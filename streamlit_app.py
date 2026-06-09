import base64
import io
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

# Load variables from a local .env file (e.g. COLAB_API_WITH_NGROK) sitting next to this
# script, if python-dotenv is installed. Streamlit does not read .env on its own.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().with_name(".env"))
except ImportError:
    pass

import numpy as np
import PIL.Image
import torch
import streamlit as st
import streamlit.elements.image as st_image
import streamlit.components.v1 as components

from datasets.mask_generator_512 import RandomMask
from datasets.shape_masks import (
    cross_mask,
    focused_random_mask,
    random_hole_range_for_scale,
    rect_mask,
    scribble_mask,
)
from demo_ui.inference_adapter import (
    CheckpointPreset,
    InferenceBackendMode,
    InferenceResult,
    apply_mask_preview,
    ensure_binary_mask,
    run_generator_on_inputs,
    run_inference,
    run_remote_inference,
)
from demo_ui.parf_compare import (
    ALL_KEYS,
    COMPARE_LABELS,
    OUTPUT_KEY,
    REFERENCE_KEYS,
    can_move,
    compare_items,
    reorder,
)

try:
    from streamlit_drawable_canvas import st_canvas
except ImportError:  # pragma: no cover - the app can still render the fallback UI without it.
    st_canvas = None

try:
    from streamlit.elements.lib.layout_utils import LayoutConfig
except ImportError:  # pragma: no cover - Streamlit 1.40 keeps image_to_url elsewhere.
    LayoutConfig = None

try:
    from streamlit.elements.lib.image_utils import image_to_url as _streamlit_image_to_url
except ImportError:  # pragma: no cover - older/newer Streamlit variants.
    _streamlit_image_to_url = None

if _streamlit_image_to_url is not None and LayoutConfig is not None and not hasattr(st_image, "image_to_url"):
    def _image_to_url_compat(image, width, clamp, channels, output_format, image_id):
        return _streamlit_image_to_url(
            image,
            LayoutConfig(width=width),
            clamp,
            channels,
            output_format,
            image_id,
        )

    st_image.image_to_url = _image_to_url_compat


PIPELINE_STEPS = ["Input&Mask", "Masked Input", "Stage 1", "Final Output"]

MASK_BASE_HOLE_RANGE = [0.20, 0.27]

MASK_POSITION_PRESETS = {
    "center": {"label": "Center", "target": (0.50, 0.50), "seed_offset": 0},
    "left": {"label": "Left", "target": (0.35, 0.50), "seed_offset": 3000},
    "right": {"label": "Right", "target": (0.65, 0.50), "seed_offset": 4000},
    "top": {"label": "Top", "target": (0.50, 0.35), "seed_offset": 5000},
    "bottom": {"label": "Bottom", "target": (0.50, 0.65), "seed_offset": 6000},
}

DEFAULT_PRESETS: Dict[str, CheckpointPreset] = {
    "Custom": CheckpointPreset(name="Custom"),
    "Defense Demo": CheckpointPreset(
        name="Defense Demo",
        description="Clean defense flow with a dedicated Stage 1 checkpoint and a final restoration checkpoint.",
    ),
}

PROJECT_ROOT = Path(__file__).resolve().parent
PHASE1_CHECKPOINT = str(Path("/home/subnh3/projects/ThuongNgo/THESIS/checkpoints/resume_phase1_from_finetune_plus_loss.pkl"))
PHASE2_FINAL_CHECKPOINT = str(
    Path("/home/subnh3/projects/ThuongNgo/PHASE_2/runs/faceart_phase2_tran_adapter/00014-train-places512-ep30-schedmedium-lr2.5e-05-lrt0.0001-pr0.1-ffl0.02-nopl-batch4-tc0.5-sm0.5-ema10-noaug-resumecustom/network-snapshot-000072.pkl")
)
MAT_BASELINE_CHECKPOINT = str(Path("/home/subnh3/projects/ThuongNgo/THESIS/checkpoints/Places_512_FullData.pkl"))
MAT_BASELINE_REMOTE_CHECKPOINT = "mat_baseline"
DEFAULT_COLAB_API_ENDPOINT = os.environ.get(
    "COLAB_API_WITH_NGROK",
    "https://salaried-easter-epileptic.ngrok-free.dev",
)


# ---------------------------------------------------------------------------
# PARF 3-step redesign (Create Input -> Input -> Output)
# ---------------------------------------------------------------------------
PARF_STEPS = [("create", "Create Input"), ("input", "Input"), ("output", "Output")]

PARF_MASK_SHAPES = [
    ("cross", "✚ Cross"),
    ("rect", "▭ Rect"),
    ("scribble", "〰 Scribble"),
    ("random", "⚄ Random"),
]
PARF_SHAPE_LABEL_TO_KEY = {label: key for key, label in PARF_MASK_SHAPES}
PARF_SHAPE_KEY_TO_LABEL = {key: label for key, label in PARF_MASK_SHAPES}

# 3x3 position grid -> (x_fraction, y_fraction) of the image for the mark center.
PARF_POSITION_PRESETS = {
    "tl": (0.30, 0.30), "tc": (0.50, 0.28), "tr": (0.70, 0.30),
    "cl": (0.28, 0.50), "center": (0.50, 0.50), "cr": (0.72, 0.50),
    "bl": (0.30, 0.70), "bc": (0.50, 0.72), "br": (0.70, 0.70),
}
PARF_POSITION_GRID = [["tl", "tc", "tr"], ["cl", "center", "cr"], ["bl", "bc", "br"]]
PARF_POSITION_GLYPHS = {
    "tl": "↖", "tc": "↑", "tr": "↗",
    "cl": "←", "center": "●", "cr": "→",
    "bl": "↙", "bc": "↓", "br": "↘",
}

# Real control ranges carried over from the wireframe handoff.
PARF_MASK_SCALE_MIN = 0.25
PARF_MASK_SCALE_MAX = 1.00
PARF_MASK_SCALE_STEP = 0.05
PARF_NUDGE_CLAMP = (0.12, 0.88)  # mark center stays within 12%-88% of the image bounds


@dataclass
class StageArtifact:
    image: PIL.Image.Image
    notes: str = ""


def init_session_state():
    defaults = {
        "selected_stage": "Input&Mask",
        "inference_result": None,
        "final_output_result": None,
        "stage1_result": None,
        "mat_original_result": None,
        "binary_mask": None,
        "input_confirmed": False,
        "confirmed_input_image": None,
        "confirmed_binary_mask": None,
        "confirmed_masked_input_image": None,
        "mask_source": "preset",
        "uploaded_image": None,
        "uploaded_mask": None,
        "backend_mode": InferenceBackendMode.REMOTE.value,
        "preset_name": "Custom",
        "stage1_checkpoint": PHASE1_CHECKPOINT,
        "final_checkpoint": PHASE2_FINAL_CHECKPOINT,
        "mat_original_checkpoint": MAT_BASELINE_CHECKPOINT,
        "remote_endpoint": DEFAULT_COLAB_API_ENDPOINT,
        "mask_position": "center",
        "mask_scale": 1.0,
        "mask_seed": 0,
        "mask_signature": None,
        "mask_canvas_base_mask": None,
        "mask_canvas_seed_pending": True,
        "include_mat_baseline": False,
        "mask_canvas_state": None,
        "device_name": "cuda" if torch.cuda.is_available() else "cpu",
        # --- PARF 3-step redesign state ---
        "parf_step": "create",
        "parf_shape": "cross",
        "parf_shape_radio": "cross",
        "parf_position": "center",
        "parf_scale": 1.0,
        "parf_move_step": 20,
        "parf_seed": 1234,
        "parf_nudge_x": 0,
        "parf_nudge_y": 0,
        "parf_mask_generated": False,
        "parf_mask_sig_cache": None,
        "parf_confirmed_sig": None,
        # --- PARF Step 3 (Output / compare / lightbox) state ---
        "parf_cmp_masked": True,
        "parf_cmp_mat": True,
        "parf_cmp_coarse": True,
        "parf_cmp_origin": True,
        "parf_compare_open": False,
        "parf_order": list(ALL_KEYS),
        "parf_removed": set(),
        "parf_lb_open": False,
        "parf_lb_layout": "row",
        "parf_lb_scope": "single",
    }
    for key, value in defaults.items():
        if key not in st.session_state or st.session_state[key] in (None, ""):
            st.session_state[key] = value


def load_pil(uploaded_file) -> Optional[PIL.Image.Image]:
    if uploaded_file is None:
        return None
    return PIL.Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")


def pil_to_data_url(image: PIL.Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")


def mask_overlay_image(mask_image: PIL.Image.Image) -> PIL.Image.Image:
    mask_np = np.array(mask_image.convert("L"), dtype=np.uint8)
    rgba = np.zeros((mask_np.shape[0], mask_np.shape[1], 4), dtype=np.uint8)
    hole = mask_np == 0
    rgba[hole, 0:3] = 0
    rgba[hole, 3] = 255
    return PIL.Image.fromarray(rgba, mode="RGBA")


def latest_snapshot(run_root: Path) -> Optional[str]:
    if not run_root.exists():
        return None

    best_path = None
    best_step = -1
    for snapshot_path in run_root.rglob("network-snapshot-*.pkl"):
        try:
            step = int(snapshot_path.stem.rsplit("-", 1)[-1])
        except ValueError:
            continue
        if step > best_step:
            best_step = step
            best_path = snapshot_path

    return str(best_path) if best_path is not None else None


def image_signature(image: Optional[PIL.Image.Image]) -> str:
    if image is None:
        return ""
    image_np = np.array(image.convert("RGB"), dtype=np.uint8)
    return hashlib.sha256(image_np.tobytes()).hexdigest()


def mask_signature(mask: Optional[PIL.Image.Image]) -> str:
    if mask is None:
        return ""
    mask_np = np.array(mask.convert("L"), dtype=np.uint8)
    return hashlib.sha256(mask_np.tobytes()).hexdigest()


def invalidate_confirmation_and_results(reset_stage: Optional[str] = None):
    st.session_state["input_confirmed"] = False
    st.session_state["confirmed_input_image"] = None
    st.session_state["confirmed_binary_mask"] = None
    st.session_state["confirmed_masked_input_image"] = None
    st.session_state["final_output_result"] = None
    st.session_state["stage1_result"] = None
    st.session_state["mat_original_result"] = None
    if reset_stage is not None:
        st.session_state["selected_stage"] = reset_stage


def open_compare_view():
    st.session_state["selected_stage"] = "Compare"


def close_compare_view():
    st.session_state["selected_stage"] = "Final Output"


def confirm_current_input(image: PIL.Image.Image, mask: PIL.Image.Image):
    masked_input = apply_tinted_overlay(image, mask)
    st.session_state["input_confirmed"] = True
    st.session_state["confirmed_input_image"] = image.copy()
    st.session_state["confirmed_binary_mask"] = mask.copy()
    st.session_state["confirmed_masked_input_image"] = masked_input


def build_mask_signature() -> tuple:
    return (
        st.session_state["mask_position"],
        float(st.session_state["mask_scale"]),
        int(st.session_state["mask_seed"]),
    )


def apply_tinted_overlay(image: PIL.Image.Image, mask_image: PIL.Image.Image) -> PIL.Image.Image:
    base = np.array(image.convert("RGB"), dtype=np.float32)
    mask = np.array(mask_image.convert("L"), dtype=np.uint8) == 0
    overlay = base.copy()
    overlay[mask] = 0
    return PIL.Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB")


def centered_shift(mask: np.ndarray, target_cy: int, target_cx: int) -> np.ndarray:
    hole = np.argwhere(mask == 0)
    if hole.size == 0:
        return mask
    cy, cx = hole.mean(axis=0)
    dy = int(round(target_cy - cy))
    dx = int(round(target_cx - cx))

    out = np.ones_like(mask, dtype=np.float32)
    ny = hole[:, 0] + dy
    nx = hole[:, 1] + dx
    valid = (ny >= 0) & (ny < mask.shape[0]) & (nx >= 0) & (nx < mask.shape[1])
    out[ny[valid], nx[valid]] = 0
    return out


def resize_mask(mask: np.ndarray, scale: float, target_shape: Tuple[int, int]) -> np.ndarray:
    scale = float(np.clip(scale, 0.1, 1.8))
    src_h, src_w = mask.shape
    dst_h, dst_w = target_shape
    new_h = max(1, int(round(src_h * scale)))
    new_w = max(1, int(round(src_w * scale)))

    mask_image = PIL.Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    resized = mask_image.resize((new_w, new_h), resample=PIL.Image.Resampling.NEAREST)
    resized_np = (np.array(resized, dtype=np.uint8) > 0).astype(np.float32)

    canvas = np.ones((dst_h, dst_w), dtype=np.float32)
    offset_y = max((dst_h - new_h) // 2, 0)
    offset_x = max((dst_w - new_w) // 2, 0)

    copy_h = min(new_h, dst_h)
    copy_w = min(new_w, dst_w)
    src_y = max((new_h - dst_h) // 2, 0)
    src_x = max((new_w - dst_w) // 2, 0)

    canvas[offset_y:offset_y + copy_h, offset_x:offset_x + copy_w] = resized_np[src_y:src_y + copy_h, src_x:src_x + copy_w]
    return canvas


def build_mask_canvas_initial_drawing(overlay_src: str, width: int, height: int) -> dict:
    return {
        "version": "4.4.0",
        "objects": [
            {
                "type": "image",
                "version": "4.4.0",
                "originX": "left",
                "originY": "top",
                "left": 0,
                "top": 0,
                "width": width,
                "height": height,
                "scaleX": 1,
                "scaleY": 1,
                "angle": 0,
                "opacity": 1,
                "crossOrigin": None,
                "src": overlay_src,
                "selectable": True,
                "evented": True,
                "hasControls": True,
                "hasBorders": True,
                "lockMovementX": False,
                "lockMovementY": False,
                "lockScalingX": False,
                "lockScalingY": False,
            }
        ],
    }


def shift_binary_mask(mask: np.ndarray, left: float, top: float, scale_x: float, scale_y: float, canvas_size: Tuple[int, int]) -> np.ndarray:
    src_h, src_w = mask.shape
    dst_w, dst_h = canvas_size
    scaled_w = max(1, int(round(src_w * scale_x)))
    scaled_h = max(1, int(round(src_h * scale_y)))
    mask_image = PIL.Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    scaled = mask_image.resize((scaled_w, scaled_h), resample=PIL.Image.Resampling.NEAREST)
    scaled_np = (np.array(scaled, dtype=np.uint8) > 0).astype(np.float32)

    canvas = np.ones((dst_h, dst_w), dtype=np.float32)
    paste_x = int(round(left))
    paste_y = int(round(top))

    src_x0 = max(0, -paste_x)
    src_y0 = max(0, -paste_y)
    dst_x0 = max(0, paste_x)
    dst_y0 = max(0, paste_y)
    copy_w = min(scaled_w - src_x0, dst_w - dst_x0)
    copy_h = min(scaled_h - src_y0, dst_h - dst_y0)
    if copy_w > 0 and copy_h > 0:
        canvas[dst_y0:dst_y0 + copy_h, dst_x0:dst_x0 + copy_w] = scaled_np[src_y0:src_y0 + copy_h, src_x0:src_x0 + copy_w]
    return canvas


def next_stage(stage: str) -> str:
    idx = PIPELINE_STEPS.index(stage)
    return PIPELINE_STEPS[min(idx + 1, len(PIPELINE_STEPS) - 1)]


def completed_stages() -> Set[str]:
    done = set()
    if st.session_state.get("uploaded_image") is not None and st.session_state.get("binary_mask") is not None:
        done.add("Input&Mask")
    if st.session_state.get("input_confirmed"):
        done.add("Masked Input")
    if st.session_state.get("stage1_result") is not None or st.session_state.get("final_output_result") is not None:
        done.add("Stage 1")
    if st.session_state.get("final_output_result") is not None:
        done.add("Final Output")
    return done


def sample_random_mask(mask_size: int, hole_range, seed_base: int, max_tries: int = 64) -> np.ndarray:
    lo, hi = hole_range
    target = 0.5 * (lo + hi)
    best_mask = None
    best_dist = float("inf")

    for attempt in range(max_tries):
        np.random.seed(seed_base + attempt)
        candidate = RandomMask(mask_size, hole_range=[0, 1])[0].astype(np.float32)
        ratio = float((candidate == 0).mean())
        if lo <= ratio <= hi:
            return candidate
        dist = abs(ratio - target)
        if dist < best_dist:
            best_dist = dist
            best_mask = candidate

    return best_mask if best_mask is not None else np.ones((mask_size, mask_size), dtype=np.float32)


def generate_preset_mask(image: PIL.Image.Image, position_key: str, seed: int) -> PIL.Image.Image:
    position_preset = MASK_POSITION_PRESETS[position_key]
    mask_size = max(1, min(image.size))
    mask = sample_random_mask(
        mask_size=mask_size,
        hole_range=MASK_BASE_HOLE_RANGE,
        seed_base=seed + position_preset["seed_offset"],
    )
    target_cx = int(round(image.size[0] * position_preset["target"][0]))
    target_cy = int(round(image.size[1] * position_preset["target"][1]))
    shifted = centered_shift(mask, target_cy, target_cx)
    scaled = resize_mask(shifted, float(st.session_state.get("mask_scale", 1.0)), (image.size[1], image.size[0]))
    mask_image = PIL.Image.fromarray((scaled * 255).astype(np.uint8), mode="L")
    return ensure_binary_mask(mask_image, image.size)


def extract_mask_from_canvas(raw_state: Optional[dict], base_mask: np.ndarray, canvas_size: Tuple[int, int]) -> np.ndarray:
    if not raw_state:
        return base_mask
    objects = raw_state.get("objects") or []
    if not objects:
        return base_mask
    obj = objects[0]
    left = float(obj.get("left", 0))
    top = float(obj.get("top", 0))
    scale_x = float(obj.get("scaleX", 1))
    scale_y = float(obj.get("scaleY", 1))
    return shift_binary_mask(base_mask, left=left, top=top, scale_x=scale_x, scale_y=scale_y, canvas_size=canvas_size)


def canvas_state_from_mask(mask_image: PIL.Image.Image, width: int, height: int) -> dict:
    overlay_src = pil_to_data_url(mask_overlay_image(mask_image))
    return build_mask_canvas_initial_drawing(overlay_src, width, height)


def render_pipeline():
    done = completed_stages()
    cols = st.columns(len(PIPELINE_STEPS))
    for idx, step in enumerate(PIPELINE_STEPS):
        label = f"✓ {step}" if step in done else step
        enabled = True
        if step == "Masked Input":
            enabled = st.session_state.get("binary_mask") is not None
        elif step == "Stage 1":
            enabled = st.session_state.get("input_confirmed") or st.session_state.get("stage1_result") is not None or st.session_state.get("final_output_result") is not None
        elif step == "Final Output":
            enabled = st.session_state.get("input_confirmed") or st.session_state.get("final_output_result") is not None or st.session_state.get("stage1_result") is not None
        if cols[idx].button(
            label,
            use_container_width=True,
            type="primary" if st.session_state["selected_stage"] == step else "secondary",
            disabled=not enabled,
        ):
            st.session_state["selected_stage"] = step


def render_mask_builder(image: Optional[PIL.Image.Image], interactive: bool = True) -> Optional[PIL.Image.Image]:
    st.caption("Generate and refine a MAT-style random mask before confirming the input package.")

    uploaded_mask = load_pil(st.session_state["uploaded_mask"]) if st.session_state["uploaded_mask"] is not None else None
    if image is None:
        st.info("Upload an image to enable mask generation.")
        return None

    position_keys = list(MASK_POSITION_PRESETS.keys())
    position_labels = [MASK_POSITION_PRESETS[key]["label"] for key in position_keys]
    current_position = st.session_state["mask_position"] if st.session_state["mask_position"] in position_keys else position_keys[0]

    cols = st.columns([1.0, 1.0, 1.0, 1.0])
    with cols[0]:
        selected_position = st.selectbox("Mask position", position_labels, index=position_keys.index(current_position))
        st.session_state["mask_position"] = position_keys[position_labels.index(selected_position)]
    with cols[1]:
        st.slider("Mask scale", min_value=0.5, max_value=1.8, value=float(st.session_state["mask_scale"]), step=0.05, key="mask_scale")
    with cols[2]:
        st.number_input("Seed", min_value=0, max_value=10_000_000, value=int(st.session_state["mask_seed"]), key="mask_seed")
    with cols[3]:
        generate_label = "Generate Mask" if st.session_state.get("binary_mask") is None else "Regenerate mask"
        generate_mask = st.button(generate_label, use_container_width=True, disabled=image is None)

    signature = build_mask_signature()

    if uploaded_mask is not None:
        st.session_state["mask_source"] = "upload"
        binary_mask = ensure_binary_mask(uploaded_mask, image.size)
        st.session_state["mask_signature"] = ("upload", signature)
        st.session_state["mask_canvas_state"] = None
        st.session_state["mask_canvas_base_mask"] = binary_mask.copy()
        st.session_state["mask_canvas_seed_pending"] = True
    else:
        st.session_state["mask_source"] = "preset"
        if generate_mask:
            binary_mask = generate_preset_mask(
                image,
                st.session_state["mask_position"],
                int(st.session_state["mask_seed"]),
            )
            st.session_state["mask_canvas_state"] = None
            st.session_state["mask_signature"] = signature
            st.session_state["binary_mask"] = binary_mask
            st.session_state["mask_canvas_base_mask"] = binary_mask.copy()
            st.session_state["mask_canvas_seed_pending"] = True
            invalidate_confirmation_and_results(reset_stage="Input&Mask")
            if st.session_state["selected_stage"] == "Input&Mask":
                st.rerun()

        binary_mask = st.session_state.get("binary_mask")
        base_mask = np.array(binary_mask.convert("L"), dtype=np.uint8) / 255.0 if binary_mask is not None else None
        if binary_mask is not None and st.session_state["mask_signature"] != signature:
            binary_mask = generate_preset_mask(
                image,
                st.session_state["mask_position"],
                int(st.session_state["mask_seed"]),
            )
            st.session_state["mask_canvas_state"] = None
            st.session_state["mask_signature"] = signature
            st.session_state["binary_mask"] = binary_mask
            st.session_state["mask_canvas_base_mask"] = binary_mask.copy()
            st.session_state["mask_canvas_seed_pending"] = True
            base_mask = np.array(binary_mask.convert("L"), dtype=np.uint8) / 255.0
            invalidate_confirmation_and_results(reset_stage=st.session_state.get("selected_stage", "Input&Mask"))

    if binary_mask is None:
        st.info("Press Generate Mask to create the binary mask.")
        return None

    move_step = st.slider("Move step (px)", min_value=1, max_value=100, value=20, step=1)
    move_cols = st.columns(4)
    move_actions = [
        ("← Left", -move_step, 0),
        ("→ Right", move_step, 0),
        ("↑ Up", 0, -move_step),
        ("↓ Down", 0, move_step),
    ]
    for col, (label, dx, dy) in zip(move_cols, move_actions):
        with col:
            if st.button(label, use_container_width=True):
                current_mask_image = st.session_state.get("mask_canvas_base_mask") or binary_mask
                current_mask = np.array(current_mask_image.convert("L"), dtype=np.uint8) / 255.0
                moved_mask = shift_binary_mask(current_mask, left=dx, top=dy, scale_x=1.0, scale_y=1.0, canvas_size=image.size)
                moved_mask_image = PIL.Image.fromarray((moved_mask * 255).astype(np.uint8), mode="L")
                st.session_state["binary_mask"] = moved_mask_image
                st.session_state["mask_canvas_base_mask"] = moved_mask_image.copy()
                st.session_state["mask_canvas_state"] = None
                st.session_state["mask_canvas_seed_pending"] = True
                invalidate_confirmation_and_results(reset_stage="Input&Mask")
                st.rerun()

    if st_canvas is None or not interactive:
        st.session_state["binary_mask"] = binary_mask
        st.session_state["mask_canvas_base_mask"] = binary_mask.copy()
        return binary_mask

    if uploaded_mask is None:
        base_mask_image = st.session_state.get("mask_canvas_base_mask") or binary_mask
        base_mask = np.array(base_mask_image.convert("L"), dtype=np.uint8) / 255.0
        initial_drawing = None
        if st.session_state.get("mask_canvas_seed_pending", False):
            overlay_image = mask_overlay_image(PIL.Image.fromarray((base_mask * 255).astype(np.uint8), mode="L"))
            overlay_src = pil_to_data_url(overlay_image)
            initial_drawing = st.session_state.get("mask_canvas_state") or build_mask_canvas_initial_drawing(
                overlay_src,
                image.size[0],
                image.size[1],
            )
        canvas_result = st_canvas(
            background_image=image,
            update_streamlit=True,
            height=image.height,
            width=image.width,
            drawing_mode="transform",
            initial_drawing=initial_drawing,
            display_toolbar=True,
            key=f"input-mask-canvas-{st.session_state['mask_position']}-{int(st.session_state['mask_scale'] * 100)}-{int(st.session_state['mask_seed'])}",
        )
        st.session_state["mask_canvas_seed_pending"] = False
        if canvas_result and canvas_result.json_data is not None:
            st.session_state["mask_canvas_state"] = canvas_result.json_data
            binary_mask = PIL.Image.fromarray(
                (extract_mask_from_canvas(canvas_result.json_data, base_mask, image.size) * 255).astype(np.uint8),
                mode="L",
            )
            st.session_state["binary_mask"] = binary_mask

    st.session_state["binary_mask"] = binary_mask
    return binary_mask

def render_masked_input_editor(image: Optional[PIL.Image.Image], mask: Optional[PIL.Image.Image]) -> Optional[PIL.Image.Image]:
    if image is None or mask is None:
        st.info("Generate a mask first to continue.")
        return None

    st.caption("Masked Input shows the finalized overlay after the mask is locked in Input&Mask.")

    masked_preview = apply_tinted_overlay(image, mask)
    st.image(masked_preview, use_container_width=True)
    return masked_preview


def describe_stage(stage: str, result: Optional[InferenceResult]) -> str:
    descriptions = {
        "Input&Mask": "Upload the image, generate the mask, and refine the problem setup before continuing.",
        "Masked Input": "Finalized restoration problem setup with the hole applied to the original image.",
        "Stage 1": "Intermediate restoration result used to show what the earlier restoration stage contributes.",
        "Final Output": "Completed restoration result used for thesis evaluation and visual comparison.",
    }
    if stage == "Stage 1" and st.session_state.get("stage1_result") is not None:
        return f"{descriptions[stage]} {st.session_state['stage1_result'].notes}".strip()
    if stage == "Final Output" and st.session_state.get("final_output_result") is not None:
        return f"{descriptions[stage]} {st.session_state['final_output_result'].notes}".strip()
    return descriptions[stage]


def get_stage_image(stage: str, image: Optional[PIL.Image.Image], mask: Optional[PIL.Image.Image], result: Optional[InferenceResult]):
    if stage == "Input&Mask":
        if image is not None and mask is not None:
            return None
        return image
    if stage == "Masked Input":
        if image is not None and mask is not None:
            return apply_tinted_overlay(image, mask)
        if st.session_state.get("confirmed_masked_input_image") is not None:
            return st.session_state["confirmed_masked_input_image"]
    if stage == "Stage 1":
        stage1_result = ensure_stage1_result(image, mask)
        return stage1_result.image if stage1_result is not None else None
    if stage == "Final Output":
        final_result = st.session_state.get("final_output_result")
        return final_result.image if final_result is not None else None
    return None


def render_stage_details(image: Optional[PIL.Image.Image], mask: Optional[PIL.Image.Image], result: Optional[InferenceResult]):
    selected_stage = st.session_state["selected_stage"]
    st.subheader(f"{selected_stage} Details")
    st.caption(describe_stage(selected_stage, result))
    if selected_stage == "Input&Mask":
        if image is None:
            st.info("Upload an image to start the pipeline.")
            return
        cols = st.columns(2)
        if mask is None:
            cols[0].info("Generate a mask to continue.")
            cols[1].info("Overlay will appear here after mask generation.")
        else:
            cols[0].image(mask, caption="Generated Mask", use_container_width=True)
            cols[1].image(apply_tinted_overlay(image, mask), caption="Overlay Preview", use_container_width=True)
        return
    if selected_stage == "Masked Input":
        if image is None or mask is None:
            st.info("Prepare the required input to view this stage.")
            return
        cols = st.columns(2)
        cols[0].image(mask, caption="Mask", use_container_width=True)
        cols[1].image(apply_tinted_overlay(image, mask), caption="Overlay Preview", use_container_width=True)
        return
    if selected_stage == "Stage 1":
        try:
            stage1_result = st.session_state.get("stage1_result") or ensure_stage1_result(image, mask)
        except Exception as exc:
            st.error(f"Stage 1 inference failed: {exc}")
            return
        if stage1_result is None:
            st.info("Stage 1 becomes available after the finetune+loss checkpoint can run on the confirmed input.")
            return
        st.image(stage1_result.image, use_container_width=True)
        render_compare_panel()
        return
    if selected_stage == "Final Output":
        final_result = st.session_state.get("final_output_result")
        if final_result is None:
            st.info("Run inference from Masked Input to generate the final Phase 2 output.")
            return
        st.image(final_result.image, use_container_width=True)
        render_compare_panel()
        return
    stage_image = get_stage_image(selected_stage, image, mask, result)
    if stage_image is None:
        st.info("Run inference after confirming the input package to view this stage.")
        return
    st.image(stage_image, use_container_width=True)


def ensure_mat_original_result(image: Optional[PIL.Image.Image], mask: Optional[PIL.Image.Image]) -> Optional[StageArtifact]:
    if image is None or mask is None:
        return None

    cache = st.session_state.setdefault("mat_original_cache", {})
    if st.session_state["backend_mode"] == InferenceBackendMode.REMOTE.value:
        endpoint = st.session_state.get("remote_endpoint", "").strip()
        if not endpoint:
            return None

        baseline_key = (
            "remote",
            endpoint,
            MAT_BASELINE_REMOTE_CHECKPOINT,
            image_signature(image),
            mask_signature(mask),
        )
        if baseline_key not in cache:
            remote_result = run_remote_inference(
                image=image,
                mask_image=mask,
                preset=build_preset_from_state(),
                checkpoint=MAT_BASELINE_REMOTE_CHECKPOINT,
            )
            cache[baseline_key] = StageArtifact(
                image=remote_result.final_image,
                notes=remote_result.pipeline_notes.get(
                    "final",
                    "Optional MAT original baseline comparison.",
                ),
            )
        st.session_state["mat_original_result"] = cache[baseline_key]
        return cache[baseline_key]

    checkpoint_path = st.session_state.get("mat_original_checkpoint", "").strip()
    if not checkpoint_path:
        return None
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        return None

    baseline_key = (
        checkpoint_path,
        st.session_state.get("device_name", "cpu"),
        image_signature(image),
        mask_signature(mask),
    )
    cache = st.session_state.setdefault("mat_original_cache", {})
    if baseline_key not in cache:
        baseline_image = run_inference(
            backend_mode=InferenceBackendMode.LOCAL,
            image=image,
            mask_image=mask,
            preset=CheckpointPreset(name="MAT original baseline", final_checkpoint=checkpoint_path),
            device_name=st.session_state.get("device_name", None),
        ).final_image
        cache[baseline_key] = StageArtifact(
            image=baseline_image,
            notes="Optional MAT original baseline comparison.",
        )
    st.session_state["mat_original_result"] = cache[baseline_key]
    return cache[baseline_key]


def ensure_stage1_result(image: Optional[PIL.Image.Image], mask: Optional[PIL.Image.Image]) -> Optional[StageArtifact]:
    if image is None or mask is None:
        return None

    if st.session_state["backend_mode"] == InferenceBackendMode.REMOTE.value:
        endpoint = st.session_state.get("remote_endpoint", "").strip()
        if not endpoint:
            return None

        preview_key = (
            "remote",
            endpoint,
            "stage1",
            image_signature(image),
            mask_signature(mask),
        )
        cache = st.session_state.setdefault("stage1_cache", {})
        if preview_key not in cache:
            remote_result = run_remote_inference(
                image=image,
                mask_image=mask,
                preset=build_preset_from_state(),
                checkpoint="stage1",
            )
            cache[preview_key] = StageArtifact(
                image=remote_result.final_image,
                notes=remote_result.pipeline_notes.get(
                    "stage1",
                    "Loaded lazily from the remote Stage 1 checkpoint.",
                ),
            )
        st.session_state["stage1_result"] = cache[preview_key]
        return cache[preview_key]

    checkpoint_path = st.session_state.get("stage1_checkpoint", "").strip()
    if not checkpoint_path:
        return None

    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        return None

    preview_key = (
        checkpoint_path,
        st.session_state.get("device_name", "cpu"),
        image_signature(image),
        mask_signature(mask),
    )
    cache = st.session_state.setdefault("stage1_cache", {})
    if preview_key not in cache:
        stage1_image = run_generator_on_inputs(
            checkpoint_path,
            image,
            mask,
            torch.device(st.session_state.get("device_name", "cpu")),
            allow_missing_params=True,
        )
        cache[preview_key] = StageArtifact(
            image=stage1_image,
            notes="Uses the finetune+loss checkpoint when Stage 1 is selected.",
        )
    st.session_state["stage1_result"] = cache[preview_key]
    return cache[preview_key]


def ensure_final_output_result(image: Optional[PIL.Image.Image], mask: Optional[PIL.Image.Image]) -> Optional[StageArtifact]:
    if image is None or mask is None:
        return None

    checkpoint_path = st.session_state.get("final_checkpoint", "").strip()
    if not checkpoint_path:
        return None

    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        return None

    preview_key = (
        checkpoint_path,
        st.session_state.get("device_name", "cpu"),
        image_signature(image),
        mask_signature(mask),
    )
    cache = st.session_state.setdefault("final_output_cache", {})
    if preview_key not in cache:
        final_image = run_generator_on_inputs(
            checkpoint_path,
            image,
            mask,
            torch.device(st.session_state.get("device_name", "cpu")),
        )
        cache[preview_key] = StageArtifact(
            image=final_image,
            notes="Uses the latest Phase 2 checkpoint for the final restoration result.",
        )
    st.session_state["final_output_result"] = cache[preview_key]
    return cache[preview_key]


def render_compare_panel():
    if st.button("Compare", use_container_width=True):
        open_compare_view()
        st.rerun()


def render_compare_page(image: Optional[PIL.Image.Image], mask: Optional[PIL.Image.Image]):
    st.subheader("Compare View")
    st.caption("Large compare view for direct visual checking across inputs and outputs.")

    top_cols = st.columns([1, 1, 1])
    with top_cols[0]:
        if st.button("Back to Final Output", use_container_width=True):
            close_compare_view()
            st.rerun()

    with st.expander("Advanced Compare"):
        include_mat_baseline = st.checkbox("Include MAT original baseline", key="include_mat_baseline")
        st.text_input(
            "MAT original checkpoint",
            key="mat_original_checkpoint",
            help="Optional original MAT checkpoint path used only for compare view.",
        )

    mat_original_result = ensure_mat_original_result(image, mask) if include_mat_baseline else None

    compare_options = []
    compare_defaults = []
    if image is not None:
        compare_options.append("Original Input")
        compare_defaults.append("Original Input")
    if image is not None and mask is not None:
        compare_options.append("Masked Input")
        compare_defaults.append("Masked Input")
    if st.session_state.get("stage1_result") is not None:
        compare_options.append("Stage 1")
        compare_defaults.append("Stage 1")
    if st.session_state.get("final_output_result") is not None:
        compare_options.append("Final Output")
        compare_defaults.append("Final Output")
    if mat_original_result is not None:
        compare_options.append("MAT original baseline")
        compare_defaults.append("MAT original baseline")

    selected_items = st.multiselect("Compare items", compare_options, default=compare_defaults)
    if include_mat_baseline and mat_original_result is None:
        st.info("MAT original baseline is unavailable until the required input or inference result exists.")

    if not selected_items:
        st.info("Select at least one item to compare.")
        return

    compare_images = []
    compare_labels = []
    for item in selected_items:
        compare_image = None
        if item == "Original Input":
            compare_image = image
        elif item == "Masked Input":
            compare_image = st.session_state.get("confirmed_masked_input_image") or (apply_tinted_overlay(image, mask) if image is not None and mask is not None else None)
        elif item == "Stage 1":
            stage1_result = ensure_stage1_result(image, mask)
            compare_image = stage1_result.image if stage1_result is not None else None
        elif item == "Final Output":
            final_result = st.session_state.get("final_output_result")
            compare_image = final_result.image if final_result is not None else None
        elif item == "MAT original baseline":
            mat_original_result = st.session_state.get("mat_original_result") or ensure_mat_original_result(image, mask)
            compare_image = mat_original_result.image if mat_original_result is not None else None

        if compare_image is None:
            compare_image = PIL.Image.new("RGB", (1024, 1024), (240, 240, 240))
            st.caption(f"{item} is unavailable until the required input or inference result exists.")

        compare_images.append(compare_image)
        compare_labels.append(item)

    compare_cols = st.columns(len(compare_images))
    for col, compare_image, compare_label in zip(compare_cols, compare_images, compare_labels):
        col.image(compare_image, caption=compare_label, use_container_width=True)


def build_preset_from_state() -> CheckpointPreset:
    return CheckpointPreset(
        name=st.session_state["preset_name"],
        stage1_checkpoint=st.session_state["stage1_checkpoint"],
        final_checkpoint=st.session_state["final_checkpoint"],
        remote_endpoint=st.session_state["remote_endpoint"],
        description=DEFAULT_PRESETS.get(st.session_state["preset_name"], DEFAULT_PRESETS["Custom"]).description,
    )


def _legacy_pipeline_demo():
    """Pre-redesign 4-tab pipeline UI. Kept for reference; not called by main()."""
    st.set_page_config(page_title="PARF Thesis Demo UI", layout="wide")
    init_session_state()

    st.title("PARF Restoration Pipeline Demo")
    st.caption("Interactive thesis-first UI for portrait artwork reconstruction with a clean defense flow and hidden advanced controls.")

    render_pipeline()

    left_col, center_col, right_col = st.columns([1.05, 1.35, 1.1])
    selected_stage = st.session_state["selected_stage"]

    with left_col:
        if selected_stage == "Input&Mask":
            st.subheader("Input Configuration")
            st.session_state["uploaded_image"] = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
            image = load_pil(st.session_state["uploaded_image"])

            with st.expander("Advanced Controls"):
                st.selectbox(
                    "Backend Mode",
                    options=[InferenceBackendMode.LOCAL.value, InferenceBackendMode.REMOTE.value],
                    key="backend_mode",
                )
                st.selectbox(
                    "Device",
                    options=["cuda", "cpu"],
                    key="device_name",
                    disabled=not torch.cuda.is_available(),
                    help="Local inference uses CUDA when available for faster run time.",
                )
                st.selectbox("Checkpoint presets", options=list(DEFAULT_PRESETS.keys()), key="preset_name")
                remote_selected = st.session_state["backend_mode"] == InferenceBackendMode.REMOTE.value
                st.text_input(
                    "Stage 1 checkpoint",
                    key="stage1_checkpoint",
                    disabled=remote_selected,
                    help="Local mode only. Remote mode sends checkpoint=stage1 to the Colab API.",
                )
                st.text_input(
                    "Final checkpoint",
                    key="final_checkpoint",
                    disabled=remote_selected,
                    help="Local mode only. Remote mode sends checkpoint=final to the Colab API.",
                )
                st.text_input("Remote endpoint", key="remote_endpoint", help="Base ngrok URL or full /infer URL used when Backend Mode is remote.")
                st.session_state["uploaded_mask"] = st.file_uploader("Upload mask file (optional)", type=["png", "jpg", "jpeg"])

            current_mask = render_mask_builder(image, interactive=True)
            if current_mask is not None:
                st.session_state["binary_mask"] = current_mask

            if image is not None:
                if st.session_state.get("binary_mask") is None:
                    st.image(image, caption="Original Input", use_container_width=True)
                else:
                    st.image(apply_tinted_overlay(image, st.session_state["binary_mask"]), caption="Masked Input", use_container_width=True)

            if image is not None and st.session_state.get("binary_mask") is not None:
                preview_cols = st.columns(2)
                preview_cols[0].image(st.session_state["binary_mask"], caption="Generated Mask", use_container_width=True)
                preview_cols[1].image(apply_tinted_overlay(image, st.session_state["binary_mask"]), caption="Overlay Preview", use_container_width=True)

            if st.session_state["binary_mask"] is not None:
                if st.button("Confirm Input", type="primary", use_container_width=True):
                    confirm_current_input(image, st.session_state["binary_mask"])
                    st.session_state["selected_stage"] = "Masked Input"

        elif selected_stage == "Masked Input":
            st.subheader("Inference")
            if st.button("Run Inference", type="primary", use_container_width=True, disabled=not st.session_state.get("input_confirmed")):
                image = load_pil(st.session_state["uploaded_image"])
                if image is None:
                    st.error("Upload an image before running inference.")
                elif st.session_state["binary_mask"] is None:
                    st.error("Generate a mask before running inference.")
                elif not st.session_state.get("input_confirmed"):
                    st.error("Confirm the input package before running inference.")
                else:
                    preset = build_preset_from_state()
                    remote_selected = st.session_state["backend_mode"] == InferenceBackendMode.REMOTE.value
                    if remote_selected and not preset.remote_endpoint:
                        st.error("Remote endpoint is required for Colab inference.")
                        return
                    if not remote_selected and not preset.stage1_checkpoint:
                        st.error("Stage 1 checkpoint path is required. Use the Phase 1 final checkpoint.")
                        return
                    if not remote_selected and not preset.final_checkpoint:
                        st.error("Final checkpoint path is required. Use the Phase 3 final checkpoint.")
                        return
                    with st.spinner("Running restoration pipeline..."):
                        try:
                            input_image = st.session_state.get("confirmed_input_image") or image
                            binary_mask = st.session_state.get("confirmed_binary_mask") or st.session_state["binary_mask"]
                            if input_image is None or binary_mask is None:
                                raise ValueError("Input image and mask are required for inference.")

                            if st.session_state["backend_mode"] == InferenceBackendMode.LOCAL.value:
                                final_artifact = ensure_final_output_result(input_image, binary_mask)
                                if final_artifact is None:
                                    raise ValueError("Final checkpoint could not be loaded.")
                            else:
                                remote_result = run_inference(
                                    backend_mode=InferenceBackendMode.REMOTE,
                                    image=input_image,
                                    mask_image=binary_mask,
                                    preset=preset,
                                    device_name=st.session_state["device_name"],
                                )
                                st.session_state["inference_result"] = remote_result
                                st.session_state["final_output_result"] = StageArtifact(
                                    image=remote_result.final_image,
                                    notes=remote_result.pipeline_notes.get("final", "Remote final output."),
                                )
                        except Exception as exc:
                            st.session_state["final_output_result"] = None
                            st.error(str(exc))
                        else:
                            st.session_state["selected_stage"] = "Final Output"
                            st.rerun()
        elif selected_stage == "Compare":
            st.subheader("Compare")
            st.write("Use the dedicated compare view to inspect outputs at a larger size.")

    image = load_pil(st.session_state["uploaded_image"])
    binary_mask = st.session_state["binary_mask"]

    with center_col:
        if selected_stage == "Compare":
            render_compare_page(image=image, mask=binary_mask)
        else:
            render_stage_details(image=image, mask=binary_mask, result=None)
        current_artifact = st.session_state.get("stage1_result") if selected_stage == "Stage 1" else st.session_state.get("final_output_result") if selected_stage == "Final Output" else None
        if current_artifact is not None:
            with st.expander("Technical Notes"):
                st.write(current_artifact.notes)
                st.write("Loss apply: training-time losses belong to the thesis explanation layer, not to the main inference pipeline UI.")
                st.write("This demo focuses on observable reconstruction stages: Input&Mask, Masked Input, Stage 1, and Final Output.")


def inject_parf_theme():
    st.markdown(
        """
        <style>
        .block-container { max-width: 1180px; padding-top: 2.2rem; }
        [data-testid="stImage"] img { border-radius: 10px; border: 1px solid #E5E7EB; }
        .parf-title { font-size: 2.35rem; font-weight: 800; letter-spacing: -0.5px; margin: 0 0 0.2rem; }
        .parf-title .accent { color: #3B6EA5; }
        .parf-eyebrow { text-transform: uppercase; letter-spacing: 0.12em; font-size: 0.72rem;
                        color: #6B7280; font-weight: 700; margin: 0 0 0.4rem; }
        .parf-oneliner { color: #6B7280; font-size: 1.0rem; margin: 0.1rem 0 1.2rem; }
        .parf-oneliner b { color: #1F2933; }
        .parf-phase-head { display: flex; align-items: center; gap: 0.55rem; margin: 0.4rem 0 0.5rem; }
        .parf-badge { width: 1.7rem; height: 1.7rem; border-radius: 50%; background: #3B6EA5;
                      color: #fff; display: flex; align-items: center; justify-content: center;
                      font-weight: 700; font-size: 0.95rem; flex: 0 0 auto; }
        .parf-badge.locked { background: #CBD5E1; }
        .parf-phase-title { font-size: 1.1rem; font-weight: 700; margin: 0; }
        .parf-chip { display: inline-flex; align-items: center; gap: 0.4rem; margin-top: 0.5rem;
                     border: 1px solid #D1D5DB; border-radius: 10px; padding: 0.4rem 0.7rem;
                     background: #F9FAFB; font-size: 0.9rem; }
        .parf-empty { border: 1.5px dashed #CBD5E1; border-radius: 10px; padding: 2.6rem 1rem;
                      text-align: center; color: #9AA5B1; font-size: 0.9rem; background: #FAFBFC; }
        .parf-locked-note { color: #9AA5B1; font-size: 0.8rem; font-style: italic; }
        /* Keep the live preview in view while the taller controls column scrolls. */
        div[data-testid="stColumn"]:has(.parf-preview-marker) {
            position: sticky; top: 1.25rem; align-self: flex-start;
        }
        .parf-output { max-width: 760px; margin: 0 auto; }
        .parf-cchip { display:inline-flex; align-items:center; gap:.4rem; border:1px solid #D1D5DB;
                      border-radius:999px; padding:.3rem .8rem; background:#F9FAFB; font-size:.85rem;
                      font-weight:600; }
        .parf-sub { color:#9AA5B1; font-size:.75rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _embed_centered(base_square: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Place a square (s, s) mask (keep=1/hole=0) centered onto an (H, W) all-keep canvas."""
    height, width = target_shape
    side = base_square.shape[0]
    canvas = np.ones((height, width), dtype=np.float32)
    offset_y = max((height - side) // 2, 0)
    offset_x = max((width - side) // 2, 0)
    copy_h = min(side, height)
    copy_w = min(side, width)
    canvas[offset_y:offset_y + copy_h, offset_x:offset_x + copy_w] = base_square[:copy_h, :copy_w]
    return canvas


def parf_base_shape_mask(shape: str, mask_size: int, scale: float, seed: int) -> np.ndarray:
    """Centered (mask_size, mask_size) mask for a geometric mark (1=keep, 0=hole).

    Random is handled separately in parf_build_mask via the MAT face-focused test spec.
    """
    if shape == "rect":
        return rect_mask(mask_size, scale)
    if shape == "scribble":
        return scribble_mask(mask_size, scale, seed)
    return cross_mask(mask_size, scale)


def parf_mask_signature() -> tuple:
    """Identity of the current mark configuration (shape/position/scale/seed/nudge)."""
    return (
        st.session_state["parf_shape"],
        st.session_state["parf_position"],
        round(float(st.session_state["parf_scale"]), 3),
        int(st.session_state["parf_seed"]),
        int(st.session_state["parf_nudge_x"]),
        int(st.session_state["parf_nudge_y"]),
    )


def parf_build_mask(image: PIL.Image.Image) -> PIL.Image.Image:
    """Build the binary mask (L mode, 255=keep, 0=hole) from the current create-step state."""
    width, height = image.size
    shape = st.session_state["parf_shape"]
    scale = float(st.session_state["parf_scale"])
    seed = int(st.session_state["parf_seed"])
    pos_x, pos_y = PARF_POSITION_PRESETS.get(st.session_state["parf_position"], (0.5, 0.5))
    nudge_x = int(st.session_state["parf_nudge_x"])
    nudge_y = int(st.session_state["parf_nudge_y"])
    lo, hi = PARF_NUDGE_CLAMP

    if shape == "random":
        # MAT face-focused test spec: RandomMask(512) -> centered_shift(target) -> hole-ratio
        # retry. Computed at 512 (the model's working resolution) then fit to the image.
        # The Scale slider drives the hole ratio; no geometric resize is applied.
        canvas = 512
        target_cx = pos_x * canvas + (nudge_x / max(1, width)) * canvas
        target_cy = pos_y * canvas + (nudge_y / max(1, height)) * canvas
        target_cx = int(np.clip(target_cx, lo * canvas, hi * canvas))
        target_cy = int(np.clip(target_cy, lo * canvas, hi * canvas))
        mask_arr = focused_random_mask(
            canvas, (target_cy, target_cx), random_hole_range_for_scale(scale), seed
        )
        mask_image = PIL.Image.fromarray((mask_arr * 255).astype(np.uint8), mode="L")
        return ensure_binary_mask(mask_image, image.size)

    # Geometric marks (cross / rect / scribble): built centered, then positioned in image space.
    mask_size = max(1, min(width, height))
    base = parf_base_shape_mask(shape, mask_size, scale, seed)
    embedded = _embed_centered(base, (height, width))
    target_cx = int(np.clip(round(width * pos_x) + nudge_x, lo * width, hi * width))
    target_cy = int(np.clip(round(height * pos_y) + nudge_y, lo * height, hi * height))
    shifted = centered_shift(embedded, target_cy, target_cx)
    mask_image = PIL.Image.fromarray((shifted * 255).astype(np.uint8), mode="L")
    return ensure_binary_mask(mask_image, image.size)


def parf_mark_regenerated():
    """Flag a freshly generated mask, advancing the seed for stochastic shapes."""
    if st.session_state["parf_shape"] in ("random", "scribble"):
        st.session_state["parf_seed"] = int(st.session_state["parf_seed"]) + 1
    st.session_state["parf_mask_generated"] = True


def parf_render_step_bar():
    confirmed = bool(st.session_state.get("input_confirmed"))
    inferred = st.session_state.get("final_output_result") is not None
    current = st.session_state.get("parf_step", "create")
    unlocked = {"create": True, "input": confirmed, "output": inferred}
    done = {"create": confirmed, "input": inferred, "output": False}
    notes = {"input": "— confirm step 1 first", "output": "— run inference first"}

    cols = st.columns(3, gap="small")
    for col, (key, label), num in zip(cols, PARF_STEPS, (1, 2, 3)):
        with col:
            prefix = "✓ " if done[key] else f"{num}.  "
            if st.button(
                f"{prefix}{label}",
                key=f"parf_nav_{key}",
                use_container_width=True,
                type="primary" if current == key else "secondary",
                disabled=not unlocked[key],
            ):
                st.session_state["parf_step"] = key
                st.rerun()
            if not unlocked[key]:
                st.markdown(f'<span class="parf-locked-note">{notes[key]}</span>', unsafe_allow_html=True)


def parf_render_upload_phase() -> Optional[PIL.Image.Image]:
    st.markdown(
        '<div class="parf-phase-head"><span class="parf-badge">1</span>'
        '<p class="parf-phase-title">Upload portrait artwork</p></div>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Drag & drop, or click to browse — PNG · JPG · JPEG (up to 200 MB)",
        type=["png", "jpg", "jpeg"],
        key="parf_upload",
    )
    st.session_state["uploaded_image"] = uploaded
    image = load_pil(uploaded)

    if uploaded is None:
        # Removing the file resets everything downstream (mask, confirm, locks).
        if st.session_state.get("binary_mask") is not None or st.session_state.get("parf_mask_generated"):
            st.session_state["binary_mask"] = None
            st.session_state["parf_mask_generated"] = False
            st.session_state["parf_mask_sig_cache"] = None
            invalidate_confirmation_and_results(reset_stage=None)
        return None

    size_kb = len(uploaded.getvalue()) / 1024.0
    size_text = f"{size_kb / 1024.0:.1f} MB" if size_kb >= 1024 else f"{size_kb:.1f} KB"
    st.markdown(
        f'<div class="parf-chip">📄 <b>{uploaded.name}</b> · {size_text}</div>',
        unsafe_allow_html=True,
    )
    return image


def parf_render_mark_phase(image: Optional[PIL.Image.Image]):
    locked = image is None
    badge_cls = "parf-badge locked" if locked else "parf-badge"
    st.markdown(
        f'<div class="parf-phase-head"><span class="{badge_cls}">2</span>'
        '<p class="parf-phase-title">Mark configuration</p></div>',
        unsafe_allow_html=True,
    )
    if locked:
        st.caption("Upload a portrait to enable mark configuration.")
        return

    st.caption("Generate a mark, choose its shape, then nudge it into place. The preview updates live.")

    gen_label = "Regenerate mask" if st.session_state["parf_mask_generated"] else "Generate mask"
    if st.button(gen_label, type="primary", key="parf_generate"):
        parf_mark_regenerated()

    # Mark shape — radio (single-select; always exactly one choice, no deselect/empty state).
    shape_keys = [key for key, _ in PARF_MASK_SHAPES]
    previous_shape = st.session_state["parf_shape"]
    choice = st.radio(
        "Mark shape",
        options=shape_keys,
        format_func=lambda key: PARF_SHAPE_KEY_TO_LABEL[key],
        horizontal=True,
        key="parf_shape_radio",
    )
    if choice != previous_shape:
        if choice in ("random", "scribble"):
            st.session_state["parf_seed"] = int(st.session_state["parf_seed"]) + 1
        st.session_state["parf_shape"] = choice
    st.caption("Random → existing MAT-style random-mask generator (face-focused test spec).")

    # Position preset — 3x3 grid.
    st.markdown("**Position preset**")
    for row in PARF_POSITION_GRID:
        grid_cols = st.columns(3, gap="small")
        for grid_col, pos_key in zip(grid_cols, row):
            selected = st.session_state["parf_position"] == pos_key
            if grid_col.button(
                PARF_POSITION_GLYPHS[pos_key],
                key=f"parf_pos_{pos_key}",
                use_container_width=True,
                type="primary" if selected else "secondary",
            ):
                st.session_state["parf_position"] = pos_key
                st.session_state["parf_nudge_x"] = 0
                st.session_state["parf_nudge_y"] = 0

    st.slider(
        "Mask scale",
        min_value=PARF_MASK_SCALE_MIN,
        max_value=PARF_MASK_SCALE_MAX,
        step=PARF_MASK_SCALE_STEP,
        key="parf_scale",
    )
    if st.session_state["parf_shape"] == "random":
        lo, hi = random_hole_range_for_scale(float(st.session_state["parf_scale"]))
        st.caption(f"Random sizes by hole-ratio (MAT test spec): ~{lo * 100:.0f}–{hi * 100:.0f}% masked.")
    st.slider("Move step (px)", min_value=1, max_value=100, step=1, key="parf_move_step")

    st.markdown("**Nudge position**")
    nudge_cols = st.columns(4, gap="small")
    step = int(st.session_state["parf_move_step"])
    nudges = [("← Left", -step, 0), ("→ Right", step, 0), ("↑ Up", 0, -step), ("↓ Down", 0, step)]
    width, height = image.size
    pos_x, pos_y = PARF_POSITION_PRESETS.get(st.session_state["parf_position"], (0.5, 0.5))
    lo, hi = PARF_NUDGE_CLAMP
    for nudge_col, (label, dx, dy) in zip(nudge_cols, nudges):
        if nudge_col.button(label, key=f"parf_nudge_{label}", use_container_width=True):
            # Clamp the accumulated offset to the usable range (mark center stays within
            # 12%-88%). This stops runaway accumulation, so reversing direction responds
            # immediately instead of the mark appearing stuck at the edge.
            nx = int(st.session_state["parf_nudge_x"]) + dx
            ny = int(st.session_state["parf_nudge_y"]) + dy
            st.session_state["parf_nudge_x"] = int(np.clip(nx, (lo - pos_x) * width, (hi - pos_x) * width))
            st.session_state["parf_nudge_y"] = int(np.clip(ny, (lo - pos_y) * height, (hi - pos_y) * height))


def parf_compute_live_mask(image: Optional[PIL.Image.Image]) -> Optional[PIL.Image.Image]:
    """Recompute the mask from current state (cached by config + image size)."""
    if image is None or not st.session_state["parf_mask_generated"]:
        if image is None:
            st.session_state["binary_mask"] = None
        return None

    cache_key = (parf_mask_signature(), image.size)
    if st.session_state.get("parf_mask_sig_cache") == cache_key and st.session_state.get("binary_mask") is not None:
        mask = st.session_state["binary_mask"]
    else:
        mask = parf_build_mask(image)
        st.session_state["binary_mask"] = mask
        st.session_state["parf_mask_sig_cache"] = cache_key

    # If the package was already confirmed and the mark changed, re-lock step 2.
    if st.session_state.get("input_confirmed") and st.session_state.get("parf_confirmed_sig") != parf_mask_signature():
        invalidate_confirmation_and_results(reset_stage=None)
    return mask


def parf_render_create_preview(image: Optional[PIL.Image.Image], mask: Optional[PIL.Image.Image]):
    st.markdown('<div class="parf-preview-marker"></div>', unsafe_allow_html=True)
    st.markdown('<p class="parf-eyebrow">Preview</p>', unsafe_allow_html=True)
    cols = st.columns(2, gap="medium")
    if image is None:
        cols[0].markdown('<div class="parf-empty">Binary Mask<br>upload a portrait to preview</div>', unsafe_allow_html=True)
        cols[1].markdown('<div class="parf-empty">Damaged Portrait Artwork<br>upload a portrait to preview</div>', unsafe_allow_html=True)
    elif mask is None:
        cols[0].markdown('<div class="parf-empty">Binary Mask<br>press “Generate mask”</div>', unsafe_allow_html=True)
        cols[1].image(image, caption="Portrait Artwork — original, no mark yet", use_container_width=True)
    else:
        cols[0].image(mask, caption="Binary Mask — white = keep · black = hole", use_container_width=True)
        cols[1].image(apply_tinted_overlay(image, mask), caption="Damaged Portrait Artwork — mark applied", use_container_width=True)

    st.write("")
    note_col, button_col = st.columns([0.62, 0.38], vertical_alignment="center")
    note_col.markdown('<span class="parf-locked-note">Confirm to lock this input package and continue to step 2.</span>', unsafe_allow_html=True)
    confirm_disabled = image is None or mask is None
    if button_col.button("Confirm →", type="primary", use_container_width=True, disabled=confirm_disabled, key="parf_confirm"):
        confirm_current_input(image, mask)
        st.session_state["parf_confirmed_sig"] = parf_mask_signature()
        st.session_state["parf_step"] = "input"
        st.rerun()


def parf_render_create_input():
    st.markdown(
        '<p class="parf-oneliner"><b>Create Damaged Portrait Artwork and Binary Mask.</b> '
        'Upload a portrait, then generate &amp; place a mark to define the damaged region.</p>',
        unsafe_allow_html=True,
    )
    controls_col, preview_col = st.columns([0.4, 0.6], gap="large")
    with controls_col:
        image = parf_render_upload_phase()
        parf_render_mark_phase(image)
    mask = parf_compute_live_mask(image)
    with preview_col:
        parf_render_create_preview(image, mask)


def parf_ensure_output(image: Optional[PIL.Image.Image], mask: Optional[PIL.Image.Image]) -> Optional[StageArtifact]:
    """Produce (and cache) the final restored Output for the confirmed package.

    Honors the selected backend mode and stores the artifact in
    ``final_output_result`` (the value the step bar uses to unlock Output).
    """
    if image is None or mask is None:
        return None

    if st.session_state["backend_mode"] == InferenceBackendMode.REMOTE.value:
        endpoint = st.session_state.get("remote_endpoint", "").strip()
        if not endpoint:
            raise ValueError("A remote endpoint is required for remote inference.")
        cache = st.session_state.setdefault("final_output_remote_cache", {})
        cache_key = ("remote", endpoint, "final", image_signature(image), mask_signature(mask))
        if cache_key not in cache:
            remote_result = run_remote_inference(
                image=image,
                mask_image=mask,
                preset=build_preset_from_state(),
                checkpoint="final",
            )
            cache[cache_key] = StageArtifact(
                image=remote_result.final_image,
                notes=remote_result.pipeline_notes.get("final", "Remote final restoration result."),
            )
        st.session_state["final_output_result"] = cache[cache_key]
        return cache[cache_key]

    # Local backend: ensure_final_output_result handles checkpoint path + caching.
    return ensure_final_output_result(image, mask)


def parf_run_inference():
    image = st.session_state.get("confirmed_input_image")
    mask = st.session_state.get("confirmed_binary_mask")
    if image is None or mask is None:
        st.error("Confirm an input package before running inference.")
        return
    try:
        with st.status("Running reconstruction…", expanded=False):
            artifact = parf_ensure_output(image, mask)
    except Exception as exc:  # noqa: BLE001 - surface any backend error to the user
        st.session_state["final_output_result"] = None
        st.error(f"Inference failed: {exc}")
        return
    if artifact is None:
        st.error("Inference did not return a result. Check the backend settings in Advanced.")
        return
    st.success("Reconstruction complete — opening Output.")
    st.session_state["parf_step"] = "output"
    st.rerun()


def parf_render_input_step():
    st.markdown(
        '<p class="parf-oneliner"><b>Input.</b> The confirmed input package — a binary mask and the '
        'damaged portrait artwork. Run inference to reconstruct the masked region.</p>',
        unsafe_allow_html=True,
    )
    action_col, package_col = st.columns([0.4, 0.6], gap="large")
    with action_col:
        if st.button("Run inference", type="primary", use_container_width=True, key="parf_run"):
            parf_run_inference()
        if st.button("← Back to Create Input", use_container_width=True, key="parf_back_to_create"):
            st.session_state["parf_step"] = "create"
            st.rerun()
        with st.expander("Advanced — backend"):
            st.selectbox(
                "Backend mode",
                options=[InferenceBackendMode.REMOTE.value, InferenceBackendMode.LOCAL.value],
                key="backend_mode",
            )
            st.text_input(
                "Remote endpoint",
                key="remote_endpoint",
                help="Base ngrok URL or full /infer URL used in remote mode.",
            )
            remote_selected = st.session_state["backend_mode"] == InferenceBackendMode.REMOTE.value
            st.text_input("Stage 1 checkpoint", key="stage1_checkpoint", disabled=remote_selected)
            st.text_input("Final checkpoint", key="final_checkpoint", disabled=remote_selected)
            st.text_input("MAT baseline checkpoint", key="mat_original_checkpoint", disabled=remote_selected)
            st.selectbox(
                "Device",
                options=["cuda", "cpu"],
                key="device_name",
                disabled=not torch.cuda.is_available(),
            )
    with package_col:
        st.markdown('<p class="parf-eyebrow">Input package</p>', unsafe_allow_html=True)
        cols = st.columns(2, gap="medium")
        mask = st.session_state.get("confirmed_binary_mask")
        damaged = st.session_state.get("confirmed_masked_input_image")
        if mask is not None:
            cols[0].image(mask, caption="Binary Mask", use_container_width=True)
        if damaged is not None:
            cols[1].image(damaged, caption="Damaged Portrait Artwork", use_container_width=True)


def parf_current_cmp() -> dict:
    """Read the Advanced compare checkboxes into a {key: bool} dict."""
    return {k: bool(st.session_state.get(f"parf_cmp_{k}", True)) for k in REFERENCE_KEYS}


def parf_sync_removed(key: str):
    """on_change for a compare checkbox: toggling it clears any prior chip-x removal."""
    st.session_state["parf_removed"] = set(st.session_state["parf_removed"]) - {key}


def parf_compare_image(key: str, image, mask):
    """Resolve a compare key to a PIL image (lazy inference for output/coarse/mat).

    Returns None when the source is not yet available. May raise on backend errors;
    callers wrap this to surface per-tile messages.
    """
    if key == "output":
        artifact = st.session_state.get("final_output_result") or parf_ensure_output(image, mask)
        return artifact.image if artifact is not None else None
    if key == "origin":
        return st.session_state.get("confirmed_input_image")
    if key == "masked":
        return st.session_state.get("confirmed_masked_input_image")
    if key == "coarse":
        artifact = ensure_stage1_result(image, mask)
        return artifact.image if artifact is not None else None
    if key == "mat":
        artifact = ensure_mat_original_result(image, mask)
        return artifact.image if artifact is not None else None
    return None


def parf_open_lightbox(scope: str):
    st.session_state["parf_lb_open"] = True
    st.session_state["parf_lb_scope"] = scope
    st.rerun()


def parf_close_lightbox():
    st.session_state["parf_lb_open"] = False


def parf_lightbox_items(image, mask) -> list:
    """Build [{title, sub, src}] for the lightbox from the current scope."""
    if st.session_state.get("parf_lb_scope", "single") == "single":
        keys = [OUTPUT_KEY]
    else:
        keys = compare_items(parf_current_cmp(), st.session_state["parf_order"], st.session_state["parf_removed"])
    items = []
    for key in keys:
        try:
            img = parf_compare_image(key, image, mask)
        except Exception:  # noqa: BLE001 - a broken tile should not blank the whole lightbox
            img = None
        if img is not None:
            title, sub = COMPARE_LABELS[key]
            items.append({"title": title, "sub": sub, "src": pil_to_data_url(img)})
    return items


def parf_render_output_step():
    image = st.session_state.get("confirmed_input_image")
    mask = st.session_state.get("confirmed_binary_mask")
    final = st.session_state.get("final_output_result")

    st.markdown('<div class="parf-output">', unsafe_allow_html=True)
    st.markdown(
        '<p class="parf-oneliner"><b>Output.</b> The restored result. '
        'Compare it against other inferences or the original.</p>',
        unsafe_allow_html=True,
    )

    head_col, zoom_col = st.columns([0.7, 0.3], vertical_alignment="center")
    head_col.markdown('<p class="parf-eyebrow">Result</p>', unsafe_allow_html=True)
    if zoom_col.button("🔍 Zoom", use_container_width=True, key="parf_zoom_single", disabled=final is None):
        parf_open_lightbox("single")

    if final is None:
        st.info("Run inference on the Input step to produce the restored result.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    st.image(final.image, use_container_width=True)
    st.caption("Result persists across the session. Zoom opens a synced view (scroll to zoom, drag to pan).")

    if st.session_state.get("parf_lb_open"):
        if st.button("✕ Close zoom", key="parf_lb_close"):
            parf_close_lightbox()
            st.rerun()
        parf_render_lightbox(parf_lightbox_items(image, mask))

    with st.expander("Advanced — compare"):
        st.caption(
            "Pick what to compare the output against — another inference, or the original "
            "image — then open the compare view."
        )
        for k in REFERENCE_KEYS:
            st.checkbox(COMPARE_LABELS[k][0], key=f"parf_cmp_{k}", on_change=parf_sync_removed, args=(k,))
        if st.button("Compare →", type="primary", key="parf_compare_open_btn"):
            st.session_state["parf_compare_open"] = True
            st.rerun()

    if st.session_state.get("parf_compare_open"):
        parf_render_compare_section(image, mask)

    st.markdown("</div>", unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="PARF — Portrait Artwork Reconstruction Framework", layout="wide")
    init_session_state()
    inject_parf_theme()

    st.markdown(
        '<h1 class="parf-title">Portrait Artwork Reconstruction Framework '
        '<span class="accent">— Demo</span></h1>',
        unsafe_allow_html=True,
    )

    parf_render_step_bar()
    st.markdown(
        "<hr style='margin:0.5rem 0 1.2rem;border:none;border-top:1px solid #E5E7EB;'>",
        unsafe_allow_html=True,
    )

    # Never render a locked step, even if state was tampered with.
    step = st.session_state.get("parf_step", "create")
    if step == "input" and not st.session_state.get("input_confirmed"):
        step = "create"
    if step == "output" and st.session_state.get("final_output_result") is None:
        step = "create"

    if step == "create":
        parf_render_create_input()
    elif step == "input":
        parf_render_input_step()
    else:
        parf_render_output_step()


if __name__ == "__main__":
    main()
