import base64
import io
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import PIL.Image
import torch
import streamlit as st
import streamlit.elements.image as st_image
import streamlit.components.v1 as components
from streamlit.elements.lib.layout_utils import LayoutConfig

from datasets.mask_generator_512 import RandomMask
from demo_ui.inference_adapter import (
    CheckpointPreset,
    InferenceBackendMode,
    InferenceResult,
    apply_mask_preview,
    ensure_binary_mask,
    run_generator_on_inputs,
    run_inference,
)

try:
    from streamlit_drawable_canvas import st_canvas
except ImportError:  # pragma: no cover - the app can still render the fallback UI without it.
    st_canvas = None

try:
    from streamlit.elements.lib.image_utils import image_to_url as _streamlit_image_to_url
except ImportError:  # pragma: no cover - older/newer Streamlit variants.
    _streamlit_image_to_url = None

if _streamlit_image_to_url is not None and not hasattr(st_image, "image_to_url"):
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


@dataclass
class StageArtifact:
    image: PIL.Image.Image
    notes: str = ""


def init_session_state():
    defaults = {
        "selected_stage": "Input&Mask",
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
        "backend_mode": InferenceBackendMode.LOCAL.value,
        "preset_name": "Custom",
        "stage1_checkpoint": PHASE1_CHECKPOINT,
        "final_checkpoint": PHASE2_FINAL_CHECKPOINT,
        "mat_original_checkpoint": MAT_BASELINE_CHECKPOINT,
        "remote_endpoint": "",
        "mask_position": "center",
        "mask_scale": 1.0,
        "mask_seed": 0,
        "mask_signature": None,
        "mask_canvas_base_mask": None,
        "mask_canvas_seed_pending": True,
        "include_mat_baseline": False,
        "mask_canvas_state": None,
        "device_name": "cuda" if torch.cuda.is_available() else "cpu",
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


def resize_mask(mask: np.ndarray, scale: float, target_shape: tuple[int, int]) -> np.ndarray:
    scale = float(np.clip(scale, 0.5, 1.8))
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


def shift_binary_mask(mask: np.ndarray, left: float, top: float, scale_x: float, scale_y: float, canvas_size: tuple[int, int]) -> np.ndarray:
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


def completed_stages() -> set[str]:
    done = set()
    if st.session_state.get("uploaded_image") is not None and st.session_state.get("binary_mask") is not None:
        done.add("Input&Mask")
    if st.session_state.get("input_confirmed"):
        done.add("Masked Input")
    if st.session_state.get("stage1_result") is not None:
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


def extract_mask_from_canvas(raw_state: Optional[dict], base_mask: np.ndarray, canvas_size: tuple[int, int]) -> np.ndarray:
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
        stage1_result = st.session_state.get("stage1_result") or ensure_stage1_result(image, mask)
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
    checkpoint_path = st.session_state.get("mat_original_checkpoint", "").strip()
    if image is None or mask is None or not checkpoint_path:
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
            preset=CheckpointPreset(name="MAT-original baseline", final_checkpoint=checkpoint_path),
            device_name=st.session_state.get("device_name", None),
        ).final_image
        cache[baseline_key] = StageArtifact(
            image=baseline_image,
            notes="Optional MAT-original baseline comparison.",
        )
    st.session_state["mat_original_result"] = cache[baseline_key]
    return cache[baseline_key]


def ensure_stage1_result(image: Optional[PIL.Image.Image], mask: Optional[PIL.Image.Image]) -> Optional[StageArtifact]:
    if image is None or mask is None:
        return None

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
        include_mat_baseline = st.checkbox("Include MAT-original baseline", key="include_mat_baseline")
        st.text_input(
            "MAT-original checkpoint",
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
        compare_options.append("MAT-original baseline")
        compare_defaults.append("MAT-original baseline")

    selected_items = st.multiselect("Compare items", compare_options, default=compare_defaults)
    if include_mat_baseline and mat_original_result is None:
        st.info("MAT-original baseline is unavailable until the required input or inference result exists.")

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
        elif item == "MAT-original baseline":
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


def main():
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
                st.text_input("Stage 1 checkpoint", key="stage1_checkpoint", help="Phase 1 final checkpoint used for the Stage 1 output.")
                st.text_input("Final checkpoint", key="final_checkpoint", help="Phase 3 final checkpoint used for the completed restoration output.")
                st.text_input("Remote endpoint", key="remote_endpoint", help="Used only when Backend Mode is remote.")
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
                    if not preset.stage1_checkpoint:
                        st.error("Stage 1 checkpoint path is required. Use the Phase 1 final checkpoint.")
                        return
                    if not preset.final_checkpoint:
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
                                st.session_state["final_output_result"] = StageArtifact(
                                    image=remote_result.final_image,
                                    notes=remote_result.pipeline_notes.get("final", "Remote final output."),
                                )
                        except ValueError as exc:
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


if __name__ == "__main__":
    main()