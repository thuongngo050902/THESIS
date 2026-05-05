import io
from typing import Dict, Optional

import numpy as np
import PIL.Image
import streamlit as st

from demo_ui.inference_adapter import (
    CheckpointPreset,
    InferenceBackendMode,
    InferenceResult,
    apply_mask_preview,
    ensure_binary_mask,
    run_inference,
)

try:
    from streamlit_drawable_canvas import st_canvas
except ImportError:  # pragma: no cover - fallback for environments without the drawing package.
    st_canvas = None


PIPELINE_STEPS = ["Input", "Mask", "Masked Input", "Stage 1", "Final Output"]

DEFAULT_PRESETS: Dict[str, CheckpointPreset] = {
    "Custom": CheckpointPreset(name="Custom"),
    "Defense Demo": CheckpointPreset(
        name="Defense Demo",
        description="Clean defense flow with a dedicated Stage 1 checkpoint and a final restoration checkpoint.",
    ),
}


def init_session_state():
    defaults = {
        "selected_stage": "Input",
        "inference_result": None,
        "binary_mask": None,
        "mask_source": "draw",
        "uploaded_image": None,
        "uploaded_mask": None,
        "backend_mode": InferenceBackendMode.LOCAL.value,
        "preset_name": "Custom",
        "stage1_checkpoint": "",
        "final_checkpoint": "",
        "remote_endpoint": "",
        "canvas_key": 0,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def load_pil(uploaded_file) -> Optional[PIL.Image.Image]:
    if uploaded_file is None:
        return None
    return PIL.Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")


def render_pipeline():
    cols = st.columns(len(PIPELINE_STEPS))
    for idx, step in enumerate(PIPELINE_STEPS):
        if cols[idx].button(
            step,
            use_container_width=True,
            type="primary" if st.session_state["selected_stage"] == step else "secondary",
        ):
            st.session_state["selected_stage"] = step


def render_mask_builder(image: Optional[PIL.Image.Image]) -> Optional[PIL.Image.Image]:
    st.caption("Draw binary mask directly on the image. White regions represent damaged areas for restoration.")
    uploaded_mask = load_pil(st.session_state["uploaded_mask"]) if st.session_state["uploaded_mask"] is not None else None
    if uploaded_mask is not None:
        st.session_state["mask_source"] = "upload"
        return ensure_binary_mask(uploaded_mask, image.size)

    st.session_state["mask_source"] = "draw"
    if image is None:
        st.info("Upload an image to start drawing the mask.")
        return None

    if st_canvas is None:
        st.warning("Install streamlit-drawable-canvas to enable drawing. For now, use the optional uploaded mask file.")
        return None

    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1.0)",
        stroke_width=18,
        stroke_color="#ffffff",
        background_image=image,
        update_streamlit=True,
        height=image.height,
        width=image.width,
        drawing_mode="freedraw",
        key=f"mask-canvas-{st.session_state['canvas_key']}",
    )
    if canvas_result.image_data is None:
        return None

    canvas_rgba = canvas_result.image_data.astype(np.uint8)
    alpha_mask = canvas_rgba[:, :, 3]
    binary = np.where(alpha_mask > 0, 255, 0).astype(np.uint8)
    binary_image = PIL.Image.fromarray(binary, mode="L")
    st.session_state["binary_mask"] = binary_image
    return binary_image


def describe_stage(stage: str, result: Optional[InferenceResult]) -> str:
    descriptions = {
        "Input": "Original image before any restoration guidance is applied.",
        "Mask": "Binary mask that defines the damaged region to be restored.",
        "Masked Input": "Visualization of the restoration problem after the damage region is applied.",
        "Stage 1": "Intermediate restoration result used to show what the earlier restoration stage contributes.",
        "Final Output": "Completed restoration result used for thesis evaluation and visual comparison.",
    }
    if result is not None and stage == "Stage 1":
        return f"{descriptions[stage]} {result.pipeline_notes.get('stage1', '')}".strip()
    if result is not None and stage == "Final Output":
        return f"{descriptions[stage]} {result.pipeline_notes.get('final', '')}".strip()
    return descriptions[stage]


def get_stage_image(stage: str, image: Optional[PIL.Image.Image], mask: Optional[PIL.Image.Image], result: Optional[InferenceResult]):
    if stage == "Input":
        return image
    if stage == "Mask":
        return mask.convert("RGB") if mask is not None else None
    if stage == "Masked Input":
        if result is not None:
            return result.masked_input_image
        if image is not None and mask is not None:
            return apply_mask_preview(image, mask)
    if stage == "Stage 1" and result is not None:
        return result.stage1_image
    if stage == "Final Output" and result is not None:
        return result.final_image
    return None


def render_stage_details(image: Optional[PIL.Image.Image], mask: Optional[PIL.Image.Image], result: Optional[InferenceResult]):
    selected_stage = st.session_state["selected_stage"]
    st.subheader(f"{selected_stage} Details")
    st.caption(describe_stage(selected_stage, result))
    stage_image = get_stage_image(selected_stage, image, mask, result)
    if stage_image is None:
        st.info("Run inference or prepare the required input to view this stage.")
        return
    st.image(stage_image, use_container_width=True)


def render_compare_panel(result: Optional[InferenceResult]):
    st.subheader("Result Viewer")
    if result is None:
        st.info("Run inference to view restoration results.")
        return

    st.image(result.final_image, caption="Final Output", use_container_width=True)
    if st.checkbox("Enable compare view"):
        compare_target = st.selectbox("Compare against", ["Input", "Masked Input", "Stage 1"])
        if compare_target == "Input":
            left = result.input_image
        elif compare_target == "Masked Input":
            left = result.masked_input_image
        else:
            left = result.stage1_image

        compare_cols = st.columns(2)
        compare_cols[0].image(left, caption=compare_target, use_container_width=True)
        compare_cols[1].image(result.final_image, caption="Final Output", use_container_width=True)


def build_preset_from_state() -> CheckpointPreset:
    return CheckpointPreset(
        name=st.session_state["preset_name"],
        stage1_checkpoint=st.session_state["stage1_checkpoint"],
        final_checkpoint=st.session_state["final_checkpoint"],
        remote_endpoint=st.session_state["remote_endpoint"],
        description=DEFAULT_PRESETS.get(st.session_state["preset_name"], DEFAULT_PRESETS["Custom"]).description,
    )


def main():
    st.set_page_config(page_title="MAT Thesis Demo UI", layout="wide")
    init_session_state()

    st.title("MAT Restoration Pipeline Demo")
    st.caption("Interactive thesis-first UI for mask-guided restoration with a clean defense flow and hidden advanced controls.")

    render_pipeline()

    left_col, center_col, right_col = st.columns([1.05, 1.35, 1.1])

    with left_col:
        st.subheader("Input Configuration")
        st.session_state["uploaded_image"] = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
        image = load_pil(st.session_state["uploaded_image"])

        with st.expander("Advanced Controls"):
            st.selectbox(
                "Backend Mode",
                options=[InferenceBackendMode.LOCAL.value, InferenceBackendMode.REMOTE.value],
                key="backend_mode",
            )
            st.selectbox("Checkpoint presets", options=list(DEFAULT_PRESETS.keys()), key="preset_name")
            st.text_input("Stage 1 checkpoint", key="stage1_checkpoint", help="Optional checkpoint path used for the Stage 1 output.")
            st.text_input("Final checkpoint", key="final_checkpoint", help="Checkpoint path used for the final output.")
            st.text_input("Remote endpoint", key="remote_endpoint", help="Used only when Backend Mode is remote.")
            st.session_state["uploaded_mask"] = st.file_uploader("Upload mask file (optional)", type=["png", "jpg", "jpeg"])
            if st.button("Clear drawn mask"):
                st.session_state["binary_mask"] = None
                st.session_state["canvas_key"] += 1

        if image is not None:
            st.image(image, caption="Input", use_container_width=True)

        current_mask = render_mask_builder(image)
        if current_mask is not None:
            st.session_state["binary_mask"] = current_mask
            st.image(current_mask, caption="Binary Mask", use_container_width=True)

        if image is not None and st.session_state["binary_mask"] is not None:
            st.image(apply_mask_preview(image, st.session_state["binary_mask"]), caption="Masked Input", use_container_width=True)

        if st.button("Run Inference", type="primary", use_container_width=True):
            if image is None:
                st.error("Upload an image before running inference.")
            elif st.session_state["binary_mask"] is None:
                st.error("Draw a binary mask or upload a mask file before running inference.")
            else:
                preset = build_preset_from_state()
                with st.spinner("Running restoration pipeline..."):
                    st.session_state["inference_result"] = run_inference(
                        backend_mode=InferenceBackendMode(st.session_state["backend_mode"]),
                        image=image,
                        mask_image=st.session_state["binary_mask"],
                        preset=preset,
                    )
                st.session_state["selected_stage"] = "Final Output"

    image = load_pil(st.session_state["uploaded_image"])
    binary_mask = st.session_state["binary_mask"]
    result = st.session_state["inference_result"]

    with center_col:
        render_stage_details(image=image, mask=binary_mask, result=result)
        if result is not None:
            with st.expander("Technical Notes"):
                st.write(result.pipeline_notes.get("backend", ""))
                st.write("Loss apply: training-time losses belong to the thesis explanation layer, not to the main inference pipeline UI.")
                st.write("This demo focuses on observable restoration stages: Input, Mask, Masked Input, Stage 1, and Final Output.")

    with right_col:
        render_compare_panel(result)


if __name__ == "__main__":
    main()
