import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "streamlit_app.py"
ADAPTER_PATH = ROOT / "demo_ui" / "inference_adapter.py"


def read_text(path):
    return path.read_text(encoding="utf-8")


class StreamlitUiContractTest(unittest.TestCase):
    def test_streamlit_app_declares_expected_pipeline_steps(self):
        source = read_text(APP_PATH)
        self.assertIn("PIPELINE_STEPS", source)
        self.assertIn('"Input&Mask"', source)
        self.assertIn('"Masked Input"', source)
        self.assertIn('"Stage 1"', source)
        self.assertIn('"Final Output"', source)

    def test_streamlit_app_mentions_gated_mask_and_compare_controls(self):
        source = read_text(APP_PATH)
        self.assertIn("Advanced Controls", source)
        self.assertIn("Upload mask file (optional)", source)
        self.assertIn("Enable compare view", source)
        self.assertIn("Backend Mode", source)
        self.assertIn("Checkpoint presets", source)
        self.assertIn("Generate Mask", source)
        self.assertIn("View Masked Input", source)
        self.assertIn("Confirm Input", source)
        self.assertIn("Regenerate Mask", source)
        self.assertIn("Mask position", source)
        self.assertIn("Mask scale", source)
        self.assertIn("Seed", source)

    def test_streamlit_app_tracks_pipeline_state_in_session(self):
        source = read_text(APP_PATH)
        self.assertIn('st.session_state["selected_stage"]', source)
        self.assertIn('st.session_state["inference_result"]', source)
        self.assertIn('st.session_state["binary_mask"]', source)
        self.assertIn('st.session_state["input_confirmed"]', source)
        self.assertIn('st.session_state["mask_generated"]', source)
        self.assertIn('st.session_state["mat_result"]', source)
        self.assertIn('st.session_state["masked_input_preview"]', source)

    def test_inference_adapter_declares_local_and_remote_surface(self):
        source = read_text(ADAPTER_PATH)
        self.assertIn("class InferenceBackendMode", source)
        self.assertIn('LOCAL = "local"', source)
        self.assertIn('REMOTE = "remote"', source)
        self.assertIn("class CheckpointPreset", source)
        self.assertIn("def run_inference(", source)
        self.assertIn("def run_local_inference(", source)
        self.assertIn("def run_remote_inference(", source)

    def test_inference_adapter_returns_stage1_and_final_outputs(self):
        source = read_text(ADAPTER_PATH)
        self.assertIn("stage1_image", source)
        self.assertIn("final_image", source)
        self.assertIn("masked_input_image", source)
        self.assertIn("pipeline_notes", source)
        self.assertIn("mat_original_image", source)

    def test_streamlit_compare_supports_dynamic_multi_select(self):
        source = read_text(APP_PATH)
        self.assertIn("st.multiselect", source)
        self.assertIn("Original Input", source)
        self.assertIn("Masked Input", source)
        self.assertIn("Stage 1", source)
        self.assertIn("Final Output", source)
        self.assertIn("MAT Original", source)


if __name__ == "__main__":
    unittest.main()
