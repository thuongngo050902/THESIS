from pathlib import Path
import unittest
import importlib.util
import sys


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "streamlit_app.py"
ADAPTER_PATH = ROOT / "demo_ui" / "inference_adapter.py"


def read_text(path):
    return path.read_text(encoding="utf-8")


class StreamlitUiContractTest(unittest.TestCase):
    def test_streamlit_app_imports_with_installed_streamlit_version(self):
        spec = importlib.util.spec_from_file_location("streamlit_app_import_contract", APP_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self.assertTrue(hasattr(module, "main"))

    def test_streamlit_app_declares_expected_pipeline_steps(self):
        source = read_text(APP_PATH)
        self.assertIn("PIPELINE_STEPS", source)
        self.assertIn('"Input&Mask"', source)
        self.assertIn('"Masked Input"', source)
        self.assertIn('"Stage 1"', source)
        self.assertIn('"Final Output"', source)

    def test_streamlit_app_mentions_advanced_controls_and_compare_toggle(self):
        source = read_text(APP_PATH)
        self.assertIn("Advanced Controls", source)
        self.assertIn("Upload mask file (optional)", source)
        self.assertIn("Compare", source)
        self.assertIn("Backend Mode", source)
        self.assertIn("Checkpoint presets", source)

    def test_streamlit_app_tracks_pipeline_state_in_session(self):
        source = read_text(APP_PATH)
        self.assertIn('st.session_state["selected_stage"]', source)
        self.assertIn('st.session_state["inference_result"]', source)
        self.assertIn('st.session_state["binary_mask"]', source)

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

    def test_remote_mode_marks_stage1_complete_after_final_output(self):
        source = read_text(APP_PATH)
        self.assertIn('done.add("Stage 1")', source)
        self.assertIn('st.session_state.get("final_output_result") is not None', source)

    def test_streamlit_app_lazy_loads_remote_stage1(self):
        source = read_text(APP_PATH)
        self.assertIn('checkpoint="stage1"', source)
        self.assertIn('st.session_state["backend_mode"] == InferenceBackendMode.REMOTE.value', source)

    def test_colab_api_allows_missing_params_for_stage1_checkpoint(self):
        source = read_text(ROOT / "colab_inference_api.py")
        self.assertIn('allow_missing_params=(checkpoint == "stage1")', source)


if __name__ == "__main__":
    unittest.main()
