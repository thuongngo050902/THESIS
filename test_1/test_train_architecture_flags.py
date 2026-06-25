import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TrainArchitectureFlagsTest(unittest.TestCase):
    def read_text(self, relative_path):
        return (ROOT / relative_path).read_text(encoding="utf-8")

    def test_setup_function_accepts_phase1_kwargs(self):
        tree = ast.parse(self.read_text("train.py"))
        setup_fn = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "setup_training_loop_kwargs"
        )
        arg_names = [arg.arg for arg in setup_fn.args.args]
        self.assertIn("enable_rel_pos_bias", arg_names)
        self.assertIn("enable_mask_bias", arg_names)
        self.assertIn("enable_deterministic_latent_gate", arg_names)

    def test_setup_function_accepts_phase2_adapter_kwargs(self):
        tree = ast.parse(self.read_text("train.py"))
        setup_fn = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "setup_training_loop_kwargs"
        )
        arg_names = [arg.arg for arg in setup_fn.args.args]
        self.assertIn("enable_tran_adapter_32", arg_names)
        self.assertIn("enable_tran_adapter_16", arg_names)

    def test_setup_threads_phase1_flags_into_synthesis_kwargs(self):
        train_text = self.read_text("train.py")
        self.assertIn("args.G_kwargs.synthesis_kwargs.enable_rel_pos_bias = enable_rel_pos_bias", train_text)
        self.assertIn("args.G_kwargs.synthesis_kwargs.enable_mask_bias = enable_mask_bias", train_text)
        self.assertIn(
            "args.G_kwargs.synthesis_kwargs.enable_deterministic_latent_gate = enable_deterministic_latent_gate",
            train_text,
        )

    def test_setup_threads_phase2_flags_into_synthesis_kwargs(self):
        train_text = self.read_text("train.py")
        self.assertIn("args.G_kwargs.synthesis_kwargs.enable_tran_adapter_32 = enable_tran_adapter_32", train_text)
        self.assertIn("args.G_kwargs.synthesis_kwargs.enable_tran_adapter_16 = enable_tran_adapter_16", train_text)

    def test_train_exposes_phase1_architecture_flags(self):
        train_text = self.read_text("train.py")
        self.assertIn("--enable-rel-pos-bias", train_text)
        self.assertIn("--enable-mask-bias", train_text)
        self.assertIn("--enable-deterministic-latent-gate", train_text)

    def test_train_exposes_phase2_architecture_flags(self):
        train_text = self.read_text("train.py")
        self.assertIn("--enable-tran-adapter-32", train_text)
        self.assertIn("--enable-tran-adapter-16", train_text)

    def test_colab_entrypoint_exposes_phase1_architecture_flags(self):
        notebook_text = self.read_text("collab/train_mat_real_(2).py")
        self.assertIn("ENABLE_REL_POS_BIAS", notebook_text)
        self.assertIn("ENABLE_MASK_BIAS", notebook_text)
        self.assertIn("ENABLE_DETERMINISTIC_LATENT_GATE", notebook_text)
        self.assertIn("--enable-rel-pos-bias", notebook_text)
        self.assertIn("--enable-mask-bias", notebook_text)
        self.assertIn("--enable-deterministic-latent-gate", notebook_text)

    def test_colab_entrypoint_exposes_phase2_architecture_flags(self):
        notebook_text = self.read_text("collab/train_mat_real_(2).py")
        self.assertIn("ENABLE_TRAN_ADAPTER_32", notebook_text)
        self.assertIn("ENABLE_TRAN_ADAPTER_16", notebook_text)
        self.assertIn("--enable-tran-adapter-32", notebook_text)
        self.assertIn("--enable-tran-adapter-16", notebook_text)


if __name__ == "__main__":
    unittest.main()
