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
        self.assertIn("enable_structure_guidance", arg_names)
        self.assertIn("enable_structure_fuse_16", arg_names)
        self.assertIn("enable_structure_fuse_stage2", arg_names)
        self.assertIn("enable_structure_fuse_32", arg_names)
        self.assertIn("enable_adaptive_structure_gate", arg_names)
        self.assertIn("if enable_structure_guidance is None:", self.read_text("train.py"))
        self.assertIn("if enable_structure_fuse_16 is None:", self.read_text("train.py"))
        self.assertIn("if enable_structure_fuse_stage2 is None:", self.read_text("train.py"))
        self.assertIn("if enable_structure_fuse_32 is None:", self.read_text("train.py"))
        self.assertIn("if enable_adaptive_structure_gate is None:", self.read_text("train.py"))

    def test_setup_threads_phase1_flags_into_synthesis_kwargs(self):
        train_text = self.read_text("train.py")
        self.assertIn("args.G_kwargs.synthesis_kwargs.enable_rel_pos_bias = enable_rel_pos_bias", train_text)
        self.assertIn("args.G_kwargs.synthesis_kwargs.enable_mask_bias = enable_mask_bias", train_text)
        self.assertIn(
            "args.G_kwargs.synthesis_kwargs.enable_deterministic_latent_gate = enable_deterministic_latent_gate",
            train_text,
        )
        self.assertIn("args.G_kwargs.synthesis_kwargs.enable_structure_guidance = enable_structure_guidance", train_text)
        self.assertIn("args.G_kwargs.synthesis_kwargs.enable_structure_fuse_16 = enable_structure_fuse_16", train_text)
        self.assertIn(
            "args.G_kwargs.synthesis_kwargs.enable_structure_fuse_stage2 = enable_structure_fuse_stage2",
            train_text,
        )
        self.assertIn("args.G_kwargs.synthesis_kwargs.enable_structure_fuse_32 = enable_structure_fuse_32", train_text)
        self.assertIn(
            "args.G_kwargs.synthesis_kwargs.enable_adaptive_structure_gate = enable_adaptive_structure_gate",
            train_text,
        )

    def test_train_exposes_phase1_architecture_flags(self):
        train_text = self.read_text("train.py")
        self.assertIn("--enable-rel-pos-bias", train_text)
        self.assertIn("--enable-mask-bias", train_text)
        self.assertIn("--enable-deterministic-latent-gate", train_text)
        self.assertIn("--enable-structure-guidance", train_text)
        self.assertIn("--enable-structure-fuse-16", train_text)
        self.assertIn("--enable-structure-fuse-stage2", train_text)
        self.assertIn("--enable-structure-fuse-32", train_text)
        self.assertIn("--enable-adaptive-structure-gate", train_text)

    def test_colab_entrypoint_exposes_phase1_architecture_flags(self):
        notebook_text = self.read_text("collab/train_mat_real_(2).py")
        self.assertIn("ENABLE_REL_POS_BIAS", notebook_text)
        self.assertIn("ENABLE_MASK_BIAS", notebook_text)
        self.assertIn("ENABLE_DETERMINISTIC_LATENT_GATE", notebook_text)
        self.assertIn("ENABLE_STRUCTURE_GUIDANCE", notebook_text)
        self.assertIn("ENABLE_STRUCTURE_FUSE_16", notebook_text)
        self.assertIn("ENABLE_STRUCTURE_FUSE_STAGE2", notebook_text)
        self.assertIn("ENABLE_STRUCTURE_FUSE_32", notebook_text)
        self.assertIn("ENABLE_ADAPTIVE_STRUCTURE_GATE", notebook_text)
        self.assertIn("--enable-rel-pos-bias", notebook_text)
        self.assertIn("--enable-mask-bias", notebook_text)
        self.assertIn("--enable-deterministic-latent-gate", notebook_text)
        self.assertIn("--enable-structure-guidance", notebook_text)
        self.assertIn("--enable-structure-fuse-16", notebook_text)
        self.assertIn("--enable-structure-fuse-stage2", notebook_text)
        self.assertIn("--enable-structure-fuse-32", notebook_text)
        self.assertIn("--enable-adaptive-structure-gate", notebook_text)


if __name__ == "__main__":
    unittest.main()