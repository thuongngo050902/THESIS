import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class MatTransformerArchitectureContractTest(unittest.TestCase):
    def read_text(self):
        return (ROOT / "networks/mat.py").read_text(encoding="utf-8")

    def test_window_attention_declares_resume_safe_biases(self):
        text = self.read_text()
        self.assertIn("relative_position_bias_table", text)
        self.assertIn("relative_position_index", text)
        self.assertIn("mask_bias", text)

    def test_window_attention_accepts_phase1_bias_flags(self):
        tree = ast.parse(self.read_text())
        init_fn = None
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "WindowAttention":
                init_fn = next(
                    child for child in node.body if isinstance(child, ast.FunctionDef) and child.name == "__init__"
                )
                break

        self.assertIsNotNone(init_fn)
        arg_names = [arg.arg for arg in init_fn.args.args]
        self.assertIn("enable_rel_pos_bias", arg_names)
        self.assertIn("enable_mask_bias", arg_names)

    def test_deterministic_latent_gate_is_repeatable(self):
        text = self.read_text()
        self.assertIn("class DeterministicLatentGate", text)
        self.assertIn("torch.sigmoid", text)
        self.assertIn("latent_gate", text)

    def test_dropout_based_latent_mixing_is_removed_from_mat(self):
        text = self.read_text()
        self.assertNotIn("F.dropout(mul_map, training=True)", text)


if __name__ == "__main__":
    unittest.main()
