import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "train.py"


class TrainOptimizerBetasTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = TRAIN_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def _adam_beta_literals(self):
        betas = []
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "EasyDict":
                continue

            class_name = None
            betas_value = None
            for keyword in node.keywords:
                if keyword.arg == "class_name" and isinstance(keyword.value, ast.Constant):
                    class_name = keyword.value.value
                if keyword.arg == "betas":
                    betas_value = keyword.value

            if class_name == "torch.optim.Adam" and betas_value is not None:
                betas.append(betas_value)
        return betas

    def test_all_adam_betas_are_explicit_floats(self):
        betas = self._adam_beta_literals()
        self.assertEqual(len(betas), 3)
        for beta_node in betas:
            self.assertIsInstance(beta_node, ast.List)
            self.assertEqual(len(beta_node.elts), 2)
            self.assertIsInstance(beta_node.elts[0], ast.Constant)
            self.assertIsInstance(beta_node.elts[1], ast.Constant)
            self.assertEqual(beta_node.elts[0].value, 0.0)
            self.assertEqual(beta_node.elts[1].value, 0.99)
            self.assertIsInstance(beta_node.elts[0].value, float)
            self.assertIsInstance(beta_node.elts[1].value, float)

    def test_legacy_int_beta_literal_is_absent(self):
        self.assertNotIn("betas=[0, 0.99]", self.source)
        self.assertNotIn("betas=(0, 0.99)", self.source)


if __name__ == "__main__":
    unittest.main()
