import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
TRAINING_LOOP_PATH = ROOT / "training" / "training_loop.py"


class TrainingLoopBatchContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = TRAINING_LOOP_PATH.read_text(encoding="utf-8-sig")
        cls.tree = ast.parse(cls.source)

    def extract_helpers(self):
        helper_names = ["_normalize_dataset_sample", "_normalize_training_batch"]
        defs = {
            node.name: node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef) and node.name in helper_names
        }
        if set(defs) != set(helper_names):
            missing = sorted(set(helper_names) - set(defs))
            self.fail(f"Missing training loop batch helper(s): {missing}")

        module = ast.Module(body=[defs[name] for name in helper_names], type_ignores=[])
        ast.fix_missing_locations(module)
        namespace = {}
        exec(compile(module, str(TRAINING_LOOP_PATH), "exec"), namespace)
        return namespace

    def test_training_loop_exposes_batch_normalization_helpers(self):
        namespace = self.extract_helpers()
        self.assertIn("_normalize_dataset_sample", namespace)
        self.assertIn("_normalize_training_batch", namespace)

    def test_dataset_sample_normalizer_accepts_image_mask_label(self):
        namespace = self.extract_helpers()
        sample = ("image", "mask", "label")
        self.assertEqual(namespace["_normalize_dataset_sample"](sample), sample)

    def test_training_batch_normalizer_accepts_dataloader_style_batch(self):
        namespace = self.extract_helpers()
        batch = ["images", "masks", "labels"]
        self.assertEqual(namespace["_normalize_training_batch"](batch), ("images", "masks", "labels"))

    def test_normalizers_reject_legacy_two_value_contract(self):
        namespace = self.extract_helpers()
        for helper_name in ["_normalize_dataset_sample", "_normalize_training_batch"]:
            with self.subTest(helper=helper_name):
                with self.assertRaisesRegex(ValueError, r"image, mask, label"):
                    namespace[helper_name](("image", "label"))

    def test_training_loop_routes_unpacking_through_helpers(self):
        self.assertIn("_normalize_dataset_sample(training_set[i])", self.source)
        self.assertIn("_normalize_training_batch(next(training_set_iterator))", self.source)


if __name__ == "__main__":
    unittest.main()
