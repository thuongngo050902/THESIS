import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ColabEntrypointContractTest(unittest.TestCase):
    def read_text(self, relative_path):
        return (ROOT / relative_path).read_text(encoding="utf-8")

    def test_generate_image_guards_pyspng_import(self):
        tree = ast.parse(self.read_text("generate_image.py"))
        guarded_import = False
        for node in tree.body:
            if not isinstance(node, ast.Try):
                continue
            imported_pyspng = any(
                isinstance(stmt, ast.Import) and any(alias.name == "pyspng" for alias in stmt.names)
                for stmt in node.body
            )
            catches_import_error = any(
                handler.type is None
                or (isinstance(handler.type, ast.Name) and handler.type.id == "ImportError")
                for handler in node.handlers
            )
            if imported_pyspng and catches_import_error:
                guarded_import = True
                break
        self.assertTrue(
            guarded_import,
            "generate_image.py should guard the pyspng import with try/except ImportError",
        )

    def test_missing_package_init_files_are_committed(self):
        for relative_path in ["datasets/__init__.py", "losses/__init__.py", "networks/__init__.py"]:
            with self.subTest(relative_path=relative_path):
                self.assertTrue((ROOT / relative_path).exists(), f"Missing {relative_path}")

    def test_notebook_no_longer_contains_a_hardcoded_token(self):
        notebook_text = self.read_text("collab/train_mat_real_(2).py")
        self.assertNotIn('token = "ghp_', notebook_text)

    def test_notebook_contains_lambda_ffl_training_config(self):
        notebook_text = self.read_text("collab/train_mat_real_(2).py")
        self.assertIn("LAMBDA_FFL", notebook_text)
        self.assertIn("Places_512_FullData.pkl", notebook_text)
        self.assertIn("python train.py", notebook_text)

    def test_train_exposes_lambda_ffl_and_legacy_alias(self):
        train_text = self.read_text("train.py")
        self.assertIn("--lambda-ffl", train_text)
        self.assertIn("--ffl-ratio", train_text)

    def test_train_setup_mentions_lambda_ffl(self):
        train_text = self.read_text("train.py")
        self.assertIn("lambda_ffl", train_text)

    def test_torch_sampler_uses_modern_super_signature(self):
        misc_text = self.read_text("torch_utils/misc.py")
        self.assertIn("super().__init__()", misc_text)
        self.assertNotIn("super().__init__(dataset)", misc_text)


if __name__ == "__main__":
    unittest.main()
