import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
GENERATE_IMAGE_PATH = ROOT / "generate_image.py"


def load_tree():
    return ast.parse(GENERATE_IMAGE_PATH.read_text(encoding="utf-8"))


def find_function(tree, name):
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


class GenerateImageLoadingContractTest(unittest.TestCase):
    def test_generate_image_has_checkpoint_selection_helpers(self):
        tree = load_tree()
        self.assertIsNotNone(find_function(tree, "select_generator_for_inference"))
        self.assertIsNotNone(find_function(tree, "load_generator_for_inference"))

    def test_generate_images_uses_loader_helper_instead_of_rebuilding_generator(self):
        tree = load_tree()
        generate_images = find_function(tree, "generate_images")
        self.assertIsNotNone(generate_images, "generate_images function is missing")

        calls_loader_helper = False
        constructs_generator = False
        strict_copy_call = False

        for node in ast.walk(generate_images):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "load_generator_for_inference":
                    calls_loader_helper = True
                if isinstance(node.func, ast.Name) and node.func.id == "Generator":
                    constructs_generator = True
                if isinstance(node.func, ast.Name) and node.func.id == "copy_params_and_buffers":
                    for keyword in node.keywords:
                        if keyword.arg == "require_all" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True:
                            strict_copy_call = True

        self.assertTrue(calls_loader_helper, "generate_images should load the saved generator through a helper")
        self.assertFalse(constructs_generator, "generate_images should not rebuild Generator for checkpoint inference")
        self.assertFalse(strict_copy_call, "generate_images should not strict-copy params in the main inference path")

    def test_copy_helper_mentions_missing_key_debug_output(self):
        source = GENERATE_IMAGE_PATH.read_text(encoding="utf-8")
        self.assertIn("Missing in source", source)
        self.assertIn("Unexpected in source", source)


if __name__ == "__main__":
    unittest.main()
