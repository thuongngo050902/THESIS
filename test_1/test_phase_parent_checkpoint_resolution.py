import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "collab" / "resolve_phase_parent_checkpoint.py"


def load_module():
    spec = importlib.util.spec_from_file_location("resolve_phase_parent_checkpoint", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PhaseParentCheckpointResolutionTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_explicit_checkpoint_path_wins(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root_dir = Path(tmpdir)
            explicit_path = root_dir / "manual_parent.pkl"
            explicit_path.write_bytes(b"checkpoint")

            resolved = self.module.resolve_parent_checkpoint(
                root_dir=root_dir,
                explicit_checkpoint_path=explicit_path,
            )

            self.assertEqual(resolved, explicit_path)

    def test_latest_phase2_snapshot_is_selected_by_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root_dir = Path(tmpdir)
            first_run = root_dir / "runs" / "faceart_phase2_ffl" / "00000-old"
            later_run = root_dir / "runs" / "faceart_phase2_ffl" / "00001-new"
            first_run.mkdir(parents=True)
            later_run.mkdir(parents=True)

            (first_run / "network-snapshot-000040.pkl").write_bytes(b"40")
            expected = later_run / "network-snapshot-000080.pkl"
            expected.write_bytes(b"80")

            resolved = self.module.resolve_parent_checkpoint(root_dir=root_dir)

            self.assertEqual(resolved, expected)

    def test_returns_none_when_no_phase2_checkpoint_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root_dir = Path(tmpdir)
            (root_dir / "runs" / "faceart_phase3_structure_guidance").mkdir(parents=True)

            resolved = self.module.resolve_parent_checkpoint(root_dir=root_dir)

            self.assertIsNone(resolved)


if __name__ == "__main__":
    unittest.main()
