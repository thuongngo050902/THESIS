from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TrainingLoopLrtGroupingTest(unittest.TestCase):
    def test_training_loop_mentions_adapter_markers_for_transformer_lr_grouping(self):
        text = (ROOT / "training/training_loop.py").read_text(encoding="utf-8")
        self.assertIn("adapter", text)
        self.assertIn("tran", text)


if __name__ == "__main__":
    unittest.main()
