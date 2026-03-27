import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ThesisMetricsPipelineTest(unittest.TestCase):
    def test_example_config_loads(self):
        from thesis_metrics.config import load_config

        config = load_config(ROOT / "thesis_metrics" / "config.example.yaml")
        self.assertEqual(config["experiment_name"], "faceart_phase_comparison")
        self.assertIn("mat_original", config["results"])
        self.assertIn("architecture_phase2", config["results"])

    def test_pair_builder_detects_mismatch(self):
        from thesis_metrics.matching import collect_dataset_pairs

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            gt_dir = root / "gt"
            pred_dir = root / "pred"
            gt_dir.mkdir()
            pred_dir.mkdir()
            (gt_dir / "sample_a.png").write_bytes(b"gt-a")
            (gt_dir / "sample_b.png").write_bytes(b"gt-b")
            (pred_dir / "sample_a.png").write_bytes(b"pred-a")

            with self.assertRaisesRegex(ValueError, "Missing prediction files"):
                collect_dataset_pairs(gt_dir=gt_dir, pred_dir=pred_dir)

    def test_summary_writer_emits_expected_files(self):
        from thesis_metrics.reporting import write_summary_outputs

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            summary = {
                "mat_original": {
                    "FID": 10.5,
                    "LPIPS": 0.21,
                    "P-IDS": 0.61,
                    "U-IDS": 0.59,
                    "PSNR": 21.1,
                    "SSIM": 0.75,
                    "L1": 0.08,
                },
                "architecture_phase2": {
                    "FID": 9.7,
                    "LPIPS": 0.18,
                    "P-IDS": 0.64,
                    "U-IDS": 0.62,
                    "PSNR": 22.5,
                    "SSIM": 0.79,
                    "L1": 0.07,
                },
            }

            write_summary_outputs(
                output_dir=output_dir,
                experiment_name="faceart_phase_comparison",
                summary=summary,
                enabled_optional_metrics=[],
                qualitative_outputs=["qualitative/panel_hardcase_01.png"],
                notes=["ArcFace disabled."],
            )

            summary_json = output_dir / "summary.json"
            summary_csv = output_dir / "summary.csv"
            summary_md = output_dir / "summary.md"

            self.assertTrue(summary_json.exists())
            self.assertTrue(summary_csv.exists())
            self.assertTrue(summary_md.exists())

            payload = json.loads(summary_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["experiment_name"], "faceart_phase_comparison")
            self.assertEqual(payload["best_by_metric"]["FID"]["model"], "architecture_phase2")
            self.assertIn("qualitative/panel_hardcase_01.png", payload["qualitative_outputs"])

            markdown = summary_md.read_text(encoding="utf-8")
            self.assertIn("FID", markdown)
            self.assertIn("architecture_phase2", markdown)
            self.assertIn("ArcFace disabled.", markdown)

    def test_readme_documents_locked_metric_stack(self):
        readme = (ROOT / "thesis_metrics" / "README.md").read_text(encoding="utf-8")
        self.assertIn("FID", readme)
        self.assertIn("LPIPS", readme)
        self.assertIn("P-IDS", readme)
        self.assertIn("U-IDS", readme)
        self.assertIn("MAT original", readme)
