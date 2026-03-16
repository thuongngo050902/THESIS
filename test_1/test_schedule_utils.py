import unittest

from training.schedule_utils import (
    compute_total_kimg_from_epochs,
    compute_warmup_ratio,
    resolve_training_schedule,
)


class ScheduleUtilsTest(unittest.TestCase):
    def test_compute_total_kimg_from_epochs_rounds_up(self):
        self.assertEqual(compute_total_kimg_from_epochs(num_images=6500, total_epochs=5), 33)
        self.assertEqual(compute_total_kimg_from_epochs(num_images=6500, total_epochs=40), 260)

    def test_short_schedule(self):
        schedule = resolve_training_schedule(total_epochs=5)
        self.assertEqual(schedule["profile"], "short")
        self.assertAlmostEqual(schedule["ffl_ratio"], 0.005)
        self.assertAlmostEqual(schedule["lr"], 1e-4)
        self.assertEqual(schedule["aug"], "noaug")
        self.assertFalse(schedule["enable_ffl_warmup"])

    def test_medium_schedule(self):
        schedule = resolve_training_schedule(total_epochs=20)
        self.assertEqual(schedule["profile"], "medium")
        self.assertAlmostEqual(schedule["ffl_ratio"], 0.01)
        self.assertAlmostEqual(schedule["lr"], 8e-5)
        self.assertEqual(schedule["aug"], "noaug")
        self.assertFalse(schedule["enable_ffl_warmup"])

    def test_long_schedule_enables_warmup(self):
        schedule = resolve_training_schedule(total_epochs=40)
        self.assertEqual(schedule["profile"], "long")
        self.assertAlmostEqual(schedule["ffl_ratio"], 0.02)
        self.assertAlmostEqual(schedule["lr"], 5e-5)
        self.assertEqual(schedule["aug"], "noaug")
        self.assertTrue(schedule["enable_ffl_warmup"])
        self.assertAlmostEqual(schedule["ffl_warmup_kimg"], 10.0)

    def test_warmup_ratio_is_linear_and_capped(self):
        self.assertAlmostEqual(compute_warmup_ratio(current_kimg=0.0, warmup_kimg=10.0), 0.0)
        self.assertAlmostEqual(compute_warmup_ratio(current_kimg=5.0, warmup_kimg=10.0), 0.5)
        self.assertAlmostEqual(compute_warmup_ratio(current_kimg=20.0, warmup_kimg=10.0), 1.0)
        self.assertAlmostEqual(compute_warmup_ratio(current_kimg=3.0, warmup_kimg=0.0), 1.0)

    def test_schedule_can_be_resolved_from_kimg_and_dataset_size(self):
        schedule = resolve_training_schedule(total_kimg=260, num_images=6500)
        self.assertEqual(schedule["profile"], "long")
        self.assertTrue(schedule["enable_ffl_warmup"])


if __name__ == "__main__":
    unittest.main()
