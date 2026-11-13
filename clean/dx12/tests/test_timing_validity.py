# SPDX-License-Identifier: MIT
"""CPU regression: invalid event spans must not become accepted timing summaries."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from time_frames import event_statistics


class EventValidity(unittest.TestCase):
    def test_positive(self):
        result = event_statistics([2.0, 3.0, 4.0])
        self.assertTrue(result["event_valid"])
        self.assertEqual(result["event_median"], 3.0)
        self.assertEqual(result["event_min"], 2.0)

    def test_invalid_whole_window(self):
        for invalid in [-2.261018991470337, 0.0, float("nan"), float("inf")]:
            with self.subTest(invalid=invalid):
                result = event_statistics([2.0, invalid, 3.0])
                self.assertFalse(result["event_valid"])
                self.assertIsNone(result["event_median"])
                self.assertIsNone(result["event_min"])
                self.assertEqual(result["invalid_event_samples"][0]["index"], 1)

    def test_missing_samples(self):
        result = event_statistics([])
        self.assertFalse(result["event_valid"])
        self.assertIsNone(result["event_median"])


if __name__ == "__main__":
    unittest.main()
