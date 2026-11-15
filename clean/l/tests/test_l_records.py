# SPDX-License-Identifier: MIT
"""Model L comparison records (clean/l/runner/l_frame_check.py): every record names the FP8 op model its oracle used,
and a recorded run's capture-time identity is verified against the files a check loads.

    python -m unittest clean.l.tests.test_l_records
"""
import os
import tempfile
import unittest

from clean.l.reference import table as T
from clean.l.runner import l_frame_check as F


def _write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


class RecordTest(unittest.TestCase):
    def test_record_without_the_op_model_fails(self):
        rec = {"frames": [{"frame": 0}, {"frame": 1}]}
        with self.assertRaises(ValueError):
            F.check_l_record(rec)                                  # no fp8_op_model at all
        rec["fp8_op_model"] = "exact"
        with self.assertRaises(ValueError):
            F.check_l_record(rec)                                  # frames without it
        for m in rec["frames"]:
            m["fp8_op_model"] = "exact"
        F.check_l_record(rec)
        rec["frames"][1]["fp8_op_model"] = "trunc13"
        with self.assertRaises(ValueError):
            F.check_l_record(rec)                                  # a frame with another model
        with self.assertRaises(ValueError):
            F.check_l_record({"fp8_op_model": "other", "frames": []})

    def test_default_op_model_is_named(self):
        self.assertEqual(T.TABLE.get("mm8"), "exact")

    def test_identity_is_verified(self):
        with tempfile.TemporaryDirectory() as d:
            files = F.identity_files(d)
            for name, path in files.items():
                _write(path, name.encode())
            ident = F.capture_identity(d)
            self.assertTrue(ident["git_commit"])                  # from git rev-parse HEAD in this checkout
            self.assertEqual(len(ident["runtime_library_sha256"]), 64)
            rec = {"identity": ident}
            self.assertEqual(F.verify_identity(rec, d), ident)
            _write(files["params_pkl"], b"other parameters")
            with self.assertRaises(ValueError):
                F.verify_identity(rec, d)                          # params.pkl replaced after the capture
            _write(files["params_pkl"], b"params_pkl")
            os.remove(files["runtime_library"])
            with self.assertRaises(ValueError):
                F.verify_identity(rec, d)                          # the library gone
            with self.assertRaises(ValueError):
                F.verify_identity({}, d)                           # a recording without identity


if __name__ == "__main__":
    unittest.main()
