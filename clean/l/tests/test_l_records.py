# SPDX-License-Identifier: GPL-3.0-only
"""Model L comparison records (clean/l/runner/l_frame_check.py): every record names the FP8 op model its oracle used,
and a recorded run's capture-time identity is verified against the files a check loads.

    python -m unittest clean.l.tests.test_l_records
"""
import copy
import json
import os
import subprocess
import tempfile
import unittest
from unittest import mock

from clean.l.reference import table as T
from clean.l.runner import l_frame_check as F
from clean.l.runner import l_record as LR


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


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


class IdentityFailureTest(unittest.TestCase):
    def _files(self, d):
        for name, path in F.identity_files(d).items():
            _write(path, name.encode())

    def test_git_status_failure_fails_the_capture(self):
        real = subprocess.run

        def fake(cmd, *a, **k):
            if "status" in cmd:
                return subprocess.CompletedProcess(cmd, 128, "", "fatal: not a git repository")
            return real(cmd, *a, **k)

        with tempfile.TemporaryDirectory() as d:
            self._files(d)
            with mock.patch("subprocess.run", side_effect=fake):
                with self.assertRaises(RuntimeError):
                    F.capture_identity(d)

    def test_git_failure_to_run_fails_the_capture(self):
        with tempfile.TemporaryDirectory() as d:
            self._files(d)
            with mock.patch("subprocess.run", side_effect=OSError("git not found")):
                with self.assertRaises(RuntimeError):
                    F.capture_identity(d)

    def test_tracked_changes_flag_required(self):
        with tempfile.TemporaryDirectory() as d:
            self._files(d)
            ident = F.capture_identity(d)
            self.assertIsInstance(ident["git_tracked_changes"], bool)
            for bad in ({k: v for k, v in ident.items() if k != "git_tracked_changes"},
                        dict(ident, git_tracked_changes=None), dict(ident, git_tracked_changes="no")):
                with self.subTest(flag=bad.get("git_tracked_changes", "missing")):
                    with self.assertRaises(ValueError):
                        F.verify_identity({"identity": bad}, d)


# The checks that write a Model L comparison record against the oracle (each goes through LR.write_l_record).
L_ORACLE_EMITTERS = ("l_frame_check", "l_enc_check", "l_ops_check", "l_c32_check", "l_default_post_check",
                     "l_d3d12_own_post_check", "l_q7_negative_control")


class WriterTest(unittest.TestCase):
    def test_defaulted_table_names_exact(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "r.json")
            LR.write_l_record({"frames": [{"frame": 0}]}, p)                         # the shipped table
            self.assertEqual(_load(p)["fp8_op_model"], "exact")
            t = copy.deepcopy(T.TABLE)
            del t["mm8"]                                                          # a table that leaves it out
            LR.write_l_record({"x": 1}, p, table=t)
            self.assertEqual(_load(p)["fp8_op_model"], "exact")
            t["mm8"] = "trunc13"
            LR.write_l_record({"x": 1}, p, table=t)
            self.assertEqual(_load(p)["fp8_op_model"], "trunc13")
            with self.assertRaises(ValueError):                                   # a frame contradicting it
                LR.write_l_record({"frames": [{"frame": 0, "fp8_op_model": "trunc13"}]}, p)

    def test_every_l_oracle_emitter_writes_through_the_writer(self):
        here = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runner")
        for name in L_ORACLE_EMITTERS:
            with self.subTest(emitter=name):
                src = _read(os.path.join(here, name + ".py"))
                self.assertIn("write_l_record(", src)
                self.assertNotIn("json.dump(res", src)      # no record written around the writer
        # and no other runner compares with the oracle without being listed
        for fn in os.listdir(here):
            if fn.endswith(".py") and fn[:-3] not in L_ORACLE_EMITTERS:
                src = _read(os.path.join(here, fn))
                uses_oracle = any(s in src for s in ("L.encode(", "L.decode(", "compare_model(", "L.swin_block(",
                                                     "L.linear(", "L.project_groups(", "L.dec_project("))
                with self.subTest(runner=fn):
                    self.assertFalse(uses_oracle and "json.dump(" in src)


if __name__ == "__main__":
    unittest.main()
