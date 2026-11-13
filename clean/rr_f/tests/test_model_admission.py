# SPDX-License-Identifier: MIT
"""Synthetic-only admission checks against independent spec-team fixtures."""
import copy
import hashlib
import json
import os
from pathlib import Path
import struct
import unittest
from unittest.mock import patch

import numpy as np

from clean.rr_f.model import admission as a

ROOT = Path(__file__).resolve().parents[3]
SPEC_DIR = Path(os.environ.get("RR_F_MODEL_SPEC_DIR", str(ROOT / "clean/rr_f/model/contracts")))


def _case(base, edits):
    value = copy.deepcopy(base)
    for edit in edits:
        op = edit["op"]
        if op == "set":
            value[edit["field"]] = edit["value"]
        elif op == "set_head":
            value["head_derivation"][edit["field"]] = edit["value"]
        elif op == "remove_parity":
            value["shifts"] = [v for v in value["shifts"] if v["parity"] != edit["parity"]]
        elif op == "set_shift":
            block, = [v for v in value["shifts"] if v["parity"] == edit["parity"]]
            row, = [v for v in block["rows"] if v["stage"] == edit["stage"]]
            row["xy"] = edit["xy"]
        else:
            rows = value[edit["collection"]]
            row, = [v for v in rows if v["id"] == edit["id"]]
            if op == "set_selected":
                row[edit["field"]] = edit["value"]
            elif op == "remove_selected":
                rows.remove(row)
            elif op == "duplicate_selected":
                rows.append(copy.deepcopy(row))
            else:
                raise ValueError("unknown independent fixture operation: " + op)
    return value


def _arrays(base):
    result = {}
    for row in base["tensors"]:
        unsigned, dtype = {"e4m3": (np.uint8, np.uint8), "f16": (np.uint16, np.float16),
                           "f32": (np.uint32, np.float32)}[row["precision"]]
        result[row["id"]] = np.full(row["shape"], int(row["synthetic_fill_bits"], 16), unsigned).view(dtype)
    return result


class ModelAdmission(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = json.loads((SPEC_DIR / "synthetic-base.json").read_text(encoding="utf-8"))
        cls.suite = json.loads((SPEC_DIR / "synthetic-cases.json").read_text(encoding="utf-8"))
        cls.contract = json.loads((SPEC_DIR / "prepared-model.contract.json").read_text(encoding="utf-8"))

    def test_independent_spec_cases(self):
        for case in self.suite["cases"]:
            with self.subTest(case=case["id"]):
                report = a.validate_metadata(_case(self.base, case["edits"]))
                self.assertEqual(report.decision, case["expected"], report.as_dict())
                if case.get("error"):
                    self.assertIn(case["error"], [v.code for v in report.issues])
                self.assertFalse(report.player_ready)
                self.assertFalse(report.payload_checked)
                if case["id"] == "attention_overflow_record_true":
                    self.assertTrue(report.z_may_overflow)

    def test_source_and_catalog_match_independent_contract(self):
        self.assertEqual(self.contract["source_head"], a.SOURCE_HEAD)
        self.assertEqual(self.contract["contract_id"], a.CONTRACT_ID)
        self.assertEqual(len(a.TENSOR_CATALOG), 197)
        self.assertEqual(a.TENSOR_CATALOG, {r["id"]: (tuple(r["shape"]), r["precision"], tuple(r["logical_axes"]))
                                           for r in self.contract["tensors"]})
        for row in self.contract["sources"]:
            self.assertEqual(a.SOURCE_SHA256[row["path"]], row["sha256"])
            # The spec pages are not part of the public repository; their hashes are checked where they exist.
            if (ROOT / row["path"]).is_file():
                self.assertEqual(hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest(), row["sha256"])

    def test_every_top_level_field_is_required_and_typed(self):
        for field in self.base:
            with self.subTest(field=field):
                value = copy.deepcopy(self.base)
                del value[field]
                self.assertEqual(a.validate_metadata(value).decision, a.REJECTED)
            for bad in (None, [], {}, 1, True, np.array([1, 2])):
                with self.subTest(field=field, bad=type(bad).__name__):
                    value = copy.deepcopy(self.base)
                    value[field] = bad
                    self.assertEqual(a.validate_metadata(value).decision, a.REJECTED)

    def test_malformed_roots_and_unknown_fields(self):
        for bad in (None, [], (), True, 1, "{}", np.array([1, 2]), {1: "bad"}):
            self.assertEqual(a.validate_metadata(bad).decision, a.REJECTED)
        for field in ("player_ready", "gpu_resource", "source_path", "physical_offset", "build_id"):
            value = copy.deepcopy(self.base)
            value[field] = True
            self.assertEqual(a.validate_metadata(value).decision, a.REJECTED)

    def test_no_bool_integer_coercion(self):
        for field, replacement in (("shape", [True, 32, 32]), ("shape", [2.0, 32, 32])):
            value = copy.deepcopy(self.base)
            value["tensors"][0][field] = replacement
            self.assertEqual(a.validate_metadata(value).decision, a.REJECTED)
        for target in ("parity", "xy"):
            value = copy.deepcopy(self.base)
            if target == "parity":
                value["shifts"][0]["parity"] = False
            else:
                value["shifts"][0]["rows"][0]["xy"] = [False, 0]
            self.assertEqual(a.validate_metadata(value).decision, a.REJECTED)

    def test_nested_malformed_objects_reject(self):
        for field in ("tensors", "symbols", "exposure_symbols", "shifts"):
            for bad in (None, [], True, {"id": []}, {"id": np.array([1, 2])}):
                value = copy.deepcopy(self.base)
                value[field][0] = bad
                self.assertEqual(a.validate_metadata(value).decision, a.REJECTED)
        for bits in ("0", "00000", "0x00", "7F", "GG", 0, False, None, []):
            value = copy.deepcopy(self.base)
            value["tensors"][0]["synthetic_fill_bits"] = bits
            self.assertEqual(a.validate_metadata(value).decision, a.REJECTED)

    def test_no_player_escalation(self):
        value = copy.deepcopy(self.base)
        value["request"] = "complete_player_model"
        self.assertEqual(a.validate_metadata(value).decision, a.BLOCKED)
        value["tensors"] = None
        self.assertEqual(a.validate_metadata(value).decision, a.BLOCKED)
        value = copy.deepcopy(self.base)
        value["complete_player_model"] = True
        self.assertEqual(a.validate_metadata(value).decision, a.REJECTED)
        value["complete_player_model"] = 0
        self.assertEqual(a.validate_metadata(value).decision, a.REJECTED)
        value = copy.deepcopy(self.base)
        value["source_frontend"]["claims_original_frontend"] = True
        self.assertEqual(a.validate_metadata(value).decision, a.REJECTED)

    def test_validation_has_no_file_io_and_does_not_mutate_input(self):
        before = copy.deepcopy(self.base)
        with patch("builtins.open", side_effect=AssertionError("unexpected file read")):
            report = a.validate_metadata(self.base)
        self.assertEqual(report.decision, a.ADMITTED)
        self.assertEqual(before, self.base)
        self.assertFalse(report.payload_checked)
        self.assertEqual(report.exposure_operational_status, "UNRESOLVED_X_OPERATIONAL_DOMAIN")
        with self.assertRaises(AttributeError):
            report.player_ready = True

    def test_actual_bindings_and_invalid_payloads(self):
        arrays = _arrays(self.base)
        report = a.validate_metadata(self.base, bindings=arrays)
        self.assertEqual(report.decision, a.ADMITTED)
        self.assertTrue(report.payload_checked)
        for bad in ({}, None, [], arrays["IN0.Wq"].reshape(-1), arrays["IN0.Wq"].astype(np.float16)):
            broken = dict(arrays)
            broken["IN0.Wq"] = bad
            report = a.validate_metadata(self.base, bindings=broken)
            self.assertEqual(report.decision, a.REJECTED)
            self.assertFalse(report.payload_checked)
        broken = dict(arrays)
        del broken["IN0.Wq"]
        self.assertEqual(a.validate_metadata(self.base, bindings=broken).decision, a.REJECTED)
        arrays["IN0.Wq"][0, 0, 0] ^= 128
        self.assertEqual(a.validate_metadata(self.base, bindings=arrays).issues[0].code, "PAYLOAD_CONTENT")

    def test_nontrivial_correctly_rounded_exponentials(self):
        # Independently familiar values, compared to nearest f32 mathematical exp.
        for h, expected in ((0.0, 0x3f800000), (1.0, 0x402df854), (-1.0, 0x3ebc5ab2),
                            (88.75, 0x7f800000), (-104.0, 0), (-103.0, 1)):
            self.assertEqual(a._exp_f32_bits(h), expected)
        self.assertEqual(a._exp_f32_bits(-0.0), a._exp_f32_bits(0.0))

    def test_source_counterexamples_retain_unresolved_diagnostic(self):
        for case in self.suite["source_constraint_counterexamples"]:
            with self.subTest(case=case["id"]):
                value = copy.deepcopy(self.base)
                for row in value["exposure_symbols"]:
                    if row["id"] in case["symbol_edits"]:
                        row["bits"] = case["symbol_edits"][row["id"]]
                report = a.validate_metadata(value)
                self.assertEqual(report.decision, a.ADMITTED)
                self.assertTrue(report.exposure_declared_domain_valid)
                self.assertEqual(report.exposure_operational_status, "UNRESOLVED_X_OPERATIONAL_DOMAIN")
                self.assertFalse(report.player_ready)


if __name__ == "__main__":
    unittest.main()
