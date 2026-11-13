# SPDX-License-Identifier: MIT
"""CPU-only negative gates for GPU evidence; no device, model or GPU work."""
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gpu_evidence as G


def process(pid, parent=1, created=None, name="test.exe"):
    return {"pid": pid, "ppid": parent, "created_filetime": pid if created is None else created, "name": name}


def counter(pid=10, value=0.0, code=0, engine=0):
    name = f"pid_{pid}_luid_0x00000000_0x00016141_phys_0_eng_{engine}_engtype_3D"
    return {"instance": name, "status": G.status(code), "value": value, **G.instance_fields(name)}


def sample(begin=0, end=1, phase="during", counters=None, processes=None, valid=True,
           raw_time=None, raw_first=0):
    counters = counters if counters is not None else [counter()]
    raw_time = end if raw_time is None else raw_time
    raw = [{"instance": r["instance"], "status": r["status"], "first": raw_first,
            "second": raw_time, "timestamp_filetime": raw_time, "multi_count": 1} for r in counters]
    return {"interval_begin_ns": begin, "interval_end_ns": end, "phase": phase, "phase_begin": phase,
            "valid": valid, "processes": {"valid": True, "rows": processes or [process(10)]},
            "counters": {"collect_status": G.status(0),
                         "formatted": {"status": G.status(0), "rows": counters},
                         "raw": {"status": G.status(0), "rows": raw}}}


def options(**overrides):
    result = dict(min_samples=2, max_interval=1.0, max_unrelated_pid=5, max_unrelated_total=10)
    result.update(overrides)
    return SimpleNamespace(**result)


def qualified_fixture():
    # Nanosecond units; exact contiguous interval boundaries across all phases.
    rows = [sample(i * 250000000, (i + 1) * 250000000,
                   phase="pre" if i < 2 else "post" if i >= 6 else "during",
                   counters=[counter(value=50 if 2 <= i < 6 else 0)]) for i in range(8)]
    running = 0
    for row in rows:
        running += int((row["interval_end_ns"] - row["interval_begin_ns"]) *
                       row["counters"]["formatted"]["rows"][0]["value"] / 100)
        row["counters"]["raw"]["rows"][0]["first"] = running
    rows = G.attribute(rows, 10, [], [])
    markers = [{"kind": kind, "monotonic_ns": t, "pid": 10, "phase": "stream", "warm_seconds": 3}
               for kind, t in (("steady_begin", 600000000), ("steady_end", 1400000000))]
    return rows, markers


class StatusTests(unittest.TestCase):
    def test_zero_and_new_data_are_both_valid(self):
        for code in (0, 1):
            self.assertTrue(G.classify_interval(sample(), sample(raw_time=2, counters=[counter(code=code)]))["valid"])

    def test_initial_formatted_rate_unavailable_does_not_hide_valid_first_interval(self):
        previous = sample()
        previous["counters"]["formatted"] = {"status": G.status(G.INVALID_DATA), "rows": []}
        self.assertTrue(G.classify_interval(previous, sample(raw_time=2))["valid"])

    def test_invalid_and_missing_instance_rejected_even_when_value_zero(self):
        for code in (G.NO_INSTANCE, G.INVALID_DATA, 0x800007D5):
            result = G.classify_interval(sample(), sample(raw_time=2, counters=[counter(value=0, code=code)]))
            self.assertFalse(result["valid"])
            self.assertIn("invalid_counter_unresolved", result["reasons"])

    def test_disappearing_process_is_classified_but_not_excused(self):
        result = G.classify_interval(sample(processes=[process(10)]),
                                     sample(raw_time=2, counters=[], processes=[process(20)]))
        self.assertFalse(result["valid"])
        self.assertEqual(result["transitions"][0]["kind"], "disappeared_process_exit")

    def test_new_instance_requires_previous_counter_even_if_formatted_status_valid(self):
        result = G.classify_interval(sample(), sample(raw_time=2, counters=[counter(), counter(20)]))
        self.assertIn("new_counter_without_previous_sample", result["reasons"])

    def test_raw_api_failure_and_status_failure_reject(self):
        for mutate in (lambda s: s["counters"]["raw"].update(status=G.status(G.NO_INSTANCE)),
                       lambda s: s["counters"]["raw"]["rows"][0].update(status=G.status(G.INVALID_DATA))):
            current = sample(raw_time=2)
            mutate(current)
            self.assertFalse(G.classify_interval(sample(), current)["valid"])


class AccountingTests(unittest.TestCase):
    def pair(self, value=50, pid=10):
        previous = sample(raw_time=10000000, raw_first=1000000, counters=[counter(pid, 0)])
        current = sample(raw_time=12500000, raw_first=1000000 + int(2500000 * value / 100),
                         counters=[counter(pid, value)])
        return previous, current

    def test_exact_retained_sign_change_rejected_despite_valid_status(self):
        previous, current = self.pair()
        previous["counters"]["raw"]["rows"][0].update(
            first=197974242, second=134365743376176503, timestamp_filetime=134365599376176503)
        current["counters"]["raw"]["rows"][0].update(
            first=-4015511905, second=134365743378674931, timestamp_filetime=134365599378674931)
        current["counters"]["formatted"]["rows"][0]["value"] = 738334027216156.1
        result = G.classify_interval(previous, current)
        self.assertFalse(result["valid"])
        self.assertIn("raw_counter_accounting_unknown", result["reasons"])
        detail = next(t for t in result["transitions"] if t["kind"] == "raw_counter_accounting_unknown")
        self.assertIn("decreased_cumulative_count", detail["problems"])
        self.assertIn("negative_cumulative_count", detail["problems"])
        self.assertEqual(detail["previous"]["first"], 197974242)
        self.assertEqual(detail["current"]["first"], -4015511905)

    def test_decreasing_positive_and_increasing_negative_counts_are_unknown(self):
        for before, after in ((100, 99), (-100, -99), (-1, 0)):
            previous, current = self.pair()
            previous["counters"]["raw"]["rows"][0]["first"] = before
            current["counters"]["raw"]["rows"][0]["first"] = after
            result = G.classify_interval(previous, current)
            self.assertFalse(result["valid"])
            self.assertIn("raw_counter_accounting_unknown", result["reasons"])

    def test_zero_reversed_negative_and_missing_raw_time_are_unknown(self):
        for field in ("second", "timestamp_filetime"):
            for bad in (10000000, 9999999, -1, None, float("nan")):
                previous, current = self.pair()
                current["counters"]["raw"]["rows"][0][field] = bad
                result = G.classify_interval(previous, current)
                self.assertFalse(result["valid"], (field, bad))
                self.assertIn("raw_counter_accounting_unknown", result["reasons"])

    def test_workload_and_unrelated_overshoot_get_the_same_unknown_reason(self):
        for pid, value in ((10, 702.012642609826), (9052, 25111.459072074962)):
            previous, current = self.pair(value, pid)
            result = G.classify_interval(previous, current)
            self.assertFalse(result["valid"])
            self.assertIn("counter_accounting_resolution_unknown", result["reasons"])
            detail = next(t for t in result["transitions"] if t["kind"] == "counter_accounting_resolution_unknown")
            self.assertEqual(detail["value"], value)

    def test_exact_100_passes_next_float_above_100_is_unknown(self):
        for value, expected in ((100.0, True), (math.nextafter(100.0, math.inf), False)):
            self.assertEqual(G.classify_interval(*self.pair(value))["valid"], expected)

    def test_two_80_percent_engines_are_not_one_160_percent_instance(self):
        previous, current = self.pair(80)
        for row in (previous, current):
            row["counters"]["formatted"]["rows"].append(counter(10, 80, engine=1))
            raw = copy.deepcopy(row["counters"]["raw"]["rows"][0])
            raw["instance"] = counter(10, engine=1)["instance"]
            row["counters"]["raw"]["rows"].append(raw)
        self.assertTrue(G.classify_interval(previous, current)["valid"])
        attributed = G.attribute([previous, current], 10, [], [])[-1]
        self.assertEqual(attributed["attribution"]["workload_total_percent"], 160)

    def test_classification_preserves_raw_values_and_statuses(self):
        previous, current = self.pair(25111.459072074962)
        before = copy.deepcopy((previous, current))
        G.classify_interval(previous, current)
        self.assertEqual((previous, current), before)

    def test_workload_overshoot_cannot_pass_qualification_or_drop_edge(self):
        rows, markers = qualified_fixture()
        original_count = len(rows)
        # Row 2 overlaps only the leading edge (500..750ms against a 600ms begin).
        rows[2]["counters"]["formatted"]["rows"][0]["value"] = 702.012642609826
        rows[2].update(G.classify_interval(rows[1], rows[2]))
        rows = G.attribute(rows, 10, [], [])
        self.assertEqual(rows[2]["attribution"]["unrelated_total_percent"], 0)
        self.assertEqual(rows[2]["attribution"]["workload_total_percent"], 702.012642609826)
        result = G.qualify(rows, markers, options())
        self.assertFalse(result["activity_qualified"])
        self.assertIn("invalid_telemetry_interval", result["windows"][0]["reasons"])
        self.assertEqual(len(rows), original_count)
        self.assertEqual(result["windows"][0]["sample_count"], 4)

    def test_numeric_formatted_corruption_is_unknown_even_with_valid_status(self):
        for value in (float("nan"), float("inf"), float("-inf"), -1, None):
            previous, current = self.pair()
            current["counters"]["formatted"]["rows"][0]["value"] = value
            self.assertFalse(G.classify_interval(previous, current)["valid"])


class AttributionTests(unittest.TestCase):
    def test_descendants_observed_and_unrelated_not_hidden(self):
        rows = [sample(counters=[counter(11, 50), counter(20, 3)],
                       processes=[process(10), process(11, 10), process(20)])]
        result = G.attribute(rows, 10, [], ["test.exe"])[0]["attribution"]
        self.assertEqual(result["workload_total_percent"], 50)
        self.assertEqual(result["unrelated_total_percent"], 3)
        self.assertEqual(result["active_declared_competitors"][0]["pid"], 20)

    def test_reused_root_pid_is_not_a_second_workload(self):
        rows = [sample(counters=[counter(10, 50)], processes=[process(10, created=10)]),
                sample(counters=[counter(10, 50)], processes=[process(10, created=20)])]
        result = G.attribute(rows, 10, [], [])[1]["attribution"]
        self.assertEqual(result["workload_total_percent"], 0)
        self.assertEqual(result["unrelated_total_percent"], 50)

    def test_protected_identity_counts_as_unrelated_not_workload(self):
        proc = process(10)
        proc["created_filetime"] = None
        proc["identity_error"] = 5
        result = G.attribute([sample(counters=[counter(10, 0.2)], processes=[proc])], 10, [], [])[0]["attribution"]
        self.assertEqual(result["unrelated_total_percent"], 0.2)
        self.assertEqual(result["workload_total_percent"], 0)
        self.assertEqual(result["unverified_creation_active_pids"], [10])

    def test_unobserved_process_remains_unknown_and_unrelated(self):
        result = G.attribute([sample(counters=[counter(22, 1)])], 10, [], [])[0]["attribution"]
        self.assertEqual(result["unknown_active_pids"], [22])
        self.assertEqual(result["unrelated_total_percent"], 1)


class QualificationTests(unittest.TestCase):
    def test_complete_valid_window_passes(self):
        rows, markers = qualified_fixture()
        self.assertTrue(G.qualify(rows, markers, options())["activity_qualified"])

    def test_deliberate_invalid_measurement_rejected(self):
        rows, markers = qualified_fixture()
        rows[3]["valid"] = False
        result = G.qualify(rows, markers, options())
        self.assertFalse(result["activity_qualified"])
        self.assertIn("invalid_telemetry_interval", result["windows"][0]["reasons"])

    def test_missing_middle_interval_rejected(self):
        rows, markers = qualified_fixture()
        del rows[3]
        result = G.qualify(rows, markers, options())
        self.assertIn("sampling_gap", result["windows"][0]["reasons"])

    def test_partial_edge_coverage_rejected(self):
        rows, markers = qualified_fixture()
        markers[0]["monotonic_ns"] = -1
        self.assertIn("window_not_fully_covered", G.qualify(rows, markers, options())["windows"][0]["reasons"])

    def test_competitor_and_thresholds_reject(self):
        for field, value in (("unrelated_max_pid_percent", 5.01), ("unrelated_total_percent", 10.01),
                             ("active_declared_competitors", [{"pid": 99}]), ("unknown_active_pids", [99])):
            rows, markers = qualified_fixture()
            rows[3]["attribution"][field] = value
            self.assertFalse(G.qualify(rows, markers, options())["activity_qualified"], field)

    def test_missing_markers_short_warmup_and_too_few_samples_reject(self):
        rows, markers = qualified_fixture()
        self.assertFalse(G.qualify(rows, [], options())["activity_qualified"])
        markers[0]["warm_seconds"] = 2.99
        self.assertFalse(G.qualify(rows, markers, options())["activity_qualified"])
        markers[0]["warm_seconds"] = 3
        self.assertFalse(G.qualify(rows, markers, options(min_samples=5))["activity_qualified"])

    def test_invalid_setup_outside_measurement_does_not_invalidate_steady_window(self):
        rows, markers = qualified_fixture()
        rows[2]["valid"] = False
        markers[0]["monotonic_ns"] = 800000000
        self.assertTrue(G.qualify(rows, markers, options())["activity_qualified"])

    def test_malformed_marker_detected(self):
        with patch.object(Path, "glob", return_value=[Path("10.jsonl")]), patch.object(
                Path, "read_text", return_value=json.dumps({"kind": "steady_begin"}) + "\n"):
            markers, errors = G.read_markers("unused")
        self.assertFalse(markers)
        self.assertTrue(errors)


if __name__ == "__main__":
    unittest.main(verbosity=2)
