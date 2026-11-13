# SPDX-License-Identifier: MIT
"""Capture Windows GPU-engine evidence without treating missing counters as idle.

Caller acquires the GPU lock (WG_GPU_LOCK, BUILDING.md) nonblocking before GPU commands. This wrapper
does not acquire a lock, stop other processes, or change any system setting.
--probe-seconds collects counters only (no GPU workload and no timing claim).

Measured children append steady_begin/steady_end JSON objects to their own
<pid>.jsonl file in WG_GPU_EVIDENCE_MARKER_DIR. Required keys: kind, phase, pid,
monotonic_ns (time.perf_counter_ns()), warm_seconds; config/round are retained.
All overlapping counter intervals must be valid; invalid edge intervals are
never silently removed from a measurement. Raw invalid values are preserved.
GPU-engine cumulative-count/time discontinuities and individual engine rates
above 100% make time-local accounting unknown, even for workload-owned counters.
No public interval-error bound is assumed: values are never capped or excused.

Original implementation from Microsoft PDH / Toolhelp API documentation:
https://learn.microsoft.com/en-us/windows/win32/perfctrs/checking-pdh-interface-return-values
https://learn.microsoft.com/en-us/windows/win32/api/pdh/nf-pdh-pdhgetformattedcounterarrayw
https://learn.microsoft.com/en-us/windows/win32/api/pdh/nf-pdh-pdhgetrawcounterarrayw
https://learn.microsoft.com/en-us/windows/win32/api/tlhelp32/ns-tlhelp32-processentry32w
https://devblogs.microsoft.com/directx/gpus-in-the-task-manager/
"""
import argparse
import collections
import ctypes as C
import datetime
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

VALID = {0, 1}  # PDH_CSTATUS_VALID_DATA and PDH_CSTATUS_NEW_DATA
MORE_DATA = 0x800007D2
NO_INSTANCE = 0x800007D1
INVALID_DATA = 0xC0000BBA
STATUS_NAMES = {0: "VALID_DATA", 1: "NEW_DATA", NO_INSTANCE: "NO_INSTANCE",
                INVALID_DATA: "INVALID_DATA", MORE_DATA: "MORE_DATA"}
COUNTER_PATH = r"\GPU Engine(*)\Utilization Percentage"
INSTANCE = re.compile(r"^pid_(\d+)_luid_(0x[0-9a-f]+)_(0x[0-9a-f]+)_phys_(\d+)_eng_(\d+)_engtype_(.+)$", re.I)
DEFAULT_COMPETITOR = r"^(python(?:w)?(?:\d+(?:\.\d+)*)?\.exe|wg_.*|.*-Win64-Shipping\.exe)$"


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def status(value):
    value = int(value) & 0xFFFFFFFF
    return {"code": value, "hex": f"0x{value:08x}", "name": STATUS_NAMES.get(value, "OTHER")}


def instance_fields(name):
    match = INSTANCE.match(name)
    if not match:
        return {"pid": None, "engine": None}
    pid, high, low, physical, engine, kind = match.groups()
    return {"pid": int(pid), "adapter_luid": f"{high}_{low}", "physical_adapter": int(physical),
            "engine": int(engine), "engine_type": kind}


# Fixed-width ABI fields; ctypes.c_long is not a Windows DWORD on every host.
DWORD, HANDLE = C.c_uint32, C.c_void_p


class FileTime(C.Structure):
    _fields_ = [("low", DWORD), ("high", DWORD)]

    def integer(self):
        return self.low | self.high << 32


class RawValue(C.Structure):
    _fields_ = [("CStatus", DWORD), ("TimeStamp", FileTime), ("FirstValue", C.c_int64),
                ("SecondValue", C.c_int64), ("MultiCount", DWORD)]


class FormattedValue(C.Structure):
    # PDH_FMT_DOUBLE selects the double member of the eight-byte union.
    _fields_ = [("CStatus", DWORD), ("doubleValue", C.c_double)]


class RawItem(C.Structure):
    _fields_ = [("name", C.c_wchar_p), ("value", RawValue)]


class FormattedItem(C.Structure):
    _fields_ = [("name", C.c_wchar_p), ("value", FormattedValue)]


class ProcessEntry(C.Structure):
    _fields_ = [("size", DWORD), ("usage", DWORD), ("pid", DWORD), ("heap", C.c_size_t),
                ("module", DWORD), ("threads", DWORD), ("ppid", DWORD),
                ("priority", C.c_int32), ("flags", DWORD), ("name", C.c_wchar * 260)]


def bind(dll, name, args, result=DWORD):
    fn = getattr(dll, name)
    fn.argtypes, fn.restype = args, result
    return fn


class Processes:
    """Names, ancestry and creation identity only; no image/command-line reads."""
    def __init__(self):
        kernel = C.WinDLL("kernel32", use_last_error=True)
        self.snapshot = bind(kernel, "CreateToolhelp32Snapshot", [DWORD, DWORD], HANDLE)
        self.first = bind(kernel, "Process32FirstW", [HANDLE, C.POINTER(ProcessEntry)], C.c_int)
        self.next = bind(kernel, "Process32NextW", [HANDLE, C.POINTER(ProcessEntry)], C.c_int)
        self.open = bind(kernel, "OpenProcess", [DWORD, C.c_int, DWORD], HANDLE)
        self.times = bind(kernel, "GetProcessTimes", [HANDLE] + [C.POINTER(FileTime)] * 4, C.c_int)
        self.close = bind(kernel, "CloseHandle", [HANDLE], C.c_int)

    def read(self):
        begin = time.perf_counter_ns()
        handle = self.snapshot(2, 0)  # TH32CS_SNAPPROCESS
        if handle == C.c_void_p(-1).value:
            raise C.WinError(C.get_last_error())
        result = []
        try:
            entry = ProcessEntry()
            entry.size = C.sizeof(entry)
            good = self.first(handle, C.byref(entry))
            while good:
                row = {"pid": entry.pid, "ppid": entry.ppid, "name": entry.name, "created_filetime": None}
                proc = self.open(0x1000, False, entry.pid)  # QUERY_LIMITED_INFORMATION
                if proc:
                    try:
                        created, exited, kernel, user = FileTime(), FileTime(), FileTime(), FileTime()
                        if self.times(proc, C.byref(created), C.byref(exited), C.byref(kernel), C.byref(user)):
                            row["created_filetime"] = created.integer()
                        else:
                            row["identity_error"] = C.get_last_error()
                    finally:
                        self.close(proc)
                else:
                    row["identity_error"] = C.get_last_error()
                result.append(row)
                good = self.next(handle, C.byref(entry))
            error = C.get_last_error()
            if error != 18:  # ERROR_NO_MORE_FILES, including an empty snapshot
                raise C.WinError(error)
        finally:
            self.close(handle)
        return {"begin_ns": begin, "end_ns": time.perf_counter_ns(), "valid": True, "rows": result}


class Counters:
    """One persistent wildcard query; return codes and item statuses are distinct."""
    def __init__(self):
        dll = C.WinDLL("pdh", use_last_error=True)
        self.close = bind(dll, "PdhCloseQuery", [HANDLE])
        self.collect = bind(dll, "PdhCollectQueryData", [HANDLE])
        self.formatted = bind(dll, "PdhGetFormattedCounterArrayW", [HANDLE, DWORD, C.POINTER(DWORD), C.POINTER(DWORD), HANDLE])
        self.raw = bind(dll, "PdhGetRawCounterArrayW", [HANDLE, C.POINTER(DWORD), C.POINTER(DWORD), HANDLE])
        self.query, self.counter = HANDLE(), HANDLE()
        opened = bind(dll, "PdhOpenQueryW", [C.c_wchar_p, C.c_size_t, C.POINTER(HANDLE)])(None, 0, C.byref(self.query))
        if opened:
            raise RuntimeError(f"PdhOpenQueryW: {status(opened)}")
        added = bind(dll, "PdhAddEnglishCounterW", [HANDLE, C.c_wchar_p, C.c_size_t, C.POINTER(HANDLE)])(
            self.query, COUNTER_PATH, 0, C.byref(self.counter))
        if added:
            self.close(self.query)
            raise RuntimeError(f"PdhAddEnglishCounterW: {status(added)}")

    def array(self, raw=False):
        fn, typ = (self.raw, RawItem) if raw else (self.formatted, FormattedItem)
        prefix = [self.counter] if raw else [self.counter, 0x200 | 0x8000]  # DOUBLE | NOCAP100
        attempts = []
        for _ in range(3):
            size, count = DWORD(), DWORD()
            rc = fn(*prefix, C.byref(size), C.byref(count), None)
            attempts.append({"step": "size", "status": status(rc), "bytes": size.value, "count": count.value})
            if rc != MORE_DATA or not size.value:
                return {"status": status(rc), "attempts": attempts, "rows": []}
            buf = C.create_string_buffer(size.value)
            rc = fn(*prefix, C.byref(size), C.byref(count), buf)
            attempts.append({"step": "values", "status": status(rc), "bytes": size.value, "count": count.value})
            if rc == MORE_DATA:
                continue  # Re-query size; never trust a truncated-buffer size.
            if rc:
                return {"status": status(rc), "attempts": attempts, "rows": []}
            items = C.cast(buf, C.POINTER(typ))
            rows = []
            for i in range(count.value):
                item = items[i]
                row = {"instance": item.name, "status": status(item.value.CStatus)}
                if raw:
                    row.update(timestamp_filetime=item.value.TimeStamp.integer(), first=item.value.FirstValue,
                               second=item.value.SecondValue, multi_count=item.value.MultiCount)
                else:
                    number = item.value.doubleValue
                    row.update(value=number if math.isfinite(number) else None,
                               nonfinite=None if math.isfinite(number) else repr(number), **instance_fields(item.name))
                rows.append(row)
            return {"status": status(rc), "attempts": attempts, "rows": rows}
        return {"status": status(MORE_DATA), "attempts": attempts, "rows": []}

    def read(self):
        begin = time.perf_counter_ns()
        rc = self.collect(self.query)
        end = time.perf_counter_ns()
        return {"collect_begin_ns": begin, "collect_end_ns": end, "collect_status": status(rc),
                "formatted": self.array(), "raw": self.array(raw=True)}


def classify_interval(previous, current):
    """Churn is a diagnosis, not permission to call an invalid interval idle."""
    reasons, transitions = [], []
    rows = current["counters"]["formatted"]["rows"]
    previous_raw = {r["instance"]: r for r in previous["counters"]["raw"]["rows"]}
    current_raw = {r["instance"]: r for r in current["counters"]["raw"]["rows"]}
    before = set(previous_raw)
    after = {r["instance"] for r in rows}
    old_pids = {p["pid"] for p in previous["processes"]["rows"]}
    new_pids = {p["pid"] for p in current["processes"]["rows"]}
    for name in sorted(before - after):
        pid = instance_fields(name)["pid"]
        transitions.append({"instance": name, "kind": "disappeared_process_exit" if pid in old_pids - new_pids else "disappeared_instance"})
        reasons.append("counter_disappeared")
    for name in sorted(after - before):
        transitions.append({"instance": name, "kind": "new_instance"})
        reasons.append("new_counter_without_previous_sample")
    for side in (previous, current):
        counters = side["counters"]
        if counters["collect_status"]["code"] != 0:
            reasons.append("collect_api_error")
        for key in (("formatted", "raw") if side is current else ("raw",)):
            if counters[key]["status"]["code"] != 0 or not counters[key]["rows"]:
                reasons.append(f"{key}_array_unavailable")
        if not side["processes"].get("valid"):
            reasons.append("process_snapshot_unavailable")
    # Previous formatted values need an earlier rate sample and can be invalid
    # on priming; current rate and both raw endpoints must be valid.
    for side in (previous, current):
        if any(r["status"]["code"] not in VALID for r in side["counters"]["raw"]["rows"]):
            reasons.append("invalid_raw_counter")
    # This query is specifically GPU Engine utilization, whose running count
    # is cumulative. PDH has returned VALID even when that count decreased and
    # its formatted unsigned-difference rate became enormous. Do not unwrap a
    # counter or infer an idle/busy interval from discontinuous accounting.
    fields = ("first", "second", "timestamp_filetime")
    for name in sorted(before & set(current_raw)):
        old, new = previous_raw[name], current_raw[name]
        if old["status"]["code"] not in VALID or new["status"]["code"] not in VALID:
            continue
        problems = []
        if any(type(row.get(field)) is not int for row in (old, new) for field in fields):
            problems.append("missing_or_noninteger_raw_endpoint")
        else:
            if old["first"] < 0 or new["first"] < 0:
                problems.append("negative_cumulative_count")
            if new["first"] < old["first"]:
                problems.append("decreased_cumulative_count")
            for field in ("second", "timestamp_filetime"):
                if old[field] < 0 or new[field] < 0 or new[field] <= old[field]:
                    problems.append("nonpositive_raw_time_delta:" + field)
        if problems:
            reasons.append("raw_counter_accounting_unknown")
            transitions.append({"instance": name, "kind": "raw_counter_accounting_unknown",
                                "problems": problems,
                                "previous": {field: old.get(field) for field in fields},
                                "current": {field: new.get(field) for field in fields}})
    for row in rows:
        number = row.get("value")
        finite = isinstance(number, (int, float)) and not isinstance(number, bool) and math.isfinite(number)
        if row["status"]["code"] not in VALID or not finite or number < 0:
            pid = row.get("pid")
            diagnosis = "process_exited_invalid_counter" if pid in old_pids - new_pids else "invalid_counter_unresolved"
            transitions.append({"instance": row["instance"], "kind": diagnosis, "status": row["status"]})
            reasons.append(diagnosis)
        elif number > 100:
            # Check each PID/adapter/physical-engine instance, never a sum across
            # engines. Completion-sized increments are observed in our captures;
            # no documented per-interval error bound licenses an overshoot here.
            # Preserve the rate and mark its temporal interpretation unknown.
            reasons.append("counter_accounting_resolution_unknown")
            transitions.append({"instance": row["instance"], "kind": "counter_accounting_resolution_unknown",
                                "value": number, "status": row["status"]})
        if row.get("pid") is None:
            reasons.append("unparsed_instance")
    if after != set(current_raw):
        reasons.append("raw_formatted_instance_mismatch")
    return {"valid": not reasons, "reasons": sorted(set(reasons)), "transitions": transitions}


def monitor(out, interval):
    out = Path(out)
    processes, counters = Processes(), Counters()
    previous = None
    try:
        with (out / "telemetry.jsonl").open("w", encoding="utf-8") as log:
            while not (out / "stop").exists():
                begin = time.perf_counter_ns()
                phase = (out / "phase.txt").read_text().strip()
                try:
                    proc = processes.read()
                except Exception as error:
                    proc = {"valid": False, "rows": [], "error": repr(error)}
                current = {"utc": utc(), "phase": phase, "processes": proc, "counters": counters.read()}
                current["interval_end_ns"] = current["counters"]["collect_begin_ns"]
                current["interval_begin_ns"] = previous["interval_end_ns"] if previous else None
                current["phase_begin"] = previous["phase"] if previous else None
                current.update(classify_interval(previous, current) if previous else {"valid": False, "reasons": ["priming"], "transitions": []})
                current["monitor_work_ns"] = time.perf_counter_ns() - begin
                log.write(json.dumps(current, allow_nan=False) + "\n")
                log.flush()
                (out / "ready").touch()
                previous = current
                time.sleep(max(0, interval - (time.perf_counter_ns() - begin) / 1e9))
    finally:
        counters.close(counters.query)


def attribute(samples, root_pid, competitor_pids, competitor_names):
    """Only observed parent chains with non-reused identities count as workload."""
    known, owned, root_identity = {}, set(), None
    previous_processes = {}
    for sample in samples:
        rows = sample.get("processes", {}).get("rows", [])
        for proc in rows:
            known[proc["pid"]] = proc
            if proc["pid"] == root_pid and proc.get("created_filetime") and root_identity is None:
                root_identity = (proc["pid"], proc["created_filetime"])
        changed = True
        while changed:
            changed = False
            for proc in rows:
                identity = (proc["pid"], proc.get("created_filetime"))
                parent = known.get(proc["ppid"])
                parent_owned = parent and (parent["pid"], parent.get("created_filetime")) in owned
                if proc.get("created_filetime") and (identity == root_identity or
                        (parent_owned and parent["created_filetime"] <= proc["created_filetime"])):
                    if identity not in owned:
                        owned.add(identity)
                        changed = True
        visible = {**previous_processes, **{p["pid"]: p for p in rows}}
        totals, workload, active, unknown, protected = collections.defaultdict(float), 0.0, [], [], []
        competitors = []
        for counter in sample.get("counters", {}).get("formatted", {}).get("rows", []):
            if counter["status"]["code"] not in VALID or counter.get("value") is None:
                continue
            value, pid = counter["value"], counter.get("pid")
            proc = visible.get(pid)
            is_owned = bool(proc and (pid, proc.get("created_filetime")) in owned)
            if is_owned:
                workload += value
            else:
                totals[pid] += value
            if value > 0:
                detail = {**counter, "process": proc, "workload": is_owned}
                active.append(detail)
                if not proc:
                    unknown.append(pid)
                elif not proc.get("created_filetime"):
                    protected.append(pid)  # Count as unrelated; cannot join workload.
                if not is_owned and (pid in competitor_pids or proc and any(re.search(pattern, proc["name"], re.I) for pattern in competitor_names)):
                    competitors.append(detail)
        sample["attribution"] = {"workload_pids": sorted({pid for pid, _ in owned}),
                "active": active, "unrelated_by_pid": dict(totals),
                "unrelated_total_percent": sum(totals.values()),
                "unrelated_max_pid_percent": max(totals.values(), default=0.0),
                "strict_zero_unrelated": sum(totals.values()) == 0,
                "workload_total_percent": workload, "unknown_active_pids": sorted(set(unknown), key=str),
                "unverified_creation_active_pids": sorted(set(protected)),
                "active_declared_competitors": competitors}
        previous_processes = {p["pid"]: p for p in rows}
    return samples


def read_markers(directory):
    markers, errors = [], []
    for path in sorted(Path(directory).glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                marker = json.loads(line)
                if (marker.get("kind") not in ("steady_begin", "steady_end") or
                        not isinstance(marker.get("monotonic_ns"), int) or not isinstance(marker.get("pid"), int)):
                    raise ValueError("invalid marker")
                markers.append(marker)
            except (ValueError, TypeError) as error:
                errors.append(f"{path.name}: {error}")
    return sorted(markers, key=lambda m: m["monotonic_ns"]), errors


def qualify(samples, markers, args):
    def gate(rows, need_workload=False):
        reasons = []
        if len(rows) < args.min_samples:
            reasons.append("insufficient_samples")
        if any(not row.get("valid") for row in rows):
            reasons.append("invalid_telemetry_interval")
        if any(row["attribution"]["unknown_active_pids"] for row in rows):
            reasons.append("unknown_active_process_identity")
        if any(row["attribution"]["unrelated_max_pid_percent"] > args.max_unrelated_pid for row in rows):
            reasons.append("unrelated_pid_threshold_exceeded")
        if any(row["attribution"]["unrelated_total_percent"] > args.max_unrelated_total for row in rows):
            reasons.append("unrelated_total_threshold_exceeded")
        if any(row["attribution"]["active_declared_competitors"] for row in rows):
            reasons.append("active_declared_game_or_project_competitor")
        if need_workload and not any(row["attribution"]["workload_total_percent"] > 0 for row in rows):
            reasons.append("no_observed_workload_gpu_activity")
        return reasons

    phases = {}
    for phase in ("pre", "post"):
        rows = [s for s in samples if s.get("phase") == phase and s.get("phase_begin") == phase]
        reasons = gate(rows)
        phases[phase] = {"qualified": not reasons, "reasons": reasons, "sample_count": len(rows)}
    pending, windows, errors = {}, [], []
    for marker in markers:
        key = (marker["pid"], marker.get("phase"))
        if marker["kind"] == "steady_begin":
            if key in pending:
                errors.append(f"duplicate begin {key}")
            pending[key] = marker
            continue
        begin = pending.pop(key, None)
        if not begin:
            errors.append(f"unmatched end {key}")
            continue
        lo, hi = begin["monotonic_ns"], marker["monotonic_ns"]
        rows = [s for s in samples if s.get("interval_begin_ns") is not None and
                s["interval_begin_ns"] < hi and s["interval_end_ns"] > lo]
        reasons = gate(rows, need_workload=True)
        if hi <= lo:
            reasons.append("nonpositive_window")
        if begin.get("warm_seconds", 0) < 3:
            reasons.append("insufficient_reported_warmup")
        if not rows or rows[0]["interval_begin_ns"] > lo or rows[-1]["interval_end_ns"] < hi:
            reasons.append("window_not_fully_covered")
        if any(right["interval_begin_ns"] != left["interval_end_ns"] for left, right in zip(rows, rows[1:])):
            reasons.append("sampling_gap")
        if any((r["interval_end_ns"] - r["interval_begin_ns"]) / 1e9 > args.max_interval for r in rows):
            reasons.append("sampling_interval_too_long")
        windows.append({"begin": begin, "end": marker, "sample_count": len(rows),
                        "qualified": not reasons, "reasons": reasons,
                        "strict_zero_unrelated": bool(rows) and all(r["attribution"]["strict_zero_unrelated"] for r in rows),
                        "max_unrelated_pid_percent": max((r["attribution"]["unrelated_max_pid_percent"] for r in rows), default=None),
                        "max_unrelated_total_percent": max((r["attribution"]["unrelated_total_percent"] for r in rows), default=None)})
    if pending:
        errors.append(f"unclosed windows: {list(pending)}")
    return {"pre_post": phases, "windows": windows, "marker_errors": errors,
            "activity_qualified": bool(windows) and not errors and all(w["qualified"] for w in windows) and
                                  all(p["qualified"] for p in phases.values())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--powershell", default="pwsh", help="Only for adapter/driver metadata; counters use PDH directly")
    ap.add_argument("--interval", type=float, default=0.25)
    ap.add_argument("--pre", type=float, default=5)
    ap.add_argument("--post", type=float, default=5)
    ap.add_argument("--post-settle", type=float, default=1,
                    help="Fixed recorded retirement interval outside measurement/post windows")
    ap.add_argument("--min-samples", type=int, default=2)
    ap.add_argument("--max-interval", type=float, default=1.0)
    ap.add_argument("--max-unrelated-pid", type=float, default=5.0)
    ap.add_argument("--max-unrelated-total", type=float, default=10.0)
    ap.add_argument("--competitor-pid", type=int, action="append", default=[])
    ap.add_argument("--competitor-name", action="append", default=[DEFAULT_COMPETITOR], help="Regex; every active match rejects")
    ap.add_argument("--probe-seconds", type=float, help="Read-only counters, no workload or timing qualification")
    ap.add_argument("--monitor", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)
    if args.interval <= 0 or args.pre <= 0 or args.post <= 0 or args.post_settle < 0 or args.min_samples < 1:
        ap.error("positive intervals and sample counts required")
    out = Path(args.out).resolve()
    if args.monitor:
        monitor(out, args.interval)
        return 0
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if bool(command) == (args.probe_seconds is not None):
        ap.error("provide exactly one command or --probe-seconds")
    out.mkdir(parents=True, exist_ok=False)
    markers = out / "markers"
    markers.mkdir()
    phase, stop = out / "phase.txt", out / "stop"
    phase.write_text("pre")
    env = os.environ.copy()
    env["WG_GPU_EVIDENCE_MARKER_DIR"] = str(markers)
    no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    started, started_ns, pid, rc, error = utc(), time.perf_counter_ns(), None, None, None
    metadata = subprocess.run([args.powershell, "-NoProfile", "-Command",
        "Get-CimInstance Win32_VideoController | Select-Object Name,PNPDeviceID,DriverVersion,DriverDate | ConvertTo-Json"],
        capture_output=True, text=True, creationflags=no_window)
    (out / "adapter.json").write_text(metadata.stdout, encoding="utf-8")
    (out / "adapter.stderr").write_text(metadata.stderr, encoding="utf-8")
    with (out / "telemetry.stderr").open("w") as terr:
        mon = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--out", str(out),
              "--interval", str(args.interval), "--monitor"], stderr=terr, stdout=terr, creationflags=no_window)
        boundaries = {"pre_begin_ns": time.perf_counter_ns()}
        try:
            deadline = time.monotonic() + 15
            while not (out / "ready").exists() and mon.poll() is None and time.monotonic() < deadline:
                time.sleep(0.05)
            if not (out / "ready").exists():
                raise RuntimeError("telemetry monitor did not become ready")
            time.sleep(args.pre)
            phase.write_text("during")
            boundaries["during_begin_ns"] = time.perf_counter_ns()
            with (out / "command.log").open("w") as log:
                if args.probe_seconds is not None:
                    time.sleep(args.probe_seconds)
                    rc = 0
                else:
                    child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                    pid = child.pid
                    rc = child.wait()
            boundaries["during_end_ns"] = time.perf_counter_ns()
            phase.write_text("retire")
            time.sleep(args.post_settle)
            boundaries["post_begin_ns"] = time.perf_counter_ns()
            phase.write_text("post")
            time.sleep(args.post)
        except Exception as caught:
            error = repr(caught)
            rc = 1
        finally:
            boundaries["post_end_ns"] = time.perf_counter_ns()
            stop.touch()
            try:
                mon.wait(timeout=10)
            except subprocess.TimeoutExpired:
                mon.terminate()  # only this wrapper's monitor process
                mon.wait()
    telemetry_file = out / "telemetry.jsonl"
    samples, parse_errors = [], []
    if telemetry_file.exists():
        for number, line in enumerate(telemetry_file.read_text(encoding="utf-8").splitlines(), 1):
            try:
                samples.append(json.loads(line))
            except ValueError:
                parse_errors.append(f"malformed telemetry line {number}")
    samples = attribute(samples, pid, args.competitor_pid, args.competitor_name)
    with (out / "attributed.jsonl").open("w", encoding="utf-8") as output:
        for sample in samples:
            output.write(json.dumps(sample, allow_nan=False) + "\n")
    events, marker_errors = read_markers(markers)
    qualification = qualify(samples, events, args)
    qualification["marker_errors"].extend(marker_errors)
    qualified = (qualification["activity_qualified"] and not marker_errors and not parse_errors and rc == 0 and mon.returncode == 0
                 and metadata.returncode == 0 and args.probe_seconds is None and error is None)
    valid_collection = bool(samples) and not parse_errors and mon.returncode == 0 and all(
        row.get("valid") for row in samples if row.get("interval_begin_ns") is not None)
    result = {"schema": 2, "started_utc": started, "finished_utc": utc(), "started_ns": started_ns,
              "boundaries": boundaries, "command": command, "pid": pid, "returncode": rc,
              "monitor_returncode": mon.returncode, "error": error,
              "lock": "GPU command caller must hold the GPU lock (WG_GPU_LOCK); probe-only does not use GPU",
              "counter_path": COUNTER_PATH, "raw_recorded": bool(samples), "parse_errors": parse_errors,
              "telemetry": "valid" if valid_collection else "unknown",
              "valid_intervals": sum(s.get("valid", False) for s in samples), "interval_count": len(samples),
              "invalid_reasons": dict(collections.Counter(reason for s in samples for reason in s.get("reasons", []))),
              "criteria": {k: getattr(args, k) for k in ("min_samples", "max_interval", "max_unrelated_pid", "max_unrelated_total", "competitor_pid", "competitor_name")},
              "post_settle_seconds": args.post_settle,
              "qualification": qualification, "timing_qualified": qualified,
              "timing_note": "Qualification is sampled activity isolation under the recorded thresholds, not absolute GPU idle. "
                             "Raw cumulative-count/time discontinuities and individual engine rates above 100% make "
                             "accounting unknown, including workload counters; no interval-error tolerance is assumed. "
                             "All engines and processes count, including desktop/browser activity. Sums across engines can exceed 100%. "
                             "Warm-up duration is reported by the harness, not measured shader clocks. "
                             "Short bursts within sampling intervals, unobserved short-lived process ancestry and CPU contention are limitations. "
                             "Owner must identify gameplay/other project processes using recorded names and explicit competitor options."}
    dump(out / "run.json", result)
    print(json.dumps(result), flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
