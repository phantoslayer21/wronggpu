# SPDX-License-Identifier: MIT
"""Validate/summarize original WG_D3D12_PROFILE diagnostic JSONL, without running GPU work.

Basis: native device's original capture schema; public Microsoft D3D12 Timing and ResolveQueryData contracts.
These timestamp intervals perturb GPU scheduling. They are attribution evidence, not a substitute for the
unprofiled interleaved whole-frame benchmark. CPU scopes are nested and must not be added together.
"""
import argparse
from collections import defaultdict
import copy
import json
import math
from pathlib import Path


KINDS = {"dispatch", "fill", "copy", "upload", "download", "barrier"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, f"invalid {label}: {value!r}")
    return value


def validate(rows, required_kinds=()):
    sessions = {}
    for line, row in enumerate(rows, 1):
        require(isinstance(row, dict), f"line {line}: expected an object")
        session = row.get("session")
        require(isinstance(session, str) and session, f"line {line}: missing session")
        kind = row.get("type")
        if kind == "meta":
            require(session not in sessions, f"line {line}: duplicate session metadata")
            require(row.get("version") == 1 and row.get("diagnostic_only") is True,
                    f"line {line}: unsupported capture version or missing diagnostic flag")
            integer(row.get("frequency_hz"), "timestamp frequency", 1)
            sessions[session] = {"meta": row, "gpu": [], "cpu": [], "end": None}
            continue
        require(session in sessions, f"line {line}: record precedes metadata")
        target = sessions[session]
        require(target["end"] is None, f"line {line}: record follows session end")
        if kind == "end":
            require(row.get("valid") is True and not row.get("error"), f"line {line}: unsuccessful capture")
            integer(row.get("gpu_operations"), "GPU operation count")
            target["end"] = row
        elif kind == "gpu":
            integer(row.get("sequence"), "operation sequence", 1)
            integer(row.get("fence"), "submission fence", 1)
            integer(row.get("bytes"), "byte count")
            start = integer(row.get("start_tick"), "start tick")
            stop = integer(row.get("stop_tick"), "stop tick")
            require(row.get("valid") is True and stop >= start, f"line {line}: invalid GPU interval")
            require(row.get("kind") in KINDS, f"line {line}: unsupported operation kind")
            require(isinstance(row.get("name"), str) and row["name"], f"line {line}: missing operation name")
            expected = (stop - start) * 1000.0 / target["meta"]["frequency_hz"]
            require(isinstance(row.get("ms"), (int, float)) and math.isfinite(row["ms"]) and
                    math.isclose(row["ms"], expected, abs_tol=1e-9, rel_tol=1e-9),
                    f"line {line}: GPU interval conversion mismatch")
            target["gpu"].append(row)
        elif kind == "cpu":
            start = integer(row.get("start_ns"), "CPU start")
            stop = integer(row.get("stop_ns"), "CPU stop")
            integer(row.get("bytes"), "CPU byte count")
            require(stop >= start and isinstance(row.get("name"), str) and row["name"],
                    f"line {line}: invalid CPU interval")
            require(isinstance(row.get("ms"), (int, float)) and math.isfinite(row["ms"]) and
                    math.isclose(row["ms"], (stop - start) / 1e6, abs_tol=1e-9, rel_tol=1e-9),
                    f"line {line}: CPU interval conversion mismatch")
            target["cpu"].append(row)
        else:
            raise ValueError(f"line {line}: unsupported record type {kind!r}")

    require(sessions, "capture has no sessions")
    gpu_totals = defaultdict(lambda: {"count": 0, "ms": 0.0, "bytes": 0})
    cpu_totals = defaultdict(lambda: {"count": 0, "ms": 0.0, "bytes": 0})
    result = []
    seen_kinds = set()
    for session, data in sessions.items():
        require(data["end"] is not None, f"session {session}: missing completion record")
        ops = sorted(data["gpu"], key=lambda r: r["sequence"])
        count = data["end"]["gpu_operations"]
        require(len(ops) == count and [r["sequence"] for r in ops] == list(range(1, count + 1)),
                f"session {session}: lost, duplicated or uncompleted GPU operations")
        previous = None
        submission = defaultdict(list)
        for op in ops:
            if previous:
                require(op["fence"] >= previous["fence"] and op["start_tick"] >= previous["stop_tick"],
                        f"session {session}: GPU ordering is inconsistent with the native queue")
            previous = op
            seen_kinds.add(op["kind"])
            total = gpu_totals[f'{op["kind"]}/{op["name"]}']
            total["count"] += 1
            total["ms"] += op["ms"]
            total["bytes"] += op["bytes"]
            submission[op["fence"]].append(op)
        for cpu in data["cpu"]:
            total = cpu_totals[cpu["name"]]
            total["count"] += 1
            total["ms"] += cpu["ms"]
            total["bytes"] += cpu["bytes"]
        result.append({
            "session": session, "gpu_operations": count,
            "submissions": [{"fence": fence, "operations": len(batch),
                             "first_to_last_operation_ms":
                                 (batch[-1]["stop_tick"] - batch[0]["start_tick"]) * 1000 /
                                 data["meta"]["frequency_hz"],
                             "summed_operation_ms": sum(r["ms"] for r in batch)}
                            for fence, batch in submission.items()],
        })
    require(set(required_kinds) <= seen_kinds, f"missing operation kinds: {sorted(set(required_kinds) - seen_kinds)}")
    return {"valid": True, "diagnostic_only": True,
            "limitations": "Instrumented intervals perturb scheduling; barrier spans are not isolated hardware cost. "
                           "CPU scopes overlap. Submission spans are not frame spans.",
            "sessions": result, "gpu_by_operation": dict(gpu_totals), "cpu_by_scope": dict(cpu_totals)}


def self_test():
    meta = {"type": "meta", "version": 1, "session": "fixture", "diagnostic_only": True, "frequency_hz": 1000}
    gpu = {"type": "gpu", "session": "fixture", "sequence": 1, "fence": 1, "kind": "dispatch",
           "name": "original_fixture", "bytes": 0, "start_tick": 10, "stop_tick": 20, "ms": 10, "valid": True}
    end = {"type": "end", "session": "fixture", "gpu_operations": 1, "valid": True, "error": ""}
    rows = [meta, gpu, end]
    validate(rows, ["dispatch"])
    mutations = []
    for key, value in (("valid", False), ("stop_tick", 9), ("ms", 9), ("sequence", 2), ("fence", 0)):
        bad = copy.deepcopy(rows)
        bad[1][key] = value
        mutations.append(bad)
    mutations.extend([rows[:-1], [meta, gpu, gpu, end], [meta, end]])
    detected = 0
    for bad in mutations:
        try:
            validate(bad)
        except ValueError:
            detected += 1
        else:
            raise AssertionError("deliberately broken profile was accepted")
    print(json.dumps({"self_test": "pass", "deliberate_mutations_detected": detected}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", nargs="*")
    parser.add_argument("--require-kind", action="append", choices=sorted(KINDS), default=[])
    parser.add_argument("--json", help="save the complete attribution summary")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    if not args.capture:
        if args.self_test:
            return 0
        parser.error("provide at least one capture or --self-test")
    rows = []
    for capture in args.capture:
        rows.extend(json.loads(line) for line in Path(capture).read_text().splitlines() if line.strip())
    summary = validate(rows, args.require_kind)
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
