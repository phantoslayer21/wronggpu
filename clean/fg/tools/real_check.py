# SPDX-License-Identifier: GPL-3.0-only
"""Frame generation (issue #96): check that our converter converts a player's own frame-generation library with the
preset-FG map, recording digests and the verdict only.

    python -m clean.fg.tools.real_check --exe <wronggpu-convert.exe> --dll <one exact library path>
                                        [--map data/fg.v1.map.json] [--work <empty folder>] [--record <file.json>]

Execute-only: the converter reads the library; this tool never opens it. It runs the converter once (no --raw, a
console-less child below normal priority), keeps from its output only the verdict, the map's SHA-256, the library's
file version and size, and the number of matching stage digests, then deletes the converted model file it wrote
(after taking its SHA-256). Offsets, values and bytes of the library are never recorded or printed; the converter's
log is not kept.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile

SUCCESS = "The model was converted successfully."
MODEL = "model.wgcm"


def parse(code, stdout, stderr):
    """The record of one converter run, from its exit code and output. Only the fields listed here are kept."""
    def one(pattern):
        m = re.search(pattern, stderr)
        return m.group(1) if m else None

    matched = one(r"all (\d+) stage digests match")
    size = one(r"\bsize: (\d+) bytes")
    rec = {
        "exit_code": code,
        "converted": code == 0 and SUCCESS in stdout,
        "message": stdout.strip().splitlines()[0] if stdout.strip() else "",
        "map_sha256": one(r"map sha256 ([0-9a-f]{64})"),
        "file_version": one(r"file version: (\d+(?:\.\d+){1,3})"),
        "file_size": int(size) if size else None,
        "stage_digests_matched": int(matched) if matched else 0,
    }
    stages = one(r"map sha256 [0-9a-f]{64}, (\d+) stages")
    rec["stages_in_map"] = int(stages) if stages else None
    rec["verdict"] = "pass" if (rec["converted"] and rec["stages_in_map"] and
                                rec["stage_digests_matched"] == rec["stages_in_map"]) else "fail"
    return rec


def _popen_flags():
    if os.name != "nt":
        return {}
    return {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS}


def run(cmd_prefix, map_path, dll, work, threads=2, timeout=600):
    """Runs `cmd_prefix + converter arguments` once and returns the record. `work`: an empty or missing folder; the
    converter writes its model file there, which is removed afterwards (file by file, only what it wrote)."""
    out = os.path.join(work, "out")
    os.makedirs(work, exist_ok=True)
    if os.path.exists(out):
        raise SystemExit("work folder already has an out/ folder: %s" % out)
    cmd = list(cmd_prefix) + ["--map", map_path, "--dll", dll, "--out", out, "--threads", str(threads)]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, **_popen_flags())
    rec = parse(p.returncode, p.stdout.decode("utf-8", "replace"), p.stderr.decode("utf-8", "replace"))
    model = os.path.join(out, MODEL)
    rec["model_sha256"] = None
    if os.path.isfile(model):
        with open(model, "rb") as f:
            rec["model_sha256"] = hashlib.sha256(f.read()).hexdigest()
        rec["model_size"] = os.path.getsize(model)
        os.remove(model)
    if os.path.isdir(out) and not os.listdir(out):
        os.rmdir(out)
    with open(map_path, "rb") as f:
        rec["map_file_sha256"] = hashlib.sha256(f.read()).hexdigest()
    return rec


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--exe", required=True, help="wronggpu-convert executable")
    ap.add_argument("--dll", required=True, help="one exact path of the frame-generation library")
    ap.add_argument("--map", default=os.path.normpath(os.path.join(here, "..", "..", "..", "data", "fg.v1.map.json")))
    ap.add_argument("--work", default=None, help="folder for the converter's output (removed again); default: a temp folder")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--record", default=None, help="write the record (JSON) here as well")
    a = ap.parse_args(argv)
    work = a.work or tempfile.mkdtemp(prefix="fg-real-check-")
    rec = run([a.exe], a.map, a.dll, work, threads=a.threads)
    text = json.dumps(rec, indent=1, sort_keys=True)
    print(text)
    if a.record:
        with open(a.record, "w", encoding="utf-8", newline="\n") as f:
            f.write(text + "\n")
    if a.work is None and not os.listdir(work):
        os.rmdir(work)
    return 0 if rec["verdict"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
