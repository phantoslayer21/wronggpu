# SPDX-License-Identifier: GPL-3.0-only
"""Model L (issue #89): negative control of the oracle's FP16 block form. One recorded run of l_frame_check (HIP, every
kernel separate: lframe_L_stages.pkl) is checked twice in one process, frame by frame on the run's own stage inputs
(l_frame_check.compare_model):
- "model_m_form": the oracle as it was before the FP16 form (both of Model L's blocks in Model M's block arithmetic
  with the kernels' GELU, the decoder block without the skip input);
- "fp16_form": the oracle as it is (clean/l/reference/table.py TABLE: both blocks in the FP16 form, the decoder block
  reading the encoder block's output as its skip, which this check takes from the encoder oracle on the run's own
  inputs), with the FP8 matrix-op model --mm8.
Kernels that compute Model M's block form pass the first and must fail the second; kernels in the FP16 form the other
way round. Identity: the recording carries what --run captured (l_frame_check.capture_identity: the checkout's git
commit from git rev-parse HEAD, and the SHA-256 of the runtime library, params.pkl and source.json); this check re-hashes
the files it loads and refuses when the recording has no identity or any hash differs (l_frame_check.verify_identity).
Output: a JSON record with counts and digests only (the verified identity, the model's and map's SHA-256 from
source.json, the SHA-256 of the recorded run file and of the frames' inputs and outputs), the command line (the build
folder as <build>), the FP8 op model of each oracle, and the per-frame alignment checks (frame counter, parity, reset,
window offsets, as compare_model checks them).

    python clean/l/runner/l_frame_check.py --build <dir> --prepare --model <model.wgcm>   # CPU
    python clean/l/runner/l_frame_check.py --build <dir> --run                            # GPU
    python clean/l/runner/l_q7_negative_control.py --build <dir> --json out.json          # CPU
"""
import argparse
import copy
import hashlib
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)

from clean.l.reference import oracle as L  # noqa: E402
from clean.l.reference import table as T  # noqa: E402
from clean.l.runner import l_frame_check as F  # noqa: E402

DEC0_RULE = "A, B and latent codes and C values each at most 0.5 % differing (l_frame_check dec0)"


def model_m_form_table():
    """The oracle before the FP16 form: Model M's block arithmetic (the kernels' GELU) in both blocks, no skip."""
    t = copy.deepcopy(T.TABLE)
    t["consts"]["default"]["form"] = "m"
    t["consts"]["default"]["gelu"] = "sigmoid"
    t["shared_form"] = "m"            # the shared stages on Model M's oracle, as before their FP16 form
    for op in t["dec_ops"]:
        op.pop("skip", None)
    return t


def fp16_form_table(mm8):
    t = copy.deepcopy(T.TABLE)
    t["mm8"] = mm8
    return t


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_arrays(arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode() + str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def frame_digests(rec):
    """SHA-256 over every frame's inputs (sorted keys) and over every frame's recorded outputs (raw bytes)."""
    ins, outs = [], []
    for fd in rec["frames"]:
        for k in sorted(fd["inputs"]):
            v = fd["inputs"][k]
            ins.append(np.asarray(v) if not isinstance(v, (bool, int, float)) else np.asarray([v]))
        for k in sorted(fd["raw"]):
            outs.append(np.frombuffer(bytes(fd["raw"][k]), np.uint8))
    return {"inputs": _sha256_arrays(ins), "outputs_raw": _sha256_arrays(outs)}


def dec0_rows(frames):
    rows = []
    for m in frames:
        d = m["dec0"]
        rows.append({"frame": m["frame"], "render": m["render"], "parity": m["parity"], "reset": m["reset"],
                     "alignment": m["state"], "enc0_features_pass": m["enc0"]["features"]["pass"],
                     "dec0_pass": d["pass"],
                     "dec0_differ": {k: v["differ"] for k, v in d.items() if isinstance(v, dict)},
                     "dec0_values": {k: v["values"] for k, v in d.items() if isinstance(v, dict)}})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--config", default="stages", help="the recorded run (lframe_L_<config>.pkl)")
    ap.add_argument("--mm8", default="exact", choices=L.MM8_MODELS)
    ap.add_argument("--json", required=True)
    a = ap.parse_args(argv)
    run_path = F.rec_path(a.build, "L", a.config)
    with open(run_path, "rb") as f:
        rec = pickle.load(f)
    with open(os.path.join(F.model_dir(a.build), "params.pkl"), "rb") as f:
        params = pickle.load(f)["L"]
    with open(os.path.join(F.model_dir(a.build), "source.json")) as f:
        source = json.load(f)
    try:
        ident = F.verify_identity(rec, a.build)   # the files this check loads are the ones the run used
    except ValueError as e:
        print("REFUSED:", e)
        return 2
    cmd = [os.path.basename(sys.argv[0])] + [("<build>" if x == a.build else ("<out>" if x == a.json else x))
                                             for x in (argv if argv is not None else sys.argv[1:])]
    res = {"what": "negative control of Model L's oracle block form: one recorded run checked against the Model "
                   "M-form oracle and the FP16-form oracle in one process",
           "command": " ".join(["python", "clean/l/runner/" + cmd[0]] + cmd[1:]),
           "runtime_commit": ident["git_commit"],
           "identity_verified": ident,
           "recorded_run": {"file": os.path.basename(run_path), "sha256": _sha256_file(run_path),
                            "frames": len(rec["frames"]), "model": rec["model"], "kernels": rec["kernels"],
                            "frame_sha256": frame_digests(rec)},
           "model_source": source, "fp8_op_model": a.mm8, "dec0_rule": DEC0_RULE, "forms": {}}
    for name, table in (("model_m_form", model_m_form_table()), ("fp16_form", fp16_form_table(a.mm8))):
        frames = F.compare_model(rec, params, table=table)
        rows = dec0_rows(frames)
        res["forms"][name] = {"fp8_op_model": table["mm8"],
                              "dec0_frames_passed": sum(r["dec0_pass"] for r in rows),
                              "enc0_frames_passed": sum(r["enc0_features_pass"] for r in rows),
                              "alignment_frames_ok": sum(bool(r["alignment"]["pass"]) for r in rows),
                              "frames": rows}
        print(name, "dec0 frames passed", res["forms"][name]["dec0_frames_passed"], "of", len(rows),
              "| enc0", res["forms"][name]["enc0_frames_passed"], "| alignment ok",
              res["forms"][name]["alignment_frames_ok"], flush=True)
    from clean.l.runner import l_record as LR
    LR.write_l_record(res, a.json, table=fp16_form_table(a.mm8))
    return 0


if __name__ == "__main__":
    sys.exit(main())
