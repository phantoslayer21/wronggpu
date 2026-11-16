# SPDX-License-Identifier: MIT
"""Model M on Model L's FP16-form shared-stage kernels (runtime WG_M_FORM=q7): the runtime's Model M frame on HIP
with its ten shared stages (enc1, enc2, b0 ... b5, dec2, dec1) on the q7 kernels and Model M's weights, checked stage by
stage against the FP16-form oracle (clean/l/reference/oracle.py run_stage_q7) on the runtime's own stage inputs.
Correctness only: no timing (the q7 kernels are correctness kernels until the Q7-FAST work lands).

    python clean/runner/m_q7_check.py --build <dir> --rocm <ROCm> --run              # GPU, inside a GPU slot
    python clean/runner/m_q7_check.py --build <dir> --compare --json out.json        # CPU
    python clean/runner/m_q7_check.py --build <dir> --rocm <ROCm> --run-d3d12        # GPU: native D3D12 + HIP pack12

<dir>: a build prepared as clean/l/runner/l_frame_check.py --prepare does (Model M's and Model L's code objects incl.
l_q7.co / l_q7w.co, the runtime library with WG_MODEL_L, <dir>/lframe_model with the synthetic Model M in raw/ and its
tensors in params.pkl). The frames, sentinels and per-stage metrics are l_frame_check's (SEQUENCE, frame_inputs,
fp8_stage). --run runs the sequence twice for Model M: WG_M_FORM=q7 (the route under test) and the default form (its
own blocks, the reference point for the deliverable's "functional M"). --compare checks, per frame and shared stage:
  - q7 route vs the FP16-form oracle with Model M's weights: every output (skip / merged / out) byte-identical
    (0 differing values; the kernels reproduce the oracle exactly as Model L's do);
  - negative control 1: the same q7 outputs vs Model M's functional reference (clean/reference/swin.py run_stage, the
    form Model M's own blocks implement) must DIFFER (the switch really changed the arithmetic);
  - negative control 2: the oracle with the stage's output bias b2[0] moved by +1.0 must DIFFER from the q7 outputs
    (the comparison can fail).
The default-form run is checked against the functional reference under Model M's per-stage rule (the existing
contract) so that both sides of the deliverable are on record.
Native D3D12 (--run-d3d12: form "q7d", backend 3, WG_M_FORM=q7 and WG_D3D12_FUSION=none so that every stage output is
written; <dir>/dx12 holds the q7 ports l_q7_*): --compare then checks per frame and shared stage
  - q7d vs the HIP q7 run: every output byte-identical (the D3D12 ports reproduce their HIP kernels);
  - q7d vs the FP16-form oracle on its own stage inputs: byte-identical;
  - negative control: q7d vs the HIP default-form run (Model M's own blocks) must DIFFER; the D3D12 run's sentinel
    (0x3C) differs from the HIP runs' (0xC3 / 0xA5), so an output the D3D12 route never wrote cannot match;
  - every raw output of the frame (stage maps, dec0's pack12 images, post, state) of q7d against form q7h12 (HIP,
    WG_M_FORM=q7, default kernels: dec0 by d0p12/coef_decoder, the decoder the D3D12 port d1p12/dec0_q7 reproduces):
    byte-identical; unwritten only the previous-frame state of frame 0 and the display image under the fused pack12
    post (both sides keep their sentinels there)."""
import argparse
import copy
import json
import os
import pickle
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from clean.l.runner import l_frame_check as LF  # noqa: E402

FORMS = ("q7", "m")
SENT = {"q7": 0xC3, "m": 0xA5, "q7d": 0x3C, "q7h12": 0x96, "hyb": 0x69, "hybmut": 0x5C}
# Model M's HYBRID evaluation form (Q11 part C): form "hyb" = WG_M_FORM=hybrid (l_q7w__hyb.co, float32 chains),
# checked against the oracle's FP8 op model "f32chain" (byte-identical per stage), negative control: the exact q7
# oracle must DIFFER; form "hybmut" = WG_M_FORM=hybrid_mutant (the Wqk chains keep an FP16 rounding per chunk): it must
# DIFFER from the f32chain oracle (the check catches the mutant).
HYB_ENV = {"hyb": "hybrid", "hybmut": "hybrid_mutant"}


def rec_path(build, form):
    return os.path.join(build, "mq7_M_%s.pkl" % form)


def run_form(build, rocm, form):
    """one form in this process (the runtime library reads WG_M_FORM once, when this process first loads it)"""
    if form != "q7d":
        from clean.runner.hip import clean_process_environment
        clean_process_environment(rocm)
    from clean.runner.pipeline import Pipeline
    d = LF.model_dir(build)
    flags = {"depth_inverted": False, "auto_exposure": True}
    # six bottleneck launches on the default form so that b0 ... b4 are written there too
    if form == "q7d":   # native D3D12 (backend 3), D3D12 fusions off (WG_D3D12_FUSION=none, set by run_d3d12)
        p = Pipeline(build, "", d, LF.MAX_RENDER, LF.OUTPUT, flags, backend="d3d12", check_packing=True, kernels={})
    elif form == "q7h12":   # HIP with the default kernels: pack12 frames decoded by d0p12/coef_decoder
        p = Pipeline(build, rocm, d, LF.MAX_RENDER, LF.OUTPUT, flags, check_packing=True, kernels={})
    else:
        p = Pipeline(build, rocm, d, LF.MAX_RENDER, LF.OUTPUT, flags, check_packing=True,
                     kernels={"pack": "none", "bneck": "none"})
    rng = np.random.default_rng(LF.SEED)
    frames = []
    for i, ((rw, rh), reset) in enumerate(LF.SEQUENCE):
        fr = LF.frame_inputs(rng, rw, rh, reset, True)
        LF.fill_sentinels(p, SENT[form], i == 0)
        out = p.frame(fr, debug=True)
        frames.append({"inputs": fr, "out": out, "raw": LF.raw_outputs(p) if form in ("q7d", "q7h12") else {}})
    rec = {"model": "M", "form": form, "sentinel": SENT[form], "kernels": p.kernels(), "notes": p.notes(), "model_source": p.model_source,
           "frames": frames, "identity": LF.capture_identity(build)}
    with open(rec_path(build, form), "wb") as f:
        pickle.dump(rec, f)
    print(form, "frames", len(frames), "m_form", (rec["kernels"] or {}).get("m_form", "-"))


def run(build, rocm):
    """both forms, each in its own process with its own environment (WG_M_FORM=q7 / unset)"""
    import subprocess
    for form in FORMS:
        env = dict(os.environ)
        env.pop("WG_M_FORM", None)
        if form == "q7":
            env["WG_M_FORM"] = "q7"
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--build", build, "--rocm", rocm,
                            "--run-form", form], env=env)
        if r.returncode != 0:
            raise SystemExit("form %s: exit %d" % (form, r.returncode))


def run_hybrid(build, rocm):
    """forms hyb and hybmut (HIP), each in its own process with its WG_M_FORM"""
    import subprocess
    for form, mf in HYB_ENV.items():
        env = dict(os.environ)
        env["WG_M_FORM"] = mf
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--build", build, "--rocm", rocm, "--run-form",
                            form], env=env)
        if r.returncode != 0:
            raise SystemExit("form %s: exit %d" % (form, r.returncode))


def run_d3d12(build, rocm):
    """form q7d (native D3D12, WG_D3D12_FUSION=none) and form q7h12 (HIP, default kernels: the pack12 decoder the D3D12
    route's dec0 port reproduces), each in its own process with WG_M_FORM=q7 (read when the runtime library loads)"""
    import subprocess
    for form in ("q7d", "q7h12"):
        env = dict(os.environ)
        env["WG_M_FORM"] = "q7"
        if form == "q7d":
            env["WG_D3D12_FUSION"] = "none"
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--build", build, "--rocm", rocm, "--run-form",
                            form], env=env)
        if r.returncode != 0:
            raise SystemExit("form %s: exit %d" % (form, r.returncode))


def _perturbed(p, stage):
    """the stage's output bias b2 of channel 0 moved by +1.0 (a change every token's channel 0 must show unless
    clamped away; a one-FP8-step change of a single W1 weight was too weak: in b3 / b5 it moved no output code)"""
    q = {k: np.array(v, copy=True) for k, v in p.items()}
    q["b2"].reshape(-1)[0] += np.float32(1.0)
    return q


def compare(build, json_path):
    from clean.l.reference import oracle as L
    from clean.l.reference import table as T
    from clean.reference import swin as S
    with open(os.path.join(LF.model_dir(build), "params.pkl"), "rb") as f:
        params = pickle.load(f)["M"]
    table = copy.deepcopy(T.TABLE)
    table["consts"]["default"]["gelu"] = "sigmoid"   # as l_frame_check (the kernels' GELU for the M block form)
    table["shared_form"] = "q7"
    table_h = copy.deepcopy(table)
    table_h["mm8"] = "f32chain"
    res = {"forms": {}}
    recs = {}
    for form in FORMS + ("q7d", "q7h12", "hyb", "hybmut"):
        if form in ("q7d", "q7h12", "hyb", "hybmut") and not os.path.exists(rec_path(build, form)):
            continue
        if form == "q7h12":   # raw outputs only (compared below)
            with open(rec_path(build, form), "rb") as f:
                recs[form] = pickle.load(f)
            continue
        with open(rec_path(build, form), "rb") as f:
            rec = recs[form] = pickle.load(f)
        frames = []
        for k, fd in enumerate(rec["frames"]):
            out = fd["out"]
            info = out["state"]["info"]
            f = int(info["parity"])
            x, skips = out["enc0"]["features"], {}
            m = {"frame": k, "stages": {}}
            for s in T.SHARED_STAGES:
                off = S.window_offset(s, f)
                kind = S.STAGES[s]["kind"]
                skip = skips.get("enc2" if s == "dec2" else "enc1") if kind == "dec" else None
                ent = {}
                if form == "q7":
                    ref = L.run_stage_q7(s, params[s], x, off, skip, table)
                    ent["vs_fp16_oracle"] = {key: LF.same_values(out[s][key], ref[key]) for key in out[s]}
                    fun = S.run_stage(s, params[s], x, off, skip, gelu="sigmoid")
                    ent["neg_vs_functional"] = {key: LF.same_values(out[s][key], fun[key]) for key in out[s]}
                    bad = L.run_stage_q7(s, _perturbed(params[s], s), x, off, skip, table)
                    ent["neg_perturbed_weight"] = {key: LF.same_values(out[s][key], bad[key]) for key in out[s]}
                    ent["pass"] = (all(v["differ"] == 0 for v in ent["vs_fp16_oracle"].values()) and
                                   not all(v["differ"] == 0 for v in ent["neg_vs_functional"].values()) and
                                   not all(v["differ"] == 0 for v in ent["neg_perturbed_weight"].values()))
                elif form in ("hyb", "hybmut"):
                    refh = L.run_stage_q7(s, params[s], x, off, skip, table_h)
                    ent["vs_f32chain_oracle"] = {key: LF.same_values(out[s][key], refh[key]) for key in out[s]}
                    if form == "hyb":
                        ref = L.run_stage_q7(s, params[s], x, off, skip, table)
                        ent["neg_vs_exact_q7_oracle"] = {key: LF.same_values(out[s][key], ref[key]) for key in out[s]}
                        ent["pass"] = (all(v["differ"] == 0 for v in ent["vs_f32chain_oracle"].values()) and
                                       not all(v["differ"] == 0 for v in ent["neg_vs_exact_q7_oracle"].values()))
                    else:   # the mutant: judged over the whole run below (it must differ somewhere)
                        ent["pass"] = True
                elif form == "q7d":
                    hq = recs["q7"]["frames"][k]["out"][s]
                    hm = recs["m"]["frames"][k]["out"][s]
                    ent["vs_hip_q7"] = {key: LF.same_values(out[s][key], hq[key]) for key in out[s]}
                    ref = L.run_stage_q7(s, params[s], x, off, skip, table)
                    ent["vs_fp16_oracle"] = {key: LF.same_values(out[s][key], ref[key]) for key in out[s]}
                    ent["neg_vs_hip_m_form"] = {key: LF.same_values(out[s][key], hm[key]) for key in out[s]}
                    ent["pass"] = (all(v["differ"] == 0 for v in ent["vs_hip_q7"].values()) and
                                   all(v["differ"] == 0 for v in ent["vs_fp16_oracle"].values()) and
                                   not all(v["differ"] == 0 for v in ent["neg_vs_hip_m_form"].values()))
                else:
                    fun = S.run_stage(s, params[s], x, off, skip, gelu="sigmoid")
                    ent["vs_functional_rule"] = {key: LF.fp8_stage(out[s][key], fun[key]) for key in out[s]}
                    ent["pass"] = all(v["pass"] for v in ent["vs_functional_rule"].values())
                m["stages"][s] = ent
                if kind == "enc":
                    skips[s] = out[s]["skip"]
                    x = out[s]["merged"]
                else:
                    x = out[s]["out"]
            m["pass"] = all(e["pass"] for e in m["stages"].values())
            frames.append(m)
        res["forms"][form] = {"kernels": rec["kernels"], "model_source": rec["model_source"],
                              "identity": rec.get("identity"), "frames": frames,
                              "pass": all(m["pass"] for m in frames)}
        if form == "hybmut":
            nd = sum(v["differ"] for fr in frames for e in fr["stages"].values() for v in e["vs_f32chain_oracle"].values())
            res["forms"][form]["mutant_differing_values"] = nd
            res["forms"][form]["pass"] = nd > 0          # caught
        if form in ("hyb", "hybmut"):
            res["forms"][form]["route_seen"] = (rec["kernels"] or {}).get("m_form") == HYB_ENV[form]
            res["forms"][form]["pass"] = res["forms"][form]["pass"] and res["forms"][form]["route_seen"]
        if form in ("q7", "q7d"):
            res["forms"][form]["route_seen"] = (rec["kernels"] or {}).get("m_form") == "q7"
            res["forms"][form]["pass"] = res["forms"][form]["pass"] and res["forms"][form]["route_seen"]
        print(form, "PASS" if res["forms"][form]["pass"] else "FAIL", "frames", len(frames))
        for m in frames:
            if not m["pass"]:
                print("  frame", m["frame"], "failing:", [s for s, e in m["stages"].items() if not e["pass"]])
    if "q7d" in recs and "q7h12" in recs:
        # every raw output of the frame (exposure ... dec0's pack12 images, post, state), D3D12 against HIP, both on q7
        # with pack12 dec0 (D3D12: d1p12/dec0_q7; HIP: d0p12/coef_decoder); the q7 route writes every stage map, so
        # only the previous-frame state of frame 0 is unwritten (both sides keep their sentinels there)
        # ... and the display image when both run the fused pack12 post (post_tf: it writes none). Both sides decode to
        # pack12, so the images are compared in that layout (no conversion).
        no_disp = all("post_tf" in str(recs[f]["kernels"].get("post_last", "")) for f in ("q7h12", "q7d"))
        unw = lambda k, n: ("both" if (k == 0 and n in LF.UNWRITTEN_ON_FIRST) or (n == "post.display" and no_disp)
                            else None)
        raw = LF.raw_compare(recs["q7h12"], recs["q7d"], SENT["q7h12"], SENT["q7d"], unw, (), convert_b=False)
        res["d3d12_vs_hip_raw"] = {"summary": LF.raw_summary(raw), "frames": raw,
                                   "kernels_hip": recs["q7h12"]["kernels"], "kernels_d3d12": recs["q7d"]["kernels"]}
        ok = all(f["identical"] for f in raw)
        print("q7d vs q7h12 raw:", "PASS" if ok else "FAIL", LF.raw_summary(raw))
        res["forms"]["q7d"]["pass"] = res["forms"]["q7d"]["pass"] and ok
    res["pass"] = all(v["pass"] for v in res["forms"].values())
    with open(json_path, "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return 0 if res["pass"] else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--rocm", default="")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--run-form", choices=FORMS + ("q7d", "q7h12", "hyb", "hybmut"), help=argparse.SUPPRESS)
    ap.add_argument("--run-hybrid", action="store_true")
    ap.add_argument("--run-d3d12", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--json", default="m_q7_check.json")
    a = ap.parse_args(argv)
    if a.run_form:
        run_form(a.build, a.rocm, a.run_form)
        return 0
    if a.run:
        run(a.build, a.rocm)
    if a.run_hybrid:
        run_hybrid(a.build, a.rocm)
    if a.run_d3d12:
        run_d3d12(a.build, a.rocm)
    if a.compare:
        return compare(a.build, a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
