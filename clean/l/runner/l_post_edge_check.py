# SPDX-License-Identifier: MIT
"""Model L (issue #89): the reconstruction kernels' history sample at the image edges and at small, odd image sizes,
D3D12 against HIP byte for byte, HIP against the oracle, with a counter that proves the D3D12 interior fast path runs.

    python clean/l/runner/l_post_edge_check.py --build <dir> --rocm "C:\\Program Files\\AMD\\ROCm\\7.2" --prepare
    python clean/l/runner/l_post_edge_check.py --build <dir> --rocm "C:\\Program Files\\AMD\\ROCm\\7.2" --run --json out.json

--prepare (CPU): l_post.co (HIP), kernel_replay.exe and four D3D12 builds of the post ports (clean/dx12/build_dx12.py
--define): <dir>/dx12 as shipped; <dir>/hscount (LP_HSCOUNT=1: each wave taking the history fast path adds 1 to the
first dword after the full history); <dir>/mut8 (counter + LP_MUT=8: the fast path's predicate always false) and
<dir>/mut9 (LP_MUT=9: the predicate without its size guard, the bug of images narrower or shorter than 4).
--run (GPU, one GPU slot). The kernels l_post_recon, l_post_recon_p12, l_post_recon_ldr and l_post_recon_p12_ldr (the
history sample, history_cr, is shared by all four), and l_post_recon_p12 again with the no-fold port
(WG_TEST_FOLD_PROBE_FAIL=1), on every size of SIZES (1 x 1 ... 97 x 35: below 4, odd, a tile's interior boundary in the
middle of a wave) and every kind of KINDS, with a valid history (no reset):
- random: Model M's random inputs (random motions around a random base);
- center / left / right / top / bottom: every motion points at one history position: the centre, or just inside
  one edge (the clamped sample on every lane);
- in_lo / in_hi / out_lo / out_hi: the position at the fast path's boundary, ix (and iy) = 1 or size - 3 (inside: the
  4 x 4 footprint fits) and 0 or size - 2 (outside);
- split: even columns point inside, odd columns at an edge (every wave mixes both); split_half: in each 32-column
  tile the first 16 columns (half the width when narrower) inside, the rest at an edge (the boundary in the middle of
  a wave).
Checks:
- raw bytes, D3D12 == HIP, every output (full history, reduced, display; the l_d3d12_ops_check rules: each back end's
  own sentinel, the pitch padding and 64 slack bytes untouched), and the full history per edge band (the two columns
  or rows at each of the four edges) reported apart;
- the counter (hscount build, same bytes otherwise): 0 at every size below 4; otherwise, for the non-random kinds, > 0
  exactly when some wave's sampling lanes all have their footprint inside (wave_rows: e.g. center, in_lo, in_hi) and
  0 when none has (left, right, out_lo, out_hi, the mixed waves of split);
- negative controls: mut8 must break the counter expectation (the fast path gone is noticed), mut9 must differ from
  HIP (the old predicate reads past a history of height or width 3), and the run's first multi-tile case with half its
  workgroups must differ;
- the oracle: l_post_recon and l_post_recon_ldr (HIP) against Model L's reconstruction (clean/l/runner/l_post_check.py
  check_case: Model L's post acceptance) on every case.
Output: counts only, no timing.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)

from clean.l.runner import l_c32_check as C32  # noqa: E402
from clean.l.runner import l_d3d12_ops_check as OPS  # noqa: E402
from clean.l.runner.l_d3d12_check import D3D12_SENTINEL, HIP_SENTINEL, prepare_replay  # noqa: E402

KERNELS = ["l_post_recon", "l_post_recon_p12", "l_post_recon_ldr", "l_post_recon_p12_ldr"]
SIZES = [(1, 1), (2, 3), (3, 2), (3, 3), (3, 5), (5, 3), (4, 4), (5, 4), (4, 5), (7, 7), (33, 9), (37, 6), (67, 13),
         (97, 35)]
KINDS = ["random", "center", "left", "right", "top", "bottom", "in_lo", "in_hi", "out_lo", "out_hi", "split",
         "split_half"]
VARIANTS = {"hscount": ["LP_HSCOUNT=1"], "mut8": ["LP_HSCOUNT=1", "LP_MUT=8"], "mut9": ["LP_MUT=9"]}
SEED = 231


def _pos(kind, n, lo):
    """History coordinate q (q - 1/2 = t, ix = floor(t)) of one axis of size n for a kind; lo: the low side."""
    if kind == "center":
        return n // 2 + 0.25
    if kind in ("in_lo", "in_hi"):   # ix = 1 or n - 3
        return 2.25 if (kind == "in_lo") == lo else n - 2.25
    if kind in ("out_lo", "out_hi"):   # ix = 0 or n - 2
        return 1.25 if (kind == "out_lo") == lo else n - 1.25
    return n // 2 + 0.25


def _point(kind, dw, dh):
    if kind == "left":
        return 0.75, dh // 2 + 0.25
    if kind == "right":
        return dw - 0.25, dh // 2 + 0.25
    if kind == "top":
        return dw // 2 + 0.25, 0.75
    if kind == "bottom":
        return dw // 2 + 0.25, dh - 0.25
    return _pos(kind, dw, True), _pos(kind, dh, True)


def edge_case(rng, dw, dh, kind):
    """Model M's random post inputs at render size = output size (ratio 1, jitter 0: render pixel x is output pixel
    x, so every candidate motion is the pixel's own) with the motions of `kind` (FP16-exact)."""
    from clean.l.runner import l_post_check as LPC
    if kind == "random":
        rw, rh = max(1, (2 * dw) // 3), max(1, (2 * dh) // 3)
        return LPC.make_case(rng, rw, rh, dw, dh, False, "random")
    x = LPC.make_case(rng, dw, dh, dw, dh, False, "random")
    x["jitter"] = np.zeros(2)
    X, Y = np.meshgrid(np.arange(dw) + 0.5, np.arange(dh) + 0.5)
    if kind in ("split", "split_half"):
        inside = (np.arange(dw) % 2 == 0) if kind == "split" else (np.arange(dw) % 32 < min(16, dw // 2))
        qi, qe = _point("center", dw, dh), _point("left", dw, dh)
        qx = np.where(inside[None, :], qi[0], qe[0]) + 0 * X
        qy = np.where(inside[None, :], qi[1], qe[1]) + 0 * Y
    else:
        q = _point(kind, dw, dh)
        qx, qy = q[0] + 0 * X, q[1] + 0 * Y
    m = np.stack([qx - X, qy - Y], -1)
    assert np.array_equal(m.astype(np.float16).astype(np.float64), m), "motions must be FP16-exact"
    for k in ("m_near", "m_center", "m_far"):
        x[k] = m.astype(np.float16)
    return x


def wave_rows(x, dw, dh):
    """The kernel's history samples of the non-random kinds, per wave and row: [(inside, outside)] counts of the lanes
    that sample the history (valid position), as l_post.hlsl's grid runs them: 32-column tiles, wave k = row pair
    k0 + k (pairs past the end repeat the last), both rows (a second row past the image is computed too: its motion is
    the last row's), lanes past the width repeat the last column. Render size = output size, jitter 0."""
    m = x["m_near"].astype(np.float64)
    npairs = (dh + 1) // 2
    out = []
    for x0 in range(0, dw, 32):
        for k0 in range(0, npairs, 8):
            for k in range(8):
                y0 = 2 * min(k0 + k, npairs - 1)
                for row in (0, 1):
                    y = y0 + row
                    ins = outs = 0
                    for lane in range(32):
                        xx = min(x0 + lane, dw - 1)
                        fy = min(max(y, 0), dh - 1)
                        qx, qy = xx + 0.5 + m[fy, xx, 0], y + 0.5 + m[fy, xx, 1]
                        if not (0.0 <= qx <= dw and 0.0 <= qy <= dh):
                            continue
                        ix, iy = int(np.floor(qx - 0.5)), int(np.floor(qy - 0.5))
                        fits = ix - 1 >= 0 and ix + 2 <= dw - 1 and iy - 1 >= 0 and iy + 2 <= dh - 1
                        ins, outs = ins + fits, outs + (not fits)
                    out.append((ins, outs))
    return out


def counter_expected(case):
    """True: the fast path must run (counter > 0: some wave's sampling lanes all fit); False: it must not (0); None:
    either (random motions and jitter, at least 4 x 4)."""
    dw, dh = case["dw"], case["dh"]
    if dw < 4 or dh < 4:
        return False
    if case["kind"] == "random":
        return None
    return any(i > 0 and o == 0 for i, o in wave_rows(case["x"], dw, dh))


def cases(kernel, rng):
    """The post cases of l_d3d12_ops_check (same argument layout, sentinels, slack) on the edge inputs."""
    from clean.runner import post as RP
    from clean.l.runner import l_post_check as LPC
    out = []
    for (dw, dh) in SIZES:
        for kind in KINDS:
            x = edge_case(rng, dw, dh, kind)
            prm = RP.post_params(x)
            data = RP.post_device_inputs(x)
            n_hist, n_red = dw * dh * 8, max(4, (dw // 2) * (dh // 2) * 4)
            n_disp, pitch = RP.display_bytes(dw, dh), RP.display_pitch(dw)
            disp_def = np.zeros((3, dh, pitch, 2), bool)
            disp_def[:, :, :dw, :] = True
            red_def = np.zeros(n_red, bool)
            red_def[:(dw // 2) * (dh // 2) * 4] = True
            defined = [np.concatenate([np.ones(n_hist, bool), np.zeros(OPS.POST_SLACK, bool)]),
                       np.concatenate([red_def, np.zeros(OPS.POST_SLACK, bool)]),
                       np.concatenate([disp_def.reshape(-1), np.zeros(OPS.POST_SLACK, bool)])]
            if kernel.startswith("l_post_recon_p12"):
                data = dict(data, **OPS._pack12(data))
            args = [("null",) if data[k] == b"" else ("buf", data[k])
                    for k in ("color", "A", "B", "C", "hist_prev", "m_near", "m_center", "m_far")]
            args += [("buf", RP.pack({"s": np.array(LPC.S)})), ("out", n_hist + OPS.POST_SLACK),
                     ("out", n_red + OPS.POST_SLACK), ("out", n_disp + OPS.POST_SLACK), ("bytes", bytes(prm)), ("null",)]
            out.append({"kernel": kernel, "co": "l_post.co", "grid": RP.recon_grid(dw, dh), "block": 256, "args": args,
                        "defined": defined, "x": x, "dw": dw, "dh": dh, "kind": kind,
                        "label": f"{dw}x{dh} {kind}"})
    return out


def prepare(build, rocm):
    C32.compile_kernels(build, rocm)
    names = KERNELS + ["l_fold_probe", "native_fill"]
    prepare_replay(build, rocm, names)
    from clean.runner.hip import hip_child_environment
    env, _ = hip_child_environment(rocm_dir=rocm)
    env = dict(env, WG_DX12_ISA_CHECK="0")   # test variants: the ISA rules are the shipped ports' (checked above)
    for tag, defs in VARIANTS.items():
        cmd = [sys.executable, os.path.join(ROOT, "clean", "dx12", "build_dx12.py"), "--out", os.path.join(build, tag),
               "--only", ",".join(names), "-j", "1"]
        for d in defs:
            cmd += ["--define", d]
        subprocess.check_call(cmd, env=env, cwd=ROOT)


def edge_bands(dw, dh):
    """Pixel masks [dh][dw] of the two columns / rows at each edge."""
    ys, xs = np.mgrid[0:dh, 0:dw]
    return {"left": xs < 2, "right": xs >= dw - 2, "top": ys < 2, "bottom": ys >= dh - 2}


def replay(build_dir, exe, cdir, lines, env_extra=None):
    with open(os.path.join(cdir, "cases.txt"), "w", newline="\n") as f:
        f.write("\n".join(lines))
    env = {k: v for k, v in os.environ.items()
           if not any(s in k.upper() for s in ("HIP", "LLVM", "ROCM", "HSA")) or k.upper() in ("HIP_PLATFORM",)}
    env["WG_D3D12_ARENA_MB"] = "512"
    env.pop("WG_TEST_FOLD_PROBE_FAIL", None)
    env.update(env_extra or {})
    r = subprocess.run([exe, build_dir, os.path.join(cdir, "cases.txt")], env=env, capture_output=True, text=True,
                       timeout=150, creationflags=0x08000000)
    note = [ln for ln in r.stdout.splitlines() if ln.startswith("device note:")]
    return {"exit": r.returncode, "note": note[0] if note else "", "tail": r.stdout[-400:] + r.stderr[-200:]}


def compare_case(cdir, name, case, wants, counter=False):
    """Raw comparison of one case's three outputs (l_d3d12_ops_check rules); counter: the hscount build, whose first
    dword after the full history holds the counter (excluded from the slack rule, returned)."""
    dw, dh = case["dw"], case["dh"]
    n_hist = dw * dh * 8
    res = {"case": name, "label": case["label"], "outputs": []}
    count = None
    for oi, want in enumerate(wants):
        path = os.path.join(cdir, f"{name}.o{oi}")
        got = open(path, "rb").read() if os.path.exists(path) else b""
        dmask = case["defined"][oi].copy()
        if counter and oi == 0 and len(got) == len(want):
            count = int(np.frombuffer(got[n_hist:n_hist + 4], np.uint32)[0]) - 0x5A5A5A5A
            g, w = bytearray(got), bytearray(want)
            g[n_hist:n_hist + 4] = bytes([D3D12_SENTINEL]) * 4
            w[n_hist:n_hist + 4] = bytes([HIP_SENTINEL]) * 4
            got, want = bytes(g), bytes(w)
        a, b = np.frombuffer(got, np.uint8), np.frombuffer(want, np.uint8)
        o = {"bytes": len(want)}
        if a.size != b.size:
            o.update(identical=False, differ=-1)
        else:
            o["differ"] = int(np.sum((a != b) & dmask))
            o["unwritten_bad"] = int(np.sum(~dmask & ((b != HIP_SENTINEL) | (a != D3D12_SENTINEL))))
            o["identical"] = o["differ"] == 0 and o["unwritten_bad"] == 0
            if oi == 0:   # the full history per edge band
                d = (a[:n_hist] != b[:n_hist]).reshape(dh, dw, 8).any(-1)
                o["edges"] = {k: {"pixels": int(m.sum()), "differ": int((d & m).sum())}
                              for k, m in edge_bands(dw, dh).items()}
        res["outputs"].append(o)
    res["identical"] = all(o["identical"] for o in res["outputs"])
    if counter:
        res["counter"] = count
    return res


def run(build, rocm, part="all"):
    """part: all, replay (D3D12 legs only) or oracle (the oracle leg only): two GPU slots when one is too short."""
    from clean.l.runner import l_post_check as LPC
    g = C32.Gpu(build, rocm)
    with open(os.path.join(build, "l_post.co"), "rb") as f:
        mod = g.hip.load_module(f.read())
    exe = os.path.join(build, "kernel_replay.exe")
    rng = np.random.default_rng(SEED)
    allc, hip_out, lines = {}, {}, []
    neg = None
    for kernel in KERNELS:
        for ci, case in enumerate(cases(kernel, rng)):
            name = f"{kernel}_{ci}"
            allc[name] = case
            if part != "oracle":
                hip_out[name] = OPS.hip_run(g, mod, case)
    res = {"sizes": SIZES, "kinds": KINDS, "part": part, "legs": {}}
    cdir_base = os.path.join(build, "replay-edge")
    for leg, bdir, env_extra, counter in () if part == "oracle" else (("d3d12", build, None, False),
                                          ("d3d12_nofold", build, {"WG_TEST_FOLD_PROBE_FAIL": "1"}, False),
                                          ("hscount", os.path.join(build, "hscount"), None, True),
                                          ("mut8", os.path.join(build, "mut8"), None, True),
                                          ("mut9", os.path.join(build, "mut9"), None, False)):
        cdir = os.path.join(cdir_base, leg)
        os.makedirs(cdir, exist_ok=True)
        names = [n for n in allc if leg != "d3d12_nofold" or allc[n]["kernel"] == "l_post_recon_p12"]
        lines = []
        for n in names:
            lines += OPS.replay_lines(n, allc[n], cdir)
            if leg == "d3d12" and neg is None and allc[n]["grid"] >= 4:
                neg = n
                lines += OPS.replay_lines(n, allc[n], cdir, grid=allc[n]["grid"] // 2, prefix="negative_")
        r = replay(bdir, exe, cdir, lines, env_extra)
        out = [compare_case(cdir, n, allc[n], hip_out[n], counter) for n in names]
        leg_res = {"replay_exit": r["exit"], "device_note": r["note"], "replay_tail": r["tail"], "cases": len(out),
                   "identical": sum(c["identical"] for c in out),
                   "edges": {k: {"pixels": sum(c["outputs"][0].get("edges", {}).get(k, {}).get("pixels", 0) for c in out),
                                 "differ": sum(c["outputs"][0].get("edges", {}).get(k, {}).get("differ", 0) for c in out)}
                             for k in ("left", "right", "top", "bottom")},
                   "differing": [c["label"] + " " + c["case"] for c in out if not c["identical"]][:40]}
        if counter:
            bad, entered = [], 0
            for c in out:
                case = allc[c["case"]]
                want = counter_expected(case)
                entered += c["counter"] > 0 if c["counter"] is not None else 0
                if c["counter"] is None or (want is True and c["counter"] <= 0) or (want is False and c["counter"] != 0):
                    bad.append(f"{c['case']} {c['label']}: counter {c['counter']}")
            leg_res["counter"] = {"cases_entering_fast_path": entered, "expectation_failures": len(bad),
                                  "first_failures": bad[:20],
                                  "must_enter": sum(counter_expected(allc[c["case"]]) is True for c in out),
                                  "must_not_enter": sum(counter_expected(allc[c["case"]]) is False for c in out)}
        if leg == "d3d12" and neg:
            wants = hip_out[neg]
            got = [open(os.path.join(cdir, f"negative_{neg}.o{oi}"), "rb").read()
                   if os.path.exists(os.path.join(cdir, f"negative_{neg}.o{oi}")) else b"" for oi in range(len(wants))]
            leg_res["negative_control"] = {"case": neg, "half_grid": True, "caught": got != list(wants)}
        res["legs"][leg] = leg_res
    # the oracle: HIP's l_post_recon / l_post_recon_ldr against Model L's reconstruction on every case's inputs
    # (cases under 100 pixels: Model L's post acceptance over all of them together, where one value is not 1 %)
    orc, small = [], {"history": [], "display": [], "reduced_max": 0}
    for name, case in allc.items() if part != "replay" else ():
        if case["kernel"] not in ("l_post_recon", "l_post_recon_ldr"):
            continue
        m = LPC.check_case(g.hip, mod, case["x"], "random", ldr=case["kernel"].endswith("_ldr"))
        tiny = case["dw"] * case["dh"] < 100
        ok = m["pass"]
        if tiny:
            for k in ("history", "display"):
                small[k].append(m[k])
            small["reduced_max"] = max(small["reduced_max"], m["reduced"]["max"])
            ok = m["slack_untouched"] and m["alpha_zero"] and m["reduced_top_zero"]
        orc.append({"case": name, "label": case["label"], "kernel": case["kernel"], "pass": bool(ok), "tiny": tiny})
    pooled = {k: {f: sum(i[f] for i in small[k]) for f in ("values", "nonfinite", "over_1", "differ")}
              for k in ("history", "display")}
    for k, v in pooled.items():
        v["pass"] = LPC.l_post_rule(v["values"], v["nonfinite"], v["over_1"], v["differ"],
                                    max((i["mean_rel_rms"] for i in small[k]), default=0.0))
    pooled["reduced_max"] = small["reduced_max"]
    res["oracle"] = {"cases": len(orc), "passed": sum(o["pass"] for o in orc), "tiny_pooled": pooled,
                     "failing": [o["kernel"] + " " + o["label"] for o in orc if not o["pass"]][:40]}
    L = res["legs"]
    checks = {}
    if part != "oracle":
        checks.update({
            "d3d12_identical": L["d3d12"]["identical"] == L["d3d12"]["cases"] and L["d3d12"]["replay_exit"] == 0,
            "fold_port_selected": "passed: the fold ports run" in L["d3d12"]["device_note"],
            "nofold_identical": L["d3d12_nofold"]["identical"] == L["d3d12_nofold"]["cases"],
            "nofold_port_selected": "the no-fold ports run" in L["d3d12_nofold"]["device_note"],
            "hscount_identical": L["hscount"]["identical"] == L["hscount"]["cases"],
            "hscount_counter_ok": L["hscount"]["counter"]["expectation_failures"] == 0,
            "mut8_caught": L["mut8"]["counter"]["expectation_failures"] > 0,
            "mut9_caught": L["mut9"]["identical"] < L["mut9"]["cases"],
            "negative_control_caught": bool(L["d3d12"].get("negative_control", {}).get("caught"))})
    if part != "replay":
        checks["oracle_pass"] = (res["oracle"]["passed"] == res["oracle"]["cases"] and pooled["history"]["pass"] and
                                 pooled["display"]["pass"] and pooled["reduced_max"] <= 1)
    res["checks"] = checks
    res["pass"] = all(checks.values())
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--rocm", required=True)
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--part", choices=("all", "replay", "oracle"), default="all")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    if a.prepare:
        prepare(a.build, a.rocm)
    if a.run:
        res = run(a.build, a.rocm, a.part)
        print(json.dumps(res["checks"], indent=1), "pass", res["pass"])
        if a.json:
            with open(a.json, "w") as f:
                json.dump(res, f, indent=1)
        return 0 if res["pass"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
