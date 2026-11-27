# SPDX-License-Identifier: GPL-3.0-only
"""Byte check of the HYBRID bottleneck in one launch (speed lever 4): swin_bneck6_hyb (clean/dx12/kernels/hyb_bneck.hlsl,
bneck.hlsl's persistent queue around swin_block_c128_hyb's body) against six swin_block_c128_hyb launches on the same
input map and the same six random hybrid blobs, on native D3D12 (kernel_replay chains).

  1. b5's output of the queue kernel == the six launches' b5 output, raw bytes, for every case and every workgroup
     count (1: one workgroup runs every item in queue order; 7, 128: the runtime's count; 300: more than fit at once),
     every output buffer starting from a sentinel fill.
  2. The sync words are all zero after every launch (the last workgroup resets the queue, the finish counter and every
     row counter; word 15 counts bounded-wait timeouts and is never reset: it must stay 0).
  3. Mutant: swin_bneck6_hyb_mut (order mutant 4: the W2 chain end one float32 ulp off before its RN16) must differ.
  4. The default leg (standing rule): kernel_replay with no form variable == WG_L_FORM=hybrid, every saved byte.
  5. Timeout control: swin_bneck6_hyb_tmo (BNECK_SPIN_LIMIT 1: a wait gives up at its first poll of a producer that
     is not done) on the two largest grids, 128 workgroups: the launch completes, the abort word (sync word 12) and the
     timeout count (word 15) are equal and nonzero (a timed-out wait refuses the frame: the runtime's frame guard reads
     word 12 right after the launch), and every other sync word is reset. The twin's launches leave both at 0 (2.).
  The block arithmetic itself is swin_block_c128_hyb's (checked against the oracle by l_hyb_stage_check.py / the S3
  checks); the queue adds no arithmetic, so the six launches are the reference. Inputs: random FP8 maps, special codes
  (NaN included: both sides are ours) in every other case; grids with partial windows, both parities' offsets.

    python clean/l/runner/l_hyb_bneck_check.py --build <dir> [--json out.json] [--time N]
      (<dir>: a build with swin_block_c128_hyb, swin_bneck6_hyb, swin_bneck6_hyb_mut, swin_bneck6_hyb_tmo and
       kernel_replay.exe; GPU)
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

from clean.l.reference import oracle as O  # noqa: E402
from clean.l.runner import l_d3d12_check as LD  # noqa: E402
from clean.l.runner.l_hyb_pack import pack_block_hyb  # noqa: E402
from clean.reference import swin as SW  # noqa: E402
from clean.reference.fp8 import fp8_encode, q8  # noqa: E402

C, HEADS, MLP = 128, 4, 512
GRIDS = [(8, 8), (12, 20), (18, 10), (30, 54), (120, 216)]   # (H8, W8); 120 x 216: the 1440p Quality bottleneck
GROUPS = (1, 7, 128, 300)
SENT = 0xAB


def nwin_of(H, W, ox, oy):
    return ((W + ox + 3) // 4) * ((H + oy + 3) // 4)


def chain_lines(cdir, n, H, W, offs, special, rng, timed, tmo=False):
    """The chains of one case: six launches (ref), the queue kernel at each workgroup count, its mutant; and (tmo) the
    timeout control's chain, returned apart (its bytes depend on the timing: not in the default leg's comparison)."""
    blobs = [pack_block_hyb(O.random_block_params(rng, C, HEADS, MLP), C, HEADS, MLP) for _ in range(6)]
    x = fp8_encode(q8(rng.normal(0, 40.0 if special else 1.5, (H, W, C)))).astype(np.uint8)
    if special:
        pick = rng.random(x.shape) < 0.05
        x[pick] = rng.choice(LD.SPECIAL_FP8, int(pick.sum()))
    for i, b in enumerate(blobs):
        with open(os.path.join(cdir, f"w{n}_{i}.bin"), "wb") as f:
            f.write(b)
    with open(os.path.join(cdir, f"x{n}.bin"), "wb") as f:
        f.write(x.tobytes())
    nb = H * W * C
    NR = max((H + oy + 3) // 4 for _, oy in offs)
    nsync = 4 * (16 + 7 * NR + 16)
    wl = [f"buf w{i} {len(blobs[i])} w{n}_{i}.bin" for i in range(6)]
    ln = [f"chain r{n}", f"buf x {nb} x{n}.bin"] + wl + [f"buf o{i} {nb} - {SENT}" for i in range(6)]
    src = "@x"
    for i in range(6):
        ox, oy = offs[i & 1]
        ln.append(f"step swin_block_c128_hyb {-(-nwin_of(H, W, ox, oy) // 8)} 256 {src} null @o{i} @w{i} "
                  f"{H} {W} {ox} {oy}")
        src = f"@o{i}"
    ln += [f"save o5 r{n}.bin"] + ([f"time {timed}"] if timed else []) + [""]
    (oxe, oye), (oxo, oyo) = offs
    legs = [("swin_bneck6_hyb", "m", g) for g in GROUPS] + [("swin_bneck6_hyb_mut", "u", 128)]
    if tmo:
        legs.append(("swin_bneck6_hyb_tmo", "t", 128))
    tl = []
    for kern, tag, groups in legs:
        nm = f"{tag}{n}_{groups}"
        if tag == "t":
            ln, tl = tl, ln
        ln += [f"chain {nm}", f"buf x {nb} x{n}.bin"] + wl + \
              [f"buf out {nb} - {SENT}", f"buf tmp {nb} - {SENT}", f"buf sync {nsync} - 0",
               f"step {kern} {groups} 256 @x @out @tmp @sync @w0 @w1 @w2 @w3 @w4 @w5 {H} {W} {oxe} {oye} {oxo} {oyo} "
               f"{groups}", f"save out {nm}.bin", f"save sync {nm}_sync.bin"] + \
              ([f"time {timed}"] if timed and tag == "m" and groups == 128 else []) + [""]
        if tag == "t":
            ln, tl = tl, ln
    return ln, tl


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True)
    ap.add_argument("--json")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--time", type=int, default=0, help="also time the 1440p case: N runs of each chain")
    a = ap.parse_args(argv)
    rng = np.random.default_rng(29)
    cdir = os.path.join(a.build, "hyb_bneck")
    os.makedirs(cdir, exist_ok=True)
    lines, tlines, meta, n = [], [], [], 0
    for seed in range(a.seeds):
        for gi, (H, W) in enumerate(GRIDS):
            for par in (0, 1):
                offs = (SW.window_offset("b0", par), SW.window_offset("b1", par))
                special = (gi + par + seed) % 2 == 1
                timed = a.time if (H, W) == GRIDS[-1] and par == 0 and seed == 0 else 0
                tmo = gi >= len(GRIDS) - 2 and seed == 0
                ln, tl = chain_lines(cdir, n, H, W, offs, special, rng, timed, tmo)
                lines += ln
                tlines += tl
                meta.append({"case": n, "H": H, "W": W, "offsets": [list(o) for o in offs], "special": special,
                             "tmo": tmo})
                n += 1
    cases = os.path.join(cdir, "cases.txt")
    with open(cases, "w") as f:
        f.write("\n".join(lines))
    env = dict(os.environ, WG_L_FORM="hybrid")
    r = subprocess.run([os.path.join(a.build, "kernel_replay.exe"), a.build, cases], env=env, capture_output=True,
                       text=True, timeout=160, creationflags=0x08000000)
    if r.returncode:
        raise SystemExit(f"kernel_replay failed: {r.stdout[-1200:]}{r.stderr[-400:]}")
    tcases = os.path.join(cdir, "cases_tmo.txt")   # the timeout control: its own run (bounded: every wait gives up)
    with open(tcases, "w") as f:
        f.write("\n".join(tlines))
    rt = subprocess.run([os.path.join(a.build, "kernel_replay.exe"), a.build, tcases], env=env, capture_output=True,
                        text=True, timeout=160, creationflags=0x08000000)
    if rt.returncode:
        raise SystemExit(f"kernel_replay (timeout control) failed: {rt.stdout[-1200:]}{rt.stderr[-400:]}")
    times = [ln for ln in r.stdout.splitlines() if ": time " in ln]
    rd = lambda nm: open(os.path.join(cdir, nm), "rb").read()  # noqa: E731
    res = {"cases": [], "groups": list(GROUPS), "times": times}
    for m in meta:
        k = m["case"]
        ref = rd(f"r{k}.bin")
        e = dict(m, ref_sentinel_left=int(np.sum(np.frombuffer(ref, np.uint8) == SENT)))
        e["identical"] = {g: rd(f"m{k}_{g}.bin") == ref for g in GROUPS}
        e["sync_zero"] = {g: not np.any(np.frombuffer(rd(f"m{k}_{g}_sync.bin"), np.uint32)) for g in GROUPS}
        sync15 = [int(np.frombuffer(rd(f"m{k}_{g}_sync.bin"), np.uint32)[15]) for g in GROUPS]
        e["timeouts"] = sum(sync15)
        mu = np.frombuffer(rd(f"u{k}_128.bin"), np.uint8)
        e["mutant_differ_bytes"] = int(np.sum(mu != np.frombuffer(ref, np.uint8)))
        if m["tmo"]:
            s = np.frombuffer(rd(f"t{k}_128_sync.bin"), np.uint32).copy()
            ab, cnt = int(s[12]), int(s[15])
            s[12] = s[15] = 0
            e["timeout_control"] = {"abort": ab, "count": cnt, "others_zero": not np.any(s),
                                    "pass": ab > 0 and ab == cnt and not np.any(s)}
        res["cases"].append(e)
    dl = LD.replay_default_leg(a.build, cases, env, timeout=160)
    res["default_leg"] = dl
    cs = res["cases"]
    res["queue_identical"] = sum(all(c["identical"].values()) for c in cs)
    res["sync_all_zero"] = all(all(c["sync_zero"].values()) for c in cs)
    res["mutant_caught"] = sum(c["mutant_differ_bytes"] > 0 for c in cs)
    tc = [c["timeout_control"] for c in cs if "timeout_control" in c]
    res["timeout_control"] = {"cases": len(tc), "pass": sum(t["pass"] for t in tc),
                              "abort": [t["abort"] for t in tc], "count": [t["count"] for t in tc]}
    res["pass"] = (res["queue_identical"] == len(cs) and res["sync_all_zero"] and res["mutant_caught"] > 0
                   and dl["pass"] and len(tc) > 0 and res["timeout_control"]["pass"] == len(tc))
    print(json.dumps({k: v for k, v in res.items() if k != "cases"}, indent=1))
    for c in cs:
        print(c["case"], c["H"], c["W"], c["offsets"], "special" if c["special"] else "", c["identical"],
              "sync0" if all(c["sync_zero"].values()) else "SYNC!", "mutant", c["mutant_differ_bytes"])
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1, default=str)
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
