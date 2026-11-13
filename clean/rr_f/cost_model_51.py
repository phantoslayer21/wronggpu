# SPDX-License-Identifier: MIT
"""Issue-slot cost model behind DESIGN-51.md section 10 (no GPU, no kernel code).

Counts, per wave-tile (16 tokens of one 8 x 8 window, the work unit of one wave), the v_wmma_f32_16x16x16_fp8_fp8
instructions and the VALU issue slots of the mapping in DESIGN-51.md, then turns them into stage times at a given
render size with a throughput calibrated on the Model M D3D12 c64 Swin block (DESIGN-51.md 10.1).

    python cost_model_51.py [--w 1706 --h 960] [--json]

Every number here is an estimate for planning; the kernels are measured, not modelled, once they exist.
"""
import argparse
import json
import math

# ---- machine / calibration (DESIGN-51.md 10.1) ----------------------------------------------------------------------
CUS = 64
SIMDS = 2 * CUS
CLOCK = 2.9e9            # warm shader clock (README: 2.6-2.9 GHz warm)
EFF = 0.83               # achieved fraction of (8 * wmma + valu) issue cycles, from the M c64 calibration
WMMA_CYC = 8             # fp8 16x16x16 on one SIMD (389 dense fp8 TFLOPS / 128 SIMDs / 2.97 GHz ~ 512 MAC/clk)
VALU_OVH = 1.10          # addresses, loop control, LDS address math, exchange glue
LONG = False             # --long: variant L (DESIGN-51.md 7.4): f32 over the whole K, one f16 rounding at the end

# ---- per-operation VALU costs (lane-instructions; one per issue cycle for wave32) ------------------------------------
def rnd(tiles, k):
    """f16 rounding of a 16x16 f32 accumulator chain over K inputs: K/32 steps; every step but the last
    rounds to f16 and widens back (2 ops per value), the last only rounds (1 op per value); 8 values per lane."""
    s = 1 if LONG else k // 32
    return tiles * (16 * (s - 1) + 8)


def q8(vals):
    """Q8 of f16 values already rounded: widen (1), clamp keeping NaN v_maximumminimum (1), pack two per op (0.5)."""
    return 2.5 * vals


def block(C, H):
    """Common core 51.3 for one wave-tile: returns (wmma, valu)."""
    w = v = 0.0
    for _ in range(H):
        # 1. q and v (no bias), 2 feature tiles each, K = C
        w += 2 * 2 * (C // 16)
        v += rnd(4, C) + q8(4 * 8)
        # 2. scores of the wave's 16 queries against 64 keys, K = 32, start value = bias (f32 from memory)
        w += 4 * 2
        v += 4 * 8                      # f32 -> f16 of the scores
        # 3. attention weight, packed f16: mul, max, min, fma, fma (5 VOP3P) + shl, and (dual-issued: 1) per pair
        v += 16 * 6
        # 4. Z (16 v_dot2_f32_f16 + exchange + add + cvt + rcp + cvt), P = Q8(R(w r)): pk_mul, widen, pack
        v += 16 + 6
        v += 16 + 32 + 16
        # 5. O = Q8(P V): 2 feature tiles, K = 64 (2 steps)
        w += 2 * 4
        v += rnd(2, 64) + q8(2 * 8)
        # 6. Dp chain step of this head: C/16 tiles, K = 32: widen previous f16 + round
        w += (C // 16) * 2
        v += 0 if LONG else (C // 16) * 16
    if LONG:
        v += (C // 16) * 8              # one rounding of Dp at the end
    # MLP, 4C/32 chunks
    for _ in range(4 * C // 32):
        w += 2 * (C // 16)              # W1 chunk, 2 hidden tiles, K = C
        v += rnd(2, C)
        v += 8 * 6 + 8                  # act on 16 values (8 pairs: max, min, and, fma, fma, mul) + opaque fence
        v += q8(16)
        w += (C // 16) * 2              # W2 chunk into every output tile, K = 32
        v += 0 if LONG else (C // 16) * 16
    if LONG:
        v += (C // 16) * 8              # one rounding of Dd at the end
    vals = C // 2                       # values per lane of one token tile x C channels
    v += vals * 1.5                     # xr = d8(x): fp8 -> f32 (pk, 0.5) -> f16 (1)
    v += vals * 0.5 + q8(vals)          # y1h = fma16, y1q
    v += vals * 0.5 + q8(vals)          # y2h = fma16, y2
    return w, v


def hidden(C, outs, decoder):
    """Hidden projection (one padded 16-row tile, K = C), sigmoid LUT, gate / update; decoder adds the bilinear."""
    w = C // 16
    v = rnd(1, C) * 3 / 8 + 12 + (40 if decoder else 10)
    return w, v


def merge(C, Cn):
    """2 x 2 merge of one window: one N tile (16 coarse tokens), Cn/16 output tiles over K = 4C, split over 4 waves."""
    tiles = Cn / 16 / 4
    return tiles * (4 * C // 16), rnd(tiles, 4 * C) + q8(tiles * 8) + C / 8   # + LDS gather glue


def upproj(C, Kc):
    """Fusion front end: 4C/16 tiles over K = Kc for the window's 16 coarse tokens (split over 4 waves), then
    xf = fma16(d8(skip), f, d8(upq)) and x = Q8(xf) for the wave's tile."""
    tiles = 4 * C / 16 / 4
    vals = C // 2
    return tiles * (Kc // 16), rnd(tiles, Kc) + q8(tiles * 8) + vals * (1.5 + 1.5 + 0.5) + q8(vals) - vals * 1.5


def adapter():
    """E1 input adapter 128 -> 64 (4 output tiles, K = 128)."""
    return 4 * 8, rnd(4, 128) + q8(32)


# ---- stages ----------------------------------------------------------------------------------------------------------
LEVEL_C = {0: 32, 1: 64, 2: 96, 3: 128, 4: 160, 5: 160}
STAGES = [
    # name, level, C, H, extras
    ("IN0 (core only)", 0, 32, 2, ["hid_e"]),
    ("E1", 1, 64, 2, ["adapter", "hid_e", ("merge", 96)]),
    ("E2", 2, 96, 4, ["hid_e", ("merge", 128)]),
    ("E3", 3, 128, 4, ["hid_e", ("merge", 160)]),
    ("E4", 4, 160, 8, ["hid_e", ("merge", 160)]),
    ("E5", 5, 160, 8, ["hid_e"]),
    ("D4-F", 4, 160, 8, [("up", 160)]),
    ("D4-R", 4, 160, 8, ["hid_d"]),
    ("D3-F", 3, 128, 4, [("up", 160)]),
    ("D3-R", 3, 128, 4, ["hid_d"]),
    ("D2-F", 2, 96, 4, [("up", 128)]),
    ("D2-R", 2, 96, 4, ["hid_d"]),
    ("D1-F", 1, 64, 2, [("up", 96)]),
    ("D1-R", 1, 64, 2, ["hid_d"]),
    ("D0-F", 0, 32, 2, [("up", 64)]),
    ("D0-R (core only)", 0, 32, 2, []),
]


def windows(n, s=4):
    return math.ceil((n + s) / 8)


def run(w_render, h_render):
    wt, ht = 32 * math.ceil(w_render / 32), 32 * math.ceil(h_render / 32)
    rows = []
    total = 0.0
    for name, lev, C, H, extras in STAGES:
        nx, ny = wt >> lev, ht >> lev
        nwin = windows(nx) * windows(ny)
        wm, va = block(C, H)
        for e in extras:
            if e == "adapter":
                a, b = adapter()
            elif e == "hid_e":
                a, b = hidden(C, 3, False)
            elif e == "hid_d":
                a, b = hidden(C, 4, True)
            elif e[0] == "merge":
                a, b = merge(C, e[1])
            elif e[0] == "up":
                a, b = upproj(C, e[1])
            wm += a
            va += b
        cyc = WMMA_CYC * wm + VALU_OVH * va
        # per CU: ceil(nwin / CUS) workgroups of 4 waves on 2 SIMDs -> 2 tile-times per workgroup per SIMD
        per_simd = math.ceil(nwin / CUS) * 2 * cyc
        tput = nwin * 4 * cyc / SIMDS
        t_ms = max(per_simd, tput) / (CLOCK * EFF) * 1e3
        total += t_ms
        rows.append(dict(stage=name, level=lev, tokens=nx * ny, windows=nwin, C=C, H=H, wmma_per_tile=round(wm),
                         valu_per_tile=round(va), wmma_share=round(WMMA_CYC * wm / cyc, 2), cycles_per_tile=round(cyc),
                         ms=round(t_ms, 3)))
    return dict(render=[w_render, h_render], tokens0=[wt, ht], stages=rows, total_ms=round(total, 2))


def calib_m_c64():
    """Model M c64 block (4 x 4 windows, C 64, H 2, MLP 256), same counting: per 16-token window."""
    w = 2 * (2 * 4 * 2 + 2 + 2 + 4 * 2) + 16 * 4 + 4 * 16
    v = 1700   # GELU (table + chain mix), softmax of 2 x 256 scores, 2 RMSNorms, packs: counted from swin_block.hlsl
    return w, v


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=int, default=1706)
    ap.add_argument("--h", type=int, default=960)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--long", action="store_true", help="variant L: no per-step f16 rounding")
    a = ap.parse_args()
    LONG = a.long
    r = run(a.w, a.h)
    if a.json:
        print(json.dumps(r, indent=1))
    else:
        print(f"render {a.w}x{a.h} -> level-0 tokens {r['tokens0'][0]}x{r['tokens0'][1]}")
        print(f"{'stage':18s} {'lev':>3s} {'tokens':>8s} {'win':>6s} {'wmma':>6s} {'valu':>6s} {'wm%':>4s} {'cyc/tile':>8s} {'ms':>7s}")
        for s in r["stages"]:
            print(f"{s['stage']:18s} {s['level']:3d} {s['tokens']:8d} {s['windows']:6d} {s['wmma_per_tile']:6d} "
                  f"{s['valu_per_tile']:6d} {int(100 * s['wmma_share']):3d}% {s['cycles_per_tile']:8d} {s['ms']:7.3f}")
        print(f"total (16 window blocks + tails, without page 52 / 55 front end and head): {r['total_ms']} ms")
        w, v = calib_m_c64()
        cyc = WMMA_CYC * w + v
        nwin = 853 * 480 // 16
        print(f"calibration: M c64 {w} wmma + ~{v} valu per window -> {cyc} cyc; {nwin} windows -> "
              f"{nwin * cyc / SIMDS / (CLOCK * EFF) * 1e3:.3f} ms (measured 0.265 ms)")
