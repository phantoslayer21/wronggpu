# SPDX-License-Identifier: MIT
"""Generator of the compact exact q8(GELU) lookup v2 (clean/dx12/hlsl/wg_gelu_lds.hlsli, a 2 KB groupshared table).

    python clean/dx12/gelu_lut/gen2.py --rle <oracle rle.bin> --out <dir> [--hlsl <header>]

Input: HIP's function h -> byte (q8(GELU(h)) of the MLP hidden path, run-length form from gelu_lut/oracle.hip, see gen.py).
Lookup v2 (per hidden value h; all FP32 operations single-rounded):
    w = fma(fma(h, h, A), h, B)            index transform: a monotone cubic h^3 + A h + B (two dual-issued v_fmaak)
    c = fp8(w)                             v_cvt_pk_fp8_f32 (RNE, no saturation): the driver channel's fragment
                                           conversion, 4 instructions per 8 values (code i in byte i & 3 of dword i >> 2)
    (a, b) = T[8 c]                        offset 8 c by one v_dot4_u32_u8 (the code dword . 8 in byte i & 3), one
                                           ds_load_b64 from the 2 KB groupshared copy
    byte = fp8(fma(h, a, b))               v_fma + v_cvt_pk_fp8_f32 (wg_pack_q8_nc: no clamp)
Per 16 values that is 16 dual-issued v_fmaak + 8 + 8 conversions + 16 v_dot4 + 16 address ANDs + 16 v_fmac (80 VALU,
no transcendentals) and 16 LDS loads, against 48 VALU + 32 transcendentals + 8 conversions.
The cubic spreads the buckets where GELU's byte changes fastest (the negative tail, -3.4 < h < -2): with the FP8 cells
of w as buckets, no bucket holds more than two of HIP's byte changes, and (a, b) reproduces up to two changes exactly
(both thresholds to the input ulp). A and B are chosen so that HIP's two one-input flips (u = 0xbf6c7530: a3 between
a2's, 0xbff0c85e: 97 between 96's) each straddle a bucket boundary. Buckets beyond the active range use (1, 0) (h >= 4.75:
the byte is fp8(h)) or (0, -0) (the byte is +-0 by the sign of h; NaN / -inf give NaN = 0xff).
Measured (warm interleaved A/B on the RX 9070 XT against the transcendental chain): swin_block_c96 -12 us, c96_add_ex
-10.6 us; with smaller MLP staging units to make room (C = 64: 2 x 8 KB, bneck: 2-tile units) swin_block_c64 -31 us,
dec1_dec0 -6 us, bneck -17 / -20 us. The fit itself is exact; the 2^32 proof is gelu_lut/run2.py.
Outputs: <out>/gelu_lds.json (A, B, statistics), <out>/gelu_lds_table.bin (256 entries (a, b) float32 at their LDS
offsets 8 c: 2048 bytes), and with --hlsl the header.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen  # noqa: E402  (fp8_hw, fma32, read_rle, byte_of)

f32, u32, fma32, fp8_hw = gen.f32, gen.u32, gen.fma32, gen.fp8_hw
J1 = (0xbf6c752f, 0xbf6c7530)   # these input pairs must fall into different buckets
J2 = (0xbff0c85d, 0xbff0c85e)
KMIN, KMAX = -0x7f800001, 0x7f800000   # ordered keys of -inf .. +inf


def key_to_u(k):
    k = np.asarray(k, np.int64)
    return np.where(k >= 0, k, (-k - 1) + 0x80000000).astype(np.uint32)


def u_to_key(u):
    u = np.asarray(u, np.int64)
    return np.where(u < 0x80000000, u, -(u - 0x80000000) - 1)


def rank(c):
    """Order of FP8 codes by value (0x80 = -0 just below 0x00 = +0; 0xff = NaN / negative overflow lowest)."""
    c = np.asarray(c, np.int64)
    return np.where(c < 0x80, c, -1 - (c - 0x80))


def index_code(h, A, B):
    return fp8_hw(fma32(fma32(h, h, np.float32(A)), h, np.float32(B)))


def offset_of(c):
    return 8 * np.asarray(c, np.int64)


def fp8_values():
    v = np.array([gen.fp8_value(c) for c in range(0x7f)], np.float64)   # 0 .. 448
    return v


def boundaries():
    v = fp8_values()
    m = 0.5 * (v[:-1] + v[1:])
    return np.concatenate([-m[::-1], m])


def separated(A, B):
    """Both one-input flips on bucket boundaries (vectorized over arrays A, B)."""
    ok = np.ones(np.shape(A), bool)
    for pair in (J1, J2):
        h0 = np.full(np.shape(A), f32(np.uint32(pair[0])), np.float32)
        h1 = np.full(np.shape(A), f32(np.uint32(pair[1])), np.float32)
        ok &= index_code(h0, A, B) != index_code(h1, A, B)
    return ok


def solve_params(arange=(0.3, 1.2), brange=0.05):
    """(A, B) float32 pairs that separate both flips: for each pair of FP8 cell boundaries (M1, M2) near w(J1), w(J2)
    the real solution of h1^3 + A h1 + B = M1, h2^3 + A h2 + B = M2, then the float32 neighbourhood checked exactly."""
    h1 = 0.5 * (float(f32(np.uint32(J1[0]))) + float(f32(np.uint32(J1[1]))))
    h2 = 0.5 * (float(f32(np.uint32(J2[0]))) + float(f32(np.uint32(J2[1]))))
    M = boundaries()
    out = []
    for M1 in M[(M > -2.5) & (M < -0.5)]:
        for M2 in M[(M > -12) & (M < -5)]:
            A = ((M1 - h1 ** 3) - (M2 - h2 ** 3)) / (h1 - h2)
            B = M1 - h1 ** 3 - A * h1
            if not (arange[0] <= A <= arange[1] and abs(B) <= brange):
                continue
            Ab = int(u32(np.float32(A)))
            As = f32(np.arange(Ab - 48, Ab + 49, dtype=np.int64).astype(np.uint32))
            for a in As:
                b0 = np.float32(M1 - h1 ** 3 - float(a) * h1)
                bb = int(u32(b0))
                Bs = f32((np.arange(bb - 600, bb + 601, dtype=np.int64)).astype(np.uint32))
                Bs = Bs[np.isfinite(Bs)]
                ok = separated(np.full(len(Bs), a, np.float32), Bs)
                for b in Bs[ok]:
                    out.append((float(a), float(b), float(M1), float(M2)))
    return out


def bucket_keys(A, B):
    """First key of every code's bucket (by value rank) and the bucket key intervals {code: (klo, khi)}."""
    codes = np.arange(256)
    r = rank(codes)
    lo = np.full(256, KMIN, np.int64)
    hi = np.full(256, KMAX + 1, np.int64)   # first key with rank(idx) >= r(c), in [lo, hi]
    for _ in range(34):
        mid = (lo + hi) // 2
        rm = rank(index_code(f32(key_to_u(np.minimum(mid, KMAX))), np.full(256, A, np.float32), np.full(256, B, np.float32)))
        ge = rm >= r
        hi = np.where(ge, mid, hi)
        lo = np.where(ge, lo, mid + 1)
    first = hi
    order = np.argsort(r)
    iv = {}
    for j, c in enumerate(order):
        nxt = first[order[j + 1]] if j + 1 < len(order) else KMAX + 1
        if first[c] < nxt:
            iv[int(c)] = (int(first[c]), int(nxt) - 1)
    return iv


class Hip:
    def __init__(self, pos, byt):
        self.pos, self.byt = pos, byt
        # change boundaries in key order: a change "at key k" means byte(k) != byte(k - 1)
        ks = set()
        for u in pos.tolist():
            if u in (0, 0x80000000) or (u & 0x7fffffff) > 0x7f800000:
                continue
            k = int(u_to_key(u))
            ks.add(k if u < 0x80000000 else k + 1)
        ks.add(0)   # -0 | +0
        self.ck = np.array(sorted(ks), np.int64)

    def byte_k(self, k):
        return gen.byte_of(self.pos, self.byt, key_to_u(k)).astype(np.int64)

    def segments(self, klo, khi):
        """[(klo, khi, byte)] of HIP's byte over keys [klo, khi]."""
        cut = self.ck[(self.ck > klo) & (self.ck <= khi)]
        starts = np.concatenate([[klo], cut])
        ends = np.concatenate([cut - 1, [khi]])
        b = self.byte_k(starts)
        segs = []
        for s, e, x in zip(starts.tolist(), ends.tolist(), b.tolist()):
            if segs and segs[-1][2] == x:
                segs[-1] = (segs[-1][0], e, x)
            else:
                segs.append((s, e, x))
        return segs


def check_entry(segs, a, b):
    """fp8(fma(h, a, b)) == HIP's byte at both ends of every segment (vectorized over arrays a, b): enough, since the
    lookup is monotone in h within a bucket."""
    a = np.atleast_1d(np.asarray(a, np.float32))
    b = np.atleast_1d(np.asarray(b, np.float32))
    ok = np.ones(np.broadcast(a, b).shape, bool)
    for s, e, x in segs:
        for k in {s, e}:
            h = np.full(ok.shape, f32(key_to_u(k)), np.float32)
            ok &= fp8_hw(fma32(h, a, b)) == x
    return ok


def fit(segs):
    """(a, b) for a bucket's segments, or None."""
    cands = [(1.0, 0.0), (0.0, -0.0)]
    vals = [x for _, _, x in segs]
    if len(set(vals)) == 1 and vals[0] not in (0x00, 0x80, 0xff):
        cands.append((0.0, gen.fp8_value(vals[0])))
    for a, b in cands:
        if check_entry(segs, a, b)[0]:
            return a, b
    has_zero = any(s <= -1 and e >= 0 for s, e, _ in segs) or any(
        segs[i][1] == -1 and segs[i + 1][0] == 0 for i in range(len(segs) - 1))
    # crossings: (h before, h after, value midpoint)
    cr = []
    for (s0, e0, x0), (s1, e1, x1) in zip(segs[:-1], segs[1:]):
        if {x0, x1} == {0x00, 0x80}:
            continue   # the sign split at h = 0: b = -0 gives it
        hp, ht = float(f32(key_to_u(e0))), float(f32(key_to_u(s1)))
        m = 0.5 * (gen.fp8_value(x0) + gen.fp8_value(x1))
        cr.append((hp, ht, m))
    if not cr:
        return None
    if has_zero:
        # b = -0: y = a h; only a free (one crossing at most)
        hp, ht, m = cr[0]
        a0 = m / (0.5 * (hp + ht))
        ab = int(u32(np.float32(a0)))
        As = f32(np.arange(ab - 4096, ab + 4097, dtype=np.int64).astype(np.uint32))
        ok = check_entry(segs, As, np.full(len(As), -0.0, np.float32))
        if ok.any():
            i = np.nonzero(ok)[0]
            return float(As[i[len(i) // 2]]), -0.0
        return None
    if len(cr) == 1:
        hp, ht, m = cr[0]
        x = 0.5 * (hp + ht)
        # slopes for which a h is near m at the crossing (b small: fine placement), then a wide b scan
        for k in (2.0, 3.0, 4.0, 1.5, 6.0, 8.0, 1.25, 12.0, 16.0, 1.0, 0.75, 0.5, 24.0, 32.0):
            sg = 1.0 if (gen.fp8_value(segs[-1][2]) - gen.fp8_value(segs[0][2])) * (segs[-1][0] - segs[0][0]) >= 0 else -1.0
            a = np.float32(sg * k * abs(m) / max(abs(x), 1e-30))
            b0 = np.float32(m - float(a) * x)
            bs = f32(((int(u32(b0)) + np.arange(-512, 513)) & 0xffffffff).astype(np.uint32))
            bs = bs[np.isfinite(bs)]
            ok = check_entry(segs, np.full(len(bs), a, np.float32), bs)
            if ok.any():
                i = np.nonzero(ok)[0]
                return float(a), float(bs[i[len(i) // 2]])
        return None
    if len(cr) == 2:
        (hp1, ht1, m1), (hp2, ht2, m2) = cr
        x1, x2 = 0.5 * (hp1 + ht1), 0.5 * (hp2 + ht2)
        a0 = (m2 - m1) / (x2 - x1)
        ab = int(u32(np.float32(a0)))
        for span in (64, 512, 4096):
            As = f32(np.arange(ab - span, ab + span + 1, dtype=np.int64).astype(np.uint32))
            for a in As:
                b0 = np.float32(m1 - float(a) * x1)
                bs = f32(((int(u32(b0)) + np.arange(-64, 65)) & 0xffffffff).astype(np.uint32))
                ok = check_entry(segs, np.full(len(bs), a, np.float32), bs)
                if ok.any():
                    i = np.nonzero(ok)[0]
                    return float(a), float(bs[i[len(i) // 2]])
        return None
    return None


def build(hip, A, B, verbose=False):
    iv = bucket_keys(A, B)
    table = np.zeros((256, 2), np.float32)
    table[:, 1] = np.float32(-0.0)
    stats = {"buckets": len(iv), "changes": {}, "kinds": {}}
    fails = []
    for c, (klo, khi) in sorted(iv.items()):
        segs = hip.segments(klo, khi)
        e = fit(segs)
        n = len(segs) - 1
        if e is None:
            fails.append((c, n, segs[:4]))
            continue
        table[c] = e
        kind = "pass" if e == (1.0, 0.0) else ("const" if e[0] == 0.0 else f"affine{min(n, 3)}")
        stats["kinds"][kind] = stats["kinds"].get(kind, 0) + 1
    return table, iv, stats, fails


def lookup(h, A, B, table):
    c = index_code(h, A, B)
    return fp8_hw(fma32(h, table[c, 0], table[c, 1]))


def model_check(hip, A, B, table, iv):
    """The lookup against HIP at every change and bucket boundary +-3 keys (FP32 denormal inputs skipped: D3D12
    flushes them). The exhaustive GPU run (run2.py) is the proof."""
    ks = [hip.ck]
    for klo, khi in iv.values():
        ks += [np.array([klo, khi], np.int64)]
    ks = np.unique(np.concatenate([k + d for k in ks for d in range(-3, 4)]))
    ks = ks[(ks >= KMIN) & (ks <= KMAX)]
    u = key_to_u(ks)
    u = u[((u & 0x7f800000) != 0) | ((u & 0x7fffff) == 0)]
    h = f32(u)
    got = lookup(h, A, B, table)
    want = gen.byte_of(hip.pos, hip.byt, u)
    bad = np.nonzero(got != want)[0]
    return len(u), [(hex(int(u[i])), hex(int(got[i])), hex(int(want[i]))) for i in bad[:10]], len(bad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hlsl")
    ap.add_argument("--max-tries", type=int, default=400)
    a = ap.parse_args()
    pos, byt = gen.read_rle(a.rle)
    assert len(pos) == 205, len(pos)
    hip = Hip(pos, byt)
    sols = solve_params()
    print(f"{len(sols)} (A, B) candidates separate both flips", flush=True)
    # prefer A near 0.5 (fewest two-change buckets), B small; one try per (M1, M2) family first
    sols.sort(key=lambda s: (abs(s[0] - 0.55), abs(s[1])))
    seen, chosen = set(), None
    tries = 0
    for A, B, M1, M2 in sols:
        fam = (M1, M2, round(A, 3))
        if fam in seen:
            continue
        seen.add(fam)
        tries += 1
        if tries > a.max_tries:
            break
        table, iv, stats, fails = build(hip, A, B)
        print(f"A={A!r} B={B!r} fails={len(fails)} {stats['kinds']}", flush=True)
        if not fails:
            n, bad, nbad = model_check(hip, A, B, table, iv)
            print(f"  model check {n} points, {nbad} bad {bad}", flush=True)
            if nbad == 0:
                chosen = (A, B, table, iv, stats, n)
                break
        else:
            for f in fails[:3]:
                print("   fail", hex(f[0]), f[1], [(hex(s), hex(e), hex(x)) for s, e, x in f[2]])
    assert chosen, "no (A, B)"
    A, B, table, iv, stats, npts = chosen
    os.makedirs(a.out, exist_ok=True)
    lds = np.zeros((256, 2), np.uint32)
    lds[offset_of(np.arange(256)) // 8] = table.view(np.uint32)
    lds.astype("<u4").tofile(os.path.join(a.out, "gelu_lds_table.bin"))
    info = {"A": A, "B": B, "A_bits": f"0x{int(u32(np.float32(A))):08x}", "B_bits": f"0x{int(u32(np.float32(B))):08x}",
            "entries": 256, "bytes": 2048, "kinds": stats["kinds"], "buckets": stats["buckets"],
            "model_check_points": npts}
    with open(os.path.join(a.out, "gelu_lds.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps(info))
    if a.hlsl:
        write_hlsl(a.hlsl, A, B, lds)


def write_hlsl(path, A, B, lds):
    rows = []
    for c in range(256):   # one entry (a, b) per line, at its LDS offset 8 c
        x, y = (int(v) for v in lds[offset_of(c) // 8])
        rows.append(f"    0x{x:08x}u, 0x{y:08x}u,   // {c:3d}  a = {float(np.uint32(x).view(np.float32))!r}, "
                    f"b = {float(np.uint32(y).view(np.float32))!r}")
    text = HLSL.replace("@A@", f"0x{int(u32(np.float32(A))):08x}u").replace("@B@", f"0x{int(u32(np.float32(B))):08x}u")
    text = text.replace("@Af@", repr(float(np.float32(A)))).replace("@Bf@", repr(float(np.float32(B))))
    text = text.replace("@TABLE@", "\n".join(rows))
    with open(path, "w", newline="\n") as f:
        f.write(text)


HLSL = r"""// Exact q8(GELU(h)) of the MLP hidden path by a 2 KB table in groupshared memory (v2). GENERATED by
// clean/dx12/gelu_lut/gen2.py: do not edit.
//
// HIP (swin.co *_nc blocks, dec1_dec0__pack12.co, swin__bmegahg.co bneck6_nc) computes per hidden value h
//   g = h * rcp(1 + exp2(h * fma(h * h, -0.10029031, -2.30816186)))   (wg_platform.h wg_gelu, WG_GELU_MODE 2)
//   byte = v_cvt_pk_fp8_f32(g)                                         (wg_pack_q8_nc: no clamp)
// Here, without transcendentals (every step one FP32 rounding):
//   w = fma(fma(h, h, A), h, B)                  a monotone cubic: spreads the buckets where the byte changes fastest
//   c = fp8(w)                                   v_cvt_pk_fp8_f32 (RNE, no saturation; the 0x32 fragment conversion)
//   (a, b) = T[8 c]                              one ds_load_b64 (offset 8 c: one v_dot4_u32_u8 per value)
//   byte = fp8(fma(h, a, b))                     v_fma_f32 + v_cvt_pk_fp8_f32 (wg_pack_q8_nc_ags)
// No FP8 cell of w holds more than two of HIP's byte changes, and (a, b) places them to the input ulp; A and B put HIP's
// two one-input flips (u = 0xbf6c7530, 0xbff0c85e) on bucket boundaries. Proven on the GPU against HIP for every FP32
// bit pattern (gelu_lut/run2.py): identical bytes for all 2^32 inputs except FP32 denormals (D3D12 flushes them).
//
// The including kernel provides the table: WG_GL_GET(off) = the uint2 at byte offset off (a multiple of 8, < 2048) of
// its 2 KB groupshared copy, filled from wg_gl_table (entry i = the uint2 at byte offset 8 i) before a barrier.
#pragma once

#define WG_GL_A asfloat(@A@)   // @Af@
#define WG_GL_B asfloat(@B@)   // @Bf@
#define WG_GL_BYTES 2048u
#ifndef WG_GL_SCALE
#define WG_GL_SCALE 8u   // offsets in bytes (8 c); 1u: entry indices (c), WG_GL_GET then takes an index
#endif

static const uint wg_gl_table[512] = {
@TABLE@
};
// uint2 at byte offset 8 i of the table (i < 256): the fill of entry i
uint2 wg_gl_entry(uint i) { return uint2(wg_gl_table[2u * i], wg_gl_table[2u * i + 1u]); }

// Byte offsets of the entries of 8 values: the cubics (stage by stage: the driver pairs them into dual-issued v_fmaak),
// one fragment conversion to FP8 (4 v_cvt_pk_fp8_f32: code of value i in byte i & 3 of dword i >> 2), then per value
// 8 c by one v_dot4_u32_u8 of the code dword with 8 in byte i & 3.
void wg_gl_offs8(v8f h, out uint o[8])
{
    v8f w;
    [unroll] for (uint i = 0; i < 8; ++i) w.v[i] = mad(h.v[i], h.v[i], WG_GL_A);
    [unroll] for (uint i1 = 0; i1 < 8; ++i1) w.v[i1] = mad(w.v[i1], h.v[i1], WG_GL_B);
    const uint2 c = wg_pack_q8_nc_ags(w);
    [unroll] for (uint i2 = 0; i2 < 4; ++i2) {
        o[i2] = dot4add_u8packed(c.x, WG_GL_SCALE << (8u * i2), 0u);
        o[4 + i2] = dot4add_u8packed(c.y, WG_GL_SCALE << (8u * i2), 0u);
    }
}

// q8(GELU) of 8 / 16 values: the offsets (two channel conversions per tile), then all loads, then the fmas and the
// output conversions (the driver never moves loads across channel calls; wave-uniform control flow).
uint2 wg_gelu_q8_lds(v8f h)
{
    uint o[8];
    wg_gl_offs8(h, o);
    uint2 e[8];
    [unroll] for (uint i1 = 0; i1 < 8; ++i1) e[i1] = WG_GL_GET(o[i1]);
    v8f y;
    [unroll] for (uint i2 = 0; i2 < 8; ++i2) y.v[i2] = mad(h.v[i2], asfloat(e[i2].x), asfloat(e[i2].y));
    return wg_pack_q8_nc_ags(y);
}
void wg_gelu_q8_lds2(v8f h0, v8f h1, out uint2 q0, out uint2 q1)
{
#if WG_GL_SPLIT
    q0 = wg_gelu_q8_lds(h0);
    q1 = wg_gelu_q8_lds(h1);
    return;
#endif
    uint o0[8], o1[8];
    wg_gl_offs8(h0, o0);
    wg_gl_offs8(h1, o1);
    uint2 e[16];
    [unroll] for (uint i1 = 0; i1 < 8; ++i1) { e[i1] = WG_GL_GET(o0[i1]); e[8 + i1] = WG_GL_GET(o1[i1]); }
    v8f y0, y1;
    [unroll] for (uint i2 = 0; i2 < 8; ++i2) {
        y0.v[i2] = mad(h0.v[i2], asfloat(e[i2].x), asfloat(e[i2].y));
        y1.v[i2] = mad(h1.v[i2], asfloat(e[8 + i2].x), asfloat(e[8 + i2].y));
    }
    q0 = wg_pack_q8_nc_ags(y0);
    q1 = wg_pack_q8_nc_ags(y1);
}
"""


if __name__ == "__main__":
    main()
