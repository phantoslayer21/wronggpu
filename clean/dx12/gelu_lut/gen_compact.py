# SPDX-License-Identifier: MIT
"""Generator of the compact exact q8(GELU) lookup (hlsl/wg_gelu_lut.hlsli, design "compact table, global").

    python clean/dx12/gelu_lut/gen_compact.py --rle <oracle rle.bin> --out <dir> [--k 5] [--layout full|clamp]
        [--hlsl <path>] [--cpp <path>]

Input: HIP's run-length form of h -> byte = v_cvt_pk_fp8_f32(wg_gelu(h)) over all 2^32 FP32 inputs (oracle.hip via
run.py; 204 changes, two one-input flips at u = 0xbf6c7530 and 0xbff0c85e).

Lookup per value h (no transcendentals):
    w = fma(h, S, B)                       index transform (one v_fmaak_f32, dual-issued)
    c = f16_rtz(w)                         v_cvt_pk_rtz_f16_f32: two values per instruction, sign in bit 15
    layout full:  i = c >> (10 - K)        all 2^(6 + K) codes: [sign][exponent 5][K mantissa bits]
    layout clamp: c clamped to the range that matters (FP16 min / max of the pair: v_minmax_num_f16), then
                  i = [sign][exponent 3][K mantissa bits] (the clamped codes never use the exponent's top two bits)
    (a, b) = table[i]                      one 8-byte load
    byte = fp8(fma(h, a, b))               v_fmac_f32 + v_cvt_pk_fp8_f32 (no clamp), as v1
Inside a bucket y = fma(h, a, b) is monotone in h, so (a, b) can reproduce a staircase with at most two steps whose
position (a, b) sets to the ulp; S and B put both one-input flips on bucket boundaries, so every bucket is monotone.
Constant buckets use a = 0 (+-0 keeps the sign of h; -inf, NaN give NaN = 0xff); buckets beyond h = 4.750001 use
(1, 0) (byte = fp8(h) there, inf -> 0x7f, NaN -> 0xff); buckets below h = -3.3954375 use (0, -0).
Both index semantics are covered: fma rounded to FP32 before the RTZ conversion, or folded into one RTZ conversion.

Outputs in --out: gelu_lut_table.bin (the table bytes from the table's base), gelu_lut.json; with --hlsl / --cpp the
headers. The exhaustive GPU check (run.py) is the proof; the model check here catches generator slips early.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen  # noqa: E402  (fma32, rtz32_of_fma, fp8_hw, fp8_value, read_rle, byte_of, fit, changes)

J1 = (0xbf6c752f, 0xbf6c7530)
J2 = (0xbff0c85d, 0xbff0c85e)
PASS_FROM = 0x40980002   # h >= 4.750001: byte = fp8(h)
ZERO_FROM = 0xc0594ed9   # h <= -3.3954375 (finite): byte 0x80


def f16_rtz(w):
    """v_cvt_pk_rtz_f16_f32 of float32 values (subnormal results kept, RTZ saturation at 0x7bff, inf, NaN)."""
    w = np.asarray(w, np.float32)
    b = gen.u32(w).astype(np.int64)
    s = (b >> 31) & 1
    e = (b >> 23) & 0xff
    m = b & 0x7fffff
    a = np.abs(w.astype(np.float64))
    sub = np.floor(a * 2.0 ** 24).astype(np.int64)
    nrm = ((e - 112) << 10) | (m >> 13)
    out = np.where(e < 113, sub, nrm)
    out = np.where(e >= 143, 0x7bff, out)
    out = np.where((e == 255) & (m == 0), 0x7c00, out)
    out = np.where((e == 255) & (m != 0), 0x7e00, out)
    return out | (s << 15)


def f16_value(c):
    c = int(c)
    s, e, m = c >> 15, (c >> 10) & 31, c & 1023
    v = m * 2.0 ** -24 if e == 0 else (1 + m / 1024) * 2.0 ** (e - 15)
    return -v if s else v


class Index:
    def __init__(self, S, B, K, layout, hp=None, hn=None):
        self.S, self.B, self.K, self.layout = np.asarray(S, np.float32), np.asarray(B, np.float32), K, layout
        self.SH = 10 - K
        self.hp, self.hn = hp, hn   # clamp layout: largest positive / negative-magnitude code kept

    def w(self, h, sem):
        h = np.asarray(h, np.float32)
        S = np.broadcast_to(self.S, h.shape).astype(np.float32)
        B = np.broadcast_to(self.B, h.shape).astype(np.float32)
        return gen.fma32(h, S, B) if sem == "a" else gen.rtz32_of_fma(h, S, B)

    def clamp(self, c):
        if self.layout != "clamp":
            return c
        neg = c >= 0x8000
        return np.where(neg, np.minimum(c, 0x8000 | self.hn), np.minimum(c, self.hp))

    def slot(self, c):
        """table slot (8-byte units from the table base) of a clamped code"""
        if self.layout == "full":
            return c >> self.SH
        return ((c >> 15) << (5 + self.K)) | ((c & 0x1fff) >> self.SH)   # offset (c >> (SH - 3)) & mask: sign at slot bit 5 + K

    def of(self, h, sem):
        return self.slot(self.clamp(f16_rtz(self.w(h, sem))))

    def nslots(self):
        return 1 << (16 - self.SH) if self.layout == "full" else (1 << (5 + self.K)) + (1 << (3 + self.K))


def inside(pos, ix):
    """{slot: set of changes u inside it} over both semantics (u inside when u - 1 and u share the slot)."""
    t = gen.changes(pos)
    t = t[(t <= PASS_FROM) | (t >= 0x80000000)]   # beyond PASS_FROM byte = fp8(h): slots there are (1, 0)
    out = {}
    for sem in "ab":
        b0 = ix.of(gen.f32(t), sem)
        b1 = ix.of(gen.f32(t - 1), sem)
        for k, u in zip(b0[b0 == b1], t[b0 == b1]):
            out.setdefault(int(k), set()).add(int(u))
    return out


def flips_split(ix):
    for x in (J1, J2):
        h = gen.f32(np.array(x, np.uint32))
        for sem in "ab":
            b = ix.of(h, sem)
            if b[0] == b[1]:
                return False
    return True


def candidates(K, Slo, Shi, layout, hp=None, hn=None):
    """(S, B) in [Slo, Shi) x small B that put a bucket boundary inside J1 (J2 checked exactly afterwards)."""
    S = np.arange(gen.u32(np.float32(Slo)), gen.u32(np.float32(Shi)), dtype=np.uint32).view(np.float32).astype(np.float64)
    ha, hb = gen.f32(np.array(J1, np.uint32)).astype(np.float64)
    pa = -ha * S
    if layout == "clamp":   # boundaries of subnormal / normal f16 grids alike: step of the grid at pa
        e = np.floor(np.log2(pa))
        step = np.where(pa < 2.0 ** -14, 2.0 ** (-24 + 10 - K), 2.0 ** (e - K))
    else:
        step = 2.0 ** (np.floor(np.log2(pa)) - K)
    out = []
    for back in range(0, 3):
        V = np.floor(pa / step) * step - back * step
        lo, hi = pa - V, -hb * S - V
        Bm = hi.astype(np.float32).astype(np.float64)
        Bm = np.where(Bm > hi, np.nextafter(hi.astype(np.float32), np.float32(0)).astype(np.float64), Bm)
        ok = (Bm > lo) & (Bm <= hi)
        out += list(zip(S[ok].astype(np.float32), Bm[ok].astype(np.float32)))
    if not out:
        return []
    Sk = np.array([o[0] for o in out], np.float32)
    Bk = np.array([o[1] for o in out], np.float32)
    keep = np.ones(len(Sk), bool)
    for x in (J1, J2):   # exact, vectorized over the candidates, both semantics
        h = gen.f32(np.array(x, np.uint32))
        for sem in "ab":
            ix = Index(Sk, Bk, K, layout, hp, hn)
            ha_ = np.full(len(Sk), h[0], np.float32)
            hb_ = np.full(len(Sk), h[1], np.float32)
            keep &= ix.of(ha_, sem) != ix.of(hb_, sem)
    return [(float(a), float(b)) for a, b in zip(Sk[keep], Bk[keep])]


def y_of(h, a, b):
    h = np.asarray(h, np.float32)
    return gen.fma32(h, np.full(h.shape, a, np.float32), np.full(h.shape, b, np.float32))


def fit_points(hs, want, a, b):
    return np.array_equal(gen.fp8_hw(y_of(np.array(hs, np.float32), np.float32(a), np.float32(b))), np.array(want))


def fit2(cross, hmin, hmax):
    """(a, b) with fp8(fma(h, a, b)) = c0 on [hmin, p1], c1 on [t1, p2], c2 on [t2, hmax]; cross = [(p, t, cp, ct)]
    in increasing h with p < t adjacent floats. y is monotone, so the six key points decide."""
    (p1, t1, c0, c1), (p2, t2, c1b, c2) = cross
    assert c1 == c1b
    hs = np.array([hmin, p1, t1, p2, t2, hmax], np.float32)
    want = np.array([c0, c0, c1, c1, c2, c2])
    m1 = 0.5 * (gen.fp8_value(c0) + gen.fp8_value(c1))
    m2 = 0.5 * (gen.fp8_value(c1) + gen.fp8_value(c2))
    a0 = np.float32((m2 - m1) / (float(t2) - float(t1)))
    ua = int(gen.u32(a0))
    span = 1 << 16
    A = gen.f32(((ua + np.arange(-span, span + 1)) & 0xffffffff).astype(np.uint32))
    A = A[np.isfinite(A) & (np.sign(A) == np.sign(a0))]
    B0 = (np.float64(m1) - A.astype(np.float64) * float(t1)).astype(np.float32)
    for db in range(-6, 7):
        Bc = gen.f32(((gen.u32(B0).astype(np.int64) + db) & 0xffffffff).astype(np.uint32))
        ok = np.ones(len(A), bool)
        for hv, wv in zip(hs, want):
            y = gen.fma32(np.full(len(A), hv, np.float32), A, Bc)
            ok &= gen.fp8_hw(y) == wv
            if not ok.any():
                break
        if ok.any():
            i = int(np.nonzero(ok)[0][0])
            return float(A[i]), float(Bc[i])
    return None


def fit1(hl, hr, cl, cr, hmin, hmax):
    """gen.fit, vectorized: (a, b) with fp8(fma(h, a, b)) = cl on [hmin, hl] and cr on [hr, hmax]."""
    vl, vr = gen.fp8_value(cl), gen.fp8_value(cr)
    sgn = 1.0 if vr > vl else -1.0
    m = 0.5 * (vl + vr)
    hs = np.array([hmin, hl, hr, hmax], np.float32)
    want = np.array([cl, cl, cr, cr])
    for k in (2.0, 3.0, 4.0, 1.5, 6.0, 8.0, 1.25, 12.0, 16.0, 1.0, 24.0, 32.0, 0.75, 0.5):
        a = np.float32(sgn * k * abs(m) / abs(float(hr)))
        b0 = np.float32(m - float(a) * float(hr))
        bs = gen.f32(((gen.u32(b0).astype(np.int64) + np.arange(-256, 257)) & 0xffffffff).astype(np.uint32))
        bs = bs[np.isfinite(bs)]
        ok = np.ones(len(bs), bool)
        for hv, wv in zip(hs, want):
            ok &= gen.fp8_hw(gen.fma32(np.full(len(bs), hv, np.float32), np.full(len(bs), a, np.float32), bs)) == wv
        if ok.any():
            return float(a), float(bs[np.nonzero(ok)[0][0]])
    return None


def bucket_h_range(ix, slot):
    """h interval [lo, hi] (float64, padded) of a slot and the sign of its codes; None for slots no code reaches."""
    K, SH = ix.K, ix.SH
    if ix.layout == "full":
        c0 = slot << SH
    else:
        sign = (slot >> (K + 5)) & 1
        if (slot >> (K + 3)) & 3:
            return None   # the gap between the two halves
        c0 = (sign << 15) | ((slot & ((1 << (K + 3)) - 1)) << SH)
    c1 = c0 + (1 << SH)
    neg = bool(c0 & 0x8000)
    lim = 0x7c00
    if ix.layout == "clamp":
        top = (0x8000 | ix.hn) if neg else ix.hp
        if c0 > top:
            return None
        if c0 <= top < c1:
            c1 = None   # the clamped slot: everything beyond
    mag0 = c0 & 0x7fff
    if mag0 >= lim:
        return ("special", neg)
    v0 = f16_value(mag0)
    v1 = f16_value(c1 & 0x7fff) if c1 is not None and (c1 & 0x7fff) <= lim else np.inf
    w_lo, w_hi = (-v1, -v0) if neg else (v0, v1)
    S, B = float(ix.S), float(ix.B)
    h_lo, h_hi = (w_lo - B) / S, (w_hi - B) / S
    return (h_lo, h_hi, neg)


def build(pos, byt, ix):
    n = ix.nslots()
    table = np.zeros((n, 2), np.float32)
    table[:, 1] = np.float32(-0.0)
    ins = inside(pos, ix)
    kinds = {"pass": 0, "zero_far": 0, "const": 0, "one": 0, "two": 0, "unused": 0, "special": 0}
    zero = lambda c: (0.0, -0.0) if c in (0x00, 0x80) else (0.0, gen.fp8_value(c))
    used = np.zeros(n, bool)
    for slot in range(n):
        r = bucket_h_range(ix, slot)
        if r is None:
            kinds["unused"] += 1
            continue
        used[slot] = True
        if r[0] == "special":   # +-inf / NaN codes
            table[slot] = (0.0, -0.0) if r[1] else (1.0, 0.0)
            kinds["special"] += 1
            continue
        h_lo, h_hi, neg = r
        if not neg and h_lo >= float(gen.f32(PASS_FROM)):
            table[slot] = (1.0, 0.0)
            kinds["pass"] += 1
            continue
        if neg and h_hi <= float(gen.f32(ZERO_FROM)):
            table[slot] = (0.0, -0.0)
            kinds["zero_far"] += 1
            continue
        hi_fin = h_hi if np.isfinite(h_hi) else 3e38
        lo_fin = h_lo if np.isfinite(h_lo) else -3e38
        pad = max(abs(lo_fin), abs(hi_fin)) * 2.0 ** -20 + 1e-45
        hmin, hmax = np.float32(lo_fin - pad), np.float32(hi_fin + pad)
        ch = sorted(ins.get(slot, ()), key=lambda u: float(gen.f32(u)))
        if not ch:
            if h_lo < 0.0 < h_hi:
                cpos = int(gen.byte_of(pos, byt, [gen.u32(np.float32(h_hi * 0.5))])[0])
                cneg = int(gen.byte_of(pos, byt, [gen.u32(np.float32(h_lo * 0.5))])[0])
                assert cpos == 0x00 and cneg == 0x80, (slot, hex(cpos), hex(cneg))
                table[slot] = (0.0, -0.0)
            else:
                hm = np.float32(0.5 * (lo_fin + hi_fin))
                table[slot] = zero(int(gen.byte_of(pos, byt, [gen.u32(hm)])[0]))
            kinds["const"] += 1
            continue
        cross = []
        for u in ch:
            h_t, h_p = gen.f32(u), gen.f32(u - 1)
            c_t, c_p = (int(x) for x in gen.byte_of(pos, byt, [u, u - 1]))
            cross.append((h_p, h_t, c_p, c_t) if h_t > h_p else (h_t, h_p, c_t, c_p))
        cross.sort(key=lambda x: float(x[0]))
        if len(cross) == 1:
            p, t, cp, ct = cross[0]
            ent = fit1(p, t, cp, ct, hmin, hmax)
            kinds["one"] += 1
        elif len(cross) == 2:
            ent = fit2(cross, hmin, hmax)
            kinds["two"] += 1
        else:
            raise AssertionError(("more than two changes", slot, [hex(u) for u in ch]))
        assert ent is not None, ("no fit", slot, [hex(u) for u in ch])
        table[slot] = ent
    return table, kinds, used


def check_points(pos, byt, ix, table):
    us = np.unique(np.concatenate([pos + d for d in (-2, -1, 0, 1, 2)]) & 0xffffffff)
    us = us[(us & 0x7fffffff) <= 0x7f800000]
    us = us[((us & 0x7f800000) != 0) | ((us & 0x7fffff) == 0)]
    h = gen.f32(us)
    want = gen.byte_of(pos, byt, us)
    bad = []
    for sem in "ab":
        i = ix.of(h, sem)
        got = gen.fp8_hw(gen.fma32(h, table[i, 0], table[i, 1]))
        bad += [(sem, hex(int(us[k])), hex(int(got[k])), hex(int(want[k]))) for k in np.nonzero(got != want)[0]]
    return len(us), bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--layout", default="full", choices=("full", "clamp"))
    ap.add_argument("--hlsl")
    ap.add_argument("--cpp")
    a = ap.parse_args()
    pos, byt = gen.read_rle(a.rle)
    assert pos[0] == 0 and len(pos) == 205, len(pos)
    K = a.k
    if a.layout == "full":
        ranges, hp, hn = ((1.0, 2.0), (0.5, 1.0)), None, None
    else:
        ranges = ((2.0 ** -10, 2.0 ** -9),)
        hp, hn = None, None
    chosen, tried = None, 0
    for Slo, Shi in ranges:
        cl = candidates(K, Slo, Shi, a.layout, 0x1fff, 0x1fff)
        print("S range", Slo, Shi, "flip-aligned candidates", len(cl), flush=True)
        for S, B in sorted(cl, key=lambda sb: abs(sb[1])):
            tried += 1
            if a.layout == "clamp":
                ixc = Index(S, B, K, "clamp", 0, 0)
                # keep every code below the pass / far-zero thresholds; clamp the rest into one slot per sign
                cp = int(f16_rtz(ixc.w(gen.f32(np.array([PASS_FROM], np.uint32)), "a"))[0])
                cn = int(f16_rtz(ixc.w(gen.f32(np.array([ZERO_FROM], np.uint32)), "a"))[0]) & 0x7fff
                hp = ((cp >> ixc.SH) + 1) << ixc.SH
                hn = ((cn >> ixc.SH) + 1) << ixc.SH
                if hp > 0x1fff or hn > 0x1fff:
                    continue
            ix = Index(S, B, K, a.layout, hp, hn)
            if not flips_split(ix):
                continue
            mx = max((len(v) for v in inside(pos, ix).values()), default=0)
            print("candidate", S, B, "max changes per slot", mx, flush=True)
            if mx > 2:
                continue
            try:
                table, kinds, used = build(pos, byt, ix)
            except AssertionError as e:
                print("skip", S, B, str(e)[:120], flush=True)
                continue
            npts, bad = check_points(pos, byt, ix, table)
            if bad:
                print("model check failed", S, B, bad[:4], flush=True)
                continue
            chosen = (ix, table, kinds, npts)
            break
        if chosen:
            break
    assert chosen, f"no (S, B) after {tried} candidates"
    ix, table, kinds, npts = chosen
    os.makedirs(a.out, exist_ok=True)
    table.astype("<f4").tofile(os.path.join(a.out, "gelu_lut_table.bin"))
    stats = {"S": float(ix.S), "B": float(ix.B), "S_bits": f"0x{int(gen.u32(ix.S)):08x}",
             "B_bits": f"0x{int(gen.u32(ix.B)):08x}", "k": K, "layout": a.layout,
             "hp": ix.hp, "hn": ix.hn, "slots": int(ix.nslots()), "table_bytes": int(table.nbytes),
             "kinds": kinds, "model_check_points": npts, "candidates_tried": tried}
    with open(os.path.join(a.out, "gelu_lut.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
