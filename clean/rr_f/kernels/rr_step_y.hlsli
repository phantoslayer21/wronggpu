// SPDX-License-Identifier: MIT
// Exact 51.1 matrix step "Y": D = R(C + sum of 32 e4m3 products), the sum exact and rounded once, on the plain fp8
// WMMA. Proof and evidence: results/2026-10-20-speed-body.md; hardware facts H1 / H2: gpu/probe_y.py.
//
// One operand of every product is split exactly into its nearest integer and the rest (both e4m3 codes):
//     b = bh + bl,  bh = rint(b) (an integer, |bh| <= 448),  bl = b - bh (|bl| <= 1/2, a multiple of 2^-9; bl has at
//     most 4 significant bits: for |b| < 1 it is b, b - 1 or b + 1 with b = (8 + m) / 16 at most; above, b's fraction
//     has at most 3 bits).
// The other operand a (weights, or P of the attention) is bounded: |a| <= A, every a a multiple of 2^g (the host checks
// 16 A < 2^(g + 15), g <= 0, and passes the bound of the start values, below). Two chains of WMMAs per step:
//     M = cb + sum a bh   (every term a multiple of 2^g,     sum of magnitudes < 2^(g + 24))
//     L =      sum a bl   (every term a multiple of 2^(g-9), sum of magnitudes <= 16 A < 2^(g + 15))
// with the start value c split as cb = rint(c) (an integer: on M's grid as g <= 0) and cs = c - cb (|cs| <= 1/2).
// (Driver rule found here: it folds (x + K) - K into x for a constant K, so no magic-constant rounding is used.)
// H1: each WMMA of a chain is exact (all its terms on one grid, everything below 2^24 grid units). Then
//     s = M + L (f32, rounded), t = L - (s - M) (exact rounding error), u = cs + t (exact),
// and C + sum = s + u exactly. H2: R(s + u) is ONE instruction, v_fma_mixlo/hi_f16 s, 1.0, u (the f32 add feeding the
// driver's f16 matrix conversion is folded into it): the exact sum rounded once, nearest even - the reference's step.
// With a zero start value the step is R(M + L) directly (one v_fma_mix, no other VALU work).
// Values outside the proven range (a start value |c| >= the host's bound, NaN / inf, NaN codes among the split
// operands) set rr_yflag; the kernel then marks its window and the window is recomputed by the fixup dispatch
// (the step-X kernel, RR_FIXUP), so such windows give exactly the previous configuration's results.
#pragma once
// RR_YABL: cost ablations for timing only, NOT exact, never in a shipped build (bits: 1 split skipped, 2 start split
// skipped, 4 one-add finish, 8 fraction chain dropped, 16 score on the integer chain only, 32 two-rounding finish,
// 64 (rr_block.hlsl, variant E) a second chain per step)
#ifndef RR_YABL
#define RR_YABL 0
#endif

struct RrSp { uint2 h, l; };          // split e4m3 fragment: integer parts h, fractions l
struct RrY { v8f m, l; };             // the two chains of a step

static uint rr_yflag;
static float rr_ymax;                 // running max of |start value|, compared with the bound at the window's end

// NaN codes (0x7f / 0xff) among 4 e4m3 bytes -> nonzero
uint rr_nan_bytes(uint x) { return ((x & 0x7f7f7f7fu) + 0x01010101u) & 0x80808080u; }

RrSp rr_split(uint2 q)
{
#if RR_YABL & 1
    { RrSp t; t.h = q; t.l = q ^ 0x01010101u; return t; }
#endif
    const v8f d = rr_d8(q);
    v8f h, l;
    [unroll] for (uint i = 0; i < 8; ++i) {
        h.v[i] = round(d.v[i]);                                      // v_rndne: nearest integer, ties to even
        l.v[i] = d.v[i] - h.v[i];
    }
    RrSp s;
    s.h = rr_q8_small(h);
    s.l = rr_q8_small(l);
    rr_yflag |= rr_nan_bytes(q.x) | rr_nan_bytes(q.y);
    return s;
}

// Start value c (f16 values as f32, a previous step's result): cb = rint(c) starts chain M, cs = c - cb is
// added at the end. max |c| goes into rr_ymax; the kernel marks the window when rr_ymax >= the bound.
// +-inf (a previous step overflowed) is clamped to +-2^20: the step's products sum to at most 32 * 448 A + 16 A <
// 2^20 - 65520 in magnitude (host check A <= 64), so the result stays +-inf, as R(+-inf + sum) of the reference.
// A start value is never NaN without an earlier mark: it is a step result (finite or +-inf, exact finite sums) of a
// window whose operands passed the NaN check.
RrY y_start(v8f c, out v8f cs)
{
    RrY r;
#if RR_YABL & 2
    r.m = c; r.l = wg_splat(0.0f); cs = wg_splat(0.0f); return r;
#endif
    float m = rr_ymax;
    [unroll] for (uint i = 0; i < 8; ++i) {
        const float cc = clamp(c.v[i], -1048576.0f, 1048576.0f);
        const float cb = round(cc);                                 // v_rndne (not a magic-constant pair)
        r.m.v[i] = cb;
        r.l.v[i] = 0.0f;
        cs.v[i] = cc - cb;
        m = max(m, abs(cc));
    }
    rr_ymax = m;
    return r;
}
// Start value from a bias vector split by the host (layout.py: [values | cb | cs], n values each; checked there).
RrY y_start_b(uint p, uint n, uint tile, out v8f cs)
{
    RrY r;
    r.m = wg_load8f(p, (n + 16u * tile + 8u * rr_hf) * 4u);
    r.l = wg_splat(0.0f);
    cs = wg_load8f(p, (2u * n + 16u * tile + 8u * rr_hf) * 4u);
    return r;
}
RrY y_zero()
{
    RrY r;
    r.m = wg_splat(0.0f);
    r.l = wg_splat(0.0f);
    return r;
}

// One 32-input step on split B operands (a: two A fragments, b: two split B fragments), and with the split operand
// as A (v^T). The calls alternate between the two accumulators (no register moves between dependent WMMAs).
RrY y_mm(uint2 a0, RrSp b0, uint2 a1, RrSp b1, RrY c)
{
#if RR_YABL & 8
    c.m = wg_wmma_fp8(a0, b0.h, c.m);
    c.m = wg_wmma_fp8(a1, b1.h, c.m);
#else
    c.m = wg_wmma_fp8(a0, b0.h, c.m);
    c.l = wg_wmma_fp8(a0, b0.l, c.l);
    c.m = wg_wmma_fp8(a1, b1.h, c.m);
    c.l = wg_wmma_fp8(a1, b1.l, c.l);
#endif
    return c;
}
RrY y_mmt(RrSp a0, uint2 b0, RrSp a1, uint2 b1, RrY c)
{
#if RR_YABL & 8
    c.m = wg_wmma_fp8(a0.h, b0, c.m);
    c.m = wg_wmma_fp8(a1.h, b1, c.m);
#else
    c.m = wg_wmma_fp8(a0.h, b0, c.m);
    c.l = wg_wmma_fp8(a0.l, b0, c.l);
    c.m = wg_wmma_fp8(a1.h, b1, c.m);
    c.l = wg_wmma_fp8(a1.l, b1, c.l);
#endif
    return c;
}

// R(M + L) (zero start value): one v_fma_mix per value.
uint4 y_fin0(RrY r)
{
    v8f s;
    [unroll] for (uint i = 0; i < 8; ++i) s.v[i] = r.m.v[i] + r.l.v[i];
    return rr_h16(s);
}
// R(M + L + cs) with the exact error of s = M + L.
uint4 y_fin(RrY r, v8f cs)
{
#if RR_YABL & 4
    return y_fin0(r);
#endif
#if RR_YABL & 32
    { v8f w2; [unroll] for (uint i2 = 0; i2 < 8; ++i2) w2.v[i2] = rr_fence(r.m.v[i2] + r.l.v[i2]) + cs.v[i2]; return rr_h16(w2); }
#endif
    v8f w;
    [unroll] for (uint i = 0; i < 8; ++i) {
        const float s = r.m.v[i] + r.l.v[i];
        const float z = s - r.m.v[i];
        const float t = r.l.v[i] - z;
        const float u = cs.v[i] + t;
        w.v[i] = s + u;                                              // folded: v_fma_mix s, 1.0, u
    }
    return rr_h16(w);
}

// The attention scores (51.3 step 2): both operands are activations (k^T and q), both split. Four chains:
//     I = ca + sum ah bh  (integers: ca = rint(c); |I| < 2^23)
//     M = sum ah bl + al bh  (multiples of 2^-9, sum of magnitudes <= 64 * 448 / 2 < 2^14)
//     L = sum al bl  (multiples of 2^-18, <= 32 / 4 = 8)
// every WMMA exact (H1). Then M = Mi + Mf (integer part, fraction: exact), I2 = I + Mi (integers < 2^24: exact),
// F = Mf + L (multiples of 2^-18 below 2^5: exact), s = I2 + F rounded with its exact error t = F - (s - I2), and
// R(c + products) = R(s + (cb + t)) with cb = c - ca (|cb| <= 1/2): u = cb + t exact, one v_fma_mix (H2).
// The score bias comes split by the host (layout.py: ca = rint(c), cb = c - ca; a finite f16, no bound needed).
uint4 y_score(RrSp a0, RrSp b0, RrSp a1, RrSp b1, v8f ca, v8f cb)
{
#if RR_YABL & 16
    { v8f q = wg_wmma_fp8(a0.h, b0.h, ca); q = wg_wmma_fp8(a1.h, b1.h, q); return rr_h16(q); }
#endif
    v8f I = ca, M, L;
    M = wg_splat(0.0f);
    L = wg_splat(0.0f);
    I = wg_wmma_fp8(a0.h, b0.h, I);
    M = wg_wmma_fp8(a0.h, b0.l, M);
    L = wg_wmma_fp8(a0.l, b0.l, L);
    M = wg_wmma_fp8(a0.l, b0.h, M);
    I = wg_wmma_fp8(a1.h, b1.h, I);
    M = wg_wmma_fp8(a1.h, b1.l, M);
    L = wg_wmma_fp8(a1.l, b1.l, L);
    M = wg_wmma_fp8(a1.l, b1.h, M);
    v8f w;
    [unroll] for (uint k = 0; k < 8; ++k) {
        const float mi = round(M.v[k]);
        const float mf = M.v[k] - mi;
        const float i2 = I.v[k] + mi;
        const float f = mf + L.v[k];
        const float s = i2 + f;
        const float z = s - i2;
        const float t = f - z;
        const float u = cb.v[k] + t;
        w.v[k] = s + u;                                              // folded: v_fma_mix s, 1.0, u
    }
    return rr_h16(w);
}

// P of the attention as the bounded operand: every code must be at most 1.0 (0x38) in magnitude.
void y_check_p(uint2 p)
{
    rr_yflag |= (((p.x & 0x7f7f7f7fu) + 0x47474747u) | ((p.y & 0x7f7f7f7fu) + 0x47474747u)) & 0x80808080u;
}

// Whole steps -> packed f16 results.
#if defined(RR_ZFAST) && RR_ZFAST >= 2
// RR_ZFAST 2 ("Y2" steps): both chains start from 0 (no start-value split, no bound on the start values), and
//     s = M + L is exact when |s| < rr_yb = 2^(g + 15): M and L are multiples of 2^(g - 9) (the fraction chain's grid,
//     the coarser integer chain's grid 2^g is a multiple of it), so every such value below 2^(g - 9 + 24) is an f32;
//     and fl(M + L) < 2^(g + 15) implies M + L < 2^(g + 15) (rounding is monotone, 2^(g + 15) is an f32).
// Then R(c + M + L) = R(c + s) is ONE v_fma_mix (H2) for every f16 start value c (+-inf included). A lane whose |s|
// reaches the bound sets rr_yflag: the window is redone by the fixup dispatch (step X). M is exact by H1 with
// 448 L1 < 2^(g + 24), L with L1 / 2 < 2^(g + 15) (stage.y_bounds). rr_yb is set per site (class g).
static float rr_yb;
uint4 y2_fin(RrY r, v8f c)
{
    v8f w;
    bool ok = true;
    [unroll] for (uint i = 0; i < 8; ++i) {
        const float s = r.m.v[i] + r.l.v[i];
        ok = ok && abs(s) < rr_yb;
        w.v[i] = s + rr_fence(c.v[i]);                               // folded: v_fma_mix s, 1.0, c
    }
    rr_yflag |= ok ? 0u : 1u;
    return rr_h16(w);
}
uint4 y_step_b(uint2 a0, RrSp b0, uint2 a1, RrSp b1, uint p, uint n, uint tile)
{
    return y2_fin(y_mm(a0, b0, a1, b1, y_zero()), wg_load8f(p, (16u * tile + 8u * rr_hf) * 4u));
}
uint4 y_step(uint2 a0, RrSp b0, uint2 a1, RrSp b1, v8f c) { return y2_fin(y_mm(a0, b0, a1, b1, y_zero()), c); }
uint4 y_stept(RrSp a0, uint2 b0, RrSp a1, uint2 b1, v8f c) { return y2_fin(y_mmt(a0, b0, a1, b1, y_zero()), c); }
uint4 y_step0(uint2 a0, RrSp b0, uint2 a1, RrSp b1) { return y_fin0(y_mm(a0, b0, a1, b1, y_zero())); }
uint4 y_step0t(RrSp a0, uint2 b0, RrSp a1, uint2 b1) { return y_fin0(y_mmt(a0, b0, a1, b1, y_zero())); }
#else
uint4 y_step_b(uint2 a0, RrSp b0, uint2 a1, RrSp b1, uint p, uint n, uint tile)
{
    v8f cs;
    const RrY r = y_mm(a0, b0, a1, b1, y_start_b(p, n, tile, cs));
    return y_fin(r, cs);
}
uint4 y_step0(uint2 a0, RrSp b0, uint2 a1, RrSp b1) { return y_fin0(y_mm(a0, b0, a1, b1, y_zero())); }
uint4 y_step0t(RrSp a0, uint2 b0, RrSp a1, uint2 b1) { return y_fin0(y_mmt(a0, b0, a1, b1, y_zero())); }
uint4 y_step(uint2 a0, RrSp b0, uint2 a1, RrSp b1, v8f c)
{
    v8f cs;
    const RrY r = y_mm(a0, b0, a1, b1, y_start(c, cs));
    return y_fin(r, cs);
}
uint4 y_stept(RrSp a0, uint2 b0, RrSp a1, uint2 b1, v8f c)
{
    v8f cs;
    const RrY r = y_mmt(a0, b0, a1, b1, y_start(c, cs));
    return y_fin(r, cs);
}
#endif

// ---- fast single-chain steps (variant Z = Y with RR_ZFAST 1) --------------------------------------------------------
// A wave takes a step on ONE chain of two WMMAs on the unsplit operands, and rounds R(c + M) with one v_fma_mix (H2),
// when the step's split operand block (the wave's 16 tokens x 32 inputs) is in one of two classes for which every
// product and the zero start value are multiples of one grid 2^q with all magnitudes summing below 2^(q + 24) (H1):
//   small: every |b| < T (host: T_class, the largest code with L1max * maxval(codes < T) < 2^(g + 15); products are
//          multiples of 2^(g - 9) since b's grid is >= 2^-9, and their magnitudes sum to at most L1max * max|b|);
//   int:   every b is 0 or |b| >= 8 (an integer; host: L1max * 448 < 2^(g + 24); products multiples of 2^g).
// L1max = the largest sum of |a| over the 32 inputs of one step of one output (host, per weight class), g = the bounded
// operand's lowest set bit. NaN codes are in neither class. The start value c never enters the chain: R(c + M) is one
// fused rounding of the exact sum (H2), for every f16 c (an infinite c gives the infinite result, as the reference).
#ifndef RR_ZFAST
#define RR_ZFAST 0
#endif
#if RR_ZFAST
uint z_tk(uint t) { return (0x80u - t) * 0x01010101u; }   // byte + tk has bit 7 set iff byte >= t (t <= 0x7f)
// per lane: bit 7 of some byte set iff one of the 8 codes is not below T
uint z_sbad(uint2 f, uint tk) { return ((f.x & 0x7f7f7f7fu) + tk) | ((f.y & 0x7f7f7f7fu) + tk); }
// per lane: bit 7 of some byte set iff one of the 8 codes is in 1 .. 0x4f (|b| in (0, 8): maybe not an integer) or NaN
uint z_ibad1(uint x)
{
    const uint v = x & 0x7f7f7f7fu;
    return ((v + 0x7f7f7f7fu) & ~(v + 0x30303030u)) | (v + 0x01010101u);
}
uint z_ibad(uint2 f) { return z_ibad1(f.x) | z_ibad1(f.y); }
// wave decision for a block whose lanes accumulated sb / ib
bool z_fast_ok(uint sb, uint ib, bool imode)
{
    const bool s = WaveActiveAllTrue((sb & 0x80808080u) == 0u);
    const bool i = WaveActiveAllTrue((ib & 0x80808080u) == 0u);
    return s || (imode && i);
}
// the step on one chain: zero start value, and R(c + M)
uint4 z_fast0(uint2 a0, uint2 b0, uint2 a1, uint2 b1)
{
    v8f m = wg_wmma_fp8(a0, b0, wg_splat(0.0f));
    m = wg_wmma_fp8(a1, b1, m);
    return rr_h16(m);
}
// Split of a weight class's operand on the fallback path: with the class bound T >= 8 (code 0x50) the partition
// h = the codes with |b| >= T (integers), l = the others (|l| < T: L1max * |l| < 2^(g + 15), the bound of the fraction
// chain); h | l = b, two masks per dword. Below 8 the nearest-integer split (rr_split). Either way the Y step's proof
// holds: h are integers (M chain on 2^g), l are multiples of 2^-9 with the fraction chain's sum below 2^(g + 15).
RrSp z_psplit(uint2 q, uint tk)
{
    const uint2 big = ((((q & 0x7f7f7f7fu) + tk) >> 7) & 0x01010101u) * 0xffu;
    RrSp s;
    s.h = q & big;
    s.l = q ^ s.h;
    rr_yflag |= rr_nan_bytes(q.x) | rr_nan_bytes(q.y);
    return s;
}
RrSp z_split(uint2 q, uint t)
{
    RrSp s;
    [branch] if (t >= 0x50u) s = z_psplit(q, z_tk(t));
    else s = rr_split(q);
    return s;
}
uint4 z_fast(uint2 a0, uint2 b0, uint2 a1, uint2 b1, v8f c)
{
    v8f m = wg_wmma_fp8(a0, b0, wg_splat(0.0f));
    m = wg_wmma_fp8(a1, b1, m);
    // folded: v_fma_mixlo/hi_f16 m, 1.0, c. The fence hides that c is a widened f16 value: otherwise the compiler
    // folds the widening into a v_fma_mix_f32 (an f32 rounding) and converts its result once more (double rounding).
    [unroll] for (uint i = 0; i < 8; ++i) m.v[i] = m.v[i] + rr_fence(c.v[i]);
    return rr_h16(m);
}
#endif

// ---- range gates (RR_ZFAST 2, levers L2 / L5 of results/2026-10-20-speed-body.md) ----------------------------------
// A wave takes a 51.1 step on ONE chain of the unsplit operands (z_fast / z_fast0) when its operand block passes
//     bm = value(am) 2^(10 - Elo) < R,   R = 2^(g + 24) / L1 (host, per weight class, rounded DOWN to an e5m3 byte),
// am = the block's largest code (magnitude), Elo = the smallest max(e, 1) over its nonzero codes (e = exponent field).
// Proof: every nonzero e4m3 value is an integer times 2^(max(e, 1) - 10), so the block's values are multiples of
// 2^(Elo - 10); every weight of the class is a multiple of 2^g; every product of the step is a multiple of
// 2^(g + Elo - 10). The 32 products of one output sum to at most L1 Bmax <= L1 value(am) in magnitude, and
// L1 value(am) < 2^(g + Elo - 10 + 24) is the gate. H1 makes both WMMAs of the chain exact (the first sums half the
// products from 0, the second adds the other half to that exact partial sum), H2 rounds c + M once (z_fast).
// An all-zero block passes (every product is 0); a NaN code (0x7f) never passes (am = 0x7f). The small and int
// classes of RR_ZFAST 1 are the cases Elo = 1 and Elo >= 10 of this gate.
#if RR_ZFAST >= 2
// per lane: low byte = largest magnitude code, second byte = max over the nonzero codes of 0x80 - code (0: all zero)
// (2026-10-22, track body-levers: the earlier form took 32-bit maxima of words holding two 16-bit lanes, so the lower
// lane's maximum could be lost and the gate could pass a block outside its class; Z2 was never verified. Bytes are now
// extracted one by one.)
uint zr_bmax(uint a) { return max(max(a & 0xffu, (a >> 8) & 0xffu), max((a >> 16) & 0xffu, a >> 24)); }
uint zr_acc(uint acc, uint2 f)
{
    const uint a0 = f.x & 0x7f7f7f7fu, a1 = f.y & 0x7f7f7f7fu;
    const uint w0 = (0x80808080u - a0) & 0x7f7f7f7fu, w1 = (0x80808080u - a1) & 0x7f7f7f7fu;   // no borrows
    const uint am = max(acc & 0xffu, max(zr_bmax(a0), zr_bmax(a1)));
    const uint wm = max((acc >> 8) & 0xffu, max(zr_bmax(w0), zr_bmax(w1)));
    return am | (wm << 8);
}
// the wave's statistics (wave-uniform)
uint zr_wave(uint acc) { return WaveActiveMax(acc & 0xffu) | (WaveActiveMax(acc & 0xff00u)); }
uint zr_merge(uint a, uint b) { return max(a & 0xffu, b & 0xffu) | max(a & 0xff00u, b & 0xff00u); }
// bm of the statistics (0 for an all-zero block, 2^30 with a NaN code)
float zr_bm(uint st)
{
    const uint am = st & 0xffu, wm = (st >> 8) & 0xffu;
    const uint e = am >> 3, elo = max((0x80u - wm) >> 3, 1u);
    const float v = ldexp((float)((e != 0u ? 8u : 0u) + (am & 7u)), (float)((int)max(e, 1u) - (int)elo));
    return am == 0x7fu ? 1073741824.0f : (wm == 0u ? 0.0f : v);
}
// R of an e5m3 byte: (8 + m) 2^(e - 3)
float zr_R(uint b) { return ldexp((float)(8u + (b & 7u)), (float)((int)(b >> 3) - 3)); }
bool zr_pass(uint st, uint rb) { return zr_bm(st) < zr_R(rb); }
// The small-class threshold T for the partition split of the slow path (z_psplit): the largest T with
// L1 value(T - 1) < 2^(g + 15), i.e. value(T - 1) < R / 2^9; from R rounded down (sound: a smaller T).
uint zr_T(uint rb)
{
    const float L = zr_R(rb) * (1.0f / 512.0f);                  // codes with value < L
    if (L > 448.0f) return 0x7fu;
    const uint u = asuint(L);
    const int E = (int)(u >> 23) - 127;
    uint c;                                                       // the largest code with value <= L
    if (E >= -6) c = (uint)(E + 7) * 8u + ((u >> 20) & 7u);
    else c = (uint)floor(L * 512.0f);
    const float vc = c >= 8u ? ldexp((float)(8u + (c & 7u)), (float)((int)(c >> 3) - 10)) : (float)c * (1.0f / 512.0f);
    return vc < L ? c + 1u : c;                                   // T - 1 = the largest code with value < L
}
#endif

#if RR_ZFAST >= 2
// Scores on two chains (RR_ZFAST 2): k unsplit (A), q split as rint / rest (B, rr_split): H = sum k qh, L = sum k ql.
// H: products multiples of 2^(Elok - 10) (qh integers), sum of magnitudes <= Kmax (L1q + 16) <= Kmax (32 Qmax + 16);
// L: multiples of 2^(Elok + Eloq - 20) (ql: q's grid, |ql| <= 1/2), sum <= 16 Kmax. The gate (zs2_ok):
// (32 Qmax + 16) bm(k) < 2^24 and bm(k) < 2^(Eloq + 10), bm(k) = value(am_k) 2^(10 - Elok) >= Kmax 2^(10 - Elok);
// then both chains are exact (H1). s = H + L is exact when |s| < 2^(Elok + Eloq + 4) (as y2_fin), and R(c + s) is one
// v_fma_mix (H2). ok is false when a lane's |s| reaches the bound (the caller then takes the four-chain step).
bool zs2_ok(uint sq, float bk)
{
    const uint am = sq & 0xffu, e = am >> 3;
    const float qmax = ldexp((float)((e != 0u ? 8u : 0u) + (am & 7u)), (float)((int)max(e, 1u) - 10));
    const uint eloq = max((0x80u - ((sq >> 8) & 0xffu)) >> 3, 1u);
    return am < 0x7fu && (32.0f * qmax + 16.0f) * bk < 16777216.0f && bk < ldexp(1.0f, (float)(eloq + 10u));
}
uint4 s2_score(uint2 a0, RrSp b0, uint2 a1, RrSp b1, v8f c, float bnd, out bool ok)
{
    v8f H = wg_wmma_fp8(a0, b0.h, wg_splat(0.0f));
    v8f L = wg_wmma_fp8(a0, b0.l, wg_splat(0.0f));
    H = wg_wmma_fp8(a1, b1.h, H);
    L = wg_wmma_fp8(a1, b1.l, L);
    bool good = true;
    v8f w;
    [unroll] for (uint i = 0; i < 8; ++i) {
        const float s = H.v[i] + L.v[i];
        good = good && abs(s) < bnd;
        w.v[i] = s + c.v[i];                                         // folded: v_fma_mix s, 1.0, c (c: f32 from memory)
    }
    ok = WaveActiveAllTrue(good);
    return rr_h16(w);
}
#endif
