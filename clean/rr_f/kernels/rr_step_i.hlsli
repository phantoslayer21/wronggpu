// SPDX-License-Identifier: MIT
// Exact 51.1 matrix step "I" (RR_STEP 4, lever L4 of results/2026-10-21-body-levers.md): D = R(C + sum of 32 e4m3
// products), the sum exact in INTEGER arithmetic on v_wmma_i32_16x16x16_iu8 (driver channel op 0x28, format 0xC: A
// and B signed int8, int32 accumulator; clean/dx12/hlsl/wg_ags.hlsli), then ONE rounding to binary16, nearest even.
//
// Integer form. Every finite e4m3 value is v = M 2^(E - 10), M = 0..15 (8 + mantissa, or the mantissa of a
// subnormal), E = max(exponent field, 1). An operand block whose nonzero values have E >= beta is exactly
//     v = v' 2^(beta - 10),   v' = M 2^(E - beta) an integer, |v'| <= 15 * 2^(Emax - beta) < 2^(4 + span),
// span = Emax - beta <= 14. v' is split into signed 7-bit digits v' = d0 + 2^7 d1 + 2^14 d2 (|d| <= 127, all with the
// sign of v'; d2 = 0 when span <= 10, d1 = d2 = 0 when span <= 3): two or three int8 planes. The WMMA sums over K
// only, so one scale per operand block is allowed: beta of the weights is per (16-output tile, 32-input step) (host
// table, layout.int_planes), beta of an activation block per (wave, step) (wave-uniform: the result scale is one
// scalar g per tile-step). With A digits a_p and B digits b_q, the step's integer sum
//     N = sum_k a'_k b'_k = sum_(p,q) 2^(7 (p + q)) sum_k a_pk b_qk
// is accumulated by Horner over the levels L = p + q, highest first: acc = (acc << 7) + one WMMA per digit pair and
// K-block. Every WMMA adds exact integer products to an exact integer (two's complement, no saturation) as long as
// no partial sum leaves int32: every partial sum of the Horner scheme is a sum of a subset of the terms
// |a'_k b'_k| 2^(-7 L) <= sum_k |a'_k| |b'_k| <= L1' Bmax', and the tile-step is taken only when
// L1' Bmax' < 2^31 - 2^15 (L1' = the largest sum of |a'| over the step of one output row (host table, or 32 Amax' for
// an activation block: its largest row / column sum), Bmax' = the block's largest |b'|; ri_ok).
// Otherwise rr_yflag marks the window and the fixup dispatch (step X) recomputes it.
//
// Rounding (ri_fin). The step value is S = c + N g, g = 2^G, G = betaA + betaB - 20; c the f16 start value (as f32).
//   |N| < 2^24 for every value of the tile (wave-uniform test): x = f32(N) is exact, x g is exact (a power of two,
//     no f32 under- or overflow: -18 <= G + 4 L <= 10 + ...), and R(x g + c) is one v_fma_mix (H2: the f32 add
//     feeding the driver's f16 conversion is ONE rounding of the exact sum; c is fenced as in z_fast).
//   otherwise (|N| < 2^31): x = RNE32(N), r = N - int(x) (exact, |r| <= 64), p = x g (exact), (s, e) = TwoSum(p, c)
//     (s + e = p + c exactly), u = e + r g, and S = s + u. u is exact in f32 when G <= -8: e and r g are multiples of
//     delta = min(g, lsb(c)) >= 2^-24 (lsb(c): c is an f16), |e| <= 2^-24 |s| <= 2^-24 (2^31 g + 65504) and
//     |r g| <= 64 g, so |u| <= 192 g + 2^-8 < 2^24 delta: for delta = g since 2^-8 < (2^24 - 192) g (g >= 2^-18);
//     for delta = lsb(c) >= 2^-24 since 192 g + 2^-8 < 1 for G <= -8. Then R(s + u) is one v_fma_mix (H2). For
//     G > -8 the window is marked (fixup). Non-finite c: R(x g + c) of the first form (x finite) is +-inf / NaN as
//     the reference's R(c + sum).
// NaN codes (0x7f / 0xff) in an operand block mark the window (fixup); weights are checked by the host.
// The hardware facts used: H1' (v_wmma_i32_16x16x16_iu8 returns A B + C exactly in int32 for |partial sums| < 2^31;
// gpu/probe_i8.py) and H2 (gpu/probe_y.py, re-checked with a power-of-two multiplier by probe_i8.py).
#pragma once

#define WG_AGS_I8_I32 0xCu                    // v_wmma_i32_16x16x16_iu8, A signed, B signed

static uint rr_yflag;
#ifdef RR_IDBG
static uint rr_isite;                         // debug: flag bits shifted by 4 for the two-activation steps
#endif                         // nonzero: the window is recomputed by the fixup dispatch (step X)

void ri_two_sum(float a, float b, out float s, out float e)   // Knuth: s + e = a + b exactly (no overflow)
{
    s = a + b;
    const float bb = s - a;
    e = (a - (s - bb)) + (b - bb);
}

struct RrIA { uint v[8]; };                   // int32 accumulator fragment (element i of lane l: D[8 (l >> 4) + i][l & 15])
struct RrDg { uint2 p0, p1, p2; };            // the three int8 digit planes of one fragment (8 values)

RrIA ri_zero() { RrIA r; [unroll] for (uint i = 0; i < 8; ++i) r.v[i] = 0u; return r; }
RrIA ri_mma(uint2 a, uint2 b, RrIA c)
{
    wg_frag cf;
    [unroll] for (uint i = 0; i < 8; ++i) cf.v[i] = c.v[i];
    const wg_frag r = wg_ags_mma(WG_AGS_I8_I32, wg_frag2(a), wg_frag2(b), cf);
    RrIA o;
    [unroll] for (uint k = 0; k < 8; ++k) o.v[k] = r.v[k];
    return o;
}
RrIA ri_shl(RrIA c) { [unroll] for (uint i = 0; i < 8; ++i) c.v[i] = c.v[i] << 7; return c; }

// NaN codes (0x7f / 0xff) among 4 e4m3 bytes -> nonzero (no carries: every masked byte + 1 <= 0x80)
uint rr_nan_bytes_i(uint x) { return ((x & 0x7f7f7f7fu) + 0x01010101u) & 0x80808080u; }

// ---- digits ------------------------------------------------------------------------------------------------------
uint ri_pack(int a, int b, int c, int d)
{
    return ((uint)a & 0xffu) | (((uint)b & 0xffu) << 8) | (((uint)c & 0xffu) << 16) | ((uint)d << 24);
}
// the digit planes of 8 codes on the scale 2^(beta - 10): s = 2^(10 - beta). v' = d s is an exact integer below 2^18;
// t2 = trunc(v' / 2^14), r = v' - 2^14 t2, t1 = trunc(r / 2^7), t0 = r - 2^7 t1 (all exact in f32, signs of v').
RrDg ri_digits_d(v8f d, float s)
{
    int l[8], m[8], h[8];
    [unroll] for (uint i = 0; i < 8; ++i) {
        const float v = d.v[i] * s;
        const float t2 = trunc(v * (1.0f / 16384.0f));
        const float r = mad(t2, -16384.0f, v);
        const float t1 = trunc(r * (1.0f / 128.0f));
        l[i] = (int)mad(t1, -128.0f, r);
        m[i] = (int)t1;
        h[i] = (int)t2;
    }
    RrDg g;
    g.p0 = uint2(ri_pack(l[0], l[1], l[2], l[3]), ri_pack(l[4], l[5], l[6], l[7]));
    g.p1 = uint2(ri_pack(m[0], m[1], m[2], m[3]), ri_pack(m[4], m[5], m[6], m[7]));
    g.p2 = uint2(ri_pack(h[0], h[1], h[2], h[3]), ri_pack(h[4], h[5], h[6], h[7]));
    return g;
}
RrDg ri_digits(uint2 q, float s) { return ri_digits_d(rr_d8(q), s); }
RrDg ri_planes2(uint2 p0, uint2 p1) { RrDg g; g.p0 = p0; g.p1 = p1; g.p2 = uint2(0u, 0u); return g; }

// ---- the integer sum of one step (two K-blocks), Horner over the digit levels ------------------------------------------
#define RI_PAIR(P, Q) acc = ri_mma(a0.p##P, b0.p##Q, acc); acc = ri_mma(a1.p##P, b1.p##Q, acc);
RrIA ri_mm22(RrDg a0, RrDg b0, RrDg a1, RrDg b1)
{
    RrIA acc = ri_zero();
    RI_PAIR(1, 1) acc = ri_shl(acc);
    RI_PAIR(1, 0) RI_PAIR(0, 1) acc = ri_shl(acc);
    RI_PAIR(0, 0)
    return acc;
}
RrIA ri_mm23(RrDg a0, RrDg b0, RrDg a1, RrDg b1)
{
    RrIA acc = ri_zero();
    RI_PAIR(1, 2) acc = ri_shl(acc);
    RI_PAIR(1, 1) RI_PAIR(0, 2) acc = ri_shl(acc);
    RI_PAIR(1, 0) RI_PAIR(0, 1) acc = ri_shl(acc);
    RI_PAIR(0, 0)
    return acc;
}
RrIA ri_mm32(RrDg a0, RrDg b0, RrDg a1, RrDg b1)
{
    RrIA acc = ri_zero();
    RI_PAIR(2, 1) acc = ri_shl(acc);
    RI_PAIR(2, 0) RI_PAIR(1, 1) acc = ri_shl(acc);
    RI_PAIR(1, 0) RI_PAIR(0, 1) acc = ri_shl(acc);
    RI_PAIR(0, 0)
    return acc;
}
RrIA ri_mm33(RrDg a0, RrDg b0, RrDg a1, RrDg b1)
{
    RrIA acc = ri_zero();
    RI_PAIR(2, 2) acc = ri_shl(acc);
    RI_PAIR(2, 1) RI_PAIR(1, 2) acc = ri_shl(acc);
    RI_PAIR(2, 0) RI_PAIR(1, 1) RI_PAIR(0, 2) acc = ri_shl(acc);
    RI_PAIR(1, 0) RI_PAIR(0, 1) acc = ri_shl(acc);
    RI_PAIR(0, 0)
    return acc;
}
// weights (two planes) x an activation block (two or three planes, wave-uniform b3)
RrIA ri_mm_w(RrDg a0, RrDg b0, RrDg a1, RrDg b1, bool b3)
{
    RrIA r;
    [branch] if (b3) r = ri_mm23(a0, b0, a1, b1);
    else r = ri_mm22(a0, b0, a1, b1);
    return r;
}
// an activation block as A (two or three planes) x weights (two planes)
RrIA ri_mm_t(RrDg a0, RrDg b0, RrDg a1, RrDg b1, bool a3)
{
    RrIA r;
    [branch] if (a3) r = ri_mm32(a0, b0, a1, b1);
    else r = ri_mm22(a0, b0, a1, b1);
    return r;
}
// two activation blocks
RrIA ri_mm_aa(RrDg a0, RrDg b0, RrDg a1, RrDg b1, bool d3)
{
    RrIA r;
    [branch] if (d3) r = ri_mm33(a0, b0, a1, b1);
    else r = ri_mm22(a0, b0, a1, b1);
    return r;
}

// ---- the one rounding ---------------------------------------------------------------------------------------------
// R(c + N 2^G) for the 8 values of a lane (see the header). ok: the int32 bound of the tile-step held (else the window
// is marked and the values are not used).
uint4 ri_fin(RrIA n, v8f c, int G, bool ok)
{
    const float g = ldexp(1.0f, (float)G);
    v8f x;
    float mx = 0.0f;
    [unroll] for (uint i = 0; i < 8; ++i) {
        x.v[i] = (float)(int)n.v[i];
        mx = max(mx, abs(x.v[i]));
    }
    v8f w;
    uint4 h;
    const bool small = WaveActiveAllTrue(mx < 16777216.0f);
#ifdef RR_ICOV                                // coverage (debug only; bits above 0xff never mark a window):
    {                                         // 0x100 TwoSum path taken unmarked, 0x400 a +-inf / NaN start
        bool nf = false;
        [unroll] for (uint q = 0; q < 8; ++q) nf = nf || !isfinite(c.v[q]);
        rr_yflag |= WaveActiveAnyTrue(nf) ? 0x400u : 0u;
        rr_yflag |= (!small && G <= -8 && ok) ? 0x100u : 0u;
    }
#endif
    // The conversion must sit in the same block as the add it folds (v_fma_mix): rr_h16 inside each branch. The exact
    // product x g is fenced, else the driver contracts x g + c into an f32 fma (an f32 rounding before the f16 one).
    [branch] if (small) {
        [unroll] for (uint k = 0; k < 8; ++k) w.v[k] = rr_fence(x.v[k] * g) + rr_fence(c.v[k]);   // v_fma_mix (H2)
        h = rr_h16(w);
    } else {
        rr_yflag |= (G <= -8 && ok) ? 0u : 1u;
#ifdef RR_IDBG
        rr_yflag |= G <= -8 ? 0u : (8u << rr_isite);
#endif
        [unroll] for (uint k = 0; k < 8; ++k) {
            const float r = (float)((int)n.v[k] - (int)x.v[k]);                  // exact, |r| <= 64
            const float p = x.v[k] * g;                                           // exact
            const float cc = rr_fence(c.v[k]);
            float s, e;
            ri_two_sum(p, cc, s, e);
            const float u = mad(r, g, e);                                         // exact (header)
            const bool fin = isfinite(cc);                                        // +-inf / NaN start: R(c + 0)
            w.v[k] = (fin ? s : cc) + (fin ? u : 0.0f);                           // v_fma_mix (H2)
        }
        h = rr_h16(w);
    }
    rr_yflag |= ok ? 0u : 1u;
#ifdef RR_IDBG
    rr_yflag |= ok ? 0u : (4u << rr_isite);
#endif
    return h;
}
// the int32 bound of a tile-step: L1' Bmax' < 2^31 - 2^15. The bound's f32 inputs may be rounded (sums of up to 32
// values, the table's f32 L1', the product): relative error below 2^-18 in total, inside the margin 2^-16.
bool ri_ok(float l1, float bmax) { return l1 * bmax < 2147450880.0f; }

// ---- an activation block of one step (two fragments: 16 tokens x 32 inputs of the wave) ---------------------------------
struct RrAB { RrDg d0, d1; int beta; bool d3; float vmax, l1; };   // l1: largest row / column sum of |v'|
// Statistics from the decoded values (exact f32): the largest |d| and the smallest nonzero |d| of the wave's block
// (as the unsigned order of the f32 bits; |d| - 1 as uint maps 0 to the largest value), NaN codes by their bytes.
// beta = max(e_f + 7, 1) with e_f the f32 exponent of the smallest nonzero |d| (an e4m3 value d = (1.m) 2^(e - 7) for
// e >= 1, below 2^-6 for e = 0), likewise Emax; vmax = max |d| 2^(10 - beta) = the largest |v'|.
RrAB ri_act(uint2 f0, uint2 f1)
{
    const v8f d0 = rr_d8(f0), d1 = rr_d8(f1);
    uint mx = 0u, mn = 0xffffffffu;
    float sa = 0.0f;                                                   // sum of |d| of this lane's 16 values
    [unroll] for (uint i = 0; i < 8; ++i) {
        const uint a = asuint(d0.v[i]) & 0x7fffffffu, b = asuint(d1.v[i]) & 0x7fffffffu;
        mx = max(mx, max(a, b));
        mn = min(mn, min(a - 1u, b - 1u));
        sa = sa + abs(d0.v[i]) + abs(d1.v[i]);
    }
    // the row (A) / column (B) of lane l & 15 holds the 32 values of lanes l and l ^ 16 (f32 sums of e4m3 values:
    // rounded up by at most 2^-24 relative per add, covered by the margin of ri_ok)
    const float rsum = WaveActiveMax(sa + WaveReadLaneAt(sa, WaveGetLaneIndex() ^ 16u));
    const uint nanb = rr_nan_bytes_i(f0.x) | rr_nan_bytes_i(f0.y) | rr_nan_bytes_i(f1.x) | rr_nan_bytes_i(f1.y);
    rr_yflag |= WaveActiveAnyTrue(nanb != 0u) ? 1u : 0u;
#ifdef RR_IDBG
    rr_yflag |= WaveActiveAnyTrue(nanb != 0u) ? 2u : 0u;
#endif
    mx = WaveActiveMax(mx);
    mn = WaveActiveMin(mn);
    RrAB r;
    const bool any = mn != 0xffffffffu;                                // a nonzero value in the block
    const int emax = max((int)(mx >> 23) - 127 + 7, 1);
    r.beta = any ? max((int)((mn + 1u) >> 23) - 127 + 7, 1) : 1;
    r.d3 = any && emax - r.beta > 10;
#ifdef RR_ICOV
    rr_yflag |= r.d3 ? 0x200u : 0u;           // coverage: three activation digits
#endif
    r.vmax = any ? asfloat(mx) * ldexp(1.0f, (float)(10 - r.beta)) : 0.0f;
    r.l1 = rsum * ldexp(1.0f, (float)(10 - r.beta));
    const float sc = ldexp(1.0f, (float)(10 - r.beta));
    r.d0 = ri_digits_d(d0, sc);
    r.d1 = ri_digits_d(d1, sc);
    return r;
}
// weights (A: K-block 0 planes a00 / a01, K-block 1 planes a10 / a11; table entry t = (beta, L1' bits)) x activations
uint4 ri_step_w(uint2 a00, uint2 a01, uint2 a10, uint2 a11, RrAB b, v8f c, uint2 t)
{
#ifdef RR_IDBG
    rr_isite = 0u;
#endif
    const RrIA n = ri_mm_w(ri_planes2(a00, a01), b.d0, ri_planes2(a10, a11), b.d1, b.d3);
    return ri_fin(n, c, (int)t.x + b.beta - 20, ri_ok(asfloat(t.y), b.vmax));
}
// activations as A x weights as B
uint4 ri_step_t(RrAB a, uint2 b00, uint2 b01, uint2 b10, uint2 b11, v8f c, uint2 t)
{
#ifdef RR_IDBG
    rr_isite = 0u;
#endif
    const RrIA n = ri_mm_t(a.d0, ri_planes2(b00, b01), a.d1, ri_planes2(b10, b11), a.d3);
    return ri_fin(n, c, (int)t.x + a.beta - 20, ri_ok(asfloat(t.y), a.vmax));
}
// two activation blocks (scores, O): sum_k |a'_k| |b'_k| <= min(L1A' Bmax', Amax' L1B')
uint4 ri_step_aa(RrAB a, RrAB b, v8f c)
{
#ifdef RR_IDBG
    rr_isite = 4u;
#endif
#ifdef RR_ICOV
    rr_yflag |= (a.d3 || b.d3) ? 0x800u : 0u; // coverage: the 3 x 3 digit form
#endif
    const RrIA n = ri_mm_aa(a.d0, b.d0, a.d1, b.d1, a.d3 || b.d3);
    return ri_fin(n, c, a.beta + b.beta - 20, ri_ok(1.0f, min(a.l1 * b.vmax, a.vmax * b.l1)));
}
