// SPDX-License-Identifier: MIT
// The hybrid block form's norm and softmax row in the Swin block templates' register layout (SW_HYB=1; included by
// kernels/swin_block.hlsl after its layout macros): lane (t, hh) of a wave holds token t's channels hh C/2 + 8 k + v in
// v8f tiles k = 0 .. KC - 1, and the scores S(t, s) of keys s = 8 hh + v (gfx12 D rows, packing.resolve_d_row).
// The sums follow the original's fixed FP16 orders (clean/l/reference/oracle.py norm_q7, softmax_q7) exactly; the
// partner half (hh ^ 1) arrives by lane exchange, and both lanes of a token compute the same sums.
// Arithmetic in native FP16 (half2, packed): v_pk_add_f16 / v_pk_mul_f16 are correctly rounded single operations (the
// original's FP16 steps). Contraction: the only multiply feeding an add is the norm's square; `precise` does NOT stop
// the driver from fusing it (gfx1201), so the square passes an OR with a zero it cannot prove (HYB_OPAQUE0), and
// build_dx12.py's fused-square scan (hyb_isa_scan.py, with its *_scanmut mutants) fails a build where it fuses anyway.
// The softmax's two fused steps are float32 (hlsl/wg_hyb.hlsli hyb_softmax_e8: exact under any lowering).
#pragma once

typedef float16_t2 h2;
struct hv8 { h2 p[4]; };   // 8 FP16 values, element 2 i + j in p[i][j] (the order of a v8f)

hv8 hyb_to_h(v8f a)        // exact for FP16 values; otherwise RN16 (the driver's conversion, checked by the probe)
{
    hv8 r;
    [unroll] for (uint i = 0; i < 4; ++i) r.p[i] = h2((float16_t)a.v[2 * i], (float16_t)a.v[2 * i + 1]);
    return r;
}
v8f hyb_to_f(hv8 a)
{
    v8f r;
    [unroll] for (uint i = 0; i < 4; ++i) { r.v[2 * i] = (float)a.p[i].x; r.v[2 * i + 1] = (float)a.p[i].y; }
    return r;
}
// RN16(a + b) of 8 FP16 values held as float32 (exact: one float rounding first, 24 >= 2 x 11 + 2)
v8f hyb_add16(v8f a, v8f b)
{
    v8f s;
    [unroll] for (uint i = 0; i < 8; ++i) s.v[i] = a.v[i] + b.v[i];
    return hyb_round16(s);
}
hv8 hyb_hadd(hv8 a, hv8 b) { hv8 r; [unroll] for (uint i = 0; i < 4; ++i) r.p[i] = a.p[i] + b.p[i]; return r; }
#ifndef HYB_OPAQUE0
#error "hyb_block.hlsli: define HYB_OPAQUE0 (a uint zero the compiler cannot prove, e.g. from a kernel argument)"
#endif
hv8 hyb_hmul(hv8 a, hv8 b) { hv8 r; [unroll] for (uint i = 0; i < 4; ++i) r.p[i] = a.p[i] * b.p[i]; return r; }
uint hyb_h2u(h2 v) { return (uint)asuint16(v.x) | ((uint)asuint16(v.y) << 16); }
h2 hyb_u2h(uint u) { return h2(asfloat16((uint16_t)(u & 0xffffu)), asfloat16((uint16_t)(u >> 16))); }
hv8 hyb_hxor16(hv8 a)
{
    hv8 r;
    [unroll] for (uint i = 0; i < 4; ++i) r.p[i] = hyb_u2h(wg_xor16_u(hyb_h2u(a.p[i])));
    return r;
}
hv8 hyb_hsel(bool c, hv8 a, hv8 b) { hv8 r; [unroll] for (uint i = 0; i < 4; ++i) r.p[i] = select(c, a.p[i], b.p[i]); return r; }
uint4 hyb_hpack(hv8 a) { return uint4(hyb_h2u(a.p[0]), hyb_h2u(a.p[1]), hyb_h2u(a.p[2]), hyb_h2u(a.p[3])); }

// _group of oracle.norm_q7 over 1 .. 4 partial tiles: p0; p0 + p1; (p0 + p1) + p2; (p0 + p1) + (p2 + p3)
hv8 hyb_group(hv8 p[4], uint n)
{
    if (n == 1u) return p[0];
    if (n == 2u) return hyb_hadd(p[0], p[1]);
    if (n == 3u) return hyb_hadd(hyb_hadd(p[0], p[1]), p[2]);
    return hyb_hadd(hyb_hadd(p[0], p[1]), hyb_hadd(p[2], p[3]));
}

// The norm of the lane's token: x = this lane's KC tiles (FP16 values), gp = g' of the same channels (FP16 values).
// y = RN16(x RN16(r g')) with r = RN16(rsqrt32(S')) and S' = RN16(S + 2^-13), S = the FP16 tree of the squares.
void hyb_norm(v8f x[KC], v8f gp[KC], uint hh, out v8f y[KC])
{
    hv8 xh[KC], q0h[KC], qph[KC];   // this lane's squares, the partner lane's (its half of the token's channels)
#if !(KC == 2 || KC == 4 || KC == 8)
    hv8 T[2 * KC];
    const bool h0 = hh == 0u;
#endif
    [unroll] for (uint k = 0; k < KC; ++k) {
        xh[k] = hyb_to_h(x[k]);
        [unroll] for (uint i = 0; i < 4; ++i) {
            precise h2 sq = xh[k].p[i] * xh[k].p[i];   // q = RN16(x x): never fused into the tree's adds
            // contraction barrier: the driver fuses a square feeding an add into v_pk_fma_f16 despite precise (seen in
            // the ISA of the select-free tree: x x + q' on one lane of a token only, RN16(x x) skipped); an OR with
            // a zero it cannot prove leaves no multiply-add pattern (one v_or_b32 per register)
            q0h[k].p[i] = hyb_u2h(hyb_h2u(sq) | HYB_OPAQUE0);
        }
        qph[k] = hyb_hxor16(q0h[k]);
#if !(KC == 2 || KC == 4 || KC == 8)
#if defined(HYB_SCAN_MUTANT) && HYB_SCAN_MUTANT
        // the fused-square scan's SUBSTITUTE mutant for KC 6 (never in a model build; wrong values by design): with
        // the selects, every square reaches the tree only through a v_cndmask or the lane exchange (RGA: no square
        // feeds an add, so even without the barrier the driver forms no multiply-add and the plain mutant shows 0).
        // Here the tree takes the lane's own squares directly, a square feeds an add with no barrier, and the scan
        // must find the fused square in this kernel's ISA.
        T[k] = q0h[k];
        T[KC + k] = q0h[k];
#else
        T[k] = hyb_hsel(h0, q0h[k], qph[k]);            // tile j = channels 8 j .. 8 j + 7: low half j < KC
        T[KC + k] = hyb_hsel(h0, qph[k], q0h[k]);       // high half
#endif
#endif
    }
    // element e = 2 t + odd of tile j is the original's Q[j] of lane t, half odd: pair sums (j, j + 2), groups of the
    // even tiles j = 0, 4, 8, 12 (left) and the odd ones j = 1, 5, 9, 13 (right)
#if KC == 2 || KC == 4 || KC == 8
    // Select-free form (lever: the same FP16 sums, no per-lane tile selects). FP16 addition is commutative (a + b == b + a
    // bit for bit), and for KC = 2, 4, 8 every add of the tree whose operands the selects would swap between the two
    // lanes of a token adds the same two values in either order:
    //   KC 2: pairs (T0, T2) = {q0, q0'} and (T1, T3) = {q1, q1'};
    //   KC 4, 8: the left / right groups are (the group of this lane's own pairs) + (the group of the partner's pairs), and
    //   the partner lane computes the same two groups and adds them in the other order.
    // (KC 6: the three-pair group mixes own and partner tiles asymmetrically; the selects stay.)
    hv8 L;
#if KC == 2
    L = hyb_hadd(hyb_hadd(q0h[0], qph[0]), hyb_hadd(q0h[1], qph[1]));
#else
    {
        hv8 ol[KC / 4], orr[KC / 4], pl2[KC / 4], pr2[KC / 4];
        [unroll] for (uint j2 = 0; j2 < KC; j2 += 4u) {
            ol[j2 / 4u] = hyb_hadd(q0h[j2], q0h[j2 + 2u]);
            orr[j2 / 4u] = hyb_hadd(q0h[j2 + 1u], q0h[j2 + 3u]);
            pl2[j2 / 4u] = hyb_hadd(qph[j2], qph[j2 + 2u]);
            pr2[j2 / 4u] = hyb_hadd(qph[j2 + 1u], qph[j2 + 3u]);
        }
#if KC == 4
        L = hyb_hadd(hyb_hadd(ol[0], pl2[0]), hyb_hadd(orr[0], pr2[0]));
#else
        L = hyb_hadd(hyb_hadd(hyb_hadd(ol[0], ol[1]), hyb_hadd(pl2[0], pl2[1])),
                     hyb_hadd(hyb_hadd(orr[0], orr[1]), hyb_hadd(pr2[0], pr2[1])));
#endif
    }
#endif
#else
    const uint tiles = 2u * KC;
    hv8 pl[4], pr[4];
    uint nl = 0u, nr = 0u;
    [unroll] for (uint j = 0; j < tiles; j += 4u) {
        pl[nl++] = hyb_hadd(T[j], T[j + 2u]);
        pr[nr++] = hyb_hadd(T[j + 1u], T[j + 3u]);
    }
    const hv8 L = hyb_hadd(hyb_group(pl, nl), hyb_group(pr, nr));   // L[2 t + odd]
#endif
    // across the original's lanes t, per half: ((t0 + t1) + (t2 + t3)); then even + odd, + 2^-13 (packed pairs: even
    // in .x, odd in .y)
    const h2 u01 = L.p[0] + L.p[1], u23 = L.p[2] + L.p[3];   // (t0 + t1), (t2 + t3) for both halves
    const h2 hv = u01 + u23;                                   // half(0), half(1)
    const float16_t S = hv.x + hv.y;
    const float16_t S2 = S + (float16_t)asfloat16((uint16_t)0x0800u);   // + 2^-13
    const float16_t r = (float16_t)hyb_rsq16((uint)asuint16(S2));
    const h2 r2 = h2(r, r);
    [unroll] for (uint k2 = 0; k2 < KC; ++k2) {
        const hv8 g = hyb_to_h(gp[k2]);
        hv8 yh;
        [unroll] for (uint i2 = 0; i2 < 4; ++i2) yh.p[i2] = xh[k2].p[i2] * (r2 * g.p[i2]);
        y[k2] = hyb_to_f(yh);
    }
}

hv8 hyb_u4_to_h(uint4 u)
{
    hv8 r;
    r.p[0] = hyb_u2h(u.x); r.p[1] = hyb_u2h(u.y); r.p[2] = hyb_u2h(u.z); r.p[3] = hyb_u2h(u.w);
    return r;
}

// Fast norm from already-packed FP16 tiles xh_u (uint4[KC], e.g. SW_Q7_CHUNK norm2 acc_h):
// avoids 8*KC v_cvt_f32_f16 + 4*KC v_cvt_pk_rtz round-trip conversions; exact same FP16 ops in the exact same order.
void hyb_norm_h(uint4 xh_u[KC], v8f gp[KC], uint hh, out v8f y[KC])
{
    hv8 xh[KC], q0h[KC], qph[KC];
#if !(KC == 2 || KC == 4 || KC == 8)
    hv8 T[2 * KC];
    const bool h0 = hh == 0u;
#endif
    [unroll] for (uint k = 0; k < KC; ++k) {
        xh[k] = hyb_u4_to_h(xh_u[k]);
        [unroll] for (uint i = 0; i < 4; ++i) {
            precise h2 sq = xh[k].p[i] * xh[k].p[i];
            q0h[k].p[i] = hyb_u2h(hyb_h2u(sq) | HYB_OPAQUE0);
        }
        qph[k] = hyb_hxor16(q0h[k]);
#if !(KC == 2 || KC == 4 || KC == 8)
#if defined(HYB_SCAN_MUTANT) && HYB_SCAN_MUTANT
        T[k] = q0h[k];
        T[KC + k] = q0h[k];
#else
        T[k] = hyb_hsel(h0, q0h[k], qph[k]);
        T[KC + k] = hyb_hsel(h0, qph[k], q0h[k]);
#endif
#endif
    }
#if KC == 2 || KC == 4 || KC == 8
    hv8 L;
#if KC == 2
    L = hyb_hadd(hyb_hadd(q0h[0], qph[0]), hyb_hadd(q0h[1], qph[1]));
#else
    {
        hv8 ol[KC / 4], orr[KC / 4], pl2[KC / 4], pr2[KC / 4];
        [unroll] for (uint j2 = 0; j2 < KC; j2 += 4u) {
            ol[j2 / 4u] = hyb_hadd(q0h[j2], q0h[j2 + 2u]);
            orr[j2 / 4u] = hyb_hadd(q0h[j2 + 1u], q0h[j2 + 3u]);
            pl2[j2 / 4u] = hyb_hadd(qph[j2], qph[j2 + 2u]);
            pr2[j2 / 4u] = hyb_hadd(qph[j2 + 1u], qph[j2 + 3u]);
        }
#if KC == 4
        L = hyb_hadd(hyb_hadd(ol[0], pl2[0]), hyb_hadd(orr[0], pr2[0]));
#else
        L = hyb_hadd(hyb_hadd(hyb_hadd(ol[0], ol[1]), hyb_hadd(pl2[0], pl2[1])),
                     hyb_hadd(hyb_hadd(orr[0], orr[1]), hyb_hadd(pr2[0], pr2[1])));
#endif
    }
#endif
#else
    const uint tiles = 2u * KC;
    hv8 pl[4], pr[4];
    uint nl = 0u, nr = 0u;
    [unroll] for (uint j = 0; j < tiles; j += 4u) {
        pl[nl++] = hyb_hadd(T[j], T[j + 2u]);
        pr[nr++] = hyb_hadd(T[j + 1u], T[j + 3u]);
    }
    const hv8 L = hyb_hadd(hyb_group(pl, nl), hyb_group(pr, nr));
#endif
    const h2 u01 = L.p[0] + L.p[1], u23 = L.p[2] + L.p[3];
    const h2 hv = u01 + u23;
    const float16_t S = hv.x + hv.y;
    const float16_t S2 = S + (float16_t)asfloat16((uint16_t)0x0800u);
    const float16_t r = (float16_t)hyb_rsq16((uint)asuint16(S2));
    const h2 r2 = h2(r, r);
    [unroll] for (uint k2 = 0; k2 < KC; ++k2) {
        const hv8 g = hyb_to_h(gp[k2]);
        hv8 yh;
        [unroll] for (uint i2 = 0; i2 < 4; ++i2) yh.p[i2] = xh[k2].p[i2] * (r2 * g.p[i2]);
        y[k2] = hyb_to_f(yh);
    }
}

// The attention weights of the lane's 8 keys from its 8 scores s (float32 chain values started at B):
// RN16(s), e = hyb_softmax_e8, the row sum in the original's order (a_t = e_2t + e_8+2t, b_t = e_2t+1 + e_9+2t;
// a = (a0 + a1) + (a2 + a3), b likewise; sum = a + b), w = RN16(rcp32(sum)), P = RN16(e w) as packed FP16.
// Packed half2 form for u = clamp(RN16(s * c), +-c'), p = FMA16(u, t, P), e_bits = (p_bits & 0x03ff03ff) << 5;
// t = RN16(fma(-u, u, T)) stays float32 so hyb_isa_scan rule 1 sees no FP16 square FMA. Bit-identical for all inputs.
struct HybSmaxPre { hv8 e; float16_t w; };
HybSmaxPre hyb_softmax_pre(v8f s)
{
    const hv8 sh = hyb_u4_to_h(hyb_rn16x8(s));
    const h2 c2 = h2(asfloat16((uint16_t)0x2466u), asfloat16((uint16_t)0x2466u));
    const h2 cl = h2(asfloat16((uint16_t)0x3873u), asfloat16((uint16_t)0x3873u));
    const h2 ncl = h2(asfloat16((uint16_t)0xb873u), asfloat16((uint16_t)0xb873u));
    hv8 uh;
    [unroll] for (uint i = 0; i < 4; ++i) uh.p[i] = min(max(sh.p[i] * c2, ncl), cl);
    const v8f u = hyb_to_f(uh);
    v8f t;
    [unroll] for (uint i2 = 0; i2 < 8; ++i2) t.v[i2] = mad(u.v[i2], -u.v[i2], WG_HYB_POLY_T);
    const hv8 th = hyb_u4_to_h(hyb_rn16x8(t));
    const h2 p0 = h2(asfloat16((uint16_t)0x3d80u), asfloat16((uint16_t)0x3d80u));
    HybSmaxPre r;
    [unroll] for (uint i3 = 0; i3 < 4; ++i3) {
        const h2 ph = mad(uh.p[i3], th.p[i3], p0);
        r.e.p[i3] = hyb_u2h((hyb_h2u(ph) & 0x03ff03ffu) << 5);
    }
    const hv8 ep = hyb_hxor16(r.e);
    const hv8 A = hyb_hadd(r.e, ep);
    const h2 ab = (A.p[0] + A.p[1]) + (A.p[2] + A.p[3]);                 // (a, b)
    const float16_t sum = ab.x + ab.y;
    r.w = (float16_t)hyb_rcp16((uint)asuint16(sum));
    return r;
}
uint4 hyb_softmax_fin(HybSmaxPre r)
{
    const h2 w2 = h2(r.w, r.w);
    hv8 P;
    [unroll] for (uint i = 0; i < 4; ++i) P.p[i] = r.e.p[i] * w2;
    return hyb_hpack(P);
}
uint4 hyb_softmax_p(v8f s, uint hh) { return hyb_softmax_fin(hyb_softmax_pre(s)); }
