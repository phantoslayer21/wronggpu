// SPDX-License-Identifier: MIT
// Helpers of the specialised window-block kernels (rr_block_c32.hlsl, rr_block_fast.hlsl): the same operations as
// rr_common.hlsli / rr_block.hlsl, arranged for fewer instructions; every result has the bits of the rr_block.hlsl
// form for every input (argued at each helper). Needs rr_common.hlsli and the kernel's static rr_lane.
#pragma once
// clean/rr-f-a14-l01 A2 smajor-elementwise (RR_SMAJOR 1, default): bv_act / bv_attn of the relaxed body written
// stage-major (all 8 clamps, then all 8 u, ...); the same operations per value (bit-identical: cap8 + cap48 accurate vs
// MXR, fast vs FAST; the exact body has no RR_B32 forms); -0.056 / -0.065 ms (1080p / 1440p Q, game_bench, 4 rounds).
// 0: the value-major text. 2: also act2 of IN0's folded front-end MLP (A3, act2_4).
#ifndef RR_SMAJOR
#define RR_SMAJOR 1
#endif

// ---- Q8 fast path ----------------------------------------------------------------------------------------------------
// The unclamped driver conversion equals Q8 (rr_q8) for every finite |x| <= 464 (values in (448, 464] round to
// +-448 = the clamp), and every other input (|x| > 464, +-inf, NaN) gives a code 0x7f / 0xff, which no Q8 result
// has. A wave that sees such a code in any value of a Q8 group recomputes the group with rr_q8 (wave-uniform branch).
// q = the unclamped conversion; bad collects, per byte, bit 7 set when the byte's low 7 bits are 0x7f (a NaN code).
#ifndef L01_Q8CLAMP
#define L01_Q8CLAMP 0         // clean/rr-f-a14-l01 A46 (gated opt-in; bit as L01_DEFERMASK, set by the l01 kernels' RR_L01BIT)
#endif
uint2 q8f(v8f d, inout uint bad)
{
#if defined(RR_L01BIT) && (L01_Q8CLAMP & RR_L01BIT) != 0
    // A46 q8-clamp-nocheck: clamp to +-448 (v_maxmin; a NaN gives one of the bounds, not 0x7f / 0xff) then the driver
    // conversion; no bad codes collected, so the group's slow path is never taken. NOT exact for NaN inputs.
    v8f c;
    [unroll] for (uint i = 0; i < 8; ++i) c.v[i] = min(max(d.v[i], -448.0f), 448.0f);
    return wg_pack_q8_nc_ags(c);
#endif
    const uint2 q = wg_pack_q8_nc_ags(d);
    bad |= ((q.x & 0x7f7f7f7fu) + 0x01010101u) | ((q.y & 0x7f7f7f7fu) + 0x01010101u);
    return q;
}
#ifndef RR_COUNT
#define RR_COUNT 0            // 1: counting build (isa_stats --dyn): fast paths only, loops unrolled; never run
#endif
#if RR_COUNT
bool q8_slow(uint bad) { return RR_COUNT == 2; }   // 2: every Q8 group on its slow path
#define RR_LOOP [unroll]
#define RR_ZL 0u
#else
bool q8_slow(uint bad) { return WaveActiveAnyTrue((bad & 0x80808080u) != 0u); }
#define RR_LOOP [loop]
#define RR_ZL rr_zero()
#endif

// Q8 of 8 packed f16 values with the driver conversion straight from f16 (v_cvt_f32_f16 + v_cvt_pk_fp8_f32, the f16
// value widened exactly). The driver folds an f16 fma feeding this conversion into v_fma_mix_f32 (no f16 rounding), so
// fma results are fenced (pk_mad); f16 products (P, the activation) are not folded (isa_stats checks every build for
// v_fma_mix). The slow path widens a fenced copy.
#if WG_WAVE == 64
// clean/rr-mp2-level0 L0a (rr_block_c32.hlsl design D): one conversion per tile (h.xy = tile 0, h.zw = tile 1; the
// wave64 f16 fragment is 2 dwords), the same per-value conversion and byte order
uint2 q8hf(uint4 h, inout uint bad)
{
    const wg_frag r0 = wg_ags_convert(rr_w64f(h.x, h.y, 0u, 0u), WG_AGS_T_F16_ACC, WG_AGS_T_FP8_B);
    const wg_frag r1 = wg_ags_convert(rr_w64f(h.z, h.w, 0u, 0u), WG_AGS_T_F16_ACC, WG_AGS_T_FP8_B);
    const uint2 q = uint2(r0.v[0], r1.v[0]);
    bad |= ((q.x & 0x7f7f7f7fu) + 0x01010101u) | ((q.y & 0x7f7f7f7fu) + 0x01010101u);
    return q;
}
#else
uint2 q8hf(uint4 h, inout uint bad)
{
    wg_frag f;
    f.v[0] = h.x; f.v[1] = h.y; f.v[2] = h.z; f.v[3] = h.w;
    [unroll] for (uint i = 4; i < 8; ++i) f.v[i] = 0u;
    const wg_frag r = wg_ags_convert(f, WG_AGS_T_F16_ACC, WG_AGS_T_FP8_B);
    const uint2 q = uint2(r.v[0], r.v[1]);
    bad |= ((q.x & 0x7f7f7f7fu) + 0x01010101u) | ((q.y & 0x7f7f7f7fu) + 0x01010101u);
    return q;
}
#endif
// Q8 groups (several Q8 calls sharing one wave-uniform slow-path decision):
//     uint bad = 0u;  Q8_TRY <fast forms, accumulating bad>  Q8_ELSE <slow forms>  Q8_END
// Both forms give rr_q8's bits for every input, so which one runs is a speed choice only. RR_STICKY 1 (default): once a
// group of a wave needed the slow form, the wave's later groups run the slow form directly (rr_sat, wave-uniform; the
// saturating data of the stand-in model needs it everywhere); 0: every group tries the fast form first.
#ifndef RR_STICKY
#define RR_STICKY 1
#endif
static bool rr_sat;
#ifdef RR_DBG_CNT     // debug: counts of Q8 groups per path (fast ok, slow, NaN patch) added to arena dwords 4..6
static uint rr_cnt0, rr_cnt1, rr_cnt2;
#define DBG_C(k) rr_cnt##k += 1u;
#define DBG_INIT rr_cnt0 = 0u; rr_cnt1 = 0u; rr_cnt2 = 0u;
#define DBG_FLUSH if (WaveGetLaneIndex() == 0u) { wg_atomic_add(0u, 16u, rr_cnt0); wg_atomic_add(0u, 20u, rr_cnt1); \
        wg_atomic_add(0u, 24u, rr_cnt2); }
#else
#define DBG_C(k)
#define DBG_INIT
#define DBG_FLUSH
#endif
#if RR_STICKY
#define Q8_TRY [branch] if (!rr_sat) {
#define Q8_ELSE rr_sat = q8_slow(bad); if (!rr_sat) { DBG_C(0) } } [branch] if (rr_sat) { DBG_C(1)
#define Q8_END }
#else
#define Q8_TRY {
#define Q8_ELSE } [branch] if (q8_slow(bad)) {
#define Q8_END }
#endif
uint4 fence4(uint4 h) { return uint4(rr_fence_u(h.x), rr_fence_u(h.y), rr_fence_u(h.z), rr_fence_u(h.w)); }
// The slow form of a Q8 group: the clamp (v_maxmin_num, the clamp of rr_q8_prep) and the driver conversion give rr_q8's
// code for every input but a NaN (beyond +-448 and +-inf: +-448; zeros keep their sign). A wave with a NaN in the tile
// (wave-uniform branch) then replaces each NaN by +-1000 of its sign, as rr_q8_prep does (code 0x7f | sign), reusing
// the clamped values of the others: the bits of rr_q8 for every input. After that, the wave's later slow groups run
// rr_q8 itself (rr_nan, sticky like rr_sat: data with NaNs has them everywhere).
static bool rr_nan;
#ifndef RR_NANSUM
#define RR_NANSUM 1
#endif
uint2 q8s(v8f d)
{
    uint2 q;
    [branch] if (rr_nan) {
        q = rr_q8(d);
    } else {
        v8f c;
#if RR_NANSUM
        // NaN test of the group without per-value compares: s = sum of |d| is NaN exactly when a value is NaN (sums of
        // non-negative values never form inf - inf; an overflow gives +inf, not NaN)
        float s = abs(d.v[0]);
        [unroll] for (uint i0 = 1; i0 < 8; ++i0) s = s + abs(d.v[i0]);
        const bool nan = isnan(s);
#else
        bool nan = false;
        [unroll] for (uint i0 = 0; i0 < 8; ++i0) nan = nan || isnan(d.v[i0]);
#endif
        [unroll] for (uint i = 0; i < 8; ++i) c.v[i] = min(max(d.v[i], -448.0f), 448.0f);
        [branch] if (WaveActiveAnyTrue(nan)) {
            DBG_C(2)
            rr_nan = RR_STICKY != 0;
            [unroll] for (uint k = 0; k < 8; ++k)
                c.v[k] = select(isnan(d.v[k]), asfloat((asuint(d.v[k]) & 0x80000000u) | 0x447a0000u), c.v[k]);
        }
        q = wg_pack_q8_nc_ags(c);
    }
    return q;
}
uint2 q8h_slow(uint4 h) { return q8s(rr_unh(fence4(h))); }
uint2 q8h_small(uint4 h)   // values known finite and within +-448 (P)
{
    uint bad = 0u;
    return q8hf(h, bad);
}

// 51.3 step 3 (rr_attn_w2) with the same operations; the bit step as a packed 16-bit shift and one 32-bit mask:
// per half (bits(p) << 5) & 0x7FE0 (the 32-bit form of rr_common shifts no bit across the halves that survives the
// mask, so the bits are those of rr_attn_w2 for every p). The mask is opaque so that the driver keeps the 32-bit and.
uint attn_w2(uint s2, RrWc k)
{
    const half2 u = rr_h2(s2) * k.ascale;
    const half2 c = min(max(u, k.aclampn), k.aclamp);
    const half2 inner = mad(-c, c, k.ap1);
    const half2 p = mad(c, inner, k.ap0);
    const uint16_t2 pb = asuint16(p) << (uint16_t)5;
    return ((uint)pb.x | ((uint)pb.y << 16)) & (0x7FE07FE0u & ~rr_zero_bits());
}
// 51.3 step 8 (rr_act2) without the closing fence: its only consumer is q8hf.
uint act2(uint x2, RrWc c)
{
    const half2 x = rr_h2(x2);
    const half2 k = min(max(x, c.actcln), c.actcl);
    const half2 u = mad(-abs(k), c.c1, c.c2);
    const half2 v = mad(k, u, c.c3);
    return rr_b2(x * v);
}

// clean/rr-f-a14-l01 A3 (RR_SMAJOR >= 2): act2 of 4 packed pairs, stage-major (all clamps, then all u, ...): the same
// operations per value as act2 (bit-identical), issued in an order that pairs independent values.
uint4 act2_4(uint4 h, RrWc c)
{
#if RR_SMAJOR >= 2
    const half2 x[4] = {rr_h2(h.x), rr_h2(h.y), rr_h2(h.z), rr_h2(h.w)};
    half2 k[4], u[4], v[4];
    [unroll] for (uint i0 = 0; i0 < 4; ++i0) k[i0] = min(max(x[i0], c.actcln), c.actcl);
    [unroll] for (uint i1 = 0; i1 < 4; ++i1) u[i1] = mad(-abs(k[i1]), c.c1, c.c2);
    [unroll] for (uint i2 = 0; i2 < 4; ++i2) v[i2] = mad(k[i2], u[i2], c.c3);
    return uint4(rr_b2(x[0] * v[0]), rr_b2(x[1] * v[1]), rr_b2(x[2] * v[2]), rr_b2(x[3] * v[3]));
#else
    return uint4(act2(h.x, c), act2(h.y, c), act2(h.z, c), act2(h.w, c));
#endif
}

uint4 pk_mad(uint4 a, half2 b, uint4 c)   // R(a b + c) per half, fenced
{
    uint4 r;
    r.x = rr_fence_u(rr_b2(mad(rr_h2(a.x), b, rr_h2(c.x))));
    r.y = rr_fence_u(rr_b2(mad(rr_h2(a.y), b, rr_h2(c.y))));
    r.z = rr_fence_u(rr_b2(mad(rr_h2(a.z), b, rr_h2(c.z))));
    r.w = rr_fence_u(rr_b2(mad(rr_h2(a.w), b, rr_h2(c.w))));
    return r;
}
uint4 d8h(uint2 q) { return rr_h16(rr_d8(q)); }   // d8 as packed f16 (exact)


// ---- borders (51.6) ----------------------------------------------------------------------------------------------------
// rr_reflect for -(N - 1) <= v <= 2 (N - 1): with P = 2 (N - 1), u = v mod P, min(u, P - u) is -v for v < 0, v up to
// N - 1 and P - v from N to P (0 at v = P), i.e. |v| reflected once. Every read of the window blocks stays in that
// range when the level extents are >= 10 (fine coordinates -7 .. N + 6; coarse ones -4 .. M + 3 with M = N / 2 >= 5);
// r1 (launch-uniform) selects it, otherwise the general rr_reflect.
int rr_reflect1(int v, int N) { const int a = abs(v); return a >= N ? 2 * (N - 1) - a : a; }
int rr_refl(int v, int N, bool r1)
{
    int r;
    [branch] if (r1) r = rr_reflect1(v, N); else r = rr_reflect(v, N);
    return r;
}

// ---- f32 -> f16 bits, round to nearest even: one v_cvt_f16_f32 through the driver channel (op 0x1C, immediate 1);
// the bits of wg_f2h for every one of the 2^32 inputs (GPU, exhaustive). The argument is fenced so that the driver
// cannot fold the operation that produced it into the conversion (one rounding instead of two).
uint f2h_hw(float x) { return wg_ags_call(0x1cu, 0u, 1u, asuint(x) ^ rr_zero_bits(), 0u) & 0xffffu; }

// ---- RR_B32 1 (opt-in, NOT exact to variant E or 51.1; tier-B speed variant, clean/rr-f-r8-body32) --------------------
// The block's in-between elementwise maths in f32 instead of f16: the step results that today are rounded to packed f16
// (step2h: MLP hidden, MLP / output-projection accumulators, scores) stay f32, the activation, attention weight
// function, softmax normalisation and both residual adds run in f32, and Q8 takes the f32 values directly (no f16
// round / widen). Needs RR_VARL 1 and RR_Z 0. The temporal parts (hidden state, its sigmoid table, gates) are unchanged.
// rrbv = the type of such a value: v8f (RR_B32) or 8 packed f16 (uint4, the exact form).
#ifndef RR_B32
#define RR_B32 0
#endif
#if RR_B32
#if !RR_VARL || RR_Z != 0
#error RR_B32 needs RR_VARL 1 and RR_Z 0
#endif
typedef v8f rrbv;
struct RrWx { float ascale, aclamp, aclampn, ap1, ap0, actcl, actcln, c1, c2, c3; };
RrWx rr_wx(RrWc c)
{
    RrWx f;
    f.ascale = (float)c.ascale.x; f.aclamp = (float)c.aclamp.x; f.aclampn = (float)c.aclampn.x; f.ap1 = (float)c.ap1.x;
    f.ap0 = (float)c.ap0.x; f.actcl = (float)c.actcl.x; f.actcln = (float)c.actcln.x; f.c1 = (float)c.c1.x;
    f.c2 = (float)c.c2.x; f.c3 = (float)c.c3.x;
    return f;
}
rrbv bv_step(uint2 a0, uint2 b0, uint2 a1, uint2 b1, v8f c) { return step2_raw(a0, b0, a1, b1, c); }
v8f bv_un(rrbv x) { return x; }
rrbv bv_of(v8f x) { return x; }
uint2 bv_q8f(rrbv x, inout uint bad) { return q8f(x, bad); }
uint2 bv_q8s(rrbv x) { return q8s(x); }
rrbv bv_d8(uint2 q) { return rr_d8(q); }
rrbv bv_mad(rrbv a, float b, rrbv c)
{
    v8f r;
    [unroll] for (uint i = 0; i < 8; ++i) r.v[i] = mad(a.v[i], b, c.v[i]);
    return r;
}
// 51.3 step 8 in f32 (same formula, no f16 roundings)
rrbv bv_act(rrbv x, RrWx c)
{
    v8f r;
#if RR_SMAJOR
    v8f k, u, v;
    [unroll] for (uint i0 = 0; i0 < 8; ++i0) k.v[i0] = min(max(x.v[i0], c.actcln), c.actcl);
    [unroll] for (uint i1 = 0; i1 < 8; ++i1) u.v[i1] = mad(-abs(k.v[i1]), c.c1, c.c2);
    [unroll] for (uint i2 = 0; i2 < 8; ++i2) v.v[i2] = mad(k.v[i2], u.v[i2], c.c3);
    [unroll] for (uint i3 = 0; i3 < 8; ++i3) r.v[i3] = x.v[i3] * v.v[i3];
#else
    [unroll] for (uint i = 0; i < 8; ++i) {
        const float k = min(max(x.v[i], c.actcln), c.actcl);
        const float u = mad(-abs(k), c.c1, c.c2);
        const float v = mad(k, u, c.c3);
        r.v[i] = x.v[i] * v;
    }
#endif
    return r;
}
// 51.3 step 3 in f32: p as in rr_attn_w2 without the f16 roundings; the bit step on the f32 bits of p in
// [1 + 1/32, 2): w = 2^(m_hi - 15) (1 + m_lo) with m_hi the top 5 mantissa bits of p and m_lo the other 18 (the f16
// form keeps 5 of them): exponent field m_hi + 112, mantissa m_lo << 5. Equal to rr_attn_w2 where p is an f16 value.
// One v_lshl_add_u32: p's exponent bits shifted to bits 31..28 are 0xF (p in [1, 2)), 0xF0000000 + 0x48000000 wraps
// to the 0x38000000 bias.
rrbv bv_attn(rrbv s, RrWx k)
{
    v8f r;
#if RR_SMAJOR
    v8f u, c, inner, p;
    [unroll] for (uint i0 = 0; i0 < 8; ++i0) u.v[i0] = s.v[i0] * k.ascale;
    [unroll] for (uint i1 = 0; i1 < 8; ++i1) c.v[i1] = min(max(u.v[i1], k.aclampn), k.aclamp);
    [unroll] for (uint i2 = 0; i2 < 8; ++i2) inner.v[i2] = mad(-c.v[i2], c.v[i2], k.ap1);
    [unroll] for (uint i3 = 0; i3 < 8; ++i3) p.v[i3] = mad(c.v[i3], inner.v[i3], k.ap0);
    [unroll] for (uint i4 = 0; i4 < 8; ++i4) r.v[i4] = asfloat((asuint(p.v[i4]) << 5) + 0x48000000u);
#else
    [unroll] for (uint i = 0; i < 8; ++i) {
        const float u = s.v[i] * k.ascale;
        const float c = min(max(u, k.aclampn), k.aclamp);
        const float inner = mad(-c, c, k.ap1);
        const float p = mad(c, inner, k.ap0);
        r.v[i] = asfloat((asuint(p) << 5) + 0x48000000u);   // = ((bits << 5) & 0x0FFFFFE0) + 0x38000000 for p in [1, 2)
    }
#endif
    return r;
}
#define BVW wx
#else
// the exact form: macros that expand to the original text (the preprocessed source, hence the build cache key and the
// DXIL, of every RR_B32 0 kernel stays that of clean/rr-f-speed7)
#define rrbv uint4
#define RrWx RrWc
#define BVW wc
#define bv_step step2h
#define bv_un rr_unh
#define bv_of rr_h16
#define bv_q8f q8hf
#define bv_q8s q8h_slow
#define bv_d8 d8h
#define bv_mad pk_mad
#define bv_act(V_, C_) uint4(act2(V_.x, C_), act2(V_.y, C_), act2(V_.z, C_), act2(V_.w, C_))
#define bv_attn(V_, C_) uint4(attn_w2(V_.x, C_), attn_w2(V_.y, C_), attn_w2(V_.z, C_), attn_w2(V_.w, C_))
#endif
