// SPDX-License-Identifier: MIT
// Probes of the two hardware facts the exact matrix step Y (rr_step_y.hlsli) is built on (gpu/probe_y.py):
//   rr_py_wmma : H1, v_wmma_f32_16x16x16_fp8_fp8 is exact in the "integer regime": D1 = A1 B1 + C, D2 = A2 B2 + D1
//                with every product and C a multiple of 2^g and the sum of all magnitudes below 2^(g + 24).
//   rr_py_mix  : H2, an f32 add feeding the driver's f16 matrix conversion compiles to v_fma_mixlo/hi_f16 x, 1.0, y
//                (one rounding of the exact x + y to f16, nearest even): the ISA is checked by the harness, the values
//                against the exact rounding.
#include "wg_platform.hlsli"
WG_ARGS_BEGIN
    uint pIn, pOut, n;
WG_ARGS_END
#include "rr_common.hlsli"

// H1. Per lane in: A1, B1, A2, B2 (8 B each), C (8 f32). Out: D1 (8 f32), D2 (8 f32).
WG_KERNEL(rr_py_wmma)
{
    WG_SETUP();
    const uint i = wg_block_id() * WG_BLOCK + wg_lane_id();
    const uint ib = pIn + i * 64u;
    const uint4 ab = wg_ld128(ib, 0u), ab2 = wg_ld128(ib, 16u);
    const v8f c = wg_load8f(ib, 32u);
    const v8f d1 = wg_wmma_fp8(ab.xy, ab.zw, c);
    const v8f d2 = wg_wmma_fp8(ab2.xy, ab2.zw, d1);
    const uint ob = pOut + i * 64u;
    [unroll] for (uint k = 0; k < 8; ++k) {
        wg_st32(ob, 4u * k, asuint(d1.v[k]));
        wg_st32(ob, 32u + 4u * k, asuint(d2.v[k]));
    }
}

// H2. Per lane in: x (8 f32), y (8 f32). Out: f16 bits of x + y (4 dwords), through rr_h16 (the step's rounding).
WG_KERNEL(rr_py_mix)
{
    WG_SETUP();
    const uint i = wg_block_id() * WG_BLOCK + wg_lane_id();
    const v8f x = wg_load8f(pIn, i * 64u), y = wg_load8f(pIn, i * 64u + 32u);
    v8f s;
    [unroll] for (uint k = 0; k < 8; ++k) s.v[k] = x.v[k] + y.v[k];
    wg_st128(pOut, i * 16u, rr_h16(s));
}

// Step tests of rr_step_y.hlsli against the exact step (gpu/probe_y.py --steps). Per lane in: A0, A1, B0, B1 (8 B each),
// C (8 f32); out: packed f16 (4 dwords) then the flag (1 dword). RR_PY_MODE 0: y_step0 (C unused), 1: y_step,
// 2: y_step0t (A split), 3: y_stept, 4: y_score (both split), 5: z_fast (one chain, R(c + M)), 6: z_fast0.
#define RR_STEP 3
#define RR_LDSW 1
#ifdef RR_PY_Z2
#define RR_ZFAST 2       // variant Z2: Y2 steps (bound 2^(g + 15) = the n argument), range gates, two-chain scores
#else
#define RR_ZFAST 1
#endif
static uint rr_hf;
#include "rr_step_y.hlsli"
#ifndef RR_PY_MODE
#define RR_PY_MODE 0
#endif
WG_KERNEL(rr_py_step)
{
    WG_SETUP();
    rr_yflag = 0u;
    rr_ymax = 0.0f;
    rr_hf = WaveGetLaneIndex() >> 4;
    const uint i = wg_block_id() * WG_BLOCK + wg_lane_id();
    const uint ib = pIn + i * 64u;
    const uint4 ab = wg_ld128(ib, 0u), bb = wg_ld128(ib, 16u);   // A0 A1 | B0 B1
    const v8f c = wg_load8f(ib, 32u);
    const float thr = asfloat(n);
#ifdef RR_PY_Z2
    rr_yb = thr;
#endif
    uint4 h;
#if RR_PY_MODE == 0
    h = y_step0(ab.xy, rr_split(bb.xy), ab.zw, rr_split(bb.zw));
#elif RR_PY_MODE == 1
#ifdef RR_PY_Z2
    h = y_step(ab.xy, rr_split(bb.xy), ab.zw, rr_split(bb.zw), rr_unh(rr_h16(c)));   // c: a widened f16 value
#else
    h = y_step(ab.xy, rr_split(bb.xy), ab.zw, rr_split(bb.zw), c);
#endif
#elif RR_PY_MODE == 2
    h = y_step0t(rr_split(ab.xy), bb.xy, rr_split(ab.zw), bb.zw);
#elif RR_PY_MODE == 3
#ifdef RR_PY_Z2
    h = y_stept(rr_split(ab.xy), bb.xy, rr_split(ab.zw), bb.zw, rr_unh(rr_h16(c)));
#else
    h = y_stept(rr_split(ab.xy), bb.xy, rr_split(ab.zw), bb.zw, c);
#endif
#elif RR_PY_MODE == 5
    h = z_fast(ab.xy, bb.xy, ab.zw, bb.zw, rr_unh(rr_h16(c)));   // c as the kernels hold it: a widened f16 value
#elif RR_PY_MODE == 6
    h = z_fast0(ab.xy, bb.xy, ab.zw, bb.zw);
#elif RR_PY_MODE == 7 || RR_PY_MODE == 8
    {   // scores of variant Z2 (A = k unsplit, B = q): 8 one chain behind its gate, 7 two chains behind zs2_ok
        const uint sq = zr_wave(zr_acc(zr_acc(0u, bb.xy), bb.zw)), sk = zr_wave(zr_acc(zr_acc(0u, ab.xy), ab.zw));
        const float bk = zr_bm(sk);
        bool ok = false;
        h = uint4(0u, 0u, 0u, 0u);
#if RR_PY_MODE == 8
        [branch] if (zr_bm(sq) * bk < 524288.0f) { h = z_fast(ab.xy, bb.xy, ab.zw, bb.zw, c); ok = true; }
#else
        [branch] if (zs2_ok(sq, bk)) {
            const uint ek = max((0x80u - ((sk >> 8) & 0xffu)) >> 3, 1u), eq = max((0x80u - ((sq >> 8) & 0xffu)) >> 3, 1u);
            h = s2_score(ab.xy, rr_split(bb.xy), ab.zw, rr_split(bb.zw), c, ldexp(1.0f, (float)(ek + eq + 4u)), ok);
        }
#endif
        rr_yflag |= ok ? 0u : 2u;
    }
#elif RR_PY_MODE == 9
    {   // the range gate of variant Z2 (A = the bounded operand of a class with R = the n argument's low byte)
        const uint sb = zr_wave(zr_acc(zr_acc(0u, bb.xy), bb.zw));
        h = uint4(0u, 0u, 0u, 0u);
        [branch] if (zr_pass(sb, n & 0xffu)) h = z_fast(ab.xy, bb.xy, ab.zw, bb.zw, rr_unh(rr_h16(c)));
        else rr_yflag |= 2u;
    }
#else
    v8f ca, cb;                                                      // the host's split of the score bias
    [unroll] for (uint k = 0; k < 8; ++k) {
        ca.v[k] = round(c.v[k]);
        cb.v[k] = c.v[k] - ca.v[k];
    }
    h = y_score(rr_split(ab.xy), rr_split(bb.xy), rr_split(ab.zw), rr_split(bb.zw), ca, cb);
#endif
#ifndef RR_PY_Z2
    rr_yflag |= rr_ymax < thr ? 0u : 1u;
#endif
    wg_st128(pOut, i * 20u, h);
    wg_st32(pOut, i * 20u + 16u, rr_yflag);
}
