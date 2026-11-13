// SPDX-License-Identifier: MIT
// Page 54b.5 quarter pool (PH2) through two exact tables (speed form of rr_resolve.hlsl rr_resolve_pool, same values
// bit for bit; the tables are built on the GPU with the same exact elementary functions, rr_resolve_math.hlsli).
//
// The pool of one PH2 texel: for each channel, acc = F(term(code_t), K12, acc) over the valid PH1 taps t (in the
// order (0,0) (1,0) (0,1) (1,1)), mean = M(RCP(count), acc), value = M(M(M(L(pmax(F(mean, K4, 1), K5)), K6), K7), K11),
// stored as the RGB10A2 code unorm(value) (alpha code 3).
//   * term(code) = A(X(M(M(clamp(M(code / 1023, C_HDEC)), K8), K9)), -1) depends on the 10-bit code only: table TERM.
//   * code(mean) is a monotone step function of the f32 mean (every operation of the chain is monotone in its
//     variable operand; FTZ and the plain max keep that): with the f32 values ordered by key(), code(mean) is the
//     number of thresholds THR[1..1023] at or below key(mean) (or 1023 minus that number for a decreasing chain);
//     a NaN mean gives the code CNAN. rr_pool_check compares this with the direct chain for all 2^32 f32 patterns
//     (gpu/pool.py --check); a model whose chain is not monotone fails that check and keeps the direct kernel.
// Table block at `tab` (u32 words): [0, 1024) TERM (f32), [1024, 2048) THR (keys; THR[0] = 0), 2048 increasing flag,
// 2049 CNAN.
// Parameter block: rr_resolve.hlsl's (gpu/resolve.py PARAMS): PH1 at 160, PQ at 168 (format 4: four uint16 codes),
// K4 25, K5 26, K7 27, K8 28, K10 29, K11 30, K12 31, C_HDEC 33.
#include "wg_platform.hlsli"
WG_ARGS_BEGIN
 uint prm, tab, n, pout;
WG_ARGS_END
#include "rr_resolve_math.hlsli"

#ifndef RR_POOL_TPL
#define RR_POOL_TPL 4          // PH2 texels per lane (the tables are staged once per workgroup)
#endif

uint PU(uint i) { return WaveReadLaneFirst(wg_ld32(prm, i * 4u)); }
float PF(uint i) { return asfloat(PU(i)); }

struct Res { uint off, w, h, pitch, fmt, ch; int ox, oy; };
Res res(uint b)
{
    Res r;
    r.off = PU(b); r.w = PU(b + 1u); r.h = PU(b + 2u); r.pitch = PU(b + 3u);
    r.fmt = PU(b + 4u); r.ch = PU(b + 5u); r.ox = (int)PU(b + 6u); r.oy = (int)PU(b + 7u);
    return r;
}
float cv(uint code, uint maxc) { return wg_div((float)code, (float)maxc); }   // as rr_resolve.hlsl
uint unorm_code(float v, uint maxc)                                            // as rr_resolve.hlsl
{
    if (nanf(v)) return 0u;
    double s = (double)min(max(v, 0.0f), 1.0f) * (double)maxc;   // exact
    uint a = (uint)s; double f = s - (double)a;
    return min(maxc, a + ((f > 0.5 || (f == 0.5 && (a & 1u) != 0u)) ? 1u : 0u));
}

// the two halves of rr_resolve_pool's per-channel arithmetic, written as there
float pool_term(uint code)
{
    const float K5 = PF(26), K10 = PF(29);
    float cc = M(cv(code, 1023u), PF(33));
    float tt = (cc > K5 || nanf(cc)) ? min_nan(cc, K10) : K5;
    return A(X(M(M(tt, PF(28)), RR_K9)), -1.0f);
}
uint pool_code(float mean)
{
    const float K5 = PF(26);
    return unorm_code(M(M(M(L(pmax(F(mean, PF(25), 1.0f), K5)), RR_K6), PF(27)), PF(30)), 1023u);
}

// total order of the non-NaN f32 values (-0 directly below +0)
uint fkey(float f) { const uint u = asuint(f); return (u >> 31) != 0u ? ~u : (u | 0x80000000u); }
float unkey(uint k) { return asfloat((k >> 31) != 0u ? (k & 0x7fffffffu) : ~k); }
#define KEY_NINF 0x007fffffu
#define KEY_PINF 0xff800000u

// ---- table build: 1024 lanes (4 workgroups of 256) ------------------------------------------------------------------
WG_KERNEL(rr_pool_tables)
{
    WG_SETUP();
    const uint i = wg_block_id() * WG_BLOCK + wg_lane_id();
    if (i >= 1024u) return;
    wg_st32(tab, i * 4u, asuint(pool_term(i)));
    const bool inc = pool_code(unkey(KEY_PINF)) >= pool_code(unkey(KEY_NINF));
    uint T = 0u;
    if (i > 0u) {
        // smallest key with s(key) >= i, s = code (increasing) or 1023 - code (decreasing)
        uint lo = KEY_NINF, hi = KEY_PINF;
        const uint slo = inc ? pool_code(unkey(lo)) : 1023u - pool_code(unkey(lo));
        const uint shi = inc ? pool_code(unkey(hi)) : 1023u - pool_code(unkey(hi));
        if (slo >= i) T = 0u;
        else if (shi < i) T = 0xffffffffu;
        else {
            [loop] for (uint it = 0; it < 40u && hi - lo > 1u; ++it) {
                const uint mid = lo + ((hi - lo) >> 1);
                const uint c = pool_code(unkey(mid));
                const uint s = inc ? c : 1023u - c;
                if (s >= i) hi = mid; else lo = mid;
            }
            T = hi;
        }
    }
    wg_st32(tab, (1024u + i) * 4u, T);
    if (i == 0u) {
        wg_st32(tab, 2048u * 4u, inc ? 1u : 0u);
        wg_st32(tab, 2049u * 4u, pool_code(asfloat(0x7fc00000u)));
    }
}

// ---- the pool --------------------------------------------------------------------------------------------------------
groupshared float gs_term[1024];
groupshared uint gs_thr[1024];

uint table_code(float mean, bool inc, uint cnan)
{
    const uint k = fkey(mean);
    uint c = 0u;
    [unroll] for (uint st = 512u; st >= 1u; st >>= 1) c += gs_thr[c + st] <= k ? st : 0u;
    c = inc ? c : 1023u - c;
    return nanf(mean) ? cnan : c;
}

WG_KERNEL(rr_pool_main)
{
    WG_SETUP();
    for (uint j = wg_lane_id(); j < 1024u; j += WG_BLOCK) {
        gs_term[j] = asfloat(wg_ld32(tab, j * 4u));
        gs_thr[j] = wg_ld32(tab, (1024u + j) * 4u);
    }
    GroupMemoryBarrierWithGroupSync();
    const bool inc = WaveReadLaneFirst(wg_ld32(tab, 2048u * 4u)) != 0u;
    const uint cnan = WaveReadLaneFirst(wg_ld32(tab, 2049u * 4u));
    const Res ph = res(160u), pq = res(168u);
    const uint w2 = (ph.w + 1u) / 2u, h2 = (ph.h + 1u) / 2u;
    const float K12 = PF(31);
    [unroll] for (uint r = 0; r < RR_POOL_TPL; ++r) {
        const uint idx = (wg_block_id() * RR_POOL_TPL + r) * WG_BLOCK + wg_lane_id();
        if (idx >= w2 * h2) break;
        const int x = (int)(idx % w2), y = (int)(idx / w2);
        float3 acc = 0.0f;
        uint cnt = 0u;
        [unroll] for (uint t = 0; t < 4u; ++t) {
            const int tx = 2 * x + (int)(t & 1u), ty = 2 * y + (int)(t >> 1);
            if (tx < (int)ph.w && ty < (int)ph.h) {
                const uint2 cw = wg_ld64(ph.off, ((uint)ty * ph.pitch + (uint)tx) * 8u);
                acc.x = F(gs_term[cw.x & 1023u], K12, acc.x);
                acc.y = F(gs_term[(cw.x >> 16) & 1023u], K12, acc.y);
                acc.z = F(gs_term[cw.y & 1023u], K12, acc.z);
                cnt += 1u;
            }
        }
        // RCP(count): the valid taps are 1 or 2 per axis, so count is 1, 2 or 4 and its reciprocal is exact
        const float rn = cnt == 4u ? 0.25f : (cnt == 2u ? 0.5f : 1.0f);
        const uint c0 = table_code(M(rn, acc.x), inc, cnan), c1 = table_code(M(rn, acc.y), inc, cnan);
        const uint c2 = table_code(M(rn, acc.z), inc, cnan);
        if (x < (int)pq.w && y < (int)pq.h)
            wg_st64(pq.off, ((uint)y * pq.pitch + (uint)x) * 8u, uint2(c0 | (c1 << 16), c2 | (3u << 16)));
    }
}

// ---- exhaustive check of the code table: lane = f32 pattern n + i (all 2^32 over 256 dispatches of 2^24) -------------
// pout: [0] patterns whose table code differs from the direct chain, [1] waves checked (32 patterns each), [2] largest
// differing pattern + 1 (0: none). The tables must be staged (one workgroup = 256 patterns).
WG_KERNEL(rr_pool_check)
{
    WG_SETUP();
    for (uint j = wg_lane_id(); j < 1024u; j += WG_BLOCK) gs_thr[j] = wg_ld32(tab, (1024u + j) * 4u);
    GroupMemoryBarrierWithGroupSync();
    const bool inc = WaveReadLaneFirst(wg_ld32(tab, 2048u * 4u)) != 0u;
    const uint cnan = WaveReadLaneFirst(wg_ld32(tab, 2049u * 4u));
    const uint u = n + wg_block_id() * WG_BLOCK + wg_lane_id();
    const float mean = asfloat(u);
    const uint bad = table_code(mean, inc, cnan) != pool_code(mean) ? 1u : 0u;
    const uint sb = WaveActiveSum(bad), lm = WaveActiveMax(bad != 0u ? u + 1u : 0u);
    if (WaveIsFirstLane()) {
        if (sb != 0u) wg_atomic_add(pout, 0u, sb);
        wg_atomic_add(pout, 4u, 1u);
        if (lm != 0u) wg_atomic_max(pout, 8u, lm);
    }
}
