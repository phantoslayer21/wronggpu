// SPDX-License-Identifier: MIT
// Page 54b on D3D12, fast resolve (rr_resolve_fast): the extended resolve of rr_resolve.hlsl (rr_resolve_main) with
// the same bits for every input, at a fraction of the cost. Own implementation; design and proofs:
// clean/rr_f/gpu/RESOLVE-FAST.md. Also: probes of the hardware operations it relies on and the exhaustive checks of
// its helpers (gpu/resolve_fast.py).
#include "rr_resolve.hlsl"     // platform, arguments (prm, aux, n), the verified kernel's helpers and kernels
#if WG_WAVE == 64
#include "wg_ags.hlsli"         // the driver channel (wg_platform.hlsli includes it for wave32 only)
#endif

// The exp2 table T[j] = 2^(j/1024) (j < 1024) as (hi, lo) f32 pairs lives in the arena (filled once by
// rr_rf_init_table); the kernels read it from there (RF_TABLE_ARENA: the fixup kernel) or from a groupshared copy.
// clean/rr-f-s10-tail: the table is read from the arena copy by default (the same words, so the same bits): no 8 KB
// groupshared copy and barrier per workgroup (LDS 28.3 -> 20.1 KB for RF_SPEC 2: 4 -> 6 groups per WGP). With the
// hardware exp2 (RF_HW 1) the fast kernels never read it; with the exact exp2 (exact mode) the memory reads measured
// faster than the groupshared copy (resolve 0.65 -> 0.58 ms at 1440p Q). RF_TABLE_ARENA=0 restores the copy.
#ifndef RF_TABLE_ARENA
#define RF_TABLE_ARENA 1
#endif
static uint rf_toff;                 // arena offset of the table
groupshared uint2 rf_t[1024];
#if RF_TABLE_ARENA
#define RF_TGET(j) wg_ld64(rf_toff, (j) * 8u)
#else
#define RF_TGET(j) rf_t[(j)]
#endif
#include "rr_resolve_fast_math.hlsli"
#if RR_ELEM_EXACT == 2
float rr_x_hook(float x)             // EXP2_CR for every input: the fast function, its rare undecided inputs exactly
{
#if RF_TABLE_ARENA
    if (rf_toff == 0u) { bool h0; return EXP2_CR(x, 0u, h0); }     // no table attached: the verified function
#endif
    bool slow = false;
    float v = rf_exp2(x, slow);
    if (slow) { bool h; v = EXP2_CR(x, 0u, h); }
    return v;
}
float rr_rcp_hook(float x) { return rf_rcp(x); }
#endif
void rf_load_table(uint toff)
{
    rf_toff = toff;
#if !RF_TABLE_ARENA
    [unroll] for (uint k = 0; k < 1024u / WG_BLOCK; ++k) {
        uint j = wg_lane_id() + k * WG_BLOCK;
        rf_t[j] = wg_ld64(toff, j * 8u);
    }
    GroupMemoryBarrierWithGroupSync();
#endif
}

// T[j] = 2^(j/1024), j < 1024: hi = the f32 nearest to 2^(j/1024), lo = the f32 nearest to 2^(j/1024) - hi, from the
// verified kernel's double-double exp2 (error below 2^-100) and its correct rounding round_dd. Run once (1024 lanes)
// into the 8 KB at prm; equal to gpu/gen_rf_tables.py's exact values (resolve_fast.py probes).
WG_KERNEL(rr_rf_init_table)
{
    WG_SETUP();
    uint j = wg_block_id() * WG_BLOCK + wg_lane_id();
    if (j >= 1024u) return;
    rr_dd V = exp2_dd((double)j * (1.0 / 1024.0));
    float hi = round_dd(V);
    double rest = (V.h - (double)hi) + V.l;             // V.h - hi exact (Sterbenz); |rest| < 2^-24
    float lo = (float)rest;
    wg_st64(prm, j * 8u, uint2(asuint(hi), asuint(lo)));
}

#include "rr_resolve_fast_main.hlsli"

// ---- probes of the hardware operations the fast path relies on ----------------------------------------------------
// f32 pairs (a, b) at prm (n records of two words); results after them, 4 words per record:
// min(a, b), max(a, b) (HLSL min / max), pmin(a, b), pmax(a, b) (rr_resolve_math.hlsli).
WG_KERNEL(rr_rf_probe_minmax)
{
    WG_SETUP();
    uint i = wg_block_id() * WG_BLOCK + wg_lane_id();
    if (i >= n) return;
    float a = asfloat(wg_ld32(prm, i * 8u)), b = asfloat(wg_ld32(prm, i * 8u + 4u));
    wg_st128(prm, n * 8u + i * 16u, uint4(asuint(min(a, b)), asuint(max(a, b)), asuint(pmin(a, b)), asuint(pmax(a, b))));
}

// Exhaustive f16 pairs: lane i covers a = i >> 16 ... (aux = first a code, n = count of a codes, all 65536 b codes):
// counts at prm: [0] half min != pmin, [1] half max != pmax, [2] packed min != pmin, [3] packed max != pmax,
// then up to 64 (a, b, which) records from word 8. NaN results compare as one class.
uint hclass(uint h) { return (h & 0x7fffu) > 0x7c00u ? 0x7e00u : h; }
void hrec(uint a, uint b, uint which)
{
    uint old, slot;
    wg_mem.InterlockedAdd(prm + which * 4u, 1u, old);
    wg_mem.InterlockedAdd(prm + 28u, 1u, slot);
    if (slot < 64u) wg_st32(prm, 32u + slot * 4u, (a << 16) | b | (which << 30));
}
WG_KERNEL(rr_rf_probe_h16)
{
    WG_SETUP();
    uint i = wg_block_id() * WG_BLOCK + wg_lane_id();
    if (i >= n * 256u) return;
    uint a = aux + i / 256u;
    float fa = f16tof32(a);
    half ha = asfloat16((uint16_t)a);
    [loop] for (uint k = 0; k < 256u; ++k) {
        uint b = (i % 256u) * 256u + k;
        float fb = f16tof32(b);
        half hb = asfloat16((uint16_t)b);
        uint wmin = hclass(wg_f2h(pmin(fa, fb))), wmax = hclass(wg_f2h(pmax(fa, fb)));
        uint gmin = hclass((uint)asuint16(min(ha, hb))), gmax = hclass((uint)asuint16(max(ha, hb)));
        half2 pa = half2(ha, hb), pb = half2(hb, ha);
        half2 pm = min(pa, pb), px = max(pa, pb);
        uint pmn = hclass((uint)asuint16(pm.x)), pmx = hclass((uint)asuint16(px.x));
        if (gmin != wmin) hrec(a, b, 0u);
        if (gmax != wmax) hrec(a, b, 1u);
        if (pmn != wmin) hrec(a, b, 2u);
        if (pmx != wmax) hrec(a, b, 3u);
    }
}

// fma16: triples of f16 codes (low halves of three words) at prm; results after them: [mad(half), wg_fma_f16,
// packed mad lane x] as three words per record.
WG_KERNEL(rr_rf_probe_fma16)
{
    WG_SETUP();
    uint i = wg_block_id() * WG_BLOCK + wg_lane_id();
    if (i >= n) return;
    uint a = wg_ld32(prm, i * 12u) & 0xffffu, b = wg_ld32(prm, i * 12u + 4u) & 0xffffu, c = wg_ld32(prm, i * 12u + 8u) & 0xffffu;
    half ha = asfloat16((uint16_t)a), hb = asfloat16((uint16_t)b), hc = asfloat16((uint16_t)c);
    half r = mad(ha, hb, hc);
    half2 rp = mad(half2(ha, hb), half2(hb, ha), half2(hc, hc));
    uint w = wg_fma_f16(f16tof32(a), f16tof32(b), f16tof32(c));
    wg_st32(prm, n * 12u + i * 12u, (uint)asuint16(r));
    wg_st32(prm, n * 12u + i * 12u + 4u, w);
    wg_st32(prm, n * 12u + i * 12u + 8u, (uint)asuint16(rp.x));
}

// ---- exhaustive checks of rr_resolve_fast_math.hlsli against rr_resolve_math.hlsli --------------------------------
bool same32(uint a, uint b) { return a == b || (nanf(asfloat(a)) && nanf(asfloat(b))); }
// prm: counters [0] mismatches (fast decided, bits differ), [1] undecided (hard), [2] checked, [3] list slot,
// [6] the exp2 table offset; list of (input, fast, reference) from word 8 (capacity 256). Inputs aux + i, i < n, 64 per
// lane.
#ifndef RF_EXH_FUNC
#define RF_EXH_FUNC 0      // 0 exp2, 1 rcp
#endif
WG_KERNEL(rr_rf_exhaustive)
{
    WG_SETUP();
    rf_load_table(wg_ld32(prm, 24u));                  // counter word 6: the table offset
    uint mism = 0u, hardc = 0u, cnt = 0u;
    uint base = (wg_block_id() * WG_BLOCK + wg_lane_id()) * 64u;
    [loop] for (uint k = 0; k < 64u; ++k) {
        uint i = base + k;
        if (i >= n) break;
        uint bits = aux + i;
        float x = asfloat(bits);
        bool slow = false;
        uint fb, rb;
#if RF_EXH_FUNC == 0
        fb = asuint(rf_exp2(x, slow));
        bool h;
        rb = asuint(EXP2_CR(x, 0u, h));
#elif RF_EXH_FUNC == 1
        fb = asuint(rf_rcp(x));
        rb = asuint(RCP(x));
#elif RF_EXH_FUNC == 2
        fb = rf_f2h(x);
        rb = wg_f2h(x);
        fb = (fb & 0x7fffu) > 0x7c00u ? 0x7fc00000u : fb;   // binary16 NaN class
        rb = (rb & 0x7fffu) > 0x7c00u ? 0x7fc00000u : rb;
#elif RF_EXH_FUNC == 8
        // the hardware log2 (v_log_f32 on the flushed input) against LOG2_CR: its largest distance in f32 steps for
        // finite results of one sign (counter word 5); "mismatch" = any other difference (special results)
        bool h;
        float lh = fence(log2(ftz(x)));
        rb = asuint(LOG2_CR(x, 0u, h));
        fb = asuint(lh);
        if (isfinite(lh) && isfinite(asfloat(rb)) && ((fb ^ rb) & 0x80000000u) == 0u) {
            uint dd = fb > rb ? fb - rb : rb - fb, o8;
            wg_mem.InterlockedMax(prm + 20u, dd, o8);
            fb = rb;                     // counted by its distance, not as a mismatch
        } else if (asfloat(fb) == 0.0f && asfloat(rb) == 0.0f) {
            fb = rb;
        }
#elif RF_EXH_FUNC == 6
        fb = asuint(X(x));               // build with RR_ELEM_EXACT=2, RF_TABLE_ARENA=1: rr_x_hook
        bool h;
        rb = asuint(EXP2_CR(x, 0u, h));
#else
        uint mc = RF_EXH_FUNC == 3 ? 255u : (RF_EXH_FUNC == 4 ? 1023u : 3u);
        fb = rf_unorm_code(x, mc);
        rb = unorm_code(x, mc);
#endif
        cnt += 1u;
        if (slow) { hardc += 1u; continue; }
        if (!same32(fb, rb)) {
            mism += 1u;
            uint slot, old;
            wg_mem.InterlockedAdd(prm + 12u, 1u, slot);
            if (slot < 256u) { wg_st32(prm, 32u + slot * 12u, bits); wg_st32(prm, 36u + slot * 12u, fb); wg_st32(prm, 40u + slot * 12u, rb); }
        }
    }
    uint old;
    mism = WaveActiveSum(mism); hardc = WaveActiveSum(hardc); cnt = WaveActiveSum(cnt);
    if (WaveIsFirstLane()) {
        wg_mem.InterlockedAdd(prm, mism, old); wg_mem.InterlockedAdd(prm + 4u, hardc, old); wg_mem.InterlockedAdd(prm + 8u, cnt, old);
    }
}

// rf_accx vs F (rr_resolve_math.hlsli) where the hardware fma and F can differ: for each v (prm word 16 + 2k: value
// bits, word 17 + 2k: the bits of FMINN / v) every normal w within aux f32 steps of FMINN / v, with acc = 0, FMINN and
// FMINN + 1 step. Counters at prm: [0] rf_accx != F, [1] hardware fma != F (the cases rf_accx repairs), [2] checked,
// [3] list slot; list of (v, w, acc index | fast << 2 ...) from word 8 as (v, w, rf_accx, F) records (capacity 2).
WG_KERNEL(rr_rf_probe_accx)
{
    WG_SETUP();
    uint i = wg_block_id() * WG_BLOCK + wg_lane_id();
    uint D = aux;
    uint mism = 0u, hwd = 0u, cnt = 0u;
    if (i < n * 2u * D) {
        uint vi = i / (2u * D);
        float v = asfloat(wg_ld32(prm, 64u + vi * 8u));
        uint wb = wg_ld32(prm, 68u + vi * 8u) + (i % (2u * D)) - D;
        float w = asfloat(wb);
        if (wb >= 0x00800000u && wb < 0x7f800000u) {
            [unroll] for (uint k = 0; k < 3u; ++k) {
                float acc = asfloat(k == 0u ? 0u : (k == 1u ? 0x00800000u : 0x00800001u));
                uint f = asuint(rf_accx(w, v, acc)), r = asuint(F(w, v, acc)), h = asuint(fence(mad(w, v, acc)));
                cnt += 1u;
                hwd += h != r ? 1u : 0u;
                if (f != r) {
                    mism += 1u;
                    uint slot;
                    wg_mem.InterlockedAdd(prm + 12u, 1u, slot);
                    if (slot < 2u) { wg_st32(prm, 32u + slot * 16u, asuint(v)); wg_st32(prm, 36u + slot * 16u, wb);
                                     wg_st32(prm, 40u + slot * 16u, f); wg_st32(prm, 44u + slot * 16u, r); }
                }
            }
        }
    }
    uint old;
    mism = WaveActiveSum(mism); hwd = WaveActiveSum(hwd); cnt = WaveActiveSum(cnt);
    if (WaveIsFirstLane()) {
        wg_mem.InterlockedAdd(prm, mism, old); wg_mem.InterlockedAdd(prm + 4u, hwd, old); wg_mem.InterlockedAdd(prm + 8u, cnt, old);
    }
}

// rf_cv vs cv (the verified IEEE division) for every code 0..maxc of maxc = 255, 1023, 3: lane i -> code i % 1024,
// maxc index i / 1024; two result words per lane at prm (fast, reference); codes above maxc store 0, 0.
WG_KERNEL(rr_rf_probe_cv)
{
    WG_SETUP();
    uint i = wg_block_id() * WG_BLOCK + wg_lane_id();
    if (i >= 3u * 1024u) return;
    uint code = i % 1024u, k = i / 1024u;
    uint mc = k == 0u ? 255u : (k == 1u ? 1023u : 3u);
    float ri = k == 0u ? RF_R255 : (k == 1u ? RF_R1023 : RF_R3);
    uint f = 0u, r = 0u;
    if (code <= mc) { f = asuint(rf_cv(code, (float)mc, ri)); r = asuint(cv(code, mc)); }
    wg_st32(prm, i * 8u, f);
    wg_st32(prm, i * 8u + 4u, r);
}
