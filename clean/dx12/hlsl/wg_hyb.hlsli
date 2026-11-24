// SPDX-License-Identifier: MIT
// Element-wise steps of Model L's / Model M's HYBRID block form (WG_L_FORM=hybrid; the block templates with SW_HYB=1):
// the original's FP16 forms (PRESET-L-P3-Q7, clean/l/reference/oracle.py norm_q7, softmax_q7, gelu_q7) around FP8
// matrix chains that accumulate in float32 over the whole chain with ONE FP16 rounding at its end.
//
// Exactness (clean/l/reference/hyb_tables.py, every fact over all possible FP16 inputs):
// - FP16 x FP16 and FP16 + FP16 done in float32 and rounded once to FP16 (round to nearest even) are the correctly
//   rounded FP16 results (24 >= 2 x 11 + 2: double rounding is innocuous), so a driver fusing the float32 operation
//   into its conversion gives the same bits;
// - the softmax weight e(u'): float32 fused multiply-adds with exact products, one FP16 rounding per step, equal the
//   FMA16 form for every clamped u' (softmax_e_f32_equals_fma16);
// - GELU16 -> FP8, the norm's rsqrt and the softmax normaliser are tables generated from the oracle (exact by
//   construction), read from the arena at WG_HYB_BASE (the runtime fills them, runtime/hyb_tables.h);
// - the chain-end double rounding float32 -> FP16 -> FP8 is two conversions (RN16, then the saturating FP8 pack of
//   the FP16 value): the build's ISA rule 2 (clean/dx12/hyb_isa_scan.py) requires both sources of every FP8
//   conversion to be FP16 values on every path, and each kernel's mutant (HYB_SCAN_MUTANT: no RN16 here) must fail it.
// Round to nearest even FP16: the native float16_t conversion (hyb_round16) or the driver's packed conversion
// (wg_to_f16_ags); f32tof16 rounds toward zero on this driver and is not used.
#pragma once

#ifndef WG_HYB_BASE
#define WG_HYB_BASE (WG_GELU_LUT_BASE + 16384u)   // after the 16 KB GELU table (runtime/hyb_tables.h kHybBase)
#endif
#define WG_HYB_GELU (WG_HYB_BASE)                // 65,536 bytes: q8(GELU16(h)) by the FP16 bits of h
#define WG_HYB_RSQ (WG_HYB_BASE + 65536u)        // 2 x 1024 FP16 (u16): r by exponent parity and mantissa of S'
#define WG_HYB_RCP (WG_HYB_RSQ + 4096u)          // 1024 FP16 (u16): w by the mantissa of the row sum
#define WG_HYB_GELU16 (WG_HYB_RCP + 2048u)       // 65,536 u16: the GELU table again, entry = the FP8 byte (high byte 0)
#define WG_HYB_BYTES (65536u + 4096u + 2048u + 131072u)

#define WG_HYB_SCORE_C asfloat(0x3c8cc000u)      // FP16 0x2466 as float32 (0.0972232 / sqrt(32) rounded)
#define WG_HYB_SCORE_CLAMP asfloat(0x3f0e6000u)  // FP16 0x3873 = 0.55615234375
#define WG_HYB_POLY_T asfloat(0x3f6d6000u)       // FP16 0x3B6B = 0.92724609375
#define WG_HYB_POLY_P asfloat(0x3fb00000u)       // FP16 0x3D80 = 1.375
#define WG_HYB_MERGE_CLAMP 6.28125f

// RN16 of 8 float32 values (the driver's round-to-nearest-even conversion), packed, and back to float32 (exact).
uint4 hyb_rn16x8(v8f d) { return wg_to_f16_ags(d); }
// The FP16 in the low half of v as float32 (exact). f16tof32 reads only the low 16 bits, so this is wg_h2f_lo without
// its "& 0xffff": the same conversion (v_cvt_f32_f16 of the low half), but the mask made the driver copy the half
// into a register of its own first (one v_mov_b16 per FP16 pair at every hybrid chain end).
float hyb_h2f_lo(uint v) { return f16tof32(v); }
v8f hyb_h2f8(uint4 h)
{
    v8f r;
    const uint w[4] = { h.x, h.y, h.z, h.w };
    [unroll] for (uint i = 0; i < 4; ++i) {
        r.v[2 * i] = hyb_h2f_lo(w[i]);
        r.v[2 * i + 1] = wg_h2f_hi(w[i]);
    }
    return r;
}
// RN16 per value by the driver's packed conversion (wg_to_f16_ags: v_cvt_f16_f32 in the default round-to-nearest-even
// mode), then back to float32 (exact). NOT (float)(float16_t)x: the driver lowers that cast to v_cvt_pk_rtz_f16_f32 under
// s_round_mode 0xc, i.e. round toward zero (RGA, v2 build). The probe's adversarial set checks every FP16 / FP8 midpoint.
#if defined(HYB_SCAN_MUTANT) && HYB_SCAN_MUTANT
// the ISA rules' mutant (*_scanmut, never in a model build; wrong values by design): the RN16 removed, so the FP8 pack
// takes the float32 value (hyb_isa_scan.py rule 2 must fail)
v8f hyb_round16(v8f d) { return d; }
#else
v8f hyb_round16(v8f d) { return hyb_h2f8(wg_to_f16_ags(d)); }
#endif

// An FP8 point of FP16 values r (lever, exact): when no value of the wave exceeds 448 in magnitude, the saturating
// pack's clamp and NaN keep change nothing, so the driver's plain conversion gives the same bytes (every NaN -> 0xff on
// both paths; the driver's max drops NaN, a wave of NaNs only takes the clamped path). Wave-uniform branch.
float hyb_absmax8(float m, v8f r)
{
    [unroll] for (uint i = 0; i < 8; ++i) m = max(m, abs(r.v[i]));
    return m;
}
bool hyb_wave_in448(v8f r)
{
    return WaveActiveAllTrue(hyb_absmax8(0.0f, r) <= 448.0f);
}
uint2 hyb_q8(v8f r)
{
    if (hyb_wave_in448(r)) return wg_pack_q8_nc_ags(r);
    return wg_pack_q8_ags(r);
}
void hyb_q8_pair(v8f r0, v8f r1, out uint2 q0, out uint2 q1)
{
    const float m = hyb_absmax8(hyb_absmax8(0.0f, r0), r1);
    if (WaveActiveAllTrue(m <= 448.0f)) {
        q0 = wg_pack_q8_nc_ags(r0);
        q1 = wg_pack_q8_nc_ags(r1);
    } else {
        q0 = wg_pack_q8_ags(r0);
        q1 = wg_pack_q8_ags(r1);
    }
}
// The chain end before an FP8 point: RN16, then the saturating FP8 pack of the FP16 value (two roundings, by design).
uint2 hyb_q8_of_chain(v8f acc) { return hyb_q8(hyb_round16(acc)); }

// GELU16 then the FP8 point, by table: h = the FP16 bits of 8 RN16'd W1 chain ends (hyb_rn16x8). Speed lever 2: the
// 16-bit-entry copy of the table, one u16 load per value at h << 1 (the byte table needed a dword address, a shift
// by the byte position and a mask per value); the same bytes (runtime/hyb_tables.h, checked by test_hyb_tables_cpp).
struct HybGeluPre { uint b[8]; };
HybGeluPre hyb_gelu_pre(uint4 h)
{
    const uint w[4] = { h.x, h.y, h.z, h.w };
    HybGeluPre r;
    [unroll] for (uint i = 0; i < 4; ++i) {
        r.b[2 * i] = (uint)wg_mem.Load<uint16_t>(WG_HYB_GELU16 + ((w[i] & 0xffffu) << 1));
        r.b[2 * i + 1] = (uint)wg_mem.Load<uint16_t>(WG_HYB_GELU16 + ((w[i] >> 16) << 1));
    }
    return r;
}
uint2 hyb_gelu_fin(HybGeluPre p)
{
    return uint2(p.b[0] | (p.b[1] << 8) | (p.b[2] << 16) | (p.b[3] << 24),
                 p.b[4] | (p.b[5] << 8) | (p.b[6] << 16) | (p.b[7] << 24));
}
uint2 hyb_gelu_q8(uint4 h) { return hyb_gelu_fin(hyb_gelu_pre(h)); }

// The softmax weight e of 8 FP16 scores s (float32 values of FP16 numbers): u = RN16(s c), clamp to +-c' (a NaN takes
// the lower clamp, as min / max return the number), t = RN16(fma(u, -u, t0)), p = RN16(fma(u, t, p0)),
// bits(e) = (bits(p) & 0x3FF) << 5. Returns e as float32 values (exact).
v8f hyb_softmax_e8(v8f s)
{
    v8f u;
    [unroll] for (uint i = 0; i < 8; ++i) u.v[i] = s.v[i] * WG_HYB_SCORE_C;
    u = hyb_round16(u);
    [unroll] for (uint i1 = 0; i1 < 8; ++i1) u.v[i1] = min(max(u.v[i1], -WG_HYB_SCORE_CLAMP), WG_HYB_SCORE_CLAMP);
    v8f t;
    [unroll] for (uint i2 = 0; i2 < 8; ++i2) t.v[i2] = mad(u.v[i2], -u.v[i2], WG_HYB_POLY_T);
    t = hyb_round16(t);
    v8f p;
    [unroll] for (uint i3 = 0; i3 < 8; ++i3) p.v[i3] = mad(u.v[i3], t.v[i3], WG_HYB_POLY_P);
    const uint4 pb = hyb_rn16x8(p);
    const uint w[4] = { pb.x, pb.y, pb.z, pb.w };
    v8f e;
    [unroll] for (uint i4 = 0; i4 < 4; ++i4) {
        e.v[2 * i4] = hyb_h2f_lo((w[i4] & 0x03ffu) << 5);
        e.v[2 * i4 + 1] = hyb_h2f_lo(((w[i4] >> 16) & 0x03ffu) << 5);
    }
    return e;
}

// r = RN16(rsqrt32(S')) for a positive normal FP16 S' (bits) in the norm's range (S' >= 2^-13), by table; S' = inf
// (the FP16 sum of squares overflowed) gives 0 and a NaN S' gives NaN, as rsqrt32 (oracle _rsqrt32, HIP 1 / sqrt):
// the table alone would read inf as 2^16 and a NaN as a finite number.
float hyb_rsq16_tab(uint sbits);
float hyb_rsq16(uint sbits)
{
    const float r = hyb_rsq16_tab(sbits);
    return (sbits & 0x7c00u) == 0x7c00u ? ((sbits & 0x03ffu) != 0u ? asfloat(0x7fc00000u) : 0.0f) : r;
}
float hyb_rsq16_tab(uint sbits)
{
    const int E = (int)((sbits >> 10) & 31u) - 15;
    const uint p = (uint)E & 1u;
    const int k = (E - (int)p) >> 1;
    const uint off = WG_HYB_RSQ + 2u * (p * 1024u + (sbits & 1023u));
    const uint rb = (wg_mem.Load(off & ~3u) >> (8u * (off & 2u))) & 0xffffu;
    return hyb_h2f_lo((uint)((int)rb - (k << 10)));
}

// w = RN16(rcp32(sum)) for a positive normal FP16 row sum (bits) with a normal result (every softmax row), by table.
float hyb_rcp16(uint sbits)
{
    const int E = (int)((sbits >> 10) & 31u) - 15;
    const uint off = WG_HYB_RCP + 2u * (sbits & 1023u);
    const uint rb = (wg_mem.Load(off & ~3u) >> (8u * (off & 2u))) & 0xffffu;
    return hyb_h2f_lo((uint)((int)rb - (E << 10)));
}

// The merge clamp before the FP8 point: max first, then min, NaN kept (Q8 B4).
float hyb_merge_clamp(float x) { return isnan(x) ? x : min(max(x, -WG_HYB_MERGE_CLAMP), WG_HYB_MERGE_CLAMP); }
// The merges' FP8 point: FP8(clamp(RN16(chain), +-6.28125)), NaN kept (oracle run_stage_q7: q8(clamp(mm8(...)))).
uint2 hyb_merge_q8(v8f acc)
{
    v8f r = hyb_round16(acc);
    [unroll] for (uint i = 0; i < 8; ++i) r.v[i] = hyb_merge_clamp(r.v[i]);
    return wg_pack_q8_ags(r);
}

// The decoder expansion's two mirrors (Q10b): the fine one for S, the coarse one for E (c = floor(p / 2)).
int hyb_mirror_fine(int p, int n) { return min(n - 1, min(abs(p), 2 * n - 2 - abs(p))); }
int hyb_mirror_coarse(int p, int n, int nc)
{
    const int c = p >> 1;   // floor(p / 2) for negative p too (arithmetic shift)
    return min(nc - 1, min(abs(c), 2 * (n >> 1) - 2 - abs(c)));
}

// ---- SW_Q7_CHUNK helpers: fast per-32-channel FP16-rounded WMMA chunks (oracle mm8="exact") ------------------------
// In-place 32-wide chunk accumulate into a packed FP16 tile (uint4): RN16(wmma(a1, b1, wmma(a0, b0, acc))).
// Only 12 VALUs + 2 WMMAs per 32-wide chunk (keeps acc packed in 4 VGPRs/tile for high occupancy).
uint4 hyb_q7c_step_h(uint2 a0, uint2 b0, uint2 a1, uint2 b1, uint4 acc_h)
{
    v8f t = wg_wmma_fp8(a0, b0, hyb_h2f8(acc_h));
    t = wg_wmma_fp8(a1, b1, t);
    return hyb_rn16x8(t);
}

// Guarded exact 16-term FP16xFP16 dot product for P V (matching l_q7w.hlsl two_sum + rn16_pair).
// Fast path: checks whether RN16(pv - err) == RN16(pv + err) for err = 2^-21 * max_finite|V| across the wave.
// Cold path: unpacked on the fly from packed uint4 p0/p1/v0/v1 (low VGPRs so c64 keeps 3 wg/CU) and only shuffles V
// for the specific v slot(s) where WaveActiveAnyTrue(need_v) holds.
void hyb_q7c_two_sum_pair(uint p_word, uint v_word, inout float s, inout float c)
{
    {
        const float pr = f16tof32(p_word) * f16tof32(v_word);
        const float u = s + pr;
        const float bb = u - s;
        c = c + ((s - (u - bb)) + (pr - bb));
        s = u;
    }
    {
        const float pr = wg_h2f_hi(p_word) * wg_h2f_hi(v_word);
        const float u = s + pr;
        const float bb = u - s;
        c = c + ((s - (u - bb)) + (pr - bb));
        s = u;
    }
}

// Guarded exact 16-term FP16xFP16 dot product for P V -> FP8 (matching l_q7w.hlsl two_sum + rn16_pair + q8).
// Because FP8(RN16(x)) is monotone, if qlo = FP8(RN16(pv - err)) == qhi = FP8(RN16(pv + err)) for err = 2^-20 * max|V|,
// then FP8(RN16(pv_exact)) == qlo is guaranteed! Since FP8 has 128x coarser spacing than FP16 (and never triggers
// near zero), the cold two_sum path executes on <0.3% of waves while guaranteeing 100% bit-exact FP8 output.
uint2 hyb_q7c_pv_q8(uint4 vt_dt, uint4 p, uint hh, uint z0, v8f pv)
{
    uint m16 = max(max(vt_dt.x & 0x7fffu, (vt_dt.x >> 16) & 0x7fffu), max(vt_dt.y & 0x7fffu, (vt_dt.y >> 16) & 0x7fffu));
    m16 = max(m16, max(max(vt_dt.z & 0x7fffu, (vt_dt.z >> 16) & 0x7fffu), max(vt_dt.w & 0x7fffu, (vt_dt.w >> 16) & 0x7fffu)));
    if (m16 > 0x7bffu) m16 = 0x7bffu;
    const float err = f16tof32(WaveActiveMax(m16)) * asfloat(0x35800000u);   // 2^-20 * max|V| (16x safety margin)
    v8f pv_lo, pv_hi;
    [unroll] for (uint i = 0u; i < 8u; ++i) { pv_lo.v[i] = pv.v[i] - err; pv_hi.v[i] = pv.v[i] + err; }
    uint2 qlo, qhi;
    hyb_q8_pair(hyb_round16(pv_lo), hyb_round16(pv_hi), qlo, qhi);
    const uint d0 = qlo.x ^ qhi.x, d1 = qlo.y ^ qhi.y;
    if (!WaveActiveAnyTrue((d0 | d1) != 0u)) return qlo;
    const uint4 po = uint4(wg_xor16_u(p.x), wg_xor16_u(p.y), wg_xor16_u(p.z), wg_xor16_u(p.w));
    const uint4 p0 = (hh == 0u) ? p : po;
    const uint4 p1 = (hh == 0u) ? po : p;
    [loop] for (uint v = 0u; v < 8u + z0; ++v) {
        const uint dw = (v < 4u) ? d0 : d1;
        const bool need_v = (((dw >> ((v & 3u) * 8u)) & 0xffu) != 0u);
        if (WaveActiveAnyTrue(need_v)) {
            const uint cl = hh * 8u + v;
            const uint4 v0 = uint4(WaveReadLaneAt(vt_dt.x, cl), WaveReadLaneAt(vt_dt.y, cl),
                                   WaveReadLaneAt(vt_dt.z, cl), WaveReadLaneAt(vt_dt.w, cl));
            const uint4 v1 = uint4(WaveReadLaneAt(vt_dt.x, cl + 16u), WaveReadLaneAt(vt_dt.y, cl + 16u),
                                   WaveReadLaneAt(vt_dt.z, cl + 16u), WaveReadLaneAt(vt_dt.w, cl + 16u));
            if (need_v) {
                float s = 0.0f, c = 0.0f;
                hyb_q7c_two_sum_pair(p0.x, v0.x, s, c);
                hyb_q7c_two_sum_pair(p0.y, v0.y, s, c);
                hyb_q7c_two_sum_pair(p0.z, v0.z, s, c);
                hyb_q7c_two_sum_pair(p0.w, v0.w, s, c);
                hyb_q7c_two_sum_pair(p1.x, v1.x, s, c);
                hyb_q7c_two_sum_pair(p1.y, v1.y, s, c);
                hyb_q7c_two_sum_pair(p1.z, v1.z, s, c);
                hyb_q7c_two_sum_pair(p1.w, v1.w, s, c);
                const float z = s + c;
                const float bb2 = z - s;
                const float tl = (s - (z - bb2)) + (c - bb2);
                const uint zb = asuint(z), tb = asuint(tl);
                const bool ok = ((zb & 0x7f800000u) != 0x7f800000u) && ((tb & 0x7fffffffu) != 0u);
                const uint opp = (zb ^ tb) >> 31;
                pv.v[v] = asfloat(ok ? ((zb - opp) | 1u) : zb);
            }
        }
    }
    return hyb_q8_of_chain(pv);
}

// Reorder N (even) FP8 fragments from Layout L (hh=0: 0..8N-1, hh=1: 8N..16N-1) to ascending kord_drow
// (dst[g]: hh=0 holds 16g..16g+7, hh=1 holds 16g+8..16g+15) using N/2 lane-xor16 exchanges.
#define HYB_Q7C_L2D(N, src, dst, hh)                                              \
    do {                                                                          \
        const bool _h0 = ((hh) == 0u);                                            \
        [unroll] for (uint _p = 0u; _p < (N) / 2u; ++_p) {                        \
            const uint2 _s = _h0 ? (src)[2u * _p + 1u] : (src)[2u * _p];          \
            const uint2 _r = uint2(wg_xor16_u(_s.x), wg_xor16_u(_s.y));           \
            (dst)[_p] = _h0 ? (src)[2u * _p] : _r;                                \
            (dst)[(N) / 2u + _p] = _h0 ? _r : (src)[2u * _p + 1u];                \
        }                                                                         \
    } while (0)
