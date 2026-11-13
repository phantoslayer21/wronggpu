// SPDX-License-Identifier: MIT
#ifndef RFX_QLD
#define RFX_QLD 0   // clean/rr-f-a14-tail D17: the Q texel as one aligned b64 + shifts (same words; measured +0.001 ms)
#endif
// rr_resolve_fast: page 54b.1-54b.4 (the work of rr_resolve_main) with the same bits for every input.
// Included by rr_resolve_fast.hlsl after rr_resolve.hlsl (helpers of the verified kernel) and the exp2 table.
//
// Work split: persistent workgroups of 256 lanes walk 16 x 16 output tiles (tile g, g + G, ...). Per tile:
//   1. the binary16 Q texels of the tile's 5x5 footprint -> groupshared;
//   2. the horizontal 5-tap pass of 54b.1 once per (Q row, output column) -> groupshared (shared by the rows);
//   3. the render-size texels of the tile's footprint (alpha taps, 3x3 neighbourhoods) -> groupshared, with the
//      per-texel products qx / qy / qxy, the alpha value, the G halves and E * layer computed once per texel by the
//      verified helpers;
//   4. per pixel: vertical pass, decode, layer blend, alpha upscaling, stores.
// Arithmetic: per pixel the f32 operations are the hardware's (one IEEE rounding, FTZ: the verified helpers'
// operations minus their flush-boundary repair), exp2 / rcp / f32->f16 / unorm codes come from
// rr_resolve_fast_math.hlsli (exhaustively equal to the verified helpers), binary16 fma / min / max are the hardware's
// (equal to wg_fma_f16 / pmin / pmax: gpu/resolve_fast.py probes). A pixel whose values could reach the one place where
// the hardware differs from the verified helpers (an exact f32 result below the smallest normal, which the hardware
// flushes before rounding) is detected and recomputed by rr_resolve_px, the verified kernel's own code; so is a pixel
// with an undecided exp2. RESOLVE-FAST.md gives the range argument behind every unchecked operation.

#ifndef RF_COUNT
#define RF_COUNT 0                    // diagnostics: count the pixels sent to the verified path at arena offset aux
#endif
#ifndef RF_TIMING
#define RF_TIMING 0                   // profiling: shader cycles per phase summed per wave at arena offset aux
#endif
#ifndef RF_EARLY
#define RF_EARLY 1                    // 1: the layer / history reads issued before the plain colour and the alpha taps
#endif
#ifndef RF_PROF
#define RF_PROF 0                     // profiling only (wrong results): 1 no alpha taps, 2 no layer, 4 no decode, 8 no render tile, 16 no Q pass, 64 no second-tier exp2 in the taps (ISA counts)
#endif
#ifndef RF_SPEC
#define RF_SPEC 0                     // 1: the standard configuration only (rf_uniforms checks it; otherwise every pixel
                                      // goes to the fixup kernel): extended + layer + alpha upscaling (any reset,
                                      // separate alpha, bounded, exposure-link flags), no opacity; P, L, MV, TH, OUT
                                      // binary16 x4, G unorm8 x4, alpha source RGB10A2 packed (channel 3), AH binary16
                                      // x1, AOUT f32 x1. Format reads without switches, no per-pixel branches.
                                      // 2: the game's configuration (exact, same bits): extended + layer WITHOUT alpha
                                      // upscaling (any reset / bounded / exposure-link flags), no opacity; L, MV, TH,
                                      // OUT binary16 x4. As 1 without the alpha work: the render tile holds only the
                                      // layer (no P / G / alpha source loads), no alpha taps / history / stores.
#endif
#ifndef RF_HOIST                      // round 13 (RF_SPEC, RF_EARLY; the same addresses and words, issued earlier): 1 the
#define RF_HOIST 0                    // motion-vector and layer reads at the tile start, the history reads after the Q pass
#endif                                // (measured slower: VGPRs 99 -> 114); 2 the motion-vector read only
#if !RF_SPEC || !RF_EARLY || RF_PROF
#undef RF_HOIST
#define RF_HOIST 0
#endif
#define RF_T 16u                      // tile side (output pixels)
#define RF_QN 29u                     // Q tile side bound (16 output pixels need at most 28 Q texels per axis)
#define RF_RN 20u                     // render tile side bound (render / output <= 1 needs at most 20)
#define RF_NR (RF_RN * RF_RN)
#define RF_TINY asfloat(0x33800000u)  // 2^-24: smallest magnitude of an "ordinary" nonzero input

groupshared uint2 rf_hr[RF_T * RF_QN];        // horizontal pass: (rg, b) binary16 per (Q row, output column)
groupshared uint4 rf_u4[1024];                // phase 1: Q tile, (rg, b) per texel (<= 29 x 29);
                                              // phase 2: render tile, [e] = (qx, qy, qxy, a), [RF_NR + e] = E * L.rgb, L.a
groupshared uint2 rf_g[RF_NR];                // phase 2: G halves (g0 | g1 << 16, g2 | not-ordinary << 16)
#ifndef RFX_NB
#define RFX_NB 3                      // clean/rr-f-a14-tail D9 (same bits; default 3): 1 = the Q tile in its own array, so nothing the
#endif                                // next tile writes before its first barrier is read after the current tile's last
#if RFX_NB == 1                       // one: the end-of-tile barrier goes (4 -> 3 per tile; LDS + 6.7 KB); 2 = negative control
groupshared uint2 rf_q[RF_QN * RF_QN];
#define RF_QW(e, v) rf_q[e] = (v).xy
#define RF_QR(i) rf_q[i]
#elif RFX_NB == 3 && RF_SPEC == 2     // 3 (RF_SPEC 2 only): the Q tile (b64 texels) and the layer tile (2 x b64 per texel)
// ---- clean/rr-mp1-resolve (same bits): tile bookkeeping by layout (mpeak probe P6) ----------------------------------
#ifndef RFX_TB        // default 23 (wave64, 256 lanes: the runtime's resolve.fast2 / resolve.fast2.x); 0 = the old code
#if WG_BLOCK == 256 && WG_WAVE == 64
#define RFX_TB 23     // bit 1: T1 (Q tile at LDS pitch 32, render tile at pitch RFX_RP, per-lane tile-geometry table,
#else                 // uniform screens Q.w >= Wh, Q.h >= Hh, Q.ch == 3); bit 2: T5 interior tiles; bit 4: T9 (no
#define RFX_TB 0      // phase-2 -> phase-3 barrier); bit 16: the compact 2-VGPR table (the 4-VGPR one sits at the
#endif                // 108-VGPR limit)
#endif
#if (RFX_TB & 1) && (WG_BLOCK != 256 || WG_WAVE != 64)
#error RFX_TB needs 256 lanes (Q-tile rows 8 per slot) and wave64 (64-tile lane table)
#endif
#ifndef RFX_RP
#define RFX_RP 20     // T1 render-tile pitch: 20 (any render / output <= 1) or 16 (PROBE: ratio-limited, nrx, nry <= 16)
#endif
#ifndef RFX_LR        // clean/rr-mp2-tail (same bits): the layer tile holds the raw binary16 words (one b64 per texel:
#define RFX_LR 1      // LDS 18,304 -> 15,104 B, so 4 groups of 4 wave64 per CU instead of 3, i.e. 8 waves per SIMD
#endif                // instead of 6 at <= 96 VGPRs); the 3x3 reads form E * widen(h) at each read with rf_slot's
                      // own operation. 0 = the old layout (E * L.rgb, L.a as f32 in two b64 per texel)
#if RFX_TB & 1
#define RF_QO 1024u                   // Q region: 32 x 32 b64 entries at pitch RF_QP
#define RF_QP 32u
#else
#define RF_QO (RF_QN * RF_QN)
#endif
#if RFX_LR && (RFX_TB & 1)
groupshared uint2 rf_w2[RF_QO + RF_NR];        // RFX_LR: the Q tile and the raw layer tile (one b64 per texel)
#else
groupshared uint2 rf_w2[RF_QO + 2u * RF_NR];   // side by side in one b64 array (LDS 20.1 -> 16.8 KB), no end barrier
#endif
#define RF_QW(e, v) rf_w2[e] = (v).xy
#define RF_QR(i) rf_w2[i]
#define RF_LW(e, v) { const uint4 v_ = (v); rf_w2[RF_QO + 2u * (e)] = v_.xy; rf_w2[RF_QO + 2u * (e) + 1u] = v_.zw; }
#define RF_LR(i) uint4(rf_w2[RF_QO + 2u * (i)], rf_w2[RF_QO + 2u * (i) + 1u])
#else
#define RF_QW(e, v) rf_u4[e] = (v)
#define RF_QR(i) rf_u4[i].xy
#endif
#ifndef RF_LW
#define RF_LW(e, v) rf_u4[RF_NR + (e)] = (v)
#define RF_LR(i) rf_u4[RF_NR + (i)]
#endif
#ifndef RFX_TB
#define RFX_TB 0
#endif
#ifndef RFX_RP
#define RFX_RP 20
#endif
#if RFX_TB && !(RFX_NB == 3 && RF_SPEC == 2)
#error RFX_TB needs RF_SPEC 2 and RFX_NB 3
#endif
#if !(RF_SPEC == 2 && (RFX_TB & 1))
#undef RFX_LR
#define RFX_LR 0
#endif
#ifndef RFX_MIX       // same bits: widen-then-multiply as v_fma_mix_f32(widen(h), g, -0): x*g + (-0) = x*g, one rounding
#if RF_SPEC == 2      // in both (+0 + -0 = +0, -0 + -0 = -0). 1 = rf_slot's E * widen (-0 in a VGPR); 3 = also the four
#define RFX_MIX 3     // phase-4 bilinear products g * widen(h) (-0 in an SGPR). Default 3 for RF_SPEC 2.
#else
#define RFX_MIX 0
#endif
#endif
static float rf_mz;   // RFX_MIX 3: an opaque -0
#if RFX_LR && !RFX_MIX
#error RFX_LR needs RFX_MIX (rf_lr4 repeats rf_slot's RFX_MIX product)
#endif

// clean/rr-f-s10-tail: the main output stored into the game's texture by resolve.fast2 and resolve.fixup (parameter
// word 16 = 1, set by the runtime instead of dispatching rt.store for it; word 39 = the destination origin x | y << 16):
// rt.store's own conversion (each binary16 code of the arena texel widened by f16tof32, the float4 through the typed
// UAV u2), for the texels rt.store would write (x < Wo, y < Ho when the stores are bounded), so the same bits.
RWTexture2D<float4> rf_gout : register(u2);
// (an opaque zero XOR keeps the f32 values: the driver would otherwise store the binary16 codes as a d16 image store,
// whose format conversion of NaN / subnormal codes is not rt.store's f32 -> format conversion)
void rf_gstore(uint2 org, int x, int y, uint2 w)
{
    const uint z = 0u - (wg_grid_ >> 31);
    rf_gout[uint2(org.x + (uint)x, org.y + (uint)y)] =
        asfloat(asuint(float4(f16tof32(w.x & 0xffffu), f16tof32(w.x >> 16), f16tof32(w.y & 0xffffu), f16tof32(w.y >> 16))) ^ z);
}

// 24-bit product (v_mul_u32_u24, full rate): a row index and a pitch (each below 2^24; their product, the texel
// index of a row start, below 2^32)
uint rf_m24(uint a, uint b) { return (a & 0xffffffu) * (b & 0xffffffu); }
bool rf_tiny(float v) { return and(v != 0.0f, abs(v) < RF_TINY); }     // nonzero below 2^-24 (NaN: false)
bool rf_zc(float r, float c) { return and(r == 0.0f, c != 0.0f); }     // a rounded result 0 from a nonzero addend
float rf_rnd(float v)                                                   // RND of rr_resolve.hlsl (hardware ops)
{
    float t = trunc(v), f = v + (-t);
    return select(abs(f) >= 0.5f, t + select(v < 0.0f, -1.0f, 1.0f), t);
}
float rf_clampHist(float h, float mn, float mx)
{
    return select(or(or(h > mn, nanf(h)), nanf(mn)), select(h >= mx, mx, h), mn);
}

// texel() of rr_resolve.hlsl with the unorm reads of rr_resolve_fast_math.hlsli; tiny |= an f32 texel channel that
// is not ordinary.
float4 rf_texel(Res r, int x, int y, inout bool tiny)
{
    float4 v = float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (x < 0 || y < 0 || x >= (int)r.w || y >= (int)r.h) return v;
    uint idx = (uint)y * r.pitch + (uint)x;
    if (r.fmt == 3u) {
        uint wv = wg_ld32(r.off, idx * 4u);
        return float4(rf_cv(wv & 1023u, 1023.0f, RF_R1023), rf_cv((wv >> 10) & 1023u, 1023.0f, RF_R1023),
                      rf_cv((wv >> 20) & 1023u, 1023.0f, RF_R1023), rf_cv(wv >> 30, 3.0f, RF_R3));
    }
    if (r.fmt == 4u) {
        uint b = r.off + idx * 8u;
        return float4(rf_cv(wg_ld16(b, 0u), 1023.0f, RF_R1023), rf_cv(wg_ld16(b, 2u), 1023.0f, RF_R1023),
                      rf_cv(wg_ld16(b, 4u), 1023.0f, RF_R1023), rf_cv(wg_ld16(b, 6u), 3.0f, RF_R3));
    }
    if (r.ch == 4u) {                                   // whole texels: one load
        if (r.fmt == 1u) {
            uint2 h = wg_ld64(r.off, idx * 8u);
            return float4(f16tof32(h.x), f16tof32(h.x >> 16), f16tof32(h.y), f16tof32(h.y >> 16));
        }
        if (r.fmt == 0u) {
            float4 f = asfloat(wg_ld128(r.off, idx * 16u));
            f = float4(ftz(f.x), ftz(f.y), ftz(f.z), ftz(f.w));
            tiny = or(tiny, or(or(rf_tiny(f.x), rf_tiny(f.y)), or(rf_tiny(f.z), rf_tiny(f.w))));
            return f;
        }
        uint b = wg_ld32(r.off, idx * 4u);
        return float4(rf_cv(b & 255u, 255.0f, RF_R255), rf_cv((b >> 8) & 255u, 255.0f, RF_R255),
                      rf_cv((b >> 16) & 255u, 255.0f, RF_R255), rf_cv(b >> 24, 255.0f, RF_R255));
    }
    [unroll] for (uint c = 0; c < 4u; ++c) {
        float t = c == 3u ? 1.0f : 0.0f;
        if (c < r.ch) {
            uint e = idx * r.ch + c;
            if (r.fmt == 0u) { t = ftz(asfloat(wg_ld32(r.off, e * 4u))); tiny = or(tiny, rf_tiny(t)); }
            else if (r.fmt == 1u) t = f16tof32(wg_ld16(r.off, e * 2u));
            else t = rf_cv(wg_ld8(r.off, e), 255.0f, RF_R255);
        }
        v[c] = t;
    }
    return v;
}
// binary16 x4 texel (RF_SPEC): rf_h4i inside the resource (clamped bilinear indices), rf_h4 with the outside rule.
float4 rf_h4i(Res r, int x, int y)
{
    uint2 h = wg_ld64(r.off, (rf_m24((uint)y, r.pitch) + (uint)x) * 8u);
    return float4(f16tof32(h.x), f16tof32(h.x >> 16), f16tof32(h.y), f16tof32(h.y >> 16));
}
float4 rf_h4(Res r, int x, int y)
{
    bool ins = and(and(x >= 0, y >= 0), and(x < (int)r.w, y < (int)r.h));
    float4 v = rf_h4i(r, clamp(x, 0, (int)r.w - 1), clamp(y, 0, (int)r.h - 1));
    return float4(select(ins, v.x, 0.0f), select(ins, v.y, 0.0f), select(ins, v.z, 0.0f), select(ins, v.w, 0.0f));
}
float rf_h1i(Res r, int x, int y) { return f16tof32(wg_ld16(r.off, (rf_m24((uint)y, r.pitch) + (uint)x) * 2u)); }

// One channel ach of rf_texel (the alpha source read of 54b.4).
float rf_texel1(Res r, int x, int y, uint ach, inout bool tiny)
{
    if (x < 0 || y < 0 || x >= (int)r.w || y >= (int)r.h) return 0.0f;
    uint idx = (uint)y * r.pitch + (uint)x;
    if (r.fmt == 3u) {
        uint c = (wg_ld32(r.off, idx * 4u) >> (10u * ach)) & select(ach == 3u, 3u, 1023u);
        return select(ach == 3u, rf_cv(c, 3.0f, RF_R3), rf_cv(c, 1023.0f, RF_R1023));
    }
    if (r.fmt == 4u) {
        uint c = wg_ld16(r.off + idx * 8u, ach * 2u);
        return select(ach == 3u, rf_cv(c, 3.0f, RF_R3), rf_cv(c, 1023.0f, RF_R1023));
    }
    if (ach >= r.ch) return ach == 3u ? 1.0f : 0.0f;
    uint e = idx * r.ch + ach;
    if (r.fmt == 0u) { float t = ftz(asfloat(wg_ld32(r.off, e * 4u))); tiny = or(tiny, rf_tiny(t)); return t; }
    if (r.fmt == 1u) return f16tof32(wg_ld16(r.off, e * 2u));
    return rf_cv(wg_ld8(r.off, e), 255.0f, RF_R255);
}

// store_px of rr_resolve.hlsl with the conversions of rr_resolve_fast_math.hlsli.
void rf_store(Res r, int x, int y, float4 v)
{
    if (r.off == 0u || x < 0 || y < 0 || x >= (int)r.w || y >= (int)r.h) return;
    uint idx = (uint)y * r.pitch + (uint)x;
    if (r.fmt == 3u) {
        wg_st32(r.off, idx * 4u, rf_unorm_code(v.x, 1023u) | (rf_unorm_code(v.y, 1023u) << 10) |
                                 (rf_unorm_code(v.z, 1023u) << 20) | (rf_unorm_code(v.w, 3u) << 30));
        return;
    }
    if (r.fmt == 4u) {
        uint b = r.off + idx * 8u;
        wg_st16(b, 0u, rf_unorm_code(v.x, 1023u)); wg_st16(b, 2u, rf_unorm_code(v.y, 1023u));
        wg_st16(b, 4u, rf_unorm_code(v.z, 1023u)); wg_st16(b, 6u, rf_unorm_code(v.w, 3u));
        return;
    }
    if (r.fmt == 2u && r.ch == 4u) {
        wg_st32(r.off, idx * 4u, rf_unorm_code(v.x, 255u) | (rf_unorm_code(v.y, 255u) << 8) |
                                 (rf_unorm_code(v.z, 255u) << 16) | (rf_unorm_code(v.w, 255u) << 24));
        return;
    }
    if (r.fmt == 1u && r.ch == 4u) {
        wg_st64(r.off, idx * 8u, uint2(rf_f2h(v.x) | (rf_f2h(v.y) << 16), rf_f2h(v.z) | (rf_f2h(v.w) << 16)));
        return;
    }
    [unroll] for (uint c = 0; c < 4u; ++c) {
        if (c < r.ch) {
            uint e = idx * r.ch + c;
            if (r.fmt == 0u) wg_st32(r.off, e * 4u, asuint(v[c]));
            else if (r.fmt == 1u) wg_st16(r.off, e * 2u, rf_f2h(v[c]));
            else st_byte(r.off + e, rf_unorm_code(v[c], 255u));
        }
    }
}

// ---- bilinear sample of 52.1.4 (bilin of rr_resolve.hlsl): coordinates once, applied to any texture of that size --
struct RfBil { int xa, xb, ya, yb; float fx, gx, fy, gy, s, t; bool nx, ny; };
RfBil rf_bil(float u, float v, uint W, uint H)
{
    RfBil b;
    u = select(nanf(u), 0.0f, select(u < 0.0f, 0.0f, select(u > 1.0f, 1.0f, u)));
    v = select(nanf(v), 0.0f, select(v < 0.0f, 0.0f, select(v > 1.0f, 1.0f, v)));
    b.s = u * (float)W + (-0.5f);
    b.t = v * (float)H + (-0.5f);
    float i0 = floor(b.s), j0 = floor(b.t);
    b.nx = b.s < 0.0f; b.ny = b.t < 0.0f;
    b.fx = select(b.nx, 0.0f, b.s + (-i0)); b.fy = select(b.ny, 0.0f, b.t + (-j0));
    b.gx = select(b.nx, -b.s, 1.0f + (-b.fx)); b.gy = select(b.ny, -b.t, 1.0f + (-b.fy));
    b.xa = clamp((int)i0, 0, (int)W - 1); b.xb = clamp((int)i0 + 1, 0, (int)W - 1);
    b.ya = clamp((int)j0, 0, (int)H - 1); b.yb = clamp((int)j0 + 1, 0, (int)H - 1);
    return b;
}
float rf_wterm(float g, float s, bool neg, float t)
{
    return select(neg, select(or(t == 0.0f, !isfinite(t)), t, mad(s, t, t)), g * t);
}
float rf_bil1(RfBil b, float t00, float t10, float t01, float t11)
{
    float r0 = b.gx * t00 + rf_wterm(b.fx, b.s, b.nx, t10);
    float r1 = b.gx * t01 + rf_wterm(b.fx, b.s, b.nx, t11);
    return b.gy * r0 + rf_wterm(b.fy, b.t, b.ny, r1);
}
float4 rf_bil4(RfBil b, float4 t00, float4 t10, float4 t01, float4 t11)
{
    float4 o;
    if (WaveActiveAnyTrue(or(b.nx, b.ny)) && RF_MUT != 3) {      // mutation 3: always the plain formula
        [unroll] for (uint c = 0; c < 4u; ++c) o[c] = rf_bil1(b, t00[c], t10[c], t01[c], t11[c]);
    } else {                                            // s, t >= 0 in the whole wave: wterm is the plain product
#if RFX_MIX >= 3
        [unroll] for (uint c = 0; c < 4u; ++c)          // all four products g * widen(h) as fma(widen(h), g, -0)
            o[c] = b.gy * (mad(t00[c], b.gx, rf_mz) + mad(t10[c], b.fx, rf_mz)) + b.fy * (mad(t01[c], b.gx, rf_mz) + mad(t11[c], b.fx, rf_mz));
#else
        [unroll] for (uint c = 0; c < 4u; ++c)
            o[c] = b.gy * (b.gx * t00[c] + b.fx * t10[c]) + b.fy * (b.gx * t01[c] + b.fx * t11[c]);
#endif
    }
    return o;
}
float rf_bil1w(RfBil b, float t00, float t10, float t01, float t11)
{
    float o;
    if (WaveActiveAnyTrue(or(b.nx, b.ny))) o = rf_bil1(b, t00, t10, t01, t11);
    else o = b.gy * (b.gx * t00 + b.fx * t10) + b.fy * (b.gx * t01 + b.fx * t11);
    return o;
}

// ---- per-dispatch values -------------------------------------------------------------------------------------------
struct RfU {
    uint We, He, Wh, Hh, flags, ntx;
    int Wr, Hr;
    float E, jx, jy, sxr, syr, sxh, syh, jxp, jyp, rWe, rHe, fac, rE, TBW, TBW2, w1a, w1b;
    float K1, K2n, K2x2n, K5, K8, K10, K12, K13;
    bool ok;      // the fast path applies to this dispatch
};
RfU rf_uniforms()
{
    RfU U;
    U.We = PU(0); U.He = PU(1); U.Wh = PU(2); U.Hh = PU(3); U.Wr = (int)PU(4); U.Hr = (int)PU(5); U.flags = PU(8);
    U.ntx = (U.We + RF_T - 1u) / RF_T;
    uint flags = U.flags;
    U.E = (flags & 256u) != 0u ? asfloat(wg_ld32(PU(15), 0u)) : PF(9);
    float Eprev = (flags & 256u) != 0u ? asfloat(wg_ld32(PU(15), 4u)) : PF(10);
    U.E = WaveReadLaneFirst(U.E); Eprev = WaveReadLaneFirst(Eprev);
    U.jx = PF(11); U.jy = PF(12);
    float fWe = (float)U.We, fHe = (float)U.He, fWr = (float)U.Wr, fHr = (float)U.Hr;
    // DIV / RCP of the verified kernel (binary64), one quotient per lane of the wave (lane k computes quotient k),
    // then read from those lanes. DIV(a, b) = rounded(a / b) for the finite nonzero sizes; RCP(x) as rr_resolve_math.
    uint k = WaveGetLaneIndex();
    float num = 1.0f, den = 1.0f;
    num = select(k == 0u, fWr, num); den = select(k == 0u, fWe, den);
    num = select(k == 1u, fHr, num); den = select(k == 1u, fHe, den);
    num = select(k == 2u, fWe, num); den = select(k == 2u, (float)U.Wh, den);
    num = select(k == 3u, fHe, num); den = select(k == 3u, (float)U.Hh, den);
    num = select(k == 4u, fWe, num); den = select(k == 4u, fWr, den);
    num = select(k == 5u, fHe, num); den = select(k == 5u, fHr, den);
    den = select(k == 6u, fWe, den); den = select(k == 7u, fHe, den);
    den = select(k == 8u, pmax(Eprev, PF(34)), den); den = select(k == 9u, U.E, den);
    float qv = select(k < 6u, DIV(num, den), RCP(den));
    U.sxr = WaveReadLaneAt(qv, 0u); U.syr = WaveReadLaneAt(qv, 1u); U.sxh = WaveReadLaneAt(qv, 2u);
    U.syh = WaveReadLaneAt(qv, 3u);
    U.jxp = M(WaveReadLaneAt(qv, 4u), U.jx); U.jyp = M(WaveReadLaneAt(qv, 5u), U.jy);
    U.rWe = WaveReadLaneAt(qv, 6u); U.rHe = WaveReadLaneAt(qv, 7u);
    U.fac = M(U.E, WaveReadLaneAt(qv, 8u));
    U.rE = WaveReadLaneAt(qv, 9u);
    U.TBW = PF(35); U.TBW2 = M(U.TBW, U.TBW); U.w1a = A(1.0f, -U.TBW); U.w1b = A(1.0f, -U.TBW2);
    U.K1 = PF(22); U.K2n = PF(23); U.K2x2n = PF(24); U.K5 = PF(26); U.K8 = PF(28); U.K10 = PF(29); U.K12 = PF(31);
    U.K13 = PF(32);
    // range screen of every per-dispatch factor (RESOLVE-FAST.md): zero or at least 2^-24 in magnitude
    bool bad = false;
    float sc[9] = { U.K5, U.K8, U.K10, U.K12, U.E, U.rE, U.K1, U.K2n, U.K2x2n };
    [unroll] for (uint i = 0; i < 9u; ++i) bad = or(bad, rf_tiny(sc[i]));
    // K13 only enters rW = RCP(max(W, K13)) (proven for every input, f32 max = pmax without denormal operands) and
    // rW only the zero-checked products Gs rW, Aw rW: K13 needs to be 0 or normal, not >= 2^-24 (F-B1: 1e-10)
    bad = or(bad, and(U.K13 != 0.0f, abs(U.K13) < asfloat(0x00800000u)));
    if ((flags & 1u) != 0u) {
        float se[10] = { U.jx, U.jy, U.jxp, U.jyp, U.sxr, U.syr, U.sxh, U.syh, U.rWe, U.rHe };
        [unroll] for (uint k = 0; k < 10u; ++k) bad = or(bad, rf_tiny(se[k]));
        bad = or(bad, or(!isfinite(U.jx), !isfinite(U.jy)));
    }
    if ((flags & 3u) == 3u) {
        float sl[5] = { U.fac, U.TBW, U.TBW2, U.w1a, U.w1b };
        [unroll] for (uint k = 0; k < 5u; ++k) bad = or(bad, rf_tiny(sl[k]));
    }
    Res q = res(40u);
    bad = or(bad, or(q.fmt != 1u, q.ch < 3u));
    bad = or(bad, (flags & 4u) != 0u);                                   // opacity layer: verified path
    if ((flags & 3u) == 3u) { Res th = res(112u), tw = res(120u); bad = or(bad, or(or(th.fmt != 1u, th.ch != 4u), and(tw.off != 0u, or(tw.fmt != 1u, tw.ch != 4u)))); }
    if ((flags & 9u) == 9u) { Res ah = res(144u), aw = res(152u); bad = or(bad, or(ah.fmt != 1u, and(aw.off != 0u, or(aw.fmt != 1u, aw.ch != 1u)))); }
    if ((flags & 11u) == 11u) { Res th = res(112u), ah = res(144u); bad = or(bad, or(th.w != ah.w, th.h != ah.h)); }
    bad = or(bad, or(PU(37) == 0u, PU(38) == 0u));                      // no fixup bitmap / exp2 table
    bad = or(bad, or(!(abs(U.K2n) <= asfloat(0x71800000u)), !(abs(U.K2x2n) <= asfloat(0x71800000u))));   // <= 2^100
#if RF_SPEC
    {
        Res P = res(96u), G = res(104u), Ar = res(80u), L = res(64u), mv = res(88u), O = res(48u), AO = res(56u);
        Res th = res(112u), tw = res(120u), ah = res(144u), aw = res(152u);
#if RF_SPEC == 2
        bool f = (flags & 15u) == 3u;                   // extended + layer; no opacity, no alpha upscaling
        f = and(f, and(L.fmt == 1u, L.ch == 4u));
        f = and(f, and(and(mv.off != 0u, mv.fmt == 1u), mv.ch == 4u));
        f = and(f, and(and(th.off != 0u, tw.off != 0u), and(tw.fmt == 1u, tw.ch == 4u)));
        f = and(f, and(O.fmt == 1u, O.ch == 4u));
#if RFX_TB & 1
        // T1 screens (a failing dispatch takes the verified path, as for every screen above)
        f = and(f, and(and(q.w >= U.Wh, q.h >= U.Hh), and(q.ch == 3u, q.pitch < (1u << 21))));
        f = and(f, and(and(U.We <= (1u << 20), U.He <= (1u << 20)), and(U.Wh < 65536u, U.Hh < 65536u)));
        f = and(f, and(U.Wr < 65536, U.Hr < 65536));
#if RFX_TB & 16
        f = and(f, and(U.We <= 32768u, U.He <= 32768u));
#endif
#endif
#else
        bool f = (flags & 11u) == 11u;
        f = and(f, and(and(P.fmt == 1u, P.ch == 4u), and(L.fmt == 1u, L.ch == 4u)));
        f = and(f, and(and(mv.off != 0u, mv.fmt == 1u), mv.ch == 4u));
        f = and(f, and(and(th.off != 0u, tw.off != 0u), and(ah.off != 0u, aw.off != 0u)));
        f = and(f, and(and(tw.fmt == 1u, tw.ch == 4u), and(aw.fmt == 1u, aw.ch == 1u)));
        f = and(f, and(and(ah.ch == 1u, G.fmt == 2u), and(G.ch == 4u, and(Ar.fmt == 3u, PU(36) == 3u))));
        f = and(f, and(and(O.fmt == 1u, O.ch == 4u), or((flags & 16u) == 0u, and(AO.fmt == 0u, AO.ch == 1u))));
#endif
        bad = or(bad, !f);
    }
#endif
    U.ok = !bad;
    return U;
}

// DECODE of rr_resolve.hlsl for the three channels: (exp2((c'*K8)*K9) - 1) * K12. The exp2 of an input the two fast
// tiers leave undecided (1314 f32 inputs in all) is EXP2_CR itself, at one call site for the three channels.
float rf_dec_t(float c, RfU U)
{
    bool nc = nanf(c);
    float mn = select(or(nc, nanf(U.K10)), asfloat(0x7fc00000u), min(c, U.K10));
    float cp = select(or(c > U.K5, nc), mn, U.K5);
    return fence(cp * U.K8) * RR_K9;
}
float3 rf_decode3(float3 c, RfU U)
{
    float3 t = float3(rf_dec_t(c.x, U), rf_dec_t(c.y, U), rf_dec_t(c.z, U)), e;
    bool3 hd = bool3(false, false, false);
    [unroll] for (uint k = 0; k < 3u; ++k) {
        bool und;
        e[k] = rf_exp2_t1(t[k], und);
        if (und) { bool h; e[k] = rf_exp2_core(t[k], h); hd[k] = h; }
    }
    if (any(hd)) rf_exp2_fix3(t, e, hd);
    return float3((e.x + (-1.0f)) * U.K12, (e.y + (-1.0f)) * U.K12, (e.z + (-1.0f)) * U.K12);
}

// ---- the kernel ----------------------------------------------------------------------------------------------------
#if RF_SPEC
#define RF_RD4(r, x, y) rf_h4i(r, x, y)                // bilinear texels (indices inside the resource)
#define RF_RD1(r, x, y) rf_h1i(r, x, y)
#else
#define RF_RD4(r, x, y) rf_texel(r, x, y, tiny)
#define RF_RD1(r, x, y) rf_texel1(r, x, y, 0u, tiny)
#endif

// Tile geometry (uniform): the tile's pixels, its Q footprint and its render footprint.
struct RfG { uint tx0, ty0, xlast, ylast, nqc, nqr, nrx, nry; int qc0, qr0, rx0, ry0; bool fits; };
RfG rf_geom(RfU U, uint tile, bool need_r, bool rfl = true)
{
    RfG g;
    g.tx0 = (tile % U.ntx) * RF_T; g.ty0 = (tile / U.ntx) * RF_T;
    g.xlast = min(g.tx0 + RF_T - 1u, U.We - 1u); g.ylast = min(g.ty0 + RF_T - 1u, U.He - 1u);
    g.qc0 = max((int)((6u * g.tx0 + 3u) >> 2) - 2, 0);
    int qc1 = min((int)((6u * g.xlast + 3u) >> 2) + 2, (int)U.Wh - 1);
    g.qr0 = max((int)((6u * g.ty0 + 3u) >> 2) - 2, 0);
    int qr1 = min((int)((6u * g.ylast + 3u) >> 2) + 2, (int)U.Hh - 1);
    g.nqc = (uint)(qc1 - g.qc0 + 1); g.nqr = (uint)(qr1 - g.qr0 + 1);
    int rx0 = 0, rx1 = -1, ry0 = 0, ry1 = -1;
    if (need_r) {
        float xa = (float)g.tx0 + 0.5f, xb = (float)g.xlast + 0.5f, ya = (float)g.ty0 + 0.5f, yb = (float)g.ylast + 0.5f;
        int kxa = (int)rf_rnd(mad(xa, U.sxr, -0.5f) + U.jx), kxb = (int)rf_rnd(mad(xb, U.sxr, -0.5f) + U.jx);
        int kya = (int)rf_rnd(mad(ya, U.syr, -0.5f) + U.jy), kyb = (int)rf_rnd(mad(yb, U.syr, -0.5f) + U.jy);
        int fxa = (int)floor(mad(xa, U.sxr, U.jx) + (-0.5f)), fxb = (int)floor(mad(xb, U.sxr, U.jx) + (-0.5f));
        int fya = (int)floor(mad(ya, U.syr, U.jy) + (-0.5f)), fyb = (int)floor(mad(yb, U.syr, U.jy) + (-0.5f));
        rx0 = clamp(min(fxa - 1, kxa - 1), 0, U.Wr - 1); rx1 = clamp(max(fxb + 2, kxb + 1), 0, U.Wr - 1);
        ry0 = clamp(min(fya - 1, kya - 1), 0, U.Hr - 1); ry1 = clamp(max(fyb + 2, kyb + 1), 0, U.Hr - 1);
        if (rfl) { rx0 = WaveReadLaneFirst(rx0); rx1 = WaveReadLaneFirst(rx1); ry0 = WaveReadLaneFirst(ry0); ry1 = WaveReadLaneFirst(ry1); }
    }
    g.rx0 = rx0; g.ry0 = ry0; g.nrx = (uint)(rx1 - rx0 + 1); g.nry = (uint)(ry1 - ry0 + 1);
    g.fits = g.nqc <= RF_QN && g.nqr <= RF_QN && g.nrx <= RF_RN && g.nry <= RF_RN;
#if (RFX_TB & 1) && RFX_RP == 16
    g.fits = g.fits && g.nrx <= 16u && g.nry <= 16u;
#endif
    return g;
}
#if RFX_TB & 1
// T1 (d): lane k of every wave holds the geometry of the group's tile number jbase + k (tile g + (jbase + k) G), packed
// in 4 VGPRs; the same rf_geom operations in another lane (no WaveReadLaneFirst). A tile that does not fit keeps its
// pixel origin and gets a dummy 1 x 1 footprint at (0, 0) (it takes the verified path; its loads stay in bounds).
uint4 rf_gpack(RfU U, uint tile, uint ntiles, bool need_r)
{
    RfG g = rf_geom(U, min(tile, ntiles - 1u), need_r, false);
    g.fits = and(g.fits, and(g.nqc != 0u, g.nqr != 0u));   // empty Q footprint: verified path (in-bounds loads)
    if (!g.fits) { g.qc0 = 0; g.qr0 = 0; g.nqc = 1u; g.nqr = 1u; g.rx0 = 0; g.ry0 = 0; g.nrx = 1u; g.nry = 1u; }
    return uint4((g.tx0 >> 4) | ((g.ty0 >> 4) << 16), (uint)g.qc0 | ((uint)g.qr0 << 16), (uint)g.rx0 | ((uint)g.ry0 << 16),
                 g.nqc | (g.nqr << 6) | (g.nrx << 12) | (g.nry << 18) | (g.fits ? (1u << 24) : 0u));
}
RfG rf_gunpack(RfU U, uint4 t, uint k)            // k wave-uniform: 4 v_readlane
{
    const uint a = WaveReadLaneAt(t.x, k), b = WaveReadLaneAt(t.y, k), c = WaveReadLaneAt(t.z, k), d = WaveReadLaneAt(t.w, k);
    RfG g;
    g.tx0 = (a & 0xffffu) * RF_T; g.ty0 = (a >> 16) * RF_T;
    g.xlast = min(g.tx0 + RF_T - 1u, U.We - 1u); g.ylast = min(g.ty0 + RF_T - 1u, U.He - 1u);
    g.qc0 = (int)(b & 0xffffu); g.qr0 = (int)(b >> 16); g.rx0 = (int)(c & 0xffffu); g.ry0 = (int)(c >> 16);
    g.nqc = d & 63u; g.nqr = (d >> 6) & 63u; g.nrx = (d >> 12) & 63u; g.nry = (d >> 18) & 63u; g.fits = ((d >> 24) & 1u) != 0u;
    return g;
}
#if RFX_TB & 16
// bit 16: 2 VGPRs per lane: rx0 | ry0 << 16 and tile x (11 bits) | tile y (11) | nrx (5) | nry (5); nrx = 0 for a tile
// that does not fit. The Q footprint is recomputed from tx0 / ty0 with rf_geom's integer formulas (SALU, uniform).
uint2 rf_gpack2(RfU U, uint tile, uint ntiles, bool need_r)
{
    RfG g = rf_geom(U, min(tile, ntiles - 1u), need_r, false);
    const uint nrx = g.fits ? g.nrx : 0u, nry = g.fits ? g.nry : 0u;
    const uint rx0 = g.fits ? (uint)g.rx0 : 0u, ry0 = g.fits ? (uint)g.ry0 : 0u;
    return uint2(rx0 | (ry0 << 16), (g.tx0 >> 4) | ((g.ty0 >> 4) << 11) | (nrx << 22) | (nry << 27));
}
RfG rf_gunpack2(RfU U, uint2 t, uint k)
{
    const uint c = WaveReadLaneAt(t.x, k), d = WaveReadLaneAt(t.y, k);
    RfG g;
    g.tx0 = (d & 2047u) * RF_T; g.ty0 = ((d >> 11) & 2047u) * RF_T;
    g.xlast = min(g.tx0 + RF_T - 1u, U.We - 1u); g.ylast = min(g.ty0 + RF_T - 1u, U.He - 1u);
    g.qc0 = max((int)((6u * g.tx0 + 3u) >> 2) - 2, 0);
    const int qc1 = min((int)((6u * g.xlast + 3u) >> 2) + 2, (int)U.Wh - 1);
    g.qr0 = max((int)((6u * g.ty0 + 3u) >> 2) - 2, 0);
    const int qr1 = min((int)((6u * g.ylast + 3u) >> 2) + 2, (int)U.Hh - 1);
    g.nqc = (uint)(qc1 - g.qc0 + 1); g.nqr = (uint)(qr1 - g.qr0 + 1);
    g.rx0 = (int)(c & 0xffffu); g.ry0 = (int)(c >> 16);
    g.nrx = (d >> 22) & 31u; g.nry = d >> 27;
    g.fits = and(g.nrx != 0u, and(g.nqc - 1u < RF_QN, g.nqr - 1u < RF_QN));   // also nqc, nqr = 0 (Q narrower than the footprint): verified path
    if (!g.fits) { g.qc0 = 0; g.qr0 = 0; g.nqc = 1u; g.nqr = 1u; g.rx0 = 0; g.ry0 = 0; g.nrx = 1u; g.nry = 1u; }
    return g;
}
#define RF_GTAB_T uint2
#define RF_GPACK rf_gpack2
#define RF_GUNPACK rf_gunpack2
#else
#define RF_GTAB_T uint4
#define RF_GPACK rf_gpack
#define RF_GUNPACK rf_gunpack
#endif
// T1 (b): render slot e (< 512) -> row e / RFX_RP, column e % RFX_RP. RFX_RP 20: q = (e * 3277) >> 16; for e < 512,
// e * 3277 / 65536 - e / 20 = e * 4 / 1310720 <= 0.0016 < 1/20, so floor gives e / 20 (exhaustive_check.py: all 512).
void rf_rsplit(uint e, out uint q, out uint r)
{
#if RFX_RP == 16
    q = e >> 4; r = e & 15u;
#else
    q = rf_m24(e, 3277u) >> 16; r = e - q * RFX_RP;
#endif
}
#endif

#if RF_SPEC
// Cross-tile prefetch (RF_SPEC): the raw words of the next tile's Q and render footprints are loaded into registers
// while the current tile's pixels are computed, then written to groupshared at the next tile's phases 1 and 3.
#define RF_QS 4u                      // Q texels per lane (29 x 29 <= 4 x 256)
#define RF_RS 2u                      // render texels per lane (20 x 20 <= 2 x 256)
struct RfPf { uint q[2u * RF_QS]; uint r[6u * RF_RS]; };
RfPf rf_prefetch(RfG g, uint lane, Res Q, Res Pr, Res Ar, Res Gr, Res L, int lbx, int lby, bool ok)
{
    // ok = false (the dispatch takes the verified path): the resources may be absent, every load reads arena word 0
    RfPf pf;
    [unroll] for (uint s = 0; s < RF_QS; ++s) {
#if RFX_TB & 1
        // T1 (a): slot s of lane l = Q-tile entry (row (l >> 5) + 8 s, column l & 31) = LDS index l + 256 s. Entries
        // outside nqc x nqr load a clamped texel of the footprint (never read). Screens: gx <= qc1 <= Wh - 1 < Q.w,
        // gy < Q.h, stride 6 (Q.ch 3), so the address is the shipped one for every used entry.
        const uint col = lane & 31u, row = (lane >> 5) + 8u * s;
        const uint gx = (uint)g.qc0 + min(col, g.nqc - 1u), gy = (uint)g.qr0 + min(row, g.nqr - 1u);
        uint a = select(ok, Q.off + rf_m24(gy, Q.pitch * 6u) + rf_m24(gx, 6u), 0u);
#else
        uint e = min(lane + s * WG_BLOCK, g.nqc * g.nqr - 1u);
        int gx = g.qc0 + (int)(e % g.nqc), gy = g.qr0 + (int)(e / g.nqc);
        uint a = select(ok, Q.off + (rf_m24((uint)min(gy, (int)Q.h - 1), Q.pitch) + (uint)min(gx, (int)Q.w - 1)) * (Q.ch * 2u), 0u);
#endif
#if RFX_QLD
        const uint2 qw = wg_ld64(a & ~3u, 0u);          // the texel's 6 bytes at byte 0 or 2 of an aligned b64
        const bool qo = (a & 2u) != 0u;
        pf.q[2u * s] = qo ? ((qw.x >> 16) | (qw.y << 16)) : qw.x;
        pf.q[2u * s + 1u] = (qo ? (qw.y >> 16) : qw.y) & 0xffffu;
#else
        pf.q[2u * s] = wg_ld16(a, 0u) | (wg_ld16(a, 2u) << 16);
        pf.q[2u * s + 1u] = wg_ld16(a, 4u);
#endif
    }
    [unroll] for (uint s2 = 0; s2 < RF_RS; ++s2) {
#if RFX_TB & 1
        uint rq, rr;                                    // T1 (b): fixed pitch; addresses clamped below (in bounds)
        rf_rsplit(lane + s2 * WG_BLOCK, rq, rr);
        int tx = g.rx0 + (int)rr, ty = g.ry0 + (int)rq;
#else
        uint e = min(lane + s2 * WG_BLOCK, g.nrx * g.nry - 1u);
        int tx = g.rx0 + (int)(e % g.nrx), ty = g.ry0 + (int)(e / g.nrx);
#endif
        uint ap = select(ok, Pr.off + (rf_m24((uint)clamp(ty, 0, (int)Pr.h - 1), Pr.pitch) + (uint)clamp(tx, 0, (int)Pr.w - 1)) * 8u, 0u);
        uint aa = select(ok, Ar.off + (rf_m24((uint)clamp(ty + Ar.oy, 0, (int)Ar.h - 1), Ar.pitch) + (uint)clamp(tx + Ar.ox, 0, (int)Ar.w - 1)) * 4u, 0u);
        uint ag = select(ok, Gr.off + (rf_m24((uint)ty, Gr.pitch) + (uint)tx) * 4u, 0u);
        uint al = select(ok, L.off + (rf_m24((uint)clamp(ty + lby, 0, (int)L.h - 1), L.pitch) + (uint)clamp(tx + lbx, 0, (int)L.w - 1)) * 8u, 0u);
#if RF_SPEC == 2
        uint2 pw = uint2(0u, 0u), lw = wg_ld64(al, 0u);   // the layer only (no alpha upscaling)
        uint aw = 0u, gw = 0u;
#else
        uint2 pw = wg_ld64(ap, 0u), lw = wg_ld64(al, 0u);
        uint aw = wg_ld32(aa, 0u), gw = wg_ld32(ag, 0u);
#endif
        pf.r[6u * s2] = pw.x; pf.r[6u * s2 + 1u] = pw.y; pf.r[6u * s2 + 2u] = aw; pf.r[6u * s2 + 3u] = gw;
        pf.r[6u * s2 + 4u] = lw.x; pf.r[6u * s2 + 5u] = lw.y;
    }
    return pf;
}
#endif

#if RFX_LR
// RFX_LR: layer-tile texel i as rf_slot (RFX_MIX) stored it without RFX_LR: E * widen(h) for rgb by the same
// mad(widen(h), E, -0) on the same operands (E = U.E, mzv), widen(h) for alpha; outside texels hold code 0 (+0, as there)
uint4 rf_lr4(uint i, float E, float mzv)
{
    const uint2 h = rf_w2[RF_QO + i];
    return uint4(asuint(mad(f16tof32(h.x), E, mzv)), asuint(mad(f16tof32(h.x >> 16), E, mzv)),
                 asuint(mad(f16tof32(h.y), E, mzv)), asuint(f16tof32(h.y >> 16)));
}
#endif
#if RF_SPEC
// One render texel of the groupshared tile from its prefetched words (RF_SPEC formats); chk: apply the outside rule
// (false when every render texel is inside P, A and L: a per-dispatch fact).
void rf_slot(uint e, int tx, int ty, uint w[6], bool chk, Res Pr, Res Ar, Res L, int lbx, int lby, float K2n, float K2x2n,
             float E, float mzv = 0.0f)
{
        bool lin3 = or(!chk, and(and(tx + lbx >= 0, ty + lby >= 0), and(tx + lbx < (int)L.w, ty + lby < (int)L.h)));
#if RFX_MIX
        // E * widen(h) as fma(widen(h), E, -0): x*E + (-0) = x*E for every product (+0 + -0 = +0, -0 + -0 = -0), one
        // rounding in both; outside texels widen code 0 (= +0.0, as select(..., 0.0f)). mzv: -0 in a VGPR.
        const uint h0 = select(lin3, w[4], 0u), h1 = select(lin3, w[5], 0u);
#if RFX_LR
        rf_w2[RF_QO + e] = uint2(h0, h1);       // the raw words; rf_lr4 forms the four values below at each read
        if (false)
#endif
        RF_LW(e, uint4(asuint(mad(f16tof32(h0), E, mzv)), asuint(mad(f16tof32(h0 >> 16), E, mzv)),
                       asuint(mad(f16tof32(h1), E, mzv)), asuint(select(lin3, f16tof32(w[5] >> 16), 0.0f))));
#else
        float4 t = float4(select(lin3, f16tof32(w[4]), 0.0f), select(lin3, f16tof32(w[4] >> 16), 0.0f),
                          select(lin3, f16tof32(w[5]), 0.0f), select(lin3, f16tof32(w[5] >> 16), 0.0f));
        RF_LW(e, uint4(asuint(E * t.x), asuint(E * t.y), asuint(E * t.z), asuint(t.w)));
#endif
#if RF_SPEC != 2
        // the outside rule of texel() (0); allin (per dispatch): every render texel is inside P, A and L
        bool pin = or(!chk, and(and(tx >= 0, ty >= 0), and(tx < (int)Pr.w, ty < (int)Pr.h)));
        bool ain = or(!chk, and(and(tx + Ar.ox >= 0, ty + Ar.oy >= 0), and(tx + Ar.ox < (int)Ar.w, ty + Ar.oy < (int)Ar.h)));
        float3 p = float3(select(pin, f16tof32(w[0]), 0.0f), select(pin, f16tof32(w[0] >> 16), 0.0f),
                          select(pin, f16tof32(w[1]), 0.0f));
        float a = select(ain, rf_cv(w[2] >> 30, 3.0f, RF_R3), 0.0f);    // RGB10A2 alpha: >= 0
        uint gw = w[3];
        uint gw0 = gw & 255u, gw1 = (gw >> 8) & 255u, gw2 = (gw >> 16) & 255u;
        // binary16 P and layer values (0 or at least 2^-24) times screened factors (0 or at least 2^-24): the
        // products are 0 or at least 2^-48, where the hardware product is M of rr_resolve_math.hlsli
        float qx = p.x * K2n, qy = p.y * K2n, qxy = p.z * K2x2n;
        uint g0 = rf_f2h(rf_cv(gw0, 255.0f, RF_R255));
        uint g1 = rf_f2h(rf_cv(gw1, 255.0f, RF_R255));
        uint g2 = rf_f2h(rf_cv(gw2, 255.0f, RF_R255));
        // not ordinary here: qx and qy of opposite signs, a non-finite q (the q and a ranges hold by the formats)
        bool nf = and(and(qx != 0.0f, qy != 0.0f), ((asuint(qx) ^ asuint(qy)) & 0x80000000u) != 0u);
        nf = or(nf, !and(and(isfinite(qx), isfinite(qy)), isfinite(qxy)));
        rf_u4[e] = uint4(asuint(qx), asuint(qy), asuint(qxy), asuint(a));
        rf_g[e] = uint2(g0 | (g1 << 16), g2 | select(nf, 0x10000u, 0u));
#endif
}
#endif

#if RFX_TB & 1
// Phase 2 at Q pitch RF_QP; inter (compile-time at each call): T5, the tile's Q footprint is interior, so
// clamp(cx + k - 2, 0, Wh - 1) = cx + k - 2 for every column (cx monotone in x; cx(tx0) >= 2, cx(xlast) + 2 <= Wh - 1).
void rf_hpass(uint lane, uint tx0, uint xlast, uint nqr, int qc0, uint Wh, half we[5], half2 hz2, bool inter)
{
    for (uint e = lane; e < (RF_T / 2u) * nqr; e += WG_BLOCK) {
        uint xa = min(tx0 + 2u * (e % (RF_T / 2u)), xlast), xb = min(xa + 1u, xlast), rl = e / (RF_T / 2u);
        int cxa = (int)((6u * xa + 3u) >> 2), cxb = (int)((6u * xb + 3u) >> 2);
        bool odda = (xa & 1u) != 0u, oddb = (xb & 1u) != 0u;
        half2 acca = hz2, accb = hz2, accc = hz2;
        half2 mna = half2(65504.0h, 65504.0h), mxa = half2(-65504.0h, -65504.0h), mnb = mna, mxb = mxa, mnc = mna, mxc = mxa;
        const uint ba = rl * RF_QP + (uint)(cxa - 2 - qc0), bb = rl * RF_QP + (uint)(cxb - 2 - qc0);
        [unroll] for (uint k = 0; k < 5u; ++k) {
            uint ia = inter ? ba + k : rl * RF_QP + (uint)(clamp(cxa + (int)k - 2, 0, (int)Wh - 1) - qc0);
            uint ib = inter ? bb + k : rl * RF_QP + (uint)(clamp(cxb + (int)k - 2, 0, (int)Wh - 1) - qc0);
            uint2 qa = RF_QR(ia);
            uint2 qb = RF_QR(ib);
            half2 ta = wg_as_h2(qa.x), tb = wg_as_h2(qb.x), tc = wg_as_h2((qa.y & 0xffffu) | (qb.y << 16));
            half wa = select(odda, we[4u - k], we[k]), wb = select(oddb, we[4u - k], we[k]);
            acca = mad(half2(wa, wa), ta, acca); accb = mad(half2(wb, wb), tb, accb); accc = mad(half2(wa, wb), tc, accc);
            mna = min(mna, ta); mxa = max(mxa, ta); mnb = min(mnb, tb); mxb = max(mxb, tb);
            mnc = min(mnc, tc); mxc = max(mxc, tc);
        }
        acca = min(max(acca, mna), mxa); accb = min(max(accb, mnb), mxb); accc = min(max(accc, mnc), mxc);
        uint cb = wg_h2_bits(accc), ra = rl * RF_T + (xa - tx0);
        rf_hr[ra] = uint2(wg_h2_bits(acca), cb & 0xffffu);
        rf_hr[ra + 1u] = uint2(wg_h2_bits(accb), cb >> 16);
    }
}
// The vertical pass of phase 4 (the shipped code); inter: T5, clamp(cy + j - 2, 0, Hh - 1) = cy + j - 2 for the tile.
float3 rf_vpass(int y, uint xc, int qr0, uint Hh, half we[5], half2 hz2, half hz, bool inter)
{
    int cy = (int)((6u * (uint)y + 3u) >> 2);
    bool oddy = (y & 1) != 0;
    half2 acc = hz2, mn2 = half2(65504.0h, 65504.0h), mx2 = half2(-65504.0h, -65504.0h);
    half accb = hz, mnb = 65504.0h, mxb = -65504.0h;
    const uint rb = (uint)(cy - 2 - qr0);
    [unroll] for (uint j = 0; j < 5u; ++j) {
        uint rl = inter ? rb + j : (uint)(clamp(cy + (int)j - 2, 0, (int)Hh - 1) - qr0);
        uint2 hv2 = rf_hr[rl * RF_T + xc];
        half2 hrg = wg_as_h2(hv2.x);
        half hbb = asfloat16((uint16_t)(hv2.y & 0xffffu));
        half wv = select(oddy, we[4u - j], we[j]);
        acc = mad(half2(wv, wv), hrg, acc); accb = mad(wv, hbb, accb);
        mn2 = min(mn2, hrg); mx2 = max(mx2, hrg); mnb = min(mnb, hbb); mxb = max(mxb, hbb);
    }
    acc = min(max(acc, mn2), mx2); accb = min(max(accb, mnb), mxb);
    return float3(acc.x, acc.y, accb);
}
#endif

WG_KERNEL(rr_resolve_fast)
{
    WG_SETUP();
    rf_load_table(PU(38));
    RfU U = rf_uniforms();
    const uint lane = wg_lane_id(), flags = U.flags;
    const bool ext = RF_SPEC || (flags & 1u) != 0u;
    const bool lay = RF_SPEC || (flags & 3u) == 3u, alp = RF_SPEC == 1 || (RF_SPEC == 0 && (flags & 9u) == 9u);
    const bool lay3 = lay && (flags & 32u) == 0u, alp3 = alp && (flags & 64u) == 0u;
    const bool lay3c = RF_SPEC || lay3, alp3c = RF_SPEC == 1 || alp3;    // computed (RF_SPEC: always, selected below)
    const uint ntiles = U.ntx * ((U.He + RF_T - 1u) / RF_T);
    half we[5];
    [unroll] for (uint k = 0; k < 5u; ++k) we[k] = asfloat16((uint16_t)(PU(17u + k) & 0xffffu));
    const uint zb = 0u - (wg_grid_ >> 31);        // 0 (the grid is below 2^31), opaque to the compiler
    const half2 hz2 = wg_as_h2(zb);                 // +0 the compiler cannot see (no mad(a, b, 0) -> a * b)
    const half hz = hz2.x;
    const float fz = asfloat(zb);
#if RFX_MIX
    const float mzv = asfloat(0x80000000u ^ (WaveGetLaneIndex() & zb));   // RFX_MIX: -0 in a VGPR (one SGPR operand)
    rf_mz = asfloat(0x80000000u ^ zb);
#else
    const float mzv = 0.0f;                             // unused
#endif
    Res Q = res(40u), Pr = res(96u), Gr = res(104u), Ar = res(80u), L = res(64u);
    Res mvr = res(88u), TH = res(112u), AH = res(144u);
    const Res THW = res(120u), AHW = res(152u), OUTR = res(48u), AOR = res(56u);   // read once, not per tile
    const int Wo = (int)PU(6), Ho = (int)PU(7);
#if RF_SPEC == 2
    const bool gst = PU(16) == 1u;
    const uint2 gorg = uint2(PU(39) & 0xffffu, PU(39) >> 16);
#endif
    const uint bmap = PU(37);
    const uint ach = PU(36);
    const int lbx = (int)PU(13), lby = (int)PU(14);
    const bool need_r = ext && (alp || lay3c);      // render tile used

#if RF_SPEC
#if RF_SPEC == 2
    const bool allin = and(and(lbx >= 0, lby >= 0), and(lbx + U.Wr <= (int)L.w, lby + U.Hr <= (int)L.h));
#else
    const bool allin = and(and(and(Pr.w >= (uint)U.Wr, Pr.h >= (uint)U.Hr), and(Ar.ox >= 0, Ar.oy >= 0)),
                           and(and(Ar.ox + U.Wr <= (int)Ar.w, Ar.oy + U.Hr <= (int)Ar.h),
                               and(and(lbx >= 0, lby >= 0), and(lbx + U.Wr <= (int)L.w, lby + U.Hr <= (int)L.h))));
#endif
#if RFX_TB & 1
    RF_GTAB_T gtab = RF_GPACK(U, wg_block_id() + WaveGetLaneIndex() * wg_grid_size(), ntiles, need_r);
    uint jn = 0u;
    RfG gnext = RF_GUNPACK(U, gtab, 0u);
#else
    RfG gnext = rf_geom(U, wg_block_id(), need_r);
#endif
    RfPf pf = rf_prefetch(gnext, lane, Q, Pr, Ar, Gr, L, lbx, lby, U.ok);
#endif
#if RF_TIMING
    uint tck[8] = { 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u };
    uint tc0 = wg_ags_call(0x1Fu, 0u, 0u, 0u, 0u);
#define RF_TICK(k) { uint tc1 = wg_ags_call(0x1Fu, 0u, 0u, 0u, 0u); tck[k] += tc1 - tc0; tc0 = tc1; }
#else
#define RF_TICK(k)
#endif
    for (uint tile = wg_block_id(); tile < ntiles; tile += wg_grid_size()) {
#if RF_SPEC
        const RfG G = gnext;
#else
        const RfG G = rf_geom(U, tile, need_r);
#endif
        const uint tx0 = G.tx0, ty0 = G.ty0, xlast = G.xlast, ylast = G.ylast;
        const uint xl = lane % RF_T, yl = lane / RF_T;
        const bool own = tx0 + xl < U.We && ty0 + yl < U.He;
        const int x = (int)min(tx0 + xl, xlast), y = (int)min(ty0 + yl, ylast);
        const int qc0 = G.qc0, qr0 = G.qr0, rx0 = G.rx0, ry0 = G.ry0;
        const int rx1 = rx0 + (int)G.nrx - 1, ry1 = ry0 + (int)G.nry - 1;
        const uint nqc = G.nqc, nqr = G.nqr, nrx = G.nrx, nry = G.nry;
        const bool tok = U.ok && G.fits;
        bool slow = !tok;
#if RFX_TB & 2
        // T5: wave-uniform interior tests on the tile's extreme indices (SALU)
        const bool intQ = and(and((int)((6u * tx0 + 3u) >> 2) >= 2, (int)((6u * xlast + 3u) >> 2) + 2 <= (int)U.Wh - 1),
                              and((int)((6u * ty0 + 3u) >> 2) >= 2, (int)((6u * ylast + 3u) >> 2) + 2 <= (int)U.Hh - 1));
        const bool intR = and(and(rx0 > 0, rx1 < U.Wr - 1), and(ry0 > 0, ry1 < U.Hr - 1));
#else
        const bool intQ = false, intR = false;
#endif
#if RFX_TB & 1
        const uint rp = RFX_RP;                         // render-tile LDS pitch
#else
        const uint rp = nrx;
#endif
        float4 lin = float4(0.0f, 0.0f, 0.0f, 0.0f);
        float oaSep = 0.0f;
#if RF_HOIST
        // the per-pixel global reads of phase 4 (the same code, moved): motion vector and layer taps here, the history
        // taps once the motion vector is in (after the Q pass), so their latency runs under phases 1-3
        const float xf = (float)x, yf = (float)y, xh = xf + 0.5f, yh = yf + 0.5f;
        int kx = 0, ky = 0;
        float mxo = 0.0f, myo = 0.0f;
        float4 mv = 0;
#if RF_HOIST == 1
        float4 l00 = 0, l10 = 0, l01 = 0, l11 = 0;
        RfBil lb = (RfBil)0;
        float w = 0.0f, w1 = 0.0f;
        RfBil hb = (RfBil)0;
        float4 h00 = 0, h10 = 0, h01 = 0, h11 = 0;
        float a00 = 0, a10 = 0, a01 = 0, a11 = 0;
#endif
        if (tok && ext) {
            kx = (int)rf_rnd(mad(xh, U.sxr, -0.5f) + U.jx); ky = (int)rf_rnd(mad(yh, U.syr, -0.5f) + U.jy);
            if (mvr.off != 0u) mv = rf_h4(mvr, clamp(kx, 0, U.Wr - 1) + mvr.ox, clamp(ky, 0, U.Hr - 1) + mvr.oy);
#if RF_HOIST == 1
            if (lay) {
                const float a = xf + U.jxp, b = yf + U.jyp;
                const bool sflag = and(abs(rf_rnd(a) + (-a)) < 0.5f, abs(rf_rnd(b) + (-b)) < 0.5f);
                w = select(sflag, U.TBW, U.TBW2); w1 = select(sflag, U.w1a, U.w1b);
                const float u = (xh + U.jxp) * U.rWe, v = (yh + U.jyp) * U.rHe;
                lb = rf_bil(u, v, L.w, L.h);
                l00 = RF_RD4(L, lb.xa, lb.ya); l10 = RF_RD4(L, lb.xb, lb.ya);
                l01 = RF_RD4(L, lb.xa, lb.yb); l11 = RF_RD4(L, lb.xb, lb.yb);
            }
#endif
        }
#endif
        if (tok) {
        // ---- 1. Q tile
#if RF_SPEC
        [unroll] for (uint s1 = 0; s1 < RF_QS; ++s1) {
#if RFX_TB & 1
            RF_QW(lane + s1 * WG_BLOCK, uint4(pf.q[2u * s1], pf.q[2u * s1 + 1u], 0u, 0u));   // T1: qin true (screens)
#else
            uint e = lane + s1 * WG_BLOCK;
            int gx = qc0 + (int)(e % nqc), gy = qr0 + (int)(e / nqc);
            bool qin = and(gx < (int)Q.w, gy < (int)Q.h);
            if (e < nqc * nqr) RF_QW(e, uint4(select(qin, pf.q[2u * s1], 0u), select(qin, pf.q[2u * s1 + 1u], 0u), 0u, 0u));
#endif
        }
#else
        for (uint e = lane; e < nqc * nqr * ((RF_PROF & 16) ? 0u : 1u); e += WG_BLOCK) {
            int gx = qc0 + (int)(e % nqc), gy = qr0 + (int)(e / nqc);
            bool qin = and(gx < (int)Q.w, gy < (int)Q.h);
            uint a = Q.off + ((uint)min(gy, (int)Q.h - 1) * Q.pitch + (uint)min(gx, (int)Q.w - 1)) * Q.ch * 2u;
            uint rg = wg_ld16(a, 0u) | (wg_ld16(a, 2u) << 16), bb = wg_ld16(a, 4u);
            RF_QW(e, uint4(select(qin, rg, 0u), select(qin, bb, 0u), 0u, 0u));
        }
#endif
        GroupMemoryBarrierWithGroupSync();
#if RF_HOIST == 1
        {
            if (ext && mvr.off != 0u) { mxo = mv.x * U.sxh; myo = mv.y * U.syh; }
            const float hu = ((xf + mxo) + 0.5f) * U.rWe, hv = ((yf + myo) + 0.5f) * U.rHe;
            hb = rf_bil(hu, hv, select(lay3c, TH.w, AH.w), select(lay3c, TH.h, AH.h));
            if (ext && lay && lay3c) {
                h00 = RF_RD4(TH, hb.xa, hb.ya); h10 = RF_RD4(TH, hb.xb, hb.ya);
                h01 = RF_RD4(TH, hb.xa, hb.yb); h11 = RF_RD4(TH, hb.xb, hb.yb);
            }
            if (ext && alp && alp3c) {
                a00 = RF_RD1(AH, hb.xa, hb.ya); a10 = RF_RD1(AH, hb.xb, hb.ya);
                a01 = RF_RD1(AH, hb.xa, hb.yb); a11 = RF_RD1(AH, hb.xb, hb.yb);
            }
        }
#endif
        RF_TICK(0);
        // ---- 2. horizontal pass (54b.1 steps 3/4 along x), one entry per (Q row, output column); a lane does the
        // columns 2i, 2i + 1 of one row with packed binary16 (the two b chains share one half2)
#if RFX_TB & 1
        if (intQ) rf_hpass(lane, tx0, xlast, nqr, qc0, U.Wh, we, hz2, true);
        else rf_hpass(lane, tx0, xlast, nqr, qc0, U.Wh, we, hz2, false);
#else
        for (uint e = lane; e < (RF_T / 2u) * nqr * ((RF_PROF & 16) ? 0u : 1u); e += WG_BLOCK) {
            uint xa = min(tx0 + 2u * (e % (RF_T / 2u)), xlast), xb = min(xa + 1u, xlast), rl = e / (RF_T / 2u);
            int cxa = (int)((6u * xa + 3u) >> 2), cxb = (int)((6u * xb + 3u) >> 2);
            bool odda = (xa & 1u) != 0u, oddb = (xb & 1u) != 0u;
            half2 acca = hz2, accb = hz2, accc = hz2;
            half2 mna = half2(65504.0h, 65504.0h), mxa = half2(-65504.0h, -65504.0h), mnb = mna, mxb = mxa, mnc = mna, mxc = mxa;
            [unroll] for (uint k = 0; k < 5u; ++k) {
                uint2 qa = RF_QR(rl * nqc + (uint)(clamp(cxa + (int)k - 2, 0, (int)U.Wh - 1) - qc0));
                uint2 qb = RF_QR(rl * nqc + (uint)(clamp(cxb + (int)k - 2, 0, (int)U.Wh - 1) - qc0));
                half2 ta = wg_as_h2(qa.x), tb = wg_as_h2(qb.x), tc = wg_as_h2((qa.y & 0xffffu) | (qb.y << 16));
                half wa = select(odda, we[4u - k], we[k]), wb = select(oddb, we[4u - k], we[k]);
                acca = mad(half2(wa, wa), ta, acca); accb = mad(half2(wb, wb), tb, accb); accc = mad(half2(wa, wb), tc, accc);
                mna = min(mna, ta); mxa = max(mxa, ta); mnb = min(mnb, tb); mxb = max(mxb, tb);
                mnc = min(mnc, tc); mxc = max(mxc, tc);
            }
            acca = min(max(acca, mna), mxa); accb = min(max(accb, mnb), mxb); accc = min(max(accc, mnc), mxc);
            uint cb = wg_h2_bits(accc), ra = rl * RF_T + (xa - tx0);
            rf_hr[ra] = uint2(wg_h2_bits(acca), cb & 0xffffu);
            rf_hr[ra + 1u] = uint2(wg_h2_bits(accb), cb >> 16);
        }
#endif
#if !(RFX_TB & 4)
        GroupMemoryBarrierWithGroupSync();
#endif
        RF_TICK(1);
        // ---- 3. render tile (overwrites the Q tile): per texel the verified helpers products
        if (need_r && (RF_PROF & 8) == 0) {
            const float K2n = U.K2n, K2x2n = U.K2x2n, E = U.E;
#if RF_SPEC
            [unroll] for (uint s3 = 0; s3 < RF_RS; ++s3) {
                uint e = lane + s3 * WG_BLOCK;
#if RFX_TB & 1
                if (e < (nry - 1u) * RFX_RP + nrx) {        // T1: the used part of the pitch-RFX_RP tile
                uint rq, rr;
                rf_rsplit(e, rq, rr);
                int tx = rx0 + (int)rr, ty = ry0 + (int)rq;
#else
                if (e < nrx * nry) {                        // the second slot: whole waves skip it (<= 256 texels)
                int tx = rx0 + (int)(e % nrx), ty = ry0 + (int)(e / nrx);
#endif
                uint w6[6] = { pf.r[6u * s3], pf.r[6u * s3 + 1u], pf.r[6u * s3 + 2u], pf.r[6u * s3 + 3u], pf.r[6u * s3 + 4u], pf.r[6u * s3 + 5u] };
                if (allin) rf_slot(e, tx, ty, w6, false, Pr, Ar, L, lbx, lby, K2n, K2x2n, E, mzv);   // no outside texels
                else rf_slot(e, tx, ty, w6, true, Pr, Ar, L, lbx, lby, K2n, K2x2n, E, mzv);
                }
            }
#else
            for (uint e = lane; e < nrx * nry; e += WG_BLOCK) {
                int tx = rx0 + (int)(e % nrx), ty = ry0 + (int)(e / nrx);
                bool nf = false;
                if (alp) {
                    float3 p = rf_texel(Pr, tx, ty, nf).xyz;
                    float a = rf_texel1(Ar, tx + Ar.ox, ty + Ar.oy, ach, nf);
                    uint gb = ((uint)ty * Gr.pitch + (uint)tx) * Gr.ch, gw0, gw1, gw2;
                    if (Gr.ch == 4u) {                          // one load (same bytes as wg_ld8 x 3)
                        uint gw = wg_ld32(Gr.off, gb);
                        gw0 = gw & 255u; gw1 = (gw >> 8) & 255u; gw2 = (gw >> 16) & 255u;
                    } else {
                        gw0 = wg_ld8(Gr.off, gb); gw1 = wg_ld8(Gr.off, gb + 1u); gw2 = wg_ld8(Gr.off, gb + 2u);
                    }
                    float qx = M(p.x, K2n), qy = M(p.y, K2n), qxy = M(p.z, K2x2n);
                    uint g0 = rf_f2h(rf_cv(gw0, 255.0f, RF_R255));
                    uint g1 = rf_f2h(rf_cv(gw1, 255.0f, RF_R255));
                    uint g2 = rf_f2h(rf_cv(gw2, 255.0f, RF_R255));
                    // not ordinary: q below 2^-48 (nonzero), qx and qy of opposite signs, a negative or (f32) below
                    // 2^-24 (RESOLVE-FAST.md)
                    const float T48 = asfloat(0x27800000u);
                    nf = or(nf, or(or(and(qx != 0.0f, abs(qx) < T48), and(qy != 0.0f, abs(qy) < T48)),
                                     or(and(qxy != 0.0f, abs(qxy) < T48), a < 0.0f)));
                    nf = or(nf, and(and(qx != 0.0f, qy != 0.0f), ((asuint(qx) ^ asuint(qy)) & 0x80000000u) != 0u));
                    nf = or(nf, !and(and(isfinite(qx), isfinite(qy)), isfinite(qxy)));
                    rf_u4[e] = uint4(asuint(qx), asuint(qy), asuint(qxy), asuint(a));
                    rf_g[e] = uint2(g0 | (g1 << 16), g2 | select(nf, 0x10000u, 0u));
                }
                if (lay3c) {
                    bool lt = false;
                    float4 t = rf_texel(L, tx + lbx, ty + lby, lt);
                    rf_u4[RF_NR + e] = uint4(asuint(M(E, t.x)), asuint(M(E, t.y)), asuint(M(E, t.z)), asuint(t.w));
                }
            }
#endif
        }
        }   // tok (phases 1-3)
        GroupMemoryBarrierWithGroupSync();
#if RF_SPEC
        // the next tile raw words (unconditional: no register copies), in flight during the pixels below
#if RFX_TB & 1
        jn += 1u;                                       // the next tile is the group's tile number jn
        if ((jn & 63u) == 0u) gtab = RF_GPACK(U, tile + wg_grid_size() + WaveGetLaneIndex() * wg_grid_size(), ntiles, need_r);
        gnext = RF_GUNPACK(U, gtab, jn & 63u);
#else
        gnext = rf_geom(U, min(tile + wg_grid_size(), ntiles - 1u), need_r);
#endif
        pf = rf_prefetch(gnext, lane, Q, Pr, Ar, Gr, L, lbx, lby, U.ok);
#endif
        RF_TICK(2);
        if (tok) {
        // ---- 4. per pixel
        bool tiny = false;
#if RF_HOIST == 2
        float mxo = 0.0f, myo = 0.0f;
        float4 l00 = 0, l10 = 0, l01 = 0, l11 = 0;
        RfBil lb = (RfBil)0;
        float w = 0.0f, w1 = 0.0f;
        if (ext) {                                       // kx, ky and the motion vector read at the tile start
            if (mvr.off != 0u) { mxo = mv.x * U.sxh; myo = mv.y * U.syh; }
            if (lay) {
                const float a = xf + U.jxp, b = yf + U.jyp;
                const bool sflag = and(abs(rf_rnd(a) + (-a)) < 0.5f, abs(rf_rnd(b) + (-b)) < 0.5f);
                w = select(sflag, U.TBW, U.TBW2); w1 = select(sflag, U.w1a, U.w1b);
                const float u = (xh + U.jxp) * U.rWe, v = (yh + U.jyp) * U.rHe;
                lb = rf_bil(u, v, L.w, L.h);
                l00 = RF_RD4(L, lb.xa, lb.ya); l10 = RF_RD4(L, lb.xb, lb.ya);
                l01 = RF_RD4(L, lb.xa, lb.yb); l11 = RF_RD4(L, lb.xb, lb.yb);
            }
        }
        const float hu = ((xf + mxo) + 0.5f) * U.rWe, hv = ((yf + myo) + 0.5f) * U.rHe;
        RfBil hb = rf_bil(hu, hv, select(lay3c, TH.w, AH.w), select(lay3c, TH.h, AH.h));
        float4 h00 = 0, h10 = 0, h01 = 0, h11 = 0;
        float a00 = 0, a10 = 0, a01 = 0, a11 = 0;
        if (ext && lay && lay3c) {
            h00 = RF_RD4(TH, hb.xa, hb.ya); h10 = RF_RD4(TH, hb.xb, hb.ya);
            h01 = RF_RD4(TH, hb.xa, hb.yb); h11 = RF_RD4(TH, hb.xb, hb.yb);
        }
        if (ext && alp && alp3c) {
            a00 = RF_RD1(AH, hb.xa, hb.ya); a10 = RF_RD1(AH, hb.xb, hb.ya);
            a01 = RF_RD1(AH, hb.xa, hb.yb); a11 = RF_RD1(AH, hb.xb, hb.yb);
        }
#elif !RF_HOIST
        const float xf = (float)x, yf = (float)y, xh = xf + 0.5f, yh = yf + 0.5f;
        int kx = 0, ky = 0;
        float mxo = 0.0f, myo = 0.0f;
        float4 l00 = 0, l10 = 0, l01 = 0, l11 = 0;
        RfBil lb = (RfBil)0;
        float w = 0.0f, w1 = 0.0f;
        if (ext) {                                       // the global reads first (their latency under the work below)
            kx = (int)rf_rnd(mad(xh, U.sxr, -0.5f) + U.jx); ky = (int)rf_rnd(mad(yh, U.syr, -0.5f) + U.jy);
            if (mvr.off != 0u) {
#if RF_SPEC
                float4 mv = rf_h4(mvr, clamp(kx, 0, U.Wr - 1) + mvr.ox, clamp(ky, 0, U.Hr - 1) + mvr.oy);
#else
                float4 mv = rf_texel(mvr, clamp(kx, 0, U.Wr - 1) + mvr.ox, clamp(ky, 0, U.Hr - 1) + mvr.oy, tiny);
#endif
                mxo = mv.x * U.sxh; myo = mv.y * U.syh;
            }
            if (lay) {
                const float a = xf + U.jxp, b = yf + U.jyp;
                const bool sflag = and(abs(rf_rnd(a) + (-a)) < 0.5f, abs(rf_rnd(b) + (-b)) < 0.5f);
                w = select(sflag, U.TBW, U.TBW2); w1 = select(sflag, U.w1a, U.w1b);
                const float u = (xh + U.jxp) * U.rWe, v = (yh + U.jyp) * U.rHe;
                lb = rf_bil(u, v, L.w, L.h);
#if RF_EARLY
                l00 = RF_RD4(L, lb.xa, lb.ya); l10 = RF_RD4(L, lb.xb, lb.ya);
                l01 = RF_RD4(L, lb.xa, lb.yb); l11 = RF_RD4(L, lb.xb, lb.yb);
#endif
            }
        }
        const float hu = ((xf + mxo) + 0.5f) * U.rWe, hv = ((yf + myo) + 0.5f) * U.rHe;
        RfBil hb = rf_bil(hu, hv, select(lay3c, TH.w, AH.w), select(lay3c, TH.h, AH.h));
        float4 h00 = 0, h10 = 0, h01 = 0, h11 = 0;
        float a00 = 0, a10 = 0, a01 = 0, a11 = 0;
#if RF_EARLY
        if (ext && lay && lay3c) {
            h00 = RF_RD4(TH, hb.xa, hb.ya); h10 = RF_RD4(TH, hb.xb, hb.ya);
            h01 = RF_RD4(TH, hb.xa, hb.yb); h11 = RF_RD4(TH, hb.xb, hb.yb);
        }
        if (ext && alp && alp3c) {
            a00 = RF_RD1(AH, hb.xa, hb.ya); a10 = RF_RD1(AH, hb.xb, hb.ya);
            a01 = RF_RD1(AH, hb.xa, hb.yb); a11 = RF_RD1(AH, hb.xb, hb.yb);
        }
#endif
#endif   // !RF_HOIST
        RF_TICK(3);
        // plain colour: vertical pass over the shared horizontal results, decode
        const uint xc = (uint)x - tx0;
        float3 col;
#if RFX_TB & 1
        {
            float3 v3;
            if (intQ) v3 = rf_vpass(y, xc, qr0, U.Hh, we, hz2, hz, true);
            else v3 = rf_vpass(y, xc, qr0, U.Hh, we, hz2, hz, false);
            col = (RF_PROF & 4) ? v3 : rf_decode3(v3, U);
        }
        if (false)
#endif
        {
            int cy = (int)((6u * (uint)y + 3u) >> 2);
            bool oddy = (y & 1) != 0;
            half2 acc = hz2, mn2 = half2(65504.0h, 65504.0h), mx2 = half2(-65504.0h, -65504.0h);
            half accb = hz, mnb = 65504.0h, mxb = -65504.0h;
            [unroll] for (uint j = 0; j < 5u; ++j) {
                uint rl = (uint)(clamp(cy + (int)j - 2, 0, (int)U.Hh - 1) - qr0);
                uint2 hv2 = rf_hr[rl * RF_T + xc];
                half2 hrg = wg_as_h2(hv2.x);
                half hbb = asfloat16((uint16_t)(hv2.y & 0xffffu));
                half wv = select(oddy, we[4u - j], we[j]);
                acc = mad(half2(wv, wv), hrg, acc); accb = mad(wv, hbb, accb);
                mn2 = min(mn2, hrg); mx2 = max(mx2, hrg); mnb = min(mnb, hbb); mxb = max(mxb, hbb);
            }
            acc = min(max(acc, mn2), mx2); accb = min(max(accb, mnb), mxb);
            col = (RF_PROF & 4) ? float3(acc.x, acc.y, accb) : rf_decode3(float3(acc.x, acc.y, accb), U);
        }
        float alphaC = 1.0f;
        if (ext) {
            // T5 (intR): kx in [kxa, kxb] (the same monotone expression as rf_geom's), rx0 = min(fxa - 1, kxa - 1) > 0 and
            // rx1 = max(fxb + 2, kxb + 1) < Wr - 1 (neither clamp bound), so the clamps are identities and the window
            // lies inside the render footprint: no check
            if ((lay3c || alp3c) && !intR) {
                int lo = clamp(kx - 1, 0, U.Wr - 1), hi = clamp(kx + 1, 0, U.Wr - 1);
                int lo2 = clamp(ky - 1, 0, U.Hr - 1), hi2 = clamp(ky + 1, 0, U.Hr - 1);
                slow = or(slow, or(or(lo < rx0, hi > rx1), or(lo2 < ry0, hi2 > ry1)));
            }
            int o3x[3], o3y[3];
            [unroll] for (int k3 = 0; k3 < 3; ++k3) {
                o3x[k3] = clamp(kx + k3 - 1, 0, U.Wr - 1) - rx0;
                o3y[k3] = (clamp(ky + k3 - 1, 0, U.Hr - 1) - ry0) * (int)rp;
            }
        RF_TICK(4);
            // ---- alpha upscaling (54b.4), taps first: their work hides the history reads
            float ah = 0.0f, oa = 0.0f;
            if (alp && (RF_PROF & 1) == 0) {
                const float px = mad(xh, U.sxr, U.jx), py = mad(yh, U.syr, U.jy);
                const float fxf = floor(px + (-0.5f)), fyf = floor(py + (-0.5f));
                const float dx0 = (fxf + 0.5f) + (-px), dy0 = (fyf + 0.5f) + (-py);
                const int fxi = (int)fxf, fyi = (int)fyf;
                {
                    int lo = clamp(fxi - 1, 0, U.Wr - 1), hi = clamp(fxi + 2, 0, U.Wr - 1);
                    int lo2 = clamp(fyi - 1, 0, U.Hr - 1), hi2 = clamp(fyi + 2, 0, U.Hr - 1);
                    slow = or(slow, or(or(lo < rx0, hi > rx1), or(lo2 < ry0, hi2 > ry1)));
                }
                float dx[4], dy[4], dxx[4], dyy[4];
                [unroll] for (int k = 0; k < 4; ++k) {
                    dx[k] = dx0 + (float)(k - 1); dy[k] = dy0 + (float)(k - 1);
                }
                [unroll] for (int k2 = 0; k2 < 4; ++k2) { dxx[k2] = fence(dx[k2] * dx[k2]); dyy[k2] = fence(dy[k2] * dy[k2]); }
                float W = fz, Aw = fz;
                float3 Gs = float3(fz, fz, fz);
                uint fl = 0u;
                int ofx[4], ofy[4];
                [unroll] for (int k3 = 0; k3 < 4; ++k3) {
                    ofx[k3] = clamp(fxi + k3 - 1, 0, U.Wr - 1) - rx0;
                    ofy[k3] = (clamp(fyi + k3 - 1, 0, U.Hr - 1) - ry0) * (int)nrx;
                }
                // two halves of 8 taps (i = 0, 1 then 2, 3; j inner: the accumulation order of 54b.4): first-tier
                // exp2 of all 8 without branches, the rare second tier, then the ordered accumulation
                [unroll] for (int hh = 0; hh < 2; ++hh) {
                    float e8[8], wt8[8], a8[8];
                    uint2 g8[8];
                    bool u8[8];
                    [unroll] for (int t = 0; t < 8; ++t) {
                        const int i = 2 * hh + t / 4, j = t % 4;
                        uint o = (uint)(ofy[j] + ofx[i]);
                        float4 qa = asfloat(rf_u4[o]);
                        g8[t] = rf_g[o];
                        a8[t] = qa.w;
                        float dxy = dx[i] * dy[j];
                        float m1 = qa.y * dyy[j];
                        float in1 = mad(qa.x, dxx[i], m1);
                        e8[t] = mad(qa.z, dxy, in1);
                        slow = or(slow, rf_zc(e8[t], in1));       // in1: qx, qy of one sign (loader), no cancellation
#if RF_SPEC
                        wt8[t] = rf_exp2_t1f(e8[t], u8[t]);    // e finite: binary16 P, |K2| <= 2^100, |d| < 3
#else
                        wt8[t] = rf_exp2_t1(e8[t], u8[t]);
#endif
                    }
                    bool anyu = false;
                    [unroll] for (int t1 = 0; t1 < 8; ++t1) anyu = or(anyu, u8[t1]);
                    if (anyu && (RF_PROF & 64) == 0) {
                        bool h8[8], anyh = false;
                        [unroll] for (int t2 = 0; t2 < 8; ++t2) {
                            h8[t2] = false;
                            if (u8[t2]) { bool h; wt8[t2] = rf_exp2_core(e8[t2], h); h8[t2] = h; }
                            anyh = or(anyh, h8[t2]);
                        }
                        if (anyh) rf_exp2_fix8(e8, wt8, h8);
                    }
                    // a weight 0 < wt < 2^-100 can give a product below FMINN (alpha, gates >= 2^-24): then the whole
                    // wave accumulates the batch with rf_accx (= F); otherwise the plain fma is F (RESOLVE-FAST.md)
                    bool tw = false;
                    [unroll] for (int t4 = 0; t4 < 8; ++t4) tw = or(tw, (asuint(wt8[t4]) - 1u) < 0x0d7fffffu);
#if RF_MUT == 4
                    tw = false;                                   // mutation: tiny weights with the plain fma
#endif
                    if (WaveActiveAnyTrue(tw)) {
                        [unroll] for (int t5 = 0; t5 < 8; ++t5) {
                            float wt = wt8[t5];
                            uint g01 = g8[t5].x, g2 = g8[t5].y;
                            fl |= g2;
                            W = W + wt;
                            Aw = rf_accx(wt, a8[t5], Aw);
                            Gs.x = rf_accx(wt, f16tof32(g01), Gs.x);
                            Gs.y = rf_accx(wt, f16tof32(g01 >> 16), Gs.y);
                            Gs.z = rf_accx(wt, f16tof32(g2 & 0xffffu), Gs.z);
                        }
                    } else {
                        [unroll] for (int t3 = 0; t3 < 8; ++t3) {
                            float wt = wt8[t3];
                            uint g01 = g8[t3].x, g2 = g8[t3].y;
                            fl |= g2;
                            W = W + wt;
                            Aw = mad(wt, a8[t3], Aw);
                            Gs.x = mad(wt, f16tof32(g01), Gs.x);
                            Gs.y = mad(wt, f16tof32(g01 >> 16), Gs.y);
                            Gs.z = mad(wt, f16tof32(g2 & 0xffffu), Gs.z);
                        }
                    }
                }
                slow = or(slow, (fl >> 16) != 0u);
                const float rW = rf_rcp(max(W, U.K13));
                float3 nq;
                [unroll] for (uint c = 0; c < 3u; ++c) {
                    nq[c] = fence(Gs[c] * rW);
                    slow = or(slow, and(nq[c] == 0.0f, and(Gs[c] != 0.0f, rW != 0.0f)));
                }
                const float WK = fence(W * U.K1);
                slow = or(slow, and(WK == 0.0f, and(W != 0.0f, U.K1 != 0.0f)));   // W >= 2^-126 only (tiny taps)
                const float gg = nq.x * WK;
                slow = or(slow, and(gg == 0.0f, and(nq.x != 0.0f, WK != 0.0f)));
                const float g = select(nanf(gg), 0.0f, min(max(gg, 0.0f), 1.0f));
                const float beta = mad(g, 1.0f + (-nq.y), nq.y), gamma = mad(g, 1.0f + (-nq.z), nq.z);
                slow = or(slow, or(rf_zc(beta, nq.y), rf_zc(gamma, nq.z)));
                const float a_up = Aw * rW;
                slow = or(slow, and(a_up == 0.0f, and(Aw != 0.0f, rW != 0.0f)));
                float hc = 0.0f;
                if (alp3c) {
#if !RF_EARLY
                    a00 = RF_RD1(AH, hb.xa, hb.ya); a10 = RF_RD1(AH, hb.xb, hb.ya);
                    a01 = RF_RD1(AH, hb.xa, hb.yb); a11 = RF_RD1(AH, hb.xb, hb.yb);
#endif
                    float h = rf_bil1w(hb, a00, a10, a01, a11);
                    float mn = RR_FMAX, mx = -RR_FMAX;
                    [unroll] for (int j = 0; j < 3; ++j) [unroll] for (int i = 0; i < 3; ++i) {
                        float t = asfloat(rf_u4[(uint)(o3y[j] + o3x[i])].w);
                        mn = min(mn, t); mx = max(mx, t);
                    }
                    hc = select(alp3, rf_clampHist(h, mn, mx), 0.0f);
                }
                const float omb = 1.0f + (-beta), omg = 1.0f + (-gamma);
                const float mb = omb * hc, mg = omg * hc;
                ah = mad(a_up, beta, mb); oa = mad(a_up, gamma, mg);
                slow = or(slow, or(and(mb == 0.0f, and(hc != 0.0f, omb != 0.0f)), and(mg == 0.0f, and(hc != 0.0f, omg != 0.0f))));
                slow = or(slow, or(and(ah == 0.0f, or(mb != 0.0f, and(a_up != 0.0f, beta != 0.0f))),
                                   and(oa == 0.0f, or(mg != 0.0f, and(a_up != 0.0f, gamma != 0.0f)))));
                ah = select(nanf(ah), a_up, ah);
                oa = select(nanf(oa), a_up, oa);
                if ((flags & 16u) != 0u) oaSep = oa; else alphaC = oa;
            }
        RF_TICK(5);
            // ---- transparency layer (54b.3)
            if (lay && (RF_PROF & 2) == 0) {
#if !RF_EARLY
                l00 = RF_RD4(L, lb.xa, lb.ya); l10 = RF_RD4(L, lb.xb, lb.ya);
                l01 = RF_RD4(L, lb.xa, lb.yb); l11 = RF_RD4(L, lb.xb, lb.yb);
                if (lay3c) {
                    h00 = RF_RD4(TH, hb.xa, hb.ya); h10 = RF_RD4(TH, hb.xb, hb.ya);
                    h01 = RF_RD4(TH, hb.xa, hb.yb); h11 = RF_RD4(TH, hb.xb, hb.yb);
                }
#endif
                float4 cur = rf_bil4(lb, l00, l10, l01, l11);
                cur.x = U.E * cur.x; cur.y = U.E * cur.y; cur.z = U.E * cur.z;
                float4 hist = cur;
                if (lay3c) {
                    float4 mn = RR_FMAX, mx = RR_FMINN;
#if RFX_TB & 2
                    if (intR) {
                        const uint b3 = (uint)((ky - 1 - ry0) * (int)rp + (kx - 1 - rx0));
                        [unroll] for (uint j = 0; j < 3u; ++j) [unroll] for (uint i = 0; i < 3u; ++i) {
#if RFX_LR
                            float4 t = asfloat(rf_lr4(b3 + j * rp + i, U.E, mzv));
#else
                            float4 t = asfloat(RF_LR(b3 + j * rp + i));
#endif
                            mn = min(mn, t); mx = max(mx, t);
                        }
                    } else
#endif
                    [unroll] for (int j = 0; j < 3; ++j) [unroll] for (int i = 0; i < 3; ++i) {
#if RFX_LR
                        float4 t = asfloat(rf_lr4((uint)(o3y[j] + o3x[i]), U.E, mzv));
#else
                        float4 t = asfloat(RF_LR((uint)(o3y[j] + o3x[i])));
#endif
                        mn = min(mn, t); mx = max(mx, t);
                    }
                    float4 Hp = rf_bil4(hb, h00, h10, h01, h11);
                    Hp.x = Hp.x * U.fac; Hp.y = Hp.y * U.fac; Hp.z = Hp.z * U.fac;
                    [unroll] for (uint c = 0; c < 4u; ++c) hist[c] = select(lay3, rf_clampHist(Hp[c], mn[c], mx[c]), cur[c]);
                }
                float4 T;
                [unroll] for (uint c = 0; c < 4u; ++c) {
                    float m = w1 * hist[c];
                    T[c] = mad(w, cur[c], m);
#if RF_SPEC
                    // binary16 layer: cur, hist 0 or at least 2^-98, w, w1 0 or at least 2^-24 (RESOLVE-FAST.md): m and
                    // w cur never fall below the normal range; only a cancellation of the two can
                    slow = or(slow, and(T[c] == 0.0f, m != 0.0f));
#else
                    slow = or(slow, or(and(m == 0.0f, and(hist[c] != 0.0f, w1 != 0.0f)),
                                       and(T[c] == 0.0f, or(m != 0.0f, and(cur[c] != 0.0f, w != 0.0f)))));
#endif
                }
                uint th0 = rf_f2h(T.x), th1 = rf_f2h(T.y), th2 = rf_f2h(T.z), th3 = rf_f2h(T.w);
                if (own && !slow) {
                    const Res tw = THW;
                    if (tw.off != 0u && x < (int)tw.w && y < (int)tw.h)
                        wg_st64(tw.off, (rf_m24((uint)y, tw.pitch) + (uint)x) * 8u, uint2(th0 | (th1 << 16), th2 | (th3 << 16)));
                }
                float4 Ts = float4(f16tof32(th0), f16tof32(th1), f16tof32(th2), f16tof32(th3));
                float oms = 1.0f + (-Ts.w);
                [unroll] for (uint c2 = 0; c2 < 3u; ++c2) {
                    float r = mad(col[c2], oms, Ts[c2]);
                    // col (decode) 0 or at least 2^-48, 1 - Ts.w 0 or at least 2^-24: only a cancellation with Ts[c2] can
                    slow = or(slow, and(r == 0.0f, Ts[c2] != 0.0f));
                    col[c2] = r;
                }
            }
            if (alp && (RF_PROF & 1) == 0 && own && !slow) {
                const Res aw = AHW;
                if (aw.off != 0u && x < (int)aw.w && y < (int)aw.h)
                    wg_st16(aw.off, (rf_m24((uint)y, aw.pitch) + (uint)x) * 2u, rf_f2h(ah));            // ch 1 (rf_uniforms)
            }
        }
        [unroll] for (uint c = 0; c < 3u; ++c) {
            lin[c] = U.rE * col[c];
            slow = or(slow, and(lin[c] == 0.0f, and(col[c] != 0.0f, U.rE != 0.0f)));
        }
        lin.w = alphaC;
        slow = or(slow, tiny);
#if RF_MUT == 2
        slow = !tok;                                    // mutation: no pixel sent to the verified code for its values
#endif
        }   // tok
        RF_TICK(6);
        if (own) {
            if (slow) {                                 // the verified kernel's own code, in rr_resolve_fixup
#if RF_COUNT
                uint oc;
                wg_mem.InterlockedAdd(aux, 1u, oc);
#endif
                uint pix = (uint)y * U.We + (uint)x, ob, bm = bmap;
                if (bm != 0u) wg_mem.InterlockedOr(bm + (pix >> 5) * 4u, 1u << (pix & 31u), ob);
            } else {
                bool inside = (flags & 128u) == 0u || (x < Wo && y < Ho);
                if (inside) {
                    const Res o = OUTR;
#if RF_SPEC
                    int ox = x + o.ox, oy = y + o.oy;
                    if (o.off != 0u && ox >= 0 && oy >= 0 && ox < (int)o.w && oy < (int)o.h) {
                        const uint2 ow = uint2(rf_f2h(lin.x) | (rf_f2h(lin.y) << 16), rf_f2h(lin.z) | (rf_f2h(lin.w) << 16));
                        wg_st64(o.off, (rf_m24((uint)oy, o.pitch) + (uint)ox) * 8u, ow);
#if RF_SPEC == 2
                        if (gst) rf_gstore(gorg, ox, oy, ow);
#endif
                    }
                    if ((flags & 25u) == 25u) {
                        const Res ao = AOR;
                        int ax = x + ao.ox, ay = y + ao.oy;
                        if (ao.off != 0u && ax >= 0 && ay >= 0 && ax < (int)ao.w && ay < (int)ao.h)
                            wg_st32(ao.off, (rf_m24((uint)ay, ao.pitch) + (uint)ax) * 4u, asuint(oaSep));
                    }
#else
                    rf_store(o, x + o.ox, y + o.oy, lin);
                    if ((flags & 25u) == 25u) { const Res ao = AOR; rf_store(ao, x + ao.ox, y + ao.oy, float4(oaSep, 0.0f, 0.0f, 0.0f)); }
#endif
                }
            }
        }
#if !(RFX_NB == 1 || RFX_NB == 2 || (RFX_NB == 3 && RF_SPEC == 2))
        GroupMemoryBarrierWithGroupSync();
#endif
        RF_TICK(7);
    }
#if RF_TIMING
    if (WaveIsFirstLane()) {
        uint o;
        [unroll] for (uint k = 0; k < 8u; ++k) wg_mem.InterlockedAdd(aux + 4u * k, tck[k], o);
        wg_mem.InterlockedAdd(aux + 32u, 1u, o);
    }
#endif
}

// One fixup pixel; with parameter word 16 = 1 also its arena output texel (just written by rr_resolve_px, or the
// texel as it was) into the game's texture, as rt.store would (OUT: binary16 x 4, origin 0, pitch We).
void rf_fix_px(uint pix, uint We)
{
    const int x = (int)(pix % We), y = (int)(pix / We);
    rr_resolve_px(x, y);
    if (PU(16) == 1u) {
        const Res o = res(48u);
        const bool inside = (PU(8) & 128u) == 0u || (x < (int)PU(6) && y < (int)PU(7));
        if (inside && o.off != 0u && x < (int)o.w && y < (int)o.h)
            rf_gstore(uint2(PU(39) & 0xffffu, PU(39) >> 16), x, y, wg_ld64(o.off, (rf_m24((uint)y, o.pitch) + (uint)x) * 8u));
    }
}
// The pixels the fast kernel left to the verified code: a bit per pixel (y * We + x) in the bitmap at parameter word
// 37, set by rr_resolve_fast; one lane per bitmap word, which it clears after resolving its pixels with rr_resolve_px.
// Run after rr_resolve_fast with ceil(ceil(We * He / 32) / WG_BLOCK) workgroups.
WG_KERNEL(rr_resolve_fixup)
{
    WG_SETUP();
    rf_toff = PU(38);                                   // the exp2 table (RF_TABLE_ARENA)
    uint We = PU(0), He = PU(1), bm = PU(37);
    uint w = wg_block_id() * WG_BLOCK + wg_lane_id();
    uint nw = (We * He + 31u) / 32u;
#if WG_WAVE == 32
    // wave32: the wave's set words one after another, its 32 lanes on the 32 pixels of one word (a lane per pixel)
    uint bits = w < nw ? wg_ld32(bm, w * 4u) : 0u;
    if (bits != 0u) wg_st32(bm, w * 4u, 0u);
    const uint lane = WaveGetLaneIndex();
    [loop] while (WaveActiveAnyTrue(bits != 0u)) {
        uint src = firstbitlow(WaveActiveBallot(bits != 0u).x);
        uint wb = WaveReadLaneAt(bits, src), ww = WaveReadLaneAt(w, src);
        if (lane == src) bits = 0u;
        uint pix = ww * 32u + lane;
        if (((wb >> lane) & 1u) != 0u && pix < We * He) rf_fix_px(pix, We);
    }
#else
    if (w >= nw) return;
    uint bits = wg_ld32(bm, w * 4u);
    if (bits == 0u) return;
    wg_st32(bm, w * 4u, 0u);
    [loop] while (bits != 0u) {
        uint b = firstbitlow(bits);
        bits &= bits - 1u;
        uint pix = w * 32u + b;
        if (pix < We * He) rf_fix_px(pix, We);
    }
#endif
}

// The quarter pooled history (54b.5): rr_resolve_pool's operations (rr_pool_px) with the exactly equal fast exp2 /
// rcp / unorm reads and stores, and a log2 that only has to give the same RGB10A2 code: the hardware log2 is within
// one f32 step of LOG2_CR for every input with a finite result (and equal otherwise; resolve_fast.py exhaustive
// hwlog2), so the value v = ((L * K6) * K7) * K11 from it is within 2 (2 + 1) ... = 23 steps of the exact one (each
// rounded product at most doubles the distance in steps and adds one); its code (round half even of clamp(v) * 1023)
// is the exact code unless clamp(v) * 1023 lies within RF_PAMB of a half-integer, and then LOG2_CR decides.
// Build with RR_ELEM_EXACT=2 (exp2 table at parameter word 38; RF_TABLE_ARENA=0: read from a groupshared copy).
#define RF_PAMB asfloat(0x39000000u)        // 2^-13: the rounding of q itself (q < 1024, ulp <= 2^-14), with margin
float rf_pool_v(float L, float K7, float K11) { return M(M(M(L, RR_K6), K7), K11); }
uint4 rf_pu4(uint i)                                   // parameter words i .. i + 3 (one load, uniform)
{
    uint4 w = wg_mem.Load4(prm + i * 4u);
    return uint4(WaveReadLaneFirst(w.x), WaveReadLaneFirst(w.y), WaveReadLaneFirst(w.z), WaveReadLaneFirst(w.w));
}
Res rf_res4(uint4 a, uint4 b)
{
    Res r;
    r.off = a.x; r.w = a.y; r.h = a.z; r.pitch = a.w; r.fmt = b.x; r.ch = b.y; r.ox = (int)b.z; r.oy = (int)b.w;
    return r;
}
WG_KERNEL(rr_resolve_pool_fast)
{
    WG_SETUP();
    uint4 p24 = rf_pu4(24u), p28 = rf_pu4(28u), p32 = rf_pu4(32u), p36 = rf_pu4(36u);
#if RF_TABLE_ARENA
    rf_toff = p36.z;
#else
    rf_load_table(p36.z);                           // groupshared copy (rr_x_hook reads rf_t)
#endif
    uint idx = wg_block_id() * WG_BLOCK + wg_lane_id();
    Res ph = rf_res4(rf_pu4(160u), rf_pu4(164u)), pq = rf_res4(rf_pu4(168u), rf_pu4(172u));
    uint w2 = (ph.w + 1u) / 2u, h2 = (ph.h + 1u) / 2u;
    if (idx >= w2 * h2) return;
    int x = (int)(idx % w2), y = (int)(idx / w2);
    float K4 = asfloat(p24.y), K5 = asfloat(p24.z), K7 = asfloat(p24.w), K8 = asfloat(p28.x), K10 = asfloat(p28.y);
    float K11 = asfloat(p28.z), K12 = asfloat(p28.w), C = asfloat(p32.y);
    float3 acc = 0.0f;
    uint cnt = 0u;
    bool tz = false;
    // the four PH1 texels first (one load each for the RGB10A2 layouts; clamped addresses, used only inside)
    uint3 cw[4];
    [unroll] for (uint t0 = 0; t0 < 4u; ++t0) {
        int tx = min(2 * x + (int)(t0 & 1u), (int)ph.w - 1), ty = min(2 * y + (int)(t0 >> 1), (int)ph.h - 1);
        uint ix = (uint)ty * ph.pitch + (uint)tx;
        if (ph.fmt == 3u) { uint wv = wg_ld32(ph.off, ix * 4u); cw[t0] = uint3(wv & 1023u, (wv >> 10) & 1023u, (wv >> 20) & 1023u); }
        else if (ph.fmt == 4u) { uint2 wv = wg_ld64(ph.off, ix * 8u); cw[t0] = uint3(wv.x & 0xffffu, wv.x >> 16, wv.y & 0xffffu); }
        else cw[t0] = uint3(0u, 0u, 0u);
    }
    [unroll] for (uint t = 0; t < 4u; ++t) {
        int tx = 2 * x + (int)(t & 1u), ty = 2 * y + (int)(t >> 1);
        if (tx < (int)ph.w && ty < (int)ph.h) {
            float3 c = (ph.fmt == 3u || ph.fmt == 4u)
                ? float3(rf_cv(cw[t].x, 1023.0f, RF_R1023), rf_cv(cw[t].y, 1023.0f, RF_R1023), rf_cv(cw[t].z, 1023.0f, RF_R1023))
                : rf_texel(ph, tx, ty, tz).xyz;
            [unroll] for (uint k = 0; k < 3u; ++k) {
                float cc = M(c[k], C);
                float tt = (cc > K5 || nanf(cc)) ? min_nan(cc, K10) : K5;
                acc[k] = F(A(X(M(M(tt, K8), RR_K9)), -1.0f), K12, acc[k]);     // X = rr_x_hook here
            }
            cnt += 1u;
        }
    }
    float rn = rf_rcp((float)cnt);
    float4 o = float4(0.0f, 0.0f, 0.0f, 1.0f);
    [unroll] for (uint k = 0; k < 3u; ++k) {
        float mean = M(rn, acc[k]);
        float yv = pmax(F(mean, K4, 1.0f), K5);
        float v = 0.0f;
        bool exact = true;
        if ((asuint(yv) & 0x80000000u) == 0u && asuint(yv) >= 0x00800000u && asuint(yv) < 0x7f800000u) {
            v = rf_pool_v(fence(log2(yv)), K7, K11);
            float q = min(max(v, 0.0f), 1.0f) * 1023.0f;
            // 23 steps of v (see above), taken as 48 steps of v' (a binade edge between them), in code units
            float amb = mad(asfloat(asuint(abs(v)) & 0x7f800000u), asfloat(0x36c00000u) * 1023.0f, RF_PAMB);
            exact = abs(q - (floor(q) + 0.5f)) <= amb;
        }
#if (RF_PROF & 128) == 0
        if (exact) { bool h; v = rf_pool_v(LOG2_CR(yv, 0u, h), K7, K11); }
#endif
        o[k] = v;
    }
    rf_store(pq, x, y, o);
}
