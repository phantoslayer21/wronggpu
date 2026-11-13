// SPDX-License-Identifier: MIT
// D3D12 port of d1p12/swin_block_c64_add_dec0_nc (clean/kernels/dec1_dec0.hip built with the decoder's pack12 flags,
// dec1_dec0__pack12.co: WG_DEC_PACK12=1, defaults otherwise; HIP ISA isa/swin_block_c64_add_dec0_nc.s).
// The last Swin block of dec1 (C = 64, 2 heads, MLP 256, decoder block: input = expanded map + skip, every inner FP8
// pack without clamp: the _nc kernel) with the coefficient decoder dec0 as its epilogue, writing the pack12 images.
// One window of 16 tokens per wave, 8 waves per workgroup, weights staged through LDS in double-buffered units
// (2 x 8 KB: MLP units of 4 hidden tiles with an exact q8(GELU) table, the default 16 KB arena table DEC_GELU 1,
// hlsl/wg_gelu_lut.hlsli, or the 2 KB groupshared table DEC_GELU 2, hlsl/wg_gelu_lds.hlsli, which takes another 2 KB;
// 2 x 16 KB = the whole 32 KB of groupshared memory with HIP's transcendental chain DEC_GELU 0); dec0 reuses the first
// 13 KB for its tables and fragments.
//
// Same operations in the same order and with the same roundings as the HIP kernel. Where HIP's ISA fused a multiply and
// an add (FP contraction in swin.hip; decoder.hip is compiled with contraction off) this file uses mad():
//   * RMSNorm sums of squares: the pair a0^2 + a1^2 is fma(a0, a0, a1 * a1), the pairs and groups are added as written;
//   * soft cap: c - c^3 k = fma(c^3, -k, c); the softmax exponent is fma(tau log2e, that, -max), while the maximum
//     itself is taken over the rounded products (tau log2e) * (...);
//   * softmax probabilities: (f16)(e * r) rounded once (v_fma_mixlo_f16 r, e, 0): round to odd in FP32, then RNE;
//     likewise dec0's C2 = (f16)((sx sy) rho).
// The FP16 conversions of whole fragments and the FP8 packs use the driver's matrix conversion (the bits of
// v_cvt_f16_f32 / v_cvt_pk_fp8_f32); everything around the matrix calls stays in wave-uniform control flow (dec0 decodes
// both pixels of every lane and only the stores are conditional). dec0's FP16 latent update is done in FP32 with an FP16
// rounding after each step, not in half arithmetic (the driver contracts half mul + add into v_fma_mix_f32).
// Bit-identical to the HIP kernel on the synthetic frames (clean/dx12/compare.py, both cases, every buffer).
// @wg-kernel d1p12/swin_block_c64_add_dec0_nc co=dec1_dec0__pack12.co block=256 wave=32 args=p,p,p,u,u,u,u,p,p,p,p,p,p,p,b56 defines=WG_GELU_T5=0
#include "wg_platform.hlsli"
#include "wg_gelu_lut.hlsli"
#include "dec1_dec0_diet.hlsli"
// Instruction diet (dec1_dec0_diet.hlsli; every switch gives the same bytes). Measured in the game path (interop hipbench,
// executor=dx12, dec0 stage stamps, 1440p): all four with DEC_GELU_PAT 11011 -15 us; GELU offsets -8, C2 -2, 2^n -1.
// Round 2 (vector-memory loads return in order, so a load issued after the next unit's weight fetch waits for that
// fetch): b1 loads ahead of the fetch -12 us, the next group's table loads before W2 -2 us, the first group's b1 one
// unit ahead + DEC_GELU_PAT 11111 -3 us (dec0 stage 0.372 -> 0.357 ms).
// Round 3 (dual issue and dec0 tail, issue 1857 -> 1773 per wave, dec0 stage 0.356 -> 0.353 ms): VOPD partners for the
// GELU table fmas (dec_gelu_fin_pre: next group's fmaaks and code masks next to them), W2 one group at a time,
// same-bits register copies of the norm scale and the softmax offset (src banks), latent codes by one FP16 fma per
// channel pair, B's FP8 codes by the matrix conversion.
#ifndef DEC_D_GELU_OFF
#define DEC_D_GELU_OFF 1   // arena-table GELU offsets with one v_bfe per pair less, stage by stage (dd_gelu_fetch8)
#endif
#ifndef DEC_D_SMAXMX
#define DEC_D_SMAXMX 1     // softmax maximum as RN(k max zp) (rounding is monotone, k > 0): one multiply instead of 8
#endif
#ifndef DEC_D_C2MIX
#define DEC_D_C2MIX 1      // dec0 C2 = (f16)(sxy rho) by the driver's folded v_fma_mixlo_f16 sxy, rho, 0 (HIP's instruction)
#endif
#ifndef DEC_D_EXP2N
#define DEC_D_EXP2N 1      // dec0 B's 2^n (integer n <= 0) by v_exp_f32: exact powers of two, 0 below 2^-126 (FTZ)
#endif

WG_ARGS_BEGIN
    uint a_x, a_xadd, a_w;                         // expanded map, skip (FP8 [H][W][64]), block weights
    int a_H, a_W, a_ox, a_oy;                      // F/2 map size, window offset
    uint a_motion, a_lprev, a_A, a_B, a_C, a_latent, a_wdec;
    int P_rw, P_rh, P_fw, P_fh, P_reset, P_pad0;   // WgDecParams (56 bytes)
    float P_dw, P_dh, P_kx, P_ky, P_jx, P_jy, P_pjx, P_pjy;
WG_ARGS_END

// ---- block layout (swin.hip BlockLayout<64, 2, 256>, WG_UNIT_BYTES 16384) ----
#define C_ 64
#define KC 4
#define HALF 32
#define L_G1 16u
#define L_G2 272u
#define L_BP 528u
#define L_B2 784u
#define L_B1 1040u
#define L_BIAS 2064u
#define L_STREAM 4112u
#define HEAD_BYTES 6144u
#ifndef DEC_GELU
#define DEC_GELU 3   // 0: transcendental, 1: 16 KB arena table (wg_gelu_lut.hlsli), 3: per-tile mix (DEC_GELU_PAT,
                     // default 11111: every tile by the arena table; MLP pipelined by one group; 10012 was measured
                     // -6 us against 1),
                     // 2: 2 KB groupshared table (wg_gelu_lds.hlsli)
#endif
#ifndef HPU
#if DEC_GELU != 0
#define HPU 4        // MLP units of 4 hidden tiles (8 KB): the staging buffers shrink to 2 x 8 KB (room for the groupshared
                     // table; with the arena table measured -15 us against the transcendental chain, -10 us with 16 KB units)
#else
#define HPU 8
#endif
#endif
#define MLP_BYTES (2048u * HPU)
#define MAXU (MLP_BYTES > 8192u ? MLP_BYTES : 8192u)   // a head unit is moved as 8192 bytes
#define MLP_ROWS (MLP_BYTES / 4096u)                    // 16-byte chunks per lane of an MLP unit
#define QKV_FRAGS 16u
#define NMLP (16 / HPU)
#define MLP_GROUP 2
#define L_HEAD(hd) (L_STREAM + (uint)(hd) * HEAD_BYTES)
#define L_MLP(u) (L_STREAM + 2u * HEAD_BYTES + (uint)(u) * MLP_BYTES)

// ---- dec0 blob (decoder.hip) ----
#define kOffTables 512u
// dec0's staged block sits 16 bytes up in the LDS: table bases are then 4 dwords past a multiple of 256, base + byte
// is not a disjoint OR, and the compiler folds the base into the ds offset (bfe + shift per lookup instead of 3 ops)
#define kLdsStaged 10240u
#define kFragLds 5136u
#define kTabA 4u
#define kTabSx 260u
#define kTabRho 516u
#define kTabZG 772u
#define kTabSn 1028u
// FP8 E4M3 -> f32 lookup table (256 dwords) for dec0 at byte 12288 of the LDS, written after dec_stage (which fills
// bytes 0 .. 12287)
#define kLut8 3072u

groupshared uint lds[MAXU / 2u];   // 2 x MAXU bytes (dec0 uses its first 13 KB after the block)
#if DEC_GELU == 3
#define DEC_GL_DIGIT(k) ((DEC_GELU_PAT / (k)) % 10 == 2)
#define DEC_GL_USED (DEC_GL_DIGIT(1000) || DEC_GL_DIGIT(100) || DEC_GL_DIGIT(10) || DEC_GL_DIGIT(1))   // a tile on gl_tab
#else
#define DEC_GL_USED (DEC_GELU == 2)
#endif
#if DEC_GL_USED
groupshared uint64_t gl_tab[256];   // the 2 KB GELU table (wg_gelu_lds.hlsli), filled at kernel start
#define WG_GL_GET(off) uint2((uint)gl_tab[(off) >> 3], (uint)(gl_tab[(off) >> 3] >> 32))
#include "wg_gelu_lds.hlsli"
#endif

// ---- staging (swin.hip Stager<MAXU, 256>): global loads early into registers, LDS stores later ----
// A head unit (384 chunks) is moved as 512 chunks: the extra 128 come from the weights that follow it and land in the
// unused part of the buffer (bytes 6144 .. 8191; the buffer being filled is idle), so no lane test is needed.
static uint4 g_pre[4];
void fetch_head(uint src, uint tid)
{
    [unroll] for (uint i = 0; i < 2u; ++i) g_pre[i] = wg_ld128(a_w, src + (tid + i * 256u) * 16u);
}
void fetch_mlp(uint src, uint tid)
{
    [unroll] for (uint i = 0; i < MLP_ROWS; ++i) g_pre[i] = wg_ld128(a_w, src + (tid + i * 256u) * 16u);
}
void lds_st128(uint byte_off, uint4 v)
{
    uint o = byte_off >> 2;
    lds[o] = v.x; lds[o + 1u] = v.y; lds[o + 2u] = v.z; lds[o + 3u] = v.w;
}
void commit(uint dst, uint tid, uint n)
{
    [unroll] for (uint i = 0; i < 4u; ++i)
        if (i < n) lds_st128(dst + (tid + i * 256u) * 16u, g_pre[i]);
}
// One staged WMMA fragment: lane l reads bytes [8l, 8l + 8) of fragment `index` of an LDS unit; uw = the unit's dword
// index + 2 lane (base + constant offsets, so that the driver makes 64-bit LDS loads of them)
// lane2: 2 lane plus an opaque zero (wg_grid_ >> 31 is 0 at run time): the compiler cannot prove the lane and unit bits
// disjoint, keeps an add instead of an OR and folds every fragment offset into the 64-bit LDS loads.
uint lane2(uint lane) { return (lane + (0u - (wg_grid_ >> 31))) * 2u; }
uint2 lfrag(uint uw, uint index) { return uint2(lds[uw + index * 64u], lds[uw + index * 64u + 1u]); }

#if DEC_GELU == 3
// Mixed GELU: tile g of the group with parity j by method digit (2 j + g) of DEC_GELU_PAT (written 1abcd; 1 = arena
// table: vector-memory loads, 2 = groupshared table: LDS loads, 0 = HIP's transcendental chain: VALU + TRANS), the
// MLP group loop unrolled. Every method gives HIP's bytes; mixing spreads the work over the units.
#ifndef DEC_GELU_PAT
#define DEC_GELU_PAT 11111   // arena table in every tile. With the b1 loads ahead of the weight fetch and the next
                             // group's table loads under W2 (game path, dec0 stamps): -3 us against 11011 (10111 -1,
                             // 11110 -1). Round 1 (with dd_gelu_fetch8): 11011 -4 us against 10012; 10011 -1.5,
                             // 10021 +4, 10001 +8.5, 11101 +3, 10111 0, 11110 +6, 11012 +1, 12011 +4, 11021 +3, 11111 +9
#endif
uint dec_gelu_m(uint j, uint g)
{
    const uint ix = (j & 1u) * 2u + g;
    return (DEC_GELU_PAT / (ix == 0u ? 1000u : ix == 1u ? 100u : ix == 2u ? 10u : 1u)) % 10u;
}
struct DecGeluPre { uint2 e[MLP_GROUP][8]; };
DecGeluPre dec_gelu_pre(v8f h[MLP_GROUP], uint j)
{
    DecGeluPre r;
    [unroll] for (uint g = 0; g < MLP_GROUP; ++g) {
        if (dec_gelu_m(j, g) == 1u) {
#if DEC_D_GELU_OFF
            const wg_gelu_lut_e a = dd_gelu_fetch8(h[g]);
#else
            const wg_gelu_lut_e a = wg_gelu_lut_fetch8(h[g]);
#endif
            [unroll] for (uint i = 0; i < 8; ++i) r.e[g][i] = a.e[i];
#if DEC_GL_USED
        } else if (dec_gelu_m(j, g) == 2u) {
            uint o[8];
            wg_gl_offs8(h[g], o);
            [unroll] for (uint i = 0; i < 8; ++i) r.e[g][i] = WG_GL_GET(o[i]);
#endif
        } else {
            [unroll] for (uint i = 0; i < 8; ++i) r.e[g][i] = uint2(0, 0);
        }
    }
    return r;
}
void dec_gelu_fin(v8f h[MLP_GROUP], DecGeluPre p, uint j, out uint2 q[MLP_GROUP])
{
    [unroll] for (uint g = 0; g < MLP_GROUP; ++g) {
        v8f y;
        if (dec_gelu_m(j, g) != 0u) {
            [unroll] for (uint i = 0; i < 8; ++i) y.v[i] = mad(h[g].v[i], asfloat(p.e[g][i].x), asfloat(p.e[g][i].y));
        } else {
            float t[8];
            [unroll] for (uint i = 0; i < 8; ++i) t[i] = h[g].v[i] * h[g].v[i];
            [unroll] for (uint i1 = 0; i1 < 8; ++i1) t[i1] = mad(t[i1], asfloat(0xbdcd6502u), asfloat(0xc013b8edu));
            [unroll] for (uint i2 = 0; i2 < 8; ++i2) t[i2] = h[g].v[i2] * t[i2];
            [unroll] for (uint i3 = 0; i3 < 8; ++i3) t[i3] = wg_exp2(t[i3]);
            [unroll] for (uint i4 = 0; i4 < 8; ++i4) t[i4] = 1.0f + t[i4];
            [unroll] for (uint i5 = 0; i5 < 8; ++i5) t[i5] = wg_rcp(t[i5]);
            [unroll] for (uint i6 = 0; i6 < 8; ++i6) y.v[i6] = h[g].v[i6] * t[i6];
        }
        q[g] = wg_pack_q8_nc_ags(y);
    }
}
#endif

// dec_gelu_fin of group j and dec_gelu_pre of group j + 1 in one stage-by-stage sequence: the next group's code masks
// (v_and, VOPD Y) written next to this group's table fmas (v_fmac with odd destinations, VOPD X), so that the driver can
// pair them. Same operations and values as the two calls.
void dec_gelu_fin_pre(v8f h[MLP_GROUP], DecGeluPre p, uint j, out uint2 q[MLP_GROUP], v8f hn[MLP_GROUP], out DecGeluPre pn)
{
    const uint z = wg_bid_ >> 31;
    const uint msk = 0xffe0ffe0u | (z << 16) | (z * 0x1cu);
    uint dn[MLP_GROUP][4];
    v8f y[MLP_GROUP];
#if WG_GELU_T5 && DEC_GELU_PAT == 11111 && MLP_GROUP == 2
    // T5 (hlsl/wg_gelu_t5.hlsli): the next tiles' table indices are the FP32 bits of w >> 18 (one shift per value; no
    // conversion, mask or offset), this tile's fmas between them
    uint ix[MLP_GROUP][8];
    {
        float w[8], w1[8];
        [unroll] for (uint i = 0; i < 8; ++i) w[i] = mad(hn[0].v[i], WG_GELU_LUT_S, WG_GELU_LUT_B);
        [unroll] for (uint i2 = 0; i2 < 8; ++i2) {
            y[0].v[i2] = mad(h[0].v[i2], asfloat(p.e[0][i2].x), asfloat(p.e[0][i2].y));
            w1[i2] = mad(hn[1].v[i2], WG_GELU_LUT_S, WG_GELU_LUT_B);
        }
        [unroll] for (uint i1 = 0; i1 < 8; ++i1) ix[0][i1] = wg_gelu_t5_index(w[i1]);
        [unroll] for (uint i4 = 0; i4 < 8; ++i4) y[1].v[i4] = mad(h[1].v[i4], asfloat(p.e[1][i4].x), asfloat(p.e[1][i4].y));
        [unroll] for (uint i3 = 0; i3 < 8; ++i3) ix[1][i3] = wg_gelu_t5_index(w1[i3]);
    }
#elif DEC_GELU_PAT == 11111 && MLP_GROUP == 2
    // every tile on the arena table: next tile 0's fmaaks; this tile 0's fmacs (odd destinations) next to next tile 1's
    // fmaaks; this tile 1's fmacs next to the 8 masks (VOPD partners for every fmac)
    {
        float w[8], w1[8];
        [unroll] for (uint i = 0; i < 8; ++i) w[i] = mad(hn[0].v[i], WG_GELU_LUT_S, WG_GELU_LUT_B);
        [unroll] for (uint i1 = 0; i1 < 4; ++i1) dn[0][i1] = f32tof16(w[2 * i1]) | (f32tof16(w[2 * i1 + 1]) << 16);
        [unroll] for (uint i2 = 0; i2 < 8; ++i2) {
            y[0].v[i2] = mad(h[0].v[i2], asfloat(p.e[0][i2].x), asfloat(p.e[0][i2].y));
            w1[i2] = mad(hn[1].v[i2], WG_GELU_LUT_S, WG_GELU_LUT_B);
        }
        [unroll] for (uint i3 = 0; i3 < 4; ++i3) dn[1][i3] = f32tof16(w1[2 * i3]) | (f32tof16(w1[2 * i3 + 1]) << 16);
        [unroll] for (uint i4 = 0; i4 < 8; ++i4) {
            y[1].v[i4] = mad(h[1].v[i4], asfloat(p.e[1][i4].x), asfloat(p.e[1][i4].y));
            dn[i4 >> 2][i4 & 3u] &= msk;
        }
    }
#else
    [unroll] for (uint g = 0; g < MLP_GROUP; ++g) {
        float w[8];
        [unroll] for (uint i = 0; i < 8; ++i) w[i] = mad(hn[g].v[i], WG_GELU_LUT_S, WG_GELU_LUT_B);
        [unroll] for (uint i1 = 0; i1 < 4; ++i1) dn[g][i1] = f32tof16(w[2 * i1]) | (f32tof16(w[2 * i1 + 1]) << 16);
    }
    uint k = 0;
    [unroll] for (uint g1 = 0; g1 < MLP_GROUP; ++g1) {
        if (dec_gelu_m(j, g1) != 0u) {
            [unroll] for (uint i = 0; i < 8; ++i) {
                y[g1].v[i] = mad(h[g1].v[i], asfloat(p.e[g1][i].x), asfloat(p.e[g1][i].y));
                [unroll] for (uint gg = 0; gg < MLP_GROUP; ++gg)
                    if (dec_gelu_m(j + 1u, gg) == 1u && k >= gg * 4u && k < gg * 4u + 4u) dn[gg][k - gg * 4u] &= msk;
                k += 1u;
            }
        }
    }
    [unroll] for (uint g2 = 0; g2 < MLP_GROUP; ++g2)
        [unroll] for (uint i2 = 0; i2 < 4; ++i2)
            if (dec_gelu_m(j + 1u, g2) == 1u && g2 * 4u + i2 >= k) dn[g2][i2] &= msk;
#endif
    [unroll] for (uint g3 = 0; g3 < MLP_GROUP; ++g3) {
        if (dec_gelu_m(j, g3) == 0u) {
            float t[8];
            [unroll] for (uint i = 0; i < 8; ++i) t[i] = h[g3].v[i] * h[g3].v[i];
            [unroll] for (uint i1 = 0; i1 < 8; ++i1) t[i1] = mad(t[i1], asfloat(0xbdcd6502u), asfloat(0xc013b8edu));
            [unroll] for (uint i2 = 0; i2 < 8; ++i2) t[i2] = h[g3].v[i2] * t[i2];
            [unroll] for (uint i3 = 0; i3 < 8; ++i3) t[i3] = wg_exp2(t[i3]);
            [unroll] for (uint i4 = 0; i4 < 8; ++i4) t[i4] = 1.0f + t[i4];
            [unroll] for (uint i5 = 0; i5 < 8; ++i5) t[i5] = wg_rcp(t[i5]);
            [unroll] for (uint i6 = 0; i6 < 8; ++i6) y[g3].v[i6] = h[g3].v[i6] * t[i6];
        }
        q[g3] = wg_pack_q8_nc_ags(y[g3]);
    }
    [unroll] for (uint g4 = 0; g4 < MLP_GROUP; ++g4) {
        if (dec_gelu_m(j + 1u, g4) == 1u) {
#if WG_GELU_T5 && DEC_GELU_PAT == 11111 && MLP_GROUP == 2
            [unroll] for (uint i5 = 0; i5 < 8; ++i5) pn.e[g4][i5] = wg_gelu_t5[ix[g4][i5]];
#else
            uint o[8];
            [unroll] for (uint i3 = 0; i3 < 4; ++i3) o[2 * i3] = WG_GELU_LUT_TAB + ((dn[g4][i3] >> 2) & 0x7fffu);
            [unroll] for (uint i4 = 0; i4 < 4; ++i4) o[2 * i4 + 1] = WG_GELU_LUT_TAB + (dn[g4][i4] >> 18);
            [unroll] for (uint i5 = 0; i5 < 8; ++i5) pn.e[g4][i5] = wg_mem.Load2(o[i5]);
#endif
        } else {
            [unroll] for (uint i6 = 0; i6 < 8; ++i6) pn.e[g4][i6] = uint2(0, 0);
        }
    }
}

int wg_mirror(int p, int n)
{
    p = p < 0 ? -p : p;
    return p >= n ? 2 * (n - 1) - p : p;
}

// The driver keeps independent chains in the order written, so elementwise code below runs one step at a time over
// all values (one value after the other stalled on every dependency).

// r = rsq((ss + 2^-13) / 64) of the token (both lane halves). ss as HIP's ISA: per 8 values the pairs
// fma(a0, a0, a1 a1), ((p01 + p23) + (p45 + p67)); the four groups added in order.
float norm_scale(v8f acc[KC])
{
    float sq[16], pp[16], h[KC];
    [unroll] for (uint i = 0; i < 16u; ++i) sq[i] = acc[i >> 2].v[2u * (i & 3u) + 1u] * acc[i >> 2].v[2u * (i & 3u) + 1u];
    [unroll] for (uint i2 = 0; i2 < 16u; ++i2)
        pp[i2] = mad(acc[i2 >> 2].v[2u * (i2 & 3u)], acc[i2 >> 2].v[2u * (i2 & 3u)], sq[i2]);
    float l1[8];   // the same sums, level by level (independent adds next to each other: dual issue)
    [unroll] for (uint i3 = 0; i3 < 8u; ++i3) l1[i3] = pp[2u * i3] + pp[2u * i3 + 1u];
    [unroll] for (uint k = 0; k < KC; ++k) h[k] = l1[2u * k] + l1[2u * k + 1u];
    float ss = ((h[0] + h[1]) + h[2]) + h[3];
    ss = ss + wg_xor16(ss);
    return wg_rsq((ss + asfloat(0x39000000u)) * asfloat(0x3c800000u));
}

// q8(acc r g) (no clamp: _nc) for the four 8-value groups, then acc += b (g, b: f32 vectors in the block header).
// norm_apply with gamma / beta loaded early (norm_load: issued before a barrier, their latency hidden); one 8-value
// group at a time (same operations)
void norm_load(uint goff, uint boff, uint vg, uint vb, out v8f g[KC], out v8f b[KC])
{
    [unroll] for (uint k = 0; k < KC; ++k) {
        g[k] = wg_load8f(vg, goff + 32u * k);
        b[k] = wg_load8f(vb, boff + 64u * k);
    }
}
void norm_apply2(inout v8f acc[KC], float r, v8f g[KC], v8f b[KC], out uint2 q[KC])
{
    [unroll] for (uint k = 0; k < KC; ++k) {
        v8f y;
        const float r2 = asfloat(asuint(r) ^ (0u - (wg_grid_ >> 31)));   // same bits, another register (dual issue)
        [unroll] for (uint i = 0; i < 8u; ++i) y.v[i] = acc[k].v[i] * ((i & 1u) ? r2 : r);
        [unroll] for (uint i2 = 0; i2 < 8u; ++i2) y.v[i2] = y.v[i2] * g[k].v[i2];
        q[k] = wg_pack_q8_nc_ags(y);
        [unroll] for (uint i3 = 0; i3 < 8u; ++i3) acc[k].v[i3] = acc[k].v[i3] + b[k].v[i3];
    }
}

// ---- dec0 (decoder.hip, contraction off: every operation as written, fmaf = mad) ----
// The lane's two pixels (x0, y) and (x0 + 1, y) are decoded together, as .x / .y of 2-vectors: the same operations per
// pixel as decoder.hip, with the two independent chains next to each other.
float recip_rn(float b)
{
    float r0 = wg_rcp(b);
    return mad(r0, mad(-b, r0, 1.0f), r0);
}
float2 div_rn(float2 a, float b, float r)
{
    float2 q0 = a * r;
    return mad(mad(-b, q0, a), r, q0);
}

struct Prev2 {
    bool2 valid, lo, hi;
    float2 fx, fy;
    uint2 r0a, r0b, r1a, r1b;   // per pixel: texel pair (a, b) of row k0 (r0) and of row k1 (r1)
};

// Position of the pixels' previous latent (section 6) and its texels (decoder.hip prev_fetch)
Prev2 prev_fetch2(int x0, int y, uint2 mv)
{
    Prev2 q;
    const float2 mx = f16tof32(mv & 0xffffu), my = f16tof32(mv >> 16);
    const float2 hx = ((float2((float)x0, (float)(x0 + 1)) + 0.5f) - P_jx) * P_kx + mx;
    const float2 hy = (((float)y + 0.5f) - P_jy) * P_ky + my;
    q.valid = and(and(and(hx >= 0.0f, hx <= P_dw), and(hy >= 0.0f, hy <= P_dh)), P_reset == 0);
    const float2 X = div_rn(hx, P_kx, recip_rn(P_kx)) + P_pjx, Y = div_rn(hy, P_ky, recip_rn(P_ky)) + P_pjy;
    const float2 tx = X - 0.5f, ty = Y - 0.5f;
    const float2 xf = floor(tx), yf = floor(ty);
    q.fx = tx - xf;
    q.fy = ty - yf;
    const int2 xi = (int2)xf, yi = (int2)yf;
    q.lo = xi < 0;
    q.hi = xi > P_rw - 2;
    const uint2 xs = (uint2)clamp(xi, 0, P_rw - 2);
    const uint2 k0 = (uint2)clamp(yi, 0, P_rh - 1), k1 = (uint2)clamp(yi + 1, 0, P_rh - 1);
    const uint2 o0 = (k0 * (uint)P_rw + xs) * 4u, o1 = (k1 * (uint)P_rw + xs) * 4u;
    const uint2 t0 = wg_ld64(a_lprev, o0.x), t1 = wg_ld64(a_lprev, o0.y);
    const uint2 t2 = wg_ld64(a_lprev, o1.x), t3 = wg_ld64(a_lprev, o1.y);
    q.r0a = uint2(t0.x, t1.x); q.r0b = uint2(t0.y, t1.y);
    q.r1a = uint2(t2.x, t3.x); q.r1b = uint2(t2.y, t3.y);
    return q;
}

// FP8 E4M3 byte -> FP32 by the hardware (op 0x36 imm 0 = v_cvt_f32_fp8, as HIP); replaces the LDS lookup table
float dec_fp8_hw(uint b) { return asfloat(wg_ags_call(0x36u, 0u, 0u, b, 0u)); }
uint2 tabu2(uint base, uint2 i) { return uint2(lds[base + i.x], lds[base + i.y]); }
float2 tabf2(uint base, uint2 i) { return asfloat(tabu2(base, i)); }
float2 sn2(uint2 t, uint ch) { return tabf2(kTabSn, (t >> (8u * ch)) & 0xffu); }

// P (before the FP16 rounding) of the four channels, 0 where invalid
void prev_values2(Prev2 q, out float2 P[4])
{
    const uint2 t00 = select(q.hi, q.r0b, q.r0a), t10 = select(q.lo, q.r0a, q.r0b);
    const uint2 t01 = select(q.hi, q.r1b, q.r1a), t11 = select(q.lo, q.r1a, q.r1b);
    const float2 w00 = (1.0f - q.fx) * (1.0f - q.fy), w10 = q.fx * (1.0f - q.fy);
    const float2 w01 = (1.0f - q.fx) * q.fy, w11 = q.fx * q.fy;
    float2 a[4], b[4], c[4], d[4];
    [unroll] for (uint ch = 0; ch < 4u; ++ch) {
        a[ch] = sn2(t00, ch); b[ch] = sn2(t10, ch); c[ch] = sn2(t01, ch); d[ch] = sn2(t11, ch);
    }
    [unroll] for (uint ch2 = 0; ch2 < 4u; ++ch2)
        P[ch2] = select(q.valid, ((w00 * a[ch2] + w10 * b[ch2]) + w01 * c[ch2]) + w11 * d[ch2], 0.0f);
}

uint code_byte(uint c[10], uint i) { return (c[i >> 2] >> (8u * (i & 3u))) & 0xffu; }
uint2 cbytes(uint c[10], uint i) { return uint2(code_byte(c, i), code_byte(c, 17u + i)); }   // byte i of both pixels
uint2 unorm8(float2 v) { return (uint2)round(v * 255.0f); }
uint lcode(float v) { return (uint)wg_f2i(wg_rint(v * 127.0f)); }

// (float)(_Float16) of every value (v_cvt_f16_f32 RNE, back exactly)
uint dec_f2h_hw(float x) { return wg_ags_call(0x1cu, 0u, 1u, asuint(x), 0u); }   // v_cvt_f16_f32 (RNE), one value
// dec1's output pack: NaN-keeping clamp in 2 ops (HIP: v_maximumminimum_f32): c = clamp(x) (NaN -> -448), then
// fma(x, +0, c) with an opaque zero. x is finite or NaN (sums of FP8 products and finite biases cannot reach inf), so a
// finite x gives x*0 = +-0 with the sign of c and the result is c bit for bit, signed zeros included; a NaN x stays NaN.
// The FP8 conversion gives 0xff for every NaN, so wg_sat448's quieting OR cannot change a byte.
uint2 dec_pack_q8_ags(v8f d)
{
    const float z0 = asfloat(0u - (wg_grid_ >> 31));
    [unroll] for (uint i = 0; i < 8; ++i) d.v[i] = mad(d.v[i], z0, min(max(d.v[i], -448.0f), 448.0f));
    return wg_pack_q8_nc_ags(d);
}
// round_f16 of the first n values only (projection tile 4: slots 2..7 are padding and never stored)
v8f round_f16_n(v8f a, uint n)
{
    v8f r = a;
    [unroll] for (uint i = 0; i < 8u; ++i) if (i < n) r.v[i] = f16tof32(dec_f2h_hw(a.v[i]));
    return r;
}
v8f round_f16(v8f a)
{
    v8f r;
    [unroll] for (uint i = 0; i < 8u; ++i) r.v[i] = f16tof32(dec_f2h_hw(a.v[i]));
    return r;
}

WG_KERNEL(swin_block_c64_add_dec0_nc)
{
    WG_SETUP();
    const uint tid = wg_lane_id();
    const uint lane = tid & 31u, t = lane & 15u, hh = lane >> 4;
    const int nwx = (a_W + a_ox + 3) >> 2, nwy = (a_H + a_oy + 3) >> 2, nwin = nwx * nwy;
    const uint wave_id = (uint)wg_uniform((int)(tid >> 5));
    const int win0 = (int)(wg_block_id() * 8u + wave_id);
    // per-lane address bases (arena base + lane part; the opaque zero keeps them additions, so every constant part
    // folds into the load's immediate offset)
    const uint oz0 = 0u - (wg_grid_ >> 31);
    const uint vw32 = a_w + hh * 32u + oz0, vw128 = a_w + hh * 128u + oz0, vwb = a_w + (t * 2u + hh) * 32u + oz0;

    fetch_head(L_HEAD(0), tid);   // unit 0
#if DEC_GL_USED
    {
        const uint2 te = wg_gl_entry(tid);   // 256 lanes = 256 entries; the first barrier orders it before the MLP
        gl_tab[tid] = (uint64_t)te.x | ((uint64_t)te.y << 32);
    }
#endif

    // Gather: token t of the window, mirrored at the borders. The token's (row, column) stay packed in prc (pr << 16 | pc,
    // ~0 when not real; a_H, a_W < 65536), so that the epilogue needs no division by a_W; the window split is unsigned
    // (win >= 0, nwx > 0) and real has no short-circuit branches (a_H, a_W > 0).
    uint prc;
    uint2 xa[KC], xb[KC];
    {
        const int win = win0 < nwin ? win0 : nwin - 1;
        const int wy = (int)((uint)win / (uint)nwx), wx = win - wy * nwx;
        const int pr = 4 * wy - a_oy + (int)(t >> 2), pc = 4 * wx - a_ox + (int)(t & 3u);
        const bool real = (win0 < nwin) & ((uint)pr < (uint)a_H) & ((uint)pc < (uint)a_W);
        prc = real ? ((uint)pr << 16) | (uint)pc : 0xffffffffu;
        const int mr = wg_mirror(pr, a_H), mc = wg_mirror(pc, a_W);
        const uint src = (uint)(mr * a_W + mc) * (uint)C_ + hh * (uint)HALF;
        uint4 q0 = wg_ld128(a_x, src), q1 = wg_ld128(a_x, src + 16u);
        uint4 r0 = wg_ld128(a_xadd, src), r1 = wg_ld128(a_xadd, src + 16u);
        xa[0] = q0.xy; xa[1] = q0.zw; xa[2] = q1.xy; xa[3] = q1.zw;
        xb[0] = r0.xy; xb[1] = r0.zw; xb[2] = r1.xy; xb[3] = r1.zw;
    }
    v8f ng[KC], nb[KC];
    norm_load(L_G1, L_BP, vw128, vw32, ng, nb);   // before the barrier (HIP issues them there too)
    commit(0u, tid, 2u);
    wg_barrier_lds();
    uint buf = 0u;

    // Steps 1-2: a = q8(RMSNorm(x + xadd; gamma1)); the residual accumulator starts as x + bp.
    uint2 af[KC];
    v8f acc[KC];
    {
        // FP8 decode by the driver's matrix element conversion (FP8 B fragment -> F32 accumulator: byte i -> value i,
        // v_cvt_pk_f32_fp8; the bits of the lookup table, NaN codes included), then x + xadd as before
        [unroll] for (uint k = 0; k < KC; ++k) {
            const v8f ua = wg_v8f_frag(wg_ags_convert(wg_frag2(xa[k]), WG_AGS_T_FP8_B, WG_AGS_T_F32_ACC));
            const v8f ub = wg_v8f_frag(wg_ags_convert(wg_frag2(xb[k]), WG_AGS_T_FP8_B, WG_AGS_T_F32_ACC));
            [unroll] for (uint v = 0; v < 8u; ++v) acc[k].v[v] = ua.v[v] + ub.v[v];
        }
        norm_apply2(acc, norm_scale(acc), ng, nb, af);
    }

    // Steps 3-7, one head at a time; one staged unit per head (QKV fragments, then the head's slice of Wp).
    [loop] for (uint hd = 0; hd < 2u; ++hd) {
        const bool last = hd == 1u;
        if (last) fetch_mlp(L_MLP(0), tid); else fetch_head(L_HEAD(hd + 1u), tid);
        const uint U = buf * (MAXU / 4u) + lane2(lane);
        uint2 uf[2];
        uint4 vt[2];
        {
            uint2 fr[2][KC][2];
            [unroll] for (uint mt = 0; mt < 2u; ++mt)
                [unroll] for (uint kc = 0; kc < KC; ++kc) {
                    fr[mt][kc][0] = lfrag(U, (mt * KC + kc) * 2u);
                    fr[mt][kc][1] = lfrag(U, (mt * KC + kc) * 2u + 1u);
                }
            v8f accq[2], accv[2];
            [unroll] for (uint mt2 = 0; mt2 < 2u; ++mt2) { accq[mt2] = wg_splat(0.0f); accv[mt2] = wg_splat(0.0f); }
            [unroll] for (uint kc2 = 0; kc2 < KC; ++kc2)
                [unroll] for (uint mt2 = 0; mt2 < 2u; ++mt2) {
                    accq[mt2] = wg_wmma_fp8(fr[mt2][kc2][0], af[kc2], accq[mt2]);
                    accv[mt2] = wg_wmma_fp8(af[kc2], fr[mt2][kc2][1], accv[mt2]);
                }
            [unroll] for (uint mt5 = 0; mt5 < 2u; ++mt5) { uf[mt5] = wg_pack_q8_nc_ags(accq[mt5]); vt[mt5] = wg_to_f16_ags(accv[mt5]); }
        }
        uint2 fp[KC][2];   // this head's Wp fragments, loaded before the attention's matrix calls
        [unroll] for (uint mt3 = 0; mt3 < KC; ++mt3) {
            fp[mt3][0] = lfrag(U, QKV_FRAGS + mt3 * 2u);
            fp[mt3][1] = lfrag(U, QKV_FRAGS + mt3 * 2u + 1u);
        }
        // Lane (t, h) holds the logits S(t, s) of 8 keys; the position bias is packed in the same order.
        const v8f bias = wg_load8f(vwb, L_BIAS + hd * 1024u);
        uint2 of[2];
        {
            v8f s = wg_wmma_fp8(uf[0], uf[0], wg_splat(0.0f));   // S = U^T U
            s = wg_wmma_fp8(uf[1], uf[1], s);
            float c[8], q[8], zp[8], z[8], e[8];
            [unroll] for (uint v = 0; v < 8u; ++v) c[v] = s.v[v] + bias.v[v];
            [unroll] for (uint v1 = 0; v1 < 8u; ++v1) c[v1] = wg_med3f(c[v1], asfloat(0xc20179dbu), asfloat(0x420179dbu));
            [unroll] for (uint v2 = 0; v2 < 8u; ++v2) q[v2] = c[v2] * c[v2];
            [unroll] for (uint v3 = 0; v3 < 8u; ++v3) q[v3] = q[v3] * c[v3];
            [unroll] for (uint v4 = 0; v4 < 8u; ++v4) zp[v4] = mad(q[v4], asfloat(0xb9a6cc38u), c[v4]);   // c - c^3 / (3 cap^2)
            [unroll] for (uint v5 = 0; v5 < 8u; ++v5) z[v5] = asfloat(0x3f028297u) * zp[v5];              // * tau log2e
#if DEC_D_SMAXMX
            // max RN(k zp) = RN(k max zp): k = tau log2e > 0 and rounding is monotone (zp is finite: the clamp drops NaN);
            // -RN(k m) = RN(-k m), so the exponent argument fma(k, zp, -mx) is zp K + nmx (one v_fmamk)
            float mx = wg_fmax(wg_fmax(wg_fmax(zp[0], zp[1]), wg_fmax(zp[2], zp[3])),
                               wg_fmax(wg_fmax(zp[4], zp[5]), wg_fmax(zp[6], zp[7])));
            mx = wg_fmax(mx, wg_xor16(mx));
            const float nmx = asfloat(0xbf028297u) * mx;
            // a second register with the same bits for the odd values (dual-issue needs the two src1 in other banks)
            const float nmx2 = asfloat(asuint(nmx) ^ (0u - (wg_grid_ >> 31)));
            [unroll] for (uint v6 = 0; v6 < 8u; ++v6) e[v6] = mad(zp[v6], asfloat(0x3f028297u), (v6 & 1u) ? nmx2 : nmx);   // fused as in HIP
#else
            float mx = wg_fmax(wg_fmax(wg_fmax(z[0], z[1]), wg_fmax(z[2], z[3])),
                               wg_fmax(wg_fmax(z[4], z[5]), wg_fmax(z[6], z[7])));
            mx = wg_fmax(mx, wg_xor16(mx));
            [unroll] for (uint v6 = 0; v6 < 8u; ++v6) e[v6] = mad(asfloat(0x3f028297u), zp[v6], -mx);   // fused as in HIP
#endif
            [unroll] for (uint v7 = 0; v7 < 8u; ++v7) e[v7] = wg_exp2(e[v7]);
            float sum = ((e[0] + e[1]) + (e[2] + e[3])) + ((e[4] + e[5]) + (e[6] + e[7]));
            sum = sum + wg_xor16(sum);
            const float rs = wg_rcp(sum);
#ifndef DEC_SMAX_MIX
#define DEC_SMAX_MIX 1
#endif
#if DEC_SMAX_MIX
            // p = (f16)(e rs) with one rounding (HIP v_fma_mixlo_f16 rs, e, 0): mad(e, rs, 0) into the driver's F32 -> F16
            // matrix conversion is folded into exactly that instruction.
            v8f pm;
            [unroll] for (uint v8 = 0; v8 < 8u; ++v8) pm.v[v8] = mad(e[v8], rs, 0.0f);
            const uint4 p = wg_to_f16_ags(pm);
#else
            // p = (f16)(e rs) with one rounding (HIP v_fma_mixlo_f16 rs, e, 0): the product rounded to odd in FP32, then
            // RNE to FP16 (24 >= 11 + 2). Round to odd for e rs >= 0 finite, branch-free (4 VALU, no compares to SGPRs):
            // pr = the nearest-even product, pe = the exact residual (never -0); truncate (u - 1 when pe < 0), then set
            // the last bit when inexact: (u + (pe >> 31)) | min(bits(pe), 1). This is the same function as "pr plus
            // one ulp toward the exact value when pr is inexact and even" (an FP32 underflow gives FP16 zero either way).
            float pr[8], pe[8];
            [unroll] for (uint v8 = 0; v8 < 8u; ++v8) pr[v8] = e[v8] * rs;
            [unroll] for (uint v9 = 0; v9 < 8u; ++v9) pe[v9] = mad(e[v9], rs, -pr[v9]);
            v8f po;
            uint rs8[8], rm8[8];
            [unroll] for (uint va = 0; va < 8u; ++va) rs8[va] = asuint(pr[va]) + (uint)(asint(pe[va]) >> 31);
            [unroll] for (uint vb = 0; vb < 8u; ++vb) rm8[vb] = min(asuint(pe[vb]), 1u);
            [unroll] for (uint vc = 0; vc < 8u; ++vc) po.v[vc] = asfloat(rs8[vc] | rm8[vc]);
            const uint4 p = wg_to_f16_ags(po);
#endif
            [unroll] for (uint dt = 0; dt < 2u; ++dt) of[dt] = wg_pack_q8_nc_ags(wg_wmma_f16(vt[dt], p, wg_splat(0.0f)));
        }
        // x1 += Wp[:, head hd] q8(o_hd)
        [unroll] for (uint mt4 = 0; mt4 < KC; ++mt4)
            [unroll] for (uint dt2 = 0; dt2 < 2u; ++dt2) acc[mt4] = wg_wmma_fp8(fp[mt4][dt2], of[dt2], acc[mt4]);
        if (last) norm_load(L_G2, L_B2, vw128, vw32, ng, nb);   // norm2's gamma / beta before the last barrier
        commit((buf ^ 1u) * MAXU, tid, last ? MLP_ROWS : 2u);
        wg_barrier_lds();
        buf ^= 1u;
    }

    // Step 8: m = q8(RMSNorm(x1; gamma2)), then b2 into the residual accumulator.
    uint2 mf[KC];
    {
        norm_apply2(acc, norm_scale(acc), ng, nb, mf);
    }

    // Steps 9-10: x2 = x1 + W2 q8(GELU(W1 m + b1)) + b2, two 16-unit tiles of the hidden layer at a time.
    v8f hb0[MLP_GROUP];   // b1 of a unit's first group, loaded one unit ahead
    [unroll] for (uint t1 = 0; t1 < (uint)MLP_GROUP; ++t1) hb0[t1] = wg_load8f(vw32, L_B1 + t1 * 64u);
    [loop] for (uint u = 0; u < (uint)NMLP; ++u) {
        const bool more = u + 1u < (uint)NMLP;
        const uint U = buf * (MAXU / 4u) + lane2(lane);
#if DEC_GELU == 3
        // the unit's b1 (the W1 accumulators' start) before the next unit's weight fetch: vector-memory loads return in
        // order, so a bias load issued after the fetch waits for it
        v8f hb[HPU];
        [unroll] for (uint t0 = 0; t0 < (uint)MLP_GROUP; ++t0) hb[t0] = hb0[t0];
        [unroll] for (uint t2 = MLP_GROUP; t2 < (uint)HPU; ++t2) hb[t2] = wg_load8f(vw32, L_B1 + (u * (uint)HPU + t2) * 64u);
        if (more) [unroll] for (uint t3 = 0; t3 < (uint)MLP_GROUP; ++t3) hb0[t3] = wg_load8f(vw32, L_B1 + ((u + 1u) * (uint)HPU + t3) * 64u);
#ifndef DEC_FETCH_LATE
#define DEC_FETCH_LATE 0
#endif
#if !DEC_FETCH_LATE
        if (more) fetch_mlp(L_MLP(u + 1u), tid);
#endif
        // software-pipelined by one group: W1 of group j0 + MLP_GROUP is issued between the table loads of group j0
        // and the rest of its GELU (same operations, same accumulation order)
        v8f hcur[MLP_GROUP];
        {
            uint2 f1[MLP_GROUP][KC];
            [unroll] for (uint g = 0; g < (uint)MLP_GROUP; ++g)
                [unroll] for (uint kc = 0; kc < KC; ++kc) f1[g][kc] = lfrag(U, g * 2u * KC + kc);
            [unroll] for (uint g1 = 0; g1 < (uint)MLP_GROUP; ++g1)
                hcur[g1] = hb[g1];
            [unroll] for (uint kc2 = 0; kc2 < KC; ++kc2)
                [unroll] for (uint g2 = 0; g2 < (uint)MLP_GROUP; ++g2) hcur[g2] = wg_wmma_fp8(f1[g2][kc2], mf[kc2], hcur[g2]);
        }
        DecGeluPre gp = dec_gelu_pre(hcur, 0u);
        [unroll] for (uint j0 = 0; j0 < (uint)HPU; j0 += (uint)MLP_GROUP) {
            uint2 f2[MLP_GROUP][KC];
            [unroll] for (uint g = 0; g < (uint)MLP_GROUP; ++g)
                [unroll] for (uint kc = 0; kc < KC; ++kc) f2[g][kc] = lfrag(U, (j0 + g) * 2u * KC + KC + kc);
            v8f hnext[MLP_GROUP];
            if (j0 + MLP_GROUP < (uint)HPU) {
                uint2 f1[MLP_GROUP][KC];
                [unroll] for (uint g = 0; g < (uint)MLP_GROUP; ++g)
                    [unroll] for (uint kc = 0; kc < KC; ++kc) f1[g][kc] = lfrag(U, (j0 + MLP_GROUP + g) * 2u * KC + kc);
                [unroll] for (uint g1 = 0; g1 < (uint)MLP_GROUP; ++g1)
                    hnext[g1] = hb[j0 + MLP_GROUP + g1];
                [unroll] for (uint kc2 = 0; kc2 < KC; ++kc2)
                    [unroll] for (uint g2 = 0; g2 < (uint)MLP_GROUP; ++g2) hnext[g2] = wg_wmma_fp8(f1[g2][kc2], mf[kc2], hnext[g2]);
            } else {
                [unroll] for (uint g7 = 0; g7 < (uint)MLP_GROUP; ++g7) hnext[g7] = hcur[g7];
            }
            uint2 hf[MLP_GROUP];
            if (j0 + MLP_GROUP < (uint)HPU) {
                DecGeluPre gpn;
                dec_gelu_fin_pre(hcur, gp, j0 / MLP_GROUP, hf, hnext, gpn);
                gp = gpn;
            } else {
                dec_gelu_fin(hcur, gp, j0 / MLP_GROUP, hf);
            }
#if DEC_FETCH_LATE
            if (more && j0 + 2u * MLP_GROUP == (uint)HPU) fetch_mlp(L_MLP(u + 1u), tid);   // after the unit's last table loads
#endif
            [unroll] for (uint g4 = 0; g4 < (uint)MLP_GROUP; ++g4)
                [unroll] for (uint mt = 0; mt < KC; ++mt) acc[mt] = wg_wmma_fp8(f2[g4][mt], hf[g4], acc[mt]);
            [unroll] for (uint g8 = 0; g8 < (uint)MLP_GROUP; ++g8) hcur[g8] = hnext[g8];
        }
#else
        if (more) fetch_mlp(L_MLP(u + 1u), tid);
        [loop] for (uint j0 = 0; j0 < (uint)HPU; j0 += (uint)MLP_GROUP) {
            uint2 f1[MLP_GROUP][KC], f2[MLP_GROUP][KC];
            v8f hacc[MLP_GROUP];
            [unroll] for (uint g = 0; g < (uint)MLP_GROUP; ++g) {
                [unroll] for (uint kc = 0; kc < KC; ++kc) {
                    f1[g][kc] = lfrag(U, (j0 + g) * 2u * KC + kc);
                    f2[g][kc] = lfrag(U, (j0 + g) * 2u * KC + KC + kc);
                }
                hacc[g] = wg_load8f(vw32, L_B1 + (u * (uint)HPU + j0 + g) * 64u);
            }
            [unroll] for (uint kc2 = 0; kc2 < KC; ++kc2)
                [unroll] for (uint g2 = 0; g2 < (uint)MLP_GROUP; ++g2) hacc[g2] = wg_wmma_fp8(f1[g2][kc2], mf[kc2], hacc[g2]);
            uint2 hf[MLP_GROUP];
#if DEC_GELU == 2
            // q8(GELU) (sigmoid form, WG_GELU_MODE 2, then wg_pack_q8_nc) of the 16 values by the exact 2 KB table
            wg_gelu_q8_lds2(hacc[0], hacc[1], hf[0], hf[1]);
#elif DEC_GELU == 1
            // q8(GELU) by the exact arena table (wg_gelu_lut.hlsli): all loads first
            wg_gelu_lut_e ge[MLP_GROUP];
            [unroll] for (uint g5 = 0; g5 < (uint)MLP_GROUP; ++g5) ge[g5] = wg_gelu_lut_fetch8(hacc[g5]);
            [unroll] for (uint g3 = 0; g3 < (uint)MLP_GROUP; ++g3) hf[g3] = wg_gelu_lut_finish8(hacc[g3], ge[g3]);
#else
            // GELU (sigmoid form, WG_GELU_MODE 2) of the 16 values: x rcp(1 + 2^(x fma(x x, -0.10029031, -2.30816186)))
            float gx[16], gt[16];
            [unroll] for (uint i = 0; i < 16u; ++i) gx[i] = hacc[i >> 3].v[i & 7u];
            [unroll] for (uint i1 = 0; i1 < 16u; ++i1) gt[i1] = gx[i1] * gx[i1];
            [unroll] for (uint i2 = 0; i2 < 16u; ++i2) gt[i2] = mad(gt[i2], asfloat(0xbdcd6502u), asfloat(0xc013b8edu));
            [unroll] for (uint i3 = 0; i3 < 16u; ++i3) gt[i3] = gx[i3] * gt[i3];
            [unroll] for (uint i4 = 0; i4 < 16u; ++i4) gt[i4] = wg_exp2(gt[i4]);
            [unroll] for (uint i5 = 0; i5 < 16u; ++i5) gt[i5] = 1.0f + gt[i5];
            [unroll] for (uint i6 = 0; i6 < 16u; ++i6) gt[i6] = wg_rcp(gt[i6]);
            v8f ge[MLP_GROUP];
            [unroll] for (uint i7 = 0; i7 < 16u; ++i7) ge[i7 >> 3].v[i7 & 7u] = gx[i7] * gt[i7];
            [unroll] for (uint g3 = 0; g3 < (uint)MLP_GROUP; ++g3) hf[g3] = wg_pack_q8_nc_ags(ge[g3]);
#endif
            [unroll] for (uint g4 = 0; g4 < (uint)MLP_GROUP; ++g4)
                [unroll] for (uint mt = 0; mt < KC; ++mt) acc[mt] = wg_wmma_fp8(f2[g4][mt], hf[g4], acc[mt]);
        }
#endif
        if (more) {
            commit((buf ^ 1u) * MAXU, tid, MLP_ROWS);
            wg_barrier_lds();
            buf ^= 1u;
        }
    }

    // ---- dec0 as the epilogue (dec1_dec0.hip Dec0Epi::run) ----
    const bool real = prc != 0xffffffffu;
    const int rr = real ? (int)(prc >> 16) : 0, cc = real ? (int)(prc & 0xffffu) : 0;
    const int x0 = 2 * cc, y = 2 * rr + (int)hh;
    const bool row_ok = real && y < P_rh;
    uint mv0, mv1;
    {
        const uint npix = (uint)(P_rw * P_rh);
        const uint pix = (uint)(wg_med3i(y, 0, P_rh - 1) * P_rw + wg_min_i(x0, P_rw - 1));
        const uint mo = pix + 1u < npix ? pix : npix - 2u;
        const uint2 mv = wg_ld64(a_motion, mo * 4u);
        mv0 = pix == mo ? mv.x : mv.y;
        mv1 = mv.y;
    }
    // dec0's B fragments from the block output, exactly the bytes dec1 would store; lane halves swap two 8-byte groups.
    uint2 o[KC];
    // wave-uniform skip of the clamp: when every value of the wave is NaN or |x| <= 448 (the driver's max drops NaN),
    // the clamp is the identity and the plain conversion gives the same bytes (NaN -> 0xff on both sides)
    {
        float am = 0.0f;
        [unroll] for (uint k0 = 0; k0 < KC; ++k0)
            am = max(max(max(am, abs(acc[k0].v[0])), max(abs(acc[k0].v[1]), abs(acc[k0].v[2]))),
                     max(max(abs(acc[k0].v[3]), abs(acc[k0].v[4])), max(max(abs(acc[k0].v[5]), abs(acc[k0].v[6])), abs(acc[k0].v[7]))));
        if (WaveActiveAllTrue(am <= 448.0f)) {
            [unroll] for (uint k = 0; k < KC; ++k) o[k] = wg_pack_q8_nc_ags(acc[k]);
        } else {
            [unroll] for (uint k = 0; k < KC; ++k) o[k] = dec_pack_q8_ags(acc[k]);
        }
    }
    const bool h1 = hh != 0u;
    uint snd0 = h1 ? o[0].x : o[2].x, snd1 = h1 ? o[0].y : o[2].y, snd2 = h1 ? o[1].x : o[3].x, snd3 = h1 ? o[1].y : o[3].y;
    const uint2 r01 = uint2(wg_xor16_u(snd0), wg_xor16_u(snd1)), r23 = uint2(wg_xor16_u(snd2), wg_xor16_u(snd3));
    uint2 bf[4];
    bf[0] = h1 ? r01 : o[0];
    bf[1] = h1 ? r23 : o[1];
    bf[2] = h1 ? o[2] : r01;
    bf[3] = h1 ? o[3] : r23;

    wg_barrier_lds();   // the block's weight buffers are free
    // dec_stage: the 640 chunks of the tables and fragments to LDS byte 16; lanes past the end repeat the last chunk
    // into the free bytes 10256 .. 12303 (no lane test)
    [unroll] for (uint i = 0; i < 3u; ++i) {
        const uint idx = tid + i * 256u;
        lds_st128(16u + idx * 16u, wg_ld128(a_wdec, kOffTables + 16u * min(idx, kLdsStaged / 16u - 1u)));
    }
    wg_barrier_lds();

    // Projection (spec 05 section 1): 5 tiles of 16 outputs, K = 64 in two FP16-rounded halves.
    uint code[10];
    {
        uint2 fd[5][4];
        [unroll] for (uint mt = 0; mt < 5u; ++mt)
            [unroll] for (uint kb = 0; kb < 4u; ++kb) fd[mt][kb] = lfrag(kFragLds / 4u + lane2(lane), mt * 4u + kb);
        [unroll] for (uint mt2 = 0; mt2 < 5u; ++mt2) {
            v8f a = wg_load8f(a_wdec + hh * 32u + oz0, mt2 * 64u);
            a = wg_wmma_fp8(fd[mt2][0], bf[0], a);
            a = wg_wmma_fp8(fd[mt2][1], bf[1], a);
            a = round_f16_n(a, mt2 < 4u ? 8u : 2u);   // u (tile 4: only slots 0 and 1 are outputs)
            a = wg_wmma_fp8(fd[mt2][2], bf[2], a);
            a = wg_wmma_fp8(fd[mt2][3], bf[3], a);
            a = round_f16_n(a, mt2 < 4u ? 8u : 2u);   // v
            const uint2 qq = wg_pack_q8_inner_ags(a);   // h = q8(v)
            code[2u * mt2] = mt2 < 4u ? qq.x : qq.x & 0xffffu;
            code[2u * mt2 + 1u] = mt2 < 4u ? qq.y : 0u;
        }
    }
    const Prev2 q = prev_fetch2(x0, y, uint2(mv0, mv1));

    // Decode the lane's pixels (x0, y) and (x0 + 1, y) (every lane, so that the FP16 roundings can use the matrix
    // conversion; decoder.hip decode_pixel), then store those inside R (pack12 images).
    // A (section 3): codes from the table.
    const uint2 A = (tabu2(kTabA, cbytes(code, 0u)) & 0xffu) | ((tabu2(kTabA, cbytes(code, 1u)) & 0xffu) << 8) |
                    ((tabu2(kTabA, cbytes(code, 2u)) & 0xffu) << 16);
    // B (section 4): softmax in single; exp(x) = 2^n 2^(r log2 e), x = n ln2 + r (two-part ln2); e / s as q = e r,
    // q += fma(-s, q, e) r with r = rcp(s) (WG_DEC_BDIV). 2^n: exact power of two (n <= 0; results below 2^-126 are
    // flushed like every FP32 denormal here; they cannot change the stored bytes).
    float2 bx[3], bn[3], br[3];
    {
        // bytes 0..7 (pixel 0: 3, 4, 5) and 20..23 (pixel 1: 20, 21, 22); byte i -> value i
        const v8f f0 = wg_v8f_frag(wg_ags_convert(wg_frag2(uint2(code[0], code[1])), WG_AGS_T_FP8_B, WG_AGS_T_F32_ACC));
        const v8f f1 = wg_v8f_frag(wg_ags_convert(wg_frag2(uint2(code[5], code[5])), WG_AGS_T_FP8_B, WG_AGS_T_F32_ACC));
        bx[0] = float2(f0.v[3], f1.v[0]); bx[1] = float2(f0.v[4], f1.v[1]); bx[2] = float2(f0.v[5], f1.v[2]);
    }
    const float2 m = max(max(bx[0], bx[1]), bx[2]);
    [unroll] for (uint i1 = 0; i1 < 3u; ++i1) bx[i1] = bx[i1] - m;
    [unroll] for (uint i2 = 0; i2 < 3u; ++i2) bn[i2] = round(bx[i2] * asfloat(0x3fb8aa3bu));
    [unroll] for (uint i3 = 0; i3 < 3u; ++i3) br[i3] = mad(-bn[i3], asfloat(0x3f317200u), bx[i3]);
    [unroll] for (uint i4 = 0; i4 < 3u; ++i4) br[i4] = mad(-bn[i4], asfloat(0x35bfbe8eu), br[i4]);
    [unroll] for (uint i5 = 0; i5 < 3u; ++i5) br[i5] = exp2(br[i5] * asfloat(0x3fb8aa3bu));
    [unroll] for (uint i6 = 0; i6 < 3u; ++i6) {
        const int2 ni = (int2)bn[i6];
#if DEC_D_EXP2N
        // e = ldexp(2^(r log2 e), n): 2^n by v_exp_f32 of the integer n (n in [-1293, 0]: bit-identical to the form
        // below for every n in [-2047, 0] and -0, GPU-checked; 0 below 2^-126)
        bx[i6] = br[i6] * exp2(bn[i6]);
#else
        // e = ldexp(2^(r log2 e), n): 2^n for n >= -126, else 0, as max(n + 127, 0) << 23 (no compare to SGPRs)
        bx[i6] = br[i6] * asfloat((uint2)max(ni + 127, 0) << 23);
#endif
    }
    const float2 bs = (bx[0] + bx[1]) + bx[2];
    const float2 rs = rcp(bs);
    float2 bq[3];
    [unroll] for (uint i7 = 0; i7 < 3u; ++i7) bq[i7] = bx[i7] * rs;
    [unroll] for (uint i8 = 0; i8 < 3u; ++i8) bq[i8] = mad(mad(-bs, bq[i8], bx[i8]), rs, bq[i8]);
    const uint2 B = unorm8(bq[0]) | (unorm8(bq[1]) << 8) | (unorm8(bq[2]) << 16);
    // C (section 5): C0, C1 from the table; C2 = RN16(sx sy rho): HIP rounds sx sy to FP32, then (sx sy) rho once to
    // FP16 (v_fma_mixlo_f16 .., 0: a zero product gives +0): round to odd in FP32, then the RNE conversion below.
    const uint2 a6 = cbytes(code, 6u), a7 = cbytes(code, 7u);
    const uint2 C0 = (tabu2(kTabA, a6) >> 16) | (tabu2(kTabA, a7) & 0xffff0000u);
    const float2 sxy = tabf2(kTabSx, a6) * tabf2(kTabSx, a7), rho = tabf2(kTabRho, cbytes(code, 8u));
#if !DEC_D_C2MIX
    const float2 cpr = sxy * rho, cpe = mad(sxy, rho, -cpr);
    const bool2 czero = or(sxy == 0.0f, rho == 0.0f);
#endif
    // the latent update's inputs (section 7): z(a) in the low half of the ZG table, g(a) in the high half
    float2 P[4];
    prev_values2(q, P);
#if DEC_D_C2MIX
    // mad(sxy, rho, 0) into the RNE conversion is folded into v_fma_mixlo_f16 sxy, rho, 0: HIP's own instruction (a zero
    // product gives +0 there as here)
    const uint2 C1 = uint2(dec_f2h_hw(mad(sxy.x, rho.x, 0.0f)), dec_f2h_hw(mad(sxy.y, rho.y, 0.0f)));
#else
    v8f cv = wg_splat(0.0f);
    cv.v[0] = asfloat(wg_round_odd(cpr.x, cpe.x));
    cv.v[1] = asfloat(wg_round_odd(cpr.y, cpe.y));
    const uint2 C1 = select(czero, 0u, uint2(dec_f2h_hw(cv.v[0]), dec_f2h_hw(cv.v[1])));
#endif
    // Gated latent update (section 7), as HIP in packed FP16 (pixel pairs per channel), every operation rounded to FP16:
    // w = 1 - z, L = (w p) + (z g), p = RN16(P); opaque links keep the multiplies and adds apart. p of the 8 values by
    // the matrix conversion (v_cvt_f16_f32 RNE straight into the halves, no repacking).
    const uint oz = 0u - (wg_grid_ >> 31);
    uint lm[4];
    v8f pv;
    [unroll] for (uint c8 = 0; c8 < 4u; ++c8) { pv.v[2u * c8] = P[c8].x; pv.v[2u * c8 + 1u] = P[c8].y; }
    const uint4 p4 = wg_to_f16_ags(pv);
    [unroll] for (uint ch = 0; ch < 4u; ++ch) {
        const uint2 tz = tabu2(kTabZG, cbytes(code, 13u + ch)), tg = tabu2(kTabZG, cbytes(code, 9u + ch));
        const half2 z = wg_as_h2((tz.x & 0xffffu) | (tz.y << 16)), g = wg_as_h2((tg.x >> 16) | (tg.y & 0xffff0000u));
        const half2 p = wg_as_h2(p4[ch]);
        const half2 w = wg_as_h2(wg_h2_bits((half2)1.0h - z) ^ oz);
        const half2 wp = wg_as_h2(wg_h2_bits(w * p) ^ oz), zg = wg_as_h2(wg_h2_bits(z * g) ^ oz);
        const half2 L = wp + zg;
        // latent code (int)rint((float)L 127) & 0xff of both pixels: fma(L, 127, 1536) rounded once to FP16 is
        // 1536 + rint(L 127) (L 127 is exact, the FP16 step there is 1, ties to even as rint) while |L 127| < 512, so
        // its low byte is the code. GPU-checked for every FP16 L with |L| < 4.035 (build\diet-dec\prove\ddp3_check.py);
        // here |L| <= |w p| + |z g| <= 2.01 (z in [0, 1], g in [-1, 1] for the clamped codes, |p| <= 1.0001: a convex
        // blend of snorm values, 0 where invalid). A NaN L (only from non-finite blend weights) is the default NaN,
        // low byte 0 = HIP's cvt_i32(NaN) code.
        lm[ch] = wg_h2_bits(mad(L, half2(127.0h, 127.0h), half2(1536.0h, 1536.0h)));
    }
    const uint L0 = (lm[0] & 0xffu) | ((lm[1] & 0xffu) << 8) | ((lm[2] & 0xffu) << 16) | (lm[3] << 24);
    const uint L1 = ((lm[0] >> 16) & 0xffu) | ((lm[1] >> 8) & 0xff00u) | (lm[2] & 0xff0000u) | ((lm[3] << 8) & 0xff000000u);
    const uint2 w0 = A | (B << 24), w1 = (B >> 8) | (C1 << 16);   // image A words of both pixels
    const int rw = P_rw;
    const uint i0 = (uint)(y * rw + x0);
    if (row_ok && x0 + 1 < rw) {
        wg_st128(a_A, i0 * 8u, uint4(w0.x, w1.x, w0.y, w1.y));
        wg_st64(a_C, i0 * 4u, C0);
        wg_st64(a_latent, i0 * 4u, uint2(L0, L1));
    } else if (row_ok && x0 + 1 == rw) {
        // Last column of an odd-width R. Not the two 32-bit stores C, latent next to each other: the driver (RX 9070 XT,
        // this DXC listing) then dropped the first of them (two dword stores at uniform base + the same offset), so the
        // 64-bit A store goes between them.
        wg_st32(a_C, i0 * 4u, C0.x);
        wg_st64(a_A, i0 * 8u, uint2(w0.x, w1.x));
        wg_st32(a_latent, i0 * 4u, L0);
    }
}
