// SPDX-License-Identifier: MIT
// D3D12 port of model K's wide block (clean/k/kernels/k_block_wide.hip, template k_block_wide<C, HEADS, KMIX_SOFTMAX,
// IN, CIN, CNEXT, NG>, FP32 accumulation; HIP build: k_block_wide.co). WrongGPU's own code. Spec 41 block (spec 40 stage
// table), exactly the HIP kernel's operations and orders:
//   - every matrix product is v_wmma_f32_16x16x16_f16 (driver channel, wg_wmma_f16) with the same A / B operands, and
//     every FP32 accumulator sums in the HIP order (bias first; q / k / v over kt; scores bias then ct; z over st; Wo
//     over the z tiles of heads 0, 1, ..; W1 over kt; W2 over hidden tiles 0, 1, ..; merge / expand over kt), rounded
//     once to FP16 to nearest even by the driver's matrix conversion (wg_to_f16_ags, op 0x32: RNE);
//   - every FP16 elementwise step is one IEEE FP16 operation (packed half2 here, v_pk_* as HIP). The driver's compiler
//     fuses an FP16 multiply into a following add / sub (v_fma_mix / v_pk_fma) and regroups multiplication chains even
//     under -Gis (wg_math.hlsli "driver exceptions"), so every multiply whose result feeds an add, a sub or another
//     multiply goes through kw_opq (an integer xor with a zero the compiler cannot see): the activation (b |t|, t p),
//     the softmax polynomial (u^2, kappa u^2), the normalization (f16(x^2), r g1);
//   - normalization: s = f16(sum f16(x_c^2)) (FP32 sum in channel order, then the lane ^ 16 half), r = 0 for s = 0 else
//     RN16(v_rsq_f32(s)) (wg_f2h: RNE; the plain f32tof16 rounds toward zero on this driver);
//   - softmax weights: w = v_exp_f32(g), FP32 sum in key order plus the lane ^ 16 half, inv = v_rcp_f32(sum), and
//     A = f16(w inv) rounded ONCE from the exact product, as HIP's compiled form (v_fma_mixlo/hi_f16 w * inv + 0, seen in
//     the k_block_wide.co ISA): k_smx.hlsli, the FP32 product and its exact residual summed into the RNE conversion
//     (folded into v_fma_mix*_f16 by the driver), or in the __rodd ports the product rounded to odd, then the conversion
//     (FP32 has 13 more bits than FP16: no double rounding);
//   - mirrored window members, ownership of the output tiles (a group without a last tile computes tile KT - 1 too and
//     does not store it), merge / expand index maps: as HIP.
// Deviations (no effect on any result): weights and LDS operands of each product group are loaded before its multiplies
// (driver channel calls are UAV atomics the compiler never moves loads across): every global weight fragment comes from
// one ring of KW_RD tiles that runs through the whole kernel in product order, across phase borders (the weight stream,
// k_block_wide.hlsl), each loaded through a per-lane base address plus a constant (the load's immediate offset);
// independent accumulators are interleaved (the scores' four key tiles, z's two channel tiles, W1's MM hidden tiles,
// the expand's tiles), each still summing in its own order; the LDS indices are masked to the array's power of two
// (kw_lr / kw_lw); the residual x of the owned tiles is reloaded at step 3 for map inputs; NG = 4 uses another LDS layout that fits D3D12's 32 KB (HIP: 56 KB): K / VT per group unpadded with an xor
// swizzle of the 16-byte chunks, and the round's Z, the MLP's A and its H buffers all start at 0, with extra barriers
// where the phases now overlap (K / VT reads done before Z is written, Z reads done before A / the next round's K / VT,
// A reads done before H, H reads done before the merge's x2). The arithmetic is the same in both layouts.
//
// Compiled once per stage by clean/dx12/build_dx12.py with the defines of the @wg-kernel line (k_block_wide.hlsl):
//   KW_C channels, KW_HEADS heads, KW_IN 0 = map input / 1 = expand input, KW_CIN expand input channels, KW_CNEXT merge
//   output channels (0: no merge), KW_NG wave groups (block = 128 KW_NG lanes).
// Arguments: KBlkParams (184 bytes, struct token s184:p19.b32: 19 pointers as arena offsets, then 8 ints).
#pragma once

static const int KT = KW_C / 16;                 // channel tiles of a token
static const int DT = KW_HEADS * 2;              // channel tiles of z
static const int HT = 4 * KW_C / 16;             // hidden tiles of the MLP
static const int NG = KW_NG;
static const int NW = 4 * NG;                    // waves of the workgroup
static const int NOWN = (KT + NG - 1) / NG;      // output tiles a group owns (at most)
static const int MM = 2;                         // hidden tiles per group and chunk
static const int CH = NG * MM;                   // hidden tiles per chunk
static const int KTI = KW_CIN / 16;              // expand input tiles
static const int TOT = 4 * KW_C / 16;            // expand output tiles
static const int KTM = 4 * KW_C / 16;            // merge input tiles
static const int NTM = KW_CNEXT / 16;            // merge output tiles
#ifndef KW_TIGHT
#define KW_TIGHT (KW_NG == 4)
#endif
static const bool TIGHT = KW_TIGHT;              // the overlapping layout (see above; required for NG = 4)

// LDS layout in halfs (HIP's for NG = 2; the overlapping one for NG = 4)
static const int KVH = TIGHT ? 4096 : 64 * 40 + 32 * 72;   // one group's K [64][32|40] + VT [32][64|72]
static const int ZS = 32 * NG + 8;
static const int OFF_Z = TIGHT ? 0 : NG * KVH;
static const int ATT = TIGHT ? (NG * KVH > 64 * ZS ? NG * KVH : 64 * ZS) : NG * KVH + 64 * ZS;
static const int OFF_A = TIGHT ? 0 : (64 * KW_C <= NG * KVH ? 0 : ATT);
static const int OFF_H = TIGHT ? 0 : OFF_A + 64 * KW_C;
static const int END_H = OFF_H + 2 * CH * 64 * 16;
static const int LDS_HALFS0 = ATT > END_H ? ATT : END_H;
static const int LDS_HALFS = LDS_HALFS0 > 64 * KW_C ? LDS_HALFS0 : 64 * KW_C;
groupshared uint4 kw_lds[LDS_HALFS / 8];         // 16-byte chunks (every access is one chunk)

// ring depth (tiles) of the merge when a wave has more than one merge output tile (4K NG = 2 e4); everything else uses
// the weight stream's ring (KW_RD, k_block_wide.hlsl)
#ifndef KW_PD
#define KW_PD 4
#endif

// ---- helpers --------------------------------------------------------------------------------------------------------
// 0 that the compiler cannot see, with no known bits (grids are below 2^31; 0 or 0xffffffff for the compiler: a 0 / 1
// value would leave the high FP16 half of a packed pair visibly unchanged, and the driver then fused it anyway)
uint kw_zero() { return 0u - (wg_grid_ >> 31); }
uint kw_opq(uint v) { return v ^ kw_zero(); }
uint4 kw_opq4(uint4 v) { return v ^ kw_zero(); }
half2 kw_h(uint v) { return wg_as_h2(v); }
uint kw_u(half2 h) { return wg_h2_bits(h); }

uint4 kw_add8(uint4 a, uint4 b)
{
    uint4 r;
    [unroll] for (uint i = 0; i < 4; ++i) r[i] = kw_u(kw_h(a[i]) + kw_h(b[i]));
    return r;
}
uint4 kw_mul8(uint4 a, uint4 b)
{
    uint4 r;
    [unroll] for (uint i = 0; i < 4; ++i) r[i] = kw_u(kw_h(a[i]) * kw_h(b[i]));
    return r;
}
uint4 kw_splat8(uint h) { const uint d = (h & 0xffffu) | (h << 16); return uint4(d, d, d, d); }

// spec 40 activation, FP16 per operation: t = min(max(x, -2), 2); p = f16(a - f16(b |t|)); q = f16(0.5 + f16(t p));
// act = f16(x q) (k_common.h wgk_act2; maxNum / minNum clamps)
uint4 kw_act8(uint4 x)
{
    const half2 lo = half2(-2.0h, -2.0h), hi = half2(2.0h, 2.0h);
    const half2 A = half2(0.412109375h, 0.412109375h), B = half2(0.0810546875h, 0.0810546875h);
    const half2 half_ = half2(0.5h, 0.5h);
    uint4 r;
    [unroll] for (uint i = 0; i < 4; ++i) {
        const half2 xv = kw_h(x[i]);
        const half2 tv = min(max(xv, lo), hi);
        const half2 bt = kw_h(kw_opq(kw_u(B * abs(tv))));
        const half2 p = A - bt;
        const half2 tp = kw_h(kw_opq(kw_u(tv * p)));
        const half2 q = half_ + tp;
        r[i] = kw_u(xv * q);
    }
    return r;
}

// softmax polynomial of one score fragment: u = clamp(s, -c, c), g = f16(u f16(1 - f16(kappa f16(u u))))
uint4 kw_softpoly8(uint4 s)
{
    const half2 c = half2(21.640625h, 21.640625h), nc = half2(-21.640625h, -21.640625h);
    const half2 kap = (half2)asfloat16((uint16_t2)0x11d5u), one = half2(1.0h, 1.0h);   // 0.0007119178771972656
    uint4 r;
    [unroll] for (uint i = 0; i < 4; ++i) {
        const half2 u = min(max(kw_h(s[i]), nc), c);
        const half2 u2 = kw_h(kw_opq(kw_u(u * u)));
        const half2 k2 = kw_h(kw_opq(kw_u(kap * u2)));
        const half2 o = one - k2;
        r[i] = kw_u(u * o);
    }
    return r;
}

v8f kw_widen8(uint4 a)
{
    v8f r;
    [unroll] for (uint i = 0; i < 4; ++i) {
        r.v[2 * i] = f16tof32(a[i] & 0xffffu);
        r.v[2 * i + 1] = f16tof32(a[i] >> 16);
    }
    return r;
}
uint4 kw_out(v8f c) { return wg_to_f16_ags(c); }
v8f kw_mma(uint4 a, uint4 b, v8f c) { return wg_wmma_f16(a, b, c); }

// The value of v computed in the current (wave-uniform) control flow: a lane read at lane ^ z with z the opaque zero is
// a convergent operation the compiler cannot simplify, so it cannot sink the producer of v (a WMMA chain) into a
// following divergent branch. Without it the merge's whole product chain moved under its store's `if (inside)`, and
// the WMMAs then ran with a partial EXEC mask (wrong results; the driver channel needs uniform control flow).
uint4 kw_pin(uint4 v)
{
    const uint l = WaveGetLaneIndex() ^ (kw_zero() & 31u);
    return uint4(WaveReadLaneAt(v.x, l), WaveReadLaneAt(v.y, l), WaveReadLaneAt(v.z, l), WaveReadLaneAt(v.w, l));
}

// global fragments: pointer (arena byte offset) + element (half) offset
uint4 kw_ldg(uint p, uint e) { return wg_ld128(p, e * 2u); }
void kw_stg(uint p, uint e, uint4 v) { wg_st128(p, e * 2u, v); }

// Mirror about the edge token, edge not repeated (spec 41; k_common.h wgk_mirror), branch free.
int kw_mirror(int p, int n)
{
    const int period = max(2 * (n - 1), 1);
    int r = p % period;
    r = r < 0 ? r + period : r;
    r = r < n ? r : period - r;
    r = (p >= 0 && p < n) ? p : r;
    return n == 1 ? 0 : r;
}

// LDS chunk indices. Every index goes through kw_lr / kw_lw, masked to the power of two at or above the array size
// (no effect: every index is inside the array). The known bound lets the driver prove the byte offsets do not wrap, so
// it keeps 16-byte fragments as ds_load_b128 / ds_load_2addr_b64 instead of splitting them into 2addr_b32 pairs.
static const uint KW_LN = LDS_HALFS / 8;
static const uint KW_LM = (KW_LN > 1024u ? 2048u : KW_LN > 512u ? 1024u : KW_LN > 256u ? 512u : 256u) - 1u;
uint4 kw_lr(uint i) { return kw_lds[i & KW_LM]; }
void kw_lw(uint i, uint4 v) { kw_lds[i & KW_LM] = v; }

uint kw_kidx(int g, int row, int c)    // K[row = token][chunk c of 4] of group g
{
    return TIGHT ? (uint)(g * 512 + row * 4 + (c ^ (row & 3))) : (uint)(g * (KVH / 8) + row * 5 + c);
}
uint kw_vidx(int g, int row, int c)    // VT[row = channel][chunk c of 8 = tokens 8 c ..] of group g
{
    return TIGHT ? (uint)(g * 512 + 256 + row * 8 + (c ^ (row & 7))) : (uint)(g * (KVH / 8) + 320 + row * 9 + c);
}
uint kw_zidx(int t, int c) { return (uint)(OFF_Z / 8 + t * (ZS / 8) + c); }
uint kw_aidx(int t, int c) { return (uint)(OFF_A / 8 + t * (KW_C / 8) + c); }
