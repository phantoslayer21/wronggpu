// SPDX-License-Identifier: GPL-3.0-only
// The HYBRID form's patch merges and expands (WG_L_FORM = hybrid; the unfused shared stages, hyb_swin.hlsl's blocks):
// the M-form kernels' WMMA chains (swin_gemm.hlsl: the bias as the first accumulator, the K groups in order, the same
// fragments of the same packed section) with the hybrid FP8 points (oracle run_stage_q7):
//   patch_merge_c64_hyb / _c96_hyb   merged = FP8(clamp(RN16(bm + Wm z), +-6.28125))   (hyb_merge_q8, NaN kept)
//   expand_c96_hyb / _c128_hyb        E = FP8(RN16(be + We y))                           (hyb_q8_of_chain)
// The weights are the stage's merge / expand section of the hybrid blob (packing.pack_merge / pack_expand layout with
// the original's values: FP16 biases stored as float32, plain FP8 matrices; nothing folded). Arguments and grids as the
// M-form kernels: in, out, w (the section), H, W (merge: the input map's size; expand: the low-resolution size), grid
// ceil(tokens / (8 waves x NT x 16)) of 256 lanes (tokens: merged H/2 x W/2, expand Hl x Wl).
// Staging (performance only, the bytes do not depend on it): the merge stages the fragments of SW_PM_SPLIT output-tile
// groups one group at a time, the expand those of one sub-pixel k at a time when SW_EX_SPLIT (expand_c128: the four
// sub-pixels' 48 KB do not fit D3D12's 32 KB).
// HIP twins: the same names in swin_hyb.co (clean/kernels/swin_hyb.hip: swin.hip's patch_merge / expand with HYB).
// The expands run with their workgroups in reverse order (WG_REV_ORDER, 1-D dispatches; the same bytes), opposite to
// the bottleneck / dec2 block before them, so each starts on the rows its producer wrote last (Model L's hybrid chain).
//
// @wg-kernel patch_merge_c64_hyb co=- block=256 wave=32 args=p,p,p,u,u defines=SW_PM_C=64,SW_PM_CO=96,SW_PM_SPLIT=1
// @wg-kernel patch_merge_c96_hyb co=- block=256 wave=32 args=p,p,p,u,u defines=SW_PM_C=96,SW_PM_CO=128,SW_PM_SPLIT=4
// @wg-kernel expand_c96_hyb co=- block=256 wave=32 args=p,p,p,u,u defines=SW_EX_CL=96,SW_EX_C=64,SW_EX_SPLIT=0,WG_REV_ORDER=1
// @wg-kernel expand_c128_hyb co=- block=256 wave=32 args=p,p,p,u,u defines=SW_EX_CL=128,SW_EX_C=96,SW_EX_SPLIT=1,WG_REV_ORDER=1
//   *_scanmut  the ISA rules' mutants (probe=1, outside the manifest; build_dx12.py requires one per model hybrid
//              kernel): no RN16 before the FP8 pack (HYB_SCAN_MUTANT=1), rule 2 must fail (no norm tree here: rule 1
//              has nothing to find)
// @wg-kernel patch_merge_c64_hyb_scanmut co=- probe=1 block=256 wave=32 args=p,p,p,u,u defines=SW_PM_C=64,SW_PM_CO=96,SW_PM_SPLIT=1,HYB_SCAN_MUTANT=1,SW_ENTRY_OVERRIDE=patch_merge_c64_hyb_scanmut
// @wg-kernel patch_merge_c96_hyb_scanmut co=- probe=1 block=256 wave=32 args=p,p,p,u,u defines=SW_PM_C=96,SW_PM_CO=128,SW_PM_SPLIT=4,HYB_SCAN_MUTANT=1,SW_ENTRY_OVERRIDE=patch_merge_c96_hyb_scanmut
// @wg-kernel expand_c96_hyb_scanmut co=- probe=1 block=256 wave=32 args=p,p,p,u,u defines=SW_EX_CL=96,SW_EX_C=64,SW_EX_SPLIT=0,HYB_SCAN_MUTANT=1,SW_ENTRY_OVERRIDE=expand_c96_hyb_scanmut
// @wg-kernel expand_c128_hyb_scanmut co=- probe=1 block=256 wave=32 args=p,p,p,u,u defines=SW_EX_CL=128,SW_EX_C=96,SW_EX_SPLIT=1,HYB_SCAN_MUTANT=1,SW_ENTRY_OVERRIDE=expand_c128_hyb_scanmut
// Per-32-chunk FP16-rounded exact q7 twins (SW_Q7_CHUNK=1): kord_drow weight packing.
// @wg-kernel patch_merge_c64_q7c_hyb co=- block=256 wave=32 args=p,p,p,u,u defines=SW_PM_C=64,SW_PM_CO=96,SW_PM_SPLIT=1,SW_Q7_CHUNK=1,SW_ENTRY_OVERRIDE=patch_merge_c64_q7c_hyb
// @wg-kernel patch_merge_c96_q7c_hyb co=- block=256 wave=32 args=p,p,p,u,u defines=SW_PM_C=96,SW_PM_CO=128,SW_PM_SPLIT=4,SW_Q7_CHUNK=1,SW_ENTRY_OVERRIDE=patch_merge_c96_q7c_hyb
// @wg-kernel expand_c96_q7c_hyb co=- block=256 wave=32 args=p,p,p,u,u defines=SW_EX_CL=96,SW_EX_C=64,SW_EX_SPLIT=0,SW_Q7_CHUNK=1,WG_REV_ORDER=1,SW_ENTRY_OVERRIDE=expand_c96_q7c_hyb
// @wg-kernel expand_c128_q7c_hyb co=- block=256 wave=32 args=p,p,p,u,u defines=SW_EX_CL=128,SW_EX_C=96,SW_EX_SPLIT=1,SW_Q7_CHUNK=1,WG_REV_ORDER=1,SW_ENTRY_OVERRIDE=expand_c128_q7c_hyb
// @wg-kernel patch_merge_c64_q7c_hyb_scanmut co=- probe=1 block=256 wave=32 args=p,p,p,u,u defines=SW_PM_C=64,SW_PM_CO=96,SW_PM_SPLIT=1,SW_Q7_CHUNK=1,HYB_SCAN_MUTANT=1,SW_ENTRY_OVERRIDE=patch_merge_c64_q7c_hyb_scanmut
// @wg-kernel patch_merge_c96_q7c_hyb_scanmut co=- probe=1 block=256 wave=32 args=p,p,p,u,u defines=SW_PM_C=96,SW_PM_CO=128,SW_PM_SPLIT=4,SW_Q7_CHUNK=1,HYB_SCAN_MUTANT=1,SW_ENTRY_OVERRIDE=patch_merge_c96_q7c_hyb_scanmut
// @wg-kernel expand_c96_q7c_hyb_scanmut co=- probe=1 block=256 wave=32 args=p,p,p,u,u defines=SW_EX_CL=96,SW_EX_C=64,SW_EX_SPLIT=0,SW_Q7_CHUNK=1,HYB_SCAN_MUTANT=1,SW_ENTRY_OVERRIDE=expand_c96_q7c_hyb_scanmut
// @wg-kernel expand_c128_q7c_hyb_scanmut co=- probe=1 block=256 wave=32 args=p,p,p,u,u defines=SW_EX_CL=128,SW_EX_C=96,SW_EX_SPLIT=1,SW_Q7_CHUNK=1,HYB_SCAN_MUTANT=1,SW_ENTRY_OVERRIDE=expand_c128_q7c_hyb_scanmut
#ifndef SW_Q7_CHUNK
#define SW_Q7_CHUNK 0
#endif
#include "wg_platform.hlsli"

WG_ARGS_BEGIN
    uint in_p, out_p, w;   // pointers: input map, output map, weights (bias first, then the fragments)
    int H, W;              // patch merge: the input map's size; expand: the low-resolution size (Hl, Wl)
WG_ARGS_END

#define NT 2
#define THREADS WG_BLOCK
#define SW_CLAMP_SKIP 0   // every FP8 point through the hybrid packs (no clamp-skip shortcut)

#if defined(SW_PM_C)
#define KZ (SW_PM_C / 4)                  // 4C / 16 input groups
#define MO (SW_PM_CO / 16)                // output tiles
#define MOP (MO / SW_PM_SPLIT)            // output tiles per pass
#define WM (4 * SW_PM_CO)                 // byte offset of the fragments (after the bias)
#define SW_LDS_BYTES (MOP * KZ * 256)
#else
#define KC (SW_EX_CL / 16)
#define G (SW_EX_C / 16)
#define WE (16 * SW_EX_C)
#if SW_EX_SPLIT
#define SW_LDS_BYTES (G * KC * 256)       // one sub-pixel's fragments
#else
#define SW_LDS_BYTES (4 * G * KC * 256)
#endif
#endif
#include "swin_common.hlsli"
#include "speed_swin.hlsli"
#include "wg_gelu_lut.hlsli"
#include "wg_hyb.hlsli"

// `bytes` (a multiple of THREADS x 16) from global memory to LDS byte 0, then a barrier.
void copy_to_lds(uint src, uint bytes, uint tid)
{
    [loop] for (uint i = tid; i < bytes / 16u; i += THREADS) sw_lds_st128(i * 16u, wg_ld128(src, i * 16u));
    wg_barrier_lds();
}

#if defined(SW_PM_C)
#if SW_PM_C == 64
#define SW_ENTRY patch_merge_c64_hyb
#else
#define SW_ENTRY patch_merge_c96_hyb
#endif
#ifdef SW_ENTRY_OVERRIDE
#undef SW_ENTRY
#define SW_ENTRY SW_ENTRY_OVERRIDE
#endif
// merged(i, j) = FP8(clamp(RN16(bm + Wm concat(TL, TR, BL, BR)))); lane half h holds the 2C bytes of input row 2i + h.
WG_KERNEL(SW_ENTRY)
{
    WG_SETUP();
    const uint tid = wg_lane_id();
    const uint lane = tid & 31u, t = lane & 15u, hh = lane >> 4;
    const int Wm = W >> 1, ntok = (H >> 1) * Wm;
    const int tok0 = (int)(wg_block_id() * (THREADS / 32) + (tid >> 5)) * NT * 16 + (int)t;
    uint2 zf[NT][KZ];
    int tok[NT];
    [unroll] for (int i = 0; i < NT; ++i) {
        tok[i] = tok0 + 16 * i;
        const int tk = tok[i] < ntok ? tok[i] : ntok - 1;
        const int r = tk / Wm, c = tk - r * Wm;
        const uint off = (uint)((2 * r + (int)hh) * W + 2 * c) * SW_PM_C;
#if SW_Q7_CHUNK
        uint2 zr[KZ];
        [unroll] for (uint q = 0; q < KZ / 2; ++q) {
            uint4 v = wg_ld128(in_p, off + 16u * q);
            zr[2 * q] = v.xy; zr[2 * q + 1] = v.zw;
        }
        HYB_Q7C_L2D(KZ, zr, zf[i], hh);
#else
        [unroll] for (uint q = 0; q < KZ / 2; ++q) {
            uint4 v = wg_ld128(in_p, off + 16u * q);
            zf[i][2 * q] = v.xy; zf[i][2 * q + 1] = v.zw;
        }
#endif
    }
    const uint bm = w + hh * 32u;
    [unroll] for (uint pass = 0; pass < SW_PM_SPLIT; ++pass) {
        if (pass > 0u) wg_barrier_lds();   // every wave is done with the previous pass's fragments
        copy_to_lds(w + WM + pass * SW_LDS_BYTES, SW_LDS_BYTES, tid);
        const uint mop = MOP + sw_zero();
        [loop] for (uint mp = 0; mp < mop; ++mp) {
            const uint mt = pass * MOP + mp;
            uint2 f[KZ];
            [unroll] for (uint g = 0; g < KZ; ++g) f[g] = sw_lfrag_b(sw_lane_base_m(0u, lane), mp * KZ + g);
            const v8f b = wg_load8f(bm, mt * 64u);
            uint2 res[NT];
#if SW_Q7_CHUNK
            v8f acc[NT];
            [unroll] for (int i2 = 0; i2 < NT; ++i2) acc[i2] = b;
            [unroll] for (uint g2 = 0; g2 < KZ; g2 += 2u) {
                [unroll] for (int i3 = 0; i3 < NT; ++i3) {
                    acc[i3] = wg_wmma_fp8(f[g2], zf[i3][g2], acc[i3]);
                    acc[i3] = wg_wmma_fp8(f[g2 + 1u], zf[i3][g2 + 1u], acc[i3]);
                    if (g2 + 2u < KZ) acc[i3] = hyb_round16(acc[i3]);
                }
            }
            [unroll] for (int i4 = 0; i4 < NT; ++i4) res[i4] = hyb_merge_q8(acc[i4]);
#else
            v8f acc[NT];
            [unroll] for (int i2 = 0; i2 < NT; ++i2) acc[i2] = b;
            [unroll] for (uint g2 = 0; g2 < KZ; ++g2)
                [unroll] for (int i3 = 0; i3 < NT; ++i3) acc[i3] = wg_wmma_fp8(f[g2], zf[i3][g2], acc[i3]);
            [unroll] for (int i4 = 0; i4 < NT; ++i4) res[i4] = hyb_merge_q8(acc[i4]);
#endif
            [unroll] for (int i5 = 0; i5 < NT; ++i5)
                if (tok[i5] < ntok) wg_st64(out_p, (uint)tok[i5] * SW_PM_CO + hh * (SW_PM_CO / 2) + 8u * mt, res[i5]);
        }
    }
}
#else
#if SW_EX_CL == 96
#define SW_ENTRY expand_c96_hyb
#else
#define SW_ENTRY expand_c128_hyb
#endif
#ifdef SW_ENTRY_OVERRIDE
#undef SW_ENTRY
#define SW_ENTRY SW_ENTRY_OVERRIDE
#endif
// E = FP8(RN16(be + We y)) per low-resolution token, as an FP8 map E [2 Hl][2 Wl][C] (sub-pixel k at
// (2r + k / 2, 2c + k % 2)).
WG_KERNEL(SW_ENTRY)
{
    WG_SETUP();
    const uint tid = wg_lane_id();
    const uint lane = tid & 31u, t = lane & 15u, hh = lane >> 4;
    const int Hl = H, Wl = W, ntok = Hl * Wl;
    const int tok0 = (int)(wg_block_id() * (THREADS / 32) + (tid >> 5)) * NT * 16 + (int)t;
    uint2 yf[NT][KC];
    int tok[NT];
    [unroll] for (int i = 0; i < NT; ++i) {
        tok[i] = tok0 + 16 * i;
        const int tk = tok[i] < ntok ? tok[i] : ntok - 1;
        const uint off = (uint)tk * SW_EX_CL + hh * (SW_EX_CL / 2);
#if SW_Q7_CHUNK
        uint2 yr[KC];
        [unroll] for (uint q = 0; q < KC / 2; ++q) {
            uint4 v = wg_ld128(in_p, off + 16u * q);
            yr[2 * q] = v.xy; yr[2 * q + 1] = v.zw;
        }
        HYB_Q7C_L2D(KC, yr, yf[i], hh);
#else
        [unroll] for (uint q = 0; q < KC / 2; ++q) {
            uint4 v = wg_ld128(in_p, off + 16u * q);
            yf[i][2 * q] = v.xy; yf[i][2 * q + 1] = v.zw;
        }
#endif
    }
#if !SW_EX_SPLIT
    copy_to_lds(w + WE, SW_LDS_BYTES, tid);
#endif
    uint ob[NT];
    [unroll] for (int i6 = 0; i6 < NT; ++i6) {
        const int r = tok[i6] / Wl, c = tok[i6] - r * Wl;
        ob[i6] = (uint)((2 * r) * (2 * Wl) + 2 * c) * SW_EX_C + hh * (SW_EX_C / 2);
    }
    const uint be = w + hh * 32u;
    [loop] for (uint k = 0; k < 4; ++k) {
#if SW_EX_SPLIT
        // sub-pixel k's fragments (tiles k G .. k G + G - 1, contiguous): wait until every wave is done with k - 1's
        if (k > 0u) wg_barrier_lds();
        copy_to_lds(w + WE + k * SW_LDS_BYTES, SW_LDS_BYTES, tid);
        const uint fk = 0u;          // fragment index base inside the staged part
#else
        const uint fk = k * G * KC;
#endif
        uint2 res[NT][G];
        [unroll] for (uint g = 0; g < G; ++g) {
            const uint mt = k * G + g;
            uint2 f[KC];
            [unroll] for (uint kc = 0; kc < KC; ++kc) f[kc] = sw_lfrag_b(sw_lane_base_m(0u, lane), fk + g * KC + kc);
            const v8f b = wg_load8f(be, mt * 64u);
#if SW_Q7_CHUNK
            v8f acc[NT];
            [unroll] for (int i2 = 0; i2 < NT; ++i2) acc[i2] = b;
            [unroll] for (uint kc2 = 0; kc2 < KC; kc2 += 2u) {
                [unroll] for (int i3 = 0; i3 < NT; ++i3) {
                    acc[i3] = wg_wmma_fp8(f[kc2], yf[i3][kc2], acc[i3]);
                    acc[i3] = wg_wmma_fp8(f[kc2 + 1u], yf[i3][kc2 + 1u], acc[i3]);
                    if (kc2 + 2u < KC) acc[i3] = hyb_round16(acc[i3]);
                }
            }
            [unroll] for (int i4 = 0; i4 < NT; ++i4) res[i4][g] = hyb_q8_of_chain(acc[i4]);
#else
            v8f acc[NT];
            [unroll] for (int i2 = 0; i2 < NT; ++i2) acc[i2] = b;
            [unroll] for (uint kc2 = 0; kc2 < KC; ++kc2)
                [unroll] for (int i3 = 0; i3 < NT; ++i3) acc[i3] = wg_wmma_fp8(f[kc2], yf[i3][kc2], acc[i3]);
            [unroll] for (int i4 = 0; i4 < NT; ++i4) res[i4][g] = hyb_q8_of_chain(acc[i4]);
#endif
        }
        [unroll] for (int i5 = 0; i5 < NT; ++i5) {
            if (tok[i5] < ntok) {
                const uint dst = ob[i5] + (uint)((int)(k >> 1) * (2 * Wl) + (int)(k & 1u)) * SW_EX_C;
                [unroll] for (uint q2 = 0; q2 < G / 2; ++q2)
                    wg_st128(out_p, dst + 16u * q2, uint4(res[i5][2 * q2], res[i5][2 * q2 + 1]));
            }
        }
    }
}
#endif
