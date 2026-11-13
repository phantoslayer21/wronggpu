// SPDX-License-Identifier: MIT
// D3D12 ports of the patch-merge and expand kernels of clean/kernels/swin.hip as built into swin.co (WG_GEMM_THREADS 256,
// WG_NT 2: two groups of 16 tokens per wave; the weight fragments staged in LDS and shared by the 8 waves).
// HIP ISA: <build>\dx12-m\isa\patch_merge_c64.s, patch_merge_c96.s, expand_c96.s. The arithmetic is the
// WMMA chains (bias as the first accumulator, K groups in order) and the clamped FP8 pack, exactly as there.
//
// patch_merge<C, CO, NT> (defines SW_PM_C, SW_PM_CO): HIP stages all of Wm (MO * KZ fragments) in LDS at once; for C = 96
// that is 48 KB, over the 32 KB of D3D12. Here the output tiles are done in SW_PM_SPLIT passes, each with the fragments of
// its tiles in LDS (C = 96: 4 passes of 2 tiles, 12 KB each; measured 0.6-0.75 us faster than 2 passes of 24 KB, 8 passes
// of 6 KB 0.4 us; C = 64 stays at one 24 KB pass: 2 or 3 passes were 1.9 / 3.2 us slower). Every output tile is computed
// exactly as in HIP (same fragments, same chain), only the staging differs, so the bytes are the same.
// expand<CL, C, NT> (defines SW_EX_CL, SW_EX_C).
//
// @wg-kernel patch_merge_c64 co=swin.co block=256 wave=32 args=p,p,p,u,u defines=SW_PM_C=64,SW_PM_CO=96,SW_PM_SPLIT=1
// @wg-kernel patch_merge_c96 co=swin.co block=256 wave=32 args=p,p,p,u,u defines=SW_PM_C=96,SW_PM_CO=128,SW_PM_SPLIT=4
// @wg-kernel expand_c96 co=swin.co block=256 wave=32 args=p,p,p,u,u defines=SW_EX_CL=96,SW_EX_C=64
#include "wg_platform.hlsli"

WG_ARGS_BEGIN
    uint in_p, out_p, w;   // pointers: input map, output map, weights (bias first, then the fragments)
    int H, W;              // patch merge: the input map's size; expand: the low-resolution size (Hl, Wl)
WG_ARGS_END

#define NT 2
// SW_CLAMP_SKIP 1: the output packs skip the +-448 clamp when the whole wave is inside it (speed_swin.hlsli sw_in448).
#ifndef SW_CLAMP_SKIP
#define SW_CLAMP_SKIP 1
#endif
#define THREADS WG_BLOCK

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
#define SW_LDS_BYTES (4 * G * KC * 256)
#endif
#include "swin_common.hlsli"
#include "speed_swin.hlsli"

// HIP copy_to_lds: `bytes` (a multiple of THREADS x 16) from global memory to LDS byte 0, then a barrier.
void copy_to_lds(uint src, uint bytes, uint tid)
{
    [loop] for (uint i = tid; i < bytes / 16u; i += THREADS) sw_lds_st128(i * 16u, wg_ld128(src, i * 16u));
    wg_barrier_lds();
}

#if defined(SW_PM_C)
#if SW_PM_C == 64
#define SW_ENTRY patch_merge_c64
#else
#define SW_ENTRY patch_merge_c96
#endif
// merged(i, j) = q8(Wm concat(TL, TR, BL, BR) + bm); lane half h holds the 2C bytes of input row 2i + h.
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
        [unroll] for (uint q = 0; q < KZ / 2; ++q) {
            uint4 v = wg_ld128(in_p, off + 16u * q);
            zf[i][2 * q] = v.xy; zf[i][2 * q + 1] = v.zw;
        }
    }
    const uint bm = w + hh * 32u;
    [unroll] for (uint pass = 0; pass < SW_PM_SPLIT; ++pass) {
        if (pass > 0u) wg_barrier_lds();   // every wave is done with the previous pass's fragments
        copy_to_lds(w + WM + pass * SW_LDS_BYTES, SW_LDS_BYTES, tid);
        // One output tile (8 channels per lane) at a time, stored right away.
        const uint mop = MOP + sw_zero();
        [loop] for (uint mp = 0; mp < mop; ++mp) {
            const uint mt = pass * MOP + mp;
            uint2 f[KZ];
            [unroll] for (uint g = 0; g < KZ; ++g) f[g] = sw_lfrag_b(sw_lane_base_m(0u, lane), mp * KZ + g);
            const v8f b = wg_load8f(bm, mt * 64u);
            v8f acc[NT];
            [unroll] for (int i2 = 0; i2 < NT; ++i2) acc[i2] = b;
            [unroll] for (uint g2 = 0; g2 < KZ; ++g2)
                [unroll] for (int i3 = 0; i3 < NT; ++i3) acc[i3] = wg_wmma_fp8(f[g2], zf[i3][g2], acc[i3]);
            uint2 res[NT];
            // C = 64: the select form measured 0.5 us faster than sw_pack_q8_v2 (C = 96: v2 0.4 us faster)
#if SW_CLAMP_SKIP
            if (sw_in448(sw_absmax8(sw_absmax8(0.0f, acc[0]), acc[1]))) {
                [unroll] for (int i4 = 0; i4 < NT; ++i4) res[i4] = wg_pack_q8_nc_ags(acc[i4]);
            } else
#endif
            {
#if SW_PM_C == 64
                [unroll] for (int i4 = 0; i4 < NT; ++i4) res[i4] = sw_pack_q8(acc[i4]);
#else
                [unroll] for (int i4 = 0; i4 < NT; ++i4) res[i4] = sw_pack_q8_v2(acc[i4]);
#endif
            }
            [unroll] for (int i5 = 0; i5 < NT; ++i5)
                if (tok[i5] < ntok) wg_st64(out_p, (uint)tok[i5] * SW_PM_CO + hh * (SW_PM_CO / 2) + 8u * mt, res[i5]);
        }
    }
}
#else
// e = q8(We y + be) per low-resolution token, as an FP8 map e [2 Hl][2 Wl][C] (sub-pixel k at (2r + k / 2, 2c + k % 2)).
WG_KERNEL(expand_c96)
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
        [unroll] for (uint q = 0; q < KC / 2; ++q) {
            uint4 v = wg_ld128(in_p, off + 16u * q);
            yf[i][2 * q] = v.xy; yf[i][2 * q + 1] = v.zw;
        }
    }
    copy_to_lds(w + WE, SW_LDS_BYTES, tid);
    // each token's output base once (row / column by one division), the sub-pixel k part per iteration
    uint ob[NT];
    [unroll] for (int i6 = 0; i6 < NT; ++i6) {
        const int r = tok[i6] / Wl, c = tok[i6] - r * Wl;
        ob[i6] = (uint)((2 * r) * (2 * Wl) + 2 * c) * SW_EX_C + hh * (SW_EX_C / 2);
    }
    const uint be = w + hh * 32u;
    [loop] for (uint k = 0; k < 4; ++k) {
        uint2 res[NT][G];
        [unroll] for (uint g = 0; g < G; ++g) {
            const uint mt = k * G + g;
            uint2 f[KC];
            [unroll] for (uint kc = 0; kc < KC; ++kc) f[kc] = sw_lfrag_b(sw_lane_base_m(0u, lane), mt * KC + kc);
            const v8f b = wg_load8f(be, mt * 64u);
            v8f acc[NT];
            [unroll] for (int i2 = 0; i2 < NT; ++i2) acc[i2] = b;
            [unroll] for (uint kc2 = 0; kc2 < KC; ++kc2)
                [unroll] for (int i3 = 0; i3 < NT; ++i3) acc[i3] = wg_wmma_fp8(f[kc2], yf[i3][kc2], acc[i3]);
#if SW_CLAMP_SKIP
            if (sw_in448(sw_absmax8(sw_absmax8(0.0f, acc[0]), acc[1]))) {
                [unroll] for (int i4 = 0; i4 < NT; ++i4) res[i4][g] = wg_pack_q8_nc_ags(acc[i4]);
            } else
#endif
            {
                [unroll] for (int i4 = 0; i4 < NT; ++i4) res[i4][g] = sw_pack_q8_v2(acc[i4]);
            }
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
