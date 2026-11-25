// SPDX-License-Identifier: MIT
// D3D12 port of exposure_auto_diet (clean/kernels/exposure.hip; HIP build: exposure.co, ISA isa/exposure_auto_diet.s).
// Auto-exposure in one launch: every workgroup sums the log luminance of 4 rows of the sample grid, the workgroup that
// finishes last (a counter at `sync`, zero before and after the launch) sums the partials and writes the exposure pair.
// Same sums in the same order and the same roundings as the HIP kernel (fused multiply-adds where its ISA has them).
// Conditions use select() / and() (HLSL 2021 '?:' and '&&' short-circuit: the compiler then sank each load into its own
// branch, one load in flight per lane); as in HIP's ISA, a lane issues its 16 loads before the first use (RGA: 58 VGPRs,
// no scratch, 48 B LDS). Checked bit for bit incl. every FP32 partial sum, 1..4 column passes, NaN / inf / negative /
// overflowing colours and the e = 0 / 65504 corners (build/dx12-m/work/exposure_stress); only FP32 denormal colours
// differ (flushed on D3D12; colours widened from FP16 never are).
// @wg-kernel exposure_auto_diet co=exposure.co block=256 wave=32 args=p,p,p,p,p,b36,u
// @wg-kernel exposure_auto_diet_direct co=exposure.co entry=exposure_auto_diet block=256 wave=32 args=p,p,p,p,p,b36,u rs=direct defines=WG_DIRECT_TEXTURES=1
// @wg-kernel f16lut_build_direct co=- block=256 wave=32 args=p,p rs=direct defines=WG_DIRECT_TEXTURES=1,F16LUT_BUILD=1
#include "wg_platform.hlsli"
#if defined(WG_DIRECT_TEXTURES)
#include "native_direct.hlsli"
#endif

#if defined(F16LUT_BUILD)
// The direct path's per-frame 64K tables (hlsl/wg_f16lut_in.hlsli: the encoder's colour table; hlsl/wg_f16lut.hlsli: the
// post output table), launched after the exposure step (ex: [e, e_prev, x, ratio, dec_scale]) and before the encoder:
// lane i of the 65536-lane grid writes every table entry of FP16 pattern i (tab: the tables, 0 = none). (Finishing the
// auto exposure here instead, every workgroup summing the partials itself, measured +2 us: not kept.)
#include "wg_f16lut.hlsli"
#include "wg_f16lut_in.hlsli"
WG_ARGS_BEGIN
    uint tab, ex;
WG_ARGS_END
WG_KERNEL(f16lut_build_direct)
{
    WG_SETUP();
    const uint i = wg_block_id() * WG_BLOCK + wg_lane_id();
    if (and(tab != 0u, i < 65536u))
        f16lut_build(tab, i, wg_ldf(ex, 0u), wg_ldf(ex, 16u), nd_color_pi, nd_output_rne16, nd_rw >> 31u);
}
#else

WG_ARGS_BEGIN
    uint color, partials, sync_p, computed, ex;   // pointers
    int P_rw, P_rh, P_nx, P_ny, P_nparts, P_pad0;  // WgExpParams (36 bytes)
    float P_inv_pi, P_K, P_inv_n;
    int first;
WG_ARGS_END

#define kRowsPerGroup 4
#define kColsPerPass 4
#define kWaves (WG_BLOCK / 32)

groupshared float red[kWaves];
groupshared uint is_last;

// Sum over the 32 lanes of a wave (result in every lane), by lane exchanges (HIP: ds_bpermute, v += other).
float wave_sum(float v)
{
    [unroll] for (uint d = 16u; d >= 1u; d >>= 1) v = v + wg_xor(v, d);
    return v;
}

// Sum over the workgroup (in every lane). Every lane must call it.
float block_sum(float v, uint tid)
{
    v = wave_sum(v);
    if ((tid & 31u) == 0u) red[tid >> 5] = v;
    wg_barrier_lds();
    float s = 0.0f;
    [unroll] for (uint i = 0; i < kWaves; ++i) s = s + red[i];
    return s;
}

// z = ln(L + 1e-6) (0 for non-finite or non-positive L), L = (0.25 c0 + 0.5 c1 + 0.25 c2) / pi. HIP's ISA:
// t = 0.5 c1; t = fma(0.25, c0, t); t = fma(0.25, c2, t); log2(fma(inv_pi, t, 1e-6)) * ln2; L = inv_pi * t.
float sample_z(float4 c)
{
    float t = 0.5f * c.y;
    t = mad(0.25f, c.x, t);
    t = mad(0.25f, c.z, t);
    float L = P_inv_pi * t;
    float z = wg_log2(mad(P_inv_pi, t, 1e-6f)) * 0.69314718055994531f;
    return select(and(L > 0.0f, L <= 3.40282347e38f), z, 0.0f);   // no short-circuit: no branch
}

float partial_sum(uint tid)
{
    int j0 = (int)wg_block_id() * kRowsPerGroup;
    uint row[kRowsPerGroup];
    bool rok[kRowsPerGroup];
#if defined(WG_DIRECT_TEXTURES)
    int Yr[kRowsPerGroup];   // the row itself: the direct texture loads take coordinates
#endif
    [unroll] for (int jj = 0; jj < kRowsPerGroup; ++jj) {
        int j = j0 + jj;
        rok[jj] = j < P_ny;
        int Y = (2 * j * (P_rh - 1) + (P_ny - 1)) / (2 * (P_ny - 1));
        row[jj] = (uint)select(rok[jj], Y, 0) * (uint)P_rw;
#if defined(WG_DIRECT_TEXTURES)
        Yr[jj] = select(rok[jj], Y, 0);
#endif
    }
    float acc = 0.0f;
    for (int base = 0; base < P_nx; base += kColsPerPass * WG_BLOCK) {
        float4 c[kColsPerPass][kRowsPerGroup];
        bool ok[kColsPerPass];
        [unroll] for (int k = 0; k < kColsPerPass; ++k) {
            int i = base + k * WG_BLOCK + (int)tid;
            ok[k] = i < P_nx;
            int X = select(ok[k], (2 * i * (P_rw - 1) + (P_nx - 1)) / (2 * (P_nx - 1)), 0);
            [unroll] for (int jj = 0; jj < kRowsPerGroup; ++jj) {
#if defined(WG_DIRECT_TEXTURES)
                c[k][jj] = nd_raw_color_xy(X, Yr[jj]);   // 0 <= X < rw, 0 <= Y < rh: the texel row Y rw + X names
#else
                c[k][jj] = wg_ld4f(color, (row[jj] + (uint)X) * 16u);
#endif
            }
        }
        [unroll] for (int k2 = 0; k2 < kColsPerPass; ++k2)
            [unroll] for (int jj = 0; jj < kRowsPerGroup; ++jj)
                acc = acc + select(and(ok[k2], rok[jj]), sample_z(c[k2][jj]), 0.0f);
    }
    return block_sum(acc, tid);
}

// e = min(RN16(K exp(-m)), 65504), m = sum / N; exp(-m) = 2^n 2^(r log2 e), -m = n ln2 + r (two-part ln2). HIP's ISA rounds
// K * 2^n 2^(...) once to FP16 (v_fma_mixlo_f16 K, x, 0).
float exposure_value(float sum)
{
    float x = -(sum * P_inv_n);
    float n = wg_rint(x * 1.44269504088896341f);
    float r = mad(-n, 0.693145751953125f, x);
    r = mad(-n, 1.42860682030941723e-06f, r);
    float l = wg_ldexp(wg_exp2(r * 1.44269504088896341f), wg_f2i(n));
    return wg_fmin(wg_h2f(wg_mul_f16(P_K, l)), 65504.0f);
}

WG_KERNEL(exposure_auto_diet)
{
    WG_SETUP();
    uint tid = wg_lane_id();
    float s = partial_sum(tid);
    if (tid == 0u) {
        wg_mem_gc.Store(partials + wg_block_id() * 4u, asuint(s));
        wg_fence_device();
        is_last = wg_atomic_add(sync_p, 0u, 1u) == (uint)(P_nparts - 1) ? 1u : 0u;
    }
    wg_barrier();
    if (is_last == 0u) return;
    wg_fence_device();
    float acc = 0.0f;
    for (int i = (int)tid; i < P_nparts; i += WG_BLOCK) acc = acc + asfloat(wg_mem_gc.Load(partials + (uint)i * 4u));
    float t = block_sum(acc, tid);
    if (tid != 0u) return;
    float e = exposure_value(t);
    wg_stf(computed, 0u, e);
    float e_prev = first != 0 ? e : wg_ldf(ex, 8u);
    float e1 = e == 0.0f ? 1.0f : e, ep1 = e_prev == 0.0f ? 1.0f : e_prev;
    wg_st128(ex, 0u, uint4(asuint(e), asuint(e_prev), asuint(e), asuint(wg_div(e1, ep1))));
    wg_stf(ex, 16u, (float)(0.01 / (double)e1));
    wg_st32_gc(sync_p, 0u, 0u);   // for the next launch (queue-ordered after this dispatch)
}
#endif   // F16LUT_BUILD
