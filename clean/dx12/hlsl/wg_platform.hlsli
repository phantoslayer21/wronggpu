// SPDX-License-Identifier: MIT
// Platform layer of the D3D12 ports: the HLSL counterparts of clean/kernels/wg_platform.h and wg_mem.h, with the same names
// and the same bits (wg_math.hlsli: arithmetic mode, rules and exceptions). Conventions: clean/dx12/README.md.
//
// A kernel file includes this header first, then declares its argument block; build_dx12.py defines WG_BLOCK (lanes
// per workgroup) and WG_WAVE (32 or 64) from the file's @wg-kernel line:
//
//     #include "wg_platform.hlsli"
//     WG_ARGS_BEGIN
//         uint color, partials;            // HIP pointers: byte offsets into the arena (0 = null)
//         int P_rw, P_rh; float P_inv_pi;  // scalars and by-value struct fields, flattened, in HIP argument order
//     WG_ARGS_END
//     WG_KERNEL(exposure_auto_diet) { WG_SETUP(); ... }
//
// Memory: one global byte buffer (the runtime's arena) with 32-bit byte offsets; loads and stores of 2 / 4 / 8 / 16 bytes
// need that alignment of the offset only to 2 / 4 bytes (as HIP's wg_ld64 / wg_ld128 with 4-byte alignment).
#pragma once

#ifndef WG_BLOCK
#error "WG_BLOCK / WG_WAVE come from the @wg-kernel line (compile with clean/dx12/build_dx12.py)"
#endif
#ifndef WG_WAVE
#define WG_WAVE 32
#endif

#include "wg_math.hlsli"

// ---- kernel interface ---------------------------------------------------------------------------------------------
// Root constants (b0): two set by the runtime, then the kernel arguments (32-bit scalars only, so that DWORD i of the
// arguments is the i-th declared scalar: no vectors, no structs, no arrays in this block).
#define WG_ARGS_BEGIN cbuffer wg_args : register(b0) { uint wg_group_base_; uint wg_grid_;
#define WG_ARGS_END };

RWByteAddressBuffer wg_mem : register(u0);                      // the arena (root UAV)
globallycoherent RWByteAddressBuffer wg_mem_gc : register(u1);  // the same memory, coherent across workgroups

static uint wg_tid_;   // lane index in the workgroup (HIP threadIdx.x)
static uint wg_bid_;   // workgroup index in the grid (HIP blockIdx.x)

#define WG_KERNEL(name) [numthreads(WG_BLOCK, 1, 1)] [WaveSize(WG_WAVE)] \
    void name(uint wg_sv_tid_ : SV_GroupThreadID, uint wg_sv_gid_ : SV_GroupID)
#define WG_SETUP() (wg_tid_ = wg_sv_tid_, wg_bid_ = wg_sv_gid_ + wg_group_base_)

uint wg_lane_id() { return wg_tid_; }           // HIP wg_lane_id(): the lane in the workgroup (0 .. WG_BLOCK - 1)
uint wg_block_id() { return wg_bid_; }          // HIP wg_block_id(): grids above 65535 run as several dispatches
#define wg_grid_size() (wg_grid_)                   // number of workgroups of the launch (after WG_ARGS_END)
uint wg_wave_lane() { return WaveGetLaneIndex(); }   // = wg_lane_id() % WG_WAVE (checked by probe/arith3)

// ---- types ----------------------------------------------------------------------------------------------------------
// HIP ext_vector types: v4f = float4, v2u = uint2, v4u = uint4; v2i (8 FP8 bytes, byte i = element i) = uint2;
// v8h (8 FP16, element i in dword i / 2, low half first) = uint4; v8f = struct v8f.
struct v8f { float v[8]; };

v8f wg_splat(float x)
{
    v8f r;
    [unroll] for (uint i = 0; i < 8; ++i) r.v[i] = x;
    return r;
}

// ---- global memory (wg_mem.h: base pointer + 32-bit byte offset) ---------------------------------------------------
uint wg_ld8(uint base, uint off) { uint a = base + off; return (wg_mem.Load(a & ~3u) >> ((a & 3u) * 8u)) & 0xffu; }
uint wg_ld16(uint base, uint off) { return (uint)wg_mem.Load<uint16_t>(base + off); }
uint wg_ld32(uint base, uint off) { return wg_mem.Load(base + off); }
uint2 wg_ld64(uint base, uint off) { return wg_mem.Load2(base + off); }
uint4 wg_ld128(uint base, uint off) { return wg_mem.Load4(base + off); }
uint4 wg_ld128_a16(uint base, uint off) { return wg_mem.Load4(base + off); }
float wg_ldf(uint base, uint off) { return asfloat(wg_mem.Load(base + off)); }
float4 wg_ld4f(uint base, uint off) { return asfloat(wg_mem.Load4(base + off)); }
void wg_st16(uint base, uint off, uint v) { wg_mem.Store<uint16_t>(base + off, (uint16_t)v); }
void wg_st32(uint base, uint off, uint v) { wg_mem.Store(base + off, v); }
void wg_st64(uint base, uint off, uint2 v) { wg_mem.Store2(base + off, v); }
void wg_st128(uint base, uint off, uint4 v) { wg_mem.Store4(base + off, v); }
void wg_stf(uint base, uint off, float v) { wg_mem.Store(base + off, asuint(v)); }
// non-temporal stores (HIP wg_st64_hint / wg_st128_hint): plain stores here (same bytes)
void wg_st64_hint(uint base, uint off, uint2 v) { wg_st64(base, off, v); }
void wg_st128_hint(uint base, uint off, uint4 v) { wg_st128(base, off, v); }
v8f wg_load8f(uint base, uint off)
{
    float4 a = wg_ld4f(base, off), b = wg_ld4f(base, off + 16u);
    v8f r;
    r.v[0] = a.x; r.v[1] = a.y; r.v[2] = a.z; r.v[3] = a.w; r.v[4] = b.x; r.v[5] = b.y; r.v[6] = b.z; r.v[7] = b.w;
    return r;
}
// Cross-workgroup communication (HIP: __hip_atomic_* at agent scope, fences "agent"): atomics and the loads / stores of
// the data they publish go through wg_mem_gc; wg_fence_device() orders them (release before the atomic, acquire after).
uint wg_atomic_add(uint base, uint off, uint v) { uint o; wg_mem_gc.InterlockedAdd(base + off, v, o); return o; }
uint wg_atomic_max(uint base, uint off, uint v) { uint o; wg_mem_gc.InterlockedMax(base + off, v, o); return o; }
uint wg_atomic_exch(uint base, uint off, uint v) { uint o; wg_mem_gc.InterlockedExchange(base + off, v, o); return o; }
uint wg_atomic_cas(uint base, uint off, uint cmp, uint v) { uint o; wg_mem_gc.InterlockedCompareExchange(base + off, cmp, v, o); return o; }
uint wg_ld32_gc(uint base, uint off) { return wg_mem_gc.Load(base + off); }
void wg_st32_gc(uint base, uint off, uint v) { wg_mem_gc.Store(base + off, v); }
void wg_fence_device() { DeviceMemoryBarrier(); }

// ---- workgroup and wave --------------------------------------------------------------------------------------------
void wg_barrier() { AllMemoryBarrierWithGroupSync(); }        // LDS and global visibility in the workgroup
void wg_barrier_lds() { GroupMemoryBarrierWithGroupSync(); }  // LDS only (HIP wg_barrier_lds)
// Value of v in lane (lane ^ d) of the wave (HIP ds_bpermute with (lane ^ d) << 2).
float wg_xor(float v, uint d) { return WaveReadLaneAt(v, WaveGetLaneIndex() ^ d); }
uint wg_xor_u(uint v, uint d) { return WaveReadLaneAt(v, WaveGetLaneIndex() ^ d); }
float wg_xor16(float v) { return wg_xor(v, 16u); }
uint wg_xor16_u(uint v) { return wg_xor_u(v, 16u); }
int wg_uniform(int v) { return WaveReadLaneFirst(v); }       // v_readfirstlane_b32

// ---- arithmetic (HIP builtins; wg_math.hlsli for the exact helpers) -------------------------------------------------
float wg_exp2(float x) { return exp2(x); }          // v_exp_f32
float wg_log2(float x) { return log2(x); }          // v_log_f32
float wg_rcp(float x) { return rcp(x); }            // v_rcp_f32
float wg_rsq(float x) { return rsqrt(x); }          // v_rsq_f32
float wg_sqrt(float x) { return sqrt(x); }          // v_sqrt_f32
float wg_rint(float x) { return round(x); }         // v_rndne_f32
float wg_floor(float x) { return floor(x); }        // v_floor_f32
float wg_fmax(float a, float b) { return max(a, b); }   // v_max_num_f32
float wg_fmin(float a, float b) { return min(a, b); }   // v_min_num_f32
// med3(x, lo, hi) for lo <= hi and x not NaN (v_med3_num_f32)
float wg_med3f(float x, float lo, float hi) { return min(max(x, lo), hi); }
int wg_f2i(float x) { return (int)x; }              // v_cvt_i32_f32: truncate, saturate, NaN -> 0
uint wg_f2u(float x) { return (uint)x; }            // v_cvt_u32_f32
int wg_med3i(int x, int lo, int hi) { return x < lo ? lo : (x > hi ? hi : x); }
int wg_min_i(int a, int b) { return a < b ? a : b; }

// ---- FP16 -----------------------------------------------------------------------------------------------------------
float wg_rn16(float x) { return wg_h2f(wg_f2h(x)); }               // (float)(_Float16) x
float wg_h2f_lo(uint v) { return f16tof32(v & 0xffffu); }
float wg_h2f_hi(uint v) { return f16tof32(v >> 16); }
uint wg_f2h_bits(float x) { return wg_f2h(x); }                    // round to nearest even
uint wg_pack_h2(float x, float y) { return wg_f2h(x) | (wg_f2h(y) << 16); }
// v_cvt_pkrtz_f16_f32 (HIP wg_h2): round toward zero, as packed bits and as half2
uint wg_h2(float a, float b) { return f32tof16(a) | (f32tof16(b) << 16); }
half2 wg_h2_h(float a, float b) { return asfloat16(uint16_t2((uint16_t)f32tof16(a), (uint16_t)f32tof16(b))); }
half2 wg_as_h2(uint v) { return asfloat16(uint16_t2((uint16_t)(v & 0xffffu), (uint16_t)(v >> 16))); }
uint wg_h2_bits(half2 h) { uint16_t2 u = asuint16(h); return (uint)u.x | ((uint)u.y << 16); }
float wg_dot2_acc(half2 a, half2 b, float c) { return dot2add(a, b, c); }   // v_dot2_f32_f16: c + a0 b0 + a1 b1
// v_dot2_f32_f16 with both slots on one half: 2 a[HI] b[HI] (exact)
float wg_dot_sel0(half2 a, half2 b) { return dot2add(a.xx, b.xx, 0.0f); }
float wg_dot_sel1(half2 a, half2 b) { return dot2add(a.yy, b.yy, 0.0f); }
half2 wg_exp2_h2(half2 x) { return exp2(x); }                      // two v_exp_f16

// 8 x (_Float16) d[i] (round to nearest even) as v8h. Software; wg_to_f16_ags is the driver's conversion (same bits).
uint4 wg_to_f16(v8f d)
{
    return uint4(wg_pack_h2(d.v[0], d.v[1]), wg_pack_h2(d.v[2], d.v[3]), wg_pack_h2(d.v[4], d.v[5]),
                 wg_pack_h2(d.v[6], d.v[7]));
}

// ---- FP8 E4M3FN -----------------------------------------------------------------------------------------------------
// The clamp of wg_pack_q8: IEEE maximum / minimum (a NaN propagates; v_maximumminimum_f32) and the clamp without NaN
// pass-through of wg_pack_q8_inner (a NaN becomes -448 as v_max_num / v_min_num do).
float wg_sat448(float x) { return isnan(x) ? asfloat(asuint(x) | 0x400000u) : min(max(x, -448.0f), 448.0f); }
float wg_sat448_inner(float x) { return min(max(x, -448.0f), 448.0f); }
// One value -> FP8 byte exactly as v_cvt_pk_fp8_f32 (round to nearest even, no saturation: |x| > 464 -> 0x7f | sign,
// NaN -> 0xff; wg_math.hlsli wg_e4m3_hw).
uint wg_fp8(float x) { return wg_e4m3_hw(x); }
// Software wg_f32_to_e4m3 / wg_e4m3_to_f32 of wg_platform.h (saturating encode; NaN -> 0x7f | sign).
uint wg_f32_to_e4m3(float f)
{
    uint b = asuint(f), s = (b >> 24) & 0x80u, a = b & 0x7fffffffu;
    uint sub = (uint)round(min(asfloat(a), 0.015625f) * 512.0f);
    uint nrm = ((a + 0x7ffffu + ((a >> 20) & 1u)) >> 20) - (120u << 3);
    uint code = a < 0x3c800000u ? sub : nrm;
    code = a >= 0x43e00000u ? 0x7eu : code;
    code = a > 0x7f800000u ? 0x7fu : code;
    return s | code;
}
// FP8 byte -> f32 as v_cvt_f32_fp8 (exact; both NaN codes give 0xffc00000 like the hardware).
float wg_e4m3_to_f32(uint c)
{
    uint em = c & 0x7fu;
    float nrm = asfloat((em + (120u << 3)) << 20), sub = (float)em * (1.0f / 512.0f);
    float v = em >= 8u ? nrm : sub;
    uint r = ((c & 0x80u) << 24) | asuint(v);
    return asfloat(em == 0x7fu ? 0xffc00000u : r);
}
uint wg_pack4_fp8(float a, float b, float c, float d) { return wg_fp8(a) | (wg_fp8(b) << 8) | (wg_fp8(c) << 16) | (wg_fp8(d) << 24); }
// HIP wg_pack_q8 / _nc / _inner (8 values -> 8 bytes, element i -> byte i); software, any control flow.
uint2 wg_pack_q8(v8f d)
{
    [unroll] for (uint i = 0; i < 8; ++i) d.v[i] = wg_sat448(d.v[i]);
    return uint2(wg_pack4_fp8(d.v[0], d.v[1], d.v[2], d.v[3]), wg_pack4_fp8(d.v[4], d.v[5], d.v[6], d.v[7]));
}
uint2 wg_pack_q8_nc(v8f d)
{
    return uint2(wg_pack4_fp8(d.v[0], d.v[1], d.v[2], d.v[3]), wg_pack4_fp8(d.v[4], d.v[5], d.v[6], d.v[7]));
}
uint2 wg_pack_q8_inner(v8f d)
{
    [unroll] for (uint i = 0; i < 8; ++i) d.v[i] = wg_sat448_inner(d.v[i]);
    return wg_pack_q8_nc(d);
}
v8f wg_unpack_fp8(uint2 p)
{
    v8f r;
    [unroll] for (uint i = 0; i < 4; ++i) {
        r.v[i] = wg_e4m3_to_f32((p.x >> (8u * i)) & 0xffu);
        r.v[4 + i] = wg_e4m3_to_f32((p.y >> (8u * i)) & 0xffu);
    }
    return r;
}

// ---- matrix instructions (driver intrinsic channel, wg_ags.hlsli; wave-uniform control flow only) ------------------
#if WG_WAVE == 32
#include "wg_ags.hlsli"
wg_frag wg_frag_v8f(v8f c) { wg_frag f; [unroll] for (uint i = 0; i < 8; ++i) f.v[i] = asuint(c.v[i]); return f; }
v8f wg_v8f_frag(wg_frag f) { v8f c; [unroll] for (uint i = 0; i < 8; ++i) c.v[i] = asfloat(f.v[i]); return c; }
wg_frag wg_frag2(uint2 a) { wg_frag f; f.v[0] = a.x; f.v[1] = a.y; [unroll] for (uint i = 2; i < 8; ++i) f.v[i] = 0u; return f; }
wg_frag wg_frag4(uint4 a)
{
    wg_frag f;
    f.v[0] = a.x; f.v[1] = a.y; f.v[2] = a.z; f.v[3] = a.w;
    [unroll] for (uint i = 4; i < 8; ++i) f.v[i] = 0u;
    return f;
}
// D = A x B + C, 16x16x16, FP8 E4M3 A / B, FP32 accumulate (v_wmma_f32_16x16x16_fp8_fp8, HIP wg_wmma_fp8).
v8f wg_wmma_fp8(uint2 a, uint2 b, v8f c) { return wg_v8f_frag(wg_ags_mma(WG_AGS_FP8_F32, wg_frag2(a), wg_frag2(b), wg_frag_v8f(c))); }
// D = A x B + C, 16x16x16, FP16 A / B, FP32 accumulate (v_wmma_f32_16x16x16_f16, HIP wg_wmma_f16).
v8f wg_wmma_f16(uint4 a, uint4 b, v8f c) { return wg_v8f_frag(wg_ags_mma(WG_AGS_F16_F32, wg_frag4(a), wg_frag4(b), wg_frag_v8f(c))); }
// wg_pack_q8* with the driver's conversion (v_cvt_pk_fp8_f32; the same bytes as the software ones above).
uint2 wg_pack_q8_nc_ags(v8f d)
{
    wg_frag r = wg_ags_convert(wg_frag_v8f(d), WG_AGS_T_F32_ACC, WG_AGS_T_FP8_B);
    return uint2(r.v[0], r.v[1]);
}
uint2 wg_pack_q8_ags(v8f d)
{
    [unroll] for (uint i = 0; i < 8; ++i) d.v[i] = wg_sat448(d.v[i]);
    return wg_pack_q8_nc_ags(d);
}
uint2 wg_pack_q8_inner_ags(v8f d)
{
    [unroll] for (uint i = 0; i < 8; ++i) d.v[i] = wg_sat448_inner(d.v[i]);
    return wg_pack_q8_nc_ags(d);
}
// wg_to_f16 with the driver's conversion (v_cvt_f16_f32, round to nearest even)
uint4 wg_to_f16_ags(v8f d)
{
    wg_frag r = wg_ags_convert(wg_frag_v8f(d), WG_AGS_T_F32_ACC, WG_AGS_T_F16_ACC);
    return uint4(r.v[0], r.v[1], r.v[2], r.v[3]);
}
#endif
