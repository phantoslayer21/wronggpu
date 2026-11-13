// SPDX-License-Identifier: MIT
// Memory-access and arithmetic helpers of the per-pixel stages (encoder.hip, decoder.hip, exposure.hip).
//
// Addressing: every access is `base pointer (uniform, SGPRs) + 32-bit unsigned byte offset (VGPR)`, which the AMDGPU
// backend turns into global_load / global_store with an SGPR base and one VGPR offset, without the 64-bit address
// arithmetic (shift, add with carry) of `ptr[int index]`. Image sizes of the network fit easily in 32 bits.
//
// Multi-dword accesses are declared with 4-byte alignment: global memory instructions on gfx12 need only dword
// alignment (the two low address bits are ignored), so e.g. two neighbouring 32-bit texels at any texel position
// can be read with one 64-bit load.
#pragma once

#include "wg_platform.h"

typedef u32 v2u_a4 __attribute__((ext_vector_type(2), aligned(4)));
typedef u32 v4u_a4 __attribute__((ext_vector_type(4), aligned(4)));
typedef u32 v2u __attribute__((ext_vector_type(2)));
typedef float v2f_t __attribute__((ext_vector_type(2)));

// (Not templates: a typedef's alignment attribute is dropped when the type is a template argument.)
WG_FN const char* wg_at(const void* base, u32 off) { return (const char*)base + (size_t)off; }
WG_FN char* wg_at(void* base, u32 off) { return (char*)base + (size_t)off; }
WG_FN u32 wg_ld16(const void* base, u32 off) { return *(const unsigned short*)wg_at(base, off); }
WG_FN u32 wg_ld32(const void* base, u32 off) { return *(const u32*)wg_at(base, off); }
WG_FN v2u wg_ld64(const void* base, u32 off) { return *(const v2u_a4*)wg_at(base, off); }    // 4-byte aligned
WG_FN v4u wg_ld128(const void* base, u32 off) { return *(const v4u_a4*)wg_at(base, off); }   // 4-byte aligned
WG_FN v4u wg_ld128_a16(const void* base, u32 off) { return *(const v4u*)wg_at(base, off); }  // 16-byte aligned
WG_FN void wg_st32(void* base, u32 off, u32 v) { *(u32*)wg_at(base, off) = v; }
WG_FN void wg_st64(void* base, u32 off, v2u v) { *(v2u_a4*)wg_at(base, off) = v; }
WG_FN void wg_st128(void* base, u32 off, v4u v) { *(v4u_a4*)wg_at(base, off) = v; }
// Stores with a cache hint: kNT = 1 marks them non-temporal (data read again only much later, by another kernel).
template <int kNT>
WG_FN void wg_st64_hint(void* base, u32 off, v2u v) {
  if (kNT)
    __builtin_nontemporal_store((v2u_a4)v, (v2u_a4*)wg_at(base, off));
  else
    wg_st64(base, off, v);
}
template <int kNT>
WG_FN void wg_st128_hint(void* base, u32 off, v4u v) {
  if (kNT)
    __builtin_nontemporal_store((v4u_a4)v, (v4u_a4*)wg_at(base, off));
  else
    wg_st128(base, off, v);
}

#if defined(__AMDGCN__)
// med3(x, lo, hi) for lo <= hi; x must not be NaN (callers guarantee it, or only use the result where a NaN x is
// harmless).
WG_FN float wg_med3f(float x, float lo, float hi) { return __builtin_amdgcn_fmed3f(x, lo, hi); }
// float -> int32 with v_cvt_i32_f32: rounds toward zero, saturates, NaN -> 0.
WG_FN int wg_f2i(float x) { return (int)x; }
WG_FN float wg_log2(float x) { return __builtin_amdgcn_logf(x); }  // v_log_f32
WG_FN float wg_sqrt(float x) { return __builtin_amdgcn_sqrtf(x); }  // v_sqrt_f32
WG_FN u32 wg_xor16_u(u32 v) {
  return (u32)__builtin_amdgcn_ds_bpermute((int)((wg_lane_id() ^ 16u) << 2), (int)v);
}
#else
WG_FN float wg_med3f(float x, float lo, float hi) { return std::fmin(std::fmax(x, lo), hi); }
WG_FN int wg_f2i(float x) {
  if (x != x) return 0;
  if (x >= 2147483647.0f) return 2147483647;
  if (x <= -2147483648.0f) return (-2147483647 - 1);
  return (int)x;
}
WG_FN float wg_log2(float x) { return std::log2(x); }
WG_FN float wg_sqrt(float x) { return std::sqrt(x); }
WG_FN u32 wg_xor16_u(u32 v) { return __builtin_bit_cast(u32, wg_xor16(__builtin_bit_cast(float, v))); }
#endif

// Scheduling fence for values: an empty volatile asm that "uses" its operands. The values must be computed before it,
// and it is not moved across other volatile operations (such as a barrier). No instructions are emitted.
#if defined(__AMDGCN__)
WG_FN void wg_pin(u32 a, u32 b, u32 c, u32 d) { asm volatile("" ::"v"(a), "v"(b), "v"(c), "v"(d)); }
#else
WG_FN void wg_pin(u32, u32, u32, u32) {}
#endif

// Machine-scheduler barrier: no instruction is moved across it (device only).
WG_FN void wg_sched_barrier() {
#if defined(__AMDGCN__)
  __builtin_amdgcn_sched_barrier(0);
#endif
}

// Compiler-only memory fence: no memory access is moved across it (no instructions are emitted).
WG_FN void wg_fence() { asm volatile("" ::: "memory"); }

WG_FN int wg_med3i(int x, int lo, int hi) { return x < lo ? lo : (x > hi ? hi : x); }
WG_FN int wg_min_i(int a, int b) { return a < b ? a : b; }
WG_FN float wg_rn16(float x) { return (float)(_Float16)x; }
WG_FN float wg_h2f_lo(u32 v) { return (float)__builtin_bit_cast(_Float16, (unsigned short)(v & 0xffffu)); }
WG_FN float wg_h2f_hi(u32 v) { return (float)__builtin_bit_cast(_Float16, (unsigned short)(v >> 16)); }
WG_FN u32 wg_f2h_bits(float x) { return (u32)__builtin_bit_cast(unsigned short, (_Float16)x); }
WG_FN u32 wg_pack_h2(float x, float y) { return wg_f2h_bits(x) | (wg_f2h_bits(y) << 16); }

// Bilinear footprint of a w x h texture (w, h >= 2) at texel-space position (tx, ty) = (u w - 0.5, v h - 0.5), with
// clamp-to-edge addressing, read as two rows of two neighbouring texels (one 64-bit load per row).
//
// The position is clamped to [0, w - 1] x [0, h - 1] and the left texel column to w - 2 (rows likewise). This gives
// exactly the clamp-to-edge result with exact weights: left of the first texel centre, standard bilinear has both
// taps on texel 0, here tap 0 has weight 1 and tap 1 weight 0 (same value); right of the last centre, the left tap
// moves to w - 2 with weight 0 and the right tap (w - 1) gets weight 1. Inside, nothing changes.
// The texel index y0 w + x0 is computed in float (exact below 2^24 texels). A NaN position still addresses memory
// inside the texture: min(NaN, w - 2) is w - 2 (IEEE minNum), so the index is finite; its weights are NaN, and
// callers either select the value away or want NaN.
struct WgFoot {
  u32 off0, off1;  // byte offsets of the two rows' left texels
  float fx, fy;    // weights of the right column / lower row
};
WG_FN WgFoot wg_footprint(float tx, float ty, int w, int h, u32 texel_bytes) {
  const float fw = (float)w, fh = (float)h;
  const float cx = wg_med3f(tx, 0.0f, fw - 1.0f), cy = wg_med3f(ty, 0.0f, fh - 1.0f);
  const float x0 = wg_fmin(__builtin_floorf(cx), fw - 2.0f), y0 = wg_fmin(__builtin_floorf(cy), fh - 2.0f);
  WgFoot f;
  f.fx = cx - x0;
  f.fy = cy - y0;
  f.off0 = (u32)__builtin_fmaf(y0, fw, x0) * texel_bytes;
  f.off1 = f.off0 + (u32)w * texel_bytes;
  return f;
}
