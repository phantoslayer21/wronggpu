// SPDX-License-Identifier: MIT
// Probe kernels for runner/selftest.py: check on the GPU the hardware facts the Swin kernels rely on, using
// synthetic data only (WMMA operand/result register layouts, the FP8 encoding of the WMMA inputs, hardware FP8
// conversions versus the software ones in wg_platform.h, and the FP8 pack/unpack functions the kernels use).
#include "wg_platform.h"

// One WMMA per wave on raw per-lane registers: d[i] = WMMA(a[i], b[i], 0), i = global lane index.
WG_KERNEL(probe_wmma_fp8)(const v2i* a, const v2i* b, v8f* d) {
  const u32 i = wg_block_id() * WG_BLOCK + wg_lane_id();
  v8f z = {};
  d[i] = wg_wmma_fp8(a[i], b[i], z);
}

WG_KERNEL(probe_wmma_f16)(const v8h* a, const v8h* b, v8f* d) {
  const u32 i = wg_block_id() * WG_BLOCK + wg_lane_id();
  v8f z = {};
  d[i] = wg_wmma_f16(a[i], b[i], z);
}

// Software FP8 conversions of wg_platform.h: enc[i] = code of x[i]; dec[c] = value of code c (c < 256).
WG_KERNEL(probe_fp8_sw)(const float* x, u32* enc, float* dec, int n) {
  const u32 i = wg_block_id() * WG_BLOCK + wg_lane_id();
  if ((int)i < n) enc[i] = wg_f32_to_e4m3(x[i]);
  if (i < 256u) dec[i] = wg_e4m3_to_f32(i);
}

// Hardware conversions (v_cvt_pk_fp8_f32, v_cvt_f32_fp8): enc[i] = raw hardware code of x[i];
// enc_sat[i] = hardware code of x[i] clamped to [-448, 448] first; dec[c] = hardware value of code c.
WG_KERNEL(probe_fp8_hw)(const float* x, u32* enc, u32* enc_sat, float* dec, int n) {
  const u32 i = wg_block_id() * WG_BLOCK + wg_lane_id();
#if defined(__AMDGCN__)
  if ((int)i < n) {
    const float v = x[i];
    enc[i] = (u32)__builtin_amdgcn_cvt_pk_fp8_f32(v, 0.0f, 0, false) & 0xffu;
    const float c = __builtin_fminf(__builtin_fmaxf(v, -448.0f), 448.0f);
    enc_sat[i] = (u32)__builtin_amdgcn_cvt_pk_fp8_f32(v != v ? v : c, 0.0f, 0, false) & 0xffu;
  }
  if (i < 256u) dec[i] = __builtin_amdgcn_cvt_f32_fp8((int)i, 0);
#else
  if ((int)i < n) enc[i] = enc_sat[i] = 0xffffffffu;  // not available in the emulator
  if (i < 256u) dec[i] = 0.0f;
#endif
}

// The packing functions the Swin kernels use (hardware or software, see wg_platform.h), on every byte position:
// packed[i] = wg_pack_q8(x[i]); unpacked[i] = wg_unpack_fp8(codes[i]); packed_inner[i] = wg_pack_q8_inner(x[i]);
// i = global lane index < n. (Argument order: inputs, then outputs, as selftest.py's call() passes them.)
WG_KERNEL(probe_fp8_pack)(const v8f* x, const v2i* codes, v2i* packed, v8f* unpacked, v2i* packed_inner, int n) {
  const u32 i = wg_block_id() * WG_BLOCK + wg_lane_id();
  if ((int)i < n) {
    packed[i] = wg_pack_q8(x[i]);
    unpacked[i] = wg_unpack_fp8(codes[i]);
    packed_inner[i] = wg_pack_q8_inner(x[i]);
  }
}

// The GELU the kernels use, on n floats: y[i] = wg_gelu(x[i]).
WG_KERNEL(probe_gelu)(const float* x, float* y, int n) {
  const u32 i = wg_block_id() * WG_BLOCK + wg_lane_id();
  if ((int)i < n) y[i] = wg_gelu(x[i]);
}

#if defined(WG_EMU)
WG_EMU_REGISTER(probe_wmma_fp8)
WG_EMU_REGISTER(probe_wmma_f16)
WG_EMU_REGISTER(probe_fp8_sw)
WG_EMU_REGISTER(probe_fp8_hw)
WG_EMU_REGISTER(probe_fp8_pack)
WG_EMU_REGISTER(probe_gelu)
#endif
