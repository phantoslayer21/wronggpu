// SPDX-License-Identifier: GPL-3.0-only
// HIP twin of clean/dx12/kernels/l_hyb_probe.hlsl (same buffer layout): the hybrid helpers of clean/kernels/wg_hyb.h
// for every FP16 input, and the chain-end double rounding of nf float32 values (clean/l/runner/l_hyb_probe_check.py
// --hip). Not used by the runtime.
#include "wg_platform.h"
#include "wg_hyb.h"

WG_KERNEL_NW(l_hyb_probe, 256, 8)(const float* src, unsigned char* dst, unsigned nf) {
  const unsigned t = wg_block_id() * 256u + wg_lane_id();
  const unsigned i0 = 8u * t;
  v8f s;
  for (int k = 0; k < 8; ++k) s[k] = hyb_from16(i0 + k);
  const v2i g = hyb_gelu_q8(s);
  *(v2i*)(dst + i0) = g;
  const v8f e = hyb_softmax_e8(s);
  unsigned short* e16 = (unsigned short*)(dst + 65536);
  unsigned short* r16 = (unsigned short*)(dst + 196608);
  unsigned short* c16 = (unsigned short*)(dst + 327680);
  float* cl = (float*)(dst + 458752);
  for (int k = 0; k < 8; ++k) {
    e16[i0 + k] = (unsigned short)hyb_bits16(e[k]);
    r16[i0 + k] = (unsigned short)hyb_bits16(hyb_rsq16(s[k]));
    c16[i0 + k] = (unsigned short)hyb_bits16(hyb_rcp16(s[k]));
    cl[i0 + k] = hyb_merge_clamp(s[k]);
  }
  const unsigned ic = i0 < nf - 8u ? i0 : nf - 8u;
  v8f x;
  for (int m = 0; m < 8; ++m) x[m] = src[ic + m];
  const v2i q = hyb_q8_of_chain(x);
  if (i0 < nf) *(v2i*)(dst + 720896u + i0) = q;
}
