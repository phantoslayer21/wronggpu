// SPDX-License-Identifier: MIT
// Original numerical audit of the ACTUAL production ki_host helper.
// Compile this separate test object from clean/k/kernels so the include resolves
// to the production source under test; do not add this entry to runtime routing.
#include "../kernels/k_io.hip"
#if defined(__AMDGCN__)
static_assert(sizeof(KiHost) == 48, "twelve uniform FP32 fields");
WG_KERNEL_N(ki_host_audit, 128)(const KFrameIn* frames, KiHost* results, unsigned* markers, unsigned n) {
  unsigned id = wg_block_id() * 128u + wg_lane_id();
  if (id >= n) return;
  results[id] = ki_host(frames[id]);
  markers[id] = 0x4b49484fu;
}
#endif
