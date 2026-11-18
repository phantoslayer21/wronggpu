// SPDX-License-Identifier: MIT
// Developer builds (WG_DEV_DIAGNOSTICS): the 8-byte alignment premise of the D3D12 post_tf ports. WrongGPU's own code.
// dx12/kernels/post.hlsl reads the history (hist_prev) and the A8 features with 8-byte loads whose address has its low 3
// bits cleared (ld64_a8 / ld128_a8), which is the requested address only when both pointers are 8-byte aligned. The
// runtime passes its own allocations (256-byte aligned); an interior pointer such as A + 4 or hist[cur] + 4 would make the
// kernel read 4 bytes earlier, silently. runtime.cpp checks both pointers before every post_tf launch (the ordinary and
// the motion-at-output form) and fails naming the pointer. Tested by clean/runtime/tests/test_post_tf_align.cpp.
#pragma once

#include <cstdint>

// The name of the first non-null pointer that is not a multiple of 8 ("A8": the A8 features, "hist_prev": the history the
// kernel samples), or nullptr when both are.
inline const char* post_tf_misaligned(uint64_t a8, uint64_t hist_prev) {
  const struct { uint64_t v; const char* name; } f[] = {{a8, "A8"}, {hist_prev, "hist_prev"}};
  for (const auto& e : f)
    if (e.v & 7u) return e.name;
  return nullptr;
}
