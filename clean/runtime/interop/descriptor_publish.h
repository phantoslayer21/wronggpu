// SPDX-License-Identifier: MIT
// Publishing descriptors that the CPU wrote into a shader-visible heap during a frame (clean/release-060 crash-cp2077).
//
// A shader-visible heap is write-combined memory. Descriptors written while recording a game's command list are read
// by the GPU once the game submits that list, often from another thread on another core. A write-combined store can
// still sit in the writing core's buffers when the submit happens; the game's thread hand-off does not necessarily
// drain them, and the GPU then reads a part-written descriptor. Cyberpunk 2077 RR hung every few minutes with a page
// fault at a random 48-bit address in the first kernel that read the game's textures (DRED, GPU breadcrumbs); with
// this fence and read back it did not (measurements in CRASH-CP2077.md; the write-combining explanation fits them but
// is not proven). Every frame-time descriptor write into a shader-visible heap therefore ends with
// wg::publish_descriptors: a store fence on the writing thread, then a read of the whole written range.
#pragma once
#include <intrin.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

namespace wg {

// Fence, then read the n bytes at p (descriptor ranges are multiples of 8 bytes): into `shadow` when given, else
// through volatile loads the compiler cannot drop. False when the range could not be read; the fence was still issued.
inline bool publish_descriptors(const void* p, size_t n, void* shadow = nullptr) {
  _mm_sfence();
  __try {
    if (shadow) {
      memcpy(shadow, p, n);
    } else {
      const volatile uint64_t* q = static_cast<const volatile uint64_t*>(p);
      uint64_t x = 0;
      for (size_t i = 0; i < n / 8; ++i) x ^= q[i];
      static volatile uint64_t sink;
      sink = x;
    }
  } __except (1) {
    return false;
  }
  return true;
}

}  // namespace wg
