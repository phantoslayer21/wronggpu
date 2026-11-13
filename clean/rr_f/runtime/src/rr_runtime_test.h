// SPDX-License-Identifier: MIT
// Test hooks of the RR runtime (not used by the proxy): exported as WrongGPU_RrRuntimeTestV1 next to the adapter.
#pragma once
#include <stdint.h>
#include <wchar.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct WgRrRuntimeTestV1 {
  uint32_t size;
  // waits (bounded) until the instance's last recorded frame has completed, copies its whole arena to `path`
  uint32_t (*dump)(void* instance, const wchar_t* path);
  // arena layout: model, CH0, CH1, PH1, PH2, EX0, EX1, fixed_end, HID, PM0, PM1, Pe, Pn, state_end, arena bytes,
  // render size (w | h << 32); with n >= 24 also AH0, AH1, TH0, TH1, OH0, OH1 (0 = not created), the last frame's
  // work-area size, flags (bit 0 work buffers aliased by lifetime)
  uint32_t (*layout)(void* instance, uint64_t* out, uint32_t n);
  // live resources made by the runtime, live instances, deferred releases, resources made in total
  void (*counters)(int64_t out[4]);
  // waits until every recorded frame has completed and releases what was deferred
  uint32_t (*drain)(uint32_t ms);
  // releases the device objects (no live instance allowed)
  uint32_t (*shutdown)(void);
  // "name offset" lines of the last recorded frame's buffers (and "tex.<role> off w h ch ox oy")
  uint32_t (*describe)(void* instance, char* buf, uint32_t size);
} WgRrRuntimeTestV1;

#ifdef __cplusplus
}
#endif
