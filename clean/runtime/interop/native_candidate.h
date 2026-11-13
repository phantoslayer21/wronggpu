// SPDX-License-Identifier: MIT
// Private, explicit-queue Model M integration candidate. Original WrongGPU implementation.
// This is not the installed frontend backend and accepts no caller command list.
#pragma once
#include "convert.h"
#include "../wg_runtime.h"
#include <memory>
#include <vector>

namespace wgi {
struct NativeFrameInventory {
  bool direct_resources = false;
  uint32_t input_conversion_passes = 0, output_conversion_passes = 0;
  uint64_t removed_io_bytes = 0, runtime_bytes = 0;
};
class NativeCandidate {
 public:
  // Supplies the actual device and a DIRECT or COMPUTE queue from that device.
  // Initially one serialized frame. No hidden queue and no HIP fallback.
  static std::unique_ptr<NativeCandidate> create(ID3D12Device*, ID3D12CommandQueue*,
      const WgRtConfig&, uint32_t create_flags, std::string* error, bool direct_resources = true);
  ~NativeCandidate();
  // Caller has submitted all texture producers and arranged an independent
  // ready/value signal (it may still be pending on another producer queue).
  // Textures stay caller-owned; candidate AddRefs them until final consumption.
  // Truthful initial states are mandatory and are restored after final output use.
  // command_list must be null. Returns an AddRef completion fence whose value
  // covers all native neural work, final output and restores. The optional staged
  // C++ mode is a same-D3D12 differential reference; the private C ABI uses direct resources.
  // After a recording/execution failure, synchronize or destroy before reusing caller
  // textures. Cleanup restores states when possible; failure quarantines dependencies.
  // A failed frame poisons the candidate and must not be resumed.
  bool dispatch(const WgFrame&, ID3D12Fence* ready, uint64_t ready_value,
                ID3D12Fence** completion, uint64_t* completion_value, std::string* error);
  bool sync(std::string* error);
  // Private qualification diagnostics: exact named runtime bytes after sync.
  bool read_buffer(const char* name, std::vector<unsigned char>* bytes, std::string* error);
  const char* provider() const;
  const NativeFrameInventory& last_frame_inventory() const;
 private:
  struct Impl;
  explicit NativeCandidate(std::unique_ptr<Impl>);
  std::unique_ptr<Impl> p_;
};
}

// Opt-in private DLL ABI. These exports do not alter wgGetUpscalerBackend or its
// selected executor. All functions return 0 on success except error().
extern "C" {
WG_RT_API int wg_native_candidate_create(void* device, void* queue, const WgRtConfig* config,
    uint32_t create_flags, void** candidate, char* error, int error_capacity);
WG_RT_API int wg_native_candidate_dispatch(void* candidate, const WgFrame* frame, void* input_ready,
    uint64_t input_ready_value, void** completion_fence, uint64_t* completion_value);
WG_RT_API int wg_native_candidate_sync(void* candidate);
WG_RT_API const char* wg_native_candidate_error(void* candidate);
// Consumes candidate even on failure. A failure means completion was unknown;
// the implementation quarantines dependent objects and emits a diagnostic.
WG_RT_API int wg_native_candidate_destroy(void* candidate);
}
