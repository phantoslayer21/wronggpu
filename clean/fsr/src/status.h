// SPDX-License-Identifier: MIT
#pragma once
// What the front end did with one upscale dispatch, reported to the platform (Platform::status) for the in-game overlay
// (overlay.h). Plain C++, no Windows headers.
#include <cstdint>
#include <functional>
#include <string>

#include "model_k.h"

namespace wgfsr {

enum class UpscaleState : int {
    Amd = 0,      // AMD's FSR upscaled the frame: upscaler = amd, or WrongGPU's upscaler is unavailable / failed (reason)
    Network = 1,  // WrongGPU's upscaler (the network runtime, or the stand-in when that is the configured upscaler)
    StandIn = 2,  // the one-frame stand-in, because the upscaler was not ready for this frame
    Compare = 3,  // AMD's FSR upscaled the frame because the player switched to it for a comparison (compare_key)
    Failed = 4,   // no upscaler successfully produced this frame; reason records the failure
};

struct UpscaleStatus {
    UpscaleState state = UpscaleState::Amd;
    std::string upscaler;  // the upscaler's name ("WrongGPU network", "WrongGPU stand-in upscaler"; "AMD FSR 3.1.4" for
                           // Compare); empty for AMD
    std::string reason;    // state Amd: why WrongGPU is not upscaling
    uint32_t renderWidth = 0, renderHeight = 0, outputWidth = 0, outputHeight = 0;
    void* device = nullptr;  // the game's ID3D12Device when known (the overlay draws with it)
};

// Platform::gpuTimer: called with an upscale dispatch's command list before (end = false) and after (end = true) the
// dispatch is recorded; the platform measures the GPU time in between (the overlay's "upscaler time").
using GpuTimerFn = std::function<void(void* commandList, bool end)>;

// Platform::compareAmd: true while the player has switched to AMD's FSR for a comparison (the overlay's compare_key).
using CompareFn = std::function<bool()>;

// Brackets one dispatch with gpuTimer calls (none when the timer or the list is missing).
class GpuTimed {
public:
    GpuTimed(const GpuTimerFn& timer, void* commandList) : timer_(timer), list_(timer && commandList ? commandList : nullptr) {
        if (list_) timer_(list_, false);
    }
    ~GpuTimed() {
        if (list_) timer_(list_, true);
    }
    GpuTimed(const GpuTimed&) = delete;
    GpuTimed& operator=(const GpuTimed&) = delete;

private:
    const GpuTimerFn& timer_;
    void* list_;
};

}  // namespace wgfsr
