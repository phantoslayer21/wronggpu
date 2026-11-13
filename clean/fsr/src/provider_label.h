// SPDX-License-Identifier: MIT
#pragma once
#include "status.h"
#include <cstring>
#include <mutex>
#include <string>

namespace wgfsr {
// Optional CPU-only label bridge for a host's FPS UI. Never performs a GPU operation.
class ProviderLabel {
public:
    void Update(const UpscaleStatus& s) {
        std::string label;
        if (s.state == UpscaleState::Network) {
#if WRONGGPU_MODEL_K
            if (s.upscaler.find("Model K") != std::string::npos) label = "WrongGPU Model K";
            else
#endif
            if (s.upscaler.find("Model M") != std::string::npos) label = "WrongGPU Model M";
            else label = "WrongGPU";
        } else if (s.state == UpscaleState::StandIn) label = "WrongGPU (stand-in)";
        else if (s.state == UpscaleState::Failed) label = "Upscaling unavailable";
        else {
            label = s.upscaler;
            if (label.empty()) label = "AMD FSR";
            else if (label.find("FSR") == std::string::npos) label = "AMD FSR " + label;
        }
        std::lock_guard<std::mutex> guard(mutex_);
        label_ = std::move(label);
    }
    // Returns required bytes including NUL. An undersized destination gets no partial/false label.
    uint32_t Copy(char* output, uint32_t capacity) {
        std::lock_guard<std::mutex> guard(mutex_);
        const uint32_t n = uint32_t(label_.size() + 1);
        if (output && capacity) {
            output[0] = 0;
            if (capacity >= n) std::memcpy(output, label_.c_str(), n);
        }
        return n;
    }
private:
    std::mutex mutex_;
    std::string label_ = "WrongGPU (no active frame)";
};
}
