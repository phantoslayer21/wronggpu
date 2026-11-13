// SPDX-License-Identifier: MIT
#pragma once
// Private, read-only CPU diagnostics. No fence signals/waits, command recording or model data.
#include <d3d12.h>
#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>

namespace wgwin {
class SplitProgress {
public:
    using LogFn = void (*)(void*, const char*);
    bool Track(uint64_t id, const void* list, ID3D12Device* device, ID3D12Fence* order, uint64_t ticket,
               ID3D12Fence* done, uint64_t doneValue, ID3D12Fence* resume, uint64_t resumeValue, uint64_t nowUs) noexcept;
    void Submitted(uint64_t id, const void* queue);
    void CpuReleased(uint64_t id);
    void Poll(uint64_t nowUs, LogFn log, void* user);
    size_t Count();
private:
    struct Entry;
    std::mutex mutex_;
    std::array<std::shared_ptr<Entry>, 64> entries_{};
};
// Process-lifetime singleton; the optional caller must pin its module before starting this worker.
SplitProgress& ProcessSplitProgress();
bool StartSplitProgressWorker(SplitProgress::LogFn log, void* user, uint64_t (*nowUs)());
}  // namespace wgwin
