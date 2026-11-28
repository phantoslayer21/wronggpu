// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// Injectable composition presentation back end for FgWinCompPresenter. The production implementation wraps
// D3D11 / DirectComposition / IPresentationManager (presentation.h); CPU tests supply a deterministic fake
// compositor implementing the same interface so all scheduling, pacing, statistics, watchdog and slot-release
// logic in FgWinCompPresenter is exercised without a GPU or display.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <d3d12.h>

#include <cstdint>

#include "fg_present_engine.h"
#include "fg_win_common.h"

namespace wgfg {

enum class FgCompStatKind : std::uint8_t {
    None = 0,
    PresentStatus,
    IndependentFlipFrame,
    CompositionFrame,
};

enum class FgCompPresentStatus : std::uint8_t {
    Queued = 0,
    Skipped,
    Canceled,
};

struct FgCompStatistic {
    FgCompStatKind kind = FgCompStatKind::None;
    std::uint64_t present_id = 0;
    FgCompPresentStatus status = FgCompPresentStatus::Queued;  // valid for PresentStatus
    FgTicks displayed_time = 0;                                // interrupt-time ticks (0 if unavailable)
    bool displayed_measured = false;                           // true only for measured hardware/compositor completion
    std::uint64_t comp_frame_id = 0;                           // GetCompositionFrameId() on PresentStatus
    std::uint32_t status_target_count = 0;                     // targets reported for comp_frame_id (first 16 statuses)
    std::uint64_t status_target_present_time = 0;              // target presentTime (QPC) if target_count > 0
    std::uint64_t status_target_completed_time = 0;            // target completedStats.time (QPC) if target_count > 0
};

struct FgCompBackendInfo {
    FgAdapterFlipCaps adapter;
    bool presentation_supported = false;
    bool independent_flip_supported = false;
    bool d3d11_displayable = false;
    std::uint32_t d3d11_misc_flags = 0;
    bool simultaneous_access = false;
    bool primed = false;
    bool hwnd_is_root = true;
    bool window_cloaked = false;
};

class IFgCompBackend {
public:
    virtual ~IFgCompBackend() = default;

    // Clock and wait handles. Optional handles may be null in single-threaded CPU tests.
    virtual FgTicks Now() const = 0;
    virtual HANDLE StatsEvent() const = 0;
    virtual HANDLE LostEvent() const = 0;
    virtual HANDLE FenceEvent() const = 0;
    virtual HANDLE SlotAvailableEvent(std::uint32_t slot) const = 0;
    virtual bool IsLost() const = 0;
    virtual bool IsWindowVisible() const = 0;
    virtual bool IsWindowCloaked() const { return false; }
    virtual std::uint64_t RetiringFenceValue() const { return 0; }
    virtual std::uint64_t PrimePresentId() const { return 0; }
    virtual bool ProbeCompFrame(std::uint64_t /*cfid*/, std::uint32_t* /*target_count*/,
                                std::uint64_t* /*completed_time*/) const {
        return false;
    }

    // Producer fences and D3D12 slot resources.
    virtual const IFgFenceSource* FenceSource() const = 0;
    virtual void WatchFence(const FgReadyFence& fence) = 0;
    virtual std::uint32_t RegisterFence(ID3D12Fence* fence, FgWinResult* why) = 0;
    virtual ID3D12Resource* SlotResource(std::uint32_t slot) const = 0;

    // Buffer and visual lifecycle.
    virtual FgCompBackendInfo BackendInfo() const = 0;
    virtual FgWinResult CreateBuffers(std::uint32_t slot_count, UINT width, UINT height, FgCompBackendInfo* info) = 0;
    virtual void DestroyBuffers() = 0;
    virtual void DrainRetiredHandles() {}
    virtual HRESULT SetSourceRect(UINT width, UINT height) = 0;
    virtual void DetachVisual() = 0;

    // Display-path operations (called on the presenter worker under mutex_).
    virtual bool IsBufferAvailable(std::uint32_t slot) const = 0;
    virtual HRESULT ForceVSyncInterrupt(bool enable) = 0;
    virtual HRESULT PresentSlot(std::uint32_t slot, FgTicks target_time, const FgReadyFence& fence,
                                std::uint64_t* platform_id_out) = 0;
    virtual HRESULT CancelPresentsFrom(std::uint64_t platform_id) = 0;
    virtual bool PopStatistic(FgCompStatistic* out) = 0;
    virtual bool ReadRefreshGrid(FgTicks period_override, FgRefreshGrid* out) = 0;
};

}  // namespace wgfg
