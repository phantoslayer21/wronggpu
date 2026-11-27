// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// DXGI flip-discard presenter: IFgPresenter over the game's IDXGISwapChain4 (DXGI_SWAP_EFFECT_FLIP_DISCARD with a
// frame-latency waitable object) on Direct3D 12, paced by the presenter's own high-resolution timer.
//
//  - Tearing (variable refresh / VSync off): the worker sleeps on a CREATE_WAITABLE_TIMER_HIGH_RESOLUTION timer until
//    each image's target time and presents it with Present(0, DXGI_PRESENT_ALLOW_TEARING). Requires
//    DXGI_FEATURE_PRESENT_ALLOW_TEARING and a chain created with DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING.
//  - VSync (fixed refresh): tear-free Present(1, 0). Every target is snapped to its vblank and the present is issued
//    half a refresh before it, so it lands on exactly that vblank. Only plans whose image interval is a whole number
//    of refreshes are accepted (Unsupported otherwise): a fixed-refresh display cannot show other cadences evenly.
//  - Slots are presenter-owned Direct3D 12 textures with the swap chain's size and format. The executor writes a
//    reserved slot and signals its producer fence; at present time the worker makes the game's queue wait for that
//    fence on the GPU, copies the slot into the current back buffer and presents. A slot is free again once its
//    copy completed (its own fence), so the executor never writes an image that is still being read.
//  - A dedicated worker thread does all waiting (timer, frame-latency object, fences): the game thread only
//    reserves, submits and cancels and never stalls on generated-frame pacing.
//  - Refuses explicitly (no fallback to another mode or back end) when the chain or the system lacks what the mode
//    needs. Device removal turns the presenter Lost; any other failed Present turns it Failed.
//
// The swap chain's buffers are referenced only while attached; Quiesce lets the worker drop them once its copies
// finished (WaitIdle), so the game's ResizeBuffers can succeed. CompleteResize takes the new buffers. Detach and the
// destructor drain the queue first (a fence signalled after the last copy and present), so the swap chain can be
// released right afterwards.
#include <d3d12.h>
#include <dxgi1_6.h>

#include <deque>
#include <memory>
#include <mutex>
#include <vector>

#include "fg_present_engine.h"
#include "fg_win_common.h"

namespace wgfg {

enum class FgDxgiPacing : std::uint8_t {
    Tearing = 0,  // Present(0, DXGI_PRESENT_ALLOW_TEARING) at each target time
    Vsync,        // Present(1, 0) issued half a refresh before each target's vblank
};

struct FgDxgiPresenterDesc {
    ID3D12Device* device = nullptr;
    ID3D12CommandQueue* queue = nullptr;  // the queue the swap chain was created on
    IDXGISwapChain4* swap_chain = nullptr;
    FgDxgiPacing pacing = FgDxgiPacing::Tearing;
    FgTicks refresh_period = 0;   // VSync: the scan-out period (0: from the compositor); tearing: optional
    FgTicks timer_lead = 0;       // wake this much before the present time
    FgTicks late_tolerance = 10000;  // a generated image later than this past its target is dropped
    std::uint32_t max_generated = kFgMaxGenerated;  // 1..5
    std::uint32_t slot_count = 0;  // 0: max_generated + 3; below that is refused
    bool allow_unordered_access = false;  // also create the slots with D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
};

struct FgDxgiPresenterInfo {
    bool tearing_supported = false;
    UINT buffer_count = 0;
    UINT width = 0;
    UINT height = 0;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    std::uint32_t slot_count = 0;
    FgRefreshGrid grid;
    std::uint64_t presents = 0;          // Present calls that succeeded
    std::uint64_t latency_timeouts = 0;  // frame-latency waits that timed out
};

class FgDxgiPresenter final : public IFgPresenter, private IFgWorkerBody {
public:
    static FgWinResult Create(const FgDxgiPresenterDesc& desc, std::unique_ptr<FgDxgiPresenter>* out);
    ~FgDxgiPresenter() override;
    FgDxgiPresenter(const FgDxgiPresenter&) = delete;
    FgDxgiPresenter& operator=(const FgDxgiPresenter&) = delete;

    // An executor fence of the same device. Returns the id for FgReadyFence::fence, 0 when refused.
    std::uint32_t RegisterFence(ID3D12Fence* fence, FgWinResult* why);
    ID3D12Resource* SlotResource(std::uint32_t slot) const;
    FgStatus ReserveRealSlot(std::uint32_t* slot);
    void ReleaseRealSlot();
    std::uint32_t CancelPresentsFrom(std::uint64_t present_id);
    std::uint64_t FirstQueuedPresentId() const;
    // Waits (bounded) until no old work is left and, while quiescing, the swap-chain buffers are released.
    bool WaitIdle(DWORD timeout_ms);
    bool HoldsSwapChainBuffers() const;
    FgDxgiPresenterInfo Info() const;
    std::uint32_t FreeSlots() const;

    FgPresenterProbe Probe() const override;
    FgStatus Attach(std::uint64_t chain_id, std::uint64_t epoch) override;
    FgStatus BeginInterval(const FgIntervalPlan& plan, FgIntervalReservation* out) override;
    FgStatus SubmitInterval(const FgIntervalSubmission& submission) override;
    FgStatus PresentRealOnly(const FgIntervalPlan& plan, const FgRealImage& real) override;
    FgStatus RevokeGenerated(std::uint64_t interval_id, FgDropReason reason) override;
    FgStatus Quiesce() override;
    FgStatus CompleteResize(bool resize_ok, std::uint64_t new_epoch) override;
    FgStatus OnDeviceLost() override;
    FgStatus Detach() override;
    FgPresenterState State() const override;
    FgPresentStats Stats() const override;

private:
    struct Planned {
        FgEnginePresent present;
        FgTicks issue_at = 0;  // when to call Present
        FgTicks shown_at = 0;  // the vblank (VSync) or the target (tearing)
    };

    FgDxgiPresenter(const FgDxgiPresenterDesc& desc, const FgEngineConfig& engine);
    FgWinResult Initialize();
    FgWinResult AcquireChain();  // buffers, size, format; (re)creates slots when they changed
    FgWinResult CreateSlots();
    void Run(HANDLE stop, HANDLE wake) override;
    void StepLocked(FgTicks now);
    bool PlanHeadLocked(FgTicks now, Planned* out);
    bool PresentHead(HANDLE stop);  // false when the worker must stop
    bool RecordCopyAndPresentLocked(const Planned& p);
    void ReleaseCopiedSlotsLocked();
    void CancelLocalLocked(std::uint64_t engine_from);
    void LoseLocked();
    bool CopiesIdleLocked() const;
    // Signals the copy fence on the game's queue after everything this presenter put there (copies and presents) and
    // waits for it (bounded), so no queued present still uses a back buffer or a slot when they are let go.
    bool DrainQueueLocked(DWORD timeout_ms);

    FgDxgiPresenterDesc desc_;
    FgDxgiPresenterInfo info_;
    Microsoft::WRL::ComPtr<ID3D12Device> device_;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> queue_;
    Microsoft::WRL::ComPtr<IDXGISwapChain4> swap_;
    std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> back_buffers_;
    std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> slots_;
    std::vector<std::uint64_t> slot_copy_value_;  // copy fence value that finishes reading the slot (0: none)
    std::vector<Microsoft::WRL::ComPtr<ID3D12CommandAllocator>> allocators_;
    std::vector<std::uint64_t> allocator_value_;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> list_;
    Microsoft::WRL::ComPtr<ID3D12Fence> copy_fence_;
    std::uint64_t copy_value_ = 0;
    std::uint32_t ring_ = 0;
    HANDLE latency_ = nullptr;
    HANDLE copy_event_ = nullptr;
    HANDLE fence_event_ = nullptr;
    HANDLE drain_event_ = nullptr;  // DrainQueueLocked only (the worker owns copy_event_)
    FgD3D12FenceTable fences_;
    FgClockMap clock_map_;
    FgHighResTimer timer_;

    mutable std::mutex mutex_;
    FgPresentEngine engine_;
    std::deque<FgEnginePresent> local_;  // queued on the engine, not yet presented (present-id order)
    bool latency_credit_ = false;        // a frame-latency count taken but not yet used by a Present
    bool have_vblank_ = false;
    FgTicks last_vblank_ = 0;
    FgTicks grid_updated_at_ = 0;
    std::uint64_t chain_ = 0;
    bool claimed_ = false;
    FgWorkerThread worker_;
};

}  // namespace wgfg
