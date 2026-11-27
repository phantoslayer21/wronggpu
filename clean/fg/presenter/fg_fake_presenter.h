// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// Deterministic fake presenter for CPU tests of the presentation boundary: a fake interrupt-time clock, fake GPU
// fences signalled at chosen times, a fixed pool of display slots, a chain-ownership registry and a display that
// flips queued presents at their target times (no earlier than the panel's shortest interval after the previous
// flip). It implements IFgPresenter with the same rules a real back end keeps, so the scheduler and the lifecycle
// (attach, intervals, real-only, late drops, suffix cancellation, resize, device loss, refusal) are tested without a
// GPU or a window. Single-threaded: Pump() stands in for the presenter's worker wakeup.
#include <cstdint>
#include <vector>

#include "fg_presenter.h"

namespace wgfg {

class FgFakeClock {
public:
    FgTicks Now() const { return now_; }
    void Set(FgTicks t) { now_ = t; }
    void Advance(FgTicks d) { now_ += d; }

private:
    FgTicks now_ = 0;
};

// Fences with values that complete at chosen times. Completed() is the highest value whose time has come.
class FgFakeGpu {
public:
    std::uint32_t CreateFence();  // ids start at 1
    bool Valid(std::uint32_t fence) const { return fence >= 1 && fence <= fences_.size(); }
    void SignalAt(std::uint32_t fence, std::uint64_t value, FgTicks when);
    std::uint64_t Completed(std::uint32_t fence, FgTicks now) const;

private:
    struct Signal {
        FgTicks when;
        std::uint64_t value;
    };
    std::vector<std::vector<Signal>> fences_;
};

// One visible owner per chain.
class FgFakeChainRegistry {
public:
    bool Claim(std::uint64_t chain, const void* owner);
    void Release(std::uint64_t chain, const void* owner);
    const void* Owner(std::uint64_t chain) const;

private:
    struct Entry {
        std::uint64_t chain;
        const void* owner;
    };
    std::vector<Entry> entries_;
};

struct FgFakeConfig {
    FgPresenterProbe probe = {FgPresenterBackend::Fake, true, true, kFgMaxGenerated, FgStatus::Ok};
    // displayed image + one interval of six + a real-only recovery slot
    std::uint32_t slot_count = 8;
    std::uint32_t real_reserve = 1;           // slots generated reservations must leave free for real images
    std::uint32_t max_pending_intervals = 2;  // intervals with generated images not yet on the display path
    FgTicks min_flip_interval = 0;            // panel ceiling: shortest time between two flips
};

struct FgFakeDisplayRecord {
    std::uint64_t present_id;
    std::uint64_t epoch;
    std::uint64_t interval_id;
    std::uint64_t frame_id;
    std::uint32_t index;
    bool real;
    FgTicks target;
    FgTicks actual;
};

struct FgFakeDropRecord {
    std::uint64_t epoch;
    std::uint64_t interval_id;
    std::uint64_t frame_id;
    std::uint32_t index;
    bool real;
    FgDropReason reason;
    FgTicks at;
};

class FgFakePresenter final : public IFgPresenter {
public:
    FgFakePresenter(const FgFakeConfig& config, FgFakeClock* clock, FgFakeGpu* gpu, FgFakeChainRegistry* registry);
    ~FgFakePresenter() override;
    FgFakePresenter(const FgFakePresenter&) = delete;
    FgFakePresenter& operator=(const FgFakePresenter&) = delete;

    FgPresenterProbe Probe() const override { return config_.probe; }
    FgStatus Attach(std::uint64_t chain_id, std::uint64_t epoch) override;
    FgStatus BeginInterval(const FgIntervalPlan& plan, FgIntervalReservation* out) override;
    FgStatus SubmitInterval(const FgIntervalSubmission& submission) override;
    FgStatus PresentRealOnly(const FgIntervalPlan& plan, const FgRealImage& real) override;
    FgStatus RevokeGenerated(std::uint64_t interval_id, FgDropReason reason) override;
    FgStatus Quiesce() override;
    FgStatus CompleteResize(bool resize_ok, std::uint64_t new_epoch) override;
    FgStatus OnDeviceLost() override;
    FgStatus Detach() override;
    FgPresenterState State() const override { return state_; }
    FgPresentStats Stats() const override { return stats_; }

    // The presenter worker's wakeup at the clock's current time: frees finished leases, hands ready images to the
    // display in target order (dropping late generated ones), and flips queued presents whose time has come.
    void Pump();
    // Cancels every queued present with an id >= present_id (a stale suffix). Real images in it are re-presented as
    // fresh endpoints; generated ones are dropped (Cancelled). Returns the number of cancelled presents.
    std::uint32_t CancelPresentsFrom(std::uint64_t present_id);
    // No old work left besides the image on screen (reserved, pending, queued or draining slots).
    bool Drained() const;

    std::uint32_t FreeSlots() const;
    std::uint32_t LeasesHeld() const;            // dropped / revoked images whose producer fence is still pending
    std::uint64_t FirstQueuedPresentId() const;  // 0 when nothing is queued
    std::uint32_t QueuedCount() const { return static_cast<std::uint32_t>(queued_.size()); }
    const std::vector<FgFakeDisplayRecord>& Displayed() const { return displayed_log_; }
    const std::vector<FgFakeDropRecord>& Dropped() const { return dropped_log_; }

private:
    enum class SlotState : std::uint8_t { Free = 0, Reserved, Pending, Queued, Displayed, Draining };
    struct Slot {
        SlotState state = SlotState::Free;
        std::uint64_t epoch = 0;
        std::uint64_t interval_id = 0;
        std::uint64_t frame_id = 0;
        std::uint32_t index = 0;
        bool real = false;
        FgTicks target = 0;
        FgTicks deadline = 0;
        FgReadyFence fence;
        std::uint64_t present_id = 0;
        FgTicks submitted_at = 0;
    };

    FgStatus Refuse(FgStatus s);
    FgStatus CheckAccepting() const;
    FgStatus CheckPlan(const FgIntervalPlan& plan) const;
    void CommitTimeline(const FgIntervalPlan& plan);
    int FindFree() const;
    bool FenceDone(const Slot& s) const;
    void Drop(std::uint32_t slot, FgDropReason reason, bool hold_lease);
    void Queue(std::uint32_t slot);
    void ReleaseAll();
    std::uint32_t CancelSuffix(std::uint64_t from_id, bool all_stale, std::uint64_t revoked_interval, FgDropReason reason);

    FgFakeConfig config_;
    FgFakeClock* clock_;
    FgFakeGpu* gpu_;
    FgFakeChainRegistry* registry_;
    FgPresenterState state_ = FgPresenterState::Detached;
    std::uint64_t chain_ = 0;
    std::uint64_t epoch_ = 0;
    std::vector<Slot> slots_;
    std::vector<std::uint32_t> pending_;  // slot numbers, target order
    std::vector<std::uint32_t> queued_;   // slot numbers, present-id order
    std::vector<std::uint32_t> scratch_;
    int displayed_ = -1;
    std::uint64_t next_present_id_ = 1;
    bool have_target_ = false;
    FgTicks last_target_ = 0;
    std::uint64_t last_interval_id_ = 0;
    bool open_ = false;
    FgIntervalReservation open_reservation_;
    bool have_flip_ = false;
    FgTicks last_flip_ = 0;
    FgPresentStats stats_;
    std::vector<FgFakeDisplayRecord> displayed_log_;
    std::vector<FgFakeDropRecord> dropped_log_;
};

}  // namespace wgfg
