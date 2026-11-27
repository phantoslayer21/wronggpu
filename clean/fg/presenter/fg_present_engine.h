// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// Platform-neutral display-slot engine shared by the Windows presenter back ends (fg_win_comp_presenter.h,
// fg_dxgi_presenter.h): the slot pool, target-order admission, late drops with held leases, stale-suffix
// cancellation, the epoch / resize / device-loss state machine and the statistics. It keeps the rules of
// fg_presenter.h exactly as the deterministic fake does, but it does not own a clock, fences or a display:
//
//  - time is passed in by the caller (interrupt-time ticks);
//  - producer fences are answered by an IFgFenceSource (D3D12 fences in the back ends, scripted ones in tests);
//  - every present the engine queues is appended to an outbox the back end drains, in order, onto its display path;
//  - a cancellation returns the first cancelled present id, which the back end forwards to its display path
//    (IPresentationManager::CancelPresentsFrom, or its own not-yet-presented queue) BEFORE submitting the outbox;
//  - the display path reports back what happened (displayed / skipped / cancelled) and when a slot's buffer is free
//    again (OnSlotReleased). A slot is never handed out again before that: the producer can never overwrite a
//    buffer the display still owns.
//
// Not thread-safe: the back end serialises every call with its own lock. Plain C++17, no platform headers, so the
// whole state machine is tested on the CPU (tests/test_fg_win_presenters.cpp).
#include <cstdint>
#include <mutex>
#include <vector>

#include "fg_presenter.h"

namespace wgfg {

// Producer fence completion as the back end sees it.
class IFgFenceSource {
public:
    virtual ~IFgFenceSource() = default;
    virtual bool FenceValid(std::uint32_t fence) const = 0;
    virtual std::uint64_t FenceCompleted(std::uint32_t fence) const = 0;
};

// Called exactly once per accepted real frame when it retires from the presenter (displayed, skipped, cancelled by
// quiesce/detach/device-loss, or dropped by epoch — never on a CancelSuffix re-present). Must not re-enter the engine.
using FgRealRetireFn = void (*)(void* ctx, std::uint64_t frame_id, std::uint64_t interval_id, bool displayed);

// Lightweight per-image trace event (A3). event_kind: 0=queued(fence_done), 1=submitted, 2=displayed, 3=dropped_pre,
// 4=dropped_post, 5=skipped, 6=display submission estimate, 7=present lease taken. Must not re-enter the engine.
using FgPresentTraceFn = void (*)(void* ctx, std::uint64_t interval_id, std::uint64_t frame_id,
                                  std::uint32_t index, bool real,
                                  std::uint8_t event_kind, FgTicks ts, FgTicks target);

struct FgEngineConfig {
    FgPresenterProbe probe;
    std::uint32_t slot_count = kFgMaxGenerated + 4;  // >= N + 3: displayed, one per target, one being released
    std::uint32_t real_reserve = 1;                  // slots generated reservations leave free for real images
    std::uint32_t max_pending_intervals = 2;         // intervals with generated images not yet on the display path
    FgTicks pending_late_tolerance = 0; // >0: allow the backend to pace a ready image past its target
    FgTicks refresh_period = 0;                      // > 0: count displayed images closer than half of it
};

// Smallest slot pool that can hold one interval of n generated images: the image on screen, n + 1 targets and one
// buffer the display path is still releasing.
constexpr std::uint32_t FgMinSlotsFor(std::uint32_t n) { return n + 3; }

// One present the back end hands to its display path, in outbox order.
struct FgEnginePresent {
    std::uint64_t present_id = 0;  // engine id: strictly increasing, never reused
    std::uint32_t slot = 0;
    std::uint64_t epoch = 0;
    std::uint64_t interval_id = 0;
    std::uint64_t frame_id = 0;
    std::uint32_t index = 0;  // 1..N generated, N + 1 real (1 for a real-only plan)
    bool real = false;
    bool represent = false;   // re-presented after a cancelled suffix
    FgTicks target = 0;
    FgTicks deadline = 0;
    FgReadyFence fence;
};

enum class FgEngineSlotState : std::uint8_t {
    Free = 0,
    Reserved,   // handed to the executor, not yet submitted
    Pending,    // submitted, waiting for its producer fence / its turn
    Queued,     // on the display path
    Presenting, // leased to the DXGI worker, outside cancellation and slot reuse
    Displayed,  // on screen
    Retired,    // left the display path (superseded, skipped, cancelled): buffer not yet released by the display
    Draining,   // dropped before the display path: lease held until its producer fence completes
};

class FgPresentEngine {
public:
    FgPresentEngine(const FgEngineConfig& config, const IFgFenceSource* fences);

    const FgEngineConfig& Config() const { return config_; }
    FgPresenterState State() const { return state_; }
    const FgPresentStats& Stats() const { return stats_; }
    std::uint64_t Epoch() const { return epoch_; }
    void SetRealRetireCallback(FgRealRetireFn fn, void* ctx) {
        real_retire_fn_ = fn;
        real_retire_ctx_ = ctx;
    }
    void SetTraceCallback(FgPresentTraceFn fn, void* ctx) {
        trace_fn_ = fn;
        trace_ctx_ = ctx;
    }

    // IFgPresenter semantics (fg_presenter.h), with the caller's time.
    FgStatus Attach(std::uint64_t epoch);
    FgStatus BeginInterval(const FgIntervalPlan& plan, FgTicks now, FgIntervalReservation* out);
    FgStatus SubmitInterval(const FgIntervalSubmission& submission, FgTicks now);
    // A real-only image must live in a slot too. ReserveRealOnlySlot hands one out (NoSlot when none is free; one at
    // a time); PresentRealOnly then presents from it. Without a reservation PresentRealOnly takes any free slot (the
    // fake's behaviour, for tests that do not model image data).
    FgStatus ReserveRealOnlySlot(std::uint32_t* slot);
    void ReleaseRealOnlySlot();
    bool HasRealOnlyReservation() const { return real_only_slot_ >= 0; }
    FgStatus PresentRealOnly(const FgIntervalPlan& plan, const FgRealImage& real);
    // *cancel_from: first display-path present id the back end must cancel (0: none).
    FgStatus RevokeGenerated(std::uint64_t interval_id, FgDropReason reason, FgTicks now, std::uint64_t* cancel_from);
    FgStatus Quiesce(FgTicks now, std::uint64_t* cancel_from);
    FgStatus CompleteResize(bool resize_ok, std::uint64_t new_epoch);
    FgStatus OnDeviceLost();
    FgStatus Detach(FgTicks now, std::uint64_t* cancel_from);
    // Stale suffix: cancels every queued present with id >= present_id. Real images are re-presented (new ids, in
    // the outbox), generated ones dropped as Cancelled. Returns the number cancelled; *cancel_from as above.
    std::uint32_t CancelPresentsFrom(std::uint64_t present_id, FgTicks now, std::uint64_t* cancel_from);
    // Records a refusal the back end itself decided (bad device argument, ...) in the statistics.
    FgStatus Refuse(FgStatus s);
    // A hard display-path failure that is not a device loss (a present call failed): every queued and pending image
    // is dropped and the engine refuses until Detach.
    void MarkFailed();

    // Worker step: frees finished leases and moves ready images to the outbox in target order, dropping generated
    // images whose deadline passed. A real image is never dropped; later images wait behind it.
    void Pump(FgTicks now);
    // Moves the outbox into *out (appended) and clears it.
    void TakeOutbox(std::vector<FgEnginePresent>* out);
    bool OutboxEmpty() const { return outbox_.empty(); }

    // Display-path feedback, by engine present id. Unknown or stale ids are ignored (returns false).
    bool OnPresentSubmitted(std::uint64_t present_id, FgTicks when);  // handed to the OS present call
    bool OnPresentDisplayed(std::uint64_t present_id, FgTicks actual, bool measured = false);
    // DXGI hands an in-flight present to the worker. Cancellation only touches the remaining queued suffix.
    bool LeasePresent(std::uint32_t slot, std::uint64_t present_id);
    void FinishLeasedPresent(std::uint32_t slot, FgTicks submitted, FgTicks estimate, bool presented);
    bool OnPresentSkipped(std::uint64_t present_id);    // the display path skipped it (a newer present was ready)
    bool OnPresentDropped(std::uint64_t present_id, FgDropReason reason);  // the back end did not present it
    // The display path no longer uses the slot's buffer (available event / copy finished). Only a Retired slot, or a
    // Queued / Displayed one the back end consumed by a copy (consumed_by_copy), becomes Free.
    bool OnSlotReleased(std::uint32_t slot, bool consumed_by_copy);

    // Earliest deadline among pending generated images (0 when none): the worker's next drop check.
    FgTicks NextDeadline() const;
    // The fence the head of the pending queue waits for (false when nothing is pending).
    bool HeadPendingFence(FgReadyFence* out) const;
    // Engine present id of a Queued slot (0 otherwise).
    std::uint64_t SlotPresentId(std::uint32_t slot) const;
    bool Drained() const;  // nothing but the image on screen (and free slots) left
    // No producer work and nothing on its way to the display: every slot that is not free is only referenced by the
    // display path (Displayed / Retired). Such buffers never block a teardown or a resize that replaces every
    // buffer: the back end drops its own references (the compositor keeps its own) and never hands them out again.
    bool OnlyDisplayHeld() const;
    // Forgets display-held slots (Displayed / Retired -> Free) once the back end has dropped those buffers.
    void ForgetDisplayHeld();
    bool HasPending() const { return !pending_.empty(); }
    std::uint32_t FreeSlots() const;
    std::uint32_t LeasesHeld() const;
    std::uint32_t QueuedCount() const { return static_cast<std::uint32_t>(queued_.size()); }
    std::uint64_t FirstQueuedPresentId() const;
    FgEngineSlotState SlotState(std::uint32_t slot) const;
    std::uint32_t SlotCount() const { return static_cast<std::uint32_t>(slots_.size()); }
    // Slots the back end must still watch for release (Queued / Displayed / Retired).
    void SlotsAwaitingRelease(std::vector<std::uint32_t>* out) const;

private:
    struct Slot {
        FgEngineSlotState state = FgEngineSlotState::Free;
        std::uint64_t epoch = 0;
        std::uint64_t interval_id = 0;
        std::uint64_t frame_id = 0;
        std::uint32_t index = 0;
        bool real = false;
        FgTicks target = 0;
        FgTicks deadline = 0;
        FgReadyFence fence;
        std::uint64_t present_id = 0;
        bool represent = false;
        bool real_credit_active = false;
    };

    void RetireReal(Slot& s, bool displayed);
    void CountSubmission(const Slot& s, FgTicks when);
    void CountDisplay(const Slot& s, FgTicks when, bool measured);
    void EmitTrace(const Slot& s, std::uint8_t event_kind, FgTicks ts) const;
    FgStatus CheckAccepting() const;
    FgStatus CheckPlan(const FgIntervalPlan& plan) const;
    void CommitTimeline(const FgIntervalPlan& plan);
    int FindFree() const;
    bool FenceDone(const Slot& s) const;
    FgTicks PendingDeadline(const Slot& s) const;
    void Drop(std::uint32_t slot, FgDropReason reason, bool hold_lease);
    void Queue(std::uint32_t slot, bool represent);
    int FindQueued(std::uint64_t present_id) const;
    void RemoveQueued(std::uint32_t slot);
    std::uint32_t CancelSuffix(std::uint64_t from_id, bool all_stale, std::uint64_t revoked_interval,
                               FgDropReason reason, FgTicks now, std::uint64_t* cancel_from);
    void CancelAllQueued(std::uint64_t* cancel_from);
    void RetireUnshown(std::uint64_t* cancel_from);
    void ReleaseAll();

    FgEngineConfig config_;
    const IFgFenceSource* fences_;
    FgPresenterState state_ = FgPresenterState::Detached;
    std::uint64_t epoch_ = 0;
    std::vector<Slot> slots_;
    std::vector<std::uint32_t> pending_;  // target order
    std::vector<std::uint32_t> queued_;   // present-id order
    std::vector<std::uint32_t> scratch_;
    std::vector<FgEnginePresent> outbox_;
    int displayed_ = -1;
    std::uint64_t next_present_id_ = 1;
    bool have_target_ = false;
    FgTicks last_target_ = 0;
    std::uint64_t last_interval_id_ = 0;
    bool open_ = false;
    FgIntervalReservation open_reservation_;
    int real_only_slot_ = -1;
    bool have_display_ = false;
    bool have_estimated_display_ = false;
    FgTicks last_display_ = 0;
    FgPresentStats stats_;
    FgRealRetireFn real_retire_fn_ = nullptr;
    void* real_retire_ctx_ = nullptr;
    FgPresentTraceFn trace_fn_ = nullptr;
    void* trace_ctx_ = nullptr;
};

// One visible owner per chain, process-wide and thread-safe (the Windows back ends share it).
class FgChainRegistry {
public:
    static FgChainRegistry& Process();
    bool Claim(std::uint64_t chain, const void* owner);
    void Release(std::uint64_t chain, const void* owner);
    const void* Owner(std::uint64_t chain) const;

private:
    struct Entry {
        std::uint64_t chain;
        const void* owner;
    };
    mutable std::mutex mutex_;
    std::vector<Entry> entries_;
};

// --- Pacing arithmetic (pure, CPU-tested) --------------------------------------------------------------------------

// A fixed-refresh scan-out grid: vblank k is at origin + k * period (interrupt-time ticks).
struct FgRefreshGrid {
    FgTicks origin = 0;
    FgTicks period = 0;  // 0: unknown (variable refresh or not measured)
};

// The vblank nearest `target` (ties go to the later one) in *vblank, and in *submit the time to request so the image
// lands on exactly that vblank with half a refresh of tolerance on either side: vblank - period / 2. An arbitrary
// target phase that sits near a vblank boundary can land on either side of it from one present to the next (two
// images in one refresh, then a double gap); a target at the middle of the acceptance window cannot.
// InvalidArgument when the grid has no period; Overflow when the arithmetic does not fit.
FgStatus FgSnapToRefreshGrid(FgTicks target, const FgRefreshGrid& grid, FgTicks* vblank, FgTicks* submit);

// k when `interval` is k refresh periods (k >= 1) within `tolerance`, else 0. A fixed-refresh display can only show
// a cadence evenly when every image stays for a whole number of refreshes.
std::uint32_t FgRefreshMultiple(FgTicks interval, FgTicks refresh_period, FgTicks tolerance);

// A fixed-refresh back end shows a generating plan evenly only when its image interval period / (N + 1) is a whole
// number of refreshes (within an eighth of a refresh); real-only plans always fit.
bool FgPlanFitsRefresh(const FgIntervalPlan& plan, FgTicks refresh_period);

// Relative due time for SetWaitableTimerEx (negative 100 ns units) to wake `lead` before `target`; 0 when that time
// has already come (present at once).
FgTicks FgTimerDue(FgTicks target, FgTicks now, FgTicks lead);

// R3B-7: tearing / VRR head planning decision shared by FgDxgiPresenter::PlanHeadLocked and
// FgWinCompPresenter::SubmitOutboxLocked. Enforces >= min_interval spacing via issue_at while giving `next` a bounded
// slip allowance min(tol/2, slot/2) so near-ceiling VRR output (320-349 Hz) tolerates ordinary 0.3-1.0 ms issue jitter
// without dropping ready generated images.
struct FgTearingPlanDecision {
    FgTicks issue_at = 0;
    bool drop_late = false;
};

inline FgTearingPlanDecision FgPlanTearingImage(FgTicks target, bool real, FgTicks now, FgTicks last_issue,
                                                FgTicks min_interval, FgTicks base_late_tolerance, bool has_next,
                                                FgTicks next_target) {
    FgTearingPlanDecision d;
    const FgTicks earliest = last_issue + min_interval;
    d.issue_at = target > earliest ? target : earliest;
    if (real) return d;
    const FgTicks slot = has_next && next_target > target ? next_target - target : 0;
    FgTicks tol = base_late_tolerance;
    if (slot > 0) {
        const FgTicks half = slot / 2;
        const FgTicks scaled = half < 15000 ? 15000 : (half > 40000 ? 40000 : half);
        if (scaled > tol) tol = scaled;
    }
    if (now > target + tol) {
        d.drop_late = true;
        return d;
    }
    if (has_next) {
        const FgTicks issue = now > d.issue_at ? now : d.issue_at;
        const FgTicks gap = min_interval > 2500 ? min_interval : 2500;
        // Absorb submission jitter without reducing the physical minimum gap between issues.
        const FgTicks slot_allow = slot - slot / 4;
        const FgTicks slip_allow = tol < slot_allow ? tol : slot_allow;
        if (issue + gap > next_target + slip_allow) d.drop_late = true;
    }
    return d;
}

}  // namespace wgfg
