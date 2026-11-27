// SPDX-License-Identifier: GPL-3.0-only
// Deterministic fake presenter: see fg_fake_presenter.h.
#include "fg_fake_presenter.h"

#include <cstddef>

#include "fg_target_scheduler.h"

namespace wgfg {

// --- Fake GPU fences -----------------------------------------------------------------------------------------------

std::uint32_t FgFakeGpu::CreateFence() {
    fences_.emplace_back();
    return static_cast<std::uint32_t>(fences_.size());
}

void FgFakeGpu::SignalAt(std::uint32_t fence, std::uint64_t value, FgTicks when) {
    if (!Valid(fence)) return;
    fences_[fence - 1].push_back(Signal{when, value});
}

std::uint64_t FgFakeGpu::Completed(std::uint32_t fence, FgTicks now) const {
    if (!Valid(fence)) return 0;
    std::uint64_t done = 0;
    for (const Signal& s : fences_[fence - 1])
        if (s.when <= now && s.value > done) done = s.value;
    return done;
}

// --- Chain ownership -----------------------------------------------------------------------------------------------

bool FgFakeChainRegistry::Claim(std::uint64_t chain, const void* owner) {
    for (const Entry& e : entries_)
        if (e.chain == chain) return e.owner == owner;
    entries_.push_back(Entry{chain, owner});
    return true;
}

void FgFakeChainRegistry::Release(std::uint64_t chain, const void* owner) {
    for (std::size_t i = 0; i < entries_.size(); ++i) {
        if (entries_[i].chain == chain && entries_[i].owner == owner) {
            entries_.erase(entries_.begin() + static_cast<std::ptrdiff_t>(i));
            return;
        }
    }
}

const void* FgFakeChainRegistry::Owner(std::uint64_t chain) const {
    for (const Entry& e : entries_)
        if (e.chain == chain) return e.owner;
    return nullptr;
}

// --- Fake presenter ------------------------------------------------------------------------------------------------

FgFakePresenter::FgFakePresenter(const FgFakeConfig& config, FgFakeClock* clock, FgFakeGpu* gpu,
                                 FgFakeChainRegistry* registry)
    : config_(config), clock_(clock), gpu_(gpu), registry_(registry) {
    if (config_.slot_count < 2) config_.slot_count = 2;
    if (config_.real_reserve < 1) config_.real_reserve = 1;
    if (config_.max_pending_intervals < 1) config_.max_pending_intervals = 1;
    slots_.resize(config_.slot_count);
    pending_.reserve(config_.slot_count);
    queued_.reserve(config_.slot_count);
    scratch_.reserve(config_.slot_count);
}

FgFakePresenter::~FgFakePresenter() {
    if (state_ != FgPresenterState::Detached && registry_) registry_->Release(chain_, this);
}

FgStatus FgFakePresenter::Refuse(FgStatus s) {
    ++stats_.refused_calls;
    return s;
}

FgStatus FgFakePresenter::CheckAccepting() const {
    switch (state_) {
        case FgPresenterState::Attached: return FgStatus::Ok;
        case FgPresenterState::Lost: return FgStatus::DeviceLost;
        case FgPresenterState::Failed: return FgStatus::Refused;
        case FgPresenterState::Detached:
        case FgPresenterState::Quiescing: return FgStatus::WrongState;
    }
    return FgStatus::WrongState;
}

// A plan must be valid in itself and continue this epoch's timeline: same epoch, a newer interval, t_base equal to
// the last committed real target (or, re-anchored, not earlier than it), every target later than everything before.
FgStatus FgFakePresenter::CheckPlan(const FgIntervalPlan& plan) const {
    if (plan.epoch != epoch_) return FgStatus::StaleEpoch;
    if (FgValidatePlan(plan) != FgStatus::Ok) return FgStatus::InvalidArgument;
    if (plan.interval_id <= last_interval_id_) return FgStatus::InvalidArgument;
    if (have_target_) {
        if (plan.reanchored ? plan.t_base < last_target_ : plan.t_base != last_target_) return FgStatus::InvalidArgument;
        if (plan.targets[0].target <= last_target_) return FgStatus::InvalidArgument;
    }
    return FgStatus::Ok;
}

void FgFakePresenter::CommitTimeline(const FgIntervalPlan& plan) {
    have_target_ = true;
    last_target_ = plan.targets[plan.target_count - 1].target;
    last_interval_id_ = plan.interval_id;
    ++stats_.game_presents;
}

int FgFakePresenter::FindFree() const {
    for (std::size_t i = 0; i < slots_.size(); ++i)
        if (slots_[i].state == SlotState::Free) return static_cast<int>(i);
    return -1;
}

std::uint32_t FgFakePresenter::FreeSlots() const {
    std::uint32_t n = 0;
    for (const Slot& s : slots_)
        if (s.state == SlotState::Free) ++n;
    return n;
}

std::uint32_t FgFakePresenter::LeasesHeld() const {
    std::uint32_t n = 0;
    for (const Slot& s : slots_)
        if (s.state == SlotState::Draining) ++n;
    return n;
}

std::uint64_t FgFakePresenter::FirstQueuedPresentId() const {
    return queued_.empty() ? 0 : slots_[queued_.front()].present_id;
}

bool FgFakePresenter::FenceDone(const Slot& s) const {
    return gpu_ && gpu_->Completed(s.fence.fence, clock_->Now()) >= s.fence.value;
}

bool FgFakePresenter::Drained() const {
    for (const Slot& s : slots_)
        if (s.state != SlotState::Free && s.state != SlotState::Displayed) return false;
    return !open_;
}

void FgFakePresenter::Drop(std::uint32_t slot, FgDropReason reason, bool hold_lease) {
    Slot& s = slots_[slot];
    dropped_log_.push_back(FgFakeDropRecord{s.epoch, s.interval_id, s.frame_id, s.index, s.real, reason, clock_->Now()});
    switch (reason) {
        case FgDropReason::Late: ++stats_.dropped_late; break;
        case FgDropReason::Pressure: ++stats_.dropped_pressure; break;
        case FgDropReason::Cancelled: break;  // counted per cancelled present
        case FgDropReason::Revoked: ++stats_.revoked; break;
        case FgDropReason::Epoch: ++stats_.dropped_epoch; break;
        case FgDropReason::None: break;
    }
    // The output lease stays held until the producer fence completes.
    s.state = hold_lease && !FenceDone(s) ? SlotState::Draining : SlotState::Free;
}

void FgFakePresenter::Queue(std::uint32_t slot) {
    Slot& s = slots_[slot];
    s.state = SlotState::Queued;
    s.present_id = next_present_id_++;
    s.submitted_at = clock_->Now();
    queued_.push_back(slot);
    if (s.real) ++stats_.real_queued;
    else ++stats_.generated_queued;
}

FgStatus FgFakePresenter::Attach(std::uint64_t chain_id, std::uint64_t epoch) {
    if (!config_.probe.supported)
        return Refuse(config_.probe.refusal == FgStatus::Ok ? FgStatus::Unsupported : config_.probe.refusal);
    if (state_ != FgPresenterState::Detached) return Refuse(FgStatus::WrongState);
    if (!registry_ || !registry_->Claim(chain_id, this)) return Refuse(FgStatus::Refused);
    chain_ = chain_id;
    epoch_ = epoch;
    state_ = FgPresenterState::Attached;
    have_target_ = false;
    last_interval_id_ = 0;
    ++stats_.epochs;
    return FgStatus::Ok;
}

FgStatus FgFakePresenter::BeginInterval(const FgIntervalPlan& plan, FgIntervalReservation* out) {
    if (!out) return Refuse(FgStatus::InvalidArgument);
    *out = FgIntervalReservation();
    FgStatus s = CheckAccepting();
    if (s != FgStatus::Ok) return Refuse(s);
    if (open_) return Refuse(FgStatus::WrongState);  // one open reservation at a time
    if (plan.generated_count == 0) return Refuse(FgStatus::InvalidArgument);  // real-only plans: PresentRealOnly
    if ((s = CheckPlan(plan)) != FgStatus::Ok) return Refuse(s);

    // The real image first, from any free slot: generated reservations always leave real_reserve slots free.
    const int real_slot = FindFree();
    if (real_slot < 0) return Refuse(FgStatus::NoSlot);

    std::uint32_t pending_intervals = 0;
    {
        std::uint64_t seen[16] = {};
        std::uint32_t seen_count = 0;
        for (const Slot& sl : slots_) {
            if (sl.real || (sl.state != SlotState::Reserved && sl.state != SlotState::Pending)) continue;
            bool known = false;
            for (std::uint32_t i = 0; i < seen_count; ++i) known = known || seen[i] == sl.interval_id;
            if (!known && seen_count < 16) seen[seen_count++] = sl.interval_id;
        }
        pending_intervals = seen_count;
    }

    const std::uint32_t n = plan.generated_count;
    out->epoch = plan.epoch;
    out->interval_id = plan.interval_id;
    out->recorder = FgOutputRecorder::Executor;
    out->count = plan.target_count;
    Slot& r = slots_[static_cast<std::size_t>(real_slot)];
    r = Slot();
    r.state = SlotState::Reserved;
    r.epoch = plan.epoch;
    r.interval_id = plan.interval_id;
    r.frame_id = plan.real_frame_id;
    r.index = n + 1;
    r.real = true;
    r.target = plan.targets[n].target;
    r.deadline = plan.targets[n].deadline;
    FgPresentSlot& rs = out->slots[n];
    rs = FgPresentSlot{plan.epoch, plan.interval_id, plan.real_frame_id, n + 1, true,
                       static_cast<std::uint32_t>(real_slot), true, FgDropReason::None};

    std::uint32_t reserved_generated = 0;
    for (std::uint32_t k = 0; k < n; ++k) {
        const FgFrameTarget& t = plan.targets[k];
        FgPresentSlot& ps = out->slots[k];
        ps = FgPresentSlot{plan.epoch, plan.interval_id, plan.real_frame_id, t.index, false, 0, false,
                           FgDropReason::None};
        if (!t.admitted) {
            ps.refused = FgDropReason::Late;
            ++stats_.dropped_late;
            dropped_log_.push_back(
                FgFakeDropRecord{plan.epoch, plan.interval_id, plan.real_frame_id, t.index, false, FgDropReason::Late,
                                 clock_->Now()});
            continue;
        }
        const int g = FreeSlots() > config_.real_reserve && pending_intervals < config_.max_pending_intervals
                          ? FindFree()
                          : -1;
        if (g < 0) {
            ps.refused = FgDropReason::Pressure;
            ++stats_.dropped_pressure;
            dropped_log_.push_back(FgFakeDropRecord{plan.epoch, plan.interval_id, plan.real_frame_id, t.index, false,
                                                    FgDropReason::Pressure, clock_->Now()});
            continue;
        }
        Slot& gs = slots_[static_cast<std::size_t>(g)];
        gs = Slot();
        gs.state = SlotState::Reserved;
        gs.epoch = plan.epoch;
        gs.interval_id = plan.interval_id;
        gs.frame_id = plan.real_frame_id;
        gs.index = t.index;
        gs.real = false;
        gs.target = t.target;
        gs.deadline = t.deadline;
        ps.slot = static_cast<std::uint32_t>(g);
        ps.reserved = true;
        ++reserved_generated;
    }
    if (reserved_generated == 0)
        ++stats_.real_only_intervals[static_cast<std::uint32_t>(FgRealOnlyReason::Pressure)];
    open_ = true;
    open_reservation_ = *out;
    CommitTimeline(plan);
    return FgStatus::Ok;
}

FgStatus FgFakePresenter::SubmitInterval(const FgIntervalSubmission& sub) {
    FgStatus s = CheckAccepting();
    if (s != FgStatus::Ok) return Refuse(s);
    if (!open_ || sub.interval_id != open_reservation_.interval_id) return Refuse(FgStatus::WrongState);
    if (sub.epoch != epoch_ || sub.epoch != open_reservation_.epoch) return Refuse(FgStatus::StaleEpoch);
    if (sub.count != open_reservation_.count) return Refuse(FgStatus::InvalidArgument);
    // Check everything before changing anything.
    for (std::uint32_t i = 0; i < sub.count; ++i) {
        const FgPresentSlot& ps = open_reservation_.slots[i];
        if (!ps.reserved) continue;
        if (ps.real && !sub.produced[i]) return Refuse(FgStatus::InvalidArgument);  // the real image is required
        if (sub.produced[i] && (!gpu_ || !gpu_->Valid(sub.ready[i].fence))) return Refuse(FgStatus::InvalidArgument);
    }
    for (std::uint32_t i = 0; i < sub.count; ++i) {
        const FgPresentSlot& ps = open_reservation_.slots[i];
        if (!ps.reserved) continue;
        Slot& sl = slots_[ps.slot];
        if (!sub.produced[i]) {  // the executor recorded nothing for this index: no lease to hold
            Drop(ps.slot, FgDropReason::Late, false);
            continue;
        }
        sl.fence = sub.ready[i];
        sl.state = SlotState::Pending;
        pending_.push_back(ps.slot);
    }
    open_ = false;
    return FgStatus::Ok;
}

FgStatus FgFakePresenter::PresentRealOnly(const FgIntervalPlan& plan, const FgRealImage& real) {
    FgStatus s = CheckAccepting();
    if (s != FgStatus::Ok) return Refuse(s);
    if (open_) return Refuse(FgStatus::WrongState);
    if (plan.generated_count != 0) return Refuse(FgStatus::InvalidArgument);
    if (real.epoch != epoch_) return Refuse(FgStatus::StaleEpoch);
    if (real.frame_id != plan.real_frame_id || !gpu_ || !gpu_->Valid(real.ready.fence))
        return Refuse(FgStatus::InvalidArgument);
    if ((s = CheckPlan(plan)) != FgStatus::Ok) return Refuse(s);
    const int slot = FindFree();
    if (slot < 0) return Refuse(FgStatus::NoSlot);
    Slot& r = slots_[static_cast<std::size_t>(slot)];
    r = Slot();
    r.state = SlotState::Pending;
    r.epoch = plan.epoch;
    r.interval_id = plan.interval_id;
    r.frame_id = plan.real_frame_id;
    r.index = 1;
    r.real = true;
    r.target = plan.targets[0].target;
    r.deadline = plan.targets[0].deadline;
    r.fence = real.ready;
    pending_.push_back(static_cast<std::uint32_t>(slot));
    ++stats_.real_only_intervals[static_cast<std::uint32_t>(plan.real_only_reason)];
    CommitTimeline(plan);
    return FgStatus::Ok;
}

// Cancels the queued suffix from `from_id`. Real images are re-presented as fresh endpoints (same target, new present
// id, original order). Generated images are dropped with `reason` when `all_stale` or when they belong to
// `revoked_interval`; other generated images are re-presented while still before their deadline (they were not
// stale), else dropped as cancelled.
std::uint32_t FgFakePresenter::CancelSuffix(std::uint64_t from_id, bool all_stale, std::uint64_t revoked_interval,
                                            FgDropReason reason) {
    scratch_.clear();
    std::size_t keep = 0;
    for (std::size_t i = 0; i < queued_.size(); ++i) {
        if (slots_[queued_[i]].present_id >= from_id) scratch_.push_back(queued_[i]);
        else queued_[keep++] = queued_[i];
    }
    queued_.resize(keep);
    const FgTicks now = clock_->Now();
    for (const std::uint32_t slot : scratch_) {
        const Slot& s = slots_[slot];
        ++stats_.cancelled;
        if (s.real) {
            ++stats_.real_represented;
            Queue(slot);
        } else if (all_stale || s.interval_id == revoked_interval) {
            Drop(slot, reason, false);  // queued images were ready: no lease to wait for
        } else if (now <= s.deadline) {
            Queue(slot);
        } else {
            Drop(slot, FgDropReason::Cancelled, false);
        }
    }
    return static_cast<std::uint32_t>(scratch_.size());
}

std::uint32_t FgFakePresenter::CancelPresentsFrom(std::uint64_t present_id) {
    if (state_ != FgPresenterState::Attached) return 0;
    return CancelSuffix(present_id, true, 0, FgDropReason::Cancelled);
}

FgStatus FgFakePresenter::RevokeGenerated(std::uint64_t interval_id, FgDropReason reason) {
    const FgStatus s = CheckAccepting();
    if (s != FgStatus::Ok) return Refuse(s);
    if (reason == FgDropReason::None) return Refuse(FgStatus::InvalidArgument);
    bool found = false;
    // not yet on the display path: drop (holding leases until their fences complete)
    std::size_t keep = 0;
    for (std::size_t i = 0; i < pending_.size(); ++i) {
        Slot& sl = slots_[pending_[i]];
        if (!sl.real && sl.interval_id == interval_id) {
            found = true;
            Drop(pending_[i], reason, true);
        } else {
            pending_[keep++] = pending_[i];
        }
    }
    pending_.resize(keep);
    // reserved but not submitted
    if (open_ && open_reservation_.interval_id == interval_id) {
        for (std::uint32_t i = 0; i < open_reservation_.count; ++i) {
            FgPresentSlot& ps = open_reservation_.slots[i];
            if (ps.real || !ps.reserved) continue;
            found = true;
            Drop(ps.slot, reason, false);
            ps.reserved = false;
            ps.refused = reason;
        }
    }
    // already queued: cancel the suffix from the interval's first queued generated present
    std::uint64_t first = 0;
    for (std::uint32_t q : queued_) {
        const Slot& sl = slots_[q];
        if (!sl.real && sl.interval_id == interval_id && (first == 0 || sl.present_id < first)) first = sl.present_id;
    }
    if (first != 0) {
        found = true;
        CancelSuffix(first, false, interval_id, reason);
    }
    return found ? FgStatus::Ok : Refuse(FgStatus::InvalidArgument);
}

void FgFakePresenter::Pump() {
    if (state_ == FgPresenterState::Lost || state_ == FgPresenterState::Detached) return;
    const FgTicks now = clock_->Now();
    // finished leases
    for (Slot& s : slots_)
        if (s.state == SlotState::Draining && FenceDone(s)) s.state = SlotState::Free;
    // ready images to the display path, in target order
    std::size_t consumed = 0;
    for (; consumed < pending_.size(); ++consumed) {
        const std::uint32_t slot = pending_[consumed];
        Slot& s = slots_[slot];
        const bool ready = FenceDone(s);
        if (s.real) {
            if (!ready) break;  // a real image is never dropped; later images wait behind it
            Queue(slot);
            continue;
        }
        if (now > s.deadline) {
            Drop(slot, FgDropReason::Late, true);
            continue;
        }
        if (!ready) break;
        Queue(slot);
    }
    pending_.erase(pending_.begin(), pending_.begin() + static_cast<std::ptrdiff_t>(consumed));
    // the display flips queued presents at their target times, no faster than the panel allows
    std::size_t flipped = 0;
    for (; flipped < queued_.size(); ++flipped) {
        Slot& s = slots_[queued_[flipped]];
        FgTicks actual = s.target > s.submitted_at ? s.target : s.submitted_at;
        if (have_flip_ && actual < last_flip_ + config_.min_flip_interval) actual = last_flip_ + config_.min_flip_interval;
        if (actual > now) break;
        if (displayed_ >= 0) slots_[static_cast<std::size_t>(displayed_)].state = SlotState::Free;  // buffer available
        s.state = SlotState::Displayed;
        displayed_ = static_cast<int>(queued_[flipped]);
        have_flip_ = true;
        last_flip_ = actual;
        displayed_log_.push_back(
            FgFakeDisplayRecord{s.present_id, s.epoch, s.interval_id, s.frame_id, s.index, s.real, s.target, actual});
        if (s.real) ++stats_.displayed_real;
        else ++stats_.displayed_generated;
        if (actual - s.target > stats_.max_slip) stats_.max_slip = actual - s.target;
    }
    queued_.erase(queued_.begin(), queued_.begin() + static_cast<std::ptrdiff_t>(flipped));
}

FgStatus FgFakePresenter::Quiesce() {
    if (state_ == FgPresenterState::Lost) return Refuse(FgStatus::DeviceLost);
    if (state_ == FgPresenterState::Detached) return Refuse(FgStatus::WrongState);
    if (state_ == FgPresenterState::Quiescing) return FgStatus::Ok;
    // stop admission and retire the epoch before anything can race it
    state_ = FgPresenterState::Quiescing;
    for (std::uint32_t q : queued_) {
        ++stats_.cancelled;
        Drop(q, FgDropReason::Cancelled, false);  // queued images were ready: no lease to wait for
    }
    queued_.clear();
    for (std::uint32_t p : pending_) Drop(p, FgDropReason::Epoch, true);
    pending_.clear();
    if (open_) {
        for (std::uint32_t i = 0; i < open_reservation_.count; ++i)
            if (open_reservation_.slots[i].reserved) Drop(open_reservation_.slots[i].slot, FgDropReason::Epoch, false);
        open_ = false;
    }
    return FgStatus::Ok;
}

FgStatus FgFakePresenter::CompleteResize(bool resize_ok, std::uint64_t new_epoch) {
    if (state_ == FgPresenterState::Lost) return Refuse(FgStatus::DeviceLost);
    if (state_ != FgPresenterState::Quiescing) return Refuse(FgStatus::WrongState);
    if (!Drained()) return Refuse(FgStatus::Busy);  // old buffers still referenced: never a fake success
    if (!resize_ok) {
        state_ = FgPresenterState::Failed;  // the resize's own result is the truth
        return Refuse(FgStatus::ResizeFailed);
    }
    if (new_epoch <= epoch_) return Refuse(FgStatus::InvalidArgument);
    if (displayed_ >= 0) {  // detach the old epoch's last displayed buffer
        slots_[static_cast<std::size_t>(displayed_)].state = SlotState::Free;
        displayed_ = -1;
    }
    epoch_ = new_epoch;
    have_target_ = false;
    last_interval_id_ = 0;
    state_ = FgPresenterState::Attached;
    ++stats_.epochs;
    return FgStatus::Ok;
}

void FgFakePresenter::ReleaseAll() {
    for (Slot& s : slots_) s = Slot();
    pending_.clear();
    queued_.clear();
    displayed_ = -1;
    open_ = false;
}

FgStatus FgFakePresenter::OnDeviceLost() {
    if (state_ == FgPresenterState::Detached) return Refuse(FgStatus::WrongState);
    // teardown semantics: nothing waits on a lost device; stale fences and callbacks are ignored from here on
    std::uint64_t unshown = 0;
    for (const Slot& s : slots_)
        if (s.state == SlotState::Pending || s.state == SlotState::Queued || s.state == SlotState::Reserved) ++unshown;
    stats_.dropped_epoch += unshown;
    ReleaseAll();
    state_ = FgPresenterState::Lost;
    return FgStatus::Ok;
}

FgStatus FgFakePresenter::Detach() {
    switch (state_) {
        case FgPresenterState::Detached: return Refuse(FgStatus::WrongState);
        case FgPresenterState::Lost:
        case FgPresenterState::Failed: break;
        case FgPresenterState::Attached:
            Quiesce();
            [[fallthrough]];
        case FgPresenterState::Quiescing:
            if (!Drained()) return Refuse(FgStatus::Busy);
            break;
    }
    ReleaseAll();
    if (registry_) registry_->Release(chain_, this);
    state_ = FgPresenterState::Detached;
    return FgStatus::Ok;
}

}  // namespace wgfg
