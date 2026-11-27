// SPDX-License-Identifier: GPL-3.0-only
// Platform-neutral display-slot engine: see fg_present_engine.h.
#include "fg_present_engine.h"

#include <cstddef>
#include <limits>

#include "fg_target_scheduler.h"

namespace wgfg {

FgPresentEngine::FgPresentEngine(const FgEngineConfig& config, const IFgFenceSource* fences)
    : config_(config), fences_(fences) {
    if (config_.slot_count < FgMinSlotsFor(1)) config_.slot_count = FgMinSlotsFor(1);
    if (config_.real_reserve < 1) config_.real_reserve = 1;
    if (config_.max_pending_intervals < 1) config_.max_pending_intervals = 1;
    slots_.resize(config_.slot_count);
    pending_.reserve(config_.slot_count);
    queued_.reserve(config_.slot_count);
    scratch_.reserve(config_.slot_count);
    outbox_.reserve(config_.slot_count * 2);
}

FgStatus FgPresentEngine::Refuse(FgStatus s) {
    ++stats_.refused_calls;
    return s;
}

// Everything not yet on screen leaves: queued presents are cancelled (buffers Retired until released), pending
// images dropped with their leases held, the open reservation and a real-only reservation returned.
void FgPresentEngine::RetireUnshown(std::uint64_t* cancel_from) {
    CancelAllQueued(cancel_from);
    for (const std::uint32_t p : pending_) Drop(p, FgDropReason::Epoch, true);
    pending_.clear();
    if (open_) {
        for (std::uint32_t i = 0; i < open_reservation_.count; ++i)
            if (open_reservation_.slots[i].reserved) Drop(open_reservation_.slots[i].slot, FgDropReason::Epoch, false);
        open_ = false;
    }
    if (real_only_slot_ >= 0) {
        slots_[static_cast<std::size_t>(real_only_slot_)] = Slot();
        real_only_slot_ = -1;
    }
}

void FgPresentEngine::MarkFailed() {
    if (state_ != FgPresenterState::Attached && state_ != FgPresenterState::Quiescing) return;
    std::uint64_t cancel = 0;
    RetireUnshown(&cancel);
    state_ = FgPresenterState::Failed;
}

FgStatus FgPresentEngine::CheckAccepting() const {
    switch (state_) {
        case FgPresenterState::Attached: return FgStatus::Ok;
        case FgPresenterState::Lost: return FgStatus::DeviceLost;
        case FgPresenterState::Failed: return FgStatus::Refused;
        case FgPresenterState::Detached:
        case FgPresenterState::Quiescing: return FgStatus::WrongState;
    }
    return FgStatus::WrongState;
}

// Same epoch, valid in itself, a newer interval, continuing the committed timeline (fg_fake_presenter.cpp keeps the
// identical rule).
FgStatus FgPresentEngine::CheckPlan(const FgIntervalPlan& plan) const {
    if (plan.epoch != epoch_) return FgStatus::StaleEpoch;
    if (FgValidatePlan(plan) != FgStatus::Ok) return FgStatus::InvalidArgument;
    if (plan.interval_id <= last_interval_id_) return FgStatus::InvalidArgument;
    if (have_target_) {
        if (plan.reanchored ? plan.t_base < last_target_ : plan.t_base != last_target_) return FgStatus::InvalidArgument;
        if (plan.targets[0].target <= last_target_) return FgStatus::InvalidArgument;
    }
    return FgStatus::Ok;
}

void FgPresentEngine::CommitTimeline(const FgIntervalPlan& plan) {
    have_target_ = true;
    last_target_ = plan.targets[plan.target_count - 1].target;
    last_interval_id_ = plan.interval_id;
    ++stats_.game_presents;
}

int FgPresentEngine::FindFree() const {
    for (std::size_t i = 0; i < slots_.size(); ++i)
        if (slots_[i].state == FgEngineSlotState::Free) return static_cast<int>(i);
    return -1;
}

std::uint32_t FgPresentEngine::FreeSlots() const {
    std::uint32_t n = 0;
    for (const Slot& s : slots_)
        if (s.state == FgEngineSlotState::Free) ++n;
    return n;
}

std::uint32_t FgPresentEngine::LeasesHeld() const {
    std::uint32_t n = 0;
    for (const Slot& s : slots_)
        if (s.state == FgEngineSlotState::Draining) ++n;
    return n;
}

std::uint64_t FgPresentEngine::FirstQueuedPresentId() const {
    return queued_.empty() ? 0 : slots_[queued_.front()].present_id;
}

FgEngineSlotState FgPresentEngine::SlotState(std::uint32_t slot) const {
    return slot < slots_.size() ? slots_[slot].state : FgEngineSlotState::Free;
}

void FgPresentEngine::SlotsAwaitingRelease(std::vector<std::uint32_t>* out) const {
    for (std::size_t i = 0; i < slots_.size(); ++i) {
        const FgEngineSlotState st = slots_[i].state;
        if (st == FgEngineSlotState::Queued || st == FgEngineSlotState::Displayed || st == FgEngineSlotState::Retired)
            out->push_back(static_cast<std::uint32_t>(i));
    }
}

bool FgPresentEngine::FenceDone(const Slot& s) const {
    return fences_ && fences_->FenceValid(s.fence.fence) && fences_->FenceCompleted(s.fence.fence) >= s.fence.value;
}

bool FgPresentEngine::Drained() const {
    for (std::size_t i = 0; i < slots_.size(); ++i) {
        const FgEngineSlotState st = slots_[i].state;
        if (st == FgEngineSlotState::Free) continue;
        if (st == FgEngineSlotState::Displayed && static_cast<int>(i) == displayed_) continue;
        return false;
    }
    return !open_ && outbox_.empty();
}

bool FgPresentEngine::OnlyDisplayHeld() const {
    if (open_ || !outbox_.empty() || !pending_.empty() || !queued_.empty() || real_only_slot_ >= 0) return false;
    for (const Slot& s : slots_)
        if (s.state != FgEngineSlotState::Free && s.state != FgEngineSlotState::Displayed &&
            s.state != FgEngineSlotState::Retired)
            return false;
    return true;
}

void FgPresentEngine::ForgetDisplayHeld() {
    for (Slot& s : slots_)
        if (s.state == FgEngineSlotState::Displayed || s.state == FgEngineSlotState::Retired) s = Slot();
    displayed_ = -1;
}

bool FgPresentEngine::HeadPendingFence(FgReadyFence* out) const {
    if (pending_.empty() || !out) return false;
    *out = slots_[pending_.front()].fence;
    return true;
}

std::uint64_t FgPresentEngine::SlotPresentId(std::uint32_t slot) const {
    return slot < slots_.size() && slots_[slot].state == FgEngineSlotState::Queued ? slots_[slot].present_id : 0;
}

FgTicks FgPresentEngine::NextDeadline() const {
    FgTicks next = 0;
    for (const std::uint32_t p : pending_) {
        const Slot& s = slots_[p];
        if (!s.real && (next == 0 || s.deadline < next)) next = s.deadline;
    }
    return next;
}

void FgPresentEngine::Drop(std::uint32_t slot, FgDropReason reason, bool hold_lease) {
    Slot& s = slots_[slot];
    switch (reason) {
        case FgDropReason::Late: ++stats_.dropped_late; break;
        case FgDropReason::Pressure: ++stats_.dropped_pressure; break;
        case FgDropReason::Cancelled: break;  // counted per cancelled present
        case FgDropReason::Revoked: ++stats_.revoked; break;
        case FgDropReason::Epoch: ++stats_.dropped_epoch; break;
        case FgDropReason::None: break;
    }
    // The output lease stays held until the producer fence completes.
    s.state = hold_lease && !FenceDone(s) ? FgEngineSlotState::Draining : FgEngineSlotState::Free;
}

void FgPresentEngine::Queue(std::uint32_t slot, bool represent) {
    Slot& s = slots_[slot];
    s.state = FgEngineSlotState::Queued;
    s.present_id = next_present_id_++;
    s.represent = represent;
    queued_.push_back(slot);
    if (s.real) ++stats_.real_queued;
    else ++stats_.generated_queued;
    FgEnginePresent p;
    p.present_id = s.present_id;
    p.slot = slot;
    p.epoch = s.epoch;
    p.interval_id = s.interval_id;
    p.frame_id = s.frame_id;
    p.index = s.index;
    p.real = s.real;
    p.represent = represent;
    p.target = s.target;
    p.deadline = s.deadline;
    p.fence = s.fence;
    outbox_.push_back(p);
}

void FgPresentEngine::TakeOutbox(std::vector<FgEnginePresent>* out) {
    if (out) out->insert(out->end(), outbox_.begin(), outbox_.end());
    outbox_.clear();
}

int FgPresentEngine::FindQueued(std::uint64_t present_id) const {
    for (std::size_t i = 0; i < queued_.size(); ++i)
        if (slots_[queued_[i]].present_id == present_id) return static_cast<int>(i);
    return -1;
}

void FgPresentEngine::RemoveQueued(std::uint32_t slot) {
    for (std::size_t i = 0; i < queued_.size(); ++i) {
        if (queued_[i] == slot) {
            queued_.erase(queued_.begin() + static_cast<std::ptrdiff_t>(i));
            return;
        }
    }
}

FgStatus FgPresentEngine::Attach(std::uint64_t epoch) {
    if (!config_.probe.supported)
        return Refuse(config_.probe.refusal == FgStatus::Ok ? FgStatus::Unsupported : config_.probe.refusal);
    if (state_ != FgPresenterState::Detached) return Refuse(FgStatus::WrongState);
    epoch_ = epoch;
    state_ = FgPresenterState::Attached;
    have_target_ = false;
    last_interval_id_ = 0;
    have_display_ = false;
    ++stats_.epochs;
    return FgStatus::Ok;
}

FgStatus FgPresentEngine::BeginInterval(const FgIntervalPlan& plan, FgTicks now, FgIntervalReservation* out) {
    if (!out) return Refuse(FgStatus::InvalidArgument);
    *out = FgIntervalReservation();
    FgStatus s = CheckAccepting();
    if (s != FgStatus::Ok) return Refuse(s);
    if (open_) return Refuse(FgStatus::WrongState);
    if (plan.generated_count == 0) return Refuse(FgStatus::InvalidArgument);  // real-only plans: PresentRealOnly
    if (plan.generated_count > config_.probe.max_generated) return Refuse(FgStatus::Unsupported);
    if ((s = CheckPlan(plan)) != FgStatus::Ok) return Refuse(s);

    const int real_slot = FindFree();
    if (real_slot < 0) return Refuse(FgStatus::NoSlot);

    std::uint32_t pending_intervals = 0;
    {
        std::uint64_t seen[64] = {};
        std::uint32_t seen_count = 0;
        for (const Slot& sl : slots_) {
            if (sl.real || (sl.state != FgEngineSlotState::Reserved && sl.state != FgEngineSlotState::Pending)) continue;
            bool known = false;
            for (std::uint32_t i = 0; i < seen_count; ++i) known = known || seen[i] == sl.interval_id;
            if (!known && seen_count < 64) seen[seen_count++] = sl.interval_id;
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
    r.state = FgEngineSlotState::Reserved;
    r.epoch = plan.epoch;
    r.interval_id = plan.interval_id;
    r.frame_id = plan.real_frame_id;
    r.index = n + 1;
    r.real = true;
    r.target = plan.targets[n].target;
    r.deadline = plan.targets[n].deadline;
    out->slots[n] = FgPresentSlot{plan.epoch, plan.interval_id, plan.real_frame_id, n + 1, true,
                                  static_cast<std::uint32_t>(real_slot), true, FgDropReason::None};

    std::uint32_t reserved_generated = 0;
    for (std::uint32_t k = 0; k < n; ++k) {
        const FgFrameTarget& t = plan.targets[k];
        FgPresentSlot& ps = out->slots[k];
        ps = FgPresentSlot{plan.epoch, plan.interval_id, plan.real_frame_id, t.index, false, 0, false,
                           FgDropReason::None};
        // admitted at commit, but the deadline may have passed since: never reserve a slot that cannot be shown
        if (!t.admitted || now > t.deadline) {
            ps.refused = FgDropReason::Late;
            ++stats_.dropped_late;
            continue;
        }
        const int g = FreeSlots() > config_.real_reserve && pending_intervals < config_.max_pending_intervals
                          ? FindFree()
                          : -1;
        if (g < 0) {
            ps.refused = FgDropReason::Pressure;
            ++stats_.dropped_pressure;
            continue;
        }
        Slot& gs = slots_[static_cast<std::size_t>(g)];
        gs = Slot();
        gs.state = FgEngineSlotState::Reserved;
        gs.epoch = plan.epoch;
        gs.interval_id = plan.interval_id;
        gs.frame_id = plan.real_frame_id;
        gs.index = t.index;
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

FgStatus FgPresentEngine::SubmitInterval(const FgIntervalSubmission& sub, FgTicks now) {
    FgStatus s = CheckAccepting();
    if (s != FgStatus::Ok) return Refuse(s);
    if (!open_ || sub.interval_id != open_reservation_.interval_id) return Refuse(FgStatus::WrongState);
    if (sub.epoch != epoch_ || sub.epoch != open_reservation_.epoch) return Refuse(FgStatus::StaleEpoch);
    if (sub.count != open_reservation_.count) return Refuse(FgStatus::InvalidArgument);
    for (std::uint32_t i = 0; i < sub.count; ++i) {
        const FgPresentSlot& ps = open_reservation_.slots[i];
        if (!ps.reserved) continue;
        if (ps.real && !sub.produced[i]) return Refuse(FgStatus::InvalidArgument);  // the real image is required
        if (sub.produced[i] && (!fences_ || !fences_->FenceValid(sub.ready[i].fence)))
            return Refuse(FgStatus::InvalidArgument);
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
        sl.state = FgEngineSlotState::Pending;
        pending_.push_back(ps.slot);
    }
    open_ = false;
    Pump(now);  // images that are already ready go out at once
    return FgStatus::Ok;
}

FgStatus FgPresentEngine::ReserveRealOnlySlot(std::uint32_t* slot) {
    if (!slot) return Refuse(FgStatus::InvalidArgument);
    const FgStatus s = CheckAccepting();
    if (s != FgStatus::Ok) return Refuse(s);
    if (real_only_slot_ >= 0) return Refuse(FgStatus::WrongState);
    const int free_slot = FindFree();
    if (free_slot < 0) return Refuse(FgStatus::NoSlot);
    Slot& r = slots_[static_cast<std::size_t>(free_slot)];
    r = Slot();
    r.state = FgEngineSlotState::Reserved;
    r.epoch = epoch_;
    r.real = true;
    real_only_slot_ = free_slot;
    *slot = static_cast<std::uint32_t>(free_slot);
    return FgStatus::Ok;
}

FgStatus FgPresentEngine::PresentRealOnly(const FgIntervalPlan& plan, const FgRealImage& real) {
    FgStatus s = CheckAccepting();
    if (s != FgStatus::Ok) return Refuse(s);
    if (open_) return Refuse(FgStatus::WrongState);
    if (plan.generated_count != 0) return Refuse(FgStatus::InvalidArgument);
    if (real.epoch != epoch_) return Refuse(FgStatus::StaleEpoch);
    if (real.frame_id != plan.real_frame_id || !fences_ || !fences_->FenceValid(real.ready.fence))
        return Refuse(FgStatus::InvalidArgument);
    if ((s = CheckPlan(plan)) != FgStatus::Ok) return Refuse(s);
    const int slot = real_only_slot_ >= 0 ? real_only_slot_ : FindFree();
    if (slot < 0) return Refuse(FgStatus::NoSlot);
    real_only_slot_ = -1;
    Slot& r = slots_[static_cast<std::size_t>(slot)];
    r = Slot();
    r.state = FgEngineSlotState::Pending;
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

// Cancels the queued suffix from `from_id`: real images are re-presented as fresh endpoints (same target, new id,
// original order); generated images are dropped when `all_stale` or when they belong to `revoked_interval`, other
// generated images re-presented while still before their deadline, else dropped as cancelled. Every cancelled
// present's buffer stays Retired until the display path releases it; a re-present reuses its own buffer.
std::uint32_t FgPresentEngine::CancelSuffix(std::uint64_t from_id, bool all_stale, std::uint64_t revoked_interval,
                                            FgDropReason reason, FgTicks now, std::uint64_t* cancel_from) {
    scratch_.clear();
    std::size_t keep = 0;
    for (std::size_t i = 0; i < queued_.size(); ++i) {
        if (slots_[queued_[i]].present_id >= from_id) scratch_.push_back(queued_[i]);
        else queued_[keep++] = queued_[i];
    }
    queued_.resize(keep);
    if (scratch_.empty()) return 0;
    std::uint64_t first = std::numeric_limits<std::uint64_t>::max();
    for (const std::uint32_t slot : scratch_)
        if (slots_[slot].present_id < first) first = slots_[slot].present_id;
    if (cancel_from && (*cancel_from == 0 || first < *cancel_from)) *cancel_from = first;
    for (const std::uint32_t slot : scratch_) {
        Slot& s = slots_[slot];
        ++stats_.cancelled;
        if (s.real) {
            ++stats_.real_represented;
            Queue(slot, true);
        } else if (all_stale || s.interval_id == revoked_interval) {
            if (reason == FgDropReason::Revoked) ++stats_.revoked;
            s.state = FgEngineSlotState::Retired;
        } else if (now <= s.deadline) {
            Queue(slot, true);
        } else {
            s.state = FgEngineSlotState::Retired;
        }
    }
    return static_cast<std::uint32_t>(scratch_.size());
}

void FgPresentEngine::CancelAllQueued(std::uint64_t* cancel_from) {
    if (queued_.empty()) return;
    std::uint64_t first = slots_[queued_.front()].present_id;
    for (const std::uint32_t q : queued_)
        if (slots_[q].present_id < first) first = slots_[q].present_id;
    if (cancel_from && (*cancel_from == 0 || first < *cancel_from)) *cancel_from = first;
    for (const std::uint32_t q : queued_) {
        ++stats_.cancelled;
        slots_[q].state = FgEngineSlotState::Retired;
    }
    queued_.clear();
    // presents still in the outbox never reached the display path: the back end discards them with the cancel
    outbox_.clear();
}

std::uint32_t FgPresentEngine::CancelPresentsFrom(std::uint64_t present_id, FgTicks now, std::uint64_t* cancel_from) {
    if (cancel_from) *cancel_from = 0;
    if (state_ != FgPresenterState::Attached) return 0;
    return CancelSuffix(present_id, true, 0, FgDropReason::Cancelled, now, cancel_from);
}

FgStatus FgPresentEngine::RevokeGenerated(std::uint64_t interval_id, FgDropReason reason, FgTicks now,
                                          std::uint64_t* cancel_from) {
    if (cancel_from) *cancel_from = 0;
    const FgStatus s = CheckAccepting();
    if (s != FgStatus::Ok) return Refuse(s);
    if (reason == FgDropReason::None) return Refuse(FgStatus::InvalidArgument);
    bool found = false;
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
    std::uint64_t first = 0;
    for (const std::uint32_t q : queued_) {
        const Slot& sl = slots_[q];
        if (!sl.real && sl.interval_id == interval_id && (first == 0 || sl.present_id < first)) first = sl.present_id;
    }
    if (first != 0) {
        found = true;
        CancelSuffix(first, false, interval_id, reason, now, cancel_from);
    }
    return found ? FgStatus::Ok : Refuse(FgStatus::InvalidArgument);
}

void FgPresentEngine::Pump(FgTicks now) {
    if (state_ == FgPresenterState::Lost || state_ == FgPresenterState::Detached) return;
    for (Slot& s : slots_)
        if (s.state == FgEngineSlotState::Draining && FenceDone(s)) s.state = FgEngineSlotState::Free;
    if (state_ != FgPresenterState::Attached) return;
    std::size_t consumed = 0;
    for (; consumed < pending_.size(); ++consumed) {
        const std::uint32_t slot = pending_[consumed];
        Slot& s = slots_[slot];
        const bool ready = FenceDone(s);
        if (s.real) {
            if (!ready) break;  // a real image is never dropped; later images wait behind it
            Queue(slot, false);
            continue;
        }
        if (now > s.deadline) {
            Drop(slot, FgDropReason::Late, true);
            continue;
        }
        if (!ready) break;
        Queue(slot, false);
    }
    pending_.erase(pending_.begin(), pending_.begin() + static_cast<std::ptrdiff_t>(consumed));
}

bool FgPresentEngine::OnPresentDisplayed(std::uint64_t present_id, FgTicks actual) {
    const int qi = FindQueued(present_id);
    if (qi < 0) return false;
    const std::uint32_t slot = queued_[static_cast<std::size_t>(qi)];
    queued_.erase(queued_.begin() + qi);
    if (displayed_ >= 0 && static_cast<std::uint32_t>(displayed_) != slot)
        slots_[static_cast<std::size_t>(displayed_)].state = FgEngineSlotState::Retired;  // superseded
    Slot& s = slots_[slot];
    s.state = FgEngineSlotState::Displayed;
    displayed_ = static_cast<int>(slot);
    if (s.real) ++stats_.displayed_real;
    else ++stats_.displayed_generated;
    if (actual - s.target > stats_.max_slip) stats_.max_slip = actual - s.target;
    if (config_.refresh_period > 0 && have_display_ && actual - last_display_ < config_.refresh_period / 2)
        ++stats_.display_collisions;
    have_display_ = true;
    last_display_ = actual;
    return true;
}

bool FgPresentEngine::OnPresentSkipped(std::uint64_t present_id) {
    const int qi = FindQueued(present_id);
    if (qi < 0) return false;
    const std::uint32_t slot = queued_[static_cast<std::size_t>(qi)];
    queued_.erase(queued_.begin() + qi);
    slots_[slot].state = FgEngineSlotState::Retired;
    ++stats_.display_skipped;
    return true;
}

bool FgPresentEngine::OnPresentDropped(std::uint64_t present_id, FgDropReason reason) {
    const int qi = FindQueued(present_id);
    if (qi < 0) return false;
    const std::uint32_t slot = queued_[static_cast<std::size_t>(qi)];
    if (slots_[slot].real || reason == FgDropReason::None) return false;  // a real image is never dropped
    queued_.erase(queued_.begin() + qi);
    Drop(slot, reason, false);  // never handed to the display: the buffer is free at once
    return true;
}

bool FgPresentEngine::OnSlotReleased(std::uint32_t slot, bool consumed_by_copy) {
    if (slot >= slots_.size()) return false;
    Slot& s = slots_[slot];
    const bool releasable =
        s.state == FgEngineSlotState::Retired ||
        (consumed_by_copy && (s.state == FgEngineSlotState::Displayed || s.state == FgEngineSlotState::Queued));
    if (!releasable) return false;
    if (s.state == FgEngineSlotState::Queued) RemoveQueued(slot);
    if (static_cast<int>(slot) == displayed_) displayed_ = -1;
    s = Slot();
    return true;
}

FgStatus FgPresentEngine::Quiesce(FgTicks now, std::uint64_t* cancel_from) {
    (void)now;
    if (cancel_from) *cancel_from = 0;
    if (state_ == FgPresenterState::Lost) return Refuse(FgStatus::DeviceLost);
    if (state_ == FgPresenterState::Detached) return Refuse(FgStatus::WrongState);
    if (state_ == FgPresenterState::Quiescing) return FgStatus::Ok;
    state_ = FgPresenterState::Quiescing;  // stop admission before anything can race it
    RetireUnshown(cancel_from);
    return FgStatus::Ok;
}

FgStatus FgPresentEngine::CompleteResize(bool resize_ok, std::uint64_t new_epoch) {
    if (state_ == FgPresenterState::Lost) return Refuse(FgStatus::DeviceLost);
    if (state_ != FgPresenterState::Quiescing) return Refuse(FgStatus::WrongState);
    if (!Drained()) return Refuse(FgStatus::Busy);  // old buffers still referenced: never a fake success
    if (!resize_ok) {
        state_ = FgPresenterState::Failed;
        return Refuse(FgStatus::ResizeFailed);
    }
    if (new_epoch <= epoch_) return Refuse(FgStatus::InvalidArgument);
    if (displayed_ >= 0) {  // the back end replaces every buffer for the new epoch
        slots_[static_cast<std::size_t>(displayed_)] = Slot();
        displayed_ = -1;
    }
    epoch_ = new_epoch;
    have_target_ = false;
    last_interval_id_ = 0;
    have_display_ = false;
    state_ = FgPresenterState::Attached;
    ++stats_.epochs;
    return FgStatus::Ok;
}

void FgPresentEngine::ReleaseAll() {
    for (Slot& s : slots_) s = Slot();
    pending_.clear();
    queued_.clear();
    outbox_.clear();
    displayed_ = -1;
    real_only_slot_ = -1;
    open_ = false;
}

FgStatus FgPresentEngine::OnDeviceLost() {
    if (state_ == FgPresenterState::Detached) return Refuse(FgStatus::WrongState);
    std::uint64_t unshown = 0;
    for (const Slot& s : slots_)
        if (s.state == FgEngineSlotState::Pending || s.state == FgEngineSlotState::Queued ||
            s.state == FgEngineSlotState::Reserved)
            ++unshown;
    stats_.dropped_epoch += unshown;
    ReleaseAll();  // nothing waits on a lost device
    state_ = FgPresenterState::Lost;
    return FgStatus::Ok;
}

FgStatus FgPresentEngine::Detach(FgTicks now, std::uint64_t* cancel_from) {
    if (cancel_from) *cancel_from = 0;
    switch (state_) {
        case FgPresenterState::Detached: return Refuse(FgStatus::WrongState);
        case FgPresenterState::Lost:
        case FgPresenterState::Failed: break;
        case FgPresenterState::Attached:
            Quiesce(now, cancel_from);
            [[fallthrough]];
        case FgPresenterState::Quiescing:
            if (!Drained()) return Refuse(FgStatus::Busy);
            break;
    }
    ReleaseAll();
    state_ = FgPresenterState::Detached;
    return FgStatus::Ok;
}

// --- Chain registry --------------------------------------------------------------------------------------------------

FgChainRegistry& FgChainRegistry::Process() {
    static FgChainRegistry registry;
    return registry;
}

bool FgChainRegistry::Claim(std::uint64_t chain, const void* owner) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const Entry& e : entries_)
        if (e.chain == chain) return e.owner == owner;
    entries_.push_back(Entry{chain, owner});
    return true;
}

void FgChainRegistry::Release(std::uint64_t chain, const void* owner) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (std::size_t i = 0; i < entries_.size(); ++i) {
        if (entries_[i].chain == chain && entries_[i].owner == owner) {
            entries_.erase(entries_.begin() + static_cast<std::ptrdiff_t>(i));
            return;
        }
    }
}

const void* FgChainRegistry::Owner(std::uint64_t chain) const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const Entry& e : entries_)
        if (e.chain == chain) return e.owner;
    return nullptr;
}

// --- Pacing arithmetic -----------------------------------------------------------------------------------------------

FgStatus FgSnapToRefreshGrid(FgTicks target, const FgRefreshGrid& grid, FgTicks* vblank, FgTicks* submit) {
    if (grid.period <= 0 || !vblank || !submit) return FgStatus::InvalidArgument;
    const FgTicks p = grid.period;
    // offset = target - origin, checked
    FgTicks offset = 0;
    if ((grid.origin < 0 && target > std::numeric_limits<FgTicks>::max() + grid.origin) ||
        (grid.origin > 0 && target < std::numeric_limits<FgTicks>::min() + grid.origin))
        return FgStatus::Overflow;
    offset = target - grid.origin;
    // k = round(offset / p), ties up, floor division for negative offsets
    FgTicks q = offset / p;
    FgTicks r = offset % p;
    if (r < 0) {
        r += p;
        --q;
    }
    if (2 * r >= p) ++q;
    // vblank = origin + q * p, checked
    if (q != 0 && (q > std::numeric_limits<FgTicks>::max() / p || q < std::numeric_limits<FgTicks>::min() / p))
        return FgStatus::Overflow;
    FgTicks v = 0;
    if (FgCheckedAdd(grid.origin, q * p, &v) != FgStatus::Ok) return FgStatus::Overflow;
    *vblank = v;
    *submit = v - p / 2;
    return FgStatus::Ok;
}

std::uint32_t FgRefreshMultiple(FgTicks interval, FgTicks refresh_period, FgTicks tolerance) {
    if (interval <= 0 || refresh_period <= 0 || tolerance < 0) return 0;
    const FgTicks k = (interval + refresh_period / 2) / refresh_period;
    if (k < 1 || k > 0xFFFF) return 0;
    const FgTicks diff = interval - k * refresh_period;
    return (diff < 0 ? -diff : diff) <= tolerance ? static_cast<std::uint32_t>(k) : 0;
}

bool FgPlanFitsRefresh(const FgIntervalPlan& plan, FgTicks refresh_period) {
    if (plan.generated_count == 0) return true;
    if (refresh_period <= 0 || plan.period <= 0) return false;
    const FgTicks interval = plan.period / static_cast<FgTicks>(plan.generated_count + 1);
    return FgRefreshMultiple(interval, refresh_period, refresh_period / 8) != 0;
}

FgTicks FgTimerDue(FgTicks target, FgTicks now, FgTicks lead) {
    const FgTicks wake = target - lead;
    return wake > now ? -(wake - now) : 0;
}

}  // namespace wgfg
