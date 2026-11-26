// SPDX-License-Identifier: MIT
// Frame generation (issue #96, F7a): the FG session. See fg_session.h for the rules.
#include "fg_session.h"

#include <cstring>
#include <utility>

#include "../proxy/fg_contract_bridge.h"
#include "../proxy/fg_proxy_core.h"

// Test builds only (CMakeLists.txt): a deliberately broken session that tests/test_fg_session.cpp must catch.
//  1 shared work recorded again for every index   2 menu latch not held for frame k + 1   3 wrong intermediate time
//  4 silent continuation after an executor refusal   5 history advanced on every index
//  6 a count change resets history
#ifndef WG_FG_SESSION_MUTANT
#define WG_FG_SESSION_MUTANT 0
#endif

namespace wgfg {

const char* FgCopyReasonName(FgCopyReason r) {
    switch (r) {
        case FgCopyReason::MenuDetected: return "menu detected";
        case FgCopyReason::MenuLatched: return "menu latched";
        case FgCopyReason::Priming: return "priming";
    }
    return "unknown";
}

const char* FgSessionErrorName(FgSessionError e) {
    switch (e) {
        case FgSessionError::None: return "none";
        case FgSessionError::BadArgument: return "bad argument";
        case FgSessionError::WrongState: return "wrong state";
        case FgSessionError::CreateInvalid: return "create description invalid";
        case FgSessionError::WorkerFailed: return "create work failed";
        case FgSessionError::ModelInvalid: return "model check refused";
        case FgSessionError::ExecutorMissing: return "no executor";
        case FgSessionError::ExecutorConfigure: return "executor refused configuration";
        case FgSessionError::PresenterSource: return "presenter source refused";
        case FgSessionError::PresenterAttach: return "presenter refused the chain";
        case FgSessionError::PresenterCall: return "presenter call failed";
        case FgSessionError::PresenterBusy: return "presenter busy";
        case FgSessionError::SchedulerRefused: return "scheduler refused";
        case FgSessionError::ExecutorRefused: return "executor refused";
        case FgSessionError::ExecutorDeviceLost: return "executor lost the device";
        case FgSessionError::PlanMismatch: return "plan mismatch";
        case FgSessionError::DeviceLost: return "device lost";
        case FgSessionError::SessionRefused: return "session refused";
        case FgSessionError::ResizeFailed: return "resize failed";
        case FgSessionError::FrameRefused: return "frame refused";
        case FgSessionError::Internal: return "internal failure";
    }
    return "unknown";
}

namespace {

bool SameBits(float a, float b) {
    uint32_t x = 0, y = 0;
    std::memcpy(&x, &a, sizeof(x));
    std::memcpy(&y, &b, sizeof(y));
    return x == y;
}

bool IsModelRefusal(FgRefusal r) {
    const uint32_t v = static_cast<uint32_t>(r);
    return v >= static_cast<uint32_t>(FgRefusal::ModelMissing) && v <= static_cast<uint32_t>(FgRefusal::ModelValue);
}

FgSessionResult Error(FgSessionError e) {
    FgSessionResult r;
    r.error = e;
    return r;
}

}  // namespace

FgSession::~FgSession() {
    if (presenter_) {
        if (presenter_->State() != FgPresenterState::Detached) {
            // a session torn down without Destroy: give the chain back as far as the presenter allows
            if (presenter_->State() == FgPresenterState::Attached) presenter_->Quiesce();
            presenter_->Detach();
        }
    }
    ReleaseAll();
}

void FgSession::ReleaseAll() {
    presenter_.reset();
    scheduler_.reset();
    history_.reset();
    state_ = FgSessionState::None;
    menu_latch_ = false;
    have_frame_ = false;
}

void FgSession::Log(const std::string& line) {
    if (line == last_log_) return;  // one line per transition
    last_log_ = line;
    if (desc_.log) desc_.log(desc_.log_ctx, line.c_str());
}

FgTicks FgSession::Now() const { return desc_.clock ? desc_.clock(desc_.clock_ctx) : 0; }

FgCreateDesc FgSession::HistoryDesc(const FgCreateDesc& create) const {
    FgCreateDesc h = create;
    h.max_generated = kFgMaxGenerated;  // N may change per game frame up to 5 without a new session
    return h;
}

bool FgSession::NeedsRebind(const FgFramePacket& p) const {
    return p.queue_epoch != queue_epoch_ || p.swapchain_epoch != swapchain_epoch_;
}

FgSessionResult FgSession::Finish(const FgSessionResult& r) {
    last_ = r;
    return r;
}

FgSessionResult FgSession::Gate() const {
    switch (state_) {
        case FgSessionState::Active: return FgSessionResult();
        case FgSessionState::None: return Error(FgSessionError::WrongState);
        case FgSessionState::Resizing: return Error(FgSessionError::WrongState);
        case FgSessionState::Lost: return Error(FgSessionError::DeviceLost);
        case FgSessionState::Refused: {
            FgSessionResult r = Error(FgSessionError::SessionRefused);
            return r;
        }
    }
    return Error(FgSessionError::WrongState);
}

FgSessionResult FgSession::Refuse(FgSessionResult r) {
    // Nothing more is submitted. What the presenter already holds (the failing interval's game frame included) still
    // reaches the display; Destroy gives the chain back.
    state_ = FgSessionState::Refused;
    refused_with_ = r.error;
    menu_latch_ = false;
    Log(std::string("fg session: refused (") + FgSessionErrorName(r.error) + "): no fallback, generation stopped");
    return Finish(r);
}

bool FgSession::SetFirstImageBudget(FgTicks budget) {
    if (!scheduler_) return false;
    scheduler_->SetFirstImageBudget(budget);
    desc_.scheduler.first_image_budget = scheduler_->Config().first_image_budget;
    return true;
}

FgSessionResult FgSession::Lose(FgSessionResult r) {
    if (presenter_ && presenter_->State() != FgPresenterState::Detached && presenter_->State() != FgPresenterState::Lost)
        presenter_->OnDeviceLost();
    if (scheduler_) scheduler_->OnDeviceLost();
    if (history_) history_->Invalidate();
    state_ = FgSessionState::Lost;
    menu_latch_ = false;
    Log("fg session: device lost");
    return Finish(r);
}

// ---- lifecycle ---------------------------------------------------------------------------------------------------

FgSessionResult FgSession::Create(const FgSessionDesc& d) {
    if (state_ != FgSessionState::None || presenter_) return Error(FgSessionError::WrongState);
    if (!d.executor) return Finish(Error(FgSessionError::ExecutorMissing));
    if (!d.presenter_source || !d.clock) return Finish(Error(FgSessionError::BadArgument));
    FgSessionResult r;
    r.refusal = FgValidateCreate(d.create);
    if (r.refusal != FgRefusal::None) {
        r.error = FgSessionError::CreateInvalid;
        return Finish(r);
    }
    try {
        // create-time work on the worker: a named callable copied into the work in this frame, never a lambda capture
        FgPendingInit::Work work;
        if (!d.model_path.empty()) work = FgModelJob(d.model_path, d.model_expect);
        else if (d.create_work) work = d.create_work;
        if (work) {
            std::shared_ptr<FgPendingInit> pending = FgPendingInit::Start(std::move(work), FgHold());
            if (!pending) {
                r.error = FgSessionError::WorkerFailed;
                r.refusal = FgRefusal::WorkerStartFailed;
                return Finish(r);
            }
            if (!pending->Wait(d.create_wait_ms)) {
                FgPendingInit::Abandon(pending, nullptr, nullptr);
                r.error = FgSessionError::WorkerFailed;
                r.refusal = FgRefusal::WorkerAbandoned;
                return Finish(r);
            }
            r.refusal = pending->Result();
            if (r.refusal != FgRefusal::None) {
                r.error = IsModelRefusal(r.refusal) ? FgSessionError::ModelInvalid : FgSessionError::WorkerFailed;
                return Finish(r);
            }
        }

        desc_ = d;
        desc_.create_work = nullptr;  // ran once; the session keeps no copy of it
        const uint32_t banks = d.history_banks < FgHistory::kMinBanks   ? FgHistory::kMinBanks
                               : d.history_banks > FgHistory::kMaxBanks ? FgHistory::kMaxBanks
                                                                        : d.history_banks;
        desc_.history_banks = banks;
        r.executor = desc_.executor->Configure(desc_.create, banks);
        if (r.executor != FgExecResult::Ok) {
            r.error = FgSessionError::ExecutorConfigure;
            return Finish(r);
        }
        std::unique_ptr<IFgPresenter> presenter;
        std::string why;
        r.presenter = desc_.presenter_source->CreatePresenter(desc_.create, &presenter, &why);
        if (r.presenter != FgStatus::Ok || !presenter) {
            if (r.presenter == FgStatus::Ok) r.presenter = FgStatus::Unsupported;
            r.error = FgSessionError::PresenterSource;
            Log("fg session: presenter refused: " + why);
            return Finish(r);
        }
        r.presenter = presenter->Attach(desc_.create.swapchain_id, 1);
        if (r.presenter != FgStatus::Ok) {
            r.error = FgSessionError::PresenterAttach;
            return Finish(r);
        }
        std::unique_ptr<FgTargetScheduler> scheduler(new FgTargetScheduler(desc_.scheduler));
        r.presenter = scheduler->BeginEpoch(1);
        if (r.presenter != FgStatus::Ok) {
            presenter->Detach();
            r.error = FgSessionError::SchedulerRefused;
            return Finish(r);
        }
        std::unique_ptr<FgHistory> history(new FgHistory(HistoryDesc(desc_.create), banks));
        presenter_ = std::move(presenter);
        scheduler_ = std::move(scheduler);
        history_ = std::move(history);
    } catch (...) {
        ReleaseAll();
        return Finish(Error(FgSessionError::Internal));
    }
    present_epoch_ = 1;
    queue_epoch_ = desc_.create.queue_epoch;
    swapchain_epoch_ = desc_.create.swapchain_epoch;
    refused_with_ = FgSessionError::None;
    counters_ = FgSessionCounters();
    state_ = FgSessionState::Active;
    Log("fg session: created");
    return Finish(r);
}

FgSessionResult FgSession::BeginResize() {
    FgSessionResult r = Gate();
    if (r.error != FgSessionError::None) return Finish(r);
    r.presenter = presenter_->Quiesce();
    if (r.presenter == FgStatus::DeviceLost) {
        r.error = FgSessionError::DeviceLost;
        return Lose(r);
    }
    if (r.presenter != FgStatus::Ok) {
        r.error = FgSessionError::PresenterCall;
        return Finish(r);
    }
    state_ = FgSessionState::Resizing;
    menu_latch_ = false;
    return Finish(r);
}

FgSessionResult FgSession::CompleteResize(bool resize_ok, const FgCreateDesc& resized) {
    FgSessionResult r;
    if (state_ != FgSessionState::Resizing) {
        r = Gate();
        if (r.error == FgSessionError::None) r.error = FgSessionError::WrongState;
        return Finish(r);
    }
    r.presenter = presenter_->CompleteResize(resize_ok, present_epoch_ + 1);
    if (r.presenter == FgStatus::Busy) {
        r.error = FgSessionError::PresenterBusy;
        return Finish(r);
    }
    if (r.presenter == FgStatus::DeviceLost) {
        r.error = FgSessionError::DeviceLost;
        return Lose(r);
    }
    if (r.presenter == FgStatus::ResizeFailed) {
        r.error = FgSessionError::ResizeFailed;
        return Refuse(r);
    }
    if (r.presenter != FgStatus::Ok) {
        r.error = FgSessionError::PresenterCall;
        return Refuse(r);
    }
    ++present_epoch_;
    r.refusal = FgValidateCreate(resized);
    if (r.refusal != FgRefusal::None) {
        r.error = FgSessionError::CreateInvalid;
        return Refuse(r);
    }
    try {
        r.executor = desc_.executor->Configure(resized, desc_.history_banks);
        if (r.executor != FgExecResult::Ok) {
            r.error = r.executor == FgExecResult::DeviceLost ? FgSessionError::ExecutorDeviceLost
                                                             : FgSessionError::ExecutorConfigure;
            return r.executor == FgExecResult::DeviceLost ? Lose(r) : Refuse(r);
        }
        r.presenter = scheduler_->BeginEpoch(present_epoch_);
        if (r.presenter != FgStatus::Ok) {
            r.error = FgSessionError::SchedulerRefused;
            return Refuse(r);
        }
        // the presenter drained every old-epoch image, so nothing reads the old banks any more
        history_.reset(new FgHistory(HistoryDesc(resized), desc_.history_banks));
    } catch (...) {
        r.error = FgSessionError::Internal;
        return Refuse(r);
    }
    desc_.create = resized;
    queue_epoch_ = resized.queue_epoch;
    swapchain_epoch_ = resized.swapchain_epoch;
    have_frame_ = false;
    ++counters_.resizes;
    state_ = FgSessionState::Active;
    Log("fg session: resized");
    return Finish(r);
}

FgSessionResult FgSession::OnDeviceLost() {
    if (state_ == FgSessionState::None) return Finish(Error(FgSessionError::WrongState));
    FgSessionResult r = Error(FgSessionError::DeviceLost);
    return Lose(r);
}

FgSessionResult FgSession::Destroy() {
    if (state_ == FgSessionState::None && !presenter_) return Finish(Error(FgSessionError::WrongState));
    FgSessionResult r;
    if (presenter_ && presenter_->State() != FgPresenterState::Detached) {
        r.presenter = presenter_->Detach();
        if (r.presenter == FgStatus::Busy) {
            r.error = FgSessionError::PresenterBusy;
            return Finish(r);
        }
    }
    ReleaseAll();
    refused_with_ = FgSessionError::None;
    Log("fg session: destroyed");
    return Finish(r);
}

// ---- per frame ---------------------------------------------------------------------------------------------------

FgSessionResult FgSession::PresentRealOnlyPlan(const FgIntervalPlan& plan, FgSessionResult r,
                                               const FgTextureRef* game_colour) {
    FgRealDispatchDesc rd;
    rd.frame_token = plan.real_frame_id;
    rd.interval_id = plan.interval_id;
    rd.real_only = true;
    rd.target = plan.targets[0];
    if (game_colour) rd.game_colour = *game_colour;
    FgReadyFence ready;
    r.executor = desc_.executor->RecordRealEndpoint(rd, &ready);
    if (r.executor == FgExecResult::DeviceLost) {
        r.error = FgSessionError::ExecutorDeviceLost;
        return Lose(r);
    }
    if (r.executor != FgExecResult::Ok) {
        r.error = FgSessionError::ExecutorRefused;
        return Refuse(r);
    }
    ++counters_.real_endpoints;
    FgRealImage image;
    image.epoch = plan.epoch;
    image.frame_id = plan.real_frame_id;
    image.ready = ready;
    r.presenter = presenter_->PresentRealOnly(plan, image);
    if (r.presenter == FgStatus::DeviceLost) {
        r.error = FgSessionError::DeviceLost;
        return Lose(r);
    }
    if (r.presenter != FgStatus::Ok) {
        r.error = FgSessionError::PresenterCall;
        return Finish(r);
    }
    ++counters_.real_only_intervals;
    ++counters_.game_frames;
    if (r.error == FgSessionError::None) r.frame = FgFrameStatus::RealOnly;  // a refused frame stays Refused
    r.plan_real_only = plan.real_only_reason;
    r.interval_id = plan.interval_id;
    return Finish(r);
}

FgSessionResult FgSession::RunInterval(const FgFramePacket& packet, const FgIngestResult& ingest,
                                       const FgIntervalPlan& plan, bool copy_output, FgCopyReason copy_reason,
                                       FgSessionResult r, bool pre_failed) {
    const uint32_t n = plan.generated_count;
    r.interval_id = plan.interval_id;
    r.scheduled = n;
    FgIntervalReservation res;
    r.presenter = presenter_->BeginInterval(plan, &res);
    if (r.presenter == FgStatus::NoSlot) {
        const FgTicks until = Now() + 1000000;  // 100 ms in 100 ns ticks
        while (r.presenter == FgStatus::NoSlot && Now() < until) {
            r.presenter = presenter_->BeginInterval(plan, &res);
        }
    }
    if (r.presenter == FgStatus::DeviceLost) {
        r.error = FgSessionError::DeviceLost;
        return Lose(r);
    }
    if (r.presenter != FgStatus::Ok) {
        r.error = FgSessionError::PresenterCall;
        return Finish(r);
    }
    if (res.count != n + 1 || res.interval_id != plan.interval_id || !res.slots[n].real || !res.slots[n].reserved) {
        r.error = FgSessionError::PlanMismatch;
        presenter_->Quiesce();  // the reservation cannot be submitted without its real slot: drop it
        return Refuse(r);
    }
    FgPresetGraph graph = {};
    FgPresetGraphOf(packet.preset, &graph);
    const bool generated_history = ingest.status == FgFrameStatus::Generated;

    FgIntervalSubmission sub;
    sub.epoch = res.epoch;
    sub.interval_id = res.interval_id;
    sub.count = res.count;
    bool failed = pre_failed;  // the shared work already failed: only the game frame is submitted
    bool stale = false;
    // An executor that records several indices more cheaply together (BatchesGenerated) gets the consecutive
    // generated indices of the interval in one call; every other path records them one by one as always. The
    // checks of each index (plan, history, slot, deadline) still run in order before it is queued; the queue is
    // recorded before anything that is not a generated index, before the history advance and at the end.
#if WG_FG_SESSION_MUTANT == 0
    const bool batching = !failed && desc_.executor->BatchesGenerated();
#else
    const bool batching = false;
#endif
    FgIndexDispatchDesc pending[kFgMaxGenerated];
    uint32_t pending_n = 0;
    const auto flush = [&]() {
        if (pending_n == 0) return true;
        uint32_t done = 0;
        const uint32_t first = pending[0].index;
        r.executor = desc_.executor->RecordGeneratedBatch(pending, pending_n, &sub.ready[first - 1], &done);
        for (uint32_t k = 0; k < done; ++k) {
            sub.produced[first - 1 + k] = true;
            ++r.generated_recorded;
            ++counters_.index_records;
        }
        pending_n = 0;
        return r.executor == FgExecResult::Ok;
    };
    for (uint32_t i = 1; i <= n && !failed; ++i) {
        const FgFrameTarget& tg = plan.targets[i - 1];
        const FgPresentSlot& slot = res.slots[i - 1];
        const bool last = FgLastIndex(i, n);
        const float t = FgIndexTime(i, n);
        if (tg.index != i || tg.real || !SameBits(t, tg.t) || !SameBits(t, FgIntermediateTime(i, n))) {
            r.error = FgSessionError::PlanMismatch;
            failed = true;
            break;
        }
        if (generated_history && !copy_output && n == ingest.generated_count) {
            // the ingest side agrees on the interval, the index and its time
            FgFramePacket idx = packet;
            idx.generated_index = i;
            const FgIngestResult ev = history_->Evaluate(idx);
            if (ev.status != FgFrameStatus::Generated || !SameBits(ev.t, t)) {
                r.error = FgSessionError::PlanMismatch;
                r.refusal = ev.refusal;
                failed = true;
                break;
            }
        }
        if (!slot.reserved) {
            if (!flush()) {
                failed = true;
                break;
            }
            ++r.not_reserved;
        } else if (stale || Now() > tg.deadline) {
            // too late to be shown: this index and every later one of the interval are cancelled
            if (!flush()) {
                failed = true;
                break;
            }
            stale = true;
            ++r.stale_cancelled;
        } else if (copy_output || !generated_history) {
            if (!flush()) {
                failed = true;
                break;
            }
            FgCopyDispatchDesc cd;
            cd.frame_token = packet.frame_token;
            cd.interval_id = plan.interval_id;
            cd.index = i;
            cd.generated_count = n;
            cd.is_last_index = last;
            cd.reason = generated_history ? copy_reason : FgCopyReason::Priming;
            cd.slot = slot;
            cd.target = tg;
            r.executor = desc_.executor->RecordMenuCopy(cd, &sub.ready[i - 1]);
            if (r.executor != FgExecResult::Ok) {
                failed = true;
                break;
            }
            sub.produced[i - 1] = true;
            ++r.copies_recorded;
            ++counters_.copy_records;
        } else {
#if WG_FG_SESSION_MUTANT == 1
            {
                FgSharedDispatchDesc again;
                again.frame_token = packet.frame_token;
                again.packet = &packet;
                FgSharedResult ignored;
                desc_.executor->RecordSharedGameFrame(again, &ignored);
            }
#endif
            FgIndexDispatchDesc id;
            id.frame_token = packet.frame_token;
            id.interval_id = plan.interval_id;
            id.history_epoch = ingest.epoch;
            id.index = i;
            id.generated_count = n;
#if WG_FG_SESSION_MUTANT == 3
            id.t = static_cast<float>(i) / static_cast<float>(n);
#else
            id.t = t;
#endif
            id.is_last_index = last;
            id.preset = packet.preset;
            id.graph = graph;
            id.previous_bank = ingest.previous_bank;
            id.current_bank = ingest.current_bank;
            id.slot = slot;
            id.target = tg;
            if (batching) {
                pending[pending_n++] = id;
            } else {
                r.executor = desc_.executor->RecordGeneratedIndex(id, &sub.ready[i - 1]);
#if WG_FG_SESSION_MUTANT == 4
                if (r.executor == FgExecResult::Refused) {
                    r.executor = FgExecResult::Ok;
                    continue;
                }
#endif
                if (r.executor != FgExecResult::Ok) {
                    failed = true;
                    break;
                }
                sub.produced[i - 1] = true;
                ++r.generated_recorded;
                ++counters_.index_records;
            }
        }
#if WG_FG_SESSION_MUTANT == 5
        if (generated_history) {
#else
        if (last && generated_history) {
#endif
            // the interval's last index is handled: history reads are fenced and the history advance is counted
            if (!flush()) {
                failed = true;
                break;
            }
            history_->MarkRead(desc_.executor->LastComputeValue());
            r.history_advanced = true;
            ++counters_.history_advances;
        }
    }
    if (pending_n != 0 && !flush()) failed = true;  // a break above (plan mismatch) left indices queued
    if (failed && r.executor == FgExecResult::DeviceLost) {
        r.error = FgSessionError::ExecutorDeviceLost;
        return Lose(r);
    }
    if (failed && r.error == FgSessionError::None) r.error = FgSessionError::ExecutorRefused;
    for (uint32_t i = 0; failed && i < n; ++i) sub.produced[i] = false;  // nothing generated after a failure

    // the game frame of the interval is always recorded and submitted when the executor can record it
    FgRealDispatchDesc rd;
    rd.frame_token = packet.frame_token;
    rd.interval_id = plan.interval_id;
    rd.real_only = false;
    rd.slot = res.slots[n];
    rd.target = plan.targets[n];
    rd.game_colour = packet.color;
    const FgExecResult real = desc_.executor->RecordRealEndpoint(rd, &sub.ready[n]);
    if (real == FgExecResult::DeviceLost) {
        r.executor = real;
        r.error = FgSessionError::ExecutorDeviceLost;
        return Lose(r);
    }
    if (real != FgExecResult::Ok) {
        r.executor = real;
        if (r.error == FgSessionError::None) r.error = FgSessionError::ExecutorRefused;
        presenter_->Quiesce();  // drops the open reservation: nothing of this interval is shown by the session
        return Refuse(r);
    }
    sub.produced[n] = true;
    ++counters_.real_endpoints;
    r.presenter = presenter_->SubmitInterval(sub);
    if (r.presenter == FgStatus::DeviceLost) {
        r.error = FgSessionError::DeviceLost;
        return Lose(r);
    }
    if (r.presenter != FgStatus::Ok) {
        if (r.error == FgSessionError::None) r.error = FgSessionError::PresenterCall;
        presenter_->Quiesce();
        return Refuse(r);
    }
    ++counters_.game_frames;
    ++counters_.generated_intervals;
    counters_.stale_cancelled += r.stale_cancelled;
    if (failed) return Refuse(r);  // the game frame was submitted; generation stops here, no fallback
    r.frame = (generated_history && !copy_output) ? FgFrameStatus::Generated : FgFrameStatus::RealOnly;
    if (r.menu_copy) ++counters_.menu_copy_frames;
    return Finish(r);
}

FgSessionResult FgSession::OnPresent(const FgFramePacket& packet) {
    FgSessionResult r = Gate();
    r.frame_token = packet.frame_token;
    r.preset = packet.preset;
    r.generated_count = packet.generated_count;
    if (r.error != FgSessionError::None) return Finish(r);
    if (packet.generated_index != 0) {
        r.error = FgSessionError::BadArgument;
        r.refusal = FgRefusal::BadGeneratedIndex;
        return Finish(r);
    }
    try {
        // identity follows the adapter core's latched epochs (it relatches them on a chain / queue replacement)
        if (NeedsRebind(packet)) {
            history_->Rebind(packet.queue_epoch, packet.swapchain_epoch);
            queue_epoch_ = packet.queue_epoch;
            swapchain_epoch_ = packet.swapchain_epoch;
        }
        if (have_frame_ && packet.preset != last_preset_) ++counters_.preset_changes;
        if (have_frame_ && packet.generated_count != last_count_) {
            ++counters_.count_changes;
#if WG_FG_SESSION_MUTANT == 6
            history_->Invalidate();
#endif
        }
        have_frame_ = true;
        last_preset_ = packet.preset;
        last_count_ = packet.generated_count;

        const FgTicks now = Now();
        const FgIngestResult ingest = history_->Accept(packet, desc_.executor->CompletedComputeValue());
        r.history_real_only = ingest.real_only;
        r.frame = ingest.status;

        FgGameFrame gf;
        gf.epoch = present_epoch_;
        gf.frame_id = packet.frame_token;
        gf.arrival = now;
        gf.requested_count = packet.generated_count;
        gf.generation_enabled = ingest.status != FgFrameStatus::Refused && ingest.current_bank >= 0;
        gf.reset = (packet.flags & kFgFlagReset) != 0;
        if (ingest.status == FgFrameStatus::Refused) {
            ++counters_.frame_refusals;
            r.error = FgSessionError::FrameRefused;
            r.refusal = ingest.refusal;
            Log(std::string("fg session: frame refused: ") + FgRefusalName(ingest.refusal));
        }
        FgIntervalPlan plan;
        r.presenter = scheduler_->Commit(gf, &plan);
        if (r.presenter == FgStatus::DeviceLost) {
            r.error = FgSessionError::DeviceLost;
            return Lose(r);
        }
        if (r.presenter != FgStatus::Ok) {
            r.error = FgSessionError::SchedulerRefused;
            return Finish(r);
        }

        // shared work once per game frame history accepted into a bank
        const bool accepted = ingest.status != FgFrameStatus::Refused && ingest.current_bank >= 0;
        bool copy_output = false;
        FgCopyReason copy_reason = FgCopyReason::MenuDetected;
        if (accepted) {
            FgSharedDispatchDesc sd;
            sd.frame_token = packet.frame_token;
            sd.interval_id = plan.generated_count > 0 ? plan.interval_id : 0;
            sd.history_epoch = ingest.epoch;
            sd.has_previous = ingest.status == FgFrameStatus::Generated;
            sd.previous_bank = ingest.previous_bank;
            sd.current_bank = ingest.current_bank;
            sd.preset = packet.preset;
            FgPresetGraphOf(packet.preset, &sd.graph);
            sd.generated_count = packet.generated_count;
            sd.generates = plan.generated_count > 0;
            sd.packet = &packet;
            FgSharedResult shared;
            shared.menu.frame_token = packet.frame_token;
            r.executor = desc_.executor->RecordSharedGameFrame(sd, &shared);
            if (r.executor == FgExecResult::DeviceLost) {
                r.error = FgSessionError::ExecutorDeviceLost;
                return Lose(r);
            }
            if (r.executor != FgExecResult::Ok) {
                r.error = FgSessionError::ExecutorRefused;
                if (plan.generated_count == 0) {
                    // nothing reserved: show the game frame if the executor can still record it, then refuse
                    FgSessionResult shown = PresentRealOnlyPlan(plan, r, packet.color.resource ? &packet.color : nullptr);
                    if (state_ != FgSessionState::Active) return shown;
                    return Refuse(shown);
                }
                // reserve, submit the game frame alone, then refuse
                return RunInterval(packet, ingest, plan, false, FgCopyReason::MenuDetected, r, true);
            }
            r.shared_recorded = true;
            ++counters_.shared_records;
            r.menu_detected = shared.menu.detected;
            if (shared.menu.detected) ++counters_.menu_detections;
            copy_output = shared.menu.detected || menu_latch_;
            copy_reason = shared.menu.detected ? FgCopyReason::MenuDetected : FgCopyReason::MenuLatched;
#if WG_FG_SESSION_MUTANT == 2
            menu_latch_ = false;
#else
            menu_latch_ = shared.menu.detected;  // frame k's decision also covers frame k + 1
#endif
            if (desc_.on_menu) desc_.on_menu(desc_.menu_ctx, shared.menu, copy_output);
        } else {
            menu_latch_ = false;  // a frame shown as itself consumes the latch
        }
        r.menu_copy = copy_output && plan.generated_count > 0;

        if (plan.generated_count == 0) {
            if (accepted && ingest.status == FgFrameStatus::Generated)
                history_->MarkRead(desc_.executor->LastComputeValue());
            return PresentRealOnlyPlan(plan, r, packet.color.resource ? &packet.color : nullptr);
        }
        return RunInterval(packet, ingest, plan, copy_output, copy_reason, r, false);
    } catch (...) {
        r.error = FgSessionError::Internal;
        return Refuse(r);
    }
}

FgSessionResult FgSession::OnGameFrameOnly(uint64_t frame_token, uint32_t adapter_real_only,
                                           uint32_t adapter_reason, const FgTextureRef* game_colour) {
    FgSessionResult r = Gate();
    r.frame_token = frame_token;
    if (r.error != FgSessionError::None) return Finish(r);
    (void)adapter_reason;  // reported by the adapter core itself; the plan carries the presenter-side reason
    menu_latch_ = false;   // the game frame is shown as itself: the latch is consumed
    FgGameFrame gf;
    gf.epoch = present_epoch_;
    gf.frame_id = frame_token;
    gf.arrival = Now();
    gf.requested_count = adapter_real_only == WG_FG_REAL_ONLY_DISABLED ? 0u : last_count_;
    gf.generation_enabled = false;
    gf.reset = adapter_real_only == WG_FG_REAL_ONLY_RESET;
    FgIntervalPlan plan;
    r.presenter = scheduler_->Commit(gf, &plan);
    if (r.presenter == FgStatus::DeviceLost) {
        r.error = FgSessionError::DeviceLost;
        return Lose(r);
    }
    if (r.presenter != FgStatus::Ok) {
        r.error = FgSessionError::SchedulerRefused;
        return Finish(r);
    }
    if (plan.generated_count != 0) {  // a scheduler that generates with generation disabled is broken
        r.error = FgSessionError::PlanMismatch;
        return Refuse(r);
    }
    r.frame = FgFrameStatus::RealOnly;
    r.history_real_only =
        static_cast<FgFrameRealOnly>(adapter_real_only <= WG_FG_REAL_ONLY_DISABLED ? adapter_real_only : 0u);
    return PresentRealOnlyPlan(plan, r, game_colour);
}

// ---- adapter hand-off --------------------------------------------------------------------------------------------

int32_t FgSession::SubmitThunk(void* self, const WgFgPacket* packet, const WgFgSessionDesc* session) {
    if (!self || !packet || !session) return 1;
    try {
        FgSession* s = static_cast<FgSession*>(self);
        const FgFramePacket p = wgfgp::ToRuntime(*packet);
        const FgSessionResult r = s->OnPresent(p);
        // 0: the session presented the game frame (generated, copied or real-only); otherwise the core shows it
        return (r.error == FgSessionError::None || r.error == FgSessionError::FrameRefused) ? 0 : 1;
    } catch (...) {
        return 1;
    }
}

WgFgExecutorV1 FgSession::ExecutorHandOff() {
    WgFgExecutorV1 t = {};
    t.struct_size = sizeof(t);
    t.abi_version = WG_FG_ADAPTER_ABI_VERSION;
    t.self = this;
    t.submit = &FgSession::SubmitThunk;
    return t;
}

int32_t FgSessionPresent(wgfgp::FgProxy* proxy, FgSession* session, const WgFgPresentDesc* desc,
                         WgFgPresentResult* out, FgSessionResult* session_out) {
    WgFgPresentResult local = {};
    WgFgPresentResult* res = out ? out : &local;
    res->struct_size = sizeof(*res);
    res->abi_version = WG_FG_ADAPTER_ABI_VERSION;
    if (session_out) *session_out = FgSessionResult();
    if (!proxy) return -1;
    const int32_t rc = proxy->present(desc, res);
    if (!session || !desc) return rc;
    const bool session_saw_it = session->LastResult().frame_token == desc->frame_token &&
                                res->reason == WG_FG_REASON_EXECUTOR_REFUSED;
    if (res->status == WG_FG_PRESENT_GENERATE || (res->status == WG_FG_PRESENT_REAL_ONLY && session_saw_it)) {
        // the session handled this frame itself (and, on a failure, already did what it could with the game frame)
        if (session_out) *session_out = session->LastResult();
    } else if (res->status == WG_FG_PRESENT_REAL_ONLY) {
        // the game's back buffer goes along: a presenter that shows slot images needs the game frame copied into one
        const FgTextureRef bb = wgfgp::ToRuntime(desc->backbuffer);
        const FgSessionResult r =
            session->OnGameFrameOnly(desc->frame_token, res->real_only, res->reason, bb.resource ? &bb : nullptr);
        if (session_out) *session_out = r;
    }
    return rc;
}

}  // namespace wgfg
