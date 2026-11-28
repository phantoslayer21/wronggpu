// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// Frame generation (issue #96, F7a): the FG session, which joins the game adapter core (clean/fg/proxy), frame ingest
// and history (fg_history.h), the multi-frame kernel executor seam (IFgKernelExecutor below) and the presentation
// boundary (clean/fg/presenter) into one per-swap-chain pipeline:
//
//   game -> FgProxy (tags, constants, present) -> WgFgExecutorV1 hand-off -> FgSession::OnPresent
//        -> FgHistory::Accept -> FgTargetScheduler::Commit -> IFgPresenter::BeginInterval
//        -> IFgKernelExecutor: RecordSharedGameFrame once, then RecordGeneratedIndex for i = 1..N in order (or
//           RecordMenuCopy while a menu is latched / history primes), then RecordRealEndpoint
//        -> IFgPresenter::SubmitInterval (one ready fence per reserved slot)
//
// Per game frame:
//  - preset (A / B / C) and N (1..5) come from the frame's packet and are frozen for every index of that frame; a new
//    preset or count takes effect at the next game frame. A count change never resets history (FgHistory keeps its
//    banks; the scheduler latches the new count at the game-frame boundary).
//  - RecordSharedGameFrame runs exactly once for every game frame history accepted into a bank (a generated interval
//    or a priming frame), whether or not the interval generates (a real-only interval keeps the next interval's
//    inputs ready). It returns the menu decision of that frame.
//  - RecordGeneratedIndex runs once per reserved index i = 1..N, in order, with t = float(i) / (float(N) + 1) (one
//    single-precision division, checked against the scheduler's plan), is_last_index = (i == N), the frame's preset
//    graph and the presenter slot the image is written into.
//  - MENU LATCH: if the menu decision of frame k says "menu", frames k and k + 1 output an exact copy of their game
//    frame in every generated slot (RecordMenuCopy); a further detection extends the latch by one frame. History keeps
//    rotating with the game frames, so nothing is re-primed when the menu goes away.
//  - The interval's history reads are marked (FgHistory::MarkRead) and the history advance is counted only when the
//    last index of the interval has been handled.
//  - A generated index whose deadline has already passed when its turn comes cancels itself and every later index of
//    the interval (a stale suffix); the game frame of the interval is always recorded and submitted.
//  - Real-only intervals (scheduler priming, variable-refresh window exit, count 0, adapter-side real-only reasons)
//    present only the game frame, at its own target, on the same presenter: no stutter, no generated work.
//
// Refusal: every failure is an explicit FgSessionError with its detail (runtime refusal, presenter status, executor
// result). A failed executor step (or a broken plan) puts the session into the sticky Refused state: the current game
// frame is still submitted when the executor can record it (and reaches the display), nothing more is submitted, and
// every later call answers SessionRefused until Destroy releases the chain. Nothing ever switches to another
// executor, presenter or route.
//
// Single caller: the present owner serialises every call (the adapter core calls the hand-off under its own lock).
// Plain C++17; no Windows or vendor headers (the Windows presenter source is fg_session_win.h). No lambda captures a
// copy that can throw (MSVC /Zc:lambda): the create work is a named callable and the hand-off is a C trampoline.
#include <cstdint>
#include <memory>
#include <string>

#include "../presenter/fg_presenter.h"
#include "../presenter/fg_target_scheduler.h"
#include "../proxy/fg_adapter_abi.h"
#include "fg_contract.h"
#include "fg_history.h"
#include "fg_model.h"
#include "fg_worker.h"

namespace wgfg {

// ---- executor seam ---------------------------------------------------------------------------------------------------

enum class FgExecResult : uint32_t {
    Ok = 0,
    Refused = 1,     // the executor cannot record this work (pipeline missing, arena too small, bad argument, ...)
    DeviceLost = 2,  // the device was removed while recording
};

// Why a generated slot receives an exact copy of the game frame instead of a generated image.
enum class FgCopyReason : uint32_t {
    MenuDetected = 0,  // the menu decision of this frame said "menu"
    MenuLatched = 1,   // the previous frame's menu decision (the one-frame latch)
    Priming = 2,       // history has no previous frame for this interval (start, reset, gap, size change, epoch)
};
const char* FgCopyReasonName(FgCopyReason r);

// The menu decision of one game frame (the executor's menu detection on that frame's inputs).
struct FgMenuDecision {
    uint64_t frame_token = 0;
    bool detected = false;
};

struct FgSharedDispatchDesc {
    uint64_t frame_token = 0;
    uint64_t interval_id = 0;  // the presenter interval this frame ends (0: real-only interval)
    uint64_t history_epoch = 0;
    bool has_previous = false;  // false: a priming frame (no previous frame to pair with)
    int previous_bank = -1, current_bank = -1;
    FgPreset preset = FgPreset::A;
    FgPresetGraph graph = {};
    uint32_t generated_count = 0;  // N scheduled for this frame after display and latency caps
    bool generates = false;        // this frame's interval schedules generated images
    const FgFramePacket* packet = nullptr;  // valid for the duration of the call
};

struct FgSharedResult {
    FgMenuDecision menu;
};

struct FgIndexDispatchDesc {
    uint64_t frame_token = 0;
    uint64_t interval_id = 0;
    uint64_t history_epoch = 0;
    uint32_t index = 0;            // i = 1..N
    uint32_t generated_count = 0;  // N
    float t = 0.0f;                // float(i) / (float(N) + 1)
    bool is_last_index = false;    // i == N
    FgPreset preset = FgPreset::A;
    FgPresetGraph graph = {};
    int previous_bank = -1, current_bank = -1;
    FgPresentSlot slot;            // the presenter's reserved slot this image is written into
    FgFrameTarget target;          // its requested display time and deadline
};

struct FgCopyDispatchDesc {
    uint64_t frame_token = 0;
    uint64_t interval_id = 0;
    uint32_t index = 0;
    uint32_t generated_count = 0;
    bool is_last_index = false;
    FgCopyReason reason = FgCopyReason::MenuDetected;
    FgPresentSlot slot;
    FgFrameTarget target;
};

struct FgRealDispatchDesc {
    uint64_t frame_token = 0;
    uint64_t interval_id = 0;
    bool real_only = false;  // true: no slot; the fence says when the game frame itself is ready to present
    FgPresentSlot slot;      // !real_only: the real endpoint's reserved slot (the game frame is copied into it)
    FgFrameTarget target;
    // real_only without shared work for this frame (an adapter-side real-only interval): the game frame itself (the
    // back buffer the game presents), so an executor whose presenter shows slot images can copy it into a reserved
    // slot. resource is null when the caller has none.
    FgTextureRef game_colour;
};

// The GPU side of one session. Every Record* call records work on the executor's queue and, where it produces an
// image, returns the fence value at which that image is complete. A call that returns anything but Ok recorded
// nothing the session relies on.
class IFgKernelExecutor {
public:
    virtual ~IFgKernelExecutor() = default;
    // Session start / size change (arenas, pipelines, history banks). Refused: the session refuses.
    virtual FgExecResult Configure(const FgCreateDesc& desc, uint32_t history_banks) = 0;
    // Once per accepted game frame: menu detection, downsample / UI, motion preparation, previous-to-current scatter /
    // push-pull and net1 (the parts every generated index shares).
    virtual FgExecResult RecordSharedGameFrame(const FgSharedDispatchDesc& desc, FgSharedResult* out) = 0;
    // Once per generated index i = 1..N (preset A / B full blend, C Lite blend), into desc.slot.
    virtual FgExecResult RecordGeneratedIndex(const FgIndexDispatchDesc& desc, FgReadyFence* ready) = 0;
    // Several consecutive generated indices of one game frame (n >= 1, ascending index, the same frame, preset and
    // count) recorded together: an executor may interleave their layers so that the barriers sit only between
    // dependent layers. ready[k] is the fence of descs[k]. *done is the number of indices recorded when it returns
    // (all of them on Ok); the default records them one by one. A session calls it only when BatchesGenerated().
    virtual FgExecResult RecordGeneratedBatch(const FgIndexDispatchDesc* descs, uint32_t n, FgReadyFence* ready,
                                              uint32_t* done) {
        uint32_t i = 0;
        FgExecResult r = FgExecResult::Ok;
        for (; i < n && r == FgExecResult::Ok; ++i) r = RecordGeneratedIndex(descs[i], &ready[i]);
        if (done) *done = r == FgExecResult::Ok ? i : i - 1;
        return r;
    }
    // True when RecordGeneratedBatch records several indices more cheaply than one by one.
    virtual bool BatchesGenerated() const { return false; }
    // An exact copy of the game frame into a generated slot (menu latch, priming).
    virtual FgExecResult RecordMenuCopy(const FgCopyDispatchDesc& desc, FgReadyFence* ready) = 0;
    // The game frame of the interval: copied into its reserved slot, or (real-only) its readiness on the queue.
    virtual FgExecResult RecordRealEndpoint(const FgRealDispatchDesc& desc, FgReadyFence* ready) = 0;
    // The compute timeline that reads history banks: the last value recorded and the highest value completed.
    virtual uint64_t LastComputeValue() const = 0;
    virtual uint64_t CompletedComputeValue() const = 0;
};

// Where the session's presenter comes from (the fake in tests, fg_session_win.h on Windows). It creates exactly the
// back end the description asks for or refuses with a reason; it never substitutes another one.
class IFgPresenterSource {
public:
    virtual ~IFgPresenterSource() = default;
    virtual FgStatus CreatePresenter(const FgCreateDesc& desc, std::unique_ptr<IFgPresenter>* out,
                                     std::string* why) = 0;
    virtual void ReleasePresenter() {}
};

// ---- session ---------------------------------------------------------------------------------------------------------

enum class FgSessionState : uint32_t {
    None = 0,     // not created (or destroyed)
    Active = 1,   // accepting presents
    Resizing = 2, // BeginResize ran: presents are refused until CompleteResize
    Lost = 3,     // the device is gone: only Destroy
    Refused = 4,  // a hard failure: only Destroy; every call answers the stored reason
};

enum class FgSessionError : uint32_t {
    None = 0,
    BadArgument = 1,           // null / malformed call arguments
    WrongState = 2,            // the call is not allowed in this state
    CreateInvalid = 3,         // the create description fails FgValidateCreate (refusal says why)
    WorkerFailed = 4,          // the create work could not run / threw / did not finish in time (refusal says why)
    ModelInvalid = 5,          // the model check refused (refusal says why)
    ExecutorMissing = 6,
    ExecutorConfigure = 7,     // the executor refused Configure
    PresenterSource = 8,       // the presenter source refused (presenter status says why)
    PresenterAttach = 9,       // the presenter refused to claim the chain
    PresenterCall = 10,        // a presenter call failed (presenter status says why)
    PresenterBusy = 11,        // old work still in flight: pump and retry (resize / destroy)
    SchedulerRefused = 12,     // the scheduler refused the frame (presenter status says why)
    ExecutorRefused = 13,      // an executor Record* call refused
    ExecutorDeviceLost = 14,
    PlanMismatch = 15,         // the plan's intermediate time / order disagrees with the contract
    DeviceLost = 16,
    SessionRefused = 17,       // the session is in the Refused state (refused_with says the original error)
    ResizeFailed = 18,         // the forwarded swap-chain resize failed
    FrameRefused = 19,         // history refused this frame's packet (refusal says why); its game frame was shown
    Internal = 20,             // an allocation failed or an exception reached the session boundary
};
const char* FgSessionErrorName(FgSessionError e);

using FgSessionClockFn = FgTicks (*)(void* ctx);
using FgSessionLogFn = void (*)(void* ctx, const char* line);
using FgSessionMenuFn = void (*)(void* ctx, const FgMenuDecision& decision, bool copy_output);

struct FgSessionDesc {
    FgCreateDesc create;                   // the runtime contract's session description (fg_contract.h)
    IFgKernelExecutor* executor = nullptr;
    IFgPresenterSource* presenter_source = nullptr;
    FgSchedulerConfig scheduler;
    uint32_t history_banks = 3;            // 2..4
    // Create-time work on the worker thread (fg_worker.h): the model check when model_path is set, else `create_work`
    // when set (a named callable, never a capturing lambda), else nothing.
    std::string model_path;
    FgModelExpectation model_expect = FgShippedModelExpectation();
    FgPendingInit::Work create_work;
    uint32_t create_wait_ms = 5000;
    FgSessionClockFn clock = nullptr;      // interrupt-time ticks; required
    void* clock_ctx = nullptr;
    FgSessionLogFn log = nullptr;          // one line per transition
    void* log_ctx = nullptr;
    void (*on_plan)(void*, const FgIntervalPlan&) = nullptr;
    void* plan_ctx = nullptr;
    FgSessionMenuFn on_menu = nullptr;     // every menu decision (and whether the frame's output is a copy)
    void* menu_ctx = nullptr;
};

// What one call did.
struct FgSessionResult {
    FgSessionError error = FgSessionError::None;
    FgRefusal refusal = FgRefusal::None;          // runtime detail
    FgStatus presenter = FgStatus::Ok;            // presenter / scheduler detail
    FgExecResult executor = FgExecResult::Ok;
    FgFrameStatus frame = FgFrameStatus::Refused;  // Generated / RealOnly / Refused
    FgFrameRealOnly history_real_only = FgFrameRealOnly::None;
    FgRealOnlyReason plan_real_only = FgRealOnlyReason::None;
    uint64_t frame_token = 0;
    uint64_t interval_id = 0;
    FgPreset preset = FgPreset::A;
    uint32_t generated_count = 0;     // N frozen for the frame
    uint32_t scheduled = 0;           // generated indices the plan scheduled
    uint32_t generated_recorded = 0;  // RecordGeneratedIndex calls
    uint32_t copies_recorded = 0;     // RecordMenuCopy calls
    uint32_t stale_cancelled = 0;     // indices cancelled as a stale suffix
    uint32_t not_reserved = 0;        // indices the scheduler / presenter did not admit
    bool shared_recorded = false;
    bool menu_detected = false;
    bool menu_copy = false;
    bool history_advanced = false;
    FgTicks fill_period = 0;
    FgTicks fill_release_wait = 0;
};

struct FgSessionCounters {
    uint64_t game_frames = 0;           // OnPresent + OnGameFrameOnly calls that presented a game frame
    uint64_t shared_records = 0;
    uint64_t index_records = 0;
    uint64_t copy_records = 0;
    uint64_t real_endpoints = 0;
    uint64_t generated_intervals = 0;   // intervals submitted with at least one generated index scheduled
    uint64_t real_only_intervals = 0;
    uint64_t history_advances = 0;      // intervals whose last index was handled
    uint64_t stale_cancelled = 0;
    uint64_t menu_detections = 0;
    uint64_t menu_copy_frames = 0;
    uint64_t frame_refusals = 0;
    uint64_t resizes = 0;
    uint64_t preset_changes = 0;
    uint64_t count_changes = 0;
};

class FgSession {
public:
    FgSession() = default;
    ~FgSession();
    FgSession(const FgSession&) = delete;
    FgSession& operator=(const FgSession&) = delete;

    // Validates the description, runs the create work on the worker (bounded wait), configures the executor, creates
    // the presenter through the source and attaches it to the chain. Any failure leaves the session in None with
    // nothing held.
    FgSessionResult Create(const FgSessionDesc& desc);

    // A game frame from the adapter core (generated_index 0). The game frame is presented through the session's
    // presenter (generated interval or real-only).
    FgSessionResult OnPresent(const FgFramePacket& packet);
    // A game frame the adapter core shows without generation (its real-only reasons: menu flag, FG off, missing or
    // mismatched inputs, epoch change, executor refused). The session presents it real-only on its presenter;
    // game_colour (optional) is the game's back buffer, which the executor copies into a reserved slot for presenters
    // that show slot images (FgRealDispatchDesc::game_colour).
    FgSessionResult OnGameFrameOnly(uint64_t frame_token, uint32_t adapter_real_only, uint32_t adapter_reason,
                                    const FgTextureRef* game_colour = nullptr);

    // Resize in two steps: BeginResize stops admission (pending presents are cancelled, leases stay held);
    // CompleteResize passes the forwarded swap-chain resize result through. PresenterBusy while old work is in flight
    // (call again later); ResizeFailed puts the session into Refused.
    FgSessionResult BeginResize();
    FgSessionResult CompleteResize(bool resize_ok, const FgCreateDesc& resized);

    // Device removed: drops all work at once; every later call but Destroy answers DeviceLost.
    FgSessionResult OnDeviceLost();
    // Releases the chain. PresenterBusy while old work is in flight on a healthy device (call again later).
    FgSessionResult Destroy();

    // The adapter core's executor hand-off (fg_adapter_abi.h): submit() converts the packet and calls OnPresent.
    // 0: the session presented the frame; non-zero: it did not (the core then reports EXECUTOR_REFUSED).
    WgFgExecutorV1 ExecutorHandOff();

    FgSessionState State() const { return state_; }
    FgSessionError RefusedWith() const { return refused_with_; }
    const FgSessionResult& LastResult() const { return last_; }
    const FgSessionCounters& Counters() const { return counters_; }
    const FgHistory* History() const { return history_.get(); }
    const FgTargetScheduler* Scheduler() const { return scheduler_.get(); }
    IFgPresenter* Presenter() const { return presenter_.get(); }
    // Sets the arrival timestamp for the next OnPresent / OnGameFrameOnly call (A3: sampled before any host wait so
    // host throttling never perturbs the scheduler's cadence estimate). Consumed once.
    void SetNextArrival(FgTicks arrival, FgTicks budget_origin = 0, FgTicks previous_release_wait = 0) {
        has_next_arrival_ = true;
        next_arrival_ = arrival;
        next_budget_origin_ = budget_origin;
        next_release_wait_ = previous_release_wait;
    }
    FgRealOnlyReason LastReductionReason() const {
        return scheduler_ ? scheduler_->LastReductionReason() : FgRealOnlyReason::None;
    }
    // The scheduler's first-image budget while the session runs (FgTargetScheduler::SetFirstImageBudget), kept for a
    // scheduler made again after a resize. false: no session.
    bool SetFirstImageBudget(FgTicks budget);
    bool SetMaxGeneratedCap(uint32_t cap);
    bool SetDisplayMinInterval(FgTicks min_interval);
    void SetOverflowReady(bool ready, uint32_t limit = 0) {
        if (scheduler_) scheduler_->SetOverflowReady(ready, limit);
    }
    void SetConsumedLeadProtection(bool active) { if (scheduler_) scheduler_->SetConsumedLeadProtection(active); }
    uint64_t PresentEpoch() const { return present_epoch_; }
    bool MenuLatchPending() const { return menu_latch_; }
    const FgCreateDesc& CreateDesc() const { return desc_.create; }

private:
    static int32_t SubmitThunk(void* self, const WgFgPacket* packet, const WgFgSessionDesc* session);

    FgSessionResult Gate() const;
    FgSessionResult Finish(const FgSessionResult& r);
    FgSessionResult Refuse(FgSessionResult r);
    FgSessionResult Lose(FgSessionResult r);
    void Log(const std::string& line);
    FgTicks Now() const;
    void SetFrameTiming(FgGameFrame* frame);
    FgSessionResult PresentRealOnlyPlan(const FgIntervalPlan& plan, FgSessionResult r,
                                        const FgTextureRef* game_colour = nullptr);
    FgSessionResult RunInterval(const FgFramePacket& packet, const FgIngestResult& ingest, const FgIntervalPlan& plan,
                                bool copy_output, FgCopyReason copy_reason, FgSessionResult r, bool pre_failed);
    bool NeedsRebind(const FgFramePacket& packet) const;
    FgCreateDesc HistoryDesc(const FgCreateDesc& create) const;
    void ReleaseAll();

    FgSessionDesc desc_;
    FgSessionState state_ = FgSessionState::None;
    FgSessionError refused_with_ = FgSessionError::None;
    std::unique_ptr<IFgPresenter> presenter_;
    std::unique_ptr<FgTargetScheduler> scheduler_;
    std::unique_ptr<FgHistory> history_;
    uint64_t present_epoch_ = 0;
    uint64_t queue_epoch_ = 0, swapchain_epoch_ = 0;  // the identity history is bound to
    bool menu_latch_ = false;      // the previous frame's menu decision: this frame's output is a copy
    bool have_frame_ = false;
    FgPreset last_preset_ = FgPreset::A;
    uint32_t last_count_ = 0;
    std::string last_log_;
    FgSessionResult last_;
    FgSessionCounters counters_;
    bool has_next_arrival_ = false;
    FgTicks next_arrival_ = 0;
    FgTicks next_budget_origin_ = 0;
    FgTicks next_release_wait_ = 0;
};

// ---- transport glue ----------------------------------------------------------------------------------------------

}  // namespace wgfg

namespace wgfgp {
class FgProxy;
}

namespace wgfg {

// One game present through the adapter core with the session as its executor: the core assembles the frame and hands
// it to the session (GENERATE), or reports a real-only interval, which this routes to FgSession::OnGameFrameOnly so the
// game frame is still presented by the chain's one presenter. A FORWARDED or REFUSED present does not reach the
// session (FG not taken / not the owner: the game's own present path stays in charge). Returns the core's result code.
int32_t FgSessionPresent(wgfgp::FgProxy* proxy, FgSession* session, const WgFgPresentDesc* desc,
                         WgFgPresentResult* out, FgSessionResult* session_out);

}  // namespace wgfg
