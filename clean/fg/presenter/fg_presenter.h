// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// Frame generation presentation boundary: the value types every presenter back end shares and the IFgPresenter
// interface (one owner of presentation per swap chain and epoch).
//
// A presenter receives, for every game frame the game presents, one interval plan from the target scheduler
// (fg_target_scheduler.h): the real image R[j] and up to five generated images between R[j-1] and R[j], each with a
// requested display time in 100 ns interrupt-time ticks. The presenter owns the display slots, the order in which
// images reach the display, late drops, cancellation of stale queued presents and the swap chain's lifecycle
// (resize, quiesce, detach, device loss).
//
// Rules every back end keeps:
//  - One visible owner per chain: Attach fails (Refused) while another presenter owns the chain.
//  - A game frame is never dropped and never waits for generation: it has its own slot credit.
//  - Generated images reach the display in target order; one that cannot meet its deadline is dropped with a reason,
//    and its output lease stays held until its producer fence completes.
//  - A refusal is an error status with a reason. Nothing returns a successful present, resize or interval that did
//    not happen, and no call falls back to another route on its own.
//
// Plain C++17, no Windows or vendor headers: the Windows back ends (fg_win_comp_presenter.h, fg_dxgi_presenter.h)
// and the deterministic fake (fg_fake_presenter.h) implement the same interface.
#include <cstdint>

#include "../fg_limits.h"  // kFgMaxGenerated: N = 1..5, 2x..6x

namespace wgfg {

// Interrupt-time ticks of 100 ns (the unit of the Windows interrupt-time clock).
using FgTicks = std::int64_t;

constexpr std::uint32_t kFgMaxTargets = kFgMaxGenerated + 1;  // generated images plus the real endpoint

// Result of every presenter / scheduler call. Only Ok means the call did what it says.
enum class FgStatus : std::uint8_t {
    Ok = 0,
    InvalidArgument,  // malformed input (a plan that fails validation, an index out of range, ...)
    WrongState,       // the call is not allowed in the current state (not attached, quiescing, ...)
    StaleEpoch,       // the input belongs to a retired swap-chain epoch
    NoSlot,           // no display slot for the real image: back-pressure, the caller must wait
    Busy,             // work of the old epoch is still in flight: retry after it completes (never a fake success)
    Unsupported,      // the probe refused this back end / mode
    Refused,          // another owner holds the chain, or the route is disabled after a hard failure
    DeviceLost,       // the device or presentation manager is gone: only Detach is still accepted
    Overflow,         // a time computation would overflow 64 bits
    ResizeFailed,     // the forwarded swap-chain resize failed: its result is passed through
};

enum class FgPresenterBackend : std::uint8_t {
    None = 0,
    WindowsComposition,  // presentation manager with per-image target times
    AmdSwapChain,        // vendor swap-chain wrapper (owns cadence itself)
    Fake,                // deterministic test presenter
    DxgiFlip,            // DXGI flip-discard swap chain paced by the presenter's own timer (fg_dxgi_presenter.h)
};

// Who records the generated output work for an interval.
enum class FgOutputRecorder : std::uint8_t {
    Executor = 0,  // the frame generation executor records into leases the presenter hands out
    Presenter,     // the presenter's back end asks for the work from inside its own callback
};

// Why an interval presents only its real image. Each reason is observable separately.
enum class FgRealOnlyReason : std::uint8_t {
    None = 0,        // the interval generates
    GenerationOff,   // the requested count is 0 (frame generation off)
    NotPrimed,       // no valid previous game frame or no stable cadence yet (start, after a reset / menu / epoch)
    MenuDisabled,    // the game or menu decision disabled generation for this interval
    Reset,           // camera cut / reset flag on this frame: history re-primes with it
    OutOfVrrWindow,  // the generated cadence leaves the display's variable refresh window
    Reanchor,        // the timeline was re-anchored after a discontinuity (hitch, cadence change, lead out of bounds)
    LatencyCap,      // the playout lead this count needs exceeds the configured cap
    Pressure,        // no slot / pending-interval credit for generated images
    EpochChanged,    // swap-chain epoch changed (resize, recreation)
    DeviceLost,      // the device is gone
    Late,            // every generated reservation missed its deadline
};
constexpr std::uint32_t kFgRealOnlyReasonCount = 12;

// Why one generated index did not reach the display.
enum class FgDropReason : std::uint8_t {
    None = 0,
    Late,       // not ready before deadline[k] = target[k] - bridge_and_submit_margin
    Pressure,   // no free display slot at reservation
    Cancelled,  // queued, then cancelled as part of a stale suffix
    Revoked,    // the interval's generation was revoked after submission (late menu / reset decision)
    Epoch,      // the epoch was retired before it was shown
};

enum class FgPresenterState : std::uint8_t {
    Detached = 0,  // owns no chain
    Attached,      // owns a chain for one epoch, accepting intervals
    Quiescing,     // admission stopped (resize / detach pending): draining old-epoch work
    Failed,        // a hard failure (failed resize): refuses until Detach
    Lost,          // device removed / presentation manager lost: refuses until Detach
};

struct FgPresenterProbe {
    FgPresenterBackend backend = FgPresenterBackend::None;
    bool supported = false;
    bool per_image_target_time = false;  // can request a display time for every image independently
    std::uint32_t max_generated = 0;     // highest N this back end qualifies (0..5)
    FgStatus refusal = FgStatus::Unsupported;  // when !supported: why
};

// One requested display time inside an interval: generated index k = 1..N, or the real endpoint (index N + 1).
struct FgFrameTarget {
    std::uint32_t index = 0;
    bool real = false;
    FgTicks target = 0;    // requested display time
    FgTicks deadline = 0;  // last time the image may be handed to the display path (target - margin)
    float t = 0.0f;        // intermediate time: float(k) / (float(N) + 1.0f); 1.0 for the real endpoint
    bool admitted = true;  // false: dropped by the scheduler at commit (already past its deadline)
};

enum class FgReanchorReason : std::uint8_t {
    None, Initial, CadenceChange, RealTooLate, ExcessLead, GeneratedDeadline, LeadShortfall, Count
};
constexpr std::uint32_t kFgReanchorReasonCount = static_cast<std::uint32_t>(FgReanchorReason::Count);
const char* FgReanchorReasonName(FgReanchorReason reason);

// The schedule for interval R[j-1] -> R[j], committed once when R[j] arrives.
struct FgIntervalPlan {
    std::uint64_t epoch = 0;
    std::uint64_t interval_id = 0;    // strictly increasing per scheduler
    std::uint64_t real_frame_id = 0;  // game frame id of R[j]
    std::uint32_t requested_count = 0;  // N the game asked for
    std::uint32_t generated_count = 0;  // N this interval actually schedules (0 = real only)
    FgRealOnlyReason real_only_reason = FgRealOnlyReason::None;
    bool reanchored = false;  // t_base is a fresh anchor, not the previous real target
    FgTicks t_base = 0;       // committed target of R[j-1] (or the new anchor)
    FgTicks period = 0;       // T used for this interval (estimate plus bounded lead correction)
    FgTicks margin = 0;       // bridge_and_submit_margin used for the deadlines
    std::uint32_t target_count = 0;  // generated_count + 1
    FgFrameTarget targets[kFgMaxTargets] = {};
    FgReanchorReason reanchor_reason = FgReanchorReason::None;
    FgTicks fill_period = 0;        // chosen overflow period, when this plan schedules the higher count
    FgTicks fill_release_wait = 0;  // additional game-frame release delay, after submission
};

// One reserved display slot.
struct FgPresentSlot {
    std::uint64_t epoch = 0;
    std::uint64_t interval_id = 0;
    std::uint64_t frame_id = 0;
    std::uint32_t index = 0;  // 1..N generated, N + 1 real
    bool real = false;
    std::uint32_t slot = 0;   // display slot number
    bool reserved = false;
    FgDropReason refused = FgDropReason::None;  // !reserved: why
};

struct FgIntervalReservation {
    std::uint64_t epoch = 0;
    std::uint64_t interval_id = 0;
    FgOutputRecorder recorder = FgOutputRecorder::Executor;
    std::uint32_t count = 0;  // plan.target_count
    FgPresentSlot slots[kFgMaxTargets] = {};
};

// A producer fence value: the image is ready when the fence reaches value.
struct FgReadyFence {
    std::uint32_t fence = 0;
    std::uint64_t value = 0;
};

// The executor's work for a reserved interval: one ready fence per reserved slot, in plan order (index 1..N, real).
struct FgIntervalSubmission {
    std::uint64_t epoch = 0;
    std::uint64_t interval_id = 0;
    std::uint32_t count = 0;
    FgReadyFence ready[kFgMaxTargets] = {};
    bool produced[kFgMaxTargets] = {};  // false: the executor did not record this index (dropped before work)
};

struct FgRealImage {
    std::uint64_t epoch = 0;
    std::uint64_t frame_id = 0;
    FgReadyFence ready;
};

// What the display path did with one present.
enum class FgPresentOutcome : std::uint8_t { Displayed = 0, Cancelled, Dropped };

struct FgPresentStats {
    std::uint64_t game_presents = 0;         // game frames the game presented through this owner
    std::uint64_t real_queued = 0;           // real images handed to the display path (includes re-presents)
    std::uint64_t real_represented = 0;      // fresh real endpoints after a cancelled suffix
    std::uint64_t generated_queued = 0;      // generated images handed to the display path
    std::uint64_t displayed_real = 0;
    std::uint64_t displayed_generated = 0;
    std::uint64_t dropped_late = 0;
    std::uint64_t dropped_pressure = 0;
    std::uint64_t cancelled = 0;             // queued presents cancelled (any kind)
    std::uint64_t revoked = 0;               // generated images revoked after submission
    std::uint64_t dropped_epoch = 0;         // pending images dropped because their epoch retired
    std::uint64_t refused_calls = 0;         // calls that returned a refusal / error
    std::uint64_t epochs = 0;                // epochs attached
    std::uint64_t real_only_intervals[kFgRealOnlyReasonCount] = {};  // per FgRealOnlyReason
    FgTicks max_slip = 0;                    // largest (actual - target) of a displayed image
    std::uint64_t display_skipped_real = 0;
    std::uint64_t display_skipped_generated = 0;
    std::uint64_t display_skipped = 0;       // queued presents the display path itself skipped (a newer one was ready)
    std::uint64_t display_collisions = 0;    // displayed less than half a refresh after the previous displayed image
    // Non-overlapping per-stage funnel counters (A4): requested -> scheduled -> reserved -> queued -> submitted ->
    // displayed / skipped. dropped_pre_queue + dropped_post_queue == dropped_late + dropped_pressure with no overlap.
    std::uint64_t generated_requested = 0;         // sum of plan.requested_count across accepted intervals
    std::uint64_t generated_scheduled = 0;         // sum of plan.generated_count across accepted intervals
    std::uint64_t generated_reserved = 0;          // generated slots reserved in BeginInterval
    std::uint64_t generated_submitted = 0;         // generated images handed to the OS present call
    std::uint64_t generated_dropped_pre_queue = 0; // dropped before reaching the outbox (reserve / submit / pump)
    std::uint64_t generated_late_pre_queue = 0;    // pre-queue drops due to lateness
    std::uint64_t generated_late_ready_pre_queue = 0; // producer fence complete when the worker found it overdue
    std::uint64_t generated_late_after_stall = 0;    // crossed its deadline or lost its slot to the prior stall
    std::uint64_t generated_dropped_post_queue = 0;// dropped after reaching the outbox (presenter worker)
    std::uint64_t real_submitted = 0;              // real images handed to the OS present call
    bool display_time_measured = false;            // true: compositor/display feedback; false: submit/vblank estimate
};

// The presentation boundary. Calls come from the game's present thread; a back end's own worker (or, in the fake,
// an explicit Pump) moves ready images to the display.
class IFgPresenter {
public:
    virtual ~IFgPresenter() = default;

    // Typed capability answer for this back end on the attached (or candidate) chain.
    virtual FgPresenterProbe Probe() const = 0;

    // Claims the chain for `epoch`. Refused while another owner holds it; Unsupported when the probe refuses.
    virtual FgStatus Attach(std::uint64_t chain_id, std::uint64_t epoch) = 0;

    // Reserves the real image's slot first, then one slot per admitted generated index. The plan is validated
    // (fg_target_scheduler.h FgValidatePlan) and must continue the epoch's timeline (strictly later targets).
    virtual FgStatus BeginInterval(const FgIntervalPlan& plan, FgIntervalReservation* out) = 0;

    // Hands over the reserved interval's ready fences. Images reach the display in target order once ready.
    virtual FgStatus SubmitInterval(const FgIntervalSubmission& submission) = 0;

    // One real present on the active display path for a real-only plan (plan.generated_count == 0).
    virtual FgStatus PresentRealOnly(const FgIntervalPlan& plan, const FgRealImage& real) = 0;

    // Revokes an interval's generated images after submission (a late menu / reset decision): pending ones are
    // dropped, queued ones are cancelled as a suffix, and the interval's real image stays (re-presented if its
    // queued present was part of the cancelled suffix).
    virtual FgStatus RevokeGenerated(std::uint64_t interval_id, FgDropReason reason) = 0;

    // Stops admission and cancels every present not yet displayed (resize, detach). Old-epoch output leases stay
    // held until their producer fences complete.
    virtual FgStatus Quiesce() = 0;

    // Size of the buffers created by the next successful CompleteResize (no-op when the back end reads the chain).
    virtual FgStatus SetResizeTarget(std::uint32_t width, std::uint32_t height) {
        (void)width;
        (void)height;
        return FgStatus::Ok;
    }

    // After Quiesce and the forwarded swap-chain resize: `resize_ok` is the resize's own result. Busy while
    // old-epoch work is in flight; ResizeFailed (and state Failed) when the resize failed; Ok and a fresh epoch
    // otherwise.
    virtual FgStatus CompleteResize(bool resize_ok, std::uint64_t new_epoch) = 0;

    // Device removed / presentation manager lost: drops all work at once, refuses everything but Detach.
    virtual FgStatus OnDeviceLost() = 0;

    // Releases the chain. Busy while old work is in flight on a healthy device; always succeeds after loss.
    virtual FgStatus Detach() = 0;

    virtual FgPresenterState State() const = 0;
    virtual FgPresentStats Stats() const = 0;
};

const char* FgStatusName(FgStatus s);
const char* FgRealOnlyReasonName(FgRealOnlyReason r);
const char* FgDropReasonName(FgDropReason r);

}  // namespace wgfg
