// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// Target-time scheduler for frame generation presentation: game-frame interval estimate, the exact per-interval target
// times for 2x..6x, the variable-refresh window guard, deadline admission and the interval / epoch state machine.
//
// Time is in 100 ns interrupt-time ticks. For interval R[j-1] -> R[j], committed once when R[j] arrives:
//   target[k] = t_base + round(k * T / (N + 1))   k = 1..N (generated)
//   target    = t_base + T                         (the game frame R[j])
//   deadline[k] = target[k] - bridge_and_submit_margin
// t_base is the committed target of R[j-1] (or a fresh anchor), T the interval's period. Every target is one rounding
// from t_base (never accumulated), computed in checked 64-bit integers, and the targets are strictly increasing.
// N is latched per interval: a new count takes effect at the next game frame.
//
// Plain C++17, no platform headers.
#include <cstdint>

#include "fg_presenter.h"

namespace wgfg {

// --- Checked integer helpers -------------------------------------------------------------------------------------

// *out = a + b; Overflow when the sum does not fit.
FgStatus FgCheckedAdd(FgTicks a, FgTicks b, FgTicks* out);
// *out = round(k * period / divisor), halves rounded up; period >= 0, divisor >= 1. Overflow when k * period does
// not fit in 64 bits.
FgStatus FgScaledOffset(std::uint32_t k, FgTicks period, std::uint32_t divisor, FgTicks* out);
// The intermediate time of generated index k of n: float(k) / (float(n) + 1.0f), one single-precision division.
float FgIntermediateTime(std::uint32_t k, std::uint32_t n);
// Fills out[0..n] (n generated targets, then the real endpoint) for one interval. admitted is left true.
FgStatus FgBuildTargets(FgTicks t_base, FgTicks period, std::uint32_t n, FgTicks margin, FgFrameTarget* out);
// Self-consistency of a plan: counts, index numbering, the exact target formula from t_base, strict order, deadlines,
// intermediate times. Every presenter runs it before accepting a plan, so a broken scheduler is refused.
FgStatus FgValidatePlan(const FgIntervalPlan& plan);

// --- Game-frame interval estimator -------------------------------------------------------------------------------

constexpr std::uint32_t kFgEstimatorWindowMax = 16;

struct FgEstimatorConfig {
    std::uint32_t window = 8;              // deltas kept (<= kFgEstimatorWindowMax)
    std::uint32_t min_samples = 3;         // deltas before the estimate counts as stable
    FgTicks min_delta = 10000;             // a delta below 1 ms is a discontinuity (duplicate / bogus present)
    FgTicks max_gap = 2500000;             // a delta above 250 ms is a discontinuity (pause, load, alt-tab)
    std::uint32_t outlier_percent = 25;    // |delta - T| above this share of T is an outlier
    std::uint32_t cadence_change_run = 3;  // this many outliers in a row on the same side: the cadence changed
};

// Median of the recent arrival deltas. Outliers do not enter the window; a run of outliers on one side, a gap or a
// non-increasing arrival re-anchors (window restarted).
class FgIntervalEstimator {
public:
    enum class Event : std::uint8_t { First = 0, Sample, Outlier, Discontinuity };

    explicit FgIntervalEstimator(const FgEstimatorConfig& config = FgEstimatorConfig());
    Event Push(FgTicks arrival);
    void Reset();  // forget everything, including the last arrival
    bool Stable() const { return count_ >= config_.min_samples; }
    FgTicks Period() const { return Stable() ? period_ : 0; }
    std::uint32_t Samples() const { return count_; }

private:
    void Enter(FgTicks delta);
    void Recompute();

    FgEstimatorConfig config_;
    bool has_last_ = false;
    FgTicks last_ = 0;
    FgTicks window_[kFgEstimatorWindowMax] = {};
    std::uint32_t count_ = 0;
    std::uint32_t head_ = 0;
    FgTicks period_ = 0;
    int run_side_ = 0;  // -1 early, +1 late
    std::uint32_t run_len_ = 0;
    FgTicks run_[kFgEstimatorWindowMax] = {};
};

// --- Scheduler -----------------------------------------------------------------------------------------------------

// The display's qualified variable-refresh window for one image's on-screen time T / (N + 1).
struct FgDisplayWindow {
    FgTicks min_interval = 27778;   // refresh ceiling (360 Hz)
    FgTicks max_interval = 208333;  // refresh floor (48 Hz)
    FgTicks margin = 500;           // jitter / transfer margin inside the window
    FgTicks hysteresis = 1000;      // extra band required to re-enter after leaving
    std::uint32_t exit_intervals = 2;     // consecutive intervals inside the raw window but outside the margin
    std::uint32_t reenter_intervals = 8;  // consecutive intervals inside the hysteresis band before re-entry
};

struct FgSchedulerConfig {
    FgEstimatorConfig estimator;
    FgDisplayWindow display;
    FgTicks bridge_and_submit_margin = 10000;  // deadline = target - this
    FgTicks first_image_budget = 20000;        // arrival -> first generated target (capture, index 1, bridge, submit)
    FgTicks real_min_lead = 10000;             // a real target closer than this to its arrival re-anchors
    FgTicks max_lead = 500000;                 // playout lead cap (real target - arrival)
    std::uint32_t max_generated = kFgMaxGenerated;
    FgTicks fixed_refresh_period = 0;   // > 0: require period / (N + 1) to fit a whole multiple of this refresh period
    bool adaptive_count = false;        // false: a fixed count outside the window suspends; true: lower the count
    bool menu_reprimes_history = true;  // a menu frame is not history for the next interval
    bool overflow_fill = false;        // host may pace the release after submitting a fill plan
    bool first_present_admission = false;  // subtract known waits for this scheduler's prior first image
};

// One game frame as the game presents it.
struct FgGameFrame {
    std::uint64_t epoch = 0;
    std::uint64_t frame_id = 0;  // strictly increasing
    FgTicks arrival = 0;         // interrupt time of the game's present
    std::uint32_t requested_count = 0;  // N: 0 = generation off, 1..5
    bool generation_enabled = true;     // menu / game decision for this interval
    bool reset = false;                 // camera cut / history reset on this frame
    FgTicks budget_origin = 0;          // post-wait submission time (0: use arrival)
    FgTicks previous_release_wait = 0;  // measured fill wait returned with the preceding game frame
};

class FgTargetScheduler {
public:
    explicit FgTargetScheduler(const FgSchedulerConfig& config);

    // Starts (or restarts after resize / recreation) the timeline for a swap-chain epoch: anchor, history and the
    // cadence estimate are dropped, the next frames prime real-only.
    FgStatus BeginEpoch(std::uint64_t epoch);

    // Commits the interval ending at this game frame. On Ok, *out is a validated plan (real-only plans have
    // generated_count 0 and a reason). Refusals leave the timeline unchanged.
    FgStatus Commit(const FgGameFrame& frame, FgIntervalPlan* out);

    // After device loss every Commit returns DeviceLost until BeginEpoch on a new device.
    void OnDeviceLost();

    // true while a target may still be handed to the display path at `now`.
    static bool Admissible(const FgFrameTarget& target, FgTicks now) { return now <= target.deadline; }

    // Playout lead an interval of count n needs at period T: round(T * n / (n + 1)) + first_image_budget.
    FgTicks RequiredLead(FgTicks period, std::uint32_t n) const;

    bool Anchored() const { return anchored_; }
    FgTicks LastRealTarget() const { return last_real_target_; }
    std::uint32_t EffectiveCount() const { return effective_; }
    // Why the last Commit reduced generated_count below requested_count (None when full requested count ran).
    FgRealOnlyReason LastReductionReason() const { return last_reduction_reason_; }
    const FgIntervalEstimator& Estimator() const { return estimator_; }
    const FgSchedulerConfig& Config() const { return config_; }
    // The arrival -> first generated target budget, changed while the timeline runs (a host that sees generated
    // images finish after their display time gives the GPU more of it). Clamped to [0, max_lead]. The next commit
    // uses the new lead. Sustained shortfall can re-anchor and excess lead drains through bounded correction.
    void SetFirstImageBudget(FgTicks budget);
    // Optional upper cap on scheduled generated count when producer misses persist.
    // 0 clears the override.
    void SetMaxGeneratedCap(std::uint32_t cap);
    std::uint32_t MaxGeneratedCap() const { return max_generated_cap_; }

private:
    FgStatus CommitImpl(const FgGameFrame& frame, FgIntervalPlan* out);
    bool Fits(std::uint32_t n, FgTicks period, FgTicks band) const;
    bool FitsRaw(std::uint32_t n, FgTicks period) const;
    std::uint32_t Largest(std::uint32_t upto, FgTicks period) const;
    std::uint32_t UpdateCount(std::uint32_t requested, FgTicks period);
    struct FillPlan {
        std::uint32_t count = 0;
        FgTicks period = 0;
        FgTicks wait = 0;
    };
    FillPlan UpdateFill(std::uint32_t requested, bool eligible, FgTicks latest_uncapped);

    FgSchedulerConfig config_;
    FgIntervalEstimator estimator_;
    FgIntervalEstimator uncapped_estimator_;
    FgTicks uncapped_arrival_ = 0;
    FgTicks cadence_arrival_ = 0, previous_credit_wait_ = 0;
    std::uint32_t fill_requested_ = 0;
    std::uint32_t fill_count_ = 0;
    std::uint32_t fill_overrun_ = 0;
    bool epoch_started_ = false;
    bool lost_ = false;
    std::uint64_t epoch_ = 0;
    std::uint64_t interval_id_ = 0;
    bool have_frame_ = false;
    std::uint64_t last_frame_id_ = 0;
    FgTicks last_arrival_ = 0;
    bool anchored_ = false;
    bool have_target_ = false;
    FgTicks last_real_target_ = 0;
    bool history_valid_ = false;
    std::uint32_t last_interval_count_ = 0;
    std::uint32_t last_requested_ = 0;
    std::uint32_t effective_ = 0;
    std::uint32_t bad_run_ = 0;
    std::uint32_t good_run_ = 0;
    std::uint32_t good_candidate_ = 0;
    std::uint32_t shortfall_run_ = 0;
    std::uint32_t vrr_floor_run_ = 0;
    bool over_lead_last_interval_ = false;
    std::uint32_t max_generated_cap_ = 0;
    FgRealOnlyReason last_reduction_reason_ = FgRealOnlyReason::None;
};

}  // namespace wgfg
