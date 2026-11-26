// SPDX-License-Identifier: MIT
// Target-time scheduler: see fg_target_scheduler.h.
#include "fg_target_scheduler.h"

#include <limits>

namespace wgfg {

namespace {

constexpr FgTicks kTicksMax = std::numeric_limits<FgTicks>::max();
constexpr FgTicks kTicksMin = std::numeric_limits<FgTicks>::min();

FgTicks Clamp(FgTicks v, FgTicks lo, FgTicks hi) { return v < lo ? lo : (v > hi ? hi : v); }

// a * b for small non-negative factors; false when the product does not fit.
bool CheckedMul(FgTicks a, FgTicks b, FgTicks* out) {
    if (a < 0 || b < 0) return false;
    if (a != 0 && b > kTicksMax / a) return false;
    *out = a * b;
    return true;
}

}  // namespace

// --- Checked integer helpers -------------------------------------------------------------------------------------

FgStatus FgCheckedAdd(FgTicks a, FgTicks b, FgTicks* out) {
    if (!out) return FgStatus::InvalidArgument;
    if ((b > 0 && a > kTicksMax - b) || (b < 0 && a < kTicksMin - b)) return FgStatus::Overflow;
    *out = a + b;
    return FgStatus::Ok;
}

FgStatus FgScaledOffset(std::uint32_t k, FgTicks period, std::uint32_t divisor, FgTicks* out) {
    if (!out || period < 0 || divisor == 0) return FgStatus::InvalidArgument;
    FgTicks product = 0;
    if (!CheckedMul(static_cast<FgTicks>(k), period, &product)) return FgStatus::Overflow;
    const FgTicks d = static_cast<FgTicks>(divisor);
    const FgTicks q = product / d;
    const FgTicks r = product % d;
    // halves round up: r / d >= 1/2  <=>  2r >= d (2r < 2d fits: d is a 32-bit value)
    *out = q + (2 * r >= d ? 1 : 0);
    return FgStatus::Ok;
}

float FgIntermediateTime(std::uint32_t k, std::uint32_t n) {
    return static_cast<float>(k) / (static_cast<float>(n) + 1.0f);
}

FgStatus FgBuildTargets(FgTicks t_base, FgTicks period, std::uint32_t n, FgTicks margin, FgFrameTarget* out) {
    if (!out || n > kFgMaxGenerated || margin < 0 || period < static_cast<FgTicks>(n) + 1)
        return FgStatus::InvalidArgument;
    for (std::uint32_t k = 1; k <= n; ++k) {
        FgFrameTarget& t = out[k - 1];
        FgTicks offset = 0;
        FgStatus s = FgScaledOffset(k, period, n + 1, &offset);
        if (s != FgStatus::Ok) return s;
        t = FgFrameTarget();
        t.index = k;
        t.real = false;
        if ((s = FgCheckedAdd(t_base, offset, &t.target)) != FgStatus::Ok) return s;
        if ((s = FgCheckedAdd(t.target, -margin, &t.deadline)) != FgStatus::Ok) return s;
        t.t = FgIntermediateTime(k, n);
        t.admitted = true;
    }
    FgFrameTarget& r = out[n];
    r = FgFrameTarget();
    r.index = n + 1;
    r.real = true;
    FgStatus s = FgCheckedAdd(t_base, period, &r.target);
    if (s != FgStatus::Ok) return s;
    if ((s = FgCheckedAdd(r.target, -margin, &r.deadline)) != FgStatus::Ok) return s;
    r.t = 1.0f;
    r.admitted = true;
    return FgStatus::Ok;
}

FgStatus FgValidatePlan(const FgIntervalPlan& plan) {
    const std::uint32_t n = plan.generated_count;
    if (n > kFgMaxGenerated || plan.requested_count > kFgMaxGenerated || n > plan.requested_count)
        return FgStatus::InvalidArgument;
    if (plan.target_count != n + 1) return FgStatus::InvalidArgument;
    // a generating interval has no real-only reason; a real-only one always has one
    if ((n == 0) != (plan.real_only_reason != FgRealOnlyReason::None)) return FgStatus::InvalidArgument;
    if (static_cast<std::uint32_t>(plan.real_only_reason) >= kFgRealOnlyReasonCount) return FgStatus::InvalidArgument;
    if (plan.period < 1 || plan.margin < 0) return FgStatus::InvalidArgument;
    FgFrameTarget expect[kFgMaxTargets];
    const FgStatus s = FgBuildTargets(plan.t_base, plan.period, n, plan.margin, expect);
    if (s != FgStatus::Ok) return s == FgStatus::Overflow ? s : FgStatus::InvalidArgument;
    FgTicks previous = plan.t_base;
    for (std::uint32_t i = 0; i <= n; ++i) {
        const FgFrameTarget& got = plan.targets[i];
        const FgFrameTarget& want = expect[i];
        if (got.index != want.index || got.real != want.real || got.target != want.target ||
            got.deadline != want.deadline || got.t != want.t)
            return FgStatus::InvalidArgument;
        if (got.target <= previous) return FgStatus::InvalidArgument;  // strictly increasing from t_base
        previous = got.target;
    }
    if (!plan.targets[n].admitted) return FgStatus::InvalidArgument;  // the real endpoint is never dropped
    return FgStatus::Ok;
}

// --- Game-frame interval estimator -------------------------------------------------------------------------------

FgIntervalEstimator::FgIntervalEstimator(const FgEstimatorConfig& config) : config_(config) {
    if (config_.window < 1) config_.window = 1;
    if (config_.window > kFgEstimatorWindowMax) config_.window = kFgEstimatorWindowMax;
    if (config_.min_samples < 1) config_.min_samples = 1;
    if (config_.min_samples > config_.window) config_.min_samples = config_.window;
    if (config_.cadence_change_run < 1) config_.cadence_change_run = 1;
    if (config_.cadence_change_run > config_.window) config_.cadence_change_run = config_.window;
    if (config_.min_delta < 1) config_.min_delta = 1;
}

void FgIntervalEstimator::Reset() {
    has_last_ = false;
    last_ = 0;
    count_ = 0;
    head_ = 0;
    period_ = 0;
    run_side_ = 0;
    run_len_ = 0;
}

void FgIntervalEstimator::Enter(FgTicks delta) {
    window_[head_] = delta;
    head_ = (head_ + 1) % config_.window;
    if (count_ < config_.window) ++count_;
}

void FgIntervalEstimator::Recompute() {
    FgTicks v[kFgEstimatorWindowMax];
    const std::uint32_t c = count_;
    for (std::uint32_t i = 0; i < c; ++i) v[i] = window_[i];  // the first count_ entries are the valid ones
    for (std::uint32_t i = 1; i < c; ++i) {                   // insertion sort (at most 16 values)
        const FgTicks x = v[i];
        std::uint32_t j = i;
        for (; j > 0 && v[j - 1] > x; --j) v[j] = v[j - 1];
        v[j] = x;
    }
    if (c == 0) period_ = 0;
    else if (c % 2 == 1) period_ = v[c / 2];
    else period_ = v[c / 2 - 1] + (v[c / 2] - v[c / 2 - 1] + 1) / 2;
}

FgIntervalEstimator::Event FgIntervalEstimator::Push(FgTicks arrival) {
    if (!has_last_) {
        has_last_ = true;
        last_ = arrival;
        return Event::First;
    }
    const bool backwards = arrival <= last_;
    const FgTicks delta = backwards ? 0 : arrival - last_;  // arrival > last_: no overflow for sane clocks
    last_ = arrival;
    if (backwards || delta < config_.min_delta || delta > config_.max_gap) {
        count_ = 0;
        head_ = 0;
        period_ = 0;
        run_side_ = 0;
        run_len_ = 0;
        return Event::Discontinuity;
    }
    if (Stable()) {
        const FgTicks dev = delta - period_;
        const FgTicks adev = dev < 0 ? -dev : dev;
        if (adev * 100 > period_ * static_cast<FgTicks>(config_.outlier_percent)) {
            const int side = dev < 0 ? -1 : 1;
            if (side != run_side_) {
                run_side_ = side;
                run_len_ = 0;
            }
            run_[run_len_++] = delta;
            if (run_len_ >= config_.cadence_change_run) {  // the cadence changed: restart from the run
                count_ = 0;
                head_ = 0;
                for (std::uint32_t i = 0; i < run_len_; ++i) Enter(run_[i]);
                run_side_ = 0;
                run_len_ = 0;
                Recompute();
                return Event::Discontinuity;
            }
            return Event::Outlier;
        }
        run_side_ = 0;
        run_len_ = 0;
    }
    Enter(delta);
    Recompute();
    return Event::Sample;
}

// --- Scheduler -----------------------------------------------------------------------------------------------------

FgTargetScheduler::FgTargetScheduler(const FgSchedulerConfig& config) : config_(config), estimator_(config.estimator) {
    if (config_.max_generated > kFgMaxGenerated) config_.max_generated = kFgMaxGenerated;
    if (config_.real_min_lead < 1) config_.real_min_lead = 1;
    if (config_.bridge_and_submit_margin < 0) config_.bridge_and_submit_margin = 0;
    if (config_.first_image_budget < 0) config_.first_image_budget = 0;
    if (config_.display.min_interval < 1) config_.display.min_interval = 1;
    if (config_.max_lead < config_.real_min_lead) config_.max_lead = config_.real_min_lead;
}

FgStatus FgTargetScheduler::BeginEpoch(std::uint64_t epoch) {
    if (epoch_started_ && !lost_ && epoch <= epoch_) return FgStatus::InvalidArgument;
    epoch_started_ = true;
    lost_ = false;
    epoch_ = epoch;
    estimator_.Reset();
    anchored_ = false;
    have_target_ = false;
    last_real_target_ = 0;
    history_valid_ = false;
    generated_in_epoch_ = false;
    last_requested_ = 0;
    effective_ = 0;
    bad_run_ = 0;
    good_run_ = 0;
    return FgStatus::Ok;
}

void FgTargetScheduler::OnDeviceLost() {
    lost_ = true;
    anchored_ = false;
    have_target_ = false;
    history_valid_ = false;
}

void FgTargetScheduler::SetFirstImageBudget(FgTicks budget) {
    if (budget < 0) budget = 0;
    if (budget > config_.max_lead) budget = config_.max_lead;
    config_.first_image_budget = budget;
}

FgTicks FgTargetScheduler::RequiredLead(FgTicks period, std::uint32_t n) const {
    if (period <= 0 || n == 0) return config_.real_min_lead;
    FgTicks offset = 0;
    FgTicks lead = 0;
    if (FgScaledOffset(n, period, n + 1, &offset) != FgStatus::Ok ||
        FgCheckedAdd(offset, config_.first_image_budget, &lead) != FgStatus::Ok)
        return kTicksMax;
    return lead < config_.real_min_lead ? config_.real_min_lead : lead;
}

bool FgTargetScheduler::Fits(std::uint32_t n, FgTicks period, FgTicks band) const {
    const FgDisplayWindow& w = config_.display;
    const FgTicks d = static_cast<FgTicks>(n) + 1;
    const FgTicks lo = w.min_interval + w.margin + band;
    const FgTicks hi = w.max_interval - w.margin - band;
    FgTicks lo_total = 0;
    FgTicks hi_total = 0;
    if (lo < 1 || hi < lo || !CheckedMul(d, lo, &lo_total) || !CheckedMul(d, hi, &hi_total)) return false;
    return period >= lo_total && period <= hi_total;
}

bool FgTargetScheduler::FitsRaw(std::uint32_t n, FgTicks period) const {
    return Fits(n, period, -config_.display.margin);
}

std::uint32_t FgTargetScheduler::Largest(std::uint32_t upto, FgTicks period) const {
    for (std::uint32_t n = upto; n >= 1; --n)
        if (Fits(n, period, 0)) return n;
    return 0;
}

// Variable-refresh guard: the count this interval may generate for the requested one. Leaving the window is
// immediate outside the raw window and takes exit_intervals inside the margin; re-entry needs reenter_intervals
// inside the stricter hysteresis band. A changed request is evaluated at once (a deliberate change, not jitter).
std::uint32_t FgTargetScheduler::UpdateCount(std::uint32_t requested, FgTicks period) {
    const FgDisplayWindow& w = config_.display;
    const bool adaptive = config_.adaptive_count;
    if (requested != last_requested_) {
        last_requested_ = requested;
        bad_run_ = 0;
        good_run_ = 0;
        effective_ = adaptive ? Largest(requested, period) : (Fits(requested, period, 0) ? requested : 0);
        return effective_;
    }
    if (effective_ > 0) {
        bool leave = false;
        if (!FitsRaw(effective_, period)) leave = true;
        else if (!Fits(effective_, period, 0)) leave = ++bad_run_ >= w.exit_intervals;
        else bad_run_ = 0;
        if (leave) {
            bad_run_ = 0;
            good_run_ = 0;
            effective_ = adaptive ? Largest(effective_ - 1, period) : 0;
        }
    }
    const std::uint32_t up = adaptive ? (effective_ < requested ? effective_ + 1 : 0) : (effective_ == 0 ? requested : 0);
    if (up > 0) {
        if (Fits(up, period, w.hysteresis)) {
            if (++good_run_ >= w.reenter_intervals) {
                effective_ = up;
                good_run_ = 0;
                bad_run_ = 0;
            }
        } else {
            good_run_ = 0;
        }
    }
    return effective_;
}

FgStatus FgTargetScheduler::Commit(const FgGameFrame& frame, FgIntervalPlan* out) {
    if (!out) return FgStatus::InvalidArgument;
    *out = FgIntervalPlan();
    if (lost_) {
        out->real_only_reason = FgRealOnlyReason::DeviceLost;
        return FgStatus::DeviceLost;
    }
    if (!epoch_started_ || frame.epoch != epoch_) {
        out->real_only_reason = FgRealOnlyReason::EpochChanged;
        return FgStatus::StaleEpoch;
    }
    if (frame.requested_count > config_.max_generated) return FgStatus::InvalidArgument;
    if (have_frame_ && (frame.frame_id <= last_frame_id_ || frame.arrival < last_arrival_))
        return FgStatus::InvalidArgument;
    // Transactional: work on a copy, keep it only when the whole commit succeeded.
    FgTargetScheduler next = *this;
    const FgStatus s = next.CommitImpl(frame, out);
    if (s != FgStatus::Ok) {
        *out = FgIntervalPlan();
        return s;
    }
    *this = next;
    return FgStatus::Ok;
}

FgStatus FgTargetScheduler::CommitImpl(const FgGameFrame& frame, FgIntervalPlan* out) {
    const FgDisplayWindow& w = config_.display;
    const FgIntervalEstimator::Event ev = estimator_.Push(frame.arrival);
    have_frame_ = true;
    last_frame_id_ = frame.frame_id;
    last_arrival_ = frame.arrival;
    if (ev == FgIntervalEstimator::Event::Discontinuity) anchored_ = false;
    const FgTicks period_estimate = estimator_.Period();

    // History: may this interval interpolate between the previous game frame and this one?
    const bool previous_valid = history_valid_;
    FgRealOnlyReason reason = FgRealOnlyReason::None;
    if (frame.reset) {
        reason = FgRealOnlyReason::Reset;
        history_valid_ = true;  // this frame primes the next interval
    } else if (!frame.generation_enabled) {
        reason = FgRealOnlyReason::MenuDisabled;
        history_valid_ = !config_.menu_reprimes_history;
    } else if (frame.requested_count == 0) {
        reason = FgRealOnlyReason::GenerationOff;
        history_valid_ = false;  // nothing is captured while generation is off
    } else {
        history_valid_ = true;
        if (!previous_valid) reason = FgRealOnlyReason::NotPrimed;
    }

    // Count for the display window (kept current while real-only, so hysteresis spans menus and suspensions).
    std::uint32_t effective = 0;
    if (frame.requested_count == 0) {
        last_requested_ = 0;
        effective_ = 0;
        bad_run_ = 0;
        good_run_ = 0;
    } else if (period_estimate > 0) {
        effective = UpdateCount(frame.requested_count, period_estimate);
    }
    if (reason == FgRealOnlyReason::None) {
        if (period_estimate == 0) reason = FgRealOnlyReason::NotPrimed;
        else if (effective == 0) reason = FgRealOnlyReason::OutOfVrrWindow;
    }
    std::uint32_t n = reason == FgRealOnlyReason::None ? effective : 0;

    // Playout lead: what this count needs (a real-only interval keeps the lead of the count it would resume with).
    FgTicks lead_need = RequiredLead(period_estimate, n > 0 ? n : effective);
    if (lead_need > config_.max_lead) {
        if (n > 0) {
            n = 0;
            reason = FgRealOnlyReason::LatencyCap;
        }
        lead_need = RequiredLead(period_estimate, 0);
    }

    bool reanchor = !anchored_ || period_estimate == 0;
    FgTicks lead = 0;
    if (!reanchor) {
        FgTicks end = 0;
        if (FgCheckedAdd(last_real_target_, period_estimate, &end) != FgStatus::Ok ||
            FgCheckedAdd(end, -frame.arrival, &lead) != FgStatus::Ok)
            return FgStatus::Overflow;
        if (lead < config_.real_min_lead) {
            reanchor = true;  // behind: the real endpoint cannot be met on this timeline
        } else if (lead > config_.max_lead) {
            if (n > 0) {  // too much queued latency: drain it real-only (the target never moves backwards)
                n = 0;
                reason = FgRealOnlyReason::LatencyCap;
            }
        } else if (n > 0) {
            const FgTicks slot = period_estimate / (static_cast<FgTicks>(n) + 1);
            if (lead + slot / 2 < lead_need) reanchor = true;  // too little lead for this count
        }
    }

    FgTicks t_base = 0;
    FgTicks period = 0;
    if (reanchor) {
        FgTicks target = 0;
        if (FgCheckedAdd(frame.arrival, lead_need, &target) != FgStatus::Ok) return FgStatus::Overflow;
        if (have_target_) {
            FgTicks floor = 0;
            if (FgCheckedAdd(last_real_target_, w.min_interval, &floor) != FgStatus::Ok) return FgStatus::Overflow;
            if (target < floor) target = floor;
            t_base = last_real_target_;
        } else {
            t_base = frame.arrival;
        }
        period = target - t_base;
        if (n > 0) {
            n = 0;
            if (reason == FgRealOnlyReason::None)
                reason = generated_in_epoch_ ? FgRealOnlyReason::Reanchor : FgRealOnlyReason::NotPrimed;
        }
        anchored_ = period_estimate > 0;
        out->reanchored = true;
    } else {
        // Bounded lead correction: keeps the lead near what the count needs without a visible step.
        t_base = last_real_target_;
        const FgTicks diff = lead_need - lead;
        const FgTicks deadband = period_estimate / 16;
        FgTicks correction = 0;
        if (diff > deadband || diff < -deadband) {
            const FgTicks limit = n > 0 ? period_estimate / 128 : period_estimate / 8;
            correction = Clamp(diff / 8, -limit, limit);
        }
        period = period_estimate + correction;
        FgTicks floor = 0;
        if (!CheckedMul(static_cast<FgTicks>(n) + 1, w.min_interval, &floor)) return FgStatus::Overflow;
        if (period < floor) period = floor;
        const FgTicks feasible = frame.arrival + config_.real_min_lead - t_base;
        if (period < feasible) period = feasible;
    }
    if (period < static_cast<FgTicks>(n) + 1) period = static_cast<FgTicks>(n) + 1;

    out->epoch = epoch_;
    out->interval_id = ++interval_id_;
    out->real_frame_id = frame.frame_id;
    out->requested_count = frame.requested_count;
    out->generated_count = n;
    out->real_only_reason = n > 0 ? FgRealOnlyReason::None : reason;
    if (n == 0 && out->real_only_reason == FgRealOnlyReason::None) out->real_only_reason = FgRealOnlyReason::NotPrimed;
    out->t_base = t_base;
    out->period = period;
    out->margin = config_.bridge_and_submit_margin;
    out->target_count = n + 1;
    FgStatus s = FgBuildTargets(t_base, period, n, out->margin, out->targets);
    if (s != FgStatus::Ok) return s;
    for (std::uint32_t k = 0; k < n; ++k) out->targets[k].admitted = Admissible(out->targets[k], frame.arrival);
    if ((s = FgValidatePlan(*out)) != FgStatus::Ok) return s;

    last_real_target_ = out->targets[n].target;
    have_target_ = true;
    if (n > 0) generated_in_epoch_ = true;
    return FgStatus::Ok;
}

}  // namespace wgfg
