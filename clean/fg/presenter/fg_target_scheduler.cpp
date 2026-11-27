// SPDX-License-Identifier: GPL-3.0-only
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
    last_interval_generated_ = false;
    last_requested_ = 0;
    effective_ = 0;
    bad_run_ = 0;
    good_run_ = 0;
    shortfall_run_ = 0;
    vrr_floor_run_ = 0;
    over_lead_last_interval_ = false;
    last_reduction_reason_ = FgRealOnlyReason::None;
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
    // A large bootstrap correction should not repeat a whole interval of misses at the old lead.
    // Ordinary controller adjustments still need two consecutive shortfalls before they re-anchor.
    if (budget - config_.first_image_budget >= 80000) shortfall_run_ = 1;
    config_.first_image_budget = budget;
}

void FgTargetScheduler::SetMaxGeneratedCap(std::uint32_t cap) {
    if (cap > config_.max_generated) cap = config_.max_generated;
    max_generated_cap_ = cap;
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
    if (period < lo_total || period > hi_total) return false;
    if (config_.fixed_refresh_period > 0) {
        const FgTicks rp = config_.fixed_refresh_period;
        const FgTicks interval = period / d;
        const FgTicks k = (interval + rp / 2) / rp;
        if (k < 1 || k > 0xFFFF) return false;
        const FgTicks diff = interval - k * rp;
        if ((diff < 0 ? -diff : diff) > rp / 8) return false;
    }
    return true;
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
    std::uint32_t up = adaptive ? (effective_ < requested ? effective_ + 1 : 0) : (effective_ == 0 ? requested : 0);
    if (up > 0) {
        // A fixed grid can support 2x and 4x with no legal 3x plan.
        if (adaptive && config_.fixed_refresh_period > 0) {
            while (up < requested && !Fits(up, period, w.hysteresis)) ++up;
        }
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
    last_reduction_reason_ = (n < frame.requested_count && reason == FgRealOnlyReason::None)
                                 ? FgRealOnlyReason::OutOfVrrWindow
                                 : reason;
    if (n > 0 && max_generated_cap_ > 0 && n > max_generated_cap_) {
        n = max_generated_cap_;
        last_reduction_reason_ = FgRealOnlyReason::LatencyCap;
    }

    const FgTicks origin = frame.budget_origin > frame.arrival ? frame.budget_origin : frame.arrival;
    const auto needed_lead = [&](std::uint32_t count) {
        return RequiredLead(period_estimate, count == 0 && last_interval_generated_ ? effective : count);
    };

    FgTicks lead_need = needed_lead(n);
    if (lead_need > config_.max_lead) {
        while (n > 0 && needed_lead(n) > config_.max_lead) --n;
        lead_need = n > 0 ? needed_lead(n) : config_.real_min_lead;
        if (lead_need > config_.max_lead) lead_need = config_.real_min_lead;
        last_reduction_reason_ = FgRealOnlyReason::LatencyCap;
        if (n == 0 && reason == FgRealOnlyReason::None) reason = FgRealOnlyReason::LatencyCap;
    }

    const bool consecutive_real_only = n == 0 && !last_interval_generated_;
    bool reanchor = !anchored_ || period_estimate == 0 || consecutive_real_only;
    bool latency_drain = false;
    FgTicks lead = 0;
    if (!reanchor) {
        FgTicks end = 0;
        if (FgCheckedAdd(last_real_target_, period_estimate, &end) != FgStatus::Ok ||
            FgCheckedAdd(end, -origin, &lead) != FgStatus::Ok)
            return FgStatus::Overflow;
        const bool over_lead = lead > config_.max_lead;
        const bool sustained_excess = over_lead && over_lead_last_interval_;
        over_lead_last_interval_ = over_lead;
        if (lead < config_.real_min_lead) {
            reanchor = true;  // behind: the real endpoint cannot be met on this timeline
            shortfall_run_ = 0;
        } else if (lead - config_.max_lead > period_estimate / 4 || sustained_excess) {
            // A small, isolated overrun can be ordinary jitter when the interpolation span is near the limit.
            // Act immediately on a large overrun and drain a small one only if it persists.
            // Re-anchor real-only to drain the excess lead and reset the cadence estimator so an L=1
            // frame-start-waitable caller cannot lock into an alternating LatencyCap limit cycle.
            reanchor = true;
            latency_drain = true;
            shortfall_run_ = 0;
            n = 0;
            lead_need = config_.real_min_lead;
            if (reason == FgRealOnlyReason::None) reason = FgRealOnlyReason::LatencyCap;
            last_reduction_reason_ = FgRealOnlyReason::LatencyCap;
        } else if (n > 0) {
            // N-B2: avoid hair-trigger reanchors on normal 2-3 ms jitter. Only reanchor immediately if index 1's
            // target would already fall inside bridge_and_submit_margin; for moderate lead shortfall beyond slack,
            // require 2 consecutive intervals so transient jitter slews cleanly via bounded lead correction.
            const FgTicks slot = period_estimate / (static_cast<FgTicks>(n) + 1);
            const FgTicks gen_span = lead_need - config_.first_image_budget;
            const FgTicks first_gen_lead = lead - gen_span;
            const FgTicks min_slack = period_estimate / 4 < 35000 ? period_estimate / 4 : 35000;
            const FgTicks slack = slot / 2 > min_slack ? slot / 2 : min_slack;
            if (first_gen_lead < config_.bridge_and_submit_margin) {
                reanchor = true;
                shortfall_run_ = 0;
            } else if (lead + slack < lead_need) {
                if (++shortfall_run_ >= 2) {
                    reanchor = true;
                    shortfall_run_ = 0;
                }
            } else {
                shortfall_run_ = 0;
            }
        } else {
            shortfall_run_ = 0;
        }
    } else {
        shortfall_run_ = 0;
    }

    FgTicks t_base = 0;
    FgTicks period = 0;
    if (reanchor) {
        over_lead_last_interval_ = false;
        for (;;) {
            FgTicks target = 0;
            FgTicks spacing = 0;
            FgTicks floor = 0;
            if (FgCheckedAdd(origin, lead_need, &target) != FgStatus::Ok ||
                !CheckedMul(static_cast<FgTicks>(n) + 1, w.min_interval, &spacing))
                return FgStatus::Overflow;
            const FgTicks previous_target = have_target_ ? last_real_target_ : frame.arrival;
            t_base = previous_target;
            if (n > 0) {
                // The previous real-only interval may have skipped all generation lead. Start a new cadence
                // after that endpoint, leaving enough time for the first generated image.
                const FgTicks base = target - period_estimate;
                if (base > t_base) t_base = base;
            }
            if (FgCheckedAdd(t_base, spacing, &floor) != FgStatus::Ok) return FgStatus::Overflow;
            if (target < floor) target = floor;
            period = target - t_base;
            if (n == 0) break;
            const FgTicks slot = period / (static_cast<FgTicks>(n) + 1);
            if (slot >= w.min_interval && slot <= w.max_interval &&
                slot >= origin - t_base + config_.bridge_and_submit_margin &&
                target - origin <= config_.max_lead &&
                (config_.fixed_refresh_period == 0 || FitsRaw(n, period)))
                break;
            --n;
            lead_need = needed_lead(n);
            if (lead_need > config_.max_lead) lead_need = config_.real_min_lead;
            last_reduction_reason_ = FgRealOnlyReason::Reanchor;
        }
        if (n == 0 && reason == FgRealOnlyReason::None) reason = FgRealOnlyReason::Reanchor;
        out->reanchored = !anchored_ || !consecutive_real_only;
        if (latency_drain) {
            anchored_ = false;
            estimator_.Reset();
            last_requested_ = 0;
            effective_ = 0;
            bad_run_ = 0;
            good_run_ = 0;
        } else {
            anchored_ = period_estimate > 0;
        }
        vrr_floor_run_ = 0;
    } else {
        // Bounded lead correction: keeps the lead near what the count needs without a visible step.
        // Excess lead (diff < 0, e.g. speedup or budget reduction) drains up to T/8 per interval so an L=1
        // waitable caller is not pinned in slow motion while excess lead bleeds out; upward lead additions
        // stay bounded at T/128 while generating so normal jitter does not stretch spacing.
        t_base = last_real_target_;
        const FgTicks diff = lead_need - lead;
        const FgTicks deadband = period_estimate / 16;
        FgTicks correction = 0;
        if (diff > deadband) {
            const FgTicks limit = n > 0 ? period_estimate / 128 : period_estimate / 8;
            correction = Clamp(diff / 8, 0, limit);
        } else if (diff < -deadband) {
            const FgTicks limit = diff < -period_estimate / 4 ? period_estimate / 4 : period_estimate / 8;
            correction = Clamp(diff / 2, -limit, 0);
        }
        period = period_estimate + correction;
        FgTicks floor = 0;
        if (!CheckedMul(static_cast<FgTicks>(n) + 1, w.min_interval, &floor)) return FgStatus::Overflow;
        if (config_.adaptive_count && config_.fixed_refresh_period == 0 && n > 0 && diff < -deadband && period < floor) {
            // Excess playout lead wants a shorter period than (n + 1) * min_interval allows; clamping at the VRR
            // floor would throttle an L=1 waitable caller to (n + 1) * min_interval forever. Require exit_intervals
            // consecutive frames so transient alternating jitter near the VRR ceiling does not step N down.
            if (++vrr_floor_run_ >= w.exit_intervals) {
                vrr_floor_run_ = 0;
                bad_run_ = 0;
                good_run_ = 0;
                while (n > 0) {
                    FgTicks f = 0;
                    if (!CheckedMul(static_cast<FgTicks>(n) + 1, w.min_interval, &f)) return FgStatus::Overflow;
                    if (period >= f) break;
                    --n;
                }
                effective_ = n;
                last_reduction_reason_ = FgRealOnlyReason::OutOfVrrWindow;
                if (n == 0 && reason == FgRealOnlyReason::None) reason = FgRealOnlyReason::OutOfVrrWindow;
                if (!CheckedMul(static_cast<FgTicks>(n) + 1, w.min_interval, &floor)) return FgStatus::Overflow;
            }
        } else {
            vrr_floor_run_ = 0;
        }
        if (period < floor) period = floor;
        const FgTicks feasible = origin + config_.real_min_lead - t_base;
        if (period < feasible) period = feasible;
        if (config_.fixed_refresh_period > 0 && n > 0 && !FitsRaw(n, period)) {
            if (period_estimate >= feasible && FitsRaw(n, period_estimate)) {
                period = period_estimate;
            } else {
                while (n > 0 && !FitsRaw(n, period)) --n;
                effective_ = n;
                bad_run_ = 0;
                good_run_ = 0;
                last_reduction_reason_ = FgRealOnlyReason::OutOfVrrWindow;
                if (n == 0 && reason == FgRealOnlyReason::None) reason = last_reduction_reason_;
            }
        }
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
    for (std::uint32_t k = 0; k < n; ++k) out->targets[k].admitted = Admissible(out->targets[k], origin);
    if ((s = FgValidatePlan(*out)) != FgStatus::Ok) return s;

    last_real_target_ = out->targets[n].target;
    have_target_ = true;
    last_interval_generated_ = n > 0;
    return FgStatus::Ok;
}

}  // namespace wgfg
