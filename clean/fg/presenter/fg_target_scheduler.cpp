// SPDX-License-Identifier: GPL-3.0-only
// Target-time scheduler: see fg_target_scheduler.h.
#include "fg_target_scheduler.h"

#include <algorithm>
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

FgTargetScheduler::FgTargetScheduler(const FgSchedulerConfig& config)
    : config_(config), estimator_(config.estimator), uncapped_estimator_(config.estimator) {
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
    uncapped_estimator_.Reset();
    uncapped_arrival_ = 0;
    cadence_arrival_ = previous_credit_wait_ = 0;
    playout_period_ = 0;
    protect_consumed_lead_ = false;
    fill_requested_ = fill_count_ = fill_overrun_ = 0;
    overflow_ready_ = false;
    overflow_cap_ = 0;
    anchored_ = false;
    have_target_ = false;
    last_real_target_ = 0;
    history_valid_ = false;
    last_interval_count_ = 0;
    last_requested_ = 0;
    effective_ = 0;
    bad_run_ = 0;
    good_run_ = 0;
    good_candidate_ = 0;
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
    // Ordinary controller adjustments still need consecutive shortfalls before they re-anchor.
    if (budget - config_.first_image_budget >= 80000) shortfall_run_ = 1;
    config_.first_image_budget = budget;
}

void FgTargetScheduler::SetMaxGeneratedCap(std::uint32_t cap) {
    if (cap > config_.max_generated) cap = config_.max_generated;
    max_generated_cap_ = cap;
}

void FgTargetScheduler::SetDisplayMinInterval(FgTicks min_interval) {
    config_.display.min_interval = min_interval < 1 ? 1 : min_interval;
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
    std::uint32_t up = effective_ < requested ? (adaptive ? effective_ + 1 : requested) : 0;
    if (up > 0) {
        // A fixed grid can support 2x and 4x with no legal 3x plan.
        if (adaptive && config_.fixed_refresh_period > 0) {
            while (up < requested && !Fits(up, period, w.hysteresis)) ++up;
        }
        if (Fits(up, period, w.hysteresis)) {
            if (good_candidate_ != up) {
                good_candidate_ = up;
                good_run_ = 0;
            }
            if (++good_run_ >= w.reenter_intervals) {
                effective_ = up;
                good_run_ = 0;
                good_candidate_ = 0;
                bad_run_ = 0;
            }
        } else {
            good_run_ = 0;
            good_candidate_ = 0;
        }
    }
    return effective_;
}

FgTargetScheduler::FillPlan FgTargetScheduler::UpdateFill(std::uint32_t requested, bool eligible,
                                                          FgTicks latest_uncapped) {
    const bool changed = requested != fill_requested_;
    fill_requested_ = requested;
    const FgTicks uncapped = uncapped_estimator_.Period();
    if (!eligible || changed || (config_.guard_overflow_delivery && !overflow_ready_) ||
        uncapped <= 0 || latest_uncapped <= 0) {
        fill_count_ = fill_overrun_ = 0;
        return {};
    }
    const auto& w = config_.display;
    const auto target_for = [&](std::uint32_t count) {
        FgTicks target = 0;
        if (!CheckedMul(static_cast<FgTicks>(count) + 1, w.min_interval + w.margin + w.hysteresis, &target))
            return FgTicks{0};
        return target;
    };
    const auto within_trim = [](FgTicks target, FgTicks period, FgTicks percent) {
        return target - period <= (period / 100) * percent + (period % 100) * percent / 100;
    };
    const FgTicks fastest = std::min(uncapped, latest_uncapped);
    const FgTicks slowest = std::max(uncapped, latest_uncapped);
    std::uint32_t limit = max_generated_cap_ ? std::min(requested, max_generated_cap_) : requested;
    if (overflow_cap_) limit = std::min(limit, overflow_cap_);
    if (fill_count_ != 0) {
        const FgTicks target = target_for(fill_count_);
        if (fill_count_ > limit || slowest >= target) {
            fill_count_ = fill_overrun_ = 0;
            return {};
        }
        if (!within_trim(target, fastest, 16)) {
            if (++fill_overrun_ >= 4) fill_count_ = fill_overrun_ = 0;
            return {};
        }
        fill_overrun_ = 0;
        // Retain the choice across jitter, while enforcing the 12% ceiling on every release.
        if (!within_trim(target, fastest, 12)) return {};
        return {fill_count_, target, target - slowest};
    }
    const auto count = Largest(limit, uncapped) + 1;
    if (count > limit) return {};
    const FgTicks target = target_for(count);
    if (target <= slowest || !within_trim(target, fastest, 12)) return {};
    fill_count_ = count;
    return {count, target, target - slowest};
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
    const FgTicks origin = std::max(frame.arrival, frame.budget_origin);
    const bool adaptive_vrr = config_.adaptive_count && config_.fixed_refresh_period == 0;
    FgTicks cadence_arrival = frame.arrival;
    const FgTicks arrival_delta = frame.arrival - last_arrival_;
    if (config_.first_present_admission) {
        FgTicks cadence_delta = arrival_delta;
        FgTicks floor = 0;
        // Near the spacing floor, first-present admission can pace the next arrival at the
        // old count's own period. Check that the sample with that wait removed still supports
        // generation. Far from this lock-in, retain the arrival cadence including natural waits.
        if (last_interval_count_ > 1 &&
            CheckedMul(static_cast<FgTicks>(last_interval_count_) + 1, w.min_interval, &floor) &&
            estimator_.Period() <= floor + floor / 16) {
            const auto without_wait = arrival_delta - Clamp(previous_credit_wait_, 0, arrival_delta);
            if (FitsRaw(1, without_wait)) cadence_delta = without_wait;
        }
        if (cadence_arrival_ > 0 &&
            FgCheckedAdd(cadence_arrival_, std::max<FgTicks>(1, cadence_delta), &cadence_arrival) != FgStatus::Ok)
            return FgStatus::Overflow;
        cadence_arrival_ = cadence_arrival;
        previous_credit_wait_ = origin - frame.arrival;
    }
    FgTicks latest_uncapped = 0;
    if (config_.overflow_fill && adaptive_vrr) {
        if (uncapped_arrival_ > 0) {
            latest_uncapped = arrival_delta - Clamp(frame.previous_release_wait, 0, arrival_delta);
            if (FgCheckedAdd(uncapped_arrival_, std::max<FgTicks>(1, latest_uncapped), &uncapped_arrival_) != FgStatus::Ok)
                return FgStatus::Overflow;
        } else {
            uncapped_arrival_ = frame.arrival;
        }
        uncapped_estimator_.Push(uncapped_arrival_);
    }
    const FgIntervalEstimator::Event ev = estimator_.Push(cadence_arrival);
    have_frame_ = true;
    last_frame_id_ = frame.frame_id;
    last_arrival_ = frame.arrival;
    if (ev == FgIntervalEstimator::Event::Discontinuity) anchored_ = false;
    FgTicks period_estimate = estimator_.Period();

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

    const FillPlan fill = UpdateFill(frame.requested_count,
                                     config_.overflow_fill && period_estimate > 0 &&
                                         reason == FgRealOnlyReason::None,
                                     latest_uncapped);
    period_estimate = std::max(period_estimate, fill.period);
    // Count for the display window (kept current while real-only, so hysteresis spans menus and suspensions).
    std::uint32_t effective = 0;
    if (frame.requested_count == 0) {
        last_requested_ = 0;
        effective_ = 0;
        bad_run_ = 0;
        good_run_ = 0;
    } else if (period_estimate > 0) {
        // Give an oversized request the same count history as an explicit fitting
        // request. Only an approved fill plan can add a count above the base fit.
        const auto requested = adaptive_vrr && config_.guard_overflow_delivery
                                   ? std::max(Largest(frame.requested_count, estimator_.Period()), fill.count)
                                   : frame.requested_count;
        effective = UpdateCount(requested, period_estimate);
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

    const bool track_playout = config_.consumed_admission && adaptive_vrr && n > 0 &&
                               period_estimate / 4 < w.min_interval;
    const bool protect_lead = track_playout && protect_consumed_lead_;
    if (track_playout) {
        const auto nominal = period_estimate;
        if (playout_period_ == 0 || ev == FgIntervalEstimator::Event::Discontinuity) {
            playout_period_ = nominal;
        } else if (arrival_delta >= nominal / 2 && arrival_delta <= nominal * 2) {
            // Track elapsed time, including ordinary slow-tail samples that the median
            // ignores. Qualification above still uses the median to select the count.
            playout_period_ += (arrival_delta - playout_period_) / 16;
        }
        const auto floor = (static_cast<FgTicks>(n) + 1) * w.min_interval;
        period_estimate = std::max({floor, fill.period,
                                    Clamp(playout_period_, nominal - nominal / 8, nominal + nominal / 8)});
    } else {
        playout_period_ = 0;
    }

    const auto needed_lead = [&](std::uint32_t count) {
        const auto budgeted = RequiredLead(period_estimate, count == 0 && last_interval_count_ > 0 ? effective : count);
        // A little headroom at high base rates absorbs the wake and bridge jitter left
        // after producer readiness. Admission still releases on consumption.
        if (!protect_lead || count == 0) return budgeted;
        const auto reserve = std::min<FgTicks>(10000, period_estimate / 8);
        return budgeted > kTicksMax - reserve ? kTicksMax : budgeted + reserve;
    };

    FgTicks lead_need = needed_lead(n);
    if (lead_need > config_.max_lead) {
        while (n > 0 && needed_lead(n) > config_.max_lead) --n;
        lead_need = n > 0 ? needed_lead(n) : config_.real_min_lead;
        if (lead_need > config_.max_lead) lead_need = config_.real_min_lead;
        last_reduction_reason_ = FgRealOnlyReason::LatencyCap;
        if (n == 0 && reason == FgRealOnlyReason::None) reason = FgRealOnlyReason::LatencyCap;
    }

    const bool consecutive_real_only = n == 0 && last_interval_count_ == 0;
    FgReanchorReason anchor_reason = FgReanchorReason::None;
    if (!anchored_ || period_estimate == 0 || consecutive_real_only) {
        anchor_reason = ev == FgIntervalEstimator::Event::Discontinuity ? FgReanchorReason::CadenceChange
                                                                     : FgReanchorReason::Initial;
    }
    FgTicks lead = 0;
    if (anchor_reason == FgReanchorReason::None) {
        FgTicks end = 0;
        if (FgCheckedAdd(last_real_target_, period_estimate, &end) != FgStatus::Ok ||
            FgCheckedAdd(end, -origin, &lead) != FgStatus::Ok)
            return FgStatus::Overflow;
        const bool over_lead = lead > config_.max_lead;
        const bool sustained_excess = over_lead && over_lead_last_interval_;
        over_lead_last_interval_ = over_lead;
        if (lead < config_.real_min_lead) {
            anchor_reason = FgReanchorReason::RealTooLate;
            shortfall_run_ = 0;
        } else if (lead - config_.max_lead > period_estimate / 4 || sustained_excess) {
            // A small, isolated overrun can be ordinary jitter when the interpolation span is near the limit.
            // Act immediately on a large overrun and drain a small one only if it persists.
            // Re-anchor real-only to drain the excess lead and reset the cadence estimator so an L=1
            // frame-start-waitable caller cannot lock into an alternating LatencyCap limit cycle.
            anchor_reason = FgReanchorReason::ExcessLead;
            shortfall_run_ = 0;
            n = 0;
            lead_need = config_.real_min_lead;
            if (reason == FgRealOnlyReason::None) reason = FgRealOnlyReason::LatencyCap;
            last_reduction_reason_ = FgRealOnlyReason::LatencyCap;
        } else if (n > 0) {
            // A missed submit margin still reanchors immediately. The soft VRR check tolerates
            // transient producer jitter and waits for three consecutive shortfalls.
            const FgTicks slot = period_estimate / (static_cast<FgTicks>(n) + 1);
            const FgTicks gen_span = RequiredLead(period_estimate, n) - config_.first_image_budget;
            const FgTicks first_gen_lead = lead - gen_span;
            const FgTicks min_slack = std::min(period_estimate / 4, adaptive_vrr ? FgTicks{25000} : FgTicks{35000});
            const FgTicks slack = adaptive_vrr ? Clamp(slot / 2, min_slack, 35000) : std::max(slot / 2, min_slack);
            if (first_gen_lead < config_.bridge_and_submit_margin) {
                anchor_reason = FgReanchorReason::GeneratedDeadline;
                shortfall_run_ = 0;
            } else if (lead + slack < lead_need) {
                if (track_playout) shortfall_run_ = 1;
                else if (++shortfall_run_ >= (adaptive_vrr ? 3u : 2u)) {
                    anchor_reason = FgReanchorReason::LeadShortfall;
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
    if (anchor_reason != FgReanchorReason::None) {
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
        if (out->reanchored) out->reanchor_reason = anchor_reason;
        if (anchor_reason == FgReanchorReason::ExcessLead) {
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
        // waitable caller is not pinned in slow motion while excess lead bleeds out. Ordinary upward
        // corrections stay at T/128 while generating. Near the VRR ceiling, a single generated
        // image can recover from a soft shortfall by up to T/16 before a third miss would re-anchor.
        t_base = last_real_target_;
        const FgTicks diff = lead_need - lead;
        const FgTicks deadband = period_estimate / 16;
        FgTicks correction = 0;
        if (diff > deadband) {
            const bool recover_single = adaptive_vrr && n == 1 && period_estimate / 3 < w.min_interval &&
                                        shortfall_run_ > 0;
            const bool recover_playout = track_playout && shortfall_run_ > 0;
            const FgTicks limit = protect_lead ? period_estimate / 4 :
                                  n == 0 || recover_playout ? period_estimate / 8 :
                                  period_estimate / (recover_single ? 16 : 128);
            correction = Clamp(diff / (protect_lead ? 2 : 8), 0, limit);
        } else if (diff < -deadband) {
            const FgTicks limit = diff < -period_estimate / 4 ? period_estimate / 4 : period_estimate / 8;
            correction = Clamp(diff / (protect_lead ? 8 : 2), -limit, 0);
        }
        period = period_estimate + correction;
        FgTicks floor = 0;
        if (!CheckedMul(static_cast<FgTicks>(n) + 1, w.min_interval, &floor)) return FgStatus::Overflow;
        // A lead correction cannot invalidate a count the estimated cadence still supports.
        if (adaptive_vrr && n > 0 && diff < -deadband && period < floor && !FitsRaw(n, period_estimate)) {
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
    if (n != 0 && n == fill.count) {
        out->fill_period = fill.period;
        out->fill_release_wait = fill.wait;
    }
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
    last_interval_count_ = n;
    return FgStatus::Ok;
}

}  // namespace wgfg
