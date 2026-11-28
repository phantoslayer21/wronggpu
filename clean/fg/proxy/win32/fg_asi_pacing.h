// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#include "../../presenter/fg_presenter.h"

namespace wgfgp::win32 {

// Readiness samples are paired by frame and their order defines the newest window.
inline int64_t Percentile(std::vector<int64_t> v, double pct) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    const auto idx = static_cast<std::size_t>(pct * static_cast<double>(v.size() - 1));
    return v[std::min(idx, v.size() - 1)];
}

template <std::size_t Capacity>
class PacingSamples {
public:
    void Append(int64_t sample) {
        if (size_ < Capacity) samples_[size_++] = sample;
    }
    bool Empty() const { return size_ == 0; }
    std::size_t Size() const { return size_; }

    int64_t Percentile(double pct, std::size_t newest = Capacity) const {
        const std::size_t count = std::min(size_, newest);
        if (count == 0) return 0;
        auto sorted = samples_;
        const auto first = sorted.begin() + (size_ - count);
        std::sort(first, sorted.begin() + size_);
        const auto idx = static_cast<std::size_t>(pct * static_cast<double>(count - 1));
        return *(first + idx);
    }

private:
    std::array<int64_t, Capacity> samples_ = {};
    std::size_t size_ = 0;
};

constexpr int64_t kFirstImageBudgetMax = 320000;  // 32 ms
constexpr int64_t kFirstImageMissMargin = 5000;

constexpr std::size_t kPresenterAdmissionDepth = 2;  // head and one following interval

constexpr bool PresenterAdmissionAtCapacity(std::size_t intervals, std::size_t slots, bool has_consumed_interval) {
    return intervals >= slots || (has_consumed_interval && intervals >= kPresenterAdmissionDepth);
}

constexpr int64_t FirstImageBudgetCap(int64_t ready_lag, uint32_t latency_limit) {
    constexpr int64_t min_cap = 80000;   // 8 ms
    if (ready_lag <= 0) return kFirstImageBudgetMax;
    const uint32_t depth = std::clamp(latency_limit, 1u, 2u);
    return std::clamp(depth * ready_lag + 35000, min_cap, kFirstImageBudgetMax);
}

constexpr int64_t FirstImageBudgetFloor(int64_t minimum, int64_t readiness_p95, int64_t miss_floor) {
    const int64_t last_miss_with_margin = miss_floor > 0 ? miss_floor + kFirstImageMissMargin : 0;
    return std::max({minimum, readiness_p95 + 3000, last_miss_with_margin});
}

struct FirstImageBudgetLimits {
    int64_t sampled_cap = kFirstImageBudgetMax;
    int64_t miss_floor = 0;
    uint32_t clean_windows = 0;

    void Observe(std::size_t count, int64_t readiness_p75, uint32_t latency_limit) {
        if (count >= 6) sampled_cap = FirstImageBudgetCap(readiness_p75, latency_limit);
    }

    void RecordMiss(int64_t budget) {
        miss_floor = std::max(miss_floor, budget);
        BudgetRaised();
    }
    void BudgetRaised() { sampled_cap = kFirstImageBudgetMax; }
    void TrackCleanWindow(bool full, bool clean, int64_t minimum, int64_t readiness_p95) {
        if (!clean || (full && readiness_p95 <= 0)) {
            clean_windows = 0;
            return;
        }
        if (!full || ++clean_windows < 3) return;
        clean_windows = 0;
        const int64_t readiness_floor = FirstImageBudgetFloor(minimum, readiness_p95, 0);
        const int64_t target = std::max<int64_t>(0, readiness_floor - kFirstImageMissMargin);
        if (miss_floor > target) miss_floor = std::max(target, miss_floor - 5000);  // at most 0.5 ms
    }
    int64_t EasingFloor(int64_t minimum, int64_t readiness_p95) const {
        return std::min(kFirstImageBudgetMax, FirstImageBudgetFloor(minimum, readiness_p95, miss_floor));
    }
    int64_t Cap(int64_t minimum, int64_t readiness_p95, bool producer_miss_window = false) const {
        if (producer_miss_window) return kFirstImageBudgetMax;
        return std::max(sampled_cap, EasingFloor(minimum, readiness_p95));
    }
};

constexpr int64_t ApplySampledBudgetCap(int64_t proposed, int64_t cap, int64_t current, int64_t easing_step) {
    return std::min(proposed, std::max(cap, current - easing_step));
}

constexpr int64_t FirstImageFastEaseStepCap(int64_t budget, bool consumed_admission, uint32_t scheduled) {
    // A first-present gate can hide the producer tail during a high-count downward step.
    if (!consumed_admission && scheduled >= 4 && budget <= 205000) return 10000;
    return budget > 225000 ? 55000 : (budget > 165000 ? 25000 : (budget > 135000 ? 17000 : 10000));
}

constexpr int64_t FirstImageReadinessSlack(int64_t budget, bool consumed_admission, uint32_t scheduled) {
    if (consumed_admission) return std::max<int64_t>(35000, budget / 2);
    // At N=1 there is more space between generated images. A clean window can probe that headroom
    // while the p95 readiness floor and the existing 1 ms near-budget easing limit still apply.
    if (scheduled == 1) return std::clamp<int64_t>(budget / 2, 35000, 60000);
    return 35000;
}

struct FirstImageProducerWindow {
    uint64_t scheduled;
    uint64_t missed;

    bool NeedsBudgetReview() const { return missed * 50 > scheduled; }
    bool ShouldRaiseBudget() const { return missed >= 2 && NeedsBudgetReview(); }
    uint32_t ReducedCount(uint32_t last_generating_count) const {
        return missed >= 2 && missed * 20 > scheduled && last_generating_count > 1 ? last_generating_count - 1 : 0;
    }
};

inline FirstImageProducerWindow ProducerWindow(const wgfg::FgPresentStats& now, const wgfg::FgPresentStats& base) {
    return {now.generated_scheduled - base.generated_scheduled, now.generated_late_pre_queue - base.generated_late_pre_queue};
}

// Output oversubscription is independent of the producer budget. A presenter miss does
// not buy more producer lead, and a clean window only earns a higher count with measured slack.
struct OverflowDelivery {
    uint32_t cap = 0;
    uint32_t clean_windows = 0;
    uint32_t bad_windows = 0;
    bool ready = false;

    void Observe(uint64_t offered, uint64_t missed, uint32_t active, uint32_t requested, bool overflow,
                 bool has_headroom, bool full_window) {
        if (offered == 0 || active == 0) return;
        if (missed * 100 >= offered) {
            ready = false;
            clean_windows = 0;
            const bool severe = missed >= 3 && missed * 100 >= offered * 8;
            if (full_window) ++bad_windows;
            if (overflow && active > 1 && (severe || bad_windows >= 2)) {
                const uint32_t next = active - 1;
                if (cap == 0 || next < cap) {
                    cap = next;
                    bad_windows = 0;
                }
            }
        } else if (full_window) {
            bad_windows = 0;
            if (!has_headroom) {
                ready = false;
                clean_windows = 0;
            } else if (++clean_windows >= 4) {
                clean_windows = 0;
                if (cap != 0) cap = cap + 1 >= requested ? 0 : cap + 1;
                ready = true;
            }
        }
    }
};

inline bool HasActiveProducerTail(bool consumed, std::size_t samples, int64_t readiness_p95, int64_t budget) {
    return consumed && samples >= 6 && readiness_p95 + 12500 >= budget;
}

inline void ResetFirstImageBudgetWindow(FirstImageBudgetLimits& limits, wgfg::FgPresentStats& baseline,
                                      const wgfg::FgPresentStats& current) {
    limits = {};
    baseline = current;
}

// Runtime health check for Windows composition (HWFQ) safety-net fallback to DXGI.
// Triggers only when the bound window is visible and composition has observed zero displayed frames across sustained
// submissions or an admission timeout past startup grace.
enum class HwfqFallbackReason : uint8_t {
    None = 0,
    ZeroDisplaySustained = 1,
    AdmissionTimeoutZeroDisplay = 2,
};

struct HwfqHealthSnapshot {
    uint64_t displayed = 0;            // displayed_real + displayed_generated
    uint64_t visible_submitted = 0;    // presents submitted while the bound window was visible
    uint64_t resolved_unseen = 0;      // display_skipped_real + display_skipped_generated
    uint64_t watchdog_retired = 0;     // queued presents retired by the 250 ms occlusion watchdog
    wgfg::FgTicks visible_elapsed_ticks = 0;  // ticks since the first visible-window present submission
    bool admission_timed_out = false;
    bool startup_grace = false;
};

constexpr uint64_t kHwfqFallbackMinVisibleSubmits = 30;
constexpr uint64_t kHwfqFallbackMinUnseen = 15;
constexpr uint64_t kHwfqFallbackMinWatchdog = 2;
constexpr uint64_t kHwfqFallbackTimeoutMinVisibleSubmits = 4;
constexpr wgfg::FgTicks kHwfqFallbackMinElapsedTicks = 7500000;  // 750 ms

constexpr HwfqFallbackReason CheckHwfqZeroDisplayFallback(const HwfqHealthSnapshot& h) {
    if (h.displayed > 0) return HwfqFallbackReason::None;
    if (h.admission_timed_out && !h.startup_grace && h.visible_submitted >= kHwfqFallbackTimeoutMinVisibleSubmits)
        return HwfqFallbackReason::AdmissionTimeoutZeroDisplay;
    if (h.visible_submitted >= kHwfqFallbackMinVisibleSubmits &&
        h.visible_elapsed_ticks >= kHwfqFallbackMinElapsedTicks &&
        (h.resolved_unseen >= kHwfqFallbackMinUnseen || h.watchdog_retired >= kHwfqFallbackMinWatchdog))
        return HwfqFallbackReason::ZeroDisplaySustained;
    return HwfqFallbackReason::None;
}

}  // namespace wgfgp::win32
