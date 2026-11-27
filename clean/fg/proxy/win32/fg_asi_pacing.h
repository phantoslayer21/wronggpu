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
        return missed * 20 > scheduled && last_generating_count > 1 ? last_generating_count - 1 : 0;
    }
};

inline FirstImageProducerWindow ProducerWindow(const wgfg::FgPresentStats& now, const wgfg::FgPresentStats& base) {
    return {now.generated_scheduled - base.generated_scheduled, now.generated_late_pre_queue - base.generated_late_pre_queue};
}

inline void ResetFirstImageBudgetWindow(FirstImageBudgetLimits& limits, wgfg::FgPresentStats& baseline,
                                      const wgfg::FgPresentStats& current) {
    limits = {};
    baseline = current;
}

}  // namespace wgfgp::win32
