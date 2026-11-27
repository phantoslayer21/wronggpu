// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

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

constexpr int64_t FirstImageBudgetCap(int64_t ready_lag, uint32_t latency_limit) {
    constexpr int64_t min_cap = 80000;   // 8 ms
    if (ready_lag <= 0) return kFirstImageBudgetMax;
    const uint32_t depth = std::clamp(latency_limit, 1u, 2u);
    return std::clamp(depth * ready_lag + 35000, min_cap, kFirstImageBudgetMax);
}

constexpr int64_t FirstImageBudgetFloor(int64_t minimum, int64_t readiness_p95, int64_t miss_floor) {
    const int64_t last_miss_with_margin = miss_floor > 0 ? miss_floor + 5000 : 0;
    return std::max({minimum, readiness_p95 + 3000, last_miss_with_margin});
}

struct FirstImageBudgetLimits {
    int64_t sampled_cap = kFirstImageBudgetMax;
    int64_t miss_floor = 0;

    void Observe(std::size_t count, int64_t readiness_p75, uint32_t latency_limit) {
        if (count >= 6) sampled_cap = FirstImageBudgetCap(readiness_p75, latency_limit);
    }

    void RecordMiss(int64_t budget) { miss_floor = std::max(miss_floor, budget); }
    int64_t EasingFloor(int64_t minimum, int64_t readiness_p95) const {
        return std::min(kFirstImageBudgetMax, FirstImageBudgetFloor(minimum, readiness_p95, miss_floor));
    }
    int64_t Cap(int64_t minimum, int64_t readiness_p95) const {
        return std::max(sampled_cap, EasingFloor(minimum, readiness_p95));
    }
};

}  // namespace wgfgp::win32
