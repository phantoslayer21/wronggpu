// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <algorithm>
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

constexpr int64_t FirstImageBudgetCap(int64_t ready_lag, uint32_t latency_limit) {
    constexpr int64_t min_cap = 80000;   // 8 ms
    constexpr int64_t max_cap = 320000;  // 32 ms
    if (ready_lag <= 0) return max_cap;  // no sample unaffected by admission yet
    const uint32_t depth = std::clamp(latency_limit, 1u, 2u);
    return std::clamp(depth * ready_lag + 35000, min_cap, max_cap);
}

constexpr int64_t FirstImageBudgetFloor(int64_t minimum, int64_t readiness_p95, int64_t miss_floor) {
    const int64_t last_miss_with_margin = miss_floor > 0 ? miss_floor + 5000 : 0;
    return std::max({minimum, readiness_p95 + 3000, last_miss_with_margin});
}

}  // namespace wgfgp::win32
