// SPDX-License-Identifier: MIT
// FNV-1a-64 of spec 08 section 4.1 (public FNV definition by Fowler, Noll and Vo).
#pragma once

#include <cstddef>
#include <cstdint>

namespace wgc {

constexpr std::uint64_t kFnvOffsetBasis = 14695981039346656037ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

inline std::uint64_t fnv1a64(const std::uint8_t* p, std::size_t n, std::uint64_t s = kFnvOffsetBasis) {
    for (std::size_t i = 0; i < n; ++i) s = (s ^ p[i]) * kFnvPrime;
    return s;
}

}  // namespace wgc
