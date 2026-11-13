// SPDX-License-Identifier: MIT
// Model K filter-shape table (spec 43 draft 2 section 7): the table T the output stage reads (spec 43 section 5.1),
// computed at initialisation, and the level the frame selects. C++ mirror of clean/k/reference/outstage.py
// filter_level / filter_table (clean/k/tests/test_k_runtime_tables.py checks that both give the same bits).
//
// Every operation is binary32 with round-to-nearest-even, in the order of the spec; this file must be compiled
// without floating-point contraction (MSVC /fp:precise, clang -ffp-contract=off; the .cpp also sets the pragmas).
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace wgrt {

constexpr int kFilterEntries = 2048;  // per set: 256 (F0 + 2 F1 + 4 F2) + 16 i + j
constexpr int kFilterValues = 8;      // per entry: (A, Cxx, Bxy, shear, O, K, kappa6, 0), FP16 bit patterns

// Spec 43 section 7: l = clamp(R(7 (sqrt(1 / ((1/2 (r_h RN32(1/H))) (1/2 (r_w RN32(1/W))))) - 1) / 3), 0, 7) in
// FP32 steps, R = round half away from zero. rw x rh render size, W x H output size. A NaN result (only for zero
// sizes) gives 0.
int k_filter_level(int rw, int rh, int W, int H);

// Set 2 level + (1 - theta) of the table: 2048 entries x 8 FP16 bit patterns, entry-major (index 8 e + c). Model K
// uses theta = 1 (the even sets); theta = 0 is the other variant of section 7.4. level in 0..7.
std::vector<uint16_t> k_filter_table(int level, int theta = 1);

// Same, into a caller buffer of kFilterEntries * kFilterValues values.
void k_filter_table(int level, int theta, uint16_t* out);

// Spec 43 section 7.8: binary32 -> binary16 by truncation toward zero; a magnitude above 65504 becomes +-infinity,
// NaN a quiet NaN (sign kept).
uint16_t k_f32_to_f16_trunc(float x);

}  // namespace wgrt
