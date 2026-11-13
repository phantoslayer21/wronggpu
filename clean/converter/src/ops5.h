// SPDX-License-Identifier: MIT
// The decode operations, value checks and attention-constant check of spec 57 section 6 (map version 5, preset F).
// Pure functions on bit patterns, so that they can be unit-tested against data/57-T5-vectors.json. Written for
// this project from spec 57 sections 2 and 6, spec 51 section 51.1 (Q8) and 51.3 steps 3-4.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "map.h"

namespace wgc {

// e4m3 code of Q8(value(h)) (spec 51.1, 57 section 2): nearest, ties to the even code, saturating at 448, the sign
// of zero kept (-0 and a negative value that rounds to zero give code 128). -1 for a non-finite h (an error).
int e4m3_code(std::uint16_t h);

// Spec 57 section 6.2. S holds H * 4096 binary16 bit patterns (the stored [H][64][64] tensor, row-major); a_bits and
// wd_bits are the binary32 bit patterns of the data words A and Wd. On success `out` holds the H * 12 * 256
// binary16 bit patterns of Tb [H][12][16][16]. Returns false (with `why`, never a value) on a non-finite stored
// value, intermediate or T.
bool score_bias_op(const std::uint16_t* S, int H, std::uint32_t a_bits, std::uint32_t wd_bits,
                   const std::vector<std::pair<int, int>>& pairs, std::vector<std::uint16_t>& out, std::string& why);

// The const operations of spec 57 section 6.1 on the words read at every listed site (all must be equal).
// `bits` receives the binary32 (const_u32_as_f32) or binary16 (the others) bit pattern. Returns false with `why`
// when the sites disagree, the halves of const_f16x2 differ, the high half of const_f16lo is not 0, or
// const_f32_to_f16 meets a non-finite value or an RN16 overflow. (The finite-output rule of 6.1 is applied by the
// caller to every float output, see output_is_finite.)
bool const_op(OpKind k, bool negate, const std::vector<std::uint32_t>& words, std::uint32_t& bits, std::string& why);

// Spec 57 section 6.4 with binary16 bit patterns. On success `wmax` is the binary16 pattern of WMAX and
// `may_overflow` tells whether 64 * WMAX >= 65520. Returns false with `why` if an input is not finite, the lower
// bound exceeds the upper one, or some clamped c gives y < 1 or y >= 31.
bool attn_wmax_op(std::uint16_t lo, std::uint16_t hi, std::uint16_t p1, std::uint16_t p0, std::uint16_t& wmax,
                  bool& may_overflow, std::string& why);

// The finite-output rule of 57 section 6.1 for one tensor's bytes (float16 and float32 only; true otherwise).
bool output_is_finite(DType d, const std::uint8_t* bytes, std::uint64_t count, std::uint64_t& first_bad);

// The `require` conditions of 57 section 6.3 (Require bits) for every element. Returns false and the first element
// that fails one, and which condition, in `why`.
bool check_require(DType d, const std::uint8_t* bytes, std::uint64_t count, unsigned req, std::string& why);

// The value of element 0 of a tensor (for the group checks): exact as binary64 (e4m3 decoded, integers as such).
double scalar_value(DType d, const std::uint8_t* bytes);

// One group check of 57 section 6.3 on exact values: ge, gt, le compare the real values; ratio_below_f16_overflow
// holds when b > 0 and |RN32(a * r+)| < 65520, r+ the binary32 number one ulp above RN32(1 / b).
bool check_relation(Rel rel, double a, double b);

}  // namespace wgc
