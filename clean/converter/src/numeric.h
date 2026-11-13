// SPDX-License-Identifier: MIT
// Number formats and exact roundings of spec 08 section 2 and the decode operations of section 6.
// Written for this project from the IEEE 754 format definitions and the spec text.
#pragma once

#include <cstdint>

namespace wgc {

// value(h): the real number a binary16 bit pattern denotes (exact in binary64; NaN/inf preserved).
double half_value(std::uint16_t h);
bool half_is_finite(std::uint16_t h);

// RN16 / RN32: round a binary64 value to binary16 (bit pattern) or binary32, ties to even.
// Overflow gives infinity, NaN stays NaN. Neither depends on the current FPU rounding mode.
std::uint16_t rn16(double x);
float rn32(double x);

// q8(x) of section 2: nearest E4M3FN value, ties to even mantissa, saturating at +-448, zero -> +0.
// x must be finite and exactly representable in binary64 with at most 52 fraction bits below
// 2^-9 (true for every binary16 value).
float q8(double x);

// f16_scaled: RN32( RN16( RN32(x * mul) ) / div ) for a binary16 value x and binary32 mul, div.
float f16_scaled(double x, float mul, float div);

// RN32(e^x), correctly rounded, for x a binary16 value (any binary64 value with |x| < 2^20 works;
// the result is computed in double-double arithmetic, about 100 bits).
float exp_rn32(double x);

// capped_softmax of spec 48 section 6 on one group of 64 binary16 values x (s = 8 i + j): binary32 arithmetic
// with RN32 after every operation, correctly rounded exp, the sum in order s = 0..63, and
// y_s = RN16(RN32(value(RN16(E_s)) / sum)). Every x must be finite. A result that is not finite (a zero sum, an
// overflow) is returned as such; the caller rejects it.
void capped_softmax64(const double* x, float cap, float div, float sub, float* y);

float bits_to_float(std::uint32_t bits);
std::uint32_t float_to_bits(float f);
std::uint64_t double_to_bits(double d);

}  // namespace wgc
