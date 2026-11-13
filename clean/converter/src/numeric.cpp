// SPDX-License-Identifier: MIT
// Exact roundings for spec 08 sections 2 and 6. Written for this project.
//
// All roundings are done with explicit integer/power-of-two arithmetic on binary64 values that are
// exact, so the results do not depend on the FPU rounding mode or on excess precision. The only
// binary64 operations whose result is rounded are the division in f16_scaled (see the comment there)
// and the double-double steps in exp_rn32 (error analysis there).
#include "numeric.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace wgc {

float bits_to_float(std::uint32_t bits) {
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

std::uint32_t float_to_bits(float f) {
    std::uint32_t b;
    std::memcpy(&b, &f, 4);
    return b;
}

std::uint64_t double_to_bits(double d) {
    std::uint64_t b;
    std::memcpy(&b, &d, 8);
    return b;
}

namespace {

// Round a non-negative binary64 integer-or-fraction v < 2^52 to the nearest integer, ties to even.
double round_half_even(double v) {
    const double f = std::floor(v);
    const double d = v - f;  // exact for v < 2^52
    if (d > 0.5) return f + 1.0;
    if (d < 0.5) return f;
    return (std::fmod(f, 2.0) == 0.0) ? f : f + 1.0;
}

// Exponent E with 2^E <= a < 2^(E+1), for finite a > 0.
int binade(double a) {
    int e;
    std::frexp(a, &e);  // a = m * 2^e, m in [0.5, 1)
    return e - 1;
}

// Round a finite binary64 magnitude a >= 0 to a binary floating format with `frac_bits` fraction bits
// and minimum normal exponent `emin`. Returns the integer significand k and the exponent of its
// unit, so that the rounded value is k * 2^unit_exp. Overflow is left to the caller.
void round_to_format(double a, int frac_bits, int emin, double& k, int& unit_exp) {
    int e = (a < std::ldexp(1.0, emin)) ? emin : binade(a);
    unit_exp = e - frac_bits;
    k = round_half_even(std::ldexp(a, -unit_exp));  // scaling by a power of two is exact here
}

}  // namespace

double half_value(std::uint16_t h) {
    const int sign = h >> 15;
    const int exp = (h >> 10) & 0x1F;
    const int man = h & 0x3FF;
    double v;
    if (exp == 0x1F) {
        v = man ? std::numeric_limits<double>::quiet_NaN() : std::numeric_limits<double>::infinity();
    } else if (exp == 0) {
        v = std::ldexp(static_cast<double>(man), -24);
    } else {
        v = std::ldexp(static_cast<double>(man + 1024), exp - 25);
    }
    return sign ? -v : v;
}

bool half_is_finite(std::uint16_t h) { return ((h >> 10) & 0x1F) != 0x1F; }

std::uint16_t rn16(double x) {
    if (std::isnan(x)) return 0x7E00;
    const std::uint16_t sign = std::signbit(x) ? 0x8000 : 0;
    const double a = std::fabs(x);
    // Largest finite 65504 = (2 - 2^-10) * 2^15; the midpoint to 2^16 is 65520 and ties go to the
    // even significand, which is 2^16, i.e. infinity.
    if (a >= 65520.0) return sign | 0x7C00;
    if (a == 0.0) return sign;
    double k;
    int unit;
    round_to_format(a, 10, -14, k, unit);
    int biased = unit + 10 + 15;  // exponent of the leading bit, biased
    if (a < std::ldexp(1.0, -14)) {
        // Subnormal: k * 2^-24 with k <= 1024 (k == 1024 is the smallest normal, encoded the same way).
        return sign | static_cast<std::uint16_t>(k);
    }
    if (k == 2048.0) {
        k = 1024.0;
        ++biased;
    }
    return sign | static_cast<std::uint16_t>((biased << 10) | (static_cast<int>(k) - 1024));
}

float rn32(double x) {
    if (std::isnan(x)) return std::numeric_limits<float>::quiet_NaN();
    const std::uint32_t sign = std::signbit(x) ? 0x80000000u : 0u;
    const double a = std::fabs(x);
    // Largest finite (2 - 2^-23) * 2^127; midpoint to 2^128 is 2^128 - 2^103; ties go to infinity.
    if (a >= std::ldexp(1.0, 128) - std::ldexp(1.0, 103)) return bits_to_float(sign | 0x7F800000u);
    if (a == 0.0) return bits_to_float(sign);
    double k;
    int unit;
    round_to_format(a, 23, -126, k, unit);
    if (a < std::ldexp(1.0, -126)) return bits_to_float(sign | static_cast<std::uint32_t>(k));
    int biased = unit + 23 + 127;
    if (k == 16777216.0) {
        k = 8388608.0;
        ++biased;
    }
    return bits_to_float(sign | (static_cast<std::uint32_t>(biased) << 23) |
                         (static_cast<std::uint32_t>(k) - 8388608u));
}

float q8(double x) {
    const double a = std::fabs(x);
    double q;
    if (a >= 448.0) {
        q = 448.0;  // saturation (values in (448, 464] would round to 448 anyway)
    } else if (a == 0.0) {
        q = 0.0;
    } else {
        // E4M3FN: 3 fraction bits, minimum normal exponent -6 (bias 7), subnormal unit 2^-9.
        double k;
        int unit;
        round_to_format(a, 3, -6, k, unit);
        q = std::ldexp(k, unit);  // k == 16 moves to the next binade; that value is still exact
        if (q > 448.0) q = 448.0;
    }
    if (q == 0.0) return 0.0f;  // a zero result is +0
    return static_cast<float>(std::signbit(x) ? -q : q);  // exact: q has at most 4 significant bits
}

float f16_scaled(double x, float mul, float div) {
    // RN32(x * mul): x has 11 significant bits and mul 24, so the binary64 product is exact.
    const float p = rn32(x * static_cast<double>(mul));
    const double h = half_value(rn16(p));
    // RN32(h / div): the binary64 quotient is correctly rounded (IEEE 754), and rounding it again to
    // binary32 gives the correctly rounded binary32 quotient because 53 >= 2 * 24 + 2 (double
    // rounding is innocuous for division under that condition; S. Figueroa, "When is double rounding
    // innocuous?", SIGNUM Newsletter 30(3), 1995).
    return rn32(h / static_cast<double>(div));
}

// ---- exp in double-double arithmetic ------------------------------------------------------------
//
// A double-double (hi, lo) represents hi + lo with |lo| <= ulp(hi)/2. The algorithms are the
// classical error-free transformations (Knuth's TwoSum, Dekker's split and product), which need only
// correctly rounded binary64 + - * and no fused multiply-add.

namespace {

struct DD {
    double hi, lo;
};

inline DD two_sum(double a, double b) {
    const double s = a + b;
    const double bb = s - a;
    const double err = (a - (s - bb)) + (b - bb);
    return {s, err};
}

inline DD quick_two_sum(double a, double b) {  // |a| >= |b|
    const double s = a + b;
    return {s, b - (s - a)};
}

inline void split(double a, double& hi, double& lo) {
    const double t = 134217729.0 * a;  // 2^27 + 1
    hi = t - (t - a);
    lo = a - hi;
}

inline DD two_prod(double a, double b) {
    const double p = a * b;
    double ah, al, bh, bl;
    split(a, ah, al);
    split(b, bh, bl);
    const double err = ((ah * bh - p) + ah * bl + al * bh) + al * bl;
    return {p, err};
}

inline DD dd_add(DD a, DD b) {
    DD s = two_sum(a.hi, b.hi);
    DD t = two_sum(a.lo, b.lo);
    s.lo += t.hi;
    s = quick_two_sum(s.hi, s.lo);
    s.lo += t.lo;
    return quick_two_sum(s.hi, s.lo);
}

inline DD dd_mul(DD a, DD b) {
    DD p = two_prod(a.hi, b.hi);
    p.lo += a.hi * b.lo + a.lo * b.hi;
    return quick_two_sum(p.hi, p.lo);
}

inline DD dd_neg(DD a) { return {-a.hi, -a.lo}; }

inline DD dd_div_small_int(DD a, double n) {  // n a small positive integer (exact in binary64)
    // Long division: three binary64 quotient digits, each remainder computed exactly.
    const double q1 = a.hi / n;
    DD r = dd_add(a, dd_neg(two_prod(q1, n)));
    const double q2 = r.hi / n;
    r = dd_add(r, dd_neg(two_prod(q2, n)));
    const double q3 = r.hi / n;
    return dd_add(quick_two_sum(q1, q2), DD{q3, 0.0});
}

// ln 2 = LN2_HI + LN2_LO with an error below 2^-110 (computed with 60 decimal digits).
const double LN2_HI = 0x1.62e42fefa39efp-1;
const double LN2_LO = 0x1.abc9e3b39803fp-56;

}  // namespace

float exp_rn32(double x) {
    if (std::isnan(x)) return std::numeric_limits<float>::quiet_NaN();
    if (x > 100.0) return std::numeric_limits<float>::infinity();  // e^88.73 already exceeds binary32
    if (x < -110.0) return 0.0f;                                   // below 2^-151: rounds to +0
    // x = k ln2 + r with |r| <= ln2/2 (+ tiny), computed in double-double.
    const double k = std::nearbyint(x / LN2_HI);
    DD kl = dd_add(two_prod(k, LN2_HI), two_prod(k, LN2_LO));
    DD r = dd_add(DD{x, 0.0}, dd_neg(kl));
    // e^r by its Taylor series; |r| < 0.35, 30 terms leave a truncation error below 2^-140.
    DD sum{1.0, 0.0};
    DD term{1.0, 0.0};
    for (int n = 1; n <= 30; ++n) {
        term = dd_div_small_int(dd_mul(term, r), static_cast<double>(n));
        sum = dd_add(sum, term);
    }
    // Scale by 2^k (exact: the result stays in the binary64 normal range for |x| <= 110).
    const int ki = static_cast<int>(k);
    const double hi = std::ldexp(sum.hi, ki);
    const double lo = std::ldexp(sum.lo, ki);
    // Round hi + lo to binary32. Every binary32 rounding boundary (midpoint) in this range is a
    // binary64 number and |lo| <= ulp(hi)/2, so hi + lo lies on the same side of every boundary as
    // hi, except when hi is itself a boundary; then the sign of lo decides.
    const float f = rn32(hi);
    const double fd = static_cast<double>(f);
    if (fd != hi && lo != 0.0) {
        // Is hi exactly halfway between f and its neighbour on the other side of hi?
        const float g = std::nextafter(f, hi > fd ? std::numeric_limits<float>::infinity() : 0.0f);
        const double mid = (fd + static_cast<double>(g)) * 0.5;  // exact (binary32 values, scaled)
        if (mid == hi) {
            // Tie on hi alone was broken to even; the true value lies strictly on lo's side.
            return (lo > 0.0) == (static_cast<double>(g) > fd) ? g : f;
        }
    }
    return f;
}

// Each binary32 operation is done in binary64 and rounded once more with rn32. For +, -, * and / of binary32
// operands that second rounding gives the correctly rounded binary32 result (53 >= 2 * 24 + 2; Figueroa, cited
// at f16_scaled), and products of two binary32 values are exact in binary64.
void capped_softmax64(const double* x, float cap, float div, float sub, float* y) {
    const double c = cap, dv = div, sb = sub;
    double e[64];
    double sum = 0.0;
    for (int s = 0; s < 64; ++s) {
        const double u = std::min(std::max(x[s], -c), c);  // a binary16 value or +-cap: exact in binary32
        double a = rn32(u * u);
        a = rn32(a / dv);
        a = rn32(1.0 - a);
        a = rn32(a * u);
        const double g = rn32(a - sb);
        e[s] = exp_rn32(g);
    }
    for (int s = 0; s < 64; ++s) sum = rn32(sum + e[s]);
    for (int s = 0; s < 64; ++s) {
        const double q = rn32(half_value(rn16(e[s])) / sum);
        y[s] = static_cast<float>(half_value(rn16(q)));  // exact: a binary16 value (or inf / NaN)
    }
}

}  // namespace wgc
