// SPDX-License-Identifier: MIT
// The decode operations and value checks of spec 57 section 6. Written for this project.
//
// Exactness: binary16 and binary32 values are exact in binary64. Sums and products of two binary16 values, and
// products of two binary32 values, are exact in binary64. Where a binary64 result is not exact (the square root, a
// sum of operands far apart in magnitude, a quotient), the rounding to the narrow format is either innocuous double
// rounding (53 >= 2p + 2 for +, -, *, /, sqrt of narrow-format operands; S. Figueroa, "When is double rounding
// innocuous?", SIGNUM Newsletter 30(3), 1995) or is done from an exact two-term sum (rn16_exact_sum below).
#include "ops5.h"

#include <cmath>
#include <cstring>
#include <limits>

#include "numeric.h"

namespace wgc {

int e4m3_code(std::uint16_t h) {
    if (!half_is_finite(h)) return -1;
    const int sign = (h & 0x8000) ? 128 : 0;
    const double q = std::fabs(static_cast<double>(q8(half_value(h))));  // nearest, ties even, saturating; +-0 -> +0
    if (q == 0.0) return sign;  // the sign of zero is kept (57 section 2)
    int e2;
    const double fr = std::frexp(q, &e2);  // q = fr * 2^e2, fr in [0.5, 1)
    const int unb = e2 - 1;                // floor(log2 q)
    if (unb < -6) return sign | static_cast<int>(std::ldexp(q, 9));  // subnormal: m * 2^-9, m in 1..7
    const int m = static_cast<int>(std::ldexp(fr, 4)) - 8;          // q = (1 + m/8) * 2^unb
    return sign | ((unb + 7) << 3) | m;
}

namespace {

std::uint16_t half_bits_of(double v) { return rn16(v); }  // v exact in binary16

// RN16(s + e) for an exact two-term value s + e with |e| <= ulp(s)/2 (s = RN64(s + e)). Every binary16 rounding
// boundary is a binary64 number, so s + e rounds like s unless s is itself a boundary; then e decides.
std::uint16_t rn16_exact_sum(double s, double e) {
    const std::uint16_t r = rn16(s);
    if (e == 0.0 || !std::isfinite(s) || s == 0.0) return r;
    const double a = std::fabs(s);
    if (a >= 65536.0) return r;  // beyond every boundary: infinity either way
    const std::uint16_t sign = std::signbit(s) ? 0x8000 : 0;
    // The two binary16 magnitudes around a (0x7C00 stands for the boundary value 2^16 of the overflow to infinity).
    auto val = [](std::uint16_t b) { return b >= 0x7C00 ? 65536.0 : half_value(b); };
    const std::uint16_t m = static_cast<std::uint16_t>(rn16(a) & 0x7FFF);
    std::uint16_t lo = m, hi = m;
    if (val(m) > a) lo = static_cast<std::uint16_t>(m - 1);
    else if (val(m) < a) hi = static_cast<std::uint16_t>(m + 1);
    else return r;  // a is a binary16 value: s + e rounds to it
    if (a != (val(lo) + val(hi)) * 0.5) return r;  // not a boundary (the midpoint is exact in binary64)
    // s is the midpoint: the tie was broken to even, but s + e lies strictly on e's side.
    const bool away = (e > 0) == (s > 0);
    return static_cast<std::uint16_t>(sign | (away ? hi : lo));
}

// Knuth's TwoSum: s + e == a + b exactly.
inline void two_sum(double a, double b, double& s, double& e) {
    s = a + b;
    const double bb = s - a;
    e = (a - (s - bb)) + (b - bb);
}

}  // namespace

bool score_bias_op(const std::uint16_t* S, int H, std::uint32_t a_bits, std::uint32_t wd_bits,
                   const std::vector<std::pair<int, int>>& pairs, std::vector<std::uint16_t>& out, std::string& why) {
    out.clear();
    for (std::size_t i = 0; i < static_cast<std::size_t>(H) * 4096; ++i) {
        if (!half_is_finite(S[i])) {
            why = "a stored score-bias value is not finite (element " + std::to_string(i) + ")";
            return false;
        }
    }
    const double A = bits_to_float(a_bits), Wd = bits_to_float(wd_bits);
    // RN32(sqrt32(Wd) * A): sqrt in binary64 rounded again to binary32 is the correctly rounded binary32 sqrt.
    const double sq = rn32(std::sqrt(Wd));
    const double f2 = rn32(sq * A);
    if (!std::isfinite(A) || !std::isfinite(Wd) || !std::isfinite(sq) || !std::isfinite(f2)) {
        why = "a score-bias intermediate is not finite";
        return false;
    }
    std::vector<std::uint16_t> T(static_cast<std::size_t>(H) * 4096);
    for (int h = 0; h < H; ++h) {
        const std::uint16_t* Sh = S + static_cast<std::size_t>(h) * 4096;
        for (int a = 0; a < 64; ++a) {
            for (int b = 0; b < 64; ++b) {
                const double x = half_value(Sh[a * 64 + b]), y = half_value(Sh[b * 64 + a]);
                const double s1 = rn32(x + y);   // the binary16 sum is exact in binary64
                const double s2 = rn32(s1 * A);  // binary32 products are exact in binary64
                const double s3 = rn32(s2 * f2);
                if (!std::isfinite(s1) || !std::isfinite(s2) || !std::isfinite(s3)) {
                    why = "a score-bias intermediate is not finite (head " + std::to_string(h) + ")";
                    return false;
                }
                const std::uint16_t t = rn16(s3);
                if (!half_is_finite(t)) {
                    why = "a score-bias value overflows binary16 (head " + std::to_string(h) + ")";
                    return false;
                }
                T[static_cast<std::size_t>(h) * 4096 + static_cast<std::size_t>(a) * 64 + static_cast<std::size_t>(b)] = t;
            }
        }
    }
    out.reserve(static_cast<std::size_t>(H) * 12 * 256);
    for (int h = 0; h < H; ++h)
        for (const auto& p : pairs)
            for (int r = 0; r < 16; ++r)
                for (int s = 0; s < 16; ++s)
                    out.push_back(T[static_cast<std::size_t>(h) * 4096 + static_cast<std::size_t>(16 * p.first + r) * 64 +
                                    static_cast<std::size_t>(16 * p.second + s)]);
    return true;
}

bool const_op(OpKind k, bool negate, const std::vector<std::uint32_t>& words, std::uint32_t& bits, std::string& why) {
    if (words.empty()) {
        why = "no site element";
        return false;
    }
    const std::uint32_t w = words[0];
    for (std::size_t i = 1; i < words.size(); ++i) {
        if (words[i] != w) {
            why = "its sites disagree (site element " + std::to_string(i) + " differs from the first)";
            return false;
        }
    }
    switch (k) {
        case OpKind::ConstU32AsF32: bits = w; return true;
        case OpKind::ConstF16x2:
            if ((w >> 16) != (w & 0xFFFF)) {
                why = "the two 16-bit halves differ";
                return false;
            }
            bits = w & 0xFFFF;
            return true;
        case OpKind::ConstF16lo:
            if ((w >> 16) != 0) {
                why = "the high 16 bits are not zero";
                return false;
            }
            bits = w & 0xFFFF;
            return true;
        case OpKind::ConstF32ToF16: {
            double v = bits_to_float(w);
            if (negate) v = -v;
            if (!std::isfinite(v)) {
                why = "the binary32 value is not finite";
                return false;
            }
            const std::uint16_t y = rn16(v);  // v is a binary32 value: one rounding
            if (!half_is_finite(y)) {
                why = "RN16 overflows";
                return false;
            }
            bits = y;
            return true;
        }
        default: why = "not a const operation"; return false;
    }
}

bool attn_wmax_op(std::uint16_t lo, std::uint16_t hi, std::uint16_t p1, std::uint16_t p0, std::uint16_t& wmax,
                  bool& may_overflow, std::string& why) {
    if (!half_is_finite(lo) || !half_is_finite(hi) || !half_is_finite(p1) || !half_is_finite(p0)) {
        why = "an attention constant is not finite";
        return false;
    }
    const double vlo = half_value(lo), vhi = half_value(hi), vp1 = half_value(p1), vp0 = half_value(p0);
    if (vlo > vhi) {
        why = "the lower clamp exceeds the upper clamp";
        return false;
    }
    double best = -1.0;
    for (std::uint32_t c16 = 0; c16 < 65536; ++c16) {
        const std::uint16_t cb = static_cast<std::uint16_t>(c16);
        if (!half_is_finite(cb)) continue;
        const double c = half_value(cb);
        if (c < vlo || c > vhi) continue;
        // inner = RN16(A_P1 - c*c): c*c is exact in binary64; the difference as an exact two-term sum.
        double s, e;
        two_sum(vp1, -(c * c), s, e);
        const std::uint16_t inner = rn16_exact_sum(s, e);
        // p = RN16(c * inner + A_P0): the product is exact in binary64.
        two_sum(c * half_value(inner), vp0, s, e);
        const std::uint16_t pb = rn16_exact_sum(s, e);
        const double p = half_value(pb);
        const double y = 32.0 * (p - 1.0);  // exact (or non-finite)
        if (!(y >= 1.0 && y < 31.0)) {
            why = "the attention polynomial leaves its range (page 51.3 step 3)";
            return false;
        }
        const double n = std::floor(y);
        const double w = std::ldexp(1.0 + y - n, static_cast<int>(n) - 15);
        if (w > best) best = w;
    }
    wmax = half_bits_of(best);
    may_overflow = 64.0 * best >= 65520.0;
    return true;
}

namespace {

std::uint32_t rd32(const std::uint8_t* p) {
    return std::uint32_t(p[0]) | (std::uint32_t(p[1]) << 8) | (std::uint32_t(p[2]) << 16) | (std::uint32_t(p[3]) << 24);
}
std::uint16_t rd16(const std::uint8_t* p) { return static_cast<std::uint16_t>(p[0] | (p[1] << 8)); }

double e4m3_value(std::uint8_t code) {
    const int s = code >> 7, e = (code >> 3) & 15, m = code & 7;
    if (e == 15 && m == 7) return std::numeric_limits<double>::quiet_NaN();
    const double v = e == 0 ? std::ldexp(m, -9) : std::ldexp(8 + m, e - 10);
    return s ? -v : v;
}

double element_value(DType d, const std::uint8_t* p) {
    switch (d) {
        case DType::Float16: return half_value(rd16(p));
        case DType::Float32: return bits_to_float(rd32(p));
        case DType::E4M3: return e4m3_value(p[0]);
        case DType::UInt16: return rd16(p);
        case DType::UInt32: return rd32(p);
        case DType::UInt8: return p[0];
        case DType::Int32: return static_cast<std::int32_t>(rd32(p));
    }
    return 0.0;
}

// Smallest positive normal value of a floating dtype (0 for integer dtypes: no subnormals).
double min_normal(DType d) {
    switch (d) {
        case DType::Float16: return std::ldexp(1.0, -14);
        case DType::Float32: return std::ldexp(1.0, -126);
        case DType::E4M3: return std::ldexp(1.0, -6);
        default: return 0.0;
    }
}

}  // namespace

bool output_is_finite(DType d, const std::uint8_t* bytes, std::uint64_t count, std::uint64_t& first_bad) {
    if (d != DType::Float16 && d != DType::Float32) return true;
    const std::uint64_t sz = dtype_size(d);
    for (std::uint64_t i = 0; i < count; ++i) {
        if (!std::isfinite(element_value(d, bytes + i * sz))) {
            first_bad = i;
            return false;
        }
    }
    return true;
}

bool check_require(DType d, const std::uint8_t* bytes, std::uint64_t count, unsigned req, std::string& why) {
    if (!req) return true;
    const std::uint64_t sz = dtype_size(d);
    const double mn = min_normal(d);
    for (std::uint64_t i = 0; i < count; ++i) {
        const double v = element_value(d, bytes + i * sz);
        const char* failed = nullptr;
        if ((req & kReqFinite) && !std::isfinite(v)) failed = "finite";
        // Zero or normal (a non-finite value is neither).
        else if ((req & kReqNotSubnormal) && !(std::isfinite(v) && (v == 0.0 || std::fabs(v) >= mn))) failed = "not_subnormal";
        else if ((req & kReqEven) && !(std::isfinite(v) && std::fmod(v, 2.0) == 0.0)) failed = "even";
        else if ((req & kReqBelow8) && !(v < 8.0)) failed = "below_8";
        if (failed) {
            why = std::string("element ") + std::to_string(i) + " fails the condition " + failed;
            return false;
        }
    }
    return true;
}

double scalar_value(DType d, const std::uint8_t* bytes) { return element_value(d, bytes); }

bool check_relation(Rel rel, double a, double b) {
    switch (rel) {
        case Rel::Ge: return a >= b;
        case Rel::Gt: return a > b;
        case Rel::Le: return a <= b;
        case Rel::RatioBelowF16Overflow: {
            if (!(b > 0.0) || !std::isfinite(a) || !std::isfinite(b)) return false;
            // RN32(1 / b): the binary64 quotient rounded again (innocuous for binary32 operands; b is a binary16 or
            // binary32 value here).
            const float r = rn32(1.0 / b);
            const float rplus = std::nextafter(r, std::numeric_limits<float>::infinity());
            const float prod = rn32(a * static_cast<double>(rplus));  // binary32 * binary32: exact in binary64
            return std::fabs(static_cast<double>(prod)) < 65520.0;
        }
    }
    return false;
}

}  // namespace wgc
