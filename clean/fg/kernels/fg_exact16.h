// SPDX-License-Identifier: MIT
// Frame generation (issue #96): exact float16 arithmetic in integers, shared by the HIP kernels (clean/fg/kernels/*.hip)
// and their D3D12 ports (clean/dx12/kernels/fg_*.hlsl), so both back ends compute the same bits by construction.
//
// The same method as the CPU reference clean/fg/reference/fp16acc.py: a finite float16 value is m * 2^-24 with an integer
// |m| < 2^40; m splits into a high and a low 20-bit part, so a 16-term dot product is exact in three 64-bit sums
// (high * high, high * low + low * high, low * low). The accumulator acc' = RN16(acc + sum of a_i * b_i) is rounded once,
// to nearest with ties to even, by integer arithmetic only. inf and NaN follow IEEE (any NaN, 0 * inf or inf - inf gives
// NaN, written as 0x7E00; otherwise an infinite term gives that infinity).
//
// The including file defines:
//   FG_FN          function qualifier (HIP: WG_FN; HLSL: nothing)
//   FG_I64         signed 64-bit integer (HIP: long long; HLSL: int64_t)
//   FG_FBH32(v)    index of the highest set bit of a non-zero 32-bit unsigned value
#ifndef FG_EXACT16_H
#define FG_EXACT16_H

#define FG_NAN16 0x7E00u
#define FG_LOW20 1048575

struct FgAcc {
    FG_I64 hh;       // sum of high * high  (units 2^-8)
    FG_I64 mid;      // sum of high * low + low * high, plus 16 * m(acc)  (units 2^-28)
    FG_I64 ll;       // sum of low * low  (units 2^-48), never negative
    uint nan_;       // 1: the result is NaN
    uint pinf;       // 1: a +inf term
    uint ninf;       // 1: a -inf term
};

FG_FN uint fg_is_nan16(uint h) { return ((h & 0x7C00u) == 0x7C00u && (h & 0x3FFu) != 0u) ? 1u : 0u; }
FG_FN uint fg_is_inf16(uint h) { return ((h & 0x7FFFu) == 0x7C00u) ? 1u : 0u; }

// floor(log2(v)) of a 64-bit value v > 0
FG_FN uint fg_log2_64(FG_I64 v) {
    const uint hi = (uint)(v >> 32);
    const uint lo = (uint)v;                       // the low 32 bits
    return hi != 0u ? 32u + FG_FBH32(hi) : FG_FBH32(lo);
}

// a finite float16 (bits) -> m with value = m * 2^-24
FG_FN FG_I64 fg_m16(uint h) {
    const uint e = (h >> 10) & 31u;
    const uint f = h & 1023u;
    FG_I64 m = (FG_I64)f;
    if (e != 0u) m = (FG_I64)(f | 1024u) * ((FG_I64)1 << (e - 1u));
    return (h & 0x8000u) != 0u ? -m : m;
}

FG_FN FgAcc fg_acc_begin(uint acc) {
    FgAcc s;
    s.hh = 0;
    s.mid = 0;
    s.ll = 0;
    s.nan_ = fg_is_nan16(acc);
    s.pinf = (fg_is_inf16(acc) != 0u && (acc & 0x8000u) == 0u) ? 1u : 0u;
    s.ninf = (fg_is_inf16(acc) != 0u && (acc & 0x8000u) != 0u) ? 1u : 0u;
    if ((acc & 0x7C00u) != 0x7C00u) s.mid = fg_m16(acc) * 16;
    return s;
}

// s + a * b (a, b float16 bits)
FG_FN FgAcc fg_acc_add(FgAcc s, uint a, uint b) {
    if (fg_is_nan16(a) != 0u || fg_is_nan16(b) != 0u) {
        s.nan_ = 1u;
    } else if (fg_is_inf16(a) != 0u || fg_is_inf16(b) != 0u) {
        if ((a & 0x7FFFu) == 0u || (b & 0x7FFFu) == 0u) {
            s.nan_ = 1u;                                            // 0 * inf
        } else if (((a ^ b) & 0x8000u) != 0u) {
            s.ninf = 1u;
        } else {
            s.pinf = 1u;
        }
    } else {
        const FG_I64 ma = fg_m16(a);
        const FG_I64 mb = fg_m16(b);
        const FG_I64 ah = ma >> 20, al = ma & FG_LOW20;
        const FG_I64 bh = mb >> 20, bl = mb & FG_LOW20;
        s.hh += ah * bh;
        s.mid += ah * bl + al * bh;
        s.ll += al * bl;
    }
    return s;
}

// RN16 of the exact value (hh * 2^40 + mid * 2^20 + ll) * 2^-48, ll >= 0, as float16 bits
FG_FN uint fg_round16(FG_I64 hh, FG_I64 mid, FG_I64 ll) {
    const FG_I64 lim = (FG_I64)1 << 30;
    const uint big = (hh > lim || hh < -lim) ? 1u : 0u;              // |value| >= 2^22: infinite
    const FG_I64 hhc = hh > lim ? lim : (hh < -lim ? -lim : hh);
    // value * 2^25 = hh * 2^17 + mid2 / 8 + llr / 2^23, with mid2 = mid + (ll >> 20) and 0 <= llr < 2^20
    const FG_I64 l1 = ll >> 20;
    const FG_I64 llr = ll & FG_LOW20;
    const FG_I64 mid2 = mid + l1;
    const FG_I64 m1 = mid2 >> 3;                                      // floor
    const FG_I64 midr = mid2 & 7;
    const FG_I64 q = hhc * 131072 + m1;                               // floor of value * 2^25
    const uint sticky = (midr != 0 || llr != 0) ? 1u : 0u;
    const uint neg = q < 0 ? 1u : 0u;
    FG_I64 mag = q;
    if (neg != 0u) mag = sticky != 0u ? -q - 1 : -q;
    // quantum in units of 2^-25: 2 below 2^-14 (subnormal), else 2^(e - 10) in the binade [2^e, 2^(e + 1)) units
    const uint e = fg_log2_64(mag > 0 ? mag : 1);
    const uint qexp = mag < 2048 ? 1u : e - 10u;
    const FG_I64 quantum = (FG_I64)1 << qexp;
    const FG_I64 r = mag & (quantum - 1);
    const FG_I64 down = mag - r;
    const FG_I64 half = quantum >> 1;
    const uint odd = (uint)((down >> qexp) & 1);
    const uint up = (r > half || (r == half && (sticky != 0u || odd != 0u))) ? 1u : 0u;
    const FG_I64 res = down + (up != 0u ? quantum : 0);              // the rounded magnitude, units 2^-25
    const uint sign = big != 0u ? (hh < 0 ? 1u : 0u) : neg;
    uint bits;
    if (big != 0u || res >= ((FG_I64)65536 << 25)) {
        bits = 0x7C00u;
    } else if (res < 2048) {
        bits = (uint)(res >> 1);                                      // subnormal (or zero)
    } else {
        const uint e2 = fg_log2_64(res);
        bits = ((e2 - 10u) << 10) | ((uint)(res >> (e2 - 10u)) & 1023u);
    }
    return bits | (sign << 15);
}

FG_FN uint fg_acc_end(FgAcc s) {
    if (s.nan_ != 0u || (s.pinf != 0u && s.ninf != 0u)) return FG_NAN16;
    if (s.pinf != 0u) return 0x7C00u;
    if (s.ninf != 0u) return 0xFC00u;
    return fg_round16(s.hh, s.mid, s.ll);
}

// the order of a float16 value for max (-0 and +0 are equal); not for NaN
FG_FN int fg_key16(uint h) { return (h & 0x8000u) != 0u ? -(int)(h & 0x7FFFu) : (int)(h & 0x7FFFu); }

// net1's activation: y = max(x, h(s * x)) with the delivered slope s (float16 0x211E); NaN stays NaN
#define FG_LEAKY_SLOPE16 0x211Eu
FG_FN uint fg_leaky16(uint x) {
    if (fg_is_nan16(x) != 0u) return x;
    const uint t = fg_acc_end(fg_acc_add(fg_acc_begin(0u), FG_LEAKY_SLOPE16, x));
    return fg_key16(t) > fg_key16(x) ? t : x;
}

#endif
