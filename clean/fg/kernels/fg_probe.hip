// SPDX-License-Identifier: GPL-3.0-only
// Frame generation (issue #96): probe of gfx12's 16 x 16 x 16 matrix instructions against net1's chunk arithmetic
// (clean/fg/runner/fg_wmma_probe.py). A spec chunk is acc' = RN16(acc + sum of 16 float16 products), one rounding. One
// WMMA has K = 16 and adds C in the same instruction, so each of its 256 outputs is one chunk when C is the accumulator.
// The question: does the instruction form that sum exactly and round once?
//
// Memory, per tile t (one wave per tile): A [t][L][m][k], B [t][L][k][n] float16 (L chunks chained), C [t][m][n]
// float16 (the starting accumulator). Fragments (gfx12 wave32, as in clean/kernels/emu/wg_emu.cpp): lane l, h = l / 16:
// A row l % 16 and B column l % 16, input element j = K index 8 h + j; accumulator element v = row v + 8 h, column
// l % 16. A wrong guess would show as mismatches in nearly every random case.
//
//   fg_probe_exact   per output element, the exact chain (clean/fg/kernels/fg_exact16.h, the method of fp16acc.py):
//                    E16 = the float16 accumulator after L chunks; E32 = RN32 of the exact last chunk (the float32
//                    a perfect float32-accumulate instruction would give on it), NaN 0x7FC00000.
//   fg_probe_mma16   v_wmma_f16_16x16x16_f16 chained L times (D fed back as C): D16 = the final accumulator.
//   fg_probe_mma32   v_wmma_f32_16x16x16_f16 chained L times with the float16 accumulator in between (C = the
//                    accumulator widened exactly, the result rounded to float16 by v_cvt_f16_f32): D32 = the raw
//                    float32 result of the last chunk; F = (flag << 16) | final float16 accumulator, flag = some
//                    chunk's float32 result lay exactly on a float16 midpoint (fg_mid16: the only place where
//                    RN16(RN32(s)) can differ from RN16(s)).
// One shared signature (A, B, C, D, F, tiles, L); fg_probe_mma16 and fg_probe_exact do not use every argument.
#include "wg_platform.h"
#include "wg_mem.h"

typedef unsigned int uint;
typedef unsigned short u16;
#define FG_FN WG_FN
#define FG_I64 long long
#define FG_FBH32(v) (31u - (uint)__builtin_clz(v))
#include "fg_exact16.h"
#include "fg_probe_mid.h"

WG_FN uint fg_f2h(float f) { return (uint)__builtin_bit_cast(u16, (_Float16)f); }      // round to nearest even
WG_FN float fg_h2f(uint h) { return (float)__builtin_bit_cast(_Float16, (u16)h); }     // exact

// RN32 of the exact value of s (finite terms only): (hh * 2^40 + mid * 2^20 + ll) * 2^-48, nearest even. A non-zero
// value is at least 2^-48 in magnitude and below 2^90, so it is a normal float32. An exact zero is +0 (fg_round16's rule).
WG_FN uint fg_round32(FgAcc s) {
    if (s.nan_ != 0u || (s.pinf != 0u && s.ninf != 0u)) return 0x7FC00000u;
    if (s.pinf != 0u) return 0x7F800000u;
    if (s.ninf != 0u) return 0xFF800000u;
    __int128 t = ((__int128)s.hh << 40) + ((__int128)s.mid << 20) + (__int128)s.ll;
    if (t == 0) return 0u;
    const uint neg = t < 0 ? 1u : 0u;
    unsigned __int128 mag = (unsigned __int128)(neg != 0u ? -t : t);
    const unsigned long long hi = (unsigned long long)(mag >> 64), lo = (unsigned long long)mag;
    const int e = hi != 0ull ? 64 + 63 - __builtin_clzll(hi) : 63 - __builtin_clzll(lo);   // leading bit index
    unsigned long long q;
    int ex = e;
    if (e <= 23) {
        q = (unsigned long long)(mag << (23 - e));
    } else {
        const int sh = e - 23;
        q = (unsigned long long)(mag >> sh);
        const unsigned __int128 rem = mag & ((((unsigned __int128)1) << sh) - 1);
        const unsigned __int128 half = ((unsigned __int128)1) << (sh - 1);
        if (rem > half || (rem == half && (q & 1ull) != 0ull)) ++q;
        if (q == (1ull << 24)) { q >>= 1; ++ex; }
    }
    const uint bexp = (uint)(ex - 48 + 127);
    return (neg << 31) | (bexp << 23) | ((uint)q & 0x7FFFFFu);
}

WG_KERNEL_N(fg_probe_exact, 256)(const u16* __restrict__ A, const u16* __restrict__ B, const u16* __restrict__ C,
                                 u16* __restrict__ E16, uint* __restrict__ E32, int tiles, int L) {
    const int idx = (int)(wg_block_id() * 256u + wg_lane_id());
    if (idx >= tiles * 256) return;
    const int t = idx >> 8, m = (idx >> 4) & 15, n = idx & 15;
    uint acc = C[idx];
    uint e32 = 0u;
    for (int l = 0; l < L; ++l) {
        const u16* a = A + ((size_t)t * L + l) * 256 + m * 16;
        const u16* b = B + ((size_t)t * L + l) * 256 + n;
        FgAcc s = fg_acc_begin(acc);
        for (int k = 0; k < 16; ++k) s = fg_acc_add(s, a[k], b[k * 16]);
        if (l == L - 1) e32 = fg_round32(s);
        acc = fg_acc_end(s);
    }
    E16[idx] = (u16)acc;
    E32[idx] = e32;
}

// this lane's A row / B column fragments of chunk l of tile t
WG_FN void fg_probe_frags(const u16* A, const u16* B, int t, int l, int L, uint lane, v8h* fa, v8h* fb) {
    const uint r = lane & 15u, h = lane >> 4;
    const u16* a = A + ((size_t)t * L + l) * 256;
    const u16* b = B + ((size_t)t * L + l) * 256;
    for (int j = 0; j < 8; ++j) {
        const uint k = 8u * h + (uint)j;
        (*fa)[j] = __builtin_bit_cast(_Float16, a[r * 16u + k]);
        (*fb)[j] = __builtin_bit_cast(_Float16, b[k * 16u + r]);
    }
}

WG_KERNEL_N(fg_probe_mma16, 256)(const u16* __restrict__ A, const u16* __restrict__ B, const u16* __restrict__ C,
                                 u16* __restrict__ D, uint* __restrict__ F, int tiles, int L) {
    const uint lane = wg_lane_id() & 31u;
    const int t = (int)(wg_block_id() * 8u + (wg_lane_id() >> 5));
    if (t >= tiles) return;                                         // wave-uniform
    const uint r = lane & 15u, h = lane >> 4;
    v8h acc;
    for (int v = 0; v < 8; ++v) acc[v] = __builtin_bit_cast(_Float16, C[(size_t)t * 256 + (8u * h + v) * 16u + r]);
    for (int l = 0; l < L; ++l) {
        v8h fa, fb;
        fg_probe_frags(A, B, t, l, L, lane, &fa, &fb);
        acc = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12(fa, fb, acc);
    }
    // (an element is copied to a scalar first: __builtin_bit_cast of a subscripted vector element read element 0)
    for (int v = 0; v < 8; ++v) {
        const _Float16 x = acc[v];
        D[(size_t)t * 256 + (8u * h + v) * 16u + r] = __builtin_bit_cast(u16, x);
    }
}

WG_KERNEL_N(fg_probe_mma32, 256)(const u16* __restrict__ A, const u16* __restrict__ B, const u16* __restrict__ C,
                                 uint* __restrict__ D, uint* __restrict__ F, int tiles, int L) {
    const uint lane = wg_lane_id() & 31u;
    const int t = (int)(wg_block_id() * 8u + (wg_lane_id() >> 5));
    if (t >= tiles) return;                                         // wave-uniform
    const uint r = lane & 15u, h = lane >> 4;
    uint acc[8], flag[8];
    v8f d;
    for (int v = 0; v < 8; ++v) {
        acc[v] = C[(size_t)t * 256 + (8u * h + v) * 16u + r];
        flag[v] = 0u;
        d[v] = 0.0f;
    }
    for (int l = 0; l < L; ++l) {
        v8h fa, fb;
        fg_probe_frags(A, B, t, l, L, lane, &fa, &fb);
        v8f c;
        for (int v = 0; v < 8; ++v) c[v] = fg_h2f(acc[v]);
        d = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(fa, fb, c);
        for (int v = 0; v < 8; ++v) {
            const float x = d[v];                   // a scalar copy (see fg_probe_mma16)
            flag[v] |= fg_mid16(__builtin_bit_cast(uint, x));
            acc[v] = fg_f2h(x);
        }
    }
    for (int v = 0; v < 8; ++v) {
        const size_t o = (size_t)t * 256 + (8u * h + v) * 16u + r;
        const float x = d[v];
        D[o] = __builtin_bit_cast(uint, x);
        F[o] = (flag[v] << 16) | acc[v];
    }
}
