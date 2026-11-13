// SPDX-License-Identifier: MIT
// Original D3D12 per-frame helpers and native-device byte fill (2026-10-15).
// color_divide / scale_rgb / timing_barrier reproduce clean/kernels/frame.hip as frame.co:
// spec/10 (pre-exposure), plus our frame.co gfx1201 ISA. RGB division uses wg_div with a narrow normal-result
// underflow-tie repair; alpha is copied as raw bits. Shared wg_math.hlsli remains unchanged.
// scale_rgb uses one FP32 multiply, plus an exact integer repair when D3D12 flushes a tiny exact product whose RNE
// result is the smallest normal. No new arithmetic exceptions: wg_math.hlsli's FP32 denormal / NaN rules apply.
// native_fill is our own byte-range algorithm: one lane owns each aligned DWORD, including partial edge DWORDs.
// The device validates the allocation range, orders prior accesses, and provides DWORD-padded allocation backing.
// The first and last masked stores preserve all bytes outside [dst, dst + count); there is no same-DWORD RMW race.
//
// @wg-kernel color_divide co=frame.co block=256 wave=32 args=p,p,u,u defines=WG_NATIVE_COLOR=1
// @wg-kernel scale_rgb co=frame.co block=256 wave=32 args=p,u,u defines=WG_NATIVE_SCALE=1
// @wg-kernel timing_barrier co=frame.co block=32 wave=32 args=u defines=WG_NATIVE_TIMING=1
// @wg-kernel native_fill co=- block=256 wave=32 args=p,u,u defines=WG_NATIVE_FILL=1
#include "wg_platform.hlsli"

#if defined(WG_NATIVE_COLOR)
WG_ARGS_BEGIN
    uint in_p, out_p;
    float pi;
    int n;
WG_ARGS_END

float native_color_div(float a, float b)
{
    const float r = wg_div(a, b);
    const uint ua = asuint(a), ub = asuint(b);
    const uint ea = (ua >> 23u) & 255u, eb = (ub >> 23u) & 255u;
    // For normal operands, a tiny quotient can round to 2^-126 only when Ea-Eb is -126 or -127.
    // At -126, Ma < Mb has an integer gap >= 1, so Ma/Mb < 1-2^-24: it cannot round into the normal range.
    // At -127, the RNE midpoint inequality is 2^23*Ma >= (2^24-1)*Mb. Since 2^23 <= Ma,Mb < 2^24,
    // only Ma=2^24-1 and Mb=2^23 attain it, exactly the tie. This covers all 127 normal exponent splits.
    // wg_div flushes its en<1 result here; restore the correctly rounded normal result without allowing subnormals.
    if ((asuint(r) & 0x7fffffffu) == 0u && ea != 0u && eb != 255u && eb == ea + 127u &&
        (ua & 0x7fffffu) == 0x7fffffu && (ub & 0x7fffffu) == 0u)
        return asfloat(((ua ^ ub) & 0x80000000u) | 0x00800000u);
    return r;
}

WG_KERNEL(color_divide)
{
    WG_SETUP();
    const uint i = wg_block_id() * 256u + wg_lane_id();
    if ((int)i >= n) return;
    const uint4 c = wg_ld128(in_p, i * 16u);
    const uint4 r = uint4(asuint(native_color_div(asfloat(c.x), pi)), asuint(native_color_div(asfloat(c.y), pi)),
                         asuint(native_color_div(asfloat(c.z), pi)), c.w);
    wg_st128(out_p, i * 16u, r);
}

#elif defined(WG_NATIVE_SCALE)
WG_ARGS_BEGIN
    uint buf;
    float pi;
    int n;
WG_ARGS_END

float native_scale(float a, float b)
{
    const float r = a * b;
    const uint ua = asuint(a), ub = asuint(b);
    const uint ea = (ua >> 23u) & 255u, eb = (ub >> 23u) & 255u;
    // The driver flushes a tiny product before rounding, including values that RNE rounds UP into the normal range.
    // For normal operands, only biased exponent sum 127 can have a tiny exact product within half an ulp of 2^-126:
    //   P = (2^23 + mantissa(a)) * (2^23 + mantissa(b)), exact value P * 2^-173.
    // The inclusive RNE lower tie is P = 2^47 - 2^23; the exclusive upper bound is P = 2^47.
    // Integer multiplication keeps the decision exact and repairs every mantissa pair in this interval, both signs.
    if ((asuint(r) & 0x7fffffffu) == 0u && ea != 0u && eb != 0u && ea + eb == 127u) {
        const uint64_t product = (uint64_t)((ua & 0x7fffffu) | 0x800000u) *
                                 (uint64_t)((ub & 0x7fffffu) | 0x800000u);
        const uint64_t normal = (uint64_t)1u << 47u;
        if (product >= normal - ((uint64_t)1u << 23u) && product < normal)
            return asfloat(((ua ^ ub) & 0x80000000u) | 0x00800000u);
    }
    return r;
}

WG_KERNEL(scale_rgb)
{
    WG_SETUP();
    const uint i = wg_block_id() * 256u + wg_lane_id();
    if ((int)i < n) wg_stf(buf, i * 4u, native_scale(wg_ldf(buf, i * 4u), pi));
}

#elif defined(WG_NATIVE_TIMING)
WG_ARGS_BEGIN
    int unused;
WG_ARGS_END

WG_KERNEL(timing_barrier)
{
    WG_SETUP();
}

#elif defined(WG_NATIVE_FILL)
WG_ARGS_BEGIN
    uint dst, count, value;  // byte destination, byte count, byte value (low eight bits)
WG_ARGS_END

WG_KERNEL(native_fill)
{
    WG_SETUP();
    if (count == 0u) return;
    const uint head = dst & 3u;
    // Quotient/remainder form avoids count + head + 3 overflowing a DWORD.
    const uint words = (count >> 2u) + (((count & 3u) + head + 3u) >> 2u);
    const uint i = wg_block_id() * 256u + wg_lane_id();
    if (i >= words) return;
    const uint addr = (dst & ~3u) + i * 4u;
    const uint tail = (head + (count & 3u)) & 3u;
    uint mask = 0xffffffffu;
    if (i == 0u) mask &= 0xffffffffu << (head * 8u);
    if (i + 1u == words) mask &= 0xffffffffu >> (((4u - tail) & 3u) * 8u);
    const uint pattern = (value & 255u) * 0x01010101u;
    uint word = pattern;
    if (mask != 0xffffffffu) word = (wg_ld32(0u, addr) & ~mask) | (pattern & mask);
    wg_st32(0u, addr, word);
}
#endif
