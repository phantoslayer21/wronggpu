// SPDX-License-Identifier: MIT
// Host emulation of the D3D12 ports' platform layer (test-only; never shipped). WrongGPU's own code.
//
// hlsl_emu.py compiles a kernel file of clean/dx12 or clean/k/dx12 as C++ against this header, which stands in for
// hlsl/wg_platform.hlsli and hlsl/wg_ags.hlsli; every other header (wg_gfx11.hlsli, wg_math.hlsli, the model K headers) is
// the real one. Each lane of a workgroup runs the kernel text as a coroutine of the lane emulator of the HIP kernels
// (clean/kernels/emu/wg_emu.cpp, through hlsl_emu_bridge.cpp): wave operations and matrix instructions are collective
// there. The matrix instruction is the emulator's model: gfx12 (wmma) for WG_GFX 12, the measured gfx11 F16 instruction
// (wmma16: row 2 v + h, A / B read from the lanes of the output's own half-wave) for WG_GFX 11, where the channel's F16
// opcode is one K = 8 instruction (K 8 .. 15 zero) and the opcode WG_AGS_F16_F32_K16 one K = 16 instruction.
//
// Arithmetic: FP32 and FP16 operations are single IEEE operations (C++ float, _Float16 through float: exact for one
// operation), mad is fused, the hardware estimates (rcp, rsqrt, exp2, log2) are the exact functions, FP32 denormals
// are kept, and the emulator sums each matrix product exactly. So it checks layouts, addressing, exchanges, borders and
// the order of the FP16 steps, not the last bit of the GPU (the GPU tests do; clean/k/dx12/GPU tests of the native work).
// Memory: one arena of bytes (the runtime's arena); an access outside it is counted (hemu_oob) and reads 0.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>

typedef unsigned int uint;
typedef std::uint16_t uint16_t;

// ---- bridge to the lane emulator (hlsl_emu_bridge.cpp) -----------------------------------------------------------------
extern "C" {
uint hemu_lane();                        // lane in the workgroup
uint hemu_block();                       // workgroup index of the launch
uint hemu_xor(uint v, uint mask);        // value of lane (lane ^ mask) of the wave (collective)
void hemu_barrier();                     // workgroup barrier
void hemu_wmma_g12(const uint a[4], const uint b[4], const float c[8], float d[8]);    // gfx12 F16 model
void hemu_wmma_g11(const uint a[8], const uint b[8], const float c[8], float d[8]);    // gfx11 F16, 16 halves
void hemu_fail(const char* what);
extern unsigned char* hemu_arena;
extern unsigned long long hemu_arena_bytes;
extern unsigned long long hemu_oob;
extern uint hemu_consts[256];
}

// ---- scalar intrinsics -------------------------------------------------------------------------------------------------
inline uint asuint(float f) { uint u; std::memcpy(&u, &f, 4); return u; }
inline uint asuint(uint u) { return u; }
inline uint asuint(int i) { return (uint)i; }
inline float asfloat(uint u) { float f; std::memcpy(&f, &u, 4); return f; }
inline float asfloat(float f) { return f; }
inline float asfloat(int i) { return asfloat((uint)i); }
inline int asint(float f) { return (int)asuint(f); }
inline float f16tof32(uint h) { std::uint16_t b = (std::uint16_t)(h & 0xffffu); _Float16 x; std::memcpy(&x, &b, 2); return (float)x; }
// f32tof16 rounds toward zero on the D3D12 driver (wg_math.hlsli "HALF CONVERSIONS")
inline uint f32tof16(float f)
{
    if (std::isnan(f)) { uint u = asuint(f); return ((u >> 16) & 0x8000u) | 0x7e00u | ((u >> 13) & 0x3ffu); }
    const float a = std::fabs(f);
    _Float16 h = (_Float16)a;                 // nearest even, then one step down if that rounded up
    std::uint16_t b; std::memcpy(&b, &h, 2);
    if ((float)h > a) --b;
    return (uint)b | (std::signbit(f) ? 0x8000u : 0u);
}
inline float mad(float a, float b, float c) { return std::fma(a, b, c); }
inline double mad(double a, double b, double c) { return std::fma(a, b, c); }
inline double fma(double a, double b, double c) { return std::fma(a, b, c); }
inline float rcp(float x) { return 1.0f / x; }
inline float rsqrt(float x) { return 1.0f / std::sqrt(x); }
inline float exp2(float x) { return std::exp2(x); }
inline float log2(float x) { return std::log2(x); }
inline float sqrt(float x) { return std::sqrt(x); }
inline float round(float x) { return std::nearbyint(x); }
inline float floor(float x) { return std::floor(x); }
inline float abs(float x) { return std::fabs(x); }
inline int abs(int x) { return x < 0 ? -x : x; }
inline bool isnan(float x) { return std::isnan(x); }
inline bool isnan(double x) { return std::isnan(x); }
inline bool isinf(float x) { return std::isinf(x); }
inline float min(float a, float b) { return std::fmin(a, b); }
inline float max(float a, float b) { return std::fmax(a, b); }
inline int min(int a, int b) { return a < b ? a : b; }
inline int max(int a, int b) { return a > b ? a : b; }
inline uint min(uint a, uint b) { return a < b ? a : b; }
inline uint max(uint a, uint b) { return a > b ? a : b; }
inline int clamp(int x, int lo, int hi) { return min(max(x, lo), hi); }
inline float clamp(float x, float lo, float hi) { return min(max(x, lo), hi); }
inline uint countbits(uint v) { return (uint)__builtin_popcount(v); }
inline uint firstbithigh(uint v) { return v ? 31u - (uint)__builtin_clz(v) : ~0u; }
inline uint firstbitlow(uint v) { return v ? (uint)__builtin_ctz(v) : ~0u; }
inline uint reversebits(uint v) { return __builtin_bitreverse32(v); }
template <class T> inline T select(bool c, T a, T b) { return c ? a : b; }
// HLSL 2021 and / or (compiled with -fno-operator-names, so these are names)
inline bool and(bool a, bool b) { return a && b; }
inline bool or(bool a, bool b) { return a || b; }

// ---- vectors -----------------------------------------------------------------------------------------------------------
template <class T>
struct hvec2 {
    T x, y;
    hvec2() = default;
    hvec2(T a, T b) : x(a), y(b) {}
    T& operator[](uint i) { return i ? y : x; }
    T operator[](uint i) const { return i ? y : x; }
};
template <class T>
struct hvec4 {
    T x, y, z, w;
    hvec4() = default;
    hvec4(T a, T b, T c, T d) : x(a), y(b), z(c), w(d) {}
    T& operator[](uint i) { return i == 0 ? x : i == 1 ? y : i == 2 ? z : w; }
    T operator[](uint i) const { return i == 0 ? x : i == 1 ? y : i == 2 ? z : w; }
};
typedef hvec2<uint> uint2;
typedef hvec4<uint> uint4;
typedef hvec2<int> int2;
typedef hvec4<float> float4;
typedef hvec2<float> float2;
typedef hvec2<std::uint16_t> uint16_t2;
inline uint4 operator^(uint4 a, uint b) { return uint4(a.x ^ b, a.y ^ b, a.z ^ b, a.w ^ b); }
inline uint4 operator^(uint4 a, uint4 b) { return uint4(a.x ^ b.x, a.y ^ b.y, a.z ^ b.z, a.w ^ b.w); }
inline uint4 operator&(uint4 a, uint b) { return uint4(a.x & b, a.y & b, a.z & b, a.w & b); }
inline uint4 operator|(uint4 a, uint4 b) { return uint4(a.x | b.x, a.y | b.y, a.z | b.z, a.w | b.w); }
inline bool operator==(uint4 a, uint4 b) { return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w; }

// FP16 pairs: each operation one IEEE FP16 operation, round to nearest even (through float: exact for +, -, *).
struct half2 {
    _Float16 x, y;
    half2() = default;
    half2(_Float16 a, _Float16 b) : x(a), y(b) {}
};
inline _Float16 h16(float f) { return (_Float16)f; }
inline half2 operator+(half2 a, half2 b) { return half2(h16((float)a.x + (float)b.x), h16((float)a.y + (float)b.y)); }
inline half2 operator-(half2 a, half2 b) { return half2(h16((float)a.x - (float)b.x), h16((float)a.y - (float)b.y)); }
inline half2 operator*(half2 a, half2 b) { return half2(h16((float)a.x * (float)b.x), h16((float)a.y * (float)b.y)); }
inline half2 operator-(half2 a) { return half2(-a.x, -a.y); }
inline _Float16 hmin(_Float16 a, _Float16 b) { return h16(std::fmin((float)a, (float)b)); }
inline _Float16 hmax(_Float16 a, _Float16 b) { return h16(std::fmax((float)a, (float)b)); }
inline half2 min(half2 a, half2 b) { return half2(hmin(a.x, b.x), hmin(a.y, b.y)); }
inline half2 max(half2 a, half2 b) { return half2(hmax(a.x, b.x), hmax(a.y, b.y)); }
inline half2 abs(half2 a) { return half2(h16(std::fabs((float)a.x)), h16(std::fabs((float)a.y))); }
inline half2 asfloat16(uint16_t2 u) { half2 h; std::memcpy(&h.x, &u.x, 2); std::memcpy(&h.y, &u.y, 2); return h; }
inline uint16_t2 asuint16(half2 h) { uint16_t2 u; std::memcpy(&u.x, &h.x, 2); std::memcpy(&u.y, &h.y, 2); return u; }
inline half2 exp2(half2 x) { return half2(h16(std::exp2((float)x.x)), h16(std::exp2((float)x.y))); }
inline float dot2add(half2 a, half2 b, float c)
{
    return (float)((double)c + (double)a.x * (double)b.x + (double)a.y * (double)b.y);
}

// ---- memory: the arena ---------------------------------------------------------------------------------------------------
struct hemu_buffer {
    static bool ok(uint a, uint n) { return (unsigned long long)a + n <= hemu_arena_bytes; }
    template <class T = uint>
    T Load(uint a) const
    {
        T v{};
        if (ok(a, sizeof(T))) std::memcpy(&v, hemu_arena + a, sizeof(T));
        else ++hemu_oob;
        return v;
    }
    uint2 Load2(uint a) const { return uint2(Load(a), Load(a + 4u)); }
    uint4 Load4(uint a) const { return uint4(Load(a), Load(a + 4u), Load(a + 8u), Load(a + 12u)); }
    template <class T>
    void Store(uint a, T v) const
    {
        if (ok(a, sizeof(T))) std::memcpy(hemu_arena + a, &v, sizeof(T));
        else ++hemu_oob;
    }
    void Store(uint a, uint v) const { Store<uint>(a, v); }
    void Store2(uint a, uint2 v) const { Store<uint>(a, v.x); Store<uint>(a + 4u, v.y); }
    void Store4(uint a, uint4 v) const { Store2(a, uint2(v.x, v.y)); Store2(a + 8u, uint2(v.z, v.w)); }
    void InterlockedAdd(uint a, uint v, uint& o) const { o = Load(a); Store(a, o + v); }
    void InterlockedMax(uint a, uint v, uint& o) const { o = Load(a); Store(a, o > v ? o : v); }
    void InterlockedExchange(uint a, uint v, uint& o) const { o = Load(a); Store(a, v); }
    void InterlockedCompareExchange(uint a, uint c, uint v, uint& o) const { o = Load(a); if (o == c) Store(a, v); }
};
static const hemu_buffer wg_mem{}, wg_mem_gc{};

#define globallycoherent
#define precise

// ---- kernel interface (wg_platform.hlsli) --------------------------------------------------------------------------------
#ifndef WG_BLOCK
#error "WG_BLOCK / WG_WAVE come from the @wg-kernel line"
#endif
#ifndef WG_WAVE
#define WG_WAVE 32
#endif
// The root constants: the argument block's scalars overlay hemu_consts (dword i = the i-th declared scalar). Everything
// after WG_ARGS_END is the body of a struct derived from the argument block (hlsl_emu.py closes it), so the kernel text
// reads its arguments by name; the object is hemu_consts itself.
#define WG_ARGS_BEGIN struct hemu_args_t { uint wg_group_base_; uint wg_grid_;
#define WG_ARGS_END }; static_assert(sizeof(hemu_args_t) <= sizeof(hemu_consts), "argument block"); \
    struct hemu_scope : hemu_args_t {
#define groupshared inline static

uint wg_lane_id() { return hemu_lane(); }
uint WaveGetLaneIndex() { return hemu_lane() % WG_WAVE; }
#if WG_REV_ORDER
uint wg_block_id() { return hemu_consts[1] - 1u - (hemu_block() + hemu_consts[0]); }
#else
uint wg_block_id() { return hemu_block() + hemu_consts[0]; }
#endif
#define wg_grid_size() (hemu_consts[1])
uint wg_wave_lane() { return WaveGetLaneIndex(); }
// hlsl_emu.py registers the WG_KERNEL entries of the file's declarations and calls them per lane
#define WG_KERNEL(name) void name()
#define WG_SETUP() ((void)0)

struct v8f { float v[8]; };
v8f wg_splat(float x) { v8f r; for (uint i = 0; i < 8; ++i) r.v[i] = x; return r; }

uint wg_ld8(uint base, uint off) { uint a = base + off; return (wg_mem.Load(a & ~3u) >> ((a & 3u) * 8u)) & 0xffu; }
uint wg_ld16(uint base, uint off) { return (uint)wg_mem.Load<std::uint16_t>(base + off); }
uint wg_ld32(uint base, uint off) { return wg_mem.Load(base + off); }
uint2 wg_ld64(uint base, uint off) { return wg_mem.Load2(base + off); }
uint4 wg_ld128(uint base, uint off) { return wg_mem.Load4(base + off); }
uint4 wg_ld128_a16(uint base, uint off) { return wg_mem.Load4(base + off); }
float wg_ldf(uint base, uint off) { return asfloat(wg_mem.Load(base + off)); }
float4 wg_ld4f(uint base, uint off) { uint4 u = wg_mem.Load4(base + off); return float4(asfloat(u.x), asfloat(u.y), asfloat(u.z), asfloat(u.w)); }
void wg_st16(uint base, uint off, uint v) { wg_mem.Store<std::uint16_t>(base + off, (std::uint16_t)v); }
void wg_st32(uint base, uint off, uint v) { wg_mem.Store(base + off, v); }
void wg_st64(uint base, uint off, uint2 v) { wg_mem.Store2(base + off, v); }
void wg_st128(uint base, uint off, uint4 v) { wg_mem.Store4(base + off, v); }
void wg_stf(uint base, uint off, float v) { wg_mem.Store(base + off, asuint(v)); }
void wg_st64_hint(uint base, uint off, uint2 v) { wg_st64(base, off, v); }
void wg_st128_hint(uint base, uint off, uint4 v) { wg_st128(base, off, v); }
v8f wg_load8f(uint base, uint off)
{
    v8f r;
    for (uint i = 0; i < 8; ++i) r.v[i] = wg_ldf(base, off + 4u * i);
    return r;
}
uint wg_atomic_add(uint base, uint off, uint v) { uint o; wg_mem_gc.InterlockedAdd(base + off, v, o); return o; }
uint wg_atomic_max(uint base, uint off, uint v) { uint o; wg_mem_gc.InterlockedMax(base + off, v, o); return o; }
uint wg_atomic_exch(uint base, uint off, uint v) { uint o; wg_mem_gc.InterlockedExchange(base + off, v, o); return o; }
uint wg_atomic_cas(uint base, uint off, uint cmp, uint v) { uint o; wg_mem_gc.InterlockedCompareExchange(base + off, cmp, v, o); return o; }
uint wg_ld32_gc(uint base, uint off) { return wg_mem_gc.Load(base + off); }
void wg_st32_gc(uint base, uint off, uint v) { wg_mem_gc.Store(base + off, v); }
void wg_fence_device() {}

void wg_barrier() { hemu_barrier(); }
void wg_barrier_lds() { hemu_barrier(); }
uint WaveReadLaneAt(uint v, uint src)
{
    // collective: every lane reads all 32 xor partners and keeps the one that is lane src
    uint r = 0u;
    const uint lane = WaveGetLaneIndex();
    for (uint m = 0; m < 32u; ++m) {
        const uint x = hemu_xor(v, m);
        if ((lane ^ m) == (src & 31u)) r = x;
    }
    return r;
}
float wg_xor(float v, uint d) { return asfloat(hemu_xor(asuint(v), d)); }
uint wg_xor_u(uint v, uint d) { return hemu_xor(v, d); }
float wg_xor16(float v) { return wg_xor(v, 16u); }
uint wg_xor16_u(uint v) { return wg_xor_u(v, 16u); }
int wg_uniform(int v) { return (int)hemu_xor((uint)v, WaveGetLaneIndex()); }   // lane 0's value
uint WaveActiveBitOr(uint v)
{
    for (uint m = 1u; m < 32u; m <<= 1) v |= hemu_xor(v, m);
    return v;
}

float wg_exp2(float x) { return exp2(x); }
float wg_log2(float x) { return log2(x); }
float wg_rcp(float x) { return rcp(x); }
float wg_rsq(float x) { return rsqrt(x); }
float wg_sqrt(float x) { return sqrt(x); }
float wg_rint(float x) { return round(x); }
float wg_floor(float x) { return floor(x); }
float wg_fmax(float a, float b) { return max(a, b); }
float wg_fmin(float a, float b) { return min(a, b); }
float wg_med3f(float x, float lo, float hi) { return min(max(x, lo), hi); }
int wg_f2i(float x) { return std::isnan(x) ? 0 : x >= 2147483648.0f ? 2147483647 : x < -2147483648.0f ? (-2147483647 - 1) : (int)x; }
uint wg_f2u(float x) { return std::isnan(x) || x <= 0.0f ? 0u : x >= 4294967296.0f ? 0xffffffffu : (uint)x; }
int wg_med3i(int x, int lo, int hi) { return x < lo ? lo : (x > hi ? hi : x); }
int wg_min_i(int a, int b) { return a < b ? a : b; }

#include "wg_math.hlsli"

float wg_rn16(float x) { return wg_h2f(wg_f2h(x)); }
float wg_h2f_lo(uint v) { return f16tof32(v & 0xffffu); }
float wg_h2f_hi(uint v) { return f16tof32(v >> 16); }
uint wg_f2h_bits(float x) { return wg_f2h(x); }
uint wg_pack_h2(float x, float y) { return wg_f2h(x) | (wg_f2h(y) << 16); }
uint wg_h2(float a, float b) { return f32tof16(a) | (f32tof16(b) << 16); }
half2 wg_as_h2(uint v) { return asfloat16(uint16_t2((std::uint16_t)(v & 0xffffu), (std::uint16_t)(v >> 16))); }
uint wg_h2_bits(half2 h) { uint16_t2 u = asuint16(h); return (uint)u.x | ((uint)u.y << 16); }
float wg_dot2_acc(half2 a, half2 b, float c) { return dot2add(a, b, c); }
uint4 wg_to_f16(v8f d)
{
    return uint4(wg_pack_h2(d.v[0], d.v[1]), wg_pack_h2(d.v[2], d.v[3]), wg_pack_h2(d.v[4], d.v[5]),
                 wg_pack_h2(d.v[6], d.v[7]));
}

// ---- the driver channel (wg_ags.hlsli): matrix instruction and conversion ---------------------------------------------
#if WG_WAVE == 32
#define WG_AGS_MMA 0x28u
#define WG_AGS_CONVERT 0x32u
#define WG_AGS_FP8_F32 0x7u
#define WG_AGS_F16_F32 0x5u
#define WG_AGS_F16_F16 0x1u
#define WG_AGS_T_F32_ACC 0x26u
#define WG_AGS_T_F16_ACC 0x24u
#define WG_AGS_T_F16_B 0x14u
#define WG_AGS_T_FP8_B 0x1au
struct wg_frag { uint v[8]; };
#if defined(WG_GFX) && WG_GFX == 11
#define HEMU_GFX11 1
#else
#define HEMU_GFX11 0
#endif
wg_frag wg_ags_mma(uint format, wg_frag a, wg_frag b, wg_frag c)
{
    float cc[8], dd[8];
    for (uint i = 0; i < 8; ++i) cc[i] = asfloat(c.v[i]);
    if (format == WG_AGS_F16_F32 && !HEMU_GFX11) {
        hemu_wmma_g12(a.v, b.v, cc, dd);
    } else if (format == WG_AGS_F16_F32 || format == 0x7eu) {
        if (format == 0x7eu && !HEMU_GFX11) hemu_fail("the K = 16 opcode on a gfx12 build");
        uint a8[8], b8[8];
        for (uint i = 0; i < 8; ++i) {
            a8[i] = format == 0x7eu || i < 4 ? a.v[i] : 0u;   // the raw K = 8 instruction: K 8 .. 15 are zero
            b8[i] = format == 0x7eu || i < 4 ? b.v[i] : 0u;
        }
        hemu_wmma_g11(a8, b8, cc, dd);
    } else {
        hemu_fail("matrix format not emulated");
    }
    wg_frag r;
    for (uint i = 0; i < 8; ++i) r.v[i] = asuint(dd[i]);
    return r;
}
wg_frag wg_ags_convert(wg_frag m, uint from_type, uint to_type)
{
    wg_frag r = {};
    if (from_type != WG_AGS_T_F32_ACC || to_type != WG_AGS_T_F16_ACC) hemu_fail("conversion not emulated");
    for (uint i = 0; i < 4; ++i) {
        const _Float16 lo = (_Float16)asfloat(m.v[2 * i]), hi = (_Float16)asfloat(m.v[2 * i + 1]);
        std::uint16_t bl, bh;
        std::memcpy(&bl, &lo, 2);
        std::memcpy(&bh, &hi, 2);
        r.v[i] = (uint)bl | ((uint)bh << 16);
    }
    return r;
}
wg_frag wg_frag_v8f(v8f c) { wg_frag f; for (uint i = 0; i < 8; ++i) f.v[i] = asuint(c.v[i]); return f; }
v8f wg_v8f_frag(wg_frag f) { v8f c; for (uint i = 0; i < 8; ++i) c.v[i] = asfloat(f.v[i]); return c; }
wg_frag wg_frag4(uint4 a)
{
    wg_frag f;
    f.v[0] = a.x; f.v[1] = a.y; f.v[2] = a.z; f.v[3] = a.w;
    for (uint i = 4; i < 8; ++i) f.v[i] = 0u;
    return f;
}
v8f wg_wmma_f16(uint4 a, uint4 b, v8f c) { return wg_v8f_frag(wg_ags_mma(WG_AGS_F16_F32, wg_frag4(a), wg_frag4(b), wg_frag_v8f(c))); }
uint4 wg_to_f16_ags(v8f d)
{
    wg_frag r = wg_ags_convert(wg_frag_v8f(d), WG_AGS_T_F32_ACC, WG_AGS_T_F16_ACC);
    return uint4(r.v[0], r.v[1], r.v[2], r.v[3]);
}
#endif
