// SPDX-License-Identifier: MIT
// Host-side lane emulator for the kernels (test-only; never shipped).
//
// Every GPU lane of a workgroup runs the real kernel code as a cooperative coroutine (ucontext). Cross-lane
// operations (WMMA, lane exchange, barriers) are collective: each lane deposits its operands, yields, and computes
// its result once all lanes of the wave have deposited. Lanes are resumed round-robin, so lanes of one wave stay in
// lockstep as on the GPU. Workgroups run one after another.
#pragma once

#include <cmath>
#include <cstddef>
#include <utility>

#define WG_FN inline
#define WG_FN_NOINLINE inline
#define WG_KERNEL(name) extern "C" void name
#define WG_KERNEL_N(name, threads) extern "C" void name
#define WG_KERNEL_NW(name, threads, waves) extern "C" void name
#define WG_LDS static  // one copy per kernel instantiation; workgroups run one after another

namespace wg_emu {

typedef void (*Thunk)(void** args);

struct Registrar {
  Registrar(const char* name, Thunk thunk);
};

template <class... A, size_t... I>
void invoke_impl(void (*f)(A...), void** args, std::index_sequence<I...>) {
  f(*(A*)args[I]...);
}
template <class... A>
void invoke(void (*f)(A...), void** args) {
  invoke_impl(f, args, std::index_sequence_for<A...>{});
}

u32 lane_id();
u32 block_id();
v8f wmma_fp8(v2i a, v2i b, v8f c);
v8f wmma_f16(v8h a, v8h b, v8f c);
float xor16(float v);
float xor_lanes(float v, u32 mask);  // value of lane (lane ^ mask)
void barrier();

}  // namespace wg_emu

#define WG_EMU_REGISTER(name) \
  static wg_emu::Registrar wg_emu_reg_##name(#name, [](void** a) { wg_emu::invoke(&name, a); });

WG_FN u32 wg_lane_id() { return wg_emu::lane_id(); }
WG_FN u32 wg_block_id() { return wg_emu::block_id(); }
WG_FN v8f wg_wmma_fp8(v2i a, v2i b, v8f c) { return wg_emu::wmma_fp8(a, b, c); }
WG_FN v8f wg_wmma_f16(v8h a, v8h b, v8f c) { return wg_emu::wmma_f16(a, b, c); }
WG_FN float wg_xor16(float v) { return wg_emu::xor16(v); }
WG_FN float wg_emu_xor(float v, u32 mask) { return wg_emu::xor_lanes(v, mask); }
WG_FN void wg_barrier() { wg_emu::barrier(); }
WG_FN void wg_barrier_lds() { wg_emu::barrier(); }
#define WG_SCHED_VALU 0
#define WG_SCHED_WMMA 0
#define WG_SCHED_TRANS 0
#define WG_SCHED(mask, size) ((void)0)
#define WG_SCHED_FENCE() ((void)0)
WG_FN float wg_exp2(float x) { return std::exp2(x); }
WG_FN float wg_rsq(float x) { return 1.0f / std::sqrt(x); }
WG_FN float wg_rcp(float x) { return 1.0f / x; }
WG_FN float wg_rint(float x) { return std::nearbyint(x); }
WG_FN float wg_fmax(float a, float b) { return std::fmax(a, b); }
WG_FN float wg_fmin(float a, float b) { return std::fmin(a, b); }
