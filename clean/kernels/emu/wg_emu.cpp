// SPDX-License-Identifier: MIT
// Host-side lane emulator runtime (test-only; never shipped). See wg_emu.h.
//
// The WMMA model below encodes the register layout assumed for gfx12 wave32 WMMA:
//   A (16x16, rows M): lane l holds row l % 16; B (16x16, columns N): lane l holds column l % 16;
//   input element j of a lane (byte j for FP8, half j for FP16) holds K index kslot(l / 16, j);
//   C/D (FP32): lane l holds column l % 16, element v holds row drow(l / 16, v).
// The kernels only depend on the row/column-per-lane part. kslot may be any permutation as long as A and B use the
// same one, and drow is absorbed by the weight packing (runner/packing.py, D_ROW). The emulator can switch both to
// other permutations (emu_set_layout) to check that independence.
#include <ucontext.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "../wg_platform.h"

namespace wg_emu {

namespace {

const int kStackBytes = 1 << 20;
const int kMaxLanes = 512;

struct Lane {
  ucontext_t ctx;
  u32 id;
  bool done;
  char* stack;
};

struct WaveBuf {
  unsigned char a[WG_WAVE][16];
  unsigned char b[WG_WAVE][16];
  float c[WG_WAVE][8];
  float x[WG_WAVE];
};

struct State {
  std::map<std::string, Thunk>* kernels = nullptr;
  Lane lanes[kMaxLanes];
  u32 nlanes = 0;
  Lane* cur = nullptr;
  ucontext_t sched;
  u32 block = 0;
  Thunk thunk = nullptr;
  void** args = nullptr;
  WaveBuf waves[kMaxLanes / WG_WAVE];
  int kvariant = 0;
  int dvariant = 0;
  long long wmma_count = 0;
  u32 bar_count = 0;
  u32 bar_gen = 0;
};

State g;

std::map<std::string, Thunk>& kernels() {
  if (!g.kernels) g.kernels = new std::map<std::string, Thunk>();
  return *g.kernels;
}

void yield() { swapcontext(&g.cur->ctx, &g.sched); }

int kslot(int h, int j) {
  if (g.kvariant == 0) return 8 * h + j;
  return j < 4 ? 4 * h + j : 8 + 4 * h + (j - 4);
}

int drow(int h, int v) {
  if (g.dvariant == 0) return v + 8 * h;
  return 2 * v + h;
}

double fp8_value(unsigned char c) {
  int e = (c >> 3) & 15, m = c & 7;
  double v = (e == 15 && m == 7) ? NAN : (e == 0 ? m * std::ldexp(1.0, -9) : (1.0 + m / 8.0) * std::ldexp(1.0, e - 7));
  return (c & 0x80) ? -v : v;
}

double f16_value(const unsigned char* p) {
  _Float16 h;
  std::memcpy(&h, p, 2);
  return (double)(float)h;
}

template <bool kFp8>
v8f wmma(const void* a, const void* b, size_t bytes, v8f c) {
  u32 lane = g.cur->id % WG_WAVE;
  WaveBuf& w = g.waves[g.cur->id / WG_WAVE];
  std::memcpy(w.a[lane], a, bytes);
  std::memcpy(w.b[lane], b, bytes);
  yield();  // every lane of the wave has deposited its operands after one round
  double A[16][16], B[16][16];
  for (int l = 0; l < WG_WAVE; ++l) {
    int r = l % 16, h = l / 16;
    for (int j = 0; j < 8; ++j) {
      int k = kslot(h, j);
      A[r][k] = kFp8 ? fp8_value(w.a[l][j]) : f16_value(&w.a[l][2 * j]);
      B[k][r] = kFp8 ? fp8_value(w.b[l][j]) : f16_value(&w.b[l][2 * j]);
    }
  }
  v8f d;
  int n = lane % 16, h = lane / 16;
  for (int v = 0; v < 8; ++v) {
    int m = drow(h, v);
    double s = 0.0;
    for (int k = 0; k < 16; ++k) s += A[m][k] * B[k][n];
    d[v] = (float)((double)c[v] + s);
  }
  if (lane == 0) ++g.wmma_count;
  yield();  // nobody may overwrite the buffers before every lane has read them
  return d;
}

void trampoline() {
  g.thunk(g.args);
  g.cur->done = true;
}

void run_block(u32 block) {
  g.block = block;
  for (u32 i = 0; i < g.nlanes; ++i) {
    Lane& L = g.lanes[i];
    if (!L.stack) L.stack = (char*)std::malloc(kStackBytes);
    getcontext(&L.ctx);
    L.ctx.uc_stack.ss_sp = L.stack;
    L.ctx.uc_stack.ss_size = kStackBytes;
    L.ctx.uc_link = &g.sched;
    L.id = i;
    L.done = false;
    makecontext(&L.ctx, trampoline, 0);
  }
  for (bool any = true; any;) {
    any = false;
    for (u32 i = 0; i < g.nlanes; ++i) {
      if (g.lanes[i].done) continue;
      any = true;
      g.cur = &g.lanes[i];
      swapcontext(&g.sched, &g.lanes[i].ctx);
    }
  }
  g.cur = nullptr;
}

}  // namespace

Registrar::Registrar(const char* name, Thunk thunk) { kernels()[name] = thunk; }

u32 lane_id() { return g.cur->id; }
u32 block_id() { return g.block; }

v8f wmma_fp8(v2i a, v2i b, v8f c) { return wmma<true>(&a, &b, 8, c); }
v8f wmma_f16(v8h a, v8h b, v8f c) { return wmma<false>(&a, &b, 16, c); }

float xor16(float v) {
  u32 lane = g.cur->id % WG_WAVE;
  WaveBuf& w = g.waves[g.cur->id / WG_WAVE];
  w.x[lane] = v;
  yield();
  float r = w.x[lane ^ 16u];
  yield();
  return r;
}

float xor_lanes(float v, u32 mask) {
  u32 lane = g.cur->id % WG_WAVE;
  WaveBuf& w = g.waves[g.cur->id / WG_WAVE];
  w.x[lane] = v;
  yield();
  float r = w.x[(lane ^ mask) % WG_WAVE];
  yield();
  return r;
}

void barrier() {
  u32 gen = g.bar_gen;
  if (++g.bar_count == g.nlanes) {
    g.bar_count = 0;
    ++g.bar_gen;
  }
  do yield();  // always yield at least once so lanes of a wave leave the barrier in the same round
  while (g.bar_gen == gen);
}

}  // namespace wg_emu

extern "C" {

// Run a registered kernel over a 1-D grid of gx workgroups of `block` lanes (a multiple of 32, at most 512).
// args: hipModuleLaunchKernel-style array of pointers to the argument values. Returns 0, or -1 on bad arguments.
int emu_launch(const char* name, unsigned gx, unsigned block, void** args) {
  using namespace wg_emu;
  auto it = kernels().find(name);
  if (it == kernels().end() || block == 0 || block > (unsigned)kMaxLanes || block % WG_WAVE) return -1;
  g.nlanes = block;
  g.thunk = it->second;
  g.args = args;
  for (unsigned b = 0; b < gx; ++b) run_block(b);
  return 0;
}

int emu_has_kernel(const char* name) { return wg_emu::kernels().count(name) ? 1 : 0; }

void emu_set_layout(int kvariant, int dvariant) {
  wg_emu::g.kvariant = kvariant;
  wg_emu::g.dvariant = dvariant;
}

long long emu_wmma_count() { return wg_emu::g.wmma_count; }

}  // extern "C"
