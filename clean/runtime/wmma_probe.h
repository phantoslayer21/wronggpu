// SPDX-License-Identifier: MIT
// The native D3D12 device's driver-channel probe (native_driver_probe, clean/dx12/kernels/native_game_helpers.hlsl):
// its operands, its buffer layout and its expected result, computed here on the CPU. WrongGPU's own code. Header only,
// no device: device_d3d12.cpp runs the probe on a borrowed device at create, and tests/test_wmma_probe.cpp checks on
// any host that a matrix-fragment layout other than the one below fails it.
//
// The probe runs one wave of 32 lanes: D = A x B + C, once as FP16 x FP16 -> FP32 and once as FP8 (E4M3) x FP8 -> FP32
// (16 x 16 x 16, the driver channel's v_wmma_f32_16x16x16_f16 / _fp8_fp8). Fragment layout (clean/FORMATS.md, "WMMA
// fragment" and "Accumulator layout", gfx12 wave32), lane l = 16 h + r:
//   A: element j of lane l holds A[r][8 h + j]       (FP16: half j of the 4 dwords, low half first; FP8: byte j)
//   B: element j of lane l holds B[8 h + j][r]
//   C, D: element v of lane l holds [v + 8 h][r]     (D_ROW "gfx12")
// A and B are small nonzero integers chosen (by a random search) so that the 256 entries of A x B are all distinct,
// and C is (32 v + l) / 512, distinct per element. Every value, product and partial sum is exact in FP32, so the result
// does not depend on the order of the hardware's additions, and every expected result is a different number. A
// different row, column, lane, element or half order of A, B or the accumulator, or A and B swapped or transposed,
// changes at least one result (tests/test_wmma_probe.cpp checks these properties and each such layout). Only a K order
// that A and B share is invisible, and it changes no entry of A x B either.
// Cost at create: one 4.5 KiB upload, one launch of one wave, one 2 KiB readback.
#pragma once

#include <cstdint>
#include <cstring>

namespace wgrt {
namespace wmma_probe {

constexpr unsigned kLanes = 32, kElements = 8;
// Byte offsets in the probe's buffer (the kernel uses the same constants). The two results come first: float
// i = 256 p + 32 v + l is element v of lane l of result p (0 = FP16 operands, 1 = FP8 operands).
constexpr uint32_t kOutF16 = 0, kOutFp8 = 1024, kInC = 2048, kInAF16 = 3072, kInBF16 = 3584, kInAFp8 = 4096,
                   kInBFp8 = 4352, kBytes = 4608;
constexpr unsigned kResults = 512;  // floats read back from offset 0

// A[m][k] and B[k][n]: nonzero integers in [-16, 16] (exact in FP8 E4M3 and in FP16).
constexpr int8_t kA[16][16] = {
    {9, 15, -8, -10, 10, 11, 1, -12, -7, 1, -12, -12, -3, 7, -4, 11},
    {-16, -3, 1, 15, -9, 11, -14, -6, 8, 3, 15, 9, 14, 11, -12, 14},
    {12, -12, 16, 8, -6, -12, -4, 14, 10, -9, -1, 12, 5, -7, -13, 16},
    {-9, 1, 7, -6, 1, -7, 11, 4, -11, -6, 7, 6, -3, -12, -12, -9},
    {10, 12, -6, 4, -1, -8, 1, -12, -1, -12, -9, -9, -16, -4, 8, 5},
    {9, 11, -15, 9, -13, 15, -14, -12, -1, -2, -5, -2, -3, 3, 4, -5},
    {9, 5, -1, -13, 11, -9, -6, 9, 12, 16, 11, 10, 8, 11, 2, 10},
    {-9, -2, 10, 8, 14, 3, -5, -2, -15, -8, 8, 12, 10, -14, -3, 11},
    {5, 13, -2, -3, -9, 11, -12, -6, -14, 1, 4, 2, 2, -10, -4, -6},
    {6, -14, 5, -2, 15, 6, -11, 6, -6, 1, -10, -2, -15, 1, 1, -6},
    {1, -11, 14, -7, 10, 5, 7, -14, 8, 5, -14, -8, 13, -11, 11, -1},
    {-6, -5, -14, -13, 11, 7, -13, 10, 10, -15, 13, 5, 4, -9, 9, -3},
    {10, 10, -4, 2, -1, -16, 5, -16, -2, -4, 5, 13, 13, -9, -7, -2},
    {-3, -12, -12, 10, -10, -4, 12, -5, -6, -8, 15, 11, 10, -3, -3, -15},
    {-6, 10, 12, -8, -13, -11, 6, 6, -3, 2, 13, 15, -12, -7, 2, -2},
    {-15, 4, 6, 16, 15, 15, -8, -4, -6, -5, -4, -3, 11, -13, 12, 11},
};
constexpr int8_t kB[16][16] = {
    {3, 13, 15, -4, 7, 15, 2, 8, 11, -2, 16, 12, -2, 16, -5, 15},
    {3, -3, -9, -11, -9, 10, 8, 15, 1, 1, 13, -8, 8, 14, 6, 12},
    {1, -1, 12, 12, 11, -4, -13, -16, 10, -13, -7, -1, -6, -8, 11, 3},
    {12, -10, 13, 9, -11, -11, -5, -10, -3, 14, 10, -13, -6, -5, 7, -13},
    {-15, 6, -11, -16, 16, 12, -3, -12, -6, 12, -5, 3, -11, -4, -12, -1},
    {9, -4, -7, -7, -9, 15, -9, -3, 9, -1, 4, 12, 1, -14, 9, -13},
    {5, 16, 14, 6, 8, -5, 10, 16, -12, 1, -12, -2, -6, -9, -12, 2},
    {-7, -6, -15, 6, -2, -8, 3, 6, -13, -3, -6, -15, -12, -13, -10, -12},
    {13, -5, 3, 15, -3, 3, -12, -12, -9, 2, 14, -5, 9, -13, -3, -4},
    {-8, -10, -4, 10, -9, 7, 10, -7, 9, 10, 7, -3, -3, 16, -16, 10},
    {15, -4, -7, -1, 16, -1, -7, 3, -7, -12, 1, -10, -9, -6, -4, 1},
    {-4, -11, 14, 13, -1, -11, -4, 11, 2, -11, -15, 2, -13, -12, 16, 2},
    {-15, -13, 14, 5, 2, -12, -5, 7, 10, -12, -10, 15, -5, 1, -7, -8},
    {-9, -16, 5, -6, -12, 15, 4, -4, -9, 13, 9, -14, -3, -8, -15, 14},
    {-7, -5, -2, -13, -1, 8, 7, 8, -5, -9, -4, 4, -16, 15, -3, -5},
    {-7, -12, -8, 13, -5, -1, 6, 3, 4, 10, -10, -14, -6, -9, 5, 7},
};

// The layout above: which matrix entry element j / v of lane l holds.
inline void a_entry(unsigned lane, unsigned j, unsigned& m, unsigned& k) { m = lane % 16; k = 8 * (lane / 16) + j; }
inline void b_entry(unsigned lane, unsigned j, unsigned& k, unsigned& n) { n = lane % 16; k = 8 * (lane / 16) + j; }
inline void d_entry(unsigned lane, unsigned v, unsigned& m, unsigned& n) { m = v + 8 * (lane / 16); n = lane % 16; }

inline float c_value(unsigned lane, unsigned v) { return static_cast<float>(32 * v + lane) / 512.0f; }

// Bits of an integer x with |x| <= 16 (exact in both formats).
inline uint16_t f16_bits(int x) {
  if (x == 0) return 0;
  const unsigned s = x < 0 ? 1u : 0u, a = static_cast<unsigned>(x < 0 ? -x : x);
  unsigned e = 0;
  while ((a >> (e + 1)) != 0) ++e;
  return static_cast<uint16_t>(s << 15 | (e + 15) << 10 | ((a << (10 - e)) & 0x3ffu));
}
inline uint8_t fp8_bits(int x) {  // E4M3, bias 7
  if (x == 0) return 0;
  const unsigned s = x < 0 ? 1u : 0u, a = static_cast<unsigned>(x < 0 ? -x : x);
  unsigned e = 0;
  while ((a >> (e + 1)) != 0) ++e;
  return static_cast<uint8_t>(s << 7 | (e + 7) << 3 | ((a << 3 >> e) & 0x7u));
}

// The probe's buffer before the launch (kBytes): A, B and C in fragment layout, the results zero.
inline void fill(uint8_t* buffer) {
  std::memset(buffer, 0, kBytes);
  for (unsigned l = 0; l < kLanes; ++l)
    for (unsigned j = 0; j < kElements; ++j) {
      unsigned m, n, k;
      a_entry(l, j, m, k);
      const uint16_t a16 = f16_bits(kA[m][k]);
      std::memcpy(buffer + kInAF16 + 16 * l + 2 * j, &a16, 2);
      buffer[kInAFp8 + 8 * l + j] = fp8_bits(kA[m][k]);
      b_entry(l, j, k, n);
      const uint16_t b16 = f16_bits(kB[k][n]);
      std::memcpy(buffer + kInBF16 + 16 * l + 2 * j, &b16, 2);
      buffer[kInBFp8 + 8 * l + j] = fp8_bits(kB[k][n]);
      const float c = c_value(l, j);
      std::memcpy(buffer + kInC + 4 * (32 * j + l), &c, 4);
    }
}

// Expected value of result float i (both results are the same exact numbers).
inline float expected(unsigned i) {
  const unsigned lane = i % 32, v = (i % 256) / 32;
  unsigned m, n;
  d_entry(lane, v, m, n);
  int sum = 0;
  for (unsigned k = 0; k < 16; ++k) sum += kA[m][k] * kB[k][n];
  return static_cast<float>(sum) + c_value(lane, v);
}

// First result float that differs from the expected one, or -1 when all kResults match.
inline int first_mismatch(const float* results) {
  for (unsigned i = 0; i < kResults; ++i)
    if (results[i] != expected(i)) return static_cast<int>(i);
  return -1;
}

}  // namespace wmma_probe
}  // namespace wgrt
