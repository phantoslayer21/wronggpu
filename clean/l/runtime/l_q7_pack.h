// SPDX-License-Identifier: GPL-3.0-only
// Model L (issue #89): the weight blobs and lookup tables of the FP16-form (q7) kernels, header-only, byte for byte
// the Python packer's (clean/l/runner/l_pack.py pack_q7_block, pack_q7_linear, q7_tables).
//   block blob:  FP16 g1' g2' bp b1 b2 B, then FP8 codes Wqk Wv Wp W1 W2 row-major (g' = RN16(float32(gamma) x
//                sqrtf(C)), the norm's gain with sqrt(C) folded in), padded to 256 bytes
//   linear blob: FP16 bias [n], FP8 rows [n][K] at the next multiple of 16 (merges Wm / bm, expansions We / be)
//   tables:      rsqrt[65536] || rcp[65536], FP16, indexed by an FP16 bit pattern x: RN16(1 / sqrtf(x)) and RN16(1 / x)
//                in float32 (a NaN keeps its payload, as numpy's conversion does)
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "../../runtime/fpconv.h"

namespace wgrt {
namespace l {
namespace q7pack {

inline void put16(std::vector<uint8_t>* out, uint16_t h) {
  out->push_back((uint8_t)(h & 0xffu));
  out->push_back((uint8_t)(h >> 8));
}

// binary32 -> binary16 as numpy converts it: round to nearest even; a NaN keeps its sign and top 10 payload bits
inline uint16_t f16_np(float f) {
  const uint32_t a = f32_bits(f) & 0x7fffffffu;
  if (a > 0x7f800000u) {
    const uint16_t m = (uint16_t)((a >> 13) & 0x3ffu);
    return (uint16_t)(((f32_bits(f) >> 16) & 0x8000u) | 0x7c00u | (m ? m : 0x200u));
  }
  return f32_to_f16(f);
}

// A weight's FP16 copy (Q7F-1): its FP8 value, exact in FP16; a NaN as 0x7e00 (as numpy writes it)
inline uint16_t f16_weight(float x) {
  const double v = q8((double)x);
  return std::isnan(v) ? (uint16_t)0x7e00u : f32_to_f16((float)v);
}

// sum |w| per 32-wide chunk and row of an FP8 matrix W [rows][K], float32 little-endian, chunk-major [K / 32][rows]
// (exact: at most 32 FP8 magnitudes, multiples of 2^-9 below 2^14; a NaN weight gives NaN, which sends its outputs to
// the exact fallback)
inline void put_wsum(std::vector<uint8_t>* out, const std::vector<float>& W, int K) {
  for (int c0 = 0; c0 < K; c0 += 32)
    for (size_t r = 0; r < W.size() / (size_t)K; ++r) {
      double s = 0.0;
      for (int k = 0; k < 32; ++k) s += std::fabs(q8((double)W[r * K + c0 + k]));
      const uint32_t b = f32_bits((float)s);
      for (int i = 0; i < 4; ++i) out->push_back((uint8_t)(b >> (8 * i)));
    }
}

inline std::vector<uint8_t> q7_tables() {
  std::vector<uint8_t> out;
  out.reserve(4 * 65536);
  std::vector<uint16_t> rs(65536), rc(65536);
  for (uint32_t b = 0; b < 65536; ++b) {
    const float x = f16_to_f32((uint16_t)b);
    rs[b] = f16_np(1.0f / std::sqrt(x));
    rc[b] = f16_np(1.0f / x);
  }
  for (uint16_t h : rs) put16(&out, h);
  for (uint16_t h : rc) put16(&out, h);
  return out;
}

// One block's blob (C, heads, mlp = Model L's / Model M's block shapes); tensors as float32 row-major vectors.
inline bool pack_block(const std::vector<float>& gamma1, const std::vector<float>& gamma2, const std::vector<float>& bp,
                       const std::vector<float>& b1, const std::vector<float>& b2, const std::vector<float>& B,
                       const std::vector<float>& Wqk, const std::vector<float>& Wv, const std::vector<float>& Wp,
                       const std::vector<float>& W1, const std::vector<float>& W2, int C, int heads, int mlp,
                       std::vector<uint8_t>* out, std::string* err) {
  const size_t c = (size_t)C, h = (size_t)heads, m = (size_t)mlp;
  if (gamma1.size() != c || gamma2.size() != c || bp.size() != c || b1.size() != m || b2.size() != c ||
      B.size() != h * 256 || Wqk.size() != h * 32 * c || Wv.size() != h * 32 * c || Wp.size() != c * 32 * h ||
      W1.size() != m * c || W2.size() != c * m)
    return *err = "q7 block: tensor sizes do not match C " + std::to_string(C) + ", heads " + std::to_string(heads) +
                  ", MLP " + std::to_string(mlp),
           false;
  out->clear();
  const float sc = std::sqrt((float)C);
  for (const std::vector<float>* g : {&gamma1, &gamma2})
    for (float v : *g) put16(out, f32_to_f16(v * sc));
  for (const std::vector<float>* v : {&bp, &b1, &b2, &B})
    for (float x : *v) put16(out, f32_to_f16(x));
  for (const std::vector<float>* w : {&Wqk, &Wv, &Wp, &W1, &W2})
    for (float x : *w) out->push_back(fp8_encode((double)x));
  // the same matrices as FP16 (exact for FP8 values), read by the scalar kernels without an FP8 decode (Q7F-1)
  for (const std::vector<float>* w : {&Wqk, &Wv, &Wp, &W1, &W2})
    for (float x : *w) put16(out, f16_weight(x));
  // per row and 32-wide chunk sum |w| (float32, exact), the matrix-instruction path's bound (Q7F-2)
  put_wsum(out, Wqk, C);
  put_wsum(out, Wv, C);
  put_wsum(out, Wp, 32 * heads);
  put_wsum(out, W1, C);
  put_wsum(out, W2, mlp);
  out->resize((out->size() + 255) / 256 * 256, 0);
  return true;
}

// A merge's or an expansion's blob: W [n][K] (K a multiple of 32), b [n].
inline bool pack_linear(const std::vector<float>& W, const std::vector<float>& b, int n, int K,
                        std::vector<uint8_t>* out, std::string* err) {
  if (n <= 0 || K <= 0 || K % 32 || b.size() != (size_t)n || W.size() != (size_t)n * K)
    return *err = "q7 linear: tensor sizes do not match", false;
  out->clear();
  for (float x : b) put16(out, f32_to_f16(x));
  out->resize(((size_t)2 * n + 15) / 16 * 16, 0);
  for (float x : W) out->push_back(fp8_encode((double)x));
  for (float x : W) put16(out, f16_weight(x));   // FP16 rows (Q7F-1)
  put_wsum(out, W, K);                            // sum |w| per row chunk (Q7F-2)
  out->resize((out->size() + 255) / 256 * 256, 0);
  return true;
}

}  // namespace q7pack
}  // namespace l
}  // namespace wgrt
