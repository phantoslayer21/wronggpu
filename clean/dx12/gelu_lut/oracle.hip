// SPDX-License-Identifier: MIT
// HIP oracle of the MLP hidden path h -> q8(GELU(h)) as the runtime's code objects compute it (swin.co *_nc Swin blocks,
// dec1_dec0__pack12.co, swin__bmegahg.co bneck6_nc): wg_platform.h wg_gelu (WG_GELU_MODE 2, sigmoid form) of each value,
// then wg_pack_q8_nc (v_cvt_pk_fp8_f32, no clamp). Built with the runtime's device flags (clean/runner/build.py
// device_command); run.py checks that its ISA has the runtime kernels' GELU chain.
//   gelu_rle:   every FP32 bit pattern u of a chunk -> one record (u << 8 | byte) wherever byte(u) != byte(u - 1)
//               (and at u = 0): the run-length form of the whole function, 2^32 inputs.
//   gelu_check: recomputes every byte of a chunk and compares it with the run-length form (binary search): proves the
//               run-length form is the function.
#include "wg_platform.h"


// bytes of wg_pack_q8_nc(wg_gelu(x[0..7])) for x = asfloat(u0 + 0..7), as two dwords (byte i = element i)
WG_FN v2i oracle8(u32 u0) {
  v8f g;
#pragma unroll
  for (int i = 0; i < 8; ++i) g[i] = wg_gelu(__builtin_bit_cast(float, u0 + (u32)i));
  return wg_pack_q8_nc(g);
}
WG_FN u32 byte_of(v2i p, int i) { return ((u32)(i < 4 ? p[0] : p[1]) >> (8 * (i & 3))) & 0xffu; }

WG_KERNEL_N(gelu_rle, 256)(u32 base, u64* rec, u32* count, u32 cap) {
  const u32 u0 = base + 8u * (wg_block_id() * 256u + wg_lane_id());
  const v2i p = oracle8(u0);
  const v2i q = oracle8(u0 - 8u);   // the 8 patterns before (only the last is used)
  u32 prev = u0 == 0u ? 0x100u : byte_of(q, 7);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const u32 b = byte_of(p, i);
    if (b != prev) {
      const u32 k = __hip_atomic_fetch_add(count, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      if (k < cap) rec[k] = ((u64)(u0 + (u32)i) << 8) | b;
    }
    prev = b;
  }
}

// run containing u: the largest j with pos[j] <= u (pos[0] = 0)
WG_FN u32 run_of(const u32* pos, u32 n, u32 u) {
  u32 lo = 0, hi = n;   // pos[lo] <= u < pos[hi] (pos[n] = 2^32)
  while (hi - lo > 1u) {
    const u32 mid = (lo + hi) >> 1;
    if (pos[mid] <= u) lo = mid; else hi = mid;
  }
  return lo;
}

// res[0] = differences, res[4 + 2k], res[5 + 2k] = (u, hip | expected << 8), k < 30
WG_KERNEL_N(gelu_check, 256)(u32 base, const u32* pos, const u32* byt, u32 n, u32* res) {
  const u32 u0 = base + 8u * (wg_block_id() * 256u + wg_lane_id());
  const v2i p = oracle8(u0);
  u32 j = run_of(pos, n, u0);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const u32 u = u0 + (u32)i;
    while (j + 1u < n && pos[j + 1u] <= u) ++j;
    const u32 b = byte_of(p, i), e = byt[j];
    if (b != e) {
      const u32 k = __hip_atomic_fetch_add(res, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      if (k < 30u) { res[4 + 2 * k] = u; res[5 + 2 * k] = b | (e << 8); }
    }
  }
}
