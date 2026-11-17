// SPDX-License-Identifier: MIT
// Q11 part C: probe of the reconstruction's two fused-rounding forms, bit for bit (clean/l/runner/l_post_c12_check.py).
// Built as l_post_c12_probe.co (WG_LP_C1 = 1, WG_CR_C2 = 1: the forms of l_post__morig.co) and
// l_post_c12_probe__mut.co (WG_LP_C1 = 0: item 3's exponent, Model L's kernel; WG_CR_C2 = 2: w1 with 2.5 f2 fused):
// the check must find the mutant's values different. Evaluation only; not in any release build.
#include "l_post.hip"

// eps[i] = l_eps(u[i], v[i], kA[i], kB[i], kC[i]) (FP16 bits in, FP16 bits out)
WG_KERNEL_N(l_probe_eps, 256)(const unsigned short* in, unsigned short* out, int n) {
  const int i = (int)(wg_block_id() * 256u + wg_lane_id());
  if (i >= n) return;
  const unsigned short* q = in + 5 * i;
  const h16 e = l_eps(__builtin_bit_cast(h16, q[0]), __builtin_bit_cast(h16, q[1]), __builtin_bit_cast(h16, q[2]),
                      __builtin_bit_cast(h16, q[3]), __builtin_bit_cast(h16, q[4]));
  out[i] = __builtin_bit_cast(unsigned short, e);
}

// w[4 i .. 4 i + 3] = cr_weights(f[i]) (float32)
WG_KERNEL_N(l_probe_cr, 256)(const float* f, float* w, int n) {
  const int i = (int)(wg_block_id() * 256u + wg_lane_id());
  if (i >= n) return;
  float r[4];
  cr_weights(f[i], r);
  for (int k = 0; k < 4; ++k) w[4 * i + k] = r[k];
}
