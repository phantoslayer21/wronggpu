// SPDX-License-Identifier: MIT
// Model K activation (spec 40, task K2): exhaustive check kernels.
//
// k_act_table:   out[i] = act(x) for the FP16 value x with bit pattern i, i = 0 .. 65535 (scalar FP16 form).
// k_act_table2:  the same through the packed form (two values per operation, pairs (2j, 2j + 1)).
// clean/k/runner/k_stages.py (stage "k.act") launches both and compares with the NumPy oracle bit for bit.
// k_fp_probe:    the FP32 double-word kf_log2 / kf_ln / kf_exp2 (k_fp.h) against their binary64 path, bit for bit, over
//                a range of input bit patterns (clean/k/tests/test_k_io_emu.py TestFp32Functions, exhaustive).
#include "k_common.h"
#include "k_fp.h"

WG_KERNEL_N(k_act_table, 128)(unsigned short* out) {
  const u32 i = wg_block_id() * 128u + wg_lane_id();
  if (i < 65536u) {
    const h16 x = __builtin_bit_cast(h16, (unsigned short)i);
    out[i] = __builtin_bit_cast(unsigned short, wgk_act(x));
  }
}

WG_KERNEL_N(k_act_table2, 128)(unsigned short* out) {
  const u32 j = wg_block_id() * 128u + wg_lane_id();
  if (j < 32768u) {
    const h16 x0 = __builtin_bit_cast(h16, (unsigned short)(2u * j));
    const h16 x1 = __builtin_bit_cast(h16, (unsigned short)(2u * j + 1u));
    const v2h x = {x0, x1};
    const v2h y = wgk_act2(x);
    const h16 y0 = y[0], y1 = y[1];  // bit-cast scalars, not vector elements (see k_common.h)
    out[2u * j] = __builtin_bit_cast(unsigned short, y0);
    out[2u * j + 1u] = __builtin_bit_cast(unsigned short, y1);
  }
}

struct KFpProbe {
  u32 base, count, fn, pad;
  u32* out;  // [0] differences, [2] next slot, [4 + 3k ..] (input, double-word result, binary64 result) bits
};

WG_KERNEL_N(k_fp_probe, 256)(KFpProbe P) {
  const u32 i = wg_block_id() * 256u + wg_lane_id();
  if (i >= P.count) return;
  const float x = __builtin_bit_cast(float, P.base + i);
  float a, b;
  if (P.fn == 0) {
    a = kf_log2(x);
    b = (float)kd_log2((double)x);
  } else if (P.fn == 1) {
    a = kf_ln(x);
    b = (float)kd_log((double)x);
  } else if (P.fn == 2) {
    a = kf_exp2(x);
    b = (float)kd_exp2((double)x);
  } else {
    a = kf_rsqrt(x);
    b = kd_rsqrt32(x);
  }
  const u32 ua = __builtin_bit_cast(u32, a), ub = __builtin_bit_cast(u32, b);
  if (ua != ub && !(a != a && b != b)) {
    __atomic_fetch_add(&P.out[0], 1u, __ATOMIC_RELAXED);
    const u32 k = __atomic_fetch_add(&P.out[2], 1u, __ATOMIC_RELAXED);
    if (k < 16u) {
      P.out[4 + 3 * k] = P.base + i;
      P.out[5 + 3 * k] = ua;
      P.out[6 + 3 * k] = ub;
    }
  }
}

// out[i] = 2^E for the FP16 bit pattern E = i as the output stage computes it on the GPU (v_exp_f16; spec 43 5.4 needs
// h(2^E) exactly; clean/k/tests/test_k_io_emu.py TestExp2Half compares all 65536 values with the exact table)
WG_KERNEL_N(k_exp2h_probe, 256)(unsigned short* out) {
  const u32 i = wg_block_id() * 256u + wg_lane_id();
  if (i >= 65536u) return;
  const h16 E = __builtin_bit_cast(h16, (unsigned short)i);
#if defined(__AMDGCN__)
  const h16 g = __builtin_elementwise_exp2(E);
#else
  const h16 g = (h16)kd_exp2((double)E);  // exact enough for the emulator; the GPU instruction is what is tested
#endif
  out[i] = __builtin_bit_cast(unsigned short, g);
}

WG_EMU_REGISTER(k_act_table)
WG_EMU_REGISTER(k_act_table2)
WG_EMU_REGISTER(k_exp2h_probe)
WG_EMU_REGISTER(k_fp_probe)
