# SPDX-License-Identifier: MIT
"""Generates kernels/rates.hip: one throughput probe per vector instruction (see runner/rates.py).

    python3 clean/runner/gen_rates.py > clean/kernels/rates.hip

Every probe runs `iters` iterations of 16 independent instruction chains (one per register), written as inline assembly
so exactly the named instruction is issued, and stores one value per wave so nothing is optimized away. Operand
conventions: x_i is the destination and, when the instruction has a source of the same kind, also a source (a dependent
chain per register, 16 of them: latency is hidden); A, B are loop-invariant registers (A = FP32 or packed halves).
"""
# name -> (register type, assembly line with D = chain register, A, B = invariant registers)
OPS = {
    # FP32 baseline and plain arithmetic
    "fma_f32": ("float", "v_fma_f32 D, D, A, B"),
    "mul_f32": ("float", "v_mul_f32 D, D, A"),
    "add_f32": ("float", "v_add_f32 D, D, A"),
    "max_f32": ("float", "v_max_f32 D, D, A"),
    "min_f32": ("float", "v_min_f32 D, D, A"),
    "med3_f32": ("float", "v_med3_f32 D, D, A, B"),
    "mad_mix_f32": ("float", "v_fma_mix_f32 D, A, B, D op_sel_hi:[1,1,0]"),   # f16 x f16 + f32 -> f32
    # transcendentals
    "exp_f32": ("float", "v_exp_f32 D, D"),
    "rcp_f32": ("float", "v_rcp_f32 D, D"),
    "rsq_f32": ("float", "v_rsq_f32 D, D"),
    "exp_f16": ("u32", "v_exp_f16 D, D"),
    "rcp_f16": ("u32", "v_rcp_f16 D, D"),
    # packed FP16
    "pk_fma_f16": ("u32", "v_pk_fma_f16 D, D, A, B"),
    "pk_mul_f16": ("u32", "v_pk_mul_f16 D, D, A"),
    "pk_add_f16": ("u32", "v_pk_add_f16 D, D, A"),
    "pk_max_f16": ("u32", "v_pk_max_f16 D, D, A"),
    "pk_min_f16": ("u32", "v_pk_min_f16 D, D, A"),
    "dot2_f32_f16": ("float", "v_dot2_f32_f16 D, A, B, D"),
    "dot4_fp8_fp8": ("float", "v_dot4_f32_fp8_fp8 D, A, B, D"),   # 4 FP8 products + FP32 accumulate (RDNA 4)
    "dot4_bf8_fp8": ("float", "v_dot4_f32_bf8_fp8 D, A, B, D"),
    "dot2_sel": ("float", "v_dot2_f32_f16 D, A, B, D op_sel:[1,1,0] op_sel_hi:[1,1,1]"),
    # conversions
    "cvt_pkrtz_f16_f32": ("u32", "v_cvt_pkrtz_f16_f32 D, A, B"),
    "cvt_f32_f16": ("float", "v_cvt_f32_f16 D, A"),
    "cvt_pk_fp8_f32": ("u32", "v_cvt_pk_fp8_f32 D, A, B"),
    "cvt_f32_fp8": ("float", "v_cvt_f32_fp8 D, A"),
    "cvt_pk_f32_fp8": ("u64", "v_cvt_pk_f32_fp8 D, A"),     # two FP8 -> two FP32 in one instruction (64-bit result)
    # integer / bit operations
    "and_b32": ("u32", "v_and_b32 D, D, A"),
    "lshl_b32": ("u32", "v_lshlrev_b32 D, 3, D"),
    "add_u32": ("u32", "v_add_nc_u32 D, D, A"),
    "bfe_u32": ("u32", "v_bfe_u32 D, D, 3, 5"),
    "perm_b32": ("u32", "v_perm_b32 D, D, A, B"),
    "mad_u32_u24": ("u32", "v_mad_u32_u24 D, D, A, B"),
    "pk_add_u16": ("u32", "v_pk_add_u16 D, D, A"),
    "pk_max_u16": ("u32", "v_pk_max_u16 D, D, A"),
    "pk_min_u16": ("u32", "v_pk_min_u16 D, D, A"),
}


def kernel(name, typ, line):
    regs = [f"x{i}" for i in range(16)]
    styp = "u32" if typ == "u64" else typ  # type of the loop-invariant source registers
    body = []
    for i in range(16):
        body.append('        "' + line.replace("D", f"%{i}").replace("A", "%16").replace("B", "%17") + '\\n"')
    out = ", ".join(f'"+v"({r})' for r in regs)
    decl = "\n".join(f"  {typ} {r} = ({typ})(seed + {i});" if typ == "float" else
                     f"  {typ} {r} = ({typ})(seed * 2654435761u + {i}u);" for i, r in enumerate(regs))
    fold = " + ".join(f"as_float({r})" for r in regs)
    return f'''WG_KERNEL_N(rt_{name}, 256)(float* out, int iters) {{
  const u32 seed = wg_lane_id() + 1u;
  {styp} A = ({styp})(seed + 100), B = ({styp})(seed + 200);
{decl}
  for (int it = 0; it < iters; ++it) {{
    __asm__ volatile(
{chr(10).join(body)}
        : {out}
        : "v"(A), "v"(B));
  }}
  out[(size_t)wg_block_id() * 256 + wg_lane_id()] = {fold};
}}
'''


MIX_K = (1, 3, 7)  # v_fma_f32 instructions per v_exp_f32 in the interleaved probes


def mix_kernel(k):
    """8 v_exp_f32 chains and 8 FMA chains in ONE wave, interleaved 1 v_exp : k v_fma_f32 (8 + 8k instructions per
    iteration). If the transcendental unit runs beside the vector ALU the mix takes max(exp alone, fma alone), if not the sum."""
    lines = []
    for i in range(8):
        lines.append(f'        "v_exp_f32 %{i}, %{i}\\n"')
        for j in range(k):
            r = 8 + (i * k + j) % 8
            lines.append(f'        "v_fma_f32 %{r}, %{r}, %16, %17\\n"')
    regs = [f"x{i}" for i in range(16)]
    out = ", ".join(f'"+v"({r})' for r in regs)
    decl = "\n".join(f"  float {r} = (float)(seed + {i});" for i, r in enumerate(regs))
    fold = " + ".join(regs)
    body = chr(10).join(lines)
    return f'''WG_KERNEL_N(rt_mix_exp_fma{k}, 256)(float* out, int iters) {{
  const u32 seed = wg_lane_id() + 1u;
  float A = (float)(seed + 100), B = (float)(seed + 200);
{decl}
  for (int it = 0; it < iters; ++it) {{
    __asm__ volatile(
{body}
        : {out}
        : "v"(A), "v"(B));
  }}
  out[(size_t)wg_block_id() * 256 + wg_lane_id()] = {fold};
}}'''


def main():
    print("// GENERATED by runner/gen_rates.py -- do not edit. Throughput probes, one kernel per vector instruction; see")
    print("// runner/rates.py. Device only.")
    print('#include "wg_platform.h"')
    print("#if defined(__AMDGCN__)")
    print("namespace {")
    print("WG_FN float as_float(float x) { return x; }")
    print("WG_FN float as_float(u32 x) { return __builtin_bit_cast(float, x); }")
    print("WG_FN float as_float(u64 x) { return __builtin_bit_cast(float, (u32)x) + __builtin_bit_cast(float, (u32)(x >> 32)); }")
    print("}  // namespace")
    for n, (t, l) in OPS.items():
        print(kernel(n, t, l))
    for k in MIX_K:
        print(mix_kernel(k))
    # the same FP32 FMA chains written in C: shows whether the compiler's dual-issue (v_dual_*) pairs change the rate
    xs = ", ".join(f"x{i} = (float)(seed + {i})" for i in range(16))
    upd = "\n".join(f"    x{i} = __builtin_fmaf(x{i}, a, b);" for i in range(16))
    fold = " + ".join(f"x{i}" for i in range(16))
    print(f'''WG_KERNEL_N(rt_fma_c, 256)(float* out, int iters) {{
  const u32 seed = wg_lane_id() + 1u;
  const float a = (float)(seed + 100) * 1e-4f, b = (float)(seed + 200) * 1e-4f;
  float {xs};
  for (int it = 0; it < iters; ++it) {{
{upd}
  }}
  out[(size_t)wg_block_id() * 256 + wg_lane_id()] = {fold};
}}''')
    print("#endif")


if __name__ == "__main__":
    main()
