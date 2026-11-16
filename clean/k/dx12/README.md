<!-- SPDX-License-Identifier: MIT -->
# Model K on Direct3D 12

The D3D12 ports of Model K's HIP kernels (issue #15). Each port reproduces one HIP kernel of `clean/k/kernels` bit for
bit; the HIP kernel stays the reference. The ports are built only with `--model-k`
(`python clean/dx12/build_dx12.py --out <dir> --model-k`), so a build without it is Model M's kernel set, and
`--model-l` Model L's, unchanged (`clean/dx12/kernel_sets.json`, `clean/dx12/tests/test_kernel_sets.py`).

| Path | What |
|---|---|
| `kernels/*.hlsl` | the ports, declared with `// @wg-kernel` header lines like the other D3D12 kernels (`co=` names the HIP code object whose kernel the port reproduces) |
| `hlsl/*.hlsli` | headers shared by the K ports (included with `clean/dx12/hlsl`) |
| `k_port_check.py` | the byte-for-byte check of a family of ports against HIP, below |

A HIP kernel that takes a struct with pointers inside by value declares that argument as `s<N>:<fields>`
(`clean/runtime/struct_args.h`): the hybrid and the native device turn each pointer into an arena offset and pass the
other bytes as they are.

## The port check

```bat
python clean\k\dx12\k_port_check.py --build <dir> --rocm "%ROCM_PATH%" --prepare --family expo,warm
python clean\k\dx12\k_port_check.py --build <dir> --rocm "%ROCM_PATH%" --run --family expo,warm --json out.json
```

`--prepare` (CPU) builds the HIP code objects the families need, the D3D12 ports and `kernel_replay.exe`; `--run` (GPU,
under the machine's GPU lock) runs every case on HIP in the process and on D3D12 through `kernel_replay` on the same
input bytes, and compares the raw bytes. Rules, as for the Model L checks: every output buffer has 64 bytes of slack
and starts filled with its run's own sentinel (HIP `0xA5`, D3D12 `0x5A`); the defined bytes must be identical and the
slack must keep its sentinel; special values are among the inputs; a negative control (the port given another input
than HIP) must be caught; Model M's `scale_rgb` runs through the same replay as a control. Output: counts and verdicts.

Families: `output` (the output stage of `k_io.co`: `k_out_color` followed by `k_out_pixel_c` or `k_out_pixel`, the
three game-output formats (RGBA32F, RGBA16F rows, RGB32F), colour alphas that are negative, above 1 or out of the FP16
range, special values in the colour, depth, motion and head map; the stored colour taps and the filter-tap values of
`k_out_color` are compared as well as the image and the histories; the raw bytes are identical in every case; the `_f`
kernels of `k_fast` and `k_out_hidden` are not ported); `input` (the exact tiled input stage `kx/k_in_tiled` of `k_io_exact.co`: 16 flag sets of HDR, low-resolution
motion, inverted depth and reset, odd and tiny base grids down to 1 x 1, the exposure scalars from a device buffer as the
runtime passes them, infinity, huge values, zero and -0 in the colour, depth, motion and histories, NaN in a third of
the cases; the raw bytes are identical in every case; the port needs the adapter's binary64 fma and division (manifest
`fp64: ext`, dropped on an adapter without them), and flushes FP32 denormals); `narrow` (the narrow network stages of `k_block.co`: `k_stage_e2`, `k_stage_e0`, `k_stage_e1`, `k_stage_d2`,
`k_stage_d1`, `k_stage2_e0`, `k_stage2_e1`, `k_stage2_d1`, `k_stage2_d0_hp`, `k_stage2_d0_hidden`) on random stage weights
and inputs, the stage runner's launch plans run on HIP and replayed on D3D12 (`k_plan_bridge.py`); the inputs of every
third case carry special values (infinity, +-65504, zero, -0, subnormals) and of every third case NaN as well; the
half-grid control launches a stage with half its workgroups. `wide` (the wide stages of `k_block_wide.co`:
`k_stagew2_e3`, `k_stagew4_e4`, `k_stagew2_e4`, `k_stagew4_m`, `k_stagew4_d4`, `k_stagew2_d4`, `k_stagew2_d3`: one window
per 128 NG lanes) is checked the same way. Not ported, because no context can reach them: the NG = 4 forms of `e3` and `d3` (`k_stagew4_e3`, `k_stagew4_d3`), which the HIP build has. `wide_ng` in `clean/k/runtime/runtime_k.cpp` gives `e3` and `d3` NG = 2 on the exact network whatever the window count, the sizes or the options (only `e4` and `d4` look at the window count), and the FP16 networks, which take other kernels, are refused on the native device; `clean/k/tests/test_k_wide_ng.py` pins the rule, the two places that build the kernel names and the declared kernel set. A replay of `k_stagew4_d3` on D3D12 was reported to differ from the HIP capture; if a change ever makes the form reachable, that test fails until the form is ported and checked. `expo` (`k_expo_scalars`) against `k_io.co`; `warm`
(`k_warm`, a cache warm-up with no HIP kernel: checked against its definition, through the struct argument path).

`legacy` (the legacy exact input kernels of `k_io_exact.co`, `kx/k_in_color`, `kx/k_in_reset`, `kx/k_in_base`, in
`kernels/k_in_legacy.hlsl`; the runtime takes them where the tiled kernel does not apply: a render size above 0.68 x
the output up to 1.0 (native AA, DLAA), a jitter outside [-0.5, 0.5], `input_tiled=0`): the same flag sets as `input`,
every fourth case at render size = output size or with a jitter outside the guard; the inputs, the intermediates `cpre`
and `cpre2` and the output `x0` are compared raw; the half-grid control launches `k_in_base` with half its workgroups).
`direct` (the forms for the native executor, which read the game's colour, depth, motion and exposure textures and write
the output texture: `kx/k_in_tiled_direct`, `kx/k_in_color_direct`, `kx/k_in_base_direct`, `k_out_color_direct`,
`k_out_pixel_c_direct`, `k_out_pixel_direct`, `k_expo_scalars_direct`, `exposure_auto_diet_direct`): the check binds
real textures (RGBA16F colour, R32F depth, RG16F motion, exposure, an RGBA16F output) through `kernel_replay`'s direct
mode and compares the output texture and the buffers with the HIP kernels' run on the raw buffer route
(`KIO_C16` / `KIO_M16`, with the motion scale and offset folded as the interop passes them).

`mixprobe` (D3D12 only, no HIP kernel: checked against the exact rounding): the softmax weight rounding of every
softmax stage port, A = f16(w inv) rounded once from the exact product (`hlsl/k_smx.hlsli`). The ports write it as the
FP32 product plus its exact residual fed into the driver's round-to-nearest-even fragment conversion, which the gfx1201
compiler folds into one `v_fma_mixlo/hi_f16` (the mix form); the `__rodd` ports round the product to odd first (the
form the ports had before; the same bits). `kernels/k_mix_probe.hlsl` runs the same helper on the records of
`k_mix_cases.py`: products whose FP32 rounding is an FP16 midpoint with a residual of either sign (normal and subnormal
FP16), exact midpoints (ties to even) and ordinary products; every result must equal the exact single rounding
(Fraction arithmetic), for `k_mix_probe` and `k_mix_probe__rodd`, and the records must tell a second rounding through
FP32 apart (the residual-dropped control `K_SMX_MUT=1` fails this family). Two more guards keep the fold: the build's
ISA check (`clean/dx12/isa_checks.json`: one `v_fma_mix*_f16` per weight in every mix-form port, none in the `__rodd`
ports) and the native D3D12 runtime, which runs `k_mix_probe` on the records of `clean/k/runtime/k_mix_probe.h` at
create and launches the `__rodd` stage ports when the result is not exact (kernels line: `smx=mix` or `smx=rodd(...)`;
test hook `WG_TEST_K_MIX_PROBE_FAIL=1`). The `__rodd` ports give the same bits, but the switch is never silent: it adds
one note of create (`wg_rt_notes`, which the executor logs; `test_k_runtime_d3d12.py`
`test_native_softmax_switch_is_logged`).

NaN: IEEE 754 does not define the sign of a NaN that a stage generates or propagates, and where the input holds NaN the
two devices can differ in it (the decoder stages). For the cases with NaN in the input the check compares the raw bytes
and, where they differ, accepts only the elements that are NaN on both sides and differ in the sign bit alone (the
exponent, the quiet bit and the payload equal), counted separately; every other differing byte, a NaN with another
payload included, fails the case (`clean/k/tests/test_k_port_check_nan.py`). Without NaN in the input the raw bytes are
identical.

Known limit, shared with Model M's ports: D3D12 flushes FP32 denormals (inputs and results) to zero and HIP keeps them.
`k_expo_scalars` only meets one with a pre-exposure above about 8.5e37, so the check's inputs keep clear of them.
