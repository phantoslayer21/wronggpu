# Clean implementation: Swin stages (milestone 1), input encoder (enc0, spec 03), coefficient decoder (dec0, spec 05), output reconstruction and filter (spec 06), auto-exposure (spec 07), per-frame runtime core and D3D12 interop (spec 10)

| Path | Content |
|---|---|
| `reference/` | NumPy oracles: spec 02 (`swin.py`: `run_stage`, `swin_block`, `patch_merge`, `expand`, FP8 helpers), spec 06 (`post.py`: `reconstruct`, `output_filter`) |
| `k/` | model K (specs 40, 41, 44, 49): reference, FP16 kernels, runner stages (`Runner(..., model="K")`), timing harness, runtime layer table; see `k/README.md` |
| `reference/encoder.py` | NumPy oracle for spec 03 (input encoder, stage enc0) |
| `reference/decoder.py` | NumPy oracle for spec 05 (coefficient decoder, stage dec0) |
| `reference/exposure.py` | NumPy oracle for spec 07 (auto-exposure, stage exposure) |
| `kernels/swin.hip` | gfx1201 kernels: fused Swin block (FP8 WMMA), patch merge, expand |
| `kernels/encoder.hip` | gfx1201 kernel `input_encoder`: spec 03 per-pixel encoding + projection (FP8 WMMA), motion surfaces |
| `kernels/decoder.hip` | gfx1201 kernel `coef_decoder`: spec 05 projection (FP8 WMMA) + per-pixel coefficients and latent |
| `kernels/exposure.hip` | gfx1201 kernels `exposure_partial`, `exposure_final`: spec 07 log-average exposure; `exposure_pair` (runtime state) |
| `runtime/` | C++ per-frame runtime (spec 10 core): device layer (HIP loaded at run time), model loading and packing, frame state and stage order; C API `wg_runtime.h` |
| `runtime/interop/` | D3D12 interop (spec 10 sections 1, 6, 8): `wgGetUpscalerBackend` of `wronggpu_runtime.dll` for the FidelityFX front end (interface version 2, the front end's command-list split), input/output passes, HIP executor; offline D3D12 test `tests/run-tests.sh` (MinGW + Wine) |
| `kernels/swin_v2.hip` | second, speed-oriented design of the five block kernels (build variants `v2*`; merge/expand from `swin.hip`) |
| `kernels/probe.hip` | probe kernels for the GPU self-test |
| `kernels/post.hip` | gfx1201 kernels of spec 06: output reconstruction (`post_recon`) and output filter (`post_filter`) |
| `kernels/wg_platform.h` | GPU primitives (clang AMDGPU builtins) and software FP8 conversion |
| `kernels/wg_mem.h` | 32-bit-offset memory access, two-load bilinear footprint, scheduling pins (enc0, dec0, exposure) |
| `kernels/dec1_dec0.hip` | prototype: dec1's last Swin block fused with dec0 (`swin_block_c64_add_dec0`), bit-identical to the separate kernels |
| `kernels/baseline/` | previous versions of the enc0/dec0/exposure kernels, built only as timing variant "v1" |
| `kernels/emu/` | host lane emulator: runs the same kernel source on a CPU (tests only) |
| `runner/` | spec 90 interface: `build.py`, `stage_runner.py` (`Runner`), `enc0.py`, `dec0.py`, `exposure.py`, `post.py` (packing/launch of those stages), `post_bench.py`, `perf_variants.py` + `perf_bench.py` (enc0/dec0/exposure variants and timing), `pipeline.py` (whole frame), `synthetic_model.py`, `hip.py` (ctypes HIP), `selftest.py` |
| `tests/` | unit tests: reference vs slow loop versions; emulated kernels vs reference |
| `FORMATS.md` | feature-map and packed-weight layouts |
| `PERF.md` | performance milestone for enc0, dec0, exposure: analysis, changes, variants |

## Spec team: build and run (Windows, HIP SDK 7.2)

```
python clean\runner\build.py --rocm "C:\Program Files\AMD\ROCm\7.2" --out build
python clean\runner\selftest.py --build build --rocm "C:\Program Files\AMD\ROCm\7.2"
```

Then, in Python:

```
import sys; sys.path.insert(0, r"<repo root>")
from clean.runner.stage_runner import Runner
r = Runner("build", r"C:\Program Files\AMD\ROCm\7.2")
w = r.pack_weights("enc1", tensors)          # spec 90 logical tensors
out = r.run("enc1", w, x, (ox, oy))          # {"skip": [H, W, 64], "merged": [H/2, W/2, 96]}
ms = r.time("enc1", w, x, (ox, oy))          # median ms over 50 repeats; inputs already on the GPU
```

Stage enc0 (spec 03): `x` is the dict of spec 90, `offset` is ignored:

```
w = r.pack_weights("enc0", {"W": W, "b": b, "T": T, "sigma1": s1, "sigma2": s2})
out = r.run("enc0", w, x, None)              # {"features": [fh/2, fw/2, 64], "m_near" / "m_far" / "m_center": [rh, rw, 2]}
ms = r.time("enc0", w, x, None)              # the input_encoder kernel only
```

Stage dec0 (spec 05): `w = r.pack_weights("dec0", {"W": W, "b": b, "S": S, "T": T})`, then
`r.run("dec0", w, x, None)` → `{"A", "B", "C": [rh, rw, 3], "latent": [rh, rw, 4]}` (spec 90 value grids).

Stage exposure (spec 07, no weights): `r.run("exposure", b"", {"color": c, "pre_exposure": 1.0, "alpha": 0.18},
None)` → `{"e": float}`.

Whole frame (spec 10 core, spec 90 "Whole frame"): `build.py` also builds the runtime library. Then

```
from clean.runner.pipeline import Pipeline
p = Pipeline("build", r"C:\Program Files\AMD\ROCm\7.2", model_dir, (1706, 960), (2560, 1440),
             {"depth_inverted": True, "auto_exposure": False})
out = p.frame(x, debug=True)   # {"out": ..., "enc0": {...}, ..., "dec0": {...}, "post": ..., "filter": ..., "state": ...}
ms = p.time_frame(x)
```

`model_dir` is the converter's output folder. The runtime loads `model.wgcm` there (checked against
`data/preset-M.v1.map.json`, or `map_path=`), else the `--raw` layout; `model_format="raw"` forces raw.
`python clean/runner/synthetic_model.py --out <dir>` writes a synthetic raw model.

Kernel choice per stage (whole-frame A/B and 10-T1 with any variant, no new build of the runtime). Defaults are the
fastest exact variants: Swin `swin.co` (gelusig), enc0 / dec0 `base` (since clean/integration built as the former
`fragglobal` / `late`), exposure `base`, post `post_recon2_r`, filter `post_filter_c1_s4`
(bit-identical to post_recon / post_filter). Build the variant code objects first
(`build.py ... --variants all --perf-variants`); post and filter kernels are all in `post.co`.

```
p = Pipeline(..., kernels={"post": "post_recon4_fast"})          # any subset of swin, enc0, dec0, exposure, post, filter
p = Pipeline(..., kernels={"post": "post_recon6"})  # pre-pass + reconstruction (bit-identical; also _bil, _eb, _eb_bil)
p = Pipeline(..., kernels={"swin": "v2"})            # second Swin design (build.py --variants all)
p = Pipeline(..., kernels={"post": "post_recon", "filter": "post_filter", "enc0": "lds", "dec0": "early"})  # previous
p.kernels()   # what is in use, e.g. {"swin": "swin.co", "enc0": "encoder.co", ..., "post_last": "post_recon2_r"}
```

Whole-frame A/B timing of kernel choices, interleaved rounds (numbers only), e.g. the Swin GELU variants (`base` =
gelusig; `geluexact` = exact; `gelu16` = packed FP16 polynomial; `gelu725` = A&S 7.1.25):

```
python clean\runner\frame_ab.py --build build --rocm "C:\Program Files\AMD\ROCm\7.2" --model <model_dir> ^
    --configs base swin=geluexact swin=gelu16 swin=gelu725 --frame <one recorded frame .npz> --json gelu_ab.json
```

A configuration is `base` or `key=value[,key=value...]` with the keys of `kernels=` below. The integration candidate
sets (spec team results on perf-enc 469572c and Swin 1d4babd; `base` already has dec0 `nt`):

```
python clean\runner\frame_ab.py --build build --rocm "C:\Program Files\AMD\ROCm\7.2" --model <model_dir> ^
    --frame <frame .npz> --json sets.json --configs base dec0=temporal swin=geluexact ^
    swin.enc2=smax16,swin.b0=gelu16,swin.b1=av8,swin.dec1=nt1 ^
    swin.enc2=smax16,swin.b0=gelu16,swin.b2=gelu16,swin.b4=gelu16,swin.b1=av8,swin.b3=av8,swin.b5=av8,swin.dec1=nt1
```

(A) `base`; reference `dec0=temporal` (the previous dec0 stores); (B) `swin=geluexact` (timing reference);
(D) the per-stage Swin bests as measured (b0 `gelu16`, b1 `av8`); (D2) the same extended to b2 … b5 by window
parity. For 10-T1 the same strings go into `Pipeline(..., kernels={...})` (keys `swin.<stage>`) or, in game,
`WG_SWIN_STAGE_VARIANTS=enc2=smax16,b0=gelu16,b1=av8,dec1=nt1`. Build with `--variants all --perf-variants`.

In-game A/B payload: `python clean\runner\build.py --rocm "C:\Program Files\AMD\ROCm\7.2" --out payload --payload`
(= `--variants all --perf-variants`) writes `wronggpu_runtime.dll`, the converter map, the default code objects
(`swin.co`, `encoder.co`, `decoder.co`, `exposure.co`, `post.co`, `post_w64.co` (the post kernels in wave64, the
runtime's default), `swin__bmegahg.co` (the b0 … b5 megakernel, the runtime's default; built by every build),
`frame.co`, `probe.co`, `overlap.co`, `rates.co`), every
candidate `swin__<name>.co` with `variants.json`, and `encoder__<name>.co` / `decoder__<name>.co` /
`exposure__<name>.co` with `perf_variants.json`, all in one folder. The DLL looks for code objects in its own folder,
so copying that folder as the game's WrongGPU folder makes every variable below work without a rebuild. All post and
filter kernels are in `post.co`.

The same choice through environment variables (also read by `wronggpu_runtime.dll` in game):
`WG_SWIN_VARIANT`, `WG_ENC0_VARIANT`, `WG_DEC0_VARIANT`, `WG_EXPOSURE_VARIANT`, `WG_POST_KERNEL`, `WG_FILTER_KERNEL`
(for example `set WG_POST_KERNEL=post_recon4_fast`). Table of values: `clean/FORMATS.md`, "Kernel choice".

In-game runtime (Windows): the same `build.py` run writes `wronggpu_runtime.dll` (core + D3D12 interop) with the code
objects and the converter map into `build`; the FidelityFX front end loads it (contract and options:
`clean/FORMATS.md`, "D3D12 interop"). Smoke test without the game: build
`clean/runtime/interop/tests/test_interop_d3d12.cpp` (one file) and run `test_interop_d3d12.exe build\wronggpu_runtime.dll`
(D3D12 passes) and `... hip` (the network, with an installed model).

`time` covers all kernels of the stage (enc: block + merge, dec: expand + block). If `selftest.py` reports a `d_row`
other than `gfx12`, set `WG_D_ROW` to the reported table (for example `set WG_D_ROW=[[0,2,4,...],[1,3,...]]`)
before running.

## Spec team: stages post and filter (spec 06)

```
r = Runner("build", r"C:\Program Files\AMD\ROCm\7.2")
w = r.pack_weights("post", {"s": s})             # s as a 0-d array
out = r.run("post", w, x)                        # {"history", "reduced", "display"}; x as in spec 90
ms = r.time("post", w, x)
out = r.run("filter", b"", {"display": d, "exposure": e, "output_size": (ow, oh)})   # {"out": [oh, ow, 3]}
ms = r.time("filter", b"", {...})
```

Timing at the in-game size and a small GPU-vs-reference check, synthetic data only (numbers only):

```
python clean\runner\post_bench.py --build build --rocm "C:\Program Files\AMD\ROCm\7.2" --json post_bench.json
```

It also times the filter variants `post_filter_s4`, `post_filter` (default, S = 8) and `post_filter_s16` (output row
pairs per lane) and checks that their outputs are identical; `r.time("filter:post_filter_s4", b"", x)` times one.

## Spec team: fused reconstruction + output filter (post_fused*)

```
python clean\runner\post_fused_bench.py --build build --rocm "C:\Program Files\AMD\ROCm\7.2" --json fused.json
```

Builds `build\post_w64.co` (the post kernels in wave64 mode) if missing, checks on the GPU that every fused variant is
bit-identical to post_recon + post_filter and within spec 09 of the reference (synthetic data), and times the
two-kernel path and the fused variants at the in-game size, wave32 and wave64 (numbers only). From Python:
`clean.runner.post.run_fused(runner, w, x, kernel="post_fused", code_object=None)` with the spec 90 "post" dict x
returns `{"history", "reduced", "out"}`. Layouts and the runtime launch: `FORMATS.md`, "Fused reconstruction + output
filter". The latency-scheduled `post_fused3*` kernels (`runner/post.py: FUSED3_VARIANTS`; exact `post_fused3_h6`,
`post_fused3_h6_o16`, and approximations) are checked and timed by the same script, and timed cold by
`post_traffic.py`.

Drop-in variants (post_recon2*, post_recon3*, post_filter_c1*: bit-identical or within spec 09; post_recon4_*: SPEED
APPROXIMATIONS for the whole-pipeline rule, listed in `runner/post.py: RECON4_VARIANTS`) and the pre-pass variants
(post_prep + post_recon5*, post_recon6*: `PREP_VARIANTS`; post_recon5 and post_recon6 bit-identical, the `_bil` /
`_eb` switches approximations): GPU checks and interleaved timing (`post_variants.py: APPROXIMATIONS` lists every
approximation with what it changes):

```
python clean\runner\post_variants.py --build build --rocm "C:\Program Files\AMD\ROCm\7.2" --rounds 3 --json variants.json
```

Memory-traffic floor of post + filter for each image layout (two-kernel / fused; history 8, 6 or 4 bytes per pixel;
output f32 or FP16), timed with pure load/store kernels at the in-game size, warm and with the last-level cache flushed,
next to the real kernel pairs:

```
python clean\runner\post_traffic.py --build build --rocm "C:\Program Files\AMD\ROCm\7.2" --rounds 3 --json traffic.json
```

Output mode D = O (runtime option recon = output): `post_recon2_r_out` reconstructs at D = O and writes the decoded
FP16 output directly (no display image, no filter); interface in `FORMATS.md`, "Output mode D = O"; timed cold by
`post_traffic.py`.

Where the time goes (timing-only ablation builds, outputs wrong on purpose; `WG_AB_*` switches in `kernels/post.hip`):

```
python clean\runner\post_ablate.py --build build --rocm "C:\Program Files\AMD\ROCm\7.2" --rounds 3 --json ablate.json
```

## Spec team: timing sweep (no real model data needed)

`--variants all` also builds the kernel variants listed in `runner/variants.py` (workgroup size, windows per wave,
merge/expand shape). `sweep.py` times every stage at the in-game sizes with synthetic data and checks that every
variant gives bit-identical output to the base build:

```
python clean\runner\build.py --rocm "C:\Program Files\AMD\ROCm\7.2" --out build --variants all
python clean\runner\sweep.py --build build --rocm "C:\Program Files\AMD\ROCm\7.2" --json sweep.json
```

Add `--ablations` to the build command to also get twenty timing-only builds with part of the block switched off
(GELU, the whole MLP, the attention softmax, the weight staging/barriers, the MLP's W1 or W2 matrix multiplies, both
of them, the FP8 conversion of the hidden values, both of GELU and the conversion, the MLP's LDS weight reads, the
RMSNorms, the global loads, the final store, the bias adds, the attention soft cap or softmax reductions, or the FP8
conversions in the norm / attention paths). Their output is wrong on purpose; the
sweep lists them separately as "removes X ms = Y % of base", which shows where the time goes.

The output (numbers only) lists, per stage and variant, the median time, the split per kernel (block / merge /
expand) and the number of mismatches against the base variant (must be 0). `Runner(build, rocm, variant="t128")`
runs a single variant.

## Spec team: enc0 / dec0 / exposure timing (performance milestone, `PERF.md`)

```
python clean\runner\build.py --rocm "C:\Program Files\AMD\ROCm\7.2" --out build --perf-variants
python clean\runner\perf_bench.py --build build --rocm "C:\Program Files\AMD\ROCm\7.2" --json perf.json
```

Times every variant and timing-only ablation of `runner/perf_variants.py` at the in-game size (synthetic data),
counts output values that differ from the default build, and checks the default build against the references.
## Spec team: can matrix and vector work overlap? (one run, synthetic data)

```
python clean\runner\build.py --rocm "C:\Program Files\AMD\ROCm\7.2" --out build      # also builds overlap.co
python clean\runner\overlap_probe.py --build build --rocm "C:\Program Files\AMD\ROCm\7.2" --json overlap.json
```

Runs micro-kernels that issue WMMA work and vector work (FMA, `v_exp`) in the same wave (interleaved 1:8, 1:32) and in
different waves, and prints an overlap fraction: 1.0 = they run side by side, 0.0 = they serialize. Also prints the
sustained WMMA rate, a dependent-chain WMMA rate and the FMA / `v_exp` rates. This decides whether the next steps are
about overlapping (variants `ovl`, `ovl64`, `ovl128`) or only about cutting vector instructions.

## Spec team: enc0 / dec0 / exposure timing (performance milestone, `PERF.md`)

```
python clean\runner\build.py --rocm "C:\Program Files\AMD\ROCm\7.2" --out build --perf-variants
python clean\runner\perf_bench.py --build build --rocm "C:\Program Files\AMD\ROCm\7.2" --json perf.json
```

Times every variant and timing-only ablation of `runner/perf_variants.py` at the in-game size (synthetic data),
counts output values that differ from the default build, and checks the default build against the references.

## Spec team: fused merge / expand (variant `fuse`)

`build.py --variants fuse` (or `all`) builds nothing new (the fused kernels are always in the .co); the variant makes the
runner launch them: patch merge is done at the end of the encoder blocks (enc1, enc2) and expand at the start of the
decoder blocks (dec2, dec1), so those stages are one kernel launch. The output is bit-identical to base
(`sweep.py` checks). Compare `fuse` with `base` per stage in the sweep; the kernel split in the output shows one entry.

## Spec team: what does each vector instruction cost? (one run, synthetic data)

```
python clean\runner\build.py --rocm "C:\Program Files\AMD\ROCm\7.2" --out build      # also builds rates.co
python clean\runner\rates.py --build build --rocm "C:\Program Files\AMD\ROCm\7.2" --json rates.json
```

One probe per instruction (16 independent chains per wave, whole GPU): v_fma_f32 (baseline), FP32 arithmetic, v_exp/rcp/rsq
(f32 and f16), packed FP16 (v_pk_fma/mul/add/max/min), v_dot2_f32_f16, v_fma_mix_f32, the conversions the hidden path uses
(v_cvt_pkrtz_f16_f32, v_cvt_f32_f16, v_cvt_pk_fp8_f32, v_cvt_f32_fp8, v_cvt_pk_f32_fp8), the FP8 dot products
(v_dot4_f32_fp8_fp8, v_dot4_f32_bf8_fp8; 4 products per instruction, compare with v_pk_fma_f16 and v_fma_f32) and a few integer / bit
operations. Prints the rate and the cost in FMA slots. The issue-slot model assumes 1 slot for everything except
transcendentals (4); the hidden-path variant gelu16 (packed polynomial, dot2, pkrtz conversions) gained much less in
enc1 than that model predicts, and these numbers show which of those instructions is not 1 slot. `kernels/rates.hip` is
generated by `runner/gen_rates.py`. The last rows interleave `v_exp_f32` with `v_fma_f32` in one wave (1:1, 1:3, 1:7) and
print an overlap fraction: 1.0 means the transcendental unit works beside the vector ALU (a transcendental then costs
almost nothing next to enough independent ALU work), 0.0 means they add up. The variants `gelumix1/2/3` (k of the 4
value pairs of every hidden tile by the packed polynomial, the rest by the sigmoid form) are for the case in between.

## Spec team: how far is the MLP tile loop from the sum of its parts? (one run, synthetic data)

```
python clean\runner\build.py --rocm "C:\Program Files\AMD\ROCm\7.2" --out build      # also builds mlpbench.co
python clean\runner\mlp_probe.py --build build --rocm "C:\Program Files\AMD\ROCm\7.2" --json mlp_probe.json
```

The tile loop of the block's MLP (W1 chain, GELU + FP8 pack, W2) alone, weights from a fixed LDS unit, no global loads, no
staging, no barriers, for C=64 (KC=4) and C=128 (KC=8), with each part removable, at 2 / 4 / 8 / 12 waves per SIMD. Prints
the SIMD time per tile, the ratio of the full loop to the sum of its parts, and `111p`: the full loop with the LDS fragments
prefetched ahead of the GELU (what the variants `pf1` / `pf2` do in the real kernel; `pf2` also loads the next tile's W1
fragments before the current tile's W2 WMMAs). If the full loop is much more than the sum of its parts, or `111p` is much
faster than `111`, the loop is paying latency (LDS round trips, WMMA to vector hand-offs) rather than issue slots.

## Spec team: scaling check (does a small grid pay a fixed or tail cost?)

```
python clean\runner\sweep.py --build build --rocm "C:\Program Files\AMD\ROCm\7.2" --scaling --json scaling.json
```

Times the base build of every stage at 0.5x, 1x, 2x and 4x the in-game map area and prints the time per window. If the
time per window falls a lot at the larger sizes, the in-game launches are paying a ramp-up or last-round cost.

## Spec team: where does each stage wait? (per-phase wave timing, synthetic data)

```
python clean\runner\build.py --rocm "C:\Program Files\AMD\ROCm\7.2" --out build --variants all --profile
python clean\runner\profile_stages.py --build build --rocm "C:\Program Files\AMD\ROCm\7.2" --json profile.json
```

`--profile` adds instrumented builds of the block kernels (exact results). A sample of the waves (every `stride`-th, about
512 per stage by default) reads a clock at the phase boundaries: load, norm1, QKV, attention, projection + residual,
head-weight staging, norm2, W1, GELU + pack, W2, MLP-weight staging, store. Version 1 instrumented every wave through the
realtime message and slowed the kernel about 7x, so the numbers described the instrumentation; now:

- `prof` (default) reads the shader cycle counter (`s_getreg`, scalar, no message traffic) and drains the memory pipeline
  at each boundary so each phase pays its own latency; `prof2` does not drain; `profrt` uses the old realtime message
  (compare with `prof` to validate the cycle counter; the tool converts cycles to microseconds with the realtime clock
  each wave also reads at its start and end).
- The tool prints, per stage, the block-kernel time of the base build, of the instrumented build with no wave
  instrumented, with the sampled stride, and with stride 1. The sampled time must be close to the base time; if it is not
  the phase numbers do not describe the normal kernel (tell us, and try `--stride 256`).
- Per stage also: waves in flight (mean / peak, from the sampled start and end times scaled by the stride), the
  distribution of wave start times (rounds of workgroups, tail), how many hardware ids were seen (decode of HW_ID1: more
  than 128 distinct ids means the decode is wrong), and per phase the mean and max over the instrumented waves and the
  share of the average wave (the cost of the clock reads is measured and removed from phases and total, so shares add up
  to 100 % with "other").

## Clean team: local checks (Linux, no GPU)

```
python3 clean/runner/build.py --out build --clang clang --no-device-libs --report   # gfx1201 compile + VGPR report
python3 -m unittest discover -s clean/tests -t .                                     # reference + emulated kernels
python3 clean/runner/isa_mix.py --co build/swin.co                                  # instruction mix per window (model)
python3 clean/runner/isa_mix.py --co build/swin.co --ops swin_block_c64_nc          # every vector opcode per window, by region of the block
```

## Second design of the Swin block kernels (`kernels/swin_v2.hip`)

Built with `--variants v2` (or `--variants all`, which builds both designs side by side) as `swin__v2.co`, plus its own
variants (`v2exact`, `v2gsig`, `v2g7`, `v2gbyte`, `v2nopipe`, geometry variants) and, with
`--ablations`, `v2ab_*` and the cycle-counter build `v2prof`. The runner
and the sweep use it like any variant: `Runner(build, rocm, variant="v2")`; the code object also contains the merge
and expand kernels, so a whole stage runs from it. `selftest.py` adds a `v2_conversion` section when `swin__v2.co`
exists. Default GELU of v2: max(x, 0) − c(|x|) with the correction c(a) = a·Φ(−a) from a 16 KB FP16 table (0.6 % of the
FP8 results one step from the exact GELU, none more; no transcendentals); `v2gsig` uses gelusig and `v2exact` the exact
GELU (for the real-data exactness check).

```
python clean\runner\build.py --rocm "C:\Program Files\AMD\ROCm\7.2" --out build --variants all --ablations
python clean\runner\build.py --rocm "C:\Program Files\AMD\ROCm\7.2" --out build_v2report --variants v2 --report
```

Per-phase cycle profile of v2 (where the waves compute and where they wait at barriers):

```
python clean\runner\build.py --rocm "C:\Program Files\AMD\ROCm\7.2" --out build --variants v2 --ablations
python clean\runner\profile_v2.py --build build --rocm "C:\Program Files\AMD\ROCm\7.2" --json prof.json
```

### Bottleneck megakernel: stage `bneck` (`swin_bneck6`, variants `bmegahg`, `bmega`, `bmegahg16`)

The six b-blocks in one persistent launch (FORMATS.md, "Bottleneck megakernel"): an ordered queue of 8-window items
with per-window-row completion counters instead of five launch boundaries, each item's weights staged through LDS.
The `bmega*` builds compute exactly as the matching `swin.hip` build (`WG_V2_BASEMATH`; `bmegahg16` as `vec16`), so
the megakernel and every other kernel of these builds give the same bytes. The runner has a stage `bneck` (b0..b5:
`pack_weights("bneck", {"b0": ..., ...})`, `run("bneck", w, x, b0's offset)`): six launches in a build without the
megakernel, one megakernel launch in a `bmega*` build; `sweep.py` includes it. `bench_bneck.py` times six launches of
each reference build against the megakernel at several grid sizes and counts output differences:

```
python clean\runner\build.py --rocm "C:\Program Files\AMD\ROCm\7.2" --out build --variants bmegahg,bmega,bmegahg16,vec16
python clean\runner\bench_bneck.py --build build --rocm "C:\Program Files\AMD\ROCm\7.2" --groups 64,96,128,160,192
python clean\runner\build.py --rocm "C:\Program Files\AMD\ROCm\7.2" --out build --variants bmegahg --ablations
python clean\runner\profile_v2.py --build build --rocm "C:\Program Files\AMD\ROCm\7.2" --variant bmegaprof --stages bneck
```

Stage `mid` (`swin_mid`, in the same builds): enc2 with its patch merge, b0..b5 and dec2 with its expand in one
persistent launch (eight phases in the same kind of queue), bit-identical to the ten launches it replaces.
`bench_bneck.py --stage mid` (or `--stage bneck,mid`) times it against those launches; `profile_v2.py --stages mid`
profiles it (the instrumented build spills, so read only the shares of its wait phases from it).
