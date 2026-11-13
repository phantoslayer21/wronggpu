# Spec 51 window block on D3D12 (RDNA 4) - kernels, test host, probes

Clean-side implementation of DESIGN-51.md (task C). Source of truth: spec 51 draft 6 only; conventions and the driver
channel from `clean/dx12/` (README there: arithmetic mode, driver exceptions, `wg_ags.hlsli`).

| Path | Content |
|---|---|
| `../kernels/rr_block.hlsl` | the window block (51.3) with the stage fronts (51.5: E1 adapter, fusion halves) and tails (51.4: merge, encoder / recurrent hidden rule, merged hidden); one source, one kernel per stage kind by defines (`RR_C`, `RR_H`, `RR_KIND`, `RR_CN`, `RR_HMERGE`, `RR_ADAPT`, `RR_KC`), Z mode `RR_Z`, step variant `RR_STEP` |
| `../kernels/rr_common.hlsli` | Q8 (51.1, NaN by sign, saturating), f16 rounding through the driver conversion, attention weight (51.3 step 3) and activation (step 8) on packed f16, the border rule (51.6), fences |
| `../kernels/rr_probe.hlsl` | probe kernels P1 (WMMA accumulation) and P2-P4 (Q8, attention weight, activation, R(1/Z) for every f16 input) |
| `rr_host.cpp` | standalone D3D12 executor (no HIP): one arena buffer as root UAVs u0 / u1, root constants, the driver-channel table; job file of kernels and dispatches; arena readback; per-dispatch GPU timestamps |
| `build.py` | compiles kernels with `clean/dx12/build_dx12.py`'s exact arithmetic mode (cached by source hash + defines), builds the host, runs a job under the GPU lock |
| `layout.py` | the runtime's buffer layouts (WMMA fragments of every weight, expanded f32 score-bias fragments, f32 biases) built from the logical tensors of 51.2 |
| `stage.py` | one stage call: arena, arguments, outputs; comparison with the vectors (`clean/rr_f/tests/vectors`) under the manifest's tolerances, and with the variant-E step model |
| `emu.py` | the reference run with variant E's step (two f32 WMMA roundings before the f16 rounding), to separate step precision from kernel errors |
| `probe_p1.py`, `probe_p4.py` | probes P1, P2-P4 (prepare / analyze, runnable alone) |
| `run_all.py` | probes + every vector case in the chosen Z modes and step variants in one GPU call |
| `bench.py` | GPU time per stage kernel at a render size (default 1440p Quality), synthetic data |

Run (repository root, Python with NumPy: any Python 3.12 with NumPy (BUILDING.md)):

```
python -m clean.rr_f.tests.gen_vectors            # once: the vector binaries (not committed)
python -m clean.rr_f.gpu.run_all --z supplied,exact,f32 --step E,X --json results.json
python -m clean.rr_f.gpu.bench --z f32 --step E
```

Every GPU call goes through the optional GPU lock (WG_GPU_LOCK) (one host process per call).

## Production kernels of the body stages (2026-10-21)

`stage.py` routes every stage to a kernel with the arguments, layouts and output bits of `rr_block.hlsl` (same
operations in the same order for every input; `WG_RR_GENERIC=1` selects `rr_block.hlsl` everywhere,
`WG_RR_ROUTE="E3=gen,D4-R=w4p15"` single stages):

| Stages | Kernel | Structure |
|---|---|---|
| IN0, D0-F | `rr_block_c32.hlsl` | C = 32 specialisation: Q8 fast path, packed attention bit step, weights from memory |
| E1, D1-F, D1-R | `rr_block_fast.hlsl` | the same diet for any C (E1 reads IN0's y2 directly: no regroup dispatch) |
| E2, E3, D2-F, D2-R, D3-F, D3-R (`wide2`; step X: `rr_block.hlsl`, `SPECIAL_X`), E4, E5 (`wide4`), D4-F, D4-R (`w2p4`) | `rr_block_wide.hlsl` | one window on G = 2 / 4 groups of 4 waves: heads and MLP chunks in rounds of G (one group each), the Dp / Dd chains advanced in order by the owner of each channel tile; `RR_PRE` = operand loads issued ahead of the matrix calls |

Shared helpers (`rr_fast.hlsli`): Q8 by the driver conversion with a wave-uniform slow path in two levels (clamp +
conversion, rr_q8 only on a wave with a NaN), the border rule with one reflection when every extent is >= 10, the
RNE f32 -> f16 conversion through driver-channel op 0x1C (the bits of `wg_f2h` for all 2^32 inputs).
`body_ab.py` times the stage kernels against a copy of the base kernels and compares every output byte
(`--variants` routes, `--data rand|synth|same|zero`, `--model`, `--step E|X`, `--render`, `--shift`).

## Kernel structure (DESIGN-51.md sections 4-7)

One workgroup = one 8 x 8 window, 4 waves; wave `wv` owns tokens 16 wv .. 16 wv + 15. Per head: q and v^T by
WMMA (weights A / tokens B, and swapped for v^T), exchanged through LDS (two 4 KB buffers, one barrier per head);
scores as key x query tiles with the expanded bias as the start value; attention weight in packed f16 plus the exact
bit construction of w; Z (f32 / supplied / exact); r = R(1/Z) by `v_rcp_f32` + RNE; P; O; one Dp step per head.
Then the residuals, the MLP in 32-wide chunks (rolled loop), y2 and the tails. Weight units (QV_h, Wo_h, MLP chunk)
are staged through two LDS buffers by all 128 lanes, fetched one phase ahead (`RR_LDSW=1`, default; 2 barriers per
head, 1 per MLP chunk); `RR_LDSW=0` reads the fragments straight from memory in every wave. Every f16 result that is widened or fed into another f16 operation is
fenced (`rr_fence_u`: XOR with a zero whose bits the compiler cannot know; a fence with a known-bit-0 zero let the
driver fold the high half of a packed fma into an f32 `v_fma_mix`, skipping the f16 rounding).

Variants: `RR_STEP=0` (E): a step is two WMMAs on an f32 accumulator, then one f16 rounding; `RR_STEP=1` (X): each
16-product sum is taken exactly as p + e with a second WMMA (e = WMMA(a, b, -p)), the step summed with error-free
transforms and rounded once (exact under probe P1's model). `RR_Z`: 0 f32 sum, 1 supplied (51.8 item 1),
2 exact sum rounded once (the reference's rule).

## Status (2026-10-18, RX 9070 XT, every GPU call under the GPU lock)

Correctness, 52 vector cases x Z modes (supplied / exact / f32) x step variants (E / X) x weight paths (LDS / memory)
= 624 stage runs (`run_all.py`, `summary.py`):

* Probes P2-P4 (51.8 items 2, 3): Q8 of every f16 pattern, attention weight and activation of every f16 input for the
  three constant sets, R(1/Z) for every f16 Z >= 2^-8 and +inf: identical.
* Variant X: every stage kind (IN0, E1-E5, D4-F .. D0-F, D4-R .. D1-R), every case, every Z mode, both weight paths:
  identical to the reference (e4m3 zero codes equal, f16 zero signs equal). The f32 Z mode happened to give the
  exact row sums in every row of the vectors.
* Variant E (default): identical to its step model (`emu.py`) in every run; against the reference: 11 stage kinds
  identical in all their cases (E2, E3, E4, E5, D4-F, D2-F, D1-F, D0-F, D3-R, D2-R, D1-R), 4 with step-precision
  differences: IN0 (hot case: 2 y2 codes of 113,664 values), E1 (r64x96: y2 21, merged 34, hnew 4, hmerged 2 of
  166,656), D3-F (r288x256: 8 of 147,456), D4-R (r288x256: 1 y2 code); 51 of these 72 differences are one code
  or one f16 step.
* Probe P1 / P1b: the fp8 WMMA is not "exact sum rounded once to f32" (QUESTIONS-51.md, task C section): variant X is
  exact on the vectors but not certified for every input.

GPU time at 1440p Quality (`bench.py`, render 1706 x 960 -> level-0 tokens 1728 x 960, synthetic data, Z f32,
best of 10; D0-R not included):

| stage | windows | E, LDS staging | E, memory | X, LDS staging |
|---|---|---|---|---|
| IN0 | 26,257 | 1.023 | 1.014 | 4.608 |
| E1 | 6,649 | 0.580 | 0.577 | 3.643 |
| E2 | 1,705 | 0.311 | 0.316 | 1.979 |
| E3 | 448 | 0.143 | 0.173 | 0.958 |
| E4 | 112 | 0.078 | 0.125 | 0.575 |
| E5 | 40 | 0.059 | 0.109 | 0.397 |
| D4-F / D4-R | 112 / 112 | 0.070 / 0.063 | 0.126 / 0.111 | 0.570 / 0.539 |
| D3-F / D3-R | 448 / 405 | 0.134 / 0.123 | 0.115 / 0.167 | 0.954 / 0.859 |
| D2-F / D2-R | 1,705 / 1,620 | 0.304 / 0.262 | 0.300 / 0.255 | 1.955 / 1.760 |
| D1-F / D1-R | 6,649 / 6,480 | 0.552 / 0.484 | 0.543 / 0.478 | 3.099 / 2.772 |
| D0-F | 26,257 | 1.120 | 1.127 | 5.032 |
| total (ms) | | 5.31 | 5.54 | 29.70 |

DESIGN-51.md estimated about 4.5 ms for these 15 calls (5.3 ms with D0-R). Levels 3-5 are latency-bound (40-448
windows for 64 CUs); levels 0-2 are VALU-bound (the f16 rounding of every step, the attention weight and every Q8 with
its NaN-sign handling).

## Connected-body continuation (2026-10-18)

`network.py` wires IN0 through D0-R in a single arena and host submission, including an exact E1 child-regroup
dispatch. Every intermediate input points to a prior GPU output; unused allocation placeholders contain NaNs,
not CPU oracle outputs. The CPU reference is used only for expected outputs and array shapes. There are sixteen
stage dispatches plus regroup, with UAV ordering supplied by the existing host.

Set `WG_RR_BUILD` to a private build directory, then run:

```
python -m clean.rr_f.gpu.network --json body.json
python -m clean.rr_f.gpu.network --mutation head --json body-mutant-head.json
python -m clean.rr_f.gpu.network --mutation encoder --json body-mutant-encoder.json
```

Normal qualification returns 0 only when all common-stage outputs are exact, head tolerances and exact special
rules pass, stored codes match the GPU's own f16 values, and head guards remain intact. Mutants must return 1 and
produce a JSON report with `passed=false`; a crash or missing report is not a successful negative control.
The fixture is one prepared 32x32 token frame with a 29x19 render crop. It uses step X for the common stages and
the existing head kernel's arithmetic. Passing this fixture does not certify matrix arithmetic for every input,
full-frame temporal behavior, performance, or game integration. Existing E/X and head-edge limitations remain.

## Exposure qualification (spec 58)

With `WG_RR_BUILD` pointing at private scratch, run `python -m clean.rr_f.gpu.exposure --json exposure.json`.
The two kernels reduce API-decoded f32 colour to a binary64 mean and produce E/Ep/invE/X3 plus the f16 storage
value. All symbols are explicit original synthetic fixtures. The host checks binary64 support before creating
these pipelines and records the physical adapter. No image-view conversion or temporal resource integration is claimed.

Native multiply and binary64-to-f32 conversion flushed a narrow underflow interval before rounding on the tested
driver. `ex_round` explicitly preserves the interval that rounds to the smallest normal, including signed zero.
Seventeen synthetic cases pass, including independent product/mean boundary cases and negative-zero X3. The case
set includes cropped inputs, reset, nonfinite input/previous exposure and sparse highlights. A skipped partial
producer (`--mutation`) fails the exact mean-bound/output comparison. This is correctness evidence, not a performance claim.

## Output and live head connection (spec 54)

`python -m clean.rr_f.gpu.output --json output.json` qualifies decoded-buffer and packed-head input modes.
Packed Cnet/PREC use half4; GATES/CANDW use byte4 decoded to the specified half readback in the shader.
`prepare(..., input_bindings=...)` uses existing arena producer offsets without uploading any reference payload.
The output and pooling dispatches produce CH half4, internal Q half3 and PH1 unpacked RGB10A2 codes.
History/motion inputs currently use decoded f32 buffers carrying half values. Binary64 support is required.

The normal suite covers 32 image/layout cases, 256 ordering patterns and 4,186 arithmetic/store rows, with
exact encoded-byte agreement across decoded/packed paths and reset-poison pairs. Order/reset/code mutations
must fail. Image tolerances do not override exact store-code domain, comparison order or reset checks.

`python -m clean.rr_f.gpu.pipeline --json pipeline.json` connects the sixteen-stage body and E1 regroup to
the output/pool pair: 19 GPU dispatches, one arena/submission, no CPU feed between the head and output.
Its synthetic prepared fixture has 32x32 tokens, a 29x19 render/output region and 45x30 history.
Exposure, prepared tokens, motion and history are supplied fixtures, not a complete temporal frontend.
`--mutation` replaces Cnet's producer connection with zero data. Exit 1 is a useful negative control only
when the body still passes and a numerical output comparison fails; otherwise the harness returns 2.
The report records bindings from actual dispatch arguments, shaders/source hashes and local/full-graph results.

Neither harness implements page 54b's game-output resolve, page 52 input construction, player-model loading,
image-view interop or game integration. Existing matrix/head arithmetic limitations remain. See the resumed
report under `results/2026-10-18-rr-gpu-output.md`; no performance claim is made.

## Optional exact matrix baseline

`gpu/exact_dot.py` qualifies a scalar binary64 dot32 and direct binary64-to-half rounding. Finite products and
the half accumulator share a 2^-24 lattice; the maximum scaled magnitude is below 2^47, so the sum is exact in
binary64. The direct conversion avoids an intervening f32 rounding. It requires binary64 shader support.

`gpu/strict_stage.py --out <private-path>` exercises the fragment adapter, common IN0 and D0-R `stage_edge`.
Common `RR_STEP=2` and head `RR_STRICT_STEP=1` are opt-in diagnostics; defaults stay E/X and the existing head.
Run `--only default-controls` for representative default regressions and `--only nan-controls` for the known
sign-policy failures. A failed negative-NaN control is retained evidence, not permission to widen tolerances.

Current finite/positive-NaN smoke and D0-R edge checks pass, but canonical positive matrix NaNs differ from the
existing CPU oracle on negative-NaN inputs/biases after Q8. There is no full-network, universal NaN-code, speed,
player-model or game qualification for this mode. Source question and exact evidence are recorded in the round-3 report.

## Input stage (spec 52), 2026-10-20

Kernels (`../kernels/rr_input.hlsl`, `rr_in_math.hlsli`, `rr_input_mlp.hlsl`, `rr_step.hlsli`), per call:

1. `rr_input_decode` (3 small dispatches): PH1 / PH2 RGB10A2 codes and the HID snorm8 codes to f32 copies, once per
   frame (the per-texel conversion of 52.1.4 step 3; the taps then read f32).
2. `rr_input_features`, fast pass (`RR_IN_EXACT=0`, one lane per tensor token): the 64 inputs of 52.3 in the
   reference's operation order, the dither quantisation (52.4, integer form), the state-stream seed (52.6), m2 with
   the B1 / B2 reconstruction (52.7.2), the PM / Pe / Pn writes (52.7). Branch-free exact f32 helpers: rcp / sqrt
   corrected with exact fma remainders, unorm / snorm decodes exact, log2 / exp2 in float-float (`ffmath.py` is the
   model) with tables staged in LDS, the constant block staged in LDS. A token whose result cannot be proven in this
   pass (a mul / fma zero that is not an exact zero, i.e. the driver's flush-before-rounding interval; a log2 / exp2
   near a rounding midpoint) is marked in a bitmap.
3. `rr_input_features`, replay (`RR_IN_EXACT=1`, `RR_IN_REPLAY=1`): only the marked tokens, with binary64 repairs and
   the binary64 log2 / exp2 fallback; it clears the bitmap words it reads.
4. `rr_input_mlp`: the front-end MLP on WMMA (step E / X / exact as the block kernel).

All per-frame values come from one 1 KB constant block (`input_stage.scalar_words`; C++ twin `../host/rr_input_host.h`,
including the float-float tables); E / Ep / X3 can be read from the exposure pass's result buffer.

```
python -m clean.rr_f.gpu.input_stage --json input.json --probe --xcheck [--mark-all]
python -m clean.rr_f.gpu.frame --json frame.json [--frames 2] [--size 77x45] [--mutation input|exposure|history]
python -m clean.rr_f.gpu.input_bench [--fast ...] [--skip ...]       # GPU time, timing-only variants
```

Evidence (RX 9070 XT): 23 frame cases bit-exact (codes, f32 features, x0, hold, PM / Pe / Pn), also with every token
forced through the replay (`--mark-all`); probes: quantisation of all 65,536 f16 x 128 dither values, the f32 helpers;
`--xcheck` (every f32 input): log2 (2.13 G inputs), exp2 (2.23 G), rcp (2.13 G) and sqrt (2.13 G) fast paths equal
to the binary64 paths wherever the fast path does not flag (flags: 13,662 log2, 2,922 exp2). `frame.py` connects
exposure -> input stage -> body -> head in one arena (IN0 reads x0 / hold from the input stage; with `--frames 2` the
second frame reads the first frame's head HID and PM); references use the values the GPU produced upstream.

GPU time at 1440p Quality (render 1706 x 960, Hd reconstruction, all guides): see the track report; the features pass
dominates (about 4.4 ms, from 12-17 ms in the first version), the front-end MLP 0.3-0.4 ms (E) / 1.4 ms (X), decode
0.15 ms, replay 0.15 ms. Breakdown by elimination (`--skip`): history taps about 2 ms (15 exp2 + 15 log2 in
float-float), quantisation 0.7 ms, hidden / motion-difference 0.9 ms, reconstruction 0.5 ms.

### Speed pass (branch clean/rr-f-speed-input, 2026-10-20; results/2026-10-20-speed-input.md)

The outputs are the same bits as before for every input (codes, x0, hold, PM / Pe / Pn, PH2); what changed:

* Certified feature codes (fast pass): the history features (hdec + ccf: exp2 and log2) and the colour, guide and
  motion-difference features (log2) are evaluated with plain f32 and the hardware exp2 / log2 (within one f32 step
  of the correctly rounded ones for every input, `--hwerr`) plus an error bound; the code is taken when the bound
  interval maps to one code (both ends quantised, or inside one binary16 cell). Otherwise the value is recomputed
  with the reference operations or its token is replayed. The debug f32 value of a certified feature is the
  approximation (debug words 72 / 73 list them; `compare` checks their codes). Negative control: `--mutation hcert`.
* Replay list: marked tokens are appended to a list (count + indices); the replay strides over it on a fixed grid.
* quant in a shorter integer form (probe "quant": every binary16 value x 128 dithers); M / FM flag tests without the
  implied finiteness tests; bmix without zero-result tests (proven: weights 0 or >= 2^-25, decoded texels); FM's
  exact-zero test behind the zero result; mdiff mixes only its two PM channels; 128-lane workgroups.
* Front-end MLP step X = RR_STEP 3: exact sums (e1 = e2 = 0, s and s2 exact) give R16(s2) without rr_exact5, the same
  result; `--mlp-diff` compares it with plain variant X (RR_STEP 1) on 25 M x0 codes, `--mutation mlp` must differ.
* PH2 pool (54b.5) through exact tables (`../kernels/rr_pool.hlsl`, `pool.py`; all 2^32 means checked per model) and
  no widening pass (the output kernel reads the binary16 motion / history images, packed_head bit 2).

```
python -m clean.rr_f.gpu.input_stage --json in.json [--probe] [--mark-all] [--mutation hcert|mlp]
python -m clean.rr_f.gpu.input_stage --json hw.json --hwerr      # hardware exp2 / log2 against the exact ones
python -m clean.rr_f.gpu.input_stage --json f2h.json --f2hcheck  # driver f32 -> f16 conversion, all f32
python -m clean.rr_f.gpu.input_stage --json mlp.json --mlp-diff [--mutation mlp]
python -m clean.rr_f.gpu.pool --json pool.json --synthetic --model <model.wgcm> ...
```
