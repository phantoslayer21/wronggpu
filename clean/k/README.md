# Model K (specs 40 draft 4, 41 draft 3, 44, 48, 49 draft 3)

A second network next to model M. All model K code lives here; the shared files only got small hooks (the runner's
model selection, the build, the runtime's create check). Model M's behaviour is unchanged.

| Path | Content |
|---|---|
| `reference/act.py`, `f16.py` | the exact activation of spec 40 (NumPy) and an exact-rational FP16 oracle that checks it on all 65536 inputs (K2) |
| `reference/layout.py` | stage table, spec 44 offsets and base size, level pyramid, buffer plan, 8×8 window partition with origin, margin and border rule (K3) |
| `reference/linear.py` | linear, 2×2 merge, 2×2 expand (K1) |
| `reference/network.py` | spec 41 block (fixed and softmax mixing), embedding, head, decoder block input with the draft 3 border rule, the eleven stages and the whole network (K4 oracle); random parameters |
| `reference/params.py` | parameter names and shapes: canonical, and the converter's of spec 48 (accepted everywhere) |
| `kernels/k_common.h` | FP16 helpers and the activation as fusable device functions (`wgk_act`, packed `wgk_act2`, `wgk_act8`) |
| `kernels/k_linear.hip` | FP16 WMMA linear / merge / expand kernels for every shape (K1) |
| `kernels/k_block.hip` | the spec 41 block (K5), and one fused kernel per stage with the embedding, expand + skip, merge and head inside; opt-in FP16-accumulating forms `k_stageh_*`, `k_stage2h_*` (see "Speed approximations") |
| `kernels/k_block_wide.hip` | wide stage kernels `k_stagew<NG>_*` for levels 3..5 (e3, e4, m, d4, d3): one window per NG groups of 4 waves, bit for bit `k_stage_*`; FP16-accumulating forms `k_stagew<NG>h_*`, bit for bit `k_stageh_*` |
| `kernels/k_window.hip` | window gather / scatter and the device helpers for window addressing (K3) |
| `kernels/k_act.hip` | activation tables for the exhaustive GPU check (K2) |
| `runner/k_pack.py`, `k_launch.py`, `k_stages.py` | weight packing, launch plans, the Runner's model K stages |
| `runner/k_net.py`, `k_bench.py` | whole network on the GPU, timing harness and GPU-vs-NumPy check (K4) |
| `runner/k_build.py` | code objects (called by `clean/runner/build.py`) and a model K-only emulator library |
| `runner/k_model.py` | reads the converter's `--raw` output (spec 48) with the preset-K map's names and shapes, and the window-offset table (`net/window_offsets`) |
| `reference/fp.py` | exact FP32 / FP16 helpers (fma by round-to-odd, round half away, truncation to FP16) |
| `reference/instage.py`, `outstage.py` | spec 42 input stage, spec 43 output stage and filter-shape table (NumPy) |
| `kernels/k_fp.h`, `k_io.hip` | the input / output stage kernels (`k_in_color`, `k_in_base`, `k_out_pixel`, `k_out_hidden`) |
| `runner/k_io.py` | their launch arguments, host scalars, the exact 2^E table |
| `runner/k_frame.py` | one whole frame: input stage → network → output stage, histories and frame index (`KFrame`), and the chained NumPy reference (`reference_frame`) |
| `runtime/model_k.h`, `.cpp` | runtime layer table (stages, sizes, offsets, buffer plan, blob sizes) and model selection |
| `runtime/k_pack.h`, `.cpp` | C++ port of `runner/k_pack.py pack_stage` (d_row gfx12) for the runtime: `k_pack_stage(stage, tensors, &err)` |
| `tests/` | reference checks, emulator kernels vs reference, C++ table vs Python |
| `FORMATS.md` | maps, weight blobs, kernels and launch geometry |

## Test interface for model K (text for spec 90; ANSWERS.md K-Q5)

### Python API: `clean/runner/stage_runner.py`, model K

```
r = Runner(build_dir, rocm_dir, model="K")
w = r.pack_weights(stage, tensors)                       # bytes
out = r.run(stage, w, x, offset, skip=None)              # dict
ms = r.time(stage, w, x, offset, skip=None, repeats=50)  # median ms of the stage's kernels
```

- `stage` is one of `e0, e1, e2, e3, e4, m, d4, d3, d2, d1, d0` (spec 40). With model M's runner (`model="M"`, the
  default) the same stages are `k.e0` … `k.d0`.
- `offset` is the stage's window offset (ox, oy) of spec 44 (the spec team applies the frame table itself).
- Feature maps are NumPy `float32` arrays `[H, W, C]` whose values are FP16 values (other values are rounded to FP16).
  Sizes: the level sizes of spec 44 (any size works; decoder stages need ⌈W/2⌉ × ⌈H/2⌉ for the coarser map).
- Inputs per stage:
  - `e0`: `x` is the 16-channel network input at level 0 `[Hb, Wb, 16]` (the embedding runs inside the stage).
  - `e1 … e4`, `m`: `x` is the previous stage's merged map, at this stage's block level `[H, W, C]`.
  - `d4 … d0`: `x` is the previous stage's block output at the coarser level `[⌈H/2⌉, ⌈W/2⌉, C_prev]`, `skip` is the
    skip map S of this level `[H, W, C]` (the block output of the encoder stage at this level). The expand, the skip
    addition and the border rule of spec 41 draft 3 run inside the stage.
- Outputs (`dict` of `float32` arrays, FP16-exact):
  - `e0 … e4`: `{"block": [H, W, C], "merged": [⌈H/2⌉, ⌈W/2⌉, C_next]}` (block = the skip map S of this level)
  - `m`, `d4 … d1`: `{"block": [H, W, C]}`
  - `d0`: `{"block": [Hb, Wb, 32], "head": [Hb, Wb, 40]}`

### Parameter tensors passed to `pack_weights`

The converter's tensors of spec 48 section 7.2, as they are (`float32`, FP16 values, output index first): `g1`, `g2`,
`Wv`, `Wo`, `bo`, `W1`, `b1`, `W2`, `b2`; softmax stages `Wq`, `Wk` `[H][32][C]` and `B` `[H][8][8][8][8]`;
fixed-mixing stages (e0, e1, d0) `M` `[H][8][8][8][8]`; e0 `We`, `be`; e0 … e4 `Wm` `[C′][4][C]`, `bm`; d4 … d0 `Wx`
`[4][C][C_prev]`, `bx` `[4][C]`; d0 `Wh`, `bh`. The flattened forms (`B`, `M` as `[H][64][64]`, `Wm` as `[C′][4C]`,
`Wx` as `[4C][C_prev]`, `bx` as `[4C]`) are accepted too. `clean/k/runner/k_model.py load_raw(dir)` reads a
converter `--raw` folder into exactly these dicts (`{stage: tensors}`).

### Other model K stages

`block:<C>:<heads>:<fixed|softmax>` (one block on its input map; encoder-style mirror border), `lin:<plain|merge|expand>:<K>x<N>`
(one linear layer), `win` (window partition round trip), `act` (the activation on all 65536 FP16 inputs on the GPU:
`out["mismatch_scalar"]`, `out["mismatch_packed"]`, both must be 0). Suffix `@g1`: linear kernels with 16 tokens per
wave.

### Input and output stages (specs 42, 43)

`r.run("input", b"", x)` → `{"x0": [Bh, Bw, 16]}` and `r.run("output", b"", x)` → `{"out": [H, W, 4] RGBA, "L", "Hc",
"Hh"}` with the frame dicts of `clean/k/reference/instage.py` / `outstage.py`: `color` (RGB or RGBA, render
resolution), `motion`, `depth` ([MVlo]), `render_size`, `output_size`, `hdr`, `mv_lowres`, `depth_inverted`, `jitter`,
`mv_scale`, `mv_offset`, `pre_exposure`, `reset`, `E_cur`, `E_prev`, `E_aux`, `e1`, `e2`, and the histories `Hc`, `Hh`
(input) or `N`, `L`, `Hc` (output; `table` optional, else generated for the frame's level). Both equal the NumPy
references bit for bit in the emulator, in all eight flag variants, and on the RX 9070 XT (all eight variants on the
test frames, and at 2560×1440 output with render 1707×960). Every FP32 → FP16 rounding of a computed value goes
through `kh_round` (k_fp.h): without it the compiler fused `(h16)(Coo * kk)` (43 section 6.6) into `v_fma_mixlo_f16`,
one rounding instead of two, and 46 of 3.7 M colour-history pixels differed by 1 ulp at 1440p.

### Whole frame: `clean/k/runner/k_frame.py`

`KFrame(dev, func, output_size, max_render_size, params, flags)` keeps the histories (double-buffered) and the frame
index of spec 44 (0 on a reset: the game's flag, the first frame, or a render-size change). `frame(x)` returns the
image, and `time_frame(x)` the GPU ms of the input stage, network, output stage and total. `params` are the
converter's tensors (`k_model.load_raw`) including `net/window_offsets`.

### Whole network: `clean/k/runner/k_bench.py`

Timing (per stage and per kernel) at the spec 44 sizes and the GPU-vs-reference check of the chained network, with
random parameters or `--model <converter output>` (see below).

## Spec team: build and run (Windows, HIP SDK 7.2)

```
python clean\runner\build.py --rocm "C:\Program Files\AMD\ROCm\7.2" --out build
python clean\k\runner\k_bench.py --build build --rocm "C:\Program Files\AMD\ROCm\7.2" --check 1920x1080 --variants fused fused2 g2 --kernels --frames --json k_bench.json
```

Spec 49 draft 4's counted whole-network check, the realistic test sequences (pan and texture, 4 frames each, 512×512
output, generated from the recipe's formulas): `k_bench.py --build build --rocm <ROCm> --sizes --sequences --model <dir>`.
Each frame goes through the GPU frame (input stage → network → output stage, histories carried over, reset on frame 0
only) and its network is compared with the reference fed the same network input, every block output, merged map and
the head against draft 4's limits (head 8·10⁻³ / 7·10⁻²; other outputs mean 1.2·10⁻², max 0.12 at levels 0–2, 0.25 at
levels 3–5). On the RX 9070 XT with random parameters: all 8 frames pass (d0 head mean ≤ 1.5·10⁻³, max ≤ 1.5·10⁻²).

`k_bench.py` first runs the whole network once at 1080p output on random parameters and compares every stage's
block output and the head output with the chained NumPy reference (non-finite count, mean and max |error| relative
to the reference RMS; pass/fail by spec 49 draft 2's whole-network rule), then times all eleven stages at 1080p, 1440p and 4K output (median of 30
whole-network runs, events between stages; `--kernels` adds every launch on its own). `--s` sets the pending
base-size factor of spec 44, `--frame f` the frame index (offsets of m, e4, d4), `--model <dir>` uses the converter's
`--raw` output (spec 48) instead of random parameters. `--frames` adds whole-frame times (input stage, network,
output stage) at the same sizes with render size = output / `--ratio` (default 1.5); `--all-flags` for all eight
flag combinations. Windows at the norm's FP16 overflow threshold are handled as
spec 49 draft 3 says (excluded and counted; a run that has any is reported as a "threshold run").

Per stage (41-T1, 41-T2), in Python:

```
from clean.runner.stage_runner import Runner
r = Runner("build", r"C:\Program Files\AMD\ROCm\7.2", model="K")
w = r.pack_weights("e2", tensors)                  # spec 48 tensors, e.g. k_model.load_raw(dir)["e2"]
out = r.run("e2", w, x, (4, 4))                    # {"block": [H, W, 64], "merged": [H/2, W/2, 96]}
out = r.run("d3", w, y4, (4, 4), skip=S3)          # {"block": [H3, W3, 96]}
out = r.run("d0", w, y1, (4, 4), skip=S0)          # {"block": [Hb, Wb, 32], "head": [Hb, Wb, 40]}
ms = r.time("e2", w, x, (4, 4))
o = r.run("act", b"", None)                        # K2 on the GPU: o["mismatch_scalar"], o["mismatch_packed"] == 0
```

Inputs are float32 arrays holding FP16 values (other values are rounded to FP16); outputs are FP16-exact float32.
A stage runs as one fused kernel when its offset is even (always with spec 44); `"e2@split"` runs the separate kernels
(also used automatically for odd offsets); `"e0@w2"`, `"d0@w2"` the fused kernel with two windows per workgroup;
`"e3@wide2"` etc. the wide kernel. All of these accumulate in FP32 (spec 41 as written) and match the reference;
`--variants fused fused2 g2` in `k_bench.py` times them side by side. A further suffix `@f16` (`"e2@f16"`,
`"e0@w2@f16"`, `"e3@wide4@f16"`; even offsets) runs the opt-in FP16-accumulating form instead (see "Speed
approximations (spec 49)"; network variants `fused16` and `<variant>@f16`).
`Runner(...)` without `model="K"` takes the same stages as `"k.e2"` etc. Other stages: `block:<C>:<heads>:<mix>`
(the block alone), `lin:<plain|merge|expand>:<K>x<N>` (one linear layer), `win` (window round trip); suffix `@g1`
selects the linear kernels with 16 tokens per wave.

The runtime (`wronggpu_runtime.dll`) runs model K with the runtime option `model=K` (in the front end's options
string or ini line, or `WG_RUNTIME_OPTIONS=model=K`; the interop passes keys it does not know on to the runtime):
`runtime/runtime_k.cpp` is the C++ port of `KFrame` (default network variant "fused", default fast input stage),
on the interop's buffers like model M (FORMATS.md "Runtime frame"). It loads `<model_dir>\K\model.wgcm` (the
converter's preset-K output; `model_k_dir=<folder>` overrides) and checks it against the preset-K map shipped next to
or inside the DLL; without them create fails with a plain message and the front end falls back to AMD FSR. Without
the option, or with `model=M`, nothing changes. `network=fused16` selects the FP16-accumulating network (variant
"fused16", the spec 49 speed approximation; opt-in, the default stays "fused"); other values fail create.
`tests/test_k_runtime_gpu.py` (GPU only) runs it through the external-I/O entry points and compares every frame with
`KFrame` (same variant) bit for bit. 1440p Quality in the runtime (timing marks, median of 100 frames after 300
warm-up, two runs each): network 0.587 ms ("fused") / 0.508 ms ("fused16"), whole frame 1.79 / 1.71 ms.

Weight blobs in the runtime: `wgrt::k_pack_stage(stage, tensors, &err)` (`runtime/k_pack.h`, compiled into the DLL)
turns a stage's tensors from `load_k_model` (spec 48 shapes; canonical shapes and `gamma1` accepted too) into the
blob the stage kernels read, `k_stage_blob_bytes` bytes, byte for byte `k_pack.pack_stage(stage, t, d_row="gfx12")`.
Values must be exactly FP16 (as in `k_pack.py`); NaN is refused. `tests/test_k_runtime_pack.py` checks all eleven
stages on random parameters (converter and canonical shapes), refusals and a converted synthetic preset-K DLL.

## Installing model K (what the installer needs)

Model K ships in the same `wronggpu_runtime.dll` as model M; nothing changes for M.

1. Library: `clean/runner/build.py --extra-archs gfx1200` builds the DLL with K's code objects (`k_block.co`,
   `k_block_wide.co`, `k_io.co`, for gfx1201 and gfx1200) and the preset-K map (`preset-K.v1.map.json`) built in;
   the build also copies the map next to the DLL (optional: the built-in copy is used when the file is absent).
2. Model: the converter with the preset-K map, the same way as for M (command line: clean/converter/README.md,
   `--map preset-K.v1.map.json --out <model_dir>\K`) writes `<model_dir>\K\model.wgcm` (default `<model_dir>` =
   `%LOCALAPPDATA%\WrongGPU\model`; the source file versions that hold preset K and the converter's message
   otherwise: spec 48 sections 3 and 8). The map file is `data/preset-K.v1.map.json` (also next to the built
   DLL). Runtime option `model_k_dir=<folder>` points elsewhere.
3. Selecting K: the front end's options / ini line `runtime_options = model=K` (or `WG_RUNTIME_OPTIONS=model=K`).
   Without it, or with `model=M`, model M runs. If the K model or map is missing, create fails with a plain message
   and the front end falls back to AMD FSR, as for M.
4. Optional runtime options (all keep spec 41-43 output bits unless noted): `network=fused16` (FP16 accumulation,
   the spec 49 speed approximation, opt-in; not bit-exact), `k_input=f32` (the FP32 input pass instead of the
   compact one), `input_copy=1` (copies instead of the input pass where the formats allow), `timing_log=1`,
   `k_diag=input2|copyin` (timing diagnostics only), `k_head=full` (the 40-channel head map between d0 and the
   output stage instead of the compact pixel head; needed to read the buffer `k.N`).

In game (Cyberpunk 2077 benchmark, 1440p Quality, RX 9070 XT; the project's measurements): model K
GPU time about 1.8 ms per frame (`fused`; 1.72 ms with `fused16`), the game at ~120 fps against ~117 fps with model M
(release 0.4.0) at the same settings, before the compact input pass; with it (`fused16`): GPU 1.588 ms, TOTAL(t3−t0)
1.796 ms, 129.3 fps (126.4 with `k_input=f32`).

## Tests (Linux, host lane emulator)

```
python -m unittest discover -s clean/k/tests -t .
```

The emulator runs the same kernel sources on the CPU (`clean/kernels/emu`); every stage of spec 41 matches the
reference on small odd maps with random offsets (mean |error| ≤ 4·10⁻⁵, max ≤ 3·10⁻³ of the RMS; spec 49 allows
2.5·10⁻³ and 3·10⁻²). The emulator sums each WMMA in double precision, so these runs check layouts, addressing,
borders and the order of the FP16 steps. The whole chained network (`KNet`, base 37×29, frame 5) passes spec 49
draft 2's whole-network rule there too (head output: mean 1.6·10⁻³, max 1.4·10⁻² of the RMS). The GPU's FP32
accumulation is checked by `k_bench.py --check`.

### The same tests on the GPU (any OS)

```
set WG_K_GPU_BUILD=<kernel build dir>
set ROCM_PATH=C:\Program Files\AMD\ROCm\7.2
python -m unittest discover -s clean/k/tests -t .
```

With `WG_K_GPU_BUILD` set, the kernel, stage, network-chain, input/output-stage and whole-frame tests run on the GPU
instead of the emulator (same assertions). `TestRealSizeGPU` (GPU only, about two minutes) adds both I/O stages bit
for bit at 2560×1440 output; rare rounding cases only show at that size. On the RX 9070 XT (2026-10-08): all pass.

## Design notes

- Transposed WMMA form everywhere: weights are the A operand, tokens the B operand. Then a lane holds one token and 8
  contiguous channels in both the input and the output fragment. The FP16 result of one product is directly the
  next product's B operand, so the block chains norm → q → scores → softmax → z → Wo and W1 → act → W2 in registers.
  Lanes exchange data only for the two reductions (norm sum, softmax sum).
- K and V of the current head go through LDS: K token-major (the A operand of the scores), V channel-major. V is
  produced directly transposed by swapping the operands (the token fragments serve as the A operand).
- Biases are the starting value of the FP32 accumulators, so `f16(W x + b)` rounds once, as spec 41 writes it (all
  default kernels; the opt-in FP16-accumulating forms round once per 16-deep step instead, next section).
- Decoder stages (spec 41 draft 3 border rule): the expand writes the block input for the stage's whole window grid
  (8 nwx × 8 nwy tokens), with the coarse source mirrored at the coarse level and the skip mirrored at the fine level;
  the block then reads that padded map without mirroring. Encoder stages read their map through the mirror.
- Normalization (spec 41 draft 2): FP32 sum of the FP16-rounded squares, rounded once; s = +inf and s = 0 give r = 0.
- Fused stages (default): one launch per stage. The embedding (e0), expand + skip (decoders), merge (encoders) and
  head (d0) run inside the block kernel, so each map is read and written once. Feature-map traffic per frame (weights
  and mirrored re-reads not counted): 0.26 / 0.48 / 1.06 GB at 1080p / 1440p / 4K output, against 0.57 / 1.03 / 2.28 GB
  with the separate kernels; 130 MiB less memory at 4K.
- Weight reads: every wave reads each weight tile it needs from the cache, for its 16 tokens. The width-32 stages at
  level 0 (e0, d0; three quarters of all tokens) hold two windows per wave set (`k_stage2_*`, default since round 2;
  `fused_w1` / `fused16_w1`: one window), so each tile serves 32 tokens (188 instead of 121 VGPRs, so fewer waves in
  flight). Where the time goes in those stages (sustained, timing builds cut after each step, `fused_w1`): e0 0.130
  ms = load and embedding 0.013, normalization 0.003, token mixing 0.036, MLP 0.058, store and merge 0.015; d0 0.142 =
  expand and skip 0.025, mixing 0.041, MLP 0.059, store and head 0.016; d1 0.103 = mixing 0.033, MLP 0.048; e1 0.078
  = mixing 0.014, MLP 0.046. The MLP (32 WMMAs and ~400 VALU instructions per wave and window at C = 32: activation,
  FP32 bias and result conversions) is within ~20 % of that instruction count's bound; FP16 accumulation (`fused16`)
  removes most of the conversions.
- Levels 3..5 (`k_block_wide.hip`, default in variant `fused`): with 6..252 windows the 128-lane stage kernels ran
  about one wave per SIMD and waited on every weight load. The wide kernels give each window NG groups of 4 waves
  (NG = 2; m: 4). Group g computes heads g, g + NG, .. (own K / VT in LDS) and writes z to LDS; the output channel
  tiles of the residual stream are owned (g, g + NG, ..: Wo, x1, W2, x2), and the MLP's hidden tiles go in chunks
  (each group W1 + act for 2 tiles, h through LDS, then W2 over the chunk for its own output tiles). Every FP32 sum
  keeps k_block.hip's order, so the results are bit for bit the same (checked by `test_wide_stages`). Merge and expand
  tiles are spread over all waves. The weight tiles needed first after a barrier are loaded before it. Variants
  `fused1` (the 128-lane kernels everywhere), `auto` (e3, e4, d4, d3 as separate kernels), `wide2`, `wide4` stay for
  timing.

## Speed approximations (spec 49)

Spec 49 accepts approximations that keep the per-stage and whole-network thresholds and asks to report them. Model K
has one, and it is **opt-in**: the default network variant `fused` (and every stage name without `@f16`) runs the
FP32-accumulating kernels, spec 41 as written, until in-game numbers and image-quality checks exist.

The approximation: FP16-accumulating stage kernels, the FP32 kernels' names with an `h` after the prefix
(`k_stageh_<name>` with m included, `k_stage2h_e0` / `_d0`, `k_stagew<NG>h_<name>`; template parameter `F16` of
`k_block` / `k_block_wide`, mask `WGK_F16ACC` = `KF_DEFAULT` of `k_common.h`; details in FORMATS.md). The embedding,
expand, q / k / v, scores, z, Wo, W1, W2 and head products run on `v_wmma_f16_16x16x16_f16`, which rounds its FP16
accumulator to nearest even after every 16-deep step (measured on the RX 9070 XT; the same throughput as the
FP32-accumulating form), so a K-deep product rounds K/16 times instead of once (the reference keeps FP16 running
sums too). The 2×2 merge keeps FP32 accumulation (K = 4C, up to 512: the longest sums); the softmax weights are FP16
(w = f16(2^g), Σ in FP32, A = f16(f16(w · f16(2^7/Σ)) · 2^−7)). What it saves: the FP32 ↔ FP16 conversions around
every product and half the accumulator registers. The LDS layout (fragment-ordered K / V^T, single buffer where it
buys workgroups, merge staging aliased) and the wide kernels' work split are the FP32 kernels'; the wide FP16 forms
keep the FP16 accumulator chains in the same order, so they are bit for bit `k_stageh_*` (`test_wide_stages`), as
the two-window forms are (`test_two_windows_per_workgroup`).

Selecting it: network variant `fused16` (k_bench `--variants fused16`, `KNet(..., variant="fused16")`, `KFrame`), the
fastest FP16 plan: `k_stageh_*` at levels 0..2 and for e3 / d3, the wide kernels with NG = 4 for e4, m, d4
(`k_net.DEFAULT_PATHS16`). `<variant>@f16` runs any other path plan with the FP16 forms (`fused@f16`: the FP32 plan's
paths, `fused1@f16`, `fused2@f16`, `wide2@f16`, `wide4@f16`; "split" stages stay FP32). Stage runner: `"<stage>@f16"`.

Measured on the GPU (RX 9070 XT, 2026-10-09) against the NumPy reference with random parameters; errors relative to
the reference RMS, no non-finite values anywhere:

| Measure | `fused16` (FP16 accumulation) | `fused` (FP32, default) | Limit (spec 49 d4) |
|---|---|---|---|
| per stage: each of the 11 stages fed the chained reference's own input at the level sizes of base 256×256, 3 seeds, frames 0..2 (even offsets); worst mean / worst max | 1.06·10⁻³ (d4) / 1.32·10⁻² (d1) | 4.4·10⁻⁵ / 5.4·10⁻³ | 2.5·10⁻³ / 3·10⁻² |
| whole network at 1920×1080 and 2560×1440 output, s = 0.5 (`--check`): d0 head mean / max | 3.4·10⁻³ / 3.5·10⁻² | 1.6·10⁻³ / 2.8·10⁻² | 8·10⁻³ / 7·10⁻² |
| same, intermediate outputs: largest mean; max at levels 0–2; max at levels 3–5 | 3.2·10⁻³; 3.9·10⁻² (d1); 2.2·10⁻² (d3) | 1.5·10⁻³; 1.4·10⁻²; 1.1·10⁻² | 1.2·10⁻²; 0.12; 0.25 |
| realistic sequences (`--sequences`, 8 frames, s = 1): d0 head mean / max; intermediates max at levels 0–2 / 3–5 | ≤ 3.2·10⁻³ / 3.2·10⁻²; 3.6·10⁻² (e2) / 2.0·10⁻² | ≤ 1.5·10⁻³ / 1.5·10⁻²; 1.3·10⁻² / 7.8·10⁻³ | 8·10⁻³ / 7·10⁻²; 0.12 / 0.25 |

All within the limits, the head at about half of them. From the earlier experiment on the FP32 kernels before the wide
kernels and the LDS diet (same masks): FP16 accumulation everywhere, merge included, also passed (d0 head 4.1·10⁻³ /
4.9·10⁻² at 1080p, s = 1) at about the same speed; adding the residual into the Wo / W2 start (spec 41 allows it)
failed the d0 head maximum (7.4·10⁻²): every rounding of the running sum then happens at the magnitude of x. Real
parameters are not available to the clean team; the spec team's measurement against its own reference decides.

Speed, 2560×1440 output, s = 0.5 (base 640×384), `k_ab_net.py` (alternating processes, best of 5 rounds of the
median of 50 runs; GFLOP per stage from `k_bench.py`); the FP32 default is bit for bit 13e3402 (all buffers of
`fused`, `fused1`, `fused2`, `auto`, `wide2`, `wide4` compared at 1080p and 1440p, frames 0 and 3):

| Stage (GFLOP) | 13e3402 `fused` ms (TFLOP/s) | `fused` now | `fused16` ms (TFLOP/s) |
|---|---|---|---|
| e0 (9.58) | 0.219 (43.7) | 0.218 | 0.189 (50.7) |
| e1 (6.41) | 0.124 (51.7) | 0.124 | 0.109 (58.8) |
| e2 (2.19) | 0.056 (39.1) | 0.055 | 0.049 (44.7) |
| e3 (1.47) | 0.032 (45.9) | 0.032 | 0.023 (63.9) |
| e4 (0.69) | 0.051 (13.5) | 0.050 | 0.045 (15.3) |
| m (0.31) | 0.022 (14.1) | 0.022 | 0.022 (14.1) |
| d4 (0.69) | 0.046 (15.0) | 0.046 | 0.028 (24.6) |
| d3 (1.47) | 0.024 (61.3) | 0.024 | 0.050 (29.4) |
| d2 (2.19) | 0.069 (31.7) | 0.068 | 0.046 (47.6) |
| d1 (8.03) | 0.159 (50.5) | 0.159 | 0.137 (58.6) |
| d0 (10.09) | 0.223 (45.2) | 0.224 | 0.181 (55.7) |
| network (43.1) | **1.026** (42.0) | **1.025** | **0.878** (49.1) |

Stage times are events between consecutive stages of one network run, so the small stages' split between neighbours
is not exact (e3..d3 together: 0.175 → 0.168 ms); per kernel on its own (`--kernels`, 100 launches, launch overhead
included) the FP16 forms are 0.03–0.04 ms faster for e0 and d0, 0.01–0.02 for e1, d1, d2, and within ±0.01 at levels
3..5. The clocks of this GPU switch between two states between processes: in the faster one the same comparison read
0.89 (`fused`) → 0.80 ms (`fused16`). Path plans for `fused16` (one process, alternating, best of 9–15 rounds):
the FP32 plan's paths with the FP16 forms (`fused@f16`) 0.576 ms; `k_stageh_e3` / `_d3` with the wide kernels for e4,
m, d4 (NG = 2 or 4 in any mix) 0.523–0.541 ms, within the noise of each other (the FP16 wide e3 and d3 with NG = 2
lose their edge over the 128-lane kernels).

## Full interop benchmark and I/O wave size

The default build now compiles `k_io.co` and `k_io_exact.co` with 64-lane waves.
The network code objects remain wave32 and FP32 accumulation remains the default.
This changes scheduling only; the formulas, grid coverage, precision and buffers
are unchanged. `k_build.py --io-wave32` builds the previous I/O scheduling for A/B.
Both wave sizes pass the exact I/O, reset, whole-frame and runtime tests.

`clean/k/runner/k_interop_bench.py` generates random model parameters and alternates
two builds through the D3D12 input pass, HIP exposure/input/network/output and
D3D12 final output copy. It requires the test executable built from
`clean/runtime/interop/tests/test_interop_d3d12.cpp` (`hipbench` case). Run under the
shared GPU lock; all model paths are explicit synthetic paths. Timings are medians
of 300-dispatch window means after excluding each process's first window, with
automatic exposure, motion, cyclic jitter, inline output and full 1440p geometry.
`results/2026-10-09-model-k-e2e-wave64.md` records the 1.814 -> 1.699 ms result.
The 1.3 ms full-path goal is not yet met; these are synthetic interop measurements,
not game results. The benchmark checks the final output checksum and finiteness;
the separate exact tests check all intermediate/output values.

The interop option `timing_stages=0` (with `timing_log=1`) retains measured D3D12
TOTAL/input/output spans while omitting HIP stage markers. The benchmark flag
`--no-stage-events` requires both compared builds to honor this mode and records
it in JSON. Keep instrumentation equal when comparing builds. Default stage
tracing remains enabled; see `results/2026-10-09-model-k-measurement-controls.md`.

## Output preparation and luma soft limiting

`KFrame` and the native runtime prepare the output colour taps (`k_out_color`)
after exposure and before input construction. These taps depend only on current
colour/exposure. The input timing window includes this output prepass; the output
window starts at `k_out_pixel`. Hidden history now belongs to the neural interval
for the fused G2 paths ("fused" and, since 2026-10-13, "fused16"); fallback variants
retain the standalone output pass.
Standalone input/output stage plans retain their original launches.

`k_out_pixel` skips luma soft-limit arithmetic when every tap in its 5x5 history
window is below 0.99. High/NaN taps retain the original exact helper. This adds no
approximation. Combined full-interop evidence: 1.6865 -> 1.665 ms, full 1440p Quality,
FP32, unchanged checksums; see `results/2026-10-09-model-k-parallel-output.md`.
The 1.3 ms goal remains unmet.

## Exact output arithmetic helpers

`ko_rcp_refined` matches native RN32 reciprocal on the validated gfx1201 build,
using two FMA corrections inside [2^-32,2^32] and native division elsewhere.
`ko_div_axis` is limited to the cubic-axis beta coefficients; it corrects the
quotient residual and is not a general division helper. The production-code GPU
probe checks all FP32 reciprocal bit patterns and the complete cubic fraction
domain using separate output kernels and an independent integer comparator.
Run `python -m unittest clean.k.tests.test_k_output_arithmetic -v` with
`WG_K_GPU_BUILD` under the shared GPU lock. `WG_K_TEST_ROCM` may select the compiler
SDK separately from the runtime. New compiler/device targets need the same gate.
Full-path result: 1.6625 -> 1.651 ms; see
`results/2026-10-09-model-k-exact-output-arithmetic.md`. The 1.3 ms goal remains open.


The output near-history path now establishes the range once for each four-sample
batch before calling `ko_rcp_output_bounded`; invalid/negative/far cases retain the
full decode/encode path. Existing history/current-colour guards also bound the
colour ratio denominator. The additional exhaustive production-helper gate covers
all 536,870,913 FP32 patterns in [2^-32,2^32]. See
`results/2026-10-09-model-k-bounded-output.md` for the proof and paired measurements.

`ko_selector_normalize` uses a domain-specific reciprocal product. All 65,536
half inputs preserve the selector index; finite quotient bits also match.
The production-helper exhaustive test is `test_k_output_selector`. Paired full
interop: 1.6415 -> 1.6375 ms; see `results/2026-10-09-model-k-selector.md`.
FP32 accumulation and exact output behavior remain the default.


## Exact complete-frame fusion

Supported HIP frames fuse colour preparation into input construction and emit
hidden history directly from the final FP32 neural stage. Full geometry and
arithmetic order are unchanged. Auto-exposure frames use 16 algorithmic HIP
launches instead of 18; the profiling marker is counted separately. Unsupported
input geometry retains both original input kernels, and `input_tiled=0` explicitly
selects that fallback. Standalone stage interfaces and FP16 network behavior remain
available. Detailed guards and timing boundaries are documented in FORMATS.md.

Five full-path pairs measured 1.633 -> 1.533 ms (6.12%), with matching outputs and
22 integrated tests passing. The 1.3 ms objective remains unmet. See
`results/2026-10-09-model-k-fusion.md` for individual controls, live launch counts,
resource costs and rejected candidates; these are synthetic GPU/interop timings.

## Exact guarded hidden normalization

On gfx1201, the fused d0 hidden-history epilogue can use hardware reciprocal
square root when all four products are provably inside the original half-rounding
intervals. Tiny, zero, nonfinite and midpoint-near results use the original
normalization; other architectures always retain it. Final half bits and FP32
matrix accumulation are unchanged. The production-source regression checks the
entire denominator domain, rounding intervals and actual fast/fallback outputs.

Five matched full-path pairs measured 1.5505 -> 1.5440 ms, with all pairs improving,
18 integrated tests passing and unchanged d0 occupancy. This small gain does not
achieve the 1.3 ms target. See `results/2026-10-09-model-k-hidden-guard.md` for the
proof, measurement scope, operating-condition caveat and rejected alternatives.

## Half-input network normalization

The gfx1201 normalization reciprocal uses hardware rsq followed by half rounding.
All 65,536 half inputs preserve non-NaN result bits and NaN classification against
the previous expression; NaN payload/sign identity is not claimed. Zero/overflow
behavior is unchanged. Other architectures retain the original formula.

The integrated change passed 20 tests and five complete-path pairs measured
1.4845 -> 1.4805 ms. Both arms retain D3D input-through-output timestamps and
disable internal HIP stage events equally. Use `k_interop_bench --no-stage-events`
for goal comparisons; stage instrumentation adds measurable overhead and belongs
in separate diagnostics. The 1.3 ms target remains unmet. See
`results/2026-10-09-model-k-normalization.md` and
`results/2026-10-09-model-k-profile-pass.md` for evidence and limits.

## Shared tiled input and output-color preparation

Supported nonreset HIP frames now produce output ctap/T while loading the input
producer tile, using `k_in_tiled_color`. Disjoint ownership prevents duplicate
border writes; output luma retains its distinct operation order. Reset, unsupported
geometry, disabled tiling and exact-input Python calls retain separate preparation.
The original and color-producing kernels share one templated consumer body.

The actual integrated build passed 15 selected tests, eight full-resolution native
changing-frame comparisons, and five final full-path pairs: 1.5085 -> 1.5010 ms,
all five faster with identical output fingerprints. Both arms use the complete
D3D input-through-output scope and disable internal HIP events equally. The
matched saving is 0.0075 ms; earlier campaigns' absolute baselines differ and
must not be subtracted to manufacture a new absolute time. The 1.3 ms objective
remains unmet. See `results/2026-10-10-model-k-tiled-color.md` for ownership,
validation scopes, final build hashes and the rejected CPU launch-cache experiment.

## Native input frame uniforms

Supported native tiled frames now precompute the twelve KiHost constants once on
the CPU and pass an explicit 48-byte extension to the input kernel. This removes
ten full FP32 divisions per workgroup. GPU exposure and the original input math
remain separate; ordinary Python KFrame still calculates KiHost on the GPU.

The host route requires nearest rounding, zero or sufficiently large finite motion
scales, and finite computed scale factors. Other cases retain the original GPU
entries. `wg_rt_kernels` reports `input_uniforms=host|gpu` and launch counts.

Sixteen selected tests and eight full native changing frames pass. The final
five-pair full-path confirmation measured 1.4720 -> 1.4625 ms, all five faster,
with unchanged output fingerprints. Game-activity checks passed around all ten
arms. An earlier campaign overlapping Cyberpunk is retained and excluded.
The 1.3 ms objective remains unmet. See
`results/2026-10-10-model-k-input-uniforms.md` for scopes, guards and hashes.

## Reset-frame input path

History resets (first frame, explicit reset, or render-size change) select `k_in_reset`
after the unchanged colour prepass. Spec 42 sections 6.1, 6.5 and 7 require each
slot's channels to be `(current, current, 0, 0)` on a reset. The kernel therefore
reads only the current own-sample channel; it skips history, motion/depth, colour
neighbourhood and disagreement work. Steady frames still use `k_in_base`.
No arithmetic approximation, reconstruction-size change or network change is added.
Both fast and exact input builds support it; the runtime uses it automatically.

Synthetic equality and alternating full-pipeline timings are in
an internal results note of 2026-10-09. Reproduce under the shared GPU lock:
`python -m clean.k.runner.k_reset_bench --build <build> --rounds 7 --frames 200 --json <record>`.
This measures input + network + output with resident inputs, excluding exposure,
transfers and D3D12 interop. It is not an in-game measurement.

## Speed steps of 2026-10-13 (exact)

Both keep every output and history bit; A/B method: whole frames back to back in one process, the two forms
alternating (`results/2026-10-13-model-k-speed-*.md`, also for what was tried and rejected).

- `fused16` writes the hidden history from d0's epilogue as `fused` does (`k_stage2h_d0_hidden`): one launch and one
  pass over the head map less. 1440p Quality 1.287 -> 1.263 ms per frame, 4K Performance 2.761 -> 2.715 ms.
- Compact pixel head (default for complete frames, both networks): d0 writes only the three head channels per slot
  that `k_out_pixel` reads, as `[2 Bh][2 Bw][4]` FP16 in display-position raster order (`k_stage2_d0_hp`,
  `k_stage2h_d0_hp`, `k_out_pixel_c`); 23.6 MB less memory traffic per frame at 1440p. 1440p Quality -10 to -12 us,
  4K Performance -20 to -25 us per frame. The 40-channel head map (buffer `k.N`) needs `k_head=full` (runtime) or a
  `debug=True` frame (`KFrame`).

## Performance on the RX 9070 XT (what helped)

Measured with `k_bench.py` under the machine's GPU lock. Clocks vary between runs (the network alone ranged 2.6–6.3 ms
at 1440p on unchanged code), so steps are compared A/B in one process, per kernel, as the median of 40 launches.

Network (spec 41 stages, variant `fused`), 2560×1440 output; s = 0.5: `k_ab_net.py` against the previous commit
(alternating runs, best of 3 rounds); s = 1: one process, `fused1` against `fused` (best of 4 rounds of 40-run medians):

| Step | s = 0.5 (base 640×384) | s = 1 (base 1280×736) |
|---|---|---|
| one 128-lane kernel per stage and window | 1.020 ms (e3 + e4 + m + d4 + d3: 0.42) | 2.48 ms (levels 3..5: 0.53) |
| levels 3..5 by the wide kernels (`k_block_wide.hip`): NG groups of 4 waves per window, heads / output tiles / hidden chunks split between the groups, first tiles after each barrier preloaded; bit-exact | **0.722 ms** (0.13) | ≈2.36 ms (0.25) |
| round 2: e0 and d0 with two windows per workgroup (`k_stage2_*`, earlier variant `fused2`) in `fused` and `fused16` (and the runtime); bit-identical (KFrame outputs and histories, 3 sizes x 5 frames); sustained whole frames, one process, 10 rounds: frame 1.729 → 1.717 ms (`fused`), 1.657 → 1.649 ms (`fused16`); stage kernels alone (200 back-to-back launches): e0 0.127 → 0.117, d0 0.141 → 0.128 ms; runtime 1440p Q network part 0.590 → 0.581 ms | −0.01 ms | |

Measured on the way (same process): e3, e4, d4, d3 as separate kernels (`auto`) gave levels 3..5 0.53 → 0.40 ms at
s = 1, the wide kernels 0.25. Four hidden tiles per group and chunk instead of two made e4 / d4 twice as slow as
before (49 KB LDS); NG = 4 is slower than 2 for the 252-window stages at s = 1 (56 KB LDS, one workgroup per CU) and
equal elsewhere; ownership without branches (a group without a last tile repeats tile KT − 1) took m from 0.05 to
0.025 ms.

Input stage (spec 42), 2560×1440 output, render 1707×960 (Quality), bit-exact at every step:

| Step | HDR | SDR |
|---|---|---|
| start (`k_in_base`: one lane per base position, double-precision log / exp with IEEE divisions) | 14.5 ms | 7.1 ms |
| double-precision exp2 without divisions (r · (1/k), constants folded), log2 with a Newton reciprocal and computed directly (not ln · 1/ln 2), explicit fma | 7.3 ms | 2.4 ms |
| `k_in_color`: Y, Co, Cg, φ(Y) and the HDR weight once per render pixel (each was computed by ~20 display positions) | 4.7 ms | ≈2.4 ms |
| `k_in_base` one lane per display position (256 VGPRs + 44 spilled → 89, no scratch); the 4 lanes meet in LDS for the hidden-history mean | 4.1 ms | 2.3 ms |
| own-sample channel 0 per render pixel in `k_in_color`; channel 1 = channel 0 when the history is invalid (Yh = Yc) | 3.7 ms | 1.8 ms |
| no FP64: log2 / ln / 2^x in FP32 double-word arithmetic with a rounding test, binary64 only for the ~10⁻⁴ undecided inputs (identical results, checked exhaustively) | 1.22 ms | 0.77 ms |
| default build "fast" (spec 42 §9: 42-T1 accepts ≥ 99.9 % within one FP16 step): v_log_f32 / v_exp_f32 and v_rcp_f32 where a division feeds the FP16 outputs; measured 100 % within one step (max 1, ≥ 99.98 % identical, 0 non-finite) on all 8 variants × steady / reset and at 1440p | **0.80 ms** | **0.58 ms** |

Stage `input` (and `KFrame`) use the fast build `k_io.co`; `input@exact` (`KFrame(..., exact_input=True)`) runs
`k_io_exact.co`, the same source built with `WGK_IN_EXACT`, bit for bit the reference.

Output stage (spec 43), same frames, bit-exact at every step (`k_out_color` + `k_out_pixel` + `k_out_hidden`):

| Step | HDR | SDR |
|---|---|---|
| start (one lane per pixel; 16 stored colour taps and 12 filter taps recomputed per pixel; 2^E from a 128 KB table; 80 luma-history reads) | 1.7 ms | 1.6 ms |
| `k_out_color` pre-pass (stored tap and tap value once per render pixel), 2^E by `v_exp_f16` (equal to the table for all 65536 inputs), 64 × 2 pixel workgroups sharing Ȳ for F2 through LDS, the 5 × 5 luma window read once | **1.23 ms** | **1.12 ms** |
| (later steps, s = 0.5: 32-bit addressing, FP32 1/√, FP16 quotient in FP32; `k_out_pixel` alone) | 0.90 ms | 0.81 ms |
| frame-uniform values from the host (`KOutUni`: 1/W, 1/H, ρ, ρ/2, jitter − ½, motion scale, 1/W₁₆, 1/H₁₆, h(−8/(ρ/2)), 1/κ, each the same correctly rounded FP32 operation; 9 FP32 divisions per pixel less) and y wave-uniform (`readfirstlane`: a wave lies in one row, so the row's values are scalar); each alone −3 %; `k_out_pixel`, A/B interleaved in one process | **0.84 ms** | **0.76 ms** |
| packed FP16 in `k_out_pixel`: sub-samples (i, 0) and (i, 1) as the halves of `v_pk_*` operations for 5.4 weights, 5.5 sums, 6.1 blend weights and β (`kh2_*` in `k_fp.h`, each half rounded like the scalar op); best-E selection by a compare-free mask (`kh2_gtmask`: sign of h(b − a) with a bit test for NaN, checked against IEEE `>` for all 2³² FP16 pairs) and `v_bfi_b32`; 6771 → 6590 instructions, 86 VGPRs; A/B interleaved in one process (2 × 500 launches): `k_out_pixel` −0.022 to −0.029 ms (2–4 %) | −2–4 % | −2–4 % |
| both together (A/B against the build before them, alternating, 2 × 80 launches; `k_out_pixel` alone; whole stage HDR ≈ 0.81 ms) | **0.69 ms** (was 0.77) | **0.62 ms** (was 0.68) |
| (s = 0.5 and later network changes; `k_out_pixel` alone, sustained: 200 back-to-back launches, median of 6 rounds) | 0.785 ms | 0.71 ms |
| best-E tracking only where it can decide (the largest E by `v_pk_max_num_f16`; the exact tracking runs again for lanes whose largest E < −12 or whose sum is NaN, after both sub-sample pairs), one 8-byte load per colour-history texel, merged stores (colour history and RGBA16F output 8 bytes, luma history 2 × 4 bytes, stored taps 8 bytes); 86 VGPRs; runtime frame 1440p Q: output part 0.89 → 0.86 ms | **0.72 ms** | **0.64 ms** |

Where `k_out_pixel`'s time goes (HDR, timing builds cut after each section, sustained): stores and launch 0.09 ms,
network channels and motion 0.015, colour-history sample 0.09, luma window and cubics 0.14, history luma 0.02 to 0.04,
current colour and F2 0.045, table entry and taps 0.11, filter (24 packed taps) 0.21 → 0.14 with the step above, the
rest 0.06. Its own DRAM traffic is about 200 MB per frame at 1440p (≈ 0.3 ms at full bandwidth): reads of the
histories, network output, motion, depth and the k_out_color buffers, writes of both histories and the output.
Also tried without gain (timing only, kept out): three dword loads per luma-window row with a funnel shift (5 instead
of 25 loads); fetching the table entry and taps before the luma history with the m0 a valid pixel has (127 VGPRs: the
lower occupancy cost more than the overlap gained; 96-VGPR caps spill); the nearest-depth motion choice precomputed per
render pixel (bound measured by skipping it: ≤ 0.03 ms, before the pre-pass's own cost); no workgroup barrier for F2
(bound: no change); two pixels per lane through a lambda (the compiler runs them one after the other); register caps
of 48–72 VGPRs (spills); integer addressing of the 12 filter taps (bound with clamped integer indices and no fallback
for large coordinates: ≤ 0.004 ms).

Tried without gain (kept out): fusing the hidden history (6.8) into `k_out_pixel` for s = 1 (the 2 × 2 group's
lanes are a base position's four slots; Hh written through LDS, `k_out_hidden` only for the uncovered base rows).
`k_out_pixel` grew by about what `k_out_hidden` costs (~0.15 ms) whichever lane computed 1/√ and wherever the five
extra reads were placed, so the separate kernel stays.

The input stage's last step reaches what it took with FP32 hardware log / exp (wrong results, measured as a bound). Ablations
(timing only) then show the rest is mostly ALU: 14.7 M fine positions per frame at 1440p (motion, a 4-texel history
sample, φ in HDR) and ~26 M correctly rounded functions (HDR); removing the nearest-depth search, the history samples
and the colour taps together leaves 0.74 / 0.44 ms. The stage's own DRAM traffic is about 120 MB (≈ 0.2 ms).

Network (fused stages, spec 49 draft 4), 2560×1440 output (base 1280×736), bit-exact at every step; per stage the
best of 3 rounds of the median of 30 launches, A/B in one process (`k_ab_net.py`):

| Step | e3 | e4 | d4 | d3 | d1 | total |
|---|---|---|---|---|---|---|
| start (K and V^T rows padded to 40 / 72 halfs, always double-buffered, merge staging separate: 13–35 KB LDS) | 0.112 | 0.082 | 0.082 | 0.111 | 0.365 | 2.46 ms |
| LDS diet: K / V^T stored tile by tile in fragment order (no padding, conflict-free: lane l reads and writes slot l), one buffer where two would cost workgroups per WGP (d1–d4), the encoders' merge staging in the K / V^T space; 8–24.5 KB (d1 / d2 27.6 → 16.4 KB, e3 / e4 31.7 / 35.8 → 16.4 KB) | **0.052** | **0.034** | **0.069** | **0.057** | **0.347** | **2.29 ms** |

The low-resolution stages gain most: they have few windows, so more resident workgroups per WGP (the kernels run in
WGP mode, 128 KB LDS per WGP) shorten their last wave. Unpadded rows without a swizzle cost e0 + d0 about 0.2 ms
(8-way bank conflicts on the V^T reads); an XOR swizzle of the 16-byte chunk was about as fast in total but needs
more address registers (m: 12 spilled VGPRs). e0, e1, d0 and d2 are unchanged within the noise (±5 %); m (24 windows,
latency-bound) is about 0.01 ms slower from a different instruction schedule.
