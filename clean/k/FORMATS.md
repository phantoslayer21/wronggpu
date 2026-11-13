# Model K formats (feature maps, packed weights, launch geometry)

Specs: 40 draft 4, 41 draft 3, 44 draft 1, 48 draft 2. Code: `clean/k/runner/k_pack.py` (Python), `clean/k/runtime/model_k.cpp`
(sizes, C++), `clean/k/runtime/k_pack.cpp` (the runtime's packer, C++, same bytes). The converter (spec 48) is to write exactly these bytes.

## Feature maps

Every map is FP16, `[H][W][C]` (channels innermost, rows of tokens), at the level sizes of spec 40/44
(`level L = ceil(Wb / 2^L) x ceil(Hb / 2^L)`, base `Wb x Hb` from spec 44 with the pending factor s = 1). The
40-channel head output is `[Hb][Wb][40]` (row of 80 bytes). Buffers are 256-byte aligned.

### Frame buffers (`layout.buffer_plan`, `model_k.cpp k_buffer_plan`)

| Buffer | Content | Size |
|---|---|---|
| X0 | network input (spec 42), level 0, 16 ch | Wb·Hb·32 B |
| S0 … S4 | skip maps = block outputs of e0 … e4 | level size · C · 2 B |
| A, B | scratch maps the stage results alternate between | largest map each holds |
| OUT | head output, level 0, 40 ch (to spec 43) | Wb·Hb·80 B |

Fused network (default; one kernel per stage, `layout.FUSED_IO`): e0 X0→S0 (block) and A (merged); e1 A→S1, B;
e2 B→S2, A; e3 A→S3, B; e4 B→S4, A; m A→B; d4 B (+S4)→A; d3 A (+S3)→B; d2 B (+S2)→A; d1 A (+S1)→B; d0 B (+S0)→OUT
(the head; d0's block output is not written). A and B hold at most a level-1 map with 64 channels. At 4K output
(base 1920×1088): X0 64 MiB, S0 128 MiB, S1…S4 88 MiB, A 64 MiB, B 64 MiB, OUT 159 MiB: 566 MiB.

Split network (separate kernels, `layout.stage_io`): e0: embed X0→A, block A→S0, merge S0→A; e1…e4: block A→S_k,
merge S_k→A; m: block A→B; d4…d1: expand B (+S) →A (padded, below), block A→B; d0: the same, then head B→OUT. A is
sized for the largest padded map, 8·⌈(W+7)/8⌉ × 8·⌈(H+7)/8⌉ tokens: 695 MiB at 4K.

### Padded decoder block input (spec 41 draft 3 border rule)

`[8·nwy][8·nwx][C]` for the stage's window grid (offset (ox, oy)): padded position (py, px) is map position
(py − oy, px − ox), inside or outside the map; it holds x = f16(e(m′(⌊r/2⌋), m′(⌊c/2⌋))[k] + S(m(r), m(c))). Written
by `k_expandp_*`, read by the block kernel with `padded = 1` (window (a, b), member (row, col) at (8b + row, 8a + col)).

## WMMA tile format (every weight matrix)

`W [N, K]` (output index first) becomes 16 × 16 tiles. Tile `(nt, kt)` holds rows `16 nt … 16 nt + 15` and inputs
`16 kt … 16 kt + 15` as 32 lanes × 8 halfs (512 bytes): lane `l = 16 h + r` holds A row `r`, inputs
`16 kt + 8 h + 0 … 7`. A row `r` holds output channel `16 nt + 8 h' + v` where `r = d_row[h'][v]` (gfx12: `r = v + 8 h'`,
so row r = channel 16 nt + r). Order "nk": tile `(nt, kt)` at index `nt · KT + kt`; order "kn": at `kt · NT + nt`.
N is padded with zero rows to a multiple of 16 (only the head: 40 → 48).

## Linear blob (embedding, merge, expand, head; `pack_linear`)

| Section | Content |
|---|---|
| w | tiles, order nk |
| b | bias, FP16 `[N padded]` (zeros if the layer has no bias) |

Every section starts at a multiple of 256 bytes.

## Block blob (`pack_block`, spec 41 "One block")

| Section | Content |
|---|---|
| g1 | `[C]` (spec 48's g1 = γ1) |
| g2 | `[C]` |
| qkv | per head h: softmax `Wq_h`, `Wk_h`, `Wv_h`; fixed `Wv_h`; each `[32, C]` as tiles `[2][C/16]` (nk) |
| mix | `[heads][64][64]`: `B_h` (softmax) or `M_h` (fixed), row t = query, column s = key (window-local 8·row + col) |
| wo | `Wo [C, D]` tiles nk `[C/16][D/16]`; input index = head-major `32 h + c` |
| bo | `[C]` |
| w1 | `W1 [4C, C]` tiles nk |
| b1 | `[4C]` |
| w2 | `W2 [C, 4C]` tiles **kn** `[4C/16][C/16]` (one hidden tile's 16 inputs for all outputs are contiguous) |
| b2 | `[C]` |

The block kernel uses the `Wv` tiles as the B operand of the swapped product `v^T = n^T Wv^T`; that needs the gfx12
accumulator layout, so the packer refuses any other `d_row` for blocks.

## Stage blob (`pack_stage`)

| Stage | Parts in order |
|---|---|
| e0 | embed (linear 32×16) · block · merge (linear 64×128) |
| e1 … e4 | block · merge (linear C_next × 4C) |
| m | block |
| d4 … d1 | expand (linear 4C × C_prev) · block |
| d0 | expand (linear 128×64) · block · head (linear 40×32, padded to 48) |

Sizes: `k_pack.stage_sections(st)["_size"]` = `model_k.cpp k_stage_blob_bytes`.

## Input / output stages and histories (specs 42, 43)

| Image | Format | Written by | Read by |
|---|---|---|---|
| network input X0 | `[Bh][Bw][16]` FP16 (channels 4k … 4k+3 = slot k) | `k_in_base`, `k_in_reset` or `k_in_tiled` | e0 |
| network output OUT | `[Bh][Bw][40]` FP16 (10 per slot) | d0 head | `k_out_pixel`; standalone hidden fallback reads it too |
| compact pixel head (in the OUT buffer; complete G2 frames by default) | `[2 Bh][2 Bw][4]` FP16 in display-position raster order: slot k of base position (x, y) at row 2y + (k >> 1), column 2x + (k & 1), holding head channels 10k, 10k + 1, 10k + 9 and a zero (the three channels `k_out_pixel` reads; the same rounded values) | `k_stage2_d0_hp` (FP32), `k_stage2h_d0_hp` ("fused16"), which also write Hh | `k_out_pixel_c` |
| colour history Hc | `[⌈H⌉₁₆][⌈W⌉₁₆][4]` FP16 (Y, Co, Cg, age) | `k_out_pixel` | next frame's `k_in_base`/`k_in_tiled`, `k_out_pixel` |
| luma history L | `[2⌈H⌉₁₆][2⌈W⌉₁₆]` FP16 | `k_out_pixel` | next frame's `k_out_pixel` |
| hidden history Hh | `[Bh][Bw][4]` FP16 | complete G2 frames: `k_stage2_d0_hidden` (FP32) or `k_stage2h_d0_hidden` ("fused16"); otherwise `k_out_hidden` | next frame's `k_in_base` or `k_in_tiled` |
| game colour / motion / depth | RGBA float32 / 2 × float32 / float32, row pitch in the launch arguments, subrect bases | the front end | both stages |
| game output | `KOutParams.omode`: 0 RGBA float32 `[H][opitch][4]` (alpha Ā); 1 RGBA16F rows of `opitch` texels (RGB rounded to nearest even, alpha 1: the interop's output copy); 2 dense RGB float32 `[H][opitch][3]` (the interop's output pass) | `k_out_pixel` | the front end |
| exposure scalars `xs` | float32 [4]: κ, ρ, λ, RN(1/κ) of spec 42 section 4 / spec 43 3.7 (`KFrameIn.xs`; null = the struct's κ, ρ, λ and `KOutUni.kinv`) | `k_expo_scalars` (runtime) | all I/O kernels |
| filter table | `[2048][8]` FP16, the frame's set (spec 43 section 7), generated on the host per level | host | `k_out_pixel` |
| 2^E table | `[65536]` FP16: h(2^E) for every binary16 bit pattern E | host (once) | `k_out_pixel` |
| input scratch `cpre` | `[rh][rw][4]` float32: (Y, Co, Cg, φ(Y)) of each render pixel (spec 42 6.2) | `k_in_color` | `k_in_base` (same frame) |
| input scratch `cpre2` | `[rh][rw][2]` float32: (HDR weight max(1/(Y+1), 10⁻⁴), own-sample channel S(T(λ L(Yc)))) | `k_in_color` | `k_in_base` (same frame) |

The histories are double-buffered (`KFrame`: frame f reads copy p and writes copy 1 − p). Launch arguments:
`KFrameIn` (shared), `KOutParams` (ending in `KOutUni`, the output stage's frame-uniform FP32 values computed on the
host by `k_io.out_uniforms`), `KInParams` in `k_io.hip` / `k_io.py`. Canonical standalone I/O uses one lane per pixel of the ⌈W⌉₁₆ × ⌈H⌉₁₆ grid
(`k_out_pixel`), per base position (`k_out_hidden`), per render pixel (`k_in_color`, launched first), per display
position (`k_in_base`: lanes 4b … 4b + 3 are the display positions of base position b), 128 lanes per workgroup.
The scratch is sized for the largest render size (`k_io.input_scratch_bytes`) and retained for fallback.
Complete-frame fused geometry and the additional d0 hidden-output argument are described below.

### Runtime frame (`wronggpu_runtime.dll`, runtime option `model=K`; `runtime/runtime_k.cpp`)

The per-frame runtime runs `KFrame`'s frame on its HIP stream (the interop's FrameIo buffers, or its own test buffers):

| Step | Kernels | Notes |
|---|---|---|
| input copy (option `input_copy=1`) or the compact input pass (interop default for K) | none: the K kernels read the copies | the game's RGBA16F colour (`KIO_C16`, not divided by π: model K applies κ), R32F depth and FP16 motion of 2 or 4 channels (`KIO_M16`, m = ms · r + mo on load: FP32 multiply then add, as `input_convert` computes it) in their 256-byte pitched rows; `KFrameIn.mch`, `rmsx`, `rmsy`, `rmox`, `rmoy`; the same values as the FP32 slot of the input pass, so the same output bits (GPU test) |
| exposure | `exposure_auto_diet` (`exposure.co`, spec 07, when auto-exposure is asked; `exposure_auto_diet_c16` on the RGBA16F copy), `k_expo_scalars` (`k_io.co`) | the exposure pair `[e, e_prev, last e]` of spec 10 (value, the game's texel on the GPU, or spec 07), then `xs` |
| preparation | `k_out_color`, then `k_in_tiled` when supported; otherwise `k_in_color` plus `k_in_base`/`k_in_reset` | `KFrameIn.flags` = [HDR] from `WgRtConfig.hdr` (front end `WG_CREATE_HDR`), always [MVlo], [inv], reset |
| network | the eleven `k_stage_*` / `k_stagew<NG>_*` launches of `k_net.py` variant "fused" | window offsets `k_stage_offsets(f, net/window_offsets)`; the G2 d0 also emits hidden history and, by default, the compact pixel head instead of the head map (`k_stage2_d0_hp`, `k_stage2h_d0_hp`) |
| output stage | `k_out_pixel_c` (compact pixel head, default) or `k_out_pixel` (runtime option `k_head=full`) | game output omode 1 (FrameIo.out_half_pitch > 0) or 2 (interop dense RGB) |

`k_expo_scalars(ex, src, xs, KExpoArgs)` (one lane): with `pair` = 1 it first updates the pair as `exposure_pair`
(e = src[0] or `value`, e_prev = e on the first frame, else the previous e); then E_cur = e, E_prev = e_prev (E_cur on
a reset frame), E_aux = E_cur, each rounded to FP16 with 0 → 1, and the FP32 operations of `k_io.scalars()` with
e₁ = e₂ = 1 and `inv_pi` = RN32(1/π) (π: the pre-exposure, 1 if missing, ≤ 0 or not finite). The kernels then read
the same values the host would give for that exposure (bit-identical to `KFrame` given that E_cur).
`KExpoArgs` = {float value; int pair, use_src, first, hdr, reset; float e1, e2, inv_pi} (36 bytes).

State: Hc, L, Hh double-buffered, zero at create; reset on the game's flag, the first frame and a render-size change;
frame index f = 0 on a reset frame, else the previous f + 1; `WgRtFrame.reset = 2` (a parked upscaler taken over by
a new context) is a reset that also starts the host frame state over as at create (frame counter, so the first-frame
exposure; f; history parity; last size; jitter), so the frames from it on equal a new runtime's; filter-shape sets of all eight levels uploaded once at
create (the frame takes level `k_filter_level(rw, rh, W, H)`); `exp2tab` allocated and zero (the GPU uses v_exp_f16).
s = ½, base `k_base_size(W, H, ½)`. Buffers for `wg_rt_buffer`: `exposure.pair`, `k.scalars` (xs), `k.x0`, `k.N`
(network output, the 40-channel head map: only with the runtime option `k_head=full`; the default frame writes the
compact pixel head instead, buffer `k.Np`, and asking for `k.N` fails with a message naming the option; `k.Np` in
turn fails with `k_head=full`),
`k.Hc`, `k.L`, `k.Hh` (the copies the next frame reads), `filter.out` (own output: dense RGB f32,
not with external I/O), `input.color`, `input.depth`, `input.motion`. Timing marks (FrameIo.timing_base,
`wg_rt_frame_io_timed`): exposure = first kernel → `k_out_color`; input = preparation → first neural stage;
network = eleven stages (including Hh in FP32 d0) → `k_out_pixel`; output = reconstruction and any standalone
hidden pass → the final empty timing marker. Full end-to-end comparisons retain all these intervals.

Model data: `<model_dir>/K/model.wgcm` (runtime option `model_k_dir=<folder>` overrides), checked against the preset-K
map `<build_dir>/preset-K.v1.map.json` (on disk, else built into the library; `model_k_map=<file>` overrides).
Network: `network=fused` (default) or `network=fused16` (FP16 accumulation, spec 49 speed approximation, opt-in:
`k_stagew4h_*` for e4, m, d4, `k_stageh_*` for the others); `wg_rt_kernels` reports `network=<variant>`.
Head layout (same output bits): `k_head=compact` (default) or `k_head=full` (the 40-channel head map and
`k_out_pixel`, for diagnostics that read `k.N`); `wg_rt_kernels` reports `head=<compact|full>`.
Timing diagnostics (same output bits): `k_diag=input2` runs the input stage once more before mark 1 (the "input" part
is then a second run, the first is counted in "exposure"); `k_diag=copyin` copies the interop's colour, depth and
motion into K's own buffers first (`hipMemcpyAsync`, before mark 0); `k_diag=input2+copyin` both.

## Kernels and launch geometry

All kernels: 128 lanes (4 waves of 32), 1-D grid, arguments as one struct by value (`k_launch.py`).

| Kernel | Code object | Grid | Work |
|---|---|---|---|
| `k_lin_<K>x<N>`, `k_merge_<K>x<N>`, `k_expand_<K>x<N>`, `k_expandp_<K>x<N>` (padded decoder input) | `k_linear.co` (32 tokens per wave), `k_linear__g1.co` (16) | ceil(tokens / (64 G)) × output chunks | a wave: 16 G tokens × 16·NWT output channels (NWT = largest of 8, 6, 5, 4, 3, 2, 1 dividing N/16) |
| `k_block_c<C>_h<heads>_<fixed\|softmax>`, `k_blockp_…` (padded input) | `k_block.co` | windows (nwx · nwy) | one 8×8 window per workgroup; wave w = window rows 2w, 2w+1; LDS: K and V^T of the current head as fragment-ordered 16 × 16 tiles, double-buffered (8 KB fixed, 16 KB softmax) |
| `k_stage_<e0…e4, d4…d0>`, `k_stage2_<e0, d0>` (two windows per workgroup) | `k_block.co` | windows (k_stage2: ⌈windows/2⌉) | the block with the stage's other operations fused in (below); LDS adds 64·C halfs for the window's block input (decoders, e0) or merge staging (encoders: shares the K / V^T space); decoders keep one K / V^T buffer when two would cost workgroups per WGP (8–24.5 KB in all) |
| `k_stagew<NG>_<e3, e4, m, d4, d3>` (NG = 2, 4; m: 4) | `k_block_wide.co` | windows (128 NG lanes each) | one window per NG groups of 4 waves (`k_block_wide.hip`): heads, output tiles and hidden chunks split between the groups; bit for bit `k_stage_*` (m: the plain block) |
| `k_stageh_<e0…e4, m, d4…d0>`, `k_stage2h_<e0, d0>`, `k_stagew<NG>h_<…>` | as above | as above | the same with FP16-accumulating matrix products (opt-in spec 49 speed approximation, below); `k_stageh_m` takes the plain block's arguments |
| `k_win_gather`, `k_win_scatter` | `k_window.co` | 16-byte chunks / 128 | window partition round trip (K3 tests) |
| `k_act_table`, `k_act_table2` | `k_act.co` | 512 / 256 | activation on all 65536 FP16 inputs (scalar / packed) |

`k_expandp`: the WMMA columns are the virtual coarse positions (⌊r/2⌋, ⌊c/2⌋) over the padded grid
(`KLinParams.vw` × ncols / vw), the source read at the mirrored coarse position, the skip at the mirrored fine position
(`fw`, `fh`), the output at the padded position (`ox`, `oy`).

Linear flags (`KLinParams.flags`): 1 bias (the FP32 accumulators start from it), 2 activation, 4 residual / skip add
at the output position, 8 ReLU. Order: f16(W x + b) → ReLU → act → f16(r + y).

Fused stage kernels (`k_stage_*`, even window offsets only, which spec 44 always gives; `KBlkParams` input mode and
epilogue fields `wa`/`ba`, `wz`/`bz`, `skip`, `merged`, `head`, `cw`/`ch`):
- e0: every window member is embedded on load (`f16(We in + be)`, ReLU) from the mirrored network-input token.
- d4…d0: the window covers 4 × 4 coarse tokens (the coarse origin is (8a − ox)/2, (8b − oy)/2), loaded through the
  mirror at the coarse size and expanded to 4C channels with one WMMA column set; each output tile goes to LDS at
  the member it feeds (sub-position s = tile·16 / C). Every member then adds the skip map (mirrored at this level).
  This is the border rule of spec 41 draft 3.
- e0…e4: after the block, the window's 16 2×2 groups (all inside one window, because the offset is even) are the
  WMMA columns of the merge; the block output goes through LDS (members outside the map as zero).
- d0: the head (`f16(Wh x2 + bh)`, 48 padded rows, 40 stored) in the epilogue.

Numerics of the `h` forms (`k_stageh_*`, `k_stage2h_*`, `k_stagew<NG>h_*`; a spec 49 speed approximation, opt-in:
network variants `fused16`, `<variant>@f16`, stage suffix `@f16`; template parameter `F16` of `k_block` /
`k_block_wide`, built with `WGK_F16ACC` = `KF_DEFAULT` of `k_common.h`, override with `-DWGK_F16ACC=<mask>`): the
embedding, expand, q / k / v, scores, z, Wo, W1, W2 and head products run on `v_wmma_f16_16x16x16_f16`, whose FP16
accumulator is rounded to nearest even after every 16-deep step (a product with K = 16 n rounds n times instead of
once); each starts at its FP16 bias (the scores at B) and its FP16 result is the next product's operand. The 2×2 merge
keeps FP32 accumulation. Softmax weights in FP16: w = f16(2^g) (`v_exp_f16`), Σ w in FP32 (`v_dot2_f32_f16`),
A = f16(f16(w · f16(2^7/Σ)) · 2^−7). Everything else (normalization, residual adds, activation, clamps), the LDS
layout and the launch geometry are as in the FP32 kernels. The accumulation order is the same in all three forms, so
`k_stage2h_*` and `k_stagew<NG>h_*` give bit for bit the results of `k_stageh_*`. The host emulator sums each FP16
WMMA in FP32 and rounds once per call.

Window of a block: workgroup id → window (a, b) = (id mod nwx, id div nwx); token t = 8·row + col of the window is
map position (8a − ox + col, 8b − oy + row), read through the mirror rule (encoder and middle stages) or from the
padded map (decoder stages, `KBlkParams.padded`); only real positions are written. Input and output maps must be
different buffers (windows read neighbouring tokens).

## Reset input kernel

`k_in_reset` uses the unchanged 200-byte `KInParams` and 128-thread geometry of
`k_in_base` (four lanes per base position). Select it only when `KIO_RESET` is set,
after `k_in_color` has produced `cpre2`. Each slot writes `(h, h, +0, +0)`, with h
the rounded current own-sample channel. No history, motion, depth, exposure-scalar
or `cpre` reads occur in this kernel; exposure has already been used by the
prepass. The final partial workgroup skips positions outside the base map.

## Whole-frame fusion and order (2026-10-09)

After exposure, `k_out_color` prepares only current colour/exposure taps. It runs
before the input stage. Supported complete HIP frames then use `k_in_tiled`:
256 lanes cover 8×8 base tokens and prepare a 24×24 current-colour tile in shared
memory. It writes the same `x0`, using the original scalar operation order and a
four-lane transpose for hidden features. Reset directly writes `(h,h,0,0)` and
reads no discarded history/motion/depth. Global cpre/cpre2 buffers remain allocated
for fallback but are null in the tiled launch arguments and unused by that kernel.

Selection requires s=0.5, positive output dimensions ≤16384, base dimensions ≤4096,
render/output ratios ≤0.68, and finite jitter in [-0.5,0.5]. An unexpected local
footprint miss computes the original tap directly. Unsupported geometry and the
native option `input_tiled=0` retain BOTH original input launches. The default is
`input_tiled=1`; `wg_rt_kernels` reports requested/actual route and launch counters.
Standalone stage plans and host-emulator input retain the canonical pair.

Complete G2 frames use `k_stage2_d0_hidden` (FP32) or `k_stage2h_d0_hidden` ("fused16"), which emit Hh directly
from the rounded head fragments before their register lifetime ends. Existing standalone
stage interfaces and G1/split paths retain `k_out_hidden`. History parity,
formats and full reconstruction sizes are unchanged. Network timing now includes
Hh for the fused path; output timing begins at `k_out_pixel`.

Supported nonreset FP32 frames additionally fuse output ctap/T preparation into
`k_in_tiled_color`. Its explicit 216-byte argument is KInParams followed by ctap
and ctapT pointers at offsets 200 and 208. Selection also requires 4*Bw >= W and
4*Bh >= H. Consecutive clamped tile origins define disjoint ownership intervals;
unclamped producer coordinates suppress duplicate border stores. The ratio and
coordinate bounds keep each owned interval within 24 samples. Input and output
retain their distinct luma operation orders. Reset, exact-input Python calls,
unsupported geometry and disabled tiling retain separate output-color preparation.
The ordinary and color-producing entries share one templated input body.

Actual application HIP launches on supported steady FP32 frames: 15 with auto
exposure (2 exposure, 1 preparation, 11 neural, 1 output), or 14 with scalar
exposure. Reset tiled frames add one separate color launch; unsupported/disabled
input tiling uses two input passes plus separate output color. Stage profiling adds one empty
`timing_barrier` separately; D3D-only timing does not. Sync operations and D3D
input/flag dispatches/output copy remain unchanged. See the source-backed API
audits and full-path evidence in `results/2026-10-09-model-k-fusion.md`.

### Native tiled input uniforms

The native runtime can supply KiHost's twelve FP32 dimension/motion values as an
explicit 48-byte argument extension. `k_in_tiled_uniform` takes unchanged
KInParams followed by KiHost at offset 200 (248bytes total).
`k_in_tiled_color_uniform` takes unchanged KInColorParams followed by KiHost at
offset 216 (264bytes total). Both retain 256threads and the original tile grid.

The native guard requires FE_TONEAREST and raw motion-scale magnitude bits equal
to zero or in [0x01000000,0x7f800000), then finite sp_x/sp_y. Integer classification
prevents DAZ from treating a subnormal scale as zero. Unsupported geometry,
non-nearest rounding, smaller scales, nonfinite scales or overflowing scale factors
retain the original GPU KiHost computation. No CPU floating-point setting is
changed. GPU-resident exposure is still supplied through KFrameIn.xs.

Existing 200/216-byte kernel APIs and ordinary Python launch plans are preserved.
`wg_rt_kernels` adds input_uniforms=host|gpu and input_uniform_launches. The number
and ordering of algorithmic launches are unchanged by this optimization. The
CPU/GPU uniform probes live in tests and are excluded from production build lists.
