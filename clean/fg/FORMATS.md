# Frame generation: formats and open items

What the clean side has for frame generation, and what it still waits for. **OPEN** marks behaviour that has not been
delivered by the spec side; the code never guesses it (it raises, or takes the choice as an explicit argument).

## Weights

- One weight set for every configuration: both FG presets (UI recomposition off and on), every multi-frame count and
  index, every combination of optional inputs. The same blobs are in every 310.x frame-generation library.
- Two networks, float16 elements, ReLU activations, no attention, no FP8:
  - **net1**: a six-level encoder-decoder on a 2 x 2-folded grid; 17 blobs, each `[weights][zero area][bias x 16]`.
  - **net2**: two small encoder-decoders (sub-nets A and B) of 3 x 3 convolutions; 25 layers, a weight blob and a bias
    blob each. Each layer's grid level relative to the network grid (`Net2Layer.level`, and `upscaled` for the two
    layers whose output is upscaled 2x) is in the table; sub-net A's last layer has no stated level (OPEN), and
    `net2_macs_per_pixel` takes it as an argument.
- Shapes and byte sizes per layer: [`layout.py`](layout.py) (checked by `tests/test_fg_layout.py`).
- Splitting a blob into its parts: [`reference/blobs.py`](reference/blobs.py). net1: the weights in stored order, the
  zero area checked to hold only +0 / -0, the bias as copy 0 of each group of eight with the other fifteen copies
  checked to be bit-identical (`check=False` skips both checks). net2: weight and bias blob sizes checked. The weights
  come out flat in stored order; [`reference/natural.py`](reference/natural.py) puts them in natural order with the
  map's own offsets.
- net1's blob 2 is a 1 x 1 convolution (32 -> 64). net2's first layer stores 16 input channels and reads 10: its
  natural weights are [32, 10, 3, 3].

## Converter map

- Map version 1 with `"preset": "FG"` (converter README, "Preset FG"): one block source per blob, located by length
  and hashes, `f16` reads. net1's zero area is skipped by the tensor's base; its bias is read as copy 0 of each group of
  eight (`layout.net1_bias_index`).
- The map is `data/fg.v1.map.json`, the spec side's revision v1.1 (digest pinned in `tests/test_fg_map.py` and the
  converter's tests): sources `A01`..`A17` (net1) and `B01w`/`B01b`..`B25w`/`B25b` (net2), stages `dl1.l01`..`dl1.l17`
  and `dl2.l01`..`dl2.l25`, in upload order, exactly the blobs and spans of `layout.py`. `"channel_order": "natural"`.
  Per stage: `W` in natural order, [out, in, 3, 3] for 3 x 3 layers and [out, in] for 1 x 1 layers, every axis by
  explicit offsets into the blob; `b` [out] (net1: copy 0 of each bias group; net2: the bias blob, natural order).
- The first revision (v1) gave the weights as stored (`"channel_order": "storage"`, 1-D `W_storage`); the converter
  still accepts that form for preset FG.
- A conversion of a real library is checked by digests only (the converter compares every stage digest the map
  gives); no value or byte of it is ever written into the repository or any public text. `tools/real_check.py` runs
  that check and keeps only the verdict and digests (`python -m clean.fg.tools.real_check --exe <converter> --dll
  <library>`).

## CPU reference (`reference/ops.py`)

- Tensors are channel-first `x[c, y, x]`; weights use the module's own canonical order `w[out, ky, kx, in]`
  (`weight_shape(layer)` for every layout layer). `natural.py` turns the map's natural `W` ([out, in, ky, kx] or
  [out, in]) into that order (`to_canonical`), and stored blob weights into natural ones with the map's offsets.
- Settled and implemented: float16 operands and outputs; convolution with bias, one float16 rounding of the output,
  then ReLU; 2x pooling over 2 x 2 blocks; bilinear 2x upscale in the align-corners form (output sample i reads input
  position i (n - 1) / (2n - 1)); element-wise add and channel concatenation; net1's features: luma with fixed RGB
  weights, warp error as the mean absolute RGB difference clamped to [0, 1], a mask and its complement; the 2 x 2 fold
  of ten features into a cell of 40 values padded to 48.
- Arithmetic: float64 sums with a single float16 rounding per output (`acc="exact"`). The original sums in float16 in
  its matrix unit's order (OPEN): `acc="f16"` raises `OpenItem`, and a later GPU comparison has to allow for that
  difference until the order is known.
- Required arguments where the behaviour is OPEN: the convolution border rule (`pad`: zero, replicate or mirror),
  ReLU after a net2 layer (`relu_after`), the pooling kind (`kind`: max or avg), net1's feature channel order
  (`order`), the fold order inside a cell (`order`). What the cell padding holds is OPEN too (zeros here).
- `ops.net1_forward` and `ops.net2_forward` raise `OpenItem`: the original's wiring is not delivered.

## Forward pass (`reference/forward.py`)

- Parameters per layer (layout names `net1_01`..`net2_25`): canonical natural weights `[out, k, k, in]` and the bias,
  from blobs (`params_from_blobs`: stored weights gathered with the map's offsets) or from the converter's `--raw`
  output of the preset-FG map (`params_from_raw`, stages `dl1.lNN` / `dl2.lNN`, tensors `W` and `b`; every value
  checked to be exact in float16).
- Also from the converter's model file (`params_from_model_file`, container version 1, checksum checked).
- `net1_forward(values, params)`: net1 with its delivered wiring and arithmetic (next section); no other arguments.
- net2 (wiring not delivered): `run_layer` (one convolution with `ops.conv2d`), `run_graph` (a list of steps: `conv`
  with its layer and ReLU flag, `pool` with its kind, `upscale`, `add`, `concat`, `slice`; checked before it runs)
  and `net2_forward`. Required arguments (OPEN): `wiring`, `pad`.

## net1 (`reference/net1.py`)

- Input: 10 float16 values per pixel of the input grid, folded 2 x 2 into cells of 48 channels: channel 4 v + q with
  q = 2 dy + dx; channels 40..47 and pixels outside the input area are zero. Output: 16 channels per cell, unfolded the
  same way to 4 values per pixel.
- 32 steps (`NET1_WIRING`, data): five levels down (convolution, then 2 x 2 max-pool), the two bottom 1 x 1
  convolutions, five levels up (upscale to the skip's size, float16 add of the skip, two 1 x 1 convolutions). Level
  sizes halve with floor; the skips are the encoder outputs after the activation.
- Convolution: stride 1, zero border, cross-correlation over the natural weights. Float16 accumulator started at the
  bias; 16-term chunks (k = (3 ky + kx) in + c, or c for 1 x 1) each summed exactly and added with one float16
  rounding (`fp16acc.acc_dot16`, integer arithmetic). Chunk orders: 1 x 1 ascending; blobs 1, 3, 4 channel block outer,
  then the taps with kx outer and ky inner; blob 5 kx outer, then channel block, then ky.
- Activation: leaky ReLU y = max(x, h(s x)) after every convolution but the last (blob 17, linear), with the delivered
  slope s = 0.0099945068359375 (float16 0x211E). Nearest-even rounding of 0.01 would give 0x211F; the delivered value
  is used as given.
- Max-pool: 2 x 2, stride 2, floor sizes, m <- (x > m) x + (x <= m) m in float16 (inf and NaN give the same result in
  any tap order; two +inf give NaN, as the formula does).
- Upscale: bilinear, align corners, from the coarser level to the skip's size: single-precision positions
  s = ((n_in - 1) / (n_out - 1)) i, x0 = trunc(s), x1 = trunc(s + 1) (both n_in - 1 at the edge), float16 weights,
  every product and sum rounded to float16. The original's ratio is an approximate division; here it is the correctly
  rounded single-precision quotient.
- Add: h(a + b).
- `tools/net1_digest.py`: net1 on a converter model file with a fixed synthetic input; records digests only.
- Not part of net1 and still open: what the 10 input values are made of (images A-D, the warps) and the input grid
  size relative to the output.

## GPU kernels (`kernels/`, code object `fg_net1.co`; D3D12 ports `clean/dx12/kernels/fg_*.hlsl`)

- Groundwork: no runtime launches them yet. Tensors float16 channel-first `[c][h][w]`, weights natural
  `[out][in][k][k]` float16 (the map's `W`), biases `[out]` float16; a 1-D grid of 256-lane workgroups, one output
  element per lane.
- `fg_conv_f16(x, w, b, y, cin, cout, H, W, k, order, act)`: one net1 convolution exactly as `net1.conv16`: stride 1,
  zero border, float16 accumulator from the bias, 16-term chunks each summed exactly and rounded once, chunk order
  `order` (0 ascending, 1 channel block / kx / ky, 2 kx / channel block / ky), `act` 0 none or 1 leaky ReLU. cin is a
  multiple of 16.
- `fg_maxpool2_f16(x, y, C, H, W)`: net1's max-pool exactly as `net1.maxpool2`: 2 x 2, stride 2, floor sizes, taps
  (0, 0), (0, 1), (1, 0), (1, 1) through the delivered step m <- (x > m) x + (x <= m) m with IEEE products and sums
  (0 * inf is NaN; a sum of zeros is -0 only when both are -0).
- `fg_upscale_f16(x, y, C, hi, wi, ho, wo, rx, ry)`: net1's upscale exactly as `net1.upscale_to`: bilinear, align
  corners; rx / ry are the ratios (n_in - 1) / (n_out - 1) as correctly rounded single-precision bits from the caller
  (0 for an extent of 1), so no division runs on the GPU. Positions in single precision one operation at a time
  (HIP: no contraction); every product and sum rounded to float16 (a product of two float16 values is exact in single
  precision, and single precision then float16 equals one rounding of a sum of two float16 values).
- `fg_add_f16(a, b, y, n)`: net1's skip add exactly as `net1.add16`: y = h(a + b) per element (single precision then
  float16, one rounding of the exact sum).
- net1 on the GPU is these four kernels in NET1_WIRING's order, one launch per step, every buffer resident on the
  device (`runner/fg_net1_gpu.py`; on D3D12 a kernel_replay `chain` block, with a UAV barrier between launches).
- The arithmetic is integer code shared by both back ends (`kernels/fg_exact16.h`), the method of `fp16acc.py`; NaN
  results are written as 0x7E00. The D3D12 build compiles `fg_` kernels only when `--only` names them, so Model M's
  and Model L's kernel sets and DXILs do not change.
- `runner/fg_d3d12_check.py` proves each kernel on HIP against D3D12 (raw bytes) and against the CPU reference;
  `runner/fg_net1_gpu.py` the whole network, every step output. These exact kernels stay as the validation mode.

## Fast path, tier B (`kernels/fg_net1_mma.hip`, code object `fg_net1_mma.co`; D3D12 `fg_mconv.hlsl`, `fg_upscale_hwc.hlsl`)

- Tensors NHWC float16 (element (y, x, c) at (y W + x) C + c), so 8 consecutive channels of a pixel are one 16-byte
  fragment load or store. Weights as WMMA A fragments in chunk order: `[cout / 16][chunks][32 lanes][8]` float16, lane
  l holding output channel 16 mt + l % 16 and input channels 16 cb + 8 (l / 16) + 0..7 of the chunk's tap
  (`runner/fg_net1_mma.py` `swizzle`).
- `fg_mconv{1,2,4}(x, wf, b, y, p, cin, cout, H, W, k, order, act, pool, tiles_x, tiles_y)`: the spec's chunk structure
  on `v_wmma_f32_16x16x16_f16`: the accumulator starts at the bias; each chunk (16 input channels of one tap, in the
  layer's chunk order) is one instruction with the float16 accumulator as C, and its result is rounded to float16
  (nearest even) before the next chunk. Not bit-exact: the instruction's sum is not the exact one
  (`records/wmma-probe-2026-11-14.json`). An exact-zero sum gives +0, the reference's convention. Leaky ReLU and, with
  `pool` 1, the 2 x 2 max-pool are the exact helpers of `fg_exact16.h`, in the epilogue. 1 / 2 / 4 output-channel tiles
  of 16 per wave; a workgroup is 8 waves over 16 x 16 pixels.
- `fg_upscale_hwc_f16`: `fg_upscale_f16`'s arithmetic on NHWC tensors. The adds are `fg_add_f16`.
- `fg_mpair_{b,d4,d3,d2,d1,d0}(x, a, wf1, b1, wf2, b2, y, C, H, W, hi, wi, rx, ry, act2)`: two chained 1 x 1
  convolutions (chunks ascending), the first with leaky ReLU, kept in registers: the result fragment of output tile mt
  (lane = pixel, 8 consecutive channels) is exactly the second convolution's input fragment of channel block mt. `_b`:
  the input is the buffer x (the bottleneck, net1_06 + net1_07). `_d<k>`: the input is net1's upscale of the coarse
  x [hi][wi][C] plus the skip a [H][W][C], computed per input fragment with the exact upscale and add arithmetic (ratio
  bits rx / ry from the host); act2 0 for net1_17 (no activation). Pixels row-major, 16 (or 32) per wave; lanes past the
  end compute the last pixel and do not store.
- Leaky ReLU on the fast path: `fg_leaky_fast16` (`kernels/fg_fast16.h`): one single-precision product with the slope,
  one float16 rounding, then the max of `fg_leaky16`; the same bits as `fg_leaky16` for all 65,536 inputs.
- Paths (`runner/fg_net1_mma.py --paths`): `fused`, 11 launches (five encoder convolutions with the pool fused, the
  bottleneck pair, five decoder pairs); `tierb`, 27 launches (every convolution, upscale and add on its own); `exact`,
  #191's 32 launches. fused and tierb compute the same arithmetic and write byte-identical common buffers. HIP and D3D12
  are byte-identical on every path; against the CPU reference the tier-B output is a report (net1's output is not
  gated; the gate is the generated frame against the oracle), the exact path is bit-exact.
- D3D12 rounds float16 with the driver channel's conversion (`wg_to_f16_ags`, the bits of HIP's `(_Float16)`), a whole
  fragment at a time (clean/dx12/README.md "Driver exceptions").

## net2 (`reference/net2.py`; D3D12 `clean/dx12/kernels/fg_net2_*.hlsl`, fast path `fg_net2_mconv.hlsl`, `fg_net2_upwarp.hlsl`)

- Grid: `grid(Wo, Ho)` from the frame size (4..8192 x 4..7680): k the smallest power of two with floor(Ho / k) <= 480,
  W2 = 16 ceil(Wo / 16k), H2 likewise; `padding` (right, bottom); `extents` of every buffer.
- Inputs: Ib, Ic [H2, W2, 4], Ia [H2, W2, 1] float16 (channel innermost) and one single-precision scalar f. Output:
  `out` [H2, W2, 8]. What the inputs, f and the 8 output channels mean is OPEN (item 7).
- `NET2_LAUNCHES`: the 28 launches as data (input merge, L1..L3 with the 2 x 2 average pool, the residual block
  L4..L11, the L12 split, the three L13 heads, the weightless upsample, the warp-merge, L14, L15, the block L16..L23, the
  L24 split, the L25 heads with their residuals); `run` returns every buffer, `forward` the output.
- Arithmetic: convolution chunks tap by tap (row-major), 16-channel blocks inside a tap, each exact with one float16
  rounding from the bias (`fp16acc.acc_matmul16`, the same bits as `acc_dot16`); leaky ReLU with slope bits 0x211F and a
  clamp to [-6, 6] except where a layer leaves it out; the 2x half-pixel upsample and the warp sample with fused
  multiply-adds rounded once (`fma16`); IEEE signed zeros outside the chunks (an exactly zero chunk result is +0).
- D3D12: `fg_net2_conv` (convolution with the chunk code of `fg_exact16.h`, activation and clamp rule, residual, fused
  pool; channel stride and offset so a launch reads one group and writes one channel range), `fg_net2_up2`,
  `fg_net2_merge`, `fg_net2_warp`; float16 helpers in `clean/dx12/hlsl/fg_net2.hlsli` (the single-rounding multiply-add
  is `wg_fma_f16`). The network is 38 launches: the upsample inside L12, L13, L24 and L25 is its own launch, and L13 /
  L25 run one launch per head. Exact (bit-identical to the CPU reference), not tuned for speed.
- Check: `runner/fg_net2_d3d12_check.py` (each kernel, an adversarial multiply-add probe, the whole network at small
  grids and at 480 x 272 / 640 x 368, negative controls and mutants; `--timing` only after a passing record).
- Fast path (tier B, D3D12, code object `fg_net2_mma.co`; the exact kernels above stay the validation mode):
  `fg_net2_mconv_<variant>` (one `v_wmma_f32_16x16x16_f16` per spec chunk in the tap-major order, the float16
  accumulator from the bias as C, rounded to float16 after every chunk; the post-ops, pool, clamp rule and residual
  exact in the epilogue; variants by output tiles per wave, 16-channel blocks per tap, workgroup size and upsample
  staging), with the 2x upsample of L12, L13, L24 and L25 computed exactly into the workgroup's LDS tile and the three
  heads of L13 / L25 in one launch; `fg_net2_upwarp` (the head upsample and the warp-merge in one launch, exact); 8-value
  float16 helpers with the bits of `fg_net2.hlsli` in `clean/dx12/hlsl/fg_net2_fast.hlsli`. The network is 27 launches.
  Only the chunk sums differ from the CPU reference.
- Check: `runner/fg_net2_mma.py` (byte gates on exactly summable values for every variant and on adversarial probes of
  every rewritten rounding with their mutants, determinism over two sentinel fills at every grid, negative controls,
  chain mutants, the exact chain alongside; differences from the CPU reference reported; `--timing` only after a passing
  record, wall and GPU time).

## Configuration (`reference/config.py`)

- No configuration value changes the networks or their weights.
- Multi-frame: count N (1 by default, at most a cap), index i in 1..N. Intermediate time t = i / (N + 1) as one
  single-precision division (`intermediate_time`, numpy float32). Flags: multi-frame active (N > 1), last index
  (i equal to the maximum index).
- UI recomposition: preset A (off) or B (on), set by the application; a profile value of 1 or 0xFFFFFE forces it off,
  2 or 0xFFFFFF forces it on, any other value leaves it (`ui_recomposition`).
- Required arguments where the behaviour is OPEN: the cap on N (`n_max`), the further mode that also accepts
  i = N + 1 (`extra_index`), the maximum that "last index" compares with (`maximum`).

## Light path, preset C (`reference/stages_common.py`, `lite.py`, `pipeline.py`; D3D12 `fg_lite_*.hlsl`)

The shared non-network stages and the light blend, delivered by the spec side as behaviour; the choices below are
the clean side's own where the delivery leaves them open or names a GPU approximation.

- **Storage** (internal images are arena buffers, row-major, channels interleaved): colour, reference and UI images
  float16 x 4; net1-grid colour float16 x 4; luma change, sigmoid map and all masks float16; quality, logit pairs,
  weights and the resolved previous-to-current field float16 x 2; processed motion float32 x 2; packed depth and flag
  images uint32; the three 64-bit scatter buffers int64; pyramid levels float16 x 3 (plus float32 x 3 partial sums,
  device only); net1's cells as `fg_net1_mma.py`'s HWC float16 (fused path) or CHW (exact path).
- **Arithmetic**: single precision, one rounding per operation in the written order, fma where fused. The
  approximate division is `a * rcp(b)` with `rcp` the correctly rounded reciprocal (`wg_div(1, b)` on the device);
  `ex2` is our own deterministic 2^y (floor, a degree-6 polynomial in fma form, an exact power-of-two scale, flush below
  2^-126); clamps follow minNum / maxNum; float-to-integer conversions truncate and saturate (NaN -> 0). Single-precision
  denormals are flushed on the device; the test data stays away from them.
- **Textures**: nearest = texel `clamp(floor(X * W))`; bilinear = the four clamped texels around `X * W - 0.5`, blended
  with fma along x, then y. Application inputs are read through their region with the clamp to its texel centres;
  the clamp bounds are rounded up / down on the host. Reads at a pixel's own centre on its own grid are integer reads
  (equal by the size bound).
- **Choices**: the scatter grid is the frame divided by 2 (by 4 above 2880 lines); the pull pyramid's coarsest level
  keeps its own values (no tile fill); the intermediate-time footprint measures the warped 3 x 3 positions from each
  neighbour's own position; pixels of an odd-width net1 image outside the 2 x 2-fold grid write no quality / occlusion
  value; the hudless detector reads the hudless bilinearly when its base offset differs from the colour's.
- **Not modelled**: the full path (presets A / B) and the distortion field (both refused), the copy of the colour into
  the application's second output, and the on-screen indicator.
## Full path stages (`reference/stages_full.py`)

- Storage: colour, UI, both candidates, the distortion pair and the output pyramid levels 4 x float16; blend weights,
  quality image, the previous -> current field and the distorted fields 2 x float16; the hole flag image and the fine
  flag image one float16 (0 / 1: only "> 0" is ever read); processed motion 2 x float32; packed depths 32-bit words;
  MA / MB 64-bit words (key high, h(vx) in bits 31..16, h(vy) in bits 15..0, empty = 0x7FFFFFFF7FFFFFFF). net2's
  inputs Ib, Ic 4 x float16 and Ia 1 x float16 per element of the padded grid, row-major.
- Arithmetic choices of the clean side (the original's approximate operations have no defined bits): division, the
  reciprocal and 2^x correctly rounded (the kernels use the IEEE division and a 65,536-entry table of the logistic of
  net2's weight channel); bilinear and nearest reads computed by hand (no sampler); block sums as a pairwise tree, x
  before y at every scale, the same at k = 8 and 16; float16 writes round to nearest even.
- Output pyramid: levels from the frame's halving chain; pull writes levels 1..4, the middle pull 5..8 when the chain
  has more than 9 levels; push fills from the coarsest written level down, each filled level in its own buffer; the
  original's tile-local fallback for an empty coarsest element is not reproduced.
- The application's distortion field is read as 4 x float32.

## OPEN (waiting for the spec side)

1. (closed: map v1.1, natural order, is `data/fg.v1.map.json`.)
2. (closed: the in-blob axis order is the map's offsets.)
3. (closed: net2's launches, residual adds, splits, upsample and merges are delivered: `reference/net2.py`.)
4. (closed: net1's blob 2 is a 1 x 1 convolution.)
5. net1's grid size relative to the output size (net2's: `net2.grid`).
6. What the UI-recomposition preset changes in the stage order.
7. The motion-field scatter, the pull pyramid and the UI handling; net1's input images A-D and the warps.
8. Whether the networks run once per real frame or once per generated frame.
9. (closed: net2's border rule, accumulation order, activation and pooling are in `reference/net2.py`; net1's in
   `reference/net1.py`.)
10. The cap on the multi-frame count; the further mode that accepts index N + 1 and what it does; which maximum the
    "last index" flag compares with; the private override that can turn UI recomposition off.
