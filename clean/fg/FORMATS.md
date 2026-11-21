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

## OPEN (waiting for the spec side)

1. (closed: map v1.1, natural order, is `data/fg.v1.map.json`.)
2. (closed: the in-blob axis order is the map's offsets.)
3. net2's fused kernels: stride inside a kernel, the residual add, the three-way splits, the upsample and merges.
   (net1's launch order is delivered.)
4. (closed: net1's blob 2 is a 1 x 1 convolution.)
5. Both networks' grid sizes relative to the output size.
6. What the UI-recomposition preset changes in the stage order.
7. The motion-field scatter, the pull pyramid and the UI handling; net1's input images A-D and the warps.
8. Whether the networks run once per real frame or once per generated frame.
9. For net2: the border rule and the accumulation order. (For net1 both are delivered, with the activation, the
   pooling kind, the fold order and the cell padding.)
10. The cap on the multi-frame count; the further mode that accepts index N + 1 and what it does; which maximum the
    "last index" flag compares with; the private override that can turn UI recomposition off.
