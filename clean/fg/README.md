# Frame generation (issue #96, milestone "Later")

Groundwork for running the two convolution networks of frame generation. Nothing here is used by the runtime, the
front end or the launcher yet. Status, open items and formats: [FORMATS.md](FORMATS.md).

| Path | Content |
|---|---|
| `layout.py` | the weight layout of both networks as structure only: per layer its shape, and the byte size and inner layout of each blob, in the library's upload order; totals |
| `reference/blobs.py` | splits blob bytes into flat float16 weights (stored order) and distinct biases: net1's zero area skipped and checked, copy 0 of each bias group taken and the other copies checked; net2's weight / bias pairs; `build_net1` for synthetic blobs |
| `tests/test_fg_blobs.py` | round trips and every check, on synthetic blobs of every layer |
| `tests/test_fg_layout.py` | the table against the totals the spec side gives |
| `reference/ops.py` | NumPy CPU reference of the networks' building blocks: float16 convolution (3 x 3, 1 x 1) with bias and ReLU, 2x pooling, bilinear 2x upscale (align corners), element-wise combines, net1's input features and 2 x 2 fold; each open behaviour is a required argument or raises `OpenItem` |
| `tests/test_fg_reference.py` | the building blocks against plain loop versions, on synthetic data |
| `reference/forward.py` | both networks' forward pass: parameters (from blobs, the converter's raw output or its model file); `net1_forward` with net1's delivered wiring; net2 from a wiring the caller supplies (conv / pool / upscale / add / concat / slice steps), the wiring and the border rule required |
| `tests/test_fg_forward.py` | every layer of both networks against a direct NumPy convolution on synthetic weights; each step kind; wiring checks; stand-in wirings against layers composed by hand; parameters from blobs and from raw files |
| `reference/net1.py` | net1 complete: the fold, the 32 steps as data, float16 chunk accumulation from the bias in each layer's chunk order, zero border, leaky ReLU, max-pool, the align-corners upscale with float16 steps, float16 adds, the unfold |
| `reference/fp16acc.py` | exact float16 chunk accumulation: RN16(acc + 16 products) with integer arithmetic |
| `tests/test_fg_net1.py`, `tests/test_fg_fp16acc.py` | net1 end to end against a hand-composed version; one convolution per chunk order against exact rationals; the upscale against the scalar rule; pooling with inf / NaN; the accumulation against exact rationals |
| `tools/net1_digest.py` | net1 on a converter model file with a fixed synthetic input; records digests of the output only |
| `kernels/fg_exact16.h` | exact float16 arithmetic in integers (chunk accumulation, one nearest-even rounding, leaky ReLU), shared by the HIP kernels and their D3D12 ports, so both compute the same bits by construction |
| `kernels/fg_net1.hip` | HIP kernels of net1's operations (code object `fg_net1.co`): `fg_conv_f16` (3 x 3 / 1 x 1, zero border, chunk orders, leaky ReLU or none) , `fg_maxpool2_f16` (2 x 2, the delivered formula), `fg_upscale_f16` (bilinear, align corners, float16 steps) and `fg_add_f16` (the skip add); D3D12 ports `clean/dx12/kernels/fg_conv.hlsl`, `fg_pool.hlsl`, `fg_upscale.hlsl`, `fg_add.hlsl` |
| `runner/fg_d3d12_check.py` | GPU check (one GPU slot): every kernel on HIP and D3D12 byte for byte (sentinels 0xA5 / 0x5A, slack bytes, a half-grid negative control), and the HIP output against the CPU reference bit for bit; net1's layer shapes on several grid sizes and special values |
| `runner/fg_net1_gpu.py` | net1's 32 steps chained on the GPU (HIP, and D3D12 through kernel_replay's `chain` blocks): every step output on both back ends byte for byte and against the CPU reference bit for bit, a half-grid negative control per kernel, synthetic weights (also odd sizes and special values) and, with `--model`, a converter output file's weights (execute-only: its per-run files are deleted, digests only); `--timing` prints net1's milliseconds at 1080p- and 1440p-sized inputs as information, after a probe rules out long single launches |
| `runner/fg_wmma_probe.py`, `kernels/fg_probe.hip`, `kernels/fg_probe_mid.h`, `clean/dx12/kernels/fg_probe.hlsl` | probe of gfx12's 16 x 16 x 16 matrix instructions (float16 and float32 accumulate) against net1's chunk (RN16 of the accumulator plus 16 products, one rounding): random, adversarial and layer-chain classes, counts and step distances per form, mismatching examples; HIP and D3D12 byte for byte. Evidence only, no runtime change |
| `records/wmma-probe-2026-11-14.json` | the probe's record (HIP): neither form equals the exact chunk. On ordinary data (random, net1-like, the layer chains; 62.4 million outputs) float32 accumulate rounded to float16 per chunk is more than one step off in 1,216 outputs and more than four in 211; float16 accumulate in 6.05 million and 1.86 million |
| `tests/test_fg_wmma_probe.py` | the probe's CPU parts: the float16-midpoint test against brute force, the RN32 reference against exact fractions, the case classes |
| `tests/test_fg_net1_gpu_plan.py` | the chain as planned on the CPU: every step output's shape against the reference's buffers, launch arguments, natural weight order, chain lines |
| `kernels/fg_net1_mma.hip`, `clean/dx12/kernels/fg_mconv.hlsl`, `fg_upscale_hwc.hlsl` | net1's fast path, tier B (code object `fg_net1_mma.co`): `fg_mconv1/2/4` (one `v_wmma_f32_16x16x16_f16` per spec chunk in the layer's chunk order, the float16 accumulator from the bias as C, the result rounded to float16 after every chunk; leaky ReLU and the fused 2 x 2 pool in the epilogue; NHWC float16) and `fg_upscale_hwc_f16`. Not bit-exact to the CPU reference (the matrix unit's chunk sum is not exact); HIP and D3D12 byte-identical |
| `kernels/fg_net1_mma.hip` `fg_mpair_*`, `clean/dx12/kernels/fg_mpair.hlsl` | two chained 1 x 1 convolutions in registers (a result fragment is the next convolution's input fragment): the bottleneck (net1_06 + net1_07) and each decoder level (net1's upscale and skip add computed per input fragment, then net1_08/09 ... net1_16/17); u / s / e are never stored |
| `kernels/fg_fast16.h`, `clean/dx12/kernels/fg_leaky_fast.hlsl` | the fast path's leaky ReLU (one single-precision product, one float16 rounding; `fg_leaky16`'s bits, checked on all 65,536 inputs) and skip add |
| `runner/fg_net1_mma.py` | net1 on three selectable paths, HIP and D3D12: `fused` (11 launches), `tierb` (27 launches) and `exact` (#191's kernels, bit-exact: the validation mode and the fallback). Gates: every buffer byte for byte between the back ends (sentinels, slack, a half-grid negative control per kernel, incl. a 768 x 768 case where every kernel has one; synthetic weights, odd sizes, special values, `--model` execute-only), fused == tierb, the leaky check. Report against the CPU reference: values over 1 and over 4 steps, largest step distance, largest absolute difference and that over the largest reference magnitude, per buffer and per output value. `--timing` at 1080p- and 1440p-sized inputs (information only, after a launch-length probe) |
| `records/net1-tierb-2026-11-14.json`, `records/net1-fused-2026-11-14.json` | the fast path's records: gate results, the report against the CPU reference (synthetic and converted weights, counts only), timing, and (fused) the D3D12 profile per launch before and after the driver conversion |
| `tests/test_fg_net1_mma_plan.py` | the fast path as planned on the CPU: every weight fragment element against the weights, chunk order and tap; buffers and launches of the 27- and 11-launch plans; the fast leaky ReLU's formula on all 65,536 inputs |
| `../converter` (preset FG) | the converter reads a version-1 map with `"preset": "FG"`: one block source per weight blob; see the converter's README, "Preset FG" |
| `../../data/fg.v1.map.json` | the spec side's converter map, revision v1.1 (weights in natural order); `tests/test_fg_map.py` pins its digest and checks it against `layout.py` |
| `reference/natural.py` | the natural weight order from the map itself: its offsets as gather indices (stored blob weights to natural), natural to canonical `[out, k, k, in]` |
| `tests/test_fg_natural.py` | natural weights scattered into synthetic blobs with the map's gathers come back unchanged, for every layer; the gathers cover each weight span |
| `tools/real_check.py` | runs the converter once on one given frame-generation library (execute-only) and records only the verdict, the map's digest, the file version and size, the matching stage digests and the converted file's digest; deletes the converted file. Offsets and bytes of the library are never recorded |
| `tests/test_fg_real_check.py` | the tool on made-up converter output and a stand-in converter |
| `reference/config.py` | the settled per-call configuration: the multi-frame intermediate time (one single-precision division), the index checks and flags, UI recomposition (FG preset A / B) with the profile values that force it |
| `tests/test_fg_config.py` | the intermediate time against exactly rounded quotients, the index rules, every forcing profile value |

```
python -m unittest discover -s clean/fg/tests -t .
```

Test data is synthetic. Model data and file bytes never go into this repository.
