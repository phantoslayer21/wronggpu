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
| `tests/test_fg_net1_gpu_plan.py` | the chain as planned on the CPU: every step output's shape against the reference's buffers, launch arguments, natural weight order, chain lines |
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
