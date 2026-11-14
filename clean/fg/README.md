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
| `../converter` (preset FG) | the converter reads a version-1 map with `"preset": "FG"`: one block source per weight blob; see the converter's README, "Preset FG" |
| `../../data/fg.v1.map.json` | the spec side's converter map v1 (weights as stored); `tests/test_fg_map.py` pins its digest and checks it against `layout.py` |
| `tools/real_check.py` | runs the converter once on one given frame-generation library (execute-only) and records only the verdict, the map's digest, the file version and size, the matching stage digests and the converted file's digest; deletes the converted file. Offsets and bytes of the library are never recorded |
| `tests/test_fg_real_check.py` | the tool on made-up converter output and a stand-in converter |
| `reference/config.py` | the settled per-call configuration: the multi-frame intermediate time (one single-precision division), the index checks and flags, UI recomposition (FG preset A / B) with the profile values that force it |
| `tests/test_fg_config.py` | the intermediate time against exactly rounded quotients, the index rules, every forcing profile value |

```
python -m unittest discover -s clean/fg/tests -t .
```

Test data is synthetic. Model data and file bytes never go into this repository.
