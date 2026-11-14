# Frame generation (issue #96, milestone "Later")

Groundwork for running the two convolution networks of frame generation. Nothing here is used by the runtime, the
front end or the launcher yet. Status, open items and formats: [FORMATS.md](FORMATS.md).

| Path | Content |
|---|---|
| `layout.py` | the weight layout of both networks as structure only: per layer its shape, and the byte size and inner layout of each blob, in the library's upload order; totals |
| `tests/test_fg_layout.py` | the table against the totals the spec side gives |
| `reference/ops.py` | NumPy CPU reference of the networks' building blocks: float16 convolution (3 x 3, 1 x 1) with bias and ReLU, 2x pooling, bilinear 2x upscale (align corners), element-wise combines, net1's input features and 2 x 2 fold; each open behaviour is a required argument or raises `OpenItem` |
| `tests/test_fg_reference.py` | the building blocks against plain loop versions, on synthetic data |
| `../converter` (preset FG) | the converter reads a version-1 map with `"preset": "FG"`: one block source per weight blob; see the converter's README, "Preset FG" |

```
python -m unittest discover -s clean/fg/tests -t .
```

Test data is synthetic. Model data and file bytes never go into this repository.
