# Model L (issue #89)

Work in progress for a new model choice, milestone 0.7.0. Nothing here is used by the runtime, the front end or the
launcher yet.

| Path | Content |
|---|---|
| `reference/table.py` | Model L's structure table: the encoder's and the decoder's op lists (pixel-level linears and width-32 block, the 2 x 2 gather projection; the decoder projection, its width-32 block and the value positions), window offsets, block constants; parameter shapes per converter-map stage; matrix work per frame (computed, not measured). Entries in `PROVISIONAL` are pending confirmation and are changed here only |
| `reference/oracle.py` | NumPy oracle (float64, only the prescribed FP8 / FP16 roundings): `encode`, `decode`, `network`, synthetic `random_params`, parameters grouped by the converter map's stages; the decoder's coefficient functions are Model M's. Both width-32 blocks run in the FP16 form (`_block_tokens_q7`, any width 32 / 64 / 96 / 128: FP16 accumulators per 32-wide FP8 op, polynomial-and-bit-shift attention weights, the FP16 GELU polynomial, the FP16 norm tree with sqrt(C) in the gain, residuals as accumulator starts; table consts `form`); the decoder block's input is RN16(projection + skip), the skip being the encoder block's FP8 output read with its own fine mirror (table `skip_map`, swin op `skip`). The arithmetic inside one FP8 matrix op is the table's `mm8` model: `exact` (the reference) or `trunc13`; records name it. The ten shared stages still run Model M's oracle (table PROVISIONAL `S8`). The C32 kernels do not yet compute the FP16 form (`l_q7_negative_control.py` shows it). The Swin stages it shares with Model M call Model M's oracle (`clean/reference/swin.py`); the output reconstruction is not written yet |
| `runner/l_pack.py` | weight packing of Model L's own launches: the encoder's launch list derived from the table (`encoder_launches`) and one blob per launch (`pack_model_l`), each in a format Model M already uses (merge / expand section, Swin block section, enc0 and dec0 blobs); decoders for tests. The ten shared stages use `clean/runner/packing.py` unchanged |
| `kernels/l_swin.hip` | the C = 32 Swin block kernels (`l_swin_block_c32[_nc]`) from Model M's block template (`clean/kernels/swin.hip`, included for its templates only), same blob format and launch geometry as Model M's block kernels |
| `runner/l_c32_check.py` | GPU check (synthetic): the C = 32 kernel against the oracle, `_nc` against the clamping kernel, and this file's C = 64 build of the template against Model M's `swin_block_c64` |
| `tests/test_l_pack.py` | packing round trips, the block packer against Model M's stage blobs, and the unpacked weights applied the kernels' way against the oracle's maps, byte for byte |
| `tests/test_l_reference.py` | self-checks on synthetic data; the main ones reduce Model L to Model M through the table and require Model M's oracle results byte for byte |

```
python -m unittest discover -s clean/l/tests -t .
```

Test data is synthetic. Model data never goes into this repository.
