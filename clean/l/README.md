# Model L (issue #89)

Work in progress for a new model choice, milestone 0.7.0. Nothing here is used by the runtime, the front end or the
launcher yet.

| Path | Content |
|---|---|
| `reference/table.py` | Model L's structure table: the encoder's op list, the decoder's channel segments and block, window offsets, block constants, map names; parameter shapes from the table; matrix work per frame (computed, not measured). Entries in `PROVISIONAL` are pending confirmation and are changed here only |
| `reference/oracle.py` | NumPy oracle (float64, only the prescribed FP8 / FP16 roundings): `encode`, `decode`, `network`, synthetic `random_params`. The Swin stages it shares with Model M call Model M's oracle (`clean/reference/swin.py`); the output reconstruction is not written yet |
| `runner/l_pack.py` | weight packing of Model L's own launches: the encoder's launch list derived from the table (`encoder_launches`) and one blob per launch (`pack_model_l`), each in a format Model M already uses (merge / expand section, Swin block section, enc0 and dec0 blobs); decoders for tests. The ten shared stages use `clean/runner/packing.py` unchanged |
| `kernels/l_swin.hip` | the C = 32 Swin block kernels (`l_swin_block_c32[_nc]`) from Model M's block template (`clean/kernels/swin.hip`, included for its templates only), same blob format and launch geometry as Model M's block kernels |
| `runner/l_c32_check.py` | GPU check (synthetic): the C = 32 kernel against the oracle, `_nc` against the clamping kernel, and this file's C = 64 build of the template against Model M's `swin_block_c64` |
| `tests/test_l_pack.py` | packing round trips, the block packer against Model M's stage blobs, and the unpacked weights applied the kernels' way against the oracle's maps, byte for byte |
| `tests/test_l_reference.py` | self-checks on synthetic data; the main ones reduce Model L to Model M through the table and require Model M's oracle results byte for byte |

```
python -m unittest discover -s clean/l/tests -t .
```

Test data is synthetic. Model data never goes into this repository.
