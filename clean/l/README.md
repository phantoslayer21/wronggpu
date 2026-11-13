# Model L (issue #89)

Work in progress for a new model choice, milestone 0.7.0. Nothing here is used by the runtime, the front end or the
launcher yet.

| Path | Content |
|---|---|
| `reference/table.py` | Model L's structure table: the encoder's op list, the decoder's channel segments and block, window offsets, block constants, map names; parameter shapes from the table; matrix work per frame (computed, not measured). Entries in `PROVISIONAL` are pending confirmation and are changed here only |
| `reference/oracle.py` | NumPy oracle (float64, only the prescribed FP8 / FP16 roundings): `encode`, `decode`, `network`, synthetic `random_params`. The Swin stages it shares with Model M call Model M's oracle (`clean/reference/swin.py`); the output reconstruction is not written yet |
| `tests/test_l_reference.py` | self-checks on synthetic data; the main ones reduce Model L to Model M through the table and require Model M's oracle results byte for byte |

```
python -m unittest discover -s clean/l/tests -t .
```

Test data is synthetic. Model data never goes into this repository.
