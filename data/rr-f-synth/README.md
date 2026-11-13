# data/rr-f-synth/: synthetic Ray Reconstruction (preset F) converter test data

Made by the spec team for the converter's map-version-5 tests (page 56.5 item 2, tests 57-T1 / T3 / T7 / T8). No
model data.

- `synth-r0.map.json` ... `synth-r5.map.json`: six synthetic maps of the preset-F layout ("SYNTHETIC test map run
  <r>"), one per test run.
- `synth-expected.json`: the expected results. Its `rule` field says how each run's synthetic input file is built: the
  test file of run r, variant v, is W_r followed by `synth-common-v<v>.bin`, where element e of W_r is the binary16
  value of ((e >> 4r) & 15); per tensor it lists the SHA-256 of the row-major little-endian bytes the converter writes.

The two shared stand-in files `synth-common-v0.bin` and `synth-common-v1.bin` are not in this repository. Tests that
need them take their folder from `WGC_SYNTH_BINS` and skip themselves without it.
