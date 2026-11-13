# data/: converter maps and synthetic test vectors

Every file in this folder was made by the **spec team** (the other half of the clean-room process, see
`docs/CLEAN-ROOM.md`), not by the clean team that wrote the code. None of them holds model weights. The maps hold
only layout information and the digests the converter uses to verify the player's own file. In the private repository they lived under `spec/data/`.

## Converter maps

`preset-M.v1.map.json` (Model M, map version 4), `preset-F.v1.map.json` (Ray Reconstruction, version 5) and
`preset-K.v1.map.json` (Model K, version 3) tell the converter (`clean/converter`) how to read the network's parameters
from the player's own DLSS file. Each map holds:

- `format`, `version`, `channel_order`, and a `model` line naming the DLSS product and the first DLSS version the map
  reads (e.g. "DLSS super-resolution preset M (first shipped in DLSS 310.5)");
- `sources`: for each block of the file the converter needs (parameter block, dither table, constants, output-filter
  weights): its element type, length, how it is located (`locate`, `gaps`, encoding such as `lz4-block`, optional
  variants) and its FNV-1a-64 and SHA-256 digests, so the converter can find and verify it;
- `key` (the label and source of the converter's key) and `perms` (index permutations of some channel groups);
- `stages` / `groups`: per network stage, each logical tensor's name, shape, data type, source block, the operation
  that converts it (for example to FP8), its base offset and axis strides, and a digest per stage.

They contain no weights. They do describe where in the vendor's DLL the parameters are and how they are laid out.

## Test vectors (synthetic)

| File | What | Read by |
|---|---|---|
| `06-vectors.json` + `06-vectors.md` | spec 06 (output reconstruction and filter), test 06-T5; seeded random inputs, made-up blend scale and filter weights, outputs computed with the spec's arithmetic | no code in this repository reads it today (kept as the spec team's vectors for 06-T5) |
| `motion-hi-vectors.json` + `motion-hi-vectors.md` | motion at output resolution (spec 03 / 06), tests 03-T4 and 06-T6; seeded random, made-up blend scale | `clean/tests/test_motion_hi_reference.py`, `clean/runner/motion_hi_check.py` |
| `57-T5-vectors.json` | converter map version 5 operations (page 57, test 57-T5); every value synthetic | converter unit tests (`wgc-unit-tests --t5-vectors`) |
| `k48-capped-softmax-test.json` | Model K `capped_softmax` (spec 48, test 48-T5) with fixed constants; inputs are binary16 bit patterns, one row per group of 64. How the spec team chose the inputs is not recorded in the file | converter tests (`clean/converter/tests/run_tests.py`) |
| `lz4-block-test.json` | block decompression (spec 08 section 4.4), made by the spec team from its own text | converter unit tests (`--lz4-vectors`) |
| `rr-f-synth/` | synthetic preset-F maps and expected results, see `rr-f-synth/README.md` | converter and RR model-file tests |

The two `.md` files are the spec team's own descriptions of their vector files, published unchanged with them.
