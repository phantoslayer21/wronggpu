# Test vectors for spec 06 draft 4 (`06-vectors.json`)

Made by the spec team from a seeded random generator. Nothing in the file comes from a DLL: the blend scale `s` and
the output-filter weights are made-up numbers, and the expected outputs were computed from these inputs with the
arithmetic of spec 06 draft 4. Use them for test 06-T5.

The file is JSON: `cases` is a list; each case has its parameters and `arrays`, an object of named arrays. Each array
has `shape`, `type` and `data` (a flat list in row-major order: the last index varies fastest). Types: `uint8` and
`uint16` are plain integers; `float16_bits` and `float32_bits` give each number as the unsigned integer of its
IEEE bit pattern (so the values are exact and NaN and infinity can be written); reinterpret them as binary16 or
binary32. Images are `[row][column][channel]`.

## Reconstruction cases (`case` = `recon/<sequence>/<frame>`)

Each frame is a complete case: the previous history is an input, so the frames do not have to be run in order.

Parameters of the case: `R` = [rw, rh], `D` = [dw, dh], `reset`, `e`, `e_prev` (exposure pair), `jitter` =
[jx, jy] (render pixels), `s` (blend scale), `note`.

| Array | Shape, type | Meaning (spec 06 names) |
|---|---|---|
| `color` | [rh][rw][3] float32 | game color, linear |
| `A` | [rh][rw][3] uint8 | codes of g, β_h, β_o (value = code / 255) |
| `B` | [rh][rw][3] uint8 | codes of σ_near, σ_center, σ_far |
| `C` | [rh][rw][3] float16 | kernel shape a, b, c |
| `m_near`, `m_center`, `m_far` | [rh][rw][2] float32 (FP16 values) | motions (x, y) in D pixels |
| `history_prev` | [dh][dw][3] float16 | previous full history (all zero on the reset frames; not read there) |
| `full_history` | [dh][dw][3] float16 | expected new full history |
| `reduced_history` | [⌈dh/2⌉][⌈dw/2⌉][3] uint16 | expected codes 0…1023 |
| `display` | [dh][dw][3] uint16 | expected display codes: R and G 0…2047, B 0…1023 |

Sequences, two frames each (frame 0 is a reset frame): `q-even` (R 32×18, D 72×40), `q-odd` (R 32×18, D 71×39),
`b-odd` (R 32×18, D 48×27). The note `edge` marks a frame with out-of-range and NaN motion, NaN / infinite /
negative / very large color values and scores of exactly 0 and 1; `zero` a frame whose current exposure is exactly
0; `bright` a frame with a large color range and a small exposure.

## Output-filter cases (`case` = `filter/<name>`)

Parameters of the case: `O` = [ow, oh], `D` = [dw, dh], `e`, `R` = [rw, rh], `color_origin` (subrect origin of
the game's color), `jitter`, `has_alpha`.

| Array | Shape, type | Meaning |
|---|---|---|
| `display` | [dh][dw][3] uint16 | display codes |
| `weights` | [5] float32 | w₋₂ … w₊₂ for an even output index (made up) |
| `color_alpha` | [rows][columns] float32 | alpha plane of the game's color, larger than R so that the origin matters; absent when `has_alpha` is false |
| `rgb` | [oh][ow][3] float32 | expected output color |
| `alpha` | [oh][ow] float32 | expected output alpha (all 1 when `has_alpha` is false) |

## Comparing

Spec 09, "Stages with 8-bit and FP16 outputs", counted over all cases together: codes and FP16 steps for
`full_history`, `reduced_history` and `display`; the float32 rule for `rgb`; `alpha` must be equal. A NaN is equal
only to a NaN (the expected arrays contain none).
