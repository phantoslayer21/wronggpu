# Test vectors for motion vectors at output resolution (`motion-hi-vectors.json`)

Made by the spec team from a seeded random generator; the blend scale is a made-up number; nothing in the file comes
from a DLL. Same JSON layout and array types as `06-vectors.json` (see `06-vectors.md`). For tests 03-T4 and 06-T6.

Common parameters of a case: `R` = [rw, rh], `M` = [Mw, Mh] (size of the motion data = output size), `D` = [dw, dh],
`motion_origin` (subrect origin inside `motion_texture`, which is larger than M in some cases), `mv_scale`, `b_mv`,
`jitter`.

## Input-encoder cases (`case` = `enc/<name>`)

| Array | Shape, type | Meaning |
|---|---|---|
| `motion_texture` | [rows][columns][2] float32 | the game's motion vectors (x, y) |
| `motion` | [rh][rw][2] float16 | expected motion surface in D pixels: the value of m_near, m_center and m_far alike |

`enc/b` has motion in screen units with a negative vertical scale, a non-zero origin and b_mv, motion far out of
range and NaN texels.

## Reconstruction cases (`case` = `recon/<name>`)

As the reconstruction cases of `06-vectors.json` (`color`, `A`, `B`, `C`, `history_prev`, `e`, `e_prev`, `jitter`,
`s`, `reset`; expected `full_history`, `reduced_history`, `display`), but without the three motion surfaces and with:

| Array | Shape, type | Meaning |
|---|---|---|
| `motion_texture` | [rows][columns][2] float32 | the game's motion vectors |
| `boundary` | [dh][dw] uint8 | 1 where the D pixel's centre lies exactly between two motion texels in x or in y (spec 06 section 3) |

`recon/even` has D = 1.5 · M exactly (more than half of its pixels are boundary pixels); `recon/odd` has none.

## Comparing

`motion`: bit for bit. Reconstruction outputs: as in `06-vectors.md`; report the pixels marked in `boundary`
separately. An implementation that evaluates the texel rule of spec 06 section 3 exactly reproduces them too.
