# Fast resolve (page 54b): design and exactness argument

`kernels/rr_resolve_fast.hlsl` (`rr_resolve_fast`, `rr_resolve_fixup`, `rr_resolve_pool_fast`, `rr_rf_init_table`)
computes the extended resolve and the quarter pooled history of `kernels/rr_resolve.hlsl` (`rr_resolve_main`,
`rr_resolve_pool`, the verified kernels) with the same bits for every input.
Host / harness: `gpu/resolve_fast.py`; frame sequence: `gpu/sequence.py` (default `--resolve fast`).

## Structure

* **Dispatches.** `rr_rf_init_table` once per arena (the exp2 table, below); per frame `rr_resolve_fast`
  (persistent workgroups, `min(tiles, 512)` groups of 256 lanes) then `rr_resolve_fixup`
  (`ceil(ceil(We * He / 32) / 256)` groups). Parameter block as `rr_resolve_main` plus word 37 (fixup bitmap, We * He
  bits, zero between frames: the fixup clears what it reads) and word 38 (exp2 table, 8 KB).
* **Tiles.** 16 x 16 output pixels; per tile four phases, each closed by a barrier:
  1. the binary16 Q footprint of the tile (at most 29 x 29 texels) into groupshared;
  2. the horizontal 5-tap pass of 54b.1 once per (Q row, output column) (`rf_hr`), two columns per lane with packed
     binary16 (`v_pk_fma_f16`, `v_pk_min/max_num_f16`); the vertical pass per pixel reads the shared rows;
  3. the render footprint (taps -1..2 and the 3x3 neighbourhoods; at most 20 x 20 texels): per texel `qx, qy, qxy`,
     the alpha value, the G halves and `E * layer` (`rf_u4`, `rf_g`);
  4. per pixel: vertical pass + decode, alpha upscaling (16 taps in two batches of 8), transparency layer, stores.
  `RF_SPEC=1` (used by the sequence) is the standard configuration's specialisation: extended + layer + alpha
  upscaling, binary16 P / L / MV / TH / OUT, unorm8 G, RGB10A2 alpha source, f32 output alpha; `rf_uniforms` checks it
  and otherwise sends every pixel to the fixup. Its phases 1 and 3 take the words of the tile's footprints from
  registers loaded during the previous tile's phase 4 (`rf_prefetch`). `RF_SPEC=0` reads every format.
* **Fallback.** A pixel the fast path cannot prove (below) sets its bit in the bitmap; `rr_resolve_fixup` recomputes
  it with `rr_resolve_px`, the verified kernel's own pixel function (factored out of `rr_resolve_main`, otherwise
  unchanged), built with `RR_ELEM_EXACT=2`: exp2 and rcp through `rr_x_hook` / `rr_rcp_hook`, which equal `EXP2_CR` /
  `RCP` on every input (exhaustive). A dispatch that fails the per-dispatch screen sends all pixels. Each wave of the
  fixup reads 32 bitmap words and walks its nonzero words one after another with all 32 lanes on the 32 pixels of one
  word (a lane per pixel; wave32), so clustered fallback pixels run in parallel, not one after another per lane.

## Why the bits are the same

The verified kernel's f32 helpers (`rr_resolve_math.hlsli`) are hardware operations (one IEEE rounding, FTZ inputs and
results) plus one repair: `M` and `F` recompute in binary64 when the hardware result is at or below FMINN, because the
hardware flushes an exact result in `[FMINN - 2^-150, FMINN)` to 0 where 54b.N rounds it to FMINN. `A` needs no repair
(an exact sum below FMINN is an f32 subnormal: both flush it). So a plain hardware product or fma equals `M` / `F`
exactly when its exact result is 0 or at least FMINN in magnitude; every result that is not 0 also has hardware
result not 0 unless it is in the flushed range. The fast path uses plain operations under these rules:

| Operation | Why the exact result is 0 or >= FMINN (or how it is checked) |
|---|---|
| Per-dispatch values (`sxr`, `sxh`, `jxp`, `rWe`, `fac`, `rE`, `TBW^2`, ...) | computed with the verified helpers (`DIV`, `RCP`, `M`, `A`), one quotient per lane |
| Screen (`rf_uniforms`) | every per-dispatch factor (K1, K2n, K2x2n, K5, K8, K10, K12, E, rE, fac, TBW, TBW^2, 1 - w, jitter, jxp, jyp, sxr, syr, sxh, syh, rWe, rHe) is 0 or >= 2^-24 in magnitude, jitter finite, `|K2n|, |K2x2n| <= 2^100`; K13 only 0 or normal (it only enters `rW = RCP(max(W, K13))`, RCP proven for every input, and rW only the zero-checked products `Gs rW`, `Aw rW`; the game model F-B1 has K13 = 1e-10, which the old 2^-24 screen sent whole to the fixup: 28 ms instead of 1 ms at 1440p Q); otherwise the whole dispatch takes the fixup |
| Decode `cp * K8`, `* K9`, `(2^t - 1) * K12` | binary16 or K5 / K10 values (>= 2^-24) times screened factors: >= 2^-48; `2^t - 1` is 0 or >= 2^-24 |
| Bilinear weights (`u*W - 0.5`, fractions, `g * t`, `(1+s) t`) | coordinates are sums of multiples of 2^-47 (screen) times `rW*` (>= 2^-24): >= 2^-71; the fractions are 0 or >= 2^-25; binary16 / unorm texels give partial results >= 2^-74, f32 texels are screened per texel (0 or >= 2^-24, `tiny`) |
| `E * cur`, `Hp * fac`, `mv * sxh`, `(x + m + 0.5) * rWe` | >= 2^-98 (binary16) / 2^-121 (screened f32) |
| Layer blend `T = fma(w, cur, w1 * hist)` | `w1 * hist` >= 2^-122 for binary16 layers (`RF_SPEC`); a zero T from a nonzero addend is checked (`slow`) |
| Colour blend `fma(col, 1 - Ts.w, Ts)` | `col * (1 - Ts.w)` >= 2^-72; a zero result with nonzero Ts is checked |
| Tap exponent `e = fma(qxy, dx dy, fma(qx, dx^2, qy dy^2))` | `|d| >= 2^-25` if nonzero, q >= 2^-48 (loader screen), so every product >= 2^-98; `qx`, `qy` of one sign (loader screen: no cancellation in the inner fma); the outer fma's zero from a nonzero addend is checked |
| Tap weights `wt` | 0 or >= 2^-126 (exp2 result); alpha >= 0 (format / loader screen) and g >= 0, 0 or >= 2^-24, so `Aw`, `Gs` sum non-negative terms. A batch of 8 taps without a weight in (0, 2^-100) has every product 0 or >= 2^-124: plain fma. Otherwise the wave accumulates that batch with `rf_accx`, which equals `F` for these operand classes (the exact result is below FMINN only for acc = 0 and w v < FMINN; that case is decided exactly from the scaled product and its fma residual; `resolve_fast.py accx`: every positive binary16 v and the RGB10A2 alpha values, every w within 2^14 steps of FMINN / v). Before this, such pixels went to the fixup: with precision values above about 10 (K2n = -1.25) most pixels did |
| `W K1` | W >= 2^-126 when nonzero: checked (a zero result with nonzero operands takes the fixup) |
| `nq = Gs rW`, `nq.x (W K1)`, `a_up = Aw rW`, `(1 - b) hc`, beta, gamma, ah, oa, `rE col` | checked: a zero result with nonzero operands / addend takes the fixup |

**Elementary functions and conversions** (`rr_resolve_fast_math.hlsli`), each compared with the verified helper on all
2^32 f32 inputs on the GPU (`resolve_fast.py exhaustive`):

* `rf_exp2` = `EXP2_CR` or "undecided". First tier: `x = n + j/1024 + r`, `2^x = 2^n T[j] (1 + p)`,
  `p = r (P1 + r (P2 + r P3))` in one f32 word, `S = Th p + Tl`; if `Th + (S - 2^-33)` and `Th + (S + 2^-33)` round to the
  same f32 that is the result (the first tier's error is below 2^-33.2). Second tier (`rf_exp2_core`): `T (1 + s + s^2/2
  + s^3/6 + s^4/24)` with `s = r ln2` and the main sums carried as exact two-word values; undecided when the exact rounding
  error of the final sum is within 2^-43 of half an ulp. 1314 of the 2^32 inputs remain undecided; for them the kernel
  calls `EXP2_CR` itself (one call site per tap batch and one for the decode).
* `rf_rcp` (hardware reciprocal + one fused Newton step; `|x| >= 2^126` decided directly) = `RCP` on all inputs.
* `rf_f2h` (driver channel op 0x1C, `v_cvt_f16_f32`) = `wg_f2h` on all inputs.
* `rf_unorm_code` (rounded product + exact fma residual) = `unorm_code` for 255, 1023 and 3 on all inputs.
* `rf_cv` = `cv` (IEEE division) for every code of the three formats (1284 codes).
* `rr_x_hook` = `EXP2_CR` on all inputs (no undecided input).
* The exp2 table `T[j] = 2^(j/1024)` is computed by `rr_rf_init_table` from the verified double-double exp2 and its
  correct rounding; it equals `gen_rf_tables.table()` (exact decimals) on all 2048 words (`resolve_fast.py probes`).

**Hardware binary16 and min / max** (`resolve_fast.py probes`): HLSL half min / max and packed half2 min / max equal
`pmin` / `pmax` on all 2^32 binary16 pairs; f32 min / max equal `pmin` / `pmax` on 199,084 special and random pairs
without a denormal operand (the fast path has no f32 denormal values); `mad` on half / half2 equals `wg_fma_f16` on
400,000 triples (20,000 from the special classes, 40,000 cancellation cases).

**Evidence for the whole kernel**: `resolve_fast.py diff` (30 configurations x 3 seeds: formats, origins, resets,
specials, tiny f32 values, extreme E and jitter, render scales 1/3 to 1, bounded stores, without motion vectors;
7,303,668 output values, bit-identical, for `RF_SPEC` 1 and 0 and for the opt-in hardware variant against
`rr_resolve_main` e0); negative controls `--mutation 1..3` each fail; the 10-frame sequences with both model variants
and the synthetic model (`gpu/sequence.py`, resolve compared with the CPU reference).

## Quarter pooled history (54b.5)

`rr_resolve_pool_fast` runs `rr_resolve_pool`'s operations with the verified f32 helpers (`M`, `F`, `A`, `pmax`), the
exactly equal fast exp2 (`rr_x_hook`, groupshared table) / rcp / unorm reads (`rf_cv`) and stores (`rf_unorm_code`).
Its log2 only has to give the same RGB10A2 code: the hardware log2 (`v_log_f32` on the flushed input) is within one
f32 step of `LOG2_CR` for every input with a finite result and equal on all others (`exhaustive hwlog2`, all 2^32
inputs). The code is `rint_even(clamp(v) * 1023)` with `v = ((L * K6) * K7) * K11`; one step in `L` becomes at most
2 (2 (2 + 1) + 1) + 1 = 23 steps in `v` (each rounded product at most doubles the distance in steps and adds one), so
the code from the hardware log2 is the exact code unless `clamp(v) * 1023` lies within 48 steps of `v` (in code units,
plus the rounding of the product) of a half-integer; then `LOG2_CR` decides. Checked against the CPU reference
(`resolve_fast.py pool`: small fixtures, packed and code layouts, edge codes) and against `rr_resolve_pool` at the
1080p, 1440p and 4K output sizes, and in every frame of the sequences (`ph2_pool` stage, bit-exact).

## Opt-in variant

`RF_HW=1` (`sequence.py --fast-resolve`): the hardware exp2 of `rr_resolve_main` built with `RR_ELEM_EXACT=0`; the
fixup is that build's pixel code. Bit-identical to the old opt-in kernel (`diff --hw`), not to the reference.

## Measurements and limits

Per-phase shader cycles (`resolve_fast.py timing`, 1440p Q): alpha upscaling about 35 %, render footprint 16 %,
layer 15 %, stores and barrier 14 %, plain colour 7 %, Q footprint and horizontal pass 10 %. The kernel issues about
2,400 VALU instructions per output pixel (roughly 860 alpha taps, 480 layer, 240 render footprint, 150 Q, 90 decode),
at about 75 % VALU utilisation, with about 250 MB of memory traffic per 1440p frame (Q at 1.5x resolution alone 50-70
MB). Limits: 32 KB groupshared (D3D12) holds 4 workgroups per WGP (8 waves per SIMD); the 1314-input exp2 tail costs
VGPRs (171) but no occupancy at this LDS size.
