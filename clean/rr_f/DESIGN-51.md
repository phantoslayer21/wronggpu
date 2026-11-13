# DESIGN-51 - the window block of spec 51 on RDNA 4 / D3D12 (kernel design, no kernel code yet)

Status: design 1 (clean team, branch clean/rr-f). Source: spec 51 draft 6 only (pages 50, 52-57 are not available here;
where they matter this design says so). Conventions and reused code: the Model M D3D12 ports in `clean/dx12/`
(README.md there, `hlsl/wg_platform.hlsli`, `hlsl/wg_ags.hlsli`, `kernels/swin_block.hlsl`, `kernels/swin_c64w2.hlsl`,
`kernels/swin_common.hlsli`). Open points of the spec: `QUESTIONS-51.md` (provisional readings marked **[Qn]** below).
Numbers of section 10 come from `cost_model_51.py` in this folder.

## 1. Summary

* **One workgroup = one 8 x 8 window, 4 waves of 32 lanes (128 lanes).** Wave w owns token tile w = tokens
  16w .. 16w+15 = window rows 2w, 2w+1. Everything token-wise (projections, MLP, residuals, hidden state, merge
  children, fusion parents) stays inside the wave; only the attention needs the other tiles' q and v, which go through
  LDS (4 KB per head, double-buffered). Same ingredients as the M Swin port: weights as the A operand, tokens as the B
  operand, staged weight units in LDS (M's Stager), the driver channel `wg_wmma_fp8` (v_wmma_f32_16x16x16_fp8_fp8).
* **Orientation that needs no shuffles:** every WMMA result tile (lane = column r = l & 15, rows 8(l >> 4) .. +7) is,
  after rounding / packing, directly the B fragment of the next product (q, O, y1q, Au, y2) or the A fragment of the
  attention products (q as keys, v computed transposed by swapping operands, as M does). The score tile has keys as
  rows and queries as columns, so a lane holds 8 keys of one query: 32 keys per lane, the row sum needs one lane
  exchange (lane ^ 16), and the P tile is directly the B operand of P V.
* **Spec 51.1's matrix step on 16-deep WMMA:** one 32-input step = two WMMAs (K-slices 2s, 2s+1) on an f32
  accumulator whose start value is the exact f32 widening of the f16 start value; after the pair: `v_cvt_f16_f32`
  (RNE) and, if the chain continues, `v_cvt_f32_f16` back. Activations are stored with the channels of every 32-block
  permuted (layout pi32, section 3) so that a lane's 16 bytes of one step are one load and every step covers exactly
  inputs 32s .. 32s+31.
* **The step rounding is the main cost:** two VALU instructions per value per step, about as many issue cycles as the
  WMMA pair itself. A spec-acknowledged variant L (f32 over the whole chain, one rounding) is designed as opt-in.
* **Attention weight in packed f16 plus two integer instructions:** `u = s * A_SCALE`, clamp, two `v_pk_fma_f16`;
  then w is exactly `(bits(p) << 5) & 0x7FE0` per half (section 8.2); no transcendental, no rounding.
* **Exact pieces by construction:** sigmoid by a 64 K-entry f16 table generated from the reference evaluation;
  `r = R(1/Z)` by `v_rcp_f32` + RNE conversion (argued exact, verified exhaustively); every f16 rounding point a single
  hardware rounding (`v_pk_fma_f16`, `v_pk_mul_f16`, `v_cvt_f16_f32`, `v_fma_mixlo_f16`).
* **Bit-exactness risk:** the f32 accumulation inside / between the two WMMAs of a step (exactness unknown, probe P1);
  Z's summation order (accepted by the spec); NaN code sign inside the network **[Q2]**. Section 9.
* **Estimate at 1440p Quality (render 1706 x 960, level-0 tokens 1728 x 960):** about **5.3 ms** for the 16 window
  blocks with their tails (variant L: 4.3 ms), half of it at level 0 (IN0, D0-F, D0-R: 2.6 ms), +-40 %. Section 10.

## 2. What is reused from the Model M D3D12 kernels

| piece | from | use here |
|---|---|---|
| WMMA / conversions through the driver channel | `wg_ags.hlsli`, `wg_platform.hlsli` (`wg_wmma_fp8`, `wg_pack_q8_nc_ags`, `wg_to_f16_ags`, `sw_unpack_fp8`) | every product; FP8 packs; F32 -> F16 |
| fragment layouts | probe arith3 (M kernels) | section 4 |
| Stager (global -> VGPR prefetch -> LDS commit, two buffers, one barrier per unit) | `swin_block.hlsl` `st_fetch` / `st_commit` | weight units |
| LDS helpers, rolled loops (`sw_zero`, `sw_zero_bits`), 64-bit LDS elements | `swin_common.hlsli` | as is |
| FP8 pack with NaN kept (`v_maximumminimum_f32` clamp) | `sw_pack_q8`, `wg_sat448` | Q8 |
| single-rounding f16 helpers | `wg_mul_f16`, `wg_fma_f16` (v_fma_mixlo_f16) | unpaired f16 operations |
| opaque-zero fences against driver regrouping / folding | `post.hlsl opq()`, `sw_zero` | every f16 rounding point that feeds an add or a conversion |
| build (`@wg-kernel`, `vgpr_limits.json`, RGA `--isa`), GPU lock, harness patterns | `build_dx12.py`, `tests/tuned_swin_diff.cpp`, `matched_stage_bench` | kernels and tests of the next tasks |
| 128-lane workgroups (4 waves, CU mode: fewer waves per barrier was faster) | `swin_c64w2.hlsl` findings | workgroup shape |

Not reused: RMSNorm, GELU tables, exp2 softmax (spec 51 has none of them); M's 4 x 4 window gather (`sw_mirror` is the
single reflection; spec 51.6 needs the repeated one for small extents, section 6).

## 3. Data layouts (runtime's own buffers, spec 51.2: "how a runtime arranges them is its own choice")

* **Activation tensors** (e4m3, one byte per channel): token-major `[Y][X][C]`, channels of every 32-block in the order
  **pi32** = (0-7, 16-23, 8-15, 24-31). Lane half hf of a token reads bytes `32s + 16 hf .. +15` = channels
  `32s + 8hf .. +7` (K-slice 2s) and `32s + 16 + 8hf .. +7` (K-slice 2s+1): one 16-byte load per step, and the two
  WMMA result tiles 2s, 2s+1 of a token store back as one 16-byte write. Concatenations (merge input z, E1's 128-channel
  input) are whole 32-blocks per child because every C is a multiple of 32.
* **Weights** (e4m3): pre-arranged once at load time (by our loader, not the converter) into 256-byte fragments
  (32 lanes x 8 bytes), one per (16-row output tile, 16-deep K-slice), grouped into the units of section 7.3.
* **Score bias**: the 12 stored blocks `Tb_h` are expanded at load time to all 16 (I, J) blocks (transposing the 4
  missing ones, 51.3 step 2) in C-fragment order, widened exactly to **f32**: `[H][I][J][lane 32][8 f32]` = 16 KB per
  head (128 KB at H = 8), L2-resident, read straight from memory (variant: f16, 8 KB per head, plus one conversion per
  value). The slot order of the 12 blocks in `Tb` is provisional **[Q3]**.
* **Hidden state** (51.4): 4 f16 per token (8 bytes), token-major at the level of the stage: `hnew_k` of IN0 / E1-E4
  (own level, before the merge; read by Dk-R as `hold`), merged hidden (level k+1, read by E(k+1) as `hold`),
  `hnew_5` of E5 (coarse hidden of D4-R), `hnew_k` of Dk-R (coarse hidden of D(k-1)-R). Nothing persists across frames.
* **Sigmoid table**: 65 536 f16 indexed by the f16 bits of x, values = the reference evaluation of 51.4 (computed on
  the host with a correctly rounded exp2 and division); 128 KB, L2-resident.
* **Constants** (ra, m, f, g, A_*, ACT_*, shifts): root constants / constant buffer per stage call.

## 4. Mapping of the products onto v_wmma_f32_16x16x16_fp8_fp8

Fragment layouts (wave32, as the M kernels): lane l, r = l & 15, hf = l >> 4.
A (16 rows x 16 K): row r, K elements 8hf .. 8hf+7 (8 bytes). B (16 K x 16 columns): column r, K 8hf .. 8hf+7.
C / D (16 x 16 f32): column r, rows 8hf .. 8hf+7 (8 VGPRs).

| product (spec) | A operand | B operand | D tile (rows x cols) | K, steps | start value |
|---|---|---|---|---|---|
| q_h (step 1) | Wq_h^T, 2 tiles of 16 features (LDS) | x of the wave's 16 tokens (VGPR, from global) | features x tokens | C, C/32 | 0 |
| v_h (step 1), **transposed** | x of the wave's tokens (same registers as B: identical layout) | Wv_h, 2 feature tiles (LDS) | tokens x features | C, C/32 | 0 |
| s_h (step 2) | q_h of key tile J (LDS exchange; the q D-tile layout is the A layout) | q_h of the wave's 16 queries (VGPR) | keys x queries | 32, 1 | B_h (f32 fragment) |
| O_h (step 5) | v_h^T of key tiles (LDS exchange; the transposed v tile is the A layout) | P (VGPR, = the score tiles after step 4) | features x queries | 64, 2 (keys 0-31, 32-63) | 0 |
| Dp (step 6) | Wo_h^T, C/16 tiles (LDS) | O_h (VGPR) | channels x tokens | 32 per head, chain h = 0 .. H-1 | bo |
| Du_j (step 8) | W1[:, 32j ..]^T, 2 tiles (LDS) | y1q (VGPR) | hidden x tokens | C, C/32 | b1 chunk |
| Dd (step 8) | W2[32j .., :]^T, C/16 tiles (LDS) | Au_j (VGPR) | channels x tokens | 32 per j, chain j = 0 .. C/8-1 | b2 |
| merged (51.4) | Wm^T, Cn/16 tiles split over the 4 waves | z of the window's 16 coarse tokens (LDS gather of y2) | out x coarse tokens | 4C, C/8 | bm |
| Dup (51.5) | Wup^T, 4C/16 tiles split over the 4 waves | cin of the window's 4 x 4 coarse tokens (global, reflected) | up-channels x coarse tokens | Kc, Kc/32 | bup |
| x of E1 (51.5) | Wa^T, 4 tiles | in (128 channels, VGPR) | channels x tokens | 128, 4 | ba |
| Dh (51.4) | Wh^T padded to one 16-row tile (rows 3 / 4 .. 15 zero) | y2 (VGPR, already the e4m3 fragments) | 16 x tokens | C, C/32 | bh (padded) |

Per head and wave the q and v tiles are 2 x 2 tiles of 16 x 16; the score phase is 4 key tiles x 2 WMMAs; O is
2 feature tiles x 4 WMMAs. Every chain's two WMMAs of a step use K-slices 2s and 2s+1 of the same 32 inputs (pi32).

## 5. Common core, step by step (51.3), per workgroup

```
load   x tile (C/32 x 16 B per lane, reflected token coords, 6) ; Stager fetch unit QV_0
for h in 0 .. H-1:
  [A] q_h, v_h^T (unit QV_h in LDS buffer a): 4 tiles x C/32 steps, round per step, Q8
      write q tile (2 frags) and v^T tile (2 frags) to exchange[h & 1]   (1 KB per wave)
      commit unit Wo_h into buffer b ; fetch unit QV_{h+1} (or MLP_0)
      barrier
  [B] for J in 0..3: s = WMMA x2 (A = q of tile J from exchange, B = own q, C = bias f32 (I = w, J))
      -> f16 -> attention weight w (8.2) ; Z = sum (v_dot2_f32_f16, lane ^ 16), r = R(1/Z)
      P = Q8(R(w r)) ; O_h = Q8(two steps of P V^T)   (A = v^T frags from exchange)
      Dp step h: widen Dp (f16 -> f32) or bo, 2 WMMAs per channel tile (unit Wo_h, buffer b), round
      commit unit QV_{h+1} into buffer a ; fetch Wo_{h+1}
      barrier
y1h = fma16(Dp, ra, xr) ; y1q = Q8(y1h)                 xr = d8(x), or xf for a fusion half (51.5)
for j in 0 .. 4C/32-1:                                   (one unit W1_j | W2_j per j, one barrier each)
  Du_j: 2 tiles x C/32 steps ; Au_j = Q8(act(Du_j)) ; Dd step j: C/16 tiles x 2 WMMAs, round
y2h = fma16(Dd, m, y1h) ; y2 = Q8(y2h) ; tails (7)
```

Two barriers per head (q/v exchange, Wo unit) and one per MLP chunk: C = 160: 16 + 20. The exchange is double-buffered
so a fast wave writing head h+1 never overwrites what a slow wave still reads of head h. Barriers wait for 4 waves
(M's measurement: 4-wave workgroups beat 8-wave ones on this driver).

Registers that live across the head loop: x (C/16 x 2 dwords), Dp kept as packed f16 between steps (C/16 x 4), Stager
prefetch. Across the MLP: y1q (C/16 x 2), y1h packed f16 (C/16 x 4), Dd packed f16 (C/16 x 4). xr = d8(x) is recomputed
from x when needed (no extra registers); a fusion half keeps xf packed f16 (C/16 x 4) instead.

## 6. Borders (51.6): one rule per read, at the read's level

`rr_reflect(v, N)`: N = 1 -> 0; else P = 2(N-1), u = v mod P (non-negative), v' = min(u, P - u). Integer remainder by
the reciprocal method (exact for these small integers), evaluated once per token and read, never per channel. For
extents >= 8 it equals the single reflection of M's `sw_mirror`; small extents (render sizes below 225 pixels) need the
general form, so the general form is the only one (cost: a few instructions per token).

| read | level, extent | coordinates |
|---|---|---|
| stage input x, own hidden `hold` | own (N) | X = 8i - sx + lx, Y = 8j - sy + ly, reflected per axis |
| E1 input (128 channels) | level 1 | level-1 coordinate reflected, the token's 128 channels as stored (children in stored order) |
| fusion skip | own (N) | as x |
| fusion cin (4 x 4 coarse tokens) | coarse (M) | `floor(X/2)` per axis, reflected with M; child index n from the unreflected fine X, Y **[Q5]** |
| recurrent-half coarse hidden Hc | coarse (M) | `xm = (X - 1) >> 1` (arithmetic), `xp = xm + 1`, each reflected with M; weights 3/4 / 1/4 by parity of X |

Writes: only tokens with 0 <= X < N, 0 <= Y < N_y (the store offset is moved out of the buffer otherwise, as M);
merged / coarse outputs only for coarse tokens inside the coarse tensor. Duplicated tokens take part in attention
unmasked (no special case: the reflected coordinates simply repeat).

Windows: the dispatch is `ceil((Nx + sx)/8) x ceil((Ny + sy)/8)` workgroups, linearised as M (grids above 65535 split by
the platform). Shift (sx, sy) per call from SH (parity block), recurrent halves `(s + 4) mod 8`; shifts are even, so a
window's 2 x 2 groups and its 4 x 4 coarse tokens are aligned with the token tiles (coarse row of tile w = w).

## 7. Matrix steps, rounding points and the tails

### 7.1 Every rounding point of spec 51 and its instruction

| spec | value | implementation (one hardware rounding each) |
|---|---|---|
| 51.1 step | `R(C + sum of 32 products)` | 2 x WMMA on f32 C, `v_cvt_f16_f32` (RNE); chain continues: `v_cvt_f32_f16` (exact) |
| 51.1 Q8 | e4m3 of an f16 h | `v_cvt_f32_f16` (exact), `v_maximumminimum_f32(h, -448, 448)` (NaN kept), `v_cvt_pk_fp8_f32` (driver conversion, RNE, two values) |
| d8 | f16 of a code | `v_cvt_pk_f32_fp8` then `v_cvt_f16_f32` (exact) or the bit form `(b & 0x7f) << 7 | sign`, times 2^8 in f16 (exact incl. subnormals) |
| step 2 | s | 51.1 step with C = bias |
| step 3 | u = R(s A_SCALE) | `v_pk_mul_f16` |
| step 3 | c = clamp | `v_pk_max_num_f16`, `v_pk_min_num_f16` (a NaN u gives A_CLAMPn, 51.1) |
| step 3 | R(A_P1 - c c) | `v_pk_fma_f16(-c, c, A_P1)` |
| step 3 | p | `v_pk_fma_f16(c, inner, A_P0)` |
| step 3 | w | integer, exact (8.2) |
| step 4 | Z | f32 sum (`v_dot2_f32_f16` with (1, 1), lane ^ 16 exchange, add), `v_cvt_f16_f32` (accepted order, 8.3) |
| step 4 | r = R(1/Z) | `v_rcp_f32` + `v_cvt_f16_f32` (exact, 8.3) |
| step 4 | P | `v_pk_mul_f16(w, r)`, then Q8 without the clamp (P <= 1.0005) |
| step 5, 6 | O, Dp | 51.1 steps; O then Q8; Dp chain over heads, start bo |
| step 7 | y1h | `v_pk_fma_f16(Dp, ra, xr)`; y1q = Q8 |
| step 8 | Du_j, act | 51.1 steps (start b1); `k`: pk max / min; `u = v_pk_fma_f16(-|k|, C1, C2)` (|k| by `and 0x7fff7fff`); `v = v_pk_fma_f16(k, u, C3)`; `act = v_pk_mul_f16(x, v)`, then an opaque fence, then Q8 |
| step 8, 9 | Dd, y2h, y2 | 51.1 steps (start b2); `v_pk_fma_f16(Dd, m, y1h)`; Q8 |
| 51.4 | merged | 51.1 steps over K = 4C (start bm), Q8 |
| 51.4 | sig | table (exact reference) |
| 51.4 | G, hnew (encoder) | `v_fma_mixlo_f16`-based `wg_mul_f16` (or packed pk_mul): `R(sig g)`, `R(hold G)`; hnew_3 = 0 |
| 51.4 | merged hidden | `q_n = v_pk_mul_f16(h, 0.25)` (= R(h/4)), fence, `add16(add16(q0, q1), add16(q2, q3))` |
| 51.4 | decoder update | `omz = R(1 - z)`; bilinear in the stated order (`R(A wx+)`, `R(. wy-)`, then fma16 per further term); `gate_j`, `hh_j = R(R(u gate) z)`, `hnew = fma16(hold, omz, hh)`, `hnew_3 = R(hold_3 omz)` |
| 51.5 | x of E1 | 51.1 steps (start ba), Q8; xr = d8(x) |
| 51.5 | upq, xf, x | 51.1 steps (start bup), Q8; `xf = v_pk_fma_f16(d8(skip), f, d8(upq))`; x = Q8(xf); xr = xf |

f16 denormals are kept by the D3D12 shaders (`wg_math.hlsli`); FP32 denormals cannot occur in any of these chains
(every nonzero partial sum is a multiple of 2^-24, far above 2^-126; the table replaces the sigmoid's f32 evaluation).

**Driver hazards** (README "Driver exceptions"): the compiler fuses an f16 multiply with a following add into
`v_fma_mix` and folds an operation into a following conversion. Rounding points where an f16 product feeds an addition
or a widening (act -> Q8, R(h/4) -> add, R(Dc wx+) -> fma addend, R(u gate) -> R(. z)) get an opaque fence
(`asuint(x) ^ sw_zero()`); every kernel's RGA ISA is checked for the expected count of `v_pk_mul_f16` / `v_pk_fma_f16`
/ `v_cvt_f16_f32` and for the absence of `v_fma_mix*` (probe P7).

### 7.2 Exactness of the matrix step on the hardware

The spec step sums 32 products and the f16 start value **exactly** and rounds once. The hardware gives f32 results of
16-deep sums; the step is `RN16(f32 chain of two WMMAs)`. Products of two e4m3 values are exact in f32 (8-bit
significands, magnitudes 2^-18 .. 2^17.6), the start value is an exact f32, so the f32 result is exact whenever the
33 terms fit in a 24-bit window; otherwise it carries an f32 rounding, which changes the f16 result only when the exact
sum lies within about 2^-24 (relative) of an f16 rounding midpoint, i.e. for roughly 2^-12 of the inexact steps, and a
one-step f16 change reaches an e4m3 code only near an e4m3 midpoint (about 2^-7 of those). This assumes each WMMA
sums exactly and rounds once (hypothesis H-exact); if the hardware rounds per addition or aligns-and-truncates, the
rate is higher. **Probe P1** measures it (adversarial inputs: 2^8-sized products with 2^-18-sized ones, cancellations,
values placed exactly at and next to f16 midpoints).

Variants, one source with defines:
* **E (default)**: the step as above. Expected result of test 51.8.1: identical except rare one-step differences,
  which is what spec 51.1's last sentence anticipates for real matrix hardware **[Q1]**.
* **X (certified exact, test builds)**: after each step the exact residual `d = D - widen(R16(D))` (exact in f32) is
  compared with half an f16 step; when it is within the error bound of P1's measured model (a function of D and of the
  first WMMA's partial, both in registers) the lane is marked, and `WaveActiveAnyTrue` (wave-uniform) runs a VALU
  recomputation of that tile's step with 64-bit integer sums (products and start value scaled by 2^24). About 6
  extra VALU per value per step on the fast path; the slow path is expected in << 1 % of tiles under H-exact.
* **L (long chain, opt-in speed variant)**: f32 accumulation over the whole K (and over all heads / all MLP chunks for
  Dp / Dd), one f16 rounding at the end. Spec 51.1 names this as differing "by an occasional one-step difference";
  whether it is acceptable is a spec decision **[Q1]**. Saves about 1.0 ms at 1440p Q (section 10).

Order inside a step: the 32 products of a step are split between two WMMAs by K-slice (pi32), the start value enters
the first; with an exact sum the split is irrelevant, under P1's error model it is part of the measured behaviour.
Order of steps: ascending k, ascending h (Dp), ascending j (Dd), keys 0-31 then 32-63 (O), as the spec states.

### 7.3 Weight units, LDS layout and staging

Units (fragment-ordered, contiguous in global memory, fetched to VGPRs during the previous phase and committed after
its barrier, M's Stager): `QV_h` = Wq_h | Wv_h (64C bytes), `Wo_h` (32C bytes), `MLP_j` = W1 chunk j | W2 chunk j
(64C bytes), tails: `M_u` (merge, 64 x Cn rows per unit), `UP_u` (up-projection), `A` (E1 adapter, 8 KB).

| C (stages) | unit (max) | weight buffers | q / v exchange | other | LDS per WG | WG per CU (LDS, 64 KB) |
|---|---|---|---|---|---|---|
| 32 (IN0, D0-F, D0-R) | - (all 14 KB of weights read from global: they stay in the vector cache) | 0 | 2 x 4 KB | D0-F: upq 2 KB | 8-10 KB | 6 |
| 64 (E1, D1-F/R) | 4 KB (QV_h, Wo_h, half MLP chunks) | 2 x 4 KB | 2 x 4 KB | merge / upq overlay | 16 KB | 4 |
| 96 (E2, D2-F/R) | 6 KB | 2 x 6 KB | 2 x 4 KB | overlay | 20 KB | 3 |
| 128 (E3, D3-F/R) | 8 KB | 2 x 8 KB | 2 x 4 KB | overlay | 24 KB | 2 |
| 160 (E4, E5, D4-F/R) | 10 KB | 2 x 10 KB | 2 x 4 KB | overlay | 28 KB | 2 |

Exchange layout per head buffer: q `[key tile 4][K-slice 2][lane 32][8 B]` then v^T `[feature tile 2][key tile 4]
[lane 32][8 B]`: every read is one 8-byte LDS load per lane at a lane-contiguous address (no bank conflicts; M's
`sw_lfrag` addressing). Tails reuse the exchange and unit buffers after the MLP (merge: y2 of the window,
64 x C bytes = 2-10 KB, then Wm units) or before the head loop (fusion: upq of the window, 16 x 4C bytes = 2-10 KB).

### 7.4 VGPRs and occupancy (estimates, to be confirmed by RGA)

Per wave: head loop = x (C/8) + Dp f16 (C/4) + attention working set (32 score f32, 8 bias f32 streamed per key
tile, 16 key-q A fragments, 16 v^T fragments, 8 P, 4 O, 4 own q: about 84) + Stager prefetch (unit / 128 lanes / 16 B
x 4: up to 20) + addresses / temporaries (about 20). MLP = y1q (C/8) + y1h (C/4) + Dd (C/4) + current f32 tile 8 +
Du 16 + Au 4 + weight fragments 16 + prefetch 20 + temporaries 20.

| C | head loop | MLP | fusion half (+ xf) | budget (waves per SIMD from LDS) | expected waves per SIMD |
|---|---|---|---|---|---|
| 32 | ~140 (no prefetch) | ~105 | ~150 | 12 (6 WG x 4 / 2) -> <= 128 VGPRs for all 12; 10 at 152 | 10-12 |
| 64 | ~150 | ~125 | ~165 | 8 -> <= 192 | 8 |
| 96 | ~160 | ~145 | ~185 | 6 -> <= 256 | 6 |
| 128 | ~170 | ~165 | ~200 | 4 -> <= 256 | 4 |
| 160 | ~185 | ~185 | ~225 | 4 -> <= 256 | 4 |

(Wave32 register file: 1536 per lane per SIMD, granule 24, `swin_c64w2.hlsl` notes.) For C = 32 the attention set
dominates; if it lands above 128 VGPRs, the score phase runs key tiles in two halves (two passes over Z: the w values
are recomputed rather than held; about +60 VALU per head).

### 7.5 Tails

* **Merge (E1-E4)**: after y2, each wave writes its y2 tile (pi32 rows) to LDS, barrier; the window's 16 coarse tokens
  form one B tile per K-slice (lane r = coarse token 4 Yc + Xc, child n = 2dy + dx picks the fine token row of the
  gather), the Cn/16 output tiles are split over the 4 waves (E1: 6 tiles -> 2/2/1/1), K = 4C in C/8 steps,
  Q8, write coarse tokens inside the coarse tensor. All four children of a coarse token lie in the same token tile
  (even shifts), so the merged hidden is wave-local.
* **Hidden state, encoder (IN0, E1-E5)**: Dh = one padded WMMA tile per step (C/32 steps), rounding only the 3 used
  rows; sigmoid table loads; G, hnew; write `hnew` at the own level; merged hidden from the lanes of the four children
  (r, r+1, r+8, r+9 within lane half 0: DPP / `wg_xor` exchanges), written at level k+1 for stages with a merge, and
  for IN0 (whose next stage reads children concatenated; its merged hidden is the hold of E1).
* **Hidden state, decoder recurrent halves**: as above with 4 rows; `hold` = encoder `hnew` of the level (own level,
  reflected), Hc = 4 corner reads of the coarse hidden (8 bytes each, reflected at the coarse level), the bilinear
  chain in the spec's order A, B, Cc, Dc. D0-R: page 55 (not available; its head replaces the projection).
* **E1 adapter**: x = Q8(MM(in; Wa; ba)) from the 128-channel tile (4 output tiles, 4 steps) before the head loop;
  xr = d8(x) (the quantised adapter output, as the spec states).
* **Fusion halves Dk-F**: before the head loop. cin of the window's 4 x 4 coarse tokens (reflected with M) as one B tile
  per K-slice; Dup tiles (4C/16) split over the 4 waves, Kc/32 steps, Q8, written to LDS as `[coarse r][4C]`; barrier;
  wave w's fine token (row 2w + (r >> 3), column r & 7) reads parent `4w + (lx >> 1)`, child `2 (r >> 3) + (lx & 1)`;
  `xf = fma16(d8(skip), f, d8(upq))`, x = Q8(xf), xr = xf.

## 8. Attention details

### 8.1 Score bias
Wave w = query tile I = w reads, per head, the 4 bias blocks (w, J) of the expanded f32 table as the C operand of the
4 score tile pairs: 4 x 32 bytes per lane, issued before the q / v phase of the head (latency hidden). The
transposition of the 4 unstored blocks is done once at load time, so all blocks are the same contiguous fragment read.

### 8.2 Attention weight without rounding (51.3 step 3)
Within the converter-checked range p is an f16 in [1 + 1/32, 1 + 31/32): exponent field 15, mantissa field m (10 bits).
Then y = 32 (p - 1) = m / 32, n = floor(y) = m >> 5 (1 .. 30), and `w = 2^(n-15) (1 + (m & 31)/32)` is the f16 with
exponent field n and mantissa field `(m & 31) << 5`: **bits(w) = (bits(p) << 5) & 0x7FE0**. For a packed pair:
`(P << 5) & 0x7FE07FE0` (the bits shifted across the halves fall under the mask): `v_lshlrev_b32` + `v_and_b32`, both
VOPD-pairable. The converter (page 57) rejects constants for which a clamped score leaves the range; our synthetic
constants are generated with the same exhaustive check over every f16 c in [A_CLAMPn, A_CLAMP] (test inputs). Probe P3
runs the whole step-3 function for all 65 536 f16 inputs against the reference (= test 51.8.2).

### 8.3 Normalisation
* Z: the lane's 32 w (16 packed pairs) summed by 16 `v_dot2_f32_f16` with (1, 1), plus the partner lane (lane ^ 16)
  through `wg_xor16`, rounded to f16. This is the spec's accepted "f32 in any order" form. Variant **Zx** (test builds):
  exact sum (w = (32 + m5) 2^(n-20): 64-bit integer accumulation in units of 2^-19, then RNE to f16), so that the result
  equals the spec reference; and a **Z-supplied** hook (define) that reads Z per (window, head, query) from a buffer for
  test 51.8.1's first run.
* r = R(1/Z): Z >= 2^-8 (spec), so 1/Z is a normal f32. For an 11-bit Z and an f16 midpoint m (12-bit significand),
  |1/Z - m| >= 2^-22 relative unless 1/Z = m, which is impossible (Z m = 1 has no such solution); `v_rcp_f32` is within
  1 f32 ulp (2^-23), so its RNE f16 conversion is the correctly rounded R(1/Z). Probe P2 checks all f16 Z >= 2^-8
  (and Z = +inf -> r = +0 -> zero P codes, 51.3 step 4).
* P = Q8(R(w r)): `v_pk_mul_f16` (one rounding of the exact product), widened, packed; the P tile is the B operand of
  the value mixing as it stands.

## 9. Bit-exactness risks and the probes that settle them

| # | risk | effect | handling |
|---|---|---|---|
| R1 | WMMA f32 accumulation not exact (7.2) | rare one-step f16 differences of a step | P1; variant X for code-for-code tests; spec clause 51.1 **[Q1]** |
| R2 | Z order | one f16 step of Z in ~22 % of rows (spec-accepted) | default f32; Zx exact variant; Z-supplied hook |
| R3 | r = R(1/Z) | none expected | argument 8.3 + P2 exhaustive |
| R4 | driver fuses / folds f16 operations | wrong rounding count | fences + ISA count (P7) |
| R5 | packed f16 ops: rounding, NaN, denormals under D3D12 | wrong attention weight / activation | P3 exhaustive (51.8.2) incl. subnormal and NaN inputs |
| R6 | Q8 of NaN: the hardware conversion gives 0xFF for every NaN | +NaN code differs from 51.1 | standalone Q8 (test 51.8.3): a NaN is replaced by +-1000 of its sign, which the unsaturated conversion encodes as 0x7F / 0xFF = the spec code; in-network NaN only after f16 overflow, sign undefined by IEEE **[Q2]** |
| R7 | negative zero codes | none (51.1: equal in-network) | P4 checks the conversion keeps 0x80 for negative values rounding to 0 |
| R8 | sigmoid | none | exact table; table generator checked independently (P5, = 51.8.4) |
| R9 | e4m3 NaN inputs / inf start values inside WMMA | only in overflow tests | P1 includes non-finite C and NaN codes; IEEE propagation expected |
| R10 | border coordinates (small extents) | wrong token | general reflection; test 51.8.5 down to a 32 x 32 tensor |
| R11 | merged hidden R(h/4) subnormal | none (pk_mul by 0.25 is RNE of the exact quotient) | P3-style exhaustive check of the mean |
| R12 | FP32 denormal flushing | none (7.1) | - |

Probes (each a small D3D12 kernel + Python check, under the GPU lock, < 1 minute):
P1 WMMA accumulation model; P2 1/Z exhaustive; P3 attention weight, activation, f16 mean, bilinear helpers, all f16
inputs; P4 Q8 exhaustive (both paths); P5 sigmoid table; P6 throughput: can VALU of one wave issue while another
wave's WMMA occupies the same SIMD (decides whether section 10's additive model is pessimistic); P7 RGA ISA counts per
kernel; P8 LDS exchange bank-conflict check.

## 10. Time estimate at 1440p Quality

Render 1706 x 960 -> level-0 tokens 1728 x 960 (Wt rounded up to 32); level k = 1728/2^k x 960/2^k; windows with a
shift of 4 (the grid of 51.6).

### 10.1 Model
Per wave-tile (16 tokens): cycles = 8 x WMMA (fp8 16x16x16 = 4096 MACs at 512 MAC per clock per SIMD) + 1.1 x VALU
issue slots (section 7.1's instruction counts). GPU: 64 CUs x 2 SIMDs at 2.9 GHz, efficiency 0.83, with per-CU load
balance (ceil(windows / 64) workgroups per CU, 4 waves on 2 SIMDs). The efficiency is calibrated with the same counting
applied to Model M's D3D12 c64 block (184 WMMA + about 1700 VALU per 16-token window, 25 590 windows: measured
0.265 ms). Uncertainty: +-40 % (the VALU counts are hand counts; whether WMMA and VALU overlap is P6).

### 10.2 Per stage (variant E; variant L in the last column)

| stage | level | tokens | windows | C/H | WMMA / tile | VALU / tile | WMMA share | us (E) | us (L) |
|---|---|---|---|---|---|---|---|---|---|
| IN0 (core + hidden; page 52 front end not included) | 0 | 1,658,880 | 26,257 | 32/2 | 90 | 1,613 | 28 % | 852 | 768 |
| E1 | 1 | 414,720 | 6,649 | 64/2 | 268 | 3,541 | 36 % | 522 | 390 |
| E2 | 2 | 103,680 | 1,705 | 96/4 | 550 | 6,633 | 38 % | 262 | 183 |
| E3 | 3 | 25,920 | 448 | 128/4 | 856 | 9,521 | 40 % | 101 | 66 |
| E4 | 4 | 6,480 | 112 | 160/8 | 1518 | 16,139 | 41 % | 50 | 31 |
| E5 | 5 | 1,620 | 40 | 160/8 | 1418 | 15,289 | 40 % | 23 | 15 |
| D4-F | 4 | 6,480 | 112 | 160/8 | 1508 | 16,520 | 40 % | 50 | 32 |
| D4-R | 4 | 6,480 | 112 | 160/8 | 1418 | 15,319 | 40 % | 47 | 30 |
| D3-F | 3 | 25,920 | 448 | 128/4 | 848 | 9,816 | 39 % | 102 | 68 |
| D3-R | 3 | 25,920 | 448 | 128/4 | 776 | 8,865 | 39 % | 93 | 62 |
| D2-F | 2 | 103,680 | 1,705 | 96/4 | 544 | 6,848 | 37 % | 267 | 189 |
| D2-R | 2 | 103,680 | 1,705 | 96/4 | 502 | 6,243 | 37 % | 244 | 174 |
| D1-F | 1 | 414,720 | 6,649 | 64/2 | 232 | 3,372 | 33 % | 481 | 371 |
| D1-R | 1 | 414,720 | 6,649 | 64/2 | 212 | 3,049 | 34 % | 436 | 338 |
| D0-F | 0 | 1,658,880 | 26,257 | 32/2 | 96 | 1,748 | 28 % | 919 | 823 |
| D0-R (core; page 55 head not included) | 0 | 1,658,880 | 26,257 | 32/2 | 88 | 1,588 | 28 % | 837 | 753 |
| **total** | | | | | | | | **~5.3 ms** | **~4.3 ms** |

Where the time goes:
* **Level 0 (2.6 ms)**: VALU-bound attention: 128 scores per token, about 7 VALU per score (f16 conversion of the
  score, 6 packed / integer instructions per pair, Z, P's multiply, widening and packing). C = 32 makes every product
  a single step, so variant L barely helps there.
* **Levels 1-2 (2.2 ms)**: the step rounding (2 VALU per value per step, as many cycles as the WMMA pair) is about a
  third; the MLP's Dd chain (C/8 steps per output value) is the largest single item.
* **Levels 3-5 (0.47 ms)**: E5 and the level-4 stages have 40-112 windows for 64 CUs: latency-bound (one or two
  workgroups per CU). Options if they matter: 8-wave workgroups splitting output tiles (2 waves per token tile, Au /
  hidden exchanged through LDS), or one persistent kernel for E4 .. D3-R with row counters (M's bottleneck pattern),
  which also removes 7 launch gaps.
* **Memory**: level-0 stages move about 80 bytes per token (input, skip / coarse, output, hidden): ~130 MB per stage,
  ~0.2 ms at the card's DRAM bandwidth; the 64 MB cache holds part of it; overlapped with the compute above.

Levers in order of expected gain: variant L (spec decision, -1.0 ms); VOPD pairing of the integer / f32 parts of
the attention (-5..-10 % at level 0); f16 bias instead of f32 when VALU is not the limit (memory vs VALU, A/B);
two token tiles per wave at C >= 128 (halves LDS weight reads; more VGPRs); persistent coarse chain (-30..-50 us).

## 11. File plan for the next tasks (clean/rr_f/)

* `reference/rr51.py`: NumPy + Python integers, the spec text literally (exact sums by integer scaling: products x 2^24
  and f16 start values x 2^24 are integers), Q8 / R / d8 / sig / attention weight / activation / border rule; the
  oracle of every test. `reference/synth51.py`: synthetic weights and constants in the spec's ranges (converter-style
  range check of p for every clamped f16 c, w_max record, shifts even and below 8, non-trivial ra / m / f / g).
* `kernels/rr_common.hlsli` (layouts, step / Q8 / f16 helpers, reflection), `kernels/rr_block.hlsl` (the core with
  `RR_C`, `RR_H`, `RR_FRONT` (none / adapter / fusion), `RR_TAIL` (merge / hidden-enc / hidden-dec) defines, variants
  `RR_STEP` E / X / L, `RR_Z` f32 / exact / supplied), `kernels/rr_probe_*.hlsl` (P1-P8).
* `tests/`: probes, per-stage code-for-code tests on random windows (51.8.1-51.8.5) under the GPU lock, timing on the
  1440p Q sizes.

## 12. Constants used by the design that the spec leaves to page 57 (test inputs here)

A_SCALE, A_CLAMP(n), A_P1, A_P0, ACT_* , ra, m, f, g, SH, Tb slot order, weights: synthetic (section 11). The design
does not depend on their values beyond the spec's stated ranges, except the bias table size (12 blocks) and the
attention range check that makes 8.2 valid.
