# Data formats used by the clean Swin-stage kernels

Status: milestone 1 (Swin stages, spec 02). Producer of the weight formats: `clean/runner/packing.py` (and later the
converter, spec 08). Consumer: `clean/kernels/swin.hip`.

All multi-byte values are little-endian. "FP8" is E4M3FN (bias 7, max 448, NaN = `0x7F`/`0xFF`, no infinities).

## Feature maps

| Map | Shape | Element |
|---|---|---|
| block input / output, skip maps S1 S2, merged maps | `[H][W][C]`, channel fastest | 1 byte FP8 |
| expanded decoder source `e` (internal, decoder stages) | `[H][W][C]` at the stage resolution | 1 byte FP8 |

A token's C channels are contiguous (C = 64, 96, 128 bytes). Buffers must be 16-byte aligned (`hipMalloc` is). The
decoder block input `x = e + S` (spec 02 "Expand") is never stored. The block kernel reads `e` and `S` and adds them
in FP32 (the sum of two FP8 values is exact in FP32, but not in FP8 or FP16).

## Packed weights: one blob per stage

`[Swin block section][zero padding to a multiple of 256][merge section (enc1, enc2) | expand section (dec2, dec1)]`

| Stage | Block section | Merge / expand section at | Total bytes |
|---|---|---|---|
| enc1 | 49168 | 49408 (merge, 24960 B) | 74368 |
| enc2 | 117776 | 118016 (merge, 49664 B) | 167680 |
| b0 … b5 | 188432 | – | 188432 |
| dec2 | 117776 | 118016 (expand, 50688 B) | 168704 |
| dec1 | 49168 | 49408 (expand, 25600 B) | 75008 |

### Common building blocks

**WMMA fragment** (256 bytes): one operand of a 16×16×16 FP8 WMMA. Lane `l` (0…31) reads bytes `[8l, 8l+8)`.
Lane `l` holds fragment row `r = l mod 16` and K slots `(h = l div 16, j = 0…7)`, slot `(h, j)` being packed K
position `8h + j`. The hardware's actual K index for a slot does not matter, as long as the A and B operands use
the same one (checked on the GPU by `runner/selftest.py`). For a weight used as the B operand (only `Wv`),
"row" means the B column.

A matrix `M` with `16·MT` rows and `16·KC` columns becomes `MT·KC` fragments. Fragment `(mt, kc)` holds
`M[R[16·mt + r], K[16·kc + 8h + j]]` at lane `16h + r`, byte `j`. `R` is the row order and `K` is the column
order (below). Fragments are stored `[mt][kc]` unless noted.

**Accumulator layout.** Element `v` of the FP32 WMMA result in lane `l` holds column `l mod 16` and row
`D_ROW[h][v]`, with `h = l div 16`. The default assumed for gfx12 is `D_ROW[h][v] = v + 8h` (`d_row="gfx12"`).
Only the packing depends on `D_ROW`. If `selftest.py` reports a different table, pass it to `Runner(d_row=...)`
or set the environment variable `WG_D_ROW`. No kernel change is needed.

**Layout L** (per-token channels in registers): lane half `h` holds channels `[h·C/2, (h+1)·C/2)`, and register
group `g` holds channels `h·C/2 + 8g + 0…7`. Orders used:

| Name | Definition |
|---|---|
| `R_L(C)` (rows → layout L) | `R[16·mt + D_ROW[h][v]] = h·C/2 + 8·mt + v` |
| `R_id` | `R[i] = i` |
| `R_E(C)` (expand rows) | `R[16·(k·C/16 + g) + D_ROW[h][v]] = k·C + h·C/2 + 8g + v` (k = sub-pixel) |
| `K_L(C)` (layout-L activations) | `K[16g + 8h + j] = h·C/2 + 8g + j` |
| `K_D(n)` (WMMA outputs) | `K[16g + 8h + j] = 16g + D_ROW[h][j]` |
| `K_Z(C)` (merge input) | `K[16g + 8h + j] = h·2C + 8g + j` (half h = image row 2i+h: left then right token) |

**Tile-ordered vector** (biases, FP32): `packed[16·mt + 8h + v] = vec[R[16·mt + D_ROW[h][v]]]`.

### Swin block section (C, heads, MLP; D = 32·heads, KC = C/16, MT = MLP/16)

A header of FP32 vectors is followed by a **fragment stream** in the exact order the kernel uses the fragments.
The stream is cut into units that the workgroup copies into LDS one after another (double-buffered).

| Offset | Content | Format |
|---|---|---|
| FLAGS = 0 | flags word (u32) + 12 bytes zero | bit 0: the weights prove that no inner FP8 pack can exceed ±460, so the runner launches the `_nc` ("no clamp") kernels, which skip the ±448 clamp and give identical results; computed by `packing.block_bounds` |
| G1 = 16 | γ1 | C × f32, natural order |
| G2 = 16 + 4C | γ2 | C × f32, natural order |
| BP = 16 + 8C | bp | tile-ordered with `R_L(C)` |
| B2 = 16 + 12C | b2 | tile-ordered with `R_L(C)` |
| B1 = 16 + 16C | b1 | tile-ordered with `R_id` |
| BIAS = 16 + 16C + 4·MLP | B | f32 `[head][t][h][v] = B[head][t][D_ROW[h][v]]` (t = query, raster index in window) |
| STREAM = BIAS + 1024·heads | fragment stream | units below, back to back |

The flag bound (`block_bounds`): the RMSNorm output has L2 norm ≤ max|γ|·√C for any input, q8 changes each element by at most 1/16, and every inner pack is a row of a projection of such a vector: |q/k| ≤ ‖Wqk row‖₂·‖a‖₂, |o| ≤ max|v| ≤ ‖Wv row‖₂·‖a‖₂, |m| ≤ 1.07 max|γ2|√C, |hidden| ≤ ‖W1 row‖₂·‖m‖₂ + |b1| (GELU never increases the magnitude).

Fragment matrices: `Wqk[head]` (32 × C, `R_id`, `K_L(C)`), `Wv[head]` (32 × C as the B operand, `R_id`, `K_L(C)`),
`Wp` (C × D, `R_L(C)`, `K_D(D)`), `W1` (MLP × C, `R_id`, `K_L(C)`), `W2` (C × MLP, `R_L(C)`, `K_D(MLP)`).

Stream units, in order (the kernel stages one unit at a time through LDS, one barrier per unit):

1. For each head `hd`, `HEAD(hd)` ((4·KC + 2·KC) fragments):
   - QKV part: for `mt` < 2, `kc` < KC: `Wqk[hd]` fragment (mt, kc), then `Wv[hd]` fragment (mt, kc).
   - WP part: for `mt` < KC, `dt` < 2: `Wp` fragment (mt, 2·hd + dt).
2. For each `u` < MT/HPU, `MLP(u)` (2·KC·HPU fragments): for `j` < HPU, with `ht = u·HPU + j`: `W1` fragments
   (ht, kc < KC), then `W2` fragments (mt < KC, ht). `HPU` is the largest divisor of MT that keeps the unit
   ≤ 16 KB.

The byte stream is the same as in earlier revisions; only the cut into units differs. The kernel always copies
`MAXU = max(unit sizes)` bytes per unit (a head unit drags along the start of the following unit; `MAXU` equals the
MLP unit size, so nothing is ever read past the end of the stream).

| C | STREAM | HEAD unit | MLP unit (HPU) | MLP units | LDS per workgroup | block size |
|---|---|---|---|---|---|---|
| 64 | 4096 | 6144 | 16384 (8) | 2 | 32768 | 49152 |
| 96 | 7168 | 9216 | 12288 (4) | 6 | 24576 | 117760 |
| 128 | 8192 | 12288 | 16384 (4) | 8 | 32768 | 188416 |

### Patch-merge section (C → C')

| Offset | Content | Format |
|---|---|---|
| 0 | bm | tile-ordered with `R_L(C')` (C' × f32) |
| 4C' | Wm | `Wm` (C' × 4C), `R_L(C')`, `K_Z(C)`; fragments `[mt<C'/16][g<C/4]` |

### Expand section (C_low → 4C)

| Offset | Content | Format |
|---|---|---|
| 0 | be | tile-ordered with `R_E(C)` (4C × f32) |
| 16C | We | `We` (4C × C_low), `R_E(C)`, `K_L(C_low)`; fragments `[mt<4C/16][kc<C_low/16]` |

## Kernels and launch geometry

The table gives the base build. `runner/variants.py` lists build variants with other workgroup sizes and windows per
wave (compile-time knobs at the top of `kernels/swin.hip`); the packed weights are identical for all of them, and
the runner reads each variant's geometry from `variants.json`.

1-D grids. The Swin block kernels use 256-lane workgroups (8 waves). The 8 waves share each weight unit staged in
LDS (2 × the largest unit: 32 KB for C=64 and 128, 24 KB for C=96). Each staged transfer moves exactly the unit's
size (head units 6 / 9 / 12 KB, MLP units 16 / 12 / 16 KB for C = 64 / 96 / 128). Patch merge and expand also use 256 lanes, and
copy their whole fragment section into LDS once per workgroup (24 KB or 48 KB).

| Kernel | Arguments | Work per wave | Grid |
|---|---|---|---|
| `swin_block_c64`, `_c64_add`, `swin_block_c96`, `_c96_add`, `swin_block_c128` | x, xadd, out, w, H, W, ox, oy | 1 window | ⌈nwin / 8⌉ |
| `patch_merge_c64` / `_c96` | s, out, w + merge offset, H, W | 32 merged tokens | ⌈(H/2·W/2) / 256⌉ |
| `expand_c128` / `_c96` | y, e, w + expand offset, H/2, W/2 | 32 source tokens | ⌈(H/2·W/2) / 256⌉ |

Every block kernel also exists as `<name>_nc` (no clamp, used when the weights' flags word proves the range; see
above) and the following fused kernels (variant `fuse`; 256 lanes, one window per wave, even window offsets only; the
runner falls back to the separate kernels for odd offsets). The weights are the same packed block + tail:

| Kernel | Arguments | What it adds |
|---|---|---|
| `swin_block_c64_mg`, `_c96_mg` (enc1, enc2) | x, 0, out (skip map), w, H, W, ox, oy, merged | after the block: the 8 windows' outputs go to LDS, the 32 merged tokens (4 windows x 4 groups per 16-column tile) are computed by the 8 waves from the merge section of `w` (at the block size rounded up to 256 bytes) and written to `merged` |
| `swin_block_c64_add_ex`, `_c96_add_ex` (dec1, dec2) | y (low-resolution map), skip, out, w, H, W, ox, oy, null | before the block: the window's 16 tokens are expanded from `y` into LDS (a window needs at most 2 x 2 low-resolution tokens, with their 4 sub-pixels each, also at mirrored border positions) and added to the skip |

`nwin = ⌈(W + ox)/4⌉ · ⌈(H + oy)/4⌉`. Windows are numbered in raster order, and window `(wy, wx)` covers rows
`4wy − oy …` and columns `4wx − ox …`. Kernels without `_add` ignore `xadd` (may be null).
Stage sequences: enc = block → merge. b = block. dec = expand → block `_add` (xadd = skip).

## Input encoder (enc0, spec 03)

Producer of the weight blob: `clean/runner/enc0.py` (`pack`). Consumer: `clean/kernels/encoder.hip`
(`input_encoder`). Sizes: render R = (rw, rh), padded F = (fw, fh), reconstruction D = (dw, dh), reduced history
(hw, hh) = (dw/2, dh/2) (integer division, as in spec 90).

### Per-frame buffers

The runner converts the spec 90 arrays into these. In the game they will come from the FSR inputs and from the
previous frame's stages, and the history formats are the storage formats of spec 01.

| Buffer | Shape | Element |
|---|---|---|
| color | `[rh][rw]` | 4 × f32: R, G, B, unused (16 bytes, 16-byte aligned) |
| depth | `[rh][rw]` | f32 |
| motion | `[rh][rw]` | 2 × f32 (x, y), raw game motion vectors |
| reduced history Hr | `[hh][hw]` | u32 R10G10B10A2 unorm: R = bits 0–9, G = 10–19, B = 20–29, A = 30–31 (not read); value k / 1023 |
| latent L | `[rh][rw]` | 4 × s8 snorm, channel c in byte c; value max(k / 127, −1) |
| motion history Mh | `[rh][rw]` | 2 × FP16 (x in the low half), reconstruction pixels |
| **out** features | `[fh/2][fw/2][64]` | FP8, the enc1 input map (layout as "Feature maps" above) |
| **out** m_near, m_far, m_center | `[rh][rw]` each | 2 × FP16 (x low). m_near is the next frame's Mh |

Frame constants are passed by value as `struct WgEncParams` (all fields 4 bytes, in this order): int `rw, rh, fw, fh,
hw, hh, frame, reset, depth_inverted, pad0`; float `dw, dh, kx, ky` (D/R), `ikx, iky` (R/D), `sx, sy`
(mv_scale · D/R), `bx, by` (mv_bias), `jx, jy`, `jpx, jpy`, `dis_scale`
(540/rh), `idw, idh` (1/D), `irw, irh` (1/R). Total 116 bytes. The exposures are not in the struct: the kernel
reads the pair [e, e_prev] (raw values, float32) from a device buffer `expo` and replaces 0 by 1 itself (spec 03), so
an exposure computed on the GPU (spec 07) needs no CPU read-back.

### Weight blob (10752 bytes)

| Offset | Content | Format |
|---|---|---|
| 0 | b | 64 × f32, tile-ordered with `R_L(64)` |
| 256 | σ1, σ2 | 2 × f32, then zero padding to 512 |
| 512 | T | u8 `[8][16][16]` (frame, y mod 16, x mod 16) |
| 2560 | W | fragments `[mt < 4][kb < 8]` (256 bytes each, 8 KB): rows `R_L(64)`, K order `K_E` below |

**K order `K_E`** (128 K slots = 8 blocks of 16, 80 used). Sub-pixel p (0…3, p = dx + 2·dy) uses blocks 2p and
2p + 1, so its partial sum can be rounded to FP16 before the next sub-pixel. Its owner lane half is o = p div 2, the
lanes that compute row dy = o of the token's 2 × 2 pixels. Value i of sub-pixel p is column 20p + i of W.

| Block | Slots (h = o, j = 0…7) | Slots (h = 1 − o, j = 0…7) |
|---|---|---|
| 2p | values 0…7 | values 8…15 |
| 2p + 1 | values 16…19 in j = 0…3; j = 4…7 zero | zero |

Zero slots have zero weights, and the kernel also sets their B operand (activations) to 0.

### Launch

`input_encoder(color, depth, motion, hist, latent, mhist, features, m_near, m_far, m_center, expo, w, params)`.
1-D grid of ⌈(fh/2 · fw/2) / 128⌉ workgroups of 256 lanes (8 waves). Wave i of the grid handles tokens
16i … 16i + 15 in raster order. Lane (t, h) computes the pixels (2c, 2r + h) and (2c + 1, 2r + h) of token t =
(r, c), and writes the motion surfaces of those pixels that lie inside R. The 8 KB of W fragments are staged in LDS
once per workgroup (since the performance milestone, round 2: read by each wave straight from global memory, no LDS;
build switch `WG_ENC_FRAG_GLOBAL`). Since the performance milestone the kernel needs rw ≥ 4, rh ≥ 2 and a reduced
history of at least 2 × 2 texels (4-column depth windows, 2 × 2 texel reads); the runner checks this.
Timing variants with other input formats (`runner/perf_variants.py`; not used by the runtime): `colorh` reads the
colour as 4 × FP16 per pixel (8 bytes), `motionh` the motion vectors as 2 × FP16 (4 bytes), `in16` both.

## Coefficient decoder (dec0, spec 05)

Producer of the weight blob: `clean/runner/dec0.py` (`pack`). Consumer: `clean/kernels/decoder.hip` (`coef_decoder`).

### Per-frame buffers

| Buffer | Shape | Element |
|---|---|---|
| features Z | `[fh/2][fw/2][64]` | FP8, the dec1 output map |
| motion M | `[rh][rw]` | 2 × FP16 (x low), the enc0 `m_near` output of the same frame |
| previous latent H | `[rh][rw]` | 4 × s8 snorm (channel c in byte c), as read by enc0 |
| **out** A, B | `[rh][rw]` each | RGBA8 unorm: codes of channels 0–2 in bytes 0–2, byte 3 = 0 |
| **out** C | `[rh][rw]` | 4 × FP16 (8 bytes): C0, C1, C2, 0 |
| **out** new latent L | `[rh][rw]` | 4 × s8 snorm (the other buffer of the double-buffered latent) |

Only pixels inside R are written. Frame constants: `struct WgDecParams` by value (56 bytes): int `rw, rh, fw, fh,
reset, pad0`; float `dw, dh, kx, ky` (RN32(D/R)), `jx, jy, pjx, pjy`.

### Weight blob (10752 bytes)

| Offset | Content | Format |
|---|---|---|
| 0 | b | 80 × f32 tile-ordered with rows `R_C` below (padding rows 0) |
| 320 | S, T | 2 × f32, zero padding to 512 |
| 512 | tables | 5 × 256 × 4 bytes, indexed by an 8-bit code (below) |
| 5632 | W | fragments `[mt < 5][kb < 4]`, 256 bytes each: rows `R_C`, K order `K_C` |

Bytes 512 … 10751 (tables and fragments) are copied to LDS once per workgroup.

**Tables** (since the performance milestone; `dec0.tables`, `model.cpp` `pack_dec0`). Table t is at byte
512 + 1024·t; entry a is the 4-byte word for the FP8 code a (NaN codes as value 0; they never occur), table 4 for the
s8 latent code. With g(a) = RN16(σ(a)) (σ in single), sx(a) = RN32(S·g(a)), th(v) = RN16(tanh(v)):

| t | Entry (4 bytes) |
|---|---|
| 0 | bits 0–7: stored code of A for a, round(RN32(g(a)·255)); bits 16–31: FP16 bits of C0 for a, RN16(RN32(sx(a)²)) (C1 is the same function of a7) |
| 1 | f32 sx(a) |
| 2 | f32 ρ(a) = RN32(tanh(RN32(T·a))) |
| 3 | bits 0–15: FP16 bits of z(a) = RN16(0.5 + 0.5·th(a/2)); bits 16–31: FP16 bits of th(a) (the candidate g) |
| 4 | f32 latent texel value RN32(k/127) of the s8 code k (−128 → −1) |

(The first kernel version, `kernels/baseline/decoder_v1.hip`, used five f32 tables g(a), th(a), th(a/2), ρ(a) and
the snorm values; `dec0.pack_v1` builds that blob for timing comparisons.)

- **`R_C`** (80 rows, 68 used): accumulator slot (tile mt, lane half h, element v) holds output 34h + 8mt + v when
  8mt + v < 34, else a zero row: `R[16·mt + D_ROW[h][v]] = 34h + 8mt + v`. Lane half h thus holds the 2 × 17
  outputs of the token's pixels in row dy = h (outputs 17s … 17s + 16 for s = 2h, 2h + 1).
- **`K_C`**: `K[16·kb + 8h + j] = 32·(kb div 2) + 16h + 8·(kb mod 2) + j`. Blocks 0–1 hold channels 0…31 and blocks
  2–3 hold channels 32…63, so the kernel rounds the accumulator to FP16 after block 1 (u) and after block 3 (v). Lane
  half h reads the 16 contiguous bytes at 16h and at 32 + 16h of the token.

### Launch

`coef_decoder(features, motion, latent_prev, A, B, C, latent, w, params)`: a 1-D grid of ⌈(fh/2 · fw/2) / 128⌉
workgroups of 256 lanes. Wave i handles tokens 16i … 16i + 15 in raster order. Lane (t, h) decodes the pixels
(2c, 2r + h) and (2c + 1, 2r + h) of token t = (r, c) that lie inside R. Floating-point contraction is disabled in
`decoder.hip` (`#pragma clang fp contract(off)`), so the single-precision steps of spec 05 are rounded one operation
at a time.

**Fused dec1 + dec0** (prototype, `kernels/dec1_dec0.hip`, not used by the runtime yet):
`swin_block_c64_add_dec0[_nc](x, xadd, w, H, W, ox, oy, motion, latent_prev, A, B, C, latent, wdec, params)`: the first
seven arguments as for `swin_block_c64_add[_nc]` (x = dec1's expanded map, xadd = skip, w = dec1 blob; no output map),
the rest as for `coef_decoder` (wdec = the dec0 blob, unchanged). Launch geometry of the block kernel: ⌈windows / 8⌉
workgroups of 256 lanes (one window per wave). `_nc` when bit 0 of dec1's FLAGS word is set, as for the block.

Timing variant `pack12` (`runner/perf_variants.py`; needs the matching read side in the output stage, not used by the
runtime): A, B and C without padding, 12 bytes per pixel in two images: image A8 `[rh][rw]` 8 bytes = codes A0 A1 A2
B0 B1 B2, then C2 as FP16; image C4 `[rh][rw]` 4 bytes = C0, C1 as FP16. Kernel arguments: A8 in place of A, C4 in
place of C, B unused.

## Auto-exposure (exposure, spec 07)

Kernels: `clean/kernels/exposure.hip` (`exposure_partial`, `exposure_final`); launch plan `clean/runner/exposure.py`.
No weights.

| Buffer | Shape | Element |
|---|---|---|
| color | `[rh][rw]` | 4 × f32 (R, G, B, unused), the same layout as the enc0 color input |
| partials (internal) | ⌈n_y / 4⌉ | f32 partial sums of z |
| **out** e | 1 | f32 holding the FP16-rounded exposure |

Frame constants: `struct WgExpParams` by value (36 bytes): int `rw, rh, nx, ny, nparts, pad0`; float `inv_pi`
(1/π), `K` (α/(1−α)), `inv_n` (1/N).

Launch: `exposure_partial(color, partials, params)` with ⌈n_y / 4⌉ workgroups of 256 lanes (4 sample rows each,
lanes strided over the n_x columns), then `exposure_final(partials, e, params)` with one workgroup of 256 lanes.
Sample positions are computed in integers: X_i = ⌊(2i(rw − 1) + n_x − 1) / (2(n_x − 1))⌋, the same for Y_j.

## Per-frame runtime (spec 10 core)

Code: `clean/runtime/` (C API in `wg_runtime.h`), built by `build.py` as `libwg_runtime.so` / `wronggpu_runtime.dll` (Windows, with the D3D12 interop below).
Python front end: `clean/runner/pipeline.py`. Buffers are allocated once at create for the maximum render size and
for O (D = ⌊(3·O + 1)/2⌋); each frame uses its own R densely (`[rh][rw]` with row pitch rw) at the start of each buffer.

### Model data

Normal use: `<model_dir>/model.wgcm`, the converter's single output file (`clean/converter/OUTPUT-FORMAT.md`). The
runtime checks, in this order: magic and version; the whole-file SHA-256 (last 32 bytes); every stage and tensor of
spec 08 section 7.2 with its dtype and shape. For each stage it also recomputes the keyed SHA-256 of the stage's
tensor data with the key stored in the file (spec 08 section 7.3) and compares it with the stored digest and with the
digest of the map shipped with the runtime (`map_path`). Then come the value checks below. Any failure makes create
fail with a plain-words message (spec 10 section 8).

The file also stores the SHA-256 of the map it was converted with. That hash alone does not decide: a file of
another map is accepted when every stage the runtime needs is in it and equals the shipped map's digest. This is the
case of a file written with an earlier version of the same map, e.g. a player's model from before the group `filter`
existed (thirteen stages; spec 08 section 7.2: such a file "counts as not present" for the output-filter weights);
`wg_rt_kernels` then reports `model_map=other` (`same` otherwise, `none` for the raw layout). A file whose stages the
shipped map does not describe is refused as before ("converted with a different map"). Stage entries that the shipped
map does not list are skipped (a file of a later map with a further optional group); in a file of the shipped map
itself they are an error.

Tests: `<model_dir>/raw/<stage>/<name>.bin` (or `<model_dir>/<stage>/<name>.bin`, the converter's `--raw` layout), used
when there is no model.wgcm or with `model_format = 1` (`"raw"`). Every tensor of spec 08 section 7.2 is read,
row-major little-endian float32 (uint8 for enc0 T). The runtime checks that every file is present with its exact
size, that all floats are finite and that T ≤ 127. Then it packs each stage once (`model.cpp`, C++ ports of the
Python packers, byte-identical; `pipeline.py` checks this at construction). dec0 W/b are read as [4][17][64] /
[4][17], the row-major layout of [68][64] / [68]. post `s` is passed to the post kernels.

Optional group `filter` (spec 08 draft 2 section 7.2; spec 06 draft 4): one tensor `output_taps` [5] float32, the
output filter's weights w₋₂ … w₊₂ for an even output index. In model.wgcm it is one more stage in the index, after
`post`, present only when the converter found the weights; in the raw layout `raw/filter/output_taps.bin` (20 bytes),
present likewise. Both loaders (`model.cpp`) take it when it is there and check it like every other tensor (exactly
this one tensor, shape [5], float32, finite values; in model.wgcm the stage's keyed digest against the stored one and
the map's). A model without it is complete. The map gives this stage a **list** of digests (one per variant of its
source, spec 08 section 5) where the other stages have one hex string; the file's stage must equal one entry. The
runtime keeps the five values on the host (`wg_rt_stage_blob("filter")`: 5 × f32 as loaded, or "no blob";
`Pipeline.output_taps()`); they are not packed into a device blob.

### State and buffers (names as accepted by `wg_rt_buffer`, for the last frame)

| Name | Format | Double-buffered (copy read / written in frame f) |
|---|---|---|
| `input.color` | `[rh][rw]` 4 × f32 (alpha unused) | – |
| `input.depth`, `input.motion` | f32; 2 × f32 raw game motion | – |
| `exposure.pair` | f32 [3]: e of this frame, e_prev, e kept for the next frame (raw values; stages replace 0 by 1) | – |
| `enc0.features` | FP8 `[fh/2][fw/2][64]` | – |
| `enc0.m_near` / `state.motion_history` | 2 × FP16 `[rh][rw]` | write 1 − f / read f |
| `enc0.m_far`, `enc0.m_center` | 2 × FP16 `[rh][rw]` | single |
| `enc1.skip`, `enc1.merged`, `enc2.skip`, `enc2.merged`, `b0.out` … `b5.out`, `dec2.out`, `dec1.out` | FP8 maps | single |
| `dec0.A`, `dec0.B` | RGBA8 unorm `[rh][rw]` | single |
| `dec0.C` | 4 × FP16 `[rh][rw]` | single |
| `dec0.latent` / `state.latent_prev` | 4 × s8 snorm `[rh][rw]` | write 1 − f / read f |
| `post.history` / `state.history_prev` | 4 × FP16 `[dh][dw]` (8 bytes per pixel) | write 1 − f / read f |
| `post.reduced` | R10G10B10A2 `[dh/2][dw/2]` (the enc0 input format) | single (enc0 reads it before post writes it) |
| `post.display` | three FP16 planes of D, row pitch display_pitch(dw) (the spec 06 kernels' format) | single |
| `filter.out` | 3 × f32 `[oh][ow]` (RGB, linear) | single |

Frame state (spec 10 section 3) is kept on the CPU (n, f, previous jitter, last R), except the exposure pair, which
lives on the GPU so that auto-exposure never needs a CPU read-back. The effective reset (game reset, first frame,
R changed) is passed to enc0, dec0 and post, which then do not read the previous-frame state.

### Experimental reconstruction size D = O (off by default)

`WgRtConfig.recon_output = 1`, or the environment variable `WG_RECON_SIZE=output` (read by the runtime core, so the
in-game DLL and `Pipeline` both see it; `Pipeline` also takes `flags={"recon": "output"}`; in game the backend option
`recon=output`), sets D = O instead of spec 01's ⌊(3·O + 1)/2⌋. Everything the spec derives from D follows: the
history and display at O, the reduced history at ⌊O/2⌋ (and enc0's reading of it), dec0's and post's D/R factors,
motion scaled to O pixels. Step 6 is then `post_copy` (kernels/post.hip) instead of `post_filter`: per pixel
Dec(display) · dec_scale (the filter's decode and exposure handling, without the 5-tap resampling and clamp), into
the same output buffer; `scale_rgb` follows as before. `wg_rt_recon_size` returns the D in use. Default (option off):
unchanged, bit for bit. The interop's shared arena holds no D-sized buffer, so only the runtime's own history and
display buffers change size.

### Stage order per frame (one HIP stream)

0. Pre-exposure π (spec 10 draft 2; 1 when ≤ 0 or not finite): if π ≠ 1, `color_divide` (kernels/frame.hip) writes
   color / π (IEEE division per channel) into a runtime buffer that enc0 and post read; exposure reads the original
   color with π. After the filter, `scale_rgb` multiplies the output by π. Both are skipped when π = 1.
1. `exposure_partial` + `exposure_final` (auto-exposure only), then
   `exposure_pair(expo, computed, value,
   use_computed, first)`: e = computed or the given value, e_prev = e on the first frame else the kept value.
2. `input_encoder` (reads latent, motion history copy f, reduced history; writes m_near copy 1 − f).
3. enc1, enc2, b0 … b5, dec2, dec1 with the spec 02 offsets for parity f (launch plan as `stage_runner.py`, geometry
   from the build variant).
4. `coef_decoder` (features = dec1 output, motion = this frame's m_near, latent copy f → copy 1 − f).
5. `post_recon2_r_band` (spec 06; kernels/post.hip; the banded form of `post_recon2_r`, a bit-identical drop-in of
   `post_recon`): color, A, B, C, history copy f → copy 1 − f, this frame's m_near, m_center, m_far, model scalar s
   (post blob: s as f32 + 12 zero bytes) → reduced history and display; launched alternately with 6. over
   `post_bands` (4) horizontal bands (see "post (banded)" below). On a frame whose render footprint does not fit
   post_recon2's LDS staging (D < about 1.43 R): `post_recon`, then `post_filter_c1_s4` once. With a chosen post
   kernel, a chosen unbanded filter or recon=output: `post_recon2_r` and `post_filter_c1_s4` (the default before).
6. `post_filter_c1_s4_band` (bit-identical drop-in of `post_filter`, per band; unbanded `post_filter_c1_s4`: grid
   ⌈ow / 40⌉ · ⌈⌈oh / 2⌉ / 8⌉ workgroups of 64 lanes): display → the output buffer.

   Steps 5 and 6 together, by default on every frame of that banded pack12 path whose size it takes: one launch of
   the true-fusion kernel `post_tf_p12` (Quality ... Ultra Performance) or `post_tf_dl_p12` (native AA), `_f32_p12`
   forms for the 3 × f32 output; the display image is not written ("True fusion" below; bit-identical; runtime option
   `post_tf=0` / `WG_POST_TF=0` turns it off).
7. Frame state update.

### Kernel choice (A/B without a rebuild)

Each stage's kernel is chosen at create: `WgRtConfig` field, else environment variable, else the default (the
fastest exact variant). `wg_rt_kernels()` / `Pipeline.kernels()` report the choice and the post kernel of the last
frame.

| Stage | Field | Variable | Default | Values |
|---|---|---|---|---|
| Swin | `swin_code_object` (file) + geometry fields (`block_threads`, `block_wpg`, `wpw*`, `swin_force_clamp`) | `WG_SWIN_VARIANT` (name in `variants.json`, geometry from there); per stage: `swin_stage_variants` / `WG_SWIN_STAGE_VARIANTS` (`enc2=smax16,b0=gelu16`) | `swin.co` | `variants.py` names (profiling builds `prof*` refused) |
| enc0 | `enc0_variant` | `WG_ENC0_VARIANT` | `base` (`encoder.co`, A fragments from global memory) | `perf_variants.py` names → `encoder__<name>.co` (`lds` = previous default; `colorh`, `motionh`, `in16` refused: other input formats) |
| dec0 | `dec0_variant` | `WG_DEC0_VARIANT` | `base` (`decoder.co`, latent loads late, non-temporal A/B/C stores) | `temporal` = previous default, `early`, `fragglobal` (`v1`, `pack12` refused: other formats) |
| exposure | `exposure_variant` | `WG_EXPOSURE_VARIANT` | `base` (two kernels) | `v1` (`exposure__v1.co`) |
| post | `post_kernel` | `WG_POST_KERNEL` | `post_recon2_r_band_p12` (banded path with dec0 pack12, below), `post_recon2_r_band` without the pack12 decoder or with another dec0 / dec1 + dec0 choice, `post_recon2_r` with recon=output or an unbanded filter | any `post_recon*` kernel of post.hip with post_recon's arguments; `post_recon5*`, `post_recon6*` (after `post_prep`, record buffer of 32 bytes per render pixel allocated at create) |
| runtime options | `runtime_options` | `WG_RUNTIME_OPTIONS` (keys win; shared with the interop's options) | off | `graph=1`: each frame's launches captured into a HIP graph (updated in place while the topology is unchanged) and launched as one, not while the timing log records; `stream_priority=high`: the stream created with the highest priority |
| D = O output | (`recon_output` / `WG_RECON_SIZE=output`) | — | — | with the default post kernel, pi = 1 and an output buffer that takes RGBA16F (own buffer or `FrameIo.out_rgba16f`): `post_recon2_r_out` writes the output in one launch; otherwise `post_recon2_r` + `post_copy` (3 × f32) |
| post (banded) | `post_kernel` / `filter_kernel` | `WG_POST_KERNEL` / `WG_FILTER_KERNEL`; bands: runtime option `post_bands` or `WG_POST_BANDS` (default 4) | default (`post_recon2_r_band` + `post_filter_c1_s4_band`, 4 bands); `WG_POST_KERNEL=post_recon2_r` = unbanded | `post_recon2_r_band`, `post_recon2_r_hp_band` with `post_filter_c1_s4_band` (chosen automatically) or `post_filter_c1_s2_band`: reconstruction and filter launched alternately over horizontal bands (bit-identical) |
| post history format | (by kernel name) | — | 8-byte RGBA16F | `_h6` 6-byte RGB16F, `_hp` planar 6-byte (R, G plane + B plane), `_h4` compact; runtime-owned buffers (`Dn · 8` bytes fit every format); reported as `history_format` |
| dec1 + dec0 | `dec1_dec0` | `WG_DEC1_DEC0` | `separate` | `fused`: dec1's block with dec0 as its epilogue, `swin_block_c64_add_dec0[_nc]` from `dec1_dec0.co` (bit-identical; dec1's map not written, dec1's expand not fused) |
| dec0 pack12 + post _p12 | `post_pack` (or the post kernel's name) | `WG_POST_PACK=p12` / `none`; `WG_POST_KERNEL=post_recon2_r_p12` (`_hp_p12`, `_band_p12`, `_hp_band_p12`) | `p12` when `decoder__pack12.co` is built (every build; emulator: `wg_emu__pack12.so`) and post / dec0 / dec1 + dec0 are not chosen otherwise, not with recon=output | dec0 writes A, B, C as the 12-byte pack12 images (`decoder__pack12.co`, tagged module `d0p12`; A buffer 8 bytes per render pixel) and post reads them (bit-identical; in game with the banded path −0.04 ms); frames the _p12 kernels do not take (D < about 1.43 R) run `decoder.co` and `post_recon`; `dec0=pack12` alone and pack12 with `WG_DEC1_DEC0=fused` are refused |
| output copy (interop) | interop option `output_copy` | `WG_RUNTIME_OPTIONS=output_copy=0` | on (when the game's output texture is RGBA16F, single-sample 2D; else the output pass) | the last kernel writes RGBA16F rows (`FrameIo.out_half_pitch`): `post_filter_c1_s4_o16` with the default filter, π = 1 and rows of 8 · ow bytes, otherwise `post_filter_h` (π folded in), `post_copy_h` with recon=output; the banded post path is not taken (unbanded kernels); byte-identical to the FP16 rounding of the f32 output; reported as `filter_last` |
| post wave size | `post_wave` | `WG_POST_WAVE` | `64` (`post_w64.co`, when present) | `32`: the post and filter kernels from `post.co` |
| Swin bottleneck | `swin_bneck` | `WG_SWIN_BNECK` | `bmegahg` (with `swin.co`, when `swin__bmegahg.co` is built: every build) | a `bmega*` build: b0 … b5 as one `swin_bneck6[_nc]` launch (bit-identical to base; b0 … b4 outputs not kept), or `none`: six block launches |
| Swin mid (persistent) | `swin_mid` | `WG_SWIN_MID` | off | `1` (= `bmegahg`) or a `bmega*` build: enc2 + merge, b0 … b5, expand + dec2 as one `swin_mid[_nc]` launch (bit-identical; scratch maps `b5.out` and the bottleneck scratch; `enc2.merged`, b0 … b4 outputs not written); dec1 stays its own launch (fused expand or `dec1_dec0` as chosen); refused with `WG_SWIN_BNECK` or a per-stage variant of enc2 … dec2 |
| Swin fused stages | `swin_fuse_stages` | `WG_SWIN_FUSE_STAGES` | `dec1` (with `swin.co`) | encoder / decoder stages, e.g. `dec1,dec2`, or `none`: the stage's block kernel with the patch merge / expand fused in (`swin_block_*_mg` / `*_add_ex`, in `swin.co`), one launch |
| post (fused) | `post_kernel` | `WG_POST_KERNEL` | — | `post_fused*`, `post_fused3*`: reconstruction + filter in one launch (no display image; history 6 or 4 bytes, `_o16` RGBA16F output — refused for an external output unless `FrameIo.out_rgba16f`, and with pre-exposure ≠ 1) |
| filter | `filter_kernel` | `WG_FILTER_KERNEL` | `post_filter_c1_s4_band` (with the banded default; else `post_filter_c1_s4`) | `post_filter[_c1][_s4\|_s16]` |
| post (true fusion) | runtime options `post_tf`, `post_tf_cpairs` | `WG_POST_TF=0` (or `WG_RUNTIME_OPTIONS=post_tf=0`) | on: frames of the default banded pack12 path (post kernels in wave64, default filter) that `tf_supported` takes run one `post_tf_p12` / `post_tf_dl_p12` launch (interop output copy: RGBA16F rows, π folded in) or `post_tf_f32_p12` / `post_tf_dl_f32_p12` (3 × f32, then `scale_rgb` when π ≠ 1); C = 68 output row pairs per workgroup, 36 for the last dispatch round | `post_tf=0`: the banded sequence; `post_tf_cpairs=<even C>`; reported as `post_tf=<C>` and `post_last` |

Both spec 06 kernels take an extra last argument `expo`: a pointer to the exposure pair. When it is not null (the
runtime), they derive e, e_prev, e/e_prev and 0.01/e from it on the GPU (0 → 1, the quotients via binary64 then
binary32, as `runner/post.py` computes them on the host); when it is null (`Runner`), the parameters are used as
before.

### External inputs and output (interop)

`runtime_create(..., external_io = true)` (C++, `runtime_core.h`; test path `wg_rt_create_external` /
`wg_rt_frame_external`) makes a runtime without `input.color`, `input.depth`, `input.motion` and `filter.out`: every
frame passes them as device pointers (`FrameIo`) in the formats of the table above, plus the exposure source
(0 = the frame's value, 1 = the f32 at `expo_src`, 2 = compute).

## D3D12 interop (spec 10 sections 1, 6, 8)

Code: `clean/runtime/interop/`, part of `wronggpu_runtime.dll` (Windows). Export `wgGetUpscalerBackend(2)` returns the
backend of `clean/fsr/include/wronggpu_upscaler.h` interface version 2 (`backend_flags = WG_BACKEND_NEEDS_SPLIT`); it
needs the front end's `split_command_list` service and installs no hooks of its own.

### Files (installer contract, spec 30)

| What | Where |
|---|---|
| kernel code objects (`swin.co`, `encoder.co`, `decoder.co`, `exposure.co`, `post.co`, `frame.co`) | next to `wronggpu_runtime.dll` (option `kernels_dir`) |
| converter map `preset-M.v1.map.json` | next to the DLL, else in the model folder (option `map`) |
| `model.wgcm` | `%LOCALAPPDATA%\WrongGPU\model\` (option `model_dir`) |
| model K (runtime option `model=K`): `model.wgcm` of preset K, map `preset-K.v1.map.json`, code objects `k_block.co`, `k_block_wide.co`, `k_io.co` | `<model_dir>\K\model.wgcm` (option `model_k_dir`); map and code objects next to the DLL or built into it (option `model_k_map`); clean/k/FORMATS.md "Runtime frame" |
| HIP runtime `amdhip64_7.dll` | System32 (AMD driver), the standard search path, the newest `C:\Program Files\AMD\ROCm\*\bin`, `%HIP_PATH%\bin` (option `rocm_dir` is tried first) |

Options in `WgCreateInfo.options` (`key=value;...`, other keys ignored and logged), then, as a diagnostics override
whose keys win, the environment variable `WG_RUNTIME_OPTIONS` (same syntax; logged at create): `sync=inline|pipelined`,
`model_dir`, `kernels_dir`, `map`, `rocm_dir`, `swin` (code object of a Swin variant), `model_format=wgcm|auto|raw`,
`share_path=auto|arena|game|underlying|own` (see "Sharing paths"), `timing_log=1` (see "Timing log"),
`ready_wait=fence|memory` (default fence since 0.4.5, see "Timing log"), `done_mode=fence|spin` (experimental, see "Timing log"),
`watchdog=1` (diagnostics: a dispatch or network frame stuck for more than 3 s is logged with the fence values and flags),
`recon=spec|output` (experimental D = O, see "Per-frame runtime"; logged as `reconstruction size: D WxH (...)`),
`input_copy=1`, `output_copy=1` (copy paths, see "Copy paths"), `executor=auto|hip|dx12` (not named = auto: model M
on an RDNA 4 Radeon RX 9070 / 9060 runs on the D3D12 executor, anything else on HIP, one log line why;
runtime/interop/executor_select.h, clean/dx12/GAME_PATH_STATUS.md), and for offline tests `executor=d3d12-test`,
`test_view=color|depth|motion`, `test_handles=1`, `test_keep=1` (the test executor keeps its arena like an importer,
see "Context teardown and re-creation"). Other keys go to the network runtime's `runtime_options`, among them
`model=K` (model K instead of M; with `model_k_dir`, `model_k_map`, `network=fused|fused16`, `k_diag=input2|copyin` (timing diagnostics),
`k_input=16|f32` (model K's input pass, below); the create flag `WG_CREATE_HDR` reaches it as
`WgRtConfig.hdr`; timing log parts then named exposure, input, network, output).

### Shared resources (created at create on the game's device, imported into HIP)

Two slots (used alternately by the frames whose network is enqueued). All ten buffers (twelve with `motion_hi`) live in ONE committed buffer,
the arena (default heap, `D3D12_HEAP_FLAG_SHARED`, `ALLOW_UNORDERED_ACCESS`, state COMMON: implicit promotion and
decay at each ExecuteCommandLists, no barriers needed), in the order slot 0 color, depth, motion, expo, out, then
slot 1, each at a 64 KB-aligned offset. One `CreateSharedHandle`, one `hipImportExternalMemory` (D3D12 resource,
dedicated) mapped once; HIP uses base + offset, D3D12 views use FirstElement = offset / element size. The arena is at
least 32 MB: the game process refuses shared objects below about 8 MB (spec team's in-process probe), so it starts
at max(needed, 32 MB) and doubles up to 512 MB while CreateSharedHandle refuses it, each try logged. Only if that
fails come the per-buffer paths below (each buffer then its own shared resource of at least 32 MB). Nothing else in
the runtime is a shared D3D12 object except the three fences.

| Buffer | Size | Content |
|---|---|---|
| color | max R × 16 | `[rh][rw]` 4 × f32 (alpha 0), the game's color in the active region |
| depth | max R × 4 | `[rh][rw]` f32, the game's depth as given |
| motion | max R × 8 | `[rh][rw]` 2 × f32, m = motion_scale · raw + motion_offset (render pixels); with `WG_CREATE_MOTION_OUTPUT_RES` the texel ⌊(x + ½)·mw/rw⌋, ⌊(y + ½)·mh/rh⌋ (our point resampling: what the network reads with `motion_outres=point`; with `original` it reads `motion_hi`) |
| motion_hi | O × 4 + 8, or none | only with `WG_CREATE_MOTION_OUTPUT_RES` (model M; with `motion_outres=point` only while a frame capture is armed): `[oh][ow]` 2 × FP16, the raw motion texels of the output-size region ("Motion vectors at output resolution"). It follows `out` in each slot; without it every offset is what it was |
| expo | 256 bytes | f32 at offset 0: texel (0, 0) channel R of the exposure texture (exposure mode texture) |
| out | O × 12 | `[oh][ow]` 3 × f32, the network's result (the runtime's `filter.out` format) |

Fences (shared, `hipImportExternalSemaphore`, D3D12 fence): `ready` (the split request's done fence: the front end's
queue signals v after the first part of the game's list, which ends with the input conversion), `done` (HIP signals v
after frame v; the request's resume fence inline) and `pipe` (pipelined: HIP signals v when it reaches frame v's work,
i.e. once frame v − 1 is done; the request's resume fence there). Every split uses new, larger values, as the front
end's contract requires.

### Copy paths (options `input_copy=1`, `output_copy=1`; off by default)

Each frame checks the formats and logs the path when it changes (`frame N: input: ...`, `frame N: output: ...`).

- `input_copy=1`, when color is RGBA16F, depth R32F / D32F, motion RGBA16F or RG16F at render size (not
  `WG_CREATE_MOTION_OUTPUT_RES`) and an exposure texture, if used, R32F: instead of the input pass, `CopyTextureRegion`
  of the active region of each texture into the slot's buffers as rows with the 256-byte aligned footprint pitch
  (format from `GetCopyableFootprints`), textures to COPY_SOURCE and back. A depth-stencil depth texture (D32F, or
with the depth-stencil flag) can only be copied whole, so it takes the copy path only when the active region is the
whole texture; otherwise the frame uses the input pass (logged). The network's first kernel
  `input_convert` (kernels/frame.hip) then writes color / π (IEEE, replacing `color_divide`), dense depth and motion
  m = motion_scale · raw + motion_offset into the runtime's own buffers. Exposure (spec 07) then reads color / π with
  1/π = 1, so with π ≠ 1 the computed exposure can differ in the last bits from the default path (identical for
  π = 1, test `test_runtime_external.CopyLayoutTest`). Not combined with `ready_wait=memory`. Model K
  (`model=K`) reads the copies directly instead (no `input_convert`; clean/k/FORMATS.md "Runtime frame"): at 1440p
  Quality its input and output parts read half the colour and motion bytes (runtime test, one process: input
  0.309 → 0.251 ms, output 0.856 → 0.838 ms against the FP32 slot).
- Model K's compact input pass (default for `model=K` when no copy is made; `k_input=f32`: the FP32 input pass):
  where the game's colour is RGBA16F or R11G11B10 and its motion RG16F or RGBA16F (FP16-representable values), the
  pass `inputs16` writes dense RGBA16F colour (rw·8-byte rows, alpha 0), R32F depth and the raw RG16F motion texel
  (rw·4-byte rows; at output size the texel under the render pixel's centre, as the FP32 pass samples it) into the
  slot's buffers, and the network reads them as it reads the copies (motion_scale / motion_offset applied on load).
  Same output bits as the FP32 pass (`test_interop_d3d12 hipsame`, render- and output-size motion); logged as
  `input: model K compact pass (...)`. Interop soak, 1440p Quality, one machine state: D3D12 input pass 0.080 →
  0.072 ms, K's input part 0.427 → 0.378 ms, output part 1.350 → 1.334 ms, TOTAL(t3−t0) 2.97 → 2.89 ms. In game
  (Cyberpunk 2077 benchmark, 1440p Quality, `fused16`, the game's formats take the compact pass; the project's
  medians of ~30 timing windows, against `k_input=f32`): D3D12 input pass 0.055 → 0.038 ms, input 0.419 → 0.333,
  output 0.796 → 0.769, network GPU 1.713 → 1.588 ms, TOTAL(t3−t0) 1.933 → 1.796 ms, 126.4 → 129.3 fps.
- `output_copy=1`, when the output texture is RGBA16F: the network's last kernel writes RGBA16F rows of pitch
  ⌈8·ow / 256⌉·256 (`post_filter_h`, or `post_copy_h` with recon=output: the same values as the default path's
  output times π, rounded to FP16 with alpha 1; `scale_rgb` is not needed), and the second part of the game's list
  does one `CopyTextureRegion` into the output texture (to COPY_DEST and back) instead of the output pass. In
  pipelined mode the layout of the result read is that of the frame that wrote it.

Slot buffers are sized for both layouts (dense or pitched, whichever is larger).

### Sharing paths

After the arena (path 0, on the game device; `share_path=arena` forces it), for the per-buffer fallback. HIP imports
the buffers and fences through NT handles. The device the game hands to FSR can be a wrapper (an
interposer layer) that creates objects but fails `CreateSharedHandle` (seen in game: create failed). Tried in order,
each failure logged with its HRESULT, the path used logged at create:

1. game device: create and `CreateSharedHandle` on the given device;
2. underlying device: the device that an object created through the given device reports (`GetDevice`); share the
   objects created through the given device there, else create and share there;
3. own device: `D3D12CreateDevice` on the given device's adapter (`GetAdapterLuid` → `EnumAdapterByLuid`); create and
   share there and `OpenSharedHandle` on the given device for the game side (or, if that device is the underlying
   device itself, use the objects directly).

The game side (command lists, split request, the runtime's own queue) always uses objects valid on the given device;
HIP always imports the handles. At the first dispatch the log says which device the game's color texture belongs to. The network frame is enqueued at the dispatch call: HIP stream
waits `ready ≥ v`, runs the frame with `mv_scale = (1, 1)`, `mv_bias = 0` (already applied), the frame's jitter,
pre-exposure and exposure, then signals `done = v`. Values only grow.

### Per dispatch

1. Input pass (compute, recorded into the game's list): game textures transitioned to NON_PIXEL_SHADER_RESOURCE
   (unless already readable) and back to the given states.
2. `split_command_list(done = ready/v, resume = done/v)` (inline, default) or `resume = pipe/v` (pipelined). On an error (e.g. `WG_ERROR_NOT_READY`) the dispatch returns it; the next network frame is a reset.
3. Network enqueued (above). If it cannot be enqueued, the runtime's own compute queue waits for the previous `done`
   value and signals `done = v` (and `pipe = v`), so the game's queue cannot wait forever; the dispatch returns an error.
4. Output pass into the re-opened list: `out` of slot v (inline) or of the previous network frame (pipelined; the
   first frame: bilinear upscale of the color) → the output texture (alpha 1; sRGB formats encoded), output
   transitioned to UNORDERED_ACCESS if needed and back (else a UAV barrier). Shaders: HLSL compiled at create with
   `d3dcompiler_47.dll` (cs_5_0); the result buffer is read through an R32_FLOAT typed UAV.

### Context teardown and re-creation (2026-10-13)

A game destroys the upscaler context and creates a new one at every change of its upscale mode (some do it far more
often). What a context holds on the GPU, and what happens to it at destroy:

| What | At destroy |
|---|---|
| the network's own HIP allocations (weights, feature maps, histories; `runtime_memory_bytes`) | `hipFree`; returned to the system only with HIP's resource cache off (`GPU_RESOURCE_CACHE_SIZE=0`, which the front end sets, 0.5.1; see below) |
| the arena (shared D3D12 buffer) and its HIP import | **kept in the arena pool** for the next context, see below |
| the three shared fences and their HIP external semaphores | released (D3D12 `Release`, `hipDestroyExternalSemaphore`) |
| HIP stream, events, code-object modules | destroyed / unloaded |
| converter (root signature, six PSOs, descriptor heap), rescue queue, timing log | released |
| the HIP runtime library `amdhip64_7.dll` | **stays loaded** for the rest of the process |

Two properties of the HIP runtime (driver copy, 2026-10; RX 9070 XT) decide this, both measured with
`test_interop_d3d12 hipleak` and `hipMemGetInfo` around each teardown step:

0. (0.5.1) **HIP's resource cache keeps what `hipFree` gives back.** `hipMemGetInfo` counts it as free, but the
   process keeps it (DXGI local usage of the process): `hipleak` M/K, 1080p, after the first destroy +581 MiB, then up
   to the most ever in use at once; with `GPU_RESOURCE_CACHE_SIZE=0` +348 MiB (HIP's instance and the pooled arena)
   and +0 per cycle from then on. A host that switches upscalers (OptiScaler: the new context is created before the
   old one is destroyed) kept Model M + Model K + two arenas, 2.8 GB at 1440p, for the rest of the game. The front end
   sets `GPU_RESOURCE_CACHE_SIZE=0` before HIP loads (win_util.cpp `PrepareHipEnvironment`, unless the player set it).

1. **Unloading the HIP library does not return what its instance holds.** A context that loads `amdhip64_7.dll` and
   frees it at destroy (0.4.5-rc1) leaves that instance's GPU memory behind, and the next context's fresh instance
   allocates its own again: about 300 MiB of HIP's own device memory per instance plus whatever the instance had not
   returned, +390 to +440 MiB of process GPU memory per create/destroy cycle (below), +135 MiB of commit charge and
   +47 handles. The device therefore takes one extra, never released reference on the library once `hipInit` has
   succeeded (`pin_library`, device.cpp); the per-device reference stays balanced.
2. **Imported memory is not returned before the process ends.** `hipFree` of the mapped buffer and
   `hipDestroyExternalMemory` both succeed, and neither `hipMemGetInfo` nor the process's commit charge moves (in any
   order, with either call left out, with the import released after the stream, without a single kernel launch, with
   and without the dedicated flag). The import keeps the shared allocation alive after D3D12 has released the arena:
   with the library kept loaded, every context still lost exactly its arena (76 to 98 MiB at 1080p output, more at
   higher output sizes) and as much commit charge. D3D12 alone is clean (the same arena with the test executor:
   +0.0 MiB per cycle, reference count 0 at release).

**Arena pool** (backend.cpp `ArenaPool`, process-wide, mutex). A context whose executor imports the arena and can keep
its import (`Executor::import_scope` non-empty: HIP, the scope names the loaded HIP library and the HIP device) does
not release the arena at destroy; arena and import (`ArenaImport`: memory object and mapped pointer, valid in every
later HIP device object of the same scope) go to the pool. The next create takes the smallest free arena of the same
device and scope that is at least as large as it needs (logged `sharing: arena N MiB (M MiB used): kept from an
earlier context, with its import`; no `CreateSharedHandle`, no import). "The same device" is the device an object made
through the given device reports (`GetDevice`), so a wrapper that is made anew per context but hands out plain objects
still matches. If no free arena fits, all free ones are given up (`sharing: the kept arena of N MiB is released (too
small | another device | another importer; ...)`: D3D12 references released, the import released as far as HIP does)
and a new arena is made; it then serves the smaller sizes too. What the process can lose is therefore bounded by the
arenas it outgrows (e.g. Performance, then Balanced, then Quality: 76 + 86 MiB once), not by the number of contexts.
The pool holds a reference on the arena and its device until the process ends (a free arena stays allocated while no
context exists). The slot buffers of a reused arena hold the previous context's data; every buffer is written before
it is read, and the flag words are reset at create (`use_ready_flag`, `use_done_flag`). Not pooled: the per-buffer
sharing paths (no arena); their imports are released at destroy and a warning says how much memory the HIP runtime may
keep (`destroy: N MiB of shared buffers were imported for this context only ...`).

Teardown diagnostics in the log (through the front end's log callback): every HIP call of the teardown that returns
an error is a warning, one line per kind with its count (`destroy: HIP teardown: hipFree failed: HIP error 1 (...), 3
times`); one information line gives HIP's device memory in use before and after the teardown (`destroy: HIP teardown:
device memory in use 1027 MiB before, 573 MiB after (of 16304 MiB, every process on the GPU)`; the difference is what
this context returned, the game's own allocations in those milliseconds aside); `destroy: arena N MiB and its import
kept for the next context`.

Measured (`hipleak`: 12 cycles of create, 30 dispatches, destroy at 1280x720, 1114x626, 960x540 → 1920x1080 in one
process; process GPU memory = DXGI local segment usage after each destroy, growth per cycle over cycles 7 to 12; the
performance counter "GPU Process Memory \ Dedicated Usage" in brackets):

| | model M | model K | M and K alternating |
|---|---|---|---|
| 0.4.5-rc1 (4b6a984) | +440.8 MiB (+559.0) | +389.7 MiB (+507.9) | +415.2 MiB (+533.4) |
| HIP library kept loaded only | +84 to +105 MiB (the arena) | | |
| with the arena pool | +0.0 MiB | +5.0 MiB | +1.2 MiB |

The remaining growth is warm-up of HIP's allocator, not a per-context loss: steps of 5 to 34 MiB while the first ten
contexts come and go (model M 686 → 725 MiB by cycle 4, model K 616 → 679 MiB by cycle 10), then nothing: 36 cycles of
either model end with +0.0 MiB per cycle over cycles 11 to 36, constant HIP memory after every destroy, commit charge
within ±1 MiB per cycle and no handle growth. The test allows 16 MiB per cycle over the second half of its cycles. The
final frame of every cycle is bit-identical between 0.4.5-rc1 and the fixed runtime (36 checksums: M, K, alternating).

### Timing log (option `timing_log=1`, off by default)

Every 300 dispatches one INFO line through the front end's log:

| Field | Measured with | Meaning |
|---|---|---|
| network GPU [enc0, Swin, dec0, post] | HIP events on the network's stream (runtime marks: start, after enc0, after the Swin stages, after dec0, end) | GPU time of the network; enc0 includes exposure and the pre-exposure division, post includes the filter |
| D3D12 input pass, output pass | D3D12 timestamps in the game's list around each pass | the interop's copy/convert passes |
| input pass end -> resume | timestamp after the input pass (first part, front end's queue) to the first command of the second part (game's queue) | everything between the copies and the resume: ready signal, network, done signal, queue latencies; "> 0.1 ms in N of M frames" counts frames whose game queue waited; "of it not network" = this span minus the network's GPU time (the sync and scheduling overhead) |
| CPU dispatch (split, network enqueue) | QueryPerformanceCounter | CPU time of the dispatch call, of the split service and of enqueueing the network |

Stage events (HIP, in stream order): [0] start of the first kernel after the input wait (color_divide if π ≠ 1, else
exposure_partial with auto-exposure, else exposure_pair), [1] start of the first Swin kernel, [2] start of
coef_decoder, [3] start of post_recon, [4] end of an empty one-wave kernel (`timing_barrier`) after the last kernel.
enc0 = [0]-[1] (color_divide, exposure, exposure_pair, input_encoder; not the D3D12 input conversion, which is the
"input pass"), Swin = [1]-[2], dec0 = [2]-[3], post = [3]-[4]. The events are bound to these kernels with AMD's
`hipExtModuleLaunchKernel` (start / stop event of one kernel): plain events recorded between launches moved the
enc0/Swin boundary by up to 0.5 ms in game. Without that function the runtime falls back to plain events and says so.
The create line and the first summary print this order.

`ready_wait=memory` (the default in 0.4.0 and 0.4.1; an explicit option since 0.4.5, because the wait stalled the GPU for
minutes in several games: reports of 2026-10-10 to 2026-10-13 and the Crimson Desert start-up stall of
`results/2026-10-10-crimson-fence-profile.md`): after the input pass a UAV barrier and a one-thread pass write v (u32)
to a flag in its own 64 KB block at the end of the arena, and the HIP stream waits with `hipStreamWaitValue32(flag >= v)`
instead of the shared `ready` fence (the fence is still signalled for the front end). Arena only, and only with
`hipStreamWaitValue32`; otherwise the fence (`ready_wait=fence`). Measured (inc-stack3 kernels, Cyberpunk 2077 benchmark,
1440p Quality, three interleaved runs each): not-network gap 0.162 -> 0.092 ms, 115.6 -> 117.3 fps; output identical
(`hipsame`). Offline qualification: soaks with idle gaps, back to back, re-create and pipelined, no hang.

Experimental `done_mode=spin` (inline sync with `ready_wait=memory` only): the HIP stream writes v to a second flag (the
next 64 KB block) after the network, and instead of waiting for the `done` fence the game's queue runs a one-thread pass
that polls that flag with an atomic (at most `spin_polls`, default 400000, polls; a timeout is counted in the word
after the flag), followed by a UAV barrier. The split's resume fence is then the `ready` fence the first part has
already signalled. If the network cannot be enqueued, the rescue queue writes the flag. It removes the gap almost
entirely (not-network 0.003 ms in game) but the game is slower with it (113.1 fps against 117.3 with the default):
the polling pass on the game's queue costs more than the hop it saves. Off by default.

Nothing waits: timestamps are read from a readback ring once the `ready` fence shows that a later frame's first part
has run, HIP events once they report completion; frames whose times are not ready in time are left out (the counts
say how many were used). The two timestamps of "input pass end -> resume" come from two queues of the same GPU; their
clocks are assumed to agree.

## Motion vectors at output resolution (spec 03 draft 4, spec 06 section 3; create flag `WG_CREATE_MOTION_OUTPUT_RES`)

Some games give their motion vectors at the output size M = O instead of the render size. The network then reads
the game's motion itself: one texel per render pixel in the input encoder (no depth neighbourhood; the three motion
surfaces receive the same value) and one texel per reconstruction pixel in the reconstruction (the three surfaces are
not read; one history sample). Reference: `clean/reference/motion_hi.py` (the FP32 rules), `encoder.py` /
`post.py` with `x["motion_output_res"]`. Results: `results/2026-10-13-motion-output-resolution.md`.

Selection (model M): interop option / runtime option `motion_outres=original|point`. `original` is the default when
the context has the create flag. `point` is our earlier stand-in, kept for A/B in a game: the D3D12 input pass
resamples the motion to render pixels (the texel ⌊(x + ½)·mw/rw⌋, ⌊(y + ½)·mh/rh⌋ under each render pixel's centre)
and the frame runs the render-resolution kernels. Without the create flag the runtime gets no such option, loads
nothing for it and every frame is what it was (same code objects, same launches). `wg_rt_kernels` reports
`motion_outres=original|point|off` and `motion_last=output|render` (what the last frame ran). Frames that the
kernels below do not take (`recon2_supported` false, raw-input copies, a post or enc0 kernel chosen by hand, the
emulator backend, model K) run on the render-size motion as with `point`; create says so in a note when that is the
case for the whole context.

### The raw motion buffer (`kernels/wg_motion_hi.h`)

The motion texels of the region M, with the subrect origin already applied by whoever fills it, as rows of `pitch`
bytes. A texel is two FP16 (x, y; 4 bytes) or two f32 (8 bytes; the test path), raw game values: no scale, no
offset. At least 4 spare bytes follow the last texel (both formats are read as two 32-bit words and selected by a
uniform flag, no branch per pixel).

Scale: s = RN32(mv_scale / RN32(M / D)) per axis with mv_scale converting the raw motion to pixels of M, and b_mv
added to the raw motion first; the host evaluates s, k_M = RN32(M / R) and RN32(1 / D) in FP32 (correctly rounded
host operations) and passes them by value.

### Input encoder: `input_encoder_mo` (`kernels/encoder_mo.hip` → `encoder_mo.co`, tag `emo`)

`encoder.hip` included with its kernel made an internal function; only the motions differ
(`encoder_motion_outres.h`): T = clamp(⌊((x + ½) − jx)·k_Mx⌋, 0, Mw − 1) (likewise y; a NaN coordinate reads texel
0), m = sx·(bx + MV(T)) with one FP32 rounding per operation, m_near = m_center = m_far = m. Arguments: as
`input_encoder` without `depth`, `motion` = the raw motion buffer, then after `WgEncParams` (whose `sx, sy` are s and
`bx, by` are b_mv here; `depth_inverted` unused) `struct WgEncMoParams`: int `mw, mh, pitch, wide` (0 = FP16
texels, 1 = f32), float `kmx, kmy`. 24 bytes. Same grid and workgroup size. 77 VGPRs, 16 waves per SIMD.

### Reconstruction: `post_tf_p12_mo`, `post_tf_f32_p12_mo`, `post_recon2_r_p12_mo` (`kernels/post_mo.hip` → `post_mo_w64.co`, `post_mo_w64__q5.co`, `post_mo_w64__orig.co`, tag `pmo`)

`post.hip` included with its kernels made internal functions; gather, history sample, blend, outputs and output
filter are that file's, the motion is `post_motion_outres.h`:

- T = clamp(trunc(RN32(RN32((x + ½) / dw)·Mw)), 0, Mw − 1), likewise y. The division is exact: from the correctly
  rounded reciprocal with one correction step (`mo_center`), checked against the division for every pixel of every
  size up to 12000.
- q = (fma(sx, bx + MVx(T), x + ½), fma(sy, by + MVy(T), y + ½)); valid if not reset and inside [0, D] (inclusive).
- One sample S (the history at q, or the fallback), H = fma(w1, S, w2·S) with the two weights of `post_select`
  (w2·S counts as 0 when w2 = 0).
- A NaN position follows spec 06 draft 4 section 4 (valid, NaN sample, both results C_lin): the blend runs with
  1 − β′ = 0, which gives C_lin exactly. (The render-resolution kernels still treat a NaN position as invalid.)
- The two texels of a lane's row pair are loaded at the start of a step (`mo_fetch`), before staging and gather, so
  that the loads overlap that work; their values are used after the gather. (Loading them where they are used cost
  0.18 ms per 1440p frame.)

`struct WgMoParams` (by value, 40 bytes): int `mw, mh, pitch, wide`; float `sx, sy, bx, by, idw, idh`
(RN32(1 / dw), RN32(1 / dh)).

| Kernel | Stands in for | Arguments |
|---|---|---|
| `post_tf_p12_mo` / `post_tf_f32_p12_mo` | `post_tf_p12_diet_vsit` / `post_tf_f32_p12_diet_vsit` (true fusion; needs `tf_supported` at cap 352 and `post_diet`) | as those, with `m_near, m_center, m_far` replaced by `m_near` (one render-size surface, staged but not used) and `motion`; `WgMoParams` after `c2`, before the weights of the `original` filter build |
| `post_recon2_r_p12_mo` | every other frame of the path (`recon2_supported`); the unbanded filter follows (`post_filter_c1_s4`, `_o16` / `post_filter_h` for the output copy, `post_filter_quality9`) | as `post_recon2_r_p12` with the three surfaces replaced by `motion`; `WgMoParams` after `expo` |
| `post_mo_probe` | test only | `motion, out, dw, dh, WgMoParams`: per D pixel tx, ty, bits of qx, bits of qy |

Resources (gate `POST_MO_GATE` in `clean/runner/build.py`): `post_tf_*_mo` 80 VGPRs, 30336 bytes of LDS, 8 waves64
per SIMD (the kernels they stand in for: 94 to 95 VGPRs); `post_recon2_r_p12_mo` 81 VGPRs, 16128 bytes, 8 waves64.
Native AA (D = 1.5 R, `post_tf_dl*`) and the frames between the two true-fusion caps take the unfused pair for now
(about 0.13 ms more per 1440p frame than the fused kernel).

`build.py` builds and embeds the four code objects for gfx1201 and gfx1200 (`--motion-outres-only` builds only
them). The hooks they need in `encoder.hip` and `post.hip` (a template argument `MoT`, off by default) leave the
machine code of every existing kernel as it was (`tools/isa_same.py`: 93 of 93 kernels of each post code object and
the encoder, both targets).

### Runtime and interop

- `FrameIo` (`runtime_core.h`): `motion_hi`, `motion_hi_pitch`, `motion_hi_wide`, `motion_hi_w / _h`, `mv_scale_hi`,
  `b_mv`. Test path: `wg_rt_upload_motion_hi(rt, mw, mh, wide, texels, mv_scale, b_mv)` (own buffer, used by every
  later frame; `mw = 0` switches back); `pipeline.py`: `x["motion_hi"]` (float32 or float16 `[mh][mw][2]`) with
  `kernels={"options": "motion_outres=original"}`.
- Interop: with the flag (model M) each slot gets a sixth buffer `motion_hi` after `out`: O × 4 + 8 bytes, the raw
  motion texels as two FP16, written by the D3D12 pass `motionhi` (typed RG16F buffer UAV; the texture's texel
  (0, 0) is the origin, the interface has no motion subrect; a texture smaller than O repeats its last row and
  column). Exact for the FP16 motion formats (RG16F, RGBA16F); another format is rounded to FP16 by the store and the
  log says so once. Cost: 2 × 14.1 MiB of the arena at 2560×1440 (2 × 31.6 MiB at 3840×2160), and about 0.06 ms of
  D3D12 time per frame at 1440p (input pass 0.15 → 0.21 ms). The render-size `motion` buffer is still filled by the
  input pass as before (the sparkle guard and the frames named above read it). With `motion_outres=point` the
  buffer exists and is filled only while a frame capture is armed.
- `mv_scale_hi` = `motion_scale` · (O / R) per axis (`WgFrame.motion_scale` converts to render pixels, the network
  wants pixels of M), b_mv = `motion_offset` / `motion_scale` (0 when the offset is 0).
- Frame capture (`capture_dir`): with the flag a sixth file `<base>_motion_hi.bin` holds the raw motion
  (`[oh][ow]` 2 × FP16) next to `<base>_motion.bin` (the render-size motion, i.e. the stand-in's input), and the
  frame's line in `frames.txt` ends with `motion_outres=original|point motion_hi=<w>x<h> motion_hi_pitch=<bytes>
  mv_scale_hi=<x>,<y> b_mv=<x>,<y>`. A captured frame can so be replayed through both paths whichever one ran.

## Output reconstruction and output filter (spec 06: stages post, filter)

Producer of the weight blob: `clean/runner/post.py` (`pack`); input conversion and launch plans: same file. Kernels:
`clean/kernels/post.hip`. Sizes: R = (rw, rh), D = (dw, dh), O = (ow, oh); every image is row-major, row 0 first.

### Weight blob (stage post)

| Offset | Content |
|---|---|
| 0 | s (f32) |
| 4 | zero padding to 16 bytes |

The filter has no weights.

### Images

| Image | Size | Element (little-endian) | Written by | Read by |
|---|---|---|---|---|
| colour | R | 4 x f32 (R, G, B, alpha ignored) | game / runner | enc0, post |
| A, B (spec 05) | R | 4 x u8 codes k (value k / 255), channels 0..2, byte 3 ignored | dec0 / runner | post |
| C (spec 05) | R | 4 x f16 (a, b, c, unused) | dec0 / runner | post |
| m_near, m_center, m_far | R | 2 x f16 (x, y), D pixels | enc0 / runner | post |
| full history | D | 4 x f16: E-encoded R, G, B, alpha 0 | post (frame n) | post (frame n + 1) |
| reduced history | floor(D / 2) | u32 R10G10B10A2: R in bits 0..9, G 10..19, B 20..29, alpha 0 | post | enc0 (next frame) |
| display image | D | three planes of f16 (all of R, then G, then B), E-encoded and clamped to [0, 2.45] (spec 06 draft 2); row pitch P = dw rounded up to a multiple of 64 elements, plane size P · dh, buffer 6 · P · dh bytes (`display_bytes` in `runner/post.py`); padding columns unused | post | filter |
| output | O | 3 x f32 (R, G, B), linear | filter | runner / game |

The full and reduced histories are double-buffered by the runtime (spec 01): `post_recon` reads the previous full
history and writes the new one to a different buffer. Reduced-history blocks that are incomplete (odd D) are not
written.

### Kernels and launch geometry

1-D grids; `post_recon` uses 256-lane workgroups, `post_filter` 64-lane ones.

| Kernel | Arguments | Work | Grid |
|---|---|---|---|
| `post_recon` | color, A, B, C, hist_prev, m_near, m_center, m_far, w, hist_out, reduced, display, params | 32 D columns x 8 row pairs per workgroup; lane = one column of one row pair | ceil(dw / 32) x ceil(ceil(dh / 2) / 8) |
| `post_filter` | display, out, dw, dh, ow, oh, 0.01 / e | 64-lane workgroups: 128 output columns x 2 strips of S row pairs; lane = 4 columns x 2S rows, S = 8 | ceil(ow / 128) x ceil(ceil(oh / 2) / (2 S)) |
| `post_filter_s4`, `post_filter_s16` | same | timing variants with S = 4 and 16 (identical output) | same with their S |

`params` is a 64-byte struct passed by value (`WgPostParams` in the kernel, `PostParams` in `runner/post.py`):
int rw, rh, dw, dh, reset, 3 x padding; float R/D per axis, jitter x, y, e, e_prev (0 replaced by 1), e / e_prev,
padding. `post_recon` stages the render footprint of its tile in LDS (32 bytes per render pixel, at most 512 pixels:
19.5 KB per workgroup); a footprint larger than that (D smaller than about 1.5 R) makes the workgroup read its taps
from global memory instead. `post_filter` reads the 10 input columns 6m - 2 ... 6m + 7 of a lane as 5 aligned dwords
per plane (hence the even row pitch); lanes at the left and right image edges read clamped 16-bit elements.

### Drop-in variants (same arguments, layouts and grid as the kernel they replace; bit-identical results)

| Kernel | Replaces | Difference |
|---|---|---|
| `post_recon2` | `post_recon` | render features (codes as floats) and the three motions staged in LDS; reduced-history means by lane exchange; one pixel at a time; history taps as 16-byte spans (half the loads) |
| `post_recon2_r` | `post_recon` | as `post_recon2`, 3 x 3 gather rows in a loop |
| `post_recon2_ns`, `post_recon2_r_ns` | `post_recon` | as above, history taps loaded one by one |
| `post_filter_c1` (S = 8), `_c1_s4`, `_c1_s16` | `post_filter` | one channel per lane (7 x 4 window instead of 7 x 12: fewer registers, more waves); grid `ceil(ow / 40) * ceil(ceil(oh / 2) / (2 S))` workgroups of 64 lanes |

| `post_recon2_r_h6` | `post_recon` | as `post_recon2_r` with the 6-byte full history (RGB16F tightly packed, as `post_fused`: `hist_prev` / `hist_out` hold `3 · (y · dw + x)` halves, `6 · dw · dh + 8` bytes; the runtime's `Dn · 8`-byte buffers are large enough); same values, bit-identical. Selectable by name in the runtime as it is (same arguments, grid and recon2 path); switching between history formats needs a reset frame |
| `post_recon2_r_hp` | `post_recon` | as `post_recon2_r` with the planar 6-byte full history: plane 0 = R, G as two FP16 per pixel at byte `4 · (y · dw + x)`, plane 1 = B as FP16 at byte `4 · dw · dh + 2 · (y · dw + x)`; buffer `6 · dw · dh + 8` bytes (fits the runtime's `Dn · 8`). Every history access is dword- or half-aligned (spans: aligned b128 / b64 loads). Same values, bit-identical. Selectable by name like `post_recon2_r`; switching history formats needs a reset frame. `post_recon2_r_hp_band`: the same for the banded path |
| `post_recon3` | `post_recon` | as `post_recon2_r`, plus the raw colour staged (fallback without a global read) and both candidates' history loads issued before either is used (one memory round trip per pixel); bit-identical |
| `post_recon3_h` | `post_recon` | as `post_recon3`, history tap weights in packed FP16, 1 / norm without `v_rcp`, the Gaussian's a u² term shared by the two rows; within spec 09 (≤ 1 FP16 step from `post_recon`) |

`post_recon2*` (and `post_recon3*`) need `recon2_supported`: D >= about 1.43 R per axis (every FSR mode) and D at
least 4 columns wide (`dw >= 4`, i.e. an output of 3 pixels or more; since clean/post-r1-latency the `post_recon2*`
kernels always read the history taps as spans, `history_cr_span`, and return at once for `dw < 4`). Since clean/post-r1-math
`recon2_supported` also requires the sizes to fit the 32-bit byte offsets of `post_recon2_body` (`offsets32_ok`:
8 dw dh, 6 display_pitch(dw) dh and 16 rw rh at most 2^32 bytes; 16 display_pitch(dw) and 16 rw below 2^24), true
far beyond 8K output. Frames outside it run `post_recon` (bit-identical). LDS per workgroup of `post_recon2*` (since clean/post-r1-latency): 336 texels of
48 bytes = 16128 bytes (8 workgroups of 256 lanes per WGP: 8 waves64 per SIMD; before: 20160 bytes, 6 waves64), one
16-byte-aligned record per staged render pixel (`R2Texel`):

| Bytes | Field | Type |
|---|---|---|
| 0 - 11 | E_in(e · colour) R, G, B | f32 x 3 |
| 12 - 23 | kernel shape C = (a, b, c) | f32 x 3 |
| 24 - 29 | B codes σ_near, σ_center, σ_far | FP16 x 3 (integers 0 ... 255, exact) |
| 30 - 35 | A codes g, 255 − β_h, 255 − β_o | FP16 x 3 (integers 0 ... 255, exact) |
| 36 - 47 | m_near, m_center, m_far as read (FP16 x, y each) | u32 x 3 |

Filter output in FP16: `post_filter_c1_s4_o16` (arguments and grid as `post_filter_c1_s4`) writes `out` as RGBA16F
(R, G, B, 1), 8 bytes per pixel, row pitch `8 · ow`: the FP16 rounding (nearest even) of `post_filter`'s f32 output,
the layout of an R16G16B16A16_FLOAT image. The runtime needs to know the name (its filter list) and that `out` is
RGBA16F.

Wave64: `post_w64.co` (`runner/post.py: build_wave64`) is the whole file built with `-mwavefrontsize64`; every kernel
has the same name, arguments and results there. (A single wave64 kernel in `post.co` is not possible: the compiler
refuses to inline the shared wave32 helpers into it.)

### dec0 "pack12" coefficients (`post_recon2_r_p12`, `_hp_p12`, `_band_p12`, `_hp_band_p12`; exact)

The reconstruction reading dec0's `pack12` output (dec0 variant on branch clean/perf-enc, 469572c): `A` = image A8,
8 bytes per render pixel (codes A0 A1 A2 B0 | B1 B2, then C2 as FP16), `C` = image C4, 4 bytes per render pixel
(C0, C1 as FP16), `B` unused (may be null). 12 instead of 16 bytes per render pixel (-6.6 MB at 1706x960 for each of
dec0's write and post's read). Same values as the padded layout, bit-identical. Arguments as `post_recon2_r` plus
`int tile0` after `expo` (0 for the unbanded kernels; the first tile of the band for the `_band` kernels). `_hp`: planar
6-byte history. The runtime switches dec0 and post together.

### Banded two-kernel path (`post_recon2_r_band` + `post_filter_c1_s4_band` / `_s2_band`; exact)

The in-game pair `post_recon2_r` + `post_filter_c1_s4`, launched alternately over n horizontal bands so that each band
of the display image (50 MB at 3840x2160) is still in L2 / the last-level cache when the filter reads it. Bit-identical
to the unbanded pair (same kernels' arithmetic). Arguments: as `post_recon2_r` / `post_filter_c1_s4`, each followed by
`int tile0` (the first workgroup tile of the launch); the grid of a launch is its tile count.

Launch sequence (`runner/post.py: band_plan(dw, dh, ow, oh, n, S)`; S = 4 for `_s4_band`, 2 for `_s2_band`):

```
TX = ceil(dw / 32), RR = ceil(ceil(dh / 2) / 8)            // recon tiles per row, recon tile rows (16 D rows each)
FX = ceil(ow / 40), FR = ceil(ceil(oh / 2) / (2 S))        // filter tiles per row, filter tile rows (2 S row pairs)
fdone = 0
for b in 0 .. n - 1:                                       // n <= RR
    r0 = b RR / n, r1 = (b + 1) RR / n                     // integer division
    launch post_recon2_r_band, grid (r1 - r0) TX, 256 lanes, tile0 = r0 TX
    d_done = min(16 r1, dh)                                // D rows 0 ... d_done - 1 complete
    f1 = fdone
    while f1 < FR and min(3 (min((f1 + 1) 2 S, ceil(oh / 2)) - 1) + 4, dh - 1) < d_done: f1 += 1
    if b == n - 1: f1 = FR
    if f1 > fdone: launch post_filter_c1_s<S>_band, grid (f1 - fdone) FX, 64 lanes, tile0 = fdone FX; fdone = f1
```

All launches on one stream, in this order (each filter launch reads display rows written by the reconstruction
launches before it). `post_traffic.py` times n = 2, 4, 8, 16, cold, in wave32 and wave64 (`post_w64.co` has the same
kernels).

### True fusion: reconstruction + output filter in one kernel (`post_tf*`; runtime default where it applies; exact)

One launch replaces the banded `post_recon2_r_band_p12` + `post_filter_c1_s4(_h)_band` sequence; the display image
never goes to memory. Bit-identical to that sequence (same functions and operation order as `post_recon2_body`,
`filter_row_c1`, `filter_strip_c1`; `runner/post_exact.py` on the GPU, `runner/post_tf_runtime.py` through the runtime).

| Kernel | Output | Staging cap (render pixels per step) |
|---|---|---|
| `post_tf_p12` | RGBA16F rows of `pitch_bytes`, value `(v · dec_scale) · π` rounded to FP16, alpha 1 (as `post_filter_c1_s4_h_band`) | 352 |
| `post_tf_f32_p12` | 3 × f32 rows of `12 · ow` bytes (as `post_filter_c1_s4_band`; π and `pitch_bytes` unused) | 352 |
| `post_tf_dl_p12`, `post_tf_dl_f32_p12` | as above | 644 (native AA, D = 1.5 R; 2 texels per lane) |
| `post_tf4_p12`, `post_tf16_p12` | as `post_tf_p12`, NW = 4 / 16 row pairs per step (experiments, slower) | 224 / 576 |

Arguments: `color, A (pack12 A8), B (unused), C (pack12 C4), hist_prev, m_near, m_center, m_far, w, hist_out, reduced,
out, params (WgPostParams), const float* expo, int ow, int oh, float dec_scale, float pi, int pitch_bytes, int cpairs,
int k1, int c2`.
`expo` (the exposure pair) overrides e, e_prev, e / e_prev and dec_scale as for the other kernels. Full history 8 bytes
(RGBA16F, alpha 0) and reduced history exactly as `post_recon2_r_p12`.

Grid: `ceil(ow / 40) · chunks` workgroups of `64 · NW` lanes (NW = 8: 512 lanes, 8 waves64), workgroup `g` = strip
`g mod ceil(ow / 40)`, chunk `t = g div ceil(ow / 40)`. Chunk t has `C_t` = `cpairs` output row pairs for t < `k1`, else
`c2` (both even), starting at pair `P0 = t · cpairs` (t < k1) or `k1 · cpairs + (t - k1) · c2`; chunks = k1 + ⌈(⌈oh / 2⌉ −
k1 · cpairs) / c2⌉. The runtime (and `post_sizes.tf_chunks`): cpairs = 68 (the 3 C + 4 D rows of a chunk are 13 full
steps of 16), c2 = min(36, cpairs) for the chunk rows dispatched last, about one round of resident workgroups
(m = ⌈128 / strips⌉ chunk rows of c2; k1 = ⌊(⌈oh / 2⌉ − m c2) / cpairs⌋): smaller workgroups at the end shorten the tail of
the launch. Strip s: output columns 40 s ... 40 s + 39 (filter cells m = 10 s ... 10 s + 9), owned D columns
X0 = 60 s ... X0 + 59, computed D columns X0 - 2 ... X0 + 61 (lane = column; columns outside D are clamped duplicates,
computed and never stored). Chunk: output row pairs P0 ... P0 + C_t - 1, owned D rows 3 P0 ... 3 (P0 + C_t) - 1, computed
D rows max(3 P0 - 2, 0) ... min(3 (P0 + C_t) + 1, dh - 1) (the last pair of the image limits both;
`tests/test_post_tf_geometry.py` checks coverage and the ring). Per step of NW row pairs (wave w = row pair w):

```
put the staged texels (loaded during the previous step) into LDS; barrier
reconstruct the wave's row pair (post_recon2_body arithmetic); owned pixels store the full and reduced history
write the 2 x 64 display values (FP16, [row][channel][64 columns]) into the wave's two ring slots
issue the next step's staging loads (registers; written to LDS at the next step's start)
wave-local exchange; horizontal pass (lane = row, channel, cell of 4 output columns: filter_row_c1); the wave's two
  ring slots become 2 x [3 channels][40 output columns] FP32; barrier
vertical pass + Dec + store for every output pair p whose rows 3p - 2 ... 3p + 4 (clamped to D) are complete
  (lane = output pixel, 5 ring rows x 3 channels)
```

LDS (NW = 8): staged texels `352 · 48` bytes (`R2Texel`) + the ring of `2 NW + 6 = 22` rows (row `r` in slot
`(r - first row) mod 22`) of 480 bytes, plus 6 mirrored rows (slots 0 ... 5 written again at 22 ... 27, so 5 consecutive
slots never wrap): 30336 bytes, 4 workgroups per WGP; 95-96 VGPRs: 8 waves64 per SIMD (`post_tf_dl*`: 41472 bytes, no
mirror, 6 waves). Precondition (`tf_supported`, runtime.cpp and runner/post.py in step): `recon2_supported` and
`(floor(63 rw / dw) + 4) · (floor(14 rh / dh) + 5) <= cap`.

### Output mode D = O (runtime option `recon = output`, `WG_RECON_SIZE=output`; experimental)

Reconstruction at D = O (outside spec 06, where D = ⌊(3·O + 1) / 2⌋): the output filter becomes the identity, so the
reconstruction writes the final output itself; no display image, no filter pass. Exact with respect to
"reconstruct at D = O, then the identity filter": the output value is the FP16 display value (spec 06) decoded as the
filter decodes a filtered value, `FP16(Dec(clamp(E, 1e-4, 3)) · 0.01 / e)` (round to nearest even), alpha 1.

| Kernel | Use |
|---|---|
| `post_recon2_r_out` | the reconstruction for this mode: `post_recon2_r` writing the output instead of the display image; bit-identical to `post_recon2_r` + `post_decode_o16` |
| `post_decode_o16` | the identity filter as a kernel (display planes → output); for D = O with a kernel that writes the display image (e.g. `post_recon` where `recon2_supported` fails), and as the test reference |

`post_recon2_r_out` arguments (as `post_recon2_r`, `display` replaced by `out`, `dec_scale` after the params):

```
post_recon2_r_out(const float4* color, const u32* A, const u32* B, const half4* C, const half4* hist_prev,
                  const u32* m_near, const u32* m_center, const u32* m_far, const float* w, half4* hist_out,
                  u32* reduced, void* out, WgPostParams p, float dec_scale, const float* expo)
grid recon_grid(dw, dh) = ceil(dw / 32) * ceil(ceil(dh / 2) / 8) workgroups of 256 lanes
```

- `p` from `post_params` with `recon_size = O` (so `dw = ow`, `dh = oh`, `rx = rw / ow`, ...); full history (8 bytes
  per pixel, as `post_recon2_r`) and reduced history (`(dw / 2) · (dh / 2)` words) at that size, unchanged otherwise.
- `out`: `ow · oh` pixels RGBA16F, row pitch `8 · ow` bytes, alpha 1 (the layout of an R16G16B16A16_FLOAT image).
- `dec_scale = 0.01 / e`; with `expo` not null both the exposures and `dec_scale` come from the exposure pair in GPU
  memory (`dec_scale = 0.01 / e`, as `post_filter`).
- Precondition `recon2_supported(rw, rh, ow, oh)` (D ≥ about 1.43 R per axis): true for FSR Quality (O = 1.5 R, the
  staging area is then exactly full) and the lower-resolution modes, false for Ultra Quality (1.3 R) and native. Then
  use `post_recon` (display image at D = O) followed by `post_decode_o16`.
- `post_decode_o16(const half* display, void* out, int dw, int dh, float dec_scale, const float* expo)`, grid
  `ceil(dw · dh / 256)` x 256 lanes; display planes as for `post_filter` (pitch `display_pitch(dw)`).
- Wave64: both kernels are also in `post_w64.co`.

### Pre-pass + reconstruction (`post_prep` + `post_recon5`)

`post_prep` (render resolution, one lane per sample, grid `ceil(rw * rh / 256)` x 256 lanes; arguments `color, A, B,
C, prep, params, expo`) writes one 32-byte record per render sample to `prep` (`rw * rh * 32` bytes): f32 E_in(e·colour)
R, G, B; then FP16 pairs (a, b), (c, σ_near code), (σ_center, σ_far), (g, 255 − β_h), (255 − β_o, 0). `post_recon5`
(arguments `prep, color, hist_prev, m_near, m_center, m_far, w, hist_out, reduced, display, params, expo`; grid and
outputs as `post_recon`) stages the records instead of decoding colour and coefficients itself; bit-identical to
`post_recon`. `post_recon5_bil`: the same with bilinear history (approximation). Both need `recon2_supported`.

`post_recon6` (same pre-pass, arguments, grid and outputs as `post_recon5`; bit-identical to `post_recon`) cuts the
per-pixel head: motions staged as FP32 pairs, the motion reads and the m_center bilinear of a lane's two pixels done
once (3 x-interpolated motion rows instead of 4), the candidate choice by selects on the motions instead of candidate
indices, validity tests without branches, 24-bit address multiplies, and unclamped LDS indices. It therefore needs
`recon6_supported`: `recon2_supported` and R < D on both axes (every upscaling mode); `prepare_post` refuses other
sizes. 28 KB LDS per workgroup. Switches (speed approximations, whole-pipeline rule):

| Kernel | Approximation |
|---|---|
| `post_recon6_bil` | bilinear history (as `post_recon4_bil`) |
| `post_recon6_eb` | step 5 blend done in the E domain: E^ = H_E + (C_E − H_E) β′ with C_E = clamp(c̄, 1e-4, 3), H_E = clamp(H, 1e-4, 3) (E(Dec(H)·e/e_prev) when the exposure changed); history and display written from E^ directly (no Dec, no E per pixel); the reduced history keeps the exact formula E(mean of Dec(E^)) |
| `post_recon6_eb_bil` | both |

The 24-bit multiplies in the shared history fetch (`cr_spans`, `cr_edge`, `bi_load`, `bi_edge`) also speed up
`post_recon3*`, `post_recon4_*` and `post_recon5*`; integer results are unchanged (all factors below 2^24).

### Speed approximations of `post_recon` (`post_recon4_*`, not spec 06-exact)

Same arguments, layouts and grid as `post_recon` (drop-in), built on `post_recon3`; judged by the whole-pipeline
rule of spec 09 draft 2 (>= 48 dB against the exact runtime, stability and edge sharpness within 2 %), not by the
per-stage rule. `runner/post.py: RECON4_VARIANTS` lists them with what each approximates:

| Kernel | Approximation |
|---|---|
| `post_recon4_bil` | history sampled bilinearly (4 taps) for both candidates instead of the 12-tap Catmull-Rom |
| `post_recon4_crbil` | Catmull-Rom for the stronger candidate, bilinear for the other |
| `post_recon4_skip16` / `_skip4` | second candidate dropped below weight 1/16 / 1/4 (spec: 1/255), its weight given to the first |
| `post_recon4_f16` | the six per-sample codes accumulated in packed FP16 (both rows of a pair in one register), history tap weights in packed FP16 |
| `post_recon4_f16x` | the whole 3 x 3 gather (exponent, exp, weight sum, colour, codes) in packed FP16, tap weights in packed FP16 |
| `post_recon4_fast` | `f16` + `crbil` + `skip16` |
| `post_recon4_fastest` | `f16x` + `bil` + `skip4` |

### Fused reconstruction + output filter (`post_fused*`, runtime switch)

One kernel replaces `post_recon` + `post_filter`; the display image stays in LDS and never goes to memory. Results
are bit-identical to the two-kernel path (same functions and operation order; checked in the emulator and by
`runner/post_fused_bench.py` on the GPU). Variants (same arguments):

| Kernel | Full history (in and out) | Output |
|---|---|---|
| `post_fused` (default) | 6 bytes per pixel: R, G, B as FP16, tightly packed (`3 · (y · dw + x)` halves); buffer `6 · dw · dh + 8` bytes (`history_bytes`) | O, 3 x f32 |
| `post_fused_o16` | 6 bytes | O, 4 x f16 (R, G, B, 1) |
| `post_fused_h8` | 8 bytes (RGBA16F, alpha 0), as `post_recon` | O, 3 x f32 |
| `post_fused_h8_o16` | 8 bytes | O, 4 x f16 |

The full history format must stay the same from frame to frame: `post_fused` / `_o16` cannot read a history written by
`post_recon` (use the `_h8` kernels, or reset, when switching paths). The reduced history is unchanged.
`post_w64.co` (`runner/post.py: build_wave64`, the same source with `-mwavefrontsize64`) holds the same kernels in
wave64 mode.

Arguments: `color, A, B, C, hist_prev, m_near, m_center, m_far, w, hist_out, reduced, out, params (WgPostParams), int ow,
int oh, float dec_scale (0.01 / e), const float* expo` (exposure pair or null, as for `post_recon` / `post_filter`).
Grid: `ceil(ow / 40) * ceil(oh / 96)` workgroups of 192 lanes, 27 KB LDS. Workgroup = output strip of 40 columns x 96
rows (D columns 60 s ... 60 s + 59 owned, 2 margin columns each side computed but not stored), walked down in bands
of 3 D row pairs (see the kernel header). Precondition (`fused_supported`): `(floor(63 rw / dw) + 4) · (floor(4 rh / dh)
+ 5) <= 336`, i.e. D >= about 1.43 R per axis; true for every FSR mode (D = 1.5 O, O >= R). Otherwise use the
two-kernel path.

Runtime launch (replacing steps 5 and 6 of `runtime.cpp`, history buffers of `Dn * 8` bytes are large enough):

```
const int grid = cdiv(ow, 40) * cdiv(oh, 96);
launch("post_fused", grid, 192, {&color, &A, &B, &C, &hist[cur], &mnear[nxt], &mcenter, &mfar, &w, &hist[nxt],
                                  &reduced, &out, &p, &ow_i, &oh_i, &dec_scale, &expo});
```

### Fused, scheduled for latency (`post_fused3*`; the path for the in-game budget)

`post_fused` is latency-bound (its bands wait on staging loads, then two history round trips per pixel, then three
barriers). `post_fused3*` keeps its geometry, arguments, grid and preconditions and changes the schedule: the next
band's render footprint is loaded into registers while the current band computes, a pixel's two history candidates are
fetched together, two barriers per band, the fallback colour is read only by pixels that fall back. Variants
(`runner/post.py: FUSED3_VARIANTS`):

| Kernel | Arithmetic | Full history | Output | Result |
|---|---|---|---|---|
| `post_fused3_h6` | exact | 6 bytes (RGB16F, as `post_fused`) | 3 x f32 | bit-identical to `post_recon` + `post_filter` |
| `post_fused3_h6_o16` | exact | 6 bytes | RGBA16F (alpha 1) | the same, rounded to FP16 (nearest even) |
| `post_fused3_fast_h6_o16` | `post_recon4_fastest` (packed FP16 gather, bilinear history, skip4) | 6 bytes | RGBA16F | approximation |
| `post_fused3_h4_o16` | exact | 4 bytes, compact (below) | RGBA16F | approximation (history quantization) |
| `post_fused3_fast_h4_o16` | `post_recon4_fastest` | 4 bytes, compact | RGBA16F | approximation |

Compact history (`h4_pack`): one little-endian 32-bit word per pixel, `y · dw + x`; bits 0-10 R, 11-21 G, 22-31 B, the
E-encoded values as fixed point over [0, 3] (R, G: code · 3 / 2047; B: code · 3 / 1023), rounded to nearest, clamped to
[0, 3]. Host codec: `encode_history4` / `decode_history4`.

The full-history format must not change between frames: switching between kernels of different formats (8, 6 or 4
bytes) needs a reset frame. The runtime's `Dn * 8`-byte history buffers are large enough for every format; the 6-byte
format reads up to 8 bytes past the last pixel's 6 (inside `Dn * 8`).

Runtime launch for every `post_fused*` / `post_fused3*` kernel (replacing steps 5 and 6 of `runtime.cpp`: no
`post_recon*` and no filter launch, no display image); the integration runtime takes these names as `post_kernel` /
`WG_POST_KERNEL` (see "Kernel choice"):

```
const int grid = cdiv(ow, 40) * cdiv(oh, 96);   // 192 lanes per workgroup
launch(post_kernel, grid, 192, {&color, &A, &B, &C, &hist[cur], &mnear[nxt], &mcenter, &mfar, &w, &hist[nxt],
                                &reduced, &out, &p, &ow_i, &oh_i, &dec_scale, &expo});
```

`out` holds `ow · oh` pixels row-major without padding: 12 bytes (3 x f32) for kernels without `_o16`, 8 bytes
(RGBA16F, alpha 1) for `_o16`. The `_o16` output has the layout of an R16G16B16A16_FLOAT image with row pitch
`8 · ow` bytes.

## Second design of the block kernels (`kernels/swin_v2.hip`, build variants `v2*`)

Same kernel names and arguments, same packed weights (this file) and feature maps; merge and expand are the kernels
of `swin.hip`. Only the block kernels' launch geometry differs. It is recorded per variant in `variants.json`, now
with an explicit **windows per workgroup** `wpg` per kernel: the grid is `⌈nwin / wpg⌉` (for the first design
`wpg = lanes/32 · windows per wave`, as above). Each block kernel exists as `name` and `name_nc` with the same
selection rule as `swin.hip` (flags bit 0 of the block section).

| Kernel (default v2) | Lanes per workgroup | Windows per workgroup | LDS per workgroup |
|---|---|---|---|
| `swin_block_c64`, `_c64_add` | 512 | 32 (2 per wave, one after another) | 64 KB: whole 48 KB block section + 16 KB GELU table |
| `swin_block_c96`, `_c96_add` | 512 | 16 (1 per wave) | 47 KB: 2 × 12 KB weight units + 7 KB header + 16 KB table |
| `swin_block_c128` | 512 | 16 (1 per wave) | 56 KB: 2 × 16 KB weight units + 8 KB header + 16 KB table |

The GELU table (FP16 corrections c(a) = a·Φ(−a), or FP8 GELU values in `v2gbyte`) is not model data: a fixed function
of GELU stored in the code object (`swin_v2_gelu_table.h`, generated by `runner/gelu_table.py`) and copied to LDS by
each workgroup. The instrumented build `v2prof` also has a device global `wg_v2_prof` (64 × u64 cycle counters).

### Bottleneck megakernel (`swin_bneck6`, `swin_bneck6_nc`; build variants `bmega*`): stage `bneck`

The `bmega*` builds are `swin_v2.hip` with `WG_V2_BASEMATH=1`: every kernel computes exactly as the `swin.hip` build
with the same arithmetic macros (same operations in the same order), so all their outputs are bit-identical to it:
`bmegahg` and `bmega` to the base build (gelusig), `bmegahg16` to the `vec16` build (packed FP16 GELU and softmax;
variants.py `MEGA_REFERENCE`). Their code object can replace `swin.co` as a whole (it also holds merge, expand and every
block kernel). They also contain one kernel that runs the six bottleneck blocks b0 … b5 in a single launch, the
stage choice **`bneck`** (= b0 … b5):

`swin_bneck6(x, out, tmp, sync, w0, w1, w2, w3, w4, w5, H, W, oxe, oye, oxo, oyo, groups)`

| Argument | Meaning |
|---|---|
| `x` | b0 input, FP8 [H, W, 128] (not modified) |
| `out` | b5 output, FP8 [H, W, 128]; also used as a ping-pong buffer during the launch |
| `tmp` | scratch FP8 [H, W, 128] (contents undefined afterwards) |
| `sync` | `16 + 6·⌊(H + 6)/4⌋` u32 words, **zero before the first launch**; every launch leaves them zero again (the last workgroup resets them), so one buffer serves all later launches on the same stream. Not shared between launches that may overlap |
| `w0` … `w5` | the packed block sections of b0 … b5 (this file) |
| `oxe, oye` / `oxo, oyo` | window offsets of the even (b0, b2, b4) / odd (b1, b3, b5) blocks (spec 02 table) |
| `groups` | the number of workgroups launched; must equal the grid size (used for the reset). Any value ≥ 1 is correct (no co-residency is assumed); fastest is about the resident capacity: by default 128 for `bmegahg` / `bmegahg16` (4 × 32 KB per WGP) and 96 for `bmega` (3 × 40 KB) |

Lanes per workgroup: 256 (all `bmega*` builds), recorded in `variants.json` as `geometry.mega = {threads, wpw, ipw,
groups}` (ipw = 8 windows per work item). Recommended: `bmegahg` (header vectors read from global memory, 32 KB LDS,
128 groups), measured 0.398-0.412 ms against 0.445-0.477 ms for six base launches at 120 x 216. Launch `swin_bneck6_nc` when flags bit 0 is set in all six block
sections, otherwise `swin_bneck6` (identical results where both are valid). Blocks alternate buffers
x → tmp → out → tmp → out → tmp → out.

Weights of the stage `bneck` (runner `pack_weights("bneck", {"b0": …, …, "b5": …})`): the six packed b-stage blobs
back to back (equal sizes), so `wk = blob + k · len/6`; a runtime that already holds the six b-stage blobs passes their
six addresses. Offset of the stage (runner `run("bneck", w, x, offset)`): b0's window offset (it fixes the parity).

Launch plan for a runtime (the same as `stage_runner.Runner` for stage `bneck`):
```
code object: swin__bmegahg.co (replaces swin.co: same stage kernels, bit-identical)
once:        tmp = alloc(H*W*128); sync = alloc(4*(16 + 6*((H+6)/4))); memset(sync, 0)
per frame:   name = all six flags bit 0 set ? "swin_bneck6_nc" : "swin_bneck6"
             launch(name, grid = groups = geometry.mega.groups (128), block = 256,
                    x = enc2 merged output, out = b5 output, tmp, sync, w0..w5, H, W,
                    window_offset(b0, f), window_offset(b1, f), groups)
```

Inside, the work is an ordered queue of items (`ipw` consecutive windows of one block, block-major); a workgroup
takes an item from `sync[0]`, waits until the window rows of the previous block that the item reads (with the border
mirror) or overwrites are complete (counters `sync[16 + 6·k … ]`, one per block and window row), runs it with the
block's weights staged through LDS, and adds its windows to its rows' counters.

### Persistent kernel `swin_mid` (stage `mid`; `bmega*` builds)

Stage **`mid`** = enc2 (block and patch merge), b0 … b5, dec2 (expand and block) in one launch: everything at F/4
and below, bit-identical to the ten launches it replaces (the merge and expand are `swin.hip`'s fused versions,
whose outputs equal the standalone kernels'). Same queue and row counters as `swin_bneck6`, eight phases:
enc2 items (with the merge of their 8 windows' tokens), b0 … b5 items, dec2 items (with the expand of their tokens).

`swin_mid(x, out, skip, A, B, sync, we, w0, w1, w2, w3, w4, w5, wd, H, W, oxe, oye, oxb0, oyb0, oxb1, oyb1, oxd, oyd, groups)`

| Argument | Meaning |
|---|---|
| `x` | enc1's merged output, FP8 [H, W, 96] (H, W = the F/4 size; not modified) |
| `out` | dec2's output, FP8 [H, W, 96] |
| `skip` | FP8 [H, W, 96]: enc2's output (the skip connection), written and read inside the launch |
| `A`, `B` | FP8 [H/2, W/2, 128] scratch: enc2's merged output, then b0 … b5 alternate A → B → A …; A holds b5's output afterwards |
| `sync` | `16 + 8·⌊(H + 6)/4⌋` u32 words (H = the F/4 height), zero before the first launch, left zero |
| `we`, `w0` … `w5`, `wd` | the packed stage blobs of enc2 (with its merge section), b0 … b5, dec2 (with its expand section) |
| offsets | of enc2, of the even b-blocks (b0's), of the odd b-blocks (b1's), of dec2 (spec 02 table, parity f) |
| `groups` | the number of workgroups launched (must equal the grid). Default 96: about 205 VGPRs, so 3 groups of 256 lanes per WGP |

`swin_mid_nc` when flags bit 0 is set in all eight blobs. Weights of the stage `mid` in the runner:
`pack_weights("mid", {"enc2": …, "b0": …, …, "b5": …, "dec2": …})` = the eight stage blobs back to back
(`packing.split_mid`); offset of `run("mid", w, x, offset)` = enc2's window offset.

Launch plan for a runtime (replaces enc2's block and merge, the six b-blocks, dec2's expand and block):
```
code object: swin__bmegahg.co (as for bneck)
once:        sync = alloc(4*(16 + 8*((H4+6)/4))); memset(sync, 0)
per frame:   launch(all eight flags bit 0 ? "swin_mid_nc" : "swin_mid", grid = groups = 96, block = 256,
                    merged1, out2, skip2, merged2, bout_tmp, sync, w_enc2, w_b0..w_b5, w_dec2, H4, W4,
                    off(enc2), off(b0), off(b1), off(dec2), groups)
```
# Private Model M output-quality9 profile (2026-10-11)

`output_filter=quality9` changes only the final display-to-output resampling at
`D = floor((3 O + 1) / 2)` per axis. The default remains `output_filter=default`.
This is an intentional quality variation of spec 06, not a claim of NVIDIA
output equivalence. Model parameters, reconstruction and both temporal history
representations retain the original arithmetic and storage.

The independent reference is `reference/output_quality.py`. For the even output
phase, evaluate normalized Lanczos-3 at `(k - 1/4) / 1.5`, `k = -4..4`, then
apply the minimum squared coefficient correction
`w += ((1/4 - dot(w,k))/dot(k,k))*k`. This preserves the coefficient sum and
sets the first moment to the requested quarter-pixel phase. The odd phase
reverses the coefficients. Each separable pass retains the local tap min/max
clamp. Coefficients are rounded to FP32 in `kernels/output_quality9.h`; GPU
accumulation uses FP32 FMA. The ordinary decode, exposure and pre-exposure
handling remain in place; half framebuffer output retains finite saturation.

`post_filter_quality9` writes RGB32F; `_o16` writes dense RGBA16F; `_h` writes
RGBA16F with a caller-supplied byte row pitch and pre-exposure. The eight-argument
base ABI matches the existing filter. `_h` adds `float pi, int pitch_bytes`.
Launch 64 threads with grid `ceil(ow/40) * ceil(ceil(oh/2)/8)`. Every lane owns
one channel of four columns and up to four output row pairs. Edge coordinates
clamp to the source image; inactive lanes never store.

The private runtime option disables banding and true reconstruction/filter
fusion and chooses the existing exact unbanded reconstruction before this
filter. It rejects `recon=output` and other incompatible reconstruction paths.
This can cost more than the default fused path even when isolated filter times
are similar. No public default, installer or release package is changed.

# Model M output-filter profile quality5 (2026-10-13)

`output_filter=quality5` keeps spec 06's filter as it is (5 taps per pass, the local min/max clamp, the same
kernels on every path, true fusion included) and changes only its 5 weights. The default stays
`output_filter=default` (spec 06). Like quality9 this is WrongGPU's own variation of the final resampling; model
parameters, reconstruction and both histories are untouched.

Why: spec 06 truncates the stretched Lanczos-3 kernel to 5 taps and renormalizes. With the +-1/4 pixel phase the
truncation is lopsided (first moment 0.337 instead of 0.25), so even and odd output pixels sample the picture at
alternately shifted positions. On captured game frames the mean |difference| across the pixel pairs (0,1), (2,3),
... is about 9 % smaller than across the pairs (1,2), (3,4), ...; with quality5 or quality9 the two are equal within
about 1 %.

Weights (even output phase, taps k = -2..2; the odd phase mirrors them): `reference/output_quality.py weights5()`,
the least-squares fit of the ideal resampling filter (delay 1/4 D pixel up to the output's Nyquist frequency, 1/3
cycle per D pixel; zero above) under the weight 1 / (f^2 + 0.02^2), with sum = 1. `kernels/output_quality5.h` holds
them rounded to FP32; it is generated (`python -m clean.reference.output_quality`, `--check` compares).

Build and run: `post_w64__q5.co` is `kernels/post.hip` built in wave64 with `WG_LZ_PROFILE=1` (build.py; the same
resource gate as `post_w64.co`). Its machine code is that of `post_w64.co` with other immediates (the 5 weights and
the two bit deltas of the per-lane weight select), so it costs the same. The runtime loads it in place of
`post_w64.co` (tag `w64`); `wg_rt_kernels` reports `output_filter=quality5` (the values today: section "Output
filter original" below). The option needs the
wave64 post kernels on the HIP backend (no emulator build of the profile). `runner/output_quality_runtime_check.py
--profile quality5` checks whole frames: state bit-identical to the default, output against the reference.

# Pre-exposure rule (2026-10-13)

Runtime option `pre_exposure=scale|original`. The default is `scale`, in the runtime library and in a game. (The
0.4.5 candidates up to r2 passed `pre_exposure=original`, the rule of spec 10 draft 4, for model M in a game; with
auto-exposure and a pre-exposure far from 1 that leaves the network input scaled by pi and weakens the accumulation:
Hogwarts Legacy, `results/2026-10-13-pre-exposure-hogwarts.md`. `original` remains an option.)

`scale` (spec 10 draft 2, section 5 step 1): the stages that read the colour get colour / pi, the exposure measurement
reads the original colour with pi, and the filter's result is multiplied by pi.

`original` (spec 10 draft 4, pending): the colour is never divided by pi and the result never multiplied by it; pi acts
only through the exposure e. Auto exposure (the original's path): spec 07 measures luminance / pi, where pi counts as
1 unless it is positive, finite and the runtime was created with the HDR flag. A supplied exposure x (our own path):
e = x / pi and e_prev = x_prev / pi, both with this frame's pi (none: x = 1), so e / e_prev = x / x_prev and the path
behaves as `scale` does: on the reference sequences the two give the same bits wherever pi = 1 and differ by at most
0.2 % (rounding) on the frames where pi is not 1. The pair kernels `exposure_pair` / `exposure_pair_diet` take 1 / pi
as a last argument `scale` (1 otherwise: the same bits) and keep the unscaled x. `wg_rt_kernels` reports
`pre_exposure=`.

Checked on the matched native reference sequences (statistics from the spec side, 1440p Quality): on the four frames
where the pre-exposure or the exposure parameter steps, the auto-exposure path is 42 to 47 dB from the reference with
`original` and 30 to 36 dB with `scale`; the other frames are unchanged.

# Sparkle guard (2026-10-13)

Backend option `sparkle_guard=<rate>` (0 < rate < 1; `on` = 0.5), off by default; `sparkle_guard_thr=<thr>` (default
0.01). WrongGPU's own pass after the network, for either model; it is not part of the models.

What it does: with t = Y / (1 + Y) of the Rec. 709 luma Y of the result and tp the guarded t of the previous frame
at the motion-compensated position (bilinear), a pixel with d = t - tp > thr gets t' = tp + thr + rate (d - thr),
its colour scaled by Y'/Y. Darkening and brightening up to thr pass unchanged. Not guarded: the first frame, frames
after a reset or a size change, pixels whose previous position is outside the image, and pixels whose reprojected
depth differs by more than 2 % (disocclusion).

Kernel `sparkle_guard` in `kernels/frame.hip` (`frame.co`): one lane per output pixel, 256 threads, grid
`ceil(ow * oh / 256)`; arguments `out16, opitch, out32, prev_in, prev_out, motion, depth, pdepth, ow, oh, rw, rh,
rate, thr, valid`. It reads the result where the network wrote it (RGBA16F rows of `opitch` bytes on the output_copy
path, else 3 x f32) and writes back only the pixels it changes. State of the executor (`interop/executor_hip.cpp`):
two buffers of 2 bytes per output pixel (the guarded t, FP16, ping-pong) and a copy of the previous frame's depth
(4 bytes per render pixel). It runs on the dense f32 inputs only (not with `input_copy=1`). A failed allocation or
launch leaves the frame as the network wrote it.

Measured (results/2026-10-13-sparkle-guard.md): on 40 captured game frames at rate 0.5, thr 0.01: 79 % fewer
single-frame sparkles, 0.4 % of the pixels changed; about 0.12 ms per 1440p frame.

# Output filter "original" (spec 06 draft 4, 2026-10-13)

Spec 06 draft 4 gives the output filter two weight sets. **original**: five weights w₋₂ … w₊₂ for an even output
index that come from the model file (group `filter` / `output_taps`, "Model data" above), mirrored for an odd index,
with FP16 arithmetic: each weight rounded to FP16 once, `a ← h(fma(ŵ_k, t_k, a))` for k = −2 … 2 from a = +0 (one
FP16 rounding per step), then the clamp to the range of the five taps; the same in both passes. **analytic**: the
Lanczos set of draft 3 with FP32 sums, the constants `kLz0 … kLz4` of `post.co` / `post_w64.co`, unchanged.

Runtime option `output_filter` (`runtime_options` / `WG_RUNTIME_OPTIONS`):

| Value | Filter |
|---|---|
| (none), `default` | `original` when the model file has the weights; else `analytic`, with a note for the log |
| `original` | the model file's weights; create fails when the model file has none, or when this build cannot run them |
| `analytic` | the analytic set, whatever the model file holds |
| `quality5`, `quality9` | WrongGPU's own profiles (sections above), unchanged |

`wg_rt_kernels` reports the filter in use, not the option: `output_filter=original|analytic|quality5|quality9`
(before this change the analytic filter was reported as `default`), `output_taps=model|none` and `model_map`. The
note ("output filter: analytic (the model file has no output filter weights, …)") is returned by `wg_rt_notes` /
`runtime_notes`; the D3D12 interop writes it to the log after "network runtime ready" (`Executor::notes`).
`Pipeline.notes()` returns it in tests.

Kernels: `post_w64__orig.co` is `kernels/post.hip` built in wave64 with `WG_LZ_PROFILE=2` (build.py, embedded for
gfx1201 and gfx1200, the same resource gate as `post_w64.co`). The runtime loads it in place of `post_w64.co` (tag
`w64`), so every post path exists for it: true fusion (`post_tf*`), the banded and unbanded two-kernel paths, the
fused kernels, the RGBA16F forms and the interop's output copy. In this build **every kernel that filters**
(`post_filter*`, `post_filter_c1*`, their `_band` / `_h` / `_o16` forms, `post_tf*`, `post_fused*`,
`post_fused3*`) takes five more arguments after the ones documented for it: `u32` each, the FP16 bit patterns of
ŵ₋₂ … ŵ₊₂ for an even output index (`WG_LZ_KARGS`; the runtime rounds the model's float32 weights to FP16, nearest
even, and refuses a weight outside the FP16 range). Kernel arguments are scalar registers: the weights are operands
of `v_fma_f16` and cost no vector register. Launch geometry, LDS layouts and the other arguments are unchanged.

Filter values between the two passes keep their 32-bit slots (registers, the LDS rings of the fused kernels). In
profiles 0 and 1 a slot holds an FP32 value. In profile 2 every value is an FP16 value and the slot holds its FP16
bit pattern in the low half, zero above; the taps of the horizontal pass are the FP16 display values as stored (no
conversion), and only the result of the vertical pass is converted to FP32 for Dec. The clamp is
`med3(a, min, max)` on FP16 values (the taps are finite).

Profiles 0 and 1 compile the source they had before: the weights and the slot handling go through macros
(`WG_LZ_PARAM`, `WG_LZ_ARG`, `WG_LZ_LO`, …) that expand to nothing, or to the previous expression, there. Rebuilding
`post.co`, `post_w64.co` and `post_w64__q5.co` from this source gives the accepted builds' sections byte for byte
(only the symbol-name sections differ, by HIP's hash of the source path), for both targets.

No emulator build and no wave32 build of profile 2: on the emulator backend or with `WG_POST_WAVE=32` the default is
`analytic` with a note, and `output_filter=original` fails. With `recon=output` (D = O) no output filter runs.

Not part of this change (spec 06 draft 4's other items): the display image as 11/11/10-bit codes (the filter's taps
are today's FP16 display values, clamp(E(Ô), 0, 2.45) rounded to FP16), the rounding points of the kernel gather and
of the history sample, the alpha from the game's colour, the reduced history of ⌈D/2⌉.

Reference: `reference/post.py` (`h_fma`, `filter_pass_original`, `encoded_filter_original`,
`output_filter_original`). Whole-frame check on the GPU: `runner/output_filter_original_check.py`; timing:
`runner/output_filter_time.py`; results: `results/2026-10-13-model-m-output-filter-original.md`.
# Formats

## FSR 3.1 to network conventions (spec 20)

Written 2026-10-05 for branch `clean/fsr-hook` (FSR front end); updated for spec 20 draft 2 and the spec team's
answers (ANSWERS.md, spec 20): network jitter = +jitterOffset, pre-exposure rule, non-linear colour → AMD. Sources: spec 20 (network input table), spec 10
section 1, specs 01/03/07, and AMD's public FidelityFX SDK 1.1.4 documentation and headers (MIT):
`docs/getting-started/ffx-api.md` and `docs/techniques/super-resolution-upscaler.md` (sections "Input resources",
"Depth buffer configurations", "Providing motion vectors", "Exposure", "Camera jitter"), `ffx_upscale.h`,
`ffx_api_types.h`, `dx12/ffx_api_dx12.h`. Open points are in `QUESTIONS.md`, "Spec 20".

The front end converts every FSR upscale dispatch (`ffxDispatchDescUpscale`, created with
`ffxCreateContextDescUpscale` + `ffxCreateBackendDX12Desc`) into one `WgFrame` of
`clean/fsr/include/wronggpu_upscaler.h`. The code is `clean/fsr/src/param_map.cpp`; unit tests in
`clean/fsr/tests/test_frontend.cpp` check each line below. Every dispatch is logged twice in `wronggpu-fsr.log`:
the raw FSR values, then the converted `WgFrame` ("WrongGPU frame: ...").

### Conventions of the interface (network side)

| Quantity | Convention in `WgFrame` |
|---|---|
| render size R | `render_width/height`: this frame's size; active region `render_x/y` .. `+R` inside the render textures |
| output size O | `output_width/height`: region of the output texture written, from (0, 0) |
| motion m | render pixels, x right, y down, `previous_position = current_position + m`; `m = motion_scale ⊙ raw + motion_offset` per axis |
| jitter j | render pixels, x right, y down: the image shift of this frame (spec 20 draft 2): render pixel (x, y) shows the scene at the unjittered position (x + ½ − jx, y + ½ − jy) |
| depth | raw game depth; `WG_CREATE_DEPTH_INVERTED` means near = 1, far = 0 |
| exposure | `exposure_mode`: TEXTURE (texel (0,0), red channel), VALUE (`exposure_value`), or COMPUTE (spec 07 from color and `pre_exposure`) |
| pre-exposure π | `pre_exposure` > 0; the exposure applies to color / π (the runtime divides the color by π and multiplies its output by π, spec 10 draft 2) |
| reset | `reset` = 1 discards history (the runtime adds its own reset rules, spec 10 section 3) |
| formats | DXGI_FORMAT to create shader views with (always typed; see "Texture formats") |
| states | D3D12_RESOURCE_STATES the game left the texture in; must be the same after the call |
| command list (version 2) | `command_list`: the game's list, open; after `split_command_list` the same pointer, reset and with the game's bindings restored; `command_list_type`: its D3D12_COMMAND_LIST_TYPE; `command_queue`: the queue it was last executed on (or null) |
| GPU order (version 2) | with the ordered split: commands recorded before the split (and everything the game submits ahead of the list) → `done` fence → backend's outside work → `resume` fence → commands recorded after the split (`clean/fsr/README.md`, "Same-frame execution") |

### Field-by-field conversion

| FSR (`ffx_upscale.h`) | WgFrame / WgCreateInfo | Conversion and reason |
|---|---|---|
| create `flags` HIGH_DYNAMIC_RANGE | `WG_CREATE_HDR` | copied |
| create `flags` DEPTH_INVERTED | `WG_CREATE_DEPTH_INVERTED` | copied (FSR: "inverted [1..0]"; network: near = 1) |
| create `flags` AUTO_EXPOSURE | `WG_CREATE_AUTO_EXPOSURE` | copied; selects `WG_EXPOSURE_COMPUTE` when no exposure texture is given |
| create `flags` DEPTH_INFINITE, DYNAMIC_RESOLUTION, NON_LINEAR_COLORSPACE | `WG_CREATE_DEPTH_INFINITE`, `_DYNAMIC_RESOLUTION`, `_NONLINEAR_COLOR` | copied, informational |
| create `flags` DISPLAY_RESOLUTION_MOTION_VECTORS | `WG_CREATE_MOTION_OUTPUT_RES` | copied; changes the motion scale (below) and where motion is sampled |
| create `flags` MOTION_VECTORS_JITTER_CANCELLATION | (motion_offset) | see "Motion" |
| create `maxRenderSize`, `maxUpscaleSize` | `max_render_*`, `output_*` | copied |
| backend `device` | `d3d12_device` | copied |
| `commandList` | `command_list` | copied (an `ID3D12GraphicsCommandList*`) |
| `color`, `depth`, `motionVectors`, `exposure`, `output` | `WgTexture` | resource pointer copied; format, size and state converted (below) |
| `reactive`, `transparencyAndComposition` | — | not used by the network (logged) |
| `jitterOffset` (x, y) | `jitter` | `j = (+jitterOffset.x, +jitterOffset.y)` (see "Jitter"); signs switchable in the .ini |
| `motionVectorScale` (sx, sy) | `motion_scale` | `(sx, sy)`, times R/O per axis with display-resolution motion vectors (see "Motion") |
| `renderSize` | `render_width/height` | copied; `render_x = render_y = 0` (FSR's render region always starts at the texture origin) |
| `upscaleSize` | `output_width/height` | copied; when 0 (optional in FSR), `maxUpscaleSize` |
| `preExposure` | `pre_exposure` | copied; 1 when ≤ 0 or not finite (FSR requires > 0) |
| `reset` | `reset` | copied |
| `frameTimeDelta` | `frame_time_ms` | copied, informational |
| `enableSharpening`, `sharpness` | — | ignored (the network has no sharpening pass); logged |
| `cameraNear`, `cameraFar`, `cameraFovAngleVertical`, `viewSpaceToMetersFactor` | — | not needed by the network; logged |
| dispatch `flags` (DRAW_DEBUG_VIEW, NON_LINEAR_COLOR_SRGB/PQ) | — | logged; NON_LINEAR_COLOR_SRGB/PQ hands the context to AMD's FSR (a note, not an error) |
| create `flags` NON_LINEAR_COLORSPACE | — | the context is created by AMD's FSR (the network expects linear colour; ANSWERS.md spec 20 item 5) |

### Motion

AMD's documentation: "FSR accepts motion vectors in 2D which encode the motion from a pixel in the current frame to
the position of that same pixel in the previous frame", provided in [−width, −height] … [width, height] after
`motionVectorScale` ("this matches screenspace"); a vector ⟨width, height⟩ at the top-left pixel comes "from the
bottom-right corner", so +y is down. Width and height are those of the motion texture.

- Render-resolution motion vectors: `m = (sx · raw.x, sy · raw.y)` render pixels, `previous = current + m`. This is
  already the network's convention (spec 20 table, spec 03 "Convention"), so no sign change.
  - In spec 03 / spec 90 terms: `mv_scale = motion_scale` and `b_mv = motion_offset / motion_scale` per axis.
- Display-resolution motion vectors (`DISPLAY_RESOLUTION_MOTION_VECTORS`): the scaled vector is in display (output)
  pixels, so `motion_scale = (sx · rw/ow, sy · rh/oh)`. The texture has size O; the motion of render pixel (x, y) is
  read at the output texel `(⌊(x + ½)·ow/rw⌋, ⌊(y + ½)·oh/rh⌋)` (nearest; confirmed, ANSWERS.md spec 20 item 3).
- Jitter cancellation (`MOTION_VECTORS_JITTER_CANCELLATION`, "the motion vectors have the jittering pattern applied
  to them"): measured between jittered positions, raw motion = true motion + (previous jitterOffset − current
  jitterOffset), so `motion_offset = jitterOffset(current) − jitterOffset(previous)` in render pixels, 0 on a
  context's first frame. (Confirmed, ANSWERS.md spec 20 item 4: consistent with the image moving by +jitterOffset.)
- Switches: `motion_sign_x/y = −1` in `wronggpu.ini` multiply `motion_scale` (test 20-T2).

### Jitter

AMD's documentation turns `jitterOffset` (unit: render pixels) into a projection offset
`(2·jx/renderWidth, −2·jy/renderHeight)` in NDC. NDC +y is up and screen +y is down, so the rendered image moves by
(+jx, +jy) screen pixels: the pixel whose center is c shows the scene point that is at c − jitterOffset in the
unjittered image. Spec 20 draft 2 (and spec 03) define the network's jitter as that image shift: render pixel (x, y)
shows the scene at (x + ½ − jx, y + ½ − jy), hence **j = +jitterOffset**. The spec team measured this on recorded
game frames (slope −0.93, correlation −0.99 of the sample offset against j; ANSWERS.md spec 20 item 1). Spec 03's
`base = (x + ½ − jx, y + ½ − jy)` is then exactly the sample's position. Switches: `jitter_sign_x/y` (default +1,
until 1d6061b −1 after spec 20 draft 1's wording) in `wronggpu.ini`. The stand-in's jitter compensation reads the
color texture at `render position + j` (= render position + jitterOffset), which removes the sub-pixel wobble when
the convention is right (test 20-T2 step 4 and `test_dll_d3d12 upscale`, where the opposite sign gives a 10× larger
error).

### Exposure

AMD: "Pre-exposure: a value by which we divide the input signal to get back to the original signal"; "Exposure: a
value which is multiplied against the result of the pre-exposed color value"; the exposure resource is a 1×1
R32_FLOAT texture, optional when AUTO_EXPOSURE is set.

| Game gives | exposure_mode | Network (spec 10 section 5 step 1) |
|---|---|---|
| exposure texture (with or without AUTO_EXPOSURE) | TEXTURE | the texel value, "as is" |
| no texture, AUTO_EXPOSURE | COMPUTE | spec 07 from color and π = `pre_exposure` |
| no texture, no AUTO_EXPOSURE | VALUE, `exposure_value = 1` | 1 |

Pre-exposure (ANSWERS.md spec 20 item 2, spec 10 draft 2 section 5): in every mode the exposure refers to the color
*without* pre-exposure (AMD's semantics). The runtime divides the game's color by π before the network and
multiplies the network's output by π. The interface passes the exposure source and π unchanged; the stand-in does
neither (it only resamples, so its output stays in the game's domain). Cyberpunk 2077: AUTO_EXPOSURE, no exposure
texture, π = 1 (20-T0).

### Sizes

`R = renderSize` each frame (≤ `maxRenderSize`; it may change every frame with dynamic resolution, which the
runtime treats as a reset, spec 10 section 3). `O = upscaleSize`, or `maxUpscaleSize` when 0. A dispatch whose R
exceeds the color texture, or whose O exceeds the output texture, is rejected (logged) and the context falls back to
AMD (spec 10 section 8). The network's padded size F and reconstruction size D follow from R and O (spec 01).

### Texture formats

The view format of each texture is chosen from the resource's own `DXGI_FORMAT` (`ID3D12Resource::GetDesc`) and the
FSR description's `FfxApiSurfaceFormat` (mapped to DXGI as in AMD's `ffxApiGetSurfaceFormatDX12`, inverted):

1. A typed resource format is used as is; depth formats become their readable view formats
   (D32_FLOAT → R32_FLOAT, D24_UNORM_S8_UINT → R24_UNORM_X8_TYPELESS, D32_FLOAT_S8X24_UINT →
   R32_FLOAT_X8X24_TYPELESS, D16_UNORM → R16_UNORM).
2. A typeless resource uses the FSR format when that is a typed member of the same family (e.g. R8G8B8A8_TYPELESS
   with R8G8B8A8_SRGB → R8G8B8A8_UNORM_SRGB).
3. Otherwise the family default: FLOAT for 16/32-bit families, UNORM for 8/10-bit families, R16_UNORM for
   R16_TYPELESS (a depth buffer), R24_UNORM_X8_TYPELESS for R24G8_TYPELESS.

FSR's own helper maps D24S8 resources to R32_UINT; rule 1 ignores that. An sRGB output cannot be a UAV: the stand-in
writes it through the UNORM view and encodes sRGB in the shader.

### Resource states

`FfxApiResource.state` (FfxApiResourceState bits) → D3D12_RESOURCE_STATES: COMMON and PRESENT → 0 (COMMON);
UNORDERED_ACCESS → UNORDERED_ACCESS; COMPUTE_READ → NON_PIXEL_SHADER_RESOURCE; PIXEL_READ → PIXEL_SHADER_RESOURCE;
COPY_SRC → COPY_SOURCE; COPY_DEST → COPY_DEST; INDIRECT_ARGUMENT → INDIRECT_ARGUMENT; RENDER_TARGET →
RENDER_TARGET; combinations bit by bit (GENERIC_READ = COPY_SOURCE | NON_PIXEL_SHADER_RESOURCE). AMD asks for inputs
in NON_PIXEL_SHADER_RESOURCE, but games may pass others; the upscaler transitions inputs to a state that includes
NON_PIXEL_SHADER_RESOURCE (adding it to read-only states) and the output to UNORDERED_ACCESS, and transitions all of
them back after its work.

### FSR helper queries on WrongGPU's own contexts

The game asks the upscaler context for its jitter sequence and render sizes. WrongGPU answers from AMD's
documentation (`super-resolution-upscaler.md`, "Scaling modes", "Camera jitter"); the game then renders with these
values and passes them back in `jitterOffset` and `renderSize`, so only internal consistency matters.

| Query | Answer |
|---|---|
| upscale ratio of a quality mode | NativeAA 1.0, Quality 1.5, Balanced 1.7, Performance 2.0, Ultra performance 3.0 |
| render resolution from quality mode | ⌊display / ratio⌋ per axis (at least 1) |
| jitter phase count | ⌊8 · (displayWidth / renderWidth)²⌋. AMD's text says ceil, but its table (18, 23, 32, 72 for the four modes at their actual render sizes) and AMD's own answer in Cyberpunk 2077 (17 for 1707 → 2560, `results/2026-10-05-fsr-20-3989fff.md`) are reproduced by rounding down (until 3989fff WrongGPU rounded to nearest, which gave 18 there) |
| jitter offset of index i | Halton(base 2, base 3) at index ((i mod n) + 1), minus 0.5 per axis |
| provider version | id 0x5747000000000001, name of the upscaler ("WrongGPU stand-in upscaler") |
| GPU memory | the upscaler's own allocations |

Contexts handed to AMD (fallback, or `upscaler = amd`) forward these queries to AMD.

### Stand-in upscaler (spec 20 "Stand-in (now)")

`clean/fsr/src/standin_d3d12.cpp`, shader `standin_shader.h` (HLSL, cs_5_0, compiled at run time with Windows'
`d3dcompiler_47.dll`). One 8×8-thread compute dispatch per frame into the game's command list. For output pixel q:
render position `u = (render_x, render_y) + (q + ½) · R / O`; texel position `t = u + j` with jitter compensation
(default) or `t = u` without; then

- bilinear: the four texels around t (texel centers at i + ½), coordinates clamped to the active render region;
- lanczos: Lanczos-2 (4 × 4 texels, weight sinc(x)·sinc(x/2), normalized), clamped to the range of the four
  nearest texels to avoid ringing.

The output keeps the input's domain (no exposure applied; alpha passed through). Debug views (`view =` in the .ini)
for test 20-T2:

| view | output |
|---|---|
| warp | `lerp(current, previous output at q + m·O/R, warp_weight)`; history is two R16G16B16A16_FLOAT textures of size O |
| warpdiff | `|current − previous upscaled image at q + m·O/R| · debug_gain`, black where motion and jitter are right |
| motion | red = 0.5 + mx·gain/64, green = 0.5 + my·gain/64 (m in render pixels) |
| depth | near bright: `v^(1/gain)` with v = depth (inverted) or 1 − depth |
| jitter | the upscale view plus a 129 × 129 box at the top-left with a dot at (64, 64) + 128·j |

The stand-in binds its own root signature, pipeline and shader-visible descriptor heap on the game's command list
(as any FSR implementation must) and does not restore the game's earlier bindings; see QUESTIONS.md spec 20 item 7.

## FSR 3.0 component DLLs to network conventions (spec 20, integration point 2)

Written 2026-10-05 after test 20-T0 showed that Cyberpunk 2077 upscales FSR 3 through its FSR 3.0 component DLLs
(`results/2026-10-05-fsr-20-3989fff.md`). Sources: AMD's FidelityFX SDK tag `fsr3-v3.0.4` (MIT) host headers
`ffx_fsr3upscaler.h`, `ffx_fsr3.h`, `ffx_types.h`, `ffx_interface.h` and the documents
`docs/techniques/super-resolution-interpolation.md` ("Frame generation requires upscaling to run, since some internal
data is shared between upscaling and interpolation") and `frame-interpolation.md` (frame interpolation consumes the
upscaler's dilated depth and dilated motion vectors). Code: `clean/fsr/src/fsr30_*.{h,cpp}`; WrongGPU's mirror of the
structures (`fsr30_types.h`) is checked against AMD's headers at compile time (`fsr30_layout_check.cpp`).

### The DLLs

| WrongGPU's file | forwards to | role |
|---|---|---|
| `ffx_fsr3upscaler_x64.dll` | `ffx_fsr3upscaler_x64.wronggpu-original.dll` | the FSR 3.0 upscaler (`ffxFsr3Upscaler*`): logged; upscale dispatches can go to the stand-in |
| `ffx_fsr3_x64.dll` | `ffx_fsr3_x64.wronggpu-original.dll` | FSR 3 = upscaler + optical flow + frame interpolation (`ffxFsr3*`): logged; frame generation always AMD's |

Each exports the FSR 3.0.x functions of its header plus the SDK 1.1 additions (`...GetEffectVersion`,
`...GetGpuMemoryUsage`, `ffxFsr3UpscalerSetConstant`, `ffxFsr3UpscalerSetGlobalDebugMessage`,
`ffxFsr3ContextDispatchFrameGenerationPrepare`), so a game linked against either version finds every import. If
AMD's DLL itself exports `...GetEffectVersion`, it is an SDK 1.1 (FSR 3.1) build whose structures differ: then
WrongGPU logs and forwards only (pointers, no decoding) and never runs the stand-in. `FfxResource` passed by value
(`...ResourceIsNull`) is forwarded as the pointer to the caller's copy that the x64 calling convention passes.

### Conversion (FSR 3.0 → FFX API form → `WgFrame`)

An FSR 3.0 upscale dispatch is converted field by field into the FFX API's `ffxDispatchDescUpscale` and then goes
through the same `MapDispatch` as integration point 1, so every convention of the section above applies unchanged
(jitter j = +jitterOffset, motion = motionVectorScale ⊙ raw, exposure, sizes, formats, states). What differs:

| FSR 3.0 | Conversion |
|---|---|
| `FfxSurfaceFormat` (3.0 numbering: no R32G32B32_FLOAT, so R32G32_FLOAT is 5, not 6, and so on) | mapped name by name to `FfxApiSurfaceFormat`, then to DXGI as before |
| `FfxResourceStates` | same bits as `FfxApiResourceState` (COMMON = 1, UAV = 2, COMPUTE_READ = 4, ...): copied |
| `FfxResource.name` (wchar_t[64]) | logged only |
| creation flags | bits 0–6 (HDR, display-resolution MV, MV jitter cancellation, depth inverted, depth infinite, auto-exposure, dynamic resolution) mean the same as in the FFX API and are copied; TEXTURE1D_USAGE, DEBUG_CHECKING and the `ffx_fsr3`-only bits (UPSCALING_ONLY, HDR/SDR final output, ASYNC_WORKLOAD_SUPPORT) are logged only |
| output size O | upscaler context: `displaySize`; `ffx_fsr3` context: `upscaleOutputSize` (or `displaySize` when 0). FSR 3.0 has no per-frame output size |
| `output` / `upscaleOutput` | the `WgFrame` output |
| `dilatedDepth`, `dilatedMotionVectors`, `reconstructedPrevNearestDepth` | not network inputs; their presence decides whether AMD's upscaler runs first (below) |
| D3D12 device | taken from the command list (`ID3D12GraphicsCommandList::GetDevice`) at the first dispatch, not from the backend interface's `FfxDevice` |

### Who upscales (settings `fsr3_route`, `fsr3_amd_first`; `upscaler = standin`)

- Frame generation off (`ffx_fsr3` context created with `FFX_FSR3_ENABLE_UPSCALING_ONLY`, as Cyberpunk 2077 does then):
  nothing reads the upscaler's dilated depth / motion / previous depth, so AMD's upscaler does not run at all (ANSWERS.md,
  spec 20, FSR 3.0 item 12). `ffx_fsr3_x64.dll` tells `ffx_fsr3upscaler_x64.dll` (export
  `wgFsr3UpscalerHintUpscalingOnly`, per thread, around AMD's `ffxFsr3ContextCreate`, which creates the upscaler
  context on that thread) that the upscaler context belongs to such a context.
- In `ffx_fsr3upscaler_x64.dll` (`fsr3_route = auto` or `upscaler`): if the dispatch carries the three frame-generation
  resources and its context does not belong to an UPSCALING_ONLY context, or `fsr3_amd_first = always`, AMD's
  upscaler runs first (it writes those resources, which AMD's frame interpolation reads, and its own output; option (a)
  of ANSWERS.md item 12), then the stand-in records its dispatch on the same command list and overwrites the output.
  Otherwise only the stand-in runs (`fsr3_amd_first = never` forces that always).
- In `ffx_fsr3_x64.dll`: AMD's `ffxFsr3ContextDispatchUpscale` runs, and with `fsr3_route = auto` the stand-in then
  overwrites `upscaleOutput` only if WrongGPU's `ffx_fsr3upscaler_x64.dll` saw no upscaler dispatch during that call
  (counted per thread through its extra export `wgFsr3UpscalerSeenOnThread`), i.e. when AMD's `ffx_fsr3` has the
  upscaler built in; with `fsr3_route = fsr3` always (and the upscaler DLL then leaves it alone). For an
  UPSCALING_ONLY context whose upscale happens at this level (known after its first frame, or `fsr3_route = fsr3`),
  AMD's call is skipped and only the stand-in runs.
- `fsr3_route = off` or `upscaler = amd`: everything is AMD's; WrongGPU only logs.
- If the stand-in cannot be created or a frame fails, that context stays on AMD's upscaler from then on (logged).

After AMD's upscaler dispatch the game's textures are assumed back in the states the game passed (AMD's DX12 backend
restores imported resources; accepted, ANSWERS.md item 13); the stand-in's barriers start from those states.
