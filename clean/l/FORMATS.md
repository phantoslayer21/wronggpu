# Model L: kernels, maps, blobs and launch order (issue #89)

This is the interface the Model L runtime frame builds against.
- A row marked **exists** has a HIP kernel that passed its GPU check (merged, or in the open Model L kernel PR).
- A row marked **planned** is the interface the kernel is being written to. Its names, arguments, grids and maps are
  fixed here, and a change goes through this file first.
- Each D3D12 port declares the same code object, kernel name and arguments as its HIP kernel (`clean/dx12/kernels/l_*.hlsl`).

The structure follows `clean/l/reference/table.py`. The launch list comes from `clean/l/runner/l_pack.py`
(`encoder_launches`, `decoder_launches`, with each map's `in_format` / `out_format`). The ten shared Swin stages
(enc1 ... dec1) use Model M's blobs, kernels and launch plan unchanged, with Model L's weights.

## Maps

Every map is `[rows][cols][C]`, channel fastest, as Model M's feature maps (`clean/FORMATS.md`, "Feature maps").
- FP8 = E4M3, 1 byte per channel. FP16 = 2 bytes per channel, little-endian.
- F = fw x fh is the padded grid (multiples of 32). R = rw x rh is the render size.
- **Halo maps** have 2 extra pixels on each side: `[fh + 4][fw + 4][C]`. Map pixel (i, j) is grid position
  (x, y) = (j - 2, i - 2). The halo pixels hold the per-pixel chain evaluated at those positions: the 20 values at
  the mirrored render position, the dither of the unmirrored position. They are what the encoder block reads for
  window members outside the grid.

| Map | Shape | Element | Written by | Read by |
|---|---|---|---|---|
| `q` | `[fh + 4][fw + 4][32]` halo | FP8: values 0..19, bytes 20..31 zero | `lev/input_encoder` | `l_linear_32_32_relu` |
| `x1` | `[fh + 4][fw + 4][32]` halo | FP8 (not stored by `l_linear_32_32_fused`) | `l_linear_32_32_relu` | `l_linear_32_32_f16` |
| `x2` | `[fh + 4][fw + 4][32]` halo | **FP16** (64 bytes per pixel) | `l_linear_32_32_f16` | `l_swin_block_c32_enc` |
| `x3` | `[fh][fw][32]` | FP8 | `l_swin_block_c32_enc` | `l_project_gather` |
| `enc0.features` | `[fh/2][fw/2][64]` | FP8 | `l_project_gather` | enc1 (Model M's stage) |
| m_near, m_far, m_center | as Model M | as Model M | `lev/input_encoder` | as Model M |
| `y` | `[fh][fw][32]` | FP8 | `l_dec_project` | `l_swin_block_c32_dec` |
| `y2` | `[fh][fw][32]` | **FP16** (64 bytes per pixel) | `l_swin_block_c32_dec` | `l_coef` |
| A, B, C, new latent | as Model M's `coef_decoder` outputs | as Model M | `l_coef` | as Model M |

Extra bytes per frame:
- 32 (fh + 4)(fw + 4) for `q`;
- 32 (fh + 4)(fw + 4) for `x1`;
- 64 (fh + 4)(fw + 4) for `x2`;
- 32 fh fw for `x3`;
- 32 fh fw for `y`;
- 64 fh fw for `y2`.

No kernel works in place.
- `x2` may reuse `q`'s memory (`q` is dead after `l_linear_32_32_relu`), if it is sized for `x2`.
- `y` and `y2` may reuse the encoder maps, which are dead after `l_project_gather`.

## Blobs (`l_pack.pack_model_l`, names = the converter map's stage / tensor names)

| Blob | Format |
|---|---|
| `enc0.values` | Model M's enc0 blob (`clean/runner/enc0.py` pack) with W = 0 and b = 0: dither table T, sigma1, sigma2 (the map's stored values) |
| `enc0.Win1`, `enc0.Win2` | merge / expand section format: bias tile vector (rows R_L(32)), then fragments [mt < 2][kc < 2], rows R_L(32), K order K_L(32). `Win1` has zero weights for inputs 20..31 |
| `enc0_block`, `dec0_block` | Model M's Swin block section (`packing.pack_block`) at C = 32, 2 heads, MLP 128, padded to 256 |
| `enc0.project` | bias tile vector (rows R_L(64)), then fragments [mt][kb], rows R_L(64). K order: block 2i + s of group i holds channel 32 i + 16 h + 8 s + j (dec0's order, `clean/runner/dec0.py` `kord_decoder`, over 4 groups) |
| `dec0.project` | bias tile vector (rows R_E(32)), then fragments [mt < 8][kb < 4], rows R_E(32) (tile 2 s + g: sub-pixel s, channels 16 g ...). K order as `enc0.project` (two groups of 32) |
| `dec0.coef` | Model M's dec0 blob (`clean/runner/dec0.py` pack) with W = 0 and b = 0: the scalars S and T |

## Code objects

| File | Source | Kernels |
|---|---|---|
| `l_enc_values.co` (loaded under the tag `lev`) | `clean/kernels/encoder.hip` with `-DWG_ENC_VALUES_OUT=1` | `input_encoder` |
| `l_ops.co` | `clean/l/kernels/l_ops.hip` | `l_linear_32_32_relu`, `l_linear_32_32_f16`, `l_project_gather`, `l_dec_project`, `l_coef`, `l_coef_pack12` |
| `l_post.co` | `clean/l/kernels/l_post.hip` (includes Model M's `clean/kernels/post.hip` for its functions; Model M's kernels come along unused) | `l_post_recon`, `l_post_recon_p12` (HDR) |
| `l_swin.co` | `clean/l/kernels/l_swin.hip` | `l_swin_block_c32[_nc]` (exists, #102: FP8 in / FP8 out, mirror border; block checks only), `l_swin_block_c32_enc[_nc]`, `l_swin_block_c32_dec[_nc]` |

All are built with `clean/runner/build.py`'s `device_command`. The build's `-cuid` comes from the output file name,
so compare code objects only between outputs of the same name. `_nc` is used when bit 0 of the block blob's first
word is set, as for Model M's blocks. D3D12 ports exist for the `_nc` forms only.

## Kernels

Workgroups are 256 lanes (8 waves of 32) throughout. "Per-pixel" kernels handle 128 pixels (or tokens) per
workgroup, in raster order of their map.

| Kernel | Status | Arguments | Grid | Does |
|---|---|---|---|---|
| `lev/input_encoder` | exists (GPU-checked) | as Model M's `input_encoder`, with `features` replaced by `q` | ceil(((fh + 4) / 2) ((fw + 4) / 2) / 128) (2 x 2 pixel groups of the halo map) | the 20 values and their dithered FP8 for every halo-map pixel, into `q`. The motion surfaces as Model M, only for pixels inside R. w = `enc0.values` |
| `l_linear_32_32_relu` | exists (GPU-checked) | `x, y, w, npix` | ceil(npix / 128), npix = (fh + 4)(fw + 4) | y = q8(max(0, RN16(b + W x))), FP8 -> FP8. w = `enc0.Win1` |
| `l_linear_32_32_f16` | exists (GPU-checked) | `x, y, w, npix` | as above | y = RN16(b + W x), FP8 -> **FP16**. w = `enc0.Win2` |
| `l_linear_32_32_fused` | exists (GPU-checked), D3D12 only | `x, y, w1, w2, npix` | as above | the two above in one kernel, `x1` kept in registers: the same bytes in `y` (`x2`), `x1` not stored. x = `q`, y = `x2` in two different buffers. w1 = `enc0.Win1`, w2 = `enc0.Win2`. The native D3D12 frame runs it in place of the two (fusion `ll`, on by default); without `ll` or without the kernel in the build, the two run |
| `l_swin_block_c32_enc[_nc]` | exists (GPU-checked) | `x, xadd (null), out, w, H, W, ox, oy` | ceil(nwin / 8), nwin = ceil((W + ox) / 4) ceil((H + oy) / 4) | the encoder block. x = `x2` (FP16 halo map; member at grid (gx, gy) read at map pixel (gy + 2, gx + 2)). out = `x3` (FP8, grid only). H = fh, W = fw. (ox, oy) = `window_offsets["first"]` for the frame parity. w = `enc0_block` |
| `l_project_gather` | exists (GPU-checked) | `x, features, w, tw, th` | ceil(tw th / 128), tw = fw / 2, th = fh / 2 | token (r, c) gathers pixels (2r + p / 2, 2c + p % 2), p = 0..3, as channels 32 p + i. Running FP16 sum per group of 32 from the bias, clamp +-6.28125, FP8 -> `enc0.features`. w = `enc0.project` |
| `l_dec_project` | exists (GPU-checked) | `z, y, w, tw, th` | ceil(tw th / 128) | per token of dec1's output z [th][tw][64]: 128 outputs (two RN16 groups of 32 inputs, then FP8). Output 32 s + k goes to pixel (2r + s / 2, 2c + s % 2), channel k, of `y`. w = `dec0.project` |
| `l_swin_block_c32_dec[_nc]` | exists (GPU-checked) | `x, xadd (null), out, w, H, W, ox, oy` | as the encoder block | the decoder block. x = `y` (FP8). A member outside the grid takes token-grid mirrored pixels with the sub-pixel kept. out = `y2` (**FP16**). (ox, oy) = `window_offsets["second"]`. w = `dec0_block` |
| `l_coef` | exists (GPU-checked) | `y2, A, B, C, latent, w, params` | ceil((fh / 2)(fw / 2) / 128), lanes as Model M's `coef_decoder` | Model L's per-pixel functions on the 13 channels `table.L_DEC_VALUES` of `y2`: A, B, C in Model M's forms, latent = tanh(candidate). Writes the same buffers and formats as `coef_decoder`, pixels inside R only. **Reads no motion, previous latent, jitter or reset.** params = Model M's `WgDecParams` (only rw, rh, fw, fh are read). w = `dec0.coef` |
| `l_coef_pack12` | exists (GPU-checked) | as `l_coef` (`B` unused, may be null) | as `l_coef` | `l_coef`'s values in Model M's dec0 "pack12" layout (`clean/kernels/decoder.hip` `WG_DEC_PACK12`, read by the output stage's `_p12` kernels): image A, 8 bytes per pixel = A0 A1 A2 B0 | B1 B2 C2 (FP16); image C, 4 bytes per pixel = C0 C1 (FP16); latent as `l_coef`. Only padding bytes are left out: unpacked, byte-identical to `l_coef` (`l_ops_check.py`) |
| `l_swin_block_c32_dec_coef12_nc` | exists (GPU-checked), D3D12 only | the decoder block's, `out` unused (null), then `A, C, latent, w_coef, rw, rh` | as the decoder block | the decoder block with `l_coef_pack12` as its epilogue: the same bytes in `A`, `C` and `latent`, `y2` not written. w = `dec0_block`, w_coef = `dec0.coef`. The native D3D12 frame runs it in place of the two (fusion `lc`, on by default); without `lc` or without the kernel in the build, the two run |

## Launch order of one frame

Model M's frame (`clean/FORMATS.md`, "Stage order per frame") with steps 2 and 4 replaced. Window offsets follow
the frame parity, as Model M's.

2. Encoder:
   1. `lev/input_encoder` -> `q` (halo), m_near, m_far, m_center
   2. `l_linear_32_32_relu` q -> `x1` (halo)
   3. `l_linear_32_32_f16` x1 -> `x2` (halo, FP16)
   4. `l_swin_block_c32_enc[_nc]` x2 -> `x3`
   5. `l_project_gather` x3 -> `enc0.features` (what enc1 reads)
3. enc1 ... dec1: Model M's stages with Model L's weights, unchanged.
4. Decoder:
   1. `l_dec_project` dec1 output -> `y`
   2. `l_swin_block_c32_dec[_nc]` y -> `y2` (FP16)
   3. `l_coef` y2 -> A, B, C, new latent. Model L's latent recurrence runs only through the next frame's input
      encoder, which samples this latent.

5. Reconstruction (post) and output filter. The float64 oracle is `clean/l/reference/recon.py`, driven by the
   table's `recon` entry. It is Model M's reconstruction with:
   - a 4 x 4 colour gather;
   - history candidates combined in linear colour;
   - the plain linear colour as the fallback of an invalid or reset sample;
   - the post stage's learned scale.

   Behaviour pinned in the oracle:
   - The tiling variant (3/2 or 3/1) follows `table.recon_variant`, a D / render ratio. Both variants compute the same
     function.
   - A zero total gather weight follows the normal formulas with NaN: history h(E(NaN)) = -1.4404296875 in HDR, NaN
     in LDR; display and reduced code 0.
   - LDR writes 11/11/10-bit display codes of the plain colour. The output stage unpacks them (code RN32(1/N), FP16)
     and filters them with no decode and no exposure (`recon.output_filter`).
   - The output filter is Model M's (the project's own taps).
   - Steps 4 to 6 (history candidates, combination, blend) run in float32 with the fused forms, one rounding per
     operation, as the spec side states the order (table `"blend": "float32"`; Model M's entry keeps float64). This
     matters where a large fallback colour (about 1e3 and up) is cancelled in the blend: H^ = fused((C - Hl) beta' +
     Hl) rounds C - Hl to float32 first, and the float64 form lands many FP16 steps away there.

   The reconstruction kernel `l_post_recon` (`l_post.co`, HDR; exists, GPU-checked by
   `clean/l/runner/l_post_check.py`):
   - arguments, buffers, layouts and grid are Model M's `post_recon` (Model M's `clean/FORMATS.md`), with `w` = the
     post blob (`w[0]` = Model L's s);
   - its colour gather runs in the model's precision, as confirmed by the spec side (the oracle's `recon.gather_half`):
     - fused target positions;
     - FP16 offsets and exponent with the fused cross term;
     - float32 weights flushed to +0 below 2^-126;
     - FP16 weights into half accumulators;
     - a float32 weight sum;
     - tap order i (columns) outer, j (rows) inner.

     So the kernel and the oracle agree on which weights underflow. Two regimes follow: total weight 0, the
     zero-weight values; and total weight > 0 with every FP16 weight 0, where the output is the candidate history.
   - its candidate combination is one fused multiply-add w1 L1 + (w2 L2), then times e;
   - its blend is the fused form beta' = fused(alpha (1 - beta-bar) + beta-bar), H^ = fused((C - Hl) beta' + Hl);
   - the decode passes a NaN through.

   `l_post_recon_p12` (`l_post.co`) is the same kernel on the pack12 coefficients of `l_coef_pack12`: `A` = image A8
   (8 bytes per render pixel), `C` = image C4 (4 bytes per render pixel), `B` unused (may be null); the other arguments,
   the grid and the outputs as `l_post_recon`, byte for byte (`clean/l/runner/l_d3d12_ops_check.py` checks it on HIP).
   The D3D12 ports of both are `clean/dx12/kernels/l_post.hlsl`.

   The LDR kernel is planned, with the same arguments. Still open for the kernel-order model: the hardware exp2,
   reciprocal and divisions, and the Catmull-Rom weight forms, which the kernel takes from Model M's `post.hip`
   (part of the Model M fidelity decision).

## Checks (GPU, synthetic data)

- `clean/l/runner/l_c32_check.py`: the C = 32 blocks (`l_swin_block_c32`, `_enc`, `_dec`) against the oracle.
- `clean/l/runner/l_ops_check.py`: the linears, both projections and `l_coef` against the oracle steps.
- `clean/l/runner/l_enc_check.py`: `lev/input_encoder` (halo map and motion surfaces) against the oracle.
- `clean/l/runner/l_d3d12_check.py` (with its D3D12 port): each D3D12 port against its HIP kernel, byte for byte.
- `clean/l/runner/l_fold_check.py`: `l_swin_block_c32_dec_coef12_nc` against `l_swin_block_c32_dec_nc` ->
  `l_coef_pack12` on D3D12 and on HIP, byte for byte.
- `clean/l/runner/l_linfuse_check.py`: `l_linear_32_32_fused` against `l_linear_32_32_relu` -> `l_linear_32_32_f16`
  on D3D12 and on HIP, byte for byte.
- One check per new kernel follows with that kernel.
