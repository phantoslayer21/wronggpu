# Model L: kernels, blobs and launch order (issue #89)

Status: HIP kernels for Model L's encoder chain, decoder front and C = 32 block; no runtime integration yet. Feature
maps are `[tokens][C]` FP8 E4M3 bytes, channel fastest, as in Model M (`clean/FORMATS.md`, "Feature maps"); a token
map at F/2 has `ntok = (fh / 2) * (fw / 2)` tokens in raster order. Blobs: `clean/l/runner/l_pack.py`
(`pack_model_l`); the ten shared Swin stages use Model M's blobs, kernels and launch plan unchanged.

## Code objects

| File | Source | Kernels |
|---|---|---|
| `l_enc_values.co` | `clean/kernels/encoder.hip` with `-DWG_ENC_VALUES_OUT=1` | `input_encoder` (values only) |
| `l_ops.co` | `clean/l/kernels/l_ops.hip` (`-I clean/kernels`) | `l_linear_80_32`, `l_linear_32_32`, `l_project_80_32`, `l_copy_channels` |
| `l_swin.co` | `clean/l/kernels/l_swin.hip` (`-I clean/kernels`) | `l_swin_block_c32`, `l_swin_block_c32_nc`, `l_swin_block_c64` (check only) |

The runtime loads `l_enc_values.co` under the tag `lev` (its kernel is launched as `lev/input_encoder`, next to Model
M's own `input_encoder`); `l_ops.co` and `l_swin.co` are untagged. These are also the names the D3D12 ports will
declare.

All are built with `clean/runner/build.py`'s `device_command` (gfx1201). The build's `-cuid` comes from the output file
name, so compare code objects only between outputs of the same name.

## Kernels

| Kernel | Arguments | Grid (workgroups of 256 lanes) | Does |
|---|---|---|---|
| `input_encoder` of `l_enc_values.co` | as Model M's `input_encoder` (color, depth, motion, hist, latent, mhist, **raw**, m_near, m_far, m_center, expo, w, `WgEncParams`) | ceil(ntok / 128) | the per-pixel values and their dithered FP8 quantization as Model M's enc0; writes `raw` [ntok][80] (sub-pixel p = dx + 2 dy, value i at byte 20 p + i) and the three motion surfaces; w = blob `enc0.values` |
| `l_linear_80_32` | x, y, w, ntok | ceil(ntok / 128) | y [ntok][32] = q8(W x + b), x [ntok][80]; w = blob `enc0.Wa` |
| `l_linear_32_32` | x, y, w, ntok | ceil(ntok / 128) | y [ntok][32] = q8(W x + b); w = blob `enc0.Wb` |
| `l_swin_block_c32[_nc]` | x, xadd (null), out, w, H, W, ox, oy | ceil(nwin / 8), nwin = ceil((W + ox) / 4) ceil((H + oy) / 4) | one Swin block at C = 32 (2 heads, MLP 128), as Model M's block kernels; `_nc` when bit 0 of the blob's first word is set; w = blob `enc0.blk` or `dec0.blk` |
| `l_project_80_32` | raw, x, features, w, ntok | ceil(ntok / 128) | features [ntok][64] = q8(clamp(s, +-6.28125)) with s = b, then s = RN16(s + W[:, g] z[g]) for each group g of 32 channels of z = [raw (80) | x (32) | 16 zeros]; w = blob `enc0.project` |
| `l_copy_channels` | src, sstride, soff, dst, dstride, doff, n, ntok | ceil(ntok n / 16 / 256) | dst[t][doff .. doff + n) = src[t][soff .. soff + n) (bytes; strides, offsets and n multiples of 16) |

## Launch order of one frame (default table)

Model M's frame (`clean/FORMATS.md`, "Stage order per frame") with steps 2 and 4 replaced:

2. Encoder (offsets of the block: table entry `window_offsets["first"]` for the frame parity):
   1. `input_encoder` (`l_enc_values.co`) -> `raw` [F/2][80], m_near, m_far, m_center
   2. `l_linear_80_32` raw -> `x1` [F/2][32]
   3. `l_linear_32_32` x1 -> `x2` [F/2][32]
   4. `l_swin_block_c32[_nc]` x2 -> `x3` [F/2][32] (H = fh / 2, W = fw / 2)
   5. `l_project_80_32` raw, x3 -> `enc0.features` [F/2][64] (what enc1 reads)
3. enc1 ... dec1: Model M's stages with Model L's weights, unchanged.
4. Decoder (offsets: `window_offsets["second"]`):
   1. `l_copy_channels` dec1.out (stride 64, offset 0) -> `y` [F/2][32]
   2. `l_swin_block_c32[_nc]` y -> `y2` [F/2][32]
   3. `l_copy_channels` y2 (32, 0) -> `z` (64, 0), n = 32; `l_copy_channels` dec1.out (64, 32) -> `z` (64, 32), n = 32
   4. Model M's `coef_decoder` with features = `z`, w = blob `dec0`

Extra buffers per frame: `raw` 80 ntok bytes, `x1` / `x2` / `x3` 32 ntok each (two can be shared), `y` / `y2` 32 ntok,
`z` 64 ntok. The launch list is derived from the table by `clean/l/runner/l_pack.py` `encoder_launches`.

## Checks (GPU, synthetic data)

`clean/l/runner/l_c32_check.py` and `clean/l/runner/l_chain_check.py`: every kernel in isolation against the oracle
step on the same input, the whole encoder chain, and the decoder front (see their docstrings for the rules).
