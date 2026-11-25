# Model L speed

**Status: under test.** Model L ships with 0.6.5 as a choice per game, off by default; Model M stays the default.
In a build from source it needs its build option (`--model-l`, described under "Building Model L" below). Speed
work is ongoing, and these numbers will change.

Model L costs more than Model M because it has a full-resolution encoder and decoder, and its own reconstruction.

All figures are milliseconds per frame of the network (the work WrongGPU does for one upscaled frame), measured on
the GPU. Every figure comes from a record in [`records/`](records/), named under its table. The records time
slightly different things (in game, harness, and an earlier harness record for Model M), so read each table on its
own and do not mix them.

## In game: Cyberpunk 2077, 1080p Quality

| Setting | Value |
|---|---|
| Game and scene | Cyberpunk 2077, gameplay street scene (standing still at one street spot, same save for every run) |
| Resolution and mode | 1080p Quality: 1280x720 upscaled to 1920x1080 |
| GPU and driver | AMD Radeon RX 9070 XT, driver 32.0.32015.7029 |
| Timer | the runtime's D3D12 timestamps in the game's command list; each sample is the mean of 300 frames, and each figure is the median over the scene's windows |
| Build | main 25429c5, release configuration |
| Model L | the release form (hybrid, the default on Direct3D 12), chosen with `set <game> model=L`: 43 windows, 12900 frames |
| Model M | the control, timed in the same sitting just before Model L, same settings, save and spot: 49 windows, 14700 frames |

| Step | Model L (ms) | Model M (ms) |
|---|---|---|
| enc0 (input encoder) | 0.528 | 0.111 |
| Swin | 1.097 | 0.582 |
| dec0 | 0.476 | 0.188 |
| post | 0.565 | 0.287 |
| **Network total** | **2.666** | **1.168** |
| Frame's full D3D12 span | 2.677 | 1.174 |

So in this scene the network takes 2.67 ms with Model L and 1.17 ms with Model M, on the same PC in the same
sitting. "Network total" runs from the start of the network to the end of its last kernel. Record:
[`cp2077-model-l-timing-2026-11-16.json`](records/cp2077-model-l-timing-2026-11-16.json). The earlier record
[`cp2077-model-l-timing-2026-11-14.json`](records/cp2077-model-l-timing-2026-11-14.json) timed an earlier form of
Model L that this release does not ship.

## Harness: Model L, 1440p Quality, Direct3D 12

| Setting | Value |
|---|---|
| Harness | WrongGPU's Direct3D 12 test harness (no game), the direct route, executor dx12 |
| Shape | 1440p Quality: 1707x960 upscaled to 2560x1440 |
| Model L | the release form with no form option set (the default), as built by the change that made it the default |
| Runs | 5 rounds of 1500 frames, interleaved with the other configurations of the record; each figure is the median of the rounds |
| GPU and driver | not stored in the record |

| Step | Model L (ms) |
|---|---|
| enc0 (input encoder) | 1.032 |
| Swin | 1.825 |
| dec0 | 0.903 |
| post | 1.081 |
| **Frame** | **4.85** |

So in the harness at 1440p Quality a Model L frame takes 4.85 ms. Record:
[`l_time_1440q_hybrid_d3d12.json`](records/l_time_1440q_hybrid_d3d12.json) (configuration `hybdef`). Model L's
other shapes will be re-measured with this form; until then this page shows no harness figures for them.

## Harness: Model M, from an earlier harness record

These come from a different record (synthetic models, per-kernel times, build main 6622148 of 2026-11-14, 5 runs per
shape), so do not compare them with the Model L harness table above.

| Shape | Frame span (ms) | enc0 | Swin | dec0 | post | other (exposure) |
|---|---|---|---|---|---|---|
| 1080p Quality | 1.32 | 0.114 | 0.589 | 0.169 | 0.424 | 0.010 |
| 1440p Quality | 2.17 | 0.202 | 0.942 | 0.335 | 0.767 | 0.020 |
| 1440p Performance | 1.59 | 0.122 | 0.596 | 0.171 | 0.697 | 0.012 |

Record: [`l-vs-m-shapes-main.json`](records/l-vs-m-shapes-main.json) (its Model L rows timed an earlier form of
Model L and are not shown).

## Fast mode (opt-in, `l_mode=fast`)

The release form of Model L on Direct3D 12 is the hybrid form (`l_mode=hybrid`, the default: float32 chains with FP16
rounding at each step, the form that matches the reference bit for bit). `l_mode=fast` is an opt-in preset for players
who prefer speed: it runs Model L's network on Model M's Swin kernels (the WMMA / FP8 `_nc` forms, with Model M's
fusions) on Model L's own weights, so its output follows the model closely but is not bit-exact to the reference. The
structure is Model L's: the decoder block adds the encoder block's output as its skip, the reconstruction and the
temporal history are the hybrid route's own kernels (the C1/C2 reconstruction, the fused output stage).

Select it with the runtime option `l_mode=fast` (in the game's `runtime_options` line) or the environment variable
`WG_L_MODE=fast`; the option wins over the variable, and nothing set (or `l_mode=hybrid`) is the hybrid default, whose
kernels and launches the preset does not touch. Fast mode runs on native Direct3D 12 only and is refused with its
reason where a kernel it needs is missing from the build (a build made with `--model-l`; the bottleneck needs the
build's `variants.json` and `swin__bmegahg.co`). No timing is published for it yet: a figure goes on this page only
with its measurement record.

The levers of the preset (each a kernel of the build named `<kernel>_fast`, launched in place of its base kernel with the
same arguments; the base kernels and everything `l_mode=hybrid` runs are untouched, and a build without a twin simply runs
the base kernel). They drop the bit-exactness of the hybrid route where it costs instructions and keeps the model's values:

| lever | kernel | what it changes |
|---|---|---|
| 1 | `w64_lc12/l_post_tf_p12[_pf]_fast` | the fused reconstruction + output filter without the driver intrinsic channel and the opaque-zero guards; native FP16 packs with the round-to-nearest bias folded in; the gather's weight is one `v_exp_f16` of the FP16 exponent and the total weight is summed in FP16 from the same weights as the nine features; no 1 KB code table; `Dec` as `exp2(q k) - 1` for every lane; the horizontal filter ring in FP16 pairs (LDS 26 752 -> 23 040 bytes) |
| 2 | `l_project_gather_fast`, `l_dec_project_fast` | the K blocks accumulate in FP32 and round once, in the FP8 pack (no FP16 rounding after every 32 inputs) |
| 4 | `l_swin_block_c32_enc_lin_nc_fast` | the two input linears of the encoder block keep their results in FP32 (no FP16 rounding of x1, x2) |
| 16 | (option of lever 1) | the weaker history candidate (weight at most 1/2) is a 4-tap bilinear of the previous history instead of the 12-tap Catmull-Rom |
| 64 | (option of lever 1) | a weaker history candidate whose motion is within 0.01 render pixel of the first one's is the same sample and is not read twice |
| 128 | (option of lever 1, off by default) | a weaker history candidate of weight below 0.05 is dropped (the hybrid route drops it below 1 / 255) |

A developer build (`dev_diagnostics`) reads the selection of levers from `WG_L_FAST_MASK` (the sum of the lever values
above; the default is 127: every lever and option but 128) for an A/B of each; a release build has no such variable. The arithmetic choices are bounded
on the CPU by `clean/l/tests/test_l_fast_numerics.py` (the quality gate itself is measured on the GPU against
`l_mode=hybrid`: tone-mapped PSNR / SSIM and temporal stability).

## Building Model L

Model L is off in a default build. It is built with the `--model-l` option of the two build scripts:

```
python clean/runner/build.py --model-l ...
python clean/dx12/build_dx12.py --model-l ...
```

The first puts Model L in the runtime library; the second builds its Direct3D 12 kernels. See
[BUILDING.md](../../BUILDING.md) for the rest of the build.
