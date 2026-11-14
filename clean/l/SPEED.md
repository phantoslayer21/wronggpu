# Model L speed (preliminary)

**Status: preliminary.** Model L is in development for 0.7.0. It is available from source with its build option
(`--model-l`, described under "Building Model L" below). Speed work is ongoing (the encoder is next), and these
numbers will change.

Model L costs more than Model M because it has a full-resolution encoder and decoder, and its own reconstruction.

All figures are milliseconds per frame of the network (the work WrongGPU does for one upscaled frame), measured on
the GPU. Every figure comes from one of two records in [`records/`](records/): the first table from the in-game
record, the other tables from the harness record. The two records time slightly different things, so read each
table on its own and do not mix them.

## In game: Cyberpunk 2077, 1080p Quality

| Setting | Value |
|---|---|
| Game and scene | Cyberpunk 2077, gameplay street scene (standing still at one street spot, same save for every run) |
| Resolution and mode | 1080p Quality: 1280x720 upscaled to 1920x1080 |
| GPU and driver | AMD Radeon RX 9070 XT, driver 32.0.32015.7029 |
| Timer | the runtime's D3D12 timestamps in the game's command list; each sample is the mean of 300 frames, and each figure is the median over the scene's windows |
| Build | main 8ed26e8 plus pull requests #155 and #156 |
| Model L | its own reconstruction (the default): 36 windows, 10800 frames |
| Model M | same session, same settings: 43 windows, 12900 frames |

| Step | Model L (ms) | Model M (ms) |
|---|---|---|
| enc0 (input encoder) | 0.503 | 0.111 |
| Swin | 0.751 | 0.585 |
| dec0 | 0.345 | 0.189 |
| post | 0.651 | 0.289 |
| **Network total** | **2.250** | **1.174** |
| Frame's full D3D12 span | 2.256 | 1.180 |

So in this scene the network takes 2.25 ms with Model L and 1.17 ms with Model M, on the same PC in the same
session. "Network total" runs from the start of the network to the end of its last kernel. Record:
[`cp2077-model-l-timing-2026-11-14.json`](records/cp2077-model-l-timing-2026-11-14.json). The record also holds
title-screen scenes and a development variant that is not a supported output; this page does not use them.

## Harness: synthetic model and frames

| Setting | Value |
|---|---|
| Model and frames | synthetic model and synthetic frames (no game, no real weights) |
| Shapes | 1080p Quality, 1440p Quality, 1440p Performance (the record's shape names) |
| GPU and driver | not stored in the harness record |
| Timer | per-kernel GPU timestamps, which add overhead, so the harness reads higher than the game does |
| Runs | 5 runs per shape, models interleaved per round, one GPU slot per run |
| Build | main 6622148 |
| Tool | `clean/l/runner/l_d3d12_profile.py` (and `tools/prof_shapes.py` for 1440p Performance) |

Frame span, Model L / Model M:

| Shape | Model L (ms) | Model M (ms) |
|---|---|---|
| 1080p Quality | 2.36 | 1.32 |
| 1440p Quality | 4.00 | 2.17 |
| 1440p Performance | 2.86 | 1.59 |

Per-step split (ms, rounded to three decimals; "other" is the exposure step):

| Shape | Model | enc0 | Swin | dec0 | post | other |
|---|---|---|---|---|---|---|
| 1080p Quality | L | 0.545 | 0.783 | 0.349 | 0.733 | 0.012 |
| 1080p Quality | M | 0.114 | 0.589 | 0.169 | 0.424 | 0.010 |
| 1440p Quality | L | 0.991 | 1.216 | 0.638 | 1.370 | 0.020 |
| 1440p Quality | M | 0.202 | 0.942 | 0.335 | 0.767 | 0.020 |
| 1440p Performance | L | 0.551 | 0.782 | 0.351 | 1.301 | 0.013 |
| 1440p Performance | M | 0.122 | 0.596 | 0.171 | 0.697 | 0.012 |

What the steps cover in the harness (from the record): enc0 is Model L's input encoder, linears, encoder block and
gather, or Model M's input encoder; dec0 is Model L's decoder projection and decoder block, or Model M's `d1p12`
block with the decoder fused in; post is post, filter, `scale_rgb` and `color_divide`; Swin is the rest (for Model L
it includes its own dec1 block). The in-game timer cuts the network at its own points, so the step values in the two
tables are not directly comparable. Record: [`l-vs-m-shapes-main.json`](records/l-vs-m-shapes-main.json).

## Building Model L

Model L is off in a default build. It is built with the `--model-l` option of the two build scripts:

```
python clean/runner/build.py --model-l ...
python clean/dx12/build_dx12.py --model-l ...
```

The first puts Model L in the runtime library; the second builds its Direct3D 12 kernels. See
[BUILDING.md](../../BUILDING.md) for the rest of the build.
