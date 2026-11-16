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

## Building Model L

Model L is off in a default build. It is built with the `--model-l` option of the two build scripts:

```
python clean/runner/build.py --model-l ...
python clean/dx12/build_dx12.py --model-l ...
```

The first puts Model L in the runtime library; the second builds its Direct3D 12 kernels. See
[BUILDING.md](../../BUILDING.md) for the rest of the build.
