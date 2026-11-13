# WrongGPU

**Model M upscaling and Ray Reconstruction on AMD Radeon RX 9000 graphics cards.**

WrongGPU runs the neural networks behind DLSS super resolution (Model M) and DLSS Ray Reconstruction on Radeon RX
9000-series cards. Its launcher sets up each of your games and gets the WrongGPU edition of OptiScaler for them. From
this source onward, WrongGPU's source code and its official builds are under the MIT licence (see [Licence](#licence)).

**Download:** the [Releases page](https://github.com/phantoslayer21/wronggpu/releases) (latest: 0.6.0, a pre-release).
**Help and bug reports:** the [WrongGPU Discord server](https://discord.gg/745z9MfKwm) (channel #bug-reports) or
[GitHub issues](https://github.com/phantoslayer21/wronggpu/issues).

**WrongGPU is a community project.** Anyone can report games, fix bugs, write tests or work on the
[roadmap](ROADMAP.md) by pull request; see [Contributing](#contributing). Start with the
[welcome discussion](https://github.com/phantoslayer21/wronggpu/discussions/1). The project owner reviews and merges,
and builds and publishes the official releases.

## For players

### What you need

- Windows 10 or 11 and a recent AMD Adrenalin driver.
- A Radeon RX 9070 or RX 9070 XT. The RX 9060 / RX 9060 XT is supported but not yet tested; older Radeon cards are not
  supported. No AMD HIP runtime is needed (only the optional sparkle guard uses it).
- A DirectX 12 game; games with anti-cheat are never supported.
- An internet connection the first time (the launcher downloads the WrongGPU edition of OptiScaler).
- DLSS files already on your PC, from any game or app: a DLSS super resolution file of version 310.5 or newer for
  Model M, and a DLSS Ray Reconstruction file of version 310.7 or newer for Ray Reconstruction. WrongGPU converts them
  on your PC; the download contains no files or model data of that vendor, and the converted model files are yours
  alone (please don't share them).

### Install in three steps

1. Download `WrongGPU-<version>.zip` from the Releases page, extract it and run `WrongGPU-<version>.exe` (it is not
   code-signed: if Windows SmartScreen warns, choose **More info → Run anyway** after checking the file name and the
   published SHA-256).
2. Pick **Install for this user** or **Run portable**, open a game in the Library and click **Install**.
3. In the game, choose DLSS as the upscaler (with DLSS Ray Reconstruction where the game offers it). The game's page in
   the launcher shows what ran under **Last run**; **More → Uninstall WrongGPU** puts the game's own files back exactly.

Borderlands 4 needs one Engine.ini setting for Ray Reconstruction (the launcher offers to add it); RESIDENT EVIL
requiem needs REFramework, which you install yourself.

### Games (0.6.0, tested on an RX 9070 XT)

| Game | Model M | Ray Reconstruction |
|---|---|---|
| Cyberpunk 2077 | yes | yes (main test game) |
| Borderlands 4 | yes (tested with 0.4.5) | yes, with the Engine.ini setting |
| RESIDENT EVIL requiem | yes | yes, experimental (needs your own REFramework) |
| Hogwarts Legacy | yes | not available since the game's November 2026 update |
| Marvel's Spider-Man 2 | yes (choose FSR in the game) | experimental opt-in, not recommended |
| LEGO Batman | yes | the game offers none |
| 007 First Light | not supported (crashes at start under OptiScaler) | |

Other games are experimental; the full list, with requirements and how to add your game, is
[COMPATIBILITY.md](COMPATIBILITY.md). Known issues in 0.6.0: Ray Reconstruction can still show occasional artifacts (ghosting
on thin lines in reflections, blotches in Cyberpunk 2077 with path tracing); in RESIDENT EVIL requiem a single frame
can very occasionally come out far too bright or dark at a scene change. Full details: the release notes
([clean/installer/RELEASE-NOTES.md](clean/installer/RELEASE-NOTES.md)).

### Something broken?

On the game's page in the launcher, click **Collect logs**. It makes a zip on your PC and uploads nothing; check it
before you share it (it can contain your Windows user name and local paths). Post it in #bug-reports on the
[Discord server](https://discord.gg/745z9MfKwm) or attach it to a [GitHub issue](https://github.com/phantoslayer21/wronggpu/issues),
with the game, the scene and your settings.

WrongGPU is free. If you want to support development, there is an optional tip jar on
[Ko-fi](https://ko-fi.com/wronggpu); it is never required.

## For developers

WrongGPU is an independent, clean-room implementation: the clean team that wrote its code never saw DLSS source code,
disassembly or another port's kernels (see [docs/CLEAN-ROOM.md](docs/CLEAN-ROOM.md), which also says who made what: the
converter maps and test vectors in `data/` come from the separate spec team, and a few launcher / front-end helpers were
adapted from WrongGPU 0.2.0 code). No NVIDIA DLSS file is included in this repository or in WrongGPU's packages. The network's weights come from the player's own DLSS file
(`nvngx_dlss.dll` / `nvngx_dlssd.dll`, for DLSS super resolution and DLSS Ray Reconstruction): WrongGPU's converter
reads it on the player's PC and writes WrongGPU's own model file there. Nothing from that file is ever redistributed.

This repository is the source of WrongGPU 0.6.0, published as one snapshot. The documents under `clean/` are the
implementation's own design and format notes (per-file provenance records are kept privately by the maintainer). They refer to the clean-room specification ("spec 02",
"page 51", ...), to the clean team's questions (`QUESTIONS.md`, `ANSWERS.md`) and to internal results notes
(`results/...`); those are not part of this repository.

### How it works

```
 player's DLSS file ──> converter (clean/converter) ──> model.wgcm  (on the player's PC, never shipped)
                                                            │
 game ──> OptiScaler / FSR interface ──>   FSR front end (clean/fsr) ──> runtime (clean/runtime, clean/rr_f/runtime)
                                                                              │
                                    Direct3D 12 kernels (clean/dx12, clean/rr_f/kernels) or HIP kernels (clean/kernels)
```

- **Converter** (`clean/converter`): finds the network's tensors in the player's file using a map
  (`data/preset-M.v1.map.json`, `data/preset-F.v1.map.json`: offsets, strides and digests, no weights) and writes
  them as WrongGPU's model file.
- **Runtime** (`clean/runtime`, Ray Reconstruction in `clean/rr_f`): runs the network once per frame on the game's
  Direct3D 12 device: input encoder, Swin transformer blocks, coefficient decoder, output reconstruction and filter,
  auto-exposure; for Ray Reconstruction its window blocks, head, output stage and resolve.
- **Kernels**: Direct3D 12 compute shaders (`clean/dx12/kernels`, `clean/rr_f/kernels`, compiled with DXC) and HIP
  kernels for gfx1201 / gfx1200 (`clean/kernels`, compiled with the AMD HIP SDK).
- **FSR front end** (`clean/fsr`): a FidelityFX API provider (`amd_fidelityfx_dx12.dll`) that OptiScaler or the game
  calls as its upscaler; it hands each frame to the runtime, or to AMD's FSR when WrongGPU cannot run.
- **Ray Reconstruction plugin** (`clean/rr_f/proxy`): an ASI plugin that answers the game's Ray Reconstruction calls.
- **Launcher** (`clean/installer`): finds the games, installs and uninstalls WrongGPU per game (byte-exact restore),
  gets OptiScaler, runs the converter, and shows what ran.
- **Reference and tests**: NumPy reference implementations (`clean/reference`, `clean/k/reference`,
  `clean/rr_f/reference`), synthetic test vectors (`data/`, `clean/**/tests`), equivalence tools (`clean/equiv`).
  No model data and no game frame captures are included. Test data is synthetic, except the launcher's log fixtures
  (`clean/installer/tests/logs/`), some of which are trimmed excerpts of WrongGPU's own logs from game sessions, with
  process ids, device pointers and paths replaced (see that folder's README.md and docs/CLEAN-ROOM.md).

### Repository layout

| Path | Content |
|---|---|
| `clean/converter` | the model converter (C++) and its tests |
| `clean/runtime`, `clean/runner`, `clean/kernels` | Model M runtime library, build script, HIP kernels |
| `clean/dx12` | Direct3D 12 kernels of Model M and their build (`build_dx12.py`) |
| `clean/rr_f` | Ray Reconstruction: reference, kernels, runtime, plugin, tests |
| `clean/k` | Model K (preset K; built and tested, not offered in 0.6.0) |
| `clean/fsr` | FSR front end (FidelityFX API provider) |
| `clean/installer` | the WrongGPU launcher |
| `clean/wgshared` | settings registry shared by the front end, the plugin and the launcher |
| `clean/reference`, `clean/equiv`, `clean/tests` | NumPy references, equivalence tools, unit tests |
| `clean/third_party` | AMD FidelityFX SDK headers (MIT) |
| `data` | converter maps and synthetic test vectors |

Building: [BUILDING.md](BUILDING.md). Security issues: [SECURITY.md](SECURITY.md). Code of conduct:
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Contributing

WrongGPU is developed in the open by its community. Ways to help:

- **Players:** test games and report them ([COMPATIBILITY.md](COMPATIBILITY.md), "How to add your game"); report
  bugs with the launcher's **Collect logs** zip.
- **Developers:** pick an item from [ROADMAP.md](ROADMAP.md) or an open issue, and open a pull request. A good first
  step: [build WrongGPU from a fresh clone](https://github.com/phantoslayer21/wronggpu/issues/2) and report where
  BUILDING.md breaks.

Every pull request: the clean-room statement and a `Signed-off-by` line on each commit ([CONTRIBUTING.md](CONTRIBUTING.md)),
passing CI (licence, line endings, CPU builds and tests on Windows), then a maintainer's clean-room scan, review and,
when the change touches kernels, the runtime, the front end or the Ray Reconstruction plugin, a test on a Radeon
RX 9000-series card before the owner merges.

## Licence

From this source onward, WrongGPU's source code and the official WrongGPU builds made from it are under the MIT licence
([LICENSE](LICENSE)); this statement covers builds of this source and later, not packages published before it.
Third-party components and their licences: [THIRD-PARTY-NOTICES](THIRD-PARTY-NOTICES). OptiScaler, which the launcher
downloads, is a separate project under its own licence (GPL-3.0).

DLSS is a trademark of NVIDIA Corporation; AMD, Radeon, FidelityFX and FSR are trademarks of Advanced Micro Devices,
Inc. WrongGPU is an independent project, not affiliated with or endorsed by either company or by any game publisher.
