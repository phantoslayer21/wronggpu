# WrongGPU

**Model M upscaling and Ray Reconstruction on AMD Radeon RX 9000 graphics cards.**

WrongGPU runs its own implementation of the neural networks behind DLSS super resolution (Model M) and DLSS Ray
Reconstruction on Radeon RX 9000-series cards. The networks' weights come from your own DLSS files; the arithmetic is
WrongGPU's, so the output is close to DLSS's but not the same. Its launcher sets up each of your games and gets the
WrongGPU edition of OptiScaler for them.

**Download:** the [Releases page](https://github.com/phantoslayer21/wronggpu/releases) (latest:
[0.6.5](https://github.com/phantoslayer21/wronggpu/releases/tag/v0.6.5), a pre-release).
**Help and bug reports:** the [WrongGPU Discord server](https://discord.gg/745z9MfKwm) (channel #bug-reports) or
[GitHub issues](https://github.com/phantoslayer21/wronggpu/issues).

## For players

### What you need

- Windows 10 or 11 and a recent AMD Adrenalin driver.
- A Radeon RX 9070 or RX 9070 XT. The RX 9060 / RX 9060 XT is supported but not yet tested; older Radeon cards are not
  supported. No AMD HIP runtime is needed for upscaling or Ray Reconstruction; the optional sparkle guard and frame
  capture use it.
- A DirectX 12 game; games with anti-cheat are never supported.
- An internet connection the first time, so the launcher can download the WrongGPU edition of OptiScaler.
- DLSS files already on your PC, from any game or app: a DLSS super resolution file of version 310.5 or newer for
  Model M and Model L, and a DLSS Ray Reconstruction file of version 310.7 or newer for Ray Reconstruction. Model K is
  also converted from a DLSS file from one of your games. WrongGPU converts them on your PC; the download contains no
  files or model data of that vendor, and the converted model files are yours alone (please don't share them).

### Install in three steps

1. Download `WrongGPU-<version>.zip` from the Releases page, extract it and run `WrongGPU-<version>.exe`. It is not
   code-signed: if Windows SmartScreen warns, choose **More info → Run anyway** after checking the file name and the
   published SHA-256.
2. Pick **Install for this user** or **Run portable**, open a game in the Library and click **Install**.
3. In the game, choose DLSS as the upscaler, with DLSS Ray Reconstruction where the game offers it. The game's page in
   the launcher shows what ran under **Last run**. **More → Uninstall WrongGPU** puts the game's own files back exactly.

Borderlands 4 needs one Engine.ini setting for Ray Reconstruction; the launcher offers to add it. RESIDENT EVIL requiem
needs REFramework, which you install yourself.

### Tested games

Tested on an RX 9070 XT, with results from 0.6.0 unless a row says otherwise.

| Game | Model M | Ray Reconstruction |
|---|---|---|
| Cyberpunk 2077 | yes | yes (main test game) |
| Borderlands 4 | yes (tested with 0.4.5) | yes, with the Engine.ini setting |
| RESIDENT EVIL requiem | yes | yes, experimental (needs your own REFramework) |
| Hogwarts Legacy | yes | not available since the game's November 2026 update |
| Marvel's Spider-Man 2 | yes (choose FSR in the game) | experimental opt-in, not recommended |
| LEGO Batman | yes | the game offers none |
| 007 First Light | not supported (crashes at start under OptiScaler) | |

0.6.5 was re-tested in Cyberpunk 2077 at 1080p: Model M at Quality and at DLAA, Model L and Model K at Quality, and
Ray Reconstruction with path tracing, with 0 failures logged.

Other games are experimental. See [COMPATIBILITY.md](COMPATIBILITY.md) for the full list, each game's requirements and
how to add your game.

### Model L and Model K

Model L and Model K are two more super-resolution models, new in 0.6.5, off by default and still under test. Model M
stays the default. Each is converted once on your PC from a DLSS model file of one of your games:

```bat
WrongGPU.exe model convert --preset=L
WrongGPU.exe model convert --preset=K
```

Choose one per game with `WrongGPU.exe set "<the game's .exe>" model=L` (or `model=K`), or in OptiScaler's menu.

In Cyberpunk 2077 at 1080p Quality on an RX 9070 XT, standing still at the same spot and measured one right after the
other, the model's GPU time per frame (all of its kernels) was 1.03 ms with Model K and 1.17 ms with Model M
([measurement](clean/k/records/cp2077-model-k-timing-2026-11-15.json)). Model L took 2.67 ms, measured the same way in
a later sitting with Model M again at 1.17 ms
([measurement](clean/l/records/cp2077-model-l-timing-2026-11-16.json)).

Faster kernels in 0.6.5 produce the same output. Model L's Swin stage takes 7.0 µs less per frame and its decoder stage
2.5 µs less at 1440p Quality, and 2.5 µs and 1.0 µs less at 1080p Quality. Model M's final (post) stage takes 2.5 µs
less at 1440p Quality and 1.5 µs less at 1080p Quality. These are median differences per stage over five rounds of an
offline benchmark running frames back to back on an RX 9070 XT; in a game the saving per frame can differ.

### Known issues and troubleshooting

- A rare crash at game start can happen; restarting the game fixes it.
- Some games or scenes can still show visual bugs, such as artifacts or flicker. With Ray Reconstruction these include
  ghosting on thin lines in reflections and blotches in Cyberpunk 2077 with path tracing.
- In RESIDENT EVIL requiem, a single frame very occasionally comes out far too bright or too dark at a scene change.

Full details are in the [0.6.5 release notes](clean/installer/RELEASE-NOTES.md). For help, click **Collect logs** on the
game's page in the launcher. It makes a zip on your PC and uploads nothing. Check the zip before sharing it: it can
contain your Windows user name and local paths. Post it in #bug-reports on [Discord](https://discord.gg/745z9MfKwm) or
attach it to a [GitHub issue](https://github.com/phantoslayer21/wronggpu/issues), with the game, scene and your settings.

WrongGPU is free. If you want to support development, there is an optional tip jar on
[Ko-fi](https://ko-fi.com/wronggpu); it is never required.

## For developers

WrongGPU is an independent, clean-room implementation: the clean team that wrote its code never saw DLSS source code,
disassembly or another port's kernels. [docs/CLEAN-ROOM.md](docs/CLEAN-ROOM.md) describes the process and who made what:
the converter maps and test vectors in `data/` come from the separate spec team, and a few launcher / front-end helpers
were adapted from WrongGPU 0.2.0 code. No NVIDIA DLSS file is included in this repository or in WrongGPU's packages. The
network's weights come from the player's own DLSS file (`nvngx_dlss.dll` / `nvngx_dlssd.dll`, for DLSS super resolution
and DLSS Ray Reconstruction): WrongGPU's converter reads it on the player's PC and writes WrongGPU's own model file
there. Nothing from that file is ever redistributed.

This repository was first published as one snapshot of 0.6.0 and now carries the source of 0.6.5 (tag `v0.6.5`). The
documents under `clean/` are the implementation's design and format notes. They refer to the clean-room specification,
internal questions and answers, and internal results notes that are not part of this repository. Per-file provenance
records are kept privately by the maintainer.

### How it works

```text
player's DLSS file ──> converter (clean/converter) ──> model.wgcm (on the player's PC, never shipped)
                                                           │
game ──> OptiScaler / FSR interface ──> FSR front end (clean/fsr) ──> runtime (clean/runtime, clean/rr_f/runtime)
                                                                         │
                       Direct3D 12 kernels (clean/dx12, clean/rr_f/kernels) or HIP kernels (clean/kernels)
```

The converter finds tensors using maps (`data/preset-M.v1.map.json`, `data/preset-F.v1.map.json`) containing offsets,
strides and digests, but no weights. The runtime runs the network once per frame on the game's Direct3D 12 device. The
FSR front end hands each frame to the runtime, or to AMD's FSR when WrongGPU cannot run. The Ray Reconstruction plugin
(`clean/rr_f/proxy`, an ASI plugin) answers the game's Ray Reconstruction calls. The launcher finds games, installs and
uninstalls WrongGPU with byte-exact restore, gets OptiScaler, runs the converter and shows what ran.

### Repository layout

| Path | Content |
|---|---|
| `clean/converter` | the model converter (C++) and its tests |
| `clean/runtime`, `clean/runner`, `clean/kernels` | Model M runtime library, build script, HIP kernels |
| `clean/dx12` | Direct3D 12 kernels of Model M and their build (`build_dx12.py`) |
| `clean/rr_f` | Ray Reconstruction: reference, kernels, runtime, plugin, tests |
| `clean/k` | Model K (a per-game choice since 0.6.5, off by default) |
| `clean/fsr` | FSR front end (FidelityFX API provider, `amd_fidelityfx_dx12.dll`) |
| `clean/installer` | the WrongGPU launcher |
| `clean/wgshared` | settings registry shared by the front end, the plugin and the launcher |
| `clean/reference`, `clean/equiv`, `clean/tests` | NumPy references, equivalence tools, unit tests |
| `clean/third_party` | AMD FidelityFX SDK headers (MIT) |
| `data` | converter maps and synthetic test vectors |

Model M's runtime includes the input encoder, Swin transformer blocks, coefficient decoder, output reconstruction and
filter, and auto-exposure. Ray Reconstruction adds its window blocks, head, output stage and resolve. Direct3D 12 compute
shaders are compiled with DXC; HIP kernels for gfx1201 / gfx1200 are compiled with the AMD HIP SDK. Model K also has
NumPy reference code in `clean/k/reference`, and Ray Reconstruction in `clean/rr_f/reference`.

Tests use synthetic vectors in `data/` and `clean/**/tests`; no model data or game frame captures are included. The only
non-synthetic test data is in `clean/installer/tests/logs/`: some fixtures are trimmed excerpts of WrongGPU's own game
logs, with process ids, device pointers and paths replaced. See that folder's README.md and docs/CLEAN-ROOM.md.

Building: [BUILDING.md](BUILDING.md). Security issues: [SECURITY.md](SECURITY.md). Code of conduct:
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Contributing

WrongGPU is developed in the open by its community. Start with the
[welcome discussion](https://github.com/phantoslayer21/wronggpu/discussions/1). The project owner reviews and merges
changes, and builds and publishes the official releases.

- **Players:** test games and report them using [COMPATIBILITY.md](COMPATIBILITY.md), or report bugs with the launcher's
  **Collect logs** zip.
- **Developers:** pick an item from [ROADMAP.md](ROADMAP.md) or an open issue and open a pull request. A good first step
  is to [build WrongGPU from a fresh clone](https://github.com/phantoslayer21/wronggpu/issues/2) and report where
  BUILDING.md breaks.

Every pull request needs the clean-room statement and a `Signed-off-by` line on each commit, as
[CONTRIBUTING.md](CONTRIBUTING.md) explains. CI checks licences, line endings, and CPU builds and tests on Windows. A
maintainer then performs a clean-room scan and review. Changes to kernels, the runtime, the front end or the Ray
Reconstruction plugin also need a test on a Radeon RX 9000-series card before the owner merges.

## License

0.7.0 and later: GPL-3.0-only. Releases up to 0.6.5 and earlier commits remain available under MIT.
See [LICENSE](LICENSE) for the GNU General Public License version 3.
Third-party components and their licences: [THIRD-PARTY-NOTICES](THIRD-PARTY-NOTICES). OptiScaler, which the launcher
downloads, is a separate project under its own licence (GPL-3.0).

DLSS is a trademark of NVIDIA Corporation; AMD, Radeon, FidelityFX and FSR are trademarks of Advanced Micro Devices,
Inc. WrongGPU is an independent project, not affiliated with or endorsed by either company or by any game publisher.
