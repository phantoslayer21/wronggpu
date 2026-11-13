# Game compatibility

These notes apply to **WrongGPU 0.6.0**: Model M and Ray Reconstruction through the **WrongGPU edition of OptiScaler**,
set up by the WrongGPU launcher. "Ray Reconstruction" means WrongGPU's own Ray Reconstruction ran in the game; Model M
means super resolution only. This page grows with the community's reports: see [How to add your game](#how-to-add-your-game).

## Tested games

Tested on a Radeon RX 9070 XT.

| Game | Model M | Ray Reconstruction | Notes |
| --- | --- | --- | --- |
| **Cyberpunk 2077** | Yes | Yes | Our main test game. |
| **Borderlands 4** | Yes (tested with 0.4.5) | Yes | Ray Reconstruction needs one Engine.ini setting, and the game removes it on every start. After the install the launcher asks: **Fix it for me** adds the setting and makes the file read-only (uninstalling WrongGPU undoes both); **Show me how** gives the steps. The game creates Ray Reconstruction itself, so it needs the WrongGPU edition of OptiScaler. About 1.6 ms a frame at 1080p output (from 642x361). |
| **RESIDENT EVIL requiem** | Yes | Yes, experimental | Needs **REFramework**, installed by you first: the game blocks OptiScaler without it. WrongGPU never installs, downloads or replaces REFramework; the launcher tells you when it is missing. Ray Reconstruction runs when the game's **Ray Tracing** setting is **Path Tracing** and its upscaler is **DLSS**. The game prepares Ray Reconstruction at its first start, not in the launcher. Very occasionally a single frame comes out far too bright or too dark at a scene change (see Known limits). |
| **Hogwarts Legacy** | Yes | Not available | Since the game's November 2026 update its Ray Reconstruction option is greyed out and WrongGPU's Ray Reconstruction isn't available, so the launcher doesn't offer it there. Ray Reconstruction worked before that update. |
| **Marvel's Spider-Man 2** | Yes (choose **FSR** as the game's upscaler: its menu doesn't offer DLSS) | Experimental opt-in, not recommended | With Ray Reconstruction the game often freezes the GPU or crashes. The release notes explain the opt-in for those who want to try it anyway. |
| **LEGO Batman** | Yes | Not offered | The game's menu has no Ray Reconstruction option. |
| **The Witcher 3 Remastered** | Earlier version (0.4.5) | — | Not retested with 0.6.0. |
| **Control** | Earlier versions | — | Not retested with 0.6.0. This result does not cover Control Resonant. |
| **Crimson Desert** | Earlier versions | — | Not retested with 0.6.0. |
| **Ratchet & Clank: Rift Apart** | Earlier versions | — | Not retested with 0.6.0. |
| **007 First Light** | Unsupported | Unsupported | Crashes at start under OptiScaler. |
| **Other games** | Experimental / unverified | Experimental / unverified | OptiScaler support and the launcher accepting a game alone do not establish WrongGPU compatibility. [Send a report](https://github.com/phantoslayer21/wronggpu/issues/new?template=game-compatibility-report.yml). |

A successful run does not cover every game build, graphics setting, quality mode or frame-generation configuration.
Results from earlier WrongGPU versions are labelled as such. The Borderlands 4 figure is Ray Reconstruction's own GPU
time per frame from WrongGPU's log of a test run.

## Requirements for another game

- A **DirectX 12 game supported by OptiScaler**, with upscaler inputs that provide the color, depth and motion
  information WrongGPU needs. Choose **DLSS** as the game's upscaler.
- **Radeon RX 9070 or RX 9070 XT** (RX 9060 XT / RX 9060 supported, not yet tested) and a recent AMD Adrenalin driver.
  No AMD HIP runtime is needed.
- A local DLSS super resolution file of version **310.5 or newer** for Model M and, for Ray Reconstruction, a **DLSS Ray
  Reconstruction file 310.7 or newer**, each from any game or app on your PC (not necessarily the game you play). The
  launcher looks for them in your installed games.
- For Ray Reconstruction, the game must use DLSS Ray Reconstruction through game interface version 2.7, 2.8 or 2.14, or
  create it directly. Games on other game interface versions keep their own denoiser. Games that create Ray
  Reconstruction themselves (for example Unreal Engine games with their own plugin) need the WrongGPU edition of
  OptiScaler.
- **No anti-cheat.** Do not bypass a launcher refusal.

Older Radeon cards, DirectX 11 and Vulkan are unsupported in this release.

## Known limits

- Ray Reconstruction can still show occasional visual artifacts, such as ghosting on thin lines in reflections, or
  blotches in Cyberpunk 2077 with path tracing.
- RESIDENT EVIL requiem: very occasionally a single frame comes out far too bright or too dark at a scene change (after
  loading a save, or during flashes of lightning). It lasts one frame.
- Preset K is not included in 0.6.0.
- Radeon RX 9060 XT / RX 9060: supported, not yet tested.
- Without a DLSS Ray Reconstruction file of version 310.7 or newer on your PC, the game keeps its own denoiser and the
  launcher says so.
- After an AMD driver update (or a WrongGPU update) the launcher prepares Ray Reconstruction again in the background; a
  game started before that is done prepares it at its first start.
- Frames the network can't use (for example a render size larger than the output, or textures too small during an
  intro) are passed through or go to AMD FSR, and WrongGPU carries on with the next normal frame.

## How to add your game

Anyone can add a game to this page.

1. Test it as [Testing WrongGPU in other games](docs/TESTING-OTHER-GAMES.md) describes: a game without anti-cheat, the
   FPS overlay or the game page's **Last run** showing that WrongGPU's Model M (or Ray Reconstruction) really ran, at
   least one longer play session, and the settings you used.
2. Report it with the [game compatibility form](https://github.com/phantoslayer21/wronggpu/issues/new?template=game-compatibility-report.yml)
   (or in #bug-reports on the [WrongGPU Discord server](https://discord.gg/745z9MfKwm)), with the zip from **Collect
   logs**.
3. Optionally open a pull request that adds a row to the table above: the game, Model M and Ray Reconstruction
   results, what you tested on (GPU, driver, WrongGPU version) and a link to your report. A timing goes in only with
   the log line it comes from (game, resolution, mode, WrongGPU version). One report marks a game as "reported"; a
   maintainer moves it to "tested" after checking it.

## Testing and reports

Follow [Testing WrongGPU in other games](docs/TESTING-OTHER-GAMES.md). AMD FSR frames and passed-through frames are not
Model M or Ray Reconstruction results: check the FPS overlay ("DLSS → WrongGPU Model M" or "DLSS → WrongGPU RR") and
the game page's **Last run**.

Identify the game build, GPU and driver, WrongGPU version, output resolution, input upscaler, quality mode and whether
Ray Reconstruction was on (and its mode). Include the support zip from **Collect logs** on the game's page in the
launcher, reproduction steps, visual problems and matched performance comparisons. Community reports help establish
compatibility, but no fixed report count automatically makes a game validated.
