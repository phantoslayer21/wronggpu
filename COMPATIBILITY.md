# Game compatibility

These results apply to **WrongGPU 0.6.5**, set up by the WrongGPU launcher. Model M is the default. Model L and Model K
are off by default. Games use the **WrongGPU edition of OptiScaler** unless a game-specific note says otherwise.

Model M means super resolution only. Ray Reconstruction means that WrongGPU's own Ray Reconstruction ran in the game.
Each result keeps the WrongGPU version that was tested. Cyberpunk 2077 was re-tested with 0.6.5. See
[How to add your game](#how-to-add-your-game) to report another result.

## Tested games

Tests used a Radeon RX 9070 XT and WrongGPU 0.6.0 unless a row says otherwise.

| Game | Model M | Ray Reconstruction | Notes |
| --- | --- | --- | --- |
| **Cyberpunk 2077** | Yes | Yes | The main test game. See the [0.6.5 retest](#cyberpunk-2077-retest). |
| **Borderlands 4** | Yes, tested with 0.4.5 | Yes | Requires one `Engine.ini` setting. See the [setup and timing notes](#borderlands-4-notes). |
| **RESIDENT EVIL requiem** | Yes | Yes, experimental | Requires your own REFramework. See the [game notes](#resident-evil-requiem-notes). |
| **Hogwarts Legacy** | Yes | Not available | The game's November 2026 update greyed out its Ray Reconstruction option. WrongGPU's Ray Reconstruction is unavailable, and the launcher does not offer it. Ray Reconstruction worked before that update. |
| **Marvel's Spider-Man 2** | Yes. Choose **FSR** because the game's menu does not offer DLSS. | Experimental opt-in, not recommended | Ray Reconstruction often freezes the GPU or crashes. The [release notes](clean/installer/RELEASE-NOTES.md#experimental-ray-reconstruction-in-spider-man-2) explain the opt-in. |
| **LEGO Batman** | Yes | Not offered | The game's menu has no Ray Reconstruction option. |
| **The Witcher 3 Remastered** | Earlier version, 0.4.5 | — | Not retested with 0.6.0 or 0.6.5. |
| **Control** | Earlier versions | — | Not retested with 0.6.0 or 0.6.5. This result does not cover Control Resonant. |
| **Crimson Desert** | Earlier versions | — | Not retested with 0.6.0 or 0.6.5. |
| **Ratchet & Clank: Rift Apart** | Earlier versions | — | Not retested with 0.6.0 or 0.6.5. |
| **007 First Light** | Unsupported | Unsupported | Crashes at start under OptiScaler. |
| **Other games** | Experimental and unverified | Experimental and unverified | OptiScaler support and the launcher accepting a game do not establish WrongGPU compatibility. [Send a report](https://github.com/phantoslayer21/wronggpu/issues/new?template=game-compatibility-report.yml). |

### Cyberpunk 2077 retest

WrongGPU 0.6.5 was tested at 1080p with Model M at Quality and DLAA, Model L and Model K at Quality, and Ray
Reconstruction with path tracing. The test logged 0 failures.

### Borderlands 4 notes

Ray Reconstruction needs one `Engine.ini` setting, which the game removes on every start. After installation, the
launcher offers two choices. **Fix it for me** adds the setting and makes the file read-only. Uninstalling WrongGPU
undoes both changes. **Show me how** provides the manual steps.

The game creates Ray Reconstruction itself, so it needs the WrongGPU edition of OptiScaler.

### RESIDENT EVIL requiem notes

Install **REFramework** yourself before installing WrongGPU. The game blocks OptiScaler without it. WrongGPU never
installs, downloads, or replaces REFramework, and the launcher reports when it is missing.

To use Ray Reconstruction, set the game's **Ray Tracing** setting to **Path Tracing** and its upscaler to **DLSS**. The
game prepares Ray Reconstruction at its first start, not in the launcher.

A single frame can very occasionally become far too bright or too dark at a scene change. See
[Known limits](#known-limits).

## Test coverage and the Borderlands 4 timing

A successful run does not cover every game build, graphics setting, quality mode or frame-generation configuration.
Results from earlier WrongGPU versions are labelled as such. The Borderlands 4 figure is the GPU time per frame that
WrongGPU's Ray Reconstruction runtime logged for its instance 1 (the `runtime: instance 1 timing` lines of
`clean/installer/tests/logs/bl4-rr-only.log`, a trimmed excerpt of WrongGPU's plugin log from a test run): about 1.59 ms
per frame in the logged timing windows, render 642x361, output 1920x1080. The instance was destroyed after 36044
frames with 0 failed. The last line of that log, "RR instances created 0, RR evaluates 0", counts only the
game-interface route. Borderlands 4 creates Ray Reconstruction itself through the direct route, so those zeros are
expected and do not mean that Ray Reconstruction did not run. The Ray Reconstruction mode and the WrongGPU version of that run are not recorded in the excerpt.

## Requirements for another game

- Use a **DirectX 12 game supported by OptiScaler**. Its upscaler inputs must provide the color, depth, and motion
  information that WrongGPU needs. Choose **DLSS** as the game's upscaler.
- Use a **Radeon RX 9070 or RX 9070 XT** and a recent AMD Adrenalin driver. The RX 9060 XT and RX 9060 are supported
  but not yet tested. Older Radeon cards are unsupported in this release.
- No AMD HIP runtime is needed for upscaling or Ray Reconstruction. The optional sparkle guard and frame capture use
  it.
- Keep a local DLSS super-resolution file of version **310.5 or newer** for Model M and Model L. Model K is converted from
  a DLSS file from one of your games. Ray Reconstruction needs a **DLSS Ray Reconstruction file of version 310.7 or
  newer**. Each file can come from any game or app on your PC, not necessarily the game you play. The launcher looks
  for these files in your installed games.
- For Ray Reconstruction, the game must use DLSS Ray Reconstruction through game interface version 2.7, 2.8, or 2.14,
  or create it directly. Games on other game interface versions keep their own denoiser. Games that create Ray
  Reconstruction themselves, such as Unreal Engine games with their own plugin, need the WrongGPU edition of
  OptiScaler.
- **Do not test a game with anti-cheat.** Do not bypass a launcher refusal.

DirectX 11 and Vulkan are unsupported in this release.

## Known limits

- A rare crash can happen at game start. Restarting the game fixes it.
- Ray Reconstruction can show visual artifacts, such as ghosting on thin lines in reflections or blotches in
  Cyberpunk 2077 with path tracing.
- In RESIDENT EVIL requiem, a single frame can very occasionally become far too bright or too dark at a scene change,
  after loading a save, or during flashes of lightning. It lasts one frame.
- Model L and Model K were added in 0.6.5, are off by default, and are still under test. Model L is tested for super
  resolution in Cyberpunk 2077 and RESIDENT EVIL requiem. Model K is tested in Cyberpunk 2077. See the
  [release notes](clean/installer/RELEASE-NOTES.md#known-issues).
- The Radeon RX 9060 XT and RX 9060 are supported but not yet tested.
- Without a DLSS Ray Reconstruction file of version 310.7 or newer on your PC, the game keeps its own denoiser and the
  launcher says so.
- After an AMD driver update or a WrongGPU update, the launcher prepares Ray Reconstruction again in the background. A
  game started before that finishes prepares it at its first start.
- WrongGPU passes through frames that the network cannot use or sends them to AMD FSR, then continues with the next
  normal frame. Examples include a render size larger than the output and textures that are too small during an intro.

## How to add your game

Anyone can add a game to this page.

1. Follow [Testing WrongGPU in other games](docs/TESTING-OTHER-GAMES.md). Use a game without anti-cheat and complete
   at least one longer play session. Confirm that the FPS overlay or the game page's **Last run** shows WrongGPU's
   Model M, Model L, Model K, or Ray Reconstruction running. Record your settings.
2. Submit the [game compatibility form](https://github.com/phantoslayer21/wronggpu/issues/new?template=game-compatibility-report.yml)
   or post in #bug-reports on the [WrongGPU Discord server](https://discord.gg/745z9MfKwm). Include the zip from
   **Collect logs**.
3. Optionally open a pull request that adds a row to the table. Include the game, the Model M and Ray Reconstruction
   results, the GPU, the driver, the WrongGPU version, and a link to your report. Include the source log line for every
   timing, together with the game, resolution, mode, and WrongGPU version. One report marks a game as "reported". A
   maintainer moves it to "tested" after checking it.

## Check what ran before reporting

AMD FSR frames and passed-through frames are not Model M, Model L, Model K, or Ray Reconstruction results. Check the
FPS overlay for **DLSS → WrongGPU Model M**, **WrongGPU Model L**, **WrongGPU Model K**, or **DLSS → WrongGPU RR**.
Also check the game page's **Last run**.

Identify the game build, GPU, driver, WrongGPU version, output resolution, input upscaler, and quality mode. State
whether Ray Reconstruction was on and include its mode. Attach the support zip from **Collect logs** on the game's
page in the launcher. Include reproduction steps, visual problems, and matched performance comparisons.

Community reports help establish compatibility, but no fixed report count automatically makes a game validated.
