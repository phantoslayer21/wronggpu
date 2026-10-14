# Game compatibility

These notes apply to **WrongGPU 0.4.5**, running Model M through **OptiScaler**. Direct replacement of the game's FSR DLLs is no longer supported.

## Tested games

| Game | Current status | Notes |
| --- | --- | --- |
| **Cyberpunk 2077** | Validated | Model M ran with the final 0.4.5 installer on an RX 9070 XT through OptiScaler 0.9.5.4. Supply an eligible DLSS 310.5+ model source if the game's bundled file is older. |
| **Hogwarts Legacy** | Experimental; final installer tested | Model M ran on an RX 9070 XT. Use DLSS or FSR input; XeSS input lacks the depth WrongGPU needs and falls back to AMD FSR. Repeated quality switches were used to verify the main memory-leak fix. |
| **Borderlands 4** | Experimental; final installer tested | Model M ran after updating an older direct installation to OptiScaler. This does not establish every quality mode or resolve every earlier visual report. |
| **The Witcher 3 Remastered** | Experimental; final installer tested | Model M ran after the command-list fix. Longer gameplay testing of dynamic-resolution spikes is still pending. |
| **Control** | Experimental; earlier 0.4.5 test build | Model M ran in an earlier test build; the final release has not received equivalent validation. This result does not cover Control Resonant. |
| **Crimson Desert** | Experimental; earlier 0.4.5 test build | Model M ran in an earlier test build; final-release stability and image quality across settings are not fully validated. |
| **Ratchet & Clank: Rift Apart** | Experimental; earlier 0.4.5 test build | Model M ran in an earlier test build; the final release has not received equivalent validation. |
| **007 First Light** | Unsupported | Startup crash with WrongGPU as OptiScaler's FSR library, including AMD-forwarding mode. OptiScaler's own built-in FSR ran in the control. |
| **Other games** | Experimental / unverified | OptiScaler support and installer acceptance alone do not establish WrongGPU compatibility. [Send a report](https://github.com/phantoslayer21/wronggpu/issues/new?template=game-compatibility-report.yml). |

A successful run does not establish every game build, graphics setting, quality mode or frame-generation configuration. Results from earlier test builds are identified separately from the final installer.

## Requirements for another game

- A **DirectX 12 game supported by OptiScaler**, with inputs that provide the color, depth and motion information WrongGPU needs.
- Your own OptiScaler archive, folder or DLL. The installer sets up WrongGPU as its FSR library.
- **Radeon RX 9070 or RX 9070 XT**, AMD Adrenalin drivers and a working AMD HIP runtime.
- An eligible local **`nvngx_dlss.dll` 310.5 or newer** for conversion.
- **No anti-cheat.** Do not bypass an installer refusal.

RX 9060 XT code is included, but physical hardware qualification is pending, especially for the 8 GB model. Older Radeon cards, DirectX 11 and Vulkan are unsupported in this release.

## Known limits

- OptiScaler's FPS header may display an FSR version even with WrongGPU selected. The menu entry is `3.1.5 WrongGPU (Model M)`; check the WrongGPU log to confirm actual neural activity.
- Inputs without depth can silently fall back to AMD FSR. Use DLSS or FSR input; the log records the handoff.
- A handful of plain upscaled frames can occur when new AMD-driver command lists are first encountered, mostly at startup.
- A rarely used memory-sharing fallback still retains memory across upscaler restarts. The log warns when it is used.
- The three-second warm pool temporarily retains about 0.8 GB of VRAM.
- Preset K is not included. Sparkle guard remains experimental and off by default.

## Testing and reports

Follow [Testing WrongGPU in other games](docs/TESTING-OTHER-GAMES.md). AMD fallback, AMD comparison mode and plain diagnostic frames are not Model M inference.

Identify the game build, GPU and driver, WrongGPU and OptiScaler versions, output resolution, input upscaler and quality mode. Include the support ZIP, reproduction steps, visual problems and matched performance comparisons. Community reports help establish compatibility, but no fixed report count automatically makes a game validated.
