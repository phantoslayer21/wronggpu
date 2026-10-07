# Game compatibility

These notes apply to **WrongGPU 0.3.1**, which runs Model M through the game's **AMD FSR** integration.

## Tested games

| Game | Current status | Notes |
| --- | --- | --- |
| **Cyberpunk 2077** | Tested | Select **AMD FSR 3**. FSR 2.1 and FSR 4 bypass WrongGPU. The game's bundled DLSS 310.1 is too old for conversion; supply a compatible 310.5+ file from another game you own. |
| **Borderlands 4** | Tested, with a known issue | The HIP startup failure observed during testing was addressed in 0.3.1. Flickering at **4K Performance** remains under investigation. |
| **Other games** | Experimental | A compatible FSR integration is required. Installer acceptance alone does not establish game compatibility. [Send a report](https://github.com/phantoslayer21/wronggpu/issues/new?template=game-compatibility-report.yml). |

“Tested” means the mod has run in the game. It does not mean every graphics setting, game version or quality mode has been verified. Path tracing and other unlisted modes have not been validated for 0.3.1.

## Requirements for another game

- The game uses **DirectX 12**.
- It contains compatible **AMD FSR 3 or newer DLLs** that the installer recognizes. An FSR option alone does not guarantee a compatible integration.
- It runs on supported hardware: **Radeon RX 9070 or RX 9070 XT**, with a working AMD HIP runtime.
- It has **no anti-cheat**. Do not bypass an installer refusal.

DirectX 11, Vulkan and FSR 2-only integrations are unsupported. RX 9060-series and older Radeon cards are unsupported in this version.

## Testing and reports

Follow [Testing WrongGPU in other games](docs/TESTING-OTHER-GAMES.md). Check that the overlay confirms **Model M running**; performance measured during startup or AMD comparison mode is not WrongGPU neural performance.

Reports should identify the game build, GPU and driver, WrongGPU version, output resolution and FSR mode. Include logs, visual problems and any matched performance comparison. Community reports help establish compatibility, but no fixed report count automatically makes a game validated.
