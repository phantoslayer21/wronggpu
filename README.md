# WrongGPU

Run **DLSS 4.5 Model M neural upscaling** on **Radeon RX 9070 and RX 9070 XT** through AMD FSR in DirectX 12 games.

**[Download 0.3.1 — Experimental](https://github.com/phantoslayer21/wronggpu/releases/tag/v0.3.1)** · [Game compatibility](COMPATIBILITY.md) · [Discord](https://discord.gg/745z9MfKwm)

## Get started

1. Close the game. Download `WrongGPU-0.3.1.zip` from the release's **Assets** and extract it.
2. Run `WrongGPU-Setup-0.3.1.exe` and select your game. The installer finds Steam, Epic and GOG games, or you can select the game's executable or folder.
3. Let the installer find a compatible `nvngx_dlss.dll`. If it cannot, use **Model file...** to select one from another game you own.
4. Click **Install**. The installer backs up the game files it replaces.
5. In the game, select **AMD FSR 3 or newer** and a quality mode. Check that the WrongGPU panel shows **Model M running**.

Model M starts in the background. While the panel says **starting**, the game uses AMD FSR; neural upscaling takes over when initialization finishes.

**Cyberpunk 2077:** select **AMD FSR 3**. Its **FSR 2.1** and **FSR 4** options bypass WrongGPU. Its bundled DLSS 310.1 is too old for model conversion, so the installer needs a compatible file from another game.

## Requirements

- **64-bit Windows 10 or 11.**
- **Radeon RX 9070 or RX 9070 XT.** RX 9060-series and older Radeon cards are not supported by 0.3.1.
- **AMD Adrenalin drivers and a working AMD HIP runtime.** The installer checks this. If the runtime is unavailable, the Windows HIP SDK may be needed.
- **A DirectX 12 game with compatible AMD FSR 3 or newer DLLs.** The installer checks the game's integration. DirectX 11, Vulkan and FSR 2-only games are unsupported.
- **An `nvngx_dlss.dll` version 310.5 or newer that you are entitled to use**, already on your PC.
- **No anti-cheat.** Do not use WrongGPU in games with anti-cheat.

The installer converts the model locally from your DLSS file. NVIDIA's DLL and model weights are **not included in the download**. Keep converted model files on your own PC; do not upload or redistribute them.

## In-game controls

| Key | Action |
| --- | --- |
| **Insert** | Hide or show the overlay. Showing it restores the full panel for 10 seconds. |
| **Home** | Switch between WrongGPU and the game's AMD FSR upscaler for comparison. |

The panel shows whether Model M is running and the upscaler's cost per frame. Use it to confirm that neural inference is active before measuring performance.

## Compatibility and image quality

0.3.1 has been tested in **Cyberpunk 2077** and **Borderlands 4**. Other games are experimental, even when the installer accepts them.

- **Borderlands 4:** flickering at **4K Performance** is still being investigated.
- Game updates, different FSR integrations and untested settings can cause compatibility or image-quality problems.
- Path tracing and other unlisted modes have not been validated for 0.3.1.

See the [compatibility page](COMPATIBILITY.md) for game-specific notes. To help test another game, follow the [testing guide](docs/TESTING-OTHER-GAMES.md).

## Performance

Model M adds a significant cost per frame. **AMD FSR 4 is faster in the recorded comparisons.** WrongGPU offers another reconstruction option; the result depends on the game, resolution, quality mode, driver and GPU.

Compare the same scene and settings with **Home**, after Model M has finished starting. Upscaler time in milliseconds is only part of the total frame time. Stage-level improvements in the release notes are not overall FPS gains.

## Troubleshooting and reports

Use **Collect logs** in the installer, or **Modify** for your game's WrongGPU entry in Windows Installed apps. Collect logs before uninstalling, since uninstall removes that installation's logs.

For a report, include the game and version, WrongGPU version, GPU and driver, resolution and FSR mode, what happened, and a screenshot or clip where useful. Review the log ZIP before sharing it: it includes system information and file paths, which may contain your Windows user name.

- [Report a bug](https://github.com/phantoslayer21/wronggpu/issues/new?template=bug-report.yml)
- [Report game compatibility](https://github.com/phantoslayer21/wronggpu/issues/new?template=game-compatibility-report.yml)
- [Get help on Discord](https://discord.gg/745z9MfKwm)

The setup log is `%LOCALAPPDATA%\WrongGPU\setup.log`. The installer makes no network calls and sends no telemetry.

## Update or uninstall

Close the game before updating. If you have an older WrongGPU installation or another upscaler mod, follow the installer's prompts rather than copying DLLs over it manually.

To uninstall, open **Settings → Apps → Installed apps → WrongGPU - your game**. The uninstaller restores the backed-up game files and removes that installation's settings and logs. Shared model data is retained while another WrongGPU installation needs it.

## Licence

WrongGPU is free for **personal, non-commercial use**. See `LICENSE.txt` in the release for its terms and `wronggpu-notices.txt` for third-party notices. This repository provides downloads, documentation and issue tracking; source code is not published here.

WrongGPU is an independent project and is not affiliated with or endorsed by NVIDIA, AMD or any game publisher. Trademarks belong to their respective owners.
