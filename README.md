# WrongGPU

**DLSS 4.5 Model M upscaling on AMD Radeon.**

WrongGPU adds Model M to OptiScaler's upscaler menu. Pick it the same way you would any other upscaler, and switch back to AMD FSR through the same menu.

**[Download 0.4.5](https://github.com/phantoslayer21/wronggpu/releases/download/v0.4.5/WrongGPU-0.4.5.zip)** · [Release notes](https://github.com/phantoslayer21/wronggpu/releases/tag/v0.4.5) · [SHA-256 checksum](https://github.com/phantoslayer21/wronggpu/releases/download/v0.4.5/WrongGPU-0.4.5.zip.sha256) · [Discord](https://discord.gg/745z9MfKwm) · [Report a bug](https://github.com/phantoslayer21/wronggpu/issues/new?template=bug-report.yml)

**0.4.5 is experimental and Model M only. Preset K isn't included yet.** This release focuses on stability, memory cleanup and compatibility. **OptiScaler is now required.**

## Before you install

- **Windows 10 or 11, 64-bit.**
- **Radeon RX 9070 or RX 9070 XT.** RX 9060 XT code is included, but has not been tested on physical hardware. More testing is needed, especially on the 8 GB model. Older Radeon cards aren't supported by this release.
- **AMD Adrenalin drivers and a working HIP runtime.** The installer checks for the runtime.
- **A DirectX 12 game supported by OptiScaler.** WrongGPU compatibility still depends on the game; see [game compatibility](COMPATIBILITY.md). Games with anti-cheat aren't supported.
- **Your own [OptiScaler download](https://github.com/optiscaler/OptiScaler/releases).** Keep the archive or extracted folder handy. WrongGPU doesn't include or download OptiScaler.
- **An `nvngx_dlss.dll` version 310.5 or newer that you are entitled to use.** The installer can look for one in your installed games.

The model is converted locally on your PC. NVIDIA's DLL and model weights aren't included in the download. Don't upload or redistribute the converted model files.

## Get it running

1. **Close the game.** Download and extract `WrongGPU-0.4.5.zip`.
2. Run **`WrongGPU-Setup-0.4.5.exe`**. Give it your OptiScaler archive or folder, then select your game.
3. Let the installer find a compatible model file, or select your own `nvngx_dlss.dll`. Follow the setup prompts and install.
4. Start the game and select **DLSS or FSR input** in its graphics settings, as supported by OptiScaler.
5. Press **Insert** to open OptiScaler, then select **`3.1.5 WrongGPU (Model M)`** from the upscaler list.

The leading `3.1.5` is a version prefix OptiScaler uses for compatibility. The upscaler is WrongGPU's Model M. Switch between Model M and AMD FSR, including FSR 4 where available, through **OptiScaler's menu**.

OptiScaler's FPS header may still show an FSR version with WrongGPU selected. The selected menu entry alone also doesn't prove neural inference stayed active: check the WrongGPU log when diagnosing a problem. Inputs without depth, such as Hogwarts Legacy's XeSS input, fall back to AMD FSR. **Use DLSS or FSR inputs.**

Direct FSR DLL replacement is no longer available. When upgrading an older direct install, the installer restores the game's original FSR DLLs and sets up the OptiScaler route.

## What's changed in 0.4.5?

- Replaced the experimental frame-input wait path implicated in reported freezes with standard GPU synchronization.
- Fixed the reproduced video-memory leak when changing quality mode, resolution or upscalers.
- Fixed command-list handling behind first-frame crashes in games such as The Witcher 3.
- Keeps a recently destroyed upscaler warm for three seconds to reduce repeated initialization work.
- Recovers from a single inconsistent frame during a resolution change instead of permanently switching to AMD FSR.
- Correctly handles motion vectors supplied at output resolution.
- Updated the output filter. The previous look is available with `output_filter=analytic` in `runtime_options`.
- Fixed installer game detection and upgrades over manually replaced WrongGPU DLLs.

[Full release notes →](https://github.com/phantoslayer21/wronggpu/releases/tag/v0.4.5)

## How well does it work?

**Cyberpunk 2077 is validated.** The final 0.4.5 installer also ran Model M in Hogwarts Legacy, Borderlands 4 and The Witcher 3 Remastered on an RX 9070 XT through OptiScaler 0.9.5.4. Those games remain experimental.

Earlier 0.4.5 test builds ran Control, Crimson Desert and Ratchet & Clank: Rift Apart. That isn't full validation of the final release in those games. **007 First Light is currently unsupported because of a startup crash.**

Known limits include an incorrect FSR label in OptiScaler's FPS header, occasional plain upscaled frames when a new AMD-driver command list is first encountered, and memory retention on a rarely used sharing fallback path. The log warns when that path is active. Longer testing of The Witcher 3's dynamic-resolution behavior is still pending.

See [game compatibility and testing notes](COMPATIBILITY.md). Running successfully in one scene doesn't establish every setting or rendering mode. Compare Model M and AMD FSR in the same scene and use whichever looks and runs better for you.

## Something broken?

Use **Collect logs** in the installer before uninstalling. It creates a support ZIP locally and doesn't upload anything itself. Check it before sharing; it can include your Windows username and other local file paths.

Include your game and version, GPU, driver, resolution, quality mode and what went wrong. A screenshot or clip helps with visual issues.

**[Report a bug](https://github.com/phantoslayer21/wronggpu/issues/new?template=bug-report.yml)** · [Report game compatibility](https://github.com/phantoslayer21/wronggpu/issues/new?template=game-compatibility-report.yml) · [Ask in Discord](https://discord.gg/745z9MfKwm)

## Updating or removing it

Close the game and run the new installer. Follow its prompts if you already have WrongGPU or another upscaler mod installed. Upgrading an older direct install keeps your `wronggpu.ini` and restores the game's original FSR files before setting up OptiScaler.

To uninstall, go to **Windows Settings → Apps → Installed apps → WrongGPU - your game**. The uninstaller restores the backed-up game files. Shared model data stays while another WrongGPU installation still needs it.

## Support WrongGPU

WrongGPU is free to download and use. If you'd like to support development, you can [leave an optional tip on Ko-fi](https://ko-fi.com/wronggpu). Donations are never required to download or use the mod.

## Licence

Free for **personal, non-commercial use** under the `LICENSE.txt` included in the release. See `wronggpu-notices.txt` for third-party notices. OptiScaler is a separate project with its own licence.

This repository is for downloads, documentation and issue tracking. Source code isn't published here.

WrongGPU is an independent project and isn't affiliated with or endorsed by NVIDIA, AMD or any game publisher. Trademarks belong to their respective owners.
