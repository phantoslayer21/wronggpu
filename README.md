# WrongGPU

**DLSS 4.5 Model M upscaling on AMD Radeon.**

WrongGPU adds Model M to OptiScaler's upscaler menu. Pick it the same way you would any other upscaler, and switch back to FSR 4.1.1 through the same menu.

**[Download 0.4.1](https://github.com/phantoslayer21/wronggpu/releases/download/v0.4.1/WrongGPU-0.4.1-Sun-Fix.zip)** · [Release notes](https://github.com/phantoslayer21/wronggpu/releases/tag/v0.4.1) · [Discord](https://discord.gg/745z9MfKwm) · [Report a bug](https://github.com/phantoslayer21/wronggpu/issues/new?template=bug-report.yml)

Still experimental. The focus right now is getting it working in more games and making it easier to use. **This release is Model M only; Preset K isn't included yet.**

## Before you install

- **Windows 10 or 11, 64-bit.**
- **Radeon RX 9070 or RX 9070 XT.** RX 9060 XT support is included, but needs more testing on actual cards, especially the 8 GB model. Older Radeon cards aren't supported by this release.
- **AMD Adrenalin drivers and a working HIP runtime.** The installer checks for the runtime.
- **A DirectX 12 game supported by OptiScaler.** Don't use this in games with anti-cheat.
- **Your own [OptiScaler download](https://github.com/optiscaler/OptiScaler/releases).** Keep the archive or extracted folder handy.
- **An `nvngx_dlss.dll` version 310.5 or newer that you are entitled to use.** The installer can look for one in your installed games.

The model is converted locally on your PC. NVIDIA's DLL and model weights aren't included in the download. Don't upload or redistribute the converted model files.

## Get it running

1. **Close the game.** Download and extract `WrongGPU-0.4.1-Sun-Fix.zip`.
2. Run **`WrongGPU-Setup-0.4.1.exe`**. Give it your OptiScaler archive or folder, then select your game.
3. Let the installer find a compatible model file, or select your own `nvngx_dlss.dll`. Follow the setup prompts and install.
4. Start the game and select **DLSS or XeSS** in its graphics settings.
5. Press **Insert** to open OptiScaler, then select **WrongGPU (Model M)** from the upscaler list.

That's the default setup now. Switch between Model M and FSR 4.1.1 in **OptiScaler's menu**. Some OptiScaler versions show an older FSR version number in the overlay header even with WrongGPU selected; check the selected provider in the menu.

Direct FSR installation is still available under the installer's advanced option if you need it.

## What's changed in 0.4.1?

Everything from the unreleased 0.4.0 is included too.

- OptiScaler is now the default, with a clearer installer layout and setup steps.
- Fixed the black dots and arc around the sun in Cyberpunk 2077.
- Fixed the reproduced Crimson Desert startup stalls.
- Added the RX 9060 XT kernels and improved HIP runtime detection.
- Includes runtime performance work since 0.3.1, compatibility fixes and safer update/uninstall cleanup.
- Better support logs through the installer's **Collect logs** button.

[Full release notes and checksum →](https://github.com/phantoslayer21/wronggpu/releases/tag/v0.4.1)

## How well does it work?

It depends on the game and settings. The Cyberpunk sun fix was checked in fresh game runs, but that doesn't mean every game or rendering mode is sorted.

- **Crimson Desert:** the startup fix is included; ground shimmer is still unresolved.
- **Borderlands 4:** flickering at 4K Performance is still a known issue.
- **RX 9060 XT:** more feedback is needed, especially from 8 GB cards.

Model M has a higher cost per frame than AMD FSR 4 in the recorded comparisons. Try both in the same scene through OptiScaler and use whichever looks and runs better for you.

## Something broken?

Use **Collect logs** in the installer before uninstalling. It creates a support ZIP locally and doesn't upload anything itself. Check it before sharing; it can include your Windows username and other local file paths.

Include your game and version, GPU, driver, resolution, quality mode and what went wrong. A screenshot or clip helps with visual issues.

**[Report a bug](https://github.com/phantoslayer21/wronggpu/issues/new?template=bug-report.yml)** · [Report game compatibility](https://github.com/phantoslayer21/wronggpu/issues/new?template=game-compatibility-report.yml) · [Ask in Discord](https://discord.gg/745z9MfKwm)

## Updating or removing it

Close the game and run the new installer. Follow its prompts if you already have WrongGPU or another upscaler mod installed.

To uninstall, go to **Windows Settings → Apps → Installed apps → WrongGPU - your game**. The uninstaller restores the backed-up game files. Shared model data stays while another WrongGPU installation still needs it.

## Licence

Free for **personal, non-commercial use** under the `LICENSE.txt` included in the release. See `wronggpu-notices.txt` for third-party notices. OptiScaler is a separate project with its own licence.

This repository is for downloads, documentation and issue tracking. Source code isn't published here.

WrongGPU is an independent project and isn't affiliated with or endorsed by NVIDIA, AMD or any game publisher. Trademarks belong to their respective owners.
