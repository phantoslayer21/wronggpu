# WrongGPU

**Model M upscaling and Ray Reconstruction on AMD Radeon RX 9000.**

WrongGPU runs Model M (super resolution) and, in games with DLSS Ray Reconstruction, its own Ray Reconstruction on Radeon RX 9000-series cards. The WrongGPU launcher sets up each game for you and gets the WrongGPU edition of OptiScaler for it. In the game, WrongGPU is its own upscaler in OptiScaler's menu, next to AMD FSR.

**[Download 0.6.0](https://github.com/phantoslayer21/wronggpu/releases/download/v0.6.0/WrongGPU-0.6.0.zip)** · [Release notes](https://github.com/phantoslayer21/wronggpu/releases/tag/v0.6.0) · [SHA-256 checksum](https://github.com/phantoslayer21/wronggpu/releases/download/v0.6.0/WrongGPU-0.6.0.zip.sha256) · [Discord](https://discord.gg/745z9MfKwm) · [Report a bug](https://github.com/phantoslayer21/wronggpu/issues/new?template=bug-report.yml)

**0.6.0 is a pre-release and still a work in progress.** In some games or scenes you may still see visual bugs, such as artifacts or flicker. If you do, please tell us (see [Something broken?](#something-broken)). Preset K returns in 0.6.1.

## Before you install

- **Windows 10 or 11, 64-bit.**
- **Radeon RX 9070 or RX 9070 XT.** The RX 9060 XT / RX 9060 is supported but hasn't been tested yet, and 8 GB cards are untested. Older Radeon cards aren't supported.
- **A recent AMD Adrenalin driver.** **No AMD HIP runtime is needed any more**: Model M runs on Direct3D 12. HIP is only used for the optional sparkle guard.
- **A DirectX 12 game.** WrongGPU compatibility depends on the game; see [game compatibility](COMPATIBILITY.md). Games with anti-cheat are never supported.
- **An internet connection the first time.** The launcher downloads the [WrongGPU edition of OptiScaler](https://github.com/phantoslayer21/OptiScaler/releases/tag/wronggpu-edition-20261102.1) (OptiScaler with the plugin interface WrongGPU's Ray Reconstruction uses), checks it against the size and SHA-256 built into the launcher, and keeps it for every game. Nothing of OptiScaler is inside WrongGPU's download. Offline, you can choose an OptiScaler download yourself in the launcher's Settings; games whose Ray Reconstruction needs the WrongGPU edition then get Model M only.
- **DLSS files that are already on your PC:** `nvngx_dlss.dll` version 310.5 or newer for Model M and, for Ray Reconstruction, a DLSS Ray Reconstruction file of version 310.7 or newer. Each can come from any game or app on your PC, not necessarily the game you play; the launcher looks for them.

The models are converted locally on your PC. No NVIDIA files or model data are in the download, and WrongGPU never downloads them. Don't upload or share the converted model files.

## Get it running

1. **Download and extract** `WrongGPU-0.6.0.zip`, then run **`WrongGPU-0.6.0.exe`**. The download isn't code-signed, so Windows may show "Windows protected your PC": click **More info**, check the file name, then **Run anyway**. You can check the download against its [SHA-256 checksum](https://github.com/phantoslayer21/wronggpu/releases/download/v0.6.0/WrongGPU-0.6.0.zip.sha256) first; the release notes explain how.
2. **First run:** the launcher checks Windows, your graphics card, the Direct3D 12 path and the model data. Then choose **Install for this user** (a Start menu entry and one entry in Installed apps) or **Run portable**.
3. **Install into a game:** the Library shows the games it finds in your Steam, Epic and GOG libraries (**Add game** takes one from anywhere else). Open a game and click **Install**. The first time, the launcher downloads the WrongGPU edition of OptiScaler. In games with Ray Reconstruction it also prepares Ray Reconstruction's GPU code once (about 45 seconds with a fresh driver).
4. **In the game:** choose **DLSS** as the upscaler in its graphics settings, and DLSS Ray Reconstruction for Ray Reconstruction. With TAA or upscaling off, the game never asks for an upscaler and WrongGPU stays idle.
5. **See it running:** turn on OptiScaler's FPS overlay with the game page's **Turn on the FPS overlay** button before you start the game. It shows **DLSS → WrongGPU Model M**, or **DLSS → WrongGPU RR** and its mode while Ray Reconstruction runs. After you quit, the game's page shows the run under **Last run**.

The launcher selects WrongGPU in OptiScaler's menu for you. Press **Insert** in the game to open OptiScaler's menu; AMD FSR stays available under OptiScaler's own FSR entries.

A few games need one more step:
- **Borderlands 4** needs one Engine.ini setting for Ray Reconstruction. Right after the install the launcher asks: **Fix it for me** adds it (and makes the file read-only, because the game removes it otherwise).
- **RESIDENT EVIL requiem** needs **REFramework**, installed by you first. WrongGPU never installs or downloads it; the launcher tells you when it is missing.

## What's new in 0.6.0?

- **The WrongGPU launcher replaces the setup.** One window for all your games: Install, Update, Repair, Roll back and Uninstall per game, a settings page per game with an Apply bar, an optional Play button, **Last run** (what the game used the last time it ran) and **Collect logs**. Games set up with 0.5.x are picked up and keep their settings.
- **No more hunting for OptiScaler.** The launcher downloads the WrongGPU edition of OptiScaler itself and checks it.
- **No AMD HIP runtime needed.** Model M runs on Direct3D 12 inside the game's own command list, in every mode up to native resolution (DLAA). Measured on an RX 9070 XT: about 1.3 ms a frame at 1080p Quality and about 2.2 ms at 1440p Quality on a synthetic benchmark; in Cyberpunk 2077, about 1.2 ms and 1.8 ms.
- **Ray Reconstruction (new).** In games that use DLSS Ray Reconstruction, WrongGPU denoises and upscales the ray-traced image with its own network. Three modes: **Fast** (default), **Accurate** and **Exact**. In Cyberpunk 2077 with path tracing at 1080p Quality on an RX 9070 XT, Fast takes about 3.35 ms a frame. The launcher prepares its GPU code for you, so the game's first start has it ready.
- **New games:** RESIDENT EVIL requiem (experimental, with Ray Reconstruction), Marvel's Spider-Man 2 and LEGO Batman (Model M).
- **Preset K** isn't part of 0.6.0; it returns in 0.6.1.

[Full release notes →](https://github.com/phantoslayer21/wronggpu/releases/tag/v0.6.0)

## How well does it work?

Tested on a Radeon RX 9070 XT:

- **Cyberpunk 2077:** Ray Reconstruction and Model M. Our main test game.
- **Borderlands 4:** Ray Reconstruction, with the one Engine.ini setting above.
- **RESIDENT EVIL requiem** (experimental): Ray Reconstruction with path tracing, and Model M. Needs your own REFramework.
- **Hogwarts Legacy:** Model M. Since the game's November 2026 update its Ray Reconstruction isn't available with WrongGPU; a fix is planned for 0.6.1.
- **Marvel's Spider-Man 2:** Model M. Ray Reconstruction there is an experimental opt-in and the game is unstable with it; we don't recommend it.
- **LEGO Batman:** Model M. The game's menu has no Ray Reconstruction option.
- **007 First Light** is not supported (it crashes at start under OptiScaler).

Known issues:
- Ray Reconstruction can still show occasional visual artifacts, such as ghosting on thin lines in reflections, or blotches in Cyberpunk 2077 with path tracing.
- RESIDENT EVIL requiem: very occasionally a single frame comes out far too bright or too dark at a scene change (after loading a save, or during flashes of lightning). We're investigating it for 0.6.1.

See [game compatibility and testing notes](COMPATIBILITY.md). Running well in one scene doesn't cover every setting or rendering mode. Compare WrongGPU and AMD FSR in the same scene and use whichever looks and runs better for you.

## Something broken?

On the game's page in the launcher, click **Collect logs**. It creates a support ZIP locally and doesn't upload anything itself. Check it before sharing; it can include your Windows username and other local file paths.

Post it in the **#bug-reports** channel of the [WrongGPU Discord server](https://discord.gg/745z9MfKwm) with a short description: the game and its version, the scene, your settings (resolution, quality mode, Ray Reconstruction on or off and its mode) and what went wrong. A screenshot or clip helps with visual issues.

**[Ask or report in Discord](https://discord.gg/745z9MfKwm)** · [Report a bug](https://github.com/phantoslayer21/wronggpu/issues/new?template=bug-report.yml) · [Report game compatibility](https://github.com/phantoslayer21/wronggpu/issues/new?template=game-compatibility-report.yml)

## Updating or removing it

Run the new version's launcher. Games set up with an older WrongGPU show an update; click **Update** on the game's page. Games set up with 0.5.x are taken over and keep their settings. Games set up with 0.4.x use the same install record and should be taken over the same way, but that hasn't been tested yet.

To remove WrongGPU from a game, use **More → Uninstall WrongGPU** on the game's page: it puts the game's own files back exactly as they were. To remove the launcher itself (installed for your user), go to **Windows Settings → Apps → Installed apps → WrongGPU**; it can remove WrongGPU from every game too. The model data stays while another WrongGPU game still needs it.

## Support WrongGPU

WrongGPU is free to download and use. If you'd like to support development, you can [leave an optional tip on Ko-fi](https://ko-fi.com/wronggpu). Donations are never required to download or use the mod.

## Licence

Free for **personal, non-commercial use** under the `LICENSE.txt` included in the release. See `wronggpu-notices.txt` for third-party notices. OptiScaler is a separate project with its own licence (GPL); the WrongGPU edition of OptiScaler has its own [release page](https://github.com/phantoslayer21/OptiScaler/releases/tag/wronggpu-edition-20261102.1) with its source and licence.

This repository is for downloads, documentation and issue tracking. Source code isn't published here.

WrongGPU is an independent project and isn't affiliated with or endorsed by NVIDIA, AMD or any game publisher. Trademarks belong to their respective owners.
