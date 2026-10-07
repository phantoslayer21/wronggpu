# WrongGPU

> **Public access to WrongGPU is paused until further notice.** Downloads are not available right now. Updates will be posted on the Discord: https://discord.gg/745z9MfKwm

Run the DLSS 4.5 Super Resolution model on your AMD RDNA 4 GPU, using its native FP8 hardware, in DirectX 12 games. Validated on Cyberpunk 2077. Other games are experimental and tested by the community. Join the Discord for support and testing: https://discord.gg/745z9MfKwm

## Download

Get the latest `WrongGPU-Installer-v<version>.zip` from the [Releases page](../../releases/latest), unzip it and run `WrongGPU-Installer.exe`. Every stable release is free for everyone.

## Install

1. Unzip the download and run **WrongGPU-Installer.exe**. Windows may say it protected your PC because the installer is not code-signed yet: click More info, then Run anyway.
2. Pick your game. The installer lists the DLSS games it finds in Steam, Epic and GOG, or you can choose any game `.exe` or folder. Cyberpunk 2077 is **Validated**. Every other game is **Experimental** and needs you to tick a confirmation box. Games with anti-cheat are refused.
3. Give it a DLSS file. The installer needs an `nvngx_dlss.dll` from DLSS **310.5 or newer** (the version is under Properties > Details). It searches your installed games first. If none of them is new enough, click **DLSS DLL...** and choose a copy you are licensed to use, for example from another game you own that ships DLSS 310.5 or newer. The installer builds the DLSS 4.5 model files on your PC from that file, in moments. The download itself contains no NVIDIA files. The converted model file (`dlss45-amd.pak` in the game folder) is for your own use on your own PC. Do not share or upload it.
4. Click Install. The installer backs up every file it replaces.
5. In the game's graphics settings, turn on DLSS Super Resolution and pick a quality mode.

To uninstall, open Settings > Apps > Installed apps, find the entry for the mod and your game, and choose Uninstall. It restores your original files. Each game has its own entry.

## What to expect

Cyberpunk 2077 (game version 2.31), RX 9070 XT, DLSS Quality, Ray Tracing Psycho, sharpening off, the game's built-in benchmark. All numbers are from the same session; FSR 4 ran through OptiScaler 0.9.4:

- **1080p:** 74.7 FPS native, **110.1 FPS** with WrongGPU (FSR 4: 118.3 FPS)
- **1440p:** 47.2 FPS native, **74.3 FPS** with WrongGPU (FSR 4: 83.5 FPS)
- **4K:** 22.9 FPS native, **37.3 FPS** with WrongGPU (FSR 4: 43.4 FPS)

Your results will vary with your system, game version and settings. Other games have not been measured. FSR 4 is faster today. This project is about running the DLSS 4.5 model on AMD at better-than-native speed, and performance is still being optimized.

This is alpha software. Expect rough edges, such as shimmer on thin geometry (fences, foliage) and screen-space reflections.

**Path tracing in Cyberpunk 2077 is currently broken with WrongGPU.** Keep path tracing off. Regular ray tracing, up to Psycho, works.

## In-game overlay

Press **Insert** in game to open or close the overlay. It shows the real presentation FPS, the model preset, the resolution, how many frames the model has processed and any errors. It can also show an FPS counter while closed.

## Requirements

- Windows 11
- AMD Radeon RX 9070 or RX 9070 XT (RDNA 4). Older cards are not supported. Validated on an RX 9070 XT.
- AMD Adrenalin driver 32.0.31035.1003 or newer (the validated driver)
- AMD HIP SDK for Windows (ROCm 7.x), installed under `C:\Program Files\AMD\ROCm`. The installer checks for it.
- An `nvngx_dlss.dll` from DLSS 310.5 or newer, from one of your games or any copy you have (see Install, step 3).
- A DirectX 12 game with a DLSS Super Resolution option. **Validated: Cyberpunk 2077.** Every other game is experimental until community reports validate it (see [COMPATIBILITY.md](COMPATIBILITY.md)).
- Your own copy of the game.
- **Do not use it in games with anti-cheat.** The mod replaces DLL files and presents an NVIDIA graphics card to the game, so anti-cheat can treat it as tampering and ban the account. Single-player only.

## Other games

The installer is built to work beyond Cyberpunk 2077, but only Cyberpunk 2077 has been validated. Every other game is untested until someone reports back, and those reports are how a game gets validated. If you want to try one, read [docs/TESTING-OTHER-GAMES.md](docs/TESTING-OTHER-GAMES.md) and send a game compatibility report. The current status of every game is in [COMPATIBILITY.md](COMPATIBILITY.md).

## Troubleshooting and logs

- **Collect logs:** click **Collect logs** in the installer, or choose **Modify** for your game in Installed apps. It makes one zip with a system summary, the mod's logs and the install record, and tells you what is inside before you share it. The file paths in it include your Windows user name.
- The installer's own log is `%LOCALAPPDATA%\DLSS45-AMD\setup.log`. After a successful install a copy is kept in the game's `dlss45-amd` folder.
- If an install fails, the installer undoes what it did, so the game folder is left as it was.
- Uninstalling removes the mod's logs, so collect logs first if you are reporting a problem.
- The installer makes no network calls and sends no telemetry. Nothing leaves your PC unless you attach the logs zip to a report yourself.

## Other mods

OptiScaler and DLSS-NR on AMD also take over DLSS, so the installer moves them into its backup while this mod is installed and puts them back when you uninstall. If another mod already uses `version.dll` (for example Cyber Engine Tweaks), the loader uses a free name instead (`winmm.dll`, `winhttp.dll` or `dbghelp.dll`). If all of them are taken, the installer refuses and changes nothing.

## How can I help?

- Join the Discord (https://discord.gg/745z9MfKwm) and report bugs. Please attach the zip from **Collect logs**.
- Test other games and post a report in #compatibility on the Discord, or open a game compatibility report here on GitHub. A good report has the game and version, your GPU and driver, whether it launches, whether the image looks right, FPS with and without the mod, and your log files (see [docs/TESTING-OTHER-GAMES.md](docs/TESTING-OTHER-GAMES.md)). This is how new games get validated.
- A Patreon is coming soon; the link will be announced on the Discord. Everything here stays free for everyone.

## FAQ

**Does this contain NVIDIA code or model files?**
The download contains no NVIDIA DLLs and no DLSS model weights. The installer builds the model files on your PC from your own `nvngx_dlss.dll`.

**Does it work in other games?**
The installer is universal, but only Cyberpunk 2077 is validated. Other DirectX 12 games are experimental. Back up first, use the uninstaller if anything goes wrong, and report what you find so more games can be validated. The status of every game is in [COMPATIBILITY.md](COMPATIBILITY.md).

**Is it better than FSR 4?**
FSR 4 is faster at the moment. WrongGPU offers the DLSS 4.5 model's reconstruction on AMD hardware. Compare them yourself and tell us what you see.

## Legal

WrongGPU is an independent project. It is not affiliated with, endorsed by, or sponsored by NVIDIA or AMD. DLSS is a trademark of NVIDIA Corporation. Radeon and RDNA are trademarks of Advanced Micro Devices, Inc. Provided as-is, without warranty. Binaries only; see `LICENSE` and `THIRD_PARTY.txt`.
