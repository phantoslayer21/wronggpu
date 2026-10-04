# Testing WrongGPU in other games

Thank you for helping. The mod is validated on Cyberpunk 2077 only. What you report decides which games get validated next. Test at your own risk, in single-player games only.

## Before you start

- **No anti-cheat.** Skip any game that uses Easy Anti-Cheat, BattlEye, Vanguard or similar, and skip online games in general. You could be banned.
- The game must use **DirectX 12** and show a **DLSS Super Resolution** option in its graphics menu on NVIDIA cards.
- Back up your save games. The installer backs up every game file it replaces, but saves are yours to protect.
- Close the game before you install.

## Install

1. Unzip the download and run `WrongGPU-Installer.exe`.
2. Pick the game from the list (the installer finds Steam, Epic and GOG games), or choose the game's `.exe` or folder.
3. Any game other than Cyberpunk 2077 is **Experimental**: tick the confirmation box to continue. If the installer refuses the game (for example because of anti-cheat), do not try to get around it.
4. The installer needs an `nvngx_dlss.dll` from DLSS 310.5 or newer. If it cannot find one in your games, click **DLSS DLL...** and choose one.
5. Click Install. It keeps a backup, so **uninstall** (Settings > Apps > Installed apps) returns the folder to how it was.

## Test

1. Start the game and open its graphics settings.
2. Turn on DLSS Super Resolution and pick **Quality**. Set the sharpening slider to 0 if the game has one.
3. Press **Insert** to open the overlay. Check that the model-submission count goes up and the error count stays at 0.
4. Play for at least 10 minutes in a busy scene. Move the camera, drive, run, change the weather if the game has it.
5. Note anything that looks wrong: shimmer or flicker, ghosting trails, black or white screen, wrong colors, UI drawn wrong, stutter, crashes.
6. If you can, measure FPS with DLSS off (native or the game's own upscaler) and with the mod, at the same resolution and settings.

## What to send

Open a [game compatibility report](../../issues/new?template=game-compatibility-report.yml) or post it in https://discord.gg/745z9MfKwm. Please attach:

- The zip from **Collect logs** (click it in the installer, or choose **Modify** for the game in Installed apps). It bundles the mod's `dlss45-*` logs, the install record and the installer log, and tells you what is inside. The installer's own log is `%LOCALAPPDATA%\DLSS45-AMD\setup.log`.
- A screenshot or short clip of any problem.
- Your graphics card, driver version and Windows version.

The logs contain file paths and technical messages. They do not contain passwords or account data, but read them before you post if you want to be sure.

## What happens next

When at least three independent reports show a game working, it moves to **Community: works**. The maintainer then reproduces it on the listed game version and, if it passes the checklist, marks it **Validated**. Games that crash, show bad artifacts, or have no usable DLSS path are listed as **Community: issues** or **Not working**, with your data as the reason.
