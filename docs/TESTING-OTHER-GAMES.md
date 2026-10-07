# Testing WrongGPU in other games

WrongGPU 0.3.1 has been tested in Cyberpunk 2077 and Borderlands 4. Other games are experimental. This guide helps produce a useful report without confusing AMD fallback with Model M inference.

## Before you start

- Use a **DirectX 12 game with compatible AMD FSR 3 or newer DLLs**. The installer checks the integration.
- **Do not test in games with anti-cheat**, and do not bypass an installer refusal.
- Use a supported **RX 9070 or RX 9070 XT** and a working AMD HIP runtime.
- Close the game and back up your saves. The installer backs up the game files it replaces.

## Install

1. Download and extract [WrongGPU 0.3.1](https://github.com/phantoslayer21/wronggpu/releases/tag/v0.3.1).
2. Run `WrongGPU-Setup-0.3.1.exe` and select the game's executable or folder.
3. Read the experimental-game prompt. If the installer cannot find a usable model source, click **Model file...** and select an `nvngx_dlss.dll` version 310.5 or newer that you are entitled to use.
4. Click **Install**.
5. Start the game and select **AMD FSR 3 or newer**, initially in **Quality** mode. In Cyberpunk 2077, select **AMD FSR 3** specifically.

## Check the game

1. Press **Insert** to show the full panel. Wait for **starting** to finish and confirm **Model M running**.
2. Play for at least 10 minutes. Include camera motion, thin geometry, foliage, moving objects, particles and scene transitions where available.
3. Check for flicker, ghosting, wrong colors, UI problems, stutter, crashes or a black screen. Record the scene and settings that trigger a problem.
4. Press **Home** to compare with the game's AMD FSR upscaler. Return to WrongGPU before recording its results.
5. Test changes to resolution or quality mode separately, and note whether the model restarts or errors appear.

## Compare performance

Use the same scene, output resolution, FSR mode and graphics settings. Keep frame generation, dynamic resolution, V-sync and frame caps the same between runs, and state whether they are enabled. For a straightforward comparison, disable frame generation and dynamic resolution.

Let the model finish starting and let the scene settle before recording. Measure AMD FSR, then WrongGPU, then AMD FSR again. Report average FPS and 1% lows if your tool provides them, plus the overlay's upscaler time. If the two AMD control runs differ substantially, repeat the comparison before claiming a gain.

Startup fallback and AMD comparison mode are not Model M results. An upscaler's time in milliseconds is not the game's total frame time.

## Send the report

Use [GitHub's game compatibility form](https://github.com/phantoslayer21/wronggpu/issues/new?template=game-compatibility-report.yml) or [Discord](https://discord.gg/745z9MfKwm). Include:

- Game name, store and version/build.
- WrongGPU version, GPU, AMD driver and Windows version.
- Output resolution, FSR mode and graphics settings.
- Whether the overlay confirmed Model M running, and whether it stayed active.
- Reproduction steps, screenshots or clips, and any matched FPS / 1% low results.
- The ZIP from **Collect logs** in the installer, or **Modify** for the game in Windows Installed apps.

Collect logs before uninstalling. If setup failed, its log is `%LOCALAPPDATA%\WrongGPU\setup.log`. Review logs before posting: system information and file paths may include your Windows user name. Do not include your DLSS DLL or converted model files.

Use **Settings → Apps → Installed apps → WrongGPU - your game** to uninstall and restore backed-up game files.
