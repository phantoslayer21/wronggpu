# Testing WrongGPU in other games

This guide applies to **WrongGPU 0.4.5 through OptiScaler**. See [game compatibility](../COMPATIBILITY.md) for current results. It helps produce a useful report without confusing AMD fallback with Model M inference.

## Before you start

- Use a **DirectX 12 game supported by OptiScaler**.
- **Do not test in games with anti-cheat**, and do not bypass an installer refusal.
- Use an **RX 9070 or RX 9070 XT**, AMD Adrenalin drivers and a working AMD HIP runtime. RX 9060 XT code is included but has not been tested on physical hardware; identify the VRAM capacity when reporting.
- Close the game and back up your saves. The installer backs up the game files it replaces.
- Keep your own [OptiScaler download](https://github.com/optiscaler/OptiScaler/releases) ready.

## Install

1. Download and extract [WrongGPU 0.4.5](https://github.com/phantoslayer21/wronggpu/releases/tag/v0.4.5).
2. Run `WrongGPU-Setup-0.4.5.exe`, supply your OptiScaler archive or folder, and select the game's executable or folder.
3. If the installer cannot find a usable model source, select an `nvngx_dlss.dll` version 310.5 or newer that you are entitled to use.
4. Follow the prompts and install. Older direct installations are migrated to OptiScaler; the game's original FSR DLLs are restored.
5. Start the game and select **DLSS or FSR input**, initially in **Quality** mode. Inputs without depth, such as Hogwarts Legacy's XeSS input, fall back to AMD FSR.
6. Press **Insert** to open OptiScaler and select **`3.1.5 WrongGPU (Model M)`**.

## Check the game

1. Let startup finish. Check the WrongGPU log for successful neural runtime activity and any AMD handoff. OptiScaler's selected menu entry or FPS header alone is insufficient to establish the active provider.
2. Play for at least 10 minutes. Include camera motion, thin geometry, foliage, moving objects, particles and scene transitions where available.
3. Check for flicker, ghosting, wrong colors, UI problems, stutter, crashes or a black screen. Record the scene and settings that trigger a problem.
4. Switch to AMD FSR through **OptiScaler's menu** for a matched visual comparison, then return to WrongGPU before recording Model M results.
5. Test resolution changes, quality switches, upscaler switches and fresh game starts separately. Note memory growth, model restarts, errors and whether Model M resumes.

## Compare performance

Use the same scene, output resolution, input upscaler, quality mode and graphics settings. Keep frame generation, dynamic resolution, V-sync and frame caps the same between runs, and state whether they are enabled. For a straightforward comparison, disable frame generation and dynamic resolution.

Let the model finish starting and the scene settle before recording. Measure AMD FSR, then WrongGPU, then AMD FSR again. Report average FPS and 1% lows if your tool provides them, plus the upscaler time and its measurement source. If the two AMD controls differ substantially, repeat the comparison before claiming a gain.

Startup fallback, AMD comparison mode and plain fallback frames are not Model M results. GPU network time, D3D12/HIP transfer and synchronization time, CPU waits, and total game frame time are different measurements. Don't infer game FPS from an isolated network timing.

## Send the report

Use [GitHub's game compatibility form](https://github.com/phantoslayer21/wronggpu/issues/new?template=game-compatibility-report.yml) or [Discord](https://discord.gg/745z9MfKwm). Include:

- Game name, store and version/build.
- WrongGPU and OptiScaler versions, GPU and VRAM, AMD driver, HIP runtime and Windows version.
- Output resolution, input upscaler, quality mode and graphics settings.
- Evidence that Model M ran, whether it stayed active, and any fallback or warning in the log.
- Reproduction steps, screenshots or clips, and matched FPS / 1% low results.
- The ZIP from **Collect logs** in the installer.

Collect logs before uninstalling. If setup failed, its log is `%LOCALAPPDATA%\WrongGPU\setup.log`. Review logs before posting: system information and file paths may include your Windows username. Do not include your DLSS DLL or converted model files.

Use **Settings → Apps → Installed apps → WrongGPU - your game** to uninstall and restore backed-up game files.
