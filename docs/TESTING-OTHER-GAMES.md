# Testing WrongGPU in other games

This guide applies to **WrongGPU 0.6.0** and its launcher. See [game compatibility](../COMPATIBILITY.md) for current results. It helps produce a useful report without confusing AMD FSR or passed-through frames with WrongGPU's own results.

## Before you start

- Use a **DirectX 12 game supported by OptiScaler**.
- **Do not test in games with anti-cheat**, and do not bypass a launcher refusal.
- Use an **RX 9070 or RX 9070 XT** and a recent AMD Adrenalin driver (no HIP runtime needed). The RX 9060 XT / RX 9060 is supported but untested; say which card and how much video memory it has when you report.
- Back up your saves. The launcher backs up every game file it replaces and puts them back when you uninstall.

## Install

1. Download and extract [WrongGPU 0.6.0](https://github.com/phantoslayer21/wronggpu/releases/tag/v0.6.0) and run `WrongGPU-0.6.0.exe`.
2. If the game isn't in the Library, use **Add game** and pick the game's folder or exe.
3. Open the game's page and click **Install**. The launcher gets the WrongGPU edition of OptiScaler, finds a usable DLSS file on your PC, from any game or app (310.5 or newer for Model M, a DLSS Ray Reconstruction file 310.7 or newer for Ray Reconstruction) and, for Ray Reconstruction, prepares its GPU code.
4. Turn on the FPS overlay with the game page's **Turn on the FPS overlay** button before you start the game.
5. Start the game and choose **DLSS** as its upscaler, initially in **Quality** mode (and DLSS Ray Reconstruction if the game has it).

## Check the game

1. Let startup finish. The FPS overlay should show **DLSS → WrongGPU Model M**, or **DLSS → WrongGPU RR** and its mode. After you quit, the game page's **Last run** shows what ran.
2. Play for at least 10 minutes. Include camera motion, thin geometry, foliage, moving objects, particles, reflections and scene transitions where available.
3. Check for flicker, ghosting, smearing, wrong colors, UI problems, stutter, crashes or a black screen. Write down the scene and settings that trigger a problem.
4. For a matched visual comparison, switch to AMD FSR in **OptiScaler's menu** (Insert), then return to WrongGPU before recording WrongGPU results.
5. Test resolution changes, quality switches, upscaler switches and fresh game starts separately. Note memory growth, errors and whether WrongGPU resumes.

## Compare performance

Use the same scene, output resolution, input upscaler, quality mode and graphics settings. Keep frame generation, dynamic resolution, V-sync and frame caps the same between runs, and say whether they are on. For a straightforward comparison, turn frame generation and dynamic resolution off.

Let the scene settle before recording. Measure AMD FSR, then WrongGPU, then AMD FSR again. Report average FPS and 1% lows if your tool provides them. If the two AMD runs differ a lot, repeat the comparison before claiming a gain. The time **Last run** shows for Ray Reconstruction is its own time per frame, not the game's frame time.

## Send the report

Use [GitHub's game compatibility form](https://github.com/phantoslayer21/wronggpu/issues/new?template=game-compatibility-report.yml) or the **#bug-reports** channel of the [WrongGPU Discord server](https://discord.gg/745z9MfKwm). Include:

- Game name, store and version/build.
- WrongGPU version, GPU and video memory, AMD driver and Windows version.
- Output resolution, input upscaler, quality mode, graphics settings, and Ray Reconstruction on or off (and its mode).
- What the FPS overlay and **Last run** showed.
- Reproduction steps, screenshots or clips, and matched FPS / 1% low results.
- The ZIP from **Collect logs** on the game's page in the launcher.

Review the ZIP before posting: system information and file paths may include your Windows username. Do not include DLSS files or converted model files.

To remove WrongGPU from the game, use **More → Uninstall WrongGPU** on the game's page; it puts the game's own files back.
