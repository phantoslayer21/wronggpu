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

## Test Model L (WrongGPU 0.6.5 and later)

Model L is a second, larger super-resolution model. It is off by default; you choose it per game. Test Model M first
(above), then Model L in the same game, so the two can be compared.

1. **Prepare Model L's data once.** It is converted on your PC from a DLSS file of version 310.5 or newer that
   contains Model L's data, from any game or app:
   `WrongGPU.exe model convert --preset=L` (add `--model=<path to nvngx_dlss.dll>` to pick the file yourself). If no
   file on your PC has the data, it says so and nothing changes.
2. **Choose Model L for the game**, either way:
   - `WrongGPU.exe set "<the game's .exe>" model=L` while the game is closed (`model=M` goes back). The game then
     starts with Model L.
   - In the game, in **OptiScaler's menu** (Insert), choose **WrongGPU Model L** as the upscaler.
3. **Check that Model L really runs.** The FPS overlay should show **DLSS → WrongGPU Model L**, and the game page's
   **Last run** should say Model L after you quit. If it shows Model M, Model L did not start: say so in the report and
   attach Collect logs (they say why).
4. **Play the same scenes as with Model M**, in the same DLSS mode and output resolution. Look especially for
   ghosting behind moving objects, smearing, flicker on thin lines and foliage, and how quickly detail returns after
   a camera cut. Note anything that looks better or worse than Model M.
5. **Measure Model L against Model M**, as in "Compare performance" below: Model M, then Model L, then Model M again,
   same scene and settings. Model L costs noticeably more GPU time than Model M, so a lower frame rate is expected;
   the release notes give measured times. Report both results.
6. In the report, say which model each result is for, and whether the game's colour was HDR or SDR, if you know
   (the game's HDR setting).

## Compare performance

Use the same scene, output resolution, input upscaler, quality mode and graphics settings. Keep frame generation, dynamic resolution, V-sync and frame caps the same between runs, and say whether they are on. For a straightforward comparison, turn frame generation and dynamic resolution off.

Let the scene settle before recording. Measure AMD FSR, then WrongGPU, then AMD FSR again. Report average FPS and 1% lows if your tool provides them. If the two AMD runs differ a lot, repeat the comparison before claiming a gain. The time **Last run** shows for Ray Reconstruction is its own time per frame, not the game's frame time.

## Send the report

Use [GitHub's game compatibility form](https://github.com/phantoslayer21/wronggpu/issues/new?template=game-compatibility-report.yml) or the **#bug-reports** channel of the [WrongGPU Discord server](https://discord.gg/745z9MfKwm). Include:

- Game name, store and version/build.
- WrongGPU version, GPU and video memory, AMD driver and Windows version.
- Output resolution, input upscaler, quality mode, graphics settings, and Ray Reconstruction on or off (and its mode).
- What the FPS overlay and **Last run** showed, and which model (Model M or Model L) each result is for.
- Reproduction steps, screenshots or clips, and matched FPS / 1% low results.
- The ZIP from **Collect logs** on the game's page in the launcher.

Review the ZIP before posting: system information and file paths may include your Windows username. Do not include DLSS files or converted model files.

To remove WrongGPU from the game, use **More → Uninstall WrongGPU** on the game's page; it puts the game's own files back.
