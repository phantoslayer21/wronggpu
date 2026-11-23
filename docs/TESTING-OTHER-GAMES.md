# Testing WrongGPU in other games

Test **WrongGPU 0.6.5** and its launcher in another DirectX 12 game. This guide helps you produce a useful report and
distinguish WrongGPU's results from AMD FSR and passed-through frames. See
[Game compatibility](../COMPATIBILITY.md) for current results.

## Before you start

- Use a **DirectX 12 game supported by OptiScaler**.
- **Do not test games with anti-cheat.** Do not bypass a launcher refusal.
- Use a **Radeon RX 9070 or RX 9070 XT** with a recent AMD Adrenalin driver. The RX 9060 XT and RX 9060 are supported
  but untested. Report which card you use and how much video memory it has. The default Direct3D 12 path needs no HIP
  runtime.
- Back up your saves. The launcher backs up every game file that it replaces and restores those files when you
  uninstall WrongGPU.

## Install WrongGPU for the game

1. Download and extract [WrongGPU 0.6.5](https://github.com/phantoslayer21/wronggpu/releases/tag/v0.6.5), then run
   `WrongGPU-0.6.5.exe`.
2. If the game is not in the Library, choose **Add game** and select the game's folder or executable.
3. Open the game's page and click **Install**. The launcher downloads the WrongGPU edition of OptiScaler. It also
   finds usable DLSS files on your PC from any game or app. Model M and Model L need a DLSS file of version 310.5 or
   newer. Model K is converted from a DLSS file from one of your games. Ray Reconstruction needs a DLSS Ray Reconstruction file
   of version 310.7 or newer. For Ray Reconstruction, the launcher also prepares its GPU code.
4. Before starting the game, click **Turn on the FPS overlay** on the game's page.
5. Start the game and choose **DLSS** as its upscaler. Start in **Quality** mode. Turn on DLSS Ray Reconstruction if
   the game offers it.

## Check the game

1. Let startup finish. The FPS overlay should show **DLSS → WrongGPU Model M**, or **DLSS → WrongGPU RR** and its
   mode. After you quit, the game page's **Last run** shows what ran.
2. Play for at least 10 minutes. Where the game allows it, include camera motion, thin geometry, foliage, moving
   objects, particles, reflections, and scene transitions.
3. Look for flicker, ghosting, smearing, wrong colors, UI problems, stutter, crashes, or a black screen. Record the
   scene and settings that trigger each problem.
4. For a matched visual comparison, switch to AMD FSR in **OptiScaler's menu** (Insert). Switch back to WrongGPU
   before recording WrongGPU results.
5. Test resolution changes, quality switches, upscaler switches, and fresh game starts separately. Record memory
   growth, errors, and whether WrongGPU resumes.

## Test Model L or Model K

Model L is a larger super-resolution model than Model M. Model K is a different super-resolution network. Both are off
by default and selected per game. Test Model M first, then test Model L or Model K in the same game so you can compare
the results.

### Prepare the model data

For Model L, find a DLSS file of version 310.5 or newer that contains Model L's data. The file can come from any game
or app. From a command prompt in WrongGPU's folder, run:

```bat
WrongGPU.exe model convert --preset=L
```

Add `--model=<path to nvngx_dlss.dll>` to choose the file yourself. If no file on your PC contains Model L's data, the
command says so and changes nothing.

For Model K, use a DLSS file from one of your games and run:

```bat
WrongGPU.exe model convert --preset=K
```

WrongGPU converts the data on your PC. It does not download or ship the model data.

### Choose the model

Close the game. To choose Model L or Model K, run the corresponding command:

```bat
WrongGPU.exe set "<the game's .exe>" model=L
WrongGPU.exe set "<the game's .exe>" model=K
```

The game then starts with that model. Use `model=M` to return to Model M.

You can instead open **OptiScaler's menu** (Insert) in the game and choose **WrongGPU Model L** or **WrongGPU Model
K** as the upscaler.

### Check and compare the model

1. Check the FPS overlay for **DLSS → WrongGPU Model L** or **DLSS → WrongGPU Model K**. After quitting, check that
   **Last run** names the same model. If it shows Model M, the selected model did not start. Say so in the report and
   attach **Collect logs**, which records why.
2. Play the same scenes as with Model M, using the same DLSS mode and output resolution. Look for ghosting behind
   moving objects, smearing, and flicker on thin lines and foliage. Check how quickly detail returns after a camera
   cut. Record anything that looks better or worse than Model M.
3. Follow [Compare performance](#compare-performance), but measure Model M, then Model L or Model K, then Model M
   again. Keep the scene and settings unchanged. The
   [0.6.5 release notes](../clean/installer/RELEASE-NOTES.md#model-l-new) record higher GPU time for Model L than
   Model M in a matched Cyberpunk 2077 test, so a lower frame rate is expected with Model L. Report both results.
4. Identify the model for every result. For Model L, also state whether the game used HDR or SDR color if you know.
   Check the game's HDR setting.

## Compare performance

Use the same scene, output resolution, input upscaler, quality mode, and graphics settings for every run. Keep frame
generation, dynamic resolution, V-sync, and frame caps unchanged, and report whether they are on. For a straightforward
comparison, turn frame generation and dynamic resolution off.

Let the scene settle before recording. Measure AMD FSR, then WrongGPU, then AMD FSR again. Report average FPS and 1%
lows if your tool provides them. If the two AMD FSR runs differ substantially, repeat the comparison before claiming a
gain.

For Model L or Model K, use the Model M, alternative model, Model M sequence described above instead.

The time that **Last run** shows for Ray Reconstruction is its own time per frame, not the game's frame time.

## Send the report

Use [GitHub's game compatibility form](https://github.com/phantoslayer21/wronggpu/issues/new?template=game-compatibility-report.yml)
or the **#bug-reports** channel on the [WrongGPU Discord server](https://discord.gg/745z9MfKwm). Include:

- The game name, store, and version or build.
- The WrongGPU version, GPU and video memory, AMD driver, and Windows version.
- The output resolution, input upscaler, quality mode, and graphics settings.
- Whether Ray Reconstruction was on and its mode.
- What the FPS overlay and **Last run** showed, and which model each result used.
- Reproduction steps, screenshots or clips, and matched average FPS and 1% low results.
- The zip from **Collect logs** on the game's page in the launcher.

Review the zip before posting it. System information and file paths can include your Windows user name. Do not include
DLSS files or converted model files.

To remove WrongGPU from the game, choose **More → Uninstall WrongGPU** on the game's page. The launcher restores the
game's own files.
