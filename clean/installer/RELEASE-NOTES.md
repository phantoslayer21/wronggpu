# WrongGPU 0.6.5

Download **WrongGPU-0.6.5.zip** from the release page and extract it, then run **WrongGPU-0.6.5.exe** (see Quick start).

WrongGPU 0.6.5 adds **Model L** and **Model K**, two more super-resolution models you can choose per game, and brings
the fixes planned for 0.6.1 (0.6.1 was not released; everything in it is in 0.6.5). Model M stays the default. If you use 0.6.0, update
your games with the launcher (each game's page offers **Update**).

**Still a work in progress.** WrongGPU is still being worked on, and 0.6.5 is a pre-release. In some games or scenes
you may still see visual bugs, such as artifacts or flicker. If you do, we'd like to hear about it: on the game's page
in the launcher, click **Collect logs**, and send the zip it makes to us in the **#bug-reports** channel of the
**WrongGPU Discord server** or in an issue on GitHub, together with a short description of what you saw (the game, the
scene, your settings) and a screenshot. Every report helps us fix it.

## What's new in 0.6.5 (since 0.6.0)

### Model L (new)
- **A new model, off by default and still under test.** Model L is a larger super-resolution model than Model M:
  it has a full-resolution encoder and decoder and its own reconstruction step. Model M stays the default for every
  game; Model L is a choice you make per game.
- **Converted on your PC.** Like Model M, Model L's data is converted once on your PC from a DLSS model file
  (`nvngx_dlss.dll`) of version 310.5 or newer from any game or app you have installed; WrongGPU never downloads or
  ships it. Convert it once from a command prompt in WrongGPU's folder: `WrongGPU.exe model convert --preset=L`.
- **Choosing it.** For one game: `WrongGPU.exe set "<the game's .exe>" model=L` (`model=M` goes back). In OptiScaler's
  menu it is the upscaler **WrongGPU Model L**, next to **WrongGPU Model M**.
- **Runs on Direct3D 12** inside the game's own command list, like Model M: no AMD HIP runtime needed.
- **GPU time.** In Cyberpunk 2077 at 1080p Quality (1280x720 upscaled to 1920x1080) on a Radeon RX 9070 XT, standing
  still at the same spot, measured in the same sitting: the model's GPU time per frame (all of its kernels) is
  2.67 ms with Model L and 1.17 ms with Model M
  ([clean/l/records/cp2077-model-l-timing-2026-11-16.json](https://github.com/phantoslayer21/wronggpu/blob/main/clean/l/records/cp2077-model-l-timing-2026-11-16.json)
  has the record).

### Model K (new)
- **Another model, off by default and still under test.** Model K is a different super-resolution network from
  Model M, with its own implementation in WrongGPU. Model M stays the default for every game; Model K is a choice you
  make per game.
- **Converted on your PC.** Like the other models, Model K's data is converted once on your PC from a DLSS model file
  (`nvngx_dlss.dll`) of one of your games; WrongGPU never downloads or ships it. From a command prompt in WrongGPU's
  folder: `WrongGPU.exe model convert --preset=K`.
- **Choosing it.** For one game: `WrongGPU.exe set "<the game's .exe>" model=K` (`model=M` goes back). In OptiScaler's
  menu it is the upscaler **WrongGPU Model K**.
- **Runs on Direct3D 12** inside the game's own command list, like Model M: no AMD HIP runtime needed. Tested in
  Cyberpunk 2077 at 1080p Quality.
- **GPU time.** In Cyberpunk 2077 at 1080p Quality (1280x720 upscaled to 1920x1080) on a Radeon RX 9070 XT, standing
  still at the same spot, measured one right after the other: the model's GPU time per frame (all of its kernels) is
  1.03 ms with Model K and 1.17 ms with Model M
  ([clean/k/records/cp2077-model-k-timing-2026-11-15.json](https://github.com/phantoslayer21/wronggpu/blob/main/clean/k/records/cp2077-model-k-timing-2026-11-15.json)
  has the record).
- **Optional faster approximation (`k_approx`), off by default.** With nothing set, Model K runs its exact path. Adding
  `k_approx=a1,a2` to the `runtime_options` line of the game's `wronggpu.ini` selects close approximations of two
  stages: the input stage's logarithm, exponent and divisions (`a1`) and the output stage's divisions (`a2`). Model K,
  1440p Quality, WrongGPU's test harness (direct route) on a Radeon RX 9070 XT: the frame takes 1.37 ms with
  `k_approx=a1,a2` and 1.58 ms on the exact path of the same build. The output differs slightly from the exact path;
  compared with the exact path and with reference images it was rated very, very close on both test sets (a Cyberpunk
  2077 sequence and a thin-line scene), with no drift over time. Frames with a render size above 0.68x the output (for
  example DLAA) or a jitter outside +-0.5 keep the exact input stage even with `a1` (one log line). It runs only on
  WrongGPU's native Direct3D 12 route (the in-game route); where it can't be used, Model K is refused at start and the
  game keeps its own upscaler.
- If you used Preset K with an older WrongGPU, the launcher no longer asks to switch that game to Model M when it
  updates it: the game stays on Model K.

### Faster kernels, same output
- **Model L:** its Swin stage takes 7.0 µs less per frame and its decoder stage 2.5 µs less at 1440p Quality (1707x960
  upscaled to 2560x1440); at 1080p Quality (1280x720 upscaled to 1920x1080) they take 2.5 µs and 1.0 µs less. The
  output does not change.
- **Model M:** its final (post) stage takes 2.5 µs less per frame at 1440p Quality and 1.5 µs less at 1080p Quality.
  The output does not change.
- Measured on a Radeon RX 9070 XT in our offline benchmark, which runs frames back to back (five rounds, the median
  difference per stage); in a game the saving per frame can differ.

### Fixed
- **Ray Reconstruction: video memory no longer grows while Ray Reconstruction is available but switched off.** In
  0.6.0, a game that sends its Ray Reconstruction inputs every frame (Cyberpunk 2077 does, from its main menu on) made
  the plugin keep a copy of each frame's inputs that was never freed while no Ray Reconstruction was running, until the
  PC ran out of video memory. The copies are now freed as soon as the GPU is done with them (#8).
- **Ray Reconstruction plugin: no more crash with an older Microsoft C++ runtime.** `WrongGPU-RR.asi` was the only part
  of WrongGPU built against the shared C++ runtime, and with an older `msvcp140.dll` (an old Visual C++ redistributable,
  or a Proton prefix) the game aborted at start or froze shortly after. The plugin now carries its own runtime, like the
  rest of WrongGPU, and needs no `msvcp140.dll` (#9).
- **Model M under Proton:** one of WrongGPU's small helper shaders used a keyword that Wine's shader compiler rejects,
  so Model M could not start under Proton. The keyword is gone (it was not needed). Model M under Proton also needs a fix
  in vkd3d-proton's shader translator; see Discussion #10 for the state of that (#7). We have not tested Model M under
  Proton ourselves.
- **Model M's driver check at start is stricter:** it now tells apart drivers or translation layers that lay out the
  matrix data differently, instead of passing them (#14).
- **Collect logs from a game's uninstaller** (Apps & features, **Modify**) no longer reports valid model data as
  invalid: it says the data is not verified there and points to Collect logs in the launcher (#24).
- **Uninstall, Update, Repair and the move out of a wrong folder never discard your files.** When a file can't go back
  as it was, because you or another program changed it after the install, it stays where it is and WrongGPU's copy of
  what was there before is kept in a `WrongGPU-originals` folder. The result says "partly restored" and where the copy
  is. A `wronggpu.ini` you edited by hand stays after an uninstall (it used to be deleted while the message said it was
  left in place) (#70, #73).
- **The launcher works next to your own OptiScaler setup:** it recognises another OptiScaler install and its plugins,
  asks before it moves anything aside, and moves only the files you agreed to, checked by their contents (#23).
- **Ray Reconstruction: GPU memory is released safely when a game ends Ray Reconstruction.** The plugin now keeps an
  instance's memory until the GPU has finished its last frame, instead of freeing it at once (#63).
- **Ray Reconstruction stays off when another Ray Reconstruction plugin is already active.** It switches itself off
  only when the other plugin is positively identified, and says so in `wronggpu-rr.log`. A different plugin it cannot
  identify keeps WrongGPU's on; `rr_other_plugin = yield` in the plugin's settings keeps WrongGPU's off (#22).
- **Ray Reconstruction under Proton:** the thread that prepares Ray Reconstruction's GPU code now has a large enough
  stack for vkd3d-proton's shader translator, and a pipeline that fails to build is logged (#149).

### Also changed
- The launcher reads the developer-only HIP runtime setting only in test builds, so a setting left on a PC can't
  change what a release installs (#25).

### Thanks
Thanks to **j0nsh1n** for the reports and the analysis behind #7, #8 and #9, for the Linux write-up in
Discussion #10, and for the Ray Reconstruction fix under Proton (#149, #196).

## Quick start

1. **Install.** Run **WrongGPU-0.6.5.exe** from the zip. Its first-run check looks at your PC, then offers **Install
   for this user** (the default; afterwards it is **WrongGPU** in the Start menu) or **Run portable**, and then shows
   your games. Open a game and click **Install** (the launcher gets OptiScaler for it the first time). The game's page
   then shows a checklist, **Turn it on in the game**, with the next two steps. **Borderlands 4** needs one Engine.ini
   setting for Ray Reconstruction: the launcher asks about it right after the install. Already on 0.6.0? Start the new
   launcher; your games show **Update**.
2. **Choose the upscaler in the game.** Start the game and choose **DLSS** as the upscaler in its graphics settings (FSR
   or XeSS also work). With TAA or upscaling off, the game never asks for an upscaler and WrongGPU stays idle. For a game
   WrongGPU runs in without OptiScaler, the game's page tells you which upscaler to pick.
3. **See it running.** OptiScaler's FPS overlay shows which upscaler runs. To turn it on, use the checklist's **Turn
   on the FPS overlay** button, or the game's page: **OptiScaler → FPS overlay → On → Apply**. Settings take effect the
   next time the game starts, so turn it on before you start the game (or quit, Apply, and start it again).
   - With the WrongGPU edition of OptiScaler (the launcher downloads it), the overlay shows **DLSS → WrongGPU Model M**
     while Model M upscales, or **DLSS → WrongGPU RR** and its mode while Ray Reconstruction runs. With another
     OptiScaler it shows AMD FSR there, so check **Last run** instead.
   - In a game WrongGPU runs in without OptiScaler there is no FPS overlay. WrongGPU's own overlay in the game's corner
     says **Model M** while it runs.
   - After you quit, the game's page shows the run under **Last run**: Model M, or Ray Reconstruction where it did the
     upscaling.
4. **Undo.** On the game's page, **More → Uninstall WrongGPU** puts the game's own files back exactly as they were.

## What came in 0.6.0 (since 0.4.5)

For players coming from 0.4.5, in short (the 0.6.0 release page has the full list):
- **The WrongGPU launcher replaces the setup wizard:** it finds your Steam, Epic and GOG games, shows what each game can
  do, installs, updates, repairs, rolls back and uninstalls with one click per game, has a settings page per game, shows
  **Last run** and makes a **Collect logs** zip for bug reports. It works with the keyboard and with screen readers.
- **No more hunting for OptiScaler:** the launcher downloads the **WrongGPU edition of OptiScaler** itself (GPL, its own
  release page: https://github.com/phantoslayer21/OptiScaler/releases/tag/wronggpu-edition-20261102.1), checks it
  against the size and SHA-256 built into the launcher, and keeps it for every game. Nothing of OptiScaler is inside
  WrongGPU's download. In OptiScaler's menu, WrongGPU is an upscaler of its own.
- **No AMD HIP runtime needed any more:** on the Radeon RX 9070 / RX 9060 series, Model M runs on **Direct3D 12 inside
  the game's own command list**, in every mode up to native resolution (DLAA). HIP is only used for the optional sparkle
  guard (and for frame capture in bug reports). Frames the network can't use go to AMD's FSR.
- **Ray Reconstruction:** in games that use DLSS Ray Reconstruction, WrongGPU denoises and upscales the ray-traced image
  with its own network. The game page has a Ray Reconstruction section: **Use WrongGPU** (on by default) and
  **Quality**: **Fast** (default, recommended), **Accurate** (higher quality, slower) and **Exact** (the most precise and
  the slowest; Settings > Model and runtime > Exact mode). Exact follows the network's math as closely as we can, but it
  can't match DLSS on an NVIDIA card pixel for pixel: different graphics cards round their arithmetic slightly
  differently. You can also switch the mode in OptiScaler's menu. The Ray Reconstruction data is converted once on your
  PC from your own DLSS Ray Reconstruction file; WrongGPU never downloads or ships it.
- **Prepared by the launcher:** the graphics driver compiles Ray Reconstruction's GPU code once per game; the launcher
  does that after it installs, updates or repairs a game ("Preparing Ray Reconstruction", Esc cancels), and again in the
  background after an AMD driver update or a WrongGPU update.
- **Borderlands 4: "Fix it for me"** for the one engine setting its Ray Reconstruction needs (see Tested games).
- **Works next to other tools:** ReShade, Special K, ASI loaders, your own OptiScaler and similar mods are found and
  kept, or moved aside with a backup only when you agree. A game with another upscaler mod gets a clear message, and
  WrongGPU leaves that game alone.

## Known issues
- A rare crash at game start can happen; restarting the game fixes it.
- RESIDENT EVIL requiem needs your own **REFramework**, installed by you first: the game blocks OptiScaler without it.
  WrongGPU never installs, downloads or replaces REFramework; the launcher tells you when it is missing.
- RESIDENT EVIL requiem: very occasionally a single frame comes out far too bright or too dark at a scene change (after
  loading a save, or during flashes of lightning). It lasts one frame; we're investigating it (#17).
- Hogwarts Legacy: since the game's November 2026 update, Ray Reconstruction isn't available with WrongGPU yet. Model M
  works. We're investigating it (#16).
- Model L: tested for super resolution in Cyberpunk 2077 and RESIDENT EVIL requiem, both of which hand the upscaler
  HDR colour. Model L's path for games that hand the upscaler SDR colour is tested offline only: every game we tried
  sends HDR colour. Ray Reconstruction always uses its own network; Model L is a super-resolution choice only.
- Radeon RX 9060 XT / RX 9060: supported, not yet tested; 8 GB cards are untested.
- RESIDENT EVIL requiem prepares Ray Reconstruction at its own first start (OptiScaler's overlay says "WrongGPU RR
  (preparing...)" meanwhile), not in the launcher.
- One Ray Reconstruction plugin serves games on game interface 2.7, 2.8 and 2.14 (it picks the route when the game
  starts). Games on other game interface versions (for example 2.6) can't use Ray Reconstruction through the game
  interface, but it still works in those that create it directly.
- Ray Reconstruction needs a DLSS Ray Reconstruction file of version 310.7 or newer somewhere on your PC (for example
  from another game). Without one, the game keeps its own denoiser and the launcher says so.
- 007 First Light: not supported (crashes at start under OptiScaler).
- Marvel's Spider-Man 2: Model M only. Ray Reconstruction there needs OptiScaler's GPU-name spoof, which makes that game
  hang (with or without WrongGPU).

## Tested games
Tested on a Radeon RX 9070 XT. "Ray Reconstruction" means WrongGPU's Ray Reconstruction ran in the game; Model M means
super resolution only. [COMPATIBILITY.md](https://github.com/phantoslayer21/wronggpu/blob/main/COMPATIBILITY.md) says
which WrongGPU version each result comes from.
- **Cyberpunk 2077**: Ray Reconstruction, Model M, Model L and Model K.
- **RESIDENT EVIL requiem** (experimental): Model M and Model L, and Ray Reconstruction when the game's **Ray
  Tracing** setting is **Path Tracing** and its upscaler is **DLSS**. Needs your own REFramework (see Known issues).
- **Borderlands 4**: Ray Reconstruction (the game creates it itself; this needs the WrongGPU edition of OptiScaler).
  Ray Reconstruction needs one setting, and Borderlands 4 removes it on every start. After you install WrongGPU into it,
  the launcher asks. **Fix it for me** adds the setting and makes the file read-only, so the game keeps it
  (uninstalling WrongGPU undoes both). **Show me how** opens the game's page with the steps below, an **Open folder**
  button, the line with **Copy**, and **Check again**, which tells you whether the file has it. **Not now** asks again
  next time. The game's page changes the choice at any time. To do it yourself:
  1. open the game's settings folder `Documents\My Games\Borderlands 4\Saved\Config\Windows` (the folder that holds
     `GameUserSettings.ini`; if your Documents folder is in OneDrive, it is there);
  2. create or open the file `Engine.ini` there with Notepad;
  3. add the line `[SystemSettings]` at the end if the file doesn't have it, and under it the line from the launcher's
     **Copy** button on the game's page (or the release page);
  4. save the file, then right-click it, choose **Properties**, tick **Read-only** and click **OK**. Without that, the
     game removes the file at its next start.

  From a command prompt: `WrongGPU.exe set "<the game's folder>\Borderlands4.exe" rr_config_fix=on` (`off` takes
  the line out again). The exact line is on the release page and behind the game page's **Copy** button.
- **Hogwarts Legacy**: Model M. Ray Reconstruction worked up to the game's November 2026 update; on the updated game it
  isn't available yet, so the launcher doesn't offer it there (#16).
- **Marvel's Spider-Man 2**: Model M (see Known issues; Ray Reconstruction is experimental there, see below).
- **LEGO Batman**: Model M. The game's menu has no Ray Reconstruction option (it would need a separate game mod), so
  WrongGPU's Ray Reconstruction isn't offered there.

## Experimental: Ray Reconstruction in Spider-Man 2
WrongGPU's Ray Reconstruction can be turned on in Marvel's Spider-Man 2, but the game is currently unstable with it: it
often freezes the GPU or crashes (at start, while loading a save, or while it prepares ray tracing), even with
ray-traced reflections off. We don't recommend it. To try it anyway:
- turn on the experimental setting from a command prompt with `WrongGPU.exe set "<the game's folder>\Spider-Man2.exe"
  sm2_rr_experimental=on` (the launcher then sets most of the OptiScaler settings this needs;
  `sm2_rr_experimental=off` puts them back, and an uninstall removes them either way);
- if the game's launch options in Steam have the option given on the release page, remove it (with it the game turns
  its game interface off and never offers Ray Reconstruction);
- in OptiScaler's menu, turn on its spoofing for the game interface and save;
- in the game's graphics settings, turn ray-traced shadows on and ray-traced reflections off;
- choose DLSS with Ray Reconstruction in the game's menu.
The exact launch option is on the release page.

## Requirements
- Windows 10 or 11, a Radeon RX 9000-series graphics card (RX 9070 series; RX 9060 XT and RX 9060 supported, not yet
  tested), a recent AMD driver.
- **No AMD HIP runtime needed:** Model M, Model L and Model K run on Direct3D 12. `executor=hip` and frame capture are
  debugging options that use HIP; both are off by default.
- An internet connection the first time, for the WrongGPU edition of OptiScaler, which the launcher downloads itself
  (offline you can choose an OptiScaler download yourself, in Settings).
- For Model M and Model L, a DLSS file of version 310.5 or newer, and for Ray Reconstruction, a DLSS Ray
  Reconstruction file of version 310.7 or newer, each from any game or app on your PC (not necessarily the game you play); WrongGPU
  converts them on your PC and never ships them.

## Downloading and starting WrongGPU safely
WrongGPU's download is not signed with a code-signing certificate (they cost money every year and this is a free
project). That has three visible effects:
- **"Windows protected your PC"** (Microsoft Defender SmartScreen) can show when you start WrongGPU-0.6.5.exe the first
  time. Click **More info**, check that the file name is WrongGPU-0.6.5.exe, then click **Run anyway**. SmartScreen
  may show this warning for an unsigned program.
- **Check that your download is the real one** before you run it. The release page lists the SHA-256 of the zip, and
  SHA256SUMS.txt in the zip lists the SHA-256 of every file in it. In PowerShell (in the folder with the file):
  `Get-FileHash .\WrongGPU-0.6.5.zip -Algorithm SHA256` (or `certutil -hashfile WrongGPU-0.6.5.zip SHA256` in a
  Command Prompt). The long number it prints must be exactly the one on the release page. For the launcher itself, use
  `Get-FileHash .\WrongGPU-0.6.5.exe -Algorithm SHA256` and compare it with its line in SHA256SUMS.txt. If a number
  differs, delete the file and download it again from the release page only.
- **If Microsoft Defender flags or removes it**: first check the SHA-256 as above. If the numbers match, the file
  matches the published checksum, but that alone does not establish whether a detection is a false positive. You can
  restore it in Windows Security > Virus & threat protection > Protection history (choose the item, Actions, Restore),
  and please report the detection to us with the name Defender showed and to Microsoft at
  https://www.microsoft.com/en-us/wdsi/filesubmission so it gets cleared for everyone. If the numbers do not match, do
  not restore it: delete it and download it again.

DLSS is a trademark of NVIDIA Corporation; AMD, Radeon, FidelityFX and FSR are trademarks of Advanced Micro Devices, Inc. WrongGPU is an independent project and is not affiliated with or endorsed by either company.
