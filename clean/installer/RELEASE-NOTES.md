# WrongGPU 0.6.1

Download **WrongGPU-0.6.1.zip** from the release page and extract it, then run **WrongGPU-0.6.1.exe** (see Quick start).

WrongGPU 0.6.1 is a bug-fix release for 0.6.0. It fixes a video-memory leak and a crash of the Ray Reconstruction
plugin, a Model M start failure under Proton, and several launcher problems with other mods. If you use 0.6.0, update
your games with the launcher (each game's page offers **Update**).

**Still a work in progress.** WrongGPU is still being worked on, and 0.6.1 is a pre-release. In some games or scenes
you may still see visual bugs, such as artifacts or flicker. If you do, we'd like to hear about it: on the game's page
in the launcher, click **Collect logs**, and send the zip it makes to us in the **#bug-reports** channel of the
**WrongGPU Discord server** or in an issue on GitHub, together with a short description of what you saw (the game, the
scene, your settings) and a screenshot. Every report helps us fix it.

## What's new in 0.6.1 (since 0.6.0)

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
  in vkd3d-proton's shader translator; see Discussion #10 for the state of that (#7).
- **Model M's driver check at start is stricter:** it now tells apart drivers or translation layers that lay out the
  matrix data differently, instead of passing them (#14).
- **Collect logs from a game's uninstaller** (Apps & features, **Modify**) no longer reports valid model data as
  invalid: it says the data is not verified there and points to Collect logs in the launcher (#24).

### Works better next to other mods
- **Other OptiScaler setups:** when a game already has an OptiScaler setup from another source, the launcher no longer
  takes it over or removes its files. Repair and Update ask before they move another setup's files aside, and an
  uninstall removes a changed file only when it is one of WrongGPU's own releases (#23).
- **Two Ray Reconstruction plugins:** if a game already has another Ray Reconstruction plugin active, WrongGPU's plugin
  stays off and writes one line saying why to `wronggpu-rr.log` (the game page's **Last run** shows it). Model M is
  not affected. To use WrongGPU's Ray Reconstruction there anyway, set `rr_other_plugin = take` in `wronggpu-rr.ini`
  next to the game's exe (#22).
- **Ignore list for the converter:** another program's installer can list the DLSS files it placed in a game, and the
  launcher's converter then never uses them; WrongGPU converts only the files that came with your own games. The list
  is `%LOCALAPPDATA%\WrongGPU\ignore-files.txt`; its format is in the launcher's README on GitHub. An empty list, or
  none, changes nothing, and the uninstalls keep it (#21).

### Also changed
- The launcher reads the developer-only HIP runtime setting only in test builds, so a setting left on a PC can't
  change what a release installs (#25).

### Thanks
Thanks to **j0nsh1n** for the reports and the analysis behind #7, #8 and #9, and for the Linux write-up in
Discussion #10.

## Quick start

1. **Install.** Run **WrongGPU-0.6.1.exe** from the zip. Its first-run check looks at your PC, then offers **Install
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
- RESIDENT EVIL requiem needs your own **REFramework**, installed by you first: the game blocks OptiScaler without it.
  WrongGPU never installs, downloads or replaces REFramework; the launcher tells you when it is missing.
- RESIDENT EVIL requiem: very occasionally a single frame comes out far too bright or too dark at a scene change (after
  loading a save, or during flashes of lightning). It lasts one frame; we're investigating it (#17).
- Hogwarts Legacy: since the game's November 2026 update, Ray Reconstruction isn't available with WrongGPU yet. Model M
  works. We're investigating it (#16).
- Preset K is not part of 0.6.1; it returns in a later version (#15). If a game still starts on Preset K from an older
  WrongGPU, the launcher asks before it updates the game: **Switch to Model M** updates it and moves it to Model M (its
  other settings stay); **Not now** leaves the game as it is.
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
super resolution only. 0.6.1 was tested in Cyberpunk 2077; the other results are from 0.6.0.
- **Cyberpunk 2077**: Ray Reconstruction and Model M.
- **RESIDENT EVIL requiem** (experimental): Ray Reconstruction when the game's **Ray Tracing** setting is **Path
  Tracing** and its upscaler is **DLSS**. Needs your own REFramework (see Known issues).
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
- Windows 10 or 11, a Radeon RX 9000-series graphics card (RX 9070 series; RX 9060 XT supported, not yet tested), a
  recent AMD driver.
- **No AMD HIP runtime needed:** Model M runs on Direct3D 12. HIP is only used for the optional sparkle guard.
- An internet connection the first time, for the WrongGPU edition of OptiScaler, which the launcher downloads itself
  (offline you can choose an OptiScaler download yourself, in Settings).
- For Model M, a DLSS file of version 310.5 or newer, and for Ray Reconstruction, a DLSS Ray Reconstruction file
  of version 310.7 or newer, each from any game or app on your PC (not necessarily the game you play); WrongGPU
  converts them on your PC and never ships them.

## Downloading and starting WrongGPU safely
WrongGPU's download is not signed with a code-signing certificate (they cost money every year and this is a free
project). That has three visible effects:
- **"Windows protected your PC"** (Microsoft Defender SmartScreen) can show when you start WrongGPU-0.6.1.exe the first
  time. Click **More info**, check that the file name is WrongGPU-0.6.1.exe, then click **Run anyway**. SmartScreen
  may show this warning for an unsigned program.
- **Check that your download is the real one** before you run it. The release page lists the SHA-256 of the zip, and
  SHA256SUMS.txt in the zip lists the SHA-256 of every file in it. In PowerShell (in the folder with the file):
  `Get-FileHash .\WrongGPU-0.6.1.zip -Algorithm SHA256` (or `certutil -hashfile WrongGPU-0.6.1.zip SHA256` in a
  Command Prompt). The long number it prints must be exactly the one on the release page. For the launcher itself, use
  `Get-FileHash .\WrongGPU-0.6.1.exe -Algorithm SHA256` and compare it with its line in SHA256SUMS.txt. If a number
  differs, delete the file and download it again from the release page only.
- **If Microsoft Defender flags or removes it**: first check the SHA-256 as above. If the numbers match, the file
  matches the published checksum, but that alone does not establish whether a detection is a false positive. You can
  restore it in Windows Security > Virus & threat protection > Protection history (choose the item, Actions, Restore),
  and please report the detection to us with the name Defender showed and to Microsoft at
  https://www.microsoft.com/en-us/wdsi/filesubmission so it gets cleared for everyone. If the numbers do not match, do
  not restore it: delete it and download it again.

DLSS is a trademark of NVIDIA Corporation; AMD, Radeon, FidelityFX and FSR are trademarks of Advanced Micro Devices, Inc. WrongGPU is an independent project and is not affiliated with or endorsed by either company.
