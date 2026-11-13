# WrongGPU 0.6.0

Download **WrongGPU-0.6.0.zip** from the release page and extract it, then run **WrongGPU-0.6.0.exe** (see Quick start).

WrongGPU 0.6.0 is the first public release since 0.4.5, and it changes how you use WrongGPU: a **launcher** replaces
the setup wizard, the launcher gets **OptiScaler** for you, Model M **no longer needs an AMD HIP runtime**, and
WrongGPU now does **Ray Reconstruction** too. It includes everything from the 0.5.x early-access builds.

**Still a work in progress.** WrongGPU is still being worked on, and 0.6.0 is a pre-release. In some games or scenes
you may still see visual bugs, such as artifacts or flicker. If you do, we'd like to hear about it: on the game's page
in the launcher, click **Collect logs**, and send the zip it makes to us in the **#bug-reports** channel of the
**WrongGPU Discord server**, together with a short description of what you saw (the game, the scene, your settings)
and a screenshot. Every report helps us fix it.

## Quick start

1. **Install.** Run **WrongGPU-0.6.0.exe** from the zip. Its first-run check looks at your PC, then offers **Install
   for this user** (the default; afterwards it is **WrongGPU** in the Start menu) or **Run portable**, and then shows
   your games. Open a game and click **Install**
   (the launcher gets OptiScaler for it the first time). The game's page then shows a checklist, **Turn it on in the
   game**, with the next two steps. **Borderlands 4** needs one Engine.ini setting for Ray Reconstruction (so did
   Hogwarts Legacy before its November 2026 update): the launcher asks about it right after the install.
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

## What's new in 0.6.0 (since 0.4.5)

### The WrongGPU launcher replaces the setup wizard
In 0.4.5 you closed the game, ran WrongGPU-Setup and picked your game. Now there is one window, the WrongGPU launcher, for
all your games:
- **Finds your games:** it reads your Steam, Epic and GOG libraries and shows every game it finds as a tile in its
  Library, with the art from your own Steam cache (nothing is downloaded or shipped for it). **Add game** takes a game
  from anywhere else, **Rescan** looks again.
- **Shows what each game can do:** can install, installed, update or repair, your own OptiScaler, another mod, an older
  install, refused (for example games with anti-cheat, which are never supported), or running.
- **One click per game:** Install, Update, Repair, Roll back (to the version before the last update) and Uninstall,
  and an optional **Play** button.
- **A settings page per game** with an Apply bar: WrongGPU on or off, what it runs on (Auto, Direct3D 12 or HIP), the
  output filter, pre-exposure, sparkle guard, OptiScaler's FPS overlay and OptiScaler's own options, and for Ray
  Reconstruction games its on/off switch and quality (see below).
- **Last run:** what the game used the last time it ran (Model M, Direct3D 12 or HIP, Ray Reconstruction and its
  cost), and **Collect logs** in one click for a bug report.
- **A first-run check** of Windows, your graphics card, the Direct3D 12 path and the model data, then your choice:
  **Install for this user** (a Start menu entry and one entry in Apps & features; the launcher uninstalls itself from
  there) or **Run portable**.
- **Picks up your earlier installs:** games you set up with the 0.5.x setup are taken over, keep their settings, and
  are updated by the launcher (we tested this). Games set up with 0.4.x use the same install record and should be
  taken over the same way, but we haven't tested that yet.
- **Works next to other tools:** ReShade, Special K, ASI loaders, your own OptiScaler and similar mods are found and
  kept, or moved aside with a backup only when you agree. A game with another upscaler mod gets a clear message, and
  WrongGPU leaves that game alone. Older WrongGPU files without an install record (for example from a 0.4.x direct
  install in an Unreal Engine game's plugin folder) are found and can be cleaned up with your OK.
- **Borderlands 4 and Hogwarts Legacy: "Fix it for me".** Their Ray Reconstruction needs one engine setting, and
  Borderlands 4 removes it on every start. After you install WrongGPU into either game, the launcher asks: **Fix it
  for me** adds the setting (and makes Borderlands 4's file read-only so the game keeps it), **Show me how** opens the
  game's page with the steps, the line to copy and a check that the file has it, **Not now** asks again next time. The
  game's page (**Engine.ini setting**: WrongGPU or Myself) changes it at any time. Uninstalling WrongGPU undoes the fix.
  The game's own Ray Reconstruction option may look greyed out or off; that's expected, the setting turns it on.
  OptiScaler's overlay shows DLSS → WrongGPU RR and Last run shows Ray Reconstruction when it's working.
  On Hogwarts Legacy since the game's November 2026 update, WrongGPU's Ray Reconstruction isn't available yet (see
  Tested games), so the launcher doesn't ask there.
  Chosen before the game's first start, the setting is added once the game has made its settings folder: start the game
  once, then open the launcher (the game's page says "Will be added after you start the game once" until then). A game
  installed in a protected folder (such as Program Files) never gets a permission prompt you didn't click for: its page
  says "Waiting for the game's first start", then "Ready to add" (the Library lists the game under **Needs attention**
  then, and the launcher says so once when it opens), and **Fix it now** adds the setting after Windows asks.
- The whole launcher works with the keyboard and with screen readers: every control has a name for them. (An update
  check is coming in a later version.)

### No more hunting for OptiScaler
In 0.4.5 you had to download OptiScaler yourself and point the setup at it. Now the launcher downloads the **WrongGPU
edition of OptiScaler** itself the first time it needs it (OptiScaler with the plugin interface WrongGPU's Ray
Reconstruction uses; GPL, its own release page:
https://github.com/phantoslayer21/OptiScaler/releases/tag/wronggpu-edition-20261102.1), checks it against the size and
SHA-256 built into the launcher, refuses a download that doesn't match, and keeps it for every game. Nothing of
OptiScaler is inside WrongGPU's download. Offline, you can choose an OptiScaler download yourself (Settings); games
whose Ray Reconstruction needs the WrongGPU edition then get Model M only until the download works.
- In OptiScaler's menu, WrongGPU is an upscaler of its own (**WrongGPU**, version **WrongGPU Model M**), and the
  launcher selects it for you. OptiScaler's overlay shows "DLSS → WrongGPU Model M", or "DLSS → WrongGPU RR" and the
  mode while Ray Reconstruction runs. AMD's FSR stays available under OptiScaler's own FSR entries.

### No AMD HIP runtime needed any more
In 0.4.5 Model M ran on HIP and needed a working AMD HIP runtime. Now, on the Radeon RX 9070 / RX 9060 series, Model M
runs on **Direct3D 12 inside the game's own command list**: it reads the game's textures and writes its output
directly, with no HIP. That now covers every mode up to **native resolution (DLAA)**, including Ultra Quality and every
ratio between Quality and native. Measured on an RX 9070 XT with the 0.6.0 runtime (GPU time of WrongGPU's work per
frame, as its timing log reports it), on a synthetic benchmark:

| Mode | 1080p output | 1440p output |
| --- | --- | --- |
| Quality | about 1.3 ms | about 2.2 ms |
| Ultra Quality | about 1.6 ms | about 2.8 ms |
| Native resolution (DLAA) | about 2.4 ms | about 4.3 ms |


HIP is only used for the optional sparkle guard (and for frame capture in bug reports). Frames the network can't use
(for example a render size larger than the output) go to AMD's FSR, which stays available next to WrongGPU.

### Ray Reconstruction (new)
- In games that use DLSS Ray Reconstruction (path tracing or ray tracing), WrongGPU now denoises and upscales the
  ray-traced image with its own network, on Radeon RX 9000 cards. Tested: Cyberpunk 2077, RESIDENT EVIL requiem
  (experimental), Borderlands 4 (with the one setting above), Hogwarts Legacy (up to the game's November 2026 update;
  see Tested games) and Marvel's Spider-Man 2 (experimental, see below).
- **How to turn it on:** install WrongGPU into the game with the launcher, then choose DLSS Ray Reconstruction in the
  game's graphics settings. The Ray Reconstruction data is converted once on your PC from your own DLSS Ray
  Reconstruction file (version 310.7 or newer, from any game); WrongGPU never downloads or ships it.
- The game page has a Ray Reconstruction section: **Use WrongGPU** (on by default; off gives the game back its own
  denoiser) and **Quality**:
  - **Fast** (default, recommended): the fastest. Fine detail can differ slightly from Accurate in motion.
  - **Accurate**: higher quality, slower. To switch to it, choose Accurate under Quality on the game's page, or in
    OptiScaler's menu.
  - **Exact**: the most precise, and the slowest. Enable Settings > Model and runtime > Exact mode to select it.
    Exact follows the network's math as closely as we can. It still can't match DLSS on an NVIDIA card pixel for
    pixel: different graphics cards round their arithmetic slightly differently, so very small differences always
    remain.
- You can also switch the mode in the game, in OptiScaler's menu.
- Games that create Ray Reconstruction themselves (for example Unreal Engine games with their own DLSS plugin) get it
  through the WrongGPU edition of OptiScaler; with another OptiScaler the launcher installs WrongGPU without Ray
  Reconstruction for such a game and says why.
- **Prepared by the launcher:** the graphics driver has to compile Ray Reconstruction's GPU code once per game. The launcher does that for you after it installs, updates or
  repairs a game ("Preparing Ray Reconstruction", Esc cancels; the game page shows Prepared / Prepare), so the game's
  first start has Ray Reconstruction ready at once.
- **After an AMD driver update** (the new driver starts with an empty shader cache) or a WrongGPU update, the launcher
  prepares those games again in the background when you open it, the default quality first. The status line at the
  top says "Preparing Ray Reconstruction after the driver update (n of m)", and the game page says the game is not
  ready until it is done. Pressing Play or Prepare on a game does that game first. If the driver drops its cache on its
  own, the game prepares it at its first start.

### New games
- **RESIDENT EVIL requiem** (experimental): path tracing with WrongGPU's Ray Reconstruction, and Model M. It needs **REFramework**, installed by you first: the game blocks OptiScaler without
  it. WrongGPU never installs, downloads or replaces REFramework; the launcher tells you when it is missing and waits
  until it is there. The game keeps its graphics driver cache under its own name, so it prepares Ray Reconstruction at
  its first start (up to about a minute with a fresh driver; OptiScaler's overlay says "WrongGPU RR (preparing...)"
  meanwhile), and later starts have it ready at once.
- **Marvel's Spider-Man 2**: Model M; its Ray Reconstruction is experimental (see below).
- **LEGO Batman**: Model M. The game's menu has no Ray Reconstruction option (it would need a separate game mod), so WrongGPU's Ray Reconstruction isn't offered there.
- **Preset K** is not part of 0.6.0; it returns in **0.6.1**. If a game still starts on Preset K from an older WrongGPU,
  the launcher asks before it updates the game: **Switch to Model M** updates it and moves it to Model M (its other
  settings stay); **Not now** leaves the game as it is.

### Also fixed since 0.5.x
- Frames the network can't use (textures too small for the frame, for example during an intro or while the window is
  minimised) are passed through unchanged, and the network carries on with the next normal frame.
- When a game creates a new upscaler before destroying the old one, WrongGPU waits for the old one to release its video
  memory first (a few seconds at most) instead of running out of memory.
- Video memory no longer grows when you switch upscalers several times in OptiScaler's menu.
- The Ray Reconstruction plugin's extra diagnostic checks (a background thread) are off by default;
  `WG_RR_SLOT_CHECKS=on` turns them on for a bug report.
- Runtime options written on lines of their own in `wronggpu.ini` are no longer ignored.
- When an uninstall removes a linked WrongGPU data folder, it removes the link and keeps the linked folder's contents.
- One WrongGPU runtime per game, as before: `wronggpu_runtime.dll` holds Model M and Ray Reconstruction.

## Known issues
- RESIDENT EVIL requiem needs your own REFramework (see above).
- RESIDENT EVIL requiem: very occasionally a single frame comes out far too bright or too dark at a scene change (after
  loading a save, or during flashes of lightning). It lasts one frame; we're investigating it for 0.6.1.
- Hogwarts Legacy: since the game's November 2026 update, Ray Reconstruction isn't available with WrongGPU yet. Model M
  works. A fix is planned for 0.6.1.
- Preset K returns in 0.6.1.
- Radeon RX 9060 XT / RX 9060: supported, not yet tested. Ray Reconstruction should take about twice as long as on an RX 9070 XT
  (an estimate); 8 GB cards are untested.
- RESIDENT EVIL requiem prepares Ray Reconstruction at its own first start (see above), not in the launcher.
- One Ray Reconstruction plugin serves games on game interface 2.7, 2.8 and 2.14 (it picks the route when the game
  starts). Games on other game interface versions (for example 2.6) can't use Ray Reconstruction through the game
  interface, but it still works in those that create it directly, as Hogwarts Legacy did before its November 2026
  update.
- Ray Reconstruction needs a DLSS Ray Reconstruction file of version 310.7 or newer somewhere on your PC (for example
  from another game). Without one, the game keeps its own denoiser and the launcher says so.
- 007 First Light: not supported (crashes at start under OptiScaler).
- Marvel's Spider-Man 2: Model M only. Ray Reconstruction there needs OptiScaler's GPU-name spoof, which makes that game
  hang (with or without WrongGPU).

## Tested games
Tested on a Radeon RX 9070 XT. "Ray Reconstruction" means WrongGPU's Ray Reconstruction ran in the game; Model M means
super resolution only.
- **Cyberpunk 2077**: Ray Reconstruction and Model M.
- **RESIDENT EVIL requiem** (experimental): Ray Reconstruction when the game's **Ray Tracing** setting is **Path
  Tracing** and its upscaler is **DLSS**. Needs your own REFramework (see above).
  Very occasionally one frame comes out far too bright or too dark at a scene change (see Known issues).
- **Borderlands 4**: Ray Reconstruction (the game creates it itself; this needs the WrongGPU edition of OptiScaler).
  Known issue with this game: Ray Reconstruction needs one setting, and Borderlands 4 removes it on every start. After
  you install WrongGPU into it, the launcher asks. **Fix it for me** adds the setting and makes the file read-only, so
  the game keeps it (uninstalling WrongGPU undoes both). **Show me how** opens the game's page with the steps below,
  an **Open folder** button, the line with **Copy**, and **Check again**, which tells you whether the file has it. **Not
  now** asks again next time. The game's page changes the choice at any time. To do it yourself:
  1. open the game's settings folder `Documents\My Games\Borderlands 4\Saved\Config\Windows` (the folder that holds
     `GameUserSettings.ini`; if your Documents folder is in OneDrive, it is there);
  2. create or open the file `Engine.ini` there with Notepad;
  3. add the line `[SystemSettings]` at the end if the file doesn't have it, and under it the line from the launcher's
     **Copy** button on the game's page (or the release page);
  4. save the file, then right-click it, choose **Properties**, tick **Read-only** and click **OK**. Without that, the
     game removes the file at its next start.

  From a command prompt: `WrongGPU.exe set "<the game's folder>\Borderlands4.exe" rr_config_fix=on` (`off` takes
  the line out again). Measured at 1080p output (from 642x361): about 1.6 ms a frame.
  The exact line is on the release page and behind the game page's **Copy** button.
- **Hogwarts Legacy**: Model M. Ray Reconstruction worked up to the game's November 2026 update. On the updated game
  its Ray Reconstruction option is greyed out and WrongGPU's Ray Reconstruction isn't available yet, so the launcher
  doesn't offer it there (a fix is planned for 0.6.1). Before that update: the game's menu didn't turn it on (the game
  sets its own Ray Reconstruction option back to off), so it needs the same setting, and the launcher asks the same question after the install
  (**Fix it for me** adds the line; the game keeps the file, so no read-only is needed). To do it yourself: open the
  folder `%LOCALAPPDATA%\Hogwarts Legacy\Saved\Config\WindowsNoEditor` (the one that holds `GameUserSettings.ini`),
  create or open `Engine.ini` there, add `[SystemSettings]` at the end if it isn't there and under it the line from the
  game page's **Copy** button (or the release page), and save. That line turns Ray Reconstruction on whatever
  the menu says.
  The exact line (the same as Borderlands 4) is on the release page and behind the game page's **Copy** button.
- **Marvel's Spider-Man 2**: Model M (see Known issues; Ray Reconstruction is experimental there, see below).
- **LEGO Batman**: Model M. The game's menu has no Ray Reconstruction option (it would need a separate game mod), so WrongGPU's Ray Reconstruction isn't offered there.

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
- **No AMD HIP runtime needed any more** (0.4.5 needed one): Model M runs on Direct3D 12. HIP is only used for the
  optional sparkle guard.
- An internet connection the first time, for the WrongGPU edition of OptiScaler, which the launcher downloads itself
  (in 0.4.5 you supplied OptiScaler yourself; offline you still can, in Settings).
- For Model M, a DLSS file of version 310.5 or newer, and for Ray Reconstruction, a DLSS Ray Reconstruction file
  of version 310.7 or newer, each from any game or app on your PC (not necessarily the game you play); WrongGPU
  converts them on your PC and never ships them.

## Downloading and starting WrongGPU safely
WrongGPU's download is not signed with a code-signing certificate (they cost money every year and this is a free
project). That has three visible effects:
- **"Windows protected your PC"** (Microsoft Defender SmartScreen) can show when you start WrongGPU-0.6.0.exe the first
  time. Click **More info**, check that the file name is WrongGPU-0.6.0.exe, then click **Run anyway**. SmartScreen
  may show this warning for an unsigned program.
- **Check that your download is the real one** before you run it. The release page lists the SHA-256 of the zip, and
  SHA256SUMS.txt in the zip lists the SHA-256 of every file in it. In PowerShell (in the folder with the file):
  `Get-FileHash .\WrongGPU-0.6.0.zip -Algorithm SHA256` (or `certutil -hashfile WrongGPU-0.6.0.zip SHA256` in a
  Command Prompt). The long number it prints must be exactly the one on the release page. For the launcher itself, use
  `Get-FileHash .\WrongGPU-0.6.0.exe -Algorithm SHA256` and compare it with its line in SHA256SUMS.txt. If a number
  differs, delete the file and download it again from the release page only.
- **If Microsoft Defender flags or removes it**: first check the SHA-256 as above. If the numbers match, the file matches the published checksum, but that alone does not establish whether a detection is a false positive. You can restore it in Windows Security > Virus & threat protection > Protection
  history (choose the item, Actions, Restore), and please report the detection to us with the name Defender showed and
  to Microsoft at https://www.microsoft.com/en-us/wdsi/filesubmission so it gets cleared for everyone. If the numbers
  do not match, do not restore it: delete it and download it again.

DLSS is a trademark of NVIDIA Corporation; AMD, Radeon, FidelityFX and FSR are trademarks of Advanced Micro Devices, Inc. WrongGPU is an independent project and is not affiliated with or endorsed by either company.
