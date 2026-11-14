# WrongGPU installer (spec 30)

One Windows executable, `WrongGPU-Setup.exe`: setup wizard, uninstaller, log collector and silent mode. Native
Win32, no runtime dependencies, never uses the network, no telemetry. Adapted from the audited 0.2.0 installer
(WrongGPU 0.2.0's installer, not included here) for the FSR design of spec 20, with
the clean converter (`clean/converter`, spec 08) linked in as a library.

## OptiScaler-first wizard (0.4.1)

The interactive flow is Welcome → Install method → Choose a game → Review → Install → Done.
OptiScaler is selected by default. Choose the player's release archive (.zip or .7z), its
unpacked folder, or its OptiScaler.dll. A missing or rejected package blocks continuation.
OptiScaler is supplied separately; WrongGPU does not bundle or download it.

Direct FSR installation is an explicit advanced choice on the method page. Switching back
retains the chosen OptiScaler package and rechecks the install folder and conflicts. Existing
silent `/S` scripting behavior is preserved; `/OPTISCALER=<path>` selects the hosted route.
`/DIRECTFSR` records the advanced wizard choice for an administrator relaunch. The relaunch
also retains the exact OptiScaler source path, including spaces and Unicode characters.

The game page has a scrollable, wrapping requirements summary, with `OK`, `FIX`, and `NOTE`
labels. Help actions are separate from navigation. The review names the selected method;
the completion instructions match that method. Hosted users choose DLSS or XeSS in the
game and `WrongGPU Model M` in OptiScaler's Insert menu. Some host versions may still
show a legacy FSR version in the menu header; this installer does not claim to fix that.

WrongGPU's OptiScaler plugin (0.4.5.1, `src/wg_core/optiscaler.h`): the hosted route installs `plugins\WrongGPU.asi` (a
payload file built with the front end) for every game and sets OptiScaler.ini `[Plugins] LoadAsiPlugins=true`,
`Path=auto` (the section is added when the player's ini has none). On the OptiScaler builds the plugin recognises,
"WrongGPU" is then an entry of OptiScaler's main upscaler list and its default, and OptiScaler's FSR 3.X/4 entry is
AMD's upscaler; on any other build the plugin changes nothing and WrongGPU stays the first entry of the FFX box.
`Dx12Upscaler=fsr31`, `FfxDx12SRPath` and `UpscalerIndex=0` stay as before. Uninstall removes the plugin, its folder
when the install made it, and its files (`wronggpu-asi.log`, `wronggpu-asi.ini` in the OptiScaler folder), and puts the
player's OptiScaler.ini back. Tests: `test_asi_plugin`.
Other setups' plugins (0.6.1, `ForeignPlugins`): `.asi` files in `plugins\`, `OptiScaler\plugins\` or `wronggpu-plugins\` that
the install record doesn't name with that hash. In a Ray Reconstruction game WrongGPU's RR stays out by default (they may
hook RR too; the result says so); with the player's OK (the window's question, `--switch-plugins`) they are moved aside
(backed up, put back on uninstall) and WrongGPU's RR goes in: exactly the plugins the question listed (`--switch-plugins`:
those there at the time); one that is not listed keeps RR out. Repair and Update keep them aside on that OK (the earlier
install's restore brings them back; byte for byte those go aside again), and RR stays out when the exe folder is outside
the install folder (nothing to move them with). Elsewhere they stay beside WrongGPU's plugin.
WrongGPU RR from another package at exactly `plugins\WrongGPU-RR.asi` is overwritten by the install's RR plugin (backed up by its copy entry, put back on uninstall); under any other name it stays.

For 007 First Light (`007FirstLight.exe`) the installer also sets OptiScaler's `[Spoofing] DxgiFactoryWrapping=true`:
with OptiScaler 0.9.5.4 the game does not start otherwise when WrongGPU is OptiScaler's FSR library (the list is
`kOptiFactoryWrappingGames` in `src/wg_core/optiscaler.h`; other games keep OptiScaler's default).
Marvel's Spider-Man 2 (`Spider-Man2.exe`): its Ray Reconstruction is an experimental opt-in, off by default (the game
is unstable with OptiScaler's GPU-name spoof, 2026-11-04). An install writes none of its keys. `set <game>
sm2_rr_experimental=on` (that game only; not on the settings page) makes ApplyProfile write `[Spoofing] Dxgi=true`,
`SpoofedDeviceId=0x2684` and `UseFakenvapi=false` (in the section where the package's ini has it; a key the file lacks
is not written) into the installed OptiScaler.ini and journal them; no GPU name is written (`kOptiGameKeys`). `off`
puts the install's values back where the value is still ours; a repair or update follows the setting, and the
OptiScaler tab's reset leaves these keys to it. The installed OptiScaler.ini is a recorded file, so an uninstall takes
them away either way. Turning it on names the player's steps (`kGameNotices`; the release notes have them).

Borderlands 4 (`Borderlands4.exe`) and Hogwarts Legacy (`HogwartsLegacy.exe`) create Ray Reconstruction only with one
engine line in the player's own `Engine.ini`, section `[SystemSettings]`; Borderlands 4 deletes that file a few seconds
after every start unless it is read-only (Brief 16, 2026-11-04; `src/wg_core/rrconfig.h`). The line is written in one
place only, `src/wg_core/rrconfig_line.h`. The file is outside the game folder: `<Documents known folder>\My
Games\Borderlands 4\Saved\Config\<sub>\Engine.ini` (the known folder, so a Documents folder redirected to OneDrive is
found) and `%LOCALAPPDATA%\Hogwarts Legacy\Saved\Config\<sub>\Engine.ini`, `<sub>` being `Windows` or `WindowsNoEditor`,
whichever holds the game's `GameUserSettings.ini` (the game makes it at its first start; until then the sub-folder isn't
known and nothing is written: see `on` below). The per-game setting `rr_config_fix` (those two
games only; `ask` by default) decides. After an install, repair or update of a game still at `ask`, the window asks
(`src/app/ui_rrconfig.cpp`, "<game> needs one more setting for Ray Reconstruction"): **Fix it for me** sets it `on`,
**Show me how** sets it `off` and opens the game's page at its steps, **Not now** leaves it at `ask`. The game page's
**Engine.ini setting** row (WrongGPU | Myself, an Apply writes it) changes it at any time: it is there for those two
games whenever WrongGPU is installed, also where this build or the game's OptiScaler can't run their Ray Reconstruction
(the line is the game's own, whatever the OptiScaler edition). With Myself it shows the steps: the folder
(Open folder), the line (Copy) and a file check (`RrConfigLineState`: the line there, missing, not read-only, or the
settings folder not there yet; Check again). Open folder never makes the folder: before the game's first start its
sub-folder is only a guess, so the page says to start the game once and come back. With WrongGPU chosen, the row shows
the same check, with Fix it again (the choice applied again) when the line is gone or the file isn't read-only. The CLI
puts the same question, the steps and
the two commands into the result (`rr_config_question`); `install --rr-config-fix=on|off` answers it, `set <game>
rr_config_fix=on|off` changes it later, and other games refuse the setting. `on` before the game's first start (no
sub-folder holds `GameUserSettings.ini`): the choice is kept and nothing is created (no guessed folder: Hogwarts Legacy uses
`WindowsNoEditor`); the result's note, the toast and the card say "Will be added after you start the game once" (`get`:
`rr_config.waiting` -1), and the next Apply adds it once the folder is there: the window applies it when it opens or shows
the game's page (`GameEntry::rrConfigDue`, `ApplyRrConfigWhenDue`, once a session) or on Check again, and Repair/Update
apply it too. The window does that by itself only where it needs no administrator permission: for a game in a protected
folder (Program Files; `GameEntry::rrConfigProtected`, a write test while the choice waits) a permission prompt would
come out of nowhere, so it never starts there (the operation's worker checks again: `OpRequest::noPrompt` runs nothing in
a protected folder). Its card says "Waiting for the game's first start" while the folder isn't there (the line then waits
for the player's click, not for itself), then "Ready to add" with **Fix it now**, and that click (or Apply, Repair,
Update) brings the prompt. A game ready to add is also in the Library's "Needs attention" with Fix it now on its tile
(`RrConfigReadyToAdd`; the same Engine.ini-only apply), and the launcher's opening says it once in a toast. Engine.ini itself is in the player's profile, but the change is undone from the install record and the
backup in the game's `wronggpu-setup` folder (the uninstall, the restorer and Repair read them there), so it is never
written without them. A line an earlier build added in a guessed sub-folder that isn't the game's moves to the game's at the next
Apply. `on` (ApplyProfile, `ApplyRrConfig`) with the folder there: the player's file is backed up in the game's `wronggpu-setup\rr-config` folder (or noted as absent), the line is added only
where it is missing (under an existing `[SystemSettings]`, else that section at the end; a key with another value gets
the line's value), Borderlands 4's file is made read-only, and the record gets an `rrconfig` line. A repair keeps the
record line and the backup and applies it again. `off` and the uninstall (`RevertRrConfig`, also from DoRestore, so the
per-game uninstaller does it too) take the line out again (or put the key's value back), remove the section if the fix
added it and it is empty, clear read-only unless the file had it before, delete a file the fix created when nothing
else is in it (and the folders it made), and restore the backup when the file is gone. The elevated worker gets the
player's Documents folder in its job (`documents=`), as it gets `localappdata=`. Test builds take the Documents folder
only from `WRONGGPU_TEST_DOCUMENTS` and LocalAppData only from a test root (never the PC's own folders; without them
the fix refuses). Tests: `test_rr_config_fix`.

Both gfx1200 (RX 9060 family) and gfx1201 (RX 9070 family) payloads are retained. Physical
RX 9060 XT qualification is separate from the installer's synthetic architecture tests.

## Install routes and the build that requires OptiScaler (`WRONGGPU_DIRECT_ROUTE`)

The installer has two routes. **Hosted** (the wizard's default, `/OPTISCALER=<package>`): the player's OptiScaler goes
next to the game exe as a proxy DLL, WrongGPU's FFX DLL as `wronggpu_ffx_dx12.dll`; the game's own FSR files stay
untouched. **Direct** (the wizard's advanced choice, `/DIRECTFSR`, and a silent install without `/OPTISCALER=`):
WrongGPU's DLLs replace the game's FidelityFX DLLs, as described under "What it does".

The CMake option `WRONGGPU_DIRECT_ROUTE` (default `ON`, like `WRONGGPU_PRESET_K`) decides whether the direct route is
in the installer. `-DWRONGGPU_DIRECT_ROUTE=OFF` builds an installer that requires OptiScaler (the 0.4.5 release:
`-DWRONGGPU_DIRECT_ROUTE=OFF -DWRONGGPU_PRESET_K=OFF`, with `WRONGGPU_TESTING` off as in every release):

- **Wizard.** No method choice and no wording of the direct route on any page (method page, game page, review,
  finish page, game list); the step is "OptiScaler" and a missing or rejected package blocks continuation as before.
- **Command line.** `/DIRECTFSR` is an unknown option (exit 10 when silent). A silent install without `/OPTISCALER=`
  stops before anything is written: "OptiScaler is required: pass /OPTISCALER=<its release archive, folder or
  OptiScaler.dll>", exit 2; a package that is rejected likewise (exit 2, with the reason).
- **Which games qualify.** None of the three FidelityFX DLLs is needed: what the hosted route accepts qualifies (a
  game exe; the check names the DLSS, XeSS or FSR files it finds, and finding none is a note, not a refusal).
  Cyberpunk 2077 is "Validated",
  every other game "Experimental" (`/EXPERIMENTAL`), as before.
- **Setup over an earlier direct install** (any earlier version, also with WrongGPU's files replaced by hand): with a
  usable package the direct install is removed first from its own record and backup, exactly as its uninstaller
  would (the game's FSR DLLs back under their names byte for byte, see "Files that changed since the install"), then
  the hosted install is made; the record is a normal hosted one and its uninstaller leaves the game as it was before
  WrongGPU. Without a package the run is refused as above and the direct install stays as it is. The player's
  `wronggpu.ini` is kept as on any re-install: only what a hosted install sets (`upscaler` when missing,
  `runtime_dll`, `runtime_options`) is written, settings that only matter on the direct route stay as they are. When
  the direct install is not in the exe folder (the game's FSR DLLs in a plugin folder), it is removed there with its
  state folder and "Apps & features" entry, and `wronggpu.ini` moves to the exe folder (`RemoveEarlierDirect`,
  `src/wg_core/install.cpp`).
- **Payload.** `amd_fidelityfx_dx12.dll` (WrongGPU's FFX DLL), the runtime, the notices and the licence are enough.
  The FSR 3.0 component DLLs (`ffx_fsr3_x64.dll`, `ffx_fsr3upscaler_x64.dll`) are not shipped; if the payload folder
  has them they are not built in (`tools/make_payload.py --hosted-only`) and so never installed.
- **Unchanged.** Uninstall (also of a direct install, with this installer or its own uninstaller), collect logs,
  `/LIST`, `/HIPCHECK`, the model conversion, `DxgiFactoryWrapping` for 007 First Light, the Preset K switch.

`-DWRONGGPU_PRESET_K=OFF` builds an installer for a release that ships Model M only (no Preset K choice or wording,
`/PRESET=K` is an unknown option, no Preset K converter map built in, one model cache, `UpscalerIndex` always 0).
With the defaults nothing of this section changes the installer.

## What it does

- **Game detection.** Steam (all libraries), Epic and GOG installs, or any game `.exe` / folder the player picks.
  A store's launch exe (Epic's `LaunchExecutable`, GOG's `exe`) is a hint only (`StoreExe`): a launcher stub
  (`gamerules.h` `kLauncherStubs`: `REDprelauncher.exe`), or one beside which the discovery finds a known game's exe it
  doesn't lead to, gives way to the discovery below, as for a Steam game; a stub picked by hand resolves to its folder.
  Which exe below a folder is the game (`FindGameExe`, `src/wg_core/detect.cpp`): the Unreal shipping exe; else, of the
  exes that are not helpers (launchers, crash reporters, redistributables, the DirectX Agility SDK's
  `D3D12StateObjectCompiler.exe` and `d3dconfig.exe`), first by where the exe is: next to one of the three FidelityFX
  DLLs below, then next to `nvngx_dlss.dll`, `libxess.dll` or `amd_fidelityfx_upscaler_dx12.dll`, then anywhere
  else, last a folder that holds `D3D12Core.dll` (the Agility SDK's folder). Among those only the large exes count
  (at least a quarter of the largest); a DX12 name before a plain one before a DX11 / Vulkan one is the tie-breaker
  (`Control_DX12.exe` next to `Control_DX11.exe`), then the larger file.
  A game qualifies if its install contains one of the FidelityFX DLLs WrongGPU replaces (spec 20):

  | DLL | the game's file is kept as | recognised as the game's by |
  |---|---|---|
  | `amd_fidelityfx_dx12.dll` (FSR 3.1 API) | `amd_fidelityfx_dx12.wronggpu-original.dll` | version resource company "Advanced Micro Devices" |
  | `ffx_fsr3_x64.dll` (FSR 3.0) | `ffx_fsr3_x64.wronggpu-original.dll` | exports `ffxFsr3ContextCreate` and `ffxFsr3ContextDispatchUpscale` |
  | `ffx_fsr3upscaler_x64.dll` (FSR 3.0) | `ffx_fsr3upscaler_x64.wronggpu-original.dll` | exports `ffxFsr3UpscalerContextCreate` and `ffxFsr3UpscalerContextDispatch` |

  The FSR 3.0 component DLLs are built by the game's developer (no AMD version resource or signature), so their
  export table decides, as in the front end. Cyberpunk 2077 upscales through them. The install folder is the exe folder
  if it has one of the three, else the first folder below the game root that has one. Every one of the three that the
  game has and the payload contains is replaced (a DLL the payload lacks is left untouched and logged). Cyberpunk 2077
  is "Validated"; every other game is "Experimental" and needs an explicit confirmation.
- **Requirements check** before anything is written: game exe; each FSR DLL is the game's (table above), not
  another mod's or a leftover test install (`wronggpu-fsr-install.json` of `install-wronggpu-fsr.ps1`: the player is
  told to remove it with that script); the Authenticode result of each FSR DLL is logged, never required (no
  revocation check, no network); no anti-cheat (known games by exe or folder name, and marker files, lists from
  0.2.0; markers count only in the game's own folders: the exe's folder up to the game's top folder (through
  binaries folders such as `bin\x64`, `bin\x64_dx12`, `Binaries\Win64`, never past the folder directly below a
  library: `steamapps\common`, a Steam library, `Epic Games`, `GOG Games`, `XboxGames`, `Program Files`, a `Games`
  folder, a drive root) and two levels below that top folder, so another game's BattlEye in the same library never
  counts; the refusal names the marker's full path: `Anti-cheat found: <full path> - games with anti-cheat are never
  supported`); 64-bit Windows
  10/11; the GPU and the HIP runtime (below); a model file (below); other upscaler mods (OptiScaler,
  DLSS-NR on AMD: listed, then stop or move aside with a backup if the player agrees); older WrongGPU / "DLSS 4.5 AMD"
  installs (0.1.x/0.2.x: removed byte-exactly from their own record and backup, or their loose files moved into the
  backup, if the player agrees); the game is not running.
- **GPU and HIP runtime (spec 30 answers 7 and 8).** The DXGI adapter name is only a pre-filter ("RX 90", "PRO R97",
  "PRO R96"). The decision comes from a child process (`/HIPPROBE`) that loads `amdhip64_7.dll`, reads each device's
  architecture from HIP's device properties and creates a context on a supported one: `gfx1201` (RX 9070 family);
  `gfx1200` (RX 9060 family) is reported as not supported yet (one line, `kSupportedArch`, when the owner decides).
  Search order: System32 (installed by the AMD Software: Adrenalin Edition driver), the PATH folders, the newest
  `C:\Program Files\AMD\ROCm\<version>\bin`, `HIP_PATH`; `WRONGGPU_HIP_RUNTIME` overrides it in a `WRONGGPU_TESTING` build only (development; since 0.6.1). The child
  runs without `LLVM_PATH`. On failure the player is pointed to AMD's driver page
  (https://www.amd.com/en/support/download/drivers.html), the HIP SDK only as a second option. No SDK is needed.
- **Model file (bring your own).** Candidates in this order: a file the player picks (`Model file...` / `/MODEL=`),
  every `nvngx_dlss.dll` in the selected game, every `nvngx_dlss.dll` in the library games. Each is tried with the
  converter (spec 08 section 8 decides, not the file version); every rejection is logged. Several files: the first
  that converts completely **with** the output-filter weights (the optional group `filter`, spec 08 draft 2) is used;
  if none has them, the first that converts completely (the rule of spec 08 section 8 "Several DLLs" as the spec
  team amended it on 2026-10-13). No other folders are searched for this. The
  conversion runs in memory during the checks and is written at install time to
  `%LOCALAPPDATA%\WrongGPU\model\model.wgcm` with `model-source.txt` (which file, its version, where it was found,
  `filter` = whether that file had the weights, and a `not-used` line for each file that converts too but was passed
  over). The DLSS file is never copied or modified.
- **A model file that is already on the PC.** A `model.wgcm` whose stage digests match the embedded map is valid. It
  is reused without searching when it has the group `filter` and was written with the map this installer carries.
  A valid model M file without the group (written before the group existed, or from a DLSS file whose filter was
  not known), or one written with another map, is converted again with the same search:
  - a file that converts with the group is found: the new model file is written beside the one in use
    (`model\model-new\`), checked there, and only then moved over it; a copy of the earlier file
    (`model.wgcm.previous`) is kept until the result has passed its check in place and is put back otherwise;
  - no usable DLSS file, only files without a known output filter, a conversion error, or a new file that cannot
    be saved or fails its check: the model file in use stays as it is and the install succeeds as before.
  The player is never left without a working model. Which case it was is logged and written to the install record
  (`model-m`: `converted`, `reused`, `replaced` or `kept`, with the reason and where the output filter comes from)
  and to the support report (`system-info.txt`: `Model M output filter: ...`, `Model M file at the last install:
  ...`). Preset K has its own map without an optional group: its file is reused while it matches, as before.
- **The player's line about the output filter (spec 08 section 8).** When the conversion that was used has no
  group, or the model file that stays has none and the DLSS file found has none either, the player sees "The output
  filter of this DLSS build is not known to this version; WrongGPU's own filter will be used." as a `Note:` on the
  last page of the wizard (with the other notes of the install; `warning:` in the log of a silent install), and in
  the progress log. It is not shown when the model file has the weights.
- **Install.** Rename each replaced FSR DLL to `<name>.wronggpu-original.dll`, write the payload (WrongGPU's FSR
  DLLs under the original names, runtime, kernels), back up everything replaced or moved, write
  `<install folder>\wronggpu-setup\install.record` before changing anything, copy itself as the uninstaller, register
  one "Apps & features" entry per game (`HKCU\...\Uninstall\WrongGPU-<hash>`, Modify = collect logs). Any failure
  rolls the folder back. Nothing is written outside the game folder, `%LOCALAPPDATA%\WrongGPU` and that entry.
- **Update (setup run over an earlier install).** The earlier install is removed first from its own record and
  backup, as an uninstall does (the player's `wronggpu.ini` stays), then the new one is made. A failure after the
  earlier install was removed leaves the game's original files and no install: no state folder, no "Apps & features"
  entry. A failure while it is being removed (a file that is in use) loses nothing: its record and backup stay, and
  setup, or its uninstaller, can be run again. The logs that the OptiScaler of WrongGPU's own earlier hosted install
  wrote (`OptiScaler.log`) belong to that install: no `/MOVEMODS`, no warning. An OptiScaler that is not in the
  install record is another mod, as before.
- **Uninstall.** Restores every original byte for byte (hash-verified), removes what WrongGPU added (by the rule below),
  including the `wronggpu-*` files WrongGPU's own programs wrote while the game ran in the folders the record lists, and
  the entry, and `%LOCALAPPDATA%\WrongGPU` (model and logs) when no other game uses it.
- **What the launcher removes or overwrites: the file-ownership rule** (0.6.1; every restore, uninstall, repair, update,
  move out of a wrong folder and cleanup; `src/wg_core/record.h`). A file is deleted or overwritten only when its
  current bytes are WrongGPU's: the hash the install record says WrongGPU last wrote there (every write of the launcher
  to `wronggpu.ini`, `OptiScaler.ini` and the RR plugin's settings updates that hash), or a WrongGPU release's
  (`known_releases.h`, size and SHA-256) or the embedded payload's. Nothing else proves it: not the file's name, not a
  file next to it, not an ini key, not "it is a log". Two exceptions, and only these: the game's FSR DLL slots, where
  WrongGPU's DLL is recognised by its version resource (the game may update that DLL; a newer WrongGPU build placed by
  hand counts as WrongGPU's), and settings, log and text files with WrongGPU's own unique names (`wronggpu-*`, the RR
  plugin's `WrongGPU\wronggpu.ini`) that the record lists as created by WrongGPU. Shared names such as `OptiScaler.log`
  or `fakenvapi.log` are never deleted by their name.
  - Any other file at a recorded path (edited by hand, saved by OptiScaler's menu, replaced by another setup or a game
    update) stays where it is, untouched, with a warning.
  - When the original from before WrongGPU cannot go back because of such a file, it is kept in
    `<install folder>\WrongGPU-originals\<time>\` (a folder no operation removes), checked byte for byte; a second
    generation goes into a folder of its own, neither is overwritten. The result then says "partly" and names each file
    and where its original is.
  - Before anything changes, every recorded file is checked: one that cannot be read or replaced (another program holds
    it, such as the running game) or a missing or damaged backup stops the operation ("Nothing was changed"). Each file
    is checked again right before it is changed. The record is removed before the backup, so a stop at any point leaves
    either the record and backup for a second run, which finishes, or every original at its path or in
    `WrongGPU-originals`.
  - The game's original of an FSR DLL comes back from the backup, or from its `*.wronggpu-original.dll` copy when the
    backup is missing or damaged, checked against the hash in the record; when neither is there, the uninstall refuses
    ("Nothing was changed"). An uninstall that leaves an OptiScaler.ini naming WrongGPU's files warns that its setup may
    need them.
  - Moving another setup's files aside needs the player's OK for those exact bytes: the switch and plugin questions
    record each listed file with its SHA-256, and a file that changed since the question is not moved
    (`--switch` / `--switch-plugins`: the files as they are when the install starts).
- **Collect logs** (button, `/COLLECT`, or "Modify" in Apps & features): one .zip with WrongGPU's logs and crash
  notes, the install record, setup log, model source, system information (Windows, GPU, driver, HIP runtime, FSR DLL
  state, model state, where the output filter comes from, file hashes) and recent Windows crash / GPU-reset events.

## Contract with the FSR front end and the runtime (confirmed in ANSWERS.md, spec 30)

- The payload folder mirrors the install folder. At its root: WrongGPU's `amd_fidelityfx_dx12.dll` (required) and,
  from the front end's build, `ffx_fsr3_x64.dll` and `ffx_fsr3upscaler_x64.dll`. Everything else (the runtime DLL the
  front end loads, e.g. `wronggpu_runtime.dll`, kernel files, notices) at any relative path. Names starting with
  `nvngx`, `nvapi` or `dlss` are refused by the build.
- WrongGPU's DLLs have ProductName `WrongGPU` in their version resource (that is how the installer tells them from
  the game's) and do not name AMD as the company.
- The game's DLLs are next to them as `<name>.wronggpu-original.dll`.
- The model data is `%LOCALAPPDATA%\WrongGPU\model\model.wgcm` (format: `clean/converter/OUTPUT-FORMAT.md`).
- Files written while the game runs are named `wronggpu-*` (logs `wronggpu-*.log`, reports `wronggpu-report*`,
  crash notes `wronggpu-crash-*`); the player's settings file is `wronggpu.ini` (kept on a re-install, removed on uninstall). Until the front
  end renames it, `wronggpu-fsr.ini` counts as one of WrongGPU's files (collected, removed on uninstall).

## Release rule (README "Releases come only from this repository")

`WrongGPU-Setup.exe` compiles only the project's own code: the engine `src/wg_core` (adapted from the audited
0.2.0 installer), the widget kit `src/wg_ui`, the wizard and command line `src/app`, `src/wizard_policy.h`, the
clean converter (`clean/converter/src`), the generated payload table and resource script, the manifest
and the icon drawn by `tools/make_payload.py`. No third-party library is compiled in (no ImGui, no MinHook, nothing
else; `clean/THIRD_PARTY.md` is not needed for it). It uses only Windows itself: the Windows SDK headers and import
libraries, and at run time Windows DLLs (kernel32, user32, gdi32, gdiplus, dwmapi, uxtheme, comctl32, shell32, ole32,
advapi32; dxgi, wevtapi, wintrust and crypt32 are loaded from System32 when needed) and the player's AMD driver (HIP
runtime). The embedded payload must follow the same rule; the build refuses `nvngx*`, `nvapi*` and `dlss*` names.

Release builds are MSVC builds (`/MT`: the compiler's own C/C++ runtime is linked statically, as in the front end, so
no runtime DLL is needed). MinGW builds (`build-mingw.sh`, the Wine tests) link GCC's runtime libraries (libstdc++,
libgcc, winpthreads) statically and are for development and tests only, not for release.

## Source layout (launcher M0, M1, M2)

The installer is being turned into the WrongGPU launcher (`build/launcher-design.md`). Its first step split the
former `src/setup.cpp` with no behaviour change:

- `src/wg_core` (static library `wg_core`, namespace `wg`): the engine. No globals: a `wg::Context` (`context.h`: log
  file, the Preset K choice, the converter map, the player's OptiScaler package, `/NOSCAN`, the embedded payload
  table) is passed to every function that needs one. Modules: `util` and `pe` (helpers, PE version and exports),
  `payload`, `record` (install record, Apps & features entry, restore / uninstall), `detect` (store scan, game exe,
  FidelityFX DLLs, `InspectGame`), `coexist` (other mods, older installs), `checks` (Windows, GPU, HIP probe,
  anti-cheat, `RunChecks`), `model`, `optiscaler`, `settings` (wronggpu.ini keys), `install` (routes, FSR 4 for the
  comparison key, `DoInstall`), `logs` (the support bundle); for the launcher (M1) `profile` (per-game settings), `logtail` (the last run from
  `wronggpu-fsr.log`), `launcher` (paths, `launcher.ini`, game list, operations, job file, its own install, Play,
  update check) and `json_writer.h`; for coexistence (M2) `fingerprint` (what a DLL is, and a game's graphics API),
  `coexist` (the rules table and the proxy picker), `optikeys` (the OptiScaler tab and the inikey journal),
  `leftovers` (WrongGPU files without a record) and `known_releases.h` (generated by `tools/make_known_releases.py`).
- `src/wg_ui` (static library `wg_ui`, namespace `wg::ui`): the widget kit - palette, owner-drawn buttons, check boxes
  and radio buttons, progress bar, cards, fonts, the wordmark, the dark title bar.
- `src/app`: the program - `main.cpp` (entry point: window, verb or the installer's switches; silent mode), `cli.cpp` (the verbs),
  `launcher_ui.cpp` (the launcher window: input, operations, `ui-dump` / `ui-render`) with `ui_shell.h`, `launcher_ui_state.h` (what it shares with `ui_activate.cpp`, what a click on a widget does), `ui_gfx.cpp`
  (Direct2D / DirectWrite / WIC drawing, PNG output), `ui_art.cpp` (game art from the player's Steam cache),
  `ui_widgets.cpp`, `ui_library.cpp`, `ui_game.cpp`, `ui_settings.cpp` (the pages) and `ui_uia.cpp` (UI Automation),
  `restore_main.cpp` (the restorer), `wizard.cpp` (the old wizard), `app.h` (their shared state), `payload_embed.cpp`
  (the generated payload table for `wg::Context::data`).

Every source file stays below the 256 KB limit of the spec side's scan (`tools/nv_leak_scan.py`):
`tools/check_source_size.py` runs when CMake configures (a file at or over the limit stops the configure) and as the
ctest `source-size`.

## The launcher (M1)

`WrongGPU.exe` replaces `WrongGPU-Setup.exe` (`build/launcher-design.md`, milestone M1). One program, three jobs:

- **Window** (no arguments; `src/app/launcher_ui.cpp`). First run: Windows, the graphics card, the D3D12 executor,
  the HIP runtime (optional now: D3D12 is the default executor on RX 9000, a missing HIP runtime is a note), the
  model (found and converted as before), the games, and an adoption pass over 0.5.x installs; then *Install for this
  user* or *Run portable*. The game list shows every game with a status (can install, installed, update, repair,
  off, your OptiScaler, other mod, old install, orphaned, refused, running) and its last run, read from the tail of
  the game's `wronggpu-fsr.log` (upscaler, `network runtime ready: executor ...`, the OptiScaler plugin, foreign
  command-list tables, errors, the last `[support] WG-...` code). Each game's page: Install / Update / Repair /
  Uninstall / Roll back / Play (optional, Settings), Open folder / Open log / Collect logs, and the Basic settings
  (below) with Apply and the outside-edit check (Keep / Revert). Settings: the OptiScaler download, the model,
  extra game folders, the Play button, the opt-in update check, install / uninstall the launcher. Since M4 the window
  is painted with Direct2D and DirectWrite (the look of `build/launcher/ui-mockup`): a Library of art tiles (the
  player's own Steam cache, `<Steam>\appcache\librarycache\<appid>`, decoded with WIC off the UI thread; nothing is
  copied or shipped; games without art get a generated tile), a page per game with cards, custom-drawn menus and
  dialogs, DPI aware per monitor. Every widget is a UI Automation element with a name (`ui_uia.cpp`); typed text
  stays in native `EDIT` windows. A game folder that needs
  administrator rights goes to the elevated worker: the window writes a job file (operation, exact paths, the
  player's `%LOCALAPPDATA%`, the profile) and starts `WrongGPU.exe apply --job=<file> --result=<file>` with `runas`;
  the window itself never runs elevated.
- **Command line** (a verb as the first argument; `src/app/cli.cpp`): `scan`, `status [game]`, `install`, `repair`,
  `update`, `uninstall`, `rollback`, `adopt [game | --all]`, `set <game> key=value ...`, `set --global key=value`,
  `get <game> [key]`, `get --global`, `collect <game> [--zip=<file>]`, `hipcheck`, `model status|convert [--preset=M|L]`
  ("Model L by name" below),
  `rr-data status|convert [--dll=<RR model file>]` (both converts also take `--ignore=`, `--ignore-list=` and `--library=`:
  "Ignore lists" below),
  `self-install [--dir=] [--startmenu=<folder>|-]`, `self-uninstall [--keep-games]`, `play <game>`, `update-check`,
  `apply --job= --result=` (the elevated worker), `version`, `ui-dump --dump=<json> [--shots=<folder>]` (the window's
  pages painted offscreen, never shown: their widgets as JSON, PNGs, for tests), `ui-render --out-dir=<folder>
  --fixture=tests/ui_fixture.ini [--size=1180x760] [--scales=1,1.5] [--frames=library,game,...]` (fixture screenshots for development: named frames, among them empty and busy states, a big library, long names, dialogs). A game is its exe, its folder or its install
  folder. Options: `--out=<file.json>` (the result as JSON; console output of a GUI program is unreliable),
  `--log=`, `--data=<folder>` (the launcher's data), `--noscan`, `--library=<folder>` (scan this folder's
  sub-folders too, this run only), `--opti=<OptiScaler download>`, `--model=<nvngx_dlss.dll>`, `--experimental`,
  `--switch` (replace the player's own OptiScaler or another upscaler mod: their files are backed up, recorded as
  `moved` and put back byte for byte; without it the exit code is 6, "ask to switch"), `--removeold`, `--force`.
  Exit codes as below.
- **The installer's switches** (`/S`, `/DIR=`, `/EXE=`, `/OPTISCALER=`, `/MOVEMODS`, `/REMOVEOLD`, `/UNINSTALL`,
  `/COLLECT`, `/LIST`, `/HIPPROBE=`, ...) are aliases with the installer's behaviour, so `run_installer_tests.py` runs
  against `WrongGPU.exe` unchanged; `/WIZARD` (or switches without `/S`) opens the old wizard.

**Per-game settings** (`src/wg_core/profile.*`): `<install folder>\wronggpu-setup\profile.ini` is the source of truth.
Basic tier: `wronggpu` (on | off: `[fsr] upscaler = network | amd`), `executor` (auto | dx12 | hip), `output_filter`
(default | original | analytic | quality5 | quality9; quality5/9 are WrongGPU variations), `pre_exposure` (scale |
original), `sparkle_guard` (off | on | a rate in (0, 1)), `fps_overlay` (keep | on | off: OptiScaler's
`[Menu] ShowFps`, written only when that key exists in the game's OptiScaler.ini). On the game page Use WrongGPU is in view; Runs on, Output filter, Pre-exposure and Sparkle guard are under
"Advanced settings (4)", shown when one of them is off its default, unsaved or changed outside the launcher (`AdvancedWanted`). Every value is checked before
anything is written. `wronggpu.ini` gets only what differs from the defaults: the managed runtime items are removed
from `runtime_options` (and from lines of their own, which the front end appends) and the non-default ones appended
in a fixed order, one line without spaces; other items, comments, unknown keys and sections stay. A value in the
files that differs from both its default and the profile is an outside edit (Keep / Revert). After every install,
update and repair the profile is written into the files again; the record learns the new hashes of `wronggpu.ini`
and `OptiScaler.ini`, so an uninstall can still prove they are WrongGPU's.

**Record v2** (`WRONGGPU-RECORD 2`, `src/wg_core/record.*`): the v1 lines plus `payload`, `proxy` and one `owner` line
per action (`wg`, `wg-optibundle`, `adopted`, `moved`); action lines keep their five fields, so a 0.5.x uninstaller
reads a v2 record. `owned-files.json` (next to the record) is the public, read-only list for other tools. Adoption
(`adopt`, and the first run) upgrades a v1 record in place (the old file stays as `install.record.v1`), imports the
settings of `wronggpu.ini` into `profile.ini`, replaces `uninstall.exe` with the restorer and, when the launcher is
installed, replaces the game's own Apps & features entry with the launcher's.

**Update and roll back**: an install over another version first keeps that version's payload files in
`wronggpu-setup\previous` (`previous.txt`: version, hashes); Roll back swaps it in (and the newer files out, so a
second roll back returns). One version is kept.

**The restorer** (`WrongGPU-Restore.exe`, `src/app/restore_main.cpp`): wg_core without the payload, the converter
data or the UI, embedded in `WrongGPU.exe` (resource 960) and written into each game as
`wronggpu-setup\uninstall.exe`. It reads v1 and v2 records; `/S`, `/DIR=`, `/EXE=`, `/OUT=`, `/COLLECT`, `/LOG=`;
without `/S` it asks first.

**Where things live**: installed for this user, the launcher is `%LOCALAPPDATA%\Programs\WrongGPU\WrongGPU.exe` with a
Start-menu shortcut and one Apps & features entry "WrongGPU" (`self-uninstall`: remove WrongGPU from every game, or
keep the games, which then get their own entries back); its data (`launcher.ini`, `launcher.log`, `logs\`) is in
`%LOCALAPPDATA%\WrongGPU`. Portable (a file `WrongGPU.portable` next to the exe): the data is in
`<exe folder>\WrongGPU-data`, and each game keeps its own Apps & features entry. The model data stays in
`%LOCALAPPDATA%\WrongGPU\model` in both modes (the runtime reads it there). `managed-games.txt` in that folder lists
the managed games, so an uninstaller keeps the model while another game uses it.

**Model L by name** (#89; `src/wg_core/context.h` `SelectPreset`): `model status --preset=L` and
`model convert --preset=L [--model=<nvngx_dlss.dll>]` use Model L's converter map and model folder. The launcher embeds the
map (CMake `WRONGGPU_MAP_L`, default `data/preset-L.v1.map.json`, the spec side's revision v1.1; empty: a build without
Model L) and loads it as preset L, so a map that names another preset is "converter data damaged". The model goes to
`%LOCALAPPDATA%\WrongGPU\model\L\model.wgcm` with its own `model-source.txt`, the folder where the runtime looks for
`model=L`. Model M's files are never touched by it. A file without Model L's data is refused: "Model L can't be prepared:
no DLSS model file with Model L's data was found.", followed by the converter's reason for the file the player chose
(exit 5). An unknown name, or Model L in a build without its map, is a bad command line (exit 10). The JSON result names
the model (`preset`) and the converter map in use (`model.map_sha256`). Without `--preset` nothing changes. The window's
model choice and the runtime settings for Model L are not part of this step. `tests/run_model_l_tests.py` checks it
end to end: once with a build of the shipped map, and once with a build of a test map in its exact layout
(`--write-test-map`).

**The model choice** (#89; CMake `WRONGGPU_MODEL_L`, default `OFF`, on only in a release whose front end and runtime run
Model L; `src/wizard_policy.h`). With it off there is no model setting anywhere (profile, `get`, the window), and
`set <game> model=L` is refused by name (exit 10). With it on, `set <game> model=L|M` (profile key `model`, values `m`, `l`):
- **L** is accepted only while Model L's data on this PC (`model\L\model.wgcm`) passes its check against Model L's converter
  data. Otherwise: "Model L can't be chosen. Model L's data on this PC is not ready (...). Convert it first: ..." (exit 5,
  nothing changed). It writes wronggpu.ini's `runtime_options`: every model item out, `model=L` last (the last model item
  wins). It also writes OptiScaler.ini's `UpscalerIndex`, so OptiScaler starts with Model L's entry: `[WrongGPU]` in
  OptiScaler 10's slot, else `[FSR]`. The index is 1, or 2 in a build with Preset K (after Model M and Model K in the
  front end's list). Both writes go through the settings apply, so the install record's hashes follow them (the
  file-ownership rule).
- **M** is the default and changes nothing. Model L's items become `model=M` again, and an `UpscalerIndex` naming Model
  L's entry goes back to 0. Anything else (the installer's `model=M`, another entry the player chose in OptiScaler's
  menu) stays as it is. profile.ini has a `model` line only for L.
- A game saved on L whose Model L data is gone (removed, or not converted before an install, repair or update) is not
  refused. Its files say Model M with a warning saying how to convert the data, and the profile keeps L.
- The window has no model control yet. The choice is made with `set` (or a `profile.model=` line of an elevated job).

Tested by `model-choice` (ctest: profile.cpp built with the choice on) and `tests/run_model_l_tests.py --launcher-choice`
(a build with the choice on: refusal, both files, record hashes, back to M byte for byte, repair without the data,
uninstall), and with it off by `launcher-core` and the same script's build without the choice.

**Ignore lists** (`src/wg_core/convert_ignore.*`): WrongGPU converts only the vendor DLL files that came with the player's
own games, never a copy that another program's installer placed in a game. Such an installer registers the files it placed
in a list the launcher always reads, and the player does nothing. Every search for a file to convert (the model and the Ray
Reconstruction data; the window, the install, `model convert`, `rr-data convert`) reads the well-known lists:

- `%LOCALAPPDATA%\WrongGPU\ignore-files.txt` (the one an installer writes to; it creates the folder if needed and adds its
  lines, keeping the lines already there);
- `ignore-files.txt` in the launcher's own data folder when that is another folder: `WrongGPU-data\ignore-files.txt` next to
  a portable launcher, or the `--data=` folder.

`model convert` and `rr-data convert` add `--ignore-list=<file>` (repeatable; read once, by the same bounded reader as the
well-known lists, so a file of 16 MB or more is refused before it is read; a file that can't be read stops the command,
exit 10, nothing converted) and `--ignore=<entry>` (repeatable; one line of the format below, `--ignore=<path>`,
`--ignore=<path>  sha256=<hex>` or `--ignore=sha256=<hex>`; an entry that is not understood stops the command, exit 10). For
both the `=` is required: a bare `--ignore` or `--ignore-list` is an error (exit 10) and never takes the next argument as
its value. `--library=<folder>` searches that
folder's sub-folders as games too (this run only; `--noscan` leaves out the store libraries). These options are read by
those two verbs only (and, in a `WRONGGPU_TESTING` build, `rr-data prepare`); on any other verb they stop it with exit 10 and
a message, so a file is never believed protected by an option that was ignored.

**`--ignore=sha256=<hex>` matches every file with that SHA-256, wherever it is: every identical copy**, also the copy that
came with one of the player's own games (another program's installer usually places an unmodified vendor DLL, byte for byte
the one the games carry). Use it only to skip a file you know, for one run. An installer lists the file it placed by its path
(and, if it wants, the path with its hash), never by the hash alone.

The format (version 1, stable): a text file, UTF-8 (a BOM is fine) or UTF-16 LE with its BOM, CRLF or LF, one entry per line:

```
wronggpu-ignore-files 1
# placed by Example Tool's installer
C:\Games\Some Game\bin\x64\nvngx_dlssd.dll
C:\Games\Other Game\bin\nvngx_dlssd.dll  sha256=<the file's SHA-256, 64 hex digits>
```

- `wronggpu-ignore-files 1`: the format version (optional, first line by convention). A newer version is read with these
  rules, with one warning.
- A line whose first non-blank character is `#` is a comment (only whole lines: `#` may be part of a path). Blank lines are fine.
- A path: a full path with a drive letter or `\\server\share` (also written `\\?\C:\...` or `\\?\UNC\server\share\...`;
  quotes around it allowed; no wildcards, no environment variables, no relative paths). It matches the candidate that is the
  same file: the same fully resolved final path (case, `/` or `\`, `.` and `..`, symbolic links and junctions resolved), or
  the same file identity on its volume (a hard link). A path that doesn't exist now is compared as written. The tests cover
  a UNC entry through the PC's own administrative share (`\\localhost\<drive>$`), where it is reachable (otherwise they say
  so and skip those checks): a UNC entry matches the file found by its local path and the other way round (same file
  identity). Shares on other machines are not covered by a test.
- A path followed by whitespace and `sha256=<hex>`: matches **only when both match**: the same file at that path **and** that
  SHA-256. The entry names a file as it was placed; an identical copy under another path (the player's own game) is not
  skipped, and neither is the same path later holding other bytes (a game update). The file is hashed only when an entry
  with a hash applies to its path.
- `sha256=<hex>` alone is **not** a list entry: it would skip every identical copy of a vendor DLL, the player's own too.
  Such a line is skipped with a warning naming the list and the line (and what to write instead). Only the command line
  takes one (`--ignore=sha256=<hex>`, above).
- A line that is not understood (not a full path, a hash that is not 64 hex digits, a wildcard, a path that is a folder) is
  skipped with a warning naming the list and the line; the rest of the list is used.

A file on a list is skipped before its conversion is tried, also when it is the file chosen with `--model=` / `--dll=`
or in the window (no exception); the next file of the search order is tried (for `rr-data convert
--dll=`, which tries only the chosen file, nothing is converted: exit 5). The player is told why: "The file you chose (...)
is not converted: it is on the ignore list <file>, line <n> (path ...)" or "... it is named by --ignore=<entry> ...", and to
remove that line if it was listed by mistake (the result's `notes`, the install's warnings, the error when nothing else was
found). Every skipped file is named with the reason (the list and its line, or the `--ignore=` argument, and the path or the
hash) in the log (`model: skipped ...`, `RR data: skipped ...`), on the console and in the JSON result (`skipped`,
`ignore_warnings`). When every file found is on a list, nothing is converted and the exit code is 5; the message itself names
the first file skipped with its list and line (or `--ignore=` argument), says how many others were skipped and where the log
is, and to remove that line if it names a file that came with one of the player's own games. A list another program wrote
never switches the conversion off without the player being told which file did it.

"An empty list, or none, changes nothing" means the **choice of file**: the candidates, their order and so the file that
converts first are the same (`tests/convert_ignore_tests.cpp` runs the real candidate search with no list and with an
empty list in each place one is read; `tests/run_convert_ignore_tests.py` converts with each). Every conversion now also
records its source file's SHA-256 in `model-source.txt` (`sha256<TAB><hex>`, below), with a list or without one: one line
more in that file; the converted data itself is unchanged.

**A list that can't be read fails closed.** A well-known list that is there but can't be read (another program holds it
open without sharing, no permission, a folder of that name, a name the system refuses) is tried 5 times, 200 ms apart, when
the cause is a sharing violation (an installer may be writing it); still unreadable, nothing is converted or reused, the
message names the file and the reason and says what to do (close the program that holds it, or fix the file or delete it),
and the exit code is 10 (`model convert`, `rr-data convert`, and an install or
repair, whose model check fails with it). So does a list that can't be decoded: only UTF-8 and UTF-16 LE with its BOM are
read (UTF-16 big-endian, UTF-16 without a BOM and bytes that are not UTF-8 are not guessed at). A list that isn't there
(file or folder not found) is simply not read. A candidate whose SHA-256 can't be computed (it can't be read) while a hash
entry applies to it (`--ignore=sha256=`, or a path + hash entry for its path) is left out and named, never converted.

**Conversions on this PC are checked before they are reused.** `model-source.txt` (the model data's, and the RR data's in
`model\F`) records the source file's SHA-256 (`sha256<TAB><hex>`) besides its path. Wherever the launcher would reuse a
conversion (install, repair, update, the wizard, `model convert`; for the RR data, the install's step), and only while some
list has entries:
- it was converted from a listed file: its recorded path is a list's path entry, or its recorded path **and** recorded
  SHA-256 are a path + hash entry (both; the same rule as for a candidate), or its recorded SHA-256 is named by
  `--ignore=sha256=`. Then it is not reused; the search runs and the first allowed file that converts replaces it (with or
  without the output-filter weights). With none, it is not used: the model check fails (exit 5) with a message that says
  why, and an RR game's install goes on with RR off and a warning. The file itself is not deleted (other installed games'
  runtime reads it; nothing in the runtime changed). Data converted from a legitimate copy that has the same bytes as a
  listed file, at another path, is reused.
- The recorded source is compared **as it was recorded**: the recorded path's text (normalised: case, `/` or `\`, `.`, `..`)
  with the entry's path, as written and as it resolves now, and the recorded SHA-256 with the entry's. The recorded path is
  never opened again: if it was later replaced by a hard link or a symbolic link to a listed file, the data converted from
  the file that was there then is still reused.
- no SHA-256 is on record (converted by 0.6.0 or earlier): the path text alone decides (a path + hash entry naming the
  recorded path lists it, since its bytes can't be checked); not listed, it is unknown: converted again when an allowed file
  is found, else kept with a warning for the player.
With no list entries, a conversion is reused exactly as before.

**The uninstalls keep the list.** `remove-data` (and the window's Remove data), the launcher's uninstall (`self-uninstall`,
Apps & features) and the last game's uninstall (also through a game's own restorer) remove `%LOCALAPPDATA%\WrongGPU` as
before but put `ignore-files.txt` back in it, and say "kept ignore-files.txt (your/integrators' ignore list): <path>" (the
console and `kept` in `remove-data`'s result; the uninstall's output and log; a game uninstall's result warnings; the
window's toast). The launcher's own data folder's list (portable, `--data=`) was never removed and is now named the same way.
Without a list, the folder goes as before. **A player's list is never deleted**: when another program made
`ignore-files.txt` again between the folder's move and the put-back, the player's list is kept beside it as
`ignore-files.txt.kept-<n>` (the first free n; WrongGPU reads only `ignore-files.txt`, so the result says to copy the lines
still needed into it); when it can't be moved or copied out at all (a file where the folder should be), it stays in the
moved-aside folder, everything else in that folder goes, nothing deletes that folder's tree, and the result names where it
is. A list that is a link is moved as the link (what it points to is never touched). The tests drive each cleanup path
(`StartCleanup`, `RemoveSharedData`, `SelfUninstall`) through both cases.

A `WRONGGPU_TESTING` build has `rr-data prepare [--library=<folder>]`: the install's RR data step (reuse when the lists allow
it, else convert) without a game, for the tests.

**Update check**: off by default and no URL is built in (`launcher.ini` `update_check`, `update_url`; https only). When
both are set, one GET at start (or "Check now") reads a release object (`tag_name`, `html_url`) or text lines
`version=` / `page=`; nothing is downloaded. No other network use.

**Play**: a game in a Steam library whose app id its `appmanifest_<id>.acf` gives (the manifest in the library's `steamapps`
folder whose `installdir` holds the game) starts through Steam, `steam://rungameid/<appid>`: as from Steam's own Play, with
the player's launch options and Steam account (their saves, Steam Cloud). Every other game, a Steam game without an app id,
a PC without the `steam:` address handler, or a Steam start that fails: the exe in its folder without `LLVM_PATH`,
`ROCM_PATH`, `HIP_PATH` and `HIP_DEVICE_LIB_PATH` (Windows' administrator prompt when its manifest asks for it). The log
says which way (`play: ... started through Steam (app id N, ...)` / `... started directly`); `play <game>` prints it and
writes `started` (`steam` / `direct`), `appid` and, for a Steam game started directly, `steam_note` to its JSON result.
Games started from Steam, Epic or GOG work exactly the same.

## Coexistence with other mods and tools (launcher M2)

**Fingerprints** (`src/wg_core/fingerprint.*`): a DLL is recognised by what it contains - its version resource
(company, product, description), its exports (`ReShadeRegisterAddon`, `hipInit`, the FidelityFX API), then text in it
(one pass, ASCII and UTF-16, most specific first: OptiScaler, UE4SS, RED4ext, Cyber Engine Tweaks, Ultimate ASI
Loader, Special K, ReShade, vkd3d-proton, DXVK, fakenvapi, DLSS-NR on AMD, DLSS 4.5 AMD) - never by its file name.
Names only decide which files are looked at (the names Windows loads from a game folder). A game's graphics API comes
from its exe's import and delay-import tables (and `d3d12.dll` named in it). `WrongGPU.exe fingerprint <file> ...`
shows the verdict and why.

**The rules table** (`src/wg_core/coexist.*`, design section 5; `WrongGPU.exe coexist <game>`, the game page, `status`
JSON `coexistence`). Decisions: managed, ask-switch, chain, warn, inform, refuse, migrate.

| Found | Decision |
|---|---|
| WrongGPU's own OptiScaler setup (record + hash) | managed; its `OptiScaler.log` / `OptiScaler.ini` and an OptiScaler DLL whose hash the record or the package being installed has are never "your OptiScaler" |
| The player's OptiScaler (any name it loads as, also the older `nvngx.dll` layout) | ask to switch (owner decision 2026-10-17): without `--switch` exit 6; with it, their files are backed up (owner `moved`) and put back byte for byte on uninstall; the switch reuses the name their OptiScaler had |
| Two OptiScaler copies loaded by one game | refuse (exit 6) until the player keeps one |
| ReShade, Special K next to the exe | chain: they stay, OptiScaler takes another free name ("untested combination"); the game page warns about AMD FSR fallbacks from their command-list tables |
| DXVK (as `d3d9`, `d3d10core`, `d3d11`, `dxgi.dll`), vkd3d-proton (as `d3d12`, `d3d12core.dll`), recognised only by their own version resource or settings names (`DXVK_CONFIG_FILE`, `dxvk.conf`, `VKD3D_CONFIG`, ...), never by a bare "dxvk" in a file; an OptiScaler build naming DXVK is OptiScaler | refuse (exit 2): Windows' own Direct3D 12 is needed |
| `nvngx.dll` whose version resource or signer is the DLSS DLL's vendor (an older OptiScaler layout's DLSS DLL copy) | warn: "Old OptiScaler layout (nvngx.dll = DLSS DLL copy)"; left alone, never written |
| an unknown `d3d12.dll` | warn |
| Ultimate ASI Loader | warn; WrongGPU's plugin goes to `wronggpu-plugins\` and OptiScaler's `[Plugins] Path` names it (when that OptiScaler build has the key; else a warning that it may load twice) |
| Cyber Engine Tweaks (`version.dll` + `plugins\cyber_engine_tweaks`), RED4ext (`winmm.dll` + `red4ext\`), UE4SS | chain: their names are never taken |
| A HIP-based tool (`neuralhub.ini`, its own HIP DLLs) | warn: keep the D3D12 executor; `version.dll` is never taken |
| DLSS-NR on AMD | refuse (exit 6); never touched, never offered for cleanup |
| A loose `nvapi64.dll` / `nvngx.dll` of another tool | warn; WrongGPU never writes those names for itself |
| RTSS, MSI Afterburner, Lossless Scaling, OBS, Discord, Overwolf, the Steam and EOS overlays (running) | inform (Lossless Scaling: upscaling twice is pointless) |
| Special K's global injector (SKIF running) | warn |
| Anti-cheat | refuse (exit 4), no override; only removing WrongGPU's own recorded install is allowed |
| An exe that imports another graphics API and shows no sign of D3D12 | refuse (exit 2) |
| An older WrongGPU / DLSS 4.5 AMD install | migrate (`--removeold`) |

Refusals hold for the installer's switches too (`/S`: the conflicts line of the checks fails). The DLSS-NR on AMD mod is
no longer moved aside by `/MOVEMODS`.

**Proxy picker**: our earlier install's name; else the name the player's OptiScaler had (a switch); else the first of
`dxgi, winmm, version, wininet, winhttp, dbghelp` that no recognised tool, unknown file or game's own Windows DLL
holds (`version` never next to a HIP-based tool); never `d3d12`, `dinput8`, `dwmapi`, `xinput*`. The record's `proxy`
line keeps the name and why ("dxgi.dll (ReShade), winmm.dll (RED4ext), ... taken"); each install also records
`coexist <tool> <fingerprint> <decision>` lines.

**OptiScaler tab** (`src/wg_core/optikeys.*`; the game page's second tab; `WrongGPU.exe opti <game>`,
`set <game> opti.<Section>.<Key>=value`, `opti <game> --reset | --keep`): WrongGPU's keys are shown locked (the ones
setup writes, `[Hotfix] MipmapBiasOverride` reserved, `[Menu] ShowFps` owned by the Settings tab). Editable: the menu
key, FPS overlay style, menu scale, DXGI factory wrapping, GPU spoofing and the input-source keys - only those the
installed OptiScaler build has. Values are checked (true/false/auto, numbers, key codes) and only the value's bytes
change. Every key the launcher changes in OptiScaler.ini is journaled in the record (`inikey <file> <section> <key>
=<original>|absent =<ours>`): Reset puts the originals back byte for byte, a value OptiScaler's own menu saved over
ours shows as an outside edit (Keep / Put ours back), and an update or repair carries the player's values over to the
new OptiScaler.ini (keys the new build lacks are named). Uninstall still restores the player's own files from the
backup byte for byte.

**Leftovers and orphans** (`src/wg_core/leftovers.*`): files without an install record count as WrongGPU's only when
their SHA-256 is one of our releases' (`known_releases.h`: 0.3.0 to 0.5.5-ea, from our release payloads) or of the
embedded payload. Status `leftovers`; *Clean up* (`clean-leftovers`) removes exactly those files (each hash checked
again; a game DLL a direct install had renamed comes back under its own name); *Adopt* (`adopt-leftovers`) makes an
install record for them (the game's renamed DLL in the backup, as the install engine records it). An install over
leftovers needs the player's OK (`--removeold`, else exit 7). Never automatic. An older direct install in an Unreal
Engine game's plugin folders (a `signedbin` folder under `Engine\Plugins` or a project's `Plugins`, up to four folders
above the exe; found in LEGO Batman) counts too: its DLLs by hash or WrongGPU's version resource, shown by their path
from the game's root. The old install's `wronggpu.ini`, `wronggpu-fsr.log` (and `.log.old`), `wronggpu-license.txt`
and `wronggpu-notices.txt` count only when they sit in the same `signedbin` folder as such a DLL (never by name
alone). *Clean up* / `--removeold` removes them and puts the game's DLL back; a FidelityFX DLL of ours
stays (with a note) when the game's own is not there to put back; *Adopt* is refused there. Orphaned installs (the game's exe
gone) are uninstalled by their folder; a file is deleted only when the record says it is ours and its hash is the
installed one, or it is provably a WrongGPU build (version resource or a known release hash; never its name); other
files are left with a warning.

**Warnings from the logs** (`src/wg_core/logtail.*`; the game page, `status` JSON `last_run.warnings`):
`foreign-vtable` (command lists recorded into another tool's tables: AMD FSR fallback), `no-depth` /
`xess-no-depth` (the game's inputs carry no depth; XeSS named when OptiScaler.log says so), `plugin-not-hooked`
(`wronggpu-asi.log`: WrongGPU stays in OptiScaler's FSR sub-list). "Last run" counts only real problems
(`IsProblemLine`): errors (`ERROR` as a word, WG-ERROR incidents), refused splits, fallbacks to AMD's FSR with their
reason; info lines (hooks installed, `upscaler info:`, runtime ready, the settings line) never count, even when they
quote `WG_ERROR_*` codes. Start-up refusals (`refused (WG_ERROR_NOT_READY)`, "until the runtime is ready": the first frames go to AMD FSR while
the runtime starts) are routine in a run whose runtime became ready (`RoutineNote`): neither counted nor shown there.

**Adoption of 0.5.x settings**: only the settings schema's keys (`kSettings`) go into `profile.ini`. Developer and test
keys (`runtime_options` items such as `timing_log`, `capture_dir`, `model_dir`, `watchdog`; diagnostics lines such as
`log_dispatches = 1`; an `executor` value outside the schema) are not imported: each is listed as `ignored: ...` /
`not imported: ...` in the result, in setup.log and in the first-run summary; wronggpu.ini keeps them as they are.

**Payload and build id**: the launcher embeds the 0.5.5-ea build (runtime `331bc00e...`, front end `3511adb5...`,
plugin `e9aab52c...`, Model M only); its version stays 0.5.5 and pre-release builds carry a build id
(`-DWRONGGPU_BUILD_ID=0.5.5-launcher-dev`, `-DWRONGGPU_PAYLOAD_NAME=0.5.5-ea`): the window's footer, `version` and
setup.log show them.
## Silent mode and exit codes

`WrongGPU.exe /S` (the installer's switches; `WrongGPU-Setup.exe` up to 0.5.1) with `/DIR=<folder>` or `/EXE=<game exe>`, `/MODEL=<nvngx_dlss.dll>`, `/EXPERIMENTAL`,
`/MOVEMODS`, `/REMOVEOLD`, `/NOSCAN`, `/FORCE` (skip the GPU and HIP checks: tests and development; not for the
player guide), `/UNINSTALL` (with `/OUT=<zip>`: collect logs first), `/COLLECT [/OUT=<zip>]`, `/LIST`, `/HIPCHECK`
(the HIP runtime check alone: DLL, device and architecture in the log; exit 0 or 2), `/LOG=<file>`. The uninstaller is
`<install folder>\wronggpu-setup\uninstall.exe` (`/S` for silent).

| Code | Meaning |
|---|---|
| 0 | done |
| 1 | install or uninstall failed (an install is rolled back) |
| 2 | a requirement is not met (game, FSR DLL missing, Windows, GPU, HIP runtime, game running, ...; in a build without the direct route: no `/OPTISCALER=` package, or one that is rejected) |
| 3 | experimental game: pass `/EXPERIMENTAL` |
| 4 | anti-cheat found (never supported) |
| 5 | no DLSS model file converted |
| 6 | another mod: pass `/MOVEMODS` (or, if it replaced `amd_fidelityfx_dx12.dll`, verify the game files) |
| 7 | an older WrongGPU / DLSS 4.5 AMD install: pass `/REMOVEOLD` |
| 8 | the installer is damaged (payload or converter data) |
| 9 | uninstall: no install record in that folder |
| 10 | bad command line (also `/DIRECTFSR` in a build without the direct route, `/PRESET=K` in one without Preset K) |

## Building

- Windows, MSVC + CMake (release):
  `cmake -S clean/installer -B build -DWRONGGPU_PAYLOAD_DIR=<release payload folder>` then
  `cmake --build build --config Release`. Output `WrongGPU.exe` (the launcher) and `WrongGPU-Restore.exe` (the restorer it embeds); the converter's `wronggpu-convert.exe`
  and tests are built alongside. `WRONGGPU_MAP` defaults to `data/preset-M.v1.map.json`. Switches (both default
  `ON`): `-DWRONGGPU_DIRECT_ROUTE=OFF` (OptiScaler is required) and `-DWRONGGPU_PRESET_K=OFF` (Model M only), see
  "Install routes and the build that requires OptiScaler". Python 3 is needed at
  build time (`tools/make_payload.py` embeds the payload and the map, writes the resource script and draws the icon).
  The application manifest is a CMake source under MSVC (the linker embeds it once, `/MANIFESTUAC:NO`); MinGW gets it
  through the resource script.
- Linux, MinGW-w64: `./build-mingw.sh <payload folder> [map] [version] [out dir]`, or CMake with a MinGW toolchain file.

## Tests

`python3 tests/run_installer_tests.py` (Linux: builds a test installer with MinGW and runs it under Wine in its own
prefix; about 50 s). It uses only mock game folders, mock Steam/Epic/GOG libraries, mock FSR DLLs
(`tests/mockpe.py`: PE files with real version resources and export tables), a mock payload with the three FSR DLLs,
mock HIP runtimes (`tests/mock_hip.c`) and synthetic model files from the converter's test generator with a matching
synthetic map (version 4, with the optional source F). 90 checks before the output-filter cases, which add 52 on
Windows (the Wine run was not repeated for them):

- 30-T1: install / collect / uninstall round trips with byte-exact restore and nothing left behind (Cyberpunk layout
  with all three FSR DLLs, FSR 3.0 components only, Unreal with the API DLL only, FSR DLL in a plugin folder, two games
  side by side, re-install), rollback after a mid-install failure, anti-cheat refusals, no FSR DLL, foreign API DLL,
  component DLL without the FSR 3.0 exports, a leftover script install, no GPU, other mod (stop, or move aside and
  restore), older 0.2.x install with record (byte-exact removal, registry entry) and without record, game updates of
  the API DLL and of `ffx_fsr3upscaler_x64.dll`.
- 30-T2: model search order (picked file, game folder, library games), rejection of old files with the version shown,
  the converted `model.wgcm` compared bit for bit with the converter tests' reference, reuse while it matches the
  map, a damaged model file not reused, model removed with the last game, the DLSS file never modified; `/LIST`.
- The output-filter weights (spec 08 draft 2; the suite's synthetic map is version 4 with source F, and its default
  source has the block of variant 1): fresh installs from a source with the block of variant 1, of variant 2 and
  without a block (group present with that variant's digest, or absent with the player's line), each compared byte
  for byte with a container the test writes itself from `OUTPUT-FORMAT.md`; an existing model file with the group
  reused without a conversion; an earlier model file (written by the test with the version-1 map) with a source
  that has the block (converted again, replaced only after the new file passed its check, nothing left beside it),
  with a source without the block (kept, the player's line), with no source or only a rejected file (kept); a new
  conversion that cannot be saved beside the file in use, and one damaged on disk before its check (a test hook of
  `WRONGGPU_TESTING` builds): the earlier file intact, the install succeeds; a file of this map without the group,
  and a file with the group from another map; several DLLs in both orders and two without the block; what the
  install record, `model-source.txt` and the support report say in each case.
- Setup over an earlier install, and files that changed since it (32 checks, run on Windows): WrongGPU's FSR DLLs and
  runtime replaced by hand with other builds of WrongGPU, then setup again (installed, the game's originals kept,
  nothing of the hand-placed files in the record or the backup, byte-exact uninstall) or uninstall (also without the
  backup folder, and with a record that has no entry for WrongGPU's DLL); controls: a game update of two FSR DLLs
  before setup is run again, another mod's DLL (refused, nothing changed); an update that fails while the new files
  are written, one that stops right after the earlier install was removed (a test hook of `WRONGGPU_TESTING` builds),
  one with a file of the earlier install in use (native Windows only: nothing lost, the second run finishes);
  WrongGPU's own OptiScaler-hosted install with the logs its OptiScaler wrote (no `/MOVEMODS`), and an OptiScaler
  that is not in the record (exit 6 as before).
- An installer without the direct route (`-DWRONGGPU_DIRECT_ROUTE=OFF`; the suite sees it in the installer itself,
  `--direct-route on|off` adds a check that it is built as expected): cases [1] to [11] and [14] are the direct
  route's and are skipped; the HIP check, the output-filter cases and the checks of a build without Preset K run
  through the hosted route with a mock OptiScaler package. Its own cases (`test_without_direct_route`, 45 checks):
  no text of the direct route or of a method choice in the program; `/DIRECTFSR` exit 10; a silent install without
  `/OPTISCALER=`, or with a folder that is not OptiScaler's, exit 2 with nothing written; hosted installs into a game
  with all three FidelityFX DLLs (untouched), with none of them, with no upscaler file at all (experimental), and
  007 First Light; collect logs and `/UNINSTALL` without a package; the FSR 3.0 DLLs not built in
  (`make_payload.py --hosted-only` on a payload folder that has them). Setup over earlier direct installs, which an
  installer with the direct route makes (`--direct-setup <WrongGPU-Setup.exe built with the defaults from the same
  sources and test map>`; without it that part fails): refused without a package with the direct install intact;
  with one the game's three FSR DLLs back byte for byte and a normal hosted install, `wronggpu.ini` kept, uninstall
  back to the game before WrongGPU; the same with WrongGPU's DLLs replaced by hand, without the backup folder, after
  a game update, with the direct install in a plugin folder, a failure while the hosted files are written, and
  `/UNINSTALL` of a direct install. `tests/wizard_policy_tests.cpp` (ctest `wizard-policy` of a `WRONGGPU_TESTING`
  build) checks the wizard's route policy and texts for the configuration it is built with.
- `tests/wg_core_tests.cpp` (ctest `wg-core`, every build): unit tests of the engine that need no game, payload or GPU
  (strings and paths, SHA-256, the install record and `Target`'s refusal of paths outside the folder, wronggpu.ini
  keys, the runtime options of a Model-M-only build, the Context's log file); ctest `source-size` checks the 256 KB
  limit.
- `tests/launcher_core_tests.cpp` (ctest `launcher-core`, every build, 72 checks): settings validation and rendering
  (one `runtime_options` line without spaces, enum and range rejection, the last value wins, runtime keys on lines
  of their own, defaults never written, comments and unknown keys kept, a second render changes nothing), outside
  edits, OptiScaler keys only where they exist, `profile.ini`, record v1 -> v2 (owners, payload and proxy lines, the
  v1 file kept byte for byte, a 0.5.x reader still sees every action line, unknown lines kept, `owned-files.json`),
  `VerifyInstall`, the log-tail parser on synthetic logs, job and result files (relative paths refused), the update
  reply and the update check staying off without a URL, the game environment of Play, Play's way (Steam app id from an appmanifest fixture, the direct start
  otherwise; the starts stubbed), `launcher.ini`.
- `tests/convert_ignore_tests.cpp` (ctest `convert-ignore`, every build): the ignore-list format (version line,
  comments, a path, a path with its hash, quotes, UTF-8 with a BOM, UTF-16 LE, LF; a hash alone, a folder and a list that
  can't be decoded refused), bad lines (each warned with its list and line, the rest read), matching (case, `/`, `.` and
  `..`, a hard link, a junction both ways, a symbolic link both ways where the account may make one, else it says so and
  skips those checks; UNC through `\\localhost\<drive>$` where reachable, else it says so), a path + hash entry matching
  only when both match (two identical stand-ins in two folders, only one listed: the other is still tried; the same path with
  other bytes is not skipped), `--ignore=sha256=` wherever the file is, a candidate that can't be hashed while a hash entry
  applies (left out), the well-known list locations, the reason naming the list and line or the `--ignore=` argument, a list
  held open without sharing (unreadable after about 0.8 s; read when let go during the retries), a folder of that name, a
  list in UTF-16 big-endian and a name the system refuses (unreadable), the recorded-source check (path, path + hash,
  hash), `StartCleanup` keeping the list, `PutIgnoreListBack` logging when it can't, and the real candidate search
  (`AllowedModelCandidates`, which `FindModel` runs before it converts) with no list and with an empty list in each place
  one is read, the same candidates in the same order, plus a list that does change it (so the test can fail).
- `tests/run_convert_ignore_tests.py --launcher <WrongGPU.exe> --work <dir> [--synth-f <folder>]` (CPU only:
  `model convert`, `rr-data convert`, `rr-data prepare`, `remove-data`, `self-install` / `self-uninstall` with
  `--startmenu=-`, always `--noscan`; no game install, no window): the installer suite's "several DLLs" sets and library-only
  sets converted with no list and with an empty list in each place one is read (the same winner, model file and
  `model-source.txt`); each list used (`%LOCALAPPDATA%`, the data folder, `--ignore-list=`, `--ignore=`), every candidate
  ignored (exit 5, nothing written, the message), `--ignore=sha256=`, a hash alone in a list (warned), a path + hash entry (two identical files, one listed: the other converts), a folder entry, bad lines, path spellings, a junction, a symbolic link,
  a hard link, UNC (`\\localhost\<drive>$`, `\\?\UNC\`), the command line's own mistakes (exit 10: a bare `--ignore`, the options on other verbs), a chosen file on a list (the
  list and line, or the argument, named); the reuse checks of the model and RR data (recorded SHA-256; listed by path, path + hash or
  `--ignore=sha256=`, with and without an allowed file; a path + hash entry with another hash or path: reused; no SHA-256 on record with and without one; no list); a list held open (exit 10
  after the retries, read when let go, also before a reuse; a folder of that name); `remove-data` and `self-uninstall`
  keeping `ignore-files.txt` and saying so.
- `tests/launcher_ignore_cases.py` (run by `run_launcher_tests.py` after its rc6 cases; installs, so it runs with that suite):
  the ignore lists on the install path in a work folder of its own: a listed model source not reused by a repair (exit 5,
  the list and line), converted again by a second game's install, a list held open (exit 10), and the last game's uninstall
  keeping `ignore-files.txt` with the kept line in its result.
- `tests/run_launcher_tests.py --launcher <WrongGPU.exe> --work <dir> [--baseline <0.5.1 WrongGPU-Setup.exe>]
  [--hip <mock>]`: the verbs against generated fixture games (names and synthetic bytes only) with a
  `WRONGGPU_TESTING` build of the suite's test map: scan statuses, refusals (anti-cheat, DLSS-NR) with nothing changed,
  install (record v2, owned-files.json, the restorer as uninstall.exe, the managed list), set/get with validation,
  OptiScaler's FPS key, an outside edit and its revert, the last run from a synthetic log, update with `previous\`,
  roll back twice, repair, the elevated worker's job file, collect, uninstall by the launcher and by the restorer
  (byte for byte), the player's OptiScaler (exit 6, then `--switch`: backed up as `moved`, restored byte for byte),
  the experimental confirmation, the model kept while another managed game uses it, adoption of a real 0.5.1
  install (v1 record, settings imported, restorer, entries), self-install / self-uninstall with one Apps & features
  entry, portable paths, the update check off, and the window built hidden (`ui-dump`: pages, names, visibility).
  Its `[direct-rr]` part: a game that ships the RR library without the game interface's RR plugin (the Unreal Engine
  plugin layout of Hogwarts Legacy, LEGO Batman, Borderlands 4) creates Ray Reconstruction itself (`rr.h`
  `RrUse::direct`; status `rr_direct`). WrongGPU's RR runs there only through WrongGPU's edition of OptiScaler (plugin
  interface 4 or newer hands the game's RR calls to the RR plugin): installed with another OptiScaler, the RR files go in,
  the result warns, `status` says `rr_available: false` with `rr_unavailable_why`, and the RR card says RR is unavailable
  with the plain reason; a repair with the edition makes it available (`optiscaler_edition`, read from the install
  record's hash of OptiScaler's DLL). Cyberpunk 2077 and RE Requiem always count as game-interface RR games.
- `tests/coexist_tests.cpp` (ctest `coexist-core`, every build): the marker scan and fingerprints on synthetic bytes,
  tool and decision names, overlays, the proxy picker on a generated folder, byte-exact OptiScaler.ini value edits,
  the OptiScaler tab's rules (locked, editable only when present, value checks), the inikey journal (revert byte for
  byte, a key back at its original, outside edits, replay onto a new ini), record v2's inikey and coexist lines, the
  log warnings (foreign tables, no depth, XeSS, plugin not hooked; the plugin's log in its own folder).
- `tests/run_coexist_tests.py --launcher <WrongGPU.exe> --work <dir>`: one case per rule of the rules table against
  generated fixture games whose other-tool files are synthetic stand-ins (`tests/coexist_fixtures.py`, built with
  `mockpe.py`: version resources, exports, imports, a few words of text; never a real third-party file): fingerprints
  under misleading names, ReShade / Special K (chain, another proxy name, byte-exact uninstall), refusals with every
  confirmation (DXVK, vkd3d-proton, DLSS-NR on AMD, anti-cheat, two OptiScaler copies, a Direct3D 11 exe; D3D12
  evidence in the imports, in the exe, in an Agility SDK folder), Ultimate ASI Loader (the plugin's own folder,
  OptiScaler's Path, its log removed on uninstall), CET / RED4ext / UE4SS, a HIP-based tool, loose nvapi64 / nvngx,
  every proxy name taken, overlays, an older WrongGPU, the player's OptiScaler (a recognised build reused by name, the
  older nvngx layout moved aside, both put back byte for byte), the OptiScaler tab (locked / editable, refusals,
  outside edit and keep, update carrying values over, the job file, reset byte for byte), log warnings, leftovers
  (hash-matched only, clean up, direct-layout clean up and adopt), an orphan with a changed file, and the window's
  game page and OptiScaler tab (`ui-dump`).- Which exe is the game (`test_game_exe`, 8 checks, both routes; installs go through a mock OptiScaler package): a
  folder shaped like The Witcher 3's `bin\x64_dx12` (the game exe with DLSS, XeSS and FSR 4 files, and a `D3D12_0`
  folder with the Agility SDK's DLLs and programs) given as `/DIR=` itself and through the game root (what the
  library scan hands over; a DX11 build of the same name in `bin\x64`), installed and uninstalled byte for byte;
  `Control_DX12.exe` next to a larger `Control_DX11.exe`; a small tool with a DX12 name against the game; a game
  with a DX11 exe only; an unknown program in the Agility SDK folder. Under Wine also `/LIST` with that game in a
  mock Epic library (not run on Windows, where the suite never scans the tester's libraries).
- The HIP check against mock runtimes: gfx1201 accepted, gfx1200 "not supported yet", gfx1100 refused, missing or
  not a HIP runtime; `LLVM_PATH` removed for the child (the mock fails when it sees it); the newest ROCm folder
  (7.10 before 7.2) tried first.

On Windows: `python tests\run_installer_tests.py --prepare-only` writes the mock payload and the synthetic map; build
the installer from them with CMake (`-DWRONGGPU_PAYLOAD_DIR=... -DWRONGGPU_MAP=...`) and run
`python tests\run_installer_tests.py --setup <WrongGPU-Setup.exe>`. Natively it points `LOCALAPPDATA` and
`ProgramData` at its work folder, always passes `/NOSCAN`, touches only "Apps & features" entries that point into its
work folder, and skips the Steam/GOG library tests (they need registry keys of their own; run them under Wine) and
the ROCm search-order test (it would write to `C:\Program Files`). The HIP mocks are built with MinGW `gcc` or, from
a Visual Studio developer prompt, MSVC `cl`. On a PC with an RX 9070-series card and the driver's HIP runtime, the
"no GPU" case instead checks that the real GPU and HIP checks pass, then uninstalls.

### For the spec team: 30-T2 with real model files

Build with the real map (default) and any payload (the mock one from `--prepare-only` is enough for this test), then
for each real `nvngx_dlss.dll`: `WrongGPU-Setup.exe /S /FORCE /EXPERIMENTAL /NOSCAN /DIR=<a mock game folder with an
AMD amd_fidelityfx_dx12.dll> /MODEL=<the DLL>`. Supported builds: exit 0 and
`%LOCALAPPDATA%\WrongGPU\model\model.wgcm` byte-identical to `wronggpu-convert --map <map> --dll <the DLL> --out <dir>`;
older builds: exit 5 with the "no preset-M model" reason in the log. Also: the same with the DLL placed in the game
folder or a library game instead of `/MODEL=`. 30-T3 (Cyberpunk 2077) needs the real FSR front end in the payload.
# Support reports

The existing Collect logs button and `/COLLECT /DIR=<game> /OUT=<report.zip>` save
`support-manifest.json` alongside available logs/settings/install metadata. The
manifest includes captured-byte SHA-256s, all enumerated adapter identities/VRAM,
driver registry versions, HIP candidate fingerprints and recent structured failure,
frame, provider, settings and loaded-module observations from the frontend. File
hashes are observed at collection time; runtime observations retain their own times.

Reads are bounded to 64 source files, 8 MiB tails and a 64 MiB source-data budget.
Truncated, omitted, unreadable and malformed evidence is explicit. Newest diagnostic
records are summarized first. Memory dumps are not automatically included. Collection
does not execute a game/GPU probe or upload a report. Reports contain local paths and
user names; the collector explains this before the user shares them. `/OUT` must name
a ZIP so collection cannot inadvertently overwrite a game DLL or settings file.
Stable WG-SETUP references supplement existing error text and exit codes.
