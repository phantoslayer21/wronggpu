# Test plan for spec 20 (FSR front end) — for the spec team

Game: Cyberpunk 2077 (Steam) on a PC with an RX 9070 XT. Install only on a PC whose owner agrees. Games with
anti-cheat are off limits (the install script refuses them).

What you test: WrongGPU replaces three of the game's FidelityFX DLLs, each forwarding to the game's original kept
next to it as `<name>.wronggpu-original.dll`:

| WrongGPU's DLL | path | what WrongGPU does |
|---|---|---|
| `amd_fidelityfx_dx12.dll` | FSR 3.1 API (integration point 1) | upscaler contexts → stand-in; everything else → AMD |
| `ffx_fsr3_x64.dll` | FSR 3.0 components (integration point 2) | logs; frame generation and optical flow always AMD's |
| `ffx_fsr3upscaler_x64.dll` | FSR 3.0 components (integration point 2) | logs; upscale dispatches → stand-in (after AMD's upscaler when frame generation needs its outputs) |

20-T0 (first run, 2026-10-05) showed that Cyberpunk 2077 upscales through the FSR 3.0 component DLLs and uses the
FidelityFX API DLL only for jitter queries. The stand-in (bilinear or Lanczos, no network yet) is softer than AMD
FSR; that is expected.

Send back (as `results/<date>-fsr-20.md`): pass/fail and numbers for each step below, any error text, and the log
files that the uninstall step moves out of the game folder (`wronggpu-fsr.log`, `wronggpu-fsr.log.old` if present,
`wronggpu.ini`). The log holds only FidelityFX call parameters (sizes, formats, flags, jitter values, pointers) and
file paths.

## 0. Build and offline tests (once per commit)

From a fresh clone, in a "x64 Native Tools Command Prompt for VS 2022":

```
git checkout clean/fsr-hook
cmake -S clean/fsr -B build/fsr -G "Visual Studio 17 2022" -A x64 -DWG_BUILD_ID=<commit>
cmake --build build/fsr --config Release
ctest --test-dir build/fsr -C Release --output-on-failure
powershell -ExecutionPolicy Bypass -File clean/fsr/tests/test-install.ps1
```

`ctest` runs the two unit tests (`fsr_frontend_unit`, `fsr30_hook_unit`) and the D3D12 end-to-end tests on the real
GPU (`fsr_dll_d3d12_*`, 30 modes, each prints its numbers; the `hooks_trace*` and `split_*` modes test the D3D12 hooks
and the command-list split of 20-T4/20-T5 offline, the `network_*` modes `upscaler = network` with a test double of the
runtime). The MSVC build runs them with the D3D12 debug layer when Windows'
"Graphics Tools" optional feature is installed (Settings → System → Optional features); each test prints
`D3D12 debug layer: on` or why not, and fails on a debug-layer error. Please install Graphics Tools for this run, and
report the `D3D12 debug layer` lines and every `D3D12 debug layer ERROR`/`warning` line. Report: pass/fail per test and the printed numbers
(`max |GPU - CPU reference|`, `max |GPU - scene|`, `warpdiff ... mean difference`, `srgb: worst difference`). The
payload for the game is the three DLLs in `build/fsr/Release` (`amd_fidelityfx_dx12.dll`, `ffx_fsr3_x64.dll`,
`ffx_fsr3upscaler_x64.dll`; `wronggpu-notices.txt` next to them is the release's notices file and is not installed).
Use the MSVC build for the game; the MinGW build is for development only. (The tests run in `build/fsr/dlltest/Release`, which also holds test doubles named
`*.wronggpu-original.dll`; never copy that folder into a game.)

## 1. Install

Close the game. In an administrator PowerShell (Steam folders are usually under Program Files):

```
cd <repo>\clean\fsr\install
.\install-wronggpu-fsr.ps1 -Action Install -GameDir "<Steam library>\steamapps\common\Cyberpunk 2077\bin\x64" -Payload <repo>\build\fsr\Release
```

It installs every one of the three DLLs that is in both folders (`-Components <name>,<name>` limits that). It refuses
(and changes nothing) if another mod's proxy DLL or upscaler mod is present, if the game's `amd_fidelityfx_dx12.dll`
is not signed by AMD, or if WrongGPU is already installed. The FSR 3.0 component DLLs are built by the game's
developer, so they need not carry AMD's signature: the script shows each one's signer; report them. If it refuses
because of another mod, remove that mod first. Report: the "Installed." lines (hashes) and the
signer lines.

In AMD Software (Adrenalin), note whether an FSR override/upgrade for the game is enabled and report it; keep it off
for these tests.

Settings live in `bin\x64\wronggpu.ini` (written with defaults and comments on the first start). Change them
with the game closed. The log is `bin\x64\wronggpu-fsr.log`.

## 20-T0. Which FSR path the game uses (forwarding only)

1. Start the game once and quit at the main menu, so `wronggpu.ini` exists. Set `upscaler = amd` (all three DLLs
   then only log and forward).
2. Start the game, set AMD FSR 3 (upscaling) on, Quality, frame generation **off**, load a save, play 1 minute (or
   the built-in benchmark), quit. Then once more with frame generation **on**.
3. The image must look exactly like AMD FSR (everything goes to AMD). Report FPS for both runs.
4. In the log (each line is tagged `[amd_fidelityfx_dx12]`, `[ffx_fsr3]` or `[ffx_fsr3upscaler]`; every call shows the
   module that made it after `from`), report:
   - the load lines of each DLL: `AMD's original DLL loaded: ... file version ...`, `AMD's DLL is an FSR 3.0.x build`
     (or `SDK 1.1`), and any `note: AMD's DLL has exports WrongGPU does not provide` line;
   - the first `ffxFsr3ContextCreate` and/or `ffxFsr3UpscalerContextCreate` line and its result, with the `from` module;
   - the first two `ffxFsr3ContextDispatchUpscale` and `ffxFsr3UpscalerContextDispatch` lines (each with its `from`);
   - with frame generation on: the first `ffxFsr3ConfigureFrameGeneration` and `ffxFsr3DispatchFrameGeneration` lines;
   - every line containing `ERROR` or `not available in AMD's DLL`.
   What the answers mean: `ffxFsr3UpscalerContextDispatch from ffx_fsr3_x64.wronggpu-original.dll` = the game calls
   `ffx_fsr3`, which calls the upscaler DLL (the stand-in will take over in the upscaler DLL);
   `ffxFsr3UpscalerContextDispatch from Cyberpunk2077.exe` = the game calls the upscaler directly; only
   `ffxFsr3ContextDispatchUpscale` lines = `ffx_fsr3` has the upscaler built in (the stand-in will take over after it).
5. If the game does not start, or a DLL never logs `loaded as`, report that and uninstall (20-T3).

## 20-T1. Stand-in upscaler, frame generation still AMD's

Set `upscaler = standin`, `view = upscale`, `filter = bilinear` (keep `fsr3_route = auto`, `fsr3_amd_first = auto`).

1. Start the game, FSR 3 upscaling on, Quality, frame generation off, 2560×1440 (or the monitor's resolution; report
   it). Load a save.
   - Expected log for Cyberpunk 2077 (FSR 3.0 path, `UPSCALING_ONLY` context):
     `[ffx_fsr3upscaler] ... -> FFX_OK (0x00000000) context C1 (created by an ffx_fsr3 context with UPSCALING_ONLY:
     AMD's upscaler is not needed)`, then `WrongGPU upscaler "WrongGPU stand-in upscaler" create`, and per frame
     `-> FFX_OK (WrongGPU stand-in; AMD's upscaler skipped: no frame generation)` and
     `[ffx_fsr3] ... (1 upscaler dispatches inside, handled in ffx_fsr3upscaler)`. No `ERROR`.
   - If you see `(AMD for the frame-generation resources, then WrongGPU stand-in)` with frame generation off, the
     UPSCALING_ONLY hint did not reach the upscaler DLL: report the `ffxFsr3ContextCreate` and
     `ffxFsr3UpscalerContextCreate` lines. If you see `stays on AMD`, `upscaler context goes to AMD FSR` or
     `now runs on AMD FSR`, the stand-in was not used: report the line before it.
   - AMD's upscaler does not run in this mode, so FPS may be above AMD FSR's (the stand-in is cheap); report it.
2. The image: correct position and size (no offset, no stretch, no black borders), no flicker, HUD normal, stable
   when standing still. Softer than AMD FSR is expected.
3. Repeat for Balanced, Performance, Ultra Performance and Native AA (if offered); change the mode in game without
   restarting at least once; change the display resolution once; Alt-Tab out and back; load another save.
4. Frame generation (if offered, from the game's menu): turn it on with the stand-in upscaling. Expected log:
   `(AMD for the frame-generation resources, then WrongGPU stand-in)` (AMD's upscaler runs underneath for frame
   generation, ANSWERS.md item 12 (a)). It must still work (frame counter about doubles, no new artifacts beyond those
   of soft upscaling). Report FPS with frame generation off and on, and the first `ffxFsr3ConfigureFrameGeneration` and
   `ffxFsr3DispatchFrameGeneration` lines (the frame-generation half of 20-T0).
5. Set `filter = lanczos`, check 2. again (slightly sharper).
6. Report per item pass/fail, FPS in a fixed spot (stand-in and, for comparison, `upscaler = amd`), and the
   `first frame recorded` line from the log.

## 20-T2. Conventions (motion vectors and jitter)

All with `upscaler = standin`, FSR Quality, frame generation off.

1. Motion: `view = warpdiff`, `debug_gain = 4`. In a well-lit street, (a) stand still, (b) turn the camera slowly
   left/right, (c) walk sideways. Right conventions: the screen stays mostly black; only object edges that appear
   or disappear, moving NPCs/cars without motion, particles and the HUD light up. Wrong conventions: the outlines of
   all static scenery light up while the camera moves.
   Then set `motion_sign_x = -1` and `motion_sign_y = -1` and repeat (b)/(c): it must get clearly worse. Set them
   back to 1. Report for each setting: mostly black / scenery outlines lit.
2. Smearing: `view = warp`, `warp_weight = 0.9`. Turn and walk: static scenery must not smear or leave trails
   (moving objects and newly revealed areas may ghost; there is no rejection in this debug view).
3. Jitter: `view = jitter`. The dot in the box at the top-left (at the box center + 128 · j, j = +jitterOffset) must
   jump to a different place every frame (it should look like a fuzzy cloud, not one still dot); in the log,
   `jitterOffset=(x,y)` and the frame line's `jitter=(...)` (now equal to it) must change every dispatch and repeat
   with the game's phase count (17 at 2560×1440 Quality).
4. Jitter compensation: `view = upscale`, `filter = bilinear`, stand still in front of high-contrast static detail
   (fences, text on signs). Compare three settings (restart between them):
   (a) `jitter_compensation = 1`, `jitter_sign_x = 1`, `jitter_sign_y = 1` (default since 0.3.2: j = +jitterOffset),
   (b) `jitter_compensation = 0`,
   (c) `jitter_compensation = 1`, `jitter_sign_x = -1`, `jitter_sign_y = -1` (the old default).
   Right convention: (a) is the most stable (no sub-pixel wobble), (b) wobbles, (c) wobbles about twice as much.
   Report which is most stable.
5. Sanity views (optional): `view = motion` (grey at rest, colors while moving) and `view = depth` (near objects
   bright). Report if anything looks inverted or offset.

## 20-T4. D3D12 hooks and submission trace (AMD FSR, nothing changed on screen)

What it shows: where Cyberpunk 2077 submits the command list that holds the FSR dispatch (queue, which
`ExecuteCommandLists` call, position in it, other lists submitted between recording and submission), and that
WrongGPU's own D3D12 hooks (COM vtable patches, needed for the network's same-frame execution) run in the game
alongside its overlays without harm.

Set `upscaler = amd` and `d3d12_hooks = on` (keep `split_test = 0`).

1. Start the game, FSR 3 upscaling on, Quality, frame generation off, load a save, play 1 minute in a fixed spot.
   The image must be exactly AMD FSR's. Report FPS, and FPS in the same spot with `d3d12_hooks = auto` (no hooks with
   `upscaler = amd`), to see the cost of the hooks. Leave the overlays you normally use (Steam, AMD Software) on; report
   which ones are on.
2. Frame generation on, 1 minute; then Alt-Tab, change the display resolution once, load another save. No crash,
   no hang. Report FPS with frame generation on.
3. Screenshot: one, in the fixed spot (hooks on).
4. Report from the log (all lines contain `d3d12 hooks:`):
   - `installing WrongGPU's D3D12 vtable hooks (d3d12_hooks = on) ...`, every `... vtable ... patched` line (if it ends
     with `already pointed outside the vtable's module: chained to them`, another program had hooked those entries
     first: report it), `command list interfaces of vtable ...`, `installed (...)`;
   - every `queue Q<n> seen: ...` line;
   - the first 10 `submit: FSR list ...` lines, 5 from the middle of the run, and 5 after turning frame generation on.
     Each says on which queue (Q<n>), in which `ExecuteCommandLists` call and at which position the FSR list was
     submitted, on which thread it was recorded and submitted, how long after the dispatch, how many lists the game
     submitted between recording the dispatch and submitting it (`before this call: N lists in M calls`; N > 0 means
     submitting at the dispatch point would have changed the game's order, which is why WrongGPU uses the "ordered
     split"), and since the previous FSR list;
   - every `trace: ... reset without being submitted`, `ERROR` or `note:` line.

## 20-T5. Stand-in upscaling through the command-list split (same-frame execution without the network)

What it shows: the whole same-frame machinery the network backend will use, with the stand-in in place of the
network. At each upscale, WrongGPU splits the game's command list:
- the commands recorded so far run on WrongGPU's own queue, once the game's queue reaches that list in the game's own
  order;
- a "done" fence follows; a second WrongGPU queue signals "resume" right after it (standing in for the network's HIP
  work);
- the list reopens with the game's state restored, the stand-in records its upscale, and the game continues.

Set `upscaler = standin`, `split_test = 1`, `view = upscale`, `filter = bilinear` (`d3d12_hooks = auto`).

1. Start the game, FSR 3 Quality, frame generation off, load a save. Expected log:
   - `d3d12 hooks: installing WrongGPU's D3D12 vtable hooks (a backend sets WG_BACKEND_NEEDS_SPLIT)`, then `installed (...)`;
   - once at the start: `split of list ... refused (WG_ERROR_NOT_READY)` and `split_test: no split this frame` (the
     first frame cannot be split; the stand-in upscales it without the split);
   - then every frame: `split S<n> of list ... first half on W1 behind order ticket <n> ... <k> bindings restored`
     and `split S<n> released in ExecuteCommandLists #<c> on Q<q>, position <p> of <m> ...`;
   - no `ERROR`. `released from the CPU` should not appear: if it does, report the lines around it.
2. The image must look like 20-T1 with the stand-in (soft, correctly placed, no flicker). The split must not break
   anything after the upscale: watch the HUD, menus, post-processing, sharpening, film grain/vignette and the photo
   mode for missing or corrupted elements, black or frozen frames, and stutter. Screenshots: the 20-T4 spot with
   `split_test = 0` and with `split_test = 1`.
3. FPS in the same spot with `split_test = 0` and `= 1` (the cost of the split itself: an extra submission and two
   cross-queue waits per frame).
4. Frame generation on (from the game's menu): expected `(AMD for the frame-generation resources, then WrongGPU
   stand-in)` plus the split lines; frame generation must still work. Report FPS.
5. Change the quality mode in game, change the display resolution, Alt-Tab, load another save, quit to the main menu
   and load again. No crash or hang.
6. Report per item pass/fail, FPS, the first 10 `split S`/`released` lines, any `refused`, `released from the CPU`,
   `ERROR` or `note:` lines, and the `<k> bindings restored` counts you see (how many bindings the game had set
   when FSR was called).

If the game hangs in 20-T5 (a split list whose resume never comes), close it with the Task Manager, set
`split_test = 0` and send the log.

## Running the network runtime (`upscaler = network`)

For the runtime session's tests: install as in step 1 with `-Runtime <path to the runtime build's
wronggpu_runtime.dll>` (or put `wronggpu_runtime.dll` into the payload folder); the script copies it next to the FSR
DLLs as `wronggpu_runtime.dll`, shows its version and removes it again on uninstall (20-T3 stays byte for byte). If the
runtime needs other files next to it, tell the front-end session first. Set `upscaler = network` (keep
`d3d12_hooks = auto`, `split_test = 0`). Expected log at the first upscale:
- `WrongGPU runtime ...\wronggpu_runtime.dll (runtime_dll = wronggpu_runtime.dll) loaded: file version ...`,
  `WrongGPU runtime upscaler "...": interface version 2, flags 0x1 (same-frame split)`;
- `d3d12 hooks: installing WrongGPU's D3D12 vtable hooks (a backend sets WG_BACKEND_NEEDS_SPLIT)`;
- `WrongGPU upscaler "..." create for C1: interface=2 ... split=available`;
- the first frame: `refused (WG_ERROR_NOT_READY)` and `note C1: upscaler not ready this frame (WG_ERROR_NOT_READY):
  stand-in for this frame`; then per frame the `split S<n>` and `released` lines of 20-T5 and
  `-> FFX_OK (<runtime name>; AMD's upscaler skipped: no frame generation)`.
Every failure is one line in words: the runtime missing or not loadable (`ERROR: WrongGPU runtime ...`, with the
Windows error text), its create or dispatch failing (`create failed: WG_ERROR_...` / `dispatch failed: WG_ERROR_...`
plus the runtime's own `upscaler ERROR:` line), and where the context went (`C1 stays on AMD`). Send those lines.

## 20-T3. Uninstall restores the files byte for byte

Close the game. In the administrator PowerShell:

```
.\install-wronggpu-fsr.ps1 -Action Uninstall -GameDir "<...>\Cyberpunk 2077\bin\x64" -LogsTo <folder for the logs>
```

It moves WrongGPU's log and settings to `-LogsTo`, restores the game's DLLs, and compares every file of `bin\x64` with
the hashes taken at install time. Report the `20-T3 PASS` or `20-T3 FAIL` line (with the listed differences).
Optional cross-check: Steam "Verify integrity of game files" reports nothing to fix.

## If something goes wrong

- The game does not start or crashes: run the uninstall step, then send the log.
- 20-T4/20-T5 problems: set `d3d12_hooks = off` (no hooks at all) to confirm WrongGPU's hooks are the cause, and send
  the log.
- A step's result is unclear: send the log lines around it; each FidelityFX call has a number `#N` (counted per DLL,
  see the `[tag]`) and is logged with every parameter and its caller, followed by its result (`-> OK` / `FFX_OK`, or
  an error and what was done about it).
