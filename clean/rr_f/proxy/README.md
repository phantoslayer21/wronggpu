# RR proxy (spec pages 60-63)

WrongGPU's own replacement of the game's Streamline interposer library (60.1). The install tool gives the proxy the
original's file name; the original is renamed and loaded by the proxy from the proxy's own directory with a full
path. Every export is forwarded unchanged; the ray-reconstruction feature (id 1001) is answered by the proxy and
runs through an owned adapter in WrongGPU's runtime. WrongGPU.asi reads and changes the RR settings through a
contract the proxy exports (60.9).

**Two deliveries, one core.** The above is the *file-replacement* transport (`src/proxy_dll.cpp`). There is also a
*hook-mode* transport (`src/hook_dll.cpp`, **HOOK-MODE.md**): an OptiScaler ASI plugin (`WrongGPU-RR.asi` /
`WrongGPU-RR-probe.asi`, per version) that leaves the game's interface file in place and redirects the
import-address-table slots of its importers for the RR path instead. In-game probing of the file-replacement proxy
showed the game verifies the interface DLL's signature before `<init>`, so an unsigned replacement is never
initialised; hook mode avoids replacing the signed file. Both transports reuse the proxy core, probe, answers,
settings contract and runtime adapter unchanged — only the delivery differs.

Status: built per Streamline version from the real export tables of annex A9 (2.7.x: 261 exports, 2.14.1: 262;
same names and ordinals, ordinal = 1-based ASCII rank; `WrongGPU_GetSettingsContract` at the ordinal above the
highest), bound by name (A9.2), version magic `0xfedc` with every version value forwarded unchanged (A9.1). The
calls A9.4 marks for the proxy are implemented with the 60.6 declarations and the argument and return types of
annex A10 (every call returns the 32-bit signed `Result`; the frame index is read only through the token's table,
A10.1); every other export goes through argument-agnostic stubs. The release build answers the game's
`<get-function>(1001)` requests with the real feature-function names (A9.5). Tested on the CPU against a fake
original with the real export names and a fake runtime. **Not usable for RR in a game yet**: the runtime library
does not exist yet (until then RR reports 7 and everything else is forwarded). Nothing here was taken from a vendor
library or header; no Streamline binary was read.

A **probe build** (build flag `WG_RR_PROBE`, `wg_rr_proxy_probe.dll`, never part of the release build) answers the
probe questions of 63.6 in one game run: it claims RR as 60.8 says, answers `<rr-optimal>` with the 61.5 rule,
writes a plain bilinear upscale of the colour input into the output for every RR evaluate (returns 0), is
read-only on every other game resource, forwards everything else, and logs names and numbers to
`wronggpu-rr-probe.log` next to itself (below). Optionally (`wronggpu-rr-probe.ini` with `capture=1` next to the
game exe) it also captures every RR input of a few evaluates to files for the offline runner (CAPTURE.md).

| Path | Content |
|---|---|
| `include/rr_proxy_abi.h` | Our declarations of the structures of 60.6 (offsets and sizes checked with `static_assert`), `WgResultCode` and the frame token of A10.1, result codes 60.5, feature ids 60.4, buffer types 62.2 |
| `include/wg_rr_adapter.h` | The owned proxy <-> runtime adapter (v1): creation decisions, a fully decided frame (roles after 62.2 routing, render size, ms, reset, parity, counter, pj, CP, settings), result (jitter consumption, camera fallback) |
| `include/wg_rr_settings_contract.h` | The 60.9 contract table and setting entries (40 / 48 bytes, contract version 1) |
| `src/proxy_core.*` | All game-interface logic of the release build: pre-init 23 / no-device 5, forwarding of other features, 60.8 answers, one viewport (61.2), options and creation decisions (61.3), evaluate contract and rejections (61.4, 62.3, 62.5), result-code precedence, tag storage and precedence (local, frame-scoped, viewport), per-token copies (60.7), lifetime-0 copies (62.1), settings snapshot per evaluate / allocate |
| `src/answers.*` | The 60.8 / 61.5 answers both builds share (`<requirements>`, `<version>`, `<rr-optimal>` per Streamline variant) |
| `src/lifecycle.h` | Host bookkeeping of page 53 (par, fc, pj, CP, pending reset generations, render size last built), ported from `clean/rr_f/reference/lifecycle.py` |
| `src/optimal.*` | 61.5 WrongGPU's optimal-settings rule and the 1/3..1 window |
| `src/formats.*` | 62.5 format classes over the public DXGI format numbers |
| `src/settings.*` | 60.9 settings, file persistence (`wronggpu-rr.ini` next to the proxy), contract table |
| `src/proxy_dll.cpp` | File-replacement transport: lazy loading of the renamed original (`<own file stem>.wg-original.dll`) and of `wronggpu_rr_runtime.dll`, role exports, feature functions, `WrongGPU_GetSettingsContract`, log `wronggpu-rr.log`; with `WG_RR_PROBE` the roles go to the probe instead |
| `src/hook_dll.cpp` | Hook-mode transport (HOOK-MODE.md): OptiScaler ASI plugin (`InitializeASI`) that finds the loaded game interface, gates on its version resource, and redirects the RR-role IAT slots of its importers to the same core; handlers forward non-RR traffic to the genuine export (keeping OptiScaler's inline patch in the chain). Runtime, ini and log next to the game exe. Reuses proxy_core/probe/answers/settings unchanged |
| `src/ngx_rr.*` | Release build only: the NGX RR provider for games that call NGX directly (HOOK-MODE.md, "The NGX route"; spec page 66): create / evaluate / release from the host's plugin interface 4 (`include/wg_host_menu_abi.h`, `WgHostNgxRr`), refusals, the same runtime and settings as the game-interface route, the host's command-list bracket, fail-safe |
| `src/probe.*`, `src/probe_log.*` | Probe build only: the probe's role logic (platform-free, GPU behind `ProbeGpu`) and its log rules |
| `src/probe_gpu_d3d12.*`, `src/probe_upscale.hlsl` | Probe build only: the probe's own D3D12 compute passes (bilinear upscale, normal patch mean, lifetime-0 copies), recorded on the game's command list; view-format choice for 62.5 classes (typeless, sRGB); capture copies into readback memory with a completion marker and the file writer next to the game exe (CAPTURE.md) |
| `tools/rr_capture_replay.py`, `CAPTURE.md` | Offline runner: a probe capture through WrongGPU's RR runtime (test host of clean/rr-f-rt-int) with the release proxy's routing, PNGs of input and RR output (CAPTURE.md) |
| `tools/gen_forwarding.py` | Generator: binding table + A9.3 export list + names files -> `.def` (names, ordinals, own export above them), ml64 stubs, C++ table, VERSIONINFO resource script, manifest; `--probe` (all API roles typed and logged, graphics stubs noted), `--hook` (hook-mode plugin: `InitializeASI` + contract `.def`, `wg_binding.inc` with `WG_BINDING_HOOK`, the interface version required and the roles redirected, WrongGPU's own VERSIONINFO; no stubs), `--output-name`; refuses a list whose ordinals are not the ASCII rank, wrong counts or groups, an API export without a role, missing feature-function names |
| `tools/probe_log_to_profile.py` | Probe log -> game profile JSON (`tests/data/<game>-<version>-traffic.json`): init sessions, setup calls with results, token pattern, constants and their changes, tags (type, native format, extent, declared state, lifetime), options, evaluate parameters, call order, jitter / camera and normal samples, totals, timing. Function names are never copied (RR names mapped to roles through `binding/rr-functions.json`) |
| `tools/build.py` | Builds, per version, the file-replacement proxy (release + probe) and the hook-mode plugin (release + probe), and runs every test (MSVC, ml64, fxc, lib from Visual Studio / the Windows SDK; no game; the probe and hook-probe tests run on WARP) |
| `asi/wg_rr_controls.h` | WrongGPU.asi side: find the proxy among loaded modules, hold a reference, build the view (controls / "WrongGPU RR not active" / nothing), apply changes through the set functions, `ControlsSink` drawing seam (open item 23) |
| `binding/` | `rr-f-2.7.json`, `rr-f-2.14.json` (tables: roles A9.4, magic A9.1, GUIDs A9.6, binding by name, output names, file_version of the VERSIONINFO), `rr-f-sl-interposer-exports.json` (crossed spec data A9.3), `rr-functions.json` (the three RR feature-function names of A9.5, by role), `interposer.json` (the game's interposer file name), `game-imports.json` (A9.2). The four names live only here (leak-scan rule, owner OK 2026-10-20); code refers to them by role |
| `tests/` | `test_proxy_core.cpp`, `test_probe_core.cpp`, `test_proxy_dll.cpp`, `test_probe_dll.cpp`, `test_hook_dll.cpp` (the fake game imports the interface by name and the ASI patches its IAT; HOOK-MODE.md), `test_replay_profile.cpp` (a game profile from `tests/data/` replayed into the release logic with the fake runtime; prints the runtime gaps), `test_ngx_rr.cpp` with `mock_ngx_params.h` (the NGX RR provider against a mock parameter block and the fake runtime; four captured frames, `data/cp2077-2.14-rr-frames.json`, through both routes), `mini_json.h`, `data/cp2077-2.7.1-traffic.json` (game A, probe run 1), `test_gen_forwarding.py`, `test_optimal.py`, fakes (`gen_fake_original.py` also models OptiScaler's inline patch on the interface init export), `dll_diag.h` |
| `HOOK-MODE.md` | The hook-mode delivery: why (in-game signature check), design, logs, install for Cyberpunk |
| `QUESTIONS-60-63.md` | Provisional readings and open questions |

## Build and test

```
python clean/rr_f/proxy/tools/build.py            # both builds, all tests; output under <build>\rr-f-proxy\out
python clean/rr_f/proxy/tools/build.py --proxy-name interposer --probe-name interposer   # built under the game's file name
python clean/rr_f/proxy/tools/build.py --binding clean/rr_f/proxy/binding/rr-f-2.14.json [--probe] [--hook]   # one build, no tests
```

Output per version `2.7` / `2.14`: `release/wg_rr_proxy.dll`, `probe/wg_rr_proxy_probe.dll` and `install.json`
(file names, SHA-256, the steps of the install tool). The release DLL is checked not to contain the probe.

Results 2026-10-20 (all exact; there are no tolerance checks in this track):

| Test | Checks |
|---|---|
| `test_gen_forwarding.py` + `test_optimal.py` | 20 unittest cases (real A9 tables: counts, groups, A9.3 ordinals, tag-for-frame shift, A9.4 routing, every game import exported, refusals; names files: one name per line, the names nowhere else in `clean/`; probe generation; output names; VERSIONINFO script and the table's file_version); the optimal rule compared with an exact-rational oracle, 0 mismatches |
| `test_proxy_core.exe` | 1015 checks, 0 failures |
| `test_replay_profile.exe` `data/cp2077-2.7.1-traffic.json` | 1,782,259 checks, 0 failures: game A's probe-run-1 traffic (setup calls, 3,654 loading frame indices, 4,499 RR frames, shutdown) replayed as file replacement and in hook mode; every result as the game got it, every evaluate 0, the frame handed to the runtime checked per evaluate (results/2026-10-20-probe1-cp2077.md) |
| `test_ngx_rr.exe` `data/cp2077-2.14-rr-frames.json` (2026-11-04) | 334 checks, 0 failures: create keys, refusals, evaluate keys, dynamic resolution, rejections, fail-safe, release and handles, the command-list bracket (also closed after a fault), int / uint keys from a host that does not convert, and four captured frames through both routes (same creation, same frames) |
| `test_probe_core.exe` | 381 checks, 0 failures (log rules, claims, 61.5 rule identical to the release answer, forwarding, evaluate routing and errors, logged fields, normal samples, view formats) |
| `test_proxy_dll.exe`, 2.7.x release, with / without runtime | 893 / 882 checks, 0 failures (with version resource and load diagnostics) |
| `test_proxy_dll.exe`, 2.14.1 release, with / without runtime | 898 / 887 checks, 0 failures |
| `test_probe_dll.exe`, 2.7.x probe (WARP, 620 frames) | 1633 checks, 0 failures (with version resource and load diagnostics) |
| `test_probe_dll.exe`, 2.14.1 probe (WARP, 620 frames) | 1634 checks, 0 failures |

`test_proxy_dll` loads the release proxy staged as the install tool places it (under the game's interposer file
name, next to the renamed fake original with the real export names, and the fake runtime). It checks the export
table, every name each game of that version imports (A9.2), forwarding of every stub export with integer, float
and stack arguments, the A9.1 version values, `<get-function>(1001)` with the real names (other names, prefixes and
other cases: 25), the A10.1 token layout, the RR path through the role exports and feature functions into the fake
runtime, other features forwarded, and WrongGPU.asi's side of the contract.

`test_probe_dll` drives the probe like a game on WARP (wrapped device and command list, 620 frames): every RR
evaluate returns 0 and the output rectangle equals a CPU bilinear reference exactly (binary16-exact values); the
output outside the rectangle and every other tagged input are unchanged afterwards; a lifetime-0 colour input is
copied on its tagging call's list; an integer output format is refused with 25 and left untouched; another feature,
a graphics entry point and the interface upgrade are forwarded; and the log holds every field below, in full for
frames 1..600 and changes only afterwards.

## Version resource and load diagnostics (both builds)

Every built proxy carries a VERSIONINFO resource (`wg_version.rc`, generated by `gen_forwarding.py` from the
table, compiled with `rc` and read back by `build.py` after the link). Other mods in the game look up the
interposer module and read its file version, so FILEVERSION / PRODUCTVERSION and the FileVersion / ProductVersion
strings are the game interposer version of the build, the numbers in the table's `file_version` (commented in
the table as the version number other mods read from the interposer file; the generator requires the `api_version`
plus `.0`): `2.7.1.0` for the 2.7 build, `2.14.1.0` for the 2.14 build. The strings are WrongGPU's own: CompanyName `WrongGPU`, ProductName
`WrongGPU RR proxy`, FileDescription `WrongGPU ray reconstruction proxy (release)` / `(probe)`, InternalName
`wg_rr_proxy`, OriginalFilename the game's interposer file name (from `binding/interposer.json`). The install
tool's double-install guard ("WrongGPU" in ProductName or FileDescription) sees both.

Load diagnostics (release: `wronggpu-rr.log`; probe: the probe log). `DLL_PROCESS_ATTACH` only records into a
fixed buffer (no library loads, no threads, no file writes under the loader lock): process id, own module path, and
whether the renamed original is present. The first call of any kind (forwarded or answered) writes those lines
(`load: process <pid> attached <path>`, `load: original <path> present: yes|no`, `load: first call <name>`) and
`load: init export patched by another module: yes|no`: the first 16 bytes of the proxy's own init export in
memory against the on-disk image (base relocations applied) and its export-table entry; when patched, the bytes,
and for a simple jump (rel8 / rel32, `jmp [rip+disp32]`, `mov rax|r11, imm64; jmp`, through up to 4 relay thunks
outside any module) the target and the module holding it. Every `<init>` entry is logged before it is forwarded
(`init entered: thread, preferences pointer, sdk_version; init export patched by another module: yes|no`), so an
init that never reaches the proxy, or one that a hook swallows, is visible.

## The probe log (`wronggpu-rr-probe.log`; the previous run's is kept as `.prev.log`)

Names and numbers only, no image data; frames counted at `<new-frame-token>`; frames 0..600 in full, then a record
only when it changes; 20 MB cap. Per record: `f=<frame> <key>: <values> ; <seq, thread, frame index, jitter>`.

- every export call: sequence number, thread id, role, feature id, result, frame index (token table slot 0); the
  first call of each role (`first <role>`: whether / when, and which RR call came first); names requested through
  `<get-function>` for every feature; graphics / Vulkan entry points by name; a call summary at `<shutdown>`
- `<init>`: SDK version value and magic, preference flags (bit 0, bit 7), feature list, render API, engine
- every chained structure: GUID (A9.6 name when known), structure version, the next-chain GUIDs
- per frame constants: camera motion included, vectors jittered, depth inverted, 3D vectors, dilated,
  orthographic, reset, motion-vector scale
- per tag (viewport, frame-scoped, local): buffer type, lifetime, tag format, resource size, extent, declared
  state, and the resource's own description (format, size, samples, flags); lifetime-0 copies
- per RR evaluate: render size, options' output size, output rectangle, mode, normal / roughness mode and the
  normal / roughness formats; the upscale's result (or the documented error and why)
- every `<rr-options>` (all fields, structure version, the release build's answer) and `<rr-optimal>` (in / out
  sizes); the colour-input extent of the first two evaluates after every `<rr-options>`, `<allocate>` or resize
- per frame: jitter, camera-view-to-clip [2][0] and [2][1] and the previous frame's pair and jitter
- 30 normal patch samples (RR evaluates 60, 80, ..., 640): mean of the 16 x 16 centre texels as stored, with the
  options' world-to-camera-view matrix
- the 16 bytes at frame-token +16 of the first token and whether they are the A9.6 frame-token GUID

## Install tool (what it must do)

Per game, in the directory of the game's interposer library (name in `binding/interposer.json`, also in
`install.json`): rename the game's file to `<stem>.wg-original.dll` (keep it), then copy the release proxy (or the
probe) of the game's Streamline version (2.7 for game A, 2.14 for game B, A9.1) there under the game's file name.
Release only: `wronggpu_rr_runtime.dll` and the model next to it once they exist. Uninstall: delete the copied
proxy and rename `<stem>.wg-original.dll` back. After a probe run, collect `wronggpu-rr-probe.log` from that
directory.

## Integration still to do

- WrongGPU's runtime implements `WrongGPU_RrRuntimeV1` (`include/wg_rr_adapter.h`) in `wronggpu_rr_runtime.dll`
  next to the proxy: preset-F passes, page 53 images, resource-state transitions (item 16), lifetime-0 copies.
  Until then the proxy reports RR unsupported (7) and forwards everything else.
- The tag-extension GUID (Q-B6) stays open (game A does not use the extension).
- Game A (probe run 1, results/2026-10-20-probe1-cp2077.md): the runtime must accept image size 0 x 0 from the
  adapter (Q-R3; the game fills no size), run the extended resolve with type 39 as the layer on every frame, and
  handle RGBA16F / RGBA8 unorm / R32 typeless / R16F inputs declared UNORDERED_ACCESS, packing 1, depth inverted 1,
  vectors not jittered; the release logic itself accepts all 4,499 evaluates of the run (`test_replay_profile`).
- WrongGPU.asi (a separate front-end branch) includes `asi/wg_rr_controls.h` and implements
  `ControlsSink` on OptiScaler's overlay once item 23 names the extension point.
- The installer (`clean/installer`) takes over the steps above.
