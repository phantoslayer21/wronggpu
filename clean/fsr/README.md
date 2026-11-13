# WrongGPU FSR front end (spec 20)

Three replacement DLLs, each forwarding to the game's original kept next to it as `<name>.wronggpu-original.dll`:

- **Integration point 1 (FSR 3.1 API):** `amd_fidelityfx_dx12.dll`, described first below.
- **Integration point 2 (FSR 3.0 component DLLs):** `ffx_fsr3upscaler_x64.dll` and `ffx_fsr3_x64.dll`, see
  "FSR 3.0 component DLLs". Cyberpunk 2077 upscales through these (20-T0, `results/2026-10-05-fsr-20-3989fff.md`).

## FSR 3.1 API DLL

A replacement `amd_fidelityfx_dx12.dll` for FSR 3.1 games. It exports the five public FidelityFX API functions
(`ffxCreateContext`, `ffxDestroyContext`, `ffxConfigure`, `ffxQuery`, `ffxDispatch`), runs **upscaler contexts** on
a WrongGPU upscaler, and forwards **every other context type and call** unchanged to AMD's original DLL, which the
installer keeps next to it as `amd_fidelityfx_dx12.wronggpu-original.dll` (spec 30). Every call and its parameters
go to a plain-text log, `wronggpu-fsr.log`, next to the DLL.

```
game ──ffx*──▶ amd_fidelityfx_dx12.dll (WrongGPU)
                 ├─ upscaler context ──▶ WrongGPU upscaler (wronggpu_upscaler.h)
                 │                         ├─ stand-in: D3D12 compute bilinear/Lanczos (now)
                 │                         └─ network runtime DLL, spec 10 (later, branch clean/runtime-core)
                 │     on create/frame failure ──▶ AMD's FSR (spec 10 section 8)
                 └─ everything else (frame generation, swap chain, version queries, ...)
                                       ──▶ amd_fidelityfx_dx12.wronggpu-original.dll (AMD)
```

## Files

| Path | What |
|---|---|
| `include/wronggpu_upscaler.h` | The narrow C interface between the front end and an upscaler (spec 20 "Boundary", spec 10 section 1) |
| `src/frontend.*` | Routing of the five calls: WrongGPU's upscaler contexts, forwarding, fallback to AMD, call log (no Windows code) |
| `src/param_map.*` | FSR parameters → `WgFrame` (conversions in `clean/FORMATS.md`), helper queries (no Windows code) |
| `src/ffx_describe.*` | Text of every FidelityFX structure for the log (no Windows code) |
| `src/config.*` | `wronggpu.ini` parser and defaults, the registry's entries and models as this build offers them (no Windows code) |
| `src/settings_bridge.*` | The data folder, the shared settings block (writer when hosted), the folder sidecar, the OptiScaler.ini poll (`clean/wgshared/include/wg_settings_io.h`) |
| `tests/test_custom_slot.cpp`, `tests/fake_opti_loader.*` | The custom upscaler package end to end on WARP, with our own OptiScaler-style loader (spec 65) |
| `src/dllmain.cpp`, `exports.def`, `version.rc` | The DLL: exports, loading AMD's DLL, choosing the upscaler, log setup |
| `src/standin_d3d12.*`, `src/standin_shader.h` | Stand-in upscaler and its debug views (D3D12 compute, HLSL compiled at run time) |
| `src/log_win.h` | Log file shared by the three DLLs, one tag per DLL (adapted from `imported/0.2.0-infra/src/common/log.h`) |
| `src/win_util.*` | Windows helpers of the three DLLs: loading AMD's originals, settings, upscaler choice, D3D12 queries |
| `src/d3d12_hooks.*` | WrongGPU's own D3D12 vtable hooks (interface version 2): command-list state tracking, the ordered split, the submission trace |
| `src/d3d12_vtbl.h`, `src/d3d12_vtbl_check.c` | Vtable positions of the hooked D3D12 methods; compile-time check against the SDK's C declarations (linked into no DLL) |
| `src/fsr30_types.h` | WrongGPU's mirror of the FSR 3.0 structures |
| `src/fsr30_layout_check.cpp`, `src/ffx_api_layout_check.cpp` | Compile-time checks of WrongGPU's mirrors against AMD's headers (object library `wg_sdk_layout_check`, linked into no DLL) |
| `src/fsr30_hook.*`, `src/fsr30_map.*` | FSR 3.0 forwarders: logging, where the stand-in takes over, FSR 3.0 → FFX API conversion (no Windows code) |
| `src/dll_fsr3upscaler.cpp`, `src/dll_fsr3.cpp`, `exports_*.def`, `version_*.rc` | The two FSR 3.0 DLLs |
| `tests/test_fsr30.cpp` | Unit tests of the FSR 3.0 forwarders without D3D12 (any OS) |
| `tests/fake_fsr30.cpp` | Fake AMD `ffx_fsr3upscaler_x64` / `ffx_fsr3_x64` for the end-to-end tests |
| `tests/test_frontend.cpp` | Unit tests without D3D12: forwarding, mapping, fallback, settings, log (any OS) |
| `tests/test_dll_d3d12.cpp`, `tests/fake_original.*` | End-to-end tests of the built DLL on a real D3D12 device with a fake AMD DLL |
| `tests/test-install.ps1` | Tests of the install script on mock game folders |
| `tests/audit-release.sh` | Checks a MinGW build against the owner's release rule (no third-party code in the DLLs) |
| `tests/run-tests.sh` | Runs all of the above that the machine can run (Linux: host, MinGW, release audit, Wine, PowerShell 7) |
| `install/install-wronggpu-fsr.ps1` | Install / uninstall / status for the spec team (tests 20-T1 to 20-T3) |
| `TESTPLAN-20.md` | The in-game test plan for the spec team |
| `NOTICES.txt` | The release's notices file (MIT licence of the FidelityFX headers), copied next to the DLLs as `wronggpu-notices.txt` |

The vendored AMD headers are in `clean/third_party/fidelityfx-sdk-1.1.4` (FFX API) and
`clean/third_party/fidelityfx-sdk-fsr3-v3.0.4` (FSR 3.0, layout check only), both MIT (see `clean/THIRD_PARTY.md`).
Owner's release rule (top-level README.md): the DLLs contain only WrongGPU's own code. They include only the four
declaration-only FidelityFX API headers; headers with code (`dx12/ffx_api_dx12.h`, the FSR 3.0 host headers) are used
only by the layout checks, which no DLL links. `tests/audit-release.sh` checks this on every MinGW build.

## Building

Windows, Visual Studio 2022, from a fresh clone (static runtime, no extra DLLs):

```
cmake -S clean/fsr -B build/fsr -G "Visual Studio 17 2022" -A x64 -DWG_BUILD_ID=<commit>
cmake --build build/fsr --config Release
ctest --test-dir build/fsr -C Release --output-on-failure
```

Linux, MinGW-w64 cross build (`apt install g++-mingw-w64-x86-64-posix cmake`):

```
clean/fsr/build-mingw.sh            # → build/fsr-mingw/payload/{the three DLLs, notices, install script, test plan, SHA256SUMS.txt}
clean/fsr/tests/run-tests.sh        # host unit test, MinGW build, release audit, Wine tests, install-script test
```

The MinGW build is for development and the Wine tests only: it links GCC's runtime (libstdc++, libgcc) and
winpthreads statically into the DLLs, which the owner's release rule does not allow. Releases and in-game tests use
the MSVC build (QUESTIONS.md, spec 20 item 15).

The Wine tests need Wine 9 with vkd3d (part of Wine), a Vulkan driver (Mesa lavapipe works:
`apt install wine64 mesa-vulkan-drivers xvfb`) and a display or `xvfb-run`.

Each DLL imports only `KERNEL32.dll` and `msvcrt.dll` (MinGW) or `KERNEL32.dll` (MSVC, static CRT). `d3d12.dll` and
`d3dcompiler_47.dll` (both part of Windows) are loaded at run time.

## Settings (`wronggpu.ini` in WrongGPU's data folder)

The player's one WrongGPU settings file (spec 30), shared by every WrongGPU DLL. Where it is (`src/settings_bridge.*`,
the rule of `clean/wgshared/include/wg_settings_io.h`): `<game exe folder>\WrongGPU\wronggpu.ini` (`WG_DATA_DIR`
overrides the folder); when the data folder has none, the old place still works with a log line: next to the DLL
(direct install) or next to the game exe (custom slot). The network runtime is loaded from the same folder (older
installs: next to the FSR DLLs). Written with defaults and comments on the first start: the registry's generated
`[wronggpu]` section (`clean/wgshared/ini/wronggpu.default.ini`: `sr_model`, `rr_mode`, the RR opt-ins, ...) followed by
the `[fsr]` keys below. The registry's `[wronggpu]` keys are read through the shared settings block (block slot >
environment > ini > default; `clean/wgshared/README.md`); the front end reads `sr_model` (the model when no entry of
the host's list chose one), `sr_fallback_to_amd` and `sr_runtime_dll` (when set they decide over `fallback_to_amd` /
`runtime_dll`) and `entry_source`. The `[fsr]` keys are read once per process; keys before any section and in
`[wronggpu]` that are not registry keys are read as before; other sections belong to other WrongGPU components.

| Key | Default | Meaning |
|---|---|---|
| `upscaler` | `standin` | `standin`, `network` (load `runtime_dll`, export `wgGetUpscalerBackend`), or `amd` (forward everything; log only) |
| `runtime_options` | empty | extra `key=value` items for the network runtime (e.g. `timing_log=1;sync=pipelined`), appended after the front end's keys to `WgCreateInfo::options`; separate with `;` and no space before it (`#` or `;` after a space starts a comment) |
| `runtime_dll` | `wronggpu_runtime.dll` | network runtime, a file name in the folder of WrongGPU's FSR DLLs (never searched elsewhere) |
| `model_dir` | empty | folder of the converted model (`model.wgcm`; Model K in `K\`), looked at first; relative paths from the data folder; then `<game exe folder>\WrongGPU\model`, then the runtime's default `%LOCALAPPDATA%\WrongGPU\model` |
| `fallback_to_amd` | `1` | FidelityFX API DLL: on a create or frame failure, give the context to AMD's FSR (spec 10 section 8); `0`: return the error. The FSR 3.0 DLLs always hand a failed context to AMD (AMD's context exists beside WrongGPU's anyway) |
| `view` | `upscale` | stand-in view: `upscale`, `warp`, `warpdiff`, `motion`, `depth`, `jitter` (FORMATS.md, TESTPLAN-20.md) |
| `filter` | `bilinear` | `bilinear` or `lanczos` |
| `jitter_compensation` | `1` | read color at the jittered sample position |
| `warp_weight`, `debug_gain` | `0.9`, `4` | debug-view parameters |
| `jitter_sign_x/y`, `motion_sign_x/y` | `1`, `1` | convention switches for test 20-T2 (jitter: j = +jitterOffset, spec 20 draft 2) |
| `fsr3_route` | `auto` | FSR 3.0 path: where the stand-in takes over: `auto`, `upscaler`, `fsr3`, `off` (FORMATS.md) |
| `fsr3_amd_first` | `auto` | FSR 3.0 path: AMD's upscaler runs first on the frame (frame generation needs its outputs): `auto` (when the dispatch carries them and the `ffx_fsr3` context is not UPSCALING_ONLY), `always`, `never` |
| `log_dispatches` | `1` | `0`: log only the first three dispatches of each context (FSR 3.0: of each function) |
| `d3d12_hooks` | `auto` | WrongGPU's D3D12 hooks (same-frame split): `auto` (when the upscaler sets `WG_BACKEND_NEEDS_SPLIT`), `on` (also at the first upscale dispatch, e.g. with `upscaler = amd` for the submission trace, test 20-T4), `off` |
| `split_test` | `0` | `1`: the stand-in upscales through the command-list split, with a WrongGPU queue signalling resume in place of the network (test 20-T5) |
| `split_timeout_ms` | `2000` | a split list the game does not submit within this time is released from the CPU |
| `log_max_mb` | `512` | above this log size, dispatches are logged once every 600 frames |

## The log

One line per event, `[time process] #call ...`. For every call: the function, the context (`U1(...)` for WrongGPU's
upscaler contexts, `amd:<TYPE>#n` for forwarded ones), every structure of the description chain with all its fields,
and the result with who handled it (`WrongGPU` or `forwarded to AMD`), plus query outputs. For upscale dispatches
also the converted frame (`WrongGPU frame: ...`). Load time: where AMD's DLL was found, its file version, any AMD
exports this DLL does not provide, and the settings in effect.

## Behaviour notes

- **Context routing:** a create whose description chain contains `CREATE_UPSCALE` becomes a WrongGPU context (a
  front-end object returned as the `ffxContext`); everything else is created by AMD and AMD's context value is
  returned unchanged, so later calls on it are forwarded without lookup cost beyond a map check. Calls with a null
  context (global queries/configuration) go to AMD.
- **Non-linear colour** (context flag NON_LINEAR_COLORSPACE, or a frame flagged sRGB/PQ) goes to AMD's FSR: the
  network expects linear colour. Logged as a note.
- **Own contexts** answer `QUERY_GET_PROVIDER_VERSION` with the upscaler's name (WrongGPU never claims to be AMD's
  FSR), answer the upscale helper queries (ratios, render size, jitter phase count/offsets) from AMD's documented
  behaviour, accept `CONFIGURE_UPSCALE_KEYVALUE` (no effect) and treat `GENERATEREACTIVEMASK` as a no-op.
- **Hosted** (the DLL is loaded under another file name by an upscaler mod, e.g. OptiScaler's custom upscaler folder
  or its `FfxDx12SRPath`): the upscaler version list (`QUERY_GET_VERSIONS`) starts with WrongGPU's entries from the
  registry (`kWgEntries`): ids `0x5747000000000001` (Model M) and `0x5747000000000002` (Model K), named
  `3.1.5 WrongGPU (Model M)` and `3.1.5 WrongGPU (Model K)`, then the RR modes `3.1.5 WrongGPU RR exact` / `accurate` /
  `fast` (`0x...0101` .. `0x...0103`); AMD's versions follow. The leading number is the FidelityFX upscaler API level WrongGPU answers to, not a product version: hosts
  read "the FidelityFX version" of the DLL out of the first listed name (`frontend.h`, `kWrongGpuListedNameM`). It
  must stay a 3.1.x number of at least 3.1.4 and below 4.0.0 (`tests/test_frontend.cpp`, `TestHostVersionHandling`).
- **Fallback:** if the upscaler cannot be created (no D3D12 backend description, shader compile failure, runtime DLL
  missing, ...) the game's own create description goes to AMD; if a frame fails, an AMD context is created from the
  saved creation parameters and that frame and all later ones go to AMD. Both are logged as errors.
- **GPU state:** inputs are transitioned to a shader-readable state and the output to UAV, and back afterwards. The
  stand-in sets its own root signature, pipeline and descriptor heap on the game's command list (QUESTIONS.md, spec
  20 item 7). Destroyed upscalers keep their D3D12 objects alive for 5 s, since the API gives no fence to wait on.
- **No NVIDIA anything:** the DLL uses only AMD's public FidelityFX API and D3D12; it does not touch signatures or
  integrity checks; the AMD DLL it forwards to is loaded unmodified.

## FSR 3.0 component DLLs

`ffx_fsr3upscaler_x64.dll` and `ffx_fsr3_x64.dll` export every function of AMD's FSR 3.0.x headers (SDK tag
fsr3-v3.0.4) plus the SDK 1.1 additions, forward each call to AMD's renamed original, and log it with all parameters
and the module that called it (`from Cyberpunk2077.exe`, `from ffx_fsr3_x64.wronggpu-original.dll`, ...). AMD's
`ffx_fsr3` loads `ffx_fsr3upscaler_x64.dll` by name when it is linked against it, so its upscaler calls also pass
through WrongGPU's DLL.

With `upscaler = standin` the stand-in takes over the upscale output (FORMATS.md, "FSR 3.0 component DLLs"):

- FSR 3.0 frame generation reads three resources the upscaler writes (dilated depth, dilated motion vectors,
  reconstructed previous depth). With frame generation on, AMD's upscaler runs first and the stand-in overwrites only
  the output, so frame generation keeps working on AMD's code. With frame generation off (`ffx_fsr3` context created
  with UPSCALING_ONLY, as Cyberpunk 2077 does) AMD's upscaler does not run at all; `ffx_fsr3_x64.dll` passes that fact
  to `ffx_fsr3upscaler_x64.dll` through its export `wgFsr3UpscalerHintUpscalingOnly` while AMD creates the upscaler
  context.
- The upscaler DLL takes over when it sees the dispatch; otherwise (AMD's `ffx_fsr3` with the upscaler built in)
  `ffx_fsr3_x64.dll` does it after AMD's `ffxFsr3ContextDispatchUpscale`.
- The upscaler's context memory belongs to the game and holds AMD's context; WrongGPU keeps its own state beside it,
  keyed by the context address, and creates the stand-in at the first dispatch with the command list's device.
- An AMD DLL that exports `...GetEffectVersion` is an SDK 1.1 build with other structures: WrongGPU then only logs
  pointers and forwards.

## Plugging in the network runtime (spec 10)

The runtime DLL exports `const WgUpscalerBackend* wgGetUpscalerBackend(uint32_t interface_version)` returning
`create`, `dispatch`, `destroy` (and optionally `gpu_memory_bytes`) as declared in `include/wronggpu_upscaler.h`.
The front end asks for interface version 2 first, then 1, and uses the first table that names itself with the version
asked for and has `name`, `create`, `dispatch` and `destroy`. `WgFrame` already holds the network's conventions
(FORMATS.md); the runtime records its D3D12 work into `frame->command_list`, restores the textures' states, and
returns a non-zero `WgResult` on failure so the front end can fall back to AMD (`WG_ERROR_NOT_READY`: the stand-in
for that frame only). Select it with `upscaler = network`.

What the log says (every line in words, no bare codes):
- loading: `WrongGPU runtime <path> (runtime_dll = ...) loaded: file version ...` and `WrongGPU runtime upscaler "<name>":
  interface version 2, flags 0x1 (same-frame split)`; or one `ERROR:` line why not: not found next to the FSR DLLs,
  could not be loaded (Windows error with its text; error 126 for a file that exists means a DLL it needs, such as the
  AMD HIP runtime, is missing), no `wgGetUpscalerBackend` export, or no usable interface (what each version asked
  returned);
- the runtime's own messages: `U<n> upscaler ERROR|warning|info: ...` (FidelityFX API DLL) or `C<n> upscaler ...`
  (FSR 3.0 DLLs);
- failures: `upscaler "<name>" create failed: WG_ERROR_DEVICE (the D3D12 or HIP runtime returned an error)` and the
  like, then where the context goes: `upscaler context goes to AMD FSR: ...` / `now runs on AMD FSR for the rest of its
  life` (FidelityFX API DLL), `(AMD after <name> failed; C<n> stays on AMD)` (FSR 3.0 DLLs);
- `note ...: upscaler not ready this frame (WG_ERROR_NOT_READY): stand-in for this frame`.

The test install script copies `wronggpu_runtime.dll` from the payload (or `-Runtime <file>`) next to the FSR DLLs
and removes it on uninstall.

## OptiScaler 10: the custom upscaler package (everything chosen in OptiScaler's menu)

OptiScaler 10.0.0-pre1 loads custom upscalers from `OptiScaler\custom\<folder>\ffx\<FFX DX12 upscaler file name>` and
shows each folder as an entry of its main DX12 upscaler list (spec 65.1; `Dx12Upscaler=<folder>`). The build makes
that package: `<build>/package/custom/WrongGPU/ffx/amd_fidelityfx_upscaler_dx12.dll` (target `wg_package`, part of
ALL; `cmake --install <build> --prefix <dir>` writes the same under `<dir>/package/`, plus
`package/WrongGPU/wronggpu.default.ini`). The file name is the one the nightly's own FFX upscaler carries (its
OptiScaler folder ships `amd_fidelityfx_upscaler_dx12.dll`; OptiScaler.ini `[Libraries] FfxDx12SRPath` names it); CMake
`WG_FFX_SLOT_FILE` changes it. The DLL is the same front end; its version resource is kept (OptiScaler logs it).

```
<game>\OptiScaler.ini                 [Upscalers] Dx12Upscaler=WrongGPU   [WrongGPU] UpscalerIndex=<n>
<game>\OptiScaler\custom\WrongGPU\ffx\amd_fidelityfx_upscaler_dx12.dll    this front end (+ wronggpu-fsr.log next to it)
<game>\WrongGPU\wronggpu.ini, wronggpu_runtime.dll, ...                   the data folder (DESIGN-i15 section 9)
```

- **Detection:** loaded under a name other than `amd_fidelityfx_dx12.dll` = hosted (as before); a path
  `...\custom\<folder>\ffx\<file>` = the custom slot: `<folder>` is the OptiScaler.ini section it watches and
  `<folder parent's parent>\OptiScaler.ini` the host's ini.
- **Entries = versions:** WrongGPU's version box lists the registry's entries: Model M, Model K, RR exact, RR accurate,
  RR fast (then AMD's versions). An SR entry runs that model and writes `sr_model`; an RR entry runs the SR context with
  the configured model and writes `rr_mode`, which the RR runtime (same process, shared settings block) applies at its
  next frame. Log: `#n the host chose WrongGPU RR fast (0x5747000000000103): settings: rr_mode=fast (menu); the SR
  context runs WrongGPU (Model K)`.
- **Writer:** hosted, this DLL claims the shared settings block as its single writer
  (`WG_SS_FLAG_HOSTED_BY_OPTISCALER`) and persists every change to `wronggpu.ini`, so a start without OptiScaler keeps
  the menu's choice. Not hosted (direct install) it only reads.
- **Persisted choice:** OptiScaler.ini `[WrongGPU] UpscalerIndex=<n>` is polled (file time at most once per second, our
  own reader, read-only); index `n` < the number of WrongGPU entries is applied as if chosen in the menu (also when the
  host never creates an upscaler, e.g. an RR-only session). An index of one of AMD's versions applies nothing.
- **Fallback, one folder per choice** (registry option `entry_source = folders`; for a host build that shows no version
  box for a custom entry): `custom\WrongGPU M\ffx\`, `custom\WrongGPU RR fast\ffx\`, ... each with the same DLL and a
  sidecar `wronggpu-entry.ini` (`entry=0x5747000000000103`). The DLL then lists that one entry and applies it at create
  (log word `folder`); the instances take turns as the same writer (`WG_SS_FLAG_FOLDER_ENTRY`). Folder names:
  `kWgEntries[].folder_label`.
- **When WrongGPU's upscaler cannot run (fix16, after the Cyberpunk test of 2026-10-30):** the game's FidelityFX DLLs
  in OptiScaler's process are OptiScaler's way in: an upscale call through them goes to OptiScaler's selected upscaler,
  this DLL. The old AMD fallback through them (no model in the game's view) re-entered the slot on the same thread at
  every dispatch, with a new context each time (37 creates, 0 destroys, 36 levels deep in 3 s), until the game froze.
  Now, in the custom slot (`Config::customSlot`): (1) AMD's FSR (fallback, list entries, comparison) only from an AMD
  DLL placed next to this one (`amd_fidelityfx_dx12.wronggpu-original.dll`), never from the game's/host's FidelityFX
  DLLs; (2) without one, a failed create returns `ERROR_RUNTIME_ERROR` and no context (OptiScaler's own upscaler takes
  over); (3) the failure is remembered for the session: a create of the same upscaler with the same options fails at
  once without a new attempt, logged in full once and then one summary line at repeats 1, 2, 3, 10, 100, ...;
  (4) everywhere: a WrongGPU upscaler create that re-enters the front end on the same thread from inside another of
  its calls, and any call on a context the thread is already inside of, are refused (`ThreadCalls` in
  `src/frontend.cpp`), so such a loop cannot start in any host.
- **D3D12 hooks only after a working create:** `Platform::services` gives the backend its split service without
  installing anything; `Platform::activateServices` installs the vtable hooks after that backend's create succeeded
  (also after a slow create, when it finishes well). A failed create, AMD's FSR or a remembered failure never patch a
  vtable (`test_custom_slot` compares the device, queue and list vtables before and after 50 failed creates). If the
  hooks cannot be installed, the upscaler is created again without the service (as before when none was offered).
- **Wrapper objects (clean/fe-wrapped-list):** a command list of another layer's class (its vtable in that layer's module,
  e.g. a game interface's interposer, every call forwarded to an inner list) is never patched: only list class tables in
  the module of the lists the upscaler's device itself creates are. Its calls are hooked once, through the inner list. A
  split is made only on a list whose device is a device an upscaler was created on and the D3D12 runtime's own object;
  otherwise it is refused with one line and the context gets AMD's FSR (else the stand-in). WrongGPU's queue is never
  made on another device (`test_wrapped_list` case F). The front end writes its log to disk right after an upscaler's
  `about to ...` line (HIP start-up, the shared-memory import) and before the first split's submission.
- **Frames the upscaler does not take:** `WG_ERROR_UNSUPPORTED` from the upscaler's dispatch (the network runtime with
  `hip_switch=off`, its default since 2026-11-04: frames above the render size checked at create) hands the context to
  AMD's FSR at once;
  without AMD's FSR the stand-in upscales such frames.
- **Model location:** the front end looks for the converted model (`model.wgcm`, Model K in `K\`) in `[fsr] model_dir`
  of `wronggpu.ini` (relative: from the data folder), then `<game exe folder>\WrongGPU\model`, and passes the first
  folder that has it to the runtime as `model_dir=`; otherwise the runtime's own default (`%LOCALAPPDATA%\WrongGPU\model`,
  not probed). A `model_dir` in `runtime_options` / `WG_RUNTIME_OPTIONS` decides over all of it. Logged once per choice:
  `model: Model M from <file> (passed to the runtime as model_dir); looked at: ...`.
- **What it never does:** no RR traffic (RR stays on the RR plugin, spec 65.3), nothing written outside its folder and
  the data folder (the AMD-provider search no longer enters OptiScaler's `custom` and `plugins` folders), no hook of
  OptiScaler (`tests/test_custom_slot.cpp` checks every file access of the DLL).
- Tests: `tests/test_custom_slot.cpp` with `tests/fake_opti_loader.*` (our own loader written from spec 65's
  description: full-path load, version query, UpscalerIndex from the folder's section, re-create on change, both
  layouts), on WARP.

## Where a host shows WrongGPU (hosted mode, OptiScaler)

Hosted (`Config::hosted`: another upscaler mod loaded this DLL under a name of its own as its FidelityFX upscaler DLL,
e.g. OptiScaler with `[Libraries] FfxDx12SRPath=<folder>\wronggpu_ffx_dx12.dll`), WrongGPU's interface to the host is
the public FidelityFX API: the host asks `QUERY_GET_VERSIONS` and creates its upscale context with `OVERRIDE_VERSION`
naming one of the ids listed. WrongGPU's entries come first (ids tagged "WG" in the top 16 bits), each name starting
with the API level the host reads (`kWrongGpuListedNameM`, frontend.h); AMD's ids and names follow unchanged
(`TestHosted`, `TestHostVersionHandling`). The list is the same with and without the OptiScaler plugin.

**Retired (i15):** with OptiScaler 10 the custom upscaler folder (above) puts WrongGPU into the main list natively,
so the menu-patching plugin below is no longer built (CMake `WG_BUILD_ASI=OFF` by default; its sources and header-only
unit tests stay for history; spec 65.6: do not install it with 10.0.0-pre1). Older OptiScaler builds kept their main
upscaler list fixed in their own code and showed what the FidelityFX DLL lists only in the
"FFX Upscaler" box. WrongGPU's own OptiScaler plugin (`asi/`, `WrongGPU.asi`) added a "WrongGPU" entry to that main
list on the OptiScaler builds it recognises; there the plugin's entry decides (`TestAsiPluginHooked`): "WrongGPU" runs
WrongGPU whatever the FFX box names, OptiScaler's FSR 3.X(/4) entry runs AMD (the FFX box's AMD version, else AMD's
newest listed one). On any other build the plugin changes nothing and the FFX box decides as before.

## Same-frame execution: the ordered split (interface version 2)

A backend that runs its work outside D3D12 (HIP) sets `WG_BACKEND_NEEDS_SPLIT` and calls
`services->split_command_list` from its dispatch (contract in `include/wronggpu_upscaler.h`; ANSWERS.md spec 20 item
16). The front end then:

1. closes the game's command list at the dispatch point and executes it on its own queue ("W1", same type), behind
   a GPU wait on an order fence, followed by a signal of the backend's `done` fence;
2. resets the list with the allocator it was last reset with and re-applies every binding the game had set (tracked
   by the hooks: descriptor heaps, pipeline state, root signatures and arguments, render targets, viewports,
   scissors, input assembler, stencil reference, blend factor, depth bounds, sample positions, shading rate,
   view-instance mask, predication; bundles merged in);
3. when the game executes that list (hooked `ExecuteCommandLists`, any queue), splits the call there: the lists before
   it, then a signal of the order fence (the first half may run now: the game's earlier work is ahead of it), then a
   wait on the backend's `resume` fence, then the list (the second half) and the rest.

```
game queue Q:   ... lists before L │ signal order=k │ wait resume>=m │ L (second half) │ rest
WrongGPU W1:    wait order>=k │ L (first half) │ signal done=n
backend (HIP):                                   wait done>=n │ network │ signal resume=m
```

The hooks are WrongGPU's own COM vtable patches (`ID3D12CommandQueue::ExecuteCommandLists`, the
`ID3D12GraphicsCommandList` setters, `Reset`, `ClearState`, queries, render passes, `ExecuteBundle`, and
`ID3D12Device::CreateCommandList`), each chained to the entry it replaces; no hook library. One of WrongGPU's DLLs per
process owns them (the first that needs them, found through a named per-process file mapping) and stays loaded. A list
whose `Reset` the hooks have not seen yet (the first frame after installing them) answers `WG_ERROR_NOT_READY`; a list
reset without submission, or not submitted within `split_timeout_ms`, is released from the CPU so nothing waits
forever. Limits: a split inside a render pass or with an open query is refused (`WG_ERROR_UNSUPPORTED`); the render
target views are re-set from the CPU descriptor handles the game passed, so a game that rewrites those descriptors
between `OMSetRenderTargets` and the FSR call would get the new ones; a list the game resets without submitting has its
first half run anyway (right after its release), and that work is not ordered against the game's own fences: in
practice it finishes long before the game reuses the list's allocator (games do that frames later, after their frame
fence), but nothing guarantees it.

`split_test = 1` runs the stand-in through this path (test 20-T5); `d3d12_hooks = on` with `upscaler = amd` installs
the hooks only for the submission trace (20-T4): each FSR list's queue, `ExecuteCommandLists` call, position, and
the lists the game submitted between recording and submitting it.
