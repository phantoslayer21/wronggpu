# WrongGPU OptiScaler plugin (`WrongGPU.asi`)

**RETIRED (round i15, 2026-10-30).** OptiScaler 10.0.0-pre1 shows custom upscalers from `OptiScaler\custom\<folder>\`
in its main list natively (spec 65.1, 65.6), and the SR front end now ships as that custom upscaler
(`clean/fsr/README.md`, "OptiScaler 10: the custom upscaler package"). This plugin is no longer built by default
(CMake `-DWG_BUILD_ASI=ON` still builds it), must not be installed with 10.0.0-pre1 (its anchors target older builds),
and the launcher stops installing it. The sources and their header-only unit tests stay for history.

An OptiScaler ASI plugin that puts WrongGPU into OptiScaler's **main** upscaler dropdown as its own entry, "WrongGPU"
(a build with Model K: "WrongGPU Model M" and "WrongGPU Model K"), in front of OptiScaler's "FSR 3.X" / "FSR 3.X/4"
entry, which then means AMD's upscaler again (FSR 4 where AMD's provider has it). WrongGPU is the default entry; the
player's choice is remembered in `wronggpu-asi.ini`. It is WrongGPU's own code only; it contains no OptiScaler code
(only the plain menu label text it searches for) and uses no third-party hooking library.

Historically recorded offline anchor scans (`tests/test_asi_scan.cpp`): OptiScaler 0.9.4 and 0.9.5-pre4
(file version 0.9.5.4). These were not revalidated for the diagnostic change and do not qualify a v10 host.
Recognition is content-based, not an allowlist of versions. Missing or ambiguous required anchors cause refusal
before patching. With no hook, WrongGPU stays in the FFX box and AMD versions remain selectable there.

What the front end does with the plugin's entry (`src/frontend.cpp`, `TestAsiPluginHooked`): "WrongGPU" runs WrongGPU
whatever the FFX box names; "FSR 3.X(/4)" runs AMD: the AMD version the FFX box names, else AMD's newest listed version.
The FFX box's list is the same with and without the plugin (WrongGPU first).

## How it works

1. OptiScaler loads the plugin and calls its exported `InitializeASI()`. `opti_module.h` finds OptiScaler's module
   in the process, whatever name it was loaded under: first the module `InitializeASI` returns into, then modules
   whose version resource names OptiScaler, then modules loaded as one of its install names (dxgi.dll, winmm.dll,
   version.dll, ..., OptiScaler.dll); never the plugin, the executable or a system DLL. The first candidate step 2
   recognises is used, and wronggpu-asi.log names it and why it was a candidate. (0.5.0 fix: the return address was
   taken inside a lambda, so it named the plugin itself; `tests/test_asi_module.cpp` now checks this in a process.)
2. `opti_resolver.h` finds the menu anchors in that image **by content**, never by file name or fixed offset:
   - `ImGui::Selectable`, as the one call target several string-literal menu entries agree on (needs >= 3 agreeing);
   - `MenuCommon::AddDx12Backends`, as the only menu function that references "FSR 3.X" but no "... w/Dx12" label
     (found through the exception directory / `.pdata`);
   - every `Selectable` call site in that function (the combo items), and `ImGui::BeginCombo` for the preview text.
3. The plugin redirects the `call` at each of those sites to a thunk of its own, through a 12-byte absolute-jump stub
   placed within rel32 reach of the module (`asi_thunk.h`). The thunk, at OptiScaler's "FSR 3.X" entry, draws the two
   WrongGPU items first, then the real entry; on a click it remembers the chosen model and returns "clicked" for
   OptiScaler's own FSR 3.X entry, so **OptiScaler's unchanged code** performs the backend switch. The plugin writes
   none of OptiScaler's data.
4. WrongGPU's front end (`amd_fidelityfx_dx12.dll`) finds the plugin by module and asks it (`WrongGpuAsiQuery`,
   `asi_abi.h`) which model is selected. The FFX list always starts with WrongGPU entries followed by AMD entries;
   when hooked, the front end creates Model M / Model K / AMD according to the plugin selection.

If a required anchor is missing or ambiguous, or a stub cannot be placed, nothing is patched and the plugin reports
"not hooked". Call sites are prepared and revalidated before publication. A failed write rolls back sites whose
bytes still match this writer's explicit store receipts; detected foreign edits are preserved and reported as
rollback failure. The hooked flag is set only after the planned transaction commits. A missing optional preview
remains a partial menu adaptation. This supplies bounded failure rollback, not foreign-thread exclusion or safe
live unloading; a read-check-write race remains possible without a host synchronization contract.
It logs to `wronggpu-asi.log` in its data directory.

## Files

| File | What |
|---|---|
| `opti_resolver.h` | Resolve the menu anchors in a mapped PE image (pure; used live and by the offline test) |
| `asi_thunk.h` | Thunk logic, free of Windows/ImGui so it is unit tested with stand-ins |
| `asi_abi.h` | The front-end <-> plugin query interface (shared with `src/asi_bridge.cpp`) |
| `asi_log.h` | `wronggpu-asi.log` writer |
| `asi_status.h` | CPU-only own diagnostic state and bounded query helper |
| `asi_patch.h` | Bounded call-site transaction, revalidation and ownership-aware rollback |
| `wronggpu_asi.cpp` | The DLL: `InitializeASI`, module identification, resolve, stub, call-site patching, the thunks, `WrongGpuAsiQuery` |
| `exports_asi.def`, `version_asi.rc` | Exports and version resource |

Tested: `tests/test_asi_thunk.cpp` (thunk logic), `tests/test_asi_scan.cpp` (offline anchor scan of an OptiScaler
binary named by `WRONGGPU_OPTISCALER_BIN`), and the front-end bridge in `tests/test_frontend.cpp`. Not tested: the
live in-process patching and menu interaction (needs the game).

## Known limitation (switching while already on FSR 3.X)

OptiScaler recreates its upscaler only when the backend code changes. WrongGPU and AMD's FSR 3.X are both the
"fsr31" backend to OptiScaler, so selecting a WrongGPU entry takes effect when coming **from another backend**
(XeSS / FSR 2.x / DLSS -> WrongGPU), which is a real backend change. Switching between AMD FSR 3.X and WrongGPU, or
between Model M and Model K, does not by itself make OptiScaler recreate the feature, so the change shows only after
the next recreate. Forcing a recreate would require writing OptiScaler's own `State::changeBackend`, which depends on
its internal layout and is therefore version-specific; the plugin does not do this. See the owner report.

## Install

WrongGPU's installer (0.4.5.1 and later) installs it for every game: `plugins\WrongGPU.asi` next to OptiScaler and
`[Plugins] LoadAsiPlugins=true`, `Path=auto` in OptiScaler.ini (in 0.9.5-pre4, `LoadAsiPlugins=auto` reads as false).
The plugin writes `wronggpu-asi.log` and `wronggpu-asi.ini` (the entry last selected) into the OptiScaler folder (the
parent of `plugins`), where WrongGPU's log collection finds them and its uninstaller removes them.

## Own diagnostic status (version 1)

`WrongGpuAsiQueryStatusV1` is a WrongGPU-owned diagnostic export, not an OptiScaler interface. Its C-compatible
header defines a 48-byte fixed-width structure. Set `size = sizeof(WrongGpuAsiStatusV1)` and `version = 1`, and
pass the actual buffer capacity as the second argument. Success returns `WG_ASI_STATUS_OK`; null, insufficient
or inconsistent sizes, and unsupported versions are refused without changing the buffer. Larger buffers are
accepted without writing past the defined structure.

The status distinguishes loaded, initialization not attempted/in progress/completed, legacy hooked, and menu
unavailable/partial/patched. Completed means the attempt returned, including failure. Menu patched describes own
write results only; visible menu correctness remains unqualified. Reason codes distinguish unrecognized host,
stub allocation failure, transaction failure, incomplete preview adaptation and rollback failure. Observations during
initialization or selection changes are not a transactional snapshot.

The selected AMD/M/K value is reported separately from effective backend, which is always unknown: this plugin
has no frontend acknowledgement. RR-ready is always false. Blocker bits explicitly identify missing RR host
interface, missing F stage/converter contracts, unavailable complete F backend, and unqualified host. There is
no F menu item, F-to-M/K fallback, or RR resource transport in this change. The existing `WrongGpuAsiQuery` ABI 1
signature, version, null-output behavior, selection domain and hooked behavior remain unchanged.

`tests/test_asi_status.cpp` tests the original pure diagnostic helper with fake lifecycle outcomes and loads only
the owned DLL to query its uninitialized state and resolve its exports. It never invokes `InitializeASI`, inspects
a foreign module, patches a real host, or uses the GPU. This is CPU ABI/status evidence only.

`tests/test_asi_patch.cpp` injects forward/rollback failures, partial byte stores, changed call sites and rel32
boundaries in owned synthetic memory. It passes 35,378 checks under both Clang and MSVC Release. The status,
patch and thunk tests are registered with CTest; run only `^fsr_asi_(patch|status|thunk)_unit$` for this CPU-only
qualification. These checks do not exercise live host patching or Windows API failure injection.

## Own-module lifetime before publication

After recognizing a host and completing patch preparation, the plugin pins its own address-owning module with
`GetModuleHandleExW(FROM_ADDRESS | PIN)` before invoking any host call-site writer. Pin failure reports reason 8,
releases only unpublished stubs and performs no host writes. A successful pin persists until process termination,
including if subsequent patching is rejected or rolled back; normal unloading is intentionally unavailable then.

The loader/caller must still keep the plugin mapped during entry and preparation before the pin. This does not
pin the host or synchronize executing threads, and it does not establish a complete live-unload or host-lifetime
protocol. Query-only loading does not pin the module and is tested to unload successfully without initialization.
`asi_lifetime.h` and `test_asi_lifetime.cpp` cover ordering/refusal with fake callbacks; native successful pinning
through a real host was not exercised. Four selected CTest targets use
`^fsr_asi_(lifetime|patch|status|thunk)_unit$`.
