# RR proxy, hook mode (`WrongGPU-RR.asi` / `WrongGPU-RR-probe.asi`)

A second delivery of the same RR proxy. It does **not** replace the game's interface file. It is an OptiScaler ASI
plugin that redirects, in the running process, the import-address-table (IAT) slots of the modules that import the
game's interface DLL, for the functions the RR path needs, to WrongGPU's handlers. The genuine interface file stays
in place and signed; the game loads and initialises it normally.

The proxy **core is reused unchanged**: `src/proxy_core.*` (release) and `src/probe.*` (probe), the shared
`src/answers.*`, the 60.9 settings contract, and the runtime adapter (`include/wg_rr_adapter.h`). Only the transport
differs. The file-replacement proxy's transport is `src/proxy_dll.cpp` (loads a renamed original, exports a full
forwarding table). The hook transport is `src/hook_dll.cpp` (no forwarding table; patches IATs instead).

## Why hook mode

From two in-game probe runs of the file-replacement proxy on Cyberpunk 2077:

- The game verifies the digital signature of the interface DLL (the file named in `binding/interposer.json`)
  before its `<init>`. An unsigned replacement is
  never initialised (no `<init>` reaches it; forwarded calls return 21) and the game turns RR off. We do **not**
  replace that file and do **not** defeat the check.
- With the genuine file in place, the interface initialises and loads its RR plugin normally.
- OptiScaler (loaded as `dxgi.dll`) loads `*.asi` files from `bin\x64\plugins\` about 1 s **before** the game's
  first interface call and before `<init>`. OptiScaler also inline-patches the interface's exported init function
  (a `rel32 jmp` into its own code) and reads the module's version resource.
- The game links the interface by import table (2.7 binding: 19 imports; RR functions are fetched by name through
  `<get-function>`).

So a plugin that OptiScaler already loads, early, can redirect the IAT without touching the interface file or its
signature.

## Design

1. **Entry.** The plugin exports `InitializeASI` (OptiScaler's ASI convention, the same entry `WrongGPU.asi` uses)
   and `WrongGPU_GetSettingsContract` (60.9). `DllMain` does only fixed-buffer bookkeeping under the loader lock
   (no loads, no threads, no file writes). All hooking runs from `InitializeASI`. The plugin does not depend on
   `WrongGPU.asi` being present.

2. **Find the interface.** `InitializeASI` looks up the already-loaded interface module by the file name in
   `binding/interposer.json`. It is loaded via the game's import table before any plugin, so it is normally present;
   the log records that it was found already loaded. If it is **not** loaded, the plugin checks the game exe's
   **delay-load** import table (some games, e.g. Spider-Man 2, link the interface that way: the linker's delay helper
   loads it on the first call). If the exe delay-loads the interface, the plugin loads the game's own interface file
   from the exe's folder (where the delay helper's search starts; the genuine, signed file, unchanged on disk) and
   patches the RR-role slots of the exe's delay-load IAT (step 4); the other slots keep their delay thunks and bind
   to the genuine interface as usual. Otherwise it logs and patches nothing (it never installs library-load hooks
   and never crashes the game). Games that load the interface at run time without any import entry (Unreal Engine's
   Streamline plugin, RE Engine) are not covered.

3. **Version gate.** The plugin reads the interface module's version resource and compares `major.minor` with the
   build's binding version (`kWgInterposerFileVersion`, from the table's `file_version`: 2.7.x or 2.14.x). On a
   mismatch it logs and does nothing.

4. **Redirect.** For every loaded module that imports the interface DLL (at minimum the game exe), the plugin walks
   the import descriptor for that DLL and, for each imported **name** that is one of the RR roles this build handles,
   overwrites the module's IAT slot with WrongGPU's handler. Imports by ordinal are ignored (the games import by
   name, A9.2). Patching is thread-safe: `VirtualProtect` to writable, an atomic pointer store, protection restored.
   The interface module itself is never patched; the plugin and the interface module are skipped during enumeration.
   - The roles redirected are exactly the ones the core answers or records: the release build's intercepted roles
     (init, shutdown, is-supported/-loaded, set-loaded, requirements, version, get-function, allocate, free,
     set-device, tag, tag-for-frame on 2.14, constants, new-frame-token, evaluate). `set-vulkan-info`,
     `upgrade-interface` and `native-interface` are **not** redirected by the release build, so those calls reach the
     interface unchanged. The probe build redirects every API role so it can log each one.
   - Handlers forward non-RR traffic by calling through to the genuine interface export (resolved by name). That is
     the same address the IAT held before patching, so **OptiScaler's own inline patch on the export stays in the
     chain**. `<get-function>` hands back WrongGPU's RR functions for the RR names and forwards every other name.

5. **`<init>` already called.** OptiScaler loads the plugin ~1 s before the game's first interface call, so in
   practice the hook is installed before `<init>`. Should the interface's `<init>` have already run when we hook, the
   core must still work: it has no preference bits from that `<init>`. The core is marked initialised with the safe
   default — **preference bit 0 taken as 0**, i.e. assume the game does **not** restore its command-list state after
   an evaluate — and RR taken as requested (lifetime-0 inputs are copied). The probe then avoids binding its own
   descriptor heap / root signature / pipeline on the game's list (it would clobber state the game will not restore):
   those passes are skipped and logged, while the evaluate still returns 0 and lifetime-0 copies (barriers and copies
   only) still run. This matches `probe_gpu_d3d12.cpp`, which already depends on the game restoring its state when
   bit 0 is set. When the game's `<init>` is instead seen through the hook (bit 0 known), the passes run as in the
   file-replacement probe. `adopt_unseen_init()` on `RrProxy`/`RrProbe` performs this; it is a no-op once a real
   `<init>` has been observed.

6. **Logs.** Release writes `wronggpu-rr.log`, probe writes `wronggpu-rr-probe.log`, both **next to the game exe**
   (`bin\x64`, the same place the file-replacement proxy used), not the plugins folder. Same format as the file
   proxy. Hook-mode lines: the module found and whether it was already loaded, its version, each IAT slot patched
   (function + module), the count redirected, anything skipped, and `init unseen` when `<init>` ran before the hook.
   `wronggpu-rr.log` is appended to across game starts; past 8 MiB it is moved to `wronggpu-rr.log.old` (replacing an
   older one) and a new file starts with a line saying so.

7. **Settings / runtime.** The release hook loads `wronggpu_runtime.dll` (since 0.6.0 Model M and RR in one DLL) next to the game exe (`bin\x64`) and
   keeps `wronggpu-rr.ini` there too — the same locations as the file proxy. Until the runtime exists, RR reports
   unsupported and non-RR traffic is unaffected.

## SetupAPI identity shim (Spider-Man 2)

`src/setupapi_identity.{h,cpp}`, started from `InitializeASI` before the game-interface hook. Marvel's Spider-Man 2 reads
the display adapter's PCI identity through SetupAPI as well as DXGI; with the host's GPU-name spoof on, the two differ.

- **When:** only in `Spider-Man2.exe` with `wronggpu-rr.ini` `setupapi_identity=auto` (default; `off` keeps it out).
  The two import slots are redirected at `InitializeASI` (start line: `N import slot(s) of Spider-Man2.exe redirected
  (...); decision deferred to the first call`); DXGI is not readable that early (attended test SH1). The decision is
  made at the game's first call of either function: active when the first hardware adapter the game sees through DXGI
  (the `dxgi.dll` loaded in the process, a host's wrapper included) reports another vendor / device pair than the
  kernel driver's identity of that adapter (D3DKMT by LUID, as the runtime reads it). While DXGI is not readable (the
  device-list query may run before the game creates its device), the host's own spoof settings are used when they turn
  the spoof on: `OptiScaler.ini` next to the exe, `[Spoofing]` `Dxgi=true` and a numeric `SpoofedDeviceId` (hex
  `0x....` or decimal), with vendor 10de (the board the host's spoof reports); the real pair then comes from the first
  hardware adapter the kernel lists. DXGI is preferred whenever readable. With neither, the decision is tried again at
  the next calls, up to 8, then taken as inactive. Once made it is cached and logged once (`setupapi identity: active:
  ...` / `inactive: <why>`); calls before it pass through unchanged.
- **What:** the game exe's import (and delay-load) slots of `setupapi.dll` for `SetupDiGetDeviceInstanceIdW` and
  `SetupDiGetDeviceRegistryPropertyW` are redirected, as for the game interface (no code patched). For a Display-class
  device whose instance ID starts with `PCI\VEN_<real>&DEV_<real>&SUBSYS_<8 hex>&REV_<2 hex>`, the instance ID and every
  hardware-ID / compatible-ID entry that starts with `PCI\VEN_<real>&DEV_<real>` get the spoofed VEN / DEV digits
  (SUBSYS, REV, the suffix and all other entries stay). Same length: the buffer, the sizes, the return value and the
  last error are the real call's. A failed call (buffer too small), other properties (the driver key: the game still
  reads the real DriverVersion), other classes and devices are passed through untouched. One log line per rewritten
  call (function, old -> new), up to 500, then one line saying the rest is not logged.
- **Tests:** `tests/test_setupapi_identity.cpp` (synthetic IDs through fake SetupAPI functions, the decision, and the
  redirect on the test exe's own imports with read-only SetupAPI queries of the PC's Display-class devices).

## Another Ray Reconstruction plugin (0.6.1)

A player can end up with two RR plugins in one game (an integrator's installer may put this plugin next to another
one; the launcher is not involved then). Default: **when another RR plugin is positively found, WrongGPU RR stays off
and says so in one line**; Model M is not touched. The guard is generic and read-only (`src/rr_guard.{h,cpp}`, wired in
`src/hook_dll.cpp`): it looks at where the game's RR calls lead, never at another plugin's files or internals. Release
and probe builds behave the same.

**What counts as a finding.** Only a positive sign stops the plugin: an address that ends in **another named module**
(directly, or through decoded hook jumps), or in **a second copy of WrongGPU RR**. Anything the scan cannot explain is
**not** a sign of another plugin: it is logged once as inconclusive (see "Log lines") and WrongGPU RR **stays active**.

**Signals, most reliable first** (each address is classified: this module, the game interface, the host = the module
that called `InitializeASI` and the module holding the host's API table, a copy of WrongGPU RR, another WrongGPU module,
the exe, code outside every module, another module; hook jumps are followed out of the interface and out of anonymous
memory, at most 4 of them per address; a fifth is "too many hops", inconclusive):
1. **A second copy of WrongGPU RR** (`S0`): a process-wide marker (`Local\WrongGPU.rr-active.v1.<pid>`, holding the
   active copy's path) taken at `InitializeASI`; a copy that finds it taken starts nothing (no hook, routes, NGX
   provider, menu box, SetupAPI shim). A slot or export that leads into a module counting as WrongGPU RR (below) is the
   same case.
2. **Import slots** (`S1`): before anything is patched, every loaded module's import slots of the game interface for
   the roles this build would patch (and the exe's delay-load slots) are read. A slot that leads into another named
   module is a finding; a slot that leads into the interface, the host, another WrongGPU module (or, for a delay slot
   not bound yet, the exe's own thunk) is fine; any other slot is inconclusive.
3. **The RR exports' first bytes** (`S2`): the exports of is-supported, get-function, allocate and evaluate are read:
   a jump out of the interface that ends in another named module is a finding (`jmp rel32/rel8`, `jmp [rip+disp32]`,
   `mov rax|r11, imm64; jmp reg`; a `jmp [rip]` through the interface's own import table is its own code). The init
   export is not read: the host patches it itself.
4. **Host routes** (plugin interface 3): `S2` on the module the host hooked (the host's trampolines, code outside every
   module, are accepted there), then each route's next function (the host's `next`) after registering: another
   plugin's route under ours leads to that plugin. Our routes cannot be taken back: they pass every call on.
5. **The NGX route** (plugin interface 4): the host keeps one RR provider; a refused registration means another
   provider holds it, **for the NGX route only** (the game-interface route goes by its own signals).
6. **Later plugins** (`S4`): a plugin that loads after ours is found at `OptiPluginInit` and at the game's first call
   into any of our handlers (decided once: concurrent first calls wait for the one decision): a slot we patched that
   no longer holds our handler, or (while active) an RR export that now jumps away. Our handlers then pass every call
   on unchanged (to the genuine export, which carries any later patch), and the NGX provider refuses creates.

**Load orders.** Ours second: found at load, nothing patched, the core and the runtime are not loaded, no NGX provider.
Ours first: found at `OptiPluginInit` or the game's first call; pass-through from then on (the runtime is loaded but no
RR instance exists). Both late (run-time interface): the host-route checks above. The SetupAPI identity shim (not part
of the RR path) runs as configured while WrongGPU RR stays off because of another plugin; a second copy starts nothing.

**Never a crash, and a positive finding only.** Every read of memory we do not own (slot values, export bytes, jump
cells, the host's next pointers, the module headers the scan walks) goes through `rr_guard`'s safe read: `VirtualQuery`
for committed, readable, non-guard pages (execute-only pages count as readable: code is read from them), then a
structured-exception-guarded copy (a cached region is re-checked by the exception guard). A failed read is **unknown**.
Unknown is not a finding. Only the positive findings above stop the plugin. These are **inconclusive**: they are
counted, one diagnostic line names the first of them, and WrongGPU RR stays active:
- memory that cannot be read (an import slot, an export, a jump cell, a PAGE_NOACCESS or guard page, a null pointer);
- a jump chain longer than 4 hops;
- code outside every module that does not jump on (a game's copy protection or a hooking library's trampoline may sit
  there);
- a slot or export that leads into the exe's own code (a delay-load helper's thunk, or a game's own wrapper).

**Fail-open cases (known).** The guard does not see these, so WrongGPU RR stays active next to such a plugin: a plugin
hooks after the game's first call (the late check runs at `OptiPluginInit` and once, at the first call, and never
again); inline hooks in forms other than the jump forms above (for example a hook placed deeper than the first bytes of
the export, or through a different instruction sequence); a plugin that hooks the function pointers the game-interface's
get-function export returns. In these cases the other plugin and WrongGPU RR may both answer RR and the log shows no
finding; remove the other plugin.

**False positives, known and accepted.** A mod that is not an RR plugin but hooks the interface's RR functions so that
its code lives in a named module (for example a frame-generation mod working through the same functions) makes WrongGPU
RR stay off too: the line names its module and `rr_other_plugin=take` puts WrongGPU RR back. Not triggers: the game's
own RR (the calls lead into the interface), the host's own init patch, WrongGPU.asi, a second copy of WrongGPU RR (its
own line). The host is the module that called `InitializeASI` and also the module holding the host's API table
(`OptiPluginInit`): with a separate ASI loader the host's own patches look like another plugin at the load check, which
runs before the table is known. When `OptiPluginInit` then shows that the module named at load holds the host's table, a
`the module the load check took for another Ray Reconstruction plugin (...) holds the host's API table` line follows and
the load check runs again with that module as the host (nothing was patched before); the earlier "stays off" line stays
in the log.

**Override** (`wronggpu-rr.ini` next to the game exe): `rr_other_plugin = off` (default; also any unknown value, with a
line) or `take`. `take` with a plugin found at load: WrongGPU RR patches on top, answers the RR calls itself and
continues the other calls of the same functions through what the slot held before (the other plugin), so the other
plugin sees the game's other calls but never an RR create: one RR evaluate per frame. In host-route mode the same
through the host's next functions. `take` cannot be applied to a plugin that loads after ours (we never patch again;
no patch fights), to the NGX route, or to a second copy of WrongGPU RR.

**Counting as WrongGPU.** A module counts as a copy of WrongGPU RR when it exports `WrongGPU_GetSettingsContract`
(the 60.9 contract export), and as another WrongGPU module (accepted, never "another plugin") when it exports
`WrongGpuAsiQuery` (WrongGPU.asi). The launcher can use the same exports to recognise WrongGPU's files.

**Log lines** (`wronggpu-rr.log`, after the time stamp; the probe log adds `f=<frame>`). The launcher's Last run page
matches the **beginnings** of the first three; keep them exactly:
- `another Ray Reconstruction plugin is active; WrongGPU RR stays off (<detail>; nothing patched; rr_other_plugin=take in wronggpu-rr.ini uses WrongGPU's instead)` (found at load);
  with host routes already installed: `...; WrongGPU's handlers pass every call on unchanged; rr_other_plugin=take ...)`;
  found later: `another Ray Reconstruction plugin is active; WrongGPU RR stays off (<detail>; it loaded after WrongGPU: WrongGPU passes every call on unchanged; to use WrongGPU RR, move it aside)`;
  NGX route: `another Ray Reconstruction plugin is active; WrongGPU RR stays off (NGX route only: the host refused WrongGPU's RR provider, another provider holds it; the game-interface route is decided on its own)`.
- `a second copy of WrongGPU RR is loaded (<active copy> is active); this copy (<this copy>) stays off`
- `a plugin loaded after WrongGPU replaced its hooks; rr_other_plugin=take cannot be applied: WrongGPU RR stays off and passes every call on unchanged (<detail>); move it aside to use WrongGPU RR`
- with take, found at load (RR is on): `another Ray Reconstruction plugin is active; rr_other_plugin=take: WrongGPU RR goes in front of it (<detail>; it gets the game's other calls, no RR work)`

`<detail>` names the place and the module file of a positive finding, e.g. `the import slot <name> of Game.exe leads to
Other.asi`, `the export <name> of <interface file> leads to Other.asi (through 2 hook jumps)`. Diagnostics (not
matched): `hook: coexistence check at load: <result> (<n> address(es) read in <t> ms)` once per run (with its measured
cost), `hook: coexistence re-check at <when>: ...`, `NGX: RR provider not registered: WrongGPU RR stays off in this
process (see the line above)`, and the stay-on line `hook: coexistence check <when>: inconclusive for <n> address(es)
(first: <place> could not be read | leads to code outside every module | leads into the exe's own code | leads through
more than 4 hook jumps); no other plugin was positively found, WrongGPU RR stays active` (it starts with `hook: `, never
with a matched beginning above).

**Tests** (CPU only): `tests/test_rr_guard.cpp` (safe reads on PAGE_NOACCESS, reserved, guard and execute-only pages and a
page made unreadable after caching; every jump form; synthetic modules; exactly 4 hops followed and a fifth one too
many; which classifications are findings and which are inconclusive; 32 threads at one decision);
`tests/test_rr_guard_dll.cpp` with our own stand-in other plugin `tests/fake_other_rr.cpp` (modes clean, first, inline,
relay, execonly, anon, noaccess, guardpage, exe, hops, hostsplit, late, late-replace, provider, owncopy, route,
route-other, route-inline; `--take`; 16 threads at the first call): the foreign-module modes switch off (release: the
RR runtime is then not loaded; in clean it is); anon, noaccess,
guardpage, exe and hops log the inconclusive line and stay active; hostsplit has a separate loader DLL call
`InitializeASI` while the real host's DLL holds the API table; owncopy loads four more copies and runs their
`InitializeASI` / `OptiPluginInit` from threads while 16 game threads make the first call, then checks that no line is
torn and that freeing an inactive copy writes no exit summary; build.py repeats owncopy 200 times. Run for every interface
version, release and probe (build.py); the existing hook runs check that no guard line appears.

## Unload

The plugin does **not** unload. OptiScaler keeps ASI plugins loaded for the process lifetime, and a patched IAT
could be entered by another thread at any time, so there is no safe live unhook. The plugin therefore installs its
patches once and leaves them; it never removes them. (The test records the patches it made and could restore them,
but the shipped plugin does not.)

## What stays the same as the file-replacement proxy

The answers (60.8/61.5), the viewport/options/creation decisions (61.2/61.3), the evaluate contract and rejections
(61.4/62.3/62.5), tag storage, per-token copies, the settings snapshot, the probe's GPU passes and log rules, and
the runtime adapter are all the shared core. Switching transport does not change any RR behaviour the game sees.

## Build and test

```
python clean/rr_f/proxy/tools/build.py              # builds all four ASIs (2.7/2.14 x release/probe) and runs every test
python clean/rr_f/proxy/tools/build.py --binding clean/rr_f/proxy/binding/rr-f-2.14.json --hook [--probe]
```

`tests/test_hook_dll.cpp` is the fake game: it statically imports the RR functions **by name** from a stand-in
interface DLL (the interposer file name, the real export names, `tests/gen_fake_original.py`), so the loader fills
its IAT before anything runs. A stand-in "OptiScaler" installs an inline patch on the interface's init export
(`wgtest_fo_set_opti_init`). The test then loads the ASI (as OptiScaler would, after the interface is loaded), calls
`InitializeASI`, and drives a full RR sequence. It checks: the interface is found already loaded; a wrong version
patches nothing (RR stays with the stand-in); on a match the RR path reaches the core and the feature functions,
non-RR traffic reaches the stand-in (and through it the OptiScaler init patch stays in the chain), and only the
RR-role IAT slots were redirected. The release build runs the RR path into a fake runtime; the probe build runs a
short sequence on the software D3D12 adapter (WARP) with every RR evaluate returning 0. The probe core itself is
validated at 620 frames by `tests/test_probe_dll.cpp` (file transport, identical core). The hook-probe test also
runs the capture option (CAPTURE.md) on WARP with the D3D12 debug layer: evaluates 4, 6, 8 captured, files compared
byte for byte with the uploaded inputs, no debug-layer error.

## Install for Cyberpunk 2077 (2.7 interface)

The game keeps its interface DLL (the file named in `binding/interposer.json`) in `...\Cyberpunk 2077\bin\x64`.
OptiScaler is installed there as `dxgi.dll` with
a `plugins\` folder (`LoadAsiPlugins=true`, `Path=auto` in OptiScaler.ini).

1. Copy `WrongGPU-RR.asi` (the 2.7 release build) into `bin\x64\plugins\`.
2. Copy `wronggpu_runtime.dll` (and its model, once it exists) into `bin\x64`, next to the game exe. Keep
   `wronggpu-rr.ini` there too.
3. Leave the game's interface DLL untouched — its genuine, signed file.
4. For a probe run, use `WrongGPU-RR-probe.asi` instead (no runtime needed); collect `wronggpu-rr-probe.log` from
   `bin\x64` afterwards.

Do not combine the hook-mode plugin with the file-replacement proxy for the same game: use one transport or the
other.

## Diagnostic environment knobs (release plugin, wronggpu-rr.ini)

`diag_env_1` .. `diag_env_4` = `WG_RR_<NAME>=<value>`: handed to the runtime as an environment variable of the game process before it is
loaded (its start-up knobs and `WG_RR_DIAG_*` switches), so a diagnostic does not need a system-wide variable. Only `WG_RR_` names are
accepted (others are logged and ignored); a variable already set in the environment wins. Not set: nothing changes.

## The player's RR switch (`rr_enabled`, release plugin and file proxy, every interface version)

The launcher's "Use WrongGPU" row of a game's Ray Reconstruction section writes `rr_enabled = 0 | 1` under
`[wronggpu]` in `<game exe folder>\WrongGPU\wronggpu.ini`. The plugin reads it once, when its core starts
(`src/rr_enabled.h`; first hit wins: that file, then `rr_enabled` or the older `wronggpu_rr` in the plugin's own
`wronggpu-rr.ini`, then on). Off: the RR files stay installed, the runtime is not loaded, `<is-supported>` for RR
answers 7 (adapter not supported) so the game uses its own path, and allocate / evaluate create nothing (7) even for a
game that does not ask first. Non-RR traffic and the menu box are unchanged. The log says
`hook: rr_enabled=0 (<file>): WrongGPU does not take RR ...` and `rr: reported unsupported: WrongGPU RR is switched
off ...`. This is an install switch, separate from the 60.9 live setting "WrongGPU RR" (which never changes
`<is-supported>`). Tests: `test_proxy_core` (`test_rr_off`) and `test_hook_dll --rr-off` / `--rr-off-own` for 2.7, 2.8
and 2.14 (import and delay-load games).

## The NGX route (plugin interface 4, games that call NGX directly)

Unreal Engine's plugin and the Nixxes ports (Hogwarts Legacy, LEGO Batman, Borderlands 4, Spider-Man 2) create ray
reconstruction through the NGX interface, not through the game interface, so the routes above never see it. With the
WrongGPU edition of OptiScaler as the game's NGX layer, `OptiPluginInit` registers WrongGPU's NGX RR provider
(`src/ngx_rr.{h,cpp}`, spec page 66; binary layout `WgHostNgxRr` / `WgHostNgxParams` in `include/wg_host_menu_abi.h`)
when the host's interface version is 4 or newer, the player's switch is on and `wronggpu_runtime.dll` is next to the game
exe. The host then reports RR available in the NGX capability parameters and hands the D3D12 RR feature's create /
evaluate / release to the provider:

- create: annex 66.A create keys -> `WgRrCreate` (`api_minor` 1000 marks the route), `set_device` + `create_instance`
  on the same runtime, rr_mode, model and `wronggpu-rr.ini` as the game-interface route. Refused (the game sees an NGX
  failure and keeps its own path): render / output size missing, denoise mode other than 1, motion vectors at output
  resolution (create flag bit 1 clear), roughness mode other than 0 / 1, render size outside 1/3..1 of the output,
  runtime / model not ready, the live setting "WrongGPU RR" off, or the game-interface route already holding RR.
- evaluate: evaluate keys -> `WgRrFrame` (subrect bases as origins, the render subrect for dynamic resolution, motion
  vector scale taken as given, jitter unchanged, previous camera pair kept by the provider, inputs declared in a
  shader-resource state and the output in the unordered-access state), all work on the game's command list inside the
  call, bracketed with the host's command-list restore (the host brackets each NGX evaluate itself; ours nests inside).
  Missing inputs, unsupported formats or a render size outside the window fail the call; a runtime failure before any
  pass gets the pass-through copy and reports success, as on the other route.
- release: `destroy_instance` (GPU memory freed after the frames in flight).
- One route per game: the NGX route refuses while the game-interface route holds an RR feature, and game-interface RR
  work answers 38 while the NGX route holds one. A fault inside the provider is caught, reported as a failure and turns
  the NGX route off for the run; the provider's bracket on the game's list is closed if the fault left it open (the
  host's begin / end stay paired), and the route guard reads a lock-free count, so it never waits for a lock the
  faulting call may still hold.
- Integer keys are read with the type the provider expects and, when the host has no value of that type, with the
  other integer type (a negative int for an unsigned key counts as not set): the provider does not rely on the host
  converting between them.

Log lines (`wronggpu-rr.log` next to the game exe): `NGX: RR provider registered with the host (plugin interface N) ...`,
`NGX: RR feature 1 created: render WxH -> output WxH, quality q (mode m), flags 0x.. ...`,
`NGX: RR feature 1 first evaluate: render ... ; runtime returned 0 ...`, `NGX: RR feature 1 released after K evaluate(s)
...`; refusals as `NGX: RR feature refused (the game keeps its own path): <reason>`. Tests: `test_ngx_rr` (mock host
parameter block and fake runtime, plus four captured Cyberpunk frames sent through both routes: the runtime receives the
same creation and the same frames; the bracket closed after a fault; int / uint keys from a host that does not convert)
and the NGX part of `test_hook_dll` (registration with a version-5 host, version 3 skipped, `--rr-off` not registered,
create / evaluate / release, route exclusion, fault guard: an access violation inside the runtime's evaluate leaves the
host's begin / end paired and the route guard answering from another thread).

## Game-interface diagnostics (clean/rr-sl-diag)

For a game where RR never starts (the first game on the 2.7 route), `wronggpu-rr.log` says what the game asked and what
it was told, without changing any answer. Each distinct call is logged once as a `diag:` line (same arguments and same
answer: one line; at most 300 such lines per run):

- every support query: for RR the adapter (LUID, the adapter info version, vendor and device from the system's DXGI) and
  the answer, with the reason of a "no" (`<init>` not seen, the player's switch, the runtime not ready with its state,
  the host restore refusal, a Vulkan adapter, an adapter the runtime does not support); for other features the feature
  id and the answer of the game interface;
- every RR feature-function lookup by name, served or not (and why not: before set-device, before `<init>`, an unknown
  name); RR set-loaded / is-loaded / requirements / version with their answers;
- every distinct RR options and optimal-settings call (mode, sizes, answer), the first tag, tag-for-frame and constants
  calls (counts, buffer types, sizes, jitter, vector scale, reset), the first RR allocate and evaluate, one line per
  other feature's evaluate answer;
- a summary at the game interface's shutdown and at process exit: `diag: summary (...): support queries N (RR: yes Y /
  no Z), RR lookups L, RR instances created C, RR evaluates E` (the exit line is written without taking any lock).

Tests: `test_proxy_core` (`test_sl_diag`, on the 2.7 route) and the release `test_hook_dll` runs (the lines and the
shutdown summary in the plugin's log).

Slot watch (clean/rr-sl-diag, second commit): every IAT slot the plugin patched and every other delay-load slot of the
game interface's descriptor are read again 10, 30, 60, 120, 240, 480 and 900 s after the hook and at the host's first
RR create of the NGX route. Off by default (release builds, decided 2026-11-04): `WG_RR_SLOT_CHECKS=on` turns it on
with those times, a comma list of seconds with its own times; unset, empty or `off` = no background thread and no
checks (the log's "watching" line then says so). The once-per-call diagnostic lines and the exit summary stay on. The first check logs every slot,
later ones only the slots that changed: a patched slot no longer holding our handler (`REWRITTEN, now <module>+0x..`),
an unpatched delay-load slot bound by the delay helper (the game called that function). Each check also lists the loaded
modules whose import or delay-import table names the game interface. Read only: nothing is patched again. The summary
at process exit adds the calls that reached each handler, the NGX route's creates and evaluates, and the checks done.
