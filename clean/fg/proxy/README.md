# Frame generation game adapter (issue #96, F8a groundwork, F8b Win32 transport)

The part of frame generation that faces the game: a versioned adapter contract, the adapter core, the per-game FG
settings and the plugin `WrongGPU-FG.asi`. The core takes frame generation only when every route fact holds; the last
one, `kFgModesQualified`, is open from 0.7.0 (`../fg_limits.h`), after the live qualification in Cyberpunk 2077 with
OptiScaler on an RX 9070 XT. The per-game switch stays `fg=off` by default.

| Path | Content |
|---|---|
| `fg_adapter_abi.h` | the host-neutral contract (plain C, size- and version-tagged): adapter identity, capability, decoded options / tags / constants, the present descriptor and result, the assembled packet and session (field for field the FG runtime contract's `FgFramePacket` / `FgCreateDesc`), the next link of the game interface (`WgFgChainV1`), the host provider entry (`WgFgProviderV1` / `WgFgHostV1`) and the executor hand-off |
| `fg_proxy_core.h/.cpp` | the core: install-off gate, feature-scoped routing, truthful capability, tag-to-packet assembly, single present owner, one diagnostic per transition |
| `fg_settings.h/.cpp` | the per-game keys `fg`, `fg_preset`, `fg_multiplier`, `fg_presenter`: strict parse, defaults, round trip, apply without touching other keys |
| `fg_contract_bridge.h` | conversion to the FG runtime contract (`clean/fg/runtime/fg_contract.h`) with compile-time checks that every shared number matches; used only where that contract is in the tree |
| `tests/test_fg_proxy.cpp` | CPU tests against a fake game interface (super resolution, ray reconstruction, frame generation and the latency features in one session), a fake host and a fake executor; synthetic values only |
| `win32/fg_asi_transport.h/.cpp` | (F8b) the Win32 hook transport: install-off gate, qualified binding and native codec, interface found and version-gated, bounded fail-open scan, import / delay-load slots redirected for the FG roles, every call routed through the core |
| `win32/fg_swapchain_hook.h/.cpp` | (F8b) the DXGI swap-chain seam: a COM-identity-preserving `IDXGISwapChain4` wrapper installed at `CreateSwapChainForHwnd`, one present owner per chain, `IFgPresentCallback` for the presenter |
| `win32/fg_asi_entry.cpp` | (F8b) `WrongGPU-FG.asi`: `InitializeASI`, `WrongGPU_FgShutdown`, `WrongGPU_FgQuery` |
| `tests/test_fg_win32_transport.cpp`, `tests/fg_mock_*` | (F8b) CPU tests against a stand-in interface DLL, stand-in game modules (static and delay-load imports), an exe that delay-loads the interface, and fake swap-chain / factory objects; synthetic values only |
| `bindings/cp2077-2.14.1.json`, `bindings/sl2-generic.json`, `bindings/fg-functions.json`, `bindings/fg_binding_gen.cmake` | (F8b) the Cyberpunk 2077 title binding on interface build 2.14.1, the generic interface 2.x binding (2.4 and later; options / state from version 1; the exe's DXGI factory creations when the game does not create its factory through the interface), the two FG feature-function names (each alone on its line), and the configure step that turns them into headers in the build tree |
| `win32/fg_title_bindings.h/.cpp` | (F8b) the bindings this build carries, as transport structs; the title binding by exe name, else the generic one when the game links the interface |
| `win32/fg_sl214_codec.h/.cpp` | (F8b) the 2.x native codec: FG options (v1..v5; nothing read past the passed version) and state (written per version) layouts, chain walk by GUID, the options checks, tags and constants into the contract's structs, the present-interval frame clock |
| `win32/fg_panel.h/.cpp` | the "WrongGPU Frame Generation" card on OptiScaler's Custom tab (`OptiPluginInit`, `wg_host_menu_abi.h`): status (off / not active / waiting / generating Nx, preset, presenter / real frames only, reason), and with `fg=on` the preset and multiplier, applied live (`FgProxy::update_live_settings`, lock-free) and saved to `wronggpu-setup\settings.ini` |
| `win32/fg_asi_session.h/.cpp` | (F9) the FG session host of `WrongGPU-FG.asi` (0.7.0: hosted by OptiScaler, `fg_presenter=auto` takes the DXGI flip presenter so OptiScaler's menu and overlay are drawn on every image; the real chain is created on a present queue of the host's own, so presents never wait behind the game's rendering; the first-image budget follows the generated images that miss their display time; the display's refresh caps the generated cadence; game frames wait for a presenter slot when the display shows fewer images than the game renders): the precheck (the local FG model validates, every kernel file of the executor is in `wronggpu-setup\fg-dxil` next to the plugin; either failing: nothing installed, `kFgRuntimeMissing`), the route facts (model / presenter after install, the RDNA 4 adapter from the game's factory, device and queue from the attached chain), facade buffers for the game (served through the wrapper's facade seam, so the presenter never races the game's rendering), the session with one `FgD3d12Executor` and the presenter `fg_presenter` names, every present through `FgSessionPresent`; anything the session does not take is copied from the facade into the real chain and presented as the game asked; resize / fullscreen / release destroy the session, three failures stop it for the run. `fg_net2=exact` (developer key) selects net2's exact path |
| `tests/test_fg_asi_session.cpp` | (F9) ctest `fg_asi_session` (CPU): RDNA 4 ids, the precheck's refusals, the swap-chain flags per presenter choice. `--gpu --dxil <dir>` adds a headless leg: a composition chain created with the host's flags and wrapped, the core on a stand-in chain, synthetic weights; copy path, session creation, generated intervals, generation off (real-only from the facade), resize, release; `--debug-layer` |
| `tests/test_fg_cp2077_binding.cpp`, `tests/fg_cp2077_asi_host.cpp` | (F8b) the binding data, the codec from raw bytes, a whole game session through the stand-in interface with the binding's own GUIDs and names, the factory hook and the present bridge; the shipped plugin in a process with the game's exe name |

## Same split as the RR plugin

`clean/rr_f/proxy` separates the logic (`proxy_core`) from the transport (`hook_dll` / `proxy_dll`); this folder is the
logic half for frame generation. A per-title transport decodes the game's calls into the contract's structs and passes
the native arguments along. The intended interposer transport is the RR plugin's hook mode (`../../rr_f/proxy/HOOK-MODE.md`):
an early `InitializeASI` entry, import slots redirected only for the roles the core handles, the genuine signed interface
file left in place, a version gate on the loaded module, and every other call forwarded to the export the slot held
before. A host that offers an FG provider registration gets `FgProxy::provider()` through `register_with_host` instead.
A title's binding (`FgBinding`: interface version, buffer-type numbers of the FG inputs, feature-function names, the
game's UI range) is filled only from approved call-shape evidence for that build, and `FgBinding::qualified` is `false`
until then, so the core forwards everything.

## The Win32 transport (F8b)

`win32/` is that transport, following the RR plugin's hook mode step by step and reusing its bounded PE view, safe reads
and scan budget (`../../rr_f/proxy/src/rr_guard.h`) unchanged:

- **Install-off gate.** `InitializeASI` reads the four FG keys. `fg=off` (the default) returns before anything is read
  from the process: no transport, no module lookups, no patches. A second `InitializeASI` call changes nothing.
- **Qualified binding only.** The transport also needs an `FgImportBinding` (interface file name and version, the export
  name of each redirected role, the FG options / state feature-function names) and an `IFgNativeCodec` (decodes that
  build's native FG structs). The plugin selects the binding by the exe's file name (below: Cyberpunk 2077); for any
  other game it logs that it carries no binding and patches nothing, even with `fg=on`.
- **Find, gate, redirect.** The interface module already loaded, or the game's own file next to the exe when the exe
  delay-loads it; its version resource against the binding; then the by-name import and delay-load slots of every other
  loaded module, for the redirected roles only (is-supported, requirements, get-function, tag, tag-for-frame, constants,
  evaluate, free). The interface module and the transport's own module are never patched; ordinal imports are left
  alone. The signed interface file is never touched on disk.
- **Bounded, fail-open.** Every header and table read stays inside the image's extent (min of `SizeOfImage` and what is
  mapped); the scan counts modules, descriptors and name-table entries (200000 entries / 100 ms). Nothing is patched
  until the whole scan finished inside its budget; over it, nothing is patched and the game keeps its genuine interface.
- **Routing.** Each handler hands the call to the core, which answers only frame generation and only while it is taken;
  everything else reaches the genuine export (resolved by name, so a host's inline patch on it stays in the chain) with
  the game's own arguments, and its result comes back unchanged. FG evaluate and free are acknowledged while FG is
  taken: generation is present-driven. A codec failure or throw never escapes a handler and never forwards a call twice.
- **Restore.** `WrongGPU_FgShutdown` puts back every slot that still holds a WrongGPU handler (a slot another module
  rewrote since stays as it is) and stops routing; a handler entered later forwards to the genuine export.

The swap-chain seam wraps each chain created through the hooked `CreateSwapChainForHwnd`. `QueryInterface` for every
swap-chain interface returns the wrapper, so the game sees one identity; the wrapper holds one reference on the real
chain. `Attach` claims the chain through the core (`claim_present`, refused while FG is not taken) or the owner
registry; refused, the wrapper stays a pass-through. A test present (`DXGI_PRESENT_TEST`) always goes straight to the
real chain. With a presenter attached, `BeginInterval` runs once per game interval and `PresentInterval` makes the real
presents; `DXGI_ERROR_WAS_STILL_DRAWING` keeps the interval open, so the game's retry presents again without a second
`BeginInterval` (no history rotation, no second submit). Flags and HRESULTs are the real chain's. Presents re-entering
from inside the presenter go straight through. Resize, fullscreen and color-space changes are reported around the real
call (the epoch moves on success); `GetBuffer`, `GetCurrentBackBufferIndex` and `GetFrameLatencyWaitableObject` can be
served by a facade presenter. A presenter that throws is detached, and the interval it was serving is presented once.

## Title binding: Cyberpunk 2077, interface build 2.14.1

`bindings/cp2077-2.14.1.json` is data only. It takes the interface's export names (roles), its file name and the shared
structure GUIDs (viewport, tag, resource, constants, requirements) from the RR binding for 2.14
(`../../rr_f/proxy/binding/`), and adds the FG part: feature 1000, the two FG feature-function names
(`bindings/fg-functions.json`), the options (version 5, 120 B) and state (version 4, 88 B) GUIDs, the FG input buffer
types (depth 0, motion vectors 1, HUD-less colour 2; UI alpha 69, distortion 50, no-warp 54 if ever tagged), the UI range
(up to 5 generated images) and the DXGI factory creation the game imports from the interface. The configure step writes
it into a header in the build tree, so no interface name appears in the code.

- **Load.** The game imports the interface by name in its static import table: the import-table transport above
  applies unchanged. The transport also redirects the game's factory-creation import; the genuine creation runs and its
  result is returned unchanged, and the swap-chain creation hook goes on that factory, so the chain the game holds is the
  wrapped one. A chain created while FG is taken is claimed for the swap-chain wrapper owner.
- **Options.** The original's checks, in its order: the options or the viewport not found in the structure chain -> 20,
  a generated count of 0 (even with the mode off) -> 25, a count above the reported maximum -> 38. Modes 0 off, 1 on,
  2 auto; 3 dynamic is refused (25) because the state reports no dynamic support. The UI-recomposition byte exactly 1
  asks for preset B; otherwise the player's preset applies. Repeated calls are accepted.
- **State.** Written only up to the version the game passed: the minimum size (100), the highest generated count (+52),
  the presented count, the status bits WrongGPU can know (resolution below the minimum, a common constant left unset),
  no VRAM estimate, no fence, no dynamic support.
- **Inputs.** Tags count only when the tag, its resource and the viewport carry the binding's GUIDs, the resource is a
  2D texture and the tag stays valid until present or only during the call. A tag valid only during the call is kept
  through the session host's tag hook (`FgAsiConfig::tag_retain`): it copies the texture on the tagging command list
  into a texture the host holds and fills a zero size or format from the resource; without that hook (the unit tests'
  synthetic pointers are never dereferenced) such a tag is not kept. A tag valid until evaluate is never admitted. The
  game's tag call carries no frame token: every tag and the constants
  between two game presents belong to one interval, and the present bridge presents that interval's token.
- **Present.** Each game present on a claimed chain hands the interval to the core first, with the buffer being presented
  as colour input, then the presenter (`FgAsiConfig::presenter`) presents; a retry after `DXGI_ERROR_WAS_STILL_DRAWING`
  submits nothing again.
- **Everything else forwards.** Super resolution, ray reconstruction, NIS, the latency features and unknown ids keep the
  genuine path with the game's own arguments; feature 1000 is never evaluated by the game, and an FG evaluate is
  acknowledged without reaching the genuine feature while FG is taken.
- **Not settled here.** A chain created before FG is taken stays a pass-through (it is claimed at creation only); the
  per-frame match through the latency markers is not decoded (the present-interval clock assumes a frame's inputs arrive
  before its present); which optional inputs the game sends once 1000 reports supported needs a probe run.

## Rules the core enforces

- **Off by default.** `fg=off` unless the player turns it on. The route needs, in this order: the install switch, a
  qualified title binding, an attached entry, a supported adapter (AMD, RDNA 4, feature probes passed, a LUID), a
  validated model, a qualified presenter and qualified modes. The first failing fact is the reason. While any fails,
  every call, frame generation's too, goes to the next link with the game's own arguments and result; the host entry is
  not even registered while the switch is off.
- **Only frame generation.** Feature-scoped calls are answered here only for the FG feature id. Super resolution, ray
  reconstruction, the latency features and unknown ids pass through. Tags and constants are shared by every feature:
  always forwarded, observed only while FG is taken. A function name the binding does not know is refused rather than
  sent to the genuine feature, so one feature never has two owners.
- **Truthful capability.** The adapter identity is the one the game's device runs on. The highest generated count is
  0 unless the route holds, then min(5, the entry ABI's maximum, the game's UI range). The player's `fg_multiplier` caps
  the effective count of the session; a request above the cap is clamped and logged once.
- **Exact matching.** A present assembles the frame's inputs by frame token and viewport, against the adapter's LUID and
  the latched queue and swap-chain epochs. Missing, stale, conflicting or mismatched inputs, an optional input against
  its declaration, missing constants, a menu frame or the game's FG switch off show the game image real-only for that
  interval with the reason (the runtime contract's numbers). Nothing falls back to another route.
- **One present owner.** `FgPresentOwners` gives each swap chain one owner (host FG wrapper, this adapter or WrongGPU's
  swap-chain wrapper); a present from anyone else is refused without consuming the interval, and an interval (frame
  token) is consumed once. Claiming is refused while FG is not taken.
- **Diagnostics.** One log line per transition of the route decision, of the frame outcome and of the options outcome.

## Settings

Stored with the other per-game settings in `<game folder>\wronggpu-setup\settings.ini`
(`clean/wgshared/include/wg_state.h`):

| Key | Values | Default |
|---|---|---|
| `fg` | `off`, `on` | `off` |
| `fg_preset` | `A`, `B`, `C` | `A` |
| `fg_multiplier` | `2`, `3`, `4`, `5`, `6` (generated images per game frame = value - 1) | `2` |
| `fg_presenter` | `auto`, `hwfq`, `dxgi_tearing`, `dxgi_vsync` | `auto` |

Only these spellings are taken. A missing key takes its default; an invalid value also keeps the default and is
reported. The launcher controls come with the end-to-end wiring.

## Tests

```
cmake -S clean/fg/proxy -B build/fg-proxy && cmake --build build/fg-proxy --config Release
ctest --test-dir build/fg-proxy -C Release --output-on-failure
```

`fg_proxy` runs the suite. `fg_proxy_mutant_1` .. `_8` build the same suite against a core with one rule broken on
purpose (install switch ignored, every feature taken, owner not checked, an interval presented twice, LUID not matched,
a case-folded setting taken, the modes gate ignored, stale inputs reported as missing); each must fail. With the FG
runtime contract in the tree the suite also passes the adapter's packets and sessions through `fg_contract_bridge.h`
and the runtime's own validation.

On Windows, `fg_win32_transport` runs the transport and seam suite: the bounded walk on synthetic images (a name whose
NUL lies past the image end with readable memory after it, a forged `SizeOfImage`, a directory that declares fewer
descriptors than follow, ordinal and VA-based entries, broken headers, the budget), the install-off gate, the binding,
version, attach and budget gates (nothing patched), FG routed to the core while super resolution, ray reconstruction
and the latency features reach the stand-in with identical arguments and results, restore, a throwing codec, null
arguments, double init, and the swap-chain wrapper (identity, reference counts to 0, test presents, retries, occlusion
and device errors, `Present1` parameters, re-entrancy, resize / fullscreen / color space, the facade, single owner,
throwing presenters, the creation hook). `fg_win32_delay_exe` covers an exe that delay-loads the interface before it is
loaded; `fg_win32_asi_off` / `_default` / `_invalid` / `_on` run the shipped plugin in its own process per settings
file. `fg_win32_mutant_1` .. `_8` build that suite against a transport or seam with one rule broken (install-off gate
ignored, a forwarded argument lost, the scan budget ignored, a retry starting a new interval, a test present reaching
the presenter, the owner not claimed, restore leaving slots, a failing presenter kept); each must fail.

`fg_cp2077_binding` runs the title-binding suite: the binding data and its selection by exe name, GUID text, the chain
walk (any order, unknown structures skipped, cycles and over-long chains bounded), the options checks and decoding and
the state per version, all read from raw bytes at the layout's offsets, tags and constants, and a whole game session
through the stand-in interface with the binding's own GUIDs, buffer types and FG function names: support and lookups
for 1000 answered while 0, 1001, 2, 3, 4 and an unknown id forward unchanged, the factory hook, the claimed chain, five
frames (generation, a frame without motion vectors, inputs valid only during the tag call refused and then kept by the
tag hook, a still-drawing retry), the
presenter after the core, restore; plus `fg=off` and FG not taken. `fg_cp2077_asi_off` / `_on` run the shipped plugin in
a process with the game's exe name. `fg_cp2077_mutant_1` .. `_8` break one rule each (a count of 0 accepted, a count
above the maximum accepted, state written past its version, the chain walk ignoring the GUID, the lifetime gate ignored,
no factory hook, no submit from the bridge, a frame clock that never advances); each must fail.
