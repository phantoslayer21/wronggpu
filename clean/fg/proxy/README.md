# Frame generation game adapter (issue #96, F8a groundwork)

The part of frame generation that faces the game: a versioned adapter contract, the adapter core and the per-game FG
settings. Nothing here is loaded by a game, the front end or the launcher yet, and nothing is advertised: the core
takes frame generation only when every route fact holds, and the last one, `kFgModesQualified`, stays `false` until
the end-to-end gate and a qualified presenter pass.

| Path | Content |
|---|---|
| `fg_adapter_abi.h` | the host-neutral contract (plain C, size- and version-tagged): adapter identity, capability, decoded options / tags / constants, the present descriptor and result, the assembled packet and session (field for field the FG runtime contract's `FgFramePacket` / `FgCreateDesc`), the next link of the game interface (`WgFgChainV1`), the host provider entry (`WgFgProviderV1` / `WgFgHostV1`) and the executor hand-off |
| `fg_proxy_core.h/.cpp` | the core: install-off gate, feature-scoped routing, truthful capability, tag-to-packet assembly, single present owner, one diagnostic per transition |
| `fg_settings.h/.cpp` | the per-game keys `fg`, `fg_preset`, `fg_multiplier`, `fg_presenter`: strict parse, defaults, round trip, apply without touching other keys |
| `fg_contract_bridge.h` | conversion to the FG runtime contract (`clean/fg/runtime/fg_contract.h`) with compile-time checks that every shared number matches; used only where that contract is in the tree |
| `tests/test_fg_proxy.cpp` | CPU tests against a fake game interface (super resolution, ray reconstruction, frame generation and the latency features in one session), a fake host and a fake executor; synthetic values only |

## Same split as the RR plugin

`clean/rr_f/proxy` separates the logic (`proxy_core`) from the transport (`hook_dll` / `proxy_dll`); this folder is the
logic half for frame generation. A per-title transport decodes the game's calls into the contract's structs and passes
the native arguments along. The intended interposer transport is the RR plugin's hook mode (`../../rr_f/proxy/HOOK-MODE.md`):
an early `InitializeASI` entry, import slots redirected only for the roles the core handles, the genuine signed interface
file left in place, a version gate on the loaded module, and every other call forwarded to the export the slot held
before. A host that offers an FG provider registration gets `FgProxy::provider()` through `register_with_host` instead.
No transport is wired in this change: a title's binding (`FgBinding`: interface version, buffer-type numbers of the FG
inputs, feature-function names, the game's UI range) is filled only from approved call-shape evidence for that build, and
`FgBinding::qualified` is `false` until then, so the core forwards everything.

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
reported. The launcher controls come with the end-to-end wiring (F8b).

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
