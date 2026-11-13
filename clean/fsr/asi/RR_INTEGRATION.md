# Preset-F integration through the WrongGPU ASI (historical design note, 2026-10-18)

This note recorded an early plan for Ray Reconstruction (preset F) through the WrongGPU ASI that OptiScaler loads from
its `plugins` folder. The shipped route is different: Ray Reconstruction is provided by the separate plugin in
`clean/rr_f/proxy` (see its README); `WrongGPU.asi` stays the menu plugin for OptiScaler builds before 0.10.

What the ASI work of that time established:

- The ASI's own `InitializeASI` / `WrongGpuAsiQuery` bridge selects Model M / K through WrongGPU's ordinary SR front
  end. The diagnostic export `WrongGpuAsiQueryStatusV1` reports the loaded / initialised state, whether the menu
  adaptation is unavailable, partial or patched, the selected AMD / M / K entry and explicit RR blockers; it exports no
  host frame interface and does not offer F.
- The menu patch is applied as one transaction: the "hooked" flag is set only after all planned call sites are
  patched, with rollback on failure; foreign changes found on the way are kept and reported. WrongGPU's own module is
  pinned before any host call site is published.
- Publicly documented ASI loading does not define a resource interface for Ray Reconstruction, so F needed its own
  host interface (later provided by the plugin interface of the WrongGPU edition of OptiScaler; see
  `clean/rr_f/proxy/include/wg_host_menu_abi.h`).

Tests of that work were CPU-only (status / load checks, menu-thunk checks, synthetic transaction checks) and loaded only
WrongGPU's own ASI. No game files were changed.
