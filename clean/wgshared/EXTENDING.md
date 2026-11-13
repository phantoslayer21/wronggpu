# Extending WrongGPU: modes, models, game-interface families, API adapters

Everything a player can choose is one row of `registry/registry.json`; every way of reaching a game is one transport.
**Status (merge of round i15):** the registry half is done and tested (sections 1, 2, 5). The RR plugin half (game-interface
families as data, a hook engine and transport adapters, section 3 and 4) was NOT delivered in round i15: the plugin
still has one build per game-interface version, and sections 3 and 4 below describe how it is extended today and
what the data-driven form is meant to look like. Rules that hold for all of them:

- Ids are append-only. A new option, value, mode, model or entry goes at the end; nothing is renumbered, reused or
  removed (`python clean/wgshared/tools/gen_registry.py --previous <old registry.json>` refuses it, and
  `tests/test_gen_registry.py` checks the committed registry against `tests/data/registry.prev.json`).
- After a registry change run `python clean/wgshared/tools/gen_registry.py` and commit its four outputs
  (`include/wg_registry_gen.h`, `include/wg_registry_ini_gen.h`, `ini/wronggpu.default.ini`, `REGISTRY.md`).
  `tests/test_gen_registry.py` fails when a committed output differs from a fresh generation.
- No DLL hard-codes a mode name, a model name, a `[wronggpu]` key or a version-list id: they read
  `wg_registry_gen.h` (`kWgOptions`, `kWgEntries`, `kWgModeRows`, `kWgModelRows`).
- The extension path is tested for modes and models: `tests/data/ext-dummy.json` adds an RR mode `dummy` and a
  game-interface family `2.99` row through the tables only (`tests/test_dummy_extension.py`; the generator accepts the
  family row, but no plugin code consumes family rows yet, see section 3).

## 1. Add an RR mode (example: `turbo`)

A mode is a choice of kernel variants, one word per variant group (`head`, `input`, `body`, `output`, `resolve`).

1. `registry/registry.json`, `modes`: append
   `{"id": "turbo", "option": "rr_mode", "head": "relaxed", "input": "lite", "body": "relaxed", "output": "fast", "resolve": "hw"}`
   and append `"turbo"` to `options[id 1].values` (it must equal the mode rows, same order; `values_from: modes`).
2. If every word already exists in `variants`, that is all: the RR runtime builds the mode from its words. A new word
   (say `body: "int8"`) is a new kernel variant: append the word to that group in `variants`, add the kernels, and add
   the word to the runtime's word table (`clean/rr_f/runtime/src/rr_modes.cpp`; its `static_assert` stops the build
   while the registry names a word the runtime lacks).
3. Regenerate. What appears without further code:
   - `WG_RR_MODE_TURBO 3`, `WG_RR_MODE_COUNT 4`, `kWgRrModeNames[3] = "turbo"`, a fourth `kWgModeRows` row;
   - a version-list entry `3.1.5 WrongGPU RR turbo`, id `0x5747000000000104` (`WG_ENTRY_ID_BASE | 1 << 8 | 4`), shown
     in OptiScaler's WrongGPU version box by the SR front end (`clean/fsr`, `OwnEntry`); choosing it writes
     `rr_mode=turbo` to the shared block and `wronggpu.ini`;
   - the one-folder-per-choice name `WrongGPU RR turbo` (`folder_label`);
   - `rr_mode = turbo` accepted in `wronggpu.ini` and `WG_RR_MODE=turbo`, listed in the ini comment and `REGISTRY.md`;
   - the RR runtime switches to it at the next frame (history reset) once its kernels are prepared.
4. Prove it: an `--extend` file like `tests/data/ext-dummy.json` with the new row must generate, and the RR runtime's
   byte replays of the existing modes must stay identical.

## 2. Add an SR model (example: Model `L`)

1. `registry/registry.json`, `models`: append
   `{"id": "L", "option": "sr_model", "runtime_option": "model=L", "kernels": "clean/l"}` (add `"build_flag"` if the
   front end may be built without it) and append `"L"` to `options[id 0].values`.
2. The SR runtime must accept `model=L` (the front end passes `runtime_option` in the create options); the model's
   kernels live with the runtime.
3. Regenerate: `WG_SR_MODEL_L 2`, `kWgModelRows[2] = {"L", "model=L"}`, the entry `3.1.5 WrongGPU (Model L)` with id
   `0x5747000000000003`. The SR entries come before the RR entries because entries are ordered by option id, so the
   index of every RR entry in the host's list moves by one; that is why the host is given ids, not indexes, and why
   OptiScaler.ini's `UpscalerIndex` is read against the list of the running build.
4. Front end: nothing, unless the model has a build flag; then one line in `ModelBuiltIn` (`clean/fsr/src/config.cpp`),
   the single place that maps a model to its CMake option (Model K: `WG_MODEL_K`).

## 3. Add a game-interface version (example: 2.15)

**Today (what exists in the tree):** one build of the plugin per interposer version. A version is one binding table
plus an entry in the build script; nothing is data-driven at run time yet.

1. `clean/rr_f/proxy/binding/rr-f-2.15.json`: copy `rr-f-2.14.json` and change `api_version`, `file_version`,
   `exports_key` (the key under which the interposer's export names and ordinals are listed in
   `rr-f-sl-interposer-exports.json`; add that key too), `release_version`, `version_magic` and the GUIDs and role
   names if the layout differs (they come from the spec annexes A9 / A10, never from a binary).
2. `clean/rr_f/proxy/tools/build.py`: add `15` to `VERSIONS`. The script builds the release proxy, the probe, the hook
   plugin and the hook probe for that version and runs every test against a fake interposer generated from the table
   (`tests/gen_fake_original.py`).
3. A version whose structure layout differs from `A10-2.7` and `A10-2.14` also needs its declarations in the plugin's ABI
   header (`include/rr_proxy_abi.h` and the table's `roles`); that is code, not data.
4. The registry carries a list of known layouts (`gi_layouts`) and an optional family index
   (`families_index`, `clean/rr_f/proxy/binding/families.json`, not present in this tree). `gen_registry.py` validates
   the index when it exists. The option `rr_accept_provisional_families` (id 8) is reserved for it and has no reader yet.

**Planned form (not built):** one plugin build carrying a row per family (`families.json`: version range, table,
layout, status) and choosing the row at run time from the interposer's version resource, so that a new version is a
row plus a table and a fake interposer. The game-interface families 2.6, 2.8 and 2.10 of the owner's games need exactly
this; until it exists they need three more per-version builds.

## 4. Add an API adapter (example: the vendor-API direct route of spec 64)

**Today:** the plugin has one transport, the hook-mode import redirection in `clean/rr_f/proxy/src/hook_dll.cpp`, on
top of the unchanged core (`proxy_core.*`, `answers.*`). A second transport (for example the vendor-API route of spec
64, or a game-interface 1.x route) is a second `*_dll.cpp` beside `hook_dll.cpp` and `proxy_dll.cpp` that calls the same
core, built from its own binding table by `tools/build.py`. It reads its settings through `include/wg_settings_reader.h`
(the plugin's read-only view of the shared block) and never adds its own ini keys: a new setting is a registry option.
The forward / consume rules of spec 65.5 are, today, the set of roles each build redirects (the table's `roles`; the
release build never redirects the tag, tag-for-frame or constants roles away from the interposer except to observe and
forward them).

**Planned form (not built):** a hook engine (import redirection) and a transport interface
(`name`, `detect`, `install_phase_a`, `install_phase_b`, `family`) with the forward / consume policy as a table
(`binding/policy.json`), so that an adapter is a folder with one detect function and its fake in the tests.

## 5. Add an option (example: `rr_sharpness`, an int 0..10)

Append `{"id": 10, "key": "rr_sharpness", "type": "int", "min": 0, "max": 10, "default": 0, "readers": ["rr-runtime"],
"live": true, "latch": "next-frame", "ini_section": "wronggpu", "doc": "..."}` to `options` (next free id; at most 64,
the block's slot count), regenerate, and read it with `wg::resolve(WG_OPT_RR_SHARPNESS, &snapshot, &ini)` (block >
environment > ini > default); log it with `wg_describe`. A `choice` option with `"menu": {"list": "versions", ...}`
also puts its values into the host's version list.
