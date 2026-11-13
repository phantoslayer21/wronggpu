# clean/wgshared: what every WrongGPU DLL shares

Everything a WrongGPU user can choose (SR model, RR mode, the RR opt-ins, ...) is one row of ONE registry, and every
WrongGPU DLL in a game process (SR front end, RR plugin, RR runtime) reads the same values from ONE shared settings
block, without depending on another DLL's internals.

| file | what |
|---|---|
| `registry/registry.json` | THE registry: providers, options (= shared-block slots), text options, SR models, RR modes, kernel-variant words, game-interface layouts. Field reference: `registry/schema.md` |
| `tools/gen_registry.py` | generates the four outputs below from the registry; `--check` compares, `--extend <json>` merges test rows into another folder, `--previous <json>` refuses anything that is not append-only |
| `include/wg_registry_gen.h` | GENERATED: `WG_OPT_*` slot ids, value names, defaults, readers, latches, `kWgEntries` (OptiScaler version list), `kWgContractSlots`, `kWgModeRows`, `kWgModelRows`, `WG_REGISTRY_HASH` |
| `include/wg_registry_ini_gen.h`, `ini/wronggpu.default.ini` | GENERATED: the `[wronggpu]` section of `wronggpu.ini` with every key commented |
| `REGISTRY.md` | GENERATED: the human table |
| `include/wg_shared_settings.h` | the shared settings block, version 1 (hand-written; its comment is normative) |
| `include/wg_settings_io.h` | header-only C++17: the one implementation of the data folder rule, the block (open / create / seed / seqlock snapshot / writer election / write + persist), the comment-preserving ini writer, the OptiScaler.ini poll and the `key=value (source)` log text. Define `WG_SETTINGS_IO_IMPLEMENTATION` in exactly one .cpp per DLL; `<windows.h>` stays inside that part |
| `EXTENDING.md` | worked examples: add an RR mode, an SR model, a game-interface family, an API adapter |
| `tests/` | `test_gen_registry.py`, `test_dummy_extension.py` (data: `tests/data/ext-dummy.json`), `test_shared_settings.cpp`, `test_cross_reader.cpp` (the SR side's block implementation against the RR side's independent reader) |

Rules:
- Regenerate after every registry change: `python clean/wgshared/tools/gen_registry.py`, and commit the outputs.
  `tests/test_gen_registry.py` fails when a committed output differs from a fresh generation.
- Option ids are shared-block slots: append only; never renumber, reuse or remove (`--previous` refuses it). Values
  of a choice are append only too (an entry's id is derived from the option id and the value index).
- Nothing outside this folder hard-codes a mode name, a model name, a `[wronggpu]` key or a version-list id.
- The data folder of every DLL is `<game exe folder>\WrongGPU\` (`wg::data_dir()`; `WG_DATA_DIR` overrides it for
  tests and harnesses); `wronggpu.ini` lives there.
- Reader precedence everywhere: block slot (its `set_mask` bit) > the option's environment variable > own ini >
  registry default. The block's single writer (the SR front end when OptiScaler hosts it) persists every change to
  `wronggpu.ini`, so the next start without OptiScaler keeps the menu's choice.

Build and test: `cmake -S clean/wgshared -B build/wgshared && cmake --build build/wgshared --config Release && ctest
--test-dir build/wgshared -C Release` (the SR front end's build includes this folder, so its ctest runs them too).
