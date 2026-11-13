# `registry.json` schema (`wronggpu.registry.v1`)

`registry.json` is the one table of everything a WrongGPU user can choose. `tools/gen_registry.py` turns it into
`include/wg_registry_gen.h` (every DLL), `include/wg_registry_ini_gen.h` + `ini/wronggpu.default.ini` (the
`[wronggpu]` section of `wronggpu.ini`) and `REGISTRY.md`. Nothing else may hard-code a mode name, a model name, an
ini key of this section, a version-list entry id or its label. `tests/test_gen_registry.py` fails when a committed
output differs from a fresh generation.

## Top level

| field | type | meaning |
|---|---|---|
| `schema` | string | `wronggpu.registry.v1`; emitted as `WG_REGISTRY_SCHEMA` |
| `api_level_prefix` | `"a.b.c"` | the FidelityFX upscaler API level every version-list name starts with (`WG_API_LEVEL_PREFIX`). The host reads "the FidelityFX version" of the DLL from the FIRST listed name, so it must stay a 3.x.y number (below 4.0.0; `clean/fsr/tests/test_frontend.cpp TestHostVersionHandling`) |
| `entry_id_base` | hex string | high bits of every version-list entry id (`WG_ENTRY_ID_BASE`, "WG") |
| `readers` | list | reader names in bit order: `WG_READER_<NAME> = 1 << index` |
| `latches` | list | latch names in value order (`WG_LATCH_*`) |
| `kinds` | list | option kinds in value order (`WG_KIND_*`): `choice`, `onoff`, `int` |
| `providers` | list | `{id, label, kind, choice_option}`: the two WrongGPU products and the option that chooses inside each |
| `options` | list | the slot options, below |
| `text_options` | list | string options, below |
| `models` | list | SR models, below |
| `variants` | list | `{group, words}`: the RR runtime's kernel-variant groups; a word's index is its `WG_VARIANT_<GROUP>_<WORD>` value |
| `modes` | list | RR modes, below |
| `gi_layouts` | list | structure-layout names the RR plugin compiles (`A10-2.7`, `A10-2.14`); a game-interface family must name one |
| `families_index` | path | the RR plugin's game-interface family table (`clean/rr_f/proxy/binding/families.json`); read for validation and `REGISTRY.md` when present |

## `options[]` (one int32 slot each in the shared block)

| field | meaning |
|---|---|
| `id` | slot index in the shared block = position in the list. **Never reused, renumbered or removed**; a new option is appended with the next id (at most 64) |
| `key` | ini key in `[wronggpu]`, log word (`rr_mode=fast (menu)`), `WG_OPT_<KEY>` |
| `type` | `choice` (value = index into `values`), `onoff` (0/1), `int` (`min`..`max`) |
| `values` | choice only: value names in index order (`WG_<KEY>_<VALUE>`, `kWg<Key>Names`). Append only |
| `values_from` | `models` / `modes`: the list must equal the ids of the model / mode rows naming this option (an extension that adds a row appends the value) |
| `default` | choice: a value name; onoff / int: a number |
| `readers` | who reads the slot (`kWgOptions[].readers`); a reader snapshots only its own slots |
| `live` | `true`: the reader applies a change without re-creating anything |
| `latch` | when a change takes effect: `next-frame`, `rr-create`, `sr-create`, `start` |
| `ini_section` | always `wronggpu` today |
| `env` | process-environment override (tests, harnesses); `""`/absent = none |
| `contract` | `true`: a row of the RR plugin's 60.9 contract table; contract index = order among contract rows (ids 2..6 -> 0..4 today) |
| `opt_in` | `true`: changes the output versus the original (60.9 flag bit 1) |
| `menu` | `{list: "versions", label, folder_label}`: every value becomes an OptiScaler version-list entry. Entry id = `entry_id_base | (id << 8) | (value index + 1)`; label = `api_level_prefix + " " + label` with `{value}` replaced; `folder_label` names the folder in the one-folder-per-choice fallback. Entries are ordered by option id, then value index, so the first entry stays `3.1.5 WrongGPU (Model M)` |
| `doc` | one line; becomes the ini comment and the `REGISTRY.md` text |

Reader precedence for a slot, in every DLL (`include/wg_settings_io.h`): shared-block slot with its `set_mask` bit >
the option's `env` > the reader's own ini > `default`.

## `text_options[]` (not slots)

`{key, default, readers, env, doc}`. They live in `wronggpu.ini` and are copied into the block's text area by the
block's creator at seeding (`key=value`, NUL separated); read-only afterwards.

## `models[]`

`{id, option, runtime_option, kernels, build_flag}`. `id` is the value name in `option` (`sr_model`);
`runtime_option` is what the SR front end passes to the SR runtime (`model=K`); `build_flag` is the compile switch
that has to be on for the model to be offered (`WRONGGPU_MODEL_K`). Emitted as `kWgModelRows`.

## `modes[]`

`{id, option, <group>: <word> for every variants group, doc}`. `id` is the value name in `option` (`rr_mode`). Each
group names one word of that group's `words`; the RR runtime maps each word to its variant bit (its word table must
know every word: a `static_assert` there fails the build otherwise). Emitted as `kWgModeRows`.

## Extensions (`--extend <file>`, tests only)

Same field names, `schema: wronggpu.registry.ext.v1`: `options`, `text_options`, `models`, `modes`, `variants`
(extra words), `gi_layouts`, and `families` (`{family, min, max, table, base_table, layout, status, note}`, the row
shape of the RR plugin's `families.json`). Everything is appended; the generator refuses an extension that changes
an existing id. `tests/data/ext-dummy.json` is the committed example (mode `dummy`, family `2.99`).
