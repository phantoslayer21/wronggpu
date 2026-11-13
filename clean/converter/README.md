# wronggpu-convert — install-time model converter (specs 08, 48 and 57)

C++17, standard library only (own SHA-256 from FIPS 180-4, own JSON reader, own Base64 decoder, own block
decompressor written from the seven steps of spec 08 section 4.4). Implements spec 08 sections 4–8 with the command
line of section 10, spec 48 (preset K) and spec 57 (preset F, Ray Reconstruction). One program reads all five
map versions:

| Map version | Model | Sources | Adds |
|---|---|---|---|
| 1 | preset M (spec 08 draft 1) | W, D, K | – |
| 2 | preset K (spec 48 draft 2) | T (Base64 text, `segments`) | tensors index W = the decoded segments of T; `capped_softmax` |
| 3 | preset K (`data/preset-K.v1.map.json`, spec 48 draft 3) | T, O (keyed, `i32le`) | op `i32`, dtype `int32`, output group `net` (`window_offsets`) |
| 4 | preset M (`data/preset-M.v1.map.json`, spec 08 draft 2) | W, D, K, and the optional F (keyed, `hex32`, `lz4-block`, `variants`) | version 1 plus source F, a `digest` list for a group read from it, output group `filter` (`output_taps`) |
| 5 | preset F (`data/preset-F.v1.map.json`, spec 57) | W (unkeyed), keyed blocks (D, SH), keyed 16-byte data-word windows, keyed code images (IMG-IN with two `variants`, no decoding) | spec 57 sections 5-7: see "Preset F" below |

Negative strides are accepted in versions 1-4 (not in 5). A map may only use what its version defines (for example `base64`
in a version-1 map, `int32` in a version-2 map, or `hex32` in a version-1 map is "converter data damaged").
Version 4 is version 1 plus source F; it does not include the preset-K additions of versions 2 and 3. Versions
above 5 are refused ("converter data damaged"). A version-5 map that uses a version 1-4-only feature, and a map of
versions 1-4 that uses a version-5-only feature (`groups`, `field`, `encoding: none`, `also`, `sites`, `require`,
`allow_nonfinite`, an axis `limit`, `max_src`, `checks`, `requires_variant`, the new ops and dtypes), is invalid
(spec 57 section 5.1); the op names `f16` and `exp` mean spec 08's operations in versions 1-4 and spec 57's in
version 5.

**The preset, by name.** A map may name its preset in the optional top-level field `"preset"` (text: `"M"`, `"K"` or
`"F"`; the table `known_presets` in `src/map.cpp`). The name decides the player's "no preset-X model" message (exit 12);
the version still decides the map's features. A map without the field gets the preset its version always meant
(versions 1 and 4: M; 2 and 3: K; 5: F), so the shipped maps are read exactly as before. A preset name that is not known,
not text, or whose preset does not use the map's version (for example `"preset": "K"` in a version-1 map) is "converter
data damaged". This is how a further preset that reuses an existing map version is added: one row in that table.

**Preset L** (Model L, issue #89) is such a row: `"preset": "L"` in a map of version 1 or 4. Its maps use the record
kinds of preset M's (no new element format, op or source kind); its parameters are a block of their own in the same
library file as preset M's, found by its own length and hashes like any source. Its map,
`data/preset-L.v1.map.json` (map version 1; sources W, D and K), has no `"preset"` field, so the caller names the
preset: `wronggpu-convert --map data\preset-L.v1.map.json --preset L --dll <nvngx_dlss.dll> --out <dir>` (API:
`load_map_bytes(..., "L")`). `--preset` reads a map without the field as that preset when the preset uses the map's
version; a map with the field must name the same preset; anything else is "converter data damaged", and an unknown
`--preset` is a usage error (exit 1). Without `--preset` the preset-L map reads as a version-1 map (preset M's message).
`tests/run_tests.py` checks the real map (validation, the messages, `--preset` mismatches) and converts our own random
data in its exact layout at two placements. It also covers preset L with a synthetic library file holding an L block
directly before an M block, and maps for both: each converts its own block, and a file without a preset's block fails
with that preset's message. It also converts an **L-shaped** map
(`tests/cases.py` `l_shaped_map`): preset M's map layout with the input encoder's output linear 64 x 128, and two new
stages of L's record kinds (two linears and a C = 32 Swin block before `enc0`, a C = 32 Swin block before `dec0`),
filled with our own random data. Its stage names and placement are stand-ins; the real map defines its own. A map of
versions 1-4 must still have the sources W, D and K: should preset L have no dither table or constants, that rule
becomes per preset.

## Preset F (map version 5, spec 57)

    wronggpu-convert --map data\preset-F.v1.map.json --dll <nvngx_dlssd.dll> [--dll <other> ...] --out <dir> [--raw]

- **Map validity (57 section 5.4)** is checked completely before any file is opened, including every source-variant
  assignment a group can be produced under (`requires_variant`): every read and site index inside its source (per
  variant), at least one selected site per const tensor, `score_bias` reading W with `axes: []` and 12 pairs in
  0..3, 32-bit windows for `a_src`, `w_src` and `max_src`, op element sizes and output dtypes, window fields, keyed
  sources, 16-digit FNV / 64-digit SHA strings, a single-string group digest, `require` and check names, check
  constants that are exactly binary32 numbers (compared as exact decimals), scalar check operands, `attn_wmax`
  inputs that are earlier scalar float16 tensors. Any failure: code 20, no file read.
- **Search (57 section 4).** W first (unkeyed), then the key, then every other source with the block search of 08
  section 4.2 in one shared scan per prefix length (`locate_blocks`: 16-byte windows share one scan, 64-byte
  prefixes another; every FNV hit is checked against its own SHA-256; per source the first match in pass order,
  so a damaged copy is skipped when an intact one follows). A window's one element is the value at its `field`.
  IMG-IN's variants are tried in map order; the variant found is logged (`variant used: IMG-IN variant 0`).
- **Decoding (57 sections 5.3, 6).** `src/convert5.cpp` and `src/ops5.*`: the index formula with `perm` and
  `limit` (zero elements read nothing), `also` (bit-identical second reads), `e4m3` (Q8 of 51.1, sign of zero
  kept), `f16`, `u16`, `u32`, `u32_as_f32`, `exp` (correctly rounded, cap after the exponential), `score_bias`
  (binary32 steps, correctly rounded sqrt, RN16), the four const operations over all site elements of the variant
  found, `attn_wmax` (every finite binary16 c in the clamp range; `attention_sum_may_overflow` is logged), the
  finite-output rule, `require` and group `checks`. Groups whose `requires_variant` is not met (`constants_b2` for
  IMG-IN variant 0) are not produced. Every group digest is compared before anything is written.
- **Output.** `model.wgcm` container version 2 (dtype codes for float16, uint16, uint32, e4m3; see
  [OUTPUT-FORMAT.md](OUTPUT-FORMAT.md)); `--raw` writes `raw/<group>/<name>.bin`. Whether `constants_b2` is present
  is the presence of its index entry. The runtime reads the file with `clean/rr_f/model/model_file.py`
  (`admit_model_file`), which recomputes every group digest with the stored key (57 section 7.3).
- **Several files (57 section 1):** the first that converts completely is used (IMG-IN variant 0 and 1 both
  "convert completely"; `constants_b2` is not an optional group in the sense of 08's `filter`).
- **Player messages (57 section 8):** see the exit-code table (codes 10-14, 20, 23).


Source F and the group `filter` (spec 08 sections 4.3, 4.4, 7.2):

- F is searched after W, D and K, each of its `variants` in map order with the keyed block search of section 4.2;
  the first variant with a match is used. The matched block is decompressed to the variant's `decoded_length`,
  checked against `decoded_sha256`, and the five fields (eight hexadecimal digits each, either case) at the
  variant's `fields` offsets become the five `u32_as_f32` elements of `filter/output_taps`. The group's digest is
  compared with the entry of the `digest` list that belongs to the variant found. Nothing else of the decoded text
  is used, kept, written or logged.
- No variant found (also: a block with one bit changed, a file that ends inside the block, hashes that were made
  with another key) is not an error: the conversion succeeds, the model file has no `filter` group
  ([OUTPUT-FORMAT.md](OUTPUT-FORMAT.md) says exactly how a reader sees that), the log says so, and the player gets
  one more line after the success message.
- A block that matches a variant's hashes but then does not decompress, does not match `decoded_sha256`, or has a
  field that is not hexadecimal can only come from damaged converter data: code 20, nothing written.
- The log names the variant ("source F at offset … (variant 1 of 2)", "filter: variant 1"), never the weights.

Preset K, for the spec team (48-T1, 48-T6):

    wronggpu-convert --map data\preset-K.v1.map.json --dll <nvngx_dlss.dll> --out <dir> --raw
    python clean\k\runner\k_bench.py --build <kernel build> --rocm "C:\Program Files\AMD\ROCm\7.2" --check 1920x1080 --model <dir>

`k_bench --model` reads `<dir>/raw` (`clean/k/runner/k_model.py`), including `net/window_offsets`.

    wronggpu-convert --map <map.json> --dll <nvngx_dlss.dll> [--dll <other> ...] --out <dir> [--raw] [--threads N]

- `--dll` may be repeated: files are tried in that order (player's choice first). The first that converts
  completely **with** the group `filter` is used; if none has it, the first that converts completely; every
  rejection is reported (section 8, "Several DLLs", as the spec team amended it on 2026-10-13: every file that
  converts gives the same other stages, so the order only decides whether the filter weights come along). Each
  file is converted in memory; only the chosen one is written. Nothing after a file with the group is read. The
  log names the file that was used (`using …`, `converted from …`). The installer walks its own list with the
  same rule (`has_every_group` in `src/api.h`).
- Output: `<dir>/model.wgcm` (format in [OUTPUT-FORMAT.md](OUTPUT-FORMAT.md)); with `--raw` also
  `<dir>/raw/<stage>/<name>.bin` (row-major little-endian float32, uint8 for M's enc0 T, int32 for K's
  `net/window_offsets`); `raw/filter/output_taps.bin` (20 bytes) only when the group `filter` exists.
- Standard output: the player message. Standard error: the log (paths, sizes, version, source offsets,
  timings, map hash; never decoded values or source bytes).
- `--threads N` limits the search threads (default: all hardware threads).

## Exit codes and messages

| Code | Condition (spec 08 section 8) | Message on standard output |
|---|---|---|
| 0 | converted, every stage digest matches | The model was converted successfully. |
| 0 | converted, but no variant of source F was found: no `filter` group (spec 08 section 8; not an error) | The model was converted successfully. — and on the next line: The output filter of this DLSS build is not known to this version; WrongGPU's own filter will be used. |
| 1 | bad command line | usage |
| 2 | output folder cannot be written | The converted model could not be saved in … |
| 10 | file missing or unreadable | This file cannot be read. |
| 11 | file smaller than W (or above 4 GiB) | This is not a DLSS model file. |
| 12 | M: W not found. K: T not found | This DLSS file has no preset-M model. Version 310.5 or newer is needed. / This DLSS file has no preset-K model. Version 310.1 or newer is needed. (+ "This file is version a.b.c.d." when the file has a version resource) |
| 12 | F: W not found (spec 57 section 8) | This DLSS file has no preset-F (Ray Reconstruction) model. Version 310.7 or newer is needed. (+ the file version when there is one) |
| 12 | L: W not found (a map with `"preset": "L"`) | This DLSS file has no preset-L model. Version 310.5 or newer is needed. (+ the file version when there is one) |
| 13 | M: W found, D or K not found. K: T found, O not found (spec 48 draft 4 section 8). F: W found, another source (or every variant of IMG-IN) not found. The log names the missing source | This DLSS build is not supported yet (model data found, but other parts differ). |
| 14 | F only (spec 57 section 8): a constant's sites disagree, a `const_f16x2` with unequal halves, a `const_f16lo` with a non-zero high half, a non-finite value where 57.6.1 or 57.6.2 forbids it, an RN16 overflow, the two reads of an `also` tensor differ, a `require` or check fails, the attention constants fail 57.6.4. The log names the tensor or check (never a value). Not an installer error: the next file is tried | This DLSS build is not supported yet (model data found, but other parts differ). |
| 20 | map invalid: unreadable, bad JSON, bad format or version, unknown op / locate / element / encoding / dtype (or one its version does not define), missing field, unknown perm, perm not dividing its axis, `axes` length ≠ `shape` length, a segment that is not a positive multiple of 4 or runs past the end of T, K sources other than T (and O), `capped_softmax` without 8 × 8 last axes, a source F that is not a keyed optional `lz4-block` block source with variants, a field outside the decoded text, variants with different numbers of fields, a `digest` list whose length is not the number of variants (or a list for a group without variants), a group that mixes tensors from F and from other sources, … Also (spec 08 section 4.3): a variant of F matched its hashes, but the block does not decompress to `decoded_length`, does not match `decoded_sha256`, or a field is not eight hexadecimal digits | Installer error: the converter data is damaged. … |
| 20 | F: every structural failure of 57 section 5.4 (an index outside its source included) and a defensive bounds failure at run time | Installer error: the converter data is damaged. … |
| 21 | M, K: an index outside its source (checked when the map is loaded; source lengths are part of the map; for W from T against the decoded length bound, and again after decoding), a `u16_shr` result above 255, a segment of a found T that is not valid Base64, or an odd decoded total | Installer error: the converter data does not fit this model … |
| 22 | M, K: a decoded value is NaN or infinite | Installer error: a converted value is not a valid number … |
| 23 | a stage / group digest does not match (for `filter`: the entry of the variant found) | Conversion self-check failed. … |

The map is validated completely before any DLL is read. Codes 2 and 20–23 are installer errors: the remaining
`--dll` files are not tried, and nothing is written even if an earlier file converted without the group. With
several files and only file-related rejections, the exit code is the first file's. The note of code 0 without the
group is printed only when the file that was used has no group. On any failure nothing is kept: output is written to a staging folder inside `<dir>` and moved into place
only after every check passed, so earlier valid output stays untouched.

## How it works

- `src/map.*` — map parsing and validation (section 5), including every index range (section 5.1).
- `src/locate.*` — source search (section 4.2). W first (offsets that are multiples of 16, then other even
  offsets, then odd ones; the lowest offset wins within a pass), then the key, then D and K. The 64-byte FNV-1a
  check runs on all threads: eight independent hash chains at a time for the 16-byte pass, and a low-16-bit
  prefilter (exact: the low 16 bits of FNV-1a depend only on low 16 bits) in 16-bit vector lanes for the other
  offsets. SHA-256 runs only on FNV hits. Pair sources (K): each gap in map order, every offset ascending.
  For preset K: T first (the key source), then O with keyed hashes; the same search. Source F (map version 4):
  after the required sources, one block search per variant in map order (`locate_variant`); a file without the
  block costs one full scan per variant.
- `src/lz4block.*` — block decompression (section 4.4), written from the spec's seven steps: bounds-checked on
  every read, never more than `decoded_length` bytes of output, lengths that could only overrun refused before
  anything is copied.
- `src/convert.*` — decode operations (section 6) and stage digests (section 7.3). Source F: `decode_variant_block`
  (decompress, `decoded_sha256`, the hexadecimal fields); a group whose source has no found variant is left out.
  For preset K, the segments of T
  are decoded (RFC 4648 section 4: standard alphabet, `=` only at the end of a segment, bits below the padding
  ignored) and concatenated in map order into W; `capped_softmax` collects each group of 64 elements (the last two
  axes, s = 8 i + j) and `i32` copies the int32 elements of O.
- `src/numeric.*` — exact roundings: RN16, RN32 and q8 by explicit integer/power-of-two arithmetic (independent
  of the FPU rounding mode); `f16_scaled` with an exact binary64 product and a binary64 quotient rounded once more
  to binary32 (innocuous double rounding since 53 ≥ 2·24 + 2); `exp` in double-double arithmetic (~100 bits)
  with a final rounding that resolves exact binary32 midpoints, i.e. correctly rounded. `capped_softmax`: every
  binary32 step in binary64 rounded once more to binary32 (innocuous for + − × ÷ of binary32 operands), the
  correctly rounded exp above, the sum in order s = 0…63, then RN16(RN32(RN16(E) / Σ)).
- `src/output.*` — container, raw files, staging and commit.
- `src/api.*` — the library interface the installer uses: `convert_file` (also gives the found variant and the
  player note), `write_converted`, and `verify_model_file`, which accepts a model file with or without the
  optional group and reports which (`ModelFileInfo`).
- `src/pe_version.*` — file version for display/logging only (never used to accept or reject).

Timing on this container (4 threads, synthetic 300 MB file): full scan of a rejected file ≈ 1.0 s (3.5 s with
one thread); a successful conversion with W at a 16-byte multiple near the end and D at an unaligned offset
≈ 1.5 s (4.1 s with one thread).

## Building

- CMake (Windows with MSVC, Linux): `cmake -S . -B build && cmake --build build --config Release`, then
  `ctest --test-dir build -C Release` (runs the unit tests, the decompression vectors of
  `data/lz4-block-test.json` and, if Python 3 is found, the synthetic tests).
- Without CMake: `./build.sh [test]` (Linux, any C++17 compiler; `CXX=...` to choose) or `build.bat [test]`
  from a Visual Studio x64 Native Tools prompt. Output in `build-plain/`.

Floating-point flags matter for exactness: no fast-math, no FMA contraction (`-ffp-contract=off`,
`/fp:precise`); the build files set them.

## Tests (synthetic only; no real model file is ever used)

- `tests/unit_tests.cpp` — SHA-256 (FIPS 180-4 example vectors, one-million-'a', incremental vs one-shot), FNV-1a
  reference values, JSON reader, RN16 round trip of every binary16, RN32 against the hardware conversion on
  2 million random doubles and at exact midpoints, q8 ties/saturation/subnormals/zero sign, `f16_scaled` against
  binary32 hardware arithmetic for every binary16 input, `exp` against binary64 `exp`. Block decompression on
  blocks written by hand for the test (literals only, matches, distance 1, length extensions, every malformed
  form, lengths far beyond the output) and on 300 000 damaged or random blocks (never more than `decoded_length`
  bytes written; run once under AddressSanitizer: no read outside the block). `decode_variant_block` (both digit
  cases, wrong hash, wrong key, fields that are not hexadecimal). With `--work <dir>`: the library interface on a
  small synthetic file and version-4 map made in the test (`convert_file` with variant 1, variant 2, both blocks,
  no block, damaged blocks; `write_converted`; `verify_model_file` with and without the group, on a file written
  with the version-1 map, and on files with a changed or swapped `filter` digest).
- **08-T7**: `wgc-unit-tests --lz4-vectors data/lz4-block-test.json` runs the spec team's 17 decompression
  cases (10 ok, 7 malformed) through the converter's decoder; `tests/run_tests.py` runs the same file through the
  independent Python decoder in `tests/synth.py`.
- `tests/run_tests.py` (Python 3 standard library) with `tests/synth.py` (an independent reference: exact
  integer-rational arithmetic, 50-digit decimal `exp`, `hashlib`) and `tests/cases.py`:
  - **feature case**: W holds all 65 536 binary16 patterns, D all 65 536 u16 values; every op over every
    finite input (`f16`, `fp8`, four `f16_scaled` parameter sets up to their overflow bound, `exp` without and
    with `max`, `exp` of ±inf, `u16_shr` with shifts 8, 9, 15, 40, `u32_as_f32` incl. −0), permutations on first,
    middle and last axes, negative and zero strides, rank 4; W found at an even non-16 offset although an
    identical copy sits earlier at an odd offset, decoys (FNV hit with SHA miss) for W, D and K, K at the third of
    three gaps, a version resource; same output for 1, 3 and all threads.
  - **shipped-like cases**: the exact layout of `data/preset-M.v1.map.json` (every stage, tensor, base,
    stride, perm and op) with our own random source content and recomputed hashes and digests; W at 16-aligned,
    even and odd offsets.
    The shipped map is version 4: each case also has our own text of the map's `decoded_length` with our own
    numbers at the map's field offsets, compressed by the test's encoder (variant 1 at a multiple of 16, variant 2
    at an odd offset, no block).
  - **the shipped map itself** validates, and gives "no preset-M" / "not a model file" on fake files.
  - **source F** (08-T6 and more; `tests/synth.py` has an independent decoder, a greedy encoder of our own and a
    generator of made-up text with hexadecimal fields): variant 1 and variant 2 (blocks with the same first 64
    bytes, as in the shipped map) at aligned, even and odd offsets, both blocks in one file (map order decides), a
    decoy, the block as the last bytes of the file; no block, one bit flipped (first 64 bytes, middle, last byte),
    the last byte changed, the file cut inside the block, block hashes made with another key or without a key: all
    convert without the group, with the log line and the player line; a version-1 map on a file with the block and
    a version-4 map without F convert as before; converting again without the block removes the group and
    `raw/filter`; several files (a file without the block before one with it: the second is used, no note; with
    the block first: nothing after it is read; variant 2 before variant 1: the first; two without the block: both
    tried, the first written, the note; an installer error in a later file: nothing written); wrong `decoded_sha256`, wrong `filter` digest, shifted fields, wrong `decoded_length`, blocks
    that match their hashes but are malformed (distance 0, distance beyond the output, truncated, random bytes,
    too long, other text), a NaN field; ~55 invalid version-4 maps. Neither the output nor the log ever shows a
    weight or decoded text.
  - **error cases** of sections 8 and 9 (08-T3 and 08-T4 analogues and more): bit flips in W (start, middle, last
    byte), D, K (both words), truncation inside W and below the W length, empty, missing and directory paths;
    ~60 corrupted maps (every required field removed in turn, unknown op/locate/element/dtype/format/version, bad
    perm name, perm not dividing, non-permutation, shape/axes mismatch, wrong digest, bad hex, index above/below
    the source, names equal ignoring case, unsafe names, invalid JSON, missing map); non-finite decoded values
    (f16 of inf, fp8 of NaN, `f16_scaled` overflow, `exp` overflow, `u32_as_f32` NaN); several DLLs; output
    folder not writable. After every failure the earlier output must be byte-identical and no staging folder may
    remain.

  - **preset K** (map versions 2 and 3; the reference adds RFC 4648 decoding with `base64`, and `capped_softmax` in
    exact rational arithmetic with 50-digit `exp`):
    - **48-T5**: the spec's vectors (`data/k48-capped-softmax-test.json`) against the reference, and through
      the converter (bit for bit);
    - **K feature case**: segments listed in another order than their order in T, each with its own padding (none,
      `=`, `==`), junk text with `=` between them; `f16` of every finite binary16, `f16_scaled` with both preset-K
      scales, negative strides, `capped_softmax` on extreme inputs (±0, subnormals, ±65504, just above and below
      the cap), transposed and negative-stride groups; `i32` (negative values, both extremes) forwards and
      backwards; T after a decoy at an even non-16 offset, O after a decoy at an odd offset; 1, 5 and all threads;
    - **K shipped-like cases**: the exact layout of `data/preset-K.v1.map.json` (134 segments, eleven stages
      and `net`) with our own random T and O; T at 16-aligned, even and odd offsets;
    - the shipped K map validates (also read as version 2), and gives "no preset-K" / "not a model file" on fake
      files;
    - **K errors** (48-T3 / 48-T4 analogues and more): bit flips in T inside and outside the segments and in its
      first 64 bytes, in O, truncation inside T; ~40 corrupted maps (segment past the end of T, bad segment
      lengths, stride leading outside W, O index 16, version/feature mismatches, missing or extra sources, keyed T,
      unkeyed O, ops on the wrong source or dtype, `capped_softmax` shape and constants, wrong `net` digest);
      invalid Base64 and an odd total in a T whose hashes match; an index beyond the decoded W but within the
      pre-decoding bound; `capped_softmax` of +inf and a zero sum; a version-2 map converts.

  Every success case compares the container (header, key, map hash, index, digests, trailer) and every raw file
  byte for byte with the reference.

Run: `python3 tests/run_tests.py --exe <path to wronggpu-convert>` (≈25 s; `--quick` skips the large cases).
With MSVC 14.51 (Visual Studio 2026), CMake and NMake: the three ctest entries pass; `run_tests.py` reports 3663 checks.
An earlier state of the converter (before source F) passed the same unit and end-to-end tests as a MinGW
cross-compile under Wine. The additions for source F (2026-10-13) were built and tested with MSVC only: the
compiler commands of `build.bat` were checked with their output in another folder; `build.sh` (GCC, Clang) lists
the new file but was not run.
