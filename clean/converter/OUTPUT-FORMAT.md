# Converter output format (`model.wgcm`)

Written by `wronggpu-convert` into the `--out` folder. One file holds every stage of spec 08 section 7.2 (preset M:
the thirteen stages and, when the DLL has a known output-filter block, the group `filter`; see "The optional group
`filter`" below) or every output group of spec 48 section 7 (preset K: the eleven stages and `net`),
so replacing it is a single rename (spec 08 section 8: all or nothing). The runtime (spec 10 section 7) reads
this file and packs the tensors into the kernels' formats (clean/FORMATS.md on the kernel branches).

All integers are little-endian. Offsets are from the start of the file.

| Offset | Size | Field |
|---|---|---|
| 0 | 8 | magic `WGCMODEL` (ASCII) |
| 8 | 4 | container version: `1` for map versions 1-4 (presets M and K), `2` for map version 5 (preset F) |
| 12 | 4 | number of stages S: the groups that are in this file (an absent optional group is not counted) |
| 16 | 32 | the key of spec 08 section 4.1 (SHA-256(label ‖ W)), so the runtime can recheck the stage digests |
| 48 | 32 | SHA-256 of the map file the converter used |
| 80 | 8 | index size I in bytes |
| 88 | I | index (below) |
| … | … | tensor data; each tensor starts at a multiple of 64, gaps are zero bytes |
| end − 32 | 32 | SHA-256 of every byte before it (whole-file integrity check without the key) |

Index: for each stage, in map order:

| Size | Field |
|---|---|
| 2 + n | name length n (u16), name (UTF-8, `[A-Za-z0-9_.-]`) |
| 32 | stage digest (keyed SHA-256 of the stage's tensors, spec 08 section 7.3; equals the map's; for `filter`, the map's entry for the variant that was found) |
| 4 | number of tensors T (u32) |

then for each of its T tensors, in map order:

| Size | Field |
|---|---|
| 2 + n | name length n (u16), name |
| 1 | dtype: 0 = float32, 1 = uint8, 2 = int32 (spec 48 `net/window_offsets`); container version 2 adds 3 = float16, 4 = uint16, 5 = uint32, 6 = e4m3 (spec 56.4.1, 57.7.3) |
| 1 | rank r |
| 4·r | axis lengths (u32 each) |
| 8 | data offset (u64, multiple of 64) |
| 8 | data size in bytes (u64) = product of the axis lengths × 4 (float32, int32, uint32), × 2 (float16, uint16) or × 1 (uint8, e4m3) |

Tensor data is row-major (last axis fastest), little-endian binary32 for float32, little-endian two's-complement
32-bit for int32. The stage digest is the
keyed SHA-256 of the concatenation of that stage's tensor data in index order, with the stored key, exactly as in
spec 08 section 7.3; a runtime that wants to verify the content recomputes it.

`--raw` additionally writes `raw/<stage>/<name>.bin` with the same bytes as each tensor's data (test aid for the
spec team). `raw/filter/output_taps.bin` exists only when the group does; a `raw` folder from an earlier run is
replaced as a whole, so no stale `raw/filter` stays behind.

## The optional group `filter` (preset M, map version 4; spec 08 sections 4.3 and 7.2)

The container layout and the container version (`1`) are unchanged: `filter` is one more stage entry of the index,
written exactly like every other stage. Nothing else in the file says whether it is there.

**Present** (a variant of source F was found in the DLL). The index has, at the group's place in map order (in
the shipped map: last, after `post`), this entry:

| Field | Value |
|---|---|
| name | `filter` |
| stage digest | keyed SHA-256 of the 20 data bytes; equal to **one** entry of the map's `digest` list for this stage (entry *i* for variant *i* of `sources.F.variants`, counted from 0) |
| number of tensors | 1 |
| tensor name | `output_taps` |
| dtype, rank, axis lengths | 0 (float32), 1, `5` |
| data offset, data size | a multiple of 64; 20 |

The data is five little-endian binary32 numbers: element 0 is w₋₂ and element 4 is w₊₂ of the output filter of
spec 06, for an even output index (spec 08 section 7.2). S counts the entry (14 for the shipped preset-M map).

**Absent** (no variant of F in the DLL; or the file was written with the version-1 map, before the group
existed). There is no index entry named `filter`, no placeholder and no flag; S is one less (13 for the shipped
map); every other entry is the same as in a file that has the group. The runtime then uses its own filter (spec
06, "analytic"). The two kinds of absent differ only in the header's map hash (offset 48).

How to read it:

- Find the group by its **name** while walking the S index entries. Do not derive it from S or from a position,
  and do not treat an entry you do not know as an error if you only need the others.
- Which variant it is: the index *i* of the map's `digest` entry that equals the stored digest. The file does not
  store the variant number anywhere else, and the reader does not need it (the five numbers are the result).
- To verify: recompute the keyed SHA-256 (key at offset 16) over the 20 bytes, compare with the stored digest, and
  check that this digest is one of the entries of the map's list. In the version-4 map the `digest` of this stage
  is a JSON list of hex strings; the `digest` of every other stage is still one hex string.
- The header's map hash (offset 48) is the SHA-256 of the version-4 map file for files written with it, and of the
  version-1 map file for older files. `verify_model_file` of the converter library accepts both for the
  version-4 map (it compares stages and digests, not this hash) and reports `absent_groups` and `same_map`
  (`ModelFileInfo` in `src/api.h`).

## Preset F (map version 5, spec 57; container version 2)

The layout above is unchanged; only the container version (`2`) and the dtype codes 3-6 are new. Spec 57 section
7.3 leaves the container to the runtime (as 08 section 7.2); this is WrongGPU's choice:

- The index lists the produced groups in map order: `IN0`, `E1` … `E5`, `D4-F`, `D4-R`, …, `D0-F`, `D0-R`,
  `constants`, and `constants_b2` only when IMG-IN variant 1 (build B2) was found. **Whether `constants_b2` is
  present is the presence of its index entry**; a file without it (also one written before the group existed) is
  read as build B1 (spec 57 section 7.3, "Reading back").
- Each group entry keeps the digest the converter checked (the map's, keyed SHA-256 of the group's tensors in index
  order with the key at offset 16). A reader recomputes the digest of every group present, `constants_b2`
  included, and refuses the file on any mismatch (`clean/rr_f/model/model_file.py`, `verify_model_file` in
  `src/api.h`).
- Tensor bytes are the row-major little-endian bytes of the dtype: e4m3 as the code byte of spec 56.3.1, float16 and
  float32 as IEEE bit patterns, uint16 / uint32 as integers. The shapes are the map's (the combined axes of 56.4
  split as 57.7.1 says: Wm [4][C][Cn], Wa [4][32][64], Wup [Kc][4][C], bup [4][C]).
- The container records no decoded value elsewhere and no source bytes; the log line `attention_sum_may_overflow`
  of spec 57.6.4 is written by the converter, and a reader can recompute it from WMAX.
