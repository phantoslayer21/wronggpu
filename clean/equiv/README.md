# Equivalence-test tools

Tools for `clean/EQUIVALENCE-PROTOCOL.md`. The comparisons against the original run on the spec team's machine. The
clean side runs these tools only on synthetic data.

| File | What it does |
|---|---|
| `compare.py` | The comparison tool. It reads our values and the original's, computes MATCHES / CLOSE / DIFFERENT by the protocol's rules and prints the verdict table first, then the numbers of parts 1–5. It never prints a value of either side. |
| `steps.py` | Steps per storage format (FP8 E4M3, FP16, FP32, integer and normalised grids). Each value maps to an ordinal, so a difference in steps is a difference of ordinals; ±0 count as equal. |
| `model_m.py` | Model M values from our side (Brief 5): chained stage dumps of every part-1 boundary, one stage in isolation from input files, and the D3D12 replay of a recorded input set. Everything is in the spec's storage formats, each file with a `.json` description. |
| `probes.py` | The seeded probe generator (part 3). It gives the same bytes for the same parameters on any machine, and `manifest.json` carries a digest over every file. `--verify DIR` regenerates the files and checks them. |

## Comparison tool

```
python -m clean.equiv.compare --manifest M.json --ours DIR --orig DIR [--json report.json]
```

The exit code is 0 when the model verdict is MATCHES, CLOSE or "numbers only", and 1 otherwise. `--json` also
writes the report (numbers only) as JSON.

### Comparison manifest

One JSON object. Every file name is relative to `--ours` and to `--orig`; both folders hold the same names, ours and
the original's. The arrays are `.npy` files (`allow_pickle` off); interface answers are `.json` files.

```
{
  "model": "Model M",                       // free text, printed in the report header
  "run": "isolated",                        // "isolated" or "chained" (see the verdict flags)
  "close_band": { ... },                    // optional, see "CLOSE band"
  "stages":        [ { ... }, ... ],        // part 1, in compute order
  "temporal":      [ { ... }, ... ],        // part 2
  "probes":        [ { ... }, ... ],        // part 3
  "perturbations": [ { ... }, ... ],        // part 4
  "sequences":     [ { ... }, ... ]         // part 5
}
```

Every list is optional. Fields of each entry (r = required):

| Field | In | Meaning |
|---|---|---|
| `name` (r) | all | Row name in the report. For a stage, it is also the key the CLOSE band is looked up by. |
| `file` (r) | stages, temporal, probes, sequences | The array (`.npy`) or, for sequences, the answer list (`.json`). |
| `format` (r) | stages, temporal, probes, perturbations | Storage format of the values (table below). |
| `layout` | stages, temporal, probes, perturbations | Axis letters in the array's order, each at most once: `F` frame, `H` row, `W` column, `C` channel. Default: `""` for stages (a 1-D array is read as `W`), `"FHWC"` for temporal images and probes, `"HWC"` for perturbations. Temporal images and probes need an `F` axis: they are judged frame by frame. |
| `border` | stages, temporal, probes | Width in pixels of the border band class. Default 8; 0 switches the class off (use it for tiles). |
| `window`, `shift` | stages, temporal, probes | `[wh, ww]` and `[sy, sx]` of a windowed stage. They add the classes "border windows / interior windows" and "window edge rows and columns / window interior". Default: none. |
| `close_steps` | stages, temporal, probes | This entry's CLOSE band in steps. It overrides `close_band`. |
| `final` | stages | `true` marks a final output, which is judged in a chained run. |
| `verdict` | stages | `true` / `false`: judge this entry or not, overriding the run's rule. |
| `case` | temporal | Case name, shown as `name / case`. |
| `before`, `after` (r) | perturbations | Output files before and after the change. The same names exist on both sides. |
| `center` (r) | perturbations | `[y, x]`: the output pixel the change is centred on. The 99 % radius is measured from it. |

Formats (`clean/equiv/steps.py`):
- `fp8`: FP8 E4M3 codes as uint8, or floats that are exactly representable.
- `fp16`: float16, or the uint16 bit patterns.
- `fp32`: float32, or the uint32 bit patterns.
- `int`: any integer codes.
- `unorm8`, `unorm10`, `unorm16`, `snorm8`, `snorm10`, `snorm16`: integer codes, or floats exactly on the grid.

A value that is not representable in the stated format makes the row ERROR.

**CLOSE band** (`close_band`): an object whose keys are stage names or shell patterns (`*`, `?`, `[...]`; for example
`"b*"`, `"enc?.skip"`, `"*"`). Each value is either one number of steps, or an object of format → steps:

```
"close_band": {"enc0.features": {"fp8": 3}, "b*": 4, "post.*": {"fp16": 5}, "*": {"unorm8": 1}}
```

An entry's band is found in this order:
1. its own `close_steps`;
2. the key equal to its `name`, if that value is a number or names the entry's format;
3. the pattern keys in manifest order, first match that is a number or names the entry's format;
4. otherwise 2 steps.

The band replaces "2 steps" in the CLOSE rule. DIFFERENT's size then reports the share off by more than the band.

**Verdict flags.**
- `"run": "isolated"`: every stage gets a verdict.
- `"run": "chained"`: stages get every number but the verdict "numbers only", which stays outside the model verdict.
  The exception is entries with `"final": true`. `"verdict": true` or `false` overrides this per entry in either kind
  of run.
- Temporal images, probes, perturbations and sequences always get verdicts.
- The model verdict is the worst verdict of all judged rows. ERROR, a row that could not be compared (a missing file,
  different shapes, values not in the format), outranks every verdict. When no row is judged, the model verdict is
  "numbers only".

**Channel rows.** Every stage, temporal image and probe whose layout has a `C` axis of 2 or more channels also gets one
row per channel, named `name / ch k`. Each is judged on that channel alone by the same rules and band. In a chained
run the channel rows of an intermediate stage are numbers only too. The verdict table groups an output's channel rows
by verdict (for example `enc0.features / channels 0-2, 4-63 MATCHES` and `enc0.features / ch 3 DIFFERENT (minor)`).
The NUMBERS part lists only channels that are not MATCHES, and chained channels only when they differ.

**Sequences** (`.json`): a list of answer records, one object per call, in call order. Records are compared field by
field. The report gives the first differing call index and field name (or "call missing on our / the original's
side") and the number of differing calls. There is no CLOSE: MATCHES when every record is equal, DIFFERENT otherwise.

- **CLOSE band per stage:** the manifest's `"close_band"` table maps a stage name, or a shell pattern such as `"b*"`,
  to a band in steps, either as one number or per format (`{"fp8": 2, "fp16": 3}`). An entry's `"close_steps"`
  overrides it; the default is 2. DIFFERENT's size then counts the share off by more than the band. The spec team
  sends the table back.
- **Chained runs:** intermediate stages get every number but no verdict ("numbers only", outside the model verdict).
  Verdicts go to isolated stages and to final outputs (`"final": true`). `"verdict": true` / `false` overrides this
  per entry.
- **Non-finite values** (spec team confirmation): NaN against NaN and an infinity against the same infinity count as
  equal, and the non-finite share is reported separately. FP8 steps count across zero with ±0 as one value, and the
  FP8 NaN code is non-finite, never a step.

What it prints: counts, shares, step histograms, the mean signed and mean absolute difference in steps, sign shares,
class shares, the largest step and the value-range class it occurred in (for example "|value| in [2^-3, 2^-2)"),
first-difference indices, perturbation statistics, and the field names of differing interface answers. The JSON report
holds the same numbers.

Tool choices that the protocol leaves open:

- **No bias** for 1-step ties means |n+ − n−| ≤ 3.29 √(n+ + n−), the two-sided 0.1 % level.
- **Classes:** a class is judged only when it has at least 10 differences.
- **Size word** for DIFFERENT:
  - minor: no value is more than 16 steps off, and under 1 % are more than 2 steps off;
  - massive: at least 10 % are more than 16 steps off;
  - major: everything else.
- **Size word** for perturbations:
  - minor: correlation ≥ 0.95 and ratio in [0.8, 1.25];
  - massive: correlation < 0.5, ratio outside [0.5, 2], or a missing radius;
  - major: everything else.
- **ERROR:** a row that could not be compared (a missing file, different shapes, a value not representable in the
  stated format) is ERROR. ERROR outranks every verdict in the model's verdict.
- **Compute order** inside a stage is frame, row, column, channel.

Per-channel verdicts: the 3-times rule can never flag a class holding a third of the values or more, so one wrong
channel of three would pass. So every stage, temporal image and probe with two or more channels also gets one verdict
row per channel, judged by the same rules on that channel alone (project decision 2026-11-04, protocol question 5).
The 3-times rule stays as well. The verdict table groups the channel rows of an output by verdict (for example
"enc0 / channels 0-2, 4-63 MATCHES" and "enc0 / ch 3 DIFFERENT (minor)"). NUMBERS lists only the channels that are
not MATCHES. In a chained run, an output with a DIFFERENT channel counts as DIFFERENT for the stages after it.

## Probe generator

```
python -m clean.equiv.probes --out DIR --model m|rr [--render 1707x960] [--output 2560x1440] [--frames 64] [--seed 1]
python -m clean.equiv.probes --verify DIR
python -m clean.equiv.probes --list --model rr
```

There are 26 cases for Model M. RR adds 10 cases (all guides, each guide absent, packed roughness) and the 33
colour-guide cases below. The input
conventions (motion-vector sign and units, jitter sequence, depth direction, subrect fill, guide formats) are written
into every manifest. The spec team maps them to the original's inputs.

Determinism comes from integer hashing (SplitMix64 finaliser) for noise, exact `Fraction` Halton jitter, and only
IEEE +, −, ×, ÷ in float64 with correctly rounded casts to float16 or float32. The test pins the digest of a small
configuration. Any change to the generated bytes must bump `VERSION` and the pinned digest together.

Size: at 1707x960 one frame of colour, depth and motion is about 23 MB, so a 64-frame case is about 1.5 GB and all
26 Model M cases are about 38 GB. The RR guides add more. Don't write full sets. A runner can call
`probes.frame_arrays` in memory, and `probes.generate(params)` (no output directory) returns the manifest and digest
without writing anything. Both sides can then compare digests instead of files.

### RR colour guides (page 52, version 2 of the generator)

33 more RR cases sit on top of `rr_guides_all`, which is the reference with every colour guide absent (absent = the
neutral 0):

- **Guides and formats:**
  - `color_before_transparency`: RGB of RGBA16F, or R11G11B10F packed in a uint32 (R bits 0–10, G 11–21, B 22–31,
    rounded to nearest even, negatives 0);
  - `sss_guide` and `dof_guide`: channel 0 of R16F, or R8_UNORM as uint8 codes;
  - `color_before_fog`: the red channel of RGBA16F.
  All are read with the colour input's origin.
- **Value classes**, each in the main format: typical, negative (own scale, results clamp), zero, small (binary16
  subnormals), very large, and NaN in one block (NaN reads as 0). Each other format gets one typical case.
- **Own origin:** each guide also has a case with a declared own origin of (5, 3), different from the colour's. The
  texture is padded right / bottom with 7.0 (255 for unorm), and the data stays at the colour's origin. The manifest
  says `expect_same_output_as` the typical case.
- **Fog over subsurface:** `rr_cg_fog_and_sss` has both attached; fog wins, so its output must equal the fog-only case.
- **Optional:** `rr_opt_spec_mv` adds specular motion vectors (channels 0–1, read with the motion vectors' origin).
  Without them, the specular hit distance of every RR case triggers the second-motion reconstruction. That needs the
  camera matrices, which every RR case record now carries (static camera; the convention is written in the record).

The version-1 cases are byte-identical: their digest alone is still pinned in the test.

## Model M values from our side (`model_m.py`, Brief 5)

The spec team runs this on its machine with the player's model. The clean side runs it only on synthetic models and
inputs. `--build` is a `clean/runner/build.py` output and `--model` the converter's output folder, as in spec 90.

```
python -m clean.equiv.model_m dump   --build B --model MODEL --render 1706x960 --output 2560x1440 --inputs FRAMES ^
       (--full --frames 2 | --tiles tiles.json) [--state] [--depth-inverted] [--auto-exposure] --out DUMP
python -m clean.equiv.model_m inputs --dump DUMP --frame 1 --stage dec2 --out dec2.json
python -m clean.equiv.model_m stage  --build B --model MODEL --spec dec2.json --out ISO
python -m clean.equiv.model_m replay --build B --model MODEL --render 1706x960 --output 2560x1440 --inputs FRAMES ^
       --out REPLAY [--then "python cmp.py {path} {frame}"] --delete-after
```

**(a) Chained dumps.**
- **What runs:** `Pipeline.frame(debug=True)`, HIP by default; `--backend d3d12` uses the native executor.
- **What is written:** one `.npy` per part-1 boundary per frame in `DUMP/fNNNN/`. The boundaries are exposure,
  enc0 features and the three motions, enc1 / enc2 skip and merged, b0–b5, dec2, dec1, dec0 A / B / C / latent, post
  history / reduced / display, and the filter output; `--state` adds the motion history.
- **Storage formats:**
  - FP8 codes as uint8, FP16 as float16;
  - the 8-bit unsigned grid as uint8 codes k (value k/255), the 8-bit signed grid as int8 codes k (k/127), the 10-bit
    grid as uint16 codes k (k/1023);
  - the filter output as float32.
  The order is natural: [H, W, C].
- **Description file:** each `.npy` has a `.json` with boundary, spec page, format, dtype, shape, layout, frame, frame
  counter n, parity, reset, render / feature / reconstruction / output size, the Swin window offsets, jitter and the
  tile.
- **Fused post kernels:** they write no display image. The missing boundary is listed in `manifest.json`; use
  `--kernels post=post_recon2_r`.
- **Comparison manifest:** `compare-manifest.json` feeds `compare.py` directly. It is a chained run, so intermediates
  get numbers only and `filter.out` is marked final.
- **Tiles:** `{"units": "render pixels", "tiles": [[x, y, w, h], ...]}`. Each boundary writes `<boundary>.tKK.npy`,
  with the rectangle mapped to its own grid:
  - feature grid: `[x // s, ceil((x + w) / s))` with s = feature width / grid width;
  - render size: as given;
  - reconstruction and output size: scaled by size / render size.
  The exposure is always whole.
- **Full images:** `--full --frames 2`. At 1706x960 one frame is about 350 MB, mostly the post / filter images.

**(b) Isolated.**
- **What runs:** one stage from a stage-input file, through spec 90's `Runner.run`.
- **Stage-input file:**
  `{"stage": "dec2", "inputs": {"x": "b5.out.npy", "skip": "enc2.skip.npy"}, "offset": [ox, oy] (or "parity": p),
  "scalars": {...}, "meta": {...}}`.
  - Paths are relative to the file.
  - The format of each input comes from its `.json` description, or inline as `{"file": ..., "format": ...}`.
  - Input keys and scalars are spec 90's: `x` / `skip` for the Swin stages; the `x` dict keys for enc0 / dec0 / post /
    filter; `color`, `pre_exposure`, `alpha` for the exposure.
- **Outputs:** `<stage>.<output>.npy`, in the same formats, with descriptions.
- **Self-check helper:** `inputs` writes the file for a Swin stage or the filter from our own chained dump, so our
  isolated output must equal our chained output (tested).

**(c) Replay.**
- **What runs:** `Pipeline.frame` on the native D3D12 executor (`--backend d3d12`) for every frame of the input set.
- **Output:** `fNNNN.rgba16f.npy`, [oh, ow, 4] float16 with alpha 1.
- **Callback:** `--then` runs a command per frame (`{path}`, `{frame}`; no shell; a non-zero exit stops the replay).
- **Disk:** `--delete-after` removes each output after the command, so the peak disk use is one frame (about 30 MB at
  2560x1440).
- **Input reader:** `--reader npz` (default) reads a folder of one `.npz` per frame, sorted, with `clean/runner/frame_ab.py`'s
  keys; `--reader module:function` plugs in another layout. The 10-17 capture layout reader follows once its
  conventions are relayed.

**Input reader** (`--reader module:function`, dump and replay). The function is imported (`module` must be importable,
for example a `.py` file in the current folder or on `PYTHONPATH`) and called once:

```
def frames(path: str) -> Iterable[dict]      # path = the --inputs value, as given
```

It yields one dict per frame, in frame order. Keys of each dict (spec 90's whole-frame input, before the input
conventions below are applied):

| Key | Type | Required | Meaning |
|---|---|---|---|
| `color` | float32 array [rh, rw, 3] | yes | Linear HDR colour (render size, or the texture size with `--subrect`). |
| `depth` | float32 array [rh, rw] | yes | Depth, in the direction set by `--depth`. |
| `motion` | float32 array [rh, rw, 2] | yes | Raw motion vectors, in the units and sign set by `--mv-units` / `--mv-sign`. |
| `jitter` | 2 numbers | yes | Jitter (x, y), in the units set by `--jitter-units`. |
| `mv_scale` | 2 numbers | no | Raw motion → render pixels, used with `--mv-units frame` (the default); 1, 1 when absent. |
| `mv_bias` | 2 numbers | no | Added before scaling (spec 90); 0, 0 when absent (or `--mv-bias`). |
| `exposure` | float | no | The frame's exposure (used with `--auto-exposure off`); 1.0 when absent. |
| `pre_exposure` | float | no | 1.0 when absent. |
| `reset` | bool | no | The game asks for the history to start over on this frame; false when absent. |

The render size may change between frames, up to `--render`. A generator works, so frames can be produced one at a
time without holding a whole set in memory. The default reader `npz` is this function over a folder of `.npz` files
with these keys, in sorted name order. In `.npz` files the scalars may be 0-d or 1-element arrays.

**Input conventions** (dump and replay). These flags turn a recorded set's conventions into spec 90's frame dict.
The defaults take the frames as given (spec 90). The settings used are written into every dump manifest and replay
description as `input_conventions`.

| Flag | Default | Meaning |
|---|---|---|
| `--mv-units frame\|render\|output\|uv` | `frame` | Unit of the raw motion. `frame` uses the frame's own `mv_scale` (1, 1 when absent). `render` means render pixels; `output` means output pixels (scale render / output); `uv` means fractions of the render size (scale = render size). The result is spec 90's `mv_scale`. |
| `--mv-sign X,Y` | `1,1` | ±1 per axis, multiplied into the scale: flips motion that points the other way than spec 03's convention. |
| `--mv-bias X,Y` | the frame's, else `0,0` | spec 90's `mv_bias`, added before scaling. |
| `--jitter-units render\|output\|uv` | `render` | Converted to render pixels. |
| `--jitter-sign X,Y` | `1,1` | ±1 per axis. |
| `--depth normal\|inverted` | `normal` (0 near, 1 far) | `--depth-inverted` is the same as `--depth inverted`. |
| `--auto-exposure off\|on` | `off` (the frame's exposure) | A bare `--auto-exposure` means on. |
| `--subrect X,Y[,W,H]` | none (origin 0,0, whole frame) | Colour, depth and motion are cut to the rectangle. Without W,H the size is `--render`, clipped to the texture. Units and jitter use the cut size. |

A leading minus needs `=`, for example `--mv-sign=-1,1`.

Tests: `python -m unittest clean.tests.test_equiv_compare clean.tests.test_equiv_probes clean.tests.test_equiv_model_m`
(CPU only).
The model_m GPU smoke test runs only with `WG_EQUIV_GPU_BUILD` and `WG_EQUIV_GPU_MODEL` set (synthetic model only).
Read the caution below before setting them.

## Caution: the GPU smoke test

The model_m GPU smoke test (`GpuSmokeTest`) loads the HIP runtime and runs the runtime on the GPU. Run it only while
you watch the PC, never unattended; run HIP and D3D12 as separate steps, and stop a step that makes no progress for
about a minute. On one run the display went black and the cause was not found. Skipping it is fine: the CPU tests
cover the tools' logic.
