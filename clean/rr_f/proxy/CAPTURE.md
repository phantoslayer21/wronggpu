# RR input capture (probe build only) and the offline runner

Purpose: judge WrongGPU's RR image quality on real game frames without the runtime being fast enough or game-ready.
The probe build (hook-mode ASI `WrongGPU-RR-probe.asi` or the file-replacement probe) copies every tagged RR input
of a few evaluates to files; `tools/rr_capture_replay.py` later feeds those frames through WrongGPU's RR runtime
offline and writes PNGs. The release build contains none of this.

## Switching the capture on

Off unless `wronggpu-rr-probe.ini` next to the game executable (for game A: `bin\x64`) says so:

```
capture=1
capture_start=300      ; RR evaluate number (1-based) of the first captured evaluate (default 300)
capture_count=8        ; evaluates captured (default 8)
capture_every=1        ; every n-th evaluate from capture_start (default 1 = consecutive, needed for temporal state)
capture_max_mb=2048    ; cap on the total readback memory of all captured evaluates (default 2048)
capture_read_delay=2   ; evaluates between the copy and the first readback attempt (default 2)
capture_on_file=1      ; diagnostics: capture_start is ignored; the window starts at the first evaluate after the trigger file appears
capture_trigger=<path> ; the trigger file (checked every 16th evaluate; removed when seen; default capture-now.txt in the game\'s working folder)
capture_shots=N        ; with capture_on_file: up to N windows (default 1), each started by a new trigger file once the previous window is complete
capture_dir=<path>     ; capture folder instead of <game exe dir>/wronggpu-rr-capture (an existing one is renamed as below)
```

`;` / `#` comments and `[sections]` are ignored; unknown keys and bad values are logged (`capture settings: ...`)
and leave the default. The settings are read once, at the first RR evaluate. Without the file, or without
`capture=1`, the probe logs `capture: off (...)` and does nothing else.

## What the probe does (src/probe.cpp, src/probe_gpu_d3d12.cpp)

For each evaluate in the window, on the evaluate's own command list, before the probe's upscale:

1. every input tag the evaluate sees (61.4 lookup: local, frame-scoped, viewport tag), the outputs (types 4, 67) left
   out, null-resource tags absent;
2. one readback buffer per evaluate holding subresource 0 of each input (`GetCopyableFootprints`); refused when it
   would exceed the remaining `capture_max_mb` (then no further evaluate is captured);
3. barriers declared state -> `COPY_SOURCE` (none when the declared state includes it), `CopyTextureRegion` per input
   (a resource tagged twice is copied once), barriers back to the declared state (61.4 step 7); nothing else is set
   on the list (no heap, root signature or pipeline), so it runs whatever preference bit 0 says;
4. a completion marker after the copies (`WriteBufferImmediate`, `MARKER_OUT`; without `ID3D12GraphicsCommandList2`
   a buffer copy after the barriers). No fence on the game's queue is used.

From `capture_read_delay` evaluates later the probe checks the marker at every evaluate; once it is there the data
is copied out, the readback buffer released and the files are queued to a writer thread (each file written under
`.part` and renamed when complete, `frame.json` last). A marker not seen within 240 more evaluates (or at
`<shutdown>`) drops that evaluate (the buffer is kept until the device changes). At `<shutdown>` pending copies are
read once more and the writer is flushed. Log lines: `capture: on ...`, `capture eval=N: copied K of K input(s)
(types ...), B bytes`, `capture eval=N: written frame_N (...)`, `capture: done ...`, `capture at shutdown: ...`.

Memory: game A at 1440p Quality (11 inputs at 1706 x 960) is about 83 MB per evaluate; 8 evaluates about 660 MB of
system memory for a moment and the same on disk. The readback / file copy stalls the game thread for a few
milliseconds per captured evaluate.

## Folder format

`<game exe dir>\wronggpu-rr-capture\` (an existing folder of an earlier run is renamed to
`wronggpu-rr-capture.<yyyymmdd-hhmmss>`, never deleted):

```
wronggpu-rr-capture\
  frame_000300\
    input_00.raw  input_01.raw  input_03.raw  ...   one file per input, named by buffer type
    frame.json                                     written last: the frame is complete when it exists
  frame_000301\ ...
```

`input_<type>.raw`: a 64-byte little-endian header, then `height` rows of `row_pitch` bytes (tight, no padding):

| offset | field |
|---|---|
| 0 | magic `WGRRCAP1` (8 bytes) |
| 8 | header size (64) |
| 12 | buffer type |
| 16 | DXGI format of the resource description |
| 20 / 24 | width / height of the resource (mip 0) |
| 28 | row pitch in this file (bytes) |
| 32 | the tag's format (0: not given) |
| 36 | declared resource state |
| 40 / 44 / 48 / 52 | tag extent left / top / width / height (0 x 0: whole resource) |
| 56 | tag lifetime |
| 60 | reserved (0) |

`frame.json`: `evaluate`, `frame_index`, `viewport`, `api_minor`, `preference_flags`, `render` (the colour's
rectangle), `output` (type 4: format, resource size, output rectangle, state; not captured), `options` (every RR
options field: mode, output size, sharpness, exposure, HDR, packing, world-to-view / view-to-world, alpha upscaling,
presets), `constants` (every constants field: the five matrices, jitter, motion-vector scale, camera vectors, planes,
the tri-states including reset), `camera` (jitter, view-to-clip, world-to-view, view-to-world, mv scale, reset as the
proxy hands them on), `inputs` (per file: type, file, format, size, row pitch, tag format, state, lifetime, extent,
tag size as the game filled it), `not_copied` (inputs that could not be copied, with the reason). Non-finite floats
are written as `NaN` / `Infinity` (read by Python's `json`).

## Offline runner (tools/rr_capture_replay.py)

```
python clean/rr_f/proxy/tools/rr_capture_replay.py <game exe dir>\wronggpu-rr-capture ^
    --model <model.wgcm> --out <dir> [--frames 300:307] [--head strict|fast] [--exposure 1]
    [--previews] [--runtime ...\wronggpu_rr_runtime.dll] [--host ...\rr_test_host.exe] [--no-gpu]
```

Defaults: the runtime and its test host of `clean/rr-f-rt-int` (`<build>\rr-f-rt-int\out`). The model
is only handed to the runtime by path (`WG_RR_MODEL`); the runner never opens it. The runner plays the release
proxy's part (creation decisions 61.3 from the first frame, routing 62.2, render size from the colour extent, ms,
camera, jitter, reset; formats the host cannot create converted to the values the API read gives: R32 typeless depth
as R32_FLOAT with the same bits, typeless families as their typed member, 24 / 16-bit unorm depth as float), runs all
frames on one runtime instance in evaluate order (temporal state carried; a warning when the evaluates are not
consecutive) on the GPU under `the GPU lock (WG_GPU_LOCK, BUILDING.md)`, and writes per frame `frame_<n>_input.png`, `frame_<n>_rr.png`,
`frame_<n>_side.png` (input bilinear to output size | RR), `frame_<n>_crop.png` (centre quarter at output pixels:
input nearest | RR), with `--previews` every input, and `summary.json`. One tonemap for every image:
`x * exposure`, Reinhard `x / (1 + x)` per channel, sRGB encoding; NaN / inf pixels magenta. `--crop X,Y,W,H` picks
another crop rectangle (both halves always show the same output rectangle); `--raw <evaluates>` also saves those
outputs as float16 `.npy`.

Convention switches (quality investigation, replay only; each default is the release proxy's behaviour and gives a
bit-identical frame file): `--layer`, `--g1`, `--normals`, `--roughness`, `--albedo`, `--hitdist`, `--colour-scale`,
`--depth-inverted`, `--mv-jittered`, `--mv-scale`, `--jitter-sign`, and the diagnostic `--colour-probe`; the
module docstring and `CONVENTIONS` give the values and the spec rule of each default. `tools/rr_replay_variants.py`
runs a list of variants on one capture and writes a contact sheet per variant and a metric table
(results/2026-10-21-rr-quality1.md).

Tests: `tests/test_capture_replay.py` (CPU unit tests; `--gpu` end to end with the synthetic model), the capture in
`tests/test_probe_core.cpp` (logic) and `tests/test_hook_dll.cpp` (the hook probe ASI on WARP with the debug layer).
