# WrongGPU RR runtime (`wronggpu_rr_runtime.dll`)

The runtime the RR proxy (`clean/rr_f/proxy`) loads from its own directory: it implements the owned adapter
`WrongGPU_RrRuntimeV1` of `clean/rr_f/proxy/include/wg_rr_adapter.h` and runs the preset-F frame of pages 50-58 on
D3D12 (RDNA 4), recorded into the game's command list. Since `clean/rr-f-speed1` the PH2 pool, input stage, body, output
stage and resolve run the speed tracks' kernels (same bits as the verified kernels; `WG_RR_PATHS=verified` at creation
runs the verified pool / output / resolve kernels instead, for A/B runs). Since branch `clean/rr-f-rt-int` it runs the integration
branch's frame (`clean/rr-f-int`, `gpu/sequence.py`): the real PH2 pool and extended resolve, the integration
exposure / input-stage kernels and the input stage's C++ host constants (`clean/rr_f/host/rr_input_host.h`, the only
implementation). Original WrongGPU code; clean side only.

## What one evaluate records (into the game's list)

| step | page | kernels | notes |
|---|---|---|---|
| game textures -> `NON_PIXEL_SHADER_RESOURCE`, outputs -> `UNORDERED_ACCESS` | 61.4 step 7 | barriers | from the declared states; restored afterwards; "not given" = no transition |
| zero-fill on creation / render-size change / TH, OH creation | 53.8, 53.9, 54b.6 | `rt.fill`, then once on creation `resolve.init` (exp2 table of the fast resolve) and `pool.tables` (the PH2 pool's exact tables, from the model's symbols) | CH pair, HID, PM pair, Pe, Pn, the output fast path's scratch (and the exposure results, PH1, PH2, AH, tables, the resolve fixup bitmap on creation; TH / OH when created) |
| copies of inputs only where a format needs one | 52.1.2 | `rt.ingest` | input stage: only a linear conversion (`WgRrImage.conversion` 1) gives an f32 copy; every other input is read through its SRV |
| PH2 quarter pool (non-reset frames) | 54b.5 | `pool.main` (exact tables; `resolve.pool` with `WG_RR_PATHS=verified`) | PH1 of the previous frame -> PH2 (RGB10A2 codes) |
| exposure | 58, 53.6 | `exposure.partial/finish` (`RR_EX_SRV=1`) | the colour read from the game's resource; the previous frame's E from its result (the exposure result pair, EX[1-par] / EX[par]) |
| input stage | 52 | `input.decode`, `input.clear` (replay list), `input.features` (certified fast features, `RR_IN_SRV=1`, 128 lanes); front-end MLP: `input.mlp.e` (variant E) with the default body, `input.mlp` (variant X, exact) with `WG_RR_BODY=Z` / `X`; `WG_RR_INPUT_MLP=E` / `X` overrides; `WG_RR_INPUT=lite` at creation (opt-in, not exact): `input.features.lite` (plain hardware f32 features, no replay marks), `input.replay` (exact, the listed tokens), `input.mlp` (step X, exact-sum fast path) | constant block from `host/rr_input_host.h`; E / Ep / X3 from the exposure result |
| body | 51 | default (clean/rr-f-speed5): `stagee.IN0` .. `stagee.D0-F` (step E, the default matrix step, not exact to 51.1; byte-identical to step Z on the 8 captured frames); `WG_RR_BODY=Z` (exact / verification mode): `stagez.*` + `stagefix.*` (step Z = step X bits); `WG_RR_BODY=X`: `stage.IN0` .. `stage.D0-F` (step X, exact row sums; the routes of `gpu/stage.py`: c32 / fast / wide kernels, E1 reading IN0's y2 directly), `regroup` only without the direct E1 | |
| final head | 55 | `head` (`RR_STRICT_STEP=1`, `RR_ZEXACT=1`) | per mode (see "Modes"): Accurate (default) and Fast run `head.fast.rx` (relaxed fp32, `RR_HVARL 7`, gate-checked, not exact); Exact runs `head.fast` (exact: the same bits as the exact 51.1 step); `WG_RR_HEAD=exact`: the exact 51.1 step |
| output stage | 54 | `output.prep/fast/fallback` (fast path; `output.main/pool` with `RR_RT=1` and `WG_RR_PATHS=verified`) | reads CH[par], PM[1-par], Pe, Pn as binary16; E / Ep / invE from the exposure result; writes CH[1-par], PH1, Q |
| resolve | 54b | `rt.ingest` (layer, opacity, alpha source in a 54b.N read format), `resolve.fast` (every format, `RF_SPEC=0`) + `resolve.fixup` (`resolve.main` with `WG_RR_PATHS=verified`) | plain, or extended (transparency layer with the colour-before-particles rule, opacity layer, alpha upscaling, separate output alpha); TH / OH / AH pairs |
| stores into the game's outputs | 54b.N | `rt.store` | the resolve's image (binary16 / binary32 / unorm8 / RGB10A2 per destination) through the typed UAV, or our texture + `CopyTextureRegion` (no UAV, sRGB) |
| timestamps, completion marker | - | `EndQuery`, `WriteBufferImmediate` | per-frame slots wait for the marker (no fences, no second queue) |

The page-53 host bookkeeping (par, fc, pj, CP, pending resets, rebuild flag) is the proxy's
(`proxy/src/lifecycle.h`); the runtime trusts the decided frame and re-creates the render-size items whenever the
render size differs from the one they were built for. All temporal items live at fixed offsets of one per-instance
arena (model image first, then the output-size items: CH pair, PH1, PH2, the exposure result pair, AH pair with alpha
upscaling, TH pair from the first frame with a layer, OH pair from the first frame with an opacity layer; then the
render-size items). The frame's work buffers are placed by their lifetimes (`WorkPlan` in `src/rr_runtime.cpp`:
buffers never live at the same pass share memory; every pass is followed by a UAV barrier on the arena).
`WG_RR_ALIAS=0` (test runs) places them one after the other so that every intermediate can be read back.

Log `wronggpu-rr.log` next to the library (the proxy's directory; the proxy appends to the same file): model,
"RR runtime ready", create / destroy, first evaluate (resolve variant), reset frames, render-size changes, TH / OH
creation, camera fallback (once per instance), failed evaluates, GPU time per 300 frames (`WG_RR_TIMING_WINDOW=N`: per N frames, tools/game_bench.py; pre, input, body, head,
output, resolve, store).

## Model

The runtime reads the converter's output itself: `model.wgcm` (`clean/converter`, container version 2, map version
5) next to the library (`WG_RR_MODEL` overrides; `wronggpu_rr_model.wgcm` and the own container
`wronggpu_rr_model.bin` are found too). Checks at load (57.7.3): the whole-file hash, the index bounds, the keyed
digest of every group (key at offset 16), the inventory of pages 56.4 / 57.7 (every tensor's shape and type);
`constants_b2` present selects build B2. `src/rr_pack.cpp` then arranges the logical tensors in the kernels' layouts
(the same device image, blob for blob, as `tools/pack_model.py pack()`; `tests/test_model_pack.py`) and computes rho
of 55.3 (binary64 tanh with the guard of `head/fp.py`; an undecided entry is evaluated in double-double). The
model-independent binary16 tables (51.4 sigmoid, the head's sigmoid / tanh) are generated at build time from the
clean reference and embedded. The own container `WGRRMOD1` (`tools/pack_model.py --synthetic / --admission`) remains
for synthetic test models.

## Files

| path | content |
|---|---|
| `src/rr_runtime.cpp` | adapter, device / pipelines / driver-channel self-test, instances, lifetime-placed work area, evaluate, transients, test hooks (`WrongGPU_RrRuntimeTestV1`, `WrongGPU_RrModelDumpV1`, `WrongGPU_RrRhoV1`) |
| `src/rr_model.*`, `src/rr_pack.cpp`, `src/rr_tables.h` | model loading: `model.wgcm` (checks, packing, rho) or `WGRRMOD1` |
| `../host/rr_input_host.h` | the input stage's constant block (52.1.1, 52.7.3, 52.8) |
| `kernels/rr_runtime.hlsl` | ingest, fill, store, self-test |
| `tools/build.py` | kernel library (exact arithmetic mode of `clean/dx12/build_dx12.py`), tables, DLL, test host |
| `tools/pack_model.py` | logical models (synthetic, admitted subset, converter output) and the own container |
| `tests/rr_test_host.cpp` | D3D12 host in the proxy's role (the proxy's own `Lifecycle`) |
| `tests/e2e.py` | scenarios, GPU runs (GPU lock), stage-by-stage CPU reference, lifecycle, aliasing, leak, timing |
| `tests/test_model_pack.py` | the runtime's `model.wgcm` packing against the Python packing; rho; damaged files (CPU only) |

Kernel changes outside this folder (opt-in, default variants unchanged): `kernels/rr_input.hlsl` `RR_IN_SRV=1`
(texture entries with word 7 set are read through an SRV array), `kernels/rr_exposure.hlsl` `RR_EX_SRV=1` (colour
argument 0: the SRV), `kernels/rr_output.hlsl` `RR_RT=1` (binary16 motion / history images),
`head/dx12/rr_d0r.hlsl` `RR_ZEXACT=1` (exact row sums); `host/rr_input_host.h` takes the proxy's ms when given.

## Modes (clean/rr-f-s10-modes): one library, `rr_mode = exact | accurate | fast`

`wronggpu_rr_runtime.dll` holds the kernels of all three modes; the mode is chosen when an instance is created.

| mode | head | input features | body | output stage | resolve |
|---|---|---|---|---|---|
| `exact` | exact | exact | exact arithmetic (`stageex.*`, front-end MLP separate) | exact | correctly rounded exp2 / reciprocal (`resolve.fast.x`, `resolve.fast2.x`) |
| `accurate` (default) | relaxed fp32, `RR_HVARL 7` (`head.fast.rx`; owner decision 2026-11-01, clean/rr-mp1-head: gate-checked; before: exact `head.fast`) | exact (history samples exact) | relaxed arithmetic (`stagee.*`, MLP folded into IN0) | exact | correctly rounded exp2 / reciprocal, as exact (`resolve.fast.x`, `resolve.fast2.x`; owner 2026-10-28, round 11) |
| `fast` | relaxed, `RR_HVARL 7` (`head.fast.rx`) | lite | relaxed (as accurate) | `OF_HWRCP` + `OF_REPROJ_MAX` (`output.*.f`) | hardware exp2 / reciprocal (`resolve.fast`, `resolve.fast2`) |

Exactness policy (owner, 2026-11-04): history, reprojection, candidates and blending stay exact in every mode except
Fast; the head is exact in Exact mode; Accurate may run the gate-checked relaxed head (owner decision 2026-11-01,
confirmed 2026-11-04). Exact here means exact with respect to our specification and our reference, not a claim about the
original.

- Where it is read: environment `WG_RR_MODE`, else `rr_mode` (or `mode`) in `wronggpu-rr-runtime.ini` next to the library, else
  `accurate`. The proxy's `rr_mode` in `wronggpu-rr.ini` reaches the runtime as `WG_RR_MODE` (set once at proxy start unless
  already set).
- When: at every instance creation (the runtime ini is re-read each time), not live inside a running instance. The runtime
  starts in the mode of its set-device call; that mode's kernels form the first compile pass (ready as fast as before), the
  other modes' alternative kernels follow in the background pass. An instance of another mode created before that pass
  finished passes frames through until it has (the opt-in variants' rule).
- Per-knob overrides (tests, environment, win over the mode): `WG_RR_INPUT` (exact | lite | split | copy), `WG_RR_HEADV`
  (exact | relaxed), `WG_RR_BODYV` (exact | relaxed), `WG_RR_OUTV` (default | fast), `WG_RR_RESV` (hw | exact), and the older
  `WG_RR_HEAD`, `WG_RR_BODY`, `WG_RR_PATHS`.
- Log: every creation writes `instance N: rr_mode=<mode> (<source>): head ..., input ..., body ..., output ..., resolve ...`.
- Build: `tools/build.py` builds each alternative exactly as the earlier separate builds did (`WG_RR_EZMODE=exact
  WG_RR_HEAD_DEFS={} WG_RR_RF_DEFINES={}` / `WG_RR_HEAD_DEFS={"RR_HVARL":7} WG_RR_OUT_DEFINES={"OF_HWRCP":1,"OF_REPROJ_MAX":1}`):
  the DXIL of each mode's kernels is byte-identical to the separate builds' (compare `kernels.json`).

## Exact tricks of round 10 (same bits in every mode)

- body (clean/rr-f-s10-body): D4-F / D4-R of the relaxed body (RR_B32) run on four groups per window (`RR_B32G4`, `WG_RR_EROUTE` build knob).
- wave64 forms (clean/rr-f-wave4b, same bits, design D of `kernels/rr_block_c32.hlsl`: one wave64 = two token tiles of the same window,
  every weight / bias / key-value fragment loaded once for both): `head.fast.rx.w64` (`head/dx12/rr_d0r_w64.hlsl`) and the relaxed
  body's `stagee.D1-R.w64` / `stagee.D1-F.w64` / `stagee.E1.w64` (`kernels/rr_block_c64_w64.hlsl`). `WG_RR_W4B` at creation (hex mask,
  default `0xf`; bit 0 head, bit 1 D1-R, bit 2 D1-F, bit 3 E1); a clear bit selects the wave32 kernel of the `WG_RR_W4` mask, so
  `WG_RR_W4B=0` is the wave-4 frame (A/B and byte checks); bit 0 does not depend on `WG_RR_W4` bit 0, bits 1-3 need its bit 1 (the
  default). Paired timing at 1440p Q (layer / plain, `fast`): `0xf` -178 / -175 us, bit 0 -71 / -71, bit 1 -45 / -44, bit 2 -42 / -39,
  bit 3 -25 / -23; all four byte-identical. The exact body (mode exact) and the dataflow forms keep wave32.
- wave-4c levers (clean/rr-f-wave4c, same bits by construction, GPU timing pending): the relaxed body's wide stages in wave64
  (`kernels/rr_block_wide_w64.hlsl`: `stagee.{E2, D2-F, D2-R, E3, D3-F, D3-R, D4-F, D4-R}.w64`, wave `2 g + tw` holds the two token tiles
  of the wave32 waves `4 g + 2 tw`, `4 g + 2 tw + 1`; E4 / E5 stay wave32), `stagee.IN0.w4c` / `stagee.D0-F.w4c` (the wave64 form of
  `rr_block_c32.hlsl` with the deferred Q8 decision `RR_DEFER`, the Dd / Wo weight prefetch and, for IN0, the interleaved front-end
  MLP), and `output.fast.f_ph1w` (OFX_PH1W alone, no OFX_LD3 / OFX_QST). `WG_RR_W4C` at creation (hex mask, default `0x19` = bits 0, 3, 4, -29 us at 1440p Q layer and plain; bits 1, 2, 5 are opt-in, they measured +23, +6 / -1 and +8 us; bit 0
  level 2, bit 1 level 3, bit 2 level-4 decoder, bit 3 IN0, bit 4 D0-F (needs `WG_RR_W4` bit 1), bit 5 `output.fast.f_ph1w` when
  `WG_RR_W4` bit 2 is off); `WG_RR_W4C=0` is the wave-4b frame. The exact body and the dataflow forms keep wave32.
- wave-5b/5c levers (clean/rr-f-wave5c, **rr_mode fast only, not byte-identical**; network/tail arithmetic relaxed while keeping temporal
  history/reprojection logic exact): `head.fast.rx.f.w64` (`head/dx12/rr_d0r_w64.hlsl`, `RR_HFAST_ALU=2`: bounded f32 ALU head tail,
  per-head Q8 cutting VGPR 96->84, branchless Q8, no dead CANDW store, exact Cnet history), `output.fast.w5` (`OFX_W5_ALU=1` on
  `output.fast.f0` base) & `output.fast.w5c` (`OFX_W5C=1, OFX_PH1W=1, OFX_LD3=1`: rolled `of_reproj` dropping VGPR 74->55 [+50% occ],
  7.2 KB LDS, BAR 4->2, zero fallback pass/barrier when tile fits), `resolve.fast2/3.w5` (`RFX_W5_ALU=1`) & `resolve.fast2/3.w5c`
  (`RFX_W5C=1`: `QN=28`/`QO=896`, wave-uniform `s1==3` skip; `fast2` adds `RFX_RP=16 -> RF_RS=1`, 12.8 KB LDS [5 WGs/CU], `RFX_LLDS=1`
  reading layer bilinear taps from LDS, `RFX_QLD=1`, 32-bit MV load, rolled 3x3), `input.features.w5c` (`input.features.lite` wave64 +
  `RR_IN_SRV_ONLY=1`: bit-identical to `input.features.lite`, CPU-guarded direct SRV, -17% INST / -65% VLD / -41% IMG, 57 VGPR, 5120 B LDS),
  `stagef.{IN0, D0-F, E1, D1-F, D1-R, E2, D2-F, D2-R}.w64` & `stagef.{E3, D3-F, D3-R, E4, E5, D4-F, D4-R}` (pure f32 `RR_B32=1` +
  branchless clamped `RR_Q8_FAST=1`, `RR_B32G4=1` + `RR_PRE=14` at L4/L5 eliminating all LDS/scratch spills, convergent merge guard
  `NAR=0`), and `pool.main.w5` (`RR_POOL_W5=1`: hardware f32 mad/mul with exact tables, 0 FP64, -69% INST, bit-identical to `quarter_pool`).
  `WG_RR_W5` at creation (hex mask, default `0xaff` = every bit but 8 and 10: bit 0 head, bit 1 output.w5, bit 2 resolve.w5, bit 3 input.w5c, bit 4 body L2,
  bit 5 body L0, bit 6 body L1, bit 7 body L3, bit 8 body L4/L5 [opt-in, see "Wave 5c results"], bit 9 pool.main.w5, bit 10 resolve.w5c [opt-in],
  bit 11 output.w5c); effective
  mask is 0 for exact and accurate (all 24 kernels are VG_OUT alternatives: only Fast compiles them in Pass 1), `WG_RR_W5=0` reproduces
  the wave-4c frame.
- tail (clean/rr-f-s10-tail): the resolve reads the exp2 table from the arena instead of copying it into LDS; 24-bit multiplies for
  addressing in the output stage; the main output's store is folded into resolve.fast2 / resolve.fixup (`WG_RR_FUSESTORE=0` at creation
  brings the separate store pass back). Also used by the exact-resolve (`_X`) kernels of mode exact.
- tail (clean/rr-f-300, same bits): the plain resolve (no layer, opacity or alpha upscaling) takes `resolve.fast3` / `resolve.fast3.x`
  (`RF_SPEC 3`: no render-size reads, no format switches; 43 / 55 VGPRs instead of resolve.fast's 162 / 191) and folds its store like
  resolve.fast2 (`WG_RR_RES3=0` at creation: resolve.fast / rt.store as before); with the store folded the fast resolve no longer also
  writes the main output's arena texels (kept for `WG_RR_ALIAS=0` test runs; `WG_RR_RESARENA=1` brings it back).

## Wave 5c results (clean/rr-f-wave5c, RX 9070 XT, D3D12, rr_mode fast; GPU-verified)

Default `WG_RR_W5=0xaff` (every bit but 8 and 10). Timings are the runtime's GPU time per frame (`WG_RR_TIMING_WINDOW`) on structured,
temporally coherent synthetic scenes through `rr_test_host` (plain: `perf`; layer: `perfcap`), paired ABBA, one host process per run
(warm-up run + measured run), no `dxc` / `clang++` running. A = the Wave 4c build (`13a53c9`), B = this build with no `WG_RR_W5` set.

1440p Quality (1707x960 -> 2560x1440), 6 ABBA rounds (12 runs per side), median:

| | Wave 4c | Wave 5c `0xaff` | delta | speedup |
|---|---|---|---|---|
| plain | 6.369 ms | **5.760 ms** | -609 us | x1.106 |
| layer | 6.600 ms | **5.963 ms** | -637 us | x1.107 |

Per-stage GPU ms at 1440p Q (mean of the measured runs; the body is one stage, the host has no per-level hook):

| | pre | input | body | head | output | resolve | total |
|---|---|---|---|---|---|---|---|
| plain, Wave 4c | 0.044 | 0.548 | 3.817 | 1.003 | 0.730 | 0.228 | 6.370 |
| plain, Wave 5b `0x077` | 0.043 | 0.550 | 3.558 | 0.780 | 0.666 | 0.212 | 5.810 |
| plain, Wave 5c `0xaff` | 0.045 | 0.536 | 3.554 | 0.790 | 0.639 | 0.200 | 5.764 |
| layer, Wave 4c | 0.044 | 0.563 | 3.829 | 1.003 | 0.727 | 0.444 | 6.610 |
| layer, Wave 5b `0x077` | 0.045 | 0.564 | 3.570 | 0.779 | 0.661 | 0.413 | 6.034 |
| layer, Wave 5c `0xaff` | 0.046 | 0.565 | 3.548 | 0.787 | 0.634 | 0.398 | 5.978 |

Scaling (medians of 3 ABBA rounds, `WG_RR_W5=0xaff`; the 1440p Q row is the 6-round default-build figure above):

| cell | plain 4c -> 5c (ms) | delta | layer 4c -> 5c (ms) | delta |
|---|---|---|---|---|
| 1080p Q (1280x720 -> 1920x1080) | 3.699 -> 3.343 | -356 us (x1.107) | 3.806 -> 3.446 | -360 us (x1.104) |
| 1440p Perf (1280x720 -> 2560x1440) | 4.144 -> 3.742 | -402 us (x1.107) | 4.328 -> 3.894 | -435 us (x1.112) |
| 1440p Q (1707x960 -> 2560x1440) | 6.369 -> 5.760 | -609 us (x1.106) | 6.600 -> 5.963 | -637 us (x1.107) |
| 1440p DLAA (2560x1440 -> 2560x1440) | 12.727 -> 11.632 | -1095 us (x1.094) | 13.104 -> 12.014 | -1090 us (x1.091) |
| 4K Q (2560x1440 -> 3840x2160) | 14.032 -> 12.794 | -1239 us (x1.097) | 14.609 -> 13.368 | -1241 us (x1.093) |

Per-bit timing at 1440p Q (3 ABBA rounds against `WG_RR_W5=0` in the same process; single bit, then cumulative / leave-one-out), us plain / layer:
bit 3 -12 / -2, bit 7 -40 / -41, bit 8 -52 / -43, bit 9 -4 / -9, bit 10 -15 / -1, bit 11 -96 / -96 on their own; `0x077` -537 / -562,
`0x07f` -552 / -567, `0x0ff` -584 / -608, `0x1ff` -626 / -647, `0x3ff` -639 / -660, `0x7ff` -637 / -662, `0xfff` -649 / -672,
`0xeff` -607 / -634, `0xaff` -607 / -635. Bit 10 is timing-neutral inside the default (`0xaff` vs `0xeff`, 6 rounds: +1.5 / +5.9 us and
-1.9 / +0.2 us in the two orders), so it stays opt-in.

Exactness against the Wave 4c build (same synthetic scenes, 12 small and medium configurations at binary16 and binary32 outputs, plus
1440p Q and 1080p Q, plain and layer, six frames with a reset on frame 3; every output and every history / arena byte):
- `exact`, `accurate`, and `fast` with `WG_RR_W5=0`: 0 byte differences everywhere, except one known pre-existing 4-byte difference in the
  work (scratch) region on 1440p Q reset-frame sequences (two atomic-append slots swapped; it also appears between two runs of the
  Wave 4c build itself, never in an output or a history buffer).
- Bits 3, 4, 5, 6, 7 and 9 are bit-identical to the Wave 4c frame (0 byte differences; `0x008`, `0x018`, `0x020`, `0x040`, `0x080`,
  `0x0f8` and `0x200` each checked against it on a 480x270 scene). Bit 11 writes the same bytes as bit 1 (`0xaff` vs `0x0ff`: 0 byte
  differences on every scene above), so the output stage differs from Wave 4c only by the Wave 5b `output.fast.w5` arithmetic. Bit 10 at
  1440p Q is bit-identical to Wave 4c on layer scenes and 81.8 dB (tone-mapped) from it on plain scenes.
- **Bit 8 (`0x100`, `stagef.{E4, E5, D4-F, D4-R}`) is not bit-identical** and is therefore opt-in: deterministic (two runs match), but
  34.7 .. 38.5 dB tone-mapped PSNR from the Wave 4c fast frame on a small scene, 37.0 / 36.9 dB on the 1440p Q reset frames and 37.2 dB
  (steady minimum, plain) against `accurate`, which fails the reset (>= 72 dB) and steady (>= 38.5 dB plain) gates.
- With no `WG_RR_W5` set, the output is byte-identical to `WG_RR_W5=0xaff`.

Quality of the default against `accurate` (1440p Q, six frames, reset on frames 0 and 3; tone map `x / (1 + x)` on the RGB, negatives
clamped to 0; PSNR over the RGB with peak 1; SSIM on the Rec. 709 luma of the tone-mapped image, 8x8 box window, L = 1, k1 = 0.01,
k2 = 0.03). 0 NaN / Inf in every output and history buffer of every frame (also at 1440p DLAA, with `0xaff`, `0xeff` and `0xfff`):

| | reset frames | steady frames (min .. max) | SSIM (steady min) |
|---|---|---|---|
| plain | 72.85 / 72.85 dB | 38.71 .. 38.87 dB | 0.9551 |
| layer | 73.91 / 73.93 dB | 43.15 .. 43.54 dB | 0.9763 |

Against the Wave 4c `fast` frame: plain 72.85 dB on the reset frames, 38.76 .. 38.87 steady; layer 73.91 dB, 43.12 .. 43.53. These are
the numbers of Wave 5b `0x077` (bit 3, 7, 9, 11 add no output difference). At 1080p Q the plain steady frames are 38.4 .. 39.2 dB and
the layer ones 42.8 .. 43.9 dB against `accurate`, the same as the Wave 4c fast frame itself (38.39 dB on one plain frame); at 1440p DLAA
Wave 5c is 72.5 .. 72.9 dB from the Wave 4c fast frame on the reset frames and 39.9 .. 40.2 dB (plain) / 44.3 .. 45.0 dB (layer) on the
steady ones; the Wave 4c fast frame is itself 42.8 dB (frame 0) and 38.6 .. 39.7 dB (the others) from `accurate` there, Wave 5c the same.

## Round 11, tail (clean/rr-f-s11-tail)

- Owner 2026-10-28: accurate's resolve is exact (`resolve.fast*.x`, as mode exact); new accurate byte reference `ref\MXR`.
- `WG_RR_RESGROUPS=N` at creation sets resolve.fast*'s persistent group count (tests / sweeps; every tile is computed
  alone, so any count gives the same bits; the round-11 sweep kept 512).
- History prep beside the previous frame's resolve (`WG_RR_TAILPREP=0` at creation: the front order): pool.main (PH1 -> PH2)
  and the PH1 / HID decodes run beside resolve.fast*, the PH2 decode beside resolve.fixup; the next frame skips its front
  pool and decodes when the key (arena, dec_* offsets, sizes, layout offsets) matches and it is a host-scheduled frame
  (not fresh / rebuild / TH or OH created / new arena / diagnostics). dec_* live through the whole frame (arena +37 MiB at 1080p
  Quality, +65 MiB at 1440p Quality). Same outputs and state; a dump after frame i holds the next frame's PH2 (e2e runs its
  stage checks with WG_RR_TAILPREP=0).

## Round 11 merged (clean/rr-f-s11-merged)

Tip of s10-merged plus clean/rr-f-s11-tail (exact resolve in accurate, history prep beside the previous frame's resolve).
Not merged: the body track (D0-F / E1 LDS cuts, wide4 for E3 / D3, per-stage RR_B32 off: none saves time) and the
right-sized resolve tile (same bits, no speed). One DLL, `rr_mode = exact | accurate | fast`; accurate is byte-identical to
the reference `ref\MXR` (MX with the exact resolve), exact to ACC, fast to FAST.

## Build and test

```
python -m clean.rr_f.runtime.tools.build                      # out: <build>\rr-f-rt-int\out
python -m clean.rr_f.runtime.tests.test_model_pack
python -m clean.rr_f.runtime.tests.e2e --calls 61 --perf 300 --jobs 14                  # synthetic model
python -m clean.rr_f.runtime.tests.e2e --model wgcm --wgcm <model.wgcm> --perf 0 --leak 0
```

Results: `results/2026-10-20-rr-rt-int.md` (earlier: `results/2026-10-20-rr-runtime.md`).
