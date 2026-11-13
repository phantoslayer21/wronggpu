# Performance milestone: enc0, dec0, exposure (branch clean/perf-enc, 2026-10-05)

Goal (results/2026-10-05-perf-targets.md, spec 10 section 9) at R 1706×960, F 1728×960, D 3840×2160, non-reset
frame: enc0 0.331 → ≤ 0.20 ms, dec0 0.152 → ≤ 0.12 ms, exposure 0.035 → ≤ 0.03 ms. Kernel interfaces, weight-blob
sizes and launch geometry are unchanged, so the runtime (clean/runtime) runs the new kernels as they are; only
`runtime/model.cpp` (`pack_dec0`) changed, together with `runner/dec0.py`, because the dec0 tables changed.

## Round 2 (after the spec team's timing of 1668645)

Measured on the RX 9070 XT (perf_bench.py): enc0 0.2531 ms (v1 0.3540), fragglobal 0.2362, reset frame 0.178;
ablations store −0.033, hist −0.028, lat −0.022, dis −0.013, depth −0.002, quant −0.001 ms. dec0 0.1337 (v1
0.1442), late 0.1321. exposure 0.0194. Real data: enc0 and dec0 accepted (03-T1/T2, 05-T1..T3).

**Reading:** enc0 and dec0 are now close to DRAM bandwidth, not to the vector rate. Bytes per frame at the in-game
size: enc0 reads colour 26.2 MB (RGBA f32), depth 6.6, motion 13.1, reduced history 8.3, latent 6.6, motion history
6.6 and writes features 26.5 + three motion surfaces 19.7: 113.6 MB, 0.176 ms at 645 GB/s (the store ablation,
−0.033 ms, is almost exactly 19.7 MB at that rate; the reset frame, 0.178 ms, moves 92 MB = 0.143 ms). dec0 moves
72.5 MB (features 26.5, motion 6.6, previous latent 6.6; A, B, C, latent 32.8), 0.112 ms; it runs at 85 % of that.
So the budgets (0.20 and 0.12 ms) are at 88 % and 93 % of the byte floors, and the remaining levers are bytes:

| Lever | Bytes saved | Where it has to change |
|---|---|---|
| colour copy in the runtime as RGBA16F (`colorh`) | 13.1 MB enc0 (also exposure and post read it) | runtime colour buffer (integration) |
| motion vectors as 2 × FP16 (`motionh`) | 6.6 MB enc0 | runtime motion buffer (integration) |
| A, B, C without padding (`pack12`) | 6.6 MB dec0 write + 6.6 MB post read | dec0 + output stage |
| dec1 + dec0 fused (the 64-channel FP8 map never goes to memory) | 26.5 MB write + 26.5 MB read | Swin dec1 kernel + dec0 |

Game colour and motion buffers are usually FP16 already; then colorh/motionh change nothing in the result.

Round-2 changes (all output-identical): **defaults** fragglobal (enc0) and late (dec0); enc0 motion surfaces are
stored with one 64-bit store per surface for both pixels of a lane; `#pragma clang fp contract(off)` in
`encoder.hip` (HIP compiles with `-ffp-contract=fast`, so whether LLVM fused a multiply and an add could depend on
the code around it: that is how `seq` gave 1020 different features. Now the rounding of every step is fixed by the
source, the variants are bit-identical by construction, and the GPU arithmetic matches the emulator's). New
variants and ablations as listed below.

## dec1 + dec0 fusion (prototype, round 3)

`kernels/dec1_dec0.hip`: kernel `swin_block_c64_add_dec0[_nc]` = the last Swin block of dec1
(`swin_block_c64_add[_nc]` of clean/swin 1d4babd, the version integration runs) followed by dec0 in the same
workgroups. The block's output q8(x2), the 64-channel FP8 map at F/2, stays in registers: per wave (one 4 × 4-token
window, 256-lane workgroups as in the block kernel's default geometry) the two lane halves swap two of their four
8-byte channel groups (dec1's layout L → dec0's K order; dec0's blob is unchanged), the block's 32 KB of LDS weight
buffers take dec0's tables and fragments after one barrier, and dec0's projection, previous-latent read, decode and
stores follow. The expand kernel before the block is unchanged. Grids match: 3240 workgroups for dec1's 25,920
windows and for dec0's 414,720 tokens. Saved per frame: the 26.5 MB write of the map by dec1 and its 26.5 MB read by
dec0, and one launch (≈ 0.08 ms of DRAM time at 645 GB/s). Costs: dec0's decode now runs at the block kernel's
occupancy (164 VGPRs, like the block's 166), and a wave's stores cover a 4 × 4-token window (8 row pieces of 32
bytes per image) instead of two whole rows; variant `fused_t` uses ordinary stores in case the non-temporal ones
handle the short pieces badly.

Bit-identical to the separate kernels (`tests/test_dec1_dec0_emu.py`: odd render sizes, padding, border windows,
all window offsets used, reset, clamped and `_nc` builds; and the separate path equals stage dec1 then stage dec0).
How it uses `swin.hip`: a template epilogue hook (`SwinStoreEpi` / `EPI::run`, 19 lines, default off; the existing
kernels compile to the same machine code) and a guard (`WG_SWIN_NO_EXPORTS`) so the file can include the templates;
the Swin session was told what the fused kernel depends on. `decoder.hip` was split into its steps (`dec_stage`,
`dec_motion`, `dec_project`, `prev_fetch`, `dec_store`); `coef_decoder` compiles to the same machine code.

Timing: `perf_bench.py` stage `dec1dec0` (default in `--stages`): expand + block + coef_decoder vs expand + fused
(and `fused_t`), per-kernel times, and output values that differ (must be 0). In the runtime the fused kernel would
replace dec1's block launch and the dec0 launch (dec1's output buffer and its debug view `dec1.out` are then not
written).

## How the time was located without a GPU

The clean team has no GPU, but clang 18 here compiles the kernels for gfx1201, so the instruction mix of every build
can be counted (static count per lane; the kernels are fully unrolled, so this is close to the dynamic count).
Cost model from the spec team's measurements (perf-targets, "Vector unit and overlap"): a stage takes its matrix time
plus its vector time; one vector instruction per wave per cycle, a transcendental (`v_exp_f32`, `v_log_f32`) about
4 times an FMA. Below, **cost = VALU instructions + 4 × transcendentals per lane** (a lane of enc0 or dec0 handles 2
pixels). The same compile-time switches that remove one part of the work (ablations) are built for the GPU, so the
spec team can measure what each part really costs (`perf_variants.py`).

Version 1 (as measured: 0.331 ms) at that cost model: 3739 per lane, i.e. about 0.28 ms of vector issue at full
occupancy plus about 0.02 ms of WMMA (6.8 GFLOP at 372 TFLOPS); the kernel was vector-bound. Where it went:

| Part of enc0 v1 (per lane) | VALU |
|---|---|
| reduced-history samples (8 per lane: 4 positions × 2 pixels) and their exp/log | ≈ 1540 |
| dithered FP8 quantization (40 values) | ≈ 900 |
| latent samples | ≈ 520 |
| 64-bit address arithmetic (inside the rows above) | ≈ 325 |
| disagreement (motion-history sample) | ≈ 230 |
| depth neighbourhood (18 single loads) | ≈ 250 |
| everything else (colour, positions, projection incl. 256 FP16 roundings) | ≈ 640 |

## enc0 (kernels/encoder.hip)

| Build | VGPRs (occupancy) | VALU | transc. | cost | loads | WMMA |
|---|---|---|---|---|---|---|
| v1 (kernels/baseline/encoder_v1.hip) | 69 (12) | 3499 | 60 | 3739 | 85 | 32 |
| **new, e_prev > 0, e ≠ e_prev (the usual frame)** | 71 (12) | 1815 | 68* | 2087 | 44 | 32 |
| new, e = e_prev | 71 (12) | 1715 | 19* | 1791 | 44 | 32 |
| new, e_prev ≤ 0 or NaN | 71 (12) | 1764 | 44* | 1940 | 44 | 32 |

\* static count; 6 of them (a second log for colour channels with e·c > 32752) are skipped unless a lane needs them.
Expected: about 0.17–0.19 ms if the kernel stays vector-bound (cost −44 %), less on frames with e = e_prev.

What changed (all exact up to FP32 rounding order; the reference is FP64):
1. **Dither in 3 integer operations.** Spec 03 draft 2 steps 1–5 (RN16, +u+δ rounded once, −u, truncation to 3
   mantissa bits) equal `(b + bit13(b) + 0xfff + ((t + 1) << 13)) & 0xfff00000` on the FP32 bits b of the value,
   after clamping to ±447.75 (derivation in `dither_q`). Checked exhaustively against the step-by-step definition
   (`tests/test_encoder_dither.py`; a C check over 6.7·10⁹ inputs found no difference). 900 → 110 VALU.
2. **Bilinear samples from two 64-bit loads** (`wg_mem.h`, `wg_footprint`): the position is clamped to the texel
   centres, which gives exactly the clamp-to-edge result, so one 8-byte load per row suffices; texel scale factors
   (1/1023, 2.45, log2(10)/0.36, 1/127) are folded into the weights. History positions use hw/dw (exact 0.5 at the
   in-game size) instead of 1/dw·hw, latent and motion-history positions p/k + j_prev − 0.5 instead of
   ((p/k + j_prev)/R)·R − 0.5: fewer roundings, slightly closer to the reference.
3. **32-bit offsets from uniform base pointers** for every access (global loads with an SGPR base): no 64-bit
   address arithmetic.
4. **The two pixels of a lane share a 3 × 4 depth window** (three 16-byte loads) and one 16-byte load of their
   centre motions.
5. **No branches in the per-pixel code**: invalid history positions are read at a clamped position and replaced by
   a select, so all loads of a pixel are in flight together; the history formula (e_prev > 0, e = e_prev,
   e_prev ≤ 0) is a template argument chosen once per frame. With e = e_prev the spec's history value simplifies
   exactly to G(h) = q = clamp(2.45·s, 10⁻⁴, 3) (1 + 100·e·h = 10^(q/0.36)), with no exp and no log.
6. **One logarithm per colour channel**: E(c) and G(c) are the same value unless e·c > 32752 (E's cap).
7. **Register pressure**: without care LLVM held all 32 A fragments (64 VGPRs) or both pixels' values at once
   (up to 143 VGPRs, occupancy 7). The pixels are processed one after the other with their lane exchange issued
   at once, an empty `asm volatile` pins the per-pixel results before the barrier (`wg_pin`), and the projection
   runs two output tiles at a time behind a compiler fence. Result: 71 VGPRs, occupancy 12, as in v1. (Round 2: the
   default reads the A fragments from global memory without LDS; clang 18 then schedules for ILP at about 112
   VGPRs; variant `w12` asks for 12 waves, 78 VGPRs.)
8. Stores: 32 contiguous feature bytes per lane as two 16-byte stores.

Accuracy (host emulator, synthetic data; features compared as FP8 values):

| Case set | Features differing from the FP64 reference: v1 | new | > 1 FP8 step: v1 | new |
|---|---|---|---|---|
| 60 random cases (sizes 22×18 … 70×40, reset, e_prev < 0, NaN exposure, NaN/inf motion) | 278 of 2,555,904 | 243 | 51 | 51 |
| unit-test cases (`test_encoder_emu.py`) | 48 of 467,840 | 10 | | |
| perf_bench check cases (6 cases up to 130×70) | 241 of 933,888 | 221 | 36 | 32 |

New vs v1 directly: 105 of 2,555,904 feature values differ (0.004 %), motion surfaces identical. On the GPU the
transcendentals are hardware approximations in both versions.

## dec0 (kernels/decoder.hip) — bit-identical to v1

| Build | VGPRs (occupancy) | VALU | transc. | cost | loads | stores |
|---|---|---|---|---|---|---|
| v1 | 79 (12) | 1047 | 17 | 1115 | 23 | 8 |
| **new** | 57 (16) | 802 | 17 | 870 | 18 | 8 (4 executed: 64/128-bit) |

v1 was more latency-bound than vector-bound (vector issue ≈ 0.06 ms of its 0.152 ms; memory floor ≈ 0.09–0.11 ms):
each wave waited for its feature token, then (after the barrier and the projection) for its motion, then for the
previous-latent texels. The first version of this rewrite loaded the motion and the previous-latent texels before
the barrier (now variant `early`; measured 0.1337 ms vs 0.1321 ms with the loads after the projection, so the
default keeps v1's order); both pixels of a lane are stored with one 64-bit (A, B, latent) or 128-bit (C) store;
occupancy rose from 12 to 16 waves. Vector work: the tables now hold the final A codes, C0/C1, sx, and z/g pairs
(`FORMATS.md`), the latent update runs in **packed FP16** (exactly the spec's per-step FP16 rounding, see the file
header), FP8 values are decoded with `v_cvt_f32_fp8`, and the padding slots of the last output tile are not
rounded. Emulator: 40 random cases (reset, borders, large motion, saturated logits) give identical A, B, C and
latent to v1.

## exposure (kernels/exposure.hip)

`exposure_partial` v1 had one load in flight per lane (a loop of ~13 dependent load → log → add steps per lane) and
an 8-barrier LDS tree. Now each lane issues all its loads (up to 4 columns × 4 rows) first, and the workgroup sum
is a lane-exchange reduction plus one barrier. Same launch (⌈n_y/4⌉ workgroups). Measured 0.0194 ms (budget 0.03).
(A single-launch variant with a last-workgroup sum measured 0.0372 ms and was removed.)

## Packed FP16

Used where it is exact (dec0's latent gate: products and sums of FP16 values, each rounded to FP16). Not used in
enc0: its remaining vector work is exact conversions (10-bit and s8 texel decode, the 256 FP16 roundings of the
projection's running sum that spec 03 prescribes), the FP32 history path, whose interpolated value is an exponent
(an error in it is multiplied by about 6.4 in G), and transcendentals; packed FP16 would save little there and
lose exactness. If enc0 misses 0.20 ms on the GPU, the next candidates are listed in QUESTIONS.md.

## Variants and ablations (`runner/perf_variants.py`, timed by `runner/perf_bench.py`)

| Stage | Variant | What it tells |
|---|---|---|
| enc0 | v1 | the previous version (same machine, same run) |
| enc0 | lds | A fragments staged in LDS with a barrier (the default of 1668645) |
| enc0 | w12 | `amdgpu_waves_per_eu(12)`: register target for 12 waves per SIMD (78 instead of ~112 VGPRs with clang 18) |
| enc0 | ntstore | non-temporal stores for m_far and m_center (read again only by the output stage) |
| enc0 | colorh, motionh, in16 | colour as RGBA16F, motion vectors as 2 × FP16, or both (19.7 MB less to read); compared with the default build on FP16-rounded inputs |
| enc0 | nopin, seq | without `wg_pin`; scheduling barrier between a lane's two pixels |
| enc0 | ab_hist, ab_lat, ab_dis, ab_depth, ab_quant, ab_proj, ab_store | timing-only: history samples, latent sample, disagreement, depth search, dither, projection (WMMA), motion stores removed |
| dec0 | v1, early | the previous version; previous-latent loads before the projection (the default of 1668645) |
| dec0 | fragglobal | A fragments from global memory, only the 5 KB of tables staged in LDS |
| dec0 | nt | non-temporal stores of A, B, C |
| dec0 | pack12 | A, B, C without padding bytes (12 instead of 16 bytes per pixel; a format change for the output stage) |
| dec0 | ab_prev, ab_soft, ab_proj, ab_store | timing-only: previous-latent read (and its use), softmax, projection, A/B/C stores removed |
| exposure | v1 | the previous version |

All non-ablation variants are checked bit-identical to the default build in the host emulator
(`tests/test_perf_variants_emu.py`; the FP16-input variants against the default build on rounded inputs, pack12 after
decoding).

Run (Windows, HIP SDK 7.2):

```
python clean\runner\build.py --rocm "C:\Program Files\AMD\ROCm\7.2" --out build --perf-variants
python clean\runner\perf_bench.py --build build --rocm "C:\Program Files\AMD\ROCm\7.2" --json perf.json
```

It prints the median time of every variant at the in-game size (non-reset frame; enc0 also a reset frame;
exposure also per kernel), how many output values of each non-ablation variant differ from the default build at
that size, and the spec 09 measures of the default build against the references on small synthetic cases. The real
data tests (03-T1/T2, 05-T1..T3, 07-T1) use the default build as before.
