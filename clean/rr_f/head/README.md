# D0-R / output head reference (spec 55, with the page-51 block at level 0)

An independent NumPy reference of spec page 55 (draft 7): the recurrent half of decoder D0 (D0-R) and the network's
output head, including the common window block of page 51.3 (draft 6) at level 0, its window grid and borders
(51.6), and the decoder recurrent-half hidden-state rule of 51.4. Written from `spec/55-rr-f-final-head.md` and
`spec/51-rr-f-window-block.md` only. Everything is synthetic: no model data, no real converter constants.

| file | content |
|---|---|
| `exactfp.py` | exact scalar arithmetic: rounding of rationals to f16 / f32 with integers, IEEE specials, correctly rounded exp / 2^x / tanh / logistic (decimal module, 60 digits), the spec-literal sigmoid of 51.4 / 55.1, e4m3 Q8 by definition, the 8-bit store rules |
| `fp.py` | vectorised formats: direct f64 -> f16 rounding, `fma16` (round-to-odd sum), Q8 / d8, unorm8 / snorm8 store and read-back, the exhaustive sig / tanh tables, guarded correctly rounded f32 tanh / expf and the CANDW logistic, the f16-step metric |
| `block.py` | page 51.3 window block (vectorised over windows), 51.6 window grid / reflection, recurrent-half shift, the converter's attention-range / `w_max` check |
| `head.py` | page 55: converter scalars s1, s2; Cnet (55.2, 51.4 rule); HID, GATES, PREC, CANDW values and stored codes (55.3); `d0r_stage` = block + head. Block-output channel picks per the page 55 erratum (below) |
| `vectors.py` | vector generator / reader (`python -m clean.rr_f.head.vectors [--check]`) |
| `tests/slow_oracle.py` | element-by-element oracle (integer matrix steps, scalar rounding, explicit loops) |
| `tests/test_scalar_rules.py` | exhaustive scalar checks (all 65,536 f16 inputs where the domain is f16) |
| `tests/test_stage.py` | vectors vs reference recomputation, vs the slow oracle, and the page-55 properties |
| `tests/vectors/` | the test vectors and `manifest.json` (format, layouts, per-output comparison rule) |
| `QUESTIONS-55.md` | provisional readings |
| `DESIGN-55.md` | the D3D12 kernel: design, rounding points, measured results |
| `dx12/rr_d0r.hlsl`, `dx12/rr_host.cpp`, `dx12/run_dx12.py` | the kernel (block + head in one dispatch), its D3D12 test host, build / run / compare / timing |

## Page 55 erratum (block-output channel numbers)

Page 55 printed its block-output channel numbers in an internal ordering; the erratum
(`spec/55-rr-f-erratum-channel-numbers.md`) gives the natural (converter) numbers, internal c -> natural
16 ((c mod 4) div 2) + 8 (c div 16) + 2 ((c mod 16) div 4) + (c mod 2), as one simultaneous renaming:

| use | page 55 | erratum (default) |
|---|---|---|
| 55.2 gate logits Dh_0..3 | y0, y1, y4, y5 | y0, y1, y2, y3 |
| 55.3 HID values 0, 1 / 2, 3 (tanh) | y28 / y2 | y14 / y16 |
| 55.3 GATES values 0, 1, 2 | y8, y9, y12 | y4, y5, y6 |
| 55.3 PREC (Px, Py, rho logits) | y13, y16, y17 | y7, y8, y9 |
| 55.3 CANDW (d = a - b) | y20, y24 | y10, y12 |

One switch covers all of it ("page 55 erratum mode"): `head.py` `P55_ERRATUM` / `head_channels(erratum)` (the
functions take `erratum=False` for the page's numbers), `rr_d0r.hlsl` `RR_P55_ERRATUM` (1 default, 0 = the page's
numbers). The page numbering is kept for comparison only. The slow oracle (`tests/slow_oracle.py`) writes both tables
out independently (`CH_ERRATUM`, `CH_PAGE`). The test vectors use the erratum numbering (the head-edge case places its
edge values on the erratum channels).

## Running

NumPy is needed (on the owner's PC: `PYTHONPATH=<folder with the packages>`).

    python -m unittest clean.rr_f.head.tests.test_scalar_rules clean.rr_f.head.tests.test_stage
    python -m clean.rr_f.head.vectors --check       # vectors reproduce bit for bit

## Rounding model

f16 values are float64 arrays holding binary16 values. Products of two f16 values and the 32-input matrix steps
of 51.1 are exact in float64 (in any summation order); `fma16` rounds the exact sum to odd in float64 before the
single rounding to f16, so every `R(...)` of the spec is one correctly rounded step. f32 steps of page 55 are NumPy
float32 operations. The sigmoid and the HID tanh are 65,536-entry tables computed from the exact scalar evaluation;
rho's tanh, CANDW's logistic and the converter's expf are float64 evaluations whose rounding is checked against a
1e-12 band, falling back to the 60-digit evaluation when the band straddles a rounding boundary (so all three are
correctly rounded).

## Vectors

8 cases (`manifest.json` lists each with its inputs, constants, outputs and comparison rule):
`stage_basic`, `stage_border` (64 x 32 tensor, render 57 x 19), `stage_shift` (s1 capped), `stage_zovf` (rows with
Z = +inf), `stage_edge` (every finite e4m3 code: f16 overflow inside the block, +-inf / NaN logits), `stage_nan`
(all inputs NaN), `head_edge` (y2h given: edge values, CANDW tie / infinite / NaN pairs, half-way store values),
`head_random` (non-power-of-two s1, s2); plus `converter_scalars` and exhaustive `tables` (sig, tanh16, unorm8 /
snorm8 store of every f16, read-back of the 256 codes, Q8 of every f16). Files are plain lower-case hex words
(no binary, each file below 256 KB).

Tolerances in the manifest follow 55.3 / 55.5: Cnet, HID, GATES, PREC 1 f16 step (target bit-exact), CANDW 3 f16
steps (provisional) with the exact pair / d = 0 / infinite-logit requirements, codes = store rule applied to the
implementation's own f16 values. `y2h` and `Z` are diagnostics (bit-exact with the reference row sums).

## D3D12 kernel

    python -m clean.rr_f.head.dx12.run_dx12 build   --work <build>/rr-f-head/work
    python -m clean.rr_f.head.dx12.run_dx12 prepare --work <build>/rr-f-head/work
    python -m clean.rr_f.head.dx12.run_dx12 gpu --work <build>/rr-f-head/work
    python -m clean.rr_f.head.dx12.run_dx12 compare --work <build>/rr-f-head/work

Results and design: `DESIGN-55.md`.
