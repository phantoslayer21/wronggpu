# Behavioural equivalence tests: our code against the original (clean team → spec team, draft 2, 2026-11-04)

## What every test answers

Each test compares WrongGPU with the original model and ends in one of three verdicts, for every stage, every temporal
image, every probe and every interface call sequence, and once for the whole model:

| Verdict | Meaning | Rule (stage values; parts 2-4 add their own rules below) |
|---|---|---|
| **MATCHES** | our code computes what the original computes | 100 % of the values equal, or at least 99.99 % equal with every other value 1 step off and no bias (rounding ties) |
| **CLOSE** | the same computation, differing only in rounding | at least 99 % equal, every difference at most 2 steps, mean signed difference within ±0.01 steps, no channel, position class or value range with more than 3 times its share of the differences |
| **DIFFERENT** | something is computed differently from the original | anything else; reported with its size: the share of values off by more than 2 steps and by more than 16 steps, and the largest difference in steps, so that a small slip and a massively different stage are told apart at a glance |

The model's verdict is the worst verdict of any row of the verdict table (stages, temporal images, probes,
perturbations, sequences). A DIFFERENT stage is located by the report (which
stage, which outputs, where in the image, which value range, from which frame on) and is treated as an error in our
specification or our code until shown otherwise: the spec team reads that page against the original again, then the
specification, our reference and our kernel change together, and the test runs again.

The tests run on the spec team's machine, because only that side may hold the original. The clean side writes the
tools (part "Tools"), including the comparison tool that computes the verdicts, so every run gives the same verdicts
from the same rules.

## Why

Every exactness test on the clean side compares our kernels with our own reference implementation of our own
specification. Those tests prove that the code implements the specification; they cannot show that the specification
describes the original model correctly. If a specification page is wrong, kernel, reference and test agree with each
other and all pass. A whole-image PSNR on real frames is the only check against the original today, and an average
can hide a wrong step.

This protocol asks for evidence that WrongGPU behaves like the original model, measured by the spec team on its machine
with the original, in five parts. It is written for Model M and for RR preset F (mode Accurate, and mode Exact where
noted). It does not change any clean-room rule:

- The original model, its outputs and its intermediate values stay on the spec team's side. Nothing computed from the
  model's weights comes back to the clean side: no images, crops, tensors or individual values of the original.
- What comes back is numbers about the comparison only: counts, rates, histograms of differences, maxima and means of
  differences, signed-difference statistics, and locations (stage, frame, channel, pixel index) of differences.
- The clean side supplies the tools that produce our side's values and a comparison tool that prints only those numbers
  (part "Tools").

Wording used below: "step" = one representable value of the stage's storage format (one FP8 E4M3 code, one FP16
ulp, one level of an 8- or 10-bit unsigned or signed grid). "Equal" = the same stored value (FP8: the same code,
negative and positive zero counted equal).

## Part 1 · Stage-by-stage agreement on real frames

Capture, on the original, the values at every stage boundary of our specification for real game frames, and run the
same stage of WrongGPU on the same inputs. Two ways, both wanted:

- **Isolated:** each WrongGPU stage receives the original's own inputs to that stage. A difference then belongs to that
  stage alone (the specification of that stage, or our kernel).
- **Chained:** WrongGPU runs the whole frame from the game's inputs, and every stage boundary is compared. Shows how
  differences carry through the network.

Stage boundaries:

| Model | Boundaries (spec page) |
|---|---|
| Model M | exposure (07) · enc0 features and the three motion surfaces (03) · enc1, enc2 skip and merged (02) · b0 … b5 (02) · dec2, dec1 (02) · dec0 A, B, C, latent (05) · post: history, reduced history, display (06) · filter output (06). Spec 90 already exposes all of them (`Runner.run`, `Pipeline.frame(debug=True)`). |
| RR preset F | exposure (58) · input stage: features, dither codes, front-end MLP output, IN0 tokens, motion surfaces (52) · each of the sixteen body stages, outputs and hidden state (51) · head D0-R outputs (55) · output stage CH, Q, PH1 (54) · resolve outputs and PH2, TH, OH, AH (54b) · every temporal image after the frame: EX, PM pair, Pe, Pn, HID, CH pair, PH1, PH2. |

Report per stage and per output, over all compared frames:

1. values compared; share equal;
2. histogram of the difference in steps: 0, 1, 2, 3–4, 5–8, 9–16, more; and the share non-finite on either side;
3. largest difference in steps, and the stage-format value range it occurred in (for example "|value| in [1, 2)"),
   not the values themselves;
4. mean signed difference in steps, overall and per channel (bias), and the share of differences with each sign;
5. where the differences are: per channel, per image border band (8 pixels) vs interior, per value range (the format's
   binades), and for windowed stages per window position (border windows, shifted windows);
6. the first difference in compute order: stage, frame, channel, pixel (index only).

Verdict per stage and output, and per channel of every output with two or more channels: MATCHES / CLOSE /
DIFFERENT by the rules at the top (the per-channel rows stop one wrong channel of three from hiding under the 3-times
rule, which cannot flag a class holding a third of the values or more). The CLOSE band (2 steps at the top) is set per
stage and format by the spec team's table. Verdicts go to isolated stages and final outputs; chained intermediates get
every number of the list above but no verdict. In the isolated run the verdict
belongs to that stage alone; in the chained run a stage can only be as good as the stages before it (a CLOSE or
DIFFERENT stage that follows a DIFFERENT one is reported as "after a DIFFERENT stage").

## Part 2 · Long sequences: the temporal state

Run at least 600 consecutive frames per case on both, chained, and compare every temporal image after every frame
(Model M: full history, reduced history, latent, motion history; RR: the temporal images listed in part 1). Cases:
steady camera; continuous camera motion; a camera cut; a reset requested by the game every 100 frames; render-size
changes (dynamic resolution) and back; a pause (the same frame repeated); moving objects with disocclusion.

Report per temporal image: the part-1 numbers for frames 1, 10, 100, 300, 600, and the share of values equal and the
mean absolute difference in steps for every frame (one number pair per frame). Verdict per temporal image and case: the
worst per-frame verdict by the rules at the top, and DIFFERENT also when the mean absolute difference of the last 100
frames is more than 1.1 times that of frames 11 to 110 (the difference grows over time).

## Part 3 · Controlled probe inputs

Synthetic inputs that each isolate one behaviour, fed identically to both (generated by our tool from a seed, so both
sides use the same bytes). The game inputs are synthetic; the model is the original's on the spec team's side and the
converted player model on ours, as in parts 1 and 2.

| Probe | Behaviour it isolates |
|---|---|
| flat colour, then one step in brightness | exposure response, history adaptation speed |
| a single bright pixel, still, then moving 1 px per frame | spatial footprint, ghosting decay |
| horizontal, vertical and diagonal edges, still and moving at 0.25 / 1 / 4 px per frame | edge reconstruction, motion handling |
| a fine checkerboard and a fine stripe pattern under the game's jitter sequence | detail accumulation over the jitter cycle |
| a disoccluding object (foreground moving over background) | disocclusion handling, history rejection |
| motion vectors deliberately wrong in one region | robustness against bad vectors |
| reset on frame k; camera cut without reset | history invalidation |
| extreme exposure (very dark, very bright, pre-exposure ≠ 1) | value range handling |
| depth inverted and not inverted; subrect origin ≠ (0, 0) | input conventions |
| RR only: each guide input (albedos, normals / roughness, hit distance, disocclusion mask, colour guides, transparency layer and opacity) present vs absent; packed vs separate roughness | guide routing |

Report per probe the part-1 numbers of the final output and of the temporal images, per frame, for 64 frames, and the
probe's verdict: the worst per-frame verdict by the rules at the top.

## Part 4 · Response to small changes

For a set of real frames: change one input (one colour pixel by +10 %, one motion vector by +0.5 px, one depth value,
the exposure by +5 %, one guide pixel for RR), run one frame on both, and compare the change each implementation makes
to its own output (output after minus output before), never the outputs themselves.

Report per perturbation: the correlation of the two change images; the ratio of their total absolute change; the radius
(in output pixels) containing 99 % of each change; and the share of pixels where the change has opposite signs.
Verdict: MATCHES when the two change images are equal; CLOSE when correlation ≥ 0.99, ratio in [0.95, 1.05], the same
radius and under 0.1 % of pixels with opposite signs; DIFFERENT otherwise, with the four numbers as its size.

## Part 5 · Game-interface behaviour

Replay the call sequences captured by our probe builds in real games (the traffic profiles in
`clean/rr_f/proxy/tests/data/`, and new captures) against the original's interface and against ours, and compare what
each answers: every result code; the optimal-settings answer (sizes, sharpness, minimum, maximum) per quality mode and
output size; requirements and version answers; and when each implementation starts its history over (reset frames,
render-size changes, option changes). For the direct-path route (spec page 66), the same for create, evaluate and
release and the capability answers. Verdict per sequence: MATCHES when every answer is the same, DIFFERENT otherwise
(there is no CLOSE for answers), with the first differing call (call index and field name) and the number of differing
calls.

## Tools (clean side)

1. **Stage dumps.** Model M: `clean/equiv/model_m.py` over spec 90 (chained dumps of every boundary, full or
   tiles; one stage in isolation from input files; the D3D12 replay with RGBA16F outputs). RR: `clean/rr_f/gpu/sequence.py` gets a spec-90-style runner (every
   boundary of part 1 as NumPy arrays in the specification's logical layout, isolated and chained, with the converted
   player model).
2. **Probe generator.** A seeded generator of the part-3 inputs, the same bytes on both sides
   (`clean/equiv/probes.py`; the manifest carries a digest over every file, `--verify` checks a directory).
3. **Comparison tool.** Reads two sets of values (ours, and the original's saved on the spec team's side), computes the
   verdicts MATCHES / CLOSE / DIFFERENT by the rules of this document, prints the verdict table first (model, then every
   stage, temporal image, probe and sequence) and the numbers of parts 1–5 after it, and never prints a value of either
   side. Includes the first-difference finder of part 1 item 6. Its own tests run on synthetic pairs built to land in
   each verdict (equal, 1-step rounding noise, a biased rule, a wrong rule in one channel, a large difference).
   `clean/equiv/compare.py`; the choices this document leaves open (the bias test for 1-step ties, the smallest
   class judged, the size words minor / major / massive, ERROR rows) are listed in `clean/equiv/README.md`.
4. **Interface replay.** The proxy's capture replay (`tools/rr_capture_replay.py`, `tests/test_replay_profile.cpp`)
   extended to print the part-5 comparison for a trace of answers the spec team records from the original.

## Order of work

1. Part 1 isolated, Model M (tools exist) and RR Accurate, on frames the spec team already captured.
2. Part 2 on the same captures, then part 1 chained.
3. Part 5 (no model data involved; quick).
4. Parts 3 and 4.

## Questions for the spec team

1. Can the original's values be captured at all the boundaries of part 1? Where not, which boundaries are reachable?
2. Which real captures exist already (games, sizes, frame counts), for Model M and for RR?
3. Are the thresholds in parts 1, 2 and 4 reasonable, or should they follow spec 09 / page 51.8 tolerance classes?
4. Is any number in the report lists above still too close to model data for the clean-room rules?
5. The 3-times rule cannot flag a class holding a third of the values or more (for example one of three colour
   channels). Decided 2026-11-04: every channel gets its own verdict row; the 3-times rule stays.
