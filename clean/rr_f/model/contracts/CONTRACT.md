# Logical prepared-model admission contract

This freezes WrongGPU's own **synthetic known-subset preparation boundary** against approved `rr-spec` commit `4e61d0747211161483c7b2218347e1f892947158`. It does not establish a complete player-model format or approve a converter. The accompanying JSON contains no learned values, extraction offsets, signatures, build identifiers or source model mapping. The separate fixtures contain only original synthetic constants and constant-fill recipes.

Sources: approved pages 50, 51 draft 6, 55 draft 7 and 58 draft 6; exact hashes are in `prepared-model.contract.json`. Earlier `results/rr-f-contracts-20261018/logical-tensors.schema.json` supplies the already-expanded values-free 197 descriptors, checked against pages 51/55. No clean implementation, learned data, vendor binary, disassembly, personal memory or transcript was inspected. No GPU work or commits occurred. All files in this directory are original project-owned contract/fixture artifacts; this is their provenance record.

## Files and scope

- `prepared-model.contract.json`: normative semantic rules, source/own-policy classification, source pin, exact descriptors and blocking gates. This is a declarative contract, not JSON Schema.
- `prepared-model.schema.json`: genuine Draft 2020-12 structural schema for the synthetic manifest. Passing it does not establish semantic validity, uniqueness, actual buffer layout or numerical correctness.
- `synthetic-base.json`: fully specified synthetic preparation recipe for the known body/head and a declared synthetic IN0 boundary. Expanding constant fills yields 197 actual arrays; a descriptor alone is not a checked allocation.
- `synthetic-cases.json`: independently authored admissible/invalid cases, plus page-58 source counterexamples. Each case starts from a deep copy of the base and applies its named edits. No fixture claims to be a supported player model.

The manifest pins both `contract_id` and `source_head`. Fields are closed. IDs are unique; arrays' lengths never substitute for the required ID set. Reject unknown fields and forbidden physical metadata. Scalar bit strings encode numerical IEEE/e4m3 codes in lowercase hexadecimal, most significant nibble first; this textual representation imposes **no disk endianness**. f16 uses four digits, f32 eight, e4m3 two. No defaulting, implicit cast or bool-as-integer conversion is allowed. A validator may report multiple faults; a fixture's error identifies the principal semantic violation, not mandatory first-error order. A malformed complete-player-model request may return structural errors, but a structurally valid request must never reach admission.

## What the known inventory does and does not cover

There are 16 blocks. Their common nine tensors contribute 144 descriptors; four encoder merges add eight, E1 adapter adds two, five fusion up-projections add ten, ten hidden projections add twenty, eleven level/single gate tensors add eleven, and D0-R s1/s2 add two: **197**. The descriptor IDs, axes, dimensions and precision are mandatory.

IN0's block and hidden projection are included; its original front end is not. The synthetic boundary supplies IN0 block input and hold, explicitly substituting for absent page 52. Its presence cannot satisfy a real front-end requirement. Each recurrent decoder consumes its level's g; fusion has no separate g. D0-R has no Wh/bh. Shared query/key means no Wk or query/key bias. h0/h1 are converter inputs and are not extra prepared tensors. ra/m/f are symbols, not tensor-count additions.

Require 32 f16 ra/m symbols (separate pairs for every block), five independent decoder fusion f symbols, and ten globally consistent f16 window symbols. Lower clamp symbols are independently supplied, never negated from the upper clamp. Numeric original symbol values remain unknown. SH contains parity 0 and 1, each with exactly eleven named stage/level rows. Each component is 0, 2, 4 or 6. Recurrent halves derive `(s+4)%8`; row order does not imply original serialized order. No separate ARCH metadata is consumed in this synthetic schema.

Logical e4m3/f16/f32 precision is part of identity. Preserve Q8 signed zeros and its saturating nearest-even definition; IEEE binary16 keeps subnormals. The network's exact matrix step rounds once per ascending 32-input chunk. These runtime precision requirements accompany preparation but cannot be verified merely from metadata. Finite stored/prepared data does not imply finite runtime intermediates, which have explicit source behavior.

## Semantic admission

After structural checks, enforce IDs/shapes/types/axes, encoded bit widths and finite synthetic stored data, symbol completeness, SH completeness/domains, and exact source version. Finite stage/window symbols and finite ordered attention/activation clamp endpoints are explicitly **own conservative admission policy**, not invented assertions about every original model. No original bound on ra/m/f/g or activation coefficients is supplied.

Enumerate every binary16 value c in the inclusive attention clamp interval. Evaluate the exact source rounding points `p=R(c*R(A_P1-c*c)+A_P0)`; require `1 <= 32*(p-1) < 31` for every c. No sampled approximation is sufficient. Compute maximum w and record `64*w_max >= 65520`. A true overflow record is **admissible**: the source specifies zero normalized attention for a row whose sum overflows. Validate signed zeros numerically as appropriate while retaining the supplied bit encoding.

For the head, require explicit finite h0/h1 f16 inputs, exact widening, and correctly rounded f32 exponential evidence/computation. Match both resulting head tensors bit-for-bit. h0 may produce +infinity before the cap; that alone is not rejection. s2 must be finite; exponential underflow to zero is allowed. The synthetic interface requires an explicitly supplied finite **f32** S1CAP and finite s1/s2, but that cap type/domain is our test convention. Original S1CAP precision/domain/value remain a page-57 gate. Do not use an unspecified host expf as proof of correctly rounded conversion. A worker lacking a suitable reference must return an unevaluated numeric gate, not fabricate success.

The nine X symbols have the page-58 finite/normal-or-zero and relational checks. No sign bound for luminance weights or X_NRM is invented. Declared-domain validation is separate from proof that the entire exposure pass remains finite. In particular it must not silently strengthen the source checks to hide the following conflicts.

## Unresolved source constraints

ARCH is contradictory: page 50.3 says the converter does not read an architecture table, while 51.2 requires reading and cross-checking ARCH. Neither later wording nor the synthetic stage table authorizes choosing one. A sanitized explicit correction is required before original extraction/admission.

Page 58's finite claims do not follow from its published scalar-domain constraints. The fixture suite supplies three independently constructed finite-symbol counterexamples:

1. `X_KEY=0x45d1962c`, `X_MFL=0x3dd1a347`, dark input. Their exact ratio is `900167630848/13738823 < 65520`; even correctly rounded direct f32 division gives `65519.99609375 < 65520`. The specified separately rounded reciprocal and multiplication instead give 65520, whose f16 rounding is infinity. Thus the ratio bound alone does not guarantee finite E.
2. `X_LMAX=maxfinite_f32`, unit red weight, one maxfinite red pixel. The mean is maxfinite; its reciprocal becomes subnormal and is flushed to zero. E_raw is zero, contradicting the claimed strict positivity, although the prescribed E fallback handles it.
3. Dark input with key 4, floor 1, Ep 1, clamp [-2,2], X_NRM=maxfinite_f32, reset false. E is 4, the exposure log is 2, and the final product overflows. All X values satisfy the published domain, but X3 is infinite.

These are not original model values and do not establish failure for either supported build. Preserve each as `UNRESOLVED_X_OPERATIONAL_DOMAIN` diagnostic evidence. Do not replace IEEE results with finite clamps or call a newly strengthened policy original behavior. A future approved rule can add operational range checks with its own source/version.

## Decisions and completeness

`ADMISSIBLE_SYNTHETIC_KNOWN_SUBSET` means only that explicit original synthetic source plus known body/head inputs satisfy this preparation profile. `REJECTED_KNOWN_SUBSET` means an applicable known rule failed. `BLOCKED_PLAYER_MODEL` is unconditional for complete player-model admission under this version, regardless of all synthetic passes. Metadata conformance is not numerical end-to-end output correctness or GPU evidence.

Complete player-model acceptance still requires page 56's whole inventory/transforms, page 57's supported build identities and extraction mappings, IN0 front-end facts, resolved ARCH and S1CAP domains, and reconciliation of page-58 constraints. Logical axes do not disclose offsets, transposes, strides, alignment, serialization, physical SH order or packing. Do not invent any of these or fill missing symbols/tensors with zero, null or synthetic data. The `complete_player_model=false` flag is mandatory even for the fixture named fully prepared: it is fully prepared only for the explicitly synthetic known-subset harness.
