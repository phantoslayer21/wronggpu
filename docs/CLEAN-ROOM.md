# How WrongGPU was written: the clean-room process

WrongGPU is an independent implementation of a temporal super-resolution network and of a ray-traced denoiser
(Ray Reconstruction) for AMD Radeon RX 9000 GPUs. It was written under a clean-room process with two separate teams.
This document describes that process and who wrote what. The specification itself is not published.

## The two teams

- **The spec team** worked out how the networks behave and wrote it down as a functional specification: the stages,
  the shapes and names of the tensors, the arithmetic and rounding of each step, tolerances, the game-facing behaviour
  and a test interface. It also made the converter maps and the test vectors in `data/` (see `data/README.md`), and ran
  the clean team's code on real data, reporting back numbers (errors, timings, pass / fail, compiler and runtime
  messages).
- **The clean team** wrote the code and the design documents under `clean/`, working only from the sources listed
  below.

## Who wrote what

- `clean/` (code, build scripts, tests, design and format notes): the clean team. Detailed per-file provenance records
  (what each file is based on and when it was written) are kept privately by the maintainer and are not published. A
  few launcher and front-end helpers (file helpers, the log file, the binary layout of the HIP loading code, the
  installer's resource layout) were adapted from WrongGPU's own earlier 0.2.0 code base after a file-by-file audit;
  that code is not included here (some comments name it `imported/0.2.0-infra`).
- `data/`: the spec team (maps and synthetic vectors). `data/06-vectors.md` and `data/motion-hi-vectors.md` are the
  spec team's own descriptions of their vector files, published word for word.
- `clean/third_party/`: AMD's FidelityFX SDK headers, unmodified (MIT); see THIRD-PARTY-NOTICES.
- The root documents (README, BUILDING, CONTRIBUTING, this file, ...) were written for the public release.

## What the clean team worked from

1. the specification pages and the spec team's written answers to its questions;
2. the spec team's numeric results;
3. public documentation: AMD's RDNA 4 ISA and WMMA documentation, HIP / ROCm documentation and headers, LLVM AMDGPU
   documentation, Microsoft's Direct3D 12 / DXIL / Win32 documentation, AMD's public FidelityFX API, and general
   machine-learning and numerical-methods literature;
4. its own design, measurements and probes on its own hardware;
5. for the Ray Reconstruction plugin's host interface: the plugin interface of the WrongGPU edition of OptiScaler (the
   project owner's GPL-licensed fork). `clean/rr_f/proxy/include/wg_host_menu_abi.h` declares the binary interface the
   plugin needs; no text was copied into it from any third-party file.

## Rules the clean team followed

1. Never look for, open or use the original upscaler's binaries (the one named in README.md), their disassembly or
   GPU assembly, its SDK sources beyond public documentation, leaked or decompiled upscaler code of any vendor, other
   projects' ports of these networks, or forks of WrongGPU.
2. Never read the model's learned weights. The unit tests and the test vectors in this repository use random or
   synthetic data against the clean team's own NumPy reference implementation of the specification. In some later
   checks (recorded in the maintainer's private provenance records), the
   clean team's built runtime and tools were run on game frames captured from WrongGPU's own runtime in a game and on a
   model file the converter had made from the vendor's DLL: the model file was handed to the runtime by path, no tool of
   the clean team read it, and only the runtime's output bytes, counters and pass / fail results were looked at. Those
   frames and that model file are not in this repository.
3. When the specification was unclear, ask the spec team in writing and wait for the written answer; never fill the
   gap from outside sources. (The questions and answers are internal and not published; the source code and the
   documents under `clean/` refer to them as `QUESTIONS.md` / `ANSWERS.md`, and to the specification as "spec NN" or
   "page NN".)
4. Record every file's basis (specification section, public document or own design) in the provenance records,
   which the maintainer keeps privately.
5. Third-party code only under a licence that allows redistribution, and only with the owner's approval: in this
   repository that is AMD's FidelityFX SDK headers (MIT), used as type declarations.
6. WrongGPU's own upscaler integration goes through AMD's public FidelityFX (FSR) interface, normally hosted by
   OptiScaler. One exception is documented in the source: for one game (Marvel's Spider-Man 2), when the host's own
   GPU-name setting makes DXGI report another adapter identity, the Ray Reconstruction plugin makes the Windows SetupAPI
   report that same identity, so the game does not see two different ones (`clean/rr_f/proxy/src/setupapi_identity.h`).

## Checks on every change

- **Leak scan.** Every commit was scanned for material that must not appear in the clean tree (vendor names outside
  the places that need them, large constant tables, oversized files). A commit with a hit was reworded, never
  respelled to pass the scan.
- **Release builds from a recorded commit.** Each release was built from a recorded commit, with a hash manifest of
  every output, and its packages were scanned before they were shared.

## Model data

WrongGPU never ships model data. The converter (`clean/converter`) reads the learned parameters from the upscaler file
the player already has on their PC, guided by a map in `data/` (block lengths, how each block is located, digests,
tensor shapes, offsets and strides, conversions, and a line naming the product and version the map reads; see
`data/README.md`), and writes WrongGPU's own model file on the player's PC.

## Test fixtures that come from real runs

- `clean/installer/tests/logs/`: inputs of the launcher's log-reading tests; `clean/installer/tests/logs/README.md`
  says which are synthetic and which are trimmed excerpts of WrongGPU's own logs (process ids, device pointers and
  incident ids neutralised).
- `clean/rr_f/proxy/tests/data/`: synthetic game-interface fixtures of the Ray Reconstruction plugin's tests (made-up
  matrices and call sequences in the schema the plugin's tools write).

## This repository

This repository is a snapshot of the source of WrongGPU 0.6.0, prepared for publication (one commit, no private
history). Left out: the
specification, the clean team's questions and answers, the spec team's results notes, the team's own tools (for
example `tools/gpu_lock.py`, `tools/nv_leak_scan.py`) and the team's own build-machine tooling, and scratch probes that
nothing builds.

## For contributors

The same rules apply to every contribution: see [CONTRIBUTING.md](../CONTRIBUTING.md).
