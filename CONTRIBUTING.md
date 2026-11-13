# Contributing to WrongGPU

WrongGPU is a community project: anyone can propose changes by pull request. It is also a clean-room implementation
(see [docs/CLEAN-ROOM.md](docs/CLEAN-ROOM.md)), and that only holds if every contribution is clean too. Please read
this whole page before you open a pull request. Ideas and plans: [ROADMAP.md](ROADMAP.md). Game results:
[COMPATIBILITY.md](COMPATIBILITY.md). New here? Read the [welcome discussion](https://github.com/phantoslayer21/wronggpu/discussions/1);
a good first task is [building from a fresh clone](https://github.com/phantoslayer21/wronggpu/issues/2).

## How a pull request gets merged

1. You open a pull request against `main`, with the clean-room statement (first pull request) and every commit signed
   off (see below). The pull-request template has checkboxes for both.
2. **CI** runs on a Windows runner (`.github/workflows/ci.yml`); it must pass:
   - licence check: [REUSE](https://reuse.software) lint (every file has its licence and copyright information);
   - line endings: files are stored exactly as committed (`.gitattributes` is `* -text`; no CR LF);
   - CPU builds and tests: the converter, the shared settings, the FSR front end, the launcher core and the Ray
     Reconstruction plugin with their CPU tests, and the CPU Python test set (`tools/ci/run_python_tests.py`).
   CI has no GPU: kernels and the runtime library are not built there, and no GPU test runs.
3. **Checks by a maintainer, by hand, before merge:**
   - **Clean-room gate:** every pull request is scanned on the maintainer's side before it is merged (a private scan
     for material that must not appear in a clean implementation), and the clean-room statement is checked.
   - **Review** by someone who did not write the change.
   - **A GPU test on real hardware** (a Radeon RX 9000-series card) when the change touches kernels, the runtime,
     the front end or the Ray Reconstruction plugin.
4. The project owner merges. Releases are built, tested on real hardware and published by the owner (see
   [RELEASING.md](RELEASING.md)).

## The clean-room statement

Every contributor makes this statement, once, in their first pull request (tick the box in the template; it then
applies to every later contribution). In it, "the original upscaler" means NVIDIA DLSS (super resolution and Ray
Reconstruction), together with its SDK (NGX, Streamline), its runtime and its model files.

> I have never seen, and while working on WrongGPU I will not look at, any of the following: the original upscaler's
> binaries or their disassembly, decompilation or GPU assembly; its source code or SDK internals that are not public
> documentation; leaked or decompiled upscaler or denoiser code of any vendor; another project's port or
> reimplementation of the original upscaler's networks or their kernels. I have not used the original upscaler's
> learned weights, or data computed from them, to write or test my contribution. My contribution is my own work, or
> third-party code under a licence listed in CONTRIBUTING.md, and I have the right to submit it under the MIT licence.

If you cannot make this statement, please do not contribute code; bug reports, game reports and test results from your
own PC are still very welcome.

## Sign-off (Developer Certificate of Origin)

Every commit carries a sign-off line, which certifies the [Developer Certificate of Origin 1.1](https://developercertificate.org/):

```
Signed-off-by: Your Name <you@example.com>
```

`git commit -s` adds it. Pull requests with commits that are not signed off are not merged.

## What you may use

- Public documentation: AMD's RDNA 4 ISA and WMMA documentation, HIP / ROCm documentation and headers, LLVM AMDGPU
  documentation, Microsoft's Direct3D 12 / DXIL / Win32 documentation, AMD's public FidelityFX API, papers and
  general literature.
- Your own measurements on your own hardware with synthetic data.
- Third-party code only under one of these licences: MIT, BSD-2-Clause, BSD-3-Clause, Apache-2.0, zlib, ISC, Boost
  (BSL-1.0). Nothing under GPL, LGPL, AGPL or a licence without the right to redistribute in source and binary form;
  nothing that forbids reverse engineering. Code that ends up in a shipped binary needs a maintainer's approval first.
  List every new component in [THIRD-PARTY-NOTICES](THIRD-PARTY-NOTICES) and in `REUSE.toml`.

Never add: files of the original upscaler of any kind (DLLs, headers, model data), model weights or data derived from
them, game files, screenshots or frame captures from games, or binaries.

## Every change

- An SPDX licence line for MIT (`SPDX-License-Identifier` followed by `MIT`, as at the top of every source file) in
  every new source file; files that cannot carry one, such as JSON, are covered in `REUSE.toml`. CI's REUSE lint fails
  otherwise.
- In the pull request, say for each new or substantially changed file what it is based on (public document, test
  result or own design).
- Tests with synthetic data for what you change; the CPU tests pass (CI runs them; locally: [BUILDING.md](BUILDING.md)).
- No absolute paths of your PC, user names or machine names in code, scripts or documents. Tools are found through the
  environment variables in BUILDING.md.
- Kernels: if a change is meant to keep the output bit-identical, say so and show the comparison (BUILDING.md, section
  6).
- Numbers in documents (timings, quality figures): only with the measurement they come from (game, resolution, mode,
  build, the log or result file). No number without its record.

## Bug reports and game reports

Use the issue templates: **Bug report** for problems (label `bug`), **Game compatibility report** for how a game
behaves (label `game-compatibility`) (see
[COMPATIBILITY.md](COMPATIBILITY.md), "How to add your game"). For a visual problem: the game, the scene, your settings,
your GPU and driver, and the zip from the launcher's **Collect logs** (check it first: it can contain your Windows user
name). Please do not attach the upscaler's DLLs or other copyrighted game files. Questions and discussion: the
[WrongGPU Discord server](https://discord.gg/745z9MfKwm).
