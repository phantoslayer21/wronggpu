# Contributing to WrongGPU

Thank you for helping. WrongGPU is a clean-room implementation (see [docs/CLEAN-ROOM.md](docs/CLEAN-ROOM.md)), and that
only holds if every contribution is clean too. Please read this whole page before you open a pull request.

## The clean-room statement

Every contributor makes this statement, once, in their first pull request (copy it into the description), and it
applies to every later contribution. In it, "the original upscaler" means NVIDIA DLSS (super resolution and Ray
Reconstruction), together with its SDK (NGX, Streamline), its runtime and its model files.

> I have never seen, and while working on WrongGPU I will not look at, any of the following: the original upscaler's
> binaries or their disassembly, decompilation or GPU assembly; its source code or SDK internals that are not public
> documentation; leaked or decompiled upscaler or denoiser code of any vendor; another project's port or
> reimplementation of the original upscaler's networks or their kernels. I have not used the original upscaler's
> learned weights, or data computed from them, to write or test my contribution. My contribution is my own work, or third-party code under a
> licence listed in CONTRIBUTING.md, and I have the right to submit it under the MIT licence.

If you cannot make this statement, please do not contribute code; bug reports and test results from your own games
are still welcome.

## Sign-off (Developer Certificate of Origin)

Every commit carries a sign-off line, which certifies the [Developer Certificate of Origin 1.1](https://developercertificate.org/):

```
Signed-off-by: Your Name <you@example.com>
```

`git commit -s` adds it. Pull requests with unsigned commits are not merged.

## What you may use

- Public documentation: AMD's RDNA 4 ISA and WMMA documentation, HIP / ROCm documentation and headers, LLVM AMDGPU
  documentation, Microsoft's Direct3D 12 / DXIL / Win32 documentation, AMD's public FidelityFX API, papers and
  general literature.
- Your own measurements on your own hardware with synthetic data.
- Third-party code only under one of these licences: MIT, BSD-2-Clause, BSD-3-Clause, Apache-2.0, zlib, ISC, Boost
  (BSL-1.0). Nothing under GPL, LGPL, AGPL or a licence without the right to redistribute in source and binary form;
  nothing that forbids reverse engineering. Code that ends up in a shipped binary needs a maintainer's approval first.
  List every new component in [THIRD-PARTY-NOTICES](THIRD-PARTY-NOTICES) and in `REUSE.toml`.

Never add: files of the original upscaler of any kind (DLLs, headers, model data), model weights or data derived from them, game files,
screenshots or frame captures from games, or binaries.

## Every change

- An `SPDX-License-Identifier: MIT` line at the top of every new source file (files that cannot carry one, such as
  JSON, are listed in `REUSE.toml`).
- In the pull request, say for each new or substantially changed file what it is based on (public document, test
  result or own design).
- Tests with synthetic data for what you change; the CPU tests of [BUILDING.md](BUILDING.md) pass.
- No absolute paths of your PC, user names or machine names in code, scripts or documents. Tools are found through the
  environment variables in BUILDING.md.
- Kernels: if a change is meant to keep the output bit-identical, say so and show the comparison (BUILDING.md, section
  6).

## Bug reports

Use the issue templates. For a visual problem in a game: the game, the scene, your settings, your GPU and driver, and
the zip from the launcher's **Collect logs**. Please do not attach the upscaler's DLLs or other copyrighted game files.
