# Contributing to WrongGPU

WrongGPU is a community project and a clean-room implementation. Every contribution must follow the rules in
[docs/CLEAN-ROOM.md](docs/CLEAN-ROOM.md). Read this page before opening a pull request.

Start with the [welcome discussion](https://github.com/phantoslayer21/wronggpu/discussions/1). You can propose a change
from [ROADMAP.md](ROADMAP.md), report a game through [COMPATIBILITY.md](COMPATIBILITY.md), or help with
[building from a fresh clone](https://github.com/phantoslayer21/wronggpu/issues/2).

## How a pull request gets merged

1. Open a pull request against `main`. Include the clean-room statement in your first pull request and sign off every
   commit. The pull-request template has checkboxes for both.
2. Wait for CI on the Windows runner defined in `.github/workflows/ci.yml`. It must pass these checks:
   - [REUSE](https://reuse.software) lint validates the licence and copyright data in `REUSE.toml` and checks that it covers every file.
   - The line-ending check preserves files exactly as committed. `.gitattributes` sets `* -text` and forbids CR LF.
   - CPU builds and tests cover the converter, shared settings, FSR front end, launcher core, and Ray Reconstruction
     plugin. CI also runs the CPU Python tests through `tools/ci/run_python_tests.py`.

   CI has no GPU. It does not build kernels or the runtime library, and it runs no GPU tests.
3. Wait for the maintainer's manual checks:
   - Before every merge, the maintainer privately scans the pull request for material that must not appear in a clean
     implementation and checks the clean-room statement.
   - Someone who did not write the change reviews it.
   - A change to kernels, the runtime, the front end, or the Ray Reconstruction plugin needs a test on a Radeon RX
     9000-series card.
4. The project owner merges the pull request. Only the owner builds, tests on real hardware, and publishes official
   releases. See [RELEASING.md](RELEASING.md).

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

Sign off every commit to certify the [Developer Certificate of Origin 1.1](https://developercertificate.org/):

```text
Signed-off-by: Your Name <you@example.com>
```

Use `git commit -s` to add the line. Pull requests with commits that are not signed off are not merged.

## What you may use

- Public documentation: AMD's RDNA 4 ISA and WMMA documentation, HIP and ROCm documentation and headers, LLVM AMDGPU
  documentation, Microsoft's Direct3D 12, DXIL, and Win32 documentation, AMD's public FidelityFX API, papers, and
  general literature.
- Your own measurements on your own hardware with synthetic data.
- Third-party code under MIT, BSD-2-Clause, BSD-3-Clause, Apache-2.0, zlib, ISC, or Boost (BSL-1.0). Do not use code
  under GPL, LGPL, AGPL, or a licence that does not allow source and binary redistribution. Do not use code whose
  licence forbids reverse engineering. Get a maintainer's approval before adding code that ends up in a shipped
  binary. List every new component in [THIRD-PARTY-NOTICES](THIRD-PARTY-NOTICES) and `REUSE.toml`.

Never add files from the original upscaler, including DLLs, headers, and model data. Never add model weights, data
derived from them, game files, game screenshots, game frame captures, or binaries.

## What every change needs

- Add an MIT SPDX licence line to every new source file. Use `SPDX-License-Identifier` followed by `MIT`, as in the
  existing source files. Cover files that cannot carry a header, such as JSON, in `REUSE.toml`. The `**` rule in
  `REUSE.toml` already covers every file, so REUSE lint passes without a header. Reviewers catch a missing header
  instead.
- In the pull request, identify the basis of each new or substantially changed file. Name the public document, test
  result, or your own design.
- Add tests with synthetic data for the changed behavior. Run the CPU tests locally with
  [BUILDING.md](BUILDING.md#6-tests-cpu). CI runs the same tests.
- Keep absolute local paths, user names, and machine names out of code, scripts, and documents. Use the environment
  variables in [BUILDING.md](BUILDING.md#2-where-the-build-finds-the-tools) to find tools.
- If a kernel change should keep the output bit-identical, say so and show the comparison described in
  [BUILDING.md](BUILDING.md#comparing-kernels-with-a-release).
- Attach every timing or quality figure in a document to its measurement record. Include the game, resolution, mode,
  build, and log or result file. Do not publish a number without its record.

## Report bugs and game results

Use the **Bug report** issue template for problems and the **Game compatibility report** template for game behavior.
They add the `bug` and `game-compatibility` labels respectively. See
[How to add your game](COMPATIBILITY.md#how-to-add-your-game) for the game-testing process.

For a visual problem, include the game, scene, settings, GPU, driver, and the zip from the launcher's **Collect logs**.
Check the zip before sharing it because it can contain your Windows user name. Do not attach the upscaler's DLLs or
other copyrighted game files.

Use the [WrongGPU Discord server](https://discord.gg/745z9MfKwm) for questions and discussion.
