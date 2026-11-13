**What this changes and why**

**What each new or substantially changed file is based on** (public document, test result or own design)

**How it was tested** (CI runs the CPU tests; say what else you ran: GPU or game tests, with synthetic data)

**Checklist**
- [ ] I make the clean-room statement of
      [CONTRIBUTING.md](https://github.com/phantoslayer21/wronggpu/blob/main/CONTRIBUTING.md#the-clean-room-statement)
      for this and all my contributions (required once, in my first pull request).
- [ ] Every commit has a `Signed-off-by:` line (`git commit -s`; Developer Certificate of Origin 1.1).
- [ ] New source files carry the SPDX licence line for MIT, or are covered in `REUSE.toml`.
- [ ] No files of the original upscaler (see CONTRIBUTING.md), no model data, game files, captures, screenshots or
      binaries; no paths, user or machine names of my PC.
- [ ] Any new third-party component is under a permitted licence and listed in THIRD-PARTY-NOTICES and REUSE.toml.
- [ ] Numbers in documents come with their measurement record.
- [ ] Kernel or runtime changes: I say whether the output is meant to stay bit-identical and show the comparison.

A maintainer scans every pull request on their side before merge (clean-room gate), reviews it, and tests it on a GPU
when the change touches kernels, the runtime, the front end or the Ray Reconstruction plugin (CONTRIBUTING.md, "How a
pull request gets merged").
