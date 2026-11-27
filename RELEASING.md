# Releasing WrongGPU

This page explains how the project owner makes an official WrongGPU release. Contributors do not perform these steps.
The process covers the work between merging a pull request and publishing a download on the
[Releases page](https://github.com/phantoslayer21/wronggpu/releases).

## Who releases

Only the project owner builds, tests, and publishes official releases. Pull requests merge into `main` as usual, but
the owner decides separately whether to make a release. Nobody else uploads release packages or creates release tags.

## 1. Identify the source

- Build the release from **one commit on `main` in this repository**. Publish that commit before the release so anyone
  can inspect exactly what was built.
- Point the release tag, such as `v0.6.5`, at that commit. Never move a tag after publishing it. Release a fix as a new
  version.
- Set the version through the build options in [BUILDING.md](BUILDING.md). Use `-DWG_BUILD_ID=...` for the front end.
  Use `-DWRONGGPU_VERSION=...`, `-DWRONGGPU_BUILD_ID=...`, and `-DWRONGGPU_PAYLOAD_NAME=...` for the launcher.

## 2. Build and test on real hardware

The owner builds from a fresh checkout of the release commit on the owner's own PC. Follow [BUILDING.md](BUILDING.md)
to build the kernels, runtime library, FSR front end, Ray Reconstruction plugin, payload, and launcher.

Before uploading anything, complete all of these checks:

- Pass the CPU tests in [BUILDING.md](BUILDING.md#6-tests-cpu). These are the same tests that CI runs on every pull
  request.
- Pass the GPU tests on a Radeon RX 9000-series card.
- Use the launcher to install, run, and uninstall the release in real games, including every game named in the release
  notes.
- Check that the package contains only WrongGPU's own files and their notices. It must contain no file from the
  original upscaler, no model data, and no OptiScaler file. The launcher downloads OptiScaler from its own release.

## 3. Prepare and check the release notes

- Keep the notes in `clean/installer/RELEASE-NOTES.md` and review them through a pull request like any other change.
- Link the notes to the release page, which lists the **SHA-256 of the zip**. Link them also to `SHA256SUMS.txt`
  inside the zip, which lists the SHA-256 of every file in the package.
- Trace every figure in the release notes, release page, and README to its measurement record. This includes timings,
  quality figures, version requirements, and hardware lists. Record the game, resolution, mode, WrongGPU build, and
  log or result file. Leave out any figure without a record.
- Before publishing, have someone who did not write the texts compare every figure across the README, release page,
  and release notes, and against its record. A mismatch or untraced figure blocks the release until someone fixes or
  removes it.

## 4. Publish the package

Only the owner can order an upload to the Releases page. Attach these items to the tag selected in step 1:

- `WrongGPU-<version>.zip`.
- The package's SHA-256 file.
- The release notes.

Mark pre-releases as pre-releases. After publishing, download the package and check it against both the SHA-256 on the
release page and the entries in `SHA256SUMS.txt`.

## Licence of releases

Official builds of 0.7.0 and later are under GPL-3.0-only, the same as the source ([LICENSE](LICENSE)).
The package carries the licence text and the third-party notices. Releases up to 0.6.5 remain available under MIT.
