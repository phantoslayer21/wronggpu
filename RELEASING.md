# Releasing WrongGPU

How an official WrongGPU release is made. Contributors don't need to do any of this; it explains what happens between
a merged pull request and a download on the [Releases page](https://github.com/phantoslayer21/wronggpu/releases).

## Who releases

Only the project owner builds, tests and publishes official releases. Pull requests are merged into `main` as usual;
a release is a separate decision. Nobody else uploads release packages or creates release tags.

## 1. The source of a release

- A release is built from **one commit on `main` of this repository**. That commit is public before the release is
  published, so anyone can see exactly what was built.
- The release tag (for example `v0.6.1`) points at that commit. Tags are never moved after publishing; a fix is a new
  version.
- The version number is set in the build options of BUILDING.md (`-DWG_BUILD_ID=...` for the front end;
  `-DWRONGGPU_VERSION=...`, `-DWRONGGPU_BUILD_ID=...`, `-DWRONGGPU_PAYLOAD_NAME=...` for the launcher).

## 2. Build and test on real hardware

- The owner builds the release on the owner's own PC from a fresh checkout of that commit, with the tools and steps of
  [BUILDING.md](BUILDING.md): kernels, runtime library, FSR front end, Ray Reconstruction plugin, payload, launcher.
- Before anything is uploaded:
  - the CPU tests of BUILDING.md pass (the same set CI runs on every pull request);
  - the GPU tests pass on a Radeon RX 9000-series card;
  - the release is installed, used and uninstalled in real games with the launcher, including the games named in the
    release notes;
  - the package contains only WrongGPU's own files and their notices: no file of the original upscaler, no model data,
    no OptiScaler file (the launcher downloads OptiScaler from its own release).

## 3. The release notes

- The notes live in the repository (`clean/installer/RELEASE-NOTES.md`) and go through a pull request like any other
  change.
- They point to the release page, which lists the **SHA-256 of the zip**, and to `SHA256SUMS.txt` inside the zip, which
  lists the SHA-256 of every file in it, so a download can be checked.
- **The number rule:** every figure in the release notes, the release page and the README (timings, quality figures,
  version requirements, hardware lists) is traced to its measurement record: which game, resolution, mode, WrongGPU
  build and log or result file. A figure without a record is left out. Before publishing, someone who did not write
  the texts compares every figure across the README, the release page and the release notes, and against its record;
  a mismatch or an untraced figure blocks the release until it is fixed or removed.

## 4. Publishing

- Uploaded to the Releases page only on the owner's order: the package (`WrongGPU-<version>.zip`), its SHA-256 file
  and the release notes, attached to the tag of step 1.
- Pre-releases are marked as such.
- After publishing, the downloaded package is checked against the SHA-256 on the release page and in `SHA256SUMS.txt`.

## Licence of releases

From the source of this repository onward, official builds are under the MIT licence, the same as the source
([LICENSE](LICENSE)); the package carries the licence text and the third-party notices.
