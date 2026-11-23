# Security policy

WrongGPU's launcher installs files into game folders, can run elevated to do so, and downloads OptiScaler from a pinned
release. The URL, size, and SHA-256 are fixed in `clean/installer/src/wg_core/optiedition.h`. The FSR front end and the
Ray Reconstruction plugin run inside games. Report security problems in any of these components privately.

## Report a vulnerability

Use GitHub's private vulnerability reporting. Open this repository's **Security** tab and choose **Report a
vulnerability**. Do not open a public issue for a security problem.

Include what you found, the affected version or commit, and reproduction steps. We will acknowledge the report and keep
you informed. We will credit you in the release notes for the fix unless you prefer otherwise.

## Supported versions

Only the latest release receives security fixes.

## Scope

In scope:

- The launcher in `clean/installer`.
- The FSR front end in `clean/fsr`.
- The Ray Reconstruction plugin in `clean/rr_f/proxy`.
- The runtime libraries and the converter.

Out of scope:

- OptiScaler itself. Report its security problems to the OptiScaler project.
- Games.
- AMD's drivers and runtimes.
- The original upscaler's files.
