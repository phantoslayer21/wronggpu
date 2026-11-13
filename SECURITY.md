# Security policy

WrongGPU's launcher installs files into game folders, can run elevated for that, and downloads OptiScaler from a
pinned release (URL, size and SHA-256 are fixed in `clean/installer/src/wg_core/optiedition.h`). The front end and the
plugin run inside games. Please report security problems in any of these privately.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting: on this repository's **Security** tab, choose **Report a
vulnerability**. Please do not open a public issue for a security problem.

Include what you found, the version (or commit), and how to reproduce it. We will acknowledge the report, keep you
informed, and credit you in the release notes of the fix unless you prefer otherwise.

## Supported versions

Only the latest release receives security fixes.

## Scope

In scope: the launcher (`clean/installer`), the FSR front end (`clean/fsr`), the Ray Reconstruction plugin
(`clean/rr_f/proxy`), the runtime libraries and the converter. Out of scope: OptiScaler itself (report to its project),
the games, AMD's drivers and runtimes, and the original upscaler's files.
