// SPDX-License-Identifier: MIT
#pragma once
// Version of the FSR front end, shown in the log and the DLL's version resource.
// 0.6.1: a release build takes the release script's one version (CMake WG_VERSION writes wg_version_override.h and
// defines WG_VERSION_OVERRIDE, for the C++ sources and the version resources alike); the values below are the default
// of a build without it.
#ifdef WG_VERSION_OVERRIDE
#include "wg_version_override.h"
#else
#define WRONGGPU_FSR_VERSION "0.6.1"
#define WRONGGPU_FSR_VERSION_RC 0, 6, 1, 0
#endif
#ifndef WRONGGPU_FSR_BUILD
#define WRONGGPU_FSR_BUILD "local build"
#endif
