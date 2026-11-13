// SPDX-License-Identifier: MIT
// WrongGPU launcher core (wg_core): the common preamble of every wg_core file. Written for this project.
//
// wg_core is the engine of the installer and of the launcher that replaces it (build/launcher-design.md section 2):
// the code that was setup.cpp, split into modules (detect, checks, model, payload, record, optiscaler, coexist,
// settings, logs, install) with a Context (context.h) passed in instead of globals. Native Win32, C++17, no
// third-party code.
#pragma once
#ifndef UNICODE
#define UNICODE
#endif
#ifndef _UNICODE
#define _UNICODE
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#include <windows.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "wizard_policy.h"  // WRONGGPU_PRESET_K, WRONGGPU_DIRECT_ROUTE (their defaults when CMake does not set them)

namespace wg {
using std::string;
using std::wstring;
using wg_wizard::kDirectRoute;  // CMake WRONGGPU_DIRECT_ROUTE: the direct FSR route is built in
}  // namespace wg
