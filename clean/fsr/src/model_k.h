// SPDX-License-Identifier: MIT
// Model K in this build of the front end: CMake option WG_MODEL_K (default ON) defines WRONGGPU_MODEL_K as 1 or 0.
// 0 (a release that ships Model M only): no Model K entry in the hosted version list, the Model K id is an unknown
// override like any other id that was not listed (logged; WrongGPU upscales with Model M), model=K from wronggpu.ini
// or WG_RUNTIME_OPTIONS is dropped with one log line (FrontEnd::IgnoreModelK), and the overlay and the provider label
// name Model M only. Sources built without the definition (single-file checks) get Model K, as before the option.
#pragma once

#ifndef WRONGGPU_MODEL_K
#define WRONGGPU_MODEL_K 1
#endif
