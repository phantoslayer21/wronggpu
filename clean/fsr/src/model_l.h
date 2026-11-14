// SPDX-License-Identifier: MIT
// Model L in this build of the front end (#89): CMake option WG_MODEL_L (default OFF) defines WRONGGPU_MODEL_L as 1 or
// 0. 1: the entry "WrongGPU Model L" in the hosted version list, its id, sr_model=L and model=L reach the runtime as its
// create option (the runtime must be built with Model L, WG_MODEL_L), the model lookup looks for <place>\L\model.wgcm,
// and the overlay and the provider label name Model L. 0 (today's releases): none of that; model=L from wronggpu.ini or
// WG_RUNTIME_OPTIONS is dropped with one log line that says so (IgnoreModelL) and Model M runs; the Model L id is an
// unknown override like any other. Sources built without the definition get no Model L (unlike Model K: Model L is not
// in a release yet).
#pragma once

#ifndef WRONGGPU_MODEL_L
#define WRONGGPU_MODEL_L 0
#endif
