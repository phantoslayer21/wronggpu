// SPDX-License-Identifier: MIT
// Fake WrongGPU RR runtime library for the end-to-end proxy test (exports the adapter entry point).
#include "fake_runtime.h"

static FakeRuntime g_rt;

extern "C" __declspec(dllexport) const WgRrRuntimeV1* WrongGPU_RrRuntimeV1(void) { return &g_rt.table; }
extern "C" __declspec(dllexport) FakeRuntime* wgtest_fr_state(void) { return &g_rt; }
