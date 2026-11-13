// SPDX-License-Identifier: MIT
// Own host mock: a distinct AMD-provider alias beside our hosted DLL, no third-party implementation.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include "../src/frontend.h"

int main() {
    HMODULE host = LoadLibraryW(L"wronggpu_ffx_dx12.dll");
    if (!host) return 1;
    auto query = reinterpret_cast<PfnFfxQuery>(GetProcAddress(host, "ffxQuery"));
    auto create = reinterpret_cast<PfnFfxCreateContext>(GetProcAddress(host, "ffxCreateContext"));
    auto destroy = reinterpret_cast<PfnFfxDestroyContext>(GetProcAddress(host, "ffxDestroyContext"));
    if (!query || !create || !destroy) return 2;
    uint64_t count = 0;
    ffxQueryDescGetVersions q{};
    q.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
    q.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    q.outputCount = &count;
    // WrongGPU's entries first: Model M, and Model K when it is built in (CMake WG_MODEL_K); then AMD's.
    const int own = wgfsr::OwnEntryCount();  // the registry's entries this build offers (SR models, RR modes)
    if (query(nullptr, &q.header) != FFX_API_RETURN_OK || count != uint64_t(own + 1)) return 3;
    uint64_t ids[16]{};
    const char* names[16]{};
    q.versionIds = ids; q.versionNames = names;
    if (query(nullptr, &q.header) != FFX_API_RETURN_OK || count != uint64_t(own + 1) ||
        ids[0] != wgfsr::kWrongGpuProviderVersionId || ids[own] != 0xFA4E ||
        std::strcmp(names[0], wgfsr::kWrongGpuListedNameM) != 0) return 4;
#if WRONGGPU_MODEL_K
    if (ids[1] != wgfsr::kWrongGpuKProviderVersionId || std::strcmp(names[1], wgfsr::kWrongGpuListedNameK) != 0) return 4;
#else  // the built DLL lists nothing of Model K
    for (int i = 0; i <= own; ++i)
        if (ids[i] == wgfsr::kWrongGpuKProviderVersionId || std::strstr(names[i], "Model K")) return 4;
#endif
    // A host reads "the FidelityFX version" of the DLL out of the first listed name (test_frontend.cpp,
    // TestHostVersionHandling): the built DLL gives it 3.1.0 hidden after "##" (shown text first, no number in it),
    // never 0.x and never 4.x.
    const char* hidden = std::strstr(names[0], "##");
    if (!hidden || std::strcmp(hidden, "##3.1.0") != 0 || std::strpbrk(std::string(names[0], hidden).c_str(), "0123456789")) return 7;
    if (!std::strstr(names[0], "WrongGPU Model M")) return 8;
#if WRONGGPU_MODEL_K
    if (!std::strstr(names[1], "WrongGPU Model K")) return 8;
#endif
    ffxOverrideVersion version{};
    version.header.type = FFX_API_DESC_TYPE_OVERRIDE_VERSION;
    version.versionId = ids[own];
    ffxCreateContextDescUpscale up{};
    up.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    up.header.pNext = &version.header;
    ffxContext context = nullptr;
    if (create(&context, &up.header, nullptr) != FFX_API_RETURN_OK || !context) return 5;
    if (destroy(&context, nullptr) != FFX_API_RETURN_OK || context) return 6;
    std::puts("PASS: hosted DLL lists Model M first with a hidden 3.1.0 level and forwards AMD selection through its distinct alias");
    return 0;
}
