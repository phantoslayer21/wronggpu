// SPDX-License-Identifier: MIT
// Test double for AMD's amd_fidelityfx_dx12.dll, built as amd_fidelityfx_dx12.wronggpu-original.dll. It does no
// GPU work: it hands out numbered contexts, answers the version queries and records every call, which the test
// reads back through the extra export fakeLog. With WG_FAKE_AMD_MIB=N in the process environment (test_dll_d3d12
// switch_real), each upscaler context with a D3D12 backend also holds an N MiB buffer on that device until it is
// destroyed, as AMD's FSR 4 holds GPU memory.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>

#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>

#include "ffx_api/ffx_api.h"
#include "ffx_api/ffx_upscale.h"
#include "ffx_api/dx12/ffx_api_dx12.h"

namespace {
std::mutex g_mutex;
std::string g_log;
int g_next = 1;
std::map<void*, ID3D12Resource*> g_memory;  // WG_FAKE_AMD_MIB: the buffer of each context

ID3D12Resource* HoldMemory(const ffxApiHeader* desc) {
    char text[32] = "";
    const unsigned long mib = GetEnvironmentVariableA("WG_FAKE_AMD_MIB", text, sizeof text) ? std::strtoul(text, nullptr, 10) : 0;
    ID3D12Device* device = nullptr;
    bool upscale = false;
    for (const ffxApiHeader* h = desc; h; h = h->pNext) {
        if (h->type == FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE) upscale = true;
        if (h->type == FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12) device = reinterpret_cast<const ffxCreateBackendDX12Desc*>(h)->device;
    }
    if (!mib || !device || !upscale) return nullptr;
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = UINT64(mib) << 20;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ID3D12Resource* r = nullptr;
    if (FAILED(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COMMON, nullptr, __uuidof(ID3D12Resource),
                                               reinterpret_cast<void**>(&r))))
        return nullptr;
    return r;
}

void Record(const char* fn, ffxContext* context, const ffxApiHeader* desc) {
    char line[160];
    std::snprintf(line, sizeof line, "%s ctx=%p type=0x%llx\n", fn, context ? *context : nullptr,
                  desc ? static_cast<unsigned long long>(desc->type) : 0ull);
    std::lock_guard<std::mutex> lock(g_mutex);
    g_log += line;
}
}  // namespace

extern "C" {

FFX_API_ENTRY ffxReturnCode_t ffxCreateContext(ffxContext* context, ffxCreateContextDescHeader* desc, const ffxAllocationCallbacks*) {
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        *context = reinterpret_cast<ffxContext>(static_cast<uintptr_t>(0xAD0000 + 0x100 * g_next++));
    }
    if (ID3D12Resource* r = HoldMemory(desc)) {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_memory[*context] = r;
    }
    Record("create", context, desc);
    return FFX_API_RETURN_OK;
}

FFX_API_ENTRY ffxReturnCode_t ffxDestroyContext(ffxContext* context, const ffxAllocationCallbacks*) {
    Record("destroy", context, nullptr);
    ID3D12Resource* held = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_memory.find(*context);
        if (it != g_memory.end()) held = it->second, g_memory.erase(it);
    }
    if (held) held->Release();
    *context = nullptr;
    return FFX_API_RETURN_OK;
}

FFX_API_ENTRY ffxReturnCode_t ffxConfigure(ffxContext* context, const ffxConfigureDescHeader* desc) {
    Record("configure", context, desc);
    return FFX_API_RETURN_OK;
}

FFX_API_ENTRY ffxReturnCode_t ffxQuery(ffxContext* context, ffxQueryDescHeader* desc) {
    Record("query", context, desc);
    if (desc && desc->type == FFX_API_QUERY_DESC_TYPE_GET_VERSIONS) {
        auto* q = reinterpret_cast<ffxQueryDescGetVersions*>(desc);
        const uint64_t capacity = q->outputCount ? *q->outputCount : 0;
        if (q->outputCount) *q->outputCount = 1;
        if (capacity >= 1 && q->versionIds) q->versionIds[0] = 0xFA4E;
        if (capacity >= 1 && q->versionNames) q->versionNames[0] = "fake AMD FSR 3.1";
    }
    if (desc && desc->type == FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTERPHASECOUNT) {
        auto* q = reinterpret_cast<ffxQueryDescUpscaleGetJitterPhaseCount*>(desc);
        if (q->pOutPhaseCount) *q->pOutPhaseCount = 99;
    }
    return FFX_API_RETURN_OK;
}

FFX_API_ENTRY ffxReturnCode_t ffxDispatch(ffxContext* context, const ffxDispatchDescHeader* desc) {
    Record("dispatch", context, desc);
    return FFX_API_RETURN_OK;
}

__declspec(dllexport) const char* fakeLog() {
    std::lock_guard<std::mutex> lock(g_mutex);
    static std::string copy;
    copy = g_log;
    return copy.c_str();
}

}  // extern "C"
