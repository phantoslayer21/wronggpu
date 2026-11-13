// SPDX-License-Identifier: MIT
// FakeOptiLoader (fake_opti_loader.h): our own OptiScaler-style loader of a custom FFX upscaler slot (spec 65).
#include "fake_opti_loader.h"

#include <cstdlib>

bool FakeOptiLoader::Load() {
    if (module) return true;
    module = LoadLibraryExW(SlotDll().c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (!module) return false;
    create = reinterpret_cast<PfnFfxCreateContext>(reinterpret_cast<void*>(GetProcAddress(module, "ffxCreateContext")));
    destroy = reinterpret_cast<PfnFfxDestroyContext>(reinterpret_cast<void*>(GetProcAddress(module, "ffxDestroyContext")));
    query = reinterpret_cast<PfnFfxQuery>(reinterpret_cast<void*>(GetProcAddress(module, "ffxQuery")));
    dispatch = reinterpret_cast<PfnFfxDispatch>(reinterpret_cast<void*>(GetProcAddress(module, "ffxDispatch")));
    return create && destroy && query && dispatch;
}

bool FakeOptiLoader::QueryVersions(void* device) {
    ids.clear();
    names.clear();
    uint64_t count = 0;
    ffxQueryDescGetVersions q{};
    q.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
    q.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    q.device = device;
    q.outputCount = &count;
    if (query(nullptr, &q.header) != FFX_API_RETURN_OK || count == 0 || count > 64) return false;
    std::vector<uint64_t> vid(size_t(count), 0);
    std::vector<const char*> vname(size_t(count), nullptr);
    q.versionIds = vid.data();
    q.versionNames = vname.data();
    if (query(nullptr, &q.header) != FFX_API_RETURN_OK) return false;
    for (uint64_t i = 0; i < count; ++i) {
        ids.push_back(vid[size_t(i)]);
        names.push_back(vname[size_t(i)] ? vname[size_t(i)] : "");
    }
    return true;
}

int FakeOptiLoader::IndexFromIni() const {
    wchar_t value[64] = {};
    GetPrivateProfileStringW(folder.c_str(), L"UpscalerIndex", L"auto", value, 64, HostIni().c_str());
    wchar_t* end = nullptr;
    const long n = std::wcstol(value, &end, 10);
    return end == value || *end ? -1 : int(n);
}

bool FakeOptiLoader::WriteIndex(int index) const {
    const std::wstring v = index < 0 ? std::wstring(L"auto") : std::to_wstring(index);
    return WritePrivateProfileStringW(folder.c_str(), L"UpscalerIndex", v.c_str(), HostIni().c_str()) != 0;
}

ffxReturnCode_t FakeOptiLoader::Select(int index, void* device, uint32_t renderW, uint32_t renderH, uint32_t outW, uint32_t outH) {
    Release();
    if (index < 0 || size_t(index) >= ids.size()) return FFX_API_RETURN_ERROR_PARAMETER;
    ffxOverrideVersion version{};
    version.header.type = FFX_API_DESC_TYPE_OVERRIDE_VERSION;
    version.versionId = ids[size_t(index)];
    wgfsr::BackendDx12DescView backend{};  // ffxCreateBackendDX12Desc's layout
    backend.header.type = wgfsr::kCreateBackendDx12;
    backend.header.pNext = &version.header;
    backend.device = device;
    ffxCreateContextDescUpscale up{};
    up.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    up.header.pNext = &backend.header;
    up.flags = FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE;
    up.maxRenderSize = {renderW, renderH};
    up.maxUpscaleSize = {outW, outH};
    const ffxReturnCode_t rc = create(&context, &up.header, nullptr);
    if (rc == FFX_API_RETURN_OK) contextVersion = version.versionId;
    else context = nullptr;
    return rc;
}

std::string FakeOptiLoader::ProviderName(uint64_t* id) {
    if (!context) return std::string();
    ffxQueryGetProviderVersion q{};
    q.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
    if (query(&context, &q.header) != FFX_API_RETURN_OK) return std::string();
    if (id) *id = q.versionId;
    return q.versionName ? q.versionName : "";
}

void FakeOptiLoader::Release() {
    if (context) destroy(&context, nullptr);
    context = nullptr;
    contextVersion = 0;
}
