// SPDX-License-Identifier: MIT
#include "amd_compare.h"

#include <cstdio>
#include <cstdlib>

namespace wgfsr {

namespace {

// "4.1.1" -> 4001001 (for picking the newest provider); 0 when the name is not a version.
uint64_t VersionRank(const char* name) {
    if (!name) return 0;
    uint64_t rank = 0;
    int parts = 0;
    const char* p = name;
    while (*p && parts < 3) {
        char* end = nullptr;
        const unsigned long v = std::strtoul(p, &end, 10);
        if (end == p) break;
        rank = rank * 1000 + (v < 1000 ? v : 999);
        ++parts;
        p = *end == '.' ? end + 1 : end;
        if (*end != '.') break;
    }
    while (parts++ < 3) rank *= 1000;
    return rank;
}

}  // namespace

bool CreateComparison(AmdComparison& c, const OriginalApi* api, const UpscaleCreateParams& params, void* device,
                      ffxApiMessage fpMessage, std::string& log) {
    DestroyComparison(c);
    if (!api || !api->createContext || !api->query || !api->dispatch || !api->destroyContext) {
        log = "no AMD FidelityFX DLL for the comparison";
        return false;
    }
    if (!device) {
        log = "no D3D12 device for the comparison";
        return false;
    }
    // The providers AMD's DLL offers for upscaling on this device; the newest is used.
    uint64_t count = 0;
    ffxQueryDescGetVersions versions{};
    versions.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
    versions.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    versions.device = device;
    versions.outputCount = &count;
    uint64_t ids[16] = {};
    const char* names[16] = {};
    uint64_t chosenId = 0;
    const char* chosenName = nullptr;
    if (api->query(nullptr, &versions.header) == FFX_API_RETURN_OK && count > 0) {
        if (count > 16) count = 16;
        versions.versionIds = ids;
        versions.versionNames = names;
        if (api->query(nullptr, &versions.header) == FFX_API_RETURN_OK) {
            log = "providers:";
            for (uint64_t i = 0; i < count; ++i) {
                log += std::string(" ") + (names[i] ? names[i] : "?");
                if (!chosenName || VersionRank(names[i]) > VersionRank(chosenName)) {
                    chosenId = ids[i];
                    chosenName = names[i];
                }
            }
        }
    }
    ffxCreateContextDescUpscale create{};
    create.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    create.flags = params.flags;
    create.maxRenderSize = params.maxRenderSize;
    create.maxUpscaleSize = params.maxUpscaleSize;
    create.fpMessage = fpMessage;
    BackendDx12DescView backend{};
    backend.header.type = kCreateBackendDx12;
    backend.device = device;
    ffxOverrideVersion version{};
    version.header.type = FFX_API_DESC_TYPE_OVERRIDE_VERSION;
    version.versionId = chosenId;
    create.header.pNext = &backend.header;
    backend.header.pNext = chosenName ? &version.header : nullptr;
    const ffxReturnCode_t rc = api->createContext(&c.context, &create.header, nullptr);
    if (rc != FFX_API_RETURN_OK || !c.context) {
        char buf[64];
        std::snprintf(buf, sizeof buf, "AMD's create failed (%u)", unsigned(rc));
        log += (log.empty() ? "" : "; ") + std::string(buf);
        c.context = nullptr;
        return false;
    }
    c.api = api;
    c.name = std::string("AMD FSR") + (chosenName && *chosenName ? std::string(" ") + chosenName : std::string());
    return true;
}

ffxReturnCode_t DispatchComparison(AmdComparison& c, const ffxDispatchDescUpscale& dispatch, bool reset) {
    if (!c.context || !c.api) return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
    ffxDispatchDescUpscale d = dispatch;
    d.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
    if (reset) d.reset = true;
    return c.api->dispatch(&c.context, &d.header);
}

void DestroyComparison(AmdComparison& c) {
    if (c.context && c.api) c.api->destroyContext(&c.context, nullptr);
    c.context = nullptr;
    c.api = nullptr;
}

}  // namespace wgfsr
