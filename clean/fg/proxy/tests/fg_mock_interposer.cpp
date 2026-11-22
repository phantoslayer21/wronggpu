// SPDX-License-Identifier: MIT
// WrongGPU frame generation (issue #96, F8b): the test stand-in for a game interface DLL. See fg_mock_interposer.h.
// Written for this project.
#define FG_MOCK_INTERPOSER_BUILD 1
#include "fg_mock_interposer.h"

#include <string.h>

namespace {

FgMockRecord g_rec = {};
IUnknown* g_factory = nullptr;
bool g_loaded = false;  // what fgmock_set_loaded stored (fgmock_is_loaded returns it)

int32_t Note(uint32_t fn, uint32_t feature, const void* a, const void* b, const void* c, const void* d, uint32_t n) {
    ++g_rec.calls[fn];
    const int32_t r = g_rec.result[fn];
    g_rec.last[fn] = FgMockCall{feature, a, b, c, d, n, r};
    return r;
}

WgResultCode MockFgOptions(const WgViewport* vp, const void* native) {
    return Note(kMockFgOptions, 0, vp, native, nullptr, nullptr, 0);
}
WgResultCode MockFgState(const WgViewport* vp, void* native, const void* options) {
    return Note(kMockFgState, 0, vp, native, options, nullptr, 0);
}
WgResultCode MockSrOptions(const WgViewport* vp, const void* native) {
    return Note(kMockSrOptions, 0, vp, native, nullptr, nullptr, 0);
}

}  // namespace

FG_MOCK_API FgMockRecord* fgmock_record(void) { return &g_rec; }
FG_MOCK_API void fgmock_reset(void) {
    memset(&g_rec, 0, sizeof g_rec);
    g_loaded = false;
}
FG_MOCK_API void* fgmock_feature_fn(uint32_t which) {
    switch (which) {
        case kMockFgOptions: return reinterpret_cast<void*>(&MockFgOptions);
        case kMockFgState: return reinterpret_cast<void*>(&MockFgState);
        case kMockSrOptions: return reinterpret_cast<void*>(&MockSrOptions);
        default: return nullptr;
    }
}

FG_MOCK_API WgResultCode fgmock_is_supported(uint32_t feature, const WgAdapterInfo* adapter) {
    return Note(kMockIsSupported, feature, adapter, nullptr, nullptr, nullptr, 0);
}
FG_MOCK_API WgResultCode fgmock_requirements(uint32_t feature, WgRequirements* req) {
    return Note(kMockRequirements, feature, req, nullptr, nullptr, nullptr, 0);
}
FG_MOCK_API WgResultCode fgmock_get_function(uint32_t feature, const char* name, void** out) {
    const int32_t r = Note(kMockGetFunction, feature, name, out, nullptr, nullptr, 0);
    if (!name || !out) return r;
    void* fn = nullptr;
    if (feature == WG_FEAT_FRAME_GENERATION && strcmp(name, FG_MOCK_FN_FG_OPTIONS) == 0) fn = fgmock_feature_fn(kMockFgOptions);
    else if (feature == WG_FEAT_FRAME_GENERATION && strcmp(name, FG_MOCK_FN_FG_STATE) == 0) fn = fgmock_feature_fn(kMockFgState);
    else if (feature == WG_FEAT_SUPER_RESOLUTION && strcmp(name, FG_MOCK_FN_SR_OPTIONS) == 0) fn = fgmock_feature_fn(kMockSrOptions);
    if (!fn) return WG_E_FEATURE_MISSING;
    *out = fn;
    return r;
}
FG_MOCK_API WgResultCode fgmock_tag(const WgViewport* vp, const WgTag* tags, uint32_t count, void* cmd) {
    return Note(kMockTag, 0, vp, tags, cmd, nullptr, count);
}
FG_MOCK_API WgResultCode fgmock_tag_for_frame(const WgFrameToken* token, const WgViewport* vp, const WgTag* tags,
                                              uint32_t count, void* cmd) {
    return Note(kMockTagForFrame, 0, token, vp, tags, cmd, count);
}
FG_MOCK_API WgResultCode fgmock_constants(const WgConstants* c, const WgFrameToken* token, const WgViewport* vp) {
    return Note(kMockConstants, 0, c, token, vp, nullptr, 0);
}
FG_MOCK_API WgResultCode fgmock_evaluate(uint32_t feature, const WgFrameToken* token, const WgHeader* const* inputs,
                                         uint32_t count, void* cmd) {
    return Note(kMockEvaluate, feature, token, inputs, cmd, nullptr, count);
}
FG_MOCK_API WgResultCode fgmock_free(uint32_t feature, const WgViewport* vp) {
    return Note(kMockFree, feature, vp, nullptr, nullptr, nullptr, 0);
}
FG_MOCK_API WgResultCode fgmock_is_loaded(uint32_t feature, bool* loaded) {
    const int32_t r = Note(kMockIsLoaded, feature, loaded, nullptr, nullptr, nullptr, 0);
    if (loaded) *loaded = g_loaded;
    return r;
}
FG_MOCK_API WgResultCode fgmock_set_loaded(uint32_t feature, bool loaded) {
    g_loaded = loaded;
    return Note(kMockSetLoaded, feature, nullptr, nullptr, nullptr, nullptr, loaded ? 1u : 0u);
}
FG_MOCK_API WgResultCode fgmock_init(uint64_t value) {
    return Note(kMockInit, static_cast<uint32_t>(value), nullptr, nullptr, nullptr, nullptr, 0);
}
FG_MOCK_API void fgmock_set_factory(IUnknown* factory) { g_factory = factory; }
FG_MOCK_API HRESULT WINAPI fgmock_create_factory2(UINT flags, REFIID riid, void** out) {
    ++g_rec.calls[kMockCreateFactory2];
    HRESULT hr = E_FAIL;
    if (!out) hr = E_POINTER;
    else if (g_factory) hr = g_factory->QueryInterface(riid, out);
    else *out = nullptr;
    g_rec.last[kMockCreateFactory2] = FgMockCall{flags, &riid, out, nullptr, nullptr, 0, static_cast<int32_t>(hr)};
    return hr;
}
