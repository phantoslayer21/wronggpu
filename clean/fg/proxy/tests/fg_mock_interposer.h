// SPDX-License-Identifier: MIT
// WrongGPU frame generation (issue #96, F8b): the test stand-in for a game interface DLL (fg_mock_interposer.dll).
// Synthetic export names, the game interface's call shapes (rr_proxy_abi.h), and a call record the tests read: every
// call's arguments and the result it returned, so a forwarded call can be compared argument by argument.
// Written for this project.
#pragma once
#include <stdint.h>

#include "../../../rr_f/proxy/include/rr_proxy_abi.h"

#if defined(FG_MOCK_INTERPOSER_BUILD)
#define FG_MOCK_API extern "C" __declspec(dllexport)
#else
#define FG_MOCK_API extern "C" __declspec(dllimport)
#endif

enum FgMockFn : uint32_t {
    kMockIsSupported = 0, kMockRequirements, kMockGetFunction, kMockTag, kMockTagForFrame, kMockConstants,
    kMockEvaluate, kMockFree, kMockInit, kMockFgOptions, kMockFgState, kMockSrOptions, kMockFnCount
};

struct FgMockCall {
    uint32_t feature;
    const void* a;
    const void* b;
    const void* c;
    const void* d;
    uint32_t count;
    int32_t returned;
};

struct FgMockRecord {
    uint32_t calls[kMockFnCount];
    FgMockCall last[kMockFnCount];
    int32_t result[kMockFnCount];  // what each function returns (a test sets distinct codes)
};

// The names the stand-in's get-function serves (synthetic).
#define FG_MOCK_FN_FG_OPTIONS "fgmock_fn_fg_options"
#define FG_MOCK_FN_FG_STATE "fgmock_fn_fg_state"
#define FG_MOCK_FN_SR_OPTIONS "fgmock_fn_sr_options"

FG_MOCK_API FgMockRecord* fgmock_record(void);
FG_MOCK_API void fgmock_reset(void);
FG_MOCK_API void* fgmock_feature_fn(uint32_t which);  // kMockFgOptions / kMockFgState / kMockSrOptions

FG_MOCK_API WgResultCode fgmock_is_supported(uint32_t feature, const WgAdapterInfo* adapter);
FG_MOCK_API WgResultCode fgmock_requirements(uint32_t feature, WgRequirements* req);
FG_MOCK_API WgResultCode fgmock_get_function(uint32_t feature, const char* name, void** out);
FG_MOCK_API WgResultCode fgmock_tag(const WgViewport* vp, const WgTag* tags, uint32_t count, void* cmd);
FG_MOCK_API WgResultCode fgmock_tag_for_frame(const WgFrameToken* token, const WgViewport* vp, const WgTag* tags,
                                              uint32_t count, void* cmd);
FG_MOCK_API WgResultCode fgmock_constants(const WgConstants* c, const WgFrameToken* token, const WgViewport* vp);
FG_MOCK_API WgResultCode fgmock_evaluate(uint32_t feature, const WgFrameToken* token, const WgHeader* const* inputs,
                                         uint32_t count, void* cmd);
FG_MOCK_API WgResultCode fgmock_free(uint32_t feature, const WgViewport* vp);
FG_MOCK_API WgResultCode fgmock_init(uint64_t value);  // never redirected (not a hooked role)

// The stand-in game module (fg_mock_game.dll / fg_mock_game_delay.dll) calls the interface through its import slots.
#define FG_MOCK_GAME_SLOTS 9  // the eight roles in FgRole order, then fgmock_init
