// SPDX-License-Identifier: GPL-3.0-only
// WrongGPU frame generation (issue #96, F8b): the test stand-in game module. It imports the stand-in interface's
// functions by name (built twice: a static import table, fg_mock_game.dll, and a delay-load table,
// fg_mock_game_delay.dll) and calls them only through those import slots, as a game does.
// Written for this project.
#include "fg_mock_interposer.h"

#define FG_GAME_API extern "C" __declspec(dllexport)

FG_GAME_API WgResultCode game_is_supported(uint32_t f, const WgAdapterInfo* a) { return fgmock_is_supported(f, a); }
FG_GAME_API WgResultCode game_requirements(uint32_t f, WgRequirements* r) { return fgmock_requirements(f, r); }
FG_GAME_API WgResultCode game_get_function(uint32_t f, const char* n, void** o) { return fgmock_get_function(f, n, o); }
FG_GAME_API WgResultCode game_tag(const WgViewport* v, const WgTag* t, uint32_t n, void* c) { return fgmock_tag(v, t, n, c); }
FG_GAME_API WgResultCode game_tag_for_frame(const WgFrameToken* k, const WgViewport* v, const WgTag* t, uint32_t n, void* c) {
    return fgmock_tag_for_frame(k, v, t, n, c);
}
FG_GAME_API WgResultCode game_constants(const WgConstants* c, const WgFrameToken* k, const WgViewport* v) {
    return fgmock_constants(c, k, v);
}
FG_GAME_API WgResultCode game_evaluate(uint32_t f, const WgFrameToken* k, const WgHeader* const* in, uint32_t n, void* c) {
    return fgmock_evaluate(f, k, in, n, c);
}
FG_GAME_API WgResultCode game_free(uint32_t f, const WgViewport* v) { return fgmock_free(f, v); }
FG_GAME_API WgResultCode game_is_loaded(uint32_t f, bool* l) { return fgmock_is_loaded(f, l); }
FG_GAME_API WgResultCode game_set_loaded(uint32_t f, bool l) { return fgmock_set_loaded(f, l); }
FG_GAME_API WgResultCode game_init(uint64_t v) { return fgmock_init(v); }
FG_GAME_API HRESULT game_create_factory2(UINT flags, REFIID riid, void** out) {
    return fgmock_create_factory2(flags, riid, out);
}

// What each import slot holds now (the address of a dllimport function is read from its slot at run time).
FG_GAME_API void game_slot_values(void** out) {
    out[0] = reinterpret_cast<void*>(&fgmock_is_supported);
    out[1] = reinterpret_cast<void*>(&fgmock_requirements);
    out[2] = reinterpret_cast<void*>(&fgmock_get_function);
    out[3] = reinterpret_cast<void*>(&fgmock_tag);
    out[4] = reinterpret_cast<void*>(&fgmock_tag_for_frame);
    out[5] = reinterpret_cast<void*>(&fgmock_constants);
    out[6] = reinterpret_cast<void*>(&fgmock_evaluate);
    out[7] = reinterpret_cast<void*>(&fgmock_free);
    out[8] = reinterpret_cast<void*>(&fgmock_is_loaded);
    out[9] = reinterpret_cast<void*>(&fgmock_set_loaded);
    out[10] = reinterpret_cast<void*>(&fgmock_init);
    out[11] = reinterpret_cast<void*>(&fgmock_create_factory2);
}
