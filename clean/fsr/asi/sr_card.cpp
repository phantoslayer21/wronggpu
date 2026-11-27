// SPDX-License-Identifier: GPL-3.0-only
// WrongGPU.asi's "WrongGPU Upscaler" card in OptiScaler's menu (sr_card.h).
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#define WG_SETTINGS_IO_IMPLEMENTATION
#include "wg_settings_io.h"  // the shared settings block (this DLL's own copy of the one implementation)

#include "sr_card.h"

#include <mutex>

#include "../../wgshared/include/wg_state.h"  // the game's state folder: settings.ini, its lock, atomic writes
#include "wg_sr_modes.h"

namespace wgasi {
namespace card {
namespace {

const WgHostMenuApi* g_api = nullptr;
WgHostMenuPlugin* g_plugin = nullptr;
bool g_card = false;  // a card of the host's Custom tab (else a tab of its own: a host without cards)
Hooks g_hooks;

std::mutex g_mutex;          // the block and the settings cache below (menu thread; tests)
wg::SharedSettings g_block;  // opened when it exists (created by the SR front end or the RR plugin; never created here)
uint64_t g_blockTried = 0;   // GetTickCount64 of the last attempt to open it
wgstate::Settings g_saved;   // settings.ini as last read
uint64_t g_savedAt = 0;      // when it was read (0: never / stale)

void Log(const std::string& line) {
    if (g_hooks.log) g_hooks.log(line);
    if (g_api && WG_HOST_MENU_HAS(g_api, log)) g_api->log(g_plugin, 1, line.c_str());
}

std::wstring GameDir() { return g_hooks.gameDir ? g_hooks.gameDir() : std::wstring(); }

// The block, opened when another module created it (retried at most once a second while it does not exist). g_mutex held.
wg::SharedSettings& BlockLocked() {
    const uint64_t now = GetTickCount64();
    if (!g_block.present() && (g_blockTried == 0 || now - g_blockTried >= 1000)) {
        g_blockTried = now;
        std::string lines;
        g_block.open(WG_SS_WRITER_SR_FRONTEND, std::wstring(), &lines, nullptr, /*existingOnly=*/true);
        if (g_block.present()) Log("card: shared settings block opened (the SR front end switches the mode from the next frame)");
    }
    return g_block;
}

const char* ModelWords(char model) { return model == 'K' ? "Model K" : model == 'L' ? "Model L" : "Model M"; }

void Draw(void*) {
    const WgHostMenuApi* api = g_api;
    if (!api) return;
    const char active = g_hooks.activeModel ? g_hooks.activeModel() : 'M';
    if (!g_card) api->section("WrongGPU Upscaler");
    static const char* const kTitles[2] = {"Fast", "Exact"};
    const char shown = wg_sr_mode_of(active) ? active : 'M';
    if (wg_sr_mode_of(active)) {
        const std::string line = std::string("Running: ") + ModelWords(active) + " (" + wg_sr_mode_title(CurrentMode(active)) + ")";
        api->text(line.c_str());
    } else {
        api->text_disabled("AMD FSR is selected in OptiScaler's upscaler list. WrongGPU's modes apply when a WrongGPU model runs.");
    }
    // the model that runs first, then the other models (clearly named)
    const char order[3] = {shown, shown == 'M' ? 'K' : 'M', shown == 'L' ? 'K' : 'L'};
    for (int i = 0; i < 3; ++i) {
        const char model = order[i];
        if (i == 1) api->text_disabled("Other models:");
        int32_t index = CurrentMode(model);
        const std::string label = std::string(ModelWords(model)) + " mode##wg_" + static_cast<char>(model - 'A' + 'a') + "_mode";
        if (api->combo(label.c_str(), &index, kTitles, 2) && (index == WG_SR_MODE_FAST || index == WG_SR_MODE_EXACT) &&
            index != CurrentMode(model))
            Choose(model, index);
        api->item_tooltip("Fast: the default. The quickest, with a picture very close to Exact.\n"
                          "Exact: the reference result, the slowest.");
    }
    api->text_disabled("Applies from the next frame and syncs to the Launcher.");
}

}  // namespace

int32_t CurrentMode(char model) {
    const WgSrModeKey* m = wg_sr_mode_of(model);
    if (!m) return WG_SR_MODE_FAST;
    std::lock_guard lock(g_mutex);
    wg::SharedSettings& block = BlockLocked();
    wg::Snapshot snap;
    if (block.present() && block.snapshot(&snap, 1ull << m->slot) && snap.is_live(m->slot) &&
        wg::value_in_range(m->slot, snap.values[m->slot]))
        return snap.values[m->slot];
    const uint64_t now = GetTickCount64();
    if (g_savedAt == 0 || now - g_savedAt >= 1000) {
        const std::wstring dir = GameDir();
        g_saved = dir.empty() ? wgstate::Settings{} : wgstate::ReadSettings(dir);
        g_savedAt = now ? now : 1;
    }
    if (const std::string* v = wgstate::Get(g_saved, m->state_key))
        if (wg_parse_sr_mode(*v) >= 0) return wg_parse_sr_mode(*v);
    std::string e;
    const WgOptionDef& o = kWgOptions[m->slot];
    if (o.env && *o.env && wg::get_env(o.env, &e) && wg_parse_sr_mode(e) >= 0) return wg_parse_sr_mode(e);
    return o.default_value;
}

std::string Choose(char model, int32_t mode) {
    const WgSrModeKey* m = wg_sr_mode_of(model);
    if (!m || (mode != WG_SR_MODE_FAST && mode != WG_SR_MODE_EXACT)) return std::string();
    bool live = false;
    {
        std::lock_guard lock(g_mutex);
        live = BlockLocked().write_owned_slot(m->slot, mode);
        g_savedAt = 0;  // re-read settings.ini after this write
    }
    const std::wstring dir = GameDir();
    std::string saved;
    if (dir.empty()) {
        saved = "not saved: the game folder is not known";
    } else {
        const wgstate::SetOutcome o = wgstate::SetSettings(dir, {{m->state_key, wg_sr_mode_name(mode)}}, 1000);
        saved = o.ok() ? std::string("saved to wronggpu-setup\\settings.ini") : "not saved: " + o.words();
    }
    const std::string line = std::string("card: ") + m->state_key + "=" + wg_sr_mode_name(mode) + " chosen in OptiScaler's menu (" +
                             (live ? "from the next frame" : "shared settings unavailable: from the next upscaler create") + ", " +
                             saved + ")";
    Log(line);
    return line;
}

bool Init(const WgHostMenuApi* api, WgHostMenuPlugin* plugin, Hooks hooks) {
    g_hooks = std::move(hooks);
    if (!api || api->version < 1 || !WG_HOST_MENU_HAS(api, combo) || !WG_HOST_MENU_HAS(api, item_tooltip) ||
        !WG_HOST_MENU_HAS(api, text) || !WG_HOST_MENU_HAS(api, text_disabled) || !WG_HOST_MENU_HAS(api, section) ||
        (!WG_HOST_MENU_HAS(api, register_custom_card) && !WG_HOST_MENU_HAS(api, register_tab))) {
        if (g_hooks.log) g_hooks.log("card: OptiScaler's plugin interface lacks what the WrongGPU Upscaler card needs; no card");
        return false;
    }
    g_api = api;
    g_plugin = plugin;
    g_card = WG_HOST_MENU_HAS(api, register_custom_card);
    const bool added = g_card ? api->register_custom_card(plugin, "WrongGPU Upscaler", &Draw, nullptr)
                              : api->register_tab(plugin, "WrongGPU Upscaler", "WrongGPU upscaler mode", &Draw, nullptr);
    if (!added) {
        g_api = nullptr;
        if (g_hooks.log) g_hooks.log("card: OptiScaler refused the WrongGPU Upscaler card");
        return false;
    }
    Log(g_card ? "card: WrongGPU Upscaler card added to OptiScaler's Custom tab"
               : "card: WrongGPU Upscaler tab added to OptiScaler's menu (this OptiScaler has no Custom tab cards)");
    return true;
}

}  // namespace card
}  // namespace wgasi
