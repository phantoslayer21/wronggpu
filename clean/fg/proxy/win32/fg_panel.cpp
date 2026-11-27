// SPDX-License-Identifier: GPL-3.0-only
// WrongGPU frame generation: WrongGPU-FG.asi's "WrongGPU Frame Generation" card in OptiScaler's menu. See fg_panel.h.
// Written for this project.
#include "fg_panel.h"

#include <mutex>

#include "../../presenter/fg_presenter_factory.h"
#include "fg_asi_transport.h"
#include "wg_state.h"

namespace wgfgp {
namespace win32 {
namespace panel {
namespace {

const WgHostMenuApi* g_api = nullptr;
WgHostMenuPlugin* g_plugin = nullptr;
bool g_card = false;  // a card of the host's Custom tab (else a tab of its own: a host without cards)
Hooks g_hooks;
std::mutex g_mu;      // the settings copy (menu thread; tests)
FgSettings g_settings;

void Log(const std::string& line) {
    if (g_hooks.log) g_hooks.log(line.c_str());
}

const char* PresenterWords(uint32_t p) {
    switch (static_cast<wgfg::FgPresenterChoice>(p)) {
        case wgfg::FgPresenterChoice::WindowsCompositionHwfq: return "Windows composition, hardware flip queue";
        case wgfg::FgPresenterChoice::DxgiFlipTearing: return "DXGI flip, tearing";
        case wgfg::FgPresenterChoice::DxgiFlipVsync: return "DXGI flip, v-sync";
        default: return nullptr;
    }
}

// The settings.ini write and its words (the live change already happened or was not possible).
std::string Save(const char* key, const char* value) {
    const std::wstring dir = g_hooks.game_dir ? g_hooks.game_dir() : std::wstring();
    if (dir.empty()) return "not saved: the game folder is not known";
    const wgstate::SetOutcome o = wgstate::SetSettings(dir, {{key, value}}, 1000);
    return o.ok() ? std::string("saved to wronggpu-setup\\settings.ini") : "not saved: " + o.words();
}

std::string Choose(bool set_preset, uint32_t preset, bool set_mult, uint32_t multiplier, const char* key,
                   const char* value) {
    bool live = false;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        if (set_preset) g_settings.preset = preset;
        if (set_mult) g_settings.multiplier = multiplier;
        live = g_hooks.apply && g_hooks.apply(g_settings.preset, g_settings.multiplier);
    }
    const std::string line = std::string("fg menu: ") + key + "=" + value + " chosen in OptiScaler's menu (" +
                             (live ? "from the next frame" : "from the next game start") + ", " + Save(key, value) +
                             ")";
    Log(line);
    return line;
}

void Draw(void*) {
    const WgHostMenuApi* api = g_api;
    if (!api) return;
    if (!g_card) api->section("WrongGPU Frame Generation");
    const Status st = g_hooks.status ? g_hooks.status() : Status();
    const FgSettings cur = Current();
    const std::string line = StatusLine(st, cur);
    if (!st.enabled) {
        api->text(line.c_str());
        api->text_disabled("Turn frame generation on in the WrongGPU Launcher; it starts with the next game start.");
        return;
    }
    const bool good = st.installed && st.have_host && st.live.phase == kFgPhaseGenerating;
    const bool bad = !st.installed || !st.have_host || st.live.phase == kFgPhaseStopped;
    if (good)
        api->text_colored(0.45f, 0.85f, 0.45f, 1.0f, line.c_str());
    else if (bad)
        api->text_colored(0.95f, 0.65f, 0.25f, 1.0f, line.c_str());
    else
        api->text(line.c_str());
    if (st.installed && st.have_host && st.live.phase == kFgPhaseWaiting)
        api->text_disabled("Frame generation starts when the game turns on DLSS Frame Generation.");

    static const char* const kPresets[3] = {"A", "B", "C"};
    int32_t p = cur.preset < 3 ? static_cast<int32_t>(cur.preset) : 0;
    if (api->combo("Preset##wg_fg_preset", &p, kPresets, 3) && p >= 0 && p < 3 && static_cast<uint32_t>(p) != cur.preset)
        ChoosePreset(static_cast<uint32_t>(p));
    api->item_tooltip("A: the full model (the default).\n"
                      "B: the full model, keeping the game's UI from the real frame (games that hand over their UI).\n"
                      "C: the light model, the fastest.\n"
                      "A game that asks for its UI to be kept gets B whatever is chosen here.");
    static const char* const kMultipliers[5] = {"2x", "3x", "4x", "5x", "6x"};
    int32_t m = cur.multiplier >= kFgMinMultiplier && cur.multiplier <= kFgMaxMultiplier
                    ? static_cast<int32_t>(cur.multiplier - kFgMinMultiplier)
                    : 0;
    if (api->combo("Multiplier##wg_fg_multiplier", &m, kMultipliers, 5) && m >= 0 && m < 5 &&
        static_cast<uint32_t>(m) + kFgMinMultiplier != cur.multiplier)
        ChooseMultiplier(static_cast<uint32_t>(m) + kFgMinMultiplier);
    api->item_tooltip("The most images shown per frame the game renders: 2x adds one generated image, 6x adds five.\n"
                      "The game's own frame generation setting asks for a count; this caps it.");
    api->text_disabled("Applies from the next frame and syncs to the Launcher.");
}

}  // namespace

std::string StatusLine(const Status& s, const FgSettings& current) {
    if (!s.enabled) return "Off (fg=off)";
    if (!s.installed)
        return std::string("Not active: ") + FgInstallResultWord(s.install_result) + " (see wronggpu-fg.log)";
    if (!s.have_host) return "Not active: no frame generation runtime for this game (see wronggpu-fg.log)";
    switch (s.live.phase) {
        case kFgPhaseGenerating: {
            const char* pr = PresenterWords(s.live.presenter);
            std::string l = "Generating: " + std::to_string(s.live.count + 1) + "x, preset " +
                            (FgPresetSpelling(s.live.preset) ? FgPresetSpelling(s.live.preset) : "?");
            if (pr) l += std::string(", ") + pr;
            if (s.live.count + 1 < current.multiplier) l += " (the game asks for fewer images)";
            return l;
        }
        case kFgPhaseRealOnly:
            return std::string("Real frames only: ") +
                   (s.live.reason ? FgReasonName(s.live.reason) : FgRealOnlyName(s.live.real_only));
        case kFgPhaseStopped:
            return "Stopped for this run: the frame generation session could not be created (see wronggpu-fg.log)";
        default:
            return "Waiting for the game";
    }
}

FgSettings Current() {
    std::lock_guard<std::mutex> lock(g_mu);
    return g_settings;
}

std::string ChoosePreset(uint32_t preset) {
    const char* spelling = FgPresetSpelling(preset);
    if (!spelling) return std::string();
    return Choose(true, preset, false, 0, kFgKeyPreset, spelling);
}

std::string ChooseMultiplier(uint32_t multiplier) {
    if (FgGeneratedCountOf(multiplier) == 0) return std::string();
    const std::string value = std::to_string(multiplier);
    return Choose(false, 0, true, multiplier, kFgKeyMultiplier, value.c_str());
}

bool Init(const WgHostMenuApi* api, WgHostMenuPlugin* plugin, const Hooks& hooks, const FgSettings& start) {
    g_hooks = hooks;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        g_settings = start;
    }
    if (!api || api->version < 1 || !WG_HOST_MENU_HAS(api, combo) || !WG_HOST_MENU_HAS(api, item_tooltip) ||
        !WG_HOST_MENU_HAS(api, text) || !WG_HOST_MENU_HAS(api, text_disabled) ||
        !WG_HOST_MENU_HAS(api, text_colored) || !WG_HOST_MENU_HAS(api, section) ||
        (!WG_HOST_MENU_HAS(api, register_custom_card) && !WG_HOST_MENU_HAS(api, register_tab))) {
        Log("fg menu: OptiScaler's plugin interface lacks what the WrongGPU Frame Generation card needs; no card");
        return false;
    }
    g_api = api;
    g_plugin = plugin;
    g_card = WG_HOST_MENU_HAS(api, register_custom_card);
    const bool added = g_card ? api->register_custom_card(plugin, "WrongGPU Frame Generation", &Draw, nullptr)
                              : api->register_tab(plugin, "WrongGPU Frame Generation", "WrongGPU frame generation",
                                                  &Draw, nullptr);
    if (!added) {
        g_api = nullptr;
        Log("fg menu: OptiScaler refused the WrongGPU Frame Generation card");
        return false;
    }
    Log(g_card ? "fg menu: WrongGPU Frame Generation card added to OptiScaler's Custom tab"
               : "fg menu: WrongGPU Frame Generation tab added to OptiScaler's menu (this OptiScaler has no Custom tab "
                 "cards)");
    return true;
}

}  // namespace panel
}  // namespace win32
}  // namespace wgfgp
