// SPDX-License-Identifier: GPL-3.0-only
// WrongGPU frame generation controls in OptiScaler's menu. See fg_panel.h.
// Written for this project.
#include "fg_panel.h"

#include <atomic>
#include <mutex>

#include "../../presenter/fg_presenter_factory.h"
#include "fg_asi_transport.h"
#include "wg_state.h"

namespace wgfgp {
namespace win32 {
namespace panel {
namespace {

const WgHostMenuApi* g_api = nullptr;
bool g_card = false;  // a card of the host's Custom tab (else a tab of its own: a host without cards)
Hooks g_hooks;
std::mutex g_mu;      // the settings copy (menu thread; tests)
FgSettings g_settings;
std::string g_note;
std::mutex g_report_mu;  // serializes a status update with shutdown
const WgHostMenuApi* g_report_api = nullptr;
WgHostMenuPlugin* g_report_plugin = nullptr;
std::atomic<uint64_t> g_report_sequence{0};
uint64_t g_published_sequence = 0;
uint64_t g_report_at = 0;

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

const char* PresenterShort(uint32_t p) {
    switch (static_cast<wgfg::FgPresenterChoice>(p)) {
        case wgfg::FgPresenterChoice::WindowsCompositionHwfq: return "HWFQ";
        case wgfg::FgPresenterChoice::DxgiFlipTearing: return "DXGI tearing";
        case wgfg::FgPresenterChoice::DxgiFlipVsync: return "DXGI V-Sync";
        default: return "";
    }
}

bool GameControlsMultiplier(const Status& status) {
    return status.live.multiplier_source == kFgMultSourceGame;
}

std::string FrameGenStatus(const Status& st) {
    if (!st.enabled) return "Off";
    if (st.host_conflict) return "Real frames only: OptiScaler already owns the FG output";
    if (st.deferred) return "Waiting";
    if (!st.installed)
        return std::string("Real frames only: ") + FgInstallResultWord(st.install_result);
    if (!st.have_host) return "Real frames only: frame generation runtime unavailable";
    if (st.live.phase == kFgPhaseStopped) return "Real frames only: frame generation session could not start";
    if (st.live.phase == kFgPhaseRealOnly) {
        if (!st.live.reason && st.live.real_only == WG_FG_REAL_ONLY_DISABLED) return "Waiting";
        return std::string("Real frames only: ") +
               (st.live.reason ? FgReasonName(st.live.reason) : FgRealOnlyName(st.live.real_only));
    }
    if (st.live.phase != kFgPhaseGenerating) return "Waiting";
    if (!st.live.scheduled_count) {
        const auto reason = static_cast<wgfg::FgRealOnlyReason>(st.live.schedule_reason);
        if (reason == wgfg::FgRealOnlyReason::None || reason == wgfg::FgRealOnlyReason::NotPrimed) return "Waiting";
        return std::string("Real frames only: ") + wgfg::FgRealOnlyReasonName(reason);
    }
    const char* preset = FgPresetSpelling(st.live.preset);
    const char* presenter = PresenterShort(st.live.presenter);
    std::string line = "Generating " + std::to_string(st.live.scheduled_count + 1) + "x";
    if (GameControlsMultiplier(st)) line += " (game)";
    line += std::string(", preset ") + (preset ? preset : "?");
    if (*presenter) line += std::string(", ") + presenter;
    return line;
}

void RememberSave(const char* key, const char* value, bool live) {
    const std::string saved = Save(key, value);
    const bool ok = saved.rfind("saved", 0) == 0;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        g_note = !ok ? saved : live ? "" : "Saved. Restart the game to apply this change.";
    }
    Log(std::string("fg menu: ") + key + "=" + value + " (" + saved + ")");
}

void DrawMultiplier(const WgHostMenuApi* api, const Status& st, const FgSettings& cur) {
    if (st.live.game_has_mfg_selector) {
        std::string label = "Multiplier";
        if (st.live.game_requested_count) label += ": " + std::to_string(st.live.game_requested_count + 1) + "x";
        label += " (set in the game's graphics menu)";
        api->text(label.c_str());
        return;
    }

    static const char* const labels[] = {"Auto", "2x", "3x", "4x", "5x", "6x"};
    static constexpr uint32_t values[] = {kFgAutoMultiplier, 2, 3, 4, 5, 6};
    int32_t m = cur.multiplier == kFgAutoMultiplier ? 0 : static_cast<int32_t>(cur.multiplier) - 1;
    if (api->combo("Multiplier##wg_fg_multiplier", &m, labels, 6) && m >= 0 && m < 6 && values[m] != cur.multiplier)
        ChooseMultiplier(values[m]);
    api->item_tooltip("Auto follows the game and uses 2x when no count is supplied.");
}

void DrawFrameGen(void*) {
    const WgHostMenuApi* api = g_api;
    if (!api) return;
    const Status st = g_hooks.status ? g_hooks.status() : Status();
    const FgSettings cur = Current();
    const std::string line = FrameGenStatus(st);
    api->text(line.c_str());

    bool enabled = cur.enabled;
    if (api->checkbox("WrongGPU Frame Generation##wg_fg_enable", &enabled) && enabled != cur.enabled) {
        const bool live = g_hooks.enable && g_hooks.enable(enabled);
        {
            std::lock_guard<std::mutex> lock(g_mu);
            g_settings.enabled = enabled;
        }
        RememberSave(kFgKeyEnable, enabled ? "on" : "off", live);
    }

    static const char* const presets[] = {"A", "B", "C"};
    static const char* const descriptions[] = {
        "A uses the full model.",
        "B uses the full model and keeps the game's UI when provided.",
        "C uses the lighter model for higher performance.",
    };
    int32_t p = cur.preset < 3 ? static_cast<int32_t>(cur.preset) : 0;
    if (api->combo("Preset##wg_fg_preset", &p, presets, 3) && p >= 0 && p < 3 &&
        static_cast<uint32_t>(p) != cur.preset)
        ChoosePreset(static_cast<uint32_t>(p));
    api->item_tooltip("A uses the full model.\nB keeps the game's UI when provided.\nC uses the lighter model.");
    if (p >= 0 && p < 3) api->text_disabled(descriptions[p]);

    static const char* const presenters[] = {"Auto", "HWFQ", "DXGI tearing", "DXGI V-Sync"};
    int32_t presenter = static_cast<int32_t>(cur.presenter);
    if (api->combo("Presenter##wg_fg_presenter", &presenter, presenters, 4)) {
        const auto choice = static_cast<FgPresenterChoice>(presenter);
        if (const char* value = FgPresenterSpelling(choice); value && choice != cur.presenter) {
            {
                std::lock_guard<std::mutex> lock(g_mu);
                g_settings.presenter = choice;
            }
            RememberSave(kFgKeyPresenter, value, false);
        }
    }
    api->item_tooltip("Presenter changes apply when the game restarts.");

    DrawMultiplier(api, st, cur);

    const char* cap_reason = nullptr;
    switch (st.live.count_cap_reason) {
        case kFgCapDisplayVrr: cap_reason = "display cap"; break;
        case kFgCapRoute: cap_reason = "route limit"; break;
        case kFgCapLatencyCap: cap_reason = "latency"; break;
        case kFgCapPressure: cap_reason = "presenter pressure"; break;
        default: break;
    }
    if (st.enabled && st.live.phase == kFgPhaseGenerating && cap_reason && st.live.scheduled_count &&
        st.live.scheduled_count < st.live.requested_count) {
        const auto cap = "Scheduled: " + std::to_string(st.live.scheduled_count + 1) + "x (" + cap_reason + ")";
        api->text_disabled(cap.c_str());
    }

    std::string note;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        note = g_note;
    }
    api->text_disabled(note.empty() ? "Changes save automatically." : note.c_str());
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
    DrawMultiplier(api, st, cur);
    api->text_disabled("Applies from the next frame and syncs to the Launcher.");
}

}  // namespace

std::string StatusLine(const Status& s, const FgSettings&) {
    if (!s.enabled) return "Off (fg=off)";
    if (s.host_conflict) return "Real frames only: OptiScaler already owns the FG output";
    if (!s.installed)
        return std::string("Not active: ") + FgInstallResultWord(s.install_result) + " (see wronggpu-fg.log)";
    if (!s.have_host) return "Not active: no frame generation runtime for this game (see wronggpu-fg.log)";
    switch (s.live.phase) {
        case kFgPhaseGenerating: {
            const char* pr = PresenterWords(s.live.presenter);
            std::string l = "Generating: " + std::to_string(s.live.count + 1) + "x, preset " +
                            (FgPresetSpelling(s.live.preset) ? FgPresetSpelling(s.live.preset) : "?");
            if (pr) l += std::string(", ") + pr;
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
    if (multiplier != kFgAutoMultiplier && FgGeneratedCountOf(multiplier) == 0) return std::string();
    const std::string value = multiplier == kFgAutoMultiplier ? "auto" : std::to_string(multiplier);
    return Choose(false, 0, true, multiplier, kFgKeyMultiplier, value.c_str());
}

bool Init(const WgHostMenuApi* api, WgHostMenuPlugin* plugin, const Hooks& hooks, const FgSettings& start) {
    g_hooks = hooks;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        g_settings = start;
        g_note.clear();
    }
    if (!api || api->version < 1 || !WG_HOST_MENU_HAS(api, combo) || !WG_HOST_MENU_HAS(api, item_tooltip) ||
        !WG_HOST_MENU_HAS(api, text) || !WG_HOST_MENU_HAS(api, text_disabled) ||
        !WG_HOST_MENU_HAS(api, text_colored) || !WG_HOST_MENU_HAS(api, section) ||
        (!HasFrameGenApi(api) && !WG_HOST_MENU_HAS(api, register_custom_card) && !WG_HOST_MENU_HAS(api, register_tab))) {
        Log("fg menu: OptiScaler's plugin interface lacks what the WrongGPU Frame Generation card needs; no card");
        return false;
    }
    g_api = api;
    if (HasFrameGenApi(api)) {
        if (!WG_HOST_MENU_HAS(api, checkbox) || !api->register_frame_gen_section(plugin, &DrawFrameGen, nullptr)) {
            g_api = nullptr;
            Log("fg menu: OptiScaler refused the WrongGPU Frame Gen section");
            return false;
        }
        Log("fg menu: WrongGPU controls added to OptiScaler's Frame Gen tab");
        return true;
    }
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

bool HasFrameGenApi(const WgHostMenuApi* api) {
    return api && api->version >= 6 && WG_HOST_MENU_HAS(api, report_frame_generation) &&
           WG_HOST_MENU_HAS(api, register_frame_gen_section);
}

bool ClaimOutput(const WgHostMenuApi* api, WgHostMenuPlugin* plugin) {
    std::lock_guard<std::mutex> lock(g_report_mu);
    if (g_report_api) return api == g_report_api && plugin == g_report_plugin;
    if (!HasFrameGenApi(api) ||
        !api->report_frame_generation(plugin, "WrongGPU", "Waiting", nullptr, "DLSS Frame Generation", 0.0f, 0.0f, false))
        return false;
    g_report_api = api;
    g_report_plugin = plugin;
    return true;
}

void ReleaseOutput() {
    std::lock_guard<std::mutex> lock(g_report_mu);
    if (g_report_api) {
        g_report_api->report_frame_generation(g_report_plugin, nullptr, nullptr, nullptr, nullptr, 0.0f, 0.0f, false);
        g_report_api = nullptr;
        g_report_plugin = nullptr;
        g_published_sequence = g_report_sequence.load(std::memory_order_relaxed);
        g_report_at = 0;
    }
}

uint64_t NextReportSequence() noexcept {
    return g_report_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
}

void Publish(const Status& status, uint64_t sequence, bool immediate) noexcept {
    try {
        std::lock_guard<std::mutex> lock(g_report_mu);
        if (!g_report_api || sequence <= g_published_sequence) return;
        const uint64_t now = GetTickCount64();
        if (!immediate && now - g_report_at < 100) return;
        g_published_sequence = sequence;
        g_report_at = now;

        const std::string text = FrameGenStatus(status);
        const bool active = text.rfind("Generating ", 0) == 0;
        const std::string state = active ? std::to_string(status.live.scheduled_count + 1) + "x"
                                         : text.rfind("Real frames only:", 0) == 0 ? "Real only" : text;
        g_report_api->report_frame_generation(g_report_plugin, "WrongGPU", state.c_str(), text.c_str(),
                                              "DLSS Frame Generation", active ? status.live.displayed_fps : 0.0f,
                                              active ? status.live.base_fps : 0.0f, active);
    } catch (...) {
        // Menu telemetry must not affect the game's Present.
    }
}

}  // namespace panel
}  // namespace win32
}  // namespace wgfgp
