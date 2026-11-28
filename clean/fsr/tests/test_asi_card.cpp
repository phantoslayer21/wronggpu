// SPDX-License-Identifier: GPL-3.0-only
// WrongGPU.asi's "WrongGPU Upscaler" card (asi/sr_card.h) through the host's plugin interface, CPU only: the built plugin is
// loaded, OptiPluginInit gets a stand-in for OptiScaler's table (clean/wgshared/include/wg_host_menu_abi.h), and a draw
// with a combo change must write the shared settings block's owned slot (live, next frame) AND the game's state folder
// (<game>\wronggpu-setup\settings.ini) under the state lock. InitializeASI is never called; no foreign module is opened.
//     test_asi_card <path of WrongGPU.asi>
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#define WG_SETTINGS_IO_IMPLEMENTATION
#include "wg_settings_io.h"

#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "../../wgshared/include/wg_host_menu_abi.h"
#include "../../wgshared/include/wg_state.h"
#include "wg_sr_modes.h"
#include "../asi/asi_abi.h"

static int failures = 0, checks = 0;
#define CHECK(x) do { ++checks; if (!(x)) { ++failures; std::printf("FAIL line %d: %s\n", __LINE__, #x); } } while (0)

namespace {

WgHostDrawFn g_draw = nullptr;
std::string g_title;
bool g_tab = false;
std::vector<std::string> g_combos;   // labels drawn this frame
std::vector<std::string> g_texts;    // text / text_disabled drawn this frame
std::string g_pick;                  // the combo label that "changes" this frame
int32_t g_pickIndex = 0;

bool RegisterCard(WgHostMenuPlugin*, const char* title, WgHostDrawFn draw, void*) {
    g_title = title ? title : "";
    g_draw = draw;
    return true;
}
bool RegisterTab(WgHostMenuPlugin*, const char* name, const char*, WgHostDrawFn draw, void*) {
    g_title = name ? name : "";
    g_draw = draw;
    g_tab = true;
    return true;
}
void Section(const char* t) { g_texts.push_back(std::string("[") + t + "]"); }
void Text(const char* t) { g_texts.push_back(t); }
void Tooltip(const char*) {}
bool Combo(const char* label, int32_t* current, const char* const* items, int32_t count) {
    g_combos.push_back(label);
    if (count != 2 || std::string(items[0]) != "Fast" || std::string(items[1]) != "Exact") return false;
    if (g_pick == label) {
        *current = g_pickIndex;
        return true;
    }
    return false;
}
void HostLog(WgHostMenuPlugin*, int32_t, const char*) {}

WgHostMenuApi FullTable() {
    WgHostMenuApi api{};
    api.size = sizeof(api);
    api.version = 5;
    api.section = &Section;
    api.item_tooltip = &Tooltip;
    api.text = &Text;
    api.text_disabled = &Text;
    api.combo = &Combo;
    api.log = &HostLog;
    api.register_tab = &RegisterTab;
    api.register_custom_card = &RegisterCard;
    return api;
}

void DrawFrame(const std::string& pick = std::string(), int32_t index = 0) {
    g_combos.clear();
    g_texts.clear();
    g_pick = pick;
    g_pickIndex = index;
    if (g_draw) g_draw(nullptr);
}

bool HasText(const std::string& needle) {
    for (const std::string& t : g_texts)
        if (t.find(needle) != std::string::npos) return true;
    return false;
}

std::string SettingOf(const std::wstring& gameDir, const char* key) {
    const wgstate::Settings s = wgstate::ReadSettings(gameDir);
    const std::string* v = wgstate::Get(s, key);
    return v ? *v : std::string();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: test_asi_card <WrongGPU.asi>\n");
        return 2;
    }
    // a game folder with its state folder (the card writes there; a missing state folder is the launcher's to make)
    wchar_t tmp[MAX_PATH];
    GetTempPathW(MAX_PATH, tmp);
    const std::filesystem::path game = std::filesystem::path(tmp) / (L"wg_asi_card_" + std::to_wstring(GetCurrentProcessId()));
    std::error_code ec;
    std::filesystem::remove_all(game, ec);
    std::filesystem::create_directories(game / L"wronggpu-setup");
    _putenv_s("WG_M_MODE", "");
    _putenv_s("WG_K_MODE", "");
    _putenv_s("WG_L_MODE", "");

    HMODULE asi = LoadLibraryA(argv[1]);
    CHECK(asi != nullptr);
    if (!asi) return 1;
    auto init = reinterpret_cast<WgHostMenuInitFn>(GetProcAddress(asi, WG_HOST_MENU_INIT_EXPORT));
    auto setRecord = reinterpret_cast<void (*)(const wchar_t*)>(GetProcAddress(asi, "WrongGpuAsiSetChoiceRecordV1"));
    auto showModel = reinterpret_cast<void (*)(char)>(GetProcAddress(asi, "WrongGpuAsiShowModelV1"));
    CHECK(init && setRecord && showModel);
    if (!init || !setRecord || !showModel) return 1;
    setRecord((game / L"wronggpu-setup" / L"model-choice.ini").c_str());

    // 1. a table without what the card needs: nothing registered
    {
        WgHostMenuApi api = FullTable();
        api.combo = nullptr;
        CHECK(!init(&api, nullptr));
        CHECK(g_draw == nullptr);
        WgHostMenuApi none{};
        CHECK(!init(nullptr, nullptr) && !init(&none, nullptr));
    }
    // 2. a version-1 host (no Custom tab cards): a tab of its own
    {
        WgHostMenuApi api = FullTable();
        api.size = static_cast<uint32_t>(offsetof(WgHostMenuApi, register_custom_card));
        api.version = 1;
        CHECK(init(&api, nullptr));
        CHECK(g_tab && g_title == "WrongGPU Upscaler" && g_draw);
        DrawFrame();
        CHECK(HasText("[WrongGPU Upscaler]"));
    }
    // 3. the full table: the card on the Custom tab
    WgHostMenuApi api = FullTable();
    g_tab = false;
    g_draw = nullptr;
    CHECK(init(&api, nullptr));
    CHECK(!g_tab && g_title == "WrongGPU Upscaler" && g_draw);

    // no shared block yet (no SR front end in this process): settings.ini only, the default shows Fast
    showModel('M');
    DrawFrame();
    CHECK(HasText("Running: Model M (Fast)"));
    CHECK(HasText("Applies from the next frame and syncs to the Launcher."));
    CHECK(g_combos.size() == 3 && g_combos[0] == "Model M mode##wg_m_mode");
    DrawFrame("Model M mode##wg_m_mode", WG_SR_MODE_EXACT);
    CHECK(SettingOf(game, "m_mode") == "exact");
    DrawFrame();
    CHECK(HasText("Running: Model M (Exact)"));

    // the SR front end's block exists (created here as the front end would): the card writes the owned slot live
    wg::SharedSettings block;
    std::string lines;
    CHECK(block.open(WG_SS_WRITER_SR_FRONTEND, std::wstring(), &lines) == wg::SharedSettings::Status::Created);
    CHECK(block.claim_writer(WG_SS_WRITER_SR_FRONTEND, 0));  // another module is the writer: the card still owns its slots
    Sleep(1100);  // the card retries opening the block at most once a second
    showModel('K');
    DrawFrame();
#if WRONGGPU_MODEL_K
    CHECK(HasText("Running: Model K (Fast)"));
    const std::vector<std::string> expectedOrder = {"Model K mode##wg_k_mode", "Model M mode##wg_m_mode",
                                                    "Model L mode##wg_l_mode"};
#else
    // A report for an unavailable model leaves the active Model M and its saved mode in place.
    CHECK(HasText("Running: Model M (Exact)"));
    const std::vector<std::string> expectedOrder = {"Model M mode##wg_m_mode", "Model K mode##wg_k_mode",
                                                    "Model L mode##wg_l_mode"};
#endif
    CHECK(g_combos == expectedOrder);
    // The card also lets players set modes for models that are not running.
    DrawFrame("Model K mode##wg_k_mode", WG_SR_MODE_EXACT);
    wg::Snapshot snap;
    CHECK(block.snapshot(&snap));
    CHECK(snap.is_live(WG_OPT_SR_K_MODE) && snap.values[WG_OPT_SR_K_MODE] == WG_SR_MODE_EXACT);
    CHECK(SettingOf(game, "k_mode") == "exact");
    CHECK(SettingOf(game, "m_mode") == "exact");  // earlier choice kept
    // back to Fast for Model L (another model's combo)
    DrawFrame("Model L mode##wg_l_mode", WG_SR_MODE_EXACT);
    DrawFrame("Model L mode##wg_l_mode", WG_SR_MODE_FAST);
    CHECK(block.snapshot(&snap));
    CHECK(snap.is_live(WG_OPT_SR_L_MODE) && snap.values[WG_OPT_SR_L_MODE] == WG_SR_MODE_FAST);
    CHECK(SettingOf(game, "l_mode") == "fast");
    // the owned slots only: rr_mode and the SR model stay the writer's
    CHECK(!block.write_owned_slot(WG_OPT_RR_MODE, WG_RR_MODE_EXACT) && !block.write_owned_slot(WG_OPT_SR_MODEL, 1));
    CHECK(!block.write_owned_slot(WG_OPT_SR_M_MODE, 2));  // out of range
    // AMD's FSR selected: the card says so and still offers the modes
    showModel(0);
    DrawFrame();
    CHECK(HasText("AMD FSR is selected") && g_combos.size() == 3);

    // rc3: once the front end reports what it actually runs (WrongGpuAsiReportSrV1), the status lines come from that report,
    // never from the saved choice: switching, running with its mode, a model that could not start, idle.
    auto report = reinterpret_cast<void (*)(const WrongGpuAsiSrV1*)>(GetProcAddress(asi, WRONGGPU_ASI_REPORT_SR));
    CHECK(report != nullptr);
    if (report) {
        WrongGpuAsiSrV1 s{};
        s.size = sizeof(s);
        s.state = WG_ASI_SR_SWITCHING;
        s.running_model = 'M';
        s.pending_model = 'L';
        s.running_mode = WG_SR_MODE_FAST;
        report(&s);
        DrawFrame();
        CHECK(HasText("Switching to Model L... Model M (Fast) keeps running until it is ready.") && !HasText("Running:"));
        s.state = WG_ASI_SR_RUNNING;
        s.pending_model = 0;
        s.error_model = 'L';
        s.error_width = 1920;
        s.error_height = 1080;
        std::snprintf(s.error_reason, sizeof s.error_reason, "%s", "no converted Model L file was found");
        report(&s);
        DrawFrame();
        CHECK(HasText("Running: Model M (Fast)"));
        CHECK(HasText("Model L couldn't start at 1920x1080: no converted Model L file was found. Model M kept."));
        // the case of 2026-11-27: OptiScaler runs its stock FSR 2.1.2; the saved choice (or a shown model) is no "Running"
        s.state = WG_ASI_SR_IDLE;
        s.running_model = 0;
        report(&s);
        showModel('M');
        DrawFrame();
        CHECK(HasText("Idle: WrongGPU is not upscaling (OptiScaler is running another upscaler).") && !HasText("Running:"));
        CHECK(HasText("Model L couldn't start at 1920x1080: no converted Model L file was found.") && !HasText("kept"));
        CHECK(g_combos.size() == 3);
        WrongGpuAsiSrV1 small = s;  // a report smaller than the structure (an older or broken writer) is ignored
        small.size = 8;
        small.state = WG_ASI_SR_RUNNING;
        small.running_model = 'M';
        report(&small);
        DrawFrame();
        CHECK(HasText("Idle:"));
        s.state = WG_ASI_SR_RUNNING;  // the mode of the last frame, as reported
        s.running_model = 'M';
        s.running_mode = WG_SR_MODE_EXACT;
        s.error_model = 0;
        report(&s);
        DrawFrame();
        CHECK(HasText("Running: Model M (Exact)") && !HasText("couldn't start"));
    }

    block.close();
    std::filesystem::remove_all(game, ec);
    std::printf("test_asi_card: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
