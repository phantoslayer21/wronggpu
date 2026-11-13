// SPDX-License-Identifier: MIT
// The RR plugin's box in OptiScaler's menu (rr_panel.h).
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "rr_panel.h"

#include <ctype.h>

#include <atomic>
#include <mutex>
#include <vector>

#include "../include/wg_settings_reader.h"
#include "settings.h"

namespace wgrr {
namespace panel {
namespace {

LogFn g_log = nullptr;
ActiveFn g_active = nullptr;
GpuMsFn g_gpu_ms = nullptr;
PreparingFn g_preparing = nullptr;
const WgHostMenuApi* g_api = nullptr;
WgHostMenuPlugin* g_plugin = nullptr;
bool g_card = false;  // the box is a card of the host's Custom tab (else a tab of its own: a host without cards)
wgss::Block g_block;
std::atomic<bool> g_exact_allowed{false};
std::atomic<Settings*> g_settings{nullptr};

void log(const std::string& s) {
  if (g_log) g_log(s);
}

std::wstring settings_file() { return wgss::data_dir() + L"wronggpu.ini"; }

// rr_mode as it stands (block slot, else WG_RR_MODE, else wronggpu.ini, else the default), as a mode id.
int32_t current_mode() { return resolved_mode().mode; }

void choose(int32_t mode) {
  const bool live = g_block.write_panel_slot(WG_OPT_RR_MODE, mode);
  const bool saved = wgss::ini_set(settings_file(), "wronggpu", "rr_mode", kWgRrModeNames[mode]);
  log(std::string("menu: rr_mode=") + kWgRrModeNames[mode] + " chosen in OptiScaler's menu (" +
      (live ? "from the next frame" : "shared settings unavailable: from the next start") +
      (saved ? ", saved to wronggpu.ini)" : ", wronggpu.ini not written)"));
}

void draw(void*) {
  const WgHostMenuApi* api = g_api;
  if (!api) return;
  if (!g_active || !g_active()) {  // no RR options while RR does not run (owner 2026-10-31)
    if (g_card) {
      api->text_disabled("Ray reconstruction is not running in this game right now.");
    } else {
      api->text("Ray reconstruction is not running in this game right now.");
      api->text_disabled("Its mode can be chosen here while the game uses ray reconstruction.");
    }
    return;
  }
  const bool exact = g_exact_allowed.load();
  int32_t modes[WG_RR_MODE_COUNT];
  const int n = offered_modes(exact, modes);
  const char* titles[WG_RR_MODE_COUNT];
  const int32_t now = effective_mode(current_mode(), exact);
  int32_t index = 0;
  for (int i = 0; i < n; ++i) {
    titles[i] = mode_title(modes[i]);
    if (modes[i] == now) index = i;
  }
  if (!g_card) api->section("Ray reconstruction");  // a card is already open, titled "WrongGPU"
  if (api->combo(g_card ? "Ray reconstruction##rr_mode" : "Mode##rr_mode", &index, titles, n) && index >= 0 && index < n &&
      modes[index] != now)
    choose(modes[index]);
  api->item_tooltip(exact ? "Exact: the reference result, the slowest.\nAccurate: nearly the same picture, faster.\n"
                            "Fast: the quickest, a little less detail."
                          : "Accurate: the default.\nFast: the quickest, a little less detail.");
  if (Settings* s = g_settings.load()) {
    bool on = s->get(kSetHitDistanceFilter) != 0;
    if (WG_HOST_MENU_HAS(api, checkbox) && api->checkbox("Reflection speckle filter##rr_hdf", &on)) {
      s->set(kSetHitDistanceFilter, on ? 1 : 0);
      log(std::string("menu: hit_distance_filter=") + (on ? "1" : "0") + " chosen in OptiScaler's menu (from the next frame)");
    }
    api->item_tooltip("Filters lone bad reflection distances from the game, which show up as dark or bright dots in\n"
                      "reflections while the camera moves (Cyberpunk). Off = native behaviour.");
  }
  api->text_disabled("Applies from the next frame.");
}

}  // namespace

int offered_modes(bool exact_allowed, int32_t* out) {
  int n = 0;  // the registry's modes in their order (a mode added to the registry shows up here with no code change)
  for (int32_t m = 0; m < WG_RR_MODE_COUNT; ++m)
    if (m != WG_RR_MODE_EXACT || exact_allowed) out[n++] = m;
  return n;
}

int32_t effective_mode(int32_t mode, bool exact_allowed) {
  if (mode < 0 || mode >= WG_RR_MODE_COUNT) return WG_RR_MODE_ACCURATE;
  return mode == WG_RR_MODE_EXACT && !exact_allowed ? WG_RR_MODE_ACCURATE : mode;
}

const char* mode_title(int32_t mode) {
  // the registry's mode name with a capital first letter ("accurate" -> "Accurate"), one copy per mode
  static std::string titles[WG_RR_MODE_COUNT];
  static std::once_flag once;
  std::call_once(once, [] {
    for (int32_t m = 0; m < WG_RR_MODE_COUNT; ++m) {
      titles[m] = kWgRrModeNames[m];
      if (!titles[m].empty()) titles[m][0] = (char)toupper((unsigned char)titles[m][0]);
    }
  });
  return mode >= 0 && mode < WG_RR_MODE_COUNT ? titles[mode].c_str() : titles[WG_RR_MODE_ACCURATE].c_str();
}

void start(LogFn log_fn, ActiveFn rr_active, GpuMsFn gpu_ms, PreparingFn preparing) {
  g_log = log_fn;
  g_preparing = preparing;
  g_active = rr_active;
  g_gpu_ms = gpu_ms;
  std::vector<std::string> lines;
  g_block.open(WG_SS_WRITER_RR_PLUGIN, true, wgss::exe_dir() + L"wronggpu-rr.ini", &lines);
  for (const std::string& l : lines) log(l);
  wgss::Snapshot snap;
  const bool have = g_block.present() && g_block.snapshot(WG_READER_RR_PLUGIN, &snap);
  std::vector<wgss::IniEntry> ini;
  if (!have) wgss::read_ini(settings_file(), &ini);
  const bool allowed = wgss::resolve(kWgOptions[WG_OPT_RR_EXACT_ALLOWED], have ? &snap : nullptr, have ? nullptr : &ini).value != 0;
  g_exact_allowed.store(allowed);
  const int32_t mode = current_mode();
  if (mode == WG_RR_MODE_EXACT && !allowed) {
    const bool live = g_block.write_panel_slot(WG_OPT_RR_MODE, WG_RR_MODE_ACCURATE);
    log(std::string("settings: rr_mode=exact without rr_exact_allowed=1 (turned on in the WrongGPU launcher): accurate runs") +
        (live ? "" : " (shared settings unavailable)"));
  }
  log(std::string("settings: exact mode ") + (allowed ? "allowed" : "not offered") + " (rr_exact_allowed, read at start)");
}

bool exact_allowed() { return g_exact_allowed.load(); }

ModeSetting resolved_mode() {
  ModeSetting m;
  wgss::Snapshot snap;
  m.shared = g_block.present() && g_block.snapshot(WG_READER_RR_PLUGIN, &snap);
  std::vector<wgss::IniEntry> ini;
  if (!m.shared) wgss::read_ini(settings_file(), &ini);
  const wgss::Resolved r = wgss::resolve(kWgOptions[WG_OPT_RR_MODE], m.shared ? &snap : nullptr,
                                         m.shared ? nullptr : &ini, "wronggpu.ini");
  m.mode = r.value;
  m.set = strcmp(r.source, "default") != 0;
  m.source = m.shared && snap.is_set(WG_OPT_RR_MODE) ? std::string("shared settings, ") + r.source : std::string(r.source);
  return m;
}

void bind_settings(Settings* settings) { g_settings.store(settings); }

bool menu_init(const WgHostMenuApi* api, WgHostMenuPlugin* plugin) {
  if (!api || api->version < 1 || !WG_HOST_MENU_HAS(api, combo) || !WG_HOST_MENU_HAS(api, item_tooltip) ||
      !WG_HOST_MENU_HAS(api, text_disabled) || !WG_HOST_MENU_HAS(api, section) ||
      (!WG_HOST_MENU_HAS(api, register_custom_card) && !WG_HOST_MENU_HAS(api, register_tab))) {
    log("menu: OptiScaler's plugin interface lacks what the WrongGPU box needs; no box");
    return false;
  }
  g_api = api;
  g_plugin = plugin;
  g_card = WG_HOST_MENU_HAS(api, register_custom_card);
  const bool added = g_card ? api->register_custom_card(plugin, "WrongGPU", &draw, nullptr)
                            : api->register_tab(plugin, "WrongGPU", "WrongGPU ray reconstruction", &draw, nullptr);
  if (!added) {
    g_api = nullptr;
    log("menu: OptiScaler refused the WrongGPU box");
    return false;
  }
  log(g_card ? "menu: WrongGPU box added to OptiScaler's Custom tab"
             : "menu: WrongGPU tab added to OptiScaler's menu (this OptiScaler has no Custom tab cards)");
  return true;
}

void frame_done() {
  const WgHostMenuApi* api = g_api;
  if (!api || !WG_HOST_MENU_HAS(api, report_upscaler)) return;
  static std::string names[WG_RR_MODE_COUNT];  // "WrongGPU RR Accurate", one copy per mode
  static std::once_flag once;
  std::call_once(once, [] {
    for (int32_t m = 0; m < WG_RR_MODE_COUNT; ++m) names[m] = std::string("WrongGPU RR ") + mode_title(m);
  });
  if (g_preparing && g_preparing()) {  // first start with a cold driver cache: the frames pass through until the kernels are ready
    api->report_upscaler(g_plugin, "WrongGPU RR (preparing...)", 0.0f);
    return;
  }
  const int32_t mode = effective_mode(current_mode(), g_exact_allowed.load());
  api->report_upscaler(g_plugin, names[mode].c_str(), g_gpu_ms ? g_gpu_ms() : 0.0f);
}

}  // namespace panel
}  // namespace wgrr
