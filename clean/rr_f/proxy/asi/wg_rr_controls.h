// SPDX-License-Identifier: MIT
// WrongGPU.asi side of the RR controls (spec 60.9): find the proxy, read its settings contract, decide what the
// overlay shows, and change settings only through the contract's set functions.
//
// Meant to be included by clean/fsr/asi/wronggpu_asi.cpp (a separate front-end branch). The overlay
// drawing itself needs OptiScaler's overlay extension point, which is spec open item 23: ControlsSink below is
// the seam; until item 23 is answered no sink is attached and nothing is drawn (60.9 "Absence").
// It records no GPU work, hooks no API, makes no game-interface call, and never loads the proxy itself.
#pragma once
#include <stdint.h>
#include <string.h>

#include <string>
#include <vector>

#include "../include/wg_rr_settings_contract.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#endif

namespace wgrr_asi {

static const char kNotActiveLine[] = "WrongGPU RR not active";
static const char kAtCreationNote[] = "applies when the game next creates RR";

struct ControlItem {
  uint32_t index = 0;          // entry index in the contract
  std::string key, label;
  uint32_t kind = 0;           // 0 on/off, 1 choice 0..max
  uint32_t max_value = 1;
  int32_t value = 0;
  int32_t default_value = 0;
  bool at_creation = false;    // show kAtCreationNote next to it
  bool opt_in = false;
};

struct ControlsView {
  enum Mode { kNothing = 0, kNotActive = 1, kControls = 2 } mode = kNothing;
  std::string line;            // kNotActive: kNotActiveLine
  std::vector<ControlItem> items;
};

// What the overlay shows for a contract (null = no loaded module exports the contract).
inline ControlsView build_view(const WgRrSettingsContract* c) {
  ControlsView v;
  if (!c) {
    v.mode = ControlsView::kNotActive;
    v.line = kNotActiveLine;
    return v;
  }
  if (c->version != WG_RR_CONTRACT_VERSION || c->size < WG_RR_CONTRACT_V1_SIZE) return v;  // unknown: nothing
  uint32_t st = c->status ? c->status() : 0u;
  if ((st & (WG_RR_STATUS_RUNTIME_READY | WG_RR_STATUS_MODEL_LOADED)) !=
      (WG_RR_STATUS_RUNTIME_READY | WG_RR_STATUS_MODEL_LOADED)) {
    v.mode = ControlsView::kNotActive;
    v.line = kNotActiveLine;
    return v;
  }
  v.mode = ControlsView::kControls;
  for (uint32_t i = 0; c->entries && i < c->count; ++i) {
    const WgRrSettingEntry& e = c->entries[i];
    if (!e.key || !e.label || !e.get || !e.set) continue;
    ControlItem it;
    it.index = i;
    it.key = e.key;
    it.label = e.label;
    it.kind = e.kind;
    it.max_value = e.kind == WG_RR_SETTING_KIND_ONOFF ? 1u : e.max_value;
    it.value = e.get();
    it.default_value = e.default_value;
    it.at_creation = (e.flags & WG_RR_SETTING_FLAG_AT_CREATION) != 0;
    it.opt_in = (e.flags & WG_RR_SETTING_FLAG_OPT_IN) != 0;
    v.items.push_back(it);
  }
  return v;
}

// Apply a user change: only through the set function (60.9). Returns the set function's result (0 / 25),
// or 25 when the index is not an entry of the contract.
inline uint32_t apply_change(const WgRrSettingsContract* c, uint32_t index, int32_t value) {
  if (!c || c->version != WG_RR_CONTRACT_VERSION || c->size < WG_RR_CONTRACT_V1_SIZE || !c->entries ||
      index >= c->count || !c->entries[index].set)
    return 25u;
  return c->entries[index].set(value);
}

// Drawing seam (open item 23). The ASI implements it on top of OptiScaler's overlay once the extension point is
// known; draw_view calls nothing when the view is kNothing.
struct ControlsSink {
  virtual ~ControlsSink() {}
  virtual void text(const char* line) = 0;
  // Returns true and sets *value when the user changed the control this frame.
  virtual bool toggle(const char* label, int32_t current, int32_t* value) = 0;
  virtual bool choice(const char* label, int32_t current, uint32_t max_value, int32_t* value) = 0;
  virtual void note(const char* text) = 0;  // small text after the preceding control
};

inline void draw_view(const WgRrSettingsContract* c, const ControlsView& v, ControlsSink& sink) {
  if (v.mode == ControlsView::kNothing) return;
  if (v.mode == ControlsView::kNotActive) {
    sink.text(v.line.c_str());
    return;
  }
  for (const ControlItem& it : v.items) {
    int32_t nv = it.value;
    bool changed = it.kind == WG_RR_SETTING_KIND_ONOFF ? sink.toggle(it.label.c_str(), it.value, &nv)
                                                       : sink.choice(it.label.c_str(), it.value, it.max_value, &nv);
    if (it.at_creation) sink.note(kAtCreationNote);
    if (changed && nv != it.value) apply_change(c, it.index, nv);
  }
}

#ifdef _WIN32
// Finds the proxy as a module already loaded in the process that exports the contract (60.9); holds a reference
// to it while attached; never loads it. Not thread-safe by itself: attach / detach from one thread.
class ProxyLink {
 public:
  ~ProxyLink() { detach(); }
  const WgRrSettingsContract* attach() {
    if (contract_) return contract_;
    HMODULE mods[1024];
    DWORD need = 0;
    if (!EnumProcessModules(GetCurrentProcess(), mods, sizeof mods, &need)) return nullptr;
    size_t n = need / sizeof(HMODULE);
    if (n > sizeof mods / sizeof mods[0]) n = sizeof mods / sizeof mods[0];
    for (size_t i = 0; i < n; ++i) {
      FARPROC f = GetProcAddress(mods[i], WG_RR_CONTRACT_EXPORT);
      if (!f) continue;
      HMODULE held = nullptr;  // add a reference to the module that owns the export
      if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, reinterpret_cast<LPCWSTR>(f), &held))
        continue;
      const WgRrSettingsContract* c = reinterpret_cast<WgRrGetSettingsContractFn>(f)();
      if (!c) {
        FreeLibrary(held);
        continue;
      }
      module_ = held;
      contract_ = c;
      return c;
    }
    return nullptr;
  }
  void detach() {
    contract_ = nullptr;
    if (module_) FreeLibrary(module_);
    module_ = nullptr;
  }
  const WgRrSettingsContract* contract() const { return contract_; }
  HMODULE module() const { return module_; }

 private:
  HMODULE module_ = nullptr;
  const WgRrSettingsContract* contract_ = nullptr;
};
#endif

}  // namespace wgrr_asi
