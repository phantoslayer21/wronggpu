// SPDX-License-Identifier: MIT
// RR settings owned by the proxy (spec 60.9): values, persistence, the contract table for WrongGPU.asi.
#pragma once
#include <stdint.h>

#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "../include/wg_rr_settings_contract.h"

namespace wgrr {

// Setting indices. Keys are stable across contract versions (60.9); the list is WrongGPU's choice from the
// opt-in variations pages 50-58 and 60-63 allow (QUESTIONS-60-63.md Q-S1).
enum SettingId : uint32_t {
  kSetWrongGpuRr = 0,          // "WrongGPU RR" on/off, default on, flag bit 0
  kSetSpecularMotionFromV,     // 62.4 / 52.7.1 SV = V, opt-in
  kSetResetAfterFailedEvaluate,// 61.6 / 53.10 deviation: force a reset on the call after a failed evaluate, opt-in
  kSetChangeFrameReset,        // 53.9 deviation: the render-size change frame itself is a reset frame, opt-in
  kSetBoundedStores,           // 54b.0.1 bounded game-facing stores, opt-in
  kSetHitDistanceFilter,     // 52.7.2 deviation: hit distance = median of its 1x4 run, opt-in
  kSettingCount
};

struct SettingDef {
  const char* key;
  const char* label;
  uint32_t kind;
  uint32_t flags;
  int32_t default_value;
  uint32_t max_value;
};
extern const SettingDef kSettingDefs[kSettingCount];

// Snapshot of all settings, taken once per evaluate / allocate (60.9).
struct SettingsSnapshot {
  int32_t v[kSettingCount];
  bool on(SettingId id) const { return v[id] != 0; }
};

class Settings {
 public:
  using LogFn = std::function<void(const std::string&)>;

  Settings();
  // Reads the file once (missing / unreadable / invalid values -> defaults; an unreadable file is logged once).
  // `path` is UTF-8; the file is opened (and saved) by its wide name, so any folder name works.
  void load(const std::string& path, const LogFn& log);
  int32_t get(uint32_t id) const;
  // 0 on success, 25 for a value outside 0..max (unchanged). Saves the file after a change of value.
  uint32_t set(uint32_t id, int32_t value);
  SettingsSnapshot snapshot() const;
  // clean/rr-f-live2: the text of a key that is not a 60.9 setting (rr_mode, rr_model); "" when absent. Such lines
  // are kept when the file is rewritten.
  std::string text(const char* key) const;
  const std::string& path() const { return path_; }
  // Saving can be disabled (tests that only exercise the values).
  void set_save_enabled(bool on) { save_enabled_ = on; }

 private:
  void save_locked();

  std::atomic<int32_t> values_[kSettingCount];
  std::string path_;
  std::vector<std::pair<std::string, std::string>> extra_;   // non-contract key=value lines, file order
  LogFn log_;
  std::mutex save_mutex_;
  bool save_enabled_ = true;
};

// The contract table (60.9). `status` and the setting accessors are bound to one process-global Settings
// instance and one status provider: the proxy's. bind_contract replaces the binding (tests rebind).
void bind_contract(Settings* settings, std::function<uint32_t()> status);
const WgRrSettingsContract* settings_contract();

}  // namespace wgrr
