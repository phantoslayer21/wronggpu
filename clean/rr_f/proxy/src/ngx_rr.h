// SPDX-License-Identifier: MIT
// WrongGPU RR for games that call the NGX interface directly (spec page 66, with page 64 for the direct path): the
// provider the RR plugin registers with the host's plugin interface version 4 (wg_host_menu_abi.h, WgHostNgxRr). The
// host hands the D3D12 RR feature's create / evaluate / release to it with read access to the game's parameter
// block; the provider maps the create keys onto WgRrCreate (66.2) and the evaluate keys onto WgRrFrame (66.3), runs
// the same runtime the game-interface route runs (rr_mode, model, settings file: one runtime per process), and keeps
// the host bookkeeping of page 53 per feature (Lifecycle, the same module the game-interface route uses).
//
// Platform-free: the plugin (hook_dll.cpp) supplies the runtime, the settings, the log, the clock and the host's
// command-list bracket; tests supply fakes and a mock parameter block.
//
// Rules taken from page 66: no tags, no frame tokens, every resource valid for the call only, all GPU work on the
// game's command list inside the call; motion-vector scale taken as given (0 counts as 1), jitter unchanged; the
// previous camera pair is the provider's (the current pair on a reset frame); layer motion vectors are not consumed;
// destroy is deferred past the frames in flight (the runtime's destroy_instance). A refused create or a failed
// evaluate returns false, which the host turns into an NGX failure code: the game falls back, nothing crashes.
#pragma once
#include <stdint.h>

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "../include/wg_host_menu_abi.h"
#include "../include/wg_rr_adapter.h"
#include "lifecycle.h"
#include "proxy_core.h"  // LogFn, ClockMsFn, CmdListBracket
#include "settings.h"

namespace wgrr {

// WgRrCreate.api_minor / WgRrFrame.api_minor of the direct route (66.2: a value of WrongGPU's choice that no
// game-interface variant uses, so the runtime can tell the route; it never selects frame-scoped tagging).
constexpr uint32_t kNgxApiMinor = 1000;

// Declared states handed to the runtime (66.4: at the evaluate call every input is in a shader-resource state and the
// output in the unordered-access state). D3D12 state bits: PIXEL | NON_PIXEL shader resource, UNORDERED_ACCESS.
constexpr uint32_t kNgxInputState = 0xC0u;
constexpr uint32_t kNgxOutputState = 0x8u;

struct NgxRrDeps {
  std::function<const WgRrRuntimeV1*()> runtime;  // loads the runtime on first use; null when it is not available
  std::function<Settings*()> settings;            // the plugin's RR settings (wronggpu-rr.ini); may return null
  LogFn log;
  ClockMsFn clock;
  // The device under a command list, for a host that hands no device at create (null when unknown).
  std::function<void*(void* command_list)> device_of;
  // True while the game-interface route holds an RR instance: one route per game, never a double evaluate.
  std::function<bool()> other_route_busy;
  // After every evaluate that reached the runtime successfully (the menu box / overlay).
  std::function<void()> frame_done;
};

class NgxRrProvider {
 public:
  explicit NgxRrProvider(NgxRrDeps deps);
  ~NgxRrProvider();
  NgxRrProvider(const NgxRrProvider&) = delete;
  NgxRrProvider& operator=(const NgxRrProvider&) = delete;

  // The provider's three calls (WgHostNgxRr).
  bool create(void* device, void* command_list, const WgHostNgxParams* params, void** context);
  bool evaluate(void* context, void* command_list, const WgHostNgxParams* params);
  void release(void* context);
  // Releases every live feature (process end, tests).
  void release_all();

  // The table registered with the host: C trampolines into this provider (SEH-guarded in the plugin build).
  WgHostNgxRr table();

  // The host's command-list bracket (plugin interface 5). The host already brackets each evaluate itself; ours nests
  // inside it (only the outermost pair restores) and covers a host that brackets nothing.
  void set_cmd_list_bracket(CmdListBracket bracket);

  // Fault guard (hook_dll.cpp): after a fault caught on this thread, ends the provider's command-list bracket if it is
  // still open (a structured exception does not always run the destructors of the frames it leaves), so the host's
  // nesting depth and root-state tracking stay balanced. No effect when no bracket of this provider is open.
  static void close_open_bracket_after_fault();

  // ---- views (tests, the plugin's route guard)
  size_t live() const;  // lock-free: the plugin's route guard may ask from any thread, also after a fault
  uint64_t evaluates() const;
  const WgRrCreate* create_desc(void* context) const;  // null for an unknown context
  const Lifecycle* lifecycle(void* context) const;
  uint32_t log_count(const std::string& needle) const;

 private:
  struct Instance;
  class Scope;
  void log(const std::string& s) const;
  bool usable(const WgRrRuntimeV1* rt) const;
  Instance* find(void* context) const;

  NgxRrDeps deps_;
  CmdListBracket bracket_;
  mutable std::recursive_mutex mu_;
  std::set<Instance*> live_;
  std::atomic<size_t> live_count_{0};  // live_.size(), readable without the lock
  uint32_t next_id_ = 1;
  void* device_set_ = nullptr;  // the device handed to the runtime's set_device last
  uint64_t evaluates_ = 0, passthroughs_ = 0;
  uint32_t last_eval_rc_ = 0, last_pass_rc_ = 0, eval_rc_lines_ = 0, refusal_lines_ = 0;
  bool logged_busy_ = false, logged_setting_off_ = false;
  mutable std::vector<std::string> log_lines_;
};

}  // namespace wgrr
