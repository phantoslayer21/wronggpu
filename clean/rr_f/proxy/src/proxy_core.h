// SPDX-License-Identifier: MIT
// The proxy's game-interface logic (spec 60-63): answers feature 1001, records/copies what the RR path needs,
// forwards everything else to the original unchanged. Platform-free: the DLL glue (proxy_dll.cpp) supplies the
// original's function pointers, the runtime adapter, the clock and the log; tests supply fakes.
#pragma once
#include <stdint.h>

#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "../include/rr_proxy_abi.h"
#include "../include/wg_rr_adapter.h"
#include "lifecycle.h"
#include "settings.h"

namespace wgrr {

// Typed role signatures (60.3.1) with the argument and return types of annex A10.2 (references are pointers; every
// call returns the 32-bit signed Result, WgResultCode). set_vulkan_info and upgrade_interface are bound only by the
// probe build, which logs every API call with its result; the release build forwards them through stubs.
struct OriginalApi {
  WgResultCode (*init)(const WgPreferences* prefs, uint64_t api_version) = nullptr;
  WgResultCode (*shutdown)(void) = nullptr;
  WgResultCode (*is_supported)(uint32_t feature, const WgAdapterInfo* adapter) = nullptr;
  WgResultCode (*is_loaded)(uint32_t feature, bool* loaded) = nullptr;
  WgResultCode (*set_loaded)(uint32_t feature, bool loaded) = nullptr;
  WgResultCode (*requirements)(uint32_t feature, WgRequirements* out) = nullptr;
  WgResultCode (*version)(uint32_t feature, WgFeatureVersion* out) = nullptr;
  WgResultCode (*get_function)(uint32_t feature, const char* name, void** out) = nullptr;
  WgResultCode (*allocate)(void* cmd_list, uint32_t feature, const WgViewport* vp) = nullptr;
  WgResultCode (*free_resources)(uint32_t feature, const WgViewport* vp) = nullptr;
  WgResultCode (*set_device)(void* device) = nullptr;
  WgResultCode (*tag)(const WgViewport* vp, const WgTag* tags, uint32_t count, void* cmd_list) = nullptr;
  WgResultCode (*tag_for_frame)(const WgFrameToken* token, const WgViewport* vp, const WgTag* tags, uint32_t count,
                                void* cmd_list) = nullptr;  // 2.14.1 only
  WgResultCode (*constants)(const WgConstants* c, const WgFrameToken* token, const WgViewport* vp) = nullptr;
  WgResultCode (*new_frame_token)(WgFrameToken** out, const uint32_t* frame_index) = nullptr;
  WgResultCode (*evaluate)(uint32_t feature, const WgFrameToken* token, const WgHeader* const* inputs, uint32_t count,
                           void* cmd_list) = nullptr;
  WgResultCode (*native_interface)(void* wrapped, void** native) = nullptr;
  WgResultCode (*set_vulkan_info)(const void* info) = nullptr;           // probe build only
  WgResultCode (*upgrade_interface)(void** interface_ptr) = nullptr;     // probe build only
};

// Structure GUIDs the proxy must recognise (annex A9.6, data from the name-binding table).
struct BindingIds {
  WgGuid viewport{};        // 60.6.1, found in the evaluate input array
  WgGuid tag{};             // 60.6.3 local tags in the evaluate input array
  WgGuid tag_conversion{};  // 60.6.3 optional chained extension of a tag
  bool has_tag_conversion = true;  // false: the extension's GUID is not known (QUESTIONS Q-B6), never recognised
};

struct ProxyConfig {
  uint32_t api_minor = 14;               // 7 = the game ships 2.7.x, 8 = 2.8.0, 14 = 2.14.1 (60.1: built per version)
  BindingIds ids;
  uint32_t release_version[3] = {0, 0, 0};  // WrongGPU's published release version (60.8 <version>)
  uint32_t version_magic = 0;  // A9.1 low 16 bits of the <init> version value; only logged, never checked
  // The three RR feature-function names (A9.5; empty while pending); <get-function>(1001) matches them exactly.
  std::string fn_optimal, fn_state, fn_options;
  size_t max_frames = 8;                 // per-frame copies kept (>= the 6 frames in flight of 60.7)
  // clean/rr-re9-pt: which frame a per-frame copy (constants, frame-scoped tags) belongs to. Default: the token's
  // frame index (A10.1) read at the call, so every token object of one frame index shares one frame. RE Requiem
  // passes one token object to <constants> and another of the same index to <tag-for-frame> / <evaluate>; keyed by
  // the token object, its evaluates found constants set hundreds of frames earlier (frozen jitter and camera).
  // true: the token object (the behaviour before; control runs only: rr_frame_by_token=1 / WG_RR_FRAME_BY_TOKEN=1).
  bool frame_by_token = false;
};

// A10.1 frame index of a token: the first function of its table (0 for a null token or table).
uint32_t frame_index_of(const WgFrameToken* token);

// The key of a token's per-frame copies (ProxyConfig::frame_by_token); 0 for a null token.
inline uint64_t frame_key(const WgFrameToken* token, bool by_token) {
  if (!token) return 0;
  return by_token ? (uint64_t)(uintptr_t)token : ((1ull << 63) | frame_index_of(token));
}

using LogFn = std::function<void(const std::string&)>;
using ClockMsFn = std::function<uint64_t()>;

// clean/rr-re9-pt: the runtime records its passes on the game's own command list and binds its own descriptor heap,
// compute root signature and root arguments there. A game that does not rebind its compute root state after the call
// (RE Engine: RE Requiem) then runs its next dispatch against our layout. A host that tracks the game's command-list
// state (OptiScaler plugin interface 5) restores it: `begin` before the first runtime call that records on the game's
// list, `end` after the last one, on every path (success, failure, pass-through). The argument is the native list.
struct CmdListBracket {
  std::function<void(void* native_cmd_list)> begin;
  std::function<void(void* native_cmd_list)> end;
};

struct StoredTag {
  WgResource res{};
  void* resource = nullptr;  // native resource, or a runtime-owned copy for a lifetime-0 tag
  void* transient = nullptr; // the copy to release (null if none)
  uint32_t type = 0;
  uint32_t lifetime = 0;
  WgExtent extent{};
  bool has_conversion = false;
  WgTagConversion conversion{};
  // probe build only: the native resource's own size (D3D12 description), for a game that fills no size in the
  // resource description and tags no extent (RE Requiem: 0 x 0 and 0 x 0); 0 x 0 when unknown
  uint32_t desc_width = 0, desc_height = 0;
};

class RrProxy {
 public:
  RrProxy(const ProxyConfig& cfg, const OriginalApi& orig, const WgRrRuntimeV1* runtime, Settings* settings,
          LogFn log, ClockMsFn clock);
  ~RrProxy();

  // ---- Streamline API roles (60.3.1) ----
  WgResultCode init(const WgPreferences* prefs, uint64_t api_version);
  WgResultCode shutdown();
  WgResultCode is_supported(uint32_t feature, const WgAdapterInfo* adapter);
  WgResultCode is_loaded(uint32_t feature, bool* loaded);
  WgResultCode set_loaded(uint32_t feature, bool loaded);
  WgResultCode requirements(uint32_t feature, WgRequirements* out);
  WgResultCode version(uint32_t feature, WgFeatureVersion* out);
  WgResultCode get_function(uint32_t feature, const char* name, void** out);
  WgResultCode allocate(void* cmd_list, uint32_t feature, const WgViewport* vp);
  WgResultCode free_resources(uint32_t feature, const WgViewport* vp);
  WgResultCode set_device(void* device);
  WgResultCode tag(const WgViewport* vp, const WgTag* tags, uint32_t count, void* cmd_list);
  WgResultCode tag_for_frame(const WgFrameToken* token, const WgViewport* vp, const WgTag* tags, uint32_t count,
                         void* cmd_list);
  WgResultCode constants(const WgConstants* c, const WgFrameToken* token, const WgViewport* vp);
  WgResultCode new_frame_token(WgFrameToken** out, const uint32_t* frame_index);
  WgResultCode evaluate(uint32_t feature, const WgFrameToken* token, const WgHeader* const* inputs, uint32_t count,
                    void* cmd_list);
  // Hook mode only: the game's <init> ran before the hook was installed (HOOK-MODE.md). Marks the proxy initialised
  // with the safe defaults: preference flags unknown (0: bit 0 clear, the game does not restore its command-list
  // state), RR taken as requested (lifetime-0 inputs are copied), feature list and render API unknown, SDK version 0.
  void adopt_unseen_init();
  bool init_adopted() const { return init_adopted_; }

  // ---- RR feature functions (60.3.2), returned through <get-function>(1001) ----
  WgResultCode rr_optimal(const WgRrOptions* options, WgRrOptimal* out);
  WgResultCode rr_state(const WgViewport* vp, WgRrState* out);
  WgResultCode rr_options(const WgViewport* vp, const WgRrOptions* options);
  // Pointers handed out by <get-function>; must stay valid until unload (the DLL passes C trampolines).
  void set_feature_function_pointers(void* optimal, void* state, void* options);

  // clean/rr-re9-pt: bracket runtime work on the game's list for the host (CmdListBracket); empty = no host.
  void set_cmd_list_bracket(CmdListBracket bracket);
  // clean/rr-re9-pt: RR refused for this game (<is-supported>(1001) answers adapter-not-supported, so the game hides
  // its RR options): set when the game needs the host's restore and the host has none (hook_dll.cpp). `why` is logged
  // once, at the first refused query.
  void set_rr_refused(bool refused, const std::string& why);
  // clean/rr-re9-pt polish: the player's install switch "Use WrongGPU" for RR is off (rr_enabled = 0, read once at start
  // by the plugin from WrongGPU\wronggpu.ini / wronggpu-rr.ini). WrongGPU does not take RR: <is-supported>(1001) answers
  // adapter-not-supported so the game uses its own path, and no RR feature is created (allocate / evaluate answer 7).
  // Independent of set_rr_refused and of the 60.9 live setting; `why` is logged once.
  void set_rr_off(bool off, const std::string& why);
  bool rr_off() const;

  // ---- 60.9 status word ----
  uint32_t status_word() const;

  // ---- clean/rr-sl-diag: game-interface call diagnostics (log only; no answer changes) ----
  // Every distinct call of the support, lookup and loading roles, and the first RR options / optimal / constants / tag /
  // allocate / evaluate calls, are logged once each as "diag: ..." lines (same arguments and same answer: one line; at most
  // kDiagLines lines per proxy instance). `describe_adapter` turns an adapter LUID into words ("vendor 0x1002 device
  // 0x7550, AMD Radeon ..."; empty when unknown): the plugin supplies it, tests may.
  using AdapterDescribeFn = std::function<std::string(const void* luid, uint32_t size)>;
  void set_adapter_describer(AdapterDescribeFn fn);
  // "diag: summary (<when>): support queries N (RR: yes Y / no Z), RR lookups L, RR instances created C, RR evaluates E".
  // Lock-free (atomic counters): the plugin may call it at process exit while another thread holds the proxy's lock.
  std::string diag_summary(const char* when) const;
  // RR support queries the game made (yes + no answers; clean/060-rr2 Spider-Man 2 hint)
  uint64_t rr_support_queries() const { return d_rr_yes_.load() + d_rr_no_.load(); }
  static constexpr size_t kDiagLines = 300;

  // ---- test / diagnostic views ----
  bool has_instance() const;
  bool holds_viewport() const;
  uint32_t held_viewport() const;
  const Lifecycle* lifecycle() const;  // null without an instance
  const WgRrCreate* instance_create_desc() const;  // null without an instance
  uint32_t log_count(const std::string& needle) const;

 private:
  struct Instance;
  struct Viewport;
  struct FrameData;
  struct Local;

  class BracketScope;
  bool feature_precheck(WgResultCode* rc, bool needs_device) const;
  void log(const std::string& s) const;
  bool runtime_usable() const;
  WgResultCode claim_viewport(uint32_t id, Viewport** out);  // 38 for a second viewport
  void store_tags(std::map<std::pair<uint32_t, uint32_t>, StoredTag>& dst, const WgViewport* vp, const WgTag* tags,
                  uint32_t count, void* cmd_list);
  void release_tag(StoredTag& t);
  void release_frame(FrameData& f);
  FrameData* frame_for(const WgFrameToken* token, bool create);
  bool lookup(uint32_t type, uint32_t vp, const std::vector<StoredTag>& locals, const FrameData* frame,
              const StoredTag** out) const;
  WgResultCode create_or_update(Viewport& v, const std::vector<StoredTag>& locals, const FrameData* frame,
                            const WgConstants* consts, const SettingsSnapshot& snap, bool from_allocate,
                            bool* creating, WgRrCreate* desc);
  void destroy_instance(Viewport& v);
  void unwrap(void* wrapped, void** native) const;
  // the bodies of rr_options / allocate / evaluate (RR), called under mu_ by the public calls, which add the diagnostics
  WgResultCode rr_options_body(const WgViewport* vp, const WgRrOptions* options);
  WgResultCode allocate_body(void* cmd_list, const WgViewport* vp);
  WgResultCode evaluate_body(const WgFrameToken* token, const WgHeader* const* inputs, uint32_t count, void* cmd_list);

  // clean/rr-sl-diag
  void diag_once(const std::string& line);  // under mu_
  bool diag_first(int which);               // under mu_: true the first time for `which` (DiagFirst)
  std::string describe(const WgAdapterInfo* a) const;
  std::set<std::string> diag_seen_;
  uint32_t diag_firsts_ = 0;
  AdapterDescribeFn describe_adapter_;
  std::atomic<uint64_t> d_support_{0}, d_rr_yes_{0}, d_rr_no_{0}, d_rr_lookups_{0}, d_rr_instances_{0}, d_rr_evals_{0};

  ProxyConfig cfg_;
  OriginalApi orig_;
  const WgRrRuntimeV1* rt_;
  // clean/rr-f-live1: evaluate results reaching the runtime, the pass-through fail-safe, rate-limited log lines
  uint64_t evaluates_ = 0, passthroughs_ = 0;
  uint32_t last_eval_rc_ = 0, last_pass_rc_ = 0, eval_rc_lines_ = 0;
  Settings* settings_;
  CmdListBracket bracket_;
  // clean/rr-re9-pt diagnostics: frame-token / constants pairing log (rr_diag_tokens = "<first evaluate>,<count>" in
  // wronggpu-rr.ini, or WG_RR_DIAG_TOKENS); off by default, logs nothing and changes nothing else
  bool diag_read_ = false;
  uint64_t diag_from_ = 0, diag_count_ = 0, diag_evals_ = 0;
  bool diag_on() const { return diag_count_ && diag_evals_ + 1 >= diag_from_ && diag_evals_ < diag_from_ + diag_count_; }
  void diag_read();
  bool rr_refused_ = false, logged_rr_refused_ = false;
  std::string rr_refused_why_;
  bool rr_off_ = false, logged_rr_off_ = false, logged_rr_off_create_ = false;
  std::string rr_off_why_;
  LogFn log_;
  ClockMsFn clock_;

  mutable std::recursive_mutex mu_;
  bool initialised_ = false;
  bool init_adopted_ = false;
  bool device_set_ = false;
  void* native_device_ = nullptr;
  uint64_t api_version_ = 0;
  uint64_t pref_flags_ = 0;
  uint32_t render_api_ = 0;
  std::vector<uint32_t> features_;
  bool rr_requested_ = false;
  bool set_loaded_value_ = true;
  bool set_loaded_given_ = false;
  void* fn_ptrs_[3] = {nullptr, nullptr, nullptr};

  std::unique_ptr<Viewport> vp_;  // the one held viewport (61.2)
  std::map<std::pair<uint32_t, uint32_t>, StoredTag> viewport_tags_;  // (viewport, type)
  std::map<uint64_t, std::unique_ptr<FrameData>> frames_;            // keyed by frame_key (60.7)
  uint64_t frame_seq_ = 0;
  std::map<const void*, uint32_t> token_index_;
  std::map<uint32_t, WgConstants> latest_constants_;  // per viewport, for an <allocate> that creates

  // logged once per proxy instance
  bool logged_second_vp_ = false, logged_rr_off_eval_ = false, logged_rr_off_alloc_ = false;
  bool logged_no_runtime_ = false, logged_no_depth_ = false;  // create_or_update's refusals (it runs at every evaluate)
  mutable std::vector<std::string> log_lines_;

  std::atomic<uint64_t> last_complete_ms_{0};
  std::atomic<bool> last_eval_completed_{false};
};

// The RR plugin's slot watch (a background thread that reads the patched game-interface slots again, and one check at
// the host's first direct-path create) is off unless WG_RR_SLOT_CHECKS asks for it (decided 2026-11-04, release
// default): unset, empty or "off" -> "" (off); "on" -> the default times; anything else -> the comma-separated seconds
// as given. The once-per-call diagnostic lines and the exit summary do not depend on it.
extern const char* const kSlotChecksDefault;
std::string slot_check_schedule(const char* env);

}  // namespace wgrr
