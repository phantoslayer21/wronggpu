// SPDX-License-Identifier: MIT
// Spec 60-63 game-interface logic of the RR proxy. Section numbers refer to spec/60-63-rr-f-interface.md.
#include "proxy_core.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>

#include "answers.h"
#include "formats.h"
#include "optimal.h"

namespace wgrr {

namespace {

constexpr uint32_t kRR = WG_FEAT_RAY_RECONSTRUCTION;

bool guid_eq(const WgGuid& a, const WgGuid& b) { return memcmp(a.b, b.b, 16) == 0; }

uint64_t version_of(const WgHeader& h) { return struct_version(h); }

// Buffer types preset F reads (62.2); lifetime-0 tags of these are copied during the tagging call (62.1).
bool f_reads(uint32_t t) {
  switch (t) {
    case WG_BT_COLOUR_IN: case WG_BT_DEPTH: case WG_BT_LINEAR_DEPTH: case WG_BT_MOTION:
    case WG_BT_DIFFUSE_ALBEDO: case WG_BT_SPECULAR_ALBEDO: case WG_BT_NORMALS: case WG_BT_ROUGHNESS:
    case WG_BT_NORMAL_ROUGHNESS: case WG_BT_SPECULAR_MOTION: case WG_BT_SPECULAR_HIT_DISTANCE:
    case WG_BT_DISOCCLUSION: case WG_BT_SSS_GUIDE: case WG_BT_COLOUR_BEFORE_FOG: case WG_BT_DOF_GUIDE:
    case WG_BT_COLOUR_BEFORE_TRANSPARENCY: case WG_BT_TRANSPARENCY_LAYER: case WG_BT_COLOUR_BEFORE_PARTICLES:
    case WG_BT_TRANSPARENCY_OPACITY: case WG_BT_INPUT_ALPHA:
      return true;
    default:
      return false;
  }
}

// A game that fills no size in the resource description (game A: 0 x 0): the size of the native resource, from the
// runtime (adapter v1 live addition describe_resource_size), so input_size and the runtime see the real size.
void fill_size(const WgRrRuntimeV1* rt, StoredTag& s) {
  if ((s.res.width && s.res.height) || !s.res.native || !rt || rt->size < WG_RR_RUNTIME_V1_LIVE_SIZE ||
      !rt->describe_resource_size)
    return;
  uint32_t w = 0, h = 0;
  if (rt->describe_resource_size(rt->ctx, s.res.native, &w, &h) == WG_OK && w && h) {
    s.res.width = w;
    s.res.height = h;
  }
}

// Size an input covers: the tag's extent, or the full resource when the extent is not set (60.6.2, 61.3).
void input_size(const StoredTag& t, uint32_t* w, uint32_t* h) {
  if (t.extent.width == 0 || t.extent.height == 0) {
    *w = t.res.width;
    *h = t.res.height;
  } else {
    *w = t.extent.width;
    *h = t.extent.height;
  }
}

WgRrOptions default_options() {
  WgRrOptions o;
  memset(&o, 0, sizeof o);
  o.output_width = WG_SIZE_NOT_SET;
  o.output_height = WG_SIZE_NOT_SET;
  o.pre_exposure = 1.0f;
  o.exposure_scale = 1.0f;
  o.colour_is_hdr = WG_TRI_TRUE;
  return o;
}

std::string u32s(uint32_t v) { return std::to_string(v); }
std::string hexs(uint32_t v) {
  char b[16];
  snprintf(b, sizeof b, "%x", v);
  return b;
}
std::string hexp(const void* p) {
  char b[24];
  snprintf(b, sizeof b, "%p", p);
  return b;
}

}  // namespace

// A10.1: the frame index is read only through the first function of the token's table (the token is the original's
// object; the game's optional index argument of <new-frame-token> is not read by the proxy).
uint32_t frame_index_of(const WgFrameToken* token) {
  if (!token || !token->vtable || !token->vtable[0]) return 0;
  return token->vtable[0](token);
}

struct RrProxy::Instance {
  void* handle = nullptr;
  WgRrCreate desc{};
  WgRrOptions opts{};      // options the instance was created with (recreate triggers, 61.3)
  uint64_t opts_version = 1;
  Lifecycle lc;
  Instance(uint32_t rw, uint32_t rh) : lc(rw, rh) {}
};

struct RrProxy::Viewport {
  uint32_t id = WG_VIEWPORT_INVALID;
  bool has_options = false;
  WgRrOptions opts = default_options();
  uint64_t opts_version = 1;
  std::unique_ptr<Instance> inst;
  // logged once per feature instance (reset when an instance is created)
  bool logged_range = false, logged_camera_motion = false, logged_mv_res = false, logged_format = false,
       logged_camera_fallback = false, logged_zero = false;
};

struct RrProxy::FrameData {
  uint64_t seq = 0;
  std::map<std::pair<uint32_t, uint32_t>, StoredTag> tags;  // (viewport, type), frame-scoped (2.14.1)
  std::map<uint32_t, WgConstants> consts;                   // per viewport
  std::map<uint32_t, uint32_t> consts_index;                // diagnostics: the token's frame index when set
};

RrProxy::RrProxy(const ProxyConfig& cfg, const OriginalApi& orig, const WgRrRuntimeV1* runtime, Settings* settings,
                 LogFn log, ClockMsFn clock)
    : cfg_(cfg), orig_(orig), rt_(runtime), settings_(settings), log_(std::move(log)), clock_(std::move(clock)) {}

// clean/rr-re9-pt: one begin/end pair around runtime work on the game's list; end runs when the scope closes (every
// return path) and only if begin ran.
class RrProxy::BracketScope {
 public:
  explicit BracketScope(const CmdListBracket& b) : b_(b) {}
  ~BracketScope() {
    if (open_ && b_.end) b_.end(cmd_);
  }
  void open(void* native_cmd) {
    if (open_ || !native_cmd || !b_.begin || !b_.end) return;
    b_.begin(native_cmd);
    cmd_ = native_cmd;
    open_ = true;
  }
  BracketScope(const BracketScope&) = delete;
  BracketScope& operator=(const BracketScope&) = delete;

 private:
  const CmdListBracket& b_;
  void* cmd_ = nullptr;
  bool open_ = false;
};

void RrProxy::diag_read() {
  if (diag_read_) return;
  diag_read_ = true;
  std::string v;
#pragma warning(suppress : 4996)
  if (const char* ev = getenv("WG_RR_DIAG_TOKENS")) v = ev;
  if (v.empty() && settings_) v = settings_->text("rr_diag_tokens");
  if (v.empty()) return;
  unsigned long long a = 0, b = 0;
  if (sscanf(v.c_str(), "%llu,%llu", &a, &b) == 2 && b) {
    diag_from_ = a;
    diag_count_ = b > 2000 ? 2000 : b;
    log("diag: frame-token pairing log for evaluates " + std::to_string(diag_from_) + ".." +
        std::to_string(diag_from_ + diag_count_ - 1));
  }
}

void RrProxy::set_cmd_list_bracket(CmdListBracket bracket) {
  std::lock_guard<std::recursive_mutex> lock(mu_);
  bracket_ = std::move(bracket);
}

void RrProxy::set_rr_refused(bool refused, const std::string& why) {
  std::lock_guard<std::recursive_mutex> lock(mu_);
  rr_refused_ = refused;
  rr_refused_why_ = why;
  logged_rr_refused_ = false;
}

void RrProxy::set_rr_off(bool off, const std::string& why) {
  std::lock_guard<std::recursive_mutex> lock(mu_);
  rr_off_ = off;
  rr_off_why_ = why;
  logged_rr_off_ = false;
  logged_rr_off_create_ = false;
}

bool RrProxy::rr_off() const {
  std::lock_guard<std::recursive_mutex> lock(mu_);
  return rr_off_;
}

RrProxy::~RrProxy() {
  std::lock_guard<std::recursive_mutex> lock(mu_);
  if (vp_) destroy_instance(*vp_);
  for (auto& kv : viewport_tags_) release_tag(kv.second);
  for (auto& kv : frames_) release_frame(*kv.second);
}

void RrProxy::log(const std::string& s) const {
  if (log_lines_.size() < 4096) log_lines_.push_back(s);
  if (log_) log_(s);
}

uint32_t RrProxy::log_count(const std::string& needle) const {
  std::lock_guard<std::recursive_mutex> lock(mu_);
  uint32_t n = 0;
  for (const auto& l : log_lines_) n += l.find(needle) != std::string::npos;
  return n;
}

// ------------------------------------------------------------------------------------------------------------
// clean/rr-sl-diag: game-interface call diagnostics (log lines only; no answer changes)

namespace {
enum DiagFirst { kFirstConstants = 0, kFirstTag, kFirstTagForFrame, kFirstAllocate, kFirstEvaluate };
std::string rcs(WgResultCode rc) { return std::to_string((uint32_t)rc) + (rc == WG_OK ? " (yes)" : ""); }
}  // namespace

void RrProxy::set_adapter_describer(AdapterDescribeFn fn) {
  std::lock_guard<std::recursive_mutex> lock(mu_);
  describe_adapter_ = std::move(fn);
}

void RrProxy::diag_once(const std::string& line) {
  if (diag_seen_.size() >= kDiagLines || !diag_seen_.insert(line).second) return;
  log("diag: " + line + (diag_seen_.size() == kDiagLines ? " (diagnostic line limit reached: later ones are not logged)" : ""));
}

bool RrProxy::diag_first(int which) {
  const uint32_t bit = 1u << which;
  if (diag_firsts_ & bit) return false;
  diag_firsts_ |= bit;
  return true;
}

std::string RrProxy::describe(const WgAdapterInfo* a) const {
  if (!a) return "no adapter info (any adapter)";
  char b[96];
  std::string s;
  if (a->luid && a->luid_size == 8) {
    uint32_t lo = 0, hi = 0;
    memcpy(&lo, a->luid, 4);
    memcpy(&hi, static_cast<const uint8_t*>(a->luid) + 4, 4);
    snprintf(b, sizeof b, "LUID %08x:%08x", hi, lo);
    s = b;
  } else if (a->luid && a->luid_size) {
    s = "LUID of " + u32s(a->luid_size) + " bytes";
  } else {
    s = "no LUID";
  }
  snprintf(b, sizeof b, " (adapter info v%llu%s)", (unsigned long long)version_of(a->h),
           a->vk_physical_device ? ", Vulkan physical device set" : "");
  s += b;
  if (describe_adapter_ && a->luid && a->luid_size) {
    const std::string d = describe_adapter_(a->luid, a->luid_size);
    if (!d.empty()) s += ", " + d;
  }
  return s;
}

std::string RrProxy::diag_summary(const char* when) const {
  char b[320];
  snprintf(b, sizeof b,
           "diag: summary (%s): support queries %llu (RR: yes %llu / no %llu), RR lookups %llu, RR instances created %llu, "
           "RR evaluates %llu",
           when ? when : "now", (unsigned long long)d_support_.load(), (unsigned long long)d_rr_yes_.load(),
           (unsigned long long)d_rr_no_.load(), (unsigned long long)d_rr_lookups_.load(),
           (unsigned long long)d_rr_instances_.load(), (unsigned long long)d_rr_evals_.load());
  return b;
}

bool RrProxy::runtime_usable() const {
  if (!rt_ || !rt_->status) return false;
  uint32_t s = rt_->status(rt_->ctx);
  return (s & (WG_RR_RT_READY | WG_RR_RT_MODEL_LOADED)) == (WG_RR_RT_READY | WG_RR_RT_MODEL_LOADED);
}

bool RrProxy::feature_precheck(WgResultCode* rc, bool needs_device) const {
  if (!initialised_) { *rc = WG_E_INIT_NOT_CALLED; return false; }       // 60.1: 23 before <init>
  if (needs_device && !device_set_) { *rc = WG_E_DEVICE_NOT_CREATED; return false; }  // 5 before <set-device>
  return true;
}

void RrProxy::unwrap(void* wrapped, void** native) const {
  *native = wrapped;
  if (!wrapped || !orig_.native_interface) return;
  void* n = nullptr;
  if (orig_.native_interface(wrapped, &n) == WG_OK && n) *native = n;
}

WgResultCode RrProxy::claim_viewport(uint32_t id, Viewport** out) {
  if (vp_ && vp_->id != id) {  // 61.2: one viewport per proxy instance
    if (!logged_second_vp_) {
      logged_second_vp_ = true;
      log("rr: second viewport " + u32s(id) + " refused while viewport " + u32s(vp_->id) + " is held (38)");
    }
    return WG_E_INVALID_STATE;
  }
  if (!vp_) {
    vp_.reset(new Viewport());
    vp_->id = id;
  }
  *out = vp_.get();
  return WG_OK;
}

void RrProxy::destroy_instance(Viewport& v) {
  if (!v.inst) return;
  if (rt_ && rt_->destroy_instance && v.inst->handle) rt_->destroy_instance(rt_->ctx, v.inst->handle);
  v.inst.reset();
}

void RrProxy::release_tag(StoredTag& t) {
  if (t.transient && rt_ && rt_->release_transient) rt_->release_transient(rt_->ctx, t.transient);
  t.transient = nullptr;
}

void RrProxy::release_frame(FrameData& f) {
  for (auto& kv : f.tags) release_tag(kv.second);
  f.tags.clear();
}

// ------------------------------------------------------------------------------------------------------------
// 60.3.1 roles

WgResultCode RrProxy::init(const WgPreferences* prefs, uint64_t api_version) {
  // Record features / flags / render API, then forward unchanged (feature list included, 60.1).
  {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    api_version_ = api_version;
    features_.clear();
    pref_flags_ = 0;
    render_api_ = 0;
    if (prefs) {
      pref_flags_ = prefs->flags;
      render_api_ = prefs->render_api;
      if (prefs->features)
        for (uint32_t i = 0; i < prefs->feature_count; ++i) features_.push_back(prefs->features[i]);
    }
    rr_requested_ = std::find(features_.begin(), features_.end(), kRR) != features_.end();
  }
  // A9.1: every version value is forwarded unchanged; an older SDK version than the interposer's (a game built
  // against 2.11.1 shipping a 2.14.1 interposer) or another magic is never rejected, only logged.
  WgResultCode rc = orig_.init ? orig_.init(prefs, api_version) : WG_E_MISSING_OR_INVALID_API;
  std::lock_guard<std::recursive_mutex> lock(mu_);
  if (rc == WG_OK) initialised_ = true;
  log("proxy: <init> version " + u32s(WG_API_VERSION_MAJOR(api_version)) + "." +
      u32s(WG_API_VERSION_MINOR(api_version)) + "." + u32s(WG_API_VERSION_PATCH(api_version)) +
      (WG_API_VERSION_MAGIC(api_version) == cfg_.version_magic ? "" : " (unexpected magic)") +
      " forwarded unchanged, result " + u32s(rc));
  return rc;
}

void RrProxy::adopt_unseen_init() {
  std::lock_guard<std::recursive_mutex> lock(mu_);
  if (initialised_) return;
  initialised_ = true;
  init_adopted_ = true;
  api_version_ = 0;
  features_.clear();
  pref_flags_ = 0;       // unknown: bit 0 clear (the game does not restore its command-list state)
  render_api_ = 0;
  rr_requested_ = true;  // unknown: lifetime-0 inputs are copied
  log("proxy: <init> unseen (ran before the hook); adopted with flags unknown (bit 0 taken as 0), RR taken as "
      "requested, SDK version unknown");
}

WgResultCode RrProxy::shutdown() {
  WgResultCode rc = orig_.shutdown ? orig_.shutdown() : WG_E_MISSING_OR_INVALID_API;  // forward, then release
  std::lock_guard<std::recursive_mutex> lock(mu_);
  log(diag_summary("shutdown"));  // clean/rr-sl-diag
  if (vp_) destroy_instance(*vp_);
  vp_.reset();
  for (auto& kv : viewport_tags_) release_tag(kv.second);
  viewport_tags_.clear();
  for (auto& kv : frames_) release_frame(*kv.second);
  frames_.clear();
  token_index_.clear();
  latest_constants_.clear();
  initialised_ = false;
  device_set_ = false;
  native_device_ = nullptr;
  last_eval_completed_ = false;
  return rc;
}

WgResultCode RrProxy::is_supported(uint32_t feature, const WgAdapterInfo* adapter) {
  if (feature != kRR) {
    const WgResultCode frc = orig_.is_supported ? orig_.is_supported(feature, adapter) : WG_E_MISSING_OR_INVALID_API;
    ++d_support_;  // clean/rr-sl-diag: the other features' queries, one line per feature and answer
    std::lock_guard<std::recursive_mutex> lock(mu_);
    diag_once("support query feature " + u32s(feature) + " (not RR, forwarded to the game interface): answer " + rcs(frc));
    return frc;
  }
  std::lock_guard<std::recursive_mutex> lock(mu_);
  // the decision exactly as before; `why` names the reason of a "no" for the diagnostic line (clean/rr-sl-diag)
  std::string why;
  auto decide = [&]() -> WgResultCode {
    WgResultCode rc;
    if (!feature_precheck(&rc, false)) {
      why = "<init> not seen yet";
      return rc;
    }
    if (rr_off_) {  // the player's switch: WrongGPU does not take RR, the game uses its own path (set_rr_off)
      if (!logged_rr_off_) {
        logged_rr_off_ = true;
        log("rr: reported unsupported: " + rr_off_why_);
      }
      why = "the player's switch is off: " + rr_off_why_;
      return WG_E_ADAPTER_NOT_SUPPORTED;
    }
    if (!runtime_usable()) {  // 60.8, 61.6
      if (!rt_ || !rt_->status) {
        why = "WrongGPU's RR runtime is not loaded";
      } else {
        const uint32_t s = rt_->status(rt_->ctx);
        why = "the RR runtime is not ready (status 0x" + hexs(s) + ": supported GPU found " +
              ((s & WG_RR_RT_READY) ? "yes" : "no") + ", model loaded " + ((s & WG_RR_RT_MODEL_LOADED) ? "yes" : "no") + ")";
      }
      return WG_E_ADAPTER_NOT_SUPPORTED;
    }
    if (rr_refused_) {  // clean/rr-re9-pt: the game would hang without the host's restore (set_rr_refused)
      if (!logged_rr_refused_) {
        logged_rr_refused_ = true;
        log("rr: reported unsupported: " + rr_refused_why_);
      }
      why = rr_refused_why_;
      return WG_E_ADAPTER_NOT_SUPPORTED;
    }
    if (adapter) {
      if (adapter->vk_physical_device) {  // Vulkan RR is not in scope
        why = "a Vulkan adapter (only D3D12 is supported)";
        return WG_E_ADAPTER_NOT_SUPPORTED;
      }
      if (adapter->luid && adapter->luid_size) {
        if (!rt_->adapter_supported || !rt_->adapter_supported(rt_->ctx, adapter->luid, adapter->luid_size)) {
          why = !rt_->adapter_supported ? "the runtime has no adapter check"
                                        : "the runtime does not support the adapter of this LUID (not an RDNA 4 GPU, or a LUID "
                                          "size other than 8)";
          return WG_E_ADAPTER_NOT_SUPPORTED;
        }
      }
    }
    return WG_OK;
  };
  const WgResultCode rc = decide();
  ++d_support_;
  ++(rc == WG_OK ? d_rr_yes_ : d_rr_no_);
  diag_once("support query RR (" + u32s(kRR) + "): " + describe(adapter) + ": answer " + rcs(rc) +
            (rc == WG_OK ? "" : " (no: " + why + ")"));
  return rc;
}

WgResultCode RrProxy::is_loaded(uint32_t feature, bool* loaded) {
  if (feature != kRR) return orig_.is_loaded ? orig_.is_loaded(feature, loaded) : WG_E_MISSING_OR_INVALID_API;
  std::lock_guard<std::recursive_mutex> lock(mu_);
  WgResultCode rc;
  if (!feature_precheck(&rc, false)) return rc;
  if (!loaded) return WG_E_INVALID_PARAMETER;
  bool model = rt_ && rt_->status && (rt_->status(rt_->ctx) & WG_RR_RT_MODEL_LOADED);
  *loaded = model ? (set_loaded_given_ ? set_loaded_value_ : true) : false;  // 60.8
  diag_once(std::string("RR is-loaded: answer 0, loaded ") + (*loaded ? "true" : "false"));  // clean/rr-sl-diag
  return WG_OK;
}

WgResultCode RrProxy::set_loaded(uint32_t feature, bool loaded) {
  if (feature != kRR) return orig_.set_loaded ? orig_.set_loaded(feature, loaded) : WG_E_MISSING_OR_INVALID_API;
  std::lock_guard<std::recursive_mutex> lock(mu_);
  WgResultCode rc;
  const std::string what = std::string("RR set-loaded(") + (loaded ? "true" : "false") + "): answer ";
  if (!feature_precheck(&rc, false)) {
    diag_once(what + rcs(rc) + " (<init> not seen yet)");
    return rc;
  }
  set_loaded_given_ = true;  // recorded; never makes RR usable by itself (60.8)
  set_loaded_value_ = loaded;
  diag_once(what + "0");
  return WG_OK;
}

WgResultCode RrProxy::requirements(uint32_t feature, WgRequirements* out) {
  if (feature != kRR) return orig_.requirements ? orig_.requirements(feature, out) : WG_E_MISSING_OR_INVALID_API;
  std::lock_guard<std::recursive_mutex> lock(mu_);
  WgResultCode rc;
  if (!feature_precheck(&rc, false)) return rc;
  rc = answer_requirements(out);  // 60.8
  diag_once("RR requirements query: answer " + rcs(rc));  // clean/rr-sl-diag
  return rc;
}

WgResultCode RrProxy::version(uint32_t feature, WgFeatureVersion* out) {
  if (feature != kRR) return orig_.version ? orig_.version(feature, out) : WG_E_MISSING_OR_INVALID_API;
  std::lock_guard<std::recursive_mutex> lock(mu_);
  WgResultCode rc;
  if (!feature_precheck(&rc, false)) return rc;
  rc = answer_version(out, api_version_, cfg_.release_version);  // 60.8
  diag_once("RR version query: answer " + rcs(rc));  // clean/rr-sl-diag
  return rc;
}

void RrProxy::set_feature_function_pointers(void* optimal, void* state, void* options) {
  fn_ptrs_[0] = optimal;
  fn_ptrs_[1] = state;
  fn_ptrs_[2] = options;
}

WgResultCode RrProxy::get_function(uint32_t feature, const char* name, void** out) {
  if (feature != kRR)
    return orig_.get_function ? orig_.get_function(feature, name, out) : WG_E_MISSING_OR_INVALID_API;
  std::lock_guard<std::recursive_mutex> lock(mu_);
  ++d_rr_lookups_;  // clean/rr-sl-diag: every RR lookup, one line per name and answer
  const std::string what = "RR function lookup \"" + std::string(name ? name : "(no name)") + "\": ";
  WgResultCode rc;
  if (!feature_precheck(&rc, true)) {
    diag_once(what + "not served, answer " + rcs(rc) +
              (rc == WG_E_DEVICE_NOT_CREATED ? " (before set-device)" : " (<init> not seen yet)"));
    return rc;
  }
  if (!name || !out) {
    diag_once(what + "not served, answer " + rcs(WG_E_INVALID_PARAMETER) + " (no name or no output pointer)");
    return WG_E_INVALID_PARAMETER;
  }
  const std::string* names[3] = {&cfg_.fn_optimal, &cfg_.fn_state, &cfg_.fn_options};
  for (int i = 0; i < 3; ++i) {
    if (!names[i]->empty() && *names[i] == name && fn_ptrs_[i]) {
      *out = fn_ptrs_[i];
      diag_once(what + "served (answer 0)");
      return WG_OK;
    }
  }
  log(std::string("rr: unknown feature function requested: ") + name);
  diag_once(what + "not served, answer " + rcs(WG_E_INVALID_PARAMETER) + " (not one of the RR feature functions)");
  return WG_E_INVALID_PARAMETER;  // provisional (QUESTIONS-60-63.md Q-L4)
}

WgResultCode RrProxy::set_device(void* device) {
  {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    void* native = nullptr;
    unwrap(device, &native);  // record the native device (60.7)
    native_device_ = native;
    device_set_ = device != nullptr;
    if (rt_ && rt_->set_device && native) rt_->set_device(rt_->ctx, native);
  }
  return orig_.set_device ? orig_.set_device(device) : WG_E_MISSING_OR_INVALID_API;
}

void RrProxy::store_tags(std::map<std::pair<uint32_t, uint32_t>, StoredTag>& dst, const WgViewport* vp,
                         const WgTag* tags, uint32_t count, void* cmd_list) {
  uint32_t vid = vp ? vp->id : WG_VIEWPORT_INVALID;
  for (uint32_t i = 0; tags && i < count; ++i) {
    const WgTag& t = tags[i];
    auto key = std::make_pair(vid, t.buffer_type);
    auto it = dst.find(key);
    if (it != dst.end()) {  // last tag wins (60.7)
      release_tag(it->second);
      dst.erase(it);
    }
    if (!t.resource) continue;  // a null resource removes the tag
    StoredTag s;
    s.res = *t.resource;
    s.resource = t.resource->native;
    fill_size(rt_, s);
    s.type = t.buffer_type;
    s.lifetime = t.lifetime;
    s.extent = t.extent;
    for (const WgHeader* p = t.h.next; p; p = p->next) {
      if (cfg_.ids.has_tag_conversion && guid_eq(p->type, cfg_.ids.tag_conversion)) {
        s.has_conversion = true;
        s.conversion = *reinterpret_cast<const WgTagConversion*>(p);
        break;
      }
    }
    if (t.lifetime == WG_LIFE_ONLY_DURING_TAG && f_reads(t.buffer_type) && rr_requested_) {
      // 62.1: consume (copy) during the tagging call, on its command list
      s.resource = nullptr;
      void* native_cmd = nullptr;
      unwrap(cmd_list, &native_cmd);
      void* copy = nullptr;
      BracketScope bracket(bracket_);  // the copy is recorded on the game's list
      if (rt_ && rt_->retain_transient && native_cmd && s.res.native) bracket.open(native_cmd);
      // bug hunt C: with the declared state when the runtime takes it (moved to COPY_SOURCE for the copy and back)
      const bool with_state = rt_ && rt_->size >= WG_RR_RUNTIME_V1_STATE_SIZE && rt_->retain_transient_state;
      if (rt_ && rt_->retain_transient && native_cmd && s.res.native &&
          (with_state ? rt_->retain_transient_state(rt_->ctx, native_cmd, s.res.native, s.res.native_format, s.res.state, &copy)
                      : rt_->retain_transient(rt_->ctx, native_cmd, s.res.native, s.res.native_format, &copy)) == WG_OK && copy) {
        s.resource = copy;
        s.transient = copy;
      } else {
        log("rr: lifetime-0 input of buffer type " + u32s(t.buffer_type) + " could not be copied");
      }
    }
    dst[key] = s;
  }
}

namespace {
// clean/rr-sl-diag: "N tag(s): type WxH, ..." of a tag call
std::string tag_list(const WgTag* tags, uint32_t count) {
  std::string l = std::to_string(count) + " tag(s):";
  for (uint32_t i = 0; tags && i < count && i < 24; ++i)
    l += " " + std::to_string(tags[i].buffer_type) + (tags[i].resource ? "" : "(removed)") + " " +
         std::to_string(tags[i].extent.width) + "x" + std::to_string(tags[i].extent.height) + " life " +
         std::to_string(tags[i].lifetime) + (i + 1 < count ? "," : "");
  return l;
}
}  // namespace

WgResultCode RrProxy::tag(const WgViewport* vp, const WgTag* tags, uint32_t count, void* cmd_list) {
  {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    store_tags(viewport_tags_, vp, tags, count, cmd_list);  // copy tags (60.6.3), then forward
  }
  const WgResultCode rc = orig_.tag ? orig_.tag(vp, tags, count, cmd_list) : WG_E_MISSING_OR_INVALID_API;
  std::lock_guard<std::recursive_mutex> lock(mu_);
  if (diag_first(kFirstTag))
    diag_once("first tag call: viewport " + (vp ? u32s(vp->id) : std::string("none")) + ", " + tag_list(tags, count) +
              "; forwarded, answer " + rcs(rc));
  return rc;
}

RrProxy::FrameData* RrProxy::frame_for(const WgFrameToken* token, bool create) {
  if (!token) return nullptr;
  const uint64_t key = frame_key(token, cfg_.frame_by_token);
  auto it = frames_.find(key);
  if (it != frames_.end()) return it->second.get();
  if (!create) return nullptr;
  while (frames_.size() >= cfg_.max_frames) {  // evict the oldest frame
    auto oldest = frames_.begin();
    for (auto j = frames_.begin(); j != frames_.end(); ++j)
      if (j->second->seq < oldest->second->seq) oldest = j;
    release_frame(*oldest->second);
    frames_.erase(oldest);
  }
  std::unique_ptr<FrameData> f(new FrameData());
  f->seq = ++frame_seq_;
  FrameData* raw = f.get();
  frames_[key] = std::move(f);
  return raw;
}

WgResultCode RrProxy::tag_for_frame(const WgFrameToken* token, const WgViewport* vp, const WgTag* tags, uint32_t count,
                                void* cmd_list) {
  {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    if (FrameData* f = frame_for(token, true)) store_tags(f->tags, vp, tags, count, cmd_list);  // keyed by frame (60.7)
    diag_read();
    if (diag_on()) {
      std::string l = "diag tag-for-frame: token " + hexp(token) + " index " + u32s(frame_index_of(token)) + ":";
      for (uint32_t i = 0; tags && i < count; ++i)
        if (tags[i].buffer_type == WG_BT_COLOUR_IN || tags[i].buffer_type == WG_BT_MOTION ||
            tags[i].buffer_type == WG_BT_DEPTH)
          l += " " + u32s(tags[i].buffer_type) + "=" + hexp(tags[i].resource ? tags[i].resource->native : nullptr);
      log(l);
    }
  }
  const WgResultCode rc =
      orig_.tag_for_frame ? orig_.tag_for_frame(token, vp, tags, count, cmd_list) : WG_E_MISSING_OR_INVALID_API;
  std::lock_guard<std::recursive_mutex> lock(mu_);
  if (diag_first(kFirstTagForFrame))
    diag_once("first tag-for-frame call: viewport " + (vp ? u32s(vp->id) : std::string("none")) + ", " +
              tag_list(tags, count) + "; forwarded, answer " + rcs(rc));
  return rc;
}

WgResultCode RrProxy::constants(const WgConstants* c, const WgFrameToken* token, const WgViewport* vp) {
  {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    if (c) {
      WgConstants copy;
      memset(&copy, 0, sizeof copy);
      size_t n = version_of(c->h) >= 2 ? sizeof(WgConstants) : 451;  // v1 ends after +450 (60.6.4)
      memcpy(&copy, c, n);
      if (n < sizeof(WgConstants)) copy.min_depth_separation = 40.0f;
      uint32_t vid = vp ? vp->id : WG_VIEWPORT_INVALID;
      if (FrameData* f = frame_for(token, true)) {
        f->consts[vid] = copy;  // a later copy replaces (60.7)
        f->consts_index[vid] = frame_index_of(token);
      }
      latest_constants_[vid] = copy;
      diag_read();
      if (diag_on()) {
        char b[200];
        snprintf(b, sizeof b, "diag constants: token %p index %u vp %u jitter (%.4f, %.4f) reset %u", (const void*)token,
                 frame_index_of(token), vid, copy.jitter[0], copy.jitter[1], (unsigned)copy.reset);
        log(b);
      }
    }
  }
  const WgResultCode rc = orig_.constants ? orig_.constants(c, token, vp) : WG_E_MISSING_OR_INVALID_API;
  std::lock_guard<std::recursive_mutex> lock(mu_);
  if (diag_first(kFirstConstants)) {  // clean/rr-sl-diag
    char b[260];
    if (c)
      snprintf(b, sizeof b,
               "first constants: viewport %s, version %llu, jitter (%.4f, %.4f), vector scale (%g, %g), reset %u, depth "
               "inverted %u, camera motion included %u; forwarded, answer %u",
               vp ? u32s(vp->id).c_str() : "none", (unsigned long long)version_of(c->h), c->jitter[0], c->jitter[1],
               c->mv_scale[0], c->mv_scale[1], (unsigned)c->reset, (unsigned)c->depth_inverted,
               (unsigned)c->camera_motion_included, (unsigned)rc);
    else
      snprintf(b, sizeof b, "first constants: no constants given; forwarded, answer %u", (unsigned)rc);
    diag_once(b);
  }
  return rc;
}

WgResultCode RrProxy::new_frame_token(WgFrameToken** out, const uint32_t* frame_index) {
  WgResultCode rc = orig_.new_frame_token ? orig_.new_frame_token(out, frame_index) : WG_E_MISSING_OR_INVALID_API;
  if (rc == WG_OK && out && *out) {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    token_index_[*out] = frame_index_of(*out);  // remembered; never used to reject a frame (60.7)
    diag_read();
    if (diag_on()) {
      char b[160];
      snprintf(b, sizeof b, "diag new-frame-token: token %p index %u (argument %s%u)", (const void*)*out,
               frame_index_of(*out), frame_index ? "" : "none ", frame_index ? *frame_index : 0u);
      log(b);
    }
  }
  return rc;
}

// ------------------------------------------------------------------------------------------------------------
// Feature functions

WgResultCode RrProxy::rr_optimal(const WgRrOptions* options, WgRrOptimal* out) {
  std::lock_guard<std::recursive_mutex> lock(mu_);
  WgResultCode rc;
  if (!feature_precheck(&rc, false)) return rc;
  rc = answer_optimal(cfg_.api_minor, options, out);  // 61.5, per Streamline variant
  char b[220];  // clean/rr-sl-diag: one line per distinct query and answer
  snprintf(b, sizeof b, "RR optimal-settings query: mode %u, output %ux%u: answer %u, render %ux%u (min %ux%u, max %ux%u)",
           options ? options->mode : 0u, options ? options->output_width : 0u, options ? options->output_height : 0u,
           (unsigned)rc, out && rc == WG_OK ? out->optimal_width : 0u, out && rc == WG_OK ? out->optimal_height : 0u,
           out && rc == WG_OK ? out->min_width : 0u, out && rc == WG_OK ? out->min_height : 0u,
           out && rc == WG_OK ? out->max_width : 0u, out && rc == WG_OK ? out->max_height : 0u);
  diag_once(b);
  return rc;
}

WgResultCode RrProxy::rr_state(const WgViewport* vp, WgRrState* out) {
  std::lock_guard<std::recursive_mutex> lock(mu_);
  WgResultCode rc;
  if (!feature_precheck(&rc, false)) return rc;
  if (!vp || !out) return WG_E_INVALID_PARAMETER;
  uint64_t bytes = 0;
  if (vp_ && vp_->id == vp->id && vp_->inst && rt_ && rt_->instance_vram)
    bytes = rt_->instance_vram(rt_->ctx, vp_->inst->handle);
  out->estimated_vram = bytes;  // 0 when no instance exists for the viewport (60.8)
  return WG_OK;
}

WgResultCode RrProxy::rr_options(const WgViewport* vp, const WgRrOptions* options) {
  std::lock_guard<std::recursive_mutex> lock(mu_);
  const WgResultCode rc = rr_options_body(vp, options);
  char b[220];  // clean/rr-sl-diag: one line per distinct set of key values and answer
  if (options)
    snprintf(b, sizeof b, "RR options: viewport %s, mode %u, output %ux%u, HDR %u, normal/roughness packing %u, version %llu: answer %u",
             vp ? u32s(vp->id).c_str() : "none", options->mode, options->output_width, options->output_height,
             (unsigned)options->colour_is_hdr, options->normal_roughness_packing,
             (unsigned long long)version_of(options->h), (unsigned)rc);
  else
    snprintf(b, sizeof b, "RR options: no options given: answer %u", (unsigned)rc);
  diag_once(b);
  return rc;
}

WgResultCode RrProxy::rr_options_body(const WgViewport* vp, const WgRrOptions* options) {
  WgResultCode rc;
  if (!feature_precheck(&rc, false)) return rc;
  if (!vp || !options) return WG_E_INVALID_PARAMETER;
  if (vp_ && vp_->id != vp->id) {
    Viewport* dummy;
    return claim_viewport(vp->id, &dummy);  // 38, logged once
  }
  // 61.3 WrongGPU's choice: output size 0 / not set and modes > 6 are rejected (nothing recorded).
  if (options->mode > WG_RR_MODE_NATIVE_AA || options->output_width == 0 || options->output_height == 0 ||
      options->output_width == WG_SIZE_NOT_SET || options->output_height == WG_SIZE_NOT_SET)
    return WG_E_INVALID_PARAMETER;
  Viewport* v;
  rc = claim_viewport(vp->id, &v);
  if (rc) return rc;
  // copy honouring the version (60.6.5): absent members keep their defaults
  WgRrOptions o = default_options();
  uint64_t ver = version_of(options->h);
  size_t end = ver >= 3 ? WG_RR_OPTIONS_V3_END : ver == 2 ? WG_RR_OPTIONS_V2_END : WG_RR_OPTIONS_V1_END;
  memcpy(reinterpret_cast<uint8_t*>(&o) + sizeof(WgHeader),
         reinterpret_cast<const uint8_t*>(options) + sizeof(WgHeader), end - sizeof(WgHeader));
  o.h.next = nullptr;
  o.h.version = ver;
  v->opts = o;
  v->opts_version = ver;
  v->has_options = true;
  return WG_OK;
}

// ------------------------------------------------------------------------------------------------------------
// Creation (61.3), allocate / free, evaluate (61.4)

bool RrProxy::lookup(uint32_t type, uint32_t vp, const std::vector<StoredTag>& locals, const FrameData* frame,
                     const StoredTag** out) const {
  // Precedence (61.4): local tag, frame-scoped tag (2.8.0 and 2.14.1: tag-for-frame), viewport tag.
  for (const StoredTag& t : locals)
    if (t.type == type) { *out = &t; return t.resource != nullptr; }
  auto key = std::make_pair(vp, type);
  if (frame && cfg_.api_minor >= 8) {
    auto it = frame->tags.find(key);
    if (it != frame->tags.end()) { *out = &it->second; return it->second.resource != nullptr; }
  }
  auto it = viewport_tags_.find(key);
  if (it != viewport_tags_.end()) { *out = &it->second; return it->second.resource != nullptr; }
  *out = nullptr;
  return false;
}

WgResultCode RrProxy::create_or_update(Viewport& v, const std::vector<StoredTag>& locals, const FrameData* frame,
                                   const WgConstants* consts, const SettingsSnapshot& snap, bool from_allocate,
                                   bool* creating, WgRrCreate* desc) {
  *creating = false;
  bool need = !v.inst;
  if (!need) {  // recreate triggers (61.3)
    const WgRrOptions& a = v.inst->opts;
    const WgRrOptions& b = v.opts;
    need = a.mode != b.mode || a.output_width != b.output_width || a.output_height != b.output_height ||
           a.normal_roughness_packing != b.normal_roughness_packing ||
           (v.opts_version >= 3 && memcmp(a.presets, b.presets, sizeof a.presets) != 0);
  }
  if (!need) return WG_OK;
  if (rr_off_) {  // the player's switch (set_rr_off): never create, even when the game skips <is-supported>
    if (!logged_rr_off_create_) {
      logged_rr_off_create_ = true;
      log("rr: the RR feature is not created (" + std::string(from_allocate ? "allocate" : "evaluate") + ", 7): " +
          rr_off_why_);
    }
    return WG_E_ADAPTER_NOT_SUPPORTED;
  }
  if (!snap.on(kSetWrongGpuRr)) {  // 60.9 / 61.3: the feature is not created
    bool& once = from_allocate ? logged_rr_off_alloc_ : logged_rr_off_eval_;
    if (!once) {
      once = true;
      log(std::string("rr: setting \"WrongGPU RR\" is off; the RR feature is not created (") +
          (from_allocate ? "allocate" : "evaluate") + ", 7)");
    }
    return WG_E_ADAPTER_NOT_SUPPORTED;
  }
  if (!runtime_usable()) {  // logged once: this runs at every evaluate while the viewport has no instance
    if (!logged_no_runtime_) {
      logged_no_runtime_ = true;
      log("rr: WrongGPU's runtime or model is not available (7)");
    }
    return WG_E_ADAPTER_NOT_SUPPORTED;
  }
  const StoredTag* lin = nullptr;
  const StoredTag* hw = nullptr;
  bool has_lin = lookup(WG_BT_LINEAR_DEPTH, v.id, locals, frame, &lin);
  bool has_hw = lookup(WG_BT_DEPTH, v.id, locals, frame, &hw);
  if (!has_lin && !has_hw) {
    if (!logged_no_depth_) {
      logged_no_depth_ = true;
      log("rr: no depth input tagged at feature creation (20)");
    }
    return WG_E_MISSING_INPUT_PARAMETER;
  }
  memset(desc, 0, sizeof *desc);
  desc->size = sizeof(WgRrCreate);
  desc->api_minor = cfg_.api_minor;
  desc->device = native_device_;
  desc->output_width = v.opts.output_width;
  desc->output_height = v.opts.output_height;
  desc->mode = v.opts.mode;
  desc->packing = v.opts.normal_roughness_packing;
  memcpy(desc->presets, v.opts.presets, sizeof desc->presets);
  desc->colour_hdr = v.opts.colour_is_hdr == WG_TRI_TRUE;                          // only == 1
  desc->depth_inverted = consts && consts->depth_inverted == WG_TRI_TRUE;           // only == 1
  desc->mv_jittered = consts && consts->mv_jittered == WG_TRI_TRUE;                 // only == 1
  desc->alpha_upscaling = v.opts_version >= 2 && v.opts.alpha_upscaling == WG_TRI_TRUE;
  desc->sharpen_request = v.opts.sharpness > 0.0f;
  desc->hw_depth = !has_lin;
  const StoredTag* colour = nullptr;
  const StoredTag* mv = nullptr;
  bool has_colour = lookup(WG_BT_COLOUR_IN, v.id, locals, frame, &colour);
  bool has_mv = lookup(WG_BT_MOTION, v.id, locals, frame, &mv);
  if (has_colour && has_mv) {
    uint32_t cw, ch, mw, mh;
    input_size(*colour, &cw, &ch);
    input_size(*mv, &mw, &mh);
    desc->mv_output_resolution = mw > cw || mh > ch;  // larger in width OR height (61.3)
  }
  if (has_colour) input_size(*colour, &desc->render_width, &desc->render_height);
  desc->latched_settings = snap.on(kSetWrongGpuRr) ? 1u : 0u;
  *creating = true;
  return WG_OK;
}

WgResultCode RrProxy::allocate(void* cmd_list, uint32_t feature, const WgViewport* vp) {
  if (feature != kRR) return orig_.allocate ? orig_.allocate(cmd_list, feature, vp) : WG_E_MISSING_OR_INVALID_API;
  std::lock_guard<std::recursive_mutex> lock(mu_);
  const WgResultCode rc = allocate_body(cmd_list, vp);
  if (diag_first(kFirstAllocate))  // clean/rr-sl-diag
    diag_once("first RR allocate: viewport " + (vp ? u32s(vp->id) : std::string("none")) + ": answer " + rcs(rc) +
              (has_instance() ? ", instance created" : ", no instance"));
  return rc;
}

WgResultCode RrProxy::allocate_body(void* cmd_list, const WgViewport* vp) {
  WgResultCode rc;
  if (!feature_precheck(&rc, true)) return rc;
  if (!vp) return WG_E_INVALID_PARAMETER;
  Viewport* v;
  if ((rc = claim_viewport(vp->id, &v)) != WG_OK) return rc;
  if (!v->has_options || v->opts.mode == WG_RR_MODE_OFF) return WG_E_INVALID_INTEGRATION;
  SettingsSnapshot snap = settings_->snapshot();  // <allocate> takes its own snapshot (60.9)
  auto lc = latest_constants_.find(vp->id);
  const WgConstants* consts = lc == latest_constants_.end() ? nullptr : &lc->second;
  std::vector<StoredTag> none;
  bool creating = false;
  WgRrCreate desc;
  if ((rc = create_or_update(*v, none, nullptr, consts, snap, true, &creating, &desc)) != WG_OK) return rc;
  if (!creating) return WG_OK;
  void* handle = nullptr;
  if ((rc = rt_->create_instance(rt_->ctx, &desc, &handle)) != WG_OK) return rc;
  ++d_rr_instances_;  // clean/rr-sl-diag
  destroy_instance(*v);
  v->inst.reset(new Instance(desc.render_width, desc.render_height));
  v->inst->handle = handle;
  v->inst->desc = desc;
  v->inst->opts = v->opts;
  v->inst->opts_version = v->opts_version;
  v->logged_range = v->logged_camera_motion = v->logged_mv_res = v->logged_format = v->logged_camera_fallback =
        v->logged_zero = false;
  return WG_OK;
}

WgResultCode RrProxy::free_resources(uint32_t feature, const WgViewport* vp) {
  if (feature != kRR) return orig_.free_resources ? orig_.free_resources(feature, vp) : WG_E_MISSING_OR_INVALID_API;
  std::lock_guard<std::recursive_mutex> lock(mu_);
  WgResultCode rc;
  if (!feature_precheck(&rc, false)) return rc;
  if (vp && vp_ && vp_->id == vp->id) {  // releases the held viewport (61.2)
    destroy_instance(*vp_);
    vp_.reset();
  }
  return WG_OK;  // a viewport that is not held: 0, nothing done
}

WgResultCode RrProxy::evaluate(uint32_t feature, const WgFrameToken* token, const WgHeader* const* inputs, uint32_t count,
                           void* cmd_list) {
  if (feature != kRR) {
    const WgResultCode frc =
        orig_.evaluate ? orig_.evaluate(feature, token, inputs, count, cmd_list) : WG_E_MISSING_OR_INVALID_API;
    std::lock_guard<std::recursive_mutex> lock(mu_);  // clean/rr-sl-diag: one line per other feature and answer
    diag_once("evaluate of feature " + u32s(feature) + " (not RR, forwarded to the game interface): answer " + rcs(frc));
    return frc;
  }
  std::lock_guard<std::recursive_mutex> lock(mu_);
  ++d_rr_evals_;
  const bool first = diag_first(kFirstEvaluate);
  const WgResultCode rc = evaluate_body(token, inputs, count, cmd_list);
  if (first)  // clean/rr-sl-diag
    diag_once("first RR evaluate: " + u32s(count) + " input(s): answer " + rcs(rc) +
              (has_instance() ? ", instance present" : ", no instance"));
  return rc;
}

WgResultCode RrProxy::evaluate_body(const WgFrameToken* token, const WgHeader* const* inputs, uint32_t count,
                                    void* cmd_list) {
  WgResultCode rc;
  if (!feature_precheck(&rc, true)) return rc;
  last_eval_completed_ = false;
  const SettingsSnapshot snap = settings_->snapshot();  // one snapshot for the whole call (60.9)

  // Step 1: the viewport handle and local tags of the input array (by GUID).
  const WgViewport* vph = nullptr;
  std::vector<StoredTag> locals;
  for (uint32_t i = 0; inputs && i < count; ++i) {
    for (const WgHeader* p = inputs[i]; p; p = p->next) {
      if (guid_eq(p->type, cfg_.ids.viewport)) {
        if (!vph) vph = reinterpret_cast<const WgViewport*>(p);
      } else if (guid_eq(p->type, cfg_.ids.tag)) {
        const WgTag* t = reinterpret_cast<const WgTag*>(p);
        StoredTag s;
        s.type = t->buffer_type;
        s.lifetime = t->lifetime;
        s.extent = t->extent;
        if (t->resource) {
          s.res = *t->resource;
          s.resource = t->resource->native;
          fill_size(rt_, s);
        }
        for (const WgHeader* q = t->h.next; q; q = q->next)
          if (cfg_.ids.has_tag_conversion && guid_eq(q->type, cfg_.ids.tag_conversion)) {
            s.has_conversion = true;
            s.conversion = *reinterpret_cast<const WgTagConversion*>(q);
            break;
          }
        locals.erase(std::remove_if(locals.begin(), locals.end(),
                                    [&](const StoredTag& x) { return x.type == s.type; }), locals.end());
        locals.push_back(s);
      }
    }
  }
  if (!vph) return WG_E_MISSING_INPUT_PARAMETER;
  Viewport* v;
  if ((rc = claim_viewport(vph->id, &v)) != WG_OK) return rc;

  // 61.3: mode off -> 19.
  if (!v->has_options || v->opts.mode == WG_RR_MODE_OFF) return WG_E_INVALID_INTEGRATION;

  const FrameData* frame = frame_for(token, false);
  const WgConstants* consts = nullptr;
  if (frame) {
    auto it = frame->consts.find(v->id);
    if (it != frame->consts.end()) consts = &it->second;
  }
  diag_read();
  ++diag_evals_;
  if (diag_on()) {
    char b[260];
    auto ci = frame ? frame->consts_index.find(v->id) : std::map<uint32_t, uint32_t>::const_iterator();
    const bool has_ci = frame && ci != frame->consts_index.end();
    snprintf(b, sizeof b, "diag evaluate %llu: token %p index %u; constants %s (set at index %d) jitter (%.4f, %.4f)",
             (unsigned long long)diag_evals_, (const void*)token, frame_index_of(token),
             consts ? "found" : "missing", has_ci ? (int)ci->second : -1, consts ? consts->jitter[0] : 0.0f,
             consts ? consts->jitter[1] : 0.0f);
    std::string l = b;
    const StoredTag* c = nullptr;
    if (lookup(WG_BT_COLOUR_IN, v->id, locals, frame, &c) && c) l += "; colour " + hexp(c->resource);
    const StoredTag* m = nullptr;
    if (lookup(WG_BT_MOTION, v->id, locals, frame, &m) && m) l += " motion " + hexp(m->resource);
    log(l);
  }

  // Create-or-update (61.3): "WrongGPU RR" off at creation -> 7; runtime unusable -> 7; no depth -> 20.
  bool creating = false;
  WgRrCreate desc;
  if ((rc = create_or_update(*v, locals, frame, consts, snap, false, &creating, &desc)) != WG_OK) return rc;
  const WgRrCreate& cd = creating ? desc : v->inst->desc;
  const uint32_t packing = cd.packing;

  // Step 2: required inputs.
  const StoredTag *colour, *output, *mv, *dif, *spec, *nrm = nullptr, *rgh = nullptr, *lin, *hwd;
  bool ok = lookup(WG_BT_COLOUR_IN, v->id, locals, frame, &colour);
  ok = lookup(WG_BT_COLOUR_OUT, v->id, locals, frame, &output) && ok;
  ok = lookup(WG_BT_MOTION, v->id, locals, frame, &mv) && ok;
  ok = lookup(WG_BT_DIFFUSE_ALBEDO, v->id, locals, frame, &dif) && ok;
  ok = lookup(WG_BT_SPECULAR_ALBEDO, v->id, locals, frame, &spec) && ok;
  if (packing == 1) {
    ok = lookup(WG_BT_NORMAL_ROUGHNESS, v->id, locals, frame, &nrm) && ok;
  } else {
    ok = lookup(WG_BT_NORMALS, v->id, locals, frame, &nrm) && ok;
    ok = lookup(WG_BT_ROUGHNESS, v->id, locals, frame, &rgh) && ok;
  }
  bool has_lin = lookup(WG_BT_LINEAR_DEPTH, v->id, locals, frame, &lin);
  bool has_hw = lookup(WG_BT_DEPTH, v->id, locals, frame, &hwd);
  ok = ok && (has_lin || has_hw);
  if (!ok) return WG_E_MISSING_INPUT_PARAMETER;
  const StoredTag* depth = has_lin ? lin : hwd;  // linear depth wins
  if (!consts) return WG_E_MISSING_CONSTANTS;    // 26 (the original's code: open item 5)

  // Step 3: one render size per call; zero size; accepted render-scale window.
  uint32_t rw, rh;
  input_size(*colour, &rw, &rh);
  const uint32_t ow = v->opts.output_width, oh = v->opts.output_height;
  if (rw == 0 || rh == 0 || ow == 0 || oh == 0) {
    if (v->inst) v->inst->lc.request_reset();  // the only rejection that makes a reset pending
    if (!v->logged_zero) {
      v->logged_zero = true;
      log("rr: zero render or output size rejected (25); reset pending");
    }
    return WG_E_INVALID_PARAMETER;
  }
  if (rw < min_render_size(ow) || rw > ow || rh < min_render_size(oh) || rh > oh) {
    if (!v->logged_range) {
      v->logged_range = true;
      log("rr: render size " + u32s(rw) + "x" + u32s(rh) + " outside 1/3..1 of output " + u32s(ow) + "x" +
          u32s(oh) + " (25)");
    }
    return WG_E_INVALID_PARAMETER;
  }
  // Step 4: camera motion not included -> rejected (WrongGPU's choice, open item 7).
  if (consts->camera_motion_included == WG_TRI_FALSE) {
    if (!v->logged_camera_motion) {
      v->logged_camera_motion = true;
      log("rr: motion vectors without camera motion are not supported (25)");
    }
    return WG_E_INVALID_PARAMETER;
  }
  // 62.3: output-resolution motion vectors -> rejected.
  if (cd.mv_output_resolution) {
    if (!v->logged_mv_res) {
      v->logged_mv_res = true;
      log("rr: output-resolution motion vectors are not supported (25)");
    }
    return WG_E_INVALID_PARAMETER;
  }

  // Routing of 62.2 (which inputs this call reads or writes).
  struct Use { WgRrRole role; const StoredTag* tag; };
  std::vector<Use> uses = {{WG_ROLE_COLOUR, colour}, {WG_ROLE_OUTPUT, output}, {WG_ROLE_DEPTH, depth},
                           {WG_ROLE_MOTION, mv}, {WG_ROLE_DIFFUSE_ALBEDO, dif}, {WG_ROLE_SPECULAR_ALBEDO, spec},
                           {WG_ROLE_NORMALS, nrm}};
  if (rgh) uses.push_back({WG_ROLE_ROUGHNESS, rgh});
  const StoredTag* t = nullptr;
  if (lookup(WG_BT_SPECULAR_MOTION, v->id, locals, frame, &t)) uses.push_back({WG_ROLE_SPECULAR_MOTION, t});
  if (lookup(WG_BT_SPECULAR_HIT_DISTANCE, v->id, locals, frame, &t)) uses.push_back({WG_ROLE_SPECULAR_HIT_DIST, t});
  if (lookup(WG_BT_DISOCCLUSION, v->id, locals, frame, &t)) uses.push_back({WG_ROLE_DISOCCLUSION, t});
  if (lookup(WG_BT_COLOUR_BEFORE_FOG, v->id, locals, frame, &t)) uses.push_back({WG_ROLE_G1, t});  // 41 replaces 58
  else if (lookup(WG_BT_SSS_GUIDE, v->id, locals, frame, &t)) uses.push_back({WG_ROLE_G1, t});
  if (lookup(WG_BT_DOF_GUIDE, v->id, locals, frame, &t)) uses.push_back({WG_ROLE_DOF_GUIDE, t});
  if (lookup(WG_BT_COLOUR_BEFORE_TRANSPARENCY, v->id, locals, frame, &t))
    uses.push_back({WG_ROLE_BEFORE_TRANSPARENCY, t});
  const StoredTag* t51 = nullptr;
  bool has51 = lookup(WG_BT_TRANSPARENCY_LAYER, v->id, locals, frame, &t51);
  const StoredTag* t39 = nullptr;
  bool has39 = lookup(WG_BT_COLOUR_BEFORE_PARTICLES, v->id, locals, frame, &t39);
  bool layer = has39 || has51;
  if (layer) uses.push_back({WG_ROLE_LAYER, has39 ? t39 : t51});  // 39 replaces 51, also alone
  if (layer && lookup(WG_BT_TRANSPARENCY_OPACITY, v->id, locals, frame, &t)) uses.push_back({WG_ROLE_OPACITY, t});
  if (cfg_.api_minor >= 14 && cd.alpha_upscaling) {  // 2.14.1 plugin with alpha upscaling only
    if (lookup(WG_BT_INPUT_ALPHA, v->id, locals, frame, &t)) uses.push_back({WG_ROLE_INPUT_ALPHA, t});
    if (lookup(WG_BT_OUTPUT_ALPHA, v->id, locals, frame, &t)) uses.push_back({WG_ROLE_OUTPUT_ALPHA, t});
  }

  // Step 9: formats (62.5) of every input this call uses.
  WgRrFrame f;
  memset(&f, 0, sizeof f);
  for (const Use& u : uses) {
    const StoredTag& s = *u.tag;
    uint32_t fmt = s.res.native_format, samples = 1;
    if (rt_->describe_resource) {
      uint32_t nf = 0, ns = 1;
      if (rt_->describe_resource(rt_->ctx, s.resource, &nf, &ns) == WG_OK) {
        if (fmt == 0) fmt = nf;
        samples = ns;
      }
    }
    uint32_t cls = samples > 1 ? (uint32_t)WG_FMT_UNSUPPORTED : format_class(fmt, u.role == WG_ROLE_DEPTH);
    if (cls == WG_FMT_UNSUPPORTED) {
      if (!v->logged_format) {
        v->logged_format = true;
        log("rr: unsupported format " + u32s(fmt) + (samples > 1 ? " (multisampled)" : "") + " for buffer type " +
            u32s(s.type) + " (25)");
      }
      return WG_E_INVALID_PARAMETER;
    }
    WgRrImage& im = f.images[u.role];
    im.resource = s.resource;
    im.buffer_type = s.type;
    im.format = fmt;
    im.format_class = cls;
    im.state = s.res.state;
    im.width = s.res.width;
    im.height = s.res.height;
    im.origin_x = s.extent.left;
    im.origin_y = s.extent.top;
    im.extent_width = s.extent.width;
    im.extent_height = s.extent.height;
    im.lifetime = s.lifetime;
    if (s.has_conversion && s.conversion.kind == 1) {
      im.conversion = 1;
      im.conversion_scale = s.conversion.scale;
      im.conversion_bias = s.conversion.bias;
    }
    f.attached |= 1u << u.role;
  }

  // Commit the creation (all rejections passed).
  if (creating) {
    desc.render_width = rw;  // created at this call's render size
    desc.render_height = rh;
    void* handle = nullptr;
    if ((rc = rt_->create_instance(rt_->ctx, &desc, &handle)) != WG_OK) return rc;
    ++d_rr_instances_;  // clean/rr-sl-diag
    destroy_instance(*v);
    v->inst.reset(new Instance(rw, rh));
    v->inst->handle = handle;
    v->inst->desc = desc;
    v->inst->opts = v->opts;
    v->inst->opts_version = v->opts_version;
    v->logged_range = v->logged_camera_motion = v->logged_mv_res = v->logged_format = v->logged_camera_fallback =
        v->logged_zero = false;
  }
  Instance& inst = *v->inst;

  // Frame plan (page 53 bookkeeping).
  CameraPair cam;
  memcpy(cam.v2c, consts->view_to_clip, sizeof cam.v2c);
  memcpy(cam.w2v, v->opts.world_to_view, sizeof cam.w2v);  // options in effect for this evaluate
  FramePlan plan = inst.lc.begin(rw, rh, consts->jitter, cam, consts->reset == WG_TRI_TRUE,
                                 snap.on(kSetChangeFrameReset));
  f.size = sizeof f;
  f.api_minor = cfg_.api_minor;
  unwrap(cmd_list, &f.command_list);
  f.render_width = rw;
  f.render_height = rh;
  f.output_width = ow;
  f.output_height = oh;
  f.jitter[0] = plan.jitter[0];
  f.jitter[1] = plan.jitter[1];
  f.prev_jitter[0] = plan.prev_jitter[0];
  f.prev_jitter[1] = plan.prev_jitter[1];
  volatile float msx = consts->mv_scale[0] * (float)rw;  // f32 (52.8)
  volatile float msy = consts->mv_scale[1] * (float)rh;
  f.mv_scale[0] = msx;
  f.mv_scale[1] = msy;
  f.reset = plan.reset;
  f.parity = plan.parity;
  f.counter = plan.counter;
  f.rebuild_render_items = plan.rebuild;
  f.depth_is_linear = has_lin;
  f.settings = (snap.on(kSetSpecularMotionFromV) ? WG_RR_SET_SPECULAR_MOTION_FROM_V : 0u) |
               (snap.on(kSetBoundedStores) ? WG_RR_SET_BOUNDED_STORES : 0u) |
               (snap.on(kSetHitDistanceFilter) ? WG_RR_SET_HIT_DISTANCE_FILTER : 0u);
  memcpy(f.view_to_clip, plan.current.v2c, sizeof f.view_to_clip);
  memcpy(f.world_to_view, plan.current.w2v, sizeof f.world_to_view);
  memcpy(f.prev_view_to_clip, plan.effective_prev.v2c, sizeof f.prev_view_to_clip);
  memcpy(f.prev_world_to_view, plan.effective_prev.w2v, sizeof f.prev_world_to_view);
  if (has51) {  // the layer origin is always type 51's; (0, 0) without it
    f.layer_origin_x = t51->extent.left;
    f.layer_origin_y = t51->extent.top;
  }

  WgRrFrameResult res;
  memset(&res, 0, sizeof res);
  res.size = sizeof res;
  BracketScope bracket(bracket_);  // clean/rr-re9-pt: evaluate and the pass-through record on the game's list
  bracket.open(f.command_list);
  rc = rt_->evaluate(rt_->ctx, inst.handle, &f, &res);
  ++evaluates_;
  if (res.camera_fallback_used && !v->logged_camera_fallback) {
    v->logged_camera_fallback = true;
    log("rr: camera matrices unusable; specular motion = motion vectors (logged once per instance)");
  }
  if (rc != WG_OK) {
    // Fail-safe (WrongGPU's choice, clean/rr-f-live1): the runtime failed before any pass of its own, so the game's
    // list holds nothing of ours; the colour is passed through into the output and the call reports success, so the
    // frame stays presentable. The next evaluate is a reset frame (the temporal state missed this frame).
    uint32_t prc = WG_E_INVALID_PARAMETER;
    if (res.passes_started == 0 && rt_->size >= WG_RR_RUNTIME_V1_LIVE_SIZE && rt_->passthrough)
      prc = rt_->passthrough(rt_->ctx, &f);
    inst.lc.fail(plan, res.input_jitter_consumed != 0, prc == WG_OK || snap.on(kSetResetAfterFailedEvaluate));
    if (rc != last_eval_rc_ || prc != last_pass_rc_) {
      if (eval_rc_lines_ < 40) {
        ++eval_rc_lines_;
        log("rr: evaluate " + u32s((uint32_t)evaluates_) + ": runtime returned " + u32s(rc) +
            (prc == WG_OK ? "; colour passed through, success reported, reset on the next evaluate"
                          : "; no pass-through (" + u32s(prc) + "), the code goes to the game"));
      }
      last_eval_rc_ = rc;
      last_pass_rc_ = prc;
    }
    if (prc == WG_OK) {
      ++passthroughs_;
      return WG_OK;
    }
    return rc;
  }
  if (last_eval_rc_ != WG_OK || evaluates_ <= 1) {
    if (eval_rc_lines_ < 40) {
      ++eval_rc_lines_;
      log("live: evaluate " + u32s((uint32_t)evaluates_) + ": runtime returned 0 (" + u32s((uint32_t)passthroughs_) +
          " frame(s) passed through so far)");
    }
    last_eval_rc_ = WG_OK;
    last_pass_rc_ = WG_OK;
  }
  inst.lc.complete(plan);
  last_complete_ms_ = clock_ ? clock_() : 0;
  last_eval_completed_ = true;
  return WG_OK;
}

// ------------------------------------------------------------------------------------------------------------

uint32_t RrProxy::status_word() const {
  uint32_t s = 0;
  if (rt_ && rt_->status) s = rt_->status(rt_->ctx) & (WG_RR_RT_READY | WG_RR_RT_MODEL_LOADED);
  if (last_eval_completed_.load()) {
    uint64_t now = clock_ ? clock_() : 0;
    if (now - last_complete_ms_.load() < 1000) s |= WG_RR_STATUS_ACTIVE;
  }
  return s;
}

bool RrProxy::has_instance() const {
  std::lock_guard<std::recursive_mutex> lock(mu_);
  return vp_ && vp_->inst;
}
bool RrProxy::holds_viewport() const {
  std::lock_guard<std::recursive_mutex> lock(mu_);
  return vp_ != nullptr;
}
uint32_t RrProxy::held_viewport() const {
  std::lock_guard<std::recursive_mutex> lock(mu_);
  return vp_ ? vp_->id : WG_VIEWPORT_INVALID;
}
const Lifecycle* RrProxy::lifecycle() const {
  std::lock_guard<std::recursive_mutex> lock(mu_);
  return vp_ && vp_->inst ? &vp_->inst->lc : nullptr;
}
const WgRrCreate* RrProxy::instance_create_desc() const {
  std::lock_guard<std::recursive_mutex> lock(mu_);
  return vp_ && vp_->inst ? &vp_->inst->desc : nullptr;
}

const char* const kSlotChecksDefault = "10,30,60,120,240,480,900";

std::string slot_check_schedule(const char* env) {
  if (!env || !*env) return std::string();
  const std::string s(env);
  if (s == "off") return std::string();
  if (s == "on") return kSlotChecksDefault;
  return s;
}

}  // namespace wgrr
