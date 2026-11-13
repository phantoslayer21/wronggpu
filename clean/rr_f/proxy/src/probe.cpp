// SPDX-License-Identifier: MIT
// Probe build of the RR proxy (see probe.h). Section numbers refer to spec/60-63-rr-f-interface.md and its annexes.
#include "probe.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <set>

#include "answers.h"
#include "formats.h"

namespace wgrr {

namespace {

constexpr uint32_t kRR = WG_FEAT_RAY_RECONSTRUCTION;

std::string u(uint64_t v) { return std::to_string(v); }

std::string hx(uint64_t v) {
  char b[24];
  snprintf(b, sizeof b, "0x%llx", (unsigned long long)v);
  return b;
}

std::string fl(float v) {
  char b[32];
  snprintf(b, sizeof b, "%.9g", (double)v);
  return b;
}

std::string rc_text(WgResultCode rc) { return std::to_string(rc); }

bool guid_eq(const WgGuid& a, const WgGuid& b) { return memcmp(a.b, b.b, 16) == 0; }

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

// The resource size: as the game filled it in the resource description, else the native description's
// (clean/rr-re9-pt: RE Requiem fills 0 x 0 and tags no extent).
uint32_t res_w(const StoredTag& t) { return t.res.width && t.res.height ? t.res.width : t.desc_width; }
uint32_t res_h(const StoredTag& t) { return t.res.width && t.res.height ? t.res.height : t.desc_height; }

// The rectangle an input covers: its extent, or the whole resource when the extent is not set (60.6.2, Q-L13).
void rect_of(const StoredTag& t, uint32_t* x, uint32_t* y, uint32_t* w, uint32_t* h) {
  if (t.extent.width == 0 || t.extent.height == 0) {
    *x = 0;
    *y = 0;
    *w = res_w(t);
    *h = res_h(t);
  } else {
    *x = t.extent.left;
    *y = t.extent.top;
    *w = t.extent.width;
    *h = t.extent.height;
  }
}

}  // namespace

struct RrProbe::Viewport {
  uint32_t id = WG_VIEWPORT_INVALID;
  bool has_options = false;
  WgRrOptions opts = default_options();
  uint64_t opts_version = 1;
  uint32_t last_render_w = 0, last_render_h = 0;
  bool has_prev = false;
  float prev_v2c20 = 0, prev_v2c21 = 0, prev_jitter[2] = {0, 0};
};

struct RrProbe::FrameData {
  uint64_t seq = 0;
  std::map<std::pair<uint32_t, uint32_t>, StoredTag> tags;  // (viewport, type), frame-scoped (2.14.1)
  std::map<uint32_t, WgConstants> consts;                   // per viewport
};

RrProbe::RrProbe(const ProbeConfig& cfg, const OriginalApi& orig, ProbeGpu* gpu, ProbeLog* log, TidFn tid)
    : cfg_(cfg), orig_(orig), gpu_(gpu), log_(log), tid_(std::move(tid)) {}

RrProbe::~RrProbe() {}

// ------------------------------------------------------------------------------------------------------------
// logging helpers

uint64_t RrProbe::begin_call() {
  std::lock_guard<std::recursive_mutex> lock(mu_);
  return ++seq_;
}

void RrProbe::first_call(const char* role, uint64_t seq) {
  std::lock_guard<std::recursive_mutex> lock(mu_);
  if (counts_[role]++ == 0)  // "whether / when" each call happens, and which RR call came first
    log_->record(std::string("first ") + role, "seq=" + u(seq) + " frame=" + u(log_->frame()));
}

void RrProbe::log_call(uint64_t seq, const char* role, const uint32_t* feature, WgResultCode rc,
                       const WgFrameToken* token, const std::string& detail) {
  first_call(role, seq);
  std::string key = std::string("call ") + role;
  if (feature) key += " feature=" + u(*feature);
  key += detail;
  std::string extra = "seq=" + u(seq) + " tid=" + u(tid_ ? tid_() : 0);
  if (token) extra += " frame_index=" + u(frame_index_of(token));
  log_->record(key, "rc=" + rc_text(rc), extra);
}

std::string RrProbe::guid_text(const WgGuid& g) const {
  char b[40];
  uint32_t d1 = (uint32_t)g.b[0] | ((uint32_t)g.b[1] << 8) | ((uint32_t)g.b[2] << 16) | ((uint32_t)g.b[3] << 24);
  uint32_t d2 = (uint32_t)g.b[4] | ((uint32_t)g.b[5] << 8);
  uint32_t d3 = (uint32_t)g.b[6] | ((uint32_t)g.b[7] << 8);
  snprintf(b, sizeof b, "%08x-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x", d1, d2, d3, g.b[8], g.b[9], g.b[10],
           g.b[11], g.b[12], g.b[13], g.b[14], g.b[15]);
  for (const auto& n : cfg_.guid_names)
    if (guid_eq(n.second, g)) return n.first + "{" + b + "}";
  return std::string("{") + b + "}";
}

std::string RrProbe::chain_text(const WgHeader* h) const {
  if (!h) return "null";
  std::string s;
  int n = 0;
  const WgHeader* p = h;
  for (; p && n < 8; p = p->next, ++n) {
    if (n) s += " -> ";
    s += guid_text(p->type) + " v" + u(p->version);
  }
  if (p) s += " -> ...";
  return s;
}

void RrProbe::log_chain(const std::string& where, const WgHeader* h) {
  log_->record("chain " + where, chain_text(h));  // GUID, structure version, next-chain GUIDs
}

void RrProbe::unwrap(void* wrapped, void** native) const {
  *native = wrapped;
  if (!wrapped || !orig_.native_interface) return;
  void* n = nullptr;
  if (orig_.native_interface(wrapped, &n) == WG_OK && n) *native = n;
}

bool RrProbe::precheck(WgResultCode* rc, bool needs_device) const {
  if (!initialised_) { *rc = WG_E_INIT_NOT_CALLED; return false; }                    // 60.1
  if (needs_device && !device_set_) { *rc = WG_E_DEVICE_NOT_CREATED; return false; }
  return true;
}

RrProbe::Viewport& RrProbe::viewport(uint32_t id) {
  auto it = vps_.find(id);
  if (it != vps_.end()) return *it->second;
  std::unique_ptr<Viewport> v(new Viewport());
  v->id = id;
  Viewport& r = *v;
  vps_[id] = std::move(v);
  return r;
}

void RrProbe::arm_watch(const char* event, uint64_t seq) {
  watch_event_ = event;
  watch_seq_ = seq;
  watch_left_ = 2;
}

uint64_t RrProbe::calls(const std::string& role) const {
  std::lock_guard<std::recursive_mutex> lock(mu_);
  auto it = counts_.find(role);
  return it == counts_.end() ? 0 : it->second;
}

uint32_t RrProbe::rr_evaluates() const {
  std::lock_guard<std::recursive_mutex> lock(mu_);
  return rr_evals_;
}

// ------------------------------------------------------------------------------------------------------------
// roles

WgResultCode RrProbe::init(const WgPreferences* prefs, uint64_t api_version) {
  uint64_t seq = begin_call();
  std::string feats;
  {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    api_version_ = api_version;
    pref_flags_ = prefs ? prefs->flags : 0;
    render_api_ = prefs ? prefs->render_api : 0;
    rr_requested_ = false;
    if (prefs && prefs->features)
      for (uint32_t i = 0; i < prefs->feature_count; ++i) {
        feats += (i ? "," : "") + u(prefs->features[i]);
        rr_requested_ |= prefs->features[i] == kRR;
      }
    log_chain("init.preferences", prefs ? &prefs->h : nullptr);
  }
  WgResultCode rc = orig_.init ? orig_.init(prefs, api_version) : WG_E_MISSING_OR_INVALID_API;  // unchanged
  {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    if (rc == WG_OK) initialised_ = true;
    std::string s = "sdk=" + u(WG_API_VERSION_MAJOR(api_version)) + "." + u(WG_API_VERSION_MINOR(api_version)) + "." +
                    u(WG_API_VERSION_PATCH(api_version)) + " value=" + hx(api_version) +
                    " magic=" + hx(WG_API_VERSION_MAGIC(api_version));
    if (prefs) {
      s += " flags=" + hx(prefs->flags) + " bit0_no_state_tracking=" + u(prefs->flags & 1) +
           " bit7_frame_tagging=" + u((prefs->flags >> 7) & 1) + " features=[" + feats + "] render_api=" +
           u(prefs->render_api) + " engine=" + u(prefs->engine) + " log_level=" + u(prefs->log_level) +
           " app_id=" + u(prefs->application_id) + " plugin_paths=" + u(prefs->plugin_path_count) +
           " prefs_version=" + u(prefs->h.version);
    } else {
      s += " prefs=null";
    }
    log_->record("init", s + " rc=" + rc_text(rc));
  }
  log_call(seq, "init", nullptr, rc, nullptr);
  return rc;
}

void RrProbe::adopt_unseen_init() {
  std::lock_guard<std::recursive_mutex> lock(mu_);
  if (initialised_) return;
  initialised_ = true;
  api_version_ = 0;
  pref_flags_ = 0;       // unknown: bit 0 taken as 0 (the game does not restore its command-list state)
  render_api_ = 0;
  rr_requested_ = true;  // unknown: lifetime-0 inputs are copied
  log_->note("init unseen: the game's <init> ran before the hook; adopted with flags unknown (bit 0 taken as 0), "
             "RR taken as requested, SDK version unknown");
}

WgResultCode RrProbe::shutdown() {
  uint64_t seq = begin_call();
  WgResultCode rc = orig_.shutdown ? orig_.shutdown() : WG_E_MISSING_OR_INVALID_API;
  log_call(seq, "shutdown", nullptr, rc, nullptr);
  std::lock_guard<std::recursive_mutex> lock(mu_);
  std::string s;
  for (const char* r : {"allocate", "free", "tag", "tag-for-frame", "requirements", "version", "get-function",
                        "constants", "new-frame-token", "evaluate", "rr-optimal", "rr-state", "rr-options",
                        "upgrade-interface", "native-interface", "set-vulkan-info"}) {
    auto it = counts_.find(r);
    s += std::string(s.empty() ? "" : " ") + r + "=" + u(it == counts_.end() ? 0 : it->second);
  }
  log_->record("summary at shutdown", s + " rr_evaluates=" + u(rr_evals_));
  if (cap_active_) {  // read back what the GPU has finished, then wait for the files
    capture_poll(rr_evals_, true);
    ProbeCaptureHost* host = gpu_ ? gpu_->capture_host() : nullptr;
    const uint64_t failed = host ? host->flush() : 0;
    log_->note("capture at shutdown: " + u(cap_written_) + " evaluate(s) written, " + u(cap_bytes_) +
               " bytes read back, " + u(failed) + " failed file write(s)");
  }
  vps_.clear();
  viewport_tags_.clear();
  frames_.clear();
  initialised_ = false;
  device_set_ = false;
  return rc;
}

WgResultCode RrProbe::is_supported(uint32_t feature, const WgAdapterInfo* adapter) {
  uint64_t seq = begin_call();
  log_chain("is-supported.adapter feature=" + u(feature), adapter ? &adapter->h : nullptr);
  WgResultCode rc;
  if (feature != kRR) {
    rc = orig_.is_supported ? orig_.is_supported(feature, adapter) : WG_E_MISSING_OR_INVALID_API;
  } else {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    if (precheck(&rc, false)) {
      // claimed as 60.8 with WrongGPU's runtime ready and the model loaded; Vulkan RR is not in scope
      rc = adapter && adapter->vk_physical_device ? WG_E_ADAPTER_NOT_SUPPORTED : WG_OK;
      log_->record("is-supported adapter", adapter ? "luid_size=" + u(adapter->luid_size) +
                                                         " vulkan=" + u(adapter->vk_physical_device != nullptr)
                                                   : "adapter=null");
    }
  }
  log_call(seq, "is-supported", &feature, rc, nullptr);
  return rc;
}

WgResultCode RrProbe::is_loaded(uint32_t feature, bool* loaded) {
  uint64_t seq = begin_call();
  WgResultCode rc;
  if (feature != kRR) {
    rc = orig_.is_loaded ? orig_.is_loaded(feature, loaded) : WG_E_MISSING_OR_INVALID_API;
  } else {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    if (precheck(&rc, false)) {
      if (!loaded) rc = WG_E_INVALID_PARAMETER;
      else { *loaded = set_loaded_given_ ? set_loaded_value_ : true; rc = WG_OK; }  // 60.8, model "loaded"
    }
  }
  log_call(seq, "is-loaded", &feature, rc, nullptr, loaded && rc == WG_OK ? " value=" + u(*loaded) : "");
  return rc;
}

WgResultCode RrProbe::set_loaded(uint32_t feature, bool loaded) {
  uint64_t seq = begin_call();
  WgResultCode rc;
  if (feature != kRR) {
    rc = orig_.set_loaded ? orig_.set_loaded(feature, loaded) : WG_E_MISSING_OR_INVALID_API;
  } else {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    if (precheck(&rc, false)) {
      set_loaded_given_ = true;
      set_loaded_value_ = loaded;
      rc = WG_OK;
    }
  }
  log_call(seq, "set-loaded", &feature, rc, nullptr, " value=" + u(loaded));
  return rc;
}

WgResultCode RrProbe::requirements(uint32_t feature, WgRequirements* out) {
  uint64_t seq = begin_call();
  log_chain("requirements.out feature=" + u(feature), out ? &out->h : nullptr);
  WgResultCode rc;
  if (feature != kRR) {
    rc = orig_.requirements ? orig_.requirements(feature, out) : WG_E_MISSING_OR_INVALID_API;
  } else {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    if (precheck(&rc, false)) rc = answer_requirements(out);  // 60.8
  }
  log_call(seq, "requirements", &feature, rc, nullptr);
  return rc;
}

WgResultCode RrProbe::version(uint32_t feature, WgFeatureVersion* out) {
  uint64_t seq = begin_call();
  log_chain("version.out feature=" + u(feature), out ? &out->h : nullptr);
  WgResultCode rc;
  if (feature != kRR) {
    rc = orig_.version ? orig_.version(feature, out) : WG_E_MISSING_OR_INVALID_API;
  } else {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    if (precheck(&rc, false)) rc = answer_version(out, api_version_, cfg_.release_version);  // 60.8
  }
  log_call(seq, "version", &feature, rc, nullptr);
  return rc;
}

void RrProbe::set_cmd_list_bracket(CmdListBracket bracket) {
  std::lock_guard<std::recursive_mutex> lock(mu_);
  bracket_ = std::move(bracket);
}

void RrProbe::set_feature_function_pointers(void* optimal, void* state, void* options) {
  fn_ptrs_[0] = optimal;
  fn_ptrs_[1] = state;
  fn_ptrs_[2] = options;
}

WgResultCode RrProbe::get_function(uint32_t feature, const char* name, void** out) {
  uint64_t seq = begin_call();
  WgResultCode rc;
  if (feature != kRR) {
    rc = orig_.get_function ? orig_.get_function(feature, name, out) : WG_E_MISSING_OR_INVALID_API;
  } else {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    if (precheck(&rc, true)) {
      rc = WG_E_INVALID_PARAMETER;  // a name that is not one of the three (Q-L4)
      if (name && out) {
        const std::string* names[3] = {&cfg_.fn_optimal, &cfg_.fn_state, &cfg_.fn_options};
        for (int i = 0; i < 3; ++i)
          if (!names[i]->empty() && *names[i] == name && fn_ptrs_[i]) {
            *out = fn_ptrs_[i];
            rc = WG_OK;
          }
      }
    }
  }
  log_call(seq, "get-function", &feature, rc, nullptr, std::string(" name=") + (name ? name : "(null)"));
  return rc;
}

WgResultCode RrProbe::allocate(void* cmd_list, uint32_t feature, const WgViewport* vp) {
  uint64_t seq = begin_call();
  log_chain("allocate.viewport feature=" + u(feature), vp ? &vp->h : nullptr);
  WgResultCode rc;
  if (feature != kRR) {
    rc = orig_.allocate ? orig_.allocate(cmd_list, feature, vp) : WG_E_MISSING_OR_INVALID_API;
  } else {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    if (precheck(&rc, true)) {
      rc = vp ? WG_OK : WG_E_INVALID_PARAMETER;
      if (vp) {
        arm_watch("allocate", seq);
        log_->record("allocate vp=" + u(vp->id), std::string("command_list=") + (cmd_list ? "given" : "null"));
      }
    }
  }
  log_call(seq, "allocate", &feature, rc, nullptr);
  return rc;
}

WgResultCode RrProbe::free_resources(uint32_t feature, const WgViewport* vp) {
  uint64_t seq = begin_call();
  WgResultCode rc;
  if (feature != kRR) {
    rc = orig_.free_resources ? orig_.free_resources(feature, vp) : WG_E_MISSING_OR_INVALID_API;
  } else {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    if (precheck(&rc, false)) rc = WG_OK;
  }
  log_call(seq, "free", &feature, rc, nullptr, vp ? " vp=" + u(vp->id) : " vp=null");
  return rc;
}

WgResultCode RrProbe::set_device(void* device) {
  uint64_t seq = begin_call();
  {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    void* native = nullptr;
    unwrap(device, &native);
    device_set_ = device != nullptr;
    if (gpu_ && native && render_api_ == 1) gpu_->set_device(native);
    log_->record("set-device", std::string("device=") + (device ? "given" : "null") +
                                   " unwrapped=" + u(native != device) + " render_api=" + u(render_api_));
  }
  WgResultCode rc = orig_.set_device ? orig_.set_device(device) : WG_E_MISSING_OR_INVALID_API;
  log_call(seq, "set-device", nullptr, rc, nullptr);
  return rc;
}

WgResultCode RrProbe::set_vulkan_info(const void* info) {
  uint64_t seq = begin_call();
  WgResultCode rc = orig_.set_vulkan_info ? orig_.set_vulkan_info(info) : WG_E_MISSING_OR_INVALID_API;
  log_call(seq, "set-vulkan-info", nullptr, rc, nullptr);
  return rc;
}

WgResultCode RrProbe::upgrade_interface(void** interface_ptr) {
  uint64_t seq = begin_call();
  WgResultCode rc = orig_.upgrade_interface ? orig_.upgrade_interface(interface_ptr) : WG_E_MISSING_OR_INVALID_API;
  log_call(seq, "upgrade-interface", nullptr, rc, nullptr);
  return rc;
}

WgResultCode RrProbe::native_interface(void* wrapped, void** native) {
  uint64_t seq = begin_call();
  WgResultCode rc = orig_.native_interface ? orig_.native_interface(wrapped, native) : WG_E_MISSING_OR_INVALID_API;
  log_call(seq, "native-interface", nullptr, rc, nullptr);
  return rc;
}

void RrProbe::note_forward(const char* export_name) {
  uint64_t seq = begin_call();
  first_call(export_name, seq);
  log_->record(std::string("call ") + export_name, "forwarded (result not seen)",
               "seq=" + u(seq) + " tid=" + u(tid_ ? tid_() : 0));
}

// ------------------------------------------------------------------------------------------------------------
// tags, constants, frame tokens

std::string RrProbe::tag_text(const WgTag& t, const ProbeTexDesc* d) const {
  std::string s = "lifetime=" + u(t.lifetime) + " extent=" + u(t.extent.top) + "," + u(t.extent.left) + "," +
                  u(t.extent.width) + "," + u(t.extent.height);
  if (!t.resource) return s + " resource=null";
  const WgResource& r = *t.resource;
  s += " kind=" + u(r.kind) + " tag_format=" + u(r.native_format) + " res=" + u(r.width) + "x" + u(r.height) +
       " state=" + hx(r.state) + " mips=" + u(r.mip_levels) + " layers=" + u(r.array_layers) +
       " res_version=" + u(r.h.version);
  if (d)
    s += " desc_format=" + u(d->format) + " desc=" + u(d->width) + "x" + u(d->height) + " desc_dim=" + u(d->dimension) +
         " desc_samples=" + u(d->samples) + " desc_flags=" + hx(d->flags) + " desc_layers=" + u(d->depth_or_array) +
         " desc_mips=" + u(d->mips);
  return s;
}

void RrProbe::store_tags(std::map<std::pair<uint32_t, uint32_t>, StoredTag>& dst, const char* via,
                         const WgViewport* vp, const WgTag* tags, uint32_t count, void* cmd_list,
                         const WgFrameToken* token) {
  uint32_t vid = vp ? vp->id : WG_VIEWPORT_INVALID;
  if (vp) log_chain(std::string(via) + ".viewport", &vp->h);
  for (uint32_t i = 0; tags && i < count; ++i) {
    const WgTag& t = tags[i];
    const std::string where = std::string(via) + " vp=" + u(vid) + " type=" + u(t.buffer_type);
    log_chain(where, &t.h);
    if (t.resource) log_chain(where + " resource", &t.resource->h);
    ProbeTexDesc d;
    bool has_desc = t.resource && t.resource->native && t.resource->kind == WG_RES_TEX2D && render_api_ == 1 &&
                    gpu_ && gpu_->describe(t.resource->native, &d);
    // 40 / 42 also name the native resource: a game that alternates two buffers shows a line per change (the
    // spec side's reflection speckle probe a)
    const bool named = t.resource && (t.buffer_type == 40u || t.buffer_type == 42u);
    log_->record(where, tag_text(t, has_desc ? &d : nullptr) +
                            (named ? " native=" + hx((uint64_t)(uintptr_t)t.resource->native) : std::string()),
                 token ? "frame_index=" + u(frame_index_of(token)) : std::string());
    auto key = std::make_pair(vid, t.buffer_type);
    dst.erase(key);  // last tag wins (60.7)
    if (!t.resource) continue;  // a null resource removes the tag
    StoredTag s;
    s.res = *t.resource;
    s.resource = t.resource->native;
    s.type = t.buffer_type;
    s.lifetime = t.lifetime;
    s.extent = t.extent;
    if (s.res.native_format == 0 && has_desc) s.res.native_format = d.format;
    if (has_desc) {
      s.desc_width = d.width;
      s.desc_height = d.height;
    }
    const bool probe_reads = t.buffer_type == WG_BT_COLOUR_IN || t.buffer_type == WG_BT_NORMALS ||
                             t.buffer_type == WG_BT_NORMAL_ROUGHNESS;
    const bool local = strcmp(via, "local-tag") == 0;  // a local tag is read during its own call
    if (t.lifetime == WG_LIFE_ONLY_DURING_TAG && probe_reads && rr_requested_ && !local) {
      // 62.1: valid only during this call: copy it on the tagging call's command list
      void* native_cmd = nullptr;
      unwrap(cmd_list, &native_cmd);
      std::string why;
      ProbeView v;
      v.resource = s.resource;
      v.format = s.res.native_format;
      v.state = s.res.state;
      uint32_t copy_state = 0;
      void* copy = native_cmd && gpu_ && s.resource ? gpu_->copy_input(native_cmd, v, t.buffer_type, &copy_state, &why)
                                                    : nullptr;
      if (!native_cmd) why = "no command list in the tagging call";
      log_->record("lifetime-0 copy " + where, copy ? "copied" : "not copied: " + why);
      s.resource = copy;  // null: the input is missing at evaluate
      s.res.state = copy_state;
    }
    dst[key] = s;
  }
}

WgResultCode RrProbe::tag(const WgViewport* vp, const WgTag* tags, uint32_t count, void* cmd_list) {
  uint64_t seq = begin_call();
  {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    store_tags(viewport_tags_, "tag", vp, tags, count, cmd_list, nullptr);
  }
  WgResultCode rc = orig_.tag ? orig_.tag(vp, tags, count, cmd_list) : WG_E_MISSING_OR_INVALID_API;
  log_call(seq, "tag", nullptr, rc, nullptr, " count=" + u(count));
  return rc;
}

RrProbe::FrameData* RrProbe::frame_for(const WgFrameToken* token, bool create) {
  if (!token) return nullptr;
  const uint64_t key = frame_key(token, cfg_.frame_by_token);
  auto it = frames_.find(key);
  if (it != frames_.end()) return it->second.get();
  if (!create) return nullptr;
  while (frames_.size() >= cfg_.max_frames) {
    auto oldest = frames_.begin();
    for (auto j = frames_.begin(); j != frames_.end(); ++j)
      if (j->second->seq < oldest->second->seq) oldest = j;
    frames_.erase(oldest);
  }
  std::unique_ptr<FrameData> f(new FrameData());
  f->seq = ++frame_seq_;
  FrameData* raw = f.get();
  frames_[key] = std::move(f);
  return raw;
}

WgResultCode RrProbe::tag_for_frame(const WgFrameToken* token, const WgViewport* vp, const WgTag* tags,
                                    uint32_t count, void* cmd_list) {
  uint64_t seq = begin_call();
  {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    if (FrameData* f = frame_for(token, true)) store_tags(f->tags, "tag-for-frame", vp, tags, count, cmd_list, token);
  }
  WgResultCode rc =
      orig_.tag_for_frame ? orig_.tag_for_frame(token, vp, tags, count, cmd_list) : WG_E_MISSING_OR_INVALID_API;
  log_call(seq, "tag-for-frame", nullptr, rc, token, " count=" + u(count));
  return rc;
}

WgResultCode RrProbe::constants(const WgConstants* c, const WgFrameToken* token, const WgViewport* vp) {
  uint64_t seq = begin_call();
  {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    uint32_t vid = vp ? vp->id : WG_VIEWPORT_INVALID;
    log_chain("constants", c ? &c->h : nullptr);
    if (c) {
      WgConstants copy;
      memset(&copy, 0, sizeof copy);
      size_t n = struct_version(c->h) >= 2 ? sizeof(WgConstants) : 451;  // v1 ends after +450 (60.6.4)
      memcpy(&copy, c, n);
      if (FrameData* f = frame_for(token, true)) f->consts[vid] = copy;
      log_->record("constants vp=" + u(vid),
                   "camera_motion_included=" + u(c->camera_motion_included) + " mv_jittered=" + u(c->mv_jittered) +
                       " depth_inverted=" + u(c->depth_inverted) + " mv_3d=" + u(c->mv_3d) + " mv_dilated=" +
                       u(c->mv_dilated) + " orthographic=" + u(c->orthographic) + " reset=" + u(c->reset) +
                       " mv_scale=" + fl(c->mv_scale[0]) + "," + fl(c->mv_scale[1]) + " version=" + u(c->h.version),
                   token ? "frame_index=" + u(frame_index_of(token)) : std::string("token=null"));
    }
  }
  WgResultCode rc = orig_.constants ? orig_.constants(c, token, vp) : WG_E_MISSING_OR_INVALID_API;
  log_call(seq, "constants", nullptr, rc, token);
  return rc;
}

WgResultCode RrProbe::new_frame_token(WgFrameToken** out, const uint32_t* frame_index) {
  uint64_t seq = begin_call();
  WgResultCode rc = orig_.new_frame_token ? orig_.new_frame_token(out, frame_index) : WG_E_MISSING_OR_INVALID_API;
  const WgFrameToken* tok = rc == WG_OK && out ? *out : nullptr;
  if (tok) {
    log_->next_frame();
    std::lock_guard<std::recursive_mutex> lock(mu_);
    if (!token_checked_) {  // A10.1: the type GUID at +16 confirms the layout (first frame only)
      token_checked_ = true;
      const uint8_t* b = reinterpret_cast<const uint8_t*>(tok) + 16;
      char hexs[40];
      for (int i = 0; i < 16; ++i) snprintf(hexs + 2 * i, 3, "%02x", b[i]);
      WgGuid g;
      memcpy(g.b, b, 16);
      uint64_t ver;
      memcpy(&ver, reinterpret_cast<const uint8_t*>(tok) + 32, 8);
      log_->record("frame-token first", std::string("bytes_at_16=") + hexs + " as_guid=" + guid_text(g) +
                                            " matches_frame_token_guid=" +
                                            u(cfg_.has_frame_token && guid_eq(g, cfg_.frame_token)) +
                                            " version_at_32=" + u(ver));
    }
    log_->record("new-frame-token", std::string("index_argument=") + (frame_index ? "given" : "null"),
                 "index_argument_value=" + (frame_index ? u(*frame_index) : std::string("-")) +
                     " index_from_table=" + u(frame_index_of(tok)));
  }
  log_call(seq, "new-frame-token", nullptr, rc, tok);
  return rc;
}

// ------------------------------------------------------------------------------------------------------------
// feature functions

WgResultCode RrProbe::rr_optimal(const WgRrOptions* options, WgRrOptimal* out) {
  uint64_t seq = begin_call();
  WgResultCode rc;
  {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    log_chain("rr-optimal.options", options ? &options->h : nullptr);
    log_chain("rr-optimal.out", out ? &out->h : nullptr);
    if (precheck(&rc, false)) rc = answer_optimal(cfg_.api_minor, options, out);  // the real 61.5 rule
    if (options) {
      std::string s = "rc=" + rc_text(rc);
      if (rc == WG_OK && out)
        s += " optimal=" + u(out->optimal_width) + "x" + u(out->optimal_height) + " min=" + u(out->min_width) + "x" +
             u(out->min_height) + " max=" + u(out->max_width) + "x" + u(out->max_height) +
             " sharpness=" + fl(out->optimal_sharpness);
      log_->record("rr-optimal mode=" + u(options->mode) + " in=" + u(options->output_width) + "x" +
                       u(options->output_height) + " version=" + u(options->h.version),
                   s);
    }
  }
  log_call(seq, "rr-optimal", nullptr, rc, nullptr);
  return rc;
}

WgResultCode RrProbe::rr_state(const WgViewport* vp, WgRrState* out) {
  uint64_t seq = begin_call();
  WgResultCode rc;
  {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    log_chain("rr-state.out", out ? &out->h : nullptr);
    if (precheck(&rc, false)) {
      rc = vp && out ? WG_OK : WG_E_INVALID_PARAMETER;
      if (rc == WG_OK) out->estimated_vram = 0;  // the probe holds no feature instance
    }
  }
  log_call(seq, "rr-state", nullptr, rc, nullptr, vp ? " vp=" + u(vp->id) : " vp=null");
  return rc;
}

WgResultCode RrProbe::rr_options(const WgViewport* vp, const WgRrOptions* o) {
  uint64_t seq = begin_call();
  WgResultCode rc;
  {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    log_chain("rr-options.options", o ? &o->h : nullptr);
    if (precheck(&rc, false)) {
      rc = vp && o ? WG_OK : WG_E_INVALID_PARAMETER;  // the probe accepts every value; the release answer is logged
      if (rc == WG_OK) {
        uint64_t ver = struct_version(o->h);
        WgResultCode release_rc = (o->mode > WG_RR_MODE_NATIVE_AA || o->output_width == 0 || o->output_height == 0 ||
                                   o->output_width == WG_SIZE_NOT_SET || o->output_height == WG_SIZE_NOT_SET)
                                      ? WG_E_INVALID_PARAMETER
                                      : WG_OK;  // 61.3 (Q-L3)
        std::string s = "mode=" + u(o->mode) + " output=" + u(o->output_width) + "x" + u(o->output_height) +
                        " sharpness=" + fl(o->sharpness) + " pre_exposure=" + fl(o->pre_exposure) +
                        " exposure_scale=" + fl(o->exposure_scale) + " hdr=" + u(o->colour_is_hdr) +
                        " invert_x=" + u(o->indicator_invert_x) + " invert_y=" + u(o->indicator_invert_y) +
                        " normal_roughness_mode=" + u(o->normal_roughness_packing) + " version=" + u(o->h.version);
        if (ver >= 2) s += " alpha_upscaling=" + u(o->alpha_upscaling);
        if (ver >= 3) {
          s += " presets=";
          for (int i = 0; i < 6; ++i) s += (i ? "," : "") + u(o->presets[i]);
        }
        s += " release_rc=" + rc_text(release_rc);
        log_->record("rr-options vp=" + u(vp->id), s);
        Viewport& v = viewport(vp->id);
        const bool resized = v.has_options && (v.opts.output_width != o->output_width ||
                                               v.opts.output_height != o->output_height);
        WgRrOptions c = default_options();
        size_t end = ver >= 3 ? WG_RR_OPTIONS_V3_END : ver == 2 ? WG_RR_OPTIONS_V2_END : WG_RR_OPTIONS_V1_END;
        memcpy(reinterpret_cast<uint8_t*>(&c) + sizeof(WgHeader), reinterpret_cast<const uint8_t*>(o) + sizeof(WgHeader),
               end - sizeof(WgHeader));
        c.h.next = nullptr;
        c.h.version = ver;
        v.opts = c;
        v.opts_version = ver;
        v.has_options = true;
        arm_watch(resized ? "rr-options (output resize)" : "rr-options", seq);
      }
    }
  }
  log_call(seq, "rr-options", nullptr, rc, nullptr, vp ? " vp=" + u(vp->id) : " vp=null");
  return rc;
}

// ------------------------------------------------------------------------------------------------------------
// evaluate

const StoredTag* RrProbe::lookup(uint32_t type, uint32_t vp, const std::vector<StoredTag>& locals,
                                 const FrameData* frame) const {
  // 61.4: local tag, frame-scoped tag (2.8.0 and 2.14.1: tag-for-frame), viewport tag
  for (const StoredTag& t : locals)
    if (t.type == type) return t.resource ? &t : nullptr;
  auto key = std::make_pair(vp, type);
  if (frame && cfg_.api_minor >= 8) {
    auto it = frame->tags.find(key);
    if (it != frame->tags.end()) return it->second.resource ? &it->second : nullptr;
  }
  auto it = viewport_tags_.find(key);
  if (it != viewport_tags_.end()) return it->second.resource ? &it->second : nullptr;
  return nullptr;
}

void RrProbe::read_patches(uint32_t now_eval) {
  for (size_t i = 0; i < pending_patches_.size();) {
    PendingPatch& p = pending_patches_[i];
    if (now_eval < p.eval + cfg_.patch_read_delay) { ++i; continue; }
    float m[4] = {0, 0, 0, 0};
    bool done = gpu_ && gpu_->read_patch(p.slot, p.marker, m);
    if (!done && now_eval < p.eval + cfg_.patch_read_delay + 120) { ++i; continue; }
    std::string s = "eval=" + u(p.eval) + " type=" + u(p.type) + " format=" + u(p.format) + " origin=" + u(p.px) +
                    "," + u(p.py) + " size=16x16";
    if (done) {
      s += " mean=" + fl(m[0]) + "," + fl(m[1]) + "," + fl(m[2]) + "," + fl(m[3]) + " world_to_view=";
      for (int k = 0; k < 16; ++k) s += (k ? "," : "") + fl(p.w2v[k]);
    } else {
      s += " not read back";
    }
    log_->record("normal-patch sample=" + u(p.slot), s);
    pending_patches_.erase(pending_patches_.begin() + i);
  }
}

WgResultCode RrProbe::evaluate(uint32_t feature, const WgFrameToken* token, const WgHeader* const* inputs,
                               uint32_t count, void* cmd_list) {
  uint64_t seq = begin_call();
  if (feature != kRR) {
    for (uint32_t i = 0; inputs && i < count; ++i)
      log_chain("evaluate.input[" + u(i) + "] feature=" + u(feature), inputs[i]);
    WgResultCode rc =
        orig_.evaluate ? orig_.evaluate(feature, token, inputs, count, cmd_list) : WG_E_MISSING_OR_INVALID_API;
    log_call(seq, "evaluate", &feature, rc, token);
    return rc;
  }
  WgResultCode rc = WG_OK;
  {
    std::lock_guard<std::recursive_mutex> lock(mu_);
    if (!precheck(&rc, true)) {
      log_call(seq, "evaluate", &feature, rc, token);
      return rc;
    }
    // inputs: the viewport handle and local tags, found by GUID (61.4 step 1); every chain is logged
    const WgViewport* vph = nullptr;
    std::vector<StoredTag> locals;
    for (uint32_t i = 0; inputs && i < count; ++i) {
      log_chain("evaluate.input[" + u(i) + "] feature=1001", inputs[i]);
      for (const WgHeader* p = inputs[i]; p; p = p->next) {
        if (guid_eq(p->type, cfg_.ids.viewport)) {
          if (!vph) vph = reinterpret_cast<const WgViewport*>(p);
        } else if (guid_eq(p->type, cfg_.ids.tag)) {
          const WgTag* t = reinterpret_cast<const WgTag*>(p);
          std::map<std::pair<uint32_t, uint32_t>, StoredTag> one;
          store_tags(one, "local-tag", vph, t, 1, cmd_list, token);
          locals.erase(std::remove_if(locals.begin(), locals.end(),
                                      [&](const StoredTag& x) { return x.type == t->buffer_type; }),
                       locals.end());
          StoredTag s;
          s.type = t->buffer_type;
          if (!one.empty()) s = one.begin()->second;  // empty: a local tag with a null resource (absent)
          locals.push_back(s);
        }
      }
    }
    void* native_cmd = nullptr;
    unwrap(cmd_list, &native_cmd);
    // clean/rr-re9-pt: one begin/end pair around everything the probe records on the game's list in this evaluate
    struct Bracket {
      const CmdListBracket& b;
      void* cl;
      ~Bracket() {
        if (cl) b.end(cl);
      }
    } bracket{bracket_, nullptr};
    if (native_cmd && bracket_.begin && bracket_.end) {
      bracket_.begin(native_cmd);
      bracket.cl = native_cmd;
    }
    if (!vph) {
      rc = WG_E_MISSING_INPUT_PARAMETER;
      log_->record("evaluate", "no viewport in the inputs (20)");
    } else {
      ++rr_evals_;
      const uint32_t n = rr_evals_;
      Viewport& v = viewport(vph->id);
      const FrameData* frame = frame_for(token, false);
      const WgConstants* consts = nullptr;
      if (frame) {
        auto it = frame->consts.find(v.id);
        if (it != frame->consts.end()) consts = &it->second;
      }
      const uint32_t packing = v.opts.normal_roughness_packing;
      const StoredTag* colour = lookup(WG_BT_COLOUR_IN, v.id, locals, frame);
      const StoredTag* output = lookup(WG_BT_COLOUR_OUT, v.id, locals, frame);
      const StoredTag* nrm = lookup(packing == 1 ? WG_BT_NORMAL_ROUGHNESS : WG_BT_NORMALS, v.id, locals, frame);
      const StoredTag* rgh = packing == 1 ? nullptr : lookup(WG_BT_ROUGHNESS, v.id, locals, frame);
      uint32_t cx = 0, cy = 0, rw = 0, rh = 0;
      if (colour) rect_of(*colour, &cx, &cy, &rw, &rh);
      // output rectangle: the output tag's origin, the options' output size (else its extent / the resource)
      uint32_t ox = 0, oy = 0, ow = 0, oh = 0;
      if (output) {
        rect_of(*output, &ox, &oy, &ow, &oh);
        const bool opt_size = v.has_options && v.opts.output_width && v.opts.output_height &&
                              v.opts.output_width != WG_SIZE_NOT_SET && v.opts.output_height != WG_SIZE_NOT_SET;
        if (opt_size) { ow = v.opts.output_width; oh = v.opts.output_height; }
        if (ox < res_w(*output) && ow > res_w(*output) - ox) ow = res_w(*output) - ox;
        if (oy < res_h(*output) && oh > res_h(*output) - oy) oh = res_h(*output) - oy;
      }
      log_->record("evaluate vp=" + u(v.id),
                   "render=" + u(rw) + "x" + u(rh) + " output=" + u(v.opts.output_width) + "x" +
                       u(v.opts.output_height) + " output_rect=" + u(ox) + "," + u(oy) + "," + u(ow) + "," + u(oh) +
                       " mode=" + u(v.opts.mode) + " options=" + u(v.has_options) + " constants=" + u(consts != nullptr) +
                       " normal_roughness_mode=" + u(packing) + " normal_type=" +
                       u(packing == 1 ? WG_BT_NORMAL_ROUGHNESS : WG_BT_NORMALS) +
                       " normal_format=" + (nrm ? u(nrm->res.native_format) : std::string("absent")) +
                       " roughness_format=" + (packing == 1 ? std::string("in-normal") :
                                               rgh ? u(rgh->res.native_format) : std::string("absent")) +
                       " colour_format=" + (colour ? u(colour->res.native_format) : std::string("absent")) +
                       " output_format=" + (output ? u(output->res.native_format) : std::string("absent")) +
                       " locals=" + u(locals.size()) + " command_list=" + (native_cmd ? "given" : "null"),
                   "frame_index=" + (token ? u(frame_index_of(token)) : std::string("-")));
      // jitter, camera-view-to-clip [2][0] / [2][1] (row-major elements 8 and 9) and the previous frame's pair
      if (consts) {
        const float a = consts->view_to_clip[8], b = consts->view_to_clip[9];
        log_->record("camera vp=" + u(v.id),
                     "v2c20=" + fl(a) + " v2c21=" + fl(b) + " prev_v2c20=" +
                         (v.has_prev ? fl(v.prev_v2c20) : std::string("-")) + " prev_v2c21=" +
                         (v.has_prev ? fl(v.prev_v2c21) : std::string("-")),
                     "jitter=" + fl(consts->jitter[0]) + "," + fl(consts->jitter[1]) + " prev_jitter=" +
                         (v.has_prev ? fl(v.prev_jitter[0]) + "," + fl(v.prev_jitter[1]) : std::string("-")) +
                         " frame_index=" + (token ? u(frame_index_of(token)) : std::string("-")));
        v.has_prev = true;
        v.prev_v2c20 = a;
        v.prev_v2c21 = b;
        v.prev_jitter[0] = consts->jitter[0];
        v.prev_jitter[1] = consts->jitter[1];
      }
      // the colour-input extent of the first two evaluates after <rr-options> / <allocate> / a resize
      if (colour && v.last_render_w && (rw != v.last_render_w || rh != v.last_render_h)) arm_watch("resize", seq);
      if (watch_left_ > 0 && colour) {
        log_->record("colour-extent-watch",
                     "after=" + watch_event_ + " evaluate=" + u(3 - watch_left_) + "/2 extent=" +
                         u(colour->extent.top) + "," + u(colour->extent.left) + "," + u(colour->extent.width) + "," +
                         u(colour->extent.height) + " res=" + u(colour->res.width) + "x" + u(colour->res.height) +
                         " render=" + u(rw) + "x" + u(rh) + " previous_render=" + u(v.last_render_w) + "x" +
                         u(v.last_render_h),
                     "event_seq=" + u(watch_seq_) + " seq=" + u(seq));
        --watch_left_;
      }
      if (colour) {
        v.last_render_w = rw;
        v.last_render_h = rh;
      }
      // normal patch samples (30 evaluates), read back later
      if (nrm && gpu_ && gpu_->has_device() && native_cmd && passes_allowed() && patches_taken_ < cfg_.patch_count &&
          n >= cfg_.patch_first && (n - cfg_.patch_first) % cfg_.patch_every == 0) {
        uint32_t nx, ny, nw, nh;
        rect_of(*nrm, &nx, &ny, &nw, &nh);
        ProbeView pv;
        pv.resource = nrm->resource;
        pv.format = nrm->res.native_format;
        pv.state = nrm->res.state;
        pv.x = nx; pv.y = ny; pv.w = nw; pv.h = nh;
        uint32_t px = nx + (nw > 16 ? (nw - 16) / 2 : 0), py = ny + (nh > 16 ? (nh - 16) / 2 : 0);
        std::string why;
        PendingPatch pp;
        pp.slot = patches_taken_;
        pp.marker = 0x5000u + n;
        pp.eval = n;
        pp.type = nrm->type;
        pp.format = nrm->res.native_format;
        pp.px = px;
        pp.py = py;
        memcpy(pp.w2v, v.opts.world_to_view, sizeof pp.w2v);
        WgResultCode prc = gpu_->patch_mean(native_cmd, pv, px, py, pp.slot, pp.marker, &why);
        if (prc == WG_OK) pending_patches_.push_back(pp);
        else log_->record("normal-patch sample=" + u(pp.slot), "rc=" + rc_text(prc) + " " + why);
        ++patches_taken_;
      }
      read_patches(n);
      // capture option: copies of every tagged input on the evaluate's list (copies and barriers only)
      capture_settings_once();
      capture_poll(n, false);
      if (cap_active_) capture_take(n, v, frame, locals, consts, token, native_cmd, colour, output);
      // the upscale: do not fail the frame; read-only on everything but the output
      std::string why;
      bool skipped = false;
      if (!colour || !output) {
        rc = WG_E_MISSING_INPUT_PARAMETER;
        why = std::string("missing") + (colour ? "" : " colour(3)") + (output ? "" : " output(4)");
      } else {
        ProbeView src, dst;
        src.resource = colour->resource;
        src.format = colour->res.native_format;
        src.state = colour->res.state;
        src.x = cx; src.y = cy; src.w = rw; src.h = rh;
        dst.resource = output->resource;
        dst.format = output->res.native_format;
        dst.state = output->res.state;
        dst.x = ox; dst.y = oy; dst.w = ow; dst.h = oh;
        const uint32_t sc = format_class(src.format, false), dc = format_class(dst.format, false);
        if (src.format && sc == WG_FMT_UNSUPPORTED) {
          rc = WG_E_INVALID_PARAMETER;  // 62.5
          why = "unsupported colour format " + u(src.format);
        } else if (dst.format && dc == WG_FMT_UNSUPPORTED) {
          rc = WG_E_INVALID_PARAMETER;
          why = "unsupported output format " + u(dst.format);
        } else if (!rw || !rh || !ow || !oh) {
          rc = WG_E_INVALID_PARAMETER;  // 61.4 step 3 zero size
          why = "zero render or output size";
        } else if (!gpu_ || !gpu_->has_device()) {
          rc = WG_E_COMPUTE_FAILED;
          why = "no D3D12 device";
        } else if (!native_cmd) {
          rc = WG_E_COMPUTE_FAILED;
          why = "no command list";
        } else if (!passes_allowed()) {
          skipped = true;  // hook mode: rc stays 0, nothing bound on the game's list
          why = "skipped: the game's command-list state restore is not known (preference bit 0 = 0 or <init> unseen)";
        } else {
          rc = gpu_->upscale(native_cmd, src, dst, &why);
        }
      }
      log_->record("upscale vp=" + u(v.id), "rc=" + rc_text(rc) + (why.empty() ? "" : " " + why) + " src=" + u(cx) +
                                                "," + u(cy) + "," + u(rw) + "x" + u(rh) + " dst=" + u(ox) + "," +
                                                u(oy) + "," + u(ow) + "x" + u(oh));
      if (rc == WG_OK && !skipped) {
        any_complete_ = true;
        last_complete_ms_ = clock_ ? clock_() : 0;
      }
    }
  }
  log_call(seq, "evaluate", &feature, rc, token);
  return rc;
}

// ------------------------------------------------------------------------------------------------------------
// capture option (CAPTURE.md)

namespace {

std::string trim(const std::string& s) {
  size_t a = 0, b = s.size();
  while (a < b && (s[a] == ' ' || s[a] == '\t' || s[a] == '\r')) ++a;
  while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\r')) --b;
  return s.substr(a, b - a);
}

bool parse_u64(const std::string& v, uint64_t* out) {
  if (v.empty() || v.size() > 19) return false;
  for (char c : v)
    if (c < '0' || c > '9') return false;
  *out = strtoull(v.c_str(), nullptr, 10);
  return true;
}

// JSON numbers: finite values as %.9g, the rest as the tokens Python's json module reads.
std::string jn(float v) {
  if (isnan(v)) return "NaN";
  if (isinf(v)) return v > 0 ? "Infinity" : "-Infinity";
  return fl(v);
}

std::string jarr(const float* p, int n) {
  std::string s = "[";
  for (int i = 0; i < n; ++i) s += (i ? "," : "") + jn(p[i]);
  return s + "]";
}

std::string jstr(const std::string& t) {
  std::string s = "\"";
  for (char c : t) {
    if (c == '"' || c == '\\') {
      s += '\\';
      s += c;
    } else if ((unsigned char)c < 0x20) {
      s += ' ';
    } else {
      s += c;
    }
  }
  return s + "\"";
}

void put32(std::vector<uint8_t>& b, uint32_t v) {
  for (int i = 0; i < 4; ++i) b.push_back((uint8_t)(v >> (8 * i)));
}

const char kCapMagic[8] = {'W', 'G', 'R', 'R', 'C', 'A', 'P', '1'};
constexpr uint32_t kCapHeaderBytes = 64;

}  // namespace

CaptureSettings parse_capture_settings(const std::string& text_in, std::string* notes) {
  CaptureSettings s;
  std::string text = text_in;
  if (text.size() >= 3 && (uint8_t)text[0] == 0xef && (uint8_t)text[1] == 0xbb && (uint8_t)text[2] == 0xbf)
    text.erase(0, 3);  // UTF-8 byte-order mark
  size_t pos = 0;
  uint32_t line_no = 0;
  auto note = [&](const std::string& n) {
    if (notes) *notes += "line " + u(line_no) + ": " + n + "\n";
  };
  while (pos < text.size()) {
    size_t e = text.find('\n', pos);
    if (e == std::string::npos) e = text.size();
    std::string line = text.substr(pos, e - pos);
    pos = e + 1;
    ++line_no;
    size_t c = line.find_first_of(";#");
    if (c != std::string::npos) line.erase(c);
    line = trim(line);
    if (line.empty() || line[0] == '[') continue;
    size_t eq = line.find('=');
    if (eq == std::string::npos) {
      note("no '=': ignored");
      continue;
    }
    std::string key = trim(line.substr(0, eq)), val = trim(line.substr(eq + 1));
    for (char& k : key) k = (char)tolower((unsigned char)k);
    uint64_t v = 0;
    const bool num = parse_u64(val, &v);
    auto bad = [&]() { note(key + "=" + val + ": not a number in range, default kept"); };
    if (key == "capture") {
      if (num && v <= 1) s.enabled = v == 1;
      else bad();
    } else if (key == "capture_start") {
      if (num && v >= 1 && v <= 0xffffffffull) s.start = (uint32_t)v;
      else bad();
    } else if (key == "capture_count") {
      if (num && v >= 1 && v <= 100000) s.count = (uint32_t)v;
      else bad();
    } else if (key == "capture_every") {
      if (num && v >= 1 && v <= 1000000) s.every = (uint32_t)v;
      else bad();
    } else if (key == "capture_max_mb") {
      if (num && v >= 1 && v <= (1ull << 20)) s.max_bytes = v << 20;
      else bad();
    } else if (key == "capture_on_file") {
      if (num && v <= 1) s.on_file = v == 1;
      else bad();
    } else if (key == "capture_shots") {
      if (num && v >= 1 && v <= 50) s.shots = (uint32_t)v;
      else bad();
    } else if (key == "capture_trigger") {
      if (!val.empty()) s.trigger = val;
      else bad();
    } else if (key == "capture_dir") {
      if (!val.empty()) s.folder = val;
      else bad();
    } else if (key == "capture_read_delay") {
      if (num && v >= 1 && v <= 1000) s.read_delay = (uint32_t)v;
      else bad();
    } else {
      note("unknown key " + key + ": ignored");
    }
  }
  return s;
}

void RrProbe::capture_settings_once() {
  if (cap_loaded_) return;
  cap_loaded_ = true;
  ProbeCaptureHost* host = gpu_ ? gpu_->capture_host() : nullptr;
  std::string text, notes;
  const bool have = host && host->read_settings(&text);
  if (have) cap_ = parse_capture_settings(text, &notes);
  for (size_t a = 0; a < notes.size();) {
    size_t e = notes.find('\n', a);
    if (e == std::string::npos) e = notes.size();
    log_->note("capture settings: " + notes.substr(a, e - a));
    a = e + 1;
  }
  if (!cap_.enabled) {
    log_->note(std::string("capture: off (") +
               (!host ? "not supported" : have ? "capture=1 not set in the probe settings file" : "no probe settings file") +
               ")");
    return;
  }
  std::string where;
  if (!cap_.folder.empty()) host->set_folder(cap_.folder);
  if (!host->open(&where)) {
    log_->note("capture: capture=1, but the capture folder could not be prepared (" + where + "); capture off");
    return;
  }
  cap_active_ = true;
  log_->note("capture: on, start=" + u(cap_.start) + " count=" + u(cap_.count) + " every=" + u(cap_.every) +
             " max_mb=" + u(cap_.max_bytes >> 20) + " read_delay=" + u(cap_.read_delay) + (cap_.on_file ? " on_file=1 trigger=" + cap_.trigger : std::string()) +
             " folder=" + where);
}

void RrProbe::capture_take(uint32_t n, const Viewport& v, const FrameData* frame, const std::vector<StoredTag>& locals,
                           const WgConstants* consts, const WgFrameToken* token, void* native_cmd,
                           const StoredTag* colour, const StoredTag* output) {
  if (cap_.on_file && cap_fired_ && cap_taken_ >= cap_.count && cap_shots_done_ + 1 < cap_.shots) {
    ++cap_shots_done_;  // the window is complete: wait for the next trigger file
    cap_fired_ = false;
    cap_taken_ = 0;
    cap_done_logged_ = false;
    log_->note("capture: window " + u(cap_shots_done_) + " of " + u(cap_.shots) + " complete at eval=" + u(n) +
               ", waiting for the trigger file again");
  }
  if (cap_.on_file && !cap_fired_) {
    if ((n & 15u) != 0) return;
    FILE* trig = fopen(cap_.trigger.c_str(), "rb");
    if (!trig) return;
    fclose(trig);
    remove(cap_.trigger.c_str());  // one trigger file, one capture window
    cap_fired_ = true;
    cap_.start = n + 1;
    log_->note("capture: trigger file seen at eval=" + u(n) + ", window starts at eval=" + u(cap_.start));
    return;
  }
  if (cap_taken_ >= cap_.count || n < cap_.start || (n - cap_.start) % cap_.every != 0) return;
  ++cap_taken_;  // this evaluate is in the window, whether or not the copy can be made
  const std::string at = "capture eval=" + u(n) + ": ";
  if (!gpu_ || !gpu_->has_device() || !native_cmd) {
    log_->note(at + "not taken: " + (!native_cmd ? "no command list" : "no D3D12 device"));
    return;
  }
  // Every tagged input this evaluate sees (61.4: local, frame-scoped, viewport tag), the outputs left out.
  std::set<uint32_t> types;
  for (const auto& kv : viewport_tags_)
    if (kv.first.first == v.id) types.insert(kv.first.second);
  if (frame && cfg_.api_minor >= 8)
    for (const auto& kv : frame->tags)
      if (kv.first.first == v.id) types.insert(kv.first.second);
  for (const StoredTag& t : locals) types.insert(t.type);
  std::vector<CaptureSource> src;
  PendingCapture p;
  for (uint32_t type : types) {
    if (type == WG_BT_COLOUR_OUT || type == WG_BT_OUTPUT_ALPHA) continue;
    const StoredTag* t = lookup(type, v.id, locals, frame);
    if (!t || !t->resource) continue;  // removed by a null resource
    CaptureSource s;
    s.type = type;
    s.view.resource = t->resource;
    s.view.format = t->res.native_format;
    s.view.state = t->res.state;
    rect_of(*t, &s.view.x, &s.view.y, &s.view.w, &s.view.h);
    src.push_back(s);
    CaptureInputMeta m;
    m.type = type;
    m.tag_format = t->res.native_format;
    m.state = t->res.state;
    m.lifetime = t->lifetime;
    m.kind = t->res.kind;
    m.extent = t->extent;
    m.tag_width = t->res.width;
    m.tag_height = t->res.height;
    p.metas.push_back(m);
  }
  if (src.empty()) {
    log_->note(at + "not taken: no tagged input");
    return;
  }
  const uint32_t marker = 0x43500000u + (n & 0xfffffu);
  const uint64_t budget = cap_.max_bytes > cap_bytes_ ? cap_.max_bytes - cap_bytes_ : 0;
  uint64_t bytes = 0;
  std::string why;
  WgResultCode rc = gpu_->capture_copy(native_cmd, n, marker, src, budget, &bytes, &p.skipped, &why);
  if (rc != WG_OK) {
    log_->note(at + "not taken: " + why);
    if (why.compare(0, 8, "size cap") == 0) {
      cap_taken_ = cap_.count;  // nothing more fits
      log_->note("capture: size cap of " + u(cap_.max_bytes >> 20) + " MB reached after " + u(cap_bytes_) +
                 " bytes; no further evaluates captured");
    }
    return;
  }
  cap_bytes_ += bytes;
  p.id = n;
  p.marker = marker;
  p.eval = n;
  p.bytes = bytes;
  // frame.json without the inputs list (added when the copies are read back)
  uint32_t cx = 0, cy = 0, rw = 0, rh = 0;
  if (colour) rect_of(*colour, &cx, &cy, &rw, &rh);
  std::string j = "{\"format\":\"wronggpu-rr-capture\",\"version\":1,\"evaluate\":" + u(n) + ",\"frame_index\":" +
                  (token ? u(frame_index_of(token)) : std::string("null")) + ",\"viewport\":" + u(v.id) +
                  ",\"api_minor\":" + u(cfg_.api_minor) + ",\"preference_flags\":" + u(pref_flags_) +
                  ",\"render\":{\"x\":" + u(cx) + ",\"y\":" + u(cy) + ",\"width\":" + u(rw) + ",\"height\":" + u(rh) +
                  "}";
  if (output) {
    uint32_t ox, oy, ow, oh;
    rect_of(*output, &ox, &oy, &ow, &oh);
    const bool opt_size = v.has_options && v.opts.output_width && v.opts.output_height &&
                          v.opts.output_width != WG_SIZE_NOT_SET && v.opts.output_height != WG_SIZE_NOT_SET;
    if (opt_size) {
      ow = v.opts.output_width;
      oh = v.opts.output_height;
    }
    ProbeTexDesc d;
    const bool has_d = render_api_ == 1 && output->resource && gpu_->describe(output->resource, &d);
    j += ",\"output\":{\"type\":4,\"tag_format\":" + u(output->res.native_format) + ",\"format\":" +
         u(has_d ? d.format : output->res.native_format) + ",\"width\":" + u(has_d ? d.width : output->res.width) +
         ",\"height\":" + u(has_d ? d.height : output->res.height) + ",\"rect\":[" + u(ox) + "," + u(oy) + "," +
         u(ow) + "," + u(oh) + "],\"state\":" + u(output->res.state) + "}";
  } else {
    j += ",\"output\":null";
  }
  const WgRrOptions& o = v.opts;
  j += ",\"options\":{\"given\":" + std::string(v.has_options ? "true" : "false") + ",\"version\":" +
       u(v.opts_version) + ",\"mode\":" + u(o.mode) + ",\"output_width\":" + u(o.output_width) +
       ",\"output_height\":" + u(o.output_height) + ",\"sharpness\":" + jn(o.sharpness) + ",\"pre_exposure\":" +
       jn(o.pre_exposure) + ",\"exposure_scale\":" + jn(o.exposure_scale) + ",\"colour_is_hdr\":" +
       u(o.colour_is_hdr) + ",\"indicator_invert_x\":" + u(o.indicator_invert_x) + ",\"indicator_invert_y\":" +
       u(o.indicator_invert_y) + ",\"normal_roughness_packing\":" + u(o.normal_roughness_packing) +
       ",\"world_to_view\":" + jarr(o.world_to_view, 16) + ",\"view_to_world\":" + jarr(o.view_to_world, 16) +
       ",\"alpha_upscaling\":" + u(o.alpha_upscaling) + ",\"presets\":[";
  for (int i = 0; i < 6; ++i) j += (i ? "," : "") + u(o.presets[i]);
  j += "]}";
  if (consts) {
    const WgConstants& c = *consts;
    j += ",\"constants\":{\"version\":" + u(struct_version(c.h)) + ",\"view_to_clip\":" + jarr(c.view_to_clip, 16) +
         ",\"clip_to_view\":" + jarr(c.clip_to_view, 16) + ",\"clip_to_lens_clip\":" + jarr(c.clip_to_lens_clip, 16) +
         ",\"clip_to_prev_clip\":" + jarr(c.clip_to_prev_clip, 16) + ",\"prev_clip_to_clip\":" +
         jarr(c.prev_clip_to_clip, 16) + ",\"jitter\":" + jarr(c.jitter, 2) + ",\"mv_scale\":" + jarr(c.mv_scale, 2) +
         ",\"pinhole_offset\":" + jarr(c.pinhole_offset, 2) + ",\"camera_pos\":" + jarr(c.camera_pos, 3) +
         ",\"camera_up\":" + jarr(c.camera_up, 3) + ",\"camera_right\":" + jarr(c.camera_right, 3) +
         ",\"camera_fwd\":" + jarr(c.camera_fwd, 3) + ",\"near_plane\":" + jn(c.near_plane) + ",\"far_plane\":" +
         jn(c.far_plane) + ",\"fov_y\":" + jn(c.fov_y) + ",\"aspect\":" + jn(c.aspect) + ",\"mv_invalid_value\":" +
         jn(c.mv_invalid_value) + ",\"depth_inverted\":" + u(c.depth_inverted) + ",\"camera_motion_included\":" +
         u(c.camera_motion_included) + ",\"mv_3d\":" + u(c.mv_3d) + ",\"reset\":" + u(c.reset) +
         ",\"orthographic\":" + u(c.orthographic) + ",\"mv_dilated\":" + u(c.mv_dilated) + ",\"mv_jittered\":" +
         u(c.mv_jittered) + ",\"min_depth_separation\":" + jn(c.min_depth_separation) + "}";
    // the camera as the release proxy hands it to the runtime (61.4: constants V2C, options W2V, jitter, ms)
    j += ",\"camera\":{\"jitter\":" + jarr(c.jitter, 2) + ",\"view_to_clip\":" + jarr(c.view_to_clip, 16) +
         ",\"world_to_view\":" + jarr(o.world_to_view, 16) + ",\"view_to_world\":" + jarr(o.view_to_world, 16) +
         ",\"mv_scale\":" + jarr(c.mv_scale, 2) + ",\"reset\":" + u(c.reset) + "}";
  } else {
    j += ",\"constants\":null,\"camera\":null";
  }
  p.json_head = j;
  std::string kinds;
  for (const CaptureSource& s : src) kinds += (kinds.empty() ? "" : ",") + u(s.type);
  std::string sk;
  for (const std::string& x : p.skipped) sk += "; not copied: " + x;
  log_->note(at + "copied " + u(src.size() - p.skipped.size()) + " of " + u(src.size()) + " input(s) (types " + kinds +
             "), " + u(bytes) + " bytes of readback memory" + sk);
  cap_pending_.push_back(std::move(p));
}

void RrProbe::capture_poll(uint32_t now, bool final_pass) {
  ProbeCaptureHost* host = gpu_ ? gpu_->capture_host() : nullptr;
  for (size_t i = 0; i < cap_pending_.size();) {
    PendingCapture& p = cap_pending_[i];
    if (!final_pass && now < p.eval + cap_.read_delay) {
      ++i;
      continue;
    }
    std::vector<CaptureImage> imgs;
    const int r = gpu_ ? gpu_->capture_read(p.id, p.marker, &imgs) : -1;
    if (r == 0 && !final_pass && now < p.eval + cap_.read_delay + cap_.give_up) {
      ++i;
      continue;
    }
    const std::string at = "capture eval=" + u(p.eval) + ": ";
    if (r == 1 && host) {
      char dir[32];
      snprintf(dir, sizeof dir, "frame_%06u", p.eval);
      std::string inputs;
      uint64_t written = 0;
      bool ok = true;
      for (const CaptureImage& im : imgs) {
        const CaptureInputMeta* m = nullptr;
        for (const CaptureInputMeta& x : p.metas)
          if (x.type == im.type) m = &x;
        if (!m) continue;
        char name[32];
        snprintf(name, sizeof name, "input_%02u.raw", im.type);
        std::vector<uint8_t> b;
        b.reserve(kCapHeaderBytes + im.data.size());
        b.insert(b.end(), kCapMagic, kCapMagic + 8);
        put32(b, kCapHeaderBytes);
        put32(b, im.type);
        put32(b, im.format);
        put32(b, im.width);
        put32(b, im.height);
        put32(b, im.row_pitch);
        put32(b, m->tag_format);
        put32(b, m->state);
        put32(b, m->extent.left);
        put32(b, m->extent.top);
        put32(b, m->extent.width);
        put32(b, m->extent.height);
        put32(b, m->lifetime);
        put32(b, 0);
        b.insert(b.end(), im.data.begin(), im.data.end());
        written += b.size();
        ok = host->write(std::string(dir) + "/" + name, std::move(b)) && ok;
        inputs += std::string(inputs.empty() ? "" : ",") + "{\"type\":" + u(im.type) + ",\"file\":" + jstr(name) +
                  ",\"format\":" + u(im.format) + ",\"width\":" + u(im.width) + ",\"height\":" + u(im.height) +
                  ",\"row_pitch\":" + u(im.row_pitch) + ",\"tag_format\":" + u(m->tag_format) + ",\"state\":" +
                  u(m->state) + ",\"lifetime\":" + u(m->lifetime) + ",\"kind\":" + u(m->kind) +
                  ",\"extent\":{\"left\":" + u(m->extent.left) + ",\"top\":" + u(m->extent.top) + ",\"width\":" +
                  u(m->extent.width) + ",\"height\":" + u(m->extent.height) + "},\"tag_size\":[" + u(m->tag_width) +
                  "," + u(m->tag_height) + "]}";
      }
      std::string sk;
      for (const std::string& x : p.skipped) sk += std::string(sk.empty() ? "" : ",") + jstr(x);
      const std::string json = p.json_head + ",\"inputs\":[" + inputs + "],\"not_copied\":[" + sk + "]}\n";
      written += json.size();
      ok = host->write(std::string(dir) + "/frame.json", std::vector<uint8_t>(json.begin(), json.end())) && ok;
      if (ok) ++cap_written_;
      log_->note(at + (ok ? "written " : "write refused for part of ") + std::string(dir) + " (" + u(imgs.size()) +
                 " input file(s) + frame.json, " + u(written) + " bytes)");
    } else if (r == 1) {
      log_->note(at + "read back, but there is no capture folder");
    } else if (r == 0) {
      if (gpu_) gpu_->capture_abandon(p.id);
      log_->note(at + "not read back (completion marker not seen " +
                 (final_pass ? std::string("at shutdown") : "after " + u(now - p.eval) + " evaluates") + "); dropped");
    } else {
      log_->note(at + "readback memory lost (device changed); dropped");
    }
    cap_pending_.erase(cap_pending_.begin() + i);
  }
  if (cap_active_ && !cap_done_logged_ && cap_taken_ >= cap_.count && cap_pending_.empty()) {
    cap_done_logged_ = true;
    log_->note("capture: done, " + u(cap_written_) + " of " + u(cap_taken_) + " evaluate(s) written, " +
               u(cap_bytes_) + " bytes read back");
  }
}

uint32_t RrProbe::status_word() const {
  std::lock_guard<std::recursive_mutex> lock(mu_);
  uint32_t s = 1u | 2u;  // claimed as 60.8: runtime ready, model loaded
  if (any_complete_ && clock_ && clock_() - last_complete_ms_ < 1000) s |= 4u;
  return s;
}

}  // namespace wgrr
