// SPDX-License-Identifier: MIT
// WrongGPU RR for games that call the NGX interface directly (spec page 66; ngx_rr.h). Section numbers refer to spec page 66
// (NGX RR provider), draft 2, 2026-11-04, as received from the spec team; the parameter keys are its annex 66.A (allowed in this
// folder by the leak-scan allow-list, owner OK 2026-10-23).
#include "ngx_rr.h"

#include <stdio.h>
#include <string.h>

#include "formats.h"
#include "optimal.h"

namespace wgrr {

namespace {

// ---- create keys (66.2)
const char* const kWidth = "Width";
const char* const kHeight = "Height";
const char* const kOutWidth = "OutWidth";
const char* const kOutHeight = "OutHeight";
const char* const kQuality = "PerfQualityValue";
const char* const kCreateFlags = "DLSS.Feature.Create.Flags";
const char* const kSubrects = "DLSS.Enable.Output.Subrects";
const char* const kDenoiseMode = "DLSS.Denoise.Mode";
const char* const kRoughnessMode = "DLSS.Roughness.Mode";
const char* const kHwDepth = "DLSS.Use.HW.Depth";
// preset hints in the order of WgRrCreate.presets (DLAA, quality, balanced, performance, ultra performance, ultra quality)
const char* const kPresetHint[6] = {
    "RayReconstruction.Hint.Render.Preset.DLAA", "RayReconstruction.Hint.Render.Preset.Quality",
    "RayReconstruction.Hint.Render.Preset.Balanced", "RayReconstruction.Hint.Render.Preset.Performance",
    "RayReconstruction.Hint.Render.Preset.UltraPerformance", "RayReconstruction.Hint.Render.Preset.UltraQuality"};

// create flag bits (66.2)
constexpr uint32_t kFlagHdr = 1u << 0, kFlagMvLowRes = 1u << 1, kFlagMvJittered = 1u << 2, kFlagDepthInverted = 1u << 3,
                   kFlagSharpen = 1u << 5, kFlagAlpha = 1u << 7;

// ---- evaluate keys (66.3)
const char* const kRenderSubW = "DLSS.Render.Subrect.Dimensions.Width";
const char* const kRenderSubH = "DLSS.Render.Subrect.Dimensions.Height";
const char* const kMvScaleX = "MV.Scale.X";
const char* const kMvScaleY = "MV.Scale.Y";
const char* const kJitterX = "Jitter.Offset.X";
const char* const kJitterY = "Jitter.Offset.Y";
const char* const kReset = "Reset";
const char* const kWorldToView = "WorldToViewMatrix";
const char* const kViewToClip = "ViewToClipMatrix";
const char* const kLayerMvecs = "DLSS.TransparencyLayerMvecs";

// One image input: the key, its subrect-base key prefix (null = none: origin 0, 0), and what it becomes.
struct Input {
  const char* key;
  const char* base;  // "<base>.Subrect.Base.X" / ".Y"
};
const Input kColour = {"Color", "DLSS.Input.Color"};
const Input kOutput = {"Output", "DLSS.Output"};
const Input kDepth = {"Depth", "DLSS.Input.Depth"};
const Input kMotion = {"MotionVectors", "DLSS.Input.MV"};
const Input kDiffuse = {"DLSS.Input.DiffuseAlbedo", "DLSS.Input.DiffuseAlbedo"};
const Input kSpecular = {"DLSS.Input.SpecularAlbedo", "DLSS.Input.SpecularAlbedo"};
const Input kNormals = {"GBuffer.Normals", "DLSS.Input.Normals"};
const Input kRoughness = {"GBuffer.Roughness", "DLSS.Input.Roughness"};
const Input kSpecularMv = {"GBuffer.SpecularMvec", nullptr};
const Input kHitDist = {"DLSSD.SpecularHitDistance", "DLSSD.SpecularHitDistance"};
const Input kDisocclusion = {"DLSS.DisocclusionMask", "DLSS.DisocclusionMask"};
const Input kBeforeFog = {"DLSSD.ColorBeforeFog", "DLSSD.ColorBeforeFog"};
const Input kSssGuide = {"DLSSD.ScreenSpaceSubsurfaceScatteringGuide", "DLSSD.ScreenSpaceSubsurfaceScatteringGuide"};
const Input kDofGuide = {"DLSSD.DepthOfFieldGuide", "DLSSD.DepthOfFieldGuide"};
const Input kBeforeTransparency = {"DLSSD.ColorBeforeTransparency", "DLSSD.ColorBeforeTransparency"};
const Input kLayer = {"DLSS.TransparencyLayer", "DLSS.TransparencyLayer"};
const Input kBeforeParticles = {"DLSSD.ColorBeforeParticles", nullptr};  // no subrect key of its own (54b.0.2)
const Input kOpacity = {"DLSS.TransparencyLayerOpacity", "DLSS.TransparencyLayerOpacity"};
const Input kInputAlpha = {"DLSSD.Alpha", "DLSSD.Alpha"};
const Input kOutputAlpha = {"DLSSD.OutputAlpha", "DLSSD.OutputAlpha"};

const char* const kRoleName[WG_ROLE_COUNT] = {
    "colour", "output", "depth", "motion", "diffuse albedo", "specular albedo", "normals", "roughness",
    "specular motion", "hit distance", "disocclusion", "colour guide", "depth-of-field guide", "before transparency",
    "layer", "opacity", "input alpha", "output alpha"};

constexpr uint32_t kMagic = 0x57475252u;  // live instances carry it (a stale context is refused, not dereferenced)

std::string u32s(uint32_t v) { return std::to_string(v); }

// Integer keys: the game may have set a key with the other integer type than the one read here. The provider does not
// rely on the host converting: when the typed getter has no value, the other integer getter is asked and its value
// taken when it fits (a negative int for an unsigned key, or an unsigned value above INT32_MAX for a signed key, counts
// as not set).
bool get_u(const WgHostNgxParams* p, const char* key, uint32_t* v) {
  if (!p) return false;
  if (p->get_uint && p->get_uint(p->block, key, v)) return true;
  int32_t i = 0;
  if (p->get_int && p->get_int(p->block, key, &i) && i >= 0) {
    *v = (uint32_t)i;
    return true;
  }
  return false;
}
bool get_i(const WgHostNgxParams* p, const char* key, int32_t* v) {
  if (!p) return false;
  if (p->get_int && p->get_int(p->block, key, v)) return true;
  uint32_t u = 0;
  if (p->get_uint && p->get_uint(p->block, key, &u) && u <= 0x7FFFFFFFu) {
    *v = (int32_t)u;
    return true;
  }
  return false;
}
bool get_f(const WgHostNgxParams* p, const char* key, float* v) {
  return p && p->get_float && p->get_float(p->block, key, v);
}
void* get_res(const WgHostNgxParams* p, const char* key) {
  void* r = nullptr;
  if (!p || !p->get_resource || !p->get_resource(p->block, key, &r)) return nullptr;
  return r;
}
uint32_t base_of(const WgHostNgxParams* p, const Input& in, char axis) {
  if (!in.base) return 0;
  std::string k = std::string(in.base) + ".Subrect.Base." + axis;
  uint32_t v = 0;
  return get_u(p, k.c_str(), &v) ? v : 0;
}
bool get_matrix(const WgHostNgxParams* p, const char* key, float out[16]) {
  void* m = nullptr;
  if (!p || !p->get_pointer || !p->get_pointer(p->block, key, &m) || !m) return false;
  memcpy(out, m, 16 * sizeof(float));  // copied at call time (66.3)
  return true;
}

// The provider's command-list bracket open on this thread (Scope), so that the fault guard can close it.
thread_local const CmdListBracket* t_open_bracket = nullptr;
thread_local void* t_open_cl = nullptr;

Settings& default_settings() {
  static Settings s;  // the defaults of 60.9 (a host without the plugin's settings: tests only)
  return s;
}

}  // namespace

struct NgxRrProvider::Instance {
  uint32_t magic = kMagic;
  uint32_t id = 0;
  void* handle = nullptr;  // the runtime's instance
  WgRrCreate desc{};
  Lifecycle lc;
  bool subrects = false;
  uint32_t quality = 0, flags = 0;
  uint64_t evaluates = 0;  // calls that reached the runtime
  // logged once per feature
  bool logged_missing = false, logged_range = false, logged_format = false, logged_fit = false,
       logged_layer_mv = false, logged_matrices = false, logged_camera_fallback = false;
  Instance(uint32_t rw, uint32_t rh) : lc(rw, rh) {}
};

// One begin/end pair around the work recorded on the game's list (every return path closes it).
class NgxRrProvider::Scope {
 public:
  Scope(const CmdListBracket& b, void* cl) : b_(b), cl_(cl) {
    if (cl_ && b_.begin && b_.end) {
      b_.begin(cl_);
      open_ = true;
      t_open_bracket = &b_;
      t_open_cl = cl_;
    }
  }
  ~Scope() {
    // closed here unless the fault guard closed it already (close_open_bracket_after_fault)
    if (open_ && t_open_bracket == &b_ && t_open_cl == cl_) {
      t_open_bracket = nullptr;
      t_open_cl = nullptr;
      b_.end(cl_);
    }
  }
  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;

 private:
  const CmdListBracket& b_;
  void* cl_;
  bool open_ = false;
};

NgxRrProvider::NgxRrProvider(NgxRrDeps deps) : deps_(std::move(deps)) {}

NgxRrProvider::~NgxRrProvider() { release_all(); }

void NgxRrProvider::log(const std::string& s) const {
  if (log_lines_.size() < 4096) log_lines_.push_back(s);
  if (deps_.log) deps_.log(s);
}

uint32_t NgxRrProvider::log_count(const std::string& needle) const {
  std::lock_guard<std::recursive_mutex> lock(mu_);
  uint32_t n = 0;
  for (const auto& l : log_lines_) n += l.find(needle) != std::string::npos;
  return n;
}

bool NgxRrProvider::usable(const WgRrRuntimeV1* rt) const {
  if (!rt || !rt->status || !rt->create_instance || !rt->evaluate || !rt->destroy_instance) return false;
  const uint32_t s = rt->status(rt->ctx);
  return (s & (WG_RR_RT_READY | WG_RR_RT_MODEL_LOADED)) == (WG_RR_RT_READY | WG_RR_RT_MODEL_LOADED);
}

NgxRrProvider::Instance* NgxRrProvider::find(void* context) const {
  Instance* in = static_cast<Instance*>(context);
  if (!in || live_.find(in) == live_.end() || in->magic != kMagic) return nullptr;
  return in;
}

void NgxRrProvider::set_cmd_list_bracket(CmdListBracket bracket) {
  std::lock_guard<std::recursive_mutex> lock(mu_);
  bracket_ = std::move(bracket);
}

void NgxRrProvider::close_open_bracket_after_fault() {
  const CmdListBracket* b = t_open_bracket;
  void* cl = t_open_cl;
  t_open_bracket = nullptr;
  t_open_cl = nullptr;
  if (b && cl && b->end) b->end(cl);
}

size_t NgxRrProvider::live() const { return live_count_.load(); }
uint64_t NgxRrProvider::evaluates() const {
  std::lock_guard<std::recursive_mutex> lock(mu_);
  return evaluates_;
}
const WgRrCreate* NgxRrProvider::create_desc(void* context) const {
  std::lock_guard<std::recursive_mutex> lock(mu_);
  Instance* in = find(context);
  return in ? &in->desc : nullptr;
}
const Lifecycle* NgxRrProvider::lifecycle(void* context) const {
  std::lock_guard<std::recursive_mutex> lock(mu_);
  Instance* in = find(context);
  return in ? &in->lc : nullptr;
}

// ------------------------------------------------------------------------------------------------------------
// create (66.1 <create-feature> feature 13, 66.2)

bool NgxRrProvider::create(void* device, void* command_list, const WgHostNgxParams* params, void** context) {
  std::lock_guard<std::recursive_mutex> lock(mu_);
  auto refuse = [&](const std::string& why) {
    if (refusal_lines_ < 40) {
      ++refusal_lines_;
      log("NGX: RR feature refused (the game keeps its own path): " + why);
    }
    return false;
  };
  if (!context) return refuse("no context pointer from the host");
  *context = nullptr;
  if (!params || !params->get_uint || !params->get_int) return refuse("no parameter access from the host");
  if (deps_.other_route_busy && deps_.other_route_busy()) {
    if (!logged_busy_) {
      logged_busy_ = true;
      log("NGX: the game-interface route already runs WrongGPU RR in this game; the NGX route stays out "
          "(one route per game, never two evaluates)");
    }
    return refuse("the game-interface route holds the RR feature");
  }
  Settings* settings = deps_.settings ? deps_.settings() : nullptr;
  const SettingsSnapshot snap = (settings ? *settings : default_settings()).snapshot();
  if (!snap.on(kSetWrongGpuRr)) {  // 60.9: "WrongGPU RR" off at creation: the feature is not created
    if (!logged_setting_off_) {
      logged_setting_off_ = true;
      log("NGX: setting \"WrongGPU RR\" is off; the RR feature is not created");
    }
    return false;
  }
  const WgRrRuntimeV1* rt = deps_.runtime ? deps_.runtime() : nullptr;
  if (!rt) return refuse("WrongGPU's RR runtime is not loaded");
  if (!device && deps_.device_of && command_list) device = deps_.device_of(command_list);
  if (device && device != device_set_ && rt->set_device) {
    const uint32_t rc = rt->set_device(rt->ctx, device);
    if (rc != WG_OK) return refuse("the runtime refused the game's device (" + u32s(rc) + ")");
    device_set_ = device;
  }
  if (!device && !device_set_) return refuse("no D3D12 device");
  if (!usable(rt)) return refuse("WrongGPU's runtime or model is not available");

  // keys
  uint32_t rw = 0, rh = 0, ow = 0, oh = 0;
  get_u(params, kWidth, &rw);
  get_u(params, kHeight, &rh);
  get_u(params, kOutWidth, &ow);
  get_u(params, kOutHeight, &oh);
  if (!rw || !rh || !ow || !oh)
    return refuse("render or output size missing (" + u32s(rw) + "x" + u32s(rh) + " -> " + u32s(ow) + "x" + u32s(oh) +
                  ")");
  int32_t quality = 2, flags = 0, denoise = 1, roughness = 0, hw_depth = 1, subrects = 0;
  const bool has_quality = get_i(params, kQuality, &quality);
  get_i(params, kCreateFlags, &flags);
  const bool has_denoise = get_i(params, kDenoiseMode, &denoise);
  get_i(params, kRoughnessMode, &roughness);
  const bool has_hw_depth = get_i(params, kHwDepth, &hw_depth);
  get_i(params, kSubrects, &subrects);
  if (has_denoise && denoise != 1) return refuse("denoise mode " + std::to_string(denoise) + " (only 1, unified)");
  const uint32_t f = (uint32_t)flags;
  if (!(f & kFlagMvLowRes))  // 62.3: vectors at output resolution are not supported
    return refuse("motion vectors at output resolution are not supported");
  if (roughness != 0 && roughness != 1) return refuse("roughness mode " + std::to_string(roughness));
  if (quality < 0 || quality > 5) quality = 2;
  if (rw > ow || rh > oh || rw < min_render_size(ow) || rh < min_render_size(oh))
    return refuse("render size " + u32s(rw) + "x" + u32s(rh) + " outside 1/3..1 of output " + u32s(ow) + "x" +
                  u32s(oh));

  WgRrCreate d;
  memset(&d, 0, sizeof d);
  d.size = sizeof d;
  d.api_minor = kNgxApiMinor;
  d.device = device ? device : device_set_;
  d.output_width = ow;
  d.output_height = oh;
  d.render_width = rw;  // the maximum when the game uses dynamic resolution
  d.render_height = rh;
  d.mode = (uint32_t)quality + 1u;  // the game-interface numbering (66.2)
  d.packing = (uint32_t)roughness;
  for (int i = 0; i < 6; ++i) {
    uint32_t v = 0;
    if (get_u(params, kPresetHint[i], &v)) d.presets[i] = v;  // accepted and ignored (we run F)
  }
  d.colour_hdr = (f & kFlagHdr) ? 1u : 0u;
  d.depth_inverted = (f & kFlagDepthInverted) ? 1u : 0u;
  d.mv_jittered = (f & kFlagMvJittered) ? 1u : 0u;
  d.mv_output_resolution = 0;
  d.hw_depth = hw_depth != 0 ? 1u : 0u;
  d.alpha_upscaling = (f & kFlagAlpha) ? 1u : 0u;
  d.sharpen_request = (f & kFlagSharpen) ? 1u : 0u;
  d.latched_settings = snap.on(kSetWrongGpuRr) ? 1u : 0u;
  void* handle = nullptr;
  const uint32_t rc = rt->create_instance(rt->ctx, &d, &handle);
  if (rc != WG_OK || !handle) return refuse("the runtime could not create the instance (" + u32s(rc) + ")");

  Instance* in = new Instance(rw, rh);
  in->id = next_id_++;
  in->handle = handle;
  in->desc = d;
  in->subrects = subrects != 0;
  in->quality = (uint32_t)quality;
  in->flags = f;
  live_.insert(in);
  live_count_.store(live_.size());
  *context = in;
  char b[420];
  snprintf(b, sizeof b,
           "NGX: RR feature %u created: render %ux%u -> output %ux%u, quality %d (mode %u), flags 0x%x (HDR %u, "
           "jittered vectors %u, depth inverted %u, alpha %u), %s depth%s, %s normals/roughness, output subrects %u%s",
           in->id, rw, rh, ow, oh, quality, d.mode, f, d.colour_hdr, d.mv_jittered, d.depth_inverted, d.alpha_upscaling,
           d.hw_depth ? "hardware" : "linear", has_hw_depth ? "" : " (key absent: hardware taken)",
           d.packing ? "packed" : "separate", subrects ? 1u : 0u,
           has_quality ? (has_denoise ? "" : "; denoise mode absent (unified taken)") : "; quality absent (2 taken)");
  log(b);
  return true;
}

// ------------------------------------------------------------------------------------------------------------
// evaluate (66.3, 66.4)

bool NgxRrProvider::evaluate(void* context, void* command_list, const WgHostNgxParams* params) {
  std::lock_guard<std::recursive_mutex> lock(mu_);
  Instance* inp = find(context);
  if (!inp) {
    if (refusal_lines_ < 40) {
      ++refusal_lines_;
      log("NGX: evaluate for an unknown or released RR feature refused");
    }
    return false;
  }
  Instance& in = *inp;
  const WgRrRuntimeV1* rt = deps_.runtime ? deps_.runtime() : nullptr;
  if (!rt || !rt->evaluate || !params || !command_list) return false;
  Settings* settings = deps_.settings ? deps_.settings() : nullptr;
  const SettingsSnapshot snap = (settings ? *settings : default_settings()).snapshot();  // one per call (60.9)
  const WgRrCreate& cd = in.desc;

  // required inputs
  struct Use {
    WgRrRole role;
    uint32_t type;
    void* res;
    uint32_t ox, oy;
  };
  std::vector<Use> uses;
  std::string missing;
  auto need = [&](WgRrRole role, uint32_t type, const Input& key) {
    void* r = get_res(params, key.key);
    if (!r) {
      missing += missing.empty() ? kRoleName[role] : std::string(", ") + kRoleName[role];
      return;
    }
    uses.push_back({role, type, r, base_of(params, key, 'X'), base_of(params, key, 'Y')});
  };
  auto opt = [&](WgRrRole role, uint32_t type, const Input& key) {
    void* r = get_res(params, key.key);
    if (r) uses.push_back({role, type, r, base_of(params, key, 'X'), base_of(params, key, 'Y')});
    return r != nullptr;
  };
  need(WG_ROLE_COLOUR, WG_BT_COLOUR_IN, kColour);
  need(WG_ROLE_OUTPUT, WG_BT_COLOUR_OUT, kOutput);
  need(WG_ROLE_DEPTH, cd.hw_depth ? WG_BT_DEPTH : WG_BT_LINEAR_DEPTH, kDepth);
  need(WG_ROLE_MOTION, WG_BT_MOTION, kMotion);
  need(WG_ROLE_DIFFUSE_ALBEDO, WG_BT_DIFFUSE_ALBEDO, kDiffuse);
  need(WG_ROLE_SPECULAR_ALBEDO, WG_BT_SPECULAR_ALBEDO, kSpecular);
  if (cd.packing == 1) {
    need(WG_ROLE_NORMALS, WG_BT_NORMAL_ROUGHNESS, kNormals);  // roughness in normals.w
  } else {
    need(WG_ROLE_NORMALS, WG_BT_NORMALS, kNormals);
    need(WG_ROLE_ROUGHNESS, WG_BT_ROUGHNESS, kRoughness);
  }
  if (!missing.empty()) {
    if (!in.logged_missing) {
      in.logged_missing = true;
      log("NGX: RR feature " + u32s(in.id) + ": evaluate without " + missing + " refused (logged once)");
    }
    return false;
  }
  // optional inputs (routing as 62.2)
  opt(WG_ROLE_SPECULAR_MOTION, WG_BT_SPECULAR_MOTION, kSpecularMv);
  opt(WG_ROLE_SPECULAR_HIT_DIST, WG_BT_SPECULAR_HIT_DISTANCE, kHitDist);
  opt(WG_ROLE_DISOCCLUSION, WG_BT_DISOCCLUSION, kDisocclusion);
  if (!opt(WG_ROLE_G1, WG_BT_COLOUR_BEFORE_FOG, kBeforeFog)) opt(WG_ROLE_G1, WG_BT_SSS_GUIDE, kSssGuide);
  opt(WG_ROLE_DOF_GUIDE, WG_BT_DOF_GUIDE, kDofGuide);
  opt(WG_ROLE_BEFORE_TRANSPARENCY, WG_BT_COLOUR_BEFORE_TRANSPARENCY, kBeforeTransparency);
  void* layer_res = get_res(params, kLayer.key);
  const uint32_t layer_ox = layer_res ? base_of(params, kLayer, 'X') : 0u;
  const uint32_t layer_oy = layer_res ? base_of(params, kLayer, 'Y') : 0u;
  const bool layer = opt(WG_ROLE_LAYER, WG_BT_COLOUR_BEFORE_PARTICLES, kBeforeParticles) ||
                     opt(WG_ROLE_LAYER, WG_BT_TRANSPARENCY_LAYER, kLayer);
  if (layer) opt(WG_ROLE_OPACITY, WG_BT_TRANSPARENCY_OPACITY, kOpacity);
  if (layer && !in.logged_layer_mv && get_res(params, kLayerMvecs)) {
    in.logged_layer_mv = true;
    log("NGX: RR feature " + u32s(in.id) + ": the game supplies transparency-layer motion vectors; not consumed "
        "(the no-layer-motion branch, 54b.3; logged once)");
  }
  if (cd.alpha_upscaling) {
    opt(WG_ROLE_INPUT_ALPHA, WG_BT_INPUT_ALPHA, kInputAlpha);
    opt(WG_ROLE_OUTPUT_ALPHA, WG_BT_OUTPUT_ALPHA, kOutputAlpha);
  }

  // sizes: the render subrect (dynamic resolution), else the create-time render size; the create-time output size
  uint32_t rw = 0, rh = 0;
  get_u(params, kRenderSubW, &rw);
  get_u(params, kRenderSubH, &rh);
  if (!rw || !rh) {
    rw = cd.render_width;
    rh = cd.render_height;
  }
  const uint32_t ow = cd.output_width, oh = cd.output_height;
  if (rw < min_render_size(ow) || rw > ow || rh < min_render_size(oh) || rh > oh) {
    if (!in.logged_range) {
      in.logged_range = true;
      log("NGX: RR feature " + u32s(in.id) + ": render size " + u32s(rw) + "x" + u32s(rh) + " outside 1/3..1 of output " +
          u32s(ow) + "x" + u32s(oh) + " refused (logged once)");
    }
    in.lc.request_reset();
    return false;
  }

  // formats (62.5) and native sizes of every input this call uses
  WgRrFrame fr;
  memset(&fr, 0, sizeof fr);
  for (const Use& u : uses) {
    uint32_t fmt = 0, samples = 1, w = 0, h = 0;
    if (rt->describe_resource) rt->describe_resource(rt->ctx, u.res, &fmt, &samples);
    if (rt->size >= WG_RR_RUNTIME_V1_LIVE_SIZE && rt->describe_resource_size)
      rt->describe_resource_size(rt->ctx, u.res, &w, &h);
    const uint32_t cls = samples > 1 ? (uint32_t)WG_FMT_UNSUPPORTED : format_class(fmt, u.role == WG_ROLE_DEPTH);
    if (cls == WG_FMT_UNSUPPORTED) {
      if (!in.logged_format) {
        in.logged_format = true;
        log("NGX: RR feature " + u32s(in.id) + ": unsupported format " + u32s(fmt) + (samples > 1 ? " (multisampled)" : "") +
            " for the " + kRoleName[u.role] + " input refused (logged once)");
      }
      return false;
    }
    const bool out_role = u.role == WG_ROLE_OUTPUT || u.role == WG_ROLE_OUTPUT_ALPHA;
    const uint32_t ew = out_role ? ow : rw, eh = out_role ? oh : rh;
    const uint32_t ox = u.role == WG_ROLE_OUTPUT && !in.subrects ? 0u : u.ox;
    const uint32_t oy = u.role == WG_ROLE_OUTPUT && !in.subrects ? 0u : u.oy;
    if (w && h && (ox + ew > w || oy + eh > h) && (u.role == WG_ROLE_COLOUR || u.role == WG_ROLE_OUTPUT)) {
      if (!in.logged_fit) {
        in.logged_fit = true;
        log("NGX: RR feature " + u32s(in.id) + ": the " + kRoleName[u.role] + " rectangle " + u32s(ox) + "," + u32s(oy) +
            " " + u32s(ew) + "x" + u32s(eh) + " does not fit its " + u32s(w) + "x" + u32s(h) + " resource; refused "
            "(logged once)");
      }
      return false;
    }
    WgRrImage& im = fr.images[u.role];
    im.resource = u.res;
    im.buffer_type = u.type;
    im.format = fmt;
    im.format_class = cls;
    im.state = out_role ? kNgxOutputState : kNgxInputState;  // 66.4
    im.width = w;
    im.height = h;
    im.origin_x = ox;
    im.origin_y = oy;
    im.extent_width = ew;
    im.extent_height = eh;
    im.lifetime = WG_LIFE_UNTIL_EVALUATE;  // valid during the call
    fr.attached |= 1u << u.role;
  }

  // camera, jitter, motion-vector scale, reset
  CameraPair cam;
  memset(&cam, 0, sizeof cam);
  const bool has_w2v = get_matrix(params, kWorldToView, cam.w2v);
  const bool has_v2c = get_matrix(params, kViewToClip, cam.v2c);
  if ((!has_w2v || !has_v2c) && !in.logged_matrices) {
    in.logged_matrices = true;
    log("NGX: RR feature " + u32s(in.id) + ": camera matrices not supplied (" + (has_w2v ? "" : "world-to-view ") +
        (has_v2c ? "" : "view-to-clip ") + "missing: zeros handed on, logged once)");
  }
  float jitter[2] = {0.0f, 0.0f}, ms[2] = {1.0f, 1.0f};
  get_f(params, kJitterX, &jitter[0]);  // unchanged: render pixels, same sign (66.3)
  get_f(params, kJitterY, &jitter[1]);
  get_f(params, kMvScaleX, &ms[0]);  // taken as given; 0 counts as 1
  get_f(params, kMvScaleY, &ms[1]);
  if (ms[0] == 0.0f) ms[0] = 1.0f;
  if (ms[1] == 0.0f) ms[1] = 1.0f;
  int32_t reset = 0;
  get_i(params, kReset, &reset);

  FramePlan plan = in.lc.begin(rw, rh, jitter, cam, reset != 0, snap.on(kSetChangeFrameReset));
  fr.size = sizeof fr;
  fr.api_minor = kNgxApiMinor;
  fr.command_list = command_list;
  fr.render_width = rw;
  fr.render_height = rh;
  fr.output_width = ow;
  fr.output_height = oh;
  fr.jitter[0] = plan.jitter[0];
  fr.jitter[1] = plan.jitter[1];
  fr.prev_jitter[0] = plan.prev_jitter[0];
  fr.prev_jitter[1] = plan.prev_jitter[1];
  fr.mv_scale[0] = ms[0];
  fr.mv_scale[1] = ms[1];
  fr.reset = plan.reset;
  fr.parity = plan.parity;
  fr.counter = plan.counter;
  fr.rebuild_render_items = plan.rebuild;
  fr.depth_is_linear = cd.hw_depth ? 0u : 1u;
  fr.settings = (snap.on(kSetSpecularMotionFromV) ? WG_RR_SET_SPECULAR_MOTION_FROM_V : 0u) |
                (snap.on(kSetBoundedStores) ? WG_RR_SET_BOUNDED_STORES : 0u) |
                (snap.on(kSetHitDistanceFilter) ? WG_RR_SET_HIT_DISTANCE_FILTER : 0u);
  memcpy(fr.view_to_clip, plan.current.v2c, sizeof fr.view_to_clip);
  memcpy(fr.world_to_view, plan.current.w2v, sizeof fr.world_to_view);
  memcpy(fr.prev_view_to_clip, plan.effective_prev.v2c, sizeof fr.prev_view_to_clip);
  memcpy(fr.prev_world_to_view, plan.effective_prev.w2v, sizeof fr.prev_world_to_view);
  if (layer_res) {  // the layer origin is always the transparency layer's; (0, 0) without it (62.2, 54b.3)
    fr.layer_origin_x = layer_ox;
    fr.layer_origin_y = layer_oy;
  }

  WgRrFrameResult res;
  memset(&res, 0, sizeof res);
  res.size = sizeof res;
  Scope scope(bracket_, command_list);  // the runtime and the pass-through record on the game's list
  uint32_t rc = rt->evaluate(rt->ctx, in.handle, &fr, &res);
  ++evaluates_;
  ++in.evaluates;
  if (res.camera_fallback_used && !in.logged_camera_fallback) {
    in.logged_camera_fallback = true;
    log("NGX: RR feature " + u32s(in.id) + ": camera matrices unusable; specular motion = motion vectors (logged once)");
  }
  if (rc != WG_OK) {
    // the same fail-safe as the game-interface route: nothing of ours on the list yet -> colour passed through, success
    uint32_t prc = WG_E_INVALID_PARAMETER;
    if (res.passes_started == 0 && rt->size >= WG_RR_RUNTIME_V1_LIVE_SIZE && rt->passthrough)
      prc = rt->passthrough(rt->ctx, &fr);
    in.lc.fail(plan, res.input_jitter_consumed != 0, prc == WG_OK || snap.on(kSetResetAfterFailedEvaluate));
    if ((rc != last_eval_rc_ || prc != last_pass_rc_) && eval_rc_lines_ < 40) {
      ++eval_rc_lines_;
      log("NGX: RR feature " + u32s(in.id) + ": evaluate " + u32s((uint32_t)in.evaluates) + ": runtime returned " +
          u32s(rc) + (prc == WG_OK ? "; colour passed through, success reported, reset on the next evaluate"
                                   : "; no pass-through (" + u32s(prc) + "), failure reported to the game"));
    }
    last_eval_rc_ = rc;
    last_pass_rc_ = prc;
    if (prc == WG_OK) {
      ++passthroughs_;
      return true;
    }
    return false;
  }
  if ((in.evaluates == 1 || last_eval_rc_ != WG_OK) && eval_rc_lines_ < 40) {
    ++eval_rc_lines_;
    std::string inputs;
    for (int r = 0; r < WG_ROLE_COUNT; ++r)
      if (fr.attached & (1u << r)) inputs += (inputs.empty() ? "" : ", ") + std::string(kRoleName[r]);
    char b[200];
    snprintf(b, sizeof b, ": render %ux%u -> output %ux%u, jitter (%.4f, %.4f), vector scale (%g, %g), reset %u", rw, rh,
             ow, oh, fr.jitter[0], fr.jitter[1], fr.mv_scale[0], fr.mv_scale[1], fr.reset);
    log("NGX: RR feature " + u32s(in.id) + (in.evaluates == 1 ? " first evaluate" : " evaluate " + u32s((uint32_t)in.evaluates)) +
        b + "; inputs: " + inputs + "; runtime returned 0 (" + u32s((uint32_t)passthroughs_) +
        " frame(s) passed through so far)");
  }
  last_eval_rc_ = WG_OK;
  last_pass_rc_ = WG_OK;
  in.lc.complete(plan);
  if (deps_.frame_done) deps_.frame_done();
  return true;
}

// ------------------------------------------------------------------------------------------------------------
// release (66.1 <release-feature>): the runtime defers GPU destruction past the frames in flight

void NgxRrProvider::release(void* context) {
  std::lock_guard<std::recursive_mutex> lock(mu_);
  Instance* in = find(context);
  if (!in) {
    log("NGX: release of an unknown RR feature ignored");
    return;
  }
  const WgRrRuntimeV1* rt = deps_.runtime ? deps_.runtime() : nullptr;
  if (rt && rt->destroy_instance && in->handle) rt->destroy_instance(rt->ctx, in->handle);
  log("NGX: RR feature " + u32s(in->id) + " released after " + u32s((uint32_t)in->evaluates) +
      " evaluate(s); its GPU memory is freed after the frames in flight");
  live_.erase(in);
  live_count_.store(live_.size());
  in->magic = 0;
  delete in;
}

void NgxRrProvider::release_all() {
  std::lock_guard<std::recursive_mutex> lock(mu_);
  while (!live_.empty()) release(*live_.begin());
}

// ------------------------------------------------------------------------------------------------------------
// the table for the host

WgHostNgxRr NgxRrProvider::table() {
  WgHostNgxRr t;
  memset(&t, 0, sizeof t);
  t.size = sizeof t;
  t.user = this;
  t.create = [](void* u, void* dev, void* cl, const WgHostNgxParams* p, void** ctx) -> bool {
    return static_cast<NgxRrProvider*>(u)->create(dev, cl, p, ctx);
  };
  t.evaluate = [](void* u, void* ctx, void* cl, const WgHostNgxParams* p) -> bool {
    return static_cast<NgxRrProvider*>(u)->evaluate(ctx, cl, p);
  };
  t.release = [](void* u, void* ctx) { static_cast<NgxRrProvider*>(u)->release(ctx); };
  return t;
}

}  // namespace wgrr
