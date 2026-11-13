// SPDX-License-Identifier: MIT
// Replay of a game profile (tests/data/<game>-<version>-traffic.json, written by tools/probe_log_to_profile.py from
// an in-game probe log) into the release proxy core (src/proxy_core.*) with the fake runtime (tests/fake_runtime.h).
//
// What is replayed (as the probe saw the game): the <init> sessions and every call before the first frame token with
// the results the game got; the loading phase (frame tokens and constants, no RR); then, per RR frame, the frame token,
// the constants (flags and their changes as logged), the game's tags in the logged order with the logged lifetime,
// extent, declared state, an empty description format / size (the game fills only the native object and the state)
// and the native resource's format (answered by the runtime's describe_resource), further tokens of the same frame
// index (the same token object, see below), <rr-options> with the logged values and <evaluate> with the viewport only.
// Not in the log, synthetic here: the jitter after the two logged samples, the camera-view-to-clip matrix (only its
// [2][0], [2][1] = 0 were logged) and the exact position of the extra tokens within a frame.
//
// Token objects: the probe stored constants under the token object and found them at every evaluate (log record
// "evaluate ... constants=1", never changed), so the evaluate's token is the object the constants were given with.
// The fake original returns one object per frame index (64-entry ring).
//
// Checks: every result equals what the game got (forwarded calls) or 0 (every RR call of the release logic); one
// instance created at the first evaluate with the creation decisions of 61.3 from these values; per evaluate the
// frame handed to the runtime (roles after 62.2 routing, buffer type, native resource, DXGI format resolved from the
// native resource, format class, declared state, origin, extent, lifetime, render / output size, jitter, motion-vector
// scale, combined reset, parity, counter, matrices). Finally it lists, without failing, where what the game sends
// differs from what WrongGPU's runtime at clean/rr-f-rt-int (b323906, read-only) handles or has tested ("runtime gap").
#include <stdio.h>
#include <string.h>

#include <map>
#include <set>
#include <string>
#include <vector>

#include "../src/formats.h"
#include "../src/proxy_core.h"
#include "fake_runtime.h"
#include "mini_json.h"
#include "test_util.h"

using namespace wgrr;

namespace {

// ---------------------------------------------------------------- fake original (forwarded calls get the logged rc)
int32_t g_rc = 0;  // result the next forwarded call returns
int g_forwarded = 0;
std::map<std::string, int> g_forward_roles;
void fwd(const char* role) { g_forwarded++; g_forward_roles[role]++; }
WgResultCode o_init(const WgPreferences*, uint64_t) { fwd("init"); return g_rc; }
WgResultCode o_shutdown() { fwd("shutdown"); return 0; }
WgResultCode o_is_supported(uint32_t, const WgAdapterInfo*) { fwd("is-supported"); return g_rc; }
WgResultCode o_is_loaded(uint32_t, bool* l) { fwd("is-loaded"); if (l) *l = true; return g_rc; }
WgResultCode o_set_loaded(uint32_t, bool) { fwd("set-loaded"); return g_rc; }
WgResultCode o_get_function(uint32_t, const char*, void** o) { fwd("get-function"); if (o) *o = nullptr; return g_rc; }
WgResultCode o_set_device(void*) { fwd("set-device"); return 0; }
WgResultCode o_tag(const WgViewport*, const WgTag*, uint32_t, void*) { fwd("tag"); return 0; }
WgResultCode o_constants(const WgConstants*, const WgFrameToken*, const WgViewport*) { fwd("constants"); return 0; }
struct FakeToken {
  WgFrameToken t;
  uint32_t index;
};
uint32_t token_index_fn(const WgFrameToken* t) { return reinterpret_cast<const FakeToken*>(t)->index; }
const WgFrameTokenIndexFn kTokenVtbl[1] = {&token_index_fn};
FakeToken g_tokens[64];
WgResultCode o_new_frame_token(WgFrameToken** out, const uint32_t* idx) {
  fwd("new-frame-token");
  uint32_t i = idx ? *idx : 0;
  FakeToken& t = g_tokens[i % 64];  // the same object for every call with this frame index
  t.t.vtable = kTokenVtbl;
  t.index = i;
  *out = &t.t;
  return 0;
}
WgResultCode o_evaluate(uint32_t, const WgFrameToken*, const WgHeader* const*, uint32_t, void*) {
  fwd("evaluate");
  return g_rc;
}
WgResultCode o_native(void* w, void** n) { *n = reinterpret_cast<uint8_t*>(w) + 1; return 0; }

OriginalApi make_orig() {
  OriginalApi o;
  o.init = o_init; o.shutdown = o_shutdown; o.is_supported = o_is_supported; o.is_loaded = o_is_loaded;
  o.set_loaded = o_set_loaded; o.get_function = o_get_function; o.set_device = o_set_device; o.tag = o_tag;
  o.constants = o_constants; o.new_frame_token = o_new_frame_token; o.evaluate = o_evaluate;
  o.native_interface = o_native;
  return o;
}

WgGuid guid(uint8_t k) { WgGuid g; memset(g.b, 0x3c, 16); g.b[15] = k; return g; }
const WgGuid kGViewport = guid(1), kGTag = guid(2);
uint8_t g_pool[512];

// ---------------------------------------------------------------- what the game sends (from the profile)
struct GameTag {
  uint32_t type, lifetime, state, native_format, desc_format, desc_w, desc_h;
  WgExtent extent;  // top, left, width, height
  bool null_resource;
  void* native;
  WgResource res;
  WgTag tag;
};

struct ConstChange {
  uint32_t frame_index, reset, cmi, jit, inv, mv3d, dil, ortho, version;
  float mvs[2];
};

struct Profile {
  uint32_t api_minor = 0;
  std::vector<GameTag> tags;
  std::vector<ConstChange> consts;
  WgRrOptions opts{};
  uint32_t release_rc = 0;
  uint32_t first_index = 0, evaluates = 0;
  uint32_t loading_tokens_per_index = 4;
  uint32_t rr_tokens_per_index = 9;
  std::vector<std::pair<uint32_t, std::vector<float>>> w2v;  // (evaluate, matrix) from the normal samples
  std::vector<std::vector<float>> jitter;                    // logged jitter of evaluates 1, 2, ...
  uint32_t render_w = 0, render_h = 0;
};

uint32_t u_at(const mj::Value& v, size_t i) { return v[i].u(); }

bool load_profile(const mj::Value& j, Profile* p) {
  p->api_minor = j["api_minor"].u();
  const mj::Value& tl = j["tags"]["list"];
  for (size_t i = 0; i < tl.size(); ++i) {
    const mj::Value& t = tl[i];
    GameTag g{};
    g.type = t["type"].u();
    g.lifetime = t["lifetime"].u();
    g.extent = {u_at(t["extent"], 0), u_at(t["extent"], 1), u_at(t["extent"], 2), u_at(t["extent"], 3)};
    g.null_resource = t["resource"].is_null();
    if (!g.null_resource) {
      const mj::Value& r = t["resource"];
      g.state = r["state"].u();
      g.desc_format = r["desc_format"].u();
      g.desc_w = u_at(r["desc_size"], 0);
      g.desc_h = u_at(r["desc_size"], 1);
      g.native_format = r["native"]["format"].u();
    }
    p->tags.push_back(g);
  }
  const mj::Value& ch = j["constants"]["changes"];
  for (size_t i = 0; i < ch.size(); ++i) {
    const mj::Value& c = ch[i];
    ConstChange k{c["frame_index"].u(), c["reset"].u(), c["camera_motion_included"].u(), c["mv_jittered"].u(),
                  c["depth_inverted"].u(), c["mv_3d"].u(), c["mv_dilated"].u(), c["orthographic"].u(),
                  c["version"].u(), {c["mv_scale"][0].f(), c["mv_scale"][1].f()}};
    p->consts.push_back(k);
  }
  const mj::Value& o = j["rr_options"];
  memset(&p->opts, 0, sizeof p->opts);
  p->opts.h.version = o["version"].u();
  p->opts.mode = o["mode"].u();
  p->opts.output_width = o["output"][0].u();
  p->opts.output_height = o["output"][1].u();
  p->opts.sharpness = o["sharpness"].f();
  p->opts.pre_exposure = o["pre_exposure"].f();
  p->opts.exposure_scale = o["exposure_scale"].f();
  p->opts.colour_is_hdr = (uint8_t)o["hdr"].u();
  p->opts.indicator_invert_x = (uint8_t)o["invert_x"].u();
  p->opts.indicator_invert_y = (uint8_t)o["invert_y"].u();
  p->opts.normal_roughness_packing = o["normal_roughness_packing"].u();
  p->opts.alpha_upscaling = (uint8_t)o["alpha_upscaling"].u();
  for (int i = 0; i < 6; ++i) p->opts.presets[i] = o["presets"][i].u();
  p->release_rc = o["release_rc"].u();
  p->first_index = j["rr_phase"]["first_evaluate_frame_index"].u();
  p->evaluates = j["rr_phase"]["evaluates"].u();
  double tpe = j["timing"]["tokens_per_evaluate_rr_phase"].d();
  p->rr_tokens_per_index = (uint32_t)(tpe + 0.5);
  const mj::Value& ns = j["normal_samples"];
  for (size_t i = 0; i < ns.size(); ++i) {
    std::vector<float> m;
    for (size_t k = 0; k < 16; ++k) m.push_back(ns[i]["world_to_view"][k].f());
    p->w2v.emplace_back(ns[i]["evaluate"].u(), m);
  }
  const mj::Value& js = j["jitter_camera"]["samples"];
  for (size_t i = 0; i < js.size(); ++i) p->jitter.push_back({js[i]["jitter"][0].f(), js[i]["jitter"][1].f()});
  p->render_w = j["evaluate"]["render"][0].u();
  p->render_h = j["evaluate"]["render"][1].u();
  return p->api_minor && !p->tags.empty() && !p->consts.empty() && p->evaluates;
}

const ConstChange& const_at(const Profile& p, uint32_t idx) {
  const ConstChange* c = &p.consts[0];
  for (const ConstChange& k : p.consts)
    if (k.frame_index <= idx) c = &k;
  return *c;
}

float halton(uint32_t i, uint32_t b) {
  float f = 1, r = 0;
  for (; i; i /= b) { f /= (float)b; r += f * (float)(i % b); }
  return r;
}

// Role the release logic routes a buffer type to (62.2), for this game's tags.
int role_of(uint32_t type, uint32_t packing) {
  switch (type) {
    case WG_BT_COLOUR_IN: return WG_ROLE_COLOUR;
    case WG_BT_COLOUR_OUT: return WG_ROLE_OUTPUT;
    case WG_BT_DEPTH: case WG_BT_LINEAR_DEPTH: return WG_ROLE_DEPTH;
    case WG_BT_MOTION: return WG_ROLE_MOTION;
    case WG_BT_DIFFUSE_ALBEDO: return WG_ROLE_DIFFUSE_ALBEDO;
    case WG_BT_SPECULAR_ALBEDO: return WG_ROLE_SPECULAR_ALBEDO;
    case WG_BT_NORMAL_ROUGHNESS: return packing == 1 ? WG_ROLE_NORMALS : -1;
    case WG_BT_NORMALS: return packing == 0 ? WG_ROLE_NORMALS : -1;
    case WG_BT_ROUGHNESS: return packing == 0 ? WG_ROLE_ROUGHNESS : -1;
    case WG_BT_SPECULAR_MOTION: return WG_ROLE_SPECULAR_MOTION;
    case WG_BT_SPECULAR_HIT_DISTANCE: return WG_ROLE_SPECULAR_HIT_DIST;
    case WG_BT_DISOCCLUSION: return WG_ROLE_DISOCCLUSION;
    case WG_BT_SSS_GUIDE: case WG_BT_COLOUR_BEFORE_FOG: return WG_ROLE_G1;
    case WG_BT_DOF_GUIDE: return WG_ROLE_DOF_GUIDE;
    case WG_BT_COLOUR_BEFORE_TRANSPARENCY: return WG_ROLE_BEFORE_TRANSPARENCY;
    case WG_BT_COLOUR_BEFORE_PARTICLES: case WG_BT_TRANSPARENCY_LAYER: return WG_ROLE_LAYER;
    case WG_BT_TRANSPARENCY_OPACITY: return WG_ROLE_OPACITY;
    default: return -1;  // "not used" types (62.2), e.g. 29
  }
}

const char* kRoleNames[WG_ROLE_COUNT] = {"colour", "output", "depth", "motion", "diffuse albedo", "specular albedo",
                                         "normals", "roughness", "specular motion", "specular hit distance",
                                         "disocclusion", "G1 guide", "DOF guide", "before transparency", "layer",
                                         "opacity", "input alpha", "output alpha"};

// ---------------------------------------------------------------- runtime snapshot (clean/rr-f-rt-int b323906)
// Transcribed from clean/rr_f/runtime/src/rr_runtime.cpp, tests/e2e.py, tests/rr_test_host.cpp and
// results/2026-10-20-rr-runtime.md at b323906 (read-only): what the runtime requires or has been tested with.
struct RuntimeSnapshot {
  bool requires_image_size = true;                 // ingest: "render rectangle ... outside its WxH resource" -> 25
  std::set<uint32_t> tested_input_formats = {41, 16, 2};  // e2e FMT: R32_FLOAT, R32G32_FLOAT, R32G32B32A32_FLOAT
  std::set<uint32_t> tested_input_states = {0xc0, 0x0};   // e2e: ALL_SHADER, one input COMMON
  std::set<uint32_t> tested_output_states = {0x0};        // test host: output in COMMON
  std::set<uint32_t> tested_packing = {0};                // e2e create: packing 0
  std::set<uint32_t> tested_depth_inverted = {0};
  std::set<uint32_t> tested_mv_jittered = {1};
  bool layer_resolve = false;  // b323906: plain resolve, MINIMAL placeholder (no layer / opacity / alpha)
};

struct Gaps {
  std::vector<std::string> list;
  void add(const std::string& s) {
    for (const auto& x : list)
      if (x == s) return;
    list.push_back(s);
  }
};

// ---------------------------------------------------------------- one replay
struct Result {
  int nonzero = 0, evaluates_ok = 0;
};

Result replay(const mj::Value& j, Profile& P, bool hook_mode, Gaps* gaps) {
  Result out;
  g_forwarded = 0;
  g_forward_roles.clear();
  memset(g_tokens, 0, sizeof g_tokens);
  FakeRuntime rt;
  Settings settings;
  settings.set_save_enabled(false);
  ProxyConfig cfg;
  cfg.api_minor = P.api_minor;
  cfg.ids.viewport = kGViewport;
  cfg.ids.tag = kGTag;
  cfg.ids.has_tag_conversion = false;  // as the real binding tables (Q-B6)
  cfg.version_magic = 0xfedc;
  cfg.fn_optimal = "fnOptimal"; cfg.fn_state = "fnState"; cfg.fn_options = "fnOptions";
  RrProxy p(cfg, make_orig(), &rt.table, &settings, nullptr, [] { return (uint64_t)1000; });
  p.set_feature_function_pointers((void*)1, (void*)2, (void*)3);
  void* dev = &g_pool[500];
  void* cmd = &g_pool[502];
  const std::map<std::string, const char*> fn_by_role = {{"rr-optimal", "fnOptimal"}, {"rr-state", "fnState"},
                                                         {"rr-options", "fnOptions"}};

  // the game's resources: one native object per tag, its format answered by describe_resource
  for (size_t i = 0; i < P.tags.size(); ++i) {
    GameTag& g = P.tags[i];
    g.native = &g_pool[16 * i];
    memset(&g.res, 0, sizeof g.res);
    g.res.h.version = 1;
    g.res.kind = WG_RES_TEX2D;
    g.res.native = g.native;
    g.res.state = g.state;
    g.res.width = g.desc_w;
    g.res.height = g.desc_h;
    g.res.native_format = g.desc_format;
    memset(&g.tag, 0, sizeof g.tag);
    g.tag.h.type = kGTag;
    g.tag.h.version = 1;
    g.tag.resource = g.null_resource ? nullptr : &g.res;
    g.tag.buffer_type = g.type;
    g.tag.lifetime = g.lifetime;
    g.tag.extent = g.extent;
    if (!g.null_resource) rt.described[g.native] = {g.native_format, 1};
  }
  WgViewport vp;
  memset(&vp, 0, sizeof vp);
  vp.h.type = kGViewport;
  vp.h.version = 1;
  vp.id = 0;

  if (hook_mode) p.adopt_unseen_init();  // hook mode: the plugin loads after the game's first <init> may have run

  // ---- calls before the first frame token, with the results the game got
  const mj::Value& sess = j["init_sessions"];
  size_t next_session = 0;
  std::vector<uint32_t> feats;
  const uint8_t luid[8] = {1, 2, 3, 4, 5, 6, 7, 8};  // the fake runtime's supported adapter
  const mj::Value& setup = j["setup_calls"];
  for (size_t i = 0; i < setup.size(); ++i) {
    const mj::Value& c = setup[i];
    const std::string role = c["role"].s;
    const int32_t want = c["rc"].i();
    const uint32_t feature = c.has("feature") ? c["feature"].u() : 0;
    g_rc = want;
    WgResultCode rc = 0;
    if (role == "init") {
      const mj::Value& s = sess[next_session++];
      feats.clear();
      for (size_t k = 0; k < s["features"].size(); ++k) feats.push_back(s["features"][k].u());
      WgPreferences pr;
      memset(&pr, 0, sizeof pr);
      pr.h.version = s["prefs_version"].u();
      pr.flags = (uint64_t)s["flags"].u();
      pr.features = feats.data();
      pr.feature_count = (uint32_t)feats.size();
      pr.render_api = s["render_api"].u();
      pr.engine = s["engine"].u();
      pr.application_id = s["app_id"].u();
      const uint64_t ver = (2ull << 48) | (7ull << 32) | (1ull << 16) | (uint64_t)s["magic"].u();
      rc = p.init(&pr, ver);
    } else if (role == "shutdown") {
      rc = p.shutdown();
    } else if (role == "upgrade-interface") {
      continue;  // not a core role: the release build forwards it through a stub
    } else if (role == "set-device") {
      rc = p.set_device(dev);
    } else if (role == "is-supported") {
      WgAdapterInfo a;
      memset(&a, 0, sizeof a);
      a.h.version = 1;
      a.luid = luid;
      a.luid_size = 8;
      rc = p.is_supported(feature, &a);
    } else if (role == "set-loaded") {
      rc = p.set_loaded(feature, c["value"].u() != 0);
    } else if (role == "is-loaded") {
      bool v = false;
      rc = p.is_loaded(feature, &v);
      CHECK_EQ((int)v, c["value"].i());
    } else if (role == "get-function") {
      void* fp = nullptr;
      const char* name = "withheld";
      if (feature == WG_FEAT_RAY_RECONSTRUCTION) name = fn_by_role.at(c["function_role"].s);
      rc = p.get_function(feature, name, &fp);
    } else {
      fprintf(stderr, "unknown setup role %s\n", role.c_str());
      CHECK(false);
      continue;
    }
    if (rc != want) fprintf(stderr, "setup seq %d %s(%u): rc %d, the game got %d\n", c["seq"].i(), role.c_str(),
                            feature, (int)rc, (int)want);
    CHECK_EQ(rc, want);
  }
  CHECK(!hook_mode || p.init_adopted());

  // ---- loading phase: frame tokens and constants, no RR
  WgConstants k;
  auto make_consts = [&](uint32_t idx, uint32_t e) {
    const ConstChange& c = const_at(P, idx);
    memset(&k, 0, sizeof k);
    k.h.version = c.version;
    // camera view to clip: synthetic values with [2][0] = [2][1] = 0 as logged (reversed-depth style)
    k.view_to_clip[0] = 1.2f; k.view_to_clip[5] = 2.1f; k.view_to_clip[11] = 1.0f; k.view_to_clip[14] = 0.05f;
    if (e && e <= P.jitter.size()) {
      k.jitter[0] = P.jitter[e - 1][0];
      k.jitter[1] = P.jitter[e - 1][1];
    } else {  // synthetic (not logged), within the logged range of +-0.5 pixel
      k.jitter[0] = halton(idx + 1, 2) - 0.5f;
      k.jitter[1] = halton(idx + 1, 3) - 0.5f;
    }
    k.mv_scale[0] = c.mvs[0];
    k.mv_scale[1] = c.mvs[1];
    k.depth_inverted = (uint8_t)c.inv;
    k.camera_motion_included = (uint8_t)c.cmi;
    k.mv_3d = (uint8_t)c.mv3d;
    k.reset = (uint8_t)c.reset;
    k.orthographic = (uint8_t)c.ortho;
    k.mv_dilated = (uint8_t)c.dil;
    k.mv_jittered = (uint8_t)c.jit;
    k.min_depth_separation = 40.0f;
  };
  WgFrameToken* tok = nullptr;
  for (uint32_t idx = 0; idx < P.first_index; ++idx) {
    for (uint32_t t = 0; t < P.loading_tokens_per_index; ++t) {
      CHECK_EQ(p.new_frame_token(&tok, &idx), 0);
      if (t == 0) {
        make_consts(idx, 0);
        CHECK_EQ(p.constants(&k, tok, &vp), 0);
      }
    }
  }
  CHECK_EQ(rt.creates, 0);
  CHECK_EQ(rt.evaluates, 0);

  // ---- RR phase
  const uint32_t packing = P.opts.normal_roughness_packing;
  uint32_t expect_mask = 0;
  for (const GameTag& g : P.tags) {
    int r = role_of(g.type, packing);
    if (r >= 0 && !g.null_resource) expect_mask |= 1u << r;
  }
  std::vector<float> prev_jitter = {0, 0};
  std::vector<float> prev_w2v(16, 0.0f), prev_v2c(16, 0.0f);
  const WgHeader* inputs[1] = {&vp.h};
  RuntimeSnapshot rs;
  for (uint32_t e = 1; e <= P.evaluates; ++e) {
    uint32_t idx = P.first_index + e - 1;
    CHECK_EQ(p.new_frame_token(&tok, &idx), 0);
    make_consts(idx, e);
    const WgConstants kc = k;
    CHECK_EQ(p.constants(&k, tok, &vp), 0);
    for (GameTag& g : P.tags) CHECK_EQ(p.tag(&vp, &g.tag, 1, cmd), 0);
    for (uint32_t t = 2; t < P.rr_tokens_per_index; ++t) CHECK_EQ(p.new_frame_token(&tok, &idx), 0);
    WgFrameToken* etok = nullptr;
    CHECK_EQ(p.new_frame_token(&etok, &idx), 0);  // the token passed to evaluate (same object)
    if (e == 1) {
      void* fp = nullptr;
      CHECK_EQ(p.get_function(WG_FEAT_RAY_RECONSTRUCTION, "fnOptions", &fp), 0);
      CHECK(fp == (void*)3);
    }
    WgRrOptions o = P.opts;
    const std::vector<float>* w = &P.w2v[0].second;
    for (const auto& s : P.w2v)
      if (s.first <= e) w = &s.second;
    for (int i = 0; i < 16; ++i) o.world_to_view[i] = (*w)[i];
    CHECK_EQ(p.rr_options(&vp, &o), (WgResultCode)P.release_rc);
    const int before = rt.evaluates;
    g_rc = 0;
    WgResultCode rc = p.evaluate(WG_FEAT_RAY_RECONSTRUCTION, etok, inputs, 1, cmd);
    if (rc != 0) {
      if (out.nonzero < 5) fprintf(stderr, "evaluate %u (frame index %u): rc %d\n", e, idx, (int)rc);
      out.nonzero++;
      continue;
    }
    CHECK_EQ(rt.evaluates, before + 1);
    if (rt.evaluates != before + 1) continue;
    out.evaluates_ok++;
    const WgRrFrame& f = rt.last();
    const bool reset_expected = e == 1 || kc.reset == WG_TRI_TRUE;
    CHECK_EQ(f.api_minor, P.api_minor);
    CHECK_EQ(f.render_width, P.render_w);
    CHECK_EQ(f.render_height, P.render_h);
    CHECK_EQ(f.output_width, P.opts.output_width);
    CHECK_EQ(f.output_height, P.opts.output_height);
    CHECK(f.jitter[0] == kc.jitter[0] && f.jitter[1] == kc.jitter[1]);
    CHECK(f.prev_jitter[0] == prev_jitter[0] && f.prev_jitter[1] == prev_jitter[1]);
    CHECK(f.mv_scale[0] == kc.mv_scale[0] * (float)P.render_w && f.mv_scale[1] == kc.mv_scale[1] * (float)P.render_h);
    CHECK_EQ(f.reset, (uint32_t)reset_expected);
    CHECK_EQ(f.parity, (e - 1) & 1u);
    CHECK_EQ(f.counter, (e - 1) % 8u);
    CHECK_EQ(f.rebuild_render_items, 0u);
    CHECK_EQ(f.depth_is_linear, 0u);
    CHECK_EQ(f.settings, 0u);
    CHECK(f.command_list == (uint8_t*)cmd + 1);
    CHECK(memcmp(f.view_to_clip, kc.view_to_clip, sizeof f.view_to_clip) == 0);
    CHECK(memcmp(f.world_to_view, o.world_to_view, sizeof f.world_to_view) == 0);
    CHECK(memcmp(f.prev_world_to_view, reset_expected ? o.world_to_view : prev_w2v.data(),
                 sizeof f.prev_world_to_view) == 0);
    CHECK(memcmp(f.prev_view_to_clip, reset_expected ? kc.view_to_clip : prev_v2c.data(),
                 sizeof f.prev_view_to_clip) == 0);
    CHECK_EQ(f.attached, expect_mask);
    CHECK_EQ(f.layer_origin_x, 0u);
    CHECK_EQ(f.layer_origin_y, 0u);
    for (const GameTag& g : P.tags) {
      int r = role_of(g.type, packing);
      if (r < 0 || g.null_resource) continue;
      const WgRrImage& im = f.images[r];
      CHECK(im.resource == g.native);
      CHECK_EQ(im.buffer_type, g.type);
      CHECK_EQ(im.format, g.desc_format ? g.desc_format : g.native_format);  // 62.5: the native resource's format
      CHECK_EQ(im.format_class, format_class(im.format, r == WG_ROLE_DEPTH));
      CHECK(im.format_class != WG_FMT_UNSUPPORTED);
      CHECK_EQ(im.state, g.state);
      CHECK_EQ(im.origin_x, g.extent.left);
      CHECK_EQ(im.origin_y, g.extent.top);
      CHECK_EQ(im.extent_width, g.extent.width);
      CHECK_EQ(im.extent_height, g.extent.height);
      CHECK_EQ(im.lifetime, g.lifetime);
      CHECK_EQ(im.conversion, 0u);
      CHECK_EQ(im.width, g.desc_w);  // as described by the game (0 x 0 here): see the runtime gaps
      CHECK_EQ(im.height, g.desc_h);
      if (e == 1 && gaps) {  // ---- what the runtime snapshot handles
        char b[400];
        const bool output = r == WG_ROLE_OUTPUT;
        if (rs.requires_image_size && (im.width == 0 || im.height == 0)) {
          snprintf(b, sizeof b, "size: every input is handed over with width x height %ux%u (the game fills no size "
                   "in the resource description); the runtime's ingest bounds check rejects the frame with 25",
                   im.width, im.height);
          gaps->add(b);
        }
        if (!output && !rs.tested_input_formats.count(im.format)) {
          snprintf(b, sizeof b, "format: %s (type %u) in DXGI format %u (class %u) - read path not exercised by the "
                   "runtime tests (32-bit float inputs only)", kRoleNames[r], g.type, im.format, im.format_class);
          gaps->add(b);
        }
        if (!output && !rs.tested_input_states.count(im.state)) {
          snprintf(b, sizeof b, "state: %s (type %u) declared 0x%x - transition to NON_PIXEL_SHADER_RESOURCE and "
                   "back not exercised by the runtime tests", kRoleNames[r], g.type, im.state);
          gaps->add(b);
        }
        if (output && !rs.tested_output_states.count(im.state)) {
          snprintf(b, sizeof b, "state: output declared 0x%x with a typed UAV (direct store, no transition) - the "
                   "runtime tests hand the output over in COMMON", im.state);
          gaps->add(b);
        }
        if (r == WG_ROLE_LAYER && !rs.layer_resolve) {
          snprintf(b, sizeof b, "layer: type %u is attached every frame, so 62.2 / 54b.2 select the extended output "
                   "resolve with it as the layer (origin (0, 0), no type 51); the runtime at b323906 runs the plain "
                   "MINIMAL placeholder resolve and ignores the layer", g.type);
          gaps->add(b);
        }
      }
    }
    for (int r = 0; r < WG_ROLE_COUNT; ++r)
      if (!(expect_mask >> r & 1u)) CHECK(f.images[r].resource == nullptr);
    prev_jitter = {kc.jitter[0], kc.jitter[1]};
    memcpy(prev_w2v.data(), o.world_to_view, sizeof o.world_to_view);
    memcpy(prev_v2c.data(), kc.view_to_clip, sizeof kc.view_to_clip);
  }

  // ---- the one instance and its creation decisions (61.3)
  CHECK_EQ(rt.creates, 1);
  CHECK_EQ(rt.retains, 0);  // every tag has lifetime 1: nothing is copied during tagging (62.1)
  if (!rt.created.empty()) {
    const WgRrCreate& d = rt.created[0];
    const ConstChange& c0 = const_at(P, P.first_index);
    CHECK_EQ(d.api_minor, P.api_minor);
    CHECK_EQ(d.output_width, P.opts.output_width);
    CHECK_EQ(d.output_height, P.opts.output_height);
    CHECK_EQ(d.render_width, P.render_w);  // created by the first evaluate at its own render size (Q-B4)
    CHECK_EQ(d.render_height, P.render_h);
    CHECK_EQ(d.mode, P.opts.mode);
    CHECK_EQ(d.packing, packing);
    CHECK_EQ(d.colour_hdr, (uint32_t)(P.opts.colour_is_hdr == WG_TRI_TRUE));
    CHECK_EQ(d.depth_inverted, (uint32_t)(c0.inv == WG_TRI_TRUE));
    CHECK_EQ(d.mv_jittered, (uint32_t)(c0.jit == WG_TRI_TRUE));
    CHECK_EQ(d.mv_output_resolution, 0u);
    CHECK_EQ(d.hw_depth, 1u);
    CHECK_EQ(d.alpha_upscaling, 0u);
    CHECK_EQ(d.sharpen_request, (uint32_t)(P.opts.sharpness > 0));
    CHECK_EQ(d.latched_settings, 1u);
    if (gaps) {
      char b[300];
      if (!rs.tested_packing.count(d.packing)) {
        snprintf(b, sizeof b, "creation: normal/roughness packing %u (roughness in the normal's 4th channel) - the "
                 "runtime tests create with packing 0 only", d.packing);
        gaps->add(b);
      }
      if (!rs.tested_depth_inverted.count(d.depth_inverted)) {
        snprintf(b, sizeof b, "creation: depth inverted = %u with hardware depth (rev = 1, 52.8) - the runtime tests "
                 "create with depth inverted = 0 only", d.depth_inverted);
        gaps->add(b);
      }
      if (!rs.tested_mv_jittered.count(d.mv_jittered)) {
        snprintf(b, sizeof b, "creation: motion vectors jittered = %u (mo = 0) - the runtime tests create with 1 only",
                 d.mv_jittered);
        gaps->add(b);
      }
    }
  }
  uint32_t resets = 0;
  for (const WgRrFrame& f : rt.frames) resets += f.reset;
  CHECK_EQ(resets, 2u);  // creation (evaluate 1) and the game's one reset frame (frame index 3967)
  CHECK_EQ(p.shutdown(), 0);  // the game's final <shutdown>
  CHECK_EQ(rt.destroys, 1);
  CHECK(rt.live.empty());
  CHECK_EQ(p.log_count("rr: "), 0u);  // no rejection, routing or format message was logged
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  std::string path;
  if (argc > 1) {
    path = argv[1];
  } else {
    path = __FILE__;
    path = path.substr(0, path.find_last_of("/\\") + 1) + "data/cp2077-2.7.1-traffic.json";
  }
  mj::Value j;
  if (!mj::load(path.c_str(), &j)) {
    fprintf(stderr, "cannot read profile %s\n", path.c_str());
    return 2;
  }
  Profile P;
  CHECK(load_profile(j, &P));
  if (g_failures) TEST_SUMMARY("test_replay_profile");
  printf("profile %s: game %s, api 2.%u, %zu tags per frame, %u RR evaluates from frame index %u, %u tokens per "
         "frame index\n", path.c_str(), j["game"].s.c_str(), P.api_minor, P.tags.size(), P.evaluates, P.first_index,
         P.rr_tokens_per_index);
  Gaps gaps;
  for (int hook = 0; hook < 2; ++hook) {
    Result r = replay(j, P, hook != 0, hook ? nullptr : &gaps);
    printf("%s: %d evaluates handed to the runtime, %d with rc != 0\n",
           hook ? "hook mode (unseen init adopted first)" : "file replacement", r.evaluates_ok, r.nonzero);
    CHECK_EQ(r.nonzero, 0);
    CHECK_EQ(r.evaluates_ok, (int)P.evaluates);
  }
  printf("runtime gaps (game traffic vs clean/rr-f-rt-int b323906; informative, not failures): %zu\n", gaps.list.size());
  for (const auto& s : gaps.list) printf("  runtime gap: %s\n", s.c_str());
  TEST_SUMMARY("test_replay_profile");
}
