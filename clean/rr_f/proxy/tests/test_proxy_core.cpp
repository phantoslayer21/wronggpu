// SPDX-License-Identifier: MIT
// Unit tests of the proxy's game-interface logic (spec 60-63) against a fake original and a fake runtime:
// return codes and their precedence, forwarding of every other feature, proxy answers for feature 1001,
// creation / reset decisions, rejections, routing of 62.2, formats of 62.5, settings of 60.9, the ASI view.
#include <dxgiformat.h>
#include <stdio.h>
#include <string.h>

#include <string>
#include <vector>

#include "../asi/wg_rr_controls.h"
#include "../src/formats.h"
#include "../src/optimal.h"
#include "../src/proxy_core.h"
#include "fake_runtime.h"
#include "test_util.h"

using namespace wgrr;

// ---------------------------------------------------------------- fake original
struct OrigLog {
  int calls = 0;
  std::string last_role;
  uint32_t last_feature = 0;
  const void* last_ptr = nullptr;
  uint64_t last_u64 = 0;
  uint32_t last_count = 0;
  void* last_cmd = nullptr;
};
static OrigLog g_orig;
static uint32_t orig_rc(uint32_t feature) { return 100u + (feature % 50u); }
static void rec(const char* role, uint32_t f, const void* p) {
  g_orig.calls++;
  g_orig.last_role = role;
  g_orig.last_feature = f;
  g_orig.last_ptr = p;
}
static WgResultCode o_init(const WgPreferences* p, uint64_t v) { rec("init", 0, p); g_orig.last_u64 = v; return 0; }
static WgResultCode o_shutdown() { rec("shutdown", 0, nullptr); return 0; }
static WgResultCode o_is_supported(uint32_t f, const WgAdapterInfo* a) { rec("is_supported", f, a); return orig_rc(f); }
static WgResultCode o_is_loaded(uint32_t f, bool* l) { rec("is_loaded", f, l); return orig_rc(f); }
static WgResultCode o_set_loaded(uint32_t f, bool l) { rec("set_loaded", f, nullptr); g_orig.last_u64 = l; return orig_rc(f); }
static WgResultCode o_requirements(uint32_t f, WgRequirements* r) { rec("requirements", f, r); return orig_rc(f); }
static WgResultCode o_version(uint32_t f, WgFeatureVersion* v) { rec("version", f, v); return orig_rc(f); }
static WgResultCode o_get_function(uint32_t f, const char* n, void** o) { rec("get_function", f, n); (void)o; return orig_rc(f); }
static WgResultCode o_allocate(void* c, uint32_t f, const WgViewport* v) { rec("allocate", f, v); g_orig.last_cmd = c; return orig_rc(f); }
static WgResultCode o_free(uint32_t f, const WgViewport* v) { rec("free", f, v); return orig_rc(f); }
static WgResultCode o_set_device(void* d) { rec("set_device", 0, d); return 0; }
static WgResultCode o_tag(const WgViewport* v, const WgTag* t, uint32_t n, void* c) {
  rec("tag", 0, t); g_orig.last_count = n; g_orig.last_cmd = c; (void)v; return 0;
}
static WgResultCode o_tag_for_frame(const WgFrameToken* k, const WgViewport* v, const WgTag* t, uint32_t n, void* c) {
  rec("tag_for_frame", 0, t); g_orig.last_count = n; g_orig.last_cmd = c; (void)k; (void)v; return 0;
}
static WgResultCode o_constants(const WgConstants* c, const WgFrameToken* k, const WgViewport* v) {
  rec("constants", 0, c); (void)k; (void)v; return 0;
}
struct FakeToken {
  WgFrameToken t;
  uint32_t index;
};
static uint32_t token_index_fn(const WgFrameToken* t) { return reinterpret_cast<const FakeToken*>(t)->index; }
static const WgFrameTokenIndexFn kTokenVtbl[1] = {&token_index_fn};
static FakeToken g_tokens[64];
static uint32_t g_next_token = 0;
static WgResultCode o_new_frame_token(WgFrameToken** out, const uint32_t* idx) {
  rec("new_frame_token", 0, out);
  FakeToken& t = g_tokens[g_next_token++ % 64];
  memset(&t, 0, sizeof t);
  t.t.vtable = kTokenVtbl;
  t.index = idx ? *idx : 0;
  *out = &t.t;
  return 0;
}
static WgResultCode o_evaluate(uint32_t f, const WgFrameToken* k, const WgHeader* const* in, uint32_t n, void* c) {
  rec("evaluate", f, in); g_orig.last_count = n; g_orig.last_cmd = c; (void)k; return orig_rc(f);
}
static WgResultCode o_native(void* w, void** n) { *n = reinterpret_cast<uint8_t*>(w) + 1; return 0; }

static OriginalApi make_orig() {
  OriginalApi o;
  o.init = o_init; o.shutdown = o_shutdown; o.is_supported = o_is_supported; o.is_loaded = o_is_loaded;
  o.set_loaded = o_set_loaded; o.requirements = o_requirements; o.version = o_version;
  o.get_function = o_get_function; o.allocate = o_allocate; o.free_resources = o_free; o.set_device = o_set_device;
  o.tag = o_tag; o.tag_for_frame = o_tag_for_frame; o.constants = o_constants; o.new_frame_token = o_new_frame_token;
  o.evaluate = o_evaluate; o.native_interface = o_native;
  return o;
}

// ---------------------------------------------------------------- helpers
static WgGuid guid(uint8_t k) { WgGuid g; memset(g.b, 0x5a, 16); g.b[15] = k; return g; }
static const WgGuid kGViewport = guid(1), kGTag = guid(2), kGConv = guid(3), kGOther = guid(9);
static const uint32_t kMagic = 0xfedc;  // A9.1 version magic

static uint8_t g_res_pool[256];
static void* fake_res(int i) { return &g_res_pool[i]; }

struct Res {
  WgResource r;
  Res(int id, uint32_t w, uint32_t h, uint32_t fmt) {
    memset(&r, 0, sizeof r);
    r.kind = WG_RES_TEX2D;
    r.native = fake_res(id);
    r.width = w;
    r.height = h;
    r.native_format = fmt;
    r.state = 4;
  }
};

static WgTag make_tag(WgResource* r, uint32_t type, uint32_t life = WG_LIFE_UNTIL_PRESENT, WgExtent e = {0, 0, 0, 0}) {
  WgTag t;
  memset(&t, 0, sizeof t);
  t.h.type = kGTag;
  t.h.version = 1;
  t.resource = r;
  t.buffer_type = type;
  t.lifetime = life;
  t.extent = e;
  return t;
}

static WgViewport make_vp(uint32_t id) {
  WgViewport v;
  memset(&v, 0, sizeof v);
  v.h.type = kGViewport;
  v.h.version = 1;
  v.id = id;
  return v;
}

static WgRrOptions make_opts(uint32_t mode = 2, uint32_t ow = 1920, uint32_t oh = 1080, uint64_t ver = 3) {
  WgRrOptions o;
  memset(&o, 0, sizeof o);
  o.h.version = ver;
  o.mode = mode;
  o.output_width = ow;
  o.output_height = oh;
  o.pre_exposure = 1;
  o.exposure_scale = 1;
  o.colour_is_hdr = 1;
  for (int i = 0; i < 16; ++i) o.world_to_view[i] = (i % 5 == 0) ? 1.0f : 0.0f;
  o.world_to_view[12] = 3.0f;
  return o;
}

static WgConstants make_consts(float jx = 0.25f, float jy = -0.25f) {
  WgConstants c;
  memset(&c, 0, sizeof c);
  c.h.version = 2;
  for (int i = 0; i < 16; ++i) c.view_to_clip[i] = (i % 5 == 0) ? 2.0f : 0.0f;
  c.jitter[0] = jx;
  c.jitter[1] = jy;
  c.mv_scale[0] = 1.0f / 1280.0f;
  c.mv_scale[1] = 1.0f / 720.0f;
  c.depth_inverted = WG_TRI_UNSET;
  c.camera_motion_included = WG_TRI_UNSET;
  c.mv_3d = WG_TRI_UNSET;
  c.reset = WG_TRI_UNSET;
  return c;
}

// One proxy with the standard scene: all required inputs tagged as viewport tags at 1280x720 -> 1920x1080.
struct Env {
  FakeRuntime rt;
  Settings settings;
  uint64_t now = 5000;
  RrProxy* p = nullptr;
  WgViewport vp = make_vp(7);
  void* cmd = fake_res(200);
  void* dev = fake_res(201);
  Res colour{0, 1280, 720, DXGI_FORMAT_R16G16B16A16_FLOAT}, output{1, 1920, 1080, DXGI_FORMAT_R16G16B16A16_FLOAT};
  Res mv{2, 1280, 720, DXGI_FORMAT_R16G16_FLOAT}, depth{3, 1280, 720, DXGI_FORMAT_R32_FLOAT};
  Res dif{4, 1280, 720, DXGI_FORMAT_R8G8B8A8_UNORM}, spec{5, 1280, 720, DXGI_FORMAT_R8G8B8A8_UNORM};
  Res nrm{6, 1280, 720, DXGI_FORMAT_R10G10B10A2_UNORM}, rgh{7, 1280, 720, DXGI_FORMAT_R8_UNORM};
  std::vector<WgTag> tags;
  WgRrOptions opts = make_opts();
  WgConstants consts = make_consts();
  uint32_t api_minor;

  explicit Env(uint32_t minor = 14, bool with_runtime = true, bool conversion_known = true) : api_minor(minor) {
    settings.set_save_enabled(false);
    ProxyConfig cfg;
    cfg.api_minor = minor;
    cfg.ids.viewport = kGViewport;
    cfg.ids.tag = kGTag;
    cfg.ids.tag_conversion = kGConv;
    cfg.ids.has_tag_conversion = conversion_known;
    cfg.version_magic = kMagic;
    cfg.release_version[0] = 0; cfg.release_version[1] = 5; cfg.release_version[2] = 3;
    cfg.fn_optimal = "fnOptimal"; cfg.fn_state = "fnState"; cfg.fn_options = "fnOptions";
    p = new RrProxy(cfg, make_orig(), with_runtime ? &rt.table : nullptr, &settings, nullptr, [this] { return now; });
    p->set_feature_function_pointers((void*)1, (void*)2, (void*)3);
    tags = {make_tag(&colour.r, WG_BT_COLOUR_IN), make_tag(&output.r, WG_BT_COLOUR_OUT),
            make_tag(&mv.r, WG_BT_MOTION), make_tag(&depth.r, WG_BT_DEPTH),
            make_tag(&dif.r, WG_BT_DIFFUSE_ALBEDO), make_tag(&spec.r, WG_BT_SPECULAR_ALBEDO),
            make_tag(&nrm.r, WG_BT_NORMALS), make_tag(&rgh.r, WG_BT_ROUGHNESS)};
  }
  ~Env() { delete p; }
  uint32_t init(bool rr = true) {
    static const uint32_t feats[2] = {0, 1001};
    WgPreferences pr;
    memset(&pr, 0, sizeof pr);
    pr.features = feats;
    pr.feature_count = rr ? 2 : 1;
    pr.render_api = 1;
    uint64_t v = (2ull << 48) | ((uint64_t)api_minor << 32) | (1ull << 16) | kMagic;
    return p->init(&pr, v);
  }
  void setup(bool options = true) {
    init();
    p->set_device(dev);
    if (options) p->rr_options(&vp, &opts);
    p->tag(&vp, tags.data(), (uint32_t)tags.size(), cmd);
  }
  WgFrameToken* token(uint32_t idx) {
    WgFrameToken* t = nullptr;
    p->new_frame_token(&t, &idx);
    return t;
  }
  bool auto_tag = true;  // tag every input before each frame, as a game does
  uint32_t frame(uint32_t idx = 0, const std::vector<const WgHeader*>& extra = {}) {
    if (auto_tag) p->tag(&vp, tags.data(), (uint32_t)tags.size(), cmd);
    WgFrameToken* t = token(idx);
    p->constants(&consts, t, &vp);
    std::vector<const WgHeader*> in = {&vp.h};
    in.insert(in.end(), extra.begin(), extra.end());
    return p->evaluate(1001, t, in.data(), (uint32_t)in.size(), cmd);
  }
  void retag(const WgTag& t) { p->tag(&vp, &t, 1, cmd); }
};

// ---------------------------------------------------------------- tests
static void test_preinit_and_device() {
  Env e;
  WgViewport vp = make_vp(1);
  WgRrOptions o = make_opts();
  WgRrOptimal os;
  WgRrState st;
  bool b;
  WgRequirements rq;
  WgFeatureVersion fv;
  void* fp;
  g_orig = OrigLog();
  CHECK_EQ(e.p->is_supported(1001, nullptr), 23u);
  CHECK_EQ(e.p->is_loaded(1001, &b), 23u);
  CHECK_EQ(e.p->set_loaded(1001, true), 23u);
  CHECK_EQ(e.p->requirements(1001, &rq), 23u);
  CHECK_EQ(e.p->version(1001, &fv), 23u);
  CHECK_EQ(e.p->get_function(1001, "fnOptimal", &fp), 23u);
  CHECK_EQ(e.p->allocate(nullptr, 1001, &vp), 23u);
  CHECK_EQ(e.p->free_resources(1001, &vp), 23u);
  CHECK_EQ(e.p->evaluate(1001, nullptr, nullptr, 0, nullptr), 23u);
  CHECK_EQ(e.p->rr_optimal(&o, &os), 23u);
  CHECK_EQ(e.p->rr_state(&vp, &st), 23u);
  CHECK_EQ(e.p->rr_options(&vp, &o), 23u);
  CHECK_EQ(g_orig.calls, 0);          // none forwarded
  CHECK(!e.p->holds_viewport());      // no state change
  // after init, before set-device: the device-needing calls return 5
  CHECK_EQ(e.init(), 0u);
  CHECK_EQ(e.p->get_function(1001, "fnOptimal", &fp), 5u);
  CHECK_EQ(e.p->allocate(nullptr, 1001, &vp), 5u);
  const WgHeader* in[1] = {&vp.h};
  CHECK_EQ(e.p->evaluate(1001, nullptr, in, 1, nullptr), 5u);
  CHECK(!e.p->holds_viewport());
  CHECK_EQ(e.p->is_supported(1001, nullptr), 0u);  // not device-needing
  CHECK_EQ(e.p->rr_options(&vp, &o), 0u);
  CHECK_EQ(g_orig.calls, 1);  // only <init>
  // after shutdown: 23 again
  e.p->set_device(e.dev);
  CHECK_EQ(e.p->shutdown(), 0u);
  CHECK_EQ(e.p->is_supported(1001, nullptr), 23u);
  CHECK(!e.p->holds_viewport());
}

static void test_forwarding() {
  Env e;
  e.setup();
  const uint32_t feats[] = {0, 1, 2, 3, 4, 5, 6, 1000, 1002, 1003, 1004, 9999, 0xffffffffu};
  WgViewport vp = make_vp(3);
  WgAdapterInfo ai;
  bool b;
  WgRequirements rq;
  WgFeatureVersion fv;
  void* fp = nullptr;
  for (uint32_t f : feats) {
    g_orig = OrigLog();
    CHECK_EQ(e.p->is_supported(f, &ai), orig_rc(f)); CHECK(g_orig.last_role == "is_supported" && g_orig.last_feature == f && g_orig.last_ptr == &ai);
    CHECK_EQ(e.p->is_loaded(f, &b), orig_rc(f)); CHECK(g_orig.last_role == "is_loaded" && g_orig.last_ptr == &b);
    CHECK_EQ(e.p->set_loaded(f, true), orig_rc(f)); CHECK(g_orig.last_role == "set_loaded" && g_orig.last_u64 == 1);
    CHECK_EQ(e.p->requirements(f, &rq), orig_rc(f)); CHECK(g_orig.last_ptr == &rq);
    CHECK_EQ(e.p->version(f, &fv), orig_rc(f)); CHECK(g_orig.last_ptr == &fv);
    CHECK_EQ(e.p->get_function(f, "x", &fp), orig_rc(f)); CHECK(g_orig.last_role == "get_function");
    CHECK_EQ(e.p->allocate(e.cmd, f, &vp), orig_rc(f)); CHECK(g_orig.last_cmd == e.cmd && g_orig.last_ptr == &vp);
    CHECK_EQ(e.p->free_resources(f, &vp), orig_rc(f)); CHECK(g_orig.last_role == "free");
    const WgHeader* in[1] = {&vp.h};
    CHECK_EQ(e.p->evaluate(f, nullptr, in, 1, e.cmd), orig_rc(f));
    CHECK(g_orig.last_role == "evaluate" && g_orig.last_ptr == in && g_orig.last_count == 1 && g_orig.last_cmd == e.cmd);
    CHECK_EQ(g_orig.calls, 9);
  }
  CHECK(!e.p->has_instance());
  CHECK_EQ(e.rt.evaluates, 0);
  // init forwarded unchanged; set-device forwarded with the wrapped device; tags / constants forwarded
  Env f;
  g_orig = OrigLog();
  WgPreferences pr;
  memset(&pr, 0, sizeof pr);
  uint32_t ff[1] = {1001};
  pr.features = ff;
  pr.feature_count = 1;
  CHECK_EQ(f.p->init(&pr, 0x0002000e00010000ull | 77), 0u);
  CHECK(g_orig.last_ptr == &pr && g_orig.last_u64 == (0x0002000e00010000ull | 77));
  CHECK(pr.feature_count == 1 && ff[0] == 1001);
  f.p->set_device(f.dev);
  CHECK(g_orig.last_role == "set_device" && g_orig.last_ptr == f.dev);
  CHECK(f.rt.device == reinterpret_cast<uint8_t*>(f.dev) + 1);  // runtime got the unwrapped native device
  f.p->tag(&f.vp, f.tags.data(), 3, f.cmd);
  CHECK(g_orig.last_role == "tag" && g_orig.last_ptr == f.tags.data() && g_orig.last_count == 3 && g_orig.last_cmd == f.cmd);
  WgFrameToken* t = f.token(4);
  f.p->tag_for_frame(t, &f.vp, f.tags.data(), 2, f.cmd);
  CHECK(g_orig.last_role == "tag_for_frame" && g_orig.last_count == 2);
  f.p->constants(&f.consts, t, &f.vp);
  CHECK(g_orig.last_role == "constants" && g_orig.last_ptr == &f.consts);
}

static void test_answers() {
  Env e;
  e.init();
  e.p->set_device(e.dev);
  // is-supported / is-loaded (60.8)
  e.rt.status_bits = WG_RR_RT_READY;
  CHECK_EQ(e.p->is_supported(1001, nullptr), 7u);
  bool b = true;
  CHECK_EQ(e.p->set_loaded(1001, true), 0u);
  CHECK_EQ(e.p->is_loaded(1001, &b), 0u); CHECK(!b);  // false until the model has loaded
  CHECK_EQ(e.p->is_supported(1001, nullptr), 7u);     // set-loaded never makes RR usable
  e.rt.status_bits = WG_RR_RT_READY | WG_RR_RT_MODEL_LOADED;
  CHECK_EQ(e.p->is_supported(1001, nullptr), 0u);
  CHECK_EQ(e.p->is_loaded(1001, &b), 0u); CHECK(b);
  CHECK_EQ(e.p->set_loaded(1001, false), 0u);
  CHECK_EQ(e.p->is_loaded(1001, &b), 0u); CHECK(!b);
  Env e2;
  e2.init();
  CHECK_EQ(e2.p->is_loaded(1001, &b), 0u); CHECK(b);  // true if the game never set one
  WgAdapterInfo ai;
  memset(&ai, 0, sizeof ai);
  uint8_t good[8] = {1, 2, 3, 4, 5, 6, 7, 8}, bad[8] = {9, 9, 9, 9, 9, 9, 9, 9};
  ai.luid = good; ai.luid_size = 8;
  CHECK_EQ(e.p->is_supported(1001, &ai), 0u);
  ai.luid = bad;
  CHECK_EQ(e.p->is_supported(1001, &ai), 7u);
  ai.luid = nullptr; ai.luid_size = 0;
  CHECK_EQ(e.p->is_supported(1001, &ai), 0u);  // "no adapter": general check
  ai.vk_physical_device = &ai;
  CHECK_EQ(e.p->is_supported(1001, &ai), 7u);
  // settings never change is-supported (60.8)
  e.settings.set(kSetWrongGpuRr, 0);
  CHECK_EQ(e.p->is_supported(1001, nullptr), 0u);
  e.settings.set(kSetWrongGpuRr, 1);
  // runtime absent
  Env n(14, false);
  n.init();
  CHECK_EQ(n.p->is_supported(1001, nullptr), 7u);
  CHECK_EQ(n.p->is_loaded(1001, &b), 0u); CHECK(!b);

  // requirements per structure version
  for (int pass = 0; pass < 2; ++pass) {
    e.rt.status_bits = pass ? (WG_RR_RT_READY | WG_RR_RT_MODEL_LOADED) : 0u;  // same before / after model load
    for (uint64_t ver = 1; ver <= 2; ++ver) {
      WgRequirements r;
      memset(&r, 0xcd, sizeof r);
      r.h.version = ver;
      CHECK_EQ(e.p->requirements(1001, &r), 0u);
      CHECK_EQ(r.flags, 2u);
      CHECK_EQ(r.cpu_threads, 0u);
      CHECK_EQ(r.max_viewports, 1u);
      CHECK_EQ(r.required_tag_count, 5u);
      const uint32_t want[5] = {3, 4, 1, 7, 8};
      CHECK(memcmp(r.required_tags, want, sizeof want) == 0);
      CHECK(r.os_required[0] == 0 && r.driver_required[2] == 0 && r.os_detected[0] == 0);
      CHECK(r.h.version == ver && r.h.next == (WgHeader*)(uintptr_t)0xcdcdcdcdcdcdcdcdull);  // header untouched
      bool vk_written = r.vulkan_only[0] == 0 && r.vulkan_only[79] == 0;
      CHECK(ver >= 2 ? vk_written : (r.vulkan_only[0] == 0xcd && r.vulkan_only[79] == 0xcd));
    }
  }
  // version: the game's version as passed to <init>, WrongGPU's release version
  WgFeatureVersion fv;
  memset(&fv, 0, sizeof fv);
  CHECK_EQ(e.p->version(1001, &fv), 0u);
  CHECK(fv.api_version[0] == 2 && fv.api_version[1] == 14 && fv.api_version[2] == 1);
  CHECK(fv.runtime_version[0] == 0 && fv.runtime_version[1] == 5 && fv.runtime_version[2] == 3);
  // get-function
  void* fp = nullptr;
  CHECK_EQ(e.p->get_function(1001, "fnOptimal", &fp), 0u); CHECK(fp == (void*)1);
  CHECK_EQ(e.p->get_function(1001, "fnState", &fp), 0u); CHECK(fp == (void*)2);
  CHECK_EQ(e.p->get_function(1001, "fnOptions", &fp), 0u); CHECK(fp == (void*)3);
  CHECK_EQ(e.p->get_function(1001, "fnOther", &fp), 25u);
  CHECK_EQ(e.p->get_function(1001, nullptr, &fp), 25u);
  // rr-state: 0 without an instance
  WgRrState st;
  st.estimated_vram = 77;
  CHECK_EQ(e.p->rr_state(&e.vp, &st), 0u); CHECK(st.estimated_vram == 0);
}

static void test_rr_optimal_versions() {
  for (uint32_t minor : {7u, 14u}) {
    Env e(minor);
    e.init();
    WgRrOptimal o;
    for (uint32_t mode : {0u, 5u}) {
      memset(&o, 0x11, sizeof o);
      WgRrOptions op = make_opts(mode);
      uint32_t rc = e.p->rr_optimal(&op, &o);
      if (minor == 14) {
        CHECK_EQ(rc, 25u);
      } else {
        CHECK_EQ(rc, 0u);
        CHECK(o.optimal_width == 0 && o.optimal_height == 0 && o.min_width == 0 && o.max_height == 0 &&
              o.optimal_sharpness == 0.0f);
      }
    }
    WgRrOptions op = make_opts(7);
    CHECK_EQ(e.p->rr_optimal(&op, &o), 25u);
    op = make_opts(4, 1, 1);  // round(1/3) = 0: zero render size
    CHECK_EQ(e.p->rr_optimal(&op, &o), minor == 14 ? 25u : 0u);
    op = make_opts(3, 3840, 2160);
    CHECK_EQ(e.p->rr_optimal(&op, &o), 0u);
    CHECK(o.optimal_width == 2560 && o.optimal_height == 1440 && o.min_width == 1280 && o.min_height == 720 &&
          o.max_width == 3840 && o.max_height == 2160 && o.optimal_sharpness == 0.0f);
    op = make_opts(6, 1920, 1080);
    CHECK_EQ(e.p->rr_optimal(&op, &o), 0u);
    CHECK(o.optimal_width == 1920 && o.min_width == 640 && o.max_width == 1920);
  }
}

static void test_options_and_viewports() {
  Env e;
  e.init();
  e.p->set_device(e.dev);
  WgRrOptions bad = make_opts(7);
  CHECK_EQ(e.p->rr_options(&e.vp, &bad), 25u);
  bad = make_opts(2, 0, 1080);
  CHECK_EQ(e.p->rr_options(&e.vp, &bad), 25u);
  bad = make_opts(2, 1920, 0xffffffffu);
  CHECK_EQ(e.p->rr_options(&e.vp, &bad), 25u);
  CHECK(!e.p->holds_viewport());  // nothing recorded
  CHECK_EQ(e.p->rr_options(&e.vp, &e.opts), 0u);
  CHECK(e.p->holds_viewport() && e.p->held_viewport() == 7);
  WgViewport other = make_vp(8);
  WgRrOptions o2 = make_opts(3);
  CHECK_EQ(e.p->rr_options(&other, &o2), 38u);
  CHECK_EQ(e.p->allocate(nullptr, 1001, &other), 38u);
  const WgHeader* in[1] = {&other.h};
  CHECK_EQ(e.p->evaluate(1001, nullptr, in, 1, e.cmd), 38u);
  CHECK_EQ(e.p->log_count("second viewport"), 1u);  // logged once per proxy instance
  CHECK_EQ(e.p->held_viewport(), 7u);
  CHECK_EQ(e.p->free_resources(1001, &other), 0u);  // not held: 0, nothing
  CHECK_EQ(e.p->held_viewport(), 7u);
  CHECK_EQ(e.p->free_resources(1001, &e.vp), 0u);
  CHECK(!e.p->holds_viewport());
  CHECK_EQ(e.p->free_resources(1001, &e.vp), 0u);   // none held
  CHECK_EQ(e.p->rr_options(&other, &o2), 0u);       // a new viewport after free
  CHECK_EQ(e.p->held_viewport(), 8u);
}

static void test_evaluate_happy_and_lifecycle() {
  Env e;
  e.setup();
  CHECK_EQ(e.frame(0), 0u);
  CHECK_EQ(e.rt.creates, 1);
  const WgRrCreate& c = e.rt.created.back();
  CHECK(c.output_width == 1920 && c.output_height == 1080 && c.render_width == 1280 && c.render_height == 720);
  CHECK(c.mode == 2 && c.packing == 0 && c.colour_hdr == 1 && c.depth_inverted == 0 && c.mv_jittered == 0);
  CHECK(c.hw_depth == 1 && c.alpha_upscaling == 0 && c.mv_output_resolution == 0 && c.api_minor == 14);
  CHECK(c.device == reinterpret_cast<uint8_t*>(e.dev) + 1);
  const WgRrFrame& f0 = e.rt.last();
  CHECK(f0.render_width == 1280 && f0.render_height == 720 && f0.output_width == 1920 && f0.output_height == 1080);
  CHECK(f0.reset == 1 && f0.parity == 0 && f0.counter == 0 && f0.rebuild_render_items == 0);
  CHECK(f0.prev_jitter[0] == 0.0f && f0.jitter[0] == 0.25f && f0.jitter[1] == -0.25f);
  CHECK(f0.mv_scale[0] == (1.0f / 1280.0f) * 1280.0f && f0.mv_scale[1] == (1.0f / 720.0f) * 720.0f);
  CHECK(f0.command_list == reinterpret_cast<uint8_t*>(e.cmd) + 1);
  CHECK(memcmp(f0.prev_view_to_clip, f0.view_to_clip, 64) == 0);  // reset frame: current pair used
  CHECK(f0.world_to_view[12] == 3.0f && f0.view_to_clip[0] == 2.0f);
  CHECK(f0.images[WG_ROLE_COLOUR].resource == fake_res(0) && f0.images[WG_ROLE_COLOUR].buffer_type == 3);
  CHECK(f0.images[WG_ROLE_DEPTH].buffer_type == 0 && f0.depth_is_linear == 0);
  CHECK(f0.images[WG_ROLE_ROUGHNESS].resource == fake_res(7));
  CHECK(f0.images[WG_ROLE_COLOUR].state == 4 && f0.images[WG_ROLE_COLOUR].format_class == WG_FMT_FLOAT);
  uint32_t req = (1u << WG_ROLE_COLOUR) | (1u << WG_ROLE_OUTPUT) | (1u << WG_ROLE_DEPTH) | (1u << WG_ROLE_MOTION) |
                 (1u << WG_ROLE_DIFFUSE_ALBEDO) | (1u << WG_ROLE_SPECULAR_ALBEDO) | (1u << WG_ROLE_NORMALS) |
                 (1u << WG_ROLE_ROUGHNESS);
  CHECK_EQ(f0.attached, req);
  // frame 2: no reset; parity / counter advanced; pj = previous jitter; CP = previous pair
  e.consts = make_consts(-0.5f, 0.125f);
  e.consts.view_to_clip[1] = 9.0f;
  e.opts.world_to_view[13] = 4.0f;
  e.p->rr_options(&e.vp, &e.opts);  // a non-trigger change of options: no recreate
  CHECK_EQ(e.frame(1), 0u);
  const WgRrFrame& f1 = e.rt.last();
  CHECK(f1.reset == 0 && f1.parity == 1 && f1.counter == 1);
  CHECK(f1.prev_jitter[0] == 0.25f && f1.prev_jitter[1] == -0.25f && f1.jitter[0] == -0.5f);
  CHECK(f1.prev_view_to_clip[1] == 0.0f && f1.view_to_clip[1] == 9.0f && f1.world_to_view[13] == 4.0f &&
        f1.prev_world_to_view[13] == 0.0f);
  CHECK_EQ(e.rt.creates, 1);
  for (int i = 2; i < 10; ++i) CHECK_EQ(e.frame(i), 0u);
  CHECK_EQ(e.rt.last().counter, 1u);  // 9 completed frames before: 9 mod 8
  CHECK_EQ(e.rt.last().parity, 1u);
  // game reset: only the value 1 resets
  for (uint8_t r : {0, 2, 1}) {
    e.consts.reset = r;
    CHECK_EQ(e.frame(20 + r), 0u);
    CHECK_EQ(e.rt.last().reset, r == 1 ? 1u : 0u);
  }
  // rr-state with an instance
  WgRrState st;
  CHECK_EQ(e.p->rr_state(&e.vp, &st), 0u);
  CHECK(st.estimated_vram == 1000000ull + 0x1000);
  // status: active after a completed evaluate, < 1 s
  CHECK_EQ(e.p->status_word(), 7u);
  e.now += 999;
  CHECK_EQ(e.p->status_word(), 7u);
  e.now += 1;
  CHECK_EQ(e.p->status_word(), 3u);
}

static void test_creation_decisions() {
  for (uint8_t v : {0, 1, 2}) {
    Env e;
    e.opts.colour_is_hdr = v;
    e.consts.depth_inverted = v;
    e.consts.mv_jittered = v;
    e.opts.alpha_upscaling = v;
    e.setup();
    CHECK_EQ(e.frame(0), 0u);
    const WgRrCreate& c = e.rt.created.back();
    CHECK_EQ(c.colour_hdr, v == 1 ? 1u : 0u);
    CHECK_EQ(c.depth_inverted, v == 1 ? 1u : 0u);
    CHECK_EQ(c.mv_jittered, v == 1 ? 1u : 0u);
    CHECK_EQ(c.alpha_upscaling, v == 1 ? 1u : 0u);
    // a later change of these inputs alone has no effect (no recreate)
    e.consts.depth_inverted = 1;
    e.opts.colour_is_hdr = 0;
    e.opts.sharpness = 0.5f;
    e.opts.pre_exposure = 3.0f;
    e.p->rr_options(&e.vp, &e.opts);
    CHECK_EQ(e.frame(1), 0u);
    CHECK_EQ(e.rt.creates, 1);
    // camera motion included: 1 and "not set" accepted, 0 rejected (61.4 step 4)
    e.consts.camera_motion_included = v;
    uint32_t before = (uint32_t)e.rt.evaluates;
    CHECK_EQ(e.frame(2), v == 0 ? 25u : 0u);
    CHECK_EQ((uint32_t)e.rt.evaluates, before + (v == 0 ? 0u : 1u));
  }
  // alpha upscaling read only with options version >= 2
  Env a;
  a.opts = make_opts(2, 1920, 1080, 1);
  a.opts.alpha_upscaling = 1;
  a.setup();
  CHECK_EQ(a.frame(0), 0u);
  CHECK_EQ(a.rt.created.back().alpha_upscaling, 0u);
  // sharpness > 0 sets the (unused) sharpening request
  Env s;
  s.opts.sharpness = 0.25f;
  s.setup();
  CHECK_EQ(s.frame(0), 0u);
  CHECK_EQ(s.rt.created.back().sharpen_request, 1u);
  // packing 1: the combined input (type 14) is required instead of 5 + 6
  Env pk;
  pk.opts.normal_roughness_packing = 1;
  pk.setup();
  CHECK_EQ(pk.frame(0), 20u);
  Res nr{9, 1280, 720, DXGI_FORMAT_R16G16B16A16_FLOAT};
  pk.retag(make_tag(&nr.r, WG_BT_NORMAL_ROUGHNESS));
  CHECK_EQ(pk.frame(1), 0u);
  CHECK(pk.rt.last().images[WG_ROLE_NORMALS].buffer_type == 14 && !(pk.rt.last().attached & (1u << WG_ROLE_ROUGHNESS)));
  // depth: linear wins; hw_depth decided at creation
  Env d;
  Res lin{10, 1280, 720, DXGI_FORMAT_R32_FLOAT};
  d.tags.push_back(make_tag(&lin.r, WG_BT_LINEAR_DEPTH));
  d.setup();
  CHECK_EQ(d.frame(0), 0u);
  CHECK(d.rt.created.back().hw_depth == 0 && d.rt.last().depth_is_linear == 1 &&
        d.rt.last().images[WG_ROLE_DEPTH].buffer_type == 49);
  Env nd;
  nd.tags.erase(nd.tags.begin() + 3);  // no depth at all: 20 at creation
  nd.setup();
  CHECK_EQ(nd.frame(0), 20u);
  CHECK_EQ(nd.rt.creates, 0);
  CHECK_EQ(nd.p->log_count("no depth"), 1u);
}

static void test_recreate_triggers() {
  struct Case { const char* name; void (*change)(WgRrOptions&); bool recreate; };
  const Case cases[] = {
    {"mode", [](WgRrOptions& o) { o.mode = 3; }, true},
    {"width", [](WgRrOptions& o) { o.output_width = 2560; }, true},
    {"height", [](WgRrOptions& o) { o.output_height = 1440; }, true},
    {"packing", [](WgRrOptions& o) { o.normal_roughness_packing = 1; }, true},
    {"preset v3", [](WgRrOptions& o) { o.presets[2] = 6; }, true},
    {"preset v2", [](WgRrOptions& o) { o.presets[2] = 6; o.h.version = 2; }, false},
    {"hdr", [](WgRrOptions& o) { o.colour_is_hdr = 0; }, false},
    {"sharpness", [](WgRrOptions& o) { o.sharpness = 1.0f; }, false},
    {"exposure", [](WgRrOptions& o) { o.pre_exposure = 2.0f; o.exposure_scale = 4.0f; }, false},
    {"alpha", [](WgRrOptions& o) { o.alpha_upscaling = 1; }, false},
  };
  for (const Case& k : cases) {
    Env e;
    Res nr{9, 1280, 720, DXGI_FORMAT_R16G16B16A16_FLOAT};
    e.tags.push_back(make_tag(&nr.r, WG_BT_NORMAL_ROUGHNESS));
    e.setup();
    CHECK_EQ(e.frame(0), 0u);
    CHECK_EQ(e.frame(1), 0u);
    k.change(e.opts);
    CHECK_EQ(e.p->rr_options(&e.vp, &e.opts), 0u);
    uint32_t rc = e.frame(2);
    if (rc != 0) fprintf(stderr, "case %s rc %u\n", k.name, rc);
    CHECK_EQ(rc, 0u);
    CHECK_EQ(e.rt.creates, k.recreate ? 2 : 1);
    CHECK_EQ(e.rt.destroys, k.recreate ? 1 : 0);
    CHECK_EQ(e.rt.last().reset, k.recreate ? 1u : 0u);   // a recreate makes a new instance: reset frame
    CHECK_EQ(e.rt.last().parity, k.recreate ? 0u : 0u);   // new: 0; old after two frames: 0
    CHECK_EQ(e.rt.last().counter, k.recreate ? 0u : 2u);
  }
}

static void test_render_size_changes() {
  Env e;
  e.setup();
  CHECK_EQ(e.frame(0), 0u);
  CHECK_EQ(e.frame(1), 0u);
  e.colour.r.width = 1600; e.colour.r.height = 900;  // larger (dynamic resolution)
  CHECK_EQ(e.frame(2), 0u);
  CHECK(e.rt.last().rebuild_render_items == 1 && e.rt.last().reset == 0 && e.rt.last().render_width == 1600);
  CHECK(e.rt.last().counter == 2);  // not changed by the size change
  CHECK_EQ(e.frame(3), 0u);
  CHECK(e.rt.last().rebuild_render_items == 0 && e.rt.last().reset == 1);  // the next frame is a reset frame
  CHECK_EQ(e.frame(4), 0u);
  CHECK_EQ(e.rt.last().reset, 0u);
  e.colour.r.width = 1000; e.colour.r.height = 600;  // smaller
  CHECK_EQ(e.frame(5), 0u);
  CHECK(e.rt.last().rebuild_render_items == 1 && e.rt.last().reset == 0);
  CHECK_EQ(e.rt.creates, 1);  // not a recreate
  // opt-in: the change frame itself is a reset frame
  e.settings.set(kSetChangeFrameReset, 1);
  CHECK_EQ(e.frame(6), 0u);  // (pending from the change: reset)
  e.colour.r.width = 1280; e.colour.r.height = 720;
  CHECK_EQ(e.frame(7), 0u);
  CHECK(e.rt.last().rebuild_render_items == 1 && e.rt.last().reset == 1);
  CHECK_EQ(e.frame(8), 0u);
  CHECK_EQ(e.rt.last().reset, 0u);
  // created by <allocate> without a colour tag: render size unknown -> the first evaluate rebuilds; frames 1, 2 reset
  Env a;
  a.init();
  a.p->set_device(a.dev);
  a.p->rr_options(&a.vp, &a.opts);
  WgTag dt = make_tag(&a.depth.r, WG_BT_DEPTH);
  a.p->tag(&a.vp, &dt, 1, a.cmd);
  CHECK_EQ(a.p->allocate(a.cmd, 1001, &a.vp), 0u);
  CHECK(a.rt.creates == 1 && a.rt.created.back().render_width == 0);
  a.p->tag(&a.vp, a.tags.data(), (uint32_t)a.tags.size(), a.cmd);
  CHECK_EQ(a.frame(0), 0u);
  CHECK(a.rt.last().rebuild_render_items == 1 && a.rt.last().reset == 1);
  CHECK_EQ(a.frame(1), 0u);
  CHECK_EQ(a.rt.last().reset, 1u);
  CHECK_EQ(a.frame(2), 0u);
  CHECK_EQ(a.rt.last().reset, 0u);
  CHECK_EQ(a.p->allocate(a.cmd, 1001, &a.vp), 0u);  // existing instance, no trigger: nothing created
  CHECK_EQ(a.rt.creates, 1);
}

static void test_rejections() {
  // zero size: 25, no pass, a reset becomes pending
  Env e;
  e.setup();
  CHECK_EQ(e.frame(0), 0u);
  CHECK_EQ(e.frame(1), 0u);
  int ev = e.rt.evaluates;
  e.colour.r.width = 0;
  CHECK_EQ(e.frame(2), 25u);
  CHECK_EQ(e.rt.evaluates, ev);
  CHECK(e.p->lifecycle()->pending_reset() && e.p->lifecycle()->parity() == 0 && e.p->lifecycle()->counter() == 2);
  e.colour.r.width = 1280;
  CHECK_EQ(e.frame(3), 0u);
  CHECK_EQ(e.rt.last().reset, 1u);
  CHECK(!e.p->lifecycle()->pending_reset());
  // render-scale window: min - 1, min, output, output + 1 (per axis)
  struct S { uint32_t w, h, rc; };
  const uint32_t mw = min_render_size(1920), mh = min_render_size(1080);
  CHECK(mw == 640 && mh == 360);
  const S sizes[] = {{mw - 1, 720, 25}, {mw, 720, 0}, {1920, 720, 0}, {1921, 720, 25},
                     {1280, mh - 1, 25}, {1280, mh, 0}, {1280, 1080, 0}, {1280, 1081, 25}};
  Env r;
  r.setup();
  CHECK_EQ(r.frame(0), 0u);
  for (const S& s : sizes) {
    r.colour.r.width = s.w;
    r.colour.r.height = s.h;
    bool pending_before = r.p->lifecycle()->pending_reset();
    uint32_t par = r.p->lifecycle()->parity();
    int evs = r.rt.evaluates;
    CHECK_EQ(r.frame(1), s.rc);
    if (s.rc) {
      CHECK_EQ(r.rt.evaluates, evs);
      CHECK(r.p->lifecycle()->pending_reset() == pending_before && r.p->lifecycle()->parity() == par);
    }
  }
  CHECK_EQ(r.p->log_count("outside 1/3..1"), 1u);  // logged once per instance
  // extent (sub-rectangle) gives the render size; extent 0 = whole resource
  Env x;
  x.tags[0].extent = {16, 8, 1100, 650};
  x.tags[2].extent = {0, 0, 1100, 650};  // motion vectors at the same (render) size
  x.setup();
  CHECK_EQ(x.frame(0), 0u);
  CHECK(x.rt.last().render_width == 1100 && x.rt.last().render_height == 650);
  CHECK(x.rt.last().images[WG_ROLE_COLOUR].origin_x == 8 && x.rt.last().images[WG_ROLE_COLOUR].origin_y == 16);
  // camera motion not included: 25 before any pass, logged once per instance, no pending reset
  Env cm;
  cm.setup();
  CHECK_EQ(cm.frame(0), 0u);
  cm.consts.camera_motion_included = 0;
  CHECK_EQ(cm.frame(1), 25u);
  CHECK_EQ(cm.frame(2), 25u);
  CHECK_EQ(cm.p->log_count("without camera motion"), 1u);
  CHECK(!cm.p->lifecycle()->pending_reset());
  // output-resolution motion vectors: every evaluate 25, nothing created, logged once
  Env mo;
  mo.mv.r.width = 1920;  // larger in width only
  mo.setup();
  CHECK_EQ(mo.frame(0), 25u);
  CHECK_EQ(mo.frame(1), 25u);
  CHECK_EQ(mo.rt.creates, 0);
  CHECK_EQ(mo.p->log_count("output-resolution motion"), 1u);
  // created by allocate for output-resolution vectors: the instance's evaluates fail with 25
  Env ma;
  ma.mv.r.height = 1080;
  ma.setup();
  CHECK_EQ(ma.p->allocate(ma.cmd, 1001, &ma.vp), 0u);
  CHECK_EQ(ma.rt.created.back().mv_output_resolution, 1u);
  CHECK_EQ(ma.frame(0), 25u);
  CHECK_EQ(ma.rt.evaluates, 0);
  // no constants for the frame: 26
  Env nc;
  nc.setup();
  WgFrameToken* t = nc.token(0);
  const WgHeader* in[1] = {&nc.vp.h};
  CHECK_EQ(nc.p->evaluate(1001, t, in, 1, nc.cmd), 26u);
  // missing viewport in the input array: 20; mode off: 19
  CHECK_EQ(nc.p->evaluate(1001, t, nullptr, 0, nc.cmd), 20u);
  WgRrOptions off = make_opts(0);
  CHECK_EQ(nc.p->rr_options(&nc.vp, &off), 0u);
  CHECK_EQ(nc.frame(1), 19u);
  // every required input missing in turn: 20
  for (size_t i = 0; i < 8; ++i) {
    Env m;
    WgTag removed = m.tags[i];
    m.tags.erase(m.tags.begin() + i);
    m.setup();
    CHECK_EQ(m.frame(0), 20u);
    (void)removed;
  }
}

static void test_failed_evaluates() {
  Env e;
  e.setup();
  // first frame fails after its passes began, past the input stage: pending reset kept, pj replaced
  e.rt.evaluate_rc = 22;
  e.rt.next_result.passes_started = 1;
  e.rt.next_result.input_jitter_consumed = 1;
  CHECK_EQ(e.frame(0), 22u);
  const Lifecycle* lc = e.p->lifecycle();
  CHECK(lc && lc->pending_reset() && lc->parity() == 0 && lc->counter() == 0);
  CHECK(lc->prev_jitter()[0] == 0.25f);
  float zero[16] = {0};
  CHECK(memcmp(lc->previous_camera().v2c, zero, sizeof zero) == 0);  // failed reset frame: CP unchanged
  CHECK_EQ(e.p->status_word() & 4u, 0u);
  e.rt.evaluate_rc = 0;
  e.consts = make_consts(0.75f, 0.5f);
  CHECK_EQ(e.frame(1), 0u);
  CHECK(e.rt.last().reset == 1 && e.rt.last().prev_jitter[0] == 0.25f);  // still a reset frame
  CHECK(!lc->pending_reset() && lc->parity() == 1);
  // failure before the input stage used pj: pj kept
  e.rt.evaluate_rc = 22;
  e.rt.next_result.passes_started = 1;
  e.rt.next_result.input_jitter_consumed = 0;
  e.consts = make_consts(-1.0f, -1.0f);
  CHECK_EQ(e.frame(2), 22u);
  CHECK(lc->prev_jitter()[0] == 0.75f && lc->parity() == 1 && lc->counter() == 1 && !lc->pending_reset());
  // the next call runs with its own reset flag (no forced reset by default)
  e.rt.evaluate_rc = 0;
  CHECK_EQ(e.frame(3), 0u);
  CHECK_EQ(e.rt.last().reset, 0u);
  CHECK(e.rt.last().prev_view_to_clip[0] == 2.0f);
  // opt-in: reset after a failed evaluate
  e.settings.set(kSetResetAfterFailedEvaluate, 1);
  e.rt.evaluate_rc = 22;
  CHECK_EQ(e.frame(4), 22u);
  e.rt.evaluate_rc = 0;
  CHECK_EQ(e.frame(5), 0u);
  CHECK_EQ(e.rt.last().reset, 1u);
  // camera fallback reported by the runtime: logged once per instance
  e.rt.next_result.camera_fallback_used = 1;
  CHECK_EQ(e.frame(6), 0u);
  CHECK_EQ(e.frame(7), 0u);
  CHECK_EQ(e.p->log_count("camera matrices unusable"), 1u);
}

static void test_settings_snapshot_and_rr_switch() {
  Env e;
  e.setup();
  e.settings.set(kSetWrongGpuRr, 0);
  CHECK_EQ(e.frame(0), 7u);  // off at creation: 7 before any pass, nothing created
  CHECK_EQ(e.frame(1), 7u);
  CHECK_EQ(e.rt.creates, 0);
  CHECK_EQ(e.p->log_count("is off"), 1u);
  CHECK_EQ(e.p->allocate(e.cmd, 1001, &e.vp), 7u);
  CHECK_EQ(e.p->allocate(e.cmd, 1001, &e.vp), 7u);
  CHECK_EQ(e.p->log_count("is off"), 2u);  // once for evaluate, once for allocate
  CHECK(!e.p->has_instance());
  e.settings.set(kSetWrongGpuRr, 1);
  CHECK_EQ(e.frame(2), 0u);
  CHECK_EQ(e.rt.creates, 1);
  // switched off while an instance exists: it keeps running until the game next creates the feature
  e.settings.set(kSetWrongGpuRr, 0);
  CHECK_EQ(e.frame(3), 0u);
  e.opts.mode = 3;
  e.p->rr_options(&e.vp, &e.opts);
  CHECK_EQ(e.frame(4), 7u);  // recreate trigger while off
  CHECK_EQ(e.rt.creates, 1);
  e.settings.set(kSetWrongGpuRr, 1);
  CHECK_EQ(e.frame(5), 0u);
  CHECK_EQ(e.rt.creates, 2);
  CHECK_EQ(e.rt.created.back().latched_settings, 1u);
  // allocate with the setting on creates and latches
  Env a;
  a.setup();
  CHECK_EQ(a.p->allocate(a.cmd, 1001, &a.vp), 0u);
  CHECK_EQ(a.rt.creates, 1);
  a.settings.set(kSetWrongGpuRr, 0);
  CHECK_EQ(a.frame(0), 0u);  // the instance exists: runs
  // a set during an evaluate applies from the next evaluate
  Env s;
  s.setup();
  s.rt.on_evaluate = [&s] { s.settings.set(kSetSpecularMotionFromV, 1); };
  CHECK_EQ(s.frame(0), 0u);
  CHECK_EQ(s.rt.last().settings & WG_RR_SET_SPECULAR_MOTION_FROM_V, 0u);
  s.rt.on_evaluate = nullptr;
  CHECK_EQ(s.frame(1), 0u);
  CHECK_EQ(s.rt.last().settings & WG_RR_SET_SPECULAR_MOTION_FROM_V, WG_RR_SET_SPECULAR_MOTION_FROM_V);
  s.settings.set(kSetBoundedStores, 1);
  CHECK_EQ(s.frame(2), 0u);
  CHECK_EQ(s.rt.last().settings, WG_RR_SET_SPECULAR_MOTION_FROM_V | WG_RR_SET_BOUNDED_STORES);
  // runtime model not loaded at creation: 7
  Env m;
  m.setup();
  m.rt.status_bits = WG_RR_RT_READY;
  CHECK_EQ(m.frame(0), 7u);
}

// clean/060-rr-plugin-hunt: create_or_update runs at every evaluate while the viewport has no instance, so its refusals
// are logged once, like the others (a game evaluating every frame without a usable runtime or a depth input)
static void test_refusals_logged_once() {
  {
    Env m;
    m.setup();
    m.rt.status_bits = WG_RR_RT_READY;  // model not loaded
    for (uint32_t i = 0; i < 5; ++i) CHECK_EQ(m.frame(i), 7u);
    CHECK_EQ(m.p->allocate(m.cmd, 1001, &m.vp), 7u);
    CHECK_EQ(m.p->log_count("runtime or model is not available (7)"), 1u);
    m.rt.status_bits = WG_RR_RT_READY | WG_RR_RT_MODEL_LOADED;  // the runtime becomes usable: created as before
    CHECK_EQ(m.frame(5), 0u);
    CHECK_EQ(m.rt.creates, 1);
  }
  {
    Env d;
    std::vector<WgTag> no_depth;
    for (const WgTag& t : d.tags)
      if (t.buffer_type != WG_BT_DEPTH && t.buffer_type != WG_BT_LINEAR_DEPTH) no_depth.push_back(t);
    const std::vector<WgTag> all = d.tags;
    d.tags = no_depth;
    d.setup();
    for (uint32_t i = 0; i < 5; ++i) CHECK_EQ(d.frame(i), 20u);
    CHECK_EQ(d.p->log_count("no depth input tagged at feature creation (20)"), 1u);
    CHECK_EQ(d.rt.creates, 0);
    d.tags = all;  // the game tags its depth: created as before
    CHECK_EQ(d.frame(5), 0u);
    CHECK_EQ(d.rt.creates, 1);
  }
}

static void test_precedence() {
  // no viewport + mode off -> 20
  { Env e; e.opts.mode = 0; e.setup();
    WgFrameToken* t = e.token(0);
    CHECK_EQ(e.p->evaluate(1001, t, nullptr, 0, e.cmd), 20u); }
  // second viewport + mode off -> 38
  { Env e; e.setup(); WgViewport o = make_vp(99); e.vp = o; CHECK_EQ(e.frame(0), 38u); }
  // mode off + RR off + missing inputs -> 19
  { Env e; e.opts.mode = 0; e.tags.clear(); e.setup(); e.settings.set(kSetWrongGpuRr, 0); CHECK_EQ(e.frame(0), 19u); }
  // RR off + missing inputs -> 7
  { Env e; e.tags.resize(2); e.setup(); e.settings.set(kSetWrongGpuRr, 0); CHECK_EQ(e.frame(0), 7u); }
  // missing inputs + zero size + camera motion off -> 20
  { Env e; e.tags.erase(e.tags.begin() + 4); e.colour.r.width = 0; e.consts.camera_motion_included = 0; e.setup();
    CHECK_EQ(e.frame(0), 20u); }
  // zero size + camera motion off + bad format -> 25 by the zero-size rule (reset pending)
  { Env e; e.setup(); CHECK_EQ(e.frame(0), 0u);
    e.colour.r.width = 0; e.consts.camera_motion_included = 0; e.colour.r.native_format = DXGI_FORMAT_R32_UINT;
    CHECK_EQ(e.frame(1), 25u); CHECK(e.p->lifecycle()->pending_reset());
    CHECK_EQ(e.p->log_count("zero render"), 1u); CHECK_EQ(e.p->log_count("unsupported format"), 0u); }
  // range + camera motion off -> 25 by the range rule (logged as range, no reset pending)
  { Env e; e.setup(); CHECK_EQ(e.frame(0), 0u);
    e.colour.r.width = 2000; e.consts.camera_motion_included = 0;
    CHECK_EQ(e.frame(1), 25u); CHECK(!e.p->lifecycle()->pending_reset());
    CHECK_EQ(e.p->log_count("outside 1/3..1"), 1u); CHECK_EQ(e.p->log_count("without camera motion"), 0u); }
  // camera motion off + output-resolution vectors + bad format -> logged as camera motion
  { Env e; e.mv.r.width = 1920; e.consts.camera_motion_included = 0; e.colour.r.native_format = DXGI_FORMAT_BC7_UNORM;
    e.setup(); CHECK_EQ(e.frame(0), 25u);
    CHECK_EQ(e.p->log_count("without camera motion"), 1u); CHECK_EQ(e.p->log_count("output-resolution"), 0u); }
  // output-resolution vectors + bad format -> logged as output-resolution
  { Env e; e.mv.r.width = 1920; e.colour.r.native_format = DXGI_FORMAT_BC7_UNORM; e.setup(); CHECK_EQ(e.frame(0), 25u);
    CHECK_EQ(e.p->log_count("output-resolution"), 1u); CHECK_EQ(e.p->log_count("unsupported format"), 0u); }
  // before init beats everything; before set-device beats viewport problems
  { Env e; WgFrameToken* t = nullptr; CHECK_EQ(e.p->evaluate(1001, t, nullptr, 0, nullptr), 23u);
    e.init(); CHECK_EQ(e.p->evaluate(1001, t, nullptr, 0, nullptr), 5u); }
}

static void test_routing() {
  Res g58{20, 1280, 720, DXGI_FORMAT_R8G8B8A8_UNORM}, g41{21, 1280, 720, DXGI_FORMAT_R11G11B10_FLOAT};
  Res l51{22, 1920, 1080, DXGI_FORMAT_R16G16B16A16_FLOAT}, l39{23, 1920, 1080, DXGI_FORMAT_R16G16B16A16_FLOAT};
  Res op52{24, 1920, 1080, DXGI_FORMAT_R8G8B8A8_UNORM}, sm10{25, 1280, 720, DXGI_FORMAT_R16G16_FLOAT};
  Res hd42{26, 1280, 720, DXGI_FORMAT_R16_FLOAT}, ex13{27, 1, 1, DXGI_FORMAT_R32_UINT};
  Res a34{28, 1280, 720, DXGI_FORMAT_R8_UNORM}, a67{29, 1920, 1080, DXGI_FORMAT_R8_UNORM};
  Res dis11{30, 1280, 720, DXGI_FORMAT_R8_UNORM}, dof64{31, 1280, 720, DXGI_FORMAT_R16_FLOAT};
  Res bt40{32, 1280, 720, DXGI_FORMAT_R16G16B16A16_FLOAT}, rm68{33, 1280, 720, DXGI_FORMAT_R8_UINT};
  // 58 alone -> G1 = 58; 41 attached -> G1 = 41 (58 neither used nor format-checked)
  { Env e; e.tags.push_back(make_tag(&g58.r, WG_BT_SSS_GUIDE)); e.setup(); CHECK_EQ(e.frame(0), 0u);
    CHECK(e.rt.last().images[WG_ROLE_G1].buffer_type == 58); }
  { Env e; g58.r.native_format = DXGI_FORMAT_R32_UINT;
    e.tags.push_back(make_tag(&g58.r, WG_BT_SSS_GUIDE));
    e.tags.push_back(make_tag(&g41.r, WG_BT_COLOUR_BEFORE_FOG, WG_LIFE_UNTIL_PRESENT, {5, 6, 0, 0}));
    e.setup(); CHECK_EQ(e.frame(0), 0u);
    CHECK(e.rt.last().images[WG_ROLE_G1].buffer_type == 41 && e.rt.last().images[WG_ROLE_G1].origin_x == 6);
    g58.r.native_format = DXGI_FORMAT_R8G8B8A8_UNORM; }
  // 39 with 51: layer 39, origin of 51; 39 alone: origin (0, 0); 51 alone: layer 51
  { Env e; e.tags.push_back(make_tag(&l51.r, WG_BT_TRANSPARENCY_LAYER, 1, {3, 4, 0, 0}));
    e.tags.push_back(make_tag(&l39.r, WG_BT_COLOUR_BEFORE_PARTICLES, 1, {7, 9, 0, 0}));
    e.tags.push_back(make_tag(&op52.r, WG_BT_TRANSPARENCY_OPACITY));
    e.setup(); CHECK_EQ(e.frame(0), 0u);
    const WgRrFrame& f = e.rt.last();
    CHECK(f.images[WG_ROLE_LAYER].buffer_type == 39 && f.layer_origin_x == 4 && f.layer_origin_y == 3);
    CHECK(f.attached & (1u << WG_ROLE_OPACITY)); }
  { Env e; e.tags.push_back(make_tag(&l39.r, WG_BT_COLOUR_BEFORE_PARTICLES, 1, {7, 9, 0, 0}));
    e.setup(); CHECK_EQ(e.frame(0), 0u);
    CHECK(e.rt.last().images[WG_ROLE_LAYER].buffer_type == 39 && e.rt.last().layer_origin_x == 0 &&
          e.rt.last().layer_origin_y == 0); }
  { Env e; e.tags.push_back(make_tag(&l51.r, WG_BT_TRANSPARENCY_LAYER)); e.setup(); CHECK_EQ(e.frame(0), 0u);
    CHECK(e.rt.last().images[WG_ROLE_LAYER].buffer_type == 51); }
  // 52 without a layer: no effect, never format-checked
  { Env e; op52.r.native_format = DXGI_FORMAT_BC1_UNORM; e.tags.push_back(make_tag(&op52.r, WG_BT_TRANSPARENCY_OPACITY));
    e.setup(); CHECK_EQ(e.frame(0), 0u); CHECK(!(e.rt.last().attached & (1u << WG_ROLE_OPACITY)));
    op52.r.native_format = DXGI_FORMAT_R8G8B8A8_UNORM; }
  // 42 with and without 10
  { Env e; e.tags.push_back(make_tag(&hd42.r, WG_BT_SPECULAR_HIT_DISTANCE)); e.setup(); CHECK_EQ(e.frame(0), 0u);
    CHECK((e.rt.last().attached & (1u << WG_ROLE_SPECULAR_HIT_DIST)) && !(e.rt.last().attached & (1u << WG_ROLE_SPECULAR_MOTION))); }
  { Env e; e.tags.push_back(make_tag(&hd42.r, WG_BT_SPECULAR_HIT_DISTANCE));
    e.tags.push_back(make_tag(&sm10.r, WG_BT_SPECULAR_MOTION)); e.setup(); CHECK_EQ(e.frame(0), 0u);
    CHECK((e.rt.last().attached & (1u << WG_ROLE_SPECULAR_HIT_DIST)) && (e.rt.last().attached & (1u << WG_ROLE_SPECULAR_MOTION))); }
  // 13 (exposure), 68 (responsivity) and other "not used" types: ignored, never checked
  { Env e; e.tags.push_back(make_tag(&ex13.r, WG_BT_EXPOSURE)); e.tags.push_back(make_tag(&rm68.r, WG_BT_RESPONSIVITY));
    e.opts.pre_exposure = 7.0f; e.opts.exposure_scale = 0.1f;
    e.setup(); CHECK_EQ(e.frame(0), 0u); CHECK_EQ(e.rt.last().attached, 0xffu); }
  // 11, 64, 40 handed on when present
  { Env e; e.tags.push_back(make_tag(&dis11.r, WG_BT_DISOCCLUSION)); e.tags.push_back(make_tag(&dof64.r, WG_BT_DOF_GUIDE));
    e.tags.push_back(make_tag(&bt40.r, WG_BT_COLOUR_BEFORE_TRANSPARENCY)); e.setup(); CHECK_EQ(e.frame(0), 0u);
    uint32_t want = (1u << WG_ROLE_DISOCCLUSION) | (1u << WG_ROLE_DOF_GUIDE) | (1u << WG_ROLE_BEFORE_TRANSPARENCY);
    CHECK_EQ(e.rt.last().attached & want, want); }
  // 34 / 67: 2.14.1 with alpha upscaling only
  for (uint32_t minor : {7u, 14u})
    for (uint8_t alpha : {0, 1}) {
      Env e(minor);
      e.opts.alpha_upscaling = alpha;
      a34.r.native_format = DXGI_FORMAT_R8_UNORM;
      e.tags.push_back(make_tag(&a34.r, WG_BT_INPUT_ALPHA));
      e.tags.push_back(make_tag(&a67.r, WG_BT_OUTPUT_ALPHA));
      e.setup();
      CHECK_EQ(e.frame(0), 0u);
      bool used = minor == 14 && alpha == 1;
      CHECK_EQ((e.rt.last().attached >> WG_ROLE_INPUT_ALPHA) & 1u, used ? 1u : 0u);
      CHECK_EQ((e.rt.last().attached >> WG_ROLE_OUTPUT_ALPHA) & 1u, used ? 1u : 0u);
      a34.r.native_format = DXGI_FORMAT_R8_UINT;  // unsupported: checked only when used
      CHECK_EQ(e.frame(1), used ? 25u : 0u);
    }
  // precedence: local tag, then frame-scoped (2.14.1), then viewport tag
  for (uint32_t minor : {7u, 14u}) {
    Env e(minor);
    e.setup();
    Res cf{40, 1280, 720, DXGI_FORMAT_R16G16B16A16_FLOAT}, cl{41, 1280, 720, DXGI_FORMAT_R16G16B16A16_FLOAT};
    WgFrameToken* t = e.token(0);
    WgTag ft = make_tag(&cf.r, WG_BT_COLOUR_IN);
    e.p->tag_for_frame(t, &e.vp, &ft, 1, e.cmd);
    e.p->constants(&e.consts, t, &e.vp);
    const WgHeader* in1[1] = {&e.vp.h};
    CHECK_EQ(e.p->evaluate(1001, t, in1, 1, e.cmd), 0u);
    CHECK(e.rt.last().images[WG_ROLE_COLOUR].resource == (minor == 14 ? fake_res(40) : fake_res(0)));
    WgTag lt = make_tag(&cl.r, WG_BT_COLOUR_IN);
    WgTagConversion conv;
    memset(&conv, 0, sizeof conv);
    conv.h.type = kGConv;
    conv.h.version = 1;
    conv.kind = 1;
    conv.scale = 2.0f;
    conv.bias = -1.0f;
    lt.h.next = &conv.h;
    const WgHeader* in2[2] = {&lt.h, &e.vp.h};
    CHECK_EQ(e.p->evaluate(1001, t, in2, 2, e.cmd), 0u);
    const WgRrImage& im = e.rt.last().images[WG_ROLE_COLOUR];
    CHECK(im.resource == fake_res(41) && im.conversion == 1 && im.conversion_scale == 2.0f && im.conversion_bias == -1.0f);
    // the local tag applies to that call only
    CHECK_EQ(e.p->evaluate(1001, t, in1, 1, e.cmd), 0u);
    CHECK(e.rt.last().images[WG_ROLE_COLOUR].resource != fake_res(41));
  }
  // tags removed with a null resource
  { Env e; e.setup(); CHECK_EQ(e.frame(0), 0u);
    e.tags[4].resource = nullptr; CHECK_EQ(e.frame(1), 20u); }
}

static void test_frames_and_tokens() {
  Env e;
  e.setup();
  // repeated, reused and out-of-order frame indices: never rejected
  const uint32_t idx[] = {5, 5, 3, 9, 1, 1, 0};
  for (uint32_t i : idx) CHECK_EQ(e.frame(i), 0u);
  // the same token used twice; a later constants copy replaces the earlier one
  WgFrameToken* t = e.token(42);
  e.p->constants(&e.consts, t, &e.vp);
  WgConstants c2 = make_consts(0.125f, 0.0625f);
  e.p->constants(&c2, t, &e.vp);
  const WgHeader* in[1] = {&e.vp.h};
  CHECK_EQ(e.p->evaluate(1001, t, in, 1, e.cmd), 0u);
  CHECK(e.rt.last().jitter[0] == 0.125f);
  CHECK_EQ(e.p->evaluate(1001, t, in, 1, e.cmd), 0u);
  // clean/rr-re9-pt: per-frame copies keyed by the token's frame index: two token objects of one index are one frame
  // (RE Requiem sets constants through one token object and tags / evaluates through another of the same index)
  WgFrameToken* a = e.token(7);
  WgFrameToken* b = e.token(7);
  CHECK(a != b);
  WgConstants ca = make_consts(1.0f, 0.0f), cb = make_consts(2.0f, 0.0f);
  e.p->constants(&ca, a, &e.vp);
  CHECK_EQ(e.p->evaluate(1001, b, in, 1, e.cmd), 0u);  // constants set through a, evaluate through b
  CHECK(e.rt.last().jitter[0] == 1.0f);
  e.p->constants(&cb, b, &e.vp);                       // a later copy of the same frame replaces it
  CHECK_EQ(e.p->evaluate(1001, a, in, 1, e.cmd), 0u);
  CHECK(e.rt.last().jitter[0] == 2.0f);
  // the next frame through the same token objects: never the previous frame's constants
  WgFrameToken* a8 = e.token(8);
  WgFrameToken* b8 = e.token(8);
  WgConstants c8 = make_consts(3.0f, 0.0f);
  e.p->constants(&c8, a8, &e.vp);
  CHECK_EQ(e.p->evaluate(1001, b8, in, 1, e.cmd), 0u);
  CHECK(e.rt.last().jitter[0] == 3.0f);
  // frame-scoped tags (2.14.1) keyed the same way
  Res c1{50, 1280, 720, DXGI_FORMAT_R16G16B16A16_FLOAT};
  WgTag ft = make_tag(&c1.r, WG_BT_COLOUR_IN);
  e.p->tag_for_frame(a, &e.vp, &ft, 1, e.cmd);
  CHECK_EQ(e.p->evaluate(1001, b, in, 1, e.cmd), 0u);
  CHECK(e.rt.last().images[WG_ROLE_COLOUR].resource == fake_res(50));
  CHECK_EQ(e.p->evaluate(1001, b8, in, 1, e.cmd), 0u);
  CHECK(e.rt.last().images[WG_ROLE_COLOUR].resource == fake_res(0));
  // old frames are evicted beyond the kept window: their constants are gone -> 26
  for (uint32_t i = 0; i < 12; ++i) e.frame(100 + i);
  CHECK_EQ(e.p->evaluate(1001, a, in, 1, e.cmd), 26u);
  // control switch (rr_frame_by_token=1): keyed by the token object as before, so the evaluate through b sees only
  // what was set through b (the RE Requiem symptom: constants of an old frame stay attached to the evaluate's token)
  {
    Settings st;
    st.set_save_enabled(false);
    FakeRuntime rt2;
    ProxyConfig cfg;
    cfg.api_minor = 14;
    cfg.ids.viewport = kGViewport;
    cfg.ids.tag = kGTag;
    cfg.ids.tag_conversion = kGConv;
    cfg.ids.has_tag_conversion = true;
    cfg.version_magic = kMagic;
    cfg.fn_optimal = "fnOptimal"; cfg.fn_state = "fnState"; cfg.fn_options = "fnOptions";
    cfg.frame_by_token = true;
    Env t;  // reuse the scene, swap in a by-token proxy
    delete t.p;
    t.p = new RrProxy(cfg, make_orig(), &rt2.table, &st, nullptr, [] { return (uint64_t)5000; });
    t.p->set_feature_function_pointers((void*)1, (void*)2, (void*)3);
    t.setup();
    WgFrameToken* ta = t.token(20);
    WgFrameToken* tb = t.token(20);
    WgConstants old = make_consts(0.5f, 0.0f), cur = make_consts(4.0f, 0.0f);
    t.p->constants(&old, tb, &t.vp);  // once, long ago, through the evaluate's token object
    t.p->constants(&cur, ta, &t.vp);
    const WgHeader* tin[1] = {&t.vp.h};
    CHECK_EQ(t.p->evaluate(1001, tb, tin, 1, t.cmd), 0u);
    CHECK(rt2.last().jitter[0] == 0.5f);
  }
  // lifetime-0 inputs are copied during the tagging call on the unwrapped command list; replaced copies released
  Env l;
  l.auto_tag = false;
  l.tags[0].lifetime = WG_LIFE_ONLY_DURING_TAG;
  l.tags[1].lifetime = WG_LIFE_ONLY_DURING_TAG;  // output: never copied
  l.setup();
  CHECK_EQ(l.rt.retains, 1);
  CHECK(l.rt.last_transient_cmd == reinterpret_cast<uint8_t*>(l.cmd) + 1);
  CHECK_EQ(l.frame(0), 0u);
  CHECK(l.rt.last().images[WG_ROLE_COLOUR].resource == reinterpret_cast<uint8_t*>(fake_res(0)) + 0x7000);
  CHECK(l.rt.last().images[WG_ROLE_OUTPUT].resource == fake_res(1));
  l.retag(l.tags[0]);
  CHECK(l.rt.retains == 2 && l.rt.releases == 1);
  // RR not requested in <init>: nothing copied
  Env nr;
  nr.tags[0].lifetime = WG_LIFE_ONLY_DURING_TAG;
  nr.init(false);
  nr.p->tag(&nr.vp, nr.tags.data(), (uint32_t)nr.tags.size(), nr.cmd);
  CHECK_EQ(nr.rt.retains, 0);
}

static void test_formats() {
  CHECK_EQ(format_class(DXGI_FORMAT_R16G16B16A16_FLOAT, false), (uint32_t)WG_FMT_FLOAT);
  CHECK_EQ(format_class(DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, false), (uint32_t)WG_FMT_SRGB);
  CHECK_EQ(format_class(DXGI_FORMAT_B8G8R8A8_UNORM_SRGB, false), (uint32_t)WG_FMT_SRGB);
  CHECK_EQ(format_class(DXGI_FORMAT_R16G16_SNORM, false), (uint32_t)WG_FMT_SNORM);
  CHECK_EQ(format_class(DXGI_FORMAT_R10G10B10A2_TYPELESS, false), (uint32_t)WG_FMT_UNORM);
  CHECK_EQ(format_class(DXGI_FORMAT_R16G16B16A16_TYPELESS, false), (uint32_t)WG_FMT_FLOAT);
  CHECK_EQ(format_class(DXGI_FORMAT_R8G8B8A8_TYPELESS, false), (uint32_t)WG_FMT_UNORM);
  CHECK_EQ(format_class(DXGI_FORMAT_R24G8_TYPELESS, false), (uint32_t)WG_FMT_UNORM);
  CHECK_EQ(format_class(DXGI_FORMAT_R24G8_TYPELESS, true), (uint32_t)WG_FMT_DEPTH_UNORM);
  CHECK_EQ(format_class(DXGI_FORMAT_R32_TYPELESS, true), (uint32_t)WG_FMT_DEPTH_FLOAT);
  CHECK_EQ(format_class(DXGI_FORMAT_R32G8X24_TYPELESS, true), (uint32_t)WG_FMT_DEPTH_FLOAT);
  CHECK_EQ(format_class(DXGI_FORMAT_D32_FLOAT, true), (uint32_t)WG_FMT_DEPTH_FLOAT);
  CHECK_EQ(format_class(DXGI_FORMAT_D24_UNORM_S8_UINT, true), (uint32_t)WG_FMT_DEPTH_UNORM);
  CHECK_EQ(format_class(DXGI_FORMAT_D16_UNORM, true), (uint32_t)WG_FMT_DEPTH_UNORM);
  CHECK_EQ(format_class(DXGI_FORMAT_D32_FLOAT_S8X24_UINT, true), (uint32_t)WG_FMT_DEPTH_FLOAT);
  CHECK_EQ(format_class(DXGI_FORMAT_D32_FLOAT, false), (uint32_t)WG_FMT_UNSUPPORTED);
  CHECK_EQ(format_class(DXGI_FORMAT_D24_UNORM_S8_UINT, false), (uint32_t)WG_FMT_UNSUPPORTED);
  CHECK_EQ(format_class(DXGI_FORMAT_R32_FLOAT, true), (uint32_t)WG_FMT_FLOAT);
  const DXGI_FORMAT ints[] = {DXGI_FORMAT_R32G32B32A32_UINT, DXGI_FORMAT_R32G32B32A32_SINT, DXGI_FORMAT_R16G16_UINT,
                              DXGI_FORMAT_R8_SINT, DXGI_FORMAT_R10G10B10A2_UINT, DXGI_FORMAT_X24_TYPELESS_G8_UINT,
                              DXGI_FORMAT_X32_TYPELESS_G8X24_UINT, DXGI_FORMAT_R32_UINT, DXGI_FORMAT_R16_SINT};
  for (DXGI_FORMAT f : ints) { CHECK_EQ(format_class(f, false), 0u); CHECK_EQ(format_class(f, true), 0u); }
  for (int f = DXGI_FORMAT_BC1_TYPELESS; f <= DXGI_FORMAT_BC5_SNORM; ++f) CHECK_EQ(format_class((uint32_t)f, false), 0u);
  for (int f = DXGI_FORMAT_BC6H_TYPELESS; f <= DXGI_FORMAT_BC7_UNORM_SRGB; ++f) CHECK_EQ(format_class((uint32_t)f, false), 0u);
  for (int f = DXGI_FORMAT_AYUV; f <= DXGI_FORMAT_A8P8; ++f) CHECK_EQ(format_class((uint32_t)f, false), 0u);
  CHECK_EQ(format_class(DXGI_FORMAT_B4G4R4A4_UNORM, false), (uint32_t)WG_FMT_UNORM);
  CHECK_EQ(format_class(DXGI_FORMAT_A4B4G4R4_UNORM, false), (uint32_t)WG_FMT_UNORM);
  CHECK_EQ(format_class(DXGI_FORMAT_P208, false), 0u);
  CHECK_EQ(format_class(DXGI_FORMAT_UNKNOWN, false), 0u);
  // in an evaluate: refused formats -> 25 before any pass, logged once per instance, state unchanged
  Env e;
  e.setup();
  CHECK_EQ(e.frame(0), 0u);
  int ev = e.rt.evaluates;
  e.spec.r.native_format = DXGI_FORMAT_R8G8B8A8_UINT;
  CHECK_EQ(e.frame(1), 25u);
  e.spec.r.native_format = DXGI_FORMAT_BC3_UNORM;
  CHECK_EQ(e.frame(2), 25u);
  CHECK_EQ(e.rt.evaluates, ev);
  CHECK_EQ(e.p->log_count("unsupported format"), 1u);
  CHECK(!e.p->lifecycle()->pending_reset() && e.p->lifecycle()->parity() == 1);
  e.spec.r.native_format = DXGI_FORMAT_R8G8B8A8_UNORM;
  // multisampled -> refused; description format 0 -> the native resource's format
  e.rt.described[fake_res(4)] = {DXGI_FORMAT_R8G8B8A8_UNORM, 4};
  CHECK_EQ(e.frame(3), 25u);
  e.rt.described[fake_res(4)] = {DXGI_FORMAT_R16G16B16A16_FLOAT, 1};
  e.dif.r.native_format = 0;
  CHECK_EQ(e.frame(4), 0u);
  CHECK(e.rt.last().images[WG_ROLE_DIFFUSE_ALBEDO].format == DXGI_FORMAT_R16G16B16A16_FLOAT);
  e.rt.described.erase(fake_res(4));
  CHECK_EQ(e.frame(5), 25u);  // format unknown and not describable
  e.dif.r.native_format = DXGI_FORMAT_R8G8B8A8_UNORM;
  // depth formats accepted for depth inputs
  e.depth.r.native_format = DXGI_FORMAT_R24G8_TYPELESS;
  CHECK_EQ(e.frame(6), 0u);
  CHECK_EQ(e.rt.last().images[WG_ROLE_DEPTH].format_class, (uint32_t)WG_FMT_DEPTH_UNORM);
  e.colour.r.native_format = DXGI_FORMAT_D32_FLOAT;
  CHECK_EQ(e.frame(7), 25u);
}

static void test_optimal_unit() {
  float r;
  CHECK(!mode_ratio(0, &r) && !mode_ratio(5, &r) && !mode_ratio(7, &r));
  CHECK(mode_ratio(6, &r) && r == 1.0f);
  CHECK_EQ(scaled_size(1920, 0.5f), 960u);
  CHECK_EQ(scaled_size(3, 0.5f), 2u);  // 1.5 -> 2 (half up)
  CHECK_EQ(scaled_size(5, 0.5f), 3u);  // 2.5 -> 3 (half up, not to even)
  CHECK_EQ(min_render_size(1920), 640u);
  CHECK_EQ(min_render_size(2), 1u);    // 0.6666 -> 1
  CHECK_EQ(min_render_size(1), 0u);
}

// ---------------------------------------------------------------- settings / contract / ASI view
static std::string tmp_dir() {
  char buf[512];
  DWORD n = GetTempPathA(sizeof buf, buf);
  std::string d(buf, n);
  d += "wg_rr_proxy_test_" + std::to_string(GetCurrentProcessId()) + "\\";
  CreateDirectoryA(d.c_str(), nullptr);
  return d;
}

static void test_settings_and_contract() {
  std::string dir = tmp_dir();
  std::string path = dir + "wronggpu-rr.ini";
  DeleteFileA(path.c_str());
  std::vector<std::string> logs;
  auto logger = [&logs](const std::string& s) { logs.push_back(s); };
  {
    Settings s;
    s.load(path, logger);  // no file: defaults, nothing logged
    CHECK(logs.empty());
    for (uint32_t i = 0; i < kSettingCount; ++i) CHECK_EQ(s.get(i), kSettingDefs[i].default_value);
    CHECK_EQ(s.get(kSetWrongGpuRr), 1);
    bind_contract(&s, [] { return 3u; });
    const WgRrSettingsContract* c = settings_contract();
    CHECK(c->size == 40 && c->version == 1 && c->count == kSettingCount && c->reserved == 0 && !c->reserved_ptr);
    CHECK_EQ(c->status(), 3u);
    for (uint32_t i = 0; i < c->count; ++i) {
      const WgRrSettingEntry& en = c->entries[i];
      CHECK(strcmp(en.key, kSettingDefs[i].key) == 0 && en.kind == 0 && en.max_value == 1);
      CHECK_EQ(en.get(), en.default_value);
      CHECK_EQ(en.flags, i == 0 ? 1u : 2u);  // only "WrongGPU RR" applies at creation; the rest are opt-ins
      CHECK_EQ(en.default_value, i == 0 ? 1 : 0);
      CHECK_EQ(en.set(2), 25u);
      CHECK_EQ(en.set(-1), 25u);
      CHECK_EQ(en.get(), en.default_value);  // unchanged
    }
    CHECK(GetFileAttributesA(path.c_str()) == INVALID_FILE_ATTRIBUTES);  // no change, no file written
    CHECK_EQ(c->entries[0].set(0), 0u);
    CHECK_EQ(c->entries[1].set(1), 0u);
    CHECK_EQ(c->entries[0].get(), 0);
    CHECK(GetFileAttributesA(path.c_str()) != INVALID_FILE_ATTRIBUTES);  // saved after a change
  }
  {
    Settings s2;  // a new proxy load reads the same values
    s2.load(path, logger);
    CHECK_EQ(s2.get(0), 0);
    CHECK_EQ(s2.get(1), 1);
    CHECK_EQ(s2.get(2), 0);
  }
  {
    FILE* f = fopen(path.c_str(), "wb");
    fputs("# test\nwronggpu_rr = 0\nspecular_motion_from_v=7\nbounded_stores=x\nunknown_key=1\n[Section]\n", f);
    fclose(f);
    Settings s3;
    s3.load(path, logger);
    CHECK_EQ(s3.get(0), 0);   // valid value read
    CHECK_EQ(s3.get(1), 0);   // invalid -> default
    CHECK_EQ(s3.get(4), 0);   // invalid -> default
    CHECK(logs.empty());
  }
  {
    std::string dpath = dir + "as_directory.ini";  // an existing path that cannot be read as a file
    CreateDirectoryA(dpath.c_str(), nullptr);
    Settings s4;
    s4.load(dpath, logger);
    CHECK_EQ(logs.size(), (size_t)1);
    CHECK(logs.size() == 1 && logs[0].find("unreadable") != std::string::npos);
    for (uint32_t i = 0; i < kSettingCount; ++i) CHECK_EQ(s4.get(i), kSettingDefs[i].default_value);
    RemoveDirectoryA(dpath.c_str());
  }
  {  // clean/060-rr-plugin-hunt: a game folder the ANSI code page cannot spell (a Cyrillic and an accented name). The
     // path is UTF-8 (as the plugin builds it); the file is read, saved and replaced by its wide name.
    wchar_t tbuf[512];
    const DWORD tn = GetTempPathW(512, tbuf);
    const std::wstring wdir = std::wstring(tbuf, tn) + L"wg_rr_proxy_test_" + std::to_wstring(GetCurrentProcessId()) +
                              L"_\u0418\u0433\u0440\u044b_Jos\u00e9\\";
    CHECK(CreateDirectoryW(wdir.c_str(), nullptr) || GetLastError() == ERROR_ALREADY_EXISTS);
    const std::wstring wini = wdir + L"wronggpu-rr.ini";
    FILE* f = _wfopen(wini.c_str(), L"wb");
    CHECK(f != nullptr);
    if (f) {
      fputs("bounded_stores=1\nrr_model=K:\\models\\rr.bin\n", f);
      fclose(f);
    }
    const int n = WideCharToMultiByte(CP_UTF8, 0, wini.c_str(), (int)wini.size(), nullptr, 0, nullptr, nullptr);
    std::string u8((size_t)n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, wini.c_str(), (int)wini.size(), &u8[0], n, nullptr, nullptr);
    const size_t before = logs.size();
    Settings s6;
    s6.load(u8, logger);
    CHECK_EQ(s6.get(kSetBoundedStores), 1);
    CHECK(s6.text("rr_model") == "K:\\models\\rr.bin");
    CHECK_EQ(logs.size(), before);                    // read ("unreadable" is not logged)
    CHECK_EQ(s6.set(kSetHitDistanceFilter, 1), 0u);  // a change: saved to <file>.tmp, moved over the file
    CHECK_EQ(logs.size(), before);                    // nor "could not be written / replaced"
    CHECK(GetFileAttributesW((wini + L".tmp").c_str()) == INVALID_FILE_ATTRIBUTES);
    Settings s7;
    s7.load(u8, logger);
    CHECK_EQ(s7.get(kSetHitDistanceFilter), 1);
    CHECK_EQ(s7.get(kSetBoundedStores), 1);
    CHECK(s7.text("rr_model") == "K:\\models\\rr.bin");  // the non-contract line kept by the rewrite
    DeleteFileW(wini.c_str());
    RemoveDirectoryW(wdir.c_str());
  }
  // ASI view (60.9 "Absence")
  Settings s5;
  s5.set_save_enabled(false);
  uint32_t status = 0;
  bind_contract(&s5, [&status] { return status; });
  const WgRrSettingsContract* c = settings_contract();
  wgrr_asi::ControlsView v = wgrr_asi::build_view(nullptr);
  CHECK(v.mode == wgrr_asi::ControlsView::kNotActive && v.line == "WrongGPU RR not active" && v.items.empty());
  v = wgrr_asi::build_view(c);
  CHECK(v.mode == wgrr_asi::ControlsView::kNotActive && v.items.empty());
  status = 1;  // model not loaded
  CHECK(wgrr_asi::build_view(c).mode == wgrr_asi::ControlsView::kNotActive);
  status = 2;
  CHECK(wgrr_asi::build_view(c).mode == wgrr_asi::ControlsView::kNotActive);
  status = 3;
  v = wgrr_asi::build_view(c);
  CHECK(v.mode == wgrr_asi::ControlsView::kControls && v.items.size() == kSettingCount);
  CHECK(v.items[0].label == "WrongGPU RR" && v.items[0].at_creation && v.items[0].value == 1);
  for (size_t i = 1; i < v.items.size(); ++i) CHECK(!v.items[i].at_creation && v.items[i].opt_in);
  WgRrSettingsContract c2 = *c;
  c2.version = 2;
  CHECK(wgrr_asi::build_view(&c2).mode == wgrr_asi::ControlsView::kNothing);
  c2 = *c;
  c2.size = 32;
  CHECK(wgrr_asi::build_view(&c2).mode == wgrr_asi::ControlsView::kNothing);
  CHECK_EQ(wgrr_asi::apply_change(&c2, 0, 0), 25u);
  CHECK_EQ(wgrr_asi::apply_change(c, 99, 0), 25u);
  CHECK_EQ(wgrr_asi::apply_change(c, 1, 5), 25u);
  CHECK_EQ(s5.get(1), 0);
  // drawing through a sink: every control drawn, note next to flag-bit-0 controls, changes go through set
  struct Sink : wgrr_asi::ControlsSink {
    std::vector<std::string> out;
    void text(const char* l) override { out.push_back(std::string("text:") + l); }
    bool toggle(const char* l, int32_t cur, int32_t* nv) override {
      out.push_back(std::string("toggle:") + l);
      if (std::string(l) == "WrongGPU RR") { *nv = cur ? 0 : 1; return true; }
      return false;
    }
    bool choice(const char*, int32_t, uint32_t, int32_t*) override { return false; }
    void note(const char* t) override { out.push_back(std::string("note:") + t); }
  } sink;
  wgrr_asi::draw_view(c, v, sink);
  CHECK_EQ(sink.out.size(), (size_t)kSettingCount + 1);
  CHECK(sink.out[1] == "note:applies when the game next creates RR");
  CHECK_EQ(s5.get(0), 0);  // toggled through the contract's set function
  Sink s_na;
  wgrr_asi::draw_view(nullptr, wgrr_asi::build_view(nullptr), s_na);
  CHECK(s_na.out.size() == 1 && s_na.out[0] == "text:WrongGPU RR not active");
  Sink s_nothing;
  wgrr_asi::draw_view(&c2, wgrr_asi::build_view(&c2), s_nothing);
  CHECK(s_nothing.out.empty());
  // RR works the same with and without the ASI: the proxy answers do not depend on the contract being read
  bind_contract(nullptr, nullptr);
  CHECK_EQ(c->entries[0].set(0), 25u);  // unbound: refused
  CHECK_EQ(c->status(), 0u);
  RemoveDirectoryA(dir.c_str());
  DeleteFileA(path.c_str());
}

// Dump of WrongGPU's optimal-settings rule for tests/test_optimal.py (oracle: exact rationals).
static int dump_optimal(const char* path) {
  FILE* f = fopen(path, "wb");
  if (!f) return 1;
  for (uint32_t mode = 0; mode <= 7; ++mode) {
    for (uint32_t o = 0; o <= 8192; ++o) {
      OptimalAnswer a = optimal_settings(mode, o, o);
      fprintf(f, "%u,%u,%d,%u,%u,%u\n", mode, o, a.has_ratio ? 1 : 0, a.width, a.min_width, a.max_width);
    }
  }
  fclose(f);
  return 0;
}

// A9.1: the proxy forwards every <init> version value unchanged and never rejects one (an older SDK version than
// the interposer's, another magic); <version>(1001) reports the value as the game passed it.
static void test_version_values() {
  struct Case { uint32_t minor; uint64_t v; uint32_t ma, mi, pa; bool magic_ok; };
  const Case cases[] = {
      {7, 0x000200070001fedcull, 2, 7, 1, true},    // game A: 2.7.1 value, 2.7.1 interposer
      {7, 0x000200070002fedcull, 2, 7, 2, true},    // 2.7.2
      {14, 0x0002000b0001fedcull, 2, 11, 1, true},  // game B: 2.11.1 value with a 2.14.1 interposer
      {14, 0x0002000e0001fedcull, 2, 14, 1, true},  // 2.14.1
      {14, 0x0002000e00011234ull, 2, 14, 1, false}, // another magic: still forwarded unchanged
      {7, 0x0001000000000000ull, 1, 0, 0, false},   // an older major: still forwarded unchanged
  };
  for (const Case& c : cases) {
    Env e(c.minor);
    WgPreferences pr;
    memset(&pr, 0, sizeof pr);
    const uint32_t feats[1] = {1001};
    pr.features = feats;
    pr.feature_count = 1;
    g_orig.last_u64 = 0;
    CHECK_EQ(e.p->init(&pr, c.v), 0u);
    CHECK(g_orig.last_u64 == c.v && g_orig.last_role == "init");
    CHECK_EQ(e.p->log_count("unexpected magic"), c.magic_ok ? 0u : 1u);
    CHECK_EQ(e.p->log_count("forwarded unchanged"), 1u);
    WgFeatureVersion fv;
    memset(&fv, 0, sizeof fv);
    fv.h.version = 1;
    CHECK_EQ(e.p->version(1001, &fv), 0u);
    CHECK(fv.api_version[0] == c.ma && fv.api_version[1] == c.mi && fv.api_version[2] == c.pa);
    CHECK_EQ(e.p->is_supported(1001, nullptr), 0u);  // initialised whatever the version value
  }
}

// Q-B6: without the tag extension's GUID a chained extension is not recognised (no conversion handed on).
static void test_unknown_tag_extension() {
  Env e(14, true, false);
  e.setup();
  Res cl{41, 1280, 720, DXGI_FORMAT_R16G16B16A16_FLOAT};
  WgFrameToken* t = e.token(0);
  e.p->constants(&e.consts, t, &e.vp);
  WgTag lt = make_tag(&cl.r, WG_BT_COLOUR_IN);
  WgTagConversion conv;
  memset(&conv, 0, sizeof conv);
  conv.h.type = kGConv;
  conv.h.version = 1;
  conv.kind = 1;
  conv.scale = 2.0f;
  lt.h.next = &conv.h;
  const WgHeader* in[2] = {&lt.h, &e.vp.h};
  CHECK_EQ(e.p->evaluate(1001, t, in, 2, e.cmd), 0u);
  CHECK(e.rt.last().images[WG_ROLE_COLOUR].resource == fake_res(41) &&
        e.rt.last().images[WG_ROLE_COLOUR].conversion == 0);
}

// clean/rr-re9-pt: runtime work on the game's list is bracketed for the host (begin / end, balanced on every path,
// around the runtime calls only, with the native list); no bracket for rejections before any runtime call; the RR
// refusal for games that need the host's restore.
struct BracketLog {
  std::vector<std::string> ev;
  std::vector<void*> lists;
  int depth = 0, max_depth = 0;
  CmdListBracket make() {
    CmdListBracket b;
    b.begin = [this](void* cl) { ev.push_back("begin"); lists.push_back(cl); max_depth = ++depth > max_depth ? depth : max_depth; };
    b.end = [this](void* cl) { ev.push_back("end"); lists.push_back(cl); --depth; };
    return b;
  }
  int count(const char* what) const {
    int n = 0;
    for (const auto& e : ev) n += e == what;
    return n;
  }
};
static std::vector<std::string>* g_pass_events = nullptr;
static uint32_t g_pass_rc = 0;
static uint32_t fake_passthrough(void*, const WgRrFrame*) {
  if (g_pass_events) g_pass_events->push_back("pass");
  return g_pass_rc;
}

static void test_host_bracket() {
  // success: one pair per evaluate, the evaluate inside it, the native (unwrapped) list on both calls
  {
    Env e;
    BracketLog b;
    e.rt.on_evaluate = [&b] { b.ev.push_back("eval"); };
    e.p->set_cmd_list_bracket(b.make());
    e.setup();
    for (uint32_t i = 0; i < 3; ++i) CHECK_EQ(e.frame(i), 0u);
    CHECK_EQ(b.count("begin"), 3);
    CHECK_EQ(b.count("end"), 3);
    CHECK_EQ(b.depth, 0);
    CHECK_EQ(b.max_depth, 1);
    CHECK(b.ev.size() == 9 && b.ev[0] == "begin" && b.ev[1] == "eval" && b.ev[2] == "end");
    bool native = true;
    for (void* cl : b.lists) native = native && cl == reinterpret_cast<uint8_t*>(e.cmd) + 1;
    CHECK(native);
  }
  // runtime failure before its passes, pass-through succeeds: one pair around evaluate + pass-through
  {
    Env e;
    BracketLog b;
    e.rt.table.passthrough = &fake_passthrough;
    g_pass_events = &b.ev;
    g_pass_rc = 0;
    e.rt.on_evaluate = [&b] { b.ev.push_back("eval"); };
    e.p->set_cmd_list_bracket(b.make());
    e.setup();
    e.rt.evaluate_rc = 22;
    CHECK_EQ(e.frame(0), 0u);  // passed through, success reported
    CHECK(b.ev.size() == 4 && b.ev[0] == "begin" && b.ev[1] == "eval" && b.ev[2] == "pass" && b.ev[3] == "end");
    // pass-through fails too: the code goes to the game, the pair is still closed
    g_pass_rc = 22;
    b.ev.clear();
    CHECK_EQ(e.frame(1), 22u);
    CHECK(b.ev.size() == 4 && b.ev[3] == "end" && b.depth == 0);
    // failure after passes began (no pass-through): still one closed pair
    e.rt.next_result.passes_started = 1;
    b.ev.clear();
    CHECK_EQ(e.frame(2), 22u);
    CHECK(b.ev.size() == 3 && b.ev[0] == "begin" && b.ev[2] == "end" && b.depth == 0);
    g_pass_events = nullptr;
  }
  // rejections before any runtime call: no pair (zero size 25, no constants 26, missing viewport 20, mode off 19)
  {
    Env e;
    BracketLog b;
    e.p->set_cmd_list_bracket(b.make());
    e.setup();
    CHECK_EQ(e.frame(0), 0u);
    b.ev.clear();
    e.colour.r.width = 0;
    CHECK_EQ(e.frame(1), 25u);
    e.colour.r.width = 1280;
    WgFrameToken* t = e.token(5);
    const WgHeader* in[1] = {&e.vp.h};
    CHECK_EQ(e.p->evaluate(1001, t, in, 1, e.cmd), 26u);
    CHECK_EQ(e.p->evaluate(1001, t, nullptr, 0, e.cmd), 20u);
    WgRrOptions off = make_opts(0);
    e.p->rr_options(&e.vp, &off);
    CHECK_EQ(e.frame(2), 19u);
    CHECK(b.ev.empty());
  }
  // lifetime-0 input copied during the tagging call: one pair around the copy
  {
    Env e;
    BracketLog b;
    e.auto_tag = false;
    e.tags[0].lifetime = WG_LIFE_ONLY_DURING_TAG;
    e.p->set_cmd_list_bracket(b.make());
    e.setup();
    CHECK_EQ(e.rt.retains, 1);
    CHECK(b.count("begin") == 1 && b.count("end") == 1 && b.depth == 0);
    CHECK(b.lists[0] == reinterpret_cast<uint8_t*>(e.cmd) + 1);
    // RR not requested: nothing copied, no pair
    Env nr;
    BracketLog nb;
    nr.tags[0].lifetime = WG_LIFE_ONLY_DURING_TAG;
    nr.p->set_cmd_list_bracket(nb.make());
    nr.init(false);
    nr.p->tag(&nr.vp, nr.tags.data(), (uint32_t)nr.tags.size(), nr.cmd);
    CHECK(nb.ev.empty());
  }
  // no host (empty bracket) and a cleared bracket: evaluates run as before
  {
    Env e;
    BracketLog b;
    e.p->set_cmd_list_bracket(b.make());
    e.p->set_cmd_list_bracket(CmdListBracket());
    e.setup();
    CHECK_EQ(e.frame(0), 0u);
    CHECK(b.ev.empty());
    CmdListBracket half;  // only one side set: never opened
    half.begin = [&b](void*) { b.ev.push_back("begin"); };
    e.p->set_cmd_list_bracket(half);
    CHECK_EQ(e.frame(1), 0u);
    CHECK(b.ev.empty());
  }
  // RR refused for a game that needs the host's restore: adapter not supported (7), logged once; lifted again
  {
    Env e;
    e.init();
    e.p->set_device(e.dev);
    CHECK_EQ(e.p->is_supported(1001, nullptr), 0u);
    e.p->set_rr_refused(true, "needs the host's restore");
    CHECK_EQ(e.p->is_supported(1001, nullptr), 7u);
    CHECK_EQ(e.p->is_supported(1001, nullptr), 7u);
    CHECK_EQ(e.p->log_count("reported unsupported: needs the host's restore"), 1u);
    CHECK_EQ(e.p->is_supported(0, nullptr), orig_rc(0));  // other features untouched
    e.p->set_rr_refused(false, "");
    CHECK_EQ(e.p->is_supported(1001, nullptr), 0u);
  }
}

// clean/rr-re9-pt polish: the player's switch (rr_enabled = 0, set_rr_off). WrongGPU does not take RR: is-supported
// answers 7 whatever the runtime and the 60.9 setting say, nothing is created by allocate or evaluate, other features
// are forwarded unchanged, and the reason is logged once per kind. Lifting it gives RR back.
static void test_rr_off() {
  {
    Env e;
    e.init();
    e.p->set_device(e.dev);
    CHECK(!e.p->rr_off());
    CHECK_EQ(e.p->is_supported(1001, nullptr), 0u);
    e.p->set_rr_off(true, "rr_enabled=0 in test.ini");
    CHECK(e.p->rr_off());
    CHECK_EQ(e.p->is_supported(1001, nullptr), 7u);
    CHECK_EQ(e.p->is_supported(1001, nullptr), 7u);
    CHECK_EQ(e.p->log_count("reported unsupported: rr_enabled=0 in test.ini"), 1u);
    CHECK_EQ(e.p->is_supported(0, nullptr), orig_rc(0));  // other features untouched
    e.settings.set(kSetWrongGpuRr, 1);                     // the 60.9 setting on does not override the switch
    CHECK_EQ(e.p->is_supported(1001, nullptr), 7u);
    // also off when the host's refusal is lifted (two independent reasons)
    e.p->set_rr_refused(false, "");
    CHECK_EQ(e.p->is_supported(1001, nullptr), 7u);
    e.p->set_rr_off(false, "");
    CHECK_EQ(e.p->is_supported(1001, nullptr), 0u);
  }
  {  // a game that skips is-supported: evaluate and allocate create nothing (7), logged once
    Env e;
    e.setup();
    e.p->set_rr_off(true, "rr_enabled=0 in test.ini");
    CHECK_EQ(e.frame(0), 7u);
    CHECK_EQ(e.frame(1), 7u);
    CHECK_EQ(e.p->allocate(e.cmd, 1001, &e.vp), 7u);
    CHECK_EQ(e.rt.creates, 0);
    CHECK_EQ(e.rt.evaluates, 0);
    CHECK(!e.p->has_instance());
    CHECK_EQ(e.p->log_count("the RR feature is not created"), 1u);
    CHECK_EQ(e.p->log_count("is off"), 0u);  // not the 60.9 setting's message
    e.p->set_rr_off(false, "");
    CHECK_EQ(e.frame(2), 0u);
    CHECK_EQ(e.rt.creates, 1);
  }
}

// clean/rr-sl-diag: the game-interface diagnostics on the 2.7 route: one line per distinct call (support queries with the
// adapter and the reason of a "no", other features' queries, RR lookups served or not, set-loaded, the first options /
// constants / tag / evaluate), answers unchanged, and the summary at shutdown (also lock-free)
static void test_sl_diag() {
  Env e(7);
  CHECK_EQ(e.p->is_supported(1001, nullptr), 23u);  // before <init>: as before
  CHECK_EQ(e.p->log_count("diag: support query RR (1001): no adapter info (any adapter): answer 23 (no: <init> not seen yet)"), 1u);
  e.init();
  CHECK_EQ(e.p->is_supported(1001, nullptr), 0u);
  CHECK_EQ(e.p->is_supported(1001, nullptr), 0u);
  CHECK_EQ(e.p->log_count("diag: support query RR (1001): no adapter info (any adapter): answer 0 (yes)"), 1u);  // once
  WgAdapterInfo ai;
  memset(&ai, 0, sizeof ai);
  ai.h.version = 1;
  uint8_t good[8] = {1, 2, 3, 4, 5, 6, 7, 8}, other[8] = {9, 9, 9, 9, 0, 0, 0, 0};
  ai.luid = good;
  ai.luid_size = 8;
  e.p->set_adapter_describer([](const void* l, uint32_t n) -> std::string {
    return n == 8 && static_cast<const uint8_t*>(l)[0] == 1 ? "vendor 0x1002 device 0x7550, test GPU" : "";
  });
  CHECK_EQ(e.p->is_supported(1001, &ai), 0u);
  CHECK_EQ(e.p->log_count("diag: support query RR (1001): LUID 08070605:04030201 (adapter info v1), vendor 0x1002 device 0x7550, test GPU: answer 0 (yes)"), 1u);
  ai.luid = other;
  CHECK_EQ(e.p->is_supported(1001, &ai), 7u);
  CHECK_EQ(e.p->log_count("LUID 00000000:09090909 (adapter info v1): answer 7 (no: the runtime does not support the adapter"), 1u);
  e.rt.status_bits = WG_RR_RT_READY;  // model not loaded: no, with the runtime's state named
  CHECK_EQ(e.p->is_supported(1001, nullptr), 7u);
  CHECK_EQ(e.p->log_count("(no: the RR runtime is not ready (status 0x1: supported GPU found yes, model loaded no))"), 1u);
  e.rt.status_bits = WG_RR_RT_READY | WG_RR_RT_MODEL_LOADED;
  CHECK_EQ(e.p->is_supported(0, nullptr), orig_rc(0));  // another feature: forwarded unchanged, one line per answer
  CHECK_EQ(e.p->is_supported(0, nullptr), orig_rc(0));
  CHECK_EQ(e.p->log_count("diag: support query feature 0 (not RR, forwarded to the game interface): answer 100"), 1u);
  void* fn = nullptr;
  CHECK_EQ(e.p->get_function(1001, "fnOptimal", &fn), 5u);  // before set-device: as before
  CHECK_EQ(e.p->log_count("diag: RR function lookup \"fnOptimal\": not served, answer 5 (before set-device)"), 1u);
  e.p->set_device(e.dev);
  CHECK(e.p->get_function(1001, "fnOptimal", &fn) == 0u && fn == (void*)1);
  CHECK(e.p->get_function(1001, "fnOptimal", &fn) == 0u);
  CHECK_EQ(e.p->log_count("diag: RR function lookup \"fnOptimal\": served (answer 0)"), 1u);
  CHECK_EQ(e.p->get_function(1001, "fnSomethingElse", &fn), 25u);
  CHECK_EQ(e.p->log_count("diag: RR function lookup \"fnSomethingElse\": not served, answer 25 (not one of the RR feature functions)"), 1u);
  CHECK_EQ(e.p->set_loaded(1001, true), 0u);
  CHECK_EQ(e.p->log_count("diag: RR set-loaded(true): answer 0"), 1u);
  bool loaded = false;
  CHECK(e.p->is_loaded(1001, &loaded) == 0u && loaded);
  CHECK_EQ(e.p->log_count("diag: RR is-loaded: answer 0, loaded true"), 1u);
  WgRrOptimal opt;
  memset(&opt, 0, sizeof opt);
  CHECK_EQ(e.p->rr_optimal(&e.opts, &opt), 0u);
  CHECK_EQ(e.p->log_count("diag: RR optimal-settings query: mode "), 1u);
  CHECK_EQ(e.p->rr_options(&e.vp, &e.opts), 0u);
  CHECK_EQ(e.p->log_count("diag: RR options: viewport 7, mode " + std::to_string(e.opts.mode) + ", output 1920x1080"), 1u);
  e.p->tag(&e.vp, e.tags.data(), (uint32_t)e.tags.size(), e.cmd);
  CHECK_EQ(e.frame(0), 0u);
  CHECK_EQ(e.frame(1), 0u);
  CHECK_EQ(e.p->log_count("diag: first tag call: viewport 7, 8 tag(s): 3 "), 1u);
  CHECK_EQ(e.p->log_count("diag: first constants: viewport 7, version 2, jitter (0.2500, -0.2500)"), 1u);
  CHECK_EQ(e.p->log_count("diag: first RR evaluate: 1 input(s): answer 0 (yes), instance present"), 1u);
  CHECK_EQ(e.p->log_count("diag: first "), 3u);  // tag, constants, evaluate: once each
  CHECK_EQ(e.rt.creates, 1);                     // the RR path itself unchanged
  const std::string mid = e.p->diag_summary("now");
  CHECK(mid == "diag: summary (now): support queries 8 (RR: yes 3 / no 3), RR lookups 4, RR instances created 1, RR evaluates 2");
  if (mid.find("support queries 8") == std::string::npos) fprintf(stderr, "summary: %s\n", mid.c_str());
  e.p->shutdown();
  CHECK_EQ(e.p->log_count("diag: summary (shutdown): support queries 8 (RR: yes 3 / no 3), RR lookups 4, RR instances created 1, "
                          "RR evaluates 2"), 1u);
}

// clean/060-rr2: the slot watch is off by default in release builds; WG_RR_SLOT_CHECKS=on or a list turns it on
static void test_slot_check_schedule() {
  using wgrr::slot_check_schedule;
  CHECK(slot_check_schedule(nullptr).empty());
  CHECK(slot_check_schedule("").empty());
  CHECK(slot_check_schedule("off").empty());
  CHECK(slot_check_schedule("on") == wgrr::kSlotChecksDefault);
  CHECK(std::string(wgrr::kSlotChecksDefault) == "10,30,60,120,240,480,900");
  CHECK(slot_check_schedule("0") == "0");
  CHECK(slot_check_schedule("5,20") == "5,20");
}

int main(int argc, char** argv) {
  if (argc == 3 && strcmp(argv[1], "--dump-optimal") == 0) return dump_optimal(argv[2]);
  test_sl_diag();
  test_preinit_and_device();
  test_forwarding();
  test_answers();
  test_rr_optimal_versions();
  test_options_and_viewports();
  test_evaluate_happy_and_lifecycle();
  test_creation_decisions();
  test_recreate_triggers();
  test_render_size_changes();
  test_rejections();
  test_failed_evaluates();
  test_settings_snapshot_and_rr_switch();
  test_refusals_logged_once();
  test_precedence();
  test_routing();
  test_frames_and_tokens();
  test_formats();
  test_optimal_unit();
  test_settings_and_contract();
  test_version_values();
  test_host_bracket();
  test_rr_off();
  test_unknown_tag_extension();
  test_slot_check_schedule();
  TEST_SUMMARY("test_proxy_core");
}
