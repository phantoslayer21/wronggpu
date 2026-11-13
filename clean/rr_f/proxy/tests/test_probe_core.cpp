// SPDX-License-Identifier: MIT
// Unit tests of the probe build's logic (src/probe.*, src/probe_log.*; build flag WG_RR_PROBE) against a fake
// original and a fake GPU: the log rules (full first frames, then changes only, size cap), the RR claims of 60.8 and
// the 61.5 rule, forwarding of everything else, the evaluate that writes the upscale and returns 0 (and touches no
// other resource), the documented errors, every logged field the probe owes, the D3D12 view-format choice, and the
// capture option (settings file parsing, which evaluates are copied, what is written, size cap, late or missing
// completion markers, the final read at shutdown, off by default).
#include <dxgiformat.h>
#include <stdio.h>
#include <string.h>

#include <map>
#include <string>
#include <vector>

#include "../src/answers.h"
#include "../src/probe.h"
#include "../src/probe_gpu_d3d12.h"
#include "../src/probe_log.h"
#include "test_util.h"

using namespace wgrr;

// ---------------------------------------------------------------- fake original
struct OrigLog {
  int calls = 0;
  std::string last_role;
  uint32_t last_feature = 0;
  const void* last_ptr = nullptr;
  uint64_t last_u64 = 0;
  void* last_cmd = nullptr;
};
static OrigLog g_orig;
static WgResultCode orig_rc(uint32_t f) { return (WgResultCode)(100u + f % 50u); }
static void rec(const char* r, uint32_t f, const void* p) {
  g_orig.calls++;
  g_orig.last_role = r;
  g_orig.last_feature = f;
  g_orig.last_ptr = p;
}
static WgResultCode o_init(const WgPreferences* p, uint64_t v) { rec("init", 0, p); g_orig.last_u64 = v; return 0; }
static WgResultCode o_shutdown() { rec("shutdown", 0, nullptr); return 0; }
static WgResultCode o_is_supported(uint32_t f, const WgAdapterInfo* a) { rec("is-supported", f, a); return orig_rc(f); }
static WgResultCode o_is_loaded(uint32_t f, bool* l) { rec("is-loaded", f, l); return orig_rc(f); }
static WgResultCode o_set_loaded(uint32_t f, bool) { rec("set-loaded", f, nullptr); return orig_rc(f); }
static WgResultCode o_requirements(uint32_t f, WgRequirements* r) { rec("requirements", f, r); return orig_rc(f); }
static WgResultCode o_version(uint32_t f, WgFeatureVersion* v) { rec("version", f, v); return orig_rc(f); }
static WgResultCode o_get_function(uint32_t f, const char* n, void**) { rec("get-function", f, n); return orig_rc(f); }
static WgResultCode o_allocate(void* c, uint32_t f, const WgViewport* v) { rec("allocate", f, v); g_orig.last_cmd = c; return orig_rc(f); }
static WgResultCode o_free(uint32_t f, const WgViewport* v) { rec("free", f, v); return orig_rc(f); }
static WgResultCode o_set_device(void* d) { rec("set-device", 0, d); return 0; }
static WgResultCode o_set_vulkan_info(const void* i) { rec("set-vulkan-info", 0, i); return 9; }
static WgResultCode o_tag(const WgViewport*, const WgTag* t, uint32_t, void* c) { rec("tag", 0, t); g_orig.last_cmd = c; return 0; }
static WgResultCode o_tag_for_frame(const WgFrameToken*, const WgViewport*, const WgTag* t, uint32_t, void* c) {
  rec("tag-for-frame", 0, t); g_orig.last_cmd = c; return 0;
}
static WgResultCode o_constants(const WgConstants* c, const WgFrameToken*, const WgViewport*) { rec("constants", 0, c); return 0; }
static const WgGuid kGToken = [] { WgGuid g; memset(g.b, 0x35, 16); g.b[0] = 0x11; return g; }();
struct FakeToken { WgFrameToken t; uint32_t index; };
static uint32_t token_index_fn(const WgFrameToken* t) { return reinterpret_cast<const FakeToken*>(t)->index; }
static const WgFrameTokenIndexFn kTokenVtbl[1] = {&token_index_fn};
static FakeToken g_tokens[64];
static uint32_t g_next_token = 0;
static WgResultCode o_new_frame_token(WgFrameToken** out, const uint32_t* idx) {
  rec("new-frame-token", 0, out);
  FakeToken& t = g_tokens[g_next_token++ % 64];
  memset(&t, 0, sizeof t);
  t.t.vtable = kTokenVtbl;
  t.t.header.type = kGToken;
  t.t.header.version = 1;
  t.index = idx ? *idx + 500 : 7;  // the index the probe logs comes from the table, not from the argument
  *out = &t.t;
  return 0;
}
static WgResultCode o_evaluate(uint32_t f, const WgFrameToken*, const WgHeader* const* in, uint32_t, void* c) {
  rec("evaluate", f, in); g_orig.last_cmd = c; return orig_rc(f);
}
static WgResultCode o_upgrade(void** i) { rec("upgrade-interface", 0, i); return 0; }
static WgResultCode o_native(void* w, void** n) { *n = reinterpret_cast<uint8_t*>(w) + 1; return 0; }

static OriginalApi make_orig() {
  OriginalApi o;
  o.init = o_init; o.shutdown = o_shutdown; o.is_supported = o_is_supported; o.is_loaded = o_is_loaded;
  o.set_loaded = o_set_loaded; o.requirements = o_requirements; o.version = o_version;
  o.get_function = o_get_function; o.allocate = o_allocate; o.free_resources = o_free; o.set_device = o_set_device;
  o.set_vulkan_info = o_set_vulkan_info; o.tag = o_tag; o.tag_for_frame = o_tag_for_frame; o.constants = o_constants;
  o.new_frame_token = o_new_frame_token; o.evaluate = o_evaluate; o.upgrade_interface = o_upgrade;
  o.native_interface = o_native;
  return o;
}

// ---------------------------------------------------------------- fake capture host
struct FakeHost : ProbeCaptureHost {
  bool have_ini = false, open_ok = true;
  std::string ini;
  int opens = 0, flushes = 0;
  std::string folder;  // set_folder
  void set_folder(const std::string& p) override { folder = p; }
  std::vector<std::pair<std::string, std::vector<uint8_t>>> files;  // in write order
  bool read_settings(std::string* t) override {
    if (!have_ini) return false;
    *t = ini;
    return true;
  }
  bool open(std::string* w) override {
    ++opens;
    *w = "fake-folder";
    return open_ok;
  }
  bool write(const std::string& rel, std::vector<uint8_t>&& d) override {
    files.emplace_back(rel, std::move(d));
    return true;
  }
  uint64_t flush() override {
    ++flushes;
    return 0;
  }
  const std::vector<uint8_t>* file(const std::string& rel) const {
    for (const auto& f : files)
      if (f.first == rel) return &f.second;
    return nullptr;
  }
  std::string text(const std::string& rel) const {
    const std::vector<uint8_t>* b = file(rel);
    return b ? std::string(b->begin(), b->end()) : std::string();
  }
};
static uint32_t le32(const std::vector<uint8_t>& b, size_t at) {
  uint32_t v = 0;
  if (at + 4 <= b.size()) memcpy(&v, b.data() + at, 4);
  return v;
}

// ---------------------------------------------------------------- fake GPU
struct FakeGpu : ProbeGpu {
  void* device = nullptr;
  int upscales = 0, patches = 0, copies = 0, describes = 0;
  ProbeView src, dst, patch_src, copy_src;
  uint32_t patch_px = 0, patch_py = 0;
  void* upscale_cmd = nullptr;
  void* copy_cmd = nullptr;
  WgResultCode upscale_rc = 0;
  std::map<uint32_t, uint32_t> written_markers;  // slot -> marker "written by the GPU"
  uint8_t copy_tex[4];
  void set_device(void* d) override { device = d; }
  bool has_device() const override { return device != nullptr; }
  bool describe(void* r, ProbeTexDesc* d) override {
    describes++;
    if (!r) return false;
    d->dimension = 3; d->format = 10; d->width = 64; d->height = 36; d->depth_or_array = 1; d->mips = 1;
    d->samples = 1; d->flags = 4;
    return true;
  }
  WgResultCode upscale(void* cmd, const ProbeView& s, const ProbeView& d, std::string* why) override {
    upscales++; src = s; dst = d; upscale_cmd = cmd;
    if (upscale_rc) *why = "fake failure";
    return upscale_rc;
  }
  WgResultCode patch_mean(void*, const ProbeView& s, uint32_t px, uint32_t py, uint32_t slot, uint32_t marker,
                          std::string*) override {
    patches++; patch_src = s; patch_px = px; patch_py = py;
    written_markers[slot] = marker;
    return 0;
  }
  bool read_patch(uint32_t slot, uint32_t marker, float mean[4]) override {
    auto it = written_markers.find(slot);
    if (it == written_markers.end() || it->second != marker) return false;
    mean[0] = 0.5f; mean[1] = 0.25f; mean[2] = 1.0f; mean[3] = 0.75f;
    return true;
  }
  void* copy_input(void* cmd, const ProbeView& s, uint32_t, uint32_t* state, std::string*) override {
    copies++; copy_src = s; copy_cmd = cmd;
    *state = 0x40;
    return copy_tex;
  }
  // capture: 4 x 2 images of 8-byte rows (type-dependent bytes), "complete" when cap_ready
  FakeHost* host = nullptr;
  bool cap_ready = true;
  uint64_t cap_bytes_each = 1000;
  void* cap_cmd = nullptr;
  std::vector<uint32_t> cap_ids, abandoned;
  std::map<uint32_t, std::pair<uint32_t, std::vector<CaptureSource>>> cap_pending;
  std::vector<CaptureSource> cap_last;
  ProbeCaptureHost* capture_host() override { return host; }
  WgResultCode capture_copy(void* cmd, uint32_t id, uint32_t marker, const std::vector<CaptureSource>& src,
                            uint64_t budget, uint64_t* bytes, std::vector<std::string>* skipped,
                            std::string* why) override {
    if (cap_bytes_each > budget) {
      *why = "size cap: " + std::to_string(cap_bytes_each) + " bytes needed, " + std::to_string(budget) + " left";
      return WG_E_INVALID_PARAMETER;
    }
    cap_cmd = cmd;
    cap_ids.push_back(id);
    cap_last = src;
    std::vector<CaptureSource> kept;
    for (const CaptureSource& c : src) {
      if (c.type == WG_BT_SPECULAR_ALBEDO && c.view.state == 0x1234) skipped->push_back("type 8: fake refusal");
      else kept.push_back(c);
    }
    cap_pending[id] = std::make_pair(marker, kept);
    *bytes = cap_bytes_each;
    return 0;
  }
  int capture_read(uint32_t id, uint32_t marker, std::vector<CaptureImage>* out) override {
    auto it = cap_pending.find(id);
    if (it == cap_pending.end()) return -1;
    if (!cap_ready || it->second.first != marker) return 0;
    for (const CaptureSource& c : it->second.second) {
      CaptureImage im;
      im.type = c.type;
      im.format = c.view.format ? c.view.format : 10;
      im.width = 4;
      im.height = 2;
      im.row_pitch = 8;
      for (int i = 0; i < 16; ++i) im.data.push_back((uint8_t)(c.type * 16 + i));
      out->push_back(im);
    }
    cap_pending.erase(it);
    return 1;
  }
  void capture_abandon(uint32_t id) override {
    abandoned.push_back(id);
    cap_pending.erase(id);
  }
};

// ---------------------------------------------------------------- helpers
static WgGuid guid(uint8_t k) { WgGuid g; memset(g.b, 0x5a, 16); g.b[15] = k; return g; }
static const WgGuid kGViewport = guid(1), kGTag = guid(2), kGExtra = guid(9);
static uint8_t g_pool[256];
static void* fake(int i) { return &g_pool[i]; }

struct Res {
  WgResource r;
  Res(int id, uint32_t w, uint32_t h, uint32_t fmt, uint32_t state = 0x40) {
    memset(&r, 0, sizeof r);
    r.h.version = 1;
    r.kind = WG_RES_TEX2D;
    r.native = fake(id);
    r.width = w;
    r.height = h;
    r.native_format = fmt;
    r.state = state;
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

struct Lines {
  std::vector<std::string> v;
  size_t count(const std::string& needle) const {
    size_t n = 0;
    for (const auto& l : v) n += l.find(needle) != std::string::npos;
    return n;
  }
  std::string last(const std::string& needle) const {
    for (size_t i = v.size(); i-- > 0;)
      if (v[i].find(needle) != std::string::npos) return v[i];
    return std::string();
  }
};

struct Env {
  Lines lines;
  ProbeLog log;
  FakeGpu gpu;
  RrProbe* p = nullptr;
  uint64_t now = 1000;
  WgViewport vp;
  void* cmd = fake(200);
  void* dev = fake(201);
  // 32x18 colour in a 40x24 resource at (left 4, top 2) -> 64x36 output in a 70x40 resource at (left 3, top 1)
  Res colour{0, 40, 24, DXGI_FORMAT_R16G16B16A16_FLOAT}, output{1, 70, 40, DXGI_FORMAT_R16G16B16A16_FLOAT, 0x8};
  Res mv{2, 32, 18, DXGI_FORMAT_R16G16_FLOAT}, depth{3, 32, 18, DXGI_FORMAT_R32_FLOAT};
  Res nrm{4, 32, 18, DXGI_FORMAT_R10G10B10A2_UNORM}, dif{5, 32, 18, DXGI_FORMAT_R8G8B8A8_UNORM};
  std::vector<WgTag> tags;
  WgRrOptions opts;
  WgConstants consts;

  explicit Env(uint32_t minor = 14, ProbeLog::Limits lim = ProbeLog::Limits(), ProbeConfig base = ProbeConfig())
      : log([this](const std::string& l) { lines.v.push_back(l); }, lim) {
    ProbeConfig cfg = base;
    cfg.api_minor = minor;
    cfg.ids.viewport = kGViewport;
    cfg.ids.tag = kGTag;
    cfg.frame_token = kGToken;
    cfg.has_frame_token = true;
    cfg.release_version[1] = 5;
    cfg.fn_optimal = "fnOptimal"; cfg.fn_state = "fnState"; cfg.fn_options = "fnOptions";
    cfg.guid_names = {{"viewport", kGViewport}, {"tag", kGTag}, {"frame_token", kGToken}};
    p = new RrProbe(cfg, make_orig(), &gpu, &log, [] { return 4242u; });
    p->set_clock([this] { return now; });
    p->set_feature_function_pointers((void*)1, (void*)2, (void*)3);
    memset(&vp, 0, sizeof vp);
    vp.h.type = kGViewport;
    vp.h.version = 1;
    vp.id = 7;
    tags = {make_tag(&colour.r, WG_BT_COLOUR_IN, WG_LIFE_UNTIL_PRESENT, {2, 4, 32, 18}),
            make_tag(&output.r, WG_BT_COLOUR_OUT, WG_LIFE_UNTIL_PRESENT, {1, 3, 64, 36}),
            make_tag(&mv.r, WG_BT_MOTION), make_tag(&depth.r, WG_BT_DEPTH), make_tag(&dif.r, WG_BT_DIFFUSE_ALBEDO),
            make_tag(&nrm.r, WG_BT_NORMAL_ROUGHNESS)};
    memset(&opts, 0, sizeof opts);
    opts.h.version = 3;
    opts.mode = 1;
    opts.output_width = 64;
    opts.output_height = 36;
    opts.colour_is_hdr = 1;
    opts.normal_roughness_packing = 1;
    for (int i = 0; i < 16; ++i) opts.world_to_view[i] = (i % 5 == 0) ? 1.0f : 0.0f;
    opts.world_to_view[12] = 0.5f;
    memset(&consts, 0, sizeof consts);
    consts.h.version = 2;
    consts.view_to_clip[8] = 0.125f;
    consts.view_to_clip[9] = -0.25f;
    consts.jitter[0] = 0.25f;
    consts.jitter[1] = -0.375f;
    consts.mv_scale[0] = 1.0f / 32.0f;
    consts.mv_scale[1] = 1.0f / 18.0f;
    consts.camera_motion_included = 1;
    consts.mv_jittered = 0;
    consts.depth_inverted = 1;
    consts.mv_3d = 2;
    consts.mv_dilated = 0;
    consts.reset = 2;
  }
  ~Env() { delete p; }
  WgResultCode init(uint64_t flags = (1ull << 7) | 1ull) {
    static const uint32_t feats[2] = {1001, 0};
    WgPreferences pr;
    memset(&pr, 0, sizeof pr);
    pr.h.version = 1;
    pr.features = feats;
    pr.feature_count = 2;
    pr.flags = flags;
    pr.render_api = 1;
    return p->init(&pr, 0x0002000b0001fedcull);
  }
  void setup() {
    init();
    p->set_device(dev);
    p->rr_options(&vp, &opts);
    p->tag(&vp, tags.data(), (uint32_t)tags.size(), cmd);
  }
  WgResultCode frame(uint32_t idx = 1, const std::vector<const WgHeader*>& extra = {}) {
    WgFrameToken* t = nullptr;
    p->new_frame_token(&t, &idx);
    p->constants(&consts, t, &vp);
    std::vector<const WgHeader*> in = {&vp.h};
    in.insert(in.end(), extra.begin(), extra.end());
    return p->evaluate(1001, t, in.data(), (uint32_t)in.size(), cmd);
  }
};

// ---------------------------------------------------------------- tests
static void test_log_rules() {
  ProbeLog::Limits def;
  CHECK_EQ(def.full_frames, 600u);           // the first 600 frames in full
  CHECK(def.cap_bytes == 20ull * 1024 * 1024);  // 20 MB cap
  std::vector<std::string> out;
  ProbeLog::Limits lim;
  lim.full_frames = 3;
  lim.cap_bytes = 1 << 20;
  ProbeLog log([&](const std::string& l) { out.push_back(l); }, lim);
  CHECK(log.record("k", "a", "seq=1"));  // frame 0 (before the first frame): full
  for (int f = 1; f <= 3; ++f) {
    log.next_frame();
    CHECK(log.full_window());
    CHECK(log.record("k", "a", "seq=" + std::to_string(f)));  // frames 1..3: every record, also unchanged ones
  }
  log.next_frame();  // frame 4: changes only
  CHECK(!log.full_window());
  CHECK(!log.record("k", "a", "seq=9"));          // unchanged stable part: not written (the extra is not compared)
  CHECK(log.record("k", "b", "seq=10"));          // changed: written
  CHECK(!log.record("k", "b"));
  CHECK(log.record("k2", "b"));                   // a new key counts as a change
  CHECK(log.record("k", "a"));                    // back to an earlier value is a change too
  CHECK(log.note("marker"));                      // notes are always written
  CHECK_EQ(out.size(), (size_t)8);
  CHECK(out[0] == "f=0 k: a ; seq=1" && out[4] == "f=4 k: b ; seq=10" && out[7] == "f=4 marker");
  CHECK_EQ(log.suppressed(), 2u);
  // the size cap: one final line, nothing afterwards, never more than the cap
  std::vector<std::string> o2;
  ProbeLog::Limits small;
  small.cap_bytes = 400;
  ProbeLog capped([&](const std::string& l) { o2.push_back(l); }, small);
  int written = 0;
  for (int i = 0; i < 100; ++i) written += capped.record("key" + std::to_string(i), std::string(20, 'x'));
  CHECK(capped.capped() && written > 0 && written < 100);
  CHECK(o2.back().find("size cap reached") != std::string::npos);
  size_t total = 0;
  for (auto& l : o2) total += l.size() + 2;
  CHECK(total <= 400);
  CHECK(!capped.record("late", "x") && !capped.note("late"));
  CHECK_EQ(o2.size(), (size_t)written + 1);
}

static void test_claims_and_forwarding() {
  Env e;
  g_orig = OrigLog();
  bool b = false;
  WgRequirements rq;
  void* fp = nullptr;
  // 60.1 before <init>: 23 for RR, nothing forwarded
  CHECK_EQ(e.p->is_supported(1001, nullptr), 23);
  CHECK_EQ(e.p->evaluate(1001, nullptr, nullptr, 0, nullptr), 23);
  CHECK_EQ(g_orig.calls, 0);
  // <init>: forwarded unchanged; preference flags (bit 7), feature list, SDK version logged
  CHECK_EQ(e.init(), 0);
  CHECK(g_orig.last_role == "init" && g_orig.last_u64 == 0x0002000b0001fedcull);
  std::string il = e.lines.last(" init: sdk=");
  CHECK(il.find("sdk=2.11.1") != std::string::npos && il.find("magic=0xfedc") != std::string::npos);
  CHECK(il.find("bit7_frame_tagging=1") != std::string::npos && il.find("bit0_no_state_tracking=1") != std::string::npos);
  CHECK(il.find("features=[1001,0]") != std::string::npos && il.find("render_api=1") != std::string::npos);
  CHECK(e.lines.count("chain init.preferences: {") == 1);
  // claims of 60.8 without any runtime; device-needing calls 5 before <set-device>
  CHECK_EQ(e.p->is_supported(1001, nullptr), 0);
  CHECK_EQ(e.p->is_loaded(1001, &b), 0);
  CHECK(b);
  CHECK_EQ(e.p->get_function(1001, "fnOptimal", &fp), 5);
  memset(&rq, 0, sizeof rq);
  rq.h.version = 2;
  CHECK_EQ(e.p->requirements(1001, &rq), 0);
  CHECK(rq.flags == 2u && rq.max_viewports == 1 && rq.required_tag_count == 5 && rq.required_tags[0] == 3);
  WgFeatureVersion fv;
  memset(&fv, 0, sizeof fv);
  CHECK_EQ(e.p->version(1001, &fv), 0);
  CHECK(fv.api_version[1] == 11 && fv.runtime_version[1] == 5);
  CHECK_EQ(e.p->set_device(e.dev), 0);
  CHECK(e.gpu.device == reinterpret_cast<uint8_t*>(e.dev) + 1);  // unwrapped through the original
  CHECK_EQ(e.p->get_function(1001, "fnOptimal", &fp), 0);
  CHECK(fp == (void*)1);
  CHECK_EQ(e.p->get_function(1001, "fnOther", &fp), 25);
  CHECK(e.lines.count("call get-function feature=1001 name=fnOptimal: rc=0") == 1);
  CHECK(e.lines.count("call get-function feature=1001 name=fnOther: rc=25") == 1);
  // every other feature and call: forwarded unchanged, logged with its result
  g_orig = OrigLog();
  CHECK_EQ(e.p->is_supported(0, nullptr), 100);
  CHECK_EQ(e.p->get_function(1000, "frameGenName", &fp), 100);
  CHECK(g_orig.last_role == "get-function" && strcmp((const char*)g_orig.last_ptr, "frameGenName") == 0);
  CHECK(e.lines.count("call get-function feature=1000 name=frameGenName: rc=100") == 1);
  CHECK_EQ(e.p->requirements(1000, &rq), 100);
  CHECK_EQ(e.p->version(1000, &fv), 100);
  CHECK_EQ(e.p->allocate(e.cmd, 1000, &e.vp), 100);
  CHECK(g_orig.last_cmd == e.cmd);
  CHECK_EQ(e.p->free_resources(1000, &e.vp), 100);
  const WgHeader* in[1] = {&e.vp.h};
  CHECK_EQ(e.p->evaluate(1000, nullptr, in, 1, e.cmd), 100);
  CHECK(g_orig.last_role == "evaluate" && g_orig.last_feature == 1000 && g_orig.last_ptr == in);
  CHECK_EQ(e.p->set_vulkan_info(&rq), 9);
  CHECK(g_orig.last_role == "set-vulkan-info" && g_orig.last_ptr == &rq);
  void* iface = nullptr;
  CHECK_EQ(e.p->upgrade_interface(&iface), 0);
  CHECK(g_orig.last_role == "upgrade-interface" && g_orig.last_ptr == &iface);
  void* nat = nullptr;
  CHECK_EQ(e.p->native_interface(fake(9), &nat), 0);
  CHECK(nat == reinterpret_cast<uint8_t*>(fake(9)) + 1);
  e.p->note_forward("GraphicsEntryX");
  CHECK(e.lines.count("call GraphicsEntryX: forwarded (result not seen) ; seq=") == 1);
  CHECK(e.lines.count("call set-vulkan-info: rc=9") == 1 && e.lines.count("call upgrade-interface: rc=0") == 1);
  CHECK(e.lines.count("call native-interface: rc=0") == 1);
  // per call: sequence number and thread id
  CHECK(e.lines.last("call is-supported feature=0").find("tid=4242") != std::string::npos);
  // <requirements> and <version> calls are recorded (whether / when)
  CHECK(e.lines.count("first requirements: seq=") == 1 && e.lines.count("first version: seq=") == 1);
  // RR allocate / free: own (not forwarded), logged
  g_orig = OrigLog();
  CHECK_EQ(e.p->allocate(e.cmd, 1001, &e.vp), 0);
  CHECK_EQ(e.p->free_resources(1001, &e.vp), 0);
  CHECK_EQ(g_orig.calls, 0);
  CHECK(e.lines.count("first allocate: seq=") == 1 && e.lines.count("first free: seq=") == 1);
  // shutdown: forwarded; summary of the calls
  CHECK_EQ(e.p->shutdown(), 0);
  std::string sm = e.lines.last("summary at shutdown");
  CHECK(sm.find("allocate=2") != std::string::npos && sm.find("free=2") != std::string::npos &&
        sm.find("tag-for-frame=0") != std::string::npos && sm.find("requirements=2") != std::string::npos);
  CHECK_EQ(e.p->is_supported(1001, nullptr), 23);
}

static void test_rr_optimal_rule() {
  for (uint32_t minor : {7u, 14u}) {
    Env e(minor);
    e.init();
    for (uint32_t mode = 0; mode <= 7; ++mode)
      for (uint32_t w : {0u, 1u, 853u, 1920u, 2560u, 3840u}) {
        WgRrOptions o;
        memset(&o, 0, sizeof o);
        o.h.version = 3;
        o.mode = mode;
        o.output_width = w;
        o.output_height = w / 2 + 1;
        WgRrOptimal a, b;
        memset(&a, 0xcd, sizeof a);
        memset(&b, 0xcd, sizeof b);
        a.h.next = b.h.next = nullptr;  // an output structure as the game passes it: a valid (empty) chain
        WgResultCode ra = e.p->rr_optimal(&o, &a);
        WgResultCode rb = answer_optimal(minor, &o, &b);  // the real 61.5 rule, per Streamline variant
        CHECK_EQ(ra, rb);
        CHECK(memcmp(&a, &b, sizeof a) == 0);
      }
    CHECK(e.lines.count("rr-optimal mode=3 in=2560x1281 version=3: rc=0 optimal=1707x854") == 1);
  }
}

static void test_evaluate_upscale_and_fields() {
  Env e;
  e.setup();
  g_orig = OrigLog();
  CHECK_EQ(e.frame(1), 0);  // does not fail: the upscale is recorded and 0 returned
  CHECK_EQ(e.gpu.upscales, 1);
  CHECK(e.gpu.upscale_cmd == reinterpret_cast<uint8_t*>(e.cmd) + 1);  // the evaluate's list, unwrapped
  CHECK(e.gpu.src.resource == fake(0) && e.gpu.src.x == 4 && e.gpu.src.y == 2 && e.gpu.src.w == 32 && e.gpu.src.h == 18);
  CHECK(e.gpu.src.state == 0x40 && e.gpu.src.format == DXGI_FORMAT_R16G16B16A16_FLOAT);
  CHECK(e.gpu.dst.resource == fake(1) && e.gpu.dst.x == 3 && e.gpu.dst.y == 1 && e.gpu.dst.w == 64 && e.gpu.dst.h == 36);
  CHECK(e.gpu.dst.state == 0x8);
  CHECK(g_orig.last_role != "evaluate");  // RR is not forwarded
  // logged fields
  std::string ev = e.lines.last("evaluate vp=7:");
  CHECK(ev.find("render=32x18") != std::string::npos && ev.find("output=64x36") != std::string::npos);
  CHECK(ev.find("output_rect=3,1,64,36") != std::string::npos && ev.find("normal_roughness_mode=1") != std::string::npos);
  CHECK(ev.find("normal_type=14") != std::string::npos && ev.find("normal_format=24") != std::string::npos);
  CHECK(ev.find("roughness_format=in-normal") != std::string::npos);
  std::string cs = e.lines.last("constants vp=7:");
  CHECK(cs.find("camera_motion_included=1 mv_jittered=0 depth_inverted=1 mv_3d=2 mv_dilated=0") != std::string::npos);
  CHECK(cs.find("reset=2 mv_scale=0.03125,") != std::string::npos && cs.find("frame_index=501") != std::string::npos);
  std::string tg = e.lines.last("tag vp=7 type=3:");
  CHECK(tg.find("lifetime=1 extent=2,4,32,18 kind=0 tag_format=10 res=40x24 state=0x40") != std::string::npos);
  CHECK(tg.find("desc_format=10 desc=64x36") != std::string::npos);  // the resource's own description
  CHECK(e.lines.count("chain tag vp=7 type=3: tag{") == 1);
  std::string cam = e.lines.last("camera vp=7:");
  CHECK(cam.find("v2c20=0.125 v2c21=-0.25 prev_v2c20=- prev_v2c21=-") != std::string::npos);
  CHECK(cam.find("jitter=0.25,-0.375 prev_jitter=-") != std::string::npos);
  std::string ro = e.lines.last(" rr-options vp=7: mode=");
  CHECK(ro.find("mode=1 output=64x36") != std::string::npos && ro.find("normal_roughness_mode=1") != std::string::npos);
  CHECK(ro.find("version=3 alpha_upscaling=0 presets=0,0,0,0,0,0 release_rc=0") != std::string::npos);
  std::string ft = e.lines.last("frame-token first:");
  CHECK(ft.find("matches_frame_token_guid=1") != std::string::npos && ft.find("version_at_32=1") != std::string::npos);
  CHECK(ft.find("bytes_at_16=11353535") != std::string::npos);
  std::string call = e.lines.last("call evaluate feature=1001:");
  CHECK(call.find("rc=0 ; seq=") != std::string::npos && call.find("frame_index=501") != std::string::npos);
  CHECK(e.lines.count("colour-extent-watch: after=rr-options evaluate=1/2 extent=2,4,32,18 res=40x24") == 1);
  // second frame: the previous pair, the second watched evaluate; status bit 2
  e.consts.view_to_clip[8] = 0.5f;
  e.consts.jitter[0] = -0.125f;
  CHECK_EQ(e.frame(2), 0);
  cam = e.lines.last("camera vp=7:");
  CHECK(cam.find("v2c20=0.5 v2c21=-0.25 prev_v2c20=0.125 prev_v2c21=-0.25") != std::string::npos);
  CHECK(cam.find("jitter=-0.125,-0.375 prev_jitter=0.25,-0.375") != std::string::npos);
  CHECK(e.lines.count("colour-extent-watch: after=rr-options evaluate=2/2") == 1);
  CHECK_EQ(e.frame(3), 0);
  CHECK(e.lines.count("colour-extent-watch") == 2);  // only the first two after an event
  CHECK_EQ(e.p->status_word(), 7u);
  e.now += 1500;
  CHECK_EQ(e.p->status_word(), 3u);
  // <allocate> and a resize re-arm the watch
  CHECK_EQ(e.p->allocate(e.cmd, 1001, &e.vp), 0);
  CHECK_EQ(e.frame(4), 0);
  CHECK(e.lines.count("colour-extent-watch: after=allocate evaluate=1/2") == 1);
  e.tags[0].extent = {2, 4, 30, 16};
  e.p->tag(&e.vp, &e.tags[0], 1, e.cmd);
  CHECK_EQ(e.frame(5), 0);
  CHECK(e.lines.count("colour-extent-watch: after=resize evaluate=1/2 extent=2,4,30,16") == 1);
  CHECK(e.gpu.src.w == 30 && e.gpu.src.h == 16);
  // which RR call came first
  CHECK(e.lines.count("first rr-options: seq=") == 1 && e.lines.count("first evaluate: seq=") == 1);
}

static void test_evaluate_errors() {
  // unsupported format (62.5): 25, logged, nothing recorded
  {
    Env e;
    e.colour.r.native_format = DXGI_FORMAT_R32_UINT;
    e.setup();
    CHECK_EQ(e.frame(), 25);
    CHECK_EQ(e.gpu.upscales, 0);
    CHECK(e.lines.count("upscale vp=7: rc=25 unsupported colour format 42") == 1);
  }
  {
    Env e;
    e.output.r.native_format = DXGI_FORMAT_BC7_UNORM;
    e.setup();
    CHECK_EQ(e.frame(), 25);
    CHECK(e.lines.count("unsupported output format 98") == 1);
  }
  // missing output: 20
  {
    Env e;
    e.tags[1].resource = nullptr;
    e.setup();
    CHECK_EQ(e.frame(), 20);
    CHECK(e.lines.count("rc=20 missing output(4)") == 1);
  }
  // no viewport: 20; a GPU failure passes through with its reason
  {
    Env e;
    e.setup();
    WgFrameToken* t = nullptr;
    e.p->new_frame_token(&t, nullptr);
    CHECK_EQ(e.p->evaluate(1001, t, nullptr, 0, e.cmd), 20);
    e.gpu.upscale_rc = WG_E_MISSING_RESOURCE_STATE;
    CHECK_EQ(e.frame(), 18);
    CHECK(e.lines.count("upscale vp=7: rc=18 fake failure") == 1);
  }
  // no device yet: 5
  {
    Env e;
    e.init();
    const WgHeader* in[1] = {&e.vp.h};
    CHECK_EQ(e.p->evaluate(1001, nullptr, in, 1, e.cmd), 5);
  }
  // options the release build would refuse: accepted by the probe, the release answer logged
  {
    Env e;
    e.init();
    e.opts.mode = 7;  // a mode the release build refuses (61.3)
    CHECK_EQ(e.p->rr_options(&e.vp, &e.opts), 0);
    CHECK(e.lines.count("mode=7 output=64x36") == 1 && e.lines.last(" rr-options vp=7: mode=").find("release_rc=25") != std::string::npos);
  }
}

static void test_local_tags_lifetime0_and_frames() {
  Env e;
  e.tags[0].lifetime = WG_LIFE_ONLY_DURING_TAG;  // colour valid only during the tagging call: copied (62.1)
  e.setup();
  CHECK_EQ(e.gpu.copies, 1);
  CHECK(e.gpu.copy_cmd == reinterpret_cast<uint8_t*>(e.cmd) + 1 && e.gpu.copy_src.resource == fake(0));
  CHECK(e.lines.count("lifetime-0 copy tag vp=7 type=3: copied") == 1);
  CHECK_EQ(e.frame(), 0);
  CHECK(e.gpu.src.resource == e.gpu.copy_tex && e.gpu.src.state == 0x40);
  // a local tag (in the evaluate's inputs) takes precedence and is not copied
  Res other{30, 32, 18, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB};
  WgTag lt = make_tag(&other.r, WG_BT_COLOUR_IN, WG_LIFE_ONLY_DURING_TAG);
  CHECK_EQ(e.frame(2, {&lt.h}), 0);
  CHECK_EQ(e.gpu.copies, 1);
  CHECK(e.gpu.src.resource == fake(30) && e.gpu.src.w == 32 && e.gpu.src.format == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB);
  CHECK(e.lines.count("local-tag vp=7 type=3:") >= 1);
  // a chained structure in the inputs is logged with its GUID and version
  WgHeader extra;
  memset(&extra, 0, sizeof extra);
  extra.type = kGExtra;
  extra.version = 4;
  lt.h.next = &extra;
  CHECK_EQ(e.frame(3, {&lt.h}), 0);
  CHECK(e.lines.count("tag{5a5a5a5a-5a5a-5a5a-5a5a-5a5a5a5a5a02} v1 -> {5a5a5a5a-5a5a-5a5a-5a5a-5a5a5a5a5a09} v4") >= 1);
  // 2.14.1 frame-scoped tags: keyed by token, logged with the frame index
  WgFrameToken* t = nullptr;
  uint32_t idx = 40;
  e.p->new_frame_token(&t, &idx);
  Res fc{31, 32, 18, DXGI_FORMAT_R16G16B16A16_FLOAT};
  WgTag ftag = make_tag(&fc.r, WG_BT_COLOUR_IN);
  CHECK_EQ(e.p->tag_for_frame(t, &e.vp, &ftag, 1, e.cmd), 0);
  CHECK(g_orig.last_role == "tag-for-frame");
  e.p->constants(&e.consts, t, &e.vp);
  const WgHeader* in[1] = {&e.vp.h};
  CHECK_EQ(e.p->evaluate(1001, t, in, 1, e.cmd), 0);
  CHECK(e.gpu.src.resource == fake(31));
  CHECK(e.lines.last("tag-for-frame vp=7 type=3:").find("frame_index=540") != std::string::npos);
  CHECK(e.lines.count("first tag-for-frame: seq=") == 1);
}

static void test_normal_patch_samples() {
  ProbeConfig pc;
  pc.patch_first = 2;
  pc.patch_every = 3;
  pc.patch_count = 2;
  pc.patch_read_delay = 2;
  Env e(14, ProbeLog::Limits(), pc);
  e.setup();
  for (uint32_t i = 1; i <= 10; ++i) CHECK_EQ(e.frame(i), 0);
  CHECK_EQ(e.gpu.patches, 2);  // evaluates 2 and 5, then no more (count 2)
  CHECK(e.gpu.patch_src.resource == fake(4) && e.gpu.patch_px == 8 && e.gpu.patch_py == 1);  // centre of 32x18
  CHECK(e.lines.count("normal-patch sample=0: eval=2 type=14 format=24 origin=8,1 size=16x16 "
                      "mean=0.5,0.25,1,0.75 world_to_view=1,0,0,0,0,1,0,0,0,0,1,0,0.5,0,0,1") == 1);
  CHECK(e.lines.count("normal-patch sample=1: eval=5") == 1);
}

static void test_full_then_changes_only() {
  ProbeLog::Limits lim;
  lim.full_frames = 5;
  Env e(14, lim);
  e.setup();
  for (uint32_t i = 1; i <= 5; ++i) e.frame(i);
  const size_t calls_full = e.lines.count("call evaluate feature=1001: rc=0");
  CHECK_EQ(calls_full, (size_t)5);
  for (uint32_t i = 6; i <= 20; ++i) e.frame(i);
  CHECK_EQ(e.lines.count("call evaluate feature=1001: rc=0"), calls_full);  // unchanged: not repeated
  CHECK_EQ(e.lines.count("camera vp=7:"), (size_t)5);
  e.consts.view_to_clip[9] = 0.75f;  // a change is logged again
  e.frame(21);
  CHECK_EQ(e.lines.count("camera vp=7:"), (size_t)6);
  e.tags[1].resource = nullptr;  // a result change is logged
  e.p->tag(&e.vp, &e.tags[1], 1, e.cmd);
  e.frame(22);
  CHECK(e.lines.count("call evaluate feature=1001: rc=20") == 1);
}

static void test_view_formats() {
  bool d = false, enc = false;
  // typed resources: the resource's own format
  CHECK_EQ(probe_read_view_format(DXGI_FORMAT_R16G16B16A16_FLOAT, 0, &d), (uint32_t)DXGI_FORMAT_R16G16B16A16_FLOAT);
  CHECK(!d);
  CHECK_EQ(probe_read_view_format(DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, &d),
           (uint32_t)DXGI_FORMAT_R8G8B8A8_UNORM_SRGB);
  CHECK(!d);
  CHECK_EQ(probe_read_view_format(DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, &d),
           (uint32_t)DXGI_FORMAT_R8G8B8A8_UNORM);
  CHECK(d);  // tagged sRGB on a unorm resource: decoded in the shader
  // typeless resources: the tag's member, else the float member, else the unorm member
  CHECK_EQ(probe_read_view_format(DXGI_FORMAT_R16G16B16A16_TYPELESS, 0, &d), (uint32_t)DXGI_FORMAT_R16G16B16A16_FLOAT);
  CHECK_EQ(probe_read_view_format(DXGI_FORMAT_R8G8B8A8_TYPELESS, 0, &d), (uint32_t)DXGI_FORMAT_R8G8B8A8_UNORM);
  CHECK_EQ(probe_read_view_format(DXGI_FORMAT_R8G8B8A8_TYPELESS, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, &d),
           (uint32_t)DXGI_FORMAT_R8G8B8A8_UNORM_SRGB);
  CHECK_EQ(probe_read_view_format(DXGI_FORMAT_R10G10B10A2_TYPELESS, 0, &d), (uint32_t)DXGI_FORMAT_R10G10B10A2_UNORM);
  // refused: integer, block-compressed, typed depth
  CHECK_EQ(probe_read_view_format(DXGI_FORMAT_R32_UINT, 0, &d), 0u);
  CHECK_EQ(probe_read_view_format(DXGI_FORMAT_BC7_UNORM, 0, &d), 0u);
  CHECK_EQ(probe_read_view_format(DXGI_FORMAT_D32_FLOAT, 0, &d), 0u);
  // writing: no sRGB unordered access; a typeless sRGB output is written through its unorm member and encoded
  CHECK_EQ(probe_write_view_format(DXGI_FORMAT_R16G16B16A16_FLOAT, 0, &enc), (uint32_t)DXGI_FORMAT_R16G16B16A16_FLOAT);
  CHECK(!enc);
  CHECK_EQ(probe_write_view_format(DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, 0, &enc), 0u);
  CHECK_EQ(probe_write_view_format(DXGI_FORMAT_R8G8B8A8_TYPELESS, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, &enc),
           (uint32_t)DXGI_FORMAT_R8G8B8A8_UNORM);
  CHECK(enc);
  CHECK_EQ(probe_write_view_format(DXGI_FORMAT_B8G8R8A8_TYPELESS, DXGI_FORMAT_B8G8R8A8_UNORM_SRGB, &enc),
           (uint32_t)DXGI_FORMAT_B8G8R8A8_UNORM);
  CHECK_EQ(probe_write_view_format(DXGI_FORMAT_R11G11B10_FLOAT, 0, &enc), (uint32_t)DXGI_FORMAT_R11G11B10_FLOAT);
  CHECK_EQ(probe_write_view_format(DXGI_FORMAT_R32G32B32A32_UINT, 0, &enc), 0u);
}

// ---------------------------------------------------------------- capture option
static void test_capture_settings() {
  std::string notes;
  CaptureSettings d = parse_capture_settings("", &notes);
  CHECK(!d.enabled && d.start == 300 && d.count == 8 && d.every == 1 && d.read_delay == 2 && d.give_up == 240);
  CHECK(d.max_bytes == 2048ull << 20 && notes.empty());
  CaptureSettings s = parse_capture_settings(
      "\xef\xbb\xbf; comment\r\n[probe]\r\n  Capture = 1 ; on\r\ncapture_start=12\ncapture_count=3\ncapture_every=4\n"
      "capture_max_mb=64\ncapture_read_delay=5\n# done\n", &notes);
  CHECK(s.enabled && s.start == 12 && s.count == 3 && s.every == 4 && s.read_delay == 5 && s.max_bytes == 64ull << 20);
  CHECK(notes.empty());
  notes.clear();
  s = parse_capture_settings("capture=2\ncapture_start=0\ncapture_count=-1\ncapture_every=x\ncapture_max_mb=0\n"
                             "capture_read_delay=1001\nnoequals\nfoo=1\n", &notes);
  CHECK(!s.enabled && s.start == 300 && s.count == 8 && s.every == 1 && s.read_delay == 2);
  CHECK(s.max_bytes == 2048ull << 20);
  CHECK(notes.find("line 1: capture=2: not a number in range, default kept") != std::string::npos);
  CHECK(notes.find("line 6: capture_read_delay=1001: not a number in range") != std::string::npos);
  CHECK(notes.find("line 7: no '=': ignored") != std::string::npos);
  CHECK(notes.find("line 8: unknown key foo: ignored") != std::string::npos);
}

static void test_capture_trigger_file() {
  std::string notes;
  CaptureSettings s = parse_capture_settings("capture=1\ncapture_on_file=1\ncapture_trigger=x.txt\ncapture_dir=out dir\n", &notes);
  CHECK(s.enabled && s.on_file && s.trigger == "x.txt" && s.folder == "out dir" && notes.empty());
  CaptureSettings d = parse_capture_settings("capture=1\n", &notes);
  CHECK(!d.on_file && d.folder.empty());
  parse_capture_settings("capture_on_file=2\ncapture_trigger=\ncapture_dir=\n", &notes);
  CHECK(notes.find("capture_on_file=2: not a number in range") != std::string::npos);
  CHECK(notes.find("capture_trigger=: not a number in range") != std::string::npos);
  // flow: no window until the trigger file exists (checked at evaluates 16, 32, ...), then count consecutive evaluates
  const std::string trig = "wg_probe_trigger_test.tmp";
  remove(trig.c_str());
  Env e;
  FakeHost h;
  h.have_ini = true;
  h.ini = "capture=1\ncapture_on_file=1\ncapture_count=2\ncapture_read_delay=1\ncapture_trigger=" + trig +
          "\ncapture_dir=somewhere\n";
  e.gpu.host = &h;
  e.setup();
  e.gpu.cap_ready = true;
  for (uint32_t i = 1; i <= 20; ++i) CHECK_EQ(e.frame(i), 0);
  CHECK(h.folder == "somewhere" && e.gpu.cap_ids.empty());
  FILE* f = fopen(trig.c_str(), "wb");
  CHECK(f != nullptr);
  if (f) fclose(f);
  for (uint32_t i = 21; i <= 31; ++i) CHECK_EQ(e.frame(i), 0);
  CHECK(e.gpu.cap_ids.empty());  // the file is only looked at every 16th evaluate
  CHECK_EQ(e.frame(32), 0);
  CHECK(e.lines.count("capture: trigger file seen at eval=32, window starts at eval=33") == 1);
  CHECK(e.gpu.cap_ids.empty());
  FILE* g = fopen(trig.c_str(), "rb");  // removed when seen
  CHECK(g == nullptr);
  if (g) fclose(g);
  for (uint32_t i = 33; i <= 40; ++i) CHECK_EQ(e.frame(i), 0);
  CHECK(e.gpu.cap_ids.size() == 2 && e.gpu.cap_ids[0] == 33 && e.gpu.cap_ids[1] == 34);
  for (uint32_t i = 41; i <= 64; ++i) CHECK_EQ(e.frame(i), 0);  // capture_shots=1: no second window
  CHECK(e.gpu.cap_ids.size() == 2);
  remove(trig.c_str());
  {  // capture_shots=2: a second trigger file starts a second window after the first one is complete
    Env e2;
    FakeHost h2;
    h2.have_ini = true;
    h2.ini = "capture=1\ncapture_on_file=1\ncapture_count=2\ncapture_shots=2\ncapture_read_delay=1\ncapture_trigger=" + trig + "\n";
    e2.gpu.host = &h2;
    e2.setup();
    e2.gpu.cap_ready = true;
    FILE* t1 = fopen(trig.c_str(), "wb");
    CHECK(t1 != nullptr);
    if (t1) fclose(t1);
    for (uint32_t i = 1; i <= 16; ++i) CHECK_EQ(e2.frame(i), 0);
    CHECK(e2.lines.count("capture: trigger file seen at eval=16, window starts at eval=17") == 1);
    for (uint32_t i = 17; i <= 20; ++i) CHECK_EQ(e2.frame(i), 0);
    CHECK(e2.gpu.cap_ids.size() == 2 && e2.gpu.cap_ids[1] == 18);
    CHECK(e2.lines.count("capture: window 1 of 2 complete at eval=19, waiting for the trigger file again") == 1);
    FILE* t2 = fopen(trig.c_str(), "wb");
    CHECK(t2 != nullptr);
    if (t2) fclose(t2);
    for (uint32_t i = 21; i <= 40; ++i) CHECK_EQ(e2.frame(i), 0);
    CHECK(e2.lines.count("capture: trigger file seen at eval=32, window starts at eval=33") == 1);
    CHECK(e2.gpu.cap_ids.size() == 4 && e2.gpu.cap_ids[2] == 33 && e2.gpu.cap_ids[3] == 34);
    FILE* t3 = fopen(trig.c_str(), "wb");  // the shots are used up: a third file does nothing
    if (t3) fclose(t3);
    for (uint32_t i = 41; i <= 64; ++i) CHECK_EQ(e2.frame(i), 0);
    CHECK(e2.gpu.cap_ids.size() == 4);
    remove(trig.c_str());
  }
}

static void test_capture_off() {
  {  // no capture host (a GPU side without capture): off, nothing copied
    Env e;
    e.setup();
    for (uint32_t i = 1; i <= 4; ++i) CHECK_EQ(e.frame(i), 0);
    CHECK(e.lines.count("capture: off (not supported)") == 1);
    CHECK(e.gpu.cap_ids.empty());
  }
  {  // no settings file
    Env e;
    FakeHost h;
    e.gpu.host = &h;
    e.setup();
    for (uint32_t i = 1; i <= 4; ++i) CHECK_EQ(e.frame(i), 0);
    CHECK(e.lines.count("capture: off (no probe settings file)") == 1);
    CHECK(e.gpu.cap_ids.empty() && h.opens == 0 && h.files.empty());
    e.p->shutdown();
    CHECK(h.flushes == 0 && e.lines.count("capture at shutdown") == 0);
  }
  {  // settings file without capture=1
    Env e;
    FakeHost h;
    h.have_ini = true;
    h.ini = "capture_start=1\n";
    e.gpu.host = &h;
    e.setup();
    for (uint32_t i = 1; i <= 4; ++i) CHECK_EQ(e.frame(i), 0);
    CHECK(e.lines.count("capture: off (capture=1 not set in the probe settings file)") == 1);
    CHECK(e.gpu.cap_ids.empty() && h.opens == 0);
  }
  {  // the folder cannot be prepared: off
    Env e;
    FakeHost h;
    h.have_ini = true;
    h.ini = "capture=1\ncapture_start=1\n";
    h.open_ok = false;
    e.gpu.host = &h;
    e.setup();
    for (uint32_t i = 1; i <= 3; ++i) CHECK_EQ(e.frame(i), 0);
    CHECK(e.lines.count("capture: capture=1, but the capture folder could not be prepared (fake-folder); capture off") == 1);
    CHECK(e.gpu.cap_ids.empty());
  }
}

static void test_capture_flow() {
  Env e;
  FakeHost h;
  h.have_ini = true;
  h.ini = "capture=1\ncapture_start=3\ncapture_count=2\ncapture_every=2\ncapture_read_delay=2\n";
  e.gpu.host = &h;
  e.setup();
  Res nulltag{60, 1, 1, 10};
  WgTag t40 = make_tag(&nulltag.r, WG_BT_COLOUR_BEFORE_TRANSPARENCY);
  t40.resource = nullptr;  // a null resource removes the tag: not captured
  e.p->tag(&e.vp, &t40, 1, e.cmd);
  Res spec{61, 32, 18, DXGI_FORMAT_R8G8B8A8_UNORM, 0x8};
  WgTag local = make_tag(&spec.r, WG_BT_SPECULAR_ALBEDO);  // a local tag of evaluate 3 only
  e.gpu.cap_ready = false;
  for (uint32_t i = 1; i <= 2; ++i) CHECK_EQ(e.frame(i), 0);
  CHECK(e.gpu.cap_ids.empty());
  CHECK(e.lines.count("capture: on, start=3 count=2 every=2 max_mb=2048 read_delay=2 folder=fake-folder") == 1);
  CHECK_EQ(h.opens, 1);
  CHECK_EQ(e.frame(3, {&local.h}), 0);
  CHECK(e.gpu.cap_ids.size() == 1 && e.gpu.cap_ids[0] == 3);
  CHECK(e.gpu.cap_cmd == reinterpret_cast<uint8_t*>(e.cmd) + 1);  // the evaluate's list, unwrapped
  // inputs: every tagged input of the evaluate, the output left out, ordered by type
  std::vector<uint32_t> types;
  for (const CaptureSource& c : e.gpu.cap_last) types.push_back(c.type);
  CHECK((types == std::vector<uint32_t>{0, 1, 3, 7, 8, 14}));
  const CaptureSource& col = e.gpu.cap_last[2];
  CHECK(col.view.resource == fake(0) && col.view.x == 4 && col.view.y == 2 && col.view.w == 32 && col.view.h == 18);
  CHECK(col.view.state == 0x40 && col.view.format == DXGI_FORMAT_R16G16B16A16_FLOAT);
  CHECK(e.gpu.cap_last[4].view.resource == fake(61) && e.gpu.cap_last[4].view.state == 0x8);
  CHECK(e.lines.count("capture eval=3: copied 6 of 6 input(s) (types 0,1,3,7,8,14), 1000 bytes of readback memory") == 1);
  CHECK_EQ(e.frame(4), 0);
  CHECK_EQ(e.frame(5), 0);  // read attempted at evaluate 5: no marker yet, kept
  CHECK(h.files.empty());
  CHECK(e.gpu.cap_ids.size() == 2 && e.gpu.cap_ids[1] == 5);
  types.clear();
  for (const CaptureSource& c : e.gpu.cap_last) types.push_back(c.type);
  CHECK((types == std::vector<uint32_t>{0, 1, 3, 7, 14}));  // no local tag this time
  e.gpu.cap_ready = true;
  CHECK_EQ(e.frame(6), 0);  // evaluate 3 read back and written; evaluate 5 not yet due (5 + 2)
  CHECK_EQ(h.files.size(), (size_t)7);
  CHECK(h.files.back().first == "frame_000003/frame.json");  // frame.json last
  const std::vector<uint8_t>* f3 = h.file("frame_000003/input_03.raw");
  CHECK(f3 && f3->size() == 64 + 16);
  if (f3 && f3->size() == 80) {
    CHECK(memcmp(f3->data(), "WGRRCAP1", 8) == 0);
    CHECK(le32(*f3, 8) == 64 && le32(*f3, 12) == 3 && le32(*f3, 16) == DXGI_FORMAT_R16G16B16A16_FLOAT);
    CHECK(le32(*f3, 20) == 4 && le32(*f3, 24) == 2 && le32(*f3, 28) == 8);
    CHECK(le32(*f3, 32) == DXGI_FORMAT_R16G16B16A16_FLOAT && le32(*f3, 36) == 0x40);
    CHECK(le32(*f3, 40) == 4 && le32(*f3, 44) == 2 && le32(*f3, 48) == 32 && le32(*f3, 52) == 18);
    CHECK(le32(*f3, 56) == WG_LIFE_UNTIL_PRESENT && le32(*f3, 60) == 0);
    CHECK((*f3)[64] == 48 && (*f3)[79] == 63);
  }
  CHECK(h.file("frame_000003/input_08.raw") && !h.file("frame_000003/input_04.raw") &&
        !h.file("frame_000003/input_40.raw"));
  const std::string j = h.text("frame_000003/frame.json");
  CHECK(j.find("{\"format\":\"wronggpu-rr-capture\",\"version\":1,\"evaluate\":3,\"frame_index\":503,\"viewport\":7,"
               "\"api_minor\":14,\"preference_flags\":129,\"render\":{\"x\":4,\"y\":2,\"width\":32,\"height\":18}") == 0);
  CHECK(j.find("\"output\":{\"type\":4,\"tag_format\":10,\"format\":10,\"width\":64,\"height\":36,\"rect\":[3,1,64,36],"
               "\"state\":8}") != std::string::npos);
  CHECK(j.find("\"options\":{\"given\":true,\"version\":3,\"mode\":1,\"output_width\":64,\"output_height\":36,") !=
        std::string::npos);
  CHECK(j.find("\"world_to_view\":[1,0,0,0,0,1,0,0,0,0,1,0,0.5,0,0,1]") != std::string::npos);
  CHECK(j.find("\"jitter\":[0.25,-0.375],\"mv_scale\":[0.03125,0.055555556]") != std::string::npos);
  CHECK(j.find("\"depth_inverted\":1,\"camera_motion_included\":1,\"mv_3d\":2,\"reset\":2") != std::string::npos);
  CHECK(j.find("\"camera\":{\"jitter\":[0.25,-0.375],\"view_to_clip\":[0,0,0,0,0,0,0,0,0.125,-0.25,") !=
        std::string::npos);
  CHECK(j.find("{\"type\":3,\"file\":\"input_03.raw\",\"format\":10,\"width\":4,\"height\":2,\"row_pitch\":8,"
               "\"tag_format\":10,\"state\":64,\"lifetime\":1,\"kind\":0,\"extent\":{\"left\":4,\"top\":2,\"width\":32,"
               "\"height\":18},\"tag_size\":[40,24]}") != std::string::npos);
  CHECK(j.find("\"not_copied\":[]}") != std::string::npos && j.back() == '\n');
  CHECK(e.lines.count("capture eval=3: written frame_000003 (6 input file(s) + frame.json, ") == 1);
  CHECK_EQ(e.frame(7), 0);
  CHECK(h.file("frame_000005/frame.json") != nullptr);
  CHECK(e.lines.count("capture: done, 2 of 2 evaluate(s) written, 2000 bytes read back") == 1);
  for (uint32_t i = 8; i <= 12; ++i) CHECK_EQ(e.frame(i), 0);
  CHECK_EQ(e.gpu.cap_ids.size(), (size_t)2);  // count 2: no more
  e.p->shutdown();
  CHECK(h.flushes == 1 && e.lines.count("capture at shutdown: 2 evaluate(s) written, 2000 bytes read back, 0 failed") == 1);
}

static void test_capture_cap_late_and_shutdown() {
  {  // size cap: the second evaluate does not fit; a marker that never comes is abandoned
    Env e;
    FakeHost h;
    h.have_ini = true;
    h.ini = "capture=1\ncapture_start=1\ncapture_count=5\ncapture_max_mb=1\n";
    e.gpu.host = &h;
    e.gpu.cap_bytes_each = 600u << 10;
    e.gpu.cap_ready = false;
    e.setup();
    for (uint32_t i = 1; i <= 4; ++i) CHECK_EQ(e.frame(i), 0);
    CHECK_EQ(e.gpu.cap_ids.size(), (size_t)1);
    CHECK(e.lines.count("capture eval=2: not taken: size cap: 614400 bytes needed, 434176 left") == 1);
    CHECK(e.lines.count("capture: size cap of 1 MB reached after 614400 bytes; no further evaluates captured") == 1);
    CHECK(e.lines.count("capture eval=3") == 0);
    for (uint32_t i = 5; i <= 242; ++i) CHECK_EQ(e.frame(i), 0);
    CHECK(e.gpu.abandoned.empty());
    CHECK_EQ(e.frame(243), 0);  // 1 + read delay 2 + 240
    CHECK(e.gpu.abandoned.size() == 1 && e.gpu.abandoned[0] == 1);
    CHECK(e.lines.count("capture eval=1: not read back (completion marker not seen after 242 evaluates); dropped") == 1);
    CHECK(e.lines.count("capture: done, 0 of 5 evaluate(s) written") == 1 && h.files.empty());
  }
  {  // a capture still pending at shutdown is read there (the game has finished its frames) and flushed
    Env e;
    FakeHost h;
    h.have_ini = true;
    h.ini = "capture=1\ncapture_start=2\ncapture_count=1\ncapture_read_delay=50\n";
    e.gpu.host = &h;
    e.setup();
    for (uint32_t i = 1; i <= 3; ++i) CHECK_EQ(e.frame(i), 0);
    CHECK(h.files.empty());
    e.p->shutdown();
    CHECK(h.file("frame_000002/frame.json") != nullptr && h.flushes == 1);
    CHECK(e.lines.count("capture at shutdown: 1 evaluate(s) written") == 1);
  }
  {  // not complete at shutdown: abandoned, nothing written
    Env e;
    FakeHost h;
    h.have_ini = true;
    h.ini = "capture=1\ncapture_start=1\ncapture_count=1\n";
    e.gpu.host = &h;
    e.gpu.cap_ready = false;
    e.setup();
    CHECK_EQ(e.frame(1), 0);
    e.p->shutdown();
    CHECK(e.gpu.abandoned.size() == 1 && h.files.empty());
    CHECK(e.lines.count("capture eval=1: not read back (completion marker not seen at shutdown); dropped") == 1);
  }
}

// clean/rr-re9-pt: a game that fills no size and tags no extent (RE Requiem) gets the native description's size; the
// probe's passes on the game's list are bracketed by the host's restore (one begin / end pair per RR evaluate).
static void test_desc_size_and_host_bracket() {
  Env e;
  std::vector<std::string> order;
  int begins = 0, ends = 0;
  void* begin_cl = nullptr;
  void* end_cl = nullptr;
  CmdListBracket b;
  b.begin = [&](void* cl) { ++begins; begin_cl = cl; order.push_back("begin"); };
  b.end = [&](void* cl) { ++ends; end_cl = cl; order.push_back("end:" + std::to_string(e.gpu.upscales)); };
  e.p->set_cmd_list_bracket(b);
  e.colour.r.width = e.colour.r.height = 0;
  e.output.r.width = e.output.r.height = 0;
  e.tags[0].extent = {0, 0, 0, 0};
  e.tags[1].extent = {0, 0, 0, 0};
  e.setup();
  CHECK_EQ(e.frame(1), 0);
  CHECK_EQ(e.gpu.upscales, 1);  // fake describe: 64 x 36
  CHECK(e.gpu.src.x == 0 && e.gpu.src.y == 0 && e.gpu.src.w == 64 && e.gpu.src.h == 36);
  CHECK(e.gpu.dst.x == 0 && e.gpu.dst.y == 0 && e.gpu.dst.w == 64 && e.gpu.dst.h == 36);
  std::string ev = e.lines.last("evaluate vp=7:");
  CHECK(ev.find("render=64x36") != std::string::npos);
  std::string tg = e.lines.last("tag vp=7 type=3:");
  CHECK(tg.find("res=0x0") != std::string::npos);  // the log still shows what the game filled
  CHECK_EQ(begins, 1);
  CHECK_EQ(ends, 1);
  CHECK(begin_cl == reinterpret_cast<uint8_t*>(e.cmd) + 1 && end_cl == begin_cl);  // the native list
  CHECK(order.size() == 2 && order[0] == "begin" && order[1] == "end:1");  // the upscale ran inside the pair
  // an evaluate that fails before any pass still closes its pair; another feature is not bracketed
  e.output.r.native_format = DXGI_FORMAT_R32_UINT;
  e.p->tag(&e.vp, &e.tags[1], 1, e.cmd);
  CHECK(e.frame(2) != 0);
  CHECK_EQ(begins, 2);
  CHECK_EQ(ends, 2);
  WgFrameToken* t = nullptr;
  uint32_t idx = 3;
  e.p->new_frame_token(&t, &idx);
  const WgHeader* in[1] = {&e.vp.h};
  e.p->evaluate(0, t, in, 1, e.cmd);
  CHECK_EQ(begins, 2);
  // no bracket set: nothing is called
  e.p->set_cmd_list_bracket(CmdListBracket());
  e.output.r.native_format = DXGI_FORMAT_R16G16B16A16_FLOAT;
  e.p->tag(&e.vp, &e.tags[1], 1, e.cmd);
  CHECK_EQ(e.frame(4), 0);
  CHECK_EQ(begins, 2);
  CHECK_EQ(ends, 2);
}

int main() {
  test_log_rules();
  test_claims_and_forwarding();
  test_rr_optimal_rule();
  test_evaluate_upscale_and_fields();
  test_evaluate_errors();
  test_local_tags_lifetime0_and_frames();
  test_normal_patch_samples();
  test_full_then_changes_only();
  test_view_formats();
  test_capture_settings();
  test_capture_off();
  test_capture_flow();
  test_capture_trigger_file();
  test_capture_cap_late_and_shutdown();
  test_desc_size_and_host_bracket();
  TEST_SUMMARY("test_probe_core");
}
