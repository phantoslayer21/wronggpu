// SPDX-License-Identifier: MIT
// Unit tests of the NGX RR provider (src/ngx_rr.*, spec page 66) against a fake runtime and a mock parameter block:
// create keys -> WgRrCreate, refusals (bad keys, MV at output resolution, runtime not ready, setting off, the other
// route busy), evaluate keys -> WgRrFrame (routing, origins, states, MV scale as given, jitter unchanged, provider-kept
// previous matrices, dynamic resolution), rejections (missing inputs, formats, sizes), the fail-safe, release and
// handle bookkeeping, the host bracket around runtime work (root-state restore) and its closing after a fault, integer
// keys read with the other integer type than the game set them (a host that does not convert), and the same captured game frames
// sent through the game-interface core (proxy_core) and through the NGX provider: the runtime must receive the same
// creation and the same frames (same runtime, so the same output).
//   test_ngx_rr.exe [<capture frames json>]   (default: tests/data/cp2077-2.14-rr-frames.json next to this file)
#include <dxgiformat.h>
#include <stdio.h>
#include <string.h>

#include <string>
#include <vector>

#include "../src/ngx_rr.h"
#include "../src/proxy_core.h"
#include "fake_runtime.h"
#include "mini_json.h"
#include "mock_ngx_params.h"
#include "test_util.h"

using namespace wgrr;

static uint8_t g_pool[512];
static void* R(int i) { return &g_pool[i]; }

static const uint32_t kFlagsStd = 0x1u | 0x2u | 0x8u;  // HDR, vectors at render resolution, depth inverted

// The standard scene: 1280x720 -> 1920x1080, packed normals, hardware depth, all required inputs.
struct Env {
  FakeRuntime rt;
  Settings settings;
  bool busy = false, rt_present = true;
  int frames_done = 0, begins = 0, ends = 0;
  void* last_bracket_cl = nullptr;
  std::vector<std::string> lines;
  NgxRrProvider* p = nullptr;
  MockNgxParams cp, ep;  // create and evaluate parameter blocks
  void* dev = R(500);
  void* cl = R(501);
  float w2v[16], v2c[16];

  Env() {
    settings.set_save_enabled(false);
    NgxRrDeps d;
    d.runtime = [this]() -> const WgRrRuntimeV1* { return rt_present ? &rt.table : nullptr; };
    d.settings = [this] { return &settings; };
    d.log = [this](const std::string& s) { lines.push_back(s); };
    d.clock = [] { return (uint64_t)1000; };
    d.other_route_busy = [this] { return busy; };
    d.frame_done = [this] { ++frames_done; };
    p = new NgxRrProvider(std::move(d));
    CmdListBracket b;
    b.begin = [this](void* c) { ++begins; last_bracket_cl = c; };
    b.end = [this](void* c) { ++ends; last_bracket_cl = c; };
    p->set_cmd_list_bracket(std::move(b));
    cp.set_u("Width", 1280);
    cp.set_u("Height", 720);
    cp.set_u("OutWidth", 1920);
    cp.set_u("OutHeight", 1080);
    cp.set_i("PerfQualityValue", 2);  // quality -> mode 3
    cp.set_i("DLSS.Feature.Create.Flags", (int32_t)kFlagsStd);
    cp.set_i("DLSS.Denoise.Mode", 1);
    cp.set_i("DLSS.Roughness.Mode", 1);
    cp.set_i("DLSS.Use.HW.Depth", 1);
    cp.set_u("RayReconstruction.Hint.Render.Preset.Quality", 6);
    add(0, "Color", DXGI_FORMAT_R16G16B16A16_FLOAT, 1280, 720);
    add(1, "Output", DXGI_FORMAT_R16G16B16A16_FLOAT, 1920, 1080);
    add(2, "Depth", DXGI_FORMAT_R32_TYPELESS, 1280, 720);
    add(3, "MotionVectors", DXGI_FORMAT_R16G16_FLOAT, 1280, 720);
    add(4, "DLSS.Input.DiffuseAlbedo", DXGI_FORMAT_R8G8B8A8_UNORM, 1280, 720);
    add(5, "DLSS.Input.SpecularAlbedo", DXGI_FORMAT_R8G8B8A8_UNORM, 1280, 720);
    add(6, "GBuffer.Normals", DXGI_FORMAT_R10G10B10A2_UNORM, 1280, 720);
    for (int i = 0; i < 16; ++i) {
      w2v[i] = (i % 5 == 0) ? 1.0f : 0.0f;
      v2c[i] = (i % 5 == 0) ? 2.0f : 0.0f;
    }
    w2v[12] = 3.0f;
    ep.set_ptr("WorldToViewMatrix", w2v);
    ep.set_ptr("ViewToClipMatrix", v2c);
    ep.set_f("Jitter.Offset.X", 0.25f);
    ep.set_f("Jitter.Offset.Y", -0.375f);
    ep.set_f("MV.Scale.X", 1280.0f);
    ep.set_f("MV.Scale.Y", 720.0f);
    ep.set_i("Reset", 0);
  }
  ~Env() { delete p; }
  void add(int id, const char* key, uint32_t fmt, uint32_t w, uint32_t h, uint32_t samples = 1) {
    ep.set_res(key, R(id));
    rt.described[R(id)] = {fmt, samples};
    rt.sized[R(id)] = {w, h};
  }
  void* create(bool expect = true) {
    WgHostNgxParams v = cp.view();
    void* ctx = (void*)0x1;
    const bool ok = p->create(dev, cl, &v, &ctx);
    CHECK_EQ(ok, expect);
    if (!ok) CHECK(ctx == nullptr);
    return ok ? ctx : nullptr;
  }
  bool eval(void* ctx) {
    WgHostNgxParams v = ep.view();
    return p->evaluate(ctx, cl, &v);
  }
  uint32_t count(const std::string& needle) const {
    uint32_t n = 0;
    for (const auto& l : lines) n += l.find(needle) != std::string::npos;
    return n;
  }
};

static void test_create() {
  Env e;
  void* ctx = e.create();
  CHECK(ctx != nullptr);
  CHECK_EQ(e.p->live(), (size_t)1);
  CHECK_EQ(e.rt.creates, 1);
  CHECK(e.rt.device == e.dev);  // the game's device went to set_device before the creation
  const WgRrCreate& d = e.rt.created.back();
  CHECK_EQ(d.size, (uint32_t)sizeof(WgRrCreate));
  CHECK_EQ(d.api_minor, kNgxApiMinor);
  CHECK(d.device == e.dev);
  CHECK(d.render_width == 1280 && d.render_height == 720 && d.output_width == 1920 && d.output_height == 1080);
  CHECK_EQ(d.mode, 3u);  // quality 2 + 1
  CHECK_EQ(d.packing, 1u);
  CHECK(d.colour_hdr == 1 && d.depth_inverted == 1 && d.mv_jittered == 0 && d.mv_output_resolution == 0);
  CHECK(d.hw_depth == 1 && d.alpha_upscaling == 0 && d.sharpen_request == 0 && d.latched_settings == 1);
  CHECK(d.presets[1] == 6 && d.presets[0] == 0);
  CHECK(e.p->create_desc(ctx) && e.p->create_desc(ctx)->mode == 3);
  CHECK(e.count("NGX: RR feature 1 created: render 1280x720 -> output 1920x1080, quality 2 (mode 3), flags 0xb") == 1);
  // the mode numbering: 0 performance .. 5 DLAA -> 1 .. 6
  for (int q = 0; q <= 5; ++q) {
    e.cp.set_i("PerfQualityValue", q);
    void* c = e.create();
    CHECK(c && e.rt.created.back().mode == (uint32_t)q + 1u);
  }
  // other flags: jittered vectors, alpha, sharpening, linear depth, separate roughness
  e.cp.set_i("DLSS.Feature.Create.Flags", (int32_t)(0x2u | 0x4u | 0x20u | 0x80u));
  e.cp.set_i("DLSS.Use.HW.Depth", 0);
  e.cp.set_i("DLSS.Roughness.Mode", 0);
  void* c2 = e.create();
  const WgRrCreate& d2 = e.rt.created.back();
  CHECK(c2 && d2.colour_hdr == 0 && d2.mv_jittered == 1 && d2.alpha_upscaling == 1 && d2.sharpen_request == 1 &&
        d2.depth_inverted == 0 && d2.hw_depth == 0 && d2.packing == 0);
  // absent hardware-depth / denoise keys: hardware depth and unified taken, logged
  e.cp.i.erase("DLSS.Use.HW.Depth");
  e.cp.i.erase("DLSS.Denoise.Mode");
  void* c3 = e.create();
  CHECK(c3 && e.rt.created.back().hw_depth == 1);
  CHECK(e.count("(key absent: hardware taken)") == 1);
  CHECK_EQ(e.p->live(), (size_t)9);
  e.p->release_all();
  CHECK_EQ(e.p->live(), (size_t)0);
  CHECK_EQ(e.rt.destroys, 9);
  CHECK(e.rt.live.empty());
}

static void test_create_refusals() {
  struct Case { const char* name; void (*setup)(Env&); const char* log; };
  const Case cases[] = {
      {"no width", [](Env& e) { e.cp.u.erase("Width"); }, "render or output size missing"},
      {"no output height", [](Env& e) { e.cp.u.erase("OutHeight"); }, "render or output size missing"},
      {"denoise 0", [](Env& e) { e.cp.set_i("DLSS.Denoise.Mode", 0); }, "denoise mode 0"},
      {"mv at output res", [](Env& e) { e.cp.set_i("DLSS.Feature.Create.Flags", 0x1 | 0x8); },
       "motion vectors at output resolution are not supported"},
      {"roughness 2", [](Env& e) { e.cp.set_i("DLSS.Roughness.Mode", 2); }, "roughness mode 2"},
      {"render > output", [](Env& e) { e.cp.set_u("Width", 2000); }, "outside 1/3..1 of output"},
      {"render < 1/3", [](Env& e) { e.cp.set_u("Height", 300); }, "outside 1/3..1 of output"},
      {"runtime missing", [](Env& e) { e.rt_present = false; }, "RR runtime is not loaded"},
      {"runtime not ready", [](Env& e) { e.rt.status_bits = WG_RR_RT_READY; }, "runtime or model is not available"},
      {"create fails", [](Env& e) { e.rt.create_rc = 11; }, "could not create the instance (11)"},
      {"other route", [](Env& e) { e.busy = true; }, "the game-interface route holds the RR feature"},
      {"setting off", [](Env& e) { e.settings.set(kSetWrongGpuRr, 0); }, "setting \"WrongGPU RR\" is off"},
  };
  for (const Case& c : cases) {
    Env e;
    c.setup(e);
    void* ctx = e.create(false);
    CHECK(ctx == nullptr);
    CHECK_EQ(e.p->live(), (size_t)0);
    CHECK(e.rt.live.empty());
    if (e.count(c.log) != 1) fprintf(stderr, "case %s: log line missing (%s)\n", c.name, c.log);
    CHECK_EQ(e.count(c.log), 1u);
  }
  {  // no parameter access, no context pointer
    Env e;
    WgHostNgxParams v = e.cp.view();
    CHECK(!e.p->create(e.dev, e.cl, nullptr, nullptr));
    CHECK(!e.p->create(e.dev, e.cl, nullptr, (void**)&v.block));
    WgHostNgxParams none = v;
    none.get_uint = nullptr;
    void* ctx = nullptr;
    CHECK(!e.p->create(e.dev, e.cl, &none, &ctx) && ctx == nullptr);
    CHECK(e.rt.creates == 0);
  }
  {  // no device from the host: taken from the command list
    Env e;
    int asked = 0;
    NgxRrDeps d;
    d.runtime = [&e]() -> const WgRrRuntimeV1* { return &e.rt.table; };
    d.device_of = [&asked](void* c) -> void* { ++asked; return (uint8_t*)c + 2; };
    NgxRrProvider p(std::move(d));
    WgHostNgxParams v = e.cp.view();
    void* ctx = nullptr;
    CHECK(p.create(nullptr, e.cl, &v, &ctx) && ctx);
    CHECK(asked == 1 && e.rt.device == (uint8_t*)e.cl + 2 && e.rt.created.back().device == (uint8_t*)e.cl + 2);
  }
}

static void test_evaluate() {
  Env e;
  void* ctx = e.create();
  CHECK(e.eval(ctx));
  CHECK_EQ(e.rt.evaluates, 1);
  CHECK_EQ(e.frames_done, 1);
  CHECK(e.begins == 1 && e.ends == 1 && e.last_bracket_cl == e.cl);  // root-state restore around the runtime's work
  const WgRrFrame& f = e.rt.last();
  CHECK_EQ(f.size, (uint32_t)sizeof(WgRrFrame));
  CHECK_EQ(f.api_minor, kNgxApiMinor);
  CHECK(f.command_list == e.cl);
  CHECK(f.render_width == 1280 && f.render_height == 720 && f.output_width == 1920 && f.output_height == 1080);
  CHECK(f.jitter[0] == 0.25f && f.jitter[1] == -0.375f);       // unchanged
  CHECK(f.mv_scale[0] == 1280.0f && f.mv_scale[1] == 720.0f);  // taken as given, no multiplication
  CHECK(f.reset == 1);                                          // creation makes a reset pending
  CHECK(f.parity == 0 && f.counter == 0 && f.rebuild_render_items == 0 && f.depth_is_linear == 0);
  CHECK(memcmp(f.world_to_view, e.w2v, sizeof e.w2v) == 0 && memcmp(f.view_to_clip, e.v2c, sizeof e.v2c) == 0);
  CHECK(memcmp(f.prev_world_to_view, e.w2v, sizeof e.w2v) == 0);  // reset frame: previous = current
  const uint32_t want = (1u << WG_ROLE_COLOUR) | (1u << WG_ROLE_OUTPUT) | (1u << WG_ROLE_DEPTH) | (1u << WG_ROLE_MOTION) |
                        (1u << WG_ROLE_DIFFUSE_ALBEDO) | (1u << WG_ROLE_SPECULAR_ALBEDO) | (1u << WG_ROLE_NORMALS);
  CHECK_EQ(f.attached, want);
  const WgRrImage& c = f.images[WG_ROLE_COLOUR];
  CHECK(c.resource == R(0) && c.buffer_type == WG_BT_COLOUR_IN && c.format == DXGI_FORMAT_R16G16B16A16_FLOAT);
  CHECK(c.format_class == WG_FMT_FLOAT && c.state == kNgxInputState && c.width == 1280 && c.height == 720);
  CHECK(c.extent_width == 1280 && c.extent_height == 720 && c.origin_x == 0 && c.lifetime == WG_LIFE_UNTIL_EVALUATE);
  const WgRrImage& o = f.images[WG_ROLE_OUTPUT];
  CHECK(o.resource == R(1) && o.buffer_type == WG_BT_COLOUR_OUT && o.state == kNgxOutputState);
  CHECK(o.extent_width == 1920 && o.extent_height == 1080);
  CHECK(f.images[WG_ROLE_DEPTH].buffer_type == WG_BT_DEPTH && f.images[WG_ROLE_DEPTH].format_class == WG_FMT_DEPTH_FLOAT);
  CHECK(f.images[WG_ROLE_NORMALS].buffer_type == WG_BT_NORMAL_ROUGHNESS);
  CHECK(e.count("NGX: RR feature 1 first evaluate: render 1280x720 -> output 1920x1080") == 1);
  CHECK(e.ep.reads > 0);
  // second frame: previous jitter / camera from the first, counters advanced, no reset
  float w2v_prev[16];
  memcpy(w2v_prev, e.w2v, sizeof w2v_prev);
  e.w2v[13] = 7.0f;
  e.ep.set_f("Jitter.Offset.X", -0.125f);
  CHECK(e.eval(ctx));
  const WgRrFrame& g = e.rt.last();
  CHECK(g.reset == 0 && g.parity == 1 && g.counter == 1);
  CHECK(g.prev_jitter[0] == 0.25f && g.jitter[0] == -0.125f);
  CHECK(memcmp(g.prev_world_to_view, w2v_prev, sizeof w2v_prev) == 0 && g.world_to_view[13] == 7.0f);
  CHECK(e.count("first evaluate") == 1);  // logged once
  // the game's reset
  e.ep.set_i("Reset", 1);
  CHECK(e.eval(ctx));
  CHECK(e.rt.last().reset == 1 && memcmp(e.rt.last().prev_world_to_view, e.w2v, sizeof e.w2v) == 0);
  e.ep.set_i("Reset", 0);
  // MV scale 0 counts as 1; absent counts as 1
  e.ep.set_f("MV.Scale.X", 0.0f);
  e.ep.f.erase("MV.Scale.Y");
  CHECK(e.eval(ctx));
  CHECK(e.rt.last().mv_scale[0] == 1.0f && e.rt.last().mv_scale[1] == 1.0f);
  CHECK(e.begins == 4 && e.ends == 4);
  // the provider reads only: no parameter was written (the mock has no setter; values unchanged)
  CHECK(e.ep.i["Reset"] == 0 && e.cp.u["Width"] == 1280);
}

static void test_routing() {
  Env e;
  e.cp.set_i("DLSS.Enable.Output.Subrects", 1);
  e.cp.set_i("DLSS.Feature.Create.Flags", (int32_t)(kFlagsStd | 0x80u));  // alpha upscaling
  void* ctx = e.create();
  // subrect bases
  e.ep.set_u("DLSS.Input.Color.Subrect.Base.X", 16);
  e.ep.set_u("DLSS.Input.Color.Subrect.Base.Y", 8);
  e.ep.set_u("DLSS.Output.Subrect.Base.X", 4);
  e.ep.set_u("DLSS.Output.Subrect.Base.Y", 2);
  e.ep.set_u("DLSS.Input.Depth.Subrect.Base.X", 3);
  e.ep.set_u("DLSS.Input.MV.Subrect.Base.Y", 5);
  e.ep.set_u("DLSS.Input.Normals.Subrect.Base.X", 9);
  e.rt.sized[R(0)] = {1296, 728};
  e.rt.sized[R(1)] = {1924, 1082};
  // optional inputs: SSS guide alone, then the fog colour wins; the layer and particles; opacity; alpha
  e.add(10, "DLSSD.ScreenSpaceSubsurfaceScatteringGuide", DXGI_FORMAT_R16G16B16A16_FLOAT, 1280, 720);
  e.add(11, "DLSSD.SpecularHitDistance", DXGI_FORMAT_R16_FLOAT, 1280, 720);
  e.add(12, "GBuffer.SpecularMvec", DXGI_FORMAT_R16G16_FLOAT, 1280, 720);
  e.add(13, "DLSS.DisocclusionMask", DXGI_FORMAT_R8_UNORM, 1280, 720);
  e.add(14, "DLSSD.DepthOfFieldGuide", DXGI_FORMAT_R16G16B16A16_FLOAT, 1280, 720);
  e.add(15, "DLSSD.ColorBeforeTransparency", DXGI_FORMAT_R16G16B16A16_FLOAT, 1280, 720);
  e.add(16, "DLSS.TransparencyLayerOpacity", DXGI_FORMAT_R8_UNORM, 1280, 720);  // no layer yet: not handed on
  e.add(17, "DLSSD.Alpha", DXGI_FORMAT_R8_UNORM, 1280, 720);
  e.add(18, "DLSSD.OutputAlpha", DXGI_FORMAT_R8_UNORM, 1920, 1080);
  e.ep.set_u("DLSSD.SpecularHitDistance.Subrect.Base.X", 6);
  CHECK(e.eval(ctx));
  {
    const WgRrFrame& f = e.rt.last();
    CHECK(f.images[WG_ROLE_COLOUR].origin_x == 16 && f.images[WG_ROLE_COLOUR].origin_y == 8);
    CHECK(f.images[WG_ROLE_OUTPUT].origin_x == 4 && f.images[WG_ROLE_OUTPUT].origin_y == 2);
    CHECK(f.images[WG_ROLE_DEPTH].origin_x == 3 && f.images[WG_ROLE_MOTION].origin_y == 5);
    CHECK(f.images[WG_ROLE_NORMALS].origin_x == 9);
    CHECK(f.images[WG_ROLE_G1].resource == R(10) && f.images[WG_ROLE_G1].buffer_type == WG_BT_SSS_GUIDE);
    CHECK(f.images[WG_ROLE_SPECULAR_HIT_DIST].resource == R(11) && f.images[WG_ROLE_SPECULAR_HIT_DIST].origin_x == 6);
    CHECK(f.images[WG_ROLE_SPECULAR_MOTION].resource == R(12) && f.images[WG_ROLE_DISOCCLUSION].resource == R(13));
    CHECK(f.images[WG_ROLE_DOF_GUIDE].resource == R(14) && f.images[WG_ROLE_BEFORE_TRANSPARENCY].resource == R(15));
    CHECK(!(f.attached & (1u << WG_ROLE_LAYER)) && !(f.attached & (1u << WG_ROLE_OPACITY)));
    CHECK(f.images[WG_ROLE_INPUT_ALPHA].resource == R(17) && f.images[WG_ROLE_OUTPUT_ALPHA].resource == R(18));
    CHECK(f.images[WG_ROLE_OUTPUT_ALPHA].state == kNgxOutputState && f.images[WG_ROLE_OUTPUT_ALPHA].extent_width == 1920);
    CHECK(f.layer_origin_x == 0 && f.layer_origin_y == 0);
  }
  e.add(19, "DLSSD.ColorBeforeFog", DXGI_FORMAT_R16G16B16A16_FLOAT, 1280, 720);
  e.add(20, "DLSS.TransparencyLayer", DXGI_FORMAT_R16G16B16A16_FLOAT, 1280, 720);
  e.ep.set_u("DLSS.TransparencyLayer.Subrect.Base.X", 2);
  e.ep.set_u("DLSS.TransparencyLayer.Subrect.Base.Y", 1);
  e.add(22, "DLSS.TransparencyLayerMvecs", DXGI_FORMAT_R16G16_FLOAT, 1280, 720);
  CHECK(e.eval(ctx));
  {
    const WgRrFrame& f = e.rt.last();
    CHECK(f.images[WG_ROLE_G1].resource == R(19) && f.images[WG_ROLE_G1].buffer_type == WG_BT_COLOUR_BEFORE_FOG);
    CHECK(f.images[WG_ROLE_LAYER].resource == R(20) && f.images[WG_ROLE_LAYER].buffer_type == WG_BT_TRANSPARENCY_LAYER);
    CHECK(f.images[WG_ROLE_OPACITY].resource == R(16));
    CHECK(f.layer_origin_x == 2 && f.layer_origin_y == 1);
  }
  CHECK(e.count("transparency-layer motion vectors; not consumed") == 1);
  e.add(21, "DLSSD.ColorBeforeParticles", DXGI_FORMAT_R16G16B16A16_FLOAT, 1280, 720);
  CHECK(e.eval(ctx));
  {
    const WgRrFrame& f = e.rt.last();
    CHECK(f.images[WG_ROLE_LAYER].resource == R(21) && f.images[WG_ROLE_LAYER].buffer_type == WG_BT_COLOUR_BEFORE_PARTICLES);
    CHECK(f.layer_origin_x == 2 && f.layer_origin_y == 1);  // the layer origin stays the transparency layer's
  }
  CHECK(e.count("transparency-layer motion vectors; not consumed") == 1);  // once
  // without subrects enabled the output origin is (0, 0) whatever the block holds
  Env e2;
  void* c2 = e2.create();
  e2.ep.set_u("DLSS.Output.Subrect.Base.X", 4);
  CHECK(e2.eval(c2) && e2.rt.last().images[WG_ROLE_OUTPUT].origin_x == 0);
  // separate roughness and linear depth
  Env e3;
  e3.cp.set_i("DLSS.Roughness.Mode", 0);
  e3.cp.set_i("DLSS.Use.HW.Depth", 0);
  void* c3 = e3.create();
  CHECK(!e3.eval(c3));  // roughness missing
  CHECK(e3.count("evaluate without roughness refused") == 1);
  e3.add(7, "GBuffer.Roughness", DXGI_FORMAT_R8_UNORM, 1280, 720);
  e3.ep.set_u("DLSS.Input.Roughness.Subrect.Base.Y", 11);
  e3.rt.described[R(2)] = {DXGI_FORMAT_R32_FLOAT, 1};
  CHECK(e3.eval(c3));
  const WgRrFrame& f3 = e3.rt.last();
  CHECK(f3.images[WG_ROLE_NORMALS].buffer_type == WG_BT_NORMALS && f3.images[WG_ROLE_ROUGHNESS].resource == R(7));
  CHECK(f3.images[WG_ROLE_ROUGHNESS].origin_y == 11);
  CHECK(f3.images[WG_ROLE_DEPTH].buffer_type == WG_BT_LINEAR_DEPTH && f3.depth_is_linear == 1);
}

static void test_dynamic_resolution() {
  Env e;
  e.cp.set_u("Width", 1920);  // DLAA-sized maximum
  e.cp.set_u("Height", 1080);
  e.rt.sized[R(0)] = {1920, 1080};
  void* ctx = e.create();
  e.ep.set_u("DLSS.Render.Subrect.Dimensions.Width", 1280);
  e.ep.set_u("DLSS.Render.Subrect.Dimensions.Height", 720);
  CHECK(e.eval(ctx));
  CHECK(e.rt.last().render_width == 1280 && e.rt.last().rebuild_render_items == 1);
  CHECK(e.rt.last().images[WG_ROLE_COLOUR].extent_width == 1280);
  CHECK(e.eval(ctx) && e.rt.last().rebuild_render_items == 0);
  CHECK(e.rt.last().reset == 1);  // the render-size change made a reset pending for the next frame (53.9)
  e.ep.set_u("DLSS.Render.Subrect.Dimensions.Width", 960);
  e.ep.set_u("DLSS.Render.Subrect.Dimensions.Height", 540);
  CHECK(e.eval(ctx) && e.rt.last().render_width == 960 && e.rt.last().rebuild_render_items == 1);
  // outside the accepted window: refused, reset pending, logged once, nothing recorded
  const int ev = e.rt.evaluates, b = e.begins;
  e.ep.set_u("DLSS.Render.Subrect.Dimensions.Width", 400);
  CHECK(!e.eval(ctx));
  CHECK(!e.eval(ctx));
  CHECK(e.rt.evaluates == ev && e.begins == b);
  CHECK(e.count("render size 400x540 outside 1/3..1 of output 1920x1080 refused") == 1);
  CHECK(e.p->lifecycle(ctx)->pending_reset());
  // absent subrect dimensions: the create-time render size
  e.ep.u.erase("DLSS.Render.Subrect.Dimensions.Width");
  e.ep.u.erase("DLSS.Render.Subrect.Dimensions.Height");
  CHECK(e.eval(ctx) && e.rt.last().render_width == 1920 && e.rt.last().render_height == 1080);
}

static void test_evaluate_rejections() {
  {  // missing required inputs
    Env e;
    void* ctx = e.create();
    e.ep.res.erase("MotionVectors");
    e.ep.res.erase("DLSS.Input.SpecularAlbedo");
    CHECK(!e.eval(ctx));
    CHECK(!e.eval(ctx));
    CHECK(e.rt.evaluates == 0 && e.begins == 0);
    CHECK(e.count("evaluate without motion, specular albedo refused") == 1);
  }
  {  // unsupported / multisampled formats
    Env e;
    void* ctx = e.create();
    e.rt.described[R(4)] = {DXGI_FORMAT_R8G8B8A8_UINT, 1};
    CHECK(!e.eval(ctx));
    CHECK(e.count("unsupported format 30 for the diffuse albedo input refused") == 1);
    e.rt.described[R(4)] = {DXGI_FORMAT_R8G8B8A8_UNORM, 4};
    CHECK(!e.eval(ctx));
    e.rt.described.erase(R(4));  // the runtime cannot describe it
    CHECK(!e.eval(ctx));
    CHECK(e.rt.evaluates == 0);
  }
  {  // the colour rectangle does not fit its resource
    Env e;
    void* ctx = e.create();
    e.ep.set_u("DLSS.Input.Color.Subrect.Base.X", 8);
    CHECK(!e.eval(ctx));
    CHECK(e.count("the colour rectangle 8,0 1280x720 does not fit its 1280x720 resource") == 1);
  }
  {  // no command list / no parameters / unknown context
    Env e;
    void* ctx = e.create();
    WgHostNgxParams v = e.ep.view();
    CHECK(!e.p->evaluate(ctx, nullptr, &v));
    CHECK(!e.p->evaluate(ctx, e.cl, nullptr));
    CHECK(!e.p->evaluate(R(300), e.cl, &v));
    CHECK(!e.p->evaluate(nullptr, e.cl, &v));
    CHECK(e.count("evaluate for an unknown or released RR feature refused") == 2);
    CHECK(e.rt.evaluates == 0);
  }
  {  // matrices absent: zeros, logged once
    Env e;
    void* ctx = e.create();
    e.ep.ptr.clear();
    CHECK(e.eval(ctx) && e.eval(ctx));
    CHECK(e.count("camera matrices not supplied (world-to-view view-to-clip missing") == 1);
    CHECK(e.rt.last().world_to_view[0] == 0.0f);
  }
}

static void test_failsafe() {
  Env e;
  void* ctx = e.create();
  CHECK(e.eval(ctx));
  // the runtime fails before any pass: the colour is passed through and success reported, reset next time
  e.rt.evaluate_rc = 22;
  e.rt.passthrough_rc = 0;
  CHECK(e.eval(ctx));
  CHECK_EQ(e.rt.passthroughs, 1);
  CHECK(e.begins == 2 && e.ends == 2);  // the bracket also closes on the failure path
  CHECK(e.count("runtime returned 22; colour passed through, success reported") == 1);
  CHECK_EQ(e.frames_done, 1);  // a passed-through frame does not count as an RR frame
  const Lifecycle* lc = e.p->lifecycle(ctx);
  CHECK(lc && lc->parity() == 1 && lc->counter() == 1 && lc->pending_reset());  // counters unchanged by the failure
  // no pass-through either: the failure goes to the game (the host reports an NGX failure code)
  e.rt.passthrough_rc = 31;
  CHECK(!e.eval(ctx));
  CHECK(e.count("no pass-through (31), failure reported to the game") == 1);
  e.rt.evaluate_rc = 0;
  CHECK(e.eval(ctx));
  CHECK(e.rt.last().reset == 1);
  CHECK(e.count("RR feature 1 evaluate 4") == 1);  // the recovery is logged
}

static void test_release_and_handles() {
  Env e;
  void* a = e.create();
  void* b = e.create();
  CHECK(a && b && a != b);
  CHECK(e.eval(a) && e.eval(b) && e.eval(a));
  CHECK(e.p->lifecycle(a)->counter() == 2 && e.p->lifecycle(b)->counter() == 1);  // independent bookkeeping
  CHECK(e.rt.frames[1].reset == 1 && e.rt.frames[2].reset == 0);
  e.p->release(a);
  CHECK_EQ(e.rt.destroys, 1);
  CHECK_EQ(e.p->live(), (size_t)1);
  CHECK(e.count("NGX: RR feature 1 released after 2 evaluate(s)") == 1);
  CHECK(!e.eval(a));  // stale context: refused, never dereferenced
  e.p->release(a);    // twice: ignored
  CHECK_EQ(e.rt.destroys, 1);
  CHECK(e.count("release of an unknown RR feature ignored") == 1);
  CHECK(e.eval(b));
  void* c = e.create();
  CHECK(c && e.count("NGX: RR feature 3 created") == 1);  // ids are never reused
  e.p->release(b);
  e.p->release(c);
  CHECK(e.rt.live.empty() && e.p->live() == 0);
  // a provider destroyed with live features releases them
  {
    Env f;
    f.create();
    f.create();
    delete f.p;
    f.p = nullptr;
    CHECK(f.rt.destroys == 2 && f.rt.live.empty());
  }
  // the host table forwards to the provider
  {
    Env g;
    WgHostNgxRr t = g.p->table();
    CHECK(t.size == sizeof(WgHostNgxRr) && t.user == g.p && t.create && t.evaluate && t.release);
    WgHostNgxParams cv = g.cp.view(), ev = g.ep.view();
    void* ctx = nullptr;
    CHECK(t.create(t.user, g.dev, g.cl, &cv, &ctx) && ctx);
    CHECK(t.evaluate(t.user, ctx, g.cl, &ev));
    t.release(t.user, ctx);
    CHECK(g.rt.creates == 1 && g.rt.evaluates == 1 && g.rt.destroys == 1);
  }
}

static void test_no_bracket() {
  // a host without the command-list restore: no bracket set, the work still runs
  FakeRuntime rt;
  Settings s;
  NgxRrDeps d;
  d.runtime = [&rt]() -> const WgRrRuntimeV1* { return &rt.table; };
  d.settings = [&s] { return &s; };
  NgxRrProvider p(std::move(d));
  Env e;  // borrow its parameter blocks and resources
  for (auto& kv : e.rt.described) rt.described[kv.first] = kv.second;
  for (auto& kv : e.rt.sized) rt.sized[kv.first] = kv.second;
  WgHostNgxParams cv = e.cp.view(), ev = e.ep.view();
  void* ctx = nullptr;
  CHECK(p.create(e.dev, e.cl, &cv, &ctx) && ctx);
  CHECK(p.evaluate(ctx, e.cl, &ev));
  CHECK(rt.evaluates == 1 && e.begins == 0);
}

static void test_bracket_after_fault() {
  // the fault guard closes the provider's open bracket once; the bracket's own end then does not run a second time
  Env e;
  void* ctx = e.create();
  int ends_inside = -1;
  e.rt.on_evaluate = [&] {
    CHECK(e.begins == 1 && e.ends == 0);  // the runtime's work runs inside the bracket
    NgxRrProvider::close_open_bracket_after_fault();
    ends_inside = e.ends;
  };
  CHECK(e.eval(ctx));
  CHECK_EQ(ends_inside, 1);
  CHECK(e.begins == 1 && e.ends == 1);  // closed by the guard, not again at the end of the call
  e.rt.on_evaluate = nullptr;
  NgxRrProvider::close_open_bracket_after_fault();  // nothing open: no effect
  CHECK(e.ends == 1);
  CHECK(e.eval(ctx));
  CHECK(e.begins == 2 && e.ends == 2);  // the next call brackets normally
  // the live count stays readable without the provider's lock (the plugin's route guard)
  CHECK_EQ(e.p->live(), (size_t)1);
  e.p->release(ctx);
  CHECK_EQ(e.p->live(), (size_t)0);
}

static void test_integer_key_types() {
  // a host that does not convert between int and uint keys: the provider asks the other integer getter itself
  Env e;
  e.cp.strict = e.ep.strict = true;
  e.cp.u.erase("Width");
  e.cp.u.erase("OutHeight");
  e.cp.set_i("Width", 1280);  // sizes set as int, flags and quality as uint
  e.cp.set_i("OutHeight", 1080);
  e.cp.i.erase("PerfQualityValue");
  e.cp.i.erase("DLSS.Feature.Create.Flags");
  e.cp.set_u("PerfQualityValue", 1);
  e.cp.set_u("DLSS.Feature.Create.Flags", kFlagsStd);
  void* ctx = e.create();
  CHECK(ctx != nullptr);
  if (ctx) {
    const WgRrCreate* d = e.p->create_desc(ctx);
    CHECK(d && d->render_width == 1280 && d->output_height == 1080 && d->mode == 2 && d->colour_hdr == 1 &&
          d->depth_inverted == 1);
    e.ep.i.erase("Reset");
    e.ep.set_u("Reset", 1);  // reset as uint
    CHECK(e.eval(ctx));
    CHECK(e.eval(ctx));
    CHECK_EQ(e.rt.last().reset, 1u);
    e.p->release(ctx);
  }
  // a negative int for an unsigned key counts as not set: the render size is missing and the feature refused
  Env f;
  f.cp.strict = true;
  f.cp.u.erase("Width");
  f.cp.set_i("Width", -1280);
  f.create(false);
  CHECK(f.count("render or output size missing (0x720") == 1);
}

// ---------------------------------------------------------------- the same captured frames through both routes
static WgGuid guid(uint8_t k) { WgGuid g; memset(g.b, 0x5a, 16); g.b[15] = k; return g; }
static WgResultCode o_ok0() { return 0; }
static WgResultCode o_init(const WgPreferences*, uint64_t) { return 0; }
static WgResultCode o_set_device(void*) { return 0; }
static WgResultCode o_tag(const WgViewport*, const WgTag*, uint32_t, void*) { return 0; }
static WgResultCode o_constants(const WgConstants*, const WgFrameToken*, const WgViewport*) { return 0; }
struct Tok { WgFrameToken t; uint32_t index; };
static uint32_t tok_index(const WgFrameToken* t) { return reinterpret_cast<const Tok*>(t)->index; }
static const WgFrameTokenIndexFn kTokVtbl[1] = {&tok_index};

static const char* ngx_key_for_type(uint32_t type) {
  switch (type) {
    case WG_BT_DEPTH: case WG_BT_LINEAR_DEPTH: return "Depth";
    case WG_BT_MOTION: return "MotionVectors";
    case WG_BT_COLOUR_IN: return "Color";
    case WG_BT_COLOUR_OUT: return "Output";
    case WG_BT_DIFFUSE_ALBEDO: return "DLSS.Input.DiffuseAlbedo";
    case WG_BT_SPECULAR_ALBEDO: return "DLSS.Input.SpecularAlbedo";
    case WG_BT_NORMALS: case WG_BT_NORMAL_ROUGHNESS: return "GBuffer.Normals";
    case WG_BT_ROUGHNESS: return "GBuffer.Roughness";
    case WG_BT_SPECULAR_MOTION: return "GBuffer.SpecularMvec";
    case WG_BT_SPECULAR_HIT_DISTANCE: return "DLSSD.SpecularHitDistance";
    case WG_BT_DISOCCLUSION: return "DLSS.DisocclusionMask";
    case WG_BT_COLOUR_BEFORE_FOG: return "DLSSD.ColorBeforeFog";
    case WG_BT_SSS_GUIDE: return "DLSSD.ScreenSpaceSubsurfaceScatteringGuide";
    case WG_BT_DOF_GUIDE: return "DLSSD.DepthOfFieldGuide";
    case WG_BT_COLOUR_BEFORE_TRANSPARENCY: return "DLSSD.ColorBeforeTransparency";
    case WG_BT_TRANSPARENCY_LAYER: return "DLSS.TransparencyLayer";
    case WG_BT_COLOUR_BEFORE_PARTICLES: return "DLSSD.ColorBeforeParticles";
    case WG_BT_TRANSPARENCY_OPACITY: return "DLSS.TransparencyLayerOpacity";
    default: return nullptr;  // not an RR input of preset F (e.g. type 29): not handed to the NGX block either
  }
}

static bool same_image(const WgRrImage& a, const WgRrImage& b) {
  return a.resource == b.resource && a.buffer_type == b.buffer_type && a.format == b.format &&
         a.format_class == b.format_class && a.width == b.width && a.height == b.height && a.origin_x == b.origin_x &&
         a.origin_y == b.origin_y && a.extent_width == b.extent_width && a.extent_height == b.extent_height &&
         a.conversion == b.conversion;
}

// Every field the runtime reads, except the route marker (api_minor), the declared states (66.4: the direct path
// declares shader-resource / unordered-access states; the game-interface path hands the game's tags' states) and the
// lifetime (informational in the runtime).
static int compare_frames(const WgRrFrame& s, const WgRrFrame& n, int k) {
  int bad = 0;
  auto chk = [&](bool ok, const char* what) {
    if (!ok) {
      ++bad;
      fprintf(stderr, "frame %d: %s differs between the routes\n", k, what);
    }
  };
  chk(s.size == n.size, "size");
  chk(s.command_list == n.command_list, "command list");
  chk(s.render_width == n.render_width && s.render_height == n.render_height, "render size");
  chk(s.output_width == n.output_width && s.output_height == n.output_height, "output size");
  chk(memcmp(s.jitter, n.jitter, sizeof s.jitter) == 0, "jitter");
  chk(memcmp(s.prev_jitter, n.prev_jitter, sizeof s.prev_jitter) == 0, "previous jitter");
  chk(memcmp(s.mv_scale, n.mv_scale, sizeof s.mv_scale) == 0, "vector scale");
  chk(s.reset == n.reset && s.parity == n.parity && s.counter == n.counter, "reset / parity / counter");
  chk(s.rebuild_render_items == n.rebuild_render_items, "rebuild");
  chk(s.depth_is_linear == n.depth_is_linear, "depth is linear");
  chk(s.settings == n.settings, "settings");
  chk(memcmp(s.view_to_clip, n.view_to_clip, sizeof s.view_to_clip) == 0, "view-to-clip");
  chk(memcmp(s.world_to_view, n.world_to_view, sizeof s.world_to_view) == 0, "world-to-view");
  chk(memcmp(s.prev_view_to_clip, n.prev_view_to_clip, sizeof s.prev_view_to_clip) == 0, "previous view-to-clip");
  chk(memcmp(s.prev_world_to_view, n.prev_world_to_view, sizeof s.prev_world_to_view) == 0, "previous world-to-view");
  chk(s.attached == n.attached, "attached roles");
  chk(s.layer_origin_x == n.layer_origin_x && s.layer_origin_y == n.layer_origin_y, "layer origin");
  for (int r = 0; r < WG_ROLE_COUNT; ++r)
    if (s.attached & (1u << r)) chk(same_image(s.images[r], n.images[r]), "an image");
  return bad;
}

static void test_capture_equivalence(const std::string& path) {
  mj::Value doc;
  const bool loaded = mj::load(path.c_str(), &doc);
  CHECK(loaded);
  if (!loaded) return;
  const mj::Value& frames = doc["frames"];
  CHECK(frames.size() >= 3);

  // the game-interface route: the release core with a fake runtime (no native-interface unwrapping)
  FakeRuntime srt, nrt;
  Settings settings;
  settings.set_save_enabled(false);
  ProxyConfig cfg;
  cfg.api_minor = 14;
  cfg.ids.viewport = guid(1);
  cfg.ids.tag = guid(2);
  cfg.ids.tag_conversion = guid(3);
  OriginalApi o;
  o.init = o_init;
  o.shutdown = o_ok0;
  o.set_device = o_set_device;
  o.tag = o_tag;
  o.constants = o_constants;
  RrProxy sl(cfg, o, &srt.table, &settings, nullptr, [] { return (uint64_t)1; });
  NgxRrDeps d;
  d.runtime = [&nrt]() -> const WgRrRuntimeV1* { return &nrt.table; };
  d.settings = [&settings] { return &settings; };
  NgxRrProvider nprov(std::move(d));
  void* dev = R(400);
  void* cl = R(401);
  CHECK_EQ(sl.init(nullptr, 0), 0);
  CHECK_EQ(sl.set_device(dev), 0);
  WgViewport vp;
  memset(&vp, 0, sizeof vp);
  vp.h.type = cfg.ids.viewport;
  vp.h.version = 1;
  vp.id = 0;
  void* nctx = nullptr;
  static Tok toks[16];
  int compared = 0, bad = 0;
  for (size_t k = 0; k < frames.size(); ++k) {
    const mj::Value& fr = frames[k];
    // resources: one fake per buffer type, described alike to both runtimes
    std::vector<WgResource> res;
    std::vector<WgTag> tags;
    res.reserve(32);
    MockNgxParams ep, cp;
    auto add_res = [&](uint32_t type, uint32_t fmt, uint32_t w, uint32_t h, uint32_t state, uint32_t life,
                       const mj::Value& ext) {
      void* r = R(100 + (int)type);
      srt.described[r] = nrt.described[r] = {fmt, 1};
      srt.sized[r] = nrt.sized[r] = {w, h};
      WgResource x;
      memset(&x, 0, sizeof x);
      x.native = r;
      x.width = w;
      x.height = h;
      x.native_format = fmt;
      x.state = state;
      res.push_back(x);
      WgTag t;
      memset(&t, 0, sizeof t);
      t.h.type = cfg.ids.tag;
      t.h.version = 1;
      t.buffer_type = type;
      t.lifetime = life;
      t.extent.left = (uint32_t)ext[0].n;
      t.extent.top = (uint32_t)ext[1].n;
      t.extent.width = (uint32_t)ext[2].n;
      t.extent.height = (uint32_t)ext[3].n;
      tags.push_back(t);
      if (const char* key = ngx_key_for_type(type)) ep.set_res(key, r);
    };
    for (size_t i = 0; i < fr["inputs"].size(); ++i) {
      const mj::Value& in = fr["inputs"][i];
      add_res((uint32_t)in["type"].n, (uint32_t)in["format"].n, (uint32_t)in["width"].n, (uint32_t)in["height"].n,
              (uint32_t)in["state"].n, (uint32_t)in["lifetime"].n, in["extent"]);
    }
    const mj::Value& out = fr["output"];
    add_res(WG_BT_COLOUR_OUT, (uint32_t)out["format"].n, (uint32_t)out["width"].n, (uint32_t)out["height"].n,
            (uint32_t)out["state"].n, 1, out["rect"]);
    for (size_t i = 0; i < tags.size(); ++i) tags[i].resource = &res[i];

    // game-interface route: options (world-to-view in the options), tags, constants, evaluate
    WgRrOptions op;
    memset(&op, 0, sizeof op);
    op.h.version = 3;
    op.mode = (uint32_t)fr["mode"].n;
    op.output_width = (uint32_t)fr["output_width"].n;
    op.output_height = (uint32_t)fr["output_height"].n;
    op.colour_is_hdr = (uint32_t)fr["colour_is_hdr"].n;
    op.normal_roughness_packing = (uint32_t)fr["packing"].n;
    op.alpha_upscaling = (uint32_t)fr["alpha_upscaling"].n;
    op.sharpness = (float)fr["sharpness"].n;
    op.pre_exposure = 1;
    op.exposure_scale = 1;
    for (int i = 0; i < 6; ++i) op.presets[i] = (uint32_t)fr["presets"][i].n;
    for (int i = 0; i < 16; ++i) op.world_to_view[i] = (float)fr["world_to_view"][i].n;
    CHECK_EQ(sl.rr_options(&vp, &op), 0);
    CHECK_EQ(sl.tag(&vp, tags.data(), (uint32_t)tags.size(), cl), 0);
    Tok& tk = toks[k % 16];
    tk.t.vtable = kTokVtbl;
    tk.index = 100 + (uint32_t)k;
    WgConstants c;
    memset(&c, 0, sizeof c);
    c.h.version = 2;
    for (int i = 0; i < 16; ++i) c.view_to_clip[i] = (float)fr["view_to_clip"][i].n;
    c.jitter[0] = (float)fr["jitter"][0].n;
    c.jitter[1] = (float)fr["jitter"][1].n;
    c.mv_scale[0] = (float)fr["mv_scale"][0].n;
    c.mv_scale[1] = (float)fr["mv_scale"][1].n;
    c.depth_inverted = (uint32_t)fr["depth_inverted"].n;
    c.mv_jittered = (uint32_t)fr["mv_jittered"].n;
    c.camera_motion_included = (uint32_t)fr["camera_motion_included"].n;
    c.reset = (uint32_t)fr["reset"].n;
    CHECK_EQ(sl.constants(&c, &tk.t, &vp), 0);
    const WgHeader* hin[1] = {&vp.h};
    CHECK_EQ(sl.evaluate(WG_FEAT_RAY_RECONSTRUCTION, &tk.t, hin, 1, cl), 0);

    // NGX route: the same frame as the direct path hands it (66.2 / 66.3)
    const uint32_t rw = (uint32_t)fr["render"][2].n, rh = (uint32_t)fr["render"][3].n;
    if (!nctx) {
      cp.set_u("Width", rw);
      cp.set_u("Height", rh);
      cp.set_u("OutWidth", op.output_width);
      cp.set_u("OutHeight", op.output_height);
      cp.set_i("PerfQualityValue", (int32_t)op.mode - 1);
      cp.set_i("DLSS.Feature.Create.Flags", (int32_t)((op.colour_is_hdr == 1 ? 0x1u : 0u) | 0x2u |
                                                        (c.mv_jittered == 1 ? 0x4u : 0u) |
                                                        (c.depth_inverted == 1 ? 0x8u : 0u) |
                                                        (op.sharpness > 0 ? 0x20u : 0u)));
      cp.set_i("DLSS.Denoise.Mode", 1);
      cp.set_i("DLSS.Roughness.Mode", (int32_t)op.normal_roughness_packing);
      cp.set_i("DLSS.Use.HW.Depth", 1);  // the capture tags type 0 (hardware depth)
      WgHostNgxParams cv = cp.view();
      CHECK(nprov.create(dev, cl, &cv, &nctx) && nctx);
    }
    float w2v[16], v2c[16];
    memcpy(w2v, op.world_to_view, sizeof w2v);
    memcpy(v2c, c.view_to_clip, sizeof v2c);
    ep.set_ptr("WorldToViewMatrix", w2v);
    ep.set_ptr("ViewToClipMatrix", v2c);
    ep.set_f("Jitter.Offset.X", c.jitter[0]);
    ep.set_f("Jitter.Offset.Y", c.jitter[1]);
    volatile float msx = c.mv_scale[0] * (float)rw;  // the game-interface scale in render pixels, as the direct path
    volatile float msy = c.mv_scale[1] * (float)rh;  // carries it (66.3: taken as given)
    ep.set_f("MV.Scale.X", msx);
    ep.set_f("MV.Scale.Y", msy);
    ep.set_i("Reset", c.reset == 1 ? 1 : 0);
    WgHostNgxParams ev = ep.view();
    CHECK(nprov.evaluate(nctx, cl, &ev));

    CHECK_EQ(srt.evaluates, (int)k + 1);
    CHECK_EQ(nrt.evaluates, (int)k + 1);
    if (srt.evaluates == (int)k + 1 && nrt.evaluates == (int)k + 1) {
      bad += compare_frames(srt.last(), nrt.last(), (int)k);
      ++compared;
      // the capture's optional inputs reach the runtime on both routes (hit distance, layer, colour guide)
      const uint32_t opt = (1u << WG_ROLE_SPECULAR_HIT_DIST) | (1u << WG_ROLE_LAYER) | (1u << WG_ROLE_G1);
      CHECK((nrt.last().attached & opt) == opt);
      CHECK_EQ(nrt.last().reset, k == 0 ? 1u : 0u);  // a fresh feature: the first frame resets, then history runs
      CHECK_EQ(nrt.last().counter, (uint32_t)k);
      if (k > 0) CHECK(nrt.last().prev_jitter[0] == (float)frames[k - 1]["jitter"][0].n);
    }
  }
  CHECK(srt.creates == 1 && nrt.creates == 1);
  if (srt.creates == 1 && nrt.creates == 1) {
    WgRrCreate a = srt.created[0], b = nrt.created[0];
    CHECK_EQ(a.api_minor, 14u);
    CHECK_EQ(b.api_minor, kNgxApiMinor);
    b.api_minor = a.api_minor;
    CHECK(memcmp(&a, &b, sizeof a) == 0);  // the same creation apart from the route marker
  }
  CHECK_EQ(bad, 0);
  CHECK_EQ(compared, (int)frames.size());
  printf("capture equivalence: %d frame(s) of %s compared, %d difference(s)\n", compared, path.c_str(), bad);
  nprov.release(nctx);
  sl.shutdown();
}

int main(int argc, char** argv) {
  std::string data;
  if (argc > 1) {
    data = argv[1];
  } else {
    data = __FILE__;
    const size_t slash = data.find_last_of("\\/");
    data = (slash == std::string::npos ? std::string() : data.substr(0, slash + 1)) + "data\\cp2077-2.14-rr-frames.json";
  }
  test_create();
  test_create_refusals();
  test_evaluate();
  test_routing();
  test_dynamic_resolution();
  test_evaluate_rejections();
  test_failsafe();
  test_release_and_handles();
  test_no_bracket();
  test_bracket_after_fault();
  test_integer_key_types();
  test_capture_equivalence(data);
  TEST_SUMMARY("test_ngx_rr");
}
