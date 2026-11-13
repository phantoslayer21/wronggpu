// SPDX-License-Identifier: MIT
// D3D12 test host of WrongGPU's RR runtime: plays the proxy's role (clean/rr_f/proxy). It loads
// wronggpu_rr_runtime.dll from a stage folder through the owned adapter (wg_rr_adapter.h), keeps the page-53 host
// bookkeeping with the proxy's own Lifecycle (proxy/src/lifecycle.h), puts synthetic inputs into real D3D12 textures,
// records every evaluate into its own direct command list (as a game would), executes it, and writes after each frame
// the runtime's arena (test hook), the output texture(s) and one line of frame metadata. Original WrongGPU code.
//
//   rr_test_host <job file>
//
// Job file, one command per line:
//   runtime <dll>                    load the runtime (its folder holds the model and receives wronggpu-rr.log)
//   outdir <dir>
//   env <NAME> <value|->             set (or with "-" clear) an environment variable (read by the runtime at create)
//   output <dxgi format> <uav 0/1>   output texture of the next instances
//   create <Wo> <Ho> <rw> <rh> <packing> <hdr> <depth_inverted> <mv_jittered> <hw_depth> <alpha>
//   frame <input file> [nodump] [reset]  one evaluate (input file: tests/e2e.py write_frame); reset: this frame is a
//                                    game reset (the host's lifecycle makes it a reset frame: i15 switch references)
//   (environment WG_TEST_BLOCK=<option>=<value>[,...]: the same before the first job line)
//   block <option> <value>           i15: the shared settings block of this process (WrongGPU.settings.v1.<pid>),
//                                    created by the host when absent (seed: nothing, creator / writer WG_SS_WRITER_TEST),
//                                    the host claims the writer and writes registry option <option> (key, e.g. rr_mode)
//                                    = <value> (a value name or number) with the seqlock: what OptiScaler's menu does
//                                    through the SR front end in game
//   fail <input file> <role>         an evaluate whose required input <role> is detached (must fail before any pass)
//   transient <input file> <role>    <role> handed over as a lifetime-0 copy (retain_transient during the tag)
//   border <input file>              bug hunt G: one frame into an output without unordered access (the runtime's
//                                    texture, then a copy): no pixel outside the store's rectangle may change (exit 11)
//   arrayin <input file> <role>      bug hunt E: <role> as a 2-slice array texture: evaluate must refuse it before any
//                                    pass, the device not removed (exit 10 otherwise)
//   retag <input file> <role>        bug hunt B: <role> copied as a lifetime-0 input and released again before any
//                                    evaluate (the game re-tags it): the copy must stay until the list that recorded it
//                                    has run (a deferred release), and be gone after it (exit 9 otherwise)
//   destroy
//   leak <cycles> <input file>       create / 2 frames / destroy per cycle; live counters and video memory
//   perfcap <passes> <file> ...       game-setup timing: the frames resident, evaluated <passes> times without waits
//   perf <Wo> <Ho> <rw> <rh> <frames> <exact|fast> [plain|ext]   GPU timing run on random inputs (the runtime logs per
//                                    300 frames); ext: alpha upscaling, an RGBA16F layer, a separate output alpha
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "../../proxy/include/rr_proxy_abi.h"
#include "../../proxy/include/wg_rr_adapter.h"
#include "../../proxy/src/lifecycle.h"
#include "rr_runtime_test.h"
#include "../../proxy/include/wg_settings_reader.h"

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")

#define CK(x) ck((x), #x, __LINE__)
static void ck(HRESULT h, const char* s, int line) {
  if (FAILED(h)) {
    fprintf(stderr, "D3D12 %s (line %d): %08x\n", s, line, (unsigned)h);
    exit(2);
  }
}

static ID3D12Device* g_dev;
static ID3D12CommandQueue* g_q;
static ID3D12CommandAllocator* g_al;
static ID3D12GraphicsCommandList* g_l;
static ID3D12Fence* g_fence;
static HANDLE g_ev;
static uint64_t g_fv;
static IDXGIAdapter3* g_adapter;
static const WgRrRuntimeV1* g_rt;
static const WgRrRuntimeTestV1* g_test;

static void submit_wait() {
  CK(g_l->Close());
  ID3D12CommandList* ll = g_l;
  g_q->ExecuteCommandLists(1, &ll);
  CK(g_q->Signal(g_fence, ++g_fv));
  CK(g_fence->SetEventOnCompletion(g_fv, g_ev));
  if (WaitForSingleObject(g_ev, 120000) != WAIT_OBJECT_0) {
    fprintf(stderr, "GPU timeout\n");
    exit(4);
  }
  CK(g_dev->GetDeviceRemovedReason());
  CK(g_al->Reset());
  CK(g_l->Reset(g_al, nullptr));
}

static ID3D12Resource* buffer(uint64_t n, D3D12_HEAP_TYPE t, D3D12_RESOURCE_STATES s) {
  D3D12_HEAP_PROPERTIES hp = {};
  hp.Type = t;
  D3D12_RESOURCE_DESC rd = {};
  rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  rd.Width = n;
  rd.Height = rd.DepthOrArraySize = rd.MipLevels = 1;
  rd.SampleDesc.Count = 1;
  rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  ID3D12Resource* r = nullptr;
  CK(g_dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, s, nullptr, IID_PPV_ARGS(&r)));
  return r;
}

static uint32_t bpp(DXGI_FORMAT f) {
  switch (f) {
    case DXGI_FORMAT_R32G32B32A32_FLOAT: return 16;
    case DXGI_FORMAT_R16G16B16A16_FLOAT: return 8;
    case DXGI_FORMAT_R32G32_FLOAT: return 8;
    case DXGI_FORMAT_R32_FLOAT: case DXGI_FORMAT_D32_FLOAT: case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: case DXGI_FORMAT_R10G10B10A2_UNORM: case DXGI_FORMAT_R11G11B10_FLOAT:
    case DXGI_FORMAT_R16G16_FLOAT: case DXGI_FORMAT_B8G8R8A8_UNORM: return 4;
    case DXGI_FORMAT_R16_FLOAT: case DXGI_FORMAT_R8G8_UNORM: return 2;
    case DXGI_FORMAT_R8_UNORM: return 1;
    default: return 0;
  }
}

struct Tex {
  ID3D12Resource* r = nullptr;
  DXGI_FORMAT f = DXGI_FORMAT_UNKNOWN;
  uint32_t w = 0, h = 0;
  D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
  bool uav = false;
};

// WG_RR_TEST_MIPS=N (0.6.0 follow-up, texture_shape.h): the game's input textures with N mips (as many as their size
// allows), the data in mip 0. Only mip 0 is read, so the results must be the same bytes as with one mip.
static uint16_t input_mips(uint32_t w, uint32_t h) {
  char v[16] = {};  // the job's "env" lines set it with SetEnvironmentVariableA (getenv would not see them)
  int n = GetEnvironmentVariableA("WG_RR_TEST_MIPS", v, sizeof v) ? atoi(v) : 1, most = 1;
  for (uint32_t m = w > h ? w : h; m > 1; m >>= 1) ++most;
  return (uint16_t)(n < 1 ? 1 : n > most ? most : n);
}

static uint16_t g_array_slices = 1;   // bug hunt E (job arrayin): the next textures made are arrays of this many slices
static Tex make_tex(uint32_t w, uint32_t h, DXGI_FORMAT f, bool uav, D3D12_RESOURCE_STATES s, uint16_t mips = 1) {
  Tex t;
  D3D12_HEAP_PROPERTIES hp = {};
  hp.Type = D3D12_HEAP_TYPE_DEFAULT;
  D3D12_RESOURCE_DESC rd = {};
  rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
  rd.Width = w;
  rd.Height = h;
  rd.DepthOrArraySize = g_array_slices;
  rd.MipLevels = mips;
  rd.Format = f;
  rd.SampleDesc.Count = 1;
  rd.Flags = uav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
  CK(g_dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, s, nullptr, IID_PPV_ARGS(&t.r)));
  t.f = f;
  t.w = w;
  t.h = h;
  t.state = s;
  t.uav = uav;
  return t;
}

static void barrier(ID3D12Resource* r, D3D12_RESOURCE_STATES a, D3D12_RESOURCE_STATES b) {
  if (a == b) return;
  D3D12_RESOURCE_BARRIER x = {};
  x.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  x.Transition.pResource = r;
  x.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
  x.Transition.StateBefore = a;
  x.Transition.StateAfter = b;
  g_l->ResourceBarrier(1, &x);
}

// upload rows of `data` (tight, w * bpp bytes per row) into the texture on g_l; the upload buffer is kept until the
// caller's submission finished (returned for release)
static ID3D12Resource* upload(Tex& t, const uint8_t* data) {
  D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp;
  UINT64 total = 0;
  const D3D12_RESOURCE_DESC rd = t.r->GetDesc();
  g_dev->GetCopyableFootprints(&rd, 0, 1, 0, &fp, nullptr, nullptr, &total);
  ID3D12Resource* up = buffer(total, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
  uint8_t* m = nullptr;
  D3D12_RANGE none{0, 0};
  CK(up->Map(0, &none, (void**)&m));
  const uint32_t row = t.w * bpp(t.f);
  for (uint32_t y = 0; y < t.h; ++y) memcpy(m + fp.Offset + (uint64_t)y * fp.Footprint.RowPitch, data + (uint64_t)y * row, row);
  up->Unmap(0, nullptr);
  barrier(t.r, t.state, D3D12_RESOURCE_STATE_COPY_DEST);
  D3D12_TEXTURE_COPY_LOCATION dst = {}, src = {};
  dst.pResource = t.r;
  dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  src.pResource = up;
  src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
  src.PlacedFootprint = fp;
  g_l->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
  barrier(t.r, D3D12_RESOURCE_STATE_COPY_DEST, t.state);
  return up;
}

static std::vector<uint8_t> readback(Tex& t) {
  D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp;
  UINT64 total = 0;
  const D3D12_RESOURCE_DESC rd = t.r->GetDesc();
  g_dev->GetCopyableFootprints(&rd, 0, 1, 0, &fp, nullptr, nullptr, &total);
  ID3D12Resource* rb = buffer(total, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
  barrier(t.r, t.state, D3D12_RESOURCE_STATE_COPY_SOURCE);
  D3D12_TEXTURE_COPY_LOCATION dst = {}, src = {};
  dst.pResource = rb;
  dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
  dst.PlacedFootprint = fp;
  src.pResource = t.r;
  src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  g_l->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
  barrier(t.r, D3D12_RESOURCE_STATE_COPY_SOURCE, t.state);
  submit_wait();
  std::vector<uint8_t> out((size_t)t.w * t.h * bpp(t.f));
  uint8_t* m = nullptr;
  D3D12_RANGE r{0, (SIZE_T)total};
  CK(rb->Map(0, &r, (void**)&m));
  const uint32_t row = t.w * bpp(t.f);
  for (uint32_t y = 0; y < t.h; ++y) memcpy(out.data() + (size_t)y * row, m + fp.Offset + (uint64_t)y * fp.Footprint.RowPitch, row);
  rb->Unmap(0, nullptr);
  rb->Release();
  return out;
}

// ---------------------------------------------------------------------------------------------------- frame input
struct RoleIn {
  uint32_t role, format, w, h, ox, oy, state, conversion;
  float scale, bias;
  std::vector<uint8_t> data;
};
struct FrameIn {
  uint32_t rw = 0, rh = 0, reset = 0, settings = 0, depth_linear = 0, lox = 0, loy = 0;
  float jitter[2] = {}, mv_scale[2] = {}, v2c[16] = {}, w2v[16] = {};
  std::vector<RoleIn> roles;
};

static bool read_frame(const std::string& path, FrameIn* f) {
  std::ifstream s(path, std::ios::binary);
  if (!s) return false;
  auto u32 = [&]() { uint32_t v = 0; s.read((char*)&v, 4); return v; };
  auto f32 = [&]() { float v = 0; s.read((char*)&v, 4); return v; };
  if (u32() != 0x32465252u) return false;   // "RRF2"
  f->rw = u32(); f->rh = u32(); f->reset = u32(); f->settings = u32(); f->depth_linear = u32();
  f->jitter[0] = f32(); f->jitter[1] = f32(); f->mv_scale[0] = f32(); f->mv_scale[1] = f32();
  for (float& v : f->v2c) v = f32();
  for (float& v : f->w2v) v = f32();
  f->lox = u32(); f->loy = u32();
  const uint32_t n = u32();
  f->roles.resize(n);
  for (RoleIn& r : f->roles) {
    r.role = u32(); r.format = u32(); r.w = u32(); r.h = u32(); r.ox = u32(); r.oy = u32(); r.state = u32();
    r.conversion = u32(); r.scale = f32(); r.bias = f32();
    r.data.resize((size_t)r.w * r.h * bpp((DXGI_FORMAT)r.format));
    s.read((char*)r.data.data(), r.data.size());
  }
  return (bool)s;
}

// ---------------------------------------------------------------------------------------------------- the proxy's role
struct Host {
  std::string outdir;
  DXGI_FORMAT out_fmt = DXGI_FORMAT_R32G32B32A32_FLOAT;
  bool out_uav = true;
  void* inst = nullptr;
  WgRrCreate desc{};
  wgrr::Lifecycle* lc = nullptr;
  std::map<uint32_t, Tex> tex;   // by role (inputs and the separate output alpha)
  Tex out;
  int frame = 0;
  FILE* meta = nullptr;
  std::vector<ID3D12Resource*> uploads;
};

static Tex& role_tex(Host& h, const RoleIn& r) {
  Tex& t = h.tex[r.role];
  const bool uav = r.role == WG_ROLE_OUTPUT_ALPHA;
  if (!t.r || t.w != r.w || t.h != r.h || t.f != (DXGI_FORMAT)r.format) {
    if (t.r) t.r->Release();
    t = make_tex(r.w, r.h, (DXGI_FORMAT)r.format, uav, (D3D12_RESOURCE_STATES)r.state, uav ? 1 : input_mips(r.w, r.h));
  }
  return t;
}

// the output with a recognisable background, written before every evaluate: the resolve must leave every pixel outside
// its stores unchanged
static void ensure_output(Host& h) {
  const uint32_t w = h.desc.output_width + 3, ht = h.desc.output_height + 2;   // larger than the output rectangle
  if (!(h.out.r && h.out.w == w && h.out.h == ht && h.out.f == h.out_fmt)) {
    if (h.out.r) h.out.r->Release();
    h.out = make_tex(w, ht, h.out_fmt, h.out_uav, D3D12_RESOURCE_STATE_COMMON);
  }
  std::vector<uint8_t> bg((size_t)w * ht * bpp(h.out_fmt), 0x3c);
  h.uploads.push_back(upload(h.out, bg.data()));
}

static uint32_t g_transient_role = 0xffffffffu;   // this role is handed over as a lifetime-0 copy (62.1)

static int evaluate(Host& h, const FrameIn& in, uint32_t drop_role, WgRrFrameResult* res, wgrr::FramePlan* plan_out,
                    bool* completed) {
  void* copy = nullptr;
  for (const RoleIn& r : in.roles) {
    if (r.role == drop_role) continue;
    Tex& t = role_tex(h, r);
    h.uploads.push_back(upload(t, r.data.data()));
    if (r.role == g_transient_role) {   // the proxy's tagging call: the runtime copies it on the tag's list
      if (g_rt->size >= WG_RR_RUNTIME_V1_STATE_SIZE && g_rt->retain_transient_state) {
        // bug hunt C: the texture stays in its own state (a shader-resource state, not COMMON); the runtime moves it to
        // COPY_SOURCE for the copy and back
        if (g_rt->retain_transient_state(g_rt->ctx, g_l, t.r, r.format, (uint32_t)t.state, &copy) != WG_OK || !copy) {
          fprintf(stderr, "retain_transient_state failed\n");
          exit(8);
        }
      } else {
        barrier(t.r, t.state, D3D12_RESOURCE_STATE_COMMON);
        if (g_rt->retain_transient(g_rt->ctx, g_l, t.r, r.format, &copy) != WG_OK || !copy) {
          fprintf(stderr, "retain_transient failed\n");
          exit(8);
        }
        t.state = D3D12_RESOURCE_STATE_COMMON;   // promoted for the copy; decays to COMMON after the submission
      }
    }
  }
  ensure_output(h);
  WgRrFrame f;
  memset(&f, 0, sizeof f);
  f.size = sizeof f;
  f.api_minor = 14;
  f.command_list = g_l;
  f.render_width = in.rw;
  f.render_height = in.rh;
  f.output_width = h.desc.output_width;
  f.output_height = h.desc.output_height;
  wgrr::CameraPair cam;
  memcpy(cam.v2c, in.v2c, sizeof cam.v2c);
  memcpy(cam.w2v, in.w2v, sizeof cam.w2v);
  wgrr::FramePlan plan = h.lc->begin(in.rw, in.rh, in.jitter, cam, in.reset != 0, false);
  f.jitter[0] = plan.jitter[0];
  f.jitter[1] = plan.jitter[1];
  f.prev_jitter[0] = plan.prev_jitter[0];
  f.prev_jitter[1] = plan.prev_jitter[1];
  volatile float msx = in.mv_scale[0] * (float)in.rw, msy = in.mv_scale[1] * (float)in.rh;  // as the proxy (52.8)
  f.mv_scale[0] = msx;
  f.mv_scale[1] = msy;
  f.reset = plan.reset;
  f.parity = plan.parity;
  f.counter = plan.counter;
  f.rebuild_render_items = plan.rebuild;
  f.depth_is_linear = in.depth_linear;
  f.settings = in.settings;
  f.layer_origin_x = in.lox;
  f.layer_origin_y = in.loy;
  memcpy(f.view_to_clip, plan.current.v2c, sizeof f.view_to_clip);
  memcpy(f.world_to_view, plan.current.w2v, sizeof f.world_to_view);
  memcpy(f.prev_view_to_clip, plan.effective_prev.v2c, sizeof f.prev_view_to_clip);
  memcpy(f.prev_world_to_view, plan.effective_prev.w2v, sizeof f.prev_world_to_view);
  for (const RoleIn& r : in.roles) {
    if (r.role == drop_role) continue;
    Tex& t = h.tex[r.role];
    WgRrImage& im = f.images[r.role];
    im.resource = t.r;
    im.format = r.format;
    im.format_class = WG_FMT_FLOAT;
    im.state = r.state;
    im.width = r.w;
    im.height = r.h;
    im.origin_x = r.ox;
    im.origin_y = r.oy;
    im.lifetime = WG_LIFE_UNTIL_PRESENT;
    im.conversion = r.conversion;
    im.conversion_scale = r.scale;
    im.conversion_bias = r.bias;
    if (r.role == g_transient_role) im.resource = copy, im.lifetime = WG_LIFE_ONLY_DURING_TAG;
    f.attached |= 1u << r.role;
  }
  WgRrImage& o = f.images[WG_ROLE_OUTPUT];
  o.resource = h.out.r;
  o.format = h.out_fmt;
  o.format_class = WG_FMT_FLOAT;
  o.state = D3D12_RESOURCE_STATE_COMMON;
  o.width = h.out.w;
  o.height = h.out.h;
  o.origin_x = 1;            // a non-zero output origin
  o.origin_y = 1;
  f.attached |= 1u << WG_ROLE_OUTPUT;
  memset(res, 0, sizeof *res);
  res->size = sizeof *res;
  const uint32_t rc = g_rt->evaluate(g_rt->ctx, h.inst, &f, res);
  if (rc == WG_OK) h.lc->complete(plan);
  else h.lc->fail(plan, res->input_jitter_consumed != 0, false);
  *completed = rc == WG_OK;
  *plan_out = plan;
  submit_wait();
  if (copy) g_rt->release_transient(g_rt->ctx, copy);   // released by the runtime once the frame has completed
  for (ID3D12Resource* u : h.uploads) u->Release();
  h.uploads.clear();
  return (int)rc;
}

static void counters(int64_t c[4]) { g_test->counters(c); }
static uint64_t vram() {
  DXGI_QUERY_VIDEO_MEMORY_INFO i = {};
  g_adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &i);
  return i.CurrentUsage;
}

static void create(Host& h, const WgRrCreate& d) {
  void* inst = nullptr;
  const uint32_t rc = g_rt->create_instance(g_rt->ctx, &d, &inst);
  if (rc != WG_OK) {
    fprintf(stderr, "create_instance failed: %u\n", rc);
    exit(5);
  }
  h.inst = inst;
  h.desc = d;
  delete h.lc;
  h.lc = new wgrr::Lifecycle(d.render_width, d.render_height);
}
static void destroy(Host& h) {
  if (h.inst) g_rt->destroy_instance(g_rt->ctx, h.inst);
  h.inst = nullptr;
}

static void perf(Host& h, uint32_t Wo, uint32_t Ho, uint32_t rw, uint32_t rh, int frames, bool ext) {
  WgRrCreate d = {};
  d.size = sizeof d;
  d.api_minor = 14;
  d.device = g_dev;
  d.output_width = Wo;
  d.output_height = Ho;
  d.render_width = rw;
  d.render_height = rh;
  d.colour_hdr = 1;
  d.hw_depth = 1;
  d.alpha_upscaling = ext ? 1 : 0;
  Host p;
  p.desc = d;
  create(p, d);
  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  FrameIn in;
  in.rw = rw;
  in.rh = rh;
  in.mv_scale[0] = 1.0f / rw;
  in.mv_scale[1] = 1.0f / rh;
  for (int i = 0; i < 16; ++i) in.v2c[i] = in.w2v[i] = (i % 5 == 0) ? 1.0f : 0.0f;
  in.v2c[11] = 1.0f;
  in.v2c[14] = -0.1f;
  in.v2c[15] = 0.0f;
  struct Rl { uint32_t role; DXGI_FORMAT f; };
  std::vector<Rl> rs = {{WG_ROLE_COLOUR, DXGI_FORMAT_R16G16B16A16_FLOAT}, {WG_ROLE_DEPTH, DXGI_FORMAT_R32_FLOAT},
                        {WG_ROLE_MOTION, DXGI_FORMAT_R16G16_FLOAT}, {WG_ROLE_DIFFUSE_ALBEDO, DXGI_FORMAT_R8G8B8A8_UNORM},
                        {WG_ROLE_SPECULAR_ALBEDO, DXGI_FORMAT_R8G8B8A8_UNORM}, {WG_ROLE_NORMALS, DXGI_FORMAT_R16G16B16A16_FLOAT},
                        {WG_ROLE_ROUGHNESS, DXGI_FORMAT_R16_FLOAT}, {WG_ROLE_SPECULAR_HIT_DIST, DXGI_FORMAT_R16_FLOAT}};
  if (ext) rs.push_back({WG_ROLE_LAYER, DXGI_FORMAT_R16G16B16A16_FLOAT});
  Tex aout;
  if (ext) aout = make_tex(Wo, Ho, DXGI_FORMAT_R16_FLOAT, true, D3D12_RESOURCE_STATE_COMMON);
  for (const Rl& r : rs) {
    RoleIn ri;
    ri.role = r.role;
    ri.format = r.f;
    ri.w = rw;
    ri.h = rh;
    ri.ox = ri.oy = 0;
    ri.state = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    ri.conversion = 0;
    ri.scale = 1.0f;
    ri.bias = 0.0f;
    ri.data.resize((size_t)rw * rh * bpp(r.f));
    for (size_t i = 0; i < ri.data.size(); ++i) ri.data[i] = (uint8_t)(rng() & 0x3f);   // small finite values
    in.roles.push_back(ri);
  }
  p.out_fmt = DXGI_FORMAT_R16G16B16A16_FLOAT;
  // upload once, then only evaluates
  for (const RoleIn& r : in.roles) p.uploads.push_back(upload(role_tex(p, r), r.data.data()));
  ensure_output(p);
  submit_wait();
  for (ID3D12Resource* x : p.uploads) x->Release();
  p.uploads.clear();
  LARGE_INTEGER t0, t1, fq;
  QueryPerformanceFrequency(&fq);
  QueryPerformanceCounter(&t0);
  int ok = 0;
  static ID3D12CommandAllocator* al2[3] = {};
  if (!al2[0]) for (int k = 0; k < 3; ++k) CK(g_dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&al2[k])));
  for (int i = 0; i < frames; ++i) {
    in.jitter[0] = u(rng) - 0.5f;
    in.jitter[1] = u(rng) - 0.5f;
    WgRrFrame f;
    memset(&f, 0, sizeof f);
    f.size = sizeof f;
    f.api_minor = 14;
    f.command_list = g_l;
    f.render_width = rw;
    f.render_height = rh;
    f.output_width = Wo;
    f.output_height = Ho;
    wgrr::CameraPair cam;
    memcpy(cam.v2c, in.v2c, sizeof cam.v2c);
    memcpy(cam.w2v, in.w2v, sizeof cam.w2v);
    wgrr::FramePlan plan = p.lc->begin(rw, rh, in.jitter, cam, false, false);
    f.jitter[0] = plan.jitter[0]; f.jitter[1] = plan.jitter[1];
    f.prev_jitter[0] = plan.prev_jitter[0]; f.prev_jitter[1] = plan.prev_jitter[1];
    f.mv_scale[0] = in.mv_scale[0] * rw; f.mv_scale[1] = in.mv_scale[1] * rh;
    f.reset = plan.reset; f.parity = plan.parity; f.counter = plan.counter; f.rebuild_render_items = plan.rebuild;
    memcpy(f.view_to_clip, plan.current.v2c, 64); memcpy(f.world_to_view, plan.current.w2v, 64);
    memcpy(f.prev_view_to_clip, plan.effective_prev.v2c, 64); memcpy(f.prev_world_to_view, plan.effective_prev.w2v, 64);
    for (const RoleIn& r : in.roles) {
      WgRrImage& im = f.images[r.role];
      im.resource = p.tex[r.role].r; im.format = r.format; im.format_class = WG_FMT_FLOAT; im.state = r.state;
      im.width = rw; im.height = rh; im.lifetime = WG_LIFE_UNTIL_PRESENT;
      f.attached |= 1u << r.role;
    }
    WgRrImage& o = f.images[WG_ROLE_OUTPUT];
    o.resource = p.out.r; o.format = p.out_fmt; o.format_class = WG_FMT_FLOAT; o.state = D3D12_RESOURCE_STATE_COMMON;
    o.width = p.out.w; o.height = p.out.h;
    f.attached |= 1u << WG_ROLE_OUTPUT;
    if (ext) {
      WgRrImage& a = f.images[WG_ROLE_OUTPUT_ALPHA];
      a.resource = aout.r; a.format = aout.f; a.format_class = WG_FMT_FLOAT; a.state = D3D12_RESOURCE_STATE_COMMON;
      a.width = Wo; a.height = Ho;
      f.attached |= 1u << WG_ROLE_OUTPUT_ALPHA;
    }
    WgRrFrameResult res = {};
    res.size = sizeof res;
    const uint32_t rc = g_rt->evaluate(g_rt->ctx, p.inst, &f, &res);
    if (rc == WG_OK) p.lc->complete(plan), ok++;
    else p.lc->fail(plan, res.input_jitter_consumed != 0, false);
    // a game submits every frame; keep up to 2 frames queued
    CK(g_l->Close());
    ID3D12CommandList* ll = g_l;
    g_q->ExecuteCommandLists(1, &ll);
    CK(g_q->Signal(g_fence, ++g_fv));
    ID3D12CommandAllocator* next = al2[g_fv % 3];
    if (g_fv >= 3) {
      CK(g_fence->SetEventOnCompletion(g_fv - 2, g_ev));
      WaitForSingleObject(g_ev, 60000);
    }
    CK(next->Reset());
    CK(g_l->Reset(next, nullptr));
  }
  CK(g_l->Close());
  CK(g_q->Signal(g_fence, ++g_fv));
  CK(g_fence->SetEventOnCompletion(g_fv, g_ev));
  WaitForSingleObject(g_ev, 60000);
  CK(g_al->Reset());
  CK(g_l->Reset(g_al, nullptr));
  QueryPerformanceCounter(&t1);
  const double wall = (double)(t1.QuadPart - t0.QuadPart) / fq.QuadPart;
  const uint64_t vr = g_rt->instance_vram(g_rt->ctx, p.inst);
  uint64_t lay[24] = {};
  g_test->layout(p.inst, lay, 24);
  destroy(p);
  g_test->drain(10000);
  std::string path = h.outdir + "\\perf.json";
  FILE* fo = fopen(path.c_str(), "ab");
  char hv[16] = {};
  GetEnvironmentVariableA("WG_RR_HEAD", hv, sizeof hv);   // unset = the runtime's default head (fast)
  const bool fast = strcmp(hv, "exact") != 0 && strcmp(hv, "strict") != 0 && strcmp(hv, "debug") != 0;
  fprintf(fo, "{\"output\":[%u,%u],\"render\":[%u,%u],\"frames\":%d,\"ok\":%d,\"wall_s\":%.3f,\"instance_vram\":%llu,"
              "\"head\":\"%s\",\"resolve\":\"%s\",\"fixed_end\":%llu,\"state_end\":%llu,\"work_bytes\":%llu,"
              "\"arena_bytes\":%llu,\"model_bytes\":%llu,\"aliased\":%llu}\n",
          Wo, Ho, rw, rh, frames, ok, wall, (unsigned long long)vr, fast ? "fast" : "strict", ext ? "ext" : "plain",
          (unsigned long long)lay[7], (unsigned long long)lay[13], (unsigned long long)lay[22], (unsigned long long)lay[14],
          (unsigned long long)(lay[1] - lay[0]), (unsigned long long)lay[23]);
  fclose(fo);
  printf("perf %ux%u -> %ux%u %s: %d frames, %d ok, %.3f s wall, instance vram %llu\n", rw, rh, Wo, Ho, ext ? "ext" : "plain",
         frames, ok, wall, (unsigned long long)vr);
  for (auto& kv : p.tex) kv.second.r->Release();
  if (p.out.r) p.out.r->Release();
  if (aout.r) aout.r->Release();
  delete p.lc;
}

// clean/rr-f-speed5: game-setup timing. The frames (frame files of tests/e2e.py write_frame / the capture replay, in the
// creation of the current instance) are uploaded once into resident textures (one set per frame) and evaluated in order
// <passes> times without waiting (up to 2 frames queued, as perf), so the clock ramps up as in a game; passes after the
// first drop the files' reset flags. The runtime logs the GPU time per WG_RR_TIMING_WINDOW frames (tools/game_bench.py).
static void perfcap(Host& h, int passes, const std::vector<std::string>& files) {
  std::vector<FrameIn> fr(files.size());
  std::vector<std::map<uint32_t, Tex>> tx(files.size());
  for (size_t i = 0; i < files.size(); ++i) {
    if (!read_frame(files[i], &fr[i])) {
      fprintf(stderr, "perfcap: cannot read %s\n", files[i].c_str());
      exit(9);
    }
    for (RoleIn& r : fr[i].roles) {
      Tex& t = tx[i][r.role];
      t = make_tex(r.w, r.h, (DXGI_FORMAT)r.format, false, (D3D12_RESOURCE_STATES)r.state, input_mips(r.w, r.h));
      h.uploads.push_back(upload(t, r.data.data()));
      r.data.clear();
      r.data.shrink_to_fit();
    }
    ensure_output(h);
    submit_wait();
    for (ID3D12Resource* u : h.uploads) u->Release();
    h.uploads.clear();
  }
  static ID3D12CommandAllocator* al2[3] = {};
  if (!al2[0]) for (int k = 0; k < 3; ++k) CK(g_dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&al2[k])));
  LARGE_INTEGER t0, t1, fq;
  QueryPerformanceFrequency(&fq);
  QueryPerformanceCounter(&t0);
  int ok = 0, n = 0;
  for (int k = 0; k < passes; ++k) {
    for (size_t i = 0; i < fr.size(); ++i, ++n) {
      const FrameIn& in = fr[i];
      WgRrFrame f;
      memset(&f, 0, sizeof f);
      f.size = sizeof f;
      f.api_minor = 14;
      f.command_list = g_l;
      f.render_width = in.rw;
      f.render_height = in.rh;
      f.output_width = h.desc.output_width;
      f.output_height = h.desc.output_height;
      wgrr::CameraPair cam;
      memcpy(cam.v2c, in.v2c, sizeof cam.v2c);
      memcpy(cam.w2v, in.w2v, sizeof cam.w2v);
      wgrr::FramePlan plan = h.lc->begin(in.rw, in.rh, in.jitter, cam, k == 0 && in.reset != 0, false);
      f.jitter[0] = plan.jitter[0];
      f.jitter[1] = plan.jitter[1];
      f.prev_jitter[0] = plan.prev_jitter[0];
      f.prev_jitter[1] = plan.prev_jitter[1];
      volatile float msx = in.mv_scale[0] * (float)in.rw, msy = in.mv_scale[1] * (float)in.rh;  // as the proxy (52.8)
      f.mv_scale[0] = msx;
      f.mv_scale[1] = msy;
      f.reset = plan.reset;
      f.parity = plan.parity;
      f.counter = plan.counter;
      f.rebuild_render_items = plan.rebuild;
      f.depth_is_linear = in.depth_linear;
      f.settings = in.settings;
      f.layer_origin_x = in.lox;
      f.layer_origin_y = in.loy;
      memcpy(f.view_to_clip, plan.current.v2c, sizeof f.view_to_clip);
      memcpy(f.world_to_view, plan.current.w2v, sizeof f.world_to_view);
      memcpy(f.prev_view_to_clip, plan.effective_prev.v2c, sizeof f.prev_view_to_clip);
      memcpy(f.prev_world_to_view, plan.effective_prev.w2v, sizeof f.prev_world_to_view);
      for (const RoleIn& r : in.roles) {
        const Tex& t = tx[i][r.role];
        WgRrImage& im = f.images[r.role];
        im.resource = t.r;
        im.format = r.format;
        im.format_class = WG_FMT_FLOAT;
        im.state = r.state;
        im.width = r.w;
        im.height = r.h;
        im.origin_x = r.ox;
        im.origin_y = r.oy;
        im.lifetime = WG_LIFE_UNTIL_PRESENT;
        im.conversion = r.conversion;
        im.conversion_scale = r.scale;
        im.conversion_bias = r.bias;
        f.attached |= 1u << r.role;
      }
      WgRrImage& o = f.images[WG_ROLE_OUTPUT];
      o.resource = h.out.r;
      o.format = h.out_fmt;
      o.format_class = WG_FMT_FLOAT;
      o.state = D3D12_RESOURCE_STATE_COMMON;
      o.width = h.out.w;
      o.height = h.out.h;
      o.origin_x = 1;
      o.origin_y = 1;
      f.attached |= 1u << WG_ROLE_OUTPUT;
      WgRrFrameResult res = {};
      res.size = sizeof res;
      const uint32_t rc = g_rt->evaluate(g_rt->ctx, h.inst, &f, &res);
      if (rc == WG_OK) h.lc->complete(plan), ok++;
      else h.lc->fail(plan, res.input_jitter_consumed != 0, false);
      CK(g_l->Close());
      ID3D12CommandList* ll = g_l;
      g_q->ExecuteCommandLists(1, &ll);
      CK(g_q->Signal(g_fence, ++g_fv));
      ID3D12CommandAllocator* next = al2[g_fv % 3];
      if (g_fv >= 3) {
        CK(g_fence->SetEventOnCompletion(g_fv - 2, g_ev));
        WaitForSingleObject(g_ev, 60000);
      }
      CK(next->Reset());
      CK(g_l->Reset(next, nullptr));
    }
  }
  CK(g_l->Close());
  CK(g_q->Signal(g_fence, ++g_fv));
  CK(g_fence->SetEventOnCompletion(g_fv, g_ev));
  WaitForSingleObject(g_ev, 60000);
  CK(g_al->Reset());
  CK(g_l->Reset(g_al, nullptr));
  QueryPerformanceCounter(&t1);
  printf("perfcap %zu frames x %d passes: %d evaluates, %d ok, %.3f s wall\n", fr.size(), passes, n, ok,
         (double)(t1.QuadPart - t0.QuadPart) / fq.QuadPart);
  for (auto& m : tx)
    for (auto& kv : m) kv.second.r->Release();
}

int main(int argc, char** argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: rr_test_host <job file>\n");
    return 1;
  }
  std::ifstream jf(argv[1]);
  if (!jf) return 1;
  IDXGIFactory6* fac = nullptr;
  CK(CreateDXGIFactory1(IID_PPV_ARGS(&fac)));
  IDXGIAdapter1* ad = nullptr;
  for (UINT i = 0;; ++i) {
    IDXGIAdapter1* x = nullptr;
    if (fac->EnumAdapters1(i, &x) == DXGI_ERROR_NOT_FOUND) break;
    DXGI_ADAPTER_DESC1 d;
    x->GetDesc1(&d);
    if (d.VendorId == 0x1002 && !(d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) {
      ad = x;
      break;
    }
    x->Release();
  }
  if (!ad) {
    fprintf(stderr, "no AMD adapter\n");
    return 2;
  }
  CK(ad->QueryInterface(IID_PPV_ARGS(&g_adapter)));
  CK(D3D12CreateDevice(ad, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&g_dev)));
  D3D12_COMMAND_QUEUE_DESC qd = {};
  qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
  CK(g_dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&g_q)));
  CK(g_dev->CreateCommandAllocator(qd.Type, IID_PPV_ARGS(&g_al)));
  CK(g_dev->CreateCommandList(0, qd.Type, g_al, nullptr, IID_PPV_ARGS(&g_l)));
  CK(g_dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&g_fence)));
  g_ev = CreateEvent(nullptr, FALSE, FALSE, nullptr);
  // i15: WG_TEST_BLOCK=<option>=<value>[,<option>=<value>...] creates the shared settings block before the job runs (the
  // runtime then finds it at its first mode read): the same as "block" job lines, for harnesses that write the job
  // themselves (gate replay, bytecmp)
  {
    char tb[512] = {};
    if (GetEnvironmentVariableA("WG_TEST_BLOCK", tb, sizeof tb)) {
      static wgss::Block blk;
      std::vector<std::string> lines;
      if (!blk.open(WG_SS_WRITER_TEST, true, std::wstring(), &lines, true) || !blk.claim_writer(WG_SS_WRITER_TEST)) {
        fprintf(stderr, "WG_TEST_BLOCK: cannot create / claim the shared settings block\n");
        return 8;
      }
      std::string all(tb), item;
      std::istringstream is(all);
      while (std::getline(is, item, ',')) {
        const size_t eq = item.find('=');
        const WgOptionDef* o = eq == std::string::npos ? nullptr : wgss::option_by_key(item.substr(0, eq));
        int32_t v = 0;
        if (!o || !wgss::parse_value(*o, item.substr(eq + 1), &v) || !blk.write_slot(WG_SS_WRITER_TEST, o->id, v)) {
          fprintf(stderr, "WG_TEST_BLOCK: bad item %s\n", item.c_str());
          return 8;
        }
        printf("block %s (WG_TEST_BLOCK)\n", wgss::describe(o->id, v, "test block").c_str());
      }
    }
  }
  Host h;
  std::string line;
  while (std::getline(jf, line)) {
    std::istringstream s(line);
    std::string cmd;
    if (!(s >> cmd) || cmd[0] == '#') continue;
    if (cmd == "runtime") {
      std::string dll;
      s >> dll;
      HMODULE m = LoadLibraryA(dll.c_str());
      if (!m) {
        fprintf(stderr, "cannot load %s (%lu)\n", dll.c_str(), GetLastError());
        return 3;
      }
      auto e = (WgRrRuntimeEntryFn)GetProcAddress(m, WG_RR_RUNTIME_EXPORT);
      auto t = (const WgRrRuntimeTestV1* (*)(void))GetProcAddress(m, "WrongGPU_RrRuntimeTestV1");
      if (!e || !t) return 3;
      g_rt = e();
      g_test = t();
      const uint32_t st = g_rt->status(g_rt->ctx);
      LUID luid = g_dev->GetAdapterLuid();
      const uint32_t sup = g_rt->adapter_supported(g_rt->ctx, &luid, sizeof luid);
      uint32_t ver[3];
      g_rt->release_version(g_rt->ctx, ver);
      const uint32_t rc = g_rt->set_device(g_rt->ctx, g_dev);
      int64_t c[4];
      counters(c);
      LUID zero = {};
      const uint32_t sup0 = g_rt->adapter_supported(g_rt->ctx, &zero, sizeof zero);
      Tex probe = make_tex(8, 4, DXGI_FORMAT_R16G16B16A16_FLOAT, false, D3D12_RESOURCE_STATE_COMMON);
      uint32_t pf = 0, ps = 0;
      const uint32_t drc = g_rt->describe_resource(g_rt->ctx, probe.r, &pf, &ps);
      probe.r->Release();
      printf("runtime %u.%u.%u status %u adapter_supported %u (unknown LUID %u) describe %u format %u samples %u "
             "set_device %u status %u live %lld\n", ver[0], ver[1], ver[2], st, sup, sup0, drc, pf, ps, rc,
             g_rt->status(g_rt->ctx), (long long)c[0]);
      if (rc != WG_OK) return 6;
    } else if (cmd == "outdir") {
      s >> h.outdir;
      CreateDirectoryA(h.outdir.c_str(), nullptr);
      h.meta = fopen((h.outdir + "\\meta.jsonl").c_str(), "wb");
    } else if (cmd == "block") {
      static wgss::Block blk;
      std::string key, val;
      s >> key >> val;
      std::vector<std::string> lines;
      if (!blk.present() && !blk.open(WG_SS_WRITER_TEST, true, std::wstring(), &lines, true)) {
        fprintf(stderr, "block: cannot open the shared settings block\n");
        return 8;
      }
      const WgOptionDef* o = wgss::option_by_key(key);
      int32_t v = 0;
      if (!o || !wgss::parse_value(*o, val, &v)) {
        fprintf(stderr, "block: unknown option or value %s=%s\n", key.c_str(), val.c_str());
        return 8;
      }
      if (!blk.claim_writer(WG_SS_WRITER_TEST) || !blk.write_slot(WG_SS_WRITER_TEST, o->id, v)) {
        fprintf(stderr, "block: not the writer (writer %u)\n", blk.writer());
        return 8;
      }
      printf("block %s (created by %s, generation %u)\n", wgss::describe(o->id, v, "test block").c_str(),
             blk.created() ? "the host" : "another module", blk.view()->header.generation);
    } else if (cmd == "env") {
      std::string n, v;
      s >> n >> v;
      SetEnvironmentVariableA(n.c_str(), v == "-" ? nullptr : v.c_str());
    } else if (cmd == "output") {
      uint32_t f, u;
      s >> f >> u;
      h.out_fmt = (DXGI_FORMAT)f;
      h.out_uav = u != 0;
    } else if (cmd == "create") {
      WgRrCreate d = {};
      d.size = sizeof d;
      d.api_minor = 14;
      d.device = g_dev;
      s >> d.output_width >> d.output_height >> d.render_width >> d.render_height >> d.packing >> d.colour_hdr >>
          d.depth_inverted >> d.mv_jittered >> d.hw_depth >> d.alpha_upscaling;
      create(h, d);
      h.frame = 0;
    } else if (cmd == "frame" || cmd == "fail" || cmd == "transient") {
      std::string file, opt, tok;
      s >> file;
      uint32_t drop_role = 0xffffffffu;
      if (cmd == "fail") s >> drop_role;
      g_transient_role = 0xffffffffu;
      if (cmd == "transient") s >> g_transient_role;
      bool force_reset = false;
      while (s >> tok) {
        if (tok == "reset") force_reset = true;
        else opt = tok;
      }
      FrameIn in;
      if (!read_frame(file, &in)) {
        fprintf(stderr, "bad frame file %s\n", file.c_str());
        return 7;
      }
      if (force_reset) in.reset = 1;
      WgRrFrameResult res;
      wgrr::FramePlan plan;
      bool completed = false;
      const int rc = evaluate(h, in, drop_role, &res, &plan, &completed);
      uint64_t lay[24] = {};
      g_test->layout(h.inst, lay, 24);
      char name[64];
      snprintf(name, sizeof name, "\\frame_%03d", h.frame);
      if (opt != "nodump") {
        std::wstring wp(h.outdir.begin(), h.outdir.end());
        wp += std::wstring(name, name + strlen(name)) + L".arena";
        if (g_test->dump(h.inst, wp.c_str()) != 0) fprintf(stderr, "dump failed\n");
      }
      {
        std::vector<uint8_t> o = readback(h.out);
        FILE* fo = fopen((h.outdir + name + ".out").c_str(), "wb");
        fwrite(o.data(), 1, o.size(), fo);
        fclose(fo);
        auto it = h.tex.find(WG_ROLE_OUTPUT_ALPHA);
        bool has_a = false;
        for (const RoleIn& r : in.roles) has_a = has_a || r.role == WG_ROLE_OUTPUT_ALPHA;
        if (has_a && it != h.tex.end()) {
          std::vector<uint8_t> a = readback(it->second);
          FILE* fa = fopen((h.outdir + name + ".aout").c_str(), "wb");
          fwrite(a.data(), 1, a.size(), fa);
          fclose(fa);
        }
      }
      fprintf(h.meta, "{\"frame\":%d,\"kind\":\"%s\",\"file\":\"%s\",\"rc\":%d,\"passes\":%u,\"jitter_consumed\":%u,"
                      "\"camera_fallback\":%u,\"reset\":%d,\"game_reset\":%u,\"parity\":%u,\"counter\":%u,\"rebuild\":%d,"
                      "\"prev_jitter\":[%.9g,%.9g],\"jitter\":[%.9g,%.9g],\"render\":[%u,%u],\"out_tex\":[%u,%u,%u],"
                      "\"layout\":[",
              h.frame, cmd.c_str(), file.c_str(), rc, res.passes_started, res.input_jitter_consumed, res.camera_fallback_used,
              plan.reset ? 1 : 0, in.reset, plan.parity, plan.counter, plan.rebuild ? 1 : 0, plan.prev_jitter[0],
              plan.prev_jitter[1], plan.jitter[0], plan.jitter[1], in.rw, in.rh, h.out.w, h.out.h, (unsigned)h.out_fmt);
      for (int i = 0; i < 24; ++i) fprintf(h.meta, "%s%llu", i ? "," : "", (unsigned long long)lay[i]);
      static char map[1 << 16];
      map[0] = 0;
      g_test->describe(h.inst, map, sizeof map);
      std::string mj;
      for (const char* c = map; *c; ++c) mj += *c == '\n' ? std::string("|") : std::string(1, *c);
      fprintf(h.meta, "],\"map\":\"%s\",\"completed\":%d}\n", mj.c_str(), completed ? 1 : 0);
      fflush(h.meta);
      printf("frame %d %s rc %d reset %d parity %u rebuild %d\n", h.frame, file.c_str(), rc, plan.reset ? 1 : 0,
             plan.parity, plan.rebuild ? 1 : 0);
      h.frame++;
    } else if (cmd == "border") {   // bug hunt G: an output without unordered access (our texture + copy into it)
      std::string file;
      s >> file;
      FrameIn in;
      if (!read_frame(file, &in)) {
        fprintf(stderr, "bad frame file %s\n", file.c_str());
        return 7;
      }
      const bool was_uav = h.out_uav;
      h.out_uav = false;
      if (h.out.r) h.out.r->Release();
      h.out = Tex();
      WgRrFrameResult res;
      wgrr::FramePlan plan;
      bool completed = false;
      const int rc = evaluate(h, in, 0xffffffffu, &res, &plan, &completed);   // the output starts as 0x3c bytes
      std::vector<uint8_t> o = readback(h.out);
      // the store's grid: We x He (the output size rounded up to even) at the origin (1, 1); nothing outside it changes
      const uint32_t bp = bpp(h.out_fmt), Wo = h.desc.output_width, Ho = h.desc.output_height;
      const uint32_t x1 = 1 + Wo + Wo % 2, y1 = 1 + Ho + Ho % 2;
      uint64_t changed = 0, inside = 0;
      for (uint32_t y = 0; y < h.out.h; ++y)
        for (uint32_t x = 0; x < h.out.w; ++x) {
          const bool in_rect = x >= 1 && y >= 1 && x < x1 && y < y1;
          bool same = true;
          for (uint32_t b = 0; b < bp; ++b) same = same && o[((size_t)y * h.out.w + x) * bp + b] == 0x3c;
          if (!in_rect && !same) ++changed;
          if (in_rect && !same) ++inside;
        }
      const bool ok = rc == WG_OK && changed == 0 && inside > 0;
      printf("border: output without UAV, rc %d: %llu pixel(s) written inside the store's rectangle, %llu changed outside: %s\n", rc,
             (unsigned long long)inside, (unsigned long long)changed, ok ? "ok" : "FAIL");
      h.out_uav = was_uav;
      if (h.out.r) h.out.r->Release();
      h.out = Tex();
      if (!ok) return 11;
    } else if (cmd == "arrayin") {   // bug hunt E: <role> as a 2-slice array texture: refused, nothing recorded
      std::string file;
      uint32_t role = 0;
      s >> file >> role;
      FrameIn in;
      if (!read_frame(file, &in)) {
        fprintf(stderr, "bad frame file %s\n", file.c_str());
        return 7;
      }
      auto drop = [&]() {
        auto it = h.tex.find(role);
        if (it != h.tex.end()) {
          if (it->second.r) it->second.r->Release();
          h.tex.erase(it);
        }
      };
      drop();
      g_array_slices = 2;
      for (const RoleIn& r : in.roles)
        if (r.role == role) role_tex(h, r);   // made now, as an array (evaluate uploads into its slice 0)
      g_array_slices = 1;
      WgRrFrameResult res;
      wgrr::FramePlan plan;
      bool completed = false;
      const int rc = evaluate(h, in, 0xffffffffu, &res, &plan, &completed);
      const HRESULT removed = g_dev->GetDeviceRemovedReason();
      const bool ok = rc != WG_OK && res.passes_started == 0 && removed == S_OK;
      printf("arrayin role %u: evaluate rc %d, passes %u, device %s: %s\n", role, rc, res.passes_started,
             removed == S_OK ? "ok" : "REMOVED", ok ? "ok" : "FAIL (an array texture was taken)");
      drop();   // the next frames get their usual textures
      if (!ok) return 10;
    } else if (cmd == "retag") {
      std::string file;
      uint32_t role = 0;
      s >> file >> role;
      FrameIn in;
      if (!read_frame(file, &in)) {
        fprintf(stderr, "bad frame file %s\n", file.c_str());
        return 7;
      }
      bool ok = false;
      for (const RoleIn& r : in.roles) {
        if (r.role != role) continue;
        Tex& t = role_tex(h, r);
        h.uploads.push_back(upload(t, r.data.data()));
        barrier(t.r, t.state, D3D12_RESOURCE_STATE_COMMON);
        t.state = D3D12_RESOURCE_STATE_COMMON;
        int64_t c0[4], c1[4], c2[4];
        counters(c0);
        void* copy = nullptr;
        if (g_rt->retain_transient(g_rt->ctx, g_l, t.r, r.format, &copy) != WG_OK || !copy) {
          fprintf(stderr, "retain_transient failed\n");
          return 8;
        }
        g_rt->release_transient(g_rt->ctx, copy);   // re-tagged before any evaluate; the list has not run yet
        counters(c1);
        submit_wait();
        g_test->drain(5000);
        counters(c2);
        ok = c1[0] == c0[0] + 1 && c1[2] == c0[2] + 1 && c2[0] == c0[0] && c2[2] == c0[2];
        printf("retag role %u: before the list ran live %lld deferred %lld (was %lld / %lld); after it live %lld deferred %lld: %s\n",
               role, (long long)c1[0], (long long)c1[2], (long long)c0[0], (long long)c0[2], (long long)c2[0], (long long)c2[2],
               ok ? "ok" : "FAIL (the copy was freed before its list ran, or never)");
      }
      for (ID3D12Resource* u : h.uploads) u->Release();
      h.uploads.clear();
      if (!ok) return 9;
    } else if (cmd == "destroy") {
      destroy(h);
    } else if (cmd == "leak") {
      int cycles;
      std::string file;
      s >> cycles >> file;
      FrameIn in;
      if (!read_frame(file, &in)) return 7;
      g_test->drain(10000);
      int64_t c0[4], c1[4];
      counters(c0);
      const uint64_t v0 = vram();
      WgRrCreate d = h.desc;
      const DXGI_FORMAT fmts[3] = {DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, DXGI_FORMAT_R11G11B10_FLOAT};
      int bad = 0;
      for (int i = 0; i < cycles; ++i) {
        h.out_fmt = fmts[i % 3];
        h.out_uav = (i % 3) != 1;
        d.alpha_upscaling = (uint32_t)(i % 2);
        create(h, d);
        for (int k = 0; k < 2; ++k) {
          WgRrFrameResult res;
          wgrr::FramePlan plan;
          bool completed;
          if (evaluate(h, in, 0xffffffffu, &res, &plan, &completed) != 0) bad++;
        }
        destroy(h);
      }
      if (h.out.r) h.out.r->Release(), h.out.r = nullptr;
      g_test->drain(10000);
      counters(c1);
      const uint64_t v1 = vram();
      FILE* fo = fopen((h.outdir + "\\leak.json").c_str(), "wb");
      fprintf(fo, "{\"cycles\":%d,\"failed_frames\":%d,\"live_before\":%lld,\"live_after\":%lld,\"instances_after\":%lld,"
                  "\"deferred_after\":%lld,\"created_total\":%lld,\"vram_before\":%llu,\"vram_after\":%llu}\n",
              cycles, bad, (long long)c0[0], (long long)c1[0], (long long)c1[1], (long long)c1[2], (long long)c1[3],
              (unsigned long long)v0, (unsigned long long)v1);
      fclose(fo);
      printf("leak: %d cycles, live %lld -> %lld, instances %lld, deferred %lld, vram %llu -> %llu\n", cycles,
             (long long)c0[0], (long long)c1[0], (long long)c1[1], (long long)c1[2], (unsigned long long)v0,
             (unsigned long long)v1);
    } else if (cmd == "perf") {
      uint32_t Wo, Ho, rw, rh;
      int frames;
      std::string mode, kind;
      s >> Wo >> Ho >> rw >> rh >> frames >> mode >> kind;
      SetEnvironmentVariableA("WG_RR_HEAD", mode == "fast" ? "fast" : "exact");
      perf(h, Wo, Ho, rw, rh, frames, kind == "ext");
      SetEnvironmentVariableA("WG_RR_HEAD", nullptr);
    } else if (cmd == "perfcap") {      // perfcap <passes> <frame file> ... (game-setup timing; tools/game_bench.py)
      int passes = 1;
      s >> passes;
      std::vector<std::string> files;
      std::string fname;
      while (s >> fname) files.push_back(fname);
      perfcap(h, passes, files);
    } else if (cmd == "end") {
      break;
    } else {
      fprintf(stderr, "unknown command %s\n", cmd.c_str());
      return 1;
    }
  }
  destroy(h);
  if (g_test) {
    g_test->drain(10000);
    int64_t c[4];
    counters(c);
    printf("end: live resources %lld, instances %lld, deferred %lld, created %lld\n", (long long)c[0], (long long)c[1],
           (long long)c[2], (long long)c[3]);
    const uint32_t rc = g_test->shutdown();
    counters(c);
    printf("shutdown %u: live resources %lld\n", rc, (long long)c[0]);
  }
  if (h.meta) fclose(h.meta);
  for (auto& kv : h.tex) kv.second.r->Release();
  if (h.out.r) h.out.r->Release();
  return 0;
}
