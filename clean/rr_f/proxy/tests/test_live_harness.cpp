// SPDX-License-Identifier: MIT
// Live-test harness (clean/rr-f-live2): a fake game that drives the REAL release hook ASI (src/hook_dll.cpp) and the
// REAL RR runtime on the hardware D3D12 device, the way game A (Cyberpunk 2077, 2.7 interface) does: it statically
// imports the interface functions by name from the stand-in interface DLL (tests/gen_fake_original.py), loads the ASI
// as OptiScaler would and calls InitializeASI; the ASI loads the runtime next to it; rr_mode in wronggpu-rr.ini (exact / accurate / fast) selects its mode, next to
// the exe. Inputs are the captured frames (CAPTURE.md) prepared by tools/live_harness.py as .lvf files: game A's
// native formats (RGBA16F, RGBA8 unorm, R32_TYPELESS depth with the depth-stencil flag, R16F, R32F, R8), its declared
// states (UNORDERED_ACCESS / 0xc0), resource descriptions WITHOUT width / height (0 x 0), packing 1, inverted
// hardware depth, non-jittered motion, type 39 every frame, the output RGBA16F with a typed UAV in UNORDERED_ACCESS.
//
//   test_live_harness.exe <frames dir> <out dir> match  [passes]       outputs of every evaluate -> out\frame_NNN.out
//   test_live_harness.exe <frames dir> <out dir> soak   [evaluates]    resets, render / output size changes, VRAM
//   test_live_harness.exe <frames dir> <out dir> inject [evaluates]    output pre-filled with a NaN sentinel every
//                                                                      frame; checks every frame was written
//   test_live_harness.exe <frames dir> <out dir> perf   [evaluates]    (clean/rr-f-s9) game-like host cost: the capture's
//                                                                      inputs are filled once, then only constants +
//                                                                      evaluate + submit with LV_INFLIGHT (default 2)
//                                                                      frames queued, no readback; per-call QPC times,
//                                                                      command-list call counts (vtable hook), thread CPU
//                                                                      time; LV_TL=1: a GPU timestamp after every dispatch
//                                                                      (tl_<n>.csv, kernel names from the PSOs' debug names)
// out\result.txt: one line per evaluate (rc, sentinel / finite checks, VRAM) and a summary line. Exit 0 = all ok.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <string>
#include <vector>

#include "wg_binding.inc"
#include "../include/rr_proxy_abi.h"

static_assert(WG_BINDING_HOOK == 1 && WG_BINDING_PROBE == 0, "the live harness drives the release hook ASI");

extern "C" {
WgResultCode slInit(const WgPreferences*, uint64_t);
WgResultCode slShutdown(void);
WgResultCode slIsFeatureSupported(uint32_t, const WgAdapterInfo*);
WgResultCode slSetD3DDevice(void*);
WgResultCode slGetFeatureFunction(uint32_t, const char*, void**);
WgResultCode slSetTag(const WgViewport*, const WgTag*, uint32_t, void*);
WgResultCode slSetConstants(const WgConstants*, const WgFrameToken*, const WgViewport*);
WgResultCode slGetNewFrameToken(WgFrameToken**, const uint32_t*);
WgResultCode slEvaluateFeature(uint32_t, const WgFrameToken*, const WgHeader* const*, uint32_t, void*);
}

static FILE* g_res = nullptr;
static void say(const char* fmt, ...) {
  char b[2048];
  va_list a;
  va_start(a, fmt);
  vsnprintf(b, sizeof b, fmt, a);
  va_end(a);
  fputs(b, stdout);
  fputs("\n", stdout);
  if (g_res) fputs(b, g_res), fputs("\n", g_res), fflush(g_res);
}
static void set_guid(WgGuid* gg, const char* name) {
  for (const WgGuidName& n : kWgGuidNames)
    if (strcmp(n.name, name) == 0) memcpy(gg->b, n.b, 16);
}
static std::vector<uint8_t> read_file(const std::string& p) {
  std::vector<uint8_t> v;
  FILE* f = fopen(p.c_str(), "rb");
  if (!f) return v;
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  v.resize((size_t)n);
  if (n > 0 && fread(v.data(), 1, (size_t)n, f) != (size_t)n) v.clear();
  fclose(f);
  return v;
}

// ---- prepared frame (.lvf, tools/live_harness.py) ----
struct LvInput {
  uint32_t type, format, width, height, row_pitch, tag_format, state, lifetime;
  uint32_t ext_left, ext_top, ext_w, ext_h;
  std::string raw;
};
struct LvFrame {
  uint32_t frame_index = 0, out_format = 0, out_w = 0, out_h = 0, out_state = 0, pref = 0, api_minor = 0;
  uint8_t constants[sizeof(WgConstants) - sizeof(WgHeader)];
  uint8_t options[sizeof(WgRrOptions) - sizeof(WgHeader)];
  uint32_t constants_version = 2, options_version = 3;
  std::vector<LvInput> inputs;
};
static bool load_lvf(const std::string& path, LvFrame* f) {
  std::vector<uint8_t> b = read_file(path);
  size_t at = 0;
  auto u32 = [&](uint32_t* v) { if (at + 4 > b.size()) return false; memcpy(v, &b[at], 4); at += 4; return true; };
  auto blob = [&](void* d, size_t n) { if (at + n > b.size()) return false; memcpy(d, &b[at], n); at += n; return true; };
  uint32_t magic = 0, n = 0, cs = 0, os = 0;
  if (!u32(&magic) || magic != 0x3146564Cu) return false;   // "LVF1"
  if (!u32(&f->frame_index) || !u32(&f->pref) || !u32(&f->api_minor) || !u32(&f->out_format) || !u32(&f->out_w) ||
      !u32(&f->out_h) || !u32(&f->out_state) || !u32(&f->constants_version) || !u32(&cs) ||
      cs != sizeof f->constants || !blob(f->constants, cs) || !u32(&f->options_version) || !u32(&os) ||
      os != sizeof f->options || !blob(f->options, os) || !u32(&n))
    return false;
  for (uint32_t i = 0; i < n; ++i) {
    LvInput in;
    uint32_t len = 0;
    if (!u32(&in.type) || !u32(&in.format) || !u32(&in.width) || !u32(&in.height) || !u32(&in.row_pitch) ||
        !u32(&in.tag_format) || !u32(&in.state) || !u32(&in.lifetime) || !u32(&in.ext_left) || !u32(&in.ext_top) ||
        !u32(&in.ext_w) || !u32(&in.ext_h) || !u32(&len) || at + len > b.size())
      return false;
    in.raw.assign((const char*)&b[at], len);
    at += len;
    f->inputs.push_back(in);
  }
  return true;
}

// ---- D3D12 host on the hardware adapter ----
struct Host {
  IDXGIAdapter3* adapter = nullptr;
  ID3D12Device* dev = nullptr;
  ID3D12CommandQueue* queue = nullptr;
  ID3D12CommandAllocator* alloc = nullptr;
  ID3D12GraphicsCommandList* cl = nullptr;
  ID3D12Fence* fence = nullptr;
  UINT64 fv = 0;
  HANDLE ev = nullptr;
  bool create() {
    IDXGIFactory4* f = nullptr;
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory4), (void**)&f))) return false;
    for (UINT i = 0;; ++i) {
      IDXGIAdapter1* a = nullptr;
      if (f->EnumAdapters1(i, &a) == DXGI_ERROR_NOT_FOUND) break;
      DXGI_ADAPTER_DESC1 d;
      a->GetDesc1(&d);
      if (!(d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) && d.VendorId == 0x1002 &&
          SUCCEEDED(D3D12CreateDevice(a, D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), (void**)&dev))) {
        a->QueryInterface(__uuidof(IDXGIAdapter3), (void**)&adapter);
        char nm[128];
        WideCharToMultiByte(CP_UTF8, 0, d.Description, -1, nm, sizeof nm, nullptr, nullptr);
        say("adapter: %s", nm);
        a->Release();
        break;
      }
      a->Release();
    }
    f->Release();
    if (!dev) return false;
    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    if (FAILED(dev->CreateCommandQueue(&qd, __uuidof(ID3D12CommandQueue), (void**)&queue))) return false;
    if (FAILED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, __uuidof(ID3D12CommandAllocator), (void**)&alloc)))
      return false;
    if (FAILED(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, nullptr,
                                      __uuidof(ID3D12GraphicsCommandList), (void**)&cl)))
      return false;
    cl->Close();
    if (FAILED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence), (void**)&fence))) return false;
    ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    return true;
  }
  void begin() { alloc->Reset(); cl->Reset(alloc, nullptr); }
  bool submit() {
    if (FAILED(cl->Close())) return false;
    ID3D12CommandList* l[1] = {cl};
    queue->ExecuteCommandLists(1, l);
    queue->Signal(fence, ++fv);
    fence->SetEventOnCompletion(fv, ev);
    WaitForSingleObject(ev, INFINITE);
    return dev->GetDeviceRemovedReason() == S_OK;
  }
  double vram_mb() {
    DXGI_QUERY_VIDEO_MEMORY_INFO i = {};
    if (!adapter || FAILED(adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &i))) return -1;
    return (double)i.CurrentUsage / (1024.0 * 1024.0);
  }
  ID3D12Resource* buffer(uint64_t size, D3D12_HEAP_TYPE t, D3D12_RESOURCE_STATES st) {
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = t;
    D3D12_RESOURCE_DESC d = {};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = size;
    d.Height = 1;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ID3D12Resource* r = nullptr;
    dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, st, nullptr, __uuidof(ID3D12Resource), (void**)&r);
    return r;
  }
  ID3D12Resource* tex(uint32_t w, uint32_t h, uint32_t fmt, D3D12_RESOURCE_FLAGS fl, D3D12_RESOURCE_STATES st) {
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC d = {};
    d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    d.Width = w;
    d.Height = h;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.Format = (DXGI_FORMAT)fmt;
    d.SampleDesc.Count = 1;
    d.Flags = fl;
    ID3D12Resource* r = nullptr;
    dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, st, nullptr, __uuidof(ID3D12Resource), (void**)&r);
    return r;
  }
};
static void barrier(ID3D12GraphicsCommandList* cl, ID3D12Resource* r, uint32_t a, uint32_t b) {
  if (a == b) return;
  D3D12_RESOURCE_BARRIER x = {};
  x.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  x.Transition.pResource = r;
  x.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
  x.Transition.StateBefore = (D3D12_RESOURCE_STATES)a;
  x.Transition.StateAfter = (D3D12_RESOURCE_STATES)b;
  cl->ResourceBarrier(1, &x);
}

// One game texture: created once in the native format, re-filled from the capture each frame (declared state kept).
struct GameTex {
  ID3D12Resource* r = nullptr;
  uint32_t w = 0, h = 0, fmt = 0, state = 0;
  WgResource res;
  WgTag tag;
};


// ---- perf scenario: command-list call counters and optional per-dispatch timestamps (vtable copy of the real list) ----
struct ClStats {
  uint64_t dispatch = 0, groups = 0, barrier_calls = 0, barrier_uav = 0, barrier_tr = 0, barrier_other = 0, set_pso = 0,
           root32 = 0, end_query = 0, copy = 0, clear = 0, heaps = 0;
};
static ClStats g_cs;
static uint64_t g_total_hook_calls = 0;
static bool g_tl = false;
static ID3D12QueryHeap* g_qh = nullptr;
static uint32_t g_qn = 0, g_qcap = 0;
struct TlRec { std::string name; uint32_t groups; uint32_t qidx; };
static std::vector<TlRec> g_tlrecs;
static std::string g_curname;
static void* g_vt[96];
static void* g_ovt[96];
typedef void(STDMETHODCALLTYPE* FnDispatch)(ID3D12GraphicsCommandList*, UINT, UINT, UINT);
typedef void(STDMETHODCALLTYPE* FnSetPso)(ID3D12GraphicsCommandList*, ID3D12PipelineState*);
typedef void(STDMETHODCALLTYPE* FnBarrier)(ID3D12GraphicsCommandList*, UINT, const D3D12_RESOURCE_BARRIER*);
typedef void(STDMETHODCALLTYPE* FnEndQuery)(ID3D12GraphicsCommandList*, ID3D12QueryHeap*, D3D12_QUERY_TYPE, UINT);
static void STDMETHODCALLTYPE h_dispatch(ID3D12GraphicsCommandList* l, UINT x, UINT y, UINT z) {
  ++g_cs.dispatch;
  g_cs.groups += (uint64_t)x * y * z;
  ((FnDispatch)g_ovt[14])(l, x, y, z);
  if (g_tl && g_qn < g_qcap) {
    ((FnEndQuery)g_ovt[53])(l, g_qh, D3D12_QUERY_TYPE_TIMESTAMP, g_qn);
    g_tlrecs.push_back({g_curname, x, g_qn});
    ++g_qn;
  }
}
static void STDMETHODCALLTYPE h_setpso(ID3D12GraphicsCommandList* l, ID3D12PipelineState* p) {
  ++g_cs.set_pso;
  if (g_tl && p) {
    static const GUID kName = {0x429b8c22, 0x9188, 0x4b0c, {0x87, 0x42, 0xac, 0xb0, 0xbf, 0x85, 0xc2, 0x00}};
    char nm[96] = {};
    UINT sz = sizeof nm - 1;
    if (SUCCEEDED(p->GetPrivateData(kName, &sz, nm))) g_curname = nm;
    else g_curname = "?";
  }
  ((FnSetPso)g_ovt[25])(l, p);
}
static void STDMETHODCALLTYPE h_barrier(ID3D12GraphicsCommandList* l, UINT n, const D3D12_RESOURCE_BARRIER* b) {
  ++g_cs.barrier_calls;
  ++g_total_hook_calls;
  for (UINT i = 0; i < n; ++i) {
    if (b[i].Type == D3D12_RESOURCE_BARRIER_TYPE_UAV) ++g_cs.barrier_uav;
    else if (b[i].Type == D3D12_RESOURCE_BARRIER_TYPE_TRANSITION) ++g_cs.barrier_tr;
    else ++g_cs.barrier_other;
  }
  ((FnBarrier)g_ovt[26])(l, n, b);
}
static void STDMETHODCALLTYPE h_endquery(ID3D12GraphicsCommandList* l, ID3D12QueryHeap* h, D3D12_QUERY_TYPE t, UINT i) {
  ++g_cs.end_query;
  ((FnEndQuery)g_ovt[53])(l, h, t, i);
}
typedef void(STDMETHODCALLTYPE* FnCopyBuf)(ID3D12GraphicsCommandList*, ID3D12Resource*, UINT64, ID3D12Resource*, UINT64, UINT64);
static void STDMETHODCALLTYPE h_copybuf(ID3D12GraphicsCommandList* l, ID3D12Resource* a, UINT64 b, ID3D12Resource* c, UINT64 d, UINT64 e) {
  ++g_cs.copy;
  ((FnCopyBuf)g_ovt[15])(l, a, b, c, d, e);
}
typedef void(STDMETHODCALLTYPE* FnRoot32)(ID3D12GraphicsCommandList*, UINT, UINT, const void*, UINT);
static void STDMETHODCALLTYPE h_root32(ID3D12GraphicsCommandList* l, UINT a, UINT b, const void* c, UINT d) {
  ++g_cs.root32;
  ((FnRoot32)g_ovt[35])(l, a, b, c, d);
}
typedef void(STDMETHODCALLTYPE* FnHeaps)(ID3D12GraphicsCommandList*, UINT, ID3D12DescriptorHeap* const*);
static void STDMETHODCALLTYPE h_heaps(ID3D12GraphicsCommandList* l, UINT n, ID3D12DescriptorHeap* const* h) {
  ++g_cs.heaps;
  ((FnHeaps)g_ovt[28])(l, n, h);
}
static void hook_list(ID3D12GraphicsCommandList* l) {
  void** vt = *(void***)l;
  if (vt == g_vt) return;
  if (!g_ovt[0]) memcpy(g_ovt, vt, sizeof(void*) * 70);
  memcpy(g_vt, vt, sizeof(void*) * 70);
  g_vt[14] = (void*)&h_dispatch;
  g_vt[15] = (void*)&h_copybuf;
  g_vt[25] = (void*)&h_setpso;
  g_vt[26] = (void*)&h_barrier;
  g_vt[28] = (void*)&h_heaps;
  g_vt[35] = (void*)&h_root32;
  g_vt[53] = (void*)&h_endquery;
  *(void***)l = g_vt;
}
static double qpc_ms() {
  static LARGE_INTEGER f = [] { LARGE_INTEGER x; QueryPerformanceFrequency(&x); return x; }();
  LARGE_INTEGER t;
  QueryPerformanceCounter(&t);
  return 1000.0 * (double)t.QuadPart / (double)f.QuadPart;
}
static double g_cyc_per_ms = 0;
static double thread_cpu_ms() {   // thread CPU time from cycle counts (GetThreadTimes ticks are 15.6 ms), calibrated by a 40 ms spin
  if (g_cyc_per_ms == 0) {
    ULONG64 c0, c1;
    QueryThreadCycleTime(GetCurrentThread(), &c0);
    const double t0 = qpc_ms();
    while (qpc_ms() - t0 < 40.0) {}
    QueryThreadCycleTime(GetCurrentThread(), &c1);
    g_cyc_per_ms = (double)(c1 - c0) / (qpc_ms() - t0);
  }
  ULONG64 c;
  QueryThreadCycleTime(GetCurrentThread(), &c);
  return (double)c / g_cyc_per_ms;
}
static double median(std::vector<double> v) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

static const D3D12_RESOURCE_STATES kCopyDst = D3D12_RESOURCE_STATE_COPY_DEST;

int main(int argc, char** argv) {
  if (argc < 4) return 2;
  const std::string frames_dir = std::string(argv[1]) + "\\", out_dir = std::string(argv[2]) + "\\";
  const std::string scen = argv[3];
  const uint32_t count_arg = argc > 4 ? (uint32_t)atoi(argv[4]) : 0;
  CreateDirectoryA(out_dir.c_str(), nullptr);
  g_res = fopen((out_dir + "result.txt").c_str(), "wb");
  // frames
  std::vector<LvFrame> frames;
  for (int i = 0;; ++i) {
    char p[32];
    snprintf(p, sizeof p, "frame_%03d.lvf", i);
    LvFrame f;
    if (!load_lvf(frames_dir + p, &f)) break;
    frames.push_back(std::move(f));
  }
  if (frames.empty()) { say("FAIL no frames in %s", frames_dir.c_str()); return 1; }
  say("frames: %zu, scenario %s", frames.size(), scen.c_str());

  Host g;
  if (!g.create()) { say("FAIL no hardware D3D12 device"); return 1; }
  const double vram0 = g.vram_mb();

  // the stand-in interface is loaded (static import); the device / list are handed natively (native-interface
  // returns the registered object itself)
  std::wstring iname;
  for (const char* q = kWgInterposerFile; *q; ++q) iname.push_back((wchar_t)*q);
  HMODULE iface = GetModuleHandleW(iname.c_str());
  auto wrap = iface ? reinterpret_cast<void (*)(void*, void*)>(GetProcAddress(iface, "wgtest_fo_wrap")) : nullptr;
  if (!wrap) { say("FAIL stand-in interface not loaded"); return 1; }
  wrap(g.dev, g.dev);
  wrap(g.cl, g.cl);
  wchar_t exe[MAX_PATH];
  GetModuleFileNameW(nullptr, exe, MAX_PATH);
  std::wstring stage(exe);
  stage = stage.substr(0, stage.find_last_of(L"\\/") + 1);
  std::wstring hname;
  for (const char* q = kWgHookFile; *q; ++q) hname.push_back((wchar_t)*q);
  HMODULE asi = LoadLibraryW((stage + hname).c_str());
  auto InitializeASI = asi ? reinterpret_cast<void (*)(void)>(GetProcAddress(asi, "InitializeASI")) : nullptr;
  if (!InitializeASI) { say("FAIL ASI not loaded"); return 1; }
  InitializeASI();

  static const uint32_t feats[2] = {1001, 0};
  WgPreferences pr;
  memset(&pr, 0, sizeof pr);
  pr.h.version = 1;
  pr.features = feats;
  pr.feature_count = 2;
  pr.flags = frames[0].pref;
  pr.render_api = 1;
  const uint64_t ver = (2ull << 48) | ((uint64_t)WG_BINDING_API_MINOR << 32) | (1ull << 16) | WG_BINDING_VERSION_MAGIC;
  int bad = 0;
  if (slInit(&pr, ver) != 0) { say("FAIL init"); return 1; }
  if (slSetD3DDevice(g.dev) != 0) { say("FAIL set-device"); return 1; }
  const WgResultCode sup = slIsFeatureSupported(1001, nullptr);
  say("is-supported(RR) = %d", sup);
  if (sup != 0) { say("FAIL RR not supported (runtime not ready?)"); return 1; }
  void *fo = nullptr, *fp = nullptr;
  slGetFeatureFunction(1001, kWgFnOptimal, &fo);
  slGetFeatureFunction(1001, kWgFnOptions, &fp);
  auto rr_options = reinterpret_cast<WgResultCode (*)(const WgViewport*, const WgRrOptions*)>(fp);
  if (!rr_options) { say("FAIL no options function"); return 1; }

  static WgViewport vp;
  memset(&vp, 0, sizeof vp);
  memcpy(vp.h.type.b, kWgGuid_viewport, 16);
  vp.h.version = 1;

  // game textures (by buffer type), the output textures (2560 x 1440 and a 1920 x 1080 one for the size change)
  std::vector<GameTex> texs(frames[0].inputs.size());
  uint64_t up_size = 0;
  for (size_t i = 0; i < texs.size(); ++i) {
    const LvInput& in = frames[0].inputs[i];
    GameTex& t = texs[i];
    t.w = in.width, t.h = in.height, t.fmt = in.format, t.state = in.state;
    D3D12_RESOURCE_FLAGS fl = D3D12_RESOURCE_FLAG_NONE;
    if (in.type == 0) fl = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;          // hardware depth: R32_TYPELESS DS
    else if (in.state & D3D12_RESOURCE_STATE_UNORDERED_ACCESS) fl = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    else fl = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;                       // motion: a render target
    t.r = g.tex(t.w, t.h, t.fmt, fl, kCopyDst);
    if (!t.r) { say("FAIL create input type %u format %u", in.type, in.format); return 1; }
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp0;
    UINT64 total = 0;
    D3D12_RESOURCE_DESC d = t.r->GetDesc();
    g.dev->GetCopyableFootprints(&d, 0, 1, 0, &fp0, nullptr, nullptr, &total);
    up_size += (total + 511) & ~511ull;
  }
  struct OutTex { ID3D12Resource* r; uint32_t w, h; WgResource res; WgTag tag; };
  OutTex outs[2] = {};
  const uint32_t ow0 = frames[0].out_w, oh0 = frames[0].out_h;
  outs[0].w = ow0, outs[0].h = oh0;
  outs[1].w = ow0 * 3 / 4, outs[1].h = oh0 * 3 / 4;
  for (OutTex& o : outs) {
    o.r = g.tex(o.w, o.h, frames[0].out_format, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                (D3D12_RESOURCE_STATES)frames[0].out_state);
    if (!o.r) { say("FAIL create output"); return 1; }
  }
  const uint64_t out_bytes = (uint64_t)ow0 * oh0 * 8;   // RGBA16F, row pitch ow0 * 8 (multiple of 256 for 2560)
  ID3D12Resource* upload = g.buffer(up_size + out_bytes + 4096, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
  ID3D12Resource* readback = g.buffer(out_bytes + 4096, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
  if (!upload || !readback) { say("FAIL staging buffers"); return 1; }
  uint8_t* up_ptr = nullptr;
  upload->Map(0, nullptr, (void**)&up_ptr);
  // sentinel: an f16 NaN pattern in every output texel (inject scenario)
  {
    uint16_t* s = (uint16_t*)(up_ptr + up_size);
    for (uint64_t i = 0; i < out_bytes / 2; ++i) s[i] = 0x7E01;
  }
  // first upload state: COPY_DEST -> declared
  g.begin();
  for (GameTex& t : texs) barrier(g.cl, t.r, kCopyDst, t.state);
  g.submit();

  uint32_t total = count_arg;
  if (scen == "match") total = (uint32_t)frames.size() * (count_arg ? count_arg : 1);
  else if (scen == "soak") total = count_arg ? count_arg : 600;
  else if (scen == "inject") total = count_arg ? count_arg : 24;
  else if (scen == "perf") total = count_arg ? count_arg : 300;
  else { say("FAIL unknown scenario"); return 1; }

  const bool perf = scen == "perf";
  const uint32_t nf = (uint32_t)frames.size();
  UINT inflight = 2;
  { char b[16]; if (GetEnvironmentVariableA("LV_INFLIGHT", b, sizeof b)) inflight = (UINT)std::max(1, std::min(4, atoi(b))); }
  const bool want_tl = perf && GetEnvironmentVariableA("LV_TL", nullptr, 0) != 0;
  static const uint32_t kRing = 6;
  ID3D12CommandAllocator* ralloc[kRing] = {};
  ID3D12GraphicsCommandList* rcl[kRing] = {};
  ID3D12Resource* tl_rb = nullptr;
  UINT64 ts_freq = 0;
  std::vector<double> v_tag, v_const, v_eval, v_sub, v_wait, v_rec, v_cpu;
  std::vector<ClStats> v_cs;
  std::vector<std::vector<TlRec>> tl_saved;
  std::vector<uint32_t> tl_n, tl_eval;
  double t_first = 0, t_last = 0;
  uint32_t n_timed = 0;
  if (perf) {
    g.queue->GetTimestampFrequency(&ts_freq);
    thread_cpu_ms();   // calibrate
    for (uint32_t k = 0; k < kRing; ++k) {
      g.dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, __uuidof(ID3D12CommandAllocator), (void**)&ralloc[k]);
      g.dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, ralloc[k], nullptr, __uuidof(ID3D12GraphicsCommandList), (void**)&rcl[k]);
      rcl[k]->Close();
      hook_list(rcl[k]);
      if (k == 0) {
        ID3D12GraphicsCommandList *q1 = nullptr;
        ID3D12GraphicsCommandList2 *q2 = nullptr;
        rcl[k]->QueryInterface(__uuidof(ID3D12GraphicsCommandList), (void**)&q1);
        rcl[k]->QueryInterface(__uuidof(ID3D12GraphicsCommandList2), (void**)&q2);
        say("perf: list %p vt %p, QI list %p vt %p, QI list2 %p vt %p (hook vt %p)", (void*)rcl[k], *(void**)rcl[k], (void*)q1, q1 ? *(void**)q1 : nullptr, (void*)q2, q2 ? *(void**)q2 : nullptr, (void*)g_vt);
      }
      wrap(rcl[k], rcl[k]);
    }
    hook_list(g.cl);
    say("perf: after hooks: g.cl vt %p, rcl[3] vt %p, g_vt %p, g_vt[26] %p, h_barrier %p, g_ovt[26] %p", *(void**)g.cl, *(void**)rcl[3], (void*)g_vt, g_vt[26], (void*)&h_barrier, g_ovt[26]);
    if (want_tl) {
      g_qcap = 4096;
      D3D12_QUERY_HEAP_DESC qd = {};
      qd.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
      qd.Count = g_qcap;
      g.dev->CreateQueryHeap(&qd, __uuidof(ID3D12QueryHeap), (void**)&g_qh);
      tl_rb = g.buffer(16ull * g_qcap * 8, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    }
    say("perf: inflight %u, ring %u, timeline %d, warm-up fills %u frames", inflight, kRing, want_tl ? 1 : 0, nf);
  }
  auto fence_wait = [&](UINT64 v) {
    if (g.fence->GetCompletedValue() >= v) return;
    g.fence->SetEventOnCompletion(v, g.ev);
    WaitForSingleObject(g.ev, INFINITE);
  };

  double vram_first = -1, vram_peak = 0;
  std::vector<double> vram_at;
  int ok_rc = 0, sentinel_left = 0, nonfinite_frames = 0;
  for (uint32_t e = 0; e < total; ++e) {
    const LvFrame& fr = frames[e % frames.size()];
    // soak schedule: evaluates 200..299 render 1280 x 720 (render-size change), 400..499 output 3/4 size with render
    // 1280 x 720 (new instance: output size change), the capture's own reset every 8th frame
    uint32_t rw = 0, rh = 0, oi = 0;
    if (scen == "soak" && e >= 200 && e < 300) rw = 1280, rh = 720;
    if (scen == "soak" && e >= 400 && e < 500) rw = 1280, rh = 720, oi = 1;
    OutTex& o = outs[oi];
    ID3D12GraphicsCommandList* cl = g.cl;
    const bool do_fill = !perf || e < nf;
    double t_wait0 = 0, t_wait1 = 0;
    if (perf) {
      t_wait0 = qpc_ms();
      if (e < nf) fence_wait(g.fv);                                  // fills share one upload buffer: previous frame done
      else if (e >= inflight) fence_wait(g.fv - (inflight - 1));     // the game's pacing: at most `inflight` frames queued
      t_wait1 = qpc_ms();
      const uint32_t k = e % kRing;
      ralloc[k]->Reset();
      rcl[k]->Reset(ralloc[k], nullptr);
      cl = rcl[k];
      hook_list(cl);   // Reset restores the object's own vtable
    } else {
      g.begin();
    }
    // re-fill the inputs from this frame (declared -> COPY_DEST -> declared, as the game's own passes would)
    uint64_t at = 0;
    for (size_t i = 0; i < texs.size(); ++i) {
      GameTex& t = texs[i];
      const LvInput& in = fr.inputs[i];
      if (do_fill) {
      D3D12_RESOURCE_DESC d = t.r->GetDesc();
      D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp0;
      UINT64 tot = 0;
      UINT rows = 0;
      UINT64 rowb = 0;
      g.dev->GetCopyableFootprints(&d, 0, 1, at, &fp0, &rows, &rowb, &tot);
      std::vector<uint8_t> raw = read_file(in.raw);
      if (raw.size() < 64 + (size_t)in.row_pitch * in.height) { say("FAIL raw %s", in.raw.c_str()); return 1; }
      for (UINT y = 0; y < rows; ++y)
        memcpy(up_ptr + fp0.Offset + (uint64_t)y * fp0.Footprint.RowPitch, raw.data() + 64 + (size_t)y * in.row_pitch,
               (size_t)std::min<UINT64>(rowb, in.row_pitch));
      barrier(cl, t.r, t.state, kCopyDst);
      D3D12_TEXTURE_COPY_LOCATION src = {}, dst = {};
      src.pResource = upload;
      src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
      src.PlacedFootprint = fp0;
      dst.pResource = t.r;
      dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
      cl->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
      barrier(cl, t.r, kCopyDst, t.state);
      at = (fp0.Offset + tot + 511) & ~511ull;
      }
      // the tag, as game A fills it: resource size 0 x 0, the native format, the declared state, lifetime 1
      memset(&t.res, 0, sizeof t.res);
      t.res.h.version = 1;
      t.res.kind = WG_RES_TEX2D;
      t.res.native = t.r;
      t.res.state = in.state;
      t.res.width = 0;
      t.res.height = 0;
      t.res.native_format = in.tag_format;
      t.res.mip_levels = 1;
      t.res.array_layers = 1;
      memset(&t.tag, 0, sizeof t.tag);
      memcpy(t.tag.h.type.b, kWgGuid_tag, 16);
      t.tag.h.version = 1;
      t.tag.resource = &t.res;
      t.tag.buffer_type = in.type;
      t.tag.lifetime = in.lifetime;
      t.tag.extent.left = in.ext_left;
      t.tag.extent.top = in.ext_top;
      t.tag.extent.width = rw ? rw : in.ext_w;
      t.tag.extent.height = rh ? rh : in.ext_h;
    }
    if (scen == "inject") {   // the output pre-filled with the sentinel
      D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp0;
      D3D12_RESOURCE_DESC d = o.r->GetDesc();
      g.dev->GetCopyableFootprints(&d, 0, 1, up_size, &fp0, nullptr, nullptr, nullptr);
      barrier(cl, o.r, fr.out_state, kCopyDst);
      D3D12_TEXTURE_COPY_LOCATION src = {}, dst = {};
      src.pResource = upload;
      src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
      src.PlacedFootprint = fp0;
      dst.pResource = o.r;
      dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
      cl->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
      barrier(cl, o.r, kCopyDst, fr.out_state);
    }
    memset(&o.res, 0, sizeof o.res);
    o.res.h.version = 1;
    o.res.kind = WG_RES_TEX2D;
    o.res.native = o.r;
    o.res.state = fr.out_state;
    o.res.native_format = fr.out_format;
    o.res.mip_levels = 1;
    o.res.array_layers = 1;
    memset(&o.tag, 0, sizeof o.tag);
    memcpy(o.tag.h.type.b, kWgGuid_tag, 16);
    o.tag.h.version = 1;
    o.tag.resource = &o.res;
    o.tag.buffer_type = WG_BT_COLOUR_OUT;
    o.tag.lifetime = WG_LIFE_UNTIL_PRESENT;
    o.tag.extent.width = o.w;
    o.tag.extent.height = o.h;

    WgFrameToken* tok = nullptr;
    const uint32_t idx = fr.frame_index + e;   // a new frame index per evaluate
    WgResultCode rc_tok = slGetNewFrameToken(&tok, &idx);
    WgRrOptions op;
    memset(&op, 0, sizeof op);
    memcpy((uint8_t*)&op + sizeof(WgHeader), fr.options, sizeof fr.options);
    op.h.version = fr.options_version;
    set_guid(&op.h.type, "rr_options");
    if (oi) op.output_width = o.w, op.output_height = o.h;
    WgResultCode rc_opt = rr_options(&vp, &op);
    std::vector<WgTag> tags;
    for (GameTex& t : texs) tags.push_back(t.tag);
    tags.push_back(o.tag);
    const bool tl_frame = want_tl && e >= nf + 40 && e < nf + 56;
    const double t_rec0 = qpc_ms();
    const double cpu0 = perf ? thread_cpu_ms() : 0;
    if (perf) {
      g_cs = ClStats();
      g_tl = tl_frame;
      g_qn = 0;
      g_tlrecs.clear();
      g_curname.clear();
      if (tl_frame) {
        ((FnEndQuery)g_ovt[53])(cl, g_qh, D3D12_QUERY_TYPE_TIMESTAMP, g_qn++);   // start stamp
      }
    }
    WgResultCode rc_tag = slSetTag(&vp, tags.data(), (uint32_t)tags.size(), cl);
    const double t_tag = qpc_ms();
    WgConstants c;
    memset(&c, 0, sizeof c);
    memcpy((uint8_t*)&c + sizeof(WgHeader), fr.constants, sizeof fr.constants);
    c.h.version = fr.constants_version;
    set_guid(&c.h.type, "constants");
    if (scen == "soak" && (e == 200 || e == 300 || e == 400 || e == 500)) c.reset = 1;   // resets around the changes
    WgResultCode rc_c = slSetConstants(&c, tok, &vp);
    const double t_const = qpc_ms();
    const WgHeader* inl[1] = {&vp.h};
    const WgResultCode rc = slEvaluateFeature(1001, tok, inl, 1, cl);
    const double t_eval = qpc_ms();
    if (perf) {
      const ClStats cs = g_cs;
      if (tl_frame) {
        ((FnEndQuery)g_ovt[53])(cl, g_qh, D3D12_QUERY_TYPE_TIMESTAMP, g_qn++);   // end stamp
        cl->ResolveQueryData(g_qh, D3D12_QUERY_TYPE_TIMESTAMP, 0, g_qn, tl_rb, (UINT64)(e - (nf + 40)) * g_qcap * 8);
        tl_saved.push_back(g_tlrecs);
        tl_n.push_back(g_qn);
        tl_eval.push_back(e);
      }
      g_tl = false;
      const double cpu1 = thread_cpu_ms();
      cl->Close();
      ID3D12CommandList* ll[1] = {cl};
      g.queue->ExecuteCommandLists(1, ll);
      g.queue->Signal(g.fence, ++g.fv);
      const double t_sub = qpc_ms();
      ok_rc += rc == 0;
      if (e == nf + 20) t_first = qpc_ms();
      if (e >= nf + 20) {
        t_last = qpc_ms();
        ++n_timed;
        v_wait.push_back(t_wait1 - t_wait0);
        v_rec.push_back(t_rec0 - t_wait1);
        v_tag.push_back(t_tag - t_rec0);
        v_const.push_back(t_const - t_tag);
        v_eval.push_back(t_eval - t_const);
        v_sub.push_back(t_sub - t_eval);
        v_cpu.push_back(cpu1 - cpu0);
        v_cs.push_back(cs);
      }
      if (e < 12 || (e + 1) % 100 == 0)
        say("perf eval %u: rc %d wait %.3f ms, tag %.3f const %.3f eval %.3f submit %.3f ms, dispatch %llu barriers %llu (uav %llu tr %llu)",
            e + 1, rc, t_wait1 - t_wait0, t_tag - t_rec0, t_const - t_tag, t_eval - t_const, t_sub - t_eval,
            (unsigned long long)cs.dispatch, (unsigned long long)cs.barrier_calls, (unsigned long long)cs.barrier_uav,
            (unsigned long long)cs.barrier_tr);
      continue;
    }
    // read the output back
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT rfp;
    D3D12_RESOURCE_DESC od = o.r->GetDesc();
    UINT64 rrowb = 0;
    g.dev->GetCopyableFootprints(&od, 0, 1, 0, &rfp, nullptr, &rrowb, nullptr);
    barrier(cl, o.r, fr.out_state, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION src = {}, dst = {};
    src.pResource = o.r;
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst.pResource = readback;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint = rfp;
    cl->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    barrier(cl, o.r, D3D12_RESOURCE_STATE_COPY_SOURCE, fr.out_state);
    const bool sub = g.submit();
    uint8_t* rb = nullptr;
    readback->Map(0, nullptr, (void**)&rb);
    uint64_t sentinel = 0, nonfinite = 0;
    for (uint32_t y = 0; y < o.h; ++y) {
      const uint16_t* row = (const uint16_t*)(rb + (uint64_t)y * rfp.Footprint.RowPitch);
      for (uint32_t x = 0; x < o.w * 4; ++x) {
        sentinel += row[x] == 0x7E01;
        nonfinite += (row[x] & 0x7C00) == 0x7C00;
      }
    }
    if (scen == "match") {
      char p[64];
      snprintf(p, sizeof p, "frame_%03u.out", e);
      FILE* f = fopen((out_dir + p).c_str(), "wb");
      for (uint32_t y = 0; y < o.h; ++y) fwrite(rb + (uint64_t)y * rfp.Footprint.RowPitch, 1, (size_t)o.w * 8, f);
      fclose(f);
    }
    readback->Unmap(0, nullptr);
    const double vr = g.vram_mb();
    if (e == 0) vram_first = vr;
    vram_peak = std::max(vram_peak, vr);
    if ((e + 1) % 50 == 0) vram_at.push_back(vr);
    ok_rc += rc == 0;
    sentinel_left += scen == "inject" && sentinel != 0;
    nonfinite_frames += nonfinite != 0;
    if (e < 24 || rc != 0 || !sub || (e + 1) % 50 == 0 || (scen == "inject" && sentinel))
      say("eval %u: rc %d (token %d options %d tag %d constants %d) submit %d out %ux%u sentinel %llu nonfinite %llu "
          "vram %.1f MB", e + 1, rc, rc_tok, rc_opt, rc_tag, rc_c, sub ? 1 : 0, o.w, o.h,
          (unsigned long long)sentinel, (unsigned long long)nonfinite, vr);
    if (!sub) { say("FAIL device removed after evaluate %u", e + 1); bad++; break; }
  }
  if (perf) {
    fence_wait(g.fv);
    for (size_t fi = 0; fi < tl_saved.size(); ++fi) {
      uint64_t* tsd = nullptr;
      D3D12_RANGE rr{(SIZE_T)(fi * g_qcap * 8), (SIZE_T)(fi * g_qcap * 8 + tl_n[fi] * 8)};
      tl_rb->Map(0, &rr, (void**)&tsd);
      tsd += fi * g_qcap;
      char fnm[64];
      snprintf(fnm, sizeof fnm, "tl_%u.csv", tl_eval[fi]);
      FILE* f = fopen((out_dir + fnm).c_str(), "wb");
      fprintf(f, "idx,name,groups,t_us_from_start,delta_us\n");
      uint64_t prev = tsd[0];
      for (size_t r = 0; r < tl_saved[fi].size(); ++r) {
        const uint64_t t = tsd[tl_saved[fi][r].qidx];
        fprintf(f, "%zu,%s,%u,%.2f,%.2f\n", r, tl_saved[fi][r].name.c_str(), tl_saved[fi][r].groups,
                1e6 * (double)(t - tsd[0]) / (double)ts_freq, 1e6 * (double)(t - prev) / (double)ts_freq);
        prev = t;
      }
      fprintf(f, "end,,,%.2f,%.2f\n", 1e6 * (double)(tsd[tl_n[fi] - 1] - tsd[0]) / (double)ts_freq,
              1e6 * (double)(tsd[tl_n[fi] - 1] - prev) / (double)ts_freq);
      fclose(f);
      D3D12_RANGE nr{0, 0};
      tl_rb->Unmap(0, &nr);
    }
    const double span = t_last - t_first;
    say("PERF frames timed %u: period %.3f ms/frame (wall, inflight %u)", n_timed, n_timed > 1 ? span / (n_timed - 1) : 0.0, inflight);
    say("PERF median ms: pacing-wait %.3f  between-calls %.3f  setTag %.3f  setConstants %.3f  evaluate %.3f  submit %.3f  | thread CPU per frame (record part) %.3f",
        median(v_wait), median(v_rec), median(v_tag), median(v_const), median(v_eval), median(v_sub), median(v_cpu));
    auto mx = [](std::vector<double> v) { return v.empty() ? 0.0 : *std::max_element(v.begin(), v.end()); };
    say("PERF max ms: setTag %.3f setConstants %.3f evaluate %.3f submit %.3f", mx(v_tag), mx(v_const), mx(v_eval), mx(v_sub));
    if (!v_cs.empty()) {
      const ClStats& c0 = v_cs[v_cs.size() / 2];
      say("PERF per evaluate (list calls): dispatch %llu (groups %llu) barrier calls %llu (uav %llu transition %llu other %llu) set-pso %llu root-const %llu set-heaps %llu copy %llu end-query %llu",
          (unsigned long long)c0.dispatch, (unsigned long long)c0.groups, (unsigned long long)c0.barrier_calls,
          (unsigned long long)c0.barrier_uav, (unsigned long long)c0.barrier_tr, (unsigned long long)c0.barrier_other,
          (unsigned long long)c0.set_pso, (unsigned long long)c0.root32, (unsigned long long)c0.heaps,
          (unsigned long long)c0.copy, (unsigned long long)c0.end_query);
    }
    say("PERF hook calls seen in total (barriers): %llu", (unsigned long long)g_total_hook_calls);
    say("PERF submissions: 1 ExecuteCommandLists + 1 Signal per frame by the game; the runtime itself submits none while evaluating");
  }
  slShutdown();
  std::string vs;
  for (double v : vram_at) { char b[32]; snprintf(b, sizeof b, " %.0f", v); vs += b; }
  say("summary: scenario %s evaluates %u rc0 %d sentinel_frames %d nonfinite_frames %d vram MB start %.0f first %.0f "
      "peak %.0f per50:%s", scen.c_str(), total, ok_rc, sentinel_left, nonfinite_frames, vram0, vram_first, vram_peak,
      vs.c_str());
  if (ok_rc != (int)total) bad++;
  if (scen == "inject" && sentinel_left) bad++;
  say(bad ? "RESULT FAIL" : "RESULT OK");
  if (g_res) fclose(g_res);
  return bad ? 1 : 0;
}
