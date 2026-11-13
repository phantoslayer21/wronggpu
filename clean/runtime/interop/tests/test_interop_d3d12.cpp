// SPDX-License-Identifier: MIT
// Offline test of the runtime DLL's D3D12 interop (spec 10 sections 1, 6, 8) through its exported backend
// (wronggpu_upscaler.h, interface version 2), on any D3D12 device (Wine's vkd3d with Mesa lavapipe, WARP, a GPU).
// The network is replaced by the DLL's "d3d12-test" executor (nearest-neighbour upscale of one converted input), and
// the front end's split service by a minimal one in this file (single-threaded, so the game's order is trivial).
//
//   clang++ -std=c++17 -O2 test_interop_d3d12.cpp -o test_interop_d3d12.exe   (Windows; MinGW: tests/run-tests.sh)
//   test_interop_d3d12.exe <path to wronggpu_runtime.dll> [case]                (WG_D3D12_DEBUG=1: debug layer)
//   test_interop_d3d12.exe <path to wronggpu_runtime.dll> hip ["key=value;..." backend options]
//   test_interop_d3d12.exe <path to wronggpu_runtime.dll> hipsame "<options A>" "<options B>" [frames]
//     (the same frames through both option sets: identical outputs; WG_TEST_MOTION_OUTRES=1: motion at output size)
//   test_interop_d3d12.exe <runtime> hipbench "<explicit synthetic model paths and options>" [frames]
//     Full D3D12/HIP/D3D12 span via timing_log=1, auto-exposure, static colour and spatial motion field, and cyclic jitter.
//     Discard the first 300-frame log window for warm-up. Prints a final output checksum and non-finite count.
//   test_interop_d3d12.exe <runtime> dx12same "<common options>" [frames] [outsrv]
//     executor=hip against executor=dx12 (the network on this test's D3D12 device, no HIP) through the full interop:
//     the output textures byte for byte; the common options name the synthetic model (see case_dx12_same).
//   hipbench / dx12same environment: WG_TEST_RENDER_WIDTH / _HEIGHT, WG_TEST_OUTPUT_WIDTH / _HEIGHT (sizes),
//     WG_TEST_TEX_PAD=n (textures n texels wider than the render size), WG_TEST_MOTION_OUTRES=1 (motion at output size,
//     WG_CREATE_MOTION_OUTPUT_RES), WG_TEST_MOTIONFMT=rg16f|rg32f, WG_TEST_PI (pre-exposure).
//   test_interop_d3d12.exe <runtime> asyncsim "<options>" <serial|ideal|pipe|inline|game|net> [frames] [pre] [post] [iters]
//     A simulated game frame on the DIRECT queue and the network (executor=dx12) in its list or on a COMPUTE queue.
//   test_interop_d3d12.exe <runtime> hipswitch "<Model M options>" "<Model K options>"
//     Creates, dispatches, destroys M -> K -> M in one process and verifies the HIP DLL stays loaded between them.
//   test_interop_d3d12.exe <runtime> hipleak "<options A>" ["<options B>" | -] [cycles] [frames] [MiB per cycle] [hold] [thread]
//     Context re-creation as at a game's upscale-mode change: `cycles` (default 12; a multiple of 6) times create ->
//     `frames` (30) dispatches -> destroy at 1280x720, 1114x626, 960x540 -> 1920x1080, options A and B alternating
//     (e.g. "" and "model=K"); fails when the process's GPU memory grows by more than the tolerance per cycle.
//
// Cases: inline, pipelined, depth, motion, motion_outres, exposure, not_ready, pipelined_not_ready, nohip,
// noservices, timing (timing_log=1 summary), ready_flag (ready_wait=memory flag pass with timing marks), arena, arena_minsize (wrapper refusing shared objects below 100 MB), arena_keep (arena pool), wrapped, wrapped_noopen, wrapped_own
// (device wrapper whose CreateSharedHandle fails); default all.
// "hip" (only when named; Windows with an RX 9000 GPU, the kernels next to the DLL and an installed model): the real
// network through the same split, at 1707x960 -> 2560x1440, inline then pipelined; checks finite output, that the
// pipelined output of frame t equals the inline output of frame t - 1 on the same inputs, and prints the CPU time
// of each dispatch call.
// WrongGPU's own code.
#if defined(__MINGW32__) && !defined(WIDL_EXPLICIT_AGGREGATE_RETURNS)
#define WIDL_EXPLICIT_AGGREGATE_RETURNS  // MinGW headers: struct returns through an out pointer (Microsoft ABI)
#endif
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <dxgi1_4.h>
#include <pdh.h>
#include <psapi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "../../../fsr/include/wronggpu_upscaler.h"

namespace {

int g_fail = 0;
#define CHECK(c)                                                    \
  do {                                                              \
    if (!(c)) {                                                     \
      std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);    \
      ++g_fail;                                                     \
    }                                                               \
  } while (0)

std::vector<std::string> g_log;
void log_fn(void*, int level, const char* m) {
  std::printf("  [runtime %d] %s\n", level, m);
  g_log.push_back(m);
}

// ---- fp16 ----
uint16_t to_half(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 16) & 0x8000;
  int e = (int)((x >> 23) & 0xff) - 127 + 15;
  uint32_t m = x & 0x7fffff;
  if (e <= 0) return (uint16_t)sign;  // test values are normal
  if (e >= 31) return (uint16_t)(sign | 0x7c00);
  uint32_t h = sign | (e << 10) | (m >> 13);
  if ((m >> 12) & 1) h++;  // round half up (test values are exact anyway)
  return (uint16_t)h;
}
float from_half(uint16_t h) {
  const uint32_t sign = (h & 0x8000u) << 16;
  const int e = (h >> 10) & 0x1f;
  const uint32_t m = h & 0x3ff;
  uint32_t x;
  if (e == 0) {
    float f = std::ldexp((float)m, -24);
    return sign ? -f : f;
  }
  if (e == 31) x = sign | 0x7f800000 | (m << 13);
  else x = sign | ((uint32_t)(e - 15 + 127) << 23) | (m << 13);
  float f;
  std::memcpy(&f, &x, 4);
  return f;
}
float q16(float f) { return from_half(to_half(f)); }

// ---- D3D12 helpers ----
// WG_TEST_LIST=compute: the game list (queue, allocators, list) is a COMPUTE list, as in games that run the upscaler on
// async compute; the input textures then sit in NON_PIXEL_SHADER_RESOURCE (PIXEL_SHADER_RESOURCE is not a compute state).
static bool test_compute_list() {
  const char* v = std::getenv("WG_TEST_LIST");
  return v && std::strcmp(v, "compute") == 0;
}
static D3D12_COMMAND_LIST_TYPE test_list_type() {
  return test_compute_list() ? D3D12_COMMAND_LIST_TYPE_COMPUTE : D3D12_COMMAND_LIST_TYPE_DIRECT;
}
struct Gpu {
  ID3D12Device* dev = nullptr;
  ID3D12CommandQueue* q = nullptr;
  ID3D12Fence* fence = nullptr;
  uint64_t fv = 0;
  HANDLE ev = nullptr;
  ID3D12CommandAllocator* alloc[2] = {};
  ID3D12GraphicsCommandList* list = nullptr;
  // game-like pacing (soak "pipe"): frames in flight on a ring of allocator pairs, no wait per frame
  ID3D12CommandAllocator* ring[8] = {};
  uint64_t ringv[4] = {};
  ID3D12CommandAllocator* split_alloc = nullptr;  // the allocator the split re-opens the list with (ring mode)
  bool begin_ring(int k) {
    if (!ring[0])
      for (auto& a : ring)
        dev->CreateCommandAllocator(test_list_type(), __uuidof(ID3D12CommandAllocator), (void**)&a);
    if (fence->GetCompletedValue() < ringv[k]) {
      fence->SetEventOnCompletion(ringv[k], ev);
      WaitForSingleObject(ev, 20000);
      if (fence->GetCompletedValue() < ringv[k]) return false;  // the frame that used these allocators never finished
    }
    ring[2 * k]->Reset();
    list->Reset(ring[2 * k], nullptr);
    split_alloc = ring[2 * k + 1];
    return true;
  }

  bool init() {
    typedef HRESULT(WINAPI * PfnCreate)(IUnknown*, D3D_FEATURE_LEVEL, REFIID, void**);
    HMODULE d3d12 = LoadLibraryW(L"d3d12.dll");
    auto create = d3d12 ? (PfnCreate)(void*)GetProcAddress(d3d12, "D3D12CreateDevice") : nullptr;
    // WG_D3D12_DEBUG=1: the D3D12 debug layer (Windows with the Graphics Tools feature)
    char dbg[8] = {0};
    if (d3d12 && GetEnvironmentVariableA("WG_D3D12_DEBUG", dbg, 8) && dbg[0] == '1') {
      typedef HRESULT(WINAPI * PfnDebug)(REFIID, void**);
      auto get = (PfnDebug)(void*)GetProcAddress(d3d12, "D3D12GetDebugInterface");
      ID3D12Debug* d = nullptr;
      if (get && SUCCEEDED(get(__uuidof(ID3D12Debug), (void**)&d)) && d) {
        d->EnableDebugLayer();
        d->Release();
        std::printf("D3D12 debug layer on\n");
      }
    }
    if (!create || FAILED(create(nullptr, D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), (void**)&dev))) return false;
    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = test_list_type();
    dev->CreateCommandQueue(&qd, __uuidof(ID3D12CommandQueue), (void**)&q);
    dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence), (void**)&fence);
    ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    for (auto& a : alloc)
      dev->CreateCommandAllocator(test_list_type(), __uuidof(ID3D12CommandAllocator), (void**)&a);
    dev->CreateCommandList(0, test_list_type(), alloc[0], nullptr, __uuidof(ID3D12GraphicsCommandList),
                           (void**)&list);
    list->Close();
    return q && fence && list;
  }
  void wait_idle() {
    q->Signal(fence, ++fv);
    if (fence->GetCompletedValue() < fv) {
      fence->SetEventOnCompletion(fv, ev);
      WaitForSingleObject(ev, 20000);
    }
  }
  void begin() {
    alloc[0]->Reset();
    list->Reset(alloc[0], nullptr);
  }
  void submit() {
    list->Close();
    ID3D12CommandList* l = list;
    q->ExecuteCommandLists(1, &l);
    wait_idle();
  }
  ID3D12Resource* buffer(uint64_t bytes, D3D12_HEAP_TYPE type) {
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = type;
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = bytes;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ID3D12Resource* r = nullptr;
    dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
                                 type == D3D12_HEAP_TYPE_UPLOAD ? D3D12_RESOURCE_STATE_GENERIC_READ
                                                                : D3D12_RESOURCE_STATE_COPY_DEST,
                                 nullptr, __uuidof(ID3D12Resource), (void**)&r);
    return r;
  }
  // mips (0.6.0 follow-up, a tester's CONTROL Resonant): the textures of case "mips" get as many mips as their size allows
  // (data in mip 0; the runtime reads and writes mip 0 and moves subresource 0 alone)
  UINT16 mips = 1;
  ID3D12Resource* texture(UINT w, UINT h, DXGI_FORMAT f, bool uav, D3D12_RESOURCE_STATES state, bool ds = false) {
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Width = w;
    rd.Height = h;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    if (mips > 1 && !ds) {
      UINT16 most = 1;
      for (UINT m = w > h ? w : h; m > 1; m >>= 1) ++most;
      rd.MipLevels = mips < most ? mips : most;
    }
    rd.Format = f;
    rd.SampleDesc.Count = 1;
    rd.Flags = uav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
                   : ds ? D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL : D3D12_RESOURCE_FLAG_NONE;
    ID3D12Resource* r = nullptr;
    dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, state, nullptr, __uuidof(ID3D12Resource), (void**)&r);
    return r;
  }
  void barrier(ID3D12Resource* r, D3D12_RESOURCE_STATES a, D3D12_RESOURCE_STATES b) {
    if (a == b) return;
    D3D12_RESOURCE_BARRIER x = {};
    x.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    x.Transition.pResource = r;
    x.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    x.Transition.StateBefore = a;
    x.Transition.StateAfter = b;
    list->ResourceBarrier(1, &x);
  }
  // Uploads texel data (rows of w * bpp bytes) into texture t (left in state `state`, created in COPY_DEST).
  void upload(ID3D12Resource* t, UINT w, UINT h, UINT bpp, const void* data, D3D12_RESOURCE_STATES state) {
    const UINT pitch = (w * bpp + 255) & ~255u;
    ID3D12Resource* up = buffer((uint64_t)pitch * h, D3D12_HEAP_TYPE_UPLOAD);
    uint8_t* p = nullptr;
    up->Map(0, nullptr, (void**)&p);
    for (UINT y = 0; y < h; ++y) std::memcpy(p + (size_t)y * pitch, (const uint8_t*)data + (size_t)y * w * bpp, w * bpp);
    up->Unmap(0, nullptr);
    D3D12_TEXTURE_COPY_LOCATION dst = {}, src = {};
    dst.pResource = t;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src.pResource = up;
    src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    {  // the footprint format D3D12 expects for this texture (e.g. for depth formats)
      D3D12_RESOURCE_DESC td = t->GetDesc();
      D3D12_PLACED_SUBRESOURCE_FOOTPRINT l = {};
      dev->GetCopyableFootprints(&td, 0, 1, 0, &l, nullptr, nullptr, nullptr);
      src.PlacedFootprint.Footprint.Format = l.Footprint.Format;
    }
    src.PlacedFootprint.Footprint.Width = w;
    src.PlacedFootprint.Footprint.Height = h;
    src.PlacedFootprint.Footprint.Depth = 1;
    src.PlacedFootprint.Footprint.RowPitch = pitch;
    begin();
    list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    barrier(t, D3D12_RESOURCE_STATE_COPY_DEST, state);
    submit();
    up->Release();
  }
};

// Records a copy of texture t (state `state`, w x h, 8 bytes per texel) into a readback buffer; returns the buffer.
ID3D12Resource* record_readback(Gpu& g, ID3D12Resource* t, UINT w, UINT h, D3D12_RESOURCE_STATES state, UINT* pitch) {
  *pitch = (w * 8 + 255) & ~255u;
  ID3D12Resource* rb = g.buffer((uint64_t)*pitch * h, D3D12_HEAP_TYPE_READBACK);
  D3D12_TEXTURE_COPY_LOCATION dst = {}, src = {};
  src.pResource = t;
  src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  dst.pResource = rb;
  dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
  dst.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
  dst.PlacedFootprint.Footprint.Width = w;
  dst.PlacedFootprint.Footprint.Height = h;
  dst.PlacedFootprint.Footprint.Depth = 1;
  dst.PlacedFootprint.Footprint.RowPitch = *pitch;
  g.barrier(t, state, D3D12_RESOURCE_STATE_COPY_SOURCE);
  g.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
  g.barrier(t, D3D12_RESOURCE_STATE_COPY_SOURCE, state);
  return rb;
}

std::vector<float> read_rgb(ID3D12Resource* rb, UINT w, UINT h, UINT pitch) {
  std::vector<float> out((size_t)w * h * 3);
  uint8_t* p = nullptr;
  rb->Map(0, nullptr, (void**)&p);
  for (UINT y = 0; y < h; ++y)
    for (UINT x = 0; x < w; ++x) {
      const uint16_t* t = (const uint16_t*)(p + (size_t)y * pitch + (size_t)x * 8);
      for (int c = 0; c < 3; ++c) out[3 * ((size_t)y * w + x) + c] = from_half(t[c]);
    }
  rb->Unmap(0, nullptr);
  return out;
}

// ---- the minimal front end: split service, frame recording ----

struct FrontEnd {
  Gpu* g = nullptr;
  // the second part of the list waits for this before it runs (set by the split)
  ID3D12Fence* resumeFence = nullptr;
  uint64_t resumeValue = 0;
  int splits = 0;
  WgResult answer = WG_OK;  // what split_command_list returns (WG_ERROR_NOT_READY to test that path)

  static WgResult split(void* ctx, const WgFrame* f, const WgSplitRequest* r) {
    FrontEnd* fe = (FrontEnd*)ctx;
    if (fe->answer != WG_OK) return fe->answer;
    if (!r || r->struct_size < sizeof(WgSplitRequest) || !r->done_fence || !r->resume_fence) return WG_ERROR_PARAMETER;
    auto* list = (ID3D12GraphicsCommandList*)f->command_list;
    if (FAILED(list->Close())) return WG_ERROR_DEVICE;
    ID3D12CommandList* l = list;
    fe->g->q->ExecuteCommandLists(1, &l);  // single-threaded: this is already the game's order
    fe->g->q->Signal((ID3D12Fence*)r->done_fence, r->done_value);
    ID3D12CommandAllocator* a = fe->g->split_alloc ? fe->g->split_alloc : fe->g->alloc[1];
    a->Reset();
    list->Reset(a, nullptr);
    fe->resumeFence = (ID3D12Fence*)r->resume_fence;
    fe->resumeValue = r->resume_value;
    fe->splits++;
    return WG_OK;
  }
  // Submits the second part behind the resume wait and returns the fence value that marks it done (no wait).
  uint64_t submit_nowait() {
    g->list->Close();
    if (resumeFence) g->q->Wait(resumeFence, resumeValue);
    resumeFence = nullptr;
    ID3D12CommandList* l = g->list;
    g->q->ExecuteCommandLists(1, &l);
    g->q->Signal(g->fence, ++g->fv);
    return g->fv;
  }
  // Submits the (second part of the) frame's list behind the resume wait, then waits for the GPU (test only).
  void submit() {
    g->list->Close();
    if (resumeFence) g->q->Wait(resumeFence, resumeValue);
    resumeFence = nullptr;
    ID3D12CommandList* l = g->list;
    g->q->ExecuteCommandLists(1, &l);
    g->wait_idle();
  }
};

struct Backend {
  const WgUpscalerBackend* b = nullptr;
  bool load(const char* dll) {
    HMODULE m = LoadLibraryA(dll);
    if (!m) return false;
    auto get = (PfnWgGetUpscalerBackend)(void*)GetProcAddress(m, "wgGetUpscalerBackend");
    b = get ? get(WG_UPSCALER_INTERFACE_VERSION) : nullptr;
    return b != nullptr;
  }
};

// Scene of one test: render textures with a region, output, values per frame.
struct Scene {
  UINT tw = 70, th = 50;              // render texture size
  UINT rx = 3, ry = 2, rw = 64, rh = 48;
  UINT ow = 96, oh = 72;
  UINT mw = 70, mh = 50;              // motion texture size (O with WG_CREATE_MOTION_OUTPUT_RES)
  ID3D12Resource *color = nullptr, *depth = nullptr, *motion = nullptr, *expo = nullptr, *out = nullptr;
  D3D12_RESOURCE_STATES colorState = test_compute_list() ? D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
                                                         : D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
  D3D12_RESOURCE_STATES depthState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
  bool depthDsv = false;  // depth as a depth-stencil texture: D32_FLOAT, or D32_FLOAT_S8X24_UINT with depthS8
  bool depthS8 = false;
  D3D12_RESOURCE_STATES motionState = test_compute_list() ? D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
                                                          : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE |
                                                                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
  D3D12_RESOURCE_STATES outState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
  // output texture: resource format, the format reported to the runtime, unordered access (dx12same variants)
  DXGI_FORMAT outRes = DXGI_FORMAT_R16G16B16A16_FLOAT, outFmt = DXGI_FORMAT_R16G16B16A16_FLOAT;
  bool outUav = true;
  std::vector<float> colorF, depthF, motionF;  // host copies (fp16-exact values), texture layout
  // motion texture format: RGBA16F (default), RG16F, or RG32F (values that are not FP16 values: a + x / 2^20)
  DXGI_FORMAT motionFmt = DXGI_FORMAT_R16G16B16A16_FLOAT;
  bool colorUploaded = false;

  static float cval(int t, UINT x, UINT y, int c) { return q16(c == 0 ? 0.25f * t + x / 64.0f : c == 1 ? y / 64.0f : 0.125f * c + 0.5f); }

  void make(Gpu& g) {
    color = g.texture(tw, th, DXGI_FORMAT_R16G16B16A16_FLOAT, false, D3D12_RESOURCE_STATE_COPY_DEST);
    depth = g.texture(tw, th,
                      depthS8 ? DXGI_FORMAT_D32_FLOAT_S8X24_UINT : depthDsv ? DXGI_FORMAT_D32_FLOAT : DXGI_FORMAT_R32_FLOAT,
                      false, D3D12_RESOURCE_STATE_COPY_DEST, depthDsv);
    motion = g.texture(mw, mh, motionFmt, false, D3D12_RESOURCE_STATE_COPY_DEST);
    expo = g.texture(1, 1, DXGI_FORMAT_R32_FLOAT, false, D3D12_RESOURCE_STATE_COPY_DEST);
    out = g.texture(ow, oh, outRes, outUav, outState);
    std::vector<float> d((size_t)tw * th);
    for (UINT y = 0; y < th; ++y)
      for (UINT x = 0; x < tw; ++x) d[(size_t)y * tw + x] = 1.0f / (1.0f + x + 3.0f * y);
    depthF = d;
    g.upload(depth, tw, th, 4, d.data(), depthState);
    std::vector<uint16_t> m((size_t)mw * mh * 4);
    motionF.assign((size_t)mw * mh * 2, 0.0f);
    for (UINT y = 0; y < mh; ++y)
      for (UINT x = 0; x < mw; ++x) {
        const float a = q16(x / 128.0f - 0.25f), b = q16(0.5f - y / 64.0f);
        uint16_t* p = &m[4 * ((size_t)y * mw + x)];
        p[0] = to_half(a), p[1] = to_half(b), p[2] = to_half(7.0f), p[3] = 0;
        motionF[2 * ((size_t)y * mw + x)] = a;
        motionF[2 * ((size_t)y * mw + x) + 1] = b;
      }
    if (motionFmt == DXGI_FORMAT_R16G16B16A16_FLOAT) {
      g.upload(motion, mw, mh, 8, m.data(), motionState);
    } else if (motionFmt == DXGI_FORMAT_R16G16_FLOAT) {
      std::vector<uint16_t> m2((size_t)mw * mh * 2);
      for (size_t i = 0; i < (size_t)mw * mh; ++i) m2[2 * i] = m[4 * i], m2[2 * i + 1] = m[4 * i + 1];
      g.upload(motion, mw, mh, 4, m2.data(), motionState);
    } else {  // RG32F
      std::vector<float> m2((size_t)mw * mh * 2);
      for (UINT y = 0; y < mh; ++y)
        for (UINT x = 0; x < mw; ++x) {
          const size_t i = (size_t)y * mw + x;
          m2[2 * i] = motionF[2 * i] + (float)x / 1048576.0f;
          m2[2 * i + 1] = motionF[2 * i + 1] - (float)y / 1048576.0f;
        }
      g.upload(motion, mw, mh, 8, m2.data(), motionState);
    }
    const float e = 0.5f;
    g.upload(expo, 1, 1, 4, &e, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
  }
  void set_color(Gpu& g, int t) {
    std::vector<uint16_t> c((size_t)tw * th * 4);
    colorF.assign((size_t)tw * th * 3, 0.0f);
    for (UINT y = 0; y < th; ++y)
      for (UINT x = 0; x < tw; ++x)
        for (int k = 0; k < 4; ++k) {
          const float v = k < 3 ? cval(t, x, y, k) : 1.0f;
          c[4 * ((size_t)y * tw + x) + k] = to_half(v);
          if (k < 3) colorF[3 * ((size_t)y * tw + x) + k] = v;
        }
    // the texture goes back to COPY_DEST for the upload (it is made in COPY_DEST: not before the first one)
    if (colorUploaded) {
      g.begin();
      g.barrier(color, colorState, D3D12_RESOURCE_STATE_COPY_DEST);
      g.submit();
    }
    colorUploaded = true;
    g.upload(color, tw, th, 8, c.data(), colorState);
  }
  WgFrame frame(Gpu& g, int t) const {
    WgFrame f = {};
    f.struct_size = sizeof f;
    f.command_list = g.list;
    f.color = {color, DXGI_FORMAT_R16G16B16A16_FLOAT, tw, th, (uint32_t)colorState};
    f.depth = {depth,
               (uint32_t)(depthS8 ? DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS
                                  : depthDsv ? DXGI_FORMAT_D32_FLOAT : DXGI_FORMAT_R32_FLOAT),
               tw, th, (uint32_t)depthState};
    f.motion = {motion, (uint32_t)motionFmt, mw, mh, (uint32_t)motionState};
    f.exposure = {expo, DXGI_FORMAT_R32_FLOAT, 1, 1, (uint32_t)D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE};
    f.output = {out, (uint32_t)outFmt, ow, oh, (uint32_t)outState};
    f.motion_scale[0] = 64.0f, f.motion_scale[1] = -32.0f;
    f.motion_offset[0] = 0.5f, f.motion_offset[1] = 0.25f;
    f.jitter[0] = 0.25f, f.jitter[1] = -0.25f;
    f.exposure_mode = WG_EXPOSURE_VALUE;
    f.exposure_value = 1.0f;
    f.pre_exposure = 1.0f;
    f.render_x = rx, f.render_y = ry, f.render_width = rw, f.render_height = rh;
    f.output_width = ow, f.output_height = oh;
    f.frame_index = (uint64_t)t;
    f.command_list_type = test_list_type();
    return f;
  }
  // nearest render pixel of output pixel (x, y) (as the test executor), in texture coordinates
  void src(UINT x, UINT y, UINT* sx, UINT* sy) const {
    *sx = rx + std::min(x * rw / ow, rw - 1);
    *sy = ry + std::min(y * rh / oh, rh - 1);
  }
};

UINT env_uint(const char* name, UINT fallback) {
  const char* text = std::getenv(name);
  if (!text || !*text) return fallback;
  char* end = nullptr;
  const unsigned long value = std::strtoul(text, &end, 10);
  return end != text && *end == '\0' && value > 0 && value <= 16384 ? (UINT)value : fallback;
}

// Optional full-path workload sizing for a matched game context: e.g. render 2560x1440, output 3840x2160,
// max-render 3840x2160. Defaults preserve every existing test size.
void apply_test_dimensions(Scene* s) {
  const UINT rw = env_uint("WG_TEST_RENDER_WIDTH", s->rw);
  const UINT rh = env_uint("WG_TEST_RENDER_HEIGHT", s->rh);
  s->rw = rw;
  s->rh = rh;
  s->tw = rw;
  s->th = rh;
  s->rx = s->ry = 0;
  s->mw = rw;
  s->mh = rh;
  s->ow = env_uint("WG_TEST_OUTPUT_WIDTH", s->ow);
  s->oh = env_uint("WG_TEST_OUTPUT_HEIGHT", s->oh);
  // WG_TEST_TEX_PAD=n: the render rectangle (0, 0, rw, rh) of colour / depth / motion textures n texels wider (a game
  // whose dynamic-resolution textures are larger than the frame's render size, e.g. 1705x960 in 1708x960)
  if (const UINT pad = env_uint("WG_TEST_TEX_PAD", 0)) s->tw = rw + pad, s->mw = s->tw;
  // WG_TEST_MOTIONFMT = rg16f | rg32f (default: RGBA16F)
  if (const char* mf = std::getenv("WG_TEST_MOTIONFMT")) {
    if (std::string(mf) == "rg16f") s->motionFmt = DXGI_FORMAT_R16G16_FLOAT;
    if (std::string(mf) == "rg32f") s->motionFmt = DXGI_FORMAT_R32G32_FLOAT;
  }
}

// WG_TEST_MOTION_OUTRES=1: motion at output resolution (create flag WG_CREATE_MOTION_OUTPUT_RES; the motion texture
// has the output size); call after apply_test_dimensions and any variant that changes the texture sizes.
bool apply_motion_outres(Scene* s) {
  if (!std::getenv("WG_TEST_MOTION_OUTRES")) return false;
  s->mw = s->ow;
  s->mh = s->oh;
  return true;
}

struct Ctx {
  ID3D12Device* device = nullptr;  // device handed to create (default: the test's own)
  Backend be;
  Gpu* g = nullptr;
  FrontEnd fe;
  WgFrontEndServices services = {};
  WgUpscaler* u = nullptr;
  WgResult create(const Scene& s, uint32_t flags, const char* options, bool withServices = true) {
    fe.g = g;
    services.struct_size = sizeof services;
    services.context = &fe;
    services.split_command_list = &FrontEnd::split;
    WgCreateInfo ci = {};
    ci.struct_size = sizeof ci;
    ci.interface_version = WG_UPSCALER_INTERFACE_VERSION;
    ci.d3d12_device = device ? device : g->dev;
    ci.max_render_width = env_uint("WG_TEST_MAX_RENDER_WIDTH", s.rw + 8);
    ci.max_render_height = env_uint("WG_TEST_MAX_RENDER_HEIGHT", s.rh + 8);
    ci.output_width = s.ow;
    ci.output_height = s.oh;
    ci.flags = flags;
    ci.log = &log_fn;
    ci.options = options;
    ci.services = withServices ? &services : nullptr;
    return be.b->create(&ci, &u);
  }
  // One frame: game work before, the dispatch, game work after (reads the output), submit. Returns the output.
  std::vector<float> run(const Scene& s, WgFrame f, WgResult* r) {
    g->begin();
    f.command_list = g->list;
    *r = be.b->dispatch(u, &f);
    UINT pitch = 0;
    ID3D12Resource* rb = record_readback(*g, s.out, s.ow, s.oh, s.outState, &pitch);  // the game's next pass
    fe.submit();
    std::vector<float> o = read_rgb(rb, s.ow, s.oh, pitch);
    rb->Release();
    return o;
  }
  ~Ctx() {
    if (u) be.b->destroy(u);
  }
};

float max_err(const std::vector<float>& got, const std::function<float(UINT, UINT, int)>& want, UINT w, UINT h) {
  float worst = 0;
  for (UINT y = 0; y < h; ++y)
    for (UINT x = 0; x < w; ++x)
      for (int c = 0; c < 3; ++c) {
        const float a = got[3 * ((size_t)y * w + x) + c], b = want(x, y, c);
        const float e = std::fabs(a - b) / std::max(1.0f, std::fabs(b));
        if (!(e <= worst)) worst = std::isnan(e) ? INFINITY : e;
      }
  return worst;
}

const float kTol = 2e-3f;  // output is fp16
// hipleak: GPU memory a create/destroy cycle may leave behind on average (MiB; allocator slack, see the case)
const double kLeakToleranceMiB = 16.0;

// ---- cases ----

bool case_sync(Gpu& g, const char* dll, bool inlineMode, const char* options = nullptr) {
  Ctx c;
  c.g = &g;
  if (!c.be.load(dll)) return std::printf("  cannot load %s\n", dll), false;
  Scene s;
  s.make(g);
  CHECK(c.create(s, WG_CREATE_HDR | WG_CREATE_DEPTH_INVERTED,
                 options ? options : inlineMode ? "executor=d3d12-test" : "executor=d3d12-test;sync=pipelined") == WG_OK);
  if (!c.u) return false;
  for (int t = 1; t <= 4; ++t) {
    s.set_color(g, t);
    WgResult r;
    std::vector<float> o = c.run(s, s.frame(g, t), &r);
    CHECK(r == WG_OK);
    CHECK(c.fe.splits == t);
    const int want_t = inlineMode ? t : t - 1;
    if (want_t >= 1) {
      const float e = max_err(o, [&](UINT x, UINT y, int k) {
        UINT sx, sy;
        s.src(x, y, &sx, &sy);
        return Scene::cval(want_t, sx, sy, k);
      }, s.ow, s.oh);
      std::printf("  frame %d: max error vs color of frame %d = %g\n", t, want_t, e);
      CHECK(e <= kTol);
    } else {  // pipelined frame 1: bilinear upscale of this frame's color (values inside the color's range)
      bool inRange = true;
      for (size_t i = 0; i < o.size(); ++i) inRange &= std::isfinite(o[i]) && o[i] >= 0.0f && o[i] <= 2.0f;
      std::printf("  frame 1: fallback upscale %s\n", inRange ? "in range" : "OUT OF RANGE");
      CHECK(inRange);
    }
  }
  return true;
}

// 0.6.0 follow-up: every texture of the frame with mips (as many as each size allows): the same results as case_sync
bool case_mips(Gpu& g, const char* dll) {
  g.mips = 16;
  const bool ok = case_sync(g, dll, true) && case_sync(g, dll, false) &&
                  case_sync(g, dll, true, "executor=d3d12-test;input_copy=1;output_copy=1");
  g.mips = 1;
  return ok;
}

// bug hunt D: an output texture without unordered access is written only by the RGBA16F copy route; any other is refused
// before anything is recorded (WG_ERROR_INPUT, the host upscales instead), never given an unordered access view
bool case_output_nouav(Gpu& g, const char* dll) {
  struct V { DXGI_FORMAT fmt; const char* options; WgResult want; const char* what; } vs[] = {
      {DXGI_FORMAT_R11G11B10_FLOAT, "executor=d3d12-test;output_copy=0", WG_ERROR_INPUT, "R11G11B10_FLOAT, output_copy=0"},
      {DXGI_FORMAT_R11G11B10_FLOAT, "executor=d3d12-test;input_copy=1;output_copy=1", WG_ERROR_INPUT, "R11G11B10_FLOAT, output_copy=1"},
      {DXGI_FORMAT_R16G16B16A16_FLOAT, "executor=d3d12-test;output_copy=0", WG_ERROR_INPUT, "RGBA16F, output_copy=0"},  // (on by default)
      {DXGI_FORMAT_R16G16B16A16_FLOAT, "executor=d3d12-test;input_copy=1;output_copy=1", WG_OK, "RGBA16F, output_copy=1 (the copy route)"},
  };
  for (const V& v : vs) {
    Ctx c;
    c.g = &g;
    if (!c.be.load(dll)) return false;
    Scene s;
    s.outRes = s.outFmt = v.fmt;
    s.outUav = false;
    s.outState = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;  // (no render-target flag on this texture)
    s.make(g);
    s.set_color(g, 1);
    CHECK(c.create(s, WG_CREATE_HDR | WG_CREATE_DEPTH_INVERTED, v.options) == WG_OK);
    if (!c.u) return false;
    g.begin();
    WgFrame f = s.frame(g, 1);
    f.command_list = g.list;
    const WgResult r = c.be.b->dispatch(c.u, &f);
    c.fe.submit();
    const HRESULT removed = g.dev->GetDeviceRemovedReason();
    std::printf("  output %s without unordered access: result %d (want %d), device %s\n", v.what, (int)r, (int)v.want,
                removed == S_OK ? "ok" : "REMOVED");
    CHECK(r == v.want);
    CHECK(removed == S_OK);
  }
  return true;
}

bool case_view(Gpu& g, const char* dll, const char* view, const char* extra = "") {
  Ctx c;
  c.g = &g;
  if (!c.be.load(dll)) return false;
  Scene s;
  const bool outres = std::strcmp(view, "motion_outres") == 0;
  if (outres) s.mw = s.ow, s.mh = s.oh;
  s.make(g);
  s.set_color(g, 1);
  std::string opt = std::string("executor=d3d12-test;test_view=") + (std::strcmp(view, "depth") == 0 ? "depth" : "motion") +
                    extra;
  CHECK(c.create(s, WG_CREATE_HDR | (outres ? WG_CREATE_MOTION_OUTPUT_RES : 0), opt.c_str()) == WG_OK);
  if (!c.u) return false;
  WgFrame f = s.frame(g, 1);
  WgResult r;
  std::vector<float> o = c.run(s, f, &r);
  CHECK(r == WG_OK);
  const float e = max_err(o, [&](UINT x, UINT y, int k) -> float {
    UINT sx, sy;
    s.src(x, y, &sx, &sy);
    if (std::strcmp(view, "depth") == 0) return s.depthF[(size_t)sy * s.tw + sx];
    if (k == 2) return 0.0f;
    UINT mx = sx, my = sy;
    if (outres) {  // the motion texel under the render pixel's centre (render pixel = (sx - rx, sy - ry))
      mx = std::min((UINT)(((sx - s.rx) + 0.5) * s.mw / s.rw), s.mw - 1);
      my = std::min((UINT)(((sy - s.ry) + 0.5) * s.mh / s.rh), s.mh - 1);
    }
    const float raw = s.motionF[2 * ((size_t)my * s.mw + mx) + k];
    return k == 0 ? f.motion_scale[0] * raw + f.motion_offset[0] : f.motion_scale[1] * raw + f.motion_offset[1];
  }, s.ow, s.oh);
  std::printf("  %s: max error = %g\n", view, e);
  CHECK(e <= kTol);
  return true;
}

bool case_exposure(Gpu& g, const char* dll, const char* extra = "") {
  Ctx c;
  c.g = &g;
  if (!c.be.load(dll)) return false;
  Scene s;
  s.make(g);
  s.set_color(g, 2);
  CHECK(c.create(s, WG_CREATE_HDR, (std::string("executor=d3d12-test") + extra).c_str()) == WG_OK);
  if (!c.u) return false;
  struct M {
    uint32_t mode;
    float value, factor;
  } modes[3] = {{WG_EXPOSURE_VALUE, 2.0f, 2.0f}, {WG_EXPOSURE_TEXTURE, 9.0f, 0.5f}, {WG_EXPOSURE_COMPUTE, 9.0f, 1.0f}};
  int t = 1;
  for (const M& m : modes) {
    WgFrame f = s.frame(g, t++);
    f.exposure_mode = m.mode;
    f.exposure_value = m.value;
    WgResult r;
    std::vector<float> o = c.run(s, f, &r);
    CHECK(r == WG_OK);
    const float e = max_err(o, [&](UINT x, UINT y, int k) {
      UINT sx, sy;
      s.src(x, y, &sx, &sy);
      return Scene::cval(2, sx, sy, k) * m.factor;
    }, s.ow, s.oh);
    std::printf("  exposure mode %u: max error = %g\n", m.mode, e);
    CHECK(e <= kTol);
  }
  return true;
}

bool case_not_ready(Gpu& g, const char* dll) {
  Ctx c;
  c.g = &g;
  if (!c.be.load(dll)) return false;
  Scene s;
  s.make(g);
  CHECK(c.create(s, WG_CREATE_HDR, "executor=d3d12-test") == WG_OK);
  if (!c.u) return false;
  s.set_color(g, 1);
  WgResult r;
  c.fe.answer = WG_ERROR_NOT_READY;
  c.run(s, s.frame(g, 1), &r);
  CHECK(r == WG_ERROR_NOT_READY);
  c.fe.answer = WG_OK;
  for (int t = 2; t <= 3; ++t) {
    s.set_color(g, t);
    std::vector<float> o = c.run(s, s.frame(g, t), &r);
    CHECK(r == WG_OK);
    const float e = max_err(o, [&](UINT x, UINT y, int k) {
      UINT sx, sy;
      s.src(x, y, &sx, &sy);
      return Scene::cval(t, sx, sy, k);
    }, s.ow, s.oh);
    std::printf("  frame %d after a not-ready frame: max error = %g\n", t, e);
    CHECK(e <= kTol);
  }
  return true;
}

// clean/fe-minimize: a minimised game window hands over tiny stand-in textures for a while (Spider-Man 2: a 4x4 motion
// texture; a 4x4 colour and depth here too). Each such frame is passed through with success (never an error, which
// makes a host tear the context down and create another while this one still holds its GPU memory): the colour
// upscaled by the output pass while the colour holds the frame, else the output is left as it was. The context stays,
// and the first frame with the game's real textures runs the network again, starting over.
// pipelined: a network frame shows the previous frame's result (case_pipelined_not_ready), so only the answers, the
// passed-through frame's image and the log are checked there
bool case_minimised(Gpu& g, const char* dll, const char* options, bool pipelined = false) {
  Ctx c;
  c.g = &g;
  if (!c.be.load(dll)) return false;
  Scene s;
  s.make(g);
  CHECK(c.create(s, WG_CREATE_HDR, options) == WG_OK);
  if (!c.u) return false;
  ID3D12Resource* tinyMotion = g.texture(4, 4, s.motionFmt, false, D3D12_RESOURCE_STATE_COPY_DEST);
  ID3D12Resource* tinyColor = g.texture(4, 4, DXGI_FORMAT_R16G16B16A16_FLOAT, false, D3D12_RESOURCE_STATE_COPY_DEST);
  ID3D12Resource* tinyDepth = g.texture(4, 4, DXGI_FORMAT_R32_FLOAT, false, D3D12_RESOURCE_STATE_COPY_DEST);
  g.barrier(tinyMotion, D3D12_RESOURCE_STATE_COPY_DEST, s.motionState);
  g.barrier(tinyColor, D3D12_RESOURCE_STATE_COPY_DEST, s.colorState);
  g.barrier(tinyDepth, D3D12_RESOURCE_STATE_COPY_DEST, s.depthState);
  g.submit();
  const size_t log0 = g_log.size();
  auto close_to = [&](int t, float tol, const std::vector<float>& o) {
    const float e = max_err(o, [&](UINT x, UINT y, int k) {
      UINT sx, sy;
      s.src(x, y, &sx, &sy);
      return Scene::cval(t, sx, sy, k);
    }, s.ow, s.oh);
    return e <= tol ? true : (std::printf("  max error vs colour of frame %d = %g (tolerance %g)\n", t, e, tol), false);
  };
  WgResult r = WG_ERROR;
  s.set_color(g, 1);
  std::vector<float> o = c.run(s, s.frame(g, 1), &r);
  CHECK(r == WG_OK && (pipelined || close_to(1, kTol, o)));
  // minimised: the motion texture is a 4x4 stand-in (the frame's colour and depth still hold it)
  s.set_color(g, 2);
  WgFrame f = s.frame(g, 2);
  f.motion = {tinyMotion, (uint32_t)s.motionFmt, 4, 4, (uint32_t)s.motionState};
  o = c.run(s, f, &r);
  CHECK(r == WG_OK);                  // success for the host: no recreate
  CHECK(close_to(2, 0.05f, o));           // the colour upscaled by the output pass (bilinear: close_to the network's nearest)
  // still minimised, colour and depth stand-ins too: success, nothing written (the output keeps frame 2's image)
  s.set_color(g, 3);
  f = s.frame(g, 3);
  f.motion = {tinyMotion, (uint32_t)s.motionFmt, 4, 4, (uint32_t)s.motionState};
  f.color = {tinyColor, DXGI_FORMAT_R16G16B16A16_FLOAT, 4, 4, (uint32_t)s.colorState};
  f.depth = {tinyDepth, DXGI_FORMAT_R32_FLOAT, 4, 4, (uint32_t)s.depthState};
  o = c.run(s, f, &r);
  CHECK(r == WG_OK);
  CHECK(close_to(2, 0.05f, o));
  // restored: the network runs again on the same context
  for (int t = 4; t <= 5; ++t) {
    s.set_color(g, t);
    o = c.run(s, s.frame(g, t), &r);
    CHECK(r == WG_OK && (pipelined || close_to(t, kTol, o)));
  }
  int passed = 0, again = 0, errors = 0;
  for (size_t i = log0; i < g_log.size(); ++i) {
    passed += g_log[i].find("passed through without the network: the game gave a 4x4 motion texture for the render region") != std::string::npos &&
              g_log[i].find("(no motion pass this frame?)") != std::string::npos;
    again += g_log[i].find("hold the frame again after 2 passed-through frame(s)") != std::string::npos;
    errors += g_log[i].find("exceeds the texture") != std::string::npos || g_log[i].find("active region outside") != std::string::npos;
  }
  std::printf("  minimised: %d pass-through line(s), %d restored line(s), %d error line(s)\n", passed, again, errors);
  CHECK(passed == 1 && again == 1 && errors == 0);  // logged once per run of such frames
  tinyMotion->Release();
  tinyColor->Release();
  tinyDepth->Release();
  return true;
}

// Pipelined with a frame the front end could not split: the next frame shows the last network result.
bool case_pipelined_not_ready(Gpu& g, const char* dll) {
  Ctx c;
  c.g = &g;
  if (!c.be.load(dll)) return false;
  Scene s;
  s.make(g);
  CHECK(c.create(s, WG_CREATE_HDR, "executor=d3d12-test;sync=pipelined") == WG_OK);
  if (!c.u) return false;
  const WgResult answers[5] = {WG_OK, WG_OK, WG_ERROR_NOT_READY, WG_OK, WG_OK};
  const int shown[5] = {0, 1, -1, 2, 4};  // color of the frame expected in the output (0: fallback, -1: none)
  for (int t = 1; t <= 5; ++t) {
    s.set_color(g, t);
    c.fe.answer = answers[t - 1];
    WgResult r;
    std::vector<float> o = c.run(s, s.frame(g, t), &r);
    CHECK(r == answers[t - 1]);
    if (shown[t - 1] > 0) {
      const int w = shown[t - 1];
      const float e = max_err(o, [&](UINT x, UINT y, int k) {
        UINT sx, sy;
        s.src(x, y, &sx, &sy);
        return Scene::cval(w, sx, sy, k);
      }, s.ow, s.oh);
      std::printf("  frame %d: max error vs color of frame %d = %g\n", t, w, e);
      CHECK(e <= kTol);
    }
  }
  return true;
}


// ---- a forwarding device wrapper (as a D3D12 interposer layer might be) whose sharing calls fail ----

class WrapDevice : public ID3D12Device {
 public:
  // minShare = 0: CreateSharedHandle always fails (E_NOTIMPL); > 0: it refuses resources smaller than minShare bytes
  // with E_INVALIDARG (as the game process does for small objects) and forwards the rest.
  WrapDevice(ID3D12Device* inner, bool failOpen, uint64_t minShare = 0)
      : in_(inner), failOpen_(failOpen), minShare_(minShare) {
    in_->AddRef();
  }
  ~WrapDevice() { in_->Release(); }
  int shareCalls = 0, openCalls = 0;
  // IUnknown
  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** out) override {
    if (riid == __uuidof(IUnknown) || riid == __uuidof(ID3D12Object) || riid == __uuidof(ID3D12Device)) {
      *out = this;
      AddRef();
      return S_OK;
    }
    return in_->QueryInterface(riid, out);
  }
  ULONG STDMETHODCALLTYPE AddRef() override { return ++refs_; }
  ULONG STDMETHODCALLTYPE Release() override {
    const ULONG r = --refs_;
    if (!r) delete this;
    return r;
  }
  // ID3D12Object
  HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID g, UINT* n, void* d) override { return in_->GetPrivateData(g, n, d); }
  HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID g, UINT n, const void* d) override { return in_->SetPrivateData(g, n, d); }
  HRESULT STDMETHODCALLTYPE SetPrivateDataInterface(REFGUID g, const IUnknown* d) override {
    return in_->SetPrivateDataInterface(g, d);
  }
  HRESULT STDMETHODCALLTYPE SetName(const WCHAR* n) override { return in_->SetName(n); }
  // ID3D12Device
  UINT STDMETHODCALLTYPE GetNodeCount() override { return in_->GetNodeCount(); }
  HRESULT STDMETHODCALLTYPE CreateCommandQueue(const D3D12_COMMAND_QUEUE_DESC* d, REFIID r, void** o) override {
    return in_->CreateCommandQueue(d, r, o);
  }
  HRESULT STDMETHODCALLTYPE CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE t, REFIID r, void** o) override {
    return in_->CreateCommandAllocator(t, r, o);
  }
  HRESULT STDMETHODCALLTYPE CreateGraphicsPipelineState(const D3D12_GRAPHICS_PIPELINE_STATE_DESC* d, REFIID r,
                                                        void** o) override {
    return in_->CreateGraphicsPipelineState(d, r, o);
  }
  HRESULT STDMETHODCALLTYPE CreateComputePipelineState(const D3D12_COMPUTE_PIPELINE_STATE_DESC* d, REFIID r,
                                                       void** o) override {
    return in_->CreateComputePipelineState(d, r, o);
  }
  HRESULT STDMETHODCALLTYPE CreateCommandList(UINT m, D3D12_COMMAND_LIST_TYPE t, ID3D12CommandAllocator* a,
                                              ID3D12PipelineState* p, REFIID r, void** o) override {
    return in_->CreateCommandList(m, t, a, p, r, o);
  }
  HRESULT STDMETHODCALLTYPE CheckFeatureSupport(D3D12_FEATURE f, void* d, UINT n) override {
    return in_->CheckFeatureSupport(f, d, n);
  }
  HRESULT STDMETHODCALLTYPE CreateDescriptorHeap(const D3D12_DESCRIPTOR_HEAP_DESC* d, REFIID r, void** o) override {
    return in_->CreateDescriptorHeap(d, r, o);
  }
  UINT STDMETHODCALLTYPE GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE t) override {
    return in_->GetDescriptorHandleIncrementSize(t);
  }
  HRESULT STDMETHODCALLTYPE CreateRootSignature(UINT m, const void* b, SIZE_T n, REFIID r, void** o) override {
    return in_->CreateRootSignature(m, b, n, r, o);
  }
  void STDMETHODCALLTYPE CreateConstantBufferView(const D3D12_CONSTANT_BUFFER_VIEW_DESC* d,
                                                  D3D12_CPU_DESCRIPTOR_HANDLE h) override {
    in_->CreateConstantBufferView(d, h);
  }
  void STDMETHODCALLTYPE CreateShaderResourceView(ID3D12Resource* r, const D3D12_SHADER_RESOURCE_VIEW_DESC* d,
                                                  D3D12_CPU_DESCRIPTOR_HANDLE h) override {
    in_->CreateShaderResourceView(r, d, h);
  }
  void STDMETHODCALLTYPE CreateUnorderedAccessView(ID3D12Resource* r, ID3D12Resource* c,
                                                   const D3D12_UNORDERED_ACCESS_VIEW_DESC* d,
                                                   D3D12_CPU_DESCRIPTOR_HANDLE h) override {
    in_->CreateUnorderedAccessView(r, c, d, h);
  }
  void STDMETHODCALLTYPE CreateRenderTargetView(ID3D12Resource* r, const D3D12_RENDER_TARGET_VIEW_DESC* d,
                                                D3D12_CPU_DESCRIPTOR_HANDLE h) override {
    in_->CreateRenderTargetView(r, d, h);
  }
  void STDMETHODCALLTYPE CreateDepthStencilView(ID3D12Resource* r, const D3D12_DEPTH_STENCIL_VIEW_DESC* d,
                                                D3D12_CPU_DESCRIPTOR_HANDLE h) override {
    in_->CreateDepthStencilView(r, d, h);
  }
  void STDMETHODCALLTYPE CreateSampler(const D3D12_SAMPLER_DESC* d, D3D12_CPU_DESCRIPTOR_HANDLE h) override {
    in_->CreateSampler(d, h);
  }
  void STDMETHODCALLTYPE CopyDescriptors(UINT a, const D3D12_CPU_DESCRIPTOR_HANDLE* b, const UINT* c, UINT d,
                                         const D3D12_CPU_DESCRIPTOR_HANDLE* e, const UINT* f,
                                         D3D12_DESCRIPTOR_HEAP_TYPE t) override {
    in_->CopyDescriptors(a, b, c, d, e, f, t);
  }
  void STDMETHODCALLTYPE CopyDescriptorsSimple(UINT n, const D3D12_CPU_DESCRIPTOR_HANDLE a,
                                               const D3D12_CPU_DESCRIPTOR_HANDLE b,
                                               D3D12_DESCRIPTOR_HEAP_TYPE t) override {
    in_->CopyDescriptorsSimple(n, a, b, t);
  }
#ifdef WIDL_EXPLICIT_AGGREGATE_RETURNS  // MinGW header form
  D3D12_RESOURCE_ALLOCATION_INFO* STDMETHODCALLTYPE GetResourceAllocationInfo(D3D12_RESOURCE_ALLOCATION_INFO* ret,
                                                                             UINT m, UINT n,
                                                                             const D3D12_RESOURCE_DESC* d) override {
    return in_->GetResourceAllocationInfo(ret, m, n, d);
  }
  D3D12_HEAP_PROPERTIES* STDMETHODCALLTYPE GetCustomHeapProperties(D3D12_HEAP_PROPERTIES* ret, UINT m,
                                                                  D3D12_HEAP_TYPE t) override {
    return in_->GetCustomHeapProperties(ret, m, t);
  }
#else  // Windows SDK form
  D3D12_RESOURCE_ALLOCATION_INFO STDMETHODCALLTYPE GetResourceAllocationInfo(UINT m, UINT n,
                                                                            const D3D12_RESOURCE_DESC* d) override {
    return in_->GetResourceAllocationInfo(m, n, d);
  }
  D3D12_HEAP_PROPERTIES STDMETHODCALLTYPE GetCustomHeapProperties(UINT m, D3D12_HEAP_TYPE t) override {
    return in_->GetCustomHeapProperties(m, t);
  }
#endif
  HRESULT STDMETHODCALLTYPE CreateCommittedResource(const D3D12_HEAP_PROPERTIES* hp, D3D12_HEAP_FLAGS f,
                                                    const D3D12_RESOURCE_DESC* d, D3D12_RESOURCE_STATES s,
                                                    const D3D12_CLEAR_VALUE* c, REFIID r, void** o) override {
    return in_->CreateCommittedResource(hp, f, d, s, c, r, o);
  }
  HRESULT STDMETHODCALLTYPE CreateHeap(const D3D12_HEAP_DESC* d, REFIID r, void** o) override {
    return in_->CreateHeap(d, r, o);
  }
  HRESULT STDMETHODCALLTYPE CreatePlacedResource(ID3D12Heap* h, UINT64 off, const D3D12_RESOURCE_DESC* d,
                                                 D3D12_RESOURCE_STATES s, const D3D12_CLEAR_VALUE* c, REFIID r,
                                                 void** o) override {
    return in_->CreatePlacedResource(h, off, d, s, c, r, o);
  }
  HRESULT STDMETHODCALLTYPE CreateReservedResource(const D3D12_RESOURCE_DESC* d, D3D12_RESOURCE_STATES s,
                                                   const D3D12_CLEAR_VALUE* c, REFIID r, void** o) override {
    return in_->CreateReservedResource(d, s, c, r, o);
  }
  HRESULT STDMETHODCALLTYPE CreateSharedHandle(ID3D12DeviceChild* o, const SECURITY_ATTRIBUTES* sa, DWORD a,
                                               const WCHAR* n, HANDLE* h) override {
    ++shareCalls;
    if (minShare_) {
      ID3D12Resource* r = nullptr;
      if (o && SUCCEEDED(o->QueryInterface(__uuidof(ID3D12Resource), (void**)&r)) && r) {
        const uint64_t width = r->GetDesc().Width;
        r->Release();
        if (width < minShare_) {
          if (h) *h = nullptr;
          return E_INVALIDARG;
        }
      }
      return in_->CreateSharedHandle(o, sa, a, n, h);
    }
    if (h) *h = nullptr;
    return E_NOTIMPL;  // the failure seen in game
  }
  HRESULT STDMETHODCALLTYPE OpenSharedHandle(HANDLE h, REFIID r, void** o) override {
    ++openCalls;
    if (failOpen_) return E_NOTIMPL;
    return in_->OpenSharedHandle(h, r, o);
  }
  HRESULT STDMETHODCALLTYPE OpenSharedHandleByName(const WCHAR* n, DWORD a, HANDLE* h) override {
    return in_->OpenSharedHandleByName(n, a, h);
  }
  HRESULT STDMETHODCALLTYPE MakeResident(UINT n, ID3D12Pageable* const* o) override { return in_->MakeResident(n, o); }
  HRESULT STDMETHODCALLTYPE Evict(UINT n, ID3D12Pageable* const* o) override { return in_->Evict(n, o); }
  HRESULT STDMETHODCALLTYPE CreateFence(UINT64 v, D3D12_FENCE_FLAGS f, REFIID r, void** o) override {
    return in_->CreateFence(v, f, r, o);
  }
  HRESULT STDMETHODCALLTYPE GetDeviceRemovedReason() override { return in_->GetDeviceRemovedReason(); }
  void STDMETHODCALLTYPE GetCopyableFootprints(const D3D12_RESOURCE_DESC* d, UINT a, UINT b, UINT64 c,
                                               D3D12_PLACED_SUBRESOURCE_FOOTPRINT* l, UINT* rc, UINT64* rs,
                                               UINT64* tb) override {
    in_->GetCopyableFootprints(d, a, b, c, l, rc, rs, tb);
  }
  HRESULT STDMETHODCALLTYPE CreateQueryHeap(const D3D12_QUERY_HEAP_DESC* d, REFIID r, void** o) override {
    return in_->CreateQueryHeap(d, r, o);
  }
  HRESULT STDMETHODCALLTYPE SetStablePowerState(BOOL e) override { return in_->SetStablePowerState(e); }
  HRESULT STDMETHODCALLTYPE CreateCommandSignature(const D3D12_COMMAND_SIGNATURE_DESC* d, ID3D12RootSignature* rs,
                                                   REFIID r, void** o) override {
    return in_->CreateCommandSignature(d, rs, r, o);
  }
  void STDMETHODCALLTYPE GetResourceTiling(ID3D12Resource* r, UINT* a, D3D12_PACKED_MIP_INFO* b, D3D12_TILE_SHAPE* c,
                                           UINT* d, UINT e, D3D12_SUBRESOURCE_TILING* f) override {
    in_->GetResourceTiling(r, a, b, c, d, e, f);
  }
#ifdef WIDL_EXPLICIT_AGGREGATE_RETURNS
  LUID* STDMETHODCALLTYPE GetAdapterLuid(LUID* ret) override { return in_->GetAdapterLuid(ret); }
#else
  LUID STDMETHODCALLTYPE GetAdapterLuid() override { return in_->GetAdapterLuid(); }
#endif

 private:
  ID3D12Device* in_;
  bool failOpen_;
  uint64_t minShare_;
  ULONG refs_ = 1;
};

// Whether this D3D12 implementation can share at all (Windows: yes; Wine's vkd3d: no).
bool sharing_supported(Gpu& g) {
  ID3D12Fence* f = nullptr;
  HANDLE h = nullptr;
  bool ok = SUCCEEDED(g.dev->CreateFence(0, D3D12_FENCE_FLAG_SHARED, __uuidof(ID3D12Fence), (void**)&f)) &&
            SUCCEEDED(g.dev->CreateSharedHandle(f, nullptr, GENERIC_ALL, nullptr, &h));
  if (h) CloseHandle(h);
  if (f) f->Release();
  return ok;
}

// The network backend on a wrapped device whose CreateSharedHandle fails (and optionally OpenSharedHandle), with
// share_path forced or automatic. Where sharing works: create succeeds through the expected fallback and frames are
// right; where it does not (Wine): create fails cleanly and every attempt is logged with its HRESULT.
bool case_wrapped(Gpu& g, const char* dll, bool failOpen, const char* path, const char* expect) {
  const bool sharing = sharing_supported(g);
  std::printf("  sharing supported by this D3D12: %d\n", (int)sharing);
  WrapDevice* w = new WrapDevice(g.dev, failOpen);
  bool ran = true;
  {
    Ctx c;
    c.g = &g;
    c.device = w;
    if (!c.be.load(dll)) {
      w->Release();
      return false;
    }
    Scene s;
    s.make(g);
    g_log.clear();
    const std::string opt = std::string("executor=d3d12-test;test_handles=1;share_path=") + path;
    const WgResult r = c.create(s, WG_CREATE_HDR, opt.c_str());
    bool usedPath = false, loggedHr = false;
    for (const auto& m : g_log) {
      usedPath |= m.find(std::string("shared buffers and fences: ") + expect) != std::string::npos;
      loggedHr |= m.find("CreateSharedHandle (buffer) 0x") != std::string::npos;
    }
    const bool own = std::strcmp(path, "own") == 0;
    if (!own) CHECK(w->shareCalls > 0);  // automatic: the wrapper's CreateSharedHandle was tried first ...
    if (!own || !sharing) CHECK(loggedHr);  // ... and its HRESULT logged
    if (own && sharing) CHECK(w->openCalls > 0);  // own device: objects opened on the game's (wrapped) device
    if (!sharing) {
      CHECK(r != WG_OK);
      CHECK(c.u == nullptr);
      std::printf("  create -> %d (no sharing here: failure path only)\n", (int)r);
    } else {
      CHECK(r == WG_OK);
      CHECK(usedPath);
      if (c.u) {
        for (int t = 1; t <= 2; ++t) {
          s.set_color(g, t);
          WgResult rr;
          std::vector<float> o = c.run(s, s.frame(g, t), &rr);
          CHECK(rr == WG_OK);
          const float e = max_err(o, [&](UINT x, UINT y, int k) {
            UINT sx, sy;
            s.src(x, y, &sx, &sy);
            return Scene::cval(t, sx, sy, k);
          }, s.ow, s.oh);
          std::printf("  frame %d: max error = %g\n", t, e);
          CHECK(e <= kTol);
        }
      }
      ran = c.u != nullptr;
    }
  }
  w->Release();
  return ran;
}

// The arena (one shared buffer for all runtime buffers), on the plain device or on a wrapper that refuses shared
// objects below minShare bytes: the arena grows from 32 MB by doubling until CreateSharedHandle accepts it (each try
// logged). Where sharing works: create succeeds with an arena of expectMiB and frames round-trip through the offsets;
// where it does not (Wine): every size up to 512 MB is tried and logged, then create fails cleanly.
bool case_arena(Gpu& g, const char* dll, uint64_t minShare, int expectMiB) {
  const bool sharing = sharing_supported(g);
  std::printf("  sharing supported by this D3D12: %d\n", (int)sharing);
  WrapDevice* w = minShare ? new WrapDevice(g.dev, false, minShare) : nullptr;
  bool ran = true;
  {
    Ctx c;
    c.g = &g;
    c.device = w;
    if (!c.be.load(dll)) {
      if (w) w->Release();
      return false;
    }
    Scene s;
    s.make(g);
    g_log.clear();
    const WgResult r = c.create(s, WG_CREATE_HDR, "executor=d3d12-test;test_handles=1;share_path=arena");
    auto logged = [&](const std::string& t) {
      for (const auto& m : g_log)
        if (m.find(t) != std::string::npos) return true;
      return false;
    };
    for (int mib = 32; mib < expectMiB; mib *= 2)  // sizes refused on the way (by the wrapper)
      CHECK(logged("arena " + std::to_string(mib) + " MiB: CreateSharedHandle 0x80070057"));
    if (!sharing) {
      CHECK(r != WG_OK);
      CHECK(c.u == nullptr);
      CHECK(logged("arena 512 MiB: CreateSharedHandle 0x"));
      std::printf("  create -> %d (no sharing here: size retries and failure path only)\n", (int)r);
    } else {
      CHECK(r == WG_OK);
      CHECK(logged("arena " + std::to_string(expectMiB) + " MiB (") && logged("shared buffers and fences: arena"));
      for (int t = 1; c.u && t <= 3; ++t) {
        s.set_color(g, t);
        WgResult rr;
        std::vector<float> o = c.run(s, s.frame(g, t), &rr);
        CHECK(rr == WG_OK);
        const float e = max_err(o, [&](UINT x, UINT y, int k) {
          UINT sx, sy;
          s.src(x, y, &sx, &sy);
          return Scene::cval(t, sx, sy, k);
        }, s.ow, s.oh);
        std::printf("  frame %d (slot %d of the arena): max error = %g\n", t, (t - 1) % 2, e);
        CHECK(e <= kTol);
      }
      ran = c.u != nullptr;
    }
  }
  if (w) w->Release();
  return ran;
}

// The arena pool (backend.cpp): an importing executor that keeps its import ("test_keep=1" makes the test executor
// behave like one) gets the arena of the previous context again instead of a new one; an arena that is too small for
// the next context is given up for a larger one, which then serves the smaller sizes too. Every context still
// produces the right frames on the arena it got (stale content of the earlier context included).
bool case_arena_keep(Gpu& g, const char* dll) {
  const bool sharing = sharing_supported(g);
  std::printf("  sharing supported by this D3D12: %d\n", (int)sharing);
  if (!sharing) return true;  // no shared arena here: nothing to keep (case arena covers the failure path)
  // render sizes of the four contexts: the same, a smaller one (fits), a much larger one (new arena), the first again
  // (the larger arena serves it)
  static const UINT kRender[4][2] = {{64, 48}, {48, 32}, {1600, 900}, {64, 48}};
  static const char* const kExpect[4] = {"CreateSharedHandle S_OK", "kept from an earlier context",
                                         "CreateSharedHandle S_OK", "kept from an earlier context"};
  for (int k = 0; k < 4; ++k) {
    Scene s;
    s.rw = kRender[k][0], s.rh = kRender[k][1];
    s.tw = s.mw = s.rw + 6, s.th = s.mh = s.rh + 2;
    if (k == 2) s.ow = 1920, s.oh = 1080;  // well over the 32 MiB minimum arena
    s.make(g);
    auto logged = [&](const std::string& t) {
      for (const auto& m : g_log)
        if (m.find(t) != std::string::npos) return true;
      return false;
    };
    {
      Ctx c;
      c.g = &g;
      if (!c.be.load(dll)) return false;
      g_log.clear();
      const WgResult r = c.create(s, WG_CREATE_HDR, "executor=d3d12-test;test_handles=1;test_keep=1;share_path=arena");
      CHECK(r == WG_OK);
      if (r != WG_OK) return false;
      CHECK(logged(kExpect[k]));
      CHECK(logged("is released (too small") == (k == 2));
      for (int t = 1; t <= 3; ++t) {
        s.set_color(g, t + k);
        WgResult rr;
        std::vector<float> o = c.run(s, s.frame(g, t), &rr);
        CHECK(rr == WG_OK);
        const float e = max_err(o, [&](UINT x, UINT y, int ch) {
          UINT sx, sy;
          s.src(x, y, &sx, &sy);
          return Scene::cval(t + k, sx, sy, ch);
        }, s.ow, s.oh);
        if (t == 3) std::printf("  context %d (render %ux%u): max error = %g\n", k + 1, s.rw, s.rh, e);
        CHECK(e <= kTol);
      }
      g_log.clear();
    }  // destroy
    CHECK(logged("its import kept for the next context"));
    for (ID3D12Resource* r : {s.color, s.depth, s.motion, s.expo, s.out})
      if (r) r->Release();
  }
  return true;
}

// timing_log=1: one summary line after 300 dispatches, with D3D12 times (the test executor has no network times).
bool case_timing(Gpu& g, const char* dll) {
  Ctx c;
  c.g = &g;
  if (!c.be.load(dll)) return false;
  Scene s;
  s.tw = s.mw = 40, s.th = s.mh = 34, s.rx = 1, s.ry = 1, s.rw = 32, s.rh = 32, s.ow = 48, s.oh = 48;
  s.make(g);
  s.set_color(g, 1);
  g_log.clear();
  // timing_log through the diagnostics override WG_RUNTIME_OPTIONS (its keys win over the front end's options)
  SetEnvironmentVariableA("WG_RUNTIME_OPTIONS", "timing_log=1");
  CHECK(c.create(s, WG_CREATE_HDR, "executor=d3d12-test;timing_log=0") == WG_OK);
  SetEnvironmentVariableA("WG_RUNTIME_OPTIONS", nullptr);
  bool fromEnv = false;
  for (const auto& m : g_log) fromEnv |= m.find("options from WG_RUNTIME_OPTIONS: timing_log=1") != std::string::npos;
  CHECK(fromEnv);
  if (!c.u) return false;
  for (int t = 1; t <= 301; ++t) {
    WgResult r;
    c.run(s, s.frame(g, t), &r);
    CHECK(r == WG_OK);
  }
  std::string line;
  for (const auto& m : g_log)
    if (m.find("timing, last 300 dispatches") != std::string::npos) line = m;
  CHECK(!line.empty());
  CHECK(line.find("D3D12 input pass n/a") == std::string::npos);  // timestamps were read back
  CHECK(line.find("CPU dispatch n/a") == std::string::npos);
  return true;
}

// Depth as a depth-stencil texture: copied whole when the active region is the whole texture, else the input pass.
bool case_depth_dsv(Gpu& g, const char* dll, bool whole, bool s8 = false) {
  Ctx c;
  c.g = &g;
  if (!c.be.load(dll)) return false;
  Scene s;
  s.depthDsv = true;
  s.depthS8 = s8;
  if (whole) s.tw = s.mw = 64, s.th = s.mh = 48, s.rx = s.ry = 0;
  s.make(g);
  s.set_color(g, 1);
  CHECK(c.create(s, WG_CREATE_HDR, "executor=d3d12-test;test_view=depth;input_copy=1;output_copy=1") == WG_OK);
  if (!c.u) return false;
  WgResult r;
  std::vector<float> o = c.run(s, s.frame(g, 1), &r);
  CHECK(r == WG_OK);
  const float e = max_err(o, [&](UINT x, UINT y, int) {
    UINT sx, sy;
    s.src(x, y, &sx, &sy);
    return s.depthF[(size_t)sy * s.tw + sx];
  }, s.ow, s.oh);
  bool copiedIn = false, passIn = false;
  for (const auto& m : g_log) {
    copiedIn |= m.find("input: copies") != std::string::npos;
    passIn |= m.find("input: conversion pass (depth is a depth-stencil texture") != std::string::npos;
  }
  std::printf("  depth (%s depth-stencil, %s region): max error = %g, copies %d\n", s8 ? "D32S8" : "D32",
              whole ? "whole" : "partial", e,
              (int)copiedIn);
  CHECK(e <= kTol);
  CHECK(whole ? copiedIn : passIn);
  return true;
}

bool case_create_fails(Gpu& g, const char* dll, bool services) {
  Ctx c;
  c.g = &g;
  if (!c.be.load(dll)) return false;
  Scene s;
  s.make(g);
  g_log.clear();
  // the HIP executor on a machine without HIP / without a model, or a front end without the split service
  const std::string dll_path(dll);
  const std::string test_root = dll_path.substr(0, dll_path.find_last_of("/\\") + 1);
  const std::string opt = "executor=hip;model_dir=" + test_root + "wronggpu-test-no-model";
  const WgResult r = c.create(s, WG_CREATE_HDR, services ? opt.c_str() : "executor=d3d12-test", services);
  std::printf("  create -> %d\n", (int)r);
  CHECK(r != WG_OK);
  CHECK(c.u == nullptr);
  CHECK(!g_log.empty());
  if (!services) CHECK(r == WG_ERROR_UNSUPPORTED);
  if (services) {
    // model, map and kernel files present (dummies): create gets as far as loading the HIP runtime
    char tmp[MAX_PATH];
    GetTempPathA(MAX_PATH, tmp);
    const std::string dir = std::string(tmp) + "wronggpu-interop-test";
    CreateDirectoryA(dir.c_str(), nullptr);
    for (const char* n : {"model.wgcm", "preset-M.v1.map.json", "encoder.co"}) {
      FILE* fp = std::fopen((dir + "\\" + n).c_str(), "wb");
      if (fp) std::fputs("dummy", fp), std::fclose(fp);
    }
    Ctx c2;
    c2.g = &g;
    c2.be.load(dll);
    g_log.clear();
    const std::string o2 = "executor=hip;model_dir=" + dir + ";kernels_dir=" + dir;
    const WgResult r2 = c2.create(s, WG_CREATE_HDR, o2.c_str());
    std::printf("  create with dummy files -> %d\n", (int)r2);
    CHECK(r2 != WG_OK);
    CHECK(c2.u == nullptr);
    // without HIP the message names it; with HIP (a Windows PC with the driver) the dummy model fails instead
    CHECK(!g_log.empty());
    std::printf("  last message: %s\n", g_log.empty() ? "" : g_log.back().c_str());
  }
  return true;
}

bool case_hip(Gpu& g, const char* dll, const std::string& options) {
  Scene s;
  s.tw = 1707, s.th = 960, s.rx = 0, s.ry = 0, s.rw = 1707, s.rh = 960, s.ow = 2560, s.oh = 1440, s.mw = 1707,
  s.mh = 960;
  apply_test_dimensions(&s);
  s.make(g);
  std::vector<std::vector<float>> outs[2];
  for (int mode = 0; mode < 2; ++mode) {
    Ctx c;
    c.g = &g;
    if (!c.be.load(dll)) return false;
    const std::string opt = "executor=hip;" + options + (mode ? ";sync=pipelined" : "");
    const WgResult r = c.create(s, WG_CREATE_HDR | WG_CREATE_DEPTH_INVERTED, opt.c_str());
    CHECK(r == WG_OK);
    if (r != WG_OK) return false;
    for (int t = 1; t <= 4; ++t) {
      s.set_color(g, t);
      WgFrame f = s.frame(g, t);
      f.motion_scale[0] = f.motion_scale[1] = 0.0f;  // static scene
      f.motion_offset[0] = f.motion_offset[1] = 0.0f;
      g.begin();
      f.command_list = g.list;
      LARGE_INTEGER a, b, hz;
      QueryPerformanceCounter(&a);
      const WgResult rr = c.be.b->dispatch(c.u, &f);
      QueryPerformanceCounter(&b);
      QueryPerformanceFrequency(&hz);
      UINT pitch = 0;
      ID3D12Resource* rb = record_readback(g, s.out, s.ow, s.oh, s.outState, &pitch);
      c.fe.submit();
      std::vector<float> o = read_rgb(rb, s.ow, s.oh, pitch);
      rb->Release();
      CHECK(rr == WG_OK);
      bool finite = true;
      for (float v : o) finite &= std::isfinite(v);
      CHECK(finite);
      std::printf("  %s frame %d: dispatch CPU %.3f ms, finite %d\n", mode ? "pipelined" : "inline", t,
                  1000.0 * (b.QuadPart - a.QuadPart) / hz.QuadPart, (int)finite);
      outs[mode].push_back(o);
    }
  }
  for (int t = 2; t <= 4; ++t) {
    const std::vector<float>& a = outs[0][t - 2];
    const std::vector<float>& b = outs[1][t - 1];
    bool same = a.size() == b.size();
    for (size_t i = 0; same && i < a.size(); ++i) same = a[i] == b[i];
    std::printf("  pipelined frame %d == inline frame %d: %d\n", t, t - 1, (int)same);
    CHECK(same);
  }
  return true;
}

// Recreate actual model contexts in one process, alternating M -> K -> M. The HIP library stays loaded between
// the contexts: the process keeps the one HIP runtime it initialised (device.cpp pin_library). Unloading it after a
// context (as 0.4.5-rc1 did) does not return that instance's GPU memory, and the next context pays for a new instance
// (case hipleak measures it).
bool case_hipswitch(Gpu& g, const char* dll, const std::string& modelM, const std::string& modelK) {
  Scene s;
  s.tw = 1707, s.th = 960, s.rx = 0, s.ry = 0, s.rw = 1707, s.rh = 960, s.ow = 2560, s.oh = 1440, s.mw = 1707,
  s.mh = 960;
  apply_test_dimensions(&s);
  s.make(g);
  const std::string options[] = {modelM, modelK, modelM};
  const char* labels[] = {"Model M", "Model K", "Model M"};
  for (int round = 0; round < 3; ++round) {
    {
      Ctx c;
      c.g = &g;
      if (!c.be.load(dll)) return false;
      const std::string opt = "executor=hip;" + options[round];
      const WgResult r = c.create(s, WG_CREATE_HDR | WG_CREATE_DEPTH_INVERTED, opt.c_str());
      CHECK(r == WG_OK);
      if (r != WG_OK) return false;
      for (int t = 1; t <= 2; ++t) {
        s.set_color(g, t + round * 2);
        WgFrame f = s.frame(g, t);
        f.motion_scale[0] = f.motion_scale[1] = 0.0f;
        f.motion_offset[0] = f.motion_offset[1] = 0.0f;
        g.begin();
        f.command_list = g.list;
        const WgResult rr = c.be.b->dispatch(c.u, &f);
        UINT pitch = 0;
        ID3D12Resource* rb = record_readback(g, s.out, s.ow, s.oh, s.outState, &pitch);
        c.fe.submit();
        const std::vector<float> out = read_rgb(rb, s.ow, s.oh, pitch);
        rb->Release();
        bool finite = rr == WG_OK;
        for (float v : out) finite &= std::isfinite(v);
        std::printf("  %s switch round %d frame %d: finite %d\n", labels[round], round + 1, t, (int)finite);
        CHECK(finite);
      }
    }  // destroy the upscaler/runtime before checking the HIP library
    HMODULE hip = GetModuleHandleA("amdhip64_7.dll");
    std::printf("  %s teardown: HIP module %s\n", labels[round], hip ? "still loaded" : "UNLOADED");
    CHECK(hip != nullptr);
    if (g_fail) return false;
  }
  return true;
}

// ---- GPU memory of this process (case hipleak) ----
// Two independent views of what the process holds on the adapter: DXGI's local segment usage of the process
// (IDXGIAdapter3::QueryVideoMemoryInfo, every device of the process on that adapter, so HIP's own allocations count)
// and the performance counter "GPU Process Memory(pid_<pid>_*)\Dedicated Usage" (what a task manager shows), plus the
// process's commit charge and handle count (host-side leaks of a context).
struct ProcessMemory {
  IDXGIAdapter3* adapter = nullptr;
  HMODULE pdh = nullptr;

  bool init(ID3D12Device* dev) {
    typedef HRESULT(WINAPI * PfnFactory)(REFIID, void**);
    HMODULE dxgi = LoadLibraryW(L"dxgi.dll");
    auto create = dxgi ? (PfnFactory)(void*)GetProcAddress(dxgi, "CreateDXGIFactory1") : nullptr;
    IDXGIFactory4* f = nullptr;
    if (create && SUCCEEDED(create(__uuidof(IDXGIFactory4), (void**)&f)) && f) {
      IDXGIAdapter1* a = nullptr;
      if (SUCCEEDED(f->EnumAdapterByLuid(dev->GetAdapterLuid(), __uuidof(IDXGIAdapter1), (void**)&a)) && a) {
        a->QueryInterface(__uuidof(IDXGIAdapter3), (void**)&adapter);
        a->Release();
      }
      f->Release();
    }
    pdh = LoadLibraryW(L"pdh.dll");
    return adapter != nullptr;
  }
  // bytes; 0 when DXGI does not report it
  uint64_t dxgi_local() const {
    DXGI_QUERY_VIDEO_MEMORY_INFO info = {};
    if (!adapter || FAILED(adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info))) return 0;
    return info.CurrentUsage;
  }
  // bytes summed over the process's counter instances; -1 when the counter is not available
  double counter_dedicated() const {
    if (!pdh) return -1;
    auto open = (decltype(&PdhOpenQueryW))(void*)GetProcAddress(pdh, "PdhOpenQueryW");
    auto add = (decltype(&PdhAddEnglishCounterW))(void*)GetProcAddress(pdh, "PdhAddEnglishCounterW");
    auto collect = (decltype(&PdhCollectQueryData))(void*)GetProcAddress(pdh, "PdhCollectQueryData");
    auto values = (decltype(&PdhGetFormattedCounterArrayW))(void*)GetProcAddress(pdh, "PdhGetFormattedCounterArrayW");
    auto close = (decltype(&PdhCloseQuery))(void*)GetProcAddress(pdh, "PdhCloseQuery");
    if (!open || !add || !collect || !values || !close) return -1;
    wchar_t path[128] = {0};
    _snwprintf(path, 127, L"\\GPU Process Memory(pid_%lu_*)\\Dedicated Usage", (unsigned long)GetCurrentProcessId());
    PDH_HQUERY q = nullptr;
    PDH_HCOUNTER c = nullptr;
    double sum = -1;
    if (open(nullptr, 0, &q) != ERROR_SUCCESS) return -1;
    if (add(q, path, 0, &c) == ERROR_SUCCESS && collect(q) == ERROR_SUCCESS) {
      DWORD bytes = 0, count = 0;
      values(c, PDH_FMT_LARGE, &bytes, &count, nullptr);  // PDH_MORE_DATA: the size
      std::vector<uint8_t> buf(bytes ? bytes : 1);
      auto* items = (PDH_FMT_COUNTERVALUE_ITEM_W*)buf.data();
      if (bytes && values(c, PDH_FMT_LARGE, &bytes, &count, items) == ERROR_SUCCESS) {
        sum = 0;
        for (DWORD i = 0; i < count; ++i) sum += (double)items[i].FmtValue.largeValue;
      }
    }
    close(q);
    return sum;
  }
  static uint64_t commit() {
    PROCESS_MEMORY_COUNTERS_EX pm = {};
    pm.cb = sizeof pm;
    return K32GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pm, sizeof pm) ? pm.PrivateUsage : 0;
  }
  static unsigned handles() {
    DWORD n = 0;
    return GetProcessHandleCount(GetCurrentProcess(), &n) ? (unsigned)n : 0;
  }
};

// The upscaler destroyed and created again as a game does at every change of its upscale mode: `cycles` times
// create -> `frames` dispatches -> destroy in one process, the render size rotating over three sizes (output
// 1920x1080) and the options alternating between `optA` and `optB` (e.g. model M and "model=K"). After every destroy
// the process's GPU memory is read. A context must give back what it took: the GPU memory after the last destroy may
// exceed the one after the destroy half way (same size and options: cycles is a multiple of 6) by at most
// `tolMiB` MiB per cycle (allocator slack; see clean/FORMATS.md "Context teardown"). Prints the final frame's
// checksum of every cycle, so that two builds of the runtime can be compared bit for bit.
// hold: the test keeps its own reference on the HIP library after the first create (diagnostics: hipMemGetInfo is read
// after every destroy). thread: every create runs on a thread of its own that ends with it, as in the front end
// (dispatch and destroy stay on the main thread).
bool case_hipleak(Gpu& g, const char* dll, const std::string& optA, const std::string& optB, int cycles, int frames,
                  double tolMiB, bool hold, bool thread) {
  static const UINT kSizes[3][2] = {{1280, 720}, {1114, 626}, {960, 540}};
  Scene scenes[3];
  for (int k = 0; k < 3; ++k) {
    Scene& s = scenes[k];
    s.tw = s.rw = s.mw = kSizes[k][0];
    s.th = s.rh = s.mh = kSizes[k][1];
    s.rx = s.ry = 0;
    s.ow = 1920, s.oh = 1080;
    s.make(g);
  }
  ProcessMemory pm;
  if (!pm.init(g.dev)) std::printf("  note: DXGI does not report the process's video memory here\n");
  const double MiB = 1048576.0;
  struct Sample {
    double dxgi, counter, commit, hipUsed;
    unsigned handles;
  };
  HMODULE held = nullptr;
  typedef int (*PfnMemInfo)(size_t*, size_t*);
  PfnMemInfo memInfo = nullptr;
  auto sample = [&]() {
    g.wait_idle();
    Sleep(400);  // deferred destruction (D3D12, the video memory manager) settles
    Sample m = {};
    m.dxgi = (double)pm.dxgi_local() / MiB;
    const double c = pm.counter_dedicated();
    m.counter = c < 0 ? -1 : c / MiB;
    m.commit = (double)ProcessMemory::commit() / MiB;
    m.handles = ProcessMemory::handles();
    m.hipUsed = -1;
    size_t freeB = 0, totalB = 0;
    if (memInfo && memInfo(&freeB, &totalB) == 0) m.hipUsed = (double)(totalB - freeB) / MiB;
    return m;
  };
  std::vector<Sample> after;
  const Sample start = sample();
  std::printf("  start: GPU memory %.0f MiB (DXGI local), %.0f MiB (counter), commit %.0f MiB, %u handles\n", start.dxgi,
              start.counter, start.commit, start.handles);
  for (int cyc = 0; cyc < cycles; ++cyc) {
    Scene& s = scenes[cyc % 3];
    const bool second = !optB.empty() && (cyc & 1);
    uint64_t hash = 14695981039346656037ull;
    size_t nonfinite = 0;
    std::string ready;
    {
      Ctx c;
      c.g = &g;
      if (!c.be.load(dll)) return false;
      const std::string opt = "executor=hip;" + (second ? optB : optA);
      g_log.clear();
      WgResult r = WG_ERROR;
      if (thread) {  // as the front end does: the create on a thread of its own that ends with it
        struct Arg {
          Ctx* c;
          const Scene* s;
          const char* opt;
          WgResult r;
        } arg = {&c, &s, opt.c_str(), WG_ERROR};
        HANDLE th = CreateThread(nullptr, 0, [](void* p) -> DWORD {
          Arg* a = (Arg*)p;
          a->r = a->c->create(*a->s, WG_CREATE_HDR | WG_CREATE_DEPTH_INVERTED, a->opt);
          return 0;
        }, &arg, 0, nullptr);
        if (th) WaitForSingleObject(th, INFINITE), CloseHandle(th);
        r = arg.r;
      } else {
        r = c.create(s, WG_CREATE_HDR | WG_CREATE_DEPTH_INVERTED, opt.c_str());
      }
      CHECK(r == WG_OK);
      if (r != WG_OK) return false;
      for (const auto& m : g_log)
        if (m.find("network runtime ready") != std::string::npos) {
          const size_t at = m.find("GPU memory ");
          if (at != std::string::npos) ready = m.substr(at, m.find(',', at) - at);
        }
      if (hold && !held && GetModuleHandleExA(0, "amdhip64_7.dll", &held) && held)
        memInfo = (PfnMemInfo)(void*)GetProcAddress(held, "hipMemGetInfo");
      for (int t = 1; t <= frames; ++t) {
        s.set_color(g, t + cyc);
        WgFrame f = s.frame(g, t);
        f.motion_scale[0] = 1.5f, f.motion_scale[1] = -0.75f;
        f.motion_offset[0] = f.motion_offset[1] = 0.0f;
        g.begin();
        f.command_list = g.list;
        const WgResult rr = c.be.b->dispatch(c.u, &f);
        CHECK(rr == WG_OK);
        if (rr != WG_OK) return false;
        if (t < frames) {
          c.fe.submit();
          continue;
        }
        UINT pitch = 0;
        ID3D12Resource* rb = record_readback(g, s.out, s.ow, s.oh, s.outState, &pitch);
        c.fe.submit();
        for (float v : read_rgb(rb, s.ow, s.oh, pitch)) {
          nonfinite += !std::isfinite(v);
          uint32_t bits;
          std::memcpy(&bits, &v, sizeof bits);
          for (int k = 0; k < 4; ++k) hash = (hash ^ ((bits >> (8 * k)) & 255)) * 1099511628211ull;
        }
        rb->Release();
      }
    }  // destroy
    CHECK(nonfinite == 0);
    const Sample m = sample();
    after.push_back(m);
    const Sample& prev = cyc ? after[cyc - 1] : start;
    char hip[64] = "";
    if (m.hipUsed >= 0) std::snprintf(hip, sizeof hip, ", HIP used %.0f MiB", m.hipUsed);
    std::printf("  cycle %2d %s %4ux%-4u (%s): after destroy GPU memory %.0f MiB (%+.0f), counter %.0f MiB (%+.0f), "
                "commit %.0f MiB (%+.0f), %u handles (%+d)%s, HIP library %s, fnv1a64=%016llx\n",
                cyc + 1, second ? "B" : "A", s.rw, s.rh, ready.c_str(), m.dxgi, m.dxgi - prev.dxgi, m.counter,
                m.counter - prev.counter, m.commit, m.commit - prev.commit, m.handles, (int)m.handles - (int)prev.handles,
                hip, GetModuleHandleA("amdhip64_7.dll") ? "loaded" : "unloaded", (unsigned long long)hash);
  }
  if (held) FreeLibrary(held);
  for (Scene& s : scenes)
    for (ID3D12Resource* r : {s.color, s.depth, s.motion, s.expo, s.out})
      if (r) r->Release();
  if (cycles < 2) return true;
  // growth over the second half (cycles a multiple of 6: the same size and options at both ends)
  const int half = cycles / 2, n = cycles - half;
  const Sample &a = after[half - 1], &b = after[cycles - 1];
  const bool useDxgi = pm.adapter != nullptr;
  const double gpu = useDxgi ? (b.dxgi - a.dxgi) / n : (b.counter - a.counter) / n;
  const double counter = a.counter < 0 || b.counter < 0 ? 0 : (b.counter - a.counter) / n;
  const double commit = (b.commit - a.commit) / n;
  std::printf("  growth per create/destroy cycle over cycles %d..%d: GPU memory %+.1f MiB (%s), counter %+.1f MiB, "
              "commit %+.1f MiB, handles %+.1f; allowed GPU memory growth %.1f MiB per cycle\n",
              half + 1, cycles, gpu, useDxgi ? "DXGI local" : "counter", counter, commit,
              ((double)b.handles - (double)a.handles) / n, tolMiB);
  CHECK(gpu <= tolMiB);
  CHECK(counter <= tolMiB);
  return gpu <= tolMiB && counter <= tolMiB;
}

// The real network with two option strings on the same frames (inline): the outputs must be identical bit for bit
// (e.g. a bit-exact kernel choice through the interop output copy). "hipsame" <opts A> <opts B>.
bool case_hip_same(Gpu& g, const char* dll, const std::string& oa, const std::string& ob, int frames) {
  Scene s;
  s.tw = 1707, s.th = 960, s.rx = 0, s.ry = 0, s.rw = 1707, s.rh = 960, s.ow = 2560, s.oh = 1440, s.mw = 1707,
  s.mh = 960;
  // motion at output size: WG_TEST_MOTION_OUTRES, or an option string that names motion_outres (original: the network
  // reads the motion texture itself; point: the input pass resamples it to render pixels)
  const bool outres = std::getenv("WG_TEST_MOTION_OUTRES") != nullptr ||
                      oa.find("motion_outres=") != std::string::npos || ob.find("motion_outres=") != std::string::npos;
  apply_test_dimensions(&s);
  if (outres) s.mw = s.ow, s.mh = s.oh;  // (after apply_test_dimensions, which sets the motion size to R)
  s.make(g);
  std::vector<std::vector<float>> outs[2];
  for (int mode = 0; mode < 2; ++mode) {
    Ctx c;
    c.g = &g;
    if (!c.be.load(dll)) return false;
    const std::string opt = "executor=hip;" + (mode ? ob : oa);
    const WgResult r = c.create(s, WG_CREATE_HDR | WG_CREATE_DEPTH_INVERTED | (outres ? WG_CREATE_MOTION_OUTPUT_RES : 0),
                                opt.c_str());
    CHECK(r == WG_OK);
    if (r != WG_OK) return false;
    for (int t = 1; t <= frames; ++t) {
      s.set_color(g, t);
      WgFrame f = s.frame(g, t);
      f.motion_scale[0] = 1.5f, f.motion_scale[1] = -0.75f;  // moving: history taps off the pixel centres
      f.motion_offset[0] = f.motion_offset[1] = 0.0f;
      g.begin();
      f.command_list = g.list;
      const WgResult rr = c.be.b->dispatch(c.u, &f);
      UINT pitch = 0;
      ID3D12Resource* rb = record_readback(g, s.out, s.ow, s.oh, s.outState, &pitch);
      c.fe.submit();
      outs[mode].push_back(read_rgb(rb, s.ow, s.oh, pitch));
      rb->Release();
      CHECK(rr == WG_OK);
    }
  }
  bool all = true;
  for (int t = 0; t < frames; ++t) {
    const std::vector<float>& a = outs[0][t];
    const std::vector<float>& b = outs[1][t];
    size_t diff = a.size() == b.size() ? 0 : 1;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) diff += std::memcmp(&a[i], &b[i], 4) != 0;
    std::printf("  frame %d: %zu of %zu values differ%c", t + 1, diff, a.size(), 10);
    all &= diff == 0;
  }
  CHECK(all);
  return all;
}

// executor=dx12 against executor=hip through the full interop entry points (create, dispatch with the split service,
// destroy), the real network on synthetic frames: the same game frames through both, the output textures compared
// byte for byte (RGBA16F texels). "dx12same" "<common options>" [frames] [outsrv]: the common options name the
// synthetic model and kernels (model_dir=...;model_format=raw;kernels_dir=...). Frames vary the exposure source
// (value, texture, compute), the pre-exposure (1 and 1.375), resets (frames 1 and 5), jitter and motion. outsrv: the
// output texture is in a shader-resource state between frames (the executors' transitions), else unordered access.
// Prints the CPU time of each dispatch call. Fails unless the dx12 context really ran the D3D12 executor and wrote the
// output itself.
// The output texture's bytes (any format): a readback copy recorded into the test's list.
ID3D12Resource* record_readback_any(Gpu& g, ID3D12Resource* t, D3D12_RESOURCE_STATES state,
                                    D3D12_PLACED_SUBRESOURCE_FOOTPRINT* fp, UINT64* row_bytes, UINT* rows) {
  const D3D12_RESOURCE_DESC d = t->GetDesc();
  UINT64 total = 0;
  g.dev->GetCopyableFootprints(&d, 0, 1, 0, fp, rows, row_bytes, &total);
  ID3D12Resource* rb = g.buffer(total, D3D12_HEAP_TYPE_READBACK);
  D3D12_TEXTURE_COPY_LOCATION dst = {}, src = {};
  src.pResource = t;
  src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  dst.pResource = rb;
  dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
  dst.PlacedFootprint = *fp;
  g.barrier(t, state, D3D12_RESOURCE_STATE_COPY_SOURCE);
  g.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
  g.barrier(t, D3D12_RESOURCE_STATE_COPY_SOURCE, state);
  return rb;
}

std::vector<uint8_t> read_bytes(ID3D12Resource* rb, const D3D12_PLACED_SUBRESOURCE_FOOTPRINT& fp, UINT64 row_bytes,
                                UINT rows) {
  std::vector<uint8_t> out((size_t)row_bytes * rows);
  uint8_t* p = nullptr;
  rb->Map(0, nullptr, (void**)&p);
  for (UINT y = 0; y < rows; ++y)
    std::memcpy(&out[(size_t)y * row_bytes], p + fp.Offset + (size_t)y * fp.Footprint.RowPitch, (size_t)row_bytes);
  rb->Unmap(0, nullptr);
  return out;
}

// D3D12 debug layer (WG_D3D12_DEBUG=1): the messages stored since the last call, errors and warnings printed; returns
// how many were errors or corruption.
int debug_messages(Gpu& g, const char* when) {
  ID3D12InfoQueue* q = nullptr;
  if (FAILED(g.dev->QueryInterface(__uuidof(ID3D12InfoQueue), (void**)&q)) || !q) return 0;
  int errors = 0;
  const UINT64 n = q->GetNumStoredMessages();
  for (UINT64 i = 0; i < n; ++i) {
    SIZE_T len = 0;
    q->GetMessage(i, nullptr, &len);
    std::vector<char> buf(len);
    auto* m = (D3D12_MESSAGE*)buf.data();
    if (FAILED(q->GetMessage(i, m, &len))) continue;
    if (m->Severity > D3D12_MESSAGE_SEVERITY_WARNING) continue;
    errors += m->Severity <= D3D12_MESSAGE_SEVERITY_ERROR;
    std::printf("  [debug layer, %s] %s: %.*s\n", when, m->Severity <= D3D12_MESSAGE_SEVERITY_ERROR ? "ERROR" : "warning",
                (int)m->DescriptionByteLength, m->pDescription);
  }
  q->ClearStoredMessages();
  q->Release();
  return errors;
}

bool case_dx12_same(Gpu& g, const char* dll, const std::string& common, int frames, bool outsrv) {
  Scene s;
  s.tw = 1707, s.th = 960, s.rx = 0, s.ry = 0, s.rw = 1707, s.rh = 960, s.ow = 2560, s.oh = 1440, s.mw = 1707,
  s.mh = 960;
  apply_test_dimensions(&s);
  if (outsrv) s.outState = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
  // Variants (environment): WG_TEST_OUTFMT = rgba16f (default) | rgba32f | r11g11b10 | rgb10a2 | rgba8 | rgba8srgb
  // (an R8G8B8A8_TYPELESS resource reported as sRGB) | rgba16f_typeless; WG_TEST_OUTNOUAV=1: the output texture allows
  // no unordered access; WG_TEST_DEPTH = r32 (default) | d32 | d32s8; WG_TEST_SUBRECT=1: the render rectangle at (5, 3)
  // of a larger texture.
  const char* of = std::getenv("WG_TEST_OUTFMT");
  const std::string ofs = of ? of : "rgba16f";
  if (ofs == "rgba32f") s.outRes = s.outFmt = DXGI_FORMAT_R32G32B32A32_FLOAT;
  else if (ofs == "r11g11b10") s.outRes = s.outFmt = DXGI_FORMAT_R11G11B10_FLOAT;
  else if (ofs == "rgb10a2") s.outRes = s.outFmt = DXGI_FORMAT_R10G10B10A2_UNORM;
  else if (ofs == "rgba8") s.outRes = s.outFmt = DXGI_FORMAT_R8G8B8A8_UNORM;
  else if (ofs == "rgba8srgb") s.outRes = DXGI_FORMAT_R8G8B8A8_TYPELESS, s.outFmt = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
  else if (ofs == "rgba16f_typeless") s.outRes = DXGI_FORMAT_R16G16B16A16_TYPELESS;
  if (std::getenv("WG_TEST_OUTNOUAV")) s.outUav = false;
  const char* dp = std::getenv("WG_TEST_DEPTH");
  if (dp && std::string(dp) == "d32") s.depthDsv = true, s.depthState = D3D12_RESOURCE_STATE_DEPTH_READ |
                                                                         D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
  if (dp && std::string(dp) == "d32s8") s.depthDsv = s.depthS8 = true,
                                        s.depthState = D3D12_RESOURCE_STATE_DEPTH_READ |
                                                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
  if (std::getenv("WG_TEST_SUBRECT")) s.tw = 1720, s.th = 970, s.rx = 5, s.ry = 3, s.mw = 1720, s.mh = 970;
  const bool outres = apply_motion_outres(&s);
  if (!s.outUav && !outsrv) s.outState = D3D12_RESOURCE_STATE_COPY_DEST;
  if (test_compute_list()) {  // compute-list states only: no PIXEL_SHADER_RESOURCE / DEPTH_READ
    if (s.depthDsv) s.depthState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    if (outsrv) s.outState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
  }
  s.make(g);
  std::printf("  output %s%s, depth %s, render %ux%u at (%u, %u) of %ux%u, motion %ux%u%s\n", ofs.c_str(),
              s.outUav ? "" : " (no unordered access)", dp ? dp : "r32", s.rw, s.rh, s.rx, s.ry, s.tw, s.th, s.mw, s.mh,
              outres ? " (WG_CREATE_MOTION_OUTPUT_RES)" : "");
  std::vector<std::vector<uint8_t>> outs[2];
  double cpu[2] = {0, 0};
  for (int mode = 0; mode < 2; ++mode) {
    g_log.clear();
    Ctx c;
    c.g = &g;
    // WG_TEST_WRAP=1: the dx12 context gets a wrapper of the device (as an interposer layer hands to FSR); the game's
    // lists and textures stay the real device's
    WrapDevice* w = mode && std::getenv("WG_TEST_WRAP") ? new WrapDevice(g.dev, false, 1) : nullptr;
    if (w) c.device = w;
    if (!c.be.load(dll)) return false;
    // WG_TEST_DX12_OPTIONS: options for the dx12 context only (e.g. a negative control: output_copy=0 there)
    // WG_TEST_DX12_DEFAULT=1: the second context names no executor (the runtime's default: dx12 on RDNA 4)
    const char* xo = std::getenv("WG_TEST_DX12_OPTIONS");
    const bool dflt = mode && std::getenv("WG_TEST_DX12_DEFAULT");
    const std::string opt = (dflt ? common : "executor=hip;" + common + (mode ? ";executor=dx12" : "")) +
                            (mode && xo ? std::string(";") + xo : std::string());
    LARGE_INTEGER c0, c1, chz;
    QueryPerformanceCounter(&c0);
    const WgResult r = c.create(s, WG_CREATE_HDR | WG_CREATE_DEPTH_INVERTED | (outres ? WG_CREATE_MOTION_OUTPUT_RES : 0),
                                opt.c_str());
    QueryPerformanceCounter(&c1);
    QueryPerformanceFrequency(&chz);
    std::printf("  %s create: %.1f ms%s\n", mode ? "dx12" : "hip", 1000.0 * (c1.QuadPart - c0.QuadPart) / chz.QuadPart,
                w ? " (through a device wrapper)" : "");
    if (w) w->Release();  // the context holds its own references
    CHECK(debug_messages(g, mode ? "dx12 create" : "hip create") == 0);
    CHECK(r == WG_OK);
    if (r != WG_OK) return false;
    for (int t = 1; t <= frames; ++t) {
      s.set_color(g, t);
      WgFrame f = s.frame(g, t);
      f.motion_scale[0] = 1.5f, f.motion_scale[1] = -0.75f;  // moving: history taps off the pixel centres
      f.motion_offset[0] = 0.25f, f.motion_offset[1] = 0.0f;
      const float jit[4][2] = {{0.0f, -1.0f / 6}, {-0.25f, 1.0f / 6}, {0.25f, -7.0f / 18}, {-0.375f, -1.0f / 18}};
      f.jitter[0] = jit[t % 4][0], f.jitter[1] = jit[t % 4][1];
      f.exposure_mode = t % 3 == 0 ? WG_EXPOSURE_COMPUTE : t % 3 == 1 ? WG_EXPOSURE_VALUE : WG_EXPOSURE_TEXTURE;
      f.exposure_value = 0.75f + (float)(t % 24) / 24.0f;
      // WG_TEST_PI: the odd frames' pre-exposure (default 1; the even frames: 1.375)
      const char* pe = std::getenv("WG_TEST_PI");
      f.pre_exposure = t % 2 ? (pe ? (float)std::atof(pe) : 1.0f) : 1.375f;
      f.reset = (t == 1 || t == 5) ? 1u : 0u;
      g.begin();
      f.command_list = g.list;
      LARGE_INTEGER a, b, hz;
      QueryPerformanceCounter(&a);
      const WgResult rr = c.be.b->dispatch(c.u, &f);
      QueryPerformanceCounter(&b);
      QueryPerformanceFrequency(&hz);
      cpu[mode] += 1000.0 * (b.QuadPart - a.QuadPart) / hz.QuadPart;
      D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp = {};
      UINT64 rowBytes = 0;
      UINT rows = 0;
      ID3D12Resource* rb = record_readback_any(g, s.out, s.outState, &fp, &rowBytes, &rows);
      c.fe.submit();
      outs[mode].push_back(read_bytes(rb, fp, rowBytes, rows));
      rb->Release();
      CHECK(rr == WG_OK);
      CHECK(debug_messages(g, mode ? "dx12 frame" : "hip frame") == 0);
    }
    // WG_TEST_EXPECT (second context): dx12 (default: the D3D12 executor ran every frame and wrote the output), switch
    // (the default executor met a frame it does not take and the HIP executor took over the context; since the owner's
    // decision of 2026-11-04 only with hip_switch=on in the options), hip (the HIP executor from create on, after the
    // D3D12 executor refused the context)
    const char* ex = std::getenv("WG_TEST_EXPECT");
    const std::string expect = ex ? ex : "dx12";
    if (mode == 1 && expect != "dx12") {
      bool ready_dx12 = false, ready_hip = false, switched = false, refused = false;
      for (const auto& m : g_log) {
        ready_dx12 |= m.find("network runtime ready: executor dx12") != std::string::npos;
        ready_hip |= m.find("network runtime ready: executor hip") != std::string::npos;
        switched |= m.find("the HIP executor runs this context from now on") != std::string::npos;
        refused |= m.find("executor dx12 (default on RDNA 4) could not take this context") != std::string::npos;
      }
      std::printf("  second context: ready dx12 %d, ready hip %d, switched to HIP %d, refused at create %d\n",
                  (int)ready_dx12, (int)ready_hip, (int)switched, (int)refused);
      if (expect == "switch") CHECK(ready_dx12 && switched && !refused);
      else CHECK(ready_hip && refused && !switched);
    }
    if (mode == 1 && expect == "dx12") {
      bool ran = false, direct_in = false, wrote = false;
      std::string route;
      for (const auto& m : g_log) {
        ran |= m.find("executor dx12:") != std::string::npos;
        direct_in |= m.find("input: read directly by the network's D3D12 kernels") != std::string::npos;
        const size_t k = m.find("output: ");
        if (k != std::string::npos && m.find("frame ") != std::string::npos) {
          route = m.substr(k);
          wrote |= m.find("bilinear fallback") == std::string::npos;
        }
      }
      std::printf("  dx12 context: D3D12 executor %d, inputs read directly %d, %s\n", (int)ran, (int)direct_in,
                  route.c_str());
      CHECK(ran);
      CHECK(direct_in);
      CHECK(wrote);
    }
  }
  std::printf("  dispatch CPU per frame: hip %.3f ms, dx12 %.3f ms\n", cpu[0] / frames, cpu[1] / frames);
  bool all = true;
  for (int t = 0; t < frames; ++t) {
    const std::vector<uint8_t>& a = outs[0][t];
    const std::vector<uint8_t>& b = outs[1][t];
    size_t diff = a.size() == b.size() ? 0 : 1;
    size_t first = (size_t)-1;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i)
      if (a[i] != b[i]) {
        if (first == (size_t)-1) first = i;
        ++diff;
      }
    std::printf("  frame %d: %zu of %zu bytes differ", t + 1, diff, a.size());
    if (diff && first != (size_t)-1) std::printf(" (first at byte %zu: hip 0x%02x, dx12 0x%02x)", first, a[first], b[first]);
    std::printf("\n");
    all &= diff == 0;
  }
  CHECK(all);
  return all;
}

// Soak test of the handover (real network): many frames with idle gaps of random length between them (loading
// screens, light frames), each frame waited for with a timeout, so a handover that never completes is reported as a
// hang at its frame instead of freezing. Options as for "hip" (e.g. "ready_wait=memory;recon=output").
bool case_soak(Gpu& g, const char* dll, const std::string& options, int frames, int first_frame = 1) {
  Scene s;
  s.tw = 1707, s.th = 960, s.rx = 0, s.ry = 0, s.rw = 1707, s.rh = 960, s.ow = 2560, s.oh = 1440, s.mw = 1707,
  s.mh = 960;
  apply_test_dimensions(&s);
  s.make(g);
  Ctx c;
  c.g = &g;
  if (!c.be.load(dll)) return false;
  const std::string opt = "executor=hip;" + options;
  const WgResult r = c.create(s, WG_CREATE_HDR | WG_CREATE_DEPTH_INVERTED, opt.c_str());
  CHECK(r == WG_OK);
  if (r != WG_OK) return false;
  uint32_t rng = 12345u;
  auto rnd = [&rng](uint32_t n) { rng = rng * 1664525u + 1013904223u; return (rng >> 8) % n; };
  LARGE_INTEGER hz, t0, t1;
  QueryPerformanceFrequency(&hz);
  QueryPerformanceCounter(&t0);
  int gaps = 0, slow = 0;
  double worst = 0.0;
  for (int t = first_frame; t < first_frame + frames; ++t) {
    s.set_color(g, t);
    WgFrame f = s.frame(g, t);
    f.motion_scale[0] = f.motion_scale[1] = 0.0f;
    f.motion_offset[0] = f.motion_offset[1] = 0.0f;
    g.begin();
    f.command_list = g.list;
    const WgResult rr = c.be.b->dispatch(c.u, &f);
    if (rr != WG_OK) { std::printf("  soak frame %d: dispatch %d\n", t, (int)rr); CHECK(rr == WG_OK); return false; }
    LARGE_INTEGER s0, s1;
    QueryPerformanceCounter(&s0);
    c.fe.submit();  // waits up to 20 s for the GPU
    QueryPerformanceCounter(&s1);
    const double fms = 1000.0 * (double)(s1.QuadPart - s0.QuadPart) / (double)hz.QuadPart;
    worst = fms > worst ? fms : worst;
    slow += fms > 20.0 ? 1 : 0;
    if (g.fence->GetCompletedValue() < g.fv) {
      std::printf("  soak HANG at frame %d: the frame did not complete within 20 s\n", t);
      CHECK(false);
      return false;
    }
    const uint32_t k = rnd(100);
    int ms = k < 60 ? 0 : k < 85 ? (int)rnd(5) : k < 97 ? (int)rnd(40) : 100 + (int)rnd(400);  // idle gaps
    if (ms && !std::getenv("WG_SOAK_NOGAP")) Sleep((DWORD)ms), gaps++;  // WG_SOAK_NOGAP: back to back (GPU-bound game)
    if (t % 500 == 0) std::printf("  soak %d frames ok (%d idle gaps)\n", t, gaps);
  }
  QueryPerformanceCounter(&t1);
  std::printf("  soak: %d frames, %d idle gaps, no hang, %.1f s; slowest frame %.1f ms, %d frames over 20 ms\n", frames,
              gaps, (double)(t1.QuadPart - t0.QuadPart) / hz.QuadPart, worst, slow);
  return true;
}

// Soak with game-like pacing: up to 3 frames in flight (no wait per frame), random CPU gaps between frames, static
// inputs. A frame whose allocators are still in use after 20 s is reported as a hang.
bool case_soak_pipe(Gpu& g, const char* dll, const std::string& options, int frames, bool benchmark = false) {
  Scene s;
  s.tw = 1707, s.th = 960, s.rx = 0, s.ry = 0, s.rw = 1707, s.rh = 960, s.ow = 2560, s.oh = 1440, s.mw = 1707,
  s.mh = 960;
  apply_test_dimensions(&s);
  const bool outres = apply_motion_outres(&s);
  s.make(g);
  s.set_color(g, 1);
  Ctx c;
  c.g = &g;
  if (!c.be.load(dll)) return false;
  // WG_TEST_DEFAULT_EXECUTOR=1: no executor key of the test's own (the runtime's default choice; the options may
  // still name one)
  const std::string opt = (std::getenv("WG_TEST_DEFAULT_EXECUTOR") ? "" : "executor=hip;") + options;
  const WgResult r = c.create(s, WG_CREATE_HDR | WG_CREATE_DEPTH_INVERTED |
                                       (benchmark ? WG_CREATE_AUTO_EXPOSURE : 0) |
                                       (outres ? WG_CREATE_MOTION_OUTPUT_RES : 0), opt.c_str());
  CHECK(r == WG_OK);
  if (r != WG_OK) return false;
  uint32_t rng = 777u;
  auto rnd = [&rng](uint32_t n) { rng = rng * 1664525u + 1013904223u; return (rng >> 8) % n; };
  int gaps = 0;
  // benchmark: wall time per frame after the warm-up (frames in flight, back to back: the GPU-bound frame rate)
  const int wall_from = frames > 600 ? 301 : 1;
  LARGE_INTEGER wall0 = {}, wall1 = {}, wall_hz = {};
  QueryPerformanceFrequency(&wall_hz);
  for (int t = 1; t <= frames; ++t) {
    if (t == wall_from) QueryPerformanceCounter(&wall0);
    const int k = t % 4;
    if (!g.begin_ring(k)) { std::printf("  soak-pipe HANG before frame %d: frame %d never finished%c", t, t - 4, 10); CHECK(false); return false; }
    WgFrame f = s.frame(g, t);
    f.motion_scale[0] = f.motion_scale[1] = 0.0f;
    f.motion_offset[0] = f.motion_offset[1] = 0.0f;
    if (benchmark) {
      f.exposure_mode = WG_EXPOSURE_COMPUTE;
      // WG_TEST_PI: a pre-exposure other than 1 (the colour divisions of the encoder and post)
      if (const char* pe = std::getenv("WG_TEST_PI")) f.pre_exposure = (float)std::atof(pe);
      f.motion_scale[0] = 1.5f;
      f.motion_scale[1] = -0.75f;
      // Spec 49 synthetic-sequence jitters; no captured game data.
      const float jitter[4][2] = {{0, -1.0f/6}, {-0.25f, 1.0f/6}, {0.25f, -7.0f/18}, {-0.375f, -1.0f/18}};
      f.jitter[0] = jitter[(t - 1) % 4][0];
      f.jitter[1] = jitter[(t - 1) % 4][1];
    }
    f.command_list = g.list;
    const WgResult rr = c.be.b->dispatch(c.u, &f);
    if (rr != WG_OK) { std::printf("  soak-pipe frame %d: dispatch %d%c", t, (int)rr, 10); CHECK(rr == WG_OK); return false; }
    g.ringv[k] = c.fe.submit_nowait();
    const uint32_t q = rnd(100);
    const int ms = q < 70 ? 0 : q < 90 ? (int)rnd(3) : q < 98 ? (int)rnd(20) : 100 + (int)rnd(300);
    if (ms && !benchmark && !std::getenv("WG_SOAK_NOGAP")) Sleep((DWORD)ms), gaps++;  // WG_SOAK_NOGAP: back to back (GPU-bound game)
    if (t % 500 == 0) std::printf("  soak-pipe %d frames ok (%d gaps)%c", t, gaps, 10);
  }
  g.wait_idle();
  QueryPerformanceCounter(&wall1);
  if (benchmark)
    std::printf("  hipbench wall: %.4f ms per frame over frames %d..%d (up to 4 frames in flight, back to back)%c",
                1000.0 * (double)(wall1.QuadPart - wall0.QuadPart) / (double)wall_hz.QuadPart / (frames - wall_from + 1),
                wall_from, frames, 10);
  const bool done = g.fence->GetCompletedValue() >= g.fv;
  if (!done) std::printf("  soak-pipe HANG at the end%c", 10);
  CHECK(done);
  g.split_alloc = nullptr;
  std::printf("  soak-pipe: %d frames, %d gaps, %s%c", frames, gaps, done ? "no hang" : "HANG", 10);
  if (benchmark && done) {
    // This readback is outside every measured upscaler span.
    g.begin();
    UINT pitch = 0;
    ID3D12Resource* rb = record_readback(g, s.out, s.ow, s.oh, s.outState, &pitch);
    g.submit();
    const auto rgb = read_rgb(rb, s.ow, s.oh, pitch);
    rb->Release();
    uint64_t hash = 14695981039346656037ull;
    size_t nonfinite = 0;
    for (float v : rgb) {
      nonfinite += !std::isfinite(v);
      uint32_t bits;
      std::memcpy(&bits, &v, sizeof bits);
      for (int k = 0; k < 4; ++k) hash = (hash ^ ((bits >> (8 * k)) & 255)) * 1099511628211ull;
    }
    std::printf("  hipbench final output: %zu values, nonfinite=%zu, fnv1a64=%016llx\n",
                rgb.size(), nonfinite, (unsigned long long)hash);
    CHECK(nonfinite == 0);
  }
  return done;
}

// The last context logged both copy paths (and no fallback to the passes).
bool copied() {
  bool in = false, out = false, pass = false;
  for (const auto& m : g_log) {
    in |= m.find("input: copies") != std::string::npos;
    out |= m.find("output: CopyTextureRegion") != std::string::npos;
    pass |= m.find("input: conversion pass") != std::string::npos || m.find("output: compute pass") != std::string::npos ||
            m.find("output: bilinear fallback") != std::string::npos;
  }
  CHECK(in);
  CHECK(out);
  CHECK(!pass);
  return in && out && !pass;
}

}  // namespace

// ---- asyncsim: a simulated game frame on the DIRECT queue and the network (executor=dx12) in the game's list or on a
// COMPUTE queue (timing only; the network's output is not consumed). Original WrongGPU test code.
//   test_interop_d3d12.exe <runtime> asyncsim "<options naming the synthetic model>" <mode> [frames] [pre] [post] [iters]
// mode: serial (pre, network, post in one DIRECT list) | ideal (network on the COMPUTE queue, no dependencies) |
//       pipe (network t waits for pre t; post t waits for network t-1) | inline (post t waits for network t) |
//       game (pre + post only) | net (network only, DIRECT list)
// pre / post: dispatches of the game load before / after the upscaler (each 4M float4 elements, `iters` ALU steps).
// Prints the wall time per frame over the frames after the first 300 (up to 3 frames in flight).
struct GameLoad {
  ID3D12RootSignature* rs = nullptr;
  ID3D12PipelineState* pso = nullptr;
  ID3D12Resource* buf = nullptr;
  ID3D12DescriptorHeap* heap = nullptr;
  uint32_t n = 4u << 20, iters = 32;
  bool init(Gpu& g) {
    typedef HRESULT(WINAPI * PfnCompile)(LPCVOID, SIZE_T, LPCSTR, const void*, void*, LPCSTR, LPCSTR, UINT, UINT,
                                         ID3DBlob**, ID3DBlob**);
    HMODULE dc = LoadLibraryW(L"d3dcompiler_47.dll");
    auto compile = dc ? (PfnCompile)(void*)GetProcAddress(dc, "D3DCompile") : nullptr;
    if (!compile) return false;
    const char* src =
        "cbuffer C : register(b0) { uint n; uint iters; };\n"
        "RWByteAddressBuffer buf : register(u0);\n"
        "[numthreads(256, 1, 1)] void main(uint3 id : SV_DispatchThreadID) {\n"
        "  if (id.x >= n) return;\n"
        "  float4 v = asfloat(buf.Load4(id.x * 16));\n"
        "  [loop] for (uint k = 0; k < iters; ++k) v = v * 0.999f + float4(0.5f, 0.25f, 0.125f, 1.0f);\n"
        "  buf.Store4(id.x * 16, asuint(v));\n"
        "}\n";
    ID3DBlob *cs = nullptr, *er = nullptr;
    if (FAILED(compile(src, std::strlen(src), "gameload", nullptr, nullptr, "main", "cs_5_0", 1u << 15, 0, &cs, &er)))
      return false;
    D3D12_DESCRIPTOR_RANGE r = {};
    r.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    r.NumDescriptors = 1;
    D3D12_ROOT_PARAMETER p[2] = {};
    p[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    p[0].Constants.Num32BitValues = 2;
    p[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    p[1].DescriptorTable.NumDescriptorRanges = 1;
    p[1].DescriptorTable.pDescriptorRanges = &r;
    D3D12_ROOT_SIGNATURE_DESC rd = {};
    rd.NumParameters = 2;
    rd.pParameters = p;
    typedef HRESULT(WINAPI * PfnSer)(const D3D12_ROOT_SIGNATURE_DESC*, D3D_ROOT_SIGNATURE_VERSION, ID3DBlob**, ID3DBlob**);
    auto ser = (PfnSer)(void*)GetProcAddress(GetModuleHandleW(L"d3d12.dll"), "D3D12SerializeRootSignature");
    ID3DBlob* blob = nullptr;
    if (!ser || FAILED(ser(&rd, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &er)) ||
        FAILED(g.dev->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                          __uuidof(ID3D12RootSignature), (void**)&rs)))
      return false;
    D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
    pd.pRootSignature = rs;
    pd.CS.pShaderBytecode = cs->GetBufferPointer();
    pd.CS.BytecodeLength = cs->GetBufferSize();
    if (FAILED(g.dev->CreateComputePipelineState(&pd, __uuidof(ID3D12PipelineState), (void**)&pso))) return false;
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC bd = {};
    bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bd.Width = (uint64_t)n * 16;
    bd.Height = 1;
    bd.DepthOrArraySize = 1;
    bd.MipLevels = 1;
    bd.SampleDesc.Count = 1;
    bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    bd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    if (FAILED(g.dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                              nullptr, __uuidof(ID3D12Resource), (void**)&buf)))
      return false;
    D3D12_DESCRIPTOR_HEAP_DESC hd = {};
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    hd.NumDescriptors = 1;
    hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (FAILED(g.dev->CreateDescriptorHeap(&hd, __uuidof(ID3D12DescriptorHeap), (void**)&heap))) return false;
    D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
    ud.Format = DXGI_FORMAT_R32_TYPELESS;
    ud.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    ud.Buffer.NumElements = n * 4;
    ud.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
    g.dev->CreateUnorderedAccessView(buf, nullptr, &ud, heap->GetCPUDescriptorHandleForHeapStart());
    return true;
  }
  // WG_SIM_LOAD=ps: a raster load instead (fullscreen triangles with blending into a 2560x1440 RGBA16F target, `iters`
  // ALU steps per pixel), as a game's lighting / post passes
  bool ps = false;
  ID3D12RootSignature* grs = nullptr;
  ID3D12PipelineState* gpso = nullptr;
  ID3D12Resource* rt = nullptr;
  ID3D12DescriptorHeap* rtv = nullptr;
  bool init_ps(Gpu& g) {
    typedef HRESULT(WINAPI * PfnCompile)(LPCVOID, SIZE_T, LPCSTR, const void*, void*, LPCSTR, LPCSTR, UINT, UINT,
                                         ID3DBlob**, ID3DBlob**);
    auto compile = (PfnCompile)(void*)GetProcAddress(LoadLibraryW(L"d3dcompiler_47.dll"), "D3DCompile");
    const char* src =
        "cbuffer C : register(b0) { uint iters; };\n"
        "float4 vs(uint id : SV_VertexID) : SV_Position {\n"
        "  float2 p = float2((id << 1) & 2, id & 2); return float4(p * float2(2, -2) + float2(-1, 1), 0, 1); }\n"
        "float4 ps(float4 pos : SV_Position) : SV_Target {\n"
        "  float4 v = float4(pos.xy * 0.001f, 0.5f, 1.0f);\n"
        "  [loop] for (uint k = 0; k < iters; ++k) v = v * 0.999f + float4(0.001f, 0.002f, 0.003f, 0.0f);\n"
        "  return v * 0.01f; }\n";
    ID3DBlob *vsb = nullptr, *psb = nullptr, *er = nullptr;
    if (!compile || FAILED(compile(src, std::strlen(src), "psload", nullptr, nullptr, "vs", "vs_5_0", 0, 0, &vsb, &er)) ||
        FAILED(compile(src, std::strlen(src), "psload", nullptr, nullptr, "ps", "ps_5_0", 0, 0, &psb, &er)))
      return false;
    D3D12_ROOT_PARAMETER p = {};
    p.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    p.Constants.Num32BitValues = 1;
    D3D12_ROOT_SIGNATURE_DESC rd = {};
    rd.NumParameters = 1;
    rd.pParameters = &p;
    typedef HRESULT(WINAPI * PfnSer)(const D3D12_ROOT_SIGNATURE_DESC*, D3D_ROOT_SIGNATURE_VERSION, ID3DBlob**, ID3DBlob**);
    auto ser = (PfnSer)(void*)GetProcAddress(GetModuleHandleW(L"d3d12.dll"), "D3D12SerializeRootSignature");
    ID3DBlob* blob = nullptr;
    if (FAILED(ser(&rd, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &er)) ||
        FAILED(g.dev->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                          __uuidof(ID3D12RootSignature), (void**)&grs)))
      return false;
    D3D12_GRAPHICS_PIPELINE_STATE_DESC gd = {};
    gd.pRootSignature = grs;
    gd.VS = {vsb->GetBufferPointer(), vsb->GetBufferSize()};
    gd.PS = {psb->GetBufferPointer(), psb->GetBufferSize()};
    gd.BlendState.RenderTarget[0].BlendEnable = TRUE;
    gd.BlendState.RenderTarget[0].SrcBlend = gd.BlendState.RenderTarget[0].DestBlend = D3D12_BLEND_ONE;
    gd.BlendState.RenderTarget[0].BlendOp = D3D12_BLEND_OP_ADD;
    gd.BlendState.RenderTarget[0].SrcBlendAlpha = gd.BlendState.RenderTarget[0].DestBlendAlpha = D3D12_BLEND_ONE;
    gd.BlendState.RenderTarget[0].BlendOpAlpha = D3D12_BLEND_OP_ADD;
    gd.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    gd.SampleMask = 0xffffffffu;
    gd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    gd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    gd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    gd.NumRenderTargets = 1;
    gd.RTVFormats[0] = DXGI_FORMAT_R16G16B16A16_FLOAT;
    gd.SampleDesc.Count = 1;
    if (FAILED(g.dev->CreateGraphicsPipelineState(&gd, __uuidof(ID3D12PipelineState), (void**)&gpso))) return false;
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC td = {};
    td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    td.Width = 2560;
    td.Height = 1440;
    td.DepthOrArraySize = 1;
    td.MipLevels = 1;
    td.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    td.SampleDesc.Count = 1;
    td.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    if (FAILED(g.dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &td, D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr,
                                              __uuidof(ID3D12Resource), (void**)&rt)))
      return false;
    D3D12_DESCRIPTOR_HEAP_DESC hd = {};
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    hd.NumDescriptors = 1;
    if (FAILED(g.dev->CreateDescriptorHeap(&hd, __uuidof(ID3D12DescriptorHeap), (void**)&rtv))) return false;
    g.dev->CreateRenderTargetView(rt, nullptr, rtv->GetCPUDescriptorHandleForHeapStart());
    ps = true;
    return true;
  }
  void record(ID3D12GraphicsCommandList* l, int dispatches) {
    if (dispatches <= 0) return;
    if (ps) {
      l->SetGraphicsRootSignature(grs);
      l->SetPipelineState(gpso);
      l->SetGraphicsRoot32BitConstant(0, iters, 0);
      D3D12_CPU_DESCRIPTOR_HANDLE h = rtv->GetCPUDescriptorHandleForHeapStart();
      l->OMSetRenderTargets(1, &h, FALSE, nullptr);
      D3D12_VIEWPORT vp = {0, 0, 2560, 1440, 0, 1};
      D3D12_RECT sc = {0, 0, 2560, 1440};
      l->RSSetViewports(1, &vp);
      l->RSSetScissorRects(1, &sc);
      l->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
      for (int i = 0; i < dispatches; ++i) l->DrawInstanced(3, 1, 0, 0);
      return;
    }
    l->SetDescriptorHeaps(1, &heap);
    l->SetComputeRootSignature(rs);
    l->SetPipelineState(pso);
    const uint32_t c[2] = {n, iters};
    l->SetComputeRoot32BitConstants(0, 2, c, 0);
    l->SetComputeRootDescriptorTable(1, heap->GetGPUDescriptorHandleForHeapStart());
    D3D12_RESOURCE_BARRIER u = {};
    u.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    u.UAV.pResource = buf;
    for (int i = 0; i < dispatches; ++i) {
      l->Dispatch((n + 255) / 256, 1, 1);
      l->ResourceBarrier(1, &u);
    }
  }
};

bool case_async_sim(Gpu& g, const char* dll, const std::string& common, const std::string& mode, int frames, int pre,
                    int post, int iters) {
  Scene s;
  s.tw = 1707, s.th = 960, s.rx = 0, s.ry = 0, s.rw = 1707, s.rh = 960, s.ow = 2560, s.oh = 1440, s.mw = 1707,
  s.mh = 960;
  apply_test_dimensions(&s);
  s.colorState = s.depthState = s.motionState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;  // valid on both queues
  s.make(g);
  s.set_color(g, 1);
  GameLoad load;
  load.iters = (uint32_t)iters;
  if (!load.init(g)) { std::puts("  asyncsim: game load shader failed"); return false; }
  const char* lk = std::getenv("WG_SIM_LOAD");
  if (lk && std::string(lk) == "ps" && !load.init_ps(g)) { std::puts("  asyncsim: raster load failed"); return false; }
  const bool net = mode != "game";
  Ctx c;
  c.g = &g;
  if (net) {
    if (!c.be.load(dll)) return false;
    const std::string opt = "executor=hip;" + common + ";executor=dx12";
    const WgResult r = c.create(s, WG_CREATE_HDR | WG_CREATE_DEPTH_INVERTED | WG_CREATE_AUTO_EXPOSURE, opt.c_str());
    CHECK(r == WG_OK);
    if (r != WG_OK) return false;
  }
  ID3D12CommandQueue* cq = nullptr;
  D3D12_COMMAND_QUEUE_DESC qd = {};
  qd.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
  if (std::getenv("WG_SIM_PRIO")) qd.Priority = D3D12_COMMAND_QUEUE_PRIORITY_HIGH;  // the network's queue high
  g.dev->CreateCommandQueue(&qd, __uuidof(ID3D12CommandQueue), (void**)&cq);
  ID3D12Fence *fpre = nullptr, *fnet = nullptr, *fend = nullptr;
  g.dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence), (void**)&fpre);
  g.dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence), (void**)&fnet);
  g.dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence), (void**)&fend);
  constexpr int kRingN = 4;
  ID3D12CommandAllocator *aa[kRingN], *ab[kRingN], *ac[kRingN];
  ID3D12GraphicsCommandList *la = nullptr, *lb = nullptr, *lc = nullptr;
  for (int i = 0; i < kRingN; ++i) {
    g.dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, __uuidof(ID3D12CommandAllocator), (void**)&aa[i]);
    g.dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, __uuidof(ID3D12CommandAllocator), (void**)&ab[i]);
    g.dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, __uuidof(ID3D12CommandAllocator), (void**)&ac[i]);
  }
  g.dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, aa[0], nullptr, __uuidof(ID3D12GraphicsCommandList),
                           (void**)&la);
  g.dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, ab[0], nullptr, __uuidof(ID3D12GraphicsCommandList),
                           (void**)&lb);
  g.dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, ac[0], nullptr, __uuidof(ID3D12GraphicsCommandList),
                           (void**)&lc);
  la->Close();
  lb->Close();
  lc->Close();
  HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
  const bool async = mode == "ideal" || mode == "pipe" || mode == "inline";
  const int from = frames > 600 ? 301 : 1;
  LARGE_INTEGER w0 = {}, w1 = {}, hz = {};
  QueryPerformanceFrequency(&hz);
  bool ok = true;
  for (int t = 1; t <= frames && ok; ++t) {
    if (t == from) QueryPerformanceCounter(&w0);
    const int k = t % kRingN;
    // frames in flight: at most 3 (the frame that used this ring slot must have ended on both queues)
    for (ID3D12Fence* fw : {fend, fnet}) {
      if (t > 3 && (fw != fnet || async) && fw->GetCompletedValue() < (uint64_t)(t - 3)) {
        fw->SetEventOnCompletion(t - 3, ev);
        if (WaitForSingleObject(ev, 20000) != WAIT_OBJECT_0) { std::puts("  asyncsim HANG"); ok = false; break; }
      }
    }
    aa[k]->Reset();
    ab[k]->Reset();
    ac[k]->Reset();
    la->Reset(aa[k], nullptr);
    WgFrame f = s.frame(g, t);
    f.exposure_mode = WG_EXPOSURE_COMPUTE;
    f.motion_scale[0] = 1.5f, f.motion_scale[1] = -0.75f;
    const float jitter[4][2] = {{0, -1.0f / 6}, {-0.25f, 1.0f / 6}, {0.25f, -7.0f / 18}, {-0.375f, -1.0f / 18}};
    f.jitter[0] = jitter[(t - 1) % 4][0], f.jitter[1] = jitter[(t - 1) % 4][1];
    load.record(la, pre);
    auto netrec = [&](ID3D12GraphicsCommandList* l) {
      f.command_list = l;
      const WgResult rr = c.be.b->dispatch(c.u, &f);
      if (rr != WG_OK) { std::printf("  asyncsim frame %d: dispatch %d\n", t, (int)rr); ok = false; }
    };
    if (!async) {
      if (net) netrec(la);
      if (mode != "net") load.record(la, post);
      la->Close();
      ID3D12CommandList* l = la;
      g.q->ExecuteCommandLists(1, &l);
      g.q->Signal(fend, t);
      continue;
    }
    la->Close();
    ID3D12CommandList* l = la;
    g.q->ExecuteCommandLists(1, &l);
    g.q->Signal(fpre, t);
    lc->Reset(ac[k], nullptr);
    netrec(lc);
    lc->Close();
    if (mode != "ideal") cq->Wait(fpre, t);
    l = lc;
    cq->ExecuteCommandLists(1, &l);
    cq->Signal(fnet, t);
    lb->Reset(ab[k], nullptr);
    load.record(lb, post);
    lb->Close();
    if (mode == "inline") g.q->Wait(fnet, t);
    if (mode == "pipe" && t > 1) g.q->Wait(fnet, t - 1);
    l = lb;
    g.q->ExecuteCommandLists(1, &l);
    g.q->Signal(fend, t);
  }
  if (async) g.q->Wait(fnet, frames);
  g.q->Signal(fend, frames + 1);
  fend->SetEventOnCompletion(frames + 1, ev);
  WaitForSingleObject(ev, 20000);
  QueryPerformanceCounter(&w1);
  const double ms = 1000.0 * (double)(w1.QuadPart - w0.QuadPart) / (double)hz.QuadPart / (frames - from + 1);
  std::printf("  asyncsim %s: %.4f ms per frame over frames %d..%d (pre %d, post %d dispatches, %d ALU steps)\n",
              mode.c_str(), ms, from, frames, pre, post, iters);
  CloseHandle(ev);
  return ok;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf("usage: test_interop_d3d12 <wronggpu_runtime.dll> [case]\n");
    return 2;
  }
  setvbuf(stdout, nullptr, _IONBF, 0);  // unbuffered: a hung run still shows how far it got
  const char* dll = argv[1];
  const std::string only = argc > 2 ? argv[2] : "";
  Gpu g;
  if (!g.init()) {
    std::printf("SKIP: no D3D12 device\n");
    return 0;
  }
  struct Case {
    const char* name;
    std::function<bool()> run;
  } cases[] = {
      {"inline", [&] { return case_sync(g, dll, true); }},
      {"pipelined", [&] { return case_sync(g, dll, false); }},
      {"depth", [&] { return case_view(g, dll, "depth"); }},
      {"motion", [&] { return case_view(g, dll, "motion"); }},
      {"motion_outres", [&] { return case_view(g, dll, "motion_outres"); }},
      {"exposure", [&] { return case_exposure(g, dll); }},
      {"not_ready", [&] { return case_not_ready(g, dll); }},
      {"minimised", [&] { return case_minimised(g, dll, "executor=d3d12-test"); }},
      {"minimised_pipelined", [&] { return case_minimised(g, dll, "executor=d3d12-test;sync=pipelined", true); }},
      {"pipelined_not_ready", [&] { return case_pipelined_not_ready(g, dll); }},
      {"nohip", [&] { return case_create_fails(g, dll, true); }},
      {"noservices", [&] { return case_create_fails(g, dll, false); }},
      {"timing", [&] { return case_timing(g, dll); }},
      {"copies", [&] { return case_sync(g, dll, true, "executor=d3d12-test;input_copy=1;output_copy=1") && copied(); }},
      {"copies_pipelined",
       [&] { return case_sync(g, dll, false, "executor=d3d12-test;sync=pipelined;input_copy=1;output_copy=1") &&
                    copied(); }},
      {"copies_depth", [&] { return case_view(g, dll, "depth", ";input_copy=1;output_copy=1") && copied(); }},
      {"copies_motion", [&] { return case_view(g, dll, "motion", ";input_copy=1;output_copy=1") && copied(); }},
      {"copies_motion_outres", [&] { return case_view(g, dll, "motion_outres", ";input_copy=1;output_copy=1"); }},
      {"copies_depth_dsv", [&] { return case_depth_dsv(g, dll, true); }},
      {"copies_depth_dsv_region", [&] { return case_depth_dsv(g, dll, false); }},
      {"copies_depth_d32s8", [&] { return case_depth_dsv(g, dll, true, true); }},
      {"copies_exposure", [&] { return case_exposure(g, dll, ";input_copy=1;output_copy=1") && copied(); }},
      {"ready_flag", [&] { return case_sync(g, dll, true, "executor=d3d12-test;ready_wait=memory;timing_log=1"); }},
      {"arena", [&] { return case_arena(g, dll, 0, 32); }},
      {"arena_minsize", [&] { return case_arena(g, dll, 100ull << 20, 128); }},
      {"arena_keep", [&] { return case_arena_keep(g, dll); }},
      {"wrapped", [&] { return case_wrapped(g, dll, false, "auto", "underlying device"); }},
      {"wrapped_noopen", [&] { return case_wrapped(g, dll, true, "auto", "underlying device"); }},
      {"wrapped_own", [&] { return case_wrapped(g, dll, false, "own", "own device"); }},
      {"mips", [&] { return case_mips(g, dll); }},
      {"output_nouav", [&] { return case_output_nouav(g, dll); }},
  };
  if (only == "hipsame") {
    const bool ok = case_hip_same(g, dll, argc > 3 ? argv[3] : "", argc > 4 ? argv[4] : "", argc > 5 ? std::atoi(argv[5]) : 6);
    std::puts(ok && !g_fail ? "PASS" : "FAIL");
    return ok && !g_fail ? 0 : 1;
  }
  if (only == "asyncsim") {
    const bool ok = case_async_sim(g, dll, argc > 3 ? argv[3] : "", argc > 4 ? argv[4] : "serial",
                                   argc > 5 ? std::atoi(argv[5]) : 1500, argc > 6 ? std::atoi(argv[6]) : 8,
                                   argc > 7 ? std::atoi(argv[7]) : 4, argc > 8 ? std::atoi(argv[8]) : 32);
    std::puts(ok && !g_fail ? "PASS" : "FAIL");
    return ok && !g_fail ? 0 : 1;
  }
  if (only == "dx12same") {
    const bool outsrv = argc > 5 && std::string(argv[5]) == "outsrv";
    const bool ok = case_dx12_same(g, dll, argc > 3 ? argv[3] : "", argc > 4 ? std::atoi(argv[4]) : 8, outsrv);
    std::puts(ok && !g_fail ? "PASS" : "FAIL");
    return ok && !g_fail ? 0 : 1;
  }
  if (only == "hipswitch") {
    if (argc < 5) { std::puts("hipswitch requires Model M and Model K option strings"); return 2; }
    const bool ok = case_hipswitch(g, dll, argv[3], argv[4]);
    std::puts(ok && !g_fail ? "PASS" : "FAIL");
    return ok && !g_fail ? 0 : 1;
  }
  if (only == "hipleak") {
    const std::string a = argc > 3 ? argv[3] : "", b = argc > 4 && std::string(argv[4]) != "-" ? argv[4] : "";
    auto flag = [&](const char* name) {
      for (int i = 8; i < argc; ++i)
        if (std::string(argv[i]) == name) return true;
      return false;
    };
    const bool ok = case_hipleak(g, dll, a, b, argc > 5 ? std::atoi(argv[5]) : 12, argc > 6 ? std::atoi(argv[6]) : 30,
                                 argc > 7 ? std::atof(argv[7]) : kLeakToleranceMiB,
                                 flag("hold"), flag("thread"));
    std::puts(ok && !g_fail ? "PASS" : "FAIL");
    return ok && !g_fail ? 0 : 1;
  }
  if (only == "hipbench") {
    const int frames = argc > 4 ? std::atoi(argv[4]) : 1500;
    if (frames < 600) { std::puts("hipbench requires >=600 frames (300 warm-up)"); return 1; }
    const bool ok = case_soak_pipe(g, dll, argc > 3 ? argv[3] : "", frames, true);
    std::puts(ok && !g_fail ? "PASS" : "FAIL");
    return ok && !g_fail ? 0 : 1;
  }
  if (only == "soakpipe") {
    const bool ok = case_soak_pipe(g, dll, argc > 3 ? argv[3] : "", argc > 4 ? std::atoi(argv[4]) : 3000);
    std::puts(ok && !g_fail ? "PASS" : "FAIL");
    return ok && !g_fail ? 0 : 1;
  }
  if (only == "soakre") {  // the upscaler destroyed and created again, dispatching right after each create
    const int rounds = argc > 4 ? std::atoi(argv[4]) : 20;
    bool ok = true;
    for (int k = 0; ok && k < rounds; ++k) {
      ok = case_soak(g, dll, argc > 3 ? argv[3] : "", 50, 1);
      std::printf("  re-create round %d/%d: %s%c", k + 1, rounds, ok ? "ok" : "HANG/FAIL", 10);
    }
    std::puts(ok && !g_fail ? "PASS" : "FAIL");
    return ok && !g_fail ? 0 : 1;
  }
  if (only == "soak") {
    const bool ok = case_soak(g, dll, argc > 3 ? argv[3] : "", argc > 4 ? std::atoi(argv[4]) : 3000);
    std::puts(ok && !g_fail ? "PASS" : "FAIL");
    return ok && !g_fail ? 0 : 1;
  }
  if (only == "hip") {
    std::printf("hip\n");
    const bool ok = case_hip(g, dll, argc > 3 ? argv[3] : "");
    std::printf("hip: %s\n%s\n", ok && !g_fail ? "passed" : "FAILED", ok && !g_fail ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return ok && !g_fail ? 0 : 1;
  }
  for (auto& c : cases) {
    if (!only.empty() && only != c.name) continue;
    const int before = g_fail;
    std::printf("%s\n", c.name);
    g_log.clear();
    const bool ran = c.run();
    std::printf("%s: %s\n", c.name, ran && g_fail == before ? "passed" : "FAILED");
    if (!ran && g_fail == before) ++g_fail;
  }
  std::printf("%s\n", g_fail ? "SOME TESTS FAILED" : "ALL TESTS PASSED");
  return g_fail ? 1 : 0;
}
