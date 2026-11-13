// SPDX-License-Identifier: MIT
// End-to-end test of the probe build (WG_RR_PROBE, wg_rr_proxy_probe.dll) with a fake game host on the software
// D3D12 adapter (WARP; no GPU, no game): the probe staged under the game's interposer file name next to the renamed
// fake original; the export table (the release build's names and ordinals); a game-like sequence of calls with
// wrapped device and command list; every RR evaluate returns 0 and writes a plain bilinear upscale of the colour
// input into the output rectangle (compared exactly with a CPU reference), nothing else is written (output outside
// the rectangle and every other tagged input compared before / after); a lifetime-0 colour input copied during its
// tagging call; an unsupported output format refused with 25; forwarding of other features and of graphics entry
// points; the probe log (every field the probe owes, the first 600 frames in full, then changes only); the version
// resource (the game interposer version, WrongGPU's strings); and the load diagnostics (process, path, original
// present, first call, the init export hooked by a stand-in at the first call, every init entry before forwarding).
//   test_probe_dll.exe <stage dir> <proxy file name in the stage> [--imports <file>]
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <fstream>
#include <string>
#include <vector>

#include "../include/rr_proxy_abi.h"
#include "../include/wg_rr_settings_contract.h"
#include "test_util.h"
#include "wg_binding.inc"  // generated from the same binding table as the proxy under test (probe build)
#include "dll_diag.h"

static_assert(WG_BINDING_PROBE == 1, "test_probe_dll needs the probe binding");

enum { R_INIT = 0, R_SHUTDOWN, R_IS_SUPPORTED, R_IS_LOADED, R_SET_LOADED, R_REQUIREMENTS, R_VERSION, R_GET_FUNCTION,
       R_ALLOCATE, R_FREE, R_SET_DEVICE, R_SET_VULKAN_INFO, R_TAG, R_TAG_FOR_FRAME, R_CONSTANTS, R_NEW_FRAME_TOKEN,
       R_EVALUATE, R_UPGRADE_INTERFACE, R_NATIVE_INTERFACE };

struct FoState { int calls; int last_index; const char* last_role; uint32_t feature; uint64_t a1, a3, a5, a6;
                 double a2, a4, a7; uint64_t u64; const void* ptr; void* cmd; uint32_t count; };

static std::wstring widen(const char* s) {
  std::wstring w;
  while (*s) w.push_back((wchar_t)*s++);
  return w;
}

template <typename F>
static F role(HMODULE p, int r) {
  return reinterpret_cast<F>(GetProcAddress(p, kWgRoleNames[r]));
}

// ---------------------------------------------------------------- half floats (exact for the test's values)
static uint16_t to_half(float f) {
  uint32_t x;
  memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000u;
  int32_t e = (int32_t)((x >> 23) & 0xff) - 127 + 15;
  uint32_t m = x & 0x7fffffu;
  if ((x & 0x7fffffffu) == 0) return (uint16_t)sign;
  if (e <= 0) return (uint16_t)sign;  // not used by the test
  return (uint16_t)(sign | ((uint32_t)e << 10) | (m >> 13));
}
static float from_half(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t e = (h >> 10) & 0x1f, m = h & 0x3ffu;
  uint32_t x;
  if (e == 0) x = sign;  // zero (no subnormals in the test)
  else x = sign | ((e - 15 + 127) << 23) | (m << 13);
  float f;
  memcpy(&f, &x, 4);
  return f;
}

// ---------------------------------------------------------------- the fake game host's D3D12 (WARP)
struct Host {
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
    IDXGIAdapter* warp = nullptr;
    HRESULT hr = f->EnumWarpAdapter(__uuidof(IDXGIAdapter), (void**)&warp);
    f->Release();
    if (FAILED(hr)) return false;
    hr = D3D12CreateDevice(warp, D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), (void**)&dev);
    warp->Release();
    if (FAILED(hr)) return false;
    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    if (FAILED(dev->CreateCommandQueue(&qd, __uuidof(ID3D12CommandQueue), (void**)&queue))) return false;
    if (FAILED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, __uuidof(ID3D12CommandAllocator),
                                           (void**)&alloc)))
      return false;
    if (FAILED(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, nullptr,
                                      __uuidof(ID3D12GraphicsCommandList), (void**)&cl)))
      return false;
    cl->Close();
    if (FAILED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence), (void**)&fence))) return false;
    ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    return true;
  }
  void begin() {
    alloc->Reset();
    cl->Reset(alloc, nullptr);
  }
  bool submit() {
    if (FAILED(cl->Close())) return false;
    ID3D12CommandList* l[1] = {cl};
    queue->ExecuteCommandLists(1, l);
    queue->Signal(fence, ++fv);
    fence->SetEventOnCompletion(fv, ev);
    WaitForSingleObject(ev, INFINITE);
    return dev->GetDeviceRemovedReason() == S_OK;
  }
};

static void barrier(ID3D12GraphicsCommandList* cl, ID3D12Resource* r, D3D12_RESOURCE_STATES a,
                    D3D12_RESOURCE_STATES b) {
  if (a == b) return;
  D3D12_RESOURCE_BARRIER br = {};
  br.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  br.Transition.pResource = r;
  br.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
  br.Transition.StateBefore = a;
  br.Transition.StateAfter = b;
  cl->ResourceBarrier(1, &br);
}

struct Tex {
  ID3D12Resource* r = nullptr;
  uint32_t w = 0, h = 0, bpp = 0;
  DXGI_FORMAT fmt = DXGI_FORMAT_UNKNOWN;
  D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;  // the state the game declares in its tag
  std::vector<uint8_t> data;                                   // what the game uploaded (tight rows)
};

static ID3D12Resource* buffer(Host& g, UINT64 size, D3D12_HEAP_TYPE type, D3D12_RESOURCE_STATES st) {
  D3D12_HEAP_PROPERTIES hp = {};
  hp.Type = type;
  D3D12_RESOURCE_DESC d = {};
  d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  d.Width = size;
  d.Height = 1;
  d.DepthOrArraySize = 1;
  d.MipLevels = 1;
  d.SampleDesc.Count = 1;
  d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  ID3D12Resource* r = nullptr;
  g.dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, st, nullptr, __uuidof(ID3D12Resource), (void**)&r);
  return r;
}

static bool make_tex(Host& g, Tex& t, uint32_t w, uint32_t h, DXGI_FORMAT fmt, uint32_t bpp, bool uav,
                     D3D12_RESOURCE_STATES declared, const std::vector<uint8_t>& data) {
  t.w = w; t.h = h; t.bpp = bpp; t.fmt = fmt; t.state = declared; t.data = data;
  D3D12_HEAP_PROPERTIES hp = {};
  hp.Type = D3D12_HEAP_TYPE_DEFAULT;
  D3D12_RESOURCE_DESC d = {};
  d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
  d.Width = w;
  d.Height = h;
  d.DepthOrArraySize = 1;
  d.MipLevels = 1;
  d.Format = fmt;
  d.SampleDesc.Count = 1;
  d.Flags = uav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
  if (FAILED(g.dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                            __uuidof(ID3D12Resource), (void**)&t.r)))
    return false;
  D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp;
  UINT64 total = 0;
  g.dev->GetCopyableFootprints(&d, 0, 1, 0, &fp, nullptr, nullptr, &total);
  ID3D12Resource* up = buffer(g, total, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
  uint8_t* p = nullptr;
  up->Map(0, nullptr, (void**)&p);
  for (uint32_t y = 0; y < h; ++y) memcpy(p + fp.Offset + (size_t)y * fp.Footprint.RowPitch, &data[(size_t)y * w * bpp], (size_t)w * bpp);
  up->Unmap(0, nullptr);
  g.begin();
  D3D12_TEXTURE_COPY_LOCATION dst = {}, src = {};
  dst.pResource = t.r;
  dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  src.pResource = up;
  src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
  src.PlacedFootprint = fp;
  g.cl->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
  barrier(g.cl, t.r, D3D12_RESOURCE_STATE_COPY_DEST, declared);
  bool ok = g.submit();
  up->Release();
  return ok;
}

// Reads the texture back (tight rows); the texture is in its declared state before and after.
static std::vector<uint8_t> read_tex(Host& g, const Tex& t) {
  D3D12_RESOURCE_DESC d = t.r->GetDesc();
  D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp;
  UINT64 total = 0;
  g.dev->GetCopyableFootprints(&d, 0, 1, 0, &fp, nullptr, nullptr, &total);
  ID3D12Resource* rb = buffer(g, total, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
  g.begin();
  barrier(g.cl, t.r, t.state, D3D12_RESOURCE_STATE_COPY_SOURCE);
  D3D12_TEXTURE_COPY_LOCATION dst = {}, src = {};
  dst.pResource = rb;
  dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
  dst.PlacedFootprint = fp;
  src.pResource = t.r;
  src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  g.cl->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
  barrier(g.cl, t.r, D3D12_RESOURCE_STATE_COPY_SOURCE, t.state);
  g.submit();
  std::vector<uint8_t> out((size_t)t.w * t.h * t.bpp);
  uint8_t* p = nullptr;
  rb->Map(0, nullptr, (void**)&p);
  for (uint32_t y = 0; y < t.h; ++y) memcpy(&out[(size_t)y * t.w * t.bpp], p + fp.Offset + (size_t)y * fp.Footprint.RowPitch, (size_t)t.w * t.bpp);
  rb->Unmap(0, nullptr);
  rb->Release();
  return out;
}

// ---------------------------------------------------------------- scene
// 32x18 colour at (left 4, top 2) of a 40x24 R16G16B16A16_FLOAT resource -> 64x36 output at (left 3, top 1) of a
// 70x40 resource. Values are multiples of 1/16 below 4, so the 2x bilinear result (weights 1/4, 3/4) is exact in
// binary16 and the comparison is exact.
static float colour_value(uint32_t x, uint32_t y, uint32_t ch) { return (float)((x * 3 + y * 5 + ch * 7) % 64) / 16.0f; }
static const float kSentinel = 7.5f;

static float ref_upscale(const std::vector<uint8_t>& col, uint32_t cw, uint32_t sx, uint32_t sy, uint32_t sw,
                         uint32_t sh, uint32_t dw, uint32_t dh, uint32_t x, uint32_t y, uint32_t ch) {
  auto texel = [&](int px, int py) {
    px = px < (int)sx ? (int)sx : px > (int)(sx + sw - 1) ? (int)(sx + sw - 1) : px;
    py = py < (int)sy ? (int)sy : py > (int)(sy + sh - 1) ? (int)(sy + sh - 1) : py;
    uint16_t h;
    memcpy(&h, &col[((size_t)py * cw + px) * 8 + ch * 2], 2);
    return from_half(h);
  };
  float s0 = ((float)x + 0.5f) * ((float)sw / (float)dw) - 0.5f;
  float s1 = ((float)y + 0.5f) * ((float)sh / (float)dh) - 0.5f;
  float f0 = floorf(s0), f1 = floorf(s1);
  float t0 = s0 - f0, t1 = s1 - f1;
  int i0 = (int)f0 + (int)sx, i1 = (int)f1 + (int)sy;
  float a = texel(i0, i1), b = texel(i0 + 1, i1), c = texel(i0, i1 + 1), d = texel(i0 + 1, i1 + 1);
  float top = a + t0 * (b - a), bot = c + t0 * (d - c);
  return top + t1 * (bot - top);
}

static std::vector<std::string> read_lines(const std::wstring& path) {
  std::vector<std::string> out;
  std::ifstream f(path.c_str());
  std::string l;
  while (std::getline(f, l)) {
    while (!l.empty() && (l.back() == '\r' || l.back() == '\n')) l.pop_back();
    out.push_back(l);
  }
  return out;
}
static size_t count(const std::vector<std::string>& v, const std::string& needle) {
  size_t n = 0;
  for (const auto& l : v) n += l.find(needle) != std::string::npos;
  return n;
}
static std::string last(const std::vector<std::string>& v, const std::string& needle) {
  for (size_t i = v.size(); i-- > 0;)
    if (v[i].find(needle) != std::string::npos) return v[i];
  return std::string();
}
static bool has(const std::string& line, const std::string& needle) { return line.find(needle) != std::string::npos; }

// The stand-in hook's target (never called: the hook is removed before <init>).
static WgResultCode hooked_init(const WgPreferences*, uint64_t) { return (WgResultCode)99; }

// A9.6 structure GUIDs by name, from the generated binding.
static void set_guid(WgGuid* g, const char* name) {
  for (const WgGuidName& n : kWgGuidNames)
    if (strcmp(n.name, name) == 0) memcpy(g->b, n.b, 16);
}

int main(int argc, char** argv) {
  if (argc < 3) return 2;
  std::string stage = argv[1];
  if (stage.back() != '\\' && stage.back() != '/') stage += '\\';
  const std::wstring wstage = widen(stage.c_str());
  const std::wstring log_path = wstage + L"wronggpu-rr-probe.log";
  DeleteFileW(log_path.c_str());
  const std::wstring proxy_file = widen(argv[2]);
  CHECK(_stricmp(argv[2], kWgInterposerFile) == 0);  // staged under the game's interposer file name
  CHECK(strcmp(kWgProxyFile, "wg_rr_proxy_probe.dll") == 0);
  const std::wstring original_file = proxy_file.substr(0, proxy_file.find_last_of(L'.')) + kWgOriginalFile;

  HMODULE proxy = LoadLibraryW((wstage + proxy_file).c_str());
  CHECK(proxy != nullptr);
  if (!proxy) TEST_SUMMARY("test_probe_dll");
  CHECK(GetModuleHandleW(original_file.c_str()) == nullptr);  // lazy, never under the loader lock
  // the release build's names and ordinals, the own export above them
  for (const WgExportInfo& e : kWgExports)
    CHECK(GetProcAddress(proxy, MAKEINTRESOURCEA(e.ordinal)) == GetProcAddress(proxy, e.name) &&
          GetProcAddress(proxy, e.name) != nullptr);
  CHECK(GetProcAddress(proxy, MAKEINTRESOURCEA(WG_BINDING_OWN_ORDINAL)) ==
        GetProcAddress(proxy, WG_RR_CONTRACT_EXPORT));

  auto init = role<WgResultCode (*)(const WgPreferences*, uint64_t)>(proxy, R_INIT);
  auto shutdown = role<WgResultCode (*)(void)>(proxy, R_SHUTDOWN);
  auto is_supported = role<WgResultCode (*)(uint32_t, const WgAdapterInfo*)>(proxy, R_IS_SUPPORTED);
  auto is_loaded = role<WgResultCode (*)(uint32_t, bool*)>(proxy, R_IS_LOADED);
  auto requirements = role<WgResultCode (*)(uint32_t, WgRequirements*)>(proxy, R_REQUIREMENTS);
  auto version = role<WgResultCode (*)(uint32_t, WgFeatureVersion*)>(proxy, R_VERSION);
  auto get_function = role<WgResultCode (*)(uint32_t, const char*, void**)>(proxy, R_GET_FUNCTION);
  auto allocate = role<WgResultCode (*)(void*, uint32_t, const WgViewport*)>(proxy, R_ALLOCATE);
  auto free_res = role<WgResultCode (*)(uint32_t, const WgViewport*)>(proxy, R_FREE);
  auto set_device = role<WgResultCode (*)(void*)>(proxy, R_SET_DEVICE);
  auto tag = role<WgResultCode (*)(const WgViewport*, const WgTag*, uint32_t, void*)>(proxy, R_TAG);
  auto constants = role<WgResultCode (*)(const WgConstants*, const WgFrameToken*, const WgViewport*)>(proxy, R_CONSTANTS);
  auto new_token = role<WgResultCode (*)(WgFrameToken**, const uint32_t*)>(proxy, R_NEW_FRAME_TOKEN);
  auto evaluate = role<WgResultCode (*)(uint32_t, const WgFrameToken*, const WgHeader* const*, uint32_t, void*)>(
      proxy, R_EVALUATE);
  auto upgrade = role<WgResultCode (*)(void**)>(proxy, R_UPGRADE_INTERFACE);

  // ---- the fake game host on WARP; the "wrapped" device and command list the original hands the game
  Host g;
  bool host_ok = g.create();
  CHECK(host_ok);
  if (!host_ok) TEST_SUMMARY("test_probe_dll (no WARP device)");
  static uint8_t wrapped[2];
  void* wrapped_device = &wrapped[0];
  void* wrapped_cmd = &wrapped[1];

  // ---- version resource: the game interposer version, WrongGPU's own strings
  wgtest::check_version_resource(wstage + proxy_file, L"probe", kWgInterposerFile);

  // ---- <init>: preference flags with bit 7, feature list, the game's SDK version. The first call of all runs while a
  // stand-in hooks the init export (load diagnostics: patched yes, the jump target in this executable).
  {
    wgtest::Hook h = wgtest::hook((void*)init, (const void*)&hooked_init, wgtest::HookForm::kAbs64);
    CHECK_EQ(is_supported(1001, nullptr), 23);  // before <init>
    wgtest::unhook(h);
  }
  HMODULE orig = GetModuleHandleW(original_file.c_str());
  CHECK(orig != nullptr);
  if (!orig) TEST_SUMMARY("test_probe_dll");
  FoState* st = reinterpret_cast<FoState* (*)(void)>(GetProcAddress(orig, "wgtest_fo_state"))();
  auto wrap = reinterpret_cast<void (*)(void*, void*)>(GetProcAddress(orig, "wgtest_fo_wrap"));
  wrap(wrapped_device, g.dev);
  wrap(wrapped_cmd, g.cl);
  static const uint32_t feats[2] = {1001, 0};
  WgPreferences pr;
  memset(&pr, 0, sizeof pr);
  pr.h.version = 1;
  set_guid(&pr.h.type, "preferences");
  pr.features = feats;
  pr.feature_count = 2;
  pr.flags = 1ull | (1ull << 3) | (1ull << 6) | (1ull << 7);
  pr.render_api = 1;
  const uint64_t ver = (2ull << 48) | ((uint64_t)WG_BINDING_API_MINOR << 32) | (1ull << 16) | WG_BINDING_VERSION_MAGIC;
  CHECK_EQ(init(&pr, ver), 0);
  CHECK(strcmp(st->last_role, "init") == 0 && st->ptr == &pr && st->u64 == ver);
  CHECK_EQ(is_supported(1001, nullptr), 0);  // claimed (60.8) without any runtime
  bool loaded = false;
  CHECK_EQ(is_loaded(1001, &loaded), 0);
  CHECK(loaded);
  WgRequirements rq;
  memset(&rq, 0, sizeof rq);
  rq.h.version = 2;
  CHECK_EQ(requirements(1001, &rq), 0);
  CHECK(rq.flags == 2 && rq.required_tag_count == 5);
  WgFeatureVersion fv;
  memset(&fv, 0, sizeof fv);
  fv.h.version = 1;
  CHECK_EQ(version(1001, &fv), 0);
  CHECK_EQ(set_device(wrapped_device), 0);
  CHECK(strcmp(st->last_role, "set-device") == 0 && st->ptr == wrapped_device);
  void *fo = nullptr, *fs = nullptr, *fp = nullptr;
  CHECK_EQ(get_function(1001, kWgFnOptimal, &fo), 0);
  CHECK_EQ(get_function(1001, kWgFnState, &fs), 0);
  CHECK_EQ(get_function(1001, kWgFnOptions, &fp), 0);
  CHECK(fo && fs && fp);
  if (!fo || !fs || !fp) TEST_SUMMARY("test_probe_dll");
  auto rr_optimal = reinterpret_cast<WgResultCode (*)(const WgRrOptions*, WgRrOptimal*)>(fo);
  auto rr_state = reinterpret_cast<WgResultCode (*)(const WgViewport*, WgRrState*)>(fs);
  auto rr_options = reinterpret_cast<WgResultCode (*)(const WgViewport*, const WgRrOptions*)>(fp);

  // ---- resources
  Tex colour, output, normals, motion, depth, albedo, spec;
  std::vector<uint8_t> cd(40 * 24 * 8), od(70 * 40 * 8), nd(32 * 18 * 4), md(32 * 18 * 4), dd(32 * 18 * 4),
      ad(32 * 18 * 4), sd(32 * 18 * 4);
  for (uint32_t y = 0; y < 24; ++y)
    for (uint32_t x = 0; x < 40; ++x)
      for (uint32_t ch = 0; ch < 4; ++ch) {
        uint16_t h = to_half(colour_value(x, y, ch));
        memcpy(&cd[((size_t)y * 40 + x) * 8 + ch * 2], &h, 2);
      }
  for (size_t i = 0; i < od.size(); i += 2) {
    uint16_t h = to_half(kSentinel);
    memcpy(&od[i], &h, 2);
  }
  for (size_t i = 0; i < nd.size(); i += 4) {
    uint32_t v = 0u | (1023u << 10) | (1023u << 20) | (0u << 30);  // R10G10B10A2_UNORM (0, 1, 1, 0)
    memcpy(&nd[i], &v, 4);
  }
  for (size_t i = 0; i < md.size(); ++i) md[i] = (uint8_t)(i * 7 + 1);
  for (size_t i = 0; i < dd.size(); ++i) dd[i] = (uint8_t)(i * 13 + 5);
  for (size_t i = 0; i < ad.size(); ++i) ad[i] = (uint8_t)(i * 3 + 9);
  for (size_t i = 0; i < sd.size(); ++i) sd[i] = (uint8_t)(i * 11 + 2);
  const D3D12_RESOURCE_STATES SRV = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
  const D3D12_RESOURCE_STATES SRV_ALL = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE |
                                        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
  bool made = make_tex(g, colour, 40, 24, DXGI_FORMAT_R16G16B16A16_FLOAT, 8, false, D3D12_RESOURCE_STATE_COPY_SOURCE, cd) &&
              make_tex(g, output, 70, 40, DXGI_FORMAT_R16G16B16A16_FLOAT, 8, true, D3D12_RESOURCE_STATE_COMMON, od) &&
              make_tex(g, normals, 32, 18, DXGI_FORMAT_R10G10B10A2_UNORM, 4, false, SRV, nd) &&
              make_tex(g, motion, 32, 18, DXGI_FORMAT_R16G16_FLOAT, 4, false, SRV_ALL, md) &&
              make_tex(g, depth, 32, 18, DXGI_FORMAT_R32_FLOAT, 4, false, SRV, dd) &&
              make_tex(g, albedo, 32, 18, DXGI_FORMAT_R8G8B8A8_UNORM, 4, false, SRV, ad) &&
              make_tex(g, spec, 32, 18, DXGI_FORMAT_R8G8B8A8_UNORM, 4, false, SRV, sd);
  CHECK(made);
  if (!made) TEST_SUMMARY("test_probe_dll");

  static WgViewport vp;
  memset(&vp, 0, sizeof vp);
  memcpy(vp.h.type.b, kWgGuid_viewport, 16);
  vp.h.version = 1;
  vp.id = 0;
  WgRrOptions op;
  memset(&op, 0, sizeof op);
  op.h.version = 3;
  set_guid(&op.h.type, "rr_options");
  op.mode = 1;  // max performance: 1/2 per axis
  op.output_width = 64;
  op.output_height = 36;
  op.colour_is_hdr = 1;
  op.normal_roughness_packing = 1;
  for (int i = 0; i < 16; ++i) op.world_to_view[i] = (i % 5 == 0) ? 1.0f : 0.0f;
  WgRrOptimal os;
  memset(&os, 0, sizeof os);
  os.h.version = 1;
  CHECK_EQ(rr_optimal(&op, &os), 0);
  CHECK(os.optimal_width == 32 && os.optimal_height == 18 && os.min_width == 21 && os.max_width == 64);
  CHECK_EQ(rr_options(&vp, &op), 0);
  WgRrState rs;
  memset(&rs, 0, sizeof rs);
  CHECK_EQ(rr_state(&vp, &rs), 0);
  CHECK_EQ(allocate(wrapped_cmd, 1001, &vp), 0);

  struct In { Tex* t; uint32_t type; WgExtent e; };
  const In ins[] = {{&colour, 3, {2, 4, 32, 18}}, {&output, 4, {1, 3, 64, 36}}, {&normals, 14, {0, 0, 0, 0}},
                    {&motion, 1, {0, 0, 0, 0}}, {&depth, 0, {0, 0, 0, 0}}, {&albedo, 7, {0, 0, 0, 0}},
                    {&spec, 8, {0, 0, 0, 0}}};
  const size_t nin = sizeof ins / sizeof ins[0];
  static WgResource res[16];
  static WgTag tags[16];
  for (size_t i = 0; i < nin; ++i) {
    memset(&res[i], 0, sizeof res[i]);
    res[i].h.version = 1;
    res[i].kind = WG_RES_TEX2D;
    res[i].native = ins[i].t->r;
    res[i].state = (uint32_t)ins[i].t->state;
    res[i].width = ins[i].t->w;
    res[i].height = ins[i].t->h;
    res[i].native_format = (uint32_t)ins[i].t->fmt;
    res[i].mip_levels = 1;
    res[i].array_layers = 1;
    memset(&tags[i], 0, sizeof tags[i]);
    memcpy(tags[i].h.type.b, kWgGuid_tag, 16);
    tags[i].h.version = 1;
    tags[i].resource = &res[i];
    tags[i].buffer_type = ins[i].type;
    tags[i].lifetime = WG_LIFE_UNTIL_PRESENT;
    tags[i].extent = ins[i].e;
  }
  g.begin();
  CHECK_EQ(tag(&vp, tags, (uint32_t)nin, wrapped_cmd), 0);
  CHECK(strcmp(st->last_role, "tag") == 0 && st->ptr == tags && st->count == nin);
  CHECK(g.submit());

  // ---- frames: 620 (the first 600 logged in full); the output compared on frames 1 and 3 (lifetime-0 copy)
  const float jit[8][2] = {{0.25f, -0.125f}, {-0.25f, 0.125f}, {0.125f, 0.375f}, {-0.375f, -0.25f},
                           {0.0625f, 0.0f}, {-0.0625f, 0.4375f}, {0.3125f, -0.4375f}, {-0.1875f, 0.1875f}};
  int eval_ok = 0, frames_ok = 0;
  bool pixels_checked = false, copy_checked = false;
  for (uint32_t f = 1; f <= 620; ++f) {
    g.begin();
    WgFrameToken* tok = nullptr;
    uint32_t idx = f;
    CHECK_EQ(new_token(&tok, &idx), 0);
    if (f == 3) {
      // the colour input valid only during its tagging call: the probe copies it on this call's command list
      tags[0].lifetime = WG_LIFE_ONLY_DURING_TAG;
      CHECK_EQ(tag(&vp, &tags[0], 1, wrapped_cmd), 0);
    }
    WgConstants c;
    memset(&c, 0, sizeof c);
    c.h.version = 2;
    set_guid(&c.h.type, "constants");
    c.view_to_clip[0] = 1.0f;
    c.view_to_clip[5] = 1.5f;
    c.view_to_clip[8] = 0.0f;
    c.view_to_clip[9] = 0.0f;
    c.jitter[0] = jit[f % 8][0];
    c.jitter[1] = jit[f % 8][1];
    c.mv_scale[0] = 1.0f / 32.0f;
    c.mv_scale[1] = 1.0f / 18.0f;
    c.camera_motion_included = 1;
    c.mv_jittered = 0;
    c.depth_inverted = 1;
    c.mv_3d = 0;
    c.mv_dilated = 0;
    c.reset = f == 1 ? 1 : 0;
    CHECK_EQ(constants(&c, tok, &vp), 0);
    const WgHeader* in[1] = {&vp.h};
    WgResultCode rc = evaluate(1001, tok, in, 1, wrapped_cmd);
    eval_ok += rc == 0;
    frames_ok += g.submit();
    if (f == 1 || f == 3) {
      std::vector<uint8_t> o = read_tex(g, output);
      size_t inside = 0, inside_ok = 0, outside = 0, outside_ok = 0;
      for (uint32_t y = 0; y < 40; ++y)
        for (uint32_t x = 0; x < 70; ++x)
          for (uint32_t ch = 0; ch < 4; ++ch) {
            uint16_t h;
            memcpy(&h, &o[((size_t)y * 70 + x) * 8 + ch * 2], 2);
            float v = from_half(h);
            if (x >= 3 && x < 67 && y >= 1 && y < 37) {
              inside++;
              inside_ok += v == ref_upscale(cd, 40, 4, 2, 32, 18, 64, 36, x - 3, y - 1, ch);
            } else {
              outside++;
              outside_ok += v == kSentinel;
            }
          }
      CHECK_EQ(inside, (size_t)64 * 36 * 4);
      CHECK_EQ(inside_ok, inside);    // the bilinear upscale, exact
      CHECK_EQ(outside_ok, outside);  // nothing written outside the output rectangle
      if (f == 1) pixels_checked = inside_ok == inside;
      else copy_checked = inside_ok == inside;
    }
  }
  CHECK_EQ(eval_ok, 620);
  CHECK_EQ(frames_ok, 620);
  CHECK(pixels_checked && copy_checked);
  CHECK(strcmp(st->last_role, "evaluate") != 0);  // RR is never forwarded
  // read-only on every other game resource: every input unchanged (and back in its declared state)
  CHECK(read_tex(g, colour) == colour.data);
  CHECK(read_tex(g, normals) == normals.data);
  CHECK(read_tex(g, motion) == motion.data);
  CHECK(read_tex(g, depth) == depth.data);
  CHECK(read_tex(g, albedo) == albedo.data);
  CHECK(read_tex(g, spec) == spec.data);

  // ---- an output format of no supported class (62.5): 25, logged, the output untouched
  {
    Tex bad;
    std::vector<uint8_t> bd(70 * 40 * 4, 0x5a);
    CHECK(make_tex(g, bad, 70, 40, DXGI_FORMAT_R32_UINT, 4, true, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, bd));
    WgResource br = res[1];
    br.native = bad.r;
    br.native_format = DXGI_FORMAT_R32_UINT;
    br.state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    WgTag bt = tags[1];
    bt.resource = &br;
    g.begin();
    WgFrameToken* tok = nullptr;
    CHECK_EQ(new_token(&tok, nullptr), 0);
    const WgHeader* in[2] = {&vp.h, &bt.h};  // as a local tag of this evaluate
    CHECK_EQ(evaluate(1001, tok, in, 2, wrapped_cmd), 25);
    CHECK(g.submit());
    CHECK(read_tex(g, bad) == bd);
    bad.r->Release();
  }

  // ---- everything else forwarded: another feature, a graphics entry point, the interface upgrade
  {
    const WgHeader* in[1] = {&vp.h};
    CHECK_EQ(evaluate(0, nullptr, in, 1, wrapped_cmd), 100);
    CHECK(strcmp(st->last_role, "evaluate") == 0 && st->feature == 0 && st->cmd == wrapped_cmd);
    typedef uint64_t (*Generic)(uint64_t, double, uint64_t, double, uint64_t, uint64_t, double);
    const char* gname = kWgForwardNames[0];
    Generic gf = reinterpret_cast<Generic>(GetProcAddress(proxy, gname));
    uint64_t r1 = gf(5, 1.5, 6, 2.5, 7, 8, 9.5);
    CHECK(st->a1 == 5 && st->a2 == 1.5 && st->a7 == 9.5 && r1 == 5 * 31u + (uint64_t)st->last_index);
    uint64_t r2 = gf(11, 0.5, 6, 2.5, 7, 8, 9.5);  // the second call (resolved slot) is noted as well
    CHECK(st->a1 == 11 && r2 == 11 * 31u + (uint64_t)st->last_index);
    void* iface = nullptr;
    upgrade(&iface);
    CHECK(strcmp(st->last_role, "generic") == 0 && st->a1 == (uint64_t)(uintptr_t)&iface);
    CHECK_EQ(free_res(1001, &vp), 0);
  }

  // ---- settings contract: the probe claims "ready, model loaded" (no controls change anything in the probe)
  auto get_contract = reinterpret_cast<WgRrGetSettingsContractFn>(GetProcAddress(proxy, WG_RR_CONTRACT_EXPORT));
  CHECK(get_contract && get_contract()->status && (get_contract()->status() & 3u) == 3u);
  CHECK_EQ(shutdown(), 0);
  FreeLibrary(proxy);
  CHECK(GetModuleHandleW(proxy_file.c_str()) == nullptr);

  // ---- the probe log: names and numbers only, every field the probe owes
  std::vector<std::string> L = read_lines(log_path);
  CHECK(L.size() > 1000);
  size_t bad_chars = 0, bytes = 0;
  for (const auto& l : L) {
    bytes += l.size() + 2;
    for (char ch : l) bad_chars += (ch < 0x20 || ch > 0x7e);
  }
  CHECK_EQ(bad_chars, (size_t)0);
  CHECK(bytes < 20u * 1024 * 1024);
  CHECK(count(L, "probe start: build Streamline 2.") == 1);
  wgtest::check_load_diagnostics(L, wstage + proxy_file, original_file, kWgRoleNames[R_IS_SUPPORTED], "abs64 jmp",
                                 "test_probe_dll.exe", ver, "call init: rc=0 ; seq=");
  CHECK(wgtest::first_index(L, "probe start: ") < wgtest::first_index(L, "load: process "));
  std::string l = last(L, " init: sdk=");
  CHECK(has(l, "bit7_frame_tagging=1") && has(l, "bit0_no_state_tracking=1") && has(l, "features=[1001,0]") &&
        has(l, "magic=0xfedc") && has(l, "flags=0xc9"));
  CHECK(count(L, "chain init.preferences: preferences{1ca10965-bf8e-432b-8da1-6716d879fb14} v1") == 1);
  l = last(L, "call init: rc=0 ; seq=");
  CHECK(has(l, " tid="));
  CHECK(count(L, std::string("call get-function feature=1001 name=") + kWgFnOptimal + ": rc=0") == 1);
  CHECK(count(L, std::string("call get-function feature=1001 name=") + kWgFnOptions + ": rc=0") == 1);
  CHECK(count(L, "first requirements: seq=") == 1 && count(L, "first version: seq=") == 1);
  CHECK(count(L, "first allocate: seq=") == 1 && count(L, "first free: seq=") == 1);
  CHECK(count(L, "first tag-for-frame") == 0);  // not called in this run
  l = last(L, "frame-token first: ");
  CHECK(has(l, "matches_frame_token_guid=1") && has(l, "bytes_at_16=350f0a8384db7141a80459b206499b18") &&
        has(l, "as_guid=frame_token{830a0f35-db84-4171-a804-59b206499b18}"));
  l = last(L, " constants vp=0: ");
  CHECK(has(l, "camera_motion_included=1 mv_jittered=0 depth_inverted=1 mv_3d=0 mv_dilated=0") &&
        has(l, "mv_scale=0.03125,0.055555556") && has(l, "frame_index="));
  CHECK(count(L, "chain constants: constants{dcd35ad7-4e4a-4bad-a90c-e0c49eb23afe} v2") >= 1);
  l = last(L, " tag vp=0 type=4: ");
  CHECK(has(l, "lifetime=1 extent=1,3,64,36 kind=0 tag_format=10 res=70x40 state=0x0") &&
        has(l, "desc_format=10 desc=70x40 desc_dim=3 desc_samples=1 desc_flags=0x4"));
  l = last(L, " tag vp=0 type=14: ");
  CHECK(has(l, "tag_format=24 res=32x18 state=0x40"));
  CHECK(count(L, "lifetime-0 copy tag vp=0 type=3: copied") == 1);
  l = last(L, " evaluate vp=0: ");
  CHECK(has(l, "render=32x18 output=64x36 output_rect=3,1,64,36 mode=1") && has(l, "normal_roughness_mode=1") &&
        has(l, "normal_type=14 normal_format=24 roughness_format=in-normal"));
  l = last(L, " rr-options vp=0: mode=");
  CHECK(has(l, "mode=1 output=64x36") && has(l, "version=3 alpha_upscaling=0 presets=0,0,0,0,0,0 release_rc=0"));
  CHECK(count(L, "rr-optimal mode=1 in=64x36 version=3: rc=0 optimal=32x18 min=21x12 max=64x36 sharpness=0") == 1);
  CHECK(count(L, "colour-extent-watch: after=rr-options evaluate=1/2 extent=2,4,32,18 res=40x24 render=32x18") == 0);
  CHECK(count(L, "colour-extent-watch: after=allocate evaluate=1/2 extent=2,4,32,18 res=40x24 render=32x18") == 1);
  CHECK(count(L, "colour-extent-watch: after=allocate evaluate=2/2") == 1);
  CHECK(count(L, "first rr-options: seq=") == 1 && count(L, "first allocate: seq=") == 1);
  l = last(L, "camera vp=0: v2c20=0 v2c21=0 prev_v2c20=0 prev_v2c21=0");
  CHECK(has(l, "; jitter=") && has(l, " prev_jitter="));
  CHECK(count(L, "camera vp=0: v2c20=0 v2c21=0 prev_v2c20=- prev_v2c21=-") == 1);
  CHECK(count(L, "normal-patch sample=0: eval=60 type=14 format=24 origin=8,1 size=16x16 mean=0,1,1,0 "
                 "world_to_view=1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1") == 1);
  CHECK(count(L, "normal-patch sample=1: eval=80") == 1);
  CHECK(count(L, "upscale vp=0: rc=25 unsupported output format 42") == 1);
  CHECK(count(L, std::string("call ") + kWgForwardNames[0] + ": forwarded (result not seen)") == 1);
  CHECK(count(L, "call upgrade-interface: rc=") == 1);
  CHECK(count(L, "call evaluate feature=0: rc=100") == 1);
  l = last(L, "summary at shutdown: ");
  CHECK(has(l, "allocate=1 free=1 tag=2 tag-for-frame=0 requirements=1 version=1") && has(l, "rr_evaluates=621"));
  // the first 600 frames in full, then changes only
  CHECK_EQ(count(L, "call evaluate feature=1001: rc=0"), (size_t)600);
  CHECK_EQ(count(L, "call new-frame-token: rc=0"), (size_t)600);  // the token calls of frames 1..600
  CHECK(count(L, "f=601 call evaluate") == 0 && count(L, "f=620 ") == 0);
  CHECK(count(L, "f=621 call evaluate feature=1001: rc=25") == 1);  // a change of result is logged again
  TEST_SUMMARY("test_probe_dll");
}
