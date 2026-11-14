// SPDX-License-Identifier: MIT
// End-to-end test of the hook-mode delivery (src/hook_dll.cpp, WrongGPU-RR.asi / WrongGPU-RR-probe.asi). The test
// executable IS the fake game: it statically imports the RR path's functions by name from a stand-in interface DLL
// (the game's interposer file name, the real export names, from tests/gen_fake_original.py), so the loader fills its
// import address table with the stand-in's exports before anything else runs. A stand-in "OptiScaler" models an
// inline patch on the interface's init export (wgtest_fo_set_opti_init): the init body calls the patch first, so a
// forwarder that reaches the export through its address keeps the patch in the chain. The test then loads the ASI
// (as OptiScaler would, after the interface is already loaded) and calls InitializeASI.
//
// Checks: the game interface is found already loaded; the version resource gates patching (a wrong version patches
// nothing, RR stays with the stand-in); on a match, the RR path reaches our core and the feature functions, non-RR
// traffic reaches the stand-in (and through it OptiScaler's init patch), and only the RR-role IAT slots were
// redirected. Release build: the RR path runs into a fake runtime. Probe build: a short sequence on the software
// D3D12 adapter (WARP, with the D3D12 debug layer when installed: no error message allowed) with every RR evaluate
// returning 0, and the capture option (CAPTURE.md) switched on by wronggpu-rr-probe.ini next to the exe: evaluates
// 4, 6, 8 captured, the input files compared byte for byte with what was uploaded, frame.json read, an earlier
// capture folder renamed (kept), the output not captured.
// Built with WG_TEST_DELAYLOAD (and linked /DELAYLOAD:<interface file>), the fake game instead links the interface by
// delay-load import, as some games do: the interface is not loaded when the ASI runs, so the ASI must load the file
// next to the exe and patch the delay-load IAT slots.
//   test_hook_dll.exe <stage dir> [--badver]
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <fstream>
#include <string>
#include <vector>

#include "wg_binding.inc"  // hook binding (same table and version as the ASI under test); defines WG_BINDING_PROBE

#if WG_BINDING_PROBE
#include <d3d12.h>
#include <dxgi1_4.h>
#endif

#include "../include/rr_proxy_abi.h"
#include "../include/wg_rr_settings_contract.h"
#include "../include/wg_host_menu_abi.h"
#include "../../../wgshared/include/wg_shared_settings.h"  // WG_SS_NAME_FMT (the --fast-noblock run takes the name)
#include "test_util.h"

#if !WG_BINDING_PROBE
#include "fake_runtime.h"
#include "mock_ngx_params.h"
#endif

static_assert(WG_BINDING_HOOK == 1, "test_hook_dll needs the hook binding");

#ifndef WG_TEST_DELAYLOAD
#define WG_TEST_DELAYLOAD 0
#endif

// The RR path functions the fake game imports from the interface DLL (resolved through the IAT the ASI patches).
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
WgResultCode slAllocateResources(void*, uint32_t, const WgViewport*);
// A non-RR API export the game also imports; the release build does not hook it, so its IAT slot must be untouched.
WgResultCode slUpgradeInterface(void**);
}

struct FoState { int calls; int last_index; const char* last_role; uint32_t feature; uint64_t a1, a3, a5, a6;
                 double a2, a4, a7; uint64_t u64; const void* ptr; void* cmd; uint32_t count; };

static std::wstring widen(const char* s) {
  std::wstring w;
  while (*s) w.push_back((wchar_t)(unsigned char)*s++);
  return w;
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

static int g_opti_init_seen = 0;
static void opti_init_patch() { ++g_opti_init_seen; }  // stands in for OptiScaler's inline patch on the init export

#if WG_BINDING_PROBE
// ---- half floats and a tiny WARP host (as test_probe_dll) ----
static uint16_t to_half(float f) {
  uint32_t x; memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000u;
  int32_t e = (int32_t)((x >> 23) & 0xff) - 127 + 15;
  uint32_t m = x & 0x7fffffu;
  if ((x & 0x7fffffffu) == 0) return (uint16_t)sign;
  if (e <= 0) return (uint16_t)sign;
  return (uint16_t)(sign | ((uint32_t)e << 10) | (m >> 13));
}
struct Host {
  ID3D12Device* dev = nullptr; ID3D12CommandQueue* queue = nullptr; ID3D12CommandAllocator* alloc = nullptr;
  ID3D12GraphicsCommandList* cl = nullptr; ID3D12Fence* fence = nullptr; UINT64 fv = 0; HANDLE ev = nullptr;
  ID3D12InfoQueue* iq = nullptr;  // debug layer messages (null when the debug layer is not installed)
  bool create() {
    ID3D12Debug* dbg = nullptr;
    if (SUCCEEDED(D3D12GetDebugInterface(__uuidof(ID3D12Debug), (void**)&dbg))) { dbg->EnableDebugLayer(); dbg->Release(); }
    IDXGIFactory4* f = nullptr;
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory4), (void**)&f))) return false;
    IDXGIAdapter* warp = nullptr;
    HRESULT hr = f->EnumWarpAdapter(__uuidof(IDXGIAdapter), (void**)&warp); f->Release();
    if (FAILED(hr)) return false;
    hr = D3D12CreateDevice(warp, D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), (void**)&dev); warp->Release();
    if (FAILED(hr)) return false;
    if (FAILED(dev->QueryInterface(__uuidof(ID3D12InfoQueue), (void**)&iq))) iq = nullptr;
    D3D12_COMMAND_QUEUE_DESC qd = {}; qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    if (FAILED(dev->CreateCommandQueue(&qd, __uuidof(ID3D12CommandQueue), (void**)&queue))) return false;
    if (FAILED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, __uuidof(ID3D12CommandAllocator), (void**)&alloc))) return false;
    if (FAILED(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, nullptr, __uuidof(ID3D12GraphicsCommandList), (void**)&cl))) return false;
    cl->Close();
    if (FAILED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence), (void**)&fence))) return false;
    ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    return true;
  }
  void begin() { alloc->Reset(); cl->Reset(alloc, nullptr); }
  bool submit() {
    if (FAILED(cl->Close())) return false;
    ID3D12CommandList* l[1] = {cl}; queue->ExecuteCommandLists(1, l);
    queue->Signal(fence, ++fv); fence->SetEventOnCompletion(fv, ev); WaitForSingleObject(ev, INFINITE);
    return dev->GetDeviceRemovedReason() == S_OK;
  }
};
static void barrier(ID3D12GraphicsCommandList* cl, ID3D12Resource* r, D3D12_RESOURCE_STATES a, D3D12_RESOURCE_STATES b) {
  if (a == b) return;
  D3D12_RESOURCE_BARRIER br = {}; br.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; br.Transition.pResource = r;
  br.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES; br.Transition.StateBefore = a; br.Transition.StateAfter = b;
  cl->ResourceBarrier(1, &br);
}
static ID3D12Resource* make_tex(Host& g, uint32_t w, uint32_t h, DXGI_FORMAT fmt, bool uav, D3D12_RESOURCE_STATES st) {
  D3D12_HEAP_PROPERTIES hp = {}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
  D3D12_RESOURCE_DESC d = {}; d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D; d.Width = w; d.Height = h;
  d.DepthOrArraySize = 1; d.MipLevels = 1; d.Format = fmt; d.SampleDesc.Count = 1;
  d.Flags = uav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
  ID3D12Resource* r = nullptr;
  g.dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, st, nullptr, __uuidof(ID3D12Resource), (void**)&r);
  return r;
}
// Upload `rows` rows of `row_bytes` bytes into subresource 0 of `tex` (in state `st`, returned to it) on g.cl; the upload
// buffer is returned for release after the submission.
static ID3D12Resource* upload_tex(Host& g, ID3D12Resource* tex, const void* data, uint32_t row_bytes, uint32_t rows,
                                  D3D12_RESOURCE_STATES st) {
  D3D12_RESOURCE_DESC d = tex->GetDesc();
  D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp;
  UINT64 total = 0;
  g.dev->GetCopyableFootprints(&d, 0, 1, 0, &fp, nullptr, nullptr, &total);
  D3D12_HEAP_PROPERTIES hp = {}; hp.Type = D3D12_HEAP_TYPE_UPLOAD;
  D3D12_RESOURCE_DESC bd = {}; bd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; bd.Width = total; bd.Height = 1;
  bd.DepthOrArraySize = 1; bd.MipLevels = 1; bd.SampleDesc.Count = 1; bd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  ID3D12Resource* up = nullptr;
  if (FAILED(g.dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                            __uuidof(ID3D12Resource), (void**)&up))) return nullptr;
  uint8_t* m = nullptr;
  up->Map(0, nullptr, (void**)&m);
  for (uint32_t y = 0; y < rows; ++y)
    memcpy(m + fp.Offset + (size_t)y * fp.Footprint.RowPitch, (const uint8_t*)data + (size_t)y * row_bytes, row_bytes);
  up->Unmap(0, nullptr);
  barrier(g.cl, tex, st, D3D12_RESOURCE_STATE_COPY_DEST);
  D3D12_TEXTURE_COPY_LOCATION dst = {}, src = {};
  dst.pResource = tex; dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  src.pResource = up; src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; src.PlacedFootprint = fp;
  g.cl->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
  barrier(g.cl, tex, D3D12_RESOURCE_STATE_COPY_DEST, st);
  return up;
}
static std::vector<uint8_t> read_file(const std::wstring& path) {
  std::vector<uint8_t> b;
  FILE* f = _wfopen(path.c_str(), L"rb");
  if (!f) return b;
  uint8_t buf[65536];
  size_t n;
  while ((n = fread(buf, 1, sizeof buf, f)) > 0) b.insert(b.end(), buf, buf + n);
  fclose(f);
  return b;
}
static uint32_t rd32(const std::vector<uint8_t>& b, size_t at) {
  uint32_t v = 0;
  if (at + 4 <= b.size()) memcpy(&v, b.data() + at, 4);
  return v;
}
// A captured input file (CAPTURE.md): header fields and the data rows against what was uploaded.
static bool check_capture_file(const std::wstring& path, uint32_t type, uint32_t fmt, uint32_t w, uint32_t h,
                               uint32_t row_bytes, const void* data, uint32_t state, uint32_t ext_left,
                               uint32_t ext_top) {
  std::vector<uint8_t> b = read_file(path);
  if (b.size() != 64 + (size_t)row_bytes * h) {
    fprintf(stderr, "capture file size %zu (type %u)\n", b.size(), type);
    return false;
  }
  bool ok = memcmp(b.data(), "WGRRCAP1", 8) == 0 && rd32(b, 8) == 64 && rd32(b, 12) == type && rd32(b, 16) == fmt &&
            rd32(b, 20) == w && rd32(b, 24) == h && rd32(b, 28) == row_bytes && rd32(b, 36) == state &&
            rd32(b, 40) == ext_left && rd32(b, 44) == ext_top && rd32(b, 60) == 0;
  if (!ok) fprintf(stderr, "capture header mismatch (type %u)\n", type);
  if (ok && data && memcmp(b.data() + 64, data, (size_t)row_bytes * h) != 0) {
    fprintf(stderr, "capture data mismatch (type %u)\n", type);
    ok = false;
  }
  return ok;
}
static bool exists(const std::wstring& p) { return GetFileAttributesW(p.c_str()) != INVALID_FILE_ATTRIBUTES; }
static std::string text_of(const std::vector<uint8_t>& b) { return std::string(b.begin(), b.end()); }

static void set_guid(WgGuid* gg, const char* name) {
  for (const WgGuidName& n : kWgGuidNames) if (strcmp(n.name, name) == 0) memcpy(gg->b, n.b, 16);
}
#endif

#if !WG_BINDING_PROBE
// ---- a stand-in for OptiScaler's menu table (wg_host_menu_abi.h): records what the WrongGPU tab draws
namespace menu {
std::string tab_name, card_title, reported;
WgHostDrawFn draw = nullptr;
int texts = 0, combos = 0, combo_count = 0, combo_index = -1, pick = -1, reports = 0, sections = 0;
std::vector<std::string> items;
bool register_tab(WgHostMenuPlugin*, const char* name, const char*, WgHostDrawFn fn, void*) {
  tab_name = name;
  draw = fn;
  return true;
}
bool register_custom_card(WgHostMenuPlugin*, const char* title, WgHostDrawFn fn, void*) {
  card_title = title;
  draw = fn;
  return true;
}
void report_upscaler(WgHostMenuPlugin*, const char* name, float) {
  reported = name;
  ++reports;
}
void section(const char*) { ++sections; }
void text(const char*) { ++texts; }
void nop_text(const char*) {}
// plugin interface 5 (clean/rr-re9-pt): the host brackets RR work on the game's list
int cl_begins = 0, cl_ends = 0;
void* cl_last = nullptr;
void cl_begin(WgHostMenuPlugin*, void* cl) { ++cl_begins; cl_last = cl; }
void cl_end(WgHostMenuPlugin*, void* cl) { ++cl_ends; cl_last = cl; }
void nop() {}
void nop_b(bool) {}
// plugin interface 4 (clean/rr-ngx): the NGX RR provider registration, one provider per process as the host does
int ngx_registers = 0;
WgHostNgxRr nprov{};
bool register_ngx(WgHostMenuPlugin* plugin, const WgHostNgxRr* p) {
  ++ngx_registers;
  if (!plugin || !p || p->size < sizeof(WgHostNgxRr) || !p->create || !p->evaluate || !p->release || nprov.create)
    return false;
  nprov = *p;
  return true;
}
bool combo(const char*, int32_t* current, const char* const* list, int32_t count) {
  ++combos;
  combo_count = count;
  combo_index = *current;
  items.assign(list, list + count);
  if (pick >= 0 && pick < count) {  // the player picks an entry
    *current = pick;
    pick = -1;
    return true;
  }
  return false;
}
void reset() { texts = combos = combo_count = sections = 0; combo_index = -1; items.clear(); }
WgHostMenuApi table() {
  WgHostMenuApi a = {};
  a.size = sizeof a;
  a.version = 2;
  a.register_tab = register_tab;
  a.register_custom_card = register_custom_card;
  a.report_upscaler = report_upscaler;
  a.section = section;
  a.separator = nop;
  a.same_line = nop;
  a.spacing = nop;
  a.disabled = nop_b;
  a.item_tooltip = nop_text;
  a.text = text;
  a.text_disabled = text;
  a.combo = combo;
  return a;
}
}  // namespace menu
static std::string file_text(const std::wstring& p) {
  std::ifstream f(p, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}
static void write_text(const std::wstring& p, const char* s) {
  FILE* f = _wfopen(p.c_str(), L"wb");
  if (f) { fputs(s, f); fclose(f); }
}
#endif

int main(int argc, char** argv) {
  if (argc < 2) return 2;
  bool badver = false, exact = false, exact_denied = false, rr_off = false, rr_off_own = false;
  // clean/060-rr-plugin-hunt: where rr_mode comes from (fresh install: WrongGPU\wronggpu.ini only; the same with the shared
  // settings block unavailable; the legacy wronggpu-rr.ini line of an older install)
  bool fast = false, fast_noblock = false, legacy_mode = false;
  bool log_cap = false;  // wronggpu-rr.log already at its size cap when the plugin starts: moved to .log.old
  for (int i = 2; i < argc; ++i) {
    if (strcmp(argv[i], "--badver") == 0) badver = true;
    if (strcmp(argv[i], "--exact") == 0) exact = true;              // the launcher allowed exact: three modes
    if (strcmp(argv[i], "--exact-denied") == 0) exact_denied = true;  // rr_mode=exact without the permission
    if (strcmp(argv[i], "--rr-off") == 0) rr_off = true;            // launcher switch off (WrongGPU\wronggpu.ini)
    if (strcmp(argv[i], "--rr-off-own") == 0) rr_off_own = true;    // rr_enabled = 0 in the plugin's wronggpu-rr.ini
    if (strcmp(argv[i], "--fast") == 0) fast = true;                  // rr_mode = fast in WrongGPU\wronggpu.ini only
    if (strcmp(argv[i], "--fast-noblock") == 0) fast_noblock = true;  // the same, the shared settings block unavailable
    if (strcmp(argv[i], "--legacy-mode") == 0) legacy_mode = true;    // rr_mode only in the plugin's wronggpu-rr.ini
    if (strcmp(argv[i], "--log-cap") == 0) log_cap = true;            // the log at its 8 MiB cap: one rotation
  }
  std::string stage = argv[1];
  if (stage.back() != '\\' && stage.back() != '/') stage += '\\';
  const std::wstring wstage = widen(stage.c_str());
  const std::wstring log_path = wstage + (WG_BINDING_PROBE ? L"wronggpu-rr-probe.log" : L"wronggpu-rr.log");
  DeleteFileW(log_path.c_str());
  DeleteFileW((log_path + L".old").c_str());
  DeleteFileW((wstage + L"wronggpu-rr.ini").c_str());
  if (log_cap) {  // an earlier session's log at 8 MiB
    FILE* lf = _wfopen(log_path.c_str(), L"wb");
    CHECK(lf != nullptr);
    if (lf) {
      fputs("00:00:00.000 earlier session marker\r\n", lf);
      const std::string filler = std::string(1022, 'x') + "\r\n";
      for (int i = 0; i < 8 * 1024; ++i) fputs(filler.c_str(), lf);
      fclose(lf);
    }
  }
#if !WG_BINDING_PROBE
  {  // diag_env_N (release hook): WG_RR_ names become environment variables for the runtime; others are refused
    FILE* ini = _wfopen((wstage + L"wronggpu-rr.ini").c_str(), L"wb");
    fputs("diag_env_1=WG_RR_DIAG_HOOKTEST=7\r\ndiag_env_2=PATH_X=1\r\ndiag_env_3 = WG_RR_DIAG_HOOKTEST2=a=b ; c\r\n", ini);
    if (rr_off_own) fputs("rr_enabled = 0\r\n", ini);
    // the launcher's wronggpu-rr.ini of 0.6.0 holds rr_model only; an older install's may still hold rr_mode
    if (fast || fast_noblock) fputs("rr_model=C:\\wg-test\\rr-model.bin\r\n", ini);
    if (fast_noblock) fputs("rr_mode = accurate\r\n", ini);  // stale: WrongGPU\wronggpu.ini wins
    if (legacy_mode) fputs("rr_mode = fast\r\n", ini);
    fclose(ini);
  }
  // WrongGPU\wronggpu.ini of this run (the data folder, WG_DATA_DIR): the menu tab's settings
  const std::wstring data = wstage + L"data-" + (exact ? L"exact" : exact_denied ? L"denied" : rr_off ? L"rroff"
                                                  : rr_off_own ? L"rroffown" : fast ? L"fast" : fast_noblock ? L"noblock"
                                                  : legacy_mode ? L"legacy" : L"default") + L"\\";
  CreateDirectoryW(data.c_str(), nullptr);
  write_text(data + L"wronggpu.ini", exact          ? "[wronggpu]\r\nrr_mode = exact\r\nrr_exact_allowed = 1\r\n"
                                     : exact_denied ? "[wronggpu]\r\nrr_mode = exact\r\n"
                                     : rr_off       ? "[wronggpu]\r\nrr_mode = accurate\r\nrr_enabled = 0\r\n"
                                     : fast || fast_noblock ? "[wronggpu]\r\nrr_mode = fast\r\n"
                                     : legacy_mode  ? "[wronggpu]\r\nrr_enabled = 1\r\n"
                                                    : "[wronggpu]\r\nrr_mode = accurate\r\n");
  SetEnvironmentVariableW(L"WG_DATA_DIR", data.c_str());
  SetEnvironmentVariableW(L"WG_RR_MODE", nullptr);  // the checks below see what the plugin hands over, nothing inherited
  HANDLE block_squatter = nullptr;  // --fast-noblock: the block's name is taken, so no module can create the block
  if (fast_noblock) {
    char nm[64];
    snprintf(nm, sizeof nm, WG_SS_NAME_FMT, (unsigned long)GetCurrentProcessId());
    block_squatter = CreateEventA(nullptr, TRUE, FALSE, nm);
    CHECK(block_squatter != nullptr);
  }
#endif

  const std::wstring interposer = widen(kWgInterposerFile);
#if WG_TEST_DELAYLOAD
  // Delay-load import: nothing has called through it yet, so the interface is not loaded before the ASI runs.
  CHECK(GetModuleHandleW(interposer.c_str()) == nullptr);
#else
  // The interface stand-in is already loaded (the exe's static import); install the OptiScaler init patch on it.
  HMODULE iface = GetModuleHandleW(interposer.c_str());
  CHECK(iface != nullptr);
  if (!iface) TEST_SUMMARY("test_hook_dll");
  FoState* st = reinterpret_cast<FoState* (*)(void)>(GetProcAddress(iface, "wgtest_fo_state"))();
  auto set_opti = reinterpret_cast<void (*)(void (*)(void))>(GetProcAddress(iface, "wgtest_fo_set_opti_init"));
  CHECK(set_opti != nullptr);
  if (set_opti) set_opti(&opti_init_patch);
#if WG_BINDING_PROBE
  auto wrap = reinterpret_cast<void (*)(void*, void*)>(GetProcAddress(iface, "wgtest_fo_wrap"));
#endif
#endif

#if WG_BINDING_PROBE
  // capture option on (evaluates 4, 6, 8) and an earlier run's capture folder, which must be renamed, not deleted
  {
    FILE* ini = _wfopen((wstage + L"wronggpu-rr-probe.ini").c_str(), L"wb");
    fputs("; capture test\r\n[capture]\r\ncapture=1\r\ncapture_start = 4\r\ncapture_count=3\r\ncapture_every=2\r\n"
          "capture_read_delay=2\r\nbogus_key=1\r\n", ini);
    fputs(("capture_dir=" + stage + "cap-out" + std::string(1, 13) + std::string(1, 10)).c_str(), ini);
    fclose(ini);
    CreateDirectoryW((wstage + L"cap-out").c_str(), nullptr);
    FILE* old = _wfopen((wstage + L"cap-out\\earlier.txt").c_str(), L"wb");
    fputs("kept", old);
    fclose(old);
  }
#endif
  // clean/rr-sl-diag 2: the slot watch's first check at once (the watch is off unless WG_RR_SLOT_CHECKS asks for it;
  // "on" = 10 s after the hook and later)
  SetEnvironmentVariableW(L"WG_RR_SLOT_CHECKS", L"0");
  // Load the ASI the way OptiScaler does, after the interface is loaded, and run its entry point.
  HMODULE asi = LoadLibraryW((wstage + widen(kWgHookFile)).c_str());
  CHECK(asi != nullptr);
  if (!asi) TEST_SUMMARY("test_hook_dll");
  auto InitializeASI = reinterpret_cast<void (*)(void)>(GetProcAddress(asi, "InitializeASI"));
  CHECK(InitializeASI != nullptr);
  if (InitializeASI) InitializeASI();
#if WG_TEST_DELAYLOAD
  // The ASI loaded the interface for the delay-load import; install the OptiScaler init patch on it now.
  HMODULE iface = GetModuleHandleW(interposer.c_str());
  CHECK(iface != nullptr);
  if (!iface) TEST_SUMMARY("test_hook_dll");
  FoState* st = reinterpret_cast<FoState* (*)(void)>(GetProcAddress(iface, "wgtest_fo_state"))();
  auto set_opti = reinterpret_cast<void (*)(void (*)(void))>(GetProcAddress(iface, "wgtest_fo_set_opti_init"));
  CHECK(set_opti != nullptr);
  if (set_opti) set_opti(&opti_init_patch);
#if WG_BINDING_PROBE
  auto wrap = reinterpret_cast<void (*)(void*, void*)>(GetProcAddress(iface, "wgtest_fo_wrap"));
#endif
#endif

#if !WG_BINDING_PROBE
  if (!badver) {  // clean/060-rr-plugin-hunt: rr_mode as the menu box resolves it, logged with its source (the launcher's
                  // Last run page reads this line); handed to the runtime as WG_RR_MODE only where it cannot find it itself
    std::vector<std::string> L = read_lines(log_path);
    char ev[16] = {};
    const DWORD en = GetEnvironmentVariableA("WG_RR_MODE", ev, sizeof ev);
    CHECK(count(L, "hook: rr_mode ") == 1);
    if (fast) {  // fresh 0.6.0 install: the launcher's rr_mode=fast in WrongGPU\wronggpu.ini, rr_model in wronggpu-rr.ini
      CHECK(count(L, "hook: rr_mode fast (shared settings, wronggpu.ini; runtime ") == 1);
      CHECK(count(L, ", model C:\\wg-test\\rr-model.bin)") == 1);
      CHECK(en == 0);  // the runtime reads the block
    } else if (fast_noblock) {
      CHECK(count(L, "hook: rr_mode fast (wronggpu.ini, shared settings unavailable, handed to the runtime as WG_RR_MODE; "
                     "runtime ") == 1);
      CHECK(count(L, "hook: the legacy rr_mode line of wronggpu-rr.ini (accurate) is not used; wronggpu.ini sets the mode") == 1);
      CHECK(en == 4 && strcmp(ev, "fast") == 0);
    } else if (legacy_mode) {
      CHECK(count(L, "hook: rr_mode fast (legacy wronggpu-rr.ini, handed to the runtime as WG_RR_MODE; runtime ") == 1);
      CHECK(en == 4 && strcmp(ev, "fast") == 0);
    } else if (exact) {
      CHECK(count(L, "hook: rr_mode exact (shared settings, wronggpu.ini; runtime ") == 1);
      CHECK(en == 0);
    } else if (exact_denied) {  // the box turned exact into accurate in the block at start
      CHECK(count(L, "hook: rr_mode accurate (shared settings, ") == 1);
      CHECK(en == 0);
    } else {
      CHECK(count(L, "hook: rr_mode accurate (shared settings, wronggpu.ini; runtime ") == 1);
      CHECK(en == 0);
    }
    if (log_cap) {  // clean/060-rr-plugin-hunt: the first line of this process moved the full log to .log.old
      const std::vector<std::string> O = read_lines(log_path + L".old");
      CHECK(!O.empty() && O[0] == "00:00:00.000 earlier session marker");
      CHECK(count(O, "hook: ") == 0);
      CHECK(count(L, "log: wronggpu-rr.log reached 8 MiB; the earlier lines are in wronggpu-rr.log.old") == 1);
      CHECK(count(L, "earlier session marker") == 0);
      WIN32_FILE_ATTRIBUTE_DATA a;
      CHECK(GetFileAttributesExW(log_path.c_str(), GetFileExInfoStandard, &a) && a.nFileSizeHigh == 0 &&
            a.nFileSizeLow < (1u << 20));
    }
    if (fast || fast_noblock || legacy_mode || log_cap) {
      if (block_squatter) CloseHandle(block_squatter);
      TEST_SUMMARY(fast           ? "test_hook_dll (release, rr_mode in WrongGPU\\wronggpu.ini)"
                   : fast_noblock ? "test_hook_dll (release, rr_mode in WrongGPU\\wronggpu.ini, no shared settings)"
                   : legacy_mode  ? "test_hook_dll (release, legacy rr_mode in wronggpu-rr.ini)"
                                  : "test_hook_dll (release, wronggpu-rr.log at its size cap)");
    }
  }
#endif

  if (badver) {
    // Wrong interface version: nothing patched, the RR roles still reach the stand-in.
    st->calls = 0;
    WgFrameToken* tok = nullptr;
    uint32_t idx = 1;
    CHECK_EQ(slGetNewFrameToken(&tok, &idx), 0);
    CHECK(st->last_role != nullptr && strcmp(st->last_role, "new-frame-token") == 0);  // reached the stand-in
    std::vector<std::string> L = read_lines(log_path);
    CHECK(count(L, "does not match this build's binding version") == 1);
    // clean/060-rr-plugin-hunt: only the game-interface route is left out; the log does not say "nothing was patched"
    CHECK(count(L, "the game-interface route is not used for this version (no import slot patched)") == 1);
    CHECK(count(L, "nothing was patched") == 0);
#if !WG_BINDING_PROBE
    CHECK(count(L, "; games that call NGX directly can still get WrongGPU RR (the NGX route)") == 1);
#endif
    CHECK(count(L, "patched IAT slot") == 0);
    CHECK(count(L, "patched delay-load IAT slot") == 0);
    CHECK(count(L, "another Ray Reconstruction plugin is active") == 0);  // RR guard: not reached, quiet
    TEST_SUMMARY("test_hook_dll (wrong version)");
  }

  // clean/rr-sl-diag 2: the slot watch (read only): every patched slot still holds our handler at the first check, the
  // slots we left to the delay helper are still its thunks (nothing called them), and the exe links the interface
  {
    std::vector<std::string> W;
    for (int i = 0; i < 100; ++i) {  // the check runs on the watch's own thread
      W = read_lines(log_path);
      if (count(W, "diag: slot check 1 (timer, 0 s after the hook): ") == 1) break;
      Sleep(30);
    }
    CHECK(count(W, "diag: watching ") == 1);
    CHECK(count(W, "diag: slot check 1 (timer, 0 s after the hook): ") == 1);
    CHECK(count(W, "slot(s) hold our handler, 0 other delay-load slot(s) bound by the helper") == 1);
    CHECK(count(W, " slot slInit of ") == 1 && count(W, "WrongGPU's handler (as patched)") >= 3);
    CHECK(count(W, "REWRITTEN") == 0);
    CHECK(count(W, "modules linking it: none") == 0);
#if WG_TEST_DELAYLOAD
    CHECK(count(W, "(delay-load); game-interface calls that reached us") == 1);  // the exe, by delay-load import
#if !WG_BINDING_PROBE
    CHECK(count(W, "delay-load slot slUpgradeInterface of ") == 1 && count(W, "unbound (the delay helper's thunk at ") >= 1);
#endif
#else
    CHECK(count(W, "(import); game-interface calls that reached us") == 1);
#endif
  }

  // Matching version: the ASI redirected our RR-role IAT slots (observed through the log).
  {
    std::vector<std::string> L = read_lines(log_path);
    // RR guard: no other module here (the stand-in host's init patch, the delay helper's thunks): the guard
    // stays quiet (one clean check line, no named module / "stays off" / second-copy / later-module line)
    CHECK(count(L, "hook: coexistence check at load: no other Ray Reconstruction plugin found (") == 1);
    CHECK(count(L, "another Ray Reconstruction plugin is active") == 0);
    CHECK(count(L, "another module hooks the game's RR calls: ") == 0);
    CHECK(count(L, "a second copy of WrongGPU RR is loaded") == 0);
    CHECK(count(L, "took the RR calls after WrongGPU") == 0);
#if WG_TEST_DELAYLOAD
    CHECK(count(L, "loaded for the exe's delay-load import") == 1);
    CHECK(count(L, "patched IAT slot") == 0);  // the exe has no plain import of the interface
    CHECK(count(L, "patched delay-load IAT slot slInit") == 1);
    CHECK(count(L, "patched delay-load IAT slot slEvaluateFeature") == 1);
    CHECK(count(L, "patched delay-load IAT slot slGetFeatureFunction") == 1);
#if !WG_BINDING_PROBE
    CHECK(count(L, "IAT slot slUpgradeInterface") == 0);  // not hooked by the release build: left to the delay helper
#endif
#else
    CHECK(count(L, "game interface module found") == 1);
    CHECK(count(L, "already loaded via import table") == 1);
    CHECK(count(L, "patched IAT slot slInit") == 1);
    CHECK(count(L, "patched IAT slot slEvaluateFeature") == 1);
    CHECK(count(L, "patched IAT slot slGetFeatureFunction") == 1);
#if !WG_BINDING_PROBE
    CHECK(count(L, "patched IAT slot slUpgradeInterface") == 0);  // not hooked by the release build: untouched
#endif
#endif
  }

#if !WG_BINDING_PROBE
  // ---- the WrongGPU tab in OptiScaler's menu (rr_panel.h): added through OptiPluginInit; no RR box before RR runs
  auto plugin_init = reinterpret_cast<WgHostMenuInitFn>(GetProcAddress(asi, WG_HOST_MENU_INIT_EXPORT));
  CHECK(plugin_init != nullptr);
  const WgHostMenuApi host_menu = menu::table();
  {
    WgHostMenuApi v1 = host_menu;  // a host without Custom tab cards (version 1): a tab of its own
    v1.size = (uint32_t)offsetof(WgHostMenuApi, register_custom_card);
    v1.version = 1;
    CHECK(plugin_init && plugin_init(&v1, nullptr));
    CHECK(menu::tab_name == "WrongGPU" && menu::card_title.empty());
    WgHostMenuApi short_table = host_menu;
    short_table.size = (uint32_t)offsetof(WgHostMenuApi, combo);  // a host table without the combo box: no box
    CHECK(plugin_init && !plugin_init(&short_table, nullptr));
  }
  menu::tab_name.clear();
  // version 5 host (a card of the Custom tab, no tab; command-list work bracketed by the host)
  static WgHostMenuApi host_v5 = host_menu;
  host_v5.version = 5;
  host_v5.begin_cmd_list_work = menu::cl_begin;
  host_v5.end_cmd_list_work = menu::cl_end;
  host_v5.register_ngx_rr = menu::register_ngx;  // the NGX route: plugin interface 4
  static int fake_plugin = 0;
  {  // clean/rr-ngx: a version 3 host has no NGX RR provider registration: logged once, nothing registered
    WgHostMenuApi v3 = host_menu;
    v3.version = 3;
    v3.register_ngx_rr = menu::register_ngx;  // present but the version says 3: not used
    plugin_init(&v3, reinterpret_cast<WgHostMenuPlugin*>(&fake_plugin));
    plugin_init(&v3, reinterpret_cast<WgHostMenuPlugin*>(&fake_plugin));
    CHECK_EQ(menu::ngx_registers, 0);
    CHECK(count(read_lines(log_path), "NGX: the host has no RR provider registration (plugin interface 3, needs 4)") == 1);
  }
  CHECK(plugin_init && plugin_init(&host_v5, reinterpret_cast<WgHostMenuPlugin*>(&fake_plugin)));
  CHECK(menu::card_title == "WrongGPU" && menu::tab_name.empty() && menu::draw != nullptr);
  if (menu::draw) {
    menu::reset();
    menu::draw(nullptr);
    CHECK(menu::combos == 0 && menu::texts >= 1);  // RR not running: a line of text, no mode box
  }
  {
    std::vector<std::string> L = read_lines(log_path);
    CHECK(count(L, exact ? "exact mode allowed" : "exact mode not offered") == 1);
    CHECK(count(L, "accurate runs") == (exact_denied ? 1u : 0u));  // exact without the launcher's permission
  }
#endif

  // ---- the RR path reaches our core (the stand-in must not see RR init/evaluate/get-function)
  st->calls = 0; st->last_role = nullptr;
  static const uint32_t feats[2] = {1001, 0};
  WgPreferences pr;
  memset(&pr, 0, sizeof pr);
  pr.h.version = 1;
  pr.features = feats;
  pr.feature_count = 2;
  pr.flags = 1ull;  // bit 0: the game restores its own command-list state (probe passes allowed)
  pr.render_api = 1;
  const uint64_t ver = (2ull << 48) | ((uint64_t)WG_BINDING_API_MINOR << 32) | (1ull << 16) | WG_BINDING_VERSION_MAGIC;
  CHECK_EQ(slInit(&pr, ver), 0);
  // init is answered by the core AND forwarded to the interface export; the OptiScaler patch on that export ran.
  CHECK(strcmp(st->last_role, "init") == 0 && st->u64 == ver);
  CHECK(g_opti_init_seen >= 1);  // OptiScaler's inline patch stayed in the chain

  // non-RR feature: forwarded straight to the stand-in
  CHECK_EQ(slIsFeatureSupported(0, nullptr), 100u);
  CHECK(strcmp(st->last_role, "is-supported") == 0);

  // a non-RR get-function is forwarded to the stand-in (needs no device)
  void* other = nullptr;
  CHECK_EQ(slGetFeatureFunction(0, kWgFnOptimal, &other), 100u);
  CHECK(strcmp(st->last_role, "get-function") == 0);

#if !WG_BINDING_PROBE
  if (rr_off || rr_off_own) {
    // ---- the player's switch is off (rr_enabled = 0): WrongGPU does not take RR. The runtime is not loaded, RR is
    // reported unsupported (7) so the game uses its own path, nothing reaches the runtime, and other features are
    // forwarded as before.
    CHECK(GetModuleHandleW(WG_RR_RUNTIME_FILE) == nullptr);
    CHECK_EQ(slIsFeatureSupported(1001, nullptr), 7);
    CHECK_EQ(slIsFeatureSupported(1001, nullptr), 7);
    CHECK_EQ(slIsFeatureSupported(0, nullptr), 100u);
    static uint8_t pool_off[16];
    WgFrameToken* tok_off = nullptr;
    uint32_t idx_off = 1;
    CHECK_EQ(slGetNewFrameToken(&tok_off, &idx_off), 0);
    static WgViewport vp_off;
    memset(&vp_off, 0, sizeof vp_off);
    memcpy(vp_off.h.type.b, kWgGuid_viewport, 16);
    vp_off.h.version = 1;
    vp_off.id = 3;
    const WgHeader* in_off[1] = {&vp_off.h};
    st->last_role = nullptr;
    CHECK(slEvaluateFeature(1001, tok_off, in_off, 1, &pool_off[0]) != 0);  // nothing created, nothing evaluated
    CHECK(st->last_role == nullptr || strcmp(st->last_role, "evaluate") != 0);
    CHECK(menu::reports == 0);
    std::vector<std::string> L = read_lines(log_path);
    CHECK(count(L, "hook: rr_enabled=0 (") == 1);
    CHECK(count(L, rr_off ? "data-rroff\\wronggpu.ini): WrongGPU does not take RR"
                          : "wronggpu-rr.ini): WrongGPU does not take RR") == 1);
    CHECK(count(L, "rr: reported unsupported: WrongGPU RR is switched off (rr_enabled=0 in ") == 1);
    CHECK(count(L, "switched off (rr_enabled=0; RR reports unsupported)") == 1);
    CHECK(count(L, "WrongGPU RR runtime not found") == 0);
    // clean/rr-ngx: the NGX route is not offered either (the host keeps reporting RR unavailable)
    CHECK_EQ(menu::ngx_registers, 0);
    CHECK(count(L, "NGX: RR provider not registered: rr_enabled=0 (") == 1);
    CHECK_EQ(slShutdown(), 0);
    TEST_SUMMARY(rr_off ? "test_hook_dll (release, rr_enabled=0)"
                        : "test_hook_dll (release, rr_enabled=0 in wronggpu-rr.ini)");
  }
  // ---- release: the RR path runs into the fake runtime staged next to the exe
  HMODULE rtm = GetModuleHandleW(WG_RR_RUNTIME_FILE);
  CHECK(rtm != nullptr);
  FakeRuntime* rt = rtm ? reinterpret_cast<FakeRuntime* (*)(void)>(GetProcAddress(rtm, "wgtest_fr_state"))() : nullptr;
  CHECK(rt != nullptr);
  CHECK_EQ(slIsFeatureSupported(1001, nullptr), 0);  // claimed: runtime ready
  {
    char ev[16] = {};
    CHECK(GetEnvironmentVariableA("WG_RR_DIAG_HOOKTEST", ev, sizeof ev) && strcmp(ev, "7") == 0);
    CHECK(GetEnvironmentVariableA("WG_RR_DIAG_HOOKTEST2", ev, sizeof ev) && strcmp(ev, "a=b") == 0);
    CHECK(GetEnvironmentVariableA("PATH_X", ev, sizeof ev) == 0);
    std::vector<std::string> L2 = read_lines(log_path);
    CHECK(count(L2, "hook: diag_env WG_RR_DIAG_HOOKTEST=7") == 1 && count(L2, "hook: diag_env_2=PATH_X=1 ignored") == 1);
  }
  static uint8_t pool[64];
  void* device = &pool[0];
  void* cmd = &pool[8];
  CHECK_EQ(slSetD3DDevice(device), 0);
  CHECK(st->ptr == device);
  if (rt) CHECK(rt->device == (uint8_t*)device + 1);  // unwrapped through the interface's native-interface export
  void *fo = nullptr, *fs = nullptr, *fp = nullptr;
  CHECK_EQ(slGetFeatureFunction(1001, kWgFnOptimal, &fo), 0);  // the RR feature functions (after set-device)
  CHECK_EQ(slGetFeatureFunction(1001, kWgFnState, &fs), 0);
  CHECK_EQ(slGetFeatureFunction(1001, kWgFnOptions, &fp), 0);
  CHECK(fo && fs && fp && fo != fs && fs != fp);
  auto rr_optimal = reinterpret_cast<WgResultCode (*)(const WgRrOptions*, WgRrOptimal*)>(fo);
  auto rr_options = reinterpret_cast<WgResultCode (*)(const WgViewport*, const WgRrOptions*)>(fp);
  WgRrOptions op;
  memset(&op, 0, sizeof op);
  op.h.version = 3;
  op.mode = 3;
  op.output_width = 2560;
  op.output_height = 1440;
  op.colour_is_hdr = 1;
  WgRrOptimal os;
  memset(&os, 0, sizeof os);
  CHECK_EQ(rr_optimal(&op, &os), 0);
  CHECK(os.optimal_width == 1707 && os.optimal_height == 960);
  static WgViewport vp;
  memset(&vp, 0, sizeof vp);
  memcpy(vp.h.type.b, kWgGuid_viewport, 16);
  vp.h.version = 1;
  vp.id = 3;
  CHECK_EQ(rr_options(&vp, &op), 0);
  WgResource res[8];
  WgTag tags[8];
  const uint32_t types[8] = {3, 4, 1, 0, 7, 8, 5, 6};
  for (int i = 0; i < 8; ++i) {
    memset(&res[i], 0, sizeof res[i]);
    res[i].native = &pool[16 + i];
    res[i].width = i == 1 ? 2560 : 1707;
    res[i].height = i == 1 ? 1440 : 960;
    res[i].native_format = 10;
    res[i].state = 4;
    memset(&tags[i], 0, sizeof tags[i]);
    memcpy(tags[i].h.type.b, kWgGuid_tag, 16);
    tags[i].h.version = 1;
    tags[i].resource = &res[i];
    tags[i].buffer_type = types[i];
    tags[i].lifetime = 1;
  }
  CHECK_EQ(slSetTag(&vp, tags, 8, cmd), 0);
  CHECK(strcmp(st->last_role, "tag") == 0 && st->ptr == tags);  // tags are also forwarded to the interface
  WgFrameToken* tok = nullptr;
  uint32_t idx = 1;
  CHECK_EQ(slGetNewFrameToken(&tok, &idx), 0);
  WgConstants c;
  memset(&c, 0, sizeof c);
  c.h.version = 2;
  c.camera_motion_included = 1;
  c.mv_scale[0] = 1.0f / 1707.0f;
  c.mv_scale[1] = 1.0f / 960.0f;
  CHECK_EQ(slSetConstants(&c, tok, &vp), 0);
  const WgHeader* in[1] = {&vp.h};
  CHECK_EQ(slEvaluateFeature(1001, tok, in, 1, cmd), 0);
  CHECK(strcmp(st->last_role, "evaluate") != 0);  // RR never reaches the interface's evaluate
  CHECK(rt && rt->evaluates == 1 && rt->last().command_list == (uint8_t*)cmd + 1);
  // the host bracketed the evaluate on the game's (native) list, and the plugin logged that it does
  CHECK(menu::cl_begins == 1 && menu::cl_ends == 1 && menu::cl_last == (uint8_t*)cmd + 1);
  CHECK(count(read_lines(log_path), "hook: host restores the game's command-list state") == 1);
  CHECK_EQ(slEvaluateFeature(0, tok, in, 1, cmd), 100u);  // another feature forwarded
  CHECK(strcmp(st->last_role, "evaluate") == 0 && st->feature == 0);
  // RR ran a frame: OptiScaler's overlay got the RR name, and the card shows the mode box inside the card (no
  // section of its own; accurate and fast, exact first only with the launcher's permission)
  CHECK(menu::reports >= 1 && menu::reported == (exact ? "WrongGPU RR Exact" : "WrongGPU RR Accurate"));
  if (menu::draw) {
    menu::reset();
    menu::draw(nullptr);
    CHECK_EQ(menu::combos, 1);
    CHECK_EQ(menu::sections, 0);
    if (exact) {
      CHECK(menu::combo_count == 3 && menu::items[0] == "Exact" && menu::items[1] == "Accurate" && menu::items[2] == "Fast");
      CHECK_EQ(menu::combo_index, 0);  // rr_mode = exact runs exact
    } else {
      CHECK(menu::combo_count == 2 && menu::items[0] == "Accurate" && menu::items[1] == "Fast");
      CHECK_EQ(menu::combo_index, 0);  // accurate (exact_denied: exact shows as the accurate that runs)
    }
    // the player picks Fast: shared settings + wronggpu.ini, and the box shows it at once
    menu::pick = exact ? 2 : 1;
    menu::draw(nullptr);
    const std::string ini = file_text(data + L"wronggpu.ini");
    CHECK(ini.find("rr_mode=fast") != std::string::npos);
    CHECK(ini.find("rr_mode = exact") == std::string::npos && ini.find("rr_mode = accurate") == std::string::npos);
    menu::reset();
    menu::draw(nullptr);
    CHECK_EQ(menu::combo_index, exact ? 2 : 1);
    std::vector<std::string> L = read_lines(log_path);
    CHECK(count(L, "menu: rr_mode=fast chosen in OptiScaler's menu (from the next frame, saved to wronggpu.ini)") == 1);
  }
  // upgrade-interface is not hooked by the release build: the call reaches the stand-in directly (unchanged slot).
  void* iface2 = nullptr;
  slUpgradeInterface(&iface2);
  CHECK(strcmp(st->last_role, "generic") == 0);  // the stand-in's generic export recorded it
  auto get_contract = reinterpret_cast<WgRrGetSettingsContractFn>(GetProcAddress(asi, WG_RR_CONTRACT_EXPORT));
  CHECK(get_contract && get_contract()->status && (get_contract()->status() & 3u) == 3u);

  // ---- clean/rr-ngx: the NGX route (plugin interface 4) through the provider the plugin registered with the host
  CHECK_EQ(menu::ngx_registers, 1);
  CHECK(menu::nprov.size == sizeof(WgHostNgxRr) && menu::nprov.create && menu::nprov.evaluate && menu::nprov.release);
  CHECK(count(read_lines(log_path), "NGX: RR provider registered with the host (plugin interface 5): games that call "
                                    "NGX directly get WrongGPU RR; the host restores") == 1);
  MockNgxParams ncp, nep;
  ncp.set_u("Width", 1707);
  ncp.set_u("Height", 960);
  ncp.set_u("OutWidth", 2560);
  ncp.set_u("OutHeight", 1440);
  ncp.set_i("PerfQualityValue", 2);
  ncp.set_i("DLSS.Feature.Create.Flags", 0x1 | 0x2 | 0x8);
  ncp.set_i("DLSS.Denoise.Mode", 1);
  ncp.set_i("DLSS.Roughness.Mode", 0);
  ncp.set_i("DLSS.Use.HW.Depth", 1);
  const char* nkeys[8] = {"Color", "Output", "MotionVectors", "Depth", "DLSS.Input.DiffuseAlbedo",
                          "DLSS.Input.SpecularAlbedo", "GBuffer.Normals", "GBuffer.Roughness"};
  static float nw2v[16], nv2c[16];
  for (int i = 0; i < 16; ++i) nw2v[i] = nv2c[i] = (i % 5 == 0) ? 1.0f : 0.0f;
  for (int i = 0; i < 8; ++i) {
    void* r = &pool[40 + i];
    nep.set_res(nkeys[i], r);
    if (rt) {
      rt->described[r] = {i == 3 ? 39u : 10u, 1u};
      rt->sized[r] = {i == 1 ? 2560u : 1707u, i == 1 ? 1440u : 960u};
    }
  }
  nep.set_ptr("WorldToViewMatrix", nw2v);
  nep.set_ptr("ViewToClipMatrix", nv2c);
  nep.set_f("MV.Scale.X", 1707.0f);
  nep.set_f("MV.Scale.Y", 960.0f);
  void* ndev = &pool[56];
  void* ncl = &pool[57];
  WgHostNgxParams ncv = ncp.view(), nev = nep.view();
  void* nctx = nullptr;
  if (menu::nprov.create) {
    // one route per game: while the game-interface route holds the RR feature, the NGX route refuses
    CHECK(!menu::nprov.create(menu::nprov.user, ndev, ncl, &ncv, &nctx) && nctx == nullptr);
    CHECK(count(read_lines(log_path), "NGX: the game-interface route already runs WrongGPU RR in this game") == 1);
  }
  CHECK_EQ(slShutdown(), 0);  // releases the game-interface feature
  if (menu::nprov.create && rt) {
    const int creates = rt->creates, evals = rt->evaluates, destroys = rt->destroys, begins = menu::cl_begins;
    const int reports = menu::reports;
    CHECK(menu::nprov.create(menu::nprov.user, ndev, ncl, &ncv, &nctx) && nctx != nullptr);
    CHECK_EQ(rt->creates, creates + 1);
    CHECK(rt->created.back().api_minor == 1000u && rt->created.back().render_width == 1707 &&
          rt->created.back().output_width == 2560 && rt->created.back().packing == 0);
    CHECK(rt->device == ndev);
    CHECK(menu::nprov.evaluate(menu::nprov.user, nctx, ncl, &nev));
    CHECK_EQ(rt->evaluates, evals + 1);
    CHECK(rt->last().command_list == ncl && rt->last().mv_scale[0] == 1707.0f && rt->last().reset == 1);
    CHECK(rt->last().images[WG_ROLE_ROUGHNESS].resource == &pool[47]);
    // the plugin's own bracket on the game's list (nested inside the host's own bracket in OptiScaler)
    CHECK(menu::cl_begins == begins + 1 && menu::cl_ends == begins + 1 && menu::cl_last == ncl);
    CHECK(menu::reports > reports);  // the RR name and GPU time reached the host's overlay
    // one route per game, the other way round: game-interface RR work is refused while the NGX route holds RR
    static WgViewport nvp;
    memset(&nvp, 0, sizeof nvp);
    memcpy(nvp.h.type.b, kWgGuid_viewport, 16);
    nvp.h.version = 1;
    nvp.id = 3;
    const WgHeader* nin[1] = {&nvp.h};
    CHECK_EQ(slEvaluateFeature(1001, tok, nin, 1, cmd), 38);
    CHECK_EQ(slAllocateResources(cmd, 1001, &nvp), 38);
    CHECK_EQ(rt->evaluates, evals + 1);
    menu::nprov.release(menu::nprov.user, nctx);
    CHECK_EQ(rt->destroys, destroys + 1);
    CHECK(!menu::nprov.evaluate(menu::nprov.user, nctx, ncl, &nev));  // released: refused
    CHECK(slEvaluateFeature(1001, tok, nin, 1, cmd) != 38);       // the game-interface route is free again
    // a fault inside the provider (here: an access violation inside the runtime's evaluate, while the provider's
    // bracket on the game's list is open) never reaches the game: a failure is reported, the bracket is closed (the
    // host's begin / end stay paired) and the NGX route is off for the rest of the run
    CHECK(menu::nprov.create(menu::nprov.user, ndev, ncl, &ncv, &nctx) && nctx != nullptr);
    const int fault_begins = menu::cl_begins;
    rt->on_evaluate = [] { (void)(*(volatile int*)nullptr == 1); };
    CHECK(!menu::nprov.evaluate(menu::nprov.user, nctx, ncl, &nev));
    rt->on_evaluate = nullptr;
    CHECK(menu::cl_begins == fault_begins + 1 && menu::cl_ends == menu::cl_begins);
    CHECK(!menu::nprov.evaluate(menu::nprov.user, nctx, ncl, &nev));
    CHECK(menu::cl_begins == fault_begins + 1);  // off: nothing recorded any more
    {  // the route guard still answers from another thread (it never waits for the provider's lock after a fault)
      struct Call { const WgHeader* const* in; void* cmd; WgFrameToken* tok; volatile LONG rc; };
      static Call c;  // static: outlives a thread that never returns (the failure this checks)
      c.in = nin;
      c.cmd = cmd;
      c.tok = tok;
      c.rc = -1;
      HANDLE th = CreateThread(nullptr, 0, [](void* a) -> DWORD {
        Call* k = static_cast<Call*>(a);
        InterlockedExchange(&k->rc, (LONG)slEvaluateFeature(1001, k->tok, k->in, 1, k->cmd));
        return 0;
      }, &c, 0, nullptr);
      CHECK(th != nullptr);
      CHECK(th && WaitForSingleObject(th, 10000) == WAIT_OBJECT_0);
      CHECK_EQ((int)c.rc, 38);  // the abandoned NGX feature still holds RR: one route per game
      if (th) CloseHandle(th);
    }
    void* nctx2 = nullptr;
    CHECK(!menu::nprov.create(menu::nprov.user, ndev, ncl, &ncv, &nctx2));
    std::vector<std::string> NL = read_lines(log_path);
    CHECK(count(NL, "NGX: RR feature 1 created: render 1707x960 -> output 2560x1440, quality 2 (mode 3)") == 1);
    CHECK(count(NL, "NGX: RR feature 1 first evaluate: render 1707x960 -> output 2560x1440") == 1);
    CHECK(count(NL, "NGX: RR feature 1 released after 1 evaluate(s)") == 1);
    CHECK(count(NL, "hook: the NGX route holds WrongGPU RR in this game; game-interface RR work refused (38)") == 1);
    CHECK(count(NL, "NGX: a fault inside the RR provider's evaluate; the NGX route is off for this run") == 1);
  }
  std::vector<std::string> L = read_lines(log_path);
  CHECK(count(L, "hook: patched after the game interface initialised") >= 0);  // informational
  // clean/rr-sl-diag: the game-interface diagnostics reach the plugin's log: the RR support query with its answer, the RR
  // lookups, and the summary at the game interface's shutdown
  CHECK(count(L, "diag: support query RR (1001): ") >= 1);
  CHECK(count(L, "diag: RR function lookup \"") >= 1);
  CHECK(count(L, "diag: summary (shutdown): support queries ") >= 1);
  CHECK(count(L, "RR instances created 1") >= 1);
  CHECK(count(L, "redirected ") == 1 && count(L, "IAT slot(s) across") == 1);
  TEST_SUMMARY("test_hook_dll (release)");
#else
  // ---- probe: a short sequence on WARP; every RR evaluate returns 0 and writes the output
  Host g;
  bool host_ok = g.create();
  CHECK(host_ok);
  if (!host_ok) TEST_SUMMARY("test_hook_dll (no WARP device)");
  static uint8_t wrapped[2];
  void* wrapped_device = &wrapped[0];
  void* wrapped_cmd = &wrapped[1];
  if (wrap) { wrap(wrapped_device, g.dev); wrap(wrapped_cmd, g.cl); }

  CHECK_EQ(slSetD3DDevice(wrapped_device), 0);
  void *fo = nullptr, *fs = nullptr, *fp = nullptr;
  CHECK_EQ(slGetFeatureFunction(1001, kWgFnOptimal, &fo), 0);
  CHECK_EQ(slGetFeatureFunction(1001, kWgFnState, &fs), 0);
  CHECK_EQ(slGetFeatureFunction(1001, kWgFnOptions, &fp), 0);
  CHECK(fo && fs && fp);
  auto rr_optimal = reinterpret_cast<WgResultCode (*)(const WgRrOptions*, WgRrOptimal*)>(fo);
  auto rr_options = reinterpret_cast<WgResultCode (*)(const WgViewport*, const WgRrOptions*)>(fp);

  static WgViewport vp;
  memset(&vp, 0, sizeof vp);
  memcpy(vp.h.type.b, kWgGuid_viewport, 16);
  vp.h.version = 1;
  vp.id = 0;
  WgRrOptions op;
  memset(&op, 0, sizeof op);
  op.h.version = 3;
  set_guid(&op.h.type, "rr_options");
  op.mode = 1;
  op.output_width = 64;
  op.output_height = 36;
  op.colour_is_hdr = 1;
  op.normal_roughness_packing = 1;
  for (int i = 0; i < 16; ++i) op.world_to_view[i] = (i % 5 == 0) ? 1.0f : 0.0f;
  WgRrOptimal os;
  memset(&os, 0, sizeof os);
  os.h.version = 1;
  CHECK_EQ(rr_optimal(&op, &os), 0);
  CHECK(os.optimal_width == 32 && os.optimal_height == 18);
  CHECK_EQ(rr_options(&vp, &op), 0);
  CHECK_EQ(slAllocateResources(wrapped_cmd, 1001, &vp), 0);

  ID3D12Resource* colour = make_tex(g, 40, 24, DXGI_FORMAT_R16G16B16A16_FLOAT, false, D3D12_RESOURCE_STATE_COPY_SOURCE);
  ID3D12Resource* output = make_tex(g, 70, 40, DXGI_FORMAT_R16G16B16A16_FLOAT, true, D3D12_RESOURCE_STATE_COMMON);
  ID3D12Resource* normals = make_tex(g, 32, 18, DXGI_FORMAT_R10G10B10A2_UNORM, false, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
  ID3D12Resource* motion = make_tex(g, 32, 18, DXGI_FORMAT_R16G16_FLOAT, false, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
  const D3D12_RESOURCE_STATES kSr = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
  ID3D12Resource* depth = nullptr;
  {  // as game A: hardware depth in an R32_TYPELESS depth-stencil resource, declared 0xc0
    D3D12_HEAP_PROPERTIES hp = {}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC d = {}; d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D; d.Width = 32; d.Height = 18;
    d.DepthOrArraySize = 1; d.MipLevels = 1; d.Format = DXGI_FORMAT_R32_TYPELESS; d.SampleDesc.Count = 1;
    d.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    g.dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, kSr, nullptr, __uuidof(ID3D12Resource), (void**)&depth);
  }
  ID3D12Resource* albedo = make_tex(g, 32, 18, DXGI_FORMAT_R8G8B8A8_UNORM, true, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  ID3D12Resource* spec = make_tex(g, 32, 18, DXGI_FORMAT_R8G8B8A8_UNORM, false, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
  CHECK(colour && output && normals && motion && depth && albedo && spec);
  struct In { ID3D12Resource* r; uint32_t w, h; uint32_t fmt; uint32_t type; uint32_t state; WgExtent e; };
  const In ins[] = {
      {colour, 40, 24, DXGI_FORMAT_R16G16B16A16_FLOAT, 3, D3D12_RESOURCE_STATE_COPY_SOURCE, {2, 4, 32, 18}},
      {output, 70, 40, DXGI_FORMAT_R16G16B16A16_FLOAT, 4, D3D12_RESOURCE_STATE_COMMON, {1, 3, 64, 36}},
      {normals, 32, 18, DXGI_FORMAT_R10G10B10A2_UNORM, 14, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, {0, 0, 0, 0}},
      {motion, 32, 18, DXGI_FORMAT_R16G16_FLOAT, 1, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, {0, 0, 0, 0}},
      {depth, 32, 18, DXGI_FORMAT_R32_TYPELESS, 0, kSr, {0, 0, 0, 0}},
      {albedo, 32, 18, DXGI_FORMAT_R8G8B8A8_UNORM, 7, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, {0, 0, 0, 0}},
      {spec, 32, 18, DXGI_FORMAT_R8G8B8A8_UNORM, 8, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, {0, 0, 0, 0}}};
  const size_t nin = sizeof ins / sizeof ins[0];
  static WgResource res[8];
  static WgTag tags[8];
  for (size_t i = 0; i < nin; ++i) {
    memset(&res[i], 0, sizeof res[i]);
    res[i].h.version = 1;
    res[i].kind = WG_RES_TEX2D;
    res[i].native = ins[i].r;
    res[i].state = ins[i].state;
    res[i].width = ins[i].w;
    res[i].height = ins[i].h;
    res[i].native_format = ins[i].fmt;
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
  // known contents for the capture comparison: colour (RGBA16F 40x24), depth (R32 typeless 32x18), albedo (RGBA8,
  // declared UNORDERED_ACCESS as game A)
  std::vector<uint16_t> colour_px(40 * 24 * 4);
  for (size_t i = 0; i < colour_px.size(); ++i) colour_px[i] = to_half((float)(i % 977) / 64.0f + 0.125f);
  std::vector<float> depth_px(32 * 18);
  for (size_t i = 0; i < depth_px.size(); ++i) depth_px[i] = 1.0f / (1.0f + (float)i);
  std::vector<uint8_t> albedo_px(32 * 18 * 4);
  for (size_t i = 0; i < albedo_px.size(); ++i) albedo_px[i] = (uint8_t)(i * 7 + 3);
  g.begin();
  ID3D12Resource* ups[3] = {upload_tex(g, colour, colour_px.data(), 40 * 8, 24, D3D12_RESOURCE_STATE_COPY_SOURCE),
                            upload_tex(g, depth, depth_px.data(), 32 * 4, 18, kSr),
                            upload_tex(g, albedo, albedo_px.data(), 32 * 4, 18, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)};
  CHECK(ups[0] && ups[1] && ups[2]);
  CHECK(g.submit());
  for (ID3D12Resource* up : ups) if (up) up->Release();
  g.begin();
  CHECK_EQ(slSetTag(&vp, tags, (uint32_t)nin, wrapped_cmd), 0);
  CHECK(g.submit());
  int eval_ok = 0, frames_ok = 0;
  for (uint32_t f = 1; f <= 12; ++f) {
    g.begin();
    WgFrameToken* tok = nullptr;
    uint32_t idx = f;
    CHECK_EQ(slGetNewFrameToken(&tok, &idx), 0);
    if (f == 3) { tags[0].lifetime = WG_LIFE_ONLY_DURING_TAG; CHECK_EQ(slSetTag(&vp, &tags[0], 1, wrapped_cmd), 0); }
    WgConstants c;
    memset(&c, 0, sizeof c);
    c.h.version = 2;
    set_guid(&c.h.type, "constants");
    c.view_to_clip[0] = 1.0f;
    c.view_to_clip[5] = 1.5f;
    c.jitter[0] = 0.25f;
    c.jitter[1] = -0.125f;
    c.mv_scale[0] = 1.0f / 32.0f;
    c.mv_scale[1] = 1.0f / 18.0f;
    c.camera_motion_included = 1;
    c.depth_inverted = 1;
    c.reset = f == 1 ? 1 : 0;
    CHECK_EQ(slSetConstants(&c, tok, &vp), 0);
    const WgHeader* in[1] = {&vp.h};
    eval_ok += slEvaluateFeature(1001, tok, in, 1, wrapped_cmd) == 0;
    frames_ok += g.submit();
  }
  CHECK_EQ(eval_ok, 12);
  CHECK_EQ(frames_ok, 12);
  CHECK(strcmp(st->last_role, "evaluate") != 0);  // RR never forwarded to the interface
  CHECK_EQ(slShutdown(), 0);
  std::vector<std::string> L = read_lines(log_path);
  CHECK(count(L, "probe start: build Streamline 2.") == 1 && count(L, "(hook mode)") >= 1);
  // delay-load: the plugin loaded the interface itself, so the game's <init> is still to come and nothing is adopted
  CHECK(count(L, "init unseen: the game's <init> ran before the hook") == (WG_TEST_DELAYLOAD ? 0u : 1u));
  CHECK(count(L, "redirected ") == 1);
  CHECK(count(L, "call evaluate feature=1001: rc=0") >= 1);
  // ---- capture option
  CHECK(count(L, "capture settings: line 8: unknown key bogus_key: ignored") == 1);
  CHECK(count(L, "capture: on, start=4 count=3 every=2 max_mb=2048 read_delay=2 folder=") == 1 && count(L, "cap-out") >= 1);
  CHECK(count(L, "(the earlier capture was renamed to ") == 1);
  CHECK(count(L, "capture eval=4: copied 6 of 6 input(s) (types 0,1,3,7,8,14)") == 1);
  CHECK(count(L, "capture eval=6: copied 6 of 6") == 1 && count(L, "capture eval=8: copied 6 of 6") == 1);
  CHECK(count(L, "capture eval=4: written frame_000004 (6 input file(s) + frame.json") == 1);
  CHECK(count(L, "capture eval=8: written frame_000008") == 1);
  CHECK(count(L, "capture eval=5") == 0 && count(L, "capture eval=10") == 0);
  CHECK(count(L, "capture: done, 3 of 3 evaluate(s) written") == 1);
  CHECK(count(L, "capture at shutdown: 3 evaluate(s) written") == 1 && count(L, " 0 failed file write(s)") == 1);
  {
    const std::wstring cap = wstage + L"cap-out\\";
    for (const wchar_t* fr : {L"frame_000004", L"frame_000006", L"frame_000008"}) {
      const std::wstring d = cap + fr + L"\\";
      // colour: the probe's lifetime-0 copy from frame 3 on (state NON_PIXEL_SHADER_RESOURCE), extent left 4, top 2
      CHECK(check_capture_file(d + L"input_03.raw", 3, DXGI_FORMAT_R16G16B16A16_FLOAT, 40, 24, 40 * 8, colour_px.data(),
                               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, 4, 2));
      CHECK(check_capture_file(d + L"input_00.raw", 0, DXGI_FORMAT_R32_TYPELESS, 32, 18, 32 * 4, depth_px.data(), kSr, 0, 0));
      CHECK(check_capture_file(d + L"input_07.raw", 7, DXGI_FORMAT_R8G8B8A8_UNORM, 32, 18, 32 * 4, albedo_px.data(),
                               D3D12_RESOURCE_STATE_UNORDERED_ACCESS, 0, 0));
      CHECK(check_capture_file(d + L"input_14.raw", 14, DXGI_FORMAT_R10G10B10A2_UNORM, 32, 18, 32 * 4, nullptr,
                               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, 0, 0));
      CHECK(exists(d + L"input_01.raw") && exists(d + L"input_08.raw"));
      CHECK(!exists(d + L"input_04.raw"));  // the output is not captured
      CHECK(!exists(d + L"frame.json.part") && !exists(d + L"input_03.raw.part"));
    }
    CHECK(!exists(cap + L"frame_000005") && !exists(cap + L"frame_000010"));
    const std::string j = text_of(read_file(cap + L"frame_000006\\frame.json"));
    CHECK(j.find("{\"format\":\"wronggpu-rr-capture\",\"version\":1,\"evaluate\":6,\"frame_index\":") == 0);
    CHECK(j.find("\"render\":{\"x\":4,\"y\":2,\"width\":32,\"height\":18}") != std::string::npos);
    CHECK(j.find("\"output\":{\"type\":4,\"tag_format\":10,\"format\":10,\"width\":70,\"height\":40,"
                 "\"rect\":[3,1,64,36],\"state\":0}") != std::string::npos);
    CHECK(j.find("\"jitter\":[0.25,-0.125]") != std::string::npos && j.find("\"depth_inverted\":1") != std::string::npos);
    CHECK(j.find("\"normal_roughness_packing\":1") != std::string::npos);
    CHECK(j.find("{\"type\":0,\"file\":\"input_00.raw\",\"format\":39,\"width\":32,\"height\":18,\"row_pitch\":128,"
                 "\"tag_format\":39,\"state\":192") != std::string::npos);
    CHECK(j.find("input_04.raw") == std::string::npos && j.find("\"not_copied\":[]}") != std::string::npos);
    // the earlier capture folder was renamed with its file, not deleted
    WIN32_FIND_DATAW fd;
    HANDLE fh = FindFirstFileW((wstage + L"cap-out.*").c_str(), &fd);
    bool kept = false;
    if (fh != INVALID_HANDLE_VALUE) {
      do {  // the pattern also matches the new folder itself
        if (wcsncmp(fd.cFileName, L"cap-out.", 8) == 0)
          kept = kept || exists(wstage + fd.cFileName + L"\\earlier.txt");
      } while (FindNextFileW(fh, &fd));
      FindClose(fh);
    }
    CHECK(kept);
    CHECK(!exists(cap + L"earlier.txt"));
  }
  // no debug-layer error or corruption message (resource states restored, copies valid)
  if (g.iq) {
    uint64_t errors = 0;
    for (UINT64 i = 0, n = g.iq->GetNumStoredMessages(); i < n; ++i) {
      SIZE_T len = 0;
      g.iq->GetMessage(i, nullptr, &len);
      std::vector<uint8_t> mb(len);
      D3D12_MESSAGE* msg = reinterpret_cast<D3D12_MESSAGE*>(mb.data());
      if (SUCCEEDED(g.iq->GetMessage(i, msg, &len)) &&
          (msg->Severity == D3D12_MESSAGE_SEVERITY_ERROR || msg->Severity == D3D12_MESSAGE_SEVERITY_CORRUPTION)) {
        if (errors < 10) fprintf(stderr, "debug layer: %s\n", msg->pDescription);
        ++errors;
      }
    }
    CHECK_EQ(errors, (uint64_t)0);
    printf("debug layer on: %llu message(s) stored\n", (unsigned long long)g.iq->GetNumStoredMessages());
  } else {
    printf("debug layer not available\n");
  }
  colour->Release(); output->Release(); normals->Release(); motion->Release(); depth->Release();
  albedo->Release(); spec->Release();
  TEST_SUMMARY("test_hook_dll (probe)");
#endif
}
