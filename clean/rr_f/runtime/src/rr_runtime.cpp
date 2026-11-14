// SPDX-License-Identifier: MIT
// WrongGPU RR runtime (wronggpu_rr_runtime.dll): the owned adapter of clean/rr_f/proxy/include/wg_rr_adapter.h for the
// preset-F ray reconstruction of specs 50-58 on D3D12 / RDNA 4. Original WrongGPU code.
//
// Per decided frame the proxy hands over (WgRrFrame), the runtime records into the GAME's command list (no second
// queue, no fences, no command-list split; the pattern of the Model M D3D12 executor) the integration branch's frame
// (clean/rr_f/gpu/sequence.py, clean/rr-f-int):
//   game textures -> NON_PIXEL_SHADER_RESOURCE / outputs -> UNORDERED_ACCESS (declared states, restored afterwards);
//   [non-reset] PH2 quarter pool (54b.5, rr_resolve_pool) -> exposure (58; the previous frame's E from its result)
//   -> input stage (52: decode of PH1 / PH2 / HID, fast features, exact replay, front-end MLP; the game's inputs read
//   through SRVs, the API read conversion of 52.1.2) -> IN0, E1 .. E5, D4-F .. D0-F (51, step X, exact row sums)
//   -> D0-R head (55) -> output stage (54: CH pair, PH1, Q) -> extended resolve (54b: transparency layer, opacity,
//   alpha upscaling, separate output alpha; TH / OH / AH pairs) -> the game's output(s) (typed UAV stores).
// Every intermediate lives in one per-instance arena buffer (root UAVs u0 / u1, byte offsets as root constants). The
// temporal items of page 53.1 have fixed offsets in it: output-size items (CH pair, PH1, PH2, the exposure result
// pair, AH / TH / OH pairs) for the instance's life, render-size items (HID, PM pair, Pe, Pn) until the next
// render-size change (53.9). The frame's work buffers are placed by their lifetimes (buffers that are never live at
// the same pass share memory). The host bookkeeping of page 53 (par, fc, pj, CP, pending resets) is the proxy's.
//
// Completion: one runtime-wide marker (WriteBufferImmediate after each recorded frame) tells which frames the game's
// queue has finished; per-frame upload / descriptor / timestamp slots and every deferred release wait for it.
// Log: wronggpu-rr.log next to this library (the proxy's directory).
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <intrin.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <winternl.h>
#include <d3dkmthk.h>

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "../../host/rr_input_host.h"
#include "../../../runtime/interop/texture_shape.h"
#include "../../proxy/include/rr_proxy_abi.h"
#include "../../proxy/include/wg_rr_adapter.h"
#include "rr_kernels.h"
#include "rr_model.h"
#include "rr_runtime_test.h"
#include "rr_modes.h"
#include "rr_retire.h"
#include "../../proxy/include/wg_settings_reader.h"
#include "../../../runtime/interop/descriptor_publish.h"

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")

#ifndef WG_RR_RUNTIME_VERSION0
#define WG_RR_RUNTIME_VERSION0 0
#define WG_RR_RUNTIME_VERSION1 6
#define WG_RR_RUNTIME_VERSION2 0
#endif

namespace wgrr_rt {
namespace {

using rr_input::f_bits;

// debug name of every pipeline state (WKPDID_D3DDebugObjectName): profilers / the live harness's perf timeline read it
static const GUID kPsoNameGuid = {0x429b8c22, 0x9188, 0x4b0c, {0x87, 0x42, 0xac, 0xb0, 0xbf, 0x85, 0xc2, 0x00}};
constexpr uint32_t kSlots = 8;          // frames in flight per instance (upload / descriptor / timestamp slots)
constexpr uint32_t kRoles = rr_input::ROLE_COUNT;   // 13 input-stage roles
constexpr uint32_t kSrvArray = 16;      // SRV array (register space 1) of the input stage / exposure: index = role
constexpr uint32_t kPairs = 12;         // {SRV t0, UAV u2} pairs per slot (ingest copies, output stores)
constexpr uint32_t kDescPerSlot = kSrvArray + 2 * kPairs;
constexpr uint32_t kTs = 8;             // timestamps per frame (passes)
constexpr uint32_t kTsIn = 3;           // + input-stage split points (after decode/clear, features, replay)
constexpr uint32_t kTq = kTs + kTsIn;   // query slots per frame
constexpr uint32_t kRootConsts = 2 + 48;   // head.fast: 40 + 8 (RR_OUT_PREP)
constexpr uint32_t kBlock = 4096;       // per-frame upload block: input constants, decode headers, resolve parameters
constexpr uint32_t kBlkInput = 0, kBlkDecode = 1024, kBlkPool = 1280, kBlkResolve = 2048;
constexpr uint32_t kBlkInputDiag = 3072;   // clean/rr-f-s9-inx: WG_RR_DIAG_INX's second input block (exact pass outputs)
constexpr uint32_t kResolveWords = 176;
// launch geometry of the clean/rr-f-speed1 kernels (gpu/input_stage.py, gpu/output.py, gpu/resolve_fast.py, gpu/pool.py)
constexpr uint32_t kFeatBlock = 128;    // input.features / input.replay lanes per workgroup (FEAT_BLOCK)
constexpr uint32_t kReplayGroups = 512; // input.replay: fixed grid over the replay list (REPLAY_GROUPS)
constexpr uint32_t kOutPrepBlock = 256, kOutFallbackGroups = 64;   // output.prep lanes, output.fallback grid
constexpr uint32_t kResTile = 16, kResGroups = 512;                // resolve.fast: 16 x 16 tiles, persistent groups
constexpr uint32_t kPoolLanes = 256 * 4;                          // pool.main: 256 lanes x 4 PH2 texels
constexpr uint32_t kPoolTableBytes = 2050 * 4, kExp2TableBytes = 8192;
// the output fast path's packed_head: packed head buffers (1), E / Ep / invE from the exposure result (2), binary16
// motion (4) and history (32) images, no linear / trace outputs (8)
constexpr uint32_t kOutFastFlags = 1u | 2u | 4u | 8u | 32u;
constexpr uint32_t kTimingWindow = 300;

const char* const kPassNames[kTs - 1] = {"pre", "input", "body", "head", "output", "resolve", "store"};

// ---------------------------------------------------------------------------------------------------- small helpers
std::string narrow(const std::wstring& w) {
  int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), nullptr, 0, nullptr, nullptr);
  std::string s(n, '\0');
  WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), &s[0], n, nullptr, nullptr);
  return s;
}
std::string fmt(const char* f, ...) {
  char buf[1024];
  va_list a;
  va_start(a, f);
  vsnprintf(buf, sizeof buf, f, a);
  va_end(a);
  return buf;
}
uint64_t align256(uint64_t v) { return (v + 255) & ~255ull; }
template <class T> void release(T*& p) {
  if (p) p->Release();
  p = nullptr;
}
uint32_t fb(float f) { return f_bits(f); }

HMODULE g_module = nullptr;
std::wstring module_dir() {
  wchar_t buf[MAX_PATH * 2];
  DWORD n = GetModuleFileNameW(g_module, buf, (DWORD)(sizeof buf / sizeof buf[0]));
  std::wstring p(buf, n);
  size_t s = p.find_last_of(L"\\/");
  return s == std::wstring::npos ? std::wstring() : p.substr(0, s + 1);
}

std::mutex g_log_mu;
void log_line(const std::string& s) {
  std::lock_guard<std::mutex> lock(g_log_mu);
  wchar_t env[MAX_PATH];
  std::wstring path = GetEnvironmentVariableW(L"WG_RR_LOG", env, MAX_PATH) ? std::wstring(env)
                                                                           : module_dir() + L"wronggpu-rr.log";
  FILE* f = _wfopen(path.c_str(), L"ab");
  if (!f) return;
  SYSTEMTIME t;
  GetLocalTime(&t);
  fprintf(f, "%02u:%02u:%02u.%03u runtime: %s\r\n", t.wHour, t.wMinute, t.wSecond, t.wMilliseconds, s.c_str());
  fclose(f);
}

// RDNA 4 boards (the rule of clean/runtime/rdna4.h on clean/rdna4-dx12, WrongGPU's own code)
bool rdna4_adapter(uint32_t vendor, bool software, const std::string& d) {
  return vendor == 0x1002 && !software && (d.find("RX 9070") != std::string::npos || d.find("RX 9060") != std::string::npos);
}

// The adapter's PCI vendor / device id as the kernel-mode display driver reports it, looked up by LUID. DXGI's
// description can be rewritten by other layers in the game process (OptiScaler's GPU spoofing, for games that only
// offer their upscaler options for another vendor id); this path does not go through DXGI.
bool kmt_device_ids(LUID luid, uint32_t* vendor, uint32_t* device) {
  D3DKMT_OPENADAPTERFROMLUID o = {};
  o.AdapterLuid = luid;
  if (D3DKMTOpenAdapterFromLuid(&o) != 0) return false;
  D3DKMT_QUERY_DEVICE_IDS ids = {};
  ids.PhysicalAdapterIndex = 0;
  D3DKMT_QUERYADAPTERINFO q = {};
  q.hAdapter = o.hAdapter;
  q.Type = KMTQAITYPE_PHYSICALADAPTERDEVICEIDS;
  q.pPrivateDriverData = &ids;
  q.PrivateDriverDataSize = sizeof ids;
  const NTSTATUS st = D3DKMTQueryAdapterInfo(&q);
  D3DKMT_CLOSEADAPTER c = {};
  c.hAdapter = o.hAdapter;
  D3DKMTCloseAdapter(&c);
  if (st != 0) return false;
  *vendor = ids.DeviceIds.VendorID;
  *device = ids.DeviceIds.DeviceID;
  return true;
}

// RDNA 4 PCI device ids (Navi 48: RX 9070 XT / 9070 / 9070 GRE; Navi 44: RX 9060 XT / 9060).
bool rdna4_device_id(uint32_t device) {
  return device == 0x7550 || device == 0x7551 || device == 0x7590 || device == 0x7591;
}

// RDNA 4 check for one adapter: DXGI's description first; when that does not say RDNA 4 (or was rewritten), the
// kernel driver's ids decide. *note gets a log line when the two disagree.
bool rdna4_adapter_desc(const DXGI_ADAPTER_DESC1& d, std::string* note) {
  const bool software = (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0;
  if (rdna4_adapter(d.VendorId, software, narrow(d.Description))) return true;
  uint32_t ven = 0, dev = 0;
  if (software || !kmt_device_ids(d.AdapterLuid, &ven, &dev)) return false;
  if (ven != 0x1002 || !rdna4_device_id(dev)) return false;
  if (note)
    *note = fmt("DXGI reports \"%s\" (vendor 0x%04x, likely spoofed by another layer in the game); the kernel driver "
                "reports %04x:%04x, an RDNA 4 board: using the driver's identity",
                narrow(d.Description).c_str(), d.VendorId, ven, dev);
  return true;
}

// ---------------------------------------------------------------------------------------------------- formats (62.5)
DXGI_FORMAT srv_format(DXGI_FORMAT f, bool depth) {
  switch (f) {
    case DXGI_FORMAT_D32_FLOAT: return DXGI_FORMAT_R32_FLOAT;
    case DXGI_FORMAT_R32_TYPELESS: return DXGI_FORMAT_R32_FLOAT;
    case DXGI_FORMAT_D24_UNORM_S8_UINT:
    case DXGI_FORMAT_R24G8_TYPELESS: return DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
    case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
    case DXGI_FORMAT_R32G8X24_TYPELESS: return DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
    case DXGI_FORMAT_D16_UNORM: return DXGI_FORMAT_R16_UNORM;
    case DXGI_FORMAT_R16_TYPELESS: return depth ? DXGI_FORMAT_R16_UNORM : DXGI_FORMAT_R16_FLOAT;
    case DXGI_FORMAT_R32G32B32A32_TYPELESS: return DXGI_FORMAT_R32G32B32A32_FLOAT;
    case DXGI_FORMAT_R32G32B32_TYPELESS: return DXGI_FORMAT_R32G32B32_FLOAT;
    case DXGI_FORMAT_R16G16B16A16_TYPELESS: return DXGI_FORMAT_R16G16B16A16_FLOAT;
    case DXGI_FORMAT_R32G32_TYPELESS: return DXGI_FORMAT_R32G32_FLOAT;
    case DXGI_FORMAT_R10G10B10A2_TYPELESS: return DXGI_FORMAT_R10G10B10A2_UNORM;
    case DXGI_FORMAT_R8G8B8A8_TYPELESS: return DXGI_FORMAT_R8G8B8A8_UNORM;
    case DXGI_FORMAT_R16G16_TYPELESS: return DXGI_FORMAT_R16G16_FLOAT;
    case DXGI_FORMAT_R8G8_TYPELESS: return DXGI_FORMAT_R8G8_UNORM;
    case DXGI_FORMAT_R8_TYPELESS: return DXGI_FORMAT_R8_UNORM;
    case DXGI_FORMAT_B8G8R8A8_TYPELESS: return DXGI_FORMAT_B8G8R8A8_UNORM;
    case DXGI_FORMAT_B8G8R8X8_TYPELESS: return DXGI_FORMAT_B8G8R8X8_UNORM;
    default: return f;
  }
}
bool is_srgb(DXGI_FORMAT f) {
  return f == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB || f == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB || f == DXGI_FORMAT_B8G8R8X8_UNORM_SRGB;
}
DXGI_FORMAT uav_format(DXGI_FORMAT f) {
  f = srv_format(f, false);
  if (f == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB) return DXGI_FORMAT_R8G8B8A8_UNORM;
  if (f == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB) return DXGI_FORMAT_B8G8R8A8_UNORM;
  if (f == DXGI_FORMAT_B8G8R8X8_UNORM_SRGB) return DXGI_FORMAT_B8G8R8X8_UNORM;
  return f;
}
bool is_f16(DXGI_FORMAT f) {
  return f == DXGI_FORMAT_R16G16B16A16_FLOAT || f == DXGI_FORMAT_R16G16_FLOAT || f == DXGI_FORMAT_R16_FLOAT;
}
bool is_unorm8x4(DXGI_FORMAT f) {
  return f == DXGI_FORMAT_R8G8B8A8_UNORM || f == DXGI_FORMAT_B8G8R8A8_UNORM || f == DXGI_FORMAT_B8G8R8X8_UNORM;
}
bool is_unorm8(DXGI_FORMAT f) { return is_unorm8x4(f) || f == DXGI_FORMAT_R8_UNORM || f == DXGI_FORMAT_R8G8_UNORM; }
bool is_rgb10a2(DXGI_FORMAT f) { return f == DXGI_FORMAT_R10G10B10A2_UNORM; }
uint32_t channels_of(DXGI_FORMAT f) {
  switch (f) {
    case DXGI_FORMAT_R32_FLOAT: case DXGI_FORMAT_R16_FLOAT: case DXGI_FORMAT_R8_UNORM: case DXGI_FORMAT_R16_UNORM:
    case DXGI_FORMAT_R24_UNORM_X8_TYPELESS: case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS:
      return 1;
    case DXGI_FORMAT_R32G32_FLOAT: case DXGI_FORMAT_R16G16_FLOAT: case DXGI_FORMAT_R8G8_UNORM: case DXGI_FORMAT_R16G16_UNORM:
      return 2;
    case DXGI_FORMAT_R32G32B32_FLOAT: case DXGI_FORMAT_R11G11B10_FLOAT:
      return 3;
    default:
      return 4;
  }
}

// ---------------------------------------------------------------------------------------------------- kernels
enum Kid {
  K_IN0 = 0, K_E1, K_E2, K_E3, K_E4, K_E5, K_D4F, K_D4R, K_D3F, K_D3R, K_D2F, K_D2R, K_D1F, K_D1R, K_D0F,
  K_REGROUP, K_HEAD, K_HEAD_FAST, K_HEAD_DEBUG, K_HEAD_WPL, K_HEAD_FIXUP, K_HEAD_STRICT, K_EXP_PART, K_EXP_FIN, K_IN_DEC, K_IN_FEAT, K_IN_REPLAY, K_IN_MLP, K_OUT_MAIN,
  K_OUT_POOL, K_RES_MAIN, K_RES_POOL, K_INGEST, K_FILL, K_STORE, K_SELFTEST,
  // clean/rr-f-speed1: the replay-list clear, the output stage's fast path, the fast resolve (+ fixup, exp2 table), the
  // PH2 pool through exact tables
  K_IN_CLEAR, K_OUT_PREP, K_OUT_FAST, K_OUT_FALLBACK, K_RES_FAST, K_RES_FIXUP, K_RES_INIT, K_POOL_TABLES, K_POOL_MAIN,
  // clean/rr-f-r2-body: the exact body step Z (rr_block.hlsl RR_STEP 3 + RR_ZFAST 1, bit-identical to step X) per stage,
  // and its fixup (step X on the windows the Z dispatch marked, RR_FIXUP), in kStages order
  K_ZSTAGE0, K_ZFIX0 = K_ZSTAGE0 + 15,
  // clean/rr-f-r3-input: the opt-in near-exact input features (WG_RR_INPUT=lite; not the default)
  K_IN_FEAT_LITE = K_ZFIX0 + 15,
  // clean/rr-f-r3-bodyE: the default matrix step (variant E, not exact to 51.1) per stage on the struct routes, opt-in
  // WG_RR_BODY=E (the sequence bench's default frame; A/B runs and image comparisons)
  K_ESTAGE0 = K_IN_FEAT_LITE + 1,
  // clean/rr-f-r4-outres: the fast resolve specialised for the game's configuration (RF_SPEC 2: extended + layer
  // without alpha upscaling, binary16 layer / output; same bits), used when the frame's resolve matches it
  K_RES_FAST2 = K_ESTAGE0 + 15,
  // clean/rr-f-speed5: the front-end MLP on the default matrix step (variant E, RR_STEP 0; not exact to 51.1): the
  // default with the default body (the 8 captured frames byte-identical to variant X); WG_RR_INPUT_MLP=X / E overrides
  K_IN_MLP_E = K_RES_FAST2 + 1,
  // clean/rr-f-r6-tail: output.prep's uniform part (OF_PREPU) when the fast head writes the render records
  K_OUT_PREPU = K_IN_MLP_E + 1,
  // clean/rr-f-live2: the pass-through of a failed evaluate (WrongGPU's fail-safe, not preset F)
  K_PASS = K_OUT_PREPU + 1,
  // clean/rr-f-s9-inx: the split input features (history samples exact through the fast pass + replay, non-history
  // colour / guide features with input lite's maths; rr_input.hlsl RR_IN_NHL)
  K_IN_FEAT_SPLIT = K_PASS + 1,
  // clean/rr-f-s10-modes: the alternative kernels of the other modes (rr_mode exact | accurate | fast); the primary names
  // above are the accurate set. head.fast.rx: relaxed head (Fast); stageex.*: exact body step E, front-end MLP not folded
  // (exact); output.*.f: OF_HWRCP + OF_REPROJ_MAX (Fast); resolve.fast*.x: correctly rounded exp2 / reciprocal (exact)
  K_HEAD_FAST_RX = K_IN_FEAT_SPLIT + 1,
  K_ESTAGEX0 = K_HEAD_FAST_RX + 1,
  K_OUT_PREP_F = K_ESTAGEX0 + 15, K_OUT_FAST_F, K_OUT_FALLBACK_F, K_OUT_PREPU_F,
  K_RES_FAST_X, K_RES_FAST2_X,
  // clean/rr-f-s12-dflow: stagee.* in the dataflow form (rr_dflow.hlsli; the stages of kRrDflowMask, the others repeat
  // stagee.*'s DXIL), used with WG_RR_DFLOW=1
  K_DSTAGE0,
  // clean/rr-f-s13x-x-flow: stageex.* in the dataflow form (the stages of kRrDflowMaskX, the others repeat stageex.*)
  K_DXSTAGE0 = K_DSTAGE0 + 15,
  // clean/rr-f-a14-frame E24: read-only cache warm-up (rr_touch.hlsl)
  K_TOUCH = K_DXSTAGE0 + 15,
  K_BLKCOPY,   // E25 step 2: argument block upload ring -> arena (root SRV 6)
  K_COUNT
};
const char* const kKernelNames[] = {
    "stage.IN0", "stage.E1", "stage.E2", "stage.E3", "stage.E4", "stage.E5", "stage.D4-F", "stage.D4-R",
    "stage.D3-F", "stage.D3-R", "stage.D2-F", "stage.D2-R", "stage.D1-F", "stage.D1-R", "stage.D0-F",
    "regroup", "head", "head.fast", "head.debug", "head.wpl", "head.fixup", "head.strict", "exposure.partial", "exposure.finish", "input.decode", "input.features",
    "input.replay", "input.mlp", "output.main", "output.pool", "resolve.main", "resolve.pool", "rt.ingest", "rt.fill",
    "rt.store", "rt.selftest", "input.clear", "output.prep", "output.fast", "output.fallback", "resolve.fast",
    "resolve.fixup", "resolve.init", "pool.tables", "pool.main",
    "stagez.IN0", "stagez.E1", "stagez.E2", "stagez.E3", "stagez.E4", "stagez.E5", "stagez.D4-F", "stagez.D4-R",
    "stagez.D3-F", "stagez.D3-R", "stagez.D2-F", "stagez.D2-R", "stagez.D1-F", "stagez.D1-R", "stagez.D0-F",
    "stagefix.IN0", "stagefix.E1", "stagefix.E2", "stagefix.E3", "stagefix.E4", "stagefix.E5", "stagefix.D4-F",
    "stagefix.D4-R", "stagefix.D3-F", "stagefix.D3-R", "stagefix.D2-F", "stagefix.D2-R", "stagefix.D1-F",
    "stagefix.D1-R", "stagefix.D0-F", "input.features.lite",
    "stagee.IN0", "stagee.E1", "stagee.E2", "stagee.E3", "stagee.E4", "stagee.E5", "stagee.D4-F", "stagee.D4-R",
    "stagee.D3-F", "stagee.D3-R", "stagee.D2-F", "stagee.D2-R", "stagee.D1-F", "stagee.D1-R", "stagee.D0-F",
    "resolve.fast2", "input.mlp.e", "output.prepu", "rt.pass", "input.features.split",
    "head.fast.rx",
    "stageex.IN0", "stageex.E1", "stageex.E2", "stageex.E3", "stageex.E4", "stageex.E5", "stageex.D4-F", "stageex.D4-R",
    "stageex.D3-F", "stageex.D3-R", "stageex.D2-F", "stageex.D2-R", "stageex.D1-F", "stageex.D1-R", "stageex.D0-F",
    "output.prep.f", "output.fast.f", "output.fallback.f", "output.prepu.f", "resolve.fast.x", "resolve.fast2.x",
    "staged.IN0", "staged.E1", "staged.E2", "staged.E3", "staged.E4", "staged.E5", "staged.D4-F", "staged.D4-R",
    "staged.D3-F", "staged.D3-R", "staged.D2-F", "staged.D2-R", "staged.D1-F", "staged.D1-R", "staged.D0-F",
    "stagedx.IN0", "stagedx.E1", "stagedx.E2", "stagedx.E3", "stagedx.E4", "stagedx.E5", "stagedx.D4-F", "stagedx.D4-R",
    "stagedx.D3-F", "stagedx.D3-R", "stagedx.D2-F", "stagedx.D2-R", "stagedx.D1-F", "stagedx.D1-R", "stagedx.D0-F",
    "rt.touch", "rt.blkcopy"};
static_assert(sizeof(kKernelNames) / sizeof(kKernelNames[0]) == K_COUNT, "kKernelNames and Kid disagree");
// clean/rr-f-flow: kernels only the opt-in variants dispatch (exact / strict / debug head, body steps X and Z,
// WG_RR_PATHS=verified); the default frame (head fast, body E, fast paths) never uses them
// the features kernel of an input mode (0 exact, 1 lite, 2 split; mode_input)
Kid input_features_kernel(int mode) { return mode == 1 ? K_IN_FEAT_LITE : (mode == 2 ? K_IN_FEAT_SPLIT : K_IN_FEAT); }
bool kernel_opt_in(int k) {
  if (k >= K_IN0 && k <= K_D0F) return true;                  // body step X
  if (k >= K_ZSTAGE0 && k < K_ZFIX0 + 15) return true;        // body step Z and its fixup
  switch (k) {
    case K_HEAD: case K_HEAD_DEBUG: case K_HEAD_WPL: case K_HEAD_FIXUP: case K_HEAD_STRICT:
    case K_OUT_MAIN: case K_OUT_POOL: case K_RES_MAIN: case K_RES_POOL:
      return true;
    default:
      return false;
  }
}
// clean/rr-f-s10-modes: the four places where the modes use different kernels. variant_group(k, &alt): the group of a
// mode-dependent kernel (0 head, 1 body, 2 output, 3 resolve) and whether k is the group's alternative; -1 for every
// other kernel. The bit of a group in a variant set is set when the alternative runs: head = relaxed (Fast), body =
// exact (exact), output = Fast's defines, resolve = correctly rounded (exact). The default frame's pass compiles only the
// kernels of the variants the runtime starts in; the others follow in the second (background) pass.
// VG_HEAD / VG_BODY / VG_OUT / VG_RES: rr_modes.h (the registry words map onto these groups)
int variant_group(int k, bool* alt) {
  *alt = false;
  if (k == K_HEAD_FAST) return VG_HEAD;
  if (k == K_HEAD_FAST_RX) { *alt = true; return VG_HEAD; }
  if (k >= K_ESTAGE0 && k < K_ESTAGE0 + 15) return VG_BODY;
  if (k >= K_DSTAGE0 && k < K_DSTAGE0 + 15) return VG_BODY;   // clean/rr-f-s12-dflow: the relaxed body's dataflow form
  if (k >= K_ESTAGEX0 && k < K_ESTAGEX0 + 15) { *alt = true; return VG_BODY; }
  if (k >= K_DXSTAGE0 && k < K_DXSTAGE0 + 15) { *alt = true; return VG_BODY; }   // clean/rr-f-s13x-x-flow: the exact body's
  if (k == K_OUT_PREP || k == K_OUT_FAST || k == K_OUT_FALLBACK || k == K_OUT_PREPU) return VG_OUT;
  if (k >= K_OUT_PREP_F && k <= K_OUT_PREPU_F) { *alt = true; return VG_OUT; }
  if (k == K_RES_FAST || k == K_RES_FAST2) return VG_RES;
  if (k == K_RES_FAST_X || k == K_RES_FAST2_X) { *alt = true; return VG_RES; }
  return -1;
}
// kernel k belongs to the default frame of the variant set vbits (no opt-in kernel, and its variant is the one in use)
bool kernel_in_default_frame(int k, unsigned vbits) {
  if (kernel_opt_in(k)) return false;
  bool alt;
  const int g = variant_group(k, &alt);
  return g < 0 || alt == (((vbits >> g) & 1u) != 0);
}
// the mode (clean/rr-f-s10-modes; i15: from the registry, rr_modes.h): exact = everything exact; accurate (default) =
// exact head, input, history / output / resolve, relaxed body arithmetic; fast = relaxed head, lite input, OF_HWRCP +
// OF_REPROJ_MAX, hardware resolve (clean/rr-f-s11-tail, owner 2026-10-28: accurate's resolve is exact too). Mode,
// mode_vbits(), mode_input(), mode_name() come from rr_modes.h.
constexpr Mode MODE_EXACT = WG_RR_MODE_EXACT, MODE_ACCURATE = WG_RR_MODE_ACCURATE, MODE_FAST = WG_RR_MODE_FAST;

// block stages in network order (rr_block.hlsl), with level, channels, kind
struct StageDef { const char* name; Kid k; int level; uint32_t C; char kind; uint32_t Cn; };
const StageDef kStages[15] = {
    {"IN0", K_IN0, 0, 32, 'i', 0},     {"E1", K_E1, 1, 64, 'e', 96},     {"E2", K_E2, 2, 96, 'e', 128},
    {"E3", K_E3, 3, 128, 'e', 160},    {"E4", K_E4, 4, 160, 'e', 160},   {"E5", K_E5, 5, 160, 'e', 0},
    {"D4-F", K_D4F, 4, 160, 'f', 0},   {"D4-R", K_D4R, 4, 160, 'r', 0},  {"D3-F", K_D3F, 3, 128, 'f', 0},
    {"D3-R", K_D3R, 3, 128, 'r', 0},   {"D2-F", K_D2F, 2, 96, 'f', 0},   {"D2-R", K_D2R, 2, 96, 'r', 0},
    {"D1-F", K_D1F, 1, 64, 'f', 0},    {"D1-R", K_D1R, 1, 64, 'r', 0},   {"D0-F", K_D0F, 0, 32, 'f', 0}};

// ---------------------------------------------------------------------------------------------------- runtime state
struct Runtime {
  std::recursive_mutex mu;
  bool init_done = false;
  bool adapter_present = false;   // an RDNA 4 adapter exists (status READY before set-device)
  bool device_failed = false;     // set_device ran and its checks failed
  // clean/rr-f-live2: the frame kernels' pipelines are made on a worker thread (a cold driver cache needs minutes
  // for them; set-device must not block the game that long). Evaluates fail (the proxy passes the frame through)
  // until kernels_ready. The worker never takes mu; drop_device cancels and joins it.
  std::thread compile_thread;
  std::atomic<bool> kernels_ready{false}, compile_cancel{false};
  // clean/rr-f-flow: kernels_ready = the default frame's kernels (compiled first); all_ready = every kernel, the opt-in
  // variants' too (WG_RR_HEAD=exact/strict/debug, WG_RR_BODY=X/Z, WG_RR_PATHS=verified), compiled after them
  std::atomic<bool> all_ready{false};
  // clean/rr-f-s10-modes: modes_ready = the kernels of the other modes' variants too (second pass, before the opt-in
  // variants' minutes-long strict / exact head compiles of the third)
  std::atomic<bool> modes_ready{false};
  // clean/rr-f-s10-modes: the variant set (mode_vbits) whose kernels the first pass compiled (the mode the runtime started
  // in); an instance of another mode, or with a per-knob override of a variant, waits for all_ready
  unsigned start_vbits = 0;
  Mode start_mode = MODE_ACCURATE;
  std::atomic<int> compile_failed{-1};       // kernel index whose pipeline failed, -1 none
  std::atomic<uint64_t> compile_ms{0};
  bool compile_logged = false;
  Model model;
  // clean/rr-f-s9-host: the model words the frame recording needs, looked up once per loaded model (the per-evaluate
  // string-map lookups were most of the evaluate's CPU time); filled by model_words()
  struct StageWords { uint32_t bo[18]; uint32_t ra_m, f_g, zok, zthr, zt0, zt1; };
  struct Words { bool ok = false; std::string key; StageWords st[15]; uint32_t sh[2][11]; uint32_t sym[32]; };
  Words words;
  std::string model_err;
  // device objects
  ID3D12Device* dev = nullptr;
  ID3D12CommandQueue* q = nullptr;
  ID3D12CommandAllocator* al = nullptr;
  ID3D12GraphicsCommandList* l = nullptr;
  ID3D12Fence* fence = nullptr;
  HANDLE ev = nullptr;
  uint64_t fv = 0;
  ID3D12RootSignature* rs = nullptr;
  ID3D12PipelineState* pso[K_COUNT] = {};
  ID3D12Resource* model_buf = nullptr;
  ID3D12Resource* ags_buf = nullptr;
  ID3D12Resource* marker = nullptr;
  volatile uint32_t* marker_map = nullptr;
  uint64_t serial = 0;            // frames recorded into the game's lists (all instances)
  UINT inc = 0;
  uint64_t ts_freq = 0;
  std::string adapter_name;
  bool ready_logged = false;
  RetireList<IUnknown> deferred;            // objects kept until the frame marker proves their frames done (rr_retire.h)
  std::vector<ID3D12Resource*> transient;   // retain_transient copies alive
  std::atomic<int64_t> live_resources{0};
  std::atomic<int64_t> created_resources{0};
  int live_instances = 0;
  uint32_t next_instance = 1;
  // clean/rr-f-live1: fail-safe state
  bool disabled = false;                     // an exception was caught inside an evaluate: evaluates fail from now on
  ID3D12DescriptorHeap* pass_heap = nullptr; // pass-through descriptors (fixed part + one pair per slot)
  ID3D12Resource* pass_dummy = nullptr;      // bound as the root UAVs of the pass-through (never accessed)
  uint64_t pass_serial[kSlots] = {};
  uint64_t passes = 0;
  bool logged_pass_refused = false;
  uint64_t evaluates_logged = 0;             // first evaluates logged in full (log_evaluates)
  uint64_t tlog_logged = 0;                  // transition evaluates logged in full (capped)
  // clean/release-060 crash-cp2077: device-removal watch (a fence the driver signals to UINT64_MAX on removal); logs
  // the GPU breadcrumbs (marker words 2, 3) and the instances' buffer addresses
  ID3D12Fence* rm_fence = nullptr;
  HANDLE rm_ev = nullptr, rm_quit = nullptr;
  std::thread rm_thread;
  bool crumbs = false;                       // WG_RR_CRUMBS=1 at set-device: GPU breadcrumbs around every dispatch (diagnostics)
  std::mutex va_mu;
  std::string va_note;                       // "instance N arena [a, b) upload [c, d)" of the live instances
  // the input stage's SRV array as written per slot (copied back from the heap right after writing) and the frame's
  // resources, compared with the heap at removal
  uint8_t dsh[kSlots][kSrvArray * 64] = {};
  uint64_t dsh_serial[kSlots] = {};
  void* dsh_res[kSlots][kSrvArray] = {};
  uintptr_t dsh_cpu[kSlots] = {};
  uint32_t ksh[kSlots][256] = {};      // each slot's input constant block as written (removal log: vs the ring now)
  uintptr_t kring_cpu[kSlots] = {};
  uint32_t dsh_inc = 0;
};
Runtime* g_rt = nullptr;
Runtime& R() {
  if (!g_rt) g_rt = new Runtime();
  return *g_rt;
}

// resources made by the runtime (counted for the leak test)
HRESULT make_buffer(uint64_t bytes, D3D12_HEAP_TYPE heap, D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state,
                    ID3D12Resource** out) {
  D3D12_HEAP_PROPERTIES hp = {};
  hp.Type = heap;
  D3D12_RESOURCE_DESC rd = {};
  rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  rd.Width = bytes;
  rd.Height = 1;
  rd.DepthOrArraySize = 1;
  rd.MipLevels = 1;
  rd.SampleDesc.Count = 1;
  rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  rd.Flags = flags;
  HRESULT h = R().dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, state, nullptr, IID_PPV_ARGS(out));
  if (SUCCEEDED(h)) R().live_resources++, R().created_resources++;
  return h;
}
void drop(ID3D12Resource*& r) {
  if (r) {
    r->Release();
    r = nullptr;
    R().live_resources--;
  }
}
uint64_t marker_done() {
  Runtime& rt = R();
  if (!rt.marker_map) return rt.serial;
  const uint32_t low = *rt.marker_map;
  uint64_t v = (rt.serial & ~0xffffffffull) | low;
  if (v > rt.serial) v -= 0x100000000ull;
  return v;
}
// releases deferred objects whose frames have completed; once the device is removed nothing of it runs any more and
// everything left is released (issue #63: kept objects go at completion, device removal or device drop, never on a timeout)
void pump() {
  Runtime& rt = R();
  rt.live_resources -= rt.deferred.pump(marker_done());
  if (!rt.deferred.empty() && rt.dev && FAILED(rt.dev->GetDeviceRemovedReason())) {
    const size_t n = rt.deferred.size();
    rt.live_resources -= rt.deferred.release_all();
    log_line(fmt("device removed: %zu kept object(s) released", n));
  }
}
void defer_release(ID3D12Resource*& r, uint64_t serial) {
  R().live_resources -= R().deferred.retire(r, serial, marker_done(), true);
  r = nullptr;
}
bool wait_serial(uint64_t serial, DWORD ms) {
  for (DWORD i = 0; i < ms && marker_done() < serial; ++i) {
    if (R().dev && FAILED(R().dev->GetDeviceRemovedReason())) return false;
    Sleep(1);
  }
  return marker_done() >= serial;
}

// our own queue: create-time work only
bool submit_wait(std::string* err) {
  Runtime& rt = R();
  if (FAILED(rt.l->Close())) return *err = "command list Close failed", false;
  ID3D12CommandList* ll = rt.l;
  rt.q->ExecuteCommandLists(1, &ll);
  ++rt.fv;
  if (FAILED(rt.q->Signal(rt.fence, rt.fv)) || FAILED(rt.fence->SetEventOnCompletion(rt.fv, rt.ev)))
    return *err = "queue signal failed", false;
  if (WaitForSingleObject(rt.ev, 20000) != WAIT_OBJECT_0) return *err = "own queue did not finish within 20 s", false;
  if (FAILED(rt.dev->GetDeviceRemovedReason())) return *err = "device removed", false;
  rt.al->Reset();
  rt.l->Reset(rt.al, nullptr);
  return true;
}

// root signature: u0, u1 root UAVs (arena), b0 constants, table {u0 space 0x7FFF0ADE} (driver channel),
// table {t0, u2} (ingest source / store destination), table {t0 .. t15 space 1} (the game's inputs, index = role)
bool make_root(std::string* err) {
  Runtime& rt = R();
  D3D12_DESCRIPTOR_RANGE ags = {};
  ags.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
  ags.NumDescriptors = 1;
  ags.RegisterSpace = 0x7FFF0ADE;
  D3D12_DESCRIPTOR_RANGE tex[2] = {};
  tex[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
  tex[0].NumDescriptors = 1;
  tex[0].OffsetInDescriptorsFromTableStart = 0;
  tex[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
  tex[1].NumDescriptors = 1;
  tex[1].BaseShaderRegister = 2;
  tex[1].OffsetInDescriptorsFromTableStart = 1;
  D3D12_DESCRIPTOR_RANGE arr = {};
  arr.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
  arr.NumDescriptors = kSrvArray;
  arr.RegisterSpace = 1;
  D3D12_ROOT_PARAMETER rp[7] = {};
  rp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
  rp[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
  rp[1].Descriptor.ShaderRegister = 1;
  rp[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
  rp[2].Constants = {0, 0, kRootConsts};
  rp[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
  rp[3].DescriptorTable = {1, &ags};
  rp[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
  rp[4].DescriptorTable = {2, tex};
  rp[5].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
  rp[5].DescriptorTable = {1, &arr};
  D3D12_ROOT_SIGNATURE_DESC rsd = {};
  // clean/rr-f-a14-frame E25: root SRV t0 space 9 (the upload ring, read by rt.blkcopy only)
  rp[6].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
  rp[6].Descriptor.ShaderRegister = 0;
  rp[6].Descriptor.RegisterSpace = 9;
  rsd.NumParameters = 7;
  rsd.pParameters = rp;
  ID3DBlob *blob = nullptr, *e = nullptr;
  if (FAILED(D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &e))) {
    *err = std::string("root signature: ") + (e ? (const char*)e->GetBufferPointer() : "?");
    release(e);
    return false;
  }
  HRESULT h = rt.dev->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&rt.rs));
  release(blob);
  return SUCCEEDED(h) ? true : (*err = "CreateRootSignature failed", false);
}

void write_ags_uav(D3D12_CPU_DESCRIPTOR_HANDLE h) {
  D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
  ud.Format = DXGI_FORMAT_R32_TYPELESS;
  ud.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
  ud.Buffer.NumElements = 64;
  ud.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
  R().dev->CreateUnorderedAccessView(R().ags_buf, nullptr, &ud, h);
}
void write_null_srv(D3D12_CPU_DESCRIPTOR_HANDLE h) {
  D3D12_SHADER_RESOURCE_VIEW_DESC sd = {};
  sd.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
  sd.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
  sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
  sd.Texture2D.MipLevels = 1;
  R().dev->CreateShaderResourceView(nullptr, &sd, h);
}
void write_null_pair(D3D12_CPU_DESCRIPTOR_HANDLE h) {
  write_null_srv(h);
  h.ptr += R().inc;
  D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
  ud.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
  ud.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
  R().dev->CreateUnorderedAccessView(nullptr, nullptr, &ud, h);
}

// heap: [0] driver channel UAV, [1, 2] null pair, [3 .. 3 + kSrvArray) null SRV array, then the slots
constexpr uint32_t kHeapFixed = 3 + kSrvArray;
void bind(ID3D12GraphicsCommandList* l, ID3D12DescriptorHeap* heap, ID3D12Resource* arena) {
  l->SetDescriptorHeaps(1, &heap);
  l->SetComputeRootSignature(R().rs);
  l->SetComputeRootUnorderedAccessView(0, arena->GetGPUVirtualAddress());
  l->SetComputeRootUnorderedAccessView(1, arena->GetGPUVirtualAddress());
  D3D12_GPU_DESCRIPTOR_HANDLE t = heap->GetGPUDescriptorHandleForHeapStart();
  l->SetComputeRootDescriptorTable(3, t);
  t.ptr += R().inc;  // the null pair after the channel UAV
  l->SetComputeRootDescriptorTable(4, t);
  t.ptr += 2 * R().inc;
  l->SetComputeRootDescriptorTable(5, t);
}
// clean/release-060 crash-cp2077: GPU breadcrumbs around every dispatch of the frame being recorded (set by evaluate,
// under rt.mu): marker word 2 = the last dispatch the GPU reached (MARKER_IN), word 3 = the last one finished
// (MARKER_OUT); value = frame serial low 12 bits << 20 | dispatch index in the frame << 8 | kernel id
static_assert(K_COUNT <= 256, "crumb kernel id is 8 bits");
ID3D12GraphicsCommandList* g_crumb_base = nullptr;
ID3D12GraphicsCommandList2* g_crumb_list = nullptr;
uint32_t g_crumb_tag = 0;
uint32_t g_crumb_n = 0;
void crumb(uint32_t word, uint32_t v, D3D12_WRITEBUFFERIMMEDIATE_MODE mode) {
  D3D12_WRITEBUFFERIMMEDIATE_PARAMETER p{R().marker->GetGPUVirtualAddress() + 4ull * word, v};
  g_crumb_list->WriteBufferImmediate(1, &p, &mode);
}
void dispatch(ID3D12GraphicsCommandList* l, Kid k, uint32_t groups, const std::vector<uint32_t>& args) {
  Runtime& rt = R();
  const bool cr = g_crumb_list && l == g_crumb_base;
  const uint32_t cv = g_crumb_tag | (std::min<uint32_t>(g_crumb_n, 4095) << 8) | (uint32_t)k;
  if (cr) g_crumb_n++, crumb(2, cv, D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_IN);
  l->SetPipelineState(rt.pso[k]);
  if (!args.empty()) l->SetComputeRoot32BitConstants(2, (UINT)args.size(), args.data(), 2);
  for (uint32_t base = 0; base < groups; base += 65535) {
    const uint32_t c[2] = {base, groups};
    l->SetComputeRoot32BitConstants(2, 2, c, 0);
    l->Dispatch(std::min<uint32_t>(65535, groups - base), 1, 1);
  }
  if (cr) crumb(3, cv, D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT);
}
std::string crumb_text(uint32_t v) {
  if (!v) return "none";
  const uint32_t k = v & 0xff;
  return fmt("frame %u #%u %s", v >> 20, (v >> 8) & 0xfff, k < K_COUNT ? kKernelNames[k] : "?");
}
bool seh_copy(void* dst, const void* src, size_t n) {
  __try {
    memcpy(dst, src, n);
    return true;
  } __except (EXCEPTION_EXECUTE_HANDLER) {
    return false;
  }
}
uint64_t desc_va(const uint8_t* d) {   // an image descriptor's base address (dwords 0..1[7:0], address >> 8)
  uint32_t w0, w1;
  memcpy(&w0, d, 4), memcpy(&w1, d + 4, 4);
  return ((uint64_t)(w1 & 0xffu) << 40) | ((uint64_t)w0 << 8);
}
// clean/release-060 crash-cp2077 fix: every frame's descriptor writes into the shader-visible heap end with a store
// fence and a read of the written range (wg::publish_descriptors, descriptor_publish.h); `shadow` keeps the SRV array
// as written for the removal log. False when the range could not be read.
bool publish_descriptors(const void* p, size_t n, void* shadow) { return wg::publish_descriptors(p, n, shadow); }
void removal_watch() {
  Runtime& rt = R();
  HANDLE hs[2] = {rt.rm_quit, rt.rm_ev};
  if (WaitForMultipleObjects(2, hs, FALSE, INFINITE) != WAIT_OBJECT_0 + 1) return;
  const HRESULT why = rt.dev->GetDeviceRemovedReason();
  volatile uint32_t* m = rt.marker_map;
  const uint32_t w0 = m ? m[0] : 0, w2 = m ? m[2] : 0, w3 = m ? m[3] : 0;
  std::string va;
  {
    std::lock_guard<std::mutex> lock(rt.va_mu);
    va = rt.va_note;
  }
  log_line(fmt("DEVICE REMOVED (0x%08x): last frame finished %u (low 12 bits %u), frames recorded %llu; RR dispatch "
               "reached: %s; finished: %s; %s",
               (unsigned)why, w0, w0 & 0xfff, (unsigned long long)rt.serial, crumb_text(w2).c_str(),
               crumb_text(w3).c_str(), va.empty() ? "no live instance" : va.c_str()));
  const uint32_t inc = std::min<uint32_t>(rt.dsh_inc, 64);
  for (uint32_t s = 0; inc && s < kSlots; ++s) {
    if (!rt.dsh_serial[s] || (rt.dsh_serial[s] & 0xfff) != (w2 ? (w2 >> 20) : ((w0 + 1) & 0xfff))) continue;   // without breadcrumbs: the frame after the last finished one
    uint8_t now[kSrvArray * 64] = {};
    const bool rd = seh_copy(now, (const void*)rt.dsh_cpu[s], (size_t)kSrvArray * inc);
    std::string m = fmt("removal: frame %llu slot %u input SRVs (role: resource, base as written / in the heap now%s):",
                        (unsigned long long)rt.dsh_serial[s], s, rd ? "" : ", heap not readable");
    for (uint32_t r = 0; r < kSrvArray; ++r) {
      const uint8_t* a = rt.dsh[s] + (size_t)r * inc;
      const uint8_t* b = now + (size_t)r * inc;
      const bool same = !rd || memcmp(a, b, inc) == 0;
      if (!rt.dsh_res[s][r] && desc_va(a) == 0 && same) continue;
      m += fmt(" %u: %p 0x%llx%s", r, rt.dsh_res[s][r], (unsigned long long)desc_va(a),
               same ? "" : fmt(" / CHANGED 0x%llx", (unsigned long long)desc_va(b)).c_str());
    }
    log_line(m);
    std::string d = fmt("removal: frame %llu slot %u input descriptors in the heap now (32-bit words):", (unsigned long long)rt.dsh_serial[s], s);
    for (uint32_t r = 0; r < kSrvArray; ++r) {
      d += fmt(" [%u]", r);
      for (uint32_t k = 0; k < inc / 4; ++k) {
        uint32_t v;
        memcpy(&v, (rd ? now : rt.dsh[s]) + (size_t)r * inc + 4 * k, 4);
        d += fmt(" %08x", v);
      }
    }
    log_line(d);
    uint32_t kn[256] = {};
    const bool krd = rt.kring_cpu[s] && seh_copy(kn, (const void*)rt.kring_cpu[s], sizeof kn);
    std::string kd = fmt("removal: frame %llu slot %u input constant block (ring now%s):", (unsigned long long)rt.dsh_serial[s], s,
                         krd ? "" : ", not readable");
    uint32_t ndiff = 0;
    for (uint32_t i = 0; i < 256; ++i) {
      kd += fmt(" %08x", krd ? kn[i] : rt.ksh[s][i]);
      if (krd && kn[i] != rt.ksh[s][i]) ndiff++;
    }
    log_line(kd);
    std::string kx = fmt("removal: frame %llu slot %u constant words differing from as written: %u", (unsigned long long)rt.dsh_serial[s], s, ndiff);
    for (uint32_t i = 0; krd && i < 256; ++i)
      if (kn[i] != rt.ksh[s][i]) kx += fmt(" w%u %08x->%08x", i, rt.ksh[s][i], kn[i]);
    log_line(kx);
  }
  for (uint32_t s = 0; inc && s < kSlots; ++s) {   // every slot's frame: each role's base address as written
    if (!rt.dsh_serial[s]) continue;
    std::string m = fmt("removal: slot %u frame %llu:", s, (unsigned long long)rt.dsh_serial[s]);
    for (uint32_t r = 0; r < kSrvArray; ++r)
      if (rt.dsh_res[s][r]) m += fmt(" %u=%p@0x%llx", r, rt.dsh_res[s][r], (unsigned long long)desc_va(rt.dsh[s] + (size_t)r * inc));
    log_line(m);
  }
}
// clean/rr-f-a14-frame E2 (WG_RR_BARK): the barrier flavour of the frame being recorded (set by evaluate, under rt.mu)
int g_bark = 0;
ID3D12GraphicsCommandList* g_bark_list = nullptr;
ID3D12GraphicsCommandList7* g_l7 = nullptr;
void uav_barrier(ID3D12GraphicsCommandList* l, ID3D12Resource* r) {
  if (g_bark >= 2 && g_l7 && l == g_bark_list) {
    if (g_bark == 2 || !r) {
      D3D12_GLOBAL_BARRIER gb = {};
      gb.SyncBefore = D3D12_BARRIER_SYNC_COMPUTE_SHADING;
      gb.SyncAfter = D3D12_BARRIER_SYNC_COMPUTE_SHADING;
      gb.AccessBefore = D3D12_BARRIER_ACCESS_UNORDERED_ACCESS;
      gb.AccessAfter = D3D12_BARRIER_ACCESS_UNORDERED_ACCESS;
      D3D12_BARRIER_GROUP g = {};
      g.Type = D3D12_BARRIER_TYPE_GLOBAL;
      g.NumBarriers = 1;
      g.pGlobalBarriers = &gb;
      g_l7->Barrier(1, &g);
      return;
    }
    if (r->GetDesc().Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
      D3D12_BUFFER_BARRIER bb = {};
      bb.SyncBefore = D3D12_BARRIER_SYNC_COMPUTE_SHADING;
      bb.SyncAfter = D3D12_BARRIER_SYNC_COMPUTE_SHADING;
      bb.AccessBefore = D3D12_BARRIER_ACCESS_UNORDERED_ACCESS;
      bb.AccessAfter = D3D12_BARRIER_ACCESS_UNORDERED_ACCESS;
      bb.pResource = r;
      bb.Offset = 0;
      bb.Size = UINT64_MAX;
      D3D12_BARRIER_GROUP g = {};
      g.Type = D3D12_BARRIER_TYPE_BUFFER;
      g.NumBarriers = 1;
      g.pBufferBarriers = &bb;
      g_l7->Barrier(1, &g);
      return;
    }
  }
  D3D12_RESOURCE_BARRIER b = {};
  b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
  b.UAV.pResource = (g_bark == 1 && l == g_bark_list) ? nullptr : r;
  l->ResourceBarrier(1, &b);
}
void transition(ID3D12GraphicsCommandList* l, ID3D12Resource* r, D3D12_RESOURCE_STATES a, D3D12_RESOURCE_STATES b) {
  if (a == b) return;
  D3D12_RESOURCE_BARRIER x = {};
  x.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  x.Transition.pResource = r;
  x.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
  x.Transition.StateBefore = a;
  x.Transition.StateAfter = b;
  l->ResourceBarrier(1, &x);
}
// The game's textures (0.6.0 follow-up, clean/runtime/interop/texture_shape.h): all subresources of a single-mip one, as
// before; subresource 0 alone of one with mips (mip 0 is what the views read and write), so no other mip leaves the
// state the game declared for the texture.
void transition_game(ID3D12GraphicsCommandList* l, ID3D12Resource* r, D3D12_RESOURCE_STATES a, D3D12_RESOURCE_STATES b) {
  if (a == b) return;
  D3D12_RESOURCE_BARRIER x = {};
  x.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  x.Transition.pResource = r;
  x.Transition.Subresource = wgi::barrier_subresource(r);
  x.Transition.StateBefore = a;
  x.Transition.StateAfter = b;
  l->ResourceBarrier(1, &x);
}

// ---------------------------------------------------------------------------------------------------- lazy init
// Runtime configuration (clean/rr-f-live1): wronggpu-rr-runtime.ini next to this library, WrongGPU's own file (the
// proxy's wronggpu-rr.ini is rewritten by the 60.9 settings contract, so the runtime keeps its own). key=value per
// line, ';' / '#' comments, [sections] ignored, keys case-insensitive. The environment variable WG_RR_<KEY> wins over
// the file (test runs). Keys: model (absolute path of model.wgcm), log_evaluates (first evaluates logged in full,
// default 20); clean/rr-f-live2: the variants (body, head, input, ...) stay on speed8's WG_RR_* environment variables;
// test_fail / test_crash (test only: evaluate numbers that fail / raise an exception, comma-separated).
std::wstring widen_utf8(const std::string& s) {
  if (s.empty()) return std::wstring();
  const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), nullptr, 0);
  std::wstring w((size_t)n, L'\0');
  MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), &w[0], n);
  return w;
}
std::string lower(std::string s) {
  for (char& c : s) c = (char)tolower((unsigned char)c);
  return s;
}
std::string trim(const std::string& s) {
  size_t a = 0, b = s.size();
  while (a < b && (s[a] == ' ' || s[a] == '\t' || s[a] == '\r' || s[a] == '\n')) ++a;
  while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\r' || s[b - 1] == '\n')) --b;
  return s.substr(a, b - a);
}
std::wstring cfg_file() { return module_dir() + L"wronggpu-rr-runtime.ini"; }
std::vector<std::pair<std::string, std::string>> read_cfg_file() {
  std::vector<std::pair<std::string, std::string>> e;
  FILE* f = _wfopen(cfg_file().c_str(), L"rb");
  if (!f) return e;
  char line[2048];
  while (fgets(line, sizeof line, f)) {
    std::string l = trim(line);
    if (l.empty() || l[0] == ';' || l[0] == '#' || l[0] == '[') continue;
    const size_t eq = l.find('=');
    if (eq == std::string::npos) continue;
    std::string v = trim(l.substr(eq + 1));
    const size_t c = v.find_first_of(";#");
    if (c != std::string::npos && (c == 0 || v[c - 1] == ' ' || v[c - 1] == '\t')) v = trim(v.substr(0, c));
    e.push_back({lower(trim(l.substr(0, eq))), v});
  }
  fclose(f);
  return e;
}
const std::vector<std::pair<std::string, std::string>>& cfg_entries() {
  static std::vector<std::pair<std::string, std::string>> e;
  static std::once_flag once;
  std::call_once(once, [] { e = read_cfg_file(); });
  return e;
}
std::string cfg(const char* key) {
  std::string env = "WG_RR_";
  for (const char* p = key; *p; ++p) env.push_back((char)toupper((unsigned char)*p));
  char buf[MAX_PATH * 4];
  const DWORD n = GetEnvironmentVariableA(env.c_str(), buf, sizeof buf);
  if (n && n < sizeof buf) return std::string(buf, n);
  std::string v;
  for (const auto& kv : cfg_entries())
    if (kv.first == key) v = kv.second;   // the last one wins
  return v;
}

// ---------------------------------------------------------------------------------------------------- prepared kernels
// release 0.6.0 (owner 2026-11-02: the kernels are prepared inside the launcher). Measured on the RDNA 4 driver: its shader
// cache is kept per program file NAME (another folder or other bytes under the same name find it warm: 47.6 s cold, 67 ms
// warm), and neither a D3D12 pipeline library nor the pipelines' cached blobs hold the compiled code (7 KB / 57 KB for 59
// kernels: references into that cache), so an application-managed cache cannot carry the kernels into a game. The
// launcher therefore runs its helper under the game's program name (WrongGPU_RrPrepareKernelsV1 below): the driver then
// finds the game's kernels ready at the game's first start.
ID3D12PipelineState* make_pso(ID3D12Device* dev, ID3D12RootSignature* rs, const RrKernelBlob* b) {
  D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
  pd.pRootSignature = rs;
  pd.CS = {b->data, b->size};
  ID3D12PipelineState* p = nullptr;
  if (FAILED(dev->CreateComputePipelineState(&pd, IID_PPV_ARGS(&p)))) return nullptr;
  return p;
}

// The helper's run (WrongGPU_RrPrepareKernelsV1): progress per kernel, cancel by the callback's answer
typedef int (*RrPrepareFn)(void* ctx, uint32_t done, uint32_t total, uint32_t pass, const char* kernel, uint32_t ms,
                           uint32_t cached);
struct Prep {
  bool on = false;
  RrPrepareFn cb = nullptr;
  void* ctx = nullptr;
  uint32_t passes = 3;
  uint32_t total = 0;
  uint32_t done = 0;
  uint32_t passes_done = 0;
};
Prep g_prep;
// i15: rr_mode = a registry mode name (exact | accurate | fast | ...). Precedence (wg_shared_settings.h reader rule):
// the shared settings block's slot WG_OPT_RR_MODE when its set_mask bit is set (the menu's choice, or wronggpu.ini as
// seeded by the creator), else WG_RR_MODE, else the key rr_mode (or mode) of wronggpu-rr-runtime.ini (re-read when its
// write time changes, checked at most once per second), else the registry default (accurate). Resolved once per
// evaluate and at instance creation (never per dispatch). Unknown text = the default (logged by the caller).
struct ModeSource {
  wgss::Block block;
  ULONGLONG block_tried_ms = 0;
  bool block_logged = false;
  wgss::Snapshot snap;          // last good snapshot (seqlock: kept on a torn read)
  ULONGLONG ini_checked_ms = 0;
  uint64_t ini_mtime = 1;       // 1 = never read
  std::string ini_mode;
};
ModeSource& MS() {
  static ModeSource m;
  return m;
}
std::string resolve_mode_text(const char** source) {
  ModeSource& ms = MS();
  const ULONGLONG now = GetTickCount64();
  if (!ms.block.present() && (ms.block_tried_ms == 0 || now - ms.block_tried_ms >= 1000)) {
    ms.block_tried_ms = now;
    std::vector<std::string> lines;
    ms.block.open(WG_SS_WRITER_NONE, true, wgss::exe_dir() + L"wronggpu-rr.ini", &lines);
    if (!ms.block_logged)
      for (const std::string& l : lines) log_line(l);
    ms.block_logged = true;
  }
  if (ms.block.present()) ms.block.snapshot(WG_READER_RR_RUNTIME, &ms.snap);
  if (ms.snap.is_set(WG_OPT_RR_MODE)) {
    const int32_t v = ms.snap.v[WG_OPT_RR_MODE];
    *source = ms.snap.source(WG_OPT_RR_MODE);
    return v >= 0 && v < kModeCount ? kWgRrModeNames[v] : "?";
  }
  char buf[64];
  const DWORD n = GetEnvironmentVariableA(kWgOptions[WG_OPT_RR_MODE].env, buf, sizeof buf);
  if (n && n < sizeof buf) { *source = kWgOptions[WG_OPT_RR_MODE].env; return lower(trim(std::string(buf, n))); }
  if (ms.ini_checked_ms == 0 || now - ms.ini_checked_ms >= 1000) {
    ms.ini_checked_ms = now;
    const uint64_t mt = wgss::file_mtime(cfg_file());
    if (mt != ms.ini_mtime) {
      ms.ini_mtime = mt;
      std::string v;
      for (const auto& kv : read_cfg_file())
        if (kv.first == "rr_mode" || kv.first == "mode") v = kv.second;   // the last one wins
      ms.ini_mode = lower(trim(v));
    }
  }
  *source = ms.ini_mode.empty() ? "default" : "wronggpu-rr-runtime.ini";
  return ms.ini_mode;
}
Mode parse_mode_text(const std::string& t, bool* known) {
  *known = true;
  if (t.empty()) return kDefaultMode;
  Mode m = kDefaultMode;
  if (wgrr_rt::parse_mode(t.c_str(), &m)) return m;
  *known = false;
  return kDefaultMode;
}
Mode resolve_mode(const char** source, std::string* text, bool* known) {
  const std::string t = resolve_mode_text(source);
  if (text) *text = t;
  return parse_mode_text(t, known);
}
bool cfg_list_has(const char* key, uint64_t n) {   // "3,7,12" contains n
  const std::string v = cfg(key);
  size_t i = 0;
  while (i < v.size()) {
    size_t j = v.find(',', i);
    if (j == std::string::npos) j = v.size();
    const std::string t = trim(v.substr(i, j - i));
    if (!t.empty() && strtoull(t.c_str(), nullptr, 10) == n) return true;
    i = j + 1;
  }
  return false;
}

// model file: cfg model (WG_RR_MODEL / ini), else model.wgcm (the converter's output) next to this library, else
// wronggpu_rr_model.bin
std::wstring model_path() {
  const std::string m = cfg("model");
  if (!m.empty()) return widen_utf8(m);
  const std::wstring dir = module_dir();
  for (const wchar_t* n : {L"model.wgcm", L"wronggpu_rr_model.wgcm", L"wronggpu_rr_model.bin"}) {
    const std::wstring p = dir + n;
    if (GetFileAttributesW(p.c_str()) != INVALID_FILE_ATTRIBUTES) return p;
  }
  return dir + L"model.wgcm";
}

// the input stage's float-float log2 / exp2 tables (rr_input_host.h ff_tables) appended to the model image
void append_tables(Model* m) {
  static float logt[256][4], expt[65][2];
  rr_input::ff_tables(logt, expt);
  auto add = [&](const char* name, const void* p, size_t n) {
    const uint64_t off = align256(m->image.size());
    m->image.resize(align256(off + n), 0);
    memcpy(m->image.data() + off, p, n);
    m->blobs[name] = {off, n};
  };
  add("in.LOGT", logt, sizeof logt);
  add("in.EXPT", expt, sizeof expt);
}

void init_once() {
  Runtime& rt = R();
  if (rt.init_done) return;
  rt.init_done = true;
  const std::wstring mp = model_path();
  std::string err;
  if (!load_model(mp, &rt.model, &err)) {
    rt.model_err = err;
    log_line("model not loaded (" + narrow(mp) + "): " + err);
  } else {
    append_tables(&rt.model);
    // every name the runtime uses must be there
    std::vector<std::string> need = {"sig", "head.wf", "head.bias", "head.tb", "head.tab", "head.rho", "in.W1", "in.W2",
                                     "in.B1", "in.B2", "in.D", "out.TR", "in.LOGT", "in.EXPT"};
    for (const StageDef& s : kStages) {
      for (const char* n : {".Wqv", ".Tb", ".Wo", ".bo", ".MLP", ".b1", ".b2"}) need.push_back(std::string(s.name) + n);
    }
    for (const std::string& n : need)
      if (rt.model.blob(n) == UINT64_MAX) {
        rt.model.ok = false;
        rt.model_err = "model lacks " + n;
        log_line("model not loaded: lacks " + n);
        break;
      }
    for (const char* n : {"res.T_EFL", "res.TBW", "K1", "sym.C_HDEC"})
      if (rt.model.ok && !rt.model.has_scalar(n)) {
        rt.model.ok = false;
        rt.model_err = std::string("model lacks the symbol ") + n;
        log_line(std::string("model not loaded: lacks the symbol ") + n);
      }
    if (rt.model.ok)
      log_line(fmt("model loaded: %s (%s, %zu bytes device image, %zu blobs, %zu scalars, %s%s, sha256 %s%s)",
                   rt.model.path.c_str(), rt.model.kind == "wgcm" ? "converter output, groups' keyed digests checked"
                                                                  : "own container",
                   rt.model.image.size(), rt.model.blobs.size(), rt.model.scalars.size(),
                   rt.model.synthetic() ? "synthetic" : "converted", rt.model.b2() ? ", build B2" : ", build B1",
                   rt.model.file_sha256.c_str(),
                   rt.model.rho_guarded ? fmt(", %u rho entries by the exact path", rt.model.rho_guarded).c_str() : ""));
  }
  IDXGIFactory1* fac = nullptr;
  if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&fac)))) {
    IDXGIAdapter1* a = nullptr;
    for (UINT i = 0; fac->EnumAdapters1(i, &a) != DXGI_ERROR_NOT_FOUND; ++i) {
      DXGI_ADAPTER_DESC1 d;
      a->GetDesc1(&d);
      std::string note;
      if (rdna4_adapter_desc(d, &note)) {
        rt.adapter_present = true;
        if (!note.empty()) log_line("adapter check: " + note);
      }
      a->Release();
    }
    fac->Release();
  }
  if (!rt.adapter_present) log_line("no RDNA 4 adapter (Radeon RX 9070 / 9060 series): RR not available");
}

void stop_compile() {
  Runtime& rt = R();
  if (rt.compile_thread.joinable()) {
    rt.compile_cancel = true;
    rt.compile_thread.join();
  }
  rt.compile_cancel = false;
  rt.kernels_ready = false;
  rt.modes_ready = false;
  rt.all_ready = false;
}

void drop_device() {
  Runtime& rt = R();
  stop_compile();
  rt.live_resources -= rt.deferred.release_all();
  release(rt.pass_heap);
  drop(rt.pass_dummy);
  for (uint64_t& s : rt.pass_serial) s = 0;
  for (int k = 0; k < K_COUNT; ++k) release(rt.pso[k]);
  release(rt.rs);
  drop(rt.model_buf);
  drop(rt.ags_buf);
  if (rt.rm_thread.joinable()) SetEvent(rt.rm_quit), rt.rm_thread.join();
  release(rt.rm_fence);
  if (rt.rm_ev) CloseHandle(rt.rm_ev), rt.rm_ev = nullptr;
  if (rt.rm_quit) CloseHandle(rt.rm_quit), rt.rm_quit = nullptr;
  if (rt.marker && rt.marker_map) rt.marker->Unmap(0, nullptr);  // only a buffer that was mapped (set_device's Map can fail)
  rt.marker_map = nullptr;
  drop(rt.marker);
  release(rt.l);
  release(rt.al);
  release(rt.fence);
  release(rt.q);
  if (rt.ev) CloseHandle(rt.ev), rt.ev = nullptr;
  release(rt.dev);
}

void init_heap(ID3D12DescriptorHeap* heap) {
  D3D12_CPU_DESCRIPTOR_HANDLE c = heap->GetCPUDescriptorHandleForHeapStart();
  write_ags_uav(c);
  c.ptr += R().inc;
  write_null_pair(c);
  c.ptr += 2 * R().inc;
  for (uint32_t i = 0; i < kSrvArray; ++i, c.ptr += R().inc) write_null_srv(c);
}

// selftest of the driver channel on this device (rr_rt_selftest: fp8 WMMA, driver f32 -> f16 conversion)
bool selftest(std::string* err) {
  Runtime& rt = R();
  ID3D12Resource *buf = nullptr, *rb = nullptr;
  ID3D12DescriptorHeap* heap = nullptr;
  bool ok = SUCCEEDED(make_buffer(4096, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                  D3D12_RESOURCE_STATE_COMMON, &buf)) &&
            SUCCEEDED(make_buffer(4096, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE,
                                  D3D12_RESOURCE_STATE_COPY_DEST, &rb));
  D3D12_DESCRIPTOR_HEAP_DESC hd = {};
  hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
  hd.NumDescriptors = kHeapFixed;
  hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
  ok = ok && SUCCEEDED(rt.dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap)));
  if (ok) {
    init_heap(heap);
    bind(rt.l, heap, buf);
    dispatch(rt.l, K_SELFTEST, 1, std::vector<uint32_t>{256u});
    uav_barrier(rt.l, buf);
    transition(rt.l, buf, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    rt.l->CopyBufferRegion(rb, 0, buf, 0, 4096);
    ok = submit_wait(err);
  }
  if (ok) {
    uint8_t* m = nullptr;
    D3D12_RANGE r{0, 4096};
    ok = SUCCEEDED(rb->Map(0, &r, (void**)&m));
    for (uint32_t lane = 0; ok && lane < 32; ++lane) {
      const uint8_t* p = m + 256 + lane * 48;
      for (int k = 0; k < 8 && ok; ++k) {
        float v;
        memcpy(&v, p + 4 * k, 4);
        ok = v == 16.5f;
      }
      for (int k = 0; k < 4 && ok; ++k) {
        uint32_t h;
        memcpy(&h, p + 32 + 4 * k, 4);
        ok = h == 0x4C204C20u;
      }
    }
    if (m) rb->Unmap(0, nullptr);
    if (!ok && err->empty()) *err = "driver-channel self-test mismatch (fp8 WMMA / f16 conversion)";
  }
  release(heap);
  drop(buf);
  drop(rb);
  return ok;
}

uint32_t set_device_locked(ID3D12Device* given) {
  Runtime& rt = R();
  init_once();
  if (!given) return WG_E_INVALID_PARAMETER;
  if (rt.dev) {
    ID3D12Device* d = nullptr;
    // the device under the given one (an interposer may wrap it): our queue tells
    ID3D12CommandQueue* tq = nullptr;
    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    bool same = SUCCEEDED(given->CreateCommandQueue(&qd, IID_PPV_ARGS(&tq))) &&
                SUCCEEDED(tq->GetDevice(IID_PPV_ARGS(&d))) && d == rt.dev;
    release(tq);
    release(d);
    if (same) return rt.device_failed ? WG_E_ADAPTER_NOT_SUPPORTED : WG_OK;
    if (rt.live_instances) {
      log_line("set-device with a different device while instances are alive: refused");
      return WG_E_INVALID_STATE;
    }
    drop_device();
  }
  rt.device_failed = true;
  std::string err;
  D3D12_COMMAND_QUEUE_DESC qd = {};
  qd.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
  if (FAILED(given->CreateCommandQueue(&qd, IID_PPV_ARGS(&rt.q))) || FAILED(rt.q->GetDevice(IID_PPV_ARGS(&rt.dev)))) {
    log_line("set-device: no compute queue on the game's device");
    return WG_E_D3D_API;
  }
  {
    LUID luid = rt.dev->GetAdapterLuid();
    IDXGIFactory4* fac = nullptr;
    IDXGIAdapter1* a = nullptr;
    if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&fac))) && SUCCEEDED(fac->EnumAdapterByLuid(luid, IID_PPV_ARGS(&a)))) {
      DXGI_ADAPTER_DESC1 d;
      a->GetDesc1(&d);
      rt.adapter_name = narrow(d.Description);
      std::string note;
      const bool rdna4 = rdna4_adapter_desc(d, &note);
      if (!note.empty()) log_line("set-device: " + note);
      if (!rdna4) {
        log_line("set-device: adapter \"" + rt.adapter_name + "\" is not an RDNA 4 board: RR not available");
        release(a);
        release(fac);
        drop_device();
        rt.device_failed = true;
        return WG_E_ADAPTER_NOT_SUPPORTED;
      }
    }
    release(a);
    release(fac);
  }
  rt.inc = rt.dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
  rt.q->GetTimestampFrequency(&rt.ts_freq);
  bool ok = SUCCEEDED(rt.dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&rt.al))) &&
            SUCCEEDED(rt.dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, rt.al, nullptr, IID_PPV_ARGS(&rt.l))) &&
            SUCCEEDED(rt.dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&rt.fence)));
  rt.ev = CreateEvent(nullptr, FALSE, FALSE, nullptr);
  if (!ok || !rt.ev) {
    log_line("set-device: command objects could not be made");
    drop_device();
    return WG_E_D3D_API;
  }
  D3D12_FEATURE_DATA_D3D12_OPTIONS o = {};
  rt.dev->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &o, sizeof o);
  if (!o.DoublePrecisionFloatShaderOps) {
    log_line("set-device: binary64 shader operations unavailable (exposure / output stage need them)");
    drop_device();
    return WG_E_ADAPTER_NOT_SUPPORTED;
  }
  if (!make_root(&err)) {
    log_line("set-device: " + err);
    drop_device();
    return WG_E_D3D_API;
  }
  std::vector<const RrKernelBlob*> blobs(K_COUNT, nullptr);
  for (int k = 0; k < K_COUNT; ++k) {
    for (uint32_t i = 0; i < kRrKernelCount; ++i)
      if (strcmp(kRrKernels[i].name, kKernelNames[k]) == 0) blobs[k] = &kRrKernels[i];
    if (!blobs[k]) {
      log_line(std::string("set-device: kernel missing from the library: ") + kKernelNames[k]);
      drop_device();
      return WG_E_D3D_API;
    }
  }
  // now: the self-test and pass-through kernels; the frame kernels on the worker (clean/rr-f-live2)
  for (int k : {(int)K_SELFTEST, (int)K_PASS}) {
    if (!(rt.pso[k] = make_pso(rt.dev, rt.rs, blobs[k]))) {
      log_line(std::string("set-device: pipeline state failed: ") + kKernelNames[k]);
      drop_device();
      return WG_E_D3D_API;
    }
    rt.pso[k]->SetPrivateData(kPsoNameGuid, (UINT)strlen(kKernelNames[k]), kKernelNames[k]);
  }
  if (FAILED(make_buffer(256, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                         D3D12_RESOURCE_STATE_COMMON, &rt.ags_buf)) ||
      FAILED(make_buffer(256, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST,
                         &rt.marker))) {
    drop_device();
    return WG_E_D3D_API;
  }
  {
    void* m = nullptr;
    D3D12_RANGE r{0, 256};
    // Issue #67: a failed Map leaves m null; the writes below would dereference it (the Model M runtime checks it too).
    if (FAILED(rt.marker->Map(0, &r, &m)) || !m) {
      drop_device();
      return WG_E_D3D_API;
    }
    rt.marker_map = (volatile uint32_t*)m;
    *rt.marker_map = 0;
    rt.marker_map[2] = rt.marker_map[3] = 0;
    rt.serial = 0;
  }
  {
    char cv[8] = {};
    rt.crumbs = GetEnvironmentVariableA("WG_RR_CRUMBS", cv, sizeof cv) && cv[0] == '1';
    if (rt.crumbs) log_line("set-device: GPU breadcrumbs around every dispatch (WG_RR_CRUMBS=1)");
  }
  if (!rt.rm_thread.joinable() && SUCCEEDED(rt.dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&rt.rm_fence)))) {
    rt.rm_ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    rt.rm_quit = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (rt.rm_ev && rt.rm_quit && SUCCEEDED(rt.rm_fence->SetEventOnCompletion(UINT64_MAX, rt.rm_ev)))
      rt.rm_thread = std::thread(removal_watch);
  }
  err.clear();
  if (!selftest(&err)) {
    log_line("set-device: " + err + " on \"" + rt.adapter_name + "\": RR not available");
    drop_device();
    rt.device_failed = true;
    return WG_E_ADAPTER_NOT_SUPPORTED;
  }
  if (rt.model.ok) {
    ID3D12Resource* up = nullptr;
    const uint64_t n = rt.model.image.size();
    ok = SUCCEEDED(make_buffer(n, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST,
                               &rt.model_buf)) &&
         SUCCEEDED(make_buffer(n, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, &up));
    if (ok) {
      void* m = nullptr;
      D3D12_RANGE none{0, 0};
      up->Map(0, &none, &m);
      memcpy(m, rt.model.image.data(), n);
      up->Unmap(0, nullptr);
      rt.l->CopyBufferRegion(rt.model_buf, 0, up, 0, n);
      transition(rt.l, rt.model_buf, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
      ok = submit_wait(&err);
    }
    drop(up);
    if (!ok) {
      log_line("set-device: model upload failed: " + err);
      drop_device();
      return WG_E_D3D_API;
    }
  }
  rt.device_failed = false;
  {
    rt.kernels_ready = false;
    rt.modes_ready = false;
    rt.all_ready = false;
    rt.compile_cancel = false;
    rt.compile_failed = -1;
    ID3D12Device* dev = rt.dev;
    ID3D12RootSignature* rs = rt.rs;
    // clean/rr-f-s10-modes: the first pass holds the default frame of the mode the runtime starts in (WG_RR_MODE / ini /
    // accurate); the kernels of the other modes' variants (head.fast.rx, stageex.*, output.*.f, resolve.fast*.x) go to
    // the second pass with the opt-in kernels
    {
      const char* msrc = "";
      bool mknown = true;
      rt.start_mode = resolve_mode(&msrc, nullptr, &mknown);
      rt.start_vbits = mode_vbits(rt.start_mode);
      const char* bad = nullptr;
      if (!modes_self_check(&bad))
        log_line(fmt("rr_mode: registry mode row \"%s\" names a kernel variant this runtime lacks; that mode is refused "
                     "(the default is used)", bad ? bad : "?"));
    }
    const unsigned start_vbits = rt.start_vbits;
    rt.compile_thread = std::thread([dev, rs, blobs, start_vbits] {
      Runtime& r = R();
      const ULONGLONG t0 = GetTickCount64();
      // the helper (g_prep.on): its passes only, progress per kernel (the two made at set-device count as done)
      const int npass = g_prep.on ? (int)std::min<uint32_t>(std::max<uint32_t>(g_prep.passes, 1u), 3u) : 3;
      if (g_prep.on) {
        g_prep.total = 2;
        g_prep.done = 2;
        for (int k = 0; k < K_COUNT; ++k)
          if (k != K_SELFTEST && k != K_PASS &&
              (kernel_in_default_frame(k, start_vbits) ? 0 : (kernel_opt_in(k) ? 2 : 1)) < npass)
            ++g_prep.total;
        if (g_prep.cb && g_prep.cb(g_prep.ctx, g_prep.done, g_prep.total, 0, "rt.selftest", 0, 1)) r.compile_cancel = true;
      }
      // three passes: the default frame's kernels (of the mode the runtime starts in), then the other modes' variant kernels,
      // then the opt-in variants' (kernel_opt_in). The strict / exact head variants alone need minutes with a cold driver
      // cache; neither the default frame nor a mode switch waits for them.
      for (int pass = 0; pass < npass && !r.compile_cancel; ++pass) {
      for (int k = 0; k < K_COUNT && !r.compile_cancel; ++k) {
        if (k == K_SELFTEST || k == K_PASS) continue;
        const int kp = kernel_in_default_frame(k, start_vbits) ? 0 : (kernel_opt_in(k) ? 2 : 1);
        if (kp != pass) continue;
        // every kernel slower than 2 s is logged (cold driver caches); WG_RR_PSO_LOG=1: every kernel, also its start
        const bool plog = GetEnvironmentVariableA("WG_RR_PSO_LOG", nullptr, 0) != 0;
        if (plog) log_line(fmt("pso %d %s: start (%u bytes)", k, kKernelNames[k], (unsigned)blobs[k]->size));
        const ULONGLONG tk = GetTickCount64();
        if (!(r.pso[k] = make_pso(dev, rs, blobs[k]))) {
          r.compile_failed = k;
          return;
        }
        r.pso[k]->SetPrivateData(kPsoNameGuid, (UINT)strlen(kKernelNames[k]), kKernelNames[k]);
        const ULONGLONG dk = GetTickCount64() - tk;
        if (plog || dk > 2000) log_line(fmt("pso %d %s: %llu ms", k, kKernelNames[k], (unsigned long long)dk));
        if (g_prep.on) {
          ++g_prep.done;
          // the driver keeps each kernel as it is made: a cancelled or interrupted run goes on from there next time
          if (g_prep.cb && g_prep.cb(g_prep.ctx, g_prep.done, g_prep.total, (uint32_t)pass, kKernelNames[k], (uint32_t)dk, dk < 1000 ? 1u : 0u))
            r.compile_cancel = true;
        }
      }
      if (g_prep.on && !r.compile_cancel) {
        g_prep.passes_done = (uint32_t)pass + 1;
        if (g_prep.cb) g_prep.cb(g_prep.ctx, g_prep.done, g_prep.total, (uint32_t)pass, nullptr, 0, 0);   // pass complete
      }
      if (pass == 0) {
        r.compile_ms = GetTickCount64() - t0;
        if (!r.compile_cancel) r.kernels_ready = true;
      } else if (pass == 1) {
        if (!r.compile_cancel) {
          r.modes_ready = true;
          log_line(fmt("other modes' kernels prepared after %llu ms", (unsigned long long)(GetTickCount64() - t0)));
        }
      } else if (!r.compile_cancel) {
        r.all_ready = true;
        log_line(fmt("opt-in variant kernels prepared after %llu ms", (unsigned long long)(GetTickCount64() - t0)));
      }
      }
    });
    // a warm driver cache needs well under a second: wait a little so that warm starts run RR from the first frame
    char wv[16] = {};
    const DWORD wait_ms = GetEnvironmentVariableA("WG_RR_COMPILE_WAIT_MS", wv, sizeof wv) ? (DWORD)atoi(wv) : 3000u;
    const ULONGLONG t0 = GetTickCount64();
    // WG_RR_COMPILE_WAIT_ALL=1 (test runs): wait for the opt-in variants' kernels too
    const bool wait_all = GetEnvironmentVariableA("WG_RR_COMPILE_WAIT_ALL", nullptr, 0) != 0;
    while (!(wait_all ? rt.all_ready : rt.kernels_ready) && rt.compile_failed < 0 && GetTickCount64() - t0 < wait_ms) Sleep(5);
    if (rt.kernels_ready)
      log_line(fmt("set-device: default frame kernels prepared in %llu ms (opt-in variants continue in the background)",
                   (unsigned long long)rt.compile_ms.load()));
    else if (rt.compile_failed < 0)
      log_line(fmt("set-device: frame kernels still being prepared after %lu ms (cold driver cache, first start: can take "
                   "minutes); frames pass through until they are ready", (unsigned long)wait_ms));
    rt.compile_logged = rt.kernels_ready;
  }
  if (!rt.ready_logged && rt.model.ok) {
    rt.ready_logged = true;
    log_line(fmt("RR runtime ready: %u.%u.%u on \"%s\" (driver-channel self-test passed, %d kernels, model %s, "
                 "speed1 frame: PH2 pool through exact tables, exposure, input stage with direct reads and the "
                 "replay list, input MLP variant E (X with WG_RR_BODY=Z / X), body step E on the struct routes%s%s (WG_RR_BODY=Z: exact step Z, "
                 "X: step X), fast head (WG_RR_HEAD=exact: exact head), output stage fast path, fast extended resolve; same bits as the verified kernels; "
                 "the per-mode variants are chosen at instance creation)",
                 WG_RR_RUNTIME_VERSION0, WG_RR_RUNTIME_VERSION1, WG_RR_RUNTIME_VERSION2, rt.adapter_name.c_str(),
                 (int)K_COUNT, rt.model.kind == "wgcm" ? "converter output" : (rt.model.synthetic() ? "synthetic" : "converted"),
                 kRrE1Direct ? " (E1 reads IN0 directly)" : "", kRrInFold ? " (front-end MLP folded into IN0 for body E)" : ""));
    log_line(fmt("rr_mode: runtime starts in %s (the other modes' kernels follow in the background; the mode is read at "
                 "every evaluate: the shared settings block (menu / wronggpu.ini), then WG_RR_MODE, then rr_mode / mode in "
                 "wronggpu-rr-runtime.ini, default %s; a change applies on the next frame with a history reset)",
                 mode_name(rt.start_mode), mode_name(kDefaultMode)));
  } else if (!rt.model.ok) {
    log_line("set-device: device ready on \"" + rt.adapter_name + "\" but no model: " + rt.model_err);
  }
  return WG_OK;
}

// ---------------------------------------------------------------------------------------------------- instances
struct Layout {
  // fixed for the instance (output size): model image at kModelBase, then the output-size items of 53.1 / 54b.6
  uint64_t model = 0, ch[2] = {}, ph1 = 0, ph2 = 0, ex[2] = {}, ah[2] = {}, th[2] = {}, oh[2] = {}, fixed_end = 0;
  // clean/rr-f-speed1: the exp2 table of the fast resolve (resolve.init), the PH2 pool's exact tables (pool.tables;
  // they depend on the model's symbols only), the fast resolve's fixup bitmap (We x He bits, zero between frames)
  uint64_t rft = 0, ptab = 0, rfix = 0;
  bool ah_on = false, th_on = false, oh_on = false;
  // render-size items (53.9)
  uint32_t rw = 0, rh = 0;
  uint32_t Wh = 0, Hh = 0;   // the history size (for the output scratch)
  uint64_t hid = 0, pm[2] = {}, pe = 0, pn = 0, state_end = 0;
  uint64_t oscr = 0;   // the output fast path's scratch (gpu/output.py scratch_bytes; zero-filled with the render items)
  uint64_t zmk[15] = {};   // body step Z: per-window fixup marks + count per stage (zero-filled with the render items;
                           // the fixup dispatch clears the marks it consumed, so they are zero between frames)
};
constexpr uint64_t kModelBase = 256;

// work buffers of one frame placed by their lifetimes (pass steps, inclusive); aliasing off: one after the other
struct WorkPlan {
  struct Buf { const char* name; uint64_t size, off; int first, last; };
  std::vector<Buf> b;
  uint64_t end = 0, work = 0;   // end: base + work (the arena size the frame needs); work: the work area's size
  int add(const char* name, uint64_t bytes, int first, int last = -1) {
    b.push_back({name, align256(std::max<uint64_t>(bytes, 4)) + 256, 0, first, last < 0 ? first : last});
    return (int)b.size() - 1;
  }
  void use(int i, int step) {
    if (i < 0) return;
    b[i].first = std::min(b[i].first, step);
    b[i].last = std::max(b[i].last, step);
  }
  void place(uint64_t base, bool alias) {
    std::vector<int> order(b.size());
    for (size_t i = 0; i < b.size(); ++i) order[i] = (int)i;
    if (!alias) {
      uint64_t o = 0;
      for (Buf& x : b) x.off = o, o += x.size;
      end = o;
    } else {
      std::stable_sort(order.begin(), order.end(), [&](int x, int y) { return b[x].size > b[y].size; });
      std::vector<int> placed;
      end = 0;
      for (int i : order) {
        std::vector<std::pair<uint64_t, uint64_t>> busy;
        for (int j : placed)
          if (b[j].first <= b[i].last && b[i].first <= b[j].last) busy.push_back({b[j].off, b[j].off + b[j].size});
        std::sort(busy.begin(), busy.end());
        uint64_t o = 0;
        for (auto& r : busy) {
          if (o + b[i].size <= r.first) break;
          o = std::max(o, r.second);
        }
        b[i].off = o;
        end = std::max(end, o + b[i].size);
        placed.push_back(i);
      }
    }
    for (Buf& x : b) x.off += base;
    work = end;
    end = base + work;
  }
  uint64_t off(int i) const { return i < 0 ? 0 : b[i].off; }
};

// pass steps of the frame (liveness of the work buffers)
enum Step {
  S_FILL = 0, S_INGEST, S_POOL, S_EXPO, S_DECODE, S_FEAT, S_REPLAY, S_MLP, S_STAGE0,
  S_HEAD = S_STAGE0 + 15, S_OUTMAIN, S_OUTPOOL, S_RINGEST, S_RESOLVE, S_STORE
};

struct OutTarget {   // one written game output (colour, separate alpha): its typed UAV or our texture + a copy
  ID3D12Resource* proxy = nullptr;
  DXGI_FORMAT proxy_fmt = DXGI_FORMAT_UNKNOWN;
  uint32_t proxy_w = 0, proxy_h = 0;
};

struct Instance {
  uint32_t id = 0;
  WgRrCreate desc{};
  uint32_t Wo = 0, Ho = 0, We = 0, He = 0, Wh = 0, Hh = 0;
  ID3D12Resource* arena = nullptr;
  uint64_t arena_bytes = 0;
  Layout lay;
  bool fresh = true;                 // first frame: every state image zero-filled before use (53.8)
  bool ah_created = false;           // alpha history created and not yet resolved (54b.6: a reset for 54b.3 / 54b.4)
  ID3D12Resource* upload = nullptr;
  uint8_t* upload_map = nullptr;
  ID3D12DescriptorHeap* heap = nullptr;
  ID3D12DescriptorHeap* null_tmpl = nullptr;   // clean/rr-f-s9-host: CPU-only heap of one slot's null descriptors
  ID3D12QueryHeap* qh = nullptr;
  ID3D12Resource* qrb = nullptr;
  uint64_t* qmap = nullptr;
  uint64_t slot_serial[kSlots] = {};
  bool slot_pending[kSlots] = {};
  uint64_t slot_frame[kSlots] = {};
  uint64_t last_serial = 0;
  OutTarget out[2];                  // 0 colour, 1 separate output alpha
  uint64_t frames = 0, failed = 0;
  std::string tlog_sig;     // clean/release-060 crash-cp2077: the frame's image shapes at the last evaluate
  uint32_t tlog_left = 0;   // evaluates still logged in full after a creation, reset frame or shape change
  double tsum[kTs - 1] = {}, tin[kTsIn + 1] = {}, ttotal = 0;
  uint32_t tcount = 0;
  uint32_t twindows = 0;   // timing windows logged (the first one is 60 frames, then kTimingWindow)
  uint64_t twin_first = 1;
  bool logged_fallback = false, logged_layer = false;
  std::string last_fail;
  std::string map_text;            // offsets of the last recorded frame's buffers (test hook)
  bool head_debug = false;         // WG_RR_HEAD=debug (test runs): strict head, also y2h [Ht][Wt][32] f16 in "dbg_y2h"
  bool head_fast = false;          // default head (unset / WG_RR_HEAD=fast): the head's default (non-strict) matrix step
  bool head_exact = false;         // WG_RR_HEAD=exact: the exact head (weight planes, integer pass, strict fixup)
  bool head18 = true;              // the head's 18-channel reading (rr_d0r.hlsl RR_HEAD18; clean/rr-f-flow default);
                                   // WG_RR_HEAD18=0: page 55's paired reading of HID / CANDW
  bool head_strict = false;        // WG_RR_HEAD=strict: the binary64 strict head alone (same bits as the default)
  bool input_lite = false;         // WG_RR_INPUT=lite (rr_mode fast default): relaxed input features, no replay marks
  bool input_split = false;        // WG_RR_INPUT=split (clean/rr-f-s9-inx): history samples exact, non-history features lite
  bool input_copy = false;         // WG_RR_INPUT=copy at creation (diagnostic): every input through an f32 arena copy, no direct SRV reads
  bool diag_inx = false;           // WG_RR_DIAG_INX=1 at creation (diagnostics, clean/rr-f-s9-inx): the exact input
                                   // features also run into a separate buffer (dx) for bit comparisons in one frame
  bool alias = true;               // WG_RR_ALIAS=0 at creation: work buffers not aliased (test runs read them back)
  bool ts_late = false;            // WG_RR_TSLATE=1 at creation (probe): stage timestamps recorded at the frame's end
  bool fuse_store = true;          // clean/rr-f-s10-tail (WG_RR_FUSESTORE=0 at creation: rt.store as before): with the
                                   // RF_SPEC 2 resolve and a binary16 main output, resolve.fast2 / resolve.fixup store
                                   // it into the game's texture themselves (rt.store's conversion, same bits)
  int ringest = 4;                 // clean/rr-f-s13x-x-flow: the body stage the resolve's ingest joins (kRIngestStage /
                                   // kRIngestStageFast by mode, apply_dflow; WG_RR_RINGEST=<0..15>, A/B)
  bool host_sched = true;          // clean/rr-f-s9-host (WG_RR_HOSTSCHED=0 at creation: the base recording order):
                                   // independent dispatches share barriers (pool + exposure + decode; the resolve's
                                   // game-input ingest inside a low-occupancy body level). Same kernels, same bits.
  bool body_e = true;              // default body (clean/rr-f-speed5; unset / WG_RR_BODY=E): step E (default matrix
                                   // step, not exact to 51.1) on the struct routes
  bool body_z = false;             // WG_RR_BODY=Z at creation: the exact body step Z (bit-identical to step X;
                                   // verification mode)
  bool input_mlp_e = true;         // front-end MLP on variant E: default with body step E; variant X (exact) with
                                   // WG_RR_BODY=Z / X; WG_RR_INPUT_MLP=E / X at creation overrides
  // clean/rr-f-s10-modes: the mode of this instance (rr_mode, fixed at creation) and the kernel variants it resolved to
  Mode mode = MODE_ACCURATE;
  // i15: next-frame switching (a_evaluate). WG_RR_TEST_HOLD_MODES=N at creation (TEST ONLY) treats the other modes'
  // kernels as not prepared for the first N evaluates of this instance (the deferred-switch test); the target of a
  // deferred switch is logged once
  uint64_t hold_modes = 0;
  int deferred_logged = -1;
  unsigned vbits = 0;              // variant set (mode_vbits, per-knob overrides applied): head relaxed / body exact / output Fast / resolve exact
  bool infold = false;             // stagee.IN0 computes the front-end MLP itself (relaxed body in a build with kRrInFold)
  uint32_t twin = kTimingWindow;           // WG_RR_TIMING_WINDOW=N at creation: frames per timing log line (benches)
  bool body_x = false;             // WG_RR_BODY=X at creation (A/B runs): body step X on the struct routes instead of
                                   // step Z (same bits)
  bool fast_paths = true;          // WG_RR_PATHS=verified at creation (A/B runs): the PH2 pool, output stage and
                                   // resolve through the verified kernels (resolve.pool, output.main / pool,
                                   // resolve.main) instead of their fast kernels (same bits)
  uint64_t work_bytes = 0;
  // clean/rr-f-s11-tail: the fast resolve's persistent group count (WG_RR_RESGROUPS=N at creation, tests / sweeps; every
  // tile is computed alone, so any count gives the same bits)
  uint32_t res_groups = kResGroups;
  // clean/rr-f-s11-tail (WG_RR_TAILPREP=0 at creation: the front order): the next frame's history prep (pool.main PH1 ->
  // PH2, the three input decodes into dec_*) is recorded beside the resolve of this frame (PH1 / HID are final after
  // the output / head barriers and nothing writes them until the next frame's output / head; the resolve never touches
  // PH1, PH2, HID or dec_*). The next frame skips its front pool / decode when tail_ready and the key matches (same
  // arena, offsets, sizes). Same kernels on the same bytes: same bits (a reset frame reads no history sample).
  bool tail_prep = true;
  bool tail_ready = false;
  // clean/rr-f-s12-dflow probe (WG_RR_DPROBE=<hex mask> at creation, TIMING ONLY, output wrong): bit s drops the
  // UAV barrier after body stage s
  uint32_t dprobe = 0;
  // clean/rr-f-s12-dflow (opt-in WG_RR_DFLOW=1 at creation since clean/release-060, see apply_dflow): the body stages of dfdrop (bit s: the UAV barrier
  // after stage s) run without their barrier; the windows of the later stages wait on per-row completion counters
  // (rr_dflow.hlsli, kernels staged.*). WG_RR_DFLOW_MASK=<hex> overrides the drop set (bits outside the stages whose
  // staged.* kernels have the dataflow form, or whose next stage's has not, are cleared). Same bits as without.
  bool dflow = false;
  uint32_t dfdrop = 0;
  int dfneg = -1;   // WG_RR_DFLOW_NEG=<stage> (negative control, output wrong): that stage's nearest producer slot one row short
  bool dfall = false;   // WG_RR_DFLOW_ALL=1: every stage with a dataflow form runs it (default: only those that wait or signal)
  uint64_t tail_key[10] = {};
  // clean/rr-f-a14-frame (round 14 track E, every knob read at creation, default off = the base recording):
  // WG_RR_ARENA=uav (E25): the arena stays UNORDERED_ACCESS between frames (no end barrier + transition to COMMON; the
  // start goes UAV -> COPY_DEST for the argument block); =probe (TIMING ONLY, output wrong): no argument-block copy and
  // no arena transitions at all after the first frame
  int arena_mode = 0;
  bool arena_uav = false;          // the arena's state after the last recorded frame (arena_mode 1 / 9)
  // WG_RR_BARK=null | enh | enhbuf (E2): uav_barrier() as a null-resource UAV barrier / an enhanced global barrier
  // (ID3D12GraphicsCommandList7::Barrier, SYNC_COMPUTE_SHADING + ACCESS_UNORDERED_ACCESS both sides) / an enhanced
  // buffer barrier on the resource
  int bark = 0;
  // WG_RR_DPROBE2=<hex> (E3, TIMING ONLY, output wrong): bit 0 drops the barrier features -> replay, 1 replay -> body,
  // 2 head -> output.fast, 3 output.fast -> fallback, 4 fallback -> resolve, 5 resolve -> end
  uint32_t dprobe2 = 0;
  // WG_RR_PREPU_AT=rep | <stage> (E21 / E23): output.prepu recorded beside the replay / body stage <stage> instead of
  // beside the head (it writes only output scratch words nothing before the output stage reads)
  int prepu_at = -1;
  // WG_RR_HIDDEC=1 (E7 f / E23): the next frame's HID decode beside output.fast (after the head barrier) instead of
  // beside the resolve
  bool hiddec_early = false;
  // gated (body network structure, default off): WG_RR_SKIP=E5 (E14 rung 0: D4-F.cin = merged[E4], D4-R.Hc =
  // hmerged[E4]), WG_RR_SHALLOW=5 (rung 1: D3-F.cin = y2[E4], D3-R.Hc = hnew[E4]), =4 (rung 2: D3-F.cin = merged[E3],
  // D3-R.Hc = hmerged[E3]); bit s of skipmask: body stage s not dispatched
  int shallow = 0;
  uint32_t skipmask = 0;
  // gated: WG_RR_DEEPALT=1 | 2 | 3 (E15): on every other frame (never a reset / first / re-laid-out frame) skip
  // E5 + D4 / E4 .. D4-R / E3 .. D3-R; the consumers read the previous frame's values (kept through the whole frame)
  int deepalt = 0;
  uint32_t altmask = 0;
  bool deep_prev_ok = false;
  // WG_RR_TPERIOD=1: also log the frame period (t0 of frame n+1 - t0 of frame n, both completed)
  bool tperiod = false;
  // WG_RR_TOUCH=<hex> (E24, read-only warm-up, same bits): bit 0 CH[1-par] + the TH pair beside D4-F, bit 1 the first
  // 24 MiB of IN0's y2 beside D2-R, bit 2 the model image beside E2, bit 3 the model image beside IN0, bits 4 / 5 the
  // weight span of E3 .. D2-R only beside E2 / IN0
  uint32_t touch = 0;
  bool barbatch = false;
  // gated: WG_RR_DEEPLAG=1 (E16): D3-F / D3-R read the PREVIOUS frame's D4-R y2 / hnew (two persistent buffers by frame
  // parity); a reset / first / re-laid-out frame reads this frame's. =2: also E4 -> E5 on the dataflow counters (the
  // deep chain then holds no barrier; D3-F .. D2-F run beside it)
  int deeplag = 0;
  bool netalt = false, net_prev_ok = false;   // WG_RR_NETALT=1 (E29, fast-only)
  bool lag_prev_ok = false;           // WG_RR_BARBATCH=1 (E2): game-texture transitions batched per direction
  uint64_t pt0[8] = {}, pfr[8] = {};
  double psum = 0;
  uint32_t pcount = 0;
};

uint32_t hbits_pair(uint32_t lo, uint32_t hi) { return (lo & 0xffffu) | (hi << 16); }

Layout fixed_layout(const Instance& in, bool th_on, bool oh_on) {
  Layout L;
  L.model = kModelBase;
  L.Wh = in.Wh;
  L.Hh = in.Hh;
  uint64_t o = align256(kModelBase + R().model.image.size());
  const uint64_t hb = (uint64_t)in.Wh * in.Hh, eb = (uint64_t)in.We * in.He;
  L.ch[0] = o; o = align256(o + hb * 8);
  L.ch[1] = o; o = align256(o + hb * 8);
  L.ph1 = o; o = align256(o + (uint64_t)((in.Wh + 1) / 2) * ((in.Hh + 1) / 2) * 8);
  L.ph2 = o; o = align256(o + (uint64_t)((in.Wh + 3) / 4) * ((in.Hh + 3) / 4) * 8);
  L.ex[0] = o; o += 256;
  L.ex[1] = o; o += 256;
  L.rft = o; o = align256(o + kExp2TableBytes);
  L.ptab = o; o = align256(o + kPoolTableBytes);
  L.rfix = o; o = align256(o + (eb + 31) / 32 * 4);
  L.ah_on = in.desc.alpha_upscaling != 0;
  if (L.ah_on) {
    L.ah[0] = o; o = align256(o + eb * 2);
    L.ah[1] = o; o = align256(o + eb * 2);
  }
  L.th_on = th_on;
  if (th_on) {
    L.th[0] = o; o = align256(o + eb * 8);
    L.th[1] = o; o = align256(o + eb * 8);
  }
  L.oh_on = oh_on;
  if (oh_on) {
    L.oh[0] = o; o = align256(o + eb * 8);
    L.oh[1] = o; o = align256(o + eb * 8);
  }
  L.fixed_end = o;
  return L;
}
void render_layout(Layout* L, uint32_t rw, uint32_t rh) {
  const uint64_t n = (uint64_t)rw * rh;
  uint64_t o = L->fixed_end;
  L->rw = rw;
  L->rh = rh;
  L->hid = o; o = align256(o + n * 4);
  L->pm[0] = o; o = align256(o + n * 8);
  L->pm[1] = o; o = align256(o + n * 8);
  L->pe = o; o = align256(o + n * 4);
  L->pn = o; o = align256(o + n * 4);
  {   // gpu/output.py scratch_bytes(wr, hr, wh, hh): counters + fallback list, then the render records
    const uint64_t hb = (uint64_t)L->Wh * L->Hh;
    const uint64_t lst = (256 + hb * 4 + 255) / 256 * 256;
    L->oscr = o; o = align256(o + lst + n * 16 + (n * 4 + 15) / 16 * 16 + n * 8 + 64);
  }
  {   // body step Z marks: windows of 8 x 8 tokens with a shift < 8 (nwx = (Nx + sx + 7) / 8), + 1 count word
    const uint32_t Wt = 32 * ((rw + 31) / 32), Ht = 32 * ((rh + 31) / 32);
    for (int s = 0; s < 15; ++s) {
      const uint64_t nw = (uint64_t)(((Wt >> kStages[s].level) + 14) / 8) * (((Ht >> kStages[s].level) + 14) / 8);
      L->zmk[s] = o; o = align256(o + (nw + 1) * 4);
    }
  }
  L->state_end = o;
}

// issue #63: every object the instance's recorded frames may use, the descriptor and query heaps included, is released
// only when the frame marker reaches the instance's last serial (pump), or at device removal / device drop. No wait here:
// a frame not yet submitted (destroy in the middle of the game's frame) keeps them instead of stalling the game thread.
void destroy_instance_resources(Instance* in, uint64_t serial) {
  Runtime& rt = R();
  if (in->upload) in->upload->Unmap(0, nullptr), in->upload_map = nullptr;
  if (in->qrb) in->qrb->Unmap(0, nullptr), in->qmap = nullptr;
  const uint64_t done = marker_done();
  if (RetireList<IUnknown>::in_flight(serial, done))
    log_line(fmt("instance %u: frame %llu still in flight at destroy (marker at %llu); its buffers and heaps are kept until "
                 "it completes, or until device removal", in->id, (unsigned long long)serial, (unsigned long long)done));
  IUnknown* bufs[] = {in->arena, in->upload, in->qrb, in->out[0].proxy, in->out[1].proxy};
  IUnknown* heaps[] = {in->heap, in->qh};
  rt.live_resources -= retire_instance(rt.deferred, bufs, 5, heaps, 2, serial, done);
  in->arena = in->upload = in->qrb = in->out[0].proxy = in->out[1].proxy = nullptr;
  in->heap = nullptr;
  in->qh = nullptr;
  release(in->null_tmpl);   // CPU-only heap, a copy source at record time: never used by the GPU
}

bool new_arena(Instance* in, uint64_t bytes, std::string* err) {
  ID3D12Resource* a = nullptr;
  if (FAILED(make_buffer(bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                         D3D12_RESOURCE_STATE_COMMON, &a)))
    return *err = fmt("arena of %llu MiB could not be made", (unsigned long long)(bytes >> 20)), false;
  in->arena = a;
  in->arena_bytes = bytes;
  return true;
}

// ---- the frame's work buffers (shared by the creation-time estimate and evaluate)
struct FrameShape {
  uint32_t rw = 0, rh = 0;
  uint64_t copy_bytes[kRoles] = {};   // input-stage roles that need an f32 copy (linear conversion), else 0
  uint64_t layer_bytes = 0, opac_bytes = 0, asrc_bytes = 0;
  uint64_t out_bytes = 0, aout_bytes = 0;
};
struct FrameBufs {
  int hflag = -1, hwpl = -1;
  int blk = -1, copy[kRoles], partials = -1, mark = -1, dec_ph1 = -1, dec_ph2 = -1, dec_hid = -1, codes = -1, hold = -1,
      x0 = -1, regroup = -1, dbg_y2h = -1, dbg_img = -1, cnet = -1, gates = -1, prec = -1, candw = -1, linear = -1, q = -1, trace = -1,
      layer = -1, opac = -1, asrc = -1, out = -1, aout = -1;
  int dfcnt = -1;   // clean/rr-f-s12-dflow: the row counters (15 x kDfRows) + the error word
  int lagy[2] = {-1, -1}, lagh[2] = {-1, -1};   // a14 E16 (WG_RR_DEEPLAG): D4-R's y2 / hnew by frame parity
  int dx = -1;   // clean/rr-f-s9-inx: WG_RR_DIAG_INX: the exact input pass's codes | hold | marks | PM | Pe | Pn
  int y2[15], merged[15], hnew[15], hmerged[15];
  uint32_t Nx[15] = {}, Ny[15] = {};
  FrameBufs() {
    for (int& v : copy) v = -1;
    for (int s = 0; s < 15; ++s) y2[s] = merged[s] = hnew[s] = hmerged[s] = -1;
  }
};
// the stage words of 51 (blob offsets + per-stage scalars) and the shift rows, once per model (same values as the
// lookups they replace)
static const char* const kSymNames[] = {"K4", "K5", "K7", "K7n", "K8", "K10", "K12", "C_HDEC",
    "C_FCLIP", "C_AMAX", "C_GCL", "C_GCLn", "C_LWrb", "C_LWg", "C_MDL", "C_MDLn", "Gm", "Gd", "Gh", "Gs", "C_SV1",
    "C_SV2", "C_SV3", "C_SV4", "C_SV4n"};
const Runtime::Words& model_words(Runtime& rt, const char* const* sym, int nsym) {
  Runtime::Words& w = rt.words;
  const Model& M = rt.model;
  const std::string key = M.file_sha256 + "|" + M.logical_sha256 + "|" + M.path;
  if (w.ok && w.key == key) return w;
  static const char* const kBo[18] = {".Wqv", ".Tb", ".Wo", ".bo", ".MLP", ".b1", ".b2", ".Wm", ".bm", ".Wa", ".ba",
                                      ".Wup", ".bup", ".Wh", ".bh", nullptr, nullptr, nullptr};
  for (int s = 0; s < 15; ++s) {
    const std::string n = kStages[s].name;
    Runtime::StageWords& t = w.st[s];
    for (int i = 0; i < 15; ++i) {
      const uint64_t v = M.blob(n + kBo[i]);
      t.bo[i] = v == UINT64_MAX ? 0u : (uint32_t)(kModelBase + v);
    }
    t.ra_m = M.scalar(n + ".ra_m");
    t.f_g = M.scalar(n + ".f_g");
    t.zok = M.scalar(n + ".zok");
    t.zthr = M.scalar(n + ".zthr");
    t.zt0 = M.scalar(n + ".zt0");
    t.zt1 = M.scalar(n + ".zt1");
  }
  for (int p = 0; p < 2; ++p)
    for (int r = 0; r < 11; ++r) w.sh[p][r] = M.scalar(fmt("SH.%d.%d", p, r));
  for (int i = 0; i < nsym && i < 32; ++i) w.sym[i] = M.scalar(std::string("sym.") + sym[i]);
  w.key = key;
  w.ok = true;
  return w;
}
// clean/rr-f-s9-host: set when a test host fetches WrongGPU_RrRuntimeTestV1 (the per-evaluate buffer map for
// t_describe is built only then; the game never fetches it)
std::atomic<bool> g_test_used{false};
// clean/rr-f-s9-host: the body stage the resolve's game-input ingest joins. clean/rr-mp1-runtime-deep D1: after E5 (5) in
// modes accurate and exact (with accurate's drop set 0x7f0 below the ingest groups fill E5's idle half of the GPU instead
// of delaying E5's launch: -0.013 / -0.019 ms bench at 1440p Q accurate / exact, byte-identical), after E4 (4) in fast
// (measured neutral there). WG_RR_RINGEST=<0..15> overrides (at creation and after a mode switch).
constexpr int kRIngestStage = 5;
constexpr int kRIngestStageFast = 4;
constexpr uint64_t kTouchMaxPx = 1200000;   // clean/rr-f-a14-frame E24: rt.touch of the model image by default up to this render size
// clean/rr-f-s12-dflow: counter rows per stage (rr_dflow.hlsli RR_DF_ROWS), the default drop set (the decoder side of the
// deep chain between rr_block_wide.hlsl stages: barriers after E5 .. D2-F)
constexpr uint32_t kDfRows = 512;
constexpr uint32_t kDfDefaultDropX = 0x7fcu;  // clean/rr-f-s13x-x-flow: the exact body (stagedx.*): after E2 .. D2-F (0x7e0: -0.051 / -0.026, 0x7fc: -0.074 / -0.019 ms with D4 on wide4)
constexpr uint32_t kDfDefaultDrop = 0x7f0u;   // clean/rr-mp1-runtime-deep D1 (mode accurate): after E4 .. D2-F (E5 waits on E4's
                                              // window rows; with the ingest after E5: kRIngestStage)
constexpr uint32_t kDfDefaultDropFast = 0x7e0u;   // mode fast: after E5 .. D2-F (s12 timing: the encoder-side drops E2 .. E4
                                                  // cost more than they win; E4 + ingest after E5 measured neutral in fast)
void plan_frame(const Instance& in, const FrameShape& f, WorkPlan* P, FrameBufs* B) {
  const uint32_t Wt = 32 * ((f.rw + 31) / 32), Ht = 32 * ((f.rh + 31) / 32);
  const uint64_t nt = (uint64_t)Wt * Ht, rpx = (uint64_t)f.rw * f.rh, hb = (uint64_t)in.Wh * in.Hh;
  const uint64_t w1 = (in.Wh + 1) / 2, h1 = (in.Hh + 1) / 2, w2 = (in.Wh + 3) / 4, h2 = (in.Hh + 3) / 4;
  B->blk = P->add("blk", kBlock, S_FILL, S_STORE);
  for (uint32_t r = 0; r < kRoles; ++r)
    if (f.copy_bytes[r]) B->copy[r] = P->add("copy", f.copy_bytes[r], S_INGEST, S_REPLAY);
  B->partials = P->add("partials", (rpx + 255) / 256 * 8, S_EXPO);
  B->mark = P->add("mark", (nt + 1) * 4, S_DECODE, S_REPLAY);   // replay list: count + marked token indices
  // clean/rr-f-s11-tail: with the tail history prep the decodes live through the whole frame (written beside the resolve
  // for the next frame, which finds them at the same offsets when its plan is the same: Instance::tail_key)
  const bool tp = in.tail_prep && in.fast_paths && !in.diag_inx;
  const int d0 = tp ? (int)S_FILL : (int)S_DECODE, d1 = tp ? (int)S_STORE : (int)S_REPLAY;
  B->dec_ph1 = P->add("dec_ph1", w1 * h1 * 16, d0, d1);
  B->dec_ph2 = P->add("dec_ph2", w2 * h2 * 16, d0, d1);
  B->dec_hid = P->add("dec_hid", rpx * 16, d0, d1);
  // clean/rr-f-s12-c32: with the front-end MLP folded into IN0 (in.infold) the codes are read by IN0 itself (S_STAGE0); a
  // lifetime ending at S_MLP let IN0's outputs alias them, so a window could overwrite codes another window still reads
  B->codes = P->add("codes", nt * 64, S_FEAT, in.infold ? (int)S_STAGE0 : (int)S_MLP);
  B->hold = P->add("hold", nt * 8, S_FEAT, S_STAGE0);
  B->x0 = P->add("x0", nt * 32, S_MLP, S_STAGE0);
  if (in.diag_inx) B->dx = P->add("dx", nt * 64 + nt * 8 + ((nt + 1) * 4 + 15) / 16 * 16 + rpx * 16, S_DECODE, S_STORE);
  auto idx = [&](const char* n) { for (int s = 0; s < 15; ++s) if (strcmp(kStages[s].name, n) == 0) return s; return -1; };
  for (int s = 0; s < 15; ++s) {
    const StageDef& d = kStages[s];
    const int st = S_STAGE0 + s;
    B->Nx[s] = Wt >> d.level;
    B->Ny[s] = Ht >> d.level;
    const uint64_t n = (uint64_t)B->Nx[s] * B->Ny[s];
    B->y2[s] = P->add("y2", n * d.C, st);
    if (d.Cn) B->merged[s] = P->add("merged", n / 4 * d.Cn, st);
    if (d.kind == 'i' || d.kind == 'e' || d.kind == 'r') B->hnew[s] = P->add("hnew", n * 8, st);
    if (d.kind == 'i' || (d.kind == 'e' && d.Cn)) B->hmerged[s] = P->add("hmerged", n / 4 * 8, st);
    // reads
    if (d.kind == 'e') {
      if (d.level == 1) {
        B->regroup = P->add("regroup", nt * 32, st);   // step Z's E1 (rr_block.hlsl) reads the regrouped children
        P->use(B->y2[0], st);
      } else {
        P->use(B->merged[s - 1], st);
      }
      P->use(B->hmerged[s - 1], st);
    } else if (d.kind == 'f') {
      const int enc = d.level == 0 ? 0 : idx(fmt("E%d", d.level).c_str());
      const int coarse = d.level == 4 ? idx("E5") : idx(fmt("D%d-R", d.level + 1).c_str());
      P->use(B->y2[enc], st);
      P->use(B->y2[coarse], st);
    } else if (d.kind == 'r') {
      const int enc = idx(fmt("E%d", d.level).c_str());
      const int coarse = d.level == 4 ? idx("E5") : idx(fmt("D%d-R", d.level + 1).c_str());
      P->use(B->y2[s - 1], st);
      P->use(B->hnew[enc], st);
      P->use(B->hnew[coarse], st);
    }
  }
  // clean/rr-f-a14-frame: the gated rewires' reads; DEEPALT's reused tensors live through the whole frame (kept for the
  // next frame)
  if (in.shallow == 6) P->use(B->merged[4], S_STAGE0 + 6), P->use(B->hmerged[4], S_STAGE0 + 7);
  if (in.shallow == 5) P->use(B->y2[4], S_STAGE0 + 8), P->use(B->hnew[4], S_STAGE0 + 9);
  if (in.shallow == 4) P->use(B->merged[3], S_STAGE0 + 8), P->use(B->hmerged[3], S_STAGE0 + 9);
  if (in.deeplag) {
    const uint64_t n7 = (uint64_t)B->Nx[7] * B->Ny[7];
    for (int k = 0; k < 2; ++k)
      B->lagy[k] = P->add("lagy", n7 * kStages[7].C, S_FILL, S_STORE), B->lagh[k] = P->add("lagh", n7 * 8, S_FILL, S_STORE);
  }
  if (in.deepalt) {
    const int k = in.deepalt == 3 ? 9 : 7;
    for (int i : {B->y2[k], B->hnew[k]}) P->use(i, S_FILL), P->use(i, S_STORE);
  }
  P->use(B->y2[14], S_HEAD);
  P->use(B->hnew[0], S_HEAD);
  P->use(B->hnew[13], S_HEAD);
  B->cnet = P->add("cnet", nt * 8, S_HEAD, S_OUTMAIN);
  // the exact head's marks ([window][wave] u32, every recurrent shift) and the weights' digit planes
  B->hflag = P->add("hflag", (uint64_t)((Wt + 7) / 8 + 1) * ((Ht + 7) / 8 + 1) * 16, S_HEAD, S_HEAD);
  B->hwpl = P->add("hwpl", 32 * 1536 + 32 * 16, S_HEAD, S_HEAD);   // 28 weight pairs + 4 merged W2 pairs (RR_HCOMPACT)
  if (in.head_debug) {   // kept to the end of the frame (test dumps)
    B->dbg_y2h = P->add("dbg_y2h", nt * 64, S_HEAD, S_STORE);
    B->dbg_img = P->add("dbg_img", rpx * 24, S_HEAD, S_STORE);
  }
  B->gates = P->add("gates", rpx * 4, S_HEAD, S_RESOLVE);
  B->prec = P->add("prec", rpx * 8, S_HEAD, S_RESOLVE);
  B->candw = P->add("candw", rpx * 4, S_HEAD, S_OUTMAIN);
  if (in.netalt)   // a14 E29: the head results live through the whole frame (read again by the next, skipped frame)
    for (int i : {B->cnet, B->gates, B->prec, B->candw}) P->use(i, S_FILL), P->use(i, S_STORE);
  B->linear = P->add("linear", hb * 12 + 64, S_OUTMAIN, S_OUTPOOL);
  B->q = P->add("q", hb * 6 + 64, S_OUTMAIN, S_RESOLVE);
  B->trace = P->add("trace", hb * 16 + 64, S_OUTMAIN);
  if (f.layer_bytes) B->layer = P->add("layer", f.layer_bytes, S_RINGEST, S_RESOLVE);
  if (f.opac_bytes) B->opac = P->add("opac", f.opac_bytes, S_RINGEST, S_RESOLVE);
  if (f.asrc_bytes) B->asrc = P->add("asrc", f.asrc_bytes, S_RINGEST, S_RESOLVE);
  if (in.host_sched) {   // the exposure partials stay live through the decode (exposure.finish runs beside it);
                         // the resolve's ingest copies are written during body stage kRIngestStage
    P->use(B->partials, S_DECODE);
    for (int i : {B->layer, B->opac, B->asrc}) P->use(i, in.ringest == 15 ? (int)S_REPLAY : (int)S_STAGE0 + in.ringest);
  }
  B->out = P->add("out", f.out_bytes, S_RESOLVE, S_STORE);
  if (f.aout_bytes) B->aout = P->add("aout", f.aout_bytes, S_RESOLVE, S_STORE);
  if (in.dflow) {
    // clean/rr-f-s12-dflow: the counters (cleared beside the input features), and no two buffers that the barrier-free
    // stretch of the body touches share memory: a slow window of an earlier stage may still read a buffer while a
    // later stage writes. Every buffer live anywhere in [first dropped stage, last dropped stage + 1] lives through
    // all of it.
    B->dfcnt = P->add("dfcnt", (uint64_t)(15 * kDfRows + 64) * 4, S_FEAT, S_STAGE0 + 14);
    int lo = 15, hi = -1;
    for (int t = 0; t < 14; ++t)
      if ((in.dfdrop >> t) & 1) lo = std::min(lo, t), hi = std::max(hi, t + 1);
    const int a = S_STAGE0 + lo, b = S_STAGE0 + hi;
    for (WorkPlan::Buf& x : P->b)
      if (x.first <= b && a <= x.last) x.first = std::min(x.first, a), x.last = std::max(x.last, b);
  }
}

// i15: the mode-dependent instance fields, set at creation and again by a next-frame mode switch (a_evaluate): the
// variant set (mode_vbits + the per-knob overrides WG_RR_HEADV=exact|relaxed, WG_RR_BODYV=exact|relaxed,
// WG_RR_OUTV=default|fast, WG_RR_RESV=hw|exact), the input features (the mode's unless WG_RR_INPUT is set: exact | lite |
// split | copy, copy = exact through f32 copies, any other value = exact; clean/rr-f-s9-inx, owner 2026-10-28: the
// history samples stay exact in the default), the front-end MLP variant and its fold. Returns the overrides text of the
// log line. Creation-only items (history size, WG_RR_HSCALE, the arena) are not touched.
std::string apply_mode(Instance* in, Mode m) {
  in->mode = m;
  in->vbits = mode_vbits(m);
  std::string vover;
  auto knob = [&](const char* name, int group, const char* alt_word, const char* base_word) {
    char kv[16] = {};
    if (!GetEnvironmentVariableA(name, kv, sizeof kv)) return;
    const bool alt = strcmp(kv, alt_word) == 0;
    if (!alt && strcmp(kv, base_word) != 0) return;
    const unsigned before = in->vbits;
    in->vbits = alt ? (in->vbits | (1u << group)) : (in->vbits & ~(1u << group));
    if (in->vbits != before) vover += std::string(" ") + name + "=" + kv;
  };
  knob("WG_RR_HEADV", VG_HEAD, "relaxed", "exact");
  knob("WG_RR_BODYV", VG_BODY, "exact", "relaxed");
  knob("WG_RR_OUTV", VG_OUT, "fast", "default");
  knob("WG_RR_RESV", VG_RES, "exact", "hw");
  char iv[16] = {};
  const bool have_iv = GetEnvironmentVariableA("WG_RR_INPUT", iv, sizeof iv) != 0;
  const int imode = !have_iv ? mode_input(m) : (strcmp(iv, "lite") == 0 ? 1 : (strcmp(iv, "split") == 0 ? 2 : 0));
  in->input_lite = imode == 1;
  in->input_split = imode == 2;
  char mv[8] = {};
  const bool mset = GetEnvironmentVariableA("WG_RR_INPUT_MLP", mv, sizeof mv) != 0;
  in->input_mlp_e = mset ? strcmp(mv, "E") == 0 : in->body_e;   // the exact modes keep variant X
  // the folded stagee.IN0 always computes the variant-E MLP itself: the relaxed body E implies MLP E in such a build.
  // clean/rr-f-lw-fold: the exact body's stageex.IN0 folds it too (kRrInFoldX: rr_block_c32.hlsl RR_INFOLD 1 without
  // RR_B32, the operations of input.mlp.e per token; mode exact's bits are unchanged, its input.mlp.e dispatch is gone)
  in->infold = in->body_e && ((in->vbits & (1u << VG_BODY)) == 0 ? kRrInFold != 0 : kRrInFoldX != 0);
  if (in->infold) in->input_mlp_e = true;
  return vover;
}
// i15: the dataflow body of the instance's variant set (opt-in WG_RR_DFLOW=1 at creation for the step-E body) with the
// a14 DEEPLAG / SKIP adjustments; at creation after the a14 knobs, and after a mode switch.
// clean/release-060 crash-cp2077: off by default. Its windows spin on producer windows of EARLIER dispatches, which is
// deadlock-free only if every group of an earlier dispatch has launched before any group of a later one; D3D12 does not
// promise that (the game's async compute, copy uploads and OS preemption all share the CUs). A whole-PC GPU hang (TDR
// 0x141, DEVICE_HUNG in the command list that runs RR) in Cyberpunk right after a save load matched that failure; the
// barriers cost ~0.035 ms at 1440p Q and give the same bits.
void apply_dflow(Instance* in) {
  in->dflow = false;
  in->dfdrop = 0;
  {   // clean/rr-mp1-runtime-deep D1: the ingest's body stage by mode (WG_RR_RINGEST overrides); it never joins a stage that
      // may be skipped (WG_RR_SKIP / SHALLOW / DEEPALT): then the stage before the first skipped one
    char rv[8] = {};
    in->ringest = in->mode == MODE_FAST ? kRIngestStageFast : kRIngestStage;
    if (GetEnvironmentVariableA("WG_RR_RINGEST", rv, sizeof rv)) in->ringest = std::max(0, std::min(15, atoi(rv)));
    const uint32_t anyskip = in->skipmask | in->altmask;
    if (anyskip && ((anyskip >> in->ringest) & 1)) {
      int f0 = 0;
      while (!((anyskip >> f0) & 1)) ++f0;
      in->ringest = std::max(0, f0 - 1);
    }
  }
  {
    char dfv[16] = {}, dmv[16] = {};
    // opt-in for the step-E body (accurate, fast: stagee.* / staged.*; clean/rr-f-s13x-x-flow: exact too, stageex.* /
    // stagedx.*, the same bits); default: every barrier, stagee.* / stageex.* only
    const unsigned dfk = (in->vbits & (1u << VG_BODY)) ? kRrDflowMaskX : kRrDflowMask;
    const bool dfopt = GetEnvironmentVariableA("WG_RR_DFLOW", dfv, sizeof dfv) && strcmp(dfv, "1") == 0;
    if (in->body_e && !dfopt)
      log_line(fmt("instance %u: body with every UAV barrier (the dataflow body is opt-in: WG_RR_DFLOW=1)", in->id));
    if (in->body_e && dfopt) {
      uint32_t want = (in->vbits & (1u << VG_BODY)) ? kDfDefaultDropX : (in->mode == MODE_FAST ? kDfDefaultDropFast : kDfDefaultDrop);
      if (GetEnvironmentVariableA("WG_RR_DFLOW_MASK", dmv, sizeof dmv)) want = (uint32_t)strtoul(dmv, nullptr, 16);
      uint32_t drop = 0;
      for (int t = 0; t < 14; ++t)
        if (((want >> t) & 1) && ((dfk >> t) & 1) && ((dfk >> (t + 1)) & 1)) drop |= 1u << t;
      in->dflow = true;   // WG_RR_DFLOW_MASK=0: the staged.* kernels with every barrier kept (their overhead alone)
      in->dfdrop = drop;
      if (GetEnvironmentVariableA("WG_RR_DFLOW_NEG", dmv, sizeof dmv)) in->dfneg = atoi(dmv);
      in->dfall = GetEnvironmentVariableA("WG_RR_DFLOW_ALL", dmv, sizeof dmv) && strcmp(dmv, "1") == 0;
      log_line(fmt("instance %u: dataflow body (WG_RR_DFLOW): barriers dropped after stages 0x%x (asked 0x%x, kernels 0x%x)",
                   in->id, drop, want, dfk));
    }
  }
  // a14 E16 DEEPLAG=2: E4 -> E5 on the counters too, so nothing in the deep chain holds a barrier and D3-F (reading the
  // previous frame's D4-R) starts beside it
  if (in->deeplag == 2 && in->dflow) in->dfdrop |= 1u << 4;
  if (in->skipmask) {   // a skipped stage signals no counters: barriers kept around the skipped run
    for (int t = 0; t < 15; ++t)
      if (((in->skipmask >> t) & 1) || (t + 1 < 15 && ((in->skipmask >> (t + 1)) & 1))) in->dfdrop &= ~(1u << t);
  }
}

uint32_t create_locked(const WgRrCreate* d, void** out) {
  Runtime& rt = R();
  init_once();
  *out = nullptr;
  if (!d || d->size < sizeof(WgRrCreate)) return WG_E_INVALID_PARAMETER;
  if (!rt.dev || rt.device_failed) return WG_E_DEVICE_NOT_CREATED;
  if (!rt.model.ok) return WG_E_FEATURE_NOT_SUPPORTED;
  if (!d->output_width || !d->output_height || d->output_width > 16384 || d->output_height > 16384)
    return WG_E_INVALID_PARAMETER;
  pump();
  Instance* in = new Instance();
  in->id = rt.next_instance++;
  in->desc = *d;
  in->Wo = d->output_width;
  in->Ho = d->output_height;
  in->We = in->Wo + (in->Wo % 2);
  in->He = in->Ho + (in->Ho % 2);
  in->Wh = 3 * in->We / 2;
  in->Hh = 3 * in->He / 2;
  in->ah_created = d->alpha_upscaling != 0;
  {
    char hv[16] = {};
    GetEnvironmentVariableA("WG_RR_HEAD", hv, sizeof hv);
    in->head_exact = strcmp(hv, "exact") == 0;
    in->head_debug = strcmp(hv, "debug") == 0;
    in->head_strict = strcmp(hv, "strict") == 0;
    in->head_fast = !in->head_exact && !in->head_debug && !in->head_strict;   // fast head is the default
    char h18[8] = {};
    in->head18 = !(GetEnvironmentVariableA("WG_RR_HEAD18", h18, sizeof h18) != 0 && h18[0] == '0');
    // clean/rr-f-s10-modes: rr_mode (exact | accurate | fast) picks the kernel variants of this instance: head (exact /
    // relaxed), body (relaxed / exact), output stage (default / OF_HWRCP + OF_REPROJ_MAX), resolve (hardware exp2 and
    // reciprocal / correctly rounded) and the input features (exact / lite). Per-knob overrides for tests (environment):
    // WG_RR_INPUT, WG_RR_HEADV=exact|relaxed, WG_RR_BODYV=exact|relaxed, WG_RR_OUTV=default|fast, WG_RR_RESV=hw|exact
    const char* msrc = "";
    std::string mtext;
    bool mknown = true;
    in->mode = resolve_mode(&msrc, &mtext, &mknown);
    if (!mknown) log_line(fmt("instance %u: rr_mode \"%s\" (%s) not recognised; using %s", in->id, mtext.c_str(), msrc,
                              mode_name(kDefaultMode)));
    {
      char hm[16] = {};
      if (GetEnvironmentVariableA("WG_RR_TEST_HOLD_MODES", hm, sizeof hm)) in->hold_modes = strtoull(hm, nullptr, 10);
    }
    {   // a14 E19 (fast-only, WG_RR_HSCALE=1 in rr_mode fast): the colour history at 1x the output instead of 1.5x
      char hsv2[8] = {};
      if (in->mode == MODE_FAST && GetEnvironmentVariableA("WG_RR_HSCALE", hsv2, sizeof hsv2) && strcmp(hsv2, "1") == 0) {
        in->Wh = in->We;
        in->Hh = in->He;
        log_line(fmt("instance %u: WG_RR_HSCALE=1: history %ux%u (fast-only test)", in->id, in->Wh, in->Hh));
      }
    }
    char iv[16] = {};
    const bool have_iv = GetEnvironmentVariableA("WG_RR_INPUT", iv, sizeof iv) != 0;
    in->input_copy = have_iv && strcmp(iv, "copy") == 0;   // WG_RR_INPUT=copy: exact through f32 copies (diagnostic)
    char av[8] = {};
    in->alias = !(GetEnvironmentVariableA("WG_RR_ALIAS", av, sizeof av) && strcmp(av, "0") == 0);
    char dxv[8] = {};
    in->diag_inx = GetEnvironmentVariableA("WG_RR_DIAG_INX", dxv, sizeof dxv) && strcmp(dxv, "1") == 0;
    char hsv[8] = {}, tlv[8] = {};
    in->ts_late = GetEnvironmentVariableA("WG_RR_TSLATE", tlv, sizeof tlv) && strcmp(tlv, "1") == 0;
    in->host_sched = !(GetEnvironmentVariableA("WG_RR_HOSTSCHED", hsv, sizeof hsv) && strcmp(hsv, "0") == 0);
    in->ringest = kRIngestStage;   // set by mode in apply_dflow (clean/rr-mp1-runtime-deep D1); 15: beside the replay (a14 E21)
    if (GetEnvironmentVariableA("WG_RR_RINGEST", hsv, sizeof hsv)) in->ringest = std::max(0, std::min(15, atoi(hsv)));
    char fsv[8] = {};
    in->fuse_store = !(GetEnvironmentVariableA("WG_RR_FUSESTORE", fsv, sizeof fsv) && strcmp(fsv, "0") == 0);
    char pv[16] = {};
    in->fast_paths = !(GetEnvironmentVariableA("WG_RR_PATHS", pv, sizeof pv) && strcmp(pv, "verified") == 0);
    char bv[8] = {};
    const bool bset = GetEnvironmentVariableA("WG_RR_BODY", bv, sizeof bv) != 0;
    in->body_x = bset && strcmp(bv, "X") == 0;
    in->body_z = bset && strcmp(bv, "Z") == 0;
    in->body_e = !in->body_x && !in->body_z;    // step E is the default body (unset, E or anything else)
    // i15: the mode's variants, input features and front-end MLP fold (shared with the next-frame switch)
    const std::string vover = apply_mode(in, in->mode);
    char tv[16] = {};
    if (GetEnvironmentVariableA("WG_RR_TIMING_WINDOW", tv, sizeof tv)) in->twin = std::max(1, std::min(100000, atoi(tv)));
    char rgv[16] = {}, tpv[8] = {};
    if (GetEnvironmentVariableA("WG_RR_RESGROUPS", rgv, sizeof rgv)) in->res_groups = (uint32_t)std::max(1, std::min(65535, atoi(rgv)));
    in->tail_prep = !(GetEnvironmentVariableA("WG_RR_TAILPREP", tpv, sizeof tpv) && strcmp(tpv, "0") == 0);
    char dpv[16] = {};
    if (GetEnvironmentVariableA("WG_RR_DPROBE", dpv, sizeof dpv)) in->dprobe = (uint32_t)strtoul(dpv, nullptr, 16);
    {   // clean/rr-f-a14-frame knobs (see Instance)
      char v[32] = {};
      if (GetEnvironmentVariableA("WG_RR_ARENA", v, sizeof v)) in->arena_mode = strcmp(v, "uav") == 0 ? 1 : strcmp(v, "probe") == 0 ? 9 : strcmp(v, "probe2") == 0 ? 8
                     : strcmp(v, "blk") == 0 ? 2 : strcmp(v, "blkgu") == 0 ? 3 : 0;
      v[0] = 0;
      if (GetEnvironmentVariableA("WG_RR_BARK", v, sizeof v))
        in->bark = strcmp(v, "null") == 0 ? 1 : (strcmp(v, "enh") == 0 ? 2 : (strcmp(v, "enhbuf") == 0 ? 3 : 0));
      if (in->bark >= 2) {
        D3D12_FEATURE_DATA_D3D12_OPTIONS12 o12 = {};
        const bool ok12 = SUCCEEDED(rt.dev->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS12, &o12, sizeof o12)) && o12.EnhancedBarriersSupported;
        log_line(fmt("instance %u: WG_RR_BARK enhanced barriers %s", in->id, ok12 ? "supported" : "NOT supported (legacy kept)"));
        if (!ok12) in->bark = 0;
      }
      if (GetEnvironmentVariableA("WG_RR_DPROBE2", v, sizeof v)) in->dprobe2 = (uint32_t)strtoul(v, nullptr, 16);
      if (GetEnvironmentVariableA("WG_RR_PREPU_AT", v, sizeof v)) in->prepu_at = strcmp(v, "rep") == 0 ? 100 : std::max(-1, std::min(14, atoi(v)));
      // default on (a14 E7 f, bit-identical: -0.011 to -0.014 / about -0.013 ms at 1080p / 1440p Q); WG_RR_HIDDEC=0: beside
      // the resolve as before
      in->hiddec_early = !(GetEnvironmentVariableA("WG_RR_HIDDEC", v, sizeof v) && strcmp(v, "0") == 0);
      if (GetEnvironmentVariableA("WG_RR_SKIP", v, sizeof v) && strcmp(v, "E5") == 0) in->shallow = 6, in->skipmask = 1u << 5;
      if (GetEnvironmentVariableA("WG_RR_SHALLOW", v, sizeof v)) {
        if (atoi(v) == 5) in->shallow = 5, in->skipmask = 0xe0u;        // E5, D4-F, D4-R
        if (atoi(v) == 4) in->shallow = 4, in->skipmask = 0xf0u;        // E4, E5, D4-F, D4-R
      }
      if (GetEnvironmentVariableA("WG_RR_DEEPALT", v, sizeof v)) {
        in->deepalt = std::max(0, std::min(3, atoi(v)));
        in->altmask = in->deepalt == 1 ? 0xe0u : (in->deepalt == 2 ? 0xf0u : (in->deepalt == 3 ? 0x3f8u : 0u));
      }
      // the resolve's ingest never joins a stage that may be skipped: the stage before the first skipped one
      const uint32_t anyskip = in->skipmask | in->altmask;
      if (anyskip && ((anyskip >> in->ringest) & 1)) {
        int f0 = 0;
        while (!((anyskip >> f0) & 1)) ++f0;
        in->ringest = std::max(0, f0 - 1);
      }
      in->tperiod = GetEnvironmentVariableA("WG_RR_TPERIOD", v, sizeof v) && strcmp(v, "1") == 0;
      if (GetEnvironmentVariableA("WG_RR_FEATPROBE", v, sizeof v) && strcmp(v, "1") == 0) {   // a14 E2 / E13 / E25 feature log
        D3D12_FEATURE_DATA_D3D12_OPTIONS12 o12 = {};
        D3D12_FEATURE_DATA_D3D12_OPTIONS16 o16 = {};
        D3D12_FEATURE_DATA_D3D12_OPTIONS21 o21 = {};
        const bool k12 = SUCCEEDED(rt.dev->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS12, &o12, sizeof o12));
        const bool k16 = SUCCEEDED(rt.dev->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS16, &o16, sizeof o16));
        const bool k21 = SUCCEEDED(rt.dev->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS21, &o21, sizeof o21));
        log_line(fmt("instance %u: features: enhanced barriers %d (query %d), GPU upload heap %d (query %d), work graphs tier 0x%x (query %d)",
                     in->id, o12.EnhancedBarriersSupported ? 1 : 0, k12 ? 1 : 0, o16.GPUUploadHeapSupported ? 1 : 0, k16 ? 1 : 0,
                     (unsigned)o21.WorkGraphsTier, k21 ? 1 : 0));
      }
      {   // a14 E24 default (bit-identical): the model-image warm-up beside E2 for render sizes up to kTouchMaxPx (1080p Q:
          // -0.017 to -0.027 ms; 1440p Q: +0.011 to +0.020 ms, so off there); WG_RR_TOUCH=<hex> overrides (0: off)
        const uint64_t crw = d->render_width ? d->render_width : d->output_width, crh = d->render_height ? d->render_height : d->output_height;
        in->touch = crw * crh <= kTouchMaxPx ? 4u : 0u;
      }
      if (GetEnvironmentVariableA("WG_RR_TOUCH", v, sizeof v)) in->touch = (uint32_t)strtoul(v, nullptr, 16);
      in->barbatch = GetEnvironmentVariableA("WG_RR_BARBATCH", v, sizeof v) && strcmp(v, "1") == 0;
      if (GetEnvironmentVariableA("WG_RR_DEEPLAG", v, sizeof v)) in->deeplag = std::max(0, std::min(2, atoi(v)));
      in->netalt = in->mode == MODE_FAST && GetEnvironmentVariableA("WG_RR_NETALT", v, sizeof v) && strcmp(v, "1") == 0;
      if (in->arena_mode || in->bark || in->dprobe2 || in->prepu_at >= 0 || in->hiddec_early || in->skipmask || in->altmask)
        log_line(fmt("instance %u: a14-frame knobs: arena %d, bark %d, dprobe2 0x%x, prepu_at %d, hiddec %d, shallow %d skip 0x%x, "
                     "deepalt %d alt 0x%x, ringest %d", in->id, in->arena_mode, in->bark, in->dprobe2, in->prepu_at,
                     in->hiddec_early ? 1 : 0, in->shallow, in->skipmask, in->deepalt, in->altmask, in->ringest));
    }
    apply_dflow(in);   // i15: the dataflow body of the mode (shared with the next-frame switch)
    log_line(fmt("instance %u: ingest after body stage %d", in->id, in->ringest));
    log_line(fmt("instance %u: rr_mode=%s (%s%s): head %s, input %s, body %s, output %s, resolve %s", in->id, mode_name(in->mode),
                 msrc, vover.c_str(), (in->vbits >> VG_HEAD) & 1 ? "relaxed (RR_HVARL 7)" : "exact",
                 in->input_lite ? "lite" : (in->input_split ? "split (history exact)" : (in->input_copy ? "exact (copy)" : "exact")),
                 (in->vbits >> VG_BODY) & 1 ? (in->infold ? "exact arithmetic (front-end MLP folded)" : "exact arithmetic (front-end MLP separate)")
                                            : (in->infold ? "relaxed arithmetic (front-end MLP folded)" : "relaxed arithmetic"),
                 (in->vbits >> VG_OUT) & 1 ? "OF_HWRCP + OF_REPROJ_MAX" : "default (exact)",
                 (in->vbits >> VG_RES) & 1 ? "correctly rounded exp2 / reciprocal" : "hardware exp2 / reciprocal"));
    log_line(fmt("instance variants: body %s, head %s (%s reading), input %s, input mlp %s, paths %s, alias %d",
                 in->body_e ? "E" : (in->body_x ? "X" : "Z (exact)"),
                 in->head_fast ? "fast" : (in->head_strict ? "strict" : (in->head_debug ? "debug" : "exact")), in->head18 ? "18-channel" : "page-55 paired",
                 in->input_lite ? "lite" : (in->input_split ? "split" : (in->input_copy ? "copy" : "exact")), in->input_mlp_e ? "E" : "X",
                 in->fast_paths ? "fast" : "verified", in->alias ? 1 : 0));
    log_line(fmt("instance tail: resolve groups %u, history prep %s", in->res_groups,
                 in->tail_prep && in->fast_paths && !in->diag_inx ? "beside the previous frame's resolve" : "at the frame's front"));
  }
  in->lay = fixed_layout(*in, false, false);
  const uint32_t rw = d->render_width ? d->render_width : in->Wo, rh = d->render_height ? d->render_height : in->Ho;
  render_layout(&in->lay, rw, rh);
  // the arena for a frame of the creation render size without copies (a larger frame grows it)
  WorkPlan P;
  FrameBufs B;
  FrameShape fs;
  fs.rw = rw;
  fs.rh = rh;
  fs.out_bytes = (uint64_t)in->We * in->He * 16;
  plan_frame(*in, fs, &P, &B);
  P.place(in->lay.state_end, in->alias);
  std::string err;
  bool ok = new_arena(in, align256(P.end), &err);
  bool gu = false;
  if (in->arena_mode == 3) {   // a14 E25: the upload ring in a GPU_UPLOAD (ReBAR) heap when the device has it
    D3D12_FEATURE_DATA_D3D12_OPTIONS16 o16 = {};
    gu = SUCCEEDED(rt.dev->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS16, &o16, sizeof o16)) && o16.GPUUploadHeapSupported;
    log_line(fmt("instance %u: WG_RR_ARENA=blkgu: GPU upload heap %s", in->id, gu ? "supported" : "NOT supported (upload heap)"));
  }
  ok = ok && SUCCEEDED(make_buffer((uint64_t)kSlots * kBlock, gu ? D3D12_HEAP_TYPE_GPU_UPLOAD : D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
                                   gu ? D3D12_RESOURCE_STATE_COMMON : D3D12_RESOURCE_STATE_GENERIC_READ, &in->upload));
  if (ok) {
    D3D12_RANGE none{0, 0};
    in->upload->Map(0, &none, (void**)&in->upload_map);
  }
  ok = ok && SUCCEEDED(make_buffer((uint64_t)kSlots * kTq * 8, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE,
                                   D3D12_RESOURCE_STATE_COPY_DEST, &in->qrb));
  if (ok) {
    D3D12_RANGE r{0, (SIZE_T)kSlots * kTq * 8};
    in->qrb->Map(0, &r, (void**)&in->qmap);
    D3D12_DESCRIPTOR_HEAP_DESC hd = {};
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    hd.NumDescriptors = kHeapFixed + kSlots * kDescPerSlot;
    hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ok = SUCCEEDED(rt.dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&in->heap)));
    D3D12_QUERY_HEAP_DESC qd = {};
    qd.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    qd.Count = kSlots * kTq;
    ok = ok && SUCCEEDED(rt.dev->CreateQueryHeap(&qd, IID_PPV_ARGS(&in->qh)));
  }
  if (ok) {   // one slot's null descriptors in a CPU-only heap: each evaluate copies them in one call (was 40 creates)
    D3D12_DESCRIPTOR_HEAP_DESC td = {};
    td.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    td.NumDescriptors = kDescPerSlot;
    if (SUCCEEDED(rt.dev->CreateDescriptorHeap(&td, IID_PPV_ARGS(&in->null_tmpl)))) {
      D3D12_CPU_DESCRIPTOR_HANDLE h = in->null_tmpl->GetCPUDescriptorHandleForHeapStart();
      for (uint32_t i = 0; i < kSrvArray; ++i, h.ptr += rt.inc) write_null_srv(h);
      for (uint32_t i = 0; i < kPairs; ++i, h.ptr += 2 * rt.inc) write_null_pair(h);
    } else {
      in->null_tmpl = nullptr;
    }
  }
  if (ok) {
    init_heap(in->heap);
    // the model image into the arena (our queue, create time)
    rt.l->CopyBufferRegion(in->arena, kModelBase, rt.model_buf, 0, rt.model.image.size());
    ok = submit_wait(&err);
  }
  if (!ok) {
    log_line(fmt("create: failed (%s)", err.empty() ? "D3D12 objects" : err.c_str()));
    destroy_instance_resources(in, 0);
    delete in;
    return WG_E_D3D_API;
  }
  if (in->head_strict)
    log_line(fmt("instance %u: WG_RR_HEAD=strict: final head through the binary64 strict step alone (same bits)", in->id));
  if (in->head_exact)
    log_line(fmt("instance %u: WG_RR_HEAD=exact: exact final head (51.1 step; slower)", in->id));
  if (in->input_lite)
    log_line(fmt("instance %u: relaxed input features (plain hardware f32 math; Fast mode, WG_RR_INPUT=exact: exact)", in->id));
  if (in->input_split)
    log_line(fmt("instance %u: split input features (history samples exact, colour / guides relaxed; WG_RR_INPUT=exact: exact)", in->id));
  if (!in->alias) log_line(fmt("instance %u: WG_RR_ALIAS=0: work buffers not aliased (test runs)", in->id));
  if (in->body_x) log_line(fmt("instance %u: WG_RR_BODY=X: body step X on the struct routes (A/B runs; same bits)", in->id));
  if (in->body_z) log_line(fmt("instance %u: WG_RR_BODY=Z: exact body step Z (bit-identical to step X; verification mode)", in->id));
  if (in->body_e) log_line(fmt("instance %u: body step E on the struct routes (default; default matrix step, not exact to 51.1; "
                               "WG_RR_BODY=Z / X: exact)", in->id));
  log_line(fmt("instance %u: front-end MLP on variant %s%s", in->id, in->input_mlp_e ? "E (not exact to 51.1)" : "X (exact)",
               in->input_mlp_e == in->body_e ? "" : " (WG_RR_INPUT_MLP)"));
  {
    std::string zs;
    for (int s = 0; s < 15; ++s)
      if (!rt.model.scalar(std::string(kStages[s].name) + ".zok")) zs += std::string(" ") + kStages[s].name;
    if (!zs.empty()) log_line(fmt("instance %u: body step Z host check fails for%s: step X there", in->id, zs.c_str()));
  }
  if (!in->fast_paths)
    log_line(fmt("instance %u: WG_RR_PATHS=verified: PH2 pool, output stage and resolve through the verified kernels "
                 "(A/B runs; same bits)", in->id));
  rt.live_instances++;
  log_line(fmt("create instance %u: output %ux%u, render %ux%u%s, mode %u, packing %u, hdr %u, depth inverted %u, "
               "jittered motion %u, %s depth, alpha upscaling %u, history %ux%u, arena %llu MiB (state %llu MiB, "
               "work %llu MiB)",
               in->id, in->Wo, in->Ho, rw, rh, d->render_width ? "" : " (unknown: output size assumed)", d->mode,
               d->packing, d->colour_hdr, d->depth_inverted, d->mv_jittered, d->hw_depth ? "hardware" : "linear",
               d->alpha_upscaling, in->Wh, in->Hh, (unsigned long long)(in->arena_bytes >> 20),
               (unsigned long long)(in->lay.state_end >> 20), (unsigned long long)((in->arena_bytes - in->lay.state_end) >> 20)));
  *out = in;
  return WG_OK;
}

// ---------------------------------------------------------------------------------------------------- timing
void log_timing(Instance* in) {
  if (!in->tcount) return;
  std::string parts;
  for (uint32_t k = 0; k + 1 < kTs; ++k) parts += fmt("%s%s %.3f", k ? ", " : "", kPassNames[k], in->tsum[k] / in->tcount);
  static const char* const kInNames[kTsIn + 1] = {"in.dec", "in.feat", "in.rep", "in.mlp"};
  for (uint32_t k = 0; k <= kTsIn; ++k) parts += fmt(", %s %.3f", kInNames[k], in->tin[k] / in->tcount);
  if (in->tperiod && in->pcount) parts += fmt(", period %.3f", in->psum / in->pcount);
  in->psum = 0;
  in->pcount = 0;
  log_line(fmt("instance %u timing, %u frames from frame %llu: GPU %.3f ms per frame (%s), render %ux%u, output %ux%u",
               in->id, in->tcount, (unsigned long long)in->twin_first, in->ttotal / in->tcount, parts.c_str(),
               in->lay.rw, in->lay.rh, in->Wo, in->Ho));
  in->tcount = 0;
  in->ttotal = 0;
  for (double& v : in->tsum) v = 0;
  for (double& v : in->tin) v = 0;
}
// GPU time of the newest completed frame (first to last timestamp), smoothed over about 30 frames; read by the
// adapter's frame_gpu_ms from any thread (the RR plugin reports it to the host's overlay)
std::atomic<float> g_frame_gpu_ms{0.0f};

// timestamps of every completed frame (never waits); one log line per 300 frames
void harvest(Instance* in) {
  const uint64_t done = marker_done();
  for (uint32_t s = 0; s < kSlots; ++s) {
    if (!in->slot_pending[s] || in->slot_serial[s] > done) continue;
    in->slot_pending[s] = false;
    const uint64_t* t = in->qmap + s * kTq;
    const double f = 1000.0 / (double)R().ts_freq;
    bool ok = true;
    for (uint32_t k = 1; k < kTs; ++k) ok = ok && t[k] >= t[k - 1];
    if (!ok) continue;
    if (!in->tcount || in->slot_frame[s] < in->twin_first) in->twin_first = in->slot_frame[s];
    for (uint32_t k = 0; k + 1 < kTs; ++k) in->tsum[k] += (double)(t[k + 1] - t[k]) * f;
    {   // input-stage split (in.dec, in.feat, in.rep, in.mlp); a non-monotonic split is clamped, never drops the frame
      uint64_t prev = t[1];
      for (uint32_t k = 0; k <= kTsIn; ++k) {
        const uint64_t nx = k < kTsIn ? std::min(std::max(t[kTs + k], prev), t[2]) : t[2];
        in->tin[k] += (double)(nx - prev) * f;
        prev = nx;
      }
    }
    in->ttotal += (double)(t[kTs - 1] - t[0]) * f;
    {
      const float ms = (float)((double)(t[kTs - 1] - t[0]) * f), prev = g_frame_gpu_ms.load(std::memory_order_relaxed);
      g_frame_gpu_ms.store(prev > 0.0f ? prev + (ms - prev) / 30.0f : ms, std::memory_order_relaxed);
    }
    if (in->tperiod) {   // clean/rr-f-a14-frame: frame period from consecutive completed frames' first timestamps
      const uint64_t fr = in->slot_frame[s];
      const uint32_t k = (uint32_t)(fr & 7), kp = (uint32_t)((fr - 1) & 7), kn = (uint32_t)((fr + 1) & 7);
      in->pt0[k] = t[0];
      in->pfr[k] = fr + 1;
      if (fr && in->pfr[kp] == fr && t[0] > in->pt0[kp]) in->psum += (double)(t[0] - in->pt0[kp]) * f, ++in->pcount;
      if (in->pfr[kn] == fr + 2 && in->pt0[kn] > t[0]) in->psum += (double)(in->pt0[kn] - t[0]) * f, ++in->pcount;
    }
    if (++in->tcount == in->twin) log_timing(in);
  }
}

void destroy_locked(Instance* in) {
  Runtime& rt = R();
  if (!in) return;
  // the last frames' timings: harvest what has completed; no wait (rt.mu is held, and a frame not yet submitted would
  // never complete during it; issue #63). Frames still in flight are left out of the last window.
  harvest(in);
  log_timing(in);   // the last, partial window
  log_line(fmt("destroy instance %u after %llu frames (%llu failed)", in->id, (unsigned long long)in->frames,
               (unsigned long long)in->failed));
  destroy_instance_resources(in, in->last_serial);
  rt.live_instances--;
  delete in;
  pump();
}

// ---------------------------------------------------------------------------------------------------- evaluate
struct Bar {
  ID3D12Resource* r;
  D3D12_RESOURCE_STATES from, to;
};

uint32_t fail(Instance* in, const std::string& why, WgRrFrameResult* res, uint32_t rc) {
  in->failed++;
  if (why != in->last_fail || in->failed <= 3) log_line(fmt("instance %u evaluate failed before any pass: %s", in->id, why.c_str()));
  in->last_fail = why;
  if (res) res->passes_started = 0, res->input_jitter_consumed = 0;
  return rc;
}

// which input-stage roles the frame supplies (62.2 routing -> page 52 roles)
const int kRoleOf[kRoles] = {WG_ROLE_COLOUR, WG_ROLE_DEPTH, WG_ROLE_MOTION, WG_ROLE_SPECULAR_MOTION,
                             WG_ROLE_DIFFUSE_ALBEDO, WG_ROLE_SPECULAR_ALBEDO, WG_ROLE_NORMALS, WG_ROLE_ROUGHNESS,
                             WG_ROLE_DISOCCLUSION, WG_ROLE_BEFORE_TRANSPARENCY, WG_ROLE_G1, WG_ROLE_DOF_GUIDE,
                             WG_ROLE_SPECULAR_HIT_DIST};
const uint32_t kChannels[kRoles] = {3, 1, 2, 2, 3, 3, 3, 1, 1, 3, 1, 1, 1};

bool nrm_decode(uint32_t f) {   // 52.8: 8-bit unorm 1/2/4 channels, 10-10-10-2 unorm, 16-bit 4-channel unorm
  switch (f) {
    case DXGI_FORMAT_R8_UNORM: case DXGI_FORMAT_R8G8_UNORM: case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8_TYPELESS: case DXGI_FORMAT_R8G8_TYPELESS: case DXGI_FORMAT_R8G8B8A8_TYPELESS:
    case DXGI_FORMAT_R10G10B10A2_UNORM: case DXGI_FORMAT_R10G10B10A2_TYPELESS:
    case DXGI_FORMAT_R16G16B16A16_UNORM:
      return true;
    default:
      return false;
  }
}

// a resolve surface copied from a game resource (54b.N read formats): resolve format code, ingest mode, bytes
struct SurfCopy { bool on = false; int role = -1; uint32_t w = 0, h = 0, ch = 0, mode = 0, fmt = 0, first = 0, max3 = 0;
                  DXGI_FORMAT srv = DXGI_FORMAT_UNKNOWN; int32_t ox = 0, oy = 0; uint64_t bytes = 0; };
SurfCopy surf_copy(const WgRrImage& t, int role, uint32_t ch, uint32_t first, int32_t ox, int32_t oy) {
  SurfCopy s;
  s.on = true;
  s.role = role;
  s.srv = srv_format((DXGI_FORMAT)t.format, false);
  s.w = t.width;
  s.h = t.height;
  s.ch = ch;
  s.first = first;
  s.ox = ox;
  s.oy = oy;
  const DXGI_FORMAT f = s.srv;
  if (ch == 4 && is_unorm8x4(f)) s.mode = 2, s.fmt = 2;
  else if (ch == 4 && is_rgb10a2(f)) s.mode = 3, s.fmt = 3;
  else if (is_unorm8(f)) s.mode = 4, s.fmt = 0, s.max3 = 255;
  else if (is_rgb10a2(f)) s.mode = 4, s.fmt = 0, s.max3 = 3;
  else if (is_f16(f)) s.mode = 1, s.fmt = 1;
  else s.mode = 0, s.fmt = 0;
  const uint64_t px = (uint64_t)s.w * s.h;
  s.bytes = (s.mode == 2 || s.mode == 3) ? px * 4 : px * ch * (s.mode == 1 ? 2 : 4);
  return s;
}

uint32_t evaluate_locked(Instance* in, const WgRrFrame* f, WgRrFrameResult* res) {
  Runtime& rt = R();
  if (res) res->passes_started = 0, res->input_jitter_consumed = 0, res->camera_fallback_used = 0;
  if (!in || !f || f->size < sizeof(WgRrFrame)) return WG_E_INVALID_PARAMETER;
  in->frames++;
  pump();
  harvest(in);
  if (!rt.dev || rt.device_failed || !rt.model.ok) return fail(in, "runtime not ready", res, WG_E_DEVICE_NOT_CREATED);
  const uint32_t rw = f->render_width, rh = f->render_height;
  if (!rw || !rh) return fail(in, "zero render size", res, WG_E_INVALID_PARAMETER);
  if (f->output_width != in->Wo || f->output_height != in->Ho)
    return fail(in, "output size differs from the instance's", res, WG_E_INVALID_PARAMETER);
  ID3D12GraphicsCommandList* list = nullptr;
  ID3D12GraphicsCommandList2* list2 = nullptr;
  if (!f->command_list || FAILED(((IUnknown*)f->command_list)->QueryInterface(IID_PPV_ARGS(&list))))
    return fail(in, "no D3D12 command list", res, WG_E_INVALID_PARAMETER);
  if (FAILED(list->QueryInterface(IID_PPV_ARGS(&list2)))) {
    release(list);
    return fail(in, "the command list has no ID3D12GraphicsCommandList2 (completion marker)", res, WG_E_D3D_API);
  }
  struct ListRef { ID3D12GraphicsCommandList *&a; ID3D12GraphicsCommandList2 *&b; ~ListRef() { release(a); release(b); } } lr{list, list2};
  const uint32_t req[] = {WG_ROLE_COLOUR, WG_ROLE_OUTPUT, WG_ROLE_DEPTH, WG_ROLE_MOTION, WG_ROLE_DIFFUSE_ALBEDO,
                          WG_ROLE_SPECULAR_ALBEDO, WG_ROLE_NORMALS};
  for (uint32_t r : req)
    if (!(f->attached & (1u << r)) || !f->images[r].resource)
      return fail(in, fmt("required input missing (role %u)", r), res, WG_E_MISSING_INPUT_PARAMETER);
  const bool rgh_packed = in->desc.packing == 1;
  if (!rgh_packed && !(f->attached & (1u << WG_ROLE_ROUGHNESS)))
    return fail(in, "roughness missing with packing 0", res, WG_E_MISSING_INPUT_PARAMETER);

  // ---- sizes and the reader origins of 52.1.2
  const uint32_t Wt = 32 * ((rw + 31) / 32), Ht = 32 * ((rh + 31) / 32);
  const uint64_t nt = (uint64_t)Wt * Ht;
  const uint32_t par = f->parity & 1u;
  const bool reset = f->reset != 0;
  const WgRrImage* img = f->images;
  auto attached = [&](int role) { return (f->attached & (1u << role)) != 0 && img[role].resource != nullptr; };
  // bug hunt E: the views below are Texture2D views of mip 0: an array, multisampled or other texture is refused before
  // anything is recorded or any descriptor written (as the SR runtime refuses it; texture_shape.h)
  for (int role = 0; role < (int)WG_ROLE_COUNT; ++role)
    if (attached(role) && !wgi::texture_shape_ok(((ID3D12Resource*)img[role].resource)->GetDesc()))
      return fail(in, fmt("role %d: %s", role, wgi::texture_shape_rule()), res, WG_E_INVALID_PARAMETER);
  struct Ing { bool on, copy; int role; uint32_t w, h, ch; int32_t ox, oy; DXGI_FORMAT fmt; };
  Ing ing[kRoles] = {};
  FrameShape fs;
  fs.rw = rw;
  fs.rh = rh;
  for (int r = 0; r < (int)kRoles; ++r) {
    const int role = kRoleOf[r];
    if (!attached(role)) continue;
    if (r == rr_input::R_R && rgh_packed) continue;
    const int src = r == rr_input::R_SV ? WG_ROLE_MOTION
                    : (r == rr_input::R_B || r == rr_input::R_G1 || r == rr_input::R_G2) ? WG_ROLE_COLOUR : role;
    const int32_t ox = (int32_t)img[src].origin_x, oy = (int32_t)img[src].origin_y;
    const WgRrImage& t = img[role];
    if (t.format_class == WG_FMT_UNSUPPORTED) return fail(in, fmt("unsupported format %u (role %d)", t.format, role), res, WG_E_INVALID_PARAMETER);
    if ((uint64_t)ox + rw > t.width || (uint64_t)oy + rh > t.height)
      return fail(in, fmt("input role %d: render rectangle at origin (%d, %d) outside its %ux%u resource", role, ox, oy,
                          t.width, t.height), res, WG_E_INVALID_PARAMETER);
    Ing& g = ing[r];
    g.on = true;
    g.role = role;
    g.ox = ox;
    g.oy = oy;
    g.ch = kChannels[r] + ((r == rr_input::R_N && rgh_packed) ? 1 : 0);
    g.fmt = srv_format((DXGI_FORMAT)t.format, role == WG_ROLE_DEPTH);
    // direct read (the input stage's SRV path) unless the value needs the linear conversion (an f32 copy)
    g.copy = t.conversion == 1 || in->input_copy;
    if (g.copy) {
      g.w = std::min<uint32_t>(t.width, (uint32_t)ox + (r == rr_input::R_R ? Wt : rw));   // roughness padding reads
      g.h = std::min<uint32_t>(t.height, (uint32_t)oy + rh);
      fs.copy_bytes[r] = (uint64_t)g.w * g.h * g.ch * 4;
    } else {
      g.w = t.width;
      g.h = t.height;
    }
  }
  if (rgh_packed) ing[rr_input::R_N].ch = 4;

  // ---- the resolve's inputs (54b): layer (type 39 / 51), opacity, alpha source; outputs
  const bool alpha_up = in->desc.alpha_upscaling != 0;
  const bool layer_on = attached(WG_ROLE_LAYER);
  const bool opac_on = layer_on && attached(WG_ROLE_OPACITY);
  SurfCopy sl, so_, sa;
  if (layer_on) sl = surf_copy(img[WG_ROLE_LAYER], WG_ROLE_LAYER, 4, 0, (int32_t)f->layer_origin_x, (int32_t)f->layer_origin_y);
  if (opac_on) so_ = surf_copy(img[WG_ROLE_OPACITY], WG_ROLE_OPACITY, 4, 0, (int32_t)img[WG_ROLE_OPACITY].origin_x,
                               (int32_t)img[WG_ROLE_OPACITY].origin_y);
  if (alpha_up) {   // 54b.4: the input alpha (channel 0), else the colour input's channel 3 (a format without alpha: 1)
    const int role = attached(WG_ROLE_INPUT_ALPHA) ? WG_ROLE_INPUT_ALPHA : WG_ROLE_COLOUR;
    sa = surf_copy(img[role], role, 1, role == WG_ROLE_INPUT_ALPHA ? 0 : 3, (int32_t)img[role].origin_x,
                   (int32_t)img[role].origin_y);
  }
  fs.layer_bytes = sl.bytes;
  fs.opac_bytes = so_.bytes;
  fs.asrc_bytes = sa.bytes;
  // outputs: the resolve stores the We x He grid into our image in the 54b.N format nearest the destination's
  ID3D12Resource* out_res[2] = {(ID3D12Resource*)img[WG_ROLE_OUTPUT].resource, nullptr};
  const bool aout_on = alpha_up && attached(WG_ROLE_OUTPUT_ALPHA);
  if (aout_on) out_res[1] = (ID3D12Resource*)img[WG_ROLE_OUTPUT_ALPHA].resource;
  struct OutInfo { bool on = false, direct = false, srgb = false; DXGI_FORMAT f = DXGI_FORMAT_UNKNOWN, uf = DXGI_FORMAT_UNKNOWN;
                   uint32_t fmt = 0, ch = 4; D3D12_RESOURCE_DESC d{}; } oi[2];
  for (int k = 0; k < 2; ++k) {
    if (!out_res[k]) continue;
    OutInfo& o = oi[k];
    const WgRrImage& t = img[k == 0 ? WG_ROLE_OUTPUT : WG_ROLE_OUTPUT_ALPHA];
    o.on = true;
    o.d = out_res[k]->GetDesc();
    o.f = srv_format((DXGI_FORMAT)(t.format ? t.format : o.d.Format), false);
    o.srgb = is_srgb(o.f);
    o.uf = uav_format(o.f);
    o.direct = (o.d.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) != 0 && !o.srgb;
    o.ch = k == 0 ? 4 : 1;
    if (o.srgb) o.fmt = 0;
    else if (is_f16(o.uf)) o.fmt = 1;
    else if (k == 0 && is_unorm8x4(o.uf)) o.fmt = 2;
    else if (k == 1 && is_unorm8(o.uf)) o.fmt = 2;
    else if (k == 0 && is_rgb10a2(o.uf)) o.fmt = 3;
    else o.fmt = 0;
  }
  const uint64_t eb = (uint64_t)in->We * in->He;
  auto out_bytes = [&](const OutInfo& o) {
    return o.fmt == 1 ? eb * o.ch * 2 : (o.fmt == 2 ? (o.ch == 4 ? eb * 4 : eb) : (o.fmt == 3 ? eb * 4 : eb * o.ch * 4));
  };
  fs.out_bytes = out_bytes(oi[0]);
  fs.aout_bytes = aout_on ? out_bytes(oi[1]) : 0;

  // ---- layout: fixed items (TH / OH created on their first frame), render-size items (53.9), the work area
  const bool th_new = layer_on && !in->lay.th_on, oh_new = opac_on && !in->lay.oh_on;
  const bool rebuild = in->fresh || f->rebuild_render_items || rw != in->lay.rw || rh != in->lay.rh;
  Layout lay = (th_new || oh_new) ? fixed_layout(*in, in->lay.th_on || th_new, in->lay.oh_on || oh_new) : in->lay;
  if (rebuild || th_new || oh_new) render_layout(&lay, rw, rh);
  WorkPlan P;
  FrameBufs Bf;
  plan_frame(*in, fs, &P, &Bf);
  P.place(lay.state_end, in->alias);
  const uint64_t need = align256(P.end);
  if (need >= 0xFFFFFF00ull) return fail(in, "frame needs more than 4 GiB of arena (32-bit offsets)", res, WG_E_INVALID_PARAMETER);
  const uint64_t blk = P.off(Bf.blk);

  // ---- output targets (typed UAV, or our texture + copy)
  for (int k = 0; k < 2; ++k) {
    OutInfo& o = oi[k];
    OutTarget& t = in->out[k];
    if (!o.on || o.direct) continue;
    if (!t.proxy || t.proxy_fmt != o.uf || t.proxy_w != o.d.Width || t.proxy_h != o.d.Height) {
      defer_release(t.proxy, in->last_serial);
      D3D12_HEAP_PROPERTIES hp = {};
      hp.Type = D3D12_HEAP_TYPE_DEFAULT;
      D3D12_RESOURCE_DESC rd = {};
      rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
      rd.Width = o.d.Width;
      rd.Height = o.d.Height;
      rd.DepthOrArraySize = 1;
      rd.MipLevels = 1;
      rd.Format = o.uf;
      rd.SampleDesc.Count = 1;
      rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
      if (FAILED(rt.dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                                 nullptr, IID_PPV_ARGS(&t.proxy))))
        return fail(in, fmt("output texture of format %u could not be made", (unsigned)o.uf), res, WG_E_D3D_API);
      rt.live_resources++, rt.created_resources++;
      t.proxy_fmt = o.uf;
      t.proxy_w = (uint32_t)o.d.Width;
      t.proxy_h = o.d.Height;
    }
  }

  // ---- slot (upload block, descriptors, timestamps): wait for the frame that used it last
  const uint64_t serial = rt.serial + 1;
  const uint32_t slot = (uint32_t)(serial % kSlots);
  if (!wait_serial(in->slot_serial[slot], 5000))
    return fail(in, "the game's command lists with earlier frames did not finish within 5 s (slot reuse)", res, WG_E_D3D_API);
  harvest(in);

  // ---- arena: a new one when the frame needs more or the fixed items move (the state is copied on the game's list)
  ID3D12Resource* old_arena = nullptr;
  const Layout old_lay = in->lay;
  if (need > in->arena_bytes || th_new || oh_new) {
    ID3D12Resource* keep = in->arena;
    std::string err;
    // a frame larger than the arena: 1/8 more (later frames of that size do not grow it again); a re-layout for
    // TH / OH: exactly what the frame needs
    const uint64_t bytes = need <= in->arena_bytes ? in->arena_bytes : (th_new || oh_new) ? need : align256(need + need / 8);
    if (!new_arena(in, bytes, &err)) {
      in->arena = keep;
      return fail(in, err, res, WG_W_OUT_OF_VIDEO_MEMORY);
    }
    old_arena = keep;
  }

  // ---- upload block: input-stage constants (rr_input_host.h), decode headers, resolve / pool parameters
  const Model& M = rt.model;
  auto B = [&](const char* n) { return (uint32_t)(kModelBase + M.blob(n)); };
  auto Bo = [&](const std::string& n) { const uint64_t v = M.blob(n); return v == UINT64_MAX ? 0u : (uint32_t)(kModelBase + v); };
  uint8_t* up = in->upload_map + (uint64_t)slot * kBlock;
  memset(up, 0, kBlock);
  const uint32_t w1 = (in->Wh + 1) / 2, h1 = (in->Hh + 1) / 2, w2 = (in->Wh + 3) / 4, h2 = (in->Hh + 3) / 4;
  bool camera_usable = false;
  {
    rr_input::Frame fr = {};
    fr.Wr = (int)rw; fr.Hr = (int)rh; fr.Wo = (int)in->Wo; fr.Ho = (int)in->Ho;
    fr.jx = f->jitter[0]; fr.jy = f->jitter[1]; fr.pjx = f->prev_jitter[0]; fr.pjy = f->prev_jitter[1];
    fr.ms_given = true; fr.ms_x = f->mv_scale[0]; fr.ms_y = f->mv_scale[1];
    fr.mv_jittered = in->desc.mv_jittered != 0;
    fr.E = 1.0f; fr.Ep = 1.0f; fr.X3 = 0.0f;            // read on the GPU from the exposure result (K_EXPO)
    fr.reset = reset; fr.par = (int)par; fr.fc = (int)f->counter;
    const Runtime::Words& MW = model_words(rt, kSymNames, rr_input::SYM_COUNT);
    for (int p = 0; p < 2; ++p)
      for (int s = 0; s < 11; ++s) {
        const uint32_t v = MW.sh[p][s];
        fr.SH[p][s][0] = v & 0xffu;
        fr.SH[p][s][1] = (v >> 8) & 0xffu;
      }
    fr.hw_depth = f->depth_is_linear == 0;
    fr.depth_inverted = in->desc.depth_inverted != 0;
    fr.nrm_dec = nrm_decode(img[WG_ROLE_NORMALS].format);
    fr.rgh_packed = rgh_packed;
    fr.specular_from_v = (f->settings & WG_RR_SET_SPECULAR_MOTION_FROM_V) != 0;
    fr.hd_filter = (f->settings & WG_RR_SET_HIT_DISTANCE_FILTER) != 0;
    fr.have_camera = true;
    memcpy(fr.V2C.m, f->view_to_clip, 64);
    memcpy(fr.W2V.m, f->world_to_view, 64);
    memcpy(fr.V2Cp.m, f->prev_view_to_clip, 64);
    memcpy(fr.W2Vp.m, f->prev_world_to_view, 64);
    for (int r = 0; r < (int)kRoles; ++r) fr.attached[r] = ing[r].on;
    fr.b2 = M.b2();
    for (int i = 0; i < rr_input::SYM_COUNT; ++i) fr.sym[i] = rr_input::bits_f(MW.sym[i]);
    uint32_t w[256] = {};
    camera_usable = rr_input::scalar_words(fr, w);
    rr_input::Texture tx[kRoles] = {};
    for (int r = 0; r < (int)kRoles; ++r) {
      if (!ing[r].on) continue;
      tx[r].offset = ing[r].copy ? (uint32_t)P.off(Bf.copy[r]) : 0u;
      tx[r].width = ing[r].w;
      tx[r].height = ing[r].h;
      tx[r].channels = ing[r].ch;
      tx[r].ox = (uint32_t)ing[r].ox;
      tx[r].oy = (uint32_t)ing[r].oy;
    }
    rr_input::texture_words(fr, tx, w + 100);
    for (int r = 0; r < (int)kRoles; ++r) {
      if (!ing[r].on) continue;
      // the reader origin decided above (52.1.2: SV reads at V's origin, B / G1 / G2 at C's), not the table's routing
      // of the entries' own origins (ing[] already holds the routed origin of every role)
      w[100 + 8 * r + 4] = (uint32_t)ing[r].ox;
      w[100 + 8 * r + 5] = (uint32_t)ing[r].oy;
      w[100 + 8 * r + 7] = ing[r].copy ? 0u : (uint32_t)r + 1u;   // RR_IN_SRV: SRV array index + 1
    }
    w[208] = (uint32_t)P.off(Bf.dec_ph1); w[209] = w1; w[210] = h1;
    w[211] = (uint32_t)P.off(Bf.dec_ph2); w[212] = w2; w[213] = h2;
    w[214] = (uint32_t)P.off(Bf.dec_hid);
    // diagnostics only (clean/rr-f-flow probes): WG_RR_DIAG_PMT2=1 = the input stage reads the PM copy of frame t-2
    w[215] = (uint32_t)lay.pm[getenv("WG_RR_DIAG_PMT2") ? 1 - par : par]; w[216] = (uint32_t)lay.pm[1 - par]; w[217] = (uint32_t)lay.pe; w[218] = (uint32_t)lay.pn;
    w[219] = B("in.D");
    w[222] = (uint32_t)P.off(Bf.codes); w[223] = (uint32_t)P.off(Bf.hold);
    w[226] = (uint32_t)lay.ex[par];
    w[227] = B("in.LOGT"); w[228] = B("in.EXPT");
    w[229] = (uint32_t)P.off(Bf.mark);
    memcpy(up + kBlkInput, w, sizeof w);
    memcpy(rt.ksh[slot], w, sizeof w);
    rt.kring_cpu[slot] = (uintptr_t)(up + kBlkInput);
    if (Bf.dx >= 0) {   // WG_RR_DIAG_INX: the same block with the outputs (codes, hold, marks, PM, Pe, Pn) in dx
      const uint64_t ntd = (uint64_t)32 * ((rw + 31) / 32) * (32 * ((rh + 31) / 32)), d0 = P.off(Bf.dx);
      const uint64_t dm = d0 + ntd * 72, dp = dm + ((ntd + 1) * 4 + 15) / 16 * 16, rpd = (uint64_t)rw * rh;
      uint32_t wd[256];
      memcpy(wd, w, sizeof wd);
      wd[222] = (uint32_t)d0; wd[223] = (uint32_t)(d0 + ntd * 64); wd[229] = (uint32_t)dm;
      wd[216] = (uint32_t)dp; wd[217] = (uint32_t)(dp + rpd * 8); wd[218] = (uint32_t)(dp + rpd * 12);
      memcpy(up + kBlkInputDiag, wd, sizeof wd);
    }
    const uint32_t dec[3][4] = {{0u, w1 * h1, (uint32_t)lay.ph1, (uint32_t)P.off(Bf.dec_ph1)},
                                {0u, w2 * h2, (uint32_t)lay.ph2, (uint32_t)P.off(Bf.dec_ph2)},
                                {1u, rw * rh, (uint32_t)lay.hid, (uint32_t)P.off(Bf.dec_hid)}};
    memcpy(up + kBlkDecode, dec, sizeof dec);
  }
  const bool th_created = th_new || oh_new;   // 54b.6: a history created this frame (zero-filled; reset for 54b.3)
  uint32_t rflags = 0;
  bool res_spec2 = false;   // the frame's resolve matches RF_SPEC 2 (resolve.fast2; its own check repeats this)
  bool fuse_store = false;  // the main output's store is done by resolve.fast2 / resolve.fixup (no rt.store for it)
  {
    // resolve parameters (kernels/rr_resolve.hlsl; gpu/resolve.py params) and the PH2 pool's
    auto common = [&](uint32_t* p, uint32_t rwv, uint32_t rhv) {
      const uint32_t s[8] = {in->We, in->He, in->Wh, in->Hh, rwv, rhv, in->Wo, in->Ho};
      memcpy(p, s, sizeof s);
      p[9] = fb(1.0f);
      p[10] = fb(1.0f);
      const uint32_t rwh[5] = {M.scalar("res.RW_B"), M.scalar("res.RW_D"), M.scalar("res.RW_0"), M.scalar("res.RW_C"),
                               M.scalar("res.RW_A")};
      for (int i = 0; i < 5; ++i) p[17 + i] = rwh[i] & 0xffffu;
      const char* ks[14] = {"K1", "out.K2n", "out.K2x2n", "out.K4", "out.K5", "out.K7", "out.K8", "out.K10", "out.K11",
                            "out.K12", "out.K13", "sym.C_HDEC", "res.T_EFL", "res.TBW"};
      for (int i = 0; i < 14; ++i) p[22 + i] = M.scalar(ks[i]);
    };
    auto desc = [&](uint32_t* p, uint32_t at, uint64_t off, uint32_t w, uint32_t h, uint32_t fmt_, uint32_t ch, int32_t ox, int32_t oy) {
      const uint32_t d[8] = {(uint32_t)off, w, h, w, fmt_, ch, (uint32_t)ox, (uint32_t)oy};
      memcpy(p + at, d, sizeof d);
    };
    uint32_t* pp = (uint32_t*)(up + kBlkPool);
    common(pp, 1, 1);
    desc(pp, 160, lay.ph1, w1, h1, 4, 4, 0, 0);
    desc(pp, 168, lay.ph2, w2, h2, 4, 4, 0, 0);
    uint32_t* p = (uint32_t*)(up + kBlkResolve);
    common(p, rw, rh);
    const bool extended = layer_on || alpha_up;
    rflags = (extended ? 1u : 0u) | (layer_on ? 2u : 0u) | (opac_on ? 4u : 0u) | (alpha_up ? 8u : 0u) |
             (aout_on ? 16u : 0u) | ((reset || th_created || in->ah_created) ? 32u : 0u) |
             ((reset || in->ah_created) ? 64u : 0u) | ((f->settings & WG_RR_SET_BOUNDED_STORES) ? 128u : 0u) | 256u;
    p[8] = rflags;
    res_spec2 = (rflags & 15u) == 3u && sl.fmt == 1u && oi[0].fmt == 1u && lay.th_on;
    {   // clean/rr-f-s10-tail: rt.store's texels (We x He, bounded: Wo x Ho inside it) at the output origin
      const WgRrImage& t = img[WG_ROLE_OUTPUT];
      const bool bnd = (f->settings & WG_RR_SET_BOUNDED_STORES) != 0;
      fuse_store = in->fuse_store && in->fast_paths && res_spec2 && oi[0].on && oi[0].fmt == 1u && oi[0].ch == 4u &&
                   !oi[0].srgb && (!bnd || (in->Wo <= in->We && in->Ho <= in->He)) && t.origin_x < 65536u &&
                   t.origin_y < 65536u;
      p[16] = fuse_store ? 1u : 0u;
      p[39] = fuse_store ? ((uint32_t)t.origin_x | ((uint32_t)t.origin_y << 16)) : 0u;
    }
    p[11] = fb(f->jitter[0]);
    p[12] = fb(f->jitter[1]);
    p[13] = f->layer_origin_x;
    p[14] = f->layer_origin_y;
    p[15] = (uint32_t)lay.ex[par];
    p[36] = 0;   // the alpha source copy holds its channel as channel 0
    p[37] = (uint32_t)lay.rfix;   // resolve.fast / resolve.fixup: the fixup bitmap
    p[38] = (uint32_t)lay.rft;    // resolve.fast: the exp2 table
    desc(p, 40, P.off(Bf.q), in->Wh, in->Hh, 1, 3, 0, 0);
    desc(p, 48, P.off(Bf.out), in->We, in->He, oi[0].fmt, 4, 0, 0);
    if (aout_on) desc(p, 56, P.off(Bf.aout), in->We, in->He, oi[1].fmt, 1, 0, 0);
    if (layer_on) desc(p, 64, P.off(Bf.layer), sl.w, sl.h, sl.fmt, 4, sl.ox, sl.oy);
    if (opac_on) desc(p, 72, P.off(Bf.opac), so_.w, so_.h, so_.fmt, 4, so_.ox, so_.oy);
    if (alpha_up) desc(p, 80, P.off(Bf.asrc), sa.w, sa.h, sa.fmt, 1, sa.ox, sa.oy);
    desc(p, 88, lay.pm[1 - par], rw, rh, 1, 4, 0, 0);
    desc(p, 96, P.off(Bf.prec), rw, rh, 1, 4, 0, 0);
    desc(p, 104, P.off(Bf.gates), rw, rh, 2, 4, 0, 0);
    if (lay.th_on) {
      desc(p, 112, lay.th[par], in->We, in->He, 1, 4, 0, 0);
      desc(p, 120, lay.th[1 - par], in->We, in->He, 1, 4, 0, 0);
    }
    if (lay.oh_on) {
      desc(p, 128, lay.oh[par], in->We, in->He, 1, 4, 0, 0);
      desc(p, 136, lay.oh[1 - par], in->We, in->He, 1, 4, 0, 0);
    }
    if (lay.ah_on) {
      desc(p, 144, lay.ah[par], in->We, in->He, 1, 1, 0, 0);
      desc(p, 152, lay.ah[1 - par], in->We, in->He, 1, 1, 0, 0);
    }
  }
  if (res) res->camera_fallback_used = attached(WG_ROLE_SPECULAR_HIT_DIST) && !attached(WG_ROLE_SPECULAR_MOTION) &&
                                       !(f->settings & WG_RR_SET_SPECULAR_MOTION_FROM_V) && !camera_usable;

  // ---- descriptors of this slot: SRV array (direct input-stage reads), {SRV, UAV} pairs (ingest copies, stores)
  D3D12_CPU_DESCRIPTOR_HANDLE hc = in->heap->GetCPUDescriptorHandleForHeapStart();
  D3D12_GPU_DESCRIPTOR_HANDLE hg = in->heap->GetGPUDescriptorHandleForHeapStart();
  const uint64_t sbase = kHeapFixed + (uint64_t)slot * kDescPerSlot;
  auto cpu = [&](uint64_t i) { D3D12_CPU_DESCRIPTOR_HANDLE h = hc; h.ptr += i * rt.inc; return h; };
  auto gpu = [&](uint64_t i) { D3D12_GPU_DESCRIPTOR_HANDLE h = hg; h.ptr += i * rt.inc; return h; };
  auto srv = [&](ID3D12Resource* r, DXGI_FORMAT fm, D3D12_CPU_DESCRIPTOR_HANDLE h) {
    D3D12_SHADER_RESOURCE_VIEW_DESC sd = {};
    sd.Format = fm;
    sd.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    sd.Texture2D.MipLevels = 1;
    rt.dev->CreateShaderResourceView(r, &sd, h);
  };
  if (in->null_tmpl) {
    rt.dev->CopyDescriptorsSimple(kDescPerSlot, cpu(sbase), in->null_tmpl->GetCPUDescriptorHandleForHeapStart(),
                                  D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
  } else {
    for (uint32_t i = 0; i < kSrvArray; ++i) write_null_srv(cpu(sbase + i));
    for (uint32_t i = 0; i < kPairs; ++i) write_null_pair(cpu(sbase + kSrvArray + 2 * i));
  }
  for (int r = 0; r < (int)kRoles; ++r)
    if (ing[r].on) srv((ID3D12Resource*)img[ing[r].role].resource, ing[r].fmt, cpu(sbase + r));
  {   // clean/release-060 crash-cp2077: what this slot's SRV array holds, for the removal log
    rt.dsh_inc = rt.inc;
    rt.dsh_serial[slot] = serial;
    rt.dsh_cpu[slot] = (uintptr_t)cpu(sbase).ptr;
    for (uint32_t r = 0; r < kSrvArray; ++r)
      rt.dsh_res[slot][r] = (r < kRoles && ing[r].on) ? img[ing[r].role].resource : nullptr;
    if (!publish_descriptors((const void*)cpu(sbase).ptr, (size_t)kSrvArray * std::min<uint32_t>(rt.inc, 64), rt.dsh[slot]))
      rt.dsh_serial[slot] = 0;
    static int shown = 0;   // the decode checked against real descriptors (first evaluates)
    if (shown < 2) {
      shown++;
      std::string m = fmt("input SRV base addresses (frame %llu, descriptor size %u%s):", (unsigned long long)serial, rt.inc,
                          rt.dsh_serial[slot] ? "" : ", heap not readable");
      for (uint32_t r = 0; r < kRoles; ++r)
        if (rt.dsh_res[slot][r]) m += fmt(" %u=0x%llx", r, (unsigned long long)desc_va(rt.dsh[slot] + (size_t)r * std::min<uint32_t>(rt.inc, 64)));
      log_line(m);
    }
  }
  uint32_t npair = 0;
  auto pair = [&]() { return sbase + kSrvArray + 2 * (npair++); };
  struct Job { uint64_t pair; uint64_t dst; uint32_t w, h, ch, conv, scale, bias, mode, first, max3; };
  std::vector<Job> pre_jobs, res_jobs;
  for (int r = 0; r < (int)kRoles; ++r) {
    if (!ing[r].on || !ing[r].copy) continue;
    const WgRrImage& t = img[ing[r].role];
    const uint64_t pr = pair();
    srv((ID3D12Resource*)t.resource, ing[r].fmt, cpu(pr));
    pre_jobs.push_back({pr, P.off(Bf.copy[r]), ing[r].w, ing[r].h, ing[r].ch, t.conversion, fb(t.conversion_scale),
                        fb(t.conversion_bias), 0, 0, 0});
  }
  for (const SurfCopy* s : {&sl, &so_, &sa}) {
    if (!s->on) continue;
    const uint64_t pr = pair();
    srv((ID3D12Resource*)img[s->role].resource, s->srv, cpu(pr));
    const uint64_t dst = s == &sl ? P.off(Bf.layer) : s == &so_ ? P.off(Bf.opac) : P.off(Bf.asrc);
    res_jobs.push_back({pr, dst, s->w, s->h, s->ch, 0, 0, 0, s->mode, s->first, s->max3});
  }
  uint64_t store_pair[2] = {};
  for (int k = 0; k < 2; ++k) {
    if (!oi[k].on) continue;
    store_pair[k] = pair();
    D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
    ud.Format = oi[k].uf;
    ud.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    rt.dev->CreateUnorderedAccessView(oi[k].direct ? out_res[k] : in->out[k].proxy, nullptr, &ud, cpu(store_pair[k] + 1));
  }
  if (npair)   // the {SRV, UAV} pairs written above (ingest copies, stores): published like the SRV array
    publish_descriptors((const void*)cpu(sbase + kSrvArray).ptr, (size_t)2 * npair * rt.inc, nullptr);

  // ---- game resource states (declared state; 0xffffffff = not given: no transition) and our transient copies
  std::vector<Bar> bars;
  auto want = [&](ID3D12Resource* r, uint32_t declared, D3D12_RESOURCE_STATES to) {
    if (!r) return;
    for (ID3D12Resource* t : rt.transient)
      if (t == r) return;   // our lifetime-0 copies stay in NON_PIXEL_SHADER_RESOURCE
    if (declared == 0xffffffffu) return;
    if ((declared & (uint32_t)to) == (uint32_t)to) return;
    for (const Bar& b : bars)
      if (b.r == r) return;
    bars.push_back({r, (D3D12_RESOURCE_STATES)declared, to});
  };
  for (int r = 0; r < (int)kRoles; ++r)
    if (ing[r].on) want((ID3D12Resource*)img[ing[r].role].resource, img[ing[r].role].state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
  for (const SurfCopy* s : {&sl, &so_, &sa})
    if (s->on) want((ID3D12Resource*)img[s->role].resource, img[s->role].state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
  for (int k = 0; k < 2; ++k)
    if (oi[k].on)
      want(out_res[k], img[k == 0 ? WG_ROLE_OUTPUT : WG_ROLE_OUTPUT_ALPHA].state,
           oi[k].direct ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS : D3D12_RESOURCE_STATE_COPY_DEST);

  // ================================================================ record into the game's list
  ID3D12GraphicsCommandList* l = list;
  ID3D12Resource* A = in->arena;
  const UINT q0 = slot * kTq;
  l->EndQuery(in->qh, D3D12_QUERY_TYPE_TIMESTAMP, q0);
  // clean/rr-f-s9-host probe: WG_RR_TSLATE=1 at creation records the stage-split timestamps at the end of the frame
  // (frame time unchanged, stage split lost): measures what the in-frame timestamps cost
  std::vector<UINT> stamps_late;
  auto stamp = [&](UINT q) {
    if (in->ts_late) stamps_late.push_back(q);
    else l->EndQuery(in->qh, D3D12_QUERY_TYPE_TIMESTAMP, q);
  };
  // clean/rr-f-a14-frame E2: barrier flavour for this recording
  g_bark = in->bark;
  g_bark_list = l;
  g_l7 = nullptr;
  if (in->bark >= 2 && FAILED(l->QueryInterface(IID_PPV_ARGS(&g_l7)))) g_l7 = nullptr;
  struct BarkRef { ~BarkRef() { if (g_l7) g_l7->Release(); g_l7 = nullptr; g_bark = 0; g_bark_list = nullptr; } } bark_ref;
  if (rt.crumbs) g_crumb_base = l, g_crumb_list = list2, g_crumb_tag = (uint32_t)(serial & 0xfff) << 20, g_crumb_n = 0;
  struct CrumbRef { ~CrumbRef() { g_crumb_base = nullptr, g_crumb_list = nullptr; } } crumb_ref;
  {   // the instance's buffer addresses, for the removal log
    std::string v = fmt("instance %u arena [0x%llx, 0x%llx) upload [0x%llx, +%llu)", in->id,
                        (unsigned long long)A->GetGPUVirtualAddress(), (unsigned long long)(A->GetGPUVirtualAddress() + in->arena_bytes),
                        (unsigned long long)in->upload->GetGPUVirtualAddress(), (unsigned long long)in->upload->GetDesc().Width);
    std::lock_guard<std::mutex> lock(rt.va_mu);
    if (rt.va_note != v) rt.va_note = v, log_line("buffer addresses: " + v);
  }
  // a14 E2 (WG_RR_BARBATCH=1): the game-texture transitions as one ResourceBarrier call per direction
  auto bar_all = [&](bool back) {
    std::vector<D3D12_RESOURCE_BARRIER> v;
    for (const Bar& b : bars) {
      D3D12_RESOURCE_BARRIER x = {};
      x.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
      x.Transition.pResource = b.r;
      x.Transition.Subresource = wgi::barrier_subresource(b.r);  // mip 0 alone of a game texture with mips (transition_game)
      x.Transition.StateBefore = back ? b.to : b.from;
      x.Transition.StateAfter = back ? b.from : b.to;
      if (x.Transition.StateBefore != x.Transition.StateAfter) v.push_back(x);
    }
    if (!v.empty()) l->ResourceBarrier((UINT)v.size(), v.data());
  };
  if (in->barbatch) bar_all(false);
  else for (const Bar& b : bars) transition_game(l, b.r, b.from, b.to);
  // clean/rr-f-a14-frame E25: the arena's state from the last frame (COMMON, or UNORDERED_ACCESS with WG_RR_ARENA)
  const D3D12_RESOURCE_STATES ast = in->arena_uav ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS : D3D12_RESOURCE_STATE_COMMON;
  const bool arena_probe = in->arena_mode == 9 && in->arena_uav && !old_arena && !in->fresh;
  // E25: probe2 (timing only: the copy into the arena while it stays UNORDERED_ACCESS, then one UAV barrier); blk / blkgu
  // (rt.blkcopy from the ring through root SRV 6, then one UAV barrier)
  const bool arena_keep = (in->arena_mode == 8 || in->arena_mode == 2 || in->arena_mode == 3) && in->arena_uav && !old_arena && !in->fresh;
  if (old_arena) {   // new arena: the model image and the fixed items keep their offsets; the render-size items move
    transition(l, old_arena, ast, D3D12_RESOURCE_STATE_COPY_SOURCE);
    transition(l, A, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    l->CopyBufferRegion(A, 0, old_arena, 0, old_lay.fixed_end);
    if (!rebuild) l->CopyBufferRegion(A, lay.fixed_end, old_arena, old_lay.fixed_end, old_lay.state_end - old_lay.fixed_end);
    transition(l, old_arena, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
  } else if (!arena_probe && !arena_keep) {
    transition(l, A, ast, D3D12_RESOURCE_STATE_COPY_DEST);
  }
  if (arena_keep && in->arena_mode == 8) {
    l->CopyBufferRegion(A, blk, in->upload, (uint64_t)slot * kBlock, kBlock);
  } else if (!arena_probe && !arena_keep) {
    l->CopyBufferRegion(A, blk, in->upload, (uint64_t)slot * kBlock, kBlock);
    transition(l, A, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  }
  bind(l, in->heap, A);
  l->SetComputeRootDescriptorTable(5, gpu(sbase));
  l->SetComputeRootShaderResourceView(6, in->upload->GetGPUVirtualAddress());
  // clean/rr-mp1-input I1: the input features / replay kernels read their 1 KB constant block through root SRV 6
  // (rr_input.hlsl RR_IN_KCB 1, scalar loads): rebound to this frame's block in the upload ring around those dispatches
  // (the arena block they also get as pK is a byte copy of it), back to the ring's start afterwards (rt.blkcopy's view).
  // The slot is not rewritten before this frame completed (wait_serial(slot_serial[slot]) above).
  const D3D12_GPU_VIRTUAL_ADDRESS ring_slot = in->upload->GetGPUVirtualAddress() + (uint64_t)slot * kBlock;
  if (arena_keep && in->arena_mode != 8) dispatch(l, K_BLKCOPY, 4, {(uint32_t)blk, slot * kBlock});
  if (arena_keep) uav_barrier(l, A);
  auto fill = [&](uint64_t off, uint64_t bytes, uint32_t value) {
    const uint32_t n = (uint32_t)(bytes / 4);
    if (n) dispatch(l, K_FILL, (n + 63) / 64, {(uint32_t)off, n, value});
  };
  if (in->fresh) {     // creation (53.8): every image zero-filled (EX too; its first read is on a reset frame)
    fill(lay.ch[0], lay.fixed_end - lay.ch[0], 0);
  } else {
    if (rebuild) fill(lay.ch[0], lay.ph1 - lay.ch[0], 0);   // render-size change (53.9): CH pair zero-filled
    if (th_new) fill(lay.th[0], 2 * (lay.th[1] - lay.th[0]), 0);   // TH / OH created zero-filled (54b.6)
    if (oh_new) fill(lay.oh[0], 2 * (lay.oh[1] - lay.oh[0]), 0);
  }
  if (rebuild) fill(lay.hid, lay.state_end - lay.hid, 0);   // HID, PM pair, Pe, Pn, output scratch zero-filled
  if (in->fresh) {     // after the creation fill: the exp2 table of the fast resolve, the PH2 pool's exact tables
    uav_barrier(l, A);
    dispatch(l, K_RES_INIT, 1024 / 256, {(uint32_t)lay.rft, 0u, 0u});
    dispatch(l, K_POOL_TABLES, 1024 / 256, {(uint32_t)(blk + kBlkPool), (uint32_t)lay.ptab, 0u, 0u});
  }
  auto ingest = [&](const Job& j) {
    l->SetComputeRootDescriptorTable(4, gpu(j.pair));
    dispatch(l, K_INGEST, (j.w * j.h + 63) / 64, {(uint32_t)j.dst, j.w, j.h, j.ch, j.conv, j.scale, j.bias, j.mode,
                                                  j.first, j.max3 == 3 ? 1023u : 255u, j.max3 ? j.max3 : 255u});
  };
  for (const Job& j : pre_jobs) ingest(j);
  // diagnostics only (clean/rr-f-flow probes): WG_RR_DIAG_ZERO = comma list of hid, ph1, ph2, pm, ch: that temporal
  // image is zero-filled before its reader on every frame from frame WG_RR_DIAG_ZERO_FROM (default 2) on
  const char* dz = getenv("WG_RR_DIAG_ZERO");
  const bool dz_on = dz && in->frames >= (uint64_t)(getenv("WG_RR_DIAG_ZERO_FROM") ? atoi(getenv("WG_RR_DIAG_ZERO_FROM")) : 2);
  // clean/rr-f-s9-host: on a frame without zero-fills, the PH2 pool (PH1 -> PH2, state images only) runs beside the
  // ingest copies and the exposure partials (colour -> partials, when the colour is read directly), exposure.finish
  // (partials -> EX[par]) beside the decode (PH1 / PH2 / HID -> dec_*; partials are kept live through the decode):
  // two barriers instead of five. Every dispatch reads only what an earlier barrier completed.
  const bool hs = in->host_sched && !in->fresh && !rebuild && !th_new && !oh_new && !dz_on;
  const bool hs_early = hs && !ing[rr_input::R_C].copy;   // exposure.partial needs no ingest copy
  // clean/rr-f-s11-tail: this frame records the next frame's history prep beside its resolve (tail_rec); it skips its own
  // front pool / decode when the previous recorded frame did so into the same arena, offsets and sizes (use_tail)
  const bool tail_rec = in->tail_prep && in->fast_paths && !in->diag_inx && !dz;
  const uint64_t tkey[10] = {(uint64_t)(uintptr_t)A, P.off(Bf.dec_ph1), P.off(Bf.dec_ph2), P.off(Bf.dec_hid),
                             (uint64_t)w1 | ((uint64_t)h1 << 32), (uint64_t)w2 | ((uint64_t)h2 << 32),
                             (uint64_t)rw | ((uint64_t)rh << 32), lay.ph1, lay.ph2, lay.hid ^ (lay.ptab << 32)};
  const bool use_tail = tail_rec && hs && in->tail_ready && !old_arena && memcmp(tkey, in->tail_key, sizeof tkey) == 0;
  in->tail_ready = false;
  if (!hs) uav_barrier(l, A);
  // PH2 quarter pool (54b.5), non-reset frames only (rr_resolve_pool: PH1 of the previous frame -> PH2)
  if (!reset && !use_tail) {
    if (in->fast_paths)   // through the exact tables (pool.main; the same bits as resolve.pool)
      dispatch(l, K_POOL_MAIN, (w2 * h2 + kPoolLanes - 1) / kPoolLanes, {(uint32_t)(blk + kBlkPool), (uint32_t)lay.ptab, 0u, 0u});
    else
      dispatch(l, K_RES_POOL, (w2 * h2 + 63) / 64, {(uint32_t)(blk + kBlkPool), 0u, 0u});
    if (!hs) uav_barrier(l, A);
  }
  auto decode1 = [&](uint32_t k) {   // 0: PH1, 1: PH2, 2: HID -> dec_*
    const uint32_t n = k == 0 ? w1 * h1 : (k == 1 ? w2 * h2 : rw * rh);
    dispatch(l, K_IN_DEC, (n + 255) / 256, {(uint32_t)(blk + kBlkDecode + 16 * k)});
  };
  auto decode = [&]() {
    if (!use_tail)   // else decoded beside the previous frame's resolve
      for (uint32_t k = 0; k < 3; ++k) decode1(k);
    dispatch(l, K_IN_CLEAR, 1, {(uint32_t)(blk + kBlkInput)});   // the replay list's count = 0
  };
  // exposure (58): reads the previous frame's result (EX[1-par] of 53.6), writes this frame's (EX[par])
  {
    const double rcp = 1.0 / ((double)rw * rh);
    uint32_t lohi[2];
    memcpy(lohi, &rcp, 8);
    const uint32_t eblocks = (uint32_t)(((uint64_t)rw * rh + 255) / 256);
    const Ing& c = ing[rr_input::R_C];
    std::vector<uint32_t> a = {c.copy ? (uint32_t)P.off(Bf.copy[rr_input::R_C]) : 0u, (uint32_t)P.off(Bf.partials),
                               (uint32_t)lay.ex[par], c.copy ? c.w : 0u, c.copy ? c.ch : 0u, (uint32_t)c.ox, (uint32_t)c.oy,
                               rw, rh, eblocks, lohi[0], lohi[1], (uint32_t)lay.ex[1 - par], 2u | (reset ? 1u : 0u)};
    for (const char* n : {"X_LWrb", "X_LWg", "X_LMAX", "X_KEY", "X_MFL", "X_EFL", "X_CL", "X_CLn", "X_NRM"})
      a.push_back(M.scalar(std::string("exp.") + n));
    if (hs_early) {        // [ingest, pool, exposure.partial] | [exposure.finish, decode] |
      dispatch(l, K_EXP_PART, eblocks, a);
      uav_barrier(l, A);
      dispatch(l, K_EXP_FIN, 1, a);
      decode();
      uav_barrier(l, A);
    } else if (hs) {       // [ingest, pool] | [exposure.partial, decode] | [exposure.finish] |
      uav_barrier(l, A);
      dispatch(l, K_EXP_PART, eblocks, a);
      decode();
      uav_barrier(l, A);
      dispatch(l, K_EXP_FIN, 1, a);
      uav_barrier(l, A);
    } else {
      dispatch(l, K_EXP_PART, eblocks, a);
      uav_barrier(l, A);
      dispatch(l, K_EXP_FIN, 1, a);
      uav_barrier(l, A);
    }
  }
  stamp(q0 + 1);
  // input stage (52): decode PH1 / PH2 / HID, fast features (marks), exact replay, front-end MLP
  auto dzh = [&](const char* k) { return dz_on && strstr(dz, k) != nullptr; };
  if (dzh("hid")) fill(lay.hid, (uint64_t)rw * rh * 4, 0);
  if (dzh("ph1")) fill(lay.ph1, (uint64_t)w1 * h1 * 8, 0);
  if (dzh("ph2")) fill(lay.ph2, (uint64_t)w2 * h2 * 8, 0);
  if (dzh("pm")) fill(lay.pm[getenv("WG_RR_DIAG_PMT2") ? 1 - par : par], (uint64_t)rw * rh * 8, 0);
  if (dzh("ch")) fill(lay.ch[par], (uint64_t)in->Wh * in->Hh * 8, 0);
  if (dz_on) uav_barrier(l, A);
  if (!hs) {
    decode();
    uav_barrier(l, A);
  }
  stamp(q0 + kTs);
  const uint32_t fgroups = (uint32_t)((nt + kFeatBlock - 1) / kFeatBlock);
  if (Bf.dx >= 0) {   // WG_RR_DIAG_INX: the exact features + replay into dx first (they write nothing the main pass reads)
    dispatch(l, K_IN_CLEAR, 1, {(uint32_t)(blk + kBlkInputDiag)});
    uav_barrier(l, A);
    l->SetComputeRootShaderResourceView(6, ring_slot + kBlkInputDiag);   // I1: the diagnostic block's view
    dispatch(l, K_IN_FEAT, fgroups, {(uint32_t)(blk + kBlkInputDiag)});
    uav_barrier(l, A);
    dispatch(l, K_IN_REPLAY, std::min(fgroups, kReplayGroups), {(uint32_t)(blk + kBlkInputDiag)});
    uav_barrier(l, A);
    l->SetComputeRootShaderResourceView(6, in->upload->GetGPUVirtualAddress());
  }
  auto OO = [&](int i) { return (uint32_t)P.off(i); };
  // the output stage's argument block (54) without its two tail words (fast-path flags + scratch / packed head)
  auto out_args = [&]() {
    std::vector<uint32_t> a = {OO(Bf.cnet), OO(Bf.gates), OO(Bf.prec), OO(Bf.candw), (uint32_t)lay.pm[1 - par],
                               (uint32_t)lay.pe, (uint32_t)lay.pn, (uint32_t)lay.ch[par], B("out.TR"), OO(Bf.linear),
                               (uint32_t)lay.ch[1 - par], OO(Bf.q), (uint32_t)lay.ph1, OO(Bf.trace), rw, rh, Wt,
                               in->Wh, in->Hh, fb(f->jitter[0]), fb(f->jitter[1]), (uint32_t)lay.ex[par], 0u, 0u,
                               reset ? 1u : 0u, M.scalar("K1")};
    for (const char* k : {"K2n", "K2x2n", "K3", "K4", "K5", "K7", "K8", "K10", "K11", "K12", "K13"})
      a.push_back(M.scalar(std::string("out.") + k));
    return a;
  };
  // clean/rr-f-r6-tail: the fast head writes the output stage's render records (rr_d0r.hlsl RR_OUT_PREP); output.prep
  // then runs as output.prepu (uniform words, fallback-count clear, TR checks) before the head
  const bool oprep = in->head_fast && in->fast_paths;
  // clean/rr-f-a14-frame E21 / E23: output.prepu (uniform words, fallback-count clear, TR checks: output scratch words
  // only) recorded where WG_RR_PREPU_AT puts it (default: beside the head)
  auto rec_prepu = [&]() {
    if (!oprep) return;
    std::vector<uint32_t> oa = out_args();
    oa.push_back(kOutFastFlags);
    oa.push_back((uint32_t)lay.oscr);
    dispatch(l, (in->vbits >> VG_OUT) & 1 ? K_OUT_PREPU_F : K_OUT_PREPU, (513 * 3) / kOutPrepBlock + 1, oa);
  };
  const uint32_t wc[5] = {M.scalar("wc0"), M.scalar("wc1"), M.scalar("wc2"), M.scalar("wc3"), M.scalar("wc4")};
  const uint32_t x0 = (uint32_t)P.off(Bf.x0);
  // clean/rr-f-s9-kernels: stagee.IN0 built with the folded front-end MLP (kRrInFold; variant E, the bits of input.mlp.e)
  // reads the codes itself; the separate MLP dispatch then only runs for the other body steps / the exact MLP
  const bool infold = in->infold;
  // a14 E29 (fast-only, WG_RR_NETALT=1 in rr_mode fast): every other frame skips the input features, the body and the
  // head; the output stage and the resolve read the previous frame's head results (kept through the whole frame)
  const bool nsk = in->netalt && in->net_prev_ok && !reset && !rebuild && !old_arena && !in->fresh && !th_new && !oh_new;
  in->net_prev_ok = in->netalt && !nsk;
  if (!nsk) {
  if (in->dflow) fill(P.off(Bf.dfcnt), (uint64_t)(15 * kDfRows + 64) * 4, 0);   // clean/rr-f-s12-dflow: row counters
  l->SetComputeRootShaderResourceView(6, ring_slot + kBlkInput);   // I1: the input block's view (RR_IN_KCB)
  dispatch(l, input_features_kernel(in->input_lite ? 1 : (in->input_split ? 2 : 0)), fgroups, {(uint32_t)(blk + kBlkInput)});
  if (!(in->dprobe2 & 1)) uav_barrier(l, A);
  stamp(q0 + kTs + 1);
  // clean/rr-mp2-input I4f: input lite (input.features.lite, RR_IN_LITE 1) appends no token to the replay list, so its
  // replay would only read the zero count input.clear wrote and return: not dispatched, and without the barrier behind it
  // (nothing else was recorded since the features' barrier). Exact / split keep the replay and both barriers.
  const bool replay = !in->input_lite;
  if (replay)
    dispatch(l, K_IN_REPLAY, std::min(fgroups, kReplayGroups), {(uint32_t)(blk + kBlkInput)});   // the listed tokens
  l->SetComputeRootShaderResourceView(6, in->upload->GetGPUVirtualAddress());
  bool after_feat = replay;   // something recorded after the features' barrier
  if (in->prepu_at == 100) rec_prepu(), after_feat = true;   // E21: beside the replay's latency tail
  if (in->host_sched && in->ringest == 15)
    for (const Job& j : res_jobs) ingest(j), after_feat = true;
  if (!(in->dprobe2 & 2) && (after_feat || (in->dprobe2 & 1))) uav_barrier(l, A);
  stamp(q0 + kTs + 2);
  if (!infold) {
    dispatch(l, in->input_mlp_e ? K_IN_MLP_E : K_IN_MLP, (uint32_t)(nt / 64), {(uint32_t)P.off(Bf.codes), B("in.W1"), B("in.W2"), B("in.B1"), B("in.B2"),
                                                 x0, (uint32_t)nt, wc[0], wc[1], wc[2], wc[3], wc[4], 0u});
    uav_barrier(l, A);
  }
  }   // !nsk
  if (res) res->passes_started = 1, res->input_jitter_consumed = 1;   // pj consumed (53.5)
  stamp(q0 + 2);
  // network body (51): IN0, E1 .. E5 (E1 reads IN0's y2 directly, or through the child regroup), D4 .. D1 fusion /
  // recurrent, D0-F
  const Runtime::Words& MW = model_words(rt, kSymNames, rr_input::SYM_COUNT);
  auto shift_row = [&](int row) { const uint32_t v = MW.sh[par][row]; return std::make_pair(v & 0xffu, (v >> 8) & 0xffu); };
  auto sidx = [&](const char* n) { for (int s = 0; s < 15; ++s) if (strcmp(kStages[s].name, n) == 0) return s; return -1; };
  auto O = [&](int i) { return (uint32_t)P.off(i); };
  const uint32_t sig = B("sig");
  // clean/rr-f-a14-frame (gated E14 / E15): body stages not dispatched this frame (WG_RR_SKIP / SHALLOW every frame;
  // WG_RR_DEEPALT every other frame after a computed one); barriers kept around a skipped run (no counters there)
  const bool alt_skip = in->altmask && in->deep_prev_ok && !reset && !rebuild && !old_arena && !in->fresh && !th_new && !oh_new;
  const uint32_t fskip = in->skipmask | (alt_skip ? in->altmask : 0u);
  uint32_t fdrop = in->dfdrop;
  for (int t = 0; t < 15; ++t)
    if (((fskip >> t) & 1) || (t + 1 < 15 && ((fskip >> (t + 1)) & 1))) fdrop &= ~(1u << t);
  in->deep_prev_ok = in->altmask && !alt_skip;
  const bool lag_ok = in->deeplag && in->lag_prev_ok && !reset && !rebuild && !old_arena && !in->fresh && !th_new && !oh_new;
  const uint32_t lpar = (uint32_t)(in->frames & 1), lsrc = lag_ok ? 1 - lpar : lpar;
  in->lag_prev_ok = in->deeplag != 0;
  // clean/rr-f-s12-dflow: every stage's window grid (shift, nwx) for the producer slots
  uint32_t dsy[15] = {}, dnwx[15] = {};
  if (in->dflow)
    for (int s = 0; s < 15; ++s) {
      const StageDef& d = kStages[s];
      const int row = d.kind == 'i' ? 0 : (d.kind == 'e' ? d.level : 6 + (4 - d.level));
      auto sh = shift_row(row);
      uint32_t sx = sh.first, sy = sh.second;
      if (d.kind == 'r') sx = (sx + 4) % 8, sy = (sy + 4) % 8;
      dsy[s] = sy;
      dnwx[s] = (Bf.Nx[s] + sx + 7) / 8;
    }
  // producer p's tensors need a wait in stage s when every barrier between them (after p .. after s - 1) is dropped
  auto dslot = [&](int p, int s, uint32_t kind) -> uint32_t {
    for (int t = p; t < s; ++t)
      if (!((fdrop >> t) & 1)) return 0xffu;
    return (uint32_t)p | (kind << 8) | (dsy[p] << 12) | (dnwx[p] << 16);
  };
  // the slots of every staged.* stage, and the stages that need the dataflow form at all: a stage with an active slot
  // (waits, coherent reads) or the producer of one (counts itself done, coherent stores). The others keep stagee.*
  // (WG_RR_DFLOW_ALL=1: every stage of kRrDflowMask on staged.*, A/B)
  uint32_t dq[15][3];
  bool dneed[15] = {};
  if (in->dflow)
    for (int s = 0; s < 15; ++s) {
      dq[s][0] = dq[s][1] = dq[s][2] = 0xffu;
      if (!((((in->vbits >> VG_BODY) & 1) ? kRrDflowMaskX : kRrDflowMask) >> s & 1)) continue;
      const StageDef& d = kStages[s];
      if (d.kind == 'e') {
        dq[s][0] = dslot(s - 1, s, 1);
      } else if (d.kind == 'f') {
        const int enc = d.level == 0 ? 0 : sidx(fmt("E%d", d.level).c_str());
        const int coarse = d.level == 4 ? sidx("E5") : sidx(fmt("D%d-R", d.level + 1).c_str());
        dq[s][0] = (lag_ok && d.level == 3) ? 0xffu : dslot(coarse, s, 2);
        dq[s][1] = dslot(enc, s, 0);
      } else if (d.kind == 'r') {
        const int enc = sidx(fmt("E%d", d.level).c_str());
        const int coarse = d.level == 4 ? sidx("E5") : sidx(fmt("D%d-R", d.level + 1).c_str());
        dq[s][0] = dslot(s - 1, s, 0);
        dq[s][1] = dslot(enc, s, 0);
        dq[s][2] = (lag_ok && d.level == 3) ? 0xffu : dslot(coarse, s, 3);
      }
      for (int k = 0; k < 3; ++k)
        if (dq[s][k] != 0xffu) dneed[s] = true, dneed[dq[s][k] & 0xffu] = true;
      if (in->dfall) dneed[s] = true;
    }
  for (int s = 0; s < 15; ++s) {
    if (((fskip >> s) & 1) || nsk) continue;
    const StageDef& d = kStages[s];
    // default: step E. WG_RR_BODY=Z: exact step Z (bit-identical to step X) unless the stage's host check failed
    // (gpu/stage.py dispatches_for; step X there): the Z dispatch, then the step-X fixup of the windows it marked
    const Runtime::StageWords& sw = MW.st[s];
    const bool zs = !in->body_x && !in->body_e && sw.zok != 0;
    uint32_t pX = 0, pCin = 0, pHold = 0, pHc = 0;
    int row = 0;
    if (d.kind == 'i') {
      pX = infold ? (uint32_t)P.off(Bf.codes) : x0;
      pHold = O(Bf.hold);
      if (infold) pCin = B("in.W1"), pHc = B("in.B2");
      row = 0;
    } else if (d.kind == 'e') {
      const int parent = s - 1;
      if (d.level == 1 && kRrE1Direct && !zs) {
        pX = O(Bf.y2[0]);   // stage.E1 (RR_ADAPT 2) reads its four children from IN0's y2
      } else if (d.level == 1) {
        dispatch(l, K_REGROUP, (uint32_t)((nt * 2 + 255) / 256), {O(Bf.y2[0]), O(Bf.regroup), Wt, Ht});
        uav_barrier(l, A);
        pX = O(Bf.regroup);
      } else {
        pX = O(Bf.merged[parent]);
      }
      pHold = O(Bf.hmerged[parent]);
      row = d.level;
    } else if (d.kind == 'f') {
      const int enc = d.level == 0 ? 0 : sidx(fmt("E%d", d.level).c_str());
      const int coarse = d.level == 4 ? sidx("E5") : sidx(fmt("D%d-R", d.level + 1).c_str());
      pX = O(Bf.y2[enc]);
      pCin = O(Bf.y2[coarse]);
      if (in->shallow == 6 && d.level == 4) pCin = O(Bf.merged[4]);   // E14 rung 0
      if (in->shallow == 5 && d.level == 3) pCin = O(Bf.y2[4]);       // rung 1
      if (in->shallow == 4 && d.level == 3) pCin = O(Bf.merged[3]);   // rung 2
      if (in->deeplag && d.level == 3) pCin = O(Bf.lagy[lsrc]);       // E16
      row = 6 + (4 - d.level);
    } else {
      const int fusion = s - 1, enc = sidx(fmt("E%d", d.level).c_str());
      const int coarse = d.level == 4 ? sidx("E5") : sidx(fmt("D%d-R", d.level + 1).c_str());
      pX = O(Bf.y2[fusion]);
      pHold = O(Bf.hnew[enc]);
      pHc = O(Bf.hnew[coarse]);
      if (in->shallow == 6 && d.level == 4) pHc = O(Bf.hmerged[4]);
      if (in->shallow == 5 && d.level == 3) pHc = O(Bf.hnew[4]);
      if (in->shallow == 4 && d.level == 3) pHc = O(Bf.hmerged[3]);
      if (in->deeplag && d.level == 3) pHc = O(Bf.lagh[lsrc]);
      row = 6 + (4 - d.level);
    }
    auto sh = shift_row(row);
    uint32_t sx = sh.first, sy = sh.second;
    if (d.kind == 'r') sx = (sx + 4) % 8, sy = (sy + 4) % 8;
    const std::string n = d.name;
    const uint32_t Nx = Bf.Nx[s], Ny = Bf.Ny[s];
    const uint32_t nwx = (Nx + sx + 7) / 8, nwy = (Ny + sy + 7) / 8;
    std::vector<uint32_t> a = {pX, pCin, pHold, pHc, sw.bo[0], sw.bo[1], sw.bo[2], sw.bo[3],
                               sw.bo[4], sw.bo[5], sw.bo[6], sw.bo[7], sw.bo[8],
                               sw.bo[9], sw.bo[10], sw.bo[11], sw.bo[12], sw.bo[13],
                               sw.bo[14], sig, 0u, (in->deeplag && s == 7) ? O(Bf.lagy[lpar]) : O(Bf.y2[s]), O(Bf.merged[s]),
                               (in->deeplag && s == 7) ? O(Bf.lagh[lpar]) : O(Bf.hnew[s]),
                               O(Bf.hmerged[s]), Nx, Ny, sx, sy, nwx, sw.ra_m,
                               sw.f_g, wc[0], wc[1], wc[2], wc[3], wc[4]};
    if (d.kind == 'i' && infold) a[15] = B("in.W2"), a[16] = B("in.B1");   // rr_block_c32.hlsl RR_INFOLD: pWup, pBup (s9 kernels)
    if (zs) {
      a.insert(a.end(), {(uint32_t)lay.zmk[s], sw.zthr, sw.zt0, sw.zt1});
      dispatch(l, (Kid)(K_ZSTAGE0 + s), nwx * nwy, a);
      uav_barrier(l, A);
      dispatch(l, (Kid)(K_ZFIX0 + s), nwx * nwy, a);
    } else {
      if (in->dflow && dneed[s]) {
        // clean/rr-f-s12-dflow: staged.<s>: counter array, own index, producer slots (rr_dflow.hlsli)
        uint32_t q[3] = {dq[s][0], dq[s][1], dq[s][2]};
        if (s == in->dfneg && q[0] != 0xffu) q[0] |= 1u << 11;
        a.insert(a.end(), {(uint32_t)P.off(Bf.dfcnt), (uint32_t)s, q[0], q[1], q[2]});
        dispatch(l, (Kid)(((in->vbits >> VG_BODY) & 1 ? K_DXSTAGE0 : K_DSTAGE0) + s), nwx * nwy, a);
      } else {
        dispatch(l, in->body_e ? (Kid)(((in->vbits >> VG_BODY) & 1 ? K_ESTAGEX0 : K_ESTAGE0) + s) : d.k, nwx * nwy, a);
      }
    }
    // clean/rr-f-s9-host: the resolve's game-input ingest (game textures -> layer / opacity / alpha copies, read
    // first by the resolve) fills the GPU beside a low-occupancy body level instead of its own barrier slot
    if (in->host_sched && s == in->ringest)
      for (const Job& j : res_jobs) ingest(j);
    if (s == in->prepu_at) rec_prepu();
    if (in->touch) {   // a14 E24: read-only cache warm-up (rt.touch), no barrier of its own
      auto touch = [&](uint64_t off, uint64_t bytes) {
        const uint32_t lines = (uint32_t)(bytes / 128);
        if (lines) dispatch(l, K_TOUCH, (lines + 1023) / 1024, {(uint32_t)off, lines, 0x5a5a5a5au, 0u});
      };
      if ((in->touch & 1) && s == 6) {
        touch(lay.ch[1 - par], (uint64_t)in->Wh * in->Hh * 8);
        if (lay.th_on) touch(lay.th[0], 2 * (lay.th[1] - lay.th[0]));
      }
      if ((in->touch & 2) && s == 11) touch(O(Bf.y2[0]), std::min<uint64_t>((uint64_t)Bf.Nx[0] * Bf.Ny[0] * 32, 24ull << 20));
      if (((in->touch & 4) && s == 2) || ((in->touch & 8) && s == 0)) touch(kModelBase, rt.model.image.size());
      if (((in->touch & 0x10) && s == 2) || ((in->touch & 0x20) && s == 0)) {   // the deep stages' weight span only (E3 .. D2-R)
        uint32_t lo = 0xffffffffu, hi = 0;
        for (int t = 3; t <= 11; ++t)
          for (int k = 0; k < 15; ++k)
            if (MW.st[t].bo[k] >= kModelBase && MW.st[t].bo[k] < kModelBase + rt.model.image.size())
              lo = std::min(lo, MW.st[t].bo[k]), hi = std::max(hi, MW.st[t].bo[k]);
        if (hi > lo) touch(lo, std::min<uint64_t>(hi - lo + 65536, kModelBase + rt.model.image.size() - lo));
      }
    }
    if (!((in->dprobe >> s) & 1) && !(in->dflow && ((fdrop >> s) & 1))) uav_barrier(l, A);
  }
  stamp(q0 + 3);
  // D0-R final head (55): Cnet, HID (into the state image), GATES, PREC, CANDW
  {
    auto sh = shift_row(10);
    const uint32_t sx = (sh.first + 4) % 8, sy = (sh.second + 4) % 8;
    const uint32_t nwx = (Wt + sx + 7) / 8, nwy = (Ht + sy + 7) / 8;
    std::vector<uint32_t> a = {O(Bf.y2[14]), B("head.wf"), B("head.bias"), B("head.tb"), O(Bf.hnew[0]),
                               O(Bf.hnew[13]), B("head.tab"), B("head.rho"), O(Bf.cnet), (uint32_t)lay.hid,
                               O(Bf.gates), O(Bf.prec), O(Bf.candw),
                               in->head_debug ? O(Bf.dbg_y2h) : 0u, in->head_debug ? O(Bf.dbg_img) : 0u, 0u, 0u,
                               Wt, Ht, rw, rh, sx | (in->head18 ? 0x100u : 0u),
                               sy, nwx};
    for (const char* k : {"k_as", "k_acl", "k_acln", "k_p1", "k_p0", "k_actcl", "k_actcln", "k_c1", "k_c2", "k_c3",
                          "k_ra", "k_m", "k_g", "k_s1"})
      a.push_back(M.scalar(std::string("head.") + k));
    a.push_back(O(Bf.hflag));
    a.push_back(O(Bf.hwpl));
    a.push_back(oprep ? (uint32_t)lay.oscr : 0u);   // head.fast (RR_OUT_PREP): output scratch, history size, constants
    a.push_back(in->Wh);
    a.push_back(in->Hh);
    for (const char* k : {"K2n", "K2x2n", "K4", "K5", "K7"}) a.push_back(M.scalar(std::string("out.") + k));
    if (in->prepu_at < 0) rec_prepu();   // no barrier: the head does not touch its words
    if (nsk) {
    } else if (!in->head_exact) {
      dispatch(l, in->head_debug ? K_HEAD_DEBUG : in->head_fast ? ((in->vbits >> VG_HEAD) & 1 ? K_HEAD_FAST_RX : K_HEAD_FAST) : K_HEAD_STRICT,
               nwx * nwy, a);
      if (!(in->dprobe2 & 4)) uav_barrier(l, A);
    } else {   // exact head: weight planes, the integer pass (marks windows it cannot take), strict fixup of those
      dispatch(l, K_HEAD_WPL, 8, a);
      uav_barrier(l, A);
      dispatch(l, K_HEAD, nwx * nwy, a);
      uav_barrier(l, A);
      dispatch(l, K_HEAD_FIXUP, nwx * nwy, a);
      uav_barrier(l, A);
    }
  }
  stamp(q0 + 4);
  // output stage (54): CH[1-par], PH1 (read earlier this frame by the pool and the decode), Q
  {
    std::vector<uint32_t> a = out_args();
    if (in->fast_paths) {   // prep, fast (PH1 included), fallback (always dispatched: it clears the prep flags)
      a.push_back(kOutFastFlags);
      a.push_back((uint32_t)lay.oscr);
      if (!oprep) {   // the render records (unless the fast head wrote them)
        dispatch(l, (in->vbits >> VG_OUT) & 1 ? K_OUT_PREP_F : K_OUT_PREP, std::max<uint32_t>(rw * rh, 513 * 3) / kOutPrepBlock + 1, a);
        uav_barrier(l, A);
      }
      if (tail_rec && in->hiddec_early) decode1(2);   // a14 E7 f: HID is final after the head barrier
      dispatch(l, (in->vbits >> VG_OUT) & 1 ? K_OUT_FAST_F : K_OUT_FAST, ((in->Wh + 31) / 32) * ((in->Hh + 7) / 8), a);
      if (!(in->dprobe2 & 8)) uav_barrier(l, A);
      dispatch(l, (in->vbits >> VG_OUT) & 1 ? K_OUT_FALLBACK_F : K_OUT_FALLBACK, kOutFallbackGroups, a);
      if (!(in->dprobe2 & 16)) uav_barrier(l, A);
    } else {
      a.push_back(3u);   // packed head buffers, E / Ep / invE from the exposure result
      a.push_back(0u);
      const uint64_t hb = (uint64_t)in->Wh * in->Hh;
      dispatch(l, K_OUT_MAIN, (uint32_t)((hb + 31) / 32), a);
      uav_barrier(l, A);
      dispatch(l, K_OUT_POOL, (uint32_t)(((uint64_t)w1 * h1 + 31) / 32), a);
      uav_barrier(l, A);
    }
  }
  stamp(q0 + 5);
  // resolve (54b): its game inputs into readable images, then the plain / extended resolve
  if (!in->host_sched) {
    for (const Job& j : res_jobs) ingest(j);
    if (!res_jobs.empty()) uav_barrier(l, A);
  }
  if (in->fast_paths) {   // fast resolve, then the verified pixel code for the pixels it marked
    const uint32_t tiles = ((in->We + kResTile - 1) / kResTile) * ((in->He + kResTile - 1) / kResTile);
    const bool rx = (in->vbits >> VG_RES) & 1;
    if (fuse_store) l->SetComputeRootDescriptorTable(4, gpu(store_pair[0]));   // u2: the main output (rf_gout)
    dispatch(l, res_spec2 ? (rx ? K_RES_FAST2_X : K_RES_FAST2) : (rx ? K_RES_FAST_X : K_RES_FAST), std::min(tiles, in->res_groups), {(uint32_t)(blk + kBlkResolve), 0u, 0u});
    if (tail_rec) {   // the next frame's history prep: PH1 -> PH2 pool, PH1 / HID decodes (PH1, HID final here)
      dispatch(l, K_POOL_MAIN, (w2 * h2 + kPoolLanes - 1) / kPoolLanes, {(uint32_t)(blk + kBlkPool), (uint32_t)lay.ptab, 0u, 0u});
      decode1(0);
      if (!in->hiddec_early) decode1(2);
    }
    uav_barrier(l, A);
    dispatch(l, K_RES_FIXUP, (uint32_t)(((eb + 31) / 32 + 255) / 256), {(uint32_t)(blk + kBlkResolve), 0u, 0u});
    if (tail_rec) {   // PH2 decode after the pool's barrier; dec_* are next read by the next frame's features
      decode1(1);
      in->tail_ready = true;
      memcpy(in->tail_key, tkey, sizeof tkey);
    }
  } else {
    dispatch(l, K_RES_MAIN, (uint32_t)((eb + 63) / 64), {(uint32_t)(blk + kBlkResolve), 0u, 0u});
  }
  if (!(in->dprobe2 & 32)) uav_barrier(l, A);
  stamp(q0 + 6);
  // stores into the game's output(s): the We x He grid (bounded stores: Wo x Ho) at the output origin
  const bool bounded = (f->settings & WG_RR_SET_BOUNDED_STORES) != 0;
  for (int k = 0; k < 2; ++k) {
    if (!oi[k].on) continue;
    if (k == 0 && fuse_store) {   // stored by the resolve
      uav_barrier(l, oi[0].direct ? out_res[0] : in->out[0].proxy);
      continue;
    }
    const WgRrImage& t = img[k == 0 ? WG_ROLE_OUTPUT : WG_ROLE_OUTPUT_ALPHA];
    const uint32_t sw = bounded ? in->Wo : in->We, sh = bounded ? in->Ho : in->He;
    l->SetComputeRootDescriptorTable(4, gpu(store_pair[k]));
    dispatch(l, K_STORE, (sw * sh + 63) / 64, {(uint32_t)P.off(k == 0 ? Bf.out : Bf.aout), sw, sh, in->We, oi[k].fmt,
                                               oi[k].ch, t.origin_x, t.origin_y, oi[k].srgb ? 1u : 0u});
    uav_barrier(l, oi[k].direct ? out_res[k] : in->out[k].proxy);
  }
  for (UINT q : stamps_late) l->EndQuery(in->qh, D3D12_QUERY_TYPE_TIMESTAMP, q);
  l->EndQuery(in->qh, D3D12_QUERY_TYPE_TIMESTAMP, q0 + 7);
  if (in->arena_mode) {   // E25: the arena stays UNORDERED_ACCESS (the next frame's start transition orders it)
    if (in->arena_mode == 9) uav_barrier(l, A);   // probe: the frames stay ordered (outside the timed span)
    in->arena_uav = true;
  } else {
    uav_barrier(l, A);
    transition(l, A, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
    in->arena_uav = false;
  }
  for (int k = 0; k < 2; ++k) {   // our texture into the game's output (same format family)
    if (!oi[k].on || oi[k].direct) continue;
    ID3D12Resource* px = in->out[k].proxy;
    transition(l, px, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION dst = {}, src = {};
    dst.pResource = out_res[k];
    dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src.pResource = px;
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    // bug hunt G: only the rectangle the store wrote (its grid at the output origin, inside the texture); the rest of
    // our texture holds nothing of this frame, and the game's pixels outside the output rectangle stay as they are
    const WgRrImage& t = img[k == 0 ? WG_ROLE_OUTPUT : WG_ROLE_OUTPUT_ALPHA];
    const uint64_t sw = bounded ? in->Wo : in->We, sh = bounded ? in->Ho : in->He;
    D3D12_BOX box = {};
    box.left = t.origin_x;
    box.top = t.origin_y;
    box.right = (UINT)std::min<uint64_t>((uint64_t)t.origin_x + sw, oi[k].d.Width);
    box.bottom = (UINT)std::min<uint64_t>((uint64_t)t.origin_y + sh, oi[k].d.Height);
    box.front = 0;
    box.back = 1;
    if (box.right > box.left && box.bottom > box.top) l->CopyTextureRegion(&dst, box.left, box.top, 0, &src, &box);
    transition(l, px, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  }
  if (in->barbatch) bar_all(true);
  else for (auto it = bars.rbegin(); it != bars.rend(); ++it) transition_game(l, it->r, it->to, it->from);
  l->ResolveQueryData(in->qh, D3D12_QUERY_TYPE_TIMESTAMP, q0, kTq, in->qrb, (uint64_t)q0 * 8);
  {
    D3D12_WRITEBUFFERIMMEDIATE_PARAMETER p{rt.marker->GetGPUVirtualAddress(), (UINT)(serial & 0xffffffffu)};
    D3D12_WRITEBUFFERIMMEDIATE_MODE mode = D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT;
    list2->WriteBufferImmediate(1, &p, &mode);
  }
  // ================================================================ bookkeeping
  if (g_test_used) {   // the buffer map of t_describe: only built once a test host fetched the test interface
    static const char* const kRoleName[kRoles] = {"C", "Z", "V", "SV", "Dd", "Ds", "N", "R", "Q", "B", "G1", "G2", "Hd"};
    std::string m;
    const std::string NL(1, (char)10);
    auto put = [&](const std::string& n, uint64_t v) { m += n + " " + std::to_string(v) + NL; };
    put("blk", blk); put("partials", P.off(Bf.partials)); put("expo", lay.ex[par]); put("codes", P.off(Bf.codes));
    put("hold", P.off(Bf.hold)); put("x0", x0); put("mark", P.off(Bf.mark)); put("dec_ph1", P.off(Bf.dec_ph1));
    put("dec_ph2", P.off(Bf.dec_ph2)); put("dec_hid", P.off(Bf.dec_hid));
    if (Bf.dx >= 0) put("dx", P.off(Bf.dx));
    for (int s = 0; s < 15; ++s) {
      const std::string n = kStages[s].name;
      put(n + ".y2", P.off(Bf.y2[s])); put(n + ".merged", P.off(Bf.merged[s])); put(n + ".hnew", P.off(Bf.hnew[s]));
      put(n + ".hmerged", P.off(Bf.hmerged[s]));
    }
    if (in->head_debug) { put("dbg_y2h", P.off(Bf.dbg_y2h)); put("dbg_img", P.off(Bf.dbg_img)); }
    put("regroup", P.off(Bf.regroup)); put("cnet", P.off(Bf.cnet)); put("gates", P.off(Bf.gates));
    put("prec", P.off(Bf.prec)); put("candw", P.off(Bf.candw)); put("linear", P.off(Bf.linear)); put("q", P.off(Bf.q));
    put("trace", P.off(Bf.trace)); put("out", P.off(Bf.out)); put("aout", P.off(Bf.aout)); put("layer", P.off(Bf.layer));
    put("opac", P.off(Bf.opac)); put("asrc", P.off(Bf.asrc));
    put("out_fmt", oi[0].fmt); put("aout_fmt", oi[1].fmt); put("rflags", rflags); put("work_bytes", P.work);
    if (sl.on) m += fmt("surf.layer %u %u %u %u %d %d" "\n", sl.w, sl.h, sl.fmt, sl.ch, sl.ox, sl.oy);
    if (so_.on) m += fmt("surf.opac %u %u %u %u %d %d" "\n", so_.w, so_.h, so_.fmt, so_.ch, so_.ox, so_.oy);
    if (sa.on) m += fmt("surf.asrc %u %u %u %u %d %d %d" "\n", sa.w, sa.h, sa.fmt, sa.ch, sa.ox, sa.oy, sa.role);
    for (int r = 0; r < (int)kRoles; ++r)
      if (ing[r].on)
        m += fmt("tex.%s %llu %u %u %u %d %d %d" "\n", kRoleName[r], (unsigned long long)(ing[r].copy ? P.off(Bf.copy[r]) : 0),
                 ing[r].w, ing[r].h, ing[r].ch, ing[r].ox, ing[r].oy, ing[r].copy ? 0 : 1);
    m += fmt("direct %d" "\n" "camera_usable %d" "\n", oi[0].direct ? 1 : 0, camera_usable ? 1 : 0);
    in->map_text = m;
  }
  rt.serial = serial;
  in->last_serial = serial;
  in->slot_serial[slot] = serial;
  in->slot_pending[slot] = true;
  in->slot_frame[slot] = in->frames;
  in->work_bytes = P.work;
  if (old_arena) {
    log_line(fmt("instance %u: arena %s, %llu MiB (render %ux%u)", in->id, (th_new || oh_new) ? "re-laid out" : "grown",
                 (unsigned long long)(in->arena_bytes >> 20), rw, rh));
    defer_release(old_arena, serial);
  }
  if (th_new || oh_new)
    log_line(fmt("instance %u frame %llu: %s created zero-filled (54b.6; this frame resets them)", in->id,
                 (unsigned long long)in->frames, th_new && oh_new ? "TH and OH" : (th_new ? "TH" : "OH")));
  if (rebuild && !in->fresh)
    log_line(fmt("instance %u frame %llu: render size %ux%u -> %ux%u: render-size images and the colour-history pair "
                 "re-created zero-filled (53.9)", in->id, (unsigned long long)in->frames, in->lay.rw, in->lay.rh, rw, rh));
  if (in->fresh)
    log_line(fmt("instance %u first evaluate: render %ux%u, tensor %ux%u, output %ux%u, %s, reset %u, resolve %s%s%s%s",
                 in->id, rw, rh, Wt, Ht, in->Wo, in->Ho, oi[0].direct ? "output written directly" : "output through our texture and a copy",
                 reset ? 1u : 0u, (layer_on || alpha_up) ? "extended" : "plain", layer_on ? " + layer" : "",
                 opac_on ? " + opacity" : "", alpha_up ? (aout_on ? " + alpha upscaling (separate output alpha)" : " + alpha upscaling") : ""));
  else if (reset)
    log_line(fmt("instance %u frame %llu: reset frame (parity %u, counter %u)", in->id, (unsigned long long)in->frames, par,
                 f->counter));
  if (res && res->camera_fallback_used && !in->logged_fallback) {
    in->logged_fallback = true;
    log_line(fmt("instance %u: camera matrices unusable; specular motion = motion vectors (62.4, logged once)", in->id));
  }
  if (alpha_up) in->ah_created = false;
  in->lay = lay;
  in->fresh = false;
  return WG_OK;
}

// ---------------------------------------------------------------------------------------------------- transients
uint32_t retain_locked(void* cmd, void* resource, uint32_t format, uint32_t state, void** out) {
  Runtime& rt = R();
  *out = nullptr;
  if (!rt.dev || !cmd || !resource) return WG_E_INVALID_PARAMETER;
  ID3D12GraphicsCommandList* l = nullptr;
  if (FAILED(((IUnknown*)cmd)->QueryInterface(IID_PPV_ARGS(&l)))) return WG_E_INVALID_PARAMETER;
  // Issue #65: the completion marker below needs ID3D12GraphicsCommandList2 (evaluate refuses a list without it). Ask
  // first: nothing is created or recorded for a list that cannot take the marker, as without the marker the copy's
  // release would not wait for the list.
  ID3D12GraphicsCommandList2* l2 = nullptr;
  if (FAILED(l->QueryInterface(IID_PPV_ARGS(&l2)))) {
    release(l);
    return WG_E_INVALID_PARAMETER;
  }
  ID3D12Resource* src = (ID3D12Resource*)resource;
  D3D12_RESOURCE_DESC d = src->GetDesc();
  // bug hunt C: a texture with mips is copied as its mip 0 (texture_shape.h: a copy uses subresource 0; the runtime reads
  // mip 0), so a barrier on subresource 0 alone covers the copy; a single-mip one whole, as before (planes included)
  const bool mip0 = d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D && d.MipLevels > 1;
  if (mip0) d.MipLevels = 1;
  d.Flags &= ~(D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL |
               D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS);
  if (format && d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D && d.Format == DXGI_FORMAT_UNKNOWN) d.Format = (DXGI_FORMAT)format;
  D3D12_HEAP_PROPERTIES hp = {};
  hp.Type = D3D12_HEAP_TYPE_DEFAULT;
  ID3D12Resource* c = nullptr;
  if (FAILED(rt.dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                             IID_PPV_ARGS(&c)))) {
    release(l2);
    release(l);
    return WG_E_D3D_API;
  }
  rt.live_resources++, rt.created_resources++;
  // bug hunt C: the source's declared state (retain_transient_state) is moved to COPY_SOURCE for the copy and back (the
  // game's barriers expect it there); without one (retain_transient) it is copied as it is (implicit promotion from COMMON)
  const D3D12_RESOURCE_STATES from = (D3D12_RESOURCE_STATES)state;
  const bool move = state != 0xffffffffu && !(state & D3D12_RESOURCE_STATE_COPY_SOURCE);
  if (move) transition_game(l, src, from, D3D12_RESOURCE_STATE_COPY_SOURCE);
  if (mip0) {
    D3D12_TEXTURE_COPY_LOCATION dl = {}, sl = {};
    dl.pResource = c;
    dl.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    sl.pResource = src;
    sl.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    l->CopyTextureRegion(&dl, 0, 0, 0, &sl, nullptr);
  } else {
    l->CopyResource(c, src);
  }
  if (move) transition_game(l, src, D3D12_RESOURCE_STATE_COPY_SOURCE, from);
  transition(l, c, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
  // bug hunt B: the copy runs when the game's list does, and a copy alone never advanced rt.serial: a release before the
  // next frame (the input re-tagged or untagged) freed it at once, before its CopyResource ran. The copy gets a serial of
  // its own, written into the same list after it (the frames' marker), so release_transient's defer waits for that list.
  if (rt.marker) {
    const uint64_t serial = rt.serial + 1;
    D3D12_WRITEBUFFERIMMEDIATE_PARAMETER p{rt.marker->GetGPUVirtualAddress(), (UINT)(serial & 0xffffffffu)};
    D3D12_WRITEBUFFERIMMEDIATE_MODE mode = D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT;
    l2->WriteBufferImmediate(1, &p, &mode);
    rt.serial = serial;
  }
  release(l2);
  release(l);
  rt.transient.push_back(c);
  *out = c;
  return WG_OK;
}
void release_transient_locked(void* copy) {
  Runtime& rt = R();
  auto it = std::find(rt.transient.begin(), rt.transient.end(), (ID3D12Resource*)copy);
  if (it == rt.transient.end()) return;
  ID3D12Resource* r = *it;
  rt.transient.erase(it);
  defer_release(r, rt.serial);
  // issue #8: create, evaluate and destroy run pump() too, but a game that tags its inputs every frame while no
  // instance exists (RR available, switched off) calls none of them, and the released copies were never freed.
  // pump() frees only resources whose frame marker the GPU has written (the same test as at evaluate), takes no lock
  // of its own and calls nothing back. The caller, a_release_transient, holds rt.mu, as every other user of the deferred
  // list does.
  pump();
}

// ---------------------------------------------------------------------------------------------------- the adapter
uint32_t a_status(void*) {
  Runtime& rt = R();
  std::lock_guard<std::recursive_mutex> lock(rt.mu);
  init_once();
  uint32_t s = 0;
  if (rt.adapter_present && !rt.device_failed) s |= WG_RR_RT_READY;
  if (rt.model.ok) s |= WG_RR_RT_MODEL_LOADED;
  // release 0.6.0: the default frame's kernels are still being created (a cold driver cache on the first start: minutes);
  // the plugin tells OptiScaler's overlay "WrongGPU RR (preparing...)" meanwhile
  if (rt.compile_thread.joinable() && !rt.kernels_ready && rt.compile_failed < 0 && !rt.compile_cancel) s |= WG_RR_RT_PREPARING;
  return s;
}
uint32_t a_adapter_supported(void*, const void* luid, uint32_t n) {
  Runtime& rt = R();
  std::lock_guard<std::recursive_mutex> lock(rt.mu);
  init_once();
  if (!luid || n != sizeof(LUID)) return 0;
  LUID l;
  memcpy(&l, luid, sizeof l);
  IDXGIFactory4* fac = nullptr;
  IDXGIAdapter1* a = nullptr;
  uint32_t ok = 0;
  if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&fac))) && SUCCEEDED(fac->EnumAdapterByLuid(l, IID_PPV_ARGS(&a)))) {
    DXGI_ADAPTER_DESC1 d;
    a->GetDesc1(&d);
    ok = rdna4_adapter_desc(d, nullptr) ? 1u : 0u;
  }
  release(a);
  release(fac);
  return ok;
}
void a_release_version(void*, uint32_t out[3]) {
  out[0] = WG_RR_RUNTIME_VERSION0;
  out[1] = WG_RR_RUNTIME_VERSION1;
  out[2] = WG_RR_RUNTIME_VERSION2;
}
uint32_t a_set_device(void*, void* d) {
  std::lock_guard<std::recursive_mutex> lock(R().mu);
  return set_device_locked((ID3D12Device*)d);
}
uint32_t a_create(void*, const WgRrCreate* d, void** out) {
  std::lock_guard<std::recursive_mutex> lock(R().mu);
  return create_locked(d, out);
}
void a_destroy(void*, void* inst) {
  std::lock_guard<std::recursive_mutex> lock(R().mu);
  destroy_locked((Instance*)inst);
}
uint64_t a_vram(void*, void* inst) {
  std::lock_guard<std::recursive_mutex> lock(R().mu);
  Instance* in = (Instance*)inst;
  if (!in) return 0;
  uint64_t b = in->arena_bytes + (uint64_t)kSlots * kBlock + (uint64_t)kSlots * kTq * 8;
  for (const OutTarget& o : in->out)
    if (o.proxy) b += (uint64_t)o.proxy_w * o.proxy_h * 16;
  return b;
}
// ---- clean/rr-f-live1: native sizes, the first evaluates in the log, the exception guard, the pass-through

// Sizes and formats the proxy could not fill (the game hands 0 x 0 / format 0 in the resource description) from the
// native resource, which the runtime holds anyway.
void fill_native(WgRrFrame* f) {
  for (int r = 0; r < WG_ROLE_COUNT; ++r) {
    WgRrImage& t = f->images[r];
    if (!(f->attached & (1u << r)) || !t.resource) continue;
    if (t.width && t.height && t.format) continue;
    const D3D12_RESOURCE_DESC d = ((ID3D12Resource*)t.resource)->GetDesc();
    if (!t.width || !t.height) t.width = (uint32_t)d.Width, t.height = d.Height;
    if (!t.format) t.format = (uint32_t)d.Format;
  }
}

void log_evaluate(uint64_t n, const Instance* in, const WgRrFrame& f, const WgRrFrame& g, uint32_t rc) {
  static const char* const kRole[WG_ROLE_COUNT] = {"colour", "output", "depth", "motion", "diffuse", "specular",
      "normals", "roughness", "spec-motion", "hit-dist", "disocclusion", "G1", "dof", "before-transp", "layer",
      "opacity", "in-alpha", "out-alpha"};
  std::string s = fmt("evaluate %llu (instance %u): rc %u, render %ux%u -> output %ux%u, reset %u, parity %u, counter %u, "
                      "jitter (%.4f, %.4f), mv scale (%.1f, %.1f), linear depth %u, settings 0x%x; images:",
                      (unsigned long long)n, in ? in->id : 0u, rc, f.render_width, f.render_height, f.output_width,
                      f.output_height, f.reset, f.parity, f.counter, f.jitter[0], f.jitter[1], f.mv_scale[0],
                      f.mv_scale[1], f.depth_is_linear, f.settings);
  for (int r = 0; r < WG_ROLE_COUNT; ++r) {
    if (!(f.attached & (1u << r))) continue;
    const WgRrImage& a = f.images[r];
    const WgRrImage& b = g.images[r];
    s += fmt(" %s[type %u fmt %u class %u state 0x%x size %ux%u%s at (%u,%u) ext %ux%u life %u res %p]", kRole[r],
             a.buffer_type, b.format, a.format_class, a.state, b.width, b.height,
             (a.width != b.width || a.height != b.height) ? " (handed 0x0: native)" : "", a.origin_x, a.origin_y,
             a.extent_width, a.extent_height, a.lifetime, b.resource);
  }
  log_line(s);
}

enum : uint32_t { kEvalException = 0xFFFFFFF0u };
__declspec(noinline) uint32_t evaluate_body(Instance* in, const WgRrFrame* f, WgRrFrameResult* res, bool crash) {
  if (crash) *(volatile int*)nullptr = 0;   // test_crash: an access violation inside the guarded evaluate
  return evaluate_locked(in, f, res);
}
// no C++ objects in this frame (SEH); an exception disables the runtime (the proxy then passes frames through)
uint32_t evaluate_guarded(Instance* in, const WgRrFrame* f, WgRrFrameResult* res, bool crash) {
  __try {
    return evaluate_body(in, f, res, crash);
  } __except (EXCEPTION_EXECUTE_HANDLER) {
    return kEvalException;
  }
}

// an instance on an opt-in variant needs every kernel (kernel_opt_in); one on another mode's variants (or a per-knob
// override of one) the second pass's (clean/rr-f-s10-modes)
bool instance_needs_modes(const Instance* in) { return in->vbits != R().start_vbits; }
bool instance_needs_opt_in(const Instance* in) {
  // the three input-features kernels (exact, lite, split) are all part of the default frame (small, compiled first)
  // clean/rr-f-s10-modes: also an instance whose variants (mode, per-knob overrides) differ from the first pass's
  return !in->head_fast || !in->body_e || !in->fast_paths;
}
uint32_t a_evaluate(void*, void* inst, const WgRrFrame* f, WgRrFrameResult* res) {
  Runtime& rt = R();
  std::lock_guard<std::recursive_mutex> lock(rt.mu);
  Instance* in = (Instance*)inst;
  if (res) res->passes_started = 0, res->input_jitter_consumed = 0, res->camera_fallback_used = 0;
  if (!in || !f || f->size < sizeof(WgRrFrame)) return WG_E_INVALID_PARAMETER;
  if (rt.kernels_ready && !rt.compile_logged) {
    rt.compile_logged = true;
    log_line(fmt("frame kernels ready after %llu ms in the background; RR runs from this frame (a reset frame)",
                 (unsigned long long)rt.compile_ms.load()));
  }
  // i15: next-frame mode switch. The mode is resolved once per evaluate (shared settings block > WG_RR_MODE > runtime
  // ini > default); a change applies on THIS frame when the other modes' kernels are prepared (modes_ready), as a reset
  // frame (no history sample is read) with the new mode's variants; the prepared next-frame history (tail_ready) belongs to the old kernels and is dropped.
  // Arena and history sizes stay (creation-time; WG_RR_HSCALE is a creation-time, fast-only test knob).
  bool switch_reset = false;
  {
    const char* msrc = "";
    bool known = true;
    const Mode want = resolve_mode(&msrc, nullptr, &known);
    if (want != in->mode) {
      const bool held = in->frames < in->hold_modes;   // WG_RR_TEST_HOLD_MODES (test only)
      if (!rt.modes_ready || held) {
        if (in->deferred_logged != want) {
          in->deferred_logged = want;
          log_line(fmt("instance %u: rr_mode switch to %s (%s) deferred: kernels still being prepared%s", in->id,
                       mode_name(want), msrc, held ? " (WG_RR_TEST_HOLD_MODES)" : ""));
        }
      } else {
        const Mode from = in->mode;
        const std::string vover = apply_mode(in, want);
        apply_dflow(in);
        char nv[8] = {};
        in->netalt = in->mode == MODE_FAST && GetEnvironmentVariableA("WG_RR_NETALT", nv, sizeof nv) && strcmp(nv, "1") == 0;
        in->tail_ready = false;
        in->deferred_logged = -1;
        switch_reset = true;
        log_line(fmt("instance %u: rr_mode %s -> %s at frame %llu (%s%s; history reset)", in->id, mode_name(from),
                     mode_name(want), (unsigned long long)(in->frames + 1), msrc, vover.c_str()));
      }
    }
  }
  WgRrFrame g = *f;
  fill_native(&g);
  // the frame the switch applies to is a reset frame: the game's reset flag set, everything else as the game sent it (the
  // camera is continuous across a mode switch), i.e. exactly what a new instance of the new mode sees on a reset frame
  if (switch_reset) g.reset = 1;
  const uint64_t n = in->frames + 1;   // this evaluate's number on the instance
  uint32_t rc;
  if (!(instance_needs_opt_in(in) ? rt.all_ready : instance_needs_modes(in) ? rt.modes_ready : rt.kernels_ready)) {
    in->frames++;
    if (rt.compile_failed >= 0)
      rc = fail(in, std::string("pipeline state failed: ") + kKernelNames[rt.compile_failed.load()], res, WG_E_D3D_API);
    else
      rc = fail(in, "frame kernels still being prepared (first start with a cold driver cache)", res,
                WG_E_COMPUTE_FAILED);
  } else if (rt.disabled) {
    in->frames++;
    rc = fail(in, "runtime disabled after an exception in an earlier evaluate", res, WG_E_EXCEPTION_HANDLER);
  } else if (cfg_list_has("test_fail", n)) {
    in->frames++;
    rc = fail(in, fmt("injected failure (test_fail, evaluate %llu)", (unsigned long long)n), res, WG_E_COMPUTE_FAILED);
  } else {
    rc = evaluate_guarded(in, &g, res, cfg_list_has("test_crash", n));
    if (rc == kEvalException) {
      rt.disabled = true;
      in->failed++;
      if (res) res->passes_started = 0;
      log_line(fmt("instance %u evaluate %llu: exception caught inside the evaluate; RR disabled for this session "
                   "(frames pass through)", in->id, (unsigned long long)n));
      rc = WG_E_EXCEPTION_HANDLER;
    }
  }
  static uint64_t limit = UINT64_MAX;
  if (limit == UINT64_MAX) {
    const std::string v = cfg("log_evaluates");
    limit = v.empty() ? 20 : strtoull(v.c_str(), nullptr, 10);
  }
  if (rt.evaluates_logged < limit) {
    rt.evaluates_logged++;
    log_evaluate(n, in, *f, g, rc);
  } else {
    // clean/release-060 crash-cp2077: every evaluate in full for 8 frames after an instance's first frame, a reset frame
    // (a save load) or a change in the images (sizes, formats, rectangles, attached roles, resources); 4000 lines at most
    std::string sig = fmt("%u %u %u %u %x", g.render_width, g.render_height, g.output_width, g.output_height, g.attached);
    for (int r = 0; r < WG_ROLE_COUNT; ++r)
      if (g.attached & (1u << r)) {
        const WgRrImage& a = g.images[r];
        sig += fmt(" %u:%u:%ux%u:%u,%u:%ux%u:%p", r, a.format, a.width, a.height, a.origin_x, a.origin_y, a.extent_width,
                   a.extent_height, a.resource);
      }
    if (n == 1 || g.reset || sig != in->tlog_sig) in->tlog_left = 8;
    in->tlog_sig = sig;
    if (in->tlog_left && rt.tlog_logged < 4000) {
      in->tlog_left--;
      rt.tlog_logged++;
      log_evaluate(n, in, *f, g, rc);
    }
  }
  return rc;
}

// Pass-through of the game's colour into the output (bilinear, render -> output rectangle): the proxy records it on
// the evaluate's command list when the runtime failed, so the frame stays presentable. Needs only the device's
// pipelines (no instance). Output through its typed UAV only (no UAV flag / sRGB: refused, 25).
uint32_t a_passthrough(void*, const WgRrFrame* f0) {
  Runtime& rt = R();
  std::lock_guard<std::recursive_mutex> lock(rt.mu);
  if (!f0 || f0->size < sizeof(WgRrFrame)) return WG_E_INVALID_PARAMETER;
  if (!rt.dev || !rt.rs || !rt.pso[K_PASS] || !rt.marker || rt.device_failed) return WG_E_DEVICE_NOT_CREATED;
  WgRrFrame f = *f0;
  fill_native(&f);
  const WgRrImage& c = f.images[WG_ROLE_COLOUR];
  const WgRrImage& o = f.images[WG_ROLE_OUTPUT];
  if (!c.resource || !o.resource || !f.command_list) return WG_E_MISSING_INPUT_PARAMETER;
  auto refuse = [&](const std::string& why) {
    if (!rt.logged_pass_refused) {
      rt.logged_pass_refused = true;
      log_line("pass-through refused (logged once): " + why);
    }
    return (uint32_t)WG_E_INVALID_PARAMETER;
  };
  ID3D12Resource* cr = (ID3D12Resource*)c.resource;
  ID3D12Resource* orr = (ID3D12Resource*)o.resource;
  const D3D12_RESOURCE_DESC od = orr->GetDesc();
  if (!wgi::texture_shape_ok(cr->GetDesc()) || !wgi::texture_shape_ok(od))  // bug hunt E (as evaluate)
    return refuse(std::string("colour or output: ") + wgi::texture_shape_rule());
  const DXGI_FORMAT of = srv_format((DXGI_FORMAT)(o.format ? o.format : od.Format), false);
  if (!(od.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) || is_srgb(of))
    return refuse("the output has no typed UAV");
  const uint32_t rw = f.render_width ? f.render_width : c.width - c.origin_x;
  const uint32_t rh = f.render_height ? f.render_height : c.height - c.origin_y;
  const uint32_t ow = f.output_width ? f.output_width : (uint32_t)od.Width - o.origin_x;
  const uint32_t oh = f.output_height ? f.output_height : od.Height - o.origin_y;
  if (!rw || !rh || !ow || !oh || c.origin_x + rw > c.width || c.origin_y + rh > c.height ||
      o.origin_x + ow > od.Width || o.origin_y + oh > od.Height)
    return refuse(fmt("rectangles outside the resources (render %ux%u, output %ux%u)", rw, rh, ow, oh));
  ID3D12GraphicsCommandList* l = nullptr;
  ID3D12GraphicsCommandList2* l2 = nullptr;
  if (FAILED(((IUnknown*)f.command_list)->QueryInterface(IID_PPV_ARGS(&l)))) return WG_E_INVALID_PARAMETER;
  if (FAILED(l->QueryInterface(IID_PPV_ARGS(&l2)))) {
    release(l);
    return refuse("no ID3D12GraphicsCommandList2");
  }
  if (!rt.pass_heap) {
    D3D12_DESCRIPTOR_HEAP_DESC hd = {};
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    hd.NumDescriptors = kHeapFixed + 2 * kSlots;
    hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (FAILED(rt.dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&rt.pass_heap))) ||
        FAILED(make_buffer(256, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                           D3D12_RESOURCE_STATE_COMMON, &rt.pass_dummy))) {
      release(rt.pass_heap);
      release(l2);
      release(l);
      return WG_E_D3D_API;
    }
    init_heap(rt.pass_heap);
  }
  const uint64_t serial = rt.serial + 1;
  const uint32_t slot = (uint32_t)(serial % kSlots);
  if (!wait_serial(rt.pass_serial[slot], 5000)) {
    release(l2);
    release(l);
    return WG_E_D3D_API;
  }
  D3D12_CPU_DESCRIPTOR_HANDLE hc = rt.pass_heap->GetCPUDescriptorHandleForHeapStart();
  D3D12_GPU_DESCRIPTOR_HANDLE hg = rt.pass_heap->GetGPUDescriptorHandleForHeapStart();
  hc.ptr += (uint64_t)(kHeapFixed + 2 * slot) * rt.inc;
  hg.ptr += (uint64_t)(kHeapFixed + 2 * slot) * rt.inc;
  D3D12_SHADER_RESOURCE_VIEW_DESC sd = {};
  sd.Format = srv_format((DXGI_FORMAT)c.format, false);
  sd.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
  sd.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
  sd.Texture2D.MipLevels = 1;
  rt.dev->CreateShaderResourceView(cr, &sd, hc);
  D3D12_CPU_DESCRIPTOR_HANDLE hu = hc;
  hu.ptr += rt.inc;
  D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
  ud.Format = uav_format(of);
  ud.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
  rt.dev->CreateUnorderedAccessView(orr, nullptr, &ud, hu);
  publish_descriptors((const void*)hc.ptr, (size_t)2 * rt.inc, nullptr);
  // declared states -> NON_PIXEL_SHADER_RESOURCE (colour) / UNORDERED_ACCESS (output), restored afterwards
  std::vector<Bar> bars;
  auto want = [&](ID3D12Resource* r, uint32_t declared, D3D12_RESOURCE_STATES to) {
    if (declared == 0xffffffffu || (declared & (uint32_t)to) == (uint32_t)to) return;
    for (ID3D12Resource* t : rt.transient)
      if (t == r) return;
    bars.push_back({r, (D3D12_RESOURCE_STATES)declared, to});
  };
  want(cr, c.state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
  want(orr, o.state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  for (const Bar& b : bars) transition_game(l, b.r, b.from, b.to);
  ID3D12DescriptorHeap* heap = rt.pass_heap;
  l->SetDescriptorHeaps(1, &heap);
  l->SetComputeRootSignature(rt.rs);
  l->SetComputeRootUnorderedAccessView(0, rt.pass_dummy->GetGPUVirtualAddress());
  l->SetComputeRootUnorderedAccessView(1, rt.pass_dummy->GetGPUVirtualAddress());
  D3D12_GPU_DESCRIPTOR_HANDLE t0 = rt.pass_heap->GetGPUDescriptorHandleForHeapStart();
  l->SetComputeRootDescriptorTable(3, t0);
  D3D12_GPU_DESCRIPTOR_HANDLE t5 = t0;
  t5.ptr += 3 * rt.inc;
  l->SetComputeRootDescriptorTable(5, t5);
  l->SetComputeRootDescriptorTable(4, hg);
  const float sx = (float)rw / (float)ow, sy = (float)rh / (float)oh;
  dispatch(l, K_PASS, (uint32_t)(((uint64_t)ow * oh + 63) / 64),
           {c.origin_x, c.origin_y, rw, rh, o.origin_x, o.origin_y, ow, oh, fb(sx), fb(sy)});
  uav_barrier(l, orr);
  for (auto it = bars.rbegin(); it != bars.rend(); ++it) transition_game(l, it->r, it->to, it->from);
  D3D12_WRITEBUFFERIMMEDIATE_PARAMETER p{rt.marker->GetGPUVirtualAddress(), (UINT)(serial & 0xffffffffu)};
  D3D12_WRITEBUFFERIMMEDIATE_MODE mode = D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT;
  l2->WriteBufferImmediate(1, &p, &mode);
  rt.serial = serial;
  rt.pass_serial[slot] = serial;
  if (++rt.passes == 1) log_line(fmt("pass-through: first frame passed through (colour %ux%u -> output %ux%u)", rw, rh, ow, oh));
  release(l2);
  release(l);
  return WG_OK;
}

uint32_t a_describe_size(void*, void* res, uint32_t* w, uint32_t* h) {
  if (!res || !w || !h) return WG_E_INVALID_PARAMETER;
  const D3D12_RESOURCE_DESC d = ((ID3D12Resource*)res)->GetDesc();
  *w = (uint32_t)d.Width;
  *h = d.Height;
  return WG_OK;
}
uint32_t a_retain(void*, void* cmd, void* res, uint32_t format, void** out) {
  std::lock_guard<std::recursive_mutex> lock(R().mu);
  return retain_locked(cmd, res, format, 0xffffffffu, out);
}
uint32_t a_retain_state(void*, void* cmd, void* res, uint32_t format, uint32_t state, void** out) {  // bug hunt C
  std::lock_guard<std::recursive_mutex> lock(R().mu);
  return retain_locked(cmd, res, format, state, out);
}
void a_release_transient(void*, void* copy) {
  std::lock_guard<std::recursive_mutex> lock(R().mu);
  release_transient_locked(copy);
}
uint32_t a_describe(void*, void* res, uint32_t* format, uint32_t* samples) {
  if (!res || !format || !samples) return WG_E_INVALID_PARAMETER;
  const D3D12_RESOURCE_DESC d = ((ID3D12Resource*)res)->GetDesc();
  *format = (uint32_t)d.Format;
  *samples = d.SampleDesc.Count;
  return WG_OK;
}

float a_frame_gpu_ms(void*) { return g_frame_gpu_ms.load(std::memory_order_relaxed); }

WgRrRuntimeV1 g_table = {sizeof(WgRrRuntimeV1), WG_RR_ADAPTER_VERSION, nullptr, a_status, a_adapter_supported,
                         a_release_version, a_set_device, a_create, a_destroy, a_vram, a_evaluate, a_retain,
                         a_release_transient, a_describe, a_describe_size, a_passthrough, a_frame_gpu_ms,
                         a_retain_state};

// ---------------------------------------------------------------------------------------------------- test hooks
uint32_t t_dump(void* inst, const wchar_t* path) {
  Runtime& rt = R();
  std::lock_guard<std::recursive_mutex> lock(rt.mu);
  Instance* in = (Instance*)inst;
  if (!in || !rt.dev) return WG_E_INVALID_PARAMETER;
  if (!wait_serial(in->last_serial, 10000)) return WG_E_D3D_API;
  ID3D12Resource* rb = nullptr;
  if (FAILED(make_buffer(in->arena_bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE,
                         D3D12_RESOURCE_STATE_COPY_DEST, &rb)))
    return WG_E_D3D_API;
  if (in->arena_uav) transition(rt.l, in->arena, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
  rt.l->CopyBufferRegion(rb, 0, in->arena, 0, in->arena_bytes);
  if (in->arena_uav) transition(rt.l, in->arena, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
  in->arena_uav = false;
  std::string err;
  uint32_t rc = submit_wait(&err) ? WG_OK : WG_E_D3D_API;
  if (rc == WG_OK) {
    void* m = nullptr;
    D3D12_RANGE r{0, (SIZE_T)in->arena_bytes};
    FILE* f = _wfopen(path, L"wb");
    if (f && SUCCEEDED(rb->Map(0, &r, &m))) {
      fwrite(m, 1, (size_t)in->arena_bytes, f);
      rb->Unmap(0, nullptr);
    } else {
      rc = WG_E_IO;
    }
    if (f) fclose(f);
  }
  drop(rb);
  return rc;
}
uint32_t t_layout(void* inst, uint64_t* out, uint32_t n) {
  std::lock_guard<std::recursive_mutex> lock(R().mu);
  Instance* in = (Instance*)inst;
  if (!in || n < 16) return WG_E_INVALID_PARAMETER;
  const Layout& L = in->lay;
  const uint64_t v[24] = {L.model, L.ch[0], L.ch[1], L.ph1, L.ph2, L.ex[0], L.ex[1], L.fixed_end,
                          L.hid, L.pm[0], L.pm[1], L.pe, L.pn, L.state_end, in->arena_bytes, (uint64_t)L.rw | ((uint64_t)L.rh << 32),
                          L.ah[0], L.ah[1], L.th[0], L.th[1], L.oh[0], L.oh[1], in->work_bytes, in->alias ? 1u : 0u};
  memcpy(out, v, sizeof(uint64_t) * std::min<uint32_t>(n, 24));
  return WG_OK;
}
uint32_t t_describe(void* inst, char* buf, uint32_t size) {
  std::lock_guard<std::recursive_mutex> lock(R().mu);
  Instance* in = (Instance*)inst;
  if (!in || !buf || size <= in->map_text.size()) return WG_E_INVALID_PARAMETER;
  memcpy(buf, in->map_text.c_str(), in->map_text.size() + 1);
  return WG_OK;
}
void t_counters(int64_t out[4]) {
  Runtime& rt = R();
  std::lock_guard<std::recursive_mutex> lock(rt.mu);
  pump();
  out[0] = rt.live_resources.load();
  out[1] = rt.live_instances;
  out[2] = (int64_t)rt.deferred.size();
  out[3] = rt.created_resources.load();
}
uint32_t t_drain(uint32_t ms) {
  Runtime& rt = R();
  std::lock_guard<std::recursive_mutex> lock(rt.mu);
  const bool ok = wait_serial(rt.serial, ms);
  pump();
  return ok ? WG_OK : WG_E_D3D_API;
}
uint32_t t_shutdown() {
  Runtime& rt = R();
  std::lock_guard<std::recursive_mutex> lock(rt.mu);
  if (rt.live_instances) return WG_E_INVALID_STATE;
  wait_serial(rt.serial, 5000);
  for (ID3D12Resource*& t : rt.transient) drop(t);
  rt.transient.clear();
  drop_device();
  return WG_OK;
}
WgRrRuntimeTestV1 g_test = {sizeof(WgRrRuntimeTestV1), t_dump, t_layout, t_counters, t_drain, t_shutdown, t_describe};

}  // namespace
}  // namespace wgrr_rt

extern "C" __declspec(dllexport) const WgRrRuntimeV1* WrongGPU_RrRuntimeV1(void) { return &wgrr_rt::g_table; }
extern "C" __declspec(dllexport) const WgRrRuntimeTestV1* WrongGPU_RrRuntimeTestV1(void) {
  wgrr_rt::g_test_used = true;
  return &wgrr_rt::g_test;
}

// Test hook (CPU only): loads a model file as the runtime does and writes <out>.img (device image) and <out>.txt
// ("blob <name> <offset> <size>", "scalar <name> <bits>", "rho_guarded <n>", "kind <kind>"); flags: reserved.
// Returns 0, or 1 with the reason in <out>.txt.
extern "C" __declspec(dllexport) uint32_t WrongGPU_RrModelDumpV1(const wchar_t* model, const wchar_t* out, uint32_t flags) {
  wgrr_rt::Model m;
  std::string err;
  const std::wstring txt = std::wstring(out) + L".txt";
  FILE* t = _wfopen(txt.c_str(), L"wb");
  if (!t) return 1;
  if (!wgrr_rt::load_model(model, &m, &err)) {
    fprintf(t, "error %s\n", err.c_str());
    fclose(t);
    return 1;
  }
  (void)flags;
  FILE* b = _wfopen((std::wstring(out) + L".img").c_str(), L"wb");
  if (b) {
    fwrite(m.image.data(), 1, m.image.size(), b);
    fclose(b);
  }
  fprintf(t, "kind %s\nflags %u\nrho_guarded %u\n", m.kind.c_str(), m.flags, m.rho_guarded);
  for (auto& kv : m.blobs) fprintf(t, "blob %s %llu %llu\n", kv.first.c_str(), (unsigned long long)kv.second.first,
                                   (unsigned long long)kv.second.second);
  for (auto& kv : m.scalars) fprintf(t, "scalar %s %u\n", kv.first.c_str(), kv.second);
  fclose(t);
  return 0;
}

// Test hook (CPU only): rho of page 55.3 for s2 (bits), 65536 f32 into out; force: every entry by the double-double
// path. Returns the number of entries the binary64 evaluation did not decide.
extern "C" __declspec(dllexport) uint32_t WrongGPU_RrRhoV1(uint32_t s2_bits, float* out, uint32_t force) {
  float s2;
  memcpy(&s2, &s2_bits, 4);
  return wgrr_rt::head_rho(s2, out, force != 0);
}

// release 0.6.0 (owner 2026-11-02: the kernels are prepared inside the launcher): the launcher's helper prepares the RR
// kernels without a game. The helper runs under the game's program file name (the driver keeps its shader cache per
// program name, see make_pso), so the game finds them ready. A device on the RDNA 4 adapter goes through set-device's own
// path and every pipeline of the first `passes` passes (1: the default frame of the default mode, 2: + the other modes'
// kernels, 3: + the opt-in variants) is created. progress(ctx, done, total, pass, kernel, ms, quick) per kernel (quick: it
// took under a second, the driver had it), and with kernel = null when a pass is complete; a nonzero answer cancels
// (what was made is kept by the driver). Returns 0 prepared, 1 cancelled, 2 no RDNA 4 adapter, 3 a device or pipeline
// failure; the details go to the log (WG_RR_LOG, else wronggpu-rr.log next to this library).
extern "C" __declspec(dllexport) uint32_t WrongGPU_RrPrepareKernelsV1(uint32_t passes, wgrr_rt::RrPrepareFn progress, void* ctx) {
  using namespace wgrr_rt;
  SetEnvironmentVariableA("WG_RR_MODE", "accurate");   // the default mode's frame is pass 0
  SetEnvironmentVariableA("WG_RR_COMPILE_WAIT_MS", "0");
  Runtime& rt = R();
  ID3D12Device* dev = nullptr;
  {
    IDXGIFactory1* fac = nullptr;
    IDXGIAdapter1* pick = nullptr;
    if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&fac)))) {
      IDXGIAdapter1* a = nullptr;
      for (UINT i = 0; !pick && fac->EnumAdapters1(i, &a) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 d;
        a->GetDesc1(&d);
        if (rdna4_adapter_desc(d, nullptr)) pick = a;
        else a->Release();
      }
      fac->Release();
    }
    if (!pick) {
      log_line("prepare: no RDNA 4 adapter (Radeon RX 9070 / 9060 series): nothing to prepare");
      return 2;
    }
    const HRESULT h = D3D12CreateDevice(pick, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&dev));
    pick->Release();
    if (FAILED(h)) {
      log_line(fmt("prepare: no D3D12 device (0x%08lx)", (unsigned long)h));
      return 3;
    }
  }
  const ULONGLONG t0 = GetTickCount64();
  uint32_t rc;
  {
    std::lock_guard<std::recursive_mutex> lock(rt.mu);
    g_prep = Prep();
    g_prep.on = true;
    g_prep.cb = progress;
    g_prep.ctx = ctx;
    g_prep.passes = passes;
    rc = set_device_locked(dev);
  }
  if (rc == WG_OK && rt.compile_thread.joinable()) rt.compile_thread.join();   // the worker never takes mu
  const bool cancelled = rt.compile_cancel.load();
  const bool failed = rc != WG_OK || rt.compile_failed >= 0;
  char exe[MAX_PATH] = {};
  GetModuleFileNameA(nullptr, exe, MAX_PATH);
  log_line(fmt("prepare: %s after %llu ms as %s (%u of %u kernels, %u of %u passes)", failed ? "failed" : cancelled ? "cancelled" : "done",
               (unsigned long long)(GetTickCount64() - t0), exe, g_prep.done, g_prep.total, g_prep.passes_done,
               std::min<uint32_t>(std::max<uint32_t>(passes, 1u), 3u)));
  {
    std::lock_guard<std::recursive_mutex> lock(rt.mu);
    drop_device();
    g_prep = Prep();
  }
  dev->Release();
  if (failed) return rc == WG_E_ADAPTER_NOT_SUPPORTED ? 2 : 3;
  return cancelled ? 1 : 0;
}

BOOL WINAPI DllMain(HINSTANCE h, DWORD reason, LPVOID) {
  if (reason == DLL_PROCESS_ATTACH) {
    wgrr_rt::g_module = h;
    DisableThreadLibraryCalls(h);
  }
  return TRUE;
}
