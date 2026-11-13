// SPDX-License-Identifier: MIT
// WrongGPU network backend for the FidelityFX front end (spec 10 sections 1, 6 and 8): the runtime DLL's export
// wgGetUpscalerBackend (clean/fsr/include/wronggpu_upscaler.h). WrongGPU's own code.
//
// Per dispatch (interface version 2), on the game's command list L:
//  1. the input conversion (game color / depth / motion / exposure -> the slot's shared buffers) is recorded into L;
//  2. the front end's split service closes L and runs its first part in the game's own submission order, makes the
//     GPU signal `ready` (the request's done fence) = v after it, and re-opens L so that what follows waits on the
//     GPU until `done` (the request's resume fence) reaches the given value;
//  3. the network is enqueued on the HIP stream right away: GPU-side wait ready >= v, the frame, signal done = v;
//  4. the output write is recorded into the re-opened L.
// "inline" (default) resumes at done = v and writes this frame's result (same frame). "pipelined" (fallback, option
// sync=pipelined) resumes at pipe = v, which the executor signals once the previous network frame is done, and
// writes the previous frame's result, so the game's queue
// does not wait for this frame's network. The CPU never waits for the GPU. The runtime installs no hooks: the
// command-list split, the queue order and the restore of the game's bindings belong to the front end.
// See clean/FORMATS.md "D3D12 interop" for the buffers and fences, and QUESTIONS.md (spec 10) for the open points.
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "../../fsr/include/wronggpu_upscaler.h"
#include "../runtime_core.h"
#include "../embedded.h"
#include "../../k/runtime/model_k.h"
#include "../model_select.h"
#include "convert.h"
#include "executor.h"
#include "executor_select.h"

#include <dxgi1_4.h>

struct WgUpscaler;

namespace wgi {

namespace {

// the converter maps and model K's folder, from the model table (clean/runtime/model_select.h)
const char* const kMapName = wgrt::model_info(wgrt::ModelId::M).map_file;   // preset-M.v1.map.json
const char* const kKMapName = wgrt::model_info(wgrt::ModelId::K).map_file;  // model K (runtime option model=K)
const char* const kKSubdir = wgrt::model_info(wgrt::ModelId::K).model_subdir;  // <model_dir>\K

// Value of `key` in a "key=value;..." string (the last one wins), "" if absent.
std::string option_value(const std::string& text, const std::string& key) {
  std::string v;
  for (size_t a = 0; a <= text.size();) {
    size_t b = text.find(';', a);
    if (b == std::string::npos) b = text.size();
    const std::string kv = text.substr(a, b - a);
    a = b + 1;
    if (kv.compare(0, key.size() + 1, key + "=") == 0) v = kv.substr(key.size() + 1);
  }
  return v;
}

// ---- strings and paths ---------------------------------------------------------------------------------------------

std::wstring widen(const std::string& s) {
  if (s.empty()) return std::wstring();
  const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), nullptr, 0);
  std::wstring w(n, L'\0');
  MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), &w[0], n);
  return w;
}

std::string utf8(const wchar_t* w) {
  const int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
  if (n <= 1) return std::string();
  std::string s(n - 1, '\0');
  WideCharToMultiByte(CP_UTF8, 0, w, -1, &s[0], n, nullptr, nullptr);
  return s;
}

bool exists(const std::string& path) { return GetFileAttributesW(widen(path).c_str()) != INVALID_FILE_ATTRIBUTES; }

// The core runtime opens files with narrow (ANSI code page) paths. A path with characters outside that code page
// (e.g. in the user name) goes through its short 8.3 form when the volume has one.
bool narrow_path(const std::string& path, std::string* out) {
  std::wstring w = widen(path);
  wchar_t shortPath[MAX_PATH * 2];
  const DWORD n = GetShortPathNameW(w.c_str(), shortPath, MAX_PATH * 2);
  if (n > 0 && n < MAX_PATH * 2) w = shortPath;
  BOOL lossy = FALSE;
  const int len = WideCharToMultiByte(CP_ACP, WC_NO_BEST_FIT_CHARS, w.c_str(), -1, nullptr, 0, nullptr, &lossy);
  if (len <= 0 || lossy) return false;
  std::string s(len - 1, '\0');
  WideCharToMultiByte(CP_ACP, WC_NO_BEST_FIT_CHARS, w.c_str(), -1, &s[0], len, nullptr, nullptr);
  *out = s;
  return true;
}

std::string module_dir() {
  HMODULE self = nullptr;
  if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                          reinterpret_cast<LPCWSTR>(&module_dir), &self))
    return ".";
  wchar_t buf[32768];
  const DWORD n = GetModuleFileNameW(self, buf, 32768);
  if (n == 0 || n >= 32768) return ".";
  std::wstring p(buf, n);
  const size_t slash = p.find_last_of(L"\\/");
  return slash == std::wstring::npos ? "." : utf8(p.substr(0, slash).c_str());
}

std::string local_appdata() {
  wchar_t buf[32768];
  const DWORD n = GetEnvironmentVariableW(L"LOCALAPPDATA", buf, 32768);
  return n > 0 && n < 32768 ? utf8(buf) : std::string();
}

// The device's adapter as DXGI reports it (description, vendor, software flag); false when it cannot be read.
bool adapter_info(ID3D12Device* device, std::string* name, uint32_t* vendor_id = nullptr, bool* software = nullptr) {
  typedef HRESULT(WINAPI * PfnCreateFactory)(REFIID, void**);
  HMODULE dxgi = LoadLibraryW(L"dxgi.dll");
  auto create =
      dxgi ? reinterpret_cast<PfnCreateFactory>(reinterpret_cast<void*>(GetProcAddress(dxgi, "CreateDXGIFactory1")))
           : nullptr;
  bool ok = false;
  name->clear();
  IDXGIFactory4* f = nullptr;
  if (create && SUCCEEDED(create(__uuidof(IDXGIFactory4), (void**)&f)) && f) {
    IDXGIAdapter1* a = nullptr;
    if (SUCCEEDED(f->EnumAdapterByLuid(device->GetAdapterLuid(), __uuidof(IDXGIAdapter1), (void**)&a)) && a) {
      DXGI_ADAPTER_DESC1 d = {};
      if (SUCCEEDED(a->GetDesc1(&d))) {
        ok = true;
        *name = utf8(d.Description);
        if (vendor_id) *vendor_id = d.VendorId;
        if (software) *software = (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0;
      }
      a->Release();
    }
    f->Release();
  }
  return ok;
}

std::string adapter_description(ID3D12Device* device) {
  std::string name;
  adapter_info(device, &name);
  return name;
}

// ---- options ("key=value;key=value", from the front end's configuration) --------------------------------------------

struct Options {
  std::string sync = "inline";      // inline | pipelined
  std::string hip_switch;           // on | off (executor_select.h refused_frame_policy); empty: the default
  std::string executor = "auto";    // auto (not named: dx12 for model M on RDNA 4, else hip; executor_select.h) | hip |
                                    // dx12 (the network on the game's D3D12 device, no HIP) | d3d12-test (tests)
  std::string model_dir;            // default %LOCALAPPDATA%\WrongGPU\model (installer contract, spec 30)
  std::string kernels_dir;          // default: the folder of this DLL
  std::string map;                  // default <kernels_dir>\preset-M.v1.map.json, then <model_dir>\...
  std::string rocm_dir;             // HIP SDK folder to try first (development)
  std::string swin;                 // Swin code object (default swin.co)
  std::string model_format = "wgcm";  // wgcm | auto | raw (tests)
  int test_view = 0;                // d3d12-test executor: 0 color, 1 depth, 2 motion
  bool test_handles = false;        // d3d12-test executor: make shared objects with NT handles (sharing paths)
  bool test_keep = false;           // d3d12-test executor with test_handles: keep the arena for later contexts (pool)
  bool timing_stages = true;        // timing_stages=0: keep D3D12 full-span timestamps only
  bool timing_log = false;          // one timing summary line every 300 dispatches
  bool watchdog = false;            // diagnostics: report a dispatch or network frame stuck for more than 3 s
  std::string ready_wait = "fence";  // fence (default: HIP waits on the shared fence) | memory (experimental: HIP
                                     // waits on a flag in the arena; stalls the GPU in several games)
  std::string recon = "spec";       // spec | output (experimental: D = O, output filter = identity)
  std::string motion_outres = "original";  // with WG_CREATE_MOTION_OUTPUT_RES (model M): original = the game's motion
                                           // read at output resolution by the encoder and the reconstruction (specs
                                           // 03 draft 4, 06); point = resampled to render pixels by the input pass
                                           // (our earlier stand-in), for A/B
  std::string done_mode = "fence";  // fence | spin (experimental: the game's queue polls a flag the network writes)
  uint32_t spin_polls = 400000;     // done_mode=spin: polls before the spin gives up (about 0.4 s)
  std::string handover = "fence";   // fence | staged (model K, inline sync; see dispatch_impl and model_k_default)
  bool handover_named = false;      // the options name handover (model K's default mode, model_k_default)
  bool input_copy = false;          // copies of the game's textures + HIP conversion instead of the input pass
  bool output_copy = true;          // RGBA16F result + CopyTextureRegion instead of the output pass (default on;
                                    // output_copy=0: the pass)
  std::string share_path = "auto";  // auto | arena | game | underlying | own (diagnostics and tests)
  std::string capture_dir;          // diagnostics: frame capture folder (off when empty)
  int capture_frames = 30;          // frames per capture
  float sparkle_guard = 0.0f;       // WrongGPU's sparkle guard: brighten rate in (0, 1); 0 = off (default)
  float sparkle_guard_thr = 0.01f;  // its threshold: brightening up to this much of t = Y / (1 + Y) passes freely
  std::string unknown;
  std::string runtime;  // the keys above did not take: "key=value;..." for the network runtime's runtime_options
};

Options parse_options(const char* text) {
  Options o;
  std::string s = text ? text : "";
  size_t pos = 0;
  while (pos <= s.size()) {
    size_t end = s.find(';', pos);
    if (end == std::string::npos) end = s.size();
    std::string item = s.substr(pos, end - pos);
    pos = end + 1;
    const size_t eq = item.find('=');
    auto trim = [](std::string t) {
      const size_t a = t.find_first_not_of(" \t"), b = t.find_last_not_of(" \t");
      return a == std::string::npos ? std::string() : t.substr(a, b - a + 1);
    };
    const std::string key = trim(item.substr(0, eq)), value = eq == std::string::npos ? "" : trim(item.substr(eq + 1));
    if (key.empty()) continue;
    if (key == "sync") o.sync = value;
    else if (key == "executor" || key == "backend") o.executor = value == "d3d12" ? "dx12" : value;
    else if (key == "hip_switch") o.hip_switch = value;
    else if (key == "model_dir") o.model_dir = value;
    else if (key == "kernels_dir") o.kernels_dir = value;
    else if (key == "map") o.map = value;
    else if (key == "rocm_dir") o.rocm_dir = value;
    else if (key == "swin") o.swin = value;
    else if (key == "model_format") o.model_format = value;
    else if (key == "test_handles") o.test_handles = value == "1";
    else if (key == "test_keep") o.test_keep = value == "1";
    else if (key == "share_path") o.share_path = value;
    else if (key == "timing_log") o.timing_log = value == "1";
    else if (key == "timing_stages") o.timing_stages = value != "0";
    else if (key == "watchdog") o.watchdog = value == "1";
    else if (key == "ready_wait") o.ready_wait = value;
    else if (key == "capture_dir") o.capture_dir = value;
    else if (key == "capture_frames") o.capture_frames = std::atoi(value.c_str());
    else if (key == "sparkle_guard") o.sparkle_guard = value == "on" ? 0.5f : (float)std::atof(value.c_str());
    else if (key == "sparkle_guard_thr") o.sparkle_guard_thr = (float)std::atof(value.c_str());
    else if (key == "recon") o.recon = value;
    else if (key == "motion_outres") o.motion_outres = value;
    else if (key == "done_mode") o.done_mode = value;
    else if (key == "spin_polls") o.spin_polls = (uint32_t)std::strtoul(value.c_str(), nullptr, 10);
    else if (key == "handover") o.handover = value, o.handover_named = true;
    else if (key == "input_copy") o.input_copy = value == "1";
    else if (key == "output_copy") o.output_copy = value == "1";
    else if (key == "test_view") o.test_view = value == "depth" ? 1 : value == "motion" ? 2 : 0;
    else {
      o.unknown += (o.unknown.empty() ? "" : ", ") + key;
      o.runtime += (o.runtime.empty() ? "" : ";") + key + (eq == std::string::npos ? "" : "=" + value);
    }
  }
  return o;
}

// Model K's default mode, in this one place for every way model K is selected (wronggpu.ini model=K from the
// installer's Preset K, the front end's ";model=K" for the OptiScaler plugin's or the host's Model K entry,
// WG_RUNTIME_OPTIONS). 0.5.1 (hotfix 2026-10-14): the exact path, network=fused;k_fast=0;handover=staged (the staged
// hand-over changes when the game's queue resumes, not the output bits). 0.5.0 made the fastest mode the default
// (network=fused8;k_fast=1;handover=staged, about 64 dB from the exact output); players saw it shimmer, so it is
// opt-in again: name it in the options. The default applies only when the options name none of network, k_fast and
// handover; naming any of them selects the mode explicitly and the keys not named keep their base values (network=
// fused, k_fast=0, handover=fence), so "network=fused8;k_fast=1;handover=staged", "network=fused16" or
// "network=fused;k_fast=0;handover=fence" select those modes. handover=staged only with inline sync (it needs it).
// Model M and the test executor: nothing changes. Returns whether the default was applied (o gains the keys).
bool model_k_default(Options& o) {
  if (o.executor != "hip" || o.handover_named || !option_value(o.runtime, "network").empty() ||
      !option_value(o.runtime, "k_fast").empty())
    return false;
#ifdef WG_NO_MODEL_K
  return false;
#else
  WgRtConfig probe = WgRtConfig();
  probe.runtime_options = o.runtime.c_str();
  wgrt::ModelId model = wgrt::ModelId::M;
  std::string err;
  if (!wgrt::model_from_options(probe, &model, &err) || model != wgrt::ModelId::K) return false;
  o.runtime += (o.runtime.empty() ? "" : ";") + std::string("network=fused;k_fast=0");
  if (o.sync == "inline") o.handover = "staged";
  return true;
#endif
}

template <class T>
void release(T*& p) {
  if (p) p->Release();
  p = nullptr;
}

std::string hr_text(HRESULT hr) {
  char b[32];
  std::snprintf(b, sizeof b, "0x%08lX", (unsigned long)hr);
  return b;
}

// ---- arena pool (process-wide) --------------------------------------------------------------------------------------
//
// The arena of a context whose executor imports it (HIP) is not released with the context: the HIP runtime keeps
// imported memory until the process ends, whatever is called on it (hipFree of the mapped buffer and
// hipDestroyExternalMemory succeed and change nothing; measured 2026-10-13, clean/FORMATS.md "Context teardown"), so
// every released arena was lost, about 100 MiB per context at 1080p. The arena and its import go back to this pool
// instead and serve the next context on the same device and import scope that fits into it. On a miss the free arenas
// that could not serve (too small, another device, another scope) are given up, and the new, larger one takes their
// place: what the process loses is bounded by the arenas it outgrows, not by the number of contexts.
// An arena is told to belong to "the same device" by the device its objects report (ID3D12DeviceChild::GetDevice of an
// object made through the device given at create): the given device itself, or the one under a wrapper that hands out
// plain objects, which stays the same when such a wrapper is made anew for a context. The pool holds one reference on
// the arena and on that device (the pointer stays unique while pooled).
struct PooledArena {
  ID3D12Device* device = nullptr;
  ID3D12Resource* resource = nullptr;
  std::string scope;  // the importing executor's import scope
  uint64_t bytes = 0, alloc_bytes = 0;
  ArenaImport import;
  bool in_use = false;
};

class ArenaPool {
 public:
  static ArenaPool& get() {
    static ArenaPool* pool = new ArenaPool();  // never destroyed: nothing of it may run at unload or process exit
    return *pool;
  }
  // A free arena of this device and scope with at least `need` bytes (the smallest), now in use. Without one: null,
  // and every free arena is handed to the caller in `unfit`, removed from the pool.
  PooledArena* acquire(ID3D12Device* device, const std::string& scope, uint64_t need,
                       std::vector<std::unique_ptr<PooledArena>>* unfit) {
    std::lock_guard<std::mutex> lock(mutex_);
    PooledArena* best = nullptr;
    for (auto& a : all_)
      if (!a->in_use && a->device == device && a->scope == scope && a->bytes >= need && (!best || a->bytes < best->bytes))
        best = a.get();
    if (best) {
      best->in_use = true;
      return best;
    }
    for (size_t i = 0; i < all_.size();) {
      if (all_[i]->in_use) {
        ++i;
        continue;
      }
      unfit->push_back(std::move(all_[i]));
      all_.erase(all_.begin() + (std::ptrdiff_t)i);
    }
    return nullptr;
  }
  // A new arena, in use by the caller.
  PooledArena* add(ID3D12Device* device, const std::string& scope, ID3D12Resource* resource, uint64_t bytes,
                   uint64_t alloc_bytes) {
    std::unique_ptr<PooledArena> a(new PooledArena());
    a->device = device;
    a->resource = resource;
    device->AddRef();
    resource->AddRef();
    a->scope = scope;
    a->bytes = bytes;
    a->alloc_bytes = alloc_bytes;
    a->in_use = true;
    std::lock_guard<std::mutex> lock(mutex_);
    all_.push_back(std::move(a));
    return all_.back().get();
  }
  // The context is done with the arena. With an import it stays for the next context (true); without one (the
  // context failed before the import) it leaves the pool and is released.
  bool release(PooledArena* arena) {
    std::unique_ptr<PooledArena> gone;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (arena->import.mem) {
        arena->in_use = false;
        return true;
      }
      for (size_t i = 0; i < all_.size(); ++i)
        if (all_[i].get() == arena) {
          gone = std::move(all_[i]);
          all_.erase(all_.begin() + (std::ptrdiff_t)i);
          break;
        }
    }
    if (gone) free_arena(gone.get());
    return false;
  }
  static void free_arena(PooledArena* a) {
    if (a->resource) a->resource->Release();
    if (a->device) a->device->Release();
    a->resource = nullptr;
    a->device = nullptr;
  }

 private:
  std::mutex mutex_;
  std::vector<std::unique_ptr<PooledArena>> all_;
};

// ---- state shared with the ExecuteCommandLists callbacks (they may run after the context is gone) ------------------

struct Core {
  WgLogFn log = nullptr;
  void* log_user = nullptr;
  void say(int level, const std::string& m) {
    if (log) log(log_user, level, m.c_str());
  }
};

// ---- timing log (option timing_log=1; nothing of this runs when it is off) ------------------------------------------

struct Stat {
  uint64_t n = 0;
  double sum = 0, max = 0;
  void add(double x) {
    ++n;
    sum += x;
    if (x > max) max = x;
  }
  std::string text() const {
    if (!n) return "n/a";
    char b[64];
    std::snprintf(b, sizeof b, "%.3f (max %.3f)", sum / n, max);
    return b;
  }
  double avg() const { return n ? sum / n : 0; }
};

// Per frame: D3D12 timestamps in the game's list (0 before the input pass, 1 after it, both in the first part; 2 at
// the start of the second part, i.e. when the resume wait is over; 3 after the output pass), resolved into a readback
// ring; the network's GPU times from the executor; CPU times of the dispatch call. Read back without waiting, once
// the `ready` fence shows that a later frame's first part has run (so this frame's list has completed). Every 300
// dispatches one summary line.
class TimingLog {
 public:
  static constexpr int kRing = kTimingRing, kPerFrame = 4, kEvery = 300;

  ~TimingLog() {
    if (readback_) readback_->Unmap(0, nullptr);
    if (readback_) readback_->Release();
    if (heap_) heap_->Release();
  }

  bool init(ID3D12Device* d, std::string* err) {
    D3D12_QUERY_HEAP_DESC qd = {};
    qd.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    qd.Count = kRing * kPerFrame;
    HRESULT hr = d->CreateQueryHeap(&qd, __uuidof(ID3D12QueryHeap), (void**)&heap_);
    if (FAILED(hr)) return *err = "timing log: CreateQueryHeap " + hr_text(hr), false;
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = 8 * kRing * kPerFrame;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    hr = d->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                    __uuidof(ID3D12Resource), (void**)&readback_);
    if (FAILED(hr)) return *err = "timing log: readback buffer " + hr_text(hr), false;
    if (FAILED(readback_->Map(0, nullptr, (void**)&ticks_))) return *err = "timing log: Map failed", false;
    LARGE_INTEGER hz;
    QueryPerformanceFrequency(&hz);
    cpu_hz_ = (double)hz.QuadPart;
    return true;
  }

  static int64_t now() {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return t.QuadPart;
  }
  double ms(int64_t a, int64_t b) const { return 1000.0 * (double)(b - a) / cpu_hz_; }

  void stamp(ID3D12GraphicsCommandList* l, uint64_t v, int k) {
    l->EndQuery(heap_, D3D12_QUERY_TYPE_TIMESTAMP, base(v) + k);
  }
  void resolve(ID3D12GraphicsCommandList* l, uint64_t v) {
    l->ResolveQueryData(heap_, D3D12_QUERY_TYPE_TIMESTAMP, base(v), kPerFrame, readback_, 8ull * base(v));
  }
  void set_info(const std::string& s) { info_ = s; }
  // Names of the four network parts (the executor's timing marks): model M enc0, Swin, dec0, post; model K exposure,
  // input, network, output.
  void set_labels(const std::vector<std::string>& l) {
    for (size_t k = 0; k < 4 && k < l.size(); ++k) labels_[k] = l[k];
  }
  void set_frequency(uint64_t hz) {
    if (hz) gpu_hz_ = (double)hz;
  }

  // One dispatch is over: its CPU times, and whether its timestamps and network times will exist.
  void frame(uint64_t v, double cpu_ms, double split_ms, double submit_ms, bool gpu, bool network) {
    st_.cpu.add(cpu_ms);
    if (split_ms >= 0) st_.split.add(split_ms);
    if (submit_ms >= 0) st_.submit.add(submit_ms);
    Entry& e = ring_[v % kRing];
    e.value = gpu ? v : 0;
    e.network = network;
    ++st_.frames;
  }

  // Reads every finished frame (never waits) and writes the summary line every kEvery dispatches.
  void harvest(uint64_t ready_done, Executor* exec, const std::function<void(const std::string&)>& say) {
    for (Entry& e : ring_) {
      if (!e.value || e.value + 1 > ready_done) continue;
      const uint64_t* t = ticks_ + base(e.value);
      if (gpu_hz_ > 0 && t[1] >= t[0] && t[3] >= t[2]) {
        const double tick = 1000.0 / gpu_hz_;
        st_.in.add((double)(t[1] - t[0]) * tick);
        st_.out.add((double)(t[3] - t[2]) * tick);
        st_.tot.add((double)(t[3] - t[0]) * tick);
        const double span = t[2] >= t[1] ? (double)(t[2] - t[1]) * tick : -1;  // first part's queue -> game queue
        if (span >= 0) {
          st_.span.add(span);
          if (span > 0.1) ++st_.waits;
          ++st_.span_frames;
        }
        NetTiming n;
        if (e.network && exec->timing(e.value, &n)) {
          st_.net.add(n.network);
          st_.enc0.add(n.enc0);
          st_.swin.add(n.swin);
          st_.dec0.add(n.dec0);
          st_.post.add(n.post);
          if (span >= 0) st_.over.add(span - n.network);
        }
      }
      e.value = 0;
    }
    if (st_.frames < kEvery) return;
    char b[160];
    std::snprintf(b, sizeof b, "timing, last %llu dispatches: ", (unsigned long long)st_.frames);
    std::string m = b;
    m += "network GPU " + st_.net.text() + " ms [" + labels_[0] + " " + fmt(st_.enc0) + ", " + labels_[1] + " " +
         fmt(st_.swin) + ", " + labels_[2] + " " + fmt(st_.dec0) + ", " + labels_[3] + " " + fmt(st_.post) + "] (" +
         std::to_string(st_.net.n) + " frames); ";
    m += "TOTAL(t3-t0) " + st_.tot.text() + " ms; ";
    m += "D3D12 input pass " + st_.in.text() + " ms, output pass " + st_.out.text() + " ms; ";
    m += "input pass end -> resume " + st_.span.text() + " ms, > 0.1 ms in " + std::to_string(st_.waits) + " of " +
         std::to_string(st_.span_frames) + " frames, of it not network " + st_.over.text() + " ms; ";
    m += "CPU dispatch " + st_.cpu.text() + " ms (split " + st_.split.text() + ", network enqueue " + st_.submit.text() + ")";
    if (!info_.empty()) {  // once, in the first summary: what the stage times are
      m += "; stage events: " + info_;
      info_.clear();
    }
    say(m);
    st_ = Stats();
  }

 private:
  struct Entry {
    uint64_t value = 0;  // frame whose timestamps are in this ring entry (0 = none)
    bool network = false;
  };
  struct Stats {
    Stat net, enc0, swin, dec0, post, in, out, span, over, cpu, split, submit, tot;
    uint64_t waits = 0, span_frames = 0, frames = 0;
  };
  static std::string fmt(const Stat& s) {
    char b[32];
    std::snprintf(b, sizeof b, "%.3f", s.avg());
    return b;
  }
  static int base(uint64_t v) { return (int)(v % kRing) * kPerFrame; }

  ID3D12QueryHeap* heap_ = nullptr;
  ID3D12Resource* readback_ = nullptr;
  uint64_t* ticks_ = nullptr;
  double cpu_hz_ = 1, gpu_hz_ = 0;
  Entry ring_[kRing];
  Stats st_;
  std::string info_;
  std::string labels_[4] = {"enc0", "Swin", "dec0", "post"};
};

// ---- the context ---------------------------------------------------------------------------------------------------

class Context {
 public:
  ~Context() { shutdown(); }

  WgResult create(const WgCreateInfo& info) {
    log_.log = info.log;
    log_.log_user = info.log_user;
    flags_ = info.flags;
    max_rw_ = info.max_render_width;
    max_rh_ = info.max_render_height;
    ow_ = info.output_width;
    oh_ = info.output_height;
    if (!info.d3d12_device || !max_rw_ || !max_rh_ || !ow_ || !oh_) return fail(WG_ERROR_PARAMETER, "invalid create info");
    services_ = info.services;
    if (FAILED(static_cast<IUnknown*>(info.d3d12_device)->QueryInterface(__uuidof(ID3D12Device), (void**)&device_)))
      return fail(WG_ERROR_PARAMETER, "d3d12_device is not an ID3D12Device");

    // Diagnostics override: WG_RUNTIME_OPTIONS (same "key=value;..." syntax) is parsed after the front end's options,
    // so its keys win (e.g. timing_log=1 while the front end cannot forward runtime keys).
    std::string text = info.options ? info.options : "";
    char env[4096] = {0};
    const DWORD n = GetEnvironmentVariableA("WG_RUNTIME_OPTIONS", env, sizeof env);
    if (n > 0 && n < sizeof env) {
      text += std::string(";") + env;
      say(WG_LOG_INFO, std::string("options from WG_RUNTIME_OPTIONS: ") + env);
    } else if (n >= sizeof env) {
      say(WG_LOG_WARN, "WG_RUNTIME_OPTIONS is longer than 4095 characters and is ignored");
    }
    Options o = parse_options(text.c_str());
    {  // the executor (executor_select.h): the option, else the default (dx12 for model M on RDNA 4); one log line why
      ExecutorRequest q;
      q.executor = o.executor;
      WgRtConfig probe = WgRtConfig();
      probe.runtime_options = o.runtime.c_str();
      wgrt::ModelId model = wgrt::ModelId::M;
      std::string merr;
      q.model_k = wgrt::model_from_options(probe, &model, &merr) && model != wgrt::ModelId::M;
      q.adapter_known = adapter_info(device_, &q.adapter, &q.vendor_id, &q.software);
      q.sparkle_guard = o.sparkle_guard > 0.0f;
      q.capture = !o.capture_dir.empty();
      const ExecutorChoice ch = choose_executor(q);
      if (ch.executor.empty()) return fail(WG_ERROR_PARAMETER, "unknown executor " + o.executor);
      if (!ch.why.empty()) say(ch.warn ? WG_LOG_WARN : WG_LOG_INFO, ch.why);
      o.executor = ch.executor;  // model K's default mode below applies with the HIP executor
      dx12_default_ = ch.by_default && ch.executor == "dx12";
      const RefusedFramePolicy pol = refused_frame_policy(o.hip_switch);  // clean/fe-wrapped-list
      hip_switch_ = pol.switch_to_hip;
      if (!pol.note.empty()) say(WG_LOG_WARN, pol.note);
    }
    k_default_ = model_k_default(o);
    const bool split = services_ && services_->struct_size >= sizeof(WgFrontEndServices) && services_->split_command_list;
    split_ = split;
    share_path_opt_ = o.share_path;
    input_copy_opt_ = o.input_copy;
    if (!split && o.executor != "dx12")
      return fail(WG_ERROR_UNSUPPORTED, "the front end offers no command-list split (interface version 2)");
    if (!o.unknown.empty()) say(WG_LOG_INFO, "options passed on to the network runtime: " + o.unknown);
    if (o.motion_outres != "original" && o.motion_outres != "point")
      return fail(WG_ERROR_PARAMETER, "motion_outres must be original or point");
    if (o.sync != "inline" && o.sync != "pipelined") return fail(WG_ERROR_PARAMETER, "sync must be inline or pipelined");
    inline_ = o.sync == "inline";

    std::string err;
    if (!conv_.init(device_, &err)) return fail(WG_ERROR_DEVICE, err);
    bool dx12 = false;
    if (o.executor == "d3d12-test") {
      exec_ = make_test_executor(o.test_view, o.test_handles, o.test_keep);
    } else if (o.executor == "hip" || o.executor == "dx12") {
      if (!configure_hip(o, &err)) return fail(WG_ERROR, err);
    } else {
      return fail(WG_ERROR_PARAMETER, "unknown executor " + o.executor);
    }
    // Motion at output resolution (model M): the network reads the game's motion itself (motion_outres=original), so
    // each slot gets the raw output-size motion next to the render-size one; with motion_outres=point only while a
    // frame capture is armed (so that a capture can be replayed through both paths).
    motion_hi_use_ = (flags_ & WG_CREATE_MOTION_OUTPUT_RES) && !model_k_ && o.motion_outres == "original";
    motion_hi_ = (flags_ & WG_CREATE_MOTION_OUTPUT_RES) && !model_k_ && (motion_hi_use_ || !o.capture_dir.empty());
    // executor dx12 (option, or the default for model M on RDNA 4): the network on this D3D12 device, recorded into
    // the game's command list. Paths without D3D12 ports, and the diagnostics that need HIP, got the HIP executor
    // above (choose_executor). (Motion at output resolution: the D3D12 ports of the emo / pmo kernels read the game's
    // motion texture with motion_outres=original; with point the executor's D3D12 pass resamples it.)
    dx12 = o.executor == "dx12";
    if (dx12) {
      exec_ = make_d3d12_executor(rt_cfg_);
      exec_->set_log_sink([this](bool warning, const std::string& m) {
        say(warning ? WG_LOG_WARN : WG_LOG_INFO, "D3D12 executor: " + m);
      });
      no_slots_ = true;  // its kernels read the game's textures: no slot buffers (fences and a 128 KB flag arena only)
      if (!exec_->prepare(&err) || !exec_->prepare_device(device_, &err) ||
          !make_shared_objects(false, "arena", &err) || !exec_->init(device_, set_, &err)) {
        say(WG_LOG_WARN, dx12_refused_line(dx12_default_, err));
        exec_.reset();
        release_set();
        no_slots_ = false;
        dx12 = false;
        dx12_default_ = false;
        err.clear();
      } else {
        say(WG_LOG_INFO, std::string("executor dx12: the network runs on the game's D3D12 device, recorded into the "
                                     "game's command list; its kernels read the game's colour, depth and motion and "
                                     "write the output texture (no input or output conversion pass, no HIP, no "
                                     "command-list split, no shared fences)") +
                             (dx12_default_ ? "; the default on RDNA 4 (executor=hip selects the HIP executor)" : ""));
      }
    }
    if (!dx12 && !exec_) {
      if (!split) return fail(WG_ERROR_UNSUPPORTED, "the front end offers no command-list split (interface version 2)");
      exec_ = make_hip_executor(rt_cfg_, adapter_);
    }
    if (!dx12) {
      // what the executor can only report (context teardown): failed device calls as warnings, the rest as information
      exec_->set_log_sink([this](bool warning, const std::string& m) {
        say(warning ? WG_LOG_WARN : WG_LOG_INFO, "destroy: " + m);
      });
      say(WG_LOG_INFO, "about to start the HIP runtime (HIP executor)");  // flushed by the front end (clean/fe-wrapped-list)
      if (!exec_->prepare(&err)) return fail(WG_ERROR, err);
      if (!make_shared_objects(exec_->needs_handles(), o.share_path, &err)) return fail(WG_ERROR_DEVICE, err);
    }
    say(WG_LOG_INFO, "shared buffers and fences: " + share_path_);
    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    if (FAILED(device_->CreateCommandQueue(&qd, __uuidof(ID3D12CommandQueue), (void**)&rescue_)))
      return fail(WG_ERROR_DEVICE, "CreateCommandQueue failed");
    if (!dx12) say(WG_LOG_INFO, "about to import the shared buffers and fences into HIP (" + share_path_ + ")");
    if (!dx12 && !exec_->init(device_, set_, &err)) return fail(WG_ERROR, err);
    close_handles();  // the importer holds its own references now
    input_copy_ = o.input_copy && !dx12;
    output_copy_ = o.output_copy;
    if (dx12 && o.input_copy) say(WG_LOG_INFO, "input_copy is not used by executor=dx12 (its kernels read the textures)");
    if (dx12 && !inline_) say(WG_LOG_INFO, "sync=pipelined is not used by executor=dx12 (the network runs in the list)");
    if (input_copy_ || output_copy_)
      say(WG_LOG_INFO, std::string("copy paths requested:") + (input_copy_ ? " input_copy" : "") +
                           (output_copy_ ? " output_copy" : "") + " (each frame falls back to the pass if the formats do "
                                                                   "not allow a copy; path changes are logged)");
    if (o.ready_wait == "memory" && input_copy_) {
      say(WG_LOG_WARN, "ready_wait=memory is not combined with input_copy; using the fence");
    } else if (o.ready_wait == "memory") {
      ready_flag_ = set_.flag.resource && exec_->use_ready_flag();
      say(ready_flag_ ? WG_LOG_INFO : WG_LOG_WARN,
          ready_flag_ ? "ready_wait=memory (experimental): the network waits for a flag in the arena, not the fence"
                      : "ready_wait=memory not available (no arena or no hipStreamWaitValue32); using the fence");
    }
    if (o.done_mode == "spin") {
      // inline mode with the memory flag only (the flag block sits next to the ready flag); the rescue path writes
      // the flag itself when the network cannot be enqueued
      done_spin_ = ready_flag_ && inline_ && make_rescue_list() && exec_->use_done_flag();
      spin_polls_ = o.spin_polls ? o.spin_polls : 1;
      say(WG_LOG_INFO, done_spin_ ? "done_mode=spin (experimental): the game's queue polls a flag the network writes, "
                                    "not the done fence"
                                  : "done_mode=spin not available (needs ready_wait=memory, inline sync); using the fence");
    }
    if (o.handover != "fence") {
      // Model K's default mode (model_k_default) or the option: the output-side hand-over overlaps the output stage
      // (see dispatch_impl).
      if (o.handover != "staged") return fail(WG_ERROR_PARAMETER, "handover must be fence or staged");
      const bool can = inline_ && model_k_ && !done_spin_ && set_.flag.resource && make_rescue_list() &&
                       exec_->use_staged_output();
      if (can) {
        staged_out_ = done_spin_ = true;  // done_spin_: the second part starts with the bounded spin on the done flag
        spin_polls_ = o.spin_polls ? o.spin_polls : 1;
      }
      say(can ? WG_LOG_INFO : WG_LOG_WARN,
          can ? "handover=staged: the game's queue waits until the network starts its output stage, then "
                "polls the done flag (at most " + std::to_string(spin_polls_) + " polls)"
              : std::string("handover=staged not available (needs model K, inline sync, the arena, no done_mode=spin, "
                            "and the HIP executor); using the done fence"));
    }
    if (o.watchdog) start_watchdog();
    if (o.sparkle_guard > 0.0f && o.sparkle_guard < 1.0f && o.sparkle_guard_thr >= 0.0f && o.sparkle_guard_thr < 1.0f) {
      const bool on = exec_->set_sparkle_guard(o.sparkle_guard, o.sparkle_guard_thr);
      say(WG_LOG_INFO, on ? "sparkle guard on: brighten rate " + std::to_string(o.sparkle_guard) + " above threshold " +
                                std::to_string(o.sparkle_guard_thr) +
                                " (motion-compensated, disocclusion by depth; WrongGPU's own addition)"
                          : std::string("sparkle guard not available with this executor"));
    }
    if (!o.capture_dir.empty()) {
      exec_->set_capture(o.capture_dir, o.capture_frames > 0 ? o.capture_frames : 30);
      say(WG_LOG_INFO, "frame capture armed: create " + o.capture_dir + "\\capture.go to write the next " +
                           std::to_string(o.capture_frames) + " network frames there (diagnostics; stalls the GPU)");
    }
    if (o.timing_log) {
      timing_.reset(new TimingLog());
      if (!timing_->init(device_, &err)) return fail(WG_ERROR_DEVICE, err);
      UINT64 hz = 0;
      if (SUCCEEDED(rescue_->GetTimestampFrequency(&hz))) timing_->set_frequency(hz);
      const bool net = o.timing_stages && exec_->enable_timing();
      if (!o.timing_stages) say(WG_LOG_INFO, "HIP kernel events disabled by timing_stages=0; D3D12 full-span timing remains active");
      timing_->set_info(net ? exec_->timing_info() : std::string());
      timing_->set_labels(exec_->timing_labels());
      say(WG_LOG_INFO, std::string("timing log on (every 300 dispatches); network GPU times ") +
                           (net ? "from the executor: " + exec_->timing_info() : "not available from this executor"));
    }

    char sizes[160];
    std::snprintf(sizes, sizeof sizes, "max render %ux%u, output %ux%u, flags 0x%X", max_rw_, max_rh_, ow_, oh_, flags_);
    say(WG_LOG_INFO, std::string("network runtime ready: executor ") + exec_->name() + ", sync " + o.sync + ", " +
                         sizes + ", GPU memory " + std::to_string(gpu_memory() >> 20) + " MiB" +
                         (exec_->describe().empty() ? "" : ", " + exec_->describe()) +
                         (model_path_.empty() ? "" : ", model " + model_path_));
    // the runtime's own notes of create, one log line each (e.g. the output filter in use when the model file has
    // no output filter weights; spec 06 draft 4)
    for (std::string notes = exec_->notes(); !notes.empty();) {
      const size_t nl = notes.find('\n');
      say(WG_LOG_INFO, "network " + notes.substr(0, nl));
      notes = nl == std::string::npos ? std::string() : notes.substr(nl + 1);
    }
    if (model_k_) {
      say(WG_LOG_INFO, std::string("network model K (runtime option model=K), ") +
                           ((flags_ & WG_CREATE_HDR) ? "HDR" : "LDR") + " colour; output alpha 1" +
                           ((flags_ & WG_CREATE_MOTION_OUTPUT_RES)
                                ? "; motion at output size is resampled to render pixels by the input pass "
                                  "(model K runs its render-resolution motion variant on it)"
                                : ""));
      const std::string net = option_value(n_rtopts_, "network"), fast = option_value(n_rtopts_, "k_fast");
      const bool exact = (net.empty() || net == "fused") && fast != "1";
      say(WG_LOG_INFO, std::string("network model K mode: network=") + (net.empty() ? "fused" : net) + ";k_fast=" +
                           (fast.empty() ? "0" : fast) + ";handover=" + (staged_out_ ? "staged" : "fence") +
                           (exact ? ", the exact path" : ", an approximation (not bit-exact)") +
                           (k_default_ ? " (the default mode; network=fused8;k_fast=1;handover=staged selects the "
                                         "fastest mode)"
                                       : " (from the options)"));
    } else {
      WgRtConfig probe = WgRtConfig();
      probe.recon_output = o.recon == "output" ? 1 : 0;
      const bool out = wgrt::recon_output_requested(probe);
      const uint32_t dw = out ? ow_ : (3 * ow_ + 1) / 2, dh = out ? oh_ : (3 * oh_ + 1) / 2;
      say(WG_LOG_INFO, "reconstruction size: D " + std::to_string(dw) + "x" + std::to_string(dh) +
                           (!out ? " (spec 01)"
                                 : o.recon == "output" ? " (option recon=output)" : " (WG_RECON_SIZE=output)"));
    }
    if ((flags_ & WG_CREATE_MOTION_OUTPUT_RES) && !model_k_ && dx12)
      say(WG_LOG_INFO, motion_hi_use_
                           ? "motion at output resolution: the network's D3D12 kernels read the game's motion texture "
                             "themselves, one texel per render pixel in the input encoder and one per reconstruction "
                             "pixel in the reconstruction (specs 03, 06; option motion_outres=point: the earlier "
                             "resampling to render pixels)"
                           : "motion at output resolution: resampled to render pixels by a D3D12 pass of the executor "
                             "(option motion_outres=point)");
    else if ((flags_ & WG_CREATE_MOTION_OUTPUT_RES) && !model_k_)
      say(WG_LOG_INFO, motion_hi_use_
                           ? "motion at output resolution: the network reads the game's motion texture itself, one "
                             "texel per render pixel in the input encoder and one per reconstruction pixel in the "
                             "reconstruction (specs 03, 06; option motion_outres=point: the earlier resampling to "
                             "render pixels); " + std::to_string((2ull * ow_ * oh_ * 4) >> 20) + " MiB of the shared "
                             "buffers hold it"
                           : "motion at output resolution: resampled to render pixels by the input pass (option "
                             "motion_outres=point)");
    if (flags_ & WG_CREATE_NONLINEAR_COLOR)
      say(WG_LOG_WARN, "the game reports non-linear color; the network expects linear color (spec 20)");
    return WG_OK;
  }

  WgResult dispatch(const WgFrame& f) {
    phase(1);
    const WgResult r = dispatch_timed(f);
    phase(0);
    return r;
  }

  WgResult dispatch_timed(const WgFrame& f) {
    if (!timing_) return dispatch_impl(f);
    const int64_t t0 = TimingLog::now();
    if (f.struct_size >= sizeof(WgFrame) && f.command_queue && !timing_hz_) {
      UINT64 hz = 0;
      if (SUCCEEDED(static_cast<ID3D12CommandQueue*>(f.command_queue)->GetTimestampFrequency(&hz)) && hz)
        timing_->set_frequency(timing_hz_ = hz);
    }
    t_split_ = t_submit_ = -1;
    t_gpu_ = t_net_ = false;
    const WgResult r = dispatch_impl(f);
    timing_->frame(value_, timing_->ms(t0, TimingLog::now()), t_split_, t_submit_, t_gpu_, t_net_);
    // frame v's list has completed: the ready fence at v + 1 (a later frame's first part ran), or (executor dx12)
    // the executor's completion marker at v
    const uint64_t done = exec_->records_into_list() ? exec_->completed() + 1 : set_.ready.fence->GetCompletedValue();
    timing_->harvest(done, exec_.get(), [this](const std::string& m) { say(WG_LOG_INFO, m); });
    return r;
  }

  WgResult dispatch_impl(const WgFrame& f) {
    std::string why;
    if (!valid(f, &why)) return fail_frame(WG_ERROR_PARAMETER, why);
    if (failed_) return WG_ERROR;
    // bug hunt D: an output texture without unordered access can only be written by the RGBA16F copy route
    // (output_copy); every other route makes an unordered access view of it (the device is removed). Refused before
    // anything is recorded, for what no later frame changes (WG_ERROR_INPUT: the host upscales instead).
    why = output_refusal(f);
    if (!why.empty()) return fail_frame(WG_ERROR_INPUT, "input refused: " + why);
    {  // clean/fe-minimize: a frame whose textures do not hold its regions is passed through, never an error
      const std::string skip = skip_reason(f);
      if (!skip.empty()) return skip_frame(f, skip);
      if (skipped_run_) {
        say(WG_LOG_INFO, "frame " + std::to_string(value_ + 1) + ": the game's textures hold the frame again after " +
                             std::to_string(skipped_run_) + " passed-through frame(s); the network starts over");
        skipped_run_ = 0;
      }
    }
    auto* list = static_cast<ID3D12GraphicsCommandList*>(f.command_list);
    if (!device_checked_) {  // once: which device the game's textures belong to (diagnostics for the sharing paths)
      device_checked_ = true;
      ID3D12Device* d = nullptr;
      if (SUCCEEDED(static_cast<ID3D12Resource*>(f.color.resource)->GetDevice(__uuidof(ID3D12Device), (void**)&d)) && d) {
        say(WG_LOG_INFO, std::string("the game's color texture belongs to ") +
                             (d == device_ ? "the device given at create"
                                           : d == underlying_ ? "the underlying device" : "another device object") +
                             "; shared objects: " + share_path_);
        d->Release();
      }
    }
    const uint64_t v = ++value_;
    // the D3D12 executor after a switch to HIP (switch_to_hip): released once the game's lists with its frames are done
    if (retired_ && retired_->completed() >= retired_last_) retired_.reset();
    if (exec_->records_into_list()) {
      bool moved = false;
      const WgResult r = dispatch_record(f, list, v, &moved);
      if (!moved) return r;
      // moved: the HIP executor runs this frame (below) and every later one
    }
    // Slots alternate with every network frame enqueued (not with v: a frame without the network must not make two
    // consecutive network frames share a slot). The slot written now was last used two network frames ago, and the
    // game's queue has waited for that frame (inline) or a later one (pipelined) before this frame's first part.
    const int s = next_slot_;
    const SharedSlot& slot = set_.slot[s];
    Job job;
    job.slot = s;
    job.value = v;
    job.frame.rw = (int)f.render_width;
    job.frame.rh = (int)f.render_height;
    job.frame.mv_scale[0] = job.frame.mv_scale[1] = 1.0f;  // the input pass already applied motion_scale / offset
    job.frame.jitter[0] = f.jitter[0];
    job.frame.jitter[1] = f.jitter[1];
    job.frame.exposure = f.exposure_value;
    job.frame.pre_exposure = f.pre_exposure;
    // reset = 2 (wronggpu_upscaler.h): the front end gave this upscaler to a new context of the game; the runtime
    // starts over with this frame, as newly created (models M and K)
    job.frame.reset = f.reset >= 2 ? 2 : (f.reset || force_reset_) ? 1 : 0;
    job.pipelined = !inline_;
    job.exposure_mode = f.exposure_mode == WG_EXPOSURE_COMPUTE                       ? 2
                        : (f.exposure_mode == WG_EXPOSURE_TEXTURE && f.exposure.resource) ? 1
                                                                                          : 0;
    if (f.exposure_mode == WG_EXPOSURE_TEXTURE && !f.exposure.resource) job.frame.exposure = 1.0f;

    // 1. inputs into the slot
    InputBuffers in;
    in.color = {slot.color.resource, slot.color.offset};
    in.depth = {slot.depth.resource, slot.depth.offset};
    in.motion = {slot.motion.resource, slot.motion.offset};
    in.expo = {slot.expo.resource, slot.expo.offset};
    if (timing_) timing_->stamp(list, v, 0);
    bool copied = false;
    if (input_copy_) {
      Converter::CopyLayout cl;
      std::string why;
      copied = conv_.record_input_copies(list, f, flags_, in, &cl, &why);
      if (copied) {
        job.raw_inputs = true;
        job.color_pitch = cl.color_pitch;
        job.depth_pitch = cl.depth_pitch;
        job.motion_pitch = cl.motion_pitch;
        job.motion_channels = cl.motion_channels;
        for (int k = 0; k < 2; ++k) job.ms[k] = f.motion_scale[k], job.mo[k] = f.motion_offset[k];
      }
      note_path(&input_path_, copied ? "input: copies (RGBA16F color, R32F depth, " +
                                           std::to_string(cl.motion_channels) + "-channel FP16 motion) + " +
                                           (model_k_ ? "read by model K" : "HIP conversion")
                                     : "input: conversion pass (" + why + ")");
    }
    // model K: the compact input pass where the formats make it exact (runtime option k_input=f32: the FP32 pass)
    if (!copied && model_k_ && k_input16_ && Converter::inputs16_ok(f)) {
      conv_.record_inputs16(list, f, flags_, in);
      copied = true;
      job.raw_inputs = true;
      job.color_pitch = (int)f.render_width * 8;
      job.depth_pitch = (int)f.render_width * 4;
      job.motion_pitch = (int)f.render_width * 4;
      job.motion_channels = 2;
      for (int k = 0; k < 2; ++k) job.ms[k] = f.motion_scale[k], job.mo[k] = f.motion_offset[k];
      note_path(&input_path_, "input: model K compact pass (RGBA16F color, R32F depth, raw RG16F motion)");
    } else if (!copied && model_k_ && k_input16_) {
      note_path(&input_path_, "input: conversion pass (model K compact pass needs FP16 color and motion)");
    }
    if (!copied) conv_.record_inputs(list, f, flags_, in);
    if (motion_hi_ && slot.motion_hi.resource) {  // the raw output-size motion for the network (or the capture)
      conv_.record_motion_hi(list, f, {slot.motion_hi.resource, slot.motion_hi.offset}, ow_, oh_);
      job.motion_hi = true;
      job.motion_hi_use = motion_hi_use_;
      job.mw = (int)ow_;
      job.mh = (int)oh_;
      job.motion_hi_pitch = (int)ow_ * 4;
      motion_hi_params(f, ow_, oh_, job.mv_scale_hi, job.b_mv);  // (executor.h; executor=dx12 uses the same)
      if (motion_hi_use_)
        note_path(&motion_path_, Converter::motion_fp16(f)
                                     ? "motion: output resolution, raw FP16 texels to the network"
                                     : "motion: output resolution; the texture format " +
                                           std::to_string(f.motion.format) +
                                           " is wider than FP16, the network reads its values rounded to FP16");
    }
    if (timing_) timing_->stamp(list, v, 1);
    if (ready_flag_) conv_.record_flag(list, {set_.flag.resource, set_.flag.offset}, (uint32_t)v);

    // 2. split: the first part signals ready = v; the rest of the list waits for done = v (this frame's result,
    //    inline) or pipe = v (signalled by the executor once the previous network frame is done, pipelined)
    // (fence values are new every frame: v only grows)
    WgSplitRequest req = {};
    req.struct_size = sizeof req;
    req.done_fence = set_.ready.fence;
    req.done_value = v;
    // done_mode=spin: the second part needs no fence wait (ready = v is signalled by the first part itself); the
    // spin pass recorded below holds it until the network's flag reaches v
    // handover=staged: the second part waits for `pipe` = v, which the HIP stream signals right before this frame's
    // output stage (model K), and the spin recorded below then holds it until the done flag reaches v. The fence
    // wait's latency is spent while the output stage runs instead of after it. The spin begins only once the HIP
    // stream has provably reached this frame's last stage, so it lasts about as long as that stage (~0.4 ms at 1440p),
    // never across the game's own work or a stalled stream; and it is bounded by spin_polls in any case (a frame that
    // times out shows what the slot holds; nothing waits forever).
    req.resume_fence = staged_out_ ? set_.pipe.fence
                       : done_spin_ ? set_.ready.fence
                       : inline_    ? set_.done.fence
                                    : set_.pipe.fence;
    req.resume_value = v;
    const int64_t ts0 = timing_ ? TimingLog::now() : 0;
    phase(2);
    const WgResult r = services_->split_command_list(services_->context, &f, &req);
    phase(3);
    if (timing_) t_split_ = timing_->ms(ts0, TimingLog::now());
    if (r != WG_OK) {
      // This frame without the network (the front end takes it); the next network frame starts over.
      if (r == WG_ERROR_NOT_READY) {
        if (not_ready_++ == 0 || not_ready_ % 600 == 0)
          say(WG_LOG_INFO, "frame " + std::to_string(v) + ": command list not ready for the split (" +
                               std::to_string(not_ready_) + " so far)");
      } else {
        say(WG_LOG_WARN, "frame " + std::to_string(v) + ": command-list split failed (" + std::to_string((int)r) + ")");
      }
      force_reset_ = true;
      return r;
    }
    force_reset_ = false;

    if (done_spin_) conv_.record_spin(list, {set_.flag.resource, set_.flag.offset + kAlign}, (uint32_t)v, spin_polls_);
    // output layout of this frame's network result: RGBA16F rows for the output copy when the output texture
    // allows it (decided now, the network writes it)
    std::string whyNot;
    const bool halfOut = output_copy_ && Converter::output_copy_ok(f, &whyNot);
    job.out_half_pitch = halfOut ? Converter::half_rgba_pitch(ow_) : 0;
    job.ow = (int)ow_;
    job.oh = (int)oh_;
    // 3. the network, enqueued now; GPU-side wait for ready >= v and signal done = v
    bool doneQueued = false;
    std::string err;
    if (timing_) timing_->stamp(list, v, 2);  // the second part starts when the resume wait is over
    const int64_t te0 = timing_ ? TimingLog::now() : 0;
    phase(4);
    const bool ok = exec_->submit(job, &doneQueued, &err);
    phase(5);
    if (wd_.joinable()) wd_sub_t_ = GetTickCount64(), wd_sub_v_ = v;
    if (timing_) t_submit_ = timing_->ms(te0, TimingLog::now());
    if (!doneQueued) {
      // done must still reach v, after every earlier value (the fence must not go backwards): from our own queue,
      // behind the network's previous frame
      if (last_submitted_) rescue_->Wait(set_.done.fence, last_submitted_);
      if (!inline_ || staged_out_) rescue_->Signal(set_.pipe.fence, v);
      rescue_->Signal(set_.done.fence, v);
      if (done_spin_) rescue_flag(v);  // the spin waits for the flag, not the fence
    }
    last_submitted_ = v;
    next_slot_ = 1 - s;
    const int prev_slot = last_run_slot_;
    const uint64_t prev_run = last_run_;
    if (ok) last_run_ = v, last_run_slot_ = s;
    slot_half_[s] = job.out_half_pitch;

    // 4. output into the re-opened list
    BufferRange result;
    int resultHalf = 0;  // layout of the result read now (pipelined: what the previous network frame wrote)
    if (inline_ && ok) result = {slot.out.resource, slot.out.offset}, resultHalf = job.out_half_pitch;
    else if (!inline_ && prev_run)  // other slot
      result = {set_.slot[prev_slot].out.resource, set_.slot[prev_slot].out.offset}, resultHalf = slot_half_[prev_slot];
    if (result.resource && resultHalf > 0) {
      std::string why;
      if (conv_.record_output_copy(list, f, result, resultHalf, &why)) {
        note_path(&output_path_, "output: CopyTextureRegion (RGBA16F rows written by the network)");
      } else {  // the pass cannot read RGBA16F rows: this frame shows the upscaled color
        note_path(&output_path_, "output: bilinear fallback (" + why + ")");
        conv_.record_output(list, f, BufferRange());
      }
    } else {
      if (output_copy_)
        note_path(&output_path_, result.resource ? "output: compute pass (" + (whyNot.empty() ? std::string(
                                                       "this frame's result is in the pass layout") : whyNot) + ")"
                                                 : "output: no network result for this frame yet (upscaled color)");
      conv_.record_output(list, f, result);
    }
    if (ok && ++net_frames_ == 2) {  // once, at the second network frame (the first one's choices are recorded)
      const std::string k = exec_->kernels();
      if (!k.empty()) say(WG_LOG_INFO, "network kernels: " + k);
    }
    if (timing_) {
      timing_->stamp(list, v, 3);
      timing_->resolve(list, v);
      t_gpu_ = true;
      t_net_ = ok;
    }
    if (!ok) return fail_frame(WG_ERROR, err);
    return WG_OK;
  }

  // executor=dx12: the network and its output, recorded into the game's list L (no split). The network's kernels read
  // the game's textures and write its output texture themselves: no input pass, no output pass (executor_d3d12.cpp).
  // *moved: the executor did not take the frame and the context now runs on the HIP executor (switch_to_hip); nothing
  // of the frame's own work is recorded yet (the caller records it on the HIP path).
  WgResult dispatch_record(const WgFrame& f, ID3D12GraphicsCommandList* list, uint64_t v, bool* moved) {
    *moved = false;
    Job job;
    job.slot = 0;
    job.value = v;
    job.frame.rw = (int)f.render_width;
    job.frame.rh = (int)f.render_height;
    job.frame.mv_scale[0] = job.frame.mv_scale[1] = 1.0f;  // motion_scale / offset: applied where the motion is read
    job.frame.jitter[0] = f.jitter[0];
    job.frame.jitter[1] = f.jitter[1];
    job.frame.exposure = f.exposure_value;
    job.frame.pre_exposure = f.pre_exposure;
    // force_reset_: the frame after passed-through frames (skip_frame) starts over (clean/fe-minimize)
    job.frame.reset = f.reset >= 2 ? 2 : (f.reset || force_reset_) ? 1 : 0;
    job.exposure_mode = f.exposure_mode == WG_EXPOSURE_COMPUTE                       ? 2
                        : (f.exposure_mode == WG_EXPOSURE_TEXTURE && f.exposure.resource) ? 1
                                                                                          : 0;
    if (f.exposure_mode == WG_EXPOSURE_TEXTURE && !f.exposure.resource) job.frame.exposure = 1.0f;
    job.ow = (int)ow_;
    job.oh = (int)oh_;
    job.game = &f;
    job.output_copy = output_copy_;
    job.conv = &conv_;
    if (timing_) timing_->stamp(list, v, 0);
    if (timing_) timing_->stamp(list, v, 1);
    bool wrote = false;
    std::string err;
    const int64_t te0 = timing_ ? TimingLog::now() : 0;
    phase(4);
    const bool ok = exec_->record(job, list, &wrote, &err);
    phase(5);
    // a frame the executor did not take (nothing of it recorded, e.g. native AA): with the default executor the HIP
    // executor takes over the context; with executor=dx12 the frame gets the bilinear fallback below
    const bool refused = !ok && err.rfind("frame not taken", 0) == 0;
    if (refused && dx12_default_ && !hip_switch_) {  // clean/fe-wrapped-list: no move to HIP; nothing recorded
      if (!no_switch_said_) {
        no_switch_said_ = true;
        say(WG_LOG_WARN, no_switch_line(v, err));
      }
      return WG_ERROR_UNSUPPORTED;  // one frame's answer: the context stays usable (failed_ not set)
    }
    if (refused && dx12_default_ && switch_to_hip(v, err)) return *moved = true, WG_OK;
    if (!refused) dx12_last_ = v;
    note_path(&input_path_, "input: read directly by the network's D3D12 kernels (executor dx12, no input pass)");
    if ((flags_ & WG_CREATE_MOTION_OUTPUT_RES) && !model_k_)
      note_path(&motion_path_, motion_hi_use_
                                   ? (Converter::motion_fp16(f)
                                          ? "motion: output resolution, raw FP16 texels read by the network's D3D12 "
                                            "kernels"
                                          : "motion: output resolution; the texture format " +
                                                std::to_string(f.motion.format) +
                                                " is wider than FP16, the network reads its values rounded to FP16")
                                   : "motion: output resolution, resampled to render pixels by the executor's D3D12 "
                                     "pass (motion_outres=point)");
    if (timing_) t_submit_ = timing_->ms(te0, TimingLog::now());
    if (timing_) timing_->stamp(list, v, 2);
    if (ok) force_reset_ = false;  // the reset after passed-through frames is in this frame (clean/fe-minimize)
    if (ok && wrote) {
      note_path(&output_path_, "output: " + d3d12_executor_output_route(exec_.get()));
    } else {
      note_path(&output_path_, "output: bilinear fallback (the network frame failed)");
      conv_.record_output(list, f, BufferRange());
    }
    if (ok && ++net_frames_ == 2) {
      const std::string k = exec_->kernels();
      if (!k.empty()) say(WG_LOG_INFO, "network kernels: " + k);
    }
    if (timing_) {
      timing_->stamp(list, v, 3);
      timing_->resolve(list, v);
      t_gpu_ = true;
      t_net_ = ok;
    }
    // a frame the executor did not take (nothing of it recorded, e.g. a render size its kernels do not take) leaves
    // the context usable; a failure inside the recorded network does not
    // 0.6.0 follow-up: the game's textures refused for what no later frame changes (device_d3d12.cpp "input refused: "):
    // WG_ERROR_INPUT, so OptiScaler's custom slot stops making new contexts for the same inputs (wronggpu_upscaler.h)
    const bool input = !ok && !refused && err.find("input refused: ") != std::string::npos;
    if (!ok) return fail_frame(refused ? WG_ERROR_PARAMETER : input ? WG_ERROR_INPUT : WG_ERROR, err);
    return WG_OK;
  }

  // The default executor on RDNA 4 (dx12_default_) met a frame its kernels do not take (nothing of it recorded, e.g.
  // native AA, a render size above the one checked at create): the HIP executor runs this frame and every later one
  // of the context, starting over (reset), as if it had been chosen at create (shared buffers, split, fences; the
  // options input_copy and share_path apply). The D3D12 executor stays alive until the game's lists holding its
  // frames have finished (retired_). One log line; tried once (on failure: the bilinear fallback for such frames).
  bool switch_to_hip(uint64_t v, const std::string& why) {
    dx12_default_ = false;
    std::string err;
    std::unique_ptr<Executor> hip;
    if (!split_) {
      err = "the front end offers no command-list split (interface version 2)";
    } else {
      hip = make_hip_executor(rt_cfg_, adapter_);
      hip->set_log_sink([this](bool warning, const std::string& m) {
        say(warning ? WG_LOG_WARN : WG_LOG_INFO, "destroy: " + m);
      });
      say(WG_LOG_INFO, "about to start the HIP runtime (frame " + std::to_string(v) + ": the context moves to the HIP executor)");
      if (!hip->prepare(&err)) hip.reset();
    }
    if (hip) {
      std::unique_ptr<Executor> d3d = std::move(exec_);
      release_set();  // the D3D12 context's fences and flag range (no frame of it uses them)
      no_slots_ = false;
      exec_ = std::move(hip);
      bool shared = make_shared_objects(exec_->needs_handles(), share_path_opt_, &err);
      if (shared) say(WG_LOG_INFO, "about to import the shared buffers and fences into HIP (" + share_path_ + ")");
      if (!shared || !exec_->init(device_, set_, &err)) {
        exec_.reset();
        release_set();
        no_slots_ = true;
        exec_ = std::move(d3d);
      } else {
        close_handles();
        retired_ = std::move(d3d);
        retired_last_ = dx12_last_;
        if (!retired_last_ || retired_->completed() >= retired_last_) retired_.reset();
      }
    }
    if (exec_->records_into_list()) {
      say(WG_LOG_WARN, "frame " + std::to_string(v) + ": executor dx12 (default on RDNA 4) does not take this frame (" +
                           why + ") and the HIP executor could not take over (" + err +
                           "); frames it does not take get the bilinear fallback");
      return false;
    }
    input_copy_ = input_copy_opt_;
    next_slot_ = 0;
    last_run_ = last_submitted_ = 0;
    force_reset_ = true;  // the HIP network starts over with this frame
    if (timing_) {
      const bool net = exec_->enable_timing();
      timing_->set_info(net ? exec_->timing_info() : std::string());
      timing_->set_labels(exec_->timing_labels());
    }
    say(WG_LOG_WARN, "frame " + std::to_string(v) + ": executor dx12 (default on RDNA 4) does not take this frame (" +
                         why + "); the HIP executor runs this context from now on (shared buffers and fences: " +
                         share_path_ + ", GPU memory " + std::to_string(gpu_memory() >> 20) +
                         " MiB; executor=dx12 keeps the D3D12 executor with the bilinear fallback for such frames)");
    return true;
  }

  uint64_t gpu_memory() const {
    uint64_t b = 0;
    std::vector<const ID3D12Resource*> seen;  // arena buffers share one resource
    for (const auto& s : set_.slot)
      for (const SharedBuffer* x : {&s.color, &s.depth, &s.motion, &s.expo, &s.out, &s.motion_hi})
        if (x->resource && std::find(seen.begin(), seen.end(), x->resource) == seen.end()) {
          seen.push_back(x->resource);
          b += x->alloc_bytes;
        }
    return b + (exec_ ? exec_->memory_bytes() : 0);
  }

 private:
  // ---- watchdog (option watchdog=1, diagnostics) ----
  // A thread that reports, once per stuck frame, a dispatch call that has not returned for 3 s (and in which step) or
  // a network frame whose done fence has not been reached 3 s after it was enqueued, with the fence values and the
  // ready / done flags read back through a queue of its own (a stale flag poll shows as flag >= v with done < v).
  void phase(int k) {
    if (!wd_.joinable()) return;
    wd_v_ = value_;
    wd_t_ = GetTickCount64();
    wd_phase_ = k;
  }
  void start_watchdog() {
    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = 256;
    rd.Height = rd.DepthOrArraySize = rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    if (FAILED(device_->CreateCommandQueue(&qd, __uuidof(ID3D12CommandQueue), (void**)&wd_queue_)) ||
        FAILED(device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, __uuidof(ID3D12CommandAllocator),
                                               (void**)&wd_alloc_)) ||
        FAILED(device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, wd_alloc_, nullptr,
                                          __uuidof(ID3D12GraphicsCommandList), (void**)&wd_list_)) ||
        FAILED(device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence), (void**)&wd_fence_)) ||
        FAILED(device_->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                __uuidof(ID3D12Resource), (void**)&wd_read_))) {
      say(WG_LOG_WARN, "watchdog: could not create its queue; off");
      return;
    }
    wd_list_->Close();
    wd_stop_ = false;
    wd_ = std::thread([this] { watchdog_loop(); });
    say(WG_LOG_INFO, "watchdog on: stuck dispatches and network frames (> 3 s) are reported");
  }
  void stop_watchdog() {
    if (wd_.joinable()) {
      wd_stop_ = true;
      wd_.join();
    }
    release(wd_read_);
    release(wd_fence_);
    release(wd_list_);
    release(wd_alloc_);
    release(wd_queue_);
  }
  // The flag words (ready flag, done flag, spin timeouts) through the watchdog's own queue; false after 1 s.
  bool read_flags(uint32_t w[3]) {
    if (!set_.flag.resource) return false;
    wd_alloc_->Reset();
    wd_list_->Reset(wd_alloc_, nullptr);
    wd_list_->CopyBufferRegion(wd_read_, 0, set_.flag.resource, set_.flag.offset, 4);
    if (done_spin_) wd_list_->CopyBufferRegion(wd_read_, 16, set_.flag.resource, set_.flag.offset + kAlign, 8);
    wd_list_->Close();
    ID3D12CommandList* l = wd_list_;
    wd_queue_->ExecuteCommandLists(1, &l);
    wd_queue_->Signal(wd_fence_, ++wd_fv_);
    for (int i = 0; i < 100 && wd_fence_->GetCompletedValue() < wd_fv_; ++i) Sleep(10);
    if (wd_fence_->GetCompletedValue() < wd_fv_) return false;
    void* p = nullptr;
    D3D12_RANGE r = {0, 32};
    if (FAILED(wd_read_->Map(0, &r, &p)) || !p) return false;
    std::memcpy(&w[0], p, 4);
    std::memcpy(&w[1], (const char*)p + 16, 8);
    D3D12_RANGE none = {0, 0};
    wd_read_->Unmap(0, &none);
    return true;
  }
  void watchdog_loop() {
    static const char* names[] = {"idle", "dispatch (before the split)", "split_command_list", "after the split",
                                  "network enqueue (submit)", "after the enqueue"};
    uint64_t reported_d = 0, reported_n = 0;
    while (!wd_stop_) {
      Sleep(250);
      const uint64_t now = GetTickCount64();
      const int ph = wd_phase_;
      const uint64_t pv = wd_v_, sv = wd_sub_v_, pt = wd_t_, st = wd_sub_t_;
      std::string what;
      if (ph != 0 && now > pt + 3000 && reported_d != pv + 1) {
        reported_d = pv + 1;
        what = "dispatch of frame " + std::to_string(pv) + " stuck for " + std::to_string((now - pt) / 1000) +
               " s in: " + names[ph < 6 ? ph : 0];
      } else if (sv && set_.done.fence && set_.done.fence->GetCompletedValue() < sv && now > st + 3000 &&
                 reported_n != sv) {
        reported_n = sv;
        what = "network frame " + std::to_string(sv) + " not done " + std::to_string((now - st) / 1000) +
               " s after it was enqueued";
      }
      if (what.empty()) continue;
      uint32_t w[3] = {0, 0, 0};
      const bool got = read_flags(w);
      what += "; ready fence " + std::to_string(set_.ready.fence ? set_.ready.fence->GetCompletedValue() : 0) +
              ", done fence " + std::to_string(set_.done.fence ? set_.done.fence->GetCompletedValue() : 0);
      if (ready_flag_) {
        if (!got) {
          what += ", flags not readable (the watchdog's own queue did not finish in 1 s)";
        } else {
          what += ", ready flag " + std::to_string(w[0]);
          if (done_spin_) what += ", done flag " + std::to_string(w[1]) + ", spin timeouts " + std::to_string(w[2]);
        }
      }
      say(WG_LOG_WARN, "watchdog: " + what);
    }
  }

  // ---- done_mode=spin: rescue list ----
  bool make_rescue_list() {
    if (!rescue_ || !set_.flag.resource) return false;
    if (FAILED(device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, __uuidof(ID3D12CommandAllocator),
                                               (void**)&rescue_alloc_)) ||
        FAILED(device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, rescue_alloc_, nullptr,
                                          __uuidof(ID3D12GraphicsCommandList), (void**)&rescue_list_)) ||
        FAILED(device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence), (void**)&rescue_fence_)))
      return false;
    rescue_list_->Close();
    return true;
  }
  // The network of frame v could not be enqueued: the done flag must still reach v (after the fence wait already
  // queued on the rescue queue), or the game's queue spins until the poll limit.
  void rescue_flag(uint64_t v) {
    if (rescue_fv_ && rescue_fence_->GetCompletedValue() < rescue_fv_) {  // the previous rescue list must be done
      HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
      if (ev && SUCCEEDED(rescue_fence_->SetEventOnCompletion(rescue_fv_, ev))) WaitForSingleObject(ev, 2000);
      if (ev) CloseHandle(ev);
    }
    rescue_alloc_->Reset();
    rescue_list_->Reset(rescue_alloc_, nullptr);
    conv_.record_flag_immediate(rescue_list_, {set_.flag.resource, set_.flag.offset + kAlign}, (uint32_t)v);
    rescue_list_->Close();
    ID3D12CommandList* l = rescue_list_;
    rescue_->ExecuteCommandLists(1, &l);
    rescue_->Signal(rescue_fence_, ++rescue_fv_);
  }

  // ---- create helpers ----

  bool configure_hip(const Options& o, std::string* err) {
    adapter_ = adapter_description(device_);
    kernels_dir_ = o.kernels_dir.empty() ? module_dir() : o.kernels_dir;
    model_dir_ = o.model_dir;
    if (model_dir_.empty()) {
      const std::string base = local_appdata();
      if (base.empty()) return *err = "LOCALAPPDATA is not set; cannot find the model", false;
      model_dir_ = base + "\\WrongGPU\\model";
    }
    // the network model: runtime option model=K (passed on to the runtime) selects model K, else model M
    {
      WgRtConfig probe = WgRtConfig();
      probe.runtime_options = o.runtime.c_str();
      wgrt::ModelId model = wgrt::ModelId::M;
      std::string merr;
      if (!wgrt::model_from_options(probe, &model, &merr)) return *err = merr, false;
#ifndef WG_NO_MODEL_K
      model_k_ = model == wgrt::ModelId::K;
#endif
    }
    {  // model K's input path: k_input=16 (default: the compact pass where exact) or f32 (the FP32 input pass)
      const std::string ki = option_value(o.runtime, "k_input");
      if (!ki.empty() && ki != "16" && ki != "f32") return *err = "unknown k_input " + ki + " (16 or f32)", false;
      k_input16_ = ki != "f32";
    }
    map_ = o.map;
    if (map_.empty()) {
      map_ = kernels_dir_ + "\\" + kMapName;
      if (!exists(map_) && exists(model_dir_ + "\\" + kMapName)) map_ = model_dir_ + "\\" + kMapName;
    }
    const int format = o.model_format == "raw" ? 1 : o.model_format == "auto" ? 0 : 2;
    if (model_k_) {
      // model K: <model_dir>\K\model.wgcm (runtime option model_k_dir=<folder> overrides) and the preset-K map
      // next to this library or built into it (model_k_map=<file> overrides); the runtime loads and checks both
      std::string kdir = option_value(o.runtime, "model_k_dir");
      if (kdir.empty()) kdir = model_dir_ + "\\" + kKSubdir;
      model_path_ = kdir + "\\model.wgcm";
      if (!exists(model_path_))
        return *err = "model K: no converted model at " + model_path_ + " (convert the preset-K model there)", false;
      const std::string kmap = option_value(o.runtime, "model_k_map");
      if (!kmap.empty() ? !exists(kmap)
                        : !wg_file_exists(kernels_dir_ + "\\" + kKMapName) && !exists(kdir + "\\" + kKMapName))
        return *err = "model K: the converter map " + (kmap.empty() ? std::string(kKMapName) : kmap) +
                      " is missing (reinstall WrongGPU)",
               false;
      if (!wg_file_exists(kernels_dir_ + "\\k_io.co"))
        return *err = "model K: the network's kernel files are missing in " + kernels_dir_ + " (reinstall WrongGPU)",
               false;
    } else if (format == 2) {
      model_path_ = model_dir_ + "\\model.wgcm";
      if (!exists(model_path_))
        return *err = "no converted model at " + model_path_ + " (run the WrongGPU installer)", false;
      if (!wg_file_exists(map_)) return *err = "the converter map " + map_ + " is missing (reinstall WrongGPU)", false;
    }
    // Code objects and the map: next to the library, else the copies built into it (embedded.h).
    if (!wg_file_exists(kernels_dir_ + "\\encoder.co"))
      return *err = "the network's kernel files are missing in " + kernels_dir_ + " (reinstall WrongGPU)", false;
    if (!narrow_path(kernels_dir_, &n_kernels_) || !narrow_path(model_dir_, &n_model_) ||
        (exists(map_) && !narrow_path(map_, &n_map_)))
      return *err = "a WrongGPU folder name cannot be represented in the system code page and has no short name", false;
    if (!exists(map_) && wg_file_exists(map_)) n_map_ = n_kernels_ + "\\" + kMapName;
    n_rocm_ = o.rocm_dir;
    if (n_rocm_.empty()) {
      // Keep the installer probe and runtime on the same candidate. Explicit test options take precedence.
      const std::wstring record = widen(local_appdata() + "\\WrongGPU\\hip-runtime.txt");
      if (FILE* fp = _wfopen(record.c_str(), L"rb")) {
        char path[32768] = {};
        const size_t count = std::fread(path, 1, sizeof(path) - 1, fp);
        std::fclose(fp);
        const std::string chosen(path, count);
        if (count == 0 || count == sizeof(path) - 1 || !exists(chosen) || !narrow_path(chosen, &n_rocm_))
          return *err = "the installer-selected HIP runtime is missing or invalid; reinstall WrongGPU", false;
      }
    }
    n_swin_ = o.swin;
    n_rtopts_ = o.runtime;
    // motion at output resolution (model M): the runtime loads the kernels of that case (original) or keeps the
    // render-size motion (point); without the create flag the runtime gets no such option and nothing changes
    if ((flags_ & WG_CREATE_MOTION_OUTPUT_RES) && !model_k_)
      n_rtopts_ += (n_rtopts_.empty() ? "" : ";") + std::string("motion_outres=") + o.motion_outres;
    // Pre-exposure: the runtime's default, pre_exposure=scale (color / pi into the network, result * pi). The rule of
    // spec 10 draft 4 (pre_exposure=original) stays an option: with auto-exposure and a pre-exposure far from 1 it
    // leaves the network input scaled by pi (Hogwarts Legacy, pi = 0.00077: twice the frame-to-frame change on
    // dithered hair, results/2026-10-13-pre-exposure-hogwarts.md); with a supplied exposure both rules agree.
    rt_cfg_ = WgRtConfig();
    rt_cfg_.build_dir = n_kernels_.c_str();
    rt_cfg_.rocm_dir = n_rocm_.empty() ? nullptr : n_rocm_.c_str();
    rt_cfg_.model_dir = n_model_.c_str();
    rt_cfg_.map_path = n_map_.empty() ? nullptr : n_map_.c_str();
    rt_cfg_.model_format = format;
    rt_cfg_.backend = 0;
    rt_cfg_.max_rw = (int)max_rw_;
    rt_cfg_.max_rh = (int)max_rh_;
    rt_cfg_.ow = (int)ow_;
    rt_cfg_.oh = (int)oh_;
    rt_cfg_.depth_inverted = (flags_ & WG_CREATE_DEPTH_INVERTED) ? 1 : 0;
    rt_cfg_.hdr = (flags_ & WG_CREATE_HDR) ? 1 : 0;  // model K's [HDR] / [LDR] variant
    rt_cfg_.auto_exposure = (flags_ & WG_CREATE_AUTO_EXPOSURE) ? 1 : 0;
    rt_cfg_.swin_code_object = n_swin_.empty() ? nullptr : n_swin_.c_str();
    rt_cfg_.recon_output = o.recon == "output" ? 1 : 0;
    rt_cfg_.runtime_options = n_rtopts_.empty() ? nullptr : n_rtopts_.c_str();  // e.g. post_diet=0 (runtime keys)
    return true;
  }

  // ---- shared buffers and fences ("sharing paths") ----
  //
  // HIP imports the shared objects through NT handles. The device the game hands to FSR can be a wrapper (a D3D12
  // interposer layer) that creates objects fine but fails CreateSharedHandle, so several paths are tried in order:
  //  1. "game device":       create and share on the device given at create;
  //  2. "underlying device": the device the created objects report through ID3D12DeviceChild::GetDevice (the device
  //                          under a wrapper): create on the given device and share there, else create and share there;
  //  3. "own device":        a device of our own on the same adapter (D3D12CreateDevice); create and share there, then
  //                          OpenSharedHandle on the given device for the game's command lists (or use the objects as
  //                          they are when that device is the underlying device itself).
  // Every failed call is logged with its HRESULT; the path used is logged. Option share_path=auto|game|underlying|own.

  struct Path {
    const char* kind;       // game | underlying | own (option share_path)
    const char* name;
    ID3D12Device* creator;  // creates the objects (with the SHARED flags)
    ID3D12Device* sharer;   // CreateSharedHandle
    ID3D12Device* opener;   // OpenSharedHandle for the game-side objects, or null: the created objects are used
  };

  bool build_buffer(const Path& p, bool handles, uint64_t bytes, SharedBuffer* b, std::string* why) {
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    // shared objects below 8 MB are refused in the game process: never make one smaller than the arena minimum
    rd.Width = handles ? std::max(bytes, kArenaMin) : bytes;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    ID3D12Resource* r = nullptr;
    HRESULT hr = p.creator->CreateCommittedResource(&hp, handles ? D3D12_HEAP_FLAG_SHARED : D3D12_HEAP_FLAG_NONE, &rd,
                                                    D3D12_RESOURCE_STATE_COMMON, nullptr, __uuidof(ID3D12Resource),
                                                    (void**)&r);
    if (FAILED(hr)) return *why = "CreateCommittedResource (" + std::to_string(rd.Width) + " bytes) " + hr_text(hr), false;
    b->resource = r;
    b->bytes = bytes;
    b->resource_bytes = rd.Width;
    b->alloc_bytes = p.creator->GetResourceAllocationInfo(0, 1, &rd).SizeInBytes;
    if (!handles) return true;
    hr = p.sharer->CreateSharedHandle(r, nullptr, GENERIC_ALL, nullptr, &b->handle);
    if (FAILED(hr)) return b->handle = nullptr, *why = "CreateSharedHandle (buffer) " + hr_text(hr), false;
    if (p.opener) {
      ID3D12Resource* g = nullptr;
      hr = p.opener->OpenSharedHandle(b->handle, __uuidof(ID3D12Resource), (void**)&g);
      if (FAILED(hr) || !g) return *why = "OpenSharedHandle (buffer) " + hr_text(hr), false;
      b->owner = r;
      b->resource = g;
    }
    return true;
  }

  bool build_fence(const Path& p, bool handles, SharedFence* f, std::string* why) {
    ID3D12Fence* x = nullptr;
    HRESULT hr = p.creator->CreateFence(0, handles ? D3D12_FENCE_FLAG_SHARED : D3D12_FENCE_FLAG_NONE,
                                        __uuidof(ID3D12Fence), (void**)&x);
    if (FAILED(hr)) return *why = "CreateFence " + hr_text(hr), false;
    f->fence = x;
    if (!handles) return true;
    hr = p.sharer->CreateSharedHandle(x, nullptr, GENERIC_ALL, nullptr, &f->handle);
    if (FAILED(hr)) return f->handle = nullptr, *why = "CreateSharedHandle (fence) " + hr_text(hr), false;
    if (p.opener) {
      ID3D12Fence* g = nullptr;
      hr = p.opener->OpenSharedHandle(f->handle, __uuidof(ID3D12Fence), (void**)&g);
      if (FAILED(hr) || !g) return *why = "OpenSharedHandle (fence) " + hr_text(hr), false;
      f->owner = x;
      f->fence = g;
    }
    return true;
  }

  // The arena: ONE shared committed buffer on the game's device holding every runtime buffer at a 64 KB-aligned
  // offset, so that there is one CreateSharedHandle and one HIP import. In the game process small shared objects are
  // refused (results of the in-process probe: E_INVALIDARG below some size, S_OK for 24 MB), so the arena is at least
  // 32 MB and its size doubles up to 512 MB while CreateSharedHandle refuses it.
  static constexpr uint64_t kAlign = 64 * 1024, kArenaMin = 32ull << 20, kArenaMax = 512ull << 20;

  // Bytes of each slot buffer: the dense formats of FORMATS.md, or the 256-byte pitched rows of the copy paths
  // (RGBA16F color / motion, R32F depth, RGBA16F output), whichever is larger.
  void buffer_sizes(uint64_t sizes[6]) const {
    if (no_slots_) {  // executor=dx12: no slot buffers
      for (int b = 0; b < 6; ++b) sizes[b] = 0;
      return;
    }
    const uint64_t R = (uint64_t)max_rw_ * max_rh_, O = (uint64_t)ow_ * oh_;
    const uint64_t p8 = ((uint64_t)max_rw_ * 8 + 255) / 256 * 256, p4 = ((uint64_t)max_rw_ * 4 + 255) / 256 * 256;
    const uint64_t po = (uint64_t)Converter::half_rgba_pitch(ow_);
    sizes[0] = std::max(R * 16, p8 * max_rh_);
    sizes[1] = std::max(R * 4, p4 * max_rh_);
    sizes[2] = std::max(R * 8, p8 * max_rh_);
    sizes[3] = 256;
    sizes[4] = std::max(O * 12, po * oh_);
    // motion at output resolution: O texels of two FP16 and 8 spare bytes (kernels/wg_motion_hi.h); 0 = no buffer.
    // It comes last in a slot and takes no room when absent, so every other offset is what it was.
    sizes[5] = motion_hi_ ? O * 4 + 8 : 0;
  }

  uint64_t layout(uint64_t offsets[2][6], uint64_t* flag_offset) const {
    uint64_t sizes[6];
    buffer_sizes(sizes);  // color, depth, motion, expo, out, motion_hi
    uint64_t at = 0;
    for (int sl = 0; sl < 2; ++sl)
      for (int b = 0; b < 6; ++b) {
        offsets[sl][b] = at;
        at += (sizes[b] + kAlign - 1) / kAlign * kAlign;
      }
    *flag_offset = at;  // the ready flag (u32) in its own 64 KB block, then the done flag (done_mode=spin) in the next
    return at + 2 * kAlign;
  }

  bool build_arena(bool handles, std::string* why) {
    uint64_t offsets[2][6], flagOffset = 0;
    const uint64_t need = layout(offsets, &flagOffset);
    if (!handles && exec_) {  // the executor's own memory (executor=dx12: the network's arena on this device)
      ID3D12Resource* r = nullptr;
      uint64_t base = 0, rbytes = 0;
      std::string e;
      if (exec_->alloc_shared(need, &r, &base, &rbytes, &e)) {
        for (auto& sl : offsets)
          for (uint64_t& x : sl) x += base;
        say(WG_LOG_INFO, "sharing: the slot buffers (" + std::to_string(need >> 20) + " MiB) are part of the "
                         "executor's arena (" + std::to_string(rbytes >> 20) + " MiB, one buffer with the network's "
                         "memory)");
        place_arena(r, nullptr, rbytes, 0, offsets, flagOffset + base);  // its memory is counted by the executor
        r->Release();
        return true;
      }
      if (!e.empty()) return *why = e, false;
    }
    uint64_t size = handles ? std::max(need, kArenaMin) : need;
    // An importing executor that can keep its import: the arena of an earlier context, with that import (arena pool).
    const std::string scope = handles && exec_ ? exec_->import_scope() : std::string();
    ID3D12Device* poolDevice = nullptr;  // the pool's notion of this device (one reference while this function runs)
    if (!scope.empty()) {
      poolDevice = underlying_device();
      if (!poolDevice) (poolDevice = device_)->AddRef();
      std::vector<std::unique_ptr<PooledArena>> unfit;
      pooled_ = ArenaPool::get().acquire(poolDevice, scope, size, &unfit);
      for (auto& a : unfit) {  // given up: they could not serve this context (the new arena takes their place)
        say(WG_LOG_INFO, "sharing: the kept arena of " + std::to_string(a->bytes >> 20) + " MiB is released (" +
                             (a->device != poolDevice ? "another device"
                                                      : a->scope != scope ? "another importer" : "too small") +
                             "; the HIP runtime may keep its memory until the process ends)");
        if (a->scope == scope) exec_->release_kept(&a->import);
        ArenaPool::free_arena(a.get());
      }
    }
    struct Unref {
      ID3D12Device* d;
      ~Unref() {
        if (d) d->Release();
      }
    } unref = {poolDevice};
    if (pooled_) {
      ID3D12Resource* kept = pooled_->resource;
      say(WG_LOG_INFO, "sharing: arena " + std::to_string(pooled_->bytes >> 20) + " MiB (" + std::to_string(need >> 20) +
                           " MiB used): kept from an earlier context, with its import");
      place_arena(kept, nullptr, pooled_->bytes, pooled_->alloc_bytes, offsets, flagOffset);
      return true;
    }
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    ID3D12Resource* arena = nullptr;
    HANDLE handle = nullptr;
    for (;;) {
      rd.Width = size;
      HRESULT hr = device_->CreateCommittedResource(&hp, handles ? D3D12_HEAP_FLAG_SHARED : D3D12_HEAP_FLAG_NONE, &rd,
                                                    D3D12_RESOURCE_STATE_COMMON, nullptr, __uuidof(ID3D12Resource),
                                                    (void**)&arena);
      if (FAILED(hr)) {
        arena = nullptr;
        *why = "arena " + std::to_string(size >> 20) + " MiB: CreateCommittedResource " + hr_text(hr);
        say(WG_LOG_WARN, "sharing: " + *why);
        return false;
      }
      if (!handles) break;
      hr = device_->CreateSharedHandle(arena, nullptr, GENERIC_ALL, nullptr, &handle);
      if (SUCCEEDED(hr)) {
        say(WG_LOG_INFO, "sharing: arena " + std::to_string(size >> 20) + " MiB (" + std::to_string(need >> 20) +
                             " MiB used): CreateSharedHandle S_OK");
        break;
      }
      handle = nullptr;
      *why = "arena " + std::to_string(size >> 20) + " MiB: CreateSharedHandle " + hr_text(hr);
      say(WG_LOG_WARN, "sharing: " + *why);
      arena->Release();
      arena = nullptr;
      if (size >= kArenaMax) return false;
      size = std::min(size * 2, kArenaMax);
      if (size < need) return false;
    }
    const uint64_t allocBytes = device_->GetResourceAllocationInfo(0, 1, &rd).SizeInBytes;
    if (!scope.empty()) pooled_ = ArenaPool::get().add(poolDevice, scope, arena, size, allocBytes);
    place_arena(arena, handle, size, allocBytes, offsets, flagOffset);
    arena->Release();
    return true;
  }

  // The slot buffers and the flag as ranges of the arena (one reference each, released per buffer).
  void place_arena(ID3D12Resource* arena, HANDLE handle, uint64_t size, uint64_t allocBytes,
                   const uint64_t offsets[2][6], uint64_t flagOffset) {
    ArenaImport* kept = pooled_ ? &pooled_->import : nullptr;
    uint64_t sizes[6];
    buffer_sizes(sizes);
    for (int sl = 0; sl < 2; ++sl) {
      SharedBuffer* bufs[6] = {&set_.slot[sl].color, &set_.slot[sl].depth, &set_.slot[sl].motion,
                               &set_.slot[sl].expo, &set_.slot[sl].out, &set_.slot[sl].motion_hi};
      for (int b = 0; b < 6; ++b) {
        if (!sizes[b]) continue;  // (motion_hi without the flag: no buffer)
        SharedBuffer& x = *bufs[b];
        arena->AddRef();  // one reference per buffer (released per buffer)
        x.resource = arena;
        x.handle = handle;
        x.offset = offsets[sl][b];
        x.bytes = sizes[b];
        x.resource_bytes = size;
        x.alloc_bytes = allocBytes;
        x.kept = kept;
      }
    }
    SharedBuffer& fl = set_.flag;
    arena->AddRef();
    fl.resource = arena;
    fl.handle = handle;
    fl.offset = flagOffset;
    fl.bytes = 4;
    fl.resource_bytes = size;
    fl.alloc_bytes = allocBytes;
    fl.kept = kept;
  }

  bool build_set(const Path& p, bool handles, std::string* why) {
    uint64_t z[6];
    buffer_sizes(z);
    for (auto& sl : set_.slot)
      if (!build_buffer(p, handles, z[0], &sl.color, why) || !build_buffer(p, handles, z[1], &sl.depth, why) ||
          !build_buffer(p, handles, z[2], &sl.motion, why) || !build_buffer(p, handles, z[3], &sl.expo, why) ||
          !build_buffer(p, handles, z[4], &sl.out, why) ||
          (z[5] && !build_buffer(p, handles, z[5], &sl.motion_hi, why)))
        return false;
    return build_fence(p, handles, &set_.ready, why) && build_fence(p, handles, &set_.done, why) &&
           build_fence(p, handles, &set_.pipe, why);
  }

  void release_set() {
    close_handles();
    for (auto& sl : set_.slot)
      for (SharedBuffer* x : {&sl.color, &sl.depth, &sl.motion, &sl.expo, &sl.out, &sl.motion_hi}) {
        release(x->resource);
        release(x->owner);
        *x = SharedBuffer();
      }
    release(set_.flag.resource);
    set_.flag = SharedBuffer();
    for (SharedFence* f : {&set_.ready, &set_.done, &set_.pipe}) {
      release(f->fence);
      release(f->owner);
      *f = SharedFence();
    }
    if (pooled_) {  // the context's references are gone; the pool's own keeps the arena for the next context
      const uint64_t mib = pooled_->bytes >> 20;
      if (ArenaPool::get().release(pooled_))
        say(WG_LOG_INFO, "destroy: arena " + std::to_string(mib) + " MiB and its import kept for the next context");
      pooled_ = nullptr;
    }
  }

  // The device under a possible wrapper: what an object created through the given device reports as its device.
  ID3D12Device* underlying_device() {
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = 256;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ID3D12Resource* r = nullptr;
    ID3D12Device* d = nullptr;
    const HRESULT hr = device_->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COMMON,
                                                        nullptr, __uuidof(ID3D12Resource), (void**)&r);
    if (FAILED(hr)) {
      say(WG_LOG_WARN, "sharing: probe buffer for the underlying device failed " + hr_text(hr));
      return nullptr;
    }
    if (FAILED(r->GetDevice(__uuidof(ID3D12Device), (void**)&d))) d = nullptr;
    r->Release();
    return d;
  }

  ID3D12Device* own_device() {
    typedef HRESULT(WINAPI * PfnCreateFactory)(REFIID, void**);
    typedef HRESULT(WINAPI * PfnCreateDevice)(IUnknown*, D3D_FEATURE_LEVEL, REFIID, void**);
    HMODULE dxgi = LoadLibraryW(L"dxgi.dll");
    HMODULE d3d12 = GetModuleHandleW(L"d3d12.dll");
    if (!d3d12) d3d12 = LoadLibraryW(L"d3d12.dll");
    auto factory = dxgi ? reinterpret_cast<PfnCreateFactory>(
                              reinterpret_cast<void*>(GetProcAddress(dxgi, "CreateDXGIFactory1")))
                        : nullptr;
    auto create = d3d12 ? reinterpret_cast<PfnCreateDevice>(
                              reinterpret_cast<void*>(GetProcAddress(d3d12, "D3D12CreateDevice")))
                        : nullptr;
    if (!factory || !create) {
      say(WG_LOG_WARN, "sharing: dxgi.dll / d3d12.dll entry points not found");
      return nullptr;
    }
    IDXGIFactory4* f = nullptr;
    IDXGIAdapter1* a = nullptr;
    ID3D12Device* d = nullptr;
    HRESULT hr = factory(__uuidof(IDXGIFactory4), (void**)&f);
    if (SUCCEEDED(hr)) hr = f->EnumAdapterByLuid(device_->GetAdapterLuid(), __uuidof(IDXGIAdapter1), (void**)&a);
    if (FAILED(hr)) say(WG_LOG_WARN, "sharing: adapter of the game's device not found " + hr_text(hr));
    if (SUCCEEDED(hr)) {
      hr = create(a, D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), (void**)&d);
      if (FAILED(hr)) say(WG_LOG_WARN, "sharing: D3D12CreateDevice on the game's adapter failed " + hr_text(hr)), d = nullptr;
    }
    if (a) a->Release();
    if (f) f->Release();
    return d;
  }

  bool make_shared_objects(bool handles, const std::string& only, std::string* err) {
    set_.max_rw = max_rw_;
    set_.max_rh = max_rh_;
    set_.ow = ow_;
    set_.oh = oh_;
    std::string tried;
    if (!handles) {  // executor without import (tests): the arena without sharing, fences on the game's device
      const Path p = {"game", "game device", device_, device_, nullptr};
      std::string why;
      if (build_arena(false, &why) && build_fence(p, false, &set_.ready, &why) &&
          build_fence(p, false, &set_.done, &why) && build_fence(p, false, &set_.pipe, &why))
        return share_path_ = "arena on the game device (not shared)", true;
      release_set();
      return *err = why, false;
    }
    if (only.empty() || only == "auto" || only == "arena") {
      const Path p = {"game", "game device", device_, device_, nullptr};
      std::string why;
      if (build_arena(true, &why) && build_fence(p, true, &set_.ready, &why) && build_fence(p, true, &set_.done, &why) &&
          build_fence(p, true, &set_.pipe, &why)) {
        share_path_ = "arena on the game device";
        return true;
      }
      release_set();
      say(WG_LOG_WARN, "sharing path \"arena\" failed: " + why + "; trying one shared object per buffer");
      tried = "arena: " + why;
    }
    auto attempt = [&](const Path& p) {
      if (!only.empty() && only != "auto" && only != p.kind) return false;  // (share_path=arena: arena only)
      std::string why;
      if (build_set(p, true, &why)) {
        share_path_ = p.name;
        return true;
      }
      release_set();
      say(WG_LOG_WARN, std::string("sharing path \"") + p.name + "\" failed: " + why);
      tried += (tried.empty() ? "" : "; ") + std::string(p.name) + ": " + why;
      return false;
    };
    if (attempt({"game", "game device", device_, device_, nullptr})) return true;
    underlying_ = underlying_device();
    if (underlying_ && underlying_ != device_) {
      if (attempt({"underlying", "underlying device (objects made through the game device)", device_, underlying_,
                   nullptr}) ||
          attempt({"underlying", "underlying device", underlying_, underlying_, nullptr}))
        return true;
    } else if (!underlying_) {
      tried += (tried.empty() ? "" : "; ") + std::string("underlying device: GetDevice failed");
    } else {
      say(WG_LOG_INFO, "sharing: the game's objects report the given device itself (no wrapper visible)");
    }
    own_ = own_device();
    if (own_) {
      if (attempt({"own", "own device, opened on the game device", own_, own_, device_})) return true;
      if (own_ == underlying_ && attempt({"own", "own device (= the underlying device)", own_, own_, nullptr})) return true;
    } else {
      tried += (tried.empty() ? "" : "; ") + std::string("own device: not created");
    }
    return *err = "no way to share buffers and fences with HIP (" + tried + ")", false;
  }

  void close_handles() {
    std::vector<HANDLE> closed;  // arena buffers share one handle
    for (auto& sl : set_.slot)
      for (SharedBuffer* x : {&sl.color, &sl.depth, &sl.motion, &sl.expo, &sl.out, &sl.motion_hi}) {
        if (x->handle && std::find(closed.begin(), closed.end(), x->handle) == closed.end()) {
          CloseHandle(x->handle);
          closed.push_back(x->handle);
        }
        x->handle = nullptr;
      }
    set_.flag.handle = nullptr;  // the arena's handle (closed above)
    for (SharedFence* f : {&set_.ready, &set_.done, &set_.pipe})
      if (f->handle) CloseHandle(f->handle), f->handle = nullptr;
  }

  // ---- per frame ----

  bool valid(const WgFrame& f, std::string* why) const {
    auto tex = [](const WgTexture& t) { return t.resource && t.format && t.width && t.height; };
    if (f.struct_size < sizeof(WgFrame)) return *why = "WgFrame too small (front end older than interface 2)", false;
    if (!f.command_list) return *why = "no command list", false;
    if (!tex(f.color) || !tex(f.depth) || !tex(f.motion) || !tex(f.output))
      return *why = "color, depth, motion and output are required", false;
    if (!f.render_width || !f.render_height || f.render_width > max_rw_ || f.render_height > max_rh_)
      return *why = "render size " + std::to_string(f.render_width) + "x" + std::to_string(f.render_height) +
                    " outside 1.." + std::to_string(max_rw_) + "x" + std::to_string(max_rh_),
             false;
    // clean/fe-minimize: textures that do not hold the render region or the output (a minimised window's stand-ins,
    // a resize in flight) are not an error of the frame: skip_reason passes such a frame through
    if (f.output_width != ow_ || f.output_height != oh_)
      return *why = "output size differs from the context's", false;
    return true;
  }

  // clean/fe-minimize: why this frame is skipped (empty: it runs): a game texture that does not hold the render region
  // (colour, depth, motion) or the output size. Some games hand over a tiny placeholder texture for frames without that
  // pass (Spider-Man 2: a 4x4 motion texture, every few seconds in play; also while minimised), a resize can do the
  // same for a frame or two. Such a frame is passed through (dispatch_impl): no network work, the context and its
  // memory stay, the host gets success, and the next network frame starts over. The reason names the likely cause: a
  // tiny texture (16x16 or less) "the game gave a 4x4 motion texture (no motion pass this frame?)", else a resize.
  std::string skip_reason(const WgFrame& f) const {
    auto holds = [&f](const WgTexture& t) {
      return (uint64_t)f.render_x + f.render_width <= t.width && (uint64_t)f.render_y + f.render_height <= t.height;
    };
    auto size = [](const char* name, const WgTexture& t) {
      return std::string(name) + " texture " + std::to_string(t.width) + "x" + std::to_string(t.height);
    };
    const std::string region = "the render region " + std::to_string(f.render_x) + "," + std::to_string(f.render_y) + " " +
                               std::to_string(f.render_width) + "x" + std::to_string(f.render_height);
    auto gave = [&region](const char* name, const WgTexture& t) {  // clean/fe-wrapped-list: the likely cause, in words
      const std::string dims = std::to_string(t.width) + "x" + std::to_string(t.height);
      if (t.width <= 16 && t.height <= 16)
        return "the game gave a " + dims + " " + name + " texture for " + region + " (no " + name + " pass this frame?)";
      return "the " + std::string(name) + " texture " + dims + " does not hold " + region + " (a resize or a minimised window?)";
    };
    if (!holds(f.color)) return gave("color", f.color);
    if (!holds(f.depth)) return gave("depth", f.depth);
    if (!holds(f.motion)) return gave("motion", f.motion);
    if (f.output.width < ow_ || f.output.height < oh_)
      return "the " + size("output", f.output) + " is smaller than the output size " + std::to_string(ow_) + "x" +
             std::to_string(oh_) + " (a resize or a minimised window?)";
    return std::string();
  }

  // ---- teardown ----

  void shutdown() {
    stop_watchdog();
    HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    // Everything enqueued by the network has finished (destroy is the only place where the CPU waits).
    if (last_submitted_ && set_.done.fence && ev && set_.done.fence->GetCompletedValue() < last_submitted_) {
      set_.done.fence->SetEventOnCompletion(last_submitted_, ev);
      if (WaitForSingleObject(ev, 5000) != WAIT_OBJECT_0)
        say(WG_LOG_WARN, "destroy: the network's last frame did not finish within 5 s");
    }
    if (exec_) exec_->drain();
    retired_.reset();  // (drains the game's lists that hold its frames, bounded)
    if (rescue_ && device_ && ev) {
      ID3D12Fence* f = nullptr;
      if (SUCCEEDED(device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence), (void**)&f))) {
        rescue_->Signal(f, 1);
        if (f->GetCompletedValue() < 1) {
          f->SetEventOnCompletion(1, ev);
          WaitForSingleObject(ev, 5000);
        }
        f->Release();
      }
    }
    if (ev) CloseHandle(ev);
    timing_.reset();
    exec_.reset();
    release_set();
    if (rescue_fence_ && rescue_fv_ && rescue_fence_->GetCompletedValue() < rescue_fv_) {
      HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
      if (ev && SUCCEEDED(rescue_fence_->SetEventOnCompletion(rescue_fv_, ev))) WaitForSingleObject(ev, 2000);
      if (ev) CloseHandle(ev);
    }
    release(rescue_list_);
    release(rescue_alloc_);
    release(rescue_fence_);
    release(rescue_);
    release(underlying_);
    release(own_);
    release(device_);
  }

  // Logs a copy-path decision when it changes.
  void note_path(std::string* last, const std::string& now) {
    if (*last == now) return;
    *last = now;
    say(WG_LOG_INFO, "frame " + std::to_string(value_) + ": " + now);
  }

  void say(int level, const std::string& m) { log_.say(level, m); }
  WgResult fail(WgResult r, const std::string& m) {
    say(WG_LOG_ERROR, "network runtime unavailable: " + m);
    return r;
  }
  // clean/fe-minimize: a frame passed through without the network (skip_reason): the colour upscaled into the output by
  // the output pass when both hold the frame, else nothing recorded (the output keeps its last image); the context and
  // its GPU memory stay; success for the host (an error makes a host tear the context down and create another while
  // this one still holds its memory); the next network frame starts over. Logged at the first frame of a run of such
  // frames and every 600th.
  WgResult skip_frame(const WgFrame& f, const std::string& why) {
    const uint64_t v = value_ + 1;  // for the log only: the frame counter (fences, slots) is not advanced
    ++skipped_total_;
    if (skipped_run_++ % 600 == 0)
      say(WG_LOG_WARN, "frame " + std::to_string(v) + ": passed through without the network: " + why +
                           "; the context stays, the host gets success, the next network frame starts over (" +
                           std::to_string(skipped_total_) + " frame(s) passed through so far)");
    const bool colour_ok = (uint64_t)f.render_x + f.render_width <= f.color.width &&
                           (uint64_t)f.render_y + f.render_height <= f.color.height;
    if (colour_ok && f.output.width >= ow_ && f.output.height >= oh_) {
      note_path(&output_path_, "output: the color upscaled by the output pass (frames passed through without the network)");
      conv_.record_output(static_cast<ID3D12GraphicsCommandList*>(f.command_list), f, BufferRange());
    } else {
      note_path(&output_path_, "output: not written (frames passed through whose textures do not hold the frame)");
    }
    force_reset_ = true;
    return WG_OK;
  }
  // bug hunt D: why the game's output texture cannot be written (empty: it can). See dispatch_impl.
  std::string output_refusal(const WgFrame& f) const {
    auto* out = static_cast<ID3D12Resource*>(f.output.resource);
    if (!out || (out->GetDesc().Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)) return std::string();
    std::string why;
    if (output_copy_ && Converter::output_copy_ok(f, &why)) return std::string();
    return "output texture (format " + std::to_string((unsigned)f.output.format) + ") has no unordered access and " +
           (output_copy_ ? "is not RGBA16F (" + why + "), the only format copied into such an output"
                         : std::string("the output copy route (output_copy) is off"));
  }

  WgResult fail_frame(WgResult r, const std::string& m) {
    if (r == WG_ERROR_PARAMETER) {  // bug hunt H: one frame's refusal, often the same every frame: each distinct message
      uint64_t n = 1;               // once, then at 10, 100, 1000... times with the count
      auto it = refusals_.find(m);
      if (it != refusals_.end()) n = ++it->second;
      else if (refusals_.size() < 256) refusals_.emplace(m, 1);
      uint64_t p = 1;
      while (p < n && p <= UINT64_MAX / 10) p *= 10;
      if (p == n)
        say(WG_LOG_ERROR, "frame " + std::to_string(value_) + ": " + m + (n > 1 ? " (" + std::to_string(n) + " times in this context)" : ""));
      return r;
    }
    say(WG_LOG_ERROR, "frame " + std::to_string(value_) + ": " + m);
    if (r != WG_ERROR_PARAMETER) failed_ = true;
    return r;
  }

  Core log_;
  const WgFrontEndServices* services_ = nullptr;
  ID3D12Device* device_ = nullptr;
  ID3D12Device* underlying_ = nullptr;    // what the given device's objects report as their device (sharing path 2)
  ID3D12Device* own_ = nullptr;           // our own device on the same adapter (sharing path 3)
  std::string share_path_;
  bool device_checked_ = false;
  std::unique_ptr<TimingLog> timing_;  // option timing_log=1
  bool ready_flag_ = false;            // option ready_wait=memory (experimental)
  bool done_spin_ = false;             // option done_mode=spin (experimental)
  bool staged_out_ = false;            // option handover=staged (opt-in)
  std::thread wd_;                     // option watchdog=1
  std::atomic<bool> wd_stop_{false};
  std::atomic<int> wd_phase_{0};
  std::atomic<uint64_t> wd_v_{0}, wd_t_{0}, wd_sub_v_{0}, wd_sub_t_{0};
  ID3D12CommandQueue* wd_queue_ = nullptr;
  ID3D12CommandAllocator* wd_alloc_ = nullptr;
  ID3D12GraphicsCommandList* wd_list_ = nullptr;
  ID3D12Fence* wd_fence_ = nullptr;
  ID3D12Resource* wd_read_ = nullptr;
  uint64_t wd_fv_ = 0;
  uint32_t spin_polls_ = 400000;
  ID3D12CommandAllocator* rescue_alloc_ = nullptr;  // done_mode=spin: the rescue path's flag write
  ID3D12GraphicsCommandList* rescue_list_ = nullptr;
  ID3D12Fence* rescue_fence_ = nullptr;
  uint64_t rescue_fv_ = 0;
  bool no_slots_ = false;  // executor=dx12: the context has no slot buffers (buffer_sizes)
  bool input_copy_ = false, output_copy_ = false;  // options input_copy / output_copy
  std::map<std::string, uint64_t> refusals_;  // fail_frame: WG_ERROR_PARAMETER messages and how often (bug hunt H)
  // WG_CREATE_MOTION_OUTPUT_RES, model M: motion_hi_ = the slots have the raw output-size motion buffer and every
  // frame fills it; motion_hi_use_ = the network reads it (option motion_outres=original, the default)
  bool motion_hi_ = false, motion_hi_use_ = false;
  std::string motion_path_;
  uint64_t net_frames_ = 0;  // network frames submitted (the kernel report is logged at the second)
  int slot_half_[2] = {0, 0};          // RGBA16F pitch of the result in each slot (0 = 3 x f32)
  std::string input_path_, output_path_;
  uint64_t timing_hz_ = 0;
  double t_split_ = -1, t_submit_ = -1;
  bool t_gpu_ = false, t_net_ = false;
  uint32_t flags_ = 0, max_rw_ = 0, max_rh_ = 0, ow_ = 0, oh_ = 0;
  bool inline_ = true;
  Converter conv_;
  std::unique_ptr<Executor> exec_;
  SharedSet set_;
  PooledArena* pooled_ = nullptr;  // the arena in set_ belongs to the arena pool (in use by this context)
  ID3D12CommandQueue* rescue_ = nullptr;  // signals `done` when the network could not be enqueued
  uint64_t value_ = 0;                    // frame value v (dispatch count, from 1)
  uint64_t last_submitted_ = 0;           // highest v whose done signal is enqueued
  uint64_t last_run_ = 0;                 // highest v whose network frame is enqueued (its result is in its slot)
  int last_run_slot_ = 0;                 // slot of that frame
  int next_slot_ = 0;
  bool failed_ = false;
  bool force_reset_ = false;
  uint64_t skipped_run_ = 0, skipped_total_ = 0;  // clean/fe-minimize: frames passed through (skip_frame)
  uint64_t not_ready_ = 0;
  // HIP executor configuration (the strings back the pointers in rt_cfg_)
  std::string adapter_, kernels_dir_, model_dir_, map_, model_path_;
#ifdef WG_NO_MODEL_K  // build without model K (build.py --no-model-k): the model K branches compile to nothing
  static constexpr bool model_k_ = false;
#else
  bool model_k_ = false;  // runtime option model=K
#endif
  bool k_default_ = false;  // model K's default mode was applied (model_k_default)
  // executor choice (executor_select.h): dx12_default_ = the D3D12 executor runs the context by the default rule (not
  // the option), so a frame it does not take moves the context to the HIP executor (switch_to_hip); what that switch
  // needs from create: the split service and the options share_path / input_copy
  bool dx12_default_ = false;
  bool hip_switch_ = kHipSwitchDefault;  // refused_frame_policy (option hip_switch; clean/fe-wrapped-list)
  bool no_switch_said_ = false;          // the hip_switch=off line was logged
  bool split_ = false;
  std::string share_path_opt_ = "auto";
  bool input_copy_opt_ = false;
  uint64_t dx12_last_ = 0;              // last frame the D3D12 executor recorded something of
  std::unique_ptr<Executor> retired_;   // the D3D12 executor after switch_to_hip, until its frames have finished
  uint64_t retired_last_ = 0;
  bool k_input16_ = true;  // model K: compact input pass (runtime option k_input=f32: the FP32 pass)
  std::string n_kernels_, n_model_, n_map_, n_rocm_, n_swin_, n_rtopts_;
  WgRtConfig rt_cfg_ = {};
};

}  // namespace
}  // namespace wgi

struct WgUpscaler {
  wgi::Context ctx;
};

namespace {

WgResult wg_create(const WgCreateInfo* info, WgUpscaler** out) {
  if (!out) return WG_ERROR_PARAMETER;
  *out = nullptr;
  if (!info || info->struct_size < sizeof(WgCreateInfo) || info->interface_version < WG_UPSCALER_INTERFACE_VERSION) {
    if (info && info->log)
      info->log(info->log_user, WG_LOG_ERROR, "network runtime: the front end is older than interface version 2");
    return WG_ERROR_UNSUPPORTED;
  }
  try {
    std::unique_ptr<WgUpscaler> u(new WgUpscaler());
    const WgResult r = u->ctx.create(*info);
    if (r != WG_OK) return r;
    *out = u.release();
    return WG_OK;
  } catch (const std::bad_alloc&) {
    if (info->log) info->log(info->log_user, WG_LOG_ERROR, "network runtime: out of memory at create");
    return WG_ERROR_MEMORY;
  } catch (...) {
    if (info->log) info->log(info->log_user, WG_LOG_ERROR, "network runtime: unexpected error at create");
    return WG_ERROR;
  }
}

WgResult wg_dispatch(WgUpscaler* u, const WgFrame* f) {
  if (!u || !f) return WG_ERROR_PARAMETER;
  try {
    return u->ctx.dispatch(*f);
  } catch (...) {
    return WG_ERROR;
  }
}

void wg_destroy(WgUpscaler* u) {
  try {
    delete u;
  } catch (...) {
  }
}

uint64_t wg_memory(WgUpscaler* u) { return u ? u->ctx.gpu_memory() : 0; }

const WgUpscalerBackend kBackend = {sizeof(WgUpscalerBackend), WG_UPSCALER_INTERFACE_VERSION, "WrongGPU network",
                                    &wg_create,  &wg_dispatch, &wg_destroy, &wg_memory, WG_BACKEND_NEEDS_SPLIT};

}  // namespace

// Interface version 2 only: the backend needs the front end's command-list split.
extern "C" __declspec(dllexport) const WgUpscalerBackend* wgGetUpscalerBackend(uint32_t interface_version) {
  return interface_version == WG_UPSCALER_INTERFACE_VERSION ? &kBackend : nullptr;
}
