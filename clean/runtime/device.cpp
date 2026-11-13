// SPDX-License-Identifier: MIT
// Device backends (see device.h). The HIP function signatures below are written from the public HIP runtime API
// documentation (hipInit, hipMalloc, hipMemcpyAsync, hipModuleLoadData, hipModuleLaunchKernel, events, streams).
#include "device.h"
#include "embedded.h"
#include "library_handle.h"

#include <cstdio>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <sstream>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace wgrt {

namespace {

// ---- dynamic library loading ----------------------------------------------------------------------------------------

void* open_library(const std::string& path) {
#if defined(_WIN32)
  return (void*)LoadLibraryExA(path.c_str(), nullptr, path.find_first_of("\\/") != std::string::npos
                                                          ? LOAD_WITH_ALTERED_SEARCH_PATH : 0);
#else
  return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
}

void close_library(void* lib) {
  if (!lib) return;
#if defined(_WIN32)
  FreeLibrary((HMODULE)lib);
#else
  dlclose(lib);
#endif
}

void* symbol(void* lib, const char* name) {
#if defined(_WIN32)
  return (void*)GetProcAddress((HMODULE)lib, name);
#else
  return dlsym(lib, name);
#endif
}

// The HIP runtime stays loaded for the rest of the process once a device has initialised it. Unloading it does not
// return what that instance holds on the GPU, and the next context's fresh instance allocates its own again (measured
// 2026-10-13, RX 9070 XT: about 300 MiB of HIP's own device memory per instance, plus whatever the instance had not
// returned; clean/FORMATS.md "Context teardown"). Imports kept for later contexts (Device::keep_memory) live in it too.
// The per-device reference (LibraryHandle) stays balanced; this is one more that is never released.
void pin_library(void* lib, const std::string& path) {
#if defined(_WIN32)
  (void)path;
  HMODULE pinned = nullptr;
  GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_PIN | GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
                     reinterpret_cast<LPCWSTR>(lib), &pinned);
#else
  static std::mutex mutex;
  static std::set<void*> pinned;
  std::lock_guard<std::mutex> lock(mutex);
  if (pinned.insert(lib).second) dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
}

std::string library_error() {
#if defined(_WIN32)
  return "Windows error " + std::to_string((unsigned long)GetLastError());
#else
  const char* e = dlerror();
  return e ? e : "unknown error";
#endif
}

// Code objects: the file next to the library, else the copy built into it (embedded.h).
bool read_file(const std::string& path, std::vector<char>* out) { return wg_read_file(path, out); }

// ---- HIP ----------------------------------------------------------------------------------------------------------

typedef int hipError_t;
typedef void* hipStream_t;
typedef void* hipModule_t;
typedef void* hipFunction_t;
typedef void* hipEvent_t;
typedef void* hipGraph_t;
typedef void* hipGraphExec_t;
typedef void* hipGraphNode_t;
const int kH2D = 1, kD2H = 2, kD2D = 3;

// Win64 HIP external-resource ABI (hipExternalMemoryHandleDesc, hipExternalMemoryBufferDesc,
// hipExternalSemaphoreHandleDesc, hipExternalSemaphoreWait/SignalParams), taken from the audited WrongGPU code in
// imported/0.2.0-infra/runtime/runtime/hip_session.h (verified there against ROCm 7.2 hip_runtime_api.h).
struct ExtMemoryDesc {
  int type{};  // 5 = hipExternalMemoryHandleTypeD3D12Resource
  union {
    int fd;
    struct {
      void* handle;
      const void* name;
    } win32;
  } handle{};
  uint64_t size{};
  uint32_t flags{}, reserved[16]{};  // flags 1 = hipExternalMemoryDedicated
};
struct ExtBufferDesc {
  uint64_t offset{}, size{};
  uint32_t flags{}, reserved[16]{};
};
struct ExtSemaphoreDesc {
  int type{};  // 4 = hipExternalSemaphoreHandleTypeD3D12Fence
  union {
    int fd;
    struct {
      void* handle;
      const void* name;
    } win32;
  } handle{};
  uint32_t flags{}, reserved[16]{};
};
struct ExtSignalParams {
  struct {
    struct {
      uint64_t value;
    } fence{};
    uint64_t nvSciSync{}, key{}, reserved[6]{};
  } params{};
  uint32_t flags{}, reserved[16]{};
};
struct ExtWaitParams {
  struct {
    struct {
      uint64_t value;
    } fence{};
    uint64_t nvSciSync{};
    struct {
      uint64_t key;
      uint32_t timeoutMs;
    } keyedMutex{};
    uint32_t reserved[10]{};
  } params{};
  uint32_t flags{}, reserved[16]{};
};
static_assert(sizeof(ExtMemoryDesc) == 104 && sizeof(ExtBufferDesc) == 88 && sizeof(ExtSemaphoreDesc) == 96,
              "HIP external-resource ABI");
static_assert(sizeof(ExtSignalParams) == 144 && sizeof(ExtWaitParams) == 144, "HIP external-semaphore ABI");

struct HipApi {
  hipError_t (*Init)(unsigned);
  hipError_t (*SetDevice)(int);
  hipError_t (*DeviceGetName)(char*, int, int);
  hipError_t (*GetDeviceProperties)(void*, int);
  hipError_t (*Malloc)(void**, size_t);
  hipError_t (*Free)(void*);
  hipError_t (*MemcpyAsync)(void*, const void*, size_t, int, hipStream_t);
  hipError_t (*MemsetAsync)(void*, int, size_t, hipStream_t);
  hipError_t (*StreamCreate)(hipStream_t*);
  hipError_t (*StreamDestroy)(hipStream_t);
  hipError_t (*StreamSynchronize)(hipStream_t);
  hipError_t (*ModuleLoadData)(hipModule_t*, const void*);
  hipError_t (*ModuleUnload)(hipModule_t);
  hipError_t (*ModuleGetFunction)(hipFunction_t*, hipModule_t, const char*);
  hipError_t (*ModuleLaunchKernel)(hipFunction_t, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                                   hipStream_t, void**, void**);
  hipError_t (*EventCreate)(hipEvent_t*);
  hipError_t (*EventDestroy)(hipEvent_t);
  hipError_t (*EventRecord)(hipEvent_t, hipStream_t);
  hipError_t (*EventSynchronize)(hipEvent_t);
  hipError_t (*EventElapsedTime)(float*, hipEvent_t, hipEvent_t);
  hipError_t (*EventQuery)(hipEvent_t);
  hipError_t (*HostMalloc)(void**, size_t, unsigned);
  hipError_t (*HostFree)(void*);
  hipError_t (*StreamWaitValue32)(hipStream_t, void*, uint32_t, unsigned, uint32_t);
  hipError_t (*StreamWriteValue32)(hipStream_t, void*, uint32_t, unsigned);
  // AMD extension: global work size in work-items, start / stop events around this kernel
  hipError_t (*ExtModuleLaunchKernel)(hipFunction_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, size_t,
                                      hipStream_t, void**, void**, hipEvent_t, hipEvent_t, uint32_t);
  const char* (*GetErrorString)(hipError_t);
  // optional
  hipError_t (*GetDeviceCount)(int*);
  hipError_t (*StreamCreateWithFlags)(hipStream_t*, unsigned);
  hipError_t (*ImportExternalMemory)(void**, const ExtMemoryDesc*);
  hipError_t (*ExternalMemoryGetMappedBuffer)(void**, void*, const ExtBufferDesc*);
  hipError_t (*DestroyExternalMemory)(void*);
  hipError_t (*ImportExternalSemaphore)(void**, const ExtSemaphoreDesc*);
  hipError_t (*DestroyExternalSemaphore)(void*);
  hipError_t (*WaitExternalSemaphoresAsync)(void* const*, const ExtWaitParams*, unsigned, hipStream_t);
  hipError_t (*SignalExternalSemaphoresAsync)(void* const*, const ExtSignalParams*, unsigned, hipStream_t);
  // graphs and stream priorities (public HIP runtime API)
  hipError_t (*StreamBeginCapture)(hipStream_t, int);  // mode: 0 global, 1 thread-local, 2 relaxed
  hipError_t (*StreamEndCapture)(hipStream_t, hipGraph_t*);
  hipError_t (*GraphInstantiate)(hipGraphExec_t*, hipGraph_t, hipGraphNode_t*, char*, size_t);
  hipError_t (*GraphExecUpdate)(hipGraphExec_t, hipGraph_t, hipGraphNode_t*, int*);
  hipError_t (*GraphLaunch)(hipGraphExec_t, hipStream_t);
  hipError_t (*GraphDestroy)(hipGraph_t);
  hipError_t (*GraphExecDestroy)(hipGraphExec_t);
  hipError_t (*DeviceGetStreamPriorityRange)(int*, int*);
  hipError_t (*StreamCreateWithPriority)(hipStream_t*, unsigned, int);
  hipError_t (*MemGetInfo)(size_t*, size_t*);  // free and total memory of the current device
};

class HipDevice : public Device {
 public:
  ~HipDevice() override {
    if (ok_) {
      if (stream_) tear(api_.StreamSynchronize(stream_), "hipStreamSynchronize");
      for (auto& kv : imports_) release_import(kv.first, kv.second);  // imports the owner did not release
      for (void* sem : semaphores_) tear(api_.DestroyExternalSemaphore(sem), "hipDestroyExternalSemaphore");
      for (auto& e : events_)
        if (e) tear(api_.EventDestroy(e), "hipEventDestroy");  // init can fail after creating only part of the event pool
      for (auto& m : modules_) tear(api_.ModuleUnload(m.handle), "hipModuleUnload");
      if (graph_exec_) tear(api_.GraphExecDestroy(graph_exec_), "hipGraphExecDestroy");
      if (stream_) tear(api_.StreamDestroy(stream_), "hipStreamDestroy");
    }
    for (const auto& kv : teardown_failures_)
      log(true, "teardown: " + kv.first + (kv.second > 1 ? ", " + std::to_string(kv.second) + " times" : ""));
    uint64_t used = 0, total = 0;
    if (marked_ && memory_in_use(&used, &total))
      log(false, "teardown: device memory in use " + std::to_string(marked_used_ >> 20) + " MiB before, " +
                     std::to_string(used >> 20) + " MiB after (of " + std::to_string(total >> 20) +
                     " MiB, every process on the GPU)");
    // lib_ releases this context's amdhip64 LoadLibrary reference after its stream/resources are gone, including when
    // initialization failed before ok_ was set. The library itself stays loaded (pin_library).
  }

  void begin_teardown() override {
    if (!ok_ || device_index_ < 0) return;
    api_.SetDevice(device_index_);  // the destroying thread need not be the one that created the device
    uint64_t total = 0;
    marked_ = memory_in_use(&marked_used_, &total);
  }
  std::string import_scope() const override {
    if (!ok_ || device_index_ < 0) return std::string();
    char b[64];
    std::snprintf(b, sizeof b, "hip:%p:%d", lib_.get(), device_index_);
    return b;
  }
  void keep_memory(void* mem) override { imports_.erase(mem); }
  void release_kept_memory(void* mem, DevPtr ptr) override {
    if (ok_ && mem) release_import(mem, ptr);
  }

  bool supports_graphs() const override {
    return api_.StreamBeginCapture && api_.StreamEndCapture && api_.GraphInstantiate && api_.GraphLaunch &&
           api_.GraphDestroy && api_.GraphExecDestroy;
  }
  bool begin_capture() override {
    if (!supports_graphs()) return fail("this HIP runtime has no stream capture / graph functions");
    return check(api_.StreamBeginCapture(stream_, 2), "hipStreamBeginCapture");
  }
  bool end_capture_launch() override {
    hipGraph_t g = nullptr;
    if (!check(api_.StreamEndCapture(stream_, &g), "hipStreamEndCapture")) return false;
    bool updated = false;
    if (graph_exec_ && api_.GraphExecUpdate) {  // same topology: only the kernel arguments changed
      hipGraphNode_t bad = nullptr;
      int result = -1;
      updated = api_.GraphExecUpdate(graph_exec_, g, &bad, &result) == 0 && result == 0;
    }
    if (!updated) {
      if (graph_exec_) api_.GraphExecDestroy(graph_exec_);
      graph_exec_ = nullptr;
      if (!check(api_.GraphInstantiate(&graph_exec_, g, nullptr, nullptr, 0), "hipGraphInstantiate")) {
        api_.GraphDestroy(g);
        return false;
      }
      ++graph_instantiations_;
    }
    api_.GraphDestroy(g);
    return check(api_.GraphLaunch(graph_exec_, stream_), "hipGraphLaunch");
  }
  void abort_capture() override {
    hipGraph_t g = nullptr;
    if (api_.StreamEndCapture && api_.StreamEndCapture(stream_, &g) == 0 && g) api_.GraphDestroy(g);
  }
  bool set_stream_priority(bool high) override {
    if (!api_.DeviceGetStreamPriorityRange || !api_.StreamCreateWithPriority)
      return fail("this HIP runtime has no stream priorities");
    int least = 0, greatest = 0;
    if (!check(api_.DeviceGetStreamPriorityRange(&least, &greatest), "hipDeviceGetStreamPriorityRange")) return false;
    hipStream_t s = nullptr;
    if (!check(api_.StreamCreateWithPriority(&s, 1, high ? greatest : least), "hipStreamCreateWithPriority"))
      return false;
    if (stream_) {
      api_.StreamSynchronize(stream_);
      api_.StreamDestroy(stream_);
    }
    stream_ = s;
    return true;
  }

  bool init(const std::string& rocm_dir, const std::string& device_name) {
    std::string tried;
#if defined(_WIN32)
    // 1. an explicit HIP SDK directory (tests); 2. the AMD driver's copy in System32; 3. the standard search path;
    // 4. the newest HIP SDK under C:\Program Files\AMD\ROCm; 5. %HIP_PATH%.
    if (!rocm_dir.empty()) {
      const bool exact = rocm_dir.size() >= 4 && rocm_dir.substr(rocm_dir.size() - 4) == ".dll";
      try_load(exact ? rocm_dir : rocm_dir + "\\bin\\amdhip64_7.dll", 0, &tried);
      if (exact && !lib_) return fail("the installer-selected HIP runtime could not be loaded; reinstall WrongGPU:" + tried);
    }
    if (!lib_) try_load("amdhip64_7.dll", LOAD_LIBRARY_SEARCH_SYSTEM32, &tried);
    if (!lib_) try_load("amdhip64_7.dll", 0, &tried);
    if (!lib_) {
      const std::string sdk = newest_rocm_bin();
      if (!sdk.empty()) try_load(sdk + "\\amdhip64_7.dll", 0, &tried);
    }
    if (!lib_) {
      char hip[MAX_PATH] = {0};
      if (GetEnvironmentVariableA("HIP_PATH", hip, MAX_PATH)) try_load(std::string(hip) + "\\bin\\amdhip64_7.dll", 0, &tried);
    }
    // A stray LLVM_PATH pointing at an unrelated LLVM makes HIP module loading crash
    // (results/2026-10-05-swin-selftest.md); it is not needed by the driver's HIP runtime.
    char llvm[MAX_PATH] = {0};
    if (lib_ && GetEnvironmentVariableA("LLVM_PATH", llvm, MAX_PATH)) {
      note_ = std::string("LLVM_PATH (") + llvm + ") cleared for this process";
      SetEnvironmentVariableA("LLVM_PATH", nullptr);
    }
#else
    std::vector<std::string> names;
    if (!rocm_dir.empty()) names.push_back(rocm_dir + "/lib/libamdhip64.so");
    names.push_back("libamdhip64.so");
    for (auto& n : names) {
      lib_.reset(open_library(n));
      lib_path_ = n;
      if (lib_) break;
      tried += "\n  " + n + ": " + library_error();
    }
#endif
    if (!lib_) return fail("the AMD HIP runtime (amdhip64) could not be loaded; tried:" + tried);
#define WG_SYM(field, name)                                                         \
  api_.field = (decltype(api_.field))symbol(lib_.get(), name);                     \
  if (!api_.field) return fail(std::string("HIP runtime has no function ") + name);
    WG_SYM(Init, "hipInit") WG_SYM(SetDevice, "hipSetDevice") WG_SYM(DeviceGetName, "hipDeviceGetName")
    WG_SYM(Malloc, "hipMalloc") WG_SYM(Free, "hipFree") WG_SYM(MemcpyAsync, "hipMemcpyAsync")
    WG_SYM(MemsetAsync, "hipMemsetAsync") WG_SYM(StreamCreate, "hipStreamCreate")
    WG_SYM(StreamDestroy, "hipStreamDestroy") WG_SYM(StreamSynchronize, "hipStreamSynchronize")
    WG_SYM(ModuleLoadData, "hipModuleLoadData") WG_SYM(ModuleUnload, "hipModuleUnload")
    WG_SYM(ModuleGetFunction, "hipModuleGetFunction") WG_SYM(ModuleLaunchKernel, "hipModuleLaunchKernel")
    WG_SYM(EventCreate, "hipEventCreate") WG_SYM(EventDestroy, "hipEventDestroy")
    WG_SYM(EventRecord, "hipEventRecord") WG_SYM(EventSynchronize, "hipEventSynchronize")
    WG_SYM(EventElapsedTime, "hipEventElapsedTime") WG_SYM(GetErrorString, "hipGetErrorString")
#undef WG_SYM
#define WG_OPT(field, name) api_.field = (decltype(api_.field))symbol(lib_.get(), name);
    WG_OPT(EventQuery, "hipEventQuery")
    WG_OPT(HostMalloc, "hipHostMalloc") WG_OPT(HostFree, "hipHostFree")
    WG_OPT(StreamWaitValue32, "hipStreamWaitValue32")
    WG_OPT(StreamWriteValue32, "hipStreamWriteValue32")
    WG_OPT(ExtModuleLaunchKernel, "hipExtModuleLaunchKernel")
    WG_OPT(GetDeviceCount, "hipGetDeviceCount") WG_OPT(StreamCreateWithFlags, "hipStreamCreateWithFlags")
    WG_OPT(GetDeviceProperties, "hipGetDevicePropertiesR0600")
    props_r0600_ = api_.GetDeviceProperties != nullptr;
    if (!api_.GetDeviceProperties) { WG_OPT(GetDeviceProperties, "hipGetDeviceProperties") }
    WG_OPT(ImportExternalMemory, "hipImportExternalMemory")
    WG_OPT(ExternalMemoryGetMappedBuffer, "hipExternalMemoryGetMappedBuffer")
    WG_OPT(DestroyExternalMemory, "hipDestroyExternalMemory")
    WG_OPT(ImportExternalSemaphore, "hipImportExternalSemaphore")
    WG_OPT(DestroyExternalSemaphore, "hipDestroyExternalSemaphore")
    WG_OPT(WaitExternalSemaphoresAsync, "hipWaitExternalSemaphoresAsync")
    WG_OPT(SignalExternalSemaphoresAsync, "hipSignalExternalSemaphoresAsync")
    WG_OPT(StreamBeginCapture, "hipStreamBeginCapture") WG_OPT(StreamEndCapture, "hipStreamEndCapture")
    WG_OPT(GraphInstantiate, "hipGraphInstantiate") WG_OPT(GraphExecUpdate, "hipGraphExecUpdate")
    WG_OPT(GraphLaunch, "hipGraphLaunch") WG_OPT(GraphDestroy, "hipGraphDestroy")
    WG_OPT(GraphExecDestroy, "hipGraphExecDestroy")
    WG_OPT(DeviceGetStreamPriorityRange, "hipDeviceGetStreamPriorityRange")
    WG_OPT(StreamCreateWithPriority, "hipStreamCreateWithPriority")
    WG_OPT(MemGetInfo, "hipMemGetInfo")
#undef WG_OPT
    ok_ = true;
    if (!check(api_.Init(0), "hipInit")) return false;
    pin_library(lib_.get(), lib_path_);
    int device = -1, count = 0;
    if (!api_.GetDeviceCount || !api_.GetDeviceProperties || api_.GetDeviceCount(&count) != 0)
      return fail("HIP cannot enumerate device architectures");
    for (int d = 0; d < count; ++d) {
      char nm[256] = {0};
      if (api_.DeviceGetName(nm, 255, d) != 0 || (!device_name.empty() && device_name != nm)) continue;
      // Same version-tolerant public-property text lookup as the clean installer probe.
      std::vector<char> props(65536, 0);
      if (api_.GetDeviceProperties(props.data(), d) != 0) continue;
      std::string arch;
      for (size_t i = 0; i + 7 < props.size(); ++i) {
        if (std::memcmp(props.data() + i, "gfx", 3) != 0) continue;
        size_t j = i + 3;
        while (j < props.size() && ((props[j] >= '0' && props[j] <= '9') || (props[j] >= 'a' && props[j] <= 'z'))) ++j;
        arch.assign(props.data() + i, j - i);
        break;
      }
      if (arch != "gfx1200" && arch != "gfx1201") continue;
      device = d;
      // hipDeviceProp_tR0600: char name[256], hipUUID uuid (16 bytes), char luid[8] (ROCm 7.2 hip_runtime_api.h)
      if (props_r0600_) std::memcpy(luid_, props.data() + 272, 8), has_luid_ = true;
      alt_arch_ = arch;
      note_ += (note_.empty() ? "" : "; ") + std::string("device and code objects: ") + arch;
      break;
    }
    if (device < 0) return fail("no matching HIP device with packaged gfx1200/gfx1201 kernels: " + device_name);
    if (!check(api_.SetDevice(device), "hipSetDevice")) return false;
    device_index_ = device;
    char nm[256] = {0};
    if (!check(api_.DeviceGetName(nm, 255, device), "hipDeviceGetName")) return false;
    name_ = nm;
    // A non-blocking stream: it never synchronizes implicitly with the legacy default stream.
    if (api_.StreamCreateWithFlags) {
      if (!check(api_.StreamCreateWithFlags(&stream_, 1), "hipStreamCreateWithFlags")) return false;
    } else if (!check(api_.StreamCreate(&stream_), "hipStreamCreate")) {
      return false;
    }
    events_.resize(4);
    for (auto& e : events_)
      if (!check(api_.EventCreate(&e), "hipEventCreate")) return false;
    return true;
  }

  std::string name() const override { return name_; }
  std::string note() const override { return note_; }
  bool adapter_luid(uint8_t luid[8]) const override {
    if (has_luid_) std::memcpy(luid, luid_, 8);
    return has_luid_;
  }

  bool import_memory(void* handle, uint64_t alloc_bytes, uint64_t bytes, DevPtr* ptr, void** mem) override {
    if (!api_.ImportExternalMemory || !api_.ExternalMemoryGetMappedBuffer)
      return fail("this HIP runtime has no external-memory support");
    ExtMemoryDesc d;
    d.type = 5;
    d.handle.win32.handle = handle;
    d.size = alloc_bytes;
    d.flags = 1;
    void* m = nullptr;
    if (!check(api_.ImportExternalMemory(&m, &d), "hipImportExternalMemory")) return false;
    ExtBufferDesc b;
    b.size = bytes;
    void* p = nullptr;
    if (!check(api_.ExternalMemoryGetMappedBuffer(&p, m, &b), "hipExternalMemoryGetMappedBuffer")) {
      api_.DestroyExternalMemory(m);
      return false;
    }
    *ptr = (DevPtr)p;
    *mem = m;
    imports_[m] = *ptr;
    return true;
  }
  void release_memory(void* mem, DevPtr ptr) override {
    if (imports_.erase(mem)) release_import(mem, ptr);
  }
  bool import_fence(void* handle, void** sem) override {
    if (!api_.ImportExternalSemaphore || !api_.WaitExternalSemaphoresAsync || !api_.SignalExternalSemaphoresAsync)
      return fail("this HIP runtime has no external-semaphore support");
    ExtSemaphoreDesc d;
    d.type = 4;
    d.handle.win32.handle = handle;
    if (!check(api_.ImportExternalSemaphore(sem, &d), "hipImportExternalSemaphore")) return false;
    semaphores_.insert(*sem);
    return true;
  }
  void release_fence(void* sem) override {
    if (semaphores_.erase(sem)) tear(api_.DestroyExternalSemaphore(sem), "hipDestroyExternalSemaphore");
  }
  bool wait_fence(void* sem, uint64_t value) override {
    ExtWaitParams w;
    w.params.fence.value = value;
    return check(api_.WaitExternalSemaphoresAsync(&sem, &w, 1, stream_), "hipWaitExternalSemaphoresAsync");
  }
  bool signal_fence(void* sem, uint64_t value) override {
    ExtSignalParams s;
    s.params.fence.value = value;
    return check(api_.SignalExternalSemaphoresAsync(&sem, &s, 1, stream_), "hipSignalExternalSemaphoresAsync");
  }

  // The mapped buffer is freed with hipFree before its memory object. Both calls succeed, but the HIP runtime of
  // 2026-10 keeps the imported memory until the process ends (clean/FORMATS.md "Context teardown"): an import is
  // better kept and used again (keep_memory) than released.
  void release_import(void* mem, DevPtr ptr) {
    if (ptr) tear(api_.Free((void*)ptr), "hipFree (mapped external memory)");
    tear(api_.DestroyExternalMemory(mem), "hipDestroyExternalMemory");
  }

  bool alloc(DevPtr* p, size_t bytes) override {
    void* q = nullptr;
    if (!check(api_.Malloc(&q, bytes ? bytes : 1), "hipMalloc")) return false;
    *p = (DevPtr)q;
    return true;
  }
  void free(DevPtr p) override {
    if (p) tear(api_.Free((void*)p), "hipFree");
  }
  bool upload(DevPtr dst, const void* src, size_t bytes) override {
    return check(api_.MemcpyAsync((void*)dst, src, bytes, kH2D, stream_), "hipMemcpyAsync (to GPU)");
  }
  bool download(void* dst, DevPtr src, size_t bytes) override {
    return check(api_.MemcpyAsync(dst, (const void*)src, bytes, kD2H, stream_), "hipMemcpyAsync (from GPU)") &&
           sync();
  }
  bool copy(DevPtr dst, DevPtr src, size_t bytes) override {
    return check(api_.MemcpyAsync((void*)dst, (const void*)src, bytes, kD2D, stream_), "hipMemcpyAsync (on GPU)");
  }
  bool memset(DevPtr dst, int value, size_t bytes) override {
    return check(api_.MemsetAsync((void*)dst, value, bytes, stream_), "hipMemsetAsync");
  }
  bool load_module(const std::string& path) override {
    Module m;
    if (alt_arch_.empty() || !read_alt(path, alt_arch_, &m.image))
      return fail("missing " + alt_arch_ + " code object: " + path);
    if (!check(api_.ModuleLoadData(&m.handle, m.image.data()),
               ("hipModuleLoadData(" + alt_arch_ + ":" + path + ")").c_str())) return false;
    modules_.push_back(std::move(m));
    return true;
  }
  static bool read_alt(const std::string& path, const std::string& arch, std::vector<char>* out) {
    const size_t cut = path.find_last_of("/\\");
    const std::string dir = cut == std::string::npos ? std::string() : path.substr(0, cut + 1);
    const std::string base = cut == std::string::npos ? path : path.substr(cut + 1);
    // "<arch>.<name>": on disk or embedded. "<arch>/<name>": on disk only (the embedded lookup goes by the file's base
    // name, which would find the primary code object again).
    if (read_file(dir + arch + "." + base, out)) return true;
    const std::string sub = dir + arch + "/" + base;
    if (std::FILE* fp = std::fopen(sub.c_str(), "rb")) {
      std::fclose(fp);
      return read_file(sub, out);
    }
    return false;
  }
  bool load_module_tagged(const std::string& path, const std::string& tag) override {
    if (!load_module(path)) return false;
    modules_.back().tag = tag;
    return true;
  }
  bool launch(const std::string& kernel, unsigned grid, unsigned block, void** args) override {
    hipFunction_t f = function(kernel);
    if (!f) return false;
    trace_launch(kernel, grid, block);
    return check(api_.ModuleLaunchKernel(f, grid, 1, 1, block, 1, 1, 0, stream_, args, nullptr),
                 ("hipModuleLaunchKernel(" + kernel + ")").c_str());
  }
  bool launch_timed(const std::string& kernel, unsigned grid, unsigned block, void** args, int start_event,
                    int stop_event) override {
    if (!api_.ExtModuleLaunchKernel) return Device::launch_timed(kernel, grid, block, args, start_event, stop_event);
    hipFunction_t f = function(kernel);
    if (!f) return false;
    trace_launch(kernel, grid, block);
    return check(api_.ExtModuleLaunchKernel(f, grid * block, 1, 1, block, 1, 1, 0, stream_, args, nullptr,
                                            start_event >= 0 ? events_.at(start_event) : nullptr,
                                            stop_event >= 0 ? events_.at(stop_event) : nullptr, 0),
                 ("hipExtModuleLaunchKernel(" + kernel + ")").c_str());
  }
  bool exact_kernel_events() const override { return api_.ExtModuleLaunchKernel != nullptr; }
  bool sync() override { return check(api_.StreamSynchronize(stream_), "hipStreamSynchronize"); }
  bool record(int ev) override { return check(api_.EventRecord(events_.at(ev), stream_), "hipEventRecord"); }
  void* alloc_host(size_t bytes) override {
    void* p = nullptr;
    return api_.HostMalloc && api_.HostMalloc(&p, bytes, 0) == 0 ? p : nullptr;
  }
  void free_host(void* p) override {
    if (p && api_.HostFree) tear(api_.HostFree(p), "hipHostFree");
  }
  bool download_async(void* dst, DevPtr src, size_t bytes) override {
    return check(api_.MemcpyAsync(dst, (const void*)src, bytes, kD2H, stream_), "hipMemcpyAsync (capture)");
  }
  bool has_wait_value32() const override { return api_.StreamWaitValue32 != nullptr; }
  bool write_value32(DevPtr ptr, uint32_t value) override {
    if (!api_.StreamWriteValue32) return fail("no hipStreamWriteValue32");
    return check(api_.StreamWriteValue32(stream_, (void*)ptr, value, 0), "hipStreamWriteValue32");
  }
  bool wait_value32(DevPtr ptr, uint32_t value) override {
    if (!api_.StreamWaitValue32) return fail("this HIP runtime has no hipStreamWaitValue32");
    // flags 0 = hipStreamWaitValueGte, mask all bits
    return check(api_.StreamWaitValue32(stream_, (void*)ptr, value, 0, 0xFFFFFFFFu), "hipStreamWaitValue32");
  }
  bool reserve_events(int n) override {
    while ((int)events_.size() < n) {
      hipEvent_t e = nullptr;
      if (!check(api_.EventCreate(&e), "hipEventCreate")) return false;
      events_.push_back(e);
    }
    return true;
  }
  int event_done(int ev) override {
    if (!api_.EventQuery) return -1;
    const hipError_t r = api_.EventQuery(events_.at(ev));
    return r == 0 ? 1 : (r == 600 ? 0 : -1);  // hipSuccess, hipErrorNotReady
  }
  bool elapsed_done_ms(int a, int b, float* ms) override {
    return check(api_.EventElapsedTime(ms, events_.at(a), events_.at(b)), "hipEventElapsedTime");
  }
  bool elapsed_ms(int a, int b, float* ms) override {
    return check(api_.EventSynchronize(events_.at(b)), "hipEventSynchronize") &&
           check(api_.EventElapsedTime(ms, events_.at(a), events_.at(b)), "hipEventElapsedTime");
  }

 private:
  hipGraphExec_t graph_exec_ = nullptr;  // end_capture_launch
  int graph_instantiations_ = 0;
  struct Module {
    std::vector<char> image;  // kept alive for the module's lifetime
    hipModule_t handle = nullptr;
    std::string tag;  // load_module_tagged: kernels addressed as "<tag>/<name>"
  };

  hipFunction_t function(const std::string& kernel) {
    auto it = functions_.find(kernel);
    if (it != functions_.end()) return it->second;
    const size_t slash = kernel.find('/');
    const std::string tag = slash == std::string::npos ? "" : kernel.substr(0, slash);
    const std::string name = slash == std::string::npos ? kernel : kernel.substr(slash + 1);
    for (auto& m : modules_) {
      hipFunction_t f = nullptr;
      if (m.tag == tag && api_.ModuleGetFunction(&f, m.handle, name.c_str()) == 0 && f) {
        functions_[kernel] = f;
        return f;
      }
    }
    fail("kernel " + kernel + " not found in the loaded code objects");
    return nullptr;
  }
  bool check(hipError_t e, const char* what) {
    if (e == 0) return true;
    const char* s = api_.GetErrorString ? api_.GetErrorString(e) : nullptr;
    return fail(std::string(what) + " failed: HIP error " + std::to_string(e) + " (" + (s ? s : "?") + ")");
  }
  bool fail(const std::string& m) {
    err_ = m;
    return false;
  }
  // Result of a teardown call: a failure cannot be returned from there, so it is counted per kind and reported
  // through the log sink when the device is destroyed.
  void tear(hipError_t e, const char* what) {
    if (e == 0) return;
    const char* s = api_.GetErrorString ? api_.GetErrorString(e) : nullptr;
    ++teardown_failures_[std::string(what) + " failed: HIP error " + std::to_string(e) + " (" + (s ? s : "?") + ")"];
  }
  bool memory_in_use(uint64_t* used, uint64_t* total) {
    size_t f = 0, t = 0;
    if (!api_.MemGetInfo || api_.MemGetInfo(&f, &t) != 0 || t < f) return false;
    *used = t - f;
    *total = t;
    return true;
  }
#if defined(_WIN32)
  void try_load(const std::string& path, DWORD flags, std::string* tried) {
    lib_.reset((void*)LoadLibraryExA(path.c_str(), nullptr, flags));
    if (!lib_) *tried += "\n  " + path + ": " + library_error();
  }
  // The bin folder of the newest C:\Program Files\AMD\ROCm\<version> that has amdhip64_7.dll ("" if none).
  static std::string newest_rocm_bin() {
    char pf[MAX_PATH] = {0};
    if (!GetEnvironmentVariableA("ProgramFiles", pf, MAX_PATH)) std::strcpy(pf, "C:\\Program Files");
    const std::string root = std::string(pf) + "\\AMD\\ROCm\\";
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA((root + "*").c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) return std::string();
    std::string best;
    std::vector<int> bestv;
    do {
      if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) || fd.cFileName[0] == '.') continue;
      std::vector<int> v;
      for (const char* p = fd.cFileName; *p;) {
        if (*p >= '0' && *p <= '9') {
          v.push_back(std::atoi(p));
          while (*p >= '0' && *p <= '9') ++p;
        } else {
          ++p;
        }
      }
      const std::string bin = root + fd.cFileName + "\\bin";
      if (GetFileAttributesA((bin + "\\amdhip64_7.dll").c_str()) == INVALID_FILE_ATTRIBUTES) continue;
      if (best.empty() || v > bestv) {
        best = bin;
        bestv = v;
      }
    } while (FindNextFileA(h, &fd));
    FindClose(h);
    return best;
  }
#endif

  LibraryHandle lib_{&close_library};
  std::string lib_path_;   // as opened (pin_library)
  int device_index_ = -1;  // HIP device ordinal
  std::map<std::string, int> teardown_failures_;  // message -> count (tear)
  bool marked_ = false;                           // begin_teardown
  uint64_t marked_used_ = 0;
  std::string note_;
  std::string alt_arch_;  // exact HIP-reported target; never mix code objects across architectures
  bool props_r0600_ = false;  // GetDeviceProperties is hipGetDevicePropertiesR0600 (its layout has the LUID)
  bool has_luid_ = false;
  uint8_t luid_[8] = {};
  std::map<void*, DevPtr> imports_;  // external memory -> mapped pointer
  std::set<void*> semaphores_;
  bool ok_ = false;
  HipApi api_ = {};
  hipStream_t stream_ = nullptr;
  std::vector<hipEvent_t> events_;
  std::vector<Module> modules_;
  std::map<std::string, hipFunction_t> functions_;
  std::string name_;
};

// ---- emulator -----------------------------------------------------------------------------------------------------

class EmuDevice : public Device {
 public:
  ~EmuDevice() override {
    for (auto& kv : blocks_) std::free(kv.second);
  }
  bool init(const std::string& lib_path) {
    lib_ = open_library(lib_path);
    if (!lib_) return fail("cannot load the emulator library " + lib_path + ": " + library_error());
    launch_ = (int (*)(const char*, unsigned, unsigned, void**))symbol(lib_, "emu_launch");
    has_ = (int (*)(const char*))symbol(lib_, "emu_has_kernel");
    if (!launch_ || !has_) return fail("emulator library lacks emu_launch / emu_has_kernel");
    times_.resize(4);
    return true;
  }
  std::string name() const override { return "host emulator"; }
  bool alloc(DevPtr* p, size_t bytes) override {
    const size_t n = (bytes + 255) / 256 * 256 + 256;
    void* raw = std::malloc(n);
    if (!raw) return fail("out of host memory");
    std::memset(raw, 0, n);
    const DevPtr a = ((DevPtr)raw + 255) & ~(DevPtr)255;
    blocks_[a] = raw;
    *p = a;
    return true;
  }
  void free(DevPtr p) override {
    auto it = blocks_.find(p);
    if (it != blocks_.end()) {
      std::free(it->second);
      blocks_.erase(it);
    }
  }
  bool upload(DevPtr dst, const void* src, size_t bytes) override {
    std::memcpy((void*)dst, src, bytes);
    return true;
  }
  bool download(void* dst, DevPtr src, size_t bytes) override {
    std::memcpy(dst, (const void*)src, bytes);
    return true;
  }
  bool copy(DevPtr dst, DevPtr src, size_t bytes) override {
    std::memmove((void*)dst, (const void*)src, bytes);
    return true;
  }
  bool memset(DevPtr dst, int value, size_t bytes) override {
    std::memset((void*)dst, value, bytes);
    return true;
  }
  bool load_module(const std::string&) override { return true; }  // all kernels are linked into the library
  bool load_module_tagged(const std::string& lib_path, const std::string& tag) override {
    Lib l;
    l.handle = open_library(lib_path);
    if (!l.handle) return fail("cannot load the emulator library " + lib_path + ": " + library_error());
    l.launch = (int (*)(const char*, unsigned, unsigned, void**))symbol(l.handle, "emu_launch");
    l.has = (int (*)(const char*))symbol(l.handle, "emu_has_kernel");
    if (!l.launch || !l.has) return fail("emulator library lacks emu_launch / emu_has_kernel");
    tagged_[tag] = l;
    return true;
  }
  bool launch(const std::string& kernel, unsigned grid, unsigned block, void** args) override {
    const size_t slash = kernel.find('/');
    if (slash != std::string::npos) {
      auto it = tagged_.find(kernel.substr(0, slash));
      const std::string name = kernel.substr(slash + 1);
      if (it == tagged_.end() || !it->second.has(name.c_str())) return fail("emulator has no kernel " + kernel);
      if (it->second.launch(name.c_str(), grid, block, args) != 0) return fail("emu_launch(" + kernel + ") failed");
      return true;
    }
    if (!has_(kernel.c_str())) return fail("emulator has no kernel " + kernel);
    if (launch_(kernel.c_str(), grid, block, args) != 0) return fail("emu_launch(" + kernel + ") failed");
    return true;
  }
  bool sync() override { return true; }
  bool record(int ev) override {
    times_.at(ev) = std::chrono::steady_clock::now();
    return true;
  }
  bool elapsed_ms(int a, int b, float* ms) override {
    *ms = std::chrono::duration<float, std::milli>(times_.at(b) - times_.at(a)).count();
    return true;
  }
  bool reserve_events(int n) override {
    if ((int)times_.size() < n) times_.resize(n);
    return true;
  }
  int event_done(int) override { return 1; }
  bool elapsed_done_ms(int a, int b, float* ms) override { return elapsed_ms(a, b, ms); }

 private:
  bool fail(const std::string& m) {
    err_ = m;
    return false;
  }
  void* lib_ = nullptr;
  int (*launch_)(const char*, unsigned, unsigned, void**) = nullptr;
  int (*has_)(const char*) = nullptr;
  struct Lib {
    void* handle = nullptr;
    int (*launch)(const char*, unsigned, unsigned, void**) = nullptr;
    int (*has)(const char*) = nullptr;
  };
  std::map<std::string, Lib> tagged_;  // load_module_tagged
  std::map<DevPtr, void*> blocks_;
  std::vector<std::chrono::steady_clock::time_point> times_;
};

}  // namespace

void trace_launch(const std::string& kernel, unsigned grid, unsigned block, const char* where) {
  static FILE* fp = [] {
    const char* p = std::getenv("WG_LAUNCH_TRACE");
    return p && *p ? std::fopen(p, "a") : (FILE*)nullptr;
  }();
  static std::mutex mutex;
  if (!fp) return;
  std::lock_guard<std::mutex> lock(mutex);
  if (where)
    std::fprintf(fp, "%s %u %u %s\n", kernel.c_str(), grid, block, where);
  else
    std::fprintf(fp, "%s %u %u\n", kernel.c_str(), grid, block);
  std::fflush(fp);
}

#if !defined(_WIN32)
std::unique_ptr<Device> make_dx12_device(const std::string&, const std::string&, std::string* err) {
  *err = "the hybrid D3D12 backend (backend 2) exists on Windows only";
  return nullptr;
}
#endif

std::unique_ptr<Device> make_hip_device(const std::string& rocm_dir, std::string* err,
                                        const std::string& device_name) {
  std::unique_ptr<HipDevice> d(new HipDevice());
  if (!d->init(rocm_dir, device_name)) {
    *err = d->err();
    return nullptr;
  }
  return std::unique_ptr<Device>(d.release());
}

std::unique_ptr<Device> make_emu_device(const std::string& lib_path, std::string* err) {
  std::unique_ptr<EmuDevice> d(new EmuDevice());
  if (!d->init(lib_path)) {
    *err = d->err();
    return nullptr;
  }
  return std::unique_ptr<Device>(d.release());
}

}  // namespace wgrt
