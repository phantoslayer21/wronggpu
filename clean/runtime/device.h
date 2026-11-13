// SPDX-License-Identifier: MIT
// Device layer of the runtime: memory, code objects, kernel launches and events on one in-order stream.
//
// Four backends:
//  - HIP: the AMD HIP runtime (amdhip64), loaded at run time (LoadLibrary / dlopen) and called through function
//    pointers declared here from the public HIP runtime API. No HIP header is needed to build the runtime, and a
//    missing HIP runtime is an ordinary, reported error (spec 10 section 8).
//  - Emulator (tests only): the host lane emulator library built by clean/runner/build.py --emu (wg_emu.so). Device
//    memory is host memory; launches run synchronously.
//  - Hybrid D3D12 (Windows; device_dx12.cpp, runtime backend 2): a HIP device whose memory is one shared D3D12 buffer;
//    kernels with a D3D12 port (clean/dx12) run on a D3D12 queue, everything else on HIP, ordered by a shared fence.
//  - Native D3D12 (Windows; device_d3d12.cpp, runtime backend 3): D3D12 allocations, transfers, timestamps and kernels;
//    unsupported kernels/variants fail explicitly, with no HIP runtime or fallback.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace wgrt {

using DevPtr = uintptr_t;

class Device {
 public:
  virtual ~Device() = default;
  virtual std::string name() const = 0;
  // All calls return false and set err() on failure.
  virtual bool alloc(DevPtr* p, size_t bytes) = 0;
  virtual void free(DevPtr p) = 0;
  virtual bool upload(DevPtr dst, const void* src, size_t bytes) = 0;      // stream-ordered
  virtual bool download(void* dst, DevPtr src, size_t bytes) = 0;          // waits for the stream
  virtual bool copy(DevPtr dst, DevPtr src, size_t bytes) = 0;             // device to device, stream-ordered
  virtual bool memset(DevPtr dst, int value, size_t bytes) = 0;            // stream-ordered
  // Diagnostics (frame capture): page-locked host memory and a stream-ordered copy into it (no wait).
  virtual void* alloc_host(size_t bytes) { (void)bytes; return nullptr; }
  virtual void free_host(void* p) { (void)p; }
  virtual bool download_async(void* dst, DevPtr src, size_t bytes) { (void)dst, (void)src, (void)bytes; return false; }
  virtual bool load_module(const std::string& path) = 0;                   // code object file; kernels by name
  // A second build of kernels that also exist under the same names (a per-stage build variant): its kernels are
  // launched as "<tag>/<kernel>" and never by their plain names. Emulator: path is that build's emulator library.
  virtual bool load_module_tagged(const std::string& path, const std::string& tag) {
    (void)path, (void)tag;
    err_ = "tagged code objects are not supported by this device";
    return false;
  }
  // args: pointers to the argument values (hipModuleLaunchKernel convention).
  virtual bool launch(const std::string& kernel, unsigned grid, unsigned block, void** args) = 0;
  // The next launch may start before the previous one has finished (native D3D12: no UAV barrier between the two; the
  // kernels synchronize themselves). Devices that always order launches ignore it.
  virtual void unordered_next() {}
  // Timing log: launch with device events bracketing exactly this kernel (start_event / stop_event: pool indices,
  // -1 = none). Default: plain events recorded before and after the launch.
  virtual bool launch_timed(const std::string& kernel, unsigned grid, unsigned block, void** args, int start_event,
                            int stop_event) {
    return (start_event < 0 || record(start_event)) && launch(kernel, grid, block, args) &&
           (stop_event < 0 || record(stop_event));
  }
  // Whether launch_timed brackets the kernel itself (hipExtModuleLaunchKernel) rather than using plain events.
  virtual bool exact_kernel_events() const { return false; }
  virtual bool sync() = 0;
  // Events: index 0..n-1 of a small pool, recorded on the stream.
  virtual bool record(int event) = 0;
  virtual bool elapsed_ms(int start, int stop, float* ms) = 0;
  // Timing without waiting (in-game timing log): grow the event pool to at least n events; whether an event has
  // completed (1), not yet (0) or failed (-1); elapsed time between two completed events.
  virtual bool reserve_events(int n) = 0;
  virtual int event_done(int event) = 0;
  virtual bool elapsed_done_ms(int start, int stop, float* ms) = 0;

  // Interop with D3D12 (HIP on Windows): import a shared committed buffer (its NT shared handle and allocation size)
  // as device memory, and a shared fence as an external semaphore that the stream can wait for and signal (GPU-side).
  // Unsupported by default (emulator, Linux).
  virtual bool import_memory(void* shared_handle, uint64_t alloc_bytes, uint64_t bytes, DevPtr* ptr, void** mem) {
    (void)shared_handle, (void)alloc_bytes, (void)bytes, (void)ptr, (void)mem;
    err_ = "external memory is not supported by this device";
    return false;
  }
  virtual void release_memory(void* mem, DevPtr ptr) { (void)mem, (void)ptr; }
  virtual bool import_fence(void* shared_handle, void** semaphore) {
    (void)shared_handle, (void)semaphore;
    err_ = "external semaphores are not supported by this device";
    return false;
  }
  virtual void release_fence(void* semaphore) { (void)semaphore; }
  virtual bool wait_fence(void* semaphore, uint64_t value) {  // stream waits until the fence reaches value
    (void)semaphore, (void)value;
    return false;
  }
  virtual bool signal_fence(void* semaphore, uint64_t value) {  // stream sets the fence to value when it gets there
    (void)semaphore, (void)value;
    return false;
  }
  // Stream waits (GPU-side) until the 32-bit value at ptr is >= value (hipStreamWaitValue32). Unsupported by default.
  virtual bool wait_value32(DevPtr ptr, uint32_t value) {
    (void)ptr, (void)value;
    err_ = "stream memory waits are not supported by this device";
    return false;
  }
  virtual bool has_wait_value32() const { return false; }
  virtual bool write_value32(DevPtr ptr, uint32_t value) { (void)ptr, (void)value; return false; }
  // Optional (HIP graphs; runtime option graph=1): the launches between begin_capture and end_capture_launch are
  // recorded into a graph (stream capture) and launched as one; the instantiated graph is kept and updated in place
  // when the next capture has the same topology (only kernel arguments differ), else instantiated again.
  // abort_capture ends a failed capture without launching.
  virtual bool supports_graphs() const { return false; }
  virtual bool begin_capture() {
    err_ = "graphs are not supported by this device";
    return false;
  }
  virtual bool end_capture_launch() { return false; }
  virtual void abort_capture() {}
  // Optional (runtime option stream_priority=high): recreate the stream with the highest priority (high = true) or
  // the default one. Call before any work is queued.
  virtual bool set_stream_priority(bool high) {
    (void)high;
    err_ = "stream priorities are not supported by this device";
    return false;
  }
  // The adapter LUID of the device (Windows; 8 bytes), for picking the matching D3D12 adapter. Default: unknown.
  virtual bool adapter_luid(uint8_t luid[8]) const {
    (void)luid;
    return false;
  }
  const std::string& err() const { return err_; }
  // Something worth logging that happened at create (e.g. an environment variable that was cleared); may be empty.
  virtual std::string note() const { return std::string(); }

  // ---- teardown (clean/FORMATS.md "Context teardown") ----
  // Teardown cannot return errors (free, free_host, release_memory, release_fence, the destructor). A device call that
  // fails there goes to this sink as a warning (one line per kind of failure, with its count), the device memory
  // around the teardown (begin_teardown) as information; both at the latest when the device is destroyed. Without a
  // sink the lines are dropped.
  using LogSink = std::function<void(bool warning, const std::string& text)>;
  void set_log_sink(LogSink sink) { log_ = std::move(sink); }
  // The owner starts to take the context down, before it frees anything: the device notes the memory in use, to report
  // it next to the value at its own end.
  virtual void begin_teardown() {}
  // Imports that outlive a device object. An imported memory object and its mapped pointer are valid in every device
  // object of the same, non-empty scope (HIP: the same loaded HIP runtime and HIP device). Empty: imports cannot be
  // kept. keep_memory: this device no longer releases the import (destructor, release_memory); it is released, if
  // ever, by release_kept_memory of any device of that scope.
  virtual std::string import_scope() const { return std::string(); }
  virtual void keep_memory(void* mem) { (void)mem; }
  virtual void release_kept_memory(void* mem, DevPtr ptr) { (void)mem, (void)ptr; }

 protected:
  void log(bool warning, const std::string& m) {
    if (log_) log_(warning, m);
  }
  std::string err_;
  LogSink log_;
};

// rocm_dir: directory of the HIP SDK (Windows: <rocm>\bin\amdhip64_7.dll), tried first when given. Otherwise (and
// after it) on Windows: amdhip64_7.dll from System32 (installed by the AMD driver), the standard search path, the
// newest C:\Program Files\AMD\ROCm\*\bin, then %HIP_PATH%\bin. device_name: select the HIP device with this
// name (e.g. the D3D12 adapter's description); empty = first supported architecture. Unknown or unmatched devices fail.
std::unique_ptr<Device> make_hip_device(const std::string& rocm_dir, std::string* err,
                                        const std::string& device_name = std::string());
// lib_path: the emulator library (wg_emu.so).
std::unique_ptr<Device> make_emu_device(const std::string& lib_path, std::string* err);
// Hybrid D3D12 device (device_dx12.cpp; Windows only, elsewhere it fails with a message): a HIP device (as
// make_hip_device(rocm_dir)) plus the D3D12 ports of <build_dir>/dx12/manifest.json. Environment: WG_DX12_KERNELS
// (comma list of kernel names as launched, "all" (default) or "none"), WG_DX12_ARENA_MB (the shared memory every
// allocation comes from; 0 = diagnostics, HIP memory and HIP kernels only), WG_DX12_BATCH (execute the D3D12 list every
// n dispatches), WG_DX12_DEBUG=1 (D3D12 debug layer). A port runs only while the runtime has loaded the HIP code object
// it reproduces (manifest "co"). clean/dx12/README.md.
std::unique_ptr<Device> make_dx12_device(const std::string& rocm_dir, const std::string& build_dir, std::string* err);

// Standalone native D3D12 (Windows, backend 3). All memory, transfers, launches and timestamps belong to D3D12;
// no HIP runtime is loaded. A missing port or code-object identity mismatch is an error, never a fallback.
// Uses <build_dir>/dx12/manifest.json and WG_D3D12_ARENA_MB (16..4095, default 2560).
std::unique_ptr<Device> make_d3d12_device(const std::string& build_dir, std::string* err);

// Diagnostics: WG_LAUNCH_TRACE=<file> appends one line per kernel launch ("kernel grid block [where]").
void trace_launch(const std::string& kernel, unsigned grid, unsigned block, const char* where = nullptr);

}  // namespace wgrt
