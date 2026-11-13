// SPDX-License-Identifier: MIT
// The part of the interop that runs the network on a frame whose inputs are already in the runtime's shared
// buffers (spec 10 section 6). Swappable:
//  - the HIP executor: the core runtime (clean/runtime) on a HIP stream, with the shared D3D12 buffers imported as
//    HIP external memory and the two shared fences as HIP external semaphores (GPU-side waits and signals only);
//  - the D3D12 test executor: a trivial nearest-neighbour "network" on its own D3D12 compute queue, for offline tests
//    of the interop (Wine / WARP) without HIP.
//  - the D3D12 executor (executor=dx12, executor_d3d12.cpp): Model M on the native D3D12 device
//    (clean/runtime/device_d3d12.h) made on the GAME's device; the frame is recorded into the game's command list
//    (records_into_list: no command-list split, no fences, no second queue, no HIP); its kernels read the game's
//    textures and write the output texture themselves (no input or output conversion pass).
#pragma once

#include "d3d12_include.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "../../fsr/include/wronggpu_upscaler.h"
#include "../wg_runtime.h"

namespace wgi {

// Shared lifetime for D3D12 timestamp slots and HIP stage events. Must exceed
// the front end's frames-in-flight count so submit cannot overwrite unharvested data.
constexpr int kTimingRing = 8;

// What an executor that imports the arena keeps with it (backend.cpp, "arena pool"): its memory object and the mapped
// pointer of the whole resource. The arena and this import outlive the context and serve a later one, because the
// importer cannot give imported memory back (clean/FORMATS.md "Context teardown").
struct ArenaImport {
  void* mem = nullptr;
  uint64_t ptr = 0;
};

// `resource` / `fence` is the object the game's command lists, the front end's split and the D3D12 executor use (on the
// game's device). When the shared object had to be made on another device (backend.cpp, "sharing paths"), `owner`
// keeps that original alive and `resource` / `fence` was opened from its handle on the game's device.
// Normally every buffer is a range of one shared arena resource (several SharedBuffers with the same resource and
// handle, different offsets); in the per-buffer fallback paths each has its own resource (offset 0).
struct SharedBuffer {
  ID3D12Resource* resource = nullptr;
  ID3D12Resource* owner = nullptr;
  HANDLE handle = nullptr;      // NT handle for HIP (null when the executor does not need it)
  uint64_t offset = 0;          // byte offset of the buffer in `resource` (multiple of 64 KB)
  uint64_t bytes = 0;           // buffer size
  uint64_t resource_bytes = 0;  // size of the whole resource
  uint64_t alloc_bytes = 0;     // allocation size of the whole resource (GetResourceAllocationInfo)
  // A resource from the arena pool: where the executor finds its import of an earlier context (mem set: `handle` is
  // null then) or leaves the one it makes now. Null: import by `handle`, released with the executor.
  ArenaImport* kept = nullptr;
};

// Buffers of one frame in flight. Formats: clean/FORMATS.md "Per-frame runtime" (color 4 x f32, depth f32,
// motion 2 x f32, all [rh][rw] dense; expo one f32; out [oh][ow] 3 x f32).
// motion_hi: only with the create flag WG_CREATE_MOTION_OUTPUT_RES (model M): the raw motion texels of the output-size
// region M = O as dense rows of two FP16 (ow * 4 bytes), plus spare bytes at the end (clean/FORMATS.md "Motion at
// output resolution"); no resource otherwise.
struct SharedSlot {
  SharedBuffer color, depth, motion, expo, out, motion_hi;
};

struct SharedFence {
  ID3D12Fence* fence = nullptr;
  ID3D12Fence* owner = nullptr;
  HANDLE handle = nullptr;
};

struct SharedSet {
  uint32_t max_rw = 0, max_rh = 0, ow = 0, oh = 0;  // sizes the buffers were made for
  SharedSlot slot[2];
  SharedFence ready;  // reaches v when frame v's inputs are in its slot
  SharedFence done;   // the executor signals v when frame v's result is in its slot
  SharedFence pipe;   // pipelined jobs: the executor signals v once all work before job v has finished;
                      // handover=staged (inline only): v when frame v's output stage starts
  SharedBuffer flag;  // arena only: u32 written to v after frame v's input pass (experimental ready_wait=memory)
};

// Exposure source of a job (as wgrt::FrameIo): 0 = frame.exposure, 1 = the f32 in the slot's expo buffer,
// 2 = compute (spec 07).
struct Job {
  int slot = 0;
  uint64_t value = 0;
  bool pipelined = false;  // signal `pipe` = value before waiting for `ready`
  WgRtFrame frame = {};
  int exposure_mode = 0;
  // Copy paths (backend options input_copy / output_copy, clean/FORMATS.md "D3D12 interop"): the slot's color, depth
  // and motion hold the game's textures as copied (RGBA16F / R32F / motion_channels x FP16 rows of the given pitches,
  // raw motion: m = ms * raw + mo); out_half_pitch > 0: the result is written as RGBA16F rows of that pitch.
  bool raw_inputs = false;
  int color_pitch = 0, depth_pitch = 0, motion_pitch = 0, motion_channels = 4;
  float ms[2] = {1, 1}, mo[2] = {0, 0};
  int out_half_pitch = 0;
  int ow = 0, oh = 0;  // output size (sparkle guard)
  // Motion at output resolution: the slot's motion_hi holds this frame's raw motion (mw x mh texels of two FP16, rows
  // of motion_hi_pitch bytes); mv_scale_hi converts it to pixels of that size, b_mv is added to it first.
  // motion_hi_use: the network reads it (option motion_outres=original); otherwise it is there for the frame capture
  // only and the network runs on the slot's render-size motion (motion_outres=point).
  bool motion_hi = false, motion_hi_use = false;
  int mw = 0, mh = 0, motion_hi_pitch = 0;
  float mv_scale_hi[2] = {1, 1}, b_mv[2] = {0, 0};
  // Executors that record into the game's list (records_into_list): the game's frame, and whether the executor writes
  // the RGBA16F output texture itself (the result in the bytes of the output copy path; else the caller's output pass
  // reads the slot's 3 x f32 result).
  const WgFrame* game = nullptr;
  bool out_direct = false;
  // executor=dx12: the backend's output_copy option (RGBA16F output in the output-copy bytes) and its Converter (the
  // output pass for formats the network's post kernel cannot store).
  bool output_copy = true;
  class Converter* conv = nullptr;
};

// Motion at output resolution (model M, both executors): motion_scale converts the game's raw motion to render pixels
// (interface); the network wants pixels of M = O (spec 20) and the offset as a term added to the raw motion (spec 03):
// m = ms (raw + mo / ms). scale = motion_scale O / R, b = motion_offset / motion_scale (0 when either is 0), per axis.
inline void motion_hi_params(const WgFrame& f, uint32_t ow, uint32_t oh, float scale[2], float b[2]) {
  const double m_r[2] = {(double)ow / f.render_width, (double)oh / f.render_height};
  for (int k = 0; k < 2; ++k) {
    scale[k] = (float)((double)f.motion_scale[k] * m_r[k]);
    b[k] = f.motion_scale[k] != 0.0f && f.motion_offset[k] != 0.0f ? (float)((double)f.motion_offset[k] / f.motion_scale[k])
                                                                    : 0.0f;
  }
}

// GPU times of one network frame (ms), from the executor's own clock (timing log). Model K fills the four parts with
// exposure, input stage, network, output stage (Executor::timing_labels).
struct NetTiming {
  float network = 0;           // first kernel to last (exposure ... filter)
  float enc0 = 0, swin = 0, dec0 = 0, post = 0;  // enc0 includes exposure and the pre-exposure division
};

class Executor {
 public:
  virtual ~Executor() = default;
  virtual const char* name() const = 0;
  // Whether the shared buffers and fences need NT handles (D3D12_HEAP_FLAG_SHARED, CreateSharedHandle).
  virtual bool needs_handles() const = 0;
  // Where the executor reports what it cannot return (teardown: a device call that fails while the context is
  // destroyed, as a warning; the device memory around the teardown, as information). Set before prepare; the sink
  // must outlive the executor.
  virtual void set_log_sink(std::function<void(bool warning, const std::string& text)> sink) { (void)sink; }
  // After prepare: non-empty when the executor's import of an arena stays valid for later executors with the same
  // scope (SharedBuffer::kept); empty when it does not import or cannot keep imports.
  virtual std::string import_scope() const { return std::string(); }
  // Releases the import kept with an arena that leaves the pool (same scope only); clears it.
  virtual void release_kept(ArenaImport* import) { *import = ArenaImport(); }
  // Checks that the executor can run here (e.g. loads the HIP runtime), before the shared resources are made.
  virtual bool prepare(std::string* err) {
    (void)err;
    return true;
  }
  // After prepare, before the shared buffers are made: executors that run on the game's device make their device and
  // the network there (the D3D12 executor).
  virtual bool prepare_device(ID3D12Device* device, std::string* err) {
    (void)device, (void)err;
    return true;
  }
  // The shared buffers as a range of the executor's own memory (one D3D12 buffer with the network's, so that its
  // kernels address both): `bytes` at a 64 KB-aligned *offset of *resource (a reference for the caller), whose size is
  // *resource_bytes. False: the caller makes its own arena.
  virtual bool alloc_shared(uint64_t bytes, ID3D12Resource** resource, uint64_t* offset, uint64_t* resource_bytes,
                            std::string* err) {
    (void)bytes, (void)resource, (void)offset, (void)resource_bytes, (void)err;
    return false;
  }
  virtual bool init(ID3D12Device* device, const SharedSet& set, std::string* err) = 0;
  // Enqueues frame `job.value` without waiting on the CPU: (pipelined jobs: signal `pipe` = value once the previous
  // work is done,) the GPU waits until `ready` >= value, writes the slot's `out`, then signals `done` = value. On
  // failure returns false; *done_queued tells whether the done (and pipe) signals were still enqueued (otherwise the
  // caller must make sure the fences reach the value).
  virtual bool submit(const Job& job, bool* done_queued, std::string* err) = 0;
  // Executors that record the network into the game's command list (no split, no fences): record() instead of submit.
  // The slot's inputs are already recorded into `list`; record() adds the network and, when job.out_direct, the output
  // write (*output_written); otherwise the caller records its output pass from the slot's out. completed(): the
  // highest frame value whose recorded work the GPU has finished (read without waiting).
  virtual bool records_into_list() const { return false; }
  virtual bool record(const Job& job, ID3D12GraphicsCommandList* list, bool* output_written, std::string* err) {
    (void)job, (void)list, (void)output_written;
    *err = "this executor does not record into the game's list";
    return false;
  }
  virtual uint64_t completed() const { return 0; }
  // Waits on the CPU until everything submitted has finished (destroy only).
  virtual void drain() = 0;
  virtual uint64_t memory_bytes() const { return 0; }
  // Timing log (off by default): record GPU timestamps around each frame from now on; timing() returns frame
  // `value`'s times once they are complete (never waits), false if not available.
  // Experimental: wait for `flag` >= value in memory instead of the `ready` fence (false if not supported).
  virtual bool use_ready_flag() { return false; }
  virtual bool use_done_flag() { return false; }
  // Option handover=staged (inline sync, model K; false if not supported): signal `pipe` = value right before the
  // frame's output stage and write the done flag after it (as use_done_flag); the game's queue waits for `pipe`, then
  // polls the done flag with a bounded spin. `pipe` reaches value also when the frame fails earlier.
  virtual bool use_staged_output() { return false; }
  virtual bool enable_timing() { return false; }
  // What the network times measure (order of the events), for the log.
  virtual std::string timing_info() const { return std::string(); }
  // Names of NetTiming's four parts enc0, swin, dec0, post as the network in use fills them (model K: exposure,
  // input, network, output).
  virtual std::vector<std::string> timing_labels() const { return {"enc0", "Swin", "dec0", "post"}; }
  // The network's kernel choices as the runtime reports them (after a frame: including that frame's post kernels).
  virtual std::string kernels() const { return std::string(); }
  virtual bool timing(uint64_t value, NetTiming* t) {
    (void)value, (void)t;
    return false;
  }
  // One line for the log after init (device, notes); may be empty.
  virtual std::string describe() const { return std::string(); }
  // The runtime's own notes of create (wgrt::runtime_notes), one per line, each logged after init; may be empty.
  virtual std::string notes() const { return std::string(); }
  // Diagnostics (backend options capture_dir / capture_frames): when the file <dir>\capture.go appears, the next
  // `frames` network frames are written to <dir> (inputs as the network reads them, the result and the frame
  // parameters; clean/FORMATS.md "Frame capture"). The GPU is synchronised for each captured frame.
  virtual void set_capture(const std::string& dir, int frames) { (void)dir, (void)frames; }
  // Sparkle guard (backend option sparkle_guard=<rate>, off by default; kernels/frame.hip sparkle_guard): after the
  // network, a pixel that brightens by more than `thr` (in t = Y / (1 + Y)) against the motion-compensated previous
  // result brightens only by `rate` of the excess per frame.
  virtual bool set_sparkle_guard(float rate, float thr) { (void)rate, (void)thr; return false; }
};

// cfg: the core runtime's configuration (strings must outlive the executor's init); adapter: the D3D12 adapter's
// description, to pick the same GPU in HIP.
std::unique_ptr<Executor> make_hip_executor(const WgRtConfig& cfg, const std::string& adapter);
// view: 0 color x exposure, 1 depth, 2 motion (see executor_test.cpp).
// handles: behave like an executor that imports the shared objects (tests of the sharing paths).
// keep (with handles): also like one whose arena import is kept for later contexts (tests of the arena pool).
std::unique_ptr<Executor> make_test_executor(int view, bool handles, bool keep = false);
// executor=dx12: cfg as for the HIP executor (its backend field is ignored: the pure D3D12 device on the game's device).
std::unique_ptr<Executor> make_d3d12_executor(const WgRtConfig& cfg);
// The D3D12 executor's output route of its last frame (one phrase for the log), empty for other executors.
std::string d3d12_executor_output_route(const Executor* e);

}  // namespace wgi
