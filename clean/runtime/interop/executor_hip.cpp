// SPDX-License-Identifier: MIT
// HIP executor (see executor.h): the core runtime on the shared D3D12 resources. WrongGPU's own code.
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <windows.h>

#include "../runtime_core.h"
#include "executor.h"

namespace wgi {

namespace {

class HipExecutor : public Executor {
 public:
  HipExecutor(const WgRtConfig& cfg, const std::string& adapter) : cfg_(cfg), adapter_(adapter) {}

  ~HipExecutor() override {
    if (!rt_) return;
    wgrt::Device* d = wgrt::runtime_device(rt_);
    d->sync();
    d->begin_teardown();
    if (!capture_dir_.empty()) capture_flush(true);
    for (CapSlot& c : cap_)
      for (void* h : c.host) d->free_host(h);
    for (wgrt::DevPtr p : guard_prev_)
      if (p) d->free(p);
    if (guard_pdepth_) d->free(guard_pdepth_);
    uint64_t lost = 0;
    for (auto& m : imports_)
      if (!m.kept) d->release_memory(m.mem, m.ptr), lost += m.bytes;
    if (lost && log_)
      log_(true, std::to_string(lost >> 20) + " MiB of shared buffers were imported for this context only (no arena); "
                 "the HIP runtime may keep that memory until the process ends");
    for (void* f : {ready_, done_, pipe_})
      if (f) d->release_fence(f);
    wgrt::runtime_destroy(rt_);  // the device goes with it and reports its teardown
  }

  const char* name() const override { return "hip"; }
  bool needs_handles() const override { return true; }

  void set_log_sink(std::function<void(bool, const std::string&)> sink) override { log_ = std::move(sink); }

  bool prepare(std::string* err) override {
    dev_ = wgrt::make_hip_device(cfg_.rocm_dir ? cfg_.rocm_dir : "", err, adapter_);
    if (!dev_) return false;
    if (log_) dev_->set_log_sink([sink = log_](bool warning, const std::string& m) { sink(warning, "HIP " + m); });
    scope_ = dev_->import_scope();
    return true;
  }

  std::string import_scope() const override { return scope_; }

  void release_kept(ArenaImport* import) override {
    wgrt::Device* d = dev_ ? dev_.get() : rt_ ? wgrt::runtime_device(rt_) : nullptr;
    if (d && import->mem) d->release_kept_memory(import->mem, (wgrt::DevPtr)import->ptr);
    *import = ArenaImport();
  }

  bool init(ID3D12Device*, const SharedSet& set, std::string* err) override {
    if (!dev_ && !prepare(err)) return false;
    std::unique_ptr<wgrt::Device> dev = std::move(dev_);
    device_name_ = dev->name();
    note_ = dev->note();
    if (device_name_ != adapter_)
      note_ += (note_.empty() ? "" : "; ") + std::string("HIP device name differs from the D3D12 adapter (") + adapter_ +
               ")";
    // Imports first: each shared resource once (the arena: one import for every buffer), buffers as pointer offsets
    // into it. An arena from the pool (SharedBuffer::kept) brings the import of an earlier context, or takes the one
    // made now with it when this context goes; other imports are owned by the device and released in the destructor
    // through runtime_device.
    for (int s = 0; s < 2; ++s) {
      const SharedBuffer* bufs[6] = {&set.slot[s].color, &set.slot[s].depth, &set.slot[s].motion, &set.slot[s].expo,
                                     &set.slot[s].out, &set.slot[s].motion_hi};
      for (int b = 0; b < 6; ++b) {
        const SharedBuffer& x = *bufs[b];
        if (!x.resource) continue;  // motion_hi: only with motion at output resolution
        const Mem* base = nullptr;
        for (const Mem& m : imports_)
          if (m.resource == x.resource) base = &m;
        if (!base) {
          Mem m;
          m.resource = x.resource;
          m.bytes = x.alloc_bytes;
          if (x.kept && x.kept->mem) {
            m.mem = x.kept->mem;
            m.ptr = (wgrt::DevPtr)x.kept->ptr;
            m.kept = true;
          } else if (!dev->import_memory(x.handle, x.alloc_bytes, x.resource_bytes, &m.ptr, &m.mem)) {
            *err = "HIP could not import a shared D3D12 buffer (" + std::to_string(x.resource_bytes >> 20) +
                   " MiB): " + dev->err();
            release_partial(dev.get());
            return false;
          } else if (x.kept) {
            dev->keep_memory(m.mem);
            x.kept->mem = m.mem;
            x.kept->ptr = m.ptr;
            m.kept = true;
          }
          imports_.push_back(m);
          base = &imports_.back();
        }
        ptrs_[s][b] = base->ptr + x.offset;
        bytes_[s][b] = x.bytes;
      }
    }
    if (set.flag.resource) {  // the flag is a range of the arena (imported above)
      for (const Mem& m : imports_)
        if (m.resource == set.flag.resource) flag_ = m.ptr + set.flag.offset;
    }
    if (!dev->import_fence(set.ready.handle, &ready_) || !dev->import_fence(set.done.handle, &done_) ||
        !dev->import_fence(set.pipe.handle, &pipe_)) {
      *err = "HIP could not import a shared D3D12 fence: " + dev->err();
      release_partial(dev.get());
      return false;
    }
    rt_ = wgrt::runtime_create(cfg_, std::move(dev), err, true);
    if (!rt_) {
      // the device (and with it the imports) is gone with the failed runtime
      imports_.clear();
      ready_ = done_ = pipe_ = nullptr;
      return false;
    }
    return true;
  }

  bool submit(const Job& job, bool* done_queued, std::string* err) override {
    *done_queued = false;
    wgrt::Device* d = wgrt::runtime_device(rt_);
    if (job.pipelined && !d->signal_fence(pipe_, job.value)) {  // stream order: after the previous frame
      *err = "HIP signal of the pipeline fence failed: " + d->err();
      return false;
    }
    if (ready_flag_ ? !d->wait_value32(flag_, (uint32_t)job.value) : !d->wait_fence(ready_, job.value)) {
      *err = "HIP wait on the input fence failed: " + d->err();
      return false;
    }
    const int tb = timing_ ? (int)(job.value % kRing) * kEvents : -1;
    if (tb >= 0) tagged_[job.value % kRing] = 0;
    const wgrt::DevPtr* p = ptrs_[job.slot];
    wgrt::FrameIo io;
    io.color = p[0];
    io.depth = p[1];
    io.motion = p[2];
    io.expo_src = p[3];
    io.out = p[4];
    io.exposure_mode = job.exposure_mode;
    io.timing_base = tb;
    io.raw_inputs = job.raw_inputs;
    io.color_pitch = job.color_pitch;
    io.depth_pitch = job.depth_pitch;
    io.motion_pitch = job.motion_pitch;
    io.motion_channels = job.motion_channels;
    for (int k = 0; k < 2; ++k) io.ms[k] = job.ms[k], io.mo[k] = job.mo[k];
    io.out_half_pitch = job.out_half_pitch;
    if (job.motion_hi && job.motion_hi_use && p[5]) {  // motion at output resolution: the raw texels (two FP16)
      io.motion_hi = p[5];
      io.motion_hi_pitch = job.motion_hi_pitch;
      io.motion_hi_wide = 0;
      io.motion_hi_w = job.mw;
      io.motion_hi_h = job.mh;
      for (int k = 0; k < 2; ++k) io.mv_scale_hi[k] = job.mv_scale_hi[k], io.b_mv[k] = job.b_mv[k];
    }
    // handover=staged, model K: the runtime signals `pipe` = v right before the output stage; the game's queue
    // waits for it and then polls the done flag (written below) with a bounded spin, so the fence hand-over overlaps
    // the output stage instead of following it
    if (staged_out_) io.out_signal = pipe_, io.out_signal_value = job.value;
    bool ok = wgrt::runtime_frame(rt_, job.frame, &io);
    // a frame that failed before its output stage: `pipe` = v here (again if the runtime did it: same value)
    if (!ok && staged_out_ && !d->signal_fence(pipe_, job.value)) {
      *err = "network frame failed and the start-of-output fence could not be signalled: " + d->err();
      return false;
    }
    if (ok && guard_rate_ > 0.0f) sparkle_guard(job, io);
    if (!ok) *err = "network frame failed: " + wgrt::runtime_error(rt_);
    if (ok && !capture_dir_.empty()) capture(job, io);
    // The done signal is enqueued even after a failure, so the game's queue never waits forever.
    // done_mode=spin: the game's queue polls this flag, so it is written even after a failure (as the done signal);
    // if it cannot be enqueued, done_queued is false and the backend's rescue path writes it
    const bool flagged = !done_flag_ || d->write_value32(flag_ + 65536, (uint32_t)job.value);
    *done_queued = d->signal_fence(done_, job.value) && flagged;
    // ready_wait=memory: an event behind every frame (as the timing marks have). Without one, the next frame's
    // hipStreamWaitValue32 can poll a stale copy of the flag forever (reproduced: test_interop_d3d12 soakpipe).
    if (ready_flag_ && !timing_) d->record(3);
    if (ok && tb >= 0 && d->record(tb + 5)) tagged_[job.value % kRing] = job.value;
    if (!*done_queued && ok) *err = "HIP signal of the output fence failed: " + d->err();
    return ok && *done_queued;
  }

  bool set_sparkle_guard(float rate, float thr) override {
    guard_rate_ = rate;
    guard_thr_ = thr;
    return true;
  }

  // Sparkle guard pass (kernels/frame.hip sparkle_guard), stream-ordered after the network: the guarded t of the
  // previous frame is ping-ponged between two own buffers (2 bytes per output pixel); the previous frame's depth is
  // copied after the pass. A failed allocation or launch leaves the frame as the network wrote it.
  void sparkle_guard(const Job& job, const wgrt::FrameIo& io) {
    if (job.raw_inputs || !job.ow || !job.oh) return;  // dense f32 depth / motion only (not the input_copy path)
    wgrt::Device* d = wgrt::runtime_device(rt_);
    int ow = job.ow, oh = job.oh, rw = job.frame.rw, rh = job.frame.rh;
    const size_t pbytes = (size_t)ow * oh * 2, dbytes = (size_t)rw * rh * 4;
    if (pbytes > guard_pcap_) {
      guard_pcap_ = 0;
      guard_valid_ = false;
      for (wgrt::DevPtr& p : guard_prev_) {
        if (p) d->free(p);
        p = 0;
        if (!d->alloc(&p, pbytes)) return;
      }
      guard_pcap_ = pbytes;
    }
    if (dbytes > guard_dcap_) {
      guard_dcap_ = 0;
      guard_valid_ = false;
      if (guard_pdepth_) d->free(guard_pdepth_);
      guard_pdepth_ = 0;
      if (!d->alloc(&guard_pdepth_, dbytes)) return;
      guard_dcap_ = dbytes;
    }
    int valid = guard_valid_ && !job.frame.reset && ow == guard_ow_ && oh == guard_oh_ && rw == guard_rw_ &&
                rh == guard_rh_;
    void* out16 = job.out_half_pitch > 0 ? (void*)io.out : nullptr;
    void* out32 = job.out_half_pitch > 0 ? nullptr : (void*)io.out;
    int opitch = job.out_half_pitch;
    void* pin = (void*)guard_prev_[guard_cur_];
    void* pout = (void*)guard_prev_[1 - guard_cur_];
    void* mot = (void*)io.motion;
    void* dep = (void*)io.depth;
    void* pdep = (void*)guard_pdepth_;
    float rate = guard_rate_, thr = guard_thr_;
    void* args[] = {&out16, &opitch, &out32, &pin, &pout, &mot, &dep, &pdep, &ow, &oh, &rw, &rh, &rate, &thr, &valid};
    guard_valid_ = false;
    if (!d->launch("sparkle_guard", (unsigned)(((size_t)ow * oh + 255) / 256), 256, args)) return;
    if (!d->copy(guard_pdepth_, io.depth, dbytes)) return;
    guard_cur_ = 1 - guard_cur_;
    guard_valid_ = true;
    guard_ow_ = ow, guard_oh_ = oh, guard_rw_ = rw, guard_rh_ = rh;
  }

  void set_capture(const std::string& dir, int frames) override {
    capture_dir_ = dir;
    capture_frames_ = frames;
    CreateDirectoryA(dir.c_str(), nullptr);
  }

  bool use_ready_flag() override {
    wgrt::Device* d = rt_ ? wgrt::runtime_device(rt_) : nullptr;
    // 0 first, and finished before this returns: a stream-ordered reset that runs late (with the first frame) would
    // overwrite the first frame's flag value written meanwhile by the game's queue, and the network would wait forever
    if (!d || !flag_ || !d->has_wait_value32() || !d->memset(flag_, 0, 4) || !d->sync()) return false;
    ready_flag_ = true;
    return true;
  }

  bool use_done_flag() override {
    wgrt::Device* d = rt_ ? wgrt::runtime_device(rt_) : nullptr;
    if (!d || !flag_ || !d->memset(flag_ + 65536, 0, 4) || !d->sync()) return false;
    done_flag_ = true;
    return true;
  }

  bool use_staged_output() override {
    if (!rt_ || !flag_ || !pipe_ || !use_done_flag()) return false;
    staged_out_ = true;
    return true;
  }

  bool enable_timing() override {
    if (!rt_ || !wgrt::runtime_device(rt_)->reserve_events(kRing * kEvents)) return false;
    timing_ = true;
    return true;
  }

  std::string kernels() const override { return rt_ ? std::string(wgrt::runtime_kernels(rt_)) : std::string(); }
  std::string notes() const override { return rt_ ? std::string(wgrt::runtime_notes(rt_)) : std::string(); }

  std::vector<std::string> timing_labels() const override {
    if (!rt_) return Executor::timing_labels();
    const char* const* l = wgrt::runtime_timing_labels(rt_);
    return {l[0], l[1], l[2], l[3]};
  }

  std::string timing_info() const override {
    const bool exact = rt_ && wgrt::runtime_device(rt_)->exact_kernel_events();
    if (const char* k = rt_ ? wgrt::runtime_timing_info(rt_) : nullptr)  // model K
      return std::string(k) + "; events " +
             (exact ? "bracket the kernels (hipExtModuleLaunchKernel)"
                    : "are plain stream events (hipExtModuleLaunchKernel missing; part boundaries may be approximate)");
    return std::string("HIP events per frame, in stream order: [0] start of the first kernel after the input wait "
                       "(color_divide if pi != 1, else exposure_partial if auto-exposure, else exposure_pair) -> "
                       "[1] start of the first Swin kernel (enc1) -> [2] start of coef_decoder -> [3] start of "
                       "post_recon -> [4] end of an empty kernel after the last one (post_filter or scale_rgb); "
                       "enc0 = [0]-[1] (color_divide, exposure, exposure_pair, input_encoder), Swin = [1]-[2], "
                       "dec0 = [2]-[3], post = [3]-[4] (post_recon, post_filter, scale_rgb); events ") +
           (exact ? "bracket the kernels (hipExtModuleLaunchKernel)"
                  : "are plain stream events (hipExtModuleLaunchKernel missing; stage boundaries may be approximate)");
  }

  bool timing(uint64_t value, NetTiming* t) override {
    if (!timing_ || tagged_[value % kRing] != value) return false;
    wgrt::Device* d = wgrt::runtime_device(rt_);
    const int b = (int)(value % kRing) * kEvents;
    if (d->event_done(b + 5) != 1) return false;
    float m[4];
    for (int k = 0; k < 4; ++k)
      if (!d->elapsed_done_ms(b + k, b + k + 1, &m[k])) return false;
    t->enc0 = m[0];
    t->swin = m[1];
    t->dec0 = m[2];
    t->post = m[3];
    t->network = m[0] + m[1] + m[2] + m[3];
    tagged_[value % kRing] = 0;
    return true;
  }

  void drain() override {
    if (rt_) wgrt::runtime_device(rt_)->sync();
    if (rt_ && !capture_dir_.empty()) capture_flush(true);
  }

  uint64_t memory_bytes() const override { return rt_ ? wgrt::runtime_memory_bytes(rt_) : 0; }

  std::string describe() const override {
    return "HIP device \"" + device_name_ + "\"" + (note_.empty() ? "" : " (" + note_ + ")");
  }

 private:
  struct Mem {
    ID3D12Resource* resource = nullptr;  // identity of the shared resource (no reference)
    wgrt::DevPtr ptr = 0;
    void* mem = nullptr;
    uint64_t bytes = 0;
    bool kept = false;  // belongs to a pooled arena (SharedBuffer::kept): not released here
  };

  // Frame capture (diagnostics): <dir>\capture.go starts it; each captured frame writes <dir>\rRR_fNNNN_<buffer>.bin
  // (the slot buffers as the network read them: color [rh][rw] 4 x f32, depth f32, motion 2 x f32 (render pixels,
  // unless raw_inputs), expo f32, out: the result buffer; with motion at output resolution also motion_hi: the raw
  // motion texels [mh][mw] 2 x FP16, whether the network read them (motion_outres=original) or the render-size
  // `motion` (point)) and one line in <dir>\frames.txt with the frame parameters.
  // Nothing here waits for the GPU: in inline mode the stream is still waiting for the game's own submission when
  // submit() returns, so a blocking download would deadlock the game. The buffers are copied after the frame on the
  // stream into page-locked host memory (a ring of kCapRing frames, an event behind each); the files are written by a
  // later submit() once that event has completed (the frame stalls the CPU for the file write only).
  static constexpr int kCapRing = 3, kCapEvent0 = 64;
  struct CapSlot {
    bool busy = false;
    void* host[6] = {};
    uint64_t cap[6] = {}, bytes[6] = {};
    std::string base, meta;
  };
  void capture_flush(bool wait) {
    wgrt::Device* d = wgrt::runtime_device(rt_);
    static const char* const kNames[6] = {"color", "depth", "motion", "expo", "out", "motion_hi"};
    for (int r = 0; r < kCapRing; ++r) {
      CapSlot& c = cap_[r];
      if (!c.busy) continue;
      const int done = d->event_done(kCapEvent0 + r);
      if (done == 0 && !wait) continue;
      if (done == 0 && wait) d->sync();
      for (int k = 0; k < 6; ++k) {
        if (!c.bytes[k]) continue;
        const std::string path = capture_dir_ + c.base + "_" + kNames[k] + ".bin";
        if (FILE* f = std::fopen(path.c_str(), "wb")) {
          std::fwrite(c.host[k], 1, (size_t)c.bytes[k], f);
          std::fclose(f);
        }
      }
      const std::string meta = capture_dir_ + "\\frames.txt";
      if (FILE* f = std::fopen(meta.c_str(), "ab")) {
        std::fputs(c.meta.c_str(), f);
        std::fclose(f);
      }
      c.busy = false;
    }
  }
  void capture(const Job& job, const wgrt::FrameIo& io) {
    capture_flush(false);
    const std::string go = capture_dir_ + "\\capture.go";
    if (capture_left_ == 0) {
      if (GetFileAttributesA(go.c_str()) == INVALID_FILE_ATTRIBUTES) return;
      DeleteFileA(go.c_str());
      capture_left_ = capture_frames_;
      ++capture_run_;
    }
    int r = -1;
    for (int i = 0; i < kCapRing && r < 0; ++i)
      if (!cap_[i].busy) r = i;
    if (r < 0) return;  // every ring slot still in flight: this frame is not captured (frames.txt shows the gap)
    --capture_left_;
    wgrt::Device* d = wgrt::runtime_device(rt_);
    if (!events_reserved_ && !(events_reserved_ = d->reserve_events(kCapEvent0 + kCapRing))) return;
    const uint64_t px = (uint64_t)job.frame.rw * job.frame.rh;
    const wgrt::DevPtr src[6] = {io.color, io.depth, io.motion, io.expo_src, io.out,
                                 job.motion_hi ? ptrs_[job.slot][5] : 0};
    const uint64_t want[6] = {job.raw_inputs ? bytes_[job.slot][0] : px * 16, job.raw_inputs ? bytes_[job.slot][1] : px * 4,
                              job.raw_inputs ? bytes_[job.slot][2] : px * 8, 4, bytes_[job.slot][4],
                              job.motion_hi ? (uint64_t)job.motion_hi_pitch * job.mh : 0};
    CapSlot& c = cap_[r];
    for (int k = 0; k < 6; ++k) {
      c.bytes[k] = 0;
      if (!src[k] || !want[k]) continue;
      if (c.cap[k] < want[k]) {
        d->free_host(c.host[k]);
        c.host[k] = d->alloc_host((size_t)want[k]);
        c.cap[k] = c.host[k] ? want[k] : 0;
      }
      if (c.host[k] && d->download_async(c.host[k], src[k], (size_t)want[k])) c.bytes[k] = want[k];
    }
    if (!d->record(kCapEvent0 + r)) return;
    char base[64];
    std::snprintf(base, sizeof base, "\\r%02d_f%04llu", capture_run_, (unsigned long long)job.value);
    c.base = base;
    const WgRtFrame& fr = job.frame;
    char line[768];
    int len = std::snprintf(line, sizeof line,
                  "run=%d value=%llu slot=%d rw=%d rh=%d jitter=%.9g,%.9g mv_scale=%.9g,%.9g mv_bias=%.9g,%.9g "
                  "exposure=%.9g pre_exposure=%.9g reset=%d exposure_mode=%d raw_inputs=%d color_pitch=%d "
                  "depth_pitch=%d motion_pitch=%d motion_channels=%d ms=%.9g,%.9g mo=%.9g,%.9g out_half_pitch=%d",
                  capture_run_, (unsigned long long)job.value, job.slot, fr.rw, fr.rh, fr.jitter[0], fr.jitter[1],
                  fr.mv_scale[0], fr.mv_scale[1], fr.mv_bias[0], fr.mv_bias[1], fr.exposure, fr.pre_exposure, fr.reset,
                  job.exposure_mode, (int)job.raw_inputs, job.color_pitch, job.depth_pitch, job.motion_pitch,
                  job.motion_channels, job.ms[0], job.ms[1], job.mo[0], job.mo[1], job.out_half_pitch);
    // motion at output resolution: the raw motion file <base>_motion_hi.bin and how the network read it
    if (job.motion_hi && len > 0 && len < (int)sizeof line)
      len += std::snprintf(line + len, sizeof line - (size_t)len,
                           " motion_outres=%s motion_hi=%dx%d motion_hi_pitch=%d mv_scale_hi=%.9g,%.9g b_mv=%.9g,%.9g",
                           job.motion_hi_use ? "original" : "point", job.mw, job.mh, job.motion_hi_pitch,
                           job.mv_scale_hi[0], job.mv_scale_hi[1], job.b_mv[0], job.b_mv[1]);
    if (len > 0 && len < (int)sizeof line - 1) line[len] = '\n', line[len + 1] = 0;
    c.meta = line;
    c.busy = true;
  }

  void release_partial(wgrt::Device* d) {
    for (auto& m : imports_)
      if (!m.kept) d->release_memory(m.mem, m.ptr);
    imports_.clear();
    for (void* f : {ready_, done_, pipe_})
      if (f) d->release_fence(f);
    ready_ = done_ = pipe_ = nullptr;
  }

  WgRtConfig cfg_;
  std::function<void(bool, const std::string&)> log_;
  std::string scope_;  // the device's import scope (prepare)
  std::unique_ptr<wgrt::Device> dev_;  // from prepare until init hands it to the runtime
  std::string adapter_, device_name_, note_;
  wgrt::Runtime* rt_ = nullptr;
  std::vector<Mem> imports_;
  wgrt::DevPtr ptrs_[2][6] = {};  // color, depth, motion, expo, out, motion_hi (0 without that buffer)
  uint64_t bytes_[2][6] = {};
  std::string capture_dir_;
  int capture_frames_ = 30, capture_left_ = 0, capture_run_ = 0;
  float guard_rate_ = 0.0f, guard_thr_ = 0.0f;
  wgrt::DevPtr guard_prev_[2] = {}, guard_pdepth_ = 0;
  size_t guard_pcap_ = 0, guard_dcap_ = 0;
  int guard_cur_ = 0, guard_ow_ = 0, guard_oh_ = 0, guard_rw_ = 0, guard_rh_ = 0;
  bool guard_valid_ = false;
  CapSlot cap_[kCapRing];
  bool events_reserved_ = false;
  // timing log: events (value % kRing) * kEvents + 0..4 = the runtime's marks, + 5 = after the done signal
  // Match TimingLog's eight-entry D3D12 ring. With four frames in flight,
  // submit(v) reused v-4 before dispatch_timed harvested its completed events.
  // Eight slots keep them alive through that harvest; 48 events remain below
  // the independent capture events beginning at kCapEvent0 (64).
  static constexpr int kRing = kTimingRing, kEvents = 6;
  static_assert(kRing * kEvents <= kCapEvent0, "timing and capture events must not overlap");
  bool timing_ = false;
  wgrt::DevPtr flag_ = 0;
  bool ready_flag_ = false;
  bool done_flag_ = false;
  bool staged_out_ = false;  // option handover=staged
  uint64_t tagged_[kRing] = {};  // frame value whose events are complete in that ring entry (0 = none)
  void* ready_ = nullptr;
  void* done_ = nullptr;
  void* pipe_ = nullptr;
};

}  // namespace

std::unique_ptr<Executor> make_hip_executor(const WgRtConfig& cfg, const std::string& adapter) {
  return std::unique_ptr<Executor>(new HipExecutor(cfg, adapter));
}

}  // namespace wgi
