// SPDX-License-Identifier: MIT
// D3D12 executor (executor=dx12; executor.h): Model M on the native D3D12 device (clean/runtime/device_d3d12.h) made on
// the GAME's D3D12 device, recorded into the game's command list. WrongGPU's own code.
//
// Per frame, on the game's list L (backend.cpp dispatch_record), with no input or output conversion pass: the network's
// direct-resource kernels read the game's colour, depth, motion (and exposure) textures through typed SRVs of the views
// the Converter would use (srv_format), and the post kernel writes the game's output texture through a typed UAV
// (rs=direct, clean/dx12/DIRECT_RESOURCE_STATUS.md). The bytes are those of the HIP executor's path:
//  - RGBA16F output with the output copy (default): the HIP output-copy rows (framebuffer_half: +-65504 clamp, round to
//    nearest even), NativeDirectParams.output_rne16; an output texture without unordered access gets them through an
//    RGBA16F texture of ours and CopyTextureRegion, as the HIP path copies its rows;
//  - other formats (and output_copy=0): the Converter output pass's typed store of the same FP32 values; sRGB formats
//    (no typed UAV store) take an RGBA32F texture of ours and the Converter's pass on it (record_output_tex).
// The game's texture states are transitioned as needed and restored. No command-list split, no fences, no second
// queue, no shared handles, no HIP: the network runs in the game's own submission order. A completion marker
// (WriteBufferImmediate) tells when the game's lists have finished (descriptor-set reuse, timing, teardown).
// Create: the device (its driver-channel probe on the game's device), the runtime twice (the first only measures how
// much arena memory the network needs; the second gets an arena of exactly that size), every pipeline state.
// Motion at output resolution (create flag WG_CREATE_MOTION_OUTPUT_RES, model M; the runtime option motion_outres the
// backend passes): original = the D3D12 ports of the emo / pmo kernels read the game's motion texture as the HIP path's
// motionhi pass does (FrameIo.motion_hi_direct; the texture's size and FP16-ness in b1 word 15); point = a D3D12 pass
// (Converter::record_motion_point, the input pass's own texel choice) writes the texel under each render pixel's centre
// into an RG32F texture of ours, which the network then reads as its render-size motion texture.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <deque>
#include <string>
#include <utility>
#include <vector>

#include "../device_d3d12.h"
#include "../runtime_core.h"
#include "convert.h"
#include "executor.h"
#include "texture_shape.h"

namespace wgi {

namespace {

template <class T>
void release_ref(T*& p) {
  if (p) p->Release();
  p = nullptr;
}

class D3d12Executor : public Executor {
 public:
  explicit D3d12Executor(const WgRtConfig& cfg) : cfg_(cfg) {}

  ~D3d12Executor() override {
    if (rt_) {
      drain();
      wgrt::runtime_device(rt_)->begin_teardown();
      wgrt::runtime_destroy(rt_);
    }
    release_ref(proxy16_);
    release_ref(proxy32_);
    release_ref(mpoint_);
    for (ID3D12Resource*& r : retired_) release_ref(r);
    release_ref(queue_);
    release_ref(device_);
  }

  const char* name() const override { return "dx12"; }
  bool needs_handles() const override { return false; }
  void set_log_sink(std::function<void(bool, const std::string&)> sink) override { log_ = std::move(sink); }

  bool prepare_device(ID3D12Device* given, std::string* err) override {
    // motion at output resolution: the last motion_outres of the runtime options (the backend appends it with the flag)
    {
      const std::string opts = cfg_.runtime_options ? cfg_.runtime_options : "";
      for (size_t a = 0; a <= opts.size();) {
        size_t b = opts.find(';', a);
        if (b == std::string::npos) b = opts.size();
        const std::string kv = opts.substr(a, b - a);
        if (kv == "motion_outres=original") mo_mode_ = 1;
        else if (kv == "motion_outres=point") mo_mode_ = 2;
        a = b + 1;
      }
    }
    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;  // create-time work only (model upload, probe); frames use the game's list
    if (FAILED(given->CreateCommandQueue(&qd, __uuidof(ID3D12CommandQueue), (void**)&queue_)))
      return *err = "D3D12 executor: CreateCommandQueue failed", false;
    // The device our objects really belong to: the given one, or the one under a wrapper an interposer layer handed to
    // the front end (the game's textures and lists report that device too).
    if (FAILED(queue_->GetDevice(__uuidof(ID3D12Device), (void**)&device_)) || !device_)
      return *err = "D3D12 executor: the queue's device is unknown", false;
    ID3D12Device* device = device_;
    WgRtConfig c = cfg_;
    c.backend = 3;
    const std::string build = cfg_.build_dir ? cfg_.build_dir : ".";
    // Pass 1 measures the arena the network needs (the default arena is large); pass 2 makes the same allocations in
    // an arena of exactly that size. The allocation sequence of runtime_create is deterministic.
    uint64_t need = 0;
    {
      std::string e;
      std::unique_ptr<wgrt::Device> dev = wgrt::make_d3d12_device_on(device, queue_, build, &e);
      if (!dev) return *err = "D3D12 device: " + e, false;
      wgrt::Runtime* rt = wgrt::runtime_create(c, std::move(dev), &e, true, true);
      if (!rt) return *err = "D3D12 runtime: " + e, false;
      need = wgrt::d3d12_ext(wgrt::runtime_device(rt))->arena_high_water();
      wgrt::runtime_destroy(rt);
    }
    std::string e;
    std::unique_ptr<wgrt::Device> dev = wgrt::make_d3d12_device_on(device, queue_, build, &e, need + (1u << 20));
    if (!dev) return *err = "D3D12 device: " + e, false;
    if (log_) dev->set_log_sink([sink = log_](bool warning, const std::string& m) { sink(warning, m); });
    device_name_ = dev->name();
    note_ = dev->note();
    rt_ = wgrt::runtime_create(c, std::move(dev), err, true, true);
    if (!rt_) return *err = "D3D12 runtime: " + *err, false;
    ext_ = wgrt::d3d12_ext(wgrt::runtime_device(rt_));
    if (!ext_->warm_up(&e) || !wgrt::runtime_device(rt_)->sync())
      return *err = "D3D12 executor: " + (e.empty() ? wgrt::runtime_device(rt_)->err() : e), false;
    return true;
  }

  bool init(ID3D12Device*, const SharedSet&, std::string* err) override {
    if (!rt_) return *err = "D3D12 executor: not prepared", false;
    // The largest frame of this context that the direct-resource kernels can take must pass. Games with dynamic
    // resolution (e.g. Cyberpunk 2077) declare max render = output size; native AA and every ratio between 1.0 and 1.5
    // run in the output kernel post_tf4_dl_* (Brief 15), so such a context passes at its declared maximum. When the
    // declared maximum is refused for another reason, the context is checked at the largest render size of the 1.5x
    // (Quality) ratio instead; frames above that are refused per frame ("frame not taken") and get the per-frame fallback.
    auto check = [&](int rw, int rh) {
      WgRtFrame fr = {};
      fr.rw = rw;
      fr.rh = rh;
      fr.mv_scale[0] = fr.mv_scale[1] = 1.0f;
      fr.exposure = 1.0f;
      fr.pre_exposure = 1.0f;
      fr.reset = 1;
      wgrt::FrameIo io;
      io.native_direct = true;
      if (mo_mode_ == 1) {  // (sizes only: the preflight reads no texture)
        io.motion_hi_direct = true;
        io.motion_hi_w = cfg_.ow;
        io.motion_hi_h = cfg_.oh;
        io.motion_hi_pitch = 4 * cfg_.ow;
      }
      for (int mode = 0; mode < 3; ++mode) {
        io.exposure_mode = mode;
        if (!wgrt::runtime_direct_preflight(rt_, fr, &io)) return false;
      }
      return true;
    };
    if (check(cfg_.max_rw, cfg_.max_rh)) return true;
    const std::string first = wgrt::runtime_error(rt_);
    const int qw = std::min(cfg_.max_rw, (int)(cfg_.ow / 1.5)), qh = std::min(cfg_.max_rh, (int)(cfg_.oh / 1.5));
    if (qw > 0 && qh > 0 && (qw < cfg_.max_rw || qh < cfg_.max_rh) && check(qw, qh)) {
      note_ += (note_.empty() ? "" : "\n") + std::string("executor dx12: declared max render ") +
               std::to_string(cfg_.max_rw) + "x" + std::to_string(cfg_.max_rh) + " is refused (" + first +
               "); checked at " + std::to_string(qw) + "x" + std::to_string(qh) +
               " (1.5x ratio) instead; frames above that are not taken by the D3D12 executor";
      return true;
    }
    return *err = first, false;
  }

  bool submit(const Job&, bool* done_queued, std::string* err) override {
    *done_queued = false;
    *err = "the D3D12 executor records into the game's command list (record)";
    return false;
  }

  bool records_into_list() const override { return true; }

  bool record(const Job& job, ID3D12GraphicsCommandList* list, bool* output_written, std::string* err) override {
    *output_written = false;
    if (!job.game) return *err = "D3D12 executor: no game frame", false;
    const WgFrame& f = *job.game;
    wgrt::FrameIo io;
    io.native_direct = true;
    io.exposure_mode = job.exposure_mode;
    if (mo_mode_ == 1) {  // motion at output resolution, original: M = O texels of the game's motion texture
      io.motion_hi_direct = true;
      io.motion_hi_w = cfg_.ow;
      io.motion_hi_h = cfg_.oh;
      io.motion_hi_pitch = 4 * cfg_.ow;
      motion_hi_params(f, (uint32_t)cfg_.ow, (uint32_t)cfg_.oh, io.mv_scale_hi, io.b_mv);
    }
    const int tb = timing_ ? (int)(job.value % kRing) * kEvents : -1;
    io.timing_base = tb;
    if (tb >= 0) tagged_[job.value % kRing] = 0;
    if (!wgrt::runtime_direct_preflight(rt_, job.frame, &io))
      return *err = "frame not taken (nothing recorded): " + wgrt::runtime_error(rt_), false;
    // The direct kernels address the game's textures by (x, y) inside the render rectangle, with the frame's render size
    // as the dense width (native_direct.hlsli nd_*_xy): the rectangle must be the frame's render size.
    if (job.frame.rw != (int)f.render_width || job.frame.rh != (int)f.render_height)
      return *err = "frame not taken (nothing recorded): render rectangle differs from the frame's render size", false;

    // The output route (see the file comment) and the texture the network's post kernel writes.
    auto* out = static_cast<ID3D12Resource*>(f.output.resource);
    if (!out) return *err = "frame not taken (nothing recorded): no output texture", false;
    const D3D12_RESOURCE_DESC od = out->GetDesc();
    const DXGI_FORMAT ofmt = (DXGI_FORMAT)f.output.format;
    const bool uav = (od.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) != 0;
    std::string why;
    const bool rows16 = job.output_copy && Converter::output_copy_ok(f, &why);
    enum { kDirect, kProxy16, kProxy32 } route = kDirect;
    if (rows16) route = uav ? kDirect : kProxy16;
    else route = (!is_srgb(ofmt) && uav) ? kDirect : kProxy32;
    ID3D12Resource* target = out;
    DXGI_FORMAT tfmt = uav_format(ofmt);
    if (route == kProxy16 && !proxy(&proxy16_, DXGI_FORMAT_R16G16B16A16_FLOAT, f, err)) return false;
    if (route == kProxy32 && !proxy(&proxy32_, DXGI_FORMAT_R32G32B32A32_FLOAT, f, err)) return false;
    if (route == kProxy16) target = proxy16_, tfmt = DXGI_FORMAT_R16G16B16A16_FLOAT;
    if (route == kProxy32) target = proxy32_, tfmt = DXGI_FORMAT_R32G32B32A32_FLOAT;
    // motion_outres=point: our render-size motion texture (texture coordinates as the game's colour: render_x + x)
    const bool point = mo_mode_ == 2;
    if (point && !job.conv) return *err = "frame not taken (nothing recorded): no Converter for the motion pass", false;
    if (point && !point_texture(f.render_x + f.render_width, f.render_y + f.render_height, err)) return false;
    route_ = route == kDirect ? (rows16 ? "written by the network's post kernel (RGBA16F, the output-copy bytes)"
                                        : "written by the network's post kernel (typed store, as the output pass)")
             : route == kProxy16 ? "the network's RGBA16F texture + CopyTextureRegion (output without unordered access)"
                                 : "the network's RGBA32F texture + output pass (sRGB or no unordered access)";

    wgrt::NativeDirectFrame df;
    df.color = static_cast<ID3D12Resource*>(f.color.resource);
    df.depth = static_cast<ID3D12Resource*>(f.depth.resource);
    df.motion = point ? mpoint_ : static_cast<ID3D12Resource*>(f.motion.resource);
    df.exposure = job.exposure_mode == 1 ? static_cast<ID3D12Resource*>(f.exposure.resource) : nullptr;
    df.output = target;
    df.color_format = srv_format((DXGI_FORMAT)f.color.format);
    df.depth_format = srv_format((DXGI_FORMAT)f.depth.format);
    df.motion_format = point ? DXGI_FORMAT_R32G32_FLOAT : srv_format((DXGI_FORMAT)f.motion.format);
    df.exposure_format = df.exposure ? srv_format((DXGI_FORMAT)f.exposure.format) : DXGI_FORMAT_R32_FLOAT;
    df.output_format = tfmt;
    df.params.rx = f.render_x;
    df.params.ry = f.render_y;
    df.params.rw = f.render_width;
    df.params.rh = f.render_height;
    df.params.msx = f.motion_scale[0];
    df.params.msy = f.motion_scale[1];
    df.params.mox = f.motion_offset[0];
    df.params.moy = f.motion_offset[1];
    df.params.color_pi = wgrt::runtime_effective_pre_exposure(rt_, f.pre_exposure);
    df.params.exposure_mode = (uint32_t)job.exposure_mode;
    df.params.output_rne16 = rows16 ? 1u : 0u;
    // native_direct.hlsli nd_color_f16 (b1 word 14, wg_f16lut_in.hlsli): every colour texel value is an FP16 value
    // (padding[0..2] = words 11..13: the pre-exposure division window, written by bind_direct_frame)
    df.params.padding[3] = df.color_format == DXGI_FORMAT_R16G16B16A16_FLOAT || df.color_format == DXGI_FORMAT_R11G11B10_FLOAT ||
                                   df.color_format == DXGI_FORMAT_R16G16_FLOAT || df.color_format == DXGI_FORMAT_R16_FLOAT
                               ? 1u : 0u;
    // wg_motion_hi.hlsli (original): b1 word 15 = FP16 view (bit 0) | (width - 1) << 1 | (height - 1) << 16, the clamp
    // size of the motionhi pass (the WgTexture size, the region's size when not given)
    if (mo_mode_ == 1) {
      const uint32_t cw = f.motion.width ? f.motion.width : (uint32_t)cfg_.ow;
      const uint32_t ch = f.motion.height ? f.motion.height : (uint32_t)cfg_.oh;
      if (cw > 32768 || ch > 32768) return *err = "frame not taken (nothing recorded): motion texture too large", false;
      df.params.padding[4] = (Converter::motion_fp16(f) ? 1u : 0u) | ((cw - 1u) << 1) | ((ch - 1u) << 16);
    }

    std::string e;
    if (!ext_->begin_external(list, &e)) return *err = "D3D12 executor: " + e, false;
    bool ok = ext_->bind_direct_frame(df, &e);
    bool restore = false;
    if (ok) {
      ok = ext_->record_pass(D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                             [&](ID3D12GraphicsCommandList* l, std::string& m) {
                               states(l, f, target, false);
                               restore = true;
                               if (point) {  // the motion texel under each render pixel, then readable by the network
                                 if (!job.conv->record_motion_point(l, f, mpoint_, &m)) return false;
                                 D3D12_RESOURCE_BARRIER b[2] = {};
                                 b[0].Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
                                 b[0].UAV.pResource = mpoint_;
                                 b[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                                 b[1].Transition.pResource = mpoint_;
                                 b[1].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                                 b[1].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                                 b[1].Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                                 l->ResourceBarrier(2, b);
                                 point_srv_ = true;
                               }
                               return true;
                             },
                             &e, "dx12_input_states");
    }
    if (ok && !wgrt::runtime_frame(rt_, job.frame, &io)) ok = false, e = "network frame failed: " + wgrt::runtime_error(rt_);
    if (restore) {  // also after a failed frame: the game's textures go back to the states it reported
      std::string e2;
      const bool r = ext_->record_pass(
          D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
          [&](ID3D12GraphicsCommandList* l, std::string& m) {
            states(l, f, target, true);
            if (point_srv_) {  // our motion texture back to unordered access for the next frame's pass
              D3D12_RESOURCE_BARRIER b = {};
              b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
              b.Transition.pResource = mpoint_;
              b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
              b.Transition.StateBefore = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
              b.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
              l->ResourceBarrier(1, &b);
              point_srv_ = false;
            }
            if (!ok) return true;
            if (route == kProxy16) copy16(l, f);
            if (route == kProxy32 && job.conv && !job.conv->record_output_tex(l, f, proxy32_, &m)) return false;
            return true;
          },
          &e2, "dx12_output");
      if (!r && ok) ok = false, e = e2;
    }
    {
      std::string e3;
      if (!ext_->end_external(&e3) && ok) ok = false, e = e3;
    }
    serials_.push_back({ext_->external_recorded(), job.value});
    while (serials_.size() > 64) serials_.pop_front();
    if (!ok) return *err = "D3D12 executor: " + e, false;
    *output_written = true;
    if (tb >= 0) tagged_[job.value % kRing] = job.value;
    return true;
  }

  uint64_t completed() const override {
    const uint64_t done = ext_ ? ext_->external_done() : 0;
    uint64_t v = done_value_;
    for (const auto& sv : serials_)
      if (sv.first <= done && sv.second > v) v = sv.second;
    done_value_ = v;
    return v;
  }

  void drain() override {
    if (!ext_) return;
    // the game's lists that hold our work: their completion markers (bounded: a list never submitted cannot finish)
    for (int i = 0; i < 5000 && ext_->external_done() < ext_->external_recorded(); ++i) Sleep(1);
    if (ext_->external_done() < ext_->external_recorded() && log_ && !drain_warned_) {
      drain_warned_ = true;
      log_(true, "the game's command lists with the network's last frames did not finish within 5 s (" +
                     std::to_string(ext_->external_done()) + " of " + std::to_string(ext_->external_recorded()) + ")");
    }
  }

  bool enable_timing() override {
    if (!rt_ || !wgrt::runtime_device(rt_)->reserve_events(kRing * kEvents)) return false;
    timing_ = true;
    return true;
  }
  std::string timing_info() const override {
    return "D3D12 timestamps in the game's command list: [0] start of the network -> [1] start of the first Swin "
           "kernel -> [2] start of dec0 -> [3] start of post -> [4] after the last kernel";
  }
  bool timing(uint64_t value, NetTiming* t) override {
    if (!timing_ || tagged_[value % kRing] != value) return false;
    wgrt::Device* d = wgrt::runtime_device(rt_);
    const int b = (int)(value % kRing) * kEvents;
    if (d->event_done(b + 4) != 1) return false;
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

  std::string kernels() const override { return rt_ ? std::string(wgrt::runtime_kernels(rt_)) : std::string(); }
  std::string notes() const override { return rt_ ? std::string(wgrt::runtime_notes(rt_)) : std::string(); }
  uint64_t memory_bytes() const override {
    uint64_t b = ext_ ? ext_->arena_capacity() : 0;
    if (mpoint_) b += (uint64_t)mpoint_w_ * mpoint_h_ * 8;
    if (proxy16_) b += (uint64_t)cfg_.ow * cfg_.oh * 8;
    if (proxy32_) b += (uint64_t)cfg_.ow * cfg_.oh * 16;
    return b;
  }
  std::string describe() const override {
    return "D3D12 device \"" + device_name_ + "\" (the game's; " + note_ + "; direct resources: no input or output "
           "conversion pass)";
  }
  std::string output_route() const { return route_; }

 private:
  // Our texture for the routes the post kernel cannot write the game's output directly (made on first use).
  bool proxy(ID3D12Resource** p, DXGI_FORMAT fmt, const WgFrame& f, std::string* err) {
    if (*p) return true;
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Width = f.output_width;
    rd.Height = f.output_height;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.Format = fmt;
    rd.SampleDesc.Count = 1;
    rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    if (FAILED(device_->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                                nullptr, __uuidof(ID3D12Resource), (void**)p)))
      return *err = "D3D12 executor: output texture of the network could not be made", false;
    return true;
  }

  // motion_outres=point: our RG32F motion texture of at least w x h (UNORDERED_ACCESS between frames). A larger frame
  // makes a new one; the old one may still be in use by the game's lists and is released with the executor.
  bool point_texture(uint32_t w, uint32_t h, std::string* err) {
    if (mpoint_ && w <= mpoint_w_ && h <= mpoint_h_) return true;
    if (mpoint_) retired_.push_back(mpoint_), mpoint_ = nullptr;
    w = std::max(w, mpoint_w_);
    h = std::max(h, mpoint_h_);
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Width = w;
    rd.Height = h;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.Format = DXGI_FORMAT_R32G32_FLOAT;
    rd.SampleDesc.Count = 1;
    rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    if (FAILED(device_->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                                nullptr, __uuidof(ID3D12Resource), (void**)&mpoint_)))
      return *err = "D3D12 executor: the motion texture of motion_outres=point could not be made", false;
    mpoint_w_ = w;
    mpoint_h_ = h;
    return true;
  }

  // The game's textures into the states the kernels use (inputs: NON_PIXEL_SHADER_RESOURCE, the output target:
  // UNORDERED_ACCESS) and back (restore; a UAV barrier on the target first, the game reads it after us).
  void states(ID3D12GraphicsCommandList* l, const WgFrame& f, ID3D12Resource* target, bool restore) {
    D3D12_RESOURCE_BARRIER b[6] = {};
    UINT n = 0;
    if (restore) {
      b[n].Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
      b[n++].UAV.pResource = target;
    }
    const bool ownTarget = target != f.output.resource;
    struct T { ID3D12Resource* r; uint32_t state; D3D12_RESOURCE_STATES want; } t[5] = {
        {static_cast<ID3D12Resource*>(f.color.resource), f.color.state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE},
        {static_cast<ID3D12Resource*>(f.depth.resource), f.depth.state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE},
        {static_cast<ID3D12Resource*>(f.motion.resource), f.motion.state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE},
        {f.exposure_mode == WG_EXPOSURE_TEXTURE ? static_cast<ID3D12Resource*>(f.exposure.resource) : nullptr,
         f.exposure.state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE},
        {ownTarget ? nullptr : target, f.output.state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS}};
    for (int i = 0; i < 5; ++i) {
      if (!t[i].r || (t[i].state & t[i].want) == (uint32_t)t[i].want) continue;
      bool dup = false;
      for (int j = 0; j < i; ++j) dup |= t[j].r == t[i].r;
      if (dup) continue;
      auto& x = b[n++];
      x.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
      x.Transition.pResource = t[i].r;
      x.Transition.Subresource = barrier_subresource(t[i].r);  // mip 0 only of a texture with mips (texture_shape.h)
      x.Transition.StateBefore = restore ? t[i].want : (D3D12_RESOURCE_STATES)t[i].state;
      x.Transition.StateAfter = restore ? (D3D12_RESOURCE_STATES)t[i].state : t[i].want;
    }
    if (n) l->ResourceBarrier(n, b);
  }

  // kProxy16: our RGBA16F texture (the output-copy bytes) into the game's output, as the HIP path's CopyTextureRegion.
  void copy16(ID3D12GraphicsCommandList* l, const WgFrame& f) {
    auto* out = static_cast<ID3D12Resource*>(f.output.resource);
    D3D12_RESOURCE_BARRIER b[2] = {};
    UINT n = 0;
    b[n].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b[n].Transition.pResource = proxy16_;
    b[n].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b[n].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    b[n++].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    if (!(f.output.state & D3D12_RESOURCE_STATE_COPY_DEST)) {
      b[n].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
      b[n].Transition.pResource = out;
      b[n].Transition.Subresource = barrier_subresource(out);  // the copy writes subresource 0 (texture_shape.h)
      b[n].Transition.StateBefore = (D3D12_RESOURCE_STATES)f.output.state;
      b[n++].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
    }
    l->ResourceBarrier(n, b);
    D3D12_TEXTURE_COPY_LOCATION dst = {}, src = {};
    dst.pResource = out;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src.pResource = proxy16_;
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    D3D12_BOX box = {0, 0, 0, f.output_width, f.output_height, 1};
    l->CopyTextureRegion(&dst, 0, 0, 0, &src, &box);
    for (UINT i = 0; i < n; ++i) std::swap(b[i].Transition.StateBefore, b[i].Transition.StateAfter);
    l->ResourceBarrier(n, b);
  }

  WgRtConfig cfg_;
  std::function<void(bool, const std::string&)> log_;
  ID3D12Device* device_ = nullptr;  // the game's (under a possible wrapper; a reference of ours)
  ID3D12CommandQueue* queue_ = nullptr;
  wgrt::Runtime* rt_ = nullptr;
  wgrt::NativeD3D12Ext* ext_ = nullptr;
  ID3D12Resource *proxy16_ = nullptr, *proxy32_ = nullptr;
  int mo_mode_ = 0;  // motion at output resolution: 0 none, 1 original, 2 point (runtime option motion_outres)
  ID3D12Resource* mpoint_ = nullptr;  // point: our render-size RG32F motion texture
  uint32_t mpoint_w_ = 0, mpoint_h_ = 0;
  bool point_srv_ = false;  // this frame moved it to NON_PIXEL_SHADER_RESOURCE (the restore pass moves it back)
  std::vector<ID3D12Resource*> retired_;  // outgrown point textures
  std::string device_name_, note_, route_;
  std::deque<std::pair<uint64_t, uint64_t>> serials_;  // (completion marker serial, frame value) of recorded frames
  mutable uint64_t done_value_ = 0;
  bool drain_warned_ = false;
  static constexpr int kRing = kTimingRing, kEvents = 5;
  bool timing_ = false;
  uint64_t tagged_[kRing] = {};
};

}  // namespace

std::unique_ptr<Executor> make_d3d12_executor(const WgRtConfig& cfg) {
  return std::unique_ptr<Executor>(new D3d12Executor(cfg));
}

std::string d3d12_executor_output_route(const Executor* e) {
  auto* x = dynamic_cast<const D3d12Executor*>(e);
  return x ? x->output_route() : std::string();
}

}  // namespace wgi
