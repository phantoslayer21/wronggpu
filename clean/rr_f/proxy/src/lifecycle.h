// SPDX-License-Identifier: MIT
// Host bookkeeping of one RR feature instance (page 53; 61.6; 63.3), ported from
// clean/rr_f/reference/lifecycle.py. No image work: the runtime does that from the plan.
#pragma once
#include <stdint.h>
#include <string.h>

namespace wgrr {

struct CameraPair {
  float v2c[16];
  float w2v[16];
};

struct FramePlan {
  uint32_t render_w = 0, render_h = 0;
  bool reset = false;
  uint64_t entry_reset_generation = 0;  // 0 = no pending reset at entry
  uint32_t parity = 0, counter = 0;
  float prev_jitter[2] = {0, 0};
  float jitter[2] = {0, 0};
  CameraPair current{};
  CameraPair effective_prev{};
  bool rebuild = false;
};

class Lifecycle {
 public:
  // Creation (53.8): par = fc = 0, pj = 0, CP = zeros, a reset is pending. `render_w/h` = the render size the
  // render-size items are built for at creation (0 x 0 when not known, open item 13).
  Lifecycle(uint32_t render_w, uint32_t render_h) : built_w_(render_w), built_h_(render_h) {
    memset(&cp_, 0, sizeof cp_);
    request_reset();
  }

  bool pending_reset() const { return pending_gen_ != 0; }
  uint32_t parity() const { return par_; }
  uint32_t counter() const { return fc_; }
  const float* prev_jitter() const { return pj_; }
  const CameraPair& previous_camera() const { return cp_; }
  uint32_t built_w() const { return built_w_; }
  uint32_t built_h() const { return built_h_; }

  void request_reset() { pending_gen_ = ++gen_; }

  // Begin a frame that passed every rejection. A render-size change rebuilds the render-size items before the
  // frame and makes a reset pending for the next frame (53.9); the change frame runs with the reset status it
  // already had, unless the opt-in `change_frame_reset` makes it a reset frame itself.
  FramePlan begin(uint32_t rw, uint32_t rh, const float jitter[2], const CameraPair& cam, bool game_reset,
                  bool change_frame_reset) {
    FramePlan p;
    p.render_w = rw;
    p.render_h = rh;
    p.entry_reset_generation = pending_gen_;
    p.reset = game_reset || pending_gen_ != 0;
    if (rw != built_w_ || rh != built_h_) {
      p.rebuild = true;
      built_w_ = rw;
      built_h_ = rh;
      request_reset();
      if (change_frame_reset) {  // opt-in: this frame consumes the reset the change made pending
        p.reset = true;
        p.entry_reset_generation = pending_gen_;
      }
    }
    p.parity = par_;
    p.counter = fc_;
    p.prev_jitter[0] = pj_[0];
    p.prev_jitter[1] = pj_[1];
    p.jitter[0] = jitter[0];
    p.jitter[1] = jitter[1];
    p.current = cam;
    p.effective_prev = p.reset ? cam : cp_;
    return p;
  }

  // End of a completed evaluate (53.7).
  void complete(const FramePlan& p) {
    pj_[0] = p.jitter[0];
    pj_[1] = p.jitter[1];
    par_ ^= 1u;
    fc_ = (fc_ + 1u) % 8u;
    cp_ = p.current;
    if (p.reset && p.entry_reset_generation != 0 && pending_gen_ == p.entry_reset_generation) pending_gen_ = 0;
  }

  // A failed evaluate (61.6, 53.10): counters, CP and pending reset unchanged; pj replaced only when the input
  // stage had used it; the opt-in forces a reset on the next call.
  void fail(const FramePlan& p, bool input_jitter_consumed, bool reset_after_failure) {
    if (input_jitter_consumed) {
      pj_[0] = p.jitter[0];
      pj_[1] = p.jitter[1];
    }
    if (reset_after_failure) request_reset();
  }

 private:
  uint32_t built_w_, built_h_;
  uint32_t par_ = 0, fc_ = 0;
  float pj_[2] = {0, 0};
  CameraPair cp_;
  uint64_t gen_ = 0;
  uint64_t pending_gen_ = 0;
};

}  // namespace wgrr
