// SPDX-License-Identifier: MIT
// Small per-frame helper kernels of the runtime (spec 10 draft 2, section 5 step 1: pre-exposure).
//
// color_divide: the network works on color / pi: out[i] = in[i] / pi per channel (IEEE division), for the stages that
//   read the color (input encoder, output reconstruction). Exposure (spec 07) reads the original color with pi.
// scale_rgb:    the output filter's result is multiplied by pi: buf[i] *= pi.
// Both are skipped when pi = 1.
// input_convert: interop input copy (see below).
// sparkle_guard: WrongGPU's own temporal guard on the result (see below).
// timing_barrier: empty, one wave; only with the timing log: launched before each timing event so that the event is
//   ordered after the completion of the previous kernel on the stream (a bare event can be stamped earlier).
#include "wg_platform.h"

WG_KERNEL_N(color_divide, 256)(const float* in, float* out, float pi, int n) {
  const int i = (int)(wg_block_id() * 256 + wg_lane_id());
  if (i >= n) return;
  const v4f c = ((const v4f*)in)[i];
  v4f r;
  r[0] = c[0] / pi;
  r[1] = c[1] / pi;
  r[2] = c[2] / pi;
  r[3] = c[3];
  ((v4f*)out)[i] = r;
}

WG_KERNEL_N(scale_rgb, 256)(float* buf, float pi, int n) {
  const int i = (int)(wg_block_id() * 256 + wg_lane_id());
  if (i < n) buf[i] = buf[i] * pi;
}

WG_KERNEL_N(timing_barrier, 32)(int unused) { (void)unused; }

// Interop input copy (option input_copy=1): the game's textures copied as they are into pitched rows (color RGBA16F,
// depth R32F, motion with mch FP16 channels; pitches in bytes) become the runtime's dense inputs: color / pi (IEEE
// division per channel, as color_divide; alpha kept), depth, and motion m = ms * raw + mo in render pixels (as the
// D3D12 input pass). One lane per render pixel, 256-lane workgroups.
WG_KERNEL_N(input_convert, 256)(const _Float16* color, int cpitch, const float* depth, int dpitch,
                                const _Float16* motion, int mpitch, int mch, float msx, float msy, float mox,
                                float moy, float pi, float* cnet, float* dep, float* mot, int rw, int rh) {
  const int i = (int)(wg_block_id() * 256 + wg_lane_id());
  if (i >= rw * rh) return;
  const int y = i / rw, x = i - y * rw;
  const _Float16* c = (const _Float16*)((const char*)color + (size_t)y * cpitch) + 4 * x;
  v4f r;
  r[0] = (float)c[0] / pi;
  r[1] = (float)c[1] / pi;
  r[2] = (float)c[2] / pi;
  r[3] = (float)c[3];
  ((v4f*)cnet)[i] = r;
  dep[i] = ((const float*)((const char*)depth + (size_t)y * dpitch))[x];
  const _Float16* m = (const _Float16*)((const char*)motion + (size_t)y * mpitch) + mch * x;
  mot[2 * i] = msx * (float)m[0] + mox;
  mot[2 * i + 1] = msy * (float)m[1] + moy;
}

// Sparkle guard (WrongGPU's own addition, not part of the model; backend option sparkle_guard=<rate>, off by default).
// Model M can make a few pixels flash for a single frame on fine, bright detail. This pass lets a pixel brighten only
// gradually: with t = Y / (1 + Y) of the Rec. 709 luma Y of the result and tp the guarded t of the previous frame at
// the motion-compensated position, d = t - tp; where d > thr the pixel gets t' = tp + thr + rate (d - thr) and its
// colour is scaled by Y'/Y (hue kept). Darkening and brightening below thr pass unchanged, so nearly all pixels are
// only read. The previous position is the pixel + the game's motion vector (render pixels, previous = current + m,
// scaled to output pixels), tp sampled bilinearly; pixels whose reprojected depth differs by more than 2 %
// (disocclusion), whose source lies outside the image, and every frame after a reset or a size change (valid = 0) are
// not guarded. One lane per output pixel, 256-lane workgroups.
//   out16 / opitch: RGBA16F rows of opitch bytes (output_copy path); else out32: [oh][ow] 3 x f32.
//   prev_in / prev_out: the guarded t of the previous / this frame, [oh][ow] FP16 (ping-pong buffers).
//   motion [rh][rw] 2 x f32, depth / pdepth [rh][rw] f32: this frame's inputs and the previous frame's depth.
WG_FN int sg_imin(int a, int b) { return a < b ? a : b; }

WG_KERNEL_N(sparkle_guard, 256)(_Float16* out16, int opitch, float* out32, const _Float16* prev_in,
                                _Float16* prev_out, const float* motion, const float* depth, const float* pdepth,
                                int ow, int oh, int rw, int rh, float rate, float thr, int valid) {
  const u32 i = wg_block_id() * 256u + wg_lane_id();
  if (i >= (u32)ow * (u32)oh) return;
  const int y = (int)(i / (u32)ow), x = (int)(i - (u32)y * (u32)ow);
  _Float16* o16 = out16 ? (_Float16*)((char*)out16 + (size_t)y * (unsigned)opitch) + 4 * x : nullptr;
  float* o32 = out16 ? nullptr : out32 + 3 * (size_t)i;
  float c[3];
  for (int k = 0; k < 3; ++k) c[k] = o16 ? (float)o16[k] : o32[k];
  const float Y = __builtin_fmaxf(0.2126f * c[0] + 0.7152f * c[1] + 0.0722f * c[2], 0.0f);
  const float t = Y / (1.0f + Y);
  float tn = t;
  if (valid) {
    const int rx = sg_imin((int)(((float)x + 0.5f) * (float)rw / (float)ow), rw - 1);
    const int ry = sg_imin((int)(((float)y + 0.5f) * (float)rh / (float)oh), rh - 1);
    const float mx = motion[2 * ((size_t)ry * rw + rx)], my = motion[2 * ((size_t)ry * rw + rx) + 1];
    const float px = (float)x + mx * (float)ow / (float)rw, py = (float)y + my * (float)oh / (float)rh;
    const int prx = (int)__builtin_floorf((float)rx + 0.5f + mx), pry = (int)__builtin_floorf((float)ry + 0.5f + my);
    bool ok = px >= 0.0f && py >= 0.0f && px <= (float)(ow - 1) && py <= (float)(oh - 1) && prx >= 0 && pry >= 0 &&
              prx < rw && pry < rh;
    if (ok) {
      const float dc = depth[(size_t)ry * rw + rx], dp = pdepth[(size_t)pry * rw + prx];
      ok = __builtin_fabsf(dp - dc) <= 0.02f * __builtin_fmaxf(__builtin_fabsf(dc), 1e-6f);
    }
    if (ok) {
      const int x0 = sg_imin((int)px, ow - 2), y0 = sg_imin((int)py, oh - 2);
      const float fx = px - (float)x0, fy = py - (float)y0;
      const _Float16* r0 = prev_in + (size_t)y0 * ow + x0;
      const _Float16* r1 = r0 + ow;
      const float tp = ((float)r0[0] * (1.0f - fx) + (float)r0[1] * fx) * (1.0f - fy) +
                       ((float)r1[0] * (1.0f - fx) + (float)r1[1] * fx) * fy;
      const float d = t - tp;
      if (d > thr) {  // then t > tp >= 0, so Y > 0
        tn = tp + thr + rate * (d - thr);
        const float s = tn / __builtin_fmaxf(1.0f - tn, 1e-6f) / Y;
        for (int k = 0; k < 3; ++k) {
          if (o16) o16[k] = (_Float16)(c[k] * s);
          else o32[k] = c[k] * s;
        }
      }
    }
  }
  prev_out[i] = (_Float16)tn;
}

#if defined(WG_EMU)
WG_EMU_REGISTER(color_divide)
WG_EMU_REGISTER(scale_rgb)
WG_EMU_REGISTER(timing_barrier)
WG_EMU_REGISTER(input_convert)
WG_EMU_REGISTER(sparkle_guard)
#endif
