// SPDX-License-Identifier: MIT
// Model L (issue #89): output reconstruction (stage "post", HDR) as Model M's post_recon (clean/kernels/post.hip,
// included for its functions; its own kernels come along unused) with Model L's three changes
// (clean/l/reference/recon.py, the oracle):
//  1. a 4 x 4 colour gather: columns ax - 1 ... ax + 2 and rows b - 1 ... b + 2 (b from the even row of the pair);
//  2. each history candidate decoded and divided by the previous exposure before the two are combined:
//     H_lin = (w1 L1 + w2 L2) e, with L_i = D(H_i) / e_prev (L2 only when w2 > 0);
//  3. an invalid or reset sample falls back to the render pixel's linear colour (no exposure, no encode).
// The colour gather runs in the model's own precision as confirmed by the spec side (oracle: recon.gather_half):
// fused target positions, FP16 offsets and exponent with the fused cross term, float32 weights flushed to +0 below
// 2^-126, FP16 weights into half accumulators, a float32 weight sum. The blend is item 3's fused form. The decode
// passes a NaN through (a zero total gather weight gives C = D(NaN) = NaN; the NaN guard then substitutes it, as the
// oracle). Arguments, layouts and grid are post_recon's; w[0] = Model L's post scale s.
#include "post.hip"

namespace {

constexpr int kLWin = 4;  // taps -1 ... 2
typedef _Float16 h16;

// D(.) with a NaN passed through.
WG_FN float l_dec(float q) { return q != q ? q : dec_e(q); }

// One render pixel's gather inputs in the model's precision: the nine features in FP16 (h(E_in(e colour)), the B and
// A values k / 255), and A = h(a -0.721325), B = h(b -0.721325), C = h(c -1.44265) (float32 product, one FP16
// rounding).
struct LTexel {
  h16 f[9];
  h16 kA, kB, kC;
};

WG_FN LTexel l_load_texel(const v4f* color, const u32* A, const u32* B, const v4h* C, int idx, float e) {
  LTexel t;
  const v4f c = color[idx];
  t.f[0] = (h16)enc_in(e * c[0]);
  t.f[1] = (h16)enc_in(e * c[1]);
  t.f[2] = (h16)enc_in(e * c[2]);
  const u32 b = B[idx], a = A[idx];
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    t.f[3 + i] = (h16)(byte_f(b, i) / 255.0f);
    t.f[6 + i] = (h16)(byte_f(a, i) / 255.0f);
  }
  const v4h k = C[idx];
  t.kA = (h16)((float)k[0] * -0.721325f);
  t.kB = (h16)((float)k[1] * -0.721325f);
  t.kC = (h16)((float)k[2] * -1.44265f);
  return t;
}

// Gathered sums of one D pixel: W = float32 sum of the unrounded weights; F = the half accumulators.
struct LGather {
  float W;
  h16 F[9];
};

WG_FN void l_gather_zero(LGather& g) {
  g.W = 0.0f;
#pragma unroll
  for (int k = 0; k < 9; ++k) g.F[k] = (h16)0.0f;
}

// One tap: eps = fused_half(h(u v), C, h(h(h(u u) A) + h(h(v v) B))), w = 2^eps (float32, +0 below 2^-126),
// W += w, F_k = fused_half(h(w), f_k, F_k). One FP16 operation per statement (contraction is off in post.hip).
WG_FN void l_gather_add(LGather& g, const LTexel& t, h16 u, h16 v) {
  const h16 uu = u * u, vv = v * v, uv = u * v;
  const h16 ta = uu * t.kA, tb = vv * t.kB;
  const h16 s = ta + tb;
  const h16 eps = __builtin_fmaf16(uv, t.kC, s);
  float w = wg_exp2((float)eps);
  if (!(w >= 0x1p-126f)) w = 0.0f;  // flush to zero (also a NaN exponent gives 0, as 2^NaN never occurs here)
  g.W += w;
  const h16 hw = (h16)w;
#pragma unroll
  for (int k = 0; k < 9; ++k) g.F[k] = __builtin_fmaf16(hw, t.f[k], g.F[k]);
}

// Normalized features and the candidate choice (Model M's rules in value units).
struct LSel {
  float fbar[9];
  int first, second;
  float w1, w2;
};

WG_FN LSel l_select(const LGather& g) {
  LSel r;
  const float inv = wg_rcp(g.W);  // rcp(0) = inf: 0 * inf = NaN features (zero total weight)
#pragma unroll
  for (int k = 0; k < 9; ++k) r.fbar[k] = inv * (float)g.F[k];
  const float* sig = r.fbar + 3;
  int first = 0;
  float best = sig[0];
  if (sig[1] > best) { first = 1; best = sig[1]; }
  if (sig[2] > best) first = 2;
  const int o1 = first == 0 ? 1 : 0, o2 = first == 2 ? 1 : 2;
  const int second = sig[o2] > sig[o1] ? o2 : o1;
  const float q1 = clampf(1.1f * sig[first] - 0.05f, 0.0f, 1.0f);
  const float q2 = clampf(1.1f * sig[second] - 0.05f, 0.0f, 1.0f);
  const float rden = wg_rcp(wg_fmax(q1 + q2, 1e-10f));
  float w2 = q2 * rden, w1;
  if (w2 < 1.0f / 255.0f) {
    w2 = 0.0f;
    w1 = 1.0f;
  } else {
    w1 = q1 * rden;
  }
  r.first = first;
  r.second = second;
  r.w1 = w1;
  r.w2 = w2;
  return r;
}

// Steps 3-5 for one D pixel (Model L). Returns the linear new history H^ in hh and the linear display value O^ in oh.
WG_FN void l_post_core(const LGather& g, int x, int y, int fx, int fy, const float mx[3], const float my[3],
                       const WgPostParams& p, float s, const v4f* color, const void* hist_prev, float hh[3],
                       float oh[3]) {
  const LSel sl = l_select(g);
  const float X = (float)x + 0.5f, Y = (float)y + 0.5f;
  const int ncand = sl.w2 != 0.0f ? 2 : 1;
  float L[2][3];
  for (int n = 0; n < ncand; ++n) {
    const int k = n == 0 ? sl.first : sl.second;
    const float qx = X + mx[k], qy = Y + my[k];
    const bool valid = !p.reset && qx >= 0.0f && qx <= (float)p.dw && qy >= 0.0f && qy <= (float)p.dh;
    if (valid) {
      float h[3];
      history_cr<false>(hist_prev, p.dw, p.dh, qx, qy, h);
#pragma unroll
      for (int i = 0; i < 3; ++i) L[n][i] = l_dec(h[i]) / p.e_prev;
    } else {
      const v4f c = color[fy * p.rw + fx];
#pragma unroll
      for (int i = 0; i < 3; ++i) L[n][i] = c[i];
    }
  }
  float Hl[3], Cl[3];
  bool hl_inf = false;
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    const float comb = ncand == 2 ? __builtin_fmaf(sl.w1, L[0][i], sl.w2 * L[1][i]) : sl.w1 * L[0][i];
    Hl[i] = comb * p.e;
    hl_inf |= __builtin_fabsf(Hl[i]) == __builtin_inff();
    Cl[i] = l_dec(sl.fbar[i]);
  }
  // 5. Blend: alpha = sat((s W) g-bar), beta' = fused(alpha (1 - beta-bar) + beta-bar), H^ = fused((C - Hl) beta' + Hl),
  // O^ likewise (item 3). sat(NaN) = 0 (zero total weight).
  const float alpha = clampf((s * g.W) * sl.fbar[6], 0.0f, 1.0f);
  const float bh2 = __builtin_fmaf(alpha, 1.0f - sl.fbar[7], sl.fbar[7]);
  const float bo2 = __builtin_fmaf(alpha, 1.0f - sl.fbar[8], sl.fbar[8]);
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    hh[i] = __builtin_fmaf(Cl[i] - Hl[i], bh2, Hl[i]);
    oh[i] = __builtin_fmaf(Cl[i] - Hl[i], bo2, Hl[i]);
  }
  const float sh = hh[0] + hh[1] + hh[2], so = oh[0] + oh[1] + oh[2];
  if (sh != sh || hl_inf) {
#pragma unroll
    for (int i = 0; i < 3; ++i) hh[i] = Cl[i];
  }
  if (so != so || hl_inf) {
#pragma unroll
    for (int i = 0; i < 3; ++i) oh[i] = Cl[i];
  }
}

// post_recon's finish_pixel with l_post_core.
WG_FN void l_finish_pixel(const LGather& g, int x, int y, float px, float py, bool store, const WgPostParams& p,
                          float s, const v4f* color, const v4h* hist_prev, const u32* m_near, const u32* m_center,
                          const u32* m_far, v4h* hist_out, _Float16* display, float hh[3]) {
  const int fx = clampi(floor_i(px), 0, p.rw - 1), fy = clampi(floor_i(py), 0, p.rh - 1);
  float mx[3], my[3];
  unpack_h2(m_near[fy * p.rw + fx], mx[0], my[0]);
  unpack_h2(m_far[fy * p.rw + fx], mx[2], my[2]);
  motion_bilinear(px, py,
                  [&](int c, int r) { return m_center[clampi(r, 0, p.rh - 1) * p.rw + clampi(c, 0, p.rw - 1)]; },
                  mx[1], my[1]);
  float oh[3];
  l_post_core(g, x, y, fx, fy, mx, my, p, s, color, hist_prev, hh, oh);
  if (store) hist_store<false>(hist_out, y * p.dw + x, enc_e(hh[0]), enc_e(hh[1]), enc_e(hh[2]));
  if (store) {
    const int pitch = display_pitch(p.dw), plane = pitch * p.dh, di = y * pitch + x;
    display[di] = display_value(oh[0]);
    display[plane + di] = display_value(oh[1]);
    display[2 * plane + di] = display_value(oh[2]);
  }
}

}  // namespace

// Stage "post" for Model L, HDR. Arguments, layouts and grid as post_recon (clean/kernels/post.hip): grid
// ceil(dw / 32) * ceil(ceil(dh / 2) / 8) workgroups of 256 lanes; w = the post blob (w[0] = s).
WG_KERNEL_N(l_post_recon, kPostThreads)(const v4f* color, const u32* A, const u32* B, const v4h* C,
                                        const v4h* hist_prev, const u32* m_near, const u32* m_center,
                                        const u32* m_far, const float* w, v4h* hist_out, u32* reduced,
                                        _Float16* display, WgPostParams p, const float* expo) {
  if (expo) {
    p.e = expo[0] == 0.0f ? 1.0f : expo[0];
    p.e_prev = expo[1] == 0.0f ? 1.0f : expo[1];
    p.ratio = (float)((double)p.e / (double)p.e_prev);
  }
  WG_LDS LTexel sh_tex[kStageCap];
  WG_LDS float sh_red[kPostThreads * 3];
  // Target positions in the model's precision: px = fused(x + 1/2, rx, jx), py = fused(ry, y + 1/2, jy).
  const auto lpx = [&](int xx) { return __builtin_fmaf((float)xx + 0.5f, p.rx, p.jx); };
  const auto lpy = [&](int yy) { return __builtin_fmaf(p.ry, (float)yy + 0.5f, p.jy); };

  const int lane = (int)wg_lane_id();
  const int tiles_x = (p.dw + kTileX - 1) / kTileX;
  const int tile = (int)wg_block_id();
  const int x0 = (tile % tiles_x) * kTileX;
  const int k0 = (tile / tiles_x) * kTilePairs;
  const int npairs = (p.dh + 1) / 2;
  const float s = w[0];

  // Render footprint of the tile: window columns ax - 1 ... ax + 2 of its first and last column, window rows of its
  // first and last row pair.
  const int xl = mini(x0 + kTileX - 1, p.dw - 1), kl = mini(k0 + kTilePairs - 1, npairs - 1);
  const int c0 = floor_i(lpx(x0) - 0.5f) - 1, c1 = floor_i(lpx(xl) - 0.5f) + (kLWin - 2);
  const int r0 = floor_i(lpy(2 * k0) - 0.5f) - 1, r1 = floor_i(lpy(2 * kl) - 0.5f) + (kLWin - 2);
  const int ncol = c1 - c0 + 1, nrow = r1 - r0 + 1;
  const bool staged = ncol > 0 && nrow > 0 && ncol <= kStageCap && nrow <= kStageCap && ncol * nrow <= kStageCap;
  if (staged) {
    for (int i = lane; i < ncol * nrow; i += kPostThreads) {
      const int r = i / ncol, c = i - r * ncol;
      const int idx = clampi(r0 + r, 0, p.rh - 1) * p.rw + clampi(c0 + c, 0, p.rw - 1);
      sh_tex[i] = l_load_texel(color, A, B, C, idx, p.e);
    }
  }
  wg_barrier();

  const bool active = x0 + (lane & 31) < p.dw && k0 + (lane >> 5) < npairs;
  const int x = mini(x0 + (lane & 31), p.dw - 1);
  const int y0 = 2 * mini(k0 + (lane >> 5), npairs - 1);
  const bool has1 = y0 + 1 < p.dh;
  const float px = lpx(x), py0 = lpy(y0), py1 = lpy(y0 + 1);
  const int ax = floor_i(px - 0.5f), b = floor_i(py0 - 0.5f);

  // 2. Gather in the model's precision. Offsets: dx = h((ax + 1/2) - px); d_y = h((b_y + 1/2) - py(y)), minus 1 (half)
  // when the row's own b_y differs from the pair's b; u = h(dx + i), v = h(d_y + j). Taps i = -1 ... 2 (outer),
  // j = -1 ... 2 (inner).
  const h16 dx = (h16)(((float)ax + 0.5f) - px);
  const int b1 = floor_i(py1 - 0.5f);
  const h16 one = (h16)1.0f;
  h16 dy0 = (h16)(((float)b + 0.5f) - py0);  // b_y of the even row is b
  h16 dy1 = (h16)(((float)b1 + 0.5f) - py1);
  if (b1 != b) dy1 = dy1 - one;
  LGather g0, g1;
  l_gather_zero(g0);
  l_gather_zero(g1);
#pragma unroll
  for (int i = -1; i <= kLWin - 2; ++i) {
    const h16 u = dx + (h16)(float)i;
#pragma unroll
    for (int j = -1; j <= kLWin - 2; ++j) {
      const h16 v0 = dy0 + (h16)(float)j, v1 = dy1 + (h16)(float)j;
      LTexel t;
      if (staged) {
        t = sh_tex[(b + j - r0) * ncol + (ax + i - c0)];
      } else {
        const int idx = clampi(b + j, 0, p.rh - 1) * p.rw + clampi(ax + i, 0, p.rw - 1);
        t = l_load_texel(color, A, B, C, idx, p.e);
      }
      l_gather_add(g0, t, u, v0);
      l_gather_add(g1, t, u, v1);
    }
  }

  float h0[3], h1[3] = {0.0f, 0.0f, 0.0f};
  l_finish_pixel(g0, x, y0, px, py0, active, p, s, color, hist_prev, m_near, m_center, m_far, hist_out, display, h0);
  if (has1)
    l_finish_pixel(g1, x, y0 + 1, px, py1, active, p, s, color, hist_prev, m_near, m_center, m_far, hist_out,
                   display, h1);

  // Reduced history: mean of H^ over the 2 x 2 block (x, x + 1) x (y0, y0 + 1); lane x + 1 is lane ^ 1.
#pragma unroll
  for (int i = 0; i < 3; ++i) sh_red[lane * 3 + i] = h0[i] + h1[i];
  wg_barrier();
  if (active && has1 && (x & 1) == 0 && x + 1 < p.dw) {
    u32 code = 0;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      const float m = (sh_red[lane * 3 + i] + sh_red[(lane ^ 1) * 3 + i]) * 0.25f;
      const float v = clampf(enc_e(m) * (1.0f / 2.45f), 0.0f, 1.0f);
      code |= (u32)wg_rint(v * 1023.0f) << (10 * i);
    }
    reduced[(y0 / 2) * (p.dw / 2) + x / 2] = code;
  }
}
