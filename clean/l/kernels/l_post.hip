// SPDX-License-Identifier: MIT
// Model L (issue #89): output reconstruction (stage "post", HDR) as Model M's post_recon (clean/kernels/post.hip,
// included for its functions; its own kernels come along unused) with Model L's three changes
// (clean/l/reference/recon.py, the oracle):
//  1. a 4 x 4 colour gather: columns ax - 1 ... ax + 2 and rows b - 1 ... b + 2 (b from the even row of the pair);
//  2. each history candidate decoded and divided by the previous exposure before the two are combined:
//     H_lin = (w1 L1 + w2 L2) e, with L_i = D(H_i) / e_prev (L2 only when w2 > 0);
//  3. an invalid or reset sample falls back to the render pixel's linear colour (no exposure, no encode).
// The decode passes a NaN through (a zero total gather weight gives C = D(NaN) = NaN; the NaN guard then substitutes
// it, as the oracle). Arguments, layouts and grid are post_recon's; w[0] = Model L's post scale s.
#include "post.hip"

namespace {

constexpr int kLWin = 4;  // taps -1 ... 2

// D(.) with a NaN passed through.
WG_FN float l_dec(float q) { return q != q ? q : dec_e(q); }

// Steps 3-5 for one D pixel (Model L). Returns the linear new history H^ in hh and the linear display value O^ in oh.
WG_FN void l_post_core(const Gather& g, int x, int y, int fx, int fy, const float mx[3], const float my[3],
                       const WgPostParams& p, float s, const v4f* color, const void* hist_prev, float hh[3],
                       float oh[3]) {
  const PostSel sl = post_select(g);
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
    Cl[i] = l_dec(sl.cbar[i]);
  }
  // 5. Blend (Model M's form, clean/kernels/post.hip post_blend): beta' = 1 - (1 - alpha)(1 - beta).
  const float alpha = g.W < __builtin_inff() ? clampf(s * (g.a[0] * (1.0f / 255.0f)), 0.0f, 1.0f) : 0.0f;
  const float ombh = (1.0f - alpha) * sl.cbh, ombo = (1.0f - alpha) * sl.cbo;
  const float bh2 = 1.0f - ombh, bo2 = 1.0f - ombo;
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    hh[i] = __builtin_fmaf(Hl[i], ombh, Cl[i] * bh2);
    oh[i] = __builtin_fmaf(Hl[i], ombo, Cl[i] * bo2);
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
WG_FN void l_finish_pixel(const Gather& g, int x, int y, float px, float py, bool store, const WgPostParams& p,
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
  WG_LDS PostTexel sh_tex[kStageCap];
  WG_LDS float sh_red[kPostThreads * 3];

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
  const int c0 = floor_i(pos_x(x0, p) - 0.5f) - 1, c1 = floor_i(pos_x(xl, p) - 0.5f) + (kLWin - 2);
  const int r0 = floor_i(pos_y(2 * k0, p) - 0.5f) - 1, r1 = floor_i(pos_y(2 * kl, p) - 0.5f) + (kLWin - 2);
  const int ncol = c1 - c0 + 1, nrow = r1 - r0 + 1;
  const bool staged = ncol > 0 && nrow > 0 && ncol <= kStageCap && nrow <= kStageCap && ncol * nrow <= kStageCap;
  if (staged) {
    for (int i = lane; i < ncol * nrow; i += kPostThreads) {
      const int r = i / ncol, c = i - r * ncol;
      const int idx = clampi(r0 + r, 0, p.rh - 1) * p.rw + clampi(c0 + c, 0, p.rw - 1);
      sh_tex[i] = load_texel(color, A, B, C, idx, p.e);
    }
  }
  wg_barrier();

  const bool active = x0 + (lane & 31) < p.dw && k0 + (lane >> 5) < npairs;
  const int x = mini(x0 + (lane & 31), p.dw - 1);
  const int y0 = 2 * mini(k0 + (lane >> 5), npairs - 1);
  const bool has1 = y0 + 1 < p.dh;
  const float px = pos_x(x, p), py0 = pos_y(y0, p), py1 = pos_y(y0 + 1, p);
  const int ax = floor_i(px - 0.5f), b = floor_i(py0 - 0.5f);

  // 2. Gather, taps i = -1 ... 2 (outer), j = -1 ... 2 (inner).
  Gather g0, g1;
  gather_zero(g0);
  gather_zero(g1);
#pragma unroll
  for (int i = -1; i <= kLWin - 2; ++i) {
    const float u = ((float)(ax + i) + 0.5f) - px;
    const float uu = kTapK * u * u;
#pragma unroll
    for (int j = -1; j <= kLWin - 2; ++j) {
      const float v0 = ((float)(b + j) + 0.5f) - py0;
      const float v1 = ((float)(b + j) + 0.5f) - py1;
      const float vv0 = kTapK * v0 * v0, vv1 = kTapK * v1 * v1;
      const float uv0 = 2.0f * kTapK * u * v0, uv1 = 2.0f * kTapK * u * v1;
      PostTexel t;
      if (staged) {
        t = sh_tex[(b + j - r0) * ncol + (ax + i - c0)];
      } else {
        const int idx = clampi(b + j, 0, p.rh - 1) * p.rw + clampi(ax + i, 0, p.rw - 1);
        t = load_texel(color, A, B, C, idx, p.e);
      }
      gather_add(g0, t, uu, vv0, uv0);
      gather_add(g1, t, uu, vv1, uv1);
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
