// SPDX-License-Identifier: GPL-3.0-only
// Model L (issue #89): output reconstruction (stage "post", HDR) as Model M's post_recon (clean/kernels/post.hip,
// included for its functions; its own kernels come along unused) with Model L's three changes
// (clean/l/reference/recon.py, the oracle):
//  1. a 4 x 4 colour gather: columns ax - 1 ... ax + 2 and rows b - 1 ... b + 2 (b from the even row of the pair);
//  2. each history candidate decoded and divided by the previous exposure before the two are combined:
//     H_lin = (w1 L1 + w2 L2) e, with L_i = D(H_i) r_prev, r_prev = 1 / e_prev (L2 only when w2 > 0; the spec's
//     division by e_prev as a product with its reciprocal: free within the post tolerance, spec answer Q5);
//  3. an invalid or reset sample falls back to the render pixel's linear colour (no exposure, no encode).
// The colour gather runs in the model's own precision as confirmed by the spec side (oracle: recon.gather_half):
// fused target positions, FP16 offsets and exponent with the fused cross term, float32 weights flushed to +0 below
// 2^-126, FP16 weights into half accumulators, a float32 weight sum. The blend is item 3's fused form. The decode
// passes a NaN through (a zero total gather weight gives C = D(NaN) = NaN; the NaN guard then substitutes it, as the
// oracle). Arguments, layouts and grid are post_recon's; w[0] = Model L's post scale s.
// l_post_recon_p12: the same kernel on the pack12 coefficients of l_coef_pack12 (A8 / C4 images, B unused), which the
// D3D12 frame reads; the same texel values, so the same outputs byte for byte.
// LDR forms (l_post_recon_ldr, l_post_recon_p12_ldr; spec preset L part 3 item 6, Q1, Q2): the same function with the
// identity transfer (E(v) = v, D(e) = e: the colour feature is h(colour), a NaN kept; no decode of the history
// candidates or of C), no exposure (the exposure pair is not read: e = e_prev = 1), the full history h(H^), the display
// image the output stage's unpacked 11 / 11 / 10-bit codes of O^ (code = floor(clamp(fused(O^ N + 1/2), 0, N)),
// N = 2047, 2047, 1023, NaN -> 0; stored h(code RN32(1 / N))) and the reduced codes rint(clamp(mean, 0, 1) 1023).
// l_post_filter_ldr / l_post_filter_ldr_h: the output stage for that display image (Q2): Model M's 5-tap filter
// (post.hip filter_row_c1, lz5) with no clamp to [1e-4, 3], no decode and no exposure; 3 x f32, or RGBA16F rows with pi
// folded in (framebuffer_half(v pi), alpha 1).
//
// Q11 part C evaluation switches (all 0 by default: Model L's kernel as above, unchanged), built together as
// l_post__morig.co = Model M's ORIGINAL reconstruction (handoff preset-M/from-spec/Q11C-M-POST-ITEMS.md), which the
// runtime runs for Model M with WG_M_POST=orig (HIP only), with M's scale s and M's coefficients:
//  WG_LP_M = 1: M's three steps (HDR; the LDR forms are the same for both presets): D1 a 3 x 3 gather (columns
//    ax - 1 ... ax + 1, rows b - 1 ... b + 1); D2 the two candidates combined in the ENCODED domain,
//    H = fused(w1 H1 + (w2 H2)) (w1 H1 when w2 = 0), then one decode H_lin = (D(H) / e_prev) e; D3 an invalid or reset
//    candidate is E(e_prev colour) (no 32752 cap).
//  WG_LP_C1 = 1: the gather exponent in the machine-code form (PRESET-M-C1C2-machine-code.md, for M and L):
//    eps = fused_half(h(u v), C, fused_half(h(v v), B, h(h(u u) A))) (0: item 3's form, Model L's kernel; one
//    rounding more, the probe's near-miss mutant).
//  WG_CR_C2 (post.hip cr_weights): the Catmull-Rom weights' C2 form.
#ifndef WG_LP_M
#define WG_LP_M 0
#endif
#ifndef WG_LP_C1
#define WG_LP_C1 0
#endif
#include "post.hip"

namespace {

constexpr int kLWin = WG_LP_M ? 3 : 4;  // taps -1 ... 2 (Model L), -1 ... 1 (WG_LP_M)
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

// The colour feature: h(E_in(e c)) (HDR) or h(c) (LDR: identity transfer, no exposure; a NaN is kept).
template <bool kLdr>
WG_FN h16 l_colour_feature(float c, float e) {
  if (kLdr) return (h16)c;
  return (h16)enc_in(e * c);
}

template <bool kLdr>
WG_FN LTexel l_load_texel(const v4f* color, const u32* A, const u32* B, const v4h* C, int idx, float e) {
  LTexel t;
  const v4f c = color[idx];
  t.f[0] = l_colour_feature<kLdr>(c[0], e);
  t.f[1] = l_colour_feature<kLdr>(c[1], e);
  t.f[2] = l_colour_feature<kLdr>(c[2], e);
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

// l_load_texel on the pack12 coefficients (l_coef_pack12, clean/l/FORMATS.md): image A8, 8 bytes per render pixel =
// codes A0 A1 A2 B0 | B1 B2, C2 (FP16); image C4, 4 bytes per render pixel = C0, C1 (FP16). The same values as
// l_load_texel on l_coef's images (only padding bytes are left out), so the same texel.
template <bool kLdr>
WG_FN LTexel l_load_texel12(const v4f* color, const u32* A8, const u32* C4, int idx, float e) {
  LTexel t;
  const v4f c = color[idx];
  t.f[0] = l_colour_feature<kLdr>(c[0], e);
  t.f[1] = l_colour_feature<kLdr>(c[1], e);
  t.f[2] = l_colour_feature<kLdr>(c[2], e);
  const u32 d0 = A8[2 * idx], d1 = A8[2 * idx + 1], c01 = C4[idx];
  const u32 a = d0 & 0xffffffu, b = (d0 >> 24) | ((d1 & 0xffffu) << 8);
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    t.f[3 + i] = (h16)(byte_f(b, i) / 255.0f);
    t.f[6 + i] = (h16)(byte_f(a, i) / 255.0f);
  }
  t.kA = (h16)((float)__builtin_bit_cast(_Float16, (unsigned short)(c01 & 0xffffu)) * -0.721325f);
  t.kB = (h16)((float)__builtin_bit_cast(_Float16, (unsigned short)(c01 >> 16)) * -0.721325f);
  t.kC = (h16)((float)__builtin_bit_cast(_Float16, (unsigned short)(d1 >> 16)) * -1.44265f);
  return t;
}

// The texel of the kernel's coefficient form: l_coef's images (A, B, C) or pack12's (A8 in A, C4 in C, B unused).
template <bool kP12, bool kLdr>
WG_FN LTexel l_texel(const v4f* color, const u32* A, const u32* B, const void* C, int idx, float e) {
  if (kP12) return l_load_texel12<kLdr>(color, A, (const u32*)C, idx, e);
  return l_load_texel<kLdr>(color, A, B, (const v4h*)C, idx, e);
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
// The tap's exponent (WG_LP_C1: the machine-code form; also called by the C1/C2 probe, l_post_c12_probe.hip).
WG_FN h16 l_eps(h16 u, h16 v, h16 kA, h16 kB, h16 kC) {
  const h16 uu = u * u, vv = v * v, uv = u * v;
#if WG_LP_C1 == 1
  const h16 ta = uu * kA;
  const h16 s = __builtin_fmaf16(vv, kB, ta);
  return __builtin_fmaf16(uv, kC, s);
#else
  const h16 ta = uu * kA, tb = vv * kB;
  const h16 s = ta + tb;
  return __builtin_fmaf16(uv, kC, s);
#endif
}

WG_FN void l_gather_add(LGather& g, const LTexel& t, h16 u, h16 v) {
  const h16 eps = l_eps(u, v, t.kA, t.kB, t.kC);
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
// LDR: no decode (the candidates and C are the values themselves) and no exposure.
template <bool kLdr>
WG_FN void l_post_core(const LGather& g, int x, int y, int fx, int fy, const float mx[3], const float my[3],
                       const WgPostParams& p, float s, const v4f* color, const void* hist_prev, float hh[3],
                       float oh[3]) {
  const LSel sl = l_select(g);
  const float X = (float)x + 0.5f, Y = (float)y + 0.5f;
  const int ncand = sl.w2 != 0.0f ? 2 : 1;
  const float rp = kLdr ? 1.0f : 1.0f / p.e_prev;  // IEEE division
  constexpr bool kEnc = WG_LP_M && !kLdr;  // D2 / D3: candidates in the encoded domain
  float L[2][3];
  for (int n = 0; n < ncand; ++n) {
    const int k = n == 0 ? sl.first : sl.second;
    const float qx = X + mx[k], qy = Y + my[k];
    const bool valid = !p.reset && qx >= 0.0f && qx <= (float)p.dw && qy >= 0.0f && qy <= (float)p.dh;
    if (valid) {
      float h[3];
      history_cr<false>(hist_prev, p.dw, p.dh, qx, qy, h);
#pragma unroll
      for (int i = 0; i < 3; ++i) L[n][i] = (kLdr || kEnc) ? h[i] : l_dec(h[i]) * rp;
    } else {
      const v4f c = color[fy * p.rw + fx];
#pragma unroll
      for (int i = 0; i < 3; ++i) L[n][i] = kEnc ? enc_e(p.e_prev * c[i]) : c[i];
    }
  }
  float Hl[3], Cl[3];
  bool hl_inf = false;
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    const float comb = ncand == 2 ? __builtin_fmaf(sl.w1, L[0][i], sl.w2 * L[1][i]) : sl.w1 * L[0][i];
    Hl[i] = kLdr ? comb : (kEnc ? (l_dec(comb) * rp) * p.e : comb * p.e);
    hl_inf |= __builtin_fabsf(Hl[i]) == __builtin_inff();
    Cl[i] = kLdr ? sl.fbar[i] : l_dec(sl.fbar[i]);
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

// LDR display value of channel i: the output stage's unpack of the 11 / 11 / 10-bit code of O^ (Q2): code =
// floor(clamp(fused(o N + 1/2), 0, N)) (the clamp maps NaN to 0), then h(code RN32(1 / N)) (one float32 product, then
// the FP16 rounding).
WG_FN _Float16 l_ldr_display_value(float o, int i) {
  const float n = i == 2 ? 1023.0f : 2047.0f;
  const float k = __builtin_bit_cast(float, i == 2 ? 0x3a802008u : 0x3a001002u);  // RN32(1/1023), RN32(1/2047)
  const float code = __builtin_floorf(clampf(__builtin_fmaf(o, n, 0.5f), 0.0f, n));
  return (_Float16)(code * k);
}

// post_recon's finish_pixel with l_post_core.
template <bool kLdr>
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
  l_post_core<kLdr>(g, x, y, fx, fy, mx, my, p, s, color, hist_prev, hh, oh);
  if (kLdr) {
    if (store) hist_store<false>(hist_out, y * p.dw + x, hh[0], hh[1], hh[2]);
    if (store) {
      const int pitch = display_pitch(p.dw), plane = pitch * p.dh, di = y * pitch + x;
      display[di] = l_ldr_display_value(oh[0], 0);
      display[plane + di] = l_ldr_display_value(oh[1], 1);
      display[2 * plane + di] = l_ldr_display_value(oh[2], 2);
    }
    return;
  }
  if (store) {
#if WG_LP_M
    // Spec 06 draft 6: full-history store rounds FP32 -> FP16 toward zero (RTZ; v_cvt_pkrtz_f16_f32),
    // keeping FP16 subnormals, +-0, and NaN. Display channels and reduced history stay RNE.
    const v2h h01 = wg_h2(enc_e(hh[0]), enc_e(hh[1]));
    const v2h h20 = wg_h2(enc_e(hh[2]), 0.0f);
    hist_out[y * p.dw + x] = v4h{h01[0], h01[1], h20[0], h20[1]};
#else
    hist_store<false>(hist_out, y * p.dw + x, enc_e(hh[0]), enc_e(hh[1]), enc_e(hh[2]));
#endif
  }
  if (store) {
    const int pitch = display_pitch(p.dw), plane = pitch * p.dh, di = y * pitch + x;
    display[di] = display_value(oh[0]);
    display[plane + di] = display_value(oh[1]);
    display[2 * plane + di] = display_value(oh[2]);
  }
}

// The kernel body; kP12 selects the coefficient form (l_texel), kLdr the LDR form (no exposure read).
template <bool kP12, bool kLdr>
WG_FN void l_post_recon_body(const v4f* color, const u32* A, const u32* B, const void* C, const v4h* hist_prev,
                             const u32* m_near, const u32* m_center, const u32* m_far, const float* w, v4h* hist_out,
                             u32* reduced, _Float16* display, WgPostParams p, const float* expo) {
  if (kLdr) {
    p.e = 1.0f;
    p.e_prev = 1.0f;
    p.ratio = 1.0f;
  } else if (expo) {
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
      sh_tex[i] = l_texel<kP12, kLdr>(color, A, B, C, idx, p.e);
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
        t = l_texel<kP12, kLdr>(color, A, B, C, idx, p.e);
      }
      l_gather_add(g0, t, u, v0);
      l_gather_add(g1, t, u, v1);
    }
  }

  float h0[3], h1[3] = {0.0f, 0.0f, 0.0f};
  l_finish_pixel<kLdr>(g0, x, y0, px, py0, active, p, s, color, hist_prev, m_near, m_center, m_far, hist_out, display,
                       h0);
  if (has1)
    l_finish_pixel<kLdr>(g1, x, y0 + 1, px, py1, active, p, s, color, hist_prev, m_near, m_center, m_far, hist_out,
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
      const float v = kLdr ? clampf(m, 0.0f, 1.0f) : clampf(enc_e(m) * (1.0f / 2.45f), 0.0f, 1.0f);
      code |= (u32)wg_rint(v * 1023.0f) << (10 * i);
    }
    reduced[(y0 / 2) * (p.dw / 2) + x / 2] = code;
  }
}

// The LDR output stage (Q2): filter_strip_c1<S> of post.hip (the same lanes, tiles, sliding window and 5-tap passes)
// with the filtered value written as it is: no clamp to [1e-4, 3], no decode, no exposure. kH: RGBA16F rows of hpitch
// bytes, framebuffer_half(v pi), the lane of channel 2 also writes alpha 1; else 3 x f32 (row-major RGB).
template <int S, bool kH>
WG_FN void l_filter_strip_ldr(const _Float16* display, void* out, int dw, int dh, int ow, int oh, float pi,
                               int hpitch) {
  const int lane = (int)wg_lane_id();
  const int wl = lane & 31;
  if (wl >= 3 * kFc1Groups) return;
  const int g = wl / 3, c = wl - 3 * g;
  const int tiles_x = (ow + 4 * kFc1Groups - 1) / (4 * kFc1Groups);
  const int tile = (int)wg_block_id();
  const int m = (tile % tiles_x) * kFc1Groups + g;
  const int p0 = ((tile / tiles_x) * kFiltStrips + (lane >> 5)) * S;
  if (4 * m >= ow || 2 * p0 >= oh) return;
  const int pitch = display_pitch(dw), plane = pitch * dh;
  const bool fast = m >= 1 && 6 * m + 7 <= dw - 1;
  const int ncols = ow - 4 * m < 4 ? ow - 4 * m : 4;

  float hr[7][4];
  filter_row_c1(display, pitch, plane, dw, clampi(3 * p0 - 2, 0, dh - 1), m, fast, c, hr[6]);
#pragma nounroll
  for (int st = -1; st < S; ++st) {
    const int p = p0 + st;
    if (2 * p >= oh) break;
#pragma unroll
    for (int k = 0; k < 4; ++k)
#pragma unroll
      for (int i = 0; i < 4; ++i) hr[k][i] = hr[k + 3][i];
#pragma unroll
    for (int k = 4; k < 7; ++k)
      filter_row_c1(display, pitch, plane, dw, clampi(3 * p - 2 + k, 0, dh - 1), m, fast, c, hr[k]);
    if (st < 0) continue;
#pragma unroll
    for (int dy = 0; dy < 2; ++dy) {
      const int y = 2 * p + dy;
      if (y >= oh) break;
      const u32 o0 = mul24((u32)y, (u32)ow) + (u32)(4 * m);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float t[5];
#pragma unroll
        for (int k = 0; k < 5; ++k) t[k] = hr[2 * dy + k][i];
        const float v = dy == 0 ? lz5<true>(t) : lz5<false>(t);
        if (i >= ncols) continue;
        if (kH) {
          _Float16* o = (_Float16*)((char*)out + (size_t)y * (u32)hpitch + (u32)(4 * m + i) * 8u);
          o[c] = framebuffer_half(v * pi);
          if (c == 2) o[3] = (_Float16)1.0f;
        } else {
          ((float*)((char*)out + (o0 + (u32)i) * 12u))[c] = v;
        }
      }
    }
  }
}

}  // namespace

// Stage "post" for Model L, HDR. Arguments, layouts and grid as post_recon (clean/kernels/post.hip): grid
// ceil(dw / 32) * ceil(ceil(dh / 2) / 8) workgroups of 256 lanes; w = the post blob (w[0] = s).
WG_KERNEL_N(l_post_recon, kPostThreads)(const v4f* color, const u32* A, const u32* B, const v4h* C,
                                        const v4h* hist_prev, const u32* m_near, const u32* m_center,
                                        const u32* m_far, const float* w, v4h* hist_out, u32* reduced,
                                        _Float16* display, WgPostParams p, const float* expo) {
  l_post_recon_body<false, false>(color, A, B, C, hist_prev, m_near, m_center, m_far, w, hist_out, reduced, display, p,
                                  expo);
}

// l_post_recon on the pack12 coefficients of l_coef_pack12: A = image A8 (8 bytes per render pixel), C = image C4
// (4 bytes per render pixel), B unused (may be null). Everything else as l_post_recon; the same outputs.
WG_KERNEL_N(l_post_recon_p12, kPostThreads)(const v4f* color, const u32* A8, const u32* B, const u32* C4,
                                            const v4h* hist_prev, const u32* m_near, const u32* m_center,
                                            const u32* m_far, const float* w, v4h* hist_out, u32* reduced,
                                            _Float16* display, WgPostParams p, const float* expo) {
  l_post_recon_body<true, false>(color, A8, B, C4, hist_prev, m_near, m_center, m_far, w, hist_out, reduced, display,
                                 p, expo);
}

// The LDR forms (the arguments of the HDR kernels; expo is not read).
WG_KERNEL_N(l_post_recon_ldr, kPostThreads)(const v4f* color, const u32* A, const u32* B, const v4h* C,
                                            const v4h* hist_prev, const u32* m_near, const u32* m_center,
                                            const u32* m_far, const float* w, v4h* hist_out, u32* reduced,
                                            _Float16* display, WgPostParams p, const float* expo) {
  l_post_recon_body<false, true>(color, A, B, C, hist_prev, m_near, m_center, m_far, w, hist_out, reduced, display, p,
                                 expo);
}
WG_KERNEL_N(l_post_recon_p12_ldr, kPostThreads)(const v4f* color, const u32* A8, const u32* B, const u32* C4,
                                                const v4h* hist_prev, const u32* m_near, const u32* m_center,
                                                const u32* m_far, const float* w, v4h* hist_out, u32* reduced,
                                                _Float16* display, WgPostParams p, const float* expo) {
  l_post_recon_body<true, true>(color, A8, B, C4, hist_prev, m_near, m_center, m_far, w, hist_out, reduced, display, p,
                                expo);
}

// The LDR output stage. Arguments as post_filter_c1_s4 (dec_scale and expo are not read), grid ceil(ow / 40) *
// ceil(ceil(oh / 2) / 8) workgroups of 64 lanes; _h as post_filter_h's arguments (pi folded in, RGBA16F rows of hpitch
// bytes) on the same grid.
WG_KERNEL_N(l_post_filter_ldr, kFiltThreads)(const _Float16* display, void* out, int dw, int dh, int ow, int oh,
                                             float dec_scale, const float* expo) {
  l_filter_strip_ldr<4, false>(display, out, dw, dh, ow, oh, 1.0f, 0);
}
WG_KERNEL_N(l_post_filter_ldr_h, kFiltThreads)(const _Float16* display, void* out, int dw, int dh, int ow, int oh,
                                               float dec_scale, const float* expo, float pi, int hpitch) {
  l_filter_strip_ldr<4, true>(display, out, dw, dh, ow, oh, pi, hpitch);
}
