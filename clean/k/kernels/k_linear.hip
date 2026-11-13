// SPDX-License-Identifier: MIT
// Model K FP16 linear layers for gfx1201 (spec 40 task K1, spec 41): per-token linear, 2x2 merge, 2x2 expand.
//
// One template, three addressing modes:
//   plain   y(t) = W x(t) + b                                 x [N][CIN], y [N][CSTORE]
//   merge   y(i, j) = W concat_s x(2i + dy, 2j + dx) + b       x [Hi][Wi][CIN/4], y [ceil(Hi/2)][ceil(Wi/2)][COUT]
//   expand  y(2i + dy, 2j + dx) = (W x(i, j) + b)[s COUT/4 ..]  x [hi][wi][CIN], y [Ho][Wo][COUT/4], ceil(Wo/2) = wi
// with s = 2 dy + dx (top-left, top-right, bottom-left, bottom-right; spec 41). Merge members outside the input map
// are zero (odd sizes only, which spec 44 sizes never have); expand outputs outside the output map are not written.
// A fourth mode builds a decoder stage's block input for the whole window grid (spec 41 draft 3 border rule):
//   expandp y_pad(r + oy, c + ox) = f16((W x(m'(R), m'(Cc)) + b)[s COUT/4 ..] + skip(m(r), m(c)))
//           for every window member (r, c) = (2R + dy, 2Cc + dx), r in [-oy, 8 nwy - oy), c likewise; R, Cc run over
//           the "virtual" coarse positions floor(r / 2), floor(c / 2) (floor semantics), m' / m mirror at the coarse /
//           fine size. y_pad is [8 nwy][8 nwx][COUT/4]; the block kernel reads it without any mirroring.
//
// Numerics (spec 41): the FP32 accumulators start from the bias (flag 1), so f16(W x + b) is rounded once. Then, each
// step rounded to FP16: ReLU (flag 8, the embedding), the activation (flag 2, spec 40 exact), + residual / skip map
// at the output position (flag 4).
//
// WMMA mapping (k_common.h): A = packed weight tiles, B = 16 tokens, D = 16 tokens x 16 output channels. A wave owns
// 16 G tokens and NWT output tiles (16 NWT channels); a 128-lane workgroup = 4 waves on consecutive token groups and
// the same output chunk, so the four waves read the same weight tiles (from the cache) at the same time. The grid is
// (token groups) x (output chunks), chunk-minor so that the chunks of one token range run together.
// COUT is padded to a multiple of 16 (the head's 40 outputs use 48 rows, the last 8 zero); CSTORE channels are stored.
//
// Weight tiles (clean/k/FORMATS.md, "Linear blob"): tile (nt, kt) of W = 16 output rows x 16 inputs at halfs
// ((nt KT + kt) 256), lane l's 8 halfs at l * 8: A row (l % 16) (output channel after the d_row permutation),
// inputs 16 kt + 8 (l / 16) + 0..7.
#include "k_common.h"

#ifndef WGK_LIN_G
#define WGK_LIN_G 2  // token groups of 16 per wave (build variant k_linear__g1: 1)
#endif

enum { KM_PLAIN = 0, KM_MERGE = 1, KM_EXPAND = 2, KM_EXPANDP = 3 };
enum { KF_BIAS = 1, KF_ACT = 2, KF_RES = 4, KF_RELU = 8 };

// Launch arguments (must match KLinParams in clean/k/runner/k_launch.py).
struct KLinParams {
  const h16* x;     // input map
  const h16* w;     // packed weight tiles
  const h16* bias;  // [COUT padded] (flag 1)
  const h16* res;   // residual / skip map, same layout as y (flag 4)
  h16* y;           // output map
  int in_w, in_h;   // input map size in tokens
  int out_w, out_h; // output map size in tokens (expandp: the padded map, 8 nwx x 8 nwy)
  int ncols;        // tokens on the WMMA column side: plain / merge: out_w out_h; expand: in_w in_h;
                    // expandp: virtual coarse positions vw vh
  int nchunks;      // output chunks of 16 NWT channels
  int flags;
  int vw;           // expandp: virtual coarse columns
  int fw, fh;       // expandp: fine (skip) map size
  int ox, oy;       // expandp: window offset of the decoder stage
};
static_assert(sizeof(KLinParams) == 88, "KLinParams must match clean/k/runner/k_launch.py");

constexpr int k_nwt(int cout) {
  return (cout / 16) % 8 == 0 ? 8 : (cout / 16) % 6 == 0 ? 6 : (cout / 16) % 5 == 0 ? 5 : (cout / 16) % 4 == 0 ? 4
       : (cout / 16) % 3 == 0 ? 3 : (cout / 16) % 2 == 0 ? 2 : 1;
}
constexpr int k_pad16(int n) { return (n + 15) / 16 * 16; }

template <int CIN, int COUT, int MODE, int G, int CSTORE>
WG_FN void k_linear(const KLinParams& P) {
  constexpr int KT = CIN / 16;
  constexpr int NWT = k_nwt(COUT);
  constexpr int CS_IN = MODE == KM_MERGE ? CIN / 4 : CIN;      // channels of one input token
  constexpr bool EXP = MODE == KM_EXPAND || MODE == KM_EXPANDP;
  constexpr int CS_OUT = EXP ? COUT / 4 : COUT;  // output channels per output token (padded)
  static_assert(CIN % 16 == 0 && COUT % 16 == 0 && CS_IN % 16 == 0 && CS_OUT % 16 == 0, "shape");
  static_assert(CSTORE % 8 == 0 && CSTORE <= CS_OUT && (!EXP || CSTORE == CS_OUT), "stored channels");
  const u32 tid = wg_lane_id();
  const u32 lane = tid & 31u, wave = tid >> 5;
  const int n16 = (int)(lane & 15u), hh = (int)(lane >> 4);
  const int chunk = (int)wg_block_id() % P.nchunks;
  const int tgroup = (int)wg_block_id() / P.nchunks;
  const int col0 = (tgroup * 4 + (int)wave) * 16 * G;

  int col[G];
  bool ok[G];
  int src[G][4];  // merge: flat input token of sub-token s, -1 outside
  int vrow[G], vcol[G];  // expandp: virtual coarse position (R, Cc)
  const int r0 = wgk_floor2(-P.oy), c0 = wgk_floor2(-P.ox);
#pragma unroll
  for (int g = 0; g < G; ++g) {
    col[g] = col0 + 16 * g + n16;
    ok[g] = col[g] < P.ncols;
    if (MODE == KM_EXPANDP) {
      vrow[g] = r0 + col[g] / P.vw;
      vcol[g] = c0 + col[g] % P.vw;
    }
    if (MODE == KM_MERGE) {
      const int cy = col[g] / P.out_w, cx = col[g] - cy * P.out_w;
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        const int fy = 2 * cy + (s >> 1), fx = 2 * cx + (s & 1);
        src[g][s] = ok[g] && fy < P.in_h && fx < P.in_w ? fy * P.in_w + fx : -1;
      }
    }
  }

  v8f acc[NWT][G];
#pragma unroll
  for (int nt = 0; nt < NWT; ++nt) {
    const v8f b0 = (P.flags & KF_BIAS) ? wgk_widen8(wgk_ld8(P.bias + (chunk * NWT + nt) * 16 + 8 * hh)) : wgk_zero8f();
#pragma unroll
    for (int g = 0; g < G; ++g) acc[nt][g] = b0;
  }

  const h16* wt = P.w + (size_t)chunk * NWT * KT * 256 + lane * 8;
#pragma unroll 2
  for (int kt = 0; kt < KT; ++kt) {
    v8h b[G];
#pragma unroll
    for (int g = 0; g < G; ++g) {
      if (MODE == KM_MERGE) {
        const int s = (16 * kt) / CS_IN, c0 = 16 * kt - s * CS_IN;
        const int t = s == 0 ? src[g][0] : s == 1 ? src[g][1] : s == 2 ? src[g][2] : src[g][3];
        b[g] = t >= 0 ? wgk_ld8(P.x + (size_t)t * CS_IN + c0 + 8 * hh) : wgk_zero8();
      } else if (MODE == KM_EXPANDP) {
        const size_t t = (size_t)wgk_mirror(vrow[g], P.in_h) * P.in_w + wgk_mirror(vcol[g], P.in_w);
        b[g] = ok[g] ? wgk_ld8(P.x + t * CIN + 16 * kt + 8 * hh) : wgk_zero8();
      } else {
        b[g] = ok[g] ? wgk_ld8(P.x + (size_t)col[g] * CIN + 16 * kt + 8 * hh) : wgk_zero8();
      }
    }
#pragma unroll
    for (int nt = 0; nt < NWT; ++nt) {
      const v8h a = wgk_ld8(wt + (size_t)(nt * KT + kt) * 256);
#pragma unroll
      for (int g = 0; g < G; ++g) acc[nt][g] = wg_wmma_f16(a, b[g], acc[nt][g]);
    }
  }

  // epilogue
#pragma unroll
  for (int nt = 0; nt < NWT; ++nt) {
    const int tile0 = (chunk * NWT + nt) * 16;  // first output channel (of COUT) of this tile
    const int n0 = tile0 + 8 * hh;              // this lane's first channel
#pragma unroll
    for (int g = 0; g < G; ++g) {
      v8h r = wgk_round8(acc[nt][g]);
      if (P.flags & KF_RELU) r = wgk_max8(r, wgk_zero8());
      if (P.flags & KF_ACT) r = wgk_act8(r);
      bool valid = ok[g];
      size_t off, roff;
      if (MODE == KM_EXPANDP) {
        const int s = tile0 / CS_OUT, c = n0 - s * CS_OUT;
        const int fy = 2 * vrow[g] + (s >> 1), fx = 2 * vcol[g] + (s & 1);  // fine position (may be outside)
        const int py = fy + P.oy, px = fx + P.ox;                            // padded position
        valid = valid && py >= 0 && py < P.out_h && px >= 0 && px < P.out_w;
        off = ((size_t)py * P.out_w + px) * CS_OUT + c;
        roff = ((size_t)wgk_mirror(fy, P.fh) * P.fw + wgk_mirror(fx, P.fw)) * CS_OUT + c;
      } else if (MODE == KM_EXPAND) {
        const int s = tile0 / CS_OUT, c = n0 - s * CS_OUT;
        const int cy = col[g] / P.in_w, cx = col[g] - cy * P.in_w;
        const int fy = 2 * cy + (s >> 1), fx = 2 * cx + (s & 1);
        valid = valid && fy < P.out_h && fx < P.out_w;
        off = roff = ((size_t)fy * P.out_w + fx) * CS_OUT + c;
      } else {
        valid = valid && n0 < CSTORE;
        off = roff = (size_t)col[g] * CSTORE + n0;
      }
      if (valid) {
        if (P.flags & KF_RES) r = wgk_add8(wgk_ld8(P.res + roff), r);
        wgk_st8(P.y + off, r);
      }
    }
  }
}

// ---- instantiations: every shape of model K (clean/k/reference/layout.py linear_shapes) -------------------------
// plain K x N: embedding 16x32, head 32x40, per block C x D (fixed: v) or C x 3D (softmax: q, k, v), D x C (output
// projection), C x 4C and 4C x C (MLP). merge 4C x C_next of e0 .. e4, expand C_prev x 4C of d4 .. d0.
#define K_PLAIN_SHAPES(X)                                                                                    \
  X(16, 32) X(32, 40) X(32, 64) X(32, 128) X(64, 32) X(64, 64) X(64, 192) X(64, 256) X(96, 384) X(128, 32)   \
  X(128, 96) X(128, 128) X(128, 384) X(128, 512) X(160, 640) X(160, 768) X(256, 64) X(256, 160) X(384, 96)   \
  X(512, 128) X(640, 160)
#define K_MERGE_SHAPES(X) X(128, 64) X(256, 64) X(256, 96) X(384, 128) X(512, 160)
#define K_EXPAND_SHAPES(X) X(160, 512) X(128, 384) X(96, 256) X(64, 256) X(64, 128)

#define K_DEF_PLAIN(ci, co)                                     \
  WG_KERNEL_N(k_lin_##ci##x##co, 128)(KLinParams P) {           \
    k_linear<ci, k_pad16(co), KM_PLAIN, WGK_LIN_G, co>(P);      \
  }
#define K_DEF_MERGE(ci, co) \
  WG_KERNEL_N(k_merge_##ci##x##co, 128)(KLinParams P) { k_linear<ci, co, KM_MERGE, WGK_LIN_G, co>(P); }
#define K_DEF_EXPAND(ci, co)                                                                                    \
  WG_KERNEL_N(k_expand_##ci##x##co, 128)(KLinParams P) { k_linear<ci, co, KM_EXPAND, WGK_LIN_G, co / 4>(P); }   \
  WG_KERNEL_N(k_expandp_##ci##x##co, 128)(KLinParams P) { k_linear<ci, co, KM_EXPANDP, WGK_LIN_G, co / 4>(P); }
K_PLAIN_SHAPES(K_DEF_PLAIN)
K_MERGE_SHAPES(K_DEF_MERGE)
K_EXPAND_SHAPES(K_DEF_EXPAND)

#if defined(WG_EMU)
#define K_REG_PLAIN(ci, co) WG_EMU_REGISTER(k_lin_##ci##x##co)
#define K_REG_MERGE(ci, co) WG_EMU_REGISTER(k_merge_##ci##x##co)
#define K_REG_EXPAND(ci, co) WG_EMU_REGISTER(k_expand_##ci##x##co) WG_EMU_REGISTER(k_expandp_##ci##x##co)
K_PLAIN_SHAPES(K_REG_PLAIN)
K_MERGE_SHAPES(K_REG_MERGE)
K_EXPAND_SHAPES(K_REG_EXPAND)
#endif
