// SPDX-License-Identifier: MIT
// Frame generation (issue #96): net1's convolutions on gfx12's 16 x 16 x 16 matrix instructions (the fast path of
// handoff note NET1-FAST-PATH, step 2). Not bit-exact to the CPU reference: the matrix unit's chunk sum is not the exact
// one (clean/fg/records/wmma-probe-2026-11-14.json); net1's output against the CPU reference is reported, not gated.
// HIP and the D3D12 port (clean/dx12/kernels/fg_mconv.hlsl) are byte-identical to each other.
//
// Layout (fast path only): tensors NHWC float16, element (y, x, c) at (y * W + x) * C + c, so the 8 consecutive
// channels of one pixel are one 16-byte fragment load or store.
//
// fg_mconv<MT>(x, wf, b, y, p, cin, cout, H, W, k, order, act, pool, tiles_x, tiles_y), tier B of the note:
//   one net1 convolution (stride 1, zero border for k = 3), the spec's chunk structure: the accumulator starts at the
//   bias and takes one WMMA per chunk (16 input channels of one tap; K = 16), in the layer's chunk order (order 0
//   ascending, 1 channel block / kx / ky, 2 kx / channel block / ky): v_wmma_f32_16x16x16_f16 with C = the float16
//   accumulator (widened exactly), its float32 result rounded to float16 after every chunk (v_cvt_f16_f32, nearest
//   even), so the accumulator stays float16 as in the spec. An exact-zero sum gives +0, the reference's convention
//   (the probe found a -0 only where the matrix unit dropped a tiny term).
//   Then leaky ReLU (act 1; fg_leaky16, exact) and the output y; with pool 1 also net1's 2 x 2 max-pool of the
//   activated output into p (floor sizes; fg_pool_step16 from tap (0, 0), (0, 1), (1, 0), (1, 1)).
//   Weights wf: A fragments in chunk order, [cout / 16][chunks][32 lanes][8] float16: lane l, row o = 16 mt + l % 16,
//   element j = input channel 16 cb + 8 (l / 16) + j of the chunk's tap (built by the runner, fg_net1_mma.py).
//   Work: a workgroup is 8 waves over a 16 x 16 pixel tile; wave w takes rows 2w and 2w + 1 (two 16-pixel N tiles) and
//   MT output-channel tiles (16 MT channels). Grid: tiles_x * tiles_y * (cout / (16 MT)) workgroups of 256 lanes;
//   block id = (mg * tiles_y + ty) * tiles_x + tx.
// fg_upscale_hwc_f16: net1's bilinear upscale (fg_upscale_f16 of fg_net1.hip, the same arithmetic) on NHWC tensors.
#include "wg_platform.h"
#include "wg_mem.h"

typedef unsigned int uint;
typedef unsigned short u16;
#define FG_FN WG_FN
#define FG_I64 long long
#define FG_FBH32(v) (31u - (uint)__builtin_clz(v))
#include "fg_exact16.h"

#pragma clang fp contract(off)

typedef uint v4u32 __attribute__((ext_vector_type(4)));

WG_FN uint fg_f2h(float f) { return (uint)__builtin_bit_cast(u16, (_Float16)f); }      // round to nearest even
WG_FN float fg_h2f(uint h) { return (float)__builtin_bit_cast(_Float16, (u16)h); }     // exact
WG_FN uint fg_xor1(uint v) {                                                           // v of lane (lane ^ 1)
    return (uint)__builtin_amdgcn_ds_bpermute((int)(((wg_lane_id() & 31u) ^ 1u) << 2), (int)v);
}

WG_FN v8h fg_as_v8h(v4u32 u) { return __builtin_bit_cast(v8h, u); }
WG_FN v4u32 fg_ld128(const u16* p, uint elem) { return *(const v4u32*)(p + elem); }

// one chunk's (channel block, ky, kx) in the layer's order
WG_FN void fg_chunk(int q, int order, int k, int nb, int* cb, int* ky, int* kx) {
    const int taps = k * k;
    if (order == 0) { *cb = q; *ky = 0; *kx = 0; }
    else if (order == 1) { *cb = q / taps; const int t = q - *cb * taps; *kx = t / k; *ky = t - *kx * k; }
    else { *kx = q / (nb * k); const int r = q - *kx * (nb * k); *cb = r / k; *ky = r - *cb * k; }
}

template <int MT>
WG_FN void fg_mconv(const u16* __restrict__ x, const u16* __restrict__ wf, const u16* __restrict__ b,
                    u16* __restrict__ y, u16* __restrict__ p, int cin, int cout, int H, int W, int k, int order,
                    int act, int pool, int tiles_x, int tiles_y) {
    const uint lane = wg_lane_id() & 31u, wave = wg_lane_id() >> 5;
    const int blk = (int)wg_block_id();
    const int tx = blk % tiles_x, rest = blk / tiles_x;
    const int ty = rest % tiles_y, mg = rest / tiles_y;
    const int n = (int)(lane & 15u), hh = (int)(lane >> 4);
    const int px = tx * 16 + n;                                      // this lane's pixel column
    const int py0 = ty * 16 + 2 * (int)wave;                         // rows py0 (N tile 0) and py0 + 1 (N tile 1)
    const int pad = k / 2, nb = cin / 16, nq = nb * k * k;
    // accumulators: [mt][nt], element v = output channel 16 (mg MT + mt) + 8 hh + v of pixel (px, py0 + nt)
    v8f a32[MT][2];
    for (int mt = 0; mt < MT; ++mt) {
        const int o0 = (mg * MT + mt) * 16 + 8 * hh;
        for (int v = 0; v < 8; ++v) {
            const float bv = fg_h2f(b[o0 + v]);
            for (int nt = 0; nt < 2; ++nt) a32[mt][nt][v] = bv;
        }
    }
    for (int q = 0; q < nq; ++q) {
        int cb, ky, kx;
        fg_chunk(q, order, k, nb, &cb, &ky, &kx);
        v8h fb[2], fa[MT];
        const int sx = px + kx - pad;
        for (int nt = 0; nt < 2; ++nt) {
            const int sy = py0 + nt + ky - pad;
            const bool in = sx >= 0 && sx < W && sy >= 0 && sy < H;
            const uint e = in ? (uint)((sy * W + sx) * cin + cb * 16 + 8 * hh) : 0u;
            v4u32 u = fg_ld128(x, e);
            if (!in) u = (v4u32){0u, 0u, 0u, 0u};                   // zero border
            fb[nt] = fg_as_v8h(u);
        }
        for (int mt = 0; mt < MT; ++mt)
            fa[mt] = fg_as_v8h(fg_ld128(wf, (uint)((((mg * MT + mt) * nq + q) * 32 + (int)lane) * 8)));
        for (int mt = 0; mt < MT; ++mt)
            for (int nt = 0; nt < 2; ++nt) {
                v8f d = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(fa[mt], fb[nt], a32[mt][nt]);
                for (int v = 0; v < 8; ++v) { const float t = d[v]; a32[mt][nt][v] = fg_h2f(fg_f2h(t)); }
            }
    }
    // epilogue: float16 bits, activation, store; pool
    for (int mt = 0; mt < MT; ++mt) {
        const int o0 = (mg * MT + mt) * 16 + 8 * hh;
        uint r[2][8];
        for (int nt = 0; nt < 2; ++nt)
            for (int v = 0; v < 8; ++v) {
                const float t = a32[mt][nt][v];
                uint hb = fg_f2h(t);
                if (act == 1) hb = fg_leaky16(hb);
                r[nt][v] = hb;
            }
        for (int nt = 0; nt < 2; ++nt) {
            const int py = py0 + nt;
            if (px < W && py < H) {
                const v4u32 u = {r[nt][0] | (r[nt][1] << 16), r[nt][2] | (r[nt][3] << 16), r[nt][4] | (r[nt][5] << 16),
                                 r[nt][6] | (r[nt][7] << 16)};
                *(v4u32*)(y + (size_t)(py * W + px) * cout + o0) = u;
            }
        }
        if (pool == 1) {
            uint m[8];
            for (int v = 0; v < 8; ++v) {
                const uint x01 = fg_xor1(r[0][v]), x11 = fg_xor1(r[1][v]);   // pixel px + 1 (lane n + 1)
                uint t = r[0][v];
                t = fg_pool_step16(t, x01);
                t = fg_pool_step16(t, r[1][v]);
                t = fg_pool_step16(t, x11);
                m[v] = t;
            }
            const int Ho = H / 2, Wo = W / 2, ox = px / 2, oy = py0 / 2;
            if ((n & 1) == 0 && ox < Wo && oy < Ho) {
                const v4u32 u = {m[0] | (m[1] << 16), m[2] | (m[3] << 16), m[4] | (m[5] << 16), m[6] | (m[7] << 16)};
                *(v4u32*)(p + (size_t)(oy * Wo + ox) * cout + o0) = u;
            }
        }
    }
}

#define FG_MCONV_ARGS const u16* __restrict__ x, const u16* __restrict__ wf, const u16* __restrict__ b, \
    u16* __restrict__ y, u16* __restrict__ p, int cin, int cout, int H, int W, int k, int order, int act, int pool, \
    int tiles_x, int tiles_y
#define FG_MCONV_CALL x, wf, b, y, p, cin, cout, H, W, k, order, act, pool, tiles_x, tiles_y
WG_KERNEL_N(fg_mconv1, 256)(FG_MCONV_ARGS) { fg_mconv<1>(FG_MCONV_CALL); }
WG_KERNEL_N(fg_mconv2, 256)(FG_MCONV_ARGS) { fg_mconv<2>(FG_MCONV_CALL); }
WG_KERNEL_N(fg_mconv4, 256)(FG_MCONV_ARGS) { fg_mconv<4>(FG_MCONV_CALL); }

// ---- upscale on NHWC (the arithmetic of fg_upscale_f16) ------------------------------------------------------------
struct FgTap { int i0, i1; float w0, w1; };
WG_FN FgTap fg_axis(float r, int o, int n_in) {
    FgTap t;
    const float s = r * (float)o;
    t.i0 = (int)s;
    t.i1 = (int)(s + 1.0f);
    if (t.i1 >= n_in) { t.i0 = n_in - 1; t.i1 = n_in - 1; }
    t.w1 = fg_h2f(fg_f2h(s - (float)t.i0));
    t.w0 = fg_h2f(fg_f2h(((float)t.i0 - s) + 1.0f));
    return t;
}
WG_FN float fg_lerp16(float wa, float a, float wb, float b) {
    return fg_h2f(fg_f2h(fg_h2f(fg_f2h(wa * a)) + fg_h2f(fg_f2h(wb * b))));
}

WG_KERNEL_N(fg_upscale_hwc_f16, 256)(const u16* __restrict__ x, u16* __restrict__ y, int C, int hi, int wi, int ho,
                                     int wo, uint rx_bits, uint ry_bits) {
    const int idx = (int)(wg_block_id() * 256u + wg_lane_id());
    if (idx >= C * ho * wo) return;
    const int c = idx % C;
    const int pix = idx / C;
    const int oy = pix / wo;
    const int ox = pix - oy * wo;
    const FgTap tx = fg_axis(__builtin_bit_cast(float, rx_bits), ox, wi);
    const FgTap ty = fg_axis(__builtin_bit_cast(float, ry_bits), oy, hi);
    const float top = fg_lerp16(tx.w0, fg_h2f(x[(ty.i0 * wi + tx.i0) * C + c]), tx.w1, fg_h2f(x[(ty.i0 * wi + tx.i1) * C + c]));
    const float bot = fg_lerp16(tx.w0, fg_h2f(x[(ty.i1 * wi + tx.i0) * C + c]), tx.w1, fg_h2f(x[(ty.i1 * wi + tx.i1) * C + c]));
    uint out = fg_f2h(fg_lerp16(ty.w0, top, ty.w1, bot));
    if (fg_is_nan16(out) != 0u) out = FG_NAN16;
    y[idx] = (u16)out;
}
