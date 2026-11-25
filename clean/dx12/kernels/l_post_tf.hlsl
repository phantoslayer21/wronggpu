// SPDX-License-Identifier: MIT
// Model L's true-fusion output kernel (D3D12 only, no HIP kernel of its own; runtime WG_L_POST_TF, default on): the
// reconstruction (l_post_recon_p12, the lc12 form: l_post.hlsl) and the output filter (w64/post_filter_c1_s4: the analytic
// Lanczos, Dec, times dec_scale) and the staged route's scale_rgb in ONE wave64 kernel of 512 lanes, the display image never
// goes to memory. The structure is Model M's post_tf (post.hlsl w64/post_tf_f32_p12_diet_vsit, POST_OUT = 0, WG_LZ_ANALYTIC):
// workgroup = (strip of 40 output columns = 64 D columns incl. the filter's halo, chunk of output row pairs); per step
// of 16 D rows each of the 8 waves64 reconstructs one row pair of the strip's 64 columns, the display values (FP16) go to
// a ring in LDS, and the horizontal and vertical Lanczos passes run on the ring. What differs is the reconstruction of a
// D pixel: Model L's own (l_post_core.hlsli, the code of l_post_recon_p12: the staged FP16 features, the 4 x 4 gather, the
// linear-colour history candidates, steps 3 to 5), per lane the same operations on the same values as l_post_recon_p12
// followed by post_filter_c1_s4 (a pixel is a function of its own inputs only; the staging only decides where a texel sits in
// LDS). The 8 FP16 roundings of a row use the driver's single-value conversion (op 0x1C: one hardware v_cvt_f16_f32, the
// bits of wg_f2h): wave64 has no matrix conversion (l_post_core.hlsli).
// The display value of D pixel (x, y) goes to the ring as the FP16 pair of lanes (x, x + 1) exactly as l_post_recon stores
// it to the display image (hc.y >> 16, hc.z), the filter's horizontal / vertical passes are post_filter_c1_s4's operations
// on those values (lz5a, clamp to [1e-4, 3], Dec) and the output is stored times dec_scale (the pre-exposure scale of the
// runtime's scale_rgb folded in, native_scale's bits: post.hlsl m1_scale; pi_arg == 1: the plain product).
// Arguments as w64/post_tf_f32_p12_diet_vsit (the analytic build): the post_recon arguments (A = image A8, B unused, C = image
// C4), the WgPostParams, expo, then ow, oh, dec_scale (read from the exposure pair at +16 when expo is not null: the diet pair),
// pi_arg (the pre-exposure; with _pf: the bits of P_pad0 divide every colour read, color_divide folded in), pitch_bytes (unused),
// cpairs, k1, c2 (post_tf's chunk plan). Staged render texels per step: at most kStageCap (host check l_tf_supported).
// @wg-kernel w64_lc12/l_post_tf_p12 co=- entry=l_post_tf_p12 block=512 wave=64 args=p,p,p,p,p,p,p,p,p,p,p,p,b64,p,u,u,u,u,u,u,u,u defines=LP_P12=1,LP_C12=1,LP_W64=1,LP_CODE=1,LP_RPUNI=1,LP_QX=1,LP_TF=1
// @wg-kernel w64_lc12/l_post_tf_p12_pf co=- entry=l_post_tf_p12 block=512 wave=64 args=p,p,p,p,p,p,p,p,p,p,p,p,b64,p,u,u,u,u,u,u,u,u defines=LP_P12=1,LP_C12=1,LP_W64=1,LP_CODE=1,LP_RPUNI=1,LP_QX=1,LP_TF=1,LP_PIDIV=1
#include "wg_platform.hlsli"
#ifndef LP_P12
#define LP_P12 1
#endif
#ifndef LP_C12
#define LP_C12 1
#endif
#define LP_LDR 0
#define LP_MORIG 0
#ifndef LP_W64
#define LP_W64 1
#endif
#ifndef LP_CODE
#define LP_CODE 1
#endif
#ifndef LP_RPUNI
#define LP_RPUNI 1
#endif
#ifndef LP_QX
#define LP_QX 1
#endif
#if LP_W64 != (WG_WAVE == 64) || WG_BLOCK != 512
#error "l_post_tf.hlsl: wave64, 512 lanes"
#endif
#if !LP_P12
#error "l_post_tf.hlsl: the pack12 inputs only"
#endif

WG_ARGS_BEGIN
    uint color, A, B, C, hist_prev, m_near, m_center, m_far, w_p, hist_out, reduced, out_p;   // pointers (A: A8, C: C4)
    int P_rw, P_rh, P_dw, P_dh, P_reset, P_pad0, P_pad1, P_pad2;                                // WgPostParams
    float P_rx, P_ry, P_jx, P_jy, P_e, P_e_prev, P_ratio, P_pad3;
    uint expo;
    int ow, oh;
    float dec_scale_arg, pi_arg;
    int pitch_bytes, cpairs, k1, c2;
WG_ARGS_END

#define kStageCap 512   // staged render texels per step (6 dwords each: 12288 bytes); one per lane
#ifndef LP_HSEQ
#define LP_HSEQ 1       // history_cr: two groups of loads (RGA: the 24 live load registers of its span form halve)
#endif
#define LP_FBLAZY 1     // the normalized features are not live together (RGA: -7 VGPRs)
#include "l_post_core.hlsli"

#define kTfCols 64
#define kTfOwn 60
#define kTfOutX 40
#define kNW 8                      // row pairs per step (= waves64 of the workgroup)
#define kRing (2 * kNW + 6)        // 22
#define kThreads (64 * kNW)        // 512
#define kRowDw (3 * kTfOutX)       // one ring row: 3 channels x 40 values = 120 dwords
#define kLdsDw ((kRing + 6) * kRowDw)   // 3360 dwords = 13440 bytes (+ 12288 staged texels + 1024 code table = 26752)
#if kThreads != WG_BLOCK
#error "l_post_tf.hlsl: the workgroup is 64 kNW lanes"
#endif

groupshared uint lds[kLdsDw];

static const float kLz0 = asfloat(0xbdba3b4bu), kLz1 = asfloat(0x3de71ca9u), kLz2 = asfloat(0x3f23b70fu);
static const float kLz3 = asfloat(0x3ed182b5u), kLz4 = asfloat(0xbd90a4acu);

float unif(float x) { return asfloat(WaveReadLaneFirst(asuint(x))); }
uint mul24(uint a, uint b) { return (a & 0xffffffu) * (b & 0xffffffu); }
int mul_i24(int a, int b) { return ((int)((uint)a << 8) >> 8) * ((int)((uint)b << 8) >> 8); }

// post_filter_c1_s4's lz5a: v = w0 t0, one fused multiply-add per further tap, clamp to the taps' range
float lz5a(float t0, float t1, float t2, float t3, float t4, float w0, float w1, float w3, float w4)
{
    float v = w0 * t0;
    v = mad(w1, t1, v);
    v = mad(kLz2, t2, v);
    v = mad(w3, t3, v);
    v = mad(w4, t4, v);
    const float mid_lo = min(min(t1, t2), t3), mid_hi = max(max(t1, t2), t3);
    const float lo = min(min(mid_lo, t0), t4), hi = max(max(mid_hi, t0), t4);
    return min(max(v, lo), hi);   // med3 (the taps are finite, lo <= hi)
}
uint lz5a_even(float t[10], int o) { return asuint(lz5a(t[o], t[o + 1], t[o + 2], t[o + 3], t[o + 4], kLz0, kLz1, kLz3, kLz4)); }
uint lz5a_odd(float t[10], int o) { return asuint(lz5a(t[o], t[o + 1], t[o + 2], t[o + 3], t[o + 4], kLz4, kLz3, kLz1, kLz0)); }

// native_scale (native_helpers.hlsl: the staged route's scale_rgb) of one value, as post.hlsl m1_scale
float m1_scale_ref(float a, float b)
{
    const float r = a * b;
    const uint ua = asuint(a), ub = asuint(b);
    const uint ea = (ua >> 23u) & 255u, eb = (ub >> 23u) & 255u;
    if ((asuint(r) & 0x7fffffffu) == 0u && ea != 0u && eb != 0u && ea + eb == 127u) {
        const uint64_t product = (uint64_t)((ua & 0x7fffffu) | 0x800000u) * (uint64_t)((ub & 0x7fffffu) | 0x800000u);
        const uint64_t normal = (uint64_t)1u << 47u;
        if (product >= normal - ((uint64_t)1u << 23u) && product < normal)
            return asfloat(((ua ^ ub) & 0x80000000u) | 0x00800000u);
    }
    return r;
}
float m1_scale(float a, float b)
{
    float r = a * b;
    [branch] if (and(abs(r) == 0.0f, abs(a) >= asfloat(0x00800000u))) r = m1_scale_ref(a, b);
    return r;
}

// the raw inputs of render pixel idx (staged one step ahead; the conversions happen when the texel is put)
struct LRaw { float4 c; uint2 d; uint c01; };
LRaw raw_fetch(int idx)
{
    LRaw r;
    const uint ui = (uint)idx;
    r.c = wg_ld4f(color, ui * 16u);
    r.d = wg_ld64(A, ui * 8u);
    r.c01 = wg_ld32(C, ui * 4u);
    return r;
}
void raw_put(LRaw r, uint dst, bool store)
{
    const uint a = r.d.x & 0xffffffu;
    const uint b = (r.d.x >> 24) | ((r.d.y & 0xffffu) << 8);
    const uint2 k = uint2(r.c01, r.d.y >> 16);
    uint t[kTexDw];
    texel_make(r.c, a, b, k, t);   // wave-uniform control flow (the driver's conversion)
    if (store) {
        [unroll] for (int d = 0; d < kTexDw; ++d) sh_tex[dst + (uint)d] = t[d];
    }
}

// The footprint of one step: rows r0 ..., ncol columns from c0; the render pixel of staged texel i (clamped)
int step_rows(int ys, int r_need, out int r0)
{
    const int yl = mini(ys + 2 * (kNW - 1), r_need) & ~1;
    r0 = floor_i(lpy(ys) - 0.5f) - 1;
    return floor_i(lpy(yl) - 0.5f) + 2 - r0 + 1;
}

WG_KERNEL(l_post_tf_p12)
{
    WG_SETUP();
    if (P_dw < 4) return;
    g_z = wg_grid_ >> 31;   // 0, opaque to the compiler
    g_e = P_e;
    g_ep = P_e_prev;
    float dec_scale = dec_scale_arg;
    if (expo != 0u) {   // the diet pair: [e, e_prev, last e, ratio, dec_scale]
        const float e0 = unif(wg_ldf(expo, 0u)), e1 = unif(wg_ldf(expo, 4u));
        g_e = e0 == 0.0f ? 1.0f : e0;
        g_ep = e1 == 0.0f ? 1.0f : e1;
        dec_scale = unif(wg_ldf(expo, 16u));
    }
    g_rp = asfloat(WaveReadLaneFirst(asuint(wg_div(1.0f, g_ep))));   // 1 / e_prev: wave-uniform operand
#if LP_PIDIV
    g_pd = pidiv_make((uint)P_pad0);
#endif
    const int lane = (int)wg_lane_id();
    const int wave = wg_uniform(lane >> 6), wl = lane & 63;   // wave index: uniform (a scalar register)
    const int strips = (ow + kTfOutX - 1) / kTfOutX;
    const int bid = (int)wg_block_id();
    // workgroup bid -> (strip, chunk) as post_tf: the strips of chunk c rotated by c (the same set of workgroups)
    const int chunk = wg_uniform(bid / strips);
    const int strip = wg_uniform((bid - chunk * strips + chunk) % strips);
    const int X0 = strip * kTfOwn;
    const int csize = chunk < k1 ? cpairs : c2;
    const int P0 = chunk < k1 ? chunk * cpairs : k1 * cpairs + (chunk - k1) * c2;
    const int p_end = wg_uniform(mini(P0 + csize, (oh + 1) / 2));
    const int own_r0 = wg_uniform(3 * P0), own_r1 = wg_uniform(mini(3 * (P0 + csize), P_dh));
    const int r_first = wg_uniform(3 * P0 - 2 < 0 ? 0 : 3 * P0 - 2);
    const int r_need = wg_uniform(mini(3 * (p_end - 1) + 4, P_dh - 1));
    const float s = wg_ldf(w_p, 0u);

    // the strip's render columns c0 ... c1 (taps ax - 1 ... ax + 2 of its first and last D column)
    const int c0 = floor_i(lpx(clampi(X0 - 2, 0, P_dw - 1)) - 0.5f) - 1;
    const int c1 = floor_i(lpx(clampi(X0 + kTfCols - 3, 0, P_dw - 1)) - 0.5f) + (kLWin - 2);
    const int ncol = c1 - c0 + 1;
    const float inv_ncol = unif(wg_div(1.0f, (float)ncol));

    // the staging of step y_step: footprint rows r0 ..., texels loaded one step ahead
    int r0, nstage;
    {
        const int nrow = step_rows(r_first, r_need, r0);
        nstage = ncol * nrow < kStageCap ? ncol * nrow : kStageCap;
    }
    LRaw raw;
    {
        const int i = mini(lane, nstage - 1);
        const int r = (int)(((float)i + 0.5f) * inv_ncol), c = i - mul_i24(r, ncol);
        raw = raw_fetch(clampi(r0 + r, 0, P_rh - 1) * P_rw + clampi(c0 + c, 0, P_rw - 1));
    }
    code16_build(lane & 255);   // the FP16 code table: waves 4 ... 7 rewrite the entries of waves 0 ... 3 with the same values
    wg_barrier_lds();           // (wave-uniform conversions: every wave runs the same instructions)

    // this lane's D column (fixed over the walk)
    const int xv = X0 - 2 + wl;
    const int x = clampi(xv, 0, P_dw - 1);
    const bool own_x = and(and(xv >= X0, xv < X0 + kTfOwn), xv < P_dw);
    const float px = lpx(x);
    const int ax = floor_i(px - 0.5f);
    const uint wlu = (uint)wl;

    int base_slot = 0;
    int p_next = P0;
    [loop] for (int y_step = r_first; y_step <= r_need; y_step += 2 * kNW) {
        // 1. stage the footprint
        [branch] if (wave * 64 < nstage) {   // wave-uniform: a wave wholly past the staged range has nothing to store
            raw_put(raw, (uint)lane * kTexDw, lane < nstage);
        }
        wg_barrier_lds();
        const int r0c = r0;

        // 2. reconstruction of the wave's row pair (yv, yv + 1)
        const int yv = y_step + 2 * wave;
        const bool wact = yv <= r_need;
        int slot = base_slot + 2 * wave;
        if (slot >= kRing) slot -= kRing;
        const uint scr = (uint)slot * kRowDw;   // dword of ring row slot (rows slot, slot + 1)
        if (wact) {
            const int y0 = yv;
            const bool has1 = y0 + 1 < P_dh;
            const float py0 = unif(lpy(y0)), py1 = unif(lpy(y0 + 1));   // wave-uniform (y0 is)
            const int b = wg_uniform(floor_i(py0 - 0.5f));
            const int b1 = wg_uniform(floor_i(py1 - 0.5f));
            // gather offsets: dx = h((ax + 1/2) - px), d_y = h((b_y + 1/2) - py(y)) minus 1 (FP16) when the row's own b_y
            // differs from the pair's b; the three offsets in one conversion (wave64: three hardware conversions)
            v8f ov;
            ov.v[0] = opq(((float)ax + 0.5f) - px);
            ov.v[1] = opq(((float)b + 0.5f) - py0);
            ov.v[2] = opq(((float)b1 + 0.5f) - py1);
            [unroll] for (int oi = 3; oi < 8; ++oi) ov.v[oi] = 0.0f;
            const uint4 oh16 = wg_to_f16_ags(ov);
            const half dx = h_bits(oh16.x), dy0 = h_bits(oh16.x >> 16);
            half dy1 = h_bits(oh16.y);
            if (b1 != b) dy1 = dy1 - (half)1.0f;
            half v0[kLWin], v1[kLWin], vv0[kLWin], vv1[kLWin];
            [unroll] for (int jj = 0; jj < kLWin; ++jj) {
                v0[jj] = dy0 + (half)(float)(jj - 1);
                v1[jj] = dy1 + (half)(float)(jj - 1);
                vv0[jj] = mulh(v0[jj], v0[jj]);
                vv1[jj] = mulh(v1[jj], v1[jj]);
            }
            // taps column by column (i outer, j inner), both rows' gathers in one pass over the staged texels
            LGather g0 = gather_zero(), g1 = gather_zero();
            const int tb0 = ((b - 1 - r0c) * ncol + (ax - 1 - c0)) * kTexDw, rstride = ncol * kTexDw;
            [loop] for (int i = -1; i <= (kLWin - 2); ++i)
                gather_col<true>(i, dx, tb0, rstride, b, ax, v0, v1, vv0, vv1, g0, g1);

            float r[3] = {0.0f, 0.0f, 0.0f};
            [loop] for (int row = 0; row < 2; ++row) {
                const bool second = row == 1;
                if (second && !has1) break;   // wave-uniform: the missing second row is not stored and adds nothing
                const int y = y0 + row;
                LGather g = g0;
                if (second) g = g1;
                float hh[3], oo[3];
                const MotRaw mot = motion_load(px, second ? py1 : py0);
                finish_pixel(g, x, y, mot, s, hh, oo);
                v8f cv;
                float el[3];
                [unroll] for (int ci = 0; ci < 3; ++ci) {
                    el[ci] = enc_log(hh[ci]);
                    cv.v[ci] = mul_odd(kEncK, el[ci]);
                    cv.v[3 + ci] = opq(clampf(enc_e(oo[ci]), 0.0f, kDisplayMax));
                }
                cv.v[6] = 0.0f;
                cv.v[7] = 0.0f;
                const uint4 hc = wg_to_f16_ags(cv);
                if (and(and(own_x, y >= own_r0), y < own_r1)) {
                    const uint q0 = mul_zero_k(el[0]) ? 0u : (hc.x & 0xffffu);
                    const uint q1 = mul_zero_k(el[1]) ? 0u : (hc.x >> 16);
                    const uint q2 = mul_zero_k(el[2]) ? 0u : (hc.y & 0xffffu);
                    wg_st64(hist_out, (uint)(y * P_dw + x) * 8u, uint2(q0 | (q1 << 16), q2));
                }
                // display values (FP16) of the 64 columns -> scratch row slot + row: lanes 2m, 2m + 1 share a dword
                const uint srow = scr + (uint)row * kRowDw;
                [unroll] for (int c = 0; c < 3; ++c) {
                    const uint dv = c == 0 ? hc.y >> 16 : (c == 1 ? hc.z & 0xffffu : hc.z >> 16);
                    const uint nbv = QuadReadAcrossX(dv);
                    if ((wlu & 1u) == 0u) lds[srow + (uint)c * 32u + (wlu >> 1)] = dv | (nbv << 16);
                }
                // reduced history: mean of H^ over the 2 x 2 block, r = h0 + h1
                [unroll] for (int ri = 0; ri < 3; ++ri) r[ri] = second ? r[ri] + hh[ri] : hh[ri];
            }
            float o[3];
            [unroll] for (int i = 0; i < 3; ++i) o[i] = QuadReadAcrossX(r[i]);   // lane ^ 1
            if (and(and(and(and(and((wl & 1) == 0, own_x), x + 1 < P_dw), has1), y0 >= own_r0), y0 + 1 < own_r1)) {
                uint code = 0u;
                [unroll] for (int i = 0; i < 3; ++i) {
                    const float m = opq((r[i] + o[i]) * 0.25f);
                    const float v = clampf(opq(enc_e(m)) * kInv245, 0.0f, 1.0f);
                    code |= wg_f2u(round(v * 1023.0f)) << (10 * i);
                }
                wg_st32(reduced, (uint)((y0 / 2) * (P_dw / 2) + x / 2) * 4u, code);
            }
        }
        // the next step's staging loads (every lane loads a valid clamped texel)
        {
            const int nrow = step_rows(y_step + 2 * kNW, r_need, r0);
            nstage = ncol * nrow < kStageCap ? ncol * nrow : kStageCap;
            const int i = mini(lane, nstage - 1 < 0 ? 0 : nstage - 1);
            const int r = (int)(((float)i + 0.5f) * inv_ncol), c = i - mul_i24(r, ncol);
            raw = raw_fetch(clampi(r0 + r, 0, P_rh - 1) * P_rw + clampi(c0 + c, 0, P_rw - 1));
        }

        // horizontal pass of the wave's two rows: lane (row hk, channel hc, cell hg), output columns 4 hg ... 4 hg + 3
        GroupMemoryBarrier();   // the 64 lanes are one wave: its LDS operations execute in order (the fence orders the compiler)
        const int hl = wl;
        const bool hact = and(wact, hl < 60);
        const int hk = hl >= 30 ? 1 : 0, hrem = hl - 30 * hk, hc = hrem >= 20 ? 2 : (hrem >= 10 ? 1 : 0),
                  hg = hrem - 10 * hc;
        uint hv[4] = { 0u, 0u, 0u, 0u };
        if (hact) {
            const uint src = scr + (uint)hk * kRowDw + (uint)hc * 32u + (uint)hg * 3u;
            float t[10];   // the FP16 display taps as FP32 values (exact); results FP32 bits
            [unroll] for (int i = 0; i < 5; ++i) {
                const uint q = lds[src + (uint)i];
                t[2 * i] = f16tof32(q);
                t[2 * i + 1] = f16tof32(q >> 16);
            }
            hv[0] = lz5a_even(t, 0);
            hv[1] = lz5a_odd(t, 2);
            hv[2] = lz5a_even(t, 3);
            hv[3] = lz5a_odd(t, 5);
        }
        GroupMemoryBarrier();
        if (hact) {
            const uint dst = scr + (uint)hk * kRowDw + (uint)hc * (uint)kTfOutX + (uint)hg * 4u;
            [unroll] for (int i = 0; i < 4; ++i) lds[dst + (uint)i] = hv[i];
            if (slot + hk < 6) {   // the first six ring rows are mirrored after the ring (wrap-free vertical taps)
                const uint dm = dst + (uint)kRing * kRowDw;
                [unroll] for (int i2 = 0; i2 < 4; ++i2) lds[dm + (uint)i2] = hv[i2];
            }
        }
        wg_barrier_lds();

        // 3. vertical pass, Dec and output of the row pairs whose input rows are complete
        const int r_done = mini(y_step + 2 * kNW, r_need + 1);
        const int p_rdy = r_done > r_need ? p_end : mini(p_end, (r_done - 5) / 3 + 1);
        const int nitems = (p_rdy - p_next) * 2 * kTfOutX;
        const bool interior = and(3 * p_next - 2 >= 0, 3 * p_rdy + 1 <= P_dh - 1);
        const int s0 = interior ? (3 * p_next - 2 - r_first) % kRing : 0;
        [loop] for (int it = lane; it < nitems; it += kThreads) {
            const int pr = (int)(mul24((uint)it, 52429u) >> 22);
            const int rem = it - (int)mul24((uint)pr, 2u * kTfOutX);
            const int dy = rem >= kTfOutX ? 1 : 0, xo = rem - dy * kTfOutX;
            const int pp = p_next + pr, y = 2 * pp + dy, xg = strip * kTfOutX + xo;
            if (or(y >= oh, xg >= ow)) continue;
            uint t[5][3];
            if (interior) {   // one ring address, the 15 taps at constant offsets (mirrored rows: no wrap)
                int u = s0 + 3 * pr + 2 * dy;
                u = u >= kRing ? u - kRing : u;
                u = u >= kRing ? u - kRing : u;
                const uint rp = mul24((uint)u, (uint)kRowDw) + (uint)xo;
                [unroll] for (int k = 0; k < 5; ++k)
                    [unroll] for (int c = 0; c < 3; ++c) t[k][c] = lds[rp + (uint)(k * kRowDw + c * kTfOutX)];
            } else {
                const int xr = 3 * pp - 2 + 2 * dy;
                const int row0 = clampi(xr, 0, P_dh - 1);
                const int sl0 = (int)((uint)(row0 - r_first) % (uint)kRing);
                [unroll] for (int k = 0; k < 5; ++k) {
                    const int st = clampi(xr + k, 0, P_dh - 1) - row0;
                    const int sl = sl0 + st >= kRing ? sl0 + st - kRing : sl0 + st;
                    [unroll] for (int c = 0; c < 3; ++c) t[k][c] = lds[(uint)sl * kRowDw + (uint)(c * kTfOutX + xo)];
                }
            }
            // emit: weights of the lane's phase by selects, lz5a, Dec, store 3 x f32
            const bool ev = dy == 0;
            const float w0 = ev ? kLz0 : kLz4, w4 = ev ? kLz4 : kLz0, w1 = ev ? kLz1 : kLz3, w3 = ev ? kLz3 : kLz1;
            float vo[3], qv[3];
            bool vsmall = false;
            [unroll] for (int c = 0; c < 3; ++c) {
                qv[c] = clampf(lz5a(asfloat(t[0][c]), asfloat(t[1][c]), asfloat(t[2][c]), asfloat(t[3][c]), asfloat(t[4][c]),
                                    w0, w1, w3, w4), k1em4, 3.0f);
                vo[c] = exp2(qv[c] * kDecK) - 1.0f;
                vsmall = or(vsmall, qv[c] < kDecPolyMax);
            }
            if (WaveActiveAnyTrue(vsmall)) {   // Dec's polynomial only when some lane needs it (post.hip's branches)
                [unroll] for (int c = 0; c < 3; ++c) vo[c] = qv[c] < kDecPolyMax ? dec_m1_poly(qv[c]) : vo[c];
            }
            const uint o = mul24((uint)y, (uint)ow * 12u) + mul24((uint)xg, 12u);
#if LP_PIDIV
            {   // _pf runs only when pi != 1: the scale_rgb fold is unconditional
                const float3 res = float3(mul_nz(vo[0], dec_scale), mul_nz(vo[1], dec_scale), mul_nz(vo[2], dec_scale));
                const float3 so = float3(m1_scale(res.x, pi_arg), m1_scale(res.y, pi_arg), m1_scale(res.z, pi_arg));
                wg_mem.Store3(out_p + o, uint3(asuint(so.x), asuint(so.y), asuint(so.z)));
            }
#else
            [branch] if (pi_arg == 1.0f) {
                wg_mem.Store3(out_p + o, uint3(asuint(vo[0] * dec_scale), asuint(vo[1] * dec_scale), asuint(vo[2] * dec_scale)));
            } else {   // the runtime's scale_rgb folded in (the division by pi done by a separate color_divide)
                const float3 res = float3(mul_nz(vo[0], dec_scale), mul_nz(vo[1], dec_scale), mul_nz(vo[2], dec_scale));
                const float3 so = float3(m1_scale(res.x, pi_arg), m1_scale(res.y, pi_arg), m1_scale(res.z, pi_arg));
                wg_mem.Store3(out_p + o, uint3(asuint(so.x), asuint(so.y), asuint(so.z)));
            }
#endif
        }
        p_next = p_rdy;
        base_slot += 2 * kNW;
        if (base_slot >= kRing) base_slot -= kRing;
    }
}
