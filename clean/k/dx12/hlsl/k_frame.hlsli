// SPDX-License-Identifier: MIT
// D3D12 port of model K's frame inputs and their fetch helpers (clean/k/kernels/k_io.hip: KFrameIn, KInParams,
// KOutParams, KiHost, kio_*). WrongGPU's own code; the same operations and bits as the HIP helpers (k_fp.hlsli rules).
//
// Arguments. The HIP kernels take the structs by value with 64-bit pointers inside; the D3D12 ports get them as root
// constants packed by the device from the manifest's struct tokens (clean/runtime/struct_args.h): every pointer one
// DWORD (the arena byte offset, 0 = null), byte runs copied, skipped runs dropped. The macros below are the matching
// argument-block declarations (between WG_ARGS_BEGIN and WG_ARGS_END, in this order) and the loaders that rebuild the
// kernel-side view:
//   KInParams         s200:p4.b128.p5            K_ARGS_FRAME_IN K_ARGS_IN_PTRS                     41 DWORDs
//   KInUniformParams  s248:p4.b128.p5.b48        K_ARGS_FRAME_IN K_ARGS_IN_PTRS K_ARGS_HOST         53
//   KOutParams        s312:p4.b12.x24.b52.x12.b24.x4.p4.x8.p4.b8.p2.b56   K_ARGS_OUT                52
//     (skipped: cbx..zby, always 0 in the runtime (runtime_k.cpp sets no subrect base for the output stage); kappa / rho
//     / lam, which the kernels replace by KFrameIn.xs (the runtime always passes xs); pad_; exp2tab, the emulator's
//     table only: on the GPU ko_exp2h is v_exp_f16. K_LOAD_FRAME_OUT sets the skipped fields to 0 / 1.)
// Field names: F_ (KFrameIn), P_ (KInParams pointers), H_ (KiHost), O_ (KOutParams), U_ (KOutUni).
//
// Include after wg_platform.hlsli, the argument block and k_fp.hlsli. With WG_DIRECT_TEXTURES (rs=direct ports) the
// colour, depth and motion fetches read the game's textures t0 / t1 / t2 of clean/dx12/hlsl/native_direct.hlsli
// (include it first) instead of the arena (kio_texel / kio_motion / kio_depth; see there).
#pragma once

#define KIO_HDR 1
#define KIO_MVLO 2
#define KIO_INV 4
#define KIO_RESET 8
#define KIO_C16 16
#define KIO_M16 32
#define KIO_OUT_RGBA32F 0
#define KIO_OUT_RGBA16F 1
#define KIO_OUT_RGB32F 2

// ---- argument blocks (root constants, DWORD order = the struct tokens) --------------------------------------------
#define K_ARGS_FRAME_PTRS uint F_color, F_motion, F_depth, F_xs;
#define K_ARGS_FRAME_IN K_ARGS_FRAME_PTRS \
    int F_cpitch, F_mpitch, F_zpitch, F_cbx, F_cby, F_mbx, F_mby, F_zbx, F_zby, F_W, F_H, F_rw, F_rh, F_Bw, F_Bh, F_flags; \
    float F_jx, F_jy, F_sx, F_sy, F_mox, F_moy, F_kappa, F_rho, F_lam, F_s; \
    int F_mch; \
    float F_rmsx, F_rmsy, F_rmox, F_rmoy; \
    int F_pad_;
#define K_ARGS_IN_PTRS uint P_Hc, P_Hh, P_x0, P_cpre, P_cpre2;
#define K_ARGS_HOST float H_xi_x, H_xi_y, H_a_x, H_a_y, H_h_x, H_h_y, H_g_x, H_g_y, H_sp_x, H_sp_y, H_mp_x, H_mp_y;
#define K_ARGS_OUT K_ARGS_FRAME_PTRS \
    int F_cpitch, F_mpitch, F_zpitch; \
    int F_W, F_H, F_rw, F_rh, F_Bw, F_Bh, F_flags; \
    float F_jx, F_jy, F_sx, F_sy, F_mox, F_moy; \
    float F_s; \
    int F_mch; \
    float F_rmsx, F_rmsy, F_rmox, F_rmoy; \
    uint O_N, O_L, O_Hc, O_table; \
    uint O_out, O_Lout, O_Hcout, O_Hhout; \
    int O_opitch, O_omode; \
    uint O_ctap, O_ctapT; \
    float U_invW, U_invH, U_rhox, U_rhoy, U_rho2x, U_rho2y, U_jmx, U_jmy, U_spx, U_spy, U_invW16, U_invH16, U_kinv; \
    uint U_kux_kvy;

// ---- kernel-side structs --------------------------------------------------------------------------------------------
struct KFrameIn {
    uint color, motion, depth, xs;     // arena byte offsets (0 = null)
    int cpitch, mpitch, zpitch;
    int cbx, cby, mbx, mby, zbx, zby;  // subrect bases
    int W, H, rw, rh, Bw, Bh;          // output, render, base sizes
    int flags;                         // KIO_*
    float jx, jy, sx, sy, mox, moy;    // jitter, motion scale, motion offset
    float kappa, rho, lam;             // replaced by xs[0..2] when xs is set (kio_frame)
    float s;                           // spec 44's base-size factor (draft 3: 1/2)
    int mch;                           // KIO_M16: FP16 channels per motion texel
    float rmsx, rmsy, rmox, rmoy;      // KIO_M16: ms, mo of the raw motion
};
struct KInParams {
    KFrameIn f;
    uint Hc, Hh, x0, cpre, cpre2;
};
struct KiHost {  // spec 42 section 8
    float xi_x, xi_y, a_x, a_y, h_x, h_y, g_x, g_y, sp_x, sp_y, mp_x, mp_y;
    float of_x, of_y;  // KI_SGPR only (k_in.hlsl ki_host_sgpr): the motion offset in raw units; not part of the host's KiHost
};

#define K_LOAD_FRAME_PTRS(F) F.color = F_color; F.motion = F_motion; F.depth = F_depth; F.xs = F_xs;
#define K_LOAD_FRAME_COMMON(F) F.cpitch = F_cpitch; F.mpitch = F_mpitch; F.zpitch = F_zpitch; \
    F.W = F_W; F.H = F_H; F.rw = F_rw; F.rh = F_rh; F.Bw = F_Bw; F.Bh = F_Bh; F.flags = F_flags; \
    F.jx = F_jx; F.jy = F_jy; F.sx = F_sx; F.sy = F_sy; F.mox = F_mox; F.moy = F_moy; F.s = F_s; F.mch = F_mch; \
    F.rmsx = F_rmsx; F.rmsy = F_rmsy; F.rmox = F_rmox; F.rmoy = F_rmoy;
// KFrameIn from K_ARGS_FRAME_IN
#define K_LOAD_FRAME_IN(F) K_LOAD_FRAME_PTRS(F) K_LOAD_FRAME_COMMON(F) \
    F.cbx = F_cbx; F.cby = F_cby; F.mbx = F_mbx; F.mby = F_mby; F.zbx = F_zbx; F.zby = F_zby; \
    F.kappa = F_kappa; F.rho = F_rho; F.lam = F_lam;
// KFrameIn from K_ARGS_OUT (the skipped fields as the runtime passes them: bases 0, scalars 1 before kio_frame)
#define K_LOAD_FRAME_OUT(F) K_LOAD_FRAME_PTRS(F) K_LOAD_FRAME_COMMON(F) \
    F.cbx = 0; F.cby = 0; F.mbx = 0; F.mby = 0; F.zbx = 0; F.zby = 0; F.kappa = 1.0f; F.rho = 1.0f; F.lam = 1.0f;
#define K_LOAD_IN_PARAMS(P) K_LOAD_FRAME_IN(P.f) P.Hc = P_Hc; P.Hh = P_Hh; P.x0 = P_x0; P.cpre = P_cpre; P.cpre2 = P_cpre2;
#define K_LOAD_HOST(k) k.xi_x = H_xi_x; k.xi_y = H_xi_y; k.a_x = H_a_x; k.a_y = H_a_y; k.h_x = H_h_x; k.h_y = H_h_y; \
    k.g_x = H_g_x; k.g_y = H_g_y; k.sp_x = H_sp_x; k.sp_y = H_sp_y; k.mp_x = H_mp_x; k.mp_y = H_mp_y;

int kio_ceil16(int n) { return (n + 15) / 16 * 16; }

// The kernels' view: with KFrameIn.xs the exposure scalars come from the GPU (k_expo_scalars)
KFrameIn kio_frame(KFrameIn f)
{
    KFrameIn F = f;
    if (f.xs != 0u) {
        F.kappa = wg_ldf(f.xs, 0u);
        F.rho = wg_ldf(f.xs, 4u);
        F.lam = wg_ldf(f.xs, 8u);
    }
    return F;
}
KInParams ki_params(KInParams p)
{
    KInParams P = p;
    P.f = kio_frame(p.f);
    return P;
}

// spec 42 section 8, computed on the GPU (k_in_tiled; k_in_tiled_uniform takes the host's equal values)
KiHost ki_host(KFrameIn F)
{
    KiHost k;
    const int W16 = kio_ceil16(F.W), H16 = kio_ceil16(F.H);
    k.xi_x = wg_div(1.0f, (float)F.W * F.s) * 0.5f;                        // RN32(RN32(1/RN32(W s)) 0.5)
    k.xi_y = wg_div(1.0f, (float)F.H * F.s) * 0.5f;
    k.a_x = (float)F.rw * k.xi_x;
    k.a_y = (float)F.rh * k.xi_y;
    k.h_x = wg_div((float)F.W, (float)W16) * k.xi_x;
    k.h_y = wg_div((float)F.H, (float)H16) * k.xi_y;
    k.g_x = wg_div(0.25f, (float)F.Bw);
    k.g_y = wg_div(0.25f, (float)F.Bh);
    const bool mvlo = (F.flags & KIO_MVLO) != 0;
    k.sp_x = wg_div(F.sx, wg_div(1.0f, (float)F.W) * (float)(mvlo ? F.rw : F.W));
    k.sp_y = wg_div(F.sy, wg_div(1.0f, (float)F.H) * (float)(mvlo ? F.rh : F.H));
    k.mp_x = k.xi_x * (float)F.W;
    k.mp_y = k.xi_y * (float)F.H;
    return k;
}

// the motion factor with a texture's scale folded in (direct forms: b1's ms; KIO_M16: rmsx ..): spec 42 section 8's
// sp = s / (RN(1 / n) r), s = sx ms (the host's KiHost sp_x with that scale), and the raw-unit offset
float ki_dsp(float sx, float ms, int n, int r) { return wg_div(sx * ms, wg_div(1.0f, (float)n) * (float)r); }
float ki_dmo(float o, float mo, float ms) { return mo == 0.0f ? o : o + wg_div(mo, ms); }

// spec 42 section 3: index = b + min(i + d, n - 1) if i + d > 0, else b. For n >= 1 (every render / output size) that
// is b + max(min(i + d, n - 1), 0) (HIP's ko_clampi form: one v_maxmin_i32 instead of a compare, a minimum and a select)
int kio_tap(int i, int d, int b, int n) { return b + ki_max(ki_min(i + d, n - 1), 0); }
// [MVlo] motion fetch: i' = b + i + d; index = min(i' + b, b + n - 1) if i' > 0, else b; for n >= 1 that is
// b + max(min(i', n - 1), 0)
int kio_mtap(int i, int d, int b, int n) { return b + ki_max(ki_min(b + i + d, n - 1), 0); }

// colour texel (x, y) of the texture (subrect base included by the caller): FP32 RGBA, or RGBA16F (KIO_C16, exact
// widening). Direct: the game's colour texture t0 at (nd_rx + x, nd_ry + y), alpha 0 (unused by K), not divided by the
// pre-exposure (K applies kappa): the values of the HIP executor's RGBA16F input copy for a 16-bit float colour texture.
float4 kio_texel(KFrameIn F, int x, int y)
{
#if WG_DIRECT_TEXTURES
    return nd_raw_color_xy(x, y);
#else
    const uint i = ((uint)y * (uint)F.cpitch + (uint)x) * 4u;
    if ((F.flags & KIO_C16) != 0) {
        const uint2 h = wg_ld64(F.color, i * 2u);
        return float4(wg_h2f_lo(h.x), wg_h2f_hi(h.x), wg_h2f_lo(h.y), wg_h2f_hi(h.y));
    }
    return wg_ld4f(F.color, i * 4u);
#endif
}
float4 kio_color(KFrameIn F, int ix, int iy)
{
    return kio_texel(F, kio_tap(ix, 0, F.cbx, F.rw), kio_tap(iy, 0, F.cby, F.rh));
}
// motion texel (x, y) (subrect base included by the caller): FP32 pairs, or KIO_M16's raw FP16 value r. Direct: the raw
// value r of the game's motion texture t2. ki_dsp / ki_dmo apply the texture's scale and offset as the buffer form
// applies WgRtFrame's (below).
// KIO_M16 (the interop's raw-buffer route: input copies or model K's compact input pass, ms / mo in KFrameIn rmsx ..,
// WgRtFrame's scale 1): the raw FP16 value r, the scale folded into the motion factor and the offset RN(mo / ms) in raw
// units, as for the direct forms (issue #26; before: ms r + mo per texel, then the factor of scale 1).
// Direct forms (issue #26): the motion scale and offset of the game's motion texture t2 come in b1 (ms, mo; the
// executor passes 1 / 0 in WgRtFrame), and they are applied as the buffer forms and the reference (clean/k/reference
// instage.py / outstage.py, spec 42 / 43) apply WgRtFrame's mv_scale / mv_bias: the texel's raw value r plus the offset
// in raw units, times the motion factor with the scale folded in (sp = RN(s / RN(RN(1 / n) r)) of the host, s = the
// scale). Before, the direct forms scaled every texel (RN(ms r) + mo: input; one fused mad(ms, r, mo): output) and
// used the factor of scale 1: with a non-unit scale the products round differently, which moved history taps (0.02-0.04 %
// of the output and history values differed from the buffer form from the first frame after a reset, in-game packs).
// Offset: b1's mo is in render pixels after the scale (mo = ms b, one FP32 product); the raw-unit offset is RN(mo / ms)
// (exactly 0 for mo = 0; equal to the buffer drive's b whenever that quotient recovers it).
void kio_motion(KFrameIn F, int x, int y, out float mvx, out float mvy)
{
#if WG_DIRECT_TEXTURES
    const float2 r = native_direct_motion_tex.Load(int3((int)nd_rx + x, (int)nd_ry + y, 0)).xy;
    mvx = r.x;
    mvy = r.y;
#else
    if ((F.flags & KIO_M16) != 0) {
        const uint o = ((uint)y * (uint)F.mpitch + (uint)x) * (uint)F.mch * 2u;
        const uint r = wg_ld32(F.motion, o);                                 // channels 0, 1 (4-byte aligned)
        mvx = wg_h2f_lo(r);
        mvy = wg_h2f_hi(r);
        return;
    }
    const float2 m = asfloat(wg_ld64(F.motion, ((uint)y * (uint)F.mpitch + (uint)x) * 8u));
    mvx = m.x;
    mvy = m.y;
#endif
}

float kio_depth(KFrameIn F, int ix, int iy, int dx, int dy)
{
    const int x = kio_tap(ix, dx, F.zbx, F.rw), y = kio_tap(iy, dy, F.zby, F.rh);
#if WG_DIRECT_TEXTURES
    return native_direct_depth_tex.Load(int3((int)nd_rx + x, (int)nd_ry + y, 0));
#else
    return wg_ldf(F.depth, ((uint)y * (uint)F.zpitch + (uint)x) * 4u);
#endif
}

// spec 42 5.2 [MVlo]: offset of the nearest-depth motion pixel
void kio_nearest(KFrameIn F, int ix, int iy, out int odx, out int ody)
{
    float c, a, b, d, e;
#if !WG_DIRECT_TEXTURES
    [branch] if (ix >= 1 && ix <= F.rw - 2 && iy >= 1 && iy <= F.rh - 2) {
        const uint zp = (uint)F.zpitch * 4u;
        const uint voff = ((uint)(F.zby + iy - 1) * (uint)F.zpitch + (uint)(F.zbx + ix - 1)) * 4u;
        c = wg_ldf(F.depth + zp + 4u, voff);
        a = wg_ldf(F.depth, voff);
        b = wg_ldf(F.depth + 8u, voff);
        d = wg_ldf(F.depth + 2u * zp, voff);
        e = wg_ldf(F.depth + 2u * zp + 8u, voff);
    } else
#endif
    {
        c = kio_depth(F, ix, iy, 0, 0);
        a = kio_depth(F, ix, iy, -1, -1);
        b = kio_depth(F, ix, iy, 1, -1);
        d = kio_depth(F, ix, iy, -1, 1);
        e = kio_depth(F, ix, iy, 1, 1);
    }
    const float k = kf_bits(0x3A83126Fu);
    const bool inv = (F.flags & KIO_INV) != 0;
    bool edge;
    if (inv)
        edge = (c < 0.5f * (a + e) - kf_opq(k * c)) || (c < 0.5f * (b + d) - kf_opq(k * c));
    else
        edge = (c > kf_fma(c, k, 0.5f * (a + e))) || (c > kf_fma(c, k, 0.5f * (b + d)));
    int bx = 0, by = 0;
    if (edge) {
        float best = c;
        const float v[4] = {a, b, d, e};
        const int ox[4] = {-1, 1, -1, 1}, oy[4] = {-1, -1, 1, 1};
        [unroll] for (int q = 0; q < 4; ++q) {
            const bool nearer = inv ? v[q] > best : v[q] < best;
            if (nearer) {
                best = v[q];
                bx = ox[q];
                by = oy[q];
            }
        }
    }
    odx = bx;
    ody = by;
}

// spec 42 decode factor f (t f is linear) and spec 43 encode e(Y) (stored = Y e(Y)), correctly rounded steps
#define K_KIO_DECODE_NEAR kf_bits(0x3F7FBE88u)  // below it: 1 / (1 - t)
#define K_KIO_ENCODE_NEAR 1000.0f               // below it: 1 / (Y + 1)
float kio_decode(float t)
{
    if (t < K_KIO_DECODE_NEAR) return wg_div(1.0f, 1.0f - t);
    const float e = kf_opq((t + kf_bits(0x3C4FAE29u)) * 9.0f) * kf_bits(0x3FB8AA3Bu);
    return wg_div(kf_exp2(e) - 1.0f, 9.0f * t);
}
float kio_encode(float Y)
{
    if (Y < K_KIO_ENCODE_NEAR) return wg_div(1.0f, Y + 1.0f);
    return wg_div(wg_div(kf_ln(kf_fma(Y, 9.0f, 1.0f)), 9.0f) - kf_bits(0x3C4FAE29u), Y);
}

// bilinear sample of a 4-channel FP16 image (arena offset img, [h][w][4]) at normalized (cx, cy), clamp to edge;
// spec 42 section 1 (draft 2): t0 = v00 + (v10 - v00) fx, t1 = v01 + (v11 - v01) fx, (t0 (1 - fy)) + (t1 fy)
float4 kio_texel_h4(uint img, uint idx)
{
    const uint2 q = wg_ld64(img, idx * 8u);
    return float4(wg_h2f_lo(q.x), wg_h2f_hi(q.x), wg_h2f_lo(q.y), wg_h2f_hi(q.y));
}
float4 kio_bilinear4(uint img, int w, int h, float cx, float cy)
{
    const float px = kf_opq(cx * (float)w) - 0.5f, py = kf_opq(cy * (float)h) - 0.5f;
    const float fx0 = floor(px), fy0 = floor(py);
    const float fx = px - fx0, fy = py - fy0;
    // HIP: finite ? (int)max(min(f, w), -1) : (+inf: w, -inf: -1, NaN: 0). The clamp already gives w / -1 for +-inf
    // (min / max are IEEE minNum / maxNum), so only NaN needs the select
    const int x0 = isnan(fx0) ? 0 : (int)kf_max(kf_min(fx0, (float)w), -1.0f);
    const int y0 = isnan(fy0) ? 0 : (int)kf_max(kf_min(fy0, (float)h), -1.0f);
    uint2 q00u, q10u, q01u, q11u;
    [branch] if ((uint)x0 <= (uint)(w - 2) && (uint)y0 <= (uint)(h - 2)) {
        const uint voff = ((uint)y0 * (uint)w + (uint)x0) * 8u, r1 = img + (uint)w * 8u;
        q00u = wg_ld64(img, voff);
        q10u = wg_ld64(img + 8u, voff);
        q01u = wg_ld64(r1, voff);
        q11u = wg_ld64(r1 + 8u, voff);
    } else {
        const int xa = ki_max(ki_min(x0, w - 1), 0), xb = ki_max(ki_min(x0 + 1, w - 1), 0);
        const int ya = ki_max(ki_min(y0, h - 1), 0), yb = ki_max(ki_min(y0 + 1, h - 1), 0);
        q00u = wg_ld64(img, ((uint)ya * (uint)w + (uint)xa) * 8u);
        q10u = wg_ld64(img, ((uint)ya * (uint)w + (uint)xb) * 8u);
        q01u = wg_ld64(img, ((uint)yb * (uint)w + (uint)xa) * 8u);
        q11u = wg_ld64(img, ((uint)yb * (uint)w + (uint)xb) * 8u);
    }
    const float omfy = 1.0f - fy;
    const float4 q00 = float4(wg_h2f_lo(q00u.x), wg_h2f_hi(q00u.x), wg_h2f_lo(q00u.y), wg_h2f_hi(q00u.y));
    const float4 q10 = float4(wg_h2f_lo(q10u.x), wg_h2f_hi(q10u.x), wg_h2f_lo(q10u.y), wg_h2f_hi(q10u.y));
    const float4 q01 = float4(wg_h2f_lo(q01u.x), wg_h2f_hi(q01u.x), wg_h2f_lo(q01u.y), wg_h2f_hi(q01u.y));
    const float4 q11 = float4(wg_h2f_lo(q11u.x), wg_h2f_hi(q11u.x), wg_h2f_lo(q11u.y), wg_h2f_hi(q11u.y));
    const float4 t0 = q00 + kf_opq4(q10 - q00) * fx;
    const float4 t1 = q01 + kf_opq4(q11 - q01) * fx;
    return kf_opq4(t0 * omfy) + kf_opq4(t1 * fy);
}

// (Y, Co, Cg) of max(RGB, 0) scaled by kappa first (spec 42 6.2) or not (spec 43 4.2: kappa after)
void kio_ycocg(float R, float G, float B, out float Y, out float Co, out float Cg)
{
    Y = (0.25f * R + 0.5f * G) + 0.25f * B;
    Co = kf_fma(R, 0.5f, kf_opq(0.0f) * G) - 0.5f * B;                     // 0 G: NaN for an infinite G, as HIP
    Cg = (0.5f * G - 0.25f * R) - 0.25f * B;
}
float kio_luma_fma(float R, float G, float B) { return kf_fma(B, 0.25f, kf_fma(R, 0.25f, 0.5f * G)); }
