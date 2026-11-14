// SPDX-License-Identifier: MIT
// D3D12 port of w64/post_tf_f32_p12_diet_vsit (clean/kernels/post.hip, the runtime's default output kernel): true fusion
// of the output reconstruction (stage "post") and the output filter (stage "filter") in one wave64 kernel of 512 lanes.
// Build variant: post_w64__orig.co (post.hip with -mwavefrontsize64 -DWG_LZ_PROFILE=2), the code object the runtime loads
// under the tag "w64" when the model file has output-filter weights (output_filter=original, the default then): the five
// FP16 filter weights are the last five kernel arguments and the filter accumulates in FP16 steps (v_fma_f16).
// Template instance: post_tf_body<NW = 8, kH = false, CAP = 352, kDiet, kVDiet, kSel, kIdx, kTrim, MoOff> with the
// build defaults WG_TF_PIPE = 2, WG_TF_VMODE = 1, WG_TF_VDP = 15, WG_TF_MIRROR = 1, WG_TF_DECANY = 0.
// The same operations in the same order with the same roundings as the HIP ISA (post.hip compiles with
// "#pragma clang fp contract(off)": only its explicit fmaf are fused, written here as mad). The places where HIP's ISA
// differs from a literal reading of the source: the history texel (_Float16)(kEncK * log2(..)) is one v_fma_mixlo_f16
// (one rounding: the driver folds mad(a, kEncK, 0) into its conversion as that instruction); (_Float16) casts are
// v_cvt_f16_f32 (round to nearest even: the driver's matrix conversion, cvt3_f16 below); 1 / ncol is an IEEE division
// (wg_div).
// Where the driver's compiler would regroup or fuse operations, opq() cuts the chain (below).
// Checked byte for byte against the HIP kernel by clean/dx12/compare.py (every stage and state buffer, both cases).
// Output forms (POST_OUT): 0 = 3 x f32 rows (post_tf_f32_p12_diet_vsit, the runtime's own output); 1 = RGBA16F rows of
// pitch_bytes with pi folded in (post_tf_p12_diet_vsit, post.hip kH: the interop output copy, FrameIo.out_half_pitch);
// 2 = the same RGBA16F values written straight into the game's output texture (RWTexture2D at u3, texel (x, y) of the
// output; D3D12 only, no HIP kernel of its own: the bytes of form 1). Form 2 needs a root signature with the ports'
// parameters plus a descriptor table holding that texture's UAV at u3 (rs=tex; clean/dx12/README.md).
// 3 = direct typed color input and typed output, reproducing form 0 followed by the runtime's optional scale_rgb and
// Converter output store. It preserves FP32 decode-scale rounding, native_scale's boundary repair and alpha 1; unlike
// forms 1/2 it does not clamp to FP16 or explicitly round to half. The UAV format supplies the final format conversion.
// Form 3 uses native_direct.hlsli's immutable frame b1 and texture table (rs=direct).
// @wg-kernel w64/post_tf_f32_p12_diet_vsit co=post_w64__orig.co entry=post_tf_f32_p12_diet_vsit block=512 wave=64 args=p,p,p,p,p,p,p,p,p,p,p,p,b64,p,u,u,u,u,u,u,u,u,u,u,u,u,u
// @wg-kernel w64/post_tf_p12_diet_vsit co=post_w64__orig.co entry=post_tf_p12_diet_vsit block=512 wave=64 args=p,p,p,p,p,p,p,p,p,p,p,p,b64,p,u,u,u,u,u,u,u,u,u,u,u,u,u defines=POST_OUT=1
// @wg-kernel w64/post_tf_p12_diet_vsit_tex co=post_w64__orig.co rs=tex entry=post_tf_p12_diet_vsit block=512 wave=64 args=p,p,p,p,p,p,p,p,p,p,p,p,b64,p,u,u,u,u,u,u,u,u,u,u,u,u,u defines=POST_OUT=2
// @wg-kernel w64/post_tf_f32_p12_diet_vsit_direct co=post_w64__orig.co rs=direct entry=post_tf_f32_p12_diet_vsit_direct block=512 wave=64 args=p,p,p,p,p,p,p,p,p,p,p,p,b64,p,u,u,u,u,u,u,u,u,u,u,u,u,u defines=WG_DIRECT_TEXTURES=1,POST_OUT=3
//
// Motion at output resolution (WG_POST_MO; HIP: post_tf_f32_p12_mo of kernels/post_mo.hip, code object
// post_mo_w64__orig.co under the tag "pmo": post_tf_body<8, false, 352, ..., MoOn>, post_motion_outres.h). Forms 0 and 3
// only. Per D pixel one point read of the raw motion at T = (clamp(trunc(RN32(RN32((x + 1/2) / dw) Mw)), 0, Mw - 1),
// likewise y), q = (fma(sx, bx + MVx(T), x + 1/2), fma(sy, by + MVy(T), y + 1/2)), one history sample S (or the fallback),
// H = fma(w1, S, w2 S) (w2 S = 0 when w2 = 0), a NaN q blends with 1 - beta' = 0 (mo_core); the motion surfaces are not
// staged. The texels of a lane's row pair are fetched at the start of each step (mo_fetch), as HIP. Arguments as the
// kernels above with (m_near, m_center, m_far) replaced by (m_near, motion) and WgMoParams (mw, mh, pitch, wide, sx, sy,
// bx, by, idw, idh) after c2. Buffer form: the raw motion buffer (wg_motion_hi.hlsli); direct form: the game's motion
// texture (its `motion` pointer is not read).
// @wg-kernel pmo/post_tf_f32_p12_mo co=post_mo_w64__orig.co entry=post_tf_f32_p12_diet_vsit block=512 wave=64 args=p,p,p,p,p,p,p,p,p,p,p,b64,p,u,u,u,u,u,u,u,u,b40,u,u,u,u,u defines=WG_POST_MO=1
// @wg-kernel pmo/post_tf_f32_p12_mo_direct co=post_mo_w64__orig.co rs=direct entry=post_tf_f32_p12_diet_vsit_direct block=512 wave=64 args=p,p,p,p,p,p,p,p,p,p,p,b64,p,u,u,u,u,u,u,u,u,b40,u,u,u,u,u defines=WG_DIRECT_TEXTURES=1,POST_OUT=3,WG_POST_MO=1
//
// Native AA and ratios between 1.0 and 1.5 (Brief 15; D3D12 only, no HIP kernel of its own): the same body with NW = 4
// row pairs per step (4 waves64, 256 lanes) and a staging cap of 448 texels (WG_TF_NW, WG_TF_CAP), two staged texels per
// lane. HIP runs such frames in post_tf_dl_* (NW = 8, cap 644: 41472 bytes of LDS, over D3D12's 32 KB of groupshared
// memory). Each output pixel's operations are those of the kernels above (the staging only decides where a texel sits in
// LDS); runtime.cpp tf4_supported is the host bound (the step footprint of NW = 4 fits 448). LDS: 448 x 48 + 20 x 480 =
// 31104 bytes. The kernels above keep their DXIL (the defaults below).
// @wg-kernel w64/post_tf4_dl_f32_p12_diet_vsit co=post_w64__orig.co entry=post_tf_f32_p12_diet_vsit block=256 wave=64 args=p,p,p,p,p,p,p,p,p,p,p,p,b64,p,u,u,u,u,u,u,u,u,u,u,u,u,u defines=WG_TF_NW=4,WG_TF_CAP=448
// @wg-kernel w64/post_tf4_dl_f32_p12_diet_vsit_direct co=post_w64__orig.co rs=direct entry=post_tf_f32_p12_diet_vsit_direct block=256 wave=64 args=p,p,p,p,p,p,p,p,p,p,p,p,b64,p,u,u,u,u,u,u,u,u,u,u,u,u,u defines=WG_DIRECT_TEXTURES=1,POST_OUT=3,WG_TF_NW=4,WG_TF_CAP=448
// @wg-kernel pmo/post_tf4_dl_f32_p12_mo co=post_mo_w64__orig.co entry=post_tf_f32_p12_diet_vsit block=256 wave=64 args=p,p,p,p,p,p,p,p,p,p,p,b64,p,u,u,u,u,u,u,u,u,b40,u,u,u,u,u defines=WG_POST_MO=1,WG_TF_NW=4,WG_TF_CAP=448
// @wg-kernel pmo/post_tf4_dl_f32_p12_mo_direct co=post_mo_w64__orig.co rs=direct entry=post_tf_f32_p12_diet_vsit_direct block=256 wave=64 args=p,p,p,p,p,p,p,p,p,p,p,b64,p,u,u,u,u,u,u,u,u,b40,u,u,u,u,u defines=WG_DIRECT_TEXTURES=1,POST_OUT=3,WG_POST_MO=1,WG_TF_NW=4,WG_TF_CAP=448
//
// Analytic output filter (WG_LZ_ANALYTIC = 1; HIP: post_w64.co, post.hip built with -mwavefrontsize64 and the default
// WG_LZ_PROFILE = 0, the build the runtime loads under "w64" when the model file has no output-filter weights, e.g.
// Model L's): the same kernels with spec 06's own Lanczos weights as FP32 constants (kLz0 ... kLz4): v = kLz0 t0, then
// one fused multiply-add per tap, clamp to the taps' range, every value FP32 (the horizontal results go to the ring as
// FP32, the vertical result straight into Dec). No weight arguments. Forms 0 and 3 (direct; without the FP16-indexed
// output table, which needs the original's FP16 filter result).
// @wg-kernel w64/post_tf_f32_p12_diet_vsit co=post_w64.co file=w64__post_tf_f32_p12_diet_vsit__analytic entry=post_tf_f32_p12_diet_vsit block=512 wave=64 args=p,p,p,p,p,p,p,p,p,p,p,p,b64,p,u,u,u,u,u,u,u,u defines=WG_LZ_ANALYTIC=1
// @wg-kernel w64/post_tf4_dl_f32_p12_diet_vsit co=post_w64.co file=w64__post_tf4_dl_f32_p12_diet_vsit__analytic entry=post_tf_f32_p12_diet_vsit block=256 wave=64 args=p,p,p,p,p,p,p,p,p,p,p,p,b64,p,u,u,u,u,u,u,u,u defines=WG_TF_NW=4,WG_TF_CAP=448,WG_LZ_ANALYTIC=1
// @wg-kernel w64/post_tf_f32_p12_diet_vsit_direct co=post_w64.co file=w64__post_tf_f32_p12_diet_vsit_direct__analytic rs=direct entry=post_tf_f32_p12_diet_vsit_direct block=512 wave=64 args=p,p,p,p,p,p,p,p,p,p,p,p,b64,p,u,u,u,u,u,u,u,u defines=WG_DIRECT_TEXTURES=1,POST_OUT=3,WG_LZ_ANALYTIC=1
// @wg-kernel w64/post_tf4_dl_f32_p12_diet_vsit_direct co=post_w64.co file=w64__post_tf4_dl_f32_p12_diet_vsit_direct__analytic rs=direct entry=post_tf_f32_p12_diet_vsit_direct block=256 wave=64 args=p,p,p,p,p,p,p,p,p,p,p,p,b64,p,u,u,u,u,u,u,u,u defines=WG_DIRECT_TEXTURES=1,POST_OUT=3,WG_TF_NW=4,WG_TF_CAP=448,WG_LZ_ANALYTIC=1
#include "wg_platform.hlsli"
#ifndef WG_LZ_ANALYTIC
#define WG_LZ_ANALYTIC 0
#endif
#if WG_LZ_ANALYTIC && ((POST_OUT != 0 && POST_OUT != 3) || defined(WG_POST_MO))
#error "analytic output filter: forms 0 and 3 (direct) only"
#endif
#if defined(WG_DIRECT_TEXTURES)
#include "native_direct.hlsli"
#include "wg_f16lut.hlsli"
#endif
#ifndef POST_OUT
#define POST_OUT 0
#endif
#if POST_OUT == 2
RWTexture2D<float4> wg_tex_out : register(u3);   // the output texture (RGBA16F), written through a typed UAV
#endif

#if defined(WG_POST_MO) && (POST_OUT == 1 || POST_OUT == 2)
#error "motion at output resolution: forms 0 and 3 only"
#endif
#if defined(WG_POST_MO)
#include "wg_motion_hi.hlsli"
#endif

WG_ARGS_BEGIN
#if defined(WG_POST_MO)
    uint color, A8, B_unused, C4, hist_prev, m_near, motion_p, w_p, hist_out, reduced, out_p;  // pointers
#else
    uint color, A8, B_unused, C4, hist_prev, m_near, m_center, m_far, w_p, hist_out, reduced, out_p;  // pointers
#endif
    int P_rw, P_rh, P_dw, P_dh, P_reset, P_pad0, P_pad1, P_pad2;                                      // WgPostParams
    float P_rx, P_ry, P_jx, P_jy, P_e, P_e_prev, P_ratio, P_pad3;
    uint expo;
    int ow, oh;
    float dec_scale_arg, pi_arg;
    int pitch_bytes, cpairs, k1, c2;
#if defined(WG_POST_MO)
    int mo_mw, mo_mh, mo_pitch, mo_wide;   // WgMoParams (40 bytes)
    float mo_sx, mo_sy, mo_bx, mo_by, mo_idw, mo_idh;
#endif
#if !WG_LZ_ANALYTIC
    uint lz_w0, lz_w1, lz_w2, lz_w3, lz_w4;   // FP16 bit patterns of the filter weights of taps -2 ... +2
#endif
WG_ARGS_END

// ---- constants (FP32 bit patterns of post.hip's constexpr values) ---------------------------------------------------
static const float kEncK = asfloat(0x3dddf184u);        // 0.36f * 0.30102999566398120f
static const float kDecK = asfloat(0x4113a428u);        // 3.32192809488736235f / 0.36f
static const float kDecLn = asfloat(0x40ccac9au);       // 2.30258509299404568f / 0.36f
static const float kTapK = asfloat(0xbf38aa3bu);        // -0.5f * kLog2E
static const float kTapK2 = asfloat(0xbfb8aa3bu);       // 2.0f * kTapK
static const float kDecPolyMax = asfloat(0x3d60233du);  // 0.35f / kDecLn
static const float kInv255 = asfloat(0x3b808081u);      // 1.0f / 255.0f
static const float kInv245 = asfloat(0x3ed0fac6u);      // 1.0f / 2.45f
static const float kP5040 = asfloat(0x39500d01u), kP720 = asfloat(0x3ab60b61u), kP120 = asfloat(0x3c088889u);
static const float kP24 = asfloat(0x3d2aaaabu), kP6 = asfloat(0x3e2aaaabu);
static const float k1em4 = asfloat(0x38d1b717u);        // 1e-4f
static const float k1em10 = asfloat(0x2edbe6ffu);       // 1e-10f
static const float k1p1 = asfloat(0x3f8ccccdu);         // 1.1f
static const float k0p05 = asfloat(0x3d4ccccdu);        // 0.05f
static const float kDisplayMax = asfloat(0x401ccccdu);  // 2.45f
static const float k0p01 = asfloat(0x3c23d70au);        // 0.01f

#define kTfCols 64
#define kTfOwn 60
#define kTfOutX 40
#ifndef WG_TF_NW
#define WG_TF_NW 8                 // row pairs per step (= waves64 of the workgroup); 4: post_tf4_dl_* (Brief 15)
#endif
#ifndef WG_TF_CAP
#define WG_TF_CAP 352              // staged render texels per step; 448: post_tf4_dl_*
#endif
#define kNW WG_TF_NW
#define kRing (2 * kNW + 6)        // 22
#define kThreads (64 * kNW)        // 512
#define kTfCap WG_TF_CAP
#if kThreads != WG_BLOCK
#error "WG_TF_NW waves64 must make the workgroup (block= of the @wg-kernel line)"
#endif
#if kTfCap > 2 * kThreads
#error "at most two staged texels per lane"
#endif
// two staged texels per lane (lane and lane + kThreads) when the cap exceeds the workgroup's lanes (post_tf4_dl_*)
#define WG_TF_TWO (kTfCap > kThreads)
#define kTexDw 12                 // R2Texel: 48 bytes
#define kRowDw (3 * kTfOutX)       // one ring row: 3 channels x 40 floats = 120 dwords (480 bytes)
#define kRingBase (kTfCap * kTexDw)                 // 4224 dwords
#define kLdsDw (kRingBase + (kRing + 6) * kRowDw)   // 4224 + 28 * 120 = 7584 dwords = 30336 bytes (HIP: 30336)

// TfShared<8, 352>: tex[352] (R2Texel, 48 bytes: ec0 ec1 ec2 ca cb cc | s0 s1 | s2 g | cbh cbo | m0 m1 m2), then
// ring[28][3][40] (FP32 / boxed FP16 horizontal results, or a wave's FP16 display scratch). Addressed in dwords.
groupshared uint lds[kLdsDw];

// ---- per-launch values ---------------------------------------------------------------------------------------------
static float g_e, g_e_prev, g_ratio;

// The driver's compiler regroups chains of multiplications / additions (it moved the constant and the uniform factor
// of (v * 0.01) * ratio together, and the per-lane factors of t.cc * ((kTapK2 * u) * v0)); post.hip's order is kept by
// passing intermediate results through opq: an integer xor with a zero the compiler cannot see (ow > 0), so the chain
// is cut there. Applied at every link where a regrouping could pay off for the compiler (shared subexpressions, a
// constant or uniform factor / term next to another one). Each opq costs one VALU per use; links where removing it left
// the ISA's operations unchanged (only the xor gone, RGA, all four forms) and outputs identical were removed (speed-post
// round 1). Kept where removal changed the ISA: c2u and inv255. Hl, cbo (one of cbh / cbo) and the output's v dec_scale
// are mul_nz instead (round 5: one fma in place of a multiply and an xor).
static uint g_z;
float opq(float x) { return asfloat(asuint(x) ^ g_z); }
// a * b as fma(a, b, -0) with the -0 opaque: exactly the product (signed zeros too), one VALU, and the driver can neither
// regroup it (not a multiplication) nor drop the addend
float mul_nz(float a, float b) { return mad(a, b, asfloat(g_z | 0x80000000u)); }
// a wave-uniform value in a scalar register (readfirstlane): its arithmetic can go to the scalar ALU
float unif(float x) { return asfloat(WaveReadLaneFirst(asuint(x))); }
#if defined(WG_DIRECT_TEXTURES)
#include "post_direct.hlsli"
#endif
// The direct form's kernel body is compiled once per pair of launch-uniform facts (post_body below): pd_rc = the dense
// render rectangle is the port's (the colour by (column, row), no index division), pd_pi1 = the pre-exposure is 1 (no
// division, no output scale). They are literals in each copy, so each copy holds only the code its launch can run.
#if POST_OUT == 3
#define PD_P , const bool pd_rc, const bool pd_pi1
#define PD_A , pd_rc, pd_pi1
#else
#define PD_P
#define PD_A
#endif
#include "post_f16.hlsli"
#if WG_WAVE == 64   // wave32: wg_platform.hlsli includes it
#include "wg_ags.hlsli"
#endif
#if POST_OUT == 3 && !defined(WG_NO_PT_SMALL)
#define WG_PT_SMALL 1   // post_posttr.hlsli: Dec's polynomial flag by one minimum
#endif
#if POST_OUT == 3 && !defined(WG_NO_PT_INF)
#define WG_PT_INF 1     // post_posttr.hlsli: blend's infinity flag by one maximum
#endif

// (_Float16) of three FP32 values, round to nearest even, by the driver's matrix element conversion (F32 accumulator ->
// F16 accumulator): one v_cvt_f16_f32 per value, the bits of HIP's casts. In wave64 an accumulator fragment holds 4
// values per lane (dwords 0 ... 3 in, two packed dwords out; the driver converts nothing past dword 3). Wave-uniform
// control flow only (wg_ags.hlsli). Returns (v0 | v1 << 16, v2).
uint2 cvt3_f16(float v0, float v1, float v2)
{
    wg_frag f;
    f.v[0] = asuint(v0); f.v[1] = asuint(v1); f.v[2] = asuint(v2);
    [unroll] for (uint i = 3; i < 8; ++i) f.v[i] = 0u;
    wg_frag r = wg_ags_convert(f, WG_AGS_T_F32_ACC, WG_AGS_T_F16_ACC);
    return uint2(r.v[0], r.v[1] & 0xffffu);
}

// ---- helpers of post.hip ---------------------------------------------------------------------------------------------
int clampi(int v, int lo, int hi) { int t = v > hi ? hi : v; return t < lo ? lo : t; }
int mini(int a, int b) { return a < b ? a : b; }
uint mul24(uint a, uint b) { return (a & 0xffffffu) * (b & 0xffffffu); }
int mul_i24(int a, int b) { return ((int)((uint)a << 8) >> 8) * ((int)((uint)b << 8) >> 8); }
int floor_i(float v) { return (int)floor(v); }
float clampf(float v, float lo, float hi) { return min(max(v, lo), hi); }
float pos_x(int x) { return opq(((float)x + 0.5f) * P_rx + P_jx); }
// pos_y: the product as mad(.., -0.0f) (== the product, bit for bit) so the driver keeps the wave-uniform value on SALU
// (s_mul_f32; a plain * goes to VALU + readfirstlane); the inner opq stops the driver fusing it with + P_jy (s_fmac)
float pos_y(int y) { return opq(opq(mad((float)y + 0.5f, P_ry, -0.0f)) + P_jy); }

// E(v) = kEncK log2(max(fma(100, v, 1), 1e-4))
float enc_arg(float v) { return max(mad(100.0f, v, 1.0f), k1em4); }
float enc_e(float v) { return kEncK * log2(enc_arg(v)); }
float enc_in(float v) { return v != v ? 0.0f : enc_e(min(v, 32752.0f)); }

#if defined(WG_PT_SMALL) || defined(WG_PT_INF)
#include "post_posttr.hlsli"
#endif

float dec_m1_poly(float qc)
{
    float y = qc * kDecLn;
    float t = mad(y, kP5040, kP720);
    t = mad(t, y, kP120);
    t = mad(t, y, kP24);
    t = mad(t, y, kP6);
    t = mad(t, y, 0.5f);
    t = mad(t, y, 1.0f);
    return (t * y);
}

float byte_f(uint v, uint i) { return (float)((v >> (8u * i)) & 0xffu); }

// fma(h, w, +0) with h an FP16 value (v_fma_mix_f32 of hw_lo / hw_hi). The +0 addend is opaque (g_z): with a literal 0
// the driver turns the fma into a multiply and fuses that with the following addition (one rounding less than HIP's
// v_fma_mix + v_add); the opaque zero keeps HIP's v_fma_mix_f32 h, w, 0 (round 5; was an opq xor after it).
float hw_lo(uint t, float w) { return mad(f16tof32(t), w, asfloat(g_z)); }
float hw_hi(uint t, float w) { return mad(f16tof32(t >> 16), w, asfloat(g_z)); }

// post.hip framebuffer_half: v clamped to +-65504 by compares (a NaN passes), then (_Float16) (v_cvt_f16_f32, RNE: the
// driver's conversion, cvt3_f16).
float fb_clamp(float v) { return select(v > 65504.0f, 65504.0f, select(v < -65504.0f, -65504.0f, v)); }
#if POST_OUT == 3
// The direct form's stored value. RGBA16F output with output_copy (nd_output_rne16): the HIP output-copy bytes,
// framebuffer_half (clamp, RNE16; NaN passes), since typed RGBA16F stores round toward zero on this driver; the typed
// store of the exact FP16 value then stores it unchanged. Call in wave-uniform control flow (cvt3_f16).
float3 pd_out16(float3 v)
{
    if (nd_output_rne16 != 0u) {
        const uint2 hb = cvt3_f16(fb_clamp(v.x), fb_clamp(v.y), fb_clamp(v.z));
        v = float3(f16tof32(hb.x & 0xffffu), f16tof32(hb.x >> 16), f16tof32(hb.y));
    }
    return v;
}
#endif

// ---- LDS texel (R2Texel) -------------------------------------------------------------------------------------------
struct TexF { float ec0, ec1, ec2, ca, cb, cc, s0, s1, s2, g, cbh, cbo; };

TexF tex_features(uint dw)   // r2_features of the texel at dword dw
{
    TexF f;
    f.ec0 = asfloat(lds[dw + 0]); f.ec1 = asfloat(lds[dw + 1]); f.ec2 = asfloat(lds[dw + 2]);
    f.ca = asfloat(lds[dw + 3]); f.cb = asfloat(lds[dw + 4]); f.cc = asfloat(lds[dw + 5]);
    uint h0 = lds[dw + 6], h1 = lds[dw + 7], h2 = lds[dw + 8];
    f.s0 = f16tof32(h0); f.s1 = f16tof32(h0 >> 16);
    f.s2 = f16tof32(h1); f.g = f16tof32(h1 >> 16);
    f.cbh = f16tof32(h2); f.cbo = f16tof32(h2 >> 16);
    return f;
}

// Raw inputs of one staged render pixel (TfRaw)
struct TfRaw { float4 c; uint d0, d1, c01, m0, m1, m2; };

TfRaw tf_fetch(int idx, int col, int row PD_P)
{
    uint ui = (uint)idx;
    TfRaw t;
#if defined(WG_DIRECT_TEXTURES)
    // the raw texel (post_direct.hlsli: by (column, row) when the dense rectangle is the port's, always on the direct
    // route); tf_put divides it by the pre-exposure (pd_normalize), so that the fetch does not wait for it
    t.c = pd_raw_color(ui, col, row, pd_rc);
#else
    t.c = wg_ld4f(color, ui * 16u);
#endif
    uint2 d = wg_ld64(A8, ui * 8u);
    t.d0 = d.x;
    t.d1 = d.y;
    t.c01 = wg_ld32(C4, ui * 4u);
#if defined(WG_POST_MO)
    t.m0 = 0u; t.m1 = 0u; t.m2 = 0u;   // the motion surfaces are not read (mo_core)
#else
    t.m0 = wg_ld32(m_near, ui * 4u);
    t.m1 = wg_ld32(m_center, ui * 4u);
    t.m2 = wg_ld32(m_far, ui * 4u);
#endif
    return t;
}

#if POST_OUT == 3
// The direct form's staging fetch of texel i = mini(lane, ilast) of footprint rows r0 ..., columns c0 ... (ncol per
// row): (row, column) = (st_r, st_col) on lane < ilast + 1 (the lane's own texel, fixed per lane), else those of texel
// ilast (wave-uniform); the same texel and clamps as post_body's other forms
TfRaw pd_stage_fetch(int lane, int ilast, int st_r, int st_col, float inv_ncol_u, int r0, int c0, int ncol PD_P)
{
    const int rl = wg_uniform((int)(((float)ilast + 0.5f) * inv_ncol_u));
    const int cl = wg_uniform(clampi(c0 + (ilast - mul_i24(rl, ncol)), 0, P_rw - 1));
    const bool own = lane <= ilast;
    const int rr = own ? st_r : rl, colc = own ? st_col : cl;
    const int rowc = clampi(r0 + rr, 0, P_rh - 1);
    return tf_fetch((int)mul24((uint)rowc, (uint)P_rw) + colc, colc, rowc PD_A);
}
#endif

#if WG_TF_TWO && POST_OUT != 3
// The buffer forms' second staged texel of the lane: i = mini(lane + kThreads, nstage - 1), the first texel's formula
TfRaw tf_stage2(int lane, int nstage, float inv_ncol, int r0, int c0, int ncol PD_P)
{
    const int i = mini(lane + kThreads, nstage - 1 < 0 ? 0 : nstage - 1);
    const int r = (int)(((float)i + 0.5f) * inv_ncol), c = i - mul_i24(r, ncol);
    const int rowc = clampi(r0 + r, 0, P_rh - 1), colc = clampi(c0 + c, 0, P_rw - 1);
    return tf_fetch((int)mul24((uint)rowc, (uint)P_rw) + colc, colc, rowc PD_A);
}
#endif

// tf_put: load_texel12 + texel_f + r2_store_features into texel i (the codes are integers 0 ... 255: exact in FP16)
void tf_put(uint i, TfRaw r, float e PD_P)
{
    uint dw = i * kTexDw;
    uint a = r.d0 & 0xffffffu;
    uint b = (r.d0 >> 24) | ((r.d1 & 0xffffu) << 8);
    uint ac = a ^ 0xffff00u;
#if defined(WG_DIRECT_TEXTURES)
    // native_direct_normalized_color's division (tf_fetch holds the raw texel): native_direct_color_div3, the per-frame
    // reciprocal form (3 operations per channel, its own exact fallback lanes), as pd_normalize
    float4 c = r.c;
    if (pd_div_on(pd_rc, pd_pi1)) c.xyz = pd_xor(pd_div3(r.c.xyz));
#else
    const float4 c = r.c;
#endif
    lds[dw + 0] = asuint(enc_in(e * c.x));
    lds[dw + 1] = asuint(enc_in(e * c.y));
    lds[dw + 2] = asuint(enc_in(e * c.z));
    lds[dw + 3] = asuint(f16tof32(r.c01));
    lds[dw + 4] = asuint(f16tof32(r.c01 >> 16));
    lds[dw + 5] = asuint(f16tof32(r.d1 >> 16));
    lds[dw + 6] = f32tof16(byte_f(b, 0u)) | (f32tof16(byte_f(b, 1u)) << 16);
    lds[dw + 7] = f32tof16(byte_f(b, 2u)) | (f32tof16(byte_f(ac, 0u)) << 16);
    lds[dw + 8] = f32tof16(byte_f(ac, 1u)) | (f32tof16(byte_f(ac, 2u)) << 16);
#if !defined(WG_POST_MO)
    lds[dw + 9] = r.m0;
    lds[dw + 10] = r.m1;
    lds[dw + 11] = r.m2;
#endif
}

// ---- gather (spec 06 step 2) ---------------------------------------------------------------------------------------
struct Gather { float W; float c[3]; float s[3]; float a[3]; };

Gather gather_zero()
{
    Gather g;
    g.W = 0.0f;
    [unroll] for (int i = 0; i < 3; ++i) { g.c[i] = 0.0f; g.s[i] = 0.0f; g.a[i] = 0.0f; }
    return g;
}

Gather gather_zero_opq()   // gather_zero with the colour sums at an opaque +0 (g_z)
{
    Gather g = gather_zero();
    [unroll] for (int i = 0; i < 3; ++i) g.c[i] = asfloat(g_z);
    return g;
}

void gather_add_f(inout Gather g, TexF t, float uu, float vv, float uv)
{
    float q = mad(t.ca, uu, mad(t.cb, vv, t.cc * uv));
    float w = exp2(q);
    g.W = g.W + w;
    g.c[0] = mad(w, t.ec0, g.c[0]);
    g.c[1] = mad(w, t.ec1, g.c[1]);
    g.c[2] = mad(w, t.ec2, g.c[2]);
    g.s[0] = mad(w, t.s0, g.s[0]);
    g.a[0] = mad(w, t.g, g.a[0]);
    g.s[1] = mad(w, t.s1, g.s[1]);
    g.a[1] = mad(w, t.cbh, g.a[1]);
    g.s[2] = mad(w, t.s2, g.s[2]);
    g.a[2] = mad(w, t.cbo, g.a[2]);
}

// ---- history: 12-tap Catmull-Rom of the previous full history (history_cr_span<false>, 8-byte RGBA16F texels) ----------
// Catmull-Rom weights of taps -1 ... 2 at fraction f (cr_weights)
float4 cr_weights(float f)
{
    float4 w;
    w.x = f * (-0.5f + f * (1.0f - 0.5f * f));
    w.y = 1.0f + f * f * (-2.5f + 1.5f * f);
    w.z = f * (0.5f + f * (2.0f - 1.5f * f));
    w.w = f * f * (0.5f * f - 0.5f);
    return w;
}
float3 h3(uint2 v) { return float3(f16tof32(v.x), f16tof32(v.x >> 16), f16tof32(v.y)); }

// The weighted sum of one sample from its twelve taps (rows 0 and 3: columns 1, 2; rows 1 and 2: columns 0 ... 3):
// r = fma(wx[i], t, r) from +0, acc = fma(wy[j], r, acc) from +0, divided by the weight sum as HIP.
float3 cr_sum(uint2 t01, uint2 t02, uint2 t10, uint2 t11, uint2 t12, uint2 t13, uint2 t20, uint2 t21, uint2 t22, uint2 t23,
              uint2 t31, uint2 t32, float4 wx, float4 wy)
{
    const float3 z = float3(0.0f, 0.0f, 0.0f);
    float3 r0 = mad(wx.z, h3(t02), mad(wx.y, h3(t01), z));
    float3 r1 = mad(wx.w, h3(t13), mad(wx.z, h3(t12), mad(wx.y, h3(t11), mad(wx.x, h3(t10), z))));
    float3 r2 = mad(wx.w, h3(t23), mad(wx.z, h3(t22), mad(wx.y, h3(t21), mad(wx.x, h3(t20), z))));
    float3 r3 = mad(wx.z, h3(t32), mad(wx.y, h3(t31), z));
    float3 acc = mad(wy.w, r3, mad(wy.z, r2, mad(wy.y, r1, mad(wy.x, r0, z))));
    // sum of the twelve tap weights; ((wx0 + wx1) + wx2) + wx3 kept apart from wx1 + wx2 (opq: no regrouping)
    float sx = ((wx.x + wx.y) + wx.z) + wx.w;
    float norm = (wx.y + wx.z) * (wy.x + wy.w) + sx * (wy.y + wy.z);
    float inv = rcp(norm);
    return acc * inv;
}

// history_cr_span<false>: HIP loads each row as one span (rows 1, 2: 4 texels from column cs = clamp(ix - 1, 0, dw - 4),
// rows 0, 3: 2 texels from cs + 1; 16-byte loads at 8-byte aligned offsets) and, at the left / right image edge
// (ix - 1 != cs), reloads the clamped taps one by one; the sum only sees the twelve texels at the clamped (column, row)
// of each tap. Here the texels are fetched in one of two ways, decided per wave (every active lane: the caller runs
// this for the lanes with a valid sample only):
// - interior wave (columns ix - 1 ... ix + 2 and rows iy - 1 ... iy + 2 of every lane inside the image): the clamps are
//   the identity, so HIP's six spans with row k at row 0 + k pitch, and no reload;
// - otherwise (a lane at an image edge): the twelve taps at their clamped (column, row) as 8-byte loads, one memory
//   round trip instead of the spans plus the edge lanes' reload (edge strips were about 10 % slower than the others,
//   and on a fixed slot their delays added up into the launch's tail).
// The same texels either way, so the same sums.
float3 history_cr_span(float qx, float qy)
{
    const float tx = qx - 0.5f, ty = qy - 0.5f;
    const float fx0 = floor(tx), fy0 = floor(ty);
    const int ix = (int)fx0, iy = (int)fy0;
    const uint pitch = (uint)P_dw * 8u;
    [branch] if (WaveActiveAllTrue(and((uint)(ix - 1) <= (uint)(P_dw - 4), (uint)(iy - 1) <= (uint)(P_dh - 4)))) {
        const uint b0 = hist_prev + (uint)(ix - 1) * 8u + mul24((uint)(iy - 1), pitch);
        const uint b1 = b0 + pitch, b2 = b0 + 2u * pitch, b3 = b0 + 3u * pitch;
        const uint4 a0 = wg_mem.Load4(b0 + 8u), a10 = wg_mem.Load4(b1), a11 = wg_mem.Load4(b1 + 16u);
        const uint4 a20 = wg_mem.Load4(b2), a21 = wg_mem.Load4(b2 + 16u), a3 = wg_mem.Load4(b3 + 8u);
        return cr_sum(a0.xy, a0.zw, a10.xy, a10.zw, a11.xy, a11.zw, a20.xy, a20.zw, a21.xy, a21.zw, a3.xy, a3.zw,
                      cr_weights(tx - fx0), cr_weights(ty - fy0));
    }
    const uint ro0 = mul24((uint)clampi(iy - 1, 0, P_dh - 1), pitch), ro1 = mul24((uint)clampi(iy, 0, P_dh - 1), pitch);
    const uint ro2 = mul24((uint)clampi(iy + 1, 0, P_dh - 1), pitch), ro3 = mul24((uint)clampi(iy + 2, 0, P_dh - 1), pitch);
    const uint c0 = hist_prev + (uint)clampi(ix - 1, 0, P_dw - 1) * 8u, c1 = hist_prev + (uint)clampi(ix, 0, P_dw - 1) * 8u;
    const uint c2 = hist_prev + (uint)clampi(ix + 1, 0, P_dw - 1) * 8u, c3 = hist_prev + (uint)clampi(ix + 2, 0, P_dw - 1) * 8u;
    return cr_sum(wg_mem.Load2(ro0 + c1), wg_mem.Load2(ro0 + c2), wg_mem.Load2(ro1 + c0), wg_mem.Load2(ro1 + c1),
                  wg_mem.Load2(ro1 + c2), wg_mem.Load2(ro1 + c3), wg_mem.Load2(ro2 + c0), wg_mem.Load2(ro2 + c1),
                  wg_mem.Load2(ro2 + c2), wg_mem.Load2(ro2 + c3), wg_mem.Load2(ro3 + c1), wg_mem.Load2(ro3 + c2),
                  cr_weights(tx - fx0), cr_weights(ty - fy0));
}


// post.hip's fallback colour: the encoded current colour of render pixel (fx, fy) at the previous exposure
float3 fallback_rgb(int fx, int fy PD_P)
{
#if defined(WG_DIRECT_TEXTURES)
    // native_direct_normalized_color of the dense texel (one load and one division site for both routes)
    const float4 c = pd_normalize(pd_raw_color(mul24((uint)fy, (uint)P_rw) + (uint)fx, fx, fy, pd_rc), pd_rc, pd_pi1);
#else
    float4 c = wg_ld4f(color, mul24((uint)fy, (uint)P_rw * 16u) + (uint)fx * 16u);
#endif
    return float3(enc_e(opq(g_e_prev * c.x)), enc_e(opq(g_e_prev * c.y)), enc_e(opq(g_e_prev * c.z)));
}

// ---- spec 06 steps 3 to 5 for one D pixel (post_core_s<false, true, false, kSel, kTrim> + post_blend<kTrim>) -----------
// and() / or() instead of && / ||: HLSL 2021 short-circuits scalar && / ||, and the driver then carries the flags as
// 0 / -1 integers through v_cndmask chains; and() / or() keep them in lane masks (the same values, about -40 VALU).
void post_core_s(Gather g, int x, int y, int fx, int fy, float mx[3], float my[3], float s, out float hh[3],
                 out float oh[3] PD_P)
{
    // post_select_m
    float invW = rcp(g.W);
    float inv255 = (invW * kInv255);
    float cbar[3], sig[3];
    [unroll] for (int i = 0; i < 3; ++i) {
        cbar[i] = g.c[i] * invW;
        sig[i] = (g.s[i] * inv255);
    }
    float cbh = (g.a[1] * inv255);
    float cbo = mul_nz(g.a[2], inv255);
    bool b1 = sig[1] > sig[0];
    float best = b1 ? sig[1] : sig[0];
    bool b2 = sig[2] > best;
    float s1 = b2 ? sig[2] : best;
    float m1x = b2 ? mx[2] : (b1 ? mx[1] : mx[0]);
    float m1y = b2 ? my[2] : (b1 ? my[1] : my[0]);
    bool f0 = !or(b1, b2);
    float so1 = f0 ? sig[1] : sig[0], so2 = b2 ? sig[1] : sig[2];
    float mo1x = f0 ? mx[1] : mx[0], mo1y = f0 ? my[1] : my[0];
    float mo2x = b2 ? mx[1] : mx[2], mo2y = b2 ? my[1] : my[2];
    bool cc = so2 > so1;
    float s2 = cc ? so2 : so1;
    float m2x = cc ? mo2x : mo1x;
    float m2y = cc ? mo2y : mo1y;
    float q1 = clampf(k1p1 * s1 - k0p05, 0.0f, 1.0f);
    float q2 = clampf(k1p1 * s2 - k0p05, 0.0f, 1.0f);
    float rden = rcp(max(q1 + q2, k1em10));
    float w2 = q2 * rden, w1;
    if (w2 < kInv255) {
        w2 = 0.0f;
        w1 = 1.0f;
    } else {
        w1 = q1 * rden;
    }

    float X = (float)x + 0.5f, Y = (float)y + 0.5f;
    bool two = w2 != 0.0f;
    float q1x = (X + m1x), q1y = (Y + m1y), q2x = (X + m2x), q2y = (Y + m2y);
    uint bw = asuint((float)P_dw), bh = asuint((float)P_dh);
    bool nreset = P_reset == 0;
    bool v1 = and(and(nreset, asuint(q1x) <= bw), asuint(q1y) <= bh);
    bool v2 = and(and(nreset, asuint(q2x) <= bw), asuint(q2y) <= bh);
    bool same = and(and(and(v1, v2), q1x == q2x), q1y == q2y);
    // the history sample of each candidate, or the current colour where the candidate is invalid (post.hip's fb; made
    // only where it is used, so that its three values are not live through the history loads)
    float H[3];
    // (if / else, not ?: on vectors: the history sample runs for the valid lanes only, so that its per-wave interior
    // test sees only them)
    float3 h;
    if (v1) h = history_cr_span(q1x, q1y);
    else h = fallback_rgb(fx, fy PD_A);
    [unroll] for (int i = 0; i < 3; ++i) H[i] = w1 * h[i];
    if (two) {
        if (!same) {
            if (v2) h = history_cr_span(q2x, q2y);
            else h = fallback_rgb(fx, fy PD_A);
        }
        [unroll] for (int i = 0; i < 3; ++i) H[i] = H[i] + w2 * h[i];
    }

    // post_blend<kTrim = true>
    float Cl[3], Hl[3];
    bool hl_inf = false;
    float qc[6], v[6];
    // Dec: 2^(kDecK q) - 1, the polynomial below kDecPolyMax. As post.hip: the exponential for every value, the
    // polynomial only when some lane of the wave needs it (a wave-uniform branch; the same values per lane).
    bool small = false;
    [unroll] for (int i = 0; i < 6; ++i) {
        qc[i] = i < 3 ? clampf(cbar[i], k1em4, 3.0f) : wg_med3f(H[i - 3], k1em4, 3.0f);
        v[i] = exp2(qc[i] * kDecK) - 1.0f;
#if !defined(WG_PT_SMALL)
        small = or(small, qc[i] < kDecPolyMax);
#endif
    }
#if defined(WG_PT_SMALL)
    small = pt_any_small6(qc);
#endif
    if (WaveActiveAnyTrue(small)) {
        [unroll] for (int i = 0; i < 6; ++i) v[i] = qc[i] < kDecPolyMax ? dec_m1_poly(qc[i]) : v[i];
    }
    [unroll] for (int i = 0; i < 3; ++i) {
        Cl[i] = (v[i] * k0p01);
        Hl[i] = mul_nz(v[3 + i], k0p01) * g_ratio;
#if !defined(WG_PT_INF)
        hl_inf = or(hl_inf, abs(Hl[i]) == asfloat(0x7f800000u));
#endif
    }
#if defined(WG_PT_INF)
    hl_inf = pt_any_inf3(Hl[0], Hl[1], Hl[2]);
#endif
    float alpha = g.W < asfloat(0x7f800000u) ? clampf(s * (g.a[0] * kInv255), 0.0f, 1.0f) : 0.0f;
    float ombh = (1.0f - alpha) * cbh, ombo = (1.0f - alpha) * cbo;
    float bh2 = 1.0f - ombh, bo2 = 1.0f - ombo;
    [unroll] for (int i = 0; i < 3; ++i) {
        hh[i] = mad(Hl[i], ombh, Cl[i] * bh2);
        oh[i] = mad(Hl[i], ombo, Cl[i] * bo2);
    }
    float sh = hh[0] + hh[1] + hh[2], so = oh[0] + oh[1] + oh[2];
    if (or(sh != sh, hl_inf)) {
        [unroll] for (int i = 0; i < 3; ++i) hh[i] = Cl[i];
    }
    if (or(so != so, hl_inf)) {
        [unroll] for (int i = 0; i < 3; ++i) oh[i] = Cl[i];
    }
}

#if defined(WG_POST_MO)
// ---- motion at output resolution (post_motion_outres.h) ---------------------------------------------------------------
// mo_center: RN32((i + 1/2) / n) from rn = RN32(1 / n) with Markstein's correction (exact; opq: the driver must not
// regroup -n (a rn) into (-n rn) a)
float mo_center(int i, int n, float rn)
{
    const float a = (float)i + 0.5f, q0 = opq(a * rn);
    return mad(mad(-(float)n, q0, a), rn, q0);
}
// mo_texel: clamp(trunc(RN32(RN32((i + 1/2) / n) m)), 0, m - 1)
int mo_texel(int i, int n, float rn, int m)
{
    const float t = mo_center(i, n, rn) * (float)m;
    const int k = (int)t;
    return k > m - 1 ? m - 1 : (k < 0 ? 0 : k);
}
// mo_fetch: the raw motion of D pixels (x, y0) and (x, y0 + 1) (a row past the image reads the last texel row)
void mo_fetch(int x, int y0, uint2 dims, out float2 mq0, out float2 mq1)
{
    const uint tx = (uint)mo_texel(x, P_dw, mo_idw, mo_mw);
    const uint ty0 = (uint)mo_texel(y0, P_dh, mo_idh, mo_mh), ty1 = (uint)mo_texel(y0 + 1, P_dh, mo_idh, mo_mh);
#if defined(WG_DIRECT_TEXTURES)
    mq0 = wg_mo_texel_direct(tx, ty0, dims);
    mq1 = wg_mo_texel_direct(tx, ty1, dims);
#else
    const uint col = mo_wide != 0 ? tx * 8u : tx * 4u, pitch = (uint)mo_pitch & 0xffffffu;
    mq0 = wg_mo_value(wg_mo_load(motion_p, (ty0 & 0xffffffu) * pitch + col), mo_wide);
    mq1 = wg_mo_value(wg_mo_load(motion_p, (ty1 & 0xffffffu) * pitch + col), mo_wide);
#endif
}

// mo_core<kTrim>: spec 06 steps 3 to 5 of one D pixel with the single motion mv (raw texel): post_select's weights,
// one history sample (or the fallback), H = fma(w1, h, w2 h), post_blend (as post_core_s's, the NaN-position rule)
void mo_core(Gather g, float2 mv, int x, int y, int fx, int fy, float s, out float hh[3], out float oh[3] PD_P)
{
    // mo_position
    const float qx = mad(mo_sx, mo_bx + mv.x, (float)x + 0.5f);
    const float qy = mad(mo_sy, mo_by + mv.y, (float)y + 0.5f);
    // post_select (the selection of post_core_s without the motions: the same sig[first], sig[second], w1, w2)
    float invW = rcp(g.W);
    float inv255 = (invW * kInv255);
    float cbar[3], sig[3];
    [unroll] for (int i = 0; i < 3; ++i) {
        cbar[i] = g.c[i] * invW;
        sig[i] = (g.s[i] * inv255);
    }
    float cbh = (g.a[1] * inv255);
    float cbo = mul_nz(g.a[2], inv255);
    bool b1 = sig[1] > sig[0];
    float best = b1 ? sig[1] : sig[0];
    bool b2 = sig[2] > best;
    float s1 = b2 ? sig[2] : best;
    bool f0 = !or(b1, b2);
    float so1 = f0 ? sig[1] : sig[0], so2 = b2 ? sig[1] : sig[2];
    float s2 = so2 > so1 ? so2 : so1;
    float q1 = clampf(k1p1 * s1 - k0p05, 0.0f, 1.0f);
    float q2 = clampf(k1p1 * s2 - k0p05, 0.0f, 1.0f);
    float rden = rcp(max(q1 + q2, k1em10));
    float w2 = q2 * rden, w1;
    if (w2 < kInv255) {
        w2 = 0.0f;
        w1 = 1.0f;
    } else {
        w1 = q1 * rden;
    }
    const bool live = P_reset == 0;
    const bool nanq = and(live, or(qx != qx, qy != qy));
    const bool valid = and(and(and(and(live, qx >= 0.0f), qx <= (float)P_dw), qy >= 0.0f), qy <= (float)P_dh);
    float3 h;
    if (valid) h = history_cr_span(qx, qy);
    else h = fallback_rgb(fx, fy PD_A);
    float H[3];
    [unroll] for (int i2 = 0; i2 < 3; ++i2) H[i2] = mad(w1, h[i2], w2 != 0.0f ? w2 * h[i2] : 0.0f);
    if (nanq) {   // 1 - beta' = 0: the blend gives C_lin for both
        cbh = 0.0f;
        cbo = 0.0f;
    }

    // post_blend<kTrim = true> (as post_core_s)
    float Cl[3], Hl[3];
    bool hl_inf = false;
    float qc[6], v[6];
    bool small = false;
    [unroll] for (int i3 = 0; i3 < 6; ++i3) {
        qc[i3] = i3 < 3 ? clampf(cbar[i3], k1em4, 3.0f) : wg_med3f(H[i3 - 3], k1em4, 3.0f);
        v[i3] = exp2(qc[i3] * kDecK) - 1.0f;
#if !defined(WG_PT_SMALL)
        small = or(small, qc[i3] < kDecPolyMax);
#endif
    }
#if defined(WG_PT_SMALL)
    small = pt_any_small6(qc);
#endif
    if (WaveActiveAnyTrue(small)) {
        [unroll] for (int i4 = 0; i4 < 6; ++i4) v[i4] = qc[i4] < kDecPolyMax ? dec_m1_poly(qc[i4]) : v[i4];
    }
    [unroll] for (int i5 = 0; i5 < 3; ++i5) {
        Cl[i5] = (v[i5] * k0p01);
        Hl[i5] = mul_nz(v[3 + i5], k0p01) * g_ratio;
#if !defined(WG_PT_INF)
        hl_inf = or(hl_inf, abs(Hl[i5]) == asfloat(0x7f800000u));
#endif
    }
#if defined(WG_PT_INF)
    hl_inf = pt_any_inf3(Hl[0], Hl[1], Hl[2]);
#endif
    float alpha = g.W < asfloat(0x7f800000u) ? clampf(s * (g.a[0] * kInv255), 0.0f, 1.0f) : 0.0f;
    float ombh = (1.0f - alpha) * cbh, ombo = (1.0f - alpha) * cbo;
    float bh2 = 1.0f - ombh, bo2 = 1.0f - ombo;
    [unroll] for (int i6 = 0; i6 < 3; ++i6) {
        hh[i6] = mad(Hl[i6], ombh, Cl[i6] * bh2);
        oh[i6] = mad(Hl[i6], ombo, Cl[i6] * bo2);
    }
    float sh = hh[0] + hh[1] + hh[2], so = oh[0] + oh[1] + oh[2];
    if (or(sh != sh, hl_inf)) {
        [unroll] for (int i7 = 0; i7 < 3; ++i7) hh[i7] = Cl[i7];
    }
    if (or(so != so, hl_inf)) {
        [unroll] for (int i8 = 0; i8 < 3; ++i8) oh[i8] = Cl[i8];
    }
}
#endif

#if WG_LZ_ANALYTIC
// ---- output filter, profile 0 ("analytic", post_w64.co): FP32 constant weights of taps -2 ... +2 for an even output
// index (post.hip kLz0 ... kLz4; mirrored for an odd one), v = w0 t0, one fma per further tap, clamp to the taps' range
static const float kLz0 = asfloat(0xbdba3b4bu), kLz1 = asfloat(0x3de71ca9u), kLz2 = asfloat(0x3f23b70fu);
static const float kLz3 = asfloat(0x3ed182b5u), kLz4 = asfloat(0xbd90a4acu);
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
// lz5<kEven> / lz5<false> on the FP32 taps t[o ... o + 4], as FP32 bits
uint lz5a_even(float t[10], int o) { return asuint(lz5a(t[o], t[o + 1], t[o + 2], t[o + 3], t[o + 4], kLz0, kLz1, kLz3, kLz4)); }
uint lz5a_odd(float t[10], int o) { return asuint(lz5a(t[o], t[o + 1], t[o + 2], t[o + 3], t[o + 4], kLz4, kLz3, kLz1, kLz0)); }
#else
// ---- output filter, profile 2 ("original"): FP16 weights, one FP16 fma per tap, clamp to the taps' range -------------
// Values are "boxes": FP16 bits in the low half of a dword.
half hbox(uint v) { return asfloat16((uint16_t)(v & 0xffffu)); }
uint hunbox(half v) { return (uint)asuint16(v); }

uint lz5_original(uint t0b, uint t1b, uint t2b, uint t3b, uint t4b, half w0, half w1, half w2, half w3, half w4)
{
    half t0 = hbox(t0b), t1 = hbox(t1b), t2 = hbox(t2b), t3 = hbox(t3b), t4 = hbox(t4b);
    half a = mad(w0, t0, (half)0.0);
    a = mad(w1, t1, a);
    a = mad(w2, t2, a);
    a = mad(w3, t3, a);
    a = mad(w4, t4, a);
    half mid_lo = min(min(t1, t2), t3), mid_hi = max(max(t1, t2), t3);
    half lo = min(min(mid_lo, t0), t4), hi = max(max(mid_hi, t0), t4);
    return hunbox(min(max(a, lo), hi));   // med3 (the taps are finite, lo <= hi)
}

// lz5<kEven> on the boxes t[o ... o + 4]
uint lz5_even(uint t[10], int o)
{
    return lz5_original(t[o], t[o + 1], t[o + 2], t[o + 3], t[o + 4], hbox(lz_w0), hbox(lz_w1), hbox(lz_w2), hbox(lz_w3),
                        hbox(lz_w4));
}
uint lz5_odd(uint t[10], int o)
{
    return lz5_original(t[o], t[o + 1], t[o + 2], t[o + 3], t[o + 4], hbox(lz_w4), hbox(lz_w3), hbox(lz_w2), hbox(lz_w1),
                        hbox(lz_w0));
}
#endif

// The reduced history's 10-bit code of one channel from its 2 x 2 sum (the chain of post.hip's reduced store)
uint rh_code(float sum)
{
    const float m = sum * 0.25f;
    const float vq = clampf((enc_e(m)) * kInv245, 0.0f, 1.0f);
    return (uint)round(vq * 1023.0f);
}

// The 64 lanes of a strip row exchange values through LDS (tf_row_sync): in wave64 they are one wave (its LDS operations
// execute in order; the fence keeps the compiler from moving them across), in wave32 two waves (a workgroup barrier).
// The kernel runs in wave64 as HIP's post_w64.co (a wave32 build measured slower).
void tf_row_sync()
{
#if WG_WAVE == 64
    GroupMemoryBarrier();
#else
    GroupMemoryBarrierWithGroupSync();
#endif
}

#if POST_OUT == 3
// One vertical item of the direct form (post_body's step 3): output pixel (xg, y), ring column xo, its first ring row
// urel = 3 pr + 2 dy past the pass's first row pair, and the lane phase's filter weights wa = (w0, w4), wb = (w1, w3) as
// FP16 bit pairs. post_body calls it with per-lane values fixed for the walk on the first pass (it = lane).
void pd_vitem(int y, int xg, int xo, int urel, uint wa, uint wb, int p_next, bool interior, int s0, int r_first, bool lut,
              float dec_scale PD_P)
{
    if (or(y >= oh, xg >= ow)) return;
    uint t[5][3];
    if (interior) {   // one ring address, the 15 taps at constant offsets (mirrored rows: no wrap)
        int u = s0 + urel;
        u = u >= kRing ? u - kRing : u;
        u = u >= kRing ? u - kRing : u;
        const uint rp = kRingBase + mul24((uint)u, (uint)kRowDw) + (uint)xo;
        [unroll] for (int k = 0; k < 5; ++k)
            [unroll] for (int c = 0; c < 3; ++c) t[k][c] = lds[rp + (uint)(k * kRowDw + c * kTfOutX)];
    } else {
        const int xr = 3 * p_next - 2 + urel;
        const int row0 = clampi(xr, 0, P_dh - 1);
        const int sl0 = (int)((uint)(row0 - r_first) % (uint)kRing);
        [unroll] for (int k = 0; k < 5; ++k) {
            const int st = clampi(xr + k, 0, P_dh - 1) - row0;
            const int sl = sl0 + st >= kRing ? sl0 + st - kRing : sl0 + st;
            [unroll] for (int c = 0; c < 3; ++c)
                t[k][c] = lds[kRingBase + (uint)sl * kRowDw + (uint)(c * kTfOutX + xo)];
        }
    }
#if WG_LZ_ANALYTIC
    // the analytic filter: wa = 1 for the even phase (pd_vsplit); FP32 results, so no FP16-indexed output table
    const bool ev = wa != 0u;
    const float w0 = ev ? kLz0 : kLz4, w4 = ev ? kLz4 : kLz0, w1 = ev ? kLz1 : kLz3, w3 = ev ? kLz3 : kLz1;
    float vo[3], qv[3];
    bool vsmall = false;
    [unroll] for (int c = 0; c < 3; ++c) {
        qv[c] = clampf(lz5a(asfloat(t[0][c]), asfloat(t[1][c]), asfloat(t[2][c]), asfloat(t[3][c]), asfloat(t[4][c]),
                            w0, w1, w3, w4), k1em4, 3.0f);
        vo[c] = exp2(qv[c] * kDecK) - 1.0f;
        vsmall = or(vsmall, qv[c] < kDecPolyMax);
    }
#else
    const half w0 = hbox(wa), w4 = hbox(wa >> 16), w1 = hbox(wb), w3 = hbox(wb >> 16);
    if (lut) {   // the output values T[r] (wg_f16lut.hlsli, lev-f16lut d7149e2): the chain below, looked up
        float3 res;
        [unroll] for (int c = 0; c < 3; ++c) {
            const uint r = lz5_original(t[0][c], t[1][c], t[2][c], t[3][c], t[4][c], w0, w1, hbox(lz_w2), w3, w4);
            res[c] = asfloat(wg_ld32(color, WG_F16LUT_DEC + (r << 2)));
        }
        native_direct_output[uint2((uint)xg, (uint)y)] = float4(res, 1.0f);
        return;
    }
    float vo[3], qv[3];
    bool vsmall = false;
    [unroll] for (int c = 0; c < 3; ++c) {
        const uint r = lz5_original(t[0][c], t[1][c], t[2][c], t[3][c], t[4][c], w0, w1, hbox(lz_w2), w3, w4);
        qv[c] = clampf(f16tof32(r), k1em4, 3.0f);
        vo[c] = exp2(qv[c] * kDecK) - 1.0f;
        vsmall = or(vsmall, qv[c] < kDecPolyMax);
    }
#endif
    if (WaveActiveAnyTrue(vsmall)) {   // Dec's polynomial only when some lane needs it (post.hip's branches)
        [unroll] for (int c = 0; c < 3; ++c) vo[c] = qv[c] < kDecPolyMax ? dec_m1_poly(qv[c]) : vo[c];
    }
    // Match the staged FP32 output, optional scale_rgb, then Converter's typed output. opq preserves the
    // first pass's rounding boundary: the driver must not regroup vo * (dec_scale * nd_color_pi).
    const float3 result = float3(mul_nz(vo[0], dec_scale), mul_nz(vo[1], dec_scale), mul_nz(vo[2], dec_scale));
    const bool scale = pd_div_on(pd_rc, pd_pi1);
    float3 so = result;
    if (scale) so = float3(result.x * nd_color_pi, result.y * nd_color_pi, result.z * nd_color_pi);
    native_direct_output[uint2((uint)xg, (uint)y)] = float4(pd_out16(so), 1.0f);
    // native_direct_scale x 3: the plain products stored above are its values except where a product is zero; those
    // lanes store native_direct_scale's values over them (a lane's stores to one texel stay in order)
    if (scale) {
        const bool zr = pd_scale_check(so);
        [branch] if (WaveActiveAnyTrue(zr)) {
            const float3 ex = pd_out16(pd_scale_exact(result, nd_color_pi));
            if (zr) native_direct_output[uint2((uint)xg, (uint)y)] = float4(ex, 1.0f);
        }
    }
}

// item it's (y - 2 p_next, xo, urel, wa, wb) (post_body's decomposition of an item index)
void pd_vsplit(int it, out int yrel, out int xo, out int urel, out uint wa, out uint wb)
{
    const int pr = (int)(mul24((uint)it, 52429u) >> 22);
    const int rem = it - (int)mul24((uint)pr, 2u * kTfOutX);
    const int dy = rem >= kTfOutX ? 1 : 0;
    xo = rem - dy * kTfOutX;
    yrel = 2 * pr + dy;
    urel = 3 * pr + 2 * dy;
    const bool ev = dy == 0;
#if WG_LZ_ANALYTIC
    wa = ev ? 1u : 0u;   // the phase only (pd_vitem selects the constant weights)
    wb = 0u;
#else
    wa = ((ev ? lz_w0 : lz_w4) & 0xffffu) | ((ev ? lz_w4 : lz_w0) << 16);
    wb = ((ev ? lz_w1 : lz_w3) & 0xffffu) | ((ev ? lz_w3 : lz_w1) << 16);
#endif
}
#endif

// ---- the kernel ------------------------------------------------------------------------------------------------------
#if POST_OUT == 3
void post_body(const bool pd_rc, const bool pd_pi1)   // the direct form's body (its kernel at the end of the file)
{
#elif POST_OUT == 0
WG_KERNEL(post_tf_f32_p12_diet_vsit)
{
    WG_SETUP();
#else
WG_KERNEL(post_tf_p12_diet_vsit)
{
    WG_SETUP();
#endif
    if (P_dw < 4) return;
    g_z = (uint)ow >> 31;   // 0 (ow > 0), unknown to the compiler
    g_e = P_e;
    g_e_prev = P_e_prev;
    g_ratio = P_ratio;
#if POST_OUT != 3
    float dec_scale = dec_scale_arg;
    if (expo != 0u) {   // kDiet: ratio and dec_scale precomputed in the exposure pair
        float e0 = unif(wg_ldf(expo, 0u)), e1 = unif(wg_ldf(expo, 4u));
        g_e = e0 == 0.0f ? 1.0f : e0;
        g_e_prev = e1 == 0.0f ? 1.0f : e1;
        g_ratio = unif(wg_ldf(expo, 12u));
        dec_scale = unif(wg_ldf(expo, 16u));
    }
#endif
    const int lane = (int)wg_lane_id();
    const int wave = wg_uniform(lane >> 6), wl = lane & 63;   // wave index: uniform (tf_uniform: a scalar register)
    const int strips = (ow + kTfOutX - 1) / kTfOutX;
    const int bid = (int)wg_block_id();
    // workgroup bid -> (strip, chunk): HIP takes strip = bid % strips; here the strips of chunk c are rotated by c (the
    // same set of workgroups, so the same values). With 64 strips and 128 resident workgroups, HIP's order gives a
    // resident slot the same strip in every round, so the two slower edge strips' delays add up on four slots and
    // their last workgroups start late (the launch's tail); rotated, they fall on different slots.
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

    const int c0 = floor_i(pos_x(clampi(X0 - 2, 0, P_dw - 1)) - 0.5f) - 1;
    const int c1 = floor_i(pos_x(clampi(X0 + kTfCols - 3, 0, P_dw - 1)) - 0.5f) + 1;
    const int ncol = c1 - c0 + 1;
    const float inv_ncol = wg_div(1.0f, (float)ncol);
    const uint rowb = (uint)(48 * ncol);

    // staging of step y_step: footprint rows r0 ... (step_rows), texels loaded one step ahead (WG_TF_PIPE = 2)
    int r0, nstage;
    {
        int ys = r_first;
        int yl = mini(ys + 2 * (kNW - 1), r_need) & ~1;
        r0 = floor_i(pos_y(ys) - 0.5f) - 1;
        int nrow = floor_i(pos_y(yl) - 0.5f) + 2 - r0 + 1;
        nstage = ncol * nrow < kTfCap ? ncol * nrow : kTfCap;
    }
    TfRaw raw;
#if POST_OUT == 3
    // Direct form: texel i = mini(lane, nstage - 1) as below, but its footprint row and clamped column are fixed per lane
    // for lane < nstage (opaque: computed once) and wave-uniform for the other lanes (texel nstage - 1: scalar ALU)
    const float inv_ncol_u = unif(inv_ncol);
    const int st_r = (int)(((float)lane + 0.5f) * inv_ncol);
    const int st_rx = st_r ^ (int)g_z, st_col = clampi(c0 + (lane - mul_i24(st_r, ncol)), 0, P_rw - 1) ^ (int)g_z;
    raw = pd_stage_fetch(lane, nstage - 1, st_rx, st_col, inv_ncol_u, r0, c0, ncol PD_A);
#if WG_TF_TWO
    // the lane's second texel, lane + kThreads (fixed per lane like the first)
    const int lane2 = lane + kThreads;
    const int st_r2 = (int)(((float)lane2 + 0.5f) * inv_ncol);
    const int st_rx2 = st_r2 ^ (int)g_z, st_col2 = clampi(c0 + (lane2 - mul_i24(st_r2, ncol)), 0, P_rw - 1) ^ (int)g_z;
    TfRaw raw2 = pd_stage_fetch(lane2, nstage - 1, st_rx2, st_col2, inv_ncol_u, r0, c0, ncol PD_A);
#endif
#else
    {
        int i = mini(lane, nstage - 1);
        int r = (int)(((float)i + 0.5f) * inv_ncol), c = i - mul_i24(r, ncol);
        const int rowc = clampi(r0 + r, 0, P_rh - 1), colc = clampi(c0 + c, 0, P_rw - 1);
        raw = tf_fetch((int)mul24((uint)rowc, (uint)P_rw) + colc, colc, rowc PD_A);
    }
#if WG_TF_TWO
    TfRaw raw2 = tf_stage2(lane, nstage, inv_ncol, r0, c0, ncol PD_A);
#endif
#endif
#if POST_OUT == 3
    // Direct form: the per-launch values after the first staging loads (lev-f16lut 1a9daed: with the exposure-pair loads
    // first, every wave waited for them before issuing any texel load)
    float dec_scale = dec_scale_arg;
    if (expo != 0u) {   // kDiet: ratio and dec_scale precomputed in the exposure pair
        float e0 = unif(wg_ldf(expo, 0u)), e1 = unif(wg_ldf(expo, 4u));
        g_e = e0 == 0.0f ? 1.0f : e0;
        g_e_prev = e1 == 0.0f ? 1.0f : e1;
        g_ratio = unif(wg_ldf(expo, 12u));
        dec_scale = unif(wg_ldf(expo, 16u));
    }
    // the output table of this frame (wg_f16lut.hlsli; color: the table, 0 = none), wave-uniform
#if WG_LZ_ANALYTIC
    const bool lut = false;   // the table is indexed by an FP16 filter result (the original's); the analytic one is FP32
#else
    const bool lut = WaveReadLaneFirst(f16lut_valid(color, dec_scale, nd_color_pi, nd_output_rne16) ? 1u : 0u) != 0u;
#endif
#endif

    // this lane's D column (fixed over the walk)
    const int xv = X0 - 2 + wl;
    const int x = clampi(xv, 0, P_dw - 1);
    const bool own_x = and(and(xv >= X0, xv < X0 + kTfOwn), xv < P_dw);
    const float px = pos_x(x);
    const int ax = floor_i(px - 0.5f);
    const int fx = clampi(floor_i(px), 0, P_rw - 1);
    const int cb0 = 48 * (ax - c0), cbf = 48 * (floor_i(px) - c0);
    const uint wlu = (uint)wl;
#if defined(WG_POST_MO) && defined(WG_DIRECT_TEXTURES)
    const uint2 mo_dims = wg_mo_dims();
#elif defined(WG_POST_MO)
    const uint2 mo_dims = uint2(0u, 0u);
#endif

#if POST_OUT == 3
    // the horizontal pass's lane roles (row hk, channel hc, cell hg) and its scratch offsets: fixed per lane
    const int hk_l = wl >= 30 ? 1 : 0, hrem_l = wl - 30 * hk_l, hc_l = hrem_l >= 20 ? 2 : (hrem_l >= 10 ? 1 : 0),
              hg_l = hrem_l - 10 * hc_l;
    // (opaque: the compiler would otherwise rebuild the sums from their parts in every step)
    const uint h_src = ((uint)hk_l * kRowDw + (uint)hc_l * 32u + (uint)hg_l * 3u) ^ g_z;
    const uint h_dst = ((uint)hk_l * kRowDw + (uint)hc_l * (uint)kTfOutX + (uint)hg_l * 4u) ^ g_z;
    const bool h_lane = wl < 60;
    // the vertical pass's first item of each step is it = lane: its values, fixed for the walk (opaque: computed once)
    int v_yrel, v_xo, v_urel;
    uint v_wa, v_wb;
    pd_vsplit(lane, v_yrel, v_xo, v_urel, v_wa, v_wb);
    v_yrel ^= (int)g_z; v_urel ^= (int)g_z;
    const int v_xg = (strip * kTfOutX + v_xo) ^ (int)g_z;
    v_xo ^= (int)g_z; v_wa ^= g_z; v_wb ^= g_z;
#endif
    int base_slot = 0;
    int p_next = P0;
    [loop] for (int y_step = r_first; y_step <= r_need; y_step += 2 * kNW) {
        // 1. stage the footprint
#if defined(WG_POST_MO)
        // the raw motion of this lane's row pair, loaded before the staging and the gather (HIP's mo_fetch)
        float2 mq0, mq1;
        mo_fetch(x, y_step + 2 * wave, mo_dims, mq0, mq1);
#endif
        if (lane < nstage) tf_put((uint)lane, raw, g_e PD_A);
#if WG_TF_TWO
        if (lane + kThreads < nstage) tf_put((uint)(lane + kThreads), raw2, g_e PD_A);
#endif
        wg_barrier_lds();
        const int r0c = r0;

        // 2. reconstruction of the wave's row pair (yv, yv + 1)
        const int yv = y_step + 2 * wave;
        const bool wact = yv <= r_need;
        int slot = base_slot + 2 * wave;
        if (slot >= kRing) slot -= kRing;
        const uint scr = kRingBase + (uint)slot * kRowDw;   // dword of ring row slot (rows slot, slot + 1)
        if (wact) {
            const int y0 = yv;
            const bool has1 = y0 + 1 < P_dh;
            const float py0 = unif(pos_y(y0)), py1 = unif(pos_y(y0 + 1));   // wave-uniform (y0 is)
            const int b = wg_uniform(floor_i(py0 - 0.5f));
            const int jend = 1 + (int)g_z;   // = 1; opaque: the driver keeps the loop rolled (as HIP's nounroll; fewer VGPRs)
            float hsum[3] = { 0.0f, 0.0f, 0.0f };
            Gather gnext;   // row y0 + 1's gather sums, made in row y0's pass
            [unroll] for (int k = 0; k < 2; ++k) {
                if (k == 1 && !has1) break;
                const int y = y0 + k;
                const float py = unif(k == 0 ? py0 : py1);
                const int fy = clampi(floor_i(py), 0, P_rh - 1);
                float hh[3], oh3[3];
#if defined(WG_POST_MO)
#else
                const int pyi = floor_i(py);
                float mx[3], my[3];
                {
                    const uint tf = ((uint)(pyi - r0c) * rowb + (uint)cbf) >> 2;
                    const uint mn = lds[tf + 9], mfar = lds[tf + 11];
                    mx[0] = f16tof32(mn); my[0] = f16tof32(mn >> 16);
                    mx[2] = f16tof32(mfar); my[2] = f16tof32(mfar >> 16);
                    // motion_bilinear_mix_rel of m_center
                    const float tx = px - 0.5f, ty = py - 0.5f;
                    const float x0f = floor(tx), y0f = floor(ty);
                    const float wx = tx - x0f, wy = ty - y0f;
                    const int rr = (int)y0f;
                    const uint ra = ((uint)(rr - r0c) * rowb + (uint)cb0) >> 2;
                    const uint rb = ((uint)(rr + 1 - r0c) * rowb + (uint)cb0) >> 2;
                    const uint ma = lds[ra + 10], mb = lds[ra + kTexDw + 10];
                    const uint mc = lds[rb + 10], md = lds[rb + kTexDw + 10];
                    const float owx = 1.0f - wx;
                    const float topx = hw_lo(ma, owx) + hw_lo(mb, wx), topy = hw_hi(ma, owx) + hw_hi(mb, wx);
                    const float botx = hw_lo(mc, owx) + hw_lo(md, wx), boty = hw_hi(mc, owx) + hw_hi(md, wx);
                    const float owy = 1.0f - wy;
                    mx[1] = topx * owy + botx * wy;
                    my[1] = topy * owy + boty * wy;
                }
#endif
                // the kernel gather of rows y0 and y0 + 1 in one pass over the texels (texel rows b - 1 ... b + 1, post.hip's
                // order): each texel's features are read from LDS once for both rows. Row y0 + 1's sums wait in gnext.
                Gather gk;
                if (k == 0) {
#if POST_OUT == 3
                    // unrolled: the first texel starts the sums from zeros as operands (no zero moves) and the loop's
                    // overhead goes (78 VGPRs). The colour sums start from an opaque +0 (gather_zero_opq): with a
                    // literal 0 the driver turns fma(w, ec, +0) into w * ec (a -0 product would stay -0).
                    Gather g0 = gather_zero_opq(), g1 = gather_zero_opq();
                    [unroll] for (int j = -1; j <= 1; ++j) {
#else
                    Gather g0 = gather_zero(), g1 = gather_zero();
                    [loop] for (int j = -1; j <= jend; ++j) {
#endif
                        const float bj = opq((float)(b + j) + 0.5f);
                        const float v0 = bj - py0, v1 = bj - py1;
                        // (kTapK * v) * v, each product as mad(.., -0.0f) (exact): the driver then keeps it on SALU
                        const float vv0 = mad(mad(kTapK, v0, -0.0f), v0, -0.0f), vv1 = mad(mad(kTapK, v1, -0.0f), v1, -0.0f);
                        const uint rowoff = (uint)(b + j - r0c) * rowb;
                        [unroll] for (int i = -1; i <= 1; ++i) {
                            const float u = opq((float)(ax + i) + 0.5f) - px;
                            const float uu = opq(opq(kTapK * u) * u), c2u = opq(kTapK2 * u);
                            const TexF t = tex_features((rowoff + (uint)(cb0 + 48 * i)) >> 2);
                            gather_add_f(g0, t, uu, vv0, mul_nz(c2u, v0));
                            gather_add_f(g1, t, uu, vv1, mul_nz(c2u, v1));
                        }
                    }
                    gk = g0;
                    gnext = g1;
                } else {
                    gk = gnext;
                }
#if defined(WG_POST_MO)
                mo_core(gk, k == 0 ? mq0 : mq1, x, y, fx, fy, s, hh, oh3 PD_A);
#else
                post_core_s(gk, x, y, fx, fy, mx, my, s, hh, oh3 PD_A);
#endif
                // the FP16 conversions here, in wave-uniform control flow (cvt3_f16): (_Float16) enc_e(hh) as HIP's
                // v_fma_mixlo_f16 log2(..), kEncK, 0 (one rounding): the driver folds mad(.., kEncK, 0) into its
                // conversion as exactly that instruction (RGA, all four forms; was post_rto_mul + v_cvt_f16_f32,
                // about 13 VALU per channel), and the display values (display_value: RNE)
                const uint2 he = cvt3_f16(mad(log2(enc_arg(hh[0])), kEncK, 0.0f), mad(log2(enc_arg(hh[1])), kEncK, 0.0f),
                                          mad(log2(enc_arg(hh[2])), kEncK, 0.0f));
                const uint2 de = cvt3_f16(clampf(enc_e(oh3[0]), 0.0f, kDisplayMax),
                                          clampf(enc_e(oh3[1]), 0.0f, kDisplayMax),
                                          clampf(enc_e(oh3[2]), 0.0f, kDisplayMax));
                if (and(and(own_x, y >= own_r0), y < own_r1))
                    wg_st64(hist_out, mul24((uint)y, (uint)P_dw * 8u) + (uint)x * 8u, he);
                // display values (FP16) of the 64 columns -> scratch row slot + k: lanes 2m, 2m + 1 share a dword
                const uint srow = scr + (uint)k * kRowDw;
#if POST_OUT == 3
                {   // the partner's two dwords (c0 | c1 << 16, c2), then the even lane's three packed dwords (the same bits)
                    const uint nx = QuadReadAcrossX(de.x), ny = QuadReadAcrossX(de.y);
                    const uint d0 = (de.x & 0xffffu) | (nx << 16), d1 = (de.x >> 16) | (nx & 0xffff0000u);
                    const uint d2 = de.y | (ny << 16);
                    if ((wlu & 1u) == 0u) {
                        const uint a = srow + (wlu >> 1);
                        lds[a] = d0;
                        lds[a + 32u] = d1;
                        lds[a + 64u] = d2;
                    }
                }
                // hsum = hh0 + hh1 (no + 0 first): it differs from (0 + hh0) + hh1 only in the sign of a zero, and the
                // reduced code of a zero is that of either sign (enc_arg(+-0) = 1)
                [unroll] for (int c2i = 0; c2i < 3; ++c2i) hsum[c2i] = k == 0 ? hh[c2i] : hsum[c2i] + hh[c2i];
#else
                [unroll] for (int c = 0; c < 3; ++c) {
                    const uint dv = c == 0 ? de.x & 0xffffu : (c == 1 ? de.x >> 16 : de.y);
                    const uint nbv = QuadReadAcrossX(dv);
                    if ((wlu & 1u) == 0u) lds[srow + (uint)c * 32u + (wlu >> 1)] = dv | (nbv << 16);
                }
                [unroll] for (int c2i = 0; c2i < 3; ++c2i) hsum[c2i] = hsum[c2i] + hh[c2i];
#endif
            }
            // reduced history: mean of H^ over the 2 x 2 block, the partner column in lane ^ 1
#if POST_OUT == 3
            // two code chains for the three channels: chain a is channel 0 on the even lane and channel 1 on the odd lane
            // (its sum hsum_o[1] + hsum_e[1]: the even lane's hsum_e[1] + hsum_o[1] commuted, equal but for a NaN's
            // payload, and every NaN m gives the code of enc_arg = 1e-4), chain b channel 2; the odd lane's code a comes
            // to the even lane by one exchange
            {
                const bool odd = (wl & 1) != 0;
                const float ha = odd ? hsum[1] : hsum[0], hs = odd ? hsum[0] : hsum[1];
                const float na = QuadReadAcrossX(hs), nb2 = QuadReadAcrossX(hsum[2]);
                const uint qa = rh_code(ha + na), qb = rh_code(hsum[2] + nb2);
                const uint q1 = QuadReadAcrossX(qa);
                if (and(and(and(and(and(!odd, own_x), x + 1 < P_dw), has1), y0 >= own_r0), y0 + 1 < own_r1))
                    wg_st32(reduced, mul24((uint)(y0 >> 1), (uint)(P_dw >> 1) * 4u) + (uint)(x >> 1) * 4u,
                            qa | (q1 << 10) | (qb << 20));
            }
#else
            float nb[3];
            [unroll] for (int c = 0; c < 3; ++c) nb[c] = QuadReadAcrossX(hsum[c]);
            if (and(and(and(and(and((wl & 1) == 0, own_x), x + 1 < P_dw), has1), y0 >= own_r0), y0 + 1 < own_r1)) {
                uint code = 0u;
                [unroll] for (int c = 0; c < 3; ++c) {
                    const float m = (hsum[c] + nb[c]) * 0.25f;
                    const float vq = clampf((enc_e(m)) * kInv245, 0.0f, 1.0f);
                    code |= (uint)round(vq * 1023.0f) << (10u * (uint)c);
                }
                wg_st32(reduced, mul24((uint)(y0 >> 1), (uint)(P_dw >> 1) * 4u) + (uint)(x >> 1) * 4u, code);
            }
#endif
        }
        // the next step's staging loads (every lane loads a valid clamped texel)
        {
            int ys = y_step + 2 * kNW;
            int yl = mini(ys + 2 * (kNW - 1), r_need) & ~1;
            r0 = floor_i(pos_y(ys) - 0.5f) - 1;
            int nrow = floor_i(pos_y(yl) - 0.5f) + 2 - r0 + 1;
            nstage = ncol * nrow < kTfCap ? ncol * nrow : kTfCap;
#if POST_OUT == 3
            raw = pd_stage_fetch(lane, nstage - 1 < 0 ? 0 : nstage - 1, st_rx, st_col, inv_ncol_u, r0, c0, ncol PD_A);
#if WG_TF_TWO
            raw2 = pd_stage_fetch(lane2, nstage - 1 < 0 ? 0 : nstage - 1, st_rx2, st_col2, inv_ncol_u, r0, c0, ncol PD_A);
#endif
#else
            int i = mini(lane, nstage - 1 < 0 ? 0 : nstage - 1);
            int r = (int)(((float)i + 0.5f) * inv_ncol), c = i - mul_i24(r, ncol);
            const int rowc = clampi(r0 + r, 0, P_rh - 1), colc = clampi(c0 + c, 0, P_rw - 1);
            raw = tf_fetch((int)mul24((uint)rowc, (uint)P_rw) + colc, colc, rowc PD_A);
#if WG_TF_TWO
            raw2 = tf_stage2(lane, nstage, inv_ncol, r0, c0, ncol PD_A);
#endif
#endif
        }

        // horizontal pass of the wave's two rows: lane (row hk, channel hc, cell hg), output columns 4 hg ... 4 hg + 3
        tf_row_sync();
#if POST_OUT == 3
        const bool hact = and(wact, h_lane);
        const int hk = hk_l;
#else
        const int hl = wl;
        const bool hact = and(wact, hl < 60);
        const int hk = hl >= 30 ? 1 : 0, hrem = hl - 30 * hk, hc = hrem >= 20 ? 2 : (hrem >= 10 ? 1 : 0),
                  hg = hrem - 10 * hc;
#endif
#if POST_OUT == 3
        uint hv[4];   // set and used on the hact lanes only
#else
        uint hv[4] = { 0u, 0u, 0u, 0u };
#endif
        if (hact) {
#if POST_OUT == 3
            const uint src = scr + h_src;
#else
            const uint src = scr + (uint)hk * kRowDw + (uint)hc * 32u + (uint)hg * 3u;
#endif
#if WG_LZ_ANALYTIC
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
#else
            uint t[10];
            [unroll] for (int i = 0; i < 5; ++i) {
                const uint q = lds[src + (uint)i];
                t[2 * i] = q & 0xffffu;
                t[2 * i + 1] = q >> 16;
            }
            hv[0] = lz5_even(t, 0);
            hv[1] = lz5_odd(t, 2);
            hv[2] = lz5_even(t, 3);
            hv[3] = lz5_odd(t, 5);
#endif
        }
        tf_row_sync();
        if (hact) {
#if POST_OUT == 3
            const uint dst = scr + h_dst;
#else
            const uint dst = scr + (uint)hk * kRowDw + (uint)hc * (uint)kTfOutX + (uint)hg * 4u;
#endif
            [unroll] for (int i = 0; i < 4; ++i) lds[dst + (uint)i] = hv[i];
            if (slot + hk < 6) {
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
#if POST_OUT == 3
        if (lane < nitems)
            pd_vitem(2 * p_next + v_yrel, v_xg, v_xo, v_urel, v_wa, v_wb, p_next, interior, s0, r_first, lut, dec_scale PD_A);
        [loop] for (int it = lane + kThreads; it < nitems; it += kThreads) {
            int yrel, xo, urel;
            uint wa, wb;
            pd_vsplit(it, yrel, xo, urel, wa, wb);
            pd_vitem(2 * p_next + yrel, strip * kTfOutX + xo, xo, urel, wa, wb, p_next, interior, s0, r_first, lut,
                     dec_scale PD_A);
        }
#else
#if POST_OUT == 0
        [loop] for (int it = lane; it < nitems; it += kThreads) {
#else
        // RGBA16F forms: every lane of a wave runs every pass (lanes past the items repeat the last item, nothing stored),
        // so that the FP16 conversions can use the driver's conversion (wave-uniform control flow)
        [loop] for (int it0 = 0; it0 < nitems; it0 += kThreads) {
            const bool item_ok = it0 + lane < nitems;
            const int it = item_ok ? it0 + lane : nitems - 1;
#endif
            const int pr = (int)(mul24((uint)it, 52429u) >> 22);
            const int rem = it - (int)mul24((uint)pr, 2u * kTfOutX);
            const int dy = rem >= kTfOutX ? 1 : 0, xo = rem - dy * kTfOutX;
            const int pp = p_next + pr, y = 2 * pp + dy, xg = strip * kTfOutX + xo;
#if POST_OUT == 0
            if (or(y >= oh, xg >= ow)) continue;
#else
            const bool store_ok = and(item_ok, and(y < oh, xg < ow));
#endif
            uint t[5][3];
            if (interior) {   // one ring address, the 15 taps at constant offsets (mirrored rows: no wrap)
                int u = s0 + 3 * pr + 2 * dy;
                u = u >= kRing ? u - kRing : u;
                u = u >= kRing ? u - kRing : u;
                const uint rp = kRingBase + mul24((uint)u, (uint)kRowDw) + (uint)xo;
                [unroll] for (int k = 0; k < 5; ++k)
                    [unroll] for (int c = 0; c < 3; ++c) t[k][c] = lds[rp + (uint)(k * kRowDw + c * kTfOutX)];
            } else {
                const int xr = 3 * pp - 2 + 2 * dy;
                const int row0 = clampi(xr, 0, P_dh - 1);
                const int sl0 = (int)((uint)(row0 - r_first) % (uint)kRing);
                [unroll] for (int k = 0; k < 5; ++k) {
                    const int st = clampi(xr + k, 0, P_dh - 1) - row0;
                    const int sl = sl0 + st >= kRing ? sl0 + st - kRing : sl0 + st;
                    [unroll] for (int c = 0; c < 3; ++c)
                        t[k][c] = lds[kRingBase + (uint)sl * kRowDw + (uint)(c * kTfOutX + xo)];
                }
            }
            // emit (n = 1): weights of the lane's phase by selects (lz_wsel), lz5w, Dec, store 3 x f32
            const bool ev = dy == 0;
#if WG_LZ_ANALYTIC
            const float w0 = ev ? kLz0 : kLz4, w4 = ev ? kLz4 : kLz0, w1 = ev ? kLz1 : kLz3, w3 = ev ? kLz3 : kLz1;
#else
            const half w0 = hbox(ev ? lz_w0 : lz_w4), w4 = hbox(ev ? lz_w4 : lz_w0);
            const half w1 = hbox(ev ? lz_w1 : lz_w3), w3 = hbox(ev ? lz_w3 : lz_w1);
#endif
            float vo[3], qv[3];
            bool vsmall = false;
            [unroll] for (int c = 0; c < 3; ++c) {
#if WG_LZ_ANALYTIC
                qv[c] = clampf(lz5a(asfloat(t[0][c]), asfloat(t[1][c]), asfloat(t[2][c]), asfloat(t[3][c]), asfloat(t[4][c]),
                                    w0, w1, w3, w4), k1em4, 3.0f);
#else
                const uint r = lz5_original(t[0][c], t[1][c], t[2][c], t[3][c], t[4][c], w0, w1, hbox(lz_w2), w3, w4);
                qv[c] = clampf(f16tof32(r), k1em4, 3.0f);
#endif
                vo[c] = exp2(qv[c] * kDecK) - 1.0f;
                vsmall = or(vsmall, qv[c] < kDecPolyMax);
            }
            if (WaveActiveAnyTrue(vsmall)) {   // Dec's polynomial only when some lane needs it (post.hip's branches)
                [unroll] for (int c = 0; c < 3; ++c) vo[c] = qv[c] < kDecPolyMax ? dec_m1_poly(qv[c]) : vo[c];
            }
#if POST_OUT == 0
            const uint o = mul24((uint)y, (uint)ow * 12u) + mul24((uint)xg, 12u);
            wg_mem.Store3(out_p + o, uint3(asuint(vo[0] * dec_scale), asuint(vo[1] * dec_scale), asuint(vo[2] * dec_scale)));
#else
            // framebuffer_half((v dec_scale) pi): clamp to +-65504 (NaN passes), round to FP16 to nearest even; alpha 1. opq:
            // the driver would otherwise regroup the two uniform factors into v (dec_scale pi)
            const uint2 hb = cvt3_f16(fb_clamp(mul_nz(vo[0], dec_scale) * pi_arg), fb_clamp(mul_nz(vo[1], dec_scale) * pi_arg),
                                      fb_clamp(mul_nz(vo[2], dec_scale) * pi_arg));
#if POST_OUT == 1
            if (store_ok) wg_st64(out_p, mul24((uint)y, (uint)pitch_bytes) + (uint)xg * 8u, uint2(hb.x, hb.y | (0x3c00u << 16)));
#else
            // the FP16 values widened to FP32 (exact); the UAV's format conversion stores them unchanged
            if (store_ok)
                wg_tex_out[uint2((uint)xg, (uint)y)] = float4(f16tof32(hb.x), f16tof32(hb.x >> 16), f16tof32(hb.y), 1.0f);
#endif
#endif
        }
#endif
        p_next = p_rdy;
        base_slot += 2 * kNW;
        if (base_slot >= kRing) base_slot -= kRing;
    }
}

#if POST_OUT == 3
WG_KERNEL(post_tf_f32_p12_diet_vsit_direct)
{
    WG_SETUP();
    // launch-uniform: one copy of the body runs (pd_rc / pd_pi1 false: the run-time tests as before)
    const bool rc = and(P_rw == (int)nd_rw, P_rh == (int)nd_rh);
    [branch] if (and(rc, nd_color_pi == 1.0f)) post_body(true, true);
    else if (rc) post_body(true, false);
    else post_body(false, false);
}
#endif
