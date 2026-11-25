// SPDX-License-Identifier: MIT
// clean/rr-f-wave4b R4b.1: D0-R of spec 55 in wave64 (design D of kernels/rr_block_c32.hlsl), the relaxed head of
// head.fast.rx only: rr_d0r.hlsl with RR_HVARL 7, RR_S9 61, RR_OUT_PREP 1, RR_HSTM 1, RR_HQ8S 1, RR_HEAD18 1,
// WG_BLOCK 128, WG_WAVE 64 (the file refuses any other configuration). Same arguments, buffer layouts and output bytes.
//
// One workgroup = one 8 x 8 window and two wave64 waves; wave w owns the window tokens of the wave32 waves wv = 2 w
// (tile 0) and 2 w + 1 (tile 1) of rr_d0r.hlsl (16 tokens each, token = 16 wv + r). Wave64 lane L = l + 32 hh holds,
// for each tile, half hh of wave32 lane l's fragment (l = 16 hf + r): fp8 A / B bytes 4 hh .. 4 hh + 3, accumulator
// elements 4 hh .. 4 hh + 3 (rr_block_c32.hlsl, the v_wmma wave64 layout). A v8f holds elements 0-3 of tile 0 and 0-3
// of tile 1; a uint2 fp8 fragment: x = tile 0, y = tile 1; a uint4 of packed f16: x, y = tile 0, z, w = tile 1. Every
// value is computed by the operation rr_d0r.hlsl computes it with (wave64 v_wmma = wave32 v_wmma bit for bit, the
// conversions are per value), so the stored bytes are those of the wave32 kernel; what moves is only where a value
// sits:
//   * the weight fragments (shared by both tiles) are loaded once, 4 bytes per lane; so are the key / V^T fragments
//     of the q / v exchange (every wave reads the same data) and the biases;
//   * the row sum Z is a sequential f32 dot2add chain over the 32 weights of a lane in wave32 (order J, then the 8
//     elements): a lane of wave32 holds elements 0-7 of every key tile, a wave64 lane only 4 of them for two tiles.
//     The other half's words of the lane's own tile are fetched from lane ^ 32 (hh 0 runs tile 0's chain, hh 1 tile 1's)
//     so the chain, and the z + z(lane ^ 16) that follows, are the wave32 operations on the wave32 operands, and the
//     reciprocal of the other tile comes back through one more lane ^ 32 exchange;
//   * the head tail (55.2 / 55.3, stores, render records) is rr_d0r.hlsl's text, run on each 32-lane half of the
//     wave64 for its tile: before it the lane gets the y2h words of its tile that the other half holds (lane ^ 32), so
//     Y0 = channels 8 hf .. 8 hf + 7 and Y1.x = channels 16 + 8 hf, 17 + 8 hf (the only y2h[1] word the tail reads)
//     are the wave32 lane's words.
// Wave-uniform votes (the Q8 clamp test, the activation's rare path, the record flag) cover both tiles: both forms of
// those groups give the same bits for every value, and the flag word is an OR.
#include "wg_platform.hlsli"
#if WG_WAVE != 64 || WG_BLOCK != 128
#error "rr_d0r_w64.hlsl: WG_BLOCK 128, WG_WAVE 64 only"
#endif
#if !defined(RR_HVARL) || RR_HVARL != 7 || !defined(RR_S9) || RR_S9 != 61 || !defined(RR_OUT_PREP) || RR_OUT_PREP != 1 || \
    !defined(RR_HSTM) || RR_HSTM != 1 || !defined(RR_HQ8S) || RR_HQ8S != 1
#error "rr_d0r_w64.hlsl: the head.fast.rx configuration only (RR_HVARL 7, RR_S9 61, RR_OUT_PREP 1, RR_HSTM 1, RR_HQ8S 1)"
#endif
#ifdef RR_STRICT_STEP
#if RR_STRICT_STEP != 0
#error "rr_d0r_w64.hlsl: RR_STRICT_STEP 0 only"
#endif
#endif
#include "wg_ags.hlsli"
wg_frag wg_frag_v8f(v8f c) { wg_frag f; [unroll] for (uint i = 0; i < 8; ++i) f.v[i] = asuint(c.v[i]); return f; }
v8f wg_v8f_frag(wg_frag f) { v8f c; [unroll] for (uint i = 0; i < 8; ++i) c.v[i] = asfloat(f.v[i]); return c; }
wg_frag rr_w64f(uint a, uint b, uint c, uint d)   // a wave64 fragment: the first 4 / 2 / 1 dwords are used
{
    wg_frag f;
    f.v[0] = a; f.v[1] = b; f.v[2] = c; f.v[3] = d;
    [unroll] for (uint i = 4; i < 8; ++i) f.v[i] = 0u;
    return f;
}
v8f wg_wmma_fp8(uint2 a, uint2 b, v8f c)   // tile 0: (a.x, b.x, c 0-3), tile 1: (a.y, b.y, c 4-7)
{
    const wg_frag r0 = wg_ags_mma(WG_AGS_FP8_F32, rr_w64f(a.x, 0u, 0u, 0u), rr_w64f(b.x, 0u, 0u, 0u),
                                  rr_w64f(asuint(c.v[0]), asuint(c.v[1]), asuint(c.v[2]), asuint(c.v[3])));
    const wg_frag r1 = wg_ags_mma(WG_AGS_FP8_F32, rr_w64f(a.y, 0u, 0u, 0u), rr_w64f(b.y, 0u, 0u, 0u),
                                  rr_w64f(asuint(c.v[4]), asuint(c.v[5]), asuint(c.v[6]), asuint(c.v[7])));
    v8f d;
    [unroll] for (uint i = 0; i < 4; ++i) { d.v[i] = asfloat(r0.v[i]); d.v[4 + i] = asfloat(r1.v[i]); }
    return d;
}
uint2 wg_pack_q8_nc_ags(v8f d)   // v_cvt_pk_fp8_f32 per value pair, element i -> byte i of its tile's dword
{
    const wg_frag r0 = wg_ags_convert(rr_w64f(asuint(d.v[0]), asuint(d.v[1]), asuint(d.v[2]), asuint(d.v[3])),
                                      WG_AGS_T_F32_ACC, WG_AGS_T_FP8_B);
    const wg_frag r1 = wg_ags_convert(rr_w64f(asuint(d.v[4]), asuint(d.v[5]), asuint(d.v[6]), asuint(d.v[7])),
                                      WG_AGS_T_F32_ACC, WG_AGS_T_FP8_B);
    return uint2(r0.v[0], r1.v[0]);
}
uint4 wg_to_f16_ags(v8f d)       // v_cvt_f16_f32 per value (round to nearest even)
{
    const wg_frag r0 = wg_ags_convert(rr_w64f(asuint(d.v[0]), asuint(d.v[1]), asuint(d.v[2]), asuint(d.v[3])),
                                      WG_AGS_T_F32_ACC, WG_AGS_T_F16_ACC);
    const wg_frag r1 = wg_ags_convert(rr_w64f(asuint(d.v[4]), asuint(d.v[5]), asuint(d.v[6]), asuint(d.v[7])),
                                      WG_AGS_T_F32_ACC, WG_AGS_T_F16_ACC);
    return uint4(r0.v[0], r0.v[1], r1.v[0], r1.v[1]);
}
v8f rr_d8(uint2 q)   // e4m3 bytes -> f32 (exact), one conversion per tile
{
    const wg_frag r0 = wg_ags_convert(rr_w64f(q.x, 0u, 0u, 0u), WG_AGS_T_FP8_B, WG_AGS_T_F32_ACC);
    const wg_frag r1 = wg_ags_convert(rr_w64f(q.y, 0u, 0u, 0u), WG_AGS_T_FP8_B, WG_AGS_T_F32_ACC);
    v8f d;
    [unroll] for (uint i = 0; i < 4; ++i) { d.v[i] = asfloat(r0.v[i]); d.v[4 + i] = asfloat(r1.v[i]); }
    return d;
}
v8f zero8() { return wg_splat(0.0f); }

#ifndef RR_P55_ERRATUM
#define RR_P55_ERRATUM 1
#endif
#define RR_HEAD18 1
#define RR_HCVB 1
#define RR_HTRIM 7

WG_ARGS_BEGIN
    uint p_x, p_wf, p_bias, p_tb, p_hold, p_hc, p_tab, p_rho;   // arena byte offsets (0 = null)
    uint p_cnet, p_hid, p_gates, p_prec, p_candw;
    uint p_dy, p_dimg, p_dz, p_zs;
    uint Wt, Ht, Wr, Hr, sx, sy, nwx;
    uint k_as, k_acl, k_acln, k_p1, k_p0, k_actcl, k_actcln, k_c1, k_c2, k_c3, k_ra, k_m, k_g;  // f16 bits
    uint k_s1;                                                                                   // f32 bits
    uint p_flag;
    uint p_wpl;
    uint p_oscr, o_wh, o_hh, k_o2n, k_o2x2n, k_o4, k_o5, k_o7;   // output scratch, history size, out.K2n / K2x2n / K4 / K5 / K7 (f32)
WG_ARGS_END

// Weight fragments (256 B each = 32 lanes x 8 B; lane l, byte i = W[16 ks + 8 (l >> 4) + i][16 nt + (l & 15)]).
#define F_Q 0u    // Wq [h][ft][ks]
#define F_V 8u    // Wv [h][ft][ks]
#define F_O 16u   // Wo [h][ct][ks]
#define F_1 24u   // W1 [nt 8][ks 2]
#define F_2 40u   // W2 [ct 2][ks 8]
// p_bias: f32 copies of bo [32] (byte 0), b1 [128] (byte 128), b2 [32] (byte 640)
// p_tab: sig of 51.4 (f16 [65536], byte 0), R(tanh) (f16 [65536], byte 131072); p_rho: rho (f32 [65536]) for this s2
#define TAB_TANH 131072u
#define TAB_EXP2 262144u    // 55-ERRATUM-2: R(2^t) for every f16 t (clean/rr-f-flow)
#define TAB_RCP16 393216u   // 55-ERRATUM-2: R(1/v) for every f16 v

uint rr_zero() { return 0u - (wg_grid_ >> 31); }           // 0 with no known bits (README "Driver exceptions")
uint rr_fence(uint v) { return v ^ rr_zero(); }
uint4 rr_fence4(uint4 v) { return v ^ rr_zero(); }
half2 H2(uint v) { return wg_as_h2(v); }
uint U2(half2 h) { return wg_h2_bits(h); }
half2 HF(half2 h) { return H2(rr_fence(U2(h))); }            // fenced f16 pair: a rounding point stays a rounding point
float FF(float f) { return asfloat(rr_fence(asuint(f))); }
uint rr_u(uint p) { return WaveReadLaneFirst(p); }       // a pointer argument read as wave-uniform (store-merge hazard)
uint rr_f2h(float x) { return wg_ags_call(0x1cu, 0u, 1u, asuint(x), 0u); }   // hardware RNE f32 -> f16 (op 0x1C imm 1)
// rr_output.hlsl / rr_output_fast.hlsli's fence, Cf, Mq and of_d255, operation for operation
float rr_ofence(float x) { return asfloat(asuint(x) ^ (0u - (wg_grid_ >> 31))); }
float rr_ocf(float x)
{
    return rr_ofence(rr_ofence(log2(max(mad(x, asfloat(k_o4), 1.0f), asfloat(k_o5))) * asfloat(0x3e9a209bu)) * asfloat(k_o7));
}
float rr_omq(float a, float b, inout bool rare)
{
    const float r = rr_ofence(a * b);
    [branch] if (r == 0) {
        const float s2 = rr_ofence(a * asfloat(0x5f800000u)) * b;
        rare = or(rare, and(abs(s2) >= asfloat(0x207ffff0u), abs(s2) <= asfloat(0x20800000u)));
    }
    return r;
}
float rr_od255(uint b)
{
    const float a = (float)b, rc = asfloat(0x3b808081u);
    const float q = a * rc;
    return mad(mad(-255.0f, q, a), rc, q);
}
half2 rr_c2(uint b) { return H2((b & 0xffffu) * 0x10001u); }

v8f rr_widen(uint4 p)
{
    v8f r;
    r.v[0] = f16tof32(p.x); r.v[1] = f16tof32(p.x >> 16); r.v[2] = f16tof32(p.y); r.v[3] = f16tof32(p.y >> 16);
    r.v[4] = f16tof32(p.z); r.v[5] = f16tof32(p.z >> 16); r.v[6] = f16tof32(p.w); r.v[7] = f16tof32(p.w >> 16);
    return r;
}

// Border rule of 51.6 for extents >= 8 (level 0 and level 1 here: Wt, Ht >= 32): the single reflection.
int rr_refl(int v, int n) { const int a = abs(v); return a >= n ? 2 * (n - 1) - a : a; }

// speed_swin.hlsli's Q8 helpers (the same text): the clamped pack, the wave-uniform skip of the clamp
uint2 sw_pack_q8_v2(v8f d)
{
    v8f c;
    [unroll] for (uint i = 0; i < 8; ++i) c.v[i] = max(min(d.v[i], 448.0f), -448.0f);
    const uint2 pc = wg_pack_q8_nc_ags(c), px = wg_pack_q8_nc_ags(d);
    const uint2 m = px & ~pc & 0x80808080u;
    return pc | m | (m >> 7);
}
float sw_absmax8(float m, v8f d)
{
    return max(max(max(m, abs(d.v[0])), max(abs(d.v[1]), abs(d.v[2]))),
               max(max(abs(d.v[3]), abs(d.v[4])), max(max(abs(d.v[5]), abs(d.v[6])), abs(d.v[7]))));
}
bool sw_in448(float m) { return WaveActiveAllTrue(m <= 448.0f); }
// Q8 of 51.1 for groups of tiles of f16 values held in f32: saturation at +-448, RNE; one wave-uniform test skips the
// clamp when no value of the wave needs it (NaN -> 0xff, see DESIGN-55 section 4).
#define RR_Q8(N, src, dst)                                                                                  \
    {                                                                                                       \
        float am_ = 0.0f;                                                                                   \
        [unroll] for (uint qi_ = 0; qi_ < N; ++qi_) am_ = sw_absmax8(am_, src[qi_]);                         \
        if (sw_in448(am_)) { [unroll] for (uint qj_ = 0; qj_ < N; ++qj_) dst[qj_] = wg_pack_q8_nc_ags(src[qj_]); } \
        else { [unroll] for (uint qk_ = 0; qk_ < N; ++qk_) dst[qk_] = sw_pack_q8_v2(src[qk_]); }            \
    }

// 51.3 step 3 on a pair of f16 scores: u = R(s A_SCALE), clamp (a NaN gives A_CLAMPn), p = R(c R(A_P1 - c c) + A_P0),
// w = 2^(n-15) (1 + y - n), y = 32 (p - 1): for p in [1 + 1/32, 1 + 31/32) the f16 bits of w are (bits(p) << 5) & 0x7fe0.
uint rr_attw(uint sh, half2 as2, half2 acln2, half2 acl2, half2 p12, half2 p02)
{
    const half2 u = H2(sh) * as2;
    const half2 c = min(max(u, acln2), acl2);
    const half2 inner = mad(-c, c, p12);
    const half2 p = mad(c, inner, p02);
    const uint16_t2 pb = asuint16(p) << (uint16_t)5;
    return ((uint)pb.x | ((uint)pb.y << 16)) & (0x7fe07fe0u & ~rr_zero());
}
// Q8 of 8 packed f16 values known finite and within +-448 (P) through the driver's f16 -> fp8 conversion; per tile
// (h.xy = tile 0, h.zw = tile 1)
uint2 rr_q8h_small(uint4 h)
{
    const wg_frag r0 = wg_ags_convert(rr_w64f(h.x, h.y, 0u, 0u), WG_AGS_T_F16_ACC, WG_AGS_T_FP8_B);
    const wg_frag r1 = wg_ags_convert(rr_w64f(h.z, h.w, 0u, 0u), WG_AGS_T_F16_ACC, WG_AGS_T_FP8_B);
    return uint2(r0.v[0], r1.v[0]);
}
// The same conversion for any f16 values, with RR_Q8's result (rr_d0r.hlsl rr_q8h)
uint2 rr_q8h(uint4 h, inout uint bad)
{
    const uint2 q = rr_q8h_small(h);
    bad |= ((q.x & 0x7f7f7f7fu) + 0x01010101u) | ((q.y & 0x7f7f7f7fu) + 0x01010101u);
    return q;
}

// 51.3 step 8 activation on a pair: k = clamp, u = R(C2 - |k| C1), v = R(k u + C3), act = R(x v)
uint rr_act(uint xh, half2 cln2, half2 cl2, half2 c12, half2 c22, half2 c32)
{
    const half2 x = H2(xh);
    const half2 k = min(max(x, cln2), cl2);
    const half2 ak = max(k, -k);
    const half2 u = mad(-ak, c12, c22);
    const half2 v = mad(k, u, c32);
    return U2(x * v);   // consumed by rr_q8h (no fold into that conversion) or fenced before a widening
}

// sigx(dd) / 2 = 0.5 / (1 + e^-dd) in f32 (CANDW, 55.3: within about 2 f32 ulp before the rounding to f16).
float rr_lhalf(float dd)
{
    const float t = min(max(-dd, -100.0f), 87.0f);   // e^t stays finite; beyond, the f16 result is 1/2 or 0 anyway
    const float L = asfloat(0x3fb8aa3bu), Llo = asfloat(0x32a57060u), LN2 = asfloat(0x3f317218u);
    const float th = t * L;
    const float tl = mad(t, L, -th) + t * Llo;        // t log2(e) = th + tl
    const float e0 = exp2(th);
    const float e = mad(e0, tl * LN2, e0);
    const float den = 1.0f + e;
    float y = rcp(den);
    y = mad(mad(-den, y, 1.0f), y, y);
    const float v = 0.5f * y;
    return isnan(dd) ? dd : v;
}

// 8-bit stores of 55.3: unsigned normalised (NaN -> 0 by the max), HID (page 53.3 as quoted on 55.3)
uint rr_u8(uint h) { const float c = min(max(f16tof32(h), 0.0f), 1.0f); return (uint)round(c * 255.0f); }
uint rr_s8(uint h)
{
    const float f = f16tof32(h);
    const float c = isnan(f) ? 0.0f : min(max(f, -1.0f), 1.0f);
    return ((uint)(int)round(c * 127.0f)) & 0xffu;
}

// the q / v exchange: the same 8192 bytes as rr_d0r.hlsl's ([head 2][wave 4][frag 4: q ft0, q ft1, vT ft0, vT ft1][lane
// 32] x 8 B), addressed per dword: wave32 slot i (8 bytes) = dwords 2 i, 2 i + 1; a wave64 lane holds dword 2 i + hh
groupshared uint rr_xq32[2048];

// per-lane offsets, fenced (the same value with no known bits) so that the compiler adds them to the base once and
// keeps the constant parts in the instructions' offset fields
static uint rr_lo;     // lp * 8 + 4 hh: byte of a fragment-ordered 8-byte slot
static uint rr_xo;     // 2 lp + hh: dword of an LDS slot row
static uint rr_t4;     // 8 hf + 4 hh: byte within a 16-channel group of a token / an output tile row
uint wfrag1(uint base, uint f) { return wg_ld32(base + rr_lo, f * 256u); }
uint2 wfrag(uint base, uint f) { const uint w = wfrag1(base, f); return uint2(w, w); }   // shared by both tiles
v8f rr_dup4(float4 a)
{
    v8f r;
    r.v[0] = a.x; r.v[1] = a.y; r.v[2] = a.z; r.v[3] = a.w; r.v[4] = a.x; r.v[5] = a.y; r.v[6] = a.z; r.v[7] = a.w;
    return r;
}
v8f rr_cat4(float4 a, float4 b)
{
    v8f r;
    r.v[0] = a.x; r.v[1] = a.y; r.v[2] = a.z; r.v[3] = a.w; r.v[4] = b.x; r.v[5] = b.y; r.v[6] = b.z; r.v[7] = b.w;
    return r;
}
v8f bias_tile(uint p, uint off)   // per tile the same bias fragment: bytes off + 4 (8 hf + 4 hh) ..
{
    return rr_dup4(wg_ld4f(p + 4u * rr_t4, off));
}
// per tile: this lane's dword of the 8-byte group at byte o0 (tile 0) / o1 (tile 1)
uint2 rr_tld(uint base, uint o0, uint o1) { return uint2(wg_ld32(base + rr_t4, o0), wg_ld32(base + rr_t4, o1)); }
uint rr_xqs(uint row) { return rr_xq32[rr_xo + 64u * row]; }   // dword of the wave32 slot (row * 32 + lp): a shared operand

WG_KERNEL(rr_d0r_head)
{
    WG_SETUP();
    const uint L = WaveGetLaneIndex();
    const uint lp = L & 31u, r = L & 15u, hf = (L >> 4) & 1u, hh = L >> 5;
    rr_lo = rr_fence(lp * 8u + 4u * hh);
    rr_xo = rr_fence(lp * 2u + hh);
    rr_t4 = rr_fence(8u * hf + 4u * hh);
    const uint wv0 = 2u * WaveReadLaneFirst(wg_lane_id() >> 6);   // wave32 wave of tile 0 (tile 1: wv0 + 1)
    const uint win = wg_block_id();
    const uint wj = win / nwx, wi = win - wj * nwx;
    const int X = (int)(8u * wi + (r & 7u)) - (int)(sx & 0xffu);
    const bool hmode = ((sx >> 8) & 1u) != 0u;   // 18-channel reading (RR_HEAD18)
    int Yt[2];
    uint tokt[2];
    [unroll] for (uint t0 = 0; t0 < 2u; ++t0) {
        Yt[t0] = (int)(8u * wj + 2u * (wv0 + t0) + (r >> 3)) - (int)sy;
        tokt[t0] = (uint)rr_refl(Yt[t0], (int)Ht) * Wt + (uint)rr_refl(X, (int)Wt);
    }

    uint4 y2h[2];
    // ---------------------------------------------------------------- 51.3 step 1: q_h, v_h (both heads)
    const uint2 xb0 = rr_tld(p_x, tokt[0] * 32u, tokt[1] * 32u), xb1 = rr_tld(p_x, tokt[0] * 32u + 16u, tokt[1] * 32u + 16u);
    uint fq[8], fv[8];
    [unroll] for (uint i0 = 0; i0 < 8; ++i0) { fq[i0] = wfrag1(p_wf, F_Q + i0); fv[i0] = wfrag1(p_wf, F_V + i0); }
    // the q / v^T tiles of the two heads quantised in two groups of four (rr_d0r.hlsl RR_HQ8S)
    uint2 qv[8];
    [unroll] for (uint h0 = 0; h0 < 2; ++h0) {
        v8f qvt[4];   // [0] q ft0, [1] q ft1, [2] vT ft0, [3] vT ft1 of head h0
        [unroll] for (uint f0 = 0; f0 < 2; ++f0) {
            const uint b = (h0 * 2u + f0) * 2u;
            v8f a = wg_wmma_fp8(uint2(fq[b], fq[b]), xb0, zero8());
            a = wg_wmma_fp8(uint2(fq[b + 1u], fq[b + 1u]), xb1, a);
            v8f c = wg_wmma_fp8(xb0, uint2(fv[b], fv[b]), zero8());
            c = wg_wmma_fp8(xb1, uint2(fv[b + 1u], fv[b + 1u]), c);
            qvt[f0] = a;
            qvt[2u + f0] = c;
        }
        uint2 q4[4];
        RR_Q8(4u, qvt, q4);
        [unroll] for (uint i7 = 0; i7 < 4; ++i7) qv[4u * h0 + i7] = q4[i7];
    }
    // slot ((head 4 + wave 4) + f) 32 + lp of wave32 wave wv0 + t: this lane's dword of tile t's 8-byte slot
    [unroll] for (uint s0 = 0; s0 < 8; ++s0) {
        const uint rb = ((s0 >> 2) * 4u + wv0) * 4u + (s0 & 3u);
        rr_xq32[rr_xo + 64u * rb] = qv[s0].x;
        rr_xq32[rr_xo + 64u * (rb + 4u)] = qv[s0].y;
    }
    GroupMemoryBarrierWithGroupSync();

    // ---------------------------------------------------------------- steps 2-5 per head
    const half2 as2 = rr_c2(k_as), acl2 = rr_c2(k_acl), acln2 = rr_c2(k_acln), p12 = rr_c2(k_p1), p02 = rr_c2(k_p0);
    const half2 one2 = H2(0x3c003c00u);
    const bool tl1 = hh != 0u;
    v8f ot[4];    // O_h tiles before Q8: [2h + ft]
    [unroll] for (uint h = 0; h < 2; ++h) {
        uint kq[4][2], vt[4][2];
        v8f bc[4];
        [unroll] for (uint J = 0; J < 4; ++J) {
            const uint rr0 = (h * 4u + J) * 4u;
            kq[J][0] = rr_xqs(rr0); kq[J][1] = rr_xqs(rr0 + 1u);
            vt[J][0] = rr_xqs(rr0 + 2u); vt[J][1] = rr_xqs(rr0 + 3u);
            // the score bias, per tile (wave32 wave wv0 + t): 8 floats of the wave32 lane, this lane's half
            const uint tb0 = (((h * 4u + wv0) * 4u + J) * 32u) * 32u, tb1 = tb0 + 4u * 32u * 32u;   // wave wv0 + 1: 4 (J) x 32 lanes x 32 B on
            bc[J] = rr_cat4(wg_ld4f(p_tb + 4u * rr_lo, tb0), wg_ld4f(p_tb + 4u * rr_lo, tb1));
        }
        // scores (keys of tile J as rows, the wave's queries as columns), start value = the score bias, 51.3 step 2
        uint4 w4[4];
        v8f s4[4];
        [unroll] for (uint J7 = 0; J7 < 4; ++J7) s4[J7] = wg_wmma_fp8(uint2(kq[J7][0], kq[J7][0]), qv[4u * h], bc[J7]);
        [unroll] for (uint J8 = 0; J8 < 4; ++J8) s4[J8] = wg_wmma_fp8(uint2(kq[J8][1], kq[J8][1]), qv[4u * h + 1u], s4[J8]);
        [unroll] for (uint J9 = 0; J9 < 4; ++J9) {
            const uint4 sh = wg_to_f16_ags(s4[J9]);
            w4[J9] = uint4(rr_attw(sh.x, as2, acln2, acl2, p12, p02), rr_attw(sh.y, as2, acln2, acl2, p12, p02),
                           rr_attw(sh.z, as2, acln2, acl2, p12, p02), rr_attw(sh.w, as2, acln2, acl2, p12, p02));
        }
        // step 4: Z in f32 (the spec's accepted form), r = R(1 / Z), P = Q8(R(w r)). Wave32's chain over the lane's 32
        // weights (J, then elements 0-7) on this lane's own tile (hh 0: tile 0, hh 1: tile 1): the other half's words of
        // that tile come from lane ^ 32 (the lane sends the other tile's words of its half)
        float z = 0.0f;
        [unroll] for (uint J2 = 0; J2 < 4; ++J2) {
            const uint4 w = w4[J2];
            const uint sx2 = tl1 ? w.x : w.z, sy2 = tl1 ? w.y : w.w;
            const uint rx = wg_xor_u(sx2, 32u), ry = wg_xor_u(sy2, 32u);
            const uint a0 = tl1 ? rx : w.x, a1 = tl1 ? ry : w.y, a2 = tl1 ? w.z : rx, a3 = tl1 ? w.w : ry;
            z = dot2add(H2(a0), one2, z); z = dot2add(H2(a1), one2, z);
            z = dot2add(H2(a2), one2, z); z = dot2add(H2(a3), one2, z);
        }
        z = z + wg_xor16(z);
        const uint rh = rr_f2h(rcp(z));
        const uint rhx = wg_xor_u(rh, 32u);            // the other tile's R(1 / Z)
        const half2 r2a = H2((tl1 ? rhx : rh) * 0x10001u), r2b = H2((tl1 ? rh : rhx) * 0x10001u);   // tile 0, tile 1
        uint2 pf[4];
        [unroll] for (uint J3 = 0; J3 < 4; ++J3) {
            const uint4 pw = uint4(U2(H2(w4[J3].x) * r2a), U2(H2(w4[J3].y) * r2a), U2(H2(w4[J3].z) * r2b),
                                   U2(H2(w4[J3].w) * r2b));
            pf[J3] = rr_q8h_small(pw);   // P <= 1 + 2^-11: no saturation
        }
        // step 5: O_h = Q8(MM(P; v_h; 0)), keys 0-31 then 32-63
        [unroll] for (uint f1 = 0; f1 < 2; ++f1) {
            v8f a = wg_wmma_fp8(uint2(vt[0][f1], vt[0][f1]), pf[0], zero8());
            a = wg_wmma_fp8(uint2(vt[1][f1], vt[1][f1]), pf[1], a);
            a = wg_wmma_fp8(uint2(vt[2][f1], vt[2][f1]), pf[2], a);
            a = wg_wmma_fp8(uint2(vt[3][f1], vt[3][f1]), pf[3], a);
            ot[2u * h + f1] = a;
        }
    }
    uint2 of[4];
    RR_Q8(4u, ot, of);

    // ---------------------------------------------------------------- step 6: Dp, chain over the heads, start bo
    uint fo[8];
    [unroll] for (uint i1 = 0; i1 < 8; ++i1) fo[i1] = wfrag1(p_wf, F_O + i1);
    v8f dpf[2];
    [unroll] for (uint c0 = 0; c0 < 2; ++c0) {
        v8f a = bias_tile(p_bias, 64u * c0);
        a = wg_wmma_fp8(uint2(fo[c0 * 2u], fo[c0 * 2u]), of[0], a);
        a = wg_wmma_fp8(uint2(fo[c0 * 2u + 1u], fo[c0 * 2u + 1u]), of[1], a);
        a = wg_wmma_fp8(uint2(fo[4u + c0 * 2u], fo[4u + c0 * 2u]), of[2], a);
        a = wg_wmma_fp8(uint2(fo[4u + c0 * 2u + 1u], fo[4u + c0 * 2u + 1u]), of[3], a);
        dpf[c0] = a;
    }
    // ---------------------------------------------------------------- step 7: y1 = Dp ra + d8(x), y1q
    uint2 y1q[2];
    v8f y1f[2];
    {
        const float ra = f16tof32(k_ra & 0xffffu);
        const v8f x0 = rr_d8(xb0), x1 = rr_d8(xb1);   // exact
        [unroll] for (uint i5 = 0; i5 < 8; ++i5) {
            y1f[0].v[i5] = mad(dpf[0].v[i5], ra, x0.v[i5]);
            y1f[1].v[i5] = mad(dpf[1].v[i5], ra, x1.v[i5]);
        }
        RR_Q8(2u, y1f, y1q);
    }
    // ---------------------------------------------------------------- step 8: MLP in 4 chunks of 32
    const half2 cl2 = rr_c2(k_actcl), cln2 = rr_c2(k_actcln), c12 = rr_c2(k_c1), c22 = rr_c2(k_c2), c32 = rr_c2(k_c3);
    v8f ddf[2];
    uint w1[4], w2[4];
    [unroll] for (uint i2 = 0; i2 < 4; ++i2) { w1[i2] = wfrag1(p_wf, F_1 + i2); w2[i2] = wfrag1(p_wf, F_2 + (i2 >> 1) * 8u + (i2 & 1u)); }
    [unroll] for (uint j = 0; j < 4; ++j) {
        uint n1[4] = { 0u, 0u, 0u, 0u }, n2[4] = n1;
        if (j < 3u) {
            [unroll] for (uint i3 = 0; i3 < 4; ++i3) {
                n1[i3] = wfrag1(p_wf, F_1 + 4u * (j + 1u) + i3);
                n2[i3] = wfrag1(p_wf, F_2 + (i3 >> 1) * 8u + 2u * (j + 1u) + (i3 & 1u));
            }
        }
        v8f du[2];
        uint4 ah[2];
        uint4 hh2[2];
        uint2 au[2];
        uint abad = 0u;
        [unroll] for (uint k0 = 0; k0 < 2; ++k0) {
            const uint nt = 2u * j + k0;
            v8f a = bias_tile(p_bias, 128u + 64u * nt);
            a = wg_wmma_fp8(uint2(w1[2u * k0], w1[2u * k0]), y1q[0], a);
            a = wg_wmma_fp8(uint2(w1[2u * k0 + 1u], w1[2u * k0 + 1u]), y1q[1], a);
            const uint4 h = wg_to_f16_ags(a);
            hh2[k0] = h;
            ah[k0] = uint4(rr_act(h.x, cln2, cl2, c12, c22, c32), rr_act(h.y, cln2, cl2, c12, c22, c32),
                           rr_act(h.z, cln2, cl2, c12, c22, c32), rr_act(h.w, cln2, cl2, c12, c22, c32));
            au[k0] = rr_q8h(ah[k0], abad);
        }
        if (WaveActiveAnyTrue((abad & 0x80808080u) != 0u)) {
            // the same activation, recomputed from the kept f16 pre-activations (rare path)
            [unroll] for (uint k1 = 0; k1 < 2; ++k1) {
                const uint4 h1 = rr_fence4(hh2[k1]);
                ah[k1] = uint4(rr_act(h1.x, cln2, cl2, c12, c22, c32), rr_act(h1.y, cln2, cl2, c12, c22, c32),
                               rr_act(h1.z, cln2, cl2, c12, c22, c32), rr_act(h1.w, cln2, cl2, c12, c22, c32));
            }
            du[0] = rr_widen(rr_fence4(ah[0]));
            du[1] = rr_widen(rr_fence4(ah[1]));
            RR_Q8(2u, du, au);
        }
        [unroll] for (uint c1 = 0; c1 < 2; ++c1) {
            v8f a;
            if (j == 0u) a = bias_tile(p_bias, 640u + 64u * c1);
            else a = ddf[c1];
            a = wg_wmma_fp8(uint2(w2[2u * c1], w2[2u * c1]), au[0], a);
            a = wg_wmma_fp8(uint2(w2[2u * c1 + 1u], w2[2u * c1 + 1u]), au[1], a);
            ddf[c1] = a;
        }
        if (j < 3u) { [unroll] for (uint i4 = 0; i4 < 4; ++i4) { w1[i4] = n1[i4]; w2[i4] = n2[i4]; } }
    }
    // ---------------------------------------------------------------- step 9: y2h = R(Dd m + y1)
    {
        const float m = f16tof32(k_m & 0xffffu);
        [unroll] for (uint c3 = 0; c3 < 2; ++c3) {
            v8f t;
            [unroll] for (uint i6 = 0; i6 < 8; ++i6) t.v[i6] = mad(ddf[c3].v[i6], m, y1f[c3].v[i6]);
            y2h[c3] = wg_to_f16_ags(t);   // RNE, one rounding
        }
    }

    // ==================================================================================== the head (spec 55)
    // From here the lanes l = lp of each 32-lane half run rr_d0r.hlsl's text for the half's tile (hh): lane (r, hf)
    // holds y2h channels 8hf .. 8hf+7 (Y0) and 16+8hf .. of token r. A wave64 lane holds channels 8 hf + 4 hh .. + 3 of
    // both tiles: the half's other 4 channels of the lane's own tile come from lane ^ 32 (its tile-hh words), and the
    // word Y1.x (channels 16 + 8 hf, 17 + 8 hf: the hh = 0 lane's x word of y2h[1], tile 0, or z word, tile 1; the
    // only word of y2h[1] the tail reads, as WT) likewise.
    const uint hsx = tl1 ? y2h[0].x : y2h[0].z, hsy = tl1 ? y2h[0].y : y2h[0].w;   // the other tile's words of Y0
    const uint hrx = wg_xor_u(hsx, 32u), hry = wg_xor_u(hsy, 32u);
    const uint wsx = tl1 ? y2h[1].x : y2h[1].z;                                      // hh 0 sends its tile-1 word
    const uint wrx = wg_xor_u(wsx, 32u);
    const int Y = tl1 ? Yt[1] : Yt[0];
    const uint tok = tl1 ? tokt[1] : tokt[0];
    const uint4 Y0 = tl1 ? uint4(hrx, hry, y2h[0].z, y2h[0].w) : uint4(y2h[0].x, y2h[0].y, hrx, hry);
    const uint4 Y1 = uint4(tl1 ? wrx : y2h[1].x, 0u, 0u, 0u);
    // natural numbering: hf0 holds y0 .. y7, y16 .. y23; hf1 y8 .. y15, y24 .. y31. Two lane swaps:
    // hf0 WA = (y0, y1), YS = (y2, y3), W1 = (y8, y9) and W2 = (y10, -) from hf1, WT = y16;
    // hf1 WA = (y4, y5) and YS = (y6, y7) from hf0, W1 = y12, W2 = y14.
    const bool h0 = hf == 0u;
    const uint X1 = wg_xor16_u(h0 ? Y0.z : Y0.x), X2 = wg_xor16_u(h0 ? Y0.w : Y0.y);
    const uint WA = h0 ? Y0.x : X1, YS = h0 ? Y0.y : X2, W1 = h0 ? X1 : Y0.z, W2 = h0 ? X2 : Y0.w, WT = Y1.x;
    const uint ptab = rr_u(p_tab);
    // table lookups: hf0 sig(g0, g1, g2, g3), sig(py), tanh(hb), rho(rho); hf1 sig(G0, G1, G2, px), tanh(ha)
    const uint S0 = wg_ld16(ptab, (WA & 0xffffu) * 2u), S1 = wg_ld16(ptab, (WA >> 16) * 2u);
    const uint S2 = wg_ld16(ptab, (YS & 0xffffu) * 2u), S3 = wg_ld16(ptab, (YS >> 16) * 2u);
    const uint S4 = wg_ld16(ptab, h0 ? (W1 & 0xffffu) * 2u : TAB_TANH + (W2 & 0xffffu) * 2u);
    const uint T2 = wg_ld16(ptab, TAB_TANH + (WT & 0xffffu) * 2u);
    const float RHO = asfloat(wg_ld32(p_rho, (W1 >> 16) * 4u));
    // the two lanes of a token swap: hf0 -> (sig g2, sig g3), la; hf1 -> (sig px, lb)
    const uint Q0 = wg_xor16_u(h0 ? (S2 | (S3 << 16)) : (S3 | ((W1 & 0xffffu) << 16)));
    const uint Q1 = wg_xor16_u(W2 & 0xffffu);
    // 18-channel reading: hf0 holds (y10, y11) in W2 and (y16, y17) in WT, hf1 (y12, y13) in W1 and (y14, y15) in W2.
    // HID: hf0 values 2, 3 = tanh(y16), tanh(y17); hf1 values 0, 1 = tanh(y14), tanh(y15) (T3 = the high halves).
    // CANDW (55-ERRATUM-2): W4.r..a from (y10, y11, y12, y13), every step binary16 (each lane gets the other lane's
    // pair and computes all four): m = max; d = R(z - m); t = R(d R(log2 e)); e = R(2^t) (table, correctly rounded);
    // S = R(R(e0 + e2) + R(e1 + e3)); q = R(1/S) (table, correctly rounded); W = R(e q). hf0 stores W0, W1, hf1 W2, W3.
    const uint T3 = wg_ld16(ptab, TAB_TANH + ((h0 ? WT : W2) >> 16) * 2u);
    const uint P18 = wg_xor16_u(h0 ? W2 : W1);
    const half2 Z01 = H2(h0 ? W2 : P18), Z23 = H2(h0 ? P18 : W1);
    const half2 Zm = max(Z01, Z23);
    const half Zmax = max(Zm.x, Zm.y);
    const half2 Zmm = half2(Zmax, Zmax);
    const half2 L2E = H2(0x3dc53dc5u);                            // R(log2 e)
    // RR_HTRIM: subtract -> multiply (no fma form), and the product only feeds the table address (integer work)
    const uint T01 = U2(((Z01 - Zmm) * L2E)), T23 = U2(((Z23 - Zmm) * L2E));
    const uint E01 = wg_ld16(ptab, TAB_EXP2 + (T01 & 0xffffu) * 2u) | (wg_ld16(ptab, TAB_EXP2 + (T01 >> 16) * 2u) << 16);
    const uint E23 = wg_ld16(ptab, TAB_EXP2 + (T23 & 0xffffu) * 2u) | (wg_ld16(ptab, TAB_EXP2 + (T23 >> 16) * 2u) << 16);
    // RR_HTRIM: an add feeding an add (no fma form), and SS only feeds the table address
    const uint SP = U2((H2(E01) + H2(E23)));                    // (R(e0 + e2), R(e1 + e3))
    const uint SS = U2((H2(SP) + H2((SP >> 16) | (SP << 16))));  // R(R(e0 + e2) + R(e1 + e3)) in both halves
    const half2 Q18 = H2(wg_ld16(ptab, TAB_RCP16 + (SS & 0xffffu) * 2u) * 0x10001u);
    const uint WL = U2(HF(H2(h0 ? E01 : E23) * Q18));
    const uint cw18a = WL & 0xffffu, cw18b = WL >> 16;

    // ---------------------------------------------------------------- 55.2 Cnet (hf0: channels 0, 1; hf1: 2, 3)
    const half2 g2 = rr_c2(k_g);
    const uint zs = h0 ? S3 : (Q0 >> 16);                         // sig(g3) = z
    const half2 z2 = H2(zs * 0x10001u);
    const half2 omz = (H2(0x3c003c00u) - z2);                  // R(1 - z) (RR_HTRIM: a multiplicand of the fma below)
    const half2 gate = HF(H2(h0 ? (S0 | (S1 << 16)) : Q0) * g2);   // R(sig(Dh_j) g): hf0 j = 0, 1; hf1 j = 2 (low)
    const int Mx = (int)(Wt >> 1), My = (int)(Ht >> 1);
    const int xm = (X - 1) >> 1, ym = (Y - 1) >> 1;               // floor((X - 1/2) / 2)
    const uint rxm = (uint)rr_refl(xm, Mx), rxp = (uint)rr_refl(xm + 1, Mx);
    const uint rym = (uint)rr_refl(ym, My), ryp = (uint)rr_refl(ym + 1, My);
    const uint phc = rr_u(p_hc), cb = 4u * hf;
    const uint cA = wg_ld32(phc, (rym * (uint)Mx + rxp) * 8u + cb), cB = wg_ld32(phc, (rym * (uint)Mx + rxm) * 8u + cb);
    const uint cC = wg_ld32(phc, (ryp * (uint)Mx + rxm) * 8u + cb), cD = wg_ld32(phc, (ryp * (uint)Mx + rxp) * 8u + cb);
    const uint hold = wg_ld32(p_hold, tok * 8u + cb);
    const half2 wxp = H2((X & 1) != 0 ? 0x34003400u : 0x3a003a00u), wxm = H2((X & 1) != 0 ? 0x3a003a00u : 0x34003400u);
    const half2 wyp = H2((Y & 1) != 0 ? 0x34003400u : 0x3a003a00u), wym = H2((Y & 1) != 0 ? 0x3a003a00u : 0x34003400u);
    half2 bt = HF(HF(H2(cA) * wxp) * wym);
    // RR_HTRIM: an fma result feeds the next fma's addend or a multiply (no merge); hh only feeds integer work (the
    // select below) before it is the fma's addend
    bt = (mad(HF(H2(cB) * wxm), wym, bt));
    bt = (mad(HF(H2(cC) * wxm), wyp, bt));
    const half2 uu = (mad(HF(H2(cD) * wxp), wyp, bt));
    uint hhn = U2((HF(uu * gate) * z2));
    hhn = h0 ? hhn : ((hhn & 0xffffu) | 0x80000000u);                 // hnew_3 = R(hold_3 omz): addend -0
    const uint hn = U2(mad(H2(hold), omz, H2(hhn)));

    // ---------------------------------------------------------------- 55.3 PREC, CANDW values (f32, then R)
    const float s1 = asfloat(k_s1);
    float cv = 0.0f;
    [branch] if (!hmode) {
        const uint la = h0 ? (W2 & 0xffffu) : Q1, lb = h0 ? (Q0 >> 16) : (W1 & 0xffffu);
        const float d = f16tof32(la) - f16tof32(lb);                   // f32(la - lb), one rounding
        cv = rr_lhalf(h0 ? d : -d);                        // hf0: w = sigx(d)/2, hf1: w' = sigx(-d)/2
    }
    const float po = FF(f16tof32(h0 ? S4 : S3) * s1);              // hf0: Py, hf1: Px
    const float sq = po * po;                                      // hf0: PREC 1, hf1: PREC 0
    const float px = FF(f16tof32(Q0 & 0xffffu) * s1);              // hf0: Px
    const float p2 = FF(px * po) * RHO;                            // hf0: PREC 2 = (Px Py) rho
    v8f cf = zero8();
    cf.v[0] = cv; cf.v[1] = sq; cf.v[2] = p2;
    const uint4 c16 = wg_to_f16_ags(cf);
    const uint cv16 = c16.x & 0xffffu, sq16 = c16.x >> 16, p216 = c16.y & 0xffffu;

    // ---------------------------------------------------------------- stores
    const bool inT = X >= 0 && Y >= 0 && X < (int)Wt && Y < (int)Ht;
    const bool inI = X >= 0 && Y >= 0 && X < (int)Wr && Y < (int)Hr;
    const uint tt = (uint)Y * Wt + (uint)X, ti = (uint)Y * Wr + (uint)X;
    const uint hv = h0 ? T2 : S4;                                  // hf0: R(tanh hb) (values 2, 3), hf1: R(tanh ha)
    const uint hv2 = hmode ? T3 : hv;                              // second value of the lane's HID pair
    const uint cwa = hmode ? cw18a : cv16, cwb = hmode ? cw18b : cv16;   // the lane's two CANDW values
    // the 8-bit codes once, shared by the stores and the render record (same function of the same values)
    const uint u8a = rr_u8(cwa), u8b = rr_u8(cwb), u80 = rr_u8(S0), u81 = rr_u8(S1), u82 = rr_u8(S2);
    // each record whole from the hf0 lane: the hf1 lane's record parts reach it (xor 16); b64 / b32 / b32 / b64 / b32 stores
    {
        const uint cw16 = (u8a | (u8b << 8)) & 0xffffu, hd16 = (rr_s8(hv) | (rr_s8(hv2) << 8)) & 0xffffu;
        const uint gt = u80 | (u81 << 8) | (u82 << 16);
        const uint hnx = wg_xor16_u(hn), cwx = wg_xor16_u(cw16), hdx = wg_xor16_u(hd16), sqx = wg_xor16_u(sq16), gtx = wg_xor16_u(gt);
        if (h0) {
            if (inT) wg_st64(rr_u(p_cnet), tt * 8u, uint2(hn, hnx));
            if (inI) {
                wg_st32(rr_u(p_candw), ti * 4u, cw16 | (cwx << 16));
                wg_st32(rr_u(p_hid), ti * 4u, hdx | (hd16 << 16));
                wg_st64(rr_u(p_prec), ti * 8u, uint2((sqx & 0xffffu) | (sq16 << 16), p216));
                wg_st32(rr_u(p_gates), ti * 4u, gtx);
            }
        }
    }
    // the render record of token ti (rr_output_prep's a / b vectors): hf0 holds Cnet 0, 1, PREC 1 (Py), PREC 2, CANDW
    // bytes 0, 1; hf1 Cnet 2, PREC 0 (Px), GATES, CANDW bytes 2, 3 (the stores above)
    [branch] if (p_oscr != 0u) {
        bool orare = false;
        v8f e = zero8();
        e.v[0] = rr_ocf(f16tof32(hn & 0xffffu));
        e.v[1] = rr_ocf(f16tof32(hn >> 16));
        e.v[2] = rr_omq(f16tof32(sq16), asfloat(k_o2n), orare);
        e.v[3] = h0 ? rr_omq(f16tof32(p216), asfloat(k_o2x2n), orare) : rr_od255(u8b);   // hf1: W half 3
        e.v[4] = rr_od255(u8a);
        e.v[5] = rr_od255(h0 ? u8b : u80);                                                       // hf0: W half 1
        e.v[6] = rr_od255(u81); e.v[7] = rr_od255(u82);
        const uint4 o16 = wg_to_f16_ags(e);
        if (inI) {
            const uint opw = (o_wh + 1u) / 2u, olb = 256u + opw * ((o_hh + 1u) / 2u) * 4u;
            const uint ra = p_oscr + ((olb + 255u) & ~255u), rb = ra + Wr * Hr * 16u, rw = rb + ((Wr * Hr * 4u + 15u) & ~15u);
            const uint w2 = (o16.z & 0xffffu) | ((h0 ? (o16.z >> 16) : (o16.y >> 16)) << 16);   // W halves (a, b)
            {
                const uint a1 = (o16.x & 0xffffu) | (o16.z & 0xffff0000u), a2 = o16.w, a3 = o16.y & 0xffffu;   // hf1: A 2-3, 4-5, 6
                const uint a1x = wg_xor16_u(a1), a2x = wg_xor16_u(a2), a3x = wg_xor16_u(a3), w2x = wg_xor16_u(w2);
                if (h0) {
                    wg_st128(rr_u(ra), ti * 16u, uint4(o16.x, a1x, a2x, a3x | (o16.y << 16)));   // A halves 0-7
                    wg_st32(rr_u(rb), ti * 4u, o16.y >> 16);                                    // B
                    wg_st64(rr_u(rw), ti * 8u, uint2(w2, w2x));                                 // W halves 0-3
                }
            }
        }
        if (WaveActiveAnyTrue(orare) && WaveIsFirstLane()) { uint o; wg_mem_gc.InterlockedOr(rr_u(p_oscr) + 4u, 1u, o); }
    }
}
