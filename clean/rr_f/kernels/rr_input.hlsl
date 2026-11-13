// SPDX-License-Identifier: MIT
// Spec 52 input stage on RDNA 4 / D3D12: the 64 front-end inputs (52.3), their ordered-dither e4m3 codes (52.4),
// the state-stream seed (52.6), the second vector m2 with the specular-motion reconstruction (52.7.1, 52.7.2) and
// the three motion surfaces (52.7). One lane per tensor token (row-major [Ht][Wt]).
// All per-frame values (52.1.1, 52.8, 52.7.3 maps), symbols, texture descriptors and buffer offsets are in one
// constant block (gpu/input_stage.py, layout K_* below). Textures are f32 [H][W][C] after the API read conversion;
// PH1 / PH2 unpacked RGB10A2 codes u16 [h][w][4]; HID snorm8 [Hr][Wr][4]; PM / Pe / Pn f16.
// Operation order follows reference/input_stage.py one operation at a time (rr_in_math.hlsli).
// a14 C19 (in-wave64-features): RR_IN_W64 1 builds the kernel in wave64 (kernel-side: the build's WG_WAVE is replaced).
// Default: the certified fast pass / split / lite features (RR_IN_EXACT 0 at 128 lanes per workgroup); the replay,
// decode, clear and the test kernels stay wave32. Every per-token value is computed alone and every wave-uniform
// branch (tap / hidden dedup, certification fallbacks) takes the same value either way, so the bits are unchanged.
#ifndef RR_IN_W64
#if defined(RR_IN_EXACT) && defined(WG_BLOCK)
#if RR_IN_EXACT == 0 && WG_BLOCK == 128 && !defined(RR_IN_REPLAY)
#define RR_IN_W64 1
#endif
#endif
#endif
#ifndef RR_IN_W64
#define RR_IN_W64 0
#endif
#if RR_IN_W64
#undef WG_WAVE
#define WG_WAVE 64
#endif
#include "wg_platform.hlsli"
#if WG_WAVE == 64
#include "wg_ags.hlsli"      // a14 C19: the per-value driver conversions in a wave64 build
#endif
WG_ARGS_BEGIN
    uint pK;
WG_ARGS_END
#ifndef RR_IN_REPLAY
#define RR_IN_REPLAY 0
#endif
#ifndef RR_IN_RSPREAD
#define RR_IN_RSPREAD 0      // replay (clean/rr-f-s9-inx): list entry i -> wave i % waves, lane i / waves (the listed tokens
#endif                       // spread over the grid's waves instead of filling each wave's 32 lanes; same per-token work)
#ifndef RR_IN_SKIP
#define RR_IN_SKIP 0         // timing experiments only (gpu/input_bench.py --skip): parts of the token left out
#endif
#ifndef RR_IN_MARK_ALL
#define RR_IN_MARK_ALL 0
#endif
#ifndef RR_IN_CDIAG
#define RR_IN_CDIAG 0        // diagnostics: DIAG word 3 counts second-level certifications (low 16) and failures (high)
#endif
#ifndef RR_IN_HCERT_MUT
#define RR_IN_HCERT_MUT 0    // negative control of the certification (test builds only)
#endif
#ifndef RR_IN_L2CERT
#define RR_IN_L2CERT 1       // fast pass: colour, guide and motion-difference features from the certified log2 (l2c)
#endif
#ifndef RR_IN_AGSH
#define RR_IN_AGSH 1         // fast pass: quant's binary16 rounding through the driver conversion (QF2H)
#endif
#ifndef RR_IN_HIDCERT
#define RR_IN_HIDCERT 0      // timing variant: hidden features from a certified approximation (measured slower)
#endif
#ifndef RR_IN_HFB
#define RR_IN_HFB 1          // 1: an uncertified history value is recomputed in place; 0: its token goes to the replay (a14 C1: 1)
#endif
#ifndef RR_IN_NHL
#define RR_IN_NHL 0          // split input (clean/rr-f-s9-inx, fast pass only): non-history features with input lite's maths
#endif
#ifndef RR_IN_HCERT
#define RR_IN_HCERT 1        // fast pass: certified codes of the history features (hist_code); 0: the reference operations
#endif
#ifndef RR_IN_DBG
// a14 C31 (in-nodbg): 0 compiles the debug / diagnostic machinery out (the per-put debug store, the approximation bits,
// the K_DIAG atomics; the runtime's words 224 / 225 are 0, so the same codes / hold / marks / surfaces); the harness
// (gpu/input_stage.py) builds 1
#define RR_IN_DBG 0
#endif
#ifndef RR_IN_IVSKIP
#define RR_IN_IVSKIP 0       // a14 C44: wave-uniform skip of hist_code when every lane's tap is invalid
#endif
#ifndef RR_IN_DLANE
#define RR_IN_DLANE 0        // a14 C8: the 3 x 3 depth window from the neighbour lanes' columns
#endif
#ifndef RR_IN_TILE2D
#define RR_IN_TILE2D 0       // a14 C16: 2-D token tiles per workgroup (fast pass)
#endif
#ifndef RR_IN_MDD
#define RR_IN_MDD 0          // a14 C41: the two motion differences share u, v, the bilinear setup and one 8 B PM load
#endif                       // when mN and m2 have the same bits in every lane of the wave
#ifndef RR_IN_DEDUP
#define RR_IN_DEDUP ((RR_IN_SKIP & 64) == 0)   // clean/rr-f-lw-zeros: Hc1 taps / hid(m2) at the same position computed once
#endif
// a14 C33 (in-addfma): the fence diet's last forms, here and not in rr_in_math.hlsli (that header is part of the head
// kernels' cache key): A / S as fma(a, opaque 1, +-b) (a 1 is a exactly, so RN(a 1 + b) = RN(a + b) under the same flush:
// one VALU instead of add + xor); FM without its result fence in the fast pass (a fused result cannot fuse again)
#ifndef RR_IN_AFMA
#define RR_IN_AFMA 1
#endif
#ifndef RR_IN_FMNF
#define RR_IN_FMNF 1
#endif
#ifndef RR_IN_EXACT
#define RR_IN_EXACT 1        // rr_in_math.hlsli's default, needed here for the renames below (decode / clear / tests)
#endif
#if RR_IN_AFMA
#define A in_A_fenced
#define S in_S_fenced
#endif
#if RR_IN_FMNF && !RR_IN_EXACT
#define FM in_FM_fenced
#endif
#ifndef RR_IN_RCP1
#define RR_IN_RCP1 1         // a14 C12: RCP by one Newton step from the hardware reciprocal (output Rf's form), no scaling
#endif                       // branches (proof: --xcheck rcp over every normal f32 with this define)
#if RR_IN_RCP1 && !(RR_IN_FAST & 2)
#define RCP in_RCP_branchy
#endif
#include "rr_in_math.hlsli"
#undef A
#undef S
#undef FM
#undef RCP
#if RR_IN_RCP1 && !(RR_IN_FAST & 2)
// the correctly rounded 1 / x of rr_in_math.hlsli's RCP: for 2^-126 <= |x| < 2^126 one Newton step from the hardware
// reciprocal (the bits of gpu/output_probe.py R_one_step, here checked against the binary64 quotient for every normal
// f32: input_stage --xcheck rcp); |x| = 2^126: mn; |x| > 2^126 or inf: signed 0; 0 / subnormal: signed inf; NaN: x
float RCP(float x)
{
    const float y = rcp(x);
    const float q = mad(mad(-x, y, 1.0f), y, y);
    const uint ax = asuint(x) & 0x7fffffffu, sg = asuint(x) & 0x80000000u;
    uint sp = ax < 0x00800000u ? (sg | 0x7f800000u) : (ax > 0x7f800000u ? asuint(x) : sg);
    sp = ax == 0x7e800000u ? sg | 0x00800000u : sp;
    return asfloat(and(ax >= 0x00800000u, ax < 0x7e800000u) ? asuint(q) : sp);
}
#endif
float in_one() { return asfloat(0x3f800000u ^ in_zb()); }
float in_nz() { return asfloat(0x80000000u ^ in_zb()); }
#if RR_IN_AFMA
#if RR_IN_FAST & 16
float A(float a, float b) { return a + b; }
float S(float a, float b) { return a - b; }
#else
float A(float a, float b) { return mad(a, in_one(), b); }
float S(float a, float b) { return mad(a, in_one(), -b); }
#endif
#endif
#if RR_IN_FMNF && !RR_IN_EXACT
float FM(float a, float b, float c)
{
    float r = mad(a, b, c);
#if RR_IN_FAST & 8
    return r;
#endif
#if !RR_IN_LITE
#if RR_IN_FMBR
    [branch] if (IN_ZT(r)) {
        const float q = FN(a * b);
        const float qe = FN(mad(a, b, -q));
        const bool exactzero = select(or(a == 0.0f, b == 0.0f), c == 0.0f,
                                      and(and(q != 0.0f, q == -c), qe == 0.0f));
        in_flag |= select(exactzero, 0u, 1u);
    }
#else
    const float q = FN(a * b);
    const float qe = FN(mad(a, b, -q));
    const bool exactzero = select(or(a == 0.0f, b == 0.0f), c == 0.0f,
                                  and(and(q != 0.0f, q == -c), qe == 0.0f));
    in_flag |= select(and(IN_ZT(r), !exactzero), 1u, 0u);
#endif
#endif
    return r;
}
#endif

#define K_WR 0
#define K_HR 1
#define K_WT 2
#define K_HT 3
#define K_WH 4
#define K_HH 5
#define K_SX 6
#define K_SY 7
#define K_FC 8
#define K_FLAGS 9
#define K_E 10
#define K_X3 12
#define K_JX 13
#define K_JY 14
#define K_PJX 15
#define K_PJY 16
#define K_MOX 17
#define K_MOY 18
#define K_MSHX 19
#define K_MSHY 20
#define K_RWHWR 21
#define K_RHHHR 22
#define K_RWRWH 23
#define K_RHRHH 24
#define K_INVWR 25
#define K_INVHR 26
#define K_INVWH 27
#define K_INVHH 28
#define K_HCSX 29
#define K_HCSY 30
#define K_HOFFX 31
#define K_HOFFY 32
#define K_ZFAR 33
#define K_V22 34
#define K_V32 35
#define K_V23 36
#define K_RCPEP 37
#define K_MPOS 40
#define K_MREP 56
#define K_SYM 72
#define K_TEX 100
#define K_PH1 208
#define K_PH2 211
#define K_HID 214
#define K_PMR 215
#define K_PMW 216
#define K_PE 217
#define K_PN 218
#define K_D 219
#define K_UN10 220
#define K_SN8 221
#define K_CODES 222
#define K_HOLD 223
#define K_DEBUG 224
#define K_DIAG 225
#define K_EXPO 226          // nonzero: E, Ep, X3 from the exposure pass's result (rr_exposure.hlsl: E, Ep, invE, X3)
#define K_LOGT 227          // float-float log2 table [256] (c, Th, Tl, 0), gpu/ffmath.py
#define K_EXPT 228          // float-float exp2 table [65] (Eh, El)
#define K_MARK 229          // replay list: word 0 = count, then the token indices the fast pass marked for the exact
                            // replay (rr_input_mark_clear sets the count to 0 before the fast pass)

// symbols (K_SYM + index)
#define S_K4 0
#define S_K5 1
#define S_K7 2
#define S_K7N 3
#define S_K8 4
#define S_K10 5
#define S_K12 6
#define S_HDEC 7
#define S_FCLIP 8
#define S_AMAX 9
#define S_GCL 10
#define S_GCLN 11
#define S_LWRB 12
#define S_LWG 13
#define S_MDL 14
#define S_MDLN 15
#define S_GM 16
#define S_GD 17
#define S_GH 18
#define S_GS 19
#define S_SV1 20
#define S_SV2 21
#define S_SV3 22
#define S_SV4 23
#define S_SV4N 24

// roles
#define R_C 0
#define R_Z 1
#define R_V 2
#define R_SV 3
#define R_DD 4
#define R_DS 5
#define R_N 6
#define R_R 7
#define R_Q 8
#define R_B 9
#define R_G1 10
#define R_G2 11
#define R_HD 12

#define F_RESET 1u
#define F_REV 2u
#define F_HWD 4u
#define F_NRMDEC 8u
#define F_RGHPK 16u
#define F_B2 128u
#define F_DIS 256u
#define F_B 512u
#define F_G1 1024u
#define F_G2 2048u
#define F_V33LT1 4096u
#define F_HDF 8192u

#define LOG10_2 asfloat(0x3e9a209bu)
#define LOG2_10 asfloat(0x40549a78u)
#define LN_2 asfloat(0x3f317218u)
#define F16MAX 65504.0f

// The features kernels read the 1 KB constant block from LDS (staged once per workgroup: arena loads are vector
// loads on this path, and dependent per-use loads dominated the kernel's time); the small kernels read memory (KD).
#if !(defined(RR_IN_KCB) && RR_IN_KCB)
groupshared uint gs_k[256];
#endif
uint KD(uint i) { return wg_ld32(pK, i * 4u); }
// RR_IN_UKW 1 (clean/rr-f-lw-fold; default in the certified fast pass and the replay, not in input lite): every word of
// the block (and the exposure words E, Ep, X3 read from memory) is the same in every lane, so it is read as wave-uniform
// (v_readfirstlane: the same value in a scalar register). Values and operations are unchanged; the per-frame values
// and what is derived from them (hc_setup, RCP(Ep), the texture descriptors, the symbols) no longer take vector
// registers (features pass 105 -> 80 VGPRs, 13 -> 14 waves). Measured (clock-normalised, 1080p / 1440p Q): features pass
// -0.013 / -0.030 ms; input lite was slower with it (+0.012 / +0.008 ms), so lite keeps the vector form.
#ifndef RR_IN_UKW
#define RR_IN_UKW (!RR_IN_LITE)
#endif
#ifndef RR_IN_KCB
#define RR_IN_KCB 0
#endif
#if RR_IN_KCB
// clean/rr-mp1-input I1 (RR_IN_KCB 1, the WrongGPU RR runtime only): the same 256 words read through scalar memory
// from the runtime's root SRV 6 (t0 space 9), which the runtime rebinds to this frame's input block in the upload ring
// (ring slot + kBlkInput, or + kBlkInputDiag for WG_RR_DIAG_INX) around the features / replay dispatches. The arena
// block at pK is a byte copy of that slot (no GPU write touches its first 1 KB), so KW(i) returns the same word: no
// LDS staging, no barrier, no v_readfirstlane. The ring slot is not rewritten before the frame that used it completed
// (rr_runtime.cpp: wait_serial(slot_serial[slot]) before the CPU fills the slot).
ByteAddressBuffer rr_kring : register(t0, space9);
uint KW(uint i) { return rr_kring.Load(i * 4u); }
#if RR_IN_UKW
uint KU(uint v) { return WaveReadLaneFirst(v); }
#else
uint KU(uint v) { return v; }
#endif
void in_stage_block() {}
#else
#if RR_IN_UKW
uint KW(uint i) { return WaveReadLaneFirst(gs_k[i]); }
uint KU(uint v) { return WaveReadLaneFirst(v); }
#else
uint KW(uint i) { return gs_k[i]; }
uint KU(uint v) { return v; }
#endif
void in_stage_block()                    // the first 64 lanes x 16 bytes = the 256 words (WG_BLOCK >= 64)
{
    if (wg_lane_id() < 64u) {
        const uint4 v = wg_ld128(pK, wg_lane_id() * 16u);
        gs_k[4u * wg_lane_id()] = v.x; gs_k[4u * wg_lane_id() + 1u] = v.y;
        gs_k[4u * wg_lane_id() + 2u] = v.z; gs_k[4u * wg_lane_id() + 3u] = v.w;
    }
    GroupMemoryBarrierWithGroupSync();
}
#endif
void wg_atomic_or(uint base, uint off, uint v) { uint o; wg_mem_gc.InterlockedOr(base + off, v, o); }
float KF(uint i) { return asfloat(KW(i)); }
float SY(uint i) { return KF(K_SYM + i); }

// RR_IN_SRV=1 (opt-in; WrongGPU's RR runtime, clean/rr_f/runtime): a texture entry whose word 7 is nonzero is read
// from the game's resource itself (SRV rr_in_srv[word 7 - 1], the API read conversion of 52.1.2) instead of an f32
// copy in the arena; the same values as the copy (the copy is made by the same SRV load). Default: unchanged.
#ifndef RR_IN_SRV
#define RR_IN_SRV 0
#endif
#if RR_IN_SRV
Texture2D<float4> rr_in_srv[16] : register(t0, space1);
#endif
// RR_IN_SRVC (clean/rr-f-r6-input): the runtime puts role r's SRV at array index r (word 7 = r + 1 whenever nonzero,
// rr_runtime.cpp), so the descriptor index is the compile-time role. Since clean/release-060 every RR_IN_SRV kernel
// reads that way (tx below); the define is still accepted and has no effect.
#ifndef RR_IN_SRVC
#define RR_IN_SRVC 0
#endif
struct Tex { uint off, w, h, ch; int ox, oy; uint srv, r; };
Tex tex(uint r)
{
    const uint b = K_TEX + 8u * r;
    Tex t;
    t.off = KW(b); t.w = KW(b + 1u); t.h = KW(b + 2u); t.ch = KW(b + 3u);
    t.ox = (int)KW(b + 4u); t.oy = (int)KW(b + 5u);
    t.r = r;
#if RR_IN_SRV
    t.srv = KW(b + 7u);
#else
    t.srv = 0u;
#endif
    return t;
}
float tx(Tex t, int x, int y, uint c)
{
#if RR_IN_SRV
    // clean/release-060 crash-cp2077: always the compile-time role's descriptor (every caller passes a constant role, and
    // the runtime's word 7 is role + 1 whenever nonzero: the same descriptor). The index read from word 7 (the old
    // default without RR_IN_SRVC, in input lite and the replay) was the only descriptor index chosen at run time;
    // Cyberpunk 2077 hung in input.features.lite with page faults at random addresses while every descriptor and
    // constant word in memory was intact. A constant index cannot address anything but the 16-entry table.
    if (t.srv != 0u) return in_ftz(rr_in_srv[t.r].Load(int3(x + t.ox, y + t.oy, 0))[c]);
#endif
    const uint idx = ((uint)(y + t.oy) * t.w + (uint)(x + t.ox)) * t.ch + c;
    return in_ftz(asfloat(wg_ld32(t.off, idx * 4u)));
}

// ---- per-frame values (uniform) --------------------------------------------------------------------------------
static int Wr, Hr, Wt, Ht;
static float Whf, Hhf;
static uint flags;
static bool reset;
static float rcpEp;
static float E, jx, jy, pjx, pjy, rWhWr, rHhHr, rWrWh, rHrHh, invWr, invHr;

// ---- 52.1.4 reference sampler -----------------------------------------------------------------------------------
struct Bil { uint i0, i1, j0, j1; float fx, fy; };
#ifndef RR_IN_M24
#define RR_IN_M24 0          // a14 C14: 24-bit texel index arithmetic (v_mad_u32_u24)
#endif
#if RR_IN_M24
#define IN_W24(w) ((w) & 0xffffffu)
#else
#define IN_W24(w) (w)
#endif
// The scaling products of bil_setup never round a nonzero product to zero (fast pass: no zero-result test): u and v are
// clamped to [0, 1] after the flush of the arithmetic that made them (0 or >= 2^-126), the sizes are integers >= 1, so a
// nonzero product is >= 2^-126 (the same value as M's).
#if RR_IN_EXACT
#define BW M
#else
#if RR_IN_AFMA && !(RR_IN_FAST & 16)
float BW(float a, float b) { return mad(a, b, in_nz()); }   // a14 C33: RN(a b + -0) = RN(a b)
#else
float BW(float a, float b) { return FN(a * b); }
#endif
#endif
Bil bil_setup(float u, float v, uint Wi, uint Hi)
{
    u = min(max(in_nan0(u), 0.0f), 1.0f);
    v = min(max(in_nan0(v), 0.0f), 1.0f);
    const float s = S(BW(u, (float)Wi), 0.5f), t = S(BW(v, (float)Hi), 0.5f);
    const float fs = floor(s), ft = floor(t);
    Bil b;
    b.fx = S(s, fs);
    b.fy = S(t, ft);
    const int i0 = (int)fs, j0 = (int)ft;
#if RR_IN_M24
    // a14 C14: the same clamp with the (uniform) upper ends limited to 2^24 - 1 (image sizes are far below), so the
    // indices are known 24-bit values and the texel addresses take v_mad_u32_u24 instead of v_mul_lo_u32 + add
    const uint wm = min(Wi - 1u, 0xffffffu), hm = min(Hi - 1u, 0xffffffu);
    b.i0 = min((uint)max(i0, 0), wm); b.i1 = min((uint)max(i0 + 1, 0), wm);
    b.j0 = min((uint)max(j0, 0), hm); b.j1 = min((uint)max(j0 + 1, 0), hm);
#else
    b.i0 = (uint)clamp(i0, 0, (int)Wi - 1); b.i1 = (uint)clamp(i0 + 1, 0, (int)Wi - 1);
    b.j0 = (uint)clamp(j0, 0, (int)Hi - 1); b.j1 = (uint)clamp(j0 + 1, 0, (int)Hi - 1);
#endif
    return b;
}
// The products of bmix never round a nonzero product to zero (so the fast pass needs no zero-result test there): the
// weights fx, 1 - fx are 0 or in [2^-25, 1] (s = M(u, W) - 1/2 is a multiple of 2^-25), the texels are decoded unorm10
// (0 or >= 1/1023), snorm8 (0 or >= 1/127) or binary16 (0, >= 2^-24, inf, NaN) values, so a nonzero product is
// >= 2^-49, a nonzero r0 / r1 >= 2^-72 (a multiple of the smaller ulp) and the outer products >= 2^-97.
#if RR_IN_EXACT
#define MW M
#else
#if RR_IN_AFMA && !(RR_IN_FAST & 16)
float MW(float a, float b) { return mad(a, b, in_nz()); }   // a14 C33: RN(a b + -0) = RN(a b)
#else
float MW(float a, float b) { return FN(a * b); }
#endif
#endif
float bmix(float t00, float t10, float t01, float t11, Bil b)
{
    const float ofx = S(1.0f, b.fx), ofy = S(1.0f, b.fy);
    const float r0 = A(MW(ofx, t00), MW(b.fx, t10));
    const float r1 = A(MW(ofx, t01), MW(b.fx, t11));
    return A(MW(ofy, r0), MW(b.fy, r1));
}
float4 bmix4(float4 t00, float4 t10, float4 t01, float4 t11, Bil b)
{
    return float4(bmix(t00.x, t10.x, t01.x, t11.x, b), bmix(t00.y, t10.y, t01.y, t11.y, b),
                  bmix(t00.z, t10.z, t01.z, t11.z, b), bmix(t00.w, t10.w, t01.w, t11.w, b));
}
// PH1 / PH2 / HID are read from their decoded f32 copies (rr_input_decode, once per frame: the same values as the
// per-texel conversion of 52.1.4 step 3, made once instead of at every tap)
float3 ph_texel(uint off, uint w, uint i, uint j)
{
    return asfloat(wg_ld128(off, (j * IN_W24(w) + i) * 16u).xyz);
}
float3 ph_sample(uint which, float u, float v)
{
    const uint b = which == 0u ? K_PH1 : K_PH2;
    const uint off = KW(b), w = KW(b + 1u), h = KW(b + 2u);
    const Bil q = bil_setup(u, v, w, h);
    const float3 a = ph_texel(off, w, q.i0, q.j0), bb = ph_texel(off, w, q.i1, q.j0);
    const float3 c = ph_texel(off, w, q.i0, q.j1), d = ph_texel(off, w, q.i1, q.j1);
    return float3(bmix(a.x, bb.x, c.x, d.x, q), bmix(a.y, bb.y, c.y, d.y, q), bmix(a.z, bb.z, c.z, d.z, q));
}
float4 hid_texel(uint i, uint j)
{
    return asfloat(wg_ld128(KW(K_HID), (j * IN_W24((uint)Wr) + i) * 16u));
}
float4 pm_texel(uint i, uint j)
{
    const uint2 v = wg_ld64(KW(K_PMR), (j * IN_W24((uint)Wr) + i) * 8u);
    return float4(f16tof32(v.x & 0xffffu), f16tof32(v.x >> 16), f16tof32(v.y & 0xffffu), f16tof32(v.y >> 16));
}
float2 pm_texel2(uint i, uint j, uint half_)          // channels (0, 1) or (2, 3) of the PM texel
{
    const uint v = wg_ld32(KW(K_PMR), (j * IN_W24((uint)Wr) + i) * 8u + 4u * half_);
    return float2(f16tof32(v & 0xffffu), f16tof32(v >> 16));
}

bool cell_ok(float v, float D);
float l2c(float y, float c1, float c2, float c3, out float D);

// ---- 52.3 scalar rules -------------------------------------------------------------------------------------------
float cc(float v)
{
    const float u = min(in_nan0(M(E, v)), F16MAX);
    const float y = max(FM(u, SY(S_K4), 1.0f), SY(S_K5));
    return M(M(log2_cr(y), LOG10_2), SY(S_K7));
}
float ccf(float v)
{
    const float u = M(E, v);
    const float y = max(FM(u, SY(S_K4), 1.0f), SY(S_K5));
    return M(M(log2_cr(y), LOG10_2), SY(S_K7));
}
float slc(float v)
{
    const float t = M(log2_cr(FM(abs(v), SY(S_K4), 1.0f)), LOG10_2);
    const float r = in_nan0(M(select(v >= 0.0f, SY(S_K7), SY(S_K7N)), t));
    return min(max(r, SY(S_GCLN)), SY(S_GCL));
}
float clipf(float v) { return min(in_nan0(v), SY(S_FCLIP)); }
// slc with the certified log2 (fast pass): D bounds the distance of the result from slc(v) (in_nan0 and the clamp
// are 1-Lipschitz; a NaN or non-normal argument gives D = inf)
float slc_c(float v, out float D)
{
    const float r = l2c(FM(abs(v), SY(S_K4), 1.0f), LOG10_2, select(v >= 0.0f, SY(S_K7), SY(S_K7N)), 1.0f, D);
    D = isnan(r) ? asfloat(0x7f800000u) : D;
    return min(max(in_nan0(r), SY(S_GCLN)), SY(S_GCL));
}
float hdec(float h)
{
    const float h1 = M(h, SY(S_HDEC));
    const float hp = select(h1 > SY(S_K5), min(h1, SY(S_K10)), SY(S_K5));
    const float e = exp2_cr(M(M(hp, SY(S_K8)), LOG2_10));
    return M(M(S(e, 1.0f), SY(S_K12)), rcpEp);
}
#if RR_IN_NHL && !RR_IN_EXACT
// Split input (RR_IN_NHL 1; clean/rr-f-s9-inx, owner rule 2026-10-23/28): the NON-history features, colour (0, 1, 4) and
// guides (30 .. 32), with input lite's operations (gpu/input_stage.py variant "lite": plain hardware f32, hardware log2,
// no fences, no zero-result tests, no replay marks). Every history sample keeps the exact fast pass + replay: the
// second-vector reconstruction (and the normal it reads), the tap positions, the five history samples and their
// fallback colour, the hidden features, the motion differences and the motion surfaces. A token that a history part
// marks is replayed whole with the exact operations (its non-history features are then the exact ones).
float nl_cc(float v)
{
    const float u = min(in_nan0(E * v), F16MAX);
    const float y = max(mad(u, SY(S_K4), 1.0f), SY(S_K5));
    return (log2(y) * LOG10_2) * SY(S_K7);
}
float nl_slc(float v)
{
    const float t = log2(mad(abs(v), SY(S_K4), 1.0f)) * LOG10_2;
    const float r = in_nan0(select(v >= 0.0f, SY(S_K7), SY(S_K7N)) * t);
    return min(max(r, SY(S_GCLN)), SY(S_GCL));
}
#endif
bool outside(float pu, float pv) { return or(or(pu < 0.0f, pu > Whf), or(pv < 0.0f, pv > Hhf)); }

float4 hidf(float2 pos)
{
    if (reset) return float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 s = float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (!outside(pos.x, pos.y)) {
        const float uh = FM(pos.x, KF(K_INVWH), KF(K_HOFFX)), vh = FM(pos.y, KF(K_INVHH), KF(K_HOFFY));
        const Bil q = bil_setup(uh, vh, (uint)Wr, (uint)Hr);
        s = bmix4(hid_texel(q.i0, q.j0), hid_texel(q.i1, q.j0), hid_texel(q.i0, q.j1), hid_texel(q.i1, q.j1), q);
    }
    const float gs = SY(S_GS);
    return float4(M(gs, s.x), M(gs, s.y), M(gs, s.z), M(gs, s.w));
}

static bool md_approx = false;
float mdiff_core(Bil q, float2 t00, float2 t10, float2 t01, float2 t11, float mrx, float mry, float dq);
float mdiff(int X, int Y, float2 mh, bool second, float dq)
{
    md_approx = false;
    if (reset) return 0.0f;
    const float mrx = M(mh.x, rWrWh), mry = M(mh.y, rHrHh);
    const float cx = S(A((float)X, 0.5f), jx), cy = S(A((float)Y, 0.5f), jy);
    const float u = M(A(A(cx, pjx), mrx), invWr), v = M(A(A(cy, pjy), mry), invHr);
    const Bil q = bil_setup(u, v, (uint)Wr, (uint)Hr);
    const uint hs = second ? 1u : 0u;
    const float2 t00 = pm_texel2(q.i0, q.j0, hs), t10 = pm_texel2(q.i1, q.j0, hs);
    const float2 t01 = pm_texel2(q.i0, q.j1, hs), t11 = pm_texel2(q.i1, q.j1, hs);
    return mdiff_core(q, t00, t10, t01, t11, mrx, mry, dq);
}
// a14 C41: both motion differences at one vector mh (= mN = m2): the same u, v, bilinear setup and texels (one 8 B load
// per texel gives the first difference's channels 0, 1 and the second's 2, 3); r1 / a1: the second's value / approx flag
float mdiff2(int X, int Y, float2 mh, float dq, out float r1, out bool a1)
{
    md_approx = false; r1 = 0.0f; a1 = false;
    if (reset) return 0.0f;
    const float mrx = M(mh.x, rWrWh), mry = M(mh.y, rHrHh);
    const float cx = S(A((float)X, 0.5f), jx), cy = S(A((float)Y, 0.5f), jy);
    const float u = M(A(A(cx, pjx), mrx), invWr), v = M(A(A(cy, pjy), mry), invHr);
    const Bil q = bil_setup(u, v, (uint)Wr, (uint)Hr);
    const float4 t00 = pm_texel(q.i0, q.j0), t10 = pm_texel(q.i1, q.j0);
    const float4 t01 = pm_texel(q.i0, q.j1), t11 = pm_texel(q.i1, q.j1);
    r1 = mdiff_core(q, t00.zw, t10.zw, t01.zw, t11.zw, mrx, mry, dq);
    a1 = md_approx;
    md_approx = false;
    return mdiff_core(q, t00.xy, t10.xy, t01.xy, t11.xy, mrx, mry, dq);
}
float mdiff_core(Bil q, float2 t00, float2 t10, float2 t01, float2 t11, float mrx, float mry, float dq)
{
    md_approx = false;
    const float p0 = bmix(t00.x, t10.x, t01.x, t11.x, q), p1 = bmix(t00.y, t10.y, t01.y, t11.y, q);
    const float dx = FM(-p0, rWrWh, mrx), dy = FM(-p1, rHrHh, mry);
    const float mdl = SY(S_MDL), mdln = SY(S_MDLN);
    const float cdx = select(dx > mdln, min(dx, mdl), mdln), cdy = select(dy > mdln, min(dy, mdl), mdln);
    const float dxs = M(cdx, SY(S_GM)), dys = M(cdy, SY(S_GM));
    const float n2 = FM(dxs, dxs, M(dys, dys));
    const float L = select(n2 > 0.0f, in_sqrt(n2), n2);
#if RR_IN_L2CERT && !RR_IN_EXACT
    const float z = A(A(1.0f, L), dq);
    float D;
    float r = min(l2c(z, LN_2, SY(S_GD), 1.0f, D), mdl);
    const bool ok = cell_ok(r, D);
    [branch] if (WaveActiveAnyTrue(!ok)) { if (!ok) r = min(M(M(log2_cr(z), LN_2), SY(S_GD)), mdl); }
    if (ok) md_approx = true;
    return r;
#else
    return min(M(M(log2_cr(A(A(1.0f, L), dq)), LN_2), SY(S_GD)), mdl);
#endif
}

float2 mv(Tex t, int x, int y)
{
    return float2(M(A(tx(t, x, y, 0u), KF(K_MOX)), KF(K_MSHX)), M(A(tx(t, x, y, 1u), KF(K_MOY)), KF(K_MSHY)));
}

// ---- 52.7.2 ------------------------------------------------------------------------------------------------------
float dot3(float3 a, float3 b) { return FM(a.z, b.z, FM(a.x, b.x, M(a.y, b.y))); }
float len3(float3 a) { return in_sqrt(dot3(a, a)); }
float3 U(float3 a) { const float r = RCP(A(len3(a), SY(S_SV3))); return float3(M(a.x, r), M(a.y, r), M(a.z, r)); }
float3 Fv(float3 a, float s, float3 b) { return float3(FM(a.x, s, b.x), FM(a.y, s, b.y), FM(a.z, s, b.z)); }
float4 Hm(float3 a, uint base)
{
    float4 r;
    [unroll] for (uint c = 0; c < 4u; ++c) {
        const float3 col = float3(KF(base + c), KF(base + 4u + c), KF(base + 8u + c));
        const float h = A(dot3(a, col), KF(base + 12u + c));
        if (c == 0u) r.x = h; else if (c == 1u) r.y = h; else if (c == 2u) r.z = h; else r.w = h;
    }
    return r;
}
float3 cross3(float3 a, float3 b)
{
    return float3(FM(a.y, b.z, -M(a.z, b.y)), FM(a.z, b.x, -M(a.x, b.z)), FM(a.x, b.y, -M(a.y, b.x)));
}
// F_HDF (WrongGPU opt-in, not the original): the specular hit distance recon() uses is the median (mean of the middle
// two) of its aligned run of four pixels in the row, so a lone outlier (Cyberpunk 2.5: scattered missed-ray values of
// 2000 among real ~45 m hits) no longer sends one pixel's reflection reprojection far away (lone speckles in motion).
// Fast pass: tokens are row-major, one lane per pixel, so the run is four neighbouring lanes (cross-lane reads, no
// memory traffic). The replay (listed tokens, one per lane) reads the same four pixels from memory: same values, same
// sorting network, same bits.
float hd_run4(Tex tH, int X, int yc, float h)
{
#if RR_IN_REPLAY
    const int x0 = X & ~3;
    const float a = tx(tH, min(x0, Wr - 1), yc, 0u), b = tx(tH, min(x0 + 1, Wr - 1), yc, 0u);
    const float c = tx(tH, min(x0 + 2, Wr - 1), yc, 0u), d = tx(tH, min(x0 + 3, Wr - 1), yc, 0u);
#else
    const uint l = WaveGetLaneIndex() & ~3u;
    const float a = WaveReadLaneAt(h, l), b = WaveReadLaneAt(h, l + 1u);
    const float c = WaveReadLaneAt(h, l + 2u), d = WaveReadLaneAt(h, l + 3u);
#endif
    const float lo1 = min(a, b), hi1 = max(a, b), lo2 = min(c, d), hi2 = max(c, d);
    return M(A(max(lo1, lo2), min(hi1, hi2)), 0.5f);
}
// returns true (and m2) unless B1 exits
bool recon(float Z, float3 nv, float Hd, float px, float py, out float2 m2)
{
    m2 = float2(0.0f, 0.0f);
    const bool b2 = (flags & F_B2) != 0u;
    float Dz = Z;
    if ((flags & F_HWD) == 0u) {
        if (!b2 && !isfinite(Z)) return false;
        const float num = FM(Z, KF(K_V22), KF(K_V32));
        Dz = (flags & F_V33LT1) != 0u ? M(num, RCP(M(Z, KF(K_V23)))) : num;
    }
    if (!b2 && Dz == KF(K_ZFAR)) return false;
    const float4 h = Hm(float3(px, py, Dz), K_MPOS);
    if (!b2 && !(h.w > 0.0f && isfinite(h.w))) return false;
    const float rw = RCP(h.w);
    const float3 p = float3(M(h.x, rw), M(h.y, rw), M(h.z, rw));
    float sl = min(Hd, SY(S_SV2));
    sl = sl > 0.0f ? sl : SY(S_SV2);
    float3 n = U(nv);
    const float3 v = U(p);
    const float dnv = dot3(n, v);
    const float t5 = A(dnv, dnv);
    const float3 s0 = Fv(-n, t5, v);
    const float3 s = float3(M(s0.x, sl), M(s0.y, sl), M(s0.z, sl));
    float c = dot3(n, s);
    if (c < 0.0f) { n = -n; c = -c; }
    const float mc = max(c, SY(S_SV1));
    const float3 J = float3(n.z, -n.x, n.y);
    const float3 t = U(Fv(-n, dot3(n, J), J));
    const float3 b = cross3(n, t);
    const float gm = RCP(S(0.0f, RCP(mc)));
    const float km = M(RCP(mc), gm);
    const float Lv = len3(float3(M(dot3(s, t), km), M(dot3(s, b), km), gm));
    const float3 q = Fv(v, Lv, p);
    const float4 r = Hm(q, K_MREP);
    if (!b2) {
        if (!(r.w > 0.0f && isfinite(r.w))) return false;
        const float rr = RCP(r.w);
        const float ox = M(FM(r.x, rr, -px), rWhWr);
        if (!isfinite(ox)) return false;
        const float oy = M(FM(r.y, rr, -py), rHhHr);
        if (!isfinite(oy)) return false;
        m2 = float2(ox, oy);
        return true;
    }
    const bool big = abs(r.w) > SY(S_SV1);
    const float rr = RCP(r.w);
    const float ux = big ? M(r.x, rr) : 0.0f, uy = big ? M(r.y, rr) : 0.0f;
    const float ex = S(ux, px), ey = S(uy, py);
    const float sv4 = SY(S_SV4), sv4n = SY(S_SV4N);
    const float cx = isnan(ex) ? ex : (ex > sv4n ? min(ex, sv4) : sv4n);
    const float cy = isnan(ey) ? ey : (ey > sv4n ? min(ey, sv4) : sv4n);
    m2 = float2(M(cx, rWhWr), M(cy, rHhHr));
    return true;
}

// ---- 52.4 ---------------------------------------------------------------------------------------------------------
// 52.4 in integer form (branch-free): for a normal binary16 x = 2^(E-15) (1 + M/1024) and dither d, x + fa is
// 2^(E-14) (2048 + M + d) / 2048 rounded to 10 mantissa bits, i.e. t2 = RNE((M + d) / 2) units of 2^(E-14) / 1024 above
// 2^(E-14); s2 = s1 - fb is exact, so its e4m3 code (truncation to 3 mantissa bits) is: at exponent E + 1 when
// t2 >= 512 (then mantissa 0, as t2 <= 575), else at exponent E with mantissa t2 >> 6. As one sum: normal code
// (E - 8) 8 + min(t2 >> 6, 8), saturated at 0x7e (e > 8, or e = 8 with mantissa 7; E >= 30 is inside that);
// below the e4m3 normal range the 4-bit significand 8 + m3 is rounded (RNE) to the subnormal grid. A zero or
// subnormal binary16 x gives |s2| < 2^-10: code 0 with the sign of x (= the sign of v). Checked exhaustively against
// the reference for every binary16 value and dither (probe "quant").
// f32 -> f16 bits, round to nearest even: the software form, or (RR_IN_AGSH, fast pass) one v_cvt_f16_f32 through the
// driver's matrix conversion (a fragment with one nonzero element; wave-uniform control flow only). The same bits for
// every non-NaN f32 (checked for all 2^32 patterns: gpu/input_stage.py --f2hcheck); quant reads NaN from v itself.
#if WG_WAVE == 32
uint f2h_ags(float x)
{
    v8f d = wg_splat(0.0f);
    d.v[0] = x;
    return wg_to_f16_ags(d).x & 0xffffu;
}
#else
uint f2h_ags(float x) { return wg_f2h(x); }   // wave64 builds: the fragment conversion is wave32-only (not used by quant)
#endif
#ifndef RR_IN_QF1C
#define RR_IN_QF1C 0         // a14: quant's binary16 rounding by the per-value op 0x1C (in_f2h_hw: wg_f2h's bits for all 2^32)
#endif                       // instead of the 8-value fragment conversion
#if RR_IN_AGSH && !RR_IN_EXACT && (RR_IN_QF1C || WG_WAVE == 64)
#define QF2H in_f2h_hw
#elif RR_IN_AGSH && !RR_IN_EXACT
#define QF2H f2h_ags
#else
#define QF2H wg_f2h
#endif
#ifndef RR_IN_QFOLD
#define RR_IN_QFOLD 1        // round 13: quant's code index from one word (the same u)
#endif
// round 13: f32 -> f16 bits, round to nearest even, as one v_cvt_f16_f32 through the driver channel (op 0x1C, immediate
// 1; rr_fast.hlsli f2h_hw): the bits of wg_f2h for every one of the 2^32 inputs, NaN included (gpu/input_stage.py
// --f2hcheck with RR_F2HC_HW 1). The argument is fenced (no fold of its producer into the conversion).
uint in_f2h_hw(float x) { return wg_ags_call(0x1cu, 0u, 1u, asuint(x) ^ (0u - (wg_grid_ >> 31)), 0u) & 0xffffu; }
#ifndef RR_IN_TF2H
#define RR_IN_TF2H 1         // round 13: the tail's hold / 52.7 conversions through in_f2h_hw (fast pass only)
#endif
#if RR_IN_TF2H && RR_IN_AGSH && !RR_IN_EXACT
#define TF2H in_f2h_hw
#else
#define TF2H wg_f2h
#endif
// RR_IN_QHW 1 (clean/rr-f-s9-kernels; fast pass with RR_IN_AGSH only, the same bits): with u = 8 E + c0 (c0 = t2 >> 6 <= 8)
// the code is the e4m3 code of the binary32 value 2^(E-15) (1 + c0/8) with the sign of v: exact in e4m3's 3 mantissa bits
// at exponent E + (c0 >> 3) in the normal range; below it the conversion's RNE to the subnormal grid is 52.4's rounding of
// 8 + m3; E = 0 gives a value below half the smallest subnormal (code 0 with the sign); saturation at 0x7e is u <= 190
// (448); NaN 0x7f with the sign. One v_cvt_pk_fp8_f32 through the driver conversion replaces the integer code steps.
// Checked for every binary16 value and dither (probe "quant", fast-pass build).
#ifndef RR_IN_QHW
#define RR_IN_QHW 1
#endif
#if RR_IN_QHW && RR_IN_AGSH && !RR_IN_EXACT
// RR_IN_Q36 1 (clean/rr-f-lw-fold, default): the code through the driver's per-value conversion (op 0x36 imm 2: one
// v_cvt_pk_fp8_f32, RNE, no saturation, code in bits 0-7) instead of the 8-value fragment conversion, whose second
// v_cvt_pk_fp8_f32 converted the constant zeros of the unused elements. The same instruction on the same value: the
// same code (checked for every binary16 value and dither, probe "quant"; lw-fold-kb/scr/qprobe.py: 0 differences in
// 8.8M; the negative control RR_IN_Q36 2 differs in 8.1M). Features pass -0.009 / -0.005 ms (1080p / 1440p Q) on top of
// RR_IN_UKW; input lite -0.010 / -0.015 ms.
#ifndef RR_IN_Q36
#define RR_IN_Q36 1
#endif
uint q8code_ags(float s)
{
#if RR_IN_Q36 == 1
    return wg_ags_call(0x36u, 0u, 2u, asuint(s), 0u) & 0xffu;
#elif RR_IN_Q36 == 2   // negative control of the probe only: the bf8 code
    return wg_ags_call(0x36u, 0u, 3u, asuint(s), 0u) & 0xffu;
#else
    v8f d = wg_splat(0.0f);
    d.v[0] = s;
    return wg_pack_q8_nc_ags(d).x & 0xffu;
#endif
}
#endif
#ifndef RR_IN_QNAN
#define RR_IN_QNAN 1         // a14 C2: a NaN's code from op 0x36 itself (u 255 -> 122880, no saturation: 0x7f | sign)
#endif
#ifndef RR_IN_H1Q
#define RR_IN_H1Q 1          // a14 C7: hist_code's certificate on the two ends' (sign, u) and one code conversion
#endif
#if RR_IN_QHW && RR_IN_AGSH && !RR_IN_EXACT
// a14: quant's index u from the binary16 bits (RR_IN_QFOLD form) and the code from (u, sign)
uint quant_u(uint xb, uint d)
{
    const uint wq = (xb & 0x7fffu) + d;
    return min((wq + ((wq >> 1) & 1u)) >> 7, 190u);
}
#ifndef RR_IN_PKQ
#define RR_IN_PKQ 0          // a14 C36: hist_code's two quant indices as one packed u16 pair (v_pk_* integer ops)
#endif
// both ends' u at once: the same integer steps on 16-bit lanes ((b & 0x7fff) + d <= 0x807e: no carry out of a lane)
uint2 quant_u2(uint blo, uint bhi, uint d)
{
    const uint16_t2 w = uint16_t2((uint16_t)(blo & 0x7fffu), (uint16_t)(bhi & 0x7fffu)) + (uint16_t)d;
    const uint16_t2 u = min((w + ((w >> (uint16_t)1) & (uint16_t)1)) >> (uint16_t)7, (uint16_t)190);
    return uint2(u.x, u.y);
}
uint quant_code(uint u, float v) { return q8code_ags(asfloat(((u << 20) + 0x38000000u) | (asuint(v) & 0x80000000u))); }
#endif
uint quant(float v, uint d)
{
    const uint xb = QF2H(v);
#if RR_IN_QHW && RR_IN_AGSH && !RR_IN_EXACT && RR_IN_QNAN && RR_IN_QFOLD
    // a quiet binary16 NaN has (xb & 0x7fff) >= 0x7e00, so ur >= 252; inf and finite values give ur <= 249
    const uint wq = (xb & 0x7fffu) + d;
    const uint ur = (wq + ((wq >> 1) & 1u)) >> 7;
    const uint u = min(ur, 190u) + ((ur + 4u) >> 8) * 65u;
    return q8code_ags(asfloat(((u << 20) + 0x38000000u) | (asuint(v) & 0x80000000u)));
#elif RR_IN_QHW && RR_IN_AGSH && !RR_IN_EXACT
#if RR_IN_QFOLD
    // round 13: w = E 1024 + t (t = mantissa + d); E 1024 is a multiple of 128 with bit 1 clear, so
    // (w + ((w >> 1) & 1)) >> 7 = 8 E + c0: the same u in fewer operations
    const uint wq = (xb & 0x7fffu) + d;
    const uint u = min((wq + ((wq >> 1) & 1u)) >> 7, 190u);
#else
    const uint t = (xb & 1023u) + d;
    const uint c0 = (t + ((t >> 1) & 1u)) >> 7;
    const uint u = min(((xb >> 7) & 0xf8u) + c0, 190u);
#endif
    const uint code = q8code_ags(asfloat(((u << 20) + 0x38000000u) | (asuint(v) & 0x80000000u)));
    return select(isnan(v), 0x7fu | ((asuint(v) >> 31) << 7), code);
#else
    const uint E = (xb >> 10) & 31u, t = (xb & 1023u) + d;
    const uint t2 = (t >> 1) + (t & (t >> 1) & 1u);
    const uint c0 = min(t2 >> 6, 8u);
    const uint Es2 = E + (c0 >> 3);
    const uint sh = (uint)max(9 - (int)Es2, 1);
    const uint mant = 8u + (c0 & 7u);
    const uint subc = (mant + (1u << (sh - 1u)) - 1u + ((mant >> sh) & 1u)) >> sh;   // RNE(mant / 2^sh)
    uint code = select(Es2 >= 9u, min(E * 8u + c0 - 64u, 0x7eu), subc);
    code = select(E == 0u, 0u, code);
    code = select(isnan(v), 0x7fu, code);
    return code | ((asuint(v) >> 31) << 7);
#endif
}

// feature indices of the five history samples (52.3.4): own Hc1, own Hc2, near, far, m2; channels r, g, b
static const uint HK[15] = {12u, 13u, 16u, 17u, 20u, 21u, 5u, 8u, 9u, 24u, 25u, 28u, 37u, 40u, 41u};
static const uint HIDK[8] = {22u, 23u, 26u, 27u, 45u, 48u, 49u, 52u};

// a14 (LDS diet, C32 / C6 family): RR_IN_GSW = the code words kept in LDS. Features 56 .. 63 (words 14, 15) are never
// written (always code 0), so 14 words suffice: gs_codes 7 KB + gs_k 1 KB = 8 KB of LDS per 128-lane workgroup, the
// limit for 16 waves per SIMD (13 at 9 KB); the two words are stored as zeros
// clean/rr-mp2-input I3 (RR_IN_CSLOT; 2 is the default in the wave64 features kernels, 1 and 0 are options; 0 is the
// old layout): only the features that are ever written get a byte (slot) in LDS.
// RR_IN_CSLOT 1: feature k <= 33 at slot k, the written features above it (36, 37, 40, 41, 44, 45, 48, 49, 52; every
// other one stays code 0) at slots 34 .. 42 (34 + 2 ((k - 36) >> 2) + (k & 1)). 11 words, 5.5 KB per 128-lane
// workgroup instead of 7 KB, so the LDS no longer caps the wave64 features passes at 9 waves per SIMD (128 KB per WGP).
// RGP shows input.features at 10 waves then (65 VGPRs, 72 allocated; 64 also allocates 72). Each code is OR-ed once into its zeroed slot as before; the stores rebuild the same 16
// words: words 0 .. 7 are slot words 0 .. 7, words 8 .. 13 the 16-bit halves of slot words 8 .. 10 (features 34, 35, 38,
// 39, 42, 43, 46, 47, 50, 51, 53 .. 55 are the zero bytes there, slot 43 is never written), words 14, 15 zero.
#ifndef RR_IN_CSLOT
#define RR_IN_CSLOT (2 * RR_IN_W64)
#endif
// RR_IN_CSLOT 2 (10 words, 5 KB; measured faster than 1 at the same 10 waves: fewer LDS operations): the guides 30 .. 32 are packed into one register by their own
// loop (live only from there to the stores), 33 and 36 (constant per frame apart from the dither) are quantised at the
// stores; the other 38 written features take slots: k <= 29 at slot k, 37, 40, 41, 44, 45, 48, 49, 52 at slots 30 .. 37
#ifndef RR_IN_GSW
#if RR_IN_CSLOT == 2
#define RR_IN_GSW 10
#elif RR_IN_CSLOT
#define RR_IN_GSW 11
#else
#define RR_IN_GSW 14
#endif
#endif
groupshared uint gs_codes[RR_IN_GSW * WG_BLOCK];
#if RR_IN_CSLOT == 2
static uint in_gq;      // guide codes 30, 31, 32 (bytes 0, 1, 2)
static float in_x3;     // feature 36's value (X3)
// slots 30 .. 37 for 37, 40, 41, 44, 45, 48, 49, 52: 37 -> 30, 40 -> 31, 41 -> 32, 44 -> 33, 45 -> 34, 48 -> 35, 49 -> 36,
// 52 -> 37 (k = 37 + 4 j - 1 + b for j >= 1: slot 29 + 2 j + b)
uint in_slot(uint k) { return k <= 29u ? k : (k == 37u ? 30u : 29u + 2u * ((k - 36u) >> 2) + (k & 1u)); }
#elif RR_IN_CSLOT
uint in_slot(uint k) { return k <= 33u ? k : 34u + 2u * ((k - 36u) >> 2) + (k & 1u); }
uint gs_code_word(uint z, uint lane)
{
    if (z < 8u) return gs_codes[z * WG_BLOCK + lane];
    if (z >= 14u) return 0u;
    const uint w = gs_codes[(8u + ((z - 8u) >> 1)) * WG_BLOCK + lane];
    return (z & 1u) == 0u ? (w & 0xffffu) : (w >> 16);
}
#else
uint in_slot(uint k) { return k; }
uint gs_code_word(uint z, uint lane) { return z < RR_IN_GSW ? gs_codes[z * WG_BLOCK + lane] : 0u; }
#endif
static uint dith;
static uint dbg_base;
static uint lane_;
static uint2 dbg_approx = uint2(0u, 0u);   // features (bit k) whose debug f32 value is a certified approximation
#if RR_IN_CSLOT == 2
uint gs_code_word(uint z, uint lane)
{
    if (z < 7u) return gs_codes[z * WG_BLOCK + lane];
    if (z >= 14u) return 0u;
    const uint s7 = gs_codes[7u * WG_BLOCK + lane], s8 = gs_codes[8u * WG_BLOCK + lane], s9 = gs_codes[9u * WG_BLOCK + lane];
    if (z == 7u) return (s7 & 0xffffu) | (in_gq << 16);                                   // 28, 29 | 30, 31
    if (z == 8u) return ((in_gq >> 16) & 0xffu) | (quant(clipf(0.0f), dith) << 8);         // 32, 33
    if (z == 9u) return quant(in_x3, dith) | ((s7 >> 8) & 0xff00u);                        // 36, 37
    if (z == 10u) return (s7 >> 24) | ((s8 & 0xffu) << 8);                                 // 40, 41
    if (z == 11u) return (s8 >> 8) & 0xffffu;                                              // 44, 45
    if (z == 12u) return (s8 >> 24) | ((s9 & 0xffu) << 8);                                 // 48, 49
    return (s9 >> 8) & 0xffu;                                                              // 52
}
#endif
void put_code(uint k, uint code, float v)
{
    const uint sl = in_slot(k);
    const uint i = (sl >> 2) * WG_BLOCK + lane_;
#if RR_IN_SKIP & 16
    gs_codes[i] = gs_codes[i] ^ asuint(v);
#else
    InterlockedOr(gs_codes[i], code << ((sl & 3u) * 8u));               // ds_or_b32 (one LDS operation)
#endif
#if RR_IN_DBG
    if (dbg_base != 0u) wg_st32(dbg_base, k * 4u, asuint(v));
#endif
}
void put(uint k, float v) { put_code(k, quant(v, dith), v); }
#if RR_IN_CSLOT == 2
// the guide features 30 + gd (gd < 3, wave-uniform): the same code into byte gd of in_gq (zeroed per token) instead of LDS
void put_guide(uint gd, float v)
{
    in_gq |= quant(v, dith) << (8u * gd);
#if RR_IN_DBG
    if (dbg_base != 0u) wg_st32(dbg_base, (30u + gd) * 4u, asuint(v));
#endif
}
#else
void put_guide(uint gd, float v) { put(30u + gd, v); }
#endif
#if RR_IN_DBG
void approx_bit(uint k) { if (k < 32u) dbg_approx.x |= 1u << k; else dbg_approx.y |= 1u << (k - 32u); }
#else
void approx_bit(uint k) {}
#endif

// An approximation v of a feature whose reference value lies within D of v has the reference code when the interval
// [v - D, v + D] lies inside one binary16 rounding cell: D below the distance to the nearest binary16 rounding boundary,
// at least min(|low13(v) - 0x1000|, 0x800) f32 steps for 2^-14 <= |v| < 2^15. D = 0 marks a value computed with the
// reference operations (v is then the reference value).
bool cell_ok(float v, float D)
{
    const uint b = asuint(v) & 0x7fffffffu;
    const int dm = abs((int)(b & 0x1fffu) - 0x1000);
    const float ulp = asfloat((b & 0x7f800000u) - 0x0b800000u);
    return or(D == 0.0f, and(b - 0x38800000u < 0x0e800000u, D < (float)min(dm, 0x800) * ulp));
}

// M(M(M(log2_cr(y), c1), c2), c3) from the hardware log2 (within one f32 step of log2_cr for every positive normal y,
// --hwerr) and plain products, with the bound D = 2^-20 |r| + 2^-124 (1 + |c2| + |c3| + |c2 c3|) (one step of L, three
// roundings in each chain, flushed products); y = 1 gives L = +0 in both, so r is the reference value (D = 0). A y that
// is not a positive normal f32 gives D = inf (the caller recomputes).
float l2c(float y, float c1, float c2, float c3, out float D)
{
    const float L = log2(y);
    const float r = ((L * c1) * c2) * c3;
    const float tail = asfloat(0x01800000u) * ((1.0f + abs(c2)) + (abs(c3) + abs(c2 * c3)));
    D = L == 0.0f ? 0.0f : mad(abs(r), asfloat(0x35800000u), tail);
    D = and(y >= asfloat(0x00800000u), y < asfloat(0x7f800000u)) ? D : asfloat(0x7f800000u);
#if RR_IN_HCERT_MUT
    D = 0.0f;                                                // negative control: no bound
#endif
    return r;
}
float c3(float3 v, uint i) { return i == 0u ? v.x : (i == 1u ? v.y : v.z); }
uint in_c3u(uint3 v, uint i) { return i == 0u ? v.x : (i == 1u ? v.y : v.z); }
float3 in_set3(float3 v, uint i, float x) { return float3(i == 0u ? x : v.x, i == 1u ? x : v.y, i == 2u ? x : v.z); }
uint3 in_set3u(uint3 v, uint i, uint x) { return uint3(i == 0u ? x : v.x, i == 1u ? x : v.y, i == 2u ? x : v.z); }
bool in_same2(float2 a, float2 b) { return and(asuint(a.x) == asuint(b.x), asuint(a.y) == asuint(b.y)); }   // same bits

// ---- certified codes of the history features (fast pass, RR_IN_HCERT) -----------------------------------------------
// The feature clipf(M(ccf(hdec(h)), gh)) of a history sample h is evaluated with plain f32 operations and the hardware
// exp2 / log2 (within one f32 step of the correctly rounded ones for every input: gpu/input_stage.py --hwerr), with a
// running bound D on its distance from the reference value (every reference operation rounds once; the bound covers
// both chains' roundings, the flush rule, the hardware functions and the propagation; eps = 2^-24):
//   D_t = 4.0001 eps |t| + tail                          t = clamp(h HDEC) K8 L2T (three roundings each chain)
//   D_e = e (2^-22 + 0.71 D_t) (1 + 2^-18)                e = 2^t (one step either way; 2^D (2^D - 1) <= 0.71 D)
//   D_a = D_e (1 + 2^-23) + 2^-23 |a| + 2^-126           a = e - 1
//   D_u = |P| D_a (1 + 2^-20) + 2^-21 (|u| + |P| D_a) + cB   u = a K12 rcpEp E (P their product)
//   D_y = |K4| D_u (1 + 2^-23) + 2^-23 (|y0| + |u K4|) + 2^-125   y = max(u K4 + 1, K5)
//   D_L = 1.443 D_y / (y - D_y) (1 + 2^-20) + 2^-21 |L| + 2^-140   L = log2 y
//   D_v = |Q| D_L (1 + 2^-20) + 2^-21 (|v| + |Q| D_L) + cC   v = min(L LOG10_2 K7 gh, FCLIP) (Q the product)
// The reference value lies in [v - D_v, v + D_v]; quant(., d) is constant on every interval whose ends give the same
// code (52.4 is monotone in the binary16 value: checked for every binary16 value and dither), so equal codes at the
// two ends are the reference code. Otherwise (and out of range) the value is recomputed with the reference operations
// (wave-uniform branch).
static float hc_aP, hc_aQ, hc_P, hc_Q, hc_C, hc_cB, hc_cC, hc_aK4, hc_aCt;
static bool hc_ok;
#ifndef RR_IN_HBL
#define RR_IN_HBL 1          // a14 C35: the chain D_t .. D_v folded into per-frame constants (a looser valid bound)
#endif
// a14 C35 (RR_IN_HBL): substituting D_t .. D_y and D_L .. D_v into each other (|u| <= |a| |P| (1 + 2^-23), |u K4| <=
// |a| |P K4| (1 + 2^-22), D_t <= 4.0001 eps |t| + aCt) gives, in real arithmetic, bounds at least as large as the chain's:
//   D_e = e (k0 + k1 |t|)                         k0 = (2^-22 + 0.71 aCt)(1 + 2^-18), k1 = 0.71 * 4.0001 eps (1 + 2^-18)
//   D_y = A1 |a| + B1 D_e + 2^-23 |y0| + G1      (A1, B1, G1: the coefficients of |a|, D_e and 1)
//   D_v2 = V1 |v| + Q1 (1.4430 D_y / (y - D_y) + 2^-21 |L| + 2^-140) + C1
// Every coefficient carries an extra (1 + 2^-18) and G1 / C1 an extra 2^-123, which cover the f32 evaluation of these
// non-negative sums (at most 8 roundings each) and flushed tiny terms. D_t < 2^-10 holds whenever |t| < 120 and hc_ok
// (aCt <= 2^-22), so that condition is implied. A value the looser bound does not certify goes to the exact path.
static float hb_k0, hb_k1, hb_A1, hb_B1, hb_G1, hb_V1, hb_QL, hb_Q21, hb_C2;
bool hc_mid(float x) { return and(abs(x) >= asfloat(0x0d800000u), abs(x) <= asfloat(0x71800000u)); }   // 2^-100 .. 2^100
void hc_setup()
{
    hc_P = (SY(S_K12) * rcpEp) * E;
    hc_aP = abs(hc_P) * asfloat(0x3f800008u);                              // (1 + 2^-20)
    hc_Q = (LOG10_2 * SY(S_K7)) * SY(S_GH);
    hc_aQ = abs(hc_Q) * asfloat(0x3f800008u);
    hc_C = SY(S_K8) * LOG2_10;
    hc_cB = asfloat(0x01000000u) * (1.0f + abs(rcpEp * E) + abs(E) + abs(SY(S_K12) * rcpEp));   // 2^-125 (...)
    hc_cC = asfloat(0x01000000u) * (1.0f + abs(SY(S_K7) * SY(S_GH)) + abs(SY(S_GH)));
    hc_aK4 = abs(SY(S_K4)) * asfloat(0x3f800001u);                         // (1 + 2^-23)
    hc_aCt = asfloat(0x02000000u) * (1.0f + abs(hc_C));                    // 2^-123 (1 + |K8 L2T|)
#if RR_IN_HBL
    {
        const float m = asfloat(0x3f800020u);                                  // 1 + 2^-18
        const float aPK = abs(hc_P) * abs(SY(S_K4));
        hb_k0 = (asfloat(0x34800000u) + 0.71f * hc_aCt) * m * m;
        hb_k1 = asfloat(0x3435c3bbu) * m * m;                                  // 0.7100001 * 4.0001 * 2^-24 = 2.84007 eps (up)
        const float pk = hc_aK4 * hc_aP * asfloat(0x3f800004u);               // aK4 aP (1 + 2^-21)
        hb_A1 = (hc_aK4 * abs(hc_P) * asfloat(0x35000001u) + pk * asfloat(0x34000000u) + aPK * asfloat(0x34000002u)) * m * m;
        hb_B1 = pk * asfloat(0x3f800001u) * m * m;
        hb_G1 = (pk * asfloat(0x00800000u) + hc_aK4 * hc_cB + asfloat(0x01000000u)) * m * m + asfloat(0x02000000u);
        const float Q1 = hc_aQ * asfloat(0x3f800008u) * asfloat(0x3f800004u);  // aQ (1 + 2^-20)(1 + 2^-21)
        hb_V1 = (asfloat(0x34000000u) + asfloat(0x35000000u) * asfloat(0x3f800008u)) * m * m;
        hb_QL = Q1 * asfloat(0x3fb8b44eu) * m * m;
        hb_Q21 = Q1 * asfloat(0x35000000u) * m * m;
        hb_C2 = (Q1 * asfloat(0x09800000u) + hc_cC * asfloat(0x3f800008u)) * m * m + asfloat(0x02000000u);
    }
#endif
    // the bound's tails assume moderate constants (wave-uniform; otherwise every token goes to the exact replay)
    hc_ok = and(and(and(hc_mid(hc_P), hc_mid(hc_Q)), and(hc_mid(hc_C), hc_mid(E))),
                and(and(hc_mid(rcpEp), hc_mid(SY(S_K12))), and(and(hc_mid(SY(S_K7)), hc_mid(SY(S_GH))),
                    or(SY(S_K4) == 0.0f, hc_mid(SY(S_K4))))));
}
uint hist_code(float h, bool iv, float fbv, out float vout, out bool ok)
{
    const float h1 = h * SY(S_HDEC);
    const float hp = select(h1 > SY(S_K5), min(h1, SY(S_K10)), SY(S_K5));
    const float t = hp * hc_C;
    const float e = in_ftz(exp2(t));
    const float a = e - 1.0f;
    const float u = a * hc_P;
    const float uk = u * SY(S_K4);
    const float y0 = uk + 1.0f;
    const float y = max(y0, SY(S_K5));
    const float L = log2(y);
    const float v = min(L * hc_Q, SY(S_FCLIP));
#if RR_IN_HBL
    const float De = e * mad(abs(t), hb_k1, hb_k0);
    const float Dy = mad(hb_A1, abs(a), mad(hb_B1, De, mad(asfloat(0x34000000u), abs(y0), hb_G1)));
    const float ylo = y - Dy;
    const float Dv2 = mad(hb_V1, abs(v), mad(hb_QL, Dy * rcp(ylo), mad(hb_Q21, abs(L), hb_C2)));
    const float Dt = 0.0f;                                                     // implied (see above)
#else
    // the bound
    const float Dt = mad(abs(t), asfloat(0x34800068u), hc_aCt);              // 4.0001 eps |t| + tail
    const float De = (e * mad(Dt, 0.71f, asfloat(0x34800000u))) * asfloat(0x3f800020u);     // (2^-22 + 0.71 D_t)(1 + 2^-18)
    const float Da = mad(De, asfloat(0x3f800001u), mad(abs(a), asfloat(0x34000000u), asfloat(0x00800000u)));
    const float PDa = hc_aP * Da;
    const float Du = mad(abs(u) + PDa, asfloat(0x35000000u), PDa + hc_cB);
    const float Dy = mad(hc_aK4, Du, mad(abs(y0) + abs(uk), asfloat(0x34000000u), asfloat(0x01000000u)));
    const float ylo = y - Dy;
    const float DL = mad(Dy * rcp(ylo), asfloat(0x3fb8b44eu), mad(abs(L), asfloat(0x35000000u), asfloat(0x09800000u)));
    const float QDL = hc_aQ * DL;
    const float Dv = mad(abs(v) + QDL, asfloat(0x35000000u), QDL + hc_cC);
    const float Dv2 = mad(abs(v), asfloat(0x34000000u), Dv * asfloat(0x3f800008u));
#endif
    // an invalid tap takes the exact fallback value fbv: both ends are fbv (no branch around the quantisations)
    vout = iv ? fbv : v;
#if RR_IN_HCERT_MUT
    const uint clo = quant(vout, dith), chi = clo;           // negative control: no bound (the qualification must fail)
#elif RR_IN_H1Q && RR_IN_QHW && RR_IN_AGSH && RR_IN_QFOLD && !RR_IN_EXACT
    // a14 C7: quant's code is a function of (u, sign); equal (u, sign) at the two ends imply equal codes (a looser test:
    // ends with different (u, sign) but the same code now fail and go to the exact path). When ok the ends are finite
    // (isfinite(L Q), D_v < 2^100), so quant's NaN case cannot occur; a failed value's code is replaced.
    const float xlo = iv ? fbv : v - Dv2, xhi = iv ? fbv : v + Dv2;
    const uint blo = QF2H(xlo), bhi = QF2H(xhi);
#if RR_IN_PKQ
    const uint2 u2 = quant_u2(blo, bhi, dith);
    const uint ulo = u2.x;
    const bool same = and(((blo ^ bhi) & 0x8000u) == 0u, u2.x == u2.y);
#else
    const uint ulo = quant_u(blo, dith);
    const bool same = and(((blo ^ bhi) & 0x8000u) == 0u, ulo == quant_u(bhi, dith));
#endif
    const uint clo = quant_code(ulo, xlo);
    const uint chi = same ? clo : ~clo;
#else
    const uint clo = quant(iv ? fbv : v - Dv2, dith), chi = quant(iv ? fbv : v + Dv2, dith);
#endif
    // conditions of the bound: |t| < 120 (e normal and finite), D_t small, y - D_y >= 2^-125 (also false for NaN),
    // finite L Q (min() would hide a NaN), D_v finite (a NaN bound would compare false)
    ok = and(and(and(hc_ok, abs(t) < 120.0f), and(Dt < asfloat(0x3a800000u), ylo >= asfloat(0x01000000u))),
             and(and(isfinite(L * hc_Q), Dv2 < asfloat(0x71800000u)), clo == chi));
    ok = or(ok, iv);
    return clo;
}

// Hidden features (52.3.6), fast pass: the bilinear mix and the gs product in plain f32 with the bound
// D = |gs| (1 + 2^-20) 2^-20 (|t00| + |t10| + |t01| + |t11|) + 2^-22 |h| + 2^-122 (1 + |gs|) (both chains: at most
// 4 eps sum|t| for the mix, weights in [0, 1]; eps |h| for the product; flushed products); a lane with a value not
// inside one binary16 cell (cell_ok) recomputes its four values with the reference operations.
void hid_codes(float2 pos, uint k0, uint k1, uint k2, uint k3)
{
    float4 s = float4(0.0f, 0.0f, 0.0f, 0.0f), at = float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (!outside(pos.x, pos.y)) {
        const float uh = FM(pos.x, KF(K_INVWH), KF(K_HOFFX)), vh = FM(pos.y, KF(K_INVHH), KF(K_HOFFY));
        const Bil q = bil_setup(uh, vh, (uint)Wr, (uint)Hr);
        const float4 t00 = hid_texel(q.i0, q.j0), t10 = hid_texel(q.i1, q.j0);
        const float4 t01 = hid_texel(q.i0, q.j1), t11 = hid_texel(q.i1, q.j1);
        const float ofx = 1.0f - q.fx, ofy = 1.0f - q.fy;
        s = ofy * (ofx * t00 + q.fx * t10) + q.fy * (ofx * t01 + q.fx * t11);
        at = (abs(t00) + abs(t10)) + (abs(t01) + abs(t11));
    }
    const float gs = SY(S_GS);
    const float ags = abs(gs) * asfloat(0x35800008u);                      // |gs| (1 + 2^-20) 2^-20
    const float tail = asfloat(0x02800000u) * (1.0f + abs(gs));             // 2^-122 (1 + |gs|)
    float4 h = gs * s;
    // all four texels +0 (snorm code 0) or outside the image: the same operations on +0 give the reference value
    const float4 D = select(at == 0.0f, 0.0f, ags * at + (abs(h) * asfloat(0x34800000u) + tail));
    const bool ok = and(and(cell_ok(h.x, D.x), cell_ok(h.y, D.y)), and(cell_ok(h.z, D.z), cell_ok(h.w, D.w)));
    [branch] if (WaveActiveAnyTrue(!ok)) {
        if (!ok) h = hidf(pos);                                    // the reference operations
    }
    put(k0, h.x); put(k1, h.y); put(k2, h.z); put(k3, h.w);
    if (ok) { approx_bit(k0); approx_bit(k1); approx_bit(k2); approx_bit(k3); }
}

void in_token(uint t);
// The fast pass: one lane per token. The replay (RR_IN_REPLAY): the listed tokens only, one lane per listed token
// (a fixed grid strides over the list), so its cost follows the number of marked tokens, not their spread.
WG_KERNEL(rr_input_features)
{
    WG_SETUP();
#if RR_IN_REPLAY
    const uint pm0 = KD(K_MARK);
    const uint cnt = wg_ld32(pm0, 0u);
#if RR_IN_RSPREAD
    if (wg_block_id() * (WG_BLOCK / WG_WAVE) >= cnt) return;  // no wave of this workgroup has an entry
#else
    if (wg_block_id() * WG_BLOCK >= cnt) return;              // uniform per workgroup, before any staging
#endif
#endif
    in_stage_block();
    in_plog = KW(K_LOGT); in_pexp = KW(K_EXPT);
    in_stage_tables();
    lane_ = wg_lane_id();
    Wr = (int)KW(K_WR); Hr = (int)KW(K_HR); Wt = (int)KW(K_WT); Ht = (int)KW(K_HT);
#if RR_IN_REPLAY && RR_IN_RSPREAD
    {   // every lane's tokens are computed alone (in_token per lane): the order of the list entries changes no value
        const uint nw = wg_grid_ * (WG_BLOCK / WG_WAVE), wv = wg_block_id() * (WG_BLOCK / WG_WAVE) + lane_ / WG_WAVE;
        [loop] for (uint i = wv + (lane_ % WG_WAVE) * nw; i < cnt; i += nw * WG_WAVE)
            in_token(wg_ld32(pm0, 4u + i * 4u));
    }
#elif RR_IN_REPLAY
    [loop] for (uint i = wg_block_id() * WG_BLOCK + lane_; i < cnt; i += wg_grid_ * WG_BLOCK)
        in_token(wg_ld32(pm0, 4u + i * 4u));
#elif RR_IN_TILE2D && WG_BLOCK == 128
    // a14 C16 (in-2d-token-tiles): a workgroup takes a 16 x 8 token tile (1: 16 x 2 per wave, 2: 8 x 4 per wave) when the
    // token grid is a multiple of it (the same token set and grid size; every token computed alone), else the row form
    {
        const uint g = wg_block_id(), ntx = (uint)Wt >> 4;
#if RR_IN_TILE2D == 2
        const uint w = lane_ >> 5, l = lane_ & 31u;
        const uint lx = (w & 1u) * 8u + (l & 7u), ly = (w >> 1) * 4u + (l >> 3);
#else
        const uint lx = lane_ & 15u, ly = lane_ >> 4;
#endif
        const uint t2 = ((g / max(ntx, 1u)) * 8u + ly) * (uint)Wt + (g % max(ntx, 1u)) * 16u + lx;
        in_token((((uint)Wt & 15u) | ((uint)Ht & 7u)) == 0u ? t2 : g * WG_BLOCK + lane_);
    }
#else
    in_token(wg_block_id() * WG_BLOCK + lane_);
#endif
}

void in_token(uint t)
{
    [unroll] for (uint z = 0; z < RR_IN_GSW; ++z) gs_codes[z * WG_BLOCK + lane_] = 0u;
    in_flag = 0u; in_slow = 0u; in_ziv = 0u; dbg_approx = uint2(0u, 0u);
#if RR_IN_CSLOT == 2
    in_gq = 0u;
#endif
    if (t >= (uint)(Wt * Ht)) return;
    Whf = (float)KW(K_WH); Hhf = (float)KW(K_HH);
    flags = KW(K_FLAGS);
    reset = (flags & F_RESET) != 0u;
    const uint pex = KW(K_EXPO);
    E = pex != 0u ? asfloat(KU(wg_ld32(pex, 0u))) : KF(K_E);
    rcpEp = pex != 0u ? RCP(asfloat(KU(wg_ld32(pex, 4u)))) : KF(K_RCPEP);
    const float X3 = pex != 0u ? asfloat(KU(wg_ld32(pex, 12u))) : KF(K_X3);
    jx = KF(K_JX); jy = KF(K_JY); pjx = KF(K_PJX); pjy = KF(K_PJY);
    rWhWr = KF(K_RWHWR); rHhHr = KF(K_RHHHR); rWrWh = KF(K_RWRWH); rHrHh = KF(K_RHRHH);
    invWr = KF(K_INVWR); invHr = KF(K_INVHR);
#if RR_IN_HCERT && !RR_IN_EXACT
    hc_setup();
#endif
    const int X = (int)(t % (uint)Wt), Y = (int)(t / (uint)Wt);
    const int xc = min(X, Wr - 1), yc = min(Y, Hr - 1);
    const uint fc = KW(K_FC);
    dith = wg_ld16(KW(K_D), (((fc * 16u + ((uint)Y & 15u)) * 16u + ((uint)X & 15u)) * 2u)) >> 9;
    const uint pdbg = KW(K_DEBUG);
    dbg_base = pdbg == 0u ? 0u : pdbg + t * 96u * 4u;
    const float amax = SY(S_AMAX);

    const Tex tC = tex(R_C);
    const float3 C = float3(tx(tC, xc, yc, 0u), tx(tC, xc, yc, 1u), tx(tC, xc, yc, 2u));
#if RR_IN_NHL && !RR_IN_EXACT
    [loop] for (uint c0 = 0; c0 < 3u + in_zl(); ++c0) put(c0 == 2u ? 4u : c0, nl_cc(c3(C, c0)));
#elif RR_IN_L2CERT && !RR_IN_EXACT
    [loop] for (uint c0 = 0; c0 < 3u + in_zl(); ++c0) {
        const float cv = c3(C, c0);
        const float u = min(in_nan0(M(E, cv)), F16MAX);
        float D;
        float r = l2c(max(FM(u, SY(S_K4), 1.0f), SY(S_K5)), LOG10_2, SY(S_K7), 1.0f, D);
        const bool ok = cell_ok(r, D);
        [branch] if (WaveActiveAnyTrue(!ok)) { if (!ok) r = cc(cv); }
        put(c0 == 2u ? 4u : c0, r);
        if (ok) approx_bit(c0 == 2u ? 4u : c0);
    }
#else
    [loop] for (uint c0 = 0; c0 < 3u + in_zl(); ++c0) put(c0 == 2u ? 4u : c0, cc(c3(C, c0)));
#endif
    const Tex tDd = tex(R_DD), tDs = tex(R_DS);
    put(2u, min(tx(tDd, xc, yc, 1u), amax)); put(3u, min(tx(tDd, xc, yc, 2u), amax));
    put(29u, min(tx(tDd, xc, yc, 0u), amax));
    put(6u, min(tx(tDs, xc, yc, 0u), amax)); put(7u, min(tx(tDs, xc, yc, 1u), amax));
    put(10u, min(tx(tDs, xc, yc, 2u), amax));
    const Tex tN = tex(R_N);
    float3 nrm = float3(tx(tN, xc, yc, 0u), tx(tN, xc, yc, 1u), tx(tN, xc, yc, 2u));
    if ((flags & F_NRMDEC) != 0u) nrm = float3(M(2.0f, S(nrm.x, 0.5f)), M(2.0f, S(nrm.y, 0.5f)), M(2.0f, S(nrm.z, 0.5f)));
    put(11u, min(nrm.x, amax)); put(14u, min(nrm.y, amax)); put(15u, min(nrm.z, amax));
    float rough;
    if ((flags & F_RGHPK) != 0u) rough = tx(tN, xc, yc, 3u);
    else {
        const Tex tR = tex(R_R);
        const bool pad = X >= Wr || Y >= Hr;
        const uint ly = ((uint)Y + KW(K_SY)) & 7u;
        if (pad && ly < 4u) rough = (uint)(X + tR.ox) < tR.w ? tx(tR, X, yc, 0u) : 0.0f;
        else rough = tx(tR, xc, yc, 0u);
    }
    put(18u, min(rough, amax));

    // 52.3.3 depth neighbourhood
    const Tex tZ = tex(R_Z);
    const bool rev = (flags & F_REV) != 0u;
    int2 nT = int2(xc, yc), fT = nT;
    float bn = tx(tZ, xc, yc, 0u), bf = bn;
#if RR_IN_DLANE
    // a14 C8 (in-depth-window-lanes): each lane loads its own column (rows yc - 1, yc + 1; the centre is bn) and takes the
    // x - 1 / x + 1 columns from the neighbour lanes when they hold exactly that column and row (else it loads them):
    // the same nine values in the same k9 order, the same comparisons
    {
        const int ym = clamp(yc - 1, 0, Hr - 1), yp = clamp(yc + 1, 0, Hr - 1);
        const float zo0 = tx(tZ, xc, ym, 0u), zo1 = bn, zo2 = tx(tZ, xc, yp, 0u);
        const uint ln = WaveGetLaneIndex(), nl = WaveGetLaneCount();
        const uint4 act = WaveActiveBallot(true);
        const uint lL = ln == 0u ? 0u : ln - 1u, lR = ln + 1u < nl ? ln + 1u : ln;
        const bool aL = ((act[lL >> 5] >> (lL & 31u)) & 1u) != 0u, aR = ((act[lR >> 5] >> (lR & 31u)) & 1u) != 0u;
        const int xL = clamp(xc - 1, 0, Wr - 1), xR = clamp(xc + 1, 0, Wr - 1);
        const int nxL = WaveReadLaneAt(xc, lL), nyL = WaveReadLaneAt(yc, lL), nxR = WaveReadLaneAt(xc, lR), nyR = WaveReadLaneAt(yc, lR);
        float zL0 = WaveReadLaneAt(zo0, lL), zL1 = WaveReadLaneAt(zo1, lL), zL2 = WaveReadLaneAt(zo2, lL);
        float zR0 = WaveReadLaneAt(zo0, lR), zR1 = WaveReadLaneAt(zo1, lR), zR2 = WaveReadLaneAt(zo2, lR);
        if (xL == xc) { zL0 = zo0; zL1 = zo1; zL2 = zo2; }
        else if (!(aL && lL != ln && nxL == xL && nyL == yc)) { zL0 = tx(tZ, xL, ym, 0u); zL1 = tx(tZ, xL, yc, 0u); zL2 = tx(tZ, xL, yp, 0u); }
        if (xR == xc) { zR0 = zo0; zR1 = zo1; zR2 = zo2; }
        else if (!(aR && lR != ln && nxR == xR && nyR == yc)) { zR0 = tx(tZ, xR, ym, 0u); zR1 = tx(tZ, xR, yc, 0u); zR2 = tx(tZ, xR, yp, 0u); }
        const float zk[8] = {zL0, zo0, zR0, zL1, zR1, zL2, zo2, zR2};
        const int qxk[8] = {xL, xc, xR, xL, xR, xL, xc, xR};
        const int qyk[8] = {ym, ym, ym, yc, yc, yp, yp, yp};
        [unroll] for (uint k = 0; k < 8u; ++k) {
            const float z = zk[k];
            if (rev ? (z > bn) : (z < bn)) { bn = z; nT = int2(qxk[k], qyk[k]); }
            if (rev ? (z < bf) : (z > bf)) { bf = z; fT = int2(qxk[k], qyk[k]); }
        }
    }
#else
    [loop] for (uint k9 = 1; k9 < 9u + in_zl(); ++k9) {
        const int dx = (int)(k9 == 1u || k9 == 4u || k9 == 6u) * -1 + (int)(k9 == 3u || k9 == 5u || k9 == 8u);
        const int dy = (k9 <= 3u) ? -1 : (k9 <= 5u ? 0 : 1);
        const int qx = clamp(xc + dx, 0, Wr - 1), qy = clamp(yc + dy, 0, Hr - 1);
        const float z = tx(tZ, qx, qy, 0u);
        if (rev ? (z > bn) : (z < bn)) { bn = z; nT = int2(qx, qy); }
        if (rev ? (z < bf) : (z > bf)) { bf = z; fT = int2(qx, qy); }
    }
#endif
    const Tex tV = tex(R_V);
    const float2 mO = mv(tV, xc, yc), mN = mv(tV, nT.x, nT.y), mF = mv(tV, fT.x, fT.y);

    // 52.7.1 the second vector
    float2 m2 = mO;
    const uint mode = (flags >> 5) & 3u;
    if (mode == 1u) m2 = mv(tex(R_SV), xc, yc);
    else if (mode == 2u) {
        const Tex tH = tex(R_HD);
        const float px = A((float)(xc + tZ.ox), 0.5f), py = A((float)(yc + tZ.oy), 0.5f);
        float2 r;
        float hd = tx(tH, xc, yc, 0u);
        if ((flags & F_HDF) != 0u) hd = hd_run4(tH, X, yc, hd);   // wave-uniform branch
#if !(RR_IN_SKIP & 1)
        if (recon(tx(tZ, xc, yc, 0u), nrm, hd, px, py, r)) m2 = r;
#endif
    }

    // 52.3.2 history features: the four taps' positions first, the fallback (own colour) only when needed
    const float cxp = S(A((float)X, 0.5f), jx), cyp = S(A((float)Y, 0.5f), jy);
    float2 pos[4];
    bool inv[4];
    bool anyinv = false;
    [unroll] for (uint tp = 0; tp < 4u; ++tp) {
        const float2 mh = tp == 0u ? mO : (tp == 1u ? mN : (tp == 2u ? mF : m2));
        pos[tp] = float2(FM(cxp, rWhWr, mh.x), FM(cyp, rHhHr, mh.y));
        inv[tp] = or(reset, outside(pos[tp].x, pos[tp].y));
        anyinv = or(anyinv, inv[tp]);
    }
    const float gh = SY(S_GH);
    float3 fb = float3(0.0f, 0.0f, 0.0f);
    if (anyinv) {
        [loop] for (uint c1 = 0; c1 < 3u + in_zl(); ++c1) {
            const float v = clipf(M(gh, ccf(c3(C, c1))));
            fb = float3(c1 == 0u ? v : fb.x, c1 == 1u ? v : fb.y, c1 == 2u ? v : fb.z);
        }
    }
#if RR_IN_DEDUP
    // clean/rr-f-lw-zeros: a Hc1 tap (near, far, m2) whose position has the own tap's bits in every lane of the wave
    // is the same computation as the own Hc1 sample (sm 0): the same texture, position, validity and fallback colour,
    // and the same dither (52.4: token position and fc only), so its three codes, values and certification bits are
    // the own sample's; the wave copies them instead of recomputing (wave-uniform branch; in_flag is unchanged,
    // sm 0 already set any mark the copy would set). Exact for every input (NaN and signed-zero positions compare by
    // bits, so they only match a bit-identical position).
    const bool dt1 = WaveActiveAllTrue(in_same2(pos[1], pos[0])), dt2 = WaveActiveAllTrue(in_same2(pos[2], pos[0])),
               dt3 = WaveActiveAllTrue(in_same2(pos[3], pos[0]));
    float3 sv0 = float3(0.0f, 0.0f, 0.0f);
    uint3 sc0 = uint3(0u, 0u, 0u);
    uint sk0 = 0u;
#endif
    [loop] for (uint sm = 0; sm < ((RR_IN_SKIP & 2) ? 0u : 5u) + in_zl(); ++sm) {
        const uint tap = sm <= 1u ? 0u : sm - 1u;
        const bool iv = tap == 0u ? inv[0] : (tap == 1u ? inv[1] : (tap == 2u ? inv[2] : inv[3]));
        const float2 ps = tap == 0u ? pos[0] : (tap == 1u ? pos[1] : (tap == 2u ? pos[2] : pos[3]));
#if RR_IN_DEDUP
        const bool dup = sm >= 2u && (tap == 1u ? dt1 : (tap == 2u ? dt2 : dt3));
        [branch] if (dup) {
            [loop] for (uint cd = 0; cd < 3u + in_zl(); ++cd) {
                put_code(HK[sm * 3u + cd], in_c3u(sc0, cd), c3(sv0, cd));
                if (((sk0 >> cd) & 1u) != 0u && !iv) approx_bit(HK[sm * 3u + cd]);
            }
            continue;
        }
#endif
        float3 h = float3(0.0f, 0.0f, 0.0f);
#if RR_IN_SKIP & 32
        if (!iv) h = float3(ps.x, ps.y, ps.x + ps.y) * 0.001f;
#else
        if (!iv) h = ph_sample(sm == 1u ? 1u : 0u, M(ps.x, KF(K_HCSX)), M(ps.y, KF(K_HCSY)));
#endif
        [loop] for (uint ch = 0; ch < 3u + in_zl(); ++ch) {
#if RR_IN_SKIP & 64
            const float v = iv ? c3(fb, ch) : c3(h, ch);
            put(HK[sm * 3u + ch], v);
#elif RR_IN_HCERT && !RR_IN_EXACT
            {
                float va;
                bool okc;
#if RR_IN_IVSKIP
                // a14 C44: every lane of the wave invalid for this tap: hist_code would return quant(fbv) certified
                uint code;
                [branch] if (WaveActiveAllTrue(iv)) { va = c3(fb, ch); code = quant(va, dith); okc = true; }
                else code = hist_code(c3(h, ch), iv, c3(fb, ch), va, okc);
#else
                uint code = hist_code(c3(h, ch), iv, c3(fb, ch), va, okc);
#endif
#if RR_IN_HFB
                [branch] if (WaveActiveAnyTrue(!okc)) {
                    if (!okc) {
                        va = clipf(M(ccf(hdec(c3(h, ch))), gh));
                        code = quant(va, dith);
                    }
                }
#else
                in_flag |= okc ? 0u : 1u;
#endif
                put_code(HK[sm * 3u + ch], code, va);
                if (okc && !iv) approx_bit(HK[sm * 3u + ch]);
#if RR_IN_DEDUP
                if (sm == 0u) { sv0 = in_set3(sv0, ch, va); sc0 = in_set3u(sc0, ch, code); sk0 |= (okc ? 1u : 0u) << ch; }
#endif
            }
#else
            const float v = iv ? c3(fb, ch) : clipf(M(ccf(hdec(c3(h, ch))), gh));
#if RR_IN_DEDUP
            const uint code = quant(v, dith);     // put(k, v) = put_code(k, quant(v, dith), v)
            put_code(HK[sm * 3u + ch], code, v);
            if (sm == 0u) { sv0 = in_set3(sv0, ch, v); sc0 = in_set3u(sc0, ch, code); }
#else
            put(HK[sm * 3u + ch], v);
#endif
#endif
        }
    }
    // 52.3.6 hidden features (near, m2), 52.3.7 motion differences
    const float dq = (flags & F_DIS) != 0u ? tx(tex(R_Q), xc, yc, 0u) : 0.0f;
#if RR_IN_DEDUP && !(RR_IN_HIDCERT && !RR_IN_EXACT)
    const bool dth = WaveActiveAllTrue(in_same2(pos[3], pos[1]));
    float4 hd0 = float4(0.0f, 0.0f, 0.0f, 0.0f);
#endif
#if RR_IN_MDD
    const bool mdd = WaveActiveAllTrue(in_same2(mN, m2));
    float mdr1 = 0.0f;
    bool mda1 = false;
#endif
    [loop] for (uint hv = 0; hv < ((RR_IN_SKIP & 4) ? 0u : 2u) + in_zl(); ++hv) {
#if RR_IN_HIDCERT && !RR_IN_EXACT
        if (reset) {
            put(HIDK[4u * hv], 0.0f); put(HIDK[4u * hv + 1u], 0.0f); put(HIDK[4u * hv + 2u], 0.0f);
            put(HIDK[4u * hv + 3u], 0.0f);
        } else hid_codes(hv == 0u ? pos[1] : pos[3], HIDK[4u * hv], HIDK[4u * hv + 1u], HIDK[4u * hv + 2u],
                         HIDK[4u * hv + 3u]);
#elif RR_IN_DEDUP
        // clean/rr-f-lw-zeros: hid(m2) at a position with hid(near)'s bits in every lane is hid(near) (hidf depends on
        // the position, reset and per-frame constants only): reused, not recomputed (wave-uniform branch)
        float4 hd;
        [branch] if (hv != 0u && dth) hd = hd0;
        else hd = hidf(hv == 0u ? pos[1] : pos[3]);
        hd0 = hd;
        put(HIDK[4u * hv], hd.x); put(HIDK[4u * hv + 1u], hd.y); put(HIDK[4u * hv + 2u], hd.z);
        put(HIDK[4u * hv + 3u], hd.w);
#else
        const float4 hd = hidf(hv == 0u ? pos[1] : pos[3]);
        put(HIDK[4u * hv], hd.x); put(HIDK[4u * hv + 1u], hd.y); put(HIDK[4u * hv + 2u], hd.z);
        put(HIDK[4u * hv + 3u], hd.w);
#endif
#if RR_IN_MDD
        float mdv;
        [branch] if (mdd) {
            if (hv == 0u) mdv = mdiff2(X, Y, mN, dq, mdr1, mda1);
            else { mdv = mdr1; md_approx = mda1; }
        } else mdv = mdiff(X, Y, hv == 0u ? mN : m2, hv != 0u, dq);
        put(hv == 0u ? 19u : 44u, mdv);
#else
        put(hv == 0u ? 19u : 44u, mdiff(X, Y, hv == 0u ? mN : m2, hv != 0u, dq));
#endif
        if (md_approx) approx_bit(hv == 0u ? 19u : 44u);
    }

    // 52.3.5 guides
    [loop] for (uint gd = 0; gd < ((RR_IN_SKIP & 8) ? 0u : 3u) + in_zl(); ++gd) {
        float g = 0.0f;
#if RR_IN_NHL && !RR_IN_EXACT
        if (gd == 0u && (flags & F_B) != 0u) {
            const Tex tB = tex(R_B);
            const float db = mad(-E, tx(tB, xc, yc, 2u), min(in_nan0(E * C.z), F16MAX));
            const float dr = mad(-E, tx(tB, xc, yc, 0u), min(in_nan0(E * C.x), F16MAX));
            const float dg = mad(-E, tx(tB, xc, yc, 1u), min(in_nan0(E * C.y), F16MAX));
            g = nl_slc(mad(db, SY(S_LWRB), mad(dr, SY(S_LWRB), dg * SY(S_LWG))));
        } else if (gd == 1u && (flags & F_G1) != 0u) g = nl_slc(E * tx(tex(R_G1), xc, yc, 0u));
        else if (gd == 2u && (flags & F_G2) != 0u) g = nl_slc(E * tx(tex(R_G2), xc, yc, 0u));
        put_guide(gd, min(in_nan0(g), SY(S_FCLIP)));
    }
#elif RR_IN_L2CERT && !RR_IN_EXACT
        float sv = 0.0f;
        bool use = false;
        if (gd == 0u && (flags & F_B) != 0u) {
            const Tex tB = tex(R_B);
            const float db = FM(-E, tx(tB, xc, yc, 2u), min(in_nan0(M(E, C.z)), F16MAX));
            const float dr = FM(-E, tx(tB, xc, yc, 0u), min(in_nan0(M(E, C.x)), F16MAX));
            const float dg = FM(-E, tx(tB, xc, yc, 1u), min(in_nan0(M(E, C.y)), F16MAX));
            sv = FM(db, SY(S_LWRB), FM(dr, SY(S_LWRB), M(dg, SY(S_LWG))));
            use = true;
        } else if (gd == 1u && (flags & F_G1) != 0u) { sv = M(E, tx(tex(R_G1), xc, yc, 0u)); use = true; }
        else if (gd == 2u && (flags & F_G2) != 0u) { sv = M(E, tx(tex(R_G2), xc, yc, 0u)); use = true; }
        float Dg = 0.0f;
        if (use) g = slc_c(sv, Dg);
        const bool okg = cell_ok(clipf(g), Dg);
        [branch] if (WaveActiveAnyTrue(!okg)) { if (!okg) g = slc(sv); }
        put_guide(gd, clipf(g));
        if (use && okg) approx_bit(30u + gd);
    }
#else
        if (gd == 0u && (flags & F_B) != 0u) {
            const Tex tB = tex(R_B);
            const float db = FM(-E, tx(tB, xc, yc, 2u), min(in_nan0(M(E, C.z)), F16MAX));
            const float dr = FM(-E, tx(tB, xc, yc, 0u), min(in_nan0(M(E, C.x)), F16MAX));
            const float dg = FM(-E, tx(tB, xc, yc, 1u), min(in_nan0(M(E, C.y)), F16MAX));
            g = slc(FM(db, SY(S_LWRB), FM(dr, SY(S_LWRB), M(dg, SY(S_LWG)))));
        } else if (gd == 1u && (flags & F_G1) != 0u) g = slc(M(E, tx(tex(R_G1), xc, yc, 0u)));
        else if (gd == 2u && (flags & F_G2) != 0u) g = slc(M(E, tx(tex(R_G2), xc, yc, 0u)));
        put_guide(gd, clipf(g));
    }
#endif
#if RR_IN_CSLOT == 2
    in_x3 = X3;                         // 33 and 36 are quantised by gs_code_word at the stores (same values, same dither)
#if RR_IN_DBG
    if (dbg_base != 0u) { wg_st32(dbg_base, 33u * 4u, asuint(clipf(0.0f))); wg_st32(dbg_base, 36u * 4u, asuint(X3)); }
#endif
#else
    put(33u, clipf(0.0f));
    put(36u, X3);
#endif
#if RR_IN_DBG
    if (dbg_base != 0u) {
        [loop] for (uint z0 = 34u; z0 < 64u; ++z0)
            if (z0 == 34u || z0 == 35u || z0 == 38u || z0 == 39u || z0 == 42u || z0 == 43u || z0 == 46u || z0 == 47u ||
                z0 == 50u || z0 == 51u || z0 == 53u || z0 >= 54u) wg_st32(dbg_base, z0 * 4u, 0u);
        const float dv[8] = {mO.x, mO.y, mN.x, mN.y, mF.x, mF.y, m2.x, m2.y};
        [unroll] for (uint z1 = 0; z1 < 8u; ++z1) wg_st32(dbg_base, (64u + z1) * 4u, asuint(dv[z1]));
        wg_st32(dbg_base, 72u * 4u, dbg_approx.x);   // features with an approximate f32 value (certified codes)
        wg_st32(dbg_base, 73u * 4u, dbg_approx.y);
    }
#endif

    // stores: codes [Ht][Wt][64], hold [Ht][Wt][4] f16
    const uint pc = (uint)wg_uniform((int)KW(K_CODES));
    [unroll] for (uint w4 = 0; w4 < 4u; ++w4) {
        const uint b = (4u * w4) * WG_BLOCK + lane_;
        wg_st128(pc, t * 64u + w4 * 16u, uint4(gs_code_word(4u * w4, lane_), gs_code_word(4u * w4 + 1u, lane_),
                                                gs_code_word(4u * w4 + 2u, lane_), gs_code_word(4u * w4 + 3u, lane_)));
    }
    const uint h0 = TF2H(min(in_nan0(M(E, C.x)), F16MAX)), h1 = TF2H(min(in_nan0(M(E, C.y)), F16MAX));
    const uint h2b = TF2H(min(in_nan0(M(E, C.z)), F16MAX));
    const uint ph = (uint)wg_uniform((int)KW(K_HOLD));
    wg_st64(ph, t * 8u, uint2(h0 | (h1 << 16), h2b));

    // 52.7 surfaces: render-area tokens, at the motion-vector texel, out-of-image writes discarded (the conversions
    // before the branch: the driver channel wants uniform control flow)
    const uint2 vPM = uint2(TF2H(mN.x) | (TF2H(mN.y) << 16), TF2H(m2.x) | (TF2H(m2.y) << 16));
    const uint vPe = TF2H(mF.x) | (TF2H(mF.y) << 16), vPn = TF2H(mO.x) | (TF2H(mO.y) << 16);
    if (X < Wr && Y < Hr) {
        const int sx_ = X + tV.ox, sy_ = Y + tV.oy;
        if (sx_ < Wr && sy_ < Hr) {
            const uint idx = (uint)(sy_ * Wr + sx_);
            const uint pPM = (uint)wg_uniform((int)KW(K_PMW));
            const uint pPe = (uint)wg_uniform((int)KW(K_PE));
            const uint pPn = (uint)wg_uniform((int)KW(K_PN));
            wg_st64(pPM, idx * 8u, vPM);
            wg_st32(pPe, idx * 4u, vPe);
            wg_st32(pPn, idx * 4u, vPn);
        }
    }
#if !RR_IN_EXACT && !RR_IN_LITE
#if RR_IN_MARK_ALL
    in_flag = 1u;                       // test variant: every token goes through the replay
#endif
    // append the marked tokens of the wave to the replay list (one atomic per wave)
    const bool mk = in_flag != 0u;
    const uint nm = WaveActiveCountBits(mk);
    if (nm != 0u) {
        uint base = 0u;
        if (WaveIsFirstLane()) base = wg_atomic_add(KW(K_MARK), 0u, nm);
        base = WaveReadLaneFirst(base);
        if (mk) wg_st32(KW(K_MARK), 4u + (base + WavePrefixCountBits(mk)) * 4u, t);
#if RR_IN_DBG
        if (mk && KW(K_DIAG) != 0u) wg_atomic_add(KW(K_DIAG), 8u, 1u);
#endif
    }
#endif
#if RR_IN_DBG
    if (KW(K_DIAG) != 0u) {
        if (in_ziv != 0u) wg_atomic_add(KW(K_DIAG), 0u, in_ziv);
        if (in_slow != 0u) wg_atomic_add(KW(K_DIAG), 4u, in_slow);
    }
#endif
}

// f2h_ags against wg_f2h for every f32 pattern (gpu/input_stage.py --f2hcheck): pK = [base, count, pout];
// pout [0] differing non-NaN patterns, [1] patterns checked, [2] largest differing pattern + 1
WG_KERNEL(rr_input_f2hcheck)
{
    WG_SETUP();
    const uint base = KD(0), count = KD(1), po = KD(2);
    const uint i = wg_block_id() * WG_BLOCK + wg_lane_id();
    const uint u = base + min(i, count - 1u);
    const float x = asfloat(u);
#if RR_F2HC_HW   // round 13: the driver's op 0x1C (in_f2h_hw) against wg_f2h, NaN included
    const uint a = in_f2h_hw(x), b = wg_f2h(x);
    const uint bad = (i < count && a != b) ? 1u : 0u;
#else
    const uint a = f2h_ags(x), b = wg_f2h(x);
    const uint bad = (i < count && !isnan(x) && a != b) ? 1u : 0u;
#endif
    const uint sb = WaveActiveSum(bad), lm = WaveActiveMax(bad != 0u ? u + 1u : 0u);
    const uint nd = WaveActiveCountBits(i < count);
    if (WaveIsFirstLane()) {
        if (sb != 0u) wg_atomic_add(po, 0u, sb);
        wg_atomic_add(po, 4u, nd);
        if (lm != 0u) wg_atomic_max(po, 8u, lm);
    }
}

// the replay list's count to 0 before the fast pass (one lane; pK as for the features kernels)
WG_KERNEL(rr_input_mark_clear)
{
    WG_SETUP();
    if (wg_block_id() == 0u && wg_lane_id() == 0u) wg_st32(KD(K_MARK), 0u, 0u);
}

// ---- probe: the scalar helpers on arrays (gpu/input_stage.py probe) ----------------------------------------------
// pK points at [op, n, pa, pb, pc, pout, plog, pexp]; op 0 quant(a, d = b bits), 1 log2, 2 exp2, 3 sqrt, 4 rcp, 5 mul,
// 6 fma, 7 add, 8 div, 9 f16 bits RNE of a, 10 unorm10 of the code a bits, 11 snorm8 of the byte a bits
WG_KERNEL(rr_input_probe)
{
    WG_SETUP();
    const uint op = KD(0), n = KD(1), pa = KD(2), pb = KD(3), pc = KD(4), po = KD(5);
    in_plog = KD(6); in_pexp = KD(7);
    const uint i = wg_block_id() * WG_BLOCK + wg_lane_id();
    if (i >= n) return;
    const uint ua = wg_ld32(pa, i * 4u);
    const float a = asfloat(ua), b = asfloat(wg_ld32(pb, i * 4u)), c = asfloat(wg_ld32(pc, i * 4u));
    const float fa = in_ftz(a), fb = in_ftz(b), fc_ = in_ftz(c);
    uint r = 0u;
    if (op == 0u) r = quant(a, wg_ld32(pb, i * 4u));
    else if (op == 1u) r = asuint(log2_cr(fa));
    else if (op == 2u) r = asuint(exp2_cr(fa));
    else if (op == 3u) r = asuint(in_sqrt(fa));
    else if (op == 4u) r = asuint(RCP(fa));
    else if (op == 5u) r = asuint(M(fa, fb));
    else if (op == 6u) r = asuint(FM(fa, fb, fc_));
    else if (op == 7u) r = asuint(A(fa, fb));
    else if (op == 12u) r = asuint(S(fa, fb));      // a14 C33 probes: S, BW, MW, FM (fast-pass forms)
#if !RR_IN_EXACT
    else if (op == 13u) r = asuint(BW(fa, fb));
    else if (op == 14u) r = asuint(MW(fa, fb));
#endif
    else if (op == 8u) r = asuint(D(fa, fb));
    else if (op == 9u) r = wg_f2h(a);
    else if (op == 10u) r = asuint(in_un10(ua));
    else if (op == 11u) r = asuint(in_sn8(ua));
    wg_st32(po, i * 4u, r);
    if (in_ziv != 0u) wg_atomic_add(po, n * 4u, in_ziv);
    if (in_slow != 0u) wg_atomic_add(po, n * 4u + 4u, in_slow);
}

// ---- exhaustive cross-check of the fast paths (gpu/input_stage.py --xcheck) ---------------------------------------
// pK: [op, base, count, pout, plog, pexp]; lane i takes the f32 bit pattern base + i. op 0: log2_ff vs log2_dbl,
// 1: exp2_ff vs exp2_dbl, 2: RCP vs the binary64 quotient. pout words: [0] results that differ while neither path
// flags uncertainty, [1] fast-path flags, [2] binary64 unproven, [3] largest differing pattern + 1 (0: none),
// [4] evaluated inputs.
WG_KERNEL(rr_input_xcheck)
{
    WG_SETUP();
    const uint op = KD(0), base = KD(1), count = KD(2), po = KD(3);
    in_plog = KD(4); in_pexp = KD(5);
    const uint i = wg_block_id() * WG_BLOCK + wg_lane_id();
    uint mism = 0u, flg = 0u, unp = 0u, done = 0u;
    const uint u = base + i;
    if (i < count) {
        const float x = asfloat(u);
        if (op == 0u) {
            if ((u & 0x7fffffu) != 0u && (u >> 23) != 0u && (u >> 23) < 255u) {
                bool slow;
                const float a = log2_ff(u, slow);
                in_ziv = 0u;
                const float b = log2_dbl(x);
                flg = slow ? 1u : 0u;
                unp = in_ziv;
                mism = (!slow && in_ziv == 0u && asuint(a) != asuint(b)) ? 1u : 0u;
                done = 1u;
            }
        } else if (op == 1u) {
            if (isfinite(x) && x >= -126.0f && x < 128.0f && x != floor(x)) {
                bool slow;
                const float a = exp2_ff(x, slow);
                in_ziv = 0u;
                const float b = exp2_dbl(x);
                flg = slow ? 1u : 0u;
                unp = in_ziv;
                mism = (!slow && in_ziv == 0u && asuint(a) != asuint(b)) ? 1u : 0u;
                done = 1u;
            }
        } else if (op == 2u) {
            if ((u >> 23) != 0u && (u >> 23) < 255u) {
                mism = asuint(RCP(x)) != asuint(in_round(1.0 / (double)x)) ? 1u : 0u;
                done = 1u;
            }
        } else if (op == 3u) {
            if ((u >> 23) != 0u && (u >> 23) < 255u) {
                mism = asuint(in_sqrt(x)) != asuint(in_sqrt_dbl(x)) ? 1u : 0u;
                done = 1u;
            }
        }
    }
    const uint sm = WaveActiveSum(mism), sf = WaveActiveSum(flg), su = WaveActiveSum(unp), sd = WaveActiveSum(done);
    const uint lm = WaveActiveMax(mism != 0u ? u + 1u : 0u);
    if (WaveIsFirstLane()) {
        if (sm != 0u) wg_atomic_add(po, 0u, sm);
        if (sf != 0u) wg_atomic_add(po, 4u, sf);
        if (su != 0u) wg_atomic_add(po, 8u, su);
        if (lm != 0u) wg_atomic_max(po, 12u, lm);
        if (sd != 0u) wg_atomic_add(po, 16u, sd);
    }
}

// texel conversion of the temporal images, once per frame (52.1.4 step 3): pK points at [kind, n, src, dst];
// kind 0: RGB10A2 codes u16 [4] -> f32 [4] (code / 1023, alpha 0), kind 1: snorm8 [4] -> f32 [4] (max(code / 127, -1))
WG_KERNEL(rr_input_decode)
{
    WG_SETUP();
    const uint kind = KD(0), n = KD(1), src = KD(2), dst = KD(3);
    const uint i = wg_block_id() * WG_BLOCK + wg_lane_id();
    if (i >= n) return;
    float4 v;
    if (kind == 0u) {
        const uint2 c = wg_ld64(src, i * 8u);
        v = float4(in_un10(c.x), in_un10(c.x >> 16), in_un10(c.y), 0.0f);
    } else {
        const uint c = wg_ld32(src, i * 4u);
        v = float4(in_sn8(c), in_sn8(c >> 8), in_sn8(c >> 16), in_sn8(c >> 24));
    }
    wg_st128(dst, i * 16u, asuint(v));
}

// exact widening of n binary16 values to f32 (motion surfaces for the output stage's decoded-f32 inputs)
// pK points at [n, src, dst]
WG_KERNEL(rr_input_widen)
{
    WG_SETUP();
    const uint n = KD(0), src = KD(1), dst = KD(2);
    const uint i = wg_block_id() * WG_BLOCK + wg_lane_id();
    if (i < n) wg_st32(dst, i * 4u, asuint(f16tof32(wg_ld16(src, i * 2u))));
}

// ---- micro-benchmark of the scalar helpers (timing only, gpu/input_bench.py --micro) ----------------------------
// pK: [op, iters, pout, plog, pexp]; each lane evaluates iters inputs and stores the XOR of the result bits.
WG_KERNEL(rr_input_mbench)
{
    WG_SETUP();
    const uint op = KD(0), iters = KD(1), po = KD(2);
    in_plog = KD(3); in_pexp = KD(4);
    in_stage_tables();
    const uint i = wg_block_id() * WG_BLOCK + wg_lane_id();
    uint acc = 0u;
    [loop] for (uint k = 0; k < iters; ++k) {
        const float x = asfloat(0x3f800001u + ((i * 7919u + k * 104729u) & 0x3fffffu));   // [1, 1.5)
        float r;
        if (op == 0u) r = log2_cr(x);
        else if (op == 1u) r = exp2_cr(FN(x * 3.0f - 4.0f));
        else if (op == 2u) r = log2(x);
        else if (op == 3u) r = in_sqrt(x);
        else r = RCP(x);
        acc ^= asuint(r);
    }
    wg_st32(po, i * 4u, acc ^ in_flag);
}

// ---- exhaustive error of the hardware exp2 / log2 against the correctly rounded ones (gpu/input_stage.py --hwerr) ----
// pK: [op, base, count, pout]; lane i takes the f32 pattern base + i. op 0: exp2 over t in [-126, 128) (results normal or
// inf), 1: log2 over positive normal x. pout words: [0] max distance in f32 steps (key order) between FN(in_ftz(exp2(t)))
// / FN(log2(x)) and exp2_cr / log2_cr where both are finite, [1] inputs where exactly one of them is non-finite or
// zero, [2] evaluated inputs, [3] in_ziv (unproven exact roundings), [4] pattern + 1 of the largest distance.
uint hw_key(float f) { const uint u = asuint(f); return (u >> 31) != 0u ? ~u : (u | 0x80000000u); }
WG_KERNEL(rr_input_hwerr)
{
    WG_SETUP();
    const uint op = KD(0), base = KD(1), count = KD(2), po = KD(3);
    in_plog = KD(4); in_pexp = KD(5);
    const uint i = wg_block_id() * WG_BLOCK + wg_lane_id();
    uint dist = 0u, special = 0u, done = 0u;
    const uint u = base + i;
    if (i < count) {
        const float x = asfloat(u);
        float hw = 0.0f, cr = 0.0f;
        bool use = false;
        if (op == 0u) {
            use = isfinite(x) && x >= -126.0f && x < 128.0f;
            if (use) { hw = FN(in_ftz(exp2(x))); cr = exp2_cr(x); }
        } else {
            use = (u >> 31) == 0u && (u >> 23) != 0u && (u >> 23) < 255u;
            if (use) { hw = FN(log2(x)); cr = log2_cr(x); }
        }
        if (use) {
            done = 1u;
            const bool fh = isfinite(hw) && hw != 0.0f, fc = isfinite(cr) && cr != 0.0f;
            if (fh && fc) {
                const uint a = hw_key(hw), b = hw_key(cr);
                dist = a > b ? a - b : b - a;
            } else if (asuint(hw) != asuint(cr)) special = 1u;
        }
    }
    const uint md = WaveActiveMax(dist), ss = WaveActiveSum(special), sd = WaveActiveSum(done);
    const uint at = WaveActiveMax(dist == md && md != 0u ? u + 1u : 0u);
    if (WaveIsFirstLane()) {
        if (md != 0u) wg_atomic_max(po, 0u, md);
        if (ss != 0u) wg_atomic_add(po, 4u, ss);
        if (sd != 0u) wg_atomic_add(po, 8u, sd);
        if (md != 0u) wg_atomic_max(po, 16u, at);
    }
    if (in_ziv != 0u) wg_atomic_add(po, 12u, in_ziv);
}
