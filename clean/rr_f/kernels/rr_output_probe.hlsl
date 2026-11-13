// SPDX-License-Identifier: MIT
// Probes of the fast output-stage arithmetic (rr_output_fast.hlsl) against the operation helpers of rr_output.hlsl
// (copied verbatim below as *_ref): exhaustive over every f32 input for the unary helpers and over every pair of f16
// values for the f16 helpers; sampled operand triples (stored by the host) for the f32 binary / ternary helpers.
#include "wg_platform.hlsli"
WG_ARGS_BEGIN
 uint in_ptr, out_ptr, n, mode;
WG_ARGS_END

// ---- rr_output.hlsl helpers (verbatim, suffix _ref) ----------------------------------------------------------------
float fence(float x) { return asfloat(asuint(x)^(0u-(wg_grid_>>31))); }
float ftz(float x) { uint u=asuint(x); return asfloat((u&0x7f800000u)==0 ? u&0x80000000u:u); }
float rounded(double x) {
 double mn=(double)asfloat(0x00800000u), boundary=mn-mn*0.000000059604644775390625;
 if(abs(x)<mn) { uint lo,hi; asuint(x,lo,hi); return asfloat((hi&0x80000000u)|(abs(x)>=boundary?0x00800000u:0u)); }
 return fence(ftz((float)x));
}
float M_ref(float a,float b) { return rounded((double)ftz(a)*(double)ftz(b)); }
float A_ref(float a,float b) { return fence(ftz(ftz(a)+ftz(b))); }
float F_ref(float a,float b,float c) {
 a=ftz(a); b=ftz(b); c=ftz(c);
 float r=fence(ftz(mad(a,b,c)));
 if((asuint(r)&0x7fffffffu)<=0x00800000u && isfinite(a) && isfinite(b) && isfinite(c)) {
   double p=(double)a*(double)b, cc=(double)c, sum=p+cc;
   double bv=sum-p, residual=(p-(sum-bv))+(cc-bv);
   double mn=(double)asfloat(0x00800000u), boundary=mn-mn*0.000000059604644775390625;
   if(sum!=0 && abs(sum)<=mn) {
     double er=sum<0?-residual:residual;
     bool normal=abs(sum)>boundary||(abs(sum)==boundary&&er>=0);
     r=asfloat((sum<0?0x80000000u:0u)|(normal?0x00800000u:0u));
   }
 }
 return r;
}
float X_ref(float x) { return fence(ftz(exp2(ftz(x)))); }
float L_ref(float x) { return fence(ftz(log2(ftz(x)))); }
float H_ref(float x) { return fence(wg_rn16(x)); }
float HA_ref(float a,float b) { return H_ref(A_ref(a,b)); }
float HM_ref(float a,float b) { return H_ref(M_ref(a,b)); }
float HF_ref(float a,float b,float c) { return fence(wg_h2f(wg_fma_f16(a,b,c))); }

// ---- candidates --------------------------------------------------------------------------------------------------
uint zb() { return 0u - (wg_grid_ >> 31); }
float fz(float x) { return asfloat(asuint(x) ^ zb()); }
half fzh(half x) { return asfloat16((uint16_t)(asuint16(x) ^ (uint16_t)zb())); }
uint f2h_hw(float x) { return wg_ags_call(0x1cu, 0u, 1u, asuint(x), 0u); }
float R_ref(float x) { return fence(ftz(wg_div(1.0f,ftz(x)))); }
// reciprocal by wg_div's iteration without the exponent scaling (operands of moderate magnitude)
float R_it(float b)
{
    float y = rcp(b);
    float e = mad(-b, y, 1.0f);
    y = mad(e, y, y);
    float r = mad(-b, y, 1.0f);
    float q = mad(r, y, y);
    r = mad(-b, q, 1.0f);
    return mad(r, y, q);
}
float R_new(float x)
{
    const uint ax = asuint(x) & 0x7fffffffu;
    const float q = R_it(x);
    // |x| outside [2^-125, 2^125]: the reference (FTZ): 0 -> inf, denormal -> inf, |x| >= 2^126 -> 0 (denormal result),
    // inf -> 0, NaN -> quieted NaN; between 2^125 and 2^126 the result is normal
    const uint sg = asuint(x) & 0x80000000u;
    uint sp = ax < 0x00800000u ? (sg | 0x7f800000u) : (ax > 0x7f800000u ? (asuint(x) | 0x400000u) : sg);
    sp = ax == 0x7e800000u ? sg | 0x00800000u : sp;                 // 1 / 2^126
    sp = ax >= 0x00800000u && ax < 0x7e800000u ? asuint(q) : sp;   // normal results: the iteration
    return fence(asfloat(sp));
}
// round 13 candidate: rf_rcp's single Newton step from the hardware reciprocal, with R_new's special values
float R_one(float x)
{
    const uint ax = asuint(x) & 0x7fffffffu;
    const float y = rcp(x);
    const float q = mad(mad(-x, y, 1.0f), y, y);
    const uint sg = asuint(x) & 0x80000000u;
    uint sp = ax < 0x00800000u ? (sg | 0x7f800000u) : (ax > 0x7f800000u ? (asuint(x) | 0x400000u) : sg);
    sp = ax == 0x7e800000u ? sg | 0x00800000u : sp;
    sp = ax >= 0x00800000u && ax < 0x7e800000u ? asuint(q) : sp;
    return fence(asfloat(sp));
}
float X_new(float x) { return fz(exp2(x)); }
float X_new2(float x) { return fz(ftz(exp2(x))); }
float L_new(float x) { return fz(log2(x)); }
float L_new2(float x) { return fz(ftz(log2(x))); }
float H_new(float x) { return fz(f16tof32(f2h_hw(x))); }
half hof(uint b) { return asfloat16((uint16_t)b); }
float fh(half h) { return f16tof32((uint)asuint16(h)); }

void count(uint k, uint index)
{
    uint o;
    wg_mem_gc.InterlockedAdd(out_ptr + k * 4u, 1u, o);
    wg_mem_gc.InterlockedMin(out_ptr + 64u + k * 4u, index, o);
}

// mode 0: unary helpers over all 2^32 f32 inputs; mode 1: f16 pair helpers over all 2^32 (a, b) f16 pairs.
// Each lane covers 256 consecutive indices from (group * WG_BLOCK + lane) * 256.
WG_KERNEL(rr_oprobe_exhaustive)
{
    WG_SETUP();
    const uint base = (wg_block_id() * WG_BLOCK + wg_lane_id()) * 256u;
    for (uint j = 0; j < 256u; ++j) {
        const uint i = base + j;
        if (mode == 0u) {
            const float x = asfloat(i);
            if (asuint(X_ref(x)) != asuint(X_new(x))) count(0, i);
            if (asuint(X_ref(x)) != asuint(X_new2(x))) count(1, i);
            if (asuint(L_ref(x)) != asuint(L_new(x))) count(2, i);
            if (asuint(L_ref(x)) != asuint(L_new2(x))) count(3, i);
            if (asuint(H_ref(x)) != asuint(H_new(x))) count(4, i);
            if (asuint(min(max(ftz(x), 0.0f), 1.0f)) != asuint(min(max(x, 0.0f), 1.0f))) count(12, i);
            if (asuint(R_ref(x)) != asuint(R_new(x))) count(14, i);
            if (asuint(R_ref(x)) != asuint(R_one(x))) count(15, i);
            const float lg = log2(x);
            if (lg != 0 && abs(lg) < asfloat(0x30800000u)) count(13, i);   // nonzero |log2 x| below 2^-30
        } else {
            const float a = f16tof32(i & 0xffffu), b = f16tof32(i >> 16);
            const half ha = hof(i & 0xffffu), hb = hof(i >> 16);
            if (asuint(HM_ref(a, b)) != asuint(fh(fzh(ha * hb)))) count(5, i);
            if (asuint(HA_ref(a, b)) != asuint(fh(fzh(ha + hb)))) count(6, i);
            // HF with the third operand a fixed function of the pair (sampled third operand)
            const uint cb = (i * 2654435761u) >> 16;
            const float c = f16tof32(cb);
            if (asuint(HF_ref(a, b, c)) != asuint(fh(fzh(mad(ha, hb, hof(cb)))))) {
                const uint nn = (uint)isnan(a) + (uint)isnan(b) + (uint)isnan(c);
                count(nn == 0u ? 7u : (nn == 1u ? 10u : 11u), i);
            }
            // M / A on f16-valued f32 operands (products and sums of f16 values in f32)
            if (asuint(M_ref(a, b)) != asuint(fz(a * b))) count(8, i);
            if (asuint(A_ref(a, b)) != asuint(fz(a + b))) count(9, i);
        }
    }
}

// mode 2: sampled f32 operand triples in_ptr[3 i .. 3 i + 2]; stores M, A, F reference and hardware results (6 words).
WG_KERNEL(rr_oprobe_triples)
{
    WG_SETUP();
    const uint i = wg_block_id() * WG_BLOCK + wg_lane_id();
    if (i >= n) return;
    const float a = asfloat(wg_ld32(in_ptr, i * 12u)), b = asfloat(wg_ld32(in_ptr, i * 12u + 4u)),
                c = asfloat(wg_ld32(in_ptr, i * 12u + 8u));
    wg_st128(out_ptr, i * 32u, uint4(asuint(M_ref(a, b)), asuint(fz(a * b)), asuint(A_ref(a, c)), asuint(fz(a + c))));
    wg_st128(out_ptr, i * 32u + 16u, uint4(asuint(F_ref(a, b, c)), asuint(fz(mad(a, b, c))), asuint(fz(mad(fz(a), fz(b), fz(c)))), 0u));
}
