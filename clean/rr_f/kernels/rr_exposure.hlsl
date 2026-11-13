// SPDX-License-Identifier: MIT
// Original spec-58 exposure pass. API-decoded f32 RGB(A) input; explicit synthetic/model symbols.
// Two-level binary64 reduction satisfies the published nonnegative-mean accuracy bound.
// RR_EX_SRV=1 (opt-in; WrongGPU's RR runtime, clean/rr_f/runtime): color == 0 reads the colour from the game's
// resource itself (SRV rr_ex_srv[0], register space 1: the API read conversion, the same values an f32 copy holds);
// image_width and channels are then unused. Default: unchanged.
#include "wg_platform.hlsli"
#ifndef RR_EX_SRV
#define RR_EX_SRV 0
#endif
#if RR_EX_SRV
Texture2D<float4> rr_ex_srv[16] : register(t0, space1);
#endif
WG_ARGS_BEGIN
    uint color, partials, result, image_width, channels, ox, oy, width, height, blocks;
    uint reciprocal_lo, reciprocal_hi;
    float previous;
    uint reset;
    float lw_rb, lw_g, lmax, key, mean_floor, exposure_floor, clamp_hi, clamp_lo, norm;
WG_ARGS_END

groupshared double ex_sum[WG_BLOCK];
float ex_fence(float x) { return asfloat(asuint(x) ^ (0u - (wg_grid_ >> 31))); }
float ex_ftz(float x) {
    uint u=asuint(x);
    return asfloat((u & 0x7f800000u)==0u ? u & 0x80000000u : u);
}
float ex_round(double value) {
    // Native f32 multiply AND binary64-to-f32 conversion can flush before
    // rounding on this driver. Make the interval rounding to min-normal explicit.
    double smallest=(double)asfloat(0x00800000u);
    double boundary=smallest-smallest*0.000000059604644775390625;
    if(abs(value)<smallest) {
        uint magnitude=abs(value)>=boundary ? 0x00800000u : 0u;
        uint lo,hi; asuint(value,lo,hi);
        return asfloat((hi & 0x80000000u)|magnitude);
    }
    return ex_fence(ex_ftz((float)value));
}
float ex_mul(float a,float b) {
    // Product of two f32 values has at most 48 significant bits: exact in f64.
    return ex_round((double)ex_ftz(a)*(double)ex_ftz(b));
}
float ex_add(float a,float b) { return ex_fence(ex_ftz(ex_ftz(a)+ex_ftz(b))); }
double ex_reduce(double value) {
    uint lane=wg_lane_id();
    ex_sum[lane]=value;
    GroupMemoryBarrierWithGroupSync();
    for(uint stride=WG_BLOCK/2;stride>0;stride>>=1) {
        if(lane<stride) ex_sum[lane]=ex_sum[lane]+ex_sum[lane+stride];
        GroupMemoryBarrierWithGroupSync();
    }
    return ex_sum[0];
}
#ifndef RR_EX_WAVE
#define RR_EX_WAVE 1
#endif
float3 ex_rgb(uint i) {
#if RR_EX_SRV
    if(color==0u) {
        float4 t=rr_ex_srv[0].Load(int3((int)(ox+i%width),(int)(oy+i/width),0));
        return t.xyz;
    }
#endif
    uint p=((oy+i/width)*image_width+ox+i%width)*channels;
    return float3(asfloat(wg_ld32(color,(p+0)*4)),asfloat(wg_ld32(color,(p+1)*4)),asfloat(wg_ld32(color,(p+2)*4)));
}
#if RR_EX_WAVE
// RR_EX_WAVE 1 (default; compile with WG_BLOCK 32): one wave per 256-pixel block, lane l holds the pixels l + 32 k
// (k = 0..7), so the block's binary64 tree (strides 128, 64, 32 in the lane's registers, 16 .. 1 across the wave) adds
// the same pairs in the same order as the 256-lane LDS tree (ex_reduce): the same partial sums, no LDS, no barriers.
// The products: a lane's three f32 products are the binary64 products rounded once (ex_mul) whenever none of them can
// underflow: every flushed channel value 0 or of magnitude >= 2^-100 and each weight 0 or >= 2^-26 in magnitude (then
// a nonzero product is >= 2^-126; overflow, inf and NaN give the same values in both). A wave with any other lane
// takes the binary64 path (ex_mul).
bool ex_safe(float x) { float f=abs(ex_ftz(x)); return or(f==0.0f, !(f<asfloat(0x0d800000u))); }
float ex_mulf(float a,float b) { return ex_fence(ex_ftz(a)*ex_ftz(b)); }
WG_KERNEL(rr_exposure_partial) {
    WG_SETUP();
    const uint lane=wg_lane_id();
    const uint base=wg_block_id()*256u+lane;
    float3 c[8];
    bool safe=and(or(abs(lw_rb)>=asfloat(0x32800000u), lw_rb==0.0f), or(abs(lw_g)>=asfloat(0x32800000u), lw_g==0.0f));
    [unroll] for(uint k=0;k<8u;++k) {
        const uint i=base+32u*k;
        c[k]=select(i<width*height, ex_rgb(min(i,width*height-1u)), float3(0.0f,0.0f,0.0f));
        safe=and(and(safe, ex_safe(c[k].x)), and(ex_safe(c[k].y), ex_safe(c[k].z)));
    }
    double w[8];
    if(WaveActiveAllTrue(safe)) {
        [unroll] for(uint k=0;k<8u;++k) {
            float v=ex_add(ex_add(ex_mulf(lw_rb,c[k].x),ex_mulf(lw_g,c[k].y)),ex_mulf(lw_rb,c[k].z));
            w[k]=select(base+32u*k<width*height, (double)min(max(v,0.0f),lmax), 0.0);
        }
    } else {
        [unroll] for(uint k=0;k<8u;++k) {
            float v=ex_add(ex_add(ex_mul(lw_rb,c[k].x),ex_mul(lw_g,c[k].y)),ex_mul(lw_rb,c[k].z));
            w[k]=select(base+32u*k<width*height, (double)min(max(v,0.0f),lmax), 0.0);
        }
    }
    // strides 128, 64, 32 (lane l + 32 k holds w[k])
    const double a0=w[0]+w[4], a1=w[1]+w[5], a2=w[2]+w[6], a3=w[3]+w[7];
    const double b0=a0+a2, b1=a1+a3;
    double t=b0+b1;
    [unroll] for(uint stride=16u;stride>0u;stride>>=1) {
        const double o=WaveReadLaneAt(t,min(lane+stride,31u));
        t=select(lane<stride, t+o, t);
    }
    if(lane==0) {
        uint lo,hi; asuint(t,lo,hi);
        wg_st64(partials,wg_block_id()*8,uint2(lo,hi));
    }
}
#else
WG_KERNEL(rr_exposure_partial) {
    WG_SETUP();
    uint i=wg_block_id()*WG_BLOCK+wg_lane_id();
    float w=0;
    if(i<width*height) {
#if RR_EX_SRV
        float r,g,b;
        if(color==0u) {
            float4 t=rr_ex_srv[0].Load(int3((int)(ox+i%width),(int)(oy+i/width),0));
            r=t.x; g=t.y; b=t.z;
        } else {
            uint p=((oy+i/width)*image_width+ox+i%width)*channels;
            r=asfloat(wg_ld32(color,(p+0)*4)); g=asfloat(wg_ld32(color,(p+1)*4)); b=asfloat(wg_ld32(color,(p+2)*4));
        }
#else
        uint p=((oy+i/width)*image_width+ox+i%width)*channels;
        float r=asfloat(wg_ld32(color,(p+0)*4));
        float g=asfloat(wg_ld32(color,(p+1)*4));
        float b=asfloat(wg_ld32(color,(p+2)*4));
#endif
        float v=ex_add(ex_add(ex_mul(lw_rb,r),ex_mul(lw_g,g)),ex_mul(lw_rb,b));
        w=min(max(v,0.0f),lmax);
    }
    double total=ex_reduce((double)w);
    if(wg_lane_id()==0) {
        uint lo,hi; asuint(total,lo,hi);
        wg_st64(partials,wg_block_id()*8,uint2(lo,hi));
    }
}
#endif
WG_KERNEL(rr_exposure_finish) {
    WG_SETUP();
    double total=0;
    for(uint i=wg_lane_id();i<blocks;i+=WG_BLOCK) {
        uint2 v=wg_ld64(partials,i*8);
        total=total+asdouble(v.x,v.y);
    }
    total=ex_reduce(total);
    if(wg_lane_id()!=0) return;
    double mean64=total*asdouble(reciprocal_lo,reciprocal_hi);
    float mean=ex_round(mean64);
    float raw=ex_mul(key,wg_div(1.0f,max(mean,mean_floor)));
    uint stored=wg_f2h(raw);
    float e=f16tof32(stored);
    if(e==0) e=1.0f;
    // Step 5 selects the input value unchanged unless its bits encode +/-zero.
    // An arbitrary f32 test input may be subnormal, although real EX is f16.
    // reset bit 1: the previous frame's E is read from its result buffer (word 0) at the arena offset given by the
    // bits of `previous` (connected frames); bit 0 is the reset of page 58.
    float prev=(reset&2u)!=0u ? asfloat(wg_ld32(asuint(previous),0u)) : previous;
    float ep=(reset&1u)!=0u || (asuint(prev)&0x7fffffffu)==0u ? 1.0f : prev;
    float inv=ex_ftz(wg_div(1.0f,e));
    float x3=0;
    if((reset&1u)==0u) {
        float ratio=ex_mul(max(e,exposure_floor),wg_div(1.0f,max(ep,exposure_floor)));
        float lg=ex_fence(ex_ftz(log2(ratio)));
        x3=ex_mul(min(max(lg,clamp_lo),clamp_hi),norm);
    }
    wg_st128(result,0,uint4(asuint(e),asuint(ep),asuint(inv),asuint(x3)));
    wg_st64(result,16,uint2(asuint(raw),asuint(mean)));
    uint lo,hi; asuint(mean64,lo,hi);
    wg_st64(result,24,uint2(lo,hi));
    wg_st32(result,32,stored);
}
