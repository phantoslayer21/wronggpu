// SPDX-License-Identifier: MIT
// Private borrowed-device qualification and Model M supplied-exposure helper.
// Original WrongGPU code; basis: own io_probe, exposure.hip exposure_pair_diet,
// spec 07/10 and the existing strict build arithmetic contract. No alternate I/O path.
// @wg-kernel native_driver_probe co=- block=32 wave=32 args=p defines=NATIVE_DRIVER_PROBE=1
// @wg-kernel exposure_pair_diet co=exposure.co block=32 wave=32 args=p,p,u,u,u,u defines=NATIVE_EXPOSURE_PAIR=1
// @wg-kernel exposure_pair_diet_direct co=exposure.co entry=exposure_pair_diet block=32 wave=32 args=p,p,u,u,u,u rs=direct defines=NATIVE_EXPOSURE_PAIR=1,WG_DIRECT_TEXTURES=1
#include "wg_platform.hlsli"
#if defined(WG_DIRECT_TEXTURES)
#include "native_direct.hlsli"
#endif
#if defined(NATIVE_DRIVER_PROBE)
WG_ARGS_BEGIN
  uint dst;
WG_ARGS_END
WG_KERNEL(native_driver_probe) {
  WG_SETUP();
  uint lane=wg_lane_id();
  v8f c=wg_splat((float)lane);
  v8f h=wg_wmma_f16(uint4(0x3c003c00u,0x3c003c00u,0x3c003c00u,0x3c003c00u),
                    uint4(0x40004000u,0x40004000u,0x40004000u,0x40004000u),c);
  v8f q=wg_wmma_fp8(uint2(0x38383838u,0x38383838u),uint2(0x40404040u,0x40404040u),c);
  [unroll] for(uint k=0;k<8u;++k) {
    wg_stf(dst,(k*32u+lane)*4u,h.v[k]);
    wg_stf(dst,1024u+(k*32u+lane)*4u,q.v[k]);
  }
}
#endif
#if defined(NATIVE_EXPOSURE_PAIR)
WG_ARGS_BEGIN
  uint ex,computed;
  float value;
  int use_computed,first;
  float scale;
WG_ARGS_END
WG_KERNEL(exposure_pair_diet) {
  WG_SETUP();
  if(wg_lane_id()!=0u) return;
#if defined(WG_DIRECT_TEXTURES)
  float x=use_computed!=0 ? (nd_exposure_mode==1u ? native_direct_exposure() : wg_ldf(computed,0u)) : value;
#else
  float x=use_computed!=0 ? wg_ldf(computed,0u) : value;
#endif
  float e=x*scale;
  float ep=first!=0 ? e : wg_ldf(ex,8u)*scale;
  float e1=e==0.0f?1.0f:e,ep1=ep==0.0f?1.0f:ep;
  // Preserve the original FP64 quotient and decimal-double constant, including
  // quotient ties near the normal/denormal boundary. No substitution by wg_div.
  float ratio=(float)((double)e1/(double)ep1);
  float dec=(float)(0.01/(double)e1);
  wg_st128(ex,0u,uint4(asuint(e),asuint(ep),asuint(x),asuint(ratio)));
  wg_stf(ex,16u,dec);
}
#endif
