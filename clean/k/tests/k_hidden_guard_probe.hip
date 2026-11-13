// SPDX-License-Identifier: MIT
// Own synthetic audit; include the actual production helper and predicate.
#include "../kernels/k_block.hip"
#pragma clang fp contract(off)
#if !defined(__gfx1201__)
#error hidden guard bound is validated only for gfx1201
#endif
WG_FN u32 hash32(u32 x) { x ^= x>>16; x*=0x7feb352du; x^=x>>15; x*=0x846ca68bu; return x^(x>>16); }
WG_KERNEL_N(make_h,128)(float* dst,u32 n,u32 mode) {
 for(u32 i=wg_block_id()*128u+wg_lane_id();i<n;i+=4096u*128u){
  float h[4];
#pragma unroll
  for(int k=0;k<4;++k){float c[5];
#pragma unroll
   for(int j=0;j<5;++j){u32 v=hash32(i*29u+97u*k+13u*j);unsigned short bits=(unsigned short)v;
    if(mode==0) bits=(bits&0x8000u)|((bits&0x7fffu)%0x7c00u);
    if(mode==2) bits=(bits&0x8000u)|(bits&0x03ffu);
    if(mode==3){const unsigned short p[5]={0x7bff,0xfbff,0x0001,0x8000,0x3c00};bits=p[(i+j+k)%5];}
    c[j]=(float)__builtin_bit_cast(h16,bits);
   }
   h[k]=0.2f*((((c[0]+c[1])+c[2])+c[3])+c[4]);
  }
  if(mode==4){unsigned short hb=(unsigned short)(0x0400u+(i/33u)%0x37efu);float t=(float)__builtin_bit_cast(h16,hb);u32 pb=__builtin_bit_cast(u32,t)+0x1000u;pb+=(int)(i%33u)-16;double target=(double)__builtin_bit_cast(float,pb);h[1]=0.37f;h[2]=0.41f;h[3]=0.73f;double k=1.0+(double)h[1]*h[1]+(double)h[2]*h[2]+(double)h[3]*h[3];h[0]=(float)(target*__builtin_sqrt(k/(1.0-target*target)));if(i&1)h[0]=-h[0];}
#pragma unroll
  for(int k=0;k<4;++k)dst[4*i+k]=h[k];
 }
}

WG_KERNEL_N(hidden_marker,128)(u32* dst) {
 if(wg_block_id()==0 && wg_lane_id()==0) dst[0]=wgk_hidden_guard_arch_enabled()?0x1201u:0u;
}
WG_KERNEL_N(hidden_reference,128)(const float* src,h16* dst,u32 n) {
 for(u32 i=wg_block_id()*128u+wg_lane_id();i<n;i+=4096u*128u)
  wgk_store_hidden_exact(dst+4*i,src+4*i,wgk_hidden_norm2(src+4*i));
}
WG_KERNEL_N(hidden_candidate,128)(const float* src,h16* dst,u32* used_fast,u32 n) {
 for(u32 i=wg_block_id()*128u+wg_lane_id();i<n;i+=4096u*128u)
  used_fast[i]=wgk_store_hidden_guarded(dst+4*i,src+4*i)?1u:0u;
}
WG_KERNEL_N(compare_hidden,128)(const unsigned short* a,const unsigned short* b,const u32* used_fast,u32* out,u32 n) {
 u32 bad=0,fast=0,fall=0,invalid=0;
 for(u32 i=wg_block_id()*128u+wg_lane_id();i<n;i+=4096u*128u) {
  if(used_fast[i]==1u)++fast;else if(used_fast[i]==0u)++fall;else++invalid;
#pragma unroll
  for(int k=0;k<4;++k)if(a[4*i+k]!=b[4*i+k])++bad;
 }
 if(bad)__atomic_fetch_add(out,bad,__ATOMIC_RELAXED);
 if(fast)__atomic_fetch_add(out+1,fast,__ATOMIC_RELAXED);
 if(fall)__atomic_fetch_add(out+2,fall,__ATOMIC_RELAXED);
 if(invalid)__atomic_fetch_add(out+3,invalid,__ATOMIC_RELAXED);
}
WG_KERNEL_N(half_interval,128)(u32* out,u32 first,u32 n){u32 accepted=0,bad=0;
 for(u32 i=wg_block_id()*128u+wg_lane_id();i<n;i+=4096u*128u){u32 b=first+i;float p=__builtin_bit_cast(float,b);if(wgk_hidden_half_safe(p)){++accepted;h16 m=kh_round(p),lo=kh_round(__builtin_bit_cast(float,b-7u)),hi=kh_round(__builtin_bit_cast(float,b+7u));if(__builtin_bit_cast(unsigned short,m)!=__builtin_bit_cast(unsigned short,lo)||__builtin_bit_cast(unsigned short,m)!=__builtin_bit_cast(unsigned short,hi))++bad;}}
 if(accepted)__atomic_fetch_add(out,accepted,__ATOMIC_RELAXED);if(bad)__atomic_fetch_add(out+1,bad,__ATOMIC_RELAXED);
}

WG_KERNEL_N(hidden_rsq_native,128)(unsigned* dst,unsigned first,unsigned count) {
 for(unsigned i=wg_block_id()*128u+wg_lane_id();i<count;i+=128u*4096u){float x=__builtin_bit_cast(float,first+i);dst[i]=__builtin_bit_cast(unsigned,kf_rsqrt(x));}
}
WG_KERNEL_N(hidden_rsq_estimate,128)(unsigned* dst,unsigned first,unsigned count) {
 for(unsigned i=wg_block_id()*128u+wg_lane_id();i<count;i+=128u*4096u){float x=__builtin_bit_cast(float,first+i);dst[i]=__builtin_bit_cast(unsigned,__builtin_amdgcn_rsqf(x));}
}
WG_KERNEL_N(hidden_rsq_compare,128)(const unsigned* a,const unsigned* b,unsigned* out,unsigned first,unsigned count) {
 unsigned bins[5]={0,0,0,0,0},maxulp=0,maxabs=0,maxrel=0,firstbad=0xffffffffu;
 for(unsigned i=wg_block_id()*128u+wg_lane_id();i<count;i+=128u*4096u){
  unsigned aa=a[i],bb=b[i];
  if(aa==0||bb==0||aa>=0x7f800000u||bb>=0x7f800000u){++bins[4];firstbad=firstbad<first+i?firstbad:first+i;continue;}
  unsigned ulp=aa>bb?aa-bb:bb-aa;++bins[ulp<3?ulp:3];
  if(ulp){firstbad=firstbad<first+i?firstbad:first+i;maxulp=maxulp>ulp?maxulp:ulp;float av=__builtin_bit_cast(float,aa),bv=__builtin_bit_cast(float,bb),err=__builtin_fabsf(av-bv);unsigned ab=__builtin_bit_cast(unsigned,err),re=__builtin_bit_cast(unsigned,err/av);maxabs=maxabs>ab?maxabs:ab;maxrel=maxrel>re?maxrel:re;}
 }
 WG_LDS unsigned p[9][128];unsigned lane=wg_lane_id();
#pragma unroll
 for(int k=0;k<5;++k)p[k][lane]=bins[k];p[5][lane]=maxulp;p[6][lane]=maxabs;p[7][lane]=maxrel;p[8][lane]=firstbad;
 wg_barrier_lds();
#pragma unroll
 for(unsigned stride=64;stride;stride>>=1){if(lane<stride){
#pragma unroll
  for(int k=0;k<5;++k)p[k][lane]+=p[k][lane+stride];
#pragma unroll
  for(int k=5;k<8;++k)p[k][lane]=p[k][lane]>p[k][lane+stride]?p[k][lane]:p[k][lane+stride];
  p[8][lane]=p[8][lane]<p[8][lane+stride]?p[8][lane]:p[8][lane+stride];
 }wg_barrier_lds();}
 if(lane==0){
#pragma unroll
  for(int k=0;k<5;++k)if(p[k][0])__atomic_fetch_add(out+k,p[k][0],__ATOMIC_RELAXED);
#pragma unroll
  for(int k=5;k<8;++k)if(p[k][0])__atomic_fetch_max(out+k,p[k][0],__ATOMIC_RELAXED);
  if(p[8][0]!=0xffffffffu)__atomic_fetch_min(out+8,p[8][0],__ATOMIC_RELAXED);
 }
}
