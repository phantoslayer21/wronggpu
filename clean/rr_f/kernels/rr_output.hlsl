// SPDX-License-Identifier: MIT
// Original implementation of published spec 54 only; no resolve or learned data.
// Inputs are decoded f32 or packed head half4/byte4 (packed_head bit 0).
// packed_head bit 1: E, Ep, invE are read from the exposure pass's result buffer (rr_exposure.hlsl: E, Ep, invE, X3)
// at the arena offset given by the bits of the E argument (connected frame, no host round trip).
// Motion / history are decoded f32, or binary16 images: packed_head bit 2 motion (PM / Pe / Pn), bit 5 history (exact
// widening at the load, the same f16tof32 as the former widening pass rr_input_widen). Bit 3 (fast path): no linear /
// trace outputs.
// Outputs store actual half/code bits.
// RR_RT=1 (WrongGPU's RR runtime, clean/rr_f/runtime): the motion pair / eroded / non-dilated motion and the
// previous history are read as the binary16 images of page 53.1 (exact widening, the same values as the decoded f32
// buffers) also without packed_head bits 2 / 5; E / Ep / invE come from the exposure result (packed_head bit 1).
#include "wg_platform.hlsli"
#ifndef RR_RT
#define RR_RT 0
#endif
WG_ARGS_BEGIN
 uint color,gates,precision,candidates,motion_pair,motion_eroded,motion_non,hprev,table;
 uint linear_buf,ch,q,ph,trace;
 uint wr,hr,wt,wh,hh;
 float jx,jy,E_arg,Ep_arg,invE_arg;
 uint reset;
 float K1,K2n,K2x2n,K3,K4,K5,K7,K8,K10,K11,K12,K13;
 uint packed_head;
 uint scratch;   // rr_output_fast.hlsli: fallback list and render records (0 when only rr_output_main runs)
WG_ARGS_END
#ifndef OF_DEBUG
#define OF_DEBUG 0
#endif
#ifndef BAD_CODE
#define BAD_CODE 0
#endif
#ifndef BAD_ORDER
#define BAD_ORDER 0
#endif
#ifndef BAD_RESET
#define BAD_RESET 0
#endif
float fence(float x) { return asfloat(asuint(x)^(0u-(wg_grid_>>31))); }
float ftz(float x) { uint u=asuint(x); return asfloat((u&0x7f800000u)==0 ? u&0x80000000u:u); }
float rounded(double x) {
 double mn=(double)asfloat(0x00800000u), boundary=mn-mn*0.000000059604644775390625;
 if(abs(x)<mn) { uint lo,hi; asuint(x,lo,hi); return asfloat((hi&0x80000000u)|(abs(x)>=boundary?0x00800000u:0u)); }
 return fence(ftz((float)x));
}
float M(float a,float b) { return rounded((double)ftz(a)*(double)ftz(b)); }
float A(float a,float b) { return fence(ftz(ftz(a)+ftz(b))); }
float F(float a,float b,float c) {
 a=ftz(a); b=ftz(b); c=ftz(c);
 float r=fence(ftz(mad(a,b,c)));
 // Exact binary64 product plus error-free TwoSum residual locates the
 // underflow midpoint without the scaled-f32 double rounding ambiguity.
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
float R(float x) { return fence(ftz(wg_div(1.0f,ftz(x)))); }
float X(float x) { return fence(ftz(exp2(ftz(x)))); }
float L(float x) { return fence(ftz(log2(ftz(x)))); }
float S(float x) { return min(max(ftz(x),0.0f),1.0f); }
float C(float x) { return M(M(L(max(F(x,K4,1),K5)),asfloat(0x3e9a209bu)),K7); }
float D(float x) { float t=(x>K5||isnan(x))?(isnan(x)||isnan(K10)?asfloat(0x7fc00000u):min(x,K10)):K5; return A(X(M(M(t,K8),asfloat(0x40549a78u))),-1); }
float H(float x) { return fence(wg_rn16(x)); }
float HA(float a,float b) { return H(A(a,b)); }
float HM(float a,float b) { return H(M(a,b)); }
float HF(float a,float b,float c) { return fence(wg_h2f(wg_fma_f16(a,b,c))); }
float4 fetch(uint base,int x,int y,uint w,uint h,uint channels) {
 float4 v=0;
 if(x>=0&&y>=0&&x<(int)w&&y<(int)h) {
  if(((packed_head&4u)!=0u&&(base==motion_pair||base==motion_eroded||base==motion_non))||((packed_head&32u)!=0u&&base==hprev)) {
   uint2 hv=channels==4u?wg_ld64(base,(y*w+x)*8):uint2(wg_ld32(base,(y*w+x)*4),0u);
   v=float4(f16tof32(hv.x),f16tof32(hv.x>>16),f16tof32(hv.y),f16tof32(hv.y>>16));
  } else for(uint c=0;c<channels;c++) {
  if((packed_head&1u)!=0u&&(base==color||base==precision)) v[c]=wg_h2f(wg_ld16(base,((y*w+x)*4+c)*2));
  else if((packed_head&1u)!=0u&&(base==gates||base==candidates)) v[c]=H(wg_div((float)wg_ld8(base,(y*w+x)*4+c),255.0f));
#if RR_RT
  else if(base==motion_pair||base==motion_eroded||base==motion_non||base==hprev) v[c]=f16tof32(wg_ld16(base,((y*w+x)*channels+c)*2));
#endif
  else v[c]=wg_ldf(base,((y*w+x)*channels+c)*4);
  }
 }
 return v;
}
float4 sample4(uint base,float s,float t,uint w,uint h,uint channels) {
 s=isnan(s)?0:s; t=isnan(t)?0:t;
 float a=A(M(s,w),-.5f),b=A(M(t,h),-.5f),i=floor(a),j=floor(b);
 float fa=A(a,-i),fb=A(b,-j),wa[4];
 wa[0]=M(A(1,-fa),A(1,-fb)); wa[1]=M(fa,A(1,-fb)); wa[2]=M(A(1,-fa),fb); wa[3]=M(fa,fb);
 // Clamp in floating point before integer conversion to avoid overflow at distant finite coordinates.
 int x0=(int)clamp(i,0.0f,(float)(w-1)),x1=(int)clamp(A(i,1),0.0f,(float)(w-1));
 int y0=(int)clamp(j,0.0f,(float)(h-1)),y1=(int)clamp(A(j,1),0.0f,(float)(h-1));
 float4 v[4]; v[0]=fetch(base,x0,y0,w,h,channels); v[1]=fetch(base,x1,y0,w,h,channels); v[2]=fetch(base,x0,y1,w,h,channels); v[3]=fetch(base,x1,y1,w,h,channels);
 float4 outv=0;
 for(uint c=0;c<channels;c++) { outv[c]=M(wa[0],v[0][c]); for(uint k=1;k<4;k++) outv[c]=A(outv[c],M(wa[k],v[k][c])); }
 return outv;
}
uint2 cmp(float4 w,uint a,uint b) { return w[a]>=w[b]||isnan(w[a])||isnan(w[b])?uint2(a,b):uint2(b,a); }
uint4 order(float4 w) {
 uint2 ab=cmp(w,0,1),cd=cmp(w,2,3),hi=cmp(w,ab.x,cd.x),lo=cmp(w,ab.y,cd.y),mid=cmp(w,lo.x,hi.y);
 uint4 o=uint4(hi.x,mid.x,mid.y,lo.y);
 #if BAD_ORDER
 uint tmp=o.y;o.y=o.z;o.z=tmp;
 #endif
 return o;
}
float3 reproj(uint x,uint y,float2 mv,float3 fallback,out uint fell) {
 float px=A(x+.5f,mv.x),py=A(y+.5f,mv.y);
 fell=(px<0||px>wh||py<0||py>hh||(reset!=0&&!BAD_RESET))?1:0;
 if(fell) return fallback;
 float kx=A(floor(A(px,-.5f)),.5f),ky=A(floor(A(py,-.5f)),.5f);
 uint ix=min((uint)F(S(A(px,-kx)),512,.5f),512u),iy=min((uint)F(S(A(py,-ky)),512,.5f),512u);
 float4 tx=wg_ld4f(table,ix*16),ty=wg_ld4f(table,iy*16);
 float weights[5]={M(tx.y,ty.x),M(tx.x,ty.y),M(tx.y,ty.y),M(tx.y,ty.z),M(tx.z,ty.y)};
 float2 pos[5]={float2(A(kx,tx.w),A(ky,-1)),float2(A(kx,-1),A(ky,ty.w)),float2(A(kx,tx.w),A(ky,ty.w)),float2(A(kx,tx.w),A(ky,2)),float2(A(kx,2),A(ky,ty.w))};
 float3 acc=0;
 for(uint k=0;k<5;k++) {
  float3 tap=sample4(hprev,M(pos[k].x,R(wh)),M(pos[k].y,R(hh)),wh,hh,4).xyz;
  for(uint c=0;c<3;c++) acc[c]=k==0?M(tap[c],weights[k]):F(tap[c],weights[k],acc[c]);
 }
 float total=A(weights[4],A(weights[3],A(weights[2],A(weights[1],weights[0]))));
 for(uint c=0;c<3;c++) acc[c]=M(acc[c],R(total));
 return acc;
}
float3 blend(float3 current,float3 hist,float alpha,out uint guarded) {
 float3 b; for(uint c=0;c<3;c++) b[c]=F(alpha,F(current[c],K12,-hist[c]),hist[c]);
 guarded=isnan(A(b.z,A(b.x,b.y)))?1:0;
 if(guarded) for(uint c=0;c<3;c++) b[c]=M(current[c],K12);
 return b;
}
WG_KERNEL(rr_output_order) {
 WG_SETUP(); uint i=wg_block_id()*WG_BLOCK+wg_lane_id(); if(i>=wr) return;
 uint4 o=order(wg_ld4f(color,i*16)); wg_st32(trace,i*4,o.x|(o.y<<2)|(o.z<<4)|(o.w<<6));
}
void out_main_pixel(uint idx) {
 float E=E_arg,Ep=Ep_arg,invE=invE_arg;
 if((packed_head&2u)!=0u) { uint pe=asuint(E_arg); E=asfloat(wg_ld32(pe,0u)); Ep=asfloat(wg_ld32(pe,4u)); invE=asfloat(wg_ld32(pe,8u)); }
 uint x=idx%wh,y=idx/wh;
 float sx=wg_div((float)wr,(float)wh),sy=wg_div((float)hr,(float)hh);
 float u=F(x+.5f,sx,jx),v=F(y+.5f,sy,jy),va=F((y/2)*2+.5f,sy,jy);
 int cx=(int)floor(A(u,-.5f)),cy=(int)floor(A(v,-.5f)),cya=(int)floor(A(va,-.5f));
 float hx=H(A(cx+.5f,-u)),hy=H(A(cy+.5f,-v)); if(cy!=cya) hy=HA(hy,-1);
 float accum[6]={0,0,0,0,0,0}; float total=0;
 for(int k=0;k<4;k++) for(int m=0;m<4;m++) {
  int xx=clamp(cx-1+k,0,(int)wr-1),yy=clamp(cya-1+m,0,(int)hr-1);
  float3 col=fetch(color,xx,yy,wt,((hr+31)/32)*32,4).xyz,g=fetch(gates,xx,yy,wr,hr,3).xyz,p=fetch(precision,xx,yy,wr,hr,3).xyz;
  float ox=HA(hx,k-1),oy=HA(hy,m-1);
  float ax=H(M(p.x,K2n)),ay=H(M(p.y,K2n)),axy=H(M(p.z,K2x2n));
  float e=HF(HM(ox,oy),axy,HF(HM(oy,oy),ay,HM(HM(ox,ox),ax)));
  float w32=X(e),w=H(w32); total=A(total,w32);
  for(uint c=0;c<3;c++) { accum[c]=HF(w,H(C(col[c])),accum[c]); accum[c+3]=HF(w,g[c],accum[c+3]); }
 }
 float norm[6]; for(uint c=0;c<6;c++) norm[c]=M(accum[c],R(max(total,K13)));
 float coverage=S(M(norm[3],M(total,K1))),ah=F(coverage,A(1,-norm[4]),norm[4]),ao=F(coverage,A(1,-norm[5]),norm[5]);
 float3 current=float3(D(norm[0]),D(norm[1]),D(norm[2]));
 float s=M(u,R(wr)),t=M(v,R(hr)); float4 weights=sample4(candidates,s,t,wr,hr,4);
 float4 pair=fetch(motion_pair,(int)floor(u),(int)floor(v),wr,hr,4);
 float2 mv[4]={pair.xy,sample4(motion_non,s,t,wr,hr,2).xy,fetch(motion_eroded,(int)floor(u),(int)floor(v),wr,hr,2).xy,pair.zw};
 uint4 ord=order(weights); float4 n; for(uint i=0;i<4;i++) n[i]=S(weights[ord[i]]);
 float sum=A(A(A(n.y,n.x),n.z),n.w); for(uint i=0;i<4;i++) n[i]=M(n[i],R(max(sum,K13)));
 float3 fallback; for(uint c=0;c<3;c++) fallback[c]=C(M(Ep,M(invE,M(current[c],K12))));
 uint fell=0,fallmask=0,usedmask=1; float3 tap=reproj(x,y,mv[ord.x],fallback,fell); fallmask=fell;
 float3 acc; for(uint c=0;c<3;c++) acc[c]=M(n.x,tap[c]); total=n.x;
 for(uint i=1;i<3;i++) { if(!(total<K3)) break; tap=reproj(x,y,mv[ord[i]],fallback,fell); fallmask|=fell<<i; usedmask|=1u<<i; for(uint c=0;c<3;c++) acc[c]=F(n[i],tap[c],acc[c]); total=A(total,n[i]); }
 float3 hist; for(uint c=0;c<3;c++) hist[c]=M(E,M(M(D(M(acc[c],R(total))),K12),R(Ep)));
 uint gh,go; float3 bh=blend(current,hist,ah,gh),bo=blend(current,hist,ao,go);
 for(uint c=0;c<3;c++) { wg_stf(linear_buf,(idx*3+c)*4,bh[c]); wg_st16(ch,(idx*4+c)*2,wg_f2h(C(bh[c]))); wg_st16(q,(idx*3+c)*2,wg_f2h(C(bo[c]))); }
 wg_st16(ch,(idx*4+3)*2,0);
 wg_st128(trace,idx*16,uint4(ord.x|(ord.y<<2)|(ord.z<<4)|(ord.w<<6),usedmask,fallmask,gh|(go<<1)));
#if OF_DEBUG
 wg_st128(trace,idx*16,uint4(asuint(OF_DEBUG==5?current.y:OF_DEBUG==6?current.z:OF_DEBUG==1?current.x:OF_DEBUG==2?norm[0]:OF_DEBUG==3?accum[0]:hist.x),asuint(OF_DEBUG==5?hist.y:OF_DEBUG==6?hist.z:OF_DEBUG==1?hist.x:OF_DEBUG==2?total:OF_DEBUG==3?accum[3]:acc.x),asuint(OF_DEBUG==5?acc.y:OF_DEBUG==6?acc.z:OF_DEBUG==1?ah:OF_DEBUG==2?coverage:OF_DEBUG==3?accum[5]:n.x),asuint(OF_DEBUG==5?fallback.y:OF_DEBUG==6?bo.y:OF_DEBUG==1?bh.x:OF_DEBUG==2?norm[3]:OF_DEBUG==3?accum[1]:fallback.x)));
#endif
}
WG_KERNEL(rr_output_main) {
 WG_SETUP(); uint idx=wg_block_id()*WG_BLOCK+wg_lane_id(); if(idx>=wh*hh) return;
 out_main_pixel(idx);
}
uint code10(float x) { double v=(double)S(x)*1023.0; uint a=(uint)v; double f=v-(double)a; return min(1023u,a+(f>.5||(f==.5&&(a&1))?1:0)+BAD_CODE); }
void out_pool_quad(uint idx) {
 uint pw=(wh+1)/2;
 uint x=(idx%pw)*2,y=(idx/pw)*2,count=min(2u,wh-x)*min(2u,hh-y);
 float3 tl=fetch(linear_buf,x,y,wh,hh,3).xyz,bl=fetch(linear_buf,x,y+1,wh,hh,3).xyz,tr=fetch(linear_buf,x+1,y,wh,hh,3).xyz,br=fetch(linear_buf,x+1,y+1,wh,hh,3).xyz;
 for(uint c=0;c<3;c++) wg_st16(ph,(idx*4+c)*2,code10(M(C(M(A(A(tl[c],bl[c]),A(tr[c],br[c])),R(count))),K11)));
 wg_st16(ph,(idx*4+3)*2,3);
}
WG_KERNEL(rr_output_pool) {
 WG_SETUP(); uint idx=wg_block_id()*WG_BLOCK+wg_lane_id(); if(idx>=((wh+1)/2)*((hh+1)/2)) return;
 out_pool_quad(idx);
}


// Standalone arithmetic/store probes keep exact rules outside image tolerances.
WG_KERNEL(rr_output_arith) {
 WG_SETUP(); uint i=wg_block_id()*WG_BLOCK+wg_lane_id(); if(i>=wr) return;
 float4 v=wg_ld4f(color,i*16);
 wg_st128(trace,i*32,uint4(asuint(F(v.x,v.y,v.z)),asuint(M(v.x,v.y)),asuint(A(v.x,v.z)),code10(v.w)));
 wg_st128(trace,i*32+16,uint4(wg_f2h(HF(H(v.x),H(v.y),H(v.z))),wg_f2h(HM(H(v.x),H(v.y))),0,0));
}

#include "rr_output_fast.hlsli"
