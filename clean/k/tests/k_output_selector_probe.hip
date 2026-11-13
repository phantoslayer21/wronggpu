// SPDX-License-Identifier: MIT
#include "../kernels/k_io.hip"
WG_KERNEL_N(k_selector_native,128)(u32* out) {
 const u32 i=wg_block_id()*128u+wg_lane_id();if(i>=65536u)return;
 const h16 nv=__builtin_bit_cast(h16,(unsigned short)i),den=kh_add((h16)1.0f,__builtin_fabsf16(nv));
 const h16 z=kh_div(nv,den);const int q=(int)(float)kh_rni(kh_mul(z,(h16)8.0008f))+8;
 out[2*i]=__builtin_bit_cast(unsigned short,z);out[2*i+1]=ki_max(ki_min(q,15),0);
}
WG_KERNEL_N(k_selector_fast,128)(u32* out) {
 const u32 i=wg_block_id()*128u+wg_lane_id();if(i>=65536u)return;
 const h16 nv=__builtin_bit_cast(h16,(unsigned short)i);
 const h16 z=ko_selector_normalize(nv);const int q=(int)(float)kh_rni(kh_mul(z,(h16)8.0008f))+8;
 out[2*i]=__builtin_bit_cast(unsigned short,z);out[2*i+1]=ki_max(ki_min(q,15),0);
}
