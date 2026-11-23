// SPDX-License-Identifier: MIT
// Model K's narrow network stages in the native RDNA 3 (gfx11) form: the k_block template of k_block.hlsl with every
// matrix product as one K = 16 instruction on operands the kernel lays out (hlsl/k_block_g11.hlsli). WrongGPU's own
// code. Developer-only and gfx11 only (build_dx12.py --gfx 11 writes them to <out>/dx12-gfx11; they need the translator's
// K = 16 opcode, a stock translator refuses them). The same instantiations, arguments, launches and weight blob as the
// RDNA 4 ports of k_block.hlsl (k_block.co); a name is the RDNA 4 port's with the suffix __g11.
// Each declaration sets the template's register and load knobs measured fastest for that stage on the RX 7900 XT
// (hlsl/k_block_g11.hlsli; numbers only, the same bytes as without them): KN_A16 everywhere (128-bit arena loads);
// e2 and e0 two-window stream z (KN_ZSTREAM), d2 and d0 load VT / Wo per channel tile (KB_GRPLATE): 192 VGPRs, 8 waves;
// e1 two-window pins x's reload and loads late (KN_XPIN, KB_GRPLATE, KB_W2LATE): 240 VGPRs, 6 waves; d1 two-window
// streams q / k / v (KN_VSTREAM, KN_NB_HOIST 2): half the spills.
//
// @wg-kernel k_stage_e2__g11 co=k_block.co entry=k_stage_e2 block=128 wave=32 args=s184:p19.b32 defines=KN_ZSTREAM=1,KN_A16=1 gfx=11
// @wg-kernel k_stage2_e0__g11 co=k_block.co entry=k_stage2_e0 block=128 wave=32 args=s184:p19.b32,p defines=KB_BIAS32=1,KN_ZSTREAM=1,KN_A16=1 gfx=11
// @wg-kernel k_stage2_e1__g11 co=k_block.co entry=k_stage2_e1 block=128 wave=32 args=s184:p19.b32,p defines=KB_BIAS32=1,KN_XPIN=1,KB_GRPLATE=1,KB_W2LATE=1,KN_A16=1 gfx=11
// @wg-kernel k_stage_d2__g11 co=k_block.co entry=k_stage_d2 block=128 wave=32 args=s184:p19.b32 defines=KB_GRPLATE=1,KN_A16=1 gfx=11
// @wg-kernel k_stage2_d1__g11 co=k_block.co entry=k_stage2_d1 block=128 wave=32 args=s184:p19.b32,p defines=KB_BIAS32=1,KB_GRPLATE=1,KN_VSTREAM=1,KN_NB_HOIST=2,KN_A16=1 gfx=11
// @wg-kernel k_stage_e0__g11 co=k_block.co entry=k_stage_e0 block=128 wave=32 args=s184:p19.b32 defines=KB_LDS_HALFS=6144,KN_A16=1 gfx=11
// @wg-kernel k_stage_e1__g11 co=k_block.co entry=k_stage_e1 block=128 wave=32 args=s184:p19.b32 defines=KB_LDS_HALFS=4096,KN_A16=1 gfx=11
// @wg-kernel k_stage_d1__g11 co=k_block.co entry=k_stage_d1 block=128 wave=32 args=s184:p19.b32 defines=KN_A16=1 gfx=11
// @wg-kernel k_stage2_d0_hp__g11 co=k_block.co entry=k_stage2_d0_hp block=128 wave=32 args=s184:p19.b32,p,p,p defines=KB_HIDDEN_ARG=1,KB_COMPACT_HEAD_ARG=1,KB_BIAS32=1,KB_GRPLATE=1,KN_A16=1 gfx=11
// @wg-kernel k_stage2_d0_hidden__g11 co=k_block.co entry=k_stage2_d0_hidden block=128 wave=32 args=s184:p19.b32,p defines=KB_HIDDEN_ARG=1,KB_GRPLATE=1,KN_A16=1 gfx=11
// The round-to-odd softmax forms (hlsl/k_smx.hlsli K_SMX 1), as k_block.hlsl's __rodd ports:
// @wg-kernel k_stage_e2__rodd__g11 co=k_block.co entry=k_stage_e2 block=128 wave=32 args=s184:p19.b32 defines=K_SMX=1,KN_ZSTREAM=1,KN_A16=1 gfx=11
// @wg-kernel k_stage_d2__rodd__g11 co=k_block.co entry=k_stage_d2 block=128 wave=32 args=s184:p19.b32 defines=K_SMX=1,KB_GRPLATE=1,KN_A16=1 gfx=11
// @wg-kernel k_stage2_d1__rodd__g11 co=k_block.co entry=k_stage2_d1 block=128 wave=32 args=s184:p19.b32,p defines=K_SMX=1,KB_BIAS32=1,KB_GRPLATE=1,KN_VSTREAM=1,KN_NB_HOIST=2,KN_A16=1 gfx=11
#include "wg_gfx11.hlsli"
#include "k_blkparams.hlsli"

WG_ARGS_BEGIN
    KBLK_PARAMS_DECL
#if KB_HIDDEN_ARG
    uint hidden_p;   // the hidden history Hh written by the d0 epilogue
#endif
#if KB_BIAS32
    uint P_b32;      // main's #288 launch interface (k_block.hlsl): the prewidened FP32 bias plane ...
#endif
#if KB_COMPACT_HEAD_ARG
    uint P_compact_head;   // ... and d0's compact head weights
#endif
WG_ARGS_END
#include "k_block_g11.hlsli"

#if KB_HIDDEN_ARG
WG_KERNEL(k_stage2_d0_hp)
{
    WG_SETUP();
#if KB_COMPACT_HEAD_ARG
    k_block_n11<32, 2, KMIX_FIXED, KIN_EXPAND, 64, 0, true, 2, true, true, false>(hidden_p, wg_block_id(), P_b32,
                                                                               P_compact_head);
#else
    k_block_n11<32, 2, KMIX_FIXED, KIN_EXPAND, 64, 0, true, 2, true, true, false>(hidden_p, wg_block_id());
#endif
}
#if !KB_COMPACT_HEAD_ARG
WG_KERNEL(k_stage2_d0_hidden)
{
    WG_SETUP();
    k_block_n11<32, 2, KMIX_FIXED, KIN_EXPAND, 64, 0, true, 2, true, false, false>(hidden_p, wg_block_id());
}
#endif
#else
WG_KERNEL(k_stage_e2)
{
    WG_SETUP();
    k_block_n11<64, 2, KMIX_SOFTMAX, KIN_MAP, 64, 96, false, 1, false, false, false>(0u, wg_block_id());
}
WG_KERNEL(k_stage2_e0)
{
    WG_SETUP();
#if KB_BIAS32
    k_block_n11<32, 2, KMIX_FIXED, KIN_EMBED, 16, 64, false, 2, false, false, false>(0u, wg_block_id(), P_b32);
#else
    k_block_n11<32, 2, KMIX_FIXED, KIN_EMBED, 16, 64, false, 2, false, false, false>(0u, wg_block_id());
#endif
}
WG_KERNEL(k_stage2_e1)
{
    WG_SETUP();
#if KB_BIAS32
    k_block_n11<64, 2, KMIX_FIXED, KIN_MAP, 64, 64, false, 2, false, false, false>(0u, wg_block_id(), P_b32);
#else
    k_block_n11<64, 2, KMIX_FIXED, KIN_MAP, 64, 64, false, 2, false, false, false>(0u, wg_block_id());
#endif
}
WG_KERNEL(k_stage_d2)
{
    WG_SETUP();
    k_block_n11<64, 2, KMIX_SOFTMAX, KIN_EXPAND, 96, 0, false, 1, false, false, false>(0u, wg_block_id());
}
WG_KERNEL(k_stage2_d1)
{
    WG_SETUP();
#if KB_BIAS32
    k_block_n11<64, 2, KMIX_SOFTMAX, KIN_EXPAND, 64, 0, false, 2, false, false, true>(0u, wg_block_id(), P_b32);
#else
    k_block_n11<64, 2, KMIX_SOFTMAX, KIN_EXPAND, 64, 0, false, 2, false, false, true>(0u, wg_block_id());
#endif
}
WG_KERNEL(k_stage_e0)
{
    WG_SETUP();
    k_block_n11<32, 2, KMIX_FIXED, KIN_EMBED, 16, 64, false, 1, false, false, false>(0u, wg_block_id());
}
WG_KERNEL(k_stage_e1)
{
    WG_SETUP();
    k_block_n11<64, 2, KMIX_FIXED, KIN_MAP, 64, 64, false, 1, false, false, false>(0u, wg_block_id());
}
WG_KERNEL(k_stage_d1)
{
    WG_SETUP();
    k_block_n11<64, 2, KMIX_SOFTMAX, KIN_EXPAND, 64, 0, false, 1, false, false, false>(0u, wg_block_id());
}
#endif
