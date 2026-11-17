// SPDX-License-Identifier: MIT
// D3D12 ports of Model L's C = 32 Swin blocks (clean/l/kernels/l_swin.hip, l_swin.co; issue #89): swin_block.hlsl
// instantiated at C = 32, 2 heads, MLP 128 (the same template as the HIP kernels'), with HIP's MLP unit cut (HPU 8 = all
// 8 hidden tiles in one unit). q8(GELU): the frame's blocks read it from the 16 KB arena table, bit-identical to HIP's
// transcendental chain + pack (swin_block.hlsl): _dec_nc with SW_GELU 1 (measured -31 / -33 us per frame against the
// chain at 1080p Quality), _enc_nc and _dec_coef12_nc with the cheaper table offsets for every tile (SW_GELU 3, pattern
// 13333: sw_gelu_fetch8_bn; a further -7 .. -9 / -13 .. -15 us). The groupshared table (SW_GELU 2) and patterns with a
// transcendental-chain tile (10333) take them to 108 .. 112 VGPRs, over their bound of 96. l_swin_block_c32_nc keeps
// HIP's transcendental chain (SW_GELU 0).
// Scheduling knobs of swin_block.hlsl (the same values, timed per kernel): _enc_lin_nc with the head loop in pairs
// (SW_HEAD_PAIRS), the softmax's staged form (SW_SMAX_STAGED) and norm1's loads before the first barrier (SW_NORM1_EARLY);
// _dec_coef12_nc with the staged softmax (SW_SMAX_ADDST, SW_SMAX_STAGED). Both stay within 96 VGPRs.
//   l_swin_block_c32_nc      FP8 map in, FP8 out, mirror border (#117)
//   l_swin_block_c32_enc_nc  the encoder block: FP16 halo map [H + 4][W + 4][32] in (SW_SRC 1), FP8 out
//                            with its workgroups in reverse order (WG_REV_ORDER, wg_platform.hlsli; the same bytes): it
//                            reads first the rows of x2 the linears wrote last, and its last-written rows of x3 are the
//                            ones l_project_gather reads first (more cache hits at both hand-offs). WG_REV_ORDER
//                            reverses the group index of a 1-D dispatch only (a multi-dimensional launch must flatten
//                            it first); every kernel declared with it here and in hyb_swin.hlsl / hyb_gemm.hlsl is 1-D
//   l_swin_block_c32_dec_nc  the decoder block: FP8 map in with the token-grid mirror (SW_SRC 2), FP16 out (SW_OUT_F16)
//   l_swin_block_c32_enc_lin_nc  D3D12 only: the encoder block with the two input linears as its prologue (SW_SRC 3;
//                            byte for byte the pair l_linear_32_32_fused -> _enc_nc, which stays the fallback route):
//                            the FP8 q halo map in its 20-byte form (lev/input_encoder_q20) in, x2 neither
//                            written nor read. Arguments: _enc_nc's (x = q), then
//                            enc0.Win1, enc0.Win2
//
// @wg-kernel l_swin_block_c32_nc co=l_swin.co block=256 wave=32 args=p,p,p,p,u,u,u,u defines=SW_C=32,SW_HEADS=2,SW_MLP=128,SW_ADD=0,SW_SAFE=1,SW_ECL=0,SW_HPU=8,SW_GELU=0,SW_ENTRY_OVERRIDE=l_swin_block_c32_nc
// @wg-kernel l_swin_block_c32_enc_nc co=l_swin.co block=256 wave=32 args=p,p,p,p,u,u,u,u defines=SW_C=32,SW_HEADS=2,SW_MLP=128,SW_ADD=0,SW_SAFE=1,SW_ECL=0,SW_HPU=8,SW_GELU=3,SW_GELU_PAT=13333,SW_SRC=1,SW_ENTRY_OVERRIDE=l_swin_block_c32_enc_nc,WG_REV_ORDER=1
// @wg-kernel l_swin_block_c32_enc_lin_nc co=- block=256 wave=32 args=p,p,p,p,u,u,u,u,p,p defines=SW_C=32,SW_HEADS=2,SW_MLP=128,SW_ADD=0,SW_SAFE=1,SW_ECL=0,SW_HPU=8,SW_GELU=3,SW_GELU_PAT=13333,SW_SRC=3,SW_ENTRY_OVERRIDE=l_swin_block_c32_enc_lin_nc,WG_REV_ORDER=1,SW_HEAD_PAIRS=1,SW_SMAX_STAGED=1,SW_NORM1_EARLY=1
// @wg-kernel l_swin_block_c32_dec_nc co=l_swin.co block=256 wave=32 args=p,p,p,p,u,u,u,u defines=SW_C=32,SW_HEADS=2,SW_MLP=128,SW_ADD=0,SW_SAFE=1,SW_ECL=0,SW_HPU=8,SW_GELU=1,SW_SRC=2,SW_OUT_F16=1,SW_ENTRY_OVERRIDE=l_swin_block_c32_dec_nc
//   l_swin_block_c32_dec_coef12_nc  D3D12 only: the decoder block with l_coef_pack12 as its epilogue (SW_OUT_COEF12;
//                                   byte for byte the pair _dec_nc -> l_coef_pack12, which stays the fallback route);
//                                   the FP16 map is not written. Arguments: _dec_nc's (out unused), then A, C, latent,
//                                   dec0.coef, rw, rh
// @wg-kernel l_swin_block_c32_dec_coef12_nc co=- block=256 wave=32 args=p,p,p,p,u,u,u,u,p,p,p,p,u,u defines=SW_C=32,SW_HEADS=2,SW_MLP=128,SW_ADD=0,SW_SAFE=1,SW_ECL=0,SW_HPU=8,SW_GELU=3,SW_GELU_PAT=13333,SW_SRC=2,SW_OUT_F16=1,SW_OUT_COEF12=1,COEF_CVT_HW=1,SW_ENTRY_OVERRIDE=l_swin_block_c32_dec_coef12_nc,SW_SMAX_ADDST=1,SW_SMAX_STAGED=1
//   l_swin_block_c32_enc_hyb / _dec_hyb  the HYBRID block form (SW_HYB, WG_L_FORM=hybrid): _enc_nc's and _dec_nc's shapes
//                          (the decoder WITH the skip: xadd = the encoder block's FP8 map x3 by the fine mirror, x by
//                          the coarse one, member input RN16(E + S), as l_q7_block_c32_dec and the oracle)
//                            (the linears and l_coef_pack12 as their own launches) with the original's FP16 norm, softmax,
//                            GELU and start values around float32 FP8 chains, one FP16 rounding at each chain's end
// @wg-kernel l_swin_block_c32_enc_hyb co=- block=256 wave=32 args=p,p,p,p,u,u,u,u defines=SW_C=32,SW_HEADS=2,SW_MLP=128,SW_ADD=0,SW_SAFE=1,SW_ECL=0,SW_HPU=8,SW_GELU=0,SW_SRC=1,SW_HYB=1,SW_ENTRY_OVERRIDE=l_swin_block_c32_enc_hyb,WG_REV_ORDER=1
// @wg-kernel l_swin_block_c32_dec_hyb co=- block=256 wave=32 args=p,p,p,p,u,u,u,u defines=SW_C=32,SW_HEADS=2,SW_MLP=128,SW_ADD=1,SW_SAFE=1,SW_ECL=0,SW_HPU=8,SW_GELU=0,SW_OUT_F16=1,SW_HYB=1,SW_ENTRY_OVERRIDE=l_swin_block_c32_dec_hyb
//   l_swin_block_c32_enc_lin_hyb / _dec_coef12_hyb  the hybrid twins of the fusions le / lc (D3D12 only, speed lever 3):
//                          _enc_hyb with the two input linears as its prologue (SW_SRC 3: the q map in its 20-byte
//                          form, enc0.Win1, enc0.Win2 as the last arguments) and _dec_hyb with l_coef_pack12 as its
//                          epilogue (SW_OUT_COEF12, COEF_CVT_HW); the same bytes as the launches they replace
// @wg-kernel l_swin_block_c32_enc_lin_hyb co=- block=256 wave=32 args=p,p,p,p,u,u,u,u,p,p defines=SW_C=32,SW_HEADS=2,SW_MLP=128,SW_ADD=0,SW_SAFE=1,SW_ECL=0,SW_HPU=8,SW_GELU=0,SW_SRC=3,SW_HYB=1,SW_ENTRY_OVERRIDE=l_swin_block_c32_enc_lin_hyb,WG_REV_ORDER=1
// @wg-kernel l_swin_block_c32_dec_coef12_hyb co=- block=256 wave=32 args=p,p,p,p,u,u,u,u,p,p,p,p,u,u defines=SW_C=32,SW_HEADS=2,SW_MLP=128,SW_ADD=1,SW_SAFE=1,SW_ECL=0,SW_HPU=8,SW_GELU=0,SW_OUT_F16=1,SW_OUT_COEF12=1,COEF_CVT_HW=1,SW_HYB=1,SW_ENTRY_OVERRIDE=l_swin_block_c32_dec_coef12_hyb,WG_REV_ORDER=1
//   ..._enc_lin_hyb_dump / _dec_coef12_hyb_dump (+ _mutdump: SW_HYB_MUT_ORDER=4)  their stage dumps (probe=1, check only:
//                          l_hyb_stage_check.py --set fused, the oracle per stage); _scanmut: the fused-square scan's
//                          mutants (no contraction barrier, built beside every hybrid build, outside the manifest)
// @wg-kernel l_swin_block_c32_enc_lin_hyb_dump co=- probe=1 block=256 wave=32 args=p,p,p,p,u,u,u,u,p,p,p defines=SW_C=32,SW_HEADS=2,SW_MLP=128,SW_ADD=0,SW_SAFE=1,SW_ECL=0,SW_HPU=8,SW_GELU=0,SW_SRC=3,SW_HYB=1,SW_HYB_DUMP=1,SW_ENTRY_OVERRIDE=l_swin_block_c32_enc_lin_hyb_dump,WG_REV_ORDER=1
// @wg-kernel l_swin_block_c32_enc_lin_hyb_mutdump co=- probe=1 block=256 wave=32 args=p,p,p,p,u,u,u,u,p,p,p defines=SW_C=32,SW_HEADS=2,SW_MLP=128,SW_ADD=0,SW_SAFE=1,SW_ECL=0,SW_HPU=8,SW_GELU=0,SW_SRC=3,SW_HYB=1,SW_HYB_DUMP=1,SW_HYB_MUT_ORDER=4,SW_ENTRY_OVERRIDE=l_swin_block_c32_enc_lin_hyb_mutdump,WG_REV_ORDER=1
// @wg-kernel l_swin_block_c32_enc_lin_hyb_scanmut co=- probe=1 block=256 wave=32 args=p,p,p,p,u,u,u,u,p,p defines=SW_C=32,SW_HEADS=2,SW_MLP=128,SW_ADD=0,SW_SAFE=1,SW_ECL=0,SW_HPU=8,SW_GELU=0,SW_SRC=3,SW_HYB=1,HYB_SCAN_MUTANT=1,SW_ENTRY_OVERRIDE=l_swin_block_c32_enc_lin_hyb_scanmut,WG_REV_ORDER=1
// @wg-kernel l_swin_block_c32_dec_coef12_hyb_dump co=- probe=1 block=256 wave=32 args=p,p,p,p,u,u,u,u,p,p,p,p,u,u,p defines=SW_C=32,SW_HEADS=2,SW_MLP=128,SW_ADD=1,SW_SAFE=1,SW_ECL=0,SW_HPU=8,SW_GELU=0,SW_OUT_F16=1,SW_OUT_COEF12=1,COEF_CVT_HW=1,SW_HYB=1,SW_HYB_DUMP=1,SW_ENTRY_OVERRIDE=l_swin_block_c32_dec_coef12_hyb_dump
// @wg-kernel l_swin_block_c32_dec_coef12_hyb_mutdump co=- probe=1 block=256 wave=32 args=p,p,p,p,u,u,u,u,p,p,p,p,u,u,p defines=SW_C=32,SW_HEADS=2,SW_MLP=128,SW_ADD=1,SW_SAFE=1,SW_ECL=0,SW_HPU=8,SW_GELU=0,SW_OUT_F16=1,SW_OUT_COEF12=1,COEF_CVT_HW=1,SW_HYB=1,SW_HYB_DUMP=1,SW_HYB_MUT_ORDER=4,SW_ENTRY_OVERRIDE=l_swin_block_c32_dec_coef12_hyb_mutdump
// @wg-kernel l_swin_block_c32_dec_coef12_hyb_scanmut co=- probe=1 block=256 wave=32 args=p,p,p,p,u,u,u,u,p,p,p,p,u,u defines=SW_C=32,SW_HEADS=2,SW_MLP=128,SW_ADD=1,SW_SAFE=1,SW_ECL=0,SW_HPU=8,SW_GELU=0,SW_OUT_F16=1,SW_OUT_COEF12=1,COEF_CVT_HW=1,SW_HYB=1,HYB_SCAN_MUTANT=1,SW_ENTRY_OVERRIDE=l_swin_block_c32_dec_coef12_hyb_scanmut
//   l_swin_block_c32_enc_hyb_dump / _dec_hyb_dump  check variants (probe=1, never in a model build): the same kernels plus
//                          the stage dump (SW_HYB_DUMP, the extra pointer argument; l_hyb_stage_check.py)
// @wg-kernel l_swin_block_c32_enc_hyb_dump co=- probe=1 block=256 wave=32 args=p,p,p,p,u,u,u,u,p defines=SW_C=32,SW_HEADS=2,SW_MLP=128,SW_ADD=0,SW_SAFE=1,SW_ECL=0,SW_HPU=8,SW_GELU=0,SW_SRC=1,SW_HYB=1,SW_HYB_DUMP=1,SW_ENTRY_OVERRIDE=l_swin_block_c32_enc_hyb_dump,WG_REV_ORDER=1
// @wg-kernel l_swin_block_c32_dec_hyb_dump co=- probe=1 block=256 wave=32 args=p,p,p,p,u,u,u,u,p defines=SW_C=32,SW_HEADS=2,SW_MLP=128,SW_ADD=1,SW_SAFE=1,SW_ECL=0,SW_HPU=8,SW_GELU=0,SW_OUT_F16=1,SW_HYB=1,SW_HYB_DUMP=1,SW_ENTRY_OVERRIDE=l_swin_block_c32_dec_hyb_dump
//   l_swin_block_c32_enc_hyb_mut / _dec_hyb_mut  the byte-identity mutant (probe=1, check only): SW_HYB_MUT_ORDER=4, the odd W2
//                          groups in a second accumulator added at the end (l_hyb_block_check.py --mutant must FAIL HIP == D3D12)
// @wg-kernel l_swin_block_c32_enc_hyb_mut co=- probe=1 block=256 wave=32 args=p,p,p,p,u,u,u,u defines=SW_C=32,SW_HEADS=2,SW_MLP=128,SW_ADD=0,SW_SAFE=1,SW_ECL=0,SW_HPU=8,SW_GELU=0,SW_SRC=1,SW_HYB=1,SW_HYB_MUT_ORDER=4,SW_ENTRY_OVERRIDE=l_swin_block_c32_enc_hyb_mut,WG_REV_ORDER=1
// @wg-kernel l_swin_block_c32_dec_hyb_mut co=- probe=1 block=256 wave=32 args=p,p,p,p,u,u,u,u defines=SW_C=32,SW_HEADS=2,SW_MLP=128,SW_ADD=1,SW_SAFE=1,SW_ECL=0,SW_HPU=8,SW_GELU=0,SW_OUT_F16=1,SW_HYB=1,SW_HYB_MUT_ORDER=4,SW_ENTRY_OVERRIDE=l_swin_block_c32_dec_hyb_mut
// @wg-kernel l_swin_block_c32_enc_hyb_mutdump co=- probe=1 block=256 wave=32 args=p,p,p,p,u,u,u,u,p defines=SW_C=32,SW_HEADS=2,SW_MLP=128,SW_ADD=0,SW_SAFE=1,SW_ECL=0,SW_HPU=8,SW_GELU=0,SW_SRC=1,SW_HYB=1,SW_HYB_MUT_ORDER=4,SW_HYB_DUMP=1,SW_ENTRY_OVERRIDE=l_swin_block_c32_enc_hyb_mutdump,WG_REV_ORDER=1
//   l_swin_block_c32_enc_hyb_scanmut  the fused-square scan's mutant (probe=1; build_dx12.py builds it beside every build
//                          of hybrid kernels, outside the manifest, and requires hyb_isa_scan.py to FAIL on it):
//                          _enc_hyb without the contraction barrier (HYB_SCAN_MUTANT=1)
// @wg-kernel l_swin_block_c32_enc_hyb_scanmut co=- probe=1 block=256 wave=32 args=p,p,p,p,u,u,u,u defines=SW_C=32,SW_HEADS=2,SW_MLP=128,SW_ADD=0,SW_SAFE=1,SW_ECL=0,SW_HPU=8,SW_GELU=0,SW_SRC=1,SW_HYB=1,HYB_SCAN_MUTANT=1,SW_ENTRY_OVERRIDE=l_swin_block_c32_enc_hyb_scanmut,WG_REV_ORDER=1
// @wg-kernel l_swin_block_c32_dec_hyb_scanmut co=- probe=1 block=256 wave=32 args=p,p,p,p,u,u,u,u defines=SW_C=32,SW_HEADS=2,SW_MLP=128,SW_ADD=1,SW_SAFE=1,SW_ECL=0,SW_HPU=8,SW_GELU=0,SW_OUT_F16=1,SW_HYB=1,HYB_SCAN_MUTANT=1,SW_ENTRY_OVERRIDE=l_swin_block_c32_dec_hyb_scanmut
#include "swin_block.hlsl"
