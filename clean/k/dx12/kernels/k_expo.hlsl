// SPDX-License-Identifier: MIT
// D3D12 port of model K's k_expo_scalars (clean/k/kernels/k_io.hip; HIP build: k_io.co). WrongGPU's own code.
// Spec 42 draft 3 section 4 / spec 43 section 3.7, exactly the HIP kernel's operations: the state st = [E_cur, E_prev,
// E_aux, E_cur kept for the next frame] (raw values), each read as FP16 (nearest even, wg_f2h; the plain f32tof16
// rounds toward zero on this driver) with 0 -> 1, then kappa = inv_pi e1 E_cur (1 when 0), rho = e1 E_cur / (e1 E_prev),
// lambda = e2 E_aux / (e1 E_cur), kinv = RN(1 / kappa); the divisions correctly rounded as HIP's (wg_div; it flushes a
// denormal quotient to zero, which these E values, FP16 numbers >= 2^-24 times the factors, do not produce in practice:
// the port check clean/k/dx12/k_port_check.py covers the runtime's cases). One lane writes xs = (kappa, rho, lambda,
// kinv), which the input / output stage kernels read through KFrameIn.xs.
// Args as the HIP kernel: st, src, measured, xs (pointers), KExpoArgs (36 bytes, runtime_k.cpp).
// k_expo_scalars_direct (rs=direct, the native executor's direct I/O; launched once runtime_k.cpp names it): E_cur of
// src_mode 1 is the game's exposure texel t3 (native_direct_exposure, the value the HIP executor copies into src[0]);
// the src pointer is unused, everything else identical.
// @wg-kernel k_expo_scalars co=k_io.co block=32 wave=32 args=p,p,p,p,b36
// @wg-kernel k_expo_scalars_direct co=k_io.co entry=k_expo_scalars block=32 wave=32 args=p,p,p,p,b36 rs=direct defines=WG_DIRECT_TEXTURES=1
#include "wg_platform.hlsli"
#ifndef WG_DIRECT_TEXTURES
#define WG_DIRECT_TEXTURES 0
#endif
#if WG_DIRECT_TEXTURES
#include "native_direct.hlsli"
#endif

WG_ARGS_BEGIN
    uint st, src, measured, xs;                       // pointers
    float A_value;                                    // KExpoArgs (36 bytes)
    int A_src_mode, A_aux, A_first, A_hdr, A_reset;
    float A_e1, A_e2, A_inv_pi;
WG_ARGS_END

float kx_ev(float e)
{
    const float h = wg_h2f(wg_f2h(e));
    return h == 0.0f ? 1.0f : h;
}

WG_KERNEL(k_expo_scalars)
{
    WG_SETUP();
    if (wg_lane_id() != 0u)
        return;
    float Ecr = 1.0f, Epr = 1.0f, Ear = 1.0f;
    if (A_hdr != 0) {
#if WG_DIRECT_TEXTURES
        Ecr = A_src_mode == 1 ? native_direct_exposure() : (A_src_mode == 0 ? A_value : 1.0f);
#else
        Ecr = A_src_mode == 1 ? wg_ldf(src, 0u) : (A_src_mode == 0 ? A_value : 1.0f);
#endif
        Ear = A_aux != 0 ? wg_ldf(measured, 0u) : Ecr;
        Epr = (A_first != 0 || A_reset != 0) ? Ecr : wg_ldf(st, 12u);
    }
    wg_stf(st, 0u, Ecr);
    wg_stf(st, 4u, Epr);
    wg_stf(st, 8u, Ear);
    wg_stf(st, 12u, Ecr);
    const float e1 = A_hdr != 0 ? A_e1 : 1.0f, e2 = A_hdr != 0 ? A_e2 : 1.0f;
    const float Ec = kx_ev(Ecr), Ep = kx_ev(Epr), Ea = kx_ev(Ear);
    const float ce = e1 * Ec;
    float kappa = 1.0f, rho = 1.0f;
    if (A_hdr != 0) {
        kappa = A_inv_pi * ce;
        if (kappa == 0.0f)
            kappa = 1.0f;
        rho = wg_div(ce, e1 * Ep);
    }
    const float lam = wg_div(e2 * Ea, ce);
    wg_stf(xs, 0u, kappa);
    wg_stf(xs, 4u, rho);
    wg_stf(xs, 8u, lam);
    wg_stf(xs, 12u, wg_div(1.0f, kappa));
}
