// SPDX-License-Identifier: MIT
// The bridge between hlsl_emu.h (the D3D12 ports' kernel text compiled as C++) and the lane emulator of the HIP kernels
// (clean/kernels/emu/wg_emu.cpp). Test-only; never shipped. WrongGPU's own code.
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "wg_platform.h"   // -DWG_EMU: the emulator's types and collective operations

extern "C" int emu_launch(const char* name, unsigned gx, unsigned block, void** args);   // wg_emu.cpp

extern "C" {
unsigned char* hemu_arena = nullptr;
unsigned long long hemu_arena_bytes = 0;
unsigned long long hemu_oob = 0;
unsigned hemu_consts[256];
static char hemu_error[256];

unsigned hemu_lane() { return wg_emu::lane_id(); }
unsigned hemu_block() { return wg_emu::block_id(); }
unsigned hemu_xor(unsigned v, unsigned mask)
{
    return __builtin_bit_cast(unsigned, wg_emu::xor_lanes(__builtin_bit_cast(float, v), mask));
}
void hemu_barrier() { wg_emu::barrier(); }
void hemu_wmma_g12(const unsigned a[4], const unsigned b[4], const float c[8], float d[8])
{
    v8h av, bv;
    v8f cv, dv;
    std::memcpy(&av, a, 16);
    std::memcpy(&bv, b, 16);
    std::memcpy(&cv, c, 32);
    dv = wg_emu::wmma_f16(av, bv, cv);
    std::memcpy(d, &dv, 32);
}
void hemu_wmma_g11(const unsigned a[8], const unsigned b[8], const float c[8], float d[8])
{
    v16h av, bv;
    v8f cv, dv;
    std::memcpy(&av, a, 32);
    std::memcpy(&bv, b, 32);
    std::memcpy(&cv, c, 32);
    dv = wg_emu::wmma16_f16(av, bv, cv);
    std::memcpy(d, &dv, 32);
}
void hemu_fail(const char* what)
{
    if (!hemu_error[0]) std::snprintf(hemu_error, sizeof hemu_error, "%s", what);
}
const char* hemu_last_error() { return hemu_error; }
void hemu_set_arena(unsigned char* p, unsigned long long n)
{
    hemu_arena = p;
    hemu_arena_bytes = n;
    hemu_oob = 0;
    hemu_error[0] = 0;
}
unsigned long long hemu_oob_count() { return hemu_oob; }
void hemu_set_consts(const unsigned* c, unsigned n)
{
    std::memset(hemu_consts, 0, sizeof hemu_consts);
    std::memcpy(hemu_consts, c, (n < 256 ? n : 256) * 4);
}
}

namespace {
void hemu_thunk(void** args) { ((void (*)())args[0])(); }
wg_emu::Registrar hemu_reg("__hemu", hemu_thunk);
}  // namespace

extern "C" int hemu_launch_fn(void (*fn)(), unsigned gx, unsigned block)
{
    void* args[1] = {(void*)fn};
    return emu_launch("__hemu", gx, block, args);
}
