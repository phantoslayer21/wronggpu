// SPDX-License-Identifier: MIT
// Developer builds (WG_DEV_GFX11 or WG_DEV_DIAGNOSTICS): the 16-byte alignment premise of the native RDNA 3 Model K
// kernels. WrongGPU's own code. Their KN_A16 form (clean/k/dx12/hlsl/k_block_g11.hlsli) clears the low 4 bits of every
// 16-byte arena address, which is the same address only when each pointer of the stage parameters (KBlkParams,
// runtime_k.cpp) is 16-byte aligned: the device allocator and the weight packer give 256, and a pointer with any of the
// low 4 bits set would make those kernels read and write other bytes. runtime_k.cpp checks every stage's parameters at
// create and every stage launch, and fails naming the pointer. Tested by clean/runtime/tests/test_k_blk_align.cpp.
#pragma once

#include <cstdint>

// The name of the first non-null pointer of p (a KBlkParams) that is not a multiple of 16, or nullptr when all are.
// The field list is KBlkParams's pointer fields in declaration order (pinned against the struct by the test).
template <class P>
const char* k_blk_misaligned(const P& p) {
  const struct { uint64_t v; const char* name; } f[] = {
      {p.x, "x"},       {p.skip, "skip"}, {p.y, "y"},   {p.merged, "merged"}, {p.head, "head"},
      {p.g1, "g1"},     {p.g2, "g2"},     {p.wqkv, "wqkv"}, {p.mix, "mix"},  {p.wo, "wo"},
      {p.bo, "bo"},     {p.w1, "w1"},     {p.b1, "b1"}, {p.w2, "w2"},     {p.b2, "b2"},
      {p.wa, "wa"},     {p.ba, "ba"},     {p.wz, "wz"}, {p.bz, "bz"}};
  for (const auto& e : f)
    if (e.v & 15u) return e.name;
  return nullptr;
}
