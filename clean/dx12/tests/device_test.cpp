// SPDX-License-Identifier: MIT
// Self-test of the hybrid D3D12 device (clean/runtime/device_dx12.cpp) with the test kernels of device_test.hlsl:
// arena sub-allocation and pointer -> offset conversion, a null pointer and a by-value struct argument, a grid above
// 65535 workgroups (split dispatches), consecutive D3D12 dispatches in one command list that read what the previous
// one wrote (UAV barriers), HIP -> D3D12 and D3D12 -> HIP ordering through the shared fence (HIP copies and memsets
// between D3D12 dispatches), free + re-allocation, and the HIP fallback for a pointer outside the arena.
//   device_test <rocm dir> <build dir with dx12/manifest.json of device_test.hlsl>      (run by run_device_test.py)
#include "../../runtime/device.h"

#include <cstdio>
#include <cstring>
#include <vector>

using namespace wgrt;

static int fails = 0;
#define EXPECT(c, ...) do { if (!(c)) { std::printf("FAIL: " __VA_ARGS__); std::printf("\n"); ++fails; } } while (0)

int main(int argc, char** argv) {
  if (argc != 3) {
    std::printf("usage: device_test <rocm dir> <build dir>\n");
    return 2;
  }
  std::string err;
  auto dev = make_dx12_device(argv[1], argv[2], &err);
  if (!dev) {
    std::printf("make_dx12_device: %s\n", err.c_str());
    return 1;
  }
  std::printf("device: %s\nnote: %s\n", dev->name().c_str(), dev->note().c_str());
  const uint32_t n = 66000u * 64u + 17u;  // 66001 workgroups of 64
  const size_t bytes = (size_t)n * 4;
  DevPtr x = 0, y = 0, z = 0, tmp = 0;
  EXPECT(dev->alloc(&tmp, 1000) && dev->alloc(&x, bytes) && dev->alloc(&y, bytes), "alloc: %s", dev->err().c_str());
  dev->free(tmp);  // a hole before x: the next allocation that fits reuses it
  EXPECT(dev->alloc(&z, bytes), "alloc z: %s", dev->err().c_str());
  std::vector<uint32_t> h(n), r(n);
  for (uint32_t i = 0; i < n; ++i) h[i] = i * 7u;
  EXPECT(dev->upload(x, h.data(), bytes), "upload");
  // D3D12: x += 1, x += 2, y[i] = x[(i + 1) % n] + 5, y += 3 (one command list, barriers between)
  uint32_t k1 = 1, k2 = 2, k3 = 3, nn = n;
  struct { uint32_t n, k; } s = {n, 5};
  DevPtr null = 0;
  void* add1[] = {&x, &nn, &k1};
  void* add2[] = {&x, &nn, &k2};
  void* shift[] = {&x, &y, &s, &null};
  void* add3[] = {&y, &nn, &k3};
  const unsigned grid = (n + 63) / 64;
  EXPECT(dev->launch("wg_test_add", grid, 64, add1) && dev->launch("wg_test_add", grid, 64, add2) &&
             dev->launch("tst/wg_test_shift", grid, 64, shift) && dev->launch("wg_test_add", grid, 64, add3),
         "launch: %s", dev->err().c_str());
  // HIP: z = y (copy), then x = 0 (memset); D3D12 again: z += 10 (must see the copy), y[i] = z[(i+1) % n] + 5
  EXPECT(dev->copy(z, y, bytes) && dev->memset(x, 0, bytes), "copy / memset: %s", dev->err().c_str());
  uint32_t k10 = 10;
  void* add10[] = {&z, &nn, &k10};
  void* shift2[] = {&z, &y, &s, &null};
  EXPECT(dev->launch("wg_test_add", grid, 64, add10) && dev->launch("tst/wg_test_shift", grid, 64, shift2),
         "launch 2: %s", dev->err().c_str());
  EXPECT(dev->download(r.data(), y, bytes), "download: %s", dev->err().c_str());
  int bad = 0;
  for (uint32_t i = 0; i < n && bad < 5; ++i) {
    const uint32_t j = (i + 1) % n, k = (j + 1) % n;
    const uint32_t zj = h[k] + 3 + 5 + 3 + 10;  // z[j] = y1[j] + 10, y1[j] = x2[(j+1)%n] + 5 + 3, x2 = h + 3
    if (r[i] != zj + 5) {
      if (bad++ < 5) std::printf("  y[%u] = %u, expected %u\n", i, r[i], zj + 5);
    }
  }
  EXPECT(bad == 0, "results after D3D12 / HIP / D3D12");
  EXPECT(dev->download(r.data(), x, bytes), "download x");
  bool zero = true;
  for (uint32_t i = 0; i < n; ++i) zero = zero && r[i] == 0;
  EXPECT(zero, "the HIP memset between the D3D12 dispatches is not ordered");
  // a pointer outside the arena (the kernel exists only on D3D12 here, so the HIP fallback must report a failure)
  DevPtr outside = 16;
  void* bad_args[] = {&outside, &nn, &k1};
  EXPECT(!dev->launch("wg_test_add", grid, 64, bad_args), "a launch with a pointer outside the arena went to D3D12");
  // block size mismatch is refused
  EXPECT(!dev->launch("wg_test_add", grid, 128, add1), "a block size mismatch was accepted");
  EXPECT(dev->sync(), "sync: %s", dev->err().c_str());
  dev->free(x), dev->free(y), dev->free(z);
  std::printf("device_test: %s\n", fails ? "FAILED" : "all checks passed");
  return fails ? 1 : 0;
}
