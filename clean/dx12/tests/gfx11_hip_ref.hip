// SPDX-License-Identifier: MIT
// HIP reference of the gfx11 helper tests (test_gfx11_helpers.py): D = A x B + C by the native v_wmma_f32_16x16x16_f16
// of gfx1100 (the HIP code's wg_wmma16_f16, clean/kernels/wg_platform.h), K = 16, one 16x16 matrix per workgroup.
//   gfx11_hip_ref <in> <out>      in: int32 N, A[N][16][16] f16, B[N][16][16] f16, C[N][16][16] f32 (row-major)
//                                 out: D[N][16][16] f32 (row-major)
// A lane holds row (A) / column (B) l % 16 with K 0 .. 15 in order; element i of lane l of D is row 2 i + l / 16.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
typedef _Float16 v16h __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));
__global__ void k(const _Float16* A, const _Float16* B, const float* C, float* D) {
  const int n = blockIdx.x, l = threadIdx.x;
  const _Float16* a = A + n * 256; const _Float16* b = B + n * 256; const float* c = C + n * 256; float* d = D + n * 256;
  v16h fa, fb; for (int i = 0; i < 16; ++i) { fa[i] = a[(l % 16) * 16 + i]; fb[i] = b[i * 16 + (l % 16)]; }
  v8f fc; for (int i = 0; i < 8; ++i) fc[i] = c[(2 * i + l / 16) * 16 + l % 16];
  const v8f r = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(fa, fb, fc);
  for (int i = 0; i < 8; ++i) d[(2 * i + l / 16) * 16 + l % 16] = r[i];
}
int main(int argc, char** argv) {
  if (argc < 3) return 1;
  FILE* f = fopen(argv[1], "rb"); if (!f) return 1;
  int N = 0; if (fread(&N, 4, 1, f) != 1) return 1;
  std::vector<_Float16> A(N * 256), B(N * 256); std::vector<float> C(N * 256), D(N * 256);
  if (fread(A.data(), 2, A.size(), f) != A.size() || fread(B.data(), 2, B.size(), f) != B.size() ||
      fread(C.data(), 4, C.size(), f) != C.size()) return 1;
  fclose(f);
  _Float16 *dA, *dB; float *dC, *dD;
  hipMalloc(&dA, N * 512); hipMalloc(&dB, N * 512); hipMalloc(&dC, N * 1024); hipMalloc(&dD, N * 1024);
  hipMemcpy(dA, A.data(), N * 512, hipMemcpyHostToDevice); hipMemcpy(dB, B.data(), N * 512, hipMemcpyHostToDevice);
  hipMemcpy(dC, C.data(), N * 1024, hipMemcpyHostToDevice);
  hipDeviceProp_t p; hipGetDeviceProperties(&p, 0); fprintf(stderr, "device %s %s\n", p.name, p.gcnArchName);
  k<<<N, 32>>>(dA, dB, dC, dD); hipDeviceSynchronize();
  hipMemcpy(D.data(), dD, N * 1024, hipMemcpyDeviceToHost);
  f = fopen(argv[2], "wb"); fwrite(D.data(), 4, D.size(), f); fclose(f);
  return 0;
}
