// SPDX-License-Identifier: MIT
// Original synthetic fused decoder differential, adapted from clean dx12-m/dec/bench.
// HIP and native D3D12 receive the same generated FP8 features, weights, motion and latent bytes.
// Checks A (8 bytes/pixel), C (4 bytes/pixel) and latent (4 bytes/pixel); B is not written by pack12.
// Raw output equality only; whole-frame/history and intermediate tests remain separate gates. No timing claim.
#include "../../runtime/device.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace wgrt;

struct DecParams { int rw, rh, fw, fh, reset, pad0; float dw, dh, kx, ky, jx, jy, pjx, pjy; };
static const char* kName = "d1p12/swin_block_c64_add_dec0_nc";

static uint16_t f2h(float f) {  // RNE, normal range (test data)
  uint32_t u; std::memcpy(&u, &f, 4);
  uint32_t s = (u >> 16) & 0x8000u, a = u & 0x7fffffffu;
  if (a < 0x38800000u) return (uint16_t)s;
  return (uint16_t)(s | ((a + 0xC8000FFFu + ((a >> 13) & 1u)) >> 13));
}

struct Data {
  int rw = 1706, rh = 960, fw = 1728, fh = 960, H = 480, W = 864, ox = 2, oy = 2;
  size_t R = 0, M = 0;
  std::vector<uint8_t> x, xa, w, wd;
  std::vector<uint32_t> mot, lat;
  void make(unsigned seed) {
    R = (size_t)rw * rh, M = (size_t)H * W * 64;
    std::mt19937 g(seed);
    std::uniform_real_distribution<float> U(-1.f, 1.f);
    auto fp8 = [&](int maxexp) { return (uint8_t)(((g() & 1) << 7) | ((g() % (maxexp + 1)) << 3) | (g() % 8)); };
    auto putf = [](std::vector<uint8_t>& v, size_t off, float f) { std::memcpy(&v[off], &f, 4); };
    x.resize(M), xa.resize(M);
    for (auto& b : x) b = fp8(8);
    for (auto& b : xa) b = fp8(8);
    w.assign(49168, 0);
    for (int i = 0; i < 64; ++i) putf(w, 16 + 4 * i, 1.0f + 0.2f * U(g)), putf(w, 272 + 4 * i, 1.0f + 0.2f * U(g));
    for (int i = 0; i < 64; ++i) putf(w, 528 + 4 * i, 0.1f * U(g)), putf(w, 784 + 4 * i, 0.1f * U(g));
    for (int i = 0; i < 256; ++i) putf(w, 1040 + 4 * i, 0.1f * U(g));
    for (int i = 0; i < 512; ++i) putf(w, 2064 + 4 * i, 0.5f * U(g));
    for (size_t i = 4112; i < w.size(); ++i) w[i] = fp8(5);
    wd.assign(10752, 0);
    for (int i = 0; i < 80; ++i) putf(wd, 4 * i, 0.2f * U(g));
    for (int a = 0; a < 256; ++a) {
      const uint32_t ta = (g() & 0xffu) | ((uint32_t)f2h(0.5f + 0.5f * U(g)) << 16);
      std::memcpy(&wd[512 + 4 * a], &ta, 4);
      putf(wd, 512 + 4 * (256 + a), 0.1f + U(g) * U(g));
      putf(wd, 512 + 4 * (512 + a), U(g));
      const uint32_t zg = f2h(0.5f + 0.5f * U(g)) | ((uint32_t)f2h(U(g)) << 16);
      std::memcpy(&wd[512 + 4 * (768 + a)], &zg, 4);
      putf(wd, 512 + 4 * (1024 + a), std::max(-1.0f, (float)(int8_t)a / 127.0f));
    }
    for (int i = 5632; i < 10752; ++i) wd[i] = fp8(6);
    mot.resize(R), lat.resize(R);
    for (auto& m : mot) m = f2h(4.0f * U(g)) | ((uint32_t)f2h(4.0f * U(g)) << 16);
    for (auto& l : lat) l = g();
  }
};


static bool execute(const Data& d, Device& dev, int reset, std::vector<uint8_t>& output) {
  DevPtr p[10]{};
  const size_t sizes[10] = {d.M, d.M, d.w.size() + 4096, 12800, d.R * 4, d.R * 4, d.R * 8, d.R * 4, d.R * 4, d.R * 4};
  for (int i = 0; i < 10; ++i)
    if (!dev.alloc(&p[i], sizes[i]) || !dev.memset(p[i], 0xCD, sizes[i])) return false;
  if (!dev.upload(p[0], d.x.data(), d.M) || !dev.upload(p[1], d.xa.data(), d.M) ||
      !dev.upload(p[2], d.w.data(), d.w.size()) || !dev.upload(p[3], d.wd.data(), d.wd.size()) ||
      !dev.upload(p[4], d.mot.data(), d.R * 4) || !dev.upload(p[5], d.lat.data(), d.R * 4)) return false;
  const float dw = (float)(d.rw == 1920 ? 5760 : (d.rw == 853 ? 1919 : 3840));
  const float dh = (float)(d.rw == 1920 ? 3240 : (d.rw == 853 ? 1082 : 2160));
  DecParams params = {d.rw, d.rh, d.fw, d.fh, reset, 0, dw, dh, dw/d.rw, dh/d.rh, 0.25f, -0.125f, -0.3f, 0.2f};
  int h=d.H, w=d.W, ox=d.ox, oy=d.oy;
  void* args[15] = {&p[0], &p[1], &p[2], &h, &w, &ox, &oy, &p[4], &p[5], &p[6], &p[7], &p[8], &p[9], &p[3], &params};
  unsigned grid = (unsigned)((((d.W + d.ox + 3) / 4) * ((d.H + d.oy + 3) / 4) + 7) / 8);
  if (!dev.launch(kName, grid, 256, args) || !dev.sync()) return false;
  output.resize(d.R * 16);
  if (!dev.download(output.data(), p[6], d.R*8) || !dev.download(output.data()+d.R*8, p[8], d.R*4) ||
      !dev.download(output.data()+d.R*12, p[9], d.R*4)) return false;
  for (auto ptr : p) dev.free(ptr);
  return true;
}
int main(int argc, char** argv) {
  if (argc != 5) return std::printf("usage: tuned_decoder_diff <rocm> <native build> <HIP build> <seeds>\n"),2;
  int seeds = std::atoi(argv[4]);
  if (seeds<1) return 2;
  std::string error;
  auto hip=make_hip_device(argv[1],&error);
  if (!hip) return std::printf("HIP device: %s\n",error.c_str()),1;
  auto dx=make_d3d12_device(argv[2],&error);
  if (!dx) return std::printf("native device: %s\n",error.c_str()),1;
  const std::string code=std::string(argv[3])+"/dec1_dec0__pack12.co";
  if (!hip->load_module_tagged(code,"d1p12") || !dx->load_module_tagged(code,"d1p12"))
    return std::printf("load: hip=%s native=%s\n",hip->err().c_str(),dx->err().c_str()),1;
  unsigned long long bad=0,total=0;
  for (int shape=0;shape<3;++shape) for (int seed=0;seed<seeds;++seed) for (int reset=0;reset<2;++reset) {
    Data d;
    d.rw=shape==0?1706:(shape==1?853:1920); d.rh=shape==0?960:(shape==1?481:1080);
    d.fw=(d.rw+31)/32*32; d.fh=(d.rh+31)/32*32; d.H=d.fh/2; d.W=d.fw/2;
    d.ox=(seed&1)*2; d.oy=(seed&2); d.make(1234u+1000u*(unsigned)seed);
    std::vector<uint8_t> a,b;
    if (!execute(d,*hip,reset,a) || !execute(d,*dx,reset,b))
      return std::printf("execute: hip=%s native=%s\n",hip->err().c_str(),dx->err().c_str()),1;
    size_t different=0,first=0;
    for (size_t i=0;i<a.size();++i) if(a[i]!=b[i]) {if(!different) first=i; ++different;}
    bad+=different; total+=a.size();
    std::printf("decoder %dx%d seed %d reset %d: %zu differing of %zu bytes",d.rw,d.rh,seed,reset,different,a.size());
    if(different) std::printf(" first %zu hip=%02x native=%02x",first,a[first],b[first]);
    std::printf("\n"); std::fflush(stdout);
  }
  std::printf("decoder_diff: %s (%llu differing of %llu bytes)\n",bad?"DIFFERENT":"IDENTICAL",bad,total);
  return bad?3:0;
}
