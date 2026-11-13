// SPDX-License-Identifier: MIT
// Packing of the converter's preset-F model (model.wgcm, logical tensors of page 56.4, symbols of 57.7.2) into the
// kernels' buffer layouts: the same device image, blob names and scalars as clean/rr_f/runtime/tools/pack_model.py
// pack() of the same logical model (checked by tests/test_model_pack.py through the WrongGPU_RrModelDumpV1 hook).
// Layouts: clean/rr_f/gpu/layout.py (block stages), head/dx12/run_dx12.py (head), gpu/input_stage.py (front end).
// 56.4.5: a re-arrangement never changes a value (e4m3 codes and binary16 values moved, binary16 widened exactly).
#include "rr_model.h"

#include <math.h>
#include <string.h>

#include <algorithm>

#include "rr_tables.h"

namespace wgrr_rt {
namespace {

// ---- binary16 / binary32 helpers
float h2f(uint16_t h) {
  const uint32_t s = (uint32_t)(h & 0x8000u) << 16, e = (h >> 10) & 31u, m = h & 1023u;
  uint32_t u;
  if (e == 0) {
    if (m == 0) {
      u = s;
    } else {   // subnormal: m * 2^-24, exact in binary32
      float v = (float)m * 5.9604644775390625e-8f;
      memcpy(&u, &v, 4);
      u |= s;
    }
  } else if (e == 31) {
    u = s | 0x7f800000u | (m << 13);
  } else {
    u = s | ((e + 112u) << 23) | (m << 13);
  }
  float f;
  memcpy(&f, &u, 4);
  return f;
}
uint32_t fbits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

// ---- inventory of pages 56.4 / 57.7 (model/model_file.py expected_inventory, from the page text)
struct Geo { int C, H, Cn, Kc; };
Geo geometry(const std::string& part) {
  const std::string lv = part.substr(0, 2);
  if (part == "IN0") return {32, 2, 0, 0};
  if (part == "E1") return {64, 2, 96, 0};
  if (part == "E2") return {96, 4, 128, 0};
  if (part == "E3") return {128, 4, 160, 0};
  if (part == "E4") return {160, 8, 160, 0};
  if (part == "E5") return {160, 8, 0, 0};
  if (lv == "D4") return {160, 8, 0, 160};
  if (lv == "D3") return {128, 4, 0, 160};
  if (lv == "D2") return {96, 4, 0, 128};
  if (lv == "D1") return {64, 2, 0, 96};
  return {32, 2, 0, 64};   // D0
}
const char* const kParts[16] = {"IN0", "E1", "E2", "E3", "E4", "E5", "D4-F", "D4-R", "D3-F", "D3-R",
                                "D2-F", "D2-R", "D1-F", "D1-R", "D0-F", "D0-R"};
const char* const kWindow[10] = {"A_SCALE", "A_CLAMP", "A_CLAMPn", "A_P1", "A_P0", "ACT_CL", "ACT_CLn", "ACT_C1",
                                 "ACT_C2", "ACT_C3"};
const char* const kF32Sym[] = {"C_AMAX", "C_FCLIP", "C_GCL", "C_GCLn", "C_HDEC", "C_LWrb", "C_LWg", "C_MDL", "C_MDLn",
                               "C_SV1", "C_SV2", "C_SV3", "Gm", "Gd", "K2n", "K2x2n", "K3", "K4", "K5", "K7", "K7n",
                               "K8", "K10", "K11", "K12", "K13", "T_EFL", "TBW", "X_CL", "X_CLn", "X_EFL", "X_KEY",
                               "X_LMAX", "X_LWrb", "X_LWg", "X_MFL", "X_NRM", "S1CAP"};
enum { DT_F32 = 0, DT_U8 = 1, DT_I32 = 2, DT_F16 = 3, DT_U16 = 4, DT_U32 = 5, DT_E4M3 = 6 };

struct Want { std::string name; std::vector<uint32_t> shape; uint32_t dtype; };

std::vector<Want> part_inventory(const std::string& part) {
  const Geo g = geometry(part);
  const uint32_t C = g.C, H = g.H;
  std::vector<Want> w = {{"Wq", {H, C, 32}, DT_E4M3}, {"Wv", {H, C, 32}, DT_E4M3}, {"Tb", {H, 12, 16, 16}, DT_F16},
                         {"Wo", {H, 32, C}, DT_E4M3},  {"bo", {C}, DT_F16},          {"W1", {C, 4 * C}, DT_E4M3},
                         {"b1", {4 * C}, DT_F16},      {"W2", {4 * C, C}, DT_E4M3}, {"b2", {C}, DT_F16}};
  const bool single = part == "IN0" || part[0] == 'E';
  if (part == "E1" || part == "E2" || part == "E3" || part == "E4") {
    w.push_back({"Wm", {4, C, (uint32_t)g.Cn}, DT_E4M3});
    w.push_back({"bm", {(uint32_t)g.Cn}, DT_F16});
  }
  if (part == "E1") {
    w.push_back({"Wa", {4, 32, 64}, DT_E4M3});
    w.push_back({"ba", {64}, DT_F16});
  }
  const bool fuse = part.size() == 4 && part[3] == 'F', rec = part.size() == 4 && part[3] == 'R';
  if (fuse) {
    w.push_back({"Wup", {(uint32_t)g.Kc, 4, C}, DT_E4M3});
    w.push_back({"bup", {4, C}, DT_F16});
  }
  if (single) {
    w.push_back({"Wh", {C, 3}, DT_E4M3});
    w.push_back({"bh", {3}, DT_F16});
  } else if (rec && part != "D0-R") {
    w.push_back({"Wh", {C, 4}, DT_E4M3});
    w.push_back({"bh", {4}, DT_F16});
  }
  if (single || rec) w.push_back({"g", {1}, DT_F16});
  if (part == "IN0") {
    for (int b = 0; b < 2; ++b) {
      w.push_back({"FE.W1_" + std::to_string(b), {64, 32}, DT_E4M3});
      w.push_back({"FE.b1_" + std::to_string(b), {32}, DT_F16});
      w.push_back({"FE.W2_" + std::to_string(b), {32, 32}, DT_E4M3});
    }
    w.push_back({"FE.b2", {32}, DT_F16});
    w.push_back({"D", {2048}, DT_U16});
  }
  if (part == "D0-R")
    for (const char* n : {"s1", "s2", "K1"}) w.push_back({n, {1}, DT_F32});
  return w;
}

std::vector<Want> constants_inventory() {
  std::vector<Want> w;
  for (const char* n : kWindow) w.push_back({n, {1}, DT_F16});
  for (const char* n : {"WMAX", "Gh", "Gs", "RW_A", "RW_B", "RW_C", "RW_D", "RW_0"}) w.push_back({n, {1}, DT_F16});
  for (const char* p : kParts) {
    w.push_back({std::string("ra.") + p, {1}, DT_F16});
    w.push_back({std::string("m.") + p, {1}, DT_F16});
  }
  for (const char* lv : {"D4", "D3", "D2", "D1", "D0"}) w.push_back({std::string("f.") + lv, {1}, DT_F16});
  for (const char* n : kF32Sym) w.push_back({n, {1}, DT_F32});
  w.push_back({"TR", {513, 4}, DT_F32});
  w.push_back({"SH", {2, 11, 2}, DT_U32});
  return w;
}

bool check_group(const WgcmGroup& g, const std::vector<Want>& want, std::string* err) {
  if (g.tensors.size() != want.size()) return *err = "model.wgcm: group " + g.name + " has another tensor set (56.4)", false;
  for (const Want& w : want) {
    const WgcmTensor* t = g.find(w.name);
    if (!t) return *err = "model.wgcm: " + g.name + "/" + w.name + " missing (56.4)", false;
    if (t->shape != w.shape || t->dtype != w.dtype)
      return *err = "model.wgcm: " + g.name + "/" + w.name + " has another shape or type (56.4 / 57.7)", false;
  }
  return true;
}

// ---- device image (tools/pack_model.py Image)
struct Image {
  std::vector<uint8_t> bytes;
  uint64_t size = 0;
  std::map<std::string, std::pair<uint64_t, uint64_t>> entries;
  void add(const std::string& name, const void* data, uint64_t n) {
    const uint64_t off = (size + 255) / 256 * 256;
    if (bytes.size() < off + n) bytes.resize(off + n, 0);
    memcpy(bytes.data() + off, data, n);
    size = off + std::max<uint64_t>(n, 4);
    entries[name] = {off, n};
  }
  void add_f32(const std::string& name, const std::vector<float>& v) { add(name, v.data(), v.size() * 4); }
  void add_u8(const std::string& name, const std::vector<uint8_t>& v) { add(name, v.data(), v.size()); }
};

// fragments of W^T for W [K][N] (e4m3 codes, row-major): [N/16][K/16][32 lanes][8], lane l, byte i =
// W[16 ks + 8 (l >> 4) + i][16 nt + (l & 15)] (layout.frags(W.T) == run_dx12.frags(W)); columns >= N read 0
// (pad_rows of layout.py: Npad >= N columns)
void frags_T(const uint8_t* W, uint32_t K, uint32_t N, uint32_t Npad, std::vector<uint8_t>* out) {
  for (uint32_t nt = 0; nt < Npad / 16; ++nt)
    for (uint32_t ks = 0; ks < K / 16; ++ks)
      for (uint32_t l = 0; l < 32; ++l)
        for (uint32_t i = 0; i < 8; ++i) {
          const uint32_t k = 16 * ks + 8 * (l >> 4) + i, n = 16 * nt + (l & 15);
          out->push_back(n < N ? W[(size_t)k * N + n] : 0);
        }
}
// [nt][ks] fragments as a flat vector; sub-block (nt, ks) of 256 bytes
const uint8_t* frag_at(const std::vector<uint8_t>& f, uint32_t KS, uint32_t nt, uint32_t ks) {
  return f.data() + ((size_t)nt * KS + ks) * 256;
}

std::vector<float> widen16(const WgcmTensor& t) {
  std::vector<float> v(t.size / 2);
  for (size_t i = 0; i < v.size(); ++i) {
    uint16_t h;
    memcpy(&h, t.data + 2 * i, 2);
    v[i] = h2f(h);
  }
  return v;
}
// [values | cb | cs] with cb = the nearest integer (ties to even) and cs = value - cb (exact): the bias layout of the
// block stages since the exact step Y (gpu/layout.py bias_split / pack_weights Tb; kernels/rr_step_y.hlsli). Steps E / X
// read the values only.
double rne_int(double x) {   // nearest integer, ties to even (numpy round), independent of the FPU rounding mode
  if (!isfinite(x)) return x;
  double r = floor(x);
  const double d = x - r;
  if (d > 0.5 || (d == 0.5 && fmod(r, 2.0) != 0.0)) r += 1.0;
  return r;
}
std::vector<float> split3(const std::vector<float>& v) {
  std::vector<float> o(v);
  o.reserve(3 * v.size());
  for (float x : v) o.push_back((float)rne_int((double)x));
  for (float x : v) o.push_back((float)((double)x - rne_int((double)x)));
  return o;
}
uint16_t bits16(const WgcmTensor& t, size_t i = 0) {
  uint16_t h;
  memcpy(&h, t.data + 2 * i, 2);
  return h;
}
uint32_t bits32(const WgcmTensor& t, size_t i = 0) {
  uint32_t u;
  memcpy(&u, t.data + 4 * i, 4);
  return u;
}

// score-bias tiles of 51.3 step 2: B_h from the 12 stored 16 x 16 blocks; out [H][4][4][32][8] f32 with element v of
// lane 16 hf + c of tile (I, J) = B_h(16 I + c, 16 J + 8 hf + v) (layout.bias_tiles == run_dx12.bias_buffers' tb)
void bias_tiles(const WgcmTensor& Tb, uint32_t H, std::vector<float>* out) {
  static const int pairs[12][2] = {{0, 0}, {1, 0}, {0, 1}, {1, 1}, {0, 2}, {1, 2}, {0, 3}, {1, 3}, {2, 2}, {3, 2}, {2, 3}, {3, 3}};
  auto stored = [&](int I, int J) { for (int k = 0; k < 12; ++k) if (pairs[k][0] == I && pairs[k][1] == J) return k; return -1; };
  for (uint32_t h = 0; h < H; ++h)
    for (int I = 0; I < 4; ++I)
      for (int J = 0; J < 4; ++J)
        for (uint32_t l = 0; l < 32; ++l)
          for (uint32_t v = 0; v < 8; ++v) {
            const int a = (int)(l & 15), b = (int)(8 * (l >> 4) + v);   // row / column within the block
            int k = stored(I, J);
            uint32_t r = a, c = b;
            if (k < 0) k = stored(J, I), r = b, c = a;
            out->push_back(h2f(bits16(Tb, (((size_t)h * 12 + k) * 16 + r) * 16 + c)));
          }
}

uint32_t h2pair(uint16_t a, uint16_t b) { return (uint32_t)a | ((uint32_t)b << 16); }

// ---- host checks of the exact body step Z (gpu/stage.py y_bounds (L1_T form) and z_params) -------------------------
// e4m3 code -> value (NaN for 0x7f / 0xff)
double d8v(uint8_t c) {
  if ((c & 0x7f) == 0x7f) return NAN;
  const int e = (c >> 3) & 15, m = c & 7;
  const double v = e ? ldexp(8.0 + m, e - 10) : ldexp((double)m, -9);
  return (c & 0x80) ? -v : v;
}
// exponent of the lowest set bit of a nonzero e4m3 value (frexp significand of at most 4 bits)
int low_bit(double a) {
  int e;
  const double m = frexp(fabs(a), &e);
  int sig = (int)ldexp(m, 8), z = 0;
  while (!(sig & 1)) sig >>= 1, ++z;
  return e - 8 + z;
}
// one logical [rows][N] e4m3 matrix (heads stacked along the rows): A = max |v|, L1 = the largest sum of |v| over the
// 32 inputs of one 51.1 step of one output, g = lowest set bit over the nonzero values
struct ZStat { bool nan = false, any = false; double A = 0.0, L1 = 0.0; int g = 0; };
void zstat_add(const WgcmTensor& t, uint32_t N, ZStat* s) {
  const size_t rows = (size_t)t.size / N;
  for (size_t b = 0; b + 32 <= rows; b += 32)
    for (uint32_t n = 0; n < N; ++n) {
      double sum = 0.0;
      for (size_t k = b; k < b + 32; ++k) {
        const double v = d8v(t.data[k * N + n]);
        if (isnan(v)) { s->nan = true; continue; }
        sum += fabs(v);
        if (v != 0.0) {
          const int g = low_bit(v);
          s->g = s->any ? std::min(s->g, g) : g;
          s->any = true;
          s->A = std::max(s->A, fabs(v));
        }
      }
      s->L1 = std::max(s->L1, sum);
    }
}
uint32_t logical_n(const std::string& k, const WgcmTensor& t, uint32_t C, uint32_t Cn) {
  if (k == "Wq" || k == "Wv") return 32;
  if (k == "Wo" || k == "W2") return C;
  if (k == "W1" || k == "Wup") return 4 * C;
  if (k == "Wm") return Cn;
  if (k == "Wa") return 64;
  return t.shape.back();   // Wh [C][P]
}
struct ZHost { bool ok = false; double thr = 0.0; uint32_t zt0 = 0, zt1 = 0; };
ZHost z_host(const WgcmGroup* g, uint32_t C, uint32_t Cn) {
  ZHost z;
  auto T = [&](const char* n) { return g->find(n); };
  // y_bounds (L1_T form): every bounded tensor finite, L1 / 2 < 2^(g + 15), g <= 0, 448 L1 + L1 / 2 < 2^20 - 65520;
  // thr = min 2^(g + 24) - 448 L1 - 64; P: L1 = 32, g = -9
  double thr = INFINITY;
  bool ok = true;
  auto bound = [&](double L1, int gg) {
    if (!(L1 / 2 < ldexp(1.0, gg + 15) && gg <= 0 && 448.0 * L1 + L1 / 2 < 1048576.0 - 65520.0)) ok = false;
    thr = std::min(thr, ldexp(1.0, gg + 24) - 448.0 * L1 - 64.0);
  };
  bound(32.0, -9);
  for (const char* k : {"Wq", "Wv", "Wo", "W1", "W2", "Wm", "Wa", "Wup", "Wh"}) {
    const WgcmTensor* t = T(k);
    if (!t) continue;
    ZStat s;
    zstat_add(*t, logical_n(k, *t, C, Cn), &s);
    if (s.nan) ok = false;
    else if (s.any) bound(s.L1, s.g);
  }
  for (const char* k : {"bo", "b1", "b2", "bm", "ba", "bup", "bh", "Tb"}) {
    const WgcmTensor* t = T(k);
    if (!t) continue;
    double mx = 0.0;
    for (float v : widen16(*t)) {
      if (!isfinite(v)) ok = false;
      mx = std::max(mx, fabs((double)v));
    }
    if (strcmp(k, "Tb") != 0 && mx + 1.0 >= thr) ok = false;
  }
  z.ok = ok && thr > 64.0;
  z.thr = thr;
  // z_params: fast-step classes (word / byte of zt0 / zt1, int bit, tensors)
  struct Cls { int word, byte; uint32_t bit; const char* k[2]; };
  static const Cls cls[7] = {{0, 0, 1, {"Wq", "Wv"}}, {0, 1, 2, {"Wo", nullptr}}, {0, 2, 4, {"W1", nullptr}},
                             {0, 3, 8, {"W2", nullptr}}, {1, 0, 16, {"Wa", "Wup"}}, {1, 1, 32, {"Wh", nullptr}},
                             {-1, 0, 64, {"Wm", nullptr}}};
  uint32_t zt[2] = {0, 0}, bits = 0;
  for (const Cls& c : cls) {
    ZStat s;
    bool have = false;
    for (const char* k : c.k) {
      if (!k || !T(k)) continue;
      have = true;
      zstat_add(*T(k), logical_n(k, *T(k), C, Cn), &s);
    }
    if (!have) continue;
    uint32_t Tc = 0;
    bool iok = false;
    if (s.nan) {
      Tc = 0, iok = false;
    } else if (!s.any) {
      Tc = 0x7f, iok = true;
    } else {
      for (uint32_t t = 1; t < 0x80; ++t)
        if (s.L1 * d8v((uint8_t)(t - 1)) < ldexp(1.0, s.g + 15)) Tc = t;
      iok = s.L1 * 448.0 < ldexp(1.0, s.g + 24);
    }
    if (c.word >= 0) zt[c.word] |= Tc << (8 * c.byte);
    if (iok) bits |= c.bit;
  }
  z.zt0 = zt[0];
  z.zt1 = zt[1] | (bits << 16);
  return z;
}
uint32_t f32bits(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  return u;
}

}  // namespace

bool pack_wgcm(const std::vector<WgcmGroup>& groups, Model* out, std::string* err) {
  // inventory (56.4 / 57.7): every part, constants, optional constants_b2; nothing else
  const WgcmGroup* G[16] = {};
  const WgcmGroup *K = nullptr, *B2 = nullptr;
  for (const WgcmGroup& g : groups) {
    bool known = false;
    for (int p = 0; p < 16; ++p)
      if (g.name == kParts[p]) G[p] = &g, known = true;
    if (g.name == "constants") K = &g, known = true;
    if (g.name == "constants_b2") B2 = &g, known = true;
    if (!known) return *err = "model.wgcm: group " + g.name + " is not of preset F", false;
  }
  for (int p = 0; p < 16; ++p) {
    if (!G[p]) return *err = std::string("model.wgcm: group ") + kParts[p] + " missing", false;
    if (!check_group(*G[p], part_inventory(kParts[p]), err)) return false;
  }
  if (!K) return *err = "model.wgcm: group constants missing", false;
  if (!check_group(*K, constants_inventory(), err)) return false;
  if (B2 && !check_group(*B2, {{"C_SV4", {1}, DT_F32}, {"C_SV4n", {1}, DT_F32}}, err)) return false;
  auto T = [&](const WgcmGroup* g, const std::string& n) { return g->find(n); };
  auto KS = [&](const std::string& n) { return T(K, n); };

  Image img;
  std::map<std::string, uint32_t>& sc = out->scalars;
  sc.clear();
  // window constants (wc_dwords of gpu/stage.py)
  for (int i = 0; i < 5; ++i) sc["wc" + std::to_string(i)] = h2pair(bits16(*KS(kWindow[2 * i])), bits16(*KS(kWindow[2 * i + 1])));
  img.add("sig", kRrSigTable, sizeof(uint16_t) * 65536);
  // block stages (layout.pack_weights)
  for (int p = 0; p < 15; ++p) {
    const std::string name = kParts[p];
    const WgcmGroup* g = G[p];
    const Geo geo = geometry(name);
    const uint32_t C = geo.C, H = geo.H;
    {   // Wqv [h][q|v][ft 2][ks C/16]
      std::vector<uint8_t> v;
      for (uint32_t h = 0; h < H; ++h) {
        frags_T(T(g, "Wq")->data + (size_t)h * C * 32, C, 32, 32, &v);
        frags_T(T(g, "Wv")->data + (size_t)h * C * 32, C, 32, 32, &v);
      }
      img.add_u8(name + ".Wqv", v);
    }
    {
      std::vector<float> tb;
      bias_tiles(*T(g, "Tb"), H, &tb);
      img.add_f32(name + ".Tb", split3(tb));
    }
    {
      std::vector<uint8_t> v;
      for (uint32_t h = 0; h < H; ++h) frags_T(T(g, "Wo")->data + (size_t)h * 32 * C, 32, C, C, &v);
      img.add_u8(name + ".Wo", v);
    }
    img.add_f32(name + ".bo", split3(widen16(*T(g, "bo"))));
    {   // MLP chunk units j: W1^T rows 32 j .. 32 j + 31 [ft][ks], then W2^T [ct][K-slice 2 j + e]
      std::vector<uint8_t> w1, w2, mlp;
      frags_T(T(g, "W1")->data, C, 4 * C, 4 * C, &w1);   // [4C/16][C/16]
      frags_T(T(g, "W2")->data, 4 * C, C, C, &w2);       // [C/16][4C/16]
      const uint32_t nj = 4 * C / 32;
      for (uint32_t j = 0; j < nj; ++j) {
        for (uint32_t nt = 2 * j; nt < 2 * j + 2; ++nt)
          for (uint32_t ks = 0; ks < C / 16; ++ks) mlp.insert(mlp.end(), frag_at(w1, C / 16, nt, ks), frag_at(w1, C / 16, nt, ks) + 256);
        for (uint32_t ct = 0; ct < C / 16; ++ct)
          for (uint32_t e = 0; e < 2; ++e)
            mlp.insert(mlp.end(), frag_at(w2, 4 * C / 16, ct, 2 * j + e), frag_at(w2, 4 * C / 16, ct, 2 * j + e) + 256);
      }
      img.add_u8(name + ".MLP", mlp);
    }
    img.add_f32(name + ".b1", split3(widen16(*T(g, "b1"))));
    img.add_f32(name + ".b2", split3(widen16(*T(g, "b2"))));
    if (T(g, "Wm")) {
      std::vector<uint8_t> v;
      frags_T(T(g, "Wm")->data, 4 * C, geo.Cn, geo.Cn, &v);
      img.add_u8(name + ".Wm", v);
      img.add_f32(name + ".bm", split3(widen16(*T(g, "bm"))));
    }
    if (T(g, "Wa")) {
      std::vector<uint8_t> v;
      frags_T(T(g, "Wa")->data, 128, 64, 64, &v);
      img.add_u8(name + ".Wa", v);
      img.add_f32(name + ".ba", split3(widen16(*T(g, "ba"))));
    }
    if (T(g, "Wup")) {
      std::vector<uint8_t> v;
      frags_T(T(g, "Wup")->data, geo.Kc, 4 * C, 4 * C, &v);
      img.add_u8(name + ".Wup", v);
      img.add_f32(name + ".bup", split3(widen16(*T(g, "bup"))));
    }
    if (T(g, "Wh")) {
      const uint32_t P = T(g, "Wh")->shape[1];
      std::vector<uint8_t> v;
      frags_T(T(g, "Wh")->data, C, P, 16, &v);
      img.add_u8(name + ".Wh", v);
      std::vector<float> bh(16, 0.0f), b = widen16(*T(g, "bh"));
      std::copy(b.begin(), b.end(), bh.begin());
      img.add_f32(name + ".bh", split3(bh));
    }
    const std::string lv = name.substr(0, 2);
    const uint16_t f = name.size() == 4 && name[3] == 'F' ? bits16(*KS("f." + lv)) : 0;
    const uint16_t gg = T(g, "g") ? bits16(*T(g, "g")) : 0;
    sc[name + ".ra_m"] = h2pair(bits16(*KS("ra." + name)), bits16(*KS("m." + name)));
    sc[name + ".f_g"] = h2pair(f, gg);
    {   // exact body step Z: host checks (stage.y_bounds / z_params); zok 0: the stage runs step X
      const ZHost z = z_host(g, C, geo.Cn);
      sc[name + ".zok"] = z.ok ? 1u : 0u;
      sc[name + ".zthr"] = z.ok ? f32bits((float)std::min(z.thr, 3.0e38)) : 0u;
      sc[name + ".zt0"] = z.zt0;
      sc[name + ".zt1"] = z.zt1;
    }
  }
  // head (page 55): D0-R weights, biases, score-bias tiles, tables
  const WgcmGroup* Hd = G[15];
  {
    std::vector<uint8_t> wf;
    for (const char* n : {"Wq", "Wv"})
      for (uint32_t h = 0; h < 2; ++h) frags_T(T(Hd, n)->data + (size_t)h * 32 * 32, 32, 32, 32, &wf);
    for (uint32_t h = 0; h < 2; ++h) frags_T(T(Hd, "Wo")->data + (size_t)h * 32 * 32, 32, 32, 32, &wf);
    frags_T(T(Hd, "W1")->data, 32, 128, 128, &wf);
    frags_T(T(Hd, "W2")->data, 128, 32, 32, &wf);
    if (wf.size() != 56 * 256) return *err = "head weight buffer size", false;
    img.add_u8("head.wf", wf);
    std::vector<float> b(192, 0.0f), bo = widen16(*T(Hd, "bo")), b1 = widen16(*T(Hd, "b1")), b2 = widen16(*T(Hd, "b2"));
    std::copy(bo.begin(), bo.end(), b.begin());
    std::copy(b1.begin(), b1.end(), b.begin() + 32);
    std::copy(b2.begin(), b2.end(), b.begin() + 160);
    img.add_f32("head.bias", b);
    std::vector<float> tb;
    bias_tiles(*T(Hd, "Tb"), 2, &tb);
    img.add_f32("head.tb", tb);
    img.add("head.tab", kRrHeadTab, sizeof(uint16_t) * kRrHeadTabCount);
    float s1, s2;
    {
      const uint32_t u1 = bits32(*T(Hd, "s1")), u2 = bits32(*T(Hd, "s2"));
      memcpy(&s1, &u1, 4);
      memcpy(&s2, &u2, 4);
    }
    std::vector<float> rho(65536);
    out->rho_guarded = head_rho(s2, rho.data());
    img.add_f32("head.rho", rho);
    static const char* const kHk[12][2] = {{"k_as", "A_SCALE"}, {"k_acl", "A_CLAMP"}, {"k_acln", "A_CLAMPn"},
                                           {"k_p1", "A_P1"}, {"k_p0", "A_P0"}, {"k_actcl", "ACT_CL"},
                                           {"k_actcln", "ACT_CLn"}, {"k_c1", "ACT_C1"}, {"k_c2", "ACT_C2"},
                                           {"k_c3", "ACT_C3"}, {"k_ra", "ra.D0-R"}, {"k_m", "m.D0-R"}};
    for (auto& k : kHk) sc[std::string("head.") + k[0]] = bits16(*KS(k[1]));
    sc["head.k_g"] = bits16(*T(Hd, "g"));
    sc["head.k_s1"] = fbits(s1);
    sc["K1"] = bits32(*T(Hd, "K1"));
  }
  // input stage (52): front-end MLP fragments, dither, read tables
  const WgcmGroup* I0 = G[0];
  {
    std::vector<uint8_t> w1, w2;
    for (int br = 0; br < 2; ++br) frags_T(T(I0, "FE.W1_" + std::to_string(br))->data, 64, 32, 32, &w1);
    for (int br = 0; br < 2; ++br) frags_T(T(I0, "FE.W2_" + std::to_string(br))->data, 32, 32, 32, &w2);
    img.add_u8("in.W1", w1);
    img.add_u8("in.W2", w2);
    std::vector<float> b1 = widen16(*T(I0, "FE.b1_0")), b11 = widen16(*T(I0, "FE.b1_1"));
    b1.insert(b1.end(), b11.begin(), b11.end());
    img.add_f32("in.B1", b1);
    img.add_f32("in.B2", widen16(*T(I0, "FE.b2")));
    img.add("in.D", T(I0, "D")->data, T(I0, "D")->size);
    std::vector<float> un10(1024), sn8(256);
    for (int c = 0; c < 1024; ++c) un10[c] = (float)((double)c / 1023.0);
    for (int c = 0; c < 256; ++c) sn8[c] = (float)std::max((double)(int8_t)(uint8_t)c / 127.0, -1.0);
    img.add_f32("in.UN10", un10);
    img.add_f32("in.SN8", sn8);
  }
  static const char* const kInSym[25] = {"K4", "K5", "K7", "K7n", "K8", "K10", "K12", "C_HDEC", "C_FCLIP", "C_AMAX",
                                         "C_GCL", "C_GCLn", "C_LWrb", "C_LWg", "C_MDL", "C_MDLn", "Gm", "Gd", "Gh", "Gs",
                                         "C_SV1", "C_SV2", "C_SV3", "C_SV4", "C_SV4n"};
  for (const char* n : kInSym) {
    const std::string s = n;
    const WgcmTensor* t = (s == "C_SV4" || s == "C_SV4n") ? (B2 ? T(B2, s) : nullptr) : KS(s);
    if (!t) sc["sym." + s] = 0;
    else sc["sym." + s] = t->dtype == DT_F16 ? fbits(h2f(bits16(*t))) : bits32(*t);
  }
  // output stage (54), exposure (58), resolve (54b)
  for (const char* n : {"K2n", "K2x2n", "K3", "K4", "K5", "K7", "K8", "K10", "K11", "K12", "K13"})
    sc[std::string("out.") + n] = bits32(*KS(n));
  img.add("out.TR", KS("TR")->data, KS("TR")->size);
  for (const char* n : {"X_LWrb", "X_LWg", "X_LMAX", "X_KEY", "X_MFL", "X_EFL", "X_CL", "X_CLn", "X_NRM"})
    sc[std::string("exp.") + n] = bits32(*KS(n));
  for (const char* n : {"RW_A", "RW_B", "RW_C", "RW_D", "RW_0"}) sc[std::string("res.") + n] = bits16(*KS(n));
  for (const char* n : {"T_EFL", "TBW"}) sc[std::string("res.") + n] = bits32(*KS(n));
  sc["sym.C_HDEC"] = bits32(*KS("C_HDEC"));
  for (int p = 0; p < 2; ++p)
    for (int s = 0; s < 11; ++s)
      sc["SH." + std::to_string(p) + "." + std::to_string(s)] = bits32(*KS("SH"), (p * 11 + s) * 2) |
                                                                  (bits32(*KS("SH"), (p * 11 + s) * 2 + 1) << 8);
  out->flags = B2 ? 1u : 0u;
  out->version = 2;
  const uint64_t n = (img.size + 255) / 256 * 256;
  img.bytes.resize(n, 0);
  out->image = img.bytes;
  out->blobs = img.entries;
  return true;
}

// ---- rho: binary64 tanh with the guard of head/fp.py _guarded; undecided entries by a double-double evaluation -----
namespace {
struct dd { double hi, lo; };
dd two_sum(double a, double b) { const double s = a + b, bb = s - a; return {s, (a - (s - bb)) + (b - bb)}; }
dd quick(double a, double b) { const double s = a + b; return {s, b - (s - a)}; }
dd add(dd a, dd b) { dd s = two_sum(a.hi, b.hi); s.lo += a.lo + b.lo; return quick(s.hi, s.lo); }
dd mul(dd a, dd b) {
  const double p = a.hi * b.hi, e = fma(a.hi, b.hi, -p);
  return quick(p, e + (a.hi * b.lo + a.lo * b.hi));
}
dd mul_d(dd a, double b) { const double p = a.hi * b, e = fma(a.hi, b, -p); return quick(p, e + a.lo * b); }
dd div(dd a, dd b) {
  const double q1 = a.hi / b.hi;
  dd r = add(a, mul_d(b, -q1));
  const double q2 = r.hi / b.hi;
  r = add(r, mul_d(b, -q2));
  const double q3 = r.hi / b.hi;
  return add(quick(q1, q2), dd{q3, 0.0});
}
// e^x - 1 for |x| <= 81 in double-double: x = k ln2 + r, |r| <= ln2 / 2, expm1(r) by its series, then
// 2^k (expm1(r) + 1) - 1
dd expm1_dd(double x) {
  static const dd ln2 = {0.6931471805599453094, 2.319046813846299558e-17};
  const double k = nearbyint(x / ln2.hi);
  const dd r = add(dd{x, 0.0}, mul_d(ln2, -k));
  dd term = r, sum = r;
  for (int n = 2; n < 40; ++n) {
    term = div(mul(term, r), dd{(double)n, 0.0});
    sum = add(sum, term);
    if (fabs(term.hi) < 1e-40 * fabs(sum.hi)) break;
  }
  if (k == 0) return sum;
  const double s = ldexp(1.0, (int)k);
  return add(dd{sum.hi * s, sum.lo * s}, dd{s - 1.0, 0.0});   // 2^k - 1 exact for |k| <= 117
}
float tanh_dd(float a) {
  if (a == 0.0f || isnan(a)) return a;
  const double x = fabs((double)a);
  if (x > 40.0) return a > 0 ? 1.0f : -1.0f;
  const dd em = expm1_dd(2.0 * x);                            // tanh x = expm1(2x) / (expm1(2x) + 2)
  const dd t = div(em, add(em, dd{2.0, 0.0}));
  // round the double-double value once to binary32: hi rounded, corrected by lo at a tie / boundary
  float r = (float)t.hi;
  const double rd = (double)r, d = t.hi - rd + t.lo;   // exact difference to the candidate
  const double ulp = (double)nextafterf(r, INFINITY) - rd, ulpd = rd - (double)nextafterf(r, -INFINITY);
  if (d > 0 && d > ulp / 2) r = nextafterf(r, INFINITY);
  else if (d < 0 && -d > ulpd / 2) r = nextafterf(r, -INFINITY);
  return a < 0 ? -r : r;
}
}  // namespace

uint32_t head_rho(float s2, float* out, bool force_exact) {
  uint32_t guarded = 0;
  for (uint32_t b = 0; b < 65536; ++b) {
    volatile float v = h2f((uint16_t)b), s = s2;
    volatile float a = v * s;                                  // f32(y17 s2), one rounding, subnormals kept
    const double t = tanh((double)a);
    float r = (float)t;
    const float lo = (float)(t * (1 - 1e-12)), hi = (float)(t * (1 + 1e-12));
    const bool bad = isfinite(t) && (lo != hi || lo != r);
    if (bad || force_exact) {
      if (bad) guarded++;
      if (!isnan(a)) r = tanh_dd(a);
    }
    if (isnan(r)) {   // NaN inputs: the quiet NaN 0x7fc00000 (as the reference table; no page gives NaN payloads)
      const uint32_t qn = 0x7fc00000u;
      memcpy(&r, &qn, 4);
    }
    out[b] = r;
  }
  return guarded;
}

}  // namespace wgrr_rt
