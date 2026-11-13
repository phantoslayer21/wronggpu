// SPDX-License-Identifier: MIT
// Model data loading and packing (see model.h). Layouts: clean/FORMATS.md.
#include "model.h"
#include "embedded.h"

#include <cmath>
#include <cstring>
#include <fstream>

#include "../converter/src/json.h"
#include "../converter/src/sha256.h"
#include "fpconv.h"

namespace wgrt {

namespace {

const int kHeadDim = 32;
const size_t kFrag = 256, kAlign = 256;

const std::map<std::string, SwinStageInfo>& swin_table() {
  static const std::map<std::string, SwinStageInfo> t = {
      {"enc1", {'e', 64, 2, 256, 96, 0}},   {"enc2", {'e', 96, 4, 384, 128, 0}},  {"b0", {'b', 128, 4, 512, 0, 0}},
      {"b1", {'b', 128, 4, 512, 0, 0}},     {"b2", {'b', 128, 4, 512, 0, 0}},     {"b3", {'b', 128, 4, 512, 0, 0}},
      {"b4", {'b', 128, 4, 512, 0, 0}},     {"b5", {'b', 128, 4, 512, 0, 0}},     {"dec2", {'d', 96, 4, 384, 0, 128}},
      {"dec1", {'d', 64, 2, 256, 0, 96}}};
  return t;
}

struct TensorSpec {
  std::string name;
  std::vector<int> shape;
  bool u8;
};

std::vector<TensorSpec> stage_specs(const std::string& stage) {
  if (stage == "enc0")
    return {{"W", {64, 80}, false}, {"b", {64}, false}, {"T", {8, 16, 16}, true}, {"sigma1", {1}, false},
            {"sigma2", {1}, false}};
  if (stage == "dec0") return {{"W", {4, 17, 64}, false}, {"b", {4, 17}, false}, {"S", {1}, false}, {"T", {1}, false}};
  if (stage == "post") return {{"s", {1}, false}};
  const SwinStageInfo& s = swin_table().at(stage);
  const int C = s.C, h = s.heads, m = s.mlp;
  std::vector<TensorSpec> v = {{"gamma1", {C}, false},         {"gamma2", {C}, false},
                               {"Wqk", {h, kHeadDim, C}, false}, {"Wv", {h, kHeadDim, C}, false},
                               {"B", {h, 16, 16}, false},       {"Wp", {C, kHeadDim * h}, false},
                               {"bp", {C}, false},              {"W1", {m, C}, false},
                               {"b1", {m}, false},              {"W2", {C, m}, false},
                               {"b2", {C}, false}};
  if (s.kind == 'e') {
    v.push_back({"Wm", {s.C_out, 4 * C}, false});
    v.push_back({"bm", {s.C_out}, false});
  } else if (s.kind == 'd') {
    v.push_back({"We", {4 * C, s.C_low}, false});
    v.push_back({"be", {4 * C}, false});
  }
  return v;
}

size_t numel(const std::vector<int>& shape) {
  size_t n = 1;
  for (int d : shape) n *= (size_t)d;
  return n;
}

bool file_exists(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  return (bool)f;
}

// ---- packing helpers (ports of clean/runner/packing.py) -------------------------------------------------------------

using Order = std::vector<int>;  // R or K order; -1 = padding (zero)

struct Mat {
  const float* d;
  int rows, cols;
  float at(int r, int c) const { return (r < 0 || c < 0) ? 0.0f : d[(size_t)r * cols + c]; }
};

// Fragments of M: [mt][kc] (row-major), each 256 bytes: lane 16h + r, byte j = M[R[16 mt + r], K[16 kc + 8h + j]].
std::vector<std::vector<uint8_t>> frags(const Mat& M, const Order& R, const Order& K) {
  const int MT = (int)R.size() / 16, KC = (int)K.size() / 16;
  std::vector<std::vector<uint8_t>> out((size_t)MT * KC, std::vector<uint8_t>(kFrag));
  for (int mt = 0; mt < MT; ++mt)
    for (int kc = 0; kc < KC; ++kc) {
      auto& f = out[(size_t)mt * KC + kc];
      for (int h = 0; h < 2; ++h)
        for (int r = 0; r < 16; ++r)
          for (int j = 0; j < 8; ++j)
            f[h * 128 + r * 8 + j] = fp8_encode(M.at(R[16 * mt + r], K[16 * kc + 8 * h + j]));
    }
  return out;
}

void put(std::vector<uint8_t>* b, const void* p, size_t n) {
  const uint8_t* q = (const uint8_t*)p;
  b->insert(b->end(), q, q + n);
}
void put_f32(std::vector<uint8_t>* b, float v) { put(b, &v, 4); }
void pad_to(std::vector<uint8_t>* b, size_t n) { b->resize(n, 0); }

// Tile-ordered vector: packed[16 mt + 8h + v] = vec[R[16 mt + drow[h][v]]].
void put_tile_vector(std::vector<uint8_t>* b, const float* vec, const Order& R, const DRow& drow) {
  for (size_t mt = 0; mt < R.size() / 16; ++mt)
    for (int h = 0; h < 2; ++h)
      for (int v = 0; v < 8; ++v) {
        const int r = R[16 * mt + drow[h][v]];
        put_f32(b, r < 0 ? 0.0f : vec[r]);
      }
}

Order rows_identity(int n) {
  Order R(n);
  for (int i = 0; i < n; ++i) R[i] = i;
  return R;
}
Order rows_layout_l(int C, const DRow& d) {
  Order R(C);
  for (int mt = 0; mt < C / 16; ++mt)
    for (int h = 0; h < 2; ++h)
      for (int v = 0; v < 8; ++v) R[16 * mt + d[h][v]] = h * (C / 2) + 8 * mt + v;
  return R;
}
Order rows_expand(int C, const DRow& d) {
  const int G = C / 16;
  Order R(4 * C);
  for (int k = 0; k < 4; ++k)
    for (int g = 0; g < G; ++g)
      for (int h = 0; h < 2; ++h)
        for (int v = 0; v < 8; ++v) R[16 * (k * G + g) + d[h][v]] = k * C + h * (C / 2) + 8 * g + v;
  return R;
}
Order kord_layout_l(int C) {
  Order K(C);
  for (int g = 0; g < C / 16; ++g)
    for (int h = 0; h < 2; ++h)
      for (int j = 0; j < 8; ++j) K[16 * g + 8 * h + j] = h * (C / 2) + 8 * g + j;
  return K;
}
Order kord_drow(int n, const DRow& d) {
  Order K(n);
  for (int g = 0; g < n / 16; ++g)
    for (int h = 0; h < 2; ++h)
      for (int j = 0; j < 8; ++j) K[16 * g + 8 * h + j] = 16 * g + d[h][j];
  return K;
}
Order kord_merge(int C) {
  Order K(4 * C);
  for (int g = 0; g < C / 4; ++g)
    for (int h = 0; h < 2; ++h)
      for (int j = 0; j < 8; ++j) K[16 * g + 8 * h + j] = h * 2 * C + 8 * g + j;
  return K;
}

struct BlockLayout {
  size_t stream, head_bytes, hpu, mlp_bytes, nmlp, size;
};
BlockLayout block_layout(int C, int heads, int mlp) {
  const size_t KC = C / 16, MT = mlp / 16;
  BlockLayout o;
  o.stream = 16 + 4 * C * 4 + 4 * mlp + 4 * heads * 256;   // FLAGS, G1, G2, BP, B2 (4C each), B1 (4 mlp), BIAS
  o.head_bytes = (4 * KC + 2 * KC) * kFrag;
  const size_t limit = 16384 / (2 * KC * kFrag);
  size_t d = limit < MT ? limit : MT;
  while (d > 1 && MT % d) --d;
  o.hpu = d;
  o.mlp_bytes = 2 * KC * o.hpu * kFrag;
  o.nmlp = MT / o.hpu;
  o.size = o.stream + heads * o.head_bytes + o.nmlp * o.mlp_bytes;
  return o;
}

size_t align_up(size_t n) { return (n + kAlign - 1) / kAlign * kAlign; }

const Tensor& T(const StageTensors& t, const char* name) { return t.at(name); }

// runner/packing.py block_bounds: true if the weights alone prove that no FP8 conversion inside the block can exceed
// 460 (every RMSNorm output has L2 norm <= max|gamma| sqrt(C) for any input, q8 adds at most 1/16 relative, and each
// projection row is bounded by its L2 norm times that), so the block kernels may skip their clamps (`_nc`).
double max_abs(const Tensor& x) {
  double m = 0.0;
  for (float v : x.f) m = std::fmax(m, std::fabs((double)v));
  return m;
}
double row_l2(const float* w, int cols) {
  double s = 0.0;
  for (int c = 0; c < cols; ++c) s += (double)w[c] * (double)w[c];
  return std::sqrt(s);
}
bool block_safe(const StageTensors& t, int C, int heads, int mlp) {
  const double root_c = std::sqrt((double)C), limit = 460.0;
  const double na = 1.07 * max_abs(T(t, "gamma1")) * root_c + 0.01 * root_c;
  const double nm = 1.07 * max_abs(T(t, "gamma2")) * root_c + 0.01 * root_c;
  double wqk = 0.0, wv = 0.0, hidden = 0.0;
  for (int r = 0; r < heads * kHeadDim; ++r) {
    wqk = std::fmax(wqk, row_l2(T(t, "Wqk").f.data() + (size_t)r * C, C));
    wv = std::fmax(wv, row_l2(T(t, "Wv").f.data() + (size_t)r * C, C));
  }
  for (int r = 0; r < mlp; ++r)
    hidden = std::fmax(hidden, row_l2(T(t, "W1").f.data() + (size_t)r * C, C) * nm +
                                   std::fabs((double)T(t, "b1").f[r]));
  return na <= limit && wqk * na <= limit && wv * na <= limit && nm <= limit && hidden <= limit;
}

void pack_block(const StageTensors& t, const SwinStageInfo& s, const DRow& d, std::vector<uint8_t>* b) {
  const int C = s.C, heads = s.heads, mlp = s.mlp, D = heads * kHeadDim, KC = C / 16, MT = mlp / 16;
  const Order RL = rows_layout_l(C, d), KL = kord_layout_l(C);
  const uint32_t flags[4] = {block_safe(t, C, heads, mlp) ? 1u : 0u, 0u, 0u, 0u};  // FLAGS (kernels/swin.hip)
  put(b, flags, sizeof(flags));
  put(b, T(t, "gamma1").f.data(), 4 * C);
  put(b, T(t, "gamma2").f.data(), 4 * C);
  put_tile_vector(b, T(t, "bp").f.data(), RL, d);
  put_tile_vector(b, T(t, "b2").f.data(), RL, d);
  put_tile_vector(b, T(t, "b1").f.data(), rows_identity(mlp), d);
  const float* B = T(t, "B").f.data();
  for (int hd = 0; hd < heads; ++hd)
    for (int q = 0; q < 16; ++q)
      for (int h = 0; h < 2; ++h)
        for (int v = 0; v < 8; ++v) put_f32(b, B[(hd * 16 + q) * 16 + d[h][v]]);
  const auto wp = frags({T(t, "Wp").f.data(), C, D}, RL, kord_drow(D, d));           // [KC][2 heads]
  const auto w1 = frags({T(t, "W1").f.data(), mlp, C}, rows_identity(mlp), KL);       // [MT][KC]
  const auto w2 = frags({T(t, "W2").f.data(), C, mlp}, RL, kord_drow(mlp, d));        // [KC][MT]
  for (int hd = 0; hd < heads; ++hd) {
    const auto wq = frags({T(t, "Wqk").f.data() + (size_t)hd * kHeadDim * C, kHeadDim, C}, rows_identity(32), KL);
    const auto wv = frags({T(t, "Wv").f.data() + (size_t)hd * kHeadDim * C, kHeadDim, C}, rows_identity(32), KL);
    for (int mt = 0; mt < 2; ++mt)
      for (int kc = 0; kc < KC; ++kc) {
        put(b, wq[mt * KC + kc].data(), kFrag);
        put(b, wv[mt * KC + kc].data(), kFrag);
      }
    for (int mt = 0; mt < KC; ++mt)
      for (int dt = 0; dt < 2; ++dt) put(b, wp[(size_t)mt * 2 * heads + 2 * hd + dt].data(), kFrag);
  }
  for (int ht = 0; ht < MT; ++ht) {
    for (int kc = 0; kc < KC; ++kc) put(b, w1[(size_t)ht * KC + kc].data(), kFrag);
    for (int mt = 0; mt < KC; ++mt) put(b, w2[(size_t)mt * MT + ht].data(), kFrag);
  }
}

void put_frags(std::vector<uint8_t>* b, const std::vector<std::vector<uint8_t>>& f) {
  for (auto& x : f) put(b, x.data(), kFrag);
}

bool pack_swin(const std::string& stage, const StageTensors& t, const DRow& d, std::vector<uint8_t>* b) {
  const SwinStageInfo& s = swin_table().at(stage);
  pack_block(t, s, d, b);
  const BlockLayout lay = block_layout(s.C, s.heads, s.mlp);
  if (b->size() != lay.size) return false;
  pad_to(b, align_up(b->size()));  // every block section is padded to 256 bytes (runner/packing.py pack_stage)
  if (s.kind == 'b') return true;
  if (s.kind == 'e') {
    const Order RL = rows_layout_l(s.C_out, d);
    put_tile_vector(b, T(t, "bm").f.data(), RL, d);
    put_frags(b, frags({T(t, "Wm").f.data(), s.C_out, 4 * s.C}, RL, kord_merge(s.C)));
  } else {
    const Order RE = rows_expand(s.C, d);
    put_tile_vector(b, T(t, "be").f.data(), RE, d);
    put_frags(b, frags({T(t, "We").f.data(), 4 * s.C, s.C_low}, RE, kord_layout_l(s.C_low)));
  }
  return true;
}

// enc0 (runner/enc0.py): K order of the projection, sub-pixel p in blocks 2p, 2p + 1, owner half o = p / 2.
void pack_enc0(const StageTensors& t, const DRow& d, std::vector<uint8_t>* b) {
  Order K(128, -1);
  for (int p = 0; p < 4; ++p) {
    const int o = p / 2;
    for (int h = 0; h < 2; ++h)
      for (int j = 0; j < 8; ++j) {
        K[16 * (2 * p) + 8 * h + j] = 20 * p + (h == o ? j : 8 + j);
        if (h == o && j < 4) K[16 * (2 * p + 1) + 8 * h + j] = 20 * p + 16 + j;
      }
  }
  const Order RL = rows_layout_l(64, d);
  put_tile_vector(b, T(t, "b").f.data(), RL, d);                    // 0
  put_f32(b, T(t, "sigma1").f[0]);                                  // 256
  put_f32(b, T(t, "sigma2").f[0]);
  pad_to(b, 512);
  put(b, T(t, "T").u.data(), 2048);                                 // 512
  put_frags(b, frags({T(t, "W").f.data(), 64, 80}, RL, K));          // 2560
}

// dec0 (runner/dec0.py): rows R_C, K order K_C, tables of functions of FP8 codes.
void pack_dec0(const StageTensors& t, const DRow& d, std::vector<uint8_t>* b) {
  Order R(80, -1);
  for (int mt = 0; mt < 5; ++mt)
    for (int h = 0; h < 2; ++h)
      for (int v = 0; v < 8; ++v) {
        const int i = 8 * mt + v;
        if (i < 34) R[16 * mt + d[h][v]] = 34 * h + i;
      }
  Order K(64);
  for (int kb = 0; kb < 4; ++kb)
    for (int h = 0; h < 2; ++h)
      for (int j = 0; j < 8; ++j) K[16 * kb + 8 * h + j] = 32 * (kb / 2) + 16 * h + 8 * (kb % 2) + j;
  put_tile_vector(b, T(t, "b").f.data(), R, d);                     // 0
  const float S = T(t, "S").f[0], Tc = T(t, "T").f[0];
  put_f32(b, S);                                                    // 320
  put_f32(b, Tc);
  pad_to(b, 512);
  // Tables of kernels/decoder.hip (FORMATS.md; runner/dec0.py tables()): A code | C0, sx, rho, z | g, snorm.
  // One float operation per statement, so that no compiler fuses a multiply-add.
  uint32_t tab[5][256];
  for (int c = 0; c < 256; ++c) {
    double av = e4m3_value((uint8_t)c);
    const float a = std::isnan(av) ? 0.0f : (float)av;
    const float g = rn16(1.0f / (1.0f + std::exp(-a)));              // RN16(sigma(a)), sigma in single
    const float sx = S * g;
    const float sx2 = sx * sx;
    const float acode = std::nearbyint(std::fmin(std::fmax(g, 0.0f), 1.0f) * 255.0f);
    tab[0][c] = (uint32_t)acode | ((uint32_t)f32_to_f16(sx2) << 16);
    tab[1][c] = f32_bits(sx);
    tab[2][c] = f32_bits((float)std::tanh((double)(Tc * a)));       // rho = RN32(tanh(RN32(T a)))
    const float th_half = rn16((float)std::tanh((double)(a / 2.0f)));  // th(a / 2); RN16 of the double tanh, see below
    const float half_th = 0.5f * th_half;
    const float z = rn16(0.5f + half_th);
    const float th = rn16((float)std::tanh((double)a));              // th(a)
    tab[3][c] = (uint32_t)f32_to_f16(z) | ((uint32_t)f32_to_f16(th) << 16);
    const int k = (int)(int8_t)(uint8_t)c;
    tab[4][c] = f32_bits((float)(k / 127.0 < -1.0 ? -1.0 : k / 127.0));  // snorm value, -128 -> -1
  }
  put(b, tab, sizeof(tab));                                         // 512
  put_frags(b, frags({T(t, "W").f.data(), 68, 64}, R, K));          // 5632
}

}  // namespace

// The th tables round the double tanh to float first and then to FP16. For FP8 arguments this equals one rounding
// of the exact value: every exact result is at least ~10 FP32 ulps from an FP16 rounding midpoint
// (clean/tests/test_decoder_reference.py, test_tables_are_unambiguous).

const std::vector<std::string>& model_stages() {
  static const std::vector<std::string> s = {"enc0", "enc1", "enc2", "b0", "b1", "b2", "b3",
                                             "b4",   "b5",   "dec2", "dec1", "dec0", "post"};
  return s;
}

const char* const kFilterStage = "filter";
const char* const kFilterTaps = "output_taps";

bool model_output_taps(const std::map<std::string, StageTensors>& model, float taps[5]) {
  const auto st = model.find(kFilterStage);
  if (st == model.end()) return false;
  const auto t = st->second.find(kFilterTaps);
  if (t == st->second.end() || t->second.f.size() != 5) return false;
  for (int k = 0; k < 5; ++k) taps[k] = t->second.f[k];
  return true;
}

const SwinStageInfo* swin_stage_info(const std::string& stage) {
  auto it = swin_table().find(stage);
  return it == swin_table().end() ? nullptr : &it->second;
}

DRow drow_gfx12() {
  DRow d;
  for (int h = 0; h < 2; ++h)
    for (int v = 0; v < 8; ++v) d[h][v] = v + 8 * h;
  return d;
}

size_t swin_tail_offset(const std::string& stage) {
  const SwinStageInfo& s = swin_table().at(stage);
  return align_up(block_layout(s.C, s.heads, s.mlp).size);
}

namespace {

// The optional group "filter" (spec 08 draft 2 section 7.2).
const TensorSpec& filter_spec() {
  static const TensorSpec s = {kFilterTaps, {5}, false};
  return s;
}

// One raw tensor file with its checks (size, value range).
bool read_raw_tensor(const std::string& path, const TensorSpec& ts, Tensor* out, std::string* err) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    *err = "model data missing: " + path;
    return false;
  }
  const size_t n = numel(ts.shape), want = n * (ts.u8 ? 1 : 4);
  const size_t got = (size_t)f.tellg();
  if (got != want) {
    *err = "model data damaged: " + path + " has " + std::to_string(got) + " bytes, expected " +
           std::to_string(want);
    return false;
  }
  f.seekg(0);
  Tensor t;
  t.shape = ts.shape;
  if (ts.u8) {
    t.u.resize(n);
    f.read((char*)t.u.data(), (std::streamsize)n);
    for (uint8_t v : t.u)
      if (v > 127) {
        *err = "model data damaged: " + path + " has a value above 127";
        return false;
      }
  } else {
    t.f.resize(n);
    f.read((char*)t.f.data(), (std::streamsize)want);  // little-endian hosts only (x86-64)
    for (float v : t.f)
      if (!std::isfinite(v)) {
        *err = "model data damaged: " + path + " has a non-finite value";
        return false;
      }
  }
  if (!f) {
    *err = "cannot read " + path;
    return false;
  }
  *out = std::move(t);
  return true;
}

}  // namespace

bool load_raw_model(const std::string& model_dir, std::map<std::string, StageTensors>* out, std::string* err) {
  std::string root = model_dir + "/raw";
  if (!file_exists(root + "/enc0/W.bin")) root = model_dir;
  for (const std::string& stage : model_stages()) {
    StageTensors st;
    for (const TensorSpec& ts : stage_specs(stage)) {
      const std::string path = root + "/" + stage + "/" + ts.name + ".bin";
      if (!read_raw_tensor(path, ts, &st[ts.name], err)) return false;
    }
    (*out)[stage] = std::move(st);
  }
  // Optional group filter: raw/filter/output_taps.bin exists only when the converter found the weights (spec 08
  // section 10).
  const std::string taps = root + "/" + kFilterStage + "/" + kFilterTaps + ".bin";
  if (file_exists(taps)) {
    StageTensors st;
    if (!read_raw_tensor(taps, filter_spec(), &st[kFilterTaps], err)) return false;
    (*out)[kFilterStage] = std::move(st);
  }
  return true;
}

namespace {

bool read_all(const std::string& path, std::vector<uint8_t>* out) {  // on disk, else built in (the map; embedded.h)
  std::vector<char> bytes;
  if (!wg_read_file(path, &bytes)) return false;
  out->assign(bytes.begin(), bytes.end());
  return true;
}

// Little-endian reader with bounds checks.
struct Reader {
  const std::vector<uint8_t>& d;
  size_t pos, end;
  bool ok = true;
  bool need(size_t n) {
    if (!ok || pos + n > end || pos + n < pos) ok = false;
    return ok;
  }
  uint64_t le(int n) {
    if (!need((size_t)n)) return 0;
    uint64_t v = 0;
    for (int i = 0; i < n; ++i) v |= (uint64_t)d[pos + i] << (8 * i);
    pos += (size_t)n;
    return v;
  }
  std::string str() {
    const size_t n = (size_t)le(2);
    if (!need(n)) return std::string();
    std::string s((const char*)&d[pos], n);
    pos += n;
    return s;
  }
  void bytes(uint8_t* out, size_t n) {
    if (!need(n)) return;
    std::memcpy(out, &d[pos], n);
    pos += n;
  }
};

bool check_values(const std::string& where, const Tensor& t, std::string* err) {
  for (float v : t.f)
    if (!std::isfinite(v)) return *err = "model data damaged: " + where + " has a non-finite value", false;
  for (uint8_t v : t.u)
    if (v > 127) return *err = "model data damaged: " + where + " has a value above 127", false;
  return true;
}

}  // namespace

bool load_wgcm_model(const std::string& path, const std::string& map_path, std::map<std::string, StageTensors>* out,
                     std::string* err, bool* other_map) {
  if (other_map) *other_map = false;
  std::vector<uint8_t> d;
  if (!read_all(path, &d)) return *err = "model data missing: " + path, false;
  const std::string bad = "model data damaged: " + path;
  if (d.size() < 88 + 32 || std::memcmp(d.data(), "WGCMODEL", 8) != 0)
    return *err = bad + " is not a WrongGPU model file", false;
  const wgc::Digest whole = wgc::Sha256::hash(d.data(), d.size() - 32);
  if (std::memcmp(whole.data(), d.data() + d.size() - 32, 32) != 0) return *err = bad + " fails its integrity check", false;
  Reader r{d, 8, d.size() - 32};
  const uint64_t version = r.le(4), nstages = r.le(4);
  if (version != 1) return *err = bad + ": unknown container version " + std::to_string(version), false;
  wgc::Digest key, map_hash;
  r.bytes(key.data(), 32);
  r.bytes(map_hash.data(), 32);
  const uint64_t isize = r.le(8);
  if (!r.ok || isize > d.size()) return *err = bad + ": bad header", false;

  // The map shipped with the runtime: its digests are the reference.
  std::vector<uint8_t> mbytes;
  if (!read_all(map_path, &mbytes)) return *err = "converter map missing: " + map_path, false;
  // The file names the map it was converted with (its SHA-256). Another map than the one shipped here is not a
  // reason to refuse the file by itself: a file written with an earlier version of the map (before the group
  // "filter" existed; spec 08 section 7.2: it "counts as not present") holds the same stages, and each of them is
  // verified below against THIS map's digests. Only a file whose content this map does not describe is refused.
  const bool same_map = wgc::Sha256::hash(mbytes.data(), mbytes.size()) == map_hash;
  const std::string not_this_map =
      "model data does not match this installation (converted with a different map): " + path;
  wgc::Json map;
  std::string jerr;
  if (!wgc::parse_json(std::string(mbytes.begin(), mbytes.end()), map, jerr))
    return *err = "converter map damaged: " + map_path + " (" + jerr + ")", false;
  // A stage's digest is one hex string or, for a group whose tensors come from a source with variants (spec 08
  // draft 2 section 5: "filter"), a list with one entry per variant; the file's stage must equal one of them.
  std::map<std::string, std::vector<wgc::Digest>> map_digest;
  const wgc::Json* stages = map.find("stages");
  if (!stages || !stages->is_array()) return *err = "converter map damaged: " + map_path + " has no stages", false;
  for (const wgc::Json& st : stages->items) {
    const wgc::Json* nm = st.find("stage");
    const wgc::Json* dg = st.find("digest");
    const std::string bad_entry = "converter map damaged: " + map_path + " (stage entry)";
    if (!nm || !nm->is_string() || !dg || !(dg->is_string() || (dg->is_array() && !dg->items.empty())))
      return *err = bad_entry, false;
    std::vector<wgc::Digest>& list = map_digest[nm->text];
    const size_t count = dg->is_array() ? dg->items.size() : 1;
    for (size_t i = 0; i < count; ++i) {
      const wgc::Json& one = dg->is_array() ? dg->items[i] : *dg;
      wgc::Digest v;
      if (!one.is_string() || !wgc::parse_digest_hex(one.text, v)) return *err = bad_entry, false;
      list.push_back(v);
    }
  }

  // Index: stages in map order, each with its digest and tensors.
  Reader ix{d, 88, 88 + (size_t)isize};
  std::map<std::string, StageTensors> got;
  for (uint64_t s = 0; s < nstages && ix.ok; ++s) {
    const std::string stage = ix.str();
    wgc::Digest stored;
    ix.bytes(stored.data(), 32);
    const uint64_t nt = ix.le(4);
    if (!ix.ok || nt > 64) break;
    StageTensors st;
    wgc::Sha256 h;
    h.update(key.data(), key.size());     // keyed SHA-256 = SHA-256(key || data), spec 08 section 4.1
    for (uint64_t t = 0; t < nt && ix.ok; ++t) {
      const std::string name = ix.str();
      const uint64_t dtype = ix.le(1), rank = ix.le(1);
      Tensor ten;
      size_t n = 1;
      for (uint64_t a = 0; a < rank && ix.ok; ++a) {
        ten.shape.push_back((int)ix.le(4));
        n *= (size_t)ten.shape.back();
      }
      const uint64_t off = ix.le(8), size = ix.le(8);
      if (!ix.ok || dtype > 1 || off % 64 || size != n * (dtype ? 1 : 4) || off > d.size() - 32 ||
          size > d.size() - 32 - off)
        return *err = bad + ": bad tensor entry " + stage + "/" + name, false;
      h.update(&d[off], (size_t)size);
      if (dtype) {
        ten.u.assign(d.begin() + (std::ptrdiff_t)off, d.begin() + (std::ptrdiff_t)(off + size));
      } else {
        ten.f.resize(n);
        std::memcpy(ten.f.data(), &d[off], (size_t)size);  // little-endian hosts only (x86-64)
      }
      st[name] = std::move(ten);
    }
    if (!ix.ok) break;
    const wgc::Digest computed = h.finish();
    if (computed != stored) return *err = bad + ": stage " + stage + " does not match its stored digest", false;
    auto md = map_digest.find(stage);
    if (md == map_digest.end()) {
      // A group this map does not list (a file of a later map with a further optional group): not needed here, so
      // not an error (clean/converter/OUTPUT-FORMAT.md). A file of the shipped map has no such group.
      if (same_map) return *err = bad + ": stage " + stage + " does not match the map's digest", false;
      continue;
    }
    bool in_map = false;
    for (const wgc::Digest& v : md->second) in_map = in_map || v == computed;
    if (!in_map) return *err = same_map ? bad + ": stage " + stage + " does not match the map's digest" : not_this_map, false;
    got[stage] = std::move(st);
  }
  if (!ix.ok) return *err = bad + ": truncated index", false;

  // Every stage and tensor the runtime needs, with the shapes of spec 08 section 7.2.
  for (const std::string& stage : model_stages()) {
    auto it = got.find(stage);
    if (it == got.end()) return *err = same_map ? bad + ": stage " + stage + " missing" : not_this_map, false;
    for (const TensorSpec& ts : stage_specs(stage)) {
      auto t = it->second.find(ts.name);
      const std::string where = stage + "/" + ts.name;
      if (t == it->second.end()) return *err = bad + ": tensor " + where + " missing", false;
      if (t->second.shape != ts.shape || (ts.u8 ? t->second.u.empty() : t->second.f.empty()))
        return *err = bad + ": tensor " + where + " has the wrong shape or type", false;
      if (!check_values(where, t->second, err)) return false;
    }
    (*out)[stage] = std::move(it->second);
  }
  // Optional group filter (spec 08 draft 2 section 7.2): in the index only when the converter found the output-filter
  // weights. Its digest was checked above like every stage's, against the map's list of variants.
  auto fit = got.find(kFilterStage);
  if (fit != got.end()) {
    const TensorSpec& ts = filter_spec();
    const std::string where = std::string(kFilterStage) + "/" + ts.name;
    auto t = fit->second.find(ts.name);
    if (t == fit->second.end()) return *err = bad + ": tensor " + where + " missing", false;
    if (fit->second.size() != 1 || t->second.shape != ts.shape || t->second.f.empty())
      return *err = bad + ": tensor " + where + " has the wrong shape or type", false;
    if (!check_values(where, t->second, err)) return false;
    (*out)[kFilterStage] = std::move(fit->second);
  }
  if (other_map) *other_map = !same_map;
  return true;
}

bool pack_stage_blob(const std::string& stage, const StageTensors& t, const DRow& drow, std::vector<uint8_t>* out,
                     std::string* err) {
  out->clear();
  if (stage == "enc0") {
    pack_enc0(t, drow, out);
    if (out->size() != 10752) return *err = "enc0 blob size mismatch", false;
  } else if (stage == "dec0") {
    pack_dec0(t, drow, out);
    if (out->size() != 10752) return *err = "dec0 blob size mismatch", false;
  } else if (swin_stage_info(stage)) {
    if (!pack_swin(stage, t, drow, out)) return *err = stage + " block size mismatch", false;
  } else {
    *err = "no packer for stage " + stage;
    return false;
  }
  return true;
}

}  // namespace wgrt
