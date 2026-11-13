// SPDX-License-Identifier: MIT
// Model K in the per-frame runtime; see runtime_k.h. Port of clean/k/runner/k_frame.py (KFrame), k_net.py (KNet,
// variant "fused"), k_launch.py (stage_fused_launch) and k_io.py (frame_in, out_uniforms, input / output launches).
#include "runtime_k.h"
#include "k_input_uniforms.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>

#include "../../runtime/embedded.h"
#include "../../runtime/fpconv.h"
#include "../../runtime/runtime_core.h"
#include "k_filter_table.h"
#include "k_pack.h"
#include "model_k.h"
#include "model_k_load.h"

// The host-side FP32 values of k_io.py out_uniforms: each operation rounded on its own (no contraction).
#if defined(_MSC_VER) && !defined(__clang__)
#pragma fp_contract(off)
#else
#pragma STDC FP_CONTRACT OFF
#endif

namespace wgrt {

namespace {

// Kernel parameter structs, passed by value; must match clean/k/kernels/k_io.hip, k_block.hip and k_block_wide.hip
// (static_asserts there and in clean/k/runner/k_io.py, k_launch.py).
struct KFrameIn {
  DevPtr color, motion, depth, xs;
  int cpitch, mpitch, zpitch, cbx, cby, mbx, mby, zbx, zby, W, H, rw, rh, Bw, Bh, flags;
  float jx, jy, sx, sy, mox, moy, kappa, rho, lam, s;
  int mch;
  float rmsx, rmsy, rmox, rmoy;
  int pad_;
};
static_assert(sizeof(KFrameIn) == 160, "KFrameIn");
struct KOutUni {
  float invW, invH, rhox, rhoy, rho2x, rho2y, jmx, jmy, spx, spy, invW16, invH16, kinv;
  uint16_t kux, kvy;
};
static_assert(sizeof(KOutUni) == 56, "KOutUni");
struct KOutParams {
  KFrameIn f;
  DevPtr N, L, Hc, table, exp2tab, out, Lout, Hcout, Hhout;
  int opitch, omode;
  DevPtr ctap, ctapT;
  KOutUni u;
};
static_assert(sizeof(KOutParams) == 312, "KOutParams");
struct KInParams {
  KFrameIn f;
  DevPtr Hc, Hh, x0, cpre, cpre2;
};
static_assert(sizeof(KInParams) == 200, "KInParams");
struct KInColorParams { KInParams input; DevPtr ctap, ctapT; };
static_assert(sizeof(KInColorParams) == 216, "KInColorParams");
struct KInUniformParams { KInParams input; KiHost host; };
struct KInColorUniformParams { KInColorParams input; KiHost host; };
static_assert(sizeof(KInUniformParams) == 248 && sizeof(KInColorUniformParams) == 264, "input host arguments");
static_assert(offsetof(KInUniformParams, host) == 200 && offsetof(KInColorUniformParams, host) == 216, "input host offsets");
struct KExpoArgs {
  float value;
  int pair, use_src, first, hdr, reset;
  float e1, e2, inv_pi;
};
static_assert(sizeof(KExpoArgs) == 36, "KExpoArgs");
struct KBlkParams {
  DevPtr x, skip, y, merged, head, g1, g2, wqkv, mix, wo, bo, w1, b1, w2, b2, wa, ba, wz, bz;
  int w, h, ox, oy, nwx, nwy, cw, ch;
};
static_assert(sizeof(KBlkParams) == 184, "KBlkParams");
struct ExpParams {  // clean/kernels/exposure.hip WgExpParams
  int rw, rh, nx, ny, nparts, pad0;
  float inv_pi, K, inv_n;
};
static_assert(sizeof(ExpParams) == 36, "ExpParams");

enum { KIO_HDR = 1, KIO_MVLO = 2, KIO_INV = 4, KIO_RESET = 8, KIO_C16 = 16, KIO_M16 = 32 };
enum { OUT_RGBA32F = 0, OUT_RGBA16F = 1, OUT_RGB32F = 2 };
constexpr int kThreads = 128;  // k_io.py THREADS, k_launch.py THREADS
constexpr float kS = 0.5f;     // spec 44 draft 3 base-size factor (clean/k/reference/layout.py DEFAULT_S)
constexpr size_t kTableBytes = (size_t)kFilterEntries * kFilterValues * 2;

int cdiv(long long a, long long b) { return (int)((a + b - 1) / b); }
int ceil16(int n) { return (n + 15) / 16 * 16; }
size_t al256(size_t n) { return (n + 255) / 256 * 256; }
size_t pad16(size_t n) { return (n + 15) / 16 * 16; }

// k_pack.py linear_size / linear_sections / block_sections / stage_sections (offsets inside a stage blob)
size_t linear_size(size_t n, size_t k) { return al256(pad16(n) * k * 2) + al256(pad16(n) * 2); }
size_t linear_bias(size_t n, size_t k) { return al256(pad16(n) * k * 2); }
struct BlockSec {
  size_t g1, g2, qkv, mix, wo, bo, w1, b1, w2, b2, size;
};
BlockSec block_sections(const KStageInfo& s) {
  const size_t C = s.C, D = s.D(), H = s.heads, HID = s.hidden();
  const size_t sizes[10] = {C * 2, C * 2, (s.fixed ? 1 : 3) * H * 32 * C * 2, H * 64 * 64 * 2, C * D * 2, C * 2,
                            HID * C * 2, HID * 2, C * HID * 2, C * 2};
  size_t off[10], o = 0;
  for (int i = 0; i < 10; ++i) {
    off[i] = o;
    o += al256(sizes[i]);
  }
  return {off[0], off[1], off[2], off[3], off[4], off[5], off[6], off[7], off[8], off[9], o};
}
struct StageSec {
  size_t embed = 0, expand = 0, block = 0, merge = 0, head = 0;
};
StageSec stage_sections(const KStageInfo& s) {  // order: embed, expand, block, merge, head
  StageSec r;
  size_t o = 0;
  if (s.embed) r.embed = o, o += linear_size(32, 16);
  if (s.kind == 'd') r.expand = o, o += linear_size(4 * s.C, s.C_prev);
  r.block = o;
  o += block_sections(s).size;
  if (s.kind == 'e') r.merge = o, o += linear_size(s.C_next, 4 * s.C);
  if (s.head) r.head = o;
  return r;
}

// ---- network=fused8: FP8 forms of the level-0/1 stages (clean/k/kernels/k_block8.h) --------------------------------
static_assert(sizeof(KF8Params) == 104, "KF8Params of k_block8.h");
float blob_h(const std::vector<uint8_t>& b, size_t off, size_t i) {
  uint16_t v;
  std::memcpy(&v, &b[off + 2 * i], 2);
  return f16_to_f32(v);
}
// largest k with m 2^k <= 448 (m > 0, finite)
int fit448(double m) { return (int)std::floor(std::log2(448.0 / m)); }
// The FP8 data of stage s (e0, e1, d1 or d0) from its FP16 blob. Scales (k_block8.h "Scales"): weights by the largest
// power of two that keeps them <= 448; n unscaled (|n_c| <= |g1_c|); v by the bound |v_j| <= sum_c |Wv_jc| |g1_c|;
// z by its bound (fixed mixing: max_t sum_s |M_ts| times v's; softmax: v's); a unscaled and clamped at 448; h 2^1.
F8Stage fp8_stage(const KStageInfo& s, const std::vector<uint8_t>& blob) {
  F8Stage r;
  const StageSec sec = stage_sections(s);
  const BlockSec bs = block_sections(s);
  const size_t blk = sec.block, C = s.C, H = s.heads, KT = C / 16, QKV = s.fixed ? 1 : 3, HID = s.hidden(),
               D = s.D();
  auto maxabs = [&](size_t off, size_t n) {
    double m = 0;
    for (size_t i = 0; i < n; ++i) m = std::max(m, (double)std::fabs(blob_h(blob, off, i)));
    return m;
  };
  std::vector<double> g1(C);
  double g1max = 0;
  for (size_t c = 0; c < C; ++c) g1max = std::max(g1max, g1[c] = std::fabs(blob_h(blob, blk + bs.g1, c)));
  // v: tile (head hd, ct, kt) of the qkv section, B layout: lane l = column j % 16 + 16 hh, element e = channel
  // 16 kt + 8 hh + e
  double wvmax = 0, bv = 0;
  for (size_t hd = 0; hd < H; ++hd)
    for (size_t j = 0; j < 32; ++j) {
      double row = 0;
      const size_t ct = j / 16;
      for (size_t c = 0; c < C; ++c) {
        const size_t kt = c / 16, hh = (c % 16) / 8, e = c % 8;
        const size_t tile = hd * QKV * 2 * KT + (QKV - 1) * 2 * KT + ct * KT + kt;
        const double w = std::fabs(blob_h(blob, blk + bs.qkv, tile * 256 + (j % 16 + 16 * hh) * 8 + e));
        wvmax = std::max(wvmax, w);
        row += w * g1[c];
      }
      bv = std::max(bv, row);
    }
  bv *= 1.01;  // |x r| <= 1 up to the FP16 roundings of s and r
  double rm = 0, mmax = 0;
  if (s.fixed)
    for (size_t t = 0; t < H * 64; ++t) {
      double row = 0;
      for (size_t k = 0; k < 64; ++k) {
        const double m = std::fabs(blob_h(blob, blk + bs.mix, t * 64 + k));
        row += m;
        mmax = std::max(mmax, m);
      }
      rm = std::max(rm, row);
    }
  const double wo = maxabs(blk + bs.wo, C * D), w1 = maxabs(blk + bs.w1, HID * C), w2 = maxabs(blk + bs.w2, C * HID);
  for (double v : {g1max, wvmax, bv, wo, w1, w2})
    if (!std::isfinite(v) || v <= 0) return r.why = "weights not finite or all zero", r;
  if (s.fixed && (!std::isfinite(rm) || rm <= 0)) return r.why = "mixing matrix not finite or zero", r;
  if (s.fixed && 1.01 * g1max >= 448.0) return r.why = "|g1| >= 448", r;
  const int kv = s.fixed ? fit448(std::max(bv, wvmax)) : 0, km = s.fixed ? fit448(mmax) : 0;
  const int kpv = fit448(bv), kz = fit448(s.fixed ? rm * bv : bv), ko = fit448(wo), k1 = fit448(w1),
            k2 = fit448(w2), kh = 1;
  const int m = kh - k1;
  r.q.pv = std::ldexp(1.0f, kpv);
  r.q.sz = s.fixed ? std::ldexp(1.0f, kz - kv - km) : std::ldexp(1.0f, kz - kpv - 8);
  r.q.sy = std::ldexp(1.0f, -(ko + kz));
  r.q.L = std::ldexp(1.0f, k1 + 1);
  r.q.Hs = std::ldexp(440.0f, -m);
  r.q.A2 = std::ldexp(0.412109375f, m - k1);
  r.q.B2 = std::ldexp(0.0810546875f, m - 2 * k1);
  r.q.c0 = std::ldexp(1.0f, m - 1);
  r.q.sx = std::ldexp(1.0f, -(k2 + kh));
  // the buffer
  std::vector<uint8_t>& b = r.bytes;
  auto part = [&](size_t n) {
    const size_t o = b.size();
    b.resize(al256(o + n), 0);
    return (DevPtr)o;
  };
  auto f8 = [&](DevPtr dst, size_t src, size_t n, int k) {
    for (size_t i = 0; i < n; ++i) b[dst + i] = fp8_encode(std::ldexp((double)blob_h(blob, src, i), k));
  };
  auto f32 = [&](DevPtr dst, size_t src, size_t n, int k) {
    for (size_t i = 0; i < n; ++i) {
      const float v = std::ldexp(blob_h(blob, src, i), k);
      std::memcpy(&b[dst + 4 * i], &v, 4);
    }
  };
  const size_t nq = QKV * H * 32 * C;
  r.q.wqkv = part(nq);
  r.q.mix = part(s.fixed ? H * 64 * 64 : 0);
  r.q.wo = part(C * D);
  r.q.w1 = part(HID * C);
  r.q.w2 = part(C * HID);
  r.q.bo = part(4 * C);
  r.q.b1 = part(4 * HID);
  r.q.b2 = part(4 * C);
  f8(r.q.wqkv, blk + bs.qkv, nq, kv);
  if (s.fixed) f8(r.q.mix, blk + bs.mix, H * 64 * 64, km);
  f8(r.q.wo, blk + bs.wo, C * D, ko);
  f8(r.q.w1, blk + bs.w1, HID * C, k1);
  f8(r.q.w2, blk + bs.w2, C * HID, k2);
  f32(r.q.bo, blk + bs.bo, C, ko + kz);
  f32(r.q.b1, blk + bs.b1, HID, k1);
  f32(r.q.b2, blk + bs.b2, C, k2 + kh);
  return r;
}
bool fp8_stage_name(const char* n) {
  return !std::strcmp(n, "e0") || !std::strcmp(n, "e1") || !std::strcmp(n, "d1") || !std::strcmp(n, "d0");
}

// layout.py FUSED_IO (the fused network: merged maps and decoder outputs alternate between A and B)
struct FusedIo {
  const char *in, *skip, *block, *merged, *head;
};
const FusedIo kFusedIo[11] = {
    {"X0", nullptr, "S0", "A", nullptr}, {"A", nullptr, "S1", "B", nullptr}, {"B", nullptr, "S2", "A", nullptr},
    {"A", nullptr, "S3", "B", nullptr},  {"B", nullptr, "S4", "A", nullptr}, {"A", nullptr, "B", nullptr, nullptr},
    {"B", "S4", "A", nullptr, nullptr},  {"A", "S3", "B", nullptr, nullptr}, {"B", "S2", "A", nullptr, nullptr},
    {"A", "S1", "B", nullptr, nullptr},  {"B", "S0", nullptr, nullptr, "OUT"},
};
// k_launch.py WIDE_DEFAULT (k_net.py DEFAULT_PATHS of variant "fused"): k_stagew<NG>_<name>; 0 = k_stage_<name>.
// f16 (variant "fused16", WIDE_DEFAULT16): k_stagew4h_<name> for e4, m, d4; 0 = k_stageh_<name>.
// FP32: e4 and d4 with NG = 4 when they have at most 24 windows (k_launch.py WIDE4_SMALL_MAX_WINDOWS; 1440p output: 24,
// faster with 4; 4K output: 40, faster with 2).
int wide_ng(const std::string& s, bool f16, int nwin) {
  if (f16) return s == "e4" || s == "m" || s == "d4" ? 4 : 0;
  if (s == "m") return 4;
  if (s == "e4" || s == "d4") return nwin <= 24 ? 4 : 2;
  if (s == "e3" || s == "d3") return 2;
  return 0;
}

// Runtime options of model K (runtime_options, then WG_RUNTIME_OPTIONS whose keys win): model_k_dir, model_k_map,
// network (fused, the default; fused16: FP16 accumulation, the spec 49 speed approximation, opt-in), k_diag (timing
// diagnostics, same output: "input2" runs the input stage once more before its timing mark, so the "input" part is a
// second run and the first is counted in "exposure"; "copyin" copies the interop's colour, depth and motion into the
// own buffers first (hipMemcpyAsync, before the first mark); "input2+copyin" both), k_head (compact, the default: d0 writes
// the compact pixel head for k_out_pixel_c; full: the 40-channel head map and k_out_pixel, same output bits, and the
// buffer "k.N" for diagnostics), k_fast (0, the default: the exact stage kernels; 1: opt-in, not bit-exact, the "_f"
// variants of the input and output stage kernels, see k_io.hip KIO_FAST).
std::string option(const WgRtConfig& c, const std::string& key) {
  const char* env = std::getenv("WG_RUNTIME_OPTIONS");
  const std::string text = std::string(c.runtime_options ? c.runtime_options : "") + ";" + (env ? env : "");
  std::string v;
  for (size_t a = 0; a <= text.size();) {
    size_t b = text.find(';', a);
    if (b == std::string::npos) b = text.size();
    const std::string kv = text.substr(a, b - a);
    a = b + 1;
    if (kv.compare(0, key.size() + 1, key + "=") == 0) v = kv.substr(key.size() + 1);
  }
  return v;
}

std::string join(const std::string& dir, const std::string& name) {
  if (dir.empty()) return name;
  const char last = dir.back();
  return last == '/' || last == '\\' ? dir + name : dir + "/" + name;
}

}  // namespace

const char* const KRuntime::kTimingLabels[4] = {"exposure", "input", "network", "output"};

const char* KRuntime::timing_info() {
  return "model K: HIP events per frame, in stream order: [0] start of the first kernel after the input wait "
         "(exposure_auto_diet, or exposure_auto_diet_c16 with input_copy, if auto-exposure, else k_expo_scalars) -> [1] start "
         "of input/color preparation -> [2] start of the first neural kernel -> [3] start of k_out_pixel (k_out_pixel_c with the compact pixel head) -> [4] end of "
         "an empty kernel after the final output-stage kernel; exposure = [0]-[1] (input conversion, exposure, k_expo_scalars), input "
         "= [1]-[2] (k_in_tiled_color[_uniform], or separate k_out_color followed by k_in_tiled[_uniform] or k_in_color plus k_in_base/k_in_reset), network = [2]-[3] (eleven stage kernels, including hidden-history output in d0), output = [3]-[4] ("
         "k_out_pixel writing the game output)";
}

bool KRuntime::alloc(DevPtr* p, size_t bytes, bool zero) {
  bytes = std::max<size_t>(bytes, 256);
  if (!alloc_(p, bytes)) return fail("model K: device memory (" + std::to_string(bytes >> 20) + " MiB): " + dev_->err());
  if (zero && !dev_->memset(*p, 0, bytes)) return dfail();
  return true;
}

bool KRuntime::create(const WgRtConfig& c, Device* dev, AllocFn alloc_fn, bool external_io) {
  cfg_ = c;
  dev_ = dev;
  alloc_ = std::move(alloc_fn);
  external_io_ = external_io;
  build_dir_ = c.build_dir ? c.build_dir : ".";
  W_ = c.ow;
  H_ = c.oh;
  max_rw_ = c.max_rw;
  max_rh_ = c.max_rh;
  hdr_ = c.hdr != 0;
  depth_inverted_ = c.depth_inverted != 0;
  if (W_ <= 0 || H_ <= 0 || max_rw_ <= 0 || max_rh_ <= 0) return fail("model K: invalid render or output size");
  // 32-bit byte offsets in k_io.hip (kio_ptr): every image below 4 GB
  if (16LL * W_ * H_ >= (1LL << 32) || 16LL * max_rw_ * max_rh_ >= (1LL << 32))
    return fail("model K: render or output size too large");
  W16_ = ceil16(W_);
  H16_ = ceil16(H_);
  k_base_size(W_, H_, kS, &Bw_, &Bh_);

  // model data (spec 48 / spec 10 section 7): load, check against the map, pack once
  if (!c.model_dir && option(c, "model_k_dir").empty()) return fail("model K: no model directory");
  std::string kdir = option(c, "model_k_dir");
  if (kdir.empty()) kdir = join(c.model_dir, "K");
  model_path_ = join(kdir, "model.wgcm");
  const std::string net = option(c, "network");
  if (!net.empty() && net != "fused" && net != "fused16" && net != "fused8")
    return fail("model K: unknown network variant " + net + " (fused, fused16 or fused8)");
  // fused8: fused16 with the FP8 forms of e0, e1, d1, d0 (k_block8.h; opt-in speed approximation, not bit-exact)
  f16_ = net == "fused16" || net == "fused8";
  f8_ = net == "fused8";
  // fp8_stages: which of e0, e1, d1, d0 run their FP8 form with network=fused8 (comma list; default all four)
  std::string f8list = option(c, "fp8_stages");
  if (f8list.empty()) f8list = "e0,e1,d1,d0";
  for (size_t a = 0; a <= f8list.size();) {
    size_t b = f8list.find(',', a);
    if (b == std::string::npos) b = f8list.size();
    const std::string n = f8list.substr(a, b - a);
    a = b + 1;
    if (n.empty() || n == "none") continue;
    if (!fp8_stage_name(n.c_str())) return fail("model K: fp8_stages: " + n + " is not one of e0, e1, d1, d0");
    f8stages_.push_back(n);
  }
  const std::string tiled = option(c, "input_tiled");
  if (!tiled.empty() && tiled != "0" && tiled != "1") return fail("input_tiled must be 0 or 1");
  tiled_input_ = tiled != "0";
  const std::string diag = option(c, "k_diag");
  if (!diag.empty() && diag != "input2" && diag != "copyin" && diag != "input2+copyin" && diag != "copyin+input2")
    return fail("model K: unknown k_diag " + diag + " (input2, copyin, input2+copyin)");
  diag_input2_ = diag.find("input2") != std::string::npos;
  diag_copyin_ = diag.find("copyin") != std::string::npos;
  const std::string khead = option(c, "k_head");
  if (!khead.empty() && khead != "compact" && khead != "full")
    return fail("model K: unknown k_head " + khead + " (compact or full)");
  compact_head_ = khead != "full";
  const std::string kfast = option(c, "k_fast");
  if (!kfast.empty() && kfast != "0" && kfast != "1") return fail("model K: k_fast must be 0 or 1");
  fast_ = kfast == "1";
  map_path_ = option(c, "model_k_map");
  if (map_path_.empty()) {
    map_path_ = join(build_dir_, "preset-K.v1.map.json");  // next to the library, else built into it
    if (!wg_file_exists(map_path_) && wg_file_exists(join(kdir, "preset-K.v1.map.json")))
      map_path_ = join(kdir, "preset-K.v1.map.json");
  }
  std::vector<char> mchars;
  if (!wg_read_file(map_path_, &mchars)) return fail("model K: the converter map " + map_path_ + " is missing");
  const std::vector<uint8_t> mbytes(mchars.begin(), mchars.end());
  KModel model;
  std::string e;
  if (!wg_file_exists(model_path_))
    return fail("model K: no converted model at " + model_path_ + " (convert the preset-K model there)");
  if (!load_k_model_map(model_path_, mbytes, map_path_, &model, &e)) return fail("model K: " + e);
  std::memcpy(offsets_table_, model.window_offsets, sizeof offsets_table_);
  for (const KStageInfo& s : k_stage_table()) {
    std::vector<uint8_t> b = k_pack_stage(s.name, model.stages[s.name], &e);
    if (b.empty()) return fail("model K: " + e);
    if (b.size() != k_stage_blob_bytes(s)) return fail(std::string("model K: blob size of stage ") + s.name);
    if (f8_ && std::find(f8stages_.begin(), f8stages_.end(), s.name) != f8stages_.end()) {
      F8Stage f8 = fp8_stage(s, b);
      if (f8.why.empty()) f8data_[s.name] = std::move(f8);
      else f8why_ += std::string(" ") + s.name + ":" + f8.why;
    }
    blobs_[s.name] = std::move(b);
  }

  // code objects: the stage kernels (k_block.co: k_stage_*, k_block_wide.co: k_stagew*), the I/O stages and the
  // exposure scalars (k_io.co), model M's exposure (spec 07) and frame helpers (timing_barrier)
  for (const char* co : {"k_block.co", "k_block_wide.co", "k_io.co", "exposure.co", "frame.co"})
    if (!dev_->load_module(join(build_dir_, co))) return fail("model K: " + dev_->err());

  // buffers
  for (auto& kv : blobs_) {
    DevPtr p;
    if (!alloc(&p, kv.second.size()) || !dev_->upload(p, kv.second.data(), kv.second.size())) return false;
    wdev_[kv.first] = p;
  }
  for (auto& kv : f8data_) {  // network=fused8: the FP8 weights and FP32 biases; device pointers into KF8Params
    DevPtr p;
    if (!alloc(&p, kv.second.bytes.size()) || !dev_->upload(p, kv.second.bytes.data(), kv.second.bytes.size()))
      return false;
    KF8Params& q = kv.second.q;
    for (DevPtr* a : {&q.wqkv, &q.mix, &q.wo, &q.w1, &q.w2, &q.bo, &q.b1, &q.b2}) *a += p;
    f8dev_[kv.first] = p;
  }
  for (const KBuffer& b : k_buffer_plan(Bw_, Bh_)) {
    DevPtr p;
    if (!alloc(&p, b.bytes)) return false;
    net_[b.name] = p;
    net_bytes_[b.name] = b.bytes;
  }
  const size_t R = (size_t)max_rw_ * max_rh_, P16 = (size_t)W16_ * H16_;
  for (int p = 0; p < 2; ++p)  // histories, zero at create (KFrame)
    if (!alloc(&Hc_[p], P16 * 4 * 2, true) || !alloc(&L_[p], 4 * P16 * 2, true) ||
        !alloc(&Hh_[p], (size_t)Bw_ * Bh_ * 4 * 2, true))
      return false;
  const int ny = std::max(256, max_rh_ / 2);
  if (!alloc(&color_, R * 16) || !alloc(&depth_, R * 4) || !alloc(&motion_, R * 8) || !alloc(&cpre_, R * 16) ||
      !alloc(&cpre2_, R * 8) || !alloc(&ctap_, R * 8) || !alloc(&ctapT_, R * 2) ||
      // exp2tab: read only by the host emulator's k_out_pixel (the GPU uses v_exp_f16); a valid, zeroed buffer
      !alloc(&exp2tab_, 65536 * 2, true) || !alloc(&tables_, 8 * kTableBytes) ||
      !alloc(&expo_, 32, true) || !alloc(&expo_partials_, 4 * (size_t)cdiv(ny, 4)) || !alloc(&expo_computed_, 4, true) ||
      !alloc(&expo_sync_, 4, true) || !alloc(&xs_, 16, true))
    return false;
  if (!external_io_ && !alloc(&out_, (size_t)W_ * H_ * 12, true)) return false;
  // the filter-shape table (spec 43 section 7) of every level, computed once; a frame reads its level's set
  std::vector<uint16_t> tab((size_t)8 * kFilterEntries * kFilterValues);
  for (int l = 0; l < 8; ++l) k_filter_table(l, 1, tab.data() + (size_t)l * kFilterEntries * kFilterValues);
  if (!dev_->upload(tables_, tab.data(), tab.size() * 2)) return dfail();
  return dev_->sync() || dfail();
}

bool KRuntime::launch(const char* kernel, unsigned grid, unsigned block, void* a0, void* a1, void* a2, void* a3,
                      void* a4, void* a5, void* a6, void* a7) {
  if (grid == 0) return true;
  void* args[8] = {a0, a1, a2, a3, a4, a5, a6, a7};
  if (pending_mark_ >= 0) {
    const int e = pending_mark_;
    pending_mark_ = -1;
    return dev_->launch_timed(kernel, grid, block, args, e, -1) || dfail();
  }
  return dev_->launch(kernel, grid, block, args) || dfail();
}

bool KRuntime::upload_inputs(int rw, int rh, const float* rgb, const float* dep, const float* mot) {
  if (rw <= 0 || rh <= 0 || rw > max_rw_ || rh > max_rh_) return fail("render size larger than the maximum");
  if (external_io_) return fail("inputs come from the interop (external_io)");
  if (!dev_->sync()) return dfail();
  const size_t n = (size_t)rw * rh;
  std::vector<float> c(n * 4, 0.0f);
  for (size_t i = 0; i < n; ++i)
    for (int k = 0; k < 3; ++k) c[4 * i + k] = rgb[3 * i + k];
  return (dev_->upload(color_, c.data(), n * 16) && dev_->upload(depth_, dep, n * 4) &&
          dev_->upload(motion_, mot, n * 8) && dev_->sync()) ||
         dfail();
}

// The eleven stages for frame index f (k_net.py KNet.stage_plan, variant "fused" or "fused16"; k_launch.py
// stage_fused_launch).
bool KRuntime::network(int f) {
  int off[11][2];
  k_stage_offsets(f, offsets_table_, off);
  const auto& table = k_stage_table();
  for (size_t i = 0; i < table.size(); ++i) {
    const KStageInfo& st = table[i];
    const FusedIo& io = kFusedIo[i];
    int lw, lh, iw, ih;
    k_level_size(Bw_, Bh_, st.level, &lw, &lh);
    k_level_size(Bw_, Bh_, st.kind == 'd' ? st.level + 1 : st.level, &iw, &ih);
    const int ox = off[i][0], oy = off[i][1];
    if (st.kind != 'm' && (ox % 2 || oy % 2)) return fail(std::string("model K: odd window offset at ") + st.name);
    const DevPtr blob = wdev_[st.name];
    const StageSec sec = stage_sections(st);
    const BlockSec bs = block_sections(st);
    const DevPtr blk = blob + sec.block;
    KBlkParams p = {};
    p.x = net_[io.in];
    p.y = io.block ? net_[io.block] : 0;
    p.g1 = blk + bs.g1;
    p.g2 = blk + bs.g2;
    p.wqkv = blk + bs.qkv;
    p.mix = blk + bs.mix;
    p.wo = blk + bs.wo;
    p.bo = blk + bs.bo;
    p.w1 = blk + bs.w1;
    p.b1 = blk + bs.b1;
    p.w2 = blk + bs.w2;
    p.b2 = blk + bs.b2;
    p.w = lw;
    p.h = lh;
    p.ox = ox;
    p.oy = oy;
    p.nwx = cdiv(lw + ox, 8);  // layout.py window_grid
    p.nwy = cdiv(lh + oy, 8);
    if (st.embed) {
      p.wa = blob + sec.embed;
      p.ba = blob + sec.embed + linear_bias(32, 16);
    }
    if (st.kind == 'd') {
      p.wa = blob + sec.expand;
      p.ba = blob + sec.expand + linear_bias(4 * st.C, st.C_prev);
      p.skip = net_[io.skip];
      p.cw = iw;
      p.ch = ih;
    }
    if (st.kind == 'e') {
      p.wz = blob + sec.merge;
      p.bz = blob + sec.merge + linear_bias(st.C_next, 4 * st.C);
      p.merged = net_[io.merged];
    }
    if (st.head) {
      p.wz = blob + sec.head;
      p.bz = blob + sec.head + linear_bias(40, st.C);
      p.head = net_[io.head];
    }
    const int ng = wide_ng(st.name, f16_, p.nwx * p.nwy);
    if (st.kind == 'm' && !ng) return fail("model K: stage m needs its wide kernel");
    const std::string h = f16_ ? "h" : "";
    // e0, e1, d1, d0: two windows per workgroup (k_stage2_*, k_launch.py WINDOWS2_STAGES; k_net.py WINDOWS2_VARIANTS;
    // d1's needs its block output map p.y, which the network always gives it)
    const bool w2 = !ng && (std::strcmp(st.name, "e0") == 0 || std::strcmp(st.name, "e1") == 0 ||
                            (std::strcmp(st.name, "d1") == 0 && p.y) || std::strcmp(st.name, "d0") == 0);
    const std::string k =
        (ng ? "k_stagew" + std::to_string(ng) : std::string(w2 ? "k_stage2" : "k_stage")) + h + "_" + st.name;
    const unsigned grid = (unsigned)(w2 ? cdiv(p.nwx * p.nwy, 2) : p.nwx * p.nwy);
    auto f8 = f8data_.find(st.name);
    if (f8_ && w2 && f8 != f8data_.end()) {  // network=fused8: k_stage2q_* (k_block8.h)
      KF8Params q = f8->second.q;
      if (st.head) {
        DevPtr hidden = Hh_[1 - parity_];
        if (!launch(compact_head_ ? "k_stage2q_d0_hp" : "k_stage2q_d0_hidden", grid, kThreads, &p, &q, &hidden))
          return false;
      } else if (!launch(("k_stage2q_" + std::string(st.name)).c_str(), grid, kThreads, &p, &q))
        return false;
    } else if (st.head) {  // d0 always uses the G2 stage above; its epilogue writes the hidden history (FP32 and FP16 forms)
      DevPtr hidden = Hh_[1 - parity_];
      // k_head=compact: p.head (the OUT buffer) receives the compact pixel head [2 Bh][2 Bw][4] instead of the head map
      const char* k0 = compact_head_ ? (f16_ ? "k_stage2h_d0_hp" : "k_stage2_d0_hp")
                                     : (f16_ ? "k_stage2h_d0_hidden" : "k_stage2_d0_hidden");
      if (!launch(k0, grid, kThreads, &p, &hidden)) return false;
    } else if (!launch(k.c_str(), grid, (unsigned)(kThreads * (ng ? ng : 1)), &p)) return false;
    if (i < 10) {
      info_.offsets[i][0] = ox;  // WgRtFrameInfo has ten rows: e0 ... d1 (d0's offset is (4, 4))
      info_.offsets[i][1] = oy;
    }
  }
  return true;
}

bool KRuntime::frame(const WgRtFrame& fr, const FrameIo* io) {
  // reset = 2 (interop: a new context of the game took this upscaler over, wronggpu_upscaler.h): this frame is the
  // first frame of a newly created K runtime, as for model M. The host state a frame leaves for the next one starts
  // over (frame counter, so the exposure's first frame too; frame index f; history parity; last size; jitter). The
  // device histories keep their contents: a reset frame reads none of them (validity false, k_in_reset, E_prev =
  // E_cur), so the outputs equal a new runtime's (test_k_runtime_gpu.py test_reset2_as_new).
  if (fr.reset >= 2) {
    frames_ = fidx_ = parity_ = last_rw_ = last_rh_ = 0;
    jprev_[0] = jprev_[1] = 0.0f;
  }
  const int rw = fr.rw, rh = fr.rh;
  if (rw <= 0 || rh <= 0 || rw > max_rw_ || rh > max_rh_) return fail("render size larger than the maximum");
  const bool reset = fr.reset || frames_ == 0 || rw != last_rw_ || rh != last_rh_;
  const int f = reset ? 0 : fidx_;
  const int p = parity_, q = 1 - parity_;
  const int tb = io ? io->timing_base : -1;
  int zero = 0;
  auto mark = [&](int k) {
    if (tb < 0) return true;
    if (k < 4) return pending_mark_ = tb + k, true;
    pending_mark_ = -1;
    void* args[1] = {&zero};
    return dev_->launch_timed("timing_barrier", 1, 32, args, -1, tb + 4) || dfail();
  };
  info_ = {};
  info_.index = frames_;
  info_.n = f;  // model K: the frame index f of spec 44 (0 on a reset frame)
  info_.parity = p;
  info_.reset = reset ? 1 : 0;
  info_.rw = rw;
  info_.rh = rh;
  info_.jitter_prev[0] = reset ? fr.jitter[0] : jprev_[0];
  info_.jitter_prev[1] = reset ? fr.jitter[1] : jprev_[1];
  if (!mark(0)) return false;

  // inputs: the caller's dense buffers (FrameIo), the interop's raw copies converted here, or the own test buffers
  DevPtr cin = io && io->color ? io->color : color_, din = io && io->depth ? io->depth : depth_;
  DevPtr min_ = io && io->motion ? io->motion : motion_;
  // input_copy (io->raw_inputs): the interop's copies of the game's textures in pitched rows, read directly (KIO_C16:
  // RGBA16F colour, not divided by pi: K applies kappa; R32F depth; KIO_M16: FP16 motion of motion_channels channels
  // with m = ms r + mo applied on load, as input_convert would have written it)
  const bool raw = io && io->raw_inputs;
  int cpitch = rw, mpitch = rw, zpitch = rw, mch = 2;
  if (raw) {
    mch = io->motion_channels;
    if ((mch != 2 && mch != 4) || io->color_pitch % 8 || io->depth_pitch % 4 || io->motion_pitch % (2 * mch))
      return fail("model K: unexpected input-copy layout");
    cpitch = io->color_pitch / 8;
    zpitch = io->depth_pitch / 4;
    mpitch = io->motion_pitch / (2 * mch);
  }
  if (!cin || !din || !min_) return fail("frame without input buffers (interop)");
  if (diag_copyin_ && !raw && cin != color_) {  // k_diag=copyin: the inputs from K's own allocations (dense, as FrameIo)
    const size_t n = (size_t)rw * rh;
    if (!dev_->copy(color_, cin, n * 16) || !dev_->copy(depth_, din, n * 4) || !dev_->copy(motion_, min_, n * 8))
      return dfail();
    cin = color_;
    din = depth_;
    min_ = motion_;
  }

  // exposure (spec 10 section 5.1, spec 07) and the scalars of spec 42 section 4 on the GPU. Pre-exposure pi: 1 if
  // missing, <= 0 or not finite (spec 10); model K does not divide the colour by pi, kappa = (1 / pi) e1 E does.
  const float pi = (std::isfinite(fr.pre_exposure) && fr.pre_exposure > 0.0f) ? fr.pre_exposure : 1.0f;
  const int mode = io && io->exposure_mode >= 0 ? io->exposure_mode : (cfg_.auto_exposure ? 2 : 0);
  int first = frames_ == 0 ? 1 : 0;
  if (mode == 2) {
    ExpParams ep = {};
    ep.rw = rw;
    ep.rh = rh;
    ep.nx = std::max(256, rw / 2);
    ep.ny = std::max(256, rh / 2);
    ep.nparts = cdiv(ep.ny, 4);
    ep.inv_pi = 1.0f / pi;
    const double a = cfg_.alpha > 0 ? cfg_.alpha : 0.18;
    ep.K = (float)(a / (1.0 - a));
    ep.inv_n = (float)(1.0 / ((double)ep.nx * ep.ny));
    if (raw ? !launch("exposure_auto_diet_c16", (unsigned)ep.nparts, 256, &cin, &cpitch, &expo_partials_, &expo_sync_,
                      &expo_computed_, &expo_, &ep, &first)
            : !launch("exposure_auto_diet", (unsigned)ep.nparts, 256, &cin, &expo_partials_, &expo_sync_,
                      &expo_computed_, &expo_, &ep, &first))
      return false;
  }
  KExpoArgs xa = {};
  xa.value = fr.exposure;
  xa.pair = mode != 2;
  xa.use_src = mode == 1;
  xa.first = first;
  xa.hdr = hdr_;
  xa.reset = reset;
  xa.e1 = 1.0f;
  xa.e2 = 1.0f;
  xa.inv_pi = 1.0f / pi;
  DevPtr src = mode == 1 && io ? io->expo_src : expo_computed_;
  if (mode == 1 && !src) return fail("exposure from the game's texture without a source buffer");
  if (!launch("k_expo_scalars", 1, 32, &expo_, &src, &xs_, &xa)) return false;

  // the frame's inputs (k_io.py frame_in): dense render-size images, motion in render pixels ([MVlo]; the interop
  // converts output-resolution motion to render pixels in its input pass)
  KFrameIn fin = {};
  fin.color = cin;
  fin.motion = min_;
  fin.depth = din;
  fin.xs = xs_;
  fin.cpitch = cpitch;
  fin.mpitch = mpitch;
  fin.zpitch = zpitch;
  fin.W = W_;
  fin.H = H_;
  fin.rw = rw;
  fin.rh = rh;
  fin.Bw = Bw_;
  fin.Bh = Bh_;
  fin.flags = (hdr_ ? KIO_HDR : 0) | KIO_MVLO | (depth_inverted_ ? KIO_INV : 0) | (reset ? KIO_RESET : 0) |
              (raw ? KIO_C16 | KIO_M16 : 0);
  fin.mch = mch;
  if (raw) {
    fin.rmsx = io->ms[0];
    fin.rmsy = io->ms[1];
    fin.rmox = io->mo[0];
    fin.rmoy = io->mo[1];
  }
  fin.jx = fr.jitter[0];
  fin.jy = fr.jitter[1];
  fin.sx = fr.mv_scale[0];
  fin.sy = fr.mv_scale[1];
  fin.mox = fr.mv_bias[0];
  fin.moy = fr.mv_bias[1];
  fin.kappa = fin.rho = fin.lam = 1.0f;  // replaced by xs on the GPU
  fin.s = kS;

  // input stage (spec 42)
  KInParams ip = {};
  ip.f = fin;
  ip.Hc = Hc_[p];
  ip.Hh = Hh_[p];
  ip.x0 = net_["X0"];
  ip.cpre = cpre_;
  ip.cpre2 = cpre2_;
  const char* input_kernel = reset ? "k_in_reset" : "k_in_base";
  // k_fast=1: the "_f" variant of an input / output stage kernel (k_io.hip KIO_FAST), else the exact kernel itself
  std::string kname_buf;
  auto kname = [&](const char* n) -> const char* {
    if (!fast_) return n;
    kname_buf = std::string(n) + "_f";
    return kname_buf.c_str();
  };
  const bool use_tiled = tiled_input_ && cfg_.backend == 0 && fin.s == 0.5f && W_ > 0 && W_ <= 16384 && H_ > 0 && H_ <= 16384 &&
      Bw_ > 0 && Bw_ <= 4096 && Bh_ > 0 && Bh_ <= 4096 && rw > 0 && rw <= .68 * W_ && rh > 0 && rh <= .68 * H_ &&
      std::isfinite(fin.jx) && std::isfinite(fin.jy) && std::fabs(fin.jx) <= .5f && std::fabs(fin.jy) <= .5f;
  const bool use_tiled_color = use_tiled && !reset && 4LL * Bw_ >= W_ && 4LL * Bh_ >= H_;
  KiHost input_host = {};
  const bool use_uniforms = use_tiled && input_uniforms_safe(fin, &input_host);
  auto input_stage = [&]() {
    if (use_tiled) {
      KInParams local = ip;
      local.cpre = local.cpre2 = 0;
      if (use_tiled_color) {
        KInColorParams fused = {local, ctap_, ctapT_};
        if (use_uniforms) {
          KInColorUniformParams supplied = {fused, input_host};
          if (!launch(kname("k_in_tiled_color_uniform"), cdiv(Bw_, 8) * cdiv(Bh_, 8), 256, &supplied)) return false;
        } else if (!launch(kname("k_in_tiled_color"), cdiv(Bw_, 8) * cdiv(Bh_, 8), 256, &fused)) return false;
        ++tiled_color_launches_;
      } else if (use_uniforms) {
        KInUniformParams supplied = {local, input_host};
        if (!launch(kname("k_in_tiled_uniform"), cdiv(Bw_, 8) * cdiv(Bh_, 8), 256, &supplied)) return false;
      } else if (!launch(kname("k_in_tiled"), cdiv(Bw_, 8) * cdiv(Bh_, 8), 256, &local)) return false;
      if (use_uniforms) ++input_uniform_launches_;
      ++tiled_launches_;
    } else {
      if (!launch("k_in_color", cdiv((long long)rw * rh, kThreads), kThreads, &ip)) return false;
      ++legacy_color_launches_;
      if (!launch(input_kernel, cdiv(4LL * Bw_ * Bh_, kThreads), kThreads, &ip)) return false;
      ++legacy_base_launches_;
    }
    return true;
  };
  if (diag_input2_ && !input_stage()) return false;
  // The output colour taps use only this frame's colour and exposure (spec 43 4.2/5.3).
  // Produce them before the input colour prepass: no dependency on network output or histories.
  // Their cost belongs to the input timing window below; the full-frame scope is unchanged.
  KOutParams color = {};
  color.f = fin;
  color.ctap = ctap_;
  color.ctapT = ctapT_;
  if (!mark(1)) return false;
  if (!use_tiled_color && !launch(kname("k_out_color"), cdiv((long long)rw * rh, kThreads), kThreads, &color)) return false;
  if (!input_stage()) return false;
  tiled_last_ = use_tiled;
  tiled_color_last_ = use_tiled_color;
  input_uniforms_last_ = use_uniforms;

  // network (spec 41 / 44)
  if (!mark(2) || !network(f)) return false;

  // output stage (spec 43) into the caller's output: RGBA16F rows (interop output copy), dense RGB f32 (interop
  // output pass, FORMATS.md "Per-frame runtime"), or the own RGB f32 buffer (tests)
  if (!mark(3)) return false;
  // interop option handover=staged: the output stage is next (FrameIo.out_signal)
  if (io && io->out_signal && !dev_->signal_fence(io->out_signal, io->out_signal_value)) return dfail();
  const int level = k_filter_level(rw, rh, W_, H_);
  KOutParams op = {};
  op.f = fin;
  op.N = net_["OUT"];
  op.L = L_[p];
  op.Hc = Hc_[p];
  op.table = tables_ + (size_t)level * kTableBytes;
  op.exp2tab = exp2tab_;
  op.Lout = L_[q];
  op.Hcout = Hc_[q];
  op.Hhout = Hh_[q];
  op.ctap = ctap_;
  op.ctapT = ctapT_;
  const int hp = io ? io->out_half_pitch : 0;
  op.out = io && io->out ? io->out : out_;
  if (!op.out) return fail("frame without an output buffer (interop)");
  if (hp > 0) {
    if (hp % 8) return fail("model K: output rows must be a multiple of 8 bytes");
    op.omode = OUT_RGBA16F;
    op.opitch = hp / 8;
  } else {
    op.omode = OUT_RGB32F;
    op.opitch = W_;
  }
  {  // k_io.py out_uniforms: FP32 operations each rounded once (kinv comes from xs on the GPU)
    KOutUni& u = op.u;
    u.invW = 1.0f / (float)W_;
    u.invH = 1.0f / (float)H_;
    u.rhox = (float)rw * u.invW;
    u.rhoy = (float)rh * u.invH;
    u.rho2x = 0.5f * u.rhox;
    u.rho2y = 0.5f * u.rhoy;
    u.jmx = fin.jx - 0.5f;
    u.jmy = fin.jy - 0.5f;
    u.spx = fin.sx / u.rhox;  // [MVlo]
    u.spy = fin.sy / u.rhoy;
    u.invW16 = 1.0f / (float)W16_;
    u.invH16 = 1.0f / (float)H16_;
    u.kinv = 1.0f / fin.kappa;
    u.kux = f32_to_f16(-8.0f / u.rho2x);
    u.kvy = f32_to_f16(-8.0f / u.rho2y);
  }
  if (!launch(kname(compact_head_ ? "k_out_pixel_c" : "k_out_pixel"), cdiv(W16_, 64) * (H16_ / 2), kThreads, &op)) return false;
  if (!mark(4)) return false;

  // frame state (KFrame._advance)
  level_last_ = level;
  omode_last_ = op.omode;
  parity_ = q;
  fidx_ = f + 1;
  last_rw_ = rw;
  last_rh_ = rh;
  jprev_[0] = fr.jitter[0];
  jprev_[1] = fr.jitter[1];
  ++frames_;
  return true;
}

const char* KRuntime::kernels() {
  kernels_text_ = std::string("model=K network=") + (f8_ ? "fused8" : f16_ ? "fused16" : "fused") +
                  (f8_ ? " fp8_stages=" + std::to_string(f8data_.size()) + (f8why_.empty() ? "" : " fp8_kept:" + f8why_)
                       : std::string()) +
                  " input=k_io.co output=k_io.co exposure=k_expo_scalars" +
                  std::string(" input_tiled=") + (tiled_input_ ? "1" : "0") +
                  " input_actual=" + (tiled_last_ ? "tiled" : "legacy") +
                  " tiled_launches=" + std::to_string(tiled_launches_) +
                  " input_uniforms=" + (input_uniforms_last_ ? "host" : "gpu") +
                  " input_uniform_launches=" + std::to_string(input_uniform_launches_) +
                  " color_actual=" + (tiled_color_last_ ? "tiled" : "separate") +
                  " tiled_color_launches=" + std::to_string(tiled_color_launches_) +
                  " legacy_color_launches=" + std::to_string(legacy_color_launches_) +
                  " legacy_base_launches=" + std::to_string(legacy_base_launches_) +
                  (fast_ ? " k_fast=1" : "") +
                  " hidden=fused_d0" + std::string(" head=") + (compact_head_ ? "compact" : "full") +
                  " base=" + std::to_string(Bw_) + "x" + std::to_string(Bh_) +
                  " hdr=" + (hdr_ ? "1" : "0") + " mv=render" +
                  " filter_level=" + std::to_string(level_last_) +
                  " out=" + (omode_last_ == OUT_RGBA16F ? "rgba16f" : "f32") +
                  std::string(compact_head_ ? " post=k_out_pixel_c filter=fused post_last=k_out_pixel_c filter_last=fused"
                                            : " post=k_out_pixel filter=fused post_last=k_out_pixel filter_last=fused") +
                  (diag_input2_ || diag_copyin_
                       ? std::string(" k_diag=") + (diag_input2_ ? "input2" : "") +
                             (diag_input2_ && diag_copyin_ ? "+" : "") + (diag_copyin_ ? "copyin" : "")
                       : std::string());
  return kernels_text_.c_str();
}

bool KRuntime::buffer(const std::string& name, DevPtr* p, size_t* bytes) {
  if (frames_ == 0) return fail("no frame has run yet");
  const int lp = parity_;  // the copies the last frame wrote (the next frame reads them)
  const size_t P16 = (size_t)W16_ * H16_, R = (size_t)info_.rw * info_.rh;
  struct E {
    DevPtr p;
    size_t b;
  };
  std::map<std::string, E> m = {
      {"exposure.pair", {expo_, 12}},       {"k.scalars", {xs_, 16}},
      {"k.x0", {net_["X0"], net_bytes_["X0"]}}, {"k.N", {net_["OUT"], net_bytes_["OUT"]}},
      {"k.Hc", {Hc_[lp], P16 * 8}},         {"k.L", {L_[lp], P16 * 8}},
      {"k.Hh", {Hh_[lp], (size_t)Bw_ * Bh_ * 8}}, {"filter.out", {out_, (size_t)W_ * H_ * 12}},
      {"input.color", {color_, R * 16}},    {"input.depth", {depth_, R * 4}},
      {"input.motion", {motion_, R * 8}},
  };
  // the head map exists only with k_head=full; the default frame holds the compact pixel head in the same buffer
  if (name == "k.N" && compact_head_)
    return fail("buffer k.N (the 40-channel head map) needs the runtime option k_head=full; the default writes the "
                "compact pixel head (buffer k.Np)");
  if (name == "k.Np") {  // ... and the compact pixel head only without it
    if (!compact_head_)
      return fail("buffer k.Np (the compact pixel head) exists only with k_head=compact, the default; with "
                  "k_head=full the frame writes the 40-channel head map (buffer k.N)");
    *p = net_["OUT"];
    *bytes = (size_t)Bw_ * Bh_ * 32;
    return true;
  }
  auto it = m.find(name);
  if (it == m.end() || !it->second.p) return fail("unknown buffer name " + name);
  *p = it->second.p;
  *bytes = it->second.b;
  return true;
}

}  // namespace wgrt
