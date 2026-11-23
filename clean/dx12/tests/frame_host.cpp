// SPDX-License-Identifier: MIT
// A whole Model M frame on the runtime library's native D3D12 executor (backend 3), driven from a plain C++ host so that it runs
// headless under Wine on Linux (no Python on the Windows side). WrongGPU's own code. Synthetic model and inputs only: the model
// folder (raw tensors) and the input files are written by frame_parity.py; nothing here reads a model.wgcm.
//
//   frame_host.exe <build dir> <spec file> <out dir>
//
// <build dir> holds wronggpu_runtime.dll (clean/runner/build.py, a Windows build) and dx12\manifest.json (the D3D12 kernels).
// <spec file>, one item per line (paths relative to the spec file; '#' starts a comment):
//   model <dir>                         the folder with the raw tensors (model_format 1)
//   config <field> <int|float|string>   a WgRtConfig field by name (see wg_runtime.h); block_threads / block_wpg / d_row as
//                                       "<field> <i> <value>"
//   frame <rw> <rh> <reset> <mv_scale0> <mv_scale1> <mv_bias0> <mv_bias1> <jitter0> <jitter1> <exposure> <pre_exposure>
//         <color.bin> <depth.bin> <motion.bin>   float32 arrays [rh][rw][3], [rh][rw], [rh][rw][2]
//   dump <buffer name>                  after each frame write the named buffer to <out dir>/f<frame>_<name>.bin
//   time <repeats>                      after the last frame, <repeats> timed frames of the last frame's inputs (no reset):
//                                       the GPU times in ms go to <out dir>/times.txt, one per line
// The runtime library is loaded at run time, as a game does; `kernels` and `info` of the last frame go to <out dir>/report.txt.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "../../runtime/wg_runtime.h"

namespace {

struct Frame {
  int rw = 0, rh = 0, reset = 0;
  float scale[2] = {}, bias[2] = {}, jitter[2] = {}, exposure = 1, pre_exposure = 1;
  std::string color, depth, motion;
};

std::vector<char> read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) { std::fprintf(stderr, "cannot read %s\n", path.c_str()); std::exit(2); }
  return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
void write_file(const std::string& path, const void* data, size_t bytes) {
  std::ofstream f(path, std::ios::binary);
  f.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
}
std::string folder_of(const std::string& path) {
  const size_t at = path.find_last_of("/\\");
  return at == std::string::npos ? std::string() : path.substr(0, at + 1);
}

template <class T> T load(HMODULE lib, const char* name) {
  T f = reinterpret_cast<T>(GetProcAddress(lib, name));
  if (!f) { std::fprintf(stderr, "runtime library has no %s\n", name); std::exit(2); }
  return f;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) { std::fprintf(stderr, "usage: frame_host.exe <build dir> <spec file> <out dir>\n"); return 2; }
  const std::string build = argv[1], spec = argv[2], out = std::string(argv[3]) + "\\", base = folder_of(spec);
  HMODULE lib = LoadLibraryA((build + "\\wronggpu_runtime.dll").c_str());
  if (!lib) { std::fprintf(stderr, "cannot load %s\\wronggpu_runtime.dll (error %lu)\n", build.c_str(), GetLastError()); return 2; }
  auto create = load<int (*)(const WgRtConfig*, void**, char*, int)>(lib, "wg_rt_create");
  auto destroy = load<void (*)(void*)>(lib, "wg_rt_destroy");
  auto rt_error = load<const char* (*)(void*)>(lib, "wg_rt_error");
  auto upload = load<int (*)(void*, int, int, const float*, const float*, const float*)>(lib, "wg_rt_upload_inputs");
  auto frame_fn = load<int (*)(void*, const WgRtFrame*)>(lib, "wg_rt_frame");
  auto timed = load<int (*)(void*, const WgRtFrame*, float*)>(lib, "wg_rt_frame_timed");
  auto sync = load<int (*)(void*)>(lib, "wg_rt_sync");
  auto info = load<int (*)(void*, WgRtFrameInfo*)>(lib, "wg_rt_info");
  auto buffer = load<int (*)(void*, const char*, uintptr_t*, size_t*)>(lib, "wg_rt_buffer");
  auto download = load<int (*)(void*, size_t, void*, size_t)>(lib, "wg_rt_download");
  auto kernels = load<const char* (*)(void*)>(lib, "wg_rt_kernels");

  WgRtConfig cfg;
  std::memset(&cfg, 0, sizeof cfg);
  cfg.backend = 3;
  cfg.model_format = 1;
  std::vector<std::string> strings;   // owns every string the config points to
  strings.reserve(64);
  auto keep = [&](const std::string& s) { strings.push_back(s); return strings.back().c_str(); };
  cfg.build_dir = keep(build);
  std::vector<Frame> frames;
  std::vector<std::string> dumps;
  unsigned repeats = 0;
  std::ifstream in(spec);
  if (!in) { std::fprintf(stderr, "cannot read %s\n", spec.c_str()); return 2; }
  for (std::string line; std::getline(in, line);) {
    if (const size_t hash = line.find('#'); hash != std::string::npos) line.resize(hash);
    std::istringstream s(line);
    std::string key;
    if (!(s >> key)) continue;
    if (key == "model") { std::string d; s >> d; cfg.model_dir = keep(d.size() > 1 && d[1] == ':' ? d : base + d); }
    else if (key == "dump") { std::string n; s >> n; dumps.push_back(n); }
    else if (key == "time") s >> repeats;
    else if (key == "frame") {
      Frame f;
      s >> f.rw >> f.rh >> f.reset >> f.scale[0] >> f.scale[1] >> f.bias[0] >> f.bias[1] >> f.jitter[0] >> f.jitter[1] >> f.exposure >>
          f.pre_exposure >> f.color >> f.depth >> f.motion;
      if (!s) { std::fprintf(stderr, "bad frame line: %s\n", line.c_str()); return 2; }
      frames.push_back(f);
    } else if (key == "config") {
      std::string field;
      s >> field;
      int i = 0;
      std::string v;
      auto indexed = [&](int* array, int n) { s >> i >> v; if (i < 0 || i >= n) { std::fprintf(stderr, "bad index\n"); std::exit(2); } array[i] = std::atoi(v.c_str()); };
      if (field == "block_threads") indexed(cfg.block_threads, 5);
      else if (field == "block_wpg") indexed(cfg.block_wpg, 5);
      else if (field == "d_row") indexed(cfg.d_row, 16);
      else {
        s >> v;
#define INT(name) if (field == #name) cfg.name = std::atoi(v.c_str()); else
#define STR(name) if (field == #name) cfg.name = keep(v); else
        INT(max_rw) INT(max_rh) INT(ow) INT(oh) INT(depth_inverted) INT(auto_exposure) INT(swin_threads) INT(gemm_threads)
        INT(nt) INT(wpw64) INT(wpw96) INT(wpw128) INT(swin_force_clamp) INT(swin_fuse) INT(post_wave) INT(recon_output) INT(hdr)
        STR(swin_code_object) STR(map_path) STR(enc0_variant) STR(dec0_variant) STR(exposure_variant) STR(post_kernel)
        STR(filter_kernel) STR(swin_stage_variants) STR(swin_fuse_stages) STR(runtime_options) STR(swin_bneck) STR(dec1_dec0)
        STR(swin_mid) STR(post_pack)
        if (field == "alpha") cfg.alpha = static_cast<float>(std::atof(v.c_str()));
        else { std::fprintf(stderr, "unknown config field %s\n", field.c_str()); return 2; }
      }
    } else { std::fprintf(stderr, "unknown spec line: %s\n", line.c_str()); return 2; }
  }
  if (frames.empty() || !cfg.model_dir) { std::fprintf(stderr, "spec needs a model and a frame\n"); return 2; }

  void* rt = nullptr;
  char err[2048] = {};
  if (create(&cfg, &rt, err, sizeof err) != 0) { std::printf("create FAILED: %s\n", err); return 1; }
  std::printf("created\n");
  auto fail = [&](const char* what) { std::printf("%s FAILED: %s\n", what, rt_error(rt)); destroy(rt); return 1; };
  WgRtFrame fr;
  for (size_t n = 0; n < frames.size(); ++n) {
    const Frame& f = frames[n];
    const std::vector<char> color = read_file(base + f.color), depth = read_file(base + f.depth), motion = read_file(base + f.motion);
    const size_t px = static_cast<size_t>(f.rw) * f.rh;
    if (color.size() != px * 12 || depth.size() != px * 4 || motion.size() != px * 8) { std::printf("input size mismatch\n"); return 2; }
    if (upload(rt, f.rw, f.rh, reinterpret_cast<const float*>(color.data()), reinterpret_cast<const float*>(depth.data()),
               reinterpret_cast<const float*>(motion.data())) != 0) return fail("upload");
    std::memset(&fr, 0, sizeof fr);
    fr.rw = f.rw; fr.rh = f.rh; fr.reset = f.reset; fr.exposure = f.exposure; fr.pre_exposure = f.pre_exposure;
    for (int i = 0; i < 2; ++i) { fr.mv_scale[i] = f.scale[i]; fr.mv_bias[i] = f.bias[i]; fr.jitter[i] = f.jitter[i]; }
    if (frame_fn(rt, &fr) != 0 || sync(rt) != 0) return fail("frame");
    for (const std::string& name : dumps) {
      uintptr_t ptr = 0;
      size_t bytes = 0;
      if (buffer(rt, name.c_str(), &ptr, &bytes) != 0) continue;   // a buffer this route does not have
      std::vector<char> data(bytes);
      if (download(rt, ptr, data.data(), bytes) != 0) return fail(("download " + name).c_str());
      write_file(out + "f" + std::to_string(n) + "_" + name + ".bin", data.data(), bytes);
    }
    std::printf("frame %zu done\n", n);
    std::fflush(stdout);
  }
  if (repeats) {
    fr.reset = 0;
    std::string times;
    for (unsigned i = 0; i < repeats; ++i) {
      float ms = 0;
      if (timed(rt, &fr, &ms) != 0) return fail("timed frame");
      times += std::to_string(ms) + "\n";
    }
    write_file(out + "times.txt", times.data(), times.size());
  }
  WgRtFrameInfo fi;
  std::string report = std::string("kernels ") + kernels(rt) + "\n";
  if (info(rt, &fi) == 0) report += "frame_index " + std::to_string(fi.index) + " render " + std::to_string(fi.rw) + "x" + std::to_string(fi.rh) + "\n";
  write_file(out + "report.txt", report.data(), report.size());
  destroy(rt);
  std::printf("DONE\n");
  return 0;
}
