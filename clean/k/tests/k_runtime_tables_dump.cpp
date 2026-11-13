// SPDX-License-Identifier: MIT
// Test aid for clean/k/tests/test_k_runtime_tables.py and test_k_runtime_pack.py: runs the runtime's C++ model K table
// generator (clean/k/runtime/k_filter_table.cpp), loader (clean/k/runtime/model_k_load.cpp) and weight packer
// (clean/k/runtime/k_pack.cpp) and writes what they produce.
//
//   dump table <out.bin>              16 sets of spec 43 section 7 (set 2 level + 1 - theta), uint16 [16][2048][8]
//   dump level <in.txt> <out.txt>     lines "rw rh W H" -> lines "level"
//   dump trunc <in.bin> <out.bin>     binary32 bit patterns (u32) -> binary16 by truncation (u16)
//   dump specs                        JSON {group: [[name, shape], ...]} of k_stage_tensor_specs
//   dump load <model.wgcm> <map> [dir]  "ok" (exit 0) or "error: <message>" (exit 1); with dir, every tensor as
//                                     dir/<stage>/<name>.bin (float32) and dir/net/window_offsets.bin (int32)
//   dump f16 <in.bin> <out.bin>       binary32 bit patterns (u32) -> binary16 round to nearest even (u16, fpconv.h)
//   dump pack <in dir> <out dir>      for every stage with a folder <in dir>/<stage>: its tensors <name>.bin (float32;
//                                     shape from <name>.shape, text "d0 d1 ...", else the spec 48 shape) -> k_pack_stage
//                                     -> <out dir>/<stage>.bin; prints "<stage> ok <bytes>" or "<stage> error: <msg>"
//   dump packwgcm <model.wgcm> <map> <out dir>  load_k_model, then the same for all eleven stages
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "fpconv.h"
#include "k_filter_table.h"
#include "k_pack.h"
#include "model_k.h"
#include "model_k_load.h"

using namespace wgrt;

static int table(const char* path) {
  std::ofstream f(path, std::ios::binary);
  for (int set = 0; set < 16; ++set) {
    const std::vector<uint16_t> t = k_filter_table(set / 2, 1 - set % 2);
    f.write((const char*)t.data(), (std::streamsize)(t.size() * 2));
  }
  return f ? 0 : 2;
}

static int level(const char* in, const char* out) {
  std::ifstream f(in);
  std::ofstream o(out);
  int rw, rh, W, H;
  while (f >> rw >> rh >> W >> H) o << k_filter_level(rw, rh, W, H) << "\n";
  return o ? 0 : 2;
}

static int trunc16(const char* in, const char* out) {
  std::ifstream f(in, std::ios::binary);
  std::ofstream o(out, std::ios::binary);
  uint32_t b;
  while (f.read((char*)&b, 4)) {
    float x;
    std::memcpy(&x, &b, 4);
    const uint16_t h = k_f32_to_f16_trunc(x);
    o.write((const char*)&h, 2);
  }
  return o ? 0 : 2;
}

static int specs() {
  std::printf("{");
  bool first = true;
  for (const std::string& g : k_model_groups()) {
    if (g == "net") continue;
    std::printf("%s\"%s\": [", first ? "" : ", ", g.c_str());
    first = false;
    const auto s = k_stage_tensor_specs(g);
    for (size_t i = 0; i < s.size(); ++i) {
      std::printf("%s[\"%s\", [", i ? ", " : "", s[i].name.c_str());
      for (size_t a = 0; a < s[i].shape.size(); ++a) std::printf("%s%d", a ? ", " : "", s[i].shape[a]);
      std::printf("]]");
    }
    std::printf("]");
  }
  std::printf("}\n");
  return 0;
}

static int load(const char* wgcm, const char* map, const char* dir) {
  KModel m;
  std::string err;
  if (!load_k_model(wgcm, map, &m, &err)) {
    std::printf("error: %s\n", err.c_str());
    return 1;
  }
  if (dir) {
    namespace fs = std::filesystem;
    for (const auto& st : m.stages) {
      fs::create_directories(fs::path(dir) / st.first);
      for (const auto& t : st.second) {
        std::ofstream f(fs::path(dir) / st.first / (t.first + ".bin"), std::ios::binary);
        f.write((const char*)t.second.f.data(), (std::streamsize)(t.second.f.size() * 4));
      }
    }
    fs::create_directories(fs::path(dir) / "net");
    std::ofstream f(fs::path(dir) / "net" / "window_offsets.bin", std::ios::binary);
    for (int r = 0; r < 8; ++r)
      for (int c = 0; c < 2; ++c) {
        const int32_t v = m.window_offsets[r][c];
        f.write((const char*)&v, 4);
      }
  }
  std::printf("ok\n");
  return 0;
}

static int f16(const char* in, const char* out) {
  std::ifstream f(in, std::ios::binary);
  std::ofstream o(out, std::ios::binary);
  uint32_t b;
  while (f.read((char*)&b, 4)) {
    float x;
    std::memcpy(&x, &b, 4);
    const uint16_t h = f32_to_f16(x);
    o.write((const char*)&h, 2);
  }
  return o ? 0 : 2;
}

static bool pack_one(const std::string& stage, const StageTensors& t, const std::filesystem::path& out) {
  std::string err;
  const std::vector<uint8_t> blob = k_pack_stage(stage, t, &err);
  if (blob.empty()) {
    std::printf("%s error: %s\n", stage.c_str(), err.c_str());
    return true;
  }
  std::ofstream f(out / (stage + ".bin"), std::ios::binary);
  f.write((const char*)blob.data(), (std::streamsize)blob.size());
  std::printf("%s ok %zu\n", stage.c_str(), blob.size());
  return (bool)f;
}

static int pack(const char* in, const char* out) {
  namespace fs = std::filesystem;
  fs::create_directories(out);
  for (const KStageInfo& s : k_stage_table()) {
    const fs::path dir = fs::path(in) / s.name;
    if (!fs::is_directory(dir)) continue;
    StageTensors t;
    for (const auto& e : fs::directory_iterator(dir)) {
      if (e.path().extension() != ".bin") continue;
      const std::string name = e.path().stem().string();
      Tensor x;
      std::ifstream sh(dir / (name + ".shape"));
      if (sh) {
        int d;
        while (sh >> d) x.shape.push_back(d);
      } else {
        for (const KTensorSpec& sp : k_stage_tensor_specs(s.name))
          if (sp.name == name) x.shape = sp.shape;
      }
      std::ifstream f(e.path(), std::ios::binary);
      const auto bytes = (size_t)fs::file_size(e.path());
      x.f.resize(bytes / 4);
      f.read((char*)x.f.data(), (std::streamsize)(x.f.size() * 4));
      t[name] = std::move(x);
    }
    if (!pack_one(s.name, t, out)) return 2;
  }
  return 0;
}

static int packwgcm(const char* wgcm, const char* map, const char* out) {
  KModel m;
  std::string err;
  if (!load_k_model(wgcm, map, &m, &err)) {
    std::printf("error: %s\n", err.c_str());
    return 1;
  }
  std::filesystem::create_directories(out);
  for (const KStageInfo& s : k_stage_table())
    if (!pack_one(s.name, m.stages[s.name], out)) return 2;
  return 0;
}

int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "";
  if (mode == "table" && argc == 3) return table(argv[2]);
  if (mode == "level" && argc == 4) return level(argv[2], argv[3]);
  if (mode == "trunc" && argc == 4) return trunc16(argv[2], argv[3]);
  if (mode == "specs") return specs();
  if (mode == "load" && (argc == 4 || argc == 5)) return load(argv[2], argv[3], argc == 5 ? argv[4] : nullptr);
  if (mode == "f16" && argc == 4) return f16(argv[2], argv[3]);
  if (mode == "pack" && argc == 4) return pack(argv[2], argv[3]);
  if (mode == "packwgcm" && argc == 5) return packwgcm(argv[2], argv[3], argv[4]);
  std::fprintf(stderr, "usage: see the comment at the top of k_runtime_tables_dump.cpp\n");
  return 2;
}
