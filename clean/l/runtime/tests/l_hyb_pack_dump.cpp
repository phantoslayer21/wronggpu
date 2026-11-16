// SPDX-License-Identifier: MIT
// Model L's HYBRID weight values (clean/l/runtime/l_hyb_pack.h) through the runtime's packers, on the CPU, for
// test_l_runtime_hyb_pack.py:
//
//   l_hyb_pack_dump hyb <tensor dir> <out dir> [mutant]
//       reads <tensor dir>/<stage>/<name>.f32 (and enc0/T.u8) for every stage folder, applies hyb_stage_values to the
//       block stages, packs as the runtime does (pack_stage_blob for the ten shared stages, l::pack_model for Model L's
//       own launches) and writes <out dir>/<blob>.bin; prints "OK <blobs>" or "ERR <message>". "mutant": the gammas
//       without the sqrtf(C) fold (the test's negative control: the gamma-bearing blobs must differ).
//   l_hyb_pack_dump wgcm <model.wgcm> <map> <out dir>
//       wgrt::l::load_wgcm_model, writes every tensor to <out dir>/tensors/<stage>/<name>.f32 (enc0 T as .u8) and the
//       hybrid blobs to <out dir>/blobs/; prints "OK <blobs>" or "ERR <message>" (no values printed: model data).
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "../l_hyb_pack.h"
#include "../l_model.h"
#include "../l_pack.h"

namespace fs = std::filesystem;

namespace {

const char* const kShared[] = {"enc1", "enc2", "b0", "b1", "b2", "b3", "b4", "b5", "dec2", "dec1"};

bool read_stage(const fs::path& dir, wgrt::StageTensors* t) {
  for (const auto& e : fs::directory_iterator(dir)) {
    const fs::path p = e.path();
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    const std::streamsize n = f.tellg();
    f.seekg(0);
    wgrt::Tensor x;
    if (p.extension() == ".f32") {
      x.f.resize((size_t)n / 4);
      f.read(reinterpret_cast<char*>(x.f.data()), n);
    } else if (p.extension() == ".u8") {
      x.u.resize((size_t)n);
      f.read(reinterpret_cast<char*>(x.u.data()), n);
    } else {
      continue;
    }
    (*t)[p.stem().string()] = x;
  }
  return true;
}

// The runtime's packing of a Model L model in the hybrid values (runtime.cpp, WG_L_FORM=hybrid on native D3D12).
bool pack_hyb(const std::map<std::string, wgrt::StageTensors>& model, bool mutant,
              std::map<std::string, std::vector<uint8_t>>* blobs, std::string* err) {
  std::map<std::string, wgrt::StageTensors> h;
  for (const auto& [s, t] : model) {
    if (!wgrt::l::hyb_block_stage(s)) {
      h[s] = t;
      continue;
    }
    h[s] = wgrt::l::hyb_stage_values(t);
    if (mutant)
      for (const char* n : {"gamma1", "gamma2"}) {
        auto it = h[s].find(n);
        const auto src = t.find(n);
        if (it != h[s].end() && src != t.end())
          for (size_t i = 0; i < it->second.f.size(); ++i) it->second.f[i] = wgrt::rn16(src->second.f[i]);
      }
  }
  for (const char* s : kShared) {
    const auto it = h.find(s);
    if (it == h.end()) return *err = std::string("missing stage ") + s, false;
    if (!wgrt::pack_stage_blob(s, it->second, wgrt::drow_gfx12(), &(*blobs)[s], err)) return false;
  }
  std::map<std::string, std::vector<uint8_t>> lb;
  if (!wgrt::l::pack_model(h, wgrt::drow_gfx12(), &lb, err)) return false;
  for (auto& kv : lb) (*blobs)[kv.first] = std::move(kv.second);
  return true;
}

void write_blobs(const fs::path& dir, const std::map<std::string, std::vector<uint8_t>>& blobs) {
  fs::create_directories(dir);
  for (const auto& kv : blobs) {
    std::ofstream f(dir / (kv.first + ".bin"), std::ios::binary);
    f.write(reinterpret_cast<const char*>(kv.second.data()), (std::streamsize)kv.second.size());
  }
}

}  // namespace

int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "";
  std::map<std::string, wgrt::StageTensors> model;
  std::map<std::string, std::vector<uint8_t>> blobs;
  std::string err;
  if (mode == "hyb" && (argc == 4 || argc == 5)) {
    for (const auto& e : fs::directory_iterator(argv[2]))
      if (e.is_directory() && e.path().filename() != "out") read_stage(e.path(), &model[e.path().filename().string()]);
    if (!pack_hyb(model, argc == 5 && std::string(argv[4]) == "mutant", &blobs, &err))
      return std::printf("ERR %s\n", err.c_str()), 0;
    write_blobs(argv[3], blobs);
    std::printf("OK %zu\n", blobs.size());
    return 0;
  }
  if (mode == "wgcm" && argc == 5) {
    if (!wgrt::l::load_wgcm_model(argv[2], argv[3], &model, &err)) return std::printf("ERR %s\n", err.c_str()), 0;
    const fs::path out = argv[4];
    for (const auto& [s, t] : model) {
      const fs::path d = out / "tensors" / s;
      fs::create_directories(d);
      for (const auto& [n, x] : t) {
        std::ofstream f(d / (n + (x.u.empty() ? ".f32" : ".u8")), std::ios::binary);
        if (x.u.empty())
          f.write(reinterpret_cast<const char*>(x.f.data()), (std::streamsize)(x.f.size() * 4));
        else
          f.write(reinterpret_cast<const char*>(x.u.data()), (std::streamsize)x.u.size());
      }
    }
    if (!pack_hyb(model, false, &blobs, &err)) return std::printf("ERR %s\n", err.c_str()), 0;
    write_blobs(out / "blobs", blobs);
    std::printf("OK %zu\n", blobs.size());
    return 0;
  }
  std::printf("ERR usage\n");
  return 2;
}
