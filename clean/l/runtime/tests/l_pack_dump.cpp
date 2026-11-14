// SPDX-License-Identifier: MIT
// Model L's runtime packer (clean/l/runtime/l_pack.cpp) on the CPU, for test_l_runtime_pack.py:
//
//   l_pack_dump pack <tensor dir> <out dir>
//       reads <tensor dir>/<stage>/<name>.f32 (float32 values, row-major) for every stage folder (enc0, enc0_block,
//       dec0, dec0_block) and <tensor dir>/enc0/T.u8 (uint8), runs wgrt::l::pack_model and writes
//       <out dir>/<blob>.bin; prints "OK <blobs>" or "ERR <message>"
//   l_pack_dump launches
//       prints the default table's encoder launches, then the decoder launches, one line each:
//       "enc|..." / "dec|..." with the fields of wgrt::l::Launch (the test builds the same lines from
//       clean/l/runner/l_pack.py)
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "../l_pack.h"

namespace fs = std::filesystem;

namespace {

bool read_stage(const fs::path& dir, wgrt::StageTensors* t) {
  if (!fs::is_directory(dir)) return false;
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

std::string join(const std::vector<int>& v) {
  std::string s;
  for (int x : v) s += (s.empty() ? "" : ",") + std::to_string(x);
  return s;
}

void print(const char* where, const wgrt::l::Launch& l) {
  char clamp[32] = "";
  if (l.clamp != 0.0f) std::snprintf(clamp, sizeof clamp, "%.9g", l.clamp);
  std::printf("%s|%s|%s|%s|%s|%d|%d|%s|%s|%s|%s|%d|%d|%d|%s|%s|%s|%s|%s|%s|%s|%s|%s\n", where, l.kind.c_str(),
              l.blob.c_str(), l.src.c_str(), l.dst.c_str(), l.C_in, l.C_out, l.w.c_str(), l.b.c_str(), l.acc.c_str(),
              l.out_round.c_str(), l.C, l.heads, l.mlp, l.offsets.c_str(), l.params.c_str(), l.in_format.c_str(),
              l.out_format.c_str(), l.border.c_str(), l.dec_pixel.c_str(), join(l.groups).c_str(), clamp,
              join(l.values).c_str());
}

}  // namespace

int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "";
  if (mode == "pack" && argc == 4) {
    std::map<std::string, wgrt::StageTensors> params;
    for (const auto& e : fs::directory_iterator(argv[2]))
      if (e.is_directory() && e.path().filename() != "out")
        read_stage(e.path(), &params[e.path().filename().string()]);
    std::map<std::string, std::vector<uint8_t>> blobs;
    std::string err;
    if (!wgrt::l::pack_model(params, wgrt::drow_gfx12(), &blobs, &err)) return std::printf("ERR %s\n", err.c_str()), 0;
    for (const auto& kv : blobs) {
      std::ofstream f(fs::path(argv[3]) / (kv.first + ".bin"), std::ios::binary);
      f.write(reinterpret_cast<const char*>(kv.second.data()), (std::streamsize)kv.second.size());
    }
    std::printf("OK %zu\n", blobs.size());
    return 0;
  }
  if (mode == "launches") {
    for (const wgrt::l::Launch& l : wgrt::l::encoder_launches()) print("enc", l);
    for (const wgrt::l::Launch& l : wgrt::l::decoder_launches()) print("dec", l);
    return 0;
  }
  std::printf("ERR usage\n");
  return 2;
}
