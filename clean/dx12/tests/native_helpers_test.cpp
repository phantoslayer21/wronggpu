// SPDX-License-Identifier: MIT
// Original synthetic helper-kernel runner. Each invocation owns one HIP or native D3D12 device; the Python caller
// compares raw FP32 bits from separate processes, keeping native module auditing independent of the HIP reference.
#include "../../runtime/device.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <tlhelp32.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

static void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error(what);
}

static void modules(const std::string& path, const char* phase) {
  HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, GetCurrentProcessId());
  require(snap != INVALID_HANDLE_VALUE, "module snapshot failed");
  MODULEENTRY32 entry{};
  entry.dwSize = sizeof(entry);
  std::ofstream out(path, std::ios::app);
  require(bool(out), "module log open failed");
  bool hip = false, any = false;
  if (Module32First(snap, &entry)) {
    do {
      any = true;
      std::string name(entry.szModule);
      std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return char(std::tolower(c)); });
      out << phase << " " << entry.szExePath << "\n";
      hip = hip || name.find("amdhip") != std::string::npos || name.find("hiprtc") != std::string::npos;
    } while (Module32Next(snap, &entry));
  }
  CloseHandle(snap);
  require(any, "module enumeration returned no modules");
  require(!hip, "native helper process loaded a HIP runtime/compiler module");
}

static void fill_checks(wgrt::Device& dev) {
  wgrt::DevPtr p = 0;
  const size_t bytes = 4096;
  require(dev.alloc(&p, bytes), "fill allocation: " + dev.err());
  std::vector<unsigned char> before(bytes), expected(bytes), actual(bytes);
  for (size_t i = 0; i < bytes; ++i) before[i] = static_cast<unsigned char>(i * 73 + 19);
  for (unsigned start : {0u, 1u, 2u, 3u, 1023u}) {
    for (unsigned count : {0u, 1u, 2u, 3u, 4u, 5u, 7u, 31u, 255u, 256u, 257u, 1025u}) {
      for (int value : {0, 1, 127, 255, 0x1a5}) {
        expected = before;
        std::fill(expected.begin() + start, expected.begin() + start + count, static_cast<unsigned char>(value));
        require(dev.upload(p, before.data(), bytes), "fill upload: " + dev.err());
        require(dev.memset(p + start, value, count), "fill dispatch: " + dev.err());
        require(dev.download(actual.data(), p, bytes), "fill readback: " + dev.err());
        require(actual == expected, "fill differs at start=" + std::to_string(start) + " count=" +
                                      std::to_string(count) + " value=" + std::to_string(value));
      }
    }
  }
  dev.free(p);
  std::printf("fill: 300 offset/count/value cases passed (zero length and masked edge bytes included)\n");
}

int main(int argc, char** argv) {
  if (argc != 7) {
    std::printf("usage: native_helpers_test hip|native <rocm> <HIP build> <DX12 build> <input.bin> <output.bin>\n");
    return 2;
  }
  try {
    const bool native = std::string(argv[1]) == "native";
    require(native || std::string(argv[1]) == "hip", "unknown backend");
#if defined(WG_HELPER_HIP_ONLY)
    require(!native, "HIP-only helper executable rejects the native backend");
#endif
    const std::string module_path = std::string(argv[6]) + ".modules.txt";
    if (native) modules(module_path, "before");
    std::ifstream input(argv[5], std::ios::binary);
    uint32_t n = 0, cases = 0;
    input.read(reinterpret_cast<char*>(&n), sizeof(n));
    input.read(reinterpret_cast<char*>(&cases), sizeof(cases));
    require(input && n > 0 && n < (1u << 24) && cases > 0 && cases <= 128, "invalid synthetic input header");
    const size_t guard = 16, color_count = size_t(n) * 4, rgb_count = size_t(n) * 3;
    std::vector<uint32_t> color(color_count + guard, 0xa5c31f7bu), rgb(rgb_count + guard, 0xa5c31f7bu), pi(cases);
    input.read(reinterpret_cast<char*>(color.data()), color_count * 4);
    input.read(reinterpret_cast<char*>(rgb.data()), rgb_count * 4);
    input.read(reinterpret_cast<char*>(pi.data()), cases * 4);
    require(bool(input), "truncated synthetic input");
    std::string error;
#if defined(WG_HELPER_HIP_ONLY)
    auto dev = wgrt::make_hip_device(argv[2], &error);
#else
    auto dev = native ? wgrt::make_d3d12_device(argv[4], &error) : wgrt::make_hip_device(argv[2], &error);
#endif
    require(bool(dev), "device create: " + error);
    std::printf("device %s: %s\n", argv[1], dev->name().c_str());
    require(dev->load_module(std::string(argv[3]) + "/frame.co"), "frame module: " + dev->err());
    wgrt::DevPtr in = 0, out = 0, scaled = 0;
    const size_t color_bytes = color.size() * 4, rgb_bytes = rgb.size() * 4;
    require(dev->alloc(&in, color_bytes) && dev->alloc(&out, color_bytes) && dev->alloc(&scaled, rgb_bytes),
            "helper allocations: " + dev->err());
    require(dev->upload(in, color.data(), color_bytes), "color upload: " + dev->err());
    std::vector<uint32_t> cresult(color.size(), 0xa5c31f7bu), sresult(rgb.size());
    std::ofstream output(argv[6], std::ios::binary);
    output.write(reinterpret_cast<const char*>(&n), sizeof(n));
    output.write(reinterpret_cast<const char*>(&cases), sizeof(cases));
    for (uint32_t k = 0; k < cases; ++k) {
      float factor;
      std::memcpy(&factor, &pi[k], sizeof(factor));
      int pixels = int(n), channels = int(rgb_count);
      std::fill(cresult.begin(), cresult.end(), 0xa5c31f7bu);
      require(dev->upload(out, cresult.data(), color_bytes) && dev->upload(scaled, rgb.data(), rgb_bytes),
              "helper poison/upload: " + dev->err());
      void* divide_args[] = {&in, &out, &factor, &pixels};
      void* scale_args[] = {&scaled, &factor, &channels};
      require(dev->launch("color_divide", (n + 255u) / 256u, 256, divide_args), "color_divide: " + dev->err());
      require(dev->launch("scale_rgb", (channels + 255u) / 256u, 256, scale_args), "scale_rgb: " + dev->err());
      require(dev->download(cresult.data(), out, color_bytes) && dev->download(sresult.data(), scaled, rgb_bytes),
              "helper readback: " + dev->err());
      output.write(reinterpret_cast<const char*>(cresult.data()), color_bytes);
      output.write(reinterpret_cast<const char*>(sresult.data()), rgb_bytes);
      require(bool(output), "helper output write failed");
    }
    if (native) {
      int unused = 0;
      void* args[] = {&unused};
      require(dev->launch("timing_barrier", 1, 32, args) && dev->sync(), "timing_barrier: " + dev->err());
      fill_checks(*dev);
      modules(module_path, "after_dispatch");
    }
    dev->free(in);
    dev->free(out);
    dev->free(scaled);
    dev.reset();
    if (native) modules(module_path, "after_destroy");
    std::printf("wrote %u cases x %u pixels; raw FP32 and guard words preserved\n", cases, n);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
