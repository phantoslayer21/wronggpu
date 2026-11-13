// SPDX-License-Identifier: MIT
// The driver-channel probe (runtime/wmma_probe.h) on the GPU, through the runtime's own borrowed-device path: creates a
// D3D12 device and DIRECT queue on the first supported RX 9070 / 9060 adapter and hands them to
// wgrt::make_d3d12_device_on, which runs native_driver_probe at create. WrongGPU's own code. No HIP runtime, no model.
//   native_probe_test.exe <build dir holding dx12\manifest.json with native_driver_probe>
// Build (the HIP SDK's clang++ as a host compiler only, as native_direct_build.py does):
//   clang++ -std=c++17 -O2 -D_CRT_SECURE_NO_WARNINGS -DWG_NO_MODEL_K -fms-runtime-lib=static native_probe_test.cpp
//     ../../runtime/{device,device_dx12,device_d3d12,embedded,model,no_model_k,runtime}.cpp
//     ../../runtime/interop/{convert,native_candidate}.cpp ../../converter/src/{json,sha256}.cpp -o native_probe_test.exe
// With WG_TEST_DRIVER_PROBE_FAIL=1 the create must fail with "probe mismatch at float 7". Exit status 0 on PASS.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>

#include <cstdio>
#include <string>

#include "../../runtime/device_d3d12.h"
#include "../../runtime/rdna4.h"

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: native_probe_test.exe <build dir>\n");
    return 2;
  }
  HMODULE dxgi = LoadLibraryExW(L"dxgi.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
  HMODULE d3d12 = LoadLibraryExW(L"d3d12.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
  if (!dxgi || !d3d12) { std::printf("FAIL system DXGI / D3D12 unavailable\n"); return 1; }
  using Factory = HRESULT(WINAPI*)(REFIID, void**);
  using Create = HRESULT(WINAPI*)(IUnknown*, D3D_FEATURE_LEVEL, REFIID, void**);
  auto factory_fn = reinterpret_cast<Factory>(GetProcAddress(dxgi, "CreateDXGIFactory1"));
  auto create = reinterpret_cast<Create>(GetProcAddress(d3d12, "D3D12CreateDevice"));
  IDXGIFactory4* factory = nullptr;
  if (!factory_fn || !create || FAILED(factory_fn(__uuidof(IDXGIFactory4), reinterpret_cast<void**>(&factory)))) {
    std::printf("FAIL DXGI factory\n");
    return 1;
  }
  IDXGIAdapter1* adapter = nullptr;
  std::string name;
  for (UINT i = 0;; ++i) {
    IDXGIAdapter1* a = nullptr;
    if (factory->EnumAdapters1(i, &a) == DXGI_ERROR_NOT_FOUND) break;
    DXGI_ADAPTER_DESC1 desc{};
    char n[256] = {};
    if (SUCCEEDED(a->GetDesc1(&desc))) WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, n, sizeof n - 1, nullptr, nullptr);
    if (wgrt::rdna4_adapter(desc.VendorId, (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0, n)) {
      adapter = a;
      name = n;
      break;
    }
    a->Release();
  }
  factory->Release();
  if (!adapter) { std::printf("SKIP no Radeon RX 9070 / 9060 adapter\n"); return 1; }
  ID3D12Device* device = nullptr;
  ID3D12CommandQueue* queue = nullptr;
  HRESULT h = create(adapter, D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), reinterpret_cast<void**>(&device));
  adapter->Release();
  if (FAILED(h)) { std::printf("FAIL D3D12CreateDevice on %s\n", name.c_str()); return 1; }
  D3D12_COMMAND_QUEUE_DESC qd{};
  qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
  if (FAILED(device->CreateCommandQueue(&qd, __uuidof(ID3D12CommandQueue), reinterpret_cast<void**>(&queue)))) {
    std::printf("FAIL CreateCommandQueue\n");
    device->Release();
    return 1;
  }
  std::printf("adapter %s\n", name.c_str());
  std::string err;
  int status = 1;
  {
    auto dev = wgrt::make_d3d12_device_on(device, queue, argv[1], &err, 64ull << 20);
    if (dev) {
      std::printf("PASS borrowed device created, driver-channel probe passed: %s\n", dev->name().c_str());
      status = 0;
    } else {
      std::printf("FAIL %s\n", err.c_str());
    }
  }
  queue->Release();
  device->Release();
  return status;
}
