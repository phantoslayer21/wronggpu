// SPDX-License-Identifier: MIT
// Assembles a DXIL text listing (LLVM IR, as written by dxc -Fc) into an unsigned DXIL container, with the DXC
// compiler library's assembler (IDxcAssembler). build_dx12.py uses it for the exact-arithmetic rewrite of
// clean/dx12/README.md ("Arithmetic mode"); dxv.exe validates and signs the result.
//   dxil_asm <dxcompiler.dll> <in.ll> <out.dxil>
#define NOMINMAX
#include <windows.h>
#include <dxcapi.h>

#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

static int fail(const char* what, HRESULT hr) {
  std::fprintf(stderr, "dxil_asm: %s failed (0x%08lx)\n", what, (unsigned long)hr);
  return 2;
}

int main(int argc, char** argv) {
  if (argc != 4) {
    std::fprintf(stderr, "usage: dxil_asm <dxcompiler.dll> <in.ll> <out.dxil>\n");
    return 1;
  }
  HMODULE lib = LoadLibraryExA(argv[1], nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
  if (!lib) return fail("LoadLibrary(dxcompiler)", HRESULT_FROM_WIN32(GetLastError()));
  auto create = (DxcCreateInstanceProc)GetProcAddress(lib, "DxcCreateInstance");
  if (!create) return fail("DxcCreateInstance lookup", E_FAIL);
  std::ifstream f(argv[2], std::ios::binary);
  std::string text((std::istreambuf_iterator<char>(f)), {});
  if (text.empty()) return fail("reading the listing", E_FAIL);
  IDxcUtils* utils = nullptr;
  IDxcAssembler* as = nullptr;
  HRESULT hr = create(CLSID_DxcUtils, __uuidof(IDxcUtils), (void**)&utils);
  if (FAILED(hr)) return fail("CLSID_DxcUtils", hr);
  hr = create(CLSID_DxcAssembler, __uuidof(IDxcAssembler), (void**)&as);
  if (FAILED(hr)) return fail("CLSID_DxcAssembler", hr);
  IDxcBlobEncoding* src = nullptr;
  hr = utils->CreateBlob(text.data(), (UINT32)text.size(), DXC_CP_UTF8, &src);
  if (FAILED(hr)) return fail("CreateBlob", hr);
  IDxcOperationResult* res = nullptr;
  hr = as->AssembleToContainer(src, &res);
  if (FAILED(hr)) return fail("AssembleToContainer", hr);
  HRESULT st = E_FAIL;
  res->GetStatus(&st);
  if (FAILED(st)) {
    IDxcBlobEncoding* err = nullptr;
    if (SUCCEEDED(res->GetErrorBuffer(&err)) && err)
      std::fprintf(stderr, "%.*s\n", (int)err->GetBufferSize(), (const char*)err->GetBufferPointer());
    return fail("assembly", st);
  }
  IDxcBlob* out = nullptr;
  hr = res->GetResult(&out);
  if (FAILED(hr) || !out) return fail("GetResult", hr);
  std::ofstream o(argv[3], std::ios::binary);
  o.write((const char*)out->GetBufferPointer(), (std::streamsize)out->GetBufferSize());
  return o ? 0 : fail("writing the container", E_FAIL);
}
