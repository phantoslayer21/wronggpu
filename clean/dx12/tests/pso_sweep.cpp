// SPDX-License-Identifier: MIT
// PSO-creation sweep: does the driver stack create a compute pipeline for each DXIL file, with the root signature the
// runtime gives that kernel? Windows build (llvm-mingw, build_kernel_replay.py), run under Wine headless.
//   pso_sweep.exe <list.txt>        list: one kernel per line "<name> <buffer|direct> <dxil file>"
// Prints "PSO <name> <rs> hr=0x........ OK|FAILED" per line, then "SWEEP DONE". The two root signatures are those of
// clean/runtime/device_d3d12.cpp (create_root_signature: buffer; create_direct_objects: direct), written out here so that
// the sweep needs neither the runtime nor its gates (the runtime refuses a card that is not RDNA 4).
#include <windows.h>
#include <d3d12.h>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

static const UINT kArgs = 57, kConstants = 2, kGeluT5Slot = 7;

int main(int argc, char** argv) {
  if (argc != 2) { std::printf("usage: pso_sweep <list.txt>\n"); return 2; }
  HMODULE m = LoadLibraryA("d3d12.dll");
  if (!m) { std::printf("no d3d12.dll\n"); return 2; }
  auto create = (PFN_D3D12_CREATE_DEVICE)GetProcAddress(m, "D3D12CreateDevice");
  auto ser = (PFN_D3D12_SERIALIZE_ROOT_SIGNATURE)GetProcAddress(m, "D3D12SerializeRootSignature");
  ID3D12Device* dev = nullptr;
  HRESULT h = create(nullptr, D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), (void**)&dev);
  if (FAILED(h)) { std::printf("D3D12CreateDevice hr=0x%08lx\n", (unsigned long)h); return 2; }

  D3D12_DESCRIPTOR_RANGE ranges[4] = {};
  D3D12_ROOT_PARAMETER pr[4] = {};
  pr[0].ParameterType = pr[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
  pr[1].Descriptor.ShaderRegister = 1;
  pr[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
  pr[2].Constants.Num32BitValues = kConstants + kArgs;
  pr[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
  D3D12_ROOT_SIGNATURE_DESC sd{};
  sd.NumParameters = 4; sd.pParameters = pr;
  auto make_rs = [&](ID3D12RootSignature** out) {
    ID3DBlob *blob = nullptr, *err = nullptr;
    HRESULT r = ser(&sd, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err);
    if (SUCCEEDED(r)) r = dev->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), __uuidof(ID3D12RootSignature), (void**)out);
    return r;
  };
  // buffer: channel UAV table (u0, space 0x7FFF0ADE) + the GELU T5 SRV (t0, space 1) at slot 7
  ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV; ranges[0].NumDescriptors = 1; ranges[0].RegisterSpace = 0x7FFF0ADE;
  ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV; ranges[1].NumDescriptors = 1; ranges[1].RegisterSpace = 1;
  ranges[1].OffsetInDescriptorsFromTableStart = kGeluT5Slot;
  pr[3].DescriptorTable.NumDescriptorRanges = 2; pr[3].DescriptorTable.pDescriptorRanges = ranges;
  ID3D12RootSignature *rs_buffer = nullptr, *rs_direct = nullptr;
  if (FAILED(h = make_rs(&rs_buffer))) { std::printf("buffer root signature hr=0x%08lx\n", (unsigned long)h); return 2; }
  // direct: channel UAV, four SRVs (t0..t3), output UAV (u3), CBV (b1)
  D3D12_DESCRIPTOR_RANGE d[4] = {};
  d[0] = ranges[0]; d[0].OffsetInDescriptorsFromTableStart = 0;
  d[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV; d[1].NumDescriptors = 4; d[1].OffsetInDescriptorsFromTableStart = 1;
  d[2].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV; d[2].NumDescriptors = 1; d[2].BaseShaderRegister = 3; d[2].OffsetInDescriptorsFromTableStart = 5;
  d[3].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_CBV; d[3].NumDescriptors = 1; d[3].BaseShaderRegister = 1; d[3].OffsetInDescriptorsFromTableStart = 6;
  pr[3].DescriptorTable.NumDescriptorRanges = 4; pr[3].DescriptorTable.pDescriptorRanges = d;
  if (FAILED(h = make_rs(&rs_direct))) { std::printf("direct root signature hr=0x%08lx\n", (unsigned long)h); return 2; }

  std::ifstream list(argv[1]);
  std::string line;
  while (std::getline(list, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    std::istringstream s(line);
    std::string name, rs, file;
    if (!(s >> name >> rs >> file)) continue;
    std::ifstream f(file, std::ios::binary);
    std::vector<char> code((std::istreambuf_iterator<char>(f)), {});
    D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};
    pd.pRootSignature = rs == "direct" ? rs_direct : rs_buffer;
    pd.CS = {code.data(), code.size()};
    ID3D12PipelineState* pso = nullptr;
    HRESULT r = code.empty() ? E_FAIL : dev->CreateComputePipelineState(&pd, __uuidof(ID3D12PipelineState), (void**)&pso);
    std::printf("PSO %s %s hr=0x%08lx %s\n", name.c_str(), rs.c_str(), (unsigned long)r, SUCCEEDED(r) ? "OK" : "FAILED");
    std::fflush(stdout);
    if (pso) pso->Release();
  }
  std::printf("SWEEP DONE\n");
  return 0;
}
