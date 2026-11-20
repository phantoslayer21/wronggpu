// SPDX-License-Identifier: MIT
// Probe host of the gfx11 helper tests (test_gfx11_helpers.py): runs ONE compute kernel (a DXIL file built by
// clean/dx12/build_dx12.py's compile_hlsl) on the native D3D12 device with one fixed root signature of the buffer kind (two root
// UAVs over one arena, root constants = group base, grid, then the kernel's arguments, one raw UAV table at space
// 0x7FFF0ADE for the driver's matrix channel) and writes the arena back. Windows build (llvm-mingw), run under Wine:
//   gfx11_probe_host.exe <kernel.dxil> <arena_in.bin> <arena_out.bin> <arena_bytes> <groups> [<arg hex> ...]
// Prints "PSO hr=... OK|FAILED", then "DONE" after a clean run. It does not build each kernel's own root signature from the manifest (kernel_replay.cpp does).
#include <windows.h>
#include <d3d12.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <fstream>
#include <iterator>
#define CK(x) do { HRESULT h_ = (x); if (FAILED(h_)) { printf("FAIL %s hr=0x%08lx\n", #x, (unsigned long)h_); return 2; } } while (0)
static std::vector<char> rd(const char* p) { std::ifstream f(p, std::ios::binary); return std::vector<char>((std::istreambuf_iterator<char>(f)), {}); }
int main(int argc, char** argv) {
  if (argc < 6) return 1;
  auto dxil = rd(argv[1]); auto in = rd(argv[2]);
  UINT64 bytes = strtoull(argv[4], 0, 0); UINT groups = atoi(argv[5]);
  std::vector<UINT> rc = {0, groups};
  for (int i = 6; i < argc; ++i) rc.push_back((UINT)strtoul(argv[i], 0, 0));
  HMODULE m = LoadLibraryA("d3d12.dll");
  auto create = (PFN_D3D12_CREATE_DEVICE)GetProcAddress(m, "D3D12CreateDevice");
  auto ser = (PFN_D3D12_SERIALIZE_ROOT_SIGNATURE)GetProcAddress(m, "D3D12SerializeRootSignature");
  ID3D12Device* dev; CK(create(nullptr, D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), (void**)&dev));
  auto buf = [&](D3D12_HEAP_TYPE t, UINT64 sz, D3D12_RESOURCE_FLAGS fl, D3D12_RESOURCE_STATES st) {
    D3D12_HEAP_PROPERTIES hp{}; hp.Type = t;
    D3D12_RESOURCE_DESC d{}; d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; d.Width = sz; d.Height = d.DepthOrArraySize = d.MipLevels = 1;
    d.SampleDesc.Count = 1; d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; d.Flags = fl;
    ID3D12Resource* r = nullptr; HRESULT h = dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, st, nullptr, __uuidof(ID3D12Resource), (void**)&r);
    if (FAILED(h)) { printf("FAIL buffer 0x%lx\n", (unsigned long)h); exit(2); } return r; };
  ID3D12Resource* arena = buf(D3D12_HEAP_TYPE_DEFAULT, bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
  ID3D12Resource* chan = buf(D3D12_HEAP_TYPE_DEFAULT, 256, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
  ID3D12Resource* up = buf(D3D12_HEAP_TYPE_UPLOAD, bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
  ID3D12Resource* rb = buf(D3D12_HEAP_TYPE_READBACK, bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
  { void* p; D3D12_RANGE r{0,0}; up->Map(0, &r, &p); memset(p, 0, bytes); memcpy(p, in.data(), in.size() < bytes ? in.size() : bytes); up->Unmap(0, nullptr); }
  D3D12_DESCRIPTOR_RANGE rg{}; rg.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV; rg.NumDescriptors = 1; rg.RegisterSpace = 0x7FFF0ADE;
  rg.OffsetInDescriptorsFromTableStart = 0;
  D3D12_ROOT_PARAMETER pr[4] = {};
  pr[0].ParameterType = pr[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV; pr[1].Descriptor.ShaderRegister = 1;
  pr[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS; pr[2].Constants.Num32BitValues = (UINT)rc.size();
  pr[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; pr[3].DescriptorTable.NumDescriptorRanges = 1; pr[3].DescriptorTable.pDescriptorRanges = &rg;
  D3D12_ROOT_SIGNATURE_DESC sd{}; sd.NumParameters = 4; sd.pParameters = pr;
  ID3DBlob *blob = nullptr, *err = nullptr; CK(ser(&sd, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err));
  ID3D12RootSignature* rs; CK(dev->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), __uuidof(ID3D12RootSignature), (void**)&rs));
  D3D12_COMPUTE_PIPELINE_STATE_DESC pd{}; pd.pRootSignature = rs; pd.CS = {dxil.data(), dxil.size()};
  ID3D12PipelineState* pso = nullptr; HRESULT h = dev->CreateComputePipelineState(&pd, __uuidof(ID3D12PipelineState), (void**)&pso);
  printf("PSO hr=0x%08lx %s\n", (unsigned long)h, SUCCEEDED(h) ? "OK" : "FAILED"); fflush(stdout);
  if (FAILED(h)) return 3;
  D3D12_DESCRIPTOR_HEAP_DESC hd{}; hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; hd.NumDescriptors = 1; hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
  ID3D12DescriptorHeap* dh; CK(dev->CreateDescriptorHeap(&hd, __uuidof(ID3D12DescriptorHeap), (void**)&dh));
  D3D12_UNORDERED_ACCESS_VIEW_DESC uv{}; uv.Format = DXGI_FORMAT_R32_TYPELESS; uv.ViewDimension = D3D12_UAV_DIMENSION_BUFFER; uv.Buffer.NumElements = 64; uv.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
  dev->CreateUnorderedAccessView(chan, nullptr, &uv, dh->GetCPUDescriptorHandleForHeapStart());
  D3D12_COMMAND_QUEUE_DESC qd{}; qd.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
  ID3D12CommandQueue* q; CK(dev->CreateCommandQueue(&qd, __uuidof(ID3D12CommandQueue), (void**)&q));
  ID3D12CommandAllocator* al; CK(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, __uuidof(ID3D12CommandAllocator), (void**)&al));
  ID3D12GraphicsCommandList* cl; CK(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, al, nullptr, __uuidof(ID3D12GraphicsCommandList), (void**)&cl));
  auto bar = [&](ID3D12Resource* r, D3D12_RESOURCE_STATES a, D3D12_RESOURCE_STATES b) {
    D3D12_RESOURCE_BARRIER x{}; x.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; x.Transition.pResource = r; x.Transition.StateBefore = a; x.Transition.StateAfter = b;
    x.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES; cl->ResourceBarrier(1, &x); };
  bar(arena, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
  cl->CopyBufferRegion(arena, 0, up, 0, bytes);
  bar(arena, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  bar(chan, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  ID3D12DescriptorHeap* hs[] = {dh}; cl->SetDescriptorHeaps(1, hs);
  cl->SetComputeRootSignature(rs); cl->SetPipelineState(pso);
  cl->SetComputeRootUnorderedAccessView(0, arena->GetGPUVirtualAddress());
  cl->SetComputeRootUnorderedAccessView(1, arena->GetGPUVirtualAddress());
  cl->SetComputeRoot32BitConstants(2, (UINT)rc.size(), rc.data(), 0);
  cl->SetComputeRootDescriptorTable(3, dh->GetGPUDescriptorHandleForHeapStart());
  LARGE_INTEGER t0,t1,fq; QueryPerformanceFrequency(&fq);
  cl->Dispatch(groups, 1, 1);
  bar(arena, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
  cl->CopyBufferRegion(rb, 0, arena, 0, bytes);
  CK(cl->Close());
  ID3D12CommandList* ls[] = {cl}; QueryPerformanceCounter(&t0); q->ExecuteCommandLists(1, ls);
  ID3D12Fence* f; CK(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence), (void**)&f));
  HANDLE ev = CreateEventA(nullptr, FALSE, FALSE, nullptr); q->Signal(f, 1); f->SetEventOnCompletion(1, ev); WaitForSingleObject(ev, 20000); QueryPerformanceCounter(&t1); printf("wall_ms %.3f\n", 1000.0*(t1.QuadPart-t0.QuadPart)/fq.QuadPart);
  printf("fence %llu removed=0x%lx\n", (unsigned long long)f->GetCompletedValue(), (unsigned long)dev->GetDeviceRemovedReason());
  void* p; D3D12_RANGE r{0, (SIZE_T)bytes}; CK(rb->Map(0, &r, &p));
  std::ofstream o(argv[3], std::ios::binary); o.write((char*)p, bytes); o.close();
  printf("DONE\n");
  return 0;
}
