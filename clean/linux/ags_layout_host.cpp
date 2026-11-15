// SPDX-License-Identifier: MIT
// D3D12 host of the ags-layout test (clean/linux/run_wine_tests.py): runs the wave32 and the wave64 entry point of
// ags_layout.hlsl once each and writes the output slots. Root signature as the runtime's kernels bind the driver
// channel: a descriptor table with one raw UAV at u0, space 0x7FFF0ADE, and a root UAV (u1) for the data.
//   ags_layout_host <main32.dxil> <main64.dxil> <inputs.bin> <outputs.bin>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void need(HRESULT hr, const char* what) {
  if (FAILED(hr)) {
    std::printf("FAIL %s 0x%08lx\n", what, (unsigned long)hr);
    std::exit(1);
  }
}

static std::vector<char> read_file(const char* path) {
  std::vector<char> d;
  FILE* f = std::fopen(path, "rb");
  if (!f) return d;
  std::fseek(f, 0, SEEK_END);
  d.resize(std::ftell(f));
  std::fseek(f, 0, SEEK_SET);
  if (std::fread(d.data(), 1, d.size(), f) != d.size()) d.clear();
  std::fclose(f);
  return d;
}

static ID3D12Resource* buffer(ID3D12Device* dev, D3D12_HEAP_TYPE type, UINT64 size, D3D12_RESOURCE_STATES state,
                              D3D12_RESOURCE_FLAGS flags) {
  D3D12_HEAP_PROPERTIES hp = {};
  hp.Type = type;
  D3D12_RESOURCE_DESC rd = {};
  rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  rd.Width = size;
  rd.Height = 1;
  rd.DepthOrArraySize = 1;
  rd.MipLevels = 1;
  rd.SampleDesc.Count = 1;
  rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  rd.Flags = flags;
  ID3D12Resource* r = nullptr;
  need(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, state, nullptr, IID_PPV_ARGS(&r)), "buffer");
  return r;
}

int main(int argc, char** argv) {
  if (argc != 5) {
    std::puts("usage: ags_layout_host <main32.dxil> <main64.dxil> <inputs.bin> <outputs.bin>");
    return 2;
  }
  const std::vector<char> cs32 = read_file(argv[1]), cs64 = read_file(argv[2]), in = read_file(argv[3]);
  const size_t in_size = 6 * 1024, out_size = 7 * 1024, io_size = in_size + out_size;
  if (cs32.empty() || cs64.empty() || in.size() != in_size) {
    std::puts("FAIL inputs");
    return 2;
  }

  ID3D12Device* dev = nullptr;
  need(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&dev)), "D3D12CreateDevice");

  D3D12_DESCRIPTOR_RANGE range = {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0x7FFF0ADE, 0};
  D3D12_ROOT_PARAMETER params[2] = {};
  params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
  params[0].DescriptorTable = {1, &range};
  params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
  params[1].Descriptor = {1, 0};
  D3D12_ROOT_SIGNATURE_DESC rsd = {2, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
  ID3DBlob *blob = nullptr, *err = nullptr;
  need(D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err), "serialize root signature");
  ID3D12RootSignature* rs = nullptr;
  need(dev->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&rs)),
       "root signature");

  D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
  pd.pRootSignature = rs;
  ID3D12PipelineState *pso32 = nullptr, *pso64 = nullptr;
  pd.CS = {cs32.data(), cs32.size()};
  need(dev->CreateComputePipelineState(&pd, IID_PPV_ARGS(&pso32)), "CreateComputePipelineState wave32");
  pd.CS = {cs64.data(), cs64.size()};
  need(dev->CreateComputePipelineState(&pd, IID_PPV_ARGS(&pso64)), "CreateComputePipelineState wave64");

  D3D12_DESCRIPTOR_HEAP_DESC hd = {D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1,
                                   D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
  ID3D12DescriptorHeap* heap = nullptr;
  need(dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap)), "descriptor heap");
  ID3D12Resource* channel = buffer(dev, D3D12_HEAP_TYPE_DEFAULT, 256, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                   D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
  D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
  ud.Format = DXGI_FORMAT_R32_TYPELESS;
  ud.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
  ud.Buffer.NumElements = 64;
  ud.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
  dev->CreateUnorderedAccessView(channel, nullptr, &ud, heap->GetCPUDescriptorHandleForHeapStart());

  ID3D12Resource* io = buffer(dev, D3D12_HEAP_TYPE_DEFAULT, io_size, D3D12_RESOURCE_STATE_COPY_DEST,
                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
  ID3D12Resource* up = buffer(dev, D3D12_HEAP_TYPE_UPLOAD, io_size, D3D12_RESOURCE_STATE_GENERIC_READ,
                              D3D12_RESOURCE_FLAG_NONE);
  ID3D12Resource* rb = buffer(dev, D3D12_HEAP_TYPE_READBACK, io_size, D3D12_RESOURCE_STATE_COPY_DEST,
                              D3D12_RESOURCE_FLAG_NONE);
  void* p = nullptr;
  need(up->Map(0, nullptr, &p), "map upload");
  std::memset(p, 0, io_size);
  std::memcpy(p, in.data(), in.size());
  up->Unmap(0, nullptr);

  D3D12_COMMAND_QUEUE_DESC qd = {D3D12_COMMAND_LIST_TYPE_DIRECT, 0, D3D12_COMMAND_QUEUE_FLAG_NONE, 0};
  ID3D12CommandQueue* q = nullptr;
  need(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&q)), "queue");
  ID3D12CommandAllocator* ca = nullptr;
  need(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&ca)), "allocator");
  ID3D12GraphicsCommandList* cl = nullptr;
  need(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, ca, pso32, IID_PPV_ARGS(&cl)), "command list");
  cl->CopyResource(io, up);
  D3D12_RESOURCE_BARRIER b = {};
  b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  b.Transition = {io, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COPY_DEST,
                  D3D12_RESOURCE_STATE_UNORDERED_ACCESS};
  cl->ResourceBarrier(1, &b);
  cl->SetComputeRootSignature(rs);
  cl->SetDescriptorHeaps(1, &heap);
  cl->SetComputeRootDescriptorTable(0, heap->GetGPUDescriptorHandleForHeapStart());
  cl->SetComputeRootUnorderedAccessView(1, io->GetGPUVirtualAddress());
  cl->Dispatch(1, 1, 1);
  cl->SetPipelineState(pso64);
  cl->Dispatch(1, 1, 1);
  b.Transition = {io, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                  D3D12_RESOURCE_STATE_COPY_SOURCE};
  cl->ResourceBarrier(1, &b);
  cl->CopyResource(rb, io);
  need(cl->Close(), "close");
  ID3D12CommandList* lists[] = {cl};
  q->ExecuteCommandLists(1, lists);
  ID3D12Fence* fence = nullptr;
  need(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "fence");
  need(q->Signal(fence, 1), "signal");
  HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
  need(fence->SetEventOnCompletion(1, ev), "event");
  if (WaitForSingleObject(ev, 10000) != WAIT_OBJECT_0) {
    std::puts("FAIL GPU timeout");
    return 3;
  }
  need(dev->GetDeviceRemovedReason(), "device removed");
  need(rb->Map(0, nullptr, &p), "map readback");
  FILE* f = std::fopen(argv[4], "wb");
  if (!f || std::fwrite(static_cast<char*>(p) + in_size, 1, out_size, f) != out_size) {
    std::puts("FAIL writing outputs");
    return 4;
  }
  std::fclose(f);
  std::puts("ags-layout outputs written");
  return 0;
}
