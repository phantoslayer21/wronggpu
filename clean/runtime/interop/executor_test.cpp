// SPDX-License-Identifier: MIT
// D3D12 test executor (see executor.h): stands in for the network in offline tests of the interop. On its own
// compute queue it waits for `ready`, writes a nearest-neighbour upscale of one converted input into the slot's
// `out`, and signals `done`. view 0: color times the exposure (so the exposure plumbing is visible); 1: depth (d, d,
// d); 2: motion in render pixels (mx, my, 0). WrongGPU's own code.
#include <cstring>
#include <vector>

#include "executor.h"

namespace wgi {

namespace {

const char kShader[] = R"HLSL(
cbuffer C : register(b0) {
  uint rw, rh, ow, oh, mode; float value; uint view, raw;
  uint cpitch, dpitch, mpitch, mch;  // raw inputs: row pitches in u32, motion FP16 channels
  float msx, msy, mox, moy;          // raw motion: m = ms * raw + mo
  uint opitch;                       // > 0: output as RGBA16F rows of opitch u32
};
// all R32_UINT typed UAVs (loads supported everywhere); floats through asfloat, halves through h2f / f2h
RWBuffer<uint> outB   : register(u0);
RWBuffer<uint> color  : register(u1);
RWBuffer<uint> expo   : register(u2);
RWBuffer<uint> depth  : register(u3);
RWBuffer<uint> motion : register(u4);
// FP16 <-> FP32 by bits (not every HLSL compiler has f16tof32 / f32tof16); round to nearest even
float h2f(uint h) {
  uint s = (h & 0x8000u) << 16, e = (h >> 10) & 31u, m = h & 1023u;
  if (e == 0u) return asfloat(s) + (s != 0u ? -1.0 : 1.0) * float(m) * (1.0 / 16777216.0);
  if (e == 31u) return asfloat(s | 0x7f800000u | (m << 13));
  return asfloat(s | ((e + 112u) << 23) | (m << 13));
}
uint f2h(float f) {
  uint x = asuint(f), s = (x >> 16) & 0x8000u, m = x & 0x7fffffu;
  int e = int((x >> 23) & 255u) - 112;
  if (e >= 31) return s | 0x7c00u;
  if (e <= 0) {
    if (e < -10) return s;
    m |= 0x800000u;
    uint sh = uint(14 - e), hm = m >> sh, rem = m & ((1u << sh) - 1u), hf = 1u << (sh - 1u);
    if (rem > hf || (rem == hf && (hm & 1u) != 0u)) hm++;
    return s | hm;
  }
  uint r = s | (uint(e) << 10) | (m >> 13), rem = m & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (r & 1u) != 0u)) r++;
  return r;
}
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
  if (id.x >= ow || id.y >= oh) return;
  uint sx = min(id.x * rw / ow, rw - 1), sy = min(id.y * rh / oh, rh - 1);
  float e = mode == 0 ? value : (mode == 1 ? asfloat(expo[0]) : 1.0);
  uint p = sy * rw + sx;
  float3 c;
  float d;
  float2 m;
  if (raw != 0) {
    uint c0 = color[sy * cpitch + 2 * sx], c1 = color[sy * cpitch + 2 * sx + 1];
    c = float3(h2f(c0 & 0xffffu), h2f(c0 >> 16), h2f(c1 & 0xffffu));
    d = asfloat(depth[sy * dpitch + sx]);
    uint mv = motion[sy * mpitch + (mch / 2) * sx];
    m = float2(msx * h2f(mv & 0xffffu) + mox, msy * h2f(mv >> 16) + moy);
  } else {
    c = float3(asfloat(color[4 * p]), asfloat(color[4 * p + 1]), asfloat(color[4 * p + 2]));
    d = asfloat(depth[p]);
    m = float2(asfloat(motion[2 * p]), asfloat(motion[2 * p + 1]));
  }
  c *= e;
  if (view == 1) c = d.xxx;
  if (view == 2) c = float3(m, 0.0);
  if (opitch != 0) {
    uint o = id.y * opitch + 2 * id.x;
    outB[o] = f2h(c.r) | (f2h(c.g) << 16);
    outB[o + 1] = f2h(c.b) | (f2h(1.0) << 16);
  } else {
    uint i = 3 * (id.y * ow + id.x);
    outB[i] = asuint(c.r); outB[i + 1] = asuint(c.g); outB[i + 2] = asuint(c.b);
  }
}
)HLSL";

typedef HRESULT(WINAPI* PfnD3DCompile)(LPCVOID, SIZE_T, LPCSTR, const void*, void*, LPCSTR, LPCSTR, UINT, UINT,
                                       ID3DBlob**, ID3DBlob**);
typedef HRESULT(WINAPI* PfnSerialize)(const D3D12_ROOT_SIGNATURE_DESC*, D3D_ROOT_SIGNATURE_VERSION, ID3DBlob**,
                                      ID3DBlob**);

template <class T>
void release(T*& p) {
  if (p) p->Release();
  p = nullptr;
}

constexpr UINT kSet = 5;  // UAVs: out, color, expo, depth, motion
constexpr UINT kConstants = 17;

class TestExecutor : public Executor {
 public:
  TestExecutor(int view, bool handles, bool keep) : view_((UINT)view), handles_(handles), keep_(handles && keep) {}
  ~TestExecutor() override {
    drain();
    for (auto& a : allocs_) release(a.allocator);
    release(list_);
    release(heap_);
    release(pso_);
    release(root_);
    release(fence_);
    release(queue_);
    if (event_) CloseHandle(event_);
  }

  const char* name() const override { return "d3d12-test"; }
  // ready_wait=memory: accepted so that the backend records its flag pass in tests; this executor still waits on the
  // fence (a D3D12 queue cannot wait on memory).
  bool use_ready_flag() override { return true; }
  bool needs_handles() const override { return handles_; }
  // keep: as an importer whose import stays with a pooled arena (a marker stands for the import; nothing to release)
  std::string import_scope() const override { return keep_ ? "d3d12-test" : std::string(); }

  bool init(ID3D12Device* device, const SharedSet& set, std::string* err) override {
    device_ = device;
    set_ = set;
    if (ArenaImport* k = set.slot[0].color.kept)
      if (!k->mem) k->mem = k;  // "imported" now; a later context finds it set
    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    if (FAILED(device->CreateCommandQueue(&qd, __uuidof(ID3D12CommandQueue), (void**)&queue_)) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence), (void**)&fence_)))
      return *err = "test executor: queue or fence creation failed", false;
    event_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);

    D3D12_DESCRIPTOR_RANGE ranges[1] = {};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[0].NumDescriptors = kSet;
    D3D12_ROOT_PARAMETER params[2] = {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants.Num32BitValues = kConstants;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable.NumDescriptorRanges = 1;
    params[1].DescriptorTable.pDescriptorRanges = ranges;
    D3D12_ROOT_SIGNATURE_DESC rd = {};
    rd.NumParameters = 2;
    rd.pParameters = params;
    HMODULE d3d12 = GetModuleHandleW(L"d3d12.dll");
    auto serialize = d3d12 ? reinterpret_cast<PfnSerialize>(
                                 reinterpret_cast<void*>(GetProcAddress(d3d12, "D3D12SerializeRootSignature")))
                           : nullptr;
    HMODULE dc = LoadLibraryW(L"d3dcompiler_47.dll");
    auto compile = dc ? reinterpret_cast<PfnD3DCompile>(reinterpret_cast<void*>(GetProcAddress(dc, "D3DCompile")))
                      : nullptr;
    if (!serialize || !compile) return *err = "test executor: d3d12 / d3dcompiler entry points missing", false;
    ID3DBlob *blob = nullptr, *msg = nullptr, *code = nullptr;
    HRESULT hr = serialize(&rd, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &msg);
    if (SUCCEEDED(hr))
      hr = device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                       __uuidof(ID3D12RootSignature), (void**)&root_);
    release(blob);
    release(msg);
    if (SUCCEEDED(hr))
      hr = compile(kShader, sizeof(kShader) - 1, "wronggpu_test_executor.hlsl", nullptr, nullptr, "main", "cs_5_0", 0,
                   0, &code, &msg);
    if (SUCCEEDED(hr)) {
      D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
      pd.pRootSignature = root_;
      pd.CS.pShaderBytecode = code->GetBufferPointer();
      pd.CS.BytecodeLength = code->GetBufferSize();
      hr = device->CreateComputePipelineState(&pd, __uuidof(ID3D12PipelineState), (void**)&pso_);
    }
    std::string detail;
    if (msg) detail.assign(static_cast<const char*>(msg->GetBufferPointer()), msg->GetBufferSize());
    release(code);
    release(msg);
    if (FAILED(hr)) return *err = "test executor: pipeline creation failed " + detail, false;
    D3D12_DESCRIPTOR_HEAP_DESC hd = {};
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    hd.NumDescriptors = 2 * kSet;
    hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (FAILED(device->CreateDescriptorHeap(&hd, __uuidof(ID3D12DescriptorHeap), (void**)&heap_)))
      return *err = "test executor: descriptor heap failed", false;
    inc_ = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    for (int s = 0; s < 2; ++s) {
      D3D12_CPU_DESCRIPTOR_HANDLE h = heap_->GetCPUDescriptorHandleForHeapStart();
      h.ptr += (SIZE_T)(kSet * s) * inc_;
      // u0 out, u1 color, u2 expo, u3 depth, u4 motion
      const SharedBuffer* bufs[kSet] = {&set.slot[s].out, &set.slot[s].color, &set.slot[s].expo, &set.slot[s].depth,
                                        &set.slot[s].motion};
      for (UINT i = 0; i < kSet; ++i) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC ud = {};
        ud.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        ud.Format = DXGI_FORMAT_R32_UINT;
        ud.Buffer.FirstElement = bufs[i]->offset / 4;
        ud.Buffer.NumElements = (UINT)(bufs[i]->bytes / 4);
        device->CreateUnorderedAccessView(bufs[i]->resource, nullptr, &ud, h);
        h.ptr += inc_;
      }
    }
    return true;
  }

  bool submit(const Job& job, bool* done_queued, std::string* err) override {
    *done_queued = false;
    ID3D12CommandAllocator* a = allocator(err);
    if (!a) return false;
    if (!list_) {
      if (FAILED(device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, a, nullptr,
                                            __uuidof(ID3D12GraphicsCommandList), (void**)&list_)))
        return *err = "test executor: CreateCommandList failed", false;
    } else if (FAILED(list_->Reset(a, nullptr))) {
      return *err = "test executor: Reset failed", false;
    }
    UINT cc[kConstants] = {(UINT)job.frame.rw, (UINT)job.frame.rh, set_.ow, set_.oh, (UINT)job.exposure_mode, 0,
                           view_, job.raw_inputs ? 1u : 0u, (UINT)job.color_pitch / 4, (UINT)job.depth_pitch / 4,
                           (UINT)job.motion_pitch / 4, (UINT)job.motion_channels, 0, 0, 0, 0,
                           (UINT)job.out_half_pitch / 4};
    std::memcpy(&cc[5], &job.frame.exposure, 4);
    std::memcpy(&cc[12], job.ms, 8);
    std::memcpy(&cc[14], job.mo, 8);
    D3D12_GPU_DESCRIPTOR_HANDLE g = heap_->GetGPUDescriptorHandleForHeapStart();
    g.ptr += (UINT64)(kSet * job.slot) * inc_;
    list_->SetDescriptorHeaps(1, &heap_);
    list_->SetComputeRootSignature(root_);
    list_->SetPipelineState(pso_);
    list_->SetComputeRoot32BitConstants(0, kConstants, cc, 0);
    list_->SetComputeRootDescriptorTable(1, g);
    list_->Dispatch((set_.ow + 7) / 8, (set_.oh + 7) / 8, 1);
    list_->Close();
    if (job.pipelined) queue_->Signal(set_.pipe.fence, job.value);  // queue order: after the previous frame
    queue_->Wait(set_.ready.fence, job.value);
    ID3D12CommandList* l = list_;
    queue_->ExecuteCommandLists(1, &l);
    queue_->Signal(set_.done.fence, job.value);
    queue_->Signal(fence_, ++fenceValue_);
    allocs_[current_].value = fenceValue_;
    *done_queued = true;
    return true;
  }

  void drain() override {
    if (!queue_ || !fence_) return;
    queue_->Signal(fence_, ++fenceValue_);
    if (fence_->GetCompletedValue() < fenceValue_ && event_) {
      fence_->SetEventOnCompletion(fenceValue_, event_);
      WaitForSingleObject(event_, 10000);
    }
  }

 private:
  // A free allocator (the test executor may wait on the CPU; it never runs in a game).
  ID3D12CommandAllocator* allocator(std::string* err) {
    for (size_t i = 0; i < allocs_.size(); ++i)
      if (fence_->GetCompletedValue() >= allocs_[i].value) {
        current_ = i;
        allocs_[i].allocator->Reset();
        return allocs_[i].allocator;
      }
    Alloc n;
    if (FAILED(device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, __uuidof(ID3D12CommandAllocator),
                                               (void**)&n.allocator))) {
      *err = "test executor: CreateCommandAllocator failed";
      return nullptr;
    }
    allocs_.push_back(n);
    current_ = allocs_.size() - 1;
    return n.allocator;
  }

  struct Alloc {
    ID3D12CommandAllocator* allocator = nullptr;
    uint64_t value = 0;
  };
  ID3D12Device* device_ = nullptr;
  SharedSet set_;
  ID3D12CommandQueue* queue_ = nullptr;
  ID3D12Fence* fence_ = nullptr;
  uint64_t fenceValue_ = 0;
  HANDLE event_ = nullptr;
  ID3D12RootSignature* root_ = nullptr;
  ID3D12PipelineState* pso_ = nullptr;
  ID3D12DescriptorHeap* heap_ = nullptr;
  UINT inc_ = 0;
  ID3D12GraphicsCommandList* list_ = nullptr;
  std::vector<Alloc> allocs_;
  size_t current_ = 0;
  UINT view_ = 0;
  bool handles_ = false;
  bool keep_ = false;
};

}  // namespace

std::unique_ptr<Executor> make_test_executor(int view, bool handles, bool keep) {
  return std::unique_ptr<Executor>(new TestExecutor(view, handles, keep));
}

}  // namespace wgi
