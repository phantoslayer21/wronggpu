// SPDX-License-Identifier: MIT
// Exhaustive GELU + FP8 table tool (clean/dx12/gelu_lut/run.py drives it).
//   host oracle <oracle.co> <rle.bin>          HIP: run-length form of q8(GELU(u)) over all 2^32 bit patterns u
//   host check  <oracle.co> <rle.bin>          HIP: recompute all 2^32 bytes, compare with rle.bin (must be 0)
//   host dx12   <verify.dxil> <rle.bin> [table.bin]
//                                              D3D12: the port under test against rle.bin over all 2^32 inputs
// rle.bin: u32 n, u32 pos[n] (ascending, pos[0] = 0), u32 byte[n]: byte(u) = byte[j] for pos[j] <= u < pos[j + 1].
// The D3D12 kernel follows the runtime's kernel convention (clean/dx12/README.md): one buffer as root UAVs u0 and u1,
// root constants b0 = (group base, grid, base, pos, byt, n, res, tab), the driver-channel table at u0 space 0x7FFF0ADE.
// Result block res: [0] differences, [1] differences whose input is an FP32 denormal, [2] differences on NaN inputs,
// then 30 examples (u, got | expected << 8).
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <hip/hip_runtime_api.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
using Microsoft::WRL::ComPtr;
static void ck(HRESULT h, const char* s) { if (FAILED(h)) { fprintf(stderr, "D3D12 %s: %08x\n", s, (unsigned)h); exit(2); } }
static void hc(hipError_t h, const char* s) { if (h != hipSuccess) { fprintf(stderr, "HIP %s: %s\n", s, hipGetErrorString(h)); exit(3); } }
#define CK(x) ck((x), #x)
#define HC(x) hc((x), #x)
static std::vector<char> slurp(const char* p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) { fprintf(stderr, "cannot read %s\n", p); exit(1); }
    return std::vector<char>((std::istreambuf_iterator<char>(f)), {});
}
static const uint32_t CHUNK_LOG = 26, CHUNKS = 1u << (32 - CHUNK_LOG), GROUPS = (1u << CHUNK_LOG) / 8 / 256;

struct Rle { std::vector<uint32_t> pos, byt; };
static Rle read_rle(const char* p) {
    auto raw = slurp(p); uint32_t n; memcpy(&n, raw.data(), 4);
    Rle r; r.pos.resize(n); r.byt.resize(n);
    memcpy(r.pos.data(), raw.data() + 4, 4ull * n); memcpy(r.byt.data(), raw.data() + 4 + 4ull * n, 4ull * n);
    return r;
}

static void print_res(const uint32_t* res) {
    printf("differences %u (denormal inputs %u, NaN inputs %u)\n", res[0], res[1], res[2]);
    for (uint32_t k = 0; k < std::min(res[0], 30u); ++k)
        printf("  u=%08x got=%02x expected=%02x\n", res[4 + 2 * k], res[5 + 2 * k] & 0xff, (res[5 + 2 * k] >> 8) & 0xff);
}

static int hip_oracle(const char* co, const char* out) {
    HC(hipInit(0)); HC(hipSetDevice(0));
    hipModule_t mod; hipFunction_t fn; HC(hipModuleLoad(&mod, co)); HC(hipModuleGetFunction(&fn, mod, "gelu_rle"));
    const uint32_t cap = 1u << 24;
    void *rec, *cnt; HC(hipMalloc(&rec, 8ull * cap)); HC(hipMalloc(&cnt, 4)); HC(hipMemset(cnt, 0, 4));
    for (uint32_t c = 0; c < CHUNKS; ++c) {
        uint32_t base = c << CHUNK_LOG; void* args[] = {&base, &rec, &cnt, (void*)&cap};
        HC(hipModuleLaunchKernel(fn, GROUPS, 1, 1, 256, 1, 1, 0, nullptr, args, nullptr));
    }
    HC(hipDeviceSynchronize());
    uint32_t n; HC(hipMemcpy(&n, cnt, 4, hipMemcpyDeviceToHost));
    if (n > cap) { fprintf(stderr, "%u records, capacity %u\n", n, cap); return 4; }
    std::vector<uint64_t> r(n); HC(hipMemcpy(r.data(), rec, 8ull * n, hipMemcpyDeviceToHost));
    std::sort(r.begin(), r.end());
    std::ofstream f(out, std::ios::binary); f.write((const char*)&n, 4);
    for (auto x : r) { uint32_t p = (uint32_t)(x >> 8); f.write((const char*)&p, 4); }
    for (auto x : r) { uint32_t b = (uint32_t)(x & 0xff); f.write((const char*)&b, 4); }
    printf("oracle: %u runs\n", n);
    return 0;
}

static int hip_check(const char* co, const char* rle_path) {
    Rle r = read_rle(rle_path); uint32_t n = (uint32_t)r.pos.size();
    HC(hipInit(0)); HC(hipSetDevice(0));
    hipModule_t mod; hipFunction_t fn; HC(hipModuleLoad(&mod, co)); HC(hipModuleGetFunction(&fn, mod, "gelu_check"));
    void *pos, *byt, *res; HC(hipMalloc(&pos, 4ull * n)); HC(hipMalloc(&byt, 4ull * n)); HC(hipMalloc(&res, 256));
    HC(hipMemcpy(pos, r.pos.data(), 4ull * n, hipMemcpyHostToDevice)); HC(hipMemcpy(byt, r.byt.data(), 4ull * n, hipMemcpyHostToDevice));
    HC(hipMemset(res, 0, 256));
    for (uint32_t c = 0; c < CHUNKS; ++c) {
        uint32_t base = c << CHUNK_LOG; void* args[] = {&base, &pos, &byt, &n, &res};
        HC(hipModuleLaunchKernel(fn, GROUPS, 1, 1, 256, 1, 1, 0, nullptr, args, nullptr));
    }
    HC(hipDeviceSynchronize());
    uint32_t h[64]; HC(hipMemcpy(h, res, 256, hipMemcpyDeviceToHost));
    printf("hip oracle vs rle: "); print_res(h);
    return h[0] ? 5 : 0;
}

static ComPtr<ID3D12Resource> buf(ID3D12Device* d, UINT64 size, D3D12_HEAP_TYPE ht, D3D12_RESOURCE_FLAGS fl, D3D12_RESOURCE_STATES st) {
    D3D12_HEAP_PROPERTIES hp{}; hp.Type = ht; D3D12_RESOURCE_DESC rd{}; rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; rd.Width = size; rd.Height = 1;
    rd.DepthOrArraySize = 1; rd.MipLevels = 1; rd.SampleDesc.Count = 1; rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; rd.Flags = fl;
    ComPtr<ID3D12Resource> r; CK(d->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, st, nullptr, IID_PPV_ARGS(&r))); return r;
}

static int dx12_verify(const char* dxil, const char* rle_path, const char* tab_path) {
    Rle r = read_rle(rle_path); const uint32_t n = (uint32_t)r.pos.size();
    std::vector<char> tab; if (tab_path) tab = slurp(tab_path);
    auto code = slurp(dxil);
    ComPtr<IDXGIFactory6> fac; CK(CreateDXGIFactory1(IID_PPV_ARGS(&fac))); ComPtr<IDXGIAdapter1> a;
    for (UINT i = 0;; i++) { ComPtr<IDXGIAdapter1> x; if (fac->EnumAdapters1(i, &x) == DXGI_ERROR_NOT_FOUND) break; DXGI_ADAPTER_DESC1 ad; x->GetDesc1(&ad);
        if (ad.VendorId == 0x1002 && !(ad.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) { a = x; break; } }
    ComPtr<ID3D12Device> dev; CK(D3D12CreateDevice(a.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&dev)));
    D3D12_COMMAND_QUEUE_DESC qd{}; qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT; ComPtr<ID3D12CommandQueue> q; CK(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&q)));
    ComPtr<ID3D12CommandAllocator> al; CK(dev->CreateCommandAllocator(qd.Type, IID_PPV_ARGS(&al)));
    ComPtr<ID3D12GraphicsCommandList> l; CK(dev->CreateCommandList(0, qd.Type, al.Get(), nullptr, IID_PPV_ARGS(&l)));
    D3D12_ROOT_PARAMETER rp[4]{};
    rp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV; rp[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV; rp[1].Descriptor.ShaderRegister = 1;
    rp[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS; rp[2].Constants = {0, 0, 8};
    D3D12_DESCRIPTOR_RANGE ags{}; ags.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV; ags.NumDescriptors = 1; ags.RegisterSpace = 0x7FFF0ADE;
    rp[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; rp[3].DescriptorTable = {1, &ags};
    D3D12_ROOT_SIGNATURE_DESC rsd{}; rsd.NumParameters = 4; rsd.pParameters = rp; ComPtr<ID3DBlob> blob, err;
    CK(D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err)); ComPtr<ID3D12RootSignature> rs;
    CK(dev->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&rs)));
    D3D12_COMPUTE_PIPELINE_STATE_DESC pd{}; pd.pRootSignature = rs.Get(); pd.CS = {code.data(), code.size()}; ComPtr<ID3D12PipelineState> pso;
    CK(dev->CreateComputePipelineState(&pd, IID_PPV_ARGS(&pso)));
    // arena: [256 null header | table at the runtime's fixed offset 256 (65536) | pos | byt | res 256]
    const UINT64 pad = 256, o_tab = pad, o_pos = o_tab + 65536, o_byt = o_pos + 4ull * n, o_res = o_byt + 4ull * n;
    const UINT64 total = o_res + 256;
    if (tab.size() > 65536) { fprintf(stderr, "table larger than 64 KB\n"); return 1; }
    std::vector<char> init(total - pad, 0);
    memcpy(init.data() + (o_pos - pad), r.pos.data(), 4ull * n); memcpy(init.data() + (o_byt - pad), r.byt.data(), 4ull * n);
    if (!tab.empty()) memcpy(init.data() + (o_tab - pad), tab.data(), tab.size());
    auto up = buf(dev.Get(), init.size(), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
    void* m; D3D12_RANGE none{0, 0}; CK(up->Map(0, &none, &m)); memcpy(m, init.data(), init.size()); up->Unmap(0, nullptr);
    auto mem = buf(dev.Get(), total, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
    auto agsb = buf(dev.Get(), 256, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
    auto rb = buf(dev.Get(), 256, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    D3D12_DESCRIPTOR_HEAP_DESC hd{}; hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; hd.NumDescriptors = 1; hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ComPtr<ID3D12DescriptorHeap> heap; CK(dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap)));
    D3D12_UNORDERED_ACCESS_VIEW_DESC ud{}; ud.Format = DXGI_FORMAT_R32_TYPELESS; ud.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    ud.Buffer.NumElements = 64; ud.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
    dev->CreateUnorderedAccessView(agsb.Get(), nullptr, &ud, heap->GetCPUDescriptorHandleForHeapStart());
    auto tr = [](ID3D12Resource* x, D3D12_RESOURCE_STATES s0, D3D12_RESOURCE_STATES s1) { D3D12_RESOURCE_BARRIER b{}; b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; b.Transition = {x, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, s0, s1}; return b; };
    ComPtr<ID3D12Fence> f; CK(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&f))); HANDLE ev = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    UINT64 fv = 0;
    auto submit = [&]() {
        CK(l->Close()); ID3D12CommandList* ll = l.Get(); q->ExecuteCommandLists(1, &ll);
        CK(q->Signal(f.Get(), ++fv)); CK(f->SetEventOnCompletion(fv, ev));
        if (WaitForSingleObject(ev, 60000) != WAIT_OBJECT_0) { fprintf(stderr, "timeout\n"); exit(4); }
        CK(dev->GetDeviceRemovedReason()); CK(al->Reset()); CK(l->Reset(al.Get(), nullptr));
    };
    D3D12_RESOURCE_BARRIER b = tr(mem.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST); l->ResourceBarrier(1, &b);
    l->CopyBufferRegion(mem.Get(), pad, up.Get(), 0, init.size());
    b = tr(mem.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS); l->ResourceBarrier(1, &b);
    submit();
    LARGE_INTEGER t0, t1, fq; QueryPerformanceFrequency(&fq); QueryPerformanceCounter(&t0);
    for (uint32_t c = 0; c < CHUNKS; ++c) {
        ID3D12DescriptorHeap* hs[] = {heap.Get()}; l->SetDescriptorHeaps(1, hs);
        l->SetPipelineState(pso.Get()); l->SetComputeRootSignature(rs.Get());
        l->SetComputeRootUnorderedAccessView(0, mem->GetGPUVirtualAddress()); l->SetComputeRootUnorderedAccessView(1, mem->GetGPUVirtualAddress());
        const uint32_t k[8] = {0, GROUPS, c << CHUNK_LOG, (uint32_t)o_pos, (uint32_t)o_byt, n, (uint32_t)o_res, (uint32_t)o_tab};
        l->SetComputeRoot32BitConstants(2, 8, k, 0); l->SetComputeRootDescriptorTable(3, heap->GetGPUDescriptorHandleForHeapStart());
        l->Dispatch(GROUPS, 1, 1);
        if ((c & 7) == 7) submit();
    }
    QueryPerformanceCounter(&t1);
    b = tr(mem.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE); l->ResourceBarrier(1, &b);
    l->CopyBufferRegion(rb.Get(), 0, mem.Get(), o_res, 256); submit();
    uint32_t res[64]; D3D12_RANGE rr{0, 256}; CK(rb->Map(0, &rr, &m)); memcpy(res, m, 256); rb->Unmap(0, &none);
    printf("d3d12 port vs rle (%u runs, %.2f s): ", n, double(t1.QuadPart - t0.QuadPart) / fq.QuadPart); print_res(res);
    return res[0] ? 5 : 0;
}

int main(int argc, char** argv) {
    if (argc >= 4 && !strcmp(argv[1], "oracle")) return hip_oracle(argv[2], argv[3]);
    if (argc >= 4 && !strcmp(argv[1], "check")) return hip_check(argv[2], argv[3]);
    if (argc >= 4 && !strcmp(argv[1], "dx12")) return dx12_verify(argv[2], argv[3], argc >= 5 ? argv[4] : nullptr);
    fprintf(stderr, "usage: host oracle|check <oracle.co> <rle.bin> | host dx12 <verify.dxil> <rle.bin> [table.bin]\n");
    return 1;
}
