// SPDX-License-Identifier: MIT
// f16lut proof tool (clean/dx12/f16lut/run.py drives it).
//   host hip  <oracle.co> <params.bin> <hip.bin>     HIP oracle: per parameter set, out[65536] then row[65536] (u32)
//   host dx12 <verify.dxil> <params.bin> <hip.bin>   D3D12: per set, orig (post chain) and gen (table generator) for all
//                                                    65536 r; prints gen != orig and orig != HIP counts
// params.bin: u32 n, then n x (dec_scale bits, pi bits, rne16).
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <hip/hip_runtime_api.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
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
static std::vector<uint32_t> params(const char* p, uint32_t* n) {
    auto raw = slurp(p); memcpy(n, raw.data(), 4);
    std::vector<uint32_t> v(3ull * *n); memcpy(v.data(), raw.data() + 4, 12ull * *n); return v;
}
static const uint32_t N = 65536;

static int hip_run(const char* co, const char* prm_path, const char* out) {
    uint32_t n; auto prm = params(prm_path, &n);
    HC(hipInit(0)); HC(hipSetDevice(0));
    hipModule_t mod; hipFunction_t fn; HC(hipModuleLoad(&mod, co)); HC(hipModuleGetFunction(&fn, mod, "f16lut_oracle"));
    void *dp, *dout, *drow;
    HC(hipMalloc(&dp, 12ull * n)); HC(hipMalloc(&dout, 4ull * N * n)); HC(hipMalloc(&drow, 4ull * N * n));
    HC(hipMemcpy(dp, prm.data(), 12ull * n, hipMemcpyHostToDevice));
    void* args[] = {&dp, &dout, &drow};
    HC(hipModuleLaunchKernel(fn, N / 256, n, 1, 256, 1, 1, 0, nullptr, args, nullptr));
    HC(hipDeviceSynchronize());
    std::vector<uint32_t> a(1ull * N * n), b(1ull * N * n);
    HC(hipMemcpy(a.data(), dout, 4ull * N * n, hipMemcpyDeviceToHost)); HC(hipMemcpy(b.data(), drow, 4ull * N * n, hipMemcpyDeviceToHost));
    std::ofstream f(out, std::ios::binary);
    for (uint32_t s = 0; s < n; ++s) { f.write((const char*)(a.data() + 1ull * N * s), 4ull * N); f.write((const char*)(b.data() + 1ull * N * s), 4ull * N); }
    printf("hip oracle: %u sets\n", n);
    return 0;
}

static ComPtr<ID3D12Resource> buf(ID3D12Device* d, UINT64 size, D3D12_HEAP_TYPE ht, D3D12_RESOURCE_FLAGS fl, D3D12_RESOURCE_STATES st) {
    D3D12_HEAP_PROPERTIES hp{}; hp.Type = ht; D3D12_RESOURCE_DESC rd{}; rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; rd.Width = size; rd.Height = 1;
    rd.DepthOrArraySize = 1; rd.MipLevels = 1; rd.SampleDesc.Count = 1; rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; rd.Flags = fl;
    ComPtr<ID3D12Resource> r; CK(d->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, st, nullptr, IID_PPV_ARGS(&r))); return r;
}
static bool denorm(float x) { return x != 0.0f && std::fabs(x) < 1.17549435e-38f; }

static int dx12_run(const char* dxil, const char* prm_path, const char* hip_path) {
    uint32_t n; auto prm = params(prm_path, &n);
    std::ifstream hf(hip_path, std::ios::binary);
    if (!hf) { fprintf(stderr, "cannot read %s\n", hip_path); return 1; }
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
    const UINT64 o_out = 256, bytes = 8ull * N, total = o_out + bytes;
    auto mem = buf(dev.Get(), total, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    auto agsb = buf(dev.Get(), 256, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
    auto rb = buf(dev.Get(), bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
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
    std::vector<uint32_t> hip(2ull * N), got(2ull * N);
    uint64_t tot_gen = 0, tot_hip = 0, tot_hip_dn = 0, values = 0;
    for (uint32_t s = 0; s < n; ++s) {
        hf.read((char*)hip.data(), 8ull * N);
        if (!hf) { fprintf(stderr, "short %s\n", hip_path); return 1; }
        ID3D12DescriptorHeap* hs[] = {heap.Get()}; l->SetDescriptorHeaps(1, hs);
        l->SetPipelineState(pso.Get()); l->SetComputeRootSignature(rs.Get());
        l->SetComputeRootUnorderedAccessView(0, mem->GetGPUVirtualAddress()); l->SetComputeRootUnorderedAccessView(1, mem->GetGPUVirtualAddress());
        const uint32_t k[8] = {0, N / 256, (uint32_t)o_out, prm[3 * s], prm[3 * s + 1], prm[3 * s + 2], 0, 0};
        l->SetComputeRoot32BitConstants(2, 8, k, 0); l->SetComputeRootDescriptorTable(3, heap->GetGPUDescriptorHandleForHeapStart());
        l->Dispatch(N / 256, 1, 1);
        D3D12_RESOURCE_BARRIER b = tr(mem.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE); l->ResourceBarrier(1, &b);
        l->CopyBufferRegion(rb.Get(), 0, mem.Get(), o_out, bytes);
        b = tr(mem.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS); l->ResourceBarrier(1, &b);
        submit();
        void* m; D3D12_RANGE rr{0, (SIZE_T)bytes}, none{0, 0}; CK(rb->Map(0, &rr, &m)); memcpy(got.data(), m, bytes); rb->Unmap(0, &none);
        uint32_t dg = 0, dh = 0, dhd = 0, ex = 0;
        float pi, dsc; memcpy(&pi, &prm[3 * s + 1], 4); memcpy(&dsc, &prm[3 * s], 4);
        const bool dn_in = denorm(pi) || denorm(dsc);   // an FP32-denormal parameter (D3D12 flushes it)
        for (uint32_t r = 0; r < N; ++r) {
            const uint32_t orig = got[r], gen = got[N + r], h = hip[r];
            if (gen != orig) {
                if (ex++ < 4) printf("  set %u r=%04x gen=%08x orig=%08x hip=%08x\n", s, r, gen, orig, h);
                ++dg;
            }
            if (orig != h) {
                float x0; memcpy(&x0, &hip[N + r], 4);
                float hf32; memcpy(&hf32, &h, 4);
                const bool dn = dn_in || denorm(x0) || denorm(x0 * pi) || denorm(hf32);
                ++dh; dhd += dn ? 1 : 0;
                if (!dn && ex++ < 4) printf("  set %u r=%04x orig=%08x hip=%08x (normal)\n", s, r, orig, h);
            }
        }
        tot_gen += dg; tot_hip += dh; tot_hip_dn += dhd; values += N;
        if (dg || dh) printf("set %u (dec %08x pi %08x rne16 %u): gen!=orig %u, orig!=hip %u (FP32-denormal intermediates %u)\n",
                             s, prm[3 * s], prm[3 * s + 1], prm[3 * s + 2], dg, dh, dhd);
    }
    printf("TOTAL sets %u values %llu: gen!=orig %llu; orig!=hip %llu, of them FP32-denormal intermediates %llu\n", n,
           (unsigned long long)values, (unsigned long long)tot_gen, (unsigned long long)tot_hip, (unsigned long long)tot_hip_dn);
    return tot_gen || tot_hip != tot_hip_dn ? 5 : 0;
}

int main(int argc, char** argv) {
    if (argc >= 5 && !strcmp(argv[1], "hip")) return hip_run(argv[2], argv[3], argv[4]);
    if (argc >= 5 && !strcmp(argv[1], "dx12")) return dx12_run(argv[2], argv[3], argv[4]);
    fprintf(stderr, "usage: host hip <oracle.co> <params.bin> <hip.bin> | host dx12 <verify.dxil> <params.bin> <hip.bin>\n");
    return 1;
}
