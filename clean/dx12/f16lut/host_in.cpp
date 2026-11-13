// SPDX-License-Identifier: MIT
// f16lut encoder colour-table proof tool (clean/dx12/f16lut/run_in.py drives it; host.cpp's structure).
//   host_in hip  <oracle_in.co> <params.bin> <hip.bin>     HIP oracle: per parameter set, (v, gc)[65536] then c / pi [65536]
//   host_in dx12 <verify_in.dxil> <params.bin> <hip.bin>   D3D12: per set, orig (the encoder's chain) and gen (the table
//                                                          entry) for all 65536 FP16 patterns; gen != orig, orig != HIP
//   host_in rt   <verify_in.dxil>                          D3D12: f16tof32 / f32tof16 round trip of all 65536 patterns
//                                                          against the exact values, and f32tof16 of every FP32 NaN
// params.bin: u32 n, then n x (e0 bits, pi bits).
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
    std::vector<uint32_t> v(2ull * *n); memcpy(v.data(), raw.data() + 4, 8ull * *n); return v;
}
static const uint32_t N = 65536;

static int hip_run(const char* co, const char* prm_path, const char* out) {
    uint32_t n; auto prm = params(prm_path, &n);
    HC(hipInit(0)); HC(hipSetDevice(0));
    hipModule_t mod; hipFunction_t fn; HC(hipModuleLoad(&mod, co)); HC(hipModuleGetFunction(&fn, mod, "f16lut_in_oracle"));
    void *dp, *dout, *drow;
    HC(hipMalloc(&dp, 8ull * n)); HC(hipMalloc(&dout, 8ull * N * n)); HC(hipMalloc(&drow, 4ull * N * n));
    HC(hipMemcpy(dp, prm.data(), 8ull * n, hipMemcpyHostToDevice));
    void* args[] = {&dp, &dout, &drow};
    HC(hipModuleLaunchKernel(fn, N / 256, n, 1, 256, 1, 1, 0, nullptr, args, nullptr));
    HC(hipDeviceSynchronize());
    std::vector<uint32_t> a(2ull * N * n), b(1ull * N * n);
    HC(hipMemcpy(a.data(), dout, 8ull * N * n, hipMemcpyDeviceToHost)); HC(hipMemcpy(b.data(), drow, 4ull * N * n, hipMemcpyDeviceToHost));
    std::ofstream f(out, std::ios::binary);
    for (uint32_t s = 0; s < n; ++s) { f.write((const char*)(a.data() + 2ull * N * s), 8ull * N); f.write((const char*)(b.data() + 1ull * N * s), 4ull * N); }
    printf("hip oracle: %u sets\n", n);
    return 0;
}

static ComPtr<ID3D12Resource> buf(ID3D12Device* d, UINT64 size, D3D12_HEAP_TYPE ht, D3D12_RESOURCE_FLAGS fl, D3D12_RESOURCE_STATES st) {
    D3D12_HEAP_PROPERTIES hp{}; hp.Type = ht; D3D12_RESOURCE_DESC rd{}; rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; rd.Width = size; rd.Height = 1;
    rd.DepthOrArraySize = 1; rd.MipLevels = 1; rd.SampleDesc.Count = 1; rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; rd.Flags = fl;
    ComPtr<ID3D12Resource> r; CK(d->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, st, nullptr, IID_PPV_ARGS(&r))); return r;
}
static bool denorm(float x) { return x != 0.0f && std::fabs(x) < 1.17549435e-38f; }

static uint32_t half_bits_to_f32(uint32_t h) {   // exact FP16 -> FP32 (host reference)
    const uint32_t s = (h & 0x8000u) << 16, e = (h >> 10) & 31u, m = h & 0x3ffu;
    if (e == 31u) return s | 0x7f800000u | (m << 13);
    if (e != 0u) return s | ((e + 112u) << 23) | (m << 13);
    if (m == 0u) return s;
    float v = std::ldexp((float)m, -24); uint32_t b; memcpy(&b, &v, 4); return s | b;
}

static int dx12_run(const char* dxil, const char* prm_path, const char* hip_path, bool rt) {
    uint32_t n = 1; std::vector<uint32_t> prm(2, 0u);
    std::ifstream hf;
    if (!rt) {
        prm = params(prm_path, &n);
        hf.open(hip_path, std::ios::binary);
        if (!hf) { fprintf(stderr, "cannot read %s\n", hip_path); return 1; }
    }
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
    const UINT64 o_out = 256, bytes = 16ull * N, total = o_out + bytes;
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
    if (rt) {
        ID3D12DescriptorHeap* hs[] = {heap.Get()}; l->SetDescriptorHeaps(1, hs);
        l->SetPipelineState(pso.Get()); l->SetComputeRootSignature(rs.Get());
        l->SetComputeRootUnorderedAccessView(0, mem->GetGPUVirtualAddress()); l->SetComputeRootUnorderedAccessView(1, mem->GetGPUVirtualAddress());
        const uint32_t k[8] = {0, N / 256, (uint32_t)o_out, 0, 0, 1, 0, 0};
        l->SetComputeRoot32BitConstants(2, 8, k, 0); l->SetComputeRootDescriptorTable(3, heap->GetGPUDescriptorHandleForHeapStart());
        l->Dispatch(N / 256, 1, 1);
        D3D12_RESOURCE_BARRIER b = tr(mem.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE); l->ResourceBarrier(1, &b);
        l->CopyBufferRegion(rb.Get(), 0, mem.Get(), o_out, 12ull * N);
        submit();
        std::vector<uint32_t> got(3ull * N);
        void* m; D3D12_RANGE rr{0, (SIZE_T)(12ull * N)}, none{0, 0}; CK(rb->Map(0, &rr, &m)); memcpy(got.data(), m, 12ull * N); rb->Unmap(0, &none);
        uint64_t bad_val = 0, bad_rt = 0, bad_nan = 0, nan_in = 0;
        for (uint32_t h = 0; h < N; ++h) {
            const bool hnan = ((h >> 10) & 31u) == 31u && (h & 0x3ffu) != 0u;
            const uint32_t x = got[h], back = got[N + h];
            if (hnan) {   // the value is a NaN; f32tof16 of it must be an FP16 NaN
                ++nan_in;
                const bool xn = (x & 0x7f800000u) == 0x7f800000u && (x & 0x7fffffu) != 0u;
                const bool bn = ((back >> 10) & 31u) == 31u && (back & 0x3ffu) != 0u && back < 0x10000u;
                if (!xn || !bn) { if (bad_val++ < 4) printf("  NaN h=%04x x=%08x back=%08x\n", h, x, back); }
                continue;
            }
            if (x != half_bits_to_f32(h)) { if (bad_val++ < 4) printf("  h=%04x f16tof32=%08x exact=%08x\n", h, x, half_bits_to_f32(h)); }
            if (back != h) { if (bad_rt++ < 4) printf("  h=%04x f32tof16(f16tof32)=%08x\n", h, back); }
        }
        for (uint32_t i = 0; i < N; ++i) bad_nan += got[2ull * N + i];
        printf("RT values %u: f16tof32 != exact %llu; round trip != h %llu (FP16 NaN patterns %llu); FP32 NaNs -> non-NaN FP16 %llu of %u\n",
               N, (unsigned long long)bad_val, (unsigned long long)bad_rt, (unsigned long long)nan_in, (unsigned long long)bad_nan,
               (1u << 24) - 2u);
        return bad_val || bad_rt || bad_nan ? 5 : 0;
    }
    std::vector<uint32_t> hip(3ull * N), got(4ull * N);
    uint64_t tot_gen = 0, tot_hip = 0, tot_hip_dn = 0, values = 0;
    for (uint32_t s = 0; s < n; ++s) {
        hf.read((char*)hip.data(), 12ull * N);
        if (!hf) { fprintf(stderr, "short %s\n", hip_path); return 1; }
        ID3D12DescriptorHeap* hs[] = {heap.Get()}; l->SetDescriptorHeaps(1, hs);
        l->SetPipelineState(pso.Get()); l->SetComputeRootSignature(rs.Get());
        l->SetComputeRootUnorderedAccessView(0, mem->GetGPUVirtualAddress()); l->SetComputeRootUnorderedAccessView(1, mem->GetGPUVirtualAddress());
        const uint32_t k[8] = {0, N / 256, (uint32_t)o_out, prm[2 * s], prm[2 * s + 1], 0, 0, 0};
        l->SetComputeRoot32BitConstants(2, 8, k, 0); l->SetComputeRootDescriptorTable(3, heap->GetGPUDescriptorHandleForHeapStart());
        l->Dispatch(N / 256, 1, 1);
        D3D12_RESOURCE_BARRIER b = tr(mem.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE); l->ResourceBarrier(1, &b);
        l->CopyBufferRegion(rb.Get(), 0, mem.Get(), o_out, bytes);
        b = tr(mem.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS); l->ResourceBarrier(1, &b);
        submit();
        void* m; D3D12_RANGE rr{0, (SIZE_T)bytes}, none{0, 0}; CK(rb->Map(0, &rr, &m)); memcpy(got.data(), m, bytes); rb->Unmap(0, &none);
        uint32_t dg = 0, dh = 0, dhd = 0, ex = 0;
        float pi, e0; memcpy(&pi, &prm[2 * s + 1], 4); memcpy(&e0, &prm[2 * s], 4);
        const float e = e0 == 0.0f ? 1.0f : e0;
        const bool dn_in = denorm(pi) || denorm(e);   // an FP32-denormal parameter (D3D12 flushes it)
        for (uint32_t r = 0; r < N; ++r) {
            const uint32_t ov = got[2 * r], og = got[2 * r + 1], gv = got[2 * N + 2 * r], gg = got[2 * N + 2 * r + 1];
            const uint32_t hv = hip[2 * r], hg = hip[2 * r + 1];
            if (gv != ov || gg != og) {
                if (ex++ < 4) printf("  set %u c=%04x gen=%08x,%08x orig=%08x,%08x hip=%08x,%08x\n", s, r, gv, gg, ov, og, hv, hg);
                ++dg;
            }
            if (ov != hv || og != hg) {
                float c; memcpy(&c, &hip[2 * N + r], 4);   // HIP's c / pi
                const bool dn = dn_in || denorm(c) || denorm(e * c);
                ++dh; dhd += dn ? 1 : 0;
                if (!dn && ex++ < 4) printf("  set %u c=%04x orig=%08x,%08x hip=%08x,%08x (normal)\n", s, r, ov, og, hv, hg);
            }
        }
        tot_gen += dg; tot_hip += dh; tot_hip_dn += dhd; values += N;
        if (dg || dh) printf("set %u (e0 %08x pi %08x): gen!=orig %u, orig!=hip %u (FP32-denormal intermediates %u)\n",
                             s, prm[2 * s], prm[2 * s + 1], dg, dh, dhd);
    }
    printf("TOTAL sets %u values %llu: gen!=orig %llu; orig!=hip %llu, of them FP32-denormal intermediates %llu\n", n,
           (unsigned long long)values, (unsigned long long)tot_gen, (unsigned long long)tot_hip, (unsigned long long)tot_hip_dn);
    return tot_gen || tot_hip != tot_hip_dn ? 5 : 0;
}

int main(int argc, char** argv) {
    if (argc >= 5 && !strcmp(argv[1], "hip")) return hip_run(argv[2], argv[3], argv[4]);
    if (argc >= 5 && !strcmp(argv[1], "dx12")) return dx12_run(argv[2], argv[3], argv[4], false);
    if (argc >= 3 && !strcmp(argv[1], "rt")) return dx12_run(argv[2], nullptr, nullptr, true);
    fprintf(stderr, "usage: host_in hip <oracle_in.co> <params.bin> <hip.bin> | dx12 <verify_in.dxil> <params.bin> <hip.bin> | rt <verify_in.dxil>\n");
    return 1;
}
