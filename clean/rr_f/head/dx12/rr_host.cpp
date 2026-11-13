// SPDX-License-Identifier: MIT
// D3D12 test host of the D0-R kernel (rr_d0r.hlsl): runs one compute kernel on one arena buffer described by a job
// file, writes the requested regions back, optionally times it. Driven by run_dx12.py (which builds this file).
//
//   rr_host <kernel.dxil> <job.bin> <out.bin>
//
// job.bin (little-endian u32 words): 'RRJ1', arena bytes, n constants, constants[n] (the kernel's argument DWORDs),
// workgroups, timing repetitions (0 = one run), warm-up milliseconds, n init regions, per region: offset, bytes,
// data (padded to 4 bytes), n read regions, per region: offset, bytes. The arena starts zeroed; init regions are
// copied in; one run (or warm-up + timed runs) is dispatched (groups > 65535 split into dispatches with a group base);
// the read regions are written to out.bin back to back. Timing prints one line
//   TIME reps <n> median_us <x> min_us <x> mean_us <x>
// measured by GPU timestamps around each dispatch (the run's dispatches separated by UAV barriers).
// Root signature (the runtime's convention, clean/dx12/README.md): u0 / u1 root UAVs = the arena, b0 = 2 runtime
// constants (group base, workgroup count) + the arguments, and the driver-intrinsic UAV (space 0x7FFF0ADE).
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>
using Microsoft::WRL::ComPtr;

static void ck(HRESULT h, const char* s)
{
    if (FAILED(h)) { fprintf(stderr, "D3D12 %s: %08x\n", s, (unsigned)h); exit(2); }
}
#define CK(x) ck((x), #x)

static std::vector<uint32_t> slurp_words(const char* p)
{
    std::ifstream f(p, std::ios::binary);
    if (!f) { fprintf(stderr, "cannot read %s\n", p); exit(1); }
    std::vector<char> b((std::istreambuf_iterator<char>(f)), {});
    std::vector<uint32_t> w((b.size() + 3) / 4, 0u);
    memcpy(w.data(), b.data(), b.size());
    return w;
}
static std::vector<char> slurp(const char* p)
{
    std::ifstream f(p, std::ios::binary);
    if (!f) { fprintf(stderr, "cannot read %s\n", p); exit(1); }
    return std::vector<char>((std::istreambuf_iterator<char>(f)), {});
}

static ComPtr<ID3D12Resource> buf(ID3D12Device* d, UINT64 size, D3D12_HEAP_TYPE ht, D3D12_RESOURCE_FLAGS fl,
                                  D3D12_RESOURCE_STATES st)
{
    D3D12_HEAP_PROPERTIES hp{}; hp.Type = ht;
    D3D12_RESOURCE_DESC rd{}; rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; rd.Width = size; rd.Height = 1;
    rd.DepthOrArraySize = 1; rd.MipLevels = 1; rd.SampleDesc.Count = 1; rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags = fl;
    ComPtr<ID3D12Resource> r;
    CK(d->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, st, nullptr, IID_PPV_ARGS(&r)));
    return r;
}

int main(int argc, char** argv)
{
    if (argc < 4) { fprintf(stderr, "usage: rr_host <kernel.dxil> <job.bin> <out.bin>\n"); return 1; }
    const auto code = slurp(argv[1]);
    const auto job = slurp_words(argv[2]);
    size_t p = 0;
    auto next = [&]() { if (p >= job.size()) { fprintf(stderr, "short job\n"); exit(1); } return job[p++]; };
    if (next() != 0x314a5252u) { fprintf(stderr, "bad job magic\n"); return 1; }
    const uint64_t arena = ((uint64_t)next() + 255u) & ~255ull;
    const uint32_t nconst = next();
    std::vector<uint32_t> consts(2 + nconst, 0u);
    for (uint32_t i = 0; i < nconst; ++i) consts[2 + i] = next();
    const uint32_t groups = next(), reps = next(), warm_ms = next();
    consts[1] = groups;
    struct Region { uint32_t off, bytes; size_t word; };
    std::vector<Region> init, rd;
    for (uint32_t n = next(), i = 0; i < n; ++i) {
        Region r; r.off = next(); r.bytes = next(); r.word = p; p += (r.bytes + 3) / 4; init.push_back(r);
        if (p > job.size() || (uint64_t)r.off + r.bytes > arena) { fprintf(stderr, "bad init region\n"); return 1; }
    }
    for (uint32_t n = next(), i = 0; i < n; ++i) {
        Region r; r.off = next(); r.bytes = next(); r.word = 0; rd.push_back(r);
        if ((uint64_t)r.off + r.bytes > arena) { fprintf(stderr, "bad read region\n"); return 1; }
    }

    ComPtr<IDXGIFactory6> fac; CK(CreateDXGIFactory1(IID_PPV_ARGS(&fac)));
    ComPtr<IDXGIAdapter1> ad;
    for (UINT i = 0;; i++) {
        ComPtr<IDXGIAdapter1> x;
        if (fac->EnumAdapters1(i, &x) == DXGI_ERROR_NOT_FOUND) break;
        DXGI_ADAPTER_DESC1 d; x->GetDesc1(&d);
        if (d.VendorId == 0x1002 && !(d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) { ad = x; break; }
    }
    if (!ad) { fprintf(stderr, "no AMD adapter\n"); return 3; }
    ComPtr<ID3D12Device> dev; CK(D3D12CreateDevice(ad.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&dev)));
    D3D12_COMMAND_QUEUE_DESC qd{}; qd.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    ComPtr<ID3D12CommandQueue> q; CK(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&q)));
    ComPtr<ID3D12CommandAllocator> al; CK(dev->CreateCommandAllocator(qd.Type, IID_PPV_ARGS(&al)));
    ComPtr<ID3D12GraphicsCommandList> l;
    CK(dev->CreateCommandList(0, qd.Type, al.Get(), nullptr, IID_PPV_ARGS(&l)));

    D3D12_ROOT_PARAMETER rp[4]{};
    rp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    rp[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV; rp[1].Descriptor.ShaderRegister = 1;
    rp[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS; rp[2].Constants = {0, 0, (UINT)consts.size()};
    D3D12_DESCRIPTOR_RANGE ags{}; ags.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV; ags.NumDescriptors = 1;
    ags.RegisterSpace = 0x7FFF0ADE;
    rp[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; rp[3].DescriptorTable = {1, &ags};
    D3D12_ROOT_SIGNATURE_DESC rsd{}; rsd.NumParameters = 4; rsd.pParameters = rp;
    ComPtr<ID3DBlob> blob, err;
    CK(D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err));
    ComPtr<ID3D12RootSignature> rs;
    CK(dev->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&rs)));
    D3D12_COMPUTE_PIPELINE_STATE_DESC pd{}; pd.pRootSignature = rs.Get(); pd.CS = {code.data(), code.size()};
    ComPtr<ID3D12PipelineState> pso; CK(dev->CreateComputePipelineState(&pd, IID_PPV_ARGS(&pso)));

    auto mem = buf(dev.Get(), arena, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                   D3D12_RESOURCE_STATE_COPY_DEST);
    auto agsb = buf(dev.Get(), 256, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COMMON);
    D3D12_DESCRIPTOR_HEAP_DESC hd{}; hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; hd.NumDescriptors = 1;
    hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ComPtr<ID3D12DescriptorHeap> heap; CK(dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap)));
    D3D12_UNORDERED_ACCESS_VIEW_DESC ud{}; ud.Format = DXGI_FORMAT_R32_TYPELESS; ud.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    ud.Buffer.NumElements = 64; ud.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
    dev->CreateUnorderedAccessView(agsb.Get(), nullptr, &ud, heap->GetCPUDescriptorHandleForHeapStart());

    ComPtr<ID3D12Fence> f; CK(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&f)));
    HANDLE ev = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    UINT64 fv = 0;
    auto submit = [&]() {
        CK(l->Close()); ID3D12CommandList* ll = l.Get(); q->ExecuteCommandLists(1, &ll);
        CK(q->Signal(f.Get(), ++fv)); CK(f->SetEventOnCompletion(fv, ev));
        if (WaitForSingleObject(ev, 120000) != WAIT_OBJECT_0) { fprintf(stderr, "timeout\n"); exit(4); }
        CK(dev->GetDeviceRemovedReason()); CK(al->Reset()); CK(l->Reset(al.Get(), nullptr));
    };
    auto tr = [](ID3D12Resource* x, D3D12_RESOURCE_STATES s0, D3D12_RESOURCE_STATES s1) {
        D3D12_RESOURCE_BARRIER b{}; b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition = {x, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, s0, s1}; return b;
    };

    // upload: zeroed arena image with the init regions
    {
        auto up = buf(dev.Get(), arena, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
        void* m; D3D12_RANGE none{0, 0}; CK(up->Map(0, &none, &m));
        memset(m, 0, (size_t)arena);
        for (const auto& r : init) memcpy((char*)m + r.off, job.data() + r.word, r.bytes);
        up->Unmap(0, nullptr);
        l->CopyBufferRegion(mem.Get(), 0, up.Get(), 0, arena);
        D3D12_RESOURCE_BARRIER b = tr(mem.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        l->ResourceBarrier(1, &b);
        submit();
    }

    auto bind = [&]() {
        ID3D12DescriptorHeap* hs[] = {heap.Get()}; l->SetDescriptorHeaps(1, hs);
        l->SetPipelineState(pso.Get()); l->SetComputeRootSignature(rs.Get());
        l->SetComputeRootUnorderedAccessView(0, mem->GetGPUVirtualAddress());
        l->SetComputeRootUnorderedAccessView(1, mem->GetGPUVirtualAddress());
        l->SetComputeRoot32BitConstants(2, (UINT)consts.size(), consts.data(), 0);
        l->SetComputeRootDescriptorTable(3, heap->GetGPUDescriptorHandleForHeapStart());
    };
    auto dispatch = [&]() {
        for (uint32_t g0 = 0; g0 < groups; g0 += 65535u) {
            const uint32_t n = std::min(65535u, groups - g0);
            l->SetComputeRoot32BitConstants(2, 1, &g0, 0);
            l->Dispatch(n, 1, 1);
        }
    };
    D3D12_RESOURCE_BARRIER uav{}; uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV; uav.UAV.pResource = mem.Get();

    if (reps == 0) {
        bind(); dispatch(); submit();
    } else {
        // warm-up: keep the GPU busy for warm_ms (the shader clock needs a few seconds of sustained load)
        const auto t0 = std::chrono::steady_clock::now();
        while (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() < warm_ms) {
            bind();
            for (int i = 0; i < 20; ++i) { dispatch(); l->ResourceBarrier(1, &uav); }
            submit();
        }
        D3D12_QUERY_HEAP_DESC qh{}; qh.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP; qh.Count = 2 * reps;
        ComPtr<ID3D12QueryHeap> qheap; CK(dev->CreateQueryHeap(&qh, IID_PPV_ARGS(&qheap)));
        auto tsb = buf(dev.Get(), 16ull * reps, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
        bind();
        for (uint32_t i = 0; i < reps; ++i) {
            l->EndQuery(qheap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 2 * i);
            dispatch();
            l->EndQuery(qheap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 2 * i + 1);
            l->ResourceBarrier(1, &uav);
        }
        l->ResolveQueryData(qheap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 2 * reps, tsb.Get(), 0);
        submit();
        UINT64 freq; CK(q->GetTimestampFrequency(&freq));
        void* m; D3D12_RANGE rr{0, (SIZE_T)(16ull * reps)}, none{0, 0}; CK(tsb->Map(0, &rr, &m));
        std::vector<double> us(reps);
        for (uint32_t i = 0; i < reps; ++i) {
            const UINT64 a = ((UINT64*)m)[2 * i], b = ((UINT64*)m)[2 * i + 1];
            us[i] = (double)(b - a) * 1e6 / (double)freq;
        }
        tsb->Unmap(0, &none);
        std::vector<double> s = us; std::sort(s.begin(), s.end());
        double mean = 0; for (double x : us) mean += x; mean /= reps;
        printf("TIME reps %u median_us %.2f min_us %.2f mean_us %.2f\n", reps, s[reps / 2], s[0], mean);
    }

    // read back
    uint64_t total = 0; for (const auto& r : rd) total += r.bytes;
    std::ofstream o(argv[3], std::ios::binary);
    if (total) {
        auto rb = buf(dev.Get(), total, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
        D3D12_RESOURCE_BARRIER b = tr(mem.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        l->ResourceBarrier(1, &b);
        uint64_t at = 0;
        for (const auto& r : rd) { l->CopyBufferRegion(rb.Get(), at, mem.Get(), r.off, r.bytes); at += r.bytes; }
        submit();
        void* m; D3D12_RANGE rr{0, (SIZE_T)total}, none{0, 0}; CK(rb->Map(0, &rr, &m));
        o.write((const char*)m, (std::streamsize)total);
        rb->Unmap(0, &none);
    }
    printf("OK groups %u read %llu bytes\n", groups, (unsigned long long)total);
    return 0;
}
