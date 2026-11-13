// SPDX-License-Identifier: MIT
// Small D3D12 executor for the spec 51 kernels and probes (no HIP): one default-heap "arena" buffer bound as root UAVs
// u0 and u1 (the kernel convention of clean/dx12/README.md), root constants b0 = (group base, grid, kernel arguments),
// the driver-channel table at u0 space 0x7FFF0ADE (clean/dx12/hlsl/wg_ags.hlsli).
//
//   rr_host <job file>
//
// Job file (whitespace separated, one command per line, '#' comments):
//   arena <bytes> <init file>              arena size; the init file is copied to offset 0 (may be shorter)
//   kernel <id> <dxil file> <arg dwords>   pipeline with <arg dwords> kernel arguments (after the 2 runtime words)
//   dispatch <id> <groups> <arg> ...       one dispatch (grids above 65535 are split, group base in word 0)
//   time <reps> [first]                    run the dispatch list (from dispatch <first>, default 0) <reps> more
//                                          times; GPU ms per dispatch
//   readback <out file>                    write the whole arena after the first run
//   checkpoint <out file>                  first run only: submit the dispatches so far as one command list, wait,
//                                          write the whole arena (the arena stays resident; e.g. one file per frame)
//   require_double                        refuse before pipelines if binary64 operations are unavailable
// The dispatch list runs once (UAV barrier between dispatches), then the readback, then the timing runs.
// Timing runs also print the span from the first dispatch's start to the last dispatch's end ("total").
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
using Microsoft::WRL::ComPtr;

static void ck(HRESULT h, const char* s, int line)
{
    if (FAILED(h)) { fprintf(stderr, "D3D12 %s (line %d): %08x\n", s, line, (unsigned)h); exit(2); }
}
#define CK(x) ck((x), #x, __LINE__)

static std::vector<char> slurp(const std::string& p)
{
    std::ifstream f(p, std::ios::binary);
    if (!f) { fprintf(stderr, "cannot read %s\n", p.c_str()); exit(1); }
    return std::vector<char>((std::istreambuf_iterator<char>(f)), {});
}

static ComPtr<ID3D12Resource> buf(ID3D12Device* d, UINT64 size, D3D12_HEAP_TYPE ht, D3D12_RESOURCE_FLAGS fl,
                                  D3D12_RESOURCE_STATES st)
{
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = ht;
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = size;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags = fl;
    ComPtr<ID3D12Resource> r;
    CK(d->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, st, nullptr, IID_PPV_ARGS(&r)));
    return r;
}

static D3D12_RESOURCE_BARRIER tr(ID3D12Resource* r, D3D12_RESOURCE_STATES x, D3D12_RESOURCE_STATES y)
{
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition = {r, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, x, y};
    return b;
}

struct Kernel { ComPtr<ID3D12RootSignature> rs; ComPtr<ID3D12PipelineState> pso; uint32_t nargs; };
struct Dispatch { std::string id; uint32_t groups; std::vector<uint32_t> args; };

int main(int argc, char** argv)
{
    if (argc != 2) { fprintf(stderr, "usage: rr_host <job file>\n"); return 1; }
    std::ifstream jf(argv[1]);
    if (!jf) { fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }

    ComPtr<IDXGIFactory6> fac;
    CK(CreateDXGIFactory1(IID_PPV_ARGS(&fac)));
    ComPtr<IDXGIAdapter1> ad;
    for (UINT i = 0;; i++) {
        ComPtr<IDXGIAdapter1> x;
        if (fac->EnumAdapters1(i, &x) == DXGI_ERROR_NOT_FOUND) break;
        DXGI_ADAPTER_DESC1 d;
        x->GetDesc1(&d);
        if (d.VendorId == 0x1002 && !(d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) { ad = x; break; }
    }
    if (!ad) { fprintf(stderr, "no AMD adapter\n"); return 2; }
    ComPtr<ID3D12Device> dev;
    CK(D3D12CreateDevice(ad.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&dev)));
    DXGI_ADAPTER_DESC1 adapterDesc{};
    CK(ad->GetDesc1(&adapterDesc));
    printf("adapter %ls vendor=%04x device=%04x\n",adapterDesc.Description,adapterDesc.VendorId,adapterDesc.DeviceId);
    D3D12_COMMAND_QUEUE_DESC qd{};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ComPtr<ID3D12CommandQueue> q;
    CK(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&q)));
    ComPtr<ID3D12CommandAllocator> al;
    CK(dev->CreateCommandAllocator(qd.Type, IID_PPV_ARGS(&al)));
    ComPtr<ID3D12GraphicsCommandList> l;
    CK(dev->CreateCommandList(0, qd.Type, al.Get(), nullptr, IID_PPV_ARGS(&l)));
    ComPtr<ID3D12Fence> fence;
    CK(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
    HANDLE ev = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    UINT64 fv = 0;
    auto submit_wait = [&]() {
        CK(l->Close());
        ID3D12CommandList* ll = l.Get();
        q->ExecuteCommandLists(1, &ll);
        CK(q->Signal(fence.Get(), ++fv));
        CK(fence->SetEventOnCompletion(fv, ev));
        if (WaitForSingleObject(ev, 60000) != WAIT_OBJECT_0) { fprintf(stderr, "timeout\n"); exit(4); }
        CK(dev->GetDeviceRemovedReason());
        CK(al->Reset());
        CK(l->Reset(al.Get(), nullptr));
    };

    std::map<std::string, Kernel> kernels;
    std::vector<Dispatch> list;
    UINT64 arena = 0;
    std::vector<char> init;
    std::string readback;
    std::vector<std::pair<size_t, std::string>> checkpoints;   // (dispatches before it, file)
    int reps = 0;
    size_t tfirst = 0;
    std::string line;
    while (std::getline(jf, line)) {
        std::istringstream s(line);
        std::string cmd;
        if (!(s >> cmd) || cmd[0] == '#') continue;
        if (cmd == "arena") {
            std::string f;
            s >> arena >> f;
            init = slurp(f);
            if (init.size() > arena) { fprintf(stderr, "init larger than arena\n"); return 1; }
        } else if (cmd == "require_double") {
            D3D12_FEATURE_DATA_D3D12_OPTIONS options{};
            CK(dev->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS,&options,sizeof(options)));
            if (!options.DoublePrecisionFloatShaderOps) {
                fprintf(stderr,"binary64 shader operations unavailable\n"); return 2;
            }
            printf("binary64 shader operations supported\n");
        } else if (cmd == "kernel") {
            std::string id, f;
            Kernel k;
            s >> id >> f >> k.nargs;
            auto code = slurp(f);
            D3D12_ROOT_PARAMETER rp[4]{};
            rp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
            rp[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
            rp[1].Descriptor.ShaderRegister = 1;
            rp[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
            rp[2].Constants = {0, 0, 2 + k.nargs};
            D3D12_DESCRIPTOR_RANGE ags{};
            ags.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
            ags.NumDescriptors = 1;
            ags.RegisterSpace = 0x7FFF0ADE;
            rp[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            rp[3].DescriptorTable = {1, &ags};
            D3D12_ROOT_SIGNATURE_DESC rsd{};
            rsd.NumParameters = 4;
            rsd.pParameters = rp;
            ComPtr<ID3DBlob> blob, err;
            if (FAILED(D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err))) {
                fprintf(stderr, "root signature: %s\n", err ? (const char*)err->GetBufferPointer() : "?");
                return 2;
            }
            CK(dev->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&k.rs)));
            D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};
            pd.pRootSignature = k.rs.Get();
            pd.CS = {code.data(), code.size()};
            CK(dev->CreateComputePipelineState(&pd, IID_PPV_ARGS(&k.pso)));
            kernels[id] = k;
        } else if (cmd == "dispatch") {
            Dispatch d;
            s >> d.id >> d.groups;
            std::string a;
            while (s >> a) d.args.push_back((uint32_t)strtoul(a.c_str(), nullptr, 0));
            if (!kernels.count(d.id) || kernels[d.id].nargs != d.args.size()) {
                fprintf(stderr, "dispatch %s: unknown kernel or %zu args\n", d.id.c_str(), d.args.size());
                return 1;
            }
            list.push_back(d);
        } else if (cmd == "time") {
            s >> reps;
            if (!(s >> tfirst)) tfirst = 0;
        } else if (cmd == "readback") {
            s >> readback;
        } else if (cmd == "checkpoint") {
            std::string f;
            s >> f;
            checkpoints.emplace_back(list.size(), f);
        } else {
            fprintf(stderr, "unknown command %s\n", cmd.c_str());
            return 1;
        }
    }
    if (!arena) { fprintf(stderr, "no arena\n"); return 1; }

    auto mem = buf(dev.Get(), arena, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                   D3D12_RESOURCE_STATE_COMMON);
    auto agsb = buf(dev.Get(), 256, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COMMON);
    D3D12_DESCRIPTOR_HEAP_DESC hd{};
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    hd.NumDescriptors = 1;
    hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ComPtr<ID3D12DescriptorHeap> heap;
    CK(dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap)));
    D3D12_UNORDERED_ACCESS_VIEW_DESC ud{};
    ud.Format = DXGI_FORMAT_R32_TYPELESS;
    ud.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    ud.Buffer.NumElements = 64;
    ud.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
    dev->CreateUnorderedAccessView(agsb.Get(), nullptr, &ud, heap->GetCPUDescriptorHandleForHeapStart());

    // upload (zero-filled beyond the init file)
    {
        auto up = buf(dev.Get(), arena, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
                      D3D12_RESOURCE_STATE_GENERIC_READ);
        void* m;
        D3D12_RANGE none{0, 0};
        CK(up->Map(0, &none, &m));
        memset(m, 0, (size_t)arena);
        if (!init.empty()) memcpy(m, init.data(), init.size());
        up->Unmap(0, nullptr);
        D3D12_RESOURCE_BARRIER b = tr(mem.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        l->ResourceBarrier(1, &b);
        l->CopyBufferRegion(mem.Get(), 0, up.Get(), 0, arena);
        b = tr(mem.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        l->ResourceBarrier(1, &b);
        submit_wait();
    }

    const UINT nq = 2 * (UINT)std::max<size_t>(list.size(), 1);
    ComPtr<ID3D12QueryHeap> qh;
    ComPtr<ID3D12Resource> qrb;
    D3D12_QUERY_HEAP_DESC qhd{};
    qhd.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    qhd.Count = nq;
    CK(dev->CreateQueryHeap(&qhd, IID_PPV_ARGS(&qh)));
    qrb = buf(dev.Get(), 8ull * nq, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);

    auto dump = [&](const std::string& file) {
        auto rb = buf(dev.Get(), arena, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE,
                      D3D12_RESOURCE_STATE_COPY_DEST);
        D3D12_RESOURCE_BARRIER b = tr(mem.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        l->ResourceBarrier(1, &b);
        l->CopyBufferRegion(rb.Get(), 0, mem.Get(), 0, arena);
        b = tr(mem.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        l->ResourceBarrier(1, &b);
        submit_wait();
        void* m;
        D3D12_RANGE r{0, (SIZE_T)arena}, none{0, 0};
        CK(rb->Map(0, &r, &m));
        std::ofstream f(file, std::ios::binary);
        f.write((const char*)m, (std::streamsize)arena);
        rb->Unmap(0, &none);
    };

    // timed: a timestamp before and after every dispatch
    auto record = [&](bool timed) {
        ID3D12DescriptorHeap* hs[] = {heap.Get()};
        l->SetDescriptorHeaps(1, hs);
        size_t ci = 0;
        for (size_t di = timed ? tfirst : 0; di < list.size(); ++di) {
            if (!timed) {   // checkpoints: submit what is recorded, write the arena, continue on the resident arena
                bool any = false;
                for (; ci < checkpoints.size() && checkpoints[ci].first == di; ++ci) {
                    if (di) {
                        D3D12_RESOURCE_BARRIER b{};
                        b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
                        b.UAV.pResource = mem.Get();
                        l->ResourceBarrier(1, &b);
                    }
                    submit_wait();
                    dump(checkpoints[ci].second);
                    any = true;
                }
                if (any) l->SetDescriptorHeaps(1, hs);
            }
            const Dispatch& d = list[di];
            const Kernel& k = kernels[d.id];
            if (di > (timed ? tfirst : 0)) {
                D3D12_RESOURCE_BARRIER b{};
                b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
                b.UAV.pResource = mem.Get();
                l->ResourceBarrier(1, &b);
            }
            if (timed) l->EndQuery(qh.Get(), D3D12_QUERY_TYPE_TIMESTAMP, (UINT)(2 * di));
            l->SetPipelineState(k.pso.Get());
            l->SetComputeRootSignature(k.rs.Get());
            l->SetComputeRootUnorderedAccessView(0, mem->GetGPUVirtualAddress());
            l->SetComputeRootUnorderedAccessView(1, mem->GetGPUVirtualAddress());
            l->SetComputeRootDescriptorTable(3, heap->GetGPUDescriptorHandleForHeapStart());
            if (!d.args.empty()) l->SetComputeRoot32BitConstants(2, (UINT)d.args.size(), d.args.data(), 2);
            for (uint32_t base = 0; base < d.groups; base += 65535) {
                uint32_t rt[2] = {base, d.groups};
                l->SetComputeRoot32BitConstants(2, 2, rt, 0);
                l->Dispatch(std::min<uint32_t>(65535, d.groups - base), 1, 1);
            }
            if (timed) l->EndQuery(qh.Get(), D3D12_QUERY_TYPE_TIMESTAMP, (UINT)(2 * di + 1));
        }
    };

    record(false);
    submit_wait();
    for (const auto& c : checkpoints)
        if (c.first >= list.size()) dump(c.second);

    if (!readback.empty()) {
        auto rb = buf(dev.Get(), arena, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE,
                      D3D12_RESOURCE_STATE_COPY_DEST);
        D3D12_RESOURCE_BARRIER b = tr(mem.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        l->ResourceBarrier(1, &b);
        l->CopyBufferRegion(rb.Get(), 0, mem.Get(), 0, arena);
        b = tr(mem.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        l->ResourceBarrier(1, &b);
        submit_wait();
        void* m;
        D3D12_RANGE r{0, (SIZE_T)arena}, none{0, 0};
        CK(rb->Map(0, &r, &m));
        std::ofstream f(readback, std::ios::binary);
        f.write((const char*)m, (std::streamsize)arena);
        rb->Unmap(0, &none);
    }

    if (reps > 0) {   // per dispatch: best and mean GPU time over the runs
        UINT64 freq = 0;
        CK(q->GetTimestampFrequency(&freq));
        std::vector<double> best(list.size(), 1e30), sum(list.size(), 0.0);
        double tbest = 1e30, tsum = 0.0;
        const size_t nt = list.size() > tfirst ? list.size() - tfirst : 0;
        for (int r = 0; r < reps && nt; ++r) {
            record(true);
            l->ResolveQueryData(qh.Get(), D3D12_QUERY_TYPE_TIMESTAMP, (UINT)(2 * tfirst), (UINT)(2 * nt), qrb.Get(), 0);
            submit_wait();
            UINT64* t;
            D3D12_RANGE rr{0, (SIZE_T)(16 * nt)}, none{0, 0};
            CK(qrb->Map(0, &rr, (void**)&t));
            for (size_t i = 0; i < nt; ++i) {
                const double ms = (double)(t[2 * i + 1] - t[2 * i]) * 1000.0 / (double)freq;
                best[tfirst + i] = std::min(best[tfirst + i], ms);
                sum[tfirst + i] += ms;
            }
            const double span = (double)(t[2 * nt - 1] - t[0]) * 1000.0 / (double)freq;
            tbest = std::min(tbest, span);
            tsum += span;
            qrb->Unmap(0, &none);
        }
        for (size_t di = tfirst; di < list.size(); ++di)
            printf("time %zu %s best %.4f ms mean %.4f ms over %d runs\n", di, list[di].id.c_str(), best[di],
                   sum[di] / reps, reps);
        printf("total best %.4f ms mean %.4f ms over %d runs\n", tbest, tsum / reps, reps);
    }
    printf("ok %zu dispatches\n", list.size());
    return 0;
}
