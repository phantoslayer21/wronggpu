// SPDX-License-Identifier: MIT
// hwfq_probe: prints, for every display adapter, what the graphics kernel reports about hardware GPU scheduling and the
// hardware flip queue. Read only: it opens adapters through the documented D3DKMT query interface (gdi32), reads
// capability structures and closes them again. It creates no device, changes no setting and does no GPU work.
//
// Sources (public Microsoft documentation):
//   D3DKMTEnumAdapters2 / D3DKMTQueryAdapterInfo / KMTQUERYADAPTERINFOTYPE  (d3dkmthk.h)
//   D3DKMT_WDDM_2_9_CAPS (HwSchSupportState, HwSchEnabled), D3DKMT_WDDM_3_0_CAPS (HwFlipQueueSupportState,
//   HwFlipQueueEnabled), D3DKMT_WDDM_3_1_CAPS, D3DKMT_FLIPQUEUEINFO, D3DKMT_DRIVERVERSION  (d3dkmdt.h / d3dkmthk.h)
//   "Hardware flip queue", Windows driver documentation.
//
// Build (x64 Native Tools prompt): cl /nologo /W4 /EHsc /std:c++17 hwfq_probe.cpp gdi32.lib dxgi.lib
#include <windows.h>
#include <winternl.h>
#include <d3dkmthk.h>
#include <dxgi1_6.h>
#include <cstdio>
#include <vector>

namespace {

const char *support_state(UINT s) {
    switch (s) {
    case 0: return "ALWAYS_OFF";
    case 1: return "EXPERIMENTAL";
    case 2: return "STABLE";
    case 3: return "ALWAYS_ON";
    }
    return "?";
}

template <class T> NTSTATUS query(D3DKMT_HANDLE h, KMTQUERYADAPTERINFOTYPE type, T *out) {
    D3DKMT_QUERYADAPTERINFO q = {};
    q.hAdapter = h;
    q.Type = type;
    q.pPrivateDriverData = out;
    q.PrivateDriverDataSize = sizeof(T);
    return D3DKMTQueryAdapterInfo(&q);
}

void print_dxgi_name(LUID luid) {
    IDXGIFactory1 *f = nullptr;
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void **>(&f)))) return;
    IDXGIAdapter1 *a = nullptr;
    for (UINT i = 0; f->EnumAdapters1(i, &a) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 d = {};
        a->GetDesc1(&d);
        if (d.AdapterLuid.LowPart == luid.LowPart && d.AdapterLuid.HighPart == luid.HighPart)
            std::printf("  dxgi description: %ls (vendor 0x%04x, device 0x%04x, flags 0x%x)\n", d.Description,
                        d.VendorId, d.DeviceId, d.Flags);
        a->Release();
    }
    f->Release();
}

}  // namespace

int main() {
    D3DKMT_ENUMADAPTERS2 e = {};
    NTSTATUS st = D3DKMTEnumAdapters2(&e);
    if (st != 0) {
        std::printf("D3DKMTEnumAdapters2 (count) failed: 0x%08lx\n", static_cast<unsigned long>(st));
        return 2;
    }
    std::vector<D3DKMT_ADAPTERINFO> info(e.NumAdapters);
    e.pAdapters = info.data();
    st = D3DKMTEnumAdapters2(&e);
    if (st != 0) {
        std::printf("D3DKMTEnumAdapters2 failed: 0x%08lx\n", static_cast<unsigned long>(st));
        return 2;
    }
    std::printf("adapters: %u\n", e.NumAdapters);
    for (ULONG i = 0; i < e.NumAdapters; ++i) {
        const D3DKMT_ADAPTERINFO &ai = info[i];
        std::printf("adapter %lu: luid %08lx:%08lx, sources %lu\n", i, static_cast<unsigned long>(ai.AdapterLuid.HighPart),
                    ai.AdapterLuid.LowPart, ai.NumOfSources);
        print_dxgi_name(ai.AdapterLuid);

        D3DKMT_DRIVERVERSION wddm = {};
        st = query(ai.hAdapter, KMTQAITYPE_DRIVERVERSION, &wddm);
        std::printf("  KMTQAITYPE_DRIVERVERSION: status 0x%08lx, value 0x%x\n", static_cast<unsigned long>(st),
                    static_cast<unsigned>(wddm));

        D3DKMT_WDDM_2_9_CAPS c29 = {};
        st = query(ai.hAdapter, KMTQAITYPE_WDDM_2_9_CAPS, &c29);
        std::printf("  KMTQAITYPE_WDDM_2_9_CAPS: status 0x%08lx, Value 0x%08x, HwSchSupportState %u (%s), "
                    "HwSchEnabled %u\n",
                    static_cast<unsigned long>(st), c29.Value, c29.HwSchSupportState,
                    support_state(c29.HwSchSupportState), c29.HwSchEnabled);

        D3DKMT_WDDM_3_0_CAPS c30 = {};
        st = query(ai.hAdapter, KMTQAITYPE_WDDM_3_0_CAPS, &c30);
        std::printf("  KMTQAITYPE_WDDM_3_0_CAPS: status 0x%08lx, Value 0x%08x, HwFlipQueueSupportState %u (%s), "
                    "HwFlipQueueEnabled %u, DisplayableSupported %u\n",
                    static_cast<unsigned long>(st), c30.Value, c30.HwFlipQueueSupportState,
                    support_state(c30.HwFlipQueueSupportState), c30.HwFlipQueueEnabled, c30.DisplayableSupported);

        D3DKMT_WDDM_3_1_CAPS c31 = {};
        st = query(ai.hAdapter, KMTQAITYPE_WDDM_3_1_CAPS, &c31);
        std::printf("  KMTQAITYPE_WDDM_3_1_CAPS: status 0x%08lx, Value 0x%08x\n", static_cast<unsigned long>(st),
                    c31.Value);

        D3DKMT_FLIPQUEUEINFO fq = {};
        st = query(ai.hAdapter, KMTQAITYPE_FLIPQUEUEINFO, &fq);
        std::printf("  KMTQAITYPE_FLIPQUEUEINFO: status 0x%08lx, MaxHardwareFlipQueueLength %u, "
                    "MaxSoftwareFlipQueueLength %u, FlipFlags.FlipInterval %u\n",
                    static_cast<unsigned long>(st), fq.MaxHardwareFlipQueueLength, fq.MaxSoftwareFlipQueueLength,
                    static_cast<unsigned>(fq.FlipFlags.FlipInterval));

        D3DKMT_CLOSEADAPTER ca = {};
        ca.hAdapter = ai.hAdapter;
        D3DKMTCloseAdapter(&ca);
    }
    return 0;
}
