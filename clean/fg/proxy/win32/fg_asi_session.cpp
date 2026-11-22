// SPDX-License-Identifier: MIT
// WrongGPU frame generation (issue #96, F9): the FG session host of WrongGPU-FG.asi. See fg_asi_session.h.
// Written for this project.
#include "fg_asi_session.h"

#include <wrl/client.h>

#include <algorithm>
#include <cstring>
#include <new>

#include "../../presenter/fg_dxgi_presenter.h"
#include "../../presenter/fg_win_common.h"
#include "../../runtime/fg_d3d12_executor.h"
#include "../../runtime/fg_model.h"
#include "../../runtime/fg_session.h"
#include "../../runtime/fg_session_win.h"
#include "../fg_contract_bridge.h"

namespace wgfgp {
namespace win32 {

using Microsoft::WRL::ComPtr;

namespace {

constexpr uint32_t kMaxSessionFailures = 3;
constexpr DWORD kDestroyWaitMs = 1000;
constexpr DWORD kLatencyWaitMs = 100;
constexpr uint32_t kCopyRing = 4;

bool FileExists(const std::string& p) {
    const DWORD a = GetFileAttributesA(p.c_str());
    return a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY);
}

wgfg::FgTicks HostClock(void*) { return wgfg::FgInterruptNow(); }

// The facade format of a swap-chain format: the typeless family where the game may view the buffer as sRGB.
DXGI_FORMAT FacadeFormat(DXGI_FORMAT f) {
    switch (f) {
        case DXGI_FORMAT_R8G8B8A8_UNORM: case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: return DXGI_FORMAT_R8G8B8A8_TYPELESS;
        case DXGI_FORMAT_B8G8R8A8_UNORM: case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: return DXGI_FORMAT_B8G8R8A8_TYPELESS;
        case DXGI_FORMAT_R10G10B10A2_UNORM: return DXGI_FORMAT_R10G10B10A2_TYPELESS;
        default: return f;
    }
}

}  // namespace

bool FgIsRdna4(uint32_t vendor_id, uint32_t device_id) {
    return vendor_id == 0x1002 &&
           (device_id == 0x7550 || device_id == 0x7551 || device_id == 0x7590 || device_id == 0x7591);
}

bool FgAsiSessionPrecheck(const FgAsiSessionConfig& cfg, wgfg::FgExecWeights* weights, std::string* why) {
    std::string w;
    if (!weights) {
        if (why) *why = "no weights output";
        return false;
    }
    const DWORD a = GetFileAttributesA(cfg.dxil_dir.c_str());
    if (cfg.dxil_dir.empty() || a == INVALID_FILE_ATTRIBUTES || !(a & FILE_ATTRIBUTE_DIRECTORY)) {
        if (why) *why = "the FG kernel folder '" + cfg.dxil_dir + "' is missing";
        return false;
    }
    const wgfg::FgNet2Path path = cfg.net2_exact ? wgfg::FgNet2Path::Exact : wgfg::FgNet2Path::Fast;
    for (const std::string& k : wgfg::FgD3d12ExecutorKernelNames(true, path)) {
        if (!FileExists(cfg.dxil_dir + "\\" + k + ".dxil")) {
            if (why) *why = "the FG kernel folder has no " + k + ".dxil";
            return false;
        }
    }
    wgfg::FgModelReport report;
    std::vector<uint8_t> bytes;
    const wgfg::FgRefusal r = wgfg::FgValidateModelFile(cfg.model_path, wgfg::FgShippedModelExpectation(), &report,
                                                        &bytes);
    if (r != wgfg::FgRefusal::None) {
        if (why) *why = std::string("the FG model did not validate: ") + wgfg::FgRefusalName(r) + " " + report.detail;
        return false;
    }
    if (!wgfg::FgExecWeightsFromModel(bytes.data(), bytes.size(), report, weights, &w)) {
        if (why) *why = "the FG model's weights: " + w;
        return false;
    }
    return true;
}

// ---- the host ------------------------------------------------------------------------------------------------------

struct FgAsiSessionHost::Impl {
    FgAsiSessionConfig cfg;
    std::unique_ptr<wgfg::FgExecWeights> weights;
    mutable std::recursive_mutex mu;
    FgProxy* core = nullptr;
    bool adapter_set = false;
    bool tearing_supported = false;  // DXGI_FEATURE_PRESENT_ALLOW_TEARING on the game's factory

    // the attached chain
    IDXGISwapChain4* inner = nullptr;  // no reference: the wrapper holds the chain while we are attached
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12Device> device;
    HWND hwnd = nullptr;
    UINT chain_flags = 0;

    // facade
    std::vector<ComPtr<ID3D12Resource>> facade;
    UINT facade_index = 0;
    HANDLE latency = nullptr;  // the facade's frame-latency semaphore
    DXGI_FORMAT chain_format = DXGI_FORMAT_UNKNOWN;
    UINT width = 0, height = 0;

    // copy path (facade -> real back buffer)
    ComPtr<ID3D12CommandAllocator> alloc[kCopyRing];
    uint64_t alloc_value[kCopyRing] = {};
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> copy_fence;
    uint64_t copy_value = 0;
    HANDLE copy_event = nullptr;
    uint32_t ring = 0;

    // session
    std::unique_ptr<wgfg::FgD3d12Executor> exec;
    std::unique_ptr<wgfg::FgWinPresenterSource> win_source;
    std::unique_ptr<wgfg::FgExecutorPresenterSource> exec_source;
    std::unique_ptr<wgfg::FgSession> session;
    bool handled = false;  // the current interval was presented by the session
    uint64_t last_values[3] = {};  // the executor's fence after the last intervals (queue-depth bound)
    FgAsiSessionStats stats;

    void Log(const std::string& line) const {
        if (cfg.log) cfg.log(cfg.log_ctx, line.c_str());
    }

    void ReleaseFacade() {
        facade.clear();
        facade_index = 0;
        stats.facade = false;
    }

    bool CreateFacade() {
        ReleaseFacade();
        if (!inner || !device) return false;
        DXGI_SWAP_CHAIN_DESC1 sd = {};
        if (FAILED(inner->GetDesc1(&sd)) || sd.BufferCount == 0 || sd.Width == 0 || sd.Height == 0) return false;
        chain_format = sd.Format;
        width = sd.Width;
        height = sd.Height;
        chain_flags = sd.Flags;
        D3D12_HEAP_PROPERTIES hp = {};
        hp.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC rd = {};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        rd.Width = sd.Width;
        rd.Height = sd.Height;
        rd.DepthOrArraySize = 1;
        rd.MipLevels = 1;
        rd.Format = FacadeFormat(sd.Format);
        rd.SampleDesc.Count = 1;
        rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        facade.resize(sd.BufferCount);
        for (auto& f : facade) {
            if (FAILED(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COMMON,
                                                       nullptr, IID_PPV_ARGS(&f)))) {
                ReleaseFacade();
                return false;
            }
        }
        facade_index = 0;
        stats.facade = true;
        return true;
    }

    bool EnsureCopyObjects() {
        if (list) return true;
        if (!device) return false;
        for (auto& a : alloc)
            if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&a)))) return false;
        if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&copy_fence)))) return false;
        if (!copy_event) copy_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!copy_event) return false;
        if (FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc[0].Get(), nullptr,
                                             IID_PPV_ARGS(&list))) ||
            FAILED(list->Close())) {
            list.Reset();
            return false;
        }
        return true;
    }

    bool WaitCopy(uint64_t v, DWORD ms) {
        if (!copy_fence || copy_fence->GetCompletedValue() >= v) return true;
        if (FAILED(copy_fence->SetEventOnCompletion(v, copy_event))) return false;
        WaitForSingleObject(copy_event, ms);
        return copy_fence->GetCompletedValue() >= v;
    }

    // The facade buffer the game presented, copied into the real chain's current back buffer (the game's queue).
    bool CopyFacadeToChain() {
        if (facade.empty() || !inner || !queue || !EnsureCopyObjects()) return false;
        const UINT bbi = inner->GetCurrentBackBufferIndex();
        ComPtr<ID3D12Resource> bb;
        if (FAILED(inner->GetBuffer(bbi, IID_PPV_ARGS(&bb)))) return false;
        const uint32_t a = ring % kCopyRing;
        if (!WaitCopy(alloc_value[a], 100)) return false;
        if (FAILED(alloc[a]->Reset()) || FAILED(list->Reset(alloc[a].Get(), nullptr))) return false;
        ID3D12Resource* src = facade[facade_index % facade.size()].Get();
        D3D12_RESOURCE_BARRIER b[2] = {};
        b[0].Type = b[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b[0].Transition = {bb.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_PRESENT,
                           D3D12_RESOURCE_STATE_COPY_DEST};
        b[1].Transition = {src, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COMMON,
                           D3D12_RESOURCE_STATE_COPY_SOURCE};
        list->ResourceBarrier(2, b);
        list->CopyResource(bb.Get(), src);
        for (D3D12_RESOURCE_BARRIER& x : b) std::swap(x.Transition.StateBefore, x.Transition.StateAfter);
        list->ResourceBarrier(2, b);
        if (FAILED(list->Close())) return false;
        ID3D12CommandList* lists[] = {list.Get()};
        queue->ExecuteCommandLists(1, lists);
        queue->Signal(copy_fence.Get(), ++copy_value);
        alloc_value[a] = copy_value;
        ++ring;
        return true;
    }

    void AdvanceFacade() {
        if (!facade.empty()) facade_index = (facade_index + 1) % static_cast<UINT>(facade.size());
        if (latency) ReleaseSemaphore(latency, 1, nullptr);
    }

    // At most two intervals of executor work in flight before the game gets its next frame (bounded wait).
    void BoundQueueDepth() {
        if (!exec) return;
        const uint64_t want = last_values[0];
        last_values[0] = last_values[1];
        last_values[1] = last_values[2];
        last_values[2] = exec->LastComputeValue();
        if (want == 0) return;
        const ULONGLONG until = GetTickCount64() + 50;
        while (exec->CompletedComputeValue() < want && GetTickCount64() < until) Sleep(0);
    }

    void DestroySession(const char* why) {
        if (!session) {
            exec_source.reset();
            win_source.reset();
            return;
        }
        wgfg::FgSessionResult r = session->Destroy();
        const ULONGLONG until = GetTickCount64() + kDestroyWaitMs;
        while (r.error == wgfg::FgSessionError::PresenterBusy && GetTickCount64() < until) {
            Sleep(2);
            r = session->Destroy();
        }
        Log(std::string("fg host: session destroyed (") + why + "): " + wgfg::FgSessionErrorName(r.error));
        if (core) core->set_executor(nullptr);
        session.reset();
        exec_source.reset();
        win_source.reset();
        for (uint64_t& v : last_values) v = 0;
        ++stats.sessions_destroyed;
    }

    void TryCreateSession(FgProxy* c) {
        if (session || stats.gave_up || !c || !c->fg_taken() || !inner || !device || !queue) return;
        WgFgSessionDesc sd = {};
        if (!c->session_desc(&sd)) return;  // options and a latched present first
        if (!exec) {
            wgfg::FgD3d12ExecutorDesc ed;
            ed.dxil_dir = cfg.dxil_dir;
            ed.weights = weights.get();
            ed.net2 = cfg.net2_exact ? wgfg::FgNet2Path::Exact : wgfg::FgNet2Path::Fast;
            ed.settings.menu_detection = true;
            exec.reset(new (std::nothrow) wgfg::FgD3d12Executor(ed));
            if (!exec) return;
        }
        wgfg::FgPresenterFactoryDesc base;
        base.device = device.Get();
        base.queue = queue.Get();
        base.swap_chain = inner;
        base.hwnd = hwnd;
        win_source.reset(new (std::nothrow) wgfg::FgWinPresenterSource(base));
        if (!win_source) return;
        exec_source.reset(new (std::nothrow) wgfg::FgExecutorPresenterSource(win_source.get(), exec.get()));
        session.reset(new (std::nothrow) wgfg::FgSession());
        if (!exec_source || !session) {
            DestroySession("allocation");
            return;
        }
        wgfg::FgSessionDesc d;
        d.create = ToRuntime(sd);
        d.executor = exec.get();
        d.presenter_source = exec_source.get();
        d.clock = &HostClock;
        d.log = cfg.log;
        d.log_ctx = cfg.log_ctx;
        const wgfg::FgSessionResult r = session->Create(d);
        if (r.error != wgfg::FgSessionError::None) {
            ++stats.session_failures;
            Log(std::string("fg host: session not created: ") + wgfg::FgSessionErrorName(r.error) + " (executor " +
                wgfg::FgExecRefusalName(exec->LastRefusal()) + ": " + exec->LastWhy() + ")");
            session.reset();
            exec_source.reset();
            win_source.reset();
            if (stats.session_failures >= kMaxSessionFailures) {
                stats.gave_up = true;
                Log("fg host: frame generation stays off for this run (session creation failed 3 times); the game's "
                    "frames are presented as it asks");
            }
            return;
        }
        WgFgExecutorV1 h = session->ExecutorHandOff();
        c->set_executor(&h);
        ++stats.sessions_created;
        Log(std::string("fg host: session created, ") + std::to_string(sd.output_width) + " x " +
            std::to_string(sd.output_height) + ", preset " + std::to_string(sd.preset) + ", N " +
            std::to_string(sd.max_generated) + ", net2 " + (cfg.net2_exact ? "exact" : "fast"));
    }
};

FgAsiSessionHost::FgAsiSessionHost(const FgAsiSessionConfig& cfg, std::unique_ptr<wgfg::FgExecWeights> weights)
    : impl_(new Impl()) {
    impl_->cfg = cfg;
    impl_->weights = std::move(weights);
}

FgAsiSessionHost::~FgAsiSessionHost() {
    std::lock_guard<std::recursive_mutex> lock(impl_->mu);
    impl_->DestroySession("unload");
    if (impl_->exec) impl_->exec->WaitIdle(kDestroyWaitMs);
    impl_->WaitCopy(impl_->copy_value, kDestroyWaitMs);
    impl_->ReleaseFacade();
    if (impl_->latency) CloseHandle(impl_->latency);
    if (impl_->copy_event) CloseHandle(impl_->copy_event);
}

void FgAsiSessionHost::OnInstalled(FgProxy* core) {
    std::lock_guard<std::recursive_mutex> lock(impl_->mu);
    impl_->core = core;
    // the precheck passed: the model validated; the presenter is decided at session creation (refused there, not
    // replaced)
    if (core) core->set_runtime_facts(true, true);
}

FgAsiSessionStats FgAsiSessionHost::Stats() const {
    std::lock_guard<std::recursive_mutex> lock(impl_->mu);
    FgAsiSessionStats s = impl_->stats;
    if (impl_->exec) {
        const wgfg::FgExecCounters& c = impl_->exec->Counters();
        s.exec_shared = c.shared;
        s.exec_indices = c.indices;
        s.exec_copies = c.copies;
        s.exec_game_colour_copies = c.game_colour_copies;
        s.exec_net2_fast = c.net2_fast;
        s.exec_net2_exact = c.net2_exact;
    }
    return s;
}

void FgAsiSessionHost::FactoryNote(void* ctx, FgProxy* core, IUnknown* factory) {
    FgAsiSessionHost* self = static_cast<FgAsiSessionHost*>(ctx);
    if (!self || !core || !factory) return;
    Impl& m = *self->impl_;
    std::lock_guard<std::recursive_mutex> lock(m.mu);
    if (m.adapter_set) return;
    {
        ComPtr<IDXGIFactory5> f5;
        BOOL tearing = FALSE;
        m.tearing_supported =
            SUCCEEDED(factory->QueryInterface(IID_PPV_ARGS(&f5))) &&
            SUCCEEDED(f5->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &tearing, sizeof(tearing))) &&
            tearing;
    }
    ComPtr<IDXGIFactory1> f1;
    if (FAILED(factory->QueryInterface(IID_PPV_ARGS(&f1)))) return;
    for (UINT i = 0; i < 16; ++i) {
        ComPtr<IDXGIAdapter1> a;
        if (f1->EnumAdapters1(i, &a) == DXGI_ERROR_NOT_FOUND) break;
        DXGI_ADAPTER_DESC1 d = {};
        if (!a || FAILED(a->GetDesc1(&d)) || (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) continue;
        if (!FgIsRdna4(d.VendorId, d.DeviceId)) continue;
        WgFgAdapterIdentity id = {};
        id.struct_size = sizeof(id);
        id.abi_version = WG_FG_ADAPTER_ABI_VERSION;
        id.vendor_id = d.VendorId;
        id.device_id = d.DeviceId;
        id.luid.low = d.AdapterLuid.LowPart;
        id.luid.high = d.AdapterLuid.HighPart;
        id.rdna4 = 1;
        id.probes_passed = 1;  // the precheck found every kernel of the executor
        core->set_adapter(id);
        m.adapter_set = true;
        char line[96];
        wsprintfA(line, "fg host: adapter vendor 0x%04x device 0x%04x (RDNA 4)", d.VendorId, d.DeviceId);
        m.Log(line);
        return;
    }
    m.Log("fg host: no RDNA 4 adapter on the game's factory; frame generation is not taken");
}

int32_t FgAsiSessionHost::Route(void* ctx, FgProxy* core, const WgFgPresentDesc* desc, WgFgPresentResult* out) {
    FgAsiSessionHost* self = static_cast<FgAsiSessionHost*>(ctx);
    if (!core) return -1;
    if (!self) return core->present(desc, out);
    Impl& m = *self->impl_;
    std::lock_guard<std::recursive_mutex> lock(m.mu);
    m.handled = false;
    try {
        m.TryCreateSession(core);
        wgfg::FgSessionResult sr;
        const int32_t rc = wgfg::FgSessionPresent(core, m.session.get(), desc, out, &sr);
        if (m.session && out) {
            const bool took = out->status == WG_FG_PRESENT_GENERATE || out->status == WG_FG_PRESENT_REAL_ONLY;
            // the session presented the game frame unless it refused before its presenter was reached
            m.handled = took && sr.frame_token == desc->frame_token &&
                        (sr.error == wgfg::FgSessionError::None || sr.error == wgfg::FgSessionError::FrameRefused);
            if (took && !m.handled) {
                m.Log(std::string("fg host: the session did not present a game frame: ") +
                      wgfg::FgSessionErrorName(sr.error));
                if (m.session->State() != wgfg::FgSessionState::Active) {
                    m.DestroySession("refused");
                    if (++m.stats.session_failures >= kMaxSessionFailures) m.stats.gave_up = true;
                }
            }
        }
        return rc;
    } catch (...) {
        m.handled = false;
        return -1;
    }
}

HRESULT FgAsiSessionHost::PresentInterval(const FgPresentCall& call, IFgInnerPresenter& inner) {
    Impl& m = *impl_;
    std::lock_guard<std::recursive_mutex> lock(m.mu);
    ++m.stats.intervals;
    if (m.handled && m.session) {
        // the session's presenter shows this interval (real and generated images) on the real chain
        m.handled = false;
        ++m.stats.session_presents;
        m.BoundQueueDepth();
        m.AdvanceFacade();
        return S_OK;
    }
    m.handled = false;
    if (m.session) {
        // the copy path presents on the real chain, which the session's presenter must not hold meanwhile
        m.DestroySession("an interval it did not take");
        if (++m.stats.session_failures >= kMaxSessionFailures && !m.stats.gave_up) {
            m.stats.gave_up = true;
            m.Log("fg host: frame generation stays off for this run (3 session failures); the game's frames are "
                  "presented as it asks");
        }
    }
    HRESULT hr;
    if (!m.facade.empty()) {
        if (m.chain_flags & DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT) {
            HANDLE w = m.inner ? m.inner->GetFrameLatencyWaitableObject() : nullptr;
            if (w) {
                WaitForSingleObject(w, kLatencyWaitMs);
                CloseHandle(w);
            }
        }
        m.CopyFacadeToChain();
        ++m.stats.copy_presents;
        m.AdvanceFacade();
    }
    hr = call.present1 ? inner.Present1(call.sync_interval, call.flags, call.params)
                       : inner.Present(call.sync_interval, call.flags);
    return hr;
}

void FgAsiSessionHost::BeforeResize() {
    std::lock_guard<std::recursive_mutex> lock(impl_->mu);
    impl_->DestroySession("resize");
    impl_->WaitCopy(impl_->copy_value, kDestroyWaitMs);
    impl_->ReleaseFacade();
}

void FgAsiSessionHost::AfterResize(HRESULT) {
    std::lock_guard<std::recursive_mutex> lock(impl_->mu);
    if (!impl_->CreateFacade()) impl_->Log("fg host: facade not recreated after the resize");
}

void FgAsiSessionHost::BeforeFullscreen(BOOL) {
    std::lock_guard<std::recursive_mutex> lock(impl_->mu);
    impl_->DestroySession("fullscreen change");
}

void FgAsiSessionHost::AfterFullscreen(BOOL, HRESULT) {}

bool FgAsiSessionHost::GetBuffer(UINT buffer, REFIID riid, void** out, HRESULT* hr) {
    std::lock_guard<std::recursive_mutex> lock(impl_->mu);
    if (impl_->facade.empty() || !out || !hr) return false;
    if (buffer >= impl_->facade.size()) {
        *hr = DXGI_ERROR_INVALID_CALL;
        return true;
    }
    *hr = impl_->facade[buffer]->QueryInterface(riid, out);
    return true;
}

bool FgAsiSessionHost::CurrentBackBufferIndex(UINT* index) {
    std::lock_guard<std::recursive_mutex> lock(impl_->mu);
    if (impl_->facade.empty() || !index) return false;
    *index = impl_->facade_index;
    return true;
}

bool FgAsiSessionHost::FrameLatencyWaitableObject(HANDLE* handle) {
    std::lock_guard<std::recursive_mutex> lock(impl_->mu);
    if (impl_->facade.empty() || !impl_->latency || !handle) return false;
    // the caller closes what it gets, as with the chain's own object
    HANDLE dup = nullptr;
    if (!DuplicateHandle(GetCurrentProcess(), impl_->latency, GetCurrentProcess(), &dup, 0, FALSE,
                         DUPLICATE_SAME_ACCESS))
        return false;
    *handle = dup;
    return true;
}

void FgAsiSessionHost::Detached() {
    std::lock_guard<std::recursive_mutex> lock(impl_->mu);
    impl_->DestroySession("chain released");
    impl_->WaitCopy(impl_->copy_value, kDestroyWaitMs);
    impl_->ReleaseFacade();
    impl_->inner = nullptr;
}

void FgAsiSessionHost::Attached(IDXGISwapChain4* inner, IUnknown* queue) {
    Impl& m = *impl_;
    std::lock_guard<std::recursive_mutex> lock(m.mu);
    if (!inner || !queue) return;
    ComPtr<ID3D12CommandQueue> q;
    ComPtr<ID3D12Device> dev;
    if (FAILED(queue->QueryInterface(IID_PPV_ARGS(&q))) || FAILED(q->GetDevice(IID_PPV_ARGS(&dev)))) {
        m.Log("fg host: the chain's creation queue is not a D3D12 queue; frame generation is not taken");
        return;
    }
    m.inner = inner;
    m.queue = q;
    m.device = dev;
    if (FAILED(inner->GetHwnd(&m.hwnd))) m.hwnd = nullptr;
    if (!m.latency) m.latency = CreateSemaphoreW(nullptr, 1, 16, nullptr);
    if (m.core) m.core->set_device(dev.Get(), q.Get());
    if (!m.CreateFacade()) {
        m.Log("fg host: facade buffers not created; the chain's own buffers stay in use");
        return;
    }
    char line[128];
    wsprintfA(line, "fg host: chain attached, %u x %u, format %u, %u buffers (facade)", m.width, m.height,
              static_cast<unsigned>(m.chain_format), static_cast<unsigned>(m.facade.size()));
    m.Log(line);
}

UINT FgAsiSessionHost::ExtraSwapChainFlags(const DXGI_SWAP_CHAIN_DESC1&) {
    // the DXGI flip presenters pace on the chain's frame-latency object; tearing is added only when the system has it
    UINT f = 0;
    const FgPresenterChoice p = impl_->cfg.settings.presenter;
    if (p == FgPresenterChoice::Auto || p == FgPresenterChoice::DxgiTearing || p == FgPresenterChoice::DxgiVsync)
        f |= DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;
    std::lock_guard<std::recursive_mutex> lock(impl_->mu);
    if ((p == FgPresenterChoice::Auto || p == FgPresenterChoice::DxgiTearing) && impl_->tearing_supported)
        f |= DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;
    return f;
}

}  // namespace win32
}  // namespace wgfgp
