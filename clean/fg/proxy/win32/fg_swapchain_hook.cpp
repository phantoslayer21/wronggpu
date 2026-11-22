// SPDX-License-Identifier: MIT
// WrongGPU frame generation (issue #96, F8b): the DXGI swap-chain hook seam. See fg_swapchain_hook.h.
// Written for this project.
#include "fg_swapchain_hook.h"

#include <map>
#include <new>
#include <vector>

// Test-only mutants (CMakeLists.txt; fg_asi_transport.cpp has 1, 2, 3, 7):
//  4 a retry begins a new interval  5 a test present reaches the callback  6 the present owner is not claimed
//  8 a failing callback stays attached
#ifndef WG_FG_WIN32_MUTANT
#define WG_FG_WIN32_MUTANT 0
#endif

namespace wgfgp {
namespace win32 {

namespace {

thread_local int t_callback_depth = 0;  // > 0: this thread is inside a callback (its presents bypass the hook)
thread_local int t_creating = 0;        // > 0: inside the original CreateSwapChainForHwnd (no second wrap)

std::mutex& RegistryMutex() {
    static std::mutex m;
    return m;
}
std::map<IDXGISwapChain4*, FgSwapChain*>& Registry() {
    static std::map<IDXGISwapChain4*, FgSwapChain*> r;
    return r;
}

class DepthScope {
public:
    DepthScope() { ++t_callback_depth; }
    ~DepthScope() { --t_callback_depth; }
    DepthScope(const DepthScope&) = delete;
    DepthScope& operator=(const DepthScope&) = delete;
};

bool PatchPointer(void** slot, void* value) {
    DWORD old = 0;
    if (!VirtualProtect(slot, sizeof(void*), PAGE_READWRITE, &old)) return false;
    InterlockedExchangePointer(slot, value);
    DWORD ignored = 0;
    VirtualProtect(slot, sizeof(void*), old, &ignored);
    FlushInstructionCache(GetCurrentProcess(), slot, sizeof(void*));
    return true;
}

bool IsChainInterface(REFIID riid) {
    return riid == __uuidof(IUnknown) || riid == __uuidof(IDXGIObject) || riid == __uuidof(IDXGIDeviceSubObject) ||
           riid == __uuidof(IDXGISwapChain) || riid == __uuidof(IDXGISwapChain1) || riid == __uuidof(IDXGISwapChain2) ||
           riid == __uuidof(IDXGISwapChain3) || riid == __uuidof(IDXGISwapChain4);
}

}  // namespace

uint64_t FgSwapChainId(IUnknown* chain) {
    if (!chain) return 0;
    IUnknown* id = nullptr;
    if (FAILED(chain->QueryInterface(__uuidof(IUnknown), reinterpret_cast<void**>(&id))) || !id) return 0;
    const uint64_t v = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(id));
    id->Release();
    return v;
}

// ---- the inner presenter handed to the callback ----------------------------------------------------------------------

class FgSwapChain::Inner_ final : public IFgInnerPresenter {
public:
    explicit Inner_(FgSwapChain* sc) : sc_(sc) {}
    HRESULT Present(UINT sync_interval, UINT flags) override { return sc_->RawPresent(false, sync_interval, flags, nullptr); }
    HRESULT Present1(UINT sync_interval, UINT flags, const DXGI_PRESENT_PARAMETERS* params) override {
        return sc_->RawPresent(true, sync_interval, flags, params);
    }
    IDXGISwapChain4* Inner() override { return sc_->inner_; }

private:
    FgSwapChain* sc_;
};

// ---- construction ---------------------------------------------------------------------------------------------------

HRESULT FgSwapChain::Create(IDXGISwapChain1* inner, IUnknown* queue, FgLogFn log, void* log_ctx, FgSwapChain** out) {
    if (!out) return E_POINTER;
    *out = nullptr;
    if (!inner) return E_POINTER;
    IDXGISwapChain4* four = nullptr;
    if (FAILED(inner->QueryInterface(__uuidof(IDXGISwapChain4), reinterpret_cast<void**>(&four))) || !four)
        return E_NOINTERFACE;
    FgSwapChain* w = new (std::nothrow) FgSwapChain(four, queue, log, log_ctx);
    if (!w) {
        four->Release();
        return E_OUTOFMEMORY;
    }
    inner->Release();  // the caller's reference is now the wrapper's (held as `four`)
    *out = w;
    return S_OK;
}

FgSwapChain::FgSwapChain(IDXGISwapChain4* inner, IUnknown* queue, FgLogFn log, void* log_ctx)
    : inner_(inner), queue_(queue), id_(FgSwapChainId(inner)), log_(log), log_ctx_(log_ctx) {
    if (queue_) queue_->AddRef();
    std::lock_guard<std::mutex> lock(RegistryMutex());
    Registry()[static_cast<IDXGISwapChain4*>(this)] = this;
}

FgSwapChain::~FgSwapChain() {
    {
        std::lock_guard<std::mutex> lock(RegistryMutex());
        Registry().erase(static_cast<IDXGISwapChain4*>(this));
    }
    inner_->Release();
    if (queue_) queue_->Release();
}

FgSwapChain* FgSwapChain::FromInterface(IUnknown* p) {
    if (!p) return nullptr;
    IDXGISwapChain4* four = nullptr;
    if (FAILED(p->QueryInterface(__uuidof(IDXGISwapChain4), reinterpret_cast<void**>(&four))) || !four) return nullptr;
    FgSwapChain* w = nullptr;
    {
        std::lock_guard<std::mutex> lock(RegistryMutex());
        auto it = Registry().find(four);
        if (it != Registry().end()) w = it->second;
    }
    four->Release();
    return w;
}

long FgSwapChain::LiveCount() {
    std::lock_guard<std::mutex> lock(RegistryMutex());
    return static_cast<long>(Registry().size());
}

size_t FgSwapChain::DetachCallback(IFgPresentCallback* callback) {
    if (!callback) return 0;
    std::vector<FgSwapChain*> held;
    {
        std::lock_guard<std::mutex> lock(RegistryMutex());
        for (auto& e : Registry()) {
            FgSwapChain* w = e.second;
            long n = w->refs_.load();
            while (n > 0 && !w->refs_.compare_exchange_weak(n, n + 1)) {
            }
            if (n > 0) held.push_back(w);  // a reference of our own: it stays alive until the Release below
        }
    }
    size_t detached = 0;
    for (FgSwapChain* w : held) {
        {
            std::lock_guard<std::mutex> lock(w->mu_);
            if (w->callback_ == callback) {
                w->DetachLocked(true);
                ++detached;
            }
        }
        w->Release();
    }
    return detached;
}

void FgSwapChain::Log(const std::string& line) const {
    if (log_) log_(log_ctx_, line.c_str());
}

// ---- ownership ------------------------------------------------------------------------------------------------------

uint32_t FgSwapChain::Attach(IFgPresentCallback* callback, FgProxy* core, FgPresentOwners* owners, uint32_t owner) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!callback || owner == WG_FG_OWNER_NONE) return WG_FG_REASON_NULL_ARGUMENT;
    if (callback_) return WG_FG_REASON_DOUBLE_ATTACH;
    FgPresentOwners* reg = owners ? owners : &FgProcessPresentOwners();
#if WG_FG_WIN32_MUTANT == 6
    const uint32_t r = WG_FG_REASON_NONE;
    (void)core;
#else
    const uint32_t r = core ? core->claim_present(id_, owner) : reg->claim(id_, owner);
#endif
    if (r != WG_FG_REASON_NONE) {
        Log("fg swap chain: present owner not claimed (reason " + std::to_string(r) + "); the chain presents as before");
        return r;
    }
    callback_ = callback;
    core_ = core;
    owners_ = reg;
    owner_ = owner;
    pending_ = false;
    Log("fg swap chain: present owner claimed");
    return WG_FG_REASON_NONE;
}

void FgSwapChain::DetachLocked(bool notify) {
    IFgPresentCallback* cb = callback_;
    if (!cb) return;
    callback_ = nullptr;
    pending_ = false;
    if (core_) core_->release_present(id_, owner_);
    else if (owners_) owners_->release(id_, owner_);
    core_ = nullptr;
    owner_ = WG_FG_OWNER_NONE;
    if (notify) {
        try {
            cb->Detached();
        } catch (...) {
            ++stats_.callback_faults;
        }
    }
}

void FgSwapChain::Detach() {
    std::lock_guard<std::mutex> lock(mu_);
    DetachLocked(true);
}

bool FgSwapChain::Attached() const {
    std::lock_guard<std::mutex> lock(mu_);
    return callback_ != nullptr;
}

uint64_t FgSwapChain::Epoch() const {
    std::lock_guard<std::mutex> lock(mu_);
    return epoch_;
}

FgSwapChainStats FgSwapChain::Stats() const {
    std::lock_guard<std::mutex> lock(mu_);
    return stats_;
}

void FgSwapChain::Fault(const char* where) {
    ++stats_.callback_faults;
    Log(std::string("fg swap chain: the present callback failed in ") + where +
        "; it is detached and the chain presents as before");
#if WG_FG_WIN32_MUTANT != 8
    DetachLocked(false);
#endif
}

template <typename F>
bool FgSwapChain::Guard(const char* where, F&& f) {
    try {
        f();
        return true;
    } catch (...) {
        Fault(where);
        return false;
    }
}

// ---- IUnknown -------------------------------------------------------------------------------------------------------

HRESULT STDMETHODCALLTYPE FgSwapChain::QueryInterface(REFIID riid, void** ppv) {
    if (!ppv) return E_POINTER;
    *ppv = nullptr;
    if (IsChainInterface(riid)) {
        *ppv = static_cast<IDXGISwapChain4*>(this);
        AddRef();
        return S_OK;
    }
    return inner_->QueryInterface(riid, ppv);
}

ULONG STDMETHODCALLTYPE FgSwapChain::AddRef() { return static_cast<ULONG>(++refs_); }

ULONG STDMETHODCALLTYPE FgSwapChain::Release() {
    const long n = --refs_;
    if (n == 0) {
        {
            std::lock_guard<std::mutex> lock(mu_);
            DetachLocked(true);  // the present owner goes with the chain
        }
        delete this;
    }
    return static_cast<ULONG>(n < 0 ? 0 : n);
}

// ---- present --------------------------------------------------------------------------------------------------------

// Straight to the real chain. Called with mu_ held by this thread (the callback's own presents included).
HRESULT FgSwapChain::RawPresent(bool present1, UINT sync_interval, UINT flags, const DXGI_PRESENT_PARAMETERS* params) {
    const HRESULT hr = present1 ? inner_->Present1(sync_interval, flags, params) : inner_->Present(sync_interval, flags);
    ++stats_.inner_presents;
    if (!(flags & DXGI_PRESENT_TEST)) ++presents_in_call_;
    return hr;
}

HRESULT FgSwapChain::PresentCommon(bool present1, UINT sync_interval, UINT flags, const DXGI_PRESENT_PARAMETERS* params) {
    if (t_callback_depth > 0) {
        // an internal presenter's present re-entering through the game-facing pointer: the lock is this thread's
        ++stats_.bypass_presents;
        return RawPresent(present1, sync_interval, flags, params);
    }
    std::lock_guard<std::mutex> lock(mu_);
    if ((flags & DXGI_PRESENT_TEST) && WG_FG_WIN32_MUTANT != 5) {  // occlusion test: no interval, no history, no callback
        ++stats_.test_presents;
        return RawPresent(present1, sync_interval, flags, params);
    }
    if (!callback_) {
        ++stats_.forwarded_presents;
        const HRESULT hr = RawPresent(present1, sync_interval, flags, params);
        if (SUCCEEDED(hr)) {
            ++interval_;
            ++stats_.game_presents;
        }
        return hr;
    }
    FgPresentCall call;
    call.swapchain_id = id_;
    call.swapchain_epoch = epoch_;
    call.sync_interval = sync_interval;
    call.flags = flags;
    call.params = params;
    call.present1 = present1;
    call.retry = pending_;
    if (!pending_ || WG_FG_WIN32_MUTANT == 4) {
        ++interval_;
        ++stats_.begin_intervals;
        call.interval = interval_;
        IFgPresentCallback* cb = callback_;
        bool begun;
        {
            DepthScope depth;
            begun = Guard("BeginInterval", [&]() { cb->BeginInterval(call); });
        }
        if (!begun) {  // detached: this interval is shown as the game gave it, once
            const HRESULT hr = RawPresent(present1, sync_interval, flags, params);
            if (SUCCEEDED(hr)) ++stats_.game_presents;
            return hr;
        }
        pending_ = true;
    } else {
        ++stats_.retries;
        call.interval = interval_;
    }
    presents_in_call_ = 0;
    HRESULT hr = DXGI_ERROR_INVALID_CALL;
    Inner_ inner(this);
    IFgPresentCallback* cb = callback_;
    bool ok;
    {
        DepthScope depth;
        ok = Guard("PresentInterval", [&]() { hr = cb->PresentInterval(call, inner); });
    }
    if (!ok) {
        // the callback is gone; never present this interval twice
        hr = presents_in_call_ == 0 ? RawPresent(present1, sync_interval, flags, params) : S_OK;
        if (SUCCEEDED(hr)) ++stats_.game_presents;
        return hr;
    }
    if (hr == DXGI_ERROR_WAS_STILL_DRAWING) return hr;  // the interval stays open for the game's retry
    pending_ = false;
    if (SUCCEEDED(hr)) ++stats_.game_presents;
    return hr;
}

HRESULT STDMETHODCALLTYPE FgSwapChain::Present(UINT sync_interval, UINT flags) {
    return PresentCommon(false, sync_interval, flags, nullptr);
}

HRESULT STDMETHODCALLTYPE FgSwapChain::Present1(UINT sync_interval, UINT flags, const DXGI_PRESENT_PARAMETERS* params) {
    return PresentCommon(true, sync_interval, flags, params);
}

// ---- resize, fullscreen, color space -------------------------------------------------------------------------------

HRESULT STDMETHODCALLTYPE FgSwapChain::ResizeBuffers(UINT count, UINT width, UINT height, DXGI_FORMAT format, UINT flags) {
    if (t_callback_depth > 0) return inner_->ResizeBuffers(count, width, height, format, flags);
    std::lock_guard<std::mutex> lock(mu_);
    pending_ = false;
    ++stats_.resizes;
    if (callback_) {
        IFgPresentCallback* cb = callback_;
        DepthScope depth;
        Guard("BeforeResize", [&]() { cb->BeforeResize(); });
    }
    const HRESULT hr = inner_->ResizeBuffers(count, width, height, format, flags);
    if (SUCCEEDED(hr)) ++epoch_;
    if (callback_) {
        IFgPresentCallback* cb = callback_;
        DepthScope depth;
        Guard("AfterResize", [&]() { cb->AfterResize(hr); });
    }
    return hr;
}

HRESULT STDMETHODCALLTYPE FgSwapChain::ResizeBuffers1(UINT count, UINT width, UINT height, DXGI_FORMAT format, UINT flags,
                                                      const UINT* node_masks, IUnknown* const* queues) {
    if (t_callback_depth > 0) return inner_->ResizeBuffers1(count, width, height, format, flags, node_masks, queues);
    std::lock_guard<std::mutex> lock(mu_);
    pending_ = false;
    ++stats_.resizes;
    if (callback_) {
        IFgPresentCallback* cb = callback_;
        DepthScope depth;
        Guard("BeforeResize", [&]() { cb->BeforeResize(); });
    }
    const HRESULT hr = inner_->ResizeBuffers1(count, width, height, format, flags, node_masks, queues);
    if (SUCCEEDED(hr)) ++epoch_;
    if (callback_) {
        IFgPresentCallback* cb = callback_;
        DepthScope depth;
        Guard("AfterResize", [&]() { cb->AfterResize(hr); });
    }
    return hr;
}

HRESULT STDMETHODCALLTYPE FgSwapChain::SetFullscreenState(BOOL fullscreen, IDXGIOutput* target) {
    if (t_callback_depth > 0) return inner_->SetFullscreenState(fullscreen, target);
    std::lock_guard<std::mutex> lock(mu_);
    pending_ = false;
    if (callback_) {
        IFgPresentCallback* cb = callback_;
        DepthScope depth;
        Guard("BeforeFullscreen", [&]() { cb->BeforeFullscreen(fullscreen); });
    }
    const HRESULT hr = inner_->SetFullscreenState(fullscreen, target);
    if (SUCCEEDED(hr)) ++epoch_;
    if (callback_) {
        IFgPresentCallback* cb = callback_;
        DepthScope depth;
        Guard("AfterFullscreen", [&]() { cb->AfterFullscreen(fullscreen, hr); });
    }
    return hr;
}

HRESULT STDMETHODCALLTYPE FgSwapChain::SetColorSpace1(DXGI_COLOR_SPACE_TYPE space) {
    if (t_callback_depth > 0) return inner_->SetColorSpace1(space);
    std::lock_guard<std::mutex> lock(mu_);
    const HRESULT hr = inner_->SetColorSpace1(space);
    if (SUCCEEDED(hr) && callback_) {
        IFgPresentCallback* cb = callback_;
        DepthScope depth;
        Guard("ColorSpaceChanged", [&]() { cb->ColorSpaceChanged(space); });
    }
    return hr;
}

// ---- facade seam ----------------------------------------------------------------------------------------------------

HRESULT STDMETHODCALLTYPE FgSwapChain::GetBuffer(UINT buffer, REFIID riid, void** surface) {
    if (t_callback_depth > 0) return inner_->GetBuffer(buffer, riid, surface);
    std::lock_guard<std::mutex> lock(mu_);
    if (callback_) {
        IFgPresentCallback* cb = callback_;
        HRESULT hr = S_OK;
        bool served = false;
        DepthScope depth;
        if (Guard("GetBuffer", [&]() { served = cb->GetBuffer(buffer, riid, surface, &hr); }) && served) return hr;
    }
    return inner_->GetBuffer(buffer, riid, surface);
}

UINT STDMETHODCALLTYPE FgSwapChain::GetCurrentBackBufferIndex() {
    if (t_callback_depth > 0) return inner_->GetCurrentBackBufferIndex();
    std::lock_guard<std::mutex> lock(mu_);
    if (callback_) {
        IFgPresentCallback* cb = callback_;
        UINT index = 0;
        bool served = false;
        DepthScope depth;
        if (Guard("CurrentBackBufferIndex", [&]() { served = cb->CurrentBackBufferIndex(&index); }) && served)
            return index;
    }
    return inner_->GetCurrentBackBufferIndex();
}

HANDLE STDMETHODCALLTYPE FgSwapChain::GetFrameLatencyWaitableObject() {
    if (t_callback_depth > 0) return inner_->GetFrameLatencyWaitableObject();
    std::lock_guard<std::mutex> lock(mu_);
    if (callback_) {
        IFgPresentCallback* cb = callback_;
        HANDLE h = nullptr;
        bool served = false;
        DepthScope depth;
        if (Guard("FrameLatencyWaitableObject", [&]() { served = cb->FrameLatencyWaitableObject(&h); }) && served)
            return h;
    }
    return inner_->GetFrameLatencyWaitableObject();
}

// ---- forwarded as they are ------------------------------------------------------------------------------------------

HRESULT STDMETHODCALLTYPE FgSwapChain::SetPrivateData(REFGUID name, UINT size, const void* data) {
    return inner_->SetPrivateData(name, size, data);
}
HRESULT STDMETHODCALLTYPE FgSwapChain::SetPrivateDataInterface(REFGUID name, const IUnknown* unknown) {
    return inner_->SetPrivateDataInterface(name, unknown);
}
HRESULT STDMETHODCALLTYPE FgSwapChain::GetPrivateData(REFGUID name, UINT* size, void* data) {
    return inner_->GetPrivateData(name, size, data);
}
HRESULT STDMETHODCALLTYPE FgSwapChain::GetParent(REFIID riid, void** parent) { return inner_->GetParent(riid, parent); }
HRESULT STDMETHODCALLTYPE FgSwapChain::GetDevice(REFIID riid, void** device) { return inner_->GetDevice(riid, device); }
HRESULT STDMETHODCALLTYPE FgSwapChain::GetFullscreenState(BOOL* fullscreen, IDXGIOutput** target) {
    return inner_->GetFullscreenState(fullscreen, target);
}
HRESULT STDMETHODCALLTYPE FgSwapChain::GetDesc(DXGI_SWAP_CHAIN_DESC* desc) { return inner_->GetDesc(desc); }
HRESULT STDMETHODCALLTYPE FgSwapChain::ResizeTarget(const DXGI_MODE_DESC* params) { return inner_->ResizeTarget(params); }
HRESULT STDMETHODCALLTYPE FgSwapChain::GetContainingOutput(IDXGIOutput** output) {
    return inner_->GetContainingOutput(output);
}
HRESULT STDMETHODCALLTYPE FgSwapChain::GetFrameStatistics(DXGI_FRAME_STATISTICS* stats) {
    return inner_->GetFrameStatistics(stats);
}
HRESULT STDMETHODCALLTYPE FgSwapChain::GetLastPresentCount(UINT* count) { return inner_->GetLastPresentCount(count); }
HRESULT STDMETHODCALLTYPE FgSwapChain::GetDesc1(DXGI_SWAP_CHAIN_DESC1* desc) { return inner_->GetDesc1(desc); }
HRESULT STDMETHODCALLTYPE FgSwapChain::GetFullscreenDesc(DXGI_SWAP_CHAIN_FULLSCREEN_DESC* desc) {
    return inner_->GetFullscreenDesc(desc);
}
HRESULT STDMETHODCALLTYPE FgSwapChain::GetHwnd(HWND* hwnd) { return inner_->GetHwnd(hwnd); }
HRESULT STDMETHODCALLTYPE FgSwapChain::GetCoreWindow(REFIID riid, void** unk) { return inner_->GetCoreWindow(riid, unk); }
BOOL STDMETHODCALLTYPE FgSwapChain::IsTemporaryMonoSupported() { return inner_->IsTemporaryMonoSupported(); }
HRESULT STDMETHODCALLTYPE FgSwapChain::GetRestrictToOutput(IDXGIOutput** output) {
    return inner_->GetRestrictToOutput(output);
}
HRESULT STDMETHODCALLTYPE FgSwapChain::SetBackgroundColor(const DXGI_RGBA* color) {
    return inner_->SetBackgroundColor(color);
}
HRESULT STDMETHODCALLTYPE FgSwapChain::GetBackgroundColor(DXGI_RGBA* color) { return inner_->GetBackgroundColor(color); }
HRESULT STDMETHODCALLTYPE FgSwapChain::SetRotation(DXGI_MODE_ROTATION rotation) { return inner_->SetRotation(rotation); }
HRESULT STDMETHODCALLTYPE FgSwapChain::GetRotation(DXGI_MODE_ROTATION* rotation) { return inner_->GetRotation(rotation); }
HRESULT STDMETHODCALLTYPE FgSwapChain::SetSourceSize(UINT width, UINT height) { return inner_->SetSourceSize(width, height); }
HRESULT STDMETHODCALLTYPE FgSwapChain::GetSourceSize(UINT* width, UINT* height) {
    return inner_->GetSourceSize(width, height);
}
HRESULT STDMETHODCALLTYPE FgSwapChain::SetMaximumFrameLatency(UINT latency) {
    return inner_->SetMaximumFrameLatency(latency);
}
HRESULT STDMETHODCALLTYPE FgSwapChain::GetMaximumFrameLatency(UINT* latency) {
    return inner_->GetMaximumFrameLatency(latency);
}
HRESULT STDMETHODCALLTYPE FgSwapChain::SetMatrixTransform(const DXGI_MATRIX_3X2_F* matrix) {
    return inner_->SetMatrixTransform(matrix);
}
HRESULT STDMETHODCALLTYPE FgSwapChain::GetMatrixTransform(DXGI_MATRIX_3X2_F* matrix) {
    return inner_->GetMatrixTransform(matrix);
}
HRESULT STDMETHODCALLTYPE FgSwapChain::CheckColorSpaceSupport(DXGI_COLOR_SPACE_TYPE space, UINT* support) {
    return inner_->CheckColorSpaceSupport(space, support);
}
HRESULT STDMETHODCALLTYPE FgSwapChain::SetHDRMetaData(DXGI_HDR_METADATA_TYPE type, UINT size, void* data) {
    return inner_->SetHDRMetaData(type, size, data);
}

// ---- creation hook --------------------------------------------------------------------------------------------------

namespace {
std::mutex g_hook_mu;
std::atomic<FgCreateForHwndFn> g_orig_create{nullptr};  // kept after Uninstall: a late call still forwards
void** g_create_slot = nullptr;
std::atomic<IFgSwapChainSink*> g_sink{nullptr};
std::atomic<FgLogFn> g_hook_log{nullptr};
std::atomic<void*> g_hook_log_ctx{nullptr};
std::atomic<uint64_t> g_wrapped{0};

void HookLog(const std::string& line) {
    FgLogFn f = g_hook_log.load();
    if (f) f(g_hook_log_ctx.load(), line.c_str());
}

class CreatingScope {
public:
    CreatingScope() { ++t_creating; }
    ~CreatingScope() { --t_creating; }
    CreatingScope(const CreatingScope&) = delete;
    CreatingScope& operator=(const CreatingScope&) = delete;
};
}  // namespace

bool FgSwapChainHook::Install(IDXGIFactory2* factory, IFgSwapChainSink* sink, FgLogFn log, void* log_ctx) {
    std::lock_guard<std::mutex> lock(g_hook_mu);
    if (!factory || g_create_slot) return false;
    void** table = *reinterpret_cast<void***>(factory);
    if (!table) return false;
    void** slot = &table[kCreateForHwndSlot];
    void* original = *slot;
    if (!original || original == reinterpret_cast<void*>(&FgSwapChainHook::CreateForHwnd)) return false;
    g_hook_log.store(log);
    g_hook_log_ctx.store(log_ctx);
    g_sink.store(sink);
    g_orig_create.store(reinterpret_cast<FgCreateForHwndFn>(original));
    if (!PatchPointer(slot, reinterpret_cast<void*>(&FgSwapChainHook::CreateForHwnd))) {
        HookLog("fg swap chain: the factory's method table could not be written; no creation hook");
        return false;
    }
    g_create_slot = slot;
    g_wrapped.store(0);
    HookLog("fg swap chain: creation hook installed");
    return true;
}

void FgSwapChainHook::Uninstall() {
    std::lock_guard<std::mutex> lock(g_hook_mu);
    if (!g_create_slot) return;
    if (*g_create_slot == reinterpret_cast<void*>(&FgSwapChainHook::CreateForHwnd))
        PatchPointer(g_create_slot, reinterpret_cast<void*>(g_orig_create.load()));
    g_create_slot = nullptr;
    g_sink.store(nullptr);
    HookLog("fg swap chain: creation hook removed");
}

bool FgSwapChainHook::Installed() {
    std::lock_guard<std::mutex> lock(g_hook_mu);
    return g_create_slot != nullptr;
}

uint64_t FgSwapChainHook::Wrapped() { return g_wrapped.load(); }

HRESULT STDMETHODCALLTYPE FgSwapChainHook::CreateForHwnd(IDXGIFactory2* self, IUnknown* device, HWND hwnd,
                                                         const DXGI_SWAP_CHAIN_DESC1* desc,
                                                         const DXGI_SWAP_CHAIN_FULLSCREEN_DESC* fullscreen,
                                                         IDXGIOutput* restrict_to, IDXGISwapChain1** out) {
    const FgCreateForHwndFn orig = g_orig_create.load();
    if (!orig) return DXGI_ERROR_INVALID_CALL;
    if (t_creating > 0) return orig(self, device, hwnd, desc, fullscreen, restrict_to, out);
    HRESULT hr;
    {
        CreatingScope creating;
        hr = orig(self, device, hwnd, desc, fullscreen, restrict_to, out);
    }
    if (FAILED(hr) || !out || !*out) return hr;
    FgSwapChain* w = nullptr;
    if (FAILED(FgSwapChain::Create(*out, device, g_hook_log.load(), g_hook_log_ctx.load(), &w)) || !w)
        return hr;  // not an IDXGISwapChain4: the game keeps the real chain
    *out = static_cast<IDXGISwapChain1*>(w);
    g_wrapped.fetch_add(1);
    IFgSwapChainSink* sink = g_sink.load();
    if (sink) {
        try {
            sink->Created(w, hwnd, desc);
        } catch (...) {
            HookLog("fg swap chain: the creation sink failed; the chain stays a pass-through");
        }
    }
    return hr;
}

}  // namespace win32
}  // namespace wgfgp
