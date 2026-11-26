// SPDX-License-Identifier: MIT
// WrongGPU frame generation (issue #96, F8b): the DXGI swap-chain hook seam for FG presentation.
//
// A per-chain wrapper installed at swap-chain creation (FG-RUNTIME-DESIGN section 3, "Intercept at swap-chain creation
// and present"). The game holds the wrapper as its IDXGISwapChain1..4; the wrapper holds the real chain.
//
//  - COM IDENTITY. QueryInterface for IUnknown, IDXGIObject, IDXGIDeviceSubObject and IDXGISwapChain..4 returns the
//    wrapper itself, so every swap-chain interface the game gets leads back to one identity; other interfaces are the
//    real chain's. The wrapper keeps its own reference count and one reference on the real chain, released (with the
//    present owner) when the game's last reference goes.
//  - ONE PRESENT OWNER PER CHAIN. attach() claims the chain through the adapter core (FgProxy::claim_present: refused
//    unless FG is taken and no other presenter holds the chain) or, without a core, through the present-owner registry
//    (fg_proxy_core.h FgPresentOwners). Refused: the wrapper stays a pure pass-through.
//  - PRESENT. Present and Present1: a test-only present (DXGI_PRESENT_TEST) always goes straight to the real chain and
//    is never an interval. Without an attached callback every present goes straight to the real chain. With one, the
//    callback's BeginInterval runs once per game interval (history rotation, generation submit) and PresentInterval
//    performs the physical presents through IFgInnerPresenter. A present that returns DXGI_ERROR_WAS_STILL_DRAWING
//    (DXGI_PRESENT_DO_NOT_WAIT) leaves the interval open: the game's retry calls PresentInterval again (retry = true)
//    without a second BeginInterval, so history never rotates and nothing is submitted twice. Flags are passed as the
//    game gave them; the HRESULT the game sees (occluded status, DXGI errors) is the real chain's, never altered.
//  - RE-ENTRANCY. While a callback runs, a present that re-enters the wrapper on that thread (an internal presenter that
//    holds the game-facing pointer) goes straight to the real chain. IFgInnerPresenter always does.
//  - RESIZE, FULLSCREEN, COLOR SPACE. ResizeBuffers / ResizeBuffers1 / SetFullscreenState close any open interval, tell
//    the callback before (release buffer references, stop admission) and after (the real result), and bump the chain
//    epoch on success. SetColorSpace1 forwards and reports a successful change. The result is always the real chain's.
//  - FACADE SEAM. GetBuffer, GetCurrentBackBufferIndex and GetFrameLatencyWaitableObject may be served by the callback
//    (a facade that owns the game-facing ring); otherwise they forward.
//  - FAILURE. Every callback call is guarded: a callback that throws is detached (owner released, logged), and the
//    present it was serving goes to the real chain once unless the callback had already presented.
//  - CREATION. FgSwapChainHook patches the CreateSwapChainForHwnd slot of a factory's method table (the original kept
//    and always called first) and wraps the chains it creates; a sink decides per chain whether to attach.
//  - PRESENT QUEUE. The sink may name a queue of its own to create the real chain on (IFgSwapChainSink::CreationQueue):
//    every present of the real chain then runs on that queue, never behind the game's rendering on the game's queue.
//    The game's queue stays the wrapper's Queue(); the present queue is PresentQueue() (a reference kept), and a
//    ResizeBuffers1 that names queues is given the present queue for every buffer. A creation that fails with it is
//    retried once with the game's queue (the chain then has no present queue).
// Written for this project.
#pragma once
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dxgi1_6.h>

#include <stdint.h>

#include <atomic>
#include <mutex>
#include <string>

#include "../fg_adapter_abi.h"
#include "../fg_proxy_core.h"

namespace wgfgp {
namespace win32 {

// The identity of a real swap chain as the present-owner registry keys it: its IUnknown pointer value.
uint64_t FgSwapChainId(IUnknown* chain);

struct FgPresentCall {
    uint64_t swapchain_id = 0;
    uint64_t interval = 0;         // 1-based game interval this present belongs to
    uint64_t swapchain_epoch = 0;  // bumped by every successful resize / fullscreen change
    UINT sync_interval = 0;
    UINT flags = 0;                // as the game passed them
    const DXGI_PRESENT_PARAMETERS* params = nullptr;  // Present1 only
    bool present1 = false;
    bool retry = false;            // the interval's previous attempt returned DXGI_ERROR_WAS_STILL_DRAWING
};

// The callback presents through this: straight to the real chain, never through the wrapper's hook.
class IFgInnerPresenter {
public:
    virtual HRESULT Present(UINT sync_interval, UINT flags) = 0;
    virtual HRESULT Present1(UINT sync_interval, UINT flags, const DXGI_PRESENT_PARAMETERS* params) = 0;
    virtual IDXGISwapChain4* Inner() = 0;  // no reference added

protected:
    ~IFgInnerPresenter() = default;
};

// The presenter seam (F6's IFgPresenter and F7's session sit behind it). Calls come on the game's present thread with
// the wrapper's lock held; a throw is caught by the wrapper (the callback is then detached).
class IFgPresentCallback {
public:
    virtual ~IFgPresentCallback() = default;
    virtual void BeginInterval(const FgPresentCall& call) = 0;
    virtual HRESULT PresentInterval(const FgPresentCall& call, IFgInnerPresenter& inner) = 0;
    virtual void BeforeResize() {}
    virtual void AfterResize(HRESULT) {}
    virtual void BeforeFullscreen(BOOL) {}
    virtual void AfterFullscreen(BOOL, HRESULT) {}
    virtual void ColorSpaceChanged(DXGI_COLOR_SPACE_TYPE) {}
    // facade: true = served (the wrapper returns *hr / the value)
    virtual bool GetBuffer(UINT, REFIID, void**, HRESULT*) { return false; }
    virtual bool CurrentBackBufferIndex(UINT*) { return false; }
    virtual bool FrameLatencyWaitableObject(HANDLE*) { return false; }
    // the callback will not be called again (detach, a fault, or the chain's last release)
    virtual void Detached() {}
    // F9: a chain was just attached to this callback (the real chain, its creation queue; no references added)
    virtual void Attached(IDXGISwapChain4*, IUnknown*) {}
    // F9: swap-chain flags the callback needs on a chain created while it would attach (for example the frame-latency
    // waitable object of a DXGI flip presenter); the hook ORs them into the creation and every resize. Default none.
    virtual UINT ExtraSwapChainFlags(const DXGI_SWAP_CHAIN_DESC1&) { return 0; }
    // A queue of the callback's own to create the real chain on instead of the game's queue (see PRESENT QUEUE above;
    // not owned: the callback keeps it alive). Null (the default): the game's queue.
    virtual IUnknown* CreationQueue(IUnknown* /*game_queue*/, const DXGI_SWAP_CHAIN_DESC1&) { return nullptr; }
    // The attached chain presents on this queue of the callback's (its CreationQueue answer); called before Attached.
    virtual void AttachedPresentQueue(IUnknown*) {}
};

struct FgSwapChainStats {
    uint64_t game_presents = 0;      // intervals completed (a successful present outside a retry)
    uint64_t test_presents = 0;      // DXGI_PRESENT_TEST, straight through
    uint64_t forwarded_presents = 0; // no callback attached
    uint64_t begin_intervals = 0;
    uint64_t retries = 0;            // presents that continued an open interval
    uint64_t inner_presents = 0;     // presents that reached the real chain (every path)
    uint64_t bypass_presents = 0;    // re-entrant presents sent straight through
    uint64_t callback_faults = 0;
    uint64_t resizes = 0;
};

class FgSwapChain final : public IDXGISwapChain4 {
public:
    // Wraps `inner` (one reference taken over from the caller). Fails (E_NOINTERFACE, nothing taken) when the chain is
    // not an IDXGISwapChain4. queue: the creation queue (a reference is kept), may be null.
    static HRESULT Create(IDXGISwapChain1* inner, IUnknown* queue, FgLogFn log, void* log_ctx, FgSwapChain** out);
    // The wrapper behind a game-facing pointer, or null when `p` is not one (no reference added).
    static FgSwapChain* FromInterface(IUnknown* p);
    static long LiveCount();  // wrappers alive in the process (tests)
    // Detaches `callback` from every live wrapper it is attached to (the present owner released, Detached() called);
    // the number detached. A wrapper being destroyed meanwhile is skipped (no reference is taken on one whose count
    // already reached 0). No lock is held while a wrapper detaches.
    static size_t DetachCallback(IFgPresentCallback* callback);

    // Claims the chain for `owner` and routes intervals to `callback` (not owned). core null: the registry `owners`
    // (null: the process registry). WG_FG_REASON_NONE, or the refusal (the wrapper then stays pass-through).
    uint32_t Attach(IFgPresentCallback* callback, FgProxy* core, FgPresentOwners* owners,
                    uint32_t owner = WG_FG_OWNER_SWAPCHAIN_WRAPPER);
    void Detach();
    bool Attached() const;
    uint64_t Id() const { return id_; }
    uint64_t Epoch() const;
    FgSwapChainStats Stats() const;
    IDXGISwapChain4* Inner() const { return inner_; }  // no reference added
    IUnknown* Queue() const { return queue_; }
    // The queue the real chain was created on instead of the game's (null: the game's queue); a reference is kept.
    void SetPresentQueue(IUnknown* queue);
    IUnknown* PresentQueue() const { return present_queue_; }
    // Flags the creation added to the game's (FgSwapChainHook, IFgSwapChainSink::ExtraCreateFlags): ORed into every
    // ResizeBuffers / ResizeBuffers1 so the chain keeps them.
    void SetExtraFlags(UINT flags) { extra_flags_.store(flags); }
    UINT ExtraFlags() const { return extra_flags_.load(); }
    ULONG RefCount() const { return static_cast<ULONG>(refs_.load()); }

    // ---- IUnknown ----
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppv) override;
    ULONG STDMETHODCALLTYPE AddRef() override;
    ULONG STDMETHODCALLTYPE Release() override;
    // ---- IDXGIObject ----
    HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID name, UINT size, const void* data) override;
    HRESULT STDMETHODCALLTYPE SetPrivateDataInterface(REFGUID name, const IUnknown* unknown) override;
    HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID name, UINT* size, void* data) override;
    HRESULT STDMETHODCALLTYPE GetParent(REFIID riid, void** parent) override;
    // ---- IDXGIDeviceSubObject ----
    HRESULT STDMETHODCALLTYPE GetDevice(REFIID riid, void** device) override;
    // ---- IDXGISwapChain ----
    HRESULT STDMETHODCALLTYPE Present(UINT sync_interval, UINT flags) override;
    HRESULT STDMETHODCALLTYPE GetBuffer(UINT buffer, REFIID riid, void** surface) override;
    HRESULT STDMETHODCALLTYPE SetFullscreenState(BOOL fullscreen, IDXGIOutput* target) override;
    HRESULT STDMETHODCALLTYPE GetFullscreenState(BOOL* fullscreen, IDXGIOutput** target) override;
    HRESULT STDMETHODCALLTYPE GetDesc(DXGI_SWAP_CHAIN_DESC* desc) override;
    HRESULT STDMETHODCALLTYPE ResizeBuffers(UINT count, UINT width, UINT height, DXGI_FORMAT format, UINT flags) override;
    HRESULT STDMETHODCALLTYPE ResizeTarget(const DXGI_MODE_DESC* params) override;
    HRESULT STDMETHODCALLTYPE GetContainingOutput(IDXGIOutput** output) override;
    HRESULT STDMETHODCALLTYPE GetFrameStatistics(DXGI_FRAME_STATISTICS* stats) override;
    HRESULT STDMETHODCALLTYPE GetLastPresentCount(UINT* count) override;
    // ---- IDXGISwapChain1 ----
    HRESULT STDMETHODCALLTYPE GetDesc1(DXGI_SWAP_CHAIN_DESC1* desc) override;
    HRESULT STDMETHODCALLTYPE GetFullscreenDesc(DXGI_SWAP_CHAIN_FULLSCREEN_DESC* desc) override;
    HRESULT STDMETHODCALLTYPE GetHwnd(HWND* hwnd) override;
    HRESULT STDMETHODCALLTYPE GetCoreWindow(REFIID riid, void** unk) override;
    HRESULT STDMETHODCALLTYPE Present1(UINT sync_interval, UINT flags, const DXGI_PRESENT_PARAMETERS* params) override;
    BOOL STDMETHODCALLTYPE IsTemporaryMonoSupported() override;
    HRESULT STDMETHODCALLTYPE GetRestrictToOutput(IDXGIOutput** output) override;
    HRESULT STDMETHODCALLTYPE SetBackgroundColor(const DXGI_RGBA* color) override;
    HRESULT STDMETHODCALLTYPE GetBackgroundColor(DXGI_RGBA* color) override;
    HRESULT STDMETHODCALLTYPE SetRotation(DXGI_MODE_ROTATION rotation) override;
    HRESULT STDMETHODCALLTYPE GetRotation(DXGI_MODE_ROTATION* rotation) override;
    // ---- IDXGISwapChain2 ----
    HRESULT STDMETHODCALLTYPE SetSourceSize(UINT width, UINT height) override;
    HRESULT STDMETHODCALLTYPE GetSourceSize(UINT* width, UINT* height) override;
    HRESULT STDMETHODCALLTYPE SetMaximumFrameLatency(UINT latency) override;
    HRESULT STDMETHODCALLTYPE GetMaximumFrameLatency(UINT* latency) override;
    HANDLE STDMETHODCALLTYPE GetFrameLatencyWaitableObject() override;
    HRESULT STDMETHODCALLTYPE SetMatrixTransform(const DXGI_MATRIX_3X2_F* matrix) override;
    HRESULT STDMETHODCALLTYPE GetMatrixTransform(DXGI_MATRIX_3X2_F* matrix) override;
    // ---- IDXGISwapChain3 ----
    UINT STDMETHODCALLTYPE GetCurrentBackBufferIndex() override;
    HRESULT STDMETHODCALLTYPE CheckColorSpaceSupport(DXGI_COLOR_SPACE_TYPE space, UINT* support) override;
    HRESULT STDMETHODCALLTYPE SetColorSpace1(DXGI_COLOR_SPACE_TYPE space) override;
    HRESULT STDMETHODCALLTYPE ResizeBuffers1(UINT count, UINT width, UINT height, DXGI_FORMAT format, UINT flags,
                                             const UINT* node_masks, IUnknown* const* queues) override;
    // ---- IDXGISwapChain4 ----
    HRESULT STDMETHODCALLTYPE SetHDRMetaData(DXGI_HDR_METADATA_TYPE type, UINT size, void* data) override;

private:
    class Inner_;
    FgSwapChain(IDXGISwapChain4* inner, IUnknown* queue, FgLogFn log, void* log_ctx);
    ~FgSwapChain();
    FgSwapChain(const FgSwapChain&) = delete;
    FgSwapChain& operator=(const FgSwapChain&) = delete;

    HRESULT PresentCommon(bool present1, UINT sync_interval, UINT flags, const DXGI_PRESENT_PARAMETERS* params);
    HRESULT RawPresent(bool present1, UINT sync_interval, UINT flags, const DXGI_PRESENT_PARAMETERS* params);
    void Fault(const char* where);  // lock held
    void DetachLocked(bool notify);
    void Log(const std::string& line) const;
    template <typename F>
    bool Guard(const char* where, F&& f);  // runs f; false (and Fault) on a throw

    IDXGISwapChain4* inner_;
    IUnknown* queue_;
    IUnknown* present_queue_ = nullptr;
    uint64_t id_;
    FgLogFn log_;
    void* log_ctx_;
    std::atomic<long> refs_{1};
    std::atomic<UINT> extra_flags_{0};

    mutable std::mutex mu_;
    IFgPresentCallback* callback_ = nullptr;
    FgProxy* core_ = nullptr;
    FgPresentOwners* owners_ = nullptr;
    uint32_t owner_ = WG_FG_OWNER_NONE;
    bool pending_ = false;        // an interval is open (its last attempt returned DXGI_ERROR_WAS_STILL_DRAWING)
    uint64_t interval_ = 0;
    uint64_t epoch_ = 0;
    uint64_t presents_in_call_ = 0;  // inner presents made during the current callback call
    FgSwapChainStats stats_;
};

// Decides, per created chain, whether to attach (and with what). Called on the creating thread, outside any lock; a
// throw is caught (the chain then stays a pass-through wrapper).
class IFgSwapChainSink {
public:
    virtual ~IFgSwapChainSink() = default;
    virtual void Created(FgSwapChain* chain, HWND hwnd, const DXGI_SWAP_CHAIN_DESC1* desc) = 0;
    // F9: flags to add to the creation (0: the game's description unchanged). A creation that fails with them is
    // retried once with the game's own description (the chain then carries no extra flags).
    virtual UINT ExtraCreateFlags(HWND, const DXGI_SWAP_CHAIN_DESC1&) { return 0; }
    // A queue to create the real chain on instead of the game's `game_queue` (null: the game's; see PRESENT QUEUE).
    virtual IUnknown* CreationQueue(HWND, const DXGI_SWAP_CHAIN_DESC1&, IUnknown* /*game_queue*/) { return nullptr; }
};

using FgCreateForHwndFn = HRESULT(STDMETHODCALLTYPE*)(IDXGIFactory2* self, IUnknown* device, HWND hwnd,
                                                       const DXGI_SWAP_CHAIN_DESC1* desc,
                                                       const DXGI_SWAP_CHAIN_FULLSCREEN_DESC* fullscreen,
                                                       IDXGIOutput* restrict_to, IDXGISwapChain1** out);

class FgSwapChainHook {
public:
    // Index of CreateSwapChainForHwnd in IDXGIFactory2's method table: IUnknown 3, IDXGIObject 4, IDXGIFactory 5,
    // IDXGIFactory1 2, then IsWindowedStereoEnabled.
    static constexpr unsigned kCreateForHwndSlot = 15;

    // Patches the factory class's CreateSwapChainForHwnd slot (shared by every factory of that class). One install per
    // process; false: already installed, null factory, or the table not writable. The sink is not owned.
    static bool Install(IDXGIFactory2* factory, IFgSwapChainSink* sink, FgLogFn log, void* log_ctx);
    // Puts the original back when the slot still holds ours. Chains already wrapped stay wrapped.
    static void Uninstall();
    static bool Installed();
    static uint64_t Wrapped();  // chains wrapped since Install

    // The handler body (public for the tests): the original first; on success the chain is wrapped and the sink asked.
    static HRESULT STDMETHODCALLTYPE CreateForHwnd(IDXGIFactory2* self, IUnknown* device, HWND hwnd,
                                                   const DXGI_SWAP_CHAIN_DESC1* desc,
                                                   const DXGI_SWAP_CHAIN_FULLSCREEN_DESC* fullscreen,
                                                   IDXGIOutput* restrict_to, IDXGISwapChain1** out);
};

}  // namespace win32
}  // namespace wgfgp
