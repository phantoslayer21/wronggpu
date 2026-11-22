// SPDX-License-Identifier: MIT
// WrongGPU frame generation (issue #96, F9): the FG session host of WrongGPU-FG.asi. It joins the Win32 transport
// (fg_asi_transport.h) to the FG session (clean/fg/runtime/fg_session.h), the D3D12 executor (fg_d3d12_executor.h) and
// the Windows presenters (clean/fg/presenter), for a qualified title binding with fg=on.
//
//  - PRECHECK (InitializeASI, no D3D work): the local FG model validates against the shipped map (fg_model.h) and the
//    kernel folder holds every DXIL file the executor loads for the chosen net2 path. Either missing: the ASI installs
//    nothing (no transport, nothing patched), it logs why.
//  - ROUTE FACTS. The model and the presenter facts, and the adapter (vendor, device, LUID of the first RDNA 4
//    hardware adapter), go to the adapter core before the transport writes its first import slot
//    (FgAsiConfig::core_ready): the adapter from a start-up probe (CreateDXGIFactory1 and the adapter list only, no
//    device), so the game's first "is frame generation supported" call already sees the route. The game's own DXGI
//    factory (FgAsiConfig::factory_note) confirms it, or supplies it when the probe found none. The device and queue
//    come when the game's swap chain is attached.
//  - DEVELOPER FRAME DUMPS (developer builds only, WRONGGPU_FG_DEV_QUALIFIED): with WRONGGPU_FG_DUMP_DIR naming a
//    folder, the executor writes the game frame and each generated frame of every 30th generating game frame (at most
//    6) as PPM files there (FgD3d12ExecutorDesc::dump_dir); off otherwise.
//  - FACADE. A claimed chain gets game-facing buffers of its own (the chain's size, format family and count, served
//    through the swap-chain hook's facade seam: GetBuffer, GetCurrentBackBufferIndex, GetFrameLatencyWaitableObject),
//    so the presenter's copies and presents into the real chain never race the game's rendering. The facade buffer
//    the game presents is the colour input of the interval.
//  - SESSION. Created on the present thread once the core has a session description (options and a latched present),
//    with one FgD3d12Executor for the host's lifetime (pipelines built once) and the presenter the settings name
//    (fg_presenter: auto / hwfq / dxgi_tearing / dxgi_vsync, through FgWinPresenterSource). Every game present goes
//    through FgSessionPresent (FgAsiConfig::present_route): generated, real-only and copied intervals are presented by
//    the session's presenter; anything the session did not take is copied from the facade into the real chain and
//    presented as the game asked. A session that cannot be created is logged; after three failures the host stops
//    trying for the process (the game keeps running on the copy path). Resize, fullscreen changes and the chain's
//    release destroy the session (bounded waits); the next present creates it again.
//  - No lambda captures a copy that can throw; no exception leaves a callback (the wrapper detaches a callback that
//    throws, and the game's present still happens).
// Written for this project.
#pragma once
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>

#include <stdint.h>

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "../fg_proxy_core.h"
#include "../fg_settings.h"
#include "fg_swapchain_hook.h"

namespace wgfg {
class FgD3d12Executor;
class FgSession;
struct FgExecWeights;
enum class FgNet2Path : uint32_t;
}  // namespace wgfg

namespace wgfgp {
namespace win32 {

struct FgAsiSessionConfig {
    FgSettings settings;
    std::string model_path;   // the local FG model (UTF-8)
    std::string dxil_dir;     // the FG kernel folder
    bool net2_exact = false;  // net2's exact path (validation) instead of the fast default
    FgLogFn log = nullptr;
    void* log_ctx = nullptr;
};

// The precheck: true when the model validates and every kernel file is there; *weights then holds the model's weights
// in the executor's layout. *why names the first failure.
bool FgAsiSessionPrecheck(const FgAsiSessionConfig& cfg, wgfg::FgExecWeights* weights, std::string* why);

// RDNA 4 by PCI ids (vendor 0x1002; the launcher's list, clean/installer/src/wg_core/checks.cpp).
bool FgIsRdna4(uint32_t vendor_id, uint32_t device_id);

struct FgAsiSessionStats {
    uint64_t intervals = 0, session_presents = 0, copy_presents = 0;
    uint64_t sessions_created = 0, session_failures = 0, sessions_destroyed = 0;
    bool gave_up = false;
    bool facade = false;
    // the executor's counters (fg_d3d12_executor.h FgExecCounters), 0 before the first session
    uint64_t exec_shared = 0, exec_indices = 0, exec_copies = 0, exec_game_colour_copies = 0, exec_net2_fast = 0,
             exec_net2_exact = 0;
};

class FgAsiSessionHost final : public IFgPresentCallback {
public:
    // Takes the weights over (copied into the executor's arena at each session creation).
    FgAsiSessionHost(const FgAsiSessionConfig& cfg, std::unique_ptr<wgfg::FgExecWeights> weights);
    ~FgAsiSessionHost() override;
    FgAsiSessionHost(const FgAsiSessionHost&) = delete;
    FgAsiSessionHost& operator=(const FgAsiSessionHost&) = delete;

    // After the transport installed: the model / presenter facts.
    void OnInstalled(FgProxy* core);
    // FgAsiConfig hooks (ctx = this host).
    static int32_t Route(void* ctx, FgProxy* core, const WgFgPresentDesc* desc, WgFgPresentResult* out);
    static void FactoryNote(void* ctx, FgProxy* core, IUnknown* factory);
    // FgAsiConfig::core_ready (ctx = this host): the model / presenter facts and the start-up adapter probe.
    static void CoreReady(void* ctx, FgProxy* core);

    FgAsiSessionStats Stats() const;

    // ---- IFgPresentCallback ----
    void BeginInterval(const FgPresentCall&) override {}
    HRESULT PresentInterval(const FgPresentCall& call, IFgInnerPresenter& inner) override;
    void BeforeResize() override;
    void AfterResize(HRESULT hr) override;
    void BeforeFullscreen(BOOL) override;
    void AfterFullscreen(BOOL, HRESULT hr) override;
    bool GetBuffer(UINT buffer, REFIID riid, void** out, HRESULT* hr) override;
    bool CurrentBackBufferIndex(UINT* index) override;
    bool FrameLatencyWaitableObject(HANDLE* handle) override;
    void Detached() override;
    void Attached(IDXGISwapChain4* inner, IUnknown* queue) override;
    UINT ExtraSwapChainFlags(const DXGI_SWAP_CHAIN_DESC1& desc) override;

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace win32
}  // namespace wgfgp
