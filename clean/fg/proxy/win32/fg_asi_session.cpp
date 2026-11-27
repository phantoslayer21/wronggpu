// SPDX-License-Identifier: GPL-3.0-only
// WrongGPU frame generation (issue #96, F9): the FG session host of WrongGPU-FG.asi. See fg_asi_session.h.
// Written for this project.
#include "fg_asi_session.h"

#include <wrl/client.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <map>
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
// Lateness control (see the host's LATENESS comment in fg_asi_session.h): the arrival -> first generated image budget
// starts at 4 ms and grows while generated images miss their display time, up to 30 ms; it eases back 0.5 ms after
// 16 quiet windows. One window: 60 generating game frames.
constexpr wgfg::FgTicks kBudgetBase = 40000;
constexpr wgfg::FgTicks kBudgetMax = 300000;
constexpr wgfg::FgTicks kBudgetStepMin = 20000;
constexpr wgfg::FgTicks kBudgetEase = 5000;
constexpr uint32_t kBudgetWindow = 60;
constexpr uint32_t kBudgetQuietWindows = 16;
// Presenter slots: enough for the images of three intervals in flight at 6x and the one on screen
constexpr uint32_t kPresenterSlots = 12;
// Back-pressure: free presenter slots a game frame waits for (its real image and the reserve), at most 50 ms
constexpr uint32_t kBackPressureSlots = 2;
constexpr ULONGLONG kBackPressureMs = 50;

// The presenter's free slots (the Windows back ends; any other: unknown, ~0u).
uint32_t PresenterFreeSlots(const wgfg::IFgPresenter* p) {
    if (const auto* d = dynamic_cast<const wgfg::FgDxgiPresenter*>(p)) return d->FreeSlots();
    if (const auto* c = dynamic_cast<const wgfg::FgWinCompPresenter*>(p)) return c->FreeSlots();
    return ~0u;
}
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
    bool adapter_confirmed = false;  // the game's factory was checked against the start-up adapter
    WgFgAdapterIdentity adapter_id = {};
    bool tearing_supported = false;  // DXGI_FEATURE_PRESENT_ALLOW_TEARING on the game's factory
    std::string dump_dir;            // developer frame dumps (empty: off)

    // the attached chain
    IDXGISwapChain4* inner = nullptr;  // no reference: the wrapper holds the chain while we are attached
    ComPtr<ID3D12CommandQueue> queue;  // the game's queue (the executor's game-queue work)
    // the present queue (CreationQueue): created once per device; chain_queue is the queue the attached chain presents
    // on (the present queue, or the game's queue for a chain created without it)
    ComPtr<ID3D12CommandQueue> present_queue;
    ComPtr<ID3D12Device> present_device;
    ComPtr<ID3D12CommandQueue> chain_queue;
    ComPtr<ID3D12CommandQueue> attached_present_queue;  // AttachedPresentQueue, taken by the next Attached
    ComPtr<ID3D12Fence> game_fence;  // the copy path: the game's work on the facade, before the present queue copies
    uint64_t game_value = 0;
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
    uint32_t probe_failures = 0;   // start-up adapter probes that found no DXGI factory (logged at most 3 times)

    // live status for a menu (FgAsiLiveStatus): written on the present thread, read lock-free from any thread
    std::atomic<bool> host_overlay{false};
    std::atomic<uint32_t> st_phase{kFgPhaseWaiting}, st_reason{0}, st_real_only{0}, st_preset{0}, st_count{0},
        st_presenter{0};
    std::atomic<uint64_t> st_presents{0};
    bool overlay_choice_logged = false;
    HANDLE wait_timer = nullptr;      // back-pressure waits (high resolution)
    uint64_t back_pressure_waits = 0; // game frames that waited for a presenter slot
    bool back_pressure_logged = false;

    // lateness control (kBudget*): kept across sessions of the run
    wgfg::FgTicks budget = kBudgetBase;
    uint32_t window_frames = 0;
    bool window_have = false;
    wgfg::FgPresentStats window_base;
    uint32_t quiet_windows = 0;
    uint32_t budget_lines = 0;

    // retained copies of transient tagged textures, one per buffer type (RetainTag); own lock: tagging runs on the
    // game's threads and must not wait behind a present
    struct TagCopy {
        ComPtr<ID3D12Resource> res;
        ComPtr<ID3D12Device> dev;
        UINT64 w = 0;
        UINT h = 0;
        DXGI_FORMAT fmt = DXGI_FORMAT_UNKNOWN;
    };
    std::mutex tag_mu;
    std::map<uint32_t, TagCopy> tag_copies;

    void Log(const std::string& line) const {
        if (cfg.log) cfg.log(cfg.log_ctx, line.c_str());
    }

    // The first RDNA 4 hardware adapter of `factory`; false when it lists none.
    static bool FirstRdna4(IUnknown* factory, DXGI_ADAPTER_DESC1* out) {
        ComPtr<IDXGIFactory1> f1;
        if (!factory || FAILED(factory->QueryInterface(IID_PPV_ARGS(&f1)))) return false;
        for (UINT i = 0; i < 16; ++i) {
            ComPtr<IDXGIAdapter1> a;
            if (f1->EnumAdapters1(i, &a) == DXGI_ERROR_NOT_FOUND) break;
            DXGI_ADAPTER_DESC1 d = {};
            if (!a || FAILED(a->GetDesc1(&d)) || (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) continue;
            if (!FgIsRdna4(d.VendorId, d.DeviceId)) continue;
            *out = d;
            return true;
        }
        return false;
    }

    // A DXGI factory's facts: the tearing support, and the adapter (from the start-up probe, or from the game's
    // factory when the probe found none); the game's factory confirms a start-up adapter once.
    void NoteFactory(FgProxy* c, IUnknown* factory, bool game_factory) {
        {
            ComPtr<IDXGIFactory5> f5;
            BOOL tearing = FALSE;
            if (SUCCEEDED(factory->QueryInterface(IID_PPV_ARGS(&f5))))
                tearing_supported =
                    SUCCEEDED(f5->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &tearing, sizeof(tearing))) &&
                    tearing;
        }
        DXGI_ADAPTER_DESC1 d = {};
        const bool found = FirstRdna4(factory, &d);
        if (adapter_set) {
            if (game_factory && !adapter_confirmed) {
                adapter_confirmed = true;
                const bool same = found && d.AdapterLuid.LowPart == adapter_id.luid.low &&
                                  d.AdapterLuid.HighPart == adapter_id.luid.high;
                Log(same ? "fg host: the game's factory confirms the start-up adapter"
                         : "fg host: the game's factory does not list the start-up adapter first; the start-up adapter "
                           "stays");
            }
            return;
        }
        if (!found) {
            Log(game_factory ? "fg host: no RDNA 4 adapter on the game's factory; frame generation is not taken"
                             : "fg host: start-up probe: no RDNA 4 adapter; the game's factory is checked when the "
                               "game creates it");
            return;
        }
        WgFgAdapterIdentity id = {};
        id.struct_size = sizeof(id);
        id.abi_version = WG_FG_ADAPTER_ABI_VERSION;
        id.vendor_id = d.VendorId;
        id.device_id = d.DeviceId;
        id.luid.low = d.AdapterLuid.LowPart;
        id.luid.high = d.AdapterLuid.HighPart;
        id.rdna4 = 1;
        id.probes_passed = 1;  // the precheck found every kernel of the executor
        adapter_id = id;
        adapter_set = true;
        adapter_confirmed = game_factory;
        char line[128];
        wsprintfA(line, "fg host: adapter vendor 0x%04x device 0x%04x (RDNA 4, %s)", d.VendorId, d.DeviceId,
                  game_factory ? "the game's factory" : "start-up probe");
        Log(line);
        c->set_adapter(id);
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
        ID3D12CommandQueue* const cq = chain_queue ? chain_queue.Get() : queue.Get();
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
        if (cq != queue.Get()) {
            // the game rendered into the facade buffer on its own queue: the present queue copies after that work
            if (!game_fence && FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&game_fence))))
                return false;
            if (FAILED(queue->Signal(game_fence.Get(), ++game_value)) || FAILED(cq->Wait(game_fence.Get(), game_value)))
                return false;
        }
        ID3D12CommandList* lists[] = {list.Get()};
        cq->ExecuteCommandLists(1, lists);
        cq->Signal(copy_fence.Get(), ++copy_value);
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
        exec->WaitForComputeValue(want, 50);
    }

    void DestroySession(const char* why) {
        if (!session) {
            exec_source.reset();
            win_source.reset();
            return;
        }
        const wgfg::FgSessionCounters sc = session->Counters();
        if (exec) {
            const wgfg::FgExecCounters& ec = exec->Counters();
            char buf[256];
            std::snprintf(buf, sizeof buf,
                          "fg host: run stats: gen_intervals=%llu shared=%llu indices=%llu copies=%llu net2_fast=%llu "
                          "menu_det=%llu menu_copy=%llu stale_cancel=%llu refusals=%llu",
                          static_cast<unsigned long long>(sc.generated_intervals),
                          static_cast<unsigned long long>(ec.shared),
                          static_cast<unsigned long long>(ec.indices),
                          static_cast<unsigned long long>(ec.copies),
                          static_cast<unsigned long long>(ec.net2_fast),
                          static_cast<unsigned long long>(sc.menu_detections),
                          static_cast<unsigned long long>(sc.menu_copy_frames),
                          static_cast<unsigned long long>(sc.stale_cancelled),
                          static_cast<unsigned long long>(sc.frame_refusals));
            Log(buf);
        }
        if (const wgfg::IFgPresenter* pr = session->Presenter()) {
            // what the display path did: images shown, dropped by pacing, skipped by the display
            const wgfg::FgPresentStats ps = pr->Stats();
            char buf[320];
            std::snprintf(buf, sizeof buf,
                          "fg host: presenter stats: displayed_real=%llu displayed_generated=%llu generated_queued=%llu "
                          "dropped_late=%llu dropped_pressure=%llu cancelled=%llu display_skipped=%llu "
                          "max_slip_us=%lld budget_ms=%.1f",
                          static_cast<unsigned long long>(ps.displayed_real),
                          static_cast<unsigned long long>(ps.displayed_generated),
                          static_cast<unsigned long long>(ps.generated_queued),
                          static_cast<unsigned long long>(ps.dropped_late),
                          static_cast<unsigned long long>(ps.dropped_pressure),
                          static_cast<unsigned long long>(ps.cancelled),
                          static_cast<unsigned long long>(ps.display_skipped),
                          static_cast<long long>(ps.max_slip / 10), budget / 10000.0);
            Log(buf);
            // why intervals were real-only (the scheduler's / presenter's reasons), and the DXGI back end's presents
            std::string reasons = "fg host: real-only intervals:";
            for (uint32_t k = 1; k < wgfg::kFgRealOnlyReasonCount; ++k)
                if (ps.real_only_intervals[k])
                    reasons += std::string(" ") + wgfg::FgRealOnlyReasonName(static_cast<wgfg::FgRealOnlyReason>(k)) + "=" +
                           std::to_string(ps.real_only_intervals[k]);
            if (const auto* dx = dynamic_cast<const wgfg::FgDxgiPresenter*>(pr)) {
                const wgfg::FgDxgiPresenterInfo in = dx->Info();
                reasons += "; dxgi presents=" + std::to_string(in.presents) +
                       " latency_timeouts=" + std::to_string(in.latency_timeouts);
            }
            reasons += "; back-pressure waits=" + std::to_string(back_pressure_waits);
            Log(reasons);
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
        window_frames = 0;
        window_have = false;
        quiet_windows = 0;
        ++stats.sessions_destroyed;
    }

    // After each generating game frame: every kBudgetWindow frames, the share of generated images that missed their
    // display time (finished after their deadline, or found no free slot) sets the first-image budget.
    void ControlLateness() {
        if (!session) return;
        const wgfg::IFgPresenter* pr = session->Presenter();
        if (!pr || ++window_frames < kBudgetWindow) return;
        window_frames = 0;
        const wgfg::FgPresentStats st = pr->Stats();
        if (!window_have) {
            window_base = st;
            window_have = true;
            return;
        }
        const uint64_t queued = st.generated_queued - window_base.generated_queued;
        const uint64_t late = st.dropped_late - window_base.dropped_late;
        const uint64_t pressure = st.dropped_pressure - window_base.dropped_pressure;
        window_base = st;
        const uint64_t missed = late + pressure, tried = queued + missed;
        if (tried < 10) return;
        wgfg::FgTicks next = budget;
        if (missed * 10 > tried) {  // more than 1 in 10 missed: give the GPU more time
            quiet_windows = 0;
            next = budget + std::max(kBudgetStepMin, budget / 2);
            if (next > kBudgetMax) next = kBudgetMax;
        } else if (missed * 50 <= tried) {  // 1 in 50 or fewer for a while: take some latency back
            if (++quiet_windows >= kBudgetQuietWindows) {
                quiet_windows = 0;
                next = budget - kBudgetEase < kBudgetBase ? kBudgetBase : budget - kBudgetEase;
            }
        } else {
            quiet_windows = 0;
        }
        if (next == budget || !session->SetFirstImageBudget(next)) return;
        if (budget_lines < 32) {
            ++budget_lines;
            char line[256];
            std::snprintf(line, sizeof line,
                          "fg host: generated images on time %llu of %llu (late %llu, no slot %llu) in the last %u "
                          "frames: first-image budget %.1f -> %.1f ms%s",
                          static_cast<unsigned long long>(queued), static_cast<unsigned long long>(tried),
                          static_cast<unsigned long long>(late), static_cast<unsigned long long>(pressure),
                          kBudgetWindow, budget / 10000.0, next / 10000.0,
                          budget_lines == 32 ? " (the last budget line of this run)" : "");
            Log(line);
        }
        budget = next;
    }

    // Before a game frame goes to the session: wait (bounded) for a presenter slot for it (BACK-PRESSURE).
    void WaitForPresenterRoom() {
        if (!session) return;
        const wgfg::IFgPresenter* pr = session->Presenter();
        if (!pr || PresenterFreeSlots(pr) >= kBackPressureSlots) return;
        if (!wait_timer)
            wait_timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
        ++back_pressure_waits;
        if (!back_pressure_logged) {
            back_pressure_logged = true;
            Log("fg host: the display shows fewer images than the game renders; game frames now wait for a presenter "
                "slot (at most 50 ms each), as on a full swap-chain queue");
        }
        const ULONGLONG until = GetTickCount64() + kBackPressureMs;
        while (PresenterFreeSlots(pr) < kBackPressureSlots && GetTickCount64() < until) {
            LARGE_INTEGER due;
            due.QuadPart = -5000;  // 0.5 ms
            if (wait_timer && SetWaitableTimer(wait_timer, &due, 0, nullptr, nullptr, FALSE))
                WaitForSingleObject(wait_timer, 2);
            else
                Sleep(1);
        }
    }

    void TryCreateSession(FgProxy* c) {
        if (session || stats.gave_up || !c || !c->fg_taken() || !inner || !device || !queue) return;
        WgFgSessionDesc sd = {};
        if (!c->session_desc(&sd)) return;  // options and a latched present first
        // Titles commonly leave options sizes/format at 0 (taken from the swap chain).
        if (sd.output_width == 0 || sd.output_height == 0) {
            sd.output_width = width;
            sd.output_height = height;
        }
        if (sd.output_format == 0) sd.output_format = static_cast<uint32_t>(chain_format);
        if (sd.render_max_width == 0 || sd.render_max_height == 0) {
            sd.dynamic_resolution = 1;
            sd.render_min_width = 100u;
            sd.render_min_height = 100u;
            sd.render_max_width = sd.output_width;
            sd.render_max_height = sd.output_height;
        }
        if (sd.render_min_width == 0 || sd.render_min_height == 0) {
            sd.render_min_width = sd.dynamic_resolution ? 100u : sd.render_max_width;
            sd.render_min_height = sd.dynamic_resolution ? 100u : sd.render_max_height;
        }
        // room for every N the route allows: a multiplier raised in the menu applies without a new session
        const uint32_t route_max = c->advertised_max_generated();
        if (route_max > sd.max_generated) sd.max_generated = route_max;
        // OptiScaler hosts the plugin: auto takes a DXGI flip presenter, so its overlay is drawn on every image
        if (sd.presenter_mode == WG_FG_PRESENTER_AUTO && host_overlay.load()) {
            const bool tear = tearing_supported && (chain_flags & DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING) != 0;
            sd.presenter_mode = tear ? WG_FG_PRESENTER_DXGI_TEARING : WG_FG_PRESENTER_DXGI_VSYNC;
            if (!overlay_choice_logged) {
                overlay_choice_logged = true;
                Log(std::string("fg host: fg_presenter=auto with OptiScaler hosting: ") +
                    (tear ? "dxgi_tearing" : "dxgi_vsync (the chain or the system has no tearing)") +
                    ", so OptiScaler's menu and overlay are drawn on every presented image");
            }
        }
        if (!exec) {
            wgfg::FgD3d12ExecutorDesc ed;
            ed.dxil_dir = cfg.dxil_dir;
            ed.weights = weights.get();
            ed.net2 = cfg.net2_exact ? wgfg::FgNet2Path::Exact : wgfg::FgNet2Path::Fast;
            ed.settings.menu_detection = false;
            ed.dump_dir = dump_dir;  // empty (off) unless a developer build found WRONGGPU_FG_DUMP_DIR
            ed.dump_log = cfg.log;
            ed.dump_log_ctx = cfg.log_ctx;
            exec.reset(new (std::nothrow) wgfg::FgD3d12Executor(ed));
            if (!exec) return;
        }
        wgfg::FgPresenterFactoryDesc base;
        base.slot_count = kPresenterSlots;
        base.device = device.Get();
        base.queue = chain_queue ? chain_queue.Get() : queue.Get();  // the queue the real chain presents on
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
        d.scheduler.first_image_budget = budget;  // 4 ms at first; the lateness control raises it as needed
        {
            // DISPLAY RATE: no generated image shorter on screen than (95 % of) one refresh; the count adapts
            wgfg::FgClockMap map;
            wgfg::FgRefreshGrid grid;
            if (wgfg::FgReadCompositorGrid(map, 0, &grid) && grid.period > 0) {
                d.scheduler.display.min_interval = grid.period - grid.period / 20;
                d.scheduler.display.max_interval = 1000000;  // 10 Hz: a fixed-refresh display has no floor here
                d.scheduler.adaptive_count = true;
                char line[160];
                std::snprintf(line, sizeof line,
                              "fg host: display refresh %.1f Hz: an interval generates only the images the display can "
                              "show (count adapted)",
                              10000000.0 / static_cast<double>(grid.period));
                Log(line);
            } else {
                Log("fg host: the display refresh is not readable; the scheduler's own output window stays");
            }
        }
        d.executor = exec.get();
        d.presenter_source = exec_source.get();
        d.clock = &HostClock;
        d.log = cfg.log;
        d.log_ctx = cfg.log_ctx;
        const wgfg::FgSessionResult r = session->Create(d);
        if (r.error != wgfg::FgSessionError::None) {
            ++stats.session_failures;
            Log(std::string("fg host: session not created: ") + wgfg::FgSessionErrorName(r.error) + " (refusal " +
                wgfg::FgRefusalName(r.refusal) + ", executor " + wgfg::FgExecRefusalName(exec->LastRefusal()) + ": " +
                exec->LastWhy() + ")");
            session.reset();
            exec_source.reset();
            win_source.reset();
            if (stats.session_failures >= kMaxSessionFailures) {
                stats.gave_up = true;
                st_phase.store(kFgPhaseStopped);
                Log("fg host: frame generation stays off for this run (session creation failed 3 times); the game's "
                    "frames are presented as it asks");
            }
            return;
        }
        WgFgExecutorV1 h = session->ExecutorHandOff();
        c->set_executor(&h);
        ++stats.sessions_created;
        const wgfg::FgPresenterChoice chosen = win_source->LastSelection().chosen;
        st_presenter.store(static_cast<uint32_t>(chosen));
        Log(std::string("fg host: session created, ") + std::to_string(sd.output_width) + " x " +
            std::to_string(sd.output_height) + ", preset " + std::to_string(sd.preset) + ", N up to " +
            std::to_string(sd.max_generated) + ", net2 " + (cfg.net2_exact ? "exact" : "fast") + ", presenter " +
            wgfg::FgPresenterChoiceName(chosen));
    }
};

FgAsiSessionHost::FgAsiSessionHost(const FgAsiSessionConfig& cfg, std::unique_ptr<wgfg::FgExecWeights> weights)
    : impl_(new Impl()) {
    impl_->cfg = cfg;
    impl_->weights = std::move(weights);
#if defined(WRONGGPU_FG_DEV_QUALIFIED) && WRONGGPU_FG_DEV_QUALIFIED
    // developer frame dumps: only with WRONGGPU_FG_DUMP_DIR naming an existing folder
    char buf[MAX_PATH * 2];
    const DWORD n = GetEnvironmentVariableA("WRONGGPU_FG_DUMP_DIR", buf, static_cast<DWORD>(sizeof buf));
    if (n > 0 && n < sizeof buf) {
        std::string dir(buf, n);
        while (dir.size() > 3 && (dir.back() == '\\' || dir.back() == '/')) dir.pop_back();
        const DWORD a = GetFileAttributesA(dir.c_str());
        if (a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_DIRECTORY)) {
            impl_->dump_dir = dir;
            impl_->Log("fg host: developer frame dumps on: every 30th generating game frame, at most 6, into " + dir);
        } else {
            impl_->Log("fg host: WRONGGPU_FG_DUMP_DIR is not an existing folder; no frame dumps");
        }
    }
#endif
}

FgAsiSessionHost::~FgAsiSessionHost() {
    std::lock_guard<std::recursive_mutex> lock(impl_->mu);
    impl_->DestroySession("unload");
    if (impl_->exec) impl_->exec->WaitIdle(kDestroyWaitMs);
    impl_->WaitCopy(impl_->copy_value, kDestroyWaitMs);
    impl_->ReleaseFacade();
    {
        std::lock_guard<std::mutex> tag_lock(impl_->tag_mu);
        impl_->tag_copies.clear();
    }
    if (impl_->latency) CloseHandle(impl_->latency);
    if (impl_->copy_event) CloseHandle(impl_->copy_event);
    if (impl_->wait_timer) CloseHandle(impl_->wait_timer);
}

void FgAsiSessionHost::OnInstalled(FgProxy* core) {
    std::lock_guard<std::recursive_mutex> lock(impl_->mu);
    impl_->core = core;
    // the precheck passed: the model validated; the presenter is decided at session creation (refused there, not
    // replaced)
    if (core) core->set_runtime_facts(true, true);
}

FgAsiLiveStatus FgAsiSessionHost::LiveStatus() const {
    const Impl& m = *impl_;
    FgAsiLiveStatus s;
    s.phase = m.st_phase.load();
    s.reason = m.st_reason.load();
    s.real_only = m.st_real_only.load();
    s.preset = m.st_preset.load();
    s.count = m.st_count.load();
    s.presenter = m.st_presenter.load();
    s.presents = m.st_presents.load();
    s.host_overlay = m.host_overlay.load();
    return s;
}

void FgAsiSessionHost::SetHostOverlay(const char* why) {
    if (impl_->host_overlay.exchange(true)) return;
    impl_->Log(std::string("fg host: OptiScaler hosts this plugin (") + (why ? why : "?") +
               "); fg_presenter=auto presents through its swap chain");
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
    m.NoteFactory(core, factory, true);
}

void FgAsiSessionHost::CoreReady(void* ctx, FgProxy* core) {
    FgAsiSessionHost* self = static_cast<FgAsiSessionHost*>(ctx);
    if (!self || !core) return;
    Impl& m = *self->impl_;
    std::lock_guard<std::recursive_mutex> lock(m.mu);
    try {
        // the precheck passed: the model validated; the presenter is decided at session creation
        core->set_runtime_facts(true, true);
        if (m.adapter_set) {
            core->set_adapter(m.adapter_id);  // a later install: the adapter found before
            return;
        }
        // read-only: a factory and its adapter list, no device
        ComPtr<IDXGIFactory1> f;
        const HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&f));
        if (FAILED(hr) || !f) {
            // InitializeASI can run inside the host's own start-up, where DXGI may refuse; the probe runs again at the
            // game's first FG support / load call (the transport calls this while FG is not taken) and the game's own
            // factory is checked when the game creates it
            if (++m.probe_failures <= 3) {
                char line[224];
                std::snprintf(line, sizeof line,
                              "fg host: start-up probe: no DXGI factory (0x%08lx, attempt %u); tried again at the "
                              "game's first FG call, and the game's factory is checked when the game creates it",
                              static_cast<unsigned long>(hr), m.probe_failures);
                m.Log(line);
            }
            return;
        }
        if (m.probe_failures) {
            m.Log("fg host: start-up probe: a DXGI factory after " + std::to_string(m.probe_failures) +
                  " failed attempt(s)");
            m.probe_failures = 0;
        }
        m.NoteFactory(core, f.Get(), false);
    } catch (...) {
        // the game's factory is still checked when the game creates it
    }
}

bool FgAsiSessionHost::RetainTag(void* ctx, uint32_t buffer_type, bool transient, void* cmd, WgFgTexture* tex,
                                 bool* copied_out) {
    if (copied_out) *copied_out = false;
    FgAsiSessionHost* self = static_cast<FgAsiSessionHost*>(ctx);
    if (!tex || !tex->resource) return false;
    const bool need_desc = tex->width == 0 || tex->height == 0 || tex->format == 0;
    if (!transient && !need_desc) return true;
    if (!self) return !transient;
    try {
        ComPtr<ID3D12Resource> src;
        if (FAILED(static_cast<IUnknown*>(tex->resource)->QueryInterface(IID_PPV_ARGS(&src))) || !src)
            return !transient;
        const D3D12_RESOURCE_DESC sd = src->GetDesc();
        if (sd.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || sd.SampleDesc.Count != 1) return false;
        tex->width = static_cast<uint32_t>(sd.Width);
        tex->height = sd.Height;
        tex->format = static_cast<uint32_t>(sd.Format);
        if (tex->rect.width == 0 || tex->rect.height == 0) tex->rect = WgFgRect{0, 0, tex->width, tex->height};
        if (!transient || !cmd) return true;

        ComPtr<ID3D12GraphicsCommandList> cl;
        if (FAILED(static_cast<IUnknown*>(cmd)->QueryInterface(IID_PPV_ARGS(&cl))) || !cl) return false;
        ComPtr<ID3D12Device> dev;
        if (FAILED(src->GetDevice(IID_PPV_ARGS(&dev))) || !dev) return false;

        ComPtr<ID3D12Resource> dst;
        {
            Impl& m = *self->impl_;
            std::lock_guard<std::mutex> lock(m.tag_mu);
            Impl::TagCopy& e = m.tag_copies[buffer_type];
            if (!e.res || e.dev.Get() != dev.Get() || e.w != sd.Width || e.h != sd.Height || e.fmt != sd.Format) {
                e = Impl::TagCopy();
                D3D12_HEAP_PROPERTIES hp = {};
                hp.Type = D3D12_HEAP_TYPE_DEFAULT;
                D3D12_RESOURCE_DESC cd = sd;
                cd.Flags = D3D12_RESOURCE_FLAG_NONE;
                cd.MipLevels = 1;
                cd.DepthOrArraySize = 1;
                ComPtr<ID3D12Resource> made;
                if (SUCCEEDED(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &cd,
                                                           D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                           IID_PPV_ARGS(&made))) && made) {
                    e.res = made;
                    e.dev = dev;
                    e.w = sd.Width;
                    e.h = sd.Height;
                    e.fmt = sd.Format;
                }
            }
            dst = e.res;
        }
        if (!dst) return false;

        const bool have_state = tex->state != WG_STATE_NOT_GIVEN;
        const D3D12_RESOURCE_STATES src_state = static_cast<D3D12_RESOURCE_STATES>(tex->state);
        const bool trans_src = have_state && src_state != D3D12_RESOURCE_STATE_COPY_SOURCE;
        D3D12_RESOURCE_BARRIER pre[2] = {};
        UINT npre = 0;
        if (trans_src) {
            pre[npre].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            pre[npre].Transition = {src.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, src_state,
                                    D3D12_RESOURCE_STATE_COPY_SOURCE};
            ++npre;
        }
        pre[npre].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        pre[npre].Transition = {dst.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COMMON,
                                D3D12_RESOURCE_STATE_COPY_DEST};
        ++npre;
        cl->ResourceBarrier(npre, pre);
        if (sd.MipLevels == 1 && sd.DepthOrArraySize == 1) {
            cl->CopyResource(dst.Get(), src.Get());
        } else {
            D3D12_TEXTURE_COPY_LOCATION dloc = {}, sloc = {};
            dloc.pResource = dst.Get();
            dloc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            dloc.SubresourceIndex = 0;
            sloc.pResource = src.Get();
            sloc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            sloc.SubresourceIndex = 0;
            cl->CopyTextureRegion(&dloc, 0, 0, 0, &sloc, nullptr);
        }
        for (UINT k = 0; k < npre; ++k) std::swap(pre[k].Transition.StateBefore, pre[k].Transition.StateAfter);
        cl->ResourceBarrier(npre, pre);
        tex->resource = dst.Get();  // the host's cache keeps the reference
        tex->state = D3D12_RESOURCE_STATE_COMMON;
        if (copied_out) *copied_out = true;
        return true;
    } catch (...) {
        return !transient;
    }
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
        m.WaitForPresenterRoom();
        wgfg::FgSessionResult sr;
        const int32_t rc = wgfg::FgSessionPresent(core, m.session.get(), desc, out, &sr);
        m.st_presents.fetch_add(1);
        if (out) {
            if (out->status == WG_FG_PRESENT_GENERATE) {
                m.st_preset.store(out->effective_preset);
                m.st_count.store(out->effective_count);
                m.st_phase.store(kFgPhaseGenerating);
                m.ControlLateness();
            } else if (out->status == WG_FG_PRESENT_REAL_ONLY || out->status == WG_FG_PRESENT_REFUSED) {
                m.st_reason.store(out->reason);
                m.st_real_only.store(out->real_only);
                m.st_phase.store(kFgPhaseRealOnly);
            } else {
                m.st_phase.store(kFgPhaseWaiting);
            }
        }
        if (m.session && out) {
            const bool took = out->status == WG_FG_PRESENT_GENERATE || out->status == WG_FG_PRESENT_REAL_ONLY;
            // the session presented the game frame unless it refused before its presenter was reached
            m.handled = took && sr.frame_token == desc->frame_token &&
                        (sr.error == wgfg::FgSessionError::None || sr.error == wgfg::FgSessionError::FrameRefused);
            if (took && !m.handled) {
                m.Log(std::string("fg host: the session did not present a game frame: ") +
                      wgfg::FgSessionErrorName(sr.error) + " (presenter " + wgfg::FgStatusName(sr.presenter) +
                      ", refusal " + wgfg::FgRefusalName(sr.refusal) + ", exec " +
                      (m.exec ? wgfg::FgExecRefusalName(m.exec->LastRefusal()) : "null") + ": " +
                      (m.exec ? m.exec->LastWhy() : std::string()) + ")");
                if (m.session->State() != wgfg::FgSessionState::Active) {
                    m.DestroySession("refused");
                    if (++m.stats.session_failures >= kMaxSessionFailures) {
                        m.stats.gave_up = true;
                        m.st_phase.store(kFgPhaseStopped);
                    }
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
            m.st_phase.store(kFgPhaseStopped);
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
    impl_->chain_queue.Reset();
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
    m.chain_queue = m.attached_present_queue ? m.attached_present_queue : q;
    m.attached_present_queue.Reset();
    if (FAILED(inner->GetHwnd(&m.hwnd))) m.hwnd = nullptr;
    if (!m.latency) m.latency = CreateSemaphoreW(nullptr, 1, 16, nullptr);
    if (m.core) m.core->set_device(dev.Get(), q.Get());
    if (!m.CreateFacade()) {
        m.Log("fg host: facade buffers not created; the chain's own buffers stay in use");
        return;
    }
    char line[128];
    wsprintfA(line, "fg host: chain attached, %u x %u, format %u, %u buffers (facade), presents on %s", m.width,
              m.height, static_cast<unsigned>(m.chain_format), static_cast<unsigned>(m.facade.size()),
              m.chain_queue.Get() != m.queue.Get() ? "the present queue" : "the game's queue");
    m.Log(line);
}

IUnknown* FgAsiSessionHost::CreationQueue(IUnknown* game_queue, const DXGI_SWAP_CHAIN_DESC1&) {
    Impl& m = *impl_;
    std::lock_guard<std::recursive_mutex> lock(m.mu);
    try {
        ComPtr<ID3D12CommandQueue> gq;
        ComPtr<ID3D12Device> dev;
        if (!game_queue || FAILED(game_queue->QueryInterface(IID_PPV_ARGS(&gq))) || !gq ||
            FAILED(gq->GetDevice(IID_PPV_ARGS(&dev))) || !dev)
            return nullptr;
        const D3D12_COMMAND_QUEUE_DESC gd = gq->GetDesc();
        if (gd.Type != D3D12_COMMAND_LIST_TYPE_DIRECT) return nullptr;
        if (m.present_queue && m.present_device.Get() == dev.Get()) return m.present_queue.Get();
        m.present_queue.Reset();
        m.present_device.Reset();
        D3D12_COMMAND_QUEUE_DESC qd = {};
        qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        qd.Priority = D3D12_COMMAND_QUEUE_PRIORITY_HIGH;
        qd.NodeMask = gd.NodeMask;
        bool high = true;
        if (FAILED(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&m.present_queue)))) {
            high = false;
            qd.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
            if (FAILED(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&m.present_queue)))) {
                m.present_queue.Reset();
                m.Log("fg host: no present queue of its own; the real swap chain presents on the game's queue");
                return nullptr;
            }
        }
        m.present_queue->SetName(L"WrongGPU FG present queue");
        m.present_device = dev;
        m.Log(std::string("fg host: the real swap chain is created on a present queue of its own (") +
              (high ? "high" : "normal") + " priority): presents never wait behind the game's rendering");
        return m.present_queue.Get();
    } catch (...) {
        return nullptr;
    }
}

void FgAsiSessionHost::AttachedPresentQueue(IUnknown* q) {
    Impl& m = *impl_;
    std::lock_guard<std::recursive_mutex> lock(m.mu);
    m.attached_present_queue.Reset();
    if (q) q->QueryInterface(IID_PPV_ARGS(&m.attached_present_queue));
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
