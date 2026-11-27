// SPDX-License-Identifier: GPL-3.0-only
// WrongGPU frame generation (issue #96, F9): the FG session host of WrongGPU-FG.asi. See fg_asi_session.h.
// Written for this project.
#include "fg_asi_session.h"
#include "fg_asi_transport.h"

#include <wrl/client.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <map>
#include <new>

#include "../../presenter/fg_dxgi_presenter.h"
#include "../../presenter/fg_win_common.h"
#include "../../presenter/fg_win_comp_presenter.h"
#include "../../runtime/fg_d3d12_executor.h"
#include "../../runtime/fg_model.h"
#include "../../runtime/fg_session.h"
#include "../../runtime/fg_session_win.h"
#include "../../../runtime/rdna4_win.h"
#include "../fg_contract_bridge.h"

namespace wgfgp {
namespace win32 {

using Microsoft::WRL::ComPtr;

namespace {

constexpr uint32_t kMaxSessionFailures = 3;
// Lateness control (O-A7 / A4): first-image budget starts at 4.0 ms and is capped at 8.0 ms (never climbs to 20-30 ms
// buying runaway latency); eases back 0.5 ms after 3 quiet windows. Uses truthful scheduled denominator (no double
// count). Switching to a faster preset (e.g. A/B -> C) decays the budget immediately (N-B7).
constexpr wgfg::FgTicks kBudgetBase = 40000;
constexpr wgfg::FgTicks kBudgetMax = 80000;
constexpr wgfg::FgTicks kBudgetStepMin = 10000;
constexpr wgfg::FgTicks kBudgetEase = 5000;
constexpr uint32_t kBudgetWindow = 60;
constexpr uint32_t kBudgetQuietWindows = 3;
// Presenter slots: enough for three intervals in flight at 6x (3 * 6 = 18) plus the one on screen and real reserve.
constexpr uint32_t kPresenterSlots = 20;
constexpr uint32_t kPresenterMaxPendingIntervals = 3;
// Back-pressure: free presenter slots a game frame waits for (its real image and the reserve), at most 50 ms
constexpr uint32_t kBackPressureSlots = 2;
constexpr ULONGLONG kBackPressureMs = 50;
constexpr uint32_t kTraceRingSize = 512;
constexpr uint32_t kTraceSummaryInterval = 300;

// The presenter's free slots (the Windows back ends; any other: unknown, ~0u).
uint32_t PresenterFreeSlots(const wgfg::IFgPresenter* p) {
    if (const auto* d = dynamic_cast<const wgfg::FgDxgiPresenter*>(p)) return d->FreeSlots();
    if (const auto* c = dynamic_cast<const wgfg::FgWinCompPresenter*>(p)) return c->FreeSlots();
    return ~0u;
}

DWORD PresenterWorkerTid(const wgfg::IFgPresenter* p) {
    if (const auto* d = dynamic_cast<const wgfg::FgDxgiPresenter*>(p)) return d->WorkerThreadId();
    if (const auto* c = dynamic_cast<const wgfg::FgWinCompPresenter*>(p)) return c->WorkerThreadId();
    return 0;
}

// Queries the active monitor's exact display mode refresh period (in 100 ns ticks) and Hz via EnumDisplaySettingsExW.
bool QueryMonitorRefresh(HWND hwnd, wgfg::FgTicks* period_out, uint32_t* hz_out) {
    HMONITOR mon = hwnd ? MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST)
                        : MonitorFromPoint(POINT{0, 0}, MONITOR_DEFAULTTOPRIMARY);
    if (!mon) return false;
    MONITORINFOEXW mi = {};
    mi.cbSize = sizeof(mi);
    if (!GetMonitorInfoW(mon, &mi)) return false;
    DEVMODEW dm = {};
    dm.dmSize = sizeof(dm);
    if (!EnumDisplaySettingsExW(mi.szDevice, ENUM_CURRENT_SETTINGS, &dm, 0)) return false;
    if (dm.dmDisplayFrequency < 24 || dm.dmDisplayFrequency > 1000) return false;
    if (hz_out) *hz_out = dm.dmDisplayFrequency;
    if (period_out)
        *period_out = (10000000ll + static_cast<wgfg::FgTicks>(dm.dmDisplayFrequency) / 2) /
                      static_cast<wgfg::FgTicks>(dm.dmDisplayFrequency);
    return true;
}

int64_t Percentile(std::vector<int64_t>& v, double pct) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    const size_t idx = static_cast<size_t>(pct * static_cast<double>(v.size() - 1));
    return v[idx < v.size() ? idx : v.size() - 1];
}
constexpr DWORD kDestroyWaitMs = 1000;
constexpr DWORD kLatencyWaitMs = 100;
constexpr uint32_t kCopyRing = 4;

bool FileExists(const std::string& p) {
    const DWORD a = GetFileAttributesA(p.c_str());
    return a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY);
}

wgfg::FgTicks HostClock(void*) { return wgfg::FgInterruptNow(); }

// The fallback facade format of a swap-chain format when castable typed textures are unavailable: typeless only for
// 8-bit UNORM families that games commonly view as sRGB. R10G10B10A2_UNORM has no sRGB view and stays typed so games
// calling CreateRenderTargetView(buf, nullptr, ...) or checking GetDesc().Format == 24 never see a typeless buffer.
DXGI_FORMAT FacadeFormat(DXGI_FORMAT f) {
    switch (f) {
        case DXGI_FORMAT_R8G8B8A8_UNORM: case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: return DXGI_FORMAT_R8G8B8A8_TYPELESS;
        case DXGI_FORMAT_B8G8R8A8_UNORM: case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: return DXGI_FORMAT_B8G8R8A8_TYPELESS;
        default: return f;
    }
}

DXGI_FORMAT RetainCopyFormat(DXGI_FORMAT f) {
    switch (f) {
        case DXGI_FORMAT_D32_FLOAT: return DXGI_FORMAT_R32_TYPELESS;
        case DXGI_FORMAT_D24_UNORM_S8_UINT: return DXGI_FORMAT_R24G8_TYPELESS;
        case DXGI_FORMAT_D32_FLOAT_S8X24_UINT: return DXGI_FORMAT_R32G8X24_TYPELESS;
        case DXGI_FORMAT_D16_UNORM: return DXGI_FORMAT_R16_TYPELESS;
        default: return f;
    }
}

bool FacadeCastablePair(DXGI_FORMAT f, DXGI_FORMAT out[2]) {
    switch (f) {
        case DXGI_FORMAT_R8G8B8A8_UNORM:
        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
            out[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
            out[1] = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
            return true;
        case DXGI_FORMAT_B8G8R8A8_UNORM:
        case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
            out[0] = DXGI_FORMAT_B8G8R8A8_UNORM;
            out[1] = DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
            return true;
        default: return false;
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
    ComPtr<ID3D12Fence> game_fence;  // facade read fence on game queue + copy-path handoff
    uint64_t game_value = 0;
    HANDLE game_fence_event = nullptr;
    ComPtr<ID3D12Device> device;
    HWND hwnd = nullptr;
    UINT chain_flags = 0;
    bool game_requested_waitable = false;
    bool pending_game_waitable = false;
    bool logged_noop_fs = false;
    uint64_t noop_fullscreen_ignored = 0;

    // facade + per-buffer GPU reuse fences (A2: kept separate from presenter slots and latency credits)
    std::vector<ComPtr<ID3D12Resource>> facade;
    std::vector<uint64_t> facade_game_fence;
    std::vector<uint64_t> facade_copy_fence;
    UINT facade_index = 0;
    DXGI_FORMAT chain_format = DXGI_FORMAT_UNKNOWN;
    uint32_t color_space = WG_FG_CS_SDR;
    UINT width = 0, height = 0;

    // A2 / O-A1: game frame-latency credit contract. Leaf lock `credit_mu` ONLY (never acquires Impl::mu or presenter
    // mutex, preventing any lock-order inversion when the presenter worker retires a real frame).
    // The `latency_sem` handle persists across resizes / fullscreen / session recreations so cached handles stay valid.
    mutable std::mutex credit_mu;
    std::condition_variable credit_cv;
    HANDLE latency_sem = nullptr;
    UINT latency_limit = 1;
    bool waitable_handed_out = false;
    std::vector<uint64_t> active_real_frames;
    int32_t in_flight_real = 0;
    int32_t credits_to_absorb = 0;
    uint64_t real_retires = 0;
    uint64_t credit_waits = 0;
    bool skip_next_copy_credit = false;

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
    DWORD registered_worker_tid = 0;
    bool handled = false;  // the current interval was presented by the session
    uint64_t last_values[2] = {};  // N-B3: aligned to 2-interval executor pipeline bound
    FgAsiSessionStats stats;
    uint32_t probe_failures = 0;   // start-up adapter probes that found no DXGI factory (logged at most 3 times)

    // live status for a menu (FgAsiLiveStatus): written on the present thread, read lock-free from any thread
    std::atomic<bool> host_overlay{false};
    std::atomic<uint32_t> st_phase{kFgPhaseWaiting}, st_reason{0}, st_real_only{0}, st_preset{0}, st_count{0},
        st_presenter{0};
    std::atomic<uint64_t> st_presents{0};
    std::atomic<uint32_t> st_mult_source{kFgMultSourceManual}, st_req_count{0}, st_sched_count{0},
        st_cap_reason{kFgCapNone}, st_refresh_hz{0}, st_sustained_mult_x100{100}, st_latency_limit{1},
        st_in_flight_real{0};
    std::atomic<bool> st_has_mfg_selector{false};
    bool overlay_choice_logged = false;
    HANDLE wait_timer = nullptr;      // back-pressure waits (high resolution)
    uint64_t back_pressure_waits = 0; // game frames that waited for a presenter slot
    bool back_pressure_logged = false;

    // lateness control (kBudget*): kept across sessions of the run
    wgfg::FgTicks budget = kBudgetBase;
    uint32_t lateness_count_cap = 0;
    uint32_t cap_miss_windows = 0;
    uint32_t window_frames = 0;
    bool window_have = false;
    wgfg::FgPresentStats window_base;
    uint32_t quiet_windows = 0;
    uint32_t budget_lines = 0;
    bool have_last_preset = false;
    uint32_t last_preset = 0;

    // A3: per-interval tracing ring (leaf lock `trace_mu`, never acquires Impl::mu)
    struct IntervalTrace {
        uint64_t frame_token = 0;
        uint64_t interval_id = 0;
        uint32_t requested_n = 0;
        uint32_t scheduled_n = 0;
        uint32_t in_flight = 0;
        uint32_t free_slots = 0;
        wgfg::FgTicks t_arrival = 0;
        wgfg::FgTicks t_after_wait = 0;
        wgfg::FgTicks t_after_record = 0;
        wgfg::FgTicks t_first_gen_ready = 0;
        wgfg::FgTicks t_first_gen_submit = 0;
        wgfg::FgTicks t_real_ready = 0;
        wgfg::FgTicks t_real_submit = 0;
        wgfg::FgTicks t_real_display = 0;
        wgfg::FgTicks max_slip = 0;
        uint32_t drops_pre = 0;
        uint32_t drops_post = 0;
        uint32_t skips = 0;
    };
    mutable std::mutex trace_mu;
    IntervalTrace trace_ring[kTraceRingSize] = {};
    uint32_t trace_head = 0;
    uint32_t trace_count = 0;
    uint32_t trace_since_summary = 0;
    wgfg::FgTicks prev_arrival = 0;
    bool trace_csv_enabled = false;
    std::string trace_csv_path;
    bool trace_csv_header_written = false;

    void InitTraceCsv() {
        char env[4] = {};
        if (GetEnvironmentVariableA("WRONGGPU_FG_TRACE", env, sizeof(env)) == 1 && env[0] == '1') {
            trace_csv_enabled = true;
            char dir[MAX_PATH * 2] = {};
            const DWORD n = GetEnvironmentVariableA("WG_STATE_GAME_DIR", dir, static_cast<DWORD>(sizeof(dir)));
            if (n > 0 && n < sizeof(dir)) {
                trace_csv_path = std::string(dir, n) + "\\wronggpu-fg-trace.csv";
            } else {
                trace_csv_path = "wronggpu-fg-trace.csv";
            }
        }
    }

    void AppendTraceCsv(const IntervalTrace& tr) {
        if (!trace_csv_enabled || trace_csv_path.empty()) return;
        FILE* fp = nullptr;
        if (fopen_s(&fp, trace_csv_path.c_str(), "ab") != 0 || !fp) return;
        if (!trace_csv_header_written) {
            trace_csv_header_written = true;
            std::fprintf(fp,
                         "frame_token,interval_id,req_n,sched_n,in_flight,free_slots,t_arrival,wait_us,record_us,"
                         "first_gen_ready_us,first_gen_sub_us,real_ready_us,real_sub_us,real_disp_us,max_slip_us,"
                         "drops_pre,drops_post,skips\n");
        }
        auto us_from = [&](wgfg::FgTicks ts) -> long long {
            return ts > 0 && tr.t_arrival > 0 ? static_cast<long long>((ts - tr.t_arrival) / 10) : -1ll;
        };
        std::fprintf(fp, "%llu,%llu,%u,%u,%u,%u,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%u,%u,%u\n",
                     static_cast<unsigned long long>(tr.frame_token),
                     static_cast<unsigned long long>(tr.interval_id), tr.requested_n, tr.scheduled_n, tr.in_flight,
                     tr.free_slots, static_cast<long long>(tr.t_arrival), us_from(tr.t_after_wait),
                     us_from(tr.t_after_record), us_from(tr.t_first_gen_ready), us_from(tr.t_first_gen_submit),
                     us_from(tr.t_real_ready), us_from(tr.t_real_submit), us_from(tr.t_real_display),
                     static_cast<long long>(tr.max_slip / 10), tr.drops_pre, tr.drops_post, tr.skips);
        std::fclose(fp);
    }

    void RecordTraceInterval(const IntervalTrace& tr) {
        std::lock_guard<std::mutex> lock(trace_mu);
        if (trace_count == kTraceRingSize) AppendTraceCsv(trace_ring[trace_head]);
        trace_ring[trace_head] = tr;
        trace_head = (trace_head + 1) % kTraceRingSize;
        if (trace_count < kTraceRingSize) ++trace_count;
    }

    static void OnTraceThunk(void* ctx, uint64_t interval_id, uint32_t index, bool real, uint8_t kind,
                             wgfg::FgTicks ts, wgfg::FgTicks target) {
        Impl* m = static_cast<Impl*>(ctx);
        if (!m || interval_id == 0) return;
        const wgfg::FgTicks now = ts != 0 ? ts : wgfg::FgInterruptNow();
        std::lock_guard<std::mutex> lock(m->trace_mu);
        // Linear scan of recent entries (at most 32 back — intervals complete in FIFO order).
        const uint32_t scan = m->trace_count < 32 ? m->trace_count : 32;
        for (uint32_t off = 1; off <= scan; ++off) {
            const uint32_t idx = (m->trace_head + kTraceRingSize - off) % kTraceRingSize;
            IntervalTrace& tr = m->trace_ring[idx];
            if (tr.interval_id != interval_id) continue;
            switch (kind) {
                case 0:  // fence done / queued to outbox
                    if (real) tr.t_real_ready = now;
                    else if (index == 1 && tr.t_first_gen_ready == 0) tr.t_first_gen_ready = now;
                    break;
                case 1:  // submitted to OS present
                    if (real) tr.t_real_submit = now;
                    else if (index == 1 && tr.t_first_gen_submit == 0) tr.t_first_gen_submit = now;
                    break;
                case 2:  // displayed (or DXGI submit estimate)
                    if (real) tr.t_real_display = now;
                    if (target > 0 && now - target > tr.max_slip) tr.max_slip = now - target;
                    break;
                case 3: ++tr.drops_pre; break;
                case 4: ++tr.drops_post; break;
                case 5: ++tr.skips; break;
                default: break;
            }
            break;
        }
    }

    void LogTraceSummary(const char* tag) {
        std::vector<IntervalTrace> snap;
        {
            std::lock_guard<std::mutex> lock(trace_mu);
            if (trace_count == 0) return;
            snap.reserve(trace_count);
            const uint32_t start = trace_count == kTraceRingSize ? trace_head : 0;
            for (uint32_t i = 0; i < trace_count; ++i)
                snap.push_back(trace_ring[(start + i) % kTraceRingSize]);
        }
        std::vector<int64_t> base_dt, wait_us, rec_us, gen_rdy_us, real_rdy_us, slip_us;
        uint32_t max_in_flight = 0;
        for (size_t i = 0; i < snap.size(); ++i) {
            const IntervalTrace& tr = snap[i];
            if (i > 0 && tr.t_arrival > snap[i - 1].t_arrival) {
                const int64_t dt = (tr.t_arrival - snap[i - 1].t_arrival) / 10;
                if (dt > 0 && dt < 250000) base_dt.push_back(dt);
            }
            if (tr.t_after_wait >= tr.t_arrival) wait_us.push_back((tr.t_after_wait - tr.t_arrival) / 10);
            if (tr.t_after_record >= tr.t_after_wait) rec_us.push_back((tr.t_after_record - tr.t_after_wait) / 10);
            if (tr.t_first_gen_ready >= tr.t_arrival)
                gen_rdy_us.push_back((tr.t_first_gen_ready - tr.t_arrival) / 10);
            if (tr.t_real_ready >= tr.t_arrival) real_rdy_us.push_back((tr.t_real_ready - tr.t_arrival) / 10);
            if (tr.t_real_display > 0) slip_us.push_back(tr.max_slip / 10);
            if (tr.in_flight > max_in_flight) max_in_flight = tr.in_flight;
        }
        char buf[384];
        std::snprintf(buf, sizeof(buf),
                      "fg trace (%s, n=%zu): base_dt_us p50=%lld p95=%lld p99=%lld | wait_us p50=%lld max=%lld | "
                      "rec_us p50=%lld p95=%lld | gen1_rdy_us p50=%lld p95=%lld | real_rdy_us p50=%lld | "
                      "slip_us p50=%lld p95=%lld max=%lld | max_inflight=%u budget_ms=%.1f",
                      tag, snap.size(), static_cast<long long>(Percentile(base_dt, 0.50)),
                      static_cast<long long>(Percentile(base_dt, 0.95)),
                      static_cast<long long>(Percentile(base_dt, 0.99)),
                      static_cast<long long>(Percentile(wait_us, 0.50)),
                      static_cast<long long>(Percentile(wait_us, 1.00)),
                      static_cast<long long>(Percentile(rec_us, 0.50)),
                      static_cast<long long>(Percentile(rec_us, 0.95)),
                      static_cast<long long>(Percentile(gen_rdy_us, 0.50)),
                      static_cast<long long>(Percentile(gen_rdy_us, 0.95)),
                      static_cast<long long>(Percentile(real_rdy_us, 0.50)),
                      static_cast<long long>(Percentile(slip_us, 0.50)),
                      static_cast<long long>(Percentile(slip_us, 0.95)),
                      static_cast<long long>(Percentile(slip_us, 1.00)), max_in_flight, budget / 10000.0);
        Log(buf);
    }

    // ---- A2 / O-A1 Credit contract helpers (leaf lock `credit_mu` ONLY) ----
    void EnsureLatencySemLocked() {
        if (!latency_sem) {
            const LONG initial = static_cast<LONG>(latency_limit) - in_flight_real;
            latency_sem = CreateSemaphoreW(nullptr, initial > 0 ? initial : 0, 16, nullptr);
        }
    }

    void ReturnOneCreditLocked() {
        if (credits_to_absorb > 0) {
            --credits_to_absorb;
        } else if (latency_sem && waitable_handed_out) {
            LONG prev = 0;
            if (ReleaseSemaphore(latency_sem, 1, &prev) && prev + 1 > static_cast<LONG>(latency_limit)) {
                // Keep semaphore count capped at latency_limit even if the game requested the waitable handle and
                // chose not to wait on it.
                WaitForSingleObject(latency_sem, 0);
            }
        }
        credit_cv.notify_all();
    }

    static void OnRealRetireThunk(void* ctx, uint64_t frame_id, uint64_t, bool) {
        Impl* m = static_cast<Impl*>(ctx);
        if (!m) return;
        std::lock_guard<std::mutex> lock(m->credit_mu);
        auto it = std::find(m->active_real_frames.begin(), m->active_real_frames.end(), frame_id);
        if (it == m->active_real_frames.end()) return;
        m->active_real_frames.erase(it);
        m->in_flight_real = static_cast<int32_t>(m->active_real_frames.size());
        ++m->real_retires;
        m->st_in_flight_real.store(static_cast<uint32_t>(m->in_flight_real));
        m->ReturnOneCreditLocked();
    }

    void BeginRealCredit(uint64_t frame_id) {
        std::lock_guard<std::mutex> lock(credit_mu);
        active_real_frames.push_back(frame_id);
        in_flight_real = static_cast<int32_t>(active_real_frames.size());
        st_in_flight_real.store(static_cast<uint32_t>(in_flight_real));
    }

    // Returns true if the credit was still active and cancelled before retirement; false if the presenter already
    // retired this frame_id (and thus already returned its semaphore credit).
    bool CancelRealCredit(uint64_t frame_id) {
        std::lock_guard<std::mutex> lock(credit_mu);
        auto it = std::find(active_real_frames.begin(), active_real_frames.end(), frame_id);
        if (it == active_real_frames.end()) return false;
        active_real_frames.erase(it);
        in_flight_real = static_cast<int32_t>(active_real_frames.size());
        st_in_flight_real.store(static_cast<uint32_t>(in_flight_real));
        credit_cv.notify_all();
        return true;
    }

    void WaitForCreditRoom(uint32_t max_ms) {
        std::unique_lock<std::mutex> lock(credit_mu);
        if (in_flight_real < static_cast<int32_t>(latency_limit)) return;
        ++credit_waits;
        credit_cv.wait_for(lock, std::chrono::milliseconds(max_ms),
                           [this]() { return in_flight_real < static_cast<int32_t>(latency_limit); });
    }

    void ReturnCopyPresentCredit() {
        std::lock_guard<std::mutex> lock(credit_mu);
        ReturnOneCreditLocked();
    }

    void DrainInFlightCredits() {
        std::lock_guard<std::mutex> lock(credit_mu);
        const size_t n = active_real_frames.size();
        active_real_frames.clear();
        in_flight_real = 0;
        for (size_t i = 0; i < n; ++i) ReturnOneCreditLocked();
        st_in_flight_real.store(0);
    }

    void ApplyInnerMaxLatency() {
        // O-A3: keep the real swap chain's latency at min(BufferCount, 2) so FgDxgiPresenter never starves on a
        // game-requested latency of 1, nor lets DXGI queue up excessive frames.
        if (!inner || !(chain_flags & DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT)) return;
        const UINT real_lat = facade.size() >= 2 ? 2u : 1u;
        inner->SetMaximumFrameLatency(real_lat);
    }

    bool WaitGameFence(uint64_t v, DWORD ms) {
        if (!game_fence || v == 0) return true;
        const uint64_t cur = game_fence->GetCompletedValue();
        if (cur == UINT64_MAX) return false;
        if (cur >= v) return true;
        if (!game_fence_event) game_fence_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!game_fence_event || FAILED(game_fence->SetEventOnCompletion(v, game_fence_event))) return false;
        WaitForSingleObject(game_fence_event, ms);
        const uint64_t after = game_fence->GetCompletedValue();
        return after != UINT64_MAX && after >= v;
    }

    void SignalFacadeReadFence(UINT idx) {
        if (!queue || !device || idx >= facade_game_fence.size()) return;
        if (!game_fence && FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&game_fence)))) return;
        if (SUCCEEDED(queue->Signal(game_fence.Get(), ++game_value))) facade_game_fence[idx] = game_value;
    }

    // retained copies of transient tagged textures, per-buffer-type ring (RetainTag); own lock: tagging runs on the
    // game's threads and must not wait behind a present
    static constexpr uint32_t kTagRing = 8;
    struct TagCopy {
        ComPtr<ID3D12Resource> res;
        ComPtr<ID3D12Device> dev;
        UINT64 w = 0;
        UINT h = 0;
        DXGI_FORMAT fmt = DXGI_FORMAT_UNKNOWN;
    };
    struct TagRing {
        TagCopy slots[kTagRing];
        uint32_t next = 0;
    };
    struct RetiredTagCopy {
        ComPtr<ID3D12Resource> res;
        uint64_t retire_present = 0;
    };
    std::mutex tag_mu;
    std::map<uint32_t, TagRing> tag_copies;
    std::vector<RetiredTagCopy> retired_tag_copies;

    void DrainRetiredTags(bool all) {
        std::vector<ComPtr<ID3D12Resource>> drop;
        {
            std::lock_guard<std::mutex> lock(tag_mu);
            if (all) {
                for (auto& e : retired_tag_copies) drop.push_back(std::move(e.res));
                retired_tag_copies.clear();
            } else {
                const uint64_t now = st_presents.load();
                auto it = retired_tag_copies.begin();
                while (it != retired_tag_copies.end()) {
                    if (now >= it->retire_present + kTagRing + 2) {
                        drop.push_back(std::move(it->res));
                        it = retired_tag_copies.erase(it);
                    } else {
                        ++it;
                    }
                }
            }
        }
    }

    void ResetDeviceState(bool reset_failures) {
        DestroySession("device reset");
        if (exec) {
            exec->WaitIdle(kDestroyWaitMs);
            exec.reset();
        }
        WaitCopy(copy_value, kDestroyWaitMs);
        WaitGameFence(game_value, kDestroyWaitMs);
        ReleaseFacade();
        list.Reset();
        for (uint32_t i = 0; i < kCopyRing; ++i) {
            alloc[i].Reset();
            alloc_value[i] = 0;
        }
        copy_fence.Reset();
        copy_value = 0;
        ring = 0;
        game_fence.Reset();
        game_value = 0;
        chain_queue.Reset();
        queue.Reset();
        device.Reset();
        present_queue.Reset();
        present_device.Reset();
        inner = nullptr;
        hwnd = nullptr;
        chain_flags = 0;
        chain_format = DXGI_FORMAT_UNKNOWN;
        color_space = WG_FG_CS_SDR;
        width = 0;
        height = 0;
        {
            std::lock_guard<std::mutex> clock(credit_mu);
            if (latency_sem) {
                CloseHandle(latency_sem);
                latency_sem = nullptr;
            }
            waitable_handed_out = false;
            active_real_frames.clear();
            in_flight_real = 0;
            credits_to_absorb = 0;
            skip_next_copy_credit = false;
            st_in_flight_real.store(0);
        }
        {
            std::lock_guard<std::mutex> tag_lock(tag_mu);
            tag_copies.clear();
            retired_tag_copies.clear();
        }
        if (reset_failures) {
            stats.session_failures = 0;
            stats.gave_up = false;
            if (st_phase.load() == kFgPhaseStopped) st_phase.store(kFgPhaseWaiting);
        }
    }

    void Log(const std::string& line) const {
        if (cfg.log) cfg.log(cfg.log_ctx, line.c_str());
    }

    // The first RDNA 4 hardware adapter of `factory`; false when it lists none.
    // Identifies hardware by the kernel-mode driver's PCI vendor/device IDs for the adapter's LUID (D3DKMTQueryAdapterInfo,
    // rdna4_win.h), so DXGI vendor spoofing (e.g. OptiScaler Dxgi=auto/true in Unreal Engine games) never hides RDNA 4.
    static bool FirstRdna4(IUnknown* factory, DXGI_ADAPTER_DESC1* out, uint32_t* real_vendor, uint32_t* real_device,
                           bool* spoofed) {
        ComPtr<IDXGIFactory1> f1;
        if (!factory || FAILED(factory->QueryInterface(IID_PPV_ARGS(&f1)))) return false;
        for (UINT i = 0; i < 16; ++i) {
            ComPtr<IDXGIAdapter1> a;
            if (f1->EnumAdapters1(i, &a) == DXGI_ERROR_NOT_FOUND) break;
            DXGI_ADAPTER_DESC1 d = {};
            if (!a || FAILED(a->GetDesc1(&d)) || (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) continue;
            const wgrt::AdapterIdentity id = wgrt::read_adapter_identity(d.AdapterLuid);
            if (wgrt::rdna4_identity(id)) {
                const bool wine = !id.kmt_known && id.wine_fallback;
                const uint32_t rv = wine ? id.wine_vendor : id.kmt_vendor;
                const uint32_t rd = wine ? id.wine_device : id.kmt_device;
                *out = d;
                if (real_vendor) *real_vendor = rv;
                if (real_device) *real_device = rd;
                if (spoofed) *spoofed = !FgIsRdna4(d.VendorId, d.DeviceId);
                return true;
            }
            // Fallback only when D3DKMT could not open the adapter (e.g. synthetic test doubles without real LUIDs).
            if (!id.kmt_known && !id.wine_fallback && FgIsRdna4(d.VendorId, d.DeviceId)) {
                *out = d;
                if (real_vendor) *real_vendor = d.VendorId;
                if (real_device) *real_device = d.DeviceId;
                if (spoofed) *spoofed = false;
                return true;
            }
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
        uint32_t rv = 0, rd = 0;
        bool spoofed = false;
        const bool found = FirstRdna4(factory, &d, &rv, &rd, &spoofed);
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
        id.vendor_id = rv;
        id.device_id = rd;
        id.luid.low = d.AdapterLuid.LowPart;
        id.luid.high = d.AdapterLuid.HighPart;
        id.rdna4 = 1;
        id.probes_passed = 1;  // the precheck found every kernel of the executor
        adapter_id = id;
        adapter_set = true;
        adapter_confirmed = game_factory;
        char line[192];
        if (spoofed)
            wsprintfA(line, "fg host: adapter vendor 0x%04x device 0x%04x (RDNA 4 via driver PCI, DXGI reported 0x%04x:0x%04x, %s)",
                      rv, rd, d.VendorId, d.DeviceId, game_factory ? "the game's factory" : "start-up probe");
        else
            wsprintfA(line, "fg host: adapter vendor 0x%04x device 0x%04x (RDNA 4, %s)", rv, rd,
                      game_factory ? "the game's factory" : "start-up probe");
        Log(line);
        c->set_adapter(id);
    }

    void ReleaseFacade() {
        facade.clear();
        facade_game_fence.clear();
        facade_copy_fence.clear();
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
        DXGI_FORMAT cast_pair[2] = {};
        const bool want_cast = FacadeCastablePair(sd.Format, cast_pair);
#ifdef __ID3D12Device10_INTERFACE_DEFINED__
        ComPtr<ID3D12Device10> dev10;
        if (want_cast) device.As(&dev10);
        D3D12_RESOURCE_DESC1 rd1 = {};
        rd1.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        rd1.Width = sd.Width;
        rd1.Height = sd.Height;
        rd1.DepthOrArraySize = 1;
        rd1.MipLevels = 1;
        rd1.Format = sd.Format;
        rd1.SampleDesc.Count = 1;
        rd1.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
#else
        (void)want_cast;
        (void)cast_pair;
#endif
        facade.resize(sd.BufferCount);
        for (auto& f : facade) {
            HRESULT hr = E_FAIL;
#ifdef __ID3D12Device10_INTERFACE_DEFINED__
            if (dev10) {
                hr = dev10->CreateCommittedResource3(&hp, D3D12_HEAP_FLAG_NONE, &rd1, D3D12_BARRIER_LAYOUT_COMMON,
                                                     nullptr, nullptr, 2, cast_pair, IID_PPV_ARGS(&f));
            }
#endif
            if (FAILED(hr)) {
                hr = device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COMMON,
                                                     nullptr, IID_PPV_ARGS(&f));
            }
            if (FAILED(hr)) {
                ReleaseFacade();
                return false;
            }
        }
        facade_game_fence.assign(sd.BufferCount, 0);
        facade_copy_fence.assign(sd.BufferCount, 0);
        facade_index = 0;
        stats.facade = true;
        ApplyInnerMaxLatency();
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
        if (!copy_fence) return true;
        const uint64_t cur = copy_fence->GetCompletedValue();
        if (cur == UINT64_MAX) return false;
        if (cur >= v) return true;
        if (FAILED(copy_fence->SetEventOnCompletion(v, copy_event))) return false;
        WaitForSingleObject(copy_event, ms);
        const uint64_t after = copy_fence->GetCompletedValue();
        return after != UINT64_MAX && after >= v;
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
        if (FAILED(list->Close())) {
            list.Reset();
            return false;
        }
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
        const UINT fi = facade_index % static_cast<UINT>(facade.size());
        if (fi < facade_copy_fence.size()) facade_copy_fence[fi] = copy_value;
        ++ring;
        return true;
    }

    // A2: advances to the next facade back buffer and gates its reuse on the GPU fences that last read it (never
    // releases the frame-latency semaphore here — credits return only when the presenter retires a real frame).
    void AdvanceFacade() {
        if (facade.empty()) return;
        facade_index = (facade_index + 1) % static_cast<UINT>(facade.size());
        if (facade_index < facade_game_fence.size() && facade_game_fence[facade_index] != 0)
            WaitGameFence(facade_game_fence[facade_index], kLatencyWaitMs);
        if (facade_index < facade_copy_fence.size() && facade_copy_fence[facade_index] != 0)
            WaitCopy(facade_copy_fence[facade_index], kLatencyWaitMs);
    }

    // N-B3: at most two intervals of executor work in flight on the compute timeline before the next submission.
    void BoundQueueDepth() {
        if (!exec) return;
        const uint64_t want = last_values[0];
        last_values[0] = last_values[1];
        last_values[1] = exec->LastComputeValue();
        if (want == 0) return;
        exec->WaitForComputeValue(want, 50);
    }

    void DestroySession(const char* why) {
        if (!session) {
            exec_source.reset();
            win_source.reset();
            DrainInFlightCredits();
            return;
        }
        if (registered_worker_tid != 0) {
            FgSwapChain::UnregisterWorkerThread(registered_worker_tid);
            registered_worker_tid = 0;
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
            // A4: truthful per-stage funnel + sustained multiplier + display-time provenance + A1 worker re-entry count
            const wgfg::FgPresentStats ps = pr->Stats();
            const double sust =
                ps.displayed_real > 0
                    ? static_cast<double>(ps.displayed_real + ps.displayed_generated) /
                          static_cast<double>(ps.displayed_real)
                    : 1.0;
            char buf[448];
            std::snprintf(buf, sizeof buf,
                          "fg host: presenter stats: displayed_real=%llu displayed_generated=%llu generated_queued=%llu "
                          "dropped_late=%llu dropped_pressure=%llu cancelled=%llu display_skipped=%llu "
                          "max_slip_us=%lld budget_ms=%.1f | funnel: req=%llu sched=%llu res=%llu sub=%llu "
                          "pre_drop=%llu post_drop=%llu real_sub=%llu | sustained_mult=%.2fx display_time=%s "
                          "worker_reentries=%llu",
                          static_cast<unsigned long long>(ps.displayed_real),
                          static_cast<unsigned long long>(ps.displayed_generated),
                          static_cast<unsigned long long>(ps.generated_queued),
                          static_cast<unsigned long long>(ps.dropped_late),
                          static_cast<unsigned long long>(ps.dropped_pressure),
                          static_cast<unsigned long long>(ps.cancelled),
                          static_cast<unsigned long long>(ps.display_skipped),
                          static_cast<long long>(ps.max_slip / 10), budget / 10000.0,
                          static_cast<unsigned long long>(ps.generated_requested),
                          static_cast<unsigned long long>(ps.generated_scheduled),
                          static_cast<unsigned long long>(ps.generated_reserved),
                          static_cast<unsigned long long>(ps.generated_submitted),
                          static_cast<unsigned long long>(ps.generated_dropped_pre_queue),
                          static_cast<unsigned long long>(ps.generated_dropped_post_queue),
                          static_cast<unsigned long long>(ps.real_submitted), sust,
                          ps.display_time_measured ? "measured" : "submit-estimate",
                          static_cast<unsigned long long>(FgSwapChain::TotalWorkerReentries()));
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
            uint64_t ret = 0, cw = 0;
            UINT lim = 0;
            {
                std::lock_guard<std::mutex> clock(credit_mu);
                ret = real_retires;
                cw = credit_waits;
                lim = latency_limit;
            }
            reasons += "; back-pressure waits=" + std::to_string(back_pressure_waits) +
                       " credit_waits=" + std::to_string(cw) + " real_retires=" + std::to_string(ret) +
                       " latency_limit=" + std::to_string(lim);
            Log(reasons);
            LogTraceSummary(why);
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
        DrainInFlightCredits();
        for (uint64_t& v : last_values) v = 0;
        window_frames = 0;
        window_have = false;
        quiet_windows = 0;
        cap_miss_windows = 0;
        lateness_count_cap = 0;
        ++stats.sessions_destroyed;
    }

    // A4 / O-A7: truthful lateness control using generated_scheduled as denominator (zero double counting),
    // budget capped at kBudgetMax (8.0 ms), and periodic A3 trace summaries + sustained multiplier update.
    void ControlLateness() {
        if (!session) return;
        const wgfg::IFgPresenter* pr = session->Presenter();
        if (!pr) return;
        if (++trace_since_summary >= kTraceSummaryInterval) {
            trace_since_summary = 0;
            LogTraceSummary("periodic");
        }
        if (++window_frames < kBudgetWindow) return;
        window_frames = 0;
        const wgfg::FgPresentStats st = pr->Stats();
        if (!window_have) {
            window_base = st;
            window_have = true;
            return;
        }
        const uint64_t sched = st.generated_scheduled - window_base.generated_scheduled;
        const uint64_t disp_g = st.displayed_generated - window_base.displayed_generated;
        const uint64_t disp_r = st.displayed_real - window_base.displayed_real;
        const uint64_t late = st.dropped_late - window_base.dropped_late;
        const uint64_t pressure = st.dropped_pressure - window_base.dropped_pressure;
        const uint64_t skipped = st.display_skipped - window_base.display_skipped;
        window_base = st;
        if (disp_r > 0) {
            const uint32_t m100 =
                static_cast<uint32_t>(((disp_r + disp_g) * 100ull + disp_r / 2) / disp_r);
            st_sustained_mult_x100.store(m100);
        }
        if (sched < 10) return;
        const uint64_t missed = late + pressure;
        const uint64_t on_time = sched > missed ? sched - missed : 0;
        wgfg::FgTicks next = budget;
        if (missed * 10 > sched) {  // more than 1 in 10 scheduled images missed
            quiet_windows = 0;
            if (budget < kBudgetMax) {
                cap_miss_windows = 0;
                next = budget + std::max(kBudgetStepMin, budget / 4);
                if (next > kBudgetMax) next = kBudgetMax;
            } else if (++cap_miss_windows >= 2) {
                // O-A7: budget is already at the 8.0 ms ceiling and misses persist -> lower scheduled count
                cap_miss_windows = 0;
                const uint32_t cur_n = st_sched_count.load();
                const uint32_t target_cap = cur_n > 1 ? cur_n - 1 : 1u;
                if (lateness_count_cap == 0 || target_cap < lateness_count_cap) {
                    lateness_count_cap = target_cap;
                    session->SetMaxGeneratedCap(lateness_count_cap);
                    char line[224];
                    std::snprintf(line, sizeof line,
                                  "fg host: budget at %.1f ms cap with %llu/%llu missed; stepping generated count cap "
                                  "down to %u",
                                  budget / 10000.0, static_cast<unsigned long long>(missed),
                                  static_cast<unsigned long long>(sched), lateness_count_cap);
                    Log(line);
                }
            }
        } else if (missed * 50 <= sched) {  // 2% or fewer missed: release cap first, then ease latency back
            cap_miss_windows = 0;
            if (++quiet_windows >= kBudgetQuietWindows) {
                quiet_windows = 0;
                if (lateness_count_cap > 0) {
                    const uint32_t route_max = core ? core->advertised_max_generated() : 5u;
                    lateness_count_cap = (lateness_count_cap + 1 >= route_max) ? 0u : lateness_count_cap + 1;
                    session->SetMaxGeneratedCap(lateness_count_cap);
                    char line[192];
                    std::snprintf(line, sizeof line,
                                  "fg host: clean pacing window; generated count cap raised to %u (0=uncapped)",
                                  lateness_count_cap);
                    Log(line);
                } else {
                    next = budget - kBudgetEase < kBudgetBase ? kBudgetBase : budget - kBudgetEase;
                }
            }
        } else {
            quiet_windows = 0;
            cap_miss_windows = 0;
        }
        if (next == budget || !session->SetFirstImageBudget(next)) return;
        if (budget_lines < 32) {
            ++budget_lines;
            char line[320];
            std::snprintf(line, sizeof line,
                          "fg host: generated on time %llu of %llu scheduled (disp %llu, late %llu, pressure %llu, "
                          "skip %llu) in %u frames: first-image budget %.1f -> %.1f ms%s",
                          static_cast<unsigned long long>(on_time), static_cast<unsigned long long>(sched),
                          static_cast<unsigned long long>(disp_g), static_cast<unsigned long long>(late),
                          static_cast<unsigned long long>(pressure), static_cast<unsigned long long>(skipped),
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
        base.max_pending_intervals = kPresenterMaxPendingIntervals;  // N-B3: aligned to 20 slots (3 full 6x intervals)
        base.dxgi_timer_lead = 2000;                                 // 0.2 ms wake lead for DXGI tearing
        base.dxgi_late_tolerance = 20000;                            // 2.0 ms base tolerance (scaled per slot in worker)
        base.device = device.Get();
        base.queue = chain_queue ? chain_queue.Get() : queue.Get();  // the queue the real chain presents on
        base.swap_chain = inner;
        base.hwnd = hwnd;
        // On tearing/VRR-capable systems let HWFQ use unsnapped SetTargetTime flip metering so arbitrary uncapped
        // framerates are paced smoothly without fixed-grid BeginInterval refusals; AUTO only enables variable_refresh
        // when the chain itself allows tearing (otherwise FgSelectPresenter would pick DxgiFlipTearing and fail).
        const bool chain_tearing = tearing_supported && (chain_flags & DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING) != 0;
        base.display.variable_refresh =
            sd.presenter_mode == WG_FG_PRESENTER_HWFQ ? tearing_supported : chain_tearing;
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
        d.scheduler.first_image_budget = budget;  // 4.0 ms base, capped at 8.0 ms (O-A7)
        d.scheduler.max_lead = 280000;            // 28 ms latency cap (~1 real frame + 8 ms budget, O-A7)
        const bool tearing_mode = sd.presenter_mode == WG_FG_PRESENTER_DXGI_TEARING ||
                                  (sd.presenter_mode == WG_FG_PRESENTER_AUTO && base.display.variable_refresh);
        if (tearing_mode) d.scheduler.bridge_and_submit_margin = 2500;  // 0.25 ms submit margin for DXGI tearing
        {
            // O-A4 / M1: query active monitor refresh via EnumDisplaySettingsExW (fallback to compositor grid).
            // Cap output strictly inside the VRR ceiling (~3% margin: ~349 Hz on a 360 Hz panel) so generated frames
            // never exceed panel refresh and tear.
            wgfg::FgTicks refresh_period = 0;
            uint32_t refresh_hz = 0;
            bool from_mode = QueryMonitorRefresh(hwnd, &refresh_period, &refresh_hz);
            if (!from_mode) {
                wgfg::FgClockMap map;
                wgfg::FgRefreshGrid grid;
                if (wgfg::FgReadCompositorGrid(map, 0, &grid) && grid.period > 0) {
                    refresh_period = grid.period;
                    refresh_hz = static_cast<uint32_t>((10000000ll + grid.period / 2) / grid.period);
                }
            }
            if (refresh_period > 0) {
                st_refresh_hz.store(refresh_hz);
                d.scheduler.display.min_interval = refresh_period + refresh_period / 33;  // ~3% ceiling headroom
                d.scheduler.display.max_interval =
                    base.display.variable_refresh ? 208333 : 1000000;  // 48 Hz VRR floor vs 10 Hz fixed
                d.scheduler.adaptive_count = true;
                const bool fixed_grid_presenter =
                    sd.presenter_mode == WG_FG_PRESENTER_DXGI_VSYNC ||
                    (!base.display.variable_refresh &&
                     (sd.presenter_mode == WG_FG_PRESENTER_HWFQ || sd.presenter_mode == WG_FG_PRESENTER_AUTO));
                if (fixed_grid_presenter) d.scheduler.fixed_refresh_period = refresh_period;
                char line[224];
                std::snprintf(line, sizeof line,
                              "fg host: display refresh %.1f Hz (%s, vrr=%s): VRR ceiling %.1f Hz (min_interval=%lld "
                              "ticks, max_interval=%lld ticks, count adapted)",
                              10000000.0 / static_cast<double>(refresh_period),
                              from_mode ? "display mode" : "compositor grid",
                              base.display.variable_refresh ? "yes" : "no",
                              10000000.0 / static_cast<double>(d.scheduler.display.min_interval),
                              static_cast<long long>(d.scheduler.display.min_interval),
                              static_cast<long long>(d.scheduler.display.max_interval));
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
        // Wire A2 real-frame retirement callback, A3 trace callback, and A1 worker thread re-entry guard.
        if (wgfg::IFgPresenter* pr = session->Presenter()) {
            if (auto* dx = dynamic_cast<wgfg::FgDxgiPresenter*>(pr)) {
                dx->SetRealRetireCallback(&Impl::OnRealRetireThunk, this);
                dx->SetTraceCallback(&Impl::OnTraceThunk, this);
            } else if (auto* cp = dynamic_cast<wgfg::FgWinCompPresenter*>(pr)) {
                cp->SetRealRetireCallback(&Impl::OnRealRetireThunk, this);
                cp->SetTraceCallback(&Impl::OnTraceThunk, this);
            }
            registered_worker_tid = PresenterWorkerTid(pr);
            if (registered_worker_tid != 0) FgSwapChain::RegisterWorkerThread(registered_worker_tid);
        }
        WgFgExecutorV1 h = session->ExecutorHandOff();
        c->set_executor(&h);
        ++stats.sessions_created;
        const wgfg::FgPresenterChoice chosen = win_source->LastSelection().chosen;
        st_presenter.store(static_cast<uint32_t>(chosen));
        Log(std::string("fg host: session created, ") + std::to_string(sd.output_width) + " x " +
            std::to_string(sd.output_height) + ", preset " + std::to_string(sd.preset) + ", N up to " +
            std::to_string(sd.max_generated) + ", net2 " + (cfg.net2_exact ? "exact" : "fast") + ", presenter " +
            wgfg::FgPresenterChoiceName(chosen) + ", latency_limit=" + std::to_string(st_latency_limit.load()));
    }
};

FgAsiSessionHost::FgAsiSessionHost(const FgAsiSessionConfig& cfg, std::unique_ptr<wgfg::FgExecWeights> weights)
    : impl_(new Impl()) {
    impl_->cfg = cfg;
    impl_->weights = std::move(weights);
    impl_->InitTraceCsv();
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
    impl_->ResetDeviceState(false);
    if (impl_->copy_event) CloseHandle(impl_->copy_event);
    if (impl_->game_fence_event) CloseHandle(impl_->game_fence_event);
    if (impl_->wait_timer) CloseHandle(impl_->wait_timer);
}

void FgAsiSessionHost::OnInstalled(FgProxy* core) {
    std::lock_guard<std::recursive_mutex> lock(impl_->mu);
    impl_->core = core;
    if (core) {
        impl_->st_has_mfg_selector.store(core->has_mfg_selector());
        impl_->st_mult_source.store(core->multiplier_source());
        // the precheck passed: the model validated; the presenter is decided at session creation (refused there, not
        // replaced)
        core->set_runtime_facts(true, true);
    }
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
    s.multiplier_source = m.st_mult_source.load();
    s.game_has_mfg_selector = m.st_has_mfg_selector.load();
    s.requested_count = m.st_req_count.load();
    s.scheduled_count = m.st_sched_count.load();
    s.count_cap_reason = m.st_cap_reason.load();
    s.display_refresh_hz = m.st_refresh_hz.load();
    s.sustained_mult_x100 = m.st_sustained_mult_x100.load();
    s.latency_limit = m.st_latency_limit.load();
    s.in_flight_real = m.st_in_flight_real.load();
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
    s.back_pressure_waits = impl_->back_pressure_waits;
    s.noop_fullscreen_ignored = impl_->noop_fullscreen_ignored;
    {
        std::lock_guard<std::mutex> clock(impl_->credit_mu);
        s.credit_waits = impl_->credit_waits;
        s.real_retires = impl_->real_retires;
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
        // read-only: prefer the system's own dxgi.dll (bypassing any game-folder proxy during start-up), else fall back
        // to CreateDXGIFactory1; FirstRdna4 resolves hardware PCI IDs via D3DKMT either way.
        ComPtr<IDXGIFactory1> f;
        HRESULT hr = E_FAIL;
        if (HMODULE sys_dxgi = wgrt::load_system_dll(L"dxgi.dll")) {
            using MakeFactory = HRESULT(WINAPI*)(REFIID, void**);
            if (auto mk = reinterpret_cast<MakeFactory>(
                    reinterpret_cast<void*>(GetProcAddress(sys_dxgi, "CreateDXGIFactory1"))))
                hr = mk(IID_PPV_ARGS(&f));
        }
        if (FAILED(hr) || !f) hr = CreateDXGIFactory1(IID_PPV_ARGS(&f));
        if (FAILED(hr) || !f) {
            // InitializeASI can run inside the host's own start-up, where DXGI may refuse; the probe runs again at the
            // game's first FG support / load call (the transport calls this while FG is not taken) and the game's own
            // factory is checked when the game creates it
            if (++m.probe_failures <= 3) {
                char line[224];
                std::snprintf(line, sizeof line,
                              "fg host: start-up probe: DXGI factory deferred during start-up (0x%08lx, attempt %u); "
                              "tried again at the game's first FG call, and the game's factory is checked when the game creates it",
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
    tex->resource = FgAsiTransport::unwrap_native(tex->resource);
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
        if (!transient) return true;
        if (!cmd || tex->state == WG_STATE_NOT_GIVEN) return false;

        void* native_cmd = FgAsiTransport::unwrap_native(cmd);
        ComPtr<ID3D12GraphicsCommandList> cl;
        if (!native_cmd || FAILED(static_cast<IUnknown*>(native_cmd)->QueryInterface(IID_PPV_ARGS(&cl))) || !cl)
            return false;
        ComPtr<ID3D12Device> dev;
        if (FAILED(src->GetDevice(IID_PPV_ARGS(&dev))) || !dev) return false;

        const DXGI_FORMAT copy_fmt = RetainCopyFormat(sd.Format);
        ComPtr<ID3D12Resource> dst;
        {
            Impl& m = *self->impl_;
            std::lock_guard<std::mutex> lock(m.tag_mu);
            Impl::TagRing& ring = m.tag_copies[buffer_type];
            Impl::TagCopy& e = ring.slots[ring.next % Impl::kTagRing];
            ring.next = (ring.next + 1u) % Impl::kTagRing;
            if (!e.res || e.dev.Get() != dev.Get() || e.w != sd.Width || e.h != sd.Height || e.fmt != sd.Format) {
                if (e.res) m.retired_tag_copies.push_back({std::move(e.res), m.st_presents.load()});
                e = Impl::TagCopy();
                D3D12_HEAP_PROPERTIES hp = {};
                hp.Type = D3D12_HEAP_TYPE_DEFAULT;
                D3D12_RESOURCE_DESC cd = sd;
                cd.Format = copy_fmt;
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

        const D3D12_RESOURCE_STATES src_state = static_cast<D3D12_RESOURCE_STATES>(tex->state);
        const bool trans_src = !(src_state & D3D12_RESOURCE_STATE_COPY_SOURCE);
        const bool full_copy = sd.MipLevels == 1 && sd.DepthOrArraySize == 1;
        const UINT src_sub = full_copy ? D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES : tex->subresource;
        D3D12_RESOURCE_BARRIER pre[2] = {};
        UINT npre = 0;
        if (trans_src) {
            pre[npre].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            pre[npre].Transition = {src.Get(), src_sub, src_state, D3D12_RESOURCE_STATE_COPY_SOURCE};
            ++npre;
        }
        pre[npre].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        pre[npre].Transition = {dst.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COMMON,
                                D3D12_RESOURCE_STATE_COPY_DEST};
        ++npre;
        cl->ResourceBarrier(npre, pre);
        if (full_copy) {
            cl->CopyResource(dst.Get(), src.Get());
        } else {
            D3D12_TEXTURE_COPY_LOCATION dloc = {}, sloc = {};
            dloc.pResource = dst.Get();
            dloc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            dloc.SubresourceIndex = 0;
            sloc.pResource = src.Get();
            sloc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            sloc.SubresourceIndex = tex->subresource;
            cl->CopyTextureRegion(&dloc, 0, 0, 0, &sloc, nullptr);
        }
        for (UINT k = 0; k < npre; ++k) std::swap(pre[k].Transition.StateBefore, pre[k].Transition.StateAfter);
        cl->ResourceBarrier(npre, pre);
        tex->resource = dst.Get();  // the host's ring keeps the reference
        tex->format = static_cast<uint32_t>(copy_fmt);
        tex->subresource = 0;
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
        const bool fg_enabled = core->state().enabled != 0;
        m.TryCreateSession(core);
        // A3: sample game present arrival AFTER one-time session creation (so cold pipeline compilation never poisons
        // frame 1's target/slip) and BEFORE any host credit/slot wait so throttling never perturbs the cadence estimate.
        const wgfg::FgTicks t_arrival = wgfg::FgInterruptNow();
        // O-A1: enforce the game's frame-latency credit limit even if the game does not wait on the waitable handle.
        if (m.session) m.WaitForCreditRoom(100);
        // N-B3: skip presenter-room back-pressure wait when the in-game FG toggle is OFF.
        if (fg_enabled) m.WaitForPresenterRoom();
        const wgfg::FgTicks t_after_wait = wgfg::FgInterruptNow();
        if (m.session) m.session->SetNextArrival(t_arrival);
        // A2: register real credit BEFORE FgSessionPresent so an immediate presenter worker retirement never races it.
        const bool credit_taken = m.session != nullptr && desc != nullptr;
        if (credit_taken) m.BeginRealCredit(desc->frame_token);
        m.skip_next_copy_credit = false;
        wgfg::FgSessionResult sr;
        const int32_t rc = wgfg::FgSessionPresent(core, m.session.get(), desc, out, &sr);
        const wgfg::FgTicks t_after_record = wgfg::FgInterruptNow();
        m.st_presents.fetch_add(1);
        m.st_has_mfg_selector.store(core->has_mfg_selector());
        m.st_mult_source.store(core->multiplier_source());
        if (out) {
            m.st_req_count.store(out->requested_count);
            if (out->status == WG_FG_PRESENT_GENERATE) {
                // N-B7: when switching to a faster preset (e.g. A/B -> C), decay the lateness budget & cap immediately.
                if (m.have_last_preset && out->effective_preset == WG_FG_PRESET_C && m.last_preset != WG_FG_PRESET_C &&
                    (m.budget > kBudgetBase || m.lateness_count_cap > 0)) {
                    m.budget = kBudgetBase;
                    m.lateness_count_cap = 0;
                    m.cap_miss_windows = 0;
                    m.quiet_windows = 0;
                    if (m.session) {
                        m.session->SetFirstImageBudget(m.budget);
                        m.session->SetMaxGeneratedCap(0);
                    }
                    m.Log("fg host: preset switched to C (Lite); first-image budget reset to 4.0 ms");
                }
                m.have_last_preset = true;
                m.last_preset = out->effective_preset;
                m.st_preset.store(out->effective_preset);
                m.st_count.store(sr.scheduled > 0 ? sr.scheduled : out->effective_count);
                m.st_sched_count.store(sr.scheduled);
                uint32_t cap_r = kFgCapNone;
                if (m.session) {
                    const wgfg::FgRealOnlyReason rr = m.session->LastReductionReason();
                    if (rr == wgfg::FgRealOnlyReason::OutOfVrrWindow) cap_r = kFgCapDisplayVrr;
                    else if (rr == wgfg::FgRealOnlyReason::LatencyCap) cap_r = kFgCapLatencyCap;
                    else if (rr == wgfg::FgRealOnlyReason::Pressure) cap_r = kFgCapPressure;
                }
                if (cap_r == kFgCapNone && out->requested_count > out->effective_count) cap_r = kFgCapRoute;
                m.st_cap_reason.store(cap_r);
                m.st_phase.store(kFgPhaseGenerating);
                m.ControlLateness();
            } else if (out->status == WG_FG_PRESENT_REAL_ONLY || out->status == WG_FG_PRESENT_REFUSED) {
                m.st_reason.store(out->reason);
                m.st_real_only.store(out->real_only);
                m.st_sched_count.store(0);
                m.st_cap_reason.store(kFgCapNone);
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
            if (m.handled) {
                // A2: signal the game-queue read fence for the current facade buffer before AdvanceFacade can reuse it.
                if (!m.facade.empty()) m.SignalFacadeReadFence(m.facade_index % static_cast<UINT>(m.facade.size()));
                Impl::IntervalTrace tr;
                tr.frame_token = desc->frame_token;
                tr.interval_id = sr.interval_id;
                tr.requested_n = out->requested_count;
                tr.scheduled_n = sr.scheduled;
                tr.in_flight = m.st_in_flight_real.load();
                tr.free_slots = PresenterFreeSlots(m.session->Presenter());
                tr.t_arrival = t_arrival;
                tr.t_after_wait = t_after_wait;
                tr.t_after_record = t_after_record;
                m.RecordTraceInterval(tr);
            } else if (credit_taken) {
                // If CancelRealCredit returns false, the presenter already retired this frame and returned its credit.
                if (!m.CancelRealCredit(desc->frame_token)) m.skip_next_copy_credit = true;
            }
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
        // the session's presenter shows this interval (real and generated images) on the real chain; credit returns
        // asynchronously via OnRealRetireThunk when the real image retires.
        m.handled = false;
        ++m.stats.session_presents;
        m.BoundQueueDepth();
        m.AdvanceFacade();
        m.DrainRetiredTags(false);
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
    bool copied = false;
    if (!m.facade.empty() && !call.retry) {
        if (m.chain_flags & DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT) {
            HANDLE w = m.inner ? m.inner->GetFrameLatencyWaitableObject() : nullptr;
            if (w) {
                WaitForSingleObject(w, kLatencyWaitMs);
                CloseHandle(w);
            }
        }
        if (m.CopyFacadeToChain()) {
            copied = true;
            ++m.stats.copy_presents;
        }
    }
    const HRESULT hr = call.present1 ? inner.Present1(call.sync_interval, call.flags, call.params)
                                     : inner.Present(call.sync_interval, call.flags);
    // A2: on DXGI_ERROR_WAS_STILL_DRAWING the interval stays open for retry: do not advance facade or return credit.
    if (hr != DXGI_ERROR_WAS_STILL_DRAWING) {
        if (copied) m.AdvanceFacade();
        if (!m.facade.empty()) {
            if (!m.skip_next_copy_credit) m.ReturnCopyPresentCredit();
            m.skip_next_copy_credit = false;
            m.DrainRetiredTags(false);
        }
    }
    return hr;
}

void FgAsiSessionHost::BeforeResize() {
    std::lock_guard<std::recursive_mutex> lock(impl_->mu);
    impl_->DestroySession("resize");
    impl_->WaitCopy(impl_->copy_value, kDestroyWaitMs);
    impl_->WaitGameFence(impl_->game_value, kDestroyWaitMs);
    impl_->ReleaseFacade();
    impl_->DrainRetiredTags(true);
}

void FgAsiSessionHost::AfterResize(HRESULT hr) {
    std::lock_guard<std::recursive_mutex> lock(impl_->mu);
    if (FAILED(hr)) {
        impl_->Log("fg host: resize failed; facade not recreated");
        return;
    }
    if (!impl_->CreateFacade()) impl_->Log("fg host: facade not recreated after the resize");
}

void FgAsiSessionHost::BeforeFullscreen(BOOL fullscreen) {
    Impl& m = *impl_;
    std::lock_guard<std::recursive_mutex> lock(m.mu);
    // N-S3: ignore no-op SetFullscreenState(FALSE) when already windowed/borderless (e.g. Cyberpunk settings changes).
    if (m.inner) {
        BOOL cur_fs = FALSE;
        IDXGIOutput* out = nullptr;
        if (SUCCEEDED(m.inner->GetFullscreenState(&cur_fs, &out))) {
            if (out) out->Release();
            if ((cur_fs != FALSE) == (fullscreen != FALSE)) {
                ++m.noop_fullscreen_ignored;
                if (!m.logged_noop_fs) {
                    m.logged_noop_fs = true;
                    m.Log(std::string("fg host: ignoring no-op SetFullscreenState(") + (fullscreen ? "TRUE" : "FALSE") +
                          ") (state unchanged; session kept)");
                }
                return;
            }
        }
    }
    m.DestroySession("fullscreen change");
}

void FgAsiSessionHost::AfterFullscreen(BOOL, HRESULT) {}

void FgAsiSessionHost::ColorSpaceChanged(DXGI_COLOR_SPACE_TYPE cs) {
    Impl& m = *impl_;
    std::lock_guard<std::recursive_mutex> lock(m.mu);
    uint32_t next_cs = WG_FG_CS_SDR;
    if (cs == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020 || cs == DXGI_COLOR_SPACE_RGB_STUDIO_G2084_NONE_P2020)
        next_cs = WG_FG_CS_HDR10_PQ;
    else if (cs == DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709)
        next_cs = WG_FG_CS_SCRGB_LINEAR;
    const bool was_hdr = m.color_space != WG_FG_CS_SDR;
    const bool now_hdr = next_cs != WG_FG_CS_SDR;
    m.color_space = next_cs;
    if (m.core) m.core->set_color_space(next_cs);
    if (was_hdr != now_hdr && m.session) m.DestroySession("color space change");
}

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
    Impl& m = *impl_;
    std::lock_guard<std::recursive_mutex> lock(m.mu);
    if (m.facade.empty() || !handle || !(m.chain_flags & DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT))
        return false;
    std::lock_guard<std::mutex> clock(m.credit_mu);
    m.EnsureLatencySemLocked();
    if (!m.latency_sem) return false;
    m.waitable_handed_out = true;
    // the caller closes what it gets, as with the chain's own object; underlying handle persists across resizes
    HANDLE dup = nullptr;
    if (!DuplicateHandle(GetCurrentProcess(), m.latency_sem, GetCurrentProcess(), &dup, 0, FALSE,
                         DUPLICATE_SAME_ACCESS))
        return false;
    *handle = dup;
    return true;
}

bool FgAsiSessionHost::SetMaximumFrameLatency(UINT latency, HRESULT* hr) {
    Impl& m = *impl_;
    std::lock_guard<std::recursive_mutex> lock(m.mu);
    if (m.facade.empty() || !hr) return false;
    if (latency < 1 || latency > 16) {
        *hr = DXGI_ERROR_INVALID_CALL;
        return true;
    }
    UINT prev_limit = 0;
    {
        std::lock_guard<std::mutex> clock(m.credit_mu);
        prev_limit = m.latency_limit;
        int32_t delta = static_cast<int32_t>(latency) - static_cast<int32_t>(m.latency_limit);
        m.latency_limit = latency;
        m.st_latency_limit.store(latency);
        if (m.latency_sem && m.waitable_handed_out) {
            if (delta > 0) {
                const int32_t cancel = delta < m.credits_to_absorb ? delta : m.credits_to_absorb;
                m.credits_to_absorb -= cancel;
                delta -= cancel;
                if (delta > 0) ReleaseSemaphore(m.latency_sem, delta, nullptr);
            } else if (delta < 0) {
                int32_t need = -delta;
                while (need > 0 && WaitForSingleObject(m.latency_sem, 0) == WAIT_OBJECT_0) --need;
                m.credits_to_absorb += need;
            }
        }
        m.credit_cv.notify_all();
    }
    if (prev_limit != latency) {
        char buf[160];
        std::snprintf(buf, sizeof(buf),
                      "fg host: game SetMaximumFrameLatency(%u) -> facade credit limit %u -> %u (real chain stays at %u)",
                      latency, prev_limit, latency, m.facade.size() >= 2 ? 2u : 1u);
        m.Log(buf);
    }
    *hr = S_OK;
    return true;
}

bool FgAsiSessionHost::GetMaximumFrameLatency(UINT* latency, HRESULT* hr) {
    Impl& m = *impl_;
    std::lock_guard<std::recursive_mutex> lock(m.mu);
    if (m.facade.empty() || !hr) return false;
    if (!latency) {
        *hr = E_POINTER;
        return true;
    }
    std::lock_guard<std::mutex> clock(m.credit_mu);
    *latency = m.latency_limit;
    *hr = S_OK;
    return true;
}

void FgAsiSessionHost::SetGameWaitable(bool requested) {
    std::lock_guard<std::recursive_mutex> lock(impl_->mu);
    impl_->pending_game_waitable = requested;
}

void FgAsiSessionHost::Detached() {
    std::lock_guard<std::recursive_mutex> lock(impl_->mu);
    impl_->ResetDeviceState(false);
}

void FgAsiSessionHost::Attached(IDXGISwapChain4* inner, IUnknown* queue) {
    Impl& m = *impl_;
    std::lock_guard<std::recursive_mutex> lock(m.mu);
    if (!inner || !queue) return;
    ComPtr<ID3D12CommandQueue> next_present_q = std::move(m.attached_present_queue);
    m.attached_present_queue.Reset();
    void* native_q = FgAsiTransport::unwrap_native(queue);
    ComPtr<ID3D12CommandQueue> q;
    ComPtr<ID3D12Device> dev;
    if (!native_q || FAILED(static_cast<IUnknown*>(native_q)->QueryInterface(IID_PPV_ARGS(&q))) ||
        FAILED(q->GetDevice(IID_PPV_ARGS(&dev)))) {
        m.Log("fg host: the chain's creation queue is not a D3D12 queue; frame generation is not taken");
        return;
    }
    const bool new_device = m.device && m.device.Get() != dev.Get();
    if (m.inner || new_device) {
        m.ResetDeviceState(new_device);
    }
    m.inner = inner;
    m.queue = q;
    m.device = dev;
    m.chain_queue = next_present_q ? next_present_q : q;
    if (FAILED(inner->GetHwnd(&m.hwnd))) m.hwnd = nullptr;
    m.game_requested_waitable = m.pending_game_waitable;
    m.pending_game_waitable = false;
    m.logged_noop_fs = false;
    {
        std::lock_guard<std::mutex> clock(m.credit_mu);
        // DXGI waitable default is 1; when WE injected WAITABLE onto a non-waitable game chain, default to 2
        // (1 interval playing out + 1 real frame rendering in parallel).
        m.latency_limit = m.game_requested_waitable ? 1u : 2u;
        m.st_latency_limit.store(m.latency_limit);
        m.in_flight_real = 0;
        m.credits_to_absorb = 0;
        m.st_in_flight_real.store(0);
    }
    if (!m.CreateFacade()) {
        m.Log("fg host: facade buffers not created; the chain's own buffers stay in use");
        m.inner = nullptr;
        m.queue.Reset();
        m.device.Reset();
        m.chain_queue.Reset();
        m.hwnd = nullptr;
        return;
    }
    m.color_space = (m.chain_format == DXGI_FORMAT_R16G16B16A16_FLOAT) ? WG_FG_CS_SCRGB_LINEAR : WG_FG_CS_SDR;
    if (m.core) {
        m.core->set_device(dev.Get(), q.Get());
        m.core->set_color_space(m.color_space);
    }
    char line[192];
    std::snprintf(line, sizeof(line),
                  "fg host: chain attached, %u x %u, format %u, %u buffers (facade), presents on %s, "
                  "game_waitable=%s, facade_latency_limit=%u, real_chain_latency=%u",
                  m.width, m.height, static_cast<unsigned>(m.chain_format), static_cast<unsigned>(m.facade.size()),
                  m.chain_queue.Get() != m.queue.Get() ? "the present queue" : "the game's queue",
                  m.game_requested_waitable ? "yes" : "no", m.st_latency_limit.load(),
                  m.facade.size() >= 2 ? 2u : 1u);
    m.Log(line);
}

IUnknown* FgAsiSessionHost::CreationQueue(IUnknown* game_queue, const DXGI_SWAP_CHAIN_DESC1&) {
    Impl& m = *impl_;
    std::lock_guard<std::recursive_mutex> lock(m.mu);
    try {
        void* native_q = FgAsiTransport::unwrap_native(game_queue);
        ComPtr<ID3D12CommandQueue> gq;
        ComPtr<ID3D12Device> dev;
        if (!native_q || FAILED(static_cast<IUnknown*>(native_q)->QueryInterface(IID_PPV_ARGS(&gq))) || !gq ||
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
