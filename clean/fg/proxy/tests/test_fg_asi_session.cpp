// SPDX-License-Identifier: GPL-3.0-only
// WrongGPU frame generation (issue #96, F9): tests of the FG session host of WrongGPU-FG.asi (win32/fg_asi_session.h).
// Every value is synthetic. Built with CI's flags (/W4 /WX /permissive-).
//
//   test_fg_asi_session                       CPU checks (ctest fg_asi_session): the RDNA 4 ids, the precheck's
//                                             refusals (no kernel folder, a kernel missing, no model) and the swap-chain
//                                             flags per presenter choice
//   test_fg_asi_session --gpu --dxil <dir>    adds a headless GPU leg (no window): a real D3D12 device and a
//   [--debug-layer] [--preset A|C]            composition swap chain created with the host's flags and wrapped by the
//                                             swap-chain hook's wrapper, the adapter core on a stand-in interface chain,
//                                             and the host as the wrapper's callback, driven as the transport's present
//                                             bridge drives it: the facade buffers the game renders, the first present
//                                             through the copy path, the session created from the core's description
//                                             (synthetic weights), generated intervals, frames with generation off
//                                             (real-only from the facade buffer), a resize (the session destroyed and
//                                             created again), and the game's last release (session destroyed, queue
//                                             drained) with the debug layer clean.
//   Both: the OptiScaler menu card (win32/fg_panel.h) through a stand-in host table: registration (a Custom tab card,
//   a tab of its own on a host without cards, refusal on an incomplete table), the status line per state, the preset
//   and multiplier combos handed to the live hook and saved to the scratch game's wronggpu-setup\settings.ini.
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "../../runtime/fg_d3d12_executor.h"
#include "../fg_proxy_core.h"
#include "../../presenter/fg_presenter_factory.h"
#include "../win32/fg_asi_pacing.h"
#include "../win32/fg_asi_session.h"
#include "../win32/fg_asi_transport.h"
#include "../win32/fg_panel.h"
#include "../win32/fg_swapchain_hook.h"
#include "wg_state.h"

using Microsoft::WRL::ComPtr;
using namespace wgfgp;
using namespace wgfgp::win32;

namespace {

// High-resolution wait so synthetic 30 fps frame pacing never quantizes to 46.9 ms (3 * 15.625 ms < 48 Hz VRR floor).
void WaitMsPrecise(DWORD ms) {
    if (ms == 0) return;
    HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
    if (!timer) {
        Sleep(ms);
        return;
    }
    LARGE_INTEGER due;
    due.QuadPart = -static_cast<LONGLONG>(ms) * 10000LL;
    if (SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE)) WaitForSingleObject(timer, ms + 50);
    else Sleep(ms);
    CloseHandle(timer);
}

int g_checks = 0, g_failed = 0;
const char* g_case = "";
#define CHECK(cond)                                                                    \
    do {                                                                               \
        ++g_checks;                                                                    \
        if (!(cond)) {                                                                 \
            ++g_failed;                                                                \
            std::printf("FAIL [%s] %s:%d: %s\n", g_case, __FILE__, __LINE__, #cond);   \
        }                                                                              \
    } while (0)

std::atomic<int> g_log_lines{0};
void PrintLog(void*, const char* line) {
    if (g_log_lines++ < 60) std::printf("      log: %s\n", line);
}

void WriteBytes(const std::string& path, const char* data, size_t n) {
    std::ofstream f(path, std::ios::binary);
    f.write(data, static_cast<std::streamsize>(n));
}

// ---- CPU ------------------------------------------------------------------------------------------------------------

void TestIds() {
    g_case = "rdna4-ids";
    CHECK(FgIsRdna4(0x1002, 0x7550) && FgIsRdna4(0x1002, 0x7551) && FgIsRdna4(0x1002, 0x7590) &&
          FgIsRdna4(0x1002, 0x7591));
    CHECK(!FgIsRdna4(0x10DE, 0x7550) && !FgIsRdna4(0x1002, 0x744C) && !FgIsRdna4(0, 0));
}

void TestPrecheck(const std::string& scratch) {
    g_case = "precheck";
    wgfg::FgExecWeights w;
    FgAsiSessionConfig c;
    c.model_path = scratch + "\\no-model.wgcm";
    c.dxil_dir = scratch + "\\no-such-folder";
    std::string why;
    CHECK(!FgAsiSessionPrecheck(c, &w, &why) && why.find("kernel folder") != std::string::npos);
    std::printf("    no folder: %s\n", why.c_str());
    // an empty folder: the first kernel the executor loads is named
    const std::string empty = scratch + "\\dxil-empty";
    CreateDirectoryA(empty.c_str(), nullptr);
    c.dxil_dir = empty;
    why.clear();
    CHECK(!FgAsiSessionPrecheck(c, &w, &why) && why.find("fg_lite_fill16.dxil") != std::string::npos);
    std::printf("    empty folder: %s\n", why.c_str());
    // every kernel file there (placeholders; the precheck only looks for them), no model: the model check refuses
    const std::string full = scratch + "\\dxil-names";
    CreateDirectoryA(full.c_str(), nullptr);
    for (const std::string& k : wgfg::FgD3d12ExecutorKernelNames(true, wgfg::FgNet2Path::Fast))
        WriteBytes(full + "\\" + k + ".dxil", "DXBC", 4);
    c.dxil_dir = full;
    why.clear();
    CHECK(!FgAsiSessionPrecheck(c, &w, &why) && why.find("did not validate") != std::string::npos);
    std::printf("    no model: %s\n", why.c_str());
    // the exact path needs its own kernels: the fast set alone is not enough
    c.net2_exact = true;
    why.clear();
    CHECK(!FgAsiSessionPrecheck(c, &w, &why) && why.find("fg_net2_conv.dxil") != std::string::npos);
    // a model file that is not a model
    c.net2_exact = false;
    WriteBytes(scratch + "\\junk.wgcm", "not a model", 11);
    c.model_path = scratch + "\\junk.wgcm";
    why.clear();
    CHECK(!FgAsiSessionPrecheck(c, &w, &why) && why.find("did not validate") != std::string::npos);
    CHECK(!FgAsiSessionPrecheck(c, nullptr, &why));
}

void TestFlags() {
    g_case = "swap-chain-flags";
    DXGI_SWAP_CHAIN_DESC1 d = {};
    const FgPresenterChoice choices[4] = {FgPresenterChoice::Auto, FgPresenterChoice::Hwfq, FgPresenterChoice::DxgiTearing,
                                          FgPresenterChoice::DxgiVsync};
    const UINT want[4] = {DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT, 0,
                          DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT,
                          DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT};
    for (int i = 0; i < 4; ++i) {
        FgAsiSessionConfig c;
        c.settings.presenter = choices[i];
        std::unique_ptr<wgfg::FgExecWeights> w(new wgfg::FgExecWeights());
        FgAsiSessionHost h(c, std::move(w));
        // no factory seen yet: tearing support unknown, so never requested
        CHECK(h.ExtraSwapChainFlags(d) == want[i]);
        const FgAsiSessionStats s = h.Stats();
        CHECK(s.sessions_created == 0 && !s.facade && !s.gave_up);
    }
}

// ---- GPU ------------------------------------------------------------------------------------------------------------

constexpr UINT kW = 320, kH = 180;
constexpr int32_t kChainRc = 55;
constexpr uint32_t kNtBackbuffer = 900, kNtMotion = 901, kNtDepth = 902, kNtHudless = 903, kNtUiAlpha = 904,
                   kNtDistortion = 905, kNtNoWarp = 906;
bool g_debug_layer = false;

struct FakeInterface {
    int presents = 0;
    static int32_t IsSupported(void*, uint32_t, const void*) { return kChainRc; }
    static int32_t GetFunction(void*, uint32_t, const char*, void** out) {
        if (out) *out = nullptr;
        return kChainRc;
    }
    static int32_t SetOptions(void*, uint32_t, uint32_t, const void*) { return kChainRc; }
    static int32_t GetState(void*, uint32_t, uint32_t, void*) { return kChainRc; }
    static int32_t Tag(void*, const void*, uint32_t) { return kChainRc; }
    static int32_t Constants(void*, const void*) { return kChainRc; }
    static int32_t Present(void* s, const WgFgPresentDesc*) {
        static_cast<FakeInterface*>(s)->presents++;
        return kChainRc;
    }
    WgFgChainV1 table() {
        WgFgChainV1 t = {};
        t.struct_size = sizeof(t);
        t.abi_version = WG_FG_ADAPTER_ABI_VERSION;
        t.self = this;
        t.max_generated = 5;
        t.is_supported = &IsSupported;
        t.get_function = &GetFunction;
        t.set_options = &SetOptions;
        t.get_state = &GetState;
        t.tag = &Tag;
        t.constants = &Constants;
        t.present = &Present;
        return t;
    }
};
int g_fn_a = 0, g_fn_b = 0;

struct Gpu {
    ComPtr<IDXGIFactory6> factory;
    ComPtr<IDXGIAdapter1> adapter;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    ComPtr<ID3D12InfoQueue> info;
    uint64_t value = 0;
    HANDLE event = nullptr;
    DXGI_ADAPTER_DESC1 desc = {};
    ~Gpu() {
        if (event) CloseHandle(event);
    }
    bool Init() {
        if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)))) return false;
        for (UINT i = 0; factory->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                                                             IID_PPV_ARGS(&adapter)) != DXGI_ERROR_NOT_FOUND;
             ++i) {
            adapter->GetDesc1(&desc);
            if (!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) break;
            adapter.Reset();
        }
        if (g_debug_layer) {
            ComPtr<ID3D12Debug> dbg;
            if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dbg)))) dbg->EnableDebugLayer();
        }
        if (!adapter || FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device))))
            return false;
        if (g_debug_layer) device.As(&info);
        D3D12_COMMAND_QUEUE_DESC q = {};
        q.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        if (FAILED(device->CreateCommandQueue(&q, IID_PPV_ARGS(&queue))) ||
            FAILED(device->CreateCommandAllocator(q.Type, IID_PPV_ARGS(&alloc))) ||
            FAILED(device->CreateCommandList(0, q.Type, alloc.Get(), nullptr, IID_PPV_ARGS(&list))) ||
            FAILED(list->Close()) || FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))))
            return false;
        event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        return event != nullptr;
    }
    bool Wait() {
        queue->Signal(fence.Get(), ++value);
        if (fence->GetCompletedValue() >= value) return true;
        fence->SetEventOnCompletion(value, event);
        return WaitForSingleObject(event, 5000) == WAIT_OBJECT_0;
    }
    int64_t DebugErrors() {
        if (!info) return -1;
        int64_t errors = 0;
        const UINT64 n = info->GetNumStoredMessages();
        for (UINT64 i = 0; i < n; ++i) {
            SIZE_T len = 0;
            info->GetMessage(i, nullptr, &len);
            std::vector<char> buf(len);
            D3D12_MESSAGE* m = reinterpret_cast<D3D12_MESSAGE*>(buf.data());
            if (FAILED(info->GetMessage(i, m, &len))) continue;
            if (m->Severity == D3D12_MESSAGE_SEVERITY_ERROR || m->Severity == D3D12_MESSAGE_SEVERITY_CORRUPTION) {
                if (errors < 8) std::printf("      debug layer: %s\n", m->pDescription);
                ++errors;
            }
        }
        info->ClearStoredMessages();
        debug_errors_seen += errors;
        return errors;
    }
    int64_t debug_errors_seen = 0;
};

// The stand-in game's own textures: motion (zero) and depth (constant), and a colour source copied into the facade
// buffer the game "renders" each frame (a synthetic pattern that moves with the frame number).
struct GameTextures {
    ComPtr<ID3D12Resource> motion, depth, colour_src, upload;
    ComPtr<ID3D12Resource> Tex(Gpu& g, DXGI_FORMAT f, D3D12_RESOURCE_STATES s) {
        D3D12_HEAP_PROPERTIES hp = {};
        hp.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC rd = {};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        rd.Width = kW;
        rd.Height = kH;
        rd.DepthOrArraySize = 1;
        rd.MipLevels = 1;
        rd.Format = f;
        rd.SampleDesc.Count = 1;
        ComPtr<ID3D12Resource> r;
        g.device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, s, nullptr, IID_PPV_ARGS(&r));
        return r;
    }
    bool Init(Gpu& g) {
        motion = Tex(g, DXGI_FORMAT_R16G16_FLOAT, D3D12_RESOURCE_STATE_COMMON);
        depth = Tex(g, DXGI_FORMAT_R32_FLOAT, D3D12_RESOURCE_STATE_COMMON);
        colour_src = Tex(g, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_STATE_COMMON);
        D3D12_HEAP_PROPERTIES hp = {};
        hp.Type = D3D12_HEAP_TYPE_UPLOAD;
        D3D12_RESOURCE_DESC rd = {};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width = 2ull * 1024 * 1024;
        rd.Height = 1;
        rd.DepthOrArraySize = 1;
        rd.MipLevels = 1;
        rd.SampleDesc.Count = 1;
        rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        g.device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                          IID_PPV_ARGS(&upload));
        return motion && depth && colour_src && upload;
    }
    // frame f's colour into `dst` (the facade buffer, left in COMMON as a presented buffer is); motion 0, depth 0.5
    bool Render(Gpu& g, int f, ID3D12Resource* dst) {
        uint8_t* p = nullptr;
        if (FAILED(upload->Map(0, nullptr, reinterpret_cast<void**>(&p)))) return false;
        const UINT pitch = (kW * 8 + 255) & ~255u;
        for (UINT y = 0; y < kH; ++y)
            for (UINT x = 0; x < kW; ++x) {
                uint16_t* t = reinterpret_cast<uint16_t*>(p + static_cast<size_t>(y) * pitch + 8ull * x);
                const UINT v = (x + 3 * static_cast<UINT>(f)) % 64;
                t[0] = static_cast<uint16_t>(0x3000 + v * 8);  // small positive float16 values
                t[1] = static_cast<uint16_t>(0x3400 + ((y / 8) % 32) * 8);
                t[2] = 0x3800;
                t[3] = 0x3C00;
            }
        upload->Unmap(0, nullptr);
        if (FAILED(g.alloc->Reset()) || FAILED(g.list->Reset(g.alloc.Get(), nullptr))) return false;
        D3D12_RESOURCE_BARRIER b = {};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition = {dst, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COMMON,
                        D3D12_RESOURCE_STATE_COPY_DEST};
        g.list->ResourceBarrier(1, &b);
        D3D12_TEXTURE_COPY_LOCATION d = {}, s = {};
        d.pResource = dst;
        d.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        s.pResource = upload.Get();
        s.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        s.PlacedFootprint.Footprint = {DXGI_FORMAT_R16G16B16A16_FLOAT, kW, kH, 1, pitch};
        g.list->CopyTextureRegion(&d, 0, 0, 0, &s, nullptr);
        std::swap(b.Transition.StateBefore, b.Transition.StateAfter);
        g.list->ResourceBarrier(1, &b);
        if (FAILED(g.list->Close())) return false;
        ID3D12CommandList* l[] = {g.list.Get()};
        g.queue->ExecuteCommandLists(1, l);
        return g.Wait();
    }
};

FgProxyConfig CoreConfig(uint32_t preset) {
    FgProxyConfig c;
    c.settings.enabled = true;
    c.settings.preset = preset;
    c.settings.multiplier = 2;
    c.settings.presenter = FgPresenterChoice::DxgiTearing;
    c.binding.qualified = true;
    c.binding.version_major = 2;
    c.binding.version_minor = 14;
    const uint32_t nt[WG_FG_ROLE_COUNT] = {kNtBackbuffer, kNtMotion, kNtDepth, kNtHudless, kNtUiAlpha, kNtDistortion,
                                           kNtNoWarp};
    for (uint32_t r = 0; r < WG_FG_ROLE_COUNT; ++r) c.binding.native_type[r] = nt[r];
    c.binding.fn_names = {"fgTestSetOptions", "fgTestGetState"};
    c.binding.ui_max_generated = 5;
    c.modes_qualified = true;  // the release gate is closed in this build; this test exercises the taken route
    return c;
}

WgFgOptions Options(bool enable, uint32_t preset) {
    WgFgOptions o = {};
    o.struct_size = sizeof(o);
    o.abi_version = WG_FG_ADAPTER_ABI_VERSION;
    o.enable = enable ? 1u : 0u;
    o.preset = preset;
    o.generated_count = 1;
    for (uint32_t i = 0; i < WG_FG_OPTIONAL_COUNT; ++i) o.optional_decl[i] = WG_FG_DECL_MAY;
    o.color_space = WG_FG_CS_SDR;
    o.output_width = kW;
    o.output_height = kH;
    o.output_format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    o.render_min_width = o.render_max_width = kW;
    o.render_min_height = o.render_max_height = kH;
    return o;
}

void RunHost(Gpu& gpu, const std::string& dxil, uint32_t preset) {
    std::printf("  host, preset %s\n", preset == 0 ? "A" : "C");
    GameTextures game;
    CHECK(game.Init(gpu));
    FgPresentOwners owners;
    FakeInterface iface;
    FgProxy core(CoreConfig(preset), &owners, &PrintLog, nullptr);
    const WgFgChainV1 ct = iface.table();
    CHECK(core.attach_interposer(&ct, 2, 14) == WG_FG_REASON_NONE);
    core.set_feature_function("fgTestSetOptions", &g_fn_a);
    core.set_feature_function("fgTestGetState", &g_fn_b);

    FgAsiSessionConfig hc;
    hc.settings = CoreConfig(preset).settings;
    hc.dxil_dir = dxil;
    hc.log = &PrintLog;
    std::unique_ptr<wgfg::FgExecWeights> w(new wgfg::FgExecWeights());
    wgfg::FgExecSyntheticWeights(11, w.get());
    std::unique_ptr<FgAsiSessionHost> host(new FgAsiSessionHost(hc, std::move(w)));
    host->OnInstalled(&core);
    FgAsiSessionHost::FactoryNote(host.get(), &core, gpu.factory.Get());
    CHECK(core.fg_taken());
    if (!core.fg_taken()) return;

    // the game's chain, created with the host's flags (as the creation hook adds them), then wrapped and claimed
    DXGI_SWAP_CHAIN_DESC1 sd = {};
    sd.Width = kW;
    sd.Height = kH;
    sd.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    sd.SampleDesc.Count = 1;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.BufferCount = 3;
    sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    sd.AlphaMode = DXGI_ALPHA_MODE_IGNORE;
    const UINT extra = host->ExtraSwapChainFlags(sd);
    CHECK(extra & DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT);
    sd.Flags |= extra;
    ComPtr<IDXGIFactory2> f2;
    CHECK(SUCCEEDED(gpu.factory.As(&f2)));
    IDXGISwapChain1* sc1 = nullptr;
    CHECK(SUCCEEDED(f2->CreateSwapChainForComposition(gpu.queue.Get(), &sd, nullptr, &sc1)) && sc1);
    if (!sc1) return;
    FgSwapChain* chain = nullptr;
    CHECK(SUCCEEDED(FgSwapChain::Create(sc1, gpu.queue.Get(), &PrintLog, nullptr, &chain)) && chain);
    if (!chain) return;
    chain->SetExtraFlags(extra);
    CHECK(chain->Attach(host.get(), &core, &owners, WG_FG_OWNER_SWAPCHAIN_WRAPPER) == WG_FG_REASON_NONE);
    host->Attached(chain->Inner(), chain->Queue());
    CHECK(host->Stats().facade);

    // the game-facing buffers are the facade's, not the chain's
    {
        ComPtr<ID3D12Resource> mine, real;
        CHECK(SUCCEEDED(chain->GetBuffer(0, IID_PPV_ARGS(&mine))) &&
              SUCCEEDED(chain->Inner()->GetBuffer(0, IID_PPV_ARGS(&real))) && mine && real && mine.Get() != real.Get());
        HANDLE lat = chain->GetFrameLatencyWaitableObject();  // the facade's (the game waits on it each frame below)
        CHECK(lat != nullptr);
        if (lat) CloseHandle(lat);
    }
    const WgFgOptions on = Options(true, preset), off = Options(false, preset);
    core.set_options(WG_FEAT_FRAME_GENERATION, 1, &on, &on);

    const int frames = 24, off_from = 8, off_to = 11, resize_at = 16;
    int present_errors = 0, latency_waits = 0;
    UINT last_index = 99;
    bool index_moves = true;
    for (int f = 0; f < frames; ++f) {
        const uint64_t tok = static_cast<uint64_t>(f) + 1;
        if (f == off_from) core.set_options(WG_FEAT_FRAME_GENERATION, 1, &off, &off);
        if (f == off_to) core.set_options(WG_FEAT_FRAME_GENERATION, 1, &on, &on);
        if (f == resize_at) {
            // the game lets go of its buffer references, then resizes (same size: a new chain epoch)
            const HRESULT hr = chain->ResizeBuffers(0, kW, kH, DXGI_FORMAT_UNKNOWN, 0);
            CHECK(SUCCEEDED(hr));  // the creation's extra flags were kept on the chain
        }
        HANDLE lat = chain->GetFrameLatencyWaitableObject();
        if (lat) {
            if (WaitForSingleObject(lat, 200) == WAIT_OBJECT_0) ++latency_waits;
            CloseHandle(lat);
        }
        const UINT bi = chain->GetCurrentBackBufferIndex();
        if (f > 0 && f != resize_at && bi != (last_index + 1) % 3) index_moves = false;
        last_index = bi;
        ComPtr<ID3D12Resource> bb;
        CHECK(SUCCEEDED(chain->GetBuffer(bi, IID_PPV_ARGS(&bb))) && bb);
        if (!bb) break;
        CHECK(game.Render(gpu, f, bb.Get()));
        WgFgTag tags[2] = {};
        ID3D12Resource* res[2] = {game.motion.Get(), game.depth.Get()};
        const uint32_t nts[2] = {kNtMotion, kNtDepth};
        const uint32_t fm[2] = {DXGI_FORMAT_R16G16_FLOAT, DXGI_FORMAT_R32_FLOAT};
        for (int t = 0; t < 2; ++t) {
            tags[t].struct_size = sizeof(WgFgTag);
            tags[t].native_type = nts[t];
            tags[t].texture.resource = res[t];
            tags[t].texture.format = fm[t];
            tags[t].texture.width = kW;
            tags[t].texture.height = kH;
            tags[t].texture.state = D3D12_RESOURCE_STATE_COMMON;
            tags[t].texture.frame_token = tok;
            tags[t].texture.viewport_id = 1;
            tags[t].texture.producer_epoch = 3;
            tags[t].texture.rect = {0, 0, kW, kH};
        }
        core.tag(tags, 2, tags);
        WgFgConstants k = {};
        k.struct_size = sizeof(k);
        k.abi_version = WG_FG_ADAPTER_ABI_VERSION;
        k.frame_token = tok;
        k.viewport_id = 1;
        for (int i = 0; i < 4; ++i) k.camera_current[5 * i] = k.camera_previous[5 * i] = 1.0f;
        k.motion_scale_x = k.motion_scale_y = 1.0f;
        k.render_rect = {0, 0, kW, kH};
        k.output_rect = {0, 0, kW, kH};
        core.constants(&k, &k);
        // the present bridge's descriptor: the facade buffer the game rendered is the colour input
        WgFgPresentDesc d = {};
        d.struct_size = sizeof(d);
        d.abi_version = WG_FG_ADAPTER_ABI_VERSION;
        d.owner = WG_FG_OWNER_SWAPCHAIN_WRAPPER;
        d.viewport_id = 1;
        d.frame_token = tok;
        d.swapchain_id = chain->Id();
        d.swapchain_epoch = chain->Epoch();
        d.window_id = 0x77;
        d.luid.low = gpu.desc.AdapterLuid.LowPart;
        d.luid.high = gpu.desc.AdapterLuid.HighPart;
        d.backbuffer.resource = bb.Get();
        d.backbuffer.format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        d.backbuffer.width = kW;
        d.backbuffer.height = kH;
        d.backbuffer.state = 0;
        d.backbuffer.frame_token = tok;
        d.backbuffer.viewport_id = 1;
        d.backbuffer.rect = {0, 0, kW, kH};
        bb.Reset();
        WgFgPresentResult r = {};
        r.struct_size = sizeof(r);
        r.abi_version = WG_FG_ADAPTER_ABI_VERSION;
        FgAsiSessionHost::Route(host.get(), &core, &d, &r);
        const HRESULT hr = chain->Present(0, 0);
        if (FAILED(hr)) ++present_errors;
        WaitMsPrecise(33);
    }
    gpu.Wait();
    Sleep(200);
    const FgAsiSessionStats s = host->Stats();
    std::printf("    intervals %llu: session %llu, copy path %llu; sessions created %llu, failures %llu, destroyed "
                "%llu%s\n",
                static_cast<unsigned long long>(s.intervals), static_cast<unsigned long long>(s.session_presents),
                static_cast<unsigned long long>(s.copy_presents), static_cast<unsigned long long>(s.sessions_created),
                static_cast<unsigned long long>(s.session_failures), static_cast<unsigned long long>(s.sessions_destroyed),
                s.gave_up ? ", gave up" : "");
    std::printf("    executor: shared %llu, indices %llu, copies %llu, real-only from the facade %llu, net2 fast %llu "
                "exact %llu; latency waits %d / %d; interface presents %d\n",
                static_cast<unsigned long long>(s.exec_shared), static_cast<unsigned long long>(s.exec_indices),
                static_cast<unsigned long long>(s.exec_copies),
                static_cast<unsigned long long>(s.exec_game_colour_copies),
                static_cast<unsigned long long>(s.exec_net2_fast), static_cast<unsigned long long>(s.exec_net2_exact),
                latency_waits, frames, iface.presents);
    CHECK(present_errors == 0 && index_moves);
    CHECK(s.intervals == static_cast<uint64_t>(frames));
    // the first present (nothing latched yet) takes the copy path, and the first after the resize may (the session is
    // created again from the core's description); a session presents every other interval, created twice (before and
    // after the resize), never refused
    CHECK(s.copy_presents >= 1 && s.copy_presents <= 2 &&
          s.session_presents == static_cast<uint64_t>(frames) - s.copy_presents);
    CHECK(s.sessions_created == 2 && s.session_failures == 0 && s.sessions_destroyed == 1 && !s.gave_up);
    // real-only from the facade buffer: every frame with generation off, plus the new epoch's first frame after the
    // resize when the session was already there to present it
    CHECK(s.exec_indices > 0 &&
          s.exec_game_colour_copies == static_cast<uint64_t>(off_to - off_from) + (2 - s.copy_presents));
    if (preset == 0) CHECK(s.exec_net2_fast > 0 && s.exec_net2_exact == 0);
    CHECK(latency_waits == frames);
    const int64_t dbg = gpu.DebugErrors();
    if (dbg >= 0) {
        std::printf("    debug layer errors during the run: %lld\n", static_cast<long long>(dbg));
        CHECK(dbg == 0);
    }
    // the game's last reference: the wrapper detaches the host (session destroyed, queue drained) and lets the chain go
    chain->Release();
    CHECK(gpu.Wait());
    const FgAsiSessionStats e = host->Stats();
    CHECK(e.sessions_destroyed == 2 && !e.facade);
    host.reset();
    CHECK(gpu.Wait());
    const int64_t dbg2 = gpu.DebugErrors();
    if (dbg2 >= 0) {
        std::printf("    debug layer errors after teardown: %lld\n", static_cast<long long>(dbg2));
        CHECK(dbg2 == 0);
    }
}

}  // namespace

// ---- the OptiScaler menu card through a stand-in host table ---------------------------------------------------------

namespace fakehost {
WgHostDrawFn g_draw = nullptr;
std::string g_title;
int g_cards = 0, g_tabs = 0, g_sections = 0;
std::string g_report_state;
std::string g_report_text;
float g_report_rate = 0, g_report_base = 0;
bool g_report_generating = false;
bool g_report_accept = true;
std::vector<std::string> g_seen;
int32_t g_force = -1;
std::string g_force_label;
bool RegisterCard(WgHostMenuPlugin*, const char* title, WgHostDrawFn draw, void*) {
    ++g_cards;
    g_title = title;
    g_draw = draw;
    return true;
}
bool RegisterTab(WgHostMenuPlugin*, const char* name, const char*, WgHostDrawFn draw, void*) {
    ++g_tabs;
    g_title = name;
    g_draw = draw;
    return true;
}
bool RegisterFrameGenSection(WgHostMenuPlugin*, WgHostDrawFn draw, void*) {
    ++g_sections;
    g_draw = draw;
    return true;
}
bool ReportFrameGeneration(WgHostMenuPlugin*, const char*, const char* state, const char* text, const char*,
                           float rate, float base, bool generating) {
    if (!g_report_accept) return false;
    g_report_state = state ? state : "";
    g_report_text = text ? text : "";
    g_report_rate = rate;
    g_report_base = base;
    g_report_generating = generating;
    return true;
}
bool Checkbox(const char* label, bool*) {
    g_seen.push_back(std::string("checkbox:") + label);
    return false;
}
void Section(const char* t) { g_seen.push_back(std::string("section:") + t); }
void Text(const char* t) { g_seen.push_back(t); }
void TextColored(float, float, float, float, const char* t) { g_seen.push_back(t); }
void Tooltip(const char*) {}
bool Combo(const char* label, int32_t* cur, const char* const* items, int32_t count) {
    g_seen.push_back(std::string("combo:") + label + "=" + (*cur >= 0 && *cur < count ? items[*cur] : "?"));
    if (g_force >= 0 && g_force_label == label) {
        *cur = g_force;
        g_force = -1;
        return true;
    }
    return false;
}
WgHostMenuApi Api(bool cards) {
    WgHostMenuApi a = {};
    a.size = sizeof(a);
    a.version = cards ? 3u : 1u;
    a.register_tab = &RegisterTab;
    a.section = &Section;
    a.item_tooltip = &Tooltip;
    a.text = &Text;
    a.text_disabled = &Text;
    a.text_colored = &TextColored;
    a.combo = &Combo;
    if (cards) a.register_custom_card = &RegisterCard;
    return a;
}
bool Seen(const std::string& what) {
    for (const auto& x : g_seen)
        if (x.find(what) != std::string::npos) return true;
    return false;
}
}  // namespace fakehost

panel::Status g_panel_status;
bool g_apply_ok = true;
uint32_t g_applied_preset = 99, g_applied_multiplier = 99, g_apply_calls = 0;
std::wstring g_panel_dir;
panel::Status PanelStatusHook() { return g_panel_status; }
bool PanelApplyHook(uint32_t preset, uint32_t multiplier) {
    ++g_apply_calls;
    g_applied_preset = preset;
    g_applied_multiplier = multiplier;
    return g_apply_ok;
}
std::wstring PanelDirHook() { return g_panel_dir; }
std::vector<std::string> g_panel_log;
void PanelLogHook(const char* line) { g_panel_log.push_back(line); }

void TestPanel(const std::string& scratch) {
    g_case = "panel";
    // the status line per state
    panel::Status st;
    FgSettings cur;
    CHECK(panel::StatusLine(st, cur) == "Off (fg=off)");
    st.enabled = true;
    st.install_result = kFgInterfaceNotFound;
    CHECK(panel::StatusLine(st, cur).rfind("Not active: game interface module not found", 0) == 0);
    st.installed = true;
    CHECK(panel::StatusLine(st, cur).rfind("Not active: no frame generation runtime", 0) == 0);
    st.have_host = true;
    CHECK(panel::StatusLine(st, cur) == "Waiting for the game");
    st.live.phase = kFgPhaseGenerating;
    st.live.count = 1;
    st.live.preset = 0;
    st.live.presenter = static_cast<uint32_t>(wgfg::FgPresenterChoice::DxgiFlipTearing);
    CHECK(panel::StatusLine(st, cur) == "Generating: 2x, preset A, DXGI flip, tearing");
    cur.multiplier = 4;
    CHECK(panel::StatusLine(st, cur) == "Generating: 2x, preset A, DXGI flip, tearing");
    st.live.count = 3;
    st.live.preset = 2;
    st.live.presenter = static_cast<uint32_t>(wgfg::FgPresenterChoice::WindowsCompositionHwfq);
    CHECK(panel::StatusLine(st, cur) == "Generating: 4x, preset C, Windows composition, hardware flip queue");
    st.live.phase = kFgPhaseRealOnly;
    st.live.reason = WG_FG_REASON_MISSING_MOTION;
    CHECK(panel::StatusLine(st, cur) == "Real frames only: no motion vectors");
    st.live.reason = 0;
    st.live.real_only = WG_FG_REAL_ONLY_DISABLED;
    CHECK(panel::StatusLine(st, cur) == "Real frames only: generation off");
    st.live.phase = kFgPhaseStopped;
    CHECK(panel::StatusLine(st, cur).rfind("Stopped for this run", 0) == 0);

    // a game folder with its state folder, and the hooks
    const std::string game = scratch + "\\panel-game";
    CreateDirectoryA(game.c_str(), nullptr);
    CreateDirectoryA((game + "\\wronggpu-setup").c_str(), nullptr);
    DeleteFileA((game + "\\wronggpu-setup\\settings.ini").c_str());
    g_panel_dir.assign(game.begin(), game.end());
    panel::Hooks hooks;
    hooks.status = &PanelStatusHook;
    hooks.apply = &PanelApplyHook;
    hooks.game_dir = &PanelDirHook;
    hooks.log = &PanelLogHook;
    FgSettings start;
    start.enabled = true;
    start.preset = 0;
    start.multiplier = 2;

    // an incomplete table: no card
    WgHostMenuApi broken = fakehost::Api(true);
    broken.combo = nullptr;
    CHECK(!panel::Init(&broken, nullptr, hooks, start));
    CHECK(fakehost::g_cards == 0 && fakehost::g_tabs == 0);
    CHECK(!panel::Init(nullptr, nullptr, hooks, start));

    // a host with cards: the card on the Custom tab
    static WgHostMenuApi api = fakehost::Api(true);
    CHECK(panel::Init(&api, nullptr, hooks, start));
    CHECK(fakehost::g_cards == 1 && fakehost::g_title == "WrongGPU Frame Generation" && fakehost::g_draw);
    if (!fakehost::g_draw) return;
    // fg=off: the status and a hint, no combos
    g_panel_status = panel::Status();
    fakehost::g_seen.clear();
    fakehost::g_draw(nullptr);
    CHECK(fakehost::Seen("Off (fg=off)") && !fakehost::Seen("combo:"));
    // generating: the status line and both combos with the start values
    g_panel_status = st;
    g_panel_status.live.phase = kFgPhaseGenerating;
    g_panel_status.live.count = 1;
    g_panel_status.live.preset = 0;
    g_panel_status.live.presenter = static_cast<uint32_t>(wgfg::FgPresenterChoice::DxgiFlipTearing);
    fakehost::g_seen.clear();
    fakehost::g_draw(nullptr);
    CHECK(fakehost::Seen("Generating: 2x, preset A, DXGI flip, tearing"));
    CHECK(fakehost::Seen("combo:Preset##wg_fg_preset=A") && fakehost::Seen("combo:Multiplier##wg_fg_multiplier=2x"));
    CHECK(!fakehost::Seen("section:"));  // a card is already titled
    // the multiplier combo: live hook, then settings.ini
    fakehost::g_force = 3;
    fakehost::g_force_label = "Multiplier##wg_fg_multiplier";
    fakehost::g_draw(nullptr);
    CHECK(g_apply_calls == 1 && g_applied_preset == 0 && g_applied_multiplier == 4);
    CHECK(panel::Current().multiplier == 4);
    wgstate::Settings saved = wgstate::ReadSettings(g_panel_dir);
    const std::string* m = wgstate::Get(saved, "fg_multiplier");
    CHECK(m && *m == "4");
    CHECK(!g_panel_log.empty() && g_panel_log.back().find("fg_multiplier=4 chosen in OptiScaler's menu (from the next "
                                                          "frame, saved to wronggpu-setup\\settings.ini)") !=
                                     std::string::npos);
    // the preset combo
    fakehost::g_force = 2;
    fakehost::g_force_label = "Preset##wg_fg_preset";
    fakehost::g_seen.clear();
    fakehost::g_draw(nullptr);
    CHECK(g_apply_calls == 2 && g_applied_preset == 2 && g_applied_multiplier == 4);
    saved = wgstate::ReadSettings(g_panel_dir);
    const std::string* pr = wgstate::Get(saved, "fg_preset");
    m = wgstate::Get(saved, "fg_multiplier");
    CHECK(pr && *pr == "C" && m && *m == "4");
    fakehost::g_seen.clear();
    fakehost::g_draw(nullptr);
    CHECK(fakehost::Seen("combo:Preset##wg_fg_preset=C") && fakehost::Seen("combo:Multiplier##wg_fg_multiplier=4x"));
    // no live core: saved for the next start
    g_apply_ok = false;
    CHECK(panel::ChoosePreset(1).find("(from the next game start, saved to") != std::string::npos);
    g_apply_ok = true;
    // values outside the lists change nothing
    const uint32_t calls = g_apply_calls;
    CHECK(panel::ChoosePreset(3).empty() && panel::ChooseMultiplier(1).empty() && panel::ChooseMultiplier(7).empty());
    CHECK(g_apply_calls == calls);
    // a host without cards: a tab of its own, titled inside
    static WgHostMenuApi tab_api = fakehost::Api(false);
    CHECK(panel::Init(&tab_api, nullptr, hooks, start));
    CHECK(fakehost::g_tabs == 1 && fakehost::g_title == "WrongGPU Frame Generation");
    fakehost::g_seen.clear();
    fakehost::g_draw(nullptr);
    CHECK(fakehost::Seen("section:WrongGPU Frame Generation"));

    fakehost::g_force = 0;
    fakehost::g_force_label = "Multiplier##wg_fg_multiplier";
    fakehost::g_draw(nullptr);
    CHECK(panel::Current().multiplier == kFgAutoMultiplier);
    CHECK(g_applied_multiplier == kFgAutoMultiplier);
    fakehost::g_seen.clear();
    fakehost::g_draw(nullptr);
    CHECK(fakehost::Seen("combo:Multiplier##wg_fg_multiplier=Auto"));

    static WgHostMenuApi v6 = fakehost::Api(true);
    v6.version = 6;
    v6.checkbox = &fakehost::Checkbox;
    v6.report_frame_generation = &fakehost::ReportFrameGeneration;
    v6.register_frame_gen_section = &fakehost::RegisterFrameGenSection;
    CHECK(panel::Init(&v6, nullptr, hooks, start));
    CHECK(fakehost::g_sections == 1 && fakehost::g_cards == 1 && fakehost::g_tabs == 1);
    CHECK(panel::ClaimOutput(&v6, nullptr));
    CHECK(fakehost::g_report_state == "Waiting");
    g_panel_status.live.game_has_mfg_selector = true;
    g_panel_status.live.game_requested_count = 2;
    g_panel_status.live.multiplier_source = kFgMultSourceGame;
    g_panel_status.live.requested_count = 2;
    g_panel_status.live.count = 2;
    g_panel_status.live.scheduled_count = 2;
    g_panel_status.live.displayed_fps = 120.0f;
    g_panel_status.live.base_fps = 41.0f;
    g_panel_status.live.displayed_generated = 10;
    g_panel_status.live.presenter = static_cast<uint32_t>(wgfg::FgPresenterChoice::WindowsCompositionHwfq);
    fakehost::g_seen.clear();
    fakehost::g_draw(nullptr);
    CHECK(fakehost::Seen("Generating 3x (game), preset A, HWFQ"));
    CHECK(fakehost::Seen("Multiplier: 3x") && fakehost::Seen("set in the game's graphics menu"));
    CHECK(!fakehost::Seen("combo:Multiplier") && fakehost::Seen("combo:Presenter"));
    CHECK(panel::PublishAt(g_panel_status, panel::NextReportSequence(), true, 1000));
    CHECK(fakehost::g_report_state == "3x" && fakehost::g_report_rate == 120.0f);
    CHECK(fakehost::g_report_base == 41.0f && fakehost::g_report_generating);

    g_panel_status.live.count_cap_reason = kFgCapDisplayVrr;
    g_panel_status.live.scheduled_count = 1;
    fakehost::g_seen.clear();
    fakehost::g_draw(nullptr);
    CHECK(fakehost::Seen("Scheduled: 2x (display cap)"));
    g_panel_status.live.count_cap_reason = kFgCapLatencyCap;
    fakehost::g_seen.clear();
    fakehost::g_draw(nullptr);
    CHECK(fakehost::Seen("Scheduled: 2x (latency)"));
    g_panel_status.live.count_cap_reason = kFgCapRoute;
    fakehost::g_seen.clear();
    fakehost::g_draw(nullptr);
    CHECK(fakehost::Seen("Scheduled: 2x (route limit)"));
    g_panel_status.live.scheduled_count = 0;
    g_panel_status.live.schedule_reason = static_cast<uint32_t>(wgfg::FgRealOnlyReason::LatencyCap);
    fakehost::g_seen.clear();
    fakehost::g_draw(nullptr);
    CHECK(fakehost::Seen("Real frames only:"));
    // N6: an isolated real-only frame within the 500 ms activity window keeps generating steady (no 10 Hz flapping)
    // and retains the last generating multiplier instead of reporting "1x".
    CHECK(panel::PublishAt(g_panel_status, panel::NextReportSequence(), false, 1100));
    CHECK(fakehost::g_report_state == "3x" && fakehost::g_report_generating);
    // Once 500 ms elapse without new presented generated images, report drops cleanly to Real only.
    CHECK(panel::PublishAt(g_panel_status, panel::NextReportSequence(), false, 1500));
    CHECK(fakehost::g_report_state == "Real only" && fakehost::g_report_rate == 0.0f);
    CHECK(fakehost::g_report_base == 0.0f && !fakehost::g_report_generating);

    g_panel_status.live.game_has_mfg_selector = false;
    g_panel_status.live.game_requested_count = 1;
    fakehost::g_seen.clear();
    fakehost::g_draw(nullptr);
    CHECK(fakehost::Seen("combo:Multiplier") && !fakehost::Seen("graphics menu"));
    g_panel_status.live.game_requested_count = 0;
    fakehost::g_seen.clear();
    fakehost::g_draw(nullptr);
    CHECK(fakehost::Seen("combo:Multiplier"));
    g_panel_status.live.scheduled_count = 2;
    const auto stale_sequence = panel::NextReportSequence();
    const auto stale_generating = g_panel_status;
    g_panel_status.enabled = false;
    CHECK(panel::PublishAt(g_panel_status, panel::NextReportSequence(), true, 1600));
    CHECK(fakehost::g_report_state == "Off" && fakehost::g_report_rate == 0.0f && !fakehost::g_report_generating);
    CHECK(panel::PublishAt(stale_generating, stale_sequence, true, 1610));
    CHECK(fakehost::g_report_state == "Off" && !fakehost::g_report_generating);

    // R3A-1: scheduled_count > 0 with zero new presented generated images (all dropped late) must report Real only,
    // even when re-enabled with a nonzero historical displayed_generated count.
    {
        panel::Status dropped = st;
        dropped.enabled = true;
        dropped.installed = true;
        dropped.have_host = true;
        dropped.live.phase = kFgPhaseGenerating;
        dropped.live.scheduled_count = 2;
        dropped.live.count = 2;
        dropped.live.displayed_fps = 60.0f;
        dropped.live.base_fps = 60.0f;
        dropped.live.displayed_generated = 10;  // unchanged since before Off!
        CHECK(panel::PublishAt(dropped, panel::NextReportSequence(), true, 2000));
        CHECK(fakehost::g_report_state == "Real only" && !fakehost::g_report_generating);
        CHECK(fakehost::g_report_rate == 0.0f && fakehost::g_report_base == 0.0f);
        // Repeated publishes with unchanged displayed_generated still stay Real only.
        CHECK(panel::PublishAt(dropped, panel::NextReportSequence(), false, 2250));
        CHECK(fakehost::g_report_state == "Real only" && !fakehost::g_report_generating);

        // Once generated images are actually presented (delta > 0), R3A-10 bypasses the 100 ms throttle immediately.
        dropped.live.displayed_generated = 15;
        dropped.live.displayed_fps = 120.0f;
        dropped.live.base_fps = 40.0f;
        CHECK(panel::PublishAt(dropped, panel::NextReportSequence(), false, 2260));
        CHECK(fakehost::g_report_state == "3x" && fakehost::g_report_generating);
        CHECK(fakehost::g_report_rate == 120.0f && fakehost::g_report_base == 40.0f);

        // Steady-state update within 100 ms with unchanged active/state is throttled (does not overwrite rate yet).
        dropped.live.displayed_fps = 123.0f;
        CHECK(panel::PublishAt(dropped, panel::NextReportSequence(), false, 2300));
        CHECK(fakehost::g_report_rate == 120.0f);

        // If all generated images now drop late for >= 500 ms (displayed_generated stuck at 15), report falls back
        // to Real only even while scheduled_count stays 2!
        CHECK(panel::PublishAt(dropped, panel::NextReportSequence(), false, 2760));
        CHECK(fakehost::g_report_state == "Real only" && !fakehost::g_report_generating);
        CHECK(fakehost::g_report_rate == 0.0f && fakehost::g_report_base == 0.0f);
    }

    // R3A-10: Generating -> Off / Waiting transitions bypass the 100 ms throttle immediately (even with immediate=false).
    {
        panel::Status gen_st = st;
        gen_st.enabled = true;
        gen_st.installed = true;
        gen_st.have_host = true;
        gen_st.live.phase = kFgPhaseGenerating;
        gen_st.live.scheduled_count = 1;
        gen_st.live.count = 1;
        gen_st.live.displayed_generated = 20;
        gen_st.live.displayed_fps = 120.0f;
        gen_st.live.base_fps = 60.0f;
        CHECK(panel::PublishAt(gen_st, panel::NextReportSequence(), true, 3000));
        CHECK(fakehost::g_report_state == "2x" && fakehost::g_report_generating);
        panel::Status wait_st = gen_st;
        wait_st.live.phase = kFgPhaseRealOnly;
        wait_st.live.reason = 0;
        wait_st.live.real_only = WG_FG_REAL_ONLY_DISABLED;
        CHECK(panel::PublishAt(wait_st, panel::NextReportSequence(), false, 3010));
        CHECK(fakehost::g_report_state == "Waiting" && !fakehost::g_report_generating);
        panel::Status off_st = wait_st;
        off_st.enabled = false;
        CHECK(panel::PublishAt(off_st, panel::NextReportSequence(), false, 3020));
        CHECK(fakehost::g_report_state == "Off" && !fakehost::g_report_generating);
    }

    // v5 card StatusLine must not say "Generating" on scheduler real-only or priming intervals.
    {
        panel::Status sched_real = st;
        sched_real.enabled = true;
        sched_real.installed = true;
        sched_real.have_host = true;
        sched_real.live.phase = kFgPhaseGenerating;
        sched_real.live.count = 2;
        sched_real.live.scheduled_count = 0;
        sched_real.live.schedule_reason = static_cast<uint32_t>(wgfg::FgRealOnlyReason::LatencyCap);
        CHECK(panel::StatusLine(sched_real, cur).rfind("Real frames only:", 0) == 0);
        sched_real.live.schedule_reason = static_cast<uint32_t>(wgfg::FgRealOnlyReason::NotPrimed);
        CHECK(panel::StatusLine(sched_real, cur) == "Waiting for the game");
    }

    // R3A-2: PublishAt returns false when the host rejects/revokes the claim (e.g. swap-chain creation began mid-install).
    fakehost::g_report_accept = false;
    CHECK(!panel::PublishAt(g_panel_status, panel::NextReportSequence(), true, 3100));
    fakehost::g_report_accept = true;

    // R3B-1: an isolated scheduler-driven real-only interval (e.g. OutOfVrrWindow display cap) between generating
    // intervals keeps the 500 ms activity hold without publishing inactive, shows the CURRENT scheduler reduction
    // reason (display cap / latency / presenter pressure) instead of a stale adapter reason, and a sustained
    // display-capped run transitions cleanly to Real only after the hold. A lost session ends the hold immediately.
    {
        panel::Status r3b = st;
        r3b.enabled = true;
        r3b.installed = true;
        r3b.have_host = true;
        r3b.live.phase = kFgPhaseGenerating;
        r3b.live.count = 2;
        r3b.live.scheduled_count = 2;
        r3b.live.schedule_reason = 0;
        r3b.live.displayed_generated = 25;
        r3b.live.displayed_fps = 180.0f;
        r3b.live.base_fps = 60.0f;
        CHECK(panel::PublishAt(r3b, panel::NextReportSequence(), true, 4000));
        CHECK(fakehost::g_report_generating && fakehost::g_report_state == "3x");

        // Isolated single-interval display cap (OutOfVrrWindow): hold kept, no inactive publish, UI names display cap.
        panel::Status capped = r3b;
        capped.live.scheduled_count = 0;
        capped.live.schedule_reason = static_cast<uint32_t>(wgfg::FgRealOnlyReason::OutOfVrrWindow);
        g_panel_status = capped;
        fakehost::g_seen.clear();
        fakehost::g_draw(nullptr);
        CHECK(fakehost::Seen("Real frames only: display cap"));
        CHECK(panel::StatusLine(capped, cur) == "Real frames only: display cap");
        CHECK(panel::PublishAt(capped, panel::NextReportSequence(), false, 4120));
        CHECK(fakehost::g_report_generating && fakehost::g_report_state == "3x");

        // Even if a snapshot arrives with kFgPhaseRealOnly + stale adapter WG_FG_REAL_ONLY_DISABLED alongside an
        // active scheduler reduction, the current reduction reason wins and the hold is kept.
        panel::Status stale_adapter = capped;
        stale_adapter.live.phase = kFgPhaseRealOnly;
        stale_adapter.live.reason = 0;
        stale_adapter.live.real_only = WG_FG_REAL_ONLY_DISABLED;
        g_panel_status = stale_adapter;
        fakehost::g_seen.clear();
        fakehost::g_draw(nullptr);
        CHECK(fakehost::Seen("Real frames only: display cap") && !fakehost::Seen("Waiting"));
        CHECK(panel::StatusLine(stale_adapter, cur) == "Real frames only: display cap");
        CHECK(panel::PublishAt(stale_adapter, panel::NextReportSequence(), false, 4200));
        CHECK(fakehost::g_report_generating && fakehost::g_report_state == "3x");

        // LatencyCap and Pressure spellings.
        capped.live.schedule_reason = static_cast<uint32_t>(wgfg::FgRealOnlyReason::LatencyCap);
        CHECK(panel::StatusLine(capped, cur) == "Real frames only: latency");
        capped.live.schedule_reason = static_cast<uint32_t>(wgfg::FgRealOnlyReason::Pressure);
        CHECK(panel::StatusLine(capped, cur) == "Real frames only: presenter pressure");

        // Next generating interval refreshes the hold seamlessly with zero inactive publish in between.
        r3b.live.displayed_generated = 27;
        CHECK(panel::PublishAt(r3b, panel::NextReportSequence(), false, 4250));
        CHECK(fakehost::g_report_generating && fakehost::g_report_state == "3x");

        // Sustained display-capped run: held inside 500 ms, then transitions cleanly to Real only at 500 ms.
        capped.live.schedule_reason = static_cast<uint32_t>(wgfg::FgRealOnlyReason::OutOfVrrWindow);
        capped.live.displayed_generated = 27;
        CHECK(panel::PublishAt(capped, panel::NextReportSequence(), false, 4600));
        CHECK(fakehost::g_report_generating && fakehost::g_report_state == "3x");
        CHECK(panel::PublishAt(capped, panel::NextReportSequence(), false, 4750));
        CHECK(!fakehost::g_report_generating && fakehost::g_report_state == "Real only");
        CHECK(fakehost::g_report_text == "Real frames only: display cap");
        CHECK(fakehost::g_report_rate == 0.0f && fakehost::g_report_base == 0.0f);

        // Lost session (phase Waiting, schedule_reason 0) ends the hold immediately even within 10 ms.
        r3b.live.displayed_generated = 30;
        CHECK(panel::PublishAt(r3b, panel::NextReportSequence(), true, 5000));
        CHECK(fakehost::g_report_generating && fakehost::g_report_state == "3x");
        panel::Status lost = r3b;
        lost.live.phase = kFgPhaseWaiting;
        lost.live.scheduled_count = 0;
        lost.live.schedule_reason = 0;
        CHECK(panel::PublishAt(lost, panel::NextReportSequence(), false, 5010));
        CHECK(!fakehost::g_report_generating && fakehost::g_report_state == "Waiting");
    }

    panel::ReleaseOutput();
}

void TestFacadeFormats(Gpu& gpu, const std::string& dxil) {
    std::printf("  facade formats (R10G10B10A2_UNORM + R8G8B8A8_UNORM)\n");
    ComPtr<IDXGIFactory2> f2;
    CHECK(SUCCEEDED(gpu.factory.As(&f2)));
    if (!f2) return;
    D3D12_DESCRIPTOR_HEAP_DESC hd = {};
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    hd.NumDescriptors = 4;
    ComPtr<ID3D12DescriptorHeap> rtv_heap;
    CHECK(SUCCEEDED(gpu.device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&rtv_heap))));
    if (!rtv_heap) return;

    const DXGI_FORMAT fmts[2] = {DXGI_FORMAT_R10G10B10A2_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM};
    for (int fi = 0; fi < 2; ++fi) {
        const DXGI_FORMAT fmt = fmts[fi];
        FgAsiSessionConfig hc;
        hc.settings = CoreConfig(0).settings;
        hc.dxil_dir = dxil;
        hc.log = &PrintLog;
        std::unique_ptr<wgfg::FgExecWeights> w(new wgfg::FgExecWeights());
        wgfg::FgExecSyntheticWeights(11, w.get());
        std::unique_ptr<FgAsiSessionHost> host(new FgAsiSessionHost(hc, std::move(w)));
        DXGI_SWAP_CHAIN_DESC1 sd = {};
        sd.Width = kW;
        sd.Height = kH;
        sd.Format = fmt;
        sd.SampleDesc.Count = 1;
        sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        sd.BufferCount = 3;
        sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        sd.AlphaMode = DXGI_ALPHA_MODE_IGNORE;
        IDXGISwapChain1* sc1 = nullptr;
        CHECK(SUCCEEDED(f2->CreateSwapChainForComposition(gpu.queue.Get(), &sd, nullptr, &sc1)) && sc1);
        if (!sc1) continue;
        ComPtr<IDXGISwapChain4> sc4;
        CHECK(SUCCEEDED(sc1->QueryInterface(IID_PPV_ARGS(&sc4))));
        sc1->Release();
        if (!sc4) continue;
        host->Attached(sc4.Get(), gpu.queue.Get());
        CHECK(host->Stats().facade);
        ComPtr<ID3D12Resource> fb;
        HRESULT ghr = E_FAIL;
        CHECK(host->GetBuffer(0, IID_PPV_ARGS(&fb), &ghr) && SUCCEEDED(ghr) && fb);
        if (fb) {
            const D3D12_RESOURCE_DESC d = fb->GetDesc();
            if (fmt == DXGI_FORMAT_R10G10B10A2_UNORM) {
                // R10G10B10A2_UNORM must stay typed (format 24), never typeless: default RTV (nullptr desc) must work.
                CHECK(d.Format == DXGI_FORMAT_R10G10B10A2_UNORM);
                gpu.device->CreateRenderTargetView(fb.Get(), nullptr, rtv_heap->GetCPUDescriptorHandleForHeapStart());
            } else {
                // R8G8B8A8_UNORM supports both UNORM and sRGB RTVs; typed when ID3D12Device10 castable formats exist.
                D3D12_RENDER_TARGET_VIEW_DESC rd = {};
                rd.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
                rd.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
                gpu.device->CreateRenderTargetView(fb.Get(), &rd, rtv_heap->GetCPUDescriptorHandleForHeapStart());
                if (d.Format == DXGI_FORMAT_R8G8B8A8_UNORM)
                    gpu.device->CreateRenderTargetView(fb.Get(), nullptr, rtv_heap->GetCPUDescriptorHandleForHeapStart());
            }
            CHECK(gpu.device->GetDeviceRemovedReason() == S_OK);
        }
        host->Detached();
        if (g_debug_layer) CHECK(gpu.DebugErrors() == 0);
    }
}

void TestFacadeCopyRetryAndReuse(Gpu& gpu) {
    g_case = "copy retry and slow facade reuse";
    ComPtr<IDXGIFactory2> factory;
    D3D12_COMMAND_QUEUE_DESC queue_desc = {};
    ComPtr<ID3D12CommandQueue> chain_queue;
    ComPtr<ID3D12Fence> gate, done;
    CHECK(SUCCEEDED(gpu.factory.As(&factory)));
    CHECK(SUCCEEDED(gpu.device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&chain_queue))));
    CHECK(SUCCEEDED(gpu.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&gate))));
    CHECK(SUCCEEDED(gpu.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&done))));
    if (!factory || !chain_queue || !gate || !done) return;
    DXGI_SWAP_CHAIN_DESC1 desc = {};
    desc.Width = kW;
    desc.Height = kH;
    desc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    desc.SampleDesc.Count = 1;
    desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    desc.BufferCount = 2;
    desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    desc.AlphaMode = DXGI_ALPHA_MODE_IGNORE;
    ComPtr<IDXGISwapChain1> base;
    ComPtr<IDXGISwapChain4> native;
    CHECK(SUCCEEDED(factory->CreateSwapChainForComposition(chain_queue.Get(), &desc, nullptr, &base)));
    if (!base || FAILED(base.As(&native))) return;

    FgAsiSessionConfig config;
    config.log = &PrintLog;
    FgAsiSessionHost host(config, nullptr);
    host.AttachedPresentQueue(chain_queue.Get());
    host.Attached(native.Get(), gpu.queue.Get());
    CHECK(host.Stats().facade);
    if (!host.Stats().facade) return;
    struct RetryInner final : IFgInnerPresenter {
        IDXGISwapChain4* chain = nullptr;
        bool retry = true;
        HRESULT Present(UINT, UINT) override {
            if (!retry) return S_OK;
            retry = false;
            return DXGI_ERROR_WAS_STILL_DRAWING;
        }
        HRESULT Present1(UINT sync, UINT flags, const DXGI_PRESENT_PARAMETERS*) override {
            return Present(sync, flags);
        }
        IDXGISwapChain4* Inner() override { return chain; }
    } inner;
    inner.chain = native.Get();
    CHECK(SUCCEEDED(chain_queue->Wait(gate.Get(), 1)));
    FgPresentCall call;
    UINT index = ~0u;
    CHECK(host.PresentInterval(call, inner) == DXGI_ERROR_WAS_STILL_DRAWING);
    CHECK(host.CurrentBackBufferIndex(&index) && index == 0);
    call.retry = true;
    CHECK(host.PresentInterval(call, inner) == S_OK);
    CHECK(host.CurrentBackBufferIndex(&index) && index == 1);
    CHECK(host.Stats().copy_presents == 1);
    call.retry = false;
    CHECK(host.PresentInterval(call, inner) == S_OK);
    CHECK(host.CurrentBackBufferIndex(&index) && index == 0);
    CHECK(SUCCEEDED(gpu.queue->Signal(done.Get(), 1)));
    Sleep(120);
    CHECK(done->GetCompletedValue() == 0);  // advancing enqueued the wait without timing out on the CPU
    CHECK(SUCCEEDED(gate->Signal(1)));
    CHECK(gpu.Wait());
    CHECK(done->GetCompletedValue() == 1);
    CHECK(host.BeforeResize() == S_OK);
    host.AfterResize(DXGI_ERROR_INVALID_CALL);
    CHECK(host.CurrentBackBufferIndex(&index) && index == 0);
    host.Detached();

    // R3B-4: when BeforeResize returns DXGI_ERROR_WAS_STILL_DRAWING on a busy waitable L=1 chain, it must replenish
    // the game's frame-start waitable credit so an L=1 caller never deadlocks.
    {
        DXGI_SWAP_CHAIN_DESC1 wdesc = desc;
        wdesc.Flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;
        ComPtr<IDXGISwapChain1> wbase;
        ComPtr<IDXGISwapChain4> wnative;
        CHECK(SUCCEEDED(factory->CreateSwapChainForComposition(chain_queue.Get(), &wdesc, nullptr, &wbase)));
        if (wbase && SUCCEEDED(wbase.As(&wnative))) {
            FgAsiSessionHost whost(config, nullptr);
            whost.AttachedPresentQueue(chain_queue.Get());
            whost.SetGameWaitable(true);
            whost.Attached(wnative.Get(), gpu.queue.Get());
            HRESULT lhr = E_FAIL;
            CHECK(whost.SetMaximumFrameLatency(1, &lhr) && SUCCEEDED(lhr));
            HANDLE wh = nullptr;
            CHECK(whost.FrameLatencyWaitableObject(&wh) && wh != nullptr);
            if (wh) {
                CHECK(WaitForSingleObject(wh, 0) == WAIT_OBJECT_0);
                CHECK(WaitForSingleObject(wh, 0) == WAIT_TIMEOUT);  // L=1 credit consumed
                ComPtr<ID3D12Fence> gate2;
                CHECK(SUCCEEDED(gpu.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&gate2))));
                if (gate2) {
                    CHECK(SUCCEEDED(chain_queue->Wait(gate2.Get(), 1)));
                    RetryInner winner;
                    winner.chain = wnative.Get();
                    winner.retry = false;
                    FgPresentCall wcall;
                    CHECK(whost.PresentInterval(wcall, winner) == S_OK);
                    CHECK(WaitForSingleObject(wh, 0) == WAIT_OBJECT_0);  // credit returned on present
                    CHECK(WaitForSingleObject(wh, 0) == WAIT_TIMEOUT);   // consume again before busy resize
                    CHECK(whost.BeforeResize() == DXGI_ERROR_WAS_STILL_DRAWING);
                    CHECK(WaitForSingleObject(wh, 0) == WAIT_OBJECT_0);  // R3B-4: replenished on busy resize
                    CHECK(SUCCEEDED(gate2->Signal(1)));
                    CHECK(gpu.Wait());
                    CHECK(whost.BeforeResize() == S_OK);
                }
                CloseHandle(wh);
            }
            whost.Detached();
        }
    }
    if (g_debug_layer) CHECK(gpu.DebugErrors() == 0);
}

// RC2 (A1 / A2 / O-A1 / O-A3 / N-B3 / N-S3 / M1): tests SetMaximumFrameLatency(1/2/3) deadlock-free with a cached
// waitable handle surviving ResizeBuffers, inner chain latency kept at 2, no-op SetFullscreenState(FALSE) ignored,
// 0 back-pressure waits when in-game FG is toggled OFF, and synthetic 2x (120 fps) & 4x (80 fps) GPU pacing legs.
void TestRc2LatencyAndPacing(Gpu& gpu, const std::string& dxil, bool run_latency, bool run_cadence) {
    std::printf("  rc2 latency (1/2/3), no-op fullscreen, fg-off back-pressure, and synthetic 2x/4x pacing\n");
    GameTextures game;
    CHECK(game.Init(gpu));
    ComPtr<IDXGIFactory2> f2;
    CHECK(SUCCEEDED(gpu.factory.As(&f2)));
    if (!f2) return;

    // Admission is also enforced when the title chooses not to wait on its handle.
    if (run_latency) {
        for (const bool wait_on_game : {true, false}) {
            for (UINT lat : {1u, 2u, 3u}) {
                FgPresentOwners owners;
                FakeInterface iface;
                FgProxyConfig pcfg = CoreConfig(2);  // Preset C
                pcfg.binding.has_mfg_selector = true;
                pcfg.settings.multiplier = 2;  // M1: Cyberpunk-like binding ignores user cap
                FgProxy core(pcfg, &owners, &PrintLog, nullptr);
                const WgFgChainV1 ct = iface.table();
                CHECK(core.attach_interposer(&ct, 2, 14) == WG_FG_REASON_NONE);
                core.set_feature_function("fgTestSetOptions", &g_fn_a);
                core.set_feature_function("fgTestGetState", &g_fn_b);

                FgAsiSessionConfig hc;
                hc.settings = pcfg.settings;
                hc.dxil_dir = dxil;
                hc.log = &PrintLog;
                std::unique_ptr<wgfg::FgExecWeights> w(new wgfg::FgExecWeights());
                wgfg::FgExecSyntheticWeights(11, w.get());
                std::unique_ptr<FgAsiSessionHost> host(new FgAsiSessionHost(hc, std::move(w)));
                host->OnInstalled(&core);
                FgAsiSessionHost::FactoryNote(host.get(), &core, gpu.factory.Get());
                CHECK(core.fg_taken());

                DXGI_SWAP_CHAIN_DESC1 sd = {};
                sd.Width = kW;
                sd.Height = kH;
                sd.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
                sd.SampleDesc.Count = 1;
                sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
                sd.BufferCount = 3;
                sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
                sd.AlphaMode = DXGI_ALPHA_MODE_IGNORE;
                sd.Flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;
                const UINT extra = host->ExtraSwapChainFlags(sd);
                sd.Flags |= extra;
                IDXGISwapChain1* sc1 = nullptr;
                CHECK(SUCCEEDED(f2->CreateSwapChainForComposition(gpu.queue.Get(), &sd, nullptr, &sc1)) && sc1);
                if (!sc1) return;
                FgSwapChain* chain = nullptr;
                CHECK(SUCCEEDED(FgSwapChain::Create(sc1, gpu.queue.Get(), &PrintLog, nullptr, &chain)) && chain);
                if (!chain) return;
                chain->SetExtraFlags(extra & ~DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT);
                host->SetGameWaitable(true);
                CHECK(chain->Attach(host.get(), &core, &owners, WG_FG_OWNER_SWAPCHAIN_WRAPPER) == WG_FG_REASON_NONE);
                host->Attached(chain->Inner(), chain->Queue());

                // A2: invalid latency values rejected; valid latency sets facade limit while inner stays at 2 (O-A3)
                CHECK(chain->SetMaximumFrameLatency(0) == DXGI_ERROR_INVALID_CALL);
                CHECK(chain->SetMaximumFrameLatency(17) == DXGI_ERROR_INVALID_CALL);
                CHECK(chain->SetMaximumFrameLatency(lat) == S_OK);
                UINT got_lat = 0, inner_lat = 0;
                CHECK(chain->GetMaximumFrameLatency(&got_lat) == S_OK && got_lat == lat);
                CHECK(chain->Inner()->GetMaximumFrameLatency(&inner_lat) == S_OK && inner_lat == 2u);

                // Cached waitable handle stays valid across ResizeBuffers
                HANDLE cached_waitable = chain->GetFrameLatencyWaitableObject();
                CHECK(cached_waitable != nullptr);

                WgFgOptions on = Options(true, 2);
                on.generated_count = 3;  // 4x request (honoured despite settings.multiplier=2 because has_mfg_selector=true)
                WgFgOptions off = Options(false, 2);
                core.set_options(WG_FEAT_FRAME_GENERATION, 1, &on, &on);

                int waits_ok = 0, present_errors = 0;
                uint64_t bp_before_off = 0, bp_after_off = 0;
                const int total_frames = 22;
                const int stall_frame = 5;  // after the scheduler has primed its generated path
                const int stall_end = stall_frame + 3;
                ComPtr<ID3D12Fence> stalled_game;
                std::thread unblock;
                bool have_next_credit = false;
                CHECK(SUCCEEDED(gpu.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&stalled_game))));
                for (int f = 0; f < total_frames; ++f) {
                    const uint64_t tok = static_cast<uint64_t>(f) + 1;
                    if (!wait_on_game && f == stall_frame)
                        core.set_options(WG_FEAT_FRAME_GENERATION, 1, &off, &off);
                    if (!wait_on_game && f == stall_end + 1)
                        core.set_options(WG_FEAT_FRAME_GENERATION, 1, &on, &on);
                    if (f == 6) {
                        // N-S3: no-op SetFullscreenState(FALSE) on already-windowed chain must not tear down the session
                        BOOL cur_fs = FALSE;
                        IDXGIOutput* cur_out = nullptr;
                        const bool can_query_fs = SUCCEEDED(chain->Inner()->GetFullscreenState(&cur_fs, &cur_out));
                        if (cur_out) cur_out->Release();
                        if (can_query_fs && !cur_fs) {
                            const uint64_t dest_before = host->Stats().sessions_destroyed;
                            chain->SetFullscreenState(FALSE, nullptr);
                            chain->SetFullscreenState(FALSE, nullptr);
                            CHECK(host->Stats().sessions_destroyed == dest_before);
                            CHECK(host->Stats().noop_fullscreen_ignored >= 2);
                        }
                    }
                    if (f == 10) {
                        const std::uint64_t epoch = chain->Epoch();
                        CHECK(FAILED(chain->ResizeBuffers(1, kW, kH, DXGI_FORMAT_UNKNOWN,
                                                          DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT)));
                        CHECK(chain->Epoch() == epoch && host->Stats().facade);
                        CHECK(SUCCEEDED(chain->ResizeBuffers(0, kW, kH, DXGI_FORMAT_UNKNOWN,
                                                             DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT)));
                    }
                    if (f == 14) {
                        bp_before_off = host->Stats().back_pressure_waits;
                        core.set_options(WG_FEAT_FRAME_GENERATION, 1, &off, &off);
                    }
                    if (f == 20) {
                        bp_after_off = host->Stats().back_pressure_waits;
                        core.set_options(WG_FEAT_FRAME_GENERATION, 1, &on, &on);
                    }
                    if (wait_on_game && cached_waitable &&
                        (have_next_credit || WaitForSingleObject(cached_waitable, 300) == WAIT_OBJECT_0))
                        ++waits_ok;
                    have_next_credit = false;
                    const UINT bi = chain->GetCurrentBackBufferIndex();
                    ComPtr<ID3D12Resource> bb;
                    CHECK(SUCCEEDED(chain->GetBuffer(bi, IID_PPV_ARGS(&bb))) && bb);
                    if (!bb) break;
                    if (wait_on_game || !stalled_game || f <= stall_frame || f > stall_end)
                        CHECK(game.Render(gpu, f, bb.Get()));
                    if (stalled_game && f == stall_frame) {
                        CHECK(SUCCEEDED(gpu.queue->Wait(stalled_game.Get(), 1)));
                        if (!wait_on_game) {
                            // A real-only stall cannot release first-present admission via a generated late drop.
                            unblock = std::thread([fence = stalled_game]() {
                                Sleep(420);
                                fence->Signal(1);
                            });
                        }
                    }
                    WgFgTag tags[2] = {};
                    ID3D12Resource* res[2] = {game.motion.Get(), game.depth.Get()};
                    const uint32_t nts[2] = {kNtMotion, kNtDepth};
                    const uint32_t fm[2] = {DXGI_FORMAT_R16G16_FLOAT, DXGI_FORMAT_R32_FLOAT};
                    for (int t = 0; t < 2; ++t) {
                        tags[t].struct_size = sizeof(WgFgTag);
                        tags[t].native_type = nts[t];
                        tags[t].texture.resource = res[t];
                        tags[t].texture.format = fm[t];
                        tags[t].texture.width = kW;
                        tags[t].texture.height = kH;
                        tags[t].texture.state = D3D12_RESOURCE_STATE_COMMON;
                        tags[t].texture.frame_token = tok;
                        tags[t].texture.viewport_id = 1;
                        tags[t].texture.producer_epoch = 3;
                        tags[t].texture.rect = {0, 0, kW, kH};
                    }
                    core.tag(tags, 2, tags);
                    WgFgConstants k = {};
                    k.struct_size = sizeof(k);
                    k.abi_version = WG_FG_ADAPTER_ABI_VERSION;
                    k.frame_token = tok;
                    k.viewport_id = 1;
                    for (int i = 0; i < 4; ++i) k.camera_current[5 * i] = k.camera_previous[5 * i] = 1.0f;
                    k.motion_scale_x = k.motion_scale_y = 1.0f;
                    k.render_rect = {0, 0, kW, kH};
                    k.output_rect = {0, 0, kW, kH};
                    core.constants(&k, &k);
                    WgFgPresentDesc d = {};
                    d.struct_size = sizeof(d);
                    d.abi_version = WG_FG_ADAPTER_ABI_VERSION;
                    d.owner = WG_FG_OWNER_SWAPCHAIN_WRAPPER;
                    d.viewport_id = 1;
                    d.frame_token = tok;
                    d.swapchain_id = chain->Id();
                    d.swapchain_epoch = chain->Epoch();
                    d.window_id = 0x77;
                    d.luid.low = gpu.desc.AdapterLuid.LowPart;
                    d.luid.high = gpu.desc.AdapterLuid.HighPart;
                    d.backbuffer.resource = bb.Get();
                    d.backbuffer.format = DXGI_FORMAT_R16G16B16A16_FLOAT;
                    d.backbuffer.width = kW;
                    d.backbuffer.height = kH;
                    d.backbuffer.frame_token = tok;
                    d.backbuffer.viewport_id = 1;
                    d.backbuffer.rect = {0, 0, kW, kH};
                    bb.Reset();
                    WgFgPresentResult r = {};
                    r.struct_size = sizeof(r);
                    r.abi_version = WG_FG_ADAPTER_ABI_VERSION;
                    FgAsiSessionHost::Route(host.get(), &core, &d, f == stall_frame ? nullptr : &r);
                    CHECK(host->LiveStatus().in_flight_real <= lat);
                    if (FAILED(chain->Present(0, 0))) ++present_errors;
                    if (wait_on_game && stalled_game && f == stall_frame) {
                        if (host->LiveStatus().scheduled_count > 0) CHECK(host->LiveStatus().in_flight_real > 0);
                        have_next_credit = cached_waitable && WaitForSingleObject(cached_waitable, 0) == WAIT_OBJECT_0;
                        CHECK(have_next_credit);
                        CHECK(SUCCEEDED(stalled_game->Signal(1)));
                    }
                    if (wait_on_game || f < stall_frame || f > stall_end) Sleep(16);
                }
                if (unblock.joinable()) unblock.join();
                gpu.Wait();
                Sleep(120);
                const FgAsiSessionStats st = host->Stats();
                const FgAsiLiveStatus live = host->LiveStatus();
                std::printf("    lat=%u: waits=%d/%d retires=%llu credit_waits=%llu bp_off_delta=%llu noop_fs=%llu "
                            "mult_src=%u req=%u worker_reentries=%llu\n",
                            lat, waits_ok, total_frames, static_cast<unsigned long long>(st.real_retires),
                            static_cast<unsigned long long>(st.credit_waits),
                            static_cast<unsigned long long>(bp_after_off - bp_before_off),
                            static_cast<unsigned long long>(st.noop_fullscreen_ignored), live.multiplier_source,
                            live.requested_count, static_cast<unsigned long long>(chain->Stats().worker_reentries));
                CHECK(present_errors == 0);
                if (wait_on_game) {
                    CHECK(waits_ok == total_frames);
                    CHECK(st.real_retires >= static_cast<uint64_t>(total_frames - 4));
                } else {
                    std::printf("    lat=%u without game waits: admission_timeouts=%llu\n", lat,
                                static_cast<unsigned long long>(st.admission_timeouts));
                    CHECK(st.admission_timeouts > 0);
                    UINT available = 0;
                    while (cached_waitable && WaitForSingleObject(cached_waitable, 0) == WAIT_OBJECT_0) ++available;
                    CHECK(available == lat);
                }
                CHECK(bp_after_off == bp_before_off);  // N-B3: 0 back-pressure waits while in-game FG toggle is OFF
                CHECK(live.multiplier_source == kFgMultSourceGame && live.game_has_mfg_selector && live.requested_count == 3);
                CHECK(chain->Stats().worker_reentries == 0);
                if (cached_waitable) CloseHandle(cached_waitable);
                chain->Release();
                CHECK(gpu.Wait());
                host.reset();
                const int64_t dbg = gpu.DebugErrors();
                if (dbg >= 0) CHECK(dbg == 0);
            }
        }

        // R3B-3: persistent admission timeouts count toward kMaxSessionFailures (3) and fall back to plain passthrough
        // instead of looping create -> 100 ms stall -> destroy forever.
        {
            FgPresentOwners owners;
            FakeInterface iface;
            FgProxyConfig pcfg = CoreConfig(2);
            pcfg.binding.has_mfg_selector = true;
            FgProxy core(pcfg, &owners, &PrintLog, nullptr);
            const WgFgChainV1 ct = iface.table();
            CHECK(core.attach_interposer(&ct, 2, 14) == WG_FG_REASON_NONE);
            core.set_feature_function("fgTestSetOptions", &g_fn_a);
            core.set_feature_function("fgTestGetState", &g_fn_b);

            FgAsiSessionConfig hc;
            hc.settings = pcfg.settings;
            hc.dxil_dir = dxil;
            hc.log = &PrintLog;
            std::unique_ptr<wgfg::FgExecWeights> w(new wgfg::FgExecWeights());
            wgfg::FgExecSyntheticWeights(11, w.get());
            std::unique_ptr<FgAsiSessionHost> host(new FgAsiSessionHost(hc, std::move(w)));
            host->OnInstalled(&core);
            FgAsiSessionHost::FactoryNote(host.get(), &core, gpu.factory.Get());

            DXGI_SWAP_CHAIN_DESC1 sd = {};
            sd.Width = kW;
            sd.Height = kH;
            sd.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
            sd.SampleDesc.Count = 1;
            sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
            sd.BufferCount = 3;
            sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
            sd.AlphaMode = DXGI_ALPHA_MODE_IGNORE;
            sd.Flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;
            const UINT extra = host->ExtraSwapChainFlags(sd);
            sd.Flags |= extra;
            IDXGISwapChain1* sc1 = nullptr;
            CHECK(SUCCEEDED(f2->CreateSwapChainForComposition(gpu.queue.Get(), &sd, nullptr, &sc1)) && sc1);
            FgSwapChain* chain = nullptr;
            if (sc1) CHECK(SUCCEEDED(FgSwapChain::Create(sc1, gpu.queue.Get(), &PrintLog, nullptr, &chain)) && chain);
            if (chain) {
                chain->SetExtraFlags(extra & ~DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT);
                host->SetGameWaitable(true);
                CHECK(chain->Attach(host.get(), &core, &owners, WG_FG_OWNER_SWAPCHAIN_WRAPPER) == WG_FG_REASON_NONE);
                host->Attached(chain->Inner(), chain->Queue());
                CHECK(chain->SetMaximumFrameLatency(1) == S_OK);
                WgFgOptions on = Options(true, 2);
                on.generated_count = 1;
                core.set_options(WG_FEAT_FRAME_GENERATION, 1, &on, &on);

                uint64_t tok = 1;
                auto submit_frame = [&](ID3D12Fence* stall_fence, uint64_t stall_val, bool render) {
                    const UINT bi = chain->GetCurrentBackBufferIndex();
                    ComPtr<ID3D12Resource> bb;
                    CHECK(SUCCEEDED(chain->GetBuffer(bi, IID_PPV_ARGS(&bb))) && bb);
                    if (!bb) return;
                    if (render) CHECK(game.Render(gpu, static_cast<int>(tok), bb.Get()));
                    if (stall_fence) CHECK(SUCCEEDED(gpu.queue->Wait(stall_fence, stall_val)));
                    WgFgTag tags[2] = {};
                    ID3D12Resource* res[2] = {game.motion.Get(), game.depth.Get()};
                    const uint32_t nts[2] = {kNtMotion, kNtDepth};
                    const uint32_t fm[2] = {DXGI_FORMAT_R16G16_FLOAT, DXGI_FORMAT_R32_FLOAT};
                    for (int t = 0; t < 2; ++t) {
                        tags[t].struct_size = sizeof(WgFgTag);
                        tags[t].native_type = nts[t];
                        tags[t].texture.resource = res[t];
                        tags[t].texture.format = fm[t];
                        tags[t].texture.width = kW;
                        tags[t].texture.height = kH;
                        tags[t].texture.state = D3D12_RESOURCE_STATE_COMMON;
                        tags[t].texture.frame_token = tok;
                        tags[t].texture.viewport_id = 1;
                        tags[t].texture.producer_epoch = 3;
                        tags[t].texture.rect = {0, 0, kW, kH};
                    }
                    core.tag(tags, 2, tags);
                    WgFgConstants k = {};
                    k.struct_size = sizeof(k);
                    k.abi_version = WG_FG_ADAPTER_ABI_VERSION;
                    k.frame_token = tok;
                    k.viewport_id = 1;
                    for (int i = 0; i < 4; ++i) k.camera_current[5 * i] = k.camera_previous[5 * i] = 1.0f;
                    k.motion_scale_x = k.motion_scale_y = 1.0f;
                    k.render_rect = {0, 0, kW, kH};
                    k.output_rect = {0, 0, kW, kH};
                    core.constants(&k, &k);
                    WgFgPresentDesc d = {};
                    d.struct_size = sizeof(d);
                    d.abi_version = WG_FG_ADAPTER_ABI_VERSION;
                    d.owner = WG_FG_OWNER_SWAPCHAIN_WRAPPER;
                    d.viewport_id = 1;
                    d.frame_token = tok;
                    d.swapchain_id = chain->Id();
                    d.swapchain_epoch = chain->Epoch();
                    d.window_id = 0x77;
                    d.luid.low = gpu.desc.AdapterLuid.LowPart;
                    d.luid.high = gpu.desc.AdapterLuid.HighPart;
                    d.backbuffer.resource = bb.Get();
                    d.backbuffer.format = DXGI_FORMAT_R16G16B16A16_FLOAT;
                    d.backbuffer.width = kW;
                    d.backbuffer.height = kH;
                    d.backbuffer.frame_token = tok;
                    d.backbuffer.viewport_id = 1;
                    d.backbuffer.rect = {0, 0, kW, kH};
                    bb.Reset();
                    WgFgPresentResult r = {};
                    r.struct_size = sizeof(r);
                    r.abi_version = WG_FG_ADAPTER_ABI_VERSION;
                    FgAsiSessionHost::Route(host.get(), &core, &d, &r);
                    CHECK(SUCCEEDED(chain->Present(0, 0)));
                    ++tok;
                };

                // First frame establishes the session description.
                submit_frame(nullptr, 0, true);
                // Three consecutive admission timeouts must hit kMaxSessionFailures and give up.
                for (int round = 0; round < 3; ++round) {
                    // Create the session cleanly on an unstalled queue before stalling the admitted frame.
                    submit_frame(nullptr, 0, true);
                    ComPtr<ID3D12Fence> stall;
                    CHECK(SUCCEEDED(gpu.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&stall))));
                    submit_frame(stall.Get(), 1, true);   // admits and stalls GPU queue
                    // Start the unblock timer AFTER submission so setup never eats the stall.
                    std::thread release([fence = stall]() {
                        Sleep(220);
                        fence->Signal(1);
                    });
                    submit_frame(nullptr, 0, false);      // skip game.Render (which calls gpu.Wait) -> hits 100 ms timeout
                    release.join();
                    gpu.Wait();
                }
                const FgAsiSessionStats gave_st = host->Stats();
                std::printf("    r3b3 persistent-stall: timeouts=%llu failures=%llu gave_up=%d created=%llu phase=%u\n",
                            static_cast<unsigned long long>(gave_st.admission_timeouts),
                            static_cast<unsigned long long>(gave_st.session_failures), gave_st.gave_up ? 1 : 0,
                            static_cast<unsigned long long>(gave_st.sessions_created), host->LiveStatus().phase);
                CHECK(gave_st.admission_timeouts == 3);
                CHECK(gave_st.session_failures == 3 && gave_st.gave_up);
                CHECK(host->LiveStatus().phase == kFgPhaseStopped);
                const uint64_t created_before = gave_st.sessions_created;
                // Subsequent presents must stay on plain passthrough with zero new sessions or admission timeouts.
                for (int extra_f = 0; extra_f < 4; ++extra_f) submit_frame(nullptr, 0, true);
                const FgAsiSessionStats after_st = host->Stats();
                CHECK(after_st.sessions_created == created_before);
                CHECK(after_st.admission_timeouts == 3 && after_st.gave_up);
                CHECK(host->LiveStatus().phase == kFgPhaseStopped);

                CHECK(gpu.Wait());
                chain->Release();
                CHECK(gpu.Wait());
                host.reset();
                const int64_t dbg = gpu.DebugErrors();
                if (dbg >= 0) CHECK(dbg == 0);
            }
        }
    }

    // Part 2: synthetic 2x (120 fps -> 240 Hz) and 4x (80 fps -> 320 Hz) GPU pacing runs (Preset C, DXGI tearing)
    struct Cadence {
        const char* name;
        uint32_t gen_n;
        DWORD sleep_ms;
    };
    const Cadence cadences[2] = {{"2x-120fps", 1u, 8u}, {"4x-80fps", 3u, 12u}};
    if (run_cadence) for (const Cadence& cad : cadences) {
        g_log_lines = 0;
        FgPresentOwners owners;
        FakeInterface iface;
        FgProxyConfig pcfg = CoreConfig(2);
        pcfg.binding.has_mfg_selector = true;
        pcfg.settings.multiplier = 2;
        FgProxy core(pcfg, &owners, &PrintLog, nullptr);
        const WgFgChainV1 ct = iface.table();
        CHECK(core.attach_interposer(&ct, 2, 14) == WG_FG_REASON_NONE);
        core.set_feature_function("fgTestSetOptions", &g_fn_a);
        core.set_feature_function("fgTestGetState", &g_fn_b);

        FgAsiSessionConfig hc;
        hc.settings = pcfg.settings;
        hc.dxil_dir = dxil;
        hc.log = &PrintLog;
        std::unique_ptr<wgfg::FgExecWeights> w(new wgfg::FgExecWeights());
        wgfg::FgExecSyntheticWeights(11, w.get());
        std::unique_ptr<FgAsiSessionHost> host(new FgAsiSessionHost(hc, std::move(w)));
        host->OnInstalled(&core);
        FgAsiSessionHost::FactoryNote(host.get(), &core, gpu.factory.Get());

        DXGI_SWAP_CHAIN_DESC1 sd = {};
        sd.Width = kW;
        sd.Height = kH;
        sd.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        sd.SampleDesc.Count = 1;
        sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        sd.BufferCount = 3;
        sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        sd.AlphaMode = DXGI_ALPHA_MODE_IGNORE;
        const UINT extra = host->ExtraSwapChainFlags(sd);
        sd.Flags |= extra;
        IDXGISwapChain1* sc1 = nullptr;
        CHECK(SUCCEEDED(f2->CreateSwapChainForComposition(gpu.queue.Get(), &sd, nullptr, &sc1)) && sc1);
        if (!sc1) return;
        FgSwapChain* chain = nullptr;
        CHECK(SUCCEEDED(FgSwapChain::Create(sc1, gpu.queue.Get(), &PrintLog, nullptr, &chain)) && chain);
        if (!chain) return;
        chain->SetExtraFlags(extra);
        host->SetGameWaitable(false);
        CHECK(chain->Attach(host.get(), &core, &owners, WG_FG_OWNER_SWAPCHAIN_WRAPPER) == WG_FG_REASON_NONE);
        host->Attached(chain->Inner(), chain->Queue());

        WgFgOptions on = Options(true, 2);
        on.generated_count = cad.gen_n;
        core.set_options(WG_FEAT_FRAME_GENERATION, 1, &on, &on);
        const int frames = 40;
        int present_errors = 0;
        for (int f = 0; f < frames; ++f) {
            const uint64_t tok = static_cast<uint64_t>(f) + 1;
            const UINT bi = chain->GetCurrentBackBufferIndex();
            ComPtr<ID3D12Resource> bb;
            CHECK(SUCCEEDED(chain->GetBuffer(bi, IID_PPV_ARGS(&bb))) && bb);
            if (!bb) break;
            CHECK(game.Render(gpu, f, bb.Get()));
            WgFgTag tags[2] = {};
            ID3D12Resource* res[2] = {game.motion.Get(), game.depth.Get()};
            const uint32_t nts[2] = {kNtMotion, kNtDepth};
            const uint32_t fm[2] = {DXGI_FORMAT_R16G16_FLOAT, DXGI_FORMAT_R32_FLOAT};
            for (int t = 0; t < 2; ++t) {
                tags[t].struct_size = sizeof(WgFgTag);
                tags[t].native_type = nts[t];
                tags[t].texture.resource = res[t];
                tags[t].texture.format = fm[t];
                tags[t].texture.width = kW;
                tags[t].texture.height = kH;
                tags[t].texture.state = D3D12_RESOURCE_STATE_COMMON;
                tags[t].texture.frame_token = tok;
                tags[t].texture.viewport_id = 1;
                tags[t].texture.producer_epoch = 3;
                tags[t].texture.rect = {0, 0, kW, kH};
            }
            core.tag(tags, 2, tags);
            WgFgConstants k = {};
            k.struct_size = sizeof(k);
            k.abi_version = WG_FG_ADAPTER_ABI_VERSION;
            k.frame_token = tok;
            k.viewport_id = 1;
            for (int i = 0; i < 4; ++i) k.camera_current[5 * i] = k.camera_previous[5 * i] = 1.0f;
            k.motion_scale_x = k.motion_scale_y = 1.0f;
            k.render_rect = {0, 0, kW, kH};
            k.output_rect = {0, 0, kW, kH};
            core.constants(&k, &k);
            WgFgPresentDesc d = {};
            d.struct_size = sizeof(d);
            d.abi_version = WG_FG_ADAPTER_ABI_VERSION;
            d.owner = WG_FG_OWNER_SWAPCHAIN_WRAPPER;
            d.viewport_id = 1;
            d.frame_token = tok;
            d.swapchain_id = chain->Id();
            d.swapchain_epoch = chain->Epoch();
            d.window_id = 0x77;
            d.luid.low = gpu.desc.AdapterLuid.LowPart;
            d.luid.high = gpu.desc.AdapterLuid.HighPart;
            d.backbuffer.resource = bb.Get();
            d.backbuffer.format = DXGI_FORMAT_R16G16B16A16_FLOAT;
            d.backbuffer.width = kW;
            d.backbuffer.height = kH;
            d.backbuffer.frame_token = tok;
            d.backbuffer.viewport_id = 1;
            d.backbuffer.rect = {0, 0, kW, kH};
            bb.Reset();
            WgFgPresentResult r = {};
            r.struct_size = sizeof(r);
            r.abi_version = WG_FG_ADAPTER_ABI_VERSION;
            FgAsiSessionHost::Route(host.get(), &core, &d, &r);
            if (FAILED(chain->Present(0, 0))) ++present_errors;
            Sleep(cad.sleep_ms);
        }
        gpu.Wait();
        Sleep(150);
        const FgAsiSessionStats st = host->Stats();
        const FgAsiLiveStatus live = host->LiveStatus();
        std::printf("    pacing %s: intervals=%llu session=%llu indices=%llu retires=%llu bp_waits=%llu "
                    "disp_gen=%llu fps=%.1f/%.1f worker_reentries=%llu\n",
                    cad.name, static_cast<unsigned long long>(st.intervals),
                    static_cast<unsigned long long>(st.session_presents),
                    static_cast<unsigned long long>(st.exec_indices),
                    static_cast<unsigned long long>(st.real_retires),
                    static_cast<unsigned long long>(st.back_pressure_waits),
                    static_cast<unsigned long long>(live.displayed_generated), live.displayed_fps, live.base_fps,
                    static_cast<unsigned long long>(chain->Stats().worker_reentries));
        CHECK(present_errors == 0 && st.sessions_created == 1 && st.session_failures == 0);
        CHECK(st.exec_indices >= static_cast<uint64_t>(cad.gen_n * 20));
        CHECK(st.real_retires >= static_cast<uint64_t>(frames - 3));
        CHECK(live.displayed_generated > 0);
        CHECK(live.displayed_fps > live.base_fps && live.base_fps > 0.0f);
        CHECK(chain->Stats().worker_reentries == 0);

        // R3B-1 end-to-end: route a scheduler-driven real-only frame through FgAsiSessionHost::Route() -> LiveStatus()
        // -> panel::PublishAt() on an active session and confirm kFgPhaseGenerating + activity hold are preserved.
        if (cad.gen_n == 1u) {
            static WgHostMenuApi v6_live = fakehost::Api(true);
            v6_live.version = 6;
            v6_live.checkbox = &fakehost::Checkbox;
            v6_live.report_frame_generation = &fakehost::ReportFrameGeneration;
            v6_live.register_frame_gen_section = &fakehost::RegisterFrameGenSection;
            CHECK(panel::ClaimOutput(&v6_live, nullptr));
            panel::Status ps;
            ps.enabled = true;
            ps.installed = true;
            ps.have_host = true;
            ps.live = host->LiveStatus();
            CHECK(ps.live.phase == kFgPhaseGenerating && ps.live.displayed_generated > 0);
            CHECK(panel::PublishAt(ps, panel::NextReportSequence(), true, 10000));
            CHECK(fakehost::g_report_generating);

            // Drive a short zero-sleep burst (>360 Hz base) so the scheduler drops scheduled_count to 0 inside the
            // active session (re-anchor / not-primed / out-of-vrr-window).
            bool saw_sched_real_only = false;
            panel::Status reduced = ps;
            for (int bf = 0; bf < 12; ++bf) {
                const uint64_t tok = static_cast<uint64_t>(frames + bf) + 1;
                const UINT bi = chain->GetCurrentBackBufferIndex();
                ComPtr<ID3D12Resource> bb;
                CHECK(SUCCEEDED(chain->GetBuffer(bi, IID_PPV_ARGS(&bb))) && bb);
                if (!bb) break;
                CHECK(game.Render(gpu, frames + bf, bb.Get()));
                WgFgTag tags[2] = {};
                ID3D12Resource* res[2] = {game.motion.Get(), game.depth.Get()};
                const uint32_t nts[2] = {kNtMotion, kNtDepth};
                const uint32_t fm[2] = {DXGI_FORMAT_R16G16_FLOAT, DXGI_FORMAT_R32_FLOAT};
                for (int t = 0; t < 2; ++t) {
                    tags[t].struct_size = sizeof(WgFgTag);
                    tags[t].native_type = nts[t];
                    tags[t].texture.resource = res[t];
                    tags[t].texture.format = fm[t];
                    tags[t].texture.width = kW;
                    tags[t].texture.height = kH;
                    tags[t].texture.state = D3D12_RESOURCE_STATE_COMMON;
                    tags[t].texture.frame_token = tok;
                    tags[t].texture.viewport_id = 1;
                    tags[t].texture.producer_epoch = 3;
                    tags[t].texture.rect = {0, 0, kW, kH};
                }
                core.tag(tags, 2, tags);
                WgFgConstants k = {};
                k.struct_size = sizeof(k);
                k.abi_version = WG_FG_ADAPTER_ABI_VERSION;
                k.frame_token = tok;
                k.viewport_id = 1;
                for (int i = 0; i < 4; ++i) k.camera_current[5 * i] = k.camera_previous[5 * i] = 1.0f;
                k.motion_scale_x = k.motion_scale_y = 1.0f;
                k.render_rect = {0, 0, kW, kH};
                k.output_rect = {0, 0, kW, kH};
                core.constants(&k, &k);
                WgFgPresentDesc d = {};
                d.struct_size = sizeof(d);
                d.abi_version = WG_FG_ADAPTER_ABI_VERSION;
                d.owner = WG_FG_OWNER_SWAPCHAIN_WRAPPER;
                d.viewport_id = 1;
                d.frame_token = tok;
                d.swapchain_id = chain->Id();
                d.swapchain_epoch = chain->Epoch();
                d.window_id = 0x77;
                d.luid.low = gpu.desc.AdapterLuid.LowPart;
                d.luid.high = gpu.desc.AdapterLuid.HighPart;
                d.backbuffer.resource = bb.Get();
                d.backbuffer.format = DXGI_FORMAT_R16G16B16A16_FLOAT;
                d.backbuffer.width = kW;
                d.backbuffer.height = kH;
                d.backbuffer.frame_token = tok;
                d.backbuffer.viewport_id = 1;
                d.backbuffer.rect = {0, 0, kW, kH};
                bb.Reset();
                WgFgPresentResult r = {};
                r.struct_size = sizeof(r);
                r.abi_version = WG_FG_ADAPTER_ABI_VERSION;
                FgAsiSessionHost::Route(host.get(), &core, &d, &r);
                const FgAsiLiveStatus cur_live = host->LiveStatus();
                if (SUCCEEDED(chain->Present(0, 0)) && r.status == WG_FG_PRESENT_GENERATE &&
                    cur_live.scheduled_count == 0 && cur_live.schedule_reason != 0) {
                    const bool is_non_priming =
                        static_cast<wgfg::FgRealOnlyReason>(cur_live.schedule_reason) !=
                        wgfg::FgRealOnlyReason::NotPrimed;
                    if (!saw_sched_real_only || is_non_priming) {
                        saw_sched_real_only = true;
                        reduced.live = cur_live;
                    }
                    if (is_non_priming) break;
                }
            }
            CHECK(saw_sched_real_only);
            if (saw_sched_real_only) {
                CHECK(reduced.live.phase == kFgPhaseGenerating);
                if (static_cast<wgfg::FgRealOnlyReason>(reduced.live.schedule_reason) ==
                    wgfg::FgRealOnlyReason::OutOfVrrWindow) {
                    CHECK(panel::StatusLine(reduced, pcfg.settings) == "Real frames only: display cap");
                }
                CHECK(panel::PublishAt(reduced, panel::NextReportSequence(), false, 10100));
                CHECK(fakehost::g_report_generating);  // activity hold preserved across live scheduler reduction
                CHECK(panel::PublishAt(reduced, panel::NextReportSequence(), false, 10600));
                CHECK(!fakehost::g_report_generating);  // expires cleanly after 500 ms
            }
            panel::ReleaseOutput();
        }

        chain->Release();
        CHECK(gpu.Wait());
        host.reset();
        const int64_t dbg = gpu.DebugErrors();
        if (dbg >= 0) CHECK(dbg == 0);
    }
}

// RC2 smoke regressions: GPU-bound synthetic game at L=1 where real_ready lags arrival by ~1 frame, with slow startup
// frames (25 ms) transitioning into steady 10.5 ms / 12.5 ms / 18.5 ms GPU-bound rendering at 2x (N=1), 4x (N=3), and
// 6x (N=5). Proves no bistable 2x LatencyCap trap, < 2% late drops at 2x, < 5% late drops at 4x/6x, base FPS >= 85%
// of FG-off, and presented output scaling cleanly with N.
void TestSmokeGpuBoundPacing(Gpu& gpu, const std::string& dxil, const std::string& only) {
    const bool admission_only = only == "admission";
    ComPtr<IDXGIFactory2> f2;
    CHECK(SUCCEEDED(gpu.factory.As(&f2)));
    GameTextures game;
    CHECK(game.Init(gpu));
    if (!f2 || !game.motion) return;

    struct SmokeLogCapture {
        std::mutex mu;
        uint64_t sched = 0;
        uint64_t sub = 0;
        uint64_t dropped_late = 0;
        uint64_t pre_drop = 0;
        uint64_t post_drop = 0;
        uint64_t real_sub = 0;
        uint64_t latency_cap_ro = 0;
        uint64_t generating_reanchors = 0;
        double final_budget_ms = 0.0;
        bool saw_stats = false;
        bool warned_unknown_mode = false;

        static bool ParseU64(const char* line, const char* key, uint64_t* out) {
            const char* p = std::strstr(line, key);
            if (!p) return false;
            p += std::strlen(key);
            char* end = nullptr;
            const unsigned long long v = std::strtoull(p, &end, 10);
            if (end == p) return false;
            *out = static_cast<uint64_t>(v);
            return true;
        }
        static bool ParseDouble(const char* line, const char* key, double* out) {
            const char* p = std::strstr(line, key);
            if (!p) return false;
            p += std::strlen(key);
            char* end = nullptr;
            const double v = std::strtod(p, &end);
            if (end == p) return false;
            *out = v;
            return true;
        }
        static void OnLog(void* ctx, const char* line) {
            if (!ctx || !line) return;
            auto* self = static_cast<SmokeLogCapture*>(ctx);
            std::lock_guard<std::mutex> lock(self->mu);
            if (std::strstr(line, "unknown WG_FG_ADMISSION value")) {
                self->warned_unknown_mode = true;
            } else if (std::strstr(line, "fg host: presenter stats:")) {
                self->saw_stats = true;
                ParseU64(line, "sched=", &self->sched);
                ParseU64(line, "sub=", &self->sub);
                ParseU64(line, "dropped_late=", &self->dropped_late);
                ParseU64(line, "pre_drop=", &self->pre_drop);
                ParseU64(line, "post_drop=", &self->post_drop);
                ParseU64(line, "real_sub=", &self->real_sub);
                ParseDouble(line, "budget_ms=", &self->final_budget_ms);
                std::printf("      log: %s\n", line);
            } else if (std::strstr(line, "fg host: real-only intervals:")) {
                ParseU64(line, "latency-cap=", &self->latency_cap_ro);
                ParseU64(line, "generating_reanchors=", &self->generating_reanchors);
                std::printf("      log: %s\n", line);
            } else if (std::strstr(line, "first-image budget") || std::strstr(line, "generated count cap") ||
                       std::strstr(line, "trace summary")) {
                std::printf("      log: %s\n", line);
            }
        }
    };

    struct AsyncGpuPacer {
        enum { kRing = 4 };
        ComPtr<ID3D12CommandAllocator> alloc[kRing];
        ComPtr<ID3D12GraphicsCommandList> list;
        ComPtr<ID3D12Resource> upload[kRing];
        ComPtr<ID3D12Fence> ring_fence;
        ComPtr<ID3D12Fence> sim_fence;
        uint64_t ring_val[kRing] = {};
        uint64_t ring_seq = 0;
        uint64_t sim_seq = 0;
        HANDLE ring_event = nullptr;
        HANDLE cpu_timer = nullptr;

        struct Job {
            uint64_t val = 0;
            std::chrono::steady_clock::time_point submit_tp;
            int64_t dur_us = 0;
        };
        std::mutex mu;
        std::condition_variable cv;
        std::deque<Job> jobs;
        bool stop = false;
        bool abort_fast = false;
        std::thread worker;

        ~AsyncGpuPacer() {
            Stop();
            if (ring_event) CloseHandle(ring_event);
            if (cpu_timer) CloseHandle(cpu_timer);
        }

        static void WaitUs(HANDLE timer, int64_t us) {
            if (us <= 0) return;
            if (timer) {
                LARGE_INTEGER due;
                due.QuadPart = -us * 10LL;
                if (SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE)) {
                    WaitForSingleObject(timer, static_cast<DWORD>(us / 1000 + 50));
                    return;
                }
            }
            Sleep(static_cast<DWORD>((us + 999) / 1000));
        }

        bool Init(Gpu& g) {
            ring_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
            cpu_timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
            if (!ring_event) return false;
            if (FAILED(g.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&ring_fence))) ||
                FAILED(g.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&sim_fence))))
                return false;
            const UINT pitch = (kW * 8 + 255) & ~255u;
            for (int s = 0; s < kRing; ++s) {
                if (FAILED(g.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc[s]))))
                    return false;
                D3D12_HEAP_PROPERTIES hp = {};
                hp.Type = D3D12_HEAP_TYPE_UPLOAD;
                D3D12_RESOURCE_DESC rd = {};
                rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
                rd.Width = static_cast<UINT64>(pitch) * kH;
                rd.Height = 1;
                rd.DepthOrArraySize = 1;
                rd.MipLevels = 1;
                rd.SampleDesc.Count = 1;
                rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
                if (FAILED(g.device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
                                                             D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                             IID_PPV_ARGS(&upload[s]))))
                    return false;
                uint8_t* p = nullptr;
                if (FAILED(upload[s]->Map(0, nullptr, reinterpret_cast<void**>(&p)))) return false;
                for (UINT y = 0; y < kH; ++y) {
                    for (UINT x = 0; x < kW; ++x) {
                        uint16_t* t = reinterpret_cast<uint16_t*>(p + static_cast<size_t>(y) * pitch + 8ull * x);
                        const UINT v = (x + 7 * static_cast<UINT>(s)) % 64;
                        t[0] = static_cast<uint16_t>(0x3000 + v * 8);
                        t[1] = static_cast<uint16_t>(0x3400 + ((y / 8) % 32) * 8);
                        t[2] = 0x3800;
                        t[3] = 0x3C00;
                    }
                }
                upload[s]->Unmap(0, nullptr);
            }
            if (FAILED(g.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc[0].Get(), nullptr,
                                                   IID_PPV_ARGS(&list))) ||
                FAILED(list->Close()))
                return false;
            worker = std::thread([this]() {
                HANDLE wt =
                    CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
                auto busy_until = std::chrono::steady_clock::now();
                for (;;) {
                    Job job;
                    bool fast = false;
                    {
                        std::unique_lock<std::mutex> lock(mu);
                        cv.wait(lock, [this]() { return stop || !jobs.empty(); });
                        if (jobs.empty()) break;
                        job = jobs.front();
                        jobs.pop_front();
                        fast = abort_fast;
                    }
                    if (!fast && job.dur_us > 0) {
                        const auto start_tp = job.submit_tp > busy_until ? job.submit_tp : busy_until;
                        const auto done_tp = start_tp + std::chrono::microseconds(job.dur_us);
                        const auto now_tp = std::chrono::steady_clock::now();
                        if (done_tp > now_tp) {
                            const int64_t rem_us =
                                std::chrono::duration_cast<std::chrono::microseconds>(done_tp - now_tp).count();
                            WaitUs(wt, rem_us);
                        }
                        busy_until = done_tp;
                    }
                    sim_fence->Signal(job.val);
                }
                if (wt) CloseHandle(wt);
            });
            return true;
        }

        bool Submit(Gpu& g, int f, ID3D12Resource* dst, int64_t gpu_us, int64_t overlap_us) {
            const int s = f & (kRing - 1);
            if (ring_val[s] > 0 && ring_fence->GetCompletedValue() < ring_val[s]) {
                ring_fence->SetEventOnCompletion(ring_val[s], ring_event);
                if (WaitForSingleObject(ring_event, 5000) != WAIT_OBJECT_0) return false;
            }
            if (FAILED(alloc[s]->Reset()) || FAILED(list->Reset(alloc[s].Get(), nullptr))) return false;
            const UINT pitch = (kW * 8 + 255) & ~255u;
            D3D12_RESOURCE_BARRIER b = {};
            b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            b.Transition = {dst, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COMMON,
                            D3D12_RESOURCE_STATE_COPY_DEST};
            list->ResourceBarrier(1, &b);
            D3D12_TEXTURE_COPY_LOCATION d = {}, src = {};
            d.pResource = dst;
            d.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            src.pResource = upload[s].Get();
            src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            src.PlacedFootprint.Footprint = {DXGI_FORMAT_R16G16B16A16_FLOAT, kW, kH, 1, pitch};
            list->CopyTextureRegion(&d, 0, 0, 0, &src, nullptr);
            std::swap(b.Transition.StateBefore, b.Transition.StateAfter);
            list->ResourceBarrier(1, &b);
            if (FAILED(list->Close())) return false;
            ID3D12CommandList* cl[] = {list.Get()};
            g.queue->ExecuteCommandLists(1, cl);
            const uint64_t sv = ++sim_seq;
            if (FAILED(g.queue->Wait(sim_fence.Get(), sv))) return false;
            {
                std::lock_guard<std::mutex> lock(mu);
                // The title can start feeding the GPU before its CPU thread reaches Present. Its serialized render
                // service time is still gpu_us. Only an idle GPU gains the submission overlap.
                jobs.push_back({sv, std::chrono::steady_clock::now() - std::chrono::microseconds(overlap_us), gpu_us});
            }
            cv.notify_one();
            ring_val[s] = ++ring_seq;
            return SUCCEEDED(g.queue->Signal(ring_fence.Get(), ring_val[s]));
        }

        void Stop() {
            {
                std::lock_guard<std::mutex> lock(mu);
                stop = true;
                abort_fast = true;
            }
            cv.notify_all();
            if (worker.joinable()) worker.join();
        }
    };

    struct SmokeRunResult {
        double steady_base_fps = 0.0;
        double late_drop_pct = 0.0;
        double steady_late_drop_pct = 0.0;
        double submitted_mult = 1.0;
        uint64_t latency_cap_ro = 0;
        uint64_t sched = 0;
        uint64_t sub = 0;
        uint64_t dropped_late = 0;
        uint64_t pre_drop = 0;
        uint64_t post_drop = 0;
        uint64_t real_sub = 0;
        double final_budget_ms = 0.0;
        int present_errors = 0;
        uint64_t session_failures = 0;
        uint64_t queue_waits = 0;
        uint32_t queue_peak = 0;
        uint64_t admission_present = 0, admission_consumed = 0;
        uint64_t session_presents = 0;
        uint32_t steady_max_scheduled = 0;
        uint64_t generating_reanchors = 0;
        uint32_t low_count_consumed = 0;
        uint32_t low_count_frames = 0;
        uint32_t recovered_high = 0;
        int64_t gap_p95_us = 0;
        bool warned_unknown_mode = false;
    };

    auto run_leg = [&](const char* label, bool fg_enable, uint32_t gen_n, int slow_frames, int64_t slow_us,
                       int fast_frames, int64_t gpu_us, int64_t cpu_us, bool switch_counts = false) -> SmokeRunResult {
        SmokeRunResult out;
        SmokeLogCapture cap;
        FgPresentOwners owners;
        FakeInterface iface;
        FgProxyConfig pcfg = CoreConfig(2);
        pcfg.binding.has_mfg_selector = true;
        pcfg.settings.multiplier = gen_n + 1;
        FgProxy core(pcfg, &owners, &SmokeLogCapture::OnLog, &cap);
        const WgFgChainV1 ct = iface.table();
        CHECK(core.attach_interposer(&ct, 2, 14) == WG_FG_REASON_NONE);
        core.set_feature_function("fgTestSetOptions", &g_fn_a);
        core.set_feature_function("fgTestGetState", &g_fn_b);

        FgAsiSessionConfig hc;
        hc.settings = pcfg.settings;
        hc.dxil_dir = dxil;
        hc.log = &SmokeLogCapture::OnLog;
        hc.log_ctx = &cap;
        std::unique_ptr<wgfg::FgExecWeights> w(new wgfg::FgExecWeights());
        wgfg::FgExecSyntheticWeights(11, w.get());
        std::unique_ptr<FgAsiSessionHost> host(new FgAsiSessionHost(hc, std::move(w)));
        host->OnInstalled(&core);
        FgAsiSessionHost::FactoryNote(host.get(), &core, gpu.factory.Get());

        DXGI_SWAP_CHAIN_DESC1 sd = {};
        sd.Width = kW;
        sd.Height = kH;
        sd.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        sd.SampleDesc.Count = 1;
        sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        sd.BufferCount = 3;
        sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        sd.AlphaMode = DXGI_ALPHA_MODE_IGNORE;
        sd.Flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;
        const UINT extra = host->ExtraSwapChainFlags(sd);
        sd.Flags |= extra;
        // Mirror the production DXGI factory hook: create the real swap chain on the host's dedicated present queue so
        // generated-image presents never serialize behind the game's next frame on gpu.queue.
        IUnknown* pq = host->CreationQueue(gpu.queue.Get(), sd);
        IDXGISwapChain1* sc1 = nullptr;
        CHECK(SUCCEEDED(f2->CreateSwapChainForComposition(pq ? pq : gpu.queue.Get(), &sd, nullptr, &sc1)) && sc1);
        if (!sc1) return out;
        FgSwapChain* chain = nullptr;
        CHECK(SUCCEEDED(FgSwapChain::Create(sc1, gpu.queue.Get(), &SmokeLogCapture::OnLog, &cap, &chain)) && chain);
        if (!chain) return out;
        chain->SetExtraFlags(extra & ~DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT);
        host->SetGameWaitable(true);
        if (pq) host->AttachedPresentQueue(pq);
        CHECK(chain->Attach(host.get(), &core, &owners, WG_FG_OWNER_SWAPCHAIN_WRAPPER) == WG_FG_REASON_NONE);
        host->Attached(chain->Inner(), chain->Queue());
        CHECK(chain->SetMaximumFrameLatency(1) == S_OK);

        WgFgOptions opt = Options(fg_enable, 2);
        opt.generated_count = gen_n;
        core.set_options(WG_FEAT_FRAME_GENERATION, 1, &opt, &opt);

        AsyncGpuPacer pacer;
        CHECK(pacer.Init(gpu));
        HANDLE lat = chain->GetFrameLatencyWaitableObject();
        CHECK(lat != nullptr);

        const int total_frames = slow_frames + fast_frames;
        // Measure steady-state base FPS over the remaining 100-180 frames after slow-startup recovery (~60 frames).
        const int steady_from = slow_frames + (admission_only ? 100 : (fast_frames > 100 ? 60 : 15));
        std::chrono::steady_clock::time_point steady_t0{};
        int steady_count = 0;
        FgAsiSessionStats steady_stats;
        uint64_t last_consumed_count = 0;
        std::vector<int64_t> admission_gaps;
        auto previous_present = std::chrono::steady_clock::now();
        const bool switch_cadence = only == "admission-switch";
        for (int f = 0; f < total_frames; ++f) {
            if (switch_cadence && (f == 80 || f == 180)) {
                opt.generated_count = f == 80 ? 2u : 3u;
                core.set_options(WG_FEAT_FRAME_GENERATION, 1, &opt, &opt);
            }
            if (switch_counts && (f == 35 || f == 75)) {
                opt.generated_count = f == 35 ? 3u : 2u;
                core.set_options(WG_FEAT_FRAME_GENERATION, 1, &opt, &opt);
            }
            if (lat) {
                const DWORD wait = WaitForSingleObject(lat, 200);
                if (admission_only) CHECK(wait == WAIT_OBJECT_0);
            }
            const bool is_slow = f < slow_frames;
            const int64_t frame_cpu_us = is_slow ? slow_us : cpu_us;
            const int64_t frame_gpu_us = is_slow ? slow_us : gpu_us;
            AsyncGpuPacer::WaitUs(pacer.cpu_timer, frame_cpu_us);
            const uint64_t tok = static_cast<uint64_t>(f) + 1;
            const UINT bi = chain->GetCurrentBackBufferIndex();
            ComPtr<ID3D12Resource> bb;
            CHECK(SUCCEEDED(chain->GetBuffer(bi, IID_PPV_ARGS(&bb))) && bb);
            if (!bb) break;
            CHECK(pacer.Submit(gpu, f, bb.Get(), frame_gpu_us, admission_only ? 1000 : 0));
            WgFgTag tags[2] = {};
            ID3D12Resource* res[2] = {game.motion.Get(), game.depth.Get()};
            const uint32_t nts[2] = {kNtMotion, kNtDepth};
            const uint32_t fm[2] = {DXGI_FORMAT_R16G16_FLOAT, DXGI_FORMAT_R32_FLOAT};
            for (int t = 0; t < 2; ++t) {
                tags[t].struct_size = sizeof(WgFgTag);
                tags[t].native_type = nts[t];
                tags[t].texture.resource = res[t];
                tags[t].texture.format = fm[t];
                tags[t].texture.width = kW;
                tags[t].texture.height = kH;
                tags[t].texture.state = D3D12_RESOURCE_STATE_COMMON;
                tags[t].texture.frame_token = tok;
                tags[t].texture.viewport_id = 1;
                tags[t].texture.producer_epoch = 3;
                tags[t].texture.rect = {0, 0, kW, kH};
            }
            core.tag(tags, 2, tags);
            WgFgConstants k = {};
            k.struct_size = sizeof(k);
            k.abi_version = WG_FG_ADAPTER_ABI_VERSION;
            k.frame_token = tok;
            k.viewport_id = 1;
            for (int i = 0; i < 4; ++i) k.camera_current[5 * i] = k.camera_previous[5 * i] = 1.0f;
            k.motion_scale_x = k.motion_scale_y = 1.0f;
            k.render_rect = {0, 0, kW, kH};
            k.output_rect = {0, 0, kW, kH};
            core.constants(&k, &k);
            WgFgPresentDesc d = {};
            d.struct_size = sizeof(d);
            d.abi_version = WG_FG_ADAPTER_ABI_VERSION;
            d.owner = WG_FG_OWNER_SWAPCHAIN_WRAPPER;
            d.viewport_id = 1;
            d.frame_token = tok;
            d.swapchain_id = chain->Id();
            d.swapchain_epoch = chain->Epoch();
            d.window_id = 0x77;
            d.luid.low = gpu.desc.AdapterLuid.LowPart;
            d.luid.high = gpu.desc.AdapterLuid.HighPart;
            d.backbuffer.resource = bb.Get();
            d.backbuffer.format = DXGI_FORMAT_R16G16B16A16_FLOAT;
            d.backbuffer.width = kW;
            d.backbuffer.height = kH;
            d.backbuffer.frame_token = tok;
            d.backbuffer.viewport_id = 1;
            d.backbuffer.rect = {0, 0, kW, kH};
            bb.Reset();
            WgFgPresentResult r = {};
            r.struct_size = sizeof(r);
            r.abi_version = WG_FG_ADAPTER_ABI_VERSION;
            FgAsiSessionHost::Route(host.get(), &core, &d, &r);
            if (FAILED(chain->Present(0, 0))) ++out.present_errors;
            if (switch_cadence) {
                const auto present = std::chrono::steady_clock::now();
                if (f >= 65)
                    admission_gaps.push_back(std::chrono::duration_cast<std::chrono::microseconds>(
                                                 present - previous_present).count());
                previous_present = present;
                const auto current = host->Stats().admission_consumed_intervals;
                const auto scheduled = host->LiveStatus().scheduled_count;
                if (f >= 100 && f < 180 && scheduled > 0 && scheduled < 3) ++out.low_count_frames;
                if (scheduled < 3 && current > last_consumed_count) ++out.low_count_consumed;
                if (f >= 200 && scheduled >= 3) ++out.recovered_high;
                last_consumed_count = current;
            }
            if (admission_only) CHECK(host->LiveStatus().in_flight_real <= 1);
            if (f == steady_from) {
                steady_t0 = std::chrono::steady_clock::now();
                steady_stats = host->Stats();
            } else if (f > steady_from) {
                ++steady_count;
                out.steady_max_scheduled = std::max(out.steady_max_scheduled, host->LiveStatus().scheduled_count);
            }
        }
        const auto steady_t1 = std::chrono::steady_clock::now();
        const double steady_s =
            std::chrono::duration_cast<std::chrono::microseconds>(steady_t1 - steady_t0).count() / 1e6;
        if (steady_s > 0.0 && steady_count > 0) out.steady_base_fps = static_cast<double>(steady_count) / steady_s;
        if (lat) CloseHandle(lat);
        // Let the last 1-2 in-flight GPU frames finish cleanly before teardown.
        Sleep(40);
        pacer.Stop();
        CHECK(gpu.Wait());
        const FgAsiSessionStats st = host->Stats();
        out.session_failures = st.session_failures;
        out.queue_waits = st.presenter_queue_waits;
        out.queue_peak = st.presenter_queue_peak;
        out.admission_present = st.admission_present_intervals;
        out.admission_consumed = st.admission_consumed_intervals;
        out.session_presents = st.session_presents;
        if (st.generated_scheduled > steady_stats.generated_scheduled && st.generated_late >= steady_stats.generated_late)
            out.steady_late_drop_pct = 100.0 * static_cast<double>(st.generated_late - steady_stats.generated_late) /
                                      static_cast<double>(st.generated_scheduled - steady_stats.generated_scheduled);
        chain->Release();
        CHECK(gpu.Wait());
        host.reset();
        const int64_t dbg = gpu.DebugErrors();
        if (dbg >= 0) CHECK(dbg == 0);

        out.sched = cap.sched;
        out.sub = cap.sub;
        out.dropped_late = cap.dropped_late;
        out.pre_drop = cap.pre_drop;
        out.post_drop = cap.post_drop;
        out.real_sub = cap.real_sub;
        out.latency_cap_ro = cap.latency_cap_ro;
        out.warned_unknown_mode = cap.warned_unknown_mode;
        out.generating_reanchors = cap.generating_reanchors;
        out.gap_p95_us = Percentile(std::move(admission_gaps), 0.95);
        out.final_budget_ms = cap.final_budget_ms;
        if (cap.sched > 0)
            out.late_drop_pct = 100.0 * static_cast<double>(cap.dropped_late) / static_cast<double>(cap.sched);
        if (cap.real_sub > 0)
            out.submitted_mult =
                static_cast<double>(cap.real_sub + cap.sub) / static_cast<double>(cap.real_sub);

        std::printf("    smoke %-22s: base=%.1f fps | sched=%llu sub=%llu late=%llu (%.2f%%, pre=%llu post=%llu) "
                    "| real_sub=%llu mult=%.2fx ro_cap=%llu budget=%.1fms steady_late=%.2f%% queue_peak=%u "
                    "queue_waits=%llu steady_max_n=%u\n",
                    label, out.steady_base_fps, static_cast<unsigned long long>(out.sched),
                    static_cast<unsigned long long>(out.sub), static_cast<unsigned long long>(out.dropped_late),
                    out.late_drop_pct, static_cast<unsigned long long>(out.pre_drop),
                    static_cast<unsigned long long>(out.post_drop), static_cast<unsigned long long>(out.real_sub),
                    out.submitted_mult, static_cast<unsigned long long>(out.latency_cap_ro), out.final_budget_ms,
                    out.steady_late_drop_pct, out.queue_peak, static_cast<unsigned long long>(out.queue_waits),
                    out.steady_max_scheduled);
        return out;
    };

    if (only == "admission-switch") {
        const auto run = run_leg("active-3-to-2-to-3", true, 3u, 8, 25000, 272, 18500, 6500);
        std::printf("    active count switch: low=%u wrongly_consumed=%u recovered=%u gap_p95=%lldus reanchors=%llu\n",
                    run.low_count_frames, run.low_count_consumed, run.recovered_high,
                    static_cast<long long>(run.gap_p95_us), static_cast<unsigned long long>(run.generating_reanchors));
        CHECK(run.present_errors == 0 && run.session_failures == 0);
        CHECK(run.low_count_frames >= 40);
        CHECK(run.low_count_consumed == 0);
        CHECK(run.recovered_high >= 40);
        CHECK(run.admission_consumed == 0);
        CHECK(run.gap_p95_us < 27000);
        CHECK(run.generating_reanchors < 35);
        return;
    }

    if (only == "admission-policy") {
        char env_mode[16] = {};
        const auto length = GetEnvironmentVariableA("WG_FG_ADMISSION", env_mode, sizeof(env_mode));
        const bool force_present = length == 7 && std::strcmp(env_mode, "present") == 0;
        const bool force_consumed = length == 8 && std::strcmp(env_mode, "consumed") == 0;
        const bool explicit_policy = length == 6 && std::strcmp(env_mode, "policy") == 0;
        for (const uint32_t requested : {0u, 1u, 2u, 3u, 5u}) {
            const auto run = run_leg("admission-policy", requested > 0, std::max(1u, requested), 0, 0, 110, 18500, 6500);
            CHECK(run.present_errors == 0 && run.session_failures == 0);
            CHECK(run.real_sub >= 105);
            if (requested > 0) CHECK(run.sched >= 75);
            const bool expects_consumed = requested > 0 && force_consumed;
            CHECK((run.queue_peak > 0) == expects_consumed);
            CHECK(run.queue_peak <= 2);
            CHECK(run.admission_consumed > 50 || !expects_consumed);
            CHECK(run.admission_consumed == 0 || expects_consumed);
            CHECK(run.admission_present > 0);
            // The host counts acceptance. Teardown may cancel the last accepted endpoint before DXGI submission.
            CHECK(run.admission_present + run.admission_consumed == run.session_presents);
            CHECK(run.warned_unknown_mode == (length != 0 && !force_present && !force_consumed && !explicit_policy));
        }
        if (!force_present && !force_consumed) {
            const auto mixed = run_leg("2x-to-4x-to-3x", true, 1u, 0, 0, 112, 18500, 6500, true);
            CHECK(mixed.present_errors == 0 && mixed.session_failures == 0);
            CHECK(mixed.admission_present >= 105 && mixed.admission_consumed == 0);
            CHECK(mixed.queue_peak == 0);
            CHECK(mixed.admission_present + mixed.admission_consumed == mixed.session_presents);
        }
        return;
    }

    if (only == "consumed-6x") {
        // RT-on windows at x6 were display-limited to about five output images per 15.8 ms interval.
        const SmokeRunResult on = run_leg("x6-15.8ms-rt", true, 5u, 12, 25000, 240, 15800, 7600);
        CHECK(on.present_errors == 0 && on.session_failures == 0);
        CHECK(on.steady_base_fps >= 56.0 && on.steady_max_scheduled >= 3);
        CHECK(on.steady_late_drop_pct < 5.0);
        return;
    }

    if (only == "real-only-admission") {
        // Match the RT-on baseline's ~13 ms render period at latency 1, with FG explicitly disabled.
        const SmokeRunResult off = run_leg("off-real-only-admission", false, 5u, 0, 0, 180, 13000, 7000);
        CHECK(off.present_errors == 0 && off.session_failures == 0);
        CHECK(off.sched == 0 && off.real_sub >= 175);
        CHECK(off.steady_base_fps >= 69.0);
        CHECK(off.queue_peak == 0 && off.queue_waits == 0);
        return;
    }

    if (admission_only) {
        char mode[16] = {};
        const bool use_consumption = GetEnvironmentVariableA("WG_FG_ADMISSION", mode, sizeof(mode)) == 8 &&
                                     std::strcmp(mode, "consumed") == 0;
        // A GPU-bound title waits at frame start, records for ~4 ms, and queues 7 ms of GPU work at Present.
        // Warm up at a slower cadence to put the first-image budget above the steady GPU period.
        const SmokeRunResult run = run_leg("admission-7ms", true, 1u, 8, 12000, 320, 7000, 4000);
        CHECK(run.present_errors == 0 && run.session_failures == 0);
        CHECK(run.sched >= 200);
        if (!use_consumption) {
            CHECK(run.queue_peak == 0 && run.queue_waits == 0);
            return;
        }
        CHECK(run.steady_late_drop_pct < 2.0);
        CHECK(run.steady_base_fps >= 1000.0 / (1.1 * 7.0));
        CHECK(run.queue_peak == 2);

        // Request 4x with a nominal 200 fps producer. The 349.4 Hz presenter must shed generated images.
        const SmokeRunResult display_bound = run_leg("4x-display-bound", true, 3u, 0, 0, 260, 5000, 4000);
        CHECK(display_bound.present_errors == 0 && display_bound.session_failures == 0);
        CHECK(display_bound.queue_peak <= 2);
        CHECK(display_bound.steady_max_scheduled < 3);
        return;
    }

    // 1. 2x (N=1) GPU-bound at 10.5 ms (~95 fps) with 8 slow 25 ms startup frames (L=1).
    const SmokeRunResult off_105 = run_leg("off-10.5ms", false, 1u, 0, 0, 70, 10500, 3500);
    const SmokeRunResult fg_2x = run_leg("2x-10.5ms-slowstart", true, 1u, 8, 25000, 240, 10500, 3500);
    CHECK(off_105.present_errors == 0 && off_105.steady_base_fps >= 70.0);
    CHECK(fg_2x.present_errors == 0 && fg_2x.session_failures == 0);
    CHECK(fg_2x.latency_cap_ro <= 2);
    CHECK(fg_2x.sched >= 200);
    CHECK(fg_2x.late_drop_pct < 2.0);
    CHECK(fg_2x.steady_base_fps >= off_105.steady_base_fps * 0.85);
    CHECK(fg_2x.submitted_mult >= 1.75);

    // 2. 4x (N=3) GPU-bound at 12.5 ms (~80 fps -> 320 Hz <= 349.4 Hz VRR ceiling) with slow startup frames (L=1).
    const SmokeRunResult off_125 = run_leg("off-12.5ms", false, 3u, 0, 0, 70, 12500, 3500);
    const SmokeRunResult fg_4x = run_leg("4x-12.5ms-slowstart", true, 3u, 8, 25000, 220, 12500, 3500);
    CHECK(off_125.present_errors == 0 && off_125.steady_base_fps >= 60.0);
    CHECK(fg_4x.present_errors == 0 && fg_4x.session_failures == 0);
    CHECK(fg_4x.latency_cap_ro <= 2);
    CHECK(fg_4x.sched >= 500);
    CHECK(fg_4x.late_drop_pct < 5.0);
    CHECK(fg_4x.steady_base_fps >= off_125.steady_base_fps * 0.85);
    CHECK(fg_4x.submitted_mult >= 3.20);

    // 3. 6x (N=5) GPU-bound at 18.5 ms (~54 fps -> 324 Hz <= 349.4 Hz VRR ceiling) with slow startup frames (L=1).
    const SmokeRunResult off_185 = run_leg("off-18.5ms", false, 5u, 0, 0, 50, 18500, 3500);
    const SmokeRunResult fg_6x = run_leg("6x-18.5ms-slowstart", true, 5u, 8, 25000, 160, 18500, 3500);
    CHECK(off_185.present_errors == 0 && off_185.steady_base_fps >= 42.0);
    CHECK(fg_6x.present_errors == 0 && fg_6x.session_failures == 0);
    CHECK(fg_6x.latency_cap_ro <= 2);
    CHECK(fg_6x.sched >= 600);
    CHECK(fg_6x.late_drop_pct < 5.0);
    CHECK(fg_6x.steady_base_fps >= off_185.steady_base_fps * 0.85);
    CHECK(fg_6x.submitted_mult >= 4.80);
}

void TestPacingPercentileKeepsFrameOrder() {
    g_case = "pacing-percentile";
    // These samples alternate between admission-delayed and active-GPU frames.
    const std::vector<int64_t> chronological = {80000, 8000, 60000, 10000, 70000};
    auto origins = chronological;
    CHECK(Percentile(origins, 0.90) == 70000);
    CHECK(origins == chronological);
}

void TestPacingSamplesRemainBounded() {
    g_case = "pacing-sample-capacity";
    PacingSamples<2> samples;
    for (const int64_t value : {100, 200, 1}) samples.Append(value);
    CHECK(samples.Size() == 2);
    if (samples.Size() == 2) CHECK(samples.Percentile(1.0) == 200);
}

void TestReadyBudgetCapAfterSpeedup() {
    g_case = "pacing-budget-cap";
    CHECK(FirstImageBudgetCap(250000, 1) == 285000);
    CHECK(FirstImageBudgetCap(70000, 1) == 105000);
    CHECK(FirstImageBudgetCap(70000, 2) == 175000);
    CHECK(FirstImageBudgetCap(10000, 1) == 80000);
    CHECK(FirstImageBudgetCap(0, 1) == 320000);
}

void TestFirstImageBudgetEasingFloor() {
    g_case = "pacing-budget-floor";
    CHECK(FirstImageBudgetFloor(40000, 5000, 70000) == 75000);
    CHECK(FirstImageBudgetFloor(40000, 80000, 70000) == 83000);
    CHECK(FirstImageBudgetFloor(40000, 5000, 0) == 40000);
}

void TestBudgetCapNeedsReliableSamplesAndPreservesFloors() {
    g_case = "pacing-budget-cap-confidence";
    FirstImageBudgetLimits limits;
    limits.Observe(6, 120000, 1);
    CHECK(limits.Cap(40000, 120000) == 155000);
    limits.Observe(5, 15000, 1);  // sparse fast samples during a warm-up transition
    CHECK(limits.Cap(40000, 120000) == 155000);
    limits.Observe(0, 0, 1);
    CHECK(limits.Cap(40000, 120000) == 155000);
    limits.RecordMiss(140000);
    limits.Observe(6, 15000, 1);  // enough fast samples, but a retained miss sets the usable floor
    CHECK(limits.miss_floor == 140000);
    CHECK(limits.Cap(40000, 120000) == 145000);
    CHECK(limits.Cap(40000, 158000) == 161000);

    // adm2-default-x2 warm-up: a sampled 8.0 ms cap, 11.4 ms p95 and a 15.5 ms running budget.
    FirstImageBudgetLimits warmup;
    warmup.Observe(6, 15000, 1);
    CHECK(warmup.Cap(40000, 114000) == 117000);
    CHECK(std::min<int64_t>(155000, warmup.Cap(40000, 114000, true)) == 155000);
}

void TestHighCountPresentBudgetEasesInSmallSteps() {
    g_case = "pacing-high-count-easing";
    CHECK(FirstImageFastEaseStepCap(194000, false, 5) == 10000);
    CHECK(FirstImageFastEaseStepCap(194000, true, 5) == 25000);
    CHECK(FirstImageFastEaseStepCap(194000, false, 3) == 25000);
    CHECK(FirstImageFastEaseStepCap(219000, false, 5) == 25000);
}

void TestBudgetControllerUsesProducerMissesAndResetsItsBaseline() {
    g_case = "pacing-producer-budget-window";
    wgfg::FgPresentStats base;
    base.generated_scheduled = 10;
    base.generated_late_pre_queue = 1;
    auto now = base;
    now.generated_scheduled += 100;
    now.generated_dropped_post_queue = 30;
    CHECK(!ProducerWindow(now, base).NeedsBudgetReview());
    CHECK(ProducerWindow(now, base).ReducedCount(5) == 0);
    now.generated_late_pre_queue += 6;
    CHECK(ProducerWindow(now, base).NeedsBudgetReview());
    CHECK(ProducerWindow(now, base).ReducedCount(2) == 1);
    CHECK(ProducerWindow(now, base).ReducedCount(1) == 0);

    FirstImageBudgetLimits limits;
    limits.Observe(6, 120000, 1);
    limits.RecordMiss(140000);
    ResetFirstImageBudgetWindow(limits, base, now);
    CHECK(limits.miss_floor == 0 && limits.sampled_cap == kFirstImageBudgetMax);
    CHECK(base.generated_scheduled == now.generated_scheduled && base.generated_late_pre_queue == now.generated_late_pre_queue);
    CHECK(base.generated_dropped_post_queue == now.generated_dropped_post_queue);
    CHECK(!ProducerWindow(now, base).NeedsBudgetReview() && ProducerWindow(now, base).ReducedCount(5) == 0);
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    bool gpu = false;
    std::string dxil, scratch = ".", only;
    int preset = -1;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--gpu") == 0) gpu = true;
        if (std::strcmp(argv[i], "--only") == 0 && i + 1 < argc) only = argv[++i];
        if (std::strcmp(argv[i], "--dxil") == 0 && i + 1 < argc) dxil = argv[++i];
        if (std::strcmp(argv[i], "--scratch") == 0 && i + 1 < argc) scratch = argv[++i];
        if (std::strcmp(argv[i], "--debug-layer") == 0) g_debug_layer = true;
        if (std::strcmp(argv[i], "--preset") == 0 && i + 1 < argc) preset = argv[i + 1][0] == 'C' ? 2 : 0, ++i;
    }
    TestIds();
    TestPrecheck(scratch);
    TestFlags();
    TestPanel(scratch);
    TestPacingPercentileKeepsFrameOrder();
    TestPacingSamplesRemainBounded();
    TestReadyBudgetCapAfterSpeedup();
    TestFirstImageBudgetEasingFloor();
    TestBudgetCapNeedsReliableSamplesAndPreservesFloors();
    TestHighCountPresentBudgetEasesInSmallSteps();
    TestBudgetControllerUsesProducerMissesAndResetsItsBaseline();
    std::printf("cpu: %d checks, %d failed\n", g_checks, g_failed);
    if (gpu) {
        g_case = "gpu-init";
        Gpu g;
        CHECK(!dxil.empty() && g.Init());
        if (!dxil.empty() && g.device) {
            char name[128] = {};
            WideCharToMultiByte(CP_UTF8, 0, g.desc.Description, -1, name, sizeof(name) - 1, nullptr, nullptr);
            std::printf("gpu: %s\n", name);
            if (!FgIsRdna4(g.desc.VendorId, g.desc.DeviceId)) {
                std::printf("  not an RDNA 4 adapter: the host's route is never taken here (GPU leg skipped)\n");
            } else {
                if (only.empty() || only == "facade") {
                    g_case = "gpu-facade-formats";
                    TestFacadeFormats(g, dxil);
                    TestFacadeCopyRetryAndReuse(g);
                }
                if ((only.empty() || only == "host") && (preset < 0 || preset == 2)) {
                    g_case = "gpu-host-lite-2x";
                    RunHost(g, dxil, 2);
                }
                if ((only.empty() || only == "host") && (preset < 0 || preset == 0)) {
                    g_case = "gpu-host-full-a-2x";
                    RunHost(g, dxil, 0);
                }
                if (preset < 0 && (only.empty() || only == "latency" || only == "cadence")) {
                    g_case = "gpu-rc2-latency-and-pacing";
                    TestRc2LatencyAndPacing(g, dxil, only.empty() || only == "latency",
                                           only.empty() || only == "cadence");
                }
                if (preset < 0 && (only.empty() || only == "smoke" || only == "admission" ||
                                   only == "real-only-admission" || only == "consumed-6x" ||
                                   only == "admission-policy" || only == "admission-switch")) {
                    g_case = "gpu-smoke-pacing";
                    TestSmokeGpuBoundPacing(g, dxil, only);
                }
            }
            if (g_debug_layer) {
                std::printf("debug layer errors in GPU tests: %lld\n", static_cast<long long>(g.debug_errors_seen));
                CHECK(g.info && g.debug_errors_seen == 0);
            }
        }
    }
    std::printf("fg_asi_session: %d checks, %d failed\n", g_checks, g_failed);
    return g_failed ? 1 : 0;
}
