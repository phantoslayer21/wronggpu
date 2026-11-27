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

#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "../../runtime/fg_d3d12_executor.h"
#include "../fg_proxy_core.h"
#include "../../presenter/fg_presenter_factory.h"
#include "../win32/fg_asi_session.h"
#include "../win32/fg_asi_transport.h"
#include "../win32/fg_panel.h"
#include "../win32/fg_swapchain_hook.h"
#include "wg_state.h"

using Microsoft::WRL::ComPtr;
using namespace wgfgp;
using namespace wgfgp::win32;

namespace {

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

int g_log_lines = 0;
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
        return errors;
    }
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
        Sleep(33);
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
int g_cards = 0, g_tabs = 0;
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
    CHECK(panel::StatusLine(st, cur) == "Generating: 2x, preset A, DXGI flip, tearing (the game asks for fewer images)");
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
    fakehost::g_force = 2;
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

int main(int argc, char** argv) {
    bool gpu = false;
    std::string dxil, scratch = ".";
    int preset = -1;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--gpu") == 0) gpu = true;
        if (std::strcmp(argv[i], "--dxil") == 0 && i + 1 < argc) dxil = argv[++i];
        if (std::strcmp(argv[i], "--scratch") == 0 && i + 1 < argc) scratch = argv[++i];
        if (std::strcmp(argv[i], "--debug-layer") == 0) g_debug_layer = true;
        if (std::strcmp(argv[i], "--preset") == 0 && i + 1 < argc) preset = argv[i + 1][0] == 'C' ? 2 : 0, ++i;
    }
    TestIds();
    TestPrecheck(scratch);
    TestFlags();
    TestPanel(scratch);
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
                g_case = "gpu-facade-formats";
                TestFacadeFormats(g, dxil);
                if (preset < 0 || preset == 2) {
                    g_case = "gpu-host-lite-2x";
                    RunHost(g, dxil, 2);
                }
                if (preset < 0 || preset == 0) {
                    g_case = "gpu-host-full-a-2x";
                    RunHost(g, dxil, 0);
                }
            }
        }
    }
    std::printf("fg_asi_session: %d checks, %d failed\n", g_checks, g_failed);
    return g_failed ? 1 : 0;
}
