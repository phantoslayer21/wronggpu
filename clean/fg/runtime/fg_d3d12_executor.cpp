// SPDX-License-Identifier: MIT
// Frame generation (issue #96, F7b): the production D3D12 executor. See fg_d3d12_executor.h.
#include "fg_d3d12_executor.h"

#include <d3d12.h>
#include <dxgi1_6.h>
#include <windows.h>
#include <wrl/client.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iterator>
#include <unordered_map>

#include "fg_frame_dump.h"

#include "../presenter/fg_dxgi_presenter.h"
#include "../presenter/fg_win_comp_presenter.h"

using Microsoft::WRL::ComPtr;

namespace wgfg {

namespace {

// ---- kernels -----------------------------------------------------------------------------------------------------

enum Kernel : uint32_t {
    kFill16, kCopy4, kDetect, kMenuOutput, kDown2, kDown4, kMvec, kScatter, kUnpack, kInpaint, kPull, kPush, kFine,
    kHoleMask, kQualityInit, kNet1Input, kNet1Output, kInterm, kWeights, kLiteBlend,
    kCapture, kPack,
    kMConv1, kMConv2, kMConv4, kMPairB, kMPairD4, kMPairD3, kMPairD2, kMPairD1, kMPairD0,
    kFullBlend, kOutPull, kOutPullMid, kOutPush, kOutPushFine,
    kNet2Conv, kNet2Up2, kNet2Merge, kNet2Warp,
    kN2m1b1, kN2m1b2, kN2m2b1, kN2m2b2, kN2m2b4, kN2m1b4s, kN2m1b2s, kN2m1b1u, kN2m1b2u, kN2m2b4u, kN2m3b2u,
    kN2m3b4u, kNet2Upwarp,
    kMConv2Narrow, kMConv1Late, kMPairBFirst, kMPairBSecond, kMPairD4First, kMPairD4Second,
    kMPairD3Narrow, kMPairD2Narrow,
    kDown2Srv, kDown4Srv, kMvecSrv, kIntermTiled, kPackPairs, kPullTiled, kPushTail, kDown2HudTile,
    kMConv2Mid, kN2h3b1u, kNet2MergePixels, kOutPullPack, kOutPushFinePack,
    kLiteBlendPack,
    kKernelCount,
};
constexpr uint32_t kFirstFull = kFullBlend;
constexpr uint32_t kFirstOptional = kMConv2Narrow;

// The fast net2 convolution variants (clean/dx12/kernels/fg_net2_mconv.hlsl declarations, in the order
// clean/fg/runner/fg_net2_mma.py VARIANTS lists them): output tiles per wave, 16-channel blocks per tap, upsample
// staging, lanes per workgroup.
struct MconvVariant {
    uint32_t kernel, mt, nb, up, lanes;
};
constexpr MconvVariant kMconv[] = {
    {kN2m1b1, 1, 1, 0, 256}, {kN2m1b2, 1, 2, 0, 256}, {kN2m2b1, 2, 1, 0, 256}, {kN2m2b2, 2, 2, 0, 256},
    {kN2m2b4, 2, 4, 0, 256}, {kN2m1b4s, 1, 4, 0, 128}, {kN2m1b2s, 1, 2, 0, 128}, {kN2m1b1u, 1, 1, 1, 256},
    {kN2m1b2u, 1, 2, 1, 256}, {kN2m2b4u, 2, 4, 1, 128}, {kN2m3b2u, 3, 2, 1, 256}, {kN2m3b4u, 3, 4, 1, 128},
};
constexpr uint32_t kSmallGrid = 16384;  // conv grids of at most this many pixels take a 128-lane one-tile variant
constexpr uint32_t kMTile = 16;

bool KernelNeeded(uint32_t k, bool full, FgNet2Path path) {
    if (k < kFirstFull || k >= kFirstOptional) return true;
    if (!full) return false;
    if (k == kNet2Conv || k == kNet2Up2 || k == kNet2Warp) return path == FgNet2Path::Exact;
    if (k >= kN2m1b1 && k <= kNet2Upwarp) return path == FgNet2Path::Fast;
    return true;
}

// fg_net2_mma.py variant(): (variant, weight tile groups, conv rows per workgroup); false when none fits.
bool ChooseMconv(uint32_t cout, uint32_t cw, uint32_t up, uint64_t pixels, MconvVariant* v, uint32_t* groups,
                 uint32_t* rows) {
    const uint32_t nb = (cw + 15) / 16, tiles = (cout + 15) / 16;
    if (nb != 1 && nb != 2 && nb != 4) return false;
    const MconvVariant* narrow = nullptr;
    const MconvVariant* big = nullptr;
    for (const MconvVariant& c : kMconv) {
        if (c.nb != nb || c.up != up || tiles % c.mt != 0) continue;
        if (c.lanes == 128 && c.mt == 1 && !narrow) narrow = &c;
        if ((up || c.lanes == 256) && (!big || c.mt > big->mt)) big = &c;  // first of the largest mt (stable)
    }
    const MconvVariant* pick = (!up && pixels <= kSmallGrid && narrow) ? narrow : big;
    if (!pick) return false;
    *v = *pick;
    *groups = tiles / pick->mt;
    *rows = 2 * pick->lanes / 32;
    return true;
}

const char* const kKernelNames[kKernelCount] = {
    "fg_lite_fill16", "fg_lite_copy4", "fg_lite_detect", "fg_lite_menu_output", "fg_lite_downsample2",
    "fg_lite_downsample4", "fg_lite_mvec", "fg_lite_p2c_scatter", "fg_lite_p2c_unpack", "fg_lite_p2c_inpaint",
    "fg_lite_p2c_pull", "fg_lite_p2c_push", "fg_lite_p2c_fine", "fg_lite_hole_mask", "fg_lite_quality_init",
    "fg_lite_net1_input", "fg_lite_net1_output", "fg_lite_interm", "fg_lite_weights", "fg_lite_blend",
    "fg_exec_capture", "fg_exec_pack",
    "fg_mconv1", "fg_mconv2", "fg_mconv4", "fg_mpair_b", "fg_mpair_d4", "fg_mpair_d3", "fg_mpair_d2", "fg_mpair_d1",
    "fg_mpair_d0",
    "fg_full_blend", "fg_out_pull", "fg_out_pull_mid", "fg_out_push", "fg_out_push_fine",
    "fg_net2_conv", "fg_net2_up2", "fg_net2_merge", "fg_net2_warp",
    "fg_net2_mconv_m1b1", "fg_net2_mconv_m1b2", "fg_net2_mconv_m2b1", "fg_net2_mconv_m2b2", "fg_net2_mconv_m2b4",
    "fg_net2_mconv_m1b4s", "fg_net2_mconv_m1b2s", "fg_net2_mconv_m1b1u", "fg_net2_mconv_m1b2u", "fg_net2_mconv_m2b4u",
    "fg_net2_mconv_m3b2u", "fg_net2_mconv_m3b4u", "fg_net2_upwarp",
    "fg_mconv2_narrow", "fg_mconv1_late", "fg_mpair_b_first", "fg_mpair_b_second",
    "fg_mpair_d4_first", "fg_mpair_d4_second", "fg_mpair_d3_narrow", "fg_mpair_d2_narrow",
    "fg_lite_downsample2_srv", "fg_lite_downsample4_srv", "fg_lite_mvec_srv", "fg_lite_interm_tiled", "fg_exec_pack_pairs",
    "fg_lite_p2c_pull_tiled", "fg_lite_p2c_push_tail", "fg_lite_downsample2_srv_hud_tile",
    "fg_mconv2_mid", "fg_net2_mconv_h3b1u", "fg_net2_merge_pixels",
    "fg_out_pull_pack", "fg_out_push_fine_pack", "fg_lite_blend_pack",
};

constexpr uint32_t kRootConstants = 59;  // 2 runtime constants + up to 57 argument DWORDs
constexpr uint32_t kMaxArgs = kRootConstants - 2;
constexpr uint32_t kChannelSpace = 0x7FFF0ADE;
constexpr uint32_t kTables = 8;          // immutable descriptor tables per command recording
constexpr uint32_t kInputSrvs = 4;
constexpr uint32_t kTableSize = 1 + kInputSrvs;  // channel UAV followed by t0..t3
constexpr uint32_t kNoClamp = 1u << 20;
constexpr uint32_t kHalfOne = 0x3C00, kHalfTwo = 0x4000;

uint64_t align(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }
uint32_t cdiv(uint64_t a, uint64_t b) { return static_cast<uint32_t>((a + b - 1) / b); }
uint32_t groups_for(uint64_t threads, uint32_t per_lane = 1) {
    const uint32_t g = cdiv(threads, 256ull * per_lane);
    return g ? g : 1;
}
uint32_t fb(float f) { return FgExecFloatBits(f); }

bool read_file(const std::string& path, std::vector<char>* out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    out->assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    return true;
}

bool dir_exists(const std::string& p) {
    const DWORD a = GetFileAttributesA(p.c_str());
    return a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_DIRECTORY);
}

// The view format of a game texture for the capture's Load (depth and typeless formats as their readable view).
DXGI_FORMAT view_format(DXGI_FORMAT f, uint32_t* channels) {
    *channels = 4;
    switch (f) {
        case DXGI_FORMAT_R16G16B16A16_TYPELESS: return DXGI_FORMAT_R16G16B16A16_FLOAT;
        case DXGI_FORMAT_R16G16B16A16_FLOAT: case DXGI_FORMAT_R16G16B16A16_UNORM: case DXGI_FORMAT_R32G32B32A32_FLOAT:
        case DXGI_FORMAT_R8G8B8A8_UNORM: case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: case DXGI_FORMAT_B8G8R8A8_UNORM:
        case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: case DXGI_FORMAT_R10G10B10A2_UNORM: case DXGI_FORMAT_R11G11B10_FLOAT:
            return f;
        case DXGI_FORMAT_R8G8B8A8_TYPELESS: return DXGI_FORMAT_R8G8B8A8_UNORM;
        case DXGI_FORMAT_B8G8R8A8_TYPELESS: return DXGI_FORMAT_B8G8R8A8_UNORM;
        case DXGI_FORMAT_R10G10B10A2_TYPELESS: return DXGI_FORMAT_R10G10B10A2_UNORM;
        case DXGI_FORMAT_R16G16_FLOAT: case DXGI_FORMAT_R32G32_FLOAT: case DXGI_FORMAT_R16G16_SNORM:
        case DXGI_FORMAT_R16G16_UNORM:
            *channels = 2;
            return f;
        case DXGI_FORMAT_R16G16_TYPELESS: *channels = 2; return DXGI_FORMAT_R16G16_FLOAT;
        case DXGI_FORMAT_R32G32_TYPELESS: *channels = 2; return DXGI_FORMAT_R32G32_FLOAT;
        case DXGI_FORMAT_R32_FLOAT: case DXGI_FORMAT_R16_FLOAT: case DXGI_FORMAT_R16_UNORM: case DXGI_FORMAT_R8_UNORM:
            *channels = 1;
            return f;
        case DXGI_FORMAT_R32_TYPELESS: case DXGI_FORMAT_D32_FLOAT: *channels = 1; return DXGI_FORMAT_R32_FLOAT;
        case DXGI_FORMAT_R16_TYPELESS: case DXGI_FORMAT_D16_UNORM: *channels = 1; return DXGI_FORMAT_R16_UNORM;
        case DXGI_FORMAT_R24G8_TYPELESS: case DXGI_FORMAT_D24_UNORM_S8_UINT:
            *channels = 1;
            return DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
        case DXGI_FORMAT_R32G8X24_TYPELESS: case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
            *channels = 1;
            return DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
        default: break;
    }
    *channels = 0;
    return DXGI_FORMAT_UNKNOWN;
}

// The motion input reads x and y from the first two channels.
constexpr uint32_t kMotionMinChannels = 2;

// fg_exec_pack's format code of a slot format, -1 when unsupported.
int pack_code(DXGI_FORMAT f, uint32_t* bpp) {
    switch (f) {
        case DXGI_FORMAT_R16G16B16A16_FLOAT: *bpp = 8; return 0;
        case DXGI_FORMAT_R8G8B8A8_UNORM: case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: *bpp = 4; return 1;
        case DXGI_FORMAT_B8G8R8A8_UNORM: case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: *bpp = 4; return 2;
        case DXGI_FORMAT_R10G10B10A2_UNORM: *bpp = 4; return 3;
        default: break;
    }
    *bpp = 0;
    return -1;
}

DXGI_FORMAT linear_copy_format(DXGI_FORMAT f) {
    switch (f) {
        case DXGI_FORMAT_R8G8B8A8_TYPELESS: case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
            return DXGI_FORMAT_R8G8B8A8_UNORM;
        case DXGI_FORMAT_B8G8R8A8_TYPELESS: case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
            return DXGI_FORMAT_B8G8R8A8_UNORM;
        case DXGI_FORMAT_R10G10B10A2_TYPELESS:
            return DXGI_FORMAT_R10G10B10A2_UNORM;
        default: return f;
    }
}

// ---- weights -----------------------------------------------------------------------------------------------------

struct Rng {
    uint64_t s;
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    float uniform(float lo, float hi) {
        const double u = static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0);
        return static_cast<float>(lo + (hi - lo) * u);
    }
};

void put16(std::vector<uint8_t>* v, size_t i, uint16_t h) {
    (*v)[2 * i] = static_cast<uint8_t>(h & 0xFF);
    (*v)[2 * i + 1] = static_cast<uint8_t>(h >> 8);
}

float ftz(float x) { return std::fabs(x) < 1.17549435e-38f ? std::copysign(0.0f, x) : x; }

}  // namespace

const char* FgExecRefusalName(FgExecRefusal r) {
    switch (r) {
        case FgExecRefusal::None: return "none";
        case FgExecRefusal::NullDevice: return "null device";
        case FgExecRefusal::NullQueue: return "null queue";
        case FgExecRefusal::DxilDirMissing: return "kernel folder missing";
        case FgExecRefusal::DxilMissing: return "kernel file missing";
        case FgExecRefusal::DxilInvalid: return "kernel file invalid";
        case FgExecRefusal::RootSignature: return "root signature";
        case FgExecRefusal::PipelineFailed: return "pipeline creation failed";
        case FgExecRefusal::WeightsMissing: return "weights missing";
        case FgExecRefusal::WeightsInvalid: return "weights invalid";
        case FgExecRefusal::SizeUnsupported: return "frame size unsupported";
        case FgExecRefusal::AllocationFailed: return "allocation failed";
        case FgExecRefusal::FenceFailed: return "fence";
        case FgExecRefusal::NotConfigured: return "not configured";
        case FgExecRefusal::SlotTargetMissing: return "no presenter slot target";
        case FgExecRefusal::SlotMissing: return "slot missing";
        case FgExecRefusal::SlotFormat: return "slot format or size";
        case FgExecRefusal::InputMissing: return "required input missing";
        case FgExecRefusal::InputFormat: return "input format";
        case FgExecRefusal::InputExtent: return "input extent";
        case FgExecRefusal::PresetUnsupported: return "preset unsupported";
        case FgExecRefusal::DistortionUnsupported: return "distortion field not modelled";
        case FgExecRefusal::CommandFailed: return "command recording failed";
        case FgExecRefusal::WaitTimeout: return "wait timeout";
        case FgExecRefusal::DeviceLost: return "device lost";
        case FgExecRefusal::BadArgument: return "bad argument";
        case FgExecRefusal::Net1Refused: return "net1 recorder refused";
    }
    return "?";
}

bool FgExecWeightsFromModel(const uint8_t* bytes, size_t size, const FgModelReport& report, FgExecWeights* out,
                            std::string* why) {
    if (!bytes || !out || report.refusal != FgRefusal::None || report.stages.size() != 17 + kFgNet2Layers) {
        if (why) *why = "model report not validated or wrong stage count";
        return false;
    }
    FgExecWeights w;
    for (uint32_t i = 0; i < 17; ++i) {
        if (!FgPackNet1Layer(bytes, size, report.stages[i], FgNet1Layers()[i], &w.net1_fragments[i],
                             &w.net1_bias[i])) {
            if (why) *why = "net1 stage " + report.stages[i].name + " does not fit";
            return false;
        }
    }
    for (uint32_t i = 0; i < kFgNet2Layers; ++i) {
        const FgModelStage& st = report.stages[17 + i];
        const FgNet2Layer& l = FgNet2Layers()[i];
        const uint32_t used = st.w.shape[1];
        if (st.w.rank != 4 || st.w.shape[0] != l.cout || used != l.cin_used || st.w.shape[2] != 3 ||
            st.w.shape[3] != 3 || st.b.rank != 1 || st.b.shape[0] != l.cout ||
            st.w.bytes != 4ull * l.cout * used * 9 || st.b.bytes != 4ull * l.cout || st.w.offset > size ||
            st.w.bytes > size - st.w.offset || st.b.offset > size || st.b.bytes > size - st.b.offset) {
            if (why) *why = "net2 stage " + st.name + " does not fit";
            return false;
        }
        w.net2_weights[i].assign(2ull * l.cout * 9 * l.cin, 0);
        for (uint32_t o = 0; o < l.cout; ++o)
            for (uint32_t c = 0; c < used; ++c)
                for (uint32_t t = 0; t < 9; ++t) {
                    float f;
                    std::memcpy(&f, bytes + st.w.offset + 4ull * ((static_cast<uint64_t>(o) * used + c) * 9 + t), 4);
                    uint16_t h = 0;
                    if (!FgFloatToHalfExact(f, &h)) {
                        if (why) *why = "net2 stage " + st.name + " value not exact in float16";
                        return false;
                    }
                    put16(&w.net2_weights[i], (static_cast<size_t>(o) * 9 + t) * l.cin + c, h);
                }
        w.net2_bias[i].assign(2ull * l.cout, 0);
        for (uint32_t o = 0; o < l.cout; ++o) {
            float f;
            std::memcpy(&f, bytes + st.b.offset + 4ull * o, 4);
            uint16_t h = 0;
            if (!FgFloatToHalfExact(f, &h)) {
                if (why) *why = "net2 stage " + st.name + " bias not exact in float16";
                return false;
            }
            put16(&w.net2_bias[i], o, h);
        }
    }
    w.has_net2 = true;
    *out = std::move(w);
    return true;
}

void FgExecSyntheticWeights(uint32_t seed, FgExecWeights* out) {
    Rng rng{0x5EEDull * 1000003ull + seed};
    FgExecWeights w;
    for (uint32_t i = 0; i < 17; ++i) {
        const FgNet1Layer& l = FgNet1Layers()[i];
        const size_t n = static_cast<size_t>(l.cout) * l.k * l.k * l.cin;
        const float s = 1.5f / std::sqrt(static_cast<float>(l.cin * l.k * l.k));
        std::vector<uint16_t> canon(n);
        for (size_t j = 0; j < n; ++j) canon[j] = FgExecHalfBits(rng.uniform(-s, s));
        FgSwizzleNet1(canon.data(), l.cout, l.k, l.cin, l.order, &w.net1_fragments[i]);
        w.net1_bias[i].assign(2ull * l.cout, 0);
        for (uint32_t o = 0; o < l.cout; ++o) put16(&w.net1_bias[i], o, FgExecHalfBits(rng.uniform(-0.1f, 0.1f)));
    }
    for (uint32_t i = 0; i < kFgNet2Layers; ++i) {
        const FgNet2Layer& l = FgNet2Layers()[i];
        const float s = 2.5f / std::sqrt(9.0f * static_cast<float>(l.cin_used));
        w.net2_weights[i].assign(2ull * l.cout * 9 * l.cin, 0);
        for (uint32_t o = 0; o < l.cout; ++o)
            for (uint32_t t = 0; t < 9; ++t)
                for (uint32_t c = 0; c < l.cin; ++c)
                    put16(&w.net2_weights[i], (static_cast<size_t>(o) * 9 + t) * l.cin + c,
                          c < l.cin_used ? FgExecHalfBits(rng.uniform(-s, s)) : 0);
        w.net2_bias[i].assign(2ull * l.cout, 0);
        for (uint32_t o = 0; o < l.cout; ++o) put16(&w.net2_bias[i], o, FgExecHalfBits(rng.uniform(-1.0f, 1.0f)));
    }
    w.has_net2 = true;
    *out = std::move(w);
}

void FgExecSigTable(std::vector<float>* out) {
    out->resize(65536);
    float neg_log2e;
    const uint32_t nb = 0xBFB8AA3Bu;
    std::memcpy(&neg_log2e, &nb, 4);
    for (uint32_t b = 0; b < 65536; ++b) {
        const float z = FgExecHalfToFloat(static_cast<uint16_t>(b));
        const volatile float m = ftz(ftz(z) * neg_log2e);
        const float e = ftz(static_cast<float>(std::exp2(static_cast<double>(m))));
        const volatile float s = ftz(e + 1.0f);
        (*out)[b] = ftz(static_cast<float>(1.0 / static_cast<double>(s)));
    }
}

// ---- net2, fast path: plan and weights ---------------------------------------------------------------------------

const char* FgNet2PathName(FgNet2Path p) {
    switch (p) {
        case FgNet2Path::Fast: return "fast";
        case FgNet2Path::Exact: return "exact";
    }
    return "?";
}

bool FgD3d12MotionFormatAccepted(uint32_t dxgi_format) {
    uint32_t channels = 0;
    view_format(static_cast<DXGI_FORMAT>(dxgi_format), &channels);
    return channels >= kMotionMinChannels;
}

std::vector<std::string> FgD3d12ExecutorKernelNames(bool enable_full, FgNet2Path net2) {
    std::vector<std::string> names;
    for (uint32_t k = 0; k < kFirstOptional; ++k)
        if (KernelNeeded(k, enable_full, net2)) names.emplace_back(kKernelNames[k]);
    return names;
}

bool FgNet2FastPlan(uint32_t W2, uint32_t H2, std::vector<FgNet2FastLaunch>* out) {
    if (!out || W2 == 0 || H2 == 0 || W2 % 16 || H2 % 16) return false;
    out->clear();
    const FgNet2Layer* L = FgNet2Layers();
    struct Step {
        uint32_t n, div, up, heads;  // conv grid = net2 grid / div
    };
    // fg_net2_mma.py fast_chain_case: conv grids (pooling layers: the input grid; upsampling layers: the output grid)
    const Step steps[] = {{1, 1, 0, 0},  {2, 2, 0, 0},  {3, 4, 0, 0},  {4, 8, 0, 0},  {5, 8, 0, 0},  {6, 8, 0, 0},
                          {7, 8, 0, 0},  {8, 8, 0, 0},  {9, 8, 0, 0},  {10, 8, 0, 0}, {11, 8, 0, 0}, {12, 4, 1, 0},
                          {13, 2, 1, 1}, {14, 1, 0, 0}, {15, 2, 0, 0}, {16, 4, 0, 0}, {17, 4, 0, 0}, {18, 4, 0, 0},
                          {19, 4, 0, 0}, {20, 4, 0, 0}, {21, 4, 0, 0}, {22, 4, 0, 0}, {23, 4, 0, 0}, {24, 2, 1, 0},
                          {25, 1, 1, 1}};
    FgNet2FastLaunch m;
    m.layer = 0;
    m.kernel = kKernelNames[kNet2Merge];
    m.groups = groups_for(10ull * W2 * H2);
    m.H = H2;
    m.W = W2;
    out->push_back(m);
    for (const Step& s : steps) {
        const FgNet2Layer& l = L[s.n - 1];
        FgNet2FastLaunch x;
        x.layer = s.n;
        x.H = H2 / s.div;
        x.W = W2 / s.div;
        MconvVariant v{};
        uint32_t groups = 0, rows = 0;
        if (!ChooseMconv(l.cout, l.cin, s.up, static_cast<uint64_t>(x.H) * x.W, &v, &groups, &rows)) return false;
        if (s.heads) groups = 3;
        x.kernel = kKernelNames[v.kernel];
        x.groups = cdiv(x.W, kMTile) * cdiv(x.H, rows) * groups;
        out->push_back(x);
        if (s.n == 13) {
            FgNet2FastLaunch u;
            u.layer = 26;
            u.kernel = kKernelNames[kNet2Upwarp];
            u.groups = std::max<uint32_t>(1, cdiv(static_cast<uint64_t>(W2) * H2, 256));
            u.H = H2;
            u.W = W2;
            out->push_back(u);
        }
    }
    return true;
}

bool FgNet2SwizzleFast(const std::vector<uint8_t>& canonical, const std::vector<uint8_t>& bias, uint32_t cout,
                       uint32_t cw, std::vector<uint8_t>* fragments, std::vector<uint8_t>* bias_padded) {
    if (!fragments || !bias_padded || cout == 0 || cw == 0) return false;
    const uint32_t nb = (cw + 15) / 16, tiles = (cout + 15) / 16;
    if ((nb != 1 && nb != 2 && nb != 4) || canonical.size() != 2ull * cout * 9 * cw || bias.size() != 2ull * cout)
        return false;
    auto get = [&canonical](size_t i) -> uint16_t {
        return static_cast<uint16_t>(canonical[2 * i] | (canonical[2 * i + 1] << 8));
    };
    fragments->assign(2ull * tiles * 9 * nb * 2 * 16 * 8, 0);
    // [mt][t][cb][h][m][j] = w[16 mt + m][t][16 cb + 8 h + j]
    size_t at = 0;
    for (uint32_t mt = 0; mt < tiles; ++mt)
        for (uint32_t t = 0; t < 9; ++t)
            for (uint32_t cb = 0; cb < nb; ++cb)
                for (uint32_t h = 0; h < 2; ++h)
                    for (uint32_t mm = 0; mm < 16; ++mm)
                        for (uint32_t j = 0; j < 8; ++j, ++at) {
                            const uint32_t o = 16 * mt + mm, c = 16 * cb + 8 * h + j;
                            if (o >= cout || c >= cw) continue;
                            const uint16_t v = get((static_cast<size_t>(o) * 9 + t) * cw + c);
                            (*fragments)[2 * at] = static_cast<uint8_t>(v & 0xFF);
                            (*fragments)[2 * at + 1] = static_cast<uint8_t>(v >> 8);
                        }
    bias_padded->assign(2ull * 16 * tiles, 0);
    std::memcpy(bias_padded->data(), bias.data(), bias.size());
    return true;
}

// ---- the executor ------------------------------------------------------------------------------------------------

namespace {

struct Entry {
    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> list;
    uint64_t fence = 0;
    uint32_t desc_base = 0;
    uint32_t query_base = 0, stage_count = 0;
    FgExecStage stages[static_cast<uint32_t>(FgExecStage::Count)] = {};
};

constexpr uint32_t kTimingQueries = static_cast<uint32_t>(FgExecStage::Count) + 1;

struct TimingQueries {
    ComPtr<ID3D12QueryHeap> heap;
    ComPtr<ID3D12Resource> readback;
    uint64_t hz = 0;
};

struct InputTex {
    ID3D12Resource* res = nullptr;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    DXGI_FORMAT view = DXGI_FORMAT_UNKNOWN;
    uint32_t channels = 0;
    uint32_t w = 0, h = 0;
    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
    FgExecRegion region;
};

// The per-game-frame values the per-index stages read.
struct FrameCtx {
    bool valid = false;
    bool colour_captured = false;
    uint64_t token = 0;
    FgPreset preset = FgPreset::C;
    bool lite = true;
    bool hud = false, ui4 = false, ui1 = false;
    InputTex colour, depth, motion, hudless, alpha;
    uint32_t alpha_kind = 4, alpha_channel = 3;
    bool inverted = false;
    float scx = 0, scy = 0;
    float clip_to_prev[16] = {}, prev_to_clip[16] = {};
    uint32_t disable = 0;
};

struct Rec {
    Entry* e = nullptr;
    ID3D12GraphicsCommandList* cl = nullptr;
    uint32_t table = 0;
    D3D12_RESOURCE_STATES arena = D3D12_RESOURCE_STATE_COMMON;
    bool dirty = false;  // a dispatch or copy wrote since the last UAV barrier
    bool open = false, direct = false, uses_arena = true;
    ID3D12Resource* written_slot = nullptr;

    ~Rec() {
        if (open) cl->Close();
    }
};

struct Net2Buf {
    uint32_t off = 0, h = 0, w = 0, c = 0;
};

}  // namespace

struct FgD3d12Executor::Impl final : public IFgComputeRecorder {
    FgD3d12ExecutorDesc desc;
    FgExecWeights weights;
    bool have_weights = false;
    FgExecRefusal refusal = FgExecRefusal::None;
    std::string why;
    FgExecCounters counters;

    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandQueue> compute_queue;
    D3D12_COMMAND_LIST_TYPE list_type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> pso[kKernelCount];
    std::vector<char> code[kKernelCount];
    bool pipelines_built = false;
    ComPtr<ID3D12Fence> fence;
    HANDLE event = nullptr;
    uint64_t next_value = 0;
    ComPtr<ID3D12Fence> handoff_fence;
    uint64_t handoff_value = 0, joined_handoff = 0, arena_value = 0;
    std::unordered_map<ID3D12Resource*, uint64_t> compute_slot_values;
    std::vector<Entry> entries;
    std::vector<Entry> direct_entries;
    ComPtr<ID3D12DescriptorHeap> heap;
    uint32_t desc_inc = 0;
    ComPtr<ID3D12Resource> channel;
    ComPtr<ID3D12Resource> arena;
    uint64_t arena_bytes = 0;
    ComPtr<ID3D12Resource> menu_rb;
    TimingQueries producer_queries, direct_queries;
    FgExecTimings timings;
    bool timing_failed = false;

    IFgSlotTarget* target = nullptr;
    uint32_t fence_id = 0;
    bool configured = false;

    FgCreateDesc create;
    FgExecGeometry g;
    FgExecHost host;
    FrameCtx frame;
    uint32_t cap_w = 0, cap_h = 0;

    // arena layout
    uint64_t bump = 0;
    uint32_t role_off[kRoleCount] = {};
    uint32_t ps_off[12] = {};
    uint32_t detres_off = 0, dis_off = 0;
    uint32_t cap_colour = 0, cap_hud = 0, cap_ui4 = 0, cap_ui1 = 0, cap_depth = 0, cap_motion = 0;
    uint32_t out_off = 0, out_stride = 0;       // generated image (float16 x 4), stride in texels
    uint32_t pack_off = 0, pack_pitch = 0;      // packed slot rows
    uint32_t fuse_pack = 0;  // slot format code (1..3) the output pyramid packs directly; 0: fg_exec_pack runs
    FgNet1Plan n1plan;
    FgArenaBinding n1arena;
    uint32_t n1_cells = 0, n1_out = 0, n1_pair_scratch = 0;
    // Full path
    uint32_t x0 = 0, x1 = 0, xw = 0, ib = 0, ic = 0, ia = 0, fine = 0, sig = 0, lv[9] = {};
    uint32_t n2w[kFgNet2Layers] = {}, n2b[kFgNet2Layers] = {};
    uint32_t n2fw[kFgNet2Layers] = {}, n2fb[kFgNet2Layers] = {};   // fast path: A fragments, padded biases
    std::vector<uint8_t> n2frag[kFgNet2Layers], n2fbias[kFgNet2Layers];
    bool fast() const { return desc.enable_full && desc.net2 == FgNet2Path::Fast; }
    Net2Buf nb_x, nb_a1, nb_a2, nb_r0, nb_t[4], nb_r[4], nb_r4u, nb_g, nb_gu, nb_hA4, nb_hA1, nb_hA3, nb_U4, nb_U1,
        nb_U3, nb_y, nb_b1, nb_s0, nb_tb[4], nb_s[4], nb_s4u, nb_e, nb_eu, nb_out;
    std::vector<uint8_t> weight_image;  // host copy of the weights region
    uint32_t weight_base = 0;

    std::vector<FgExecOutputRecord> outputs;
    struct PendingRb {
        FgExecOutputRecord rec;
        ComPtr<ID3D12Resource> image, colour, slot;
    };
    std::vector<PendingRb> pending;

    // developer frame dumps (FgD3d12ExecutorDesc::dump_dir)
    struct DumpRb {
        ComPtr<ID3D12Resource> buf;
        uint64_t fence_value = 0;  // 0 until the recording that copies it is submitted
        bool direct = false;
        uint32_t w = 0, h = 0, pitch = 0, format = 0;
        std::string name;
    };
    std::vector<DumpRb> dumps;
    uint64_t dump_seen_token = 0, dump_token = 0, dump_generating = 0;
    bool dump_seen = false, dump_armed = false;
    uint32_t dump_frames = 0;

    void DumpLog(const std::string& line) const {
        if (desc.dump_log) desc.dump_log(desc.dump_log_ctx, line.c_str());
    }

    // Once per game frame: clear any previous dump arming; actual arming happens in RecordGeneratedIndex
    // when a real generated frame is recorded so every dump is a matched (gen_N_i, game_N) pair.
    void ArmDump(uint64_t token, bool generates) {
        if (desc.dump_dir.empty() || (dump_seen && token == dump_seen_token)) return;
        dump_seen = true;
        dump_seen_token = token;
        dump_armed = false;
        (void)generates;
    }

    void ArmDumpOnGenerated(uint64_t token) {
        if (desc.dump_dir.empty() || (dump_armed && dump_token == token)) return;
        if (dump_frames >= desc.dump_max || desc.dump_every == 0) return;
        if (++dump_generating % desc.dump_every != 0) return;
        dump_armed = true;
        dump_token = token;
        ++dump_frames;
    }

    // After the slot write (the slot in COPY_DEST): a copy of the slot into a readback buffer, back to COPY_DEST.
    void RecordDump(Rec& r, ID3D12Resource* res, const D3D12_PLACED_SUBRESOURCE_FOOTPRINT& fp, uint32_t bpp,
                    char kind, uint64_t token, uint32_t index) {
        if (!dump_armed || token != dump_token || (kind != 'g' && kind != 'r' && kind != 'm')) return;
        DumpRb d;
        d.direct = r.direct;
        d.w = fp.Footprint.Width;
        d.h = fp.Footprint.Height;
        d.format = static_cast<uint32_t>(fp.Footprint.Format);
        d.pitch = static_cast<uint32_t>(align(static_cast<uint64_t>(bpp) * d.w, 256));
        d.name = kind == 'r' ? "game_" + std::to_string(dump_frames) + ".ppm"
               : kind == 'm' ? "menu_" + std::to_string(dump_frames) + "_" + std::to_string(index) + ".ppm"
                             : "gen_" + std::to_string(dump_frames) + "_" + std::to_string(index) + ".ppm";
        d.buf = Buffer(static_cast<uint64_t>(d.pitch) * d.h, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE,
                       D3D12_RESOURCE_STATE_COPY_DEST);
        if (!d.buf) {
            DumpLog("fg dump: " + d.name + " not written (no readback buffer)");
            return;
        }
        Transition(r, res, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
        D3D12_TEXTURE_COPY_LOCATION s = {};
        s.pResource = res;
        s.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        D3D12_TEXTURE_COPY_LOCATION t = {};
        t.pResource = d.buf.Get();
        t.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        t.PlacedFootprint.Footprint = fp.Footprint;
        t.PlacedFootprint.Footprint.RowPitch = d.pitch;
        r.cl->CopyTextureRegion(&t, 0, 0, 0, &s, nullptr);
        Transition(r, res, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
        dumps.push_back(std::move(d));
    }

    // Writes every dump whose fence passed (wait: a bounded wait for each first). Never throws.
    void FlushDumps(bool wait) {
        if (dumps.empty() || !fence) return;
        try {
            for (size_t i = 0; i < dumps.size();) {
                DumpRb& d = dumps[i];
                ID3D12Fence* completion = d.direct ? handoff_fence.Get() : fence.Get();
                if (d.fence_value == 0 || (completion->GetCompletedValue() < d.fence_value &&
                                           !(wait && WaitFence(completion, d.fence_value, desc.wait_ms)))) {
                    ++i;
                    continue;
                }
                const std::string path = desc.dump_dir + "\\" + d.name;
                void* p = nullptr;
                const D3D12_RANGE all = {0, static_cast<SIZE_T>(d.pitch) * d.h};
                std::vector<uint8_t> rgb;
                bool ok = false;
                if (SUCCEEDED(d.buf->Map(0, &all, &p)) && p) {
                    ok = FgDumpToRgb8(static_cast<const uint8_t*>(p), d.pitch, d.w, d.h, d.format, &rgb);
                    const D3D12_RANGE none = {0, 0};
                    d.buf->Unmap(0, &none);
                }
                ok = ok && FgDumpWritePpm(path, d.w, d.h, rgb);
                DumpLog(ok ? "fg dump: wrote " + path + " (" + std::to_string(d.w) + " x " + std::to_string(d.h) +
                                 ", slot format " + std::to_string(d.format) + ")"
                           : "fg dump: " + path + " not written (map, format " + std::to_string(d.format) +
                                 " or file)");
                dumps.erase(dumps.begin() + static_cast<std::ptrdiff_t>(i));
            }
        } catch (...) {
            dumps.clear();
        }
    }

    // current recording
    Rec* cur = nullptr;

    ~Impl() override {
        if (event) CloseHandle(event);
    }

    FgExecResult Fail(FgExecRefusal r, const std::string& w) {
        refusal = r;
        why = w;
        if (!desc.dump_dir.empty()) DumpLog(std::string("fg exec fail: ") + FgExecRefusalName(r) + ": " + w);
        return r == FgExecRefusal::DeviceLost ? FgExecResult::DeviceLost : FgExecResult::Refused;
    }

    bool DeviceRemoved() const { return device && device->GetDeviceRemovedReason() != S_OK; }

    // ---- setup -----------------------------------------------------------------------------------------------
    FgExecResult LoadKernels() {
        if (desc.dxil_dir.empty() || !dir_exists(desc.dxil_dir))
            return Fail(FgExecRefusal::DxilDirMissing, "kernel folder '" + desc.dxil_dir + "' missing");
        for (uint32_t k = 0; k < kKernelCount; ++k) {
            if (!KernelNeeded(k, desc.enable_full, desc.net2)) continue;
            const std::string path = desc.dxil_dir + "\\" + kKernelNames[k] + ".dxil";
            std::vector<char> c;
            if (!read_file(path, &c)) {
                if (k >= kFirstOptional) continue;
                return Fail(FgExecRefusal::DxilMissing, std::string(kKernelNames[k]) + ".dxil");
            }
            if (c.size() < 32 || c.size() > 64u * 1024u * 1024u || std::memcmp(c.data(), "DXBC", 4) != 0)
                return Fail(FgExecRefusal::DxilInvalid, std::string(kKernelNames[k]) + ".dxil is not a DXIL container");
            code[k] = std::move(c);
        }
        return FgExecResult::Ok;
    }

    FgExecResult BuildPipelines() {
        D3D12_DESCRIPTOR_RANGE ranges[2] = {};
        ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        ranges[0].NumDescriptors = 1;
        ranges[0].RegisterSpace = kChannelSpace;
        ranges[0].OffsetInDescriptorsFromTableStart = 0;
        ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        ranges[1].NumDescriptors = kInputSrvs;
        ranges[1].OffsetInDescriptorsFromTableStart = 1;
        D3D12_ROOT_PARAMETER p[4] = {};
        p[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        p[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        p[1].Descriptor.ShaderRegister = 1;
        p[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        p[2].Constants.Num32BitValues = kRootConstants;
        p[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        p[3].DescriptorTable.NumDescriptorRanges = 2;
        p[3].DescriptorTable.pDescriptorRanges = ranges;
        D3D12_ROOT_SIGNATURE_DESC rs = {};
        rs.NumParameters = 4;
        rs.pParameters = p;
        ComPtr<ID3DBlob> blob, err;
        if (FAILED(D3D12SerializeRootSignature(&rs, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err)) ||
            FAILED(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                               IID_PPV_ARGS(&root))))
            return Fail(FgExecRefusal::RootSignature, "root signature creation failed");
        for (uint32_t k = 0; k < kKernelCount; ++k) {
            if (code[k].empty()) continue;
            D3D12_COMPUTE_PIPELINE_STATE_DESC pd = {};
            pd.pRootSignature = root.Get();
            pd.CS.pShaderBytecode = code[k].data();
            pd.CS.BytecodeLength = code[k].size();
            if (FAILED(device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&pso[k]))))
                return Fail(FgExecRefusal::PipelineFailed, std::string("pipeline ") + kKernelNames[k]);
        }
        pipelines_built = true;
        return FgExecResult::Ok;
    }

    uint32_t Alloc(uint64_t bytes, uint64_t alignment = 256) {
        bump = align(bump, alignment);
        const uint64_t at = bump;
        bump += align(bytes ? bytes : 4, 4);
        return static_cast<uint32_t>(at);
    }
    Net2Buf N2(uint32_t div, uint32_t c) {
        Net2Buf b;
        b.h = g.H2 / div;
        b.w = g.W2 / div;
        b.c = c;
        b.off = Alloc(2ull * b.h * b.w * c);
        return b;
    }

    FgExecResult Layout() {
        bump = 256;  // offset 0 is the null pointer
        for (uint32_t r = 0; r < kRoleCount; ++r) {
            const uint64_t n = FgRoleBytes(g, r);
            role_off[r] = n ? Alloc(n) : 0;
        }
        for (uint32_t l = 1; l <= g.top; ++l) ps_off[l] = Alloc(12ull * g.lw[l] * g.lh[l]);
        detres_off = Alloc(32);
        dis_off = detres_off + 16;
        const uint64_t cap = static_cast<uint64_t>(cap_w) * cap_h;
        cap_colour = Alloc(cap * 8);
        cap_hud = Alloc(cap * 8);
        cap_ui4 = Alloc(cap * 8);
        cap_ui1 = Alloc(cap * 2);
        cap_depth = Alloc(cap * 4);
        cap_motion = Alloc(cap * 8);
        out_stride = static_cast<uint32_t>(align(g.wo, 32));
        out_off = Alloc(8ull * out_stride * g.ho, 512);
        pack_pitch = static_cast<uint32_t>(align(4ull * g.wo, 256));
        pack_off = Alloc(static_cast<uint64_t>(pack_pitch) * g.ho, 512);
        // net1: input, weights and scratch regions
        const FgRefusal pr = FgBuildNet1Plan(g.H0, g.W0, &n1plan);
        if (pr != FgRefusal::None) return Fail(FgExecRefusal::SizeUnsupported, std::string("net1 plan: ") + FgRefusalName(pr));
        n1arena = FgArenaBinding();
        n1arena.region_base[0] = Alloc(n1plan.region_bytes[0]);
        n1arena.region_base[2] = Alloc(n1plan.region_bytes[2]);
        n1_cells = static_cast<uint32_t>(n1arena.region_base[0] + n1plan.buffers[n1plan.Find("cells")].offset);
        n1_out = static_cast<uint32_t>(n1arena.region_base[2] + n1plan.buffers[n1plan.Find("out")].offset);
        // The two coarse pairs run in sequence and can share the first convolution's output buffer.
        const bool split_b = pso[kMPairBFirst] && pso[kMPairBSecond];
        const bool split_d4 = pso[kMPairD4First] && pso[kMPairD4Second];
        n1_pair_scratch = 0;
        if (split_b || split_d4) {
            const uint64_t b_bytes = split_b ? 2 * n1plan.buffers[n1plan.Find("p5")].bytes : 0;
            const uint64_t d4_bytes = split_d4 ? n1plan.buffers[n1plan.Find("a4")].bytes : 0;
            n1_pair_scratch = Alloc(std::max(b_bytes, d4_bytes));
        }
        if (desc.enable_full) {
            const uint64_t fr = static_cast<uint64_t>(g.wo) * g.ho, n2 = static_cast<uint64_t>(g.W2) * g.H2;
            x0 = Alloc(fr * 8);
            x1 = Alloc(fr * 8);
            xw = Alloc(fr * 2);
            ib = Alloc(n2 * 8);
            ic = Alloc(n2 * 8);
            ia = Alloc(n2 * 2);
            fine = Alloc(fr * 2);
            for (uint32_t l = 1; l <= 8; ++l) {
                uint32_t w, h;
                FgExecOutLevelSize(g, l, &w, &h);
                lv[l] = Alloc(8ull * w * h);
            }
            nb_x = N2(1, 10);
            nb_a1 = N2(2, 32);
            nb_a2 = N2(4, 32);
            nb_r0 = N2(8, 64);
            for (int u = 0; u < 4; ++u) {
                nb_t[u] = N2(8, 64);
                nb_r[u] = N2(8, 64);
            }
            if (!fast()) nb_r4u = N2(4, 64);   // the fast path stages the upsamples inside the convolutions
            nb_g = N2(4, 96);
            if (!fast()) nb_gu = N2(2, 96);
            nb_hA4 = N2(2, 4);
            nb_hA1 = N2(2, 1);
            nb_hA3 = N2(2, 3);
            nb_U4 = N2(1, 4);
            nb_U1 = N2(1, 1);
            nb_U3 = N2(1, 3);
            nb_y = N2(1, 18);
            nb_b1 = N2(2, 16);
            nb_s0 = N2(4, 32);
            for (int u = 0; u < 4; ++u) {
                nb_tb[u] = N2(4, 32);
                nb_s[u] = N2(4, 32);
            }
            if (!fast()) nb_s4u = N2(2, 32);
            nb_e = N2(2, 48);
            if (!fast()) nb_eu = N2(1, 48);
            nb_out = N2(1, 8);
        }
        // the weights region last: net1's weights region, net2's weights, the logistic table
        weight_base = Alloc(0, 256);
        bump = weight_base;
        n1arena.region_base[1] = Alloc(n1plan.region_bytes[1]);
        if (desc.enable_full) {
            for (uint32_t i = 0; i < kFgNet2Layers; ++i) {
                const FgNet2Layer& l = FgNet2Layers()[i];
                if (fast()) {
                    n2fw[i] = Alloc(n2frag[i].size());
                    n2fb[i] = Alloc(n2fbias[i].size());
                } else {
                    n2w[i] = Alloc(2ull * l.cout * 9 * l.cin);
                    n2b[i] = Alloc(2ull * l.cout);
                }
            }
            sig = Alloc(65536ull * 4);
        }
        bump = align(bump, 65536);
        if (bump > 0xFFFF0000ull) return Fail(FgExecRefusal::SizeUnsupported, "arena above 32-bit offsets");
        arena_bytes = bump;
        n1arena.bound = true;
        n1arena.arena_bytes = arena_bytes;
        return FgExecResult::Ok;
    }

    FgExecResult FillWeightImage() {
        weight_image.assign(arena_bytes - weight_base, 0);
        auto put = [this](uint64_t off, const std::vector<uint8_t>& v, uint64_t expect) -> bool {
            if (v.size() != expect || off < weight_base || off + v.size() > arena_bytes) return false;
            std::memcpy(weight_image.data() + (off - weight_base), v.data(), v.size());
            return true;
        };
        for (uint32_t i = 0; i < 17; ++i) {
            const FgNet1Layer& l = FgNet1Layers()[i];
            const int wfi = n1plan.Find(std::string("wf_") + l.name), bi = n1plan.Find(std::string("b_") + l.name);
            if (wfi < 0 || bi < 0) return Fail(FgExecRefusal::WeightsInvalid, "net1 plan has no weight buffer");
            const FgBufferSpec& wf = n1plan.buffers[static_cast<size_t>(wfi)];
            const FgBufferSpec& b = n1plan.buffers[static_cast<size_t>(bi)];
            if (!put(n1arena.region_base[1] + wf.offset, weights.net1_fragments[i], wf.bytes) ||
                !put(n1arena.region_base[1] + b.offset, weights.net1_bias[i], b.bytes))
                return Fail(FgExecRefusal::WeightsInvalid, std::string("net1 layer ") + l.name + " size");
        }
        if (desc.enable_full) {
            for (uint32_t i = 0; i < kFgNet2Layers; ++i) {
                const FgNet2Layer& l = FgNet2Layers()[i];
                const bool ok = fast() ? put(n2fw[i], n2frag[i], n2frag[i].size()) &&
                                             put(n2fb[i], n2fbias[i], n2fbias[i].size())
                                       : put(n2w[i], weights.net2_weights[i], 2ull * l.cout * 9 * l.cin) &&
                                             put(n2b[i], weights.net2_bias[i], 2ull * l.cout);
                if (!ok) return Fail(FgExecRefusal::WeightsInvalid, "net2 layer " + std::to_string(i + 1) + " size");
            }
            std::vector<float> table;
            FgExecSigTable(&table);
            std::memcpy(weight_image.data() + (sig - weight_base), table.data(), table.size() * 4);
        }
        return FgExecResult::Ok;
    }

    ComPtr<ID3D12Resource> Buffer(uint64_t bytes, D3D12_HEAP_TYPE type, D3D12_RESOURCE_FLAGS flags,
                                  D3D12_RESOURCE_STATES state) {
        D3D12_HEAP_PROPERTIES hp = {};
        hp.Type = type;
        D3D12_RESOURCE_DESC rd = {};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width = bytes;
        rd.Height = 1;
        rd.DepthOrArraySize = 1;
        rd.MipLevels = 1;
        rd.SampleDesc.Count = 1;
        rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        rd.Flags = flags;
        ComPtr<ID3D12Resource> r;
        if (FAILED(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, state, nullptr, IID_PPV_ARGS(&r))))
            r.Reset();
        return r;
    }

    FgExecResult CreateObjects() {
        if (!fence) {
            if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&fence))))
                return Fail(FgExecRefusal::FenceFailed, "shared fence creation failed");
        }
        if (!event) {
            event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
            if (!event) return Fail(FgExecRefusal::FenceFailed, "event creation failed");
        }
        if (!heap) {
            const uint32_t n = std::max<uint32_t>(4, desc.allocators);
            const uint32_t count = compute_queue ? 2 * n : n;
            D3D12_DESCRIPTOR_HEAP_DESC hd = {};
            hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
            hd.NumDescriptors = count * kTables * kTableSize;
            hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
            if (FAILED(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap))))
                return Fail(FgExecRefusal::AllocationFailed, "descriptor heap");
            desc_inc = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            channel = Buffer(256, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                             D3D12_RESOURCE_STATE_COMMON);
            menu_rb = Buffer(256, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
            if (!channel || !menu_rb) return Fail(FgExecRefusal::AllocationFailed, "channel / readback buffer");
            for (uint32_t d = 0; d < hd.NumDescriptors; ++d) {
                D3D12_CPU_DESCRIPTOR_HANDLE h = heap->GetCPUDescriptorHandleForHeapStart();
                h.ptr += static_cast<SIZE_T>(d) * desc_inc;
                if (d % kTableSize == 0) {
                    D3D12_UNORDERED_ACCESS_VIEW_DESC uav = {};
                    uav.Format = DXGI_FORMAT_R32_TYPELESS;
                    uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
                    uav.Buffer.NumElements = 64;
                    uav.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
                    device->CreateUnorderedAccessView(channel.Get(), nullptr, &uav, h);
                } else {
                    D3D12_SHADER_RESOURCE_VIEW_DESC sv = {};
                    sv.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
                    sv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
                    sv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                    sv.Texture2D.MipLevels = 1;
                    device->CreateShaderResourceView(nullptr, &sv, h);
                }
            }
            if (!CreateEntries(entries, n, 0, ProducerQueue()) ||
                (compute_queue && !CreateEntries(direct_entries, n, n, queue.Get())))
                return Fail(FgExecRefusal::AllocationFailed, "command allocator / list");
            if (desc.timing) {
                CreateTimingQueries(producer_queries, n, ProducerQueue());
                if (compute_queue) CreateTimingQueries(direct_queries, n, queue.Get());
            }
        }
        return FgExecResult::Ok;
    }

    ID3D12CommandQueue* ProducerQueue() const { return compute_queue ? compute_queue.Get() : queue.Get(); }

    bool CreateEntries(std::vector<Entry>& ring, uint32_t count, uint32_t first, ID3D12CommandQueue* q) {
        const auto type = q->GetDesc().Type;
        ring.resize(count);
        for (uint32_t i = 0; i < count; ++i) {
            Entry& e = ring[i];
            if (FAILED(device->CreateCommandAllocator(type, IID_PPV_ARGS(&e.alloc))) ||
                FAILED(device->CreateCommandList(0, type, e.alloc.Get(), nullptr, IID_PPV_ARGS(&e.list))) ||
                FAILED(e.list->Close()))
                return false;
            e.desc_base = (first + i) * kTables * kTableSize;
            e.query_base = i * kTimingQueries;
        }
        return true;
    }

    void CreateTimingQueries(TimingQueries& q, uint32_t count, ID3D12CommandQueue* source) {
        if (FAILED(source->GetTimestampFrequency(&q.hz)) || !q.hz) return;
        D3D12_QUERY_HEAP_DESC qd = {};
        qd.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        qd.Count = count * kTimingQueries;
        if (SUCCEEDED(device->CreateQueryHeap(&qd, IID_PPV_ARGS(&q.heap))))
            q.readback = Buffer(qd.Count * sizeof(uint64_t), D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE,
                                D3D12_RESOURCE_STATE_COPY_DEST);
    }

    void TimeStage(Rec& r, FgExecStage stage) {
        const auto& q = r.direct ? direct_queries : producer_queries;
        if (!q.readback) return;
        Entry& e = *r.e;
        if (e.stage_count && e.stages[e.stage_count - 1] == stage) return;
        r.cl->EndQuery(q.heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, e.query_base + e.stage_count);
        e.stages[e.stage_count++] = stage;
    }

    void ResolveTimings(Rec& r) {
        const Entry& e = *r.e;
        if (!e.stage_count) return;
        const auto& q = r.direct ? direct_queries : producer_queries;
        r.cl->EndQuery(q.heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, e.query_base + e.stage_count);
        r.cl->ResolveQueryData(q.heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, e.query_base, e.stage_count + 1,
                              q.readback.Get(), e.query_base * sizeof(uint64_t));
    }

    void CollectTimings(Entry& e, const TimingQueries& q) {
        if (!e.stage_count || !e.fence) return;
        const D3D12_RANGE range = {e.query_base * sizeof(uint64_t),
                                   (e.query_base + e.stage_count + 1) * sizeof(uint64_t)};
        void* p = nullptr;
        if (SUCCEEDED(q.readback->Map(0, &range, &p))) {
            const auto* ts = static_cast<const uint64_t*>(p) + e.query_base;
            for (uint32_t i = 0; i < e.stage_count; ++i)
                timings.ms[static_cast<size_t>(e.stages[i])] += (ts[i + 1] - ts[i]) * 1000.0 / q.hz;
            const D3D12_RANGE none = {0, 0};
            q.readback->Unmap(0, &none);
        } else {
            timing_failed = true;
        }
        e.stage_count = 0;
    }

    bool WaitFence(ID3D12Fence* f, uint64_t v, uint32_t ms) {
        const ULONGLONG until = GetTickCount64() + ms;
        if (!v || f->GetCompletedValue() >= v) return true;
        if (!ms || !event || FAILED(f->SetEventOnCompletion(v, event))) return false;
        // An earlier timed-out registration can still signal this event.
        while (f->GetCompletedValue() < v) {
            const ULONGLONG now = GetTickCount64();
            if (now >= until || WaitForSingleObject(event, static_cast<DWORD>(until - now)) != WAIT_OBJECT_0)
                return f->GetCompletedValue() >= v;
        }
        return true;
    }

    bool WaitValue(uint64_t v, uint32_t ms) { return WaitFence(fence.Get(), v, ms); }

    bool WaitIdle(uint32_t ms) {
        const ULONGLONG until = GetTickCount64() + ms;
        if (!WaitValue(next_value, ms)) return false;
        const ULONGLONG now = GetTickCount64();
        return WaitFence(handoff_fence.Get(), handoff_value, now < until ? static_cast<uint32_t>(until - now) : 0);
    }

    FgExecResult Upload() {
        ComPtr<ID3D12Resource> up = Buffer(weight_image.size(), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
                                           D3D12_RESOURCE_STATE_GENERIC_READ);
        if (!up) return Fail(FgExecRefusal::AllocationFailed, "upload buffer");
        void* p = nullptr;
        D3D12_RANGE none = {0, 0};
        if (FAILED(up->Map(0, &none, &p))) return Fail(FgExecRefusal::AllocationFailed, "upload map");
        std::memcpy(p, weight_image.data(), weight_image.size());
        up->Unmap(0, nullptr);
        Rec r;
        FgExecResult b = Begin(&r);
        if (b != FgExecResult::Ok) return b;
        ArenaTo(r, D3D12_RESOURCE_STATE_COPY_DEST);
        r.cl->CopyBufferRegion(arena.Get(), weight_base, up.Get(), 0, weight_image.size());
        uint64_t v = 0;
        b = Submit(&r, &v);
        if (b != FgExecResult::Ok) return b;
        if (!WaitValue(v, desc.wait_ms * 4)) return Fail(FgExecRefusal::WaitTimeout, "weight upload");
        return FgExecResult::Ok;
    }

    // ---- recording -------------------------------------------------------------------------------------------
    FgExecResult Begin(Rec* r, bool on_game_queue = false, bool uses_arena = true) {
        r->direct = compute_queue && on_game_queue;
        r->uses_arena = uses_arena;
        auto& ring = r->direct ? direct_entries : entries;
        ID3D12Fence* reuse_fence = r->direct ? handoff_fence.Get() : fence.Get();
        Entry* e = nullptr;
        const uint64_t done = reuse_fence->GetCompletedValue();
        for (Entry& x : ring)
            if (x.fence <= done) {
                e = &x;
                break;
            }
        if (!e) {
            Entry* oldest = &ring[0];
            for (Entry& x : ring)
                if (x.fence < oldest->fence) oldest = &x;
            if (!WaitFence(reuse_fence, oldest->fence, desc.wait_ms)) {
                if (DeviceRemoved()) return Fail(FgExecRefusal::DeviceLost, "device removed");
                return Fail(FgExecRefusal::WaitTimeout, "no free command allocator");
            }
            e = oldest;
        }
        CollectTimings(*e, r->direct ? direct_queries : producer_queries);
        e->fence = 0;
        e->stage_count = 0;
        if (FAILED(e->alloc->Reset()) || FAILED(e->list->Reset(e->alloc.Get(), nullptr)))
            return Fail(FgExecRefusal::CommandFailed, "command list reset");
        r->e = e;
        r->cl = e->list.Get();
        r->open = true;
        r->table = 0;
        r->dirty = false;
        if (!uses_arena) return FgExecResult::Ok;
        ID3D12DescriptorHeap* heaps[] = {heap.Get()};
        r->cl->SetDescriptorHeaps(1, heaps);
        r->cl->SetComputeRootSignature(root.Get());
        r->cl->SetComputeRootUnorderedAccessView(0, arena->GetGPUVirtualAddress());
        r->cl->SetComputeRootUnorderedAccessView(1, arena->GetGPUVirtualAddress());
        SetTable(*r, 0);
        r->arena = D3D12_RESOURCE_STATE_COMMON;
        ArenaTo(*r, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        cur = r;
        return FgExecResult::Ok;
    }

    void SetTable(Rec& r, uint32_t table) {
        D3D12_GPU_DESCRIPTOR_HANDLE h = heap->GetGPUDescriptorHandleForHeapStart();
        h.ptr += static_cast<UINT64>(r.e->desc_base + kTableSize * table) * desc_inc;
        r.cl->SetComputeRootDescriptorTable(3, h);
    }

    void ArenaTo(Rec& r, D3D12_RESOURCE_STATES s) {
        if (r.arena == s) return;
        D3D12_RESOURCE_BARRIER b = {};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition.pResource = arena.Get();
        b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        b.Transition.StateBefore = r.arena;
        b.Transition.StateAfter = s;
        r.cl->ResourceBarrier(1, &b);
        r.arena = s;
        r.dirty = false;
    }

    void Transition(Rec& r, ID3D12Resource* res, D3D12_RESOURCE_STATES a, D3D12_RESOURCE_STATES b) {
        if (a == b) return;
        D3D12_RESOURCE_BARRIER x = {};
        x.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        x.Transition.pResource = res;
        x.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        x.Transition.StateBefore = a;
        x.Transition.StateAfter = b;
        r.cl->ResourceBarrier(1, &x);
    }

    void Barrier(Rec& r) {
        if (!r.dirty) return;
        D3D12_RESOURCE_BARRIER b = {};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        b.UAV.pResource = nullptr;
        r.cl->ResourceBarrier(1, &b);
        r.dirty = false;
    }

    void Dispatch(Rec& r, uint32_t k, uint32_t groups, const uint32_t* args, uint32_t n, bool barrier = true) {
        ArenaTo(r, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        if (barrier) Barrier(r);
        r.cl->SetPipelineState(pso[k].Get());
        uint32_t c[kRootConstants] = {};
        c[1] = groups;
        std::memcpy(c + 2, args, 4ull * std::min(n, kMaxArgs));
        for (uint32_t base = 0; base < groups; base += 65535) {
            c[0] = base;
            r.cl->SetComputeRoot32BitConstants(2, 2 + n, c, 0);
            r.cl->Dispatch(std::min<uint32_t>(groups - base, 65535), 1, 1);
        }
        r.dirty = true;
        ++counters.dispatches;
    }
    void D(Rec& r, uint32_t k, uint32_t groups, std::initializer_list<uint32_t> a, bool barrier = true) {
        uint32_t v[kMaxArgs] = {};
        uint32_t n = 0;
        for (uint32_t x : a)
            if (n < kMaxArgs) v[n++] = x;
        Dispatch(r, k, groups, v, n, barrier);
    }

    FgExecResult Submit(Rec* r, uint64_t* value) {
        cur = nullptr;
        // Buffers decay at submission boundaries. Make the Direct-to-Compute release explicit.
        if (r->direct && r->uses_arena) ArenaTo(*r, D3D12_RESOURCE_STATE_COMMON);
        ResolveTimings(*r);
        r->open = false;
        if (FAILED(r->cl->Close())) return Fail(FgExecRefusal::CommandFailed, "command list close");
        ID3D12CommandQueue* q = r->direct ? queue.Get() : ProducerQueue();
        if (r->direct) {
            uint64_t dependency = r->uses_arena ? arena_value : 0;
            const auto slot = compute_slot_values.find(r->written_slot);
            if (slot != compute_slot_values.end()) dependency = std::max(dependency, slot->second);
            if (dependency > fence->GetCompletedValue() && FAILED(q->Wait(fence.Get(), dependency)))
                return QueueFailure("wait for compute writes");
        }
        if (!r->direct && !JoinGameQueue()) return QueueFailure("wait for game queue handoff");
        ID3D12CommandList* l[] = {r->cl};
        q->ExecuteCommandLists(1, l);
        uint64_t v = 0;
        if (!(r->direct ? SignalHandoff(&v) : SignalProducer(&v))) return QueueFailure("queue signal");
        if (compute_queue && !r->direct) {
            if (r->uses_arena) arena_value = v;
            if (r->written_slot) compute_slot_values[r->written_slot] = v;
        }
        r->e->fence = v;
        ++counters.submissions;
        *value = v;
        if (DeviceRemoved()) return Fail(FgExecRefusal::DeviceLost, "device removed");
        return FgExecResult::Ok;
    }

    FgExecResult QueueFailure(const char* operation) {
        if (DeviceRemoved()) return Fail(FgExecRefusal::DeviceLost, "device removed");
        return Fail(FgExecRefusal::CommandFailed, operation);
    }

    bool SignalHandoff(uint64_t* value) {
        if (FAILED(queue->Signal(handoff_fence.Get(), handoff_value + 1))) return false;
        *value = ++handoff_value;
        return true;
    }

    bool JoinGameQueue() {
        if (!compute_queue || joined_handoff == handoff_value) return true;
        if (FAILED(compute_queue->Wait(handoff_fence.Get(), handoff_value))) return false;
        joined_handoff = handoff_value;
        return true;
    }

    bool SignalProducer(uint64_t* value) {
        // Only the producer queue publishes values, so a direct copy cannot overtake earlier compute work.
        if (FAILED(ProducerQueue()->Signal(fence.Get(), next_value + 1))) return false;
        *value = ++next_value;
        return true;
    }

    // IFgComputeRecorder (net1)
    void BindArena() override {}
    void UavBarrier() override {
        if (cur) {
            cur->dirty = true;
            Barrier(*cur);
        }
    }
    FgNet1Kernel n1_record_kernel{};
    std::array<uint32_t, kFgNet1Args + 2> n1_record_constants{};

    void SetPipeline(FgNet1Kernel k) override { n1_record_kernel = k; }
    void SetConstants(const uint32_t* dwords, uint32_t count) override {
        std::copy_n(dwords, count, n1_record_constants.begin());
    }

    void Net1PairSplit(Rec& r, Kernel first, Kernel second, uint32_t first_tiles, uint32_t second_tiles,
                       const uint32_t* args) {
        const uint32_t pixel_groups = cdiv(static_cast<uint64_t>(args[8]) * args[9], 32);
        const uint32_t dst = args[6];
        uint32_t a[kFgNet1Args];
        std::copy_n(args, kFgNet1Args, a);
        a[6] = n1_pair_scratch;
        Dispatch(r, first, pixel_groups * (first_tiles / 2), a, kFgNet1Args, false);
        a[0] = n1_pair_scratch;
        a[6] = dst;
        Dispatch(r, second, pixel_groups * (second_tiles / 2), a, kFgNet1Args);
    }

    void Dispatch(uint32_t groups_x) override {
        if (!cur) return;
        const auto& c = n1_record_constants;
        const uint32_t* args = c.data() + 2;
        const FgNet1Kernel kernel = n1_record_kernel;
        // The validator records the original launch. Retile complete launches when a tuned PSO is available.
        if (c[0] == 0 && c[1] == groups_x) {
            if (kernel == FgNet1Kernel::MPairB && pso[kMPairBFirst] && pso[kMPairBSecond] && n1_pair_scratch) {
                Net1PairSplit(*cur, kMPairBFirst, kMPairBSecond, 32, 16, args);
                return;
            }
            if (kernel == FgNet1Kernel::MPairD4 && pso[kMPairD4First] && pso[kMPairD4Second] && n1_pair_scratch) {
                Net1PairSplit(*cur, kMPairD4First, kMPairD4Second, 16, 8, args);
                return;
            }
            if ((kernel == FgNet1Kernel::MPairD3 && pso[kMPairD3Narrow]) ||
                (kernel == FgNet1Kernel::MPairD2 && pso[kMPairD2Narrow])) {
                const Kernel tuned = kernel == FgNet1Kernel::MPairD3 ? kMPairD3Narrow : kMPairD2Narrow;
                Dispatch(*cur, tuned, cdiv(static_cast<uint64_t>(args[8]) * args[9], 32), args, kFgNet1Args, false);
                return;
            }

            Kernel tuned = kKernelCount;
            uint32_t tiles_per_wave = 0;
            uint32_t tile_h = 4;
            const bool small_conv = kernel == FgNet1Kernel::MConv4 && args[9] == 3 && groups_x <= 128;
            if (small_conv && groups_x <= 48 && pso[kMConv1Late]) {
                tuned = kMConv1Late;
                tiles_per_wave = 1;
            } else if (kernel == FgNet1Kernel::MConv2 && args[9] == 3 && args[10] == 1 && pso[kMConv2Mid]) {
                tuned = kMConv2Mid;
                tiles_per_wave = 2;
                tile_h = 8;
            } else if ((kernel == FgNet1Kernel::MConv2 || small_conv) && pso[kMConv2Narrow]) {
                tuned = kMConv2Narrow;
                tiles_per_wave = 2;
            }
            if (tuned != kKernelCount) {
                uint32_t a[kFgNet1Args];
                std::copy_n(args, kFgNet1Args, a);
                a[14] = cdiv(a[7], tile_h);
                const uint32_t groups = a[13] * a[14] * (a[6] / (16 * tiles_per_wave));
                Dispatch(*cur, tuned, groups, a, kFgNet1Args, false);
                return;
            }
        }
        cur->cl->SetPipelineState(pso[kMConv1 + static_cast<uint32_t>(kernel)].Get());
        cur->cl->SetComputeRoot32BitConstants(2, static_cast<uint32_t>(c.size()), c.data(), 0);
        cur->cl->Dispatch(groups_x, 1, 1);
        cur->dirty = true;
        ++counters.dispatches;
    }

    // ---- stages ----------------------------------------------------------------------------------------------
    uint32_t R(uint32_t role) const { return role_off[role]; }

    void Fill16(Rec& r, uint32_t off, uint64_t bytes, const uint32_t* p, uint32_t period, bool barrier = true) {
        const uint32_t n16 = static_cast<uint32_t>(bytes / 2);
        D(r, kFill16, groups_for(n16, 8), {off, n16, p[0], p[1], p[2], p[3], period}, barrier);
    }

    void RegArgs(const InputTex& t, bool present, uint32_t* out10) const {
        if (present) {
            t.region.Args(out10);
        } else {
            for (int i = 0; i < 10; ++i) out10[i] = 0;
        }
    }

    // One schedule operation. Returns false when a step cannot be recorded (net1 refusal).
    bool RecordOp(Rec& r, const FgExecOp& op, bool input_srvs = false) {
        const FrameCtx& f = frame;
        const uint32_t col = cap_colour, ctw = f.colour.w;
        const InputTex& cr = f.colour;
        const uint32_t hud = f.hud ? cap_hud : 0, hw = f.hud ? f.hudless.w : 0, hh = f.hud ? f.hudless.h : 0;
        const bool alpha = f.ui1 || f.ui4;
        const uint32_t aid = f.ui1 ? cap_ui1 : (f.ui4 ? cap_ui4 : 0);
        const uint32_t aw = alpha ? f.alpha.w : 0, ah = alpha ? f.alpha.h : 0;
        uint32_t hreg[10], areg[10];
        RegArgs(f.hudless, f.hud, hreg);
        RegArgs(f.alpha, alpha, areg);
        const uint32_t nsc = g.w1 * g.h1;
        float t1 = 0, t2 = 0;
        FgExecThresholds(g.ho, &t1, &t2);
        const volatile float one = 1.0f;
        const float step_x = one / static_cast<float>(f.depth.region.ew), step_y = one / static_cast<float>(f.depth.region.eh);
        const FgExecSettings& st = desc.settings;
        const uint32_t inv = f.inverted ? 1u : 0u;
        switch (op.op) {
            case FgOp::Clear:
            case FgOp::Fill64:
                // History clears have distinct destinations, S is independent of downsample and motion prep,
                // and the per-index MA and MB clears are independent of each other.
                Fill16(r, R(op.role), FgRoleBytes(g, op.role), op.value, op.period, false);
                return true;
            case FgOp::SetDisable: {
                const uint32_t p[4] = {op.base & 0xFFFFu, op.base >> 16, 0, 0};
                Fill16(r, dis_off, 4, p, 2, false);
                frame.disable = op.base;
                return true;
            }
            case FgOp::Detect: {
                uint32_t a[29] = {};
                if (op.detect_kind == 0) {
                    const uint32_t v[15] = {0, aid, aw, f.alpha_kind, f.alpha_channel, f.alpha.region.bx,
                                            f.alpha.region.by, f.alpha.region.ew, f.alpha.region.eh,
                                            fb(st.alpha_threshold), fb(st.fraction), 0, 0, 0, 0};
                    std::memcpy(a, v, sizeof(v));
                } else {
                    const uint32_t same = (f.hudless.region.bx == cr.region.bx && f.hudless.region.by == cr.region.by);
                    const uint32_t v[15] = {1, col, ctw, 4, 0, cr.region.bx, cr.region.by, cr.region.ew, cr.region.eh,
                                            0, 0, hud, hw, hh, same};
                    std::memcpy(a, v, sizeof(v));
                    std::memcpy(a + 15, hreg, sizeof(hreg));
                }
                a[25] = detres_off;
                a[26] = dis_off;
                a[27] = op.base;
                a[28] = op.hysteresis_off ? 1u : 0u;
                Dispatch(r, kDetect, 1, a, 29);
                return true;
            }
            case FgOp::Downsample: {
                uint32_t a[44];
                uint32_t n = 0;
                const uint32_t h1[7] = {col, ctw, cr.region.bx, cr.region.by, hud, hw, hh};
                for (uint32_t x : h1) a[n++] = x;
                for (uint32_t x : hreg) a[n++] = x;
                const uint32_t h2[5] = {aid, aw, ah, f.alpha_kind, f.alpha_channel};
                for (uint32_t x : h2) a[n++] = x;
                for (uint32_t x : areg) a[n++] = x;
                const uint32_t h3[12] = {R(kRoleCp), R(kRoleCc), R(kRoleUc), R(kRoleLc), R(kRoleDc), R(kRoleLd),
                                         R(kRoleUm), g.wo, g.ho, g.wd, g.hd, host.Frame().r3_bits};
                for (uint32_t x : h3) a[n++] = x;
                const uint64_t nb = static_cast<uint64_t>(cdiv(g.wo, g.d)) * cdiv(g.ho, g.d);
                // A reset cleared the history/UI roles this dispatch accesses. With history, the preceding
                // SetDisable writes a separate word, or the detector already ended its own submission.
                if (input_srvs)
                    BindInputs(r, {&cr, f.hud ? &f.hudless : nullptr, alpha ? &f.alpha : nullptr});
                uint32_t kernel = input_srvs ? (g.d == 4 ? kDown4Srv : kDown2Srv) : (g.d == 4 ? kDown4 : kDown2);
                uint32_t groups = groups_for(nb);
                const auto& hr = f.hudless.region;
                if (kernel == kDown2Srv && pso[kDown2HudTile] && (host.Frame().r3_bits & 16u) &&
                    hw == g.wo && hh == g.ho && !hr.bx && !hr.by && hr.ew == hw && hr.eh == hh) {
                    kernel = kDown2HudTile;
                    groups = cdiv(g.wo, 32) * cdiv(g.ho, 8);
                }
                Dispatch(r, kernel, groups, a, n, !host.Frame().history);
                return true;
            }
            case FgOp::Mvec: {
                uint32_t a[34];
                uint32_t n = 0;
                uint32_t dreg[10], mreg[10];
                f.depth.region.Args(dreg);
                f.motion.region.Args(mreg);
                a[n++] = cap_depth;
                a[n++] = f.depth.w;
                a[n++] = f.depth.h;
                for (uint32_t x : dreg) a[n++] = x;
                a[n++] = cap_motion;
                a[n++] = f.motion.w;
                a[n++] = f.motion.h;
                for (uint32_t x : mreg) a[n++] = x;
                const uint32_t t[8] = {R(kRoleZc), R(kRoleMv), g.w1, g.h1, inv, fb(st.s_k), fb(f.scx), fb(f.scy)};
                for (uint32_t x : t) a[n++] = x;
                // Downsample writes colour/UI roles. Motion prep writes depth and motion roles.
                if (input_srvs) BindInputs(r, {&f.depth, &f.motion});
                Dispatch(r, input_srvs ? kMvecSrv : kMvec, groups_for(nsc), a, n, false);
                if (input_srvs) SetTable(r, 0);
                return true;
            }
            case FgOp::P2cScatter: {
                uint32_t a[28];
                uint32_t n = 0;
                const uint32_t h[6] = {R(kRoleZc), R(kRoleMv), R(kRoleS), R(kRoleKc), g.w1, g.h1};
                for (uint32_t x : h) a[n++] = x;
                for (float m : f.clip_to_prev) a[n++] = fb(m);
                const uint32_t t[6] = {fb(t1), fb(t2), fb(st.s_k), inv, fb(step_x), fb(step_y)};
                for (uint32_t x : t) a[n++] = x;
                Dispatch(r, kScatter, groups_for(nsc), a, n);
                return true;
            }
            case FgOp::P2cUnpack: {
                uint32_t a[32];
                uint32_t n = 0;
                const uint32_t h[10] = {R(kRoleS), R(kRoleZp), R(kRoleKp), R(kRoleMv), R(kRoleF),
                                        R(kRoleE), R(kRoleFl), R(kRoleVf), g.w1, g.h1};
                for (uint32_t x : h) a[n++] = x;
                for (float m : f.prev_to_clip) a[n++] = fb(m);
                const uint32_t t[6] = {fb(st.s_k), inv, fb(step_x), fb(step_y), fb(st.kmax), fb(st.tau)};
                for (uint32_t x : t) a[n++] = x;
                Dispatch(r, kUnpack, groups_for(nsc), a, n);
                return true;
            }
            case FgOp::P2cInpaint:
                D(r, kInpaint, groups_for(nsc), {R(kRoleFl), R(kRoleG), g.w1, g.h1});
                return true;
            case FgOp::P2cPull:
                if (pso[kPullTiled]) {
                    for (uint32_t l : {1u, 5u}) {
                        if (l > g.top) break;
                        const uint32_t w = g.lw[l], h = g.lh[l];
                        D(r, kPullTiled, cdiv(w, 16) * cdiv(h, 16),
                          {l == 1 ? 0u : 2u, l == 1 ? R(kRoleG) : R(kRoleP1 + 3), R(kRoleVf), 0u,
                           R(kRoleP1 + l - 1), g.lw[l - 1], g.lh[l - 1], w, h, fb(4.0f),
                           R(kRoleP1 + l), R(kRoleP1 + l + 1), R(kRoleP1 + l + 2), std::min(4u, g.top - l + 1)});
                    }
                    return true;
                }
                for (uint32_t l = 1; l <= g.top; ++l) {
                    const uint32_t wl = g.lw[l], hl = g.lh[l], ws = g.lw[l - 1], hs = g.lh[l - 1];
                    uint32_t mode, sa, sb = 0;
                    float n = 4.0f;
                    if (l == 1) {
                        mode = 0;
                        sa = R(kRoleG);
                        sb = R(kRoleVf);
                    } else if (l == 5) {
                        mode = 2;
                        sa = R(kRoleP1 + 3);
                    } else {
                        mode = 1;
                        sa = ps_off[l - 1];
                        const uint32_t e = l <= 4 ? l : l - 4;
                        n = static_cast<float>(1u << (2 * e));
                    }
                    D(r, kPull, groups_for(static_cast<uint64_t>(wl) * hl),
                      {mode, sa, sb, ps_off[l], R(kRoleP1 + l - 1), ws, hs, wl, hl, fb(n)});
                }
                return true;
            case FgOp::P2cPush: {
                uint32_t l = g.top - 1;
                if (pso[kPushTail]) {
                    uint32_t args[25] = {};
                    for (uint32_t j = 0; j <= l && g.lw[g.top - j] * g.lh[g.top - j] <= 256; ++j) {
                        const uint32_t level = g.top - j;
                        args[0] = j;
                        args[1 + j] = R(kRoleP1 + level - 1);
                        args[9 + j] = g.lw[level];
                        args[17 + j] = g.lh[level];
                    }
                    if (args[0] > 1) {
                        Dispatch(r, kPushTail, 1, args, 25);
                        l -= args[0];
                    }
                }
                for (; l >= 1; --l) {
                    D(r, kPush, groups_for(static_cast<uint64_t>(g.lw[l]) * g.lh[l]),
                      {R(kRoleP1 + l), g.lw[l + 1], g.lh[l + 1], R(kRoleP1 + l - 1), g.lw[l], g.lh[l]});
                    if (l == 1) break;
                }
                D(r, kFine, groups_for(nsc), {R(kRoleP1), g.lw[1], g.lh[1], R(kRoleG), R(kRoleF), g.w1, g.h1});
                return true;
            }
            case FgOp::HoleMask: {
                const uint32_t p[4] = {0x3C00u, 0, 0, 0};
                Fill16(r, R(kRoleHc), FgRoleBytes(g, kRoleHc), p, 1, false);
                D(r, kHoleMask, groups_for(static_cast<uint64_t>(g.wd) * g.hd),
                  {R(kRoleF), g.w1, g.h1, R(kRoleHc), g.wd, g.hd});
                return true;
            }
            case FgOp::QualityInit:
                D(r, kQualityInit, groups_for(static_cast<uint64_t>(g.wd) * g.hd),
                  {R(kRoleDc), R(kRoleQc), g.wd * g.hd}, false);
                return true;
            case FgOp::Net1Input:
                D(r, kNet1Input, groups_for(4ull * g.W0 * g.H0),
                  {R(kRoleLd), R(kRoleDc), R(kRoleDp), R(kRoleF), R(kRoleMv), R(kRoleE), R(kRoleHc), R(kRoleHp),
                   R(kRoleQp), R(kRoleM), n1_cells, R(kRoleQc), R(kRoleOm), g.wd, g.hd, g.w1, g.h1, g.W0, g.H0,
                   g.pad_l, g.pad_t, 1u, 48u});
                return true;
            case FgOp::Net1: {
                FgPipelineSet ps;
                for (uint32_t k = 0; k < kFgNet1KernelCount; ++k) ps.ready[k] = pso[kMConv1 + k] != nullptr;
                ArenaTo(r, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                r.dirty = true;
                std::string w;
                const FgRefusal rr = FgRecordNet1(n1plan, ps, n1arena, *this, &w);
                if (rr != FgRefusal::None) {
                    Fail(FgExecRefusal::Net1Refused, std::string(FgRefusalName(rr)) + ": " + w);
                    return false;
                }
                r.cl->SetPipelineState(pso[kFill16].Get());  // nothing reads it; keeps the state defined
                r.dirty = true;
                return true;
            }
            case FgOp::Net1Output:
                D(r, kNet1Output, groups_for(static_cast<uint64_t>(g.wd) * g.hd),
                  {n1_out, R(kRoleOm), R(kRoleUm), R(kRoleUmp), R(kRoleNA), R(kRoleNB), R(kRoleM), g.wd, g.hd,
                   g.pad_l, g.pad_t, g.W0, g.H0, 1u, 16u, op.ui_forcing ? 1u : 0u});
                return true;
            case FgOp::Interm: {
                const uint32_t kernel = pso[kIntermTiled] ? kIntermTiled : kInterm;
                const uint32_t groups = kernel == kIntermTiled ? cdiv(g.w1, 16) * cdiv(g.h1, 16) : groups_for(nsc);
                D(r, kernel, groups,
                  {0u, R(kRoleMv), R(kRoleZc), R(kRoleMA), g.w1, g.h1, fb(op.t), fb(st.s_k), inv, fb(t2), fb(st.tau)});
                D(r, kernel, groups,
                  {1u, R(kRoleF), R(kRoleZp), R(kRoleMB), g.w1, g.h1, fb(op.t), fb(st.s_k), inv, fb(t2), fb(st.tau)},
                  false);
                return true;
            }
            case FgOp::Weights:
                D(r, kWeights, groups_for(nsc),
                  {R(kRoleMA), R(kRoleMB), R(kRoleNA), R(kRoleNB), R(kRoleW), g.w1, g.h1, g.wd, g.hd});
                return true;
            case FgOp::Prefill:
                D(r, kCopy4, groups_for(static_cast<uint64_t>(g.wo) * g.ho),
                  {out_off, out_stride, 0, 0, col, ctw, 0, 0, g.wo, g.ho});
                return true;
            case FgOp::LiteBlend: {
                FgExecRegion fr;
                FgExecMakeRegion(g.wo, g.ho, 0, 0, g.wo, g.ho, &fr);
                uint32_t rg[10];
                fr.Args(rg);
                uint32_t a[30];
                uint32_t n = 0;
                // fuse_pack: the slot's packed rows (pack_pitch bytes, format code in the ox argument)
                const bool packed = fuse_pack != 0;
                const uint32_t h[20] = {R(kRoleCp), R(kRoleCc), R(kRoleMA), R(kRoleMB), R(kRoleW), R(kRoleUp),
                                        R(kRoleUc), R(kRoleLp), R(kRoleLc), packed ? pack_off : out_off,
                                        packed ? pack_pitch : out_stride, packed ? fuse_pack : 0u, 0, g.wo, g.ho,
                                        g.w1, g.h1, fb(op.t), op.flag1 ? 1u : 0u, op.flag2 ? 1u : 0u};
                for (uint32_t x : h) a[n++] = x;
                for (uint32_t x : rg) a[n++] = x;
                Dispatch(r, packed ? kLiteBlendPack : kLiteBlend, groups_for(static_cast<uint64_t>(g.wo) * g.ho), a, n);
                return true;
            }
            case FgOp::FullBlend:
                D(r, kFullBlend, (g.Wp / 16) * (g.Hp / 16),
                  {R(kRoleCp), R(kRoleCc), R(kRoleW), R(kRoleQc), R(kRoleMA), R(kRoleMB), op.ui ? R(kRoleUp) : 0u,
                   op.ui ? R(kRoleUc) : 0u, x0, x1, xw, ib, ic, ia, g.wo, g.ho, g.w1, g.h1, g.wd, g.hd, g.k,
                   g.Wp / 16, op.ui ? 1u : 0u});
                return true;
            case FgOp::Net2:
                if (fast())
                    RecordNet2Fast(r, op.t, false);
                else
                    RecordNet2(r, op.t);
                return true;
            case FgOp::OutPyramid:
                RecordOutPyramid(r);
                return true;
            case FgOp::MenuOutput: {
                uint32_t a[39];
                uint32_t n = 0;
                const uint32_t h[14] = {detres_off, col, ctw, cr.region.bx, cr.region.by, out_off, out_stride, 0, 0,
                                        g.wo, g.ho, hud, hw, hh};
                for (uint32_t x : h) a[n++] = x;
                for (uint32_t x : hreg) a[n++] = x;
                const uint32_t h2[5] = {aid, aw, ah, f.alpha_kind, f.alpha_channel};
                for (uint32_t x : h2) a[n++] = x;
                for (uint32_t x : areg) a[n++] = x;
                Dispatch(r, kMenuOutput, groups_for(static_cast<uint64_t>(g.wo) * g.ho), a, n);
                return true;
            }
            case FgOp::FinalCopy:
                D(r, kCopy4, groups_for(static_cast<uint64_t>(g.wo) * g.ho),
                  {out_off, out_stride, 0, 0, col, ctw, cr.region.bx, cr.region.by, g.wo, g.ho});
                return true;
        }
        return true;
    }

    void Conv(Rec& r, uint32_t layer, const Net2Buf& x, const Net2Buf& y, uint32_t in_cs, uint32_t in_co,
              uint32_t cin, uint32_t co_lo, uint32_t co_n, uint32_t out_cs, uint32_t out_co, uint32_t H, uint32_t W,
              uint32_t pool, uint32_t leaky, uint32_t clamp_from, uint32_t res, uint32_t res_mul, uint32_t skip,
              uint32_t skip_cs, bool barrier = true) {
        const FgNet2Layer& l = FgNet2Layers()[layer - 1];
        const uint64_t threads = static_cast<uint64_t>(pool ? H / 2 : H) * (pool ? W / 2 : W) * co_n;
        D(r, kNet2Conv, groups_for(threads),
          {x.off, n2w[layer - 1], n2b[layer - 1], y.off, skip, in_cs, in_co, cin, l.cin, co_lo, co_n, out_cs, out_co,
           H, W, pool, leaky, clamp_from, res, res_mul, skip_cs, 0u}, barrier);
    }
    // conv(n, src, dst, cin, ...) of the reference chain: the layer's full output, input channels from 0
    void ConvL(Rec& r, uint32_t layer, const Net2Buf& src, const Net2Buf& dst, uint32_t cin, uint32_t pool,
               uint32_t clamp_from, uint32_t res, const Net2Buf* skip) {
        const FgNet2Layer& l = FgNet2Layers()[layer - 1];
        Conv(r, layer, src, dst, src.c, 0, cin, 0, l.cout, l.cout, 0, src.h, src.w, pool, 1, clamp_from, res, kHalfOne,
             skip ? skip->off : 0u, skip ? cin : 0u);
    }
    void Up2(Rec& r, const Net2Buf& x, const Net2Buf& y, bool barrier = true) {
        D(r, kNet2Up2, groups_for(4ull * x.h * x.w * x.c), {x.off, y.off, x.c, x.h, x.w}, barrier);
    }

    void MergeNet2Inputs(Rec& r, uint32_t npx, float f) {
        if (pso[kNet2MergePixels]) {
            D(r, kNet2MergePixels, groups_for(npx), {ib, ic, ia, nb_x.off, npx, fb(f)});
        } else {
            D(r, kNet2Merge, groups_for(10ull * npx), {ib, ic, ia, nb_x.off, npx, fb(f)});
        }
    }

    void RecordNet2(Rec& r, float f) {
        const uint32_t npx = g.W2 * g.H2;
        MergeNet2Inputs(r, npx, f);
        ConvL(r, 1, nb_x, nb_a1, 10, 1, 0, 0, nullptr);
        ConvL(r, 2, nb_a1, nb_a2, 32, 1, 0, 0, nullptr);
        ConvL(r, 3, nb_a2, nb_r0, 32, 1, 0, 0, nullptr);
        const Net2Buf* src = &nb_r0;
        for (uint32_t u = 0; u < 4; ++u) {
            ConvL(r, 4 + 2 * u, *src, nb_t[u], 64, 0, 0, 1, nullptr);
            ConvL(r, 5 + 2 * u, nb_t[u], nb_r[u], 64, 0, 0, 2, src);
            src = &nb_r[u];
        }
        Up2(r, nb_r[3], nb_r4u);
        ConvL(r, 12, nb_r4u, nb_g, 64, 0, 32, 0, nullptr);
        Up2(r, nb_g, nb_gu);
        const Net2Buf* heads[3] = {&nb_hA4, &nb_hA1, &nb_hA3};
        const uint32_t lo[3] = {0, 4, 5}, cnt[3] = {4, 1, 3}, cf[3] = {kNoClamp, 0, 0};
        for (uint32_t j = 0; j < 3; ++j)
            Conv(r, 13, nb_gu, *heads[j], 96, 32 * j, 32, lo[j], cnt[j], cnt[j], 0, nb_gu.h, nb_gu.w, 0, 0, cf[j], 0,
                 kHalfOne, 0, 0, j == 0);
        Up2(r, nb_hA4, nb_U4);
        Up2(r, nb_hA1, nb_U1, false);
        Up2(r, nb_hA3, nb_U3, false);
        D(r, kNet2Warp, groups_for(18ull * npx), {ib, ic, ia, nb_U4.off, nb_U1.off, nb_U3.off, nb_y.off, g.W2, g.H2, fb(f)});
        ConvL(r, 14, nb_y, nb_b1, 18, 1, 0, 0, nullptr);
        ConvL(r, 15, nb_b1, nb_s0, 16, 1, 0, 0, nullptr);
        src = &nb_s0;
        for (uint32_t u = 0; u < 4; ++u) {
            ConvL(r, 16 + 2 * u, *src, nb_tb[u], 32, 0, 0, 1, nullptr);
            ConvL(r, 17 + 2 * u, nb_tb[u], nb_s[u], 32, 0, 0, 2, src);
            src = &nb_s[u];
        }
        Up2(r, nb_s[3], nb_s4u);
        ConvL(r, 24, nb_s4u, nb_e, 32, 0, kNoClamp, 0, nullptr);
        Up2(r, nb_e, nb_eu);
        const Net2Buf* us[3] = {&nb_U4, &nb_U1, &nb_U3};
        const uint32_t rm[3] = {kHalfTwo, kHalfOne, kHalfOne};
        for (uint32_t j = 0; j < 3; ++j)
            Conv(r, 25, nb_eu, nb_out, 48, 16 * j, 16, lo[j], cnt[j], 8, lo[j], g.H2, g.W2, 0, 0, cf[j], 2, rm[j],
                 us[j]->off, cnt[j], j == 0);
        ++counters.net2_exact;
    }

    // ---- net2, fast path (fg_net2_mma.py fast_chain_case) ------------------------------------------------------
    struct MOpt {
        uint32_t pool = 0, leaky = 1, clamp_from = 0, res = 0, res_mul = kHalfOne, up = 0, heads = 0;
        uint32_t co_n = 0;          // 0: the layer's outputs
        int32_t out_cs = -1;        // -1: co_n
        const Net2Buf* skip = nullptr;
        uint32_t skip_cs = 0;
        const Net2Buf* y1 = nullptr;
        const Net2Buf* y2 = nullptr;
        const Net2Buf* skip1 = nullptr;
        const Net2Buf* skip2 = nullptr;
    };
    uint32_t mconv_half_last = 0;  // Net2SelfTest negative control: the next MConv on half its workgroups
    // One fg_net2_mconv launch: layer n from x into y, cin input channels, conv grid H x W.
    void MConv(Rec& r, uint32_t n, const Net2Buf& x, const Net2Buf& y, uint32_t cin, uint32_t H, uint32_t W,
               const MOpt& o) {
        const FgNet2Layer& l = FgNet2Layers()[n - 1];
        MconvVariant v{};
        uint32_t groups = 0, rows = 0;
        // Configure checked every layer's variant (FgNet2FastPlan)
        if (!ChooseMconv(l.cout, l.cin, o.up, static_cast<uint64_t>(H) * W, &v, &groups, &rows)) return;
        if (n == 12 && static_cast<uint64_t>(H) * W <= 8192) {
            // Two output tiles per wave give the smaller L12 grids more workgroups.
            v.kernel = kN2m2b4u;
            groups = l.cout / (2 * kMTile);
        }
        if (o.heads) {
            if (n == 25 && pso[kN2h3b1u]) {
                v.kernel = kN2h3b1u;
                groups = 1;
            } else {
                groups = 3;
            }
        }
        const uint32_t co_n = o.co_n ? o.co_n : l.cout;
        const uint32_t out_cs = o.out_cs < 0 ? co_n : static_cast<uint32_t>(o.out_cs);
        const uint32_t tx = cdiv(W, kMTile), ty = cdiv(H, rows);
        uint32_t wg = tx * ty * groups;
        if (mconv_half_last) {
            wg = std::max<uint32_t>(1, wg / 2);
            mconv_half_last = 0;
        }
        auto off = [](const Net2Buf* b) { return b ? b->off : 0u; };
        D(r, v.kernel, wg,
          {x.off, n2fw[n - 1], n2fb[n - 1], y.off, off(o.skip), off(o.y1), off(o.y2), off(o.skip1), off(o.skip2), x.c,
           0u, cin, H, W, 0u, co_n, out_cs, 0u, o.pool, o.leaky, o.clamp_from, o.res, o.res_mul, o.skip_cs, 0u, tx, ty,
           o.heads});
    }

    void RecordNet2Fast(Rec& r, float f, bool half_last) {
        const uint32_t npx = g.W2 * g.H2;
        MergeNet2Inputs(r, npx, f);
        MOpt pool;
        pool.pool = 1;
        MConv(r, 1, nb_x, nb_a1, 10, g.H2, g.W2, pool);
        MConv(r, 2, nb_a1, nb_a2, 32, nb_a1.h, nb_a1.w, pool);
        MConv(r, 3, nb_a2, nb_r0, 32, nb_a2.h, nb_a2.w, pool);
        const Net2Buf* src = &nb_r0;
        for (uint32_t u = 0; u < 4; ++u) {
            MOpt a;
            a.res = 1;
            MConv(r, 4 + 2 * u, *src, nb_t[u], 64, src->h, src->w, a);
            MOpt b;
            b.res = 2;
            b.skip = src;
            b.skip_cs = 64;
            MConv(r, 5 + 2 * u, nb_t[u], nb_r[u], 64, src->h, src->w, b);
            src = &nb_r[u];
        }
        MOpt l12;
        l12.up = 1;
        l12.clamp_from = 32;
        MConv(r, 12, nb_r[3], nb_g, 64, nb_g.h, nb_g.w, l12);
        MOpt l13;
        l13.up = 1;
        l13.heads = 1;
        l13.co_n = 8;
        l13.out_cs = 0;
        l13.leaky = 0;
        l13.clamp_from = 4;
        l13.y1 = &nb_hA1;
        l13.y2 = &nb_hA3;
        MConv(r, 13, nb_g, nb_hA4, 32, nb_hA4.h, nb_hA4.w, l13);
        D(r, kNet2Upwarp, std::max<uint32_t>(1, cdiv(npx, 256)),
          {ib, ic, ia, nb_hA4.off, nb_hA1.off, nb_hA3.off, nb_U4.off, nb_U1.off, nb_U3.off, nb_y.off, g.W2, g.H2,
           fb(f)});
        MConv(r, 14, nb_y, nb_b1, 18, g.H2, g.W2, pool);
        MConv(r, 15, nb_b1, nb_s0, 16, nb_b1.h, nb_b1.w, pool);
        src = &nb_s0;
        for (uint32_t u = 0; u < 4; ++u) {
            MOpt a;
            a.res = 1;
            MConv(r, 16 + 2 * u, *src, nb_tb[u], 32, src->h, src->w, a);
            MOpt b;
            b.res = 2;
            b.skip = src;
            b.skip_cs = 32;
            MConv(r, 17 + 2 * u, nb_tb[u], nb_s[u], 32, src->h, src->w, b);
            src = &nb_s[u];
        }
        MOpt l24;
        l24.up = 1;
        l24.clamp_from = kNoClamp;
        MConv(r, 24, nb_s[3], nb_e, 32, nb_e.h, nb_e.w, l24);
        MOpt l25;
        l25.up = 1;
        l25.heads = 1;
        l25.co_n = 8;
        l25.out_cs = 8;
        l25.leaky = 0;
        l25.clamp_from = 4;
        l25.res = 2;
        l25.res_mul = kHalfTwo;
        l25.skip = &nb_U4;
        l25.skip1 = &nb_U1;
        l25.skip2 = &nb_U3;
        if (half_last) mconv_half_last = 1;
        MConv(r, 25, nb_e, nb_out, 16, g.H2, g.W2, l25);
        ++counters.net2_fast;
    }

    void RecordOutPyramid(Rec& r) {
        const uint32_t tx = cdiv(g.wo, 16), ty = cdiv(g.ho, 16);
        const volatile float kf = static_cast<float>(g.k);
        const float rk = 1.0f / kf;
        // fuse_pack: the pull and push-fine kernels write the slot's packed rows (pack_off, pack_pitch bytes, format
        // code in the out_x argument) instead of the float16 image
        const bool packed = fuse_pack != 0;
        D(r, packed ? kOutPullPack : kOutPull, tx * ty,
          {x0, x1, xw, nb_out.off, sig, packed ? pack_off : out_off, fine, lv[1], lv[2], lv[3], lv[4], g.wo, g.ho,
           g.W2, g.H2, g.k, tx, packed ? pack_pitch : out_stride, packed ? fuse_pack : 0u, 0u, fb(rk)});
        const uint32_t lc = g.out_coarsest;
        if (lc == 8) {
            uint32_t w4, h4;
            FgExecOutLevelSize(g, 4, &w4, &h4);
            D(r, kOutPullMid, cdiv(w4, 16) * cdiv(h4, 16), {lv[4], lv[5], lv[6], lv[7], lv[8], w4, h4, cdiv(w4, 16), g.wo, g.ho});
        }
        for (uint32_t l = lc - 1; l >= 1; --l) {
            uint32_t wf, hf, wc, hc;
            FgExecOutLevelSize(g, l, &wf, &hf);
            FgExecOutLevelSize(g, l + 1, &wc, &hc);
            D(r, kOutPush, groups_for(static_cast<uint64_t>(wf) * hf),
              {lv[l], lv[l + 1], lv[l], wf, hf, wc, hc});
            if (l == 1) break;
        }
        uint32_t w1o, h1o;
        FgExecOutLevelSize(g, 1, &w1o, &h1o);
        D(r, packed ? kOutPushFinePack : kOutPushFine, groups_for(static_cast<uint64_t>(g.wo) * g.ho),
          {fine, lv[1], packed ? pack_off : out_off, g.wo, g.ho, w1o, h1o, packed ? pack_pitch : out_stride,
           packed ? fuse_pack : 0u, 0u});
    }

    // ---- inputs ----------------------------------------------------------------------------------------------
    bool CanTransitionInput(D3D12_RESOURCE_STATES state) const {
        constexpr uint32_t compute_states = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE |
            D3D12_RESOURCE_STATE_COPY_SOURCE | D3D12_RESOURCE_STATE_COPY_DEST |
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS | D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT |
            D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
        return list_type == D3D12_COMMAND_LIST_TYPE_DIRECT || !(state & ~compute_states);
    }

    FgExecRefusal TakeInput(const FgTextureRef& t, InputTex* out, std::string* w, const char* name) {
        if (!t.resource) {
            *w = std::string(name) + " missing";
            return FgExecRefusal::InputMissing;
        }
        ID3D12Resource* res = static_cast<ID3D12Resource*>(t.resource);
        const D3D12_RESOURCE_DESC d = res->GetDesc();
        if (d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || d.SampleDesc.Count != 1 || d.DepthOrArraySize != 1 ||
            t.subresource != 0 || (d.Flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE)) {
            *w = std::string(name) + ": not a single-sample 2D texture readable by shaders";
            return FgExecRefusal::InputFormat;
        }
        InputTex x;
        x.res = res;
        x.format = d.Format;
        x.view = view_format(d.Format, &x.channels);
        if (x.view == DXGI_FORMAT_UNKNOWN) {
            *w = std::string(name) + ": format " + std::to_string(static_cast<uint32_t>(d.Format)) + " unsupported";
            return FgExecRefusal::InputFormat;
        }
        x.w = static_cast<uint32_t>(d.Width);
        x.h = d.Height;
        x.state = static_cast<D3D12_RESOURCE_STATES>(t.state);
        if (NeedsTransition(x.state) && !CanTransitionInput(x.state)) {
            *w = std::string(name) + ": state transition requires a direct queue";
            return FgExecRefusal::InputFormat;
        }
        if (x.w > cap_w || x.h > cap_h) {
            *w = std::string(name) + ": texture larger than the capture area";
            return FgExecRefusal::InputExtent;
        }
        const uint32_t rw = t.rect.width ? t.rect.width : x.w - t.rect.x;
        const uint32_t rh = t.rect.height ? t.rect.height : x.h - t.rect.y;
        if (!FgExecMakeRegion(x.w, x.h, t.rect.x, t.rect.y, rw, rh, &x.region)) {
            *w = std::string(name) + ": region outside its texture";
            return FgExecRefusal::InputExtent;
        }
        *out = x;
        return FgExecRefusal::None;
    }

    void BindInputs(Rec& r, std::initializer_list<const InputTex*> inputs) {
        const uint32_t table = ++r.table;
        D3D12_CPU_DESCRIPTOR_HANDLE h = heap->GetCPUDescriptorHandleForHeapStart();
        h.ptr += static_cast<SIZE_T>(r.e->desc_base + kTableSize * table + 1) * desc_inc;
        D3D12_SHADER_RESOURCE_VIEW_DESC sv = {};
        sv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        sv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        sv.Texture2D.MipLevels = 1;
        for (const InputTex* t : inputs) {
            sv.Format = t ? t->view : DXGI_FORMAT_R16G16B16A16_FLOAT;
            device->CreateShaderResourceView(t ? t->res : nullptr, &sv, h);
            h.ptr += desc_inc;
        }
        SetTable(r, table);
    }

    void Capture(Rec& r, const InputTex& t, uint32_t dst, uint32_t kind) {
        BindInputs(r, {&t});
        D(r, kCapture, groups_for(static_cast<uint64_t>(t.w) * t.h), {dst, t.w, t.h, kind}, false);
        ++counters.captures;
    }

    static bool NeedsTransition(D3D12_RESOURCE_STATES s) {
        return s != D3D12_RESOURCE_STATE_COMMON && !(s & D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }

    void CaptureColour(Rec& r) {
        const InputTex& src = frame.colour;
        const bool transition = NeedsTransition(src.state);
        if (transition) Transition(r, src.res, src.state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        Capture(r, src, cap_colour, 4);
        if (transition) Transition(r, src.res, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, src.state);
        SetTable(r, 0);
    }

    // ---- slot output -----------------------------------------------------------------------------------------
    ID3D12Resource* DirectCopySlot(uint32_t slot) const {
        const InputTex& src = frame.colour;
        if (desc.keep_outputs || !target || !src.res || !CanTransitionInput(src.state) ||
            src.w != g.wo || src.h != g.ho || src.region.bx || src.region.by ||
            src.region.ew != g.wo || src.region.eh != g.ho)
            return nullptr;
        ID3D12Resource* dst = target->SlotResource(slot);
        if (!dst || dst == src.res) return nullptr;
        const auto in = src.res->GetDesc();
        const auto out = dst->GetDesc();
        if (out.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || out.Width != in.Width || out.Height != in.Height ||
            out.DepthOrArraySize != 1 || out.MipLevels != 1 || in.MipLevels != 1 ||
            out.SampleDesc.Count != 1 || out.SampleDesc.Quality != in.SampleDesc.Quality)
            return nullptr;
        // Capture linearizes an sRGB SRV, and its float16 conversion quiets signaling NaNs. Only the linear
        // UNORM formats round-trip every input code through capture and pack.
        const DXGI_FORMAT format = linear_copy_format(out.Format);
        if (src.view != format ||
            (format != DXGI_FORMAT_R8G8B8A8_UNORM && format != DXGI_FORMAT_B8G8R8A8_UNORM &&
             format != DXGI_FORMAT_R10G10B10A2_UNORM))
            return nullptr;
        return dst;
    }

    void CopySlot(Rec& r, ID3D12Resource* dst, char kind, uint64_t token, uint32_t index) {
        const InputTex& src = frame.colour;
        const auto out = dst->GetDesc();
        Transition(r, src.res, src.state, D3D12_RESOURCE_STATE_COPY_SOURCE);
        Transition(r, dst, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        r.cl->CopyResource(dst, src.res);
        r.written_slot = dst;
        Transition(r, src.res, D3D12_RESOURCE_STATE_COPY_SOURCE, src.state);
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp = {};
        fp.Footprint = {linear_copy_format(out.Format), g.wo, g.ho, 1, 0};
        RecordDump(r, dst, fp, 4, kind, token, index);
        Transition(r, dst, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
    }

    // The slot format code the last output kernel of this generated index (the Full path's output pyramid or the light
    // path's blend) can pack directly, else 0: that kernel is the last writer of the output image (no menu output,
    // prefill or copy after it), outputs are not kept for readback, the slot is an 8 / 10-bit UNORM image of the output
    // size and the packing kernels are present.
    uint32_t PackFusion(const std::vector<FgExecOp>& ops, uint32_t slot) const {
        if (desc.keep_outputs || !target) return 0;
        bool pyramid = false, blend = false;
        for (const FgExecOp& op : ops) {
            if (op.op == FgOp::OutPyramid) pyramid = true;
            else if (op.op == FgOp::LiteBlend) blend = true;
            else if (op.op == FgOp::MenuOutput || op.op == FgOp::Prefill || op.op == FgOp::FinalCopy) return 0;
        }
        if (pyramid ? !(pso[kOutPullPack] && pso[kOutPushFinePack]) : !(blend && pso[kLiteBlendPack])) return 0;
        ID3D12Resource* res = target->SlotResource(slot);
        if (!res) return 0;
        const D3D12_RESOURCE_DESC d = res->GetDesc();
        uint32_t bpp = 0;
        const int code_ = pack_code(d.Format, &bpp);
        if (d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || d.Width != g.wo || d.Height != g.ho || code_ <= 0)
            return 0;
        return static_cast<uint32_t>(code_);
    }

    FgExecResult WriteSlot(Rec& r, uint32_t slot, char kind, uint64_t token, uint32_t index) {
        const uint32_t fused = fuse_pack;  // consumed here, so a refused recording cannot leak it into the next slot
        fuse_pack = 0;
        if (!target) return Fail(FgExecRefusal::SlotTargetMissing, "no presenter bound to the executor");
        ID3D12Resource* res = target->SlotResource(slot);
        if (!res) return Fail(FgExecRefusal::SlotMissing, "slot " + std::to_string(slot) + " has no texture");
        const D3D12_RESOURCE_DESC d = res->GetDesc();
        uint32_t bpp = 0;
        const int code_ = pack_code(d.Format, &bpp);
        if (d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || d.Width != g.wo || d.Height != g.ho || code_ < 0)
            return Fail(FgExecRefusal::SlotFormat, "slot texture " + std::to_string(d.Width) + " x " +
                                                       std::to_string(d.Height) + " format " +
                                                       std::to_string(static_cast<uint32_t>(d.Format)));
        uint64_t src_off = out_off;
        uint32_t pitch = out_stride * 8;
        if (code_ != 0) {
            const bool pairs = pso[kPackPairs] != nullptr;
            const uint32_t columns = pairs ? cdiv(g.wo, 2) : g.wo;
            // the output pyramid already wrote the packed rows
            if (fused != static_cast<uint32_t>(code_))
                D(r, pairs ? kPackPairs : kPack, groups_for(static_cast<uint64_t>(columns) * g.ho),
                  {out_off, out_stride, pack_off, pack_pitch, g.wo, g.ho, static_cast<uint32_t>(code_)});
            src_off = pack_off;
            pitch = pack_pitch;
        }
        ArenaTo(r, D3D12_RESOURCE_STATE_COPY_SOURCE);
        Transition(r, res, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        D3D12_TEXTURE_COPY_LOCATION dst = {};
        dst.pResource = res;
        dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        dst.SubresourceIndex = 0;
        D3D12_TEXTURE_COPY_LOCATION src = {};
        src.pResource = arena.Get();
        src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        src.PlacedFootprint.Offset = src_off;
        src.PlacedFootprint.Footprint.Format = d.Format;
        src.PlacedFootprint.Footprint.Width = g.wo;
        src.PlacedFootprint.Footprint.Height = g.ho;
        src.PlacedFootprint.Footprint.Depth = 1;
        src.PlacedFootprint.Footprint.RowPitch = pitch;
        r.cl->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        r.written_slot = res;
        if (desc.keep_outputs) {
            PendingRb p;
            p.rec.kind = kind;
            p.rec.frame_token = token;
            p.rec.index = index;
            p.rec.width = g.wo;
            p.rec.height = g.ho;
            p.rec.slot_format = static_cast<uint32_t>(d.Format);
            p.rec.slot_pitch = static_cast<uint32_t>(align(static_cast<uint64_t>(bpp) * g.wo, 256));
            p.rec.colour_width = frame.colour.w;
            p.rec.colour_height = frame.colour.h;
            p.rec.menu_disable = frame.disable;
            const uint64_t img = 8ull * out_stride * g.ho, colb = 8ull * frame.colour.w * frame.colour.h,
                           slotb = static_cast<uint64_t>(p.rec.slot_pitch) * g.ho;
            p.image = Buffer(img, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
            p.colour = Buffer(colb, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
            p.slot = Buffer(slotb, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
            if (p.image && p.colour && p.slot) {
                r.cl->CopyBufferRegion(p.image.Get(), 0, arena.Get(), out_off, img);
                r.cl->CopyBufferRegion(p.colour.Get(), 0, arena.Get(), cap_colour, colb);
                Transition(r, res, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
                D3D12_TEXTURE_COPY_LOCATION s2 = {};
                s2.pResource = res;
                s2.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                D3D12_TEXTURE_COPY_LOCATION d2 = {};
                d2.pResource = p.slot.Get();
                d2.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                d2.PlacedFootprint.Footprint = src.PlacedFootprint.Footprint;
                d2.PlacedFootprint.Footprint.RowPitch = p.rec.slot_pitch;
                r.cl->CopyTextureRegion(&d2, 0, 0, 0, &s2, nullptr);
                Transition(r, res, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
                pending.push_back(std::move(p));
            }
        }
        RecordDump(r, res, src.PlacedFootprint, bpp, kind, token, index);
        Transition(r, res, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
        return FgExecResult::Ok;
    }

    FgExecResult Finish(Rec& r, FgReadyFence* ready) {
        uint64_t v = 0;
        const FgExecResult s = Submit(&r, &v);
        if (!v) {
            dumps.erase(std::remove_if(dumps.begin(), dumps.end(), [](const DumpRb& d) { return d.fence_value == 0; }),
                        dumps.end());
        }
        for (DumpRb& d : dumps)
            if (d.fence_value == 0) d.fence_value = v;
        if (s != FgExecResult::Ok) return s;
        if (r.direct && (!JoinGameQueue() || !SignalProducer(&v)))
            return QueueFailure("publish the game queue handoff");
        if (desc.keep_outputs)
            for (PendingRb& p : pending)
                if (p.rec.fence_value == 0) p.rec.fence_value = v;
        if (ready) {
            ready->fence = fence_id;
            ready->value = v;
        }
        return FgExecResult::Ok;
    }

    FgExecResult CopyFrameToSlot(uint32_t slot, char kind, uint64_t token, uint32_t index, FgReadyFence* ready) {
        ID3D12Resource* dst = DirectCopySlot(slot);
        const bool capture = !dst && !frame.colour_captured;
        Rec first;
        Rec after_capture;
        Rec* r = &first;
        FgExecResult s = Begin(r, dst || capture, !dst);
        if (s != FgExecResult::Ok) return s;
        if (kind == 'r') TimeStage(*r, FgExecStage::RealEndpoint);
        if (dst) {
            CopySlot(*r, dst, kind, token, index);
        } else {
            if (capture) {
                CaptureColour(*r);
                if (compute_queue) {
                    uint64_t v = 0;
                    s = Submit(r, &v);
                    if (s != FgExecResult::Ok) return s;
                    r = &after_capture;
                    s = Begin(r);
                    if (s != FgExecResult::Ok) return s;
                    if (kind == 'r') TimeStage(*r, FgExecStage::RealEndpoint);
                }
            }
            FgExecOp op;
            op.op = FgOp::FinalCopy;
            RecordOp(*r, op);
            s = WriteSlot(*r, slot, kind, token, index);
            if (s != FgExecResult::Ok) return s;
        }
        s = Finish(*r, ready);
        if (s == FgExecResult::Ok && capture) frame.colour_captured = true;
        return s;
    }
};

// ---- public ------------------------------------------------------------------------------------------------------

FgD3d12Executor::FgD3d12Executor(const FgD3d12ExecutorDesc& desc) : impl_(new Impl()) {
    impl_->desc = desc;
    char timing[2] = {};
    impl_->desc.timing |= GetEnvironmentVariableA("WRONGGPU_FG_TIMING", timing, sizeof(timing)) == 1 && timing[0] == '1';
    if (desc.weights) {
        impl_->weights = *desc.weights;
        impl_->have_weights = true;
    }
    impl_->desc.weights = nullptr;
}

FgD3d12Executor::~FgD3d12Executor() {
    if (impl_ && impl_->fence) {
        impl_->WaitIdle(impl_->desc.wait_ms);
        impl_->FlushDumps(true);
    }
}

FgExecRefusal FgD3d12Executor::LastRefusal() const { return impl_->refusal; }
const std::string& FgD3d12Executor::LastWhy() const { return impl_->why; }
const FgExecCounters& FgD3d12Executor::Counters() const { return impl_->counters; }
const FgExecGeometry& FgD3d12Executor::Geometry() const { return impl_->g; }
uint64_t FgD3d12Executor::ArenaBytes() const { return impl_->arena_bytes; }

FgExecResult FgD3d12Executor::Configure(const FgCreateDesc& cd, uint32_t /*history_banks*/) {
    Impl& m = *impl_;
    m.configured = false;
    m.frame = FrameCtx();
    // Checks that need no device come first: size, weights, then the device and queue, then the kernel files.
    FgExecGeometry g;
    std::string w;
    if (!FgExecMakeGeometry(cd.output_width, cd.output_height, &g, &w))
        return m.Fail(FgExecRefusal::SizeUnsupported, w);
    if (!m.have_weights) return m.Fail(FgExecRefusal::WeightsMissing, "no weights given");
    for (uint32_t i = 0; i < 17; ++i)
        if (m.weights.net1_fragments[i].empty() || m.weights.net1_bias[i].empty())
            return m.Fail(FgExecRefusal::WeightsInvalid, "net1 layer " + std::to_string(i + 1) + " empty");
    if (m.desc.enable_full && !m.weights.has_net2)
        return m.Fail(FgExecRefusal::WeightsMissing, "the Full path needs net2's weights");
    if (m.fast()) {
        std::vector<FgNet2FastLaunch> plan;
        if (!FgNet2FastPlan(g.W2, g.H2, &plan))
            return m.Fail(FgExecRefusal::SizeUnsupported, "no fast net2 plan for this grid");
        for (uint32_t i = 0; i < kFgNet2Layers; ++i) {
            const FgNet2Layer& l = FgNet2Layers()[i];
            if (!FgNet2SwizzleFast(m.weights.net2_weights[i], m.weights.net2_bias[i], l.cout, l.cin, &m.n2frag[i],
                                   &m.n2fbias[i]))
                return m.Fail(FgExecRefusal::WeightsInvalid, "net2 layer " + std::to_string(i + 1) + " fragments");
        }
    }
    if (cd.optional[kFgOptDistortion] == FgOptionalDecl::Always && !m.desc.settings.ignore_distortion)
        return m.Fail(FgExecRefusal::DistortionUnsupported, "the session declares the distortion field always");
    if (!cd.device) return m.Fail(FgExecRefusal::NullDevice, "no device");
    if (!cd.queue) return m.Fail(FgExecRefusal::NullQueue, "no queue");
    if (!m.pipelines_built) {
        const FgExecResult k = m.LoadKernels();
        if (k != FgExecResult::Ok) return k;
    }
    ID3D12Device* dev = static_cast<ID3D12Device*>(cd.device);
    ID3D12CommandQueue* q = static_cast<ID3D12CommandQueue*>(cd.queue);
    if (m.device && m.device.Get() != dev) return m.Fail(FgExecRefusal::BadArgument, "another device than before");
    const D3D12_COMMAND_QUEUE_DESC queue_desc = q->GetDesc();
    if (queue_desc.Type != D3D12_COMMAND_LIST_TYPE_DIRECT && queue_desc.Type != D3D12_COMMAND_LIST_TYPE_COMPUTE)
        return m.Fail(FgExecRefusal::NullQueue, "the queue is neither a direct nor a compute queue");
    if (!m.WaitIdle(m.desc.wait_ms)) return m.Fail(FgExecRefusal::WaitTimeout, "reconfigure while work is in flight");
    m.FlushDumps(false);
    m.device = dev;
    m.queue = q;
    m.list_type = queue_desc.Type;
    m.entries.clear();
    m.direct_entries.clear();
    m.heap.Reset();
    m.producer_queries = {};
    m.direct_queries = {};
    m.timings = {};
    m.timing_failed = false;
    m.compute_queue.Reset();
    m.handoff_fence.Reset();
    m.handoff_value = m.joined_handoff = m.arena_value = 0;
    m.compute_slot_values.clear();
    char async_compute[2] = {};
    const bool async_off = GetEnvironmentVariableA("WRONGGPU_FG_ASYNC_COMPUTE", async_compute, sizeof(async_compute)) ==
                               1 && async_compute[0] == '0';
    if (m.list_type == D3D12_COMMAND_LIST_TYPE_DIRECT && !async_off) {
        D3D12_COMMAND_QUEUE_DESC compute_desc = {};
        compute_desc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
        compute_desc.NodeMask = queue_desc.NodeMask;
        if (FAILED(dev->CreateCommandQueue(&compute_desc, IID_PPV_ARGS(&m.compute_queue))))
            return m.Fail(FgExecRefusal::AllocationFailed, "compute queue");
        if (FAILED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m.handoff_fence))))
            return m.Fail(FgExecRefusal::FenceFailed, "game queue handoff fence");
    }
    if (!m.pipelines_built) {
        const FgExecResult p = m.BuildPipelines();
        if (p != FgExecResult::Ok) return p;
    }
    FgExecResult s = m.CreateObjects();
    if (s != FgExecResult::Ok) return s;
    m.create = cd;
    m.g = g;
    m.cap_w = std::max(cd.output_width, cd.render_max_width);
    m.cap_h = std::max(cd.output_height, cd.render_max_height);
    s = m.Layout();
    if (s != FgExecResult::Ok) return s;
    s = m.FillWeightImage();
    if (s != FgExecResult::Ok) return s;
    m.arena = m.Buffer(m.arena_bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                       D3D12_RESOURCE_STATE_COMMON);
    if (!m.arena) return m.Fail(FgExecRefusal::AllocationFailed, "arena of " + std::to_string(m.arena_bytes) + " bytes");
    s = m.Upload();
    if (s != FgExecResult::Ok) return s;
    m.host.Configure(m.desc.settings, g);
    m.configured = true;
    m.refusal = FgExecRefusal::None;
    m.why.clear();
    ++m.counters.configures;
    return FgExecResult::Ok;
}

bool FgD3d12Executor::BindSlotTarget(IFgSlotTarget* target) {
    Impl& m = *impl_;
    if (!target || !m.fence) {
        m.Fail(FgExecRefusal::SlotTargetMissing, "no slot target or no fence");
        return false;
    }
    std::string w;
    const uint32_t id = target->RegisterFence(m.fence.Get(), &w);
    if (!id) {
        m.Fail(FgExecRefusal::FenceFailed, "the presenter refused the producer fence: " + w);
        return false;
    }
    m.target = target;
    m.fence_id = id;
    return true;
}

FgExecResult FgD3d12Executor::RecordSharedGameFrame(const FgSharedDispatchDesc& d, FgSharedResult* out) {
    Impl& m = *impl_;
    if (!m.configured) return m.Fail(FgExecRefusal::NotConfigured, "Configure has not succeeded");
    if (!d.packet || !out) return m.Fail(FgExecRefusal::BadArgument, "null packet / result");
    if (m.DeviceRemoved()) return m.Fail(FgExecRefusal::DeviceLost, "device removed");
    m.FlushDumps(false);
    m.ArmDump(d.frame_token, d.generates);
    const FgFramePacket& p = *d.packet;
    FrameCtx f;
    f.token = d.frame_token;
    f.preset = d.preset;
    f.lite = d.preset == FgPreset::C;
    if (d.preset != FgPreset::A && d.preset != FgPreset::B && d.preset != FgPreset::C)
        return m.Fail(FgExecRefusal::PresetUnsupported, "preset outside A / B / C");
    if (!f.lite && !m.desc.enable_full) return m.Fail(FgExecRefusal::PresetUnsupported, "the Full path is disabled");
    if ((p.optional_present & FgOptBit(kFgOptDistortion)) && p.optional[kFgOptDistortion].resource &&
        !m.desc.settings.ignore_distortion)
        return m.Fail(FgExecRefusal::DistortionUnsupported, "the frame carries a distortion field");
    std::string w;
    FgExecRefusal rr = m.TakeInput(p.color, &f.colour, &w, "colour");
    if (rr == FgExecRefusal::None) rr = m.TakeInput(p.depth, &f.depth, &w, "depth");
    if (rr == FgExecRefusal::None) rr = m.TakeInput(p.motion, &f.motion, &w, "motion");
    if (rr == FgExecRefusal::None && (p.optional_present & FgOptBit(kFgOptHudless)) &&
        p.optional[kFgOptHudless].resource) {
        rr = m.TakeInput(p.optional[kFgOptHudless], &f.hudless, &w, "hudless colour");
        f.hud = rr == FgExecRefusal::None;
    }
    if (rr == FgExecRefusal::None && (p.optional_present & FgOptBit(kFgOptUiAlpha)) &&
        p.optional[kFgOptUiAlpha].resource) {
        rr = m.TakeInput(p.optional[kFgOptUiAlpha], &f.alpha, &w, "UI alpha");
        if (rr == FgExecRefusal::None) {
            f.ui1 = f.alpha.channels == 1;
            f.ui4 = f.alpha.channels == 4;
            if (!f.ui1 && !f.ui4) {
                rr = FgExecRefusal::InputFormat;
                w = "UI alpha: one or four channels";
            }
            f.alpha_kind = f.ui1 ? 1u : 4u;
            f.alpha_channel = f.ui1 ? 0u : 3u;
        }
    }
    if (rr == FgExecRefusal::None && f.colour.channels != 4) {
        rr = FgExecRefusal::InputFormat;
        w = "colour: four channels";
    }
    if (rr == FgExecRefusal::None && f.hud && f.hudless.channels != 4) {
        rr = FgExecRefusal::InputFormat;
        w = "hudless colour: four channels";
    }
    if (rr == FgExecRefusal::None && f.motion.channels < kMotionMinChannels) {
        rr = FgExecRefusal::InputFormat;
        w = "motion: at least two channels";
    }
    if (rr == FgExecRefusal::None && (f.colour.region.ew != m.g.wo || f.colour.region.eh != m.g.ho)) {
        rr = FgExecRefusal::InputExtent;
        w = "the colour region's extent must be the output size";
    }
    if (rr != FgExecRefusal::None) return m.Fail(rr, w);
    f.inverted = (p.flags & kFgFlagDepthReversed) != 0;
    {
        const volatile float ew = static_cast<float>(f.motion.region.ew), eh = static_cast<float>(f.motion.region.eh);
        f.scx = p.motion_scale_x / ew;
        f.scy = p.motion_scale_y / eh;
    }
    std::memcpy(f.clip_to_prev, p.camera_current, sizeof(f.clip_to_prev));
    std::memcpy(f.prev_to_clip, p.camera_previous, sizeof(f.prev_to_clip));

    const InputTex* ins[5] = {&f.colour, &f.depth, &f.motion, f.hud ? &f.hudless : nullptr,
                             (f.ui1 || f.ui4) ? &f.alpha : nullptr};
    bool transition[5] = {};
    for (int i = 0; i < 5; ++i) {
        if (!ins[i]) continue;
        transition[i] = Impl::NeedsTransition(ins[i]->state);
        for (int j = 0; j < i; ++j) {
            if (!ins[j] || ins[i]->res != ins[j]->res) continue;
            if (ins[i]->state != ins[j]->state)
                return m.Fail(FgExecRefusal::BadArgument, "aliased inputs report different resource states");
            transition[i] = false;
            break;
        }
    }

    // the previous game frame's last-index part, then this frame's schedule
    std::vector<uint32_t> pairs;
    m.host.EndFrame(&pairs);
    for (size_t i = 0; i + 1 < pairs.size(); i += 2) std::swap(m.role_off[pairs[i]], m.role_off[pairs[i + 1]]);
    std::vector<FgExecOp> ops;
    const bool reset = !d.has_previous || (p.flags & kFgFlagReset) != 0;
    m.host.BeginFrame(f.lite, d.preset == FgPreset::B, f.hud, f.ui4, f.ui1, reset, &ops);
    m.frame = f;
    if (f.lite)
        ++m.counters.lite_frames;
    else
        ++m.counters.full_frames;

    // Menu detection and the corresponding generated MenuOutput both consume the captured inputs.
    const bool input_srvs = !m.host.Frame().detect_ran && !m.desc.keep_outputs &&
                            m.pso[m.g.d == 4 ? kDown4Srv : kDown2Srv] && m.pso[kMvecSrv];
    Rec inputs;
    FgExecResult s = m.Begin(&inputs, true);
    if (s != FgExecResult::Ok) return s;
    m.TimeStage(inputs, input_srvs ? FgExecStage::SharedPreNet1 : FgExecStage::Capture);
    size_t next_op = 0;
    if (input_srvs) {
        // History clears must precede downsample. Keep the input transitions around the two SRV users.
        while (ops[next_op].op != FgOp::Downsample) {
            if (!m.RecordOp(inputs, ops[next_op++])) return FgExecResult::Refused;
        }
    }
    for (int i = 0; i < 5; ++i)
        if (transition[i])
            m.Transition(inputs, ins[i]->res, ins[i]->state, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    if (input_srvs) {
        if (!m.RecordOp(inputs, ops[next_op++], true) || !m.RecordOp(inputs, ops[next_op++], true))
            return FgExecResult::Refused;
    } else {
        const uint32_t dst[5] = {m.cap_colour, m.cap_depth, m.cap_motion, m.cap_hud, f.ui1 ? m.cap_ui1 : m.cap_ui4};
        const uint32_t kind[5] = {4, 6, 5, 4, f.ui1 ? 1u : 4u};
        for (int i = 0; i < 5; ++i)
            if (ins[i]) m.Capture(inputs, *ins[i], dst[i], kind[i]);
        m.SetTable(inputs, 0);
    }
    for (int i = 0; i < 5; ++i)
        if (transition[i])
            m.Transition(inputs, ins[i]->res, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, ins[i]->state);
    Rec compute;
    Rec post_detect;
    Rec* r = &inputs;
    uint64_t v = 0, menu_value = 0;
    if (m.compute_queue) {
        s = m.Submit(&inputs, &v);
        if (s != FgExecResult::Ok) return s;
        r = &compute;
        s = m.Begin(r);
        if (s != FgExecResult::Ok) return s;
    } else if (!input_srvs) {
        m.Barrier(inputs);
    }
    for (; next_op < ops.size(); ++next_op) {
        const FgExecOp& op = ops[next_op];
        m.TimeStage(*r, op.op == FgOp::Net1 ? FgExecStage::Net1 :
                        op.op == FgOp::Net1Output ? FgExecStage::SharedPostNet1 : FgExecStage::SharedPreNet1);
        if (!m.RecordOp(*r, op)) return FgExecResult::Refused;
        if (op.op == FgOp::Detect) {
            m.ArenaTo(*r, D3D12_RESOURCE_STATE_COPY_SOURCE);
            r->cl->CopyBufferRegion(m.menu_rb.Get(), 0, m.arena.Get(), m.detres_off, 32);
            s = m.Submit(r, &menu_value);
            if (s != FgExecResult::Ok) return s;
            r = &post_detect;
            s = m.Begin(r);
            if (s != FgExecResult::Ok) return s;
        }
    }
    s = m.Submit(r, &v);
    if (s != FgExecResult::Ok) return s;
    out->menu.frame_token = d.frame_token;
    out->menu.detected = false;
    if (menu_value) {
        ++m.counters.menu_waits;
        if (!m.WaitValue(menu_value, m.desc.wait_ms)) {
            if (m.DeviceRemoved()) return m.Fail(FgExecRefusal::DeviceLost, "device removed");
            return m.Fail(FgExecRefusal::WaitTimeout, "menu decision");
        }
        void* ptr = nullptr;
        D3D12_RANGE rg = {0, 32};
        if (FAILED(m.menu_rb->Map(0, &rg, &ptr))) return m.Fail(FgExecRefusal::CommandFailed, "readback map");
        uint32_t words[8];
        std::memcpy(words, ptr, sizeof(words));
        D3D12_RANGE none = {0, 0};
        m.menu_rb->Unmap(0, &none);
        out->menu.detected = words[1] != 0;   // this frame's detection (the session latches it)
        m.frame.disable = words[4];
    }
    ++m.counters.shared;
    m.frame.valid = true;
    m.frame.colour_captured = !input_srvs;
    return FgExecResult::Ok;
}

FgExecResult FgD3d12Executor::RecordGeneratedIndex(const FgIndexDispatchDesc& d, FgReadyFence* ready) {
    Impl& m = *impl_;
    if (!m.configured) return m.Fail(FgExecRefusal::NotConfigured, "Configure has not succeeded");
    if (!ready) return m.Fail(FgExecRefusal::BadArgument, "null ready fence");
    if (!m.frame.valid || m.frame.token != d.frame_token || !m.host.InFrame())
        return m.Fail(FgExecRefusal::BadArgument, "no shared work recorded for this game frame");
    if (d.preset != m.frame.preset) return m.Fail(FgExecRefusal::BadArgument, "preset differs from the game frame's");
    if (!m.target) return m.Fail(FgExecRefusal::SlotTargetMissing, "no presenter bound to the executor");
    m.FlushDumps(false);
    m.ArmDumpOnGenerated(d.frame_token);
    std::vector<FgExecOp> ops;
    m.host.IndexOps(d.t, m.frame.hud, m.frame.ui4, false, &ops);
    m.fuse_pack = m.PackFusion(ops, d.slot.slot);
    const bool capture = !m.frame.colour_captured && std::any_of(ops.begin(), ops.end(), [](const FgExecOp& op) {
        return op.op == FgOp::FinalCopy || op.op == FgOp::Prefill;
    });
    if (capture && m.compute_queue) {
        Rec inputs;
        FgExecResult s = m.Begin(&inputs, true);
        if (s != FgExecResult::Ok) return s;
        m.TimeStage(inputs, FgExecStage::GenPreNet2);
        m.CaptureColour(inputs);
        uint64_t value = 0;
        s = m.Submit(&inputs, &value);
        if (s != FgExecResult::Ok) return s;
        m.frame.colour_captured = true;
    }
    Rec r;
    FgExecResult s = m.Begin(&r);
    if (s != FgExecResult::Ok) return s;
    FgExecStage stage = FgExecStage::GenPreNet2;
    if (capture && !m.frame.colour_captured) {
        m.TimeStage(r, stage);
        m.CaptureColour(r);
    }
    for (const FgExecOp& op : ops) {
        if (op.op == FgOp::Net2) stage = FgExecStage::Net2;
        if (op.op == FgOp::OutPyramid) stage = FgExecStage::OutPyramid;
        m.TimeStage(r, stage);
        if (!m.RecordOp(r, op)) return FgExecResult::Refused;
    }
    m.TimeStage(r, FgExecStage::PackCopy);
    s = m.WriteSlot(r, d.slot.slot, 'g', d.frame_token, d.index);
    if (s != FgExecResult::Ok) return s;
    s = m.Finish(r, ready);
    if (s == FgExecResult::Ok) {
        if (capture) m.frame.colour_captured = true;
        ++m.counters.indices;
    }
    return s;
}

FgExecResult FgD3d12Executor::RecordMenuCopy(const FgCopyDispatchDesc& d, FgReadyFence* ready) {
    Impl& m = *impl_;
    if (!m.configured) return m.Fail(FgExecRefusal::NotConfigured, "Configure has not succeeded");
    if (!ready) return m.Fail(FgExecRefusal::BadArgument, "null ready fence");
    if (!m.frame.valid || m.frame.token != d.frame_token)
        return m.Fail(FgExecRefusal::BadArgument, "no captured game frame for this token");
    const FgExecResult s = m.CopyFrameToSlot(d.slot.slot, 'm', d.frame_token, d.index, ready);
    if (s == FgExecResult::Ok) ++m.counters.copies;
    return s;
}

FgExecResult FgD3d12Executor::RecordRealEndpoint(const FgRealDispatchDesc& d, FgReadyFence* ready) {
    Impl& m = *impl_;
    if (!m.configured) return m.Fail(FgExecRefusal::NotConfigured, "Configure has not succeeded");
    if (!ready) return m.Fail(FgExecRefusal::BadArgument, "null ready fence");
    m.FlushDumps(false);
    uint32_t slot = d.slot.slot;
    bool write = !d.real_only && d.slot.reserved;
    const bool have_frame = m.frame.valid && m.frame.token == d.frame_token;
    // an adapter-side real-only frame (no shared work): the game's back buffer, when it is a colour image of the
    // output size the executor can read
    InputTex bb;
    bool from_bb = false;
    std::string bb_w;
    if (m.target && !have_frame && d.game_colour.resource) {
        from_bb = m.TakeInput(d.game_colour, &bb, &bb_w, "game colour") == FgExecRefusal::None && bb.channels == 4 &&
                  bb.region.ew == m.g.wo && bb.region.eh == m.g.ho;
    }
    if (d.real_only && m.target && (have_frame || from_bb)) {
        // a real-only present shows a slot image: reserve one and copy the game frame into it (no slot free: the
        // presenter answers the present itself)
        FgStatus rs = m.target->ReserveRealSlotStatus(&slot);
        if (rs == FgStatus::NoSlot) {
            const ULONGLONG until = GetTickCount64() + 100;
            while (rs == FgStatus::NoSlot && GetTickCount64() < until) {
                Sleep(1);
                rs = m.target->ReserveRealSlotStatus(&slot);
            }
        }
        write = rs == FgStatus::Ok;
        if (!write) m.why = std::string("ReserveRealSlot failed: ") + FgStatusName(rs);
    } else if (d.real_only && m.target && !write) {
        m.why = std::string("real-only skip: have_frame=") + (have_frame ? "1" : "0") +
                " res=" + (d.game_colour.resource ? "1" : "0") +
                " from_bb=" + (from_bb ? "1" : "0") + " (" + bb_w + ")";
    }
    if (write) {
        if (!have_frame && !from_bb)
            return m.Fail(FgExecRefusal::BadArgument, "no captured game frame for this token");
        if (from_bb) {
            FrameCtx f;
            f.token = d.frame_token;
            f.colour = bb;
            f.valid = true;
            m.frame = f;
        }
        const FgExecResult s = m.CopyFrameToSlot(slot, 'r', d.frame_token, 0, ready);
        if (s == FgExecResult::Ok) {
            ++m.counters.real_endpoints;
            if (from_bb) ++m.counters.game_colour_copies;
        }
        if (s != FgExecResult::Ok && from_bb) m.frame.valid = false;
        return s;
    }
    uint64_t v = 0;
    if (m.compute_queue && (!m.SignalHandoff(&v) || !m.JoinGameQueue()))
        return m.QueueFailure("publish the game queue handoff");
    if (!m.SignalProducer(&v)) return m.QueueFailure("queue signal");
    ready->fence = m.fence_id;
    ready->value = v;
    ++m.counters.real_endpoints;
    return FgExecResult::Ok;
}

FgExecResult FgD3d12Executor::Net2SelfTest(uint32_t seed, float t, uint32_t runs, bool half_last,
                                           std::vector<uint16_t>* out, double* gpu_ms) {
    Impl& m = *impl_;
    if (!m.configured) return m.Fail(FgExecRefusal::NotConfigured, "Configure has not succeeded");
    if (!m.desc.enable_full) return m.Fail(FgExecRefusal::PresetUnsupported, "the Full path is disabled");
    if (!out) return m.Fail(FgExecRefusal::BadArgument, "null output");
    if (half_last && !m.fast()) return m.Fail(FgExecRefusal::BadArgument, "the control is a fast-path launch");
    if (gpu_ms) *gpu_ms = -1.0;
    const uint64_t n2 = static_cast<uint64_t>(m.g.W2) * m.g.H2;
    // synthetic smooth inputs: per channel four sinusoids of 1 to 4 cycles across the grid, peak about 8
    std::vector<uint16_t> in(n2 * 9);
    uint32_t s = seed * 2654435761u + 0x9E3779B9u;
    auto rnd = [&s]() {
        s = s * 1664525u + 1013904223u;
        return static_cast<double>(s >> 8) / 16777216.0;
    };
    const uint32_t chans[3] = {4, 4, 1};
    size_t at = 0;
    for (uint32_t b = 0; b < 3; ++b) {
        std::vector<double> img(n2 * chans[b], 0.0);
        for (uint32_t c = 0; c < chans[b]; ++c)
            for (int k = 0; k < 4; ++k) {
                const double fy = 1 + 3 * rnd(), fx = 1 + 3 * rnd(), ph = 6.283185307179586 * rnd();
                for (uint32_t y = 0; y < m.g.H2; ++y)
                    for (uint32_t x = 0; x < m.g.W2; ++x)
                        img[(static_cast<size_t>(y) * m.g.W2 + x) * chans[b] + c] +=
                            2.0 * std::sin(6.283185307179586 * (fy * y / m.g.H2 + fx * x / m.g.W2) + ph);
            }
        for (double v : img) in[at++] = FgExecHalfBits(static_cast<float>(v));
    }
    ComPtr<ID3D12Resource> up = m.Buffer(in.size() * 2, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
                                         D3D12_RESOURCE_STATE_GENERIC_READ);
    const uint64_t out_bytes = 2ull * n2 * 8;
    ComPtr<ID3D12Resource> rb = m.Buffer(out_bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE,
                                         D3D12_RESOURCE_STATE_COPY_DEST);
    if (!up || !rb) return m.Fail(FgExecRefusal::AllocationFailed, "self-test buffers");
    void* p = nullptr;
    D3D12_RANGE none = {0, 0};
    if (FAILED(up->Map(0, &none, &p))) return m.Fail(FgExecRefusal::AllocationFailed, "self-test map");
    std::memcpy(p, in.data(), in.size() * 2);
    up->Unmap(0, nullptr);

    Rec r;
    FgExecResult st = m.Begin(&r);
    if (st != FgExecResult::Ok) return st;
    m.ArenaTo(r, D3D12_RESOURCE_STATE_COPY_DEST);
    r.cl->CopyBufferRegion(m.arena.Get(), m.ib, up.Get(), 0, n2 * 8);
    r.cl->CopyBufferRegion(m.arena.Get(), m.ic, up.Get(), n2 * 8, n2 * 8);
    r.cl->CopyBufferRegion(m.arena.Get(), m.ia, up.Get(), n2 * 16, n2 * 2);
    const uint32_t sentinel[4] = {0x5A5A, 0x5A5A, 0x5A5A, 0x5A5A};
    m.Fill16(r, m.nb_out.off, out_bytes, sentinel, 1);
    if (m.fast())
        m.RecordNet2Fast(r, t, half_last);
    else
        m.RecordNet2(r, t);
    m.ArenaTo(r, D3D12_RESOURCE_STATE_COPY_SOURCE);
    r.cl->CopyBufferRegion(rb.Get(), 0, m.arena.Get(), m.nb_out.off, out_bytes);
    uint64_t v = 0;
    st = m.Submit(&r, &v);
    if (st != FgExecResult::Ok) return st;
    if (!m.WaitValue(v, m.desc.wait_ms * 8)) return m.Fail(FgExecRefusal::WaitTimeout, "self-test run");
    if (FAILED(rb->Map(0, nullptr, &p))) return m.Fail(FgExecRefusal::CommandFailed, "self-test readback");
    out->resize(n2 * 8);
    std::memcpy(out->data(), p, out_bytes);
    rb->Unmap(0, &none);

    if (runs == 0 || !gpu_ms) return FgExecResult::Ok;
    UINT64 freq = 0;
    if (FAILED(m.ProducerQueue()->GetTimestampFrequency(&freq)) || freq == 0) return FgExecResult::Ok;
    D3D12_QUERY_HEAP_DESC qd = {};
    qd.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    qd.Count = 2;
    ComPtr<ID3D12QueryHeap> qh;
    ComPtr<ID3D12Resource> qrb = m.Buffer(16, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE,
                                          D3D12_RESOURCE_STATE_COPY_DEST);
    if (FAILED(m.device->CreateQueryHeap(&qd, IID_PPV_ARGS(&qh))) || !qrb) return FgExecResult::Ok;
    double total = 0;
    // recordings that only warm the clocks up (back to back in one list; the exact path's are long: two)
    const uint32_t warm = m.fast() ? 8u : 2u;
    for (uint32_t i = 0; i <= runs; ++i) {  // i = 0: the warm-up list
        Rec rt;
        st = m.Begin(&rt);
        if (st != FgExecResult::Ok) return st;
        m.ArenaTo(rt, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        for (uint32_t k = 0; i == 0 && k < warm; ++k) {
            if (m.fast())
                m.RecordNet2Fast(rt, t, false);
            else
                m.RecordNet2(rt, t);
        }
        rt.cl->EndQuery(qh.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
        if (m.fast())
            m.RecordNet2Fast(rt, t, false);
        else
            m.RecordNet2(rt, t);
        rt.cl->EndQuery(qh.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
        rt.cl->ResolveQueryData(qh.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, qrb.Get(), 0);
        st = m.Submit(&rt, &v);
        if (st != FgExecResult::Ok) return st;
        if (!m.WaitValue(v, m.desc.wait_ms * 8)) return m.Fail(FgExecRefusal::WaitTimeout, "self-test timing");
        uint64_t ts[2] = {};
        if (FAILED(qrb->Map(0, nullptr, &p))) return m.Fail(FgExecRefusal::CommandFailed, "timestamp readback");
        std::memcpy(ts, p, sizeof(ts));
        qrb->Unmap(0, &none);
        if (i > 0) total += static_cast<double>(ts[1] - ts[0]) * 1000.0 / static_cast<double>(freq);
    }
    *gpu_ms = total / runs;
    return FgExecResult::Ok;
}

uint64_t FgD3d12Executor::LastComputeValue() const { return impl_->next_value; }
uint64_t FgD3d12Executor::CompletedComputeValue() const {
    return impl_->fence ? impl_->fence->GetCompletedValue() : 0;
}

bool FgD3d12Executor::WaitForComputeValue(uint64_t value, uint32_t timeout_ms) {
    return impl_->fence && impl_->WaitValue(value, timeout_ms);
}

bool FgD3d12Executor::WaitIdle(uint32_t timeout_ms) {
    const bool idle = impl_->WaitIdle(timeout_ms);
    if (idle) impl_->FlushDumps(false);
    return idle;
}

bool FgD3d12Executor::TakeTimings(FgExecTimings* out, uint32_t timeout_ms) {
    Impl& m = *impl_;
    if (!out || !m.producer_queries.readback || (m.compute_queue && !m.direct_queries.readback) ||
        !WaitIdle(timeout_ms))
        return false;
    for (Entry& e : m.entries) m.CollectTimings(e, m.producer_queries);
    for (Entry& e : m.direct_entries) m.CollectTimings(e, m.direct_queries);
    if (m.timing_failed) return false;
    *out = m.timings;
    m.timings = {};
    return true;
}

bool FgD3d12Executor::TakeOutputs(std::vector<FgExecOutputRecord>* out) {
    Impl& m = *impl_;
    if (!out || !WaitIdle(m.desc.wait_ms * 4)) return false;
    for (Impl::PendingRb& p : m.pending) {
        void* ptr = nullptr;
        if (SUCCEEDED(p.image->Map(0, nullptr, &ptr))) {
            p.rec.image.resize(4ull * p.rec.width * p.rec.height);
            const uint8_t* src = static_cast<const uint8_t*>(ptr);
            for (uint32_t y = 0; y < p.rec.height; ++y)
                std::memcpy(p.rec.image.data() + 4ull * p.rec.width * y, src + 8ull * m.out_stride * y,
                            8ull * p.rec.width);
            p.image->Unmap(0, nullptr);
        }
        if (SUCCEEDED(p.colour->Map(0, nullptr, &ptr))) {
            p.rec.colour.resize(4ull * p.rec.colour_width * p.rec.colour_height);
            std::memcpy(p.rec.colour.data(), ptr, p.rec.colour.size() * 2);
            p.colour->Unmap(0, nullptr);
        }
        if (SUCCEEDED(p.slot->Map(0, nullptr, &ptr))) {
            p.rec.slot.resize(static_cast<size_t>(p.rec.slot_pitch) * p.rec.height);
            std::memcpy(p.rec.slot.data(), ptr, p.rec.slot.size());
            p.slot->Unmap(0, nullptr);
        }
        out->push_back(std::move(p.rec));
    }
    m.pending.clear();
    return true;
}

// ---- presenter binding ---------------------------------------------------------------------------------------------

namespace {

template <class P>
class SlotTargetOf final : public IFgSlotTarget {
public:
    explicit SlotTargetOf(P* p) : p_(p) {}
    ID3D12Resource* SlotResource(uint32_t slot) const override { return p_->SlotResource(slot); }
    uint32_t RegisterFence(ID3D12Fence* fence, std::string* why) override {
        FgWinResult r;
        const uint32_t id = p_->RegisterFence(fence, &r);
        if (!id && why) *why = FgWinRefusalName(r.refusal);
        return id;
    }
    bool ReserveRealSlot(uint32_t* slot) override { return p_->ReserveRealSlot(slot) == FgStatus::Ok; }
    FgStatus ReserveRealSlotStatus(uint32_t* slot) override { return p_->ReserveRealSlot(slot); }

private:
    P* p_;
};

}  // namespace

FgExecutorPresenterSource::~FgExecutorPresenterSource() = default;

FgStatus FgExecutorPresenterSource::CreatePresenter(const FgCreateDesc& desc, std::unique_ptr<IFgPresenter>* out,
                                                    std::string* why) {
    if (!out || !inner_ || !executor_) return FgStatus::InvalidArgument;
    out->reset();
    std::unique_ptr<IFgPresenter> p;
    const FgStatus s = inner_->CreatePresenter(desc, &p, why);
    if (s != FgStatus::Ok || !p) return s == FgStatus::Ok ? FgStatus::Unsupported : s;
    std::unique_ptr<IFgSlotTarget> t;
    if (FgWinCompPresenter* c = dynamic_cast<FgWinCompPresenter*>(p.get())) {
        t.reset(new SlotTargetOf<FgWinCompPresenter>(c));
    } else if (FgDxgiPresenter* x = dynamic_cast<FgDxgiPresenter*>(p.get())) {
        t.reset(new SlotTargetOf<FgDxgiPresenter>(x));
    } else {
        if (why) *why = "the presenter has no D3D12 slots for the executor";
        return FgStatus::Unsupported;
    }
    if (!executor_->BindSlotTarget(t.get())) {
        if (why) *why = std::string("executor binding: ") + executor_->LastWhy();
        return FgStatus::Unsupported;
    }
    target_ = std::move(t);
    *out = std::move(p);
    return FgStatus::Ok;
}

}  // namespace wgfg
