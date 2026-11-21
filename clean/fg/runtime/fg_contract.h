// SPDX-License-Identifier: MIT
#pragma once
// Frame generation (issue #96): the versioned session / frame / result contract of the FG runtime.
//
// Every game adapter (a game-facing interposer, a host-owned provider, a fake host in tests) translates its own
// inputs into these structs. They are plain data with a leading size and version, so an older adapter's smaller struct
// is recognised and refused instead of being read past its end. Every refusal carries an explicit reason; nothing
// here falls back to another route. Plain C++, no Windows or D3D12 headers: resources are opaque pointers that the
// D3D12 side interprets, and their formats are DXGI_FORMAT values carried as integers.
#include <cstdint>

namespace wgfg {

inline constexpr uint32_t kFgAbiVersion = 1;
inline constexpr uint32_t kFgMaxGenerated = 5;  // N = 1..5 generated images per base interval (2x .. 6x)

// The release gate: no FG mode is advertised until the end-to-end oracle gate (F5) and a qualified presenter pass.
// FgAdvertisedMaxGenerated() reports 0 while this is false.
inline constexpr bool kFgModesQualified = false;

enum class FgPreset : uint32_t {
    A = 0,  // full blend, UI recomposition off
    B = 1,  // full blend, UI recomposition on
    C = 2,  // Lite blend, UI recomposition off (shares the model)
};
inline constexpr uint32_t kFgPresetCount = 3;

struct FgPresetGraph {
    bool full_blend;
    bool ui_recompose;
    bool lite_blend;
};
// The graph selection of a preset; false for a value outside A/B/C.
bool FgPresetGraphOf(FgPreset preset, FgPresetGraph* out);

// Optional inputs a session declares at create time. Always: every frame must carry it (missing = input error).
// May: a frame says per frame whether it carries it (a presence bit). Never: a frame must not carry it.
enum class FgOptionalDecl : uint32_t { Never = 0, May = 1, Always = 2 };
enum FgOptionalInput : uint32_t {
    kFgOptHudless = 0,
    kFgOptUiAlpha = 1,
    kFgOptDistortion = 2,
    kFgOptNoWarp = 3,
    kFgOptCount = 4,
};
inline constexpr uint32_t FgOptBit(FgOptionalInput i) { return 1u << static_cast<uint32_t>(i); }

// Every reason a create, a frame or a route can be refused. Stable numbers: they reach logs and the front end.
enum class FgRefusal : uint32_t {
    None = 0,
    // contract shape
    StructSize = 1,
    AbiVersion = 2,
    NullDevice = 3,
    NullQueue = 4,
    BadLuid = 5,
    BadOutputSize = 6,
    BadRenderBounds = 7,
    BadPreset = 8,
    BadGeneratedCount = 9,
    BadGeneratedIndex = 10,
    BadOptionalDecl = 11,
    BadFormat = 12,
    // frame identity and inputs
    TokenNotMonotonic = 20,
    ViewportMismatch = 21,
    LuidMismatch = 22,
    QueueEpochMismatch = 23,
    SwapchainEpochMismatch = 24,
    MissingColor = 25,
    MissingMotion = 26,
    MissingDepth = 27,
    MissingPromisedOptional = 28,
    UndeclaredOptional = 29,
    ConflictingTag = 30,
    StaleTag = 31,
    TagTokenMismatch = 32,
    TextureExtent = 33,
    RenderRectOutsideBounds = 34,
    OutputRectMismatch = 35,
    NoFrameOpen = 36,
    UnknownTag = 37,
    // history
    HistoryBankBusy = 40,
    HistoryNotPrimed = 41,
    IndexOutsideInterval = 42,
    IntervalMismatch = 43,
    // model and package
    ModelMissing = 50,
    ModelContainer = 51,
    ModelChecksum = 52,
    ModelMapHash = 53,
    ModelStageCount = 54,
    ModelStageName = 55,
    ModelTensorShape = 56,
    ModelTensorType = 57,
    ModelStageDigest = 58,
    ModelValue = 59,
    // route and plan
    InstallOff = 60,
    UnsupportedAdapter = 61,
    ModelNotValidated = 62,
    PresenterNotQualified = 63,
    RouteNotQualified = 64,
    Net1GridUnsupported = 65,
    Net1PipelineMissing = 66,
    Net1ArenaMissing = 67,
    Net1ArenaTooSmall = 68,
    Net1PlanInvalid = 69,
    Net1RegionOverlap = 70,
    Net1RegionAlign = 71,
    // worker
    WorkerStartFailed = 80,
    WorkerInitFailed = 81,
    WorkerAbandoned = 82,
};
const char* FgRefusalName(FgRefusal r);

struct FgLuid {
    uint32_t low = 0;
    int32_t high = 0;
};
inline bool operator==(const FgLuid& a, const FgLuid& b) { return a.low == b.low && a.high == b.high; }
inline bool operator!=(const FgLuid& a, const FgLuid& b) { return !(a == b); }

struct FgRect {
    uint32_t x = 0, y = 0, width = 0, height = 0;
};

// Session creation. One session per active viewport and swap chain.
struct FgCreateDesc {
    uint32_t struct_size = sizeof(FgCreateDesc);
    uint32_t abi_version = kFgAbiVersion;
    void* device = nullptr;           // the game's ID3D12Device
    void* queue = nullptr;            // its render / present queue
    FgLuid adapter_luid;
    uint64_t swapchain_id = 0;        // opaque identity of the chain (its canonical IUnknown)
    uint64_t window_id = 0;           // opaque identity of its window
    uint64_t queue_epoch = 0;         // epochs at create time (FgHistory::Rebind moves them)
    uint64_t swapchain_epoch = 0;
    uint32_t viewport_id = 0;
    uint32_t output_width = 0, output_height = 0;
    uint32_t output_format = 0;       // DXGI_FORMAT
    uint32_t render_min_width = 0, render_min_height = 0;
    uint32_t render_max_width = 0, render_max_height = 0;
    uint32_t dynamic_resolution = 0;  // 1: render size may change per frame inside [min, max]
    FgPreset preset = FgPreset::A;
    uint32_t max_generated = 1;       // 1..5
    FgOptionalDecl optional[kFgOptCount] = {};
    uint32_t hdr = 0;
    uint32_t presenter_mode = 0;      // reserved for the presenter contract (F6)
};

// One texture handed over for a frame: the resource plus what the adapter knows of its lifetime.
struct FgTextureRef {
    void* resource = nullptr;         // ID3D12Resource
    uint32_t format = 0;              // DXGI_FORMAT
    uint32_t width = 0, height = 0;   // resource extent
    uint32_t subresource = 0;
    uint32_t state = 0;               // D3D12_RESOURCE_STATES the producer leaves it in
    uint64_t producer_epoch = 0;      // producer queue / command-list epoch
    uint64_t frame_token = 0;         // the frame the tag belongs to
    uint32_t viewport_id = 0;
    FgRect rect;                      // the valid area inside the resource
};

enum FgFrameFlag : uint32_t {
    kFgFlagReset = 1u << 0,           // reset / camera cut
    kFgFlagMenu = 1u << 1,            // menu or other discontinuity: generation disabled for this frame
    kFgFlagDepthReversed = 1u << 2,
    kFgFlagMotionRenderRes = 1u << 3, // motion vectors at render (not output) resolution
    kFgFlagHdr = 1u << 4,
    kFgFlagRealOutput = 1u << 5,      // the host asks for the real image as an output too
};

// One base (game) frame, or one explicit per-index evaluation of it (generated_index 1..N).
struct FgFramePacket {
    uint32_t struct_size = sizeof(FgFramePacket);
    uint32_t abi_version = kFgAbiVersion;
    uint64_t frame_token = 0;
    uint32_t viewport_id = 0;
    FgLuid adapter_luid;
    uint64_t queue_epoch = 0;
    uint64_t swapchain_epoch = 0;
    FgTextureRef color;               // final backbuffer color (required)
    FgTextureRef motion;              // current-to-previous motion (required)
    FgTextureRef depth;               // current depth (required)
    FgTextureRef optional[kFgOptCount];
    uint32_t optional_present = 0;    // FgOptBit() per optional input carried
    FgRect render_rect;
    FgRect output_rect;
    float jitter_x = 0, jitter_y = 0;
    float motion_scale_x = 1, motion_scale_y = 1;
    float camera_current[16] = {};
    float camera_previous[16] = {};
    uint32_t flags = 0;               // FgFrameFlag
    FgPreset preset = FgPreset::A;
    uint32_t generated_count = 1;     // N, 1..max_generated
    uint32_t generated_index = 0;     // 0: the game frame (present-driven); 1..N: an explicit evaluator's index i
};

enum class FgStatus : uint32_t { Generated = 0, RealOnly = 1, Refused = 2 };

// Why a frame was shown real-only (observably distinct states).
enum class FgRealOnlyReason : uint32_t {
    None = 0,
    Prime = 1,          // first valid frame of a history epoch
    Reset = 2,          // reset / camera cut
    Menu = 3,           // menu or discontinuity flag
    Resize = 4,         // incompatible size or format change
    Gap = 5,            // a skipped frame token
    EpochChange = 6,    // device, queue or swap-chain epoch replaced
    HistoryBusy = 7,    // no free history bank (a bank is still read by compute)
    Disabled = 8,       // FG off / not qualified
};

struct FgFrameResult {
    uint32_t struct_size = sizeof(FgFrameResult);
    uint32_t abi_version = kFgAbiVersion;
    FgStatus status = FgStatus::Refused;
    FgRefusal refusal = FgRefusal::None;
    FgRealOnlyReason real_only = FgRealOnlyReason::None;
    FgPreset requested_preset = FgPreset::A, effective_preset = FgPreset::A;
    uint32_t requested_count = 0, effective_count = 0;
    uint64_t history_epoch = 0;
    uint64_t base_presents = 0;       // game presents accounted
    uint64_t generated_queued = 0;    // generated presents queued
    uint64_t displayed = 0;           // images actually displayed (presenter reports)
};

// ---- checks (plain functions; FgRefusal::None = accepted) ------------------------------------------------------------

FgRefusal FgValidateCreate(const FgCreateDesc& desc);
// Shape of a packet for a session: size/version, N and i ranges, preset. Identity and inputs: FgHistory (fg_history.h).
FgRefusal FgValidatePacketShape(const FgCreateDesc& session, const FgFramePacket& packet);

// t of generated index i of N: the single-precision quotient float(i) / (float(N) + 1), rounded once to nearest.
float FgIndexTime(uint32_t i, uint32_t n);
inline bool FgMultiFrame(uint32_t n) { return n > 1; }
inline bool FgLastIndex(uint32_t i, uint32_t n) { return i == n; }

// The release route: every fact must hold, checked in this order, first failure returned. No alternate route exists.
struct FgRouteFacts {
    bool install_on = false;
    bool adapter_supported = false;   // AMD RX 9000 / RDNA 4 with the required feature probes, by LUID
    bool model_validated = false;     // fg_model.h FgValidateModelFile passed for this device
    bool presenter_qualified = false; // a qualified presenter for this title and mode
    bool modes_qualified = kFgModesQualified;
};
FgRefusal FgCheckRoute(const FgRouteFacts& facts);
// The highest generated count the runtime advertises: 0 while the modes are not qualified or the route is refused.
uint32_t FgAdvertisedMaxGenerated(const FgRouteFacts& facts);

}  // namespace wgfg
