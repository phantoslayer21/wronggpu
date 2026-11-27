// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// Frame generation (issue #96, F7b): FgD3d12Executor, the production D3D12 implementation of IFgKernelExecutor
// (fg_session.h). Windows only.
//
// Device side:
//  - One root signature for every FG kernel (clean/dx12/hlsl/wg_platform.hlsli): the arena as root UAV u0 and its
//    coherent alias u1, 59 root constants at b0 ([group base, workgroups, arguments...]) and a descriptor table with
//    the driver channel's raw UAV (u0, space 0x7FFF0ADE, clean/dx12/hlsl/wg_ags.hlsli) and four input texture
//    views (t0..t3). Kernels address the arena by 32-bit byte offsets; offset 0 is never a buffer (null).
//  - Configure (off the present path): loads the DXIL of every kernel it needs from FgD3d12ExecutorDesc::dxil_dir
//    (<dir>/<kernel>.dxil, built by clean/dx12/build_dx12.py --only), builds every pipeline once, sizes the arena for
//    the frame size (every light-path image, the pyramid, net1's three regions, the input capture areas, the generated
//    image, and with the Full path its images, net2's tensors, the output pyramid and the logistic table), uploads the
//    weights (net1's fragments, net2's canonical weights) and creates the shared producer fence, the command
//    allocators, the descriptor heap and the readback buffer for the menu decision.
//  - RecordSharedGameFrame: runs the history clears, R3 downsample / UI and R4 motion and depth. The SRV kernels read
//    the game textures on the game queue. Menu detection and output readbacks use arena captures. The remaining
//    work uses the arena: previous-to-current scatter, unpack, hole rule, pull / push, hole mask, and net1's input,
//    network and output (or the quality image without history). Menu detection waits for its decision readback.
//  - RecordGeneratedIndex: the per-index stages (intermediate-time scatters MA / MB, blend weights, then the light
//    blend (preset C) or the Full blend, net2 and the output pyramid (presets A / B), the menu output) into the
//    generated image, which is written into the presenter slot's texture (fg_exec_pack when the slot is not float16,
//    then a copy; the slot goes COMMON -> COPY_DEST -> COMMON), followed by a signal of the shared producer fence.
//  - RecordMenuCopy / RecordRealEndpoint: copy matching linear UNORM textures on the game queue, or capture colour
//    on demand and convert it to the slot format. An endpoint without a slot signals the game frame's queue work.
//  - Every refusal names its reason (LastRefusal / LastWhy); nothing falls back to another path.
// The executor reads game textures on the session's queue (FgCreateDesc::queue), ordered after the game's producers.
// A direct game queue hands arena work to a dedicated compute queue by default
// (WRONGGPU_FG_ASYNC_COMPUTE=0 selects the game queue for all work). Single caller (the session's present owner).
//
// Conventions the frame contract leaves to the executor (until the adapter contract names them):
//  - FgFramePacket::camera_current is the clip-to-previous-clip matrix and camera_previous the previous-clip-to-clip
//    matrix, 16 floats as the light path's scatter / unpack kernels take them (clean/fg/reference/scene.py matrices);
//    motion is in pixels of the motion texture's region, scaled by motion_scale_x / _y.
//  - The colour region's extent is the output size; the generated image is the output size at offset (0, 0).
//  - net2's scalar input f is the index's intermediate time t.
//  - A real-only interval shows the game frame from a slot the executor reserves (ReserveRealSlot) and fills with an
//    exact copy, when the session recorded that frame's shared work, or, for an adapter-side real-only frame (FG off,
//    menu flag, missing inputs), from the game's back buffer (FgRealDispatchDesc::game_colour) when it is a
//    four-channel image of the output size; otherwise only the fence is signalled.
//  - A frame that carries the distortion field is refused (not modelled) unless the setting ignores it.
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "fg_exec_plan.h"
#include "fg_model.h"
#include "fg_net1_plan.h"
#include "fg_session.h"

struct ID3D12Resource;
struct ID3D12Fence;

namespace wgfg {

// The presenter side the executor writes: a reserved slot's texture and the registration of the producer fence.
class IFgSlotTarget {
public:
    virtual ~IFgSlotTarget() = default;
    virtual ID3D12Resource* SlotResource(uint32_t slot) const = 0;
    // The fence id for FgReadyFence::fence (0: refused, *why says why).
    virtual uint32_t RegisterFence(ID3D12Fence* fence, std::string* why) = 0;
    // A slot for a real-only present (the executor copies the game frame into it); false when none is free.
    virtual bool ReserveRealSlot(uint32_t* slot) {
        (void)slot;
        return false;
    }
    virtual FgStatus ReserveRealSlotStatus(uint32_t* slot) {
        return ReserveRealSlot(slot) ? FgStatus::Ok : FgStatus::WrongState;
    }
    virtual void ReleaseRealSlot() {}
};

enum class FgExecRefusal : uint32_t {
    None = 0,
    NullDevice = 1,
    NullQueue = 2,
    DxilDirMissing = 3,
    DxilMissing = 4,
    DxilInvalid = 5,
    RootSignature = 6,
    PipelineFailed = 7,
    WeightsMissing = 8,
    WeightsInvalid = 9,
    SizeUnsupported = 10,
    AllocationFailed = 11,
    FenceFailed = 12,
    NotConfigured = 13,
    SlotTargetMissing = 14,
    SlotMissing = 15,
    SlotFormat = 16,
    InputMissing = 17,
    InputFormat = 18,
    InputExtent = 19,
    PresetUnsupported = 20,
    DistortionUnsupported = 21,
    CommandFailed = 22,
    WaitTimeout = 23,
    DeviceLost = 24,
    BadArgument = 25,
    Net1Refused = 26,
};
const char* FgExecRefusalName(FgExecRefusal r);

// Weights in the arena's layouts: net1's 17 layers as fragments (fg_net1_plan.h FgSwizzleNet1) and float16 biases;
// net2's 25 layers as canonical float16 [out][3][3][stored in] and float16 biases (clean/fg/reference/net2.py).
struct FgExecWeights {
    std::vector<uint8_t> net1_fragments[17], net1_bias[17];
    std::vector<uint8_t> net2_weights[kFgNet2Layers], net2_bias[kFgNet2Layers];
    bool has_net2 = false;
};
// From a model file that passed FgValidateModelBytes (the 17 net1 stages, then the 25 net2 stages); false with *why
// when a stage does not fit or a value is not exact in float16.
bool FgExecWeightsFromModel(const uint8_t* bytes, size_t size, const FgModelReport& report, FgExecWeights* out,
                            std::string* why);
// Synthetic weights (tests): uniform values scaled by the fan-in, from a seed.
void FgExecSyntheticWeights(uint32_t seed, FgExecWeights* out);

// The Full path's logistic table: sig(z) = 1 / (2^(z * -1.44269502) + 1) of every float16 bit pattern as float32
// (clean/fg/reference/stages_full.py sig_table: single precision, subnormals flushed).
void FgExecSigTable(std::vector<float>* out);

// net2's path inside the Full presets (A / B).
//  - Fast (default): 27 launches on the matrix unit, tier B (clean/dx12/kernels/fg_net2_mconv.hlsl: one matrix
//    instruction per chunk in net2's chunk order, the accumulator rounded to float16 after every chunk; the 2x
//    upsample of L12 / L13 / L24 / L25 staged inside the convolution, each head layer as one launch) and
//    fg_net2_upwarp.hlsl (head upsample + warp-merge, exact). Only the chunk sums differ from the CPU reference.
//  - Exact: the 38-launch chain of fg_net2_conv / _up2 / _merge / _warp, bit-exact to the CPU reference (validation).
enum class FgNet2Path : uint32_t { Fast = 0, Exact = 1 };
const char* FgNet2PathName(FgNet2Path p);

// One launch of the fast net2 plan for a net2 grid (CPU side, tested on its own): the baseline kernel variant for the
// layer (clean/fg/runner/fg_net2_mma.py variant()), its workgroup count and the conv grid.
struct FgNet2FastLaunch {
    uint32_t layer = 0;       // 1..25, 0: the input merge, 26: the upwarp
    const char* kernel = "";  // DXIL name
    uint32_t groups = 0;      // workgroups
    uint32_t H = 0, W = 0;    // conv grid
};
// The 27 launches for a W2 x H2 net2 grid (multiples of 16); false when a layer has no fast variant.
bool FgNet2FastPlan(uint32_t W2, uint32_t H2, std::vector<FgNet2FastLaunch>* out);
// net2's canonical float16 weights [cout][9][cw] -> the matrix unit's A fragments [cout / 16][9][cw / 16][32][8]
// (zero-padded to whole 16-output tiles and 16-channel blocks); the bias padded to whole tiles.
bool FgNet2SwizzleFast(const std::vector<uint8_t>& canonical, const std::vector<uint8_t>& bias, uint32_t cout,
                       uint32_t cw, std::vector<uint8_t>* fragments, std::vector<uint8_t>* bias_padded);

// The required DXIL files (without ".dxil") for these choices. The executor also loads available net1 tuning variants.
std::vector<std::string> FgD3d12ExecutorKernelNames(bool enable_full, FgNet2Path net2);

// Whether a game texture of this DXGI format is accepted as the motion-vector input: it must be readable as a colour
// view with at least two channels (x and y are read from the first two; a four-channel format is fine). A single
// channel, a depth format or an unknown format is not.
bool FgD3d12MotionFormatAccepted(uint32_t dxgi_format);

struct FgD3d12ExecutorDesc {
    std::string dxil_dir;                   // the folder holding <kernel>.dxil
    FgExecSettings settings;
    const FgExecWeights* weights = nullptr; // copied into the arena at Configure (may be released afterwards)
    bool enable_full = true;                // presets A / B: needs the Full / net2 kernels and net2's weights
    FgNet2Path net2 = FgNet2Path::Fast;     // net2's path in presets A / B (only that path's kernels are loaded)
    uint32_t wait_ms = 2000;                // bounded waits (menu decision, allocator reuse, idle)
    uint32_t allocators = 24;               // command allocators in the ring
    bool keep_outputs = false;              // tests: keep a host copy of every image written into a slot
    bool timing = false;                    // also enabled by WRONGGPU_FG_TIMING=1
    // Per-launch profile (TakeKernelProfile): one timestamp before every dispatch of every recording. It serialises
    // the queue a little, so it is a profile, not a frame time; also enabled by WRONGGPU_FG_KPROF=1 (implies timing).
    bool kernel_profile = false;
    // Turbo v2 levers (all exact: every one only changes where and when independent work runs). -1 takes the
    // environment (default on); 0 / 1 force it (tests, paired A / B runs).
    //  - batch (WG_FG_BATCH=0 off): RecordGeneratedBatch records layer k of every index back to back, so the barriers
    //    sit only between dependent layers (needs one scratch bank per index: max_generated banks).
    //  - hoist (WG_FG_HOIST=0 off): the per-index MA / MB fills and the intermediate-time scatters run on the second
    //    compute queue while the game frame's net1 runs (needs the second queue and banks).
    //  - dualq (WG_FG_DUALQ=0 off): the downsample runs on the game queue beside the motion / p2c chain on the
    //    producer queue (two recordings and a handoff each).
    //  - fuse (WG_FG_FUSE=0 off): the fused / retiled kernels fg_lite_p2c_fine_mask (fine pass + hole mask),
    //    fg_lite_p2c_inpaint_tiled and fg_lite_interm_both (both intermediate-time passes, LDS-binned scatters) replace
    //    their separate kernels when their DXIL is in the kernel folder.
    //  - fuse2 (WG_FG_FUSE2=0 off; Wave 1B, each exact against the Wave 1 kernels it replaces): 0 off, 1 (the default) the levers
    //    that won in paired timing (1 + 16: net1 cells and the pyramid tail), any other value a mask of levers (63 = every lever
    //    whose kernels are in the folder; 2, 4, 8, 32 are exact but cost GPU time): 1 net1_01 builds its cells in LDS
    //    (fg_mconv2_mid_cells, no net1 input launch), 2 net2's residual pairs in one launch (fg_net2_mpair_res_c64 / _c32),
    //    4 the upwarp computed in L14's staging (fg_net2_mconv_m1b2_uw), 8 the blend weights built in the blends
    //    (fg_lite_blend_w / _pack_w, fg_full_blend_*_w), 16 the output pyramid's coarse pushes in one workgroup
    //    (fg_out_tail) and level 1 pushed in the fine pass (fg_out_push_fine_l1), 32 X0 / X1 stored as RGB
    //    (fg_full_blend_*_x3, fg_out_pull_x3). Independent of fuse.
    //    Wave 2 (all bit-exact; 64 and 512 are in the default, the others are opt-in; WG_FG_FUSE2=17 is the Wave 1B default
    //    without them): 64 fg_lite_fill16_p2 (A15:
    //    the fill without integer divisions), 128 the bilinear taps of the blends, the light weights and the output pull as
    //    16 / 8-byte row loads (A16: the _b kernels), 256 the light blend on 32 x 8 tiles with 8-byte pair stores (A17: the _t
    //    kernels), 512 the output pull's division skipped on hole-free tiles (A17: the _q kernels), 1024 net2's convolutions of
    //    the indices of a batch as one (groups, 1, N) launch (A18: the _z kernels; needs batch; WG_FG_NET2_BATCH=0 turns it
    //    off whatever the mask says). The default mask is 593 = 17 + 64 + 512 (paired ABBA at 1440p: 128 and 256 regress, 1024
    //    is neutral); 2001 is 17 + all of Wave 2 and 2047 is every lever.
    int batch = -1, hoist = -1, dualq = -1, fuse = -1, fuse2 = -1;
    // Post-composite HUD restoration when a HUD-less buffer is tagged (-1: WG_FG_UI, default on; 0: off; 1: on).
    int ui_composite = -1;
    // N-B1: per-index fence completion for RecordGeneratedBatch (true: each index signals its own producer fence so
    // index 1 can present immediately without waiting for indices 2..N; false: single-submission interleaved batch).
    bool per_index_fences = true;
    // Developer frame dumps (off while dump_dir is empty; the ASI sets it only in a developer build): of the game
    // frames that generate, every dump_every-th one (at most dump_max), the game frame's slot image (game_<n>.ppm) and
    // each generated slot image (gen_<n>_<i>.ppm) are copied to a readback buffer after the slot write, and written as
    // 8-bit binary PPM files (fg_frame_dump.h) into dump_dir once their fence passed (checked at the next recording,
    // never waited for on the present path). Each file written, or not written, is one dump_log line.
    std::string dump_dir;
    uint32_t dump_every = 30, dump_max = 6;
    void (*dump_log)(void* ctx, const char* line) = nullptr;
    void* dump_log_ctx = nullptr;
};

// A host copy of one image the executor wrote (keep_outputs).
struct FgExecOutputRecord {
    char kind = 'g';             // 'g' generated index, 'm' menu / priming copy, 'r' real endpoint
    uint64_t frame_token = 0;
    uint32_t index = 0;
    uint64_t fence_value = 0;
    uint32_t width = 0, height = 0;
    std::vector<uint16_t> image;  // the generated image (float16 x 4, width x height, tightly packed)
    std::vector<uint16_t> colour; // the game frame's captured colour (float16 x 4, its texture extent)
    uint32_t colour_width = 0, colour_height = 0;
    uint32_t slot_format = 0;     // DXGI_FORMAT
    uint32_t slot_pitch = 0;
    std::vector<uint8_t> slot;    // the slot texture's bytes after the write (rows of slot_pitch)
    uint32_t menu_disable = 0;    // the frame's generation-disabled word
};

struct FgExecCounters {
    uint64_t configures = 0, shared = 0, indices = 0, copies = 0, real_endpoints = 0;
    uint64_t dispatches = 0, submissions = 0, menu_waits = 0, captures = 0;
    uint64_t lite_frames = 0, full_frames = 0;
    uint64_t game_colour_copies = 0;   // real-only frames copied from the game's back buffer (no shared work)
    uint64_t net2_fast = 0, net2_exact = 0;  // net2 recordings per path
    uint64_t net2_z_launches = 0, net2_z_saved = 0;   // A18: z-sliced net2 launches recorded, and the launches they replaced (nf - 1 each)
    uint64_t ui_composites = 0;        // generated frames composited with the post-composite HUD pass
};

enum class FgExecStage : uint32_t {
    Capture, SharedPreNet1, Net1, SharedPostNet1, GenPreNet2, Net2, OutPyramid, PackCopy, RealEndpoint, Count
};

struct FgExecTimings {
    // Sum of GPU timestamp intervals for each stage since the previous collection, in milliseconds.
    std::array<double, static_cast<size_t>(FgExecStage::Count)> ms = {};
    // The honest multi-queue numbers over the same window (every queue's recordings on one clock): the wall-clock
    // span from the first recording's start to the last one's end (idle gaps included), the time at least one
    // recording ran (the union of the recordings' intervals: overlap between queues counted once), and the number
    // of recordings. The per-stage sums above double count work that two queues overlap.
    double span_ms = 0.0, busy_ms = 0.0;
    uint32_t recordings = 0;
};

// One row of the per-launch profile (kernel_profile), summed over the window since the previous collection. Rows of
// one kernel are told apart by the ordinal of the launch inside its recording.
struct FgExecKernelProfileRow {
    std::string name;
    uint32_t ordinal = 0;   // n-th launch of this kernel in a recording (net1's repeated kernels: its layers in order)
    uint64_t calls = 0;     // launches summed
    uint64_t groups = 0;    // workgroups summed over the calls (groups / calls: the launch's workgroup count)
    double ms = 0.0;        // summed time from this launch's timestamp to the next one's
};

class FgD3d12Executor final : public IFgKernelExecutor {
public:
    explicit FgD3d12Executor(const FgD3d12ExecutorDesc& desc);
    ~FgD3d12Executor() override;
    FgD3d12Executor(const FgD3d12Executor&) = delete;
    FgD3d12Executor& operator=(const FgD3d12Executor&) = delete;

    FgExecResult Configure(const FgCreateDesc& desc, uint32_t history_banks) override;
    FgExecResult RecordSharedGameFrame(const FgSharedDispatchDesc& desc, FgSharedResult* out) override;
    FgExecResult RecordGeneratedIndex(const FgIndexDispatchDesc& desc, FgReadyFence* ready) override;
    FgExecResult RecordGeneratedBatch(const FgIndexDispatchDesc* descs, uint32_t n, FgReadyFence* ready,
                                      uint32_t* done) override;
    bool BatchesGenerated() const override;
    FgExecResult RecordMenuCopy(const FgCopyDispatchDesc& desc, FgReadyFence* ready) override;
    FgExecResult RecordRealEndpoint(const FgRealDispatchDesc& desc, FgReadyFence* ready) override;
    uint64_t LastComputeValue() const override;
    uint64_t CompletedComputeValue() const override;

    // The presenter the session created (FgExecutorPresenterSource binds it); registers the producer fence with it.
    bool BindSlotTarget(IFgSlotTarget* target);
    void UnbindSlotTarget();
    FgExecRefusal LastRefusal() const;
    const std::string& LastWhy() const;
    const FgExecCounters& Counters() const;
    const FgExecGeometry& Geometry() const;
    uint64_t ArenaBytes() const;
    // Bounded wait for a value on the producer fence.
    bool WaitForComputeValue(uint64_t value, uint32_t timeout_ms);
    bool GateGameQueueOnCompute(uint64_t value);
    bool PerIndexFences() const;
    // Bounded wait until every submission completed.
    bool WaitIdle(uint32_t timeout_ms);
    // Collect and clear the timings through the latest submission. Returns false when profiling is unavailable.
    bool TakeTimings(FgExecTimings* out, uint32_t timeout_ms);
    // kernel_profile: waits idle, collects and clears the per-launch profile (rows in first-seen order). Returns
    // false when it is off or unavailable.
    bool TakeKernelProfile(std::vector<FgExecKernelProfileRow>* out, uint32_t timeout_ms);
    // The levers that are active after Configure (a lever the device or the arena cannot take stays off).
    bool BatchActive() const;
    bool HoistActive() const;
    // The Wave 1B levers that run after Configure (bits as FgD3d12ExecutorDesc::fuse2 lists them: 1 net1 cells, 2 net2
    // pairs, 4 upwarp in L14, 8 blend weights, 16 pyramid tail, 32 X0 / X1 as RGB); 0 when off or their kernels are missing.
    uint32_t Fuse2Active() const;
    bool DualQueueActive() const;
    bool UiCompositeActive() const;
    uint32_t ScratchBanks() const;
    // keep_outputs: waits idle and moves the host copies out (oldest first).
    bool TakeOutputs(std::vector<FgExecOutputRecord>* out);

    // Diagnostics (configured executor with the Full path, off the present path): net2 alone on synthetic smooth
    // inputs from `seed` with scalar input t, on the configured path; its output (net2 grid x 8 float16, the output
    // buffer filled with 0x5A5A first) and, with runs > 0, the mean GPU time of `runs` timed recordings after one
    // warm-up (device timestamps around net2's launches; -1 when the queue has none). half_last: the last launch on
    // half of its workgroups (negative control). Waits idle; Refused with a reason otherwise.
    FgExecResult Net2SelfTest(uint32_t seed, float t, uint32_t runs, bool half_last, std::vector<uint16_t>* out,
                              double* gpu_ms);

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

// An IFgPresenterSource that creates the presenter through `inner` and binds the executor to it (the composition and
// the DXGI back ends of clean/fg/presenter). A presenter of another type, or a refused fence registration, refuses the
// creation with a reason; the presenter is released and nothing else is tried.
class FgExecutorPresenterSource final : public IFgPresenterSource {
public:
    FgExecutorPresenterSource(IFgPresenterSource* inner, FgD3d12Executor* executor)
        : inner_(inner), executor_(executor) {}
    ~FgExecutorPresenterSource() override;
    FgStatus CreatePresenter(const FgCreateDesc& desc, std::unique_ptr<IFgPresenter>* out, std::string* why) override;
    void ReleasePresenter() override;

private:
    IFgPresenterSource* inner_;
    FgD3d12Executor* executor_;
    std::unique_ptr<IFgSlotTarget> target_;
};

}  // namespace wgfg
