// SPDX-License-Identifier: MIT
#pragma once
// Frame generation (issue #96, F7b): FgD3d12Executor, the production D3D12 implementation of IFgKernelExecutor
// (fg_session.h). Windows only.
//
// Device side:
//  - One root signature for every FG kernel (clean/dx12/hlsl/wg_platform.hlsli): the arena as root UAV u0 and its
//    coherent alias u1, 59 root constants at b0 ([group base, workgroups, arguments...]) and a descriptor table with
//    the driver channel's raw UAV (u0, space 0x7FFF0ADE, clean/dx12/hlsl/wg_ags.hlsli) and one texture view (t0) for
//    the input capture. Kernels address the arena by 32-bit byte offsets; offset 0 is never a buffer (null).
//  - Configure (off the present path): loads the DXIL of every kernel it needs from FgD3d12ExecutorDesc::dxil_dir
//    (<dir>/<kernel>.dxil, built by clean/dx12/build_dx12.py --only), builds every pipeline once, sizes the arena for
//    the frame size (every light-path image, the pyramid, net1's three regions, the input capture areas, the generated
//    image, and with the Full path its images, net2's tensors, the output pyramid and the logistic table), uploads the
//    weights (net1's fragments, net2's canonical weights) and creates the shared producer fence, the command
//    allocators, the descriptor heap and the readback buffer for the menu decision.
//  - RecordSharedGameFrame: captures the game frame's colour, motion, depth and optional hudless colour / UI alpha
//    into the arena (fg_exec_io.hlsl fg_exec_capture), then runs the once-per-game-frame stages of fg_exec_plan.h's
//    schedule: history clears, menu detection, R3 downsample / UI, R4 motion and depth, the previous-to-current scatter
//    (signed 64-bit atomic minimum), unpack, hole rule, pull / push, the hole mask, and net1's input pack, the fused
//    11-launch net1 (fg_net1_plan.h FgRecordNet1) and its output, or the quality image without history. The capture
//    and the detection are submitted first; when a detector ran, the executor waits (bounded) for that submission
//    only and reads the detection back, so the session gets this frame's decision.
//  - RecordGeneratedIndex: the per-index stages (intermediate-time scatters MA / MB, blend weights, then the light
//    blend (preset C) or the Full blend, net2 and the output pyramid (presets A / B), the menu output) into the
//    generated image, which is written into the presenter slot's texture (fg_exec_pack when the slot is not float16,
//    then a copy; the slot goes COMMON -> COPY_DEST -> COMMON), followed by a signal of the shared producer fence.
//  - RecordMenuCopy / RecordRealEndpoint: the game frame's colour, exactly, into the slot; a real-only endpoint only
//    signals the fence after the game frame's work on the queue.
//  - Every refusal names its reason (LastRefusal / LastWhy); nothing falls back to another path.
// The executor records on the session's queue (FgCreateDesc::queue), so the game's work that produced the inputs is
// ordered before it. Single caller (the session's present owner). No lambda captures a copy that can throw.
//
// Conventions the frame contract leaves to the executor (until the adapter contract names them):
//  - FgFramePacket::camera_current is the clip-to-previous-clip matrix and camera_previous the previous-clip-to-clip
//    matrix, 16 floats as the light path's scatter / unpack kernels take them (clean/fg/reference/scene.py matrices);
//    motion is in pixels of the motion texture's region, scaled by motion_scale_x / _y.
//  - The colour region's extent is the output size; the generated image is the output size at offset (0, 0).
//  - net2's scalar input f is the index's intermediate time t.
//  - A real-only interval shows the game frame from a slot the executor reserves (ReserveRealSlot) and fills with an
//    exact copy, when the session recorded that frame's shared work; otherwise only the fence is signalled.
//  - A frame that carries the distortion field is refused (not modelled) unless the setting ignores it.
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

struct FgD3d12ExecutorDesc {
    std::string dxil_dir;                   // the folder holding <kernel>.dxil
    FgExecSettings settings;
    const FgExecWeights* weights = nullptr; // copied into the arena at Configure (may be released afterwards)
    bool enable_full = true;                // presets A / B: needs the Full / net2 kernels and net2's weights
    uint32_t wait_ms = 2000;                // bounded waits (menu decision, allocator reuse, idle)
    uint32_t allocators = 24;               // command allocators in the ring
    bool keep_outputs = false;              // tests: keep a host copy of every image written into a slot
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
    FgExecResult RecordMenuCopy(const FgCopyDispatchDesc& desc, FgReadyFence* ready) override;
    FgExecResult RecordRealEndpoint(const FgRealDispatchDesc& desc, FgReadyFence* ready) override;
    uint64_t LastComputeValue() const override;
    uint64_t CompletedComputeValue() const override;

    // The presenter the session created (FgExecutorPresenterSource binds it); registers the producer fence with it.
    bool BindSlotTarget(IFgSlotTarget* target);
    FgExecRefusal LastRefusal() const;
    const std::string& LastWhy() const;
    const FgExecCounters& Counters() const;
    const FgExecGeometry& Geometry() const;
    uint64_t ArenaBytes() const;
    // Bounded wait until every submission completed.
    bool WaitIdle(uint32_t timeout_ms);
    // keep_outputs: waits idle and moves the host copies out (oldest first).
    bool TakeOutputs(std::vector<FgExecOutputRecord>* out);

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

private:
    IFgPresenterSource* inner_;
    FgD3d12Executor* executor_;
    std::unique_ptr<IFgSlotTarget> target_;
};

}  // namespace wgfg
