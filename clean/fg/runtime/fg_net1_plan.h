// SPDX-License-Identifier: MIT
#pragma once
// Frame generation (issue #96): net1's fused tier B D3D12 plan (11 launches) for an in-game recorder.
//
// A C++ port of plan_fused in clean/fg/runner/fg_net1_mma.py: the five encoder convolutions (fg_mconv1/2/4 with the
// 2 x 2 max-pool in the epilogue), the bottleneck pair net1_06 + net1_07 (fg_mpair_b) and the five decoder levels
// (upscale + skip add + two 1 x 1 convolutions, fg_mpair_d4 .. d0). Buffers are float16 HWC tensors in the native
// arena; kernels address them by 32-bit arena byte offsets (clean/dx12/hlsl/wg_platform.hlsli: root constants b0 =
// [group base, workgroups, arguments...], the arena as root UAV u0 and its coherent alias u1).
//
// Buffers live in three regions: Input (the folded cells, written by the feature packer per index), Weights (the packed
// fragments and biases: immutable, once per device and model) and Scratch (the activations, the skips and "out").
// Inside a region every buffer starts on a 256-byte boundary (the native allocator's alignment), in the replay's
// allocation order. The plan is data; FgValidateNet1Plan re-derives every grid, shape, argument and offset
// independently, and FgRecordNet1 checks plan, pipelines and arena before it records anything: a refusal records
// nothing and there is no alternate dispatch.
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "fg_contract.h"

namespace wgfg {

enum class FgNet1Kernel : uint32_t {
    MConv1 = 0, MConv2, MConv4, MPairB, MPairD4, MPairD3, MPairD2, MPairD1, MPairD0,
};
inline constexpr uint32_t kFgNet1KernelCount = 9;
const char* FgNet1KernelName(FgNet1Kernel k);

enum class FgRegion : uint32_t { Input = 0, Weights = 1, Scratch = 2 };
inline constexpr uint32_t kFgRegionCount = 3;
inline constexpr uint64_t kFgArenaAlign = 256;
inline constexpr uint32_t kFgBlock = 256;            // threads per workgroup of every net1 kernel
inline constexpr uint32_t kFgMaxGroupsPerDispatch = 65535;
inline constexpr uint32_t kFgNet1Steps = 11;
inline constexpr uint32_t kFgNet1Args = 15;          // every fused kernel takes 15 argument DWORDs
inline constexpr uint32_t kFgCellChannels = 48;

struct FgBufferSpec {
    std::string name;     // the replay's buffer id (cells, wf_net1_NN, b_net1_NN, a0 .. p5, d5 .. d1, out)
    FgRegion region = FgRegion::Scratch;
    uint64_t bytes = 0;
    uint64_t offset = 0;  // inside its region
    uint32_t h = 0, w = 0, c = 0;  // tensor shape (HWC float16); weights: c = elements, h = w = 1
};

struct FgArg {
    bool buffer = false;  // true: the arena offset of buffers[index]; false: the value
    uint32_t value = 0;
};

struct FgNet1Step {
    FgNet1Kernel kernel = FgNet1Kernel::MConv1;
    uint32_t grid = 0;    // HIP workgroups (= D3D12 workgroups: block 256, wave 32)
    FgArg args[kFgNet1Args];
    uint32_t nouts = 0;
    uint32_t outs[2] = {};  // buffer indices written
};

struct FgNet1Plan {
    uint32_t h0 = 0, w0 = 0;
    std::vector<FgBufferSpec> buffers;
    std::vector<FgNet1Step> steps;
    uint64_t region_bytes[kFgRegionCount] = {};
    int Find(const std::string& name) const;
};

// The plan for a level-0 cell grid H0 x W0 (both >= 32 so that the five floor halvings stay >= 1).
FgRefusal FgBuildNet1Plan(uint32_t h0, uint32_t w0, FgNet1Plan* out);
// Independent re-derivation of every structural fact; FgRefusal::Net1PlanInvalid / Net1RegionOverlap /
// Net1RegionAlign with *why naming the first difference.
FgRefusal FgValidateNet1Plan(const FgNet1Plan& plan, std::string* why);

// The align-corners ratio (n_in - 1) / (n_out - 1) as single-precision bits (0 when n_out is 1).
uint32_t FgRatioBits(uint32_t n_in, uint32_t n_out);

// ---- recording -------------------------------------------------------------------------------------------------------

// The command surface net1 needs; the D3D12 implementation forwards to ID3D12GraphicsCommandList (root signature with
// the arena root UAVs, SetComputeRoot32BitConstants(2, ...), SetPipelineState, Dispatch, a global UAV barrier). Tests
// use a fake that only records the calls.
class IFgComputeRecorder {
public:
    virtual ~IFgComputeRecorder() = default;
    virtual void BindArena() = 0;
    virtual void UavBarrier() = 0;
    virtual void SetPipeline(FgNet1Kernel kernel) = 0;
    virtual void SetConstants(const uint32_t* dwords, uint32_t count) = 0;  // [group base, workgroups, args...]
    virtual void Dispatch(uint32_t groups_x) = 0;
};

struct FgPipelineSet {
    bool ready[kFgNet1KernelCount] = {};  // a pipeline state exists for the kernel on this device
};

struct FgArenaBinding {
    bool bound = false;
    uint64_t arena_bytes = 0;
    uint64_t region_base[kFgRegionCount] = {};  // byte offsets of the three regions in the arena
};

// Records net1 into `rec`, or refuses with nothing recorded. A launch above max_groups workgroups is split into
// dispatches of at most that many, each with its group base in constant 0 (the device's 65535 limit; tests lower it).
FgRefusal FgRecordNet1(const FgNet1Plan& plan, const FgPipelineSet& pipelines, const FgArenaBinding& arena,
                       IFgComputeRecorder& rec, std::string* why,
                       uint32_t max_groups = kFgMaxGroupsPerDispatch);

// ---- weight packing --------------------------------------------------------------------------------------------------

enum class FgChunkOrder : uint32_t { Ascending = 0, BlockKxKy = 1, KxBlockKy = 2 };

struct FgNet1Layer {
    const char* name;
    uint32_t cout, k, cin;
    FgChunkOrder order;
    bool leaky;
};
// net1's 17 convolutions in upload order (net1_01 .. net1_17).
const FgNet1Layer* FgNet1Layers();

// swizzle(): canonical float16 weights [cout][k][k][cin] (bit patterns) -> the A fragments in chunk order:
// [cout / 16][chunks][32 lanes][8], lane l holding row 16 mt + l % 16 and input channels 16 cb + 8 (l / 16) + 0..7.
bool FgSwizzleNet1(const uint16_t* canonical, uint32_t cout, uint32_t k, uint32_t cin, FgChunkOrder order,
                   std::vector<uint8_t>* out);

// One net1 layer from a validated model file (fg_model.h): the map's natural W ([out, in, 3, 3] or [out, in], float32
// exact in float16) to the canonical order, swizzled, and the bias as float16. False when the stage does not fit the
// layer (shape) or a value is not exact in float16.
struct FgModelStage;
bool FgPackNet1Layer(const uint8_t* model_bytes, size_t model_size, const FgModelStage& stage,
                     const FgNet1Layer& layer, std::vector<uint8_t>* fragments, std::vector<uint8_t>* bias);

}  // namespace wgfg
