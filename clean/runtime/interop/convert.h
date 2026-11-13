// SPDX-License-Identifier: MIT
// D3D12 copy/convert passes of the interop (spec 10 section 6, spec 20 conversions), recorded into the game's
// command list:
//  - inputs:   the game's color, depth and motion textures (any typed format, active region) into the runtime's
//              dense float buffers (clean/FORMATS.md "Per-frame runtime": color 4 x f32, depth f32, motion 2 x f32 in
//              render pixels, m = motion_scale * raw + motion_offset), and the exposure texel when given as a texture;
//  - output:   the network's result buffer (O, 3 x f32 RGB) into the game's output texture (alpha 1, sRGB encoded for
//              sRGB formats);
//  - fallback: a bilinear upscale of the game's color into the output texture, for frames without a network result.
// HLSL compiled at run time with the Windows component d3dcompiler_47.dll (cs_5_0). The game's resource states are
// transitioned as needed and restored to the states the front end reported.
#pragma once

#include "d3d12_include.h"

#include <cstdint>
#include <string>

#include "../../fsr/include/wronggpu_upscaler.h"

namespace wgi {

// A runtime buffer: a range of a D3D12 buffer resource (the shared arena), starting at a byte offset that is a
// multiple of 64 KB (so of every element size used here).
struct BufferRange {
  ID3D12Resource* resource = nullptr;
  uint64_t offset = 0;
};

struct InputBuffers {
  BufferRange color;   // R32G32B32A32_FLOAT elements, >= rw * rh
  BufferRange depth;   // R32_FLOAT
  BufferRange motion;  // R32G32_FLOAT
  BufferRange expo;    // R32_FLOAT, 1 element (exposure texel)
};

class Converter {
 public:
  ~Converter();
  bool init(ID3D12Device* device, std::string* err);
  // Records the input conversion of frame f into list (barriers on the game's textures and back).
  void record_inputs(ID3D12GraphicsCommandList* list, const WgFrame& f, uint32_t create_flags, const InputBuffers& b);
  // Model K's compact input pass: dense rows of RGBA16F colour (rw * 8 bytes), R32F depth and the raw RG16F motion
  // texel (rw * 4 bytes; the network applies motion_scale / motion_offset), into the same buffers. Exact when
  // inputs16_ok: colour RGBA16F or R11G11B10, motion RG16F or RGBA16F (FP16-representable values).
  void record_inputs16(ID3D12GraphicsCommandList* list, const WgFrame& f, uint32_t create_flags,
                       const InputBuffers& b);
  static bool inputs16_ok(const WgFrame& f);
  // Motion at output resolution (create flag WG_CREATE_MOTION_OUTPUT_RES; specs 03 draft 4, 06 section 3, 20): the raw
  // motion texels of the region mw x mh from the texture's texel (0, 0), unchanged (no scale, no offset, no
  // resampling), into `dst` as dense rows of RG16F elements (mw * 4 bytes per row). Exact for the FP16 motion formats
  // (motion_fp16); a wider format is rounded to FP16 by the store. A texture smaller than the region repeats its last
  // row / column.
  void record_motion_hi(ID3D12GraphicsCommandList* list, const WgFrame& f, const BufferRange& dst, uint32_t mw,
                        uint32_t mh);
  static bool motion_fp16(const WgFrame& f);
  // executor=dx12 with motion_outres=point (motion at output resolution, our stand-in): the raw motion texel under each
  // render pixel's centre, chosen as the input pass chooses it (record_inputs with WG_CREATE_MOTION_OUTPUT_RES), into
  // `dst` (an R32G32_FLOAT texture with unordered access, of at least (render_x + render_width) x (render_y +
  // render_height), UNORDERED_ACCESS) at the render pixel's texture coordinate. The game's motion texture must be in a
  // shader-resource state; no barriers are recorded. False (*err) if its shader fails to compile.
  bool record_motion_point(ID3D12GraphicsCommandList* list, const WgFrame& f, ID3D12Resource* dst, std::string* err);
  // Records the output write: from the result buffer (R32_FLOAT elements, 3 per pixel) or, when result is null, the
  // bilinear fallback from the game's color texture (result.resource null).
  void record_output(ID3D12GraphicsCommandList* list, const WgFrame& f, const BufferRange& result);
  // The output pass from an RGBA32F texture holding the result (executor=dx12; the texture in UNORDERED_ACCESS before
  // and after): the operations of record_output's pass on the same FP32 values. False (*err) if its shader fails.
  bool record_output_tex(ID3D12GraphicsCommandList* list, const WgFrame& f, ID3D12Resource* proxy, std::string* err);
  // Copy paths (options input_copy / output_copy): plain CopyTextureRegion instead of the compute passes, where the
  // formats allow it. Each returns false and records nothing if this frame's textures are not eligible (*why says
  // why). Row pitches are the 256-byte aligned D3D12 footprint pitches, also what the network side reads / writes.
  struct CopyLayout {
    int color_pitch = 0, depth_pitch = 0, motion_pitch = 0, motion_channels = 0;
  };
  bool record_input_copies(ID3D12GraphicsCommandList* list, const WgFrame& f, uint32_t create_flags,
                           const InputBuffers& b, CopyLayout* layout, std::string* why);
  bool record_output_copy(ID3D12GraphicsCommandList* list, const WgFrame& f, const BufferRange& result,
                          int pitch, std::string* why);
  // Whether record_output_copy can copy into this frame's output texture (RGBA16F, single-sample 2D); checked before
  // the network writes RGBA16F rows, so that a frame it fails for takes the output pass instead.
  static bool output_copy_ok(const WgFrame& f, std::string* why);
  // RGBA16F rows of ow pixels: the pitch both sides use.
  static int half_rgba_pitch(uint32_t ow) { return (int)((ow * 8 + 255) / 256 * 256); }

  // Experimental (ready_wait=memory): after a UAV barrier, writes `value` (u32) at `flag`, so that HIP can wait on the
  // value in memory instead of the shared fence.
  void record_flag(ID3D12GraphicsCommandList* list, const BufferRange& flag, uint32_t value);
  // done_mode=spin: a one-thread pass that polls the u32 at `flag` until it is >= value (at most `polls` polls; a
  // timeout increments the u32 after it), then a UAV barrier, so the rest of the list waits for the network's flag
  // instead of a fence. record_flag_immediate: the CP writes `value` at `flag` (WriteBufferImmediate, after the work
  // before it), e.g. from a rescue list.
  void record_spin(ID3D12GraphicsCommandList* list, const BufferRange& flag, uint32_t value, uint32_t polls);
  void record_flag_immediate(ID3D12GraphicsCommandList* list, const BufferRange& flag, uint32_t value);

 private:
  struct Constants {
    uint32_t rx, ry, rw, rh;
    float msx, msy, mox, moy;
    uint32_t mw, mh, flags, ow;
    uint32_t oh, srgb, cw, ch;
  };
  UINT next_set();
  void record_inputs_pass(ID3D12GraphicsCommandList* list, const WgFrame& f, uint32_t create_flags,
                          const InputBuffers& b, bool compact);
  void bind(ID3D12GraphicsCommandList* list, ID3D12PipelineState* pso, const Constants& c, UINT set);

  ID3D12Device* device_ = nullptr;
  ID3D12RootSignature* root_ = nullptr;
  ID3D12PipelineState* inputs_ = nullptr;
  ID3D12PipelineState* inputs16_ = nullptr;
  ID3D12PipelineState* motionhi_ = nullptr;
  ID3D12PipelineState* motionpt_ = nullptr;  // compiled on first use (record_motion_point)
  ID3D12PipelineState* output_ = nullptr;
  ID3D12PipelineState* outtex_ = nullptr;
  ID3D12PipelineState* fallback_ = nullptr;
  ID3D12PipelineState* flag_ = nullptr;
  ID3D12PipelineState* spin_ = nullptr;
  ID3D12DescriptorHeap* heap_ = nullptr;
  UINT increment_ = 0;
  UINT ring_ = 0;
};

// Format helpers (also used by validation): SRV format for a (possibly depth or typeless) texture format, UAV format
// for an output format (sRGB -> UNORM with manual encoding), and whether a format is sRGB.
DXGI_FORMAT srv_format(DXGI_FORMAT f);
DXGI_FORMAT uav_format(DXGI_FORMAT f);
bool is_srgb(DXGI_FORMAT f);

}  // namespace wgi
