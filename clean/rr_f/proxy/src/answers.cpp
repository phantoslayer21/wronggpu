// SPDX-License-Identifier: MIT
// Spec 60.8 / 61.5 answers shared by the release and the probe build.
#include "answers.h"

#include <string.h>

#include "optimal.h"

namespace wgrr {

namespace {
constexpr size_t kRequirementsV1End = 104;  // Q-L1: v1 = the members before the Vulkan-only block
const uint32_t kRequiredTags[5] = {WG_BT_COLOUR_IN, WG_BT_COLOUR_OUT, WG_BT_MOTION, WG_BT_DIFFUSE_ALBEDO,
                                   WG_BT_SPECULAR_ALBEDO};  // 60.8
}  // namespace

WgResultCode answer_requirements(WgRequirements* out) {
  if (!out) return WG_E_INVALID_PARAMETER;
  WgRequirements r;
  memset(&r, 0, sizeof r);
  r.flags = 1u << 1;  // D3D12 only; no v-sync or OS-scheduling bit
  r.cpu_threads = 0;
  r.max_viewports = 1;
  r.required_tag_count = 5;
  r.required_tags = kRequiredTags;  // valid until unload
  // os/driver detected: 0 = not available; required: 0 = no version gate; Vulkan-only members 0
  size_t end = struct_version(out->h) >= 2 ? sizeof(WgRequirements) : kRequirementsV1End;
  memcpy(reinterpret_cast<uint8_t*>(out) + sizeof(WgHeader), reinterpret_cast<const uint8_t*>(&r) + sizeof(WgHeader),
         end - sizeof(WgHeader));
  return WG_OK;
}

WgResultCode answer_version(WgFeatureVersion* out, uint64_t api_version, const uint32_t release_version[3]) {
  if (!out) return WG_E_INVALID_PARAMETER;
  out->api_version[0] = WG_API_VERSION_MAJOR(api_version);  // the game's version as passed to <init>
  out->api_version[1] = WG_API_VERSION_MINOR(api_version);
  out->api_version[2] = WG_API_VERSION_PATCH(api_version);
  for (int i = 0; i < 3; ++i) out->runtime_version[i] = release_version[i];
  return WG_OK;
}

WgResultCode answer_optimal(uint32_t api_minor, const WgRrOptions* options, WgRrOptimal* out) {
  if (!options || !out) return WG_E_INVALID_PARAMETER;
  if (options->mode > WG_RR_MODE_NATIVE_AA) return WG_E_INVALID_PARAMETER;
  uint32_t w = options->output_width == WG_SIZE_NOT_SET ? 0 : options->output_width;
  uint32_t h = options->output_height == WG_SIZE_NOT_SET ? 0 : options->output_height;
  OptimalAnswer a = optimal_settings(options->mode, w, h);  // 61.5, WrongGPU's rule
  bool zero = !a.has_ratio || a.width == 0 || a.height == 0;
  if (zero && api_minor >= 14) return WG_E_INVALID_PARAMETER;  // 2.14.1 plugin
  out->optimal_width = a.width;  // 2.7.1: success with the zero sizes copied out
  out->optimal_height = a.height;
  out->optimal_sharpness = 0.0f;
  out->min_width = a.min_width;
  out->min_height = a.min_height;
  out->max_width = a.max_width;
  out->max_height = a.max_height;
  return WG_OK;
}

}  // namespace wgrr
