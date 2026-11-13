// SPDX-License-Identifier: MIT
// Model K layer table of the per-frame runtime (spec 40 draft 4, spec 41, spec 44): stages, sizes, window offsets,
// buffer plan and weight-blob sizes. C++ mirror of clean/k/reference/layout.py and clean/k/runner/k_pack.py
// (clean/k/tests/test_k_runtime_table.py checks that both give the same numbers).
//
// Model selection: runtime option "model=K" (WgRtConfig.runtime_options or WG_RUNTIME_OPTIONS; the environment wins,
// as for the other options). Default and "model=M": model M, unchanged. With model K the runtime runs
// clean/k/runtime/runtime_k.h (a create that fails, e.g. without the converted model, falls back to AMD FSR through the
// front end, spec 10 section 8).
#pragma once

#include <cstddef>
#include <string>
#include <vector>

struct WgRtConfig;

namespace wgrt {

enum class ModelId { M, K };

struct KStageInfo {
  const char* name;
  char kind;      // 'e' encoder (block then 2x2 merge), 'm' middle, 'd' decoder (2x2 expand + skip, then block)
  int level;      // block level (0 = base map)
  int C, heads;   // block width, heads (each 32 wide)
  bool fixed;     // token mixing: fixed (true) or softmax attention (false)
  int C_next;     // encoder: channels after the merge (level + 1)
  int C_prev;     // decoder: channels of the coarser input (level + 1)
  bool embed;     // e0: embedding 16 -> 32 (+ReLU) before the block
  bool head;      // d0: output head 32 -> 40 after the block
  int D() const { return 32 * heads; }
  int hidden() const { return 4 * C; }
};

// The eleven stages in run order: e0 e1 e2 e3 e4 m d4 d3 d2 d1 d0.
const std::vector<KStageInfo>& k_stage_table();
const KStageInfo* k_stage(const std::string& name);

// Spec 44 base map size for an output size; s is the pending factor (1 until settled).
void k_base_size(int ow, int oh, double s, int* wb, int* hb);
// Level L size: ceil(Wb / 2^L) x ceil(Hb / 2^L).
void k_level_size(int wb, int hb, int level, int* w, int* h);
// Spec 44 window offsets (ox, oy) of the eleven stages (table order) for frame index f. table: the window-offset
// table of stage m from the converted model (spec 48 net/window_offsets; spec 44 draft 2: never in our code or data).
void k_stage_offsets(int f, const int table[8][2], int off[11][2]);
// Windows of a stage map: ceil((W + ox) / 8) x ceil((H + oy) / 8).
int k_window_count(int w, int h, int ox, int oy);

// FP16 maps of one frame (clean/k/FORMATS.md "Frame buffers"): X0, S0..S4, A, B, OUT with their sizes in bytes,
// each rounded up to 256 bytes, for the fused stage kernels (clean/k/reference/layout.py FUSED_IO).
struct KBuffer {
  std::string name;
  size_t bytes;
};
std::vector<KBuffer> k_buffer_plan(int wb, int hb);

// Size in bytes of a stage's packed weight blob (clean/k/runner/k_pack.py stage_sections).
size_t k_stage_blob_bytes(const KStageInfo& st);

// Model chosen by the runtime options (see above). Returns false with *err for an unknown value.
bool model_from_options(const WgRtConfig& c, ModelId* model, std::string* err);
// Whether the runtime options name a known model (M or K); false with *err otherwise.
bool model_k_create_check(const WgRtConfig& c, std::string* err);

}  // namespace wgrt
