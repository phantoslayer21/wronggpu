// SPDX-License-Identifier: MIT
// Model K weight packing for the runtime: a stage's float32 tensors (the converter's spec 48 section 7.2 names and
// shapes, as clean/k/runtime/model_k_load.h load_k_model returns them) -> the blob the stage kernels read
// (clean/k/FORMATS.md "WMMA tile format", "Linear blob", "Block blob", "Stage blob").
//
// C++ port of clean/k/runner/k_pack.py pack_stage(stage, tensors, d_row="gfx12"), including the shape normalisation of
// clean/k/reference/params.py normalize; byte-identical blobs (clean/k/tests/test_k_runtime_pack.py).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "../../runtime/model.h"  // wgrt::Tensor, wgrt::StageTensors (header-only types)

namespace wgrt {

// Packs stage ("e0" ... "d0") for the gfx12 WMMA accumulator layout (row r = channel 16 nt + r, the only layout the
// model K block kernels support). Tensors may have the converter's shapes (B / M [H, 8, 8, 8, 8], Wm [C_next, 4, C],
// Wx [4, C, C_prev], bx [4, C]) or the canonical ones (params.py); g1 may be called gamma1; extra tensors are ignored.
// Every value must be exactly representable in FP16 (infinities allowed, NaN refused). The blob has
// k_stage_blob_bytes(*k_stage(stage)) bytes. On failure returns an empty vector with a message in *err.
std::vector<uint8_t> k_pack_stage(const std::string& stage, const StageTensors& t, std::string* err);

}  // namespace wgrt
