// SPDX-License-Identifier: MIT
// WrongGPU installer: what this executable embeds - the payload table, the converter maps, the wordmark and the
// version that tools/make_payload.py generates (payload_table.h) - for wg::Context::data. Written for this project.
#pragma once
#include "wg_core/context.h"

const wg::EmbeddedData& EmbeddedPayload();
