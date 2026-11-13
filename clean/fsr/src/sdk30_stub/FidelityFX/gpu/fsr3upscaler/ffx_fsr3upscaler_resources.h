// SPDX-License-Identifier: MIT
#pragma once
// WrongGPU stub. AMD's ffx_fsr3upscaler.h (fsr3-v3.0.4) includes this GPU-side header, but its host declarations do
// not use anything from it. The layout check (fsr30_layout_check.cpp) only needs the host declarations, so this empty
// file stands in for it and no GPU code of the SDK is vendored.
