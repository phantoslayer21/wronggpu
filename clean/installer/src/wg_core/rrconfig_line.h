// SPDX-License-Identifier: MIT
// WrongGPU launcher core: the one engine setting Ray Reconstruction needs in Borderlands 4 and Hogwarts Legacy (Brief 16,
// rrconfig.h). The only place it is written: the code, the steps the player reads and the Copy button take it from here.
// It is the game's own Engine.ini setting, the section [SystemSettings] (owner OK 2026-11-04). Written for this project.
#pragma once

namespace wg {
inline const char kRrConfigLine[] = "r.NGX.DLSS.DenoiserMode=1";
}  // namespace wg
