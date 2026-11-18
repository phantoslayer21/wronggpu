// SPDX-License-Identifier: MIT
// WrongGPU launcher core: the native Linux launcher's own removal. An AppImage is one file ($APPIMAGE) and the launcher keeps its
// data in one folder (WrongGPU, under $XDG_DATA_HOME); there is no installer entry, so "Uninstall WrongGPU" in Settings deletes
// those two things after one confirmation and nothing else. Written for this project.
#pragma once
#include "context.h"

namespace wg {

// The AppImage file this process runs from, when it can be told to be WrongGPU's own. All of these must hold, or false with *why in
// plain words (nothing is deleted on a false): $APPIMAGE names a regular file (not a link, not a folder) whose name starts with
// WrongGPU and ends in .AppImage, the file is an AppImage (an ELF file with the type 2 AppImage mark), and this running program
// is inside that AppImage's own mount ($APPDIR), so the file is the one this launcher came from and not another program's.
// *file is the path as the core spells it (Z:\...).
bool OwnAppImage(wstring& file, wstring* why);

// Uninstall: the AppImage file (OwnAppImage) and WrongGPU's data folder (RemoveSharedData: the converted model data, the
// games' list, the launcher's settings and logs; refused while a game still has WrongGPU, which reads that data). Nothing is
// removed unless both can be: the AppImage is checked first, then the data goes (all or nothing), then the file. *removedFile:
// the file's path; *kept: what RemoveSharedData kept (the player's ignore list). Never throws.
bool UninstallAppImage(Context& ctx, wstring* why, wstring* removedFile = nullptr, wstring* kept = nullptr);

}  // namespace wg
