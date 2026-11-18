// SPDX-License-Identifier: MIT
// WrongGPU launcher core: the native Linux launcher's own removal. An AppImage is one file ($APPIMAGE) and the launcher keeps its
// data in one folder (WrongGPU, under $XDG_DATA_HOME); there is no installer entry, so "Uninstall WrongGPU" in Settings deletes
// those two things after one confirmation and nothing else. Written for this project.
#pragma once
#include "context.h"

namespace wg {

// The AppImage file this process runs from, when it can be told to be WrongGPU's own. All of these must hold, or false with *why in
// plain words (nothing is deleted on a false): $APPIMAGE names a regular file (not a link, not a folder) whose name starts with
// WrongGPU and ends in .AppImage, the file is an AppImage (an ELF file with the type 2 AppImage mark), and the system itself ties
// it to this process: $APPDIR (case-sensitive, links resolved) is a FUSE mount in the system's mount table, this running program is
// on it, and a live process whose executable is that very file (same device and inode) holds the mount's /dev/fuse connection. The
// AppImage runtime mounts its own file that way, so $APPIMAGE and $APPDIR from the environment alone prove nothing. Where the
// system gives no such proof (the process folder cannot be read, the AppImage was unpacked instead of mounted) the answer is no.
// *file is the real path as the core spells it (Z:\...).
// *host (when given): the same file in its Linux spelling (/home/x/WrongGPU.AppImage, links resolved), what a shell is given.
bool OwnAppImage(wstring& file, wstring* why, wstring* host = nullptr);

// Whether this program runs from an AppImage at all, a weaker question than OwnAppImage, for words only (never for a delete):
// $APPIMAGE and $APPDIR are set and this running program is inside $APPDIR. True too for an AppImage that was extracted and run
// (APPIMAGE_EXTRACT_AND_RUN), which has no FUSE mount to prove; false for the bare binary started with another AppImage's variables.
bool RunsInsideAnAppImage();

// Native Linux: this program's own file as the system names it (/proc/self/exe), with no round trip through the core's Windows-style
// spelling (which turns a backslash in a Linux name into a separator) and no length limit. Empty when it cannot be read.
std::string HostSelfPath();

// Uninstall: the AppImage file (OwnAppImage) and WrongGPU's data folder (RemoveSharedData: the converted model data, the
// games' list, the launcher's settings and logs; refused while a game still has WrongGPU, which reads that data). The file is
// proven first and its identity (device, inode, header) kept; a game still using the data is looked for next; then, just before the
// delete, the path must still be that same plain file (a file renamed in between is refused); the file goes first, then the data.
// *removedFile: the file's path; *kept: what RemoveSharedData kept (the player's ignore list). Never throws.
bool UninstallAppImage(Context& ctx, wstring* why, wstring* removedFile = nullptr, wstring* kept = nullptr);

}  // namespace wg
