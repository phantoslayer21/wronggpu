// SPDX-License-Identifier: MIT
// WrongGPU launcher core: leftover WrongGPU files in a game folder that has no install record (build/launcher-design.md
// section 6, "Leftovers", milestone M2): a file counts only when its SHA-256 is one of WrongGPU's releases'
// (known_releases.h, generated from our release builds) or of the payload this launcher embeds; never by its name.
// The player chooses "Clean up" (only those files go; a game DLL a direct install of ours had renamed comes back under
// its own name) or "Adopt" (an install record is made, the game becomes a managed install). Never automatic.
// Written for this project.
#pragma once
#include "context.h"
#include "record.h"

namespace wg {

// rel: relative to the folder (shown to the player); version: the release it belongs to. abs: set for a leftover outside
// the folder (an older direct install in the game's engine plugin folders): the file itself, and rel is then its path
// from the game's root (for example Engine\Plugins\FSR\...\signedbin\amd_fidelityfx_dx12.dll).
struct Leftover { wstring rel, sha, version, abs; };
// The leftovers in `dir` (and its plugins\ and wronggpu-plugins\ folders), and an older direct install of ours in an
// Unreal Engine game's plugin folders: the DLLs of every "signedbin" folder under Engine\Plugins or a project's Plugins
// folder, up to four folders above `dir` (the old direct route replaced the FidelityFX DLL there). There a DLL counts when
// its hash is ours or its version resource says WrongGPU (ProvablyOurs); the old install's wronggpu.ini, wronggpu-fsr.log
// (.old), wronggpu-license.txt and wronggpu-notices.txt count there too, but only in a signedbin folder that holds such a
// DLL (the one case that goes by name). Empty when `dir` has an install record.
std::vector<Leftover> FindLeftovers(Context& ctx, const wstring& dir);
wstring LeftoverSummary(const std::vector<Leftover>& files);  // "3 files of WrongGPU 0.4.5 (wronggpu_runtime.dll, ...)"
// Clean up: each file's hash is checked again right before it is removed; anything else stays. A FidelityFX DLL of ours
// in an engine plugin folder is removed only when the game's own DLL (kept as <name>.wronggpu-original.dll) is there to
// put back; otherwise it stays and a note says why. Returns notes.
std::vector<wstring> CleanLeftovers(Context& ctx, const wstring& dir, const std::vector<Leftover>& files);
// Adopt: refused for leftovers in the engine plugin folders (Clean up only). An install record (v2, owner wg) for the files, the restorer as uninstall.exe, owned-files.json, the managed
// list (and, with no launcher installed, the game's own Apps & features entry). Returns notes.
std::vector<wstring> AdoptLeftovers(Context& ctx, const wstring& exe, const wstring& dir, const std::vector<Leftover>& files);
// A file that is provably a WrongGPU build although the record names another hash (a newer build placed by hand): its
// version resource says WrongGPU, or its hash is one of our releases' or the embedded payload's. Never by its name.
bool ProvablyOurs(Context& ctx, const wstring& path);

}  // namespace wg
