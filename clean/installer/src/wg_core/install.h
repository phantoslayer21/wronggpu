// SPDX-License-Identifier: MIT
// WrongGPU launcher core: the install engine: the routes, AMD's FSR 4 for the comparison key, the install itself.
// Written for this project.
#pragma once
#include "checks.h"
#include "detect.h"
#include "model.h"

namespace wg {

struct CompareDll { wstring path, version, game; std::vector<int> v; };
// movePlugins (0.6.1): other setups' plugins aside (ForeignPlugins): those in `plugins` (paths relative to the exe folder, as
// the player's question listed them, each with the SHA-256 it had then), or every one there when it is empty
// (--switch-plugins: as they are when the install decides, before anything changes). moveMods: the same for the other
// setup's files of the switch question (`mods`, relative to the install folder). A file is moved only while its bytes are
// the ones agreed to (or byte for byte what the earlier install moved aside with the OK); one changed since is never moved.
using AgreedFiles = std::vector<std::pair<wstring, wstring>>;  // (path, SHA-256)
struct Consent {
    bool moveMods = false, removeOld = false, movePlugins = false;
    AgreedFiles plugins, mods;
};

void RemoveEarlierDirect(Context& ctx, const GameTarget& target, Ui& ui, std::vector<wstring>& warnings);
wstring CompareRecordFile();
std::vector<int> VersionParts(const wstring& s);
bool CompareCandidate(const wstring& path, CompareDll& out, wstring& why);
CompareDll FindCompareDll(Context& ctx, const std::vector<StoreGame>& libraries);
CompareDll RecordCompareDll(Context& ctx, Ui& ui, const wstring& gameRoot);
bool IsAmdFidelityFxName(const wstring& rel);
wstring GameAmdFfxDll(Context& ctx, const wstring& dir, const wstring& exeDir);
std::vector<wstring> DoInstall(Context& ctx, const GameTarget& target, const ModelState& model, const Consent& consent, Ui& ui, const HipInfo& hip);
}  // namespace wg
