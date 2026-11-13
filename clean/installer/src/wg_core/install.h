// SPDX-License-Identifier: MIT
// WrongGPU launcher core: the install engine: the routes, AMD's FSR 4 for the comparison key, the install itself.
// Written for this project.
#pragma once
#include "checks.h"
#include "detect.h"
#include "model.h"

namespace wg {

struct CompareDll { wstring path, version, game; std::vector<int> v; };
struct Consent { bool moveMods = false, removeOld = false, movePlugins = false; };  // movePlugins (0.6.1): other setups' plugins aside (ForeignPlugins)

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
