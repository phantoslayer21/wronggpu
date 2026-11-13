// SPDX-License-Identifier: MIT
// WrongGPU launcher core: the model data (spec 30 "Model file", spec 08): the model file on this PC, the search for the
// player's DLSS model file, the conversion and what replaces what. Written for this project.
#pragma once
#include "detect.h"

namespace wg {

// ------------------------------------------------------------------------------------------------
// Model data (spec 30 "Model file", spec 08). The player's own nvngx_dlss.dll is converted once per PC into
// %LOCALAPPDATA%\WrongGPU\model\model.wgcm, reused while its digests match the embedded map. During the checks the
// conversion happens in memory; the file is written at install time. The DLL is never copied or modified.
// ------------------------------------------------------------------------------------------------
struct ModelState {
    enum Status { NONE, SEARCHING, READY, FAILED } status = NONE;
    bool reused = false;           // the converted file already on this PC matches the map
    bool damaged = false;          // the installer's converter data is damaged
    wstring dll, version, origin;  // the file that was (or will be) used
    wstring error;
    std::vector<wstring> rejections;  // "path: reason" for every file tried
    std::shared_ptr<wgc::Converted> converted;
    // The optional output-filter weights of the model M file (spec 08 draft 2, group `filter`; README "Model file").
    bool stale = false;            // the file on this PC is valid, but lacks the group or was written with other converter data
    bool replaces = false;         // `converted` is a second conversion that has the group: it replaces that file once it is saved and checked
    wstring outcome;               // which case this is, for the log, the install record and the support report
    std::vector<wstring> notes;    // one line each for the player (spec 08 section 8: no known output filter in this DLSS build)
    std::vector<wstring> passedOver;  // files that convert, but without the weights, and were not used (a file with them was preferred, or an earlier one without)
    bool ok() const { return status == READY; }
};
// What became of a model file at install time (WriteModel): the case and the lines for the player.
struct ModelResult { wstring outcome; std::vector<wstring> notes; };
struct ModelCandidate { wstring path, origin; };

wstring ReadModelSource(Context& ctx);
bool ExistingModelValid(Context& ctx, wstring* why = nullptr, wgc::ModelFileInfo* info = nullptr);
bool MapHasOptionalGroup(const wgc::Map& map);
wstring FilterText(Context& ctx, bool present);
std::vector<ModelCandidate> ModelCandidates(const GameTarget* t, const wstring& userDll, const std::vector<StoreGame>& libraries);
ModelState FindModel(Context& ctx, const GameTarget* t, const wstring& userDll, const std::vector<StoreGame>& libraries);
bool ReplaceModel(Context& ctx, const ModelState& m, wstring& problem);
ModelResult WriteModel(Context& ctx, const ModelState& m, Ui& ui);
bool WriteOtherModel(Context& ctx, const GameTarget& target, const ModelState& selected, Ui& ui, bool noScan, ModelResult& result);
string OnlyModelM(const string& options);
}  // namespace wg
