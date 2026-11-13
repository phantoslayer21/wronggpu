// SPDX-License-Identifier: MIT
// WrongGPU launcher core: the model data (spec 30 "Model file", spec 08): the model file on this PC, the search for the
// player's DLSS model file, the conversion and what replaces what. Written for this project.
#include "model.h"
#include <filesystem>
#include "optiscaler.h"

namespace wg {

wstring ReadModelSource(Context& ctx) {
    string raw;
    if (!ReadAll(ModelSourceFile(ctx), raw)) return L"";
    wstring t = FromUtf8(raw), src;
    size_t p = t.find(L"source\t");
    if (p != wstring::npos) { src = t.substr(p + 7, t.find(L'\n', p) - p - 7); while (!src.empty() && (src.back() == L'\r' || src.back() == L' ')) src.pop_back(); }
    return src;
}
bool ExistingModelValid(Context& ctx, wstring* why, wgc::ModelFileInfo* info) {
    if (info) *info = wgc::ModelFileInfo();
    if (!LoadEmbeddedMap(ctx) || !IsFile(ModelFile(ctx))) { if (why) *why = L"no converted model on this PC"; return false; }
    string w;
    const bool ok = wgc::verify_model_file(*ctx.map.map, std::filesystem::path(ModelFile(ctx)), w, info);
    if (why) *why = FromUtf8(w);
    return ok;
}
// True when a valid model file may lack a group of this map (preset M, map version 4: the output-filter weights).
bool MapHasOptionalGroup(const wgc::Map& map) {
    for (auto& st : map.stages) if (st.optional()) return true;
    return false;
}
// The output-filter part of a model case ("" for a map without the optional group, as preset K's).
wstring FilterText(Context& ctx, bool present) {
    if (!ctx.map.map || !MapHasOptionalGroup(*ctx.map.map)) return L"";
    return present ? L"; output filter: from the DLSS file" : L"; output filter: WrongGPU's own";
}
// Search order of spec 30: the file the player picked; the selected game's folder; every game folder of the libraries.
std::vector<ModelCandidate> ModelCandidates(const GameTarget* t, const wstring& userDll, const std::vector<StoreGame>& libraries) {
    std::vector<ModelCandidate> c;
    std::set<wstring> seen;
    auto add = [&](const wstring& p, const wstring& origin) { if (seen.insert(Lower(FullPath(p))).second) c.push_back({FullPath(p), origin}); };
    if (!userDll.empty()) add(userDll, L"chosen by you");
    if (t && !t->root.empty()) for (auto& f : FindFilesNamed(t->root, L"nvngx_dlss.dll")) add(f, L"this game");
    for (auto& g : libraries) for (auto& f : FindFilesNamed(g.root, L"nvngx_dlss.dll", 30000)) add(f, L"another game (" + g.name + L")");
    return c;
}
ModelState FindModel(Context& ctx, const GameTarget* t, const wstring& userDll, const std::vector<StoreGame>& libraries) {
    ModelState m;
    if (!LoadEmbeddedMap(ctx)) { m.status = ModelState::FAILED; m.damaged = true; m.error = L"The installer is damaged (" + ctx.mapError + L"). Download it again."; return m; }
    wstring why;
    wgc::ModelFileInfo info;
    const bool existing = ExistingModelValid(ctx, &why, &info);
    // A valid file is used as it is, with one exception (spec 08 draft 2): a file without the output-filter weights
    // (written before they existed, or from a DLSS file whose filter was not known), or one written with other
    // converter data, is converted again with the search of a first install. Only a conversion that has the weights
    // replaces it (WriteModel); in every other case the file stays and the install goes on with it.
    const bool optional = MapHasOptionalGroup(*ctx.map.map);
    const bool inFile = existing && info.absent_groups.empty();  // the file has the weights
    wstring staleWhy;
    if (existing && optional && !inFile) staleWhy = L"the file has no output-filter weights";
    else if (existing && optional && !info.same_map) staleWhy = L"the file was written with other converter data";
    const wstring source = existing ? ReadModelSource(ctx) : wstring();
    auto keep = [&](const wstring& outcome) {
        m.status = ModelState::READY; m.reused = true; m.dll = source; m.origin = L"converted earlier on this PC";
        m.stale = !staleWhy.empty(); m.outcome = outcome + FilterText(ctx, inFile);
    };
    if (existing && staleWhy.empty()) {
        keep(L"reused");
        LogLine(ctx, L"model: reusing " + ModelFile(ctx) + L" (matches the converter data; made from " + (m.dll.empty() ? wstring(L"an unknown file") : m.dll) + L")");
        return m;
    }
    if (existing) LogLine(ctx, L"model: " + ModelFile(ctx) + L" matches the converter data, but " + staleWhy + L"; searching for a DLSS model file to convert it again");
    else LogLine(ctx, L"model: " + why + L"; searching for a DLSS model file");
    // Several DLLs (spec 08 section 8): the first that converts completely with the output-filter weights is used; if
    // none has them, the first that converts completely. Every file that converts gives the same other stages, so
    // the order only decides whether the weights come along. The candidates are the same as before.
    wgc::SearchOptions opt;
    std::shared_ptr<wgc::Converted> chosen, without;  // `without`: the first that converts without the weights
    ModelCandidate chosenCand, withoutCand;
    for (auto& cand : ModelCandidates(t, userDll, libraries)) {
        auto conv = std::make_shared<wgc::Converted>();
        const wgc::Result r = wgc::convert_file(*ctx.map.map, std::filesystem::path(cand.path), opt, *conv,
                                                [&ctx](const string& s) { LogLine(ctx, L"converter: " + FromUtf8(s)); });
        if (r.code == wgc::kOk) {
            if (wgc::has_every_group(*ctx.map.map, *conv)) { chosen = conv; chosenCand = cand; break; }
            if (!without) {
                without = conv; withoutCand = cand;
                LogLine(ctx, L"model: " + cand.path + L" converts (" + cand.origin + L"), without output-filter weights; trying the remaining files for one that has them");
            } else {
                LogLine(ctx, L"model: " + cand.path + L" converts (" + cand.origin + L"), without output-filter weights too");
            }
            m.passedOver.push_back(cand.path);
            continue;
        }
        m.rejections.push_back(cand.path + L": " + FromUtf8(r.message));
        LogLine(ctx, L"model: rejected " + cand.path + L": " + FromUtf8(r.message));
        if (wgc::is_installer_error(r.code)) {
            if (existing) {  // the file on this PC is valid and stays in use
                keep(L"kept: " + staleWhy + L"; converting it again failed (" + FromUtf8(r.message) + L")");
                LogLine(ctx, L"model: keeping " + ModelFile(ctx));
                return m;
            }
            m.status = ModelState::FAILED; m.damaged = true; m.error = FromUtf8(r.message);
            return m;
        }
    }
    const bool hasGroup = chosen != nullptr;
    if (!chosen && without) {
        chosen = without; chosenCand = withoutCand;
        m.passedOver.erase(m.passedOver.begin());  // the first of them is the one used
    }
    if (chosen) {
        std::vector<wstring> notes;
        for (auto& n : chosen->notes) notes.push_back(FromUtf8(n));
        const wstring others = m.passedOver.empty() ? wstring() : L" (" + std::to_wstring(m.passedOver.size()) + L" other file(s) convert without output-filter weights)";
        if (existing && !hasGroup) {  // nothing to gain: the file on this PC stays
            keep(L"kept: " + staleWhy + L"; the DLSS file found (" + chosenCand.path + L") converts without output-filter weights");
            if (!inFile) m.notes = notes;
            m.passedOver.clear();
            LogLine(ctx, L"model: no DLSS file with output-filter weights was found; keeping " + ModelFile(ctx));
            return m;
        }
        m.status = ModelState::READY; m.dll = chosenCand.path; m.origin = chosenCand.origin; m.version = FromUtf8(chosen->version); m.converted = chosen;
        m.replaces = existing; m.notes = notes;
        m.outcome = (existing ? L"replaced: " + staleWhy + L"; converted again from " : wstring(L"converted from ")) + chosenCand.path + FilterText(ctx, hasGroup) +
                    (optional && !hasGroup ? L" (not known for this DLSS build)" : L"");
        LogLine(ctx, L"model: using " + chosenCand.path + L" (" + chosenCand.origin + L")" + (optional ? hasGroup ? L", with output-filter weights" : L", without output-filter weights" : L"") +
                others + (existing ? L"; it will replace " + ModelFile(ctx) : wstring()));
        return m;
    }
    if (existing) {
        keep(L"kept: " + staleWhy + L"; no usable DLSS file was found to convert it again");
        LogLine(ctx, L"model: no DLSS model file to convert again; keeping " + ModelFile(ctx) + L" (made from " + (m.dll.empty() ? wstring(L"an unknown file") : m.dll) + L")");
        return m;
    }
    m.status = ModelState::FAILED;
    m.error = L"No DLSS model file that WrongGPU can use was found. WrongGPU needs the DLSS model file (nvngx_dlss.dll) of "
              L"version 310.5 or newer, which comes with many recent games and apps. Install or update such a game, or choose its "
              L"nvngx_dlss.dll yourself. The file stays where it is: only the model data is read, on this PC.";
    return m;
}
// A second conversion replaces a valid model file (ModelState::replaces). The player is never left without a working
// model: the new file is written beside the one in use, checked there, and only then moved over it; a copy of the
// earlier file is kept until the result has passed its check in place, and is put back otherwise.
// False: the new file was not used (`problem` says why) and the earlier file is where it was.
bool ReplaceModel(Context& ctx, const ModelState& m, wstring& problem) {
    const wstring side = Join(ModelDir(ctx), L"model-new"), sideFile = Join(side, FromUtf8(wgc::kModelFileName));
    const wstring previous = ModelFile(ctx) + L".previous";
    auto cleanup = [&] { if (IsDir(side)) RemoveTree(side); DeleteFileW(previous.c_str()); };
    cleanup();  // what an interrupted earlier run left behind
    const wgc::Result r = wgc::write_converted(*ctx.map.map, *m.converted, std::filesystem::path(side), false,
                                               [&ctx](const string& s) { LogLine(ctx, L"converter: " + FromUtf8(s)); });
    if (r.code != wgc::kOk) { problem = L"it could not be saved next to the model data in use"; cleanup(); return false; }
#ifdef WRONGGPU_TESTING
    if (!EnvVar(L"WRONGGPU_TEST_DAMAGE_NEW_MODEL").empty()) {  // tests: the new file is damaged before its check
        string bytes;
        if (ReadAll(sideFile, bytes) && bytes.size() > 200) { bytes[200] ^= 1; WriteFileAtomic(sideFile, bytes.data(), bytes.size()); }
    }
#endif
    string w;
    wgc::ModelFileInfo info;
    if (!wgc::verify_model_file(*ctx.map.map, std::filesystem::path(sideFile), w, &info) || !info.absent_groups.empty() || !info.same_map) {
        problem = L"the new file failed its check" + (w.empty() ? wstring() : L" (" + FromUtf8(w) + L")");
        cleanup(); return false;
    }
    LogLine(ctx, L"model: the new file " + sideFile + L" passed its check");
    if (!CopyFileW(ModelFile(ctx).c_str(), previous.c_str(), FALSE)) { problem = L"the model data in use could not be copied first: " + ErrText(GetLastError()); cleanup(); return false; }
    if (!MoveFileExW(sideFile.c_str(), ModelFile(ctx).c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        problem = L"the model data in use could not be replaced: " + ErrText(GetLastError());
        cleanup(); return false;
    }
    wstring why;
    if (!ExistingModelValid(ctx, &why, &info) || !info.absent_groups.empty()) {  // not expected: the same bytes passed a moment ago
        const bool back = MoveFileExW(previous.c_str(), ModelFile(ctx).c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != FALSE;
        problem = L"the new file failed its check after it was moved into place (" + why + L")" + (back ? L"; the earlier file was put back" : L"");
        cleanup(); return false;
    }
    LogLine(ctx, L"model: " + ModelFile(ctx) + L" replaced by the new file");
    cleanup();
    return true;
}
// Writes a freshly converted model (staged by the converter), checks it and records which file it came from.
// Returns which case it was and the lines for the player.
ModelResult WriteModel(Context& ctx, const ModelState& m, Ui& ui) {
    ModelResult res{m.outcome, m.notes};
    if (m.reused) {
        wstring why;
        if (!ExistingModelValid(ctx, &why)) Fail(L"The converted model on this PC changed while setup was running (" + why + L"). Run setup again.");
        if (m.stale) ui.log(L"model data: the file on this PC stays in use (" + m.outcome + L")");
        return res;
    }
    if (!m.converted) Fail(L"No model data to install.", EXIT_NO_MODEL);
    ui.step(0, 1, L"Saving the model data");
    if (m.replaces && ExistingModelValid(ctx)) {  // (a file that is gone or damaged by now is simply written again, below)
        wstring problem, why;
        wgc::ModelFileInfo info;
        if (!ReplaceModel(ctx, m, problem)) {
            // The file that was there has to be valid still; the install then goes on with it.
            if (!ExistingModelValid(ctx, &why, &info)) Fail(L"The converted model on this PC changed while setup was running (" + why + L"). Run setup again.");
            res.outcome = L"kept: the new conversion from " + m.dll + L" was not used (" + problem + L")" + FilterText(ctx, info.absent_groups.empty());
            res.notes.clear();
            ui.log(L"model data: the file on this PC stays in use (" + res.outcome + L")");
            return res;
        }
    } else {
        const wgc::Result r = wgc::write_converted(*ctx.map.map, *m.converted, std::filesystem::path(ModelDir(ctx)), false,
                                                   [&ctx](const string& s) { LogLine(ctx, L"converter: " + FromUtf8(s)); });
        if (r.code != wgc::kOk) Fail(FromUtf8(r.message));
        wstring why;
        if (!ExistingModelValid(ctx, &why)) Fail(L"The saved model data failed its check (" + why + L").");
    }
    // model-source.txt: which file the model was made from; with the optional group in the map, whether that file
    // had the output-filter weights, and the files that convert too but were not used.
    wstring text = L"source\t" + m.dll + L"\r\nversion\t" + m.version + L"\r\norigin\t" + m.origin + L"\r\nconverted\t" + NowStamp(false) +
                   L"\r\nmap\t" + FromUtf8(wgc::to_hex(ctx.map.map->file_sha256)) + L"\r\n";
    if (MapHasOptionalGroup(*ctx.map.map)) {
        const bool hasGroup = wgc::has_every_group(*ctx.map.map, *m.converted);
        text += wstring(L"filter\t") + (hasGroup ? L"from this file" : L"not known for this file; WrongGPU's own filter is used") + L"\r\n";
        for (auto& p : m.passedOver) text += L"not-used\t" + p + L" (converts, without output-filter weights)\r\n";
    }
    const string info = ToUtf8(text);
    WriteFileAtomic(ModelSourceFile(ctx), info.data(), info.size());
    ui.log(L"model data converted from " + m.dll + L" (" + m.origin + L") into " + ModelDir(ctx) + (m.replaces ? L", replacing the earlier file" : L""));
    return res;
}

#if WRONGGPU_PRESET_K
// Prepare both local caches for the two in-game providers. The wizard choice is only the startup default.
// `result`: what became of the other model's file (unchanged when it is not prepared).
bool WriteOtherModel(Context& ctx, const GameTarget& target, const ModelState& selected, Ui& ui, bool noScan, ModelResult& result) {
    const bool selectedK = ctx.presetK;
    wgc::LoadedMap selectedMap = std::move(ctx.map);
    wstring selectedMapError = ctx.mapError;
    struct Restore {
        Context& ctx;
        bool preset;
        wgc::LoadedMap& map;
        wstring& error;
        ~Restore() { ctx.presetK = preset; ctx.map = std::move(map); ctx.mapError = error; }
    } restore{ctx, selectedK, selectedMap, selectedMapError};
    ctx.presetK = !selectedK;
    ctx.mapError.clear();
    ModelState other = FindModel(ctx, &target, selected.dll, noScan ? std::vector<StoreGame>{} : ScanLibraries());
    if (other.damaged) Fail(other.error, EXIT_DAMAGED);
    if (!other.ok()) {
        ui.log(wstring(L"The other in-game provider (") + (ctx.presetK ? L"Preset K" : L"Model M") +
               L") needs a matching local model file. It is not prepared: " + other.error);
        return false;
    }
    result = WriteModel(ctx, other, ui);
    ui.log(L"Both Model M and Preset K model caches are prepared for OptiScaler's in-game provider menu.");
    return true;
}
#else
// An installer without Preset K (CMake WRONGGPU_PRESET_K=OFF; Context::presetK): Model M only. No Preset K choice
// or wording, /PRESET=K is an unknown option (exit 10 when silent), no Preset K map built in, one model cache
// (%LOCALAPPDATA%\WrongGPU\model; a model\K folder of an earlier build is not touched by an install and goes with
// the model folder at the last uninstall), OptiScaler's UpscalerIndex always 0 (kOptiIni; OptiScaler.ini is written
// anew from the package's on every install).
// The runtime options for wronggpu.ini: model items of an earlier install (another build's choice) go, the one model
// is named; the player's other options stay.
string OnlyModelM(const string& options) {
    string kept;
    for (size_t a = 0; a <= options.size();) {
        size_t e = options.find(';', a);
        if (e == string::npos) e = options.size();
        const string item = options.substr(a, e - a);
        if (!item.empty() && item.rfind("model=", 0) != 0) kept += (kept.empty() ? "" : ";") + item;
        a = e + 1;
    }
    return kept + (kept.empty() ? "" : ";") + "model=M";
}
#endif

}  // namespace wg
