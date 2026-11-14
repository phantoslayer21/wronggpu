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
// ModelCandidates minus the files the ignore lists leave out (the order of the others stays); the files left out and why go to
// m.ignored, the player's line for a chosen file that is left out to m.chosenSkipped. With no entries it is ModelCandidates.
std::vector<ModelCandidate> AllowedModelCandidates(Context& ctx, const IgnoreLists& ign, const GameTarget* t, const wstring& userDll,
                                                   const std::vector<StoreGame>& libraries, ModelState& m) {
    const std::vector<ModelCandidate> all = ModelCandidates(t, userDll, libraries);
    std::vector<ModelCandidate> cands;
    std::vector<wstring> paths;
    for (auto& c : all) paths.push_back(c.path);
    const std::vector<bool> skip = IgnoredMask(ctx, ign, paths, L"model", m.ignored);
    for (size_t i = 0; i < all.size(); ++i) if (!skip[i]) cands.push_back(all[i]);
    if (!userDll.empty() && !all.empty() && skip[0]) {
        const wstring& s = m.ignored.skipped.front();
        m.chosenSkipped = L"The file you chose (" + all[0].path + L") is not converted: it is " + s.substr(std::min(s.size(), all[0].path.size() + 2)) +
                          L". If it was listed by mistake, remove that line (or that --ignore argument).";
        LogLine(ctx, L"model: " + m.chosenSkipped);
    }
    return cands;
}
ModelState FindModel(Context& ctx, const GameTarget* t, const wstring& userDll, const std::vector<StoreGame>& libraries) {
    ModelState m;
    if (!LoadEmbeddedMap(ctx)) { m.status = ModelState::FAILED; m.damaged = true; m.error = L"The installer is damaged (" + ctx.mapError + L"). Download it again."; return m; }
    // The ignore lists (convert_ignore.h), read first: they decide what the search may convert and whether the file converted
    // earlier may be reused. A list that is there but can't be read: nothing is converted or reused (fail closed).
    const IgnoreLists ign = LoadConvertIgnore(ctx, L"model", &m.ignored);
    if (!ign.unreadable.empty()) { m.status = ModelState::FAILED; m.code = EXIT_USAGE; m.error = ign.unreadable; return m; }
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
    // A second exception, only while a list has entries: the file it was converted from (path and SHA-256 recorded in
    // model-source.txt) is on a list (LISTED), or its SHA-256 is not on record (UNKNOWN: an earlier version converted it).
    // Then it is converted again from an allowed file whatever that file has; with none, a LISTED file is not used and an
    // UNKNOWN one stays with a note. With no list entries nothing changes here.
    wstring distrustWhy;
    if (existing && !ign.empty()) {
        const wstring listed = RecordedSourceReason(ign, source, SourceField(ModelSourceFile(ctx), L"sha256"));
        if (!listed.empty()) {
            m.trust = ModelState::LISTED;
            distrustWhy = L"it was converted from " + (source.empty() ? wstring(L"a file") : source) + L", which is " + listed;
        } else if (SourceField(ModelSourceFile(ctx), L"sha256").empty()) {
            m.trust = ModelState::UNKNOWN;
            distrustWhy = L"the SHA-256 of the file it was converted from is not on record (an earlier version converted it), so the ignore lists can't check it";
        }
    }
    const bool trusted = m.trust == ModelState::TRUSTED;
    auto keep = [&](const wstring& outcome) {
        m.status = ModelState::READY; m.reused = true; m.dll = source; m.origin = L"converted earlier on this PC";
        m.stale = !staleWhy.empty() || !trusted; m.outcome = outcome + FilterText(ctx, inFile);
    };
    if (existing && staleWhy.empty() && trusted) {
        keep(L"reused");
        LogLine(ctx, L"model: reusing " + ModelFile(ctx) + L" (matches the converter data; made from " + (m.dll.empty() ? wstring(L"an unknown file") : m.dll) + L")");
        return m;
    }
    const wstring againWhy = trusted ? staleWhy : distrustWhy;  // why the file on this PC is converted again
    if (existing) LogLine(ctx, L"model: " + ModelFile(ctx) + L" matches the converter data, but " + againWhy + L"; searching for a DLSS model file to convert it again");
    else LogLine(ctx, L"model: " + why + L"; searching for a DLSS model file");
    // An UNKNOWN file that stays (nothing allowed was found): the player's line.
    auto keepUnknown = [&](const wstring& what) {
        keep(L"kept: " + distrustWhy + L"; " + what);
        m.notes.push_back(L"The model data on this PC was converted by an earlier version of WrongGPU, which did not record the SHA-256 of the DLSS file "
                          L"it came from, so the ignore lists can't check it. No other DLSS model file was found to convert it again, so it stays in use.");
        LogLine(ctx, L"model: keeping " + ModelFile(ctx) + L" (its source can't be checked against the ignore lists; " + what + L")");
    };
    // Several DLLs (spec 08 section 8): the first that converts completely with the output-filter weights is used; if
    // none has them, the first that converts completely. Every file that converts gives the same other stages, so
    // the order only decides whether the weights come along. The candidates are the same as before.
    // A file on a list is left out before anything is tried; the order of the others stays. The rule is absolute:
    // a file the player chose is left out too, and the player is told which list and line (or --ignore=) matched.
    const std::vector<ModelCandidate> cands = AllowedModelCandidates(ctx, ign, t, userDll, libraries, m);
    wgc::SearchOptions opt;
    std::shared_ptr<wgc::Converted> chosen, without;  // `without`: the first that converts without the weights
    ModelCandidate chosenCand, withoutCand;
    for (auto& cand : cands) {
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
            if (existing && trusted) {  // the file on this PC is valid and stays in use
                keep(L"kept: " + staleWhy + L"; converting it again failed (" + FromUtf8(r.message) + L")");
                LogLine(ctx, L"model: keeping " + ModelFile(ctx));
                return m;
            }
            if (existing && m.trust == ModelState::UNKNOWN) { keepUnknown(L"converting it again failed (" + FromUtf8(r.message) + L")"); return m; }
            m.status = ModelState::FAILED; m.damaged = true; m.error = FromUtf8(r.message);
            return m;
        }
    }
    if (!m.chosenSkipped.empty()) m.notes.push_back(m.chosenSkipped);
    const bool hasGroup = chosen != nullptr;
    if (!chosen && without) {
        chosen = without; chosenCand = withoutCand;
        m.passedOver.erase(m.passedOver.begin());  // the first of them is the one used
    }
    if (chosen) {
        std::vector<wstring> notes;
        for (auto& n : chosen->notes) notes.push_back(FromUtf8(n));
        const wstring others = m.passedOver.empty() ? wstring() : L" (" + std::to_wstring(m.passedOver.size()) + L" other file(s) convert without output-filter weights)";
        if (existing && !hasGroup && trusted) {  // nothing to gain: the file on this PC stays
            keep(L"kept: " + staleWhy + L"; the DLSS file found (" + chosenCand.path + L") converts without output-filter weights");
            if (!inFile) m.notes.insert(m.notes.end(), notes.begin(), notes.end());
            m.passedOver.clear();
            LogLine(ctx, L"model: no DLSS file with output-filter weights was found; keeping " + ModelFile(ctx));
            return m;
        }
        m.status = ModelState::READY; m.dll = chosenCand.path; m.origin = chosenCand.origin; m.version = FromUtf8(chosen->version); m.converted = chosen;
        m.replaces = existing; m.notes.insert(m.notes.end(), notes.begin(), notes.end());
        m.dllSha256 = Lower(Sha256File(chosenCand.path));
        m.outcome = (existing ? L"replaced: " + againWhy + L"; converted again from " : wstring(L"converted from ")) + chosenCand.path + FilterText(ctx, hasGroup) +
                    (optional && !hasGroup ? L" (not known for this DLSS build)" : L"");
        LogLine(ctx, L"model: using " + chosenCand.path + L" (" + chosenCand.origin + L")" + (optional ? hasGroup ? L", with output-filter weights" : L", without output-filter weights" : L"") +
                others + (existing ? L"; it will replace " + ModelFile(ctx) : wstring()));
        return m;
    }
    const size_t skipped = m.ignored.skipped.size();
    if (existing && m.trust == ModelState::LISTED) {  // never reused; nothing allowed to convert it again from
        m.status = ModelState::FAILED; m.code = EXIT_NO_MODEL;
        m.error = L"The model data on this PC (" + ModelFile(ctx) + L") is not used: " + distrustWhy + L". No other DLSS model file that WrongGPU may "
                  L"convert was found. WrongGPU converts only the DLSS model file (nvngx_dlss.dll) that came with one of your own games: choose "
                  L"the nvngx_dlss.dll that came with one of your own games (Model file...), or install or update such a game." + (m.chosenSkipped.empty() ? wstring() : L" " + m.chosenSkipped);
        LogLine(ctx, L"model: " + m.error);
        return m;
    }
    if (existing && m.trust == ModelState::UNKNOWN) { keepUnknown(L"no other DLSS model file was found"); return m; }
    if (existing) {
        keep(L"kept: " + staleWhy + L"; no usable DLSS file was found to convert it again" +
             (skipped ? L" (" + std::to_wstring(skipped) + L" file(s) on an ignore list were skipped)" : wstring()));
        LogLine(ctx, L"model: no DLSS model file to convert again; keeping " + ModelFile(ctx) + L" (made from " + (m.dll.empty() ? wstring(L"an unknown file") : m.dll) + L")");
        return m;
    }
    m.status = ModelState::FAILED;
    if (skipped && cands.empty()) {  // every file found is on an ignore list: nothing was tried
        m.code = EXIT_NO_MODEL;
        m.error = !m.chosenSkipped.empty() && skipped == 1
                      ? m.chosenSkipped + L" No other DLSS model file was found."
                      : L"Every DLSS model file found (" + std::to_wstring(skipped) + L") is on an ignore list, so none was converted. "
                        L"WrongGPU converts only the DLSS model file (nvngx_dlss.dll) that came with one of your own games, never a copy "
                        L"another program placed. " + IgnoredExample(ctx, m.ignored) +
                        (m.chosenSkipped.empty() ? wstring() : L" " + m.chosenSkipped);
        LogLine(ctx, L"model: every DLSS model file found is on an ignore list; nothing converted");
        return m;
    }
    const wstring skippedText = skipped ? L" " + std::to_wstring(skipped) + L" file(s) on an ignore list were not tried. " + IgnoredExample(ctx, m.ignored) : wstring();
    if (ctx.presetL) {  // Model L: said by name, with the converter's reason for the file the player chose
        const wstring picked = userDll.empty() ? wstring() : FullPath(userDll);
        wstring why;
        for (auto& r : m.rejections)
            if (!picked.empty() && r.size() > picked.size() + 2 && IEq(r.substr(0, picked.size() + 2), picked + L": ")) { why = r.substr(picked.size() + 2); break; }
        m.error = L"Model L can't be prepared: no DLSS model file with Model L's data was found." +
                  (why.empty() ? wstring() : L" The file you chose (" + picked + L"): " + why) +
                  L" Model L is converted from the DLSS model file (nvngx_dlss.dll) of version 310.5 or newer. The file stays where it is: "
                  L"only the model data is read, on this PC." + skippedText + (m.chosenSkipped.empty() ? wstring() : L" " + m.chosenSkipped);
        LogLine(ctx, L"model: " + m.error);
        return m;
    }
    m.error = L"No DLSS model file that WrongGPU can use was found. WrongGPU needs the DLSS model file (nvngx_dlss.dll) of "
              L"version 310.5 or newer, which comes with many recent games and apps. Install or update such a game, or choose its "
              L"nvngx_dlss.dll yourself. The file stays where it is: only the model data is read, on this PC." +
              (skipped ? L" " + std::to_wstring(skipped) + L" file(s) on an ignore list were not tried. " + IgnoredExample(ctx, m.ignored) : wstring()) +
              (m.chosenSkipped.empty() ? wstring() : L" " + m.chosenSkipped);
    return m;
}
// A second conversion replaces a valid model file (ModelState::replaces). The player is never left without a working
// model: the new file is written beside the one in use, checked there, and only then moved over it; a copy of the
// earlier file is kept until the result has passed its check in place, and is put back otherwise.
// False: the new file was not used (`problem` says why) and the earlier file is where it was. A file converted again because
// the ignore lists don't allow (or can't check) the earlier one is used whether it has the output-filter weights or not.
bool ReplaceModel(Context& ctx, const ModelState& m, wstring& problem) {
    const bool needGroups = m.trust == ModelState::TRUSTED;
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
    if (!wgc::verify_model_file(*ctx.map.map, std::filesystem::path(sideFile), w, &info) || (needGroups && !info.absent_groups.empty()) || !info.same_map) {
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
    if (!ExistingModelValid(ctx, &why, &info) || (needGroups && !info.absent_groups.empty())) {  // not expected: the same bytes passed a moment ago
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
            // a file converted from a listed file is never kept in its place
            if (m.trust == ModelState::LISTED)
                Fail(L"The model data on this PC was converted from a file on an ignore list, and the new conversion from " + m.dll + L" could not replace it (" +
                     problem + L"). Nothing was changed; try again.", EXIT_NO_MODEL);
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
    // the source file's SHA-256: a later reuse is checked against the ignore lists with it (FindModel)
    if (!m.dllSha256.empty()) text += L"sha256\t" + m.dllSha256 + L"\r\n";
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
