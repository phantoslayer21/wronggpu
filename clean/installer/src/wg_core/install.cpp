// SPDX-License-Identifier: MIT
// WrongGPU launcher core: the install engine: the routes, AMD's FSR 4 for the comparison key, the install itself.
// Written for this project.
#include "install.h"

#include "coexist.h"
#include "gamerules.h"
#include "optiscaler.h"
#include "payload.h"
#include "pe.h"
#include "record.h"
#include "rr.h"
#include "settings.h"

namespace wg {

// The install routes (setup_route.h of the installer). Two routes. Hosted: the player's OptiScaler next to the game
// exe, WrongGPU's FFX DLL under its own name, the game's FSR files untouched (/OPTISCALER=, the wizard's default).
// Direct: WrongGPU's DLLs under the names of the game's FidelityFX DLLs, the game's kept as
// <name>.wronggpu-original.dll (the wizard's advanced choice, /DIRECTFSR, a silent install without /OPTISCALER=).
// CMake option WRONGGPU_DIRECT_ROUTE=OFF builds an installer that requires OptiScaler (README "Install routes"): an
// earlier direct install is removed first, as its uninstaller would, then the hosted install is made: in the exe
// folder by the update step of DoInstall, in another folder (FSR DLLs in a plugin folder) by RemoveEarlierDirect.
#if WRONGGPU_DIRECT_ROUTE
void RemoveEarlierDirect(Context&, const GameTarget&, Ui&, std::vector<wstring>&) {}
#else
// An earlier direct install in another folder than the exe folder (the game keeps its FidelityFX DLLs in a plugin
// folder; the hosted install goes next to the exe): removed from its own record and backup like an uninstall, with its
// state folder and "Apps & features" entry. The player's wronggpu.ini moves to the exe folder, where the hosted install
// keeps it (and where its uninstaller removes it). A failure while it is removed loses nothing: record and backup stay.
void RemoveEarlierDirect(Context& ctx, const GameTarget& target, Ui& ui, std::vector<wstring>& warnings) {
    std::vector<wstring> dirs;
    for (auto& sl : kSlots)
        for (auto& f : FindFilesNamed(target.root, sl.name, 60000, 8)) {
            const wstring d = Parent(f);
            if (!IEq(d, target.dir) && std::find_if(dirs.begin(), dirs.end(), [&](const wstring& x) { return IEq(x, d); }) == dirs.end()) dirs.push_back(d);
        }
    for (auto& old : dirs) {
        Record r;
        if (!LoadRecord(RecordPath(old), r) || r.legacy || HostedRecord(r)) continue;
        ui.log(L"An earlier installation was found in " + old + L" - restoring the original files first.");
        MigrateSettings(ctx, old, r.version, ui);
        try {
            for (auto& w : DoRestore(ctx, old, ui, false, true))
                if (!EndsWithI(w, L": wronggpu.ini")) warnings.push_back(w);  // (the settings file is moved below)
        } catch (const SetupError&) {
            ui.log(L"The earlier installation could not be removed; its record and backup were left as they are. Close anything that uses the game's files, then run setup again or uninstall it.");
            throw;
        }
        RemoveTree(StateDir(old));
        const wstring from = Join(old, L"wronggpu.ini"), to = Join(target.dir, L"wronggpu.ini");
        if (IsFile(from)) {
            if (!IsFile(to) && MoveFileExW(from.c_str(), to.c_str(), MOVEFILE_COPY_ALLOWED | MOVEFILE_WRITE_THROUGH)) ui.log(L"wronggpu.ini (your settings) moved to " + target.dir);
            else { ForceDelete(from); ui.log(L"wronggpu.ini of the earlier installation removed from " + old); }
        }
    }
}
#endif

// ------------------------------------------------------------------------------------------------
// FSR 4 for the comparison key (Home in game): AMD's FidelityFX SDK 2 upscaler DLL (amd_fidelityfx_upscaler_dx12.dll,
// version 4 or newer, exporting the FidelityFX API, Authenticode-signed by Advanced Micro Devices) from another game on
// this PC. WrongGPU never ships or copies it: its path goes to %LOCALAPPDATA%\WrongGPU\compare.txt
// ("fsr4<TAB>path", UTF-8), which the front end reads when the player's compare_dll setting is empty. No such DLL: no
// record, and the comparison uses the game's own AMD FSR as before.
// ------------------------------------------------------------------------------------------------
wstring CompareRecordFile() { return Join(AppDataDir(), L"compare.txt"); }
std::vector<int> VersionParts(const wstring& s) {
    std::vector<int> v;
    int cur = -1;
    for (wchar_t c : s) {
        if (c >= L'0' && c <= L'9') cur = (cur < 0 ? 0 : cur * 10) + (c - L'0');
        else if (cur >= 0) { v.push_back(cur); cur = -1; if (v.size() == 4) break; }
    }
    if (cur >= 0 && v.size() < 4) v.push_back(cur);
    return v;
}
// Usable as AMD's FSR 4: version >= 4, the FidelityFX API exports, a valid Authenticode signature of AMD. why: if not.
bool CompareCandidate(const wstring& path, CompareDll& out, wstring& why) {
    out.path = path;
    out.version = FileVersion(path);
    out.v = VersionParts(out.version);
    if (out.v.empty() || out.v[0] < 4) { why = L"version " + out.version + L" (FSR 4 needs 4.0 or newer)"; return false; }
    {
        MappedFile f(path, 256ull << 20);
        if (!f.data) { why = L"cannot be read"; return false; }
        const std::set<string> ex = ReadPeExports(f.data, f.size);
        for (const char* need : {"ffxCreateContext", "ffxDestroyContext", "ffxQuery", "ffxDispatch"})
            if (!ex.count(need)) { why = L"no " + FromUtf8(need) + L" export"; return false; }
    }
    const wstring sig = SignatureInfo(path);
    if (sig.find(L"signature valid") == wstring::npos || sig.find(L"Advanced Micro Devices") == wstring::npos) { why = sig; return false; }
    return true;
}
// The newest usable DLL; equal versions: one outside a mod folder (OptiScaler copies), then the shorter path.
CompareDll FindCompareDll(Context& ctx, const std::vector<StoreGame>& libraries) {
    CompareDll best;
    std::set<wstring> seen;
    auto modCopy = [](const wstring& p) { return Lower(p).find(L"optiscaler") != wstring::npos; };
    for (auto& game : libraries) {
        for (auto& f : FindFilesNamed(game.root, L"amd_fidelityfx_upscaler_dx12.dll", 30000)) {
            if (!seen.insert(Lower(FullPath(f))).second) continue;
            CompareDll c;
            wstring why;
            if (!CompareCandidate(FullPath(f), c, why)) { LogLine(ctx, L"AMD FSR 4 search: " + f + L" not used: " + why); continue; }
            c.game = game.name;
            const bool better = best.path.empty() || c.v > best.v ||
                                (c.v == best.v && (modCopy(best.path) != modCopy(c.path) ? modCopy(best.path)
                                                                                         : c.path.size() < best.path.size()));
            LogLine(ctx, L"AMD FSR 4 search: candidate " + c.path + L" (" + c.version + L", " + game.name + L")");
            if (better) best = c;
        }
    }
    return best;
}
// Best FSR 4 DLL of the libraries and of gameRoot (also with /NOSCAN, record then kept). Empty: none.
CompareDll RecordCompareDll(Context& ctx, Ui& ui, const wstring& gameRoot) {
    std::vector<StoreGame> where;
    if (!gameRoot.empty() && IsDir(gameRoot)) where.push_back({L"this game", FullPath(gameRoot), L"", L"install target"});
    if (ctx.noLibraryScan) {
        LogLine(ctx, L"AMD FSR 4 search: libraries not searched (/NOSCAN); record left as it is");
        return FindCompareDll(ctx, where);
    }
    ui.step(0, 1, L"Looking for AMD FSR 4 in your other games");
    std::vector<StoreGame> libraries = ScanLibraries();
    for (auto& w : where) {
        bool known = false;
        for (auto& l : libraries) known = known || IEq(FullPath(l.root), w.root);
        if (!known) libraries.push_back(w);
    }
    const CompareDll c = FindCompareDll(ctx, libraries);
    if (c.path.empty()) {
        DeleteFileW(CompareRecordFile().c_str());
        ui.log(L"No AMD FSR 4 DLL was found in your other games.");
        return c;
    }
    const string rec = ToUtf8(L"fsr4\t" + c.path + L"\r\nversion\t" + c.version + L"\r\ngame\t" + c.game + L"\r\nfound\t" + NowStamp(false) + L"\r\n");
    WriteFileAtomic(CompareRecordFile(), rec.data(), rec.size());
    ui.log(L"AMD FSR " + c.version + L" found on this PC (AMD's DLL of " + c.game + L": " + c.path + L")");
    return c;
}

// A game's AMD FidelityFX DLLs: never deleted, renamed, moved aside or overwritten by the hosted install.
bool IsAmdFidelityFxName(const wstring& rel) { return StartsWithI(BaseName(rel), L"amd_fidelityfx_"); }
// Private name of the AMD DLL that gives the hosted front end AMD's versions (first in its chain, amd_provider.h):
// OptiScaler redirects or hooks AMD's own file names, not this one.
const wchar_t* const kAmdCompare = L"wronggpu-amd-compare.dll";
// OptiScaler 10's custom upscaler slot (optiscaler.h kOptiSlotFrontEnd): a package with OptiScaler's own folder whose
// OptiScaler.dll is WrongGPU's edition or version 10 or newer (a DLL without a version: the folder decides). An older
// OptiScaler with such a folder keeps the FidelityFX library route.
static bool OptiSlotPackage(const Context& ctx) {
    if (!ctx.opti.ok() || !IsDir(Join(ctx.opti.dir, L"OptiScaler"))) return false;
    if (ctx.opti.edition) return true;
    const wstring& v = ctx.opti.version;
    return v.empty() || v[0] < L'0' || v[0] > L'9' || _wtoi(v.c_str()) >= 10;
}
// The game's own amd_fidelityfx_dx12.dll (AMD's, in the install folder or next to the exe), or empty.
wstring GameAmdFfxDll(Context& ctx, const wstring& dir, const wstring& exeDir) {
    for (const wstring& d : {dir, exeDir}) {
        if (d.empty()) continue;
        const wstring p = Join(d, kSlots[0].name);
        wstring info;
        if (ClassifySlot(kSlots[0], p, nullptr, info) == FFX_GAME) return p;
        if (IsFile(p)) LogLine(ctx, L"AMD provider: " + p + L" is not used (" + info + L")");
    }
    return wstring();
}

// ------------------------------------------------------------------------------------------------
// Install engine
// ------------------------------------------------------------------------------------------------
std::vector<wstring> DoInstall(Context& ctx, const GameTarget& target, const ModelState& model, const Consent& consent, Ui& ui, const HipInfo& hip) {
    std::vector<wstring> warnings;
    const wstring dir = target.dir;
    // 0. The embedded package and converter data must be intact before anything is touched.
    ui.step(0, 1, L"Verifying the package");
    for (auto& f : *ctx.data) {
        DWORD sz = 0; const void* d = PayloadData(f, sz);
        if (!d || sz != f.size || Sha256Bytes(d, sz) != f.sha256) Fail(L"The installer is damaged (" + wstring(f.rel) + L"). Download it again.", EXIT_DAMAGED);
    }
    // Ray Reconstruction (rr.h): a game that uses it gets this build's RR files too (OptiScaler setup only: the RR plugin is
    // one of OptiScaler's plugins)
    const bool rrGame = (!kDirectRoute || ctx.opti.ok()) && RrPayloadAvailable(ctx) && DetectRr(target.exe).uses;
    // 0.6.1: other setups' plugins in OptiScaler's plugin folders may hook RR too: WrongGPU's RR goes in only when every one of
    // them is moved aside (backed up as moved, below): those the player agreed to move (the plugins the question listed;
    // --switch-plugins: those in the folder now), and those the earlier install moved aside with that OK (its record's
    // disable entries, byte for byte: they come back with its restore and go aside again). Any other one keeps WrongGPU's RR
    // out, as does one outside the install folder (nothing to move it with). Decided here, before anything changes, against
    // what the folder holds now (the plugins moved before are still in the backup; one put back by hand since is asked
    // about like any other); checked again after the earlier install's restore (below). The earlier install's own plugins
    // are not "another setup's".
    std::map<wstring, wstring> pluginsMovedBefore;  // lower-case path relative to dir -> the hash the player's OK was for
    if (target.installed)
        for (auto& e : target.record.e)
            if (e.action == L"disable" && !e.orig.empty() && EndsWithI(e.rel, L".asi")) pluginsMovedBefore[Lower(e.rel)] = e.orig;
    const std::vector<wstring> pluginsNow = rrGame ? ForeignPlugins(target.exeDir, target.installed ? &target.record : nullptr, dir) : std::vector<wstring>{};
    const std::vector<wstring> pluginsOk = !consent.movePlugins ? std::vector<wstring>{} : consent.plugins.empty() ? pluginsNow : consent.plugins;
    auto pluginsKeepRrOut = [&](const PluginPlan& p) {
        wstring list;
        for (auto& f : p.blocking) list += (list.empty() ? L"" : L", ") + f;
        const wstring w = p.outside ? L"Other OptiScaler plugins are in this game (" + list + L") outside the folder WrongGPU installs to, so they "
                                      L"can't be moved aside: WrongGPU's Ray Reconstruction was left out (they may hook Ray Reconstruction as WrongGPU's RR plugin does)."
                                    : ForeignPluginsText(p.blocking);
        LogLine(ctx, L"Ray Reconstruction: " + w);
        warnings.push_back(w);
    };
    bool rr = rrGame;
    if (rr) {
        const PluginPlan before = PlanPluginsAside(target.exeDir, dir, pluginsNow, pluginsOk, {});
        if (!before.blocking.empty()) { rr = false; pluginsKeepRrOut(before); }
    }
    if (rr)
        for (size_t i = 0; i < ctx.data->rrCount; ++i) {
            const PayloadFile& f = ctx.data->rrFiles[i];
            DWORD sz = 0; const void* d = PayloadData(f, sz);
            if (!d || sz != f.size || Sha256Bytes(d, sz) != f.sha256) Fail(L"The installer is damaged (" + wstring(f.rel) + L"). Download it again.", EXIT_DAMAGED);
        }
    if (!LoadEmbeddedMap(ctx)) Fail(L"The installer is damaged (" + ctx.mapError + L"). Download it again.", EXIT_DAMAGED);
    RequireOptiScaler(ctx);
    if (!target.antiCheat.empty()) Fail(AntiCheatMessage(target.antiCheat), EXIT_ANTICHEAT);
    // A game that stops OptiScaler unless the player's REFramework is installed (gamerules.h): no install without it. An
    // install that is there already is updated or repaired all the same, with a warning (it does not start either way).
    const GameRule* rule = GameRuleFor(target.exe);
    if (rule && rule->needsReFramework) {
        const ReFrameworkState refw = DetectReFramework(target.exeDir);
        if (!refw.found && !IsFile(RecordPath(dir))) Fail(NeedsReFrameworkText(*rule), EXIT_REFUSED);
        if (!refw.found) {
            const wstring w = L"REFramework isn't in this game any more: " + wstring(rule->name) + L" blocks OptiScaler, and WrongGPU with it, until REFramework is installed again.";
            LogLine(ctx, w);
            warnings.push_back(w);
        } else {
            LogLine(ctx, L"REFramework found (" + refw.evidence + L"): " + rule->name + L" runs OptiScaler with it");
        }
    }
    if (!model.ok()) Fail(model.error.empty() ? L"No usable DLSS model file." : model.error, model.damaged ? EXIT_DAMAGED : EXIT_NO_MODEL);
    // (the player's OK for other mods' files, before anything changes: target.conflicts is the folder as the earlier install's
    // restore leaves it, apart from the files that restore brings back, which are those the player's OK was for)
    if (target.conflicts.modsFound() && !consent.moveMods) Fail(L"Other upscaler mods were found. They have to be moved aside first (with a backup).", EXIT_OTHER_MODS);
    if (target.conflicts.oldFound() && !consent.removeOld) Fail(L"An older WrongGPU (DLSS 4.5 AMD) install was found. It has to be removed first.", EXIT_OLD_INSTALL);
    if (DWORD e = WriteTest(dir)) Fail(L"Cannot write to the game folder: " + ErrText(e));

    // 1. Model data (in %LOCALAPPDATA%\WrongGPU, shared by all games); AMD's FSR 4 DLL for the comparison key.
    const ModelResult selectedModel = WriteModel(ctx, model, ui);
#if WRONGGPU_PRESET_K
    ModelResult otherModel;
    if (!WriteOtherModel(ctx, target, model, ui, ctx.noLibraryScan, otherModel))
        warnings.push_back(L"Only the selected model is prepared. To use the other model in OptiScaler, run setup with a compatible local model file for that preset.");
    // Model M's file, whichever preset starts first: its case goes into the install record, its line to the player.
    const ModelResult& modelM = ctx.presetK ? otherModel : selectedModel;
#else
    const ModelResult& modelM = selectedModel;
#endif
    for (auto& n : modelM.notes) { ui.log(n); warnings.push_back(n); }
    // The RR data (shared by every RR game, %LOCALAPPDATA%\WrongGPU\model\F): the file on this PC, else this game's own RR
    // model file converted now. None that converts: the RR files go in all the same, with RR off until the data is there
    // (Settings > Ray Reconstruction data, or the game page's Convert).
    bool rrData = false;
    if (rr) {
        ui.step(0, 1, L"Preparing the Ray Reconstruction data");
        const RrDataState rd = FindRrData(ctx, &target, L"", {});
        wstring why;
        rrData = rd.ok && WriteRrData(ctx, rd, &why);
        if (!rrData) {  // nothing fails silently: the operation's result says it too (the window's note after its success)
            const wstring w = L"Ray Reconstruction data isn't converted yet (" + (why.empty() ? rd.error : why) + L"): WrongGPU's Ray Reconstruction stays off in this game until it is.";
            LogLine(ctx, L"RR data: " + w);
            ui.log(w);
            warnings.push_back(w);
        } else if (!rd.reused) {
            // converted now: the RR games installed before the data existed get RR too (their settings files said off). A
            // folder this process may not write is logged; the window writes it again with permission (SyncRrSettings).
            for (auto& f : RefreshAllRrSettings(ctx, dir)) LogLine(ctx, L"RR data: the settings of " + f.dir + L" are to be written again: " + f.why);
        }
    }
    // OptiScaler 10 (the custom slot): OptiScaler lists AMD's FSR itself and WrongGPU's entry holds WrongGPU only, so no
    // AMD DLL is looked for or copied (the AMD provider is for the front end under OptiScaler before 10)
    const bool slotPackage = OptiSlotPackage(ctx);
    if (slotPackage) LogLine(ctx, L"OptiScaler 10: no AMD FSR 4 search, no AMD provider copy (OptiScaler lists AMD's FSR itself)");
    const CompareDll pcFsr4 = slotPackage ? CompareDll{} : RecordCompareDll(ctx, ui, target.root.empty() ? target.dir : target.root);
    if (hip.ok) {
        const string chosen = ToUtf8(hip.dll);
        WriteFileAtomic(Join(AppDataDir(), L"hip-runtime.txt"), chosen.data(), chosen.size());
        ui.log(L"Recorded usable HIP runtime: " + hip.dll);
    }

    // 2. An older WrongGPU install with a record, then an earlier install of this version: back to the original files.
    if (!target.conflicts.legacyRecordDir.empty()) for (auto& w : LegacyRestore(ctx, target.conflicts.legacyRecordDir, ui)) warnings.push_back(w);
    RemoveEarlierDirect(ctx, target, ui, warnings);
    // Once the earlier installation is removed, a failure leaves the game's original files and no install: what is
    // left of its state folder (the uninstaller, which has no record any more) goes too. rc6 L4: never the player's settings
    // there (profile.ini, previous\ for Roll back, rr-config\): an update or repair that fails keeps them for the next try,
    // as does a failed install where they were before it (a profile the launcher saved first).
    const bool keepSettings = StateHasSettings(dir) || IsFile(RecordPath(dir));
    struct EarlierGone {
        const wstring& dir; Ui& ui; bool armed = false;
        ~EarlierGone() {
            if (!armed || IsFile(RecordPath(dir))) return;
            RemoveStateKeepingSettings(dir);
            ui.log(L"The earlier installation was removed and the game's original files are back; the new version was not installed (your settings were kept).");
        }
    } earlierGone{dir, ui};
    std::map<wstring, wstring> keptByRestore;  // recorded files the restore left in place (changed since the install), with its warning
    if (IsFile(RecordPath(dir))) {
        ui.log(L"An earlier installation was found - restoring the original files first.");
        Record old;
        if (LoadRecord(RecordPath(dir), old)) MigrateSettings(ctx, dir, old.version, ui);
        earlierGone.armed = true;
        try {
            for (auto& w : DoRestore(ctx, dir, ui, false, true, &keptByRestore)) warnings.push_back(w);
        } catch (const SetupError&) {  // nothing is lost: the restore can be run again from the record and the backup
            ui.log(L"The earlier installation could not be removed; its record and backup were left as they are. Close anything that uses the game's files, then run setup again or uninstall it.");
            throw;
        }
#ifdef WRONGGPU_TESTING
        if (!EnvVar(L"WRONGGPU_TEST_FAIL_AFTER_RESTORE").empty()) Fail(L"Test: stopped after the earlier installation was removed.");
#endif
    }
    // 3. What is in the folder now (after the restores) decides the plan.
    // Every FidelityFX DLL the game has and the payload replaces (spec 20: the FFX API DLL and the FSR 3.0 components).
    const bool opti = !kDirectRoute || ctx.opti.ok();
    std::vector<const FfxSlot*> replace;
    for (auto& st : opti ? std::vector<SlotState>{} : InspectSlots(ctx, dir, nullptr)) {  // OptiScaler mode: the game's FSR stays
        if (st.kind == FFX_NONE) continue;
        if (st.kind != FFX_GAME) Fail(wstring(st.slot->name) + L" in " + dir + L" is not the game's FSR DLL (" + st.info + L"). Verify the game files and run setup again.", EXIT_OTHER_MODS);
        LogLine(ctx, wstring(st.slot->name) + L": " + st.info + L"; " + SignatureInfo(Join(dir, st.slot->name)));
        if (st.inPayload) replace.push_back(st.slot);
        else LogLine(ctx, wstring(st.slot->name) + L" is left untouched: this installer has no replacement for it");
    }
    if (!opti && replace.empty()) Fail(L"None of the game's FSR DLLs in " + dir + L" can be replaced by this installer.", EXIT_REFUSED);
    auto replaced = [&](const wstring& rel) { for (auto* sl : replace) if (IEq(rel, sl->name)) return true; return false; };
    // OptiScaler mode: WrongGPU's FFX API DLL under its own name, no FSR 3.0 forwarders
    // OptiScaler 10 (optiscaler.h kOptiSlotFrontEnd): the front end in OptiScaler's custom upscaler slot, no menu plugin
    const bool slot = opti && OptiSlotPackage(ctx);
    auto payloadUsed = [&](const PayloadFile& f) {
        if (slot && OptiOnlyPayload(f.rel)) return false;
        return opti ? !IsSlotName(f.rel) || IEq(f.rel, L"amd_fidelityfx_dx12.dll") : !OptiOnlyPayload(f.rel) && (!IsSlotName(f.rel) || replaced(f.rel));
    };
    // The plugin's place: plugins\ next to OptiScaler, or a folder of its own when Ultimate ASI Loader would start it a
    // second time from plugins\ (set below, before anything is planned or written).
    wstring pluginRel = kAsiPlugin;
    auto payloadRel = [&](const PayloadFile& f) {
        return opti && IEq(f.rel, L"amd_fidelityfx_dx12.dll") ? wstring(slot ? kOptiSlotFrontEnd : kOptiFrontEnd) : opti && IEq(f.rel, kAsiPlugin) ? pluginRel : wstring(f.rel);
    };
    // OptiScaler's files (OptiScaler.dll as the proxy, OptiScaler.ini set up for WrongGPU from a temporary copy)
    struct OptiFile { wstring rel, src; };
    std::vector<OptiFile> optiFiles;
    wstring proxy, proxyReason;
    CoexistReport co;
    string selectedIni;
    ReadAll(Join(dir, L"wronggpu.ini"), selectedIni);
    string existingUpscaler, existingSection;
    if (!IniGetFrontEnd(selectedIni, "upscaler", &existingUpscaler, &existingSection))
        selectedIni = IniSet(selectedIni, "fsr", "upscaler", "network");
    selectedIni = IniSet(selectedIni, "fsr", "runtime_dll", "wronggpu_runtime.dll");
    // Preserve user options, then append the explicit installer choice (last model item wins).
    string selectedOptions, selectedSection;
    IniGetFrontEnd(selectedIni, "runtime_options", &selectedOptions, &selectedSection);
#if WRONGGPU_PRESET_K
    selectedOptions += selectedOptions.empty() ? "" : ";";
    selectedOptions += ctx.presetK ? "model=K" : "model=M";
#else
    if (target.presetK) {  // the launcher asked first (PresetKText); the command line says it in the log
        const wstring w = L"Preset K isn't in this version of WrongGPU: " + target.name + L" now starts on Model M.";
        LogLine(ctx, w);
        ui.log(w);
    }
    selectedOptions = OnlyModelM(selectedOptions);
#endif
    selectedIni = IniSet(selectedIni, "fsr", "runtime_options", selectedOptions);
    const wstring selectedIniSrc = Join(OptiTempDir(ctx), L"WrongGPU.selected.ini");
    WriteFileAtomic(selectedIniSrc, selectedIni.data(), selectedIni.size());
    optiFiles.push_back({L"wronggpu.ini", selectedIniSrc});
    // Where the game's Ray Reconstruction needs the host to restore the game's command-list state (plugin interface 5,
    // gamerules.h), OptiScaler must be WrongGPU's edition of OptiScaler (optiedition.h: the launcher downloads it; never
    // shipped with WrongGPU). With another OptiScaler (the player's own download, used when the edition couldn't be
    // downloaded) the RR plugin reports RR unsupported there, so the game keeps its own ray tracing.
    if (opti && rr && rule && rule->needsHostRestore) {
        if (ctx.opti.edition) {
            LogLine(ctx, L"OptiScaler.dll: WrongGPU's edition of OptiScaler (plugin interface 5): " + wstring(rule->name) + L"'s Ray Reconstruction needs it");
        } else {
            const wstring w = L"This OptiScaler is no OptiScaler edition with plugin interface 5: WrongGPU's Ray Reconstruction isn't offered in " +
                              wstring(rule->name) + L", and the game keeps its own ray tracing.";
            LogLine(ctx, w);
            warnings.push_back(w);
        }
    }
    // A game that creates Ray Reconstruction itself, not through the game interface (rr.h RrUse::direct: Unreal Engine's
    // plugin layout): only WrongGPU's edition of OptiScaler hands its RR calls to the RR plugin (plugin interface 4 or newer).
    // With another OptiScaler the RR files are installed all the same (a later repair with the edition makes them work) and
    // the result says that RR isn't offered there; the window's RR card says it too (launcher.h optiEdition).
    if (opti && rr && !(rule && rule->needsHostRestore)) {
        const RrUse ru = DetectRr(target.exe);
        if (ru.direct) {
            const wstring game = !target.name.empty() ? target.name : GameDisplayName(target.exe);
            if (ctx.opti.edition) {
                LogLine(ctx, L"Ray Reconstruction: " + game + L" creates it itself (" + ru.why + L", no game interface RR plugin); WrongGPU's edition of "
                             L"OptiScaler (plugin interface 4 or newer) hands it to WrongGPU's RR plugin");
            } else {
                const wstring w = RrDirectNoEditionText(game);
                LogLine(ctx, w + L" (" + ru.why + L": the game creates Ray Reconstruction itself)");
                warnings.push_back(w);
            }
        }
    }
    if (opti) {
        // The rules table on the folder as it is now (after the earlier install's restore): refusals, the proxy name
        // (design section 5), the plugin's place.
        co = AssessCoexistence(ctx, target.exe, dir, nullptr);
        if (const Finding* no = co.refusal()) Fail(no->what + L": " + no->message, no->code ? no->code : EXIT_REFUSED);
        proxy = co.proxy.name;
        proxyReason = co.proxy.reason;
        if (proxy.empty()) Fail(L"OptiScaler needs a free name next to the game exe (dxgi.dll, winmm.dll, version.dll, wininet.dll, winhttp.dll or dbghelp.dll): " + co.proxy.reason, EXIT_OTHER_MODS);
        string ini;
        if (!ReadAll(Join(ctx.opti.dir, L"OptiScaler.ini"), ini)) Fail(L"Cannot read OptiScaler.ini of " + ctx.opti.source);
        const bool pathKey = IniHasKey(ini, "Plugins", "Path");
        for (auto& k : kOptiIni) {
            if (slot && (!strcmp(k.key, "FfxDx12SRPath") || (!strcmp(k.section, "FSR") && !strcmp(k.key, "UpscalerIndex")))) continue;
            ini = IniSet(ini, k.section, k.key, *k.value ? string(k.value) : ToUtf8(Join(dir, kOptiFrontEnd)));
        }
        if (slot) {
            for (auto& k : kOptiSlotIni) ini = IniSet(ini, k.section, k.key, k.value);
            LogLine(ctx, L"OptiScaler 10: WrongGPU in its custom upscaler slot (" + wstring(kOptiSlotFrontEnd) + L"), Dx12Upscaler=WrongGPU, [WrongGPU] UpscalerIndex=0");
        }
        if (co.asiLoader) {
            if (pathKey) {
                pluginRel = Join(kAsiPluginOwnDir, BaseName(kAsiPlugin));
                ini = IniSet(ini, "Plugins", "Path", ToUtf8(Join(dir, kAsiPluginOwnDir)));
                LogLine(ctx, L"Ultimate ASI Loader found: WrongGPU's plugin goes to " + pluginRel + L" (OptiScaler.ini [Plugins] Path), so the loader does not start it from plugins\\ a second time");
            } else {
                const wstring w = L"Ultimate ASI Loader is in this game and this OptiScaler build has no [Plugins] Path setting: the loader may start WrongGPU's plugin a second time.";
                LogLine(ctx, w);
                warnings.push_back(w);
            }
        } else if (IsDir(Join(ctx.opti.dir, L"OptiScaler"))) {
            // OptiScaler 10 (the WrongGPU edition) takes the game's OptiScaler\ folder as its own when there is one (its
            // package brings it) and then looks for plugins in OptiScaler\plugins\ for Path=auto: plugins\WrongGPU.asi and
            // plugins\WrongGPU-RR.asi would not load. The folder by name, as for Ultimate ASI Loader above.
            ini = IniSet(ini, "Plugins", "Path", ToUtf8(Join(dir, L"plugins")));
            LogLine(ctx, L"OptiScaler.ini [Plugins] Path: " + Join(dir, L"plugins") + L" (this OptiScaler's own folder is OptiScaler\\)");
        }
        for (auto& w : co.Warnings()) { warnings.push_back(w); LogLine(ctx, L"coexistence: " + w); }
#if WRONGGPU_PRESET_K
        ini = IniSet(ini, slot ? "WrongGPU" : "FSR", "UpscalerIndex", ctx.presetK ? "1" : "0");
#endif
        // (else 0, from kOptiIni)
        // Games whose start-up fails inside OptiScaler's DXGI factory hooks when WrongGPU is its FSR library (seen with
        // OptiScaler 0.9.5.4 in 007 First Light, 2026-10-13): OptiScaler's own factory wrapping avoids those hooks.
        for (auto exe : kOptiFactoryWrappingGames)
            if (IEq(BaseName(target.exe), exe)) {
                ini = IniSet(ini, "Spoofing", "DxgiFactoryWrapping", "true");
                LogLine(ctx, L"OptiScaler.ini: DxgiFactoryWrapping=true for " + BaseName(target.exe) + L" (this game does not start without it)");
            }
        // A game's experimental opt-in keys (optiscaler.h kOptiGameKeys) are not part of the installed file: ApplyProfile
        // writes them when the player turns the setting on.
        const wstring iniSrc = Join(OptiTempDir(ctx), L"OptiScaler.wronggpu.ini");
        WriteFileAtomic(iniSrc, ini.data(), ini.size());
        wstring packageFsr4;
        for (auto& rel : ctx.opti.files) {
            const bool fsr4 = IEq(rel, L"amd_fidelityfx_upscaler_dx12.dll");
            if (fsr4 && slot) { LogLine(ctx, L"OptiScaler's " + rel + L" not placed (OptiScaler 10: AMD's FSR comes from AMD's driver)"); continue; }
            if (!fsr4 && IsAmdFidelityFxName(rel) && (IsFile(Join(dir, rel)) || IsFile(Join(target.exeDir, BaseName(rel))))) {
                LogLine(ctx, L"OptiScaler's " + rel + L" not placed: the game has its own (left as it is)");
                continue;
            }
            const wstring dest = IEq(rel, L"OptiScaler.dll") ? proxy : fsr4 ? wstring(kAmdCompare) : rel;
            if (fsr4) packageFsr4 = Join(ctx.opti.dir, rel);
            optiFiles.push_back({dest, IEq(rel, L"OptiScaler.ini") ? iniSrc : Join(ctx.opti.dir, rel)});
        }
        // AMD's provider under kAmdCompare: the package's FSR 4; else FSR 4 found on this PC; else the game's own
        // amd_fidelityfx_dx12.dll (AMD's driver upgrades it to FSR 4). Copies; the originals stay.
        wstring provider;
        if (slot) {
        } else if (!packageFsr4.empty()) {
            provider = L"AMD FSR " + FileVersion(packageFsr4) + L" of the OptiScaler package (" + packageFsr4 + L")";
        } else if (!pcFsr4.path.empty()) {
            optiFiles.push_back({kAmdCompare, pcFsr4.path});
            provider = L"AMD FSR " + pcFsr4.version + L" found on this PC (" + pcFsr4.game + L": " + pcFsr4.path + L")";
        } else if (const wstring own = GameAmdFfxDll(ctx, dir, target.exeDir); !own.empty()) {
            optiFiles.push_back({kAmdCompare, own});
            provider = L"the game's own amd_fidelityfx_dx12.dll, version " + FileVersion(own) + L" (" + own + L"; FSR 4 when AMD Software upgrades it)";
        }
        if (slot) {
        } else if (provider.empty()) {
            const wstring w = L"FSR 4 / AMD versions not available: no AMD FidelityFX DLL was found (the OptiScaler package has no "
                              L"amd_fidelityfx_upscaler_dx12.dll, no FSR 4 DLL on this PC, the game has no amd_fidelityfx_dx12.dll). "
                              L"OptiScaler lists only WrongGPU.";
            LogLine(ctx, L"AMD provider: " + w);
            ui.log(w);
        } else {
            LogLine(ctx, L"AMD provider: " + provider + L", copied as " + kAmdCompare);
            ui.log(L"AMD's FSR versions in OptiScaler's list come from " + provider + L" (copied as " + kAmdCompare + L")");
        }
        LogLine(ctx, L"OptiScaler mode: OptiScaler " + ctx.opti.version + L" as " + proxy + L", WrongGPU as " + (slot ? kOptiSlotFrontEnd : kOptiFrontEnd) + L", " + std::to_wstring(optiFiles.size()) + L" OptiScaler files");
    }
    // 0.6.1: the other setups' plugins as the restore left the folder (the ones the earlier install moved aside are back):
    // each goes aside again only with the OK it had (the player's, or byte for byte the earlier install's move); any other
    // one keeps WrongGPU's RR out after all (nothing is moved without that OK)
    PluginPlan pluginsAside;
    if (rr) {
        pluginsAside = PlanPluginsAside(target.exeDir, dir, ForeignPlugins(target.exeDir, nullptr), pluginsOk, pluginsMovedBefore);
        if (!pluginsAside.blocking.empty()) { rr = false; pluginsKeepRrOut(pluginsAside); pluginsAside = PluginPlan(); }
    }
    if (rr) {
        // the RR plugin beside WrongGPU's plugin (OptiScaler's plugins folder, or the plugin's own next to Ultimate ASI
        // Loader), the RR runtime next to the exe, wronggpu-rr.ini (the RR data) and the plugin's settings file: from a
        // temporary copy, like OptiScaler's files (record entries: backed up, rolled back, removed by an uninstall)
        const wstring tmp = Join(OptiTempDir(ctx), L"rr");
        // the plugin built for the version of the game interface it hooks, where this build has one, else the general one;
        // by the game's own file, not by the game. Since 0.6.0 the general one serves the interfaces 2.7, 2.8 and 2.14 (it
        // picks its route at run time) and the release has no other; a per-version build still wins if a build has one
        wstring ifaceFile, forVersion;
        const wstring iface = GameInterfaceVersion(ctx, target.exe, &ifaceFile);
        const PayloadFile* plugin = RrPluginFor(ctx, iface, &forVersion);
        LogLine(ctx, L"Ray Reconstruction plugin: " + (!forVersion.empty() ? L"the build for game interface " + forVersion
                                                                          : wstring(L"the one build for game interfaces 2.7, 2.8 and 2.14 (it picks the route at run time)")) +
                         (iface.empty() ? L" (no game interface file found: the game calls Ray Reconstruction directly)"
                                        : L" (the game's " + ifaceFile + L" is " + iface + L")"));
        for (size_t i = 0; i < ctx.data->rrCount; ++i) {
            const PayloadFile& f = ctx.data->rrFiles[i];
            if (IsRrPayloadPlugin(f.rel) && &f != plugin) continue;
            DWORD sz = 0; const void* d = PayloadData(f, sz);
            const wstring src = Join(tmp, f.rel);
            EnsureDir(Parent(src));
            WriteFileAtomic(src, d, sz);
            optiFiles.push_back({IsRrPayloadPlugin(f.rel) ? Join(Parent(pluginRel), kRrPluginName) : wstring(f.rel), src});
        }
        const string rini = RrIniText(RrDataFile());
        const wstring rsrc = Join(tmp, kRrIni);
        WriteFileAtomic(rsrc, rini.data(), rini.size());
        optiFiles.push_back({kRrIni, rsrc});
        Profile rp = DefaultProfile();
        if (!LoadProfile(ProfilePath(dir), rp)) rp = DefaultProfile();
        const string sini = RenderRrSettings(rp, rrData, "");
        const wstring ssrc = Join(tmp, L"wronggpu.ini");
        WriteFileAtomic(ssrc, sini.data(), sini.size());
        optiFiles.push_back({kRrSettings, ssrc});
        LogLine(ctx, L"Ray Reconstruction: " + Join(Parent(pluginRel), kRrPluginName) + L", " + kRrRuntime + L" (the RR runtime is part of it), " + kRrIni + L" (data " + RrDataFile() +
                         (rrData ? L")" : L", not converted yet: RR off)") + L", " + kRrSettings);
    }
    const Conflicts now = FindConflicts(dir, target.exeDir);
    if (!now.outside.empty() || !now.legacyRecordDir.empty()) Fail(L"Files of another mod or an older install could not be handled automatically: " +
                                                                   (now.outside.empty() ? now.legacyRecordDir : now.outside[0]), EXIT_OTHER_MODS);
    // (0.6.1: other mods' files after the restore need no question here: the player's OK was checked before anything changed,
    // at the top, against target.conflicts. The restore leaves in place only files that were there then (another setup's
    // OptiScaler over WrongGPU's proxy is among them, so it was asked about), and what it brings back is byte for byte what
    // the earlier install moved aside or replaced with the player's OK (its disable entries, its copies over a file that
    // was there): those go aside again below on that OK.)
    Record m; m.version = ctx.data->version; m.game = dir; m.exe = target.exe;
    m.status = target.validated ? L"validated" : L"experimental"; m.regkey = ctx.perGameEntry ? RegKeyFor(dir) : L""; m.model = ModelFile(ctx); m.modelM = modelM.outcome;
    LogLine(ctx, L"target " + target.exe + L" | install folder " + dir + L" | status " + m.status);
    // Folders the payload needs that do not exist yet (shallow first).
    std::vector<wstring> newDirs;
    auto addDirs = [&](const wstring& rel) {
        std::vector<wstring> chain;
        for (wstring p = Parent(Target(dir, rel)); p.size() > dir.size() && !IsDir(p); p = Parent(p)) chain.push_back(p);
        for (auto it = chain.rbegin(); it != chain.rend(); ++it)
            if (std::find_if(newDirs.begin(), newDirs.end(), [&](const wstring& d) { return IEq(d, *it); }) == newDirs.end()) newDirs.push_back(*it);
    };
    for (auto& f : *ctx.data) if (payloadUsed(f)) addDirs(payloadRel(f));
    for (auto& of : optiFiles) addDirs(of.rel);
    for (auto& d : newDirs) { Entry e; e.action = L"dir"; e.rel = d.substr(dir.size() + 1); m.e.push_back(e); }
    for (auto* sl : replace) {  // the game's DLL keeps working under its new name; WrongGPU's forwards to it (spec 20)
        Entry e; e.action = L"rename"; e.rel = sl->original; e.inst = Sha256File(Join(dir, sl->name));
        const wstring tp = Target(dir, e.rel);
        if (Exists(tp)) { e.existed = true; e.orig = Sha256File(tp); }
        m.e.push_back(e);
    }
    for (auto& f : *ctx.data) {
        if (!payloadUsed(f)) continue;
        Entry e; e.action = L"copy"; e.rel = payloadRel(f); e.inst = f.sha256;
        wstring tp = Target(dir, e.rel);
        if (Exists(tp)) { e.existed = true; e.orig = Sha256File(tp); if (e.orig.empty()) Fail(L"Cannot read the existing file " + tp); }
        m.e.push_back(e);
    }
    for (auto& of : optiFiles) {
        Entry e; e.action = L"copy"; e.rel = of.rel; e.inst = Sha256File(of.src);
        if (e.inst.empty()) Fail(L"Cannot read OptiScaler's " + of.rel);
        wstring tp = Target(dir, e.rel);
        if (Exists(tp)) { e.existed = true; e.orig = Sha256File(tp); if (e.orig.empty()) Fail(L"Cannot read the existing file " + tp); }
        m.e.push_back(e);
    }
    // REFramework's mirror (_storage_\ next to the exe, gamerules.h): the game loads the root folder's DLLs from there, so
    // the install's root DLLs go there too, at once. A copy there that is not byte for byte ours is backed up and put back
    // by an uninstall; one that is ours already (the same bytes) counts as the install's own.
    std::vector<std::pair<wstring, wstring>> mirrors;  // (the file in the root, its copy in the mirror)
    if (IsDir(Join(dir, kMirrorDir)) && IEq(FullPath(dir), FullPath(target.exeDir))) {
        const size_t n = m.e.size();
        for (size_t i = 0; i < n; ++i) {
            const Entry src = m.e[i];
            if (src.action != L"copy" || !MirroredRel(src.rel)) continue;
            Entry e; e.action = L"copy"; e.rel = MirrorRel(src.rel); e.inst = src.inst;
            const wstring tp = Target(dir, e.rel);
            if (Exists(tp)) {
                e.orig = Sha256File(tp);
                if (e.orig.empty()) Fail(L"Cannot read the existing file " + tp);
                e.existed = e.orig != e.inst;
                if (!e.existed) e.orig.clear();
            }
            m.e.push_back(e);
            mirrors.push_back({src.rel, e.rel});
        }
        LogLine(ctx, L"REFramework's mirror (" + wstring(kMirrorDir) + L"): " + std::to_wstring(mirrors.size()) + L" of the install's DLLs go there too");
    }
    std::vector<wstring> disable;  // another mod's files, except those OptiScaler's own files replace (backed up as copies)
    for (auto& f : now.modFiles)
        if (!IsAmdFidelityFxName(f) && std::find_if(optiFiles.begin(), optiFiles.end(), [&](const OptiFile& o) { return IEq(o.rel, f); }) == optiFiles.end())
            disable.push_back(f);
    for (auto& f : now.legacyFiles) disable.push_back(f);
    // other setups' plugins, moved aside with the player's OK so WrongGPU's RR plugin is the only one (exactly those the plan
    // above found: the ones the player agreed to, and the ones the earlier install had moved aside; paths relative to the
    // install folder); one at a path this install writes is backed up by its copy entry instead
    if (rr)
        for (auto& rel : pluginsAside.moveAside) {
            if (std::find_if(optiFiles.begin(), optiFiles.end(), [&](const OptiFile& o) { return IEq(o.rel, rel); }) != optiFiles.end()) continue;
            bool written = false;
            for (auto& pf : *ctx.data) written = written || (payloadUsed(pf) && IEq(payloadRel(pf), rel));
            if (written || std::find_if(disable.begin(), disable.end(), [&](const wstring& d) { return IEq(d, rel); }) != disable.end()) continue;
            disable.push_back(rel);
            warnings.push_back(L"Moved aside with your OK: " + rel + L" (another OptiScaler plugin; kept in the backup and put back when WrongGPU is uninstalled).");
        }
    for (auto& name : disable) {
        wstring tp = Target(dir, name);
        if (!IsFile(tp)) continue;
        Entry e; e.action = L"disable"; e.rel = name; e.existed = true; e.orig = Sha256File(tp);
        if (e.orig.empty()) Fail(L"Cannot read the existing file " + tp);
        m.e.push_back(e);
        // a file the restore left in place (another setup's, over WrongGPU's) that the switch now moves aside: said as that
        const auto k = keptByRestore.find(Lower(name));
        if (k != keptByRestore.end())
            for (auto& w : warnings)
                if (w == k->second) w = L"Moved aside with your OK: " + name + L" (another setup's file that had replaced WrongGPU's; kept in the backup and put back when WrongGPU is uninstalled).";
    }
    // another setup's file the restore left in place (its OptiScaler.ini, beside its OptiScaler) that this install now replaces with the
    // switch question's OK: backed up (an entry that existed) and said as that
    for (auto& e : m.e) {
        const auto k = e.action == L"copy" && e.existed ? keptByRestore.find(Lower(e.rel)) : keptByRestore.end();
        if (k != keptByRestore.end())
            for (auto& w : warnings)
                if (w == k->second) w = L"Replaced with your OK: " + e.rel + L" (another setup's; kept in the backup and put back when WrongGPU is uninstalled).";
    }
    InferOwners(ctx, m);  // record v2: who owns each file, what this install wrote, the proxy and why
    SetPayloadLine(ctx, m);
    if (opti) {
        m.proxy = proxy; m.proxyReason = proxyReason;
        for (auto& f : co.findings) m.coexist.push_back({FromUtf8(ToolKey(f.tool)), f.what + (f.evidence.empty() ? wstring() : L" [" + f.evidence + L"]"), DecisionName(f.decision)});
    }
    int backups = 0; for (auto& e : m.e) if (e.existed) ++backups;
    int total = backups + (int)m.e.size() + 3, done = 1;
    ui.step(done, total, L"Backing up existing files");

    const wstring state = StateDir(dir), rpath = RecordPath(dir);
    EnsureDir(state);
    bool recorded = false;
    try {
        for (auto& e : m.e) {
            if (!e.existed) continue;
            wstring b = BackupPath(dir, e.rel);
            CopyChecked(Target(dir, e.rel), b);
            if (Sha256File(b) != e.orig) Fail(L"Backup verification failed for " + e.rel);
            ui.step(++done, total, L"Backed up " + e.rel);
            ui.log(L"backup " + e.rel);
        }
        SaveRecord(rpath, m);  // written before any change so an interrupted install can still be undone
        recorded = true;
        for (auto& e : m.e) {
            const wstring tp = Target(dir, e.rel);
            if (e.action == L"dir") {
                ui.step(++done, total, L"Creating " + e.rel);
                EnsureDir(tp);
            } else if (e.action == L"disable") {
                ui.step(++done, total, L"Moving aside " + e.rel);
                ForceDelete(tp);
                ui.log(L"moved aside " + e.rel + L" (original kept in the backup)");
            } else if (e.action == L"rename") {
                const FfxSlot* sl = nullptr;
                for (auto* r : replace) if (IEq(r->original, e.rel)) sl = r;
                if (!sl) Fail(L"Internal error: no FSR DLL for " + e.rel);
                const wstring src = Join(dir, sl->name);
                ui.step(++done, total, L"Keeping the game's " + wstring(sl->name) + L" as " + e.rel);
                if (Exists(tp)) ForceDelete(tp);
                if (!MoveFileExW(src.c_str(), tp.c_str(), MOVEFILE_WRITE_THROUGH)) Fail(L"Cannot rename " + src + L": " + ErrText(GetLastError()) + L". Is the game running?");
                if (Sha256File(tp) != e.inst) Fail(L"Verification failed after renaming " + wstring(sl->name));
                ui.log(L"renamed " + wstring(sl->name) + L" to " + e.rel);
            }
        }
        for (auto& f : *ctx.data) {
            if (!payloadUsed(f)) continue;
            const wstring rel = payloadRel(f);
            ui.step(++done, total, L"Installing " + rel);
            DWORD sz = 0; const void* d = PayloadData(f, sz);
            const wstring tp = Target(dir, rel);
            WriteFileAtomic(tp, d, sz);
            if (Sha256File(tp) != f.sha256) Fail(L"Verification failed after writing " + rel);
            ui.log(L"installed " + rel);
        }
        for (auto& of : optiFiles) {
            const bool ours = StartsWithI(BaseName(of.rel), L"wronggpu");  // the settings file and the Ray Reconstruction files
            ui.step(++done, total, (ours ? L"Installing " : L"Installing OptiScaler: ") + of.rel);
            const wstring tp = Target(dir, of.rel);
            CopyChecked(of.src, tp);
            ui.log((ours ? L"installed " : L"installed OptiScaler's ") + of.rel);
        }
        for (auto& mr : mirrors) {
            ui.step(++done, total, L"Installing " + mr.second);
            CopyChecked(Target(dir, mr.first), Target(dir, mr.second));
            ui.log(L"installed " + mr.second + L" (REFramework's copy of " + mr.first + L")");
        }
        if (opti) ui.log(L"OptiScaler " + ctx.opti.version + L" is set up for WrongGPU (OptiScaler.dll as " + proxy +
                         (slot ? L"; WrongGPU is its own upscaler there, selected). " : L"; WrongGPU is its default FSR upscaler). ") +
                         L"In the game, choose NVIDIA DLSS (or Intel XeSS) as the upscaler; OptiScaler's menu (Insert) switches upscalers.");
        ui.step(++done, total, L"Creating the uninstaller");
        // The uninstaller: the small restorer (WrongGPU-Restore.exe) the launcher embeds, else a copy of this program.
        if (ctx.restorer && ctx.restorerSize) {
            const wstring un = Join(state, L"uninstall.exe");
            WriteFileAtomic(un, ctx.restorer, ctx.restorerSize);
            if (Sha256File(un) != Sha256Bytes(ctx.restorer, ctx.restorerSize)) Fail(L"Verification failed after writing the uninstaller");
        } else {
            CopyChecked(SelfPath(), Join(state, L"uninstall.exe"));
        }
        if (ctx.perGameEntry) RegisterUninstall(ctx, m, state, target.name);
        WriteOwnedFiles(dir, m);
#if WRONGGPU_PRESET_K
        if (opti && !slot) SeedAsiSelection(ctx, dir, ctx.presetK);
#endif
        ui.log(L"WrongGPU installed in " + dir + L" (" + m.status + L"); model data: " + ModelFile(ctx));
        ui.step(total, total, L"Done");
    } catch (const SetupError& ex) {
        ui.log(L"FAILED: " + ex.msg + L" - rolling back");
        try {
            // (rc6 L4: the settings that were there before this install stay: the failed install's record and backup go only)
            if (recorded) DoRestore(ctx, dir, ui, true, keepSettings);
            else if (!keepSettings) RemoveTree(state);
            if (keepSettings && !IsFile(rpath)) RemoveStateKeepingSettings(dir);  // (a backup DoRestore had to keep stays with its record)
            ui.log(L"Rollback complete; the game folder is back to its previous state.");
        } catch (const SetupError& r) { ui.log(L"Rollback problem: " + r.msg); }
        throw;
    }
    if (!ctx.logFile.empty() && IsFile(ctx.logFile)) CopyFileW(ctx.logFile.c_str(), Join(state, L"setup.log").c_str(), FALSE);
    return warnings;
}

}  // namespace wg
