// SPDX-License-Identifier: MIT
// Unit tests of the launcher's part of wg_core (build/launcher-design.md section 7.1) that need no game, no payload and
// no GPU: settings validation and rendering (runtime_options as one line without spaces, enum rejection, the last value
// wins, runtime keys on lines of their own), ini round trips that keep comments, order and unknown keys, outside-edit
// detection, the profile file, record v1 -> v2 (owners, payload line, owned-files.json, a 0.5.x reader still reads
// it), the log-tail parser on synthetic logs, the job and result files, the update reply, the game environment, Ray
// Reconstruction (its settings, the RR settings file round trip, which games use it, the RR log).
// Written for this project. Run by ctest ("launcher-core").
#include <cstdio>
#include <iostream>
#include <stdexcept>

#include "app/ui_palette.h"
#include "wg_core/detect.h"
#include "wg_core/gamerules.h"
#include "wg_core/launcher.h"
#include "wg_core/logtail.h"
#include "wg_core/optiedition.h"
#include "wg_core/pe.h"
#include "wg_core/profile.h"
#include "wg_core/record.h"
#include "wg_core/rr.h"
#include "wg_core/rrconfig.h"
#include "wg_core/settings.h"

using namespace wg;

namespace {
int checks = 0;
void require(bool ok, const char* what) {
    if (!ok) throw std::runtime_error(what);
    ++checks;
}
wstring TempFolder() {
    wchar_t tmp[MAX_PATH] = {};
    GetTempPathW(MAX_PATH, tmp);
    wstring dir = Join(tmp, L"launcher-core-tests-" + std::to_wstring(GetCurrentProcessId()));
    RemoveTree(dir);
    EnsureDir(dir);
    return dir;
}
string OptionsOf(const string& ini) { string v, s; IniGetFrontEnd(ini, "runtime_options", &v, &s); return v; }
}  // namespace

int main() {
    try {
        // ---- validation -----------------------------------------------------------------------------------------
        string n; wstring why;
        const SettingDef* exec = FindSetting("executor");
        const SettingDef* filter = FindSetting("output_filter");
        const SettingDef* sparkle = FindSetting("sparkle_guard");
        require(exec && filter && sparkle && FindSetting("wronggpu") && FindSetting("pre_exposure") && FindSetting("fps_overlay"), "the Basic tier has its six settings");
        require(ValidateSetting(*exec, "D3D12", &n, &why) && n == "dx12", "executor d3d12 -> dx12");
        require(!ValidateSetting(*exec, "vulkan", &n, &why) && why.find(L"auto, dx12, hip") != wstring::npos, "a bad executor is refused and the allowed values named");
        require(!ValidateSetting(*filter, "quality7", &n, &why), "output_filter: only the runtime's values");
        require(ValidateSetting(*filter, "quality9", &n, &why) && n == "quality9", "output_filter quality9");
        require(ValidateSetting(*sparkle, "0.25", &n, &why) && n == "0.25", "sparkle_guard rate");
        require(ValidateSetting(*sparkle, "0", &n, &why) && n == "off" && ValidateSetting(*sparkle, "on", &n, &why) && n == "on", "sparkle_guard 0 = off, on");
        require(!ValidateSetting(*sparkle, "1.5", &n, &why) && !ValidateSetting(*sparkle, "-0.1", &n, &why) && !ValidateSetting(*sparkle, "abc", &n, &why), "sparkle_guard rate must be in (0, 1)");
        Profile p = DefaultProfile();
        require(!SetSetting(p, "nosuchkey", "1", &why) && !SetSetting(p, "pre_exposure", "bright", &why), "unknown keys and bad values are refused by set");
        require(SetSetting(p, "Executor", " HIP ", &why) && p.get("executor") == "hip", "set trims and lowers");

        // ---- rendering wronggpu.ini ---------------------------------------------------------------------------
        const string front = "# WrongGPU settings\r\n[fsr]\r\n# who upscales\r\nupscaler = network\r\nruntime_dll = wronggpu_runtime.dll\r\n"
                             "runtime_options = model=M ; the model\r\nfoo_unknown = 7\r\n[overlay]\r\nexecutor = keep-me\r\n";
        p = DefaultProfile();
        string out;
        require(RenderWronggpuIni(p, front, out, &why) && out == front, "the default profile changes nothing (defaults are never written)");
        SetSetting(p, "executor", "dx12", nullptr);
        SetSetting(p, "output_filter", "quality5", nullptr);
        SetSetting(p, "sparkle_guard", "0.3", nullptr);
        require(RenderWronggpuIni(p, front, out, &why), "render");
        require(OptionsOf(out) == "model=M;executor=dx12;output_filter=quality5;sparkle_guard=0.3", "runtime_options: unmanaged items first, managed ones appended in order");
        require(out.find("runtime_options=model=M;executor=dx12;output_filter=quality5;sparkle_guard=0.3\r\n") != string::npos, "runtime_options is one line with no spaces");
        require(out.find("# who upscales\r\n") != string::npos && out.find("foo_unknown = 7") != string::npos && out.find("[overlay]\r\nexecutor = keep-me") != string::npos,
                "comments, unknown keys and other sections are kept");
        string again;
        require(RenderWronggpuIni(p, out, again, &why) && again == out, "rendering twice gives the same file (round trip)");
        auto eff = EffectiveSettings(out, "");
        require(eff["executor"] == "dx12" && eff["output_filter"] == "quality5" && eff["sparkle_guard"] == "0.3" && eff["wronggpu"] == "on", "the file reads back as the profile");
        // a runtime key on a line of its own wins (the front end appends it); rendering removes such lines
        const string loose = "[fsr]\r\nruntime_options = executor=hip;output_filter=analytic;executor=dx12\r\noutput_filter = quality9\r\n";
        eff = EffectiveSettings(loose, "");
        require(eff["executor"] == "dx12", "the last value of a key in runtime_options wins");
        require(eff["output_filter"] == "quality9", "a runtime key on its own line comes after runtime_options and wins");
        Profile q = DefaultProfile();
        SetSetting(q, "output_filter", "original", nullptr);
        require(RenderWronggpuIni(q, loose, out, &why) && OptionsOf(out) == "output_filter=original" && out.find("output_filter = quality9") == string::npos,
                "rendering folds the loose line away and writes one value");
        // WrongGPU off
        Profile off = DefaultProfile();
        SetSetting(off, "wronggpu", "off", nullptr);
        require(RenderWronggpuIni(off, front, out, &why) && out.find("upscaler=amd") != string::npos && out.find("upscaler = network") == string::npos, "off writes upscaler=amd in place");
        require(RenderWronggpuIni(DefaultProfile(), out, again, &why) && EffectiveSettings(again, "")["wronggpu"] == "on", "on again: upscaler=network");
        // a profile value that does not validate is never written
        Profile bad = DefaultProfile();
        bad.v["executor"] = "warp";
        require(!RenderWronggpuIni(bad, front, out, &why) && why.find(L"warp") != wstring::npos, "an invalid profile value stops the render");
        // empty file: nothing for defaults, an [fsr] section for the rest
        require(RenderWronggpuIni(DefaultProfile(), "", out, &why) && out.empty(), "defaults into an empty file: nothing");
        require(RenderWronggpuIni(p, "", out, &why) && OptionsOf(out) == "executor=dx12;output_filter=quality5;sparkle_guard=0.3", "a new [fsr] section");

        // ---- outside edits ------------------------------------------------------------------------------------------
        Profile prof = DefaultProfile();
        SetSetting(prof, "output_filter", "quality5", nullptr);
        const string edited = "[fsr]\r\nruntime_options = output_filter=quality9;pre_exposure=scale\r\n";
        auto edits = OutsideEdits(prof, edited, "");
        require(edits.size() == 1 && edits[0].key == "output_filter" && edits[0].fileValue == "quality9" && edits[0].profileValue == "quality5",
                "a value that differs from the default and the profile is an outside edit");
        require(OutsideEdits(prof, "[fsr]\r\nruntime_options = output_filter=default\r\n", "").empty(), "a value equal to its default is ignored");
        require(OutsideEdits(prof, "[fsr]\r\nruntime_options = output_filter=quality5\r\n", "").empty(), "the profile's own value is not an outside edit");

        // ---- OptiScaler.ini -------------------------------------------------------------------------------------------
        const string opti = "[Menu]\r\n; the overlay\r\nShowFps=auto\r\nShortcutKey=auto\r\n[FSR]\r\nUpscalerIndex=0\r\n";
        Profile fps = DefaultProfile();
        require(RenderOptiIni(fps, opti, out) && out == opti, "fps_overlay keep: OptiScaler.ini unchanged");
        SetSetting(fps, "fps_overlay", "on", nullptr);
        require(RenderOptiIni(fps, opti, out) && out.find("ShowFps=true\r\n") != string::npos && out.find("; the overlay\r\n") != string::npos && out.find("ShortcutKey=auto") != string::npos,
                "fps_overlay on: ShowFps=true, the rest kept");
        require(EffectiveSettings("", out)["fps_overlay"] == "on", "ShowFps reads back");
        std::vector<wstring> notes;
        const string noKey = "[Menu]\r\nShortcutKey=auto\r\n";
        require(RenderOptiIni(fps, noKey, out, &notes) && out == noKey && notes.size() == 1, "a key the OptiScaler build does not have is not added");
        require(IniHasKey(opti, "menu", "showfps") && !IniHasKey(opti, "FSR", "ShowFps"), "IniHasKey: section and key without case, only in that section");

        // ---- import (adoption) ----------------------------------------------------------------------------------------
        const Profile imp = ImportProfile("[fsr]\r\nupscaler = standin\r\nruntime_options = model=M;output_filter=quality9;timing_log=1\r\n", "", &notes);
        require(imp.get("wronggpu") == "on" && imp.get("output_filter") == "quality9" && imp.get("executor") == "auto", "import: standin counts as on, the player's filter kept");

        // ---- profile.ini -------------------------------------------------------------------------------------------------
        const wstring dir = TempFolder();
        const wstring pp = ProfilePath(dir);
        Profile saved = DefaultProfile();
        SetSetting(saved, "pre_exposure", "original", nullptr);
        saved.v["future_key"] = "x";
        SaveProfile(pp, saved);
        Profile loaded;
        require(LoadProfile(pp, loaded) && loaded.get("pre_exposure") == "original" && loaded.get("future_key") == "x" && loaded.get("executor") == "auto",
                "profile.ini round trip keeps unknown keys");
        { const string broken = "[profile]\r\nexecutor=warp\r\n"; WriteFileAtomic(pp, broken.data(), broken.size()); }
        std::vector<wstring> problems;
        require(LoadProfile(pp, loaded, &problems) && loaded.get("executor") == "auto" && problems.size() == 1, "a bad profile value falls back to the default, reported");

        // ---- record v1 -> v2 -------------------------------------------------------------------------------------------
        const wstring game = Join(dir, L"game");
        EnsureDir(StateDir(game));
        const string v1 = "WRONGGPU-RECORD 1\r\nversion\t0.5.1\r\ngame\t" + ToUtf8(game) + "\r\nexe\t" + ToUtf8(Join(game, L"Game.exe")) +
                          "\r\nstatus\texperimental\r\nregkey\tWrongGPU-12345678\r\nmodel\tm.wgcm\r\n"
                          "dir\t0\t-\t-\tplugins\r\ncopy\t0\t-\taaa\twronggpu_ffx_dx12.dll\r\ncopy\t0\t-\tbbb\twronggpu_runtime.dll\r\n"
                          "copy\t0\t-\tccc\tplugins\\WrongGPU.asi\r\ncopy\t0\t-\tddd\tdxgi.dll\r\ncopy\t1\teee\tfff\tOptiScaler.ini\r\n"
                          "copy\t0\t-\t111\twronggpu.ini\r\ndisable\t1\t999\t-\twinmm.dll\r\n";
        WriteFileAtomic(RecordPath(game), v1.data(), v1.size());
        Context ctx;
        static const EmbeddedData noPayload{nullptr, 0, 0, L"", 0, L"", 0, L"0.5.5"};
        ctx.data = &noPayload;
        Record r1;
        require(LoadRecord(RecordPath(game), r1) && r1.format == 1 && r1.e.size() == 8, "v1 record read");
        require(UpgradeRecord(ctx, game, &notes), "v1 -> v2");
        require(IsFile(RecordPath(game) + L".v1"), "the v1 record is kept as install.record.v1");
        string rawv1; ReadAll(RecordPath(game) + L".v1", rawv1);
        require(rawv1 == v1, "install.record.v1 is the old file byte for byte");
        Record r2;
        require(LoadRecord(RecordPath(game), r2) && r2.format == 2 && r2.e.size() == 8 && r2.regkey == L"WrongGPU-12345678", "v2 record read back");
        auto owner = [&](const wchar_t* rel) { for (auto& e : r2.e) if (IEq(e.rel, rel)) return e.owner; return wstring(L"?"); };
        require(owner(L"wronggpu_ffx_dx12.dll") == L"wg" && owner(L"wronggpu_runtime.dll") == L"wg" && owner(L"plugins\\WrongGPU.asi") == L"wg" && owner(L"plugins") == L"wg",
                "owners: our payload is wg");
        require(owner(L"dxgi.dll") == L"wg-optibundle" && owner(L"OptiScaler.ini") == L"wg-optibundle", "owners: OptiScaler's files of the bundle are wg-optibundle");
        require(owner(L"winmm.dll") == L"moved" && owner(L"wronggpu.ini") == L"wg", "owners: a file moved aside is moved");
        require(r2.payloadVersion == L"0.5.1" && r2.runtimeSha == L"bbb" && r2.frontEndSha == L"aaa" && r2.pluginSha == L"ccc" && r2.proxy == L"dxgi.dll",
                "payload and proxy lines from the v1 entries");
        require(!UpgradeRecord(ctx, game, nullptr), "a v2 record is not upgraded again");
        string rawv2; ReadAll(RecordPath(game), rawv2);
        require(rawv2.rfind("WRONGGPU-RECORD 2\r\n", 0) == 0, "v2 header");
        // a 0.5.x reader (header prefix, five-field action lines) still sees every action line of a v2 record
        size_t actions = 0;
        for (size_t a = 0; a < rawv2.size();) {
            size_t e = rawv2.find('\n', a); if (e == string::npos) e = rawv2.size();
            const string line = rawv2.substr(a, e - a); a = e + 1;
            size_t tabs = 0; for (char c : line) tabs += c == '\t';
            const string act = line.substr(0, line.find('\t'));
            if ((act == "copy" || act == "disable" || act == "rename" || act == "dir") && tabs == 4) ++actions;
        }
        require(actions == 8, "v2 keeps the five-field action lines of v1");
        // a line of a later version survives a rewrite
        string withExtra = rawv2 + "coexist\tReShade\tfp\twarn\r\n";
        WriteFileAtomic(RecordPath(game), withExtra.data(), withExtra.size());
        Record r3; LoadRecord(RecordPath(game), r3); SaveRecord(RecordPath(game), r3);
        string rawv3; ReadAll(RecordPath(game), rawv3);
        require(rawv3.find("coexist\tReShade\tfp\twarn\r\n") != string::npos, "unknown v2 lines are written back");
        const string js = OwnedFilesJson(r2);
        require(js.find("\"format\": \"wronggpu-owned-files/1\"") != string::npos && js.find("\"owner\": \"wg-optibundle\"") != string::npos &&
                js.find("plugins\\\\WrongGPU.asi") != string::npos && js.find("\"original_sha256\": \"999\"") != string::npos, "owned-files.json");
        // VerifyInstall
        { const string b = "runtime"; WriteFileAtomic(Join(game, L"wronggpu_runtime.dll"), b.data(), b.size()); }
        auto vp = VerifyInstall(game, r2);
        require(std::find(vp.begin(), vp.end(), wstring(L"changed: wronggpu_runtime.dll")) != vp.end() &&
                std::find(vp.begin(), vp.end(), wstring(L"missing: wronggpu_ffx_dx12.dll")) != vp.end(), "VerifyInstall: changed and missing files");

        // ---- the payload line names the RR plugin build (rc6 L6): another RR plugin build of the same version is another build ----
        {
            static const PayloadFile pfiles[] = {{1, L"wronggpu_runtime.dll", L"r1", 1}, {2, L"amd_fidelityfx_dx12.dll", L"f1", 1}};
            static const PayloadFile prr[] = {{3, L"plugins\\WrongGPU-RR.asi", L"p1", 1}};
            EmbeddedData pd{pfiles, 2, 0, L"", 0, L"", 0, L"0.6.0"};
            pd.rrFiles = prr; pd.rrCount = 1;
            Context pc; pc.data = &pd;
            const wstring pg = Join(dir, L"payload-line");
            Record m; m.version = L"0.6.0"; m.game = pg; m.exe = Join(pg, L"Game.exe");
            Entry rre; rre.action = L"copy"; rre.rel = L"plugins\\WrongGPU-RR.asi"; rre.inst = L"p1"; m.e.push_back(rre);
            SetPayloadLine(pc, m);
            require(m.rrPluginSha == L"p1" && m.runtimeSha == L"r1" && !PayloadDiffers(pc, m), "payload line: the RR plugin build this payload has for the game");
            SaveRecord(RecordPath(pg), m);
            string rawp; ReadAll(RecordPath(pg), rawp);
            require(rawp.find("payload\t0.6.0\tr1\tf1\t-\tp1\r\n") != string::npos, "payload line: the RR plugin's hash is its sixth field");
            Record back;
            require(LoadRecord(RecordPath(pg), back) && back.rrPluginSha == L"p1" && back.pluginSha.empty() && !PayloadDiffers(pc, back), "payload line: read back");
            back.rrPluginSha = L"p0";
            require(PayloadDiffers(pc, back), "payload line: another RR plugin build of the same version differs");
            back.rrPluginSha.clear();
            back.e[0].inst = L"p0";
            require(PayloadDiffers(pc, back), "payload line before rc6 (no sixth field): the RR plugin entry's hash decides");
            back.e[0].inst = L"p1";
            require(!PayloadDiffers(pc, back), "payload line before rc6: the same RR plugin is the same build");
            Record norr; norr.version = L"0.6.0"; norr.game = pg; norr.exe = m.exe;
            SetPayloadLine(pc, norr);
            SaveRecord(RecordPath(pg), norr);
            ReadAll(RecordPath(pg), rawp);
            require(norr.rrPluginSha.empty() && rawp.find("payload\t0.6.0\tr1\tf1\t-\r\n") != string::npos && !PayloadDiffers(pc, norr),
                    "payload line: an install without the RR files keeps five fields");
        }

        // ---- a moved game folder (rc6 L1): the record names the old folder; it is the new folder's record all the same ------
        {
            const wstring oldLad = EnvVar(L"LOCALAPPDATA"), lad = Join(dir, L"moved-localappdata");
            auto file = [](const wstring& p, const string& text) { EnsureDir(Parent(p)); WriteFileAtomic(p, text.data(), text.size()); };
            file(Join(lad, L"WrongGPU\\launcher.ini"), "[launcher]\r\n");  // a launcher uses the shared data: an uninstall leaves it
            SetEnvironmentVariableW(L"LOCALAPPDATA", lad.c_str());
            const wstring oldDir = Join(dir, L"OldLib\\steamapps\\common\\Moved Game"), newDir = Join(dir, L"NewLib\\steamapps\\common\\Moved Game");
            const string rt = "runtime", ini = "[FSR]\r\nFfxDx12SRPath = " + ToUtf8(oldDir) + "\\wronggpu_ffx_dx12.dll\r\nUpscalerIndex=0\r\n[Plugins]\r\nPath=" +
                                               ToUtf8(oldDir) + "\\plugins\r\n";
            file(Join(newDir, L"Game.exe"), "MZ");
            file(Join(newDir, L"wronggpu_runtime.dll"), rt);
            file(Join(newDir, L"OptiScaler.ini"), ini);
            Record m; m.version = L"0.6.0"; m.game = oldDir; m.exe = Join(oldDir, L"Game.exe"); m.status = L"experimental"; m.model = L"m.wgcm";
            Entry a; a.action = L"copy"; a.rel = L"wronggpu_runtime.dll"; a.inst = Sha256Bytes(rt.data(), rt.size()); a.owner = L"wg"; m.e.push_back(a);
            Entry b; b.action = L"copy"; b.rel = L"OptiScaler.ini"; b.inst = Sha256Bytes(ini.data(), ini.size()); b.owner = L"wg-optibundle"; m.e.push_back(b);
            IniKeyChange k; k.file = L"OptiScaler.ini"; k.section = "Plugins"; k.key = "Path"; k.original = "auto"; k.ours = ToUtf8(oldDir) + "\\plugins"; m.inikeys.push_back(k);
            SaveRecord(RecordPath(newDir), m);
            AddManagedGame(oldDir);
            Record r;
            require(LoadRecord(RecordPath(newDir), r) && IEq(r.game, newDir) && IEq(r.exe, Join(newDir, L"Game.exe")) && IEq(r.movedFrom, oldDir),
                    "moved game: the record read from the new folder names it (game, exe), movedFrom the old one");
            require(r.inikeys.size() == 1 && r.inikeys[0].ours == ToUtf8(newDir) + "\\plugins" && r.inikeys[0].original == "auto", "moved game: the journal's paths follow");
            // another drive letter: only the changed part of the path is replaced
            Record d = m; d.game = L"Q:\\Games\\Moved Game"; d.exe = L"Q:\\Games\\Moved Game\\Game.exe"; d.model = L"Q:\\model.wgcm";
            SaveRecord(RecordPath(newDir), d);
            Record dr;
            require(LoadRecord(RecordPath(newDir), dr) && IEq(dr.exe, Join(newDir, L"Game.exe")) && dr.model == L"Q:\\model.wgcm" && dr.inikeys[0].ours == ToUtf8(oldDir) + "\\plugins",
                    "moved game: a drive letter change; the model path and values of other folders stay");
            Record own = m, same;
            own.game = newDir; own.exe = Join(newDir, L"Game.exe");
            SaveRecord(RecordPath(newDir), own);
            require(LoadRecord(RecordPath(newDir), same) && same.movedFrom.empty() && same.game == newDir, "moved game: a record of its own folder is not rebased");
            // written back: the record, OptiScaler.ini's paths, the managed list (the old folder is gone)
            SaveRecord(RecordPath(newDir), m);
            Context mc; mc.logFile = Join(dir, L"moved.log");
            mc.data = &noPayload;  // an empty payload table (the scan and EntryFor read it)
            require(LoadRecord(RecordPath(newDir), r) && SaveRebasedRecord(mc, newDir, r) && r.movedFrom.empty(), "moved game: the rebased record is written");
            Record savedRec;
            string rawRec, iniNow;
            ReadAll(RecordPath(newDir), rawRec);
            require(rawRec.find("game\t" + ToUtf8(newDir) + "\r\n") != string::npos && LoadRecord(RecordPath(newDir), savedRec) && savedRec.movedFrom.empty(), "moved game: the saved record names the new folder");
            require(ReadAll(Join(newDir, L"OptiScaler.ini"), iniNow) && iniNow.find("FfxDx12SRPath = " + ToUtf8(newDir) + "\\wronggpu_ffx_dx12.dll\r\n") != string::npos &&
                        iniNow.find("Path=" + ToUtf8(newDir) + "\\plugins\r\n") != string::npos && iniNow.find("UpscalerIndex=0\r\n") != string::npos,
                    "moved game: OptiScaler.ini's paths into the old folder point into the new one");
            require(savedRec.e[1].inst == Sha256Bytes(iniNow.data(), iniNow.size()), "moved game: the record's OptiScaler.ini hash follows the rewritten file");
            auto managed = [&](const wstring& f) { for (auto& l : ManagedGames()) if (IEq(FullPath(l), FullPath(f))) return true; return false; };
            require(managed(newDir) && !managed(oldDir), "moved game: the new folder is on the managed list in the old one's place");
            // Uninstall (and uninstall.exe) of a record that still names the old folder: no refusal, the new folder's files go
            SaveRecord(RecordPath(newDir), m);
            Ui quiet{[](int, int, const wstring&) {}, [](const wstring&) {}};
            bool refused = false;
            try { DoRestore(mc, newDir, quiet, false, false); } catch (const SetupError&) { refused = true; }
            require(!refused && !IsFile(Join(newDir, L"wronggpu_runtime.dll")) && !IsDir(StateDir(newDir)) && IsFile(Join(newDir, L"Game.exe")),
                    "moved game: the uninstall works in the new folder");
            // the scan takes managed folders whose record is gone off the list, unless their drive isn't connected
            wstring offline;
            const DWORD drives = GetLogicalDrives();
            for (wchar_t c = L'Z'; c >= L'D' && offline.empty(); --c) if (!(drives & (1u << (c - L'A'))) ) offline = wstring(1, c) + L":\\Games\\Offline Game";
            AddManagedGame(oldDir);
            if (!offline.empty()) AddManagedGame(offline);
            GlobalSettings sgs;
            ScanGames(mc, sgs, true);
            require(!managed(oldDir) && (offline.empty() || managed(offline)), "moved game: the scan drops a managed folder without its record (not one on a drive that is not connected)");

            // ---- rc6 L2: an install next to a launcher stub (before rc5) is in the wrong place; a stub alone is its game ----
            const wstring cp = Join(dir, L"Stub Lib\\Cyberpunk 2077"), stub = Join(cp, L"REDprelauncher.exe"), real = Join(cp, L"bin\\x64\\Cyberpunk2077.exe");
            file(stub, "MZ stub");
            file(real, "MZ game");
            file(Join(cp, L"bin\\x64\\amd_fidelityfx_dx12.dll"), "MZ");
            const wstring hl = Join(dir, L"Stub Lib\\Hogwarts Legacy"), starter = Join(hl, L"HogwartsLegacy.exe"), hlGame = Join(hl, L"Phoenix\\Binaries\\Win64\\HogwartsLegacy.exe");
            file(starter, "MZ starter");
            file(hlGame, "MZ game");
            require(IEq(MisplacedFor(stub), real) && IEq(MisplacedFor(starter), hlGame), "MisplacedFor: a launcher stub's game, a root starter's game");
            require(MisplacedFor(real).empty() && MisplacedFor(hlGame).empty() && MisplacedFor(Join(newDir, L"Game.exe")).empty(), "MisplacedFor: none for the game itself");
            const GameEntry alone = EntryFor(mc, stub, L"", L"Added");
            require(IEq(alone.exe, real) && !alone.hasRecord && alone.realExe.empty(), "a stub without an install next to it is read as its game");
            Record sr; sr.version = L"0.6.0"; sr.game = cp; sr.exe = stub; sr.status = L"validated";
            Entry se; se.action = L"copy"; se.rel = L"wronggpu_runtime.dll"; se.inst = Sha256Bytes(rt.data(), rt.size()); se.owner = L"wg"; sr.e.push_back(se);
            file(Join(cp, L"wronggpu_runtime.dll"), rt);
            file(Join(cp, L"wronggpu.ini"), "[fsr]\r\n");
            SaveRecord(RecordPath(cp), sr);
            AddManagedGame(cp);
            const GameEntry wrong = EntryFor(mc, stub, L"", L"Managed");
            require(wrong.hasRecord && IEq(wrong.exe, stub) && IEq(wrong.realExe, real) && wrong.status == GameStatus::NeedsRepair && !wrong.problems.empty() &&
                        IsMisplacedProblem(wrong.problems[0]) && wrong.problems[0].find(L"bin\\x64\\Cyberpunk2077.exe") != wstring::npos,
                    "an install next to the stub: Needs repair, in the wrong place, the game's program named");
            sgs.addedGames = {real};
            const auto listed = ScanGames(mc, sgs, true);
            size_t cpEntries = 0;
            for (auto& g : listed) cpEntries += IEq(g.exe, stub) || IEq(g.exe, real);
            require(cpEntries == 1, "the scan lists the game once: the wrong-place install's entry, not the game a second time");
            RemoveTree(cp); RemoveManagedGame(cp);

            // ---- rc6 L4: what a failed install or update leaves in the state folder: the player's settings only ----------------
            const wstring fs = Join(dir, L"failed-state");
            require(!StateHasSettings(fs), "L4: no state folder, no settings");
            file(Join(StateDir(fs), L"uninstall.exe"), "MZ");
            file(Join(StateDir(fs), L"owned-files.json"), "{}");
            file(Join(StateDir(fs), L"backup\\x.dll"), "x");
            require(!StateHasSettings(fs), "L4: an uninstaller and a backup are no settings");
            RemoveStateKeepingSettings(fs);
            require(!IsDir(StateDir(fs)), "L4: without settings the state folder goes");
            file(Join(StateDir(fs), L"uninstall.exe"), "MZ");
            file(ProfilePath(fs), "[profile]\r\n");
            file(Join(StateDir(fs), L"previous\\previous.txt"), "version=0.5.9\r\n");
            file(Join(StateDir(fs), L"rr-config\\Engine.ini"), "[x]\r\n");
            file(Join(StateDir(fs), L"backup\\x.dll"), "x");
            require(StateHasSettings(fs), "L4: profile.ini, previous\\ and rr-config\\ are settings");
            RemoveStateKeepingSettings(fs);
            require(IsFile(ProfilePath(fs)) && IsFile(Join(StateDir(fs), L"previous\\previous.txt")) && IsFile(Join(StateDir(fs), L"rr-config\\Engine.ini")) &&
                        !IsFile(Join(StateDir(fs), L"uninstall.exe")) && !IsDir(Join(StateDir(fs), L"backup")),
                    "L4: the settings stay, the rest goes");
            SetEnvironmentVariableW(L"LOCALAPPDATA", oldLad.c_str());
        }

        // ---- log tail -----------------------------------------------------------------------------------------------------
        const string log =
            "[10:00:00.000 100] [amd_fidelityfx_dx12] ==== session start: Old.exe (process 100) ====\r\n"
            "[10:00:00.100 100] [amd_fidelityfx_dx12] settings: upscaler=amd runtime_dll=x\r\n"
            "[11:00:00.000 200] [amd_fidelityfx_dx12] ==== session start: Game.exe (process 200) ====\r\n"
            "[11:00:00.100 200] [amd_fidelityfx_dx12] settings: upscaler=network runtime_dll=wronggpu_runtime.dll fallback_to_amd=1\r\n"
            "[11:00:01.000 200] [info] executor dx12: the network runs on the game's D3D12 device\r\n"
            "[11:00:01.200 200] [info] network runtime ready: executor dx12, sync inline, max render 1280x720\r\n"
            "[11:00:02.000 200] [amd_fidelityfx_dx12] #4 OptiScaler plugin: WrongGPU selected in the host's upscaler list\r\n"
            "[11:00:03.000 200] [d3d12] own table: list 0x1 records into its own vtable 0x2 (x), which WrongGPU does not take over: reason\r\n"
            "[11:00:04.000 200] [amd_fidelityfx_dx12] [support] WG-HOOK-TABLE incident=200-1\r\n";
        LastRun lr = ParseLastRun(log);
        require(lr.found && lr.process == L"Game.exe" && lr.upscaler == L"network" && lr.ready && lr.executor == L"dx12" && lr.optiPlugin && lr.support == L"WG-HOOK-TABLE",
                "log tail: the last session's upscaler, executor, plugin and support code");
        require(lr.notes.size() == 1 && lr.notes[0].find(L"foreign table") != wstring::npos, "log tail: foreign command-list tables noted");
        require(lr.Summary().find(L"Model M (dx12)") == 0, "summary");
        LastRun none = ParseLastRun("");
        require(!none.found && none.Summary() == L"Not run yet", "no log: not run yet");
        LastRun amd = ParseLastRun("[1] [x] ==== session start: G.exe (process 1) ====\r\n[1] [x] settings: upscaler=amd\r\n");
        require(amd.found && !amd.ready && amd.Summary().find(L"AMD FSR") == 0, "WrongGPU off: AMD FSR");
        LastRun failed = ParseLastRun("[1] [x] ==== session start: G.exe (process 1) ====\r\n[1] [x] settings: upscaler=network\r\n[2] [x] ERROR: create failed -> AMD's FSR takes over\r\n");
        require(failed.found && !failed.ready && failed.notes.size() == 1 && failed.Summary().find(L"did not start") != wstring::npos, "a failed start");
        LastRun sized = ParseLastRun("[1] [x] ==== session start: G.exe (process 1) ====\r\n[1] [x] settings: upscaler=network\r\n"
                                     "[2] [fe] [U1] WrongGPU frame: frame=0 R=1707x960 at (0,0) O=2560x1440 jitter=(0,0) motion=raw*(1,1)+(0,0)\r\n"
                                     "[3] [fe] [U1] WrongGPU frame: frame=9 R=1280x720 at (0,0) O=2560x1440 jitter=(0,0)\r\n");
        require(sized.renderSize == L"1280x720" && sized.outputSize == L"2560x1440" && sized.notes.empty(), "log tail: the last frame's render and output sizes (the window's stat tiles)");
        LastRun oldFrame = ParseLastRun("[1] [x] ==== session start: G.exe (process 1) ====\r\n[3] [fe] [up] WrongGPU frame: 1280x720 notes: none\r\n");
        require(oldFrame.renderSize.empty() && oldFrame.outputSize.empty(), "log tail: a frame line without R= / O= gives no sizes");

        // ---- adoption: developer / test keys never become settings (2026-10-17) ---------------------------------------------
        {
            const string dev = "[fsr]\r\nupscaler = network\r\nruntime_options=model=M;timing_log=1;output_filter=quality5;capture_dir=C:\\cap;executor=d3d12-test\r\n"
                               "log_dispatches = 1\r\nsplit_test = 0\r\nview = upscale\r\n";
            const auto ign = IgnoredImportKeys(dev);
            require(ign == std::vector<string>{"log_dispatches=1", "timing_log=1", "capture_dir=C:\\cap"}, "ignored keys: diagnostics lines off their default, non-schema runtime items (model= excepted)");
            std::vector<wstring> pn;
            const Profile ip = ImportProfile(dev, "", &pn);
            size_t ignoredNotes = 0;
            bool notImported = false;
            for (auto& note : pn) { ignoredNotes += note.rfind(kIgnoredPrefix, 0) == 0; notImported |= note.rfind(L"not imported:", 0) == 0 && note.find(L"d3d12-test") != wstring::npos; }
            require(ip.get("output_filter") == "quality5" && ip.get("executor") == "auto" && !ip.v.count("timing_log") && !ip.v.count("capture_dir") && ignoredNotes == 3 && notImported,
                    "import: the schema's keys only; every other key listed (ignored / not imported)");
            require(ProfileText(ip).find("timing_log") == string::npos && ProfileText(ip).find("capture_dir") == string::npos, "profile.ini gets no developer key");
        }

        // ---- job and result files -----------------------------------------------------------------------------------------
        OpRequest job; job.op = Op::Update; job.exe = L"C:\\Games\\X\\x.exe"; job.dir = L"C:\\Games\\X"; job.optiPackage = L"C:\\dl\\Opti.7z"; job.moveMods = true;
        job.hasProfile = true; job.profile = DefaultProfile(); SetSetting(job.profile, "executor", "hip", nullptr);
        const string jt = JobText(job, L"C:\\Users\\p\\AppData\\Local", L"C:\\log.txt");
        OpRequest back; wstring lad, lf;
        require(ParseJob(jt, back, lad, lf, &why) && back.op == Op::Update && back.exe == job.exe && back.dir == job.dir && back.moveMods && !back.removeOld &&
                back.hasProfile && back.profile.get("executor") == "hip" && lad == L"C:\\Users\\p\\AppData\\Local" && lf == L"C:\\log.txt", "job file round trip");
        {   // the player's account and whether a new install gets its own entry: carried to the worker (another account may run it)
            wstring sid; int per = 7;
            require(ParseJob(JobText(job, L"C:\\Users\\p\\AppData\\Local", L"", L"S-1-5-21-1-2-3-1001", 0), back, lad, lf, &why, &sid, &per) && sid == L"S-1-5-21-1-2-3-1001" && per == 0 &&
                        ParseJob(jt, back, lad, lf, &why, &sid, &per) && sid.empty() && per == -1, "job file: usersid and perentry round trip (absent: empty, -1)");
            OpRequest d; d.op = Op::Detach; d.dir = L"C:\\Games\\X";
            require(ParseJob(JobText(d, L"C:\\x", L""), back, lad, lf, &why) && back.op == Op::Detach, "job file: the detach operation (the launcher's uninstall, games kept)");
            OpRequest o; o.op = Op::Apply; o.dir = job.dir; o.optiKeep = o.optiOnly = true;
            require(ParseJob(JobText(o, lad, L""), back, lad, lf, &why) && back.optiKeep && back.optiOnly && !back.hasProfile &&
                        ParseJob(jt, back, lad, lf, &why) && !back.optiOnly, "job file: an OptiScaler-only apply round trip (absent: the profile is applied too)");
        }
        require(!ParseJob("op=install\r\nexe=relative\\x.exe\r\nlocalappdata=C:\\x\r\n", back, lad, lf, &why), "a job with a relative path is refused");
        require(!ParseJob("op=format\r\nlocalappdata=C:\\x\r\n", back, lad, lf, &why) && !ParseJob("exe=C:\\x.exe\r\n", back, lad, lf, &why), "unknown or missing operation refused");
        require(!ParseJob("op=apply\r\ndir=C:\\g\r\nlocalappdata=C:\\x\r\nprofile.executor=warp\r\n", back, lad, lf, &why), "a job with a bad setting is refused");
        OpResult res; res.code = 6; res.message = L"switch?"; res.warnings = {L"w1"}; res.notes = {L"n1", L"n2"};
        OpResult rb;
        require(ParseResult(ResultText(res), rb) && rb.code == 6 && rb.message == L"switch?" && rb.warnings.size() == 1 && rb.notes.size() == 2, "result file round trip");

        // ---- update reply -----------------------------------------------------------------------------------------------
        wstring ver, page;
        require(ParseUpdateReply("{\"tag_name\": \"v0.6.0\", \"html_url\": \"https:\\/\\/example.invalid\\/r\\/0.6.0\"}", ver, page) && ver == L"0.6.0" && page == L"https://example.invalid/r/0.6.0",
                "update reply: a release object");
        require(ParseUpdateReply("version=0.5.6\npage=https://example.invalid/x\n", ver, page) && ver == L"0.5.6", "update reply: text lines");
        require(!ParseUpdateReply("<html>", ver, page) && ParseUpdateReply("version=1.0\npage=http://plain\n", ver, page) && page.empty(), "update reply: garbage refused, non-https page dropped");
        GlobalSettings gs;
        require(!CheckForUpdate(gs, L"0.5.5").checked, "update check off by default: no request");
        gs.updateCheck = true;
        const UpdateInfo noUrl = CheckForUpdate(gs, L"0.5.5");
        require(!noUrl.checked && noUrl.message.find(L"no release address") != wstring::npos, "update check on without a URL: no request");
        require(!SetGlobal(gs, "update_url", "http://example.invalid/x", &why) && SetGlobal(gs, "update_url", "https://example.invalid/x", &why), "update URL must be https");

        // ---- game environment ---------------------------------------------------------------------------------------------
        SetEnvironmentVariableW(L"HIP_PATH", L"C:\\sdk");
        SetEnvironmentVariableW(L"ROCM_PATH", L"C:\\sdk");
        SetEnvironmentVariableW(L"WG_KEEP_ME", L"1");
        const auto block = GameEnvironment({L"AMD_LOG_LEVEL=4"});
        wstring all;
        for (wchar_t c : block) all += c ? c : L'|';
        require(all.find(L"|HIP_PATH=") == wstring::npos && all.find(L"|ROCM_PATH=") == wstring::npos && all.rfind(L"HIP_PATH=", 0) != 0, "Play: the HIP SDK variables are not passed to the game");
        require(all.find(L"WG_KEEP_ME=1") != wstring::npos && all.find(L"AMD_LOG_LEVEL=4") != wstring::npos, "Play: other variables and the extra ones are passed");

        // ---- Play: through Steam for a game with a Steam app id, else directly (the starts stubbed: no Steam, no game) ----------
        {
            const wstring lib = Join(dir, L"steamlib\\steamapps");
            auto file = [](const wstring& p, const string& text) { EnsureDir(Parent(p)); WriteFileAtomic(p, text.data(), text.size()); };
            file(Join(lib, L"appmanifest_1234560.acf"), "\"AppState\"\n{\n\t\"appid\"\t\t\"1234560\"\n\t\"name\"\t\t\"Quarry Nine\"\n\t\"installdir\"\t\t\"Quarry Nine\"\n}\n");
            file(Join(lib, L"appmanifest_777.acf"), "\"AppState\"\n{\n\t\"name\"\t\t\"Odd Layout\"\n\t\"installdir\"\t\t\"Odd Layout\"\n}\n");  // no appid key
            const wstring unreal = Join(lib, L"common\\Quarry Nine\\QuarryNine\\Binaries\\Win64\\WgPlayTestQuarry-Win64-Shipping.exe");
            const wstring odd = Join(lib, L"common\\Odd Layout\\engine\\bin\\retail\\WgPlayTestOdd.exe");     // not a known store layout
            const wstring unlisted = Join(lib, L"common\\Unlisted\\WgPlayTestUnlisted.exe");                  // no appmanifest names it
            const wstring other = Join(dir, L"Games\\Other Game\\WgPlayTestOther.exe");                         // no Steam library
            for (auto& e : {unreal, odd, unlisted, other}) file(e, "MZ");
            require(SteamAppId(unreal) == L"1234560", "Play: the app id of the appmanifest whose installdir holds the game (Unreal layout)");
            require(SteamAppId(odd) == L"777", "Play: any depth below steamapps\\common\\<installdir>; the id from the file name without an appid key");
            require(SteamAppId(unlisted).empty() && SteamAppId(other).empty() && SteamAppId(Join(lib, L"common\\WgLoose.exe")).empty(), "Play: no app id without a matching appmanifest or outside a Steam library");

            PlayWay w = ChoosePlayWay(unreal, true);
            require(w.steam && w.appid == L"1234560" && w.note.empty(), "Play: a Steam game with an app id starts through Steam");
            w = ChoosePlayWay(unreal, false);
            require(!w.steam && w.appid == L"1234560" && !w.note.empty(), "Play: directly when Windows has no steam: address handler");
            w = ChoosePlayWay(other, true);
            require(!w.steam && w.appid.empty() && w.note.empty(), "Play: a game outside Steam starts directly");
            w = ChoosePlayWay(unlisted, true);
            require(!w.steam && w.appid.empty(), "Play: a Steam folder without its app id starts directly");

            std::vector<wstring> urls, directs;
            bool urlOk = true, steamKnown = true;
            PlayStarters stub;
            stub.steamKnown = [&] { return steamKnown; };
            stub.openUrl = [&](const wstring& u, wstring* why) { urls.push_back(u); if (!urlOk && why) *why = L"no Steam"; return urlOk; };
            stub.direct = [&](const wstring& e, wstring* why) { directs.push_back(e); if (why) why->clear(); return true; };
            Context pc;
            pc.logFile = Join(dir, L"play.log");
            auto logText = [&] { string t; ReadAll(pc.logFile, t); return t; };
            PlayWay how;
            require(PlayGame(unreal, &why, &how, &pc, stub) && how.steam && how.appid == L"1234560", "Play: started through Steam");
            require(urls.size() == 1 && urls[0] == L"steam://rungameid/1234560" && directs.empty(), "Play: steam://rungameid/<appid>, no direct start");
            require(logText().find("started through Steam (app id 1234560") != string::npos, "Play: the log says it started through Steam and the app id");
            require(PlayGame(other, &why, &how, &pc, stub) && !how.steam && how.appid.empty(), "Play: a non-Steam game started directly");
            require(urls.size() == 1 && directs.size() == 1 && directs[0] == other, "Play: the direct start for a game without an app id, no steam: address");
            require(logText().find("WgPlayTestOther.exe started directly") != string::npos, "Play: the log says it started directly");
            require(PlayGame(unlisted, &why, &how, &pc, stub) && !how.steam && urls.size() == 1 && directs.size() == 2, "Play: the direct start when no app id is found");
            steamKnown = false;
            require(PlayGame(unreal, &why, &how, &pc, stub) && !how.steam && how.appid == L"1234560" && urls.size() == 1 && directs.size() == 3, "Play: the direct start without a steam: handler");
            steamKnown = true; urlOk = false;
            require(PlayGame(unreal, &why, &how, &pc, stub) && !how.steam && urls.size() == 2 && directs.size() == 4 && how.note.find(L"no Steam") != wstring::npos,
                    "Play: the direct start when starting through Steam fails, and why");
            require(logText().find("started directly (Steam app id 1234560: starting it through Steam failed") != string::npos, "Play: the log says why a Steam game started directly");
            urlOk = true;
            require(!PlayGame(Join(dir, L"Games\\Missing\\WgPlayTestMissing.exe"), &why, &how, &pc, stub) && why.find(L"not found") != wstring::npos && urls.size() == 2 && directs.size() == 4,
                    "Play: a missing exe is refused before either start");

            // rc6 L7: an Epic game starts through Epic (com.epicgames.launcher://apps/<AppName>), else directly; the starts stubbed
            const wstring oldPd = EnvVar(L"ProgramData"), pd = Join(dir, L"programdata");
            const wstring epicGame = Join(dir, L"Epic Games\\Quarry Two\\Binaries\\Win64\\WgPlayTestEpic.exe");
            file(epicGame, "MZ");
            auto jsonPath = [](const wstring& p) { string e; for (char c : ToUtf8(p)) { if (c == '\\') e += '\\'; e += c; } return e; };  // JSON escapes '\'
            file(Join(pd, L"Epic\\EpicGamesLauncher\\Data\\Manifests\\ABC.item"),
                 "{\n\t\"DisplayName\": \"Quarry Two\",\n\t\"InstallLocation\": \"" + jsonPath(Join(dir, L"Epic Games\\Quarry Two")) + "\",\n\t\"AppName\": \"QuarryTwo_App-1.0\"\n}\n");
            file(Join(pd, L"Epic\\EpicGamesLauncher\\Data\\Manifests\\BAD.item"),
                 "{\n\t\"InstallLocation\": \"" + jsonPath(Join(dir, L"Epic Games\\Odd")) + "\",\n\t\"AppName\": \"a b&c\"\n}\n");
            file(Join(dir, L"Epic Games\\Odd\\WgOdd.exe"), "MZ");
            SetEnvironmentVariableW(L"ProgramData", pd.c_str());
            require(EpicAppName(epicGame) == L"QuarryTwo_App-1.0" && EpicAppName(other).empty() && EpicAppName(Join(dir, L"Epic Games\\Quarry Twofold\\x.exe")).empty(),
                    "Play: the Epic AppName of the manifest whose InstallLocation holds the game (none elsewhere)");
            require(EpicAppName(Join(dir, L"Epic Games\\Odd\\WgOdd.exe")).empty(), "Play: an AppName that can't go into an address is none");
            w = ChoosePlayWay(epicGame, true, true, L"QuarryTwo_App-1.0");
            require(w.epic && !w.steam && w.epicApp == L"QuarryTwo_App-1.0" && w.note.empty(), "Play: an Epic game starts through Epic");
            w = ChoosePlayWay(epicGame, true, false, L"QuarryTwo_App-1.0");
            require(!w.epic && !w.note.empty(), "Play: directly when Windows has no com.epicgames.launcher: handler");
            w = ChoosePlayWay(unreal, true, true, L"QuarryTwo_App-1.0");
            require(w.steam && !w.epic, "Play: a Steam app id goes first");
            bool epicKnown = true;
            stub.epicKnown = [&] { return epicKnown; };
            stub.epicApp = [](const wstring& e) { return EpicAppName(e); };
            urls.clear(); directs.clear();
            require(PlayGame(epicGame, &why, &how, &pc, stub) && how.epic && !how.steam && how.epicApp == L"QuarryTwo_App-1.0" && directs.empty() && urls.size() == 1 &&
                        urls[0] == L"com.epicgames.launcher://apps/QuarryTwo_App-1.0?action=launch&silent=true",
                    "Play: com.epicgames.launcher://apps/<AppName>?action=launch&silent=true, no direct start");
            require(logText().find("started through Epic (app QuarryTwo_App-1.0") != string::npos, "Play: the log says it started through Epic and the app");
            urlOk = false;
            require(PlayGame(epicGame, &why, &how, &pc, stub) && !how.epic && how.epicApp == L"QuarryTwo_App-1.0" && urls.size() == 2 && directs.size() == 1 &&
                        how.note.find(L"through Epic failed") != wstring::npos,
                    "Play: the direct start when starting through Epic fails, and why");
            require(logText().find("started directly (Epic app QuarryTwo_App-1.0: starting it through Epic failed") != string::npos, "Play: the log says why an Epic game started directly");
            urlOk = true; epicKnown = false;
            require(PlayGame(epicGame, &why, &how, &pc, stub) && !how.epic && urls.size() == 2 && directs.size() == 2, "Play: the direct start without an Epic handler");
            epicKnown = true;
            require(PlayGame(other, &why, &how, &pc, stub) && !how.epic && how.epicApp.empty() && urls.size() == 2 && directs.size() == 3, "Play: a game outside Epic and Steam starts directly");
            SetEnvironmentVariableW(L"ProgramData", oldPd.c_str());
        }

        // ---- global settings round trip --------------------------------------------------------------------------------------
        LauncherPaths lp; lp.dataDir = Join(dir, L"data");
        GlobalSettings g1;
        g1.optiPackage = L"C:\\dl\\Opti.7z"; g1.extraFolders = {L"K:\\Games"}; g1.playButton = false; g1.updateCheck = true; g1.updateUrl = L"https://example.invalid/u";
        g1.appearance = L"light"; g1.accent = L"graphite"; g1.exactMode = true;
        SetSetting(g1.defaults, "output_filter", "analytic", nullptr);
        SaveGlobal(lp, g1);
        const GlobalSettings g2 = LoadGlobal(lp);
        require(g2.optiPackage == g1.optiPackage && g2.extraFolders == g1.extraFolders && !g2.playButton && g2.updateCheck && g2.updateUrl == g1.updateUrl &&
                g2.defaults.get("output_filter") == "analytic" && g2.appearance == L"light" && g2.accent == L"graphite" && g2.exactMode, "launcher.ini round trip");
        GlobalSettings fresh = LoadGlobal(LauncherPaths{L"", Join(dir, L"none")});
        require(!fresh.updateCheck && fresh.updateUrl.empty() && fresh.playButton && fresh.appearance == L"dark" && fresh.accent == L"green" && !fresh.exactMode,
                "defaults: update check off, no URL, Play button shown, the dark theme, the green accent, Exact mode off");
        {
            GlobalSettings x;
            require(SetGlobal(x, "exact_mode", "1", nullptr) && x.exactMode && SetGlobal(x, "exact_mode", "0", nullptr) && !x.exactMode && !SetGlobal(x, "exact_mode", "maybe", nullptr),
                    "exact_mode: 0 or 1");
        }
        {
            GlobalSettings a;
            require(SetGlobal(a, "appearance", "Light", nullptr) && a.appearance == L"light" && !SetGlobal(a, "appearance", "blue", nullptr) && a.appearance == L"light",
                    "appearance: system, light or dark only");
            require(SetGlobal(a, "accent", "Blue", nullptr) && a.accent == L"blue" && SetGlobal(a, "accent", "windows", nullptr) && !SetGlobal(a, "accent", "red", nullptr) &&
                        !SetGlobal(a, "accent", "orange", nullptr) && a.accent == L"windows",
                    "accent: green, blue, purple, pink, cyan, graphite or windows only (red and amber keep their meaning)");
        }

        // ---- the accent colours: every accent, both themes (BE-wide notes.md "Accent color"): white words on the filled
        // buttons 4.5:1, the accent's words 4.5:1 on the page, a cell, the fill and its tint, its graphics (switch track, the
        // chosen radio and swatch ring) 3:1 on the page and a cell; the working green the same; and the Windows accent made
        // into tokens by the same rules from colours that are hard to read (yellow, near white, near black, Windows' red)
        {
            using namespace wg::app::lui;
            const Col white = Hex(0xffffff);
            auto check = [&](const AccentTokens& k, bool dark, const char* what) {
                const Surfaces s = SurfacesOf(dark);
                const bool ok = Contrast(white, k.f) >= 4.5f && Contrast(k.t, s.bg) >= 4.5f && Contrast(k.t, s.cell) >= 4.5f && Contrast(k.t, s.fill) >= 4.5f &&
                                Contrast(k.t, k.tint) >= 4.5f && Contrast(k.t, k.tintHover) >= 4.5f && Contrast(k.c, s.bg) >= 3.f && Contrast(k.c, s.cell) >= 3.f;
                if (!ok) std::printf("  contrast: %s (%s): white/fill %.2f text/page %.2f text/cell %.2f text/fill %.2f text/tint %.2f graphics/page %.2f graphics/cell %.2f\n",
                                     what, dark ? "dark" : "light", Contrast(white, k.f), Contrast(k.t, s.bg), Contrast(k.t, s.cell), Contrast(k.t, s.fill),
                                     Contrast(k.t, k.tintHover), Contrast(k.c, s.bg), Contrast(k.c, s.cell));
                return ok;
            };
            bool contrastOk = true;
            for (int dark = 0; dark < 2; ++dark) {
                for (int a = 0; a < A_COUNT; ++a) contrastOk = check(AccentFor(a, dark != 0, 0x0078d4), dark != 0, "accent") && contrastOk;
                for (unsigned w : {0x0078d4u, 0xffb900u, 0xfff4cdu, 0x101010u,  // Windows blue, yellow, near white, near black
                                   0xe81123u, 0x00cc6au, 0xffffffu, 0x000000u}) // Windows red, green, white, black
                    contrastOk = check(AccentFor(A_WINDOWS, dark != 0, w), dark != 0, "windows accent") && contrastOk;
                const Surfaces s = SurfacesOf(dark != 0);
                contrastOk = contrastOk && Contrast(StatusGreen(dark != 0), s.bg) >= 3.f && Contrast(StatusGreen(dark != 0), s.cell) >= 3.f &&
                      Contrast(StatusGreenText(dark != 0), s.bg) >= 4.5f && Contrast(StatusGreenText(dark != 0), s.cell) >= 4.5f;
            }
            require(contrastOk, "accent colours: 4.5:1 for words and filled buttons, 3:1 for graphics, every accent and theme");
            // a filled button that is off, red or any accent, either theme: one rule (DisabledFilled): its label 3:1 or better
            // and under 4:1 on its faded fill, and weaker than the live label on the live fill (never reads as live)
            bool offOk = true;
            for (int dark = 0; dark < 2; ++dark) {
                const Surfaces s = SurfacesOf(dark != 0);
                std::vector<Col> fills = {Hex(dark ? 0xc9362a : 0xc42b1c)};  // the red (needs you)
                for (int a = 0; a < A_COUNT; ++a) fills.push_back(AccentFor(a, dark != 0, 0x0078d4).f);
                for (unsigned w : {0xffb900u, 0x101010u, 0xe81123u}) fills.push_back(AccentFor(A_WINDOWS, dark != 0, w).f);
                for (const Col& f : fills) {
                    const FilledOff o = DisabledFilled(f, s.cell, white, dark != 0);
                    const float c = Contrast(o.label, o.fill);
                    if (c < 3.f || c >= 4.f || c >= Contrast(white, f)) {
                        offOk = false;
                        std::printf("  disabled filled (%s): label/fill %.2f, live %.2f\n", dark ? "dark" : "light", c, Contrast(white, f));
                    }
                }
            }
            require(offOk, "disabled filled buttons: one rule for red and every accent, label 3:1 to 4:1 on its fill, weaker than live");
            require(AccentFromKey(L"graphite") == A_GRAPHITE && AccentFromKey(L"nonsense") == A_GREEN && std::wstring(AccentKey(A_CYAN)) == L"cyan",
                    "accent keys round trip; an unknown one is the default");
        }

        // ---- Ray Reconstruction (rr-notes.md) -------------------------------------------------------------------------
        {
            // the profile's two RR settings: validated, normalised, and never written into the upscaler's wronggpu.ini
            const SettingDef* rrOn = FindSetting("rr_enabled");
            const SettingDef* rrMode = FindSetting("rr_mode");
            require(rrOn && rrMode && rrOn->layer == Layer::Rr && rrMode->layer == Layer::Rr, "rr_enabled and rr_mode are profile settings of their own layer");
            require(ValidateSetting(*rrOn, "1", &n, &why) && n == "on" && ValidateSetting(*rrOn, "false", &n, &why) && n == "off", "rr_enabled 1 = on, false = off");
            require(ValidateSetting(*rrMode, "Fast", &n, &why) && n == "fast" && !ValidateSetting(*rrMode, "turbo", &n, &why), "rr_mode: exact, accurate or fast only");
            Profile rp = DefaultProfile();
            require(rp.get("rr_enabled") == "on" && rp.get("rr_mode") == "fast", "RR defaults: on, Fast (owner 2026-11-05)");
            SetSetting(rp, "rr_mode", "fast", nullptr);
            SetSetting(rp, "rr_enabled", "off", nullptr);
            require(RenderWronggpuIni(rp, front, out, &why) && out == front, "the RR settings leave the upscaler's wronggpu.ini alone");
            require(OutsideEdits(rp, front, "").empty(), "RR settings are never outside edits of the upscaler's file");

            // WrongGPU\wronggpu.ini: rendered from the profile, read back, the plugin's own keys and comments kept
            require(ParseRrSettings(RenderRrSettings(DefaultProfile(), true, "")).mode == "fast", "a new RR settings file from the default profile: rr_mode=fast written (rc5)");
            Profile acc = DefaultProfile();
            SetSetting(acc, "rr_mode", "accurate", nullptr);  // a player's Accurate (rc5: no longer the default)
            const string rrNew = RenderRrSettings(acc, true, "");
            RrFileState st = ParseRrSettings(rrNew);
            require(st.present && st.enabled && st.mode == "accurate", "a new RR settings file: on, Accurate");
            require(rrNew.find("[wronggpu]\r\n") != string::npos && rrNew.find("rr_enabled=1\r\n") != string::npos && rrNew.find("rr_exact_allowed=0\r\n") != string::npos,
                    "rr_enabled=1, rr_exact_allowed=0 in [wronggpu], CRLF");
            require(RenderRrSettings(acc, true, rrNew) == rrNew, "rendering twice gives the same file (round trip)");
            require(ParseRrSettings(RenderRrSettings(acc, false, rrNew)).enabled == false, "no RR data: rr_enabled=0 whatever the profile says");
            Profile ex = DefaultProfile();
            SetSetting(ex, "rr_mode", "exact", nullptr);
            const string exText = RenderRrSettings(ex, true, rrNew);
            require(ParseRrSettings(exText).mode == "exact" && exText.find("rr_exact_allowed=1\r\n") != string::npos, "exact: rr_mode=exact with rr_exact_allowed=1");
            require(RenderRrSettings(acc, true, exText).find("rr_exact_allowed=0\r\n") != string::npos, "back from exact: rr_exact_allowed=0 again");
            const string plugin = "# the plugin's\r\n[wronggpu]\r\nrr_mode = fast\r\nrr_hud = 1\r\n[other]\r\nrr_mode = exact\r\n";
            st = ParseRrSettings(plugin);
            require(st.mode == "fast" && st.enabled, "only [wronggpu] is read; a missing rr_enabled is on");
            const string kept = RenderRrSettings(acc, true, plugin);
            require(kept.find("# the plugin's\r\n") != string::npos && kept.find("rr_hud = 1") != string::npos && kept.find("[other]\r\nrr_mode = exact") != string::npos &&
                        ParseRrSettings(kept).mode == "accurate",
                    "the plugin's comment, keys and other sections stay; rr_mode follows the profile");
            require(RrIniText(L"C:\\Users\\P\\AppData\\Local\\WrongGPU\\model\\F\\model.wgcm") ==
                        "# WrongGPU Ray Reconstruction (written by the WrongGPU launcher; the RR plugin adds its own keys)\r\nrr_model=C:\\Users\\P\\AppData\\Local\\WrongGPU\\model\\F\\model.wgcm\r\n",
                    "wronggpu-rr.ini names the RR data");

            // the mode switched in the game (OptiScaler's menu) becomes the profile's
            const wstring rg = Join(dir, L"rrgame");
            EnsureDir(Join(rg, kRrDataDir));
            Profile fold = DefaultProfile();
            require(!FoldRrFromGame(rg, fold), "no RR settings file: nothing to fold");
            // (rc5: the file's mode differs from the default, Fast)
            { const string t = "[wronggpu]\r\nrr_enabled=1\r\nrr_mode=accurate\r\n"; WriteFileAtomic(Join(rg, kRrSettings), t.data(), t.size()); }
            require(FoldRrFromGame(rg, fold) && fold.get("rr_mode") == "accurate" && !FoldRrFromGame(rg, fold), "the game's rr_mode folds into the profile once");
            require(ReadRrSettings(rg).present && ReadRrSettings(Join(dir, L"nothing")).present == false, "ReadRrSettings: present only with the file");

            // which games: the RR files next to the exe, deeper in the game's folder, a game known to call it, none
            auto file = [](const wstring& p) { EnsureDir(Parent(p)); WriteFileAtomic(p, "x", 1); };
            const wstring a = Join(dir, L"rr-a"), b = Join(dir, L"rr-b"), c = Join(dir, L"rr-c"), d = Join(dir, L"rr-d");
            file(Join(a, L"bin\\x64\\Game.exe")); file(Join(a, L"bin\\x64\\nvngx_dlssd.dll"));  // DLSS Ray Reconstruction file
            file(Join(b, L"Project\\Binaries\\Win64\\Project-Win64-Shipping.exe")); file(Join(b, L"Engine\\Plugins\\RR\\Binaries\\sl.dlss_d.dll"));  // DLSS
            file(Join(c, L"HogwartsLegacy.exe"));
            file(Join(d, L"Game.exe")); file(Join(d, L"nvngx_dlss.dll"));  // DLSS (the upscaler's file alone: no RR)
            RrUse ru = DetectRrUncached(Join(a, L"bin\\x64\\Game.exe"));
            require(ru.uses && IEq(ru.why, L"nvngx_dlssd.dll"), "the RR model file next to the exe: uses RR");  // DLSS
            ru = DetectRrUncached(Join(b, L"Project\\Binaries\\Win64\\Project-Win64-Shipping.exe"));
            require(ru.uses && ru.why.find(L"sl.dlss_d.dll") != wstring::npos, "the RR plugin deeper in the game's folder: uses RR");  // DLSS
            require(DetectRrUncached(Join(c, L"HogwartsLegacy.exe")).uses, "a game known to call RR directly");
            require(!DetectRrUncached(Join(d, L"Game.exe")).uses && !DetectRrUncached(Join(d, L"Missing.exe")).uses, "a game without RR (and a missing exe): no");
            require(DetectRr(Join(a, L"bin\\x64\\Game.exe")).uses && !DetectRr(Join(d, L"Game.exe")).uses, "the cached detection agrees");
            // direct-RR games (2026-11-04): a game that creates RR itself (the RR library without the game interface's RR plugin) is
            // "direct"; the game interface's RR plugin anywhere in the folder makes it not direct; the known games' table decides
            // for its games (Hogwarts Legacy direct, RE Requiem through the game interface)
            require(DetectRrUncached(Join(a, L"bin\\x64\\Game.exe")).direct, "the RR library alone next to the exe: direct");
            require(!DetectRrUncached(Join(b, L"Project\\Binaries\\Win64\\Project-Win64-Shipping.exe")).direct, "the game interface's RR plugin: not direct");
            {
                const wstring u = Join(dir, L"rr-unreal"), cp = Join(dir, L"rr-both"), r9 = Join(dir, L"rr-re9"), late = Join(dir, L"rr-late");
                file(Join(u, L"Lego\\Binaries\\Win64\\Lego-Win64-Shipping.exe"));
                file(Join(u, L"Engine\\Plugins\\Runtime\\Vendor\\RR\\Binaries\\ThirdParty\\Win64\\nvngx_dlssd.dll"));  // DLSS RR (the Unreal plugin layout)
                RrUse ur = DetectRrUncached(Join(u, L"Lego\\Binaries\\Win64\\Lego-Win64-Shipping.exe"));
                require(ur.uses && ur.direct && ur.why.find(L"nvngx_dlssd.dll") != wstring::npos, "the Unreal plugin layout (RR library deep in Engine): direct, the library named");
                file(Join(cp, L"bin\\x64\\Game.exe")); file(Join(cp, L"bin\\x64\\nvngx_dlssd.dll")); file(Join(cp, L"bin\\x64\\sl.dlss_d.dll"));  // DLSS
                ur = DetectRrUncached(Join(cp, L"bin\\x64\\Game.exe"));
                require(ur.uses && !ur.direct && IEq(ur.why, L"sl.dlss_d.dll"), "the RR library and the game interface's RR plugin next to the exe: not direct");  // DLSS
                file(Join(late, L"bin\\x64\\Game.exe")); file(Join(late, L"bin\\x64\\nvngx_dlssd.dll")); file(Join(late, L"zz\\plugins\\sl.dlss_d.dll"));  // DLSS
                require(!DetectRrUncached(Join(late, L"bin\\x64\\Game.exe")).direct, "the library next to the exe, the game interface's RR plugin elsewhere in the folder: not direct");
                require(DetectRrUncached(Join(c, L"HogwartsLegacy.exe")).direct, "Hogwarts Legacy (known game): direct");
                file(Join(r9, L"re9.exe")); file(Join(r9, L"nvngx_dlssd.dll"));  // DLSS: the library alone, but the table says RE Requiem is not direct
                require(DetectRrUncached(Join(r9, L"re9.exe")).uses && !DetectRrUncached(Join(r9, L"re9.exe")).direct, "RE Requiem (known game): never direct");
                const wstring cpx = Join(r9, L"cp\\Cyberpunk2077.exe"), cqx = Join(r9, L"cq\\Cyberpunk2077.exe");
                file(cpx); file(Join(r9, L"cp\\nvngx_dlssd.dll"));  // DLSS
                file(cqx);
                require(DetectRrUncached(cpx).uses && !DetectRrUncached(cpx).direct && !DetectRrUncached(cqx).uses,
                        "Cyberpunk 2077: never direct; RR only by its files (its row adds no use)");
            }
            {   // the install's OptiScaler from its record: the proxy's copy entry has the edition DLL's hash
                const wstring ed = PinnedEdition().dllSha256;
                Record er; er.proxy = L"dxgi.dll";
                er.e.push_back({L"copy", L"dxgi.dll", false, L"-", ed, L"wg-optibundle"});
                require(!ed.empty() && RecordHasEdition(er), "a record whose OptiScaler DLL has the edition's hash: the edition");
                er.e.back().inst = L"0000000000000000000000000000000000000000000000000000000000000000";
                require(!RecordHasEdition(er), "another OptiScaler DLL: not the edition");
                Record nr; nr.e.push_back({L"copy", L"winmm.dll", false, L"-", ed, L"wg-optibundle"});
                require(RecordHasEdition(nr), "no proxy line: OptiScaler's DLL under a proxy name with the edition's hash");
                Record pr; pr.proxy = L"dxgi.dll"; pr.e.push_back({L"copy", L"plugins\\WrongGPU.asi", false, L"-", ed, L"wg"});
                require(!RecordHasEdition(pr), "the hash on another file than the proxy: not the edition");
                require(RrDirectNoEditionText(L"Some Game").find(L"no OptiScaler edition") != wstring::npos &&
                        RrDirectNoEditionText(L"Some Game").find(L"isn't offered in Some Game, and the game") != wstring::npos,
                        "the warning carries the words the window turns into the player's sentence (ui_words.cpp)");
            }
            ForgetRrDetection();

            // the last run's RR log: the session after the last "attached", the mode it ran, the GPU time a frame
            LastRun rl;
            ParseRrLog("[10:00:00] load: process 100 attached\r\n[10:00:01] runtime: create instance\r\nrr_mode=fast\r\n"
                       "[10:02:00] load: process 200 attached\r\n[10:02:01] rr_mode=accurate\r\n[10:02:02] runtime: create instance\r\n"
                       "[10:02:03] rr_mode accurate -> fast\r\n[10:03:00] timing, 600 frames at 2560x1440: GPU 6.84 ms per frame\r\n", rl);
            require(rl.rrFound && rl.rrUsed && rl.rrMode == L"fast" && rl.rrMs == L"6.8", "RR log: used, the mode switched in game, 6.8 ms a frame");
            LastRun lr2;
            ParseRrLog("load: process 1 attached\r\nruntime: create instance\r\nload: process 2 attached\r\nrr_mode=exact\r\nrr_exact_allowed=0: using accurate\r\n", lr2);
            require(lr2.rrFound && !lr2.rrUsed && lr2.rrMode == L"accurate" && lr2.rrMs.empty(), "RR log: the last session only; a refused Exact runs Accurate; no timing, no cost");
            LastRun lr3;
            ParseRrLog("nothing here\r\n", lr3);
            require(!lr3.rrFound && !lr3.rrUsed, "no session in the log: no RR line");
            LastRun lr4;  // a long session: its "attached" line is out of the 256 KB tail, its timing lines are not
            ParseRrLog("ames at 2560x1440: GPU 7.10 ms per frame\r\n[11:00:00] timing, 600 frames at 2560x1440: GPU 6.84 ms per frame\r\n", lr4);
            require(lr4.rrFound && lr4.rrUsed && lr4.rrMode.empty() && lr4.rrMs == L"6.8", "RR log: the attach line scrolled out of the tail: still used, its last cost");
            LastRun lr5;
            ParseRrLog("load: process 7 detached\r\n[11:00:00] timing, 600 frames at 2560x1440: GPU 6.84 ms per frame\r\n", lr5);
            require(!lr5.rrFound && !lr5.rrUsed, "RR log: another load line without an attach in the tail: not guessed");

            // which build of the RR plugin a game gets: by the version of the game interface it hooks (its major.minor), the
            // general plugin otherwise; never by the game's name
            require(RrPluginVersion(L"plugins\\2.8\\WrongGPU-RR.asi") == L"2.8" && RrPluginVersion(kRrPayloadPlugin).empty() &&
                    RrPluginVersion(L"plugins\\x\\WrongGPU-RR.asi").empty() && RrPluginVersion(L"plugins\\2.8.1\\WrongGPU-RR.asi").empty() &&
                    RrPluginVersion(L"other\\2.8\\WrongGPU-RR.asi").empty(), "the plugin's builds: plugins\\<major>.<minor>\\ only");
            require(IsRrPayloadPlugin(kRrPayloadPlugin) && IsRrPayloadPlugin(L"plugins\\2.14\\WrongGPU-RR.asi") && !IsRrPayloadPlugin(kRrRuntime) &&
                    !IsRrPayloadPlugin(kRrPayloadEdition), "IsRrPayloadPlugin");
            const std::vector<wstring> rels{kRrPayloadPlugin, L"plugins\\2.8\\WrongGPU-RR.asi", L"plugins\\2.7\\WrongGPU-RR.asi", kRrRuntime, kRrPayloadEdition};
            require(IEq(PickRrPlugin(rels, L"2.8.0.0"), L"plugins\\2.8\\WrongGPU-RR.asi") && IEq(PickRrPlugin(rels, L"2.7.1.0"), L"plugins\\2.7\\WrongGPU-RR.asi"),
                    "a game interface 2.8.0 / 2.7.1: the build for its version");
            require(IEq(PickRrPlugin(rels, L"2.14.1.0"), kRrPayloadPlugin) && IEq(PickRrPlugin(rels, L""), kRrPayloadPlugin) &&
                    IEq(PickRrPlugin(rels, L"2.80.0.0"), kRrPayloadPlugin), "no build for its version (or no interface file): the general plugin");
            require(PickRrPlugin({kRrRuntime}, L"2.8.0.0").empty(), "no plugin at all: none");

            // the per-game rules: RE Requiem needs REFramework, its RR needs the host's restore and comes with path tracing only
            const GameRule* re9 = GameRuleFor(L"K:\\Games\\RE\\re9.exe");
            require(re9 && re9->needsReFramework && re9->needsHostRestore && re9->rrOnlyPathTracing && GameRuleFor(L"RE9DEMO.EXE") &&
                    !GameRuleFor(L"Cyberpunk2077.exe"), "GameRuleFor: re9.exe and its demo, nothing for other games");
            require(NeedsReFrameworkText(*re9).find(L"needs REFramework installed first") != wstring::npos, "the refusal's sentence");
            // REFramework: dinput8.dll whose own words say REFramework (never the name alone), the folder as evidence
            const wstring rf = Join(dir, L"refw");
            file(Join(rf, L"dinput8.dll"));
            require(!DetectReFramework(rf).found, "another dinput8.dll: not REFramework");
            { const string t = "MZ synthetic loader stand-in: REFramework v1.5 (test)"; WriteFileAtomic(Join(rf, L"dinput8.dll"), t.data(), t.size()); }
            ReFrameworkState rs = DetectReFramework(rf);
            require(rs.found && rs.evidence == L"dinput8.dll", "REFramework's loader by its words");
            EnsureDir(Join(rf, L"reframework"));
            rs = DetectReFramework(rf);
            require(rs.found && rs.evidence.find(L"folder reframework") != wstring::npos, "its folder named as evidence");
            require(!DetectReFramework(Join(dir, L"nothing")).found, "no dinput8.dll: not found");
            // its mirror: the install folder's root DLLs only
            require(MirroredRel(L"dxgi.dll") && MirroredRel(L"wronggpu_rr_runtime.dll") && !MirroredRel(L"plugins\\WrongGPU-RR.asi") &&
                    !MirroredRel(L"OptiScaler.ini") && IsMirrorRel(MirrorRel(L"dxgi.dll")) && !IsMirrorRel(L"dxgi.dll") &&
                    MirrorRel(L"dxgi.dll") == L"_storage_\\dxgi.dll", "REFramework's mirror: root DLLs, _storage_\\<name>");
            // an install record's mirror entries are REFramework's to rewrite: never a problem of the install
            Record mr;
            Entry me; me.action = L"copy"; me.rel = MirrorRel(L"dxgi.dll"); me.inst = L"00"; mr.e.push_back(me);
            require(VerifyInstall(Join(dir, L"nothing"), mr).empty(), "VerifyInstall: a missing mirror copy is no problem");
        }
        {
            // an install an earlier build set up to start on Preset K (0.5.0 / 0.5.1): the plugin's choice first, then the
            // runtime options' last model item, then the record's model file
            const wstring pk = Join(dir, L"presetk");
            EnsureDir(pk);
            auto put = [&](const wchar_t* name, const string& text) { WriteFileAtomic(Join(pk, name), text.data(), text.size()); };
            Record noRec;
            require(!StartsOnPresetK(pk, &noRec), "Preset K: nothing there, not K");
            put(L"wronggpu.ini", "[fsr]\r\nupscaler = network\r\nruntime_options = output_filter=original;model=K\r\n");
            require(StartsOnPresetK(pk, &noRec), "Preset K: runtime_options model=K");
            put(L"wronggpu.ini", "[fsr]\r\nruntime_options = model=K;model=M\r\n");
            require(!StartsOnPresetK(pk, &noRec), "Preset K: the last model item wins (M)");
            put(L"wronggpu-asi.ini", "; WrongGPU OptiScaler plugin\r\nselection=wronggpu-k\r\n");
            require(StartsOnPresetK(pk, &noRec), "Preset K: the plugin's choice decides over wronggpu.ini");
            put(L"wronggpu-asi.ini", "selection=wronggpu\r\n");
            put(L"wronggpu.ini", "[fsr]\r\nruntime_options = model=K\r\n");
            require(!StartsOnPresetK(pk, &noRec), "Preset K: the plugin on WrongGPU (Model M) decides over model=K");
            DeleteFileW(Join(pk, L"wronggpu-asi.ini").c_str());
            put(L"wronggpu.ini", "[fsr]\r\nupscaler = network\r\n");
            Record rk; rk.model = Join(Join(ModelBaseDir(), L"K"), L"model.wgcm");
            Record rm; rm.model = Join(ModelBaseDir(), L"model.wgcm");
            require(StartsOnPresetK(pk, &rk) && !StartsOnPresetK(pk, &rm), "Preset K: the record's model file in the K cache");
        }

        // ---- the Engine.ini file check: a file it can't read is "couldn't read", never "no line" (r3 #8) -------------------
        {
            Context rc;
            rc.playerDocuments = Join(dir, L"docs");  // a test root, never this PC's Documents
            const RrConfigGame* bl4 = RrConfigGameFor(L"Borderlands4.exe");
            const wstring cfg = Join(Join(Join(Join(Join(rc.playerDocuments, L"My Games"), L"Borderlands 4"), L"Saved"), L"Config"), L"Windows");
            EnsureDir(cfg);
            { const string gus = "[ScalabilityGroups]\r\n"; WriteFileAtomic(Join(cfg, L"GameUserSettings.ini"), gus.data(), gus.size()); }
            const wstring ini = Join(cfg, L"Engine.ini");
            wstring f, rwhy;
            require(bl4 && RrConfigLineState(rc, *bl4, &f, &rwhy) == 0 && IEq(f, ini), "Engine.ini check: no file is no line (0)");
            const string utf16("\xFF\xFE[\0S\0", 6);
            WriteFileAtomic(ini, utf16.data(), utf16.size());
            rwhy.clear();
            require(RrConfigLineState(rc, *bl4, &f, &rwhy) == 3 && rwhy.find(L"UTF-16") != wstring::npos, "Engine.ini check: UTF-16 is couldn't read (3), why says so");
            const string line = RrConfigNewFile();
            WriteFileAtomic(ini, line.data(), line.size());
            HANDLE held = CreateFileW(ini.c_str(), GENERIC_READ, 0, nullptr, OPEN_EXISTING, 0, nullptr);  // another program has it, no sharing
            rwhy.clear();
            const int st = RrConfigLineState(rc, *bl4, &f, &rwhy);
            if (held != INVALID_HANDLE_VALUE) CloseHandle(held);
            require(held != INVALID_HANDLE_VALUE && st == 3 && rwhy.find(L"another program") != wstring::npos, "Engine.ini check: a file held open is couldn't read (3)");
            require(RrConfigLineState(rc, *bl4, &f, &rwhy) == 2, "Engine.ini check: readable again, the line there, not read-only (2)");
            // r4 #5: saved as UTF-8 with a byte order mark (Notepad's "UTF-8 with BOM"): the section right after it is found
            const string bom = "\xEF\xBB\xBF";
            const string withLine = bom + line;
            WriteFileAtomic(ini, withLine.data(), withLine.size());
            require(RrConfigLineState(rc, *bl4, &f, &rwhy) == 2, "Engine.ini check with a BOM: the line in its [SystemSettings] is there (2), not missing (r4 #5)");
            const string noLine = bom + "[SystemSettings]\r\nr.Other=1\r\n\r\n[Core.Log]\r\nLogNet=Log\r\n";
            WriteFileAtomic(ini, noLine.data(), noLine.size());
            require(RrConfigLineState(rc, *bl4, &f, &rwhy) == 0, "Engine.ini check with a BOM, the section without the line: no line (0) (r4 #5)");
            // adding: into that section (no second [SystemSettings]), the BOM kept; taking it out again: byte for byte as it was
            RrConfigFix fx;
            bool changed = false;
            const string added = RrConfigAddText(noLine, fx, &changed, true);
            size_t sections = 0;
            for (size_t at = added.find("[SystemSettings]"); at != string::npos; at = added.find("[SystemSettings]", at + 1)) ++sections;
            const string lineOnly = line.substr(line.find('\n') + 1);  // the line itself, with its CRLF
            require(changed && sections == 1 && !fx.sectionAdded && added.compare(0, 3, bom) == 0 &&
                        added == bom + "[SystemSettings]\r\nr.Other=1\r\n" + lineOnly + "\r\n[Core.Log]\r\nLogNet=Log\r\n",
                    "Engine.ini add with a BOM: the line under the existing [SystemSettings], no second section, the BOM kept (r4 #5)");
            require(RrConfigRemoveText(added, fx) == noLine, "Engine.ini revert with a BOM: byte for byte as before, the BOM kept (r4 #5)");
            bool againBom = true;
            RrConfigFix fx2;
            require(RrConfigAddText(withLine, fx2, &againBom, true) == withLine && !againBom, "Engine.ini add with a BOM and the line there: nothing to change (r4 #5)");
            // ...a file with the BOM alone: the section added after it, and taken out again leaves the BOM alone (reverted as empty)
            RrConfigFix fx3;
            const string onlyBom = RrConfigAddText(bom, fx3, &changed, true);
            require(onlyBom == bom + line && fx3.sectionAdded && RrConfigRemoveText(onlyBom, fx3) == bom,
                    "Engine.ini add to a file of a BOM alone: BOM, section, line; the revert leaves the BOM alone (r4 #5)");
        }

        // rc5: the launcher next to an installed copy decides by build at the same version (DecideLauncherStart), never by
        // the version string alone. Synthetic PE files: a COFF header with a link time, a tag byte that makes the files differ
        {
            auto pe = [&](const wchar_t* name, uint32_t linkTime, char tag) {
                string b(512, '\0');
                b[0] = 'M'; b[1] = 'Z';
                b[0x3C] = 0x40;                                   // e_lfanew
                b[0x40] = 'P'; b[0x41] = 'E';                     // the signature (two zero bytes follow)
                auto byte = [](unsigned v) { return static_cast<char>(static_cast<unsigned char>(v)); };
                b[0x44] = byte(0x64); b[0x45] = byte(0x86);       // Machine x64
                for (int i = 0; i < 4; ++i) b[0x48 + i] = byte((linkTime >> (8 * i)) & 0xFF);   // TimeDateStamp
                b[0x54] = byte(0xF0);                             // SizeOfOptionalHeader
                b[0x58] = 0x0B; b[0x59] = 0x02;                   // PE32+ magic
                b[400] = tag;
                const wstring p = Join(dir, name);
                WriteFileAtomic(p, b.data(), b.size());
                return p;
            };
            const uint32_t nov2 = 1793660000u, nov5 = 1793920000u;
            const wstring self = pe(L"self.exe", nov5, 'A'), same = pe(L"same.exe", nov5, 'A'), older = pe(L"older.exe", nov2, 'B'),
                          newer = pe(L"newer.exe", nov5 + 3600, 'C'), notPe = Join(dir, L"notpe.exe");
            WriteFileAtomic(notPe, "not a PE file", 13);
            require(FileLinkTime(self) == nov5 && FileLinkTime(notPe) == 0, "FileLinkTime: the COFF header's link time; 0 for a file that isn't a PE image");
            auto d = DecideLauncherStart(self, L"0.6.0", same, L"0.6.0");
            require(d.choice == LauncherStart::OpenInstalled && d.sameVersion && d.sameBuild, "same version, same build (same bytes): the installed copy opens");
            d = DecideLauncherStart(self, L"0.6.0", older, L"0.6.0");
            require(d.choice == LauncherStart::OfferReplace && d.sameVersion && !d.sameBuild && d.installedBuilt == nov2 && d.why.find(L"older build") != wstring::npos,
                    "same version, a different and older build installed: this copy offers to replace it (never opens the older build)");
            d = DecideLauncherStart(self, L"0.6.0", newer, L"0.6.0");
            require(d.choice == LauncherStart::OpenInstalled && d.sameVersion && !d.sameBuild && d.why.find(L"newer build") != wstring::npos,
                    "same version, a newer build installed: the installed copy opens");
            // an installed copy that isn't a readable program (corrupt, cut short): this copy runs, whatever its entry's version
            for (const wchar_t* entryVersion : {L"0.6.0", L"0.6.1", L"0.4.5"}) {
                d = DecideLauncherStart(self, L"0.6.0", notPe, entryVersion);
                require(d.choice == LauncherStart::RunHere && d.installedBuilt == 0 && d.why.find(L"isn't a readable program") != wstring::npos,
                        "an unreadable installed copy (same, newer or older version in its entry): this copy runs");
            }
            d = DecideLauncherStart(self, L"0.6.0", newer, L"0.5.5");
            require(d.choice == LauncherStart::OfferReplace && !d.sameVersion, "an older version installed (even a newer file): this copy offers to replace it");
            d = DecideLauncherStart(self, L"0.6.0", older, L"0.4.5");
            require(d.choice == LauncherStart::OfferReplace && !d.sameVersion && d.why.find(L"older version") != wstring::npos,
                    "0.4.5 installed: a newer running copy never defers to it (offers to replace it)");
            d = DecideLauncherStart(self, L"0.6.0", older, L"0.6.1");
            require(d.choice == LauncherStart::OpenInstalled && !d.sameVersion, "a newer version installed (even an older file): the installed copy opens");
            require(BuildTimeText(0) == L"unknown" && BuildTimeText(nov5).size() == 20 && BuildTimeText(nov5) == L"2026-11-05 23:06 UTC", "BuildTimeText: \"YYYY-MM-DD hh:mm UTC\"");
        }

        RemoveTree(dir);
        std::printf("launcher-core: %d checks passed\n", checks);
        return 0;
    } catch (const std::exception& e) {
        std::printf("launcher-core FAILED after %d checks: %s\n", checks, e.what());
        return 1;
    } catch (const SetupError& e) {
        std::printf("launcher-core FAILED after %d checks: %ls\n", checks, e.msg.c_str());
        return 1;
    }
}
