// SPDX-License-Identifier: MIT
// Unit tests of the launcher's coexistence part of wg_core (build/launcher-design.md sections 5 and 7.1, milestone M2):
// the marker scan and fingerprints on synthetic bytes (never a third-party file), the proxy picker on a generated
// folder, byte-exact OptiScaler.ini value edits and the inikey journal (revert, outside edits, replay onto a new ini),
// the OptiScaler tab's key rules, record v2's inikey and coexist lines, the log warnings (foreign tables, no depth,
// XeSS, plugin not hooked). Written for this project. Run by ctest ("coexist-core").
#include <cstdio>
#include <stdexcept>

#include "wg_core/coexist.h"
#include "wg_core/fingerprint.h"
#include "wg_core/logtail.h"
#include "wg_core/optikeys.h"
#include "wg_core/record.h"

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
    wstring dir = Join(tmp, L"coexist-tests-" + std::to_wstring(GetCurrentProcessId()));
    RemoveTree(dir);
    EnsureDir(dir);
    return dir;
}
void Put(const wstring& path, const string& data) { EnsureDir(Parent(path)); WriteFileAtomic(path, data.data(), data.size()); }
Tool ToolOf(const string& bytes, const wstring& name = L"") { return IdentifyBytes((const uint8_t*)bytes.data(), bytes.size(), name).tool; }
string Utf16(const char* s) { string r; for (; *s; ++s) { r += *s; r += '\0'; } return r; }
}  // namespace

int main() {
    try {
        // ---- marker scan -------------------------------------------------------------------------------------------
        const char* const needles[] = {"reshade", "dxvk", "optiscaler"};
        const string text = "xx RESHADE yy" + Utf16("DxVk") + "zz";
        require(ScanMarkers((const uint8_t*)text.data(), text.size(), needles, 3) == 3u, "markers: ASCII and UTF-16LE, any case");
        require(ScanMarkers((const uint8_t*)"resha", 5, needles, 3) == 0u, "markers: a cut-off needle is not found");

        // ---- fingerprints by content (the name is never used) --------------------------------------------------------
        require(ToolOf("MZ synthetic OptiScaler proxy") == Tool::OptiScaler, "OptiScaler by its text");
        require(ToolOf("MZ OptiScaler also loads ReShade and Special K and checks dxvk") == Tool::OptiScaler, "OptiScaler wins over the tools it mentions");
        require(ToolOf("MZ SpecialK has a ReShade add-on") == Tool::SpecialK, "Special K wins over ReShade");
        require(ToolOf("MZ reshade by crosire") == Tool::ReShade, "ReShade by its text");
        require(ToolOf("MZ vkd3d-proton d3d12", L"d3d12.dll") == Tool::Vkd3d && ToolOf("MZ DXVK_LOG_LEVEL", L"D3D11.dll") == Tool::Dxvk &&
                    ToolOf("MZ dxvk.conf", L"dxgi.dll") == Tool::Dxvk && ToolOf("MZ VKD3D_CONFIG", L"d3d12core.dll") == Tool::Vkd3d, "vkd3d-proton and DXVK by their own settings names");
        // 2026-10-17 regression (a false "DXVK as nvngx.dll" refusal): DXVK / vkd3d-proton only under their own DLL names and
        // only by their own markers; a bare "dxvk" (four letters a large binary holds by chance, or another tool's text) is not DXVK
        require(ToolOf("MZ DXVK_LOG_LEVEL dxvk.conf", L"nvngx.dll") == Tool::Unknown && ToolOf("MZ VKD3D_CONFIG", L"dxgi.dll") == Tool::Unknown &&
                    ToolOf("MZ DXVK_LOG_LEVEL") == Tool::Unknown, "DXVK markers under another name (or none) are not DXVK");
        require(ToolOf("MZ data dXvK data vkd3d", L"dxgi.dll") == Tool::Unknown && ToolOf("MZ dxvk d3d11", L"d3d11.dll") == Tool::Unknown, "a bare dxvk / vkd3d is not DXVK");
        require(ToolOf("MZ OptiScaler checks DXVK: DXVK_CONFIG_FILE dxvk.conf", L"dxgi.dll") == Tool::OptiScaler, "an OptiScaler build naming DXVK's settings is OptiScaler");
        require(ToolOf("Ultimate ASI Loader") == Tool::AsiLoader && ToolOf("by ThirteenAG") == Tool::AsiLoader, "Ultimate ASI Loader");
        require(ToolOf(Utf16("cyber_engine_tweaks")) == Tool::Cet && ToolOf("RED4ext.dll") == Tool::Red4ext && ToolOf("UE4SS proxy") == Tool::Ue4ss, "CET, RED4ext, UE4SS");
        require(ToolOf("fakenvapi") == Tool::FakeNvapi, "fakenvapi");
        require(ToolOf("dlssnr_on_amd") == Tool::NrOnAmd, "the DLSS-NR on AMD mod");
        require(ToolOf("an unrelated proxy of another tool") == Tool::Unknown, "unknown stays unknown");
        require(ToolOf("") == Tool::Unknown, "empty file");
        Tool t;
        for (Tool x : {Tool::OptiScaler, Tool::ReShade, Tool::AsiLoader, Tool::NrOnAmd, Tool::GameApi}) require(ParseToolKey(ToolKey(x), t) && t == x, "tool keys round trip");
        Decision d;
        for (Decision x : {Decision::Managed, Decision::AskSwitch, Decision::Chain, Decision::Warn, Decision::Inform, Decision::Refuse, Decision::Migrate})
            require(ParseDecision(DecisionName(x), d) && d == x, "decision names round trip");
        const wchar_t* msg = nullptr;
        require(OverlayFor(L"rtss.EXE", d, &msg) && d == Decision::Inform, "RTSS: inform");
        require(OverlayFor(L"LosslessScaling.exe", d, &msg) && wstring(msg).find(L"pointless") != wstring::npos, "Lossless Scaling: upscaling twice is pointless");
        require(OverlayFor(L"SKIF.exe", d, &msg) && d == Decision::Warn, "Special K's global injector: warn");
        require(!OverlayFor(L"notepad.exe", d, &msg), "other processes: nothing");

        // ---- the proxy picker ------------------------------------------------------------------------------------------
        const wstring dir = TempFolder();
        const wstring g = Join(dir, L"game");
        EnsureDir(g);
        require(PickProxy(g, nullptr, L"", false, false).name == L"dxgi.dll", "an empty folder: dxgi.dll");
        Put(Join(g, L"dxgi.dll"), "MZ reshade by crosire");
        Put(Join(g, L"winmm.dll"), "MZ an unknown proxy");
        ProxyChoice pc = PickProxy(g, nullptr, L"", false, true);
        require(pc.name == L"version.dll" && pc.reason.find(L"dxgi.dll (ReShade)") != wstring::npos && pc.reason.find(L"winmm.dll (an unknown file)") != wstring::npos &&
                    pc.untested && pc.reason.find(L"untested") != wstring::npos, "ReShade and an unknown file are skipped; the reason names them");
        require(PickProxy(g, nullptr, L"", true, false).name == L"wininet.dll", "next to a HIP-based tool version.dll is never taken");
        Record ours; ours.proxy = L"winhttp.dll";
        require(PickProxy(g, &ours, L"", false, false).name == L"winhttp.dll", "our earlier install's name (free) is reused");
        Put(Join(g, L"winhttp.dll"), "MZ someone else's file");
        require(PickProxy(g, &ours, L"", false, false).name == L"version.dll", "our earlier name is not reused when another file holds it");
        for (auto n : {L"version.dll", L"wininet.dll", L"winhttp.dll", L"dbghelp.dll"}) Put(Join(g, n), "MZ taken");
        pc = PickProxy(g, nullptr, L"", false, false);
        require(pc.name.empty() && pc.reason.rfind(L"no free name", 0) == 0, "every name taken: none, and why");
        require(PickProxy(g, nullptr, L"winmm.dll", false, false).name == L"winmm.dll", "a switch takes the player's OptiScaler name only when no other is free");
        for (auto n : kNeverProxy) for (auto o : kProxyOrder) require(!IEq(n, o), "d3d12, dinput8, dwmapi and xinput are never proxy names");

        // ---- byte-exact ini edits --------------------------------------------------------------------------------------
        const string ini = "; OptiScaler (synthetic)\r\n[Menu]\r\nShortcutKey = auto  \r\nShowFps=auto\r\n[Inputs]\r\nEnableXeSSInputs=auto\r\n\r\n[Upscalers]\r\nDx12Upscaler=auto\r\n";
        IniSpan sp = IniFind(ini, "menu", "shortcutkey");
        require(sp.found && IniValueAt(ini, sp) == "auto", "find a key ignoring case; the value without spaces");
        string edited = IniReplaceValue(ini, sp, "0x2D");
        require(edited == "; OptiScaler (synthetic)\r\n[Menu]\r\nShortcutKey = 0x2D  \r\nShowFps=auto\r\n[Inputs]\r\nEnableXeSSInputs=auto\r\n\r\n[Upscalers]\r\nDx12Upscaler=auto\r\n",
                "only the value's bytes change");
        require(!IniFind(ini, "Menu", "Scale").found && !IniFind(ini, "Inputs", "ShortcutKey").found, "keys of other sections or absent keys are not found");
        require(IniEntries(ini).size() == 4, "every key line once");

        // ---- the OptiScaler tab's rules ------------------------------------------------------------------------------------
        string n; wstring why;
        const OptiKeyDef* key = FindOptiKey("Menu", "ShortcutKey");
        require(key && ValidateOptiValue(*key, "0x2d", &n, &why) && n == "0x2D" && ValidateOptiValue(*key, "45", &n, &why) && ValidateOptiValue(*key, "AUTO", &n, &why) && n == "auto",
                "key codes: hex, decimal, auto");
        require(!ValidateOptiValue(*key, "0x1FF", &n, &why) && !ValidateOptiValue(*key, "insert", &n, &why) && !ValidateOptiValue(*key, "0", &n, &why), "bad key codes refused");
        const OptiKeyDef* xess = FindOptiKey("inputs", "enablexessinputs");
        require(xess && ValidateOptiValue(*xess, "On", &n, &why) && n == "true" && !ValidateOptiValue(*xess, "maybe", &n, &why) && !ValidateOptiValue(*xess, "true;x", &n, &why),
                "booleans: true/false/auto only, no comment characters");
        require(IsOptiLocked("upscalers", "dx12upscaler") && IsOptiLocked("Menu", "ShowFps") && IsOptiLocked("Plugins", "Path") && !IsOptiLocked("Menu", "ShortcutKey"),
                "locked: the keys WrongGPU writes, and the Settings page's FPS overlay");
        for (size_t i = 0; i < kOptiEditableCount; ++i) require(!IsOptiLocked(kOptiEditable[i].section, kOptiEditable[i].key), "no key is both editable and locked");
        const auto rows = OptiTabRows(ini);
        size_t locked = 0, editable = 0;
        for (auto& r : rows) (r.locked ? locked : editable)++;
        require(locked == 2 && editable == 2, "the tab offers only keys this ini has (2 locked, 2 editable)");
        string out;
        require(!ApplyOptiEdits(ini, {{"Upscalers.Dx12Upscaler", "xess"}}, out, &why) && why.find(L"managed by WrongGPU") != wstring::npos, "a locked key is refused");
        require(!ApplyOptiEdits(ini, {{"Menu.Scale", "1.5"}}, out, &why) && why.find(L"has no") != wstring::npos, "a key this build lacks is refused");
        require(!ApplyOptiEdits(ini, {{"Menu.NoSuch", "1"}}, out, &why) && !ApplyOptiEdits(ini, {{"nodot", "1"}}, out, &why), "unknown keys refused");
        require(ApplyOptiEdits(ini, {{"menu.shortcutkey", "0x2d"}, {"Inputs.EnableXeSSInputs", "false"}}, out, &why), "two edits");

        // ---- the journal --------------------------------------------------------------------------------------------------
        std::vector<IniKeyChange> j;
        JournalChanges(j, L"OptiScaler.ini", ini, out);
        require(j.size() == 2 && j[0].original == "auto" && j[0].ours == "0x2D" && j[1].ours == "false", "the journal: originals and ours");
        require(RevertJournal(out, j, L"OptiScaler.ini", nullptr) == ini, "revert: byte for byte the ini before the edits");
        string back;
        ApplyOptiEdits(out, {{"Inputs.EnableXeSSInputs", "auto"}}, back, &why);
        JournalChanges(j, L"OptiScaler.ini", out, back);
        require(j.size() == 1 && j[0].key == "ShortcutKey", "a key back at its original leaves the journal");
        const string menuSaved = IniReplaceValue(back, IniFind(back, "Menu", "ShortcutKey"), "0x2E");
        const auto outside = JournalOutside(menuSaved, j, L"OptiScaler.ini");
        require(outside.size() == 1 && outside[0].fileValue == "0x2E" && outside[0].ours == "0x2D", "a value saved over ours is an outside edit");
        require(JournalOutside(back, j, L"OptiScaler.ini").empty(), "no outside edit when the file has ours");
        std::vector<wstring> notes;
        require(RevertJournal(menuSaved, j, L"OptiScaler.ini", &notes) == ini && notes.size() == 1, "revert after an outside edit: the original, with a note");
        const string fresh = "[Menu]\nShortcutKey=auto\nScale=1.0\n[Upscalers]\nDx12Upscaler=fsr31\n";  // an update's new ini, another layout
        std::vector<IniKeyChange> j2;
        std::vector<IniKeyChange> old = j;
        old.push_back({L"OptiScaler.ini", "Inputs", "EnableXeSSInputs", false, "auto", "false"});
        old.push_back({L"OptiScaler.ini", "Upscalers", "Dx12Upscaler", false, "auto", "fsr31"});
        notes.clear();
        const string replayed = ReplayJournal(fresh, old, L"OptiScaler.ini", j2, &notes);
        require(replayed == "[Menu]\nShortcutKey=0x2D\nScale=1.0\n[Upscalers]\nDx12Upscaler=fsr31\n" && j2.size() == 1 && j2[0].original == "auto" && notes.size() == 1,
                "replay onto a new ini: the player's value carried over, a key the build lacks noted, locked keys left to the install");

        // ---- record v2: inikey and coexist lines ----------------------------------------------------------------------------
        Record m;
        m.version = L"0.5.5"; m.game = Join(dir, L"rec"); m.exe = Join(m.game, L"Game.exe");
        m.inikeys = {{L"OptiScaler.ini", "Menu", "ShortcutKey", false, "", "0x2D"}, {L"OptiScaler.ini", "Menu", "Added", true, "", "on"}};
        m.coexist = {{L"reshade", L"ReShade 6.3.3 as dxgi.dll [version resource: ReShade]", L"chain"}};
        Entry e; e.action = L"copy"; e.rel = L"dxgi.dll"; e.inst = L"ab"; e.owner = L"wg-optibundle"; m.e.push_back(e);
        SaveRecord(RecordPath(m.game), m);
        Record r2;
        require(LoadRecord(RecordPath(m.game), r2) && r2.inikeys.size() == 2 && r2.inikeys[0].original.empty() && !r2.inikeys[0].absent && r2.inikeys[0].ours == "0x2D" &&
                    r2.inikeys[1].absent && r2.coexist.size() == 1 && r2.coexist[0].decision == L"chain" && r2.extra.empty() && r2.e.size() == 1 && r2.e[0].owner == L"wg-optibundle",
                "record v2: inikey (an empty original, an added key) and coexist lines round trip, not as unknown lines");
        string raw;
        ReadAll(RecordPath(m.game), raw);
        require(raw.find("inikey\tOptiScaler.ini\tMenu\tShortcutKey\t=\t=0x2D\r\n") != string::npos && raw.find("\tabsent\t=on\r\n") != string::npos,
                "inikey lines: \"=value\", \"absent\"");

        // ---- log lines: only real warnings count (2026-10-17: an info line was shown with "!" and counted) -----------------
        require(!IsProblemLine("d3d12 hooks: installed (device 1, queue 1, command list 3 vtables); split timeout 2000 ms. Command lists already being recorded "
                               "are tracked from their next Reset; until then a split of them answers WG_ERROR_NOT_READY"), "hooks installed: info, not a warning");
        require(!IsProblemLine("  U1 upscaler info: copy paths requested: output_copy (each frame falls back to the pass if the formats do not allow a copy)") &&
                    !IsProblemLine("  U1 upscaler info: network runtime ready: executor dx12 (no HIP runtime or fallback)") &&
                    !IsProblemLine("settings: upscaler=network fallback_to_amd=1"), "upscaler info, runtime ready and the settings line: info");
        require(IsProblemLine("ERROR: hooks incomplete (queue vtables 0, command list vtables 0): no split, no trace") &&
                    IsProblemLine("  U1 upscaler ERROR: no HIP device found") && IsProblemLine("[support] WG-ERROR incident=1-2") &&
                    IsProblemLine("split of list 0000 refused (WG_ERROR_NOT_READY): the list's Reset has not been seen") &&
                    IsProblemLine("C1 upscaler context goes to AMD FSR: the runtime did not start") && IsProblemLine("ffxDispatch -> FFX_OK (AMD's output kept; C1 stays on AMD)"),
                "errors, WG-ERROR incidents, refused splits and fallback reasons count");
        const LastRun info = ParseLastRun(
            "[1] [x] ==== session start: Cyberpunk2077.exe (process 1) ====\r\n[1] [x] settings: upscaler=network\r\n"
            "[2] [amd_fidelityfx_dx12] d3d12 hooks: installed (device 1, queue 1, command list 3 vtables); split timeout 2000 ms. until then a split of them answers WG_ERROR_NOT_READY\r\n"
            "[3] [amd_fidelityfx_dx12]   U1 upscaler info: network runtime ready: executor dx12, sync inline\r\n");
        require(info.notes.empty() && info.Summary().find(L"warning") == wstring::npos, "a clean run with info lines: no warning on the game page");
        // start-up refusals (the first frames go to AMD FSR while the runtime starts) are routine once it is ready: not counted
        // (UI review #8); in a run whose runtime never started they stay, they say why
        const string startup =
            "[1] [x] ==== session start: Cyberpunk2077.exe (process 1) ====\r\n[1] [x] settings: upscaler=network\r\n"
            "[2] [fe] upscaler: frame 1 refused (WG_ERROR_NOT_READY): goes to AMD FSR until the runtime is ready\r\n"
            "[2] [hooks] split of list 0000 refused (WG_ERROR_NOT_READY): the list's Reset has not been seen\r\n";
        const LastRun ready = ParseLastRun(startup + "[3] [x]   U1 upscaler info: network runtime ready: executor dx12, sync inline\r\n");
        require(ready.ready && ready.notes.empty() && ready.routine.size() == 2 && ready.Summary().find(L"warning") == wstring::npos,
                "start-up refusals of a run whose runtime became ready: routine, no note on the game page");
        const LastRun failed = ParseLastRun(startup + "[3] [x] upscaler ERROR: the runtime did not start\r\n");
        require(!failed.ready && failed.notes.size() == 3 && failed.routine.empty(), "start-up refusals of a run whose runtime never started: they stay notes");
        const LastRun mixed = ParseLastRun(startup + "[3] [x]   U1 upscaler info: network runtime ready: executor dx12, sync inline\r\n"
                                                     "[4] [x] C1 upscaler context goes to AMD FSR: a resource is missing\r\n");
        require(mixed.notes.size() == 1 && mixed.notes[0].find(L"resource is missing") != wstring::npos, "a real fallback after the start still counts");
        // the same code once the runtime is ready is a real refusal (a list's table WrongGPU can't take over, a split lane in
        // use): repeated refusals after the start stay notes, and so does a start-up line that comes again after it (UI review r2 #4)
        const LastRun late = ParseLastRun(startup + "[3] [x]   U1 upscaler info: network runtime ready: executor dx12, sync inline\r\n"
                                                    "[4] [hooks] split of list 0001 refused (WG_ERROR_NOT_READY): the list's own vtable could not be taken over\r\n"
                                                    "[5] [hooks] split of list 0001 refused (WG_ERROR_NOT_READY): the list's own vtable could not be taken over\r\n"
                                                    "[6] [hooks] split of list 0002 refused (WG_ERROR_NOT_READY): 4 other lists of this type have splits waiting for their submission\r\n"
                                                    "[7] [hooks] split of list 0000 refused (WG_ERROR_NOT_READY): the list's Reset has not been seen\r\n");
        require(late.ready && late.notes.size() == 3 && late.routine.size() == 1 &&
                    late.notes[0].find(L"list 0000") != wstring::npos && late.notes[1].find(L"list 0001") != wstring::npos &&
                    late.notes[2].find(L"list 0002") != wstring::npos && late.routine[0].find(L"frame 1 refused") != wstring::npos &&
                    late.Summary().find(L"3 warning(s)") != wstring::npos,
                "NOT_READY refusals after the runtime was ready (repeated, and a start-up line again): notes; the start-up refusal alone is routine");
        // a frame upscaled by Model M (UI review r2 #10): the runtime logs "ready" before any frame, so ready alone is not
        // enough; a frame logged once it was ready with nothing sending it elsewhere is, and so is a late start's line
        const string readyLine = "[3] [x]   U1 upscaler info: network runtime ready: executor dx12, sync inline\r\n";
        const LastRun readyOnly = ParseLastRun(startup + readyLine);
        const LastRun framed = ParseLastRun(startup + readyLine + "[4] [fe] #12 WrongGPU frame: R=1280x720 O=2560x1440 reset=1\r\n[4] [fe] #12 -> OK (WrongGPU)\r\n");
        const LastRun standIn = ParseLastRun("[1] [x] ==== session start: Cyberpunk2077.exe (process 1) ====\r\n[1] [x] settings: upscaler=network\r\n" + readyLine +
                                             "[4] [fe] #12 WrongGPU frame: R=1280x720 O=2560x1440 reset=1\r\n"
                                             "[4] [fe] #12 note U1: upscaler not ready this frame (WG_ERROR_NOT_READY): stand-in for this frame\r\n[4] [fe] #12 -> OK (WrongGPU)\r\n");
        const LastRun early = ParseLastRun("[1] [x] ==== session start: Cyberpunk2077.exe (process 1) ====\r\n[1] [x] settings: upscaler=network\r\n"
                                           "[2] [fe] #3 WrongGPU frame: R=1280x720 O=2560x1440 reset=1\r\n[2] [fe] #3 -> OK (AMD FSR)\r\n" + readyLine);
        // a late start: its line is logged before the frame's dispatch, so it alone is not a frame upscaled (r3 #2)
        const string lateLine = "[5] [fe] #40: WrongGPU's upscaler \"network\" started after 2.1 s; it upscales from this frame on\r\n";
        const LastRun late2 = ParseLastRun(startup + readyLine + lateLine);
        require(readyOnly.ready && !readyOnly.upscaled && framed.upscaled && !standIn.upscaled && !early.upscaled && !late2.upscaled,
                "Model M upscaled a frame: a frame after ready whose dispatch ended OK (WrongGPU) with nothing sending it elsewhere; not ready alone, a "
                "stand-in frame, a frame before ready, or a late start's announcement alone (r3 #2)");
        // ...a late start followed only by refusals and fallbacks: no frame upscaled (r3 #2)
        const LastRun lateFallback = ParseLastRun(startup + readyLine + lateLine +
            "[6] [fe] #41 note U1: frame 130: upscaler \"network\" dispatch failed: WG_ERROR_UNSUPPORTED (the upscaler does not take this frame): AMD FSR upscales this frame -> OK; "
            "the context stays WrongGPU's (AMD FSR takes it over after 30 such frames in a row)\r\n"
            "[6] [hooks] split of list 0003 refused (WG_ERROR_NOT_READY): the list's own vtable could not be taken over\r\n"
            "[7] [fe] #70 U1 now runs on AMD FSR for the rest of its life\r\n");
        require(lateFallback.ready && !lateFallback.upscaled && !lateFallback.notes.empty(), "a late start, then only refusals and fallbacks: not upscaled (r3 #2)");
        // ...a late start whose frames went on: the front end's "upscales again" after a run it couldn't take, or the runtime's
        // timing line counting network frames (frames after the third are logged only with log_dispatches); a stand-in's note
        // just before "upscales again", or a timing line of no frames, isn't (r3 #2)
        const LastRun lateAgain = ParseLastRun(startup + readyLine + lateLine +
            "[6] [fe] #41 note U1: frame 131: WrongGPU upscales again after 2 frame(s) it could not take (history reset)\r\n");
        const LastRun lateTiming = ParseLastRun(startup + readyLine + lateLine +
            "[8] [x]   U1 upscaler info: timing, last 300 dispatches: network GPU 1.108 (max 1.296) ms [enc0 0.102, Swin 0.573] (298 frames); TOTAL(t3-t0) 1.114 ms\r\n");
        const LastRun noFrames = ParseLastRun(startup + readyLine + lateLine +
            "[8] [x]   U1 upscaler info: timing, last 300 dispatches: network GPU 0.000 (max 0.000) ms (0 frames); TOTAL(t3-t0) 0.000 ms\r\n");
        const LastRun againAfterStandIn = ParseLastRun(startup + readyLine + lateLine +
            "[6] [fe] #41 note U1: upscaler not ready this frame (WG_ERROR_NOT_READY): the stand-in upscales it\r\n"
            "[6] [fe] #41 note U1: frame 131: WrongGPU upscales again after 2 frame(s) it could not take (history reset)\r\n");
        require(lateAgain.upscaled && lateTiming.upscaled && !noFrames.upscaled && !againAfterStandIn.upscaled,
                "after a late start: upscales again, or a timing line of network frames, is a frame upscaled; a timing line of no frames or "
                "upscales again right after a stand-in's note is not (r3 #2)");
        // r4 #3: the direct route's FSR 3.0 hook (clean/fsr fsr30_hook.cpp) logs a frame WrongGPU upscaled as "-> FFX_OK (WrongGPU
        // network)" (with AMD first for frame generation, or AMD's own call skipped), never "OK (WrongGPU)"; AMD's output kept, and
        // the stand-in, are not upscaled by Model M
        {
            const string d0 = "[1] [x] ==== session start: Game.exe (process 1) ====\r\n[1] [x] settings: upscaler=network\r\n"
                              "[2] [fsr3] #1 ffxFsr3UpscalerContextCreate context=1 (C1)\r\n" + readyLine +
                              "[4] [fsr3] #9 WrongGPU frame: R=1280x720 O=2560x1440 reset=1\r\n";
            const char* const yes[] = {"[4] [fsr3] #9 -> FFX_OK (WrongGPU network)",
                                       "[4] [fsr3] #9 -> FFX_OK (AMD for the frame-generation resources, then WrongGPU network)",
                                       "[4] [fsr3] #9 -> FFX_OK (WrongGPU network; AMD's upscaler skipped: no frame generation)",
                                       "[4] [fsr3] #9 -> FFX_OK (WrongGPU network; AMD's ffxFsr3ContextDispatchUpscale skipped: UPSCALING_ONLY)",
                                       "[4] [fsr3] #9 -> FFX_OK (the WrongGPU upscaler)"};
            const char* const no[] = {"[4] [fsr3] #9 -> FFX_OK (AMD's output kept while WrongGPU's upscaler starts)",
                                      "[4] [fsr3] #9 -> FFX_OK (AMD's output kept; C1 stays on AMD)", "[4] [fsr3] #9 -> FFX_OK (WrongGPU stand-in)",
                                      "[4] [fsr3] #9 -> FFX_OK (AMD)", "[4] [fsr3] #9 -> FFX_OK (AMD after WrongGPU network failed; C1 stays on AMD)"};
            bool allYes = true, anyNo = false;
            for (const char* l : yes) { const LastRun one = ParseLastRun(d0 + l + "\r\n"); allYes = allYes && one.upscaled && one.created && one.ready; }
            for (const char* l : no) anyNo = anyNo || ParseLastRun(d0 + l + "\r\n").upscaled;
            const LastRun beforeReady = ParseLastRun("[1] [x] ==== session start: Game.exe (process 1) ====\r\n[1] [x] settings: upscaler=network\r\n"
                                                     "[2] [fsr3] #9 WrongGPU frame: R=1280x720 O=2560x1440 reset=1\r\n[2] [fsr3] #9 -> FFX_OK (WrongGPU network)\r\n" + readyLine);
            const LastRun createOnly = ParseLastRun("[1] [x] ==== session start: Game.exe (process 1) ====\r\n[1] [x] settings: upscaler=network\r\n"
                                                    "[2] [fsr3] #1 ffxFsr3UpscalerContextCreate context=1 (C1)\r\n");
            require(allYes && !anyNo && !beforeReady.upscaled && createOnly.created && !createOnly.upscaled,
                    "the direct route's FSR 3.0 success lines (FFX_OK with WrongGPU's network) count as a frame upscaled; AMD's output kept, AMD, a "
                    "failure and the stand-in don't; nor a frame before the runtime was ready; its create counts as the game asking (r4 #3)");
        }
        // today's real Cyberpunk 2077 runs of the release candidate (tests/logs: trimmed, paths replaced): Model M Quality, native
        // and with Ray Reconstruction: started on Direct3D 12 and upscaled (r3 #2)
        for (const wchar_t* f : {L"cyberpunk-quality-fsr.log", L"cyberpunk-native-fsr.log", L"cyberpunk-rr-fsr.log"}) {
            string logText;
            require(ReadAll(Join(FromUtf8(WG_TEST_LOGS), f), logText), "the real-log fixture is there");
            const LastRun rl = ParseLastRun(logText);
            require(rl.found && rl.created && rl.ready && rl.upscaled && rl.executor == L"dx12" && rl.process == L"Cyberpunk2077.exe" && rl.upscaler == L"network" &&
                        rl.notes.empty() && rl.warnings.empty(),
                    "a real run (tests/logs): created, ready on dx12, upscaled, no notes or warnings");
            // WrongGPU's edition of OptiScaler, its custom upscaler slot: chosen by name, so the page doesn't say the overlay shows AMD FSR (r3 #3)
            require(rl.optiPlugin, "a real run through the edition's custom upscaler slot: WrongGPU chosen by name");
        }
        // the whole logs of those runs, where this PC has them (WRONGGPU_TEST_REALLOGS: build-ui/reallogs, not committed): the
        // same, read as the launcher reads a game's folder (ReadLastRun: the front end's log tail, the RR log, OptiScaler.log)
        if (const wstring real = EnvVar(L"WRONGGPU_TEST_REALLOGS"); !real.empty()) {
            for (const wchar_t* run : {L"cyberpunk_rc3-quality", L"cyberpunk_rc060-native", L"cyberpunk_rc060-rr-exit", L"bl4_rc3-cold1", L"hogwarts_rc3-rr-final",
                                       L"re9_rc3-rr", L"re9_rc060-rr-exit", L"cyberpunk_rc6-steam-play", L"cyberpunk_rc6-steam-rr"}) {
                // (rc6: the folders this PC has; cyberpunk_rc6-steam-play: Play through Steam and REDlauncher, the custom slot's
                // log and the RR log whose last attach is a helper process of the game: one session, Model M upscaled)
                if (!IsDir(Join(real, run))) { std::printf("real log %ls: not on this PC, skipped\n", run); continue; }
                const LastRun rl = ReadLastRun(Join(real, run));
                std::printf("real log %ls: found=%d created=%d ready=%d upscaled=%d by_name=%d executor=%ls notes=%zu warnings=%zu rr_found=%d rr_used=%d rr_mode=%ls "
                            "rr_ms=%ls rr_frames=%lld rr_only=%d rr_upscaled=%d rr_note=%ls size=%ls>%ls summary=%ls\n",
                            run, rl.found, rl.created, rl.ready, rl.upscaled, rl.optiPlugin, rl.executor.c_str(), rl.notes.size(), rl.warnings.size(), rl.rrFound, rl.rrUsed,
                            rl.rrMode.c_str(), rl.rrMs.c_str(), rl.rrFrames, rl.rrOnly, rl.rrUpscaled, rl.rrNote.c_str(), rl.renderSize.c_str(), rl.outputSize.c_str(), rl.Summary().c_str());
                const wstring rn = run;
                // (r5 #2: in cyberpunk_rc060-rr-exit the RR log's last session is a later process, 16784, with no front-end session:
                // the newest session alone, an RR-only run, not the front end's older session merged with it)
                if (rn == L"cyberpunk_rc060-rr-exit")
                    require(rl.found && rl.upscaled && rl.rrOnly && !rl.ready && rl.rrFrames == 30962 && rl.rrClock.pid == 16784 && !rl.rrMs.empty(),
                            "a whole real log whose RR session is a later launch than the front end's: that RR session alone (r5 #2)");
                // (rc6: the next launch, Ray Reconstruction alone through Steam and REDlauncher, the front end's log the earlier
                // session's: that RR session alone, newer than the front end's; its last attach a helper process of the game)
                else if (rn == L"cyberpunk_rc6-steam-rr")
                    require(rl.found && rl.rrOnly && rl.rrUsed && rl.rrFrames == 13538 && rl.rrClock.pid == 34144 && !rl.when.empty(),
                            "a whole real log: the newer RR-only launch, not the earlier Model M session (rc6)");
                else if (rn.rfind(L"cyberpunk", 0) == 0)
                    require(rl.found && rl.ready && rl.upscaled && rl.executor == L"dx12" && rl.optiPlugin && !rl.rrOnly,
                            "a whole real log: ready on dx12, upscaled, chosen by name");
                else if (rn == L"re9_rc060-rr-exit")  // (that run's folder came back empty: nothing to read)
                    require(!rl.found && !rl.rrFound, "an empty real-log folder: no run");
                else  // r4 #1: the RR games ran Ray Reconstruction only (no front-end log): a run, upscaled (RR frames evaluated)
                    require(rl.found && rl.created && rl.upscaled && rl.rrOnly && rl.rrUsed && rl.rrFrames > 0 && !rl.ready && !rl.rrMs.empty() && !rl.when.empty(),
                            "a whole real RR-only log: a run with Ray Reconstruction frames, its time, its cost (r4 #1)");
                // r7 #4: Borderlands 4's real log cut right after the direct route's successful first evaluate (as a run that ended there,
                // before any timing or destroy line): that line alone is a frame
                if (rn == L"bl4_rc3-cold1") {
                    string whole;
                    require(ReadAll(Join(Join(real, run), L"wronggpu-rr.log"), whole), "the real Borderlands 4 RR log is there");
                    // (the runtime's own "instance 1 first evaluate" line comes just before it: cut after the route's line)
                    const size_t fe = whole.find("; runtime returned 0 (");
                    require(fe != string::npos && whole.rfind(" first evaluate: ", fe) != string::npos &&
                                whole.rfind(" first evaluate: ", fe) > whole.rfind('\n', fe), "...it has the direct route's successful first evaluate");
                    LastRun cut;
                    ParseRrLog(whole.substr(0, whole.find('\n', fe) + 1), cut);
                    std::printf("real log %ls cut after its first evaluate: rr_used=%d rr_frames=%lld\n", run, cut.rrUsed, cut.rrFrames);
                    require(CountRrRun(cut) && cut.rrFrames == 1 && cut.upscaled, "...cut after it: one frame, a run that worked (r7 #4)");
                }
            }
        }
        // r4 #1: the trimmed RR-only logs of those runs (tests/logs: the RR plugin's log alone, as the launcher finds it in a game's
        // folder): a run, created and upscaled (RR frames evaluated), its time from the RR log, the cost and size its timing gives
        {
            struct RrRun { const wchar_t* file; long long frames; const wchar_t* ms; const wchar_t* render; const wchar_t* output; };
            const RrRun runs[] = {{L"bl4-rr-only.log", 36044, L"1.6", L"642x361", L"1920x1080"},
                                  {L"hogwarts-rr-only.log", 9754, L"6.2", L"1279x720", L"1920x1080"},
                                  {L"re9-rr-only.log", 12317, L"4.0", L"1280x720", L"1920x1080"}};
            for (const RrRun& run : runs) {
                string logText;
                require(ReadAll(Join(FromUtf8(WG_TEST_LOGS), run.file), logText), "the RR-only fixture is there");
                const wstring gd = Join(dir, L"rr-only");
                RemoveTree(gd);
                Put(Join(gd, L"wronggpu-rr.log"), logText);
                const LastRun rl = ReadLastRun(gd);
                require(rl.found && rl.created && rl.upscaled && rl.rrOnly && rl.rrUsed && rl.rrFound && !rl.ready && rl.rrFrames == run.frames && rl.rrMs == run.ms &&
                            rl.renderSize == run.render && rl.outputSize == run.output && rl.when.size() == 16 && rl.notes.empty() && rl.warnings.empty() &&
                            rl.Summary().rfind(L"Ray Reconstruction (" + wstring(run.ms) + L" ms a frame)", 0) == 0,
                        "an RR-only run (tests/logs): found, created, upscaled, its frames, cost and size, its time; the summary says Ray Reconstruction (r4 #1)");
            }
            // ...an RR session that evaluated nothing (the game never asked for RR): not a run on its own
            const wstring gd = Join(dir, L"rr-none");
            RemoveTree(gd);
            Put(Join(gd, L"wronggpu-rr.log"), "00:00:01.000 load: process 7 attached C:\\Games\\X\\plugins\\WrongGPU-RR.asi (hook mode)\r\n"
                                              "00:10:00.000 diag: summary (process exit): support queries 6 (RR: yes 0 / no 1), RR lookups 0, RR instances created 0, RR evaluates 0\r\n");
            const LastRun none = ReadLastRun(gd);
            require(!none.found && !none.upscaled && none.rrFound && !none.rrUsed && none.rrFrames == 0 && none.when.empty(),
                    "an RR session with no frame evaluated: no run (Not played yet stays) (r4 #1)");
            // r5 #1: only success counts as frames. The plugin's evaluate counter counts calls before they are evaluated, so a session
            // whose evaluations all failed (tests/logs/rr-failed-only.log: "destroy ... after 100 frames (100 failed)", "RR evaluates
            // 100") or were all passed through (rr-passthrough-only.log; also without its destroy line, a process that ended before it)
            // has no frames and is not upscaled; the plugin's "live: evaluate N: runtime returned 0" is a frame evaluated.
            // r6 #2: such a session is still an attempt (the game asked for RR): a run that says no frame succeeded, with the log's
            // reason, never "Not played yet"; no Worked (upscaled false)
            for (const wchar_t* f : {L"rr-failed-only.log", L"rr-passthrough-only.log"}) {
                string lt;
                require(ReadAll(Join(FromUtf8(WG_TEST_LOGS), f), lt), "the failed RR fixture is there");
                for (int cut = 0; cut < 2; ++cut) {
                    string t2 = lt;
                    if (cut) { const size_t at = t2.find("runtime: destroy instance"); t2.erase(t2.rfind('\n', at) + 1, t2.find('\n', at) - t2.rfind('\n', at)); }
                    const wstring fd = Join(dir, L"rr-failed");
                    RemoveTree(fd);
                    Put(Join(fd, L"wronggpu-rr.log"), t2);
                    const LastRun fl = ReadLastRun(fd);
                    const bool passed = wstring(f) == L"rr-passthrough-only.log";
                    require(fl.found && fl.created && !fl.upscaled && !fl.AnyUpscaled() && fl.rrOnly && fl.rrFound && fl.rrUsed && fl.rrFrames == 0 &&
                                fl.when.size() == 16 && fl.Summary().rfind(L"Ray Reconstruction: no successful frame", 0) == 0 &&
                                fl.rrNote == (passed ? L"runtime not ready" : L"frame kernels still being prepared (first start with a cold driver cache)") &&
                                fl.rrPassThrough == passed,
                            "an RR session whose evaluations all failed or were passed through: no frames, not upscaled (r5 #1); an RR attempt: a run "
                            "with its time, the summary says no successful frame, the runtime's reason and the pass-through kept (r6 #2)");
                }
                LastRun ok;
                ParseRrLog(lt + "21:02:11.000 live: evaluate 101: runtime returned 0 (100 frame(s) passed through so far)\r\n", ok);
                require(ok.rrFrames == 1 && CountRrRun(ok) && ok.rrOnly && ok.upscaled, "...with the plugin's line for an evaluate the runtime finished: a frame, a run (r5 #1)");
            }
            // r7 #4: the direct route's own line for an evaluate the runtime finished (tests/logs/rr-direct-short.log: its exact format
            // after the route's prefix, a short run that ended before any timing or destroy line): a frame, a run that worked; with a
            // timing line too, the larger count (never both added); a failure line of the route is no frame
            {
                string sl;
                require(ReadAll(Join(FromUtf8(WG_TEST_LOGS), L"rr-direct-short.log"), sl), "the short direct-route RR fixture is there");
                const wstring fd = Join(dir, L"rr-direct-short");
                RemoveTree(fd);
                Put(Join(fd, L"wronggpu-rr.log"), sl);
                const LastRun sr = ReadLastRun(fd);
                require(sr.found && sr.rrOnly && sr.rrUsed && sr.rrFrames == 1 && sr.upscaled && sr.AnyUpscaled() && sr.rrNote.empty() &&
                            sr.Summary().rfind(L"Ray Reconstruction", 0) == 0 && sr.Summary().find(L"no successful") == wstring::npos,
                        "a short direct-route run with only its successful first evaluate: one frame, upscaled, not a failed attempt (r7 #4)");
                LastRun both;
                ParseRrLog(sl + "00:25:57.882 runtime: instance 1 timing, 300 frames from frame 1: GPU 1.581 ms per frame (pre 0.012), render 642x361, output 1920x1080\r\n", both);
                require(both.rrFrames == 300, "...with a timing line of 300 frames: 300, the larger count, not 301 (r7 #4)");
                LastRun bad;
                ParseRrLog(sl.substr(0, sl.find("00:25:55.321")) + "00:25:55.321 RR feature 1: evaluate 1: runtime returned 13; colour passed through, success reported, "
                                                                   "reset on the next evaluate\r\n", bad);
                require(bad.rrUsed && bad.rrFrames == 0 && bad.rrPassThrough, "...the route's failure line instead: no frame, the pass-through (r7 #4)");
            }
            // r7 #3: the plugin's direct route refusing RR before any instance (tests/logs/rr-refused-only.log: the runtime or model not available) is
            // an attempt that failed: a run with its reason, never "Not played yet"; the direct route's provider merely registered is no attempt,
            // and neither is a refusal because the game-interface route already runs RR in the game
            {
                string rl;
                require(ReadAll(Join(FromUtf8(WG_TEST_LOGS), L"rr-refused-only.log"), rl), "the refused RR fixture is there");
                const wstring fd = Join(dir, L"rr-refused");
                RemoveTree(fd);
                Put(Join(fd, L"wronggpu-rr.log"), rl);
                const LastRun fl = ReadLastRun(fd);
                require(fl.found && fl.created && fl.rrOnly && fl.rrUsed && fl.rrFrames == 0 && !fl.upscaled && !fl.AnyUpscaled() &&
                            fl.rrClock.pid == 5200 && fl.when.size() == 16 && !fl.rrPassThrough &&
                            fl.rrNote == L"the game's request was refused (WrongGPU's runtime or model is not available)" &&
                            fl.Summary().rfind(L"Ray Reconstruction: no successful frame", 0) == 0,
                        "the direct route's refusal alone: an RR attempt without a frame, a run with its time and reason (r7 #3)");
                LastRun reg;
                ParseRrLog(rl.substr(0, rl.find("RR feature refused")), reg);
                require(reg.rrFound && !reg.rrUsed && reg.rrNote.empty() && !CountRrRun(reg), "...the provider registered alone: no attempt (r7 #3)");
                LastRun busy;
                ParseRrLog("20:00:00.000 load: process 5300 attached C:\\Games\\X\\plugins\\WrongGPU-RR.asi (hook mode)\r\n"
                           "20:00:10.000 RR feature refused (the game keeps its own path): the game-interface route holds the RR feature\r\n", busy);
                require(busy.rrFound && !busy.rrUsed && busy.rrNote.empty(), "...a refusal because the game-interface route runs RR: not a failed attempt (r7 #3)");
            }
            // r6 #2: a timing line without its frame count (an older runtime's) is still frames read back: at least one, never "no
            // successful frame" beside its cost
            {
                LastRun tl;
                ParseRrLog("20:00:00.000 load: process 4400 attached C:\\Games\\X\\plugins\\WrongGPU-RR.asi (hook mode)\r\n"
                           "20:10:00.000 runtime: instance 1 timing, last 300: GPU 2.500 ms per frame (pre 0.010)\r\n", tl);
                require(tl.rrUsed && tl.rrMs == L"2.5" && tl.rrFrames == 1 && CountRrRun(tl) && tl.upscaled,
                        "a timing line without its frame count: one frame at least, a run with RR frames (r6 #2)");
            }
            // r6 #2: the runtime couldn't make the instance at all: still an attempt (the game asked for RR), its reason kept
            {
                LastRun cf;
                ParseRrLog("20:00:00.000 load: process 4300 attached C:\\Games\\X\\plugins\\WrongGPU-RR.asi (hook mode)\r\n"
                           "20:00:05.000 runtime: create: failed (D3D12 objects)\r\n", cf);
                require(cf.rrUsed && cf.rrFrames == 0 && cf.rrNote == L"the instance couldn't be made (D3D12 objects)" && CountRrRun(cf) && cf.rrOnly && cf.found &&
                            !cf.upscaled,
                        "an RR session whose instance couldn't be made: an attempt, a run without a frame, the reason kept (r6 #2)");
            }
            // r5 #2: the front end's session and the RR plugin's session are combined only when they are one game session (the same
            // process id, their times overlapping); otherwise the newer alone counts. Logs on a test clock (local times, set as the
            // launcher reads them: the files' last writes)
            auto touch = [](const wstring& p, WORD day, WORD h, WORD mi) {
                SYSTEMTIME lt{2026, 10, 0, day, h, mi, 0, 0};
                FILETIME lft{}, ft{};
                SystemTimeToFileTime(&lt, &lft);
                LocalFileTimeToFileTime(&lft, &ft);
                HANDLE fh = CreateFileW(p.c_str(), FILE_WRITE_ATTRIBUTES, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
                if (fh != INVALID_HANDLE_VALUE) { SetFileTime(fh, nullptr, nullptr, &ft); CloseHandle(fh); }
            };
            auto hh = [](int h) { char b[4]; snprintf(b, sizeof b, "%02d", h); return string(b); };
            // the front end: started at h:00, the runtime ready, no frame of its own logged as upscaled; its last line at h:20
            auto feLog = [&](int pid, int h) {
                const string p = std::to_string(pid), t = hh(h);
                return "[" + t + ":00:00.000 " + p + "] [x] ==== session start: Game.exe (process " + p + ") ====\r\n[" + t + ":00:00.100 " + p +
                       "] [x] settings: upscaler=network\r\n[" + t + ":00:05.000 " + p + "] [x]   U1 upscaler info: network runtime ready: executor dx12, sync inline\r\n[" +
                       t + ":20:00.000 " + p + "] [x] #9 context destroyed\r\n";
            };
            // the RR plugin: attached at h:mm, 36044 frames evaluated, its last line at h:20
            auto rrLog = [&](int pid, int h, int m0) {
                const string t = hh(h), m = hh(m0);
                return t + ":" + m + ":00.000 load: process " + std::to_string(pid) + " attached C:\\Games\\Example\\plugins\\WrongGPU-RR.asi (hook mode)\r\n" + t + ":" + m +
                       ":06.000 runtime: create instance 1: output 1920x1080, render 1280x720\r\n" + t +
                       ":19:59.000 runtime: instance 1 timing, 300 frames from frame 35701: GPU 2.000 ms per frame (pre 0.010), render 1280x720, output 1920x1080\r\n" + t +
                       ":20:00.500 runtime: destroy instance 1 after 36044 frames (0 failed)\r\n";
            };
            auto pair = [&](const wchar_t* name, int fePid, WORD feDay, int feH, int rrPid, WORD rrDay, int rrH, int rrM) {
                const wstring d = Join(dir, name);
                RemoveTree(d);
                Put(Join(d, L"wronggpu-fsr.log"), feLog(fePid, feH));
                Put(Join(d, L"wronggpu-rr.log"), rrLog(rrPid, rrH, rrM));
                touch(Join(d, L"wronggpu-fsr.log"), feDay, (WORD)feH, 20);
                touch(Join(d, L"wronggpu-rr.log"), rrDay, (WORD)rrH, 20);
                return ReadLastRun(d);
            };
            // the front end's session newer (another process, two hours later): the stale RR evidence is left out
            const LastRun feNew = pair(L"rr-older", 1500, 2, 20, 1400, 2, 18, 0);
            require(feNew.found && feNew.ready && !feNew.rrOnly && !feNew.rrFound && !feNew.rrUsed && feNew.rrFrames == 0 && feNew.rrMs.empty() &&
                        feNew.clock.pid == 1500 && feNew.when == L"2026-10-02 20:20",
                    "the front end's session newer than the RR plugin's (another process): the front end's run alone, no RR evidence, its time (r5 #2)");
            // the RR session newer (another process, after the front end's): an RR-only run, the front end's evidence left out
            const LastRun rrNew = pair(L"rr-newer", 1500, 2, 18, 1600, 2, 21, 0);
            require(rrNew.found && rrNew.created && rrNew.upscaled && rrNew.rrOnly && !rrNew.ready && rrNew.upscaler.empty() && rrNew.executor.empty() &&
                        rrNew.rrFrames == 36044 && rrNew.rrClock.pid == 1600 && rrNew.when == L"2026-10-02 21:20" &&
                        rrNew.Summary().rfind(L"Ray Reconstruction (2.0 ms a frame)", 0) == 0,
                    "the RR plugin's session newer than the front end's (another process): an RR-only run, its time; nothing of the older front end (r5 #2)");
            // r6 #3: Open log opens the log of the run Last run shows: the RR log for the newer RR-only session, the front end's otherwise
            require(LastRunLog(Join(dir, L"rr-newer"), rrNew) == Join(Join(dir, L"rr-newer"), L"wronggpu-rr.log") &&
                        LastRunLog(Join(dir, L"rr-older"), feNew) == Join(Join(dir, L"rr-older"), L"wronggpu-fsr.log"),
                    "LastRunLog: the RR plugin's log for an RR-only run beside an older front-end log; the front end's for its own newer run (r6 #3)");
            // ...the same with an RR session that evaluated nothing: no RR-only run; the front end's older run stays, without RR
            {
                const wstring noneDir = Join(dir, L"rr-newer-none");
                RemoveTree(noneDir);
                Put(Join(noneDir, L"wronggpu-fsr.log"), feLog(1500, 18));
                Put(Join(noneDir, L"wronggpu-rr.log"), "21:00:00.000 load: process 1600 attached C:\\Games\\Example\\plugins\\WrongGPU-RR.asi (hook mode)\r\n"
                                                 "21:20:00.000 diag: summary (process exit): support queries 6 (RR: yes 0 / no 1), RR lookups 0, RR instances created 0, RR evaluates 0\r\n");
                touch(Join(noneDir, L"wronggpu-fsr.log"), 2, 18, 20);
                touch(Join(noneDir, L"wronggpu-rr.log"), 2, 21, 20);
                const LastRun noneRun = ReadLastRun(noneDir);
                require(noneRun.found && noneRun.ready && !noneRun.rrOnly && !noneRun.rrFound && noneRun.when == L"2026-10-02 18:20",
                        "a newer RR session that evaluated nothing: the front end's run, without that session's RR lines (r5 #2)");
            }
            // r6 #2: a newer RR attempt that failed (another process, after the front end's successful run): that attempt is the last
            // run, never the older success ("Worked last time you played" would be about another launch)
            {
                string failedLog;
                require(ReadAll(Join(FromUtf8(WG_TEST_LOGS), L"rr-failed-only.log"), failedLog), "the failed RR fixture is there");
                const wstring failDir = Join(dir, L"rr-newer-failed");
                RemoveTree(failDir);
                Put(Join(failDir, L"wronggpu-fsr.log"), feLog(1500, 18) + "[18:20:01.000 1500] [fe] #12 WrongGPU frame: R=1280x720 O=1920x1080\r\n"
                                                                    "[18:20:01.000 1500] [fe] #12 -> OK (WrongGPU)\r\n");
                Put(Join(failDir, L"wronggpu-rr.log"), failedLog);  // (process 4100, 20:00 to 20:01)
                touch(Join(failDir, L"wronggpu-fsr.log"), 2, 18, 21);
                touch(Join(failDir, L"wronggpu-rr.log"), 2, 20, 2);
                const LastRun failRun = ReadLastRun(failDir);
                require(failRun.found && failRun.rrOnly && !failRun.upscaled && !failRun.AnyUpscaled() && !failRun.ready && failRun.upscaler.empty() && failRun.rrUsed && failRun.rrFrames == 0 &&
                            failRun.rrClock.pid == 4100 && failRun.when == L"2026-10-02 20:02" && !failRun.rrNote.empty(),
                        "a newer RR attempt that failed after the front end's successful run: the failed attempt is the last run (no older success, "
                        "no Worked), its time and reason (r6 #2)");
            }
            // r7 #3: the direct route's refusal newer than the front end's successful run (another process, later): the refused attempt is
            // the last run, never the older success; and inside one game session where Model M upscaled: Model M's run, RR's attempt
            // and its reason kept for the page
            {
                string refused;
                require(ReadAll(Join(FromUtf8(WG_TEST_LOGS), L"rr-refused-only.log"), refused), "the refused RR fixture is there");
                const wstring refDir = Join(dir, L"rr-newer-refused");
                RemoveTree(refDir);
                Put(Join(refDir, L"wronggpu-fsr.log"), feLog(1500, 18) + "[18:20:01.000 1500] [fe] #12 WrongGPU frame: R=1280x720 O=1920x1080\r\n"
                                                                    "[18:20:01.000 1500] [fe] #12 -> OK (WrongGPU)\r\n");
                Put(Join(refDir, L"wronggpu-rr.log"), refused);  // (process 5200, 19:30 to 19:35)
                touch(Join(refDir, L"wronggpu-fsr.log"), 2, 18, 21);
                touch(Join(refDir, L"wronggpu-rr.log"), 2, 19, 36);
                const LastRun refRun = ReadLastRun(refDir);
                require(refRun.found && refRun.rrOnly && !refRun.upscaled && !refRun.AnyUpscaled() && !refRun.ready && refRun.rrUsed && refRun.rrFrames == 0 && refRun.rrClock.pid == 5200 &&
                            refRun.when == L"2026-10-02 19:36" && refRun.rrNote.find(L"refused") != wstring::npos,
                        "the direct route's refusal newer than the front end's successful run: the refusal is the last run (no older success, no "
                        "Worked), its time and reason (r7 #3)");
                const wstring d2 = Join(dir, L"rr-same-refused");
                RemoveTree(d2);
                Put(Join(d2, L"wronggpu-fsr.log"), feLog(1500, 20) + "[20:20:01.000 1500] [fe] #12 WrongGPU frame: R=1280x720 O=1920x1080\r\n"
                                                                     "[20:20:01.000 1500] [fe] #12 -> OK (WrongGPU)\r\n");
                Put(Join(d2, L"wronggpu-rr.log"), "20:00:00.000 load: process 1500 attached C:\\Games\\Example\\plugins\\WrongGPU-RR.asi (hook mode)\r\n"
                                                  "20:00:06.000 RR feature refused (the game keeps its own path): render or output size missing (0x0 -> 1920x1080)\r\n"
                                                  "20:20:00.500 diag: summary (process exit): direct route: RR creates 1 (created 0), evaluates 0\r\n");
                touch(Join(d2, L"wronggpu-fsr.log"), 2, 20, 21);
                touch(Join(d2, L"wronggpu-rr.log"), 2, 20, 21);
                const LastRun refSame = ReadLastRun(d2);
                require(refSame.found && refSame.ready && refSame.upscaled && !refSame.rrOnly && !refSame.rrUpscaled && refSame.rrUsed && refSame.rrFrames == 0 &&
                            refSame.rrNote == L"the game's request was refused (render or output size missing (0x0 -> 1920x1080))",
                        "one session, Model M upscaled, the direct route refused RR: Model M's run, RR's attempt and its reason kept (r7 #3)");
            }
            // one game session (the same process id, the same launch): combined, not RR-only
            const LastRun mix = pair(L"rr-same", 1500, 2, 20, 1500, 2, 20, 0);
            require(mix.found && mix.ready && mix.created && !mix.rrOnly && mix.rrFound && mix.rrUsed && mix.rrFrames == 36044 && !mix.upscaled &&
                        mix.rrUpscaled && mix.AnyUpscaled() && mix.when == L"2026-10-02 20:20" && mix.Summary().rfind(L"Ray Reconstruction (2.0 ms a frame)", 0) == 0,
                    "one game session in both logs: combined (not RR-only), the RR frames there; Model M's late start stays without a frame of its "
                    "own (r5 #2, r5 #3 case B); RR did the upscaling there (r6 #1: rrUpscaled, the summary says Ray Reconstruction)");
            // r5 #3: in one game session the front end's own evidence stays as it is; RR's frames never stand for Model M's
            {
                const string rrSame = rrLog(1500, 20, 0);
                auto sameRun = [&](const wchar_t* name, const string& fe) {
                    const wstring d = Join(dir, name);
                    RemoveTree(d);
                    Put(Join(d, L"wronggpu-fsr.log"), fe);
                    Put(Join(d, L"wronggpu-rr.log"), rrSame);
                    touch(Join(d, L"wronggpu-fsr.log"), 2, 20, 20);
                    touch(Join(d, L"wronggpu-rr.log"), 2, 20, 20);
                    return ReadLastRun(d);
                };
                // case A: the front end set to Model M, the game asked for no upscaler (no create), RR ran: Model M "not used"
                const LastRun a = sameRun(L"rr-same-unused", "[20:00:00.000 1500] [x] ==== session start: Game.exe (process 1500) ====\r\n"
                                                             "[20:00:00.100 1500] [x] settings: upscaler=network\r\n[20:20:00.000 1500] [x] exit\r\n");
                require(a.found && !a.created && !a.ready && !a.upscaled && !a.rrOnly && a.rrFrames == 36044 && a.upscaler == L"network" &&
                            a.rrUpscaled && a.AnyUpscaled() && a.Summary().rfind(L"Ray Reconstruction (2.0 ms a frame)", 0) == 0,
                        "one session, the game asked Model M for nothing, RR ran: not created, not upscaled (r5 #3 case A); RR did the upscaling "
                        "(r6 #1: rrUpscaled, the summary says Ray Reconstruction, not WrongGPU not used)");
                // ...and a frame Model M upscaled: upscaled, as the front end logged it
                const LastRun ok = sameRun(L"rr-same-framed", feLog(1500, 20) + "[20:20:01.000 1500] [fe] #12 WrongGPU frame: R=1280x720 O=1920x1080\r\n"
                                                                                "[20:20:01.000 1500] [fe] #12 -> OK (WrongGPU)\r\n");
                require(ok.found && ok.ready && ok.upscaled && !ok.rrOnly && !ok.rrUpscaled && ok.rrFrames == 36044 && ok.renderSize == L"1280x720",
                        "one session, Model M upscaled a frame, RR ran: upscaled, the front end's size (r5 #3); not RR's upscaling (r6 #1)");
                // r6 #2: one session where Model M upscaled and RR was attempted without a successful frame: Model M's run, RR's attempt
                // kept for the Denoiser row (no frames, its reason), not RR's upscaling
                {
                    const wstring rfDir = Join(dir, L"rr-same-rrfailed");
                    RemoveTree(rfDir);
                    Put(Join(rfDir, L"wronggpu-fsr.log"), feLog(1500, 20) + "[20:20:01.000 1500] [fe] #12 WrongGPU frame: R=1280x720 O=1920x1080\r\n"
                                                                        "[20:20:01.000 1500] [fe] #12 -> OK (WrongGPU)\r\n");
                    Put(Join(rfDir, L"wronggpu-rr.log"), "20:00:00.000 load: process 1500 attached C:\\Games\\Example\\plugins\\WrongGPU-RR.asi (hook mode)\r\n"
                                                     "20:00:06.000 runtime: create instance 1: output 1920x1080, render 1280x720\r\n"
                                                     "20:00:06.100 runtime: instance 1 evaluate failed before any pass: runtime not ready\r\n"
                                                     "20:20:00.500 runtime: destroy instance 1 after 20 frames (20 failed)\r\n");
                    touch(Join(rfDir, L"wronggpu-fsr.log"), 2, 20, 21);
                    touch(Join(rfDir, L"wronggpu-rr.log"), 2, 20, 21);
                    const LastRun rf = ReadLastRun(rfDir);
                    require(rf.found && rf.ready && rf.upscaled && !rf.rrOnly && !rf.rrUpscaled && rf.rrUsed && rf.rrFrames == 0 && rf.rrNote == L"runtime not ready",
                            "one session, Model M upscaled, RR attempted without a frame: Model M's run, RR's attempt and reason kept, not RR's upscaling (r6 #2)");
                    // r7 #1: the game asked Model M for nothing in that session (RR instead): the summary says RR's story, not "not used"
                    Put(Join(rfDir, L"wronggpu-fsr.log"), "[20:00:00.000 1500] [x] ==== session start: Game.exe (process 1500) ====\r\n"
                                                      "[20:00:00.100 1500] [x] settings: upscaler=network\r\n[20:20:00.000 1500] [x] exit\r\n");
                    touch(Join(rfDir, L"wronggpu-fsr.log"), 2, 20, 21);
                    const LastRun ru = ReadLastRun(rfDir);
                    require(ru.found && !ru.created && !ru.rrOnly && !ru.rrUpscaled && ru.rrUsed && ru.rrFrames == 0 &&
                                ru.Summary().rfind(L"Ray Reconstruction: no successful frame", 0) == 0 && ru.Summary().find(L"not used") == wstring::npos,
                            "one session, Model M asked for nothing, RR attempted without a frame: the summary says no successful RR frame, not "
                            "WrongGPU not used (r7 #1)");
                }
                // r6 #1: Model M failed to start (created, never ready) or was turned off (AMD FSR): their own story stays, RR's frames
                // don't make the run RR's upscaling
                const LastRun ns = sameRun(L"rr-same-nostart", "[20:00:00.000 1500] [x] ==== session start: Game.exe (process 1500) ====\r\n"
                                                              "[20:00:00.100 1500] [x] settings: upscaler=network\r\n"
                                                              "[20:00:02.000 1500] [x] #1 ffxCreateContext upscale\r\n[20:20:00.000 1500] [x] exit\r\n");
                require(ns.found && ns.created && !ns.ready && !ns.upscaled && !ns.rrUpscaled && !ns.AnyUpscaled() && ns.rrFrames == 36044,
                        "one session, Model M created but never ready, RR ran: not RR's upscaling, Model M didn't start stays (r6 #1)");
                const LastRun off = sameRun(L"rr-same-amd", "[20:00:00.000 1500] [x] ==== session start: Game.exe (process 1500) ====\r\n"
                                                           "[20:00:00.100 1500] [x] settings: upscaler=amd\r\n[20:20:00.000 1500] [x] exit\r\n");
                require(off.found && off.upscaler == L"amd" && !off.rrUpscaled && !off.AnyUpscaled() && off.rrFrames == 36044,
                        "one session, Use WrongGPU off (AMD FSR), RR ran: not RR's upscaling (r6 #1)");
                // the parse alone (ui-dump and ui-render use it so): CountRrRun keeps the front end's created and upscaled
                LastRun pa = ParseLastRun("[20:00:00.000 1500] [x] ==== session start: Game.exe (process 1500) ====\r\n[20:00:00.100 1500] [x] settings: upscaler=network\r\n");
                ParseRrLog(rrSame, pa);
                require(CountRrRun(pa) && pa.found && !pa.created && !pa.upscaled && !pa.rrOnly && pa.rrUpscaled,
                        "CountRrRun with a front-end session: its evidence unchanged (r5 #3); RR's upscaling where Model M upscaled none (r6 #1)");
            }
            // the same process id a day apart (Windows gives ids out again): not one session
            const LastRun reused = pair(L"rr-pid-again", 1500, 2, 20, 1500, 1, 20, 0);
            require(reused.found && reused.ready && !reused.rrFound && !reused.rrOnly && reused.when == L"2026-10-02 20:20",
                    "the same process id on another day: two sessions, the newer (the front end's) alone (r5 #2)");
            // JoinRrSession's rules: an id missing (the RR plugin's "attached" line scrolled out of the tail): the times decide
            SessionSpan f{1500, 1000, 2000}, rs{0, 500, 2100};
            require(JoinRrSession(f, rs) == RrJoin::Same, "no RR process id, the times overlapping: one session (r5 #2)");
            rs = {0, 2200, 3000};
            require(JoinRrSession(f, rs) == RrJoin::RrNewer, "no RR process id, the RR session after the front end's: the RR one newer (r5 #2)");
            rs = {0, 100, 800};
            require(JoinRrSession(f, rs) == RrJoin::FrontNewer, "no RR process id, the RR session before the front end's: the front end newer (r5 #2)");
            rs = {1600, 900, 2100};
            require(JoinRrSession(f, rs) == RrJoin::FrontNewer, "two process ids, overlapping times: not one session; the later start newer (r5 #2)");
            // logs copied later (both written the next day at 03:28, as build-ui/reallogs): the times of day in them place the
            // sessions the day before, and the RR session that started after the front end's is the newer
            {
                SYSTEMTIME lt{2026, 11, 0, 5, 3, 28, 41, 0};
                FILETIME lft{}, ft{};
                SystemTimeToFileTime(&lt, &lft);
                LocalFileTimeToFileTime(&lft, &ft);
                const LastRun::Clock fc{1000, 19 * 3600 + 25 * 60 + 6, 19 * 3600 + 36 * 60 + 15}, rc{1001, 19 * 3600 + 39 * 60 + 10, 19 * 3600 + 48 * 60 + 12};
                const SessionSpan fs = SpanOf(fc, ft), rs2 = SpanOf(rc, ft);
                require(fs.end - fs.start == 11 * 60 + 9 && rs2.start - fs.start == 14 * 60 + 4 && JoinRrSession(fs, rs2) == RrJoin::RrNewer,
                        "copied logs: the sessions placed the day before by their times of day; the later RR session newer (r5 #2)");
            }
        }
        // a session where the game never asked for an upscaler (TAA in its settings): WrongGPU wasn't used, nothing failed (UI review #9)
        const LastRun unused = ParseLastRun("[1] [x] ==== session start: Cyberpunk2077.exe (process 1) ====\r\n[1] [x] settings: upscaler=network\r\n");
        require(unused.found && !unused.created && !unused.ready && unused.Summary().find(L"not used") != wstring::npos, "a run without an upscaler: not used");
        // a create, a frame refused or sent to AMD FSR, or the runtime ready: the game asked for one (a start that failed is not "not used")
        require(ready.created && failed.created && ParseLastRun(startup).created, "a run with a create, a refused frame or the runtime ready: the game asked for an upscaler");

        // ---- log warnings ----------------------------------------------------------------------------------------------------
        const string fsrLog =
            "[10:00:00.000 1] [x] ==== session start: Hogwarts.exe (process 1) ====\r\n"
            "[10:00:00.100 1] [x] settings: upscaler=network\r\n"
            "[10:00:01.000 1] [info] network runtime ready: executor dx12, sync inline\r\n"
            "[10:00:02.000 1] [hooks] own table: list 0000 records into its own vtable 0001 (a), which WrongGPU does not take over: ReShade\r\n"
            "[10:00:02.100 1] [hooks] own table: list 0002 records into its own vtable 0003 (a), which WrongGPU does not take over: ReShade\r\n"
            "[10:00:03.000 1] [fe] [up] WrongGPU frame: 1920x1080 notes: no depth; no motion vectors\r\n";
        const LastRun lr = ParseLastRun(fsrLog);
        require(lr.warnings.size() == 2 && lr.warnings[0].kind == L"foreign-vtable" && lr.warnings[1].kind == L"no-depth", "foreign tables and no depth: one warning each");
        LastRun pl;
        ParseAsiLog("[1] [WrongGPU.asi] WrongGPU.asi loaded\r\n[2] [WrongGPU.asi] hooked: 2 of 2 Selectable call sites patched\r\n"
                    "[3] [WrongGPU.asi] WrongGPU.asi loaded\r\n[4] [WrongGPU.asi] not hooked: OptiScaler's DX12 upscaler menu was not recognised in 3 candidate module(s)\r\n", pl);
        require(pl.plugin == L"not-hooked" && pl.warnings.size() == 1 && pl.warnings[0].kind == L"plugin-not-hooked", "the plugin's last load: not hooked");
        LastRun ok;
        ParseAsiLog("[1] [WrongGPU.asi] WrongGPU.asi loaded\r\n[2] [WrongGPU.asi] hooked: 2 of 2 Selectable call sites patched\r\n", ok);
        require(ok.plugin == L"hooked" && ok.warnings.empty(), "the plugin hooked: no warning");
        require(OptiLogMentionsXess("[info] Inputs: XeSS (libxess) init") && !OptiLogMentionsXess("[info] Inputs: DLSS"), "OptiScaler.log naming XeSS");
        const wstring lg = Join(dir, L"logs");
        Put(Join(lg, L"wronggpu-fsr.log"), fsrLog);
        Put(Join(lg, L"OptiScaler.log"), "Inputs: XeSS");
        Put(Join(Join(lg, kAsiPluginOwnDir), L"wronggpu-asi.log"), "[4] [WrongGPU.asi] not hooked: no Selectable call site could be patched\r\n");
        const LastRun all = ReadLastRun(lg);
        bool xessWarn = false, plug = false;
        for (auto& w : all.warnings) { xessWarn |= w.kind == L"xess-no-depth"; plug |= w.kind == L"plugin-not-hooked"; }
        require(all.found && xessWarn && plug && all.plugin == L"not-hooked", "ReadLastRun: XeSS without depth (OptiScaler.log), the plugin's log in its own folder");

        RemoveTree(dir);
        std::printf("coexist-core: %d checks passed\n", checks);
        return 0;
    } catch (const std::exception& e) {
        std::printf("coexist-core FAILED after %d checks: %s\n", checks, e.what());
        return 1;
    } catch (const SetupError& e) {
        std::printf("coexist-core FAILED after %d checks: %ls\n", checks, e.msg.c_str());
        return 1;
    }
}
