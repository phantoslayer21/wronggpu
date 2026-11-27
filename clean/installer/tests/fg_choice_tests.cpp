// SPDX-License-Identifier: GPL-3.0-only
// The frame generation choice of the launcher (#96, wg_core/fg.h; profile.cpp built with WRONGGPU_FG on, whatever the launcher
// build's switch): the four settings fg, fg_preset, fg_multiplier and fg_presenter in their exact spellings and defaults (off,
// A, 2, auto), profile.ini round trips beside the upscaler, RR and other keys, the job file, the global defaults, and the keys
// a game's settings.ini gets: none for a game never opted in; fg off there while frame generation's data is not ready. A game
// that never opts in sees no change anywhere. Synthetic folders and values only. Written for this project. Run by ctest
// ("fg-choice").
#include <iostream>
#include <stdexcept>

#include "wg_core/launcher.h"
#include "wg_core/profile.h"

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
    wstring dir = Join(tmp, L"fg-choice-tests-" + std::to_wstring(GetCurrentProcessId()));
    RemoveTree(dir);
    EnsureDir(dir);
    return dir;
}
Profile With(std::initializer_list<std::pair<const char*, const char*>> kv) {
    Profile p = DefaultProfile();
    for (auto& x : kv) {
        wstring why;
        if (!SetSetting(p, x.first, x.second, &why)) throw std::runtime_error("test setup: " + string(x.first));
    }
    return p;
}
}  // namespace

int main() {
    try {
        // ---- the settings: four keys, off by default (`set`, and the game page's Frame Generation section) -------------
        const char* const defs[][2] = {{"fg", "off"}, {"fg_preset", "A"}, {"fg_multiplier", "auto"}, {"fg_presenter", "auto"}};
        for (auto& kd : defs) {
            const SettingDef* d = FindSetting(kd[0]);
            require(d && d->layer == Layer::Fg && d->tier == Tier::Basic && string(d->def) == kd[1] && string(d->values) == FgValues(kd[0]),
                    "an FG setting: Layer::Fg, Basic (on the game page since 0.7.0), its default and spellings (fg.h)");
        }
        const Profile def = DefaultProfile();
        require(def.get("fg") == "off" && def.get("fg_preset") == "A" && def.get("fg_multiplier") == "auto" && def.get("fg_presenter") == "auto" &&
                    def.isDefault("fg") && !FgOptedIn(def),
                "the default profile: frame generation off, A, 2, auto (not opted in)");

        // ---- set: exact spellings; a refusal names the allowed ones and changes nothing ---------------------------------
        Profile p = DefaultProfile();
        wstring why;
        require(SetSetting(p, "fg", "on", &why) && p.get("fg") == "on" && SetSetting(p, "FG", " off ", &why) && p.get("fg") == "off",
                "set fg=on / off (the key in either case, as every key; spaces around the value dropped)");
        require(SetSetting(p, "fg_preset", "B", &why) && SetSetting(p, "fg_multiplier", "5", &why) && SetSetting(p, "fg_presenter", "dxgi_vsync", &why) &&
                    p.get("fg_preset") == "B" && p.get("fg_multiplier") == "5" && p.get("fg_presenter") == "dxgi_vsync",
                "set fg_preset=B fg_multiplier=5 fg_presenter=dxgi_vsync");
        const Profile kept = p;
        const char* const bad[][2] = {{"fg", "On"}, {"fg", "ON"}, {"fg", "1"}, {"fg", "true"}, {"fg_preset", "b"}, {"fg_preset", "D"},
                                      {"fg_multiplier", "1"}, {"fg_multiplier", "7"}, {"fg_multiplier", "02"}, {"fg_multiplier", "+3"},
                                      {"fg_presenter", "Auto"}, {"fg_presenter", "vsync"}, {"fg_presenter", "DXGI_TEARING"}};
        for (auto& b : bad) {
            require(!SetSetting(p, b[0], b[1], &why) && why.find(L"allowed, exactly as written") != wstring::npos, "a bad spelling is refused, the spellings named");
            require(p.v == kept.v, "a refused value changes nothing");
        }
        require(!SetSetting(p, "fg_preset", "c", &why) && why == L"\"c\" is not a value of fg_preset (allowed, exactly as written: A, B, C)",
                "set fg_preset=c: refused in words (no case folding)");
        require(!SetSetting(p, "fg_presenter", "flip", &why) && why.find(L"auto, hwfq, dxgi_tearing, dxgi_vsync") != wstring::npos, "fg_presenter's spellings named");
        require(!SetSetting(p, "fg_mode", "on", &why) && why.find(L"unknown setting") != wstring::npos, "a key that is not one of the four: unknown");

        // ---- profile.ini: a game never opted in has no FG line; the chosen values round trip beside the other keys -------
        require(ProfileText(def).find("fg") == string::npos, "profile.ini of a game never opted in: no FG line (as before the choice)");
        const Profile chosen = With({{"fg", "on"}, {"fg_preset", "C"}, {"pre_exposure", "original"}, {"rr_mode", "exact"}, {"rr_enabled", "off"}});
        const string ct = ProfileText(chosen);
        require(ct.find("fg=on\r\n") != string::npos && ct.find("fg_preset=C\r\n") != string::npos && ct.find("fg_multiplier") == string::npos &&
                    ct.find("fg_presenter") == string::npos,
                "profile.ini: the FG values chosen written; the ones at their defaults not");
        const wstring dir = TempFolder();
        const wstring prof = Join(dir, L"profile.ini");
        Profile withUnknown = chosen;
        withUnknown.v["later_key"] = "kept";
        SaveProfile(prof, withUnknown);
        Profile back;
        std::vector<wstring> problems;
        require(LoadProfile(prof, back, &problems) && problems.empty() && back.v == withUnknown.v,
                "profile.ini round trip: the FG keys, the SR / RR keys and an unknown key all read back as written");
        const string edited = "[profile]\r\nformat=1\r\nfg=ON\r\nfg_preset=b\r\nfg_multiplier=4\r\nfg_presenter=hwfq\r\npre_exposure=original\r\nrr_mode=accurate\r\n";
        WriteFileAtomic(prof, edited.data(), edited.size());
        problems.clear();
        require(LoadProfile(prof, back, &problems) && problems.size() == 2 && back.get("fg") == "off" && back.get("fg_preset") == "A" &&
                    back.get("fg_multiplier") == "4" && back.get("fg_presenter") == "hwfq" && back.get("pre_exposure") == "original" && back.get("rr_mode") == "accurate" &&
                    problems[0].find(L"\"ON\" is not a value of fg") != wstring::npos && problems[0].find(L"the default is used") != wstring::npos,
                "profile.ini with bad FG spellings: those keep their defaults (each listed); the valid FG and other keys are read");

        // ---- the game's files the launcher writes from the profile: frame generation is never in wronggpu.ini or OptiScaler.ini
        const string wini = "# WrongGPU settings\r\n[fsr]\r\nupscaler = network\r\nruntime_options = model=M\r\n";
        string out;
        require(RenderWronggpuIni(chosen, wini, out, &why), "the profile with frame generation renders");
        string winiDef;
        require(RenderWronggpuIni(With({{"pre_exposure", "original"}, {"rr_mode", "exact"}, {"rr_enabled", "off"}}), wini, winiDef, &why) && out == winiDef &&
                    out.find("fg") == string::npos,
                "wronggpu.ini: the same with and without frame generation (no FG key in it)");
        const string oini = "[FSR]\r\nUpscalerIndex=0\r\n[Menu]\r\nShowFps=auto\r\n";
        require(RenderOptiIni(chosen, oini, out) && out == oini, "OptiScaler.ini: unchanged by frame generation");
        for (auto& kv : FrontEndStateKeys(chosen)) require(!IsFgKey(kv.first), "the front end's state keys hold no FG key");
        require(ImportProfile(wini, oini).v == def.v && OutsideEdits(chosen, wini, oini).empty(),
                "an import never invents an FG value; no outside edit is reported for frame generation");
        require(EffectiveSettings(wini, oini)["fg"] == "off", "wronggpu.ini says nothing about frame generation: off");

        // ---- settings.ini's FG keys from the profile (the launcher's apply) ----------------------------------------------
        require(FgStateKeys(def, {{"upscaler", "network"}, {"rr_mode", "fast"}}, true).empty(), "a game never opted in: no FG key for settings.ini");
        const Profile on = With({{"fg", "on"}, {"fg_multiplier", "3"}});
        auto k = FgStateKeys(on, {}, true);
        require(k == std::vector<std::pair<string, string>>{{"fg", "on"}, {"fg_preset", "A"}, {"fg_multiplier", "3"}, {"fg_presenter", "auto"}},
                "fg=on, data ready: the four keys");
        k = FgStateKeys(on, {}, false);
        require(k.size() == 4 && k[0].second == "off" && k[2].second == "3" && on.get("fg") == "on",
                "fg=on, data not ready: settings.ini says off; the profile keeps the choice");
        k = FgStateKeys(def, {{"fg", "on"}}, true);
        require(k.size() == 4 && k[0].second == "off", "back to the defaults while settings.ini says on: off written down");

        // ---- the elevated worker's job file and the global defaults --------------------------------------------------
        OpRequest job;
        job.op = Op::Apply;
        job.dir = L"C:\\Games\\Synthetic";
        job.hasProfile = true;
        job.profile = With({{"fg", "on"}, {"fg_preset", "B"}, {"fg_multiplier", "6"}, {"fg_presenter", "dxgi_tearing"}, {"rr_mode", "exact"}});
        OpRequest jb;
        wstring lad, lf;
        require(ParseJob(JobText(job, L"C:\\Users\\p\\AppData\\Local", L""), jb, lad, lf, &why) && jb.hasProfile && jb.profile.v == job.profile.v,
                "job file: the FG keys round trip with the rest of the profile");
        require(!ParseJob("op=apply\r\ndir=C:\\g\r\nlocalappdata=C:\\x\r\nprofile.fg_preset=b\r\n", jb, lad, lf, &why), "job file: a bad FG spelling is refused");
        GlobalSettings gs;
        require(SetGlobal(gs, "defaults.fg_multiplier", "4", &why) && gs.defaults.get("fg_multiplier") == "4" && gs.defaults.get("fg") == "off" &&
                    !SetGlobal(gs, "defaults.fg", "yes", &why),
                "global defaults: an FG key by its exact spelling; frame generation itself stays off");
        RemoveTree(dir);
    } catch (const std::exception& e) {
        std::cerr << "fg-choice FAILED after " << checks << " checks: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "fg-choice: " << checks << " checks passed" << std::endl;
    return 0;
}
