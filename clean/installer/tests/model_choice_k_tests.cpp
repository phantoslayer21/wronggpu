// SPDX-License-Identifier: GPL-3.0-only
// The Model K choice of the launcher (profile.cpp built with WRONGGPU_MODEL_K on, whatever the launcher build's switch;
// CMake builds this file twice: with the Model L choice too, and without it). The model setting accepts k, profile.ini
// says model=k, wronggpu.ini's model item is model=K (last, every other model item out) and OptiScaler's UpscalerIndex
// names Model K's entry. Model M changes nothing in either file, byte for byte; Model L (when built in) behaves as in
// model_choice_tests.cpp. Written for this project. Run by ctest ("model-choice-k", "model-choice-k-only").
#include <iostream>
#include <stdexcept>

#include "wg_core/profile.h"

using namespace wg;

namespace {
int checks = 0;
void require(bool ok, const char* what) {
    if (!ok) throw std::runtime_error(what);
    ++checks;
}
Profile With(const char* model) {
    Profile p = DefaultProfile();
    wstring why;
    if (!SetSetting(p, "model", model, &why)) throw std::runtime_error("test setup: model");
    return p;
}
string Wini(const Profile& p, const string& in) {
    string out;
    wstring why;
    if (!RenderWronggpuIni(p, in, out, &why)) throw std::runtime_error("test setup: render");
    return out;
}
string Oini(const Profile& p, const string& in, std::vector<wstring>* notes = nullptr) {
    string out;
    RenderOptiIni(p, in, out, notes);
    return out;
}
}  // namespace

int main() {
    try {
        const bool withL = WRONGGPU_MODEL_L != 0;
        const string K = kModelKEntryIndex, L = kModelLEntryIndex;
        require(K == "1" && L == "2", "Model K's entry is 1 (after Model M's), Model L's is 2 (after Model K's)");

        // ---- the setting ---------------------------------------------------------------------------------------------
        const SettingDef* d = FindSetting("model");
        require(d && d->layer == Layer::Model && string(d->def) == "m", "the model setting: Model M by default");
        require(string(d->values) == (withL ? "m|k|l" : "m|k"), "the values: m|k, and l with the Model L choice");
        Profile p = DefaultProfile();
        wstring why;
        require(SetSetting(p, "model", "K", &why) && p.get("model") == "k", "model=K accepted (either case)");
        require(SetSetting(p, "model", " k ", &why) && p.get("model") == "k", "model=k with spaces accepted");
        require(!SetSetting(p, "model", "Q", &why) && p.get("model") == "k" && why.find(withL ? L"allowed: m, k, l" : L"allowed: m, k") != wstring::npos,
                "a bad value is refused with the allowed ones named; the choice stays");
        if (withL) {
            require(SetSetting(p, "model", "L", &why) && p.get("model") == "l", "model=L still accepted (the Model L control)");
        } else {
            require(!SetSetting(p, "model", "L", &why) && why.find(L"allowed: m, k") != wstring::npos, "without the Model L choice, model=L is not a value");
        }
        require(SetSetting(p, "model", "M", &why) && p.get("model") == "m" && p.isDefault("model"), "model=M accepted, the default again");

        // ---- profile.ini: Model M as before the choice existed; K written and read back -----------------------------------
        require(ProfileText(DefaultProfile()).find("model=") == string::npos, "profile.ini of a Model M game: no model line (as before)");
        require(ProfileText(With("k")).find("model=k\r\n") != string::npos, "profile.ini of a Model K game: model=k");

        // ---- wronggpu.ini --------------------------------------------------------------------------------------------
        const string fresh = "# WrongGPU settings\r\n[fsr]\r\nupscaler = network\r\nruntime_dll = wronggpu_runtime.dll\r\nruntime_options = model=M ; the model\r\n";
        require(Wini(DefaultProfile(), fresh) == fresh, "Model M: the installer's wronggpu.ini unchanged, byte for byte");
        const string toK = Wini(With("k"), fresh);
        require(toK.find("runtime_options=model=K") != string::npos && toK.find("model=M") == string::npos &&
                    toK.find("upscaler = network") != string::npos && toK.find("# WrongGPU settings") != string::npos,
                "Model K: the model item is model=K; the other lines stay");
        require(Wini(With("k"), toK) == toK, "Model K again: unchanged (the line is rewritten only when its items change)");
        const string mixed = "[fsr]\r\nruntime_options = model=M;timing_log=1;model=L\r\n";
        require(Wini(With("k"), mixed).find("runtime_options=timing_log=1;model=K\r\n") != string::npos,
                "Model K: every model item out (an L item too), model=K last (the last one wins); timing_log kept");
        const string backM = Wini(DefaultProfile(), "[fsr]\r\nruntime_options = timing_log=1;model=K\r\n");
        require(backM.find("runtime_options=timing_log=1;model=M\r\n") != string::npos, "back to Model M: Model K's item becomes model=M");
        require(Wini(DefaultProfile(), toK) == fresh.substr(0, fresh.find("runtime_options")) + "runtime_options=model=M\r\n",
                "K then M: only the runtime_options line is rewritten (the installer's comment on it is gone as it is for L)");
        const string withLItem = "[fsr]\r\nruntime_options = model=L\r\n";
        if (withL) {
            require(Wini(DefaultProfile(), withLItem).find("model=M") != string::npos, "Model M control: Model L's item still goes back to model=M");
            require(Wini(With("l"), toK).find("runtime_options=model=L") != string::npos && Wini(With("l"), toK).find("model=K") == string::npos,
                    "Model L control: K to L, the K item is out, model=L is last");
            require(Wini(With("k"), Wini(With("l"), toK)).find("runtime_options=model=K") != string::npos, "L to K again: model=K");
        } else {
            require(Wini(DefaultProfile(), withLItem) == withLItem, "without the Model L choice, Model L's item is not managed: it stays");
        }
        require(EffectiveSettings(toK, "")["model"] == "k" && EffectiveSettings(fresh, "")["model"] == "m",
                "the file's model: K from its last model item, M otherwise");
        require(EffectiveSettings("[fsr]\r\nruntime_options = model=K;model=M\r\n", "")["model"] == "m", "the file's model: the last model item counts");
        // ---- the model choice record's inputs (launcher.cpp ApplyProfile)
        require(ModelOfWronggpuIni(toK) == 'K', "ModelOfWronggpuIni: model=K");
        {
            string v;
            const string toK = SetOptiModelEntry("[WrongGPU]\r\nUpscalerIndex=5\r\n", 'K');
            require(IniHasKey(toK, "WrongGPU", "UpscalerIndex", &v) && v == K, "SetOptiModelEntry: AMD's value to K's entry (1)");
            const string toL = SetOptiModelEntry("[FSR]\r\nUpscalerIndex=0\r\n", 'L');
            require(IniHasKey(toL, "FSR", "UpscalerIndex", &v) && v == (withL ? L : string("0")),
                    "SetOptiModelEntry: L's entry (2) only with the L choice (else Model M's, 0)");
        }
        require(EffectiveSettings(withLItem, "")["model"] == (withL ? "l" : "m"), "the file's model: L only with the Model L choice");
        // the profile a game's files imply (no profile.ini): K when the files say K (an install that started on Preset K)
        require(ImportProfile(toK, "").get("model") == "k" && ImportProfile(fresh, "").get("model") == "m", "the imported profile: model=k from model=K");
        require(OutsideEdits(With("k"), toK, "").empty() && OutsideEdits(DefaultProfile(), fresh, "").empty(),
                "files that say what the profile says (K and K, M and M) are no outside edit");
        require(OutsideEdits(DefaultProfile(), toK, "").size() == 1 && OutsideEdits(DefaultProfile(), toK, "")[0].key == "model",
                "a file that says model=K under a profile of Model M is an outside edit (an install that starts on Preset K saves K)");

        // ---- OptiScaler.ini: the entry OptiScaler starts with -----------------------------------------------------------
        const string slot = "[WrongGPU]\r\nUpscalerIndex=0\r\n[FSR]\r\nUpscalerIndex=0\r\n[Menu]\r\nShowFps=auto\r\n";
        require(Oini(DefaultProfile(), slot) == slot, "Model M: OptiScaler.ini unchanged, byte for byte");
        const string slotK = Oini(With("k"), slot);
        require(slotK.find("[WrongGPU]\r\nUpscalerIndex=" + K + "\r\n") != string::npos && slotK.find("[FSR]\r\nUpscalerIndex=0\r\n") != string::npos,
                "Model K, OptiScaler 10's slot: [WrongGPU] UpscalerIndex names Model K's entry; AMD's [FSR] entry is left to OptiScaler");
        require(Oini(With("k"), slotK) == slotK, "Model K again: unchanged");
        require(Oini(DefaultProfile(), slotK) == slot, "back to Model M: [WrongGPU] UpscalerIndex=0 again, the rest as it was");
        const string other = "[WrongGPU]\r\nUpscalerIndex=7\r\n";
        require(Oini(DefaultProfile(), other) == other, "Model M: another entry the player chose in OptiScaler's menu stays");
        require(Oini(With("k"), "[FSR]\r\nUpscalerIndex=0\r\n") == "[FSR]\r\nUpscalerIndex=" + K + "\r\n",
                "Model K, WrongGPU's FFX DLL as OptiScaler's FSR: [FSR] UpscalerIndex");
        const string slotLEntry = "[WrongGPU]\r\nUpscalerIndex=" + L + "\r\n";
        if (withL) {
            require(Oini(DefaultProfile(), slotLEntry) == "[WrongGPU]\r\nUpscalerIndex=0\r\n", "Model M control: Model L's entry still goes back to 0");
            require(Oini(With("l"), slotK) == "[WrongGPU]\r\nUpscalerIndex=" + L + "\r\n[FSR]\r\nUpscalerIndex=0\r\n[Menu]\r\nShowFps=auto\r\n",
                    "Model L control: K's entry to L's");
            require(Oini(With("k"), Oini(With("l"), slot)) == slotK, "L to K: K's entry again");
        } else {
            require(Oini(DefaultProfile(), slotLEntry) == slotLEntry, "without the Model L choice, Model L's entry is not managed: it stays");
        }
        std::vector<wstring> notes;
        const string none = "[Menu]\r\nShowFps=auto\r\n";
        require(Oini(With("k"), none, &notes) == none && notes.size() == 1 && notes[0].find(L"Model K") != wstring::npos &&
                    notes[0].find(L"no UpscalerIndex key") != wstring::npos,
                "Model K without an UpscalerIndex key: nothing written, the player is told to choose it in OptiScaler's menu");
        Profile fps = With("k");
        SetSetting(fps, "fps_overlay", "on", &why);
        require(Oini(fps, slot).find("ShowFps=true") != string::npos && Oini(fps, slot).find("UpscalerIndex=" + K) != string::npos,
                "the FPS overlay setting still applies with the model");
    } catch (const std::exception& e) {
        std::cerr << "model-choice-k FAILED after " << checks << " checks: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "model-choice-k (" << (WRONGGPU_MODEL_L ? "with" : "without") << " the Model L choice): " << checks << " checks passed" << std::endl;
    return 0;
}
