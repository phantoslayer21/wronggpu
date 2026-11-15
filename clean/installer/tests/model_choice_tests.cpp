// SPDX-License-Identifier: MIT
// The Model L choice of the launcher (#89; profile.cpp built with WRONGGPU_MODEL_L on, whatever the launcher build's
// switch): the model setting, profile.ini, wronggpu.ini's model item and the OptiScaler entry OptiScaler starts with
// (UpscalerIndex). Model M changes nothing in either file, byte for byte. Written for this project. Run by ctest
// ("model-choice").
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
string Wini(const Profile& p, const string& in, uint64_t now = 0, bool chosen = false) {
    string out;
    wstring why;
    if (!RenderWronggpuIni(p, in, out, &why, now, chosen)) throw std::runtime_error("test setup: render");
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
        // ---- the setting ---------------------------------------------------------------------------------------------
        const SettingDef* d = FindSetting("model");
        require(d && d->layer == Layer::Model && string(d->def) == "m", "the model setting: Model M by default");
        Profile p = DefaultProfile();
        wstring why;
        require(SetSetting(p, "model", "L", &why) && p.get("model") == "l", "model=L accepted (either case)");
        require(!SetSetting(p, "model", "K", &why) && why.find(L"allowed: m, l") != wstring::npos, "model=K is not a choice of the launcher");
        require(!SetSetting(p, "model", "Q", &why) && p.get("model") == "l", "a bad value is refused; the choice stays");

        // ---- profile.ini: Model M as before the choice existed; L written and read back ---------------------------------
        require(ProfileText(DefaultProfile()).find("model=") == string::npos, "profile.ini of a Model M game: no model line (as before)");
        const string lt = ProfileText(With("l"));
        require(lt.find("model=l\r\n") != string::npos, "profile.ini of a Model L game: model=l");

        // ---- wronggpu.ini --------------------------------------------------------------------------------------------
        const string fresh = "# WrongGPU settings\r\n[fsr]\r\nupscaler = network\r\nruntime_dll = wronggpu_runtime.dll\r\nruntime_options = model=M ; the model\r\n";
        require(Wini(DefaultProfile(), fresh) == fresh, "Model M: the installer's wronggpu.ini unchanged, byte for byte");
        const string withK = "[fsr]\r\nruntime_options = model=K\r\n";
        require(Wini(DefaultProfile(), withK) == withK, "Model M: Model K's item of a build with it stays as it is");
        const string toL = Wini(With("l"), fresh);
        require(toL.find("runtime_options=model=L") != string::npos && toL.find("model=M") == string::npos &&
                    toL.find("upscaler = network") != string::npos && toL.find("# WrongGPU settings") != string::npos,
                "Model L: the model item is model=L; the other lines stay");
        require(Wini(With("l"), toL) == toL, "Model L again: unchanged (the line is rewritten only when its items change)");
        // (timing_log: an item the launcher does not manage, kept where it is)
        const string mixed = "[fsr]\r\nruntime_options = model=M;timing_log=1;model=K\r\n";
        const string mixedL = Wini(With("l"), mixed);
        require(mixedL.find("runtime_options=timing_log=1;model=L\r\n") != string::npos, "Model L: every model item out, model=L last (the last one wins)");
        const string backM = Wini(DefaultProfile(), "[fsr]\r\nruntime_options = timing_log=1;model=L\r\n");
        require(backM.find("runtime_options=timing_log=1;model=M\r\n") != string::npos, "back to Model M: Model L's item becomes model=M");
        require(EffectiveSettings(toL, "")["model"] == "l" && EffectiveSettings(fresh, "")["model"] == "m" && EffectiveSettings(withK, "")["model"] == "m",
                "the file's model: L from its last model item, M otherwise");
        require(EffectiveSettings("[fsr]\r\nruntime_options = model=L;model=M\r\n", "")["model"] == "m", "the file's model: the last model item counts");

        // ---- wronggpu.ini model_chosen: the time of the launcher's model choice (0.6.5; the front end weighs it against
        // the OptiScaler menu plugin's saved choice, clean/fsr/src/model_precedence.h) --------------------------------------
        require(Wini(DefaultProfile(), fresh, 1000) == fresh, "Model M on the installer's file: no model_chosen (nothing changed, byte for byte)");
        const string toLt = Wini(With("l"), fresh, 1000);
        require(toLt.find("model_chosen=1000\r\n") != string::npos && toLt.find("runtime_options=model=L") != string::npos,
                "Model L chosen: model_chosen = the time of the change, next to runtime_options");
        require(Wini(With("l"), toLt, 2000) == toLt, "Model L again: model_chosen stays (the model did not change)");
        const string otherSetting = Wini([] { Profile q = With("l"); wstring w; SetSetting(q, "sparkle_guard", "on", &w); return q; }(), toLt, 3000);
        require(otherSetting.find("model_chosen=1000\r\n") != string::npos && otherSetting.find("model_chosen=3000") == string::npos,
                "another setting changed: model_chosen stays (only a model change is a model choice)");
        const string backMt = Wini(DefaultProfile(), toLt, 4000);
        require(backMt.find("model=M") != string::npos && backMt.find("model_chosen=4000\r\n") != string::npos && backMt.find("model_chosen=1000") == string::npos,
                "back to Model M: model_chosen = the time of that change (replaced, one line)");
        require(Wini(With("l"), "[fsr]\r\nruntime_options = model=K\r\n", 5000).find("model_chosen=5000") != string::npos,
                "from another model's item (Model K) to Model L: stamped");
        require(Wini(DefaultProfile(), withK, 6000) == withK, "Model K's item left as it is by Model M: no model_chosen");
        require(Wini(With("l"), "[fsr]\r\nupscaler = network\r\n", 7000).find("model_chosen=7000") != string::npos,
                "a file without runtime_options gets model=L and model_chosen");
        require(Wini(With("l"), fresh).find("model_chosen=") != string::npos, "now = 0: the current time is written");
        // #213 review P1: the player explicitly applies the SAME model again (set <game> model=...): stamped again, newest
        const string againL = Wini(With("l"), toLt, 3000, true);
        require(againL.find("model_chosen=3000\r\n") != string::npos && againL.find("model_chosen=1000") == string::npos &&
                    againL.find("runtime_options=model=L\r\n") != string::npos,
                "Model L applied again by the player: model_chosen = the new time (the model item unchanged)");
        const string chosenM = Wini(DefaultProfile(), fresh, 3000, true);
        require(chosenM.find("model_chosen=3000\r\n") != string::npos && chosenM.find("runtime_options = model=M ; the model\r\n") != string::npos,
                "Model M explicitly applied on the installer's file: its model=M line kept as it is, stamped (an explicit M, not the default)");
        const string againM = Wini(DefaultProfile(), chosenM, 4000, true);
        require(againM.find("model_chosen=4000\r\n") != string::npos && againM.find("model_chosen=3000") == string::npos,
                "Model M applied again: stamped again");
        require(Wini(DefaultProfile(), chosenM, 5000) == chosenM, "another apply that is not a model choice: model_chosen stays");
        require(NowFileTime() > 133000000000000000ull, "NowFileTime: FILETIME units (100 ns since 1601), after 2022");

        // ---- OptiScaler.ini: the entry OptiScaler starts with -----------------------------------------------------------
        const string L = kModelLEntryIndex;
        const string slot = "[WrongGPU]\r\nUpscalerIndex=0\r\n[FSR]\r\nUpscalerIndex=0\r\n[Menu]\r\nShowFps=auto\r\n";
        require(Oini(DefaultProfile(), slot) == slot, "Model M: OptiScaler.ini unchanged, byte for byte");
        const string slotL = Oini(With("l"), slot);
        require(slotL.find("[WrongGPU]\r\nUpscalerIndex=" + L + "\r\n") != string::npos && slotL.find("[FSR]\r\nUpscalerIndex=0\r\n") != string::npos,
                "Model L, OptiScaler 10's slot: [WrongGPU] UpscalerIndex names Model L's entry; AMD's [FSR] entry is left to OptiScaler");
        require(Oini(With("l"), slotL) == slotL, "Model L again: unchanged");
        require(Oini(DefaultProfile(), slotL) == slot, "back to Model M: [WrongGPU] UpscalerIndex=0 again, the rest as it was");
        const string other = "[WrongGPU]\r\nUpscalerIndex=7\r\n";
        require(Oini(DefaultProfile(), other) == other, "Model M: another entry the player chose in OptiScaler's menu stays");
        const string fsr = "[FSR]\r\nUpscalerIndex=0\r\n";
        require(Oini(With("l"), fsr) == "[FSR]\r\nUpscalerIndex=" + L + "\r\n", "Model L, WrongGPU's FFX DLL as OptiScaler's FSR: [FSR] UpscalerIndex");
        std::vector<wstring> notes;
        const string none = "[Menu]\r\nShowFps=auto\r\n";
        require(Oini(With("l"), none, &notes) == none && notes.size() == 1 && notes[0].find(L"no UpscalerIndex key") != wstring::npos,
                "Model L without an UpscalerIndex key: nothing written, the player is told to choose it in OptiScaler's menu");
        Profile fps = With("l");
        SetSetting(fps, "fps_overlay", "on", &why);
        require(Oini(fps, slot).find("ShowFps=true") != string::npos && Oini(fps, slot).find("UpscalerIndex=" + L) != string::npos,
                "the FPS overlay setting still applies with the model");
    } catch (const std::exception& e) {
        std::cerr << "model-choice FAILED after " << checks << " checks: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "model-choice: " << checks << " checks passed" << std::endl;
    return 0;
}
