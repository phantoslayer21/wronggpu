// SPDX-License-Identifier: MIT
#pragma once
// Which model runs when WrongGPU's OptiScaler menu plugin (clean/fsr/asi) has hooked the host's upscaler list: the
// plugin's saved choice or the launcher's (0.6.5). Plain C++, no Windows headers, so the unit tests can check it on
// any host (tests/test_model_precedence.cpp; the front end's own tests check it through CreateUpscale).
//
// The two choices:
//  - the plugin's: the entry the player last clicked in OptiScaler's upscaler list (wronggpu-asi.ini: selection= and,
//    since 0.6.5, chosen= with the click's time);
//  - the launcher's: the last model item of wronggpu.ini's runtime_options (model=M / K / L), which the launcher
//    writes, with model_chosen= (the time the launcher last changed that item; since 0.6.5).
// Times are seconds since 1970-01-01 UTC on this PC's clock; 0 means "not recorded" (files of earlier versions,
// installers, a hand-edited file).
//
// A choice is a DEFAULT when nobody chose it: on the plugin's side no file, no selection line, or Model M with no
// time (what installers seed and what earlier versions saved); on the launcher's side no model item, or model=M with
// no time (the installer's model=M). Anything else is EXPLICIT.
//
// The rule:
//  1. both default: Model M (as before);
//  2. only one explicit: that one (the launcher's choice wins over the plugin's default, and the reverse);
//  3. both explicit and the same model: that model;
//  4. both explicit and different (a CONFLICT): the more recent one, by its time; with equal times (also both not
//     recorded) the plugin's, the choice the player made in the game. The front end logs one line naming both choices
//     and which one won.
// AMD (the plugin's "FSR 3.X(/4)" entry, model 0) is a choice like the models. The launcher's "WrongGPU off"
// (upscaler = amd) is not a model choice: it is a master switch and still turns WrongGPU off whatever is chosen here.
#include <cstdint>
#include <string>

namespace wgfsr {

constexpr char kChoiceAmd = 0;     // AMD's FSR (the plugin's FSR 3.X entry)
constexpr char kChoiceNone = '-';  // the launcher stated no model

struct PluginModelChoice {
    char model = 'M';             // 0 (AMD), 'M', 'K', 'L'
    bool explicitChoice = false;  // false: the plugin's default entry
    uint64_t chosenAt = 0;
};

struct LauncherModelChoice {
    char model = kChoiceNone;     // 'M', 'K', 'L' (the last model item), or kChoiceNone
    uint64_t chosenAt = 0;        // wronggpu.ini model_chosen
    bool Explicit() const { return model != kChoiceNone && (model != 'M' || chosenAt != 0); }
};

enum class ModelWinner { Default, Plugin, Launcher };

struct ModelDecision {
    char model = 'M';
    ModelWinner winner = ModelWinner::Default;
    bool conflict = false;
    std::string line;  // the conflict line (empty without a conflict)
};

inline std::string ChoiceName(char model) {
    if (model == kChoiceAmd) return "AMD FSR";
    if (model == kChoiceNone) return "no model";
    return std::string("Model ") + model;
}

inline std::string ChoiceTime(uint64_t t) { return t ? "chosen=" + std::to_string(t) : "no time recorded"; }

inline ModelDecision DecideModel(const PluginModelChoice& plugin, const LauncherModelChoice& launcher) {
    ModelDecision d;
    const bool p = plugin.explicitChoice, l = launcher.Explicit();
    if (!p && !l) {
        d.model = launcher.model == kChoiceNone ? 'M' : launcher.model;  // (an unstamped model=M: Model M)
        return d;
    }
    if (p && !l) {
        d.model = plugin.model;
        d.winner = ModelWinner::Plugin;
        return d;
    }
    if (!p && l) {
        d.model = launcher.model;
        d.winner = ModelWinner::Launcher;
        return d;
    }
    if (plugin.model == launcher.model) {
        d.model = plugin.model;
        d.winner = ModelWinner::Plugin;
        return d;
    }
    const bool launcherNewer = launcher.chosenAt > plugin.chosenAt;
    d.conflict = true;
    d.winner = launcherNewer ? ModelWinner::Launcher : ModelWinner::Plugin;
    d.model = launcherNewer ? launcher.model : plugin.model;
    d.line = "model choice conflict: OptiScaler menu plugin chose " + ChoiceName(plugin.model) + " (wronggpu-asi.ini, " +
             ChoiceTime(plugin.chosenAt) + "), the launcher chose " + ChoiceName(launcher.model) +
             " (wronggpu.ini runtime_options, model_chosen " + (launcher.chosenAt ? std::to_string(launcher.chosenAt) : "not recorded") +
             "); " + (launcherNewer ? "the launcher's" : "the plugin's") + " choice wins (" +
             (launcher.chosenAt == plugin.chosenAt ? "same time or none recorded: the in-game choice is kept"
                                                   : "the more recent explicit choice") +
             "): " + ChoiceName(d.model) + " runs";
    return d;
}

}  // namespace wgfsr
