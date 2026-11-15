// SPDX-License-Identifier: MIT
#pragma once
// Which model runs when WrongGPU's OptiScaler menu plugin (clean/fsr/asi) has hooked the host's upscaler list (0.6.5).
// Plain C++, no Windows headers, so the unit tests can check it on any host (tests/test_frontend.cpp,
// TestAsiLauncherPrecedence; the front end applies it in CreateUpscale).
//
// Three places can choose the model, each with the time it was chosen:
//  - the LAUNCHER: the last model item of wronggpu.ini's runtime_options (model=M / K / L) and model_chosen=, which the
//    launcher writes whenever it changes that item and whenever the player explicitly applies a model (also the same
//    one again: `WrongGPU.exe set <game> model=<m>`);
//  - the OPTISCALER INDEX: the WrongGPU entry the player picked in OptiScaler's FFX upscaler box (the version id
//    OptiScaler passes with each create; [FSR] UpscalerIndex). The front end saves the entry it saw (wronggpu-ffx-choice.ini)
//    and stamps the time when it sees a different one (FfxChoiceSeen below);
//  - the PLUGIN MENU: the entry the player last clicked in the plugin's items of OptiScaler's main list
//    (wronggpu-asi.ini selection= and chosen=).
// Times are Windows FILETIME units: 100 ns since 1601-01-01 UTC, on this PC's clock, so two choices practically never
// share a time. 0 means "not recorded" (files of earlier versions, installers, a hand edit, the first entry the front
// end ever saw): such a choice ranks as the OLDEST.
//
// A choice is a DEFAULT when nobody made it: plugin menu with no file / no selection line / Model M with no time; launcher
// with no model item or model=M with no time (the installer's item); OptiScaler index with no time (the first entry seen,
// or none). Anything else is EXPLICIT.
//
// The rule:
//  1. no explicit choice: the launcher's model item if it has one, else Model M (as before);
//  2. otherwise the explicit choice with the latest time wins; untimed explicit choices are the oldest;
//  3. on an exact tie (also: all untimed) the launcher wins, then the OptiScaler index, then the plugin menu (the launcher is
//     the deliberate settings tool);
//  4. explicit choices that name different models are a CONFLICT: the front end logs one line naming each, its time, and
//     the winner with the reason.
// AMD (the plugin's "FSR 3.X(/4)" entry, model 0) is a choice like the models. The launcher's "WrongGPU off"
// (upscaler = amd) is not a model choice: it is a master switch and still turns WrongGPU off whatever is chosen here.
// Known limit: the order follows the PC's clock; a clock set back between two choices can order them wrongly.
#include <cstdint>
#include <string>

namespace wgfsr {

constexpr char kChoiceAmd = 0;     // AMD's FSR (the plugin's FSR 3.X entry)
constexpr char kChoiceNone = '-';  // no model stated

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

// The WrongGPU entry picked in OptiScaler's FFX upscaler box, as the front end last saw it.
struct FfxModelChoice {
    char model = kChoiceNone;  // 'M', 'K', 'L', or kChoiceNone (no WrongGPU entry seen)
    uint64_t chosenAt = 0;     // when the front end saw it change; 0: the first entry seen (not counted as a choice)
    bool Explicit() const { return model != kChoiceNone && chosenAt != 0; }
};

// The saved FFX entry after a create that names `seen` (a WrongGPU model's entry, or kChoiceNone) at time `now`: the
// first entry ever seen is saved untimed (it is what the launcher or the install set); a different entry than the saved
// one is the player's choice in OptiScaler's box, made now; the same entry keeps its time. No entry seen: unchanged.
inline FfxModelChoice FfxChoiceSeen(const FfxModelChoice& saved, char seen, uint64_t now) {
    if (seen == kChoiceNone) return saved;
    if (saved.model == kChoiceNone) return FfxModelChoice{seen, 0};
    if (saved.model != seen) return FfxModelChoice{seen, now};
    return saved;
}

// wronggpu-ffx-choice.ini (next to wronggpu.ini): "model=<M|K|L>" and "chosen=<FILETIME>" (0 or none: untimed).
inline FfxModelChoice ParseFfxChoiceFile(const std::string& text) {
    FfxModelChoice c;
    size_t at = 0;
    while (at < text.size()) {
        size_t end = text.find('\n', at);
        if (end == std::string::npos) end = text.size();
        std::string line;
        for (size_t i = at; i < end; ++i)
            if (text[i] != ' ' && text[i] != '\t' && text[i] != '\r') line += text[i];
        at = end + 1;
        if (line.size() == 7 && line.compare(0, 6, "model=") == 0) {
            const char m = char(line[6] >= 'a' && line[6] <= 'z' ? line[6] - 32 : line[6]);
            if (m == 'M' || m == 'K' || m == 'L') c.model = m;
        } else if (line.compare(0, 7, "chosen=") == 0 && line.size() > 7 && line.size() <= 27) {
            uint64_t t = 0;
            bool ok = true;
            for (size_t i = 7; i < line.size(); ++i) {
                if (line[i] < '0' || line[i] > '9') { ok = false; break; }
                t = t * 10 + uint64_t(line[i] - '0');
            }
            if (ok) c.chosenAt = t;
        }
    }
    if (c.model == kChoiceNone) c.chosenAt = 0;
    return c;
}

inline std::string FormatFfxChoiceFile(const FfxModelChoice& c) {
    return std::string("; WrongGPU front end: the WrongGPU entry last seen in OptiScaler's FFX upscaler box, and when it changed\r\n"
                       "; (chosen: Windows FILETIME, 100 ns since 1601 UTC; 0 = the first entry seen, not counted as a choice).\r\n"
                       "model=") + (c.model == kChoiceNone ? std::string() : std::string(1, c.model)) + "\r\nchosen=" +
           std::to_string(c.chosenAt) + "\r\n";
}

enum class ModelWinner { Default, Plugin, Launcher, Ffx };

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

inline std::string ChoiceTime(uint64_t t) { return t ? "chosen=" + std::to_string(t) : "no time recorded, counted as the oldest"; }

inline ModelDecision DecideModel(const PluginModelChoice& plugin, const LauncherModelChoice& launcher,
                                 const FfxModelChoice& ffx = FfxModelChoice{}) {
    struct Source {
        bool on;
        char model;
        uint64_t at;
        ModelWinner who;
        const char* name;
    };
    // tie order: launcher, OptiScaler index, plugin menu
    const Source s[3] = {
        {launcher.Explicit(), launcher.model, launcher.chosenAt, ModelWinner::Launcher, "the launcher"},
        {ffx.Explicit(), ffx.model, ffx.chosenAt, ModelWinner::Ffx, "OptiScaler's upscaler box"},
        {plugin.explicitChoice, plugin.model, plugin.chosenAt, ModelWinner::Plugin, "the OptiScaler menu plugin"},
    };
    ModelDecision d;
    const Source* best = nullptr;
    for (const Source& c : s)
        if (c.on && (!best || c.at > best->at)) best = &c;  // strictly later wins: an equal time keeps the earlier in tie order
    if (!best) {
        d.model = launcher.model == kChoiceNone ? 'M' : launcher.model;  // (an unstamped model=M: Model M)
        return d;
    }
    d.model = best->model;
    d.winner = best->who;
    int tied = 0;
    for (const Source& c : s) {
        if (!c.on) continue;
        if (c.model != best->model) d.conflict = true;
        if (c.at == best->at) ++tied;
    }
    if (!d.conflict) return d;
    static const char* const where[3] = {" (wronggpu.ini runtime_options, model_chosen)", " (wronggpu-ffx-choice.ini)",
                                         " (wronggpu-asi.ini)"};
    d.line = "model choice conflict:";
    bool first = true;
    for (int i = 0; i < 3; ++i) {
        if (!s[i].on) continue;
        d.line += std::string(first ? " " : ", ") + s[i].name + " chose " + ChoiceName(s[i].model) + where[i] + " " +
                  ChoiceTime(s[i].at);
        first = false;
    }
    const char* why = best->at == 0 ? "no choice recorded a time; the launcher, then OptiScaler's box, then the plugin is taken"
                      : tied > 1    ? "the same time; the launcher, then OptiScaler's box, then the plugin is taken"
                                    : "the most recent explicit choice";
    d.line += std::string("; ") + best->name + "'s choice wins (" + why + "): " + ChoiceName(d.model) + " runs";
    return d;
}

}  // namespace wgfsr
