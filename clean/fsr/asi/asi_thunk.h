// SPDX-License-Identifier: MIT
#pragma once
// WrongGPU OptiScaler ASI plugin: the logic of the two menu thunks, free of Windows and of ImGui so it can be unit
// tested with function-pointer stand-ins (tests/test_asi_thunk.cpp). The live plugin (wronggpu_asi.cpp) gives these
// the real ImGui::Selectable / ImGui::BeginCombo it resolved in OptiScaler's image.
//
// It writes none of OptiScaler's data. When the player clicks a WrongGPU item, the Selectable thunk returns "clicked"
// for OptiScaler's own "FSR 3.X" entry, so OptiScaler's unchanged code sets its backend to FSR 3.X (which loads
// WrongGPU's FidelityFX DLL); the plugin only remembers which entry the player chose (and when), and the front end
// reads that back. The front end decides which model runs (the plugin's choice or the launcher's: frontend
// model_precedence.h) and tells the plugin, so the menu marks the model that actually runs. The WrongGPU labels are
// plain UI text.
#include <atomic>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

#ifndef WRONGGPU_MODEL_K
#define WRONGGPU_MODEL_K 1
#endif
#ifndef WRONGGPU_MODEL_L
#define WRONGGPU_MODEL_L 0
#endif

namespace wgasi {

constexpr char kModelAmd = 0;  // no WrongGPU model: OptiScaler's own FSR 3.X (AMD) is selected
constexpr char kModelM = 'M';
constexpr char kModelK = 'K';
constexpr char kModelL = 'L';
constexpr char kDefaultModel = kModelM;  // WrongGPU is the default entry wherever the plugin hooks
constexpr char kShowSaved = 1;           // PluginState::shown: no model reported by the front end, mark the saved one

// The entries the plugin adds in front of OptiScaler's FSR 3.X entry. A build with only Model M has one, "WrongGPU".
struct Entry {
    char model;
    const char* label;
};
#if WRONGGPU_MODEL_K && WRONGGPU_MODEL_L
constexpr Entry kEntries[] = {{kModelM, "WrongGPU Model M"}, {kModelK, "WrongGPU Model K"}, {kModelL, "WrongGPU Model L"}};
#elif WRONGGPU_MODEL_K
constexpr Entry kEntries[] = {{kModelM, "WrongGPU Model M"}, {kModelK, "WrongGPU Model K"}};
#elif WRONGGPU_MODEL_L
constexpr Entry kEntries[] = {{kModelM, "WrongGPU Model M"}, {kModelL, "WrongGPU Model L"}};
#else
constexpr Entry kEntries[] = {{kModelM, "WrongGPU"}};
#endif
constexpr size_t kEntryCount = sizeof(kEntries) / sizeof(kEntries[0]);

inline const char* LabelOf(char model) {
    for (const Entry& e : kEntries)
        if (e.model == model) return e.label;
    return nullptr;
}

// Shared plugin state. `hooked` is read by the front end through the exported query. `model` is the player's saved
// choice within OptiScaler's FSR 3.X entry; `explicitChoice` says whether the player (or an earlier version's file)
// actually chose it, as opposed to the default entry; `chosenAt` is when the player clicked it (seconds since
// 1970-01-01 UTC, 0 = not recorded: a file written before 0.6.5, or by an installer). `shown` is the model the front
// end reported as running (kShowSaved until it reports one): the menu marks that one, so a launcher choice that won
// over the plugin's default is what the player sees checked.
struct PluginState {
    std::atomic<bool> hooked{false};
    std::atomic<char> model{kDefaultModel};
    std::atomic<bool> explicitChoice{false};
    std::atomic<uint64_t> chosenAt{0};
    std::atomic<char> shown{kShowSaved};
};

inline char ShownModel(const PluginState& st) {
    const char s = st.shown.load();
    return s == kShowSaved ? st.model.load() : s;
}

inline bool IsFsr3xLabel(const char* label) {
    return label && (std::strcmp(label, "FSR 3.X") == 0 || std::strcmp(label, "FSR 3.X/4") == 0);
}

// The Selectable thunk. `sel` is the real ImGui::Selectable(label, selected, flags, size). For OptiScaler's FSR 3.X
// entry it draws the WrongGPU items first, then the original entry, records any click as the player's explicit choice
// (made at `now`, seconds since 1970 UTC), and returns true (click) to OptiScaler when any of them was clicked so
// OptiScaler switches to its FSR 3.X backend. `selected` (OptiScaler's backend is FSR 3.X) marks the WrongGPU item or
// the FSR 3.X item of the model shown (ShownModel). Every other entry passes through unchanged. *clickedOut (when
// given) is set to true on a click on one of these items, so the caller saves the choice.
template <class SelFn>
bool SelectableThunkLogic(SelFn&& sel, const char* label, bool selected, int flags, const void* size, PluginState& st,
                          uint64_t now = 0, bool* clickedOut = nullptr) {
    if (clickedOut) *clickedOut = false;
    if (!IsFsr3xLabel(label)) return sel(label, selected, flags, size);
    const char model = ShownModel(st);
    char clicked = 1;  // 1: nothing clicked
    for (const Entry& e : kEntries)
        if (sel(e.label, selected && model == e.model, flags, size) && clicked == 1) clicked = e.model;
    if (sel(label, selected && model == kModelAmd, flags, size) && clicked == 1) clicked = kModelAmd;
    if (clicked == 1) return false;
    st.model.store(clicked);
    st.explicitChoice.store(true);
    st.chosenAt.store(now);
    st.shown.store(kShowSaved);  // the player's click is the newest choice: it runs and is marked
    if (clickedOut) *clickedOut = true;
    return true;
}

// The BeginCombo thunk: when a WrongGPU model is shown and OptiScaler would show its FSR 3.X name as the closed
// combo's preview, show the WrongGPU entry's name instead. `bc` is the real ImGui::BeginCombo(label, preview, flags).
template <class BcFn>
bool BeginComboThunkLogic(BcFn&& bc, const char* label, const char* preview, int flags, PluginState& st) {
    if (IsFsr3xLabel(preview))
        if (const char* own = LabelOf(ShownModel(st))) preview = own;
    return bc(label, preview, flags);
}

// The remembered choice (wronggpu-asi.ini next to the plugin): "selection=wronggpu" (Model M), "wronggpu-k",
// "wronggpu-l" or "amd", and since 0.6.5 "chosen=<seconds since 1970 UTC>", the time of the player's click.
struct Selection {
    char model = kDefaultModel;
    bool explicitChoice = false;  // false: the default entry (no file, no selection line, an unknown value, or
                                  // Model M with no recorded time: what installers write and what 0.6.0 saved for M)
    uint64_t chosenAt = 0;        // 0: not recorded
};

// Anything else, or no file, is the default (WrongGPU Model M, not an explicit choice). A model this build lacks
// (Model K or L) reads as the default too, so the launcher's choice decides.
inline Selection ParseSelectionFile(const std::string& text) {
    Selection s;
    bool stated = false;
    size_t at = 0;
    while (at < text.size()) {
        size_t end = text.find('\n', at);
        if (end == std::string::npos) end = text.size();
        std::string line;
        for (size_t i = at; i < end; ++i)
            if (!std::isspace(static_cast<unsigned char>(text[i]))) line += char(std::tolower(static_cast<unsigned char>(text[i])));
        at = end + 1;
        if (!stated && line.rfind("selection=", 0) == 0) {
            stated = true;
            const std::string v = line.substr(10);
            if (v == "amd") s.model = kModelAmd;
            else if (v == "wronggpu-k" && LabelOf(kModelK)) s.model = kModelK;
            else if (v == "wronggpu-l" && LabelOf(kModelL)) s.model = kModelL;
            else if (v == "wronggpu") s.model = kModelM;
            else stated = false;  // an unknown entry, or one this build lacks: the default
        } else if (line.rfind("chosen=", 0) == 0) {
            const std::string v = line.substr(7);
            uint64_t t = 0;
            bool digits = !v.empty() && v.size() <= 19;
            for (char c : v) digits = digits && c >= '0' && c <= '9';
            if (digits) t = std::strtoull(v.c_str(), nullptr, 10);
            s.chosenAt = t;
        }
    }
    if (!stated) return Selection{};
    s.explicitChoice = s.model != kDefaultModel || s.chosenAt != 0;
    return s;
}

// The model alone (the earlier interface).
inline char ParseSelection(const std::string& text) { return ParseSelectionFile(text).model; }

inline const char* SelectionValue(char model) {
    return model == kModelAmd ? "amd" : model == kModelK ? "wronggpu-k" : model == kModelL ? "wronggpu-l" : "wronggpu";
}

// `chosenAt` 0 writes no time (the default entry's file, as installers write it).
inline std::string FormatSelection(char model, uint64_t chosenAt = 0) {
    std::string text = std::string("; WrongGPU OptiScaler plugin: the entry last selected in OptiScaler's upscaler list\r\n") +
                       "; (wronggpu, wronggpu-k, wronggpu-l or amd) and when (chosen: seconds since 1970 UTC).\r\n"
                       "; Deleting this file selects WrongGPU's default; the launcher's model choice then decides.\r\n"
                       "selection=" + SelectionValue(model) + "\r\n";
    if (chosenAt) text += "chosen=" + std::to_string(chosenAt) + "\r\n";
    return text;
}

}  // namespace wgasi
