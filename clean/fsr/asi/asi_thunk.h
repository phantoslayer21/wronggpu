// SPDX-License-Identifier: MIT
#pragma once
// WrongGPU OptiScaler ASI plugin: the logic of the two menu thunks, free of Windows and of ImGui so it can be unit
// tested with function-pointer stand-ins (tests/test_asi_thunk.cpp). The live plugin (wronggpu_asi.cpp) gives these
// the real ImGui::Selectable / ImGui::BeginCombo it resolved in OptiScaler's image.
//
// It writes none of OptiScaler's data. When the player clicks a WrongGPU item, the Selectable thunk returns "clicked"
// for OptiScaler's own "FSR 3.X" entry, so OptiScaler's unchanged code sets its backend to FSR 3.X (which loads
// WrongGPU's FidelityFX DLL). The plugin remembers whether WrongGPU or AMD was chosen (wronggpu-asi.ini); a click on a
// WrongGPU model is also written to the game's model choice record (clean/fsr/src/model_choice_record.h), which decides
// the model with the launcher's and OptiScaler's list's choices. The front end reports the model it runs, so the menu
// marks that one. The WrongGPU labels are plain UI text.
#include <atomic>
#include <cctype>
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

// Shared plugin state. `hooked` is read by the front end through the exported query; `model` is the player's choice
// within OptiScaler's FSR 3.X entry (0: AMD). `shown` is the model the front end reported as running (kShowSaved until
// it reports one): the menu marks that one, so the model the choice record decided is what the player sees checked.
// `unsaved`: a model click the plugin could not write to the model choice record yet (0: none); its entry says so in
// the menu, "(not saved yet, retrying)", or with `refused` "(not saved: the choice record is full)".
struct PluginState {
    std::atomic<bool> hooked{false};
    std::atomic<char> model{kDefaultModel};
    std::atomic<char> shown{kShowSaved};
    std::atomic<char> unsaved{0};
    std::atomic<bool> refused{false};
};

constexpr const char* kUnsavedSuffix = " (not saved yet, retrying)";
constexpr const char* kRefusedSuffix = " (not saved: the choice record is full, see the plugin's log)";

// The label drawn for entry `e`: its name, with the not-saved note when its click is not in the record yet.
inline std::string EntryLabel(const Entry& e, const PluginState& st) {
    if (st.unsaved.load() != e.model) return e.label;
    return std::string(e.label) + (st.refused.load() ? kRefusedSuffix : kUnsavedSuffix);
}

inline char ShownModel(const PluginState& st) {
    const char s = st.shown.load();
    return s == kShowSaved ? st.model.load() : s;
}

inline bool IsFsr3xLabel(const char* label) {
    return label && (std::strcmp(label, "FSR 3.X") == 0 || std::strcmp(label, "FSR 3.X/4") == 0);
}

// The Selectable thunk. `sel` is the real ImGui::Selectable(label, selected, flags, size). For OptiScaler's FSR 3.X
// entry it draws the WrongGPU items first, then the original entry, records any click as the selected model, and
// returns true (click) to OptiScaler when any of them was clicked so OptiScaler switches to its FSR 3.X backend.
// `selected` (OptiScaler's backend is FSR 3.X) marks the item of the model shown (ShownModel). Every other entry passes
// through unchanged. *clickedOut (when given) is set to true on a click on one of these items (the caller saves it).
template <class SelFn>
bool SelectableThunkLogic(SelFn&& sel, const char* label, bool selected, int flags, const void* size, PluginState& st,
                          bool* clickedOut = nullptr) {
    if (clickedOut) *clickedOut = false;
    if (!IsFsr3xLabel(label)) return sel(label, selected, flags, size);
    const char model = ShownModel(st);
    char clicked = 1;  // 1: nothing clicked
    for (const Entry& e : kEntries)
        if (sel(EntryLabel(e, st).c_str(), selected && model == e.model, flags, size) && clicked == 1) clicked = e.model;
    if (sel(label, selected && model == kModelAmd, flags, size) && clicked == 1) clicked = kModelAmd;
    if (clicked == 1) return false;
    st.model.store(clicked);
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

// The remembered entry (wronggpu-asi.ini next to the plugin): "selection=wronggpu" (Model M), "wronggpu-k", "wronggpu-l"
// or "amd". Anything else, or no file, is the default (WrongGPU). A model this build lacks reads as WrongGPU (Model M).
// Which WrongGPU model runs is the model choice record's (model_choice_record.h); this file says WrongGPU or AMD.
inline char ParseSelection(const std::string& text) {
    size_t at = 0;
    while (at < text.size()) {
        size_t end = text.find('\n', at);
        if (end == std::string::npos) end = text.size();
        std::string line;
        for (size_t i = at; i < end; ++i)
            if (!std::isspace(static_cast<unsigned char>(text[i]))) line += char(std::tolower(static_cast<unsigned char>(text[i])));
        at = end + 1;
        if (line.rfind("selection=", 0) != 0) continue;
        const std::string v = line.substr(10);
        if (v == "amd") return kModelAmd;
        if (v == "wronggpu-k") return LabelOf(kModelK) ? kModelK : kModelM;
        if (v == "wronggpu-l") return LabelOf(kModelL) ? kModelL : kModelM;
        return kDefaultModel;
    }
    return kDefaultModel;
}

inline std::string FormatSelection(char model) {
    const char* v = model == kModelAmd ? "amd" : model == kModelK ? "wronggpu-k" : model == kModelL ? "wronggpu-l" : "wronggpu";
    return std::string("; WrongGPU OptiScaler plugin: the entry last selected in OptiScaler's upscaler list\r\n") +
           "; (wronggpu, wronggpu-k, wronggpu-l or amd). Deleting this file selects WrongGPU.\r\nselection=" + v + "\r\n";
}

}  // namespace wgasi
