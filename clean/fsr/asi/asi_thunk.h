// SPDX-License-Identifier: MIT
#pragma once
// WrongGPU OptiScaler ASI plugin: the logic of the two menu thunks, free of Windows and of ImGui so it can be unit
// tested with function-pointer stand-ins (tests/test_asi_thunk.cpp). The live plugin (wronggpu_asi.cpp) gives these
// the real ImGui::Selectable / ImGui::BeginCombo it resolved in OptiScaler's image.
//
// It writes none of OptiScaler's data. When the player clicks a WrongGPU item, the Selectable thunk returns "clicked"
// for OptiScaler's own "FSR 3.X" entry, so OptiScaler's unchanged code sets its backend to FSR 3.X (which loads
// WrongGPU's FidelityFX DLL); the plugin only remembers which entry the player chose, and the front end reads that
// back. The WrongGPU labels are plain UI text.
#include <atomic>
#include <cctype>
#include <cstring>
#include <string>

#ifndef WRONGGPU_MODEL_K
#define WRONGGPU_MODEL_K 1
#endif

namespace wgasi {

constexpr char kModelAmd = 0;  // no WrongGPU model: OptiScaler's own FSR 3.X (AMD) is selected
constexpr char kModelM = 'M';
constexpr char kModelK = 'K';
constexpr char kDefaultModel = kModelM;  // WrongGPU is the default entry wherever the plugin hooks

// The entries the plugin adds in front of OptiScaler's FSR 3.X entry. A build without Model K has one, "WrongGPU".
struct Entry {
    char model;
    const char* label;
};
#if WRONGGPU_MODEL_K
constexpr Entry kEntries[] = {{kModelM, "WrongGPU Model M"}, {kModelK, "WrongGPU Model K"}};
#else
constexpr Entry kEntries[] = {{kModelM, "WrongGPU"}};
#endif
constexpr size_t kEntryCount = sizeof(kEntries) / sizeof(kEntries[0]);

inline const char* LabelOf(char model) {
    for (const Entry& e : kEntries)
        if (e.model == model) return e.label;
    return nullptr;
}

// Shared plugin state. `hooked` is read by the front end through the exported query; `model` is the player's current
// choice within OptiScaler's FSR 3.X entry.
struct PluginState {
    std::atomic<bool> hooked{false};
    std::atomic<char> model{kDefaultModel};
};

inline bool IsFsr3xLabel(const char* label) {
    return label && (std::strcmp(label, "FSR 3.X") == 0 || std::strcmp(label, "FSR 3.X/4") == 0);
}

// The Selectable thunk. `sel` is the real ImGui::Selectable(label, selected, flags, size). For OptiScaler's FSR 3.X
// entry it draws the WrongGPU items first, then the original entry, records any click as the selected model, and
// returns true (click) to OptiScaler when any of them was clicked so OptiScaler switches to its FSR 3.X backend.
// `selected` (OptiScaler's backend is FSR 3.X) marks the WrongGPU item or the FSR 3.X item, whichever is chosen.
// Every other entry passes through unchanged.
template <class SelFn>
bool SelectableThunkLogic(SelFn&& sel, const char* label, bool selected, int flags, const void* size, PluginState& st) {
    if (!IsFsr3xLabel(label)) return sel(label, selected, flags, size);
    const char model = st.model.load();
    char clicked = 1;  // 1: nothing clicked
    for (const Entry& e : kEntries)
        if (sel(e.label, selected && model == e.model, flags, size) && clicked == 1) clicked = e.model;
    if (sel(label, selected && model == kModelAmd, flags, size) && clicked == 1) clicked = kModelAmd;
    if (clicked == 1) return false;
    st.model.store(clicked);
    return true;
}

// The BeginCombo thunk: when a WrongGPU model is selected and OptiScaler would show its FSR 3.X name as the closed
// combo's preview, show the WrongGPU entry's name instead. `bc` is the real ImGui::BeginCombo(label, preview, flags).
template <class BcFn>
bool BeginComboThunkLogic(BcFn&& bc, const char* label, const char* preview, int flags, PluginState& st) {
    if (IsFsr3xLabel(preview))
        if (const char* own = LabelOf(st.model.load())) preview = own;
    return bc(label, preview, flags);
}

// The remembered choice (wronggpu-asi.ini next to the plugin): "selection=wronggpu", "wronggpu-k" or "amd". Anything
// else, or no file, is the default (WrongGPU). Model K in a build without it reads as WrongGPU.
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
        return kDefaultModel;
    }
    return kDefaultModel;
}

inline std::string FormatSelection(char model) {
    const char* v = model == kModelAmd ? "amd" : model == kModelK ? "wronggpu-k" : "wronggpu";
    return std::string("; WrongGPU OptiScaler plugin: the entry last selected in OptiScaler's upscaler list\r\n") +
           "; (wronggpu or amd). Deleting this file selects WrongGPU.\r\nselection=" + v + "\r\n";
}

}  // namespace wgasi
