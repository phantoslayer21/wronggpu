// SPDX-License-Identifier: MIT
// Unit tests of the WrongGPU ASI plugin thunk logic (clean/fsr/asi/asi_thunk.h), with function-pointer stand-ins for
// ImGui::Selectable / ImGui::BeginCombo. No Windows, no ImGui: builds on any host.
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../asi/asi_thunk.h"

using namespace wgasi;

static int g_failures = 0, g_checks = 0;
#define CHECK(cond)                                                            \
    do {                                                                       \
        ++g_checks;                                                            \
        if (!(cond)) { ++g_failures; std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); } \
    } while (0)

namespace {

// A fake ImGui::Selectable that records each (label, selected) and returns true for one chosen label (a click).
struct FakeSelectable {
    struct Call { std::string label; bool selected; };
    std::vector<Call> calls;
    std::string clickLabel;  // the label whose Selectable returns true this frame
    bool operator()(const char* label, bool selected, int /*flags*/, const void* /*size*/) {
        calls.push_back({label ? label : "", selected});
        return clickLabel == (label ? label : "");
    }
};

struct FakeBeginCombo {
    std::string label, preview;
    bool operator()(const char* l, const char* p, int /*flags*/) {
        label = l ? l : "";
        preview = p ? p : "";
        return true;
    }
};

void TestPassThrough() {
    PluginState st;
    FakeSelectable sel;
    const int size = 0;
    const bool r = SelectableThunkLogic(sel, "XeSS", true, 0, &size, st);
    CHECK(sel.calls.size() == 1);
    CHECK(sel.calls[0].label == "XeSS" && sel.calls[0].selected == true);
    CHECK(r == false);  // XeSS was not the click label
    CHECK(st.model.load() == kDefaultModel);
    // Labels that merely contain the FSR 3.X text (other APIs' "... w/Dx12" entries) pass through too.
    FakeSelectable other;
    SelectableThunkLogic(other, "FSR 3.X w/Dx12", true, 0, &size, st);
    CHECK(other.calls.size() == 1 && other.calls[0].label == "FSR 3.X w/Dx12");
}

void TestDefaultIsWrongGpu() {
    PluginState st;
    CHECK(st.model.load() == kModelM && kDefaultModel == kModelM);
    FakeSelectable sel;
    const int size = 0;
    SelectableThunkLogic(sel, "FSR 3.X", true, 0, &size, st);  // OptiScaler's backend is FSR 3.X
    CHECK(sel.calls.size() == kEntryCount + 1);
    CHECK(sel.calls[0].label == kEntries[0].label && sel.calls[0].selected);
    CHECK(sel.calls.back().label == "FSR 3.X" && !sel.calls.back().selected);
#if WRONGGPU_MODEL_K
    CHECK(kEntryCount == 2 && sel.calls[0].label == "WrongGPU Model M" && sel.calls[1].label == "WrongGPU Model K");
#else
    CHECK(kEntryCount == 1 && sel.calls[0].label == "WrongGPU");
#endif
    // Another backend is active (selected == false for the FSR 3.X row): no WrongGPU row is checked.
    FakeSelectable off;
    SelectableThunkLogic(off, "FSR 3.X", false, 0, &size, st);
    for (const auto& c : off.calls) CHECK(!c.selected);
}

void TestInjectOrderAndSelection() {
    PluginState st;
    st.model.store(kModelAmd);
    FakeSelectable sel;
    const int size = 0;
    const bool r = SelectableThunkLogic(sel, "FSR 3.X", true, 0, &size, st);
    CHECK(sel.calls.size() == kEntryCount + 1);
    for (size_t i = 0; i < kEntryCount; ++i) CHECK(sel.calls[i].label == kEntries[i].label && !sel.calls[i].selected);
    CHECK(sel.calls.back().label == "FSR 3.X" && sel.calls.back().selected);  // AMD active -> checked
    CHECK(r == false);
    CHECK(st.model.load() == kModelAmd);
#if WRONGGPU_MODEL_K
    st.model.store(kModelK);
    FakeSelectable k;
    SelectableThunkLogic(k, "FSR 3.X", true, 0, &size, st);
    CHECK(!k.calls[0].selected && k.calls[1].selected && !k.calls[2].selected);
#endif
}

void TestFsr3x4Name() {
    PluginState st;
    FakeSelectable sel;
    const int size = 0;
    SelectableThunkLogic(sel, "FSR 3.X/4", true, 0, &size, st);  // the FSR4-capable label is recognised too
    CHECK(sel.calls.size() == kEntryCount + 1 && sel.calls.back().label == "FSR 3.X/4");
}

void TestClickWrongGpu() {
    PluginState st;
    st.model.store(kModelAmd);
    FakeSelectable sel;
    sel.clickLabel = kEntries[0].label;
    const int size = 0;
    const bool r = SelectableThunkLogic(sel, "FSR 3.X", false, 0, &size, st);
    CHECK(r == true);                   // reported as a click so OptiScaler switches to its FSR 3.X backend
    CHECK(st.model.load() == kModelM);  // and we remember WrongGPU
}

void TestClickAmd() {
    PluginState st;
    FakeSelectable sel;
    sel.clickLabel = "FSR 3.X/4";  // player picks OptiScaler's own FSR 3.X/4 entry
    const int size = 0;
    const bool r = SelectableThunkLogic(sel, "FSR 3.X/4", true, 0, &size, st);
    CHECK(r == true);
    CHECK(st.model.load() == kModelAmd);
    CHECK(sel.calls.back().selected == false);  // AMD shown unchecked while WrongGPU was active
}

void TestPreviewRelabel() {
    PluginState st;
    FakeBeginCombo bc;
    BeginComboThunkLogic(bc, "", "FSR 3.X/4", 0, st);
    CHECK(bc.preview == kEntries[0].label);  // closed combo shows WrongGPU

    st.model.store(kModelAmd);
    FakeBeginCombo bc2;
    BeginComboThunkLogic(bc2, "", "FSR 3.X", 0, st);
    CHECK(bc2.preview == "FSR 3.X");  // AMD selected -> unchanged

    st.model.store(kModelM);
    FakeBeginCombo bc3;
    BeginComboThunkLogic(bc3, "", "XeSS", 0, st);
    CHECK(bc3.preview == "XeSS");  // a non-FSR-3.X preview is never relabelled
}

void TestSelectionFile() {
    CHECK(ParseSelection("") == kModelM);
    CHECK(ParseSelection("selection=amd\r\n") == kModelAmd);
    CHECK(ParseSelection("; c\r\n Selection = AMD \r\n") == kModelAmd);
    CHECK(ParseSelection("selection=wronggpu") == kModelM);
    CHECK(ParseSelection("selection=garbage") == kModelM);
    CHECK(ParseSelection("selection=wronggpu-k") == (WRONGGPU_MODEL_K ? kModelK : kModelM));
    CHECK(ParseSelection(FormatSelection(kModelAmd)) == kModelAmd);
    CHECK(ParseSelection(FormatSelection(kModelM)) == kModelM);
    CHECK(FormatSelection(kModelM).find("selection=wronggpu\r\n") != std::string::npos);
}

}  // namespace

int main() {
    TestPassThrough();
    TestDefaultIsWrongGpu();
    TestInjectOrderAndSelection();
    TestFsr3x4Name();
    TestClickWrongGpu();
    TestClickAmd();
    TestPreviewRelabel();
    TestSelectionFile();
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
