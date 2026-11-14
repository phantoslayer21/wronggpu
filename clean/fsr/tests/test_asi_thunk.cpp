// SPDX-License-Identifier: MIT
// Unit tests of the WrongGPU ASI plugin thunk logic (clean/fsr/asi/asi_thunk.h), with function-pointer stand-ins for
// ImGui::Selectable / ImGui::BeginCombo. No Windows, no ImGui: builds on any host.
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifdef WRONGGPU_MODEL_K_OVERRIDE  // test_asi_thunk_kl: Model K on, whatever the build's WG_MODEL_K says
#undef WRONGGPU_MODEL_K
#define WRONGGPU_MODEL_K 1
#endif
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
#if WRONGGPU_MODEL_K && WRONGGPU_MODEL_L
    CHECK(kEntryCount == 3 && sel.calls[0].label == "WrongGPU Model M" && sel.calls[1].label == "WrongGPU Model K" &&
          sel.calls[2].label == "WrongGPU Model L");
#elif WRONGGPU_MODEL_K
    CHECK(kEntryCount == 2 && sel.calls[0].label == "WrongGPU Model M" && sel.calls[1].label == "WrongGPU Model K");
#elif WRONGGPU_MODEL_L
    CHECK(kEntryCount == 2 && sel.calls[0].label == "WrongGPU Model M" && sel.calls[1].label == "WrongGPU Model L");
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
    CHECK(ParseSelection("selection=wronggpu-l") == (WRONGGPU_MODEL_L ? kModelL : kModelM));
}

// 0.6.5: the saved choice says whether the player chose it and when (chosen=), so the front end can weigh it against
// the launcher's model choice (src/model_precedence.h).
void TestSelectionExplicitAndTime() {
    Selection s = ParseSelectionFile("");
    CHECK(s.model == kModelM && !s.explicitChoice && s.chosenAt == 0);  // no file: the default entry
    s = ParseSelectionFile("; c\r\nselection=wronggpu\r\n");             // what installers seed / 0.6.0 saved for M
    CHECK(s.model == kModelM && !s.explicitChoice && s.chosenAt == 0);
    s = ParseSelectionFile("selection=wronggpu\r\nchosen=1763150000\r\n");  // a click on Model M, with its time
    CHECK(s.model == kModelM && s.explicitChoice && s.chosenAt == 1763150000ull);
    s = ParseSelectionFile("chosen=77\r\nselection=amd\r\n");  // order does not matter
    CHECK(s.model == kModelAmd && s.explicitChoice && s.chosenAt == 77);
    s = ParseSelectionFile("selection=amd\r\n");  // an earlier version's AMD: explicit, no time
    CHECK(s.model == kModelAmd && s.explicitChoice && s.chosenAt == 0);
    s = ParseSelectionFile("selection=wronggpu\r\nchosen=soon\r\n");  // a bad time: not recorded
    CHECK(s.model == kModelM && !s.explicitChoice && s.chosenAt == 0);
    s = ParseSelectionFile("selection=garbage\r\nchosen=5\r\n");  // an unknown entry: the default, whatever the time
    CHECK(s.model == kModelM && !s.explicitChoice && s.chosenAt == 0);
    s = ParseSelectionFile("selection=wronggpu-k\r\n");
    CHECK(WRONGGPU_MODEL_K ? (s.model == kModelK && s.explicitChoice) : (s.model == kModelM && !s.explicitChoice));
    s = ParseSelectionFile("selection=wronggpu-l\r\nchosen=9\r\n");  // Model L; in a build without it, the default
    CHECK(WRONGGPU_MODEL_L ? (s.model == kModelL && s.explicitChoice && s.chosenAt == 9) : (s.model == kModelM && !s.explicitChoice));
    // round trip of every entry with and without a time
    for (const Entry& e : kEntries) {
        const Selection t = ParseSelectionFile(FormatSelection(e.model, 1234));
        CHECK(t.model == e.model && t.explicitChoice && t.chosenAt == 1234);
        const Selection u = ParseSelectionFile(FormatSelection(e.model));
        CHECK(u.model == e.model && u.chosenAt == 0 && u.explicitChoice == (e.model != kModelM));
    }
    CHECK(FormatSelection(kModelL, 5).find("selection=wronggpu-l\r\nchosen=5\r\n") != std::string::npos);
    CHECK(FormatSelection(kModelM).find("chosen=") == std::string::npos);
}

// A click records an explicit choice with its time and reports it; the menu marks the model the front end reported
// (ShownModel) until the next click.
void TestClickRecordsTimeAndShownModel() {
    PluginState st;
    FakeSelectable none;
    bool clicked = true;
    const int size = 0;
    SelectableThunkLogic(none, "FSR 3.X", true, 0, &size, st, 100, &clicked);
    CHECK(!clicked && !st.explicitChoice.load() && st.chosenAt.load() == 0);  // drawing alone records nothing
    // the front end reports that another model runs (the launcher's choice won over the plugin's default)
    st.shown.store(kEntries[kEntryCount - 1].model);
    FakeSelectable marks;
    SelectableThunkLogic(marks, "FSR 3.X", true, 0, &size, st);
    for (size_t i = 0; i < kEntryCount; ++i) CHECK(marks.calls[i].selected == (i == kEntryCount - 1));
    FakeBeginCombo bc;
    BeginComboThunkLogic(bc, "", "FSR 3.X", 0, st);
    CHECK(bc.preview == kEntries[kEntryCount - 1].label);
    // the player clicks Model M: explicit, timed, and marked again
    FakeSelectable click;
    click.clickLabel = kEntries[0].label;
    CHECK(SelectableThunkLogic(click, "FSR 3.X", true, 0, &size, st, 1763150000ull, &clicked));
    CHECK(clicked && st.model.load() == kModelM && st.explicitChoice.load() && st.chosenAt.load() == 1763150000ull);
    CHECK(st.shown.load() == kShowSaved && ShownModel(st) == kModelM);
    // a click on the entry already saved is still an explicit choice with a new time
    FakeSelectable again;
    again.clickLabel = kEntries[0].label;
    CHECK(SelectableThunkLogic(again, "FSR 3.X", true, 0, &size, st, 1763150500ull, &clicked) && clicked);
    CHECK(st.chosenAt.load() == 1763150500ull);
#if WRONGGPU_MODEL_L
    FakeSelectable l;
    l.clickLabel = "WrongGPU Model L";
    CHECK(SelectableThunkLogic(l, "FSR 3.X", true, 0, &size, st, 1763151000ull, &clicked) && clicked && st.model.load() == kModelL);
#endif
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
    TestSelectionExplicitAndTime();
    TestClickRecordsTimeAndShownModel();
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
