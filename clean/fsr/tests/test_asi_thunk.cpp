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
    // WithSelection: only the selection line ParseSelection reads changes; every other byte stays (the player's comments,
    // a later selection line ParseSelection never reads, line ends, a last line without one)
    const std::string mine = "# my notes: AMD for screenshots\r\n\r\n Selection = AMD \r\n; keep\nselection=wronggpu-k\n# end";
    CHECK(WithSelection(mine, kModelM) == "# my notes: AMD for screenshots\r\n\r\nselection=wronggpu\r\n; keep\nselection=wronggpu-k\n# end");
    CHECK(ParseSelection(WithSelection(mine, kModelM)) == kModelM);
    CHECK(WithSelection("selection=amd", kModelM) == "selection=wronggpu");                   // no line end: none added
    CHECK(WithSelection("; only a comment\n", kModelM) == "; only a comment\nselection=wronggpu\n");  // none there: appended
    CHECK(WithSelection("; no line end", kModelAmd) == "; no line end\r\nselection=amd\r\n");
    CHECK(WithSelection("", kModelM) == "selection=wronggpu\r\n");
    CHECK(WithSelection(FormatSelection(kModelAmd), kModelM) == FormatSelection(kModelM));    // WrongGPU's own file
}

// A click records the model and reports it; the menu marks the model the front end reported (ShownModel) until the
// next click. (The click's entry in the game's model choice record is the live plugin's: wronggpu_asi.cpp.)
void TestClickAndShownModel() {
    PluginState st;
    FakeSelectable none;
    bool clicked = true;
    const int size = 0;
    SelectableThunkLogic(none, "FSR 3.X", true, 0, &size, st, &clicked);
    CHECK(!clicked && st.model.load() == kModelM);  // drawing alone records nothing
    // the front end reports that another model runs (the model choice record decided)
    st.shown.store(kEntries[kEntryCount - 1].model);
    FakeSelectable marks;
    SelectableThunkLogic(marks, "FSR 3.X", true, 0, &size, st);
    for (size_t i = 0; i < kEntryCount; ++i) CHECK(marks.calls[i].selected == (i == kEntryCount - 1));
    FakeBeginCombo bc;
    BeginComboThunkLogic(bc, "", "FSR 3.X", 0, st);
    CHECK(bc.preview == kEntries[kEntryCount - 1].label);
    // the player clicks the first entry: recorded, marked again
    FakeSelectable click;
    click.clickLabel = kEntries[0].label;
    CHECK(SelectableThunkLogic(click, "FSR 3.X", true, 0, &size, st, &clicked));
    CHECK(clicked && st.model.load() == kModelM && st.shown.load() == kShowSaved && ShownModel(st) == kModelM);
#if WRONGGPU_MODEL_L
    FakeSelectable l;
    l.clickLabel = "WrongGPU Model L";
    CHECK(SelectableThunkLogic(l, "FSR 3.X", true, 0, &size, st, &clicked) && clicked && st.model.load() == kModelL);
    CHECK(ParseSelection(FormatSelection(kModelL)) == kModelL);
#endif
#if WRONGGPU_MODEL_K
    FakeSelectable k;
    k.clickLabel = "WrongGPU Model K";
    CHECK(SelectableThunkLogic(k, "FSR 3.X", true, 0, &size, st, &clicked) && clicked && st.model.load() == kModelK);
#endif
}

// A model click the plugin cannot write to the model choice record (the lock not taken in time, a failed write, the end of
// the sequence numbers): NOTHING changes (the selection, the mark, no click passed on to OptiScaler, nothing kept for
// later) and the entry says "(not saved, try again)" ("not saved: the choice record is full" when refused) until a click
// that is saved. The AMD entry never writes the record.
void TestUnsavedClick() {
    PluginState st;
    const int size = 0;
    const Entry& first = kEntries[0];
    const Entry& last = kEntries[kEntryCount - 1];
    st.model.store(first.model);
    st.shown.store(first.model);  // the front end runs the first model
    int saves = 0;
    auto fail = [&](char) { ++saves; return Save::NotSaved; };
    FakeSelectable click;
    click.clickLabel = last.label;
    bool clicked = true;
    CHECK(!SelectableThunkLogic(click, "FSR 3.X", true, 0, &size, st, fail, &clicked) && !clicked && saves == 1);
    CHECK(st.model.load() == first.model && st.shown.load() == first.model && st.unsaved.load() == last.model);
    FakeSelectable draw;
    SelectableThunkLogic(draw, "FSR 3.X", true, 0, &size, st, fail);
    CHECK(draw.calls.size() == kEntryCount + 1 && draw.calls[kEntryCount - 1].label == std::string(last.label) + kUnsavedSuffix);
    CHECK(draw.calls[0].selected == true || kEntryCount == 1);  // the model that runs stays marked
    for (size_t i = 0; i + 1 < kEntryCount; ++i) CHECK(draw.calls[i].label == kEntries[i].label);
    // refused (the end of the sequence numbers): said so
    auto refuse = [&](char) { return Save::Refused; };
    FakeSelectable click2;
    click2.clickLabel = std::string(last.label) + kUnsavedSuffix;
    CHECK(!SelectableThunkLogic(click2, "FSR 3.X", true, 0, &size, st, refuse, &clicked) && st.model.load() == first.model);
    FakeSelectable draw2;
    SelectableThunkLogic(draw2, "FSR 3.X", true, 0, &size, st, refuse);
    CHECK(draw2.calls[kEntryCount - 1].label == std::string(last.label) + kRefusedSuffix);
    // the AMD entry: no record write, takes effect
    int amdSaves = 0;
    FakeSelectable amd;
    amd.clickLabel = "FSR 3.X";
    CHECK(SelectableThunkLogic(amd, "FSR 3.X", true, 0, &size, st, [&](char) { ++amdSaves; return Save::NotSaved; }, &clicked) &&
          clicked && amdSaves == 0 && st.model.load() == kModelAmd && st.unsaved.load() == 0);
    // a click that is saved: takes effect, the note goes
    FakeSelectable ok;
    ok.clickLabel = last.label;
    CHECK(SelectableThunkLogic(ok, "FSR 3.X", true, 0, &size, st, [](char) { return Save::Saved; }, &clicked) && clicked &&
          st.model.load() == last.model && st.unsaved.load() == 0);
    FakeSelectable plain;
    SelectableThunkLogic(plain, "FSR 3.X", true, 0, &size, st);
    CHECK(plain.calls[kEntryCount - 1].label == last.label);
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
    TestClickAndShownModel();
    TestUnsavedClick();
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
