// SPDX-License-Identifier: MIT
// WrongGPU launcher window: UI Automation (ui_shell.h). The window paints its widgets itself, so it tells UI
// Automation about them: a fragment root for the window and one element per widget of the current frame (its hit:
// name, control type, place, enabled, focus, help text, heading level, place in its set), with Invoke for buttons, items,
// links and menu items, Toggle for switches ("toggle switch") and a menu's checked item, SelectionItem for segments, radio
// choices, the toolbar's tabs and a select's choices (which one is selected), ExpandCollapse and Value for selects,
// ExpandCollapse for what opens or folds content (More, "Show 12 more", the log's Show, "More OptiScaler settings", the
// managed keys, the first run's passed checks), RangeValue for progress bars, ScrollItem on every element (a screen
// reader's scan mode brings text into view too, not only Tab) and Scroll on the window (the page's scroll; the thumb is
// the mouse's and not an element). The sheets are containers: an open dialog (a Window with the Window pattern, a dialog)
// holds its text and buttons, the first run's sheet (a dialog too) everything on it, an open menu its items (a select's
// menu is a List of its choices, with Selection). Sets of choices have a container of their own (not a hit: an element
// whose id is below 0): a segmented control's or a radio group's choices are a Group with Selection named by the control
// ("Runs on"), the toolbar's two tabs a Tab, a shelf's lockups a List named by its heading, Settings' section index a List
// ("Sections") of ListItems with Selection (the section in view), and a lockup holds its action (the pill, or its progress
// while it works); a toast's words are an element while it shows; the rest are the window's children in reading order.
// Segments, the toolbar's tabs, menu items, a shelf's lockups and Settings' section index say their place in their set
// ("2 of 3").
// After every frame (UiaFrame) the changes are reported as UI Automation events: the focus (Tab, the arrows in a group or a
// menu, a page change, a dialog or the first run opening and closing), toggle, selection, value, fold and name changes, a
// menu, dialog or sheet that opened or closed, the page's elements replaced; toasts, a dialog's question, a game's page
// opening and the unsaved capsule are announced
// (notification events; an action's result as ActionCompleted or ActionAborted). Screen readers and cua-driver tests use
// these names (build/launcher-design.md principle 4). Typed text stays in native EDIT children: each gets its field's name
// (direct annotation), and while it is on screen the field's element is the EDIT itself, put in the tree where the
// painted field is (IRawElementProviderHwndOverride: in reading order, not ahead of the toolbar); the painted field stands
// in for it only while it is hidden (scrolled out, under a dialog). The caption buttons are the system title bar's
// (WM_GETTITLEBARINFOEX gives it their places: launcher_ui.cpp), so they are not listed a second time here.
// UIAutomationCore and Oleacc (Windows) only. Written for this project.
#include "app/ui_shell.h"
#include "app/launcher_window.h"

#include <oleacc.h>
#include <uiautomation.h>

#ifdef _MSC_VER
#pragma comment(lib, "uiautomationcore.lib")
#endif

namespace wg::app::lui {

namespace {

const Hit* FindHit(int id) {
    for (auto& h : v.hits) if (h.id == id) return &h;
    return nullptr;
}
int IndexOf(int id) {
    for (size_t i = 0; i < v.hits.size(); ++i) if (v.hits[i].id == id) return (int)i;
    return -1;
}
RECT ScreenRect(const Rc& r) {
    const UINT dpi = GetDpiForWindow(v.hwnd);
    POINT a{(LONG)std::lround(r.x * dpi / 96.f), (LONG)std::lround(r.y * dpi / 96.f)}, b{(LONG)std::lround(r.r() * dpi / 96.f), (LONG)std::lround(r.b() * dpi / 96.f)};
    ClientToScreen(v.hwnd, &a); ClientToScreen(v.hwnd, &b);
    return RECT{a.x, a.y, b.x, b.y};
}
// The caption buttons drawn in the toolbar are the system title bar's elements (Minimize, Maximize, Close: their places
// from WM_GETTITLEBARINFOEX), so they are left out here (Narrator met them twice); so is the overlay scroll track (the
// mouse's: the window's Scroll pattern scrolls the page, and scan mode never stops on a track among the page's words)
bool Exposed(const Hit& h) { return !(h.id >= ID_CAP_MIN && h.id <= ID_CAP_CLOSE) && h.role != R_SCROLL; }
// A text field whose native EDIT is on screen: its element is the EDIT (named, holding the keys, its own Value), placed
// where the painted field is (the window's override for that child window: Root::GetOverrideProviderForHwnd)
HWND HostedEdit(const Hit& h) {
    if (h.role != R_EDIT) return nullptr;
    const HWND e = EditOf(h.id);
    return e && IsWindowVisible(e) ? e : nullptr;
}
// The sheets open in this frame. A container's children are its own: an open menu's items, a dialog's text and buttons,
// everything on the first run's sheet; parent 0 is the window (the fragment root).
struct Sheets { bool menu = false, dialog = false, first = false; };
Sheets SheetsNow() {
    Sheets s;
    for (auto& h : v.hits) { s.menu |= h.id == ID_MENUBOX; s.dialog |= h.id == ID_DIALOGBOX; s.first |= h.id == ID_FR_SHEET; }
    return s;
}
int SheetOf(const Hit& h, const Sheets& s) {
    if (s.menu && h.id >= ID_MENU && h.id < ID_MENUBOX) return ID_MENUBOX;
    if (s.dialog && ((h.id >= ID_DIALOG && h.id < ID_DIALOGBOX) || h.id == ID_DIALOGBODY)) return ID_DIALOGBOX;
    if (s.first && h.id != ID_FR_SHEET) return ID_FR_SHEET;
    return 0;
}
bool IsTile(int id) { return id >= ID_TILE && id < ID_TILE + kTileSpan; }
bool Inside(const Rc& a, const Rc& b) { return a.x >= b.x - 2 && a.y >= b.y - 2 && a.r() <= b.r() + 2 && a.b() <= b.b() + 2; }
// Settings' section index: one Tab stop whose Up and Down move between its entries and whose Return or Space opens one (as
// WinUI's NavigationView), so UI Automation sees what the keyboard does: a List ("Sections") of ListItems, the section in
// view selected, not three links side by side (Narrator's "link, 2 of 3" sent Tab users past the other two)
bool IndexEntry(const Hit& h) { return h.role == R_LINK && h.id >= ID_S_INDEX0 && h.id < ID_S_INDEX0 + 3; }
// Words numbered as a frame draws them (AddText's ID_AUTO0 + n): the same id can hold other words a frame later (a callout
// above them appeared), so such an element is known by its id and its words (Elem::key_), never another sentence's
bool Worded(const Hit& h) { return h.role == R_TEXT && h.id >= ID_AUTO0 && h.id < ID_AUTO0 + kAutoSpan; }
int WordsKey(const wstring& s) { return (int)(std::hash<wstring>()(s) | 1); }
// The Library's places (a lockup and its action: ID_TILE / ID_TILEQA + its game's index in v.games; the featured banner's
// ids: whichever game it shows) name a game by where it is, and a scan's result moves the games. Such an element is known by
// its id and its game (Elem::key_, as words are by their words): one a screen reader or voice access cached for a game
// never reads or presses another game's control, its runtime id says which game it is, and an Invoke is checked against
// the game again when it runs (UiaInvokeMessage; posted messages run before the next frame). The mouse lets go of places
// (ForgetPlaces), the keyboard's focus goes with its game (HoldPlaceFocus, FollowPlaceGame).
bool Placed(int id) { return id >= ID_TILE || (id >= ID_FEAT0 && id < ID_FEAT0 + 3); }
std::map<int, int> framePlaces;  // the last frame's places -> their games' keys (UiaFrame; UiaFocusChanged during a frame)
std::map<int, int> PlacesNow() {
    std::map<int, int> m;
    for (auto& h : v.hits) if (Placed(h.id)) m[h.id] = WordsKey(win::PlaceGame(h.id));
    return m;
}
// the game a place named in the last frame, while it still names it (0: not a place drawn then, or the list changed since)
int PlaceKey(int id) {
    const auto it = framePlaces.find(id);
    if (it == framePlaces.end()) return 0;
    const wstring game = win::PlaceGame(id);
    return !game.empty() && WordsKey(game) == it->second ? it->second : 0;
}
// A toast's words: an element of their own while it shows (not a hit: it takes no clicks), the window's last child, so a
// screen reader that missed the announcement finds the result again before it fades (as WinUI's InfoBar stays readable)
const int kToastId = -0x7FFF0000;
bool ToastShown() { return !v.toast.text.empty() && v.toastRc.w > 0 && v.toastRc.h > 0; }
// The frame's elements as a tree. A container that is not a hit has an id below 0: minus the id of the hit that keys it (a
// radio group's first choice, the Library tab, a shelf's heading, the section index's first entry). Built from the hits when
// UI Automation asks (they are a few hundred: one pass, no state kept between frames).
enum GroupKind { G_RADIO = 1, G_TABS, G_SHELF, G_INDEX };
struct Tree {
    std::vector<int> order;     // every element in reading order, a container just before its first member
    std::map<int, int> parent;  // element -> its parent (0: the window)
    std::map<int, int> kind;    // container -> GroupKind
    std::map<int, size_t> at;   // hit element -> its index in v.hits
    bool Has(int id) const { return parent.count(id) != 0; }
    int Parent(int id) const { const auto it = parent.find(id); return it == parent.end() ? 0 : it->second; }
    const Hit* HitOf(int id) const { const auto it = at.find(id); return it == at.end() ? nullptr : &v.hits[it->second]; }
    int Kind(int id) const { const auto it = kind.find(id); return it == kind.end() ? 0 : it->second; }
    // the next (step 1) or previous (-1) element with the same parent; 0 none
    int Sibling(int id, int step) const {
        int i = -1;
        for (size_t k = 0; k < order.size(); ++k) if (order[k] == id) { i = (int)k; break; }
        if (i < 0) return 0;
        const int p = Parent(id);
        for (int k = i + step; k >= 0 && k < (int)order.size(); k += step) if (Parent(order[(size_t)k]) == p) return order[(size_t)k];
        return 0;
    }
    int Child(int id, bool last) const {  // id 0: the window's
        if (!last) { for (int x : order) if (Parent(x) == id) return x; }
        else for (size_t k = order.size(); k-- > 0;) if (Parent(order[k]) == id) return order[k];
        return 0;
    }
    std::vector<const Hit*> Members(int id) const {
        std::vector<const Hit*> m;
        for (int x : order) if (x > 0 && Parent(x) == id) if (const Hit* h = HitOf(x)) m.push_back(h);
        return m;
    }
};
Tree BuildTree() {
    Tree t;
    const Sheets s = SheetsNow();
    std::map<int, int> radioGroup;  // a choice's id -> its group (the choices of one control have consecutive ids)
    int shelf = 0, tile = -1;       // the shelf whose lockups follow (its heading's container); the last lockup's hit index
    for (size_t i = 0; i < v.hits.size(); ++i) {
        const Hit& h = v.hits[i];
        if (!Exposed(h)) continue;
        const int sheet = SheetOf(h, s);
        int p = sheet, group = 0, kind = 0;
        if (h.id >= ID_SHELF0 && h.id < ID_SHELF0 + 4) { shelf = -h.id; tile = -1; }
        if (h.role == R_RADIO) {
            const auto it = radioGroup.find(h.id - 1);
            group = it != radioGroup.end() ? it->second : -h.id;
            radioGroup[h.id] = group;
            kind = G_RADIO;
        } else if (h.id == ID_NAV_LIBRARY || h.id == ID_NAV_SETTINGS) {
            group = -ID_NAV_LIBRARY; kind = G_TABS;
        } else if (IndexEntry(h) && !sheet) {
            group = -ID_S_INDEX0; kind = G_INDEX;
        } else if (IsTile(h.id) && !sheet) {
            if (shelf) { group = shelf; kind = G_SHELF; }
            tile = (int)i;
        } else if (tile >= 0 && !sheet && Inside(h.full, v.hits[(size_t)tile].full)) p = v.hits[(size_t)tile].id;  // a lockup's action or progress
        else tile = -1;
        if (group) {
            if (!t.kind.count(group)) { t.kind[group] = kind; t.parent[group] = sheet; t.order.push_back(group); }
            p = group;
        }
        t.parent[h.id] = p;
        t.at[h.id] = i;
        t.order.push_back(h.id);
    }
    if (ToastShown()) { t.parent[kToastId] = 0; t.order.push_back(kToastId); }
    return t;
}
// A container's name: a choice group by its control ("Runs on: Auto" -> "Runs on"; the first run's "How it runs"), the
// toolbar's tabs, a shelf by its heading ("Needs attention, 7 games")
wstring GroupName(const Tree& t, int id) {
    if (t.Kind(id) == G_TABS) return L"Pages";
    if (t.Kind(id) == G_INDEX) return L"Sections";
    if (t.Kind(id) == G_SHELF) { const Hit* hd = t.HitOf(-id); return hd ? hd->name : wstring(); }
    const auto m = t.Members(id);
    if (m.empty()) return wstring();
    if (m[0]->id == ID_FR_INSTALL || m[0]->id == ID_FR_PORTABLE) return L"How it runs";
    const size_t colon = m[0]->name.find(L": ");
    return colon == wstring::npos ? m[0]->name : m[0]->name.substr(0, colon);
}
// An open select's menu: its items are its choices (a List of ListItems, the chosen one selected), not commands
bool ChoiceMenu() {
    if (!v.menu.open) return false;
    const Hit* o = FindHit(v.menu.owner);
    return o && o->role == R_COMBO;
}
// A hit with a selected state: a segment, a radio choice, a toolbar tab (an item with a state), a select's choice, the
// section index's entry (the section in view)
bool Selectable(const Hit& h) { return h.role == R_RADIO || (h.role == R_ITEM && h.state >= 0) || (h.role == R_MENUITEM && ChoiceMenu()) || IndexEntry(h); }
// Its place in its set (1-based) and the set's size: a segmented control's or the first run's choices (consecutive radio
// ids), the toolbar's two tabs, Settings' section index, an open menu's items, a shelf's lockups (between its heading and
// the next). False: not in a set.
bool SetPlace(const Hit& h, int& pos, int& size) {
    if (h.role == R_RADIO) {
        auto radio = [](int id) { const Hit* x = FindHit(id); return x && x->role == R_RADIO; };
        int lo = h.id, hi = h.id;
        while (radio(lo - 1)) --lo;
        while (radio(hi + 1)) ++hi;
        pos = h.id - lo + 1;
        size = hi - lo + 1;
        return true;
    }
    if (h.id == ID_NAV_LIBRARY || h.id == ID_NAV_SETTINGS) { pos = h.id - ID_NAV_LIBRARY + 1; size = 2; return true; }
    if (h.id >= ID_S_INDEX0 && h.id < ID_S_INDEX0 + 3 && h.role == R_LINK) { pos = h.id - ID_S_INDEX0 + 1; size = 3; return true; }
    const bool item = h.role == R_MENUITEM, tile = h.id >= ID_TILE && h.id < ID_TILE + kTileSpan;
    if (!item && !tile) return false;
    const int at = IndexOf(h.id);
    if (at < 0) return false;
    auto member = [&](const Hit& x) { return item ? x.role == R_MENUITEM : x.id >= ID_TILE && x.id < ID_TILE + kTileSpan; };
    auto heading = [&](const Hit& x) { return !item && x.id >= ID_SHELF0 && x.id < ID_SHELF0 + 4; };  // a shelf's heading ends its set
    int a = at, b = at;
    while (a > 0 && !heading(v.hits[(size_t)a - 1])) --a;
    while (b + 1 < (int)v.hits.size() && !heading(v.hits[(size_t)b + 1])) ++b;
    pos = size = 0;
    for (int i = a; i <= b; ++i)
        if (member(v.hits[(size_t)i])) { ++size; if (i <= at) ++pos; }
    return size > 0;
}
// What opens or folds content: 1 open, 0 folded, -1 nothing to open
int ExpandState(const Hit& h) {
    const bool menu = v.menu.open && v.menu.owner == h.id;
    if (h.role == R_COMBO || h.id == ID_G_MORE || h.id == ID_S_MODELMORE || h.id == ID_S_RRMORE) return menu ? 1 : 0;
    switch (h.id) {
        case ID_ATTNMORE: return v.attnAll ? 1 : 0;
        case ID_G_DETAILS: return v.detailsOpen ? 1 : 0;
        case ID_G_LOCKED: return v.lockedOpen ? 1 : 0;
        case ID_G_OPTIMORE: return v.optiMore ? 1 : 0;
        case ID_G_ADVMORE: return v.advOpen ? 1 : 0;
        case ID_FR_CHECKS: return v.first.checksOpen ? 1 : 0;
        default: return -1;
    }
}
long ControlType(const Hit& h) {
    switch (h.role) {
        case R_BUTTON: return UIA_ButtonControlTypeId;
        case R_TOGGLE: return UIA_ButtonControlTypeId;  // with Toggle and "toggle switch", as Windows' own switches (WinUI ToggleSwitch) are
        case R_RADIO: return UIA_RadioButtonControlTypeId;
        case R_ITEM: return h.state >= 0 ? UIA_TabItemControlTypeId : UIA_ListItemControlTypeId;
        case R_SCROLL: return UIA_ThumbControlTypeId;
        case R_COMBO: return UIA_ComboBoxControlTypeId;
        case R_LINK:  // what folds content is a button (a link goes somewhere); the section index's entries are a list's items
            return IndexEntry(h) ? UIA_ListItemControlTypeId : ExpandState(h) >= 0 ? UIA_ButtonControlTypeId : UIA_HyperlinkControlTypeId;
        case R_MENUITEM: return ChoiceMenu() ? UIA_ListItemControlTypeId : UIA_MenuItemControlTypeId;
        case R_EDIT: return UIA_EditControlTypeId;
        case R_DIALOG: return UIA_WindowControlTypeId;
        case R_MENU: return ChoiceMenu() ? UIA_ListControlTypeId : UIA_MenuControlTypeId;
        case R_PROGRESS: return UIA_ProgressBarControlTypeId;
        default: return UIA_TextControlTypeId;
    }
}
const UINT kInvokeMsg = WM_APP + 27;
const UINT kScrollMsg = WM_APP + 29;  // the window's Scroll pattern: wParam 0 by an amount (lParam: a ScrollAmount), 1 to a percent (lParam)
const UINT kFocusMsg = WM_APP + 31;   // an element's SetFocus: wParam its id
const UINT kRevealMsg = WM_APP + 35;  // an element's ScrollIntoView: wParam its id
const UINT kCloseMsg = WM_APP + 37;   // an open dialog's Window pattern Close (as Escape: its cancel answer)
// The page's scroll as UI Automation sees it (0..100 of its range; -1 when it does not scroll)
double ScrollPercent() {
    const float range = std::max(0.f, v.contentH[v.page] - v.H);
    return range <= 0 ? (double)UIA_ScrollPatternNoScroll : 100.0 * v.target[v.page] / range;
}
// The element that has the keyboard focus as UI Automation sees it: an open menu's keyboard item, else the focused
// control, else an open dialog's or the first run's sheet (the first run while its checks run); 0 none
int FocusTarget() {
    if (v.menu.open && v.menu.hot >= 0 && FindHit(ID_MENU + v.menu.hot)) return ID_MENU + v.menu.hot;
    if (v.focus && FindHit(v.focus)) return v.focus;
    if (v.dialog.open && FindHit(ID_DIALOGBOX)) return ID_DIALOGBOX;
    if (v.first.open && FindHit(ID_FR_SHEET)) return ID_FR_SHEET;
    return 0;
}

class Root;
Root* root = nullptr;

class Elem : public IRawElementProviderSimple, public IRawElementProviderFragment, public IInvokeProvider, public IToggleProvider,
             public ISelectionItemProvider, public IExpandCollapseProvider, public IValueProvider, public IRangeValueProvider,
             public ISelectionProvider, public IScrollItemProvider, public IWindowProvider {
public:
    explicit Elem(int id) : id_(id) {
        const Hit* h = id > 0 ? FindHit(id) : nullptr;
        if (h && Placed(id)) key_ = PlaceKey(id);
        else if (h && Worded(*h)) key_ = WordsKey(h->name);
    }
    // its hit in this frame: for words numbered by the frame (Worded), only while the id still holds the same words (a
    // screen reader's cached element or scan-mode cursor never reads another sentence); for a Library place (Placed), only
    // while it names the same game
    const Hit* Me() const {
        const Hit* h = FindHit(id_);
        if (h && Placed(id_)) return key_ && PlaceKey(id_) == key_ ? h : nullptr;
        return h && (!key_ || (Worded(*h) && WordsKey(h->name) == key_)) ? h : nullptr;
    }
    // IUnknown
    ULONG STDMETHODCALLTYPE AddRef() override { return InterlockedIncrement(&ref_); }
    ULONG STDMETHODCALLTYPE Release() override { const LONG r = InterlockedDecrement(&ref_); if (!r) delete this; return r; }
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void** out) override {
        if (iid == __uuidof(IUnknown) || iid == __uuidof(IRawElementProviderSimple)) *out = static_cast<IRawElementProviderSimple*>(this);
        else if (iid == __uuidof(IRawElementProviderFragment)) *out = static_cast<IRawElementProviderFragment*>(this);
        else if (iid == __uuidof(IInvokeProvider)) *out = static_cast<IInvokeProvider*>(this);
        else if (iid == __uuidof(IToggleProvider)) *out = static_cast<IToggleProvider*>(this);
        else if (iid == __uuidof(ISelectionItemProvider)) *out = static_cast<ISelectionItemProvider*>(this);
        else if (iid == __uuidof(IExpandCollapseProvider)) *out = static_cast<IExpandCollapseProvider*>(this);
        else if (iid == __uuidof(IValueProvider)) *out = static_cast<IValueProvider*>(this);
        else if (iid == __uuidof(IRangeValueProvider)) *out = static_cast<IRangeValueProvider*>(this);
        else if (iid == __uuidof(ISelectionProvider)) *out = static_cast<ISelectionProvider*>(this);
        else if (iid == __uuidof(IScrollItemProvider)) *out = static_cast<IScrollItemProvider*>(this);
        else if (iid == __uuidof(IWindowProvider)) *out = static_cast<IWindowProvider*>(this);
        else { *out = nullptr; return E_NOINTERFACE; }
        AddRef();
        return S_OK;
    }
    // IRawElementProviderSimple
    // a text field's EDIT on screen: this element overrides the EDIT's own (its place in the tree), and everything else (name,
    // value, bounds, focus, runtime id) is the EDIT's (HostedEdit)
    HRESULT STDMETHODCALLTYPE get_ProviderOptions(ProviderOptions* o) override {
        const Hit* h = id_ > 0 ? Me() : nullptr;
        *o = (ProviderOptions)(ProviderOptions_ServerSideProvider | ProviderOptions_UseComThreading | (h && HostedEdit(*h) ? ProviderOptions_OverrideProvider : 0));
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE GetPatternProvider(PATTERNID p, IUnknown** out) override {
        *out = nullptr;
        if (id_ == kToastId) return ToastShown() ? S_OK : UIA_E_ELEMENTNOTAVAILABLE;  // words only
        if (id_ < 0) {  // a container: a choice group's or the tabs' Selection; ScrollItem (its first member)
            const Tree t = BuildTree();
            const int k = t.Kind(id_);
            if (!k) return UIA_E_ELEMENTNOTAVAILABLE;
            if (p == UIA_SelectionPatternId && (k == G_RADIO || k == G_TABS || k == G_INDEX)) *out = static_cast<ISelectionProvider*>(this);
            else if (p == UIA_ScrollItemPatternId) *out = static_cast<IScrollItemProvider*>(this);
            if (*out) AddRef();
            return S_OK;
        }
        const Hit* h = Me();
        if (!h) return UIA_E_ELEMENTNOTAVAILABLE;
        if (HostedEdit(*h)) return S_OK;
        if (p == UIA_TogglePatternId && (h->role == R_TOGGLE || (h->role == R_MENUITEM && h->state == 1 && !ChoiceMenu()))) *out = static_cast<IToggleProvider*>(this);
        else if (p == UIA_SelectionItemPatternId && Selectable(*h)) *out = static_cast<ISelectionItemProvider*>(this);
        else if (p == UIA_SelectionPatternId && h->role == R_MENU && ChoiceMenu()) *out = static_cast<ISelectionProvider*>(this);  // a select's list
        else if (p == UIA_WindowPatternId && h->role == R_DIALOG) *out = static_cast<IWindowProvider*>(this);  // a Window has it (UI Automation's rule)
        else if (p == UIA_ScrollItemPatternId && h->role != R_DIALOG && h->role != R_MENU) *out = static_cast<IScrollItemProvider*>(this);
        else if (p == UIA_ExpandCollapsePatternId && ExpandState(*h) >= 0) *out = static_cast<IExpandCollapseProvider*>(this);
        else if (p == UIA_ValuePatternId && (h->role == R_COMBO || h->role == R_EDIT)) *out = static_cast<IValueProvider*>(this);  // a field's text while its EDIT is hidden
        else if (p == UIA_RangeValuePatternId && h->role == R_PROGRESS && h->state >= 0) *out = static_cast<IRangeValueProvider*>(this);
        else if (p == UIA_InvokePatternId && h->role != R_TOGGLE && h->role != R_EDIT && h->role != R_SCROLL && !Passive(h->role))
            *out = static_cast<IInvokeProvider*>(this);
        if (*out) AddRef();
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE GetPropertyValue(PROPERTYID p, VARIANT* out) override;
    HRESULT STDMETHODCALLTYPE get_HostRawElementProvider(IRawElementProviderSimple** out) override {
        *out = nullptr;
        const Hit* h = Me();
        if (const HWND e = h ? HostedEdit(*h) : nullptr) return UiaHostProviderFromHwnd(e, out);
        return S_OK;
    }
    // IRawElementProviderFragment
    HRESULT STDMETHODCALLTYPE Navigate(NavigateDirection d, IRawElementProviderFragment** out) override;
    HRESULT STDMETHODCALLTYPE GetRuntimeId(SAFEARRAY** out) override {
        *out = nullptr;
        const Hit* h = Me();
        if (h && HostedEdit(*h)) return S_OK;  // the EDIT's own
        const int ids[3] = {UiaAppendRuntimeId, id_, key_};  // words numbered by the frame: their words too (Worded); a place: its game
        const LONG n = key_ ? 3 : 2;
        *out = SafeArrayCreateVector(VT_I4, 0, (ULONG)n);
        if (!*out) return E_OUTOFMEMORY;
        for (LONG i = 0; i < n; ++i) SafeArrayPutElement(*out, &i, const_cast<int*>(&ids[i]));
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE get_BoundingRectangle(UiaRect* out) override {
        if (id_ == kToastId) {
            if (!ToastShown()) return UIA_E_ELEMENTNOTAVAILABLE;
            const RECT r = ScreenRect(v.toastRc);
            *out = UiaRect{(double)r.left, (double)r.top, (double)(r.right - r.left), (double)(r.bottom - r.top)};
            return S_OK;
        }
        if (id_ < 0) {  // a container: around its members in view (all of them when none is)
            const Tree t = BuildTree();
            if (!t.Kind(id_)) return UIA_E_ELEMENTNOTAVAILABLE;
            const auto m = t.Members(id_);
            bool any = false;
            for (const Hit* x : m) any |= x->shown;
            Rc u;
            bool first = true;
            for (const Hit* x : m) {
                if (any && !x->shown) continue;
                if (first) { u = x->r; first = false; continue; }
                const float l = std::min(u.x, x->r.x), tp = std::min(u.y, x->r.y), r = std::max(u.r(), x->r.r()), b = std::max(u.b(), x->r.b());
                u = Rc{l, tp, r - l, b - tp};
            }
            if (first) { *out = UiaRect{0, 0, 0, 0}; return S_OK; }
            const RECT sr = ScreenRect(u);
            *out = UiaRect{(double)sr.left, (double)sr.top, (double)(sr.right - sr.left), (double)(sr.bottom - sr.top)};
            return S_OK;
        }
        const Hit* h = Me();
        if (!h) return UIA_E_ELEMENTNOTAVAILABLE;
        if (HostedEdit(*h)) { *out = UiaRect{0, 0, 0, 0}; return S_OK; }  // the EDIT's own
        const RECT r = ScreenRect(h->r);
        *out = UiaRect{(double)r.left, (double)r.top, (double)(r.right - r.left), (double)(r.bottom - r.top)};
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE GetEmbeddedFragmentRoots(SAFEARRAY** out) override { *out = nullptr; return S_OK; }
    HRESULT STDMETHODCALLTYPE SetFocus() override {
        if (id_ == kToastId) return ToastShown() ? S_OK : UIA_E_ELEMENTNOTAVAILABLE;  // it takes no focus and is always in view
        if (id_ < 0) return ScrollIntoView();  // a container takes no focus: brought into view
        if (!Me()) return UIA_E_ELEMENTNOTAVAILABLE;
        // as Tab moves the focus (FocusId): scrolled into view, a text field's native EDIT gets the keys, the focus-changed
        // event is raised; on the window's thread, after this call returns. Words, which take no focus, are brought into view.
        PostMessageW(v.hwnd, kFocusMsg, (WPARAM)id_, (LPARAM)key_);  // a place: its game, checked again there
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE get_FragmentRoot(IRawElementProviderFragmentRoot** out) override;
    // IInvokeProvider
    HRESULT STDMETHODCALLTYPE Invoke() override {
        const Hit* h = Me();
        if (!h) return UIA_E_ELEMENTNOTAVAILABLE;
        if (!h->enabled) return UIA_E_ELEMENTNOTENABLED;
        PostMessageW(v.hwnd, kInvokeMsg, (WPARAM)id_, (LPARAM)key_);  // on the window's thread, after this call returns (a place: its game)
        return S_OK;
    }
    // IToggleProvider (a switch; a menu's checked item, whose Toggle chooses it again)
    HRESULT STDMETHODCALLTYPE Toggle() override { return Invoke(); }
    HRESULT STDMETHODCALLTYPE get_ToggleState(ToggleState* out) override {
        const Hit* h = Me();
        if (!h) return UIA_E_ELEMENTNOTAVAILABLE;
        *out = h->state == 1 ? ToggleState_On : ToggleState_Off;
        return S_OK;
    }
    // ISelectionItemProvider (a segment, a radio choice, a toolbar tab: one of a group is selected; choosing one is a click)
    HRESULT STDMETHODCALLTYPE Select() override { return Invoke(); }
    HRESULT STDMETHODCALLTYPE AddToSelection() override { return UIA_E_INVALIDOPERATION; }
    HRESULT STDMETHODCALLTYPE RemoveFromSelection() override { return UIA_E_INVALIDOPERATION; }
    HRESULT STDMETHODCALLTYPE get_IsSelected(BOOL* out) override {
        const Hit* h = Me();
        if (!h) return UIA_E_ELEMENTNOTAVAILABLE;
        *out = h->state == 1;
        return S_OK;
    }
    // its group (a segmented control's or a radio group's Group, the toolbar's Tab), or a select's list
    HRESULT STDMETHODCALLTYPE get_SelectionContainer(IRawElementProviderSimple** out) override {
        *out = nullptr;
        const Hit* h = Me();
        if (!h) return UIA_E_ELEMENTNOTAVAILABLE;
        int c = 0;
        if (h->role == R_MENUITEM) c = FindHit(ID_MENUBOX) ? ID_MENUBOX : 0;
        else { const Tree t = BuildTree(); const int p = t.Parent(id_); if (p < 0 && (t.Kind(p) == G_RADIO || t.Kind(p) == G_TABS || t.Kind(p) == G_INDEX)) c = p; }
        if (c) *out = static_cast<IRawElementProviderSimple*>(new Elem(c));
        return S_OK;
    }
    // ISelectionProvider (a choice group, the toolbar's tabs, a select's list): its one chosen item
    HRESULT STDMETHODCALLTYPE GetSelection(SAFEARRAY** out) override {
        *out = nullptr;
        std::vector<int> sel;
        if (id_ == ID_MENUBOX) {
            if (!Me()) return UIA_E_ELEMENTNOTAVAILABLE;
            for (auto& h : v.hits) if (h.role == R_MENUITEM && h.state == 1) sel.push_back(h.id);
        } else {
            const Tree t = BuildTree();
            if (!t.Kind(id_)) return UIA_E_ELEMENTNOTAVAILABLE;
            for (const Hit* x : t.Members(id_)) if (x->state == 1) sel.push_back(x->id);
        }
        *out = SafeArrayCreateVector(VT_UNKNOWN, 0, (ULONG)sel.size());
        if (!*out) return E_OUTOFMEMORY;
        for (LONG i = 0; i < (LONG)sel.size(); ++i) {
            Elem* e = new Elem(sel[(size_t)i]);
            SafeArrayPutElement(*out, &i, static_cast<IRawElementProviderSimple*>(e));  // the array holds its own reference
            e->Release();
        }
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE get_CanSelectMultiple(BOOL* out) override { *out = FALSE; return S_OK; }
    HRESULT STDMETHODCALLTYPE get_IsSelectionRequired(BOOL* out) override { *out = TRUE; return S_OK; }
    // IScrollItemProvider: into view as Tab brings a control (the toolbar, the pinned header and the unsaved capsule kept
    // clear), words too; a container by its first member; on the window's thread
    HRESULT STDMETHODCALLTYPE ScrollIntoView() override {
        if (id_ == kToastId) return ToastShown() ? S_OK : UIA_E_ELEMENTNOTAVAILABLE;
        int to = id_;
        if (id_ < 0) {
            const Tree t = BuildTree();
            if (!t.Kind(id_)) return UIA_E_ELEMENTNOTAVAILABLE;
            to = t.Child(id_, false);
        } else if (!Me()) return UIA_E_ELEMENTNOTAVAILABLE;
        if (to > 0) PostMessageW(v.hwnd, kRevealMsg, (WPARAM)to, 0);
        return S_OK;
    }
    // IWindowProvider (an open dialog, the first run's sheet): modal, neither sized nor moved; Close answers a dialog as
    // Escape does (its cancel button); the first run has no answer to close it with
    HRESULT STDMETHODCALLTYPE SetVisualState(WindowVisualState) override { return UIA_E_INVALIDOPERATION; }
    HRESULT STDMETHODCALLTYPE Close() override {
        if (!Me()) return UIA_E_ELEMENTNOTAVAILABLE;
        if (id_ != ID_DIALOGBOX) return UIA_E_INVALIDOPERATION;
        PostMessageW(v.hwnd, kCloseMsg, 0, 0);
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE WaitForInputIdle(int, BOOL* out) override { *out = TRUE; return S_OK; }
    HRESULT STDMETHODCALLTYPE get_CanMaximize(BOOL* out) override { *out = FALSE; return S_OK; }
    HRESULT STDMETHODCALLTYPE get_CanMinimize(BOOL* out) override { *out = FALSE; return S_OK; }
    HRESULT STDMETHODCALLTYPE get_IsModal(BOOL* out) override { *out = TRUE; return S_OK; }
    HRESULT STDMETHODCALLTYPE get_WindowVisualState(WindowVisualState* out) override { *out = WindowVisualState_Normal; return S_OK; }
    HRESULT STDMETHODCALLTYPE get_WindowInteractionState(WindowInteractionState* out) override {
        *out = WindowInteractionState_ReadyForUserInteraction;
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE get_IsTopmost(BOOL* out) override { *out = FALSE; return S_OK; }
    // IExpandCollapseProvider (a select's menu, More's menu, a fold): opening and folding are a click
    HRESULT STDMETHODCALLTYPE Expand() override {
        const Hit* h = Me();
        if (!h) return UIA_E_ELEMENTNOTAVAILABLE;
        return ExpandState(*h) == 1 ? S_OK : Invoke();
    }
    HRESULT STDMETHODCALLTYPE Collapse() override {
        const Hit* h = Me();
        if (!h) return UIA_E_ELEMENTNOTAVAILABLE;
        if (ExpandState(*h) != 1) return S_OK;
        if (v.menu.open && v.menu.owner == id_) {  // as Escape closes it: the focus back on its control (UiaFrame says so)
            v.menu.open = false; v.focus = id_; v.hot = 0;
            Invalidate();
            return S_OK;
        }
        return Invoke();
    }
    HRESULT STDMETHODCALLTYPE get_ExpandCollapseState(ExpandCollapseState* out) override {
        const Hit* h = Me();
        if (!h) return UIA_E_ELEMENTNOTAVAILABLE;
        *out = ExpandState(*h) == 1 ? ExpandCollapseState_Expanded : ExpandCollapseState_Collapsed;
        return S_OK;
    }
    // IValueProvider (a select's shown choice, changed through its menu; a text field's text, typed into its native EDIT)
    HRESULT STDMETHODCALLTYPE SetValue(LPCWSTR) override { return UIA_E_INVALIDOPERATION; }
    HRESULT STDMETHODCALLTYPE get_Value(BSTR* out) override {
        const Hit* h = Me();
        if (!h) return UIA_E_ELEMENTNOTAVAILABLE;
        *out = SysAllocString(h->value.c_str());
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE get_IsReadOnly(BOOL* out) override { *out = TRUE; return S_OK; }
    // IRangeValueProvider (a progress bar: 0..100, read only)
    HRESULT STDMETHODCALLTYPE SetValue(double) override { return UIA_E_INVALIDOPERATION; }
    HRESULT STDMETHODCALLTYPE get_Value(double* out) override {
        const Hit* h = Me();
        if (!h) return UIA_E_ELEMENTNOTAVAILABLE;
        *out = std::max(0, h->state);
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE get_Maximum(double* out) override { *out = 100; return S_OK; }
    HRESULT STDMETHODCALLTYPE get_Minimum(double* out) override { *out = 0; return S_OK; }
    HRESULT STDMETHODCALLTYPE get_LargeChange(double* out) override { *out = 0; return S_OK; }
    HRESULT STDMETHODCALLTYPE get_SmallChange(double* out) override { *out = 0; return S_OK; }
    int id_;
    int key_ = 0;  // words numbered by the frame: their words' key when the element was made (Worded); 0 any other element

private:
    LONG ref_ = 1;
};

class Root : public IRawElementProviderSimple, public IRawElementProviderFragment, public IRawElementProviderFragmentRoot, public IScrollProvider,
             public IRawElementProviderHwndOverride {
public:
    explicit Root(HWND h) : hwnd_(h) {}
    ULONG STDMETHODCALLTYPE AddRef() override { return InterlockedIncrement(&ref_); }
    ULONG STDMETHODCALLTYPE Release() override { const LONG r = InterlockedDecrement(&ref_); if (!r) delete this; return r; }
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void** out) override {
        if (iid == __uuidof(IUnknown) || iid == __uuidof(IRawElementProviderSimple)) *out = static_cast<IRawElementProviderSimple*>(this);
        else if (iid == __uuidof(IRawElementProviderFragment)) *out = static_cast<IRawElementProviderFragment*>(this);
        else if (iid == __uuidof(IRawElementProviderFragmentRoot)) *out = static_cast<IRawElementProviderFragmentRoot*>(this);
        else if (iid == __uuidof(IScrollProvider)) *out = static_cast<IScrollProvider*>(this);
        else if (iid == __uuidof(IRawElementProviderHwndOverride)) *out = static_cast<IRawElementProviderHwndOverride*>(this);
        else { *out = nullptr; return E_NOINTERFACE; }
        AddRef();
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE get_ProviderOptions(ProviderOptions* o) override { *o = (ProviderOptions)(ProviderOptions_ServerSideProvider | ProviderOptions_UseComThreading); return S_OK; }
    HRESULT STDMETHODCALLTYPE GetPatternProvider(PATTERNID p, IUnknown** out) override {
        *out = nullptr;
        if (p == UIA_ScrollPatternId) { *out = static_cast<IScrollProvider*>(this); AddRef(); }
        return S_OK;
    }
    // IScrollProvider: the page's vertical scroll (the window's thread does it)
    HRESULT STDMETHODCALLTYPE Scroll(ScrollAmount h, ScrollAmount vert) override {
        if (h != ScrollAmount_NoAmount) return UIA_E_INVALIDOPERATION;
        if (vert != ScrollAmount_NoAmount) PostMessageW(hwnd_, kScrollMsg, 0, (LPARAM)vert);
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE SetScrollPercent(double h, double vert) override {
        if (h != UIA_ScrollPatternNoScroll) return UIA_E_INVALIDOPERATION;
        if (vert != UIA_ScrollPatternNoScroll) {
            if (vert < 0 || vert > 100) return E_INVALIDARG;
            PostMessageW(hwnd_, kScrollMsg, 1, (LPARAM)std::lround(vert * 100));
        }
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE get_HorizontalScrollPercent(double* out) override { *out = UIA_ScrollPatternNoScroll; return S_OK; }
    HRESULT STDMETHODCALLTYPE get_VerticalScrollPercent(double* out) override { *out = ScrollPercent(); return S_OK; }
    HRESULT STDMETHODCALLTYPE get_HorizontalViewSize(double* out) override { *out = 100; return S_OK; }
    HRESULT STDMETHODCALLTYPE get_VerticalViewSize(double* out) override {
        const float ch = v.contentH[v.page];
        *out = ch > v.H ? 100.0 * v.H / ch : 100.0;
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE get_HorizontallyScrollable(BOOL* out) override { *out = FALSE; return S_OK; }
    HRESULT STDMETHODCALLTYPE get_VerticallyScrollable(BOOL* out) override { *out = v.contentH[v.page] > v.H + 0.5f; return S_OK; }
    HRESULT STDMETHODCALLTYPE GetPropertyValue(PROPERTYID p, VARIANT* out) override {
        out->vt = VT_EMPTY;
        if (p == UIA_NamePropertyId) { out->vt = VT_BSTR; out->bstrVal = SysAllocString(L"WrongGPU"); }
        else if (p == UIA_AutomationIdPropertyId) { out->vt = VT_BSTR; out->bstrVal = SysAllocString(L"WrongGPULauncher"); }
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE get_HostRawElementProvider(IRawElementProviderSimple** out) override { return UiaHostProviderFromHwnd(hwnd_, out); }
    HRESULT STDMETHODCALLTYPE Navigate(NavigateDirection d, IRawElementProviderFragment** out) override {
        *out = nullptr;
        if (d != NavigateDirection_FirstChild && d != NavigateDirection_LastChild) return S_OK;
        const int c = BuildTree().Child(0, d == NavigateDirection_LastChild);
        if (c) *out = new Elem(c);
        return S_OK;
    }
    // IRawElementProviderHwndOverride: a text field's EDIT on screen is the field's element, where the painted field is in
    // the reading order (the window's child windows would otherwise come first, ahead of the toolbar)
    HRESULT STDMETHODCALLTYPE GetOverrideProviderForHwnd(HWND child, IRawElementProviderSimple** out) override {
        *out = nullptr;
        for (auto& h : v.hits)
            if (HostedEdit(h) == child) { *out = static_cast<IRawElementProviderSimple*>(new Elem(h.id)); break; }
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE GetRuntimeId(SAFEARRAY** out) override { *out = nullptr; return S_OK; }
    HRESULT STDMETHODCALLTYPE get_BoundingRectangle(UiaRect* out) override { *out = UiaRect{0, 0, 0, 0}; return S_OK; }
    HRESULT STDMETHODCALLTYPE GetEmbeddedFragmentRoots(SAFEARRAY** out) override { *out = nullptr; return S_OK; }
    HRESULT STDMETHODCALLTYPE SetFocus() override { return S_OK; }
    HRESULT STDMETHODCALLTYPE get_FragmentRoot(IRawElementProviderFragmentRoot** out) override { *out = this; AddRef(); return S_OK; }
    HRESULT STDMETHODCALLTYPE ElementProviderFromPoint(double x, double y, IRawElementProviderFragment** out) override {
        *out = nullptr;
        POINT pt{(LONG)x, (LONG)y};
        ScreenToClient(hwnd_, &pt);
        const UINT dpi = GetDpiForWindow(hwnd_);
        const float cx = pt.x * 96.f / dpi, cy = pt.y * 96.f / dpi;
        for (size_t i = v.hits.size(); i-- > 0;)
            if (v.hits[i].shown && Exposed(v.hits[i]) && !HostedEdit(v.hits[i]) && v.hits[i].r.has(cx, cy)) { *out = new Elem(v.hits[i].id); return S_OK; }
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE GetFocus(IRawElementProviderFragment** out) override {
        *out = nullptr;
        const int f = FocusTarget();
        if (f && ::GetFocus() == hwnd_) *out = new Elem(f);  // the keys in a native EDIT: UI Automation asks the EDIT
        return S_OK;
    }
    HWND hwnd_;

private:
    LONG ref_ = 1;
};

HRESULT Elem::GetPropertyValue(PROPERTYID p, VARIANT* out) {
    out->vt = VT_EMPTY;
    auto str = [&](const wstring& s) { out->vt = VT_BSTR; out->bstrVal = SysAllocString(s.c_str()); };
    auto boolean = [&](bool b) { out->vt = VT_BOOL; out->boolVal = b ? VARIANT_TRUE : VARIANT_FALSE; };
    auto integer = [&](int x) { out->vt = VT_I4; out->lVal = x; };
    if (id_ == kToastId) {  // a toast's words, while it shows
        if (!ToastShown()) return UIA_E_ELEMENTNOTAVAILABLE;
        switch (p) {
            case UIA_NamePropertyId: str(v.toast.text); break;
            case UIA_ControlTypePropertyId: integer(UIA_TextControlTypeId); break;
            case UIA_AutomationIdPropertyId: str(L"wg-toast"); break;
            case UIA_IsEnabledPropertyId: case UIA_IsControlElementPropertyId: case UIA_IsContentElementPropertyId: boolean(true); break;
            case UIA_IsKeyboardFocusablePropertyId: case UIA_HasKeyboardFocusPropertyId: case UIA_IsOffscreenPropertyId: boolean(false); break;
            case UIA_FrameworkIdPropertyId: str(L"WrongGPU"); break;
            case UIA_ProcessIdPropertyId: integer((int)GetCurrentProcessId()); break;
            default: break;
        }
        return S_OK;
    }
    if (id_ < 0) {  // a container (a choice group, the toolbar's tabs, a shelf's list): named, no focus of its own
        const Tree t = BuildTree();
        const int k = t.Kind(id_);
        if (!k) return UIA_E_ELEMENTNOTAVAILABLE;
        const auto m = t.Members(id_);
        bool enabled = false, shown = false;
        for (const Hit* x : m) { enabled |= x->enabled; shown |= x->shown; }
        switch (p) {
            case UIA_NamePropertyId: str(GroupName(t, id_)); break;
            case UIA_ControlTypePropertyId: integer(k == G_TABS ? UIA_TabControlTypeId : k == G_SHELF || k == G_INDEX ? UIA_ListControlTypeId : UIA_GroupControlTypeId); break;
            case UIA_AutomationIdPropertyId: str(L"wg-group-" + std::to_wstring(-id_)); break;
            case UIA_IsEnabledPropertyId: boolean(enabled || k == G_SHELF); break;
            case UIA_IsKeyboardFocusablePropertyId: case UIA_HasKeyboardFocusPropertyId: boolean(false); break;
            case UIA_IsOffscreenPropertyId: boolean(!shown); break;
            case UIA_IsControlElementPropertyId: case UIA_IsContentElementPropertyId: boolean(true); break;
            case UIA_FrameworkIdPropertyId: str(L"WrongGPU"); break;
            case UIA_ProcessIdPropertyId: integer((int)GetCurrentProcessId()); break;
            default: break;
        }
        return S_OK;
    }
    const Hit* h = Me();
    if (!h) return UIA_E_ELEMENTNOTAVAILABLE;
    if (HostedEdit(*h)) return S_OK;  // the EDIT's own (its name given by direct annotation, its text, its focus)
    // a field whose value Apply refused: invalid, its reason first (the native EDIT says the same: UiaMarkEdit)
    const wstring bad = h->role == R_EDIT ? FieldInvalidWhy(h->id) : wstring();
    const wstring help = !bad.empty() ? bad : !h->help.empty() ? h->help : h->role == R_COMBO ? h->value : wstring();
    int pos = 0, size = 0;
    switch (p) {
        case UIA_NamePropertyId: str(h->name); break;
        case UIA_ControlTypePropertyId: out->vt = VT_I4; out->lVal = ControlType(*h); break;
        case UIA_AutomationIdPropertyId: {  // a text field: its EDIT's (the control id), scrolled out or on screen the same element
            const HWND e = h->role == R_EDIT ? EditOf(h->id) : nullptr;
            const int ctl = e ? GetDlgCtrlID(e) : 0;
            str(ctl ? std::to_wstring(ctl) : L"wg-" + std::to_wstring(h->id));
            break;
        }
        case UIA_IsEnabledPropertyId: boolean(h->enabled); break;
        // off: not a stop (Tab passes it over, as Windows' own disabled buttons say); a busy control keeps the focus it had (PaintAll)
        case UIA_IsKeyboardFocusablePropertyId: boolean((h->focusable && h->enabled) || FocusTarget() == h->id); break;
        case UIA_HasKeyboardFocusPropertyId: boolean(FocusTarget() == h->id && ::GetFocus() == v.hwnd); break;
        case UIA_IsOffscreenPropertyId: boolean(!h->shown); break;
        case UIA_IsControlElementPropertyId: case UIA_IsContentElementPropertyId: boolean(true); break;
        case UIA_ValueValuePropertyId: if (h->role == R_COMBO || h->role == R_EDIT) str(h->value); break;
        case UIA_HelpTextPropertyId: if (!help.empty()) str(help); break;
        case UIA_FullDescriptionPropertyId: if (!h->help.empty()) str(h->help); break;
        case UIA_HeadingLevelPropertyId:
            out->vt = VT_I4;
            out->lVal = h->role == R_TEXT && h->state >= 1 && h->state <= 9 ? HeadingLevel1 + h->state - 1 : HeadingLevel_None;
            break;
        case UIA_IsDialogPropertyId: boolean(h->role == R_DIALOG); break;
        case UIA_LocalizedControlTypePropertyId: if (h->role == R_TOGGLE) str(L"toggle switch"); break;  // as Windows' own switches say it
        case UIA_PositionInSetPropertyId: if (SetPlace(*h, pos, size)) integer(pos); break;
        case UIA_SizeOfSetPropertyId: if (SetPlace(*h, pos, size)) integer(size); break;
        case UIA_AcceleratorKeyPropertyId: if (h->id == ID_G_APPLY) str(L"Ctrl+S"); break;
        case UIA_IsDataValidForFormPropertyId: if (h->role == R_EDIT) boolean(bad.empty()); break;
        case UIA_FrameworkIdPropertyId: str(L"WrongGPU"); break;
        case UIA_ProcessIdPropertyId: out->vt = VT_I4; out->lVal = (LONG)GetCurrentProcessId(); break;
        default: break;
    }
    return S_OK;
}
HRESULT Elem::Navigate(NavigateDirection d, IRawElementProviderFragment** out) {
    *out = nullptr;
    const Tree t = BuildTree();
    if (!t.Has(id_) || (id_ > 0 && !Me())) return UIA_E_ELEMENTNOTAVAILABLE;
    const int parent = t.Parent(id_);
    if (d == NavigateDirection_Parent) {
        if (parent) *out = new Elem(parent);
        else if (root) { *out = root; root->AddRef(); }
        return S_OK;
    }
    // children: a sheet's, a container's members, a lockup's action (BuildTree)
    int j = 0;
    switch (d) {
        case NavigateDirection_NextSibling: j = t.Sibling(id_, 1); break;
        case NavigateDirection_PreviousSibling: j = t.Sibling(id_, -1); break;
        case NavigateDirection_FirstChild: j = t.Child(id_, false); break;
        case NavigateDirection_LastChild: j = t.Child(id_, true); break;
        default: break;
    }
    if (j) *out = new Elem(j);
    return S_OK;
}
HRESULT Elem::get_FragmentRoot(IRawElementProviderFragmentRoot** out) {
    *out = root;
    if (root) root->AddRef();
    return S_OK;
}

// ---- events after a frame (UiaFrame) ---------------------------------------------------------------------------------------
struct Seen { Role role; int state; int expand; wstring name, value; bool edit; };  // edit: the element is its native EDIT (HostedEdit)
std::map<int, Seen> seen;   // the last reported frame's elements by id
int seenFocus = 0;          // the focus last reported (FocusTarget)
int seenMenuOwner = 0;      // the open menu last reported (0: none)
ULONGLONG seenDialogAt = 0;  // the open dialog last reported (its entrance time; 0 none): a dialog over a dialog is new too
bool seenSheet = false;     // the first run's sheet was there
wstring seenGame;           // the game whose page was open (its folder; empty: another page)
wstring seenToast;          // the toast's words that were an element (empty: none shown)
void Raise(int id, EVENTID ev) {
    Elem* e = new Elem(id);
    UiaRaiseAutomationEvent(e, ev);
    e->Release();
}
void RaiseProperty(int id, PROPERTYID p, VARIANT before, VARIANT after) {
    Elem* e = new Elem(id);
    UiaRaiseAutomationPropertyChangedEvent(e, p, before, after);
    e->Release();
}
VARIANT VInt(int x) { VARIANT r; VariantInit(&r); r.vt = VT_I4; r.lVal = x; return r; }
VARIANT VBool(bool b) { VARIANT r; VariantInit(&r); r.vt = VT_BOOL; r.boolVal = b ? VARIANT_TRUE : VARIANT_FALSE; return r; }
void RaiseString(int id, PROPERTYID p, const wstring& before, const wstring& after) {
    VARIANT a, b;
    VariantInit(&a); VariantInit(&b);
    a.vt = b.vt = VT_BSTR;
    a.bstrVal = SysAllocString(before.c_str());
    b.bstrVal = SysAllocString(after.c_str());
    RaiseProperty(id, p, a, b);
    VariantClear(&a);
    VariantClear(&b);
}
// The focus element for an event: a text field's painted hit stands for its native EDIT only while that is hidden (the
// EDIT raises its own focus event when it takes the keys); nothing while the keys are in an EDIT or another window
bool FocusEventFor(int id) {
    const Hit* h = FindHit(id);
    return h && Exposed(*h) && !HostedEdit(*h) && ::GetFocus() == v.hwnd;
}

// UiaRaiseNotificationEvent: Windows 10 1709 and later (looked up, so the launcher still starts on an older Windows)
using NotifyFn = HRESULT(WINAPI*)(IRawElementProviderSimple*, NotificationKind, NotificationProcessing, BSTR, BSTR);
NotifyFn NotifyProc() {
    static const NotifyFn fn = [] {
        HMODULE m = GetModuleHandleW(L"uiautomationcore.dll");
        return m ? reinterpret_cast<NotifyFn>(reinterpret_cast<void*>(GetProcAddress(m, "UiaRaiseNotificationEvent"))) : nullptr;
    }();
    return fn;
}

// Direct annotation (Oleacc): a native EDIT's name, which UI Automation's own provider for the EDIT reads
IAccPropServices* AccProps() {
    static IAccPropServices* props = nullptr;
    static bool tried = false;
    if (!tried) {
        tried = true;
        if (FAILED(CoCreateInstance(__uuidof(CAccPropServices), nullptr, CLSCTX_INPROC_SERVER, __uuidof(IAccPropServices), (void**)&props))) props = nullptr;
    }
    return props;
}
const GUID kAccName = {0x608d3df8, 0x8128, 0x4aa7, {0xa4, 0x28, 0xf5, 0x5e, 0x49, 0x26, 0x72, 0x91}};  // PROPID_ACC_NAME (oleacc.h)
const GUID kUiaName = {0xc3a6921b, 0x4a99, 0x44f1, {0xbc, 0xa6, 0x61, 0x18, 0x70, 0x52, 0xc4, 0x31}};  // Name_Property_GUID (UIAutomationCoreApi.h)
const GUID kUiaHelp = {0x08555685, 0x0977, 0x45c7, {0xa7, 0xa6, 0xab, 0xaf, 0x56, 0x84, 0x12, 0x1a}};  // HelpText_Property_GUID
const GUID kUiaValid = {0x445ac684, 0xc3fc, 0x4dd9, {0xac, 0xf8, 0x84, 0x5a, 0x57, 0x92, 0x96, 0xba}};  // IsDataValidForForm_Property_GUID
std::map<HWND, wstring> editMarks;  // the EDITs marked invalid (UiaMarkEdit) and the reason each was given

}  // namespace

LRESULT UiaGetObject(HWND hwnd, WPARAM wp, LPARAM lp) {
    if ((LONG)lp != (LONG)UiaRootObjectId || !v.live) return 0;
    if (!root) { root = new Root(hwnd); framePlaces = PlacesNow(); }  // the places of the frame on screen (UiaFrame keeps them after)
    return UiaReturnRawElementProvider(hwnd, wp, lp, root);
}

void UiaFocusChanged(int id) {
    seenFocus = id;  // UiaFrame does not report it again
    // a place the focus went to during a frame (FollowPlaceGame: its game's new place) is known by the game it names now,
    // before UiaFrame records the frame's places
    if (root && Placed(id)) framePlaces[id] = WordsKey(win::PlaceGame(id));
    if (!root || !UiaClientsAreListening() || !FocusEventFor(id)) return;
    Raise(id, UIA_AutomationFocusChangedEventId);
}

void UiaNotify(const wstring& text, bool important, int kind) {
    if (!root || text.empty() || !UiaClientsAreListening()) return;
    const NotifyFn fn = NotifyProc();
    if (!fn) return;
    // an action's result is said as one (a screen reader can treat results and failures apart from other news)
    BSTR t = SysAllocString(text.c_str()), a = SysAllocString(kind == 1 ? L"WrongGPU.Result" : kind == 2 ? L"WrongGPU.Failure" : L"WrongGPU");
    fn(root, kind == 1 ? NotificationKind_ActionCompleted : kind == 2 ? NotificationKind_ActionAborted : NotificationKind_Other,
       important ? NotificationProcessing_ImportantAll : NotificationProcessing_All, t, a);
    SysFreeString(t);
    SysFreeString(a);
}

void UiaFrame() {
    const wstring game = v.page == P_GAME ? v.cur.dir : wstring();
    // the Library's places and their games in this frame (Placed): an element is known by both, clients listening or not
    static std::map<int, int> placesBefore;  // as the last reported frame had them
    if (root) framePlaces = PlacesNow();
    if (!root || !UiaClientsAreListening()) {
        seen.clear(); seenFocus = FocusTarget(); seenMenuOwner = 0; seenDialogAt = v.dialog.open ? v.dialog.at : 0;
        seenSheet = FindHit(ID_FR_SHEET) != nullptr; seenGame = game; seenToast = ToastShown() ? v.toast.text : wstring();
        return;
    }
    std::map<int, Seen> now;
    bool replaced = false;
    for (auto& h : v.hits) {
        if (!Exposed(h)) continue;  // not an element (the caption buttons, the scroll track)
        const Seen s{h.role, h.state, ExpandState(h), h.name, h.value, HostedEdit(h) != nullptr};
        now[h.id] = s;
        const auto it = seen.find(h.id);
        if (it == seen.end()) { replaced = true; continue; }
        const Seen& o = it->second;
        if (o.role != s.role || o.edit != s.edit) { replaced = true; continue; }  // a field's element became its EDIT or the painted one
        const auto pb = placesBefore.find(h.id);  // a place that names another game now: another element (its runtime id)
        if (pb != placesBefore.end() && pb->second != framePlaces[h.id]) { replaced = true; continue; }
        if (o.state != s.state) {
            if (h.role == R_TOGGLE)
                RaiseProperty(h.id, UIA_ToggleToggleStatePropertyId, VInt(o.state == 1 ? ToggleState_On : ToggleState_Off), VInt(s.state == 1 ? ToggleState_On : ToggleState_Off));
            else if (Selectable(h)) {
                RaiseProperty(h.id, UIA_SelectionItemIsSelectedPropertyId, VBool(o.state == 1), VBool(s.state == 1));
                // the section index follows the scroll: its entry is announced selected only when the keyboard chose it
                if (s.state == 1 && (!IndexEntry(h) || FocusTarget() == h.id)) Raise(h.id, UIA_SelectionItem_ElementSelectedEventId);
            } else if (h.role == R_PROGRESS && s.state >= 0) RaiseProperty(h.id, UIA_RangeValueValuePropertyId, VInt(std::max(0, o.state)), VInt(s.state));
        }
        if (o.expand != s.expand && s.expand >= 0)
            RaiseProperty(h.id, UIA_ExpandCollapseExpandCollapseStatePropertyId, VInt(o.expand == 1 ? ExpandCollapseState_Expanded : ExpandCollapseState_Collapsed),
                          VInt(s.expand == 1 ? ExpandCollapseState_Expanded : ExpandCollapseState_Collapsed));
        if (o.value != s.value && h.role == R_COMBO) RaiseString(h.id, UIA_ValueValuePropertyId, o.value, s.value);
        // "Show 12 more" -> "Show fewer"; not when its fold state changed with it (Show/Hide details): the state's event says
        // it once, Narrator does not hear the change twice (the name stays the visible words, which voice access says)
        if (o.name != s.name && !Passive(h.role) && !(o.expand != s.expand && s.expand >= 0)) RaiseString(h.id, UIA_NamePropertyId, o.name, s.name);
        if (o.name != s.name && Worded(h)) replaced = true;  // other words under the same number: another element (Elem::key_)
    }
    // a toast's words appeared, changed or went (their element: BuildTree)
    const wstring toast = ToastShown() ? v.toast.text : wstring();
    if (toast != seenToast) { replaced = true; seenToast = toast; }
    // the unsaved capsule appearing or its count changing ("1 unsaved change"): announced, the focus stays where it is
    const Hit* unsaved = FindHit(ID_G_UNSAVED);
    const auto was = seen.find(ID_G_UNSAVED);
    if (unsaved && (was == seen.end() || was->second.name != unsaved->name)) UiaNotify(unsaved->name);
    if (!replaced) for (auto& o : seen) if (!now.count(o.first)) { replaced = true; break; }
    seen = std::move(now);
    placesBefore = framePlaces;
    if (replaced) UiaRaiseStructureChangedEvent(root, StructureChangeType_ChildrenInvalidated, nullptr, 0);
    // a menu that opened or closed (Escape, Tab, a choice, a click outside: Narrator leaves its menu mode); a dialog that
    // opened (its question announced: Narrator reads the focused button, not the text) or closed. A closed sheet's element
    // is gone: the event carries its runtime id, which is all a client can use then (UI Automation's rule for these).
    const int menuOwner = v.menu.open && FindHit(ID_MENUBOX) ? v.menu.owner : 0;
    if (seenMenuOwner && menuOwner != seenMenuOwner) Raise(ID_MENUBOX, UIA_MenuClosedEventId);
    if (menuOwner && menuOwner != seenMenuOwner) Raise(ID_MENUBOX, UIA_MenuOpenedEventId);
    seenMenuOwner = menuOwner;
    const bool dialog = v.dialog.open && FindHit(ID_DIALOGBOX);
    const bool opened = dialog && v.dialog.at != seenDialogAt;
    if (seenDialogAt && !dialog) Raise(ID_DIALOGBOX, UIA_Window_WindowClosedEventId);
    if (opened) Raise(ID_DIALOGBOX, UIA_Window_WindowOpenedEventId);
    const bool sheet = FindHit(ID_FR_SHEET) != nullptr;  // the first run's sheet (Welcome to WrongGPU)
    if (sheet && !seenSheet) Raise(ID_FR_SHEET, UIA_Window_WindowOpenedEventId);
    if (!sheet && seenSheet && !v.dialog.open) Raise(ID_FR_SHEET, UIA_Window_WindowClosedEventId);
    if (!v.dialog.open) seenSheet = sheet;  // a dialog over it hides it for a while: it does not open again after
    // the focus: moved without Tab (an arrow in a group or a menu, a page that opened, a dialog that opened or closed)
    const int f = FocusTarget();
    if (f != seenFocus) {
        seenFocus = f;
        if (f && FocusEventFor(f)) Raise(f, UIA_AutomationFocusChangedEventId);
    }
    if (opened) {
        const wstring& t = v.dialog.title;
        UiaNotify(t.empty() ? v.dialog.body : v.dialog.body.empty() ? t : t + L" " + v.dialog.body);
    }
    seenDialogAt = dialog ? v.dialog.at : 0;
    // a game's page that opened: its name after the focus (Back), so the page is named, not only "Back to Library"
    if (!v.dialog.open && game != seenGame) {
        if (!game.empty()) UiaNotify(v.cur.name);
        seenGame = game;
    }
}

HWND EditOf(int id) {
    if (id == ID_SEARCH) return v.edSearch.h;
    if (id == ID_S_URL) return v.edUrl.h;
    if (id >= ID_G_OPTI0 && id < ID_G_OPTI_END && (id - ID_G_OPTI0) % 10 == 0) {
        const size_t row = (size_t)(id - ID_G_OPTI0) / 10;
        if (row < v.edOpti.size()) return v.edOpti[row].h;
    }
    return nullptr;
}

void UiaNameEdit(HWND h, const wstring& name) {
    IAccPropServices* a = h && !name.empty() ? AccProps() : nullptr;
    if (!a) return;
    a->SetHwndPropStr(h, (DWORD)OBJID_CLIENT, CHILDID_SELF, kAccName, name.c_str());
    a->SetHwndPropStr(h, (DWORD)OBJID_CLIENT, CHILDID_SELF, kUiaName, name.c_str());
}

void UiaForgetEdit(HWND h) {
    editMarks.erase(h);
    IAccPropServices* a = h ? AccProps() : nullptr;
    if (!a) return;
    MSAAPROPID ids[4] = {kAccName, kUiaName, kUiaHelp, kUiaValid};
    a->ClearHwndProps(h, (DWORD)OBJID_CLIENT, CHILDID_SELF, ids, 4);
}

// A value Apply refused (FocusOptiField, then each frame while it stays): the native EDIT says it is invalid, and why, by
// direct annotation (UI Automation's own provider for the EDIT reads them); once its text changes, valid again
void UiaMarkEdit(HWND h, const wstring& why) {
    if (!h) return;
    const auto it = editMarks.find(h);
    if (it == editMarks.end() ? why.empty() : it->second == why) return;  // as it is
    IAccPropServices* a = AccProps();
    if (!a) return;
    if (why.empty()) {
        editMarks.erase(it);
        MSAAPROPID ids[2] = {kUiaHelp, kUiaValid};
        a->ClearHwndProps(h, (DWORD)OBJID_CLIENT, CHILDID_SELF, ids, 2);
        return;
    }
    editMarks[h] = why;
    VARIANT no; VariantInit(&no); no.vt = VT_BOOL; no.boolVal = VARIANT_FALSE;
    a->SetHwndProp(h, (DWORD)OBJID_CLIENT, CHILDID_SELF, kUiaValid, no);
    a->SetHwndPropStr(h, (DWORD)OBJID_CLIENT, CHILDID_SELF, kUiaHelp, why.c_str());
}

wstring FieldInvalidWhy(int id) {
    const auto& b = v.optiBad;
    if (id < ID_G_OPTI0 || id >= ID_G_OPTI_END || (id - ID_G_OPTI0) % 10 != 0 || v.page != P_GAME) return L"";
    const size_t row = (size_t)(id - ID_G_OPTI0) / 10;
    if (row != b.row || row >= v.optiValues.size() || v.optiValues[row] != b.value || b.dir.empty() || !IEq(b.dir, v.cur.dir)) return L"";
    return b.why;
}

void UiaShutdown(HWND hwnd) {
    seen.clear();
    framePlaces.clear();
    if (!root) return;
    UiaReturnRawElementProvider(hwnd, 0, 0, nullptr);
    UiaDisconnectProvider(root);
    root->Release();
    root = nullptr;
}

bool UiaInvokeMessage(UINT msg, WPARAM wp, LPARAM lp) {
    if (msg == kScrollMsg) {  // the page by a step or a page, or to a percent of its range
        const Page p = v.page;
        const float range = std::max(0.f, v.contentH[p] - v.H);
        float to = v.target[p];
        if (wp == 1) to = range * (float)lp / 10000.f;
        else switch ((ScrollAmount)lp) {
            case ScrollAmount_LargeDecrement: to -= v.H * 0.85f; break;
            case ScrollAmount_SmallDecrement: to -= 56; break;
            case ScrollAmount_LargeIncrement: to += v.H * 0.85f; break;
            case ScrollAmount_SmallIncrement: to += 56; break;
            default: break;
        }
        v.target[p] = std::min(std::max(0.f, to), range);
        v.scroll[p] = v.target[p];
        Invalidate();
        return true;
    }
    if (msg == kFocusMsg) {
        // an open menu's item: the menu's keyboard item moves there (FocusTarget reports it, Return and Escape act from it),
        // as the arrows move it
        const int id = (int)wp;
        if (Placed(id) && (!lp || PlaceKey(id) != (int)lp)) return true;  // the place names another game now (a scan's result)
        if (v.menu.open && id >= ID_MENU && id < ID_MENUBOX) {
            const size_t i = (size_t)(id - ID_MENU);
            if (i < v.menu.items.size() && !v.menu.items[i].sep && v.menu.items[i].enabled) {
                v.menu.hot = (int)i;
                MenuShowItem((int)i);
                Invalidate();
            }
            return true;
        }
        FocusId(id);
        return true;
    }
    if (msg == kRevealMsg) { RevealId((int)wp); return true; }
    if (msg == kCloseMsg) {  // a dialog's Close: its cancel answer, as Escape gives it
        if (v.dialog.open) { v.dialog.result = v.dialog.cancel; v.dialog.open = false; Invalidate(); }
        return true;
    }
    if (msg != kInvokeMsg) return false;
    // checked again here: the control may have changed since Invoke was called (an operation started, a pill became progress),
    // and a Library place may name another game now (a scan's result landed first: Placed)
    const int id = (int)wp;
    const Hit* h = FindHit(id);
    if (!h || !h->enabled || (Placed(id) && (!lp || PlaceKey(id) != (int)lp))) return true;
    // as a click: the focus moves to the control pressed (no ring, unless the keyboard was on it already) and a text field
    // gives up the keys, so when the result reflows the page it stays where the user is, not with an old focus off screen
    // (an open menu's item: the focus goes back to its control, as Activate has it)
    if (v.focus != id && !(id >= ID_MENU && id < ID_MENUBOX)) { v.focus = id; v.focusVisible = false; v.editFocusId = 0; }
    if (v.live && IsChild(v.hwnd, ::GetFocus())) SetFocus(v.hwnd);
    Activate(id);
    return true;
}

}  // namespace wg::app::lui

// ---- the keyboard focus on the Library's places --------------------------------------------------------------------------
// A lockup's ids (ID_TILE, ID_TILEQA + its index) are its game's place in v.games, and the featured banner's (ID_FEAT0 + 1, + 2)
// name whichever game it shows. When the list changes (a scan's result, a game that left) or the banner takes another game
// (its own Repair finished and another game leads now), the focus stays with the game the player chose, so a second Return
// never acts on one they did not choose; also the focus kept under an open dialog (a scan that lands while it asks).
namespace wg::app::win {
using namespace wg::app::lui;

std::vector<int*> focusUnderDialogs;

namespace {
wstring PlaceKey(const GameEntry& e) { return Lower(e.dir) + L"|" + Lower(e.exe); }
int PlaceIndex(const wstring& key) {
    for (size_t i = 0; i < v.games.size() && i < (size_t)kTileSpan; ++i) if (PlaceKey(v.games[i]) == key) return (int)i;
    return -1;
}
struct HeldPlace { int* focus; int id, base; wstring game; };  // base: ID_TILE or ID_TILEQA, 0 the banner's
struct Hold { std::vector<HeldPlace> places; wstring featured; };
std::vector<Hold> holds;  // nested: a scan's result puts back the games operations finished on (ReloadEntry) in between
}  // namespace

wstring PlaceGame(int id) {
    const int i = id >= ID_TILEQA && id < ID_TILEQA + kTileSpan ? id - ID_TILEQA : id >= ID_TILE && id < ID_TILE + kTileSpan ? id - ID_TILE
                : id >= ID_FEAT0 && id < ID_FEAT0 + 3 ? v.featIndex : -1;
    return i >= 0 && i < (int)v.games.size() ? PlaceKey(v.games[(size_t)i]) : wstring();
}

void HoldPlaceFocus() {
    Hold h;
    auto add = [&](int* f) {
        const wstring game = PlaceGame(*f);
        if (game.empty()) return;
        h.places.push_back({f, *f, *f >= ID_TILEQA && *f < ID_TILEQA + kTileSpan ? ID_TILEQA : *f >= ID_TILE && *f < ID_TILE + kTileSpan ? ID_TILE : 0, game});
    };
    add(&v.focus);
    for (int* f : focusUnderDialogs) add(f);
    h.featured = PlaceGame(ID_FEAT0 + 1);
    holds.push_back(std::move(h));
}

// A lockup: its game's new place. The banner's controls stay while its game is listed (FollowPlaceGame moves the focus if
// the banner takes another game when drawn). A game that left: -1, the stop nearest to where it was (PaintAll).
void RestorePlaceFocus() {
    if (holds.empty()) return;
    const Hold hold = std::move(holds.back());
    holds.pop_back();
    for (const HeldPlace& h : hold.places) {
        const int to = PlaceIndex(h.game);
        *h.focus = to < 0 ? -1 : h.base ? h.base + to : h.id;
    }
    v.featIndex = hold.featured.empty() ? -1 : PlaceIndex(hold.featured);  // Return on the banner before the next frame: its game
}

// The same id names another game now than in the last Library frame: to that game's control of the same kind (its action:
// a lockup's pill or the banner's Play / Repair / Update; else what opens its page: the lockup or the banner's Game
// settings), UI Automation told the focus moved. Its game left the Library: -1, the nearest stop (PaintAll).
void FollowPlaceGame(int lastId, const wstring& lastGame) {
    if (!v.focus || v.focus != lastId || lastGame.empty() || PlaceGame(v.focus) == lastGame) return;
    const int gi = PlaceIndex(lastGame);
    if (gi < 0) { v.focus = -1; return; }
    const bool action = (v.focus >= ID_TILEQA && v.focus < ID_TILEQA + kTileSpan) || v.focus == ID_FEAT0 + 1, feat = gi == v.featIndex;
    const int want[2] = {feat ? (action ? ID_FEAT0 + 1 : ID_FEAT0 + 2) : (action ? ID_TILEQA : ID_TILE) + gi,
                         feat ? (action ? ID_FEAT0 + 2 : ID_FEAT0 + 1) : (action ? ID_TILE : ID_TILEQA) + gi};
    const Hit* to = nullptr;
    for (int w : want)
        for (auto& h : v.hits) if (!to && h.id == w && h.focusable && h.enabled) to = &h;
    for (auto& h : v.hits) if (!to && h.id == want[0] && h.focusable && h.role != R_TEXT) to = &h;  // busy: kept there, shown off
    if (!to) return;
    v.focus = to->id;
    revealFocus = v.focusVisible;  // the page follows a ring, not a click
    UiaFocusChanged(to->id);
}

// The control the keyboard is on moved on the page itself, not by a scroll (an operation that finished moved its game to
// another shelf: Install from Available up to With WrongGPU, Update or Repair down from Needs attention): the page follows
// it into view after the frame, so the ring is never off screen (SHIP-BAR: focus never hidden). Not while the page scrolls,
// enters or the window is resized, not for a click's focus (no ring), not for the game page's right column (it stops
// under the header as the page scrolls).
void RevealIfMoved(const Hit* at, bool samePage) {
    static int id = 0;
    static float y0 = 0, scroll0 = 0, w0 = 0, h0 = 0;
    if (!at) { id = 0; return; }
    const Page p = v.page;
    const float y = at->full.y + at->scroll;
    if (at->id == id && samePage && v.focusVisible && !at->sticky && w0 == v.W && h0 == v.H && scroll0 == v.scroll[p] && v.target[p] == v.scroll[p] &&
        Entrance(v.pageChangedAt, kPageMs) >= 1 && std::fabs(y - y0) > 1)
        revealFocus = true;
    id = at->id; y0 = y; scroll0 = v.scroll[p]; w0 = v.W; h0 = v.H;
}

// ---- the window's keyboard model: what launcher_ui.cpp's WndProc asks here first -------------------------------------------
int spacePress = 0;

void SpaceUp() {
    const int p = spacePress;
    spacePress = 0;
    if (!p || p != v.focus || v.menu.open) return;  // the focus moved away while Space was down: the press is cancelled
    for (auto& h : v.hits)
        if (h.id == p) { if (h.enabled) Activate(p); return; }
}

bool AccessMessage(UINT msg, WPARAM wp, LPARAM lp, LRESULT& r) {
    // Alt pressed and let go on its own (or let go after Alt+Left or Alt+Down, which OnKey takes, so the system saw a lone
    // Alt): no invisible system-menu mode that swallows the next key (the window draws no system menu icon to show it is
    // in it); F10 already goes to OnKey. Alt+Space (lParam ' ') still opens the window menu.
    if (msg == WM_SYSCOMMAND && (wp & 0xFFF0) == SC_KEYMENU && lp == 0) { r = 0; return true; }
    // Back from a game's page: a keyboard's Browser Back key, or a mouse or keyboard sending it as an app command (from a text
    // field too: the EDIT passes it up), as the mouse's Back button and Alt+Left
    if ((msg == WM_KEYDOWN && wp == VK_BROWSER_BACK) || (msg == WM_APPCOMMAND && GET_APPCOMMAND_LPARAM(lp) == APPCOMMAND_BROWSER_BACKWARD)) {
        if (v.page == P_GAME && !v.dialog.open && !v.first.open && !v.menu.open && IsWindowEnabled(v.hwnd)) Navigate(P_LIBRARY);
        r = msg == WM_APPCOMMAND ? TRUE : 0;
        return true;
    }
    if (msg == WM_KEYUP && wp == VK_SPACE) { SpaceUp(); r = 0; return true; }
    if (msg == WM_KILLFOCUS) spacePress = 0;  // the keys went elsewhere while Space was down: its release here presses nothing
    return false;
}

void FocusOptiField(const string& name, const wstring& why) {
    for (size_t i = 0; i < v.opti.editable.size(); ++i) {
        const OptiRow& row = v.opti.editable[i];
        if (row.section + "." + row.key != name) continue;
        const OptiKind kind = row.def ? row.def->kind : OptiKind::Number;
        const string val = i < v.optiValues.size() ? v.optiValues[i] : row.value;
        const int id = ID_G_OPTI0 + (int)i * 10 + (kind == OptiKind::Bool ? (val == "true" ? 1 : val == "false" ? 2 : 0) : 0);
        if (kind != OptiKind::Key) v.optiMore = true;  // the row shows (it is under "More OptiScaler settings")
        v.focus = id;
        v.focusVisible = true;
        revealFocus = true;  // into view after the next frame, out from under the pinned header and the unsaved capsule
        v.optiBad = View::OptiBad{v.cur.dir, i, val, why};  // its field says it is invalid until its text changes (FieldInvalidWhy)
        const HWND e = EditOf(id);
        v.editFocusId = e ? id : 0;  // a number field: its EDIT takes the keys once shown (SyncEdits), the wrong value selected
        if (e) SendMessageW(e, EM_SETSEL, 0, -1);
        UiaMarkEdit(e, why);  // said with its name as it takes the keys (before SyncEdits gives them)
        if (v.live && GetFocus() != v.hwnd) SetFocus(v.hwnd);
        Invalidate();
        return;
    }
}

// Apply from the keyboard: Ctrl+S anywhere on a game's page, Return in an OptiScaler number field (the capsule's filled
// Apply is the page's default button, as a form's Save is on Windows and macOS). The keys the field had come back to it
// once Apply has run (SyncEdits gives them when its row shows unlocked), so digits typed after go on there and Backspace
// corrects one, never leaves the page: a save shortcut never moves the keyboard out of the field being typed in. Apply
// waiting (the game runs, an operation works): said, not a silent key (its tooltip's reason).
void KeyApply() {
    if (v.page != P_GAME || v.dialog.open || v.first.open) return;
    const Hit* apply = nullptr;
    for (auto& h : v.hits) if (h.id == ID_G_APPLY) apply = &h;
    if (!apply) return;
    if (!apply->enabled) { MessageBeep(MB_OK); ShowToast(ApplyWaitsWords(v.cur), 0); return; }
    const HWND e = EditOf(v.focus);
    const int typing = e && GetFocus() == e ? v.focus : 0;
    if (GetFocus() != v.hwnd) SetFocus(v.hwnd);
    Activate(ID_G_APPLY);
    if (typing && v.focus == typing && !v.editFocusId) { v.editFocusId = typing; Invalidate(); }  // a refused value's field has them already (FocusOptiField)
}

// The toolbar's tab whose page is open: after Ctrl+1, Ctrl+2 or an arrow on the tabs, the focus goes to the tab only when its
// page opened. A game's page kept (its leave question cancelled, or Apply refused a value) keeps the focus where the dialog
// put it back, or on the field to fix (FocusOptiField).
bool OnTabPage(int tab) { return v.page == (tab == ID_NAV_SETTINGS ? P_SETTINGS : P_LIBRARY); }
}  // namespace wg::app::win
