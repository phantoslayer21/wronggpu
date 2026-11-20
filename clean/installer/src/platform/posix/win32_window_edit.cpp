// SPDX-License-Identifier: MIT
// Linux: the Win32 "EDIT" control as the launcher uses it: a single line with a caret and a selection, keyboard editing (arrows,
// Home/End, Backspace/Delete, Ctrl+A/C/X/V/Z, word moves), the mouse (click, drag, double click for a word), the clipboard, the
// messages the app sends it (WM_SETTEXT / WM_GETTEXT / WM_GETTEXTLENGTH, EM_SETSEL, EM_GETSEL, EM_REPLACESEL, EM_SETMARGINS,
// EM_GETRECT, EM_SETCUEBANNER, WM_SETFONT, WM_GETFONT, WM_CUT / WM_COPY / WM_PASTE / WM_CLEAR / WM_UNDO) and the notifications it
// expects (EN_CHANGE, EN_SETFOCUS, EN_KILLFOCUS as WM_COMMAND to the parent). The field draws nothing itself: the Linux Gfx paints
// it after the app's frame from wgwin::EditViews. Not emulated: several lines, password and number styles, drag and drop, IME
// composition, right-to-left text.
#include <algorithm>
#include <cwctype>

#include "win32_window_internal.h"

namespace wgwin {

namespace {

bool WordCh(wchar_t c) { return iswalnum((wint_t)c) || c == L'_'; }
int PrevWord(const std::wstring& t, int i) {
    while (i > 0 && !WordCh(t[(size_t)i - 1])) --i;
    while (i > 0 && WordCh(t[(size_t)i - 1])) --i;
    return i;
}
int NextWord(const std::wstring& t, int i) {
    const int n = (int)t.size();
    while (i < n && WordCh(t[(size_t)i])) ++i;
    while (i < n && !WordCh(t[(size_t)i])) ++i;
    return i;
}

void Notify(Win* w, UINT code) {
    if (w->parent) Send((HWND)w->parent, WM_COMMAND, MAKEWPARAM((WORD)(w->ctrlId & 0xFFFF), (WORD)code), (LPARAM)(HWND)w->id);
}
void Touch(Win* w) {
    w->edit->lastInputMs = NowMs();
    InvalidateWin(w);
}
int Len(Win* w) { return (int)w->text.size(); }
int Clamp(Win* w, int i) { return std::max(0, std::min(i, Len(w))); }
int SelLo(EditState& s) { return std::min(s.anchor, s.caret); }
int SelHi(EditState& s) { return std::max(s.anchor, s.caret); }

void Snapshot(Win* w) {
    EditState& s = *w->edit;
    s.undoText = w->text; s.undoAnchor = s.anchor; s.undoCaret = s.caret;
}

// the text between [lo, hi) replaced; the caret after what went in
void Replace(Win* w, int lo, int hi, const std::wstring& ins) {
    EditState& s = *w->edit;
    lo = Clamp(w, lo); hi = Clamp(w, hi);
    if (hi < lo) std::swap(lo, hi);
    std::wstring clean;
    for (wchar_t c : ins) if (c != L'\r' && c != L'\n') clean += c;  // a single line
    if (lo == hi && clean.empty()) return;
    Snapshot(w);
    w->text.replace((size_t)lo, (size_t)(hi - lo), clean);
    s.anchor = s.caret = lo + (int)clean.size();
    Touch(w);
    Notify(w, EN_CHANGE);
}

void DeleteSel(Win* w) { EditState& s = *w->edit; Replace(w, SelLo(s), SelHi(s), L""); }

void Copy(Win* w) {
    EditState& s = *w->edit;
    if (SelLo(s) != SelHi(s)) ClipSet(w->text.substr((size_t)SelLo(s), (size_t)(SelHi(s) - SelLo(s))));
}
void Paste(Win* w) {
    EditState& s = *w->edit;
    const std::wstring t = ClipGet();
    if (!t.empty()) Replace(w, SelLo(s), SelHi(s), t);
}
void Undo(Win* w) {
    EditState& s = *w->edit;
    if (s.undoText == w->text) return;
    std::swap(s.undoText, w->text);
    std::swap(s.undoAnchor, s.anchor);
    std::swap(s.undoCaret, s.caret);
    s.anchor = Clamp(w, s.anchor); s.caret = Clamp(w, s.caret);
    Touch(w);
    Notify(w, EN_CHANGE);
}

// the character boundary nearest to x (pixels from the field's left edge)
int IndexAt(Win* w, int x) {
    EditState& s = *w->edit;
    const float rel = (float)(x - s.marginL + s.scroll);
    int lo = 0, hi = Len(w);
    if (rel <= 0) return 0;
    while (lo < hi) {  // the first boundary at or past rel
        const int mid = (lo + hi) / 2;
        if (EditMeasure(w->text, s.fontPx, mid) < rel) lo = mid + 1; else hi = mid;
    }
    if (lo > 0 && lo <= Len(w)) {
        const float a = EditMeasure(w->text, s.fontPx, lo - 1), b = EditMeasure(w->text, s.fontPx, lo);
        if (rel - a < b - rel) --lo;
    }
    return Clamp(w, lo);
}

void Move(Win* w, int to, bool extend) {
    EditState& s = *w->edit;
    s.caret = Clamp(w, to);
    if (!extend) s.anchor = s.caret;
    Touch(w);
}

LRESULT Key(Win* w, WPARAM vk) {
    EditState& s = *w->edit;
    const bool shift = GetKeyState(VK_SHIFT) < 0, ctrl = GetKeyState(VK_CONTROL) < 0;
    const bool sel = SelLo(s) != SelHi(s);
    switch (vk) {
        case VK_LEFT:
            if (sel && !shift && !ctrl) Move(w, SelLo(s), false);
            else Move(w, ctrl ? PrevWord(w->text, s.caret) : s.caret - 1, shift);
            return 0;
        case VK_RIGHT:
            if (sel && !shift && !ctrl) Move(w, SelHi(s), false);
            else Move(w, ctrl ? NextWord(w->text, s.caret) : s.caret + 1, shift);
            return 0;
        case VK_HOME: Move(w, 0, shift); return 0;
        case VK_END: Move(w, Len(w), shift); return 0;
        case VK_BACK:
            if (sel) DeleteSel(w);
            else Replace(w, ctrl ? PrevWord(w->text, s.caret) : s.caret - 1, s.caret, L"");
            return 0;
        case VK_DELETE:
            if (shift && sel) { Copy(w); DeleteSel(w); }
            else if (sel) DeleteSel(w);
            else Replace(w, s.caret, ctrl ? NextWord(w->text, s.caret) : s.caret + 1, L"");
            return 0;
        case VK_INSERT:
            if (ctrl) Copy(w); else if (shift) Paste(w);
            return 0;
        case 'A': if (ctrl) { s.anchor = 0; s.caret = Len(w); Touch(w); } return 0;
        case 'C': if (ctrl) Copy(w); return 0;
        case 'X': if (ctrl) { Copy(w); DeleteSel(w); } return 0;
        case 'V': if (ctrl) Paste(w); return 0;
        case 'Z': if (ctrl) Undo(w); return 0;
        default: return 0;
    }
}

}  // namespace

void EditInit(Win* w) {
    w->edit = std::make_unique<EditState>();
    w->edit->lastInputMs = NowMs();
}

void EditSetText(Win* w, const std::wstring& text, bool notify) {
    EditState& s = *w->edit;
    if (notify) Snapshot(w);
    w->text = text;
    s.anchor = s.caret = 0;  // as a Win32 edit (and Wine's): setting the text leaves the caret at its start; the app selects
                             // all itself where it wants that (EM_SETSEL 0, -1 on the search box)
    s.scroll = 0;
    Touch(w);
    if (notify) Notify(w, EN_CHANGE);
}

LRESULT EditClassProc(Win* w, UINT m, WPARAM wp, LPARAM lp) {
    EditState& s = *w->edit;
    const HWND h = (HWND)w->id;
    switch (m) {
        case WM_GETDLGCODE: return DLGC_WANTCHARS | DLGC_HASSETSEL | DLGC_WANTARROWS;
        case WM_SETTEXT: EditSetText(w, lp ? (const wchar_t*)lp : L"", true); return TRUE;
        case WM_GETTEXT: {
            wchar_t* out = (wchar_t*)lp;
            if (!out || wp == 0) return 0;
            const size_t n = std::min<size_t>(w->text.size(), wp - 1);
            for (size_t i = 0; i < n; ++i) out[i] = w->text[i];
            out[n] = 0;
            return (LRESULT)n;
        }
        case WM_GETTEXTLENGTH: return (LRESULT)w->text.size();
        case WM_SETFONT: {
            LOGFONTW lf{};
            if (GdiFont((HFONT)wp, &lf)) { s.font = (HFONT)wp; s.fontPx = std::max(1, (int)std::abs(lf.lfHeight)); }
            if (lp) InvalidateWin(w);
            return 0;
        }
        case WM_GETFONT: return (LRESULT)s.font;
        case EM_SETMARGINS:
            if (wp & EC_LEFTMARGIN) s.marginL = (int)LOWORD(lp);
            if (wp & EC_RIGHTMARGIN) s.marginR = (int)HIWORD(lp);
            InvalidateWin(w);
            return 0;
        case EM_GETRECT:
            if (lp) *(RECT*)lp = RECT{s.marginL, 0, std::max(0, w->w - s.marginR), w->h};
            return 0;
        case EM_SETSEL: {
            const int a = (int)wp, b = (int)lp;
            if (a == -1) { s.anchor = s.caret; }
            else { s.anchor = Clamp(w, a); s.caret = b < 0 ? Len(w) : Clamp(w, b); }
            Touch(w);
            return 0;
        }
        case EM_GETSEL:
            if (wp) *(DWORD*)wp = (DWORD)SelLo(s);
            if (lp) *(DWORD*)lp = (DWORD)SelHi(s);
            return MAKELONG(SelLo(s), SelHi(s));
        case EM_REPLACESEL: Replace(w, SelLo(s), SelHi(s), lp ? (const wchar_t*)lp : L""); return 0;
        case EM_SETCUEBANNER: s.cue = lp ? (const wchar_t*)lp : L""; return TRUE;
        case EM_GETCUEBANNER: return FALSE;
        case WM_CUT: Copy(w); DeleteSel(w); return 0;
        case WM_COPY: Copy(w); return 0;
        case WM_PASTE: Paste(w); return 0;
        case WM_CLEAR: DeleteSel(w); return 0;
        case WM_UNDO: case EM_UNDO: Undo(w); return TRUE;
        case WM_SETFOCUS: s.lastInputMs = NowMs(); Notify(w, EN_SETFOCUS); InvalidateWin(w); return 0;
        case WM_KILLFOCUS: s.dragging = false; Notify(w, EN_KILLFOCUS); InvalidateWin(w); return 0;
        case WM_KEYDOWN: case WM_SYSKEYDOWN: return Key(w, wp);
        case WM_CHAR:
            if (wp >= 32 && wp != 127 && GetKeyState(VK_CONTROL) >= 0) Replace(w, SelLo(s), SelHi(s), std::wstring(1, (wchar_t)wp));
            return 0;
        case WM_SETCURSOR: SetCursor(LoadCursorW(nullptr, IDC_IBEAM)); return TRUE;
        case WM_LBUTTONDOWN: {
            SetFocus(h);
            const int i = IndexAt(w, GET_X_LPARAM(lp));
            Move(w, i, GetKeyState(VK_SHIFT) < 0);
            s.dragging = true;
            SetCapture(h);
            return 0;
        }
        case WM_LBUTTONDBLCLK: {
            const int i = IndexAt(w, GET_X_LPARAM(lp));
            int lo = i, hi = i;
            while (lo > 0 && WordCh(w->text[(size_t)lo - 1])) --lo;
            while (hi < Len(w) && WordCh(w->text[(size_t)hi])) ++hi;
            s.anchor = lo; s.caret = hi;
            Touch(w);
            return 0;
        }
        case WM_MOUSEMOVE:
            if (s.dragging) Move(w, IndexAt(w, GET_X_LPARAM(lp)), true);
            return 0;
        case WM_LBUTTONUP:
            if (s.dragging) { s.dragging = false; if (GetCapture() == h) ReleaseCapture(); }
            return 0;
        case WM_CAPTURECHANGED: s.dragging = false; return 0;
        case WM_NCDESTROY: if (g_focus == h) g_focus = nullptr; return 0;
        default: return DefWindowProcW(h, m, wp, lp);
    }
}

}  // namespace wgwin
