// SPDX-License-Identifier: MIT
// Linux: the smaller Win32 calls the launcher's window and command line make that need no SDL: the GDI stand-ins behind the text
// fields' font, the clipboard (text), the command line main() hands over, console output and a few conversions (win32_window.h).
#include <unistd.h>

#include <cmath>

#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

#include "win32_posix_internal.h"
#include "win32_window_internal.h"

// ---- GDI: a font is its LOGFONT, a brush its colour, a device context what was selected into it ------------------------------------
namespace {
struct GdiObj {
    enum Type { Font, Brush, Dc } type;
    LOGFONTW lf{};
    COLORREF color = 0, text = 0, bk = 0;
    HGDIOBJ selected = nullptr;
};
std::mutex g_gdiMu;
std::unordered_set<GdiObj*> g_gdi;
GdiObj* Obj(HANDLE h) {
    std::lock_guard<std::mutex> l(g_gdiMu);
    auto* o = static_cast<GdiObj*>(h);
    return h && g_gdi.count(o) ? o : nullptr;
}
GdiObj* NewObj(GdiObj::Type t) {
    auto* o = new GdiObj;
    o->type = t;
    std::lock_guard<std::mutex> l(g_gdiMu);
    g_gdi.insert(o);
    return o;
}
}  // namespace

namespace wgwin {
bool GdiFont(HFONT f, LOGFONTW* out) {
    GdiObj* o = Obj(f);
    if (!o || o->type != GdiObj::Font) return false;
    *out = o->lf;
    return true;
}
}  // namespace wgwin

HFONT CreateFontIndirectW(const LOGFONTW* lf) {
    GdiObj* o = NewObj(GdiObj::Font);
    if (lf) o->lf = *lf;
    return o;
}
HBRUSH CreateSolidBrush(COLORREF c) { GdiObj* o = NewObj(GdiObj::Brush); o->color = c; return o; }
BOOL DeleteObject(HGDIOBJ h) {
    GdiObj* o = Obj(h);
    if (!o) return FALSE;
    {
        std::lock_guard<std::mutex> l(g_gdiMu);
        g_gdi.erase(o);
    }
    delete o;
    return TRUE;
}
HDC GetDC(HWND) { return NewObj(GdiObj::Dc); }
int ReleaseDC(HWND, HDC dc) { DeleteObject(dc); return 1; }
HGDIOBJ SelectObject(HDC dc, HGDIOBJ o) {
    GdiObj* d = Obj(dc);
    if (!d) return nullptr;
    HGDIOBJ old = d->selected;
    d->selected = o;
    return old;
}
int GetTextFaceW(HDC dc, int n, LPWSTR out) {
    GdiObj* d = Obj(dc);
    GdiObj* f = d ? Obj(d->selected) : nullptr;
    if (!f || f->type != GdiObj::Font || !out || n <= 0) return 0;
    int i = 0;
    for (; i < n - 1 && f->lf.lfFaceName[i]; ++i) out[i] = f->lf.lfFaceName[i];
    out[i] = 0;
    return i;
}
BOOL GetTextMetricsW(HDC dc, TEXTMETRICW* tm) {
    GdiObj* d = Obj(dc);
    GdiObj* f = d ? Obj(d->selected) : nullptr;
    if (!f || !tm) return FALSE;
    const double em = std::abs((double)f->lf.lfHeight);
    // Selawik, as Segoe UI: ascent 1.08 em, descent 0.25 em
    tm->tmAscent = (LONG)std::lround(em * 1.08);
    tm->tmDescent = (LONG)std::lround(em * 0.25);
    tm->tmHeight = tm->tmAscent + tm->tmDescent;
    tm->tmInternalLeading = tm->tmExternalLeading = 0;
    tm->tmAveCharWidth = (LONG)std::lround(em * 0.55);
    return TRUE;
}
COLORREF SetTextColor(HDC dc, COLORREF c) { GdiObj* d = Obj(dc); if (!d) return 0; const COLORREF o = d->text; d->text = c; return o; }
COLORREF SetBkColor(HDC dc, COLORREF c) { GdiObj* d = Obj(dc); if (!d) return 0; const COLORREF o = d->bk; d->bk = c; return o; }
BOOL InitCommonControlsEx(const INITCOMMONCONTROLSEX*) { return TRUE; }

// ---- clipboard (text): the layer's own text is held until the next one ---------------------------------------------------------------------
namespace {
std::wstring g_pending;
std::wstring g_got;  // what GetClipboardData handed out (the system owns it: it stays valid until the next call)
}  // namespace
BOOL OpenClipboard(HWND) { return TRUE; }
BOOL EmptyClipboard() { g_pending.clear(); return TRUE; }
HANDLE SetClipboardData(UINT format, HANDLE mem) {
    if (format != CF_UNICODETEXT || !mem) return nullptr;
    g_pending = static_cast<const wchar_t*>(mem);
    wgwin::ClipSet(g_pending);
    return mem;
}
HANDLE GetClipboardData(UINT format) {
    if (format != CF_UNICODETEXT) return nullptr;
    g_got = wgwin::ClipGet();
    return (HANDLE)g_got.c_str();
}
BOOL CloseClipboard() { return TRUE; }
HGLOBAL GlobalAlloc(UINT, SIZE_T bytes) { return calloc(1, bytes ? bytes : 1); }
LPVOID GlobalLock(HGLOBAL h) { return h; }
BOOL GlobalUnlock(HGLOBAL) { return TRUE; }
HGLOBAL GlobalFree(HGLOBAL h) { free(h); return nullptr; }

// ---- the command line ------------------------------------------------------------------------------------------------------------------------
namespace {
std::wstring g_cmd;
std::wstring Quoted(const std::wstring& a) {
    if (!a.empty() && a.find_first_of(L" \t\"") == std::wstring::npos) return a;
    std::wstring o = L"\"";
    size_t bs = 0;
    for (wchar_t c : a) {
        if (c == L'\\') { ++bs; continue; }
        if (c == L'"') { o.append(bs * 2 + 1, L'\\'); o += L'"'; }
        else { o.append(bs, L'\\'); o += c; }
        bs = 0;
    }
    o.append(bs * 2, L'\\');
    return o + L"\"";
}
}  // namespace
void WgSetCommandLine(int argc, char** argv) {
    g_cmd.clear();
    for (int i = 0; i < argc; ++i) {
        if (i) g_cmd += L' ';
        g_cmd += Quoted(wgposix::FromUtf8(argv[i], strlen(argv[i])));
    }
}
LPWSTR GetCommandLineW() { return g_cmd.empty() ? const_cast<wchar_t*>(L"") : g_cmd.data(); }
// the way CommandLineToArgvW reads it: the first word has no escapes; then backslashes before a quote are halved
LPWSTR* CommandLineToArgvW(LPCWSTR cmd, int* argc) {
    std::vector<std::wstring> args;
    const wchar_t* p = cmd ? cmd : L"";
    if (*p == L'"') { ++p; std::wstring a; while (*p && *p != L'"') a += *p++; if (*p) ++p; args.push_back(a); }
    else { std::wstring a; while (*p && *p != L' ' && *p != L'\t') a += *p++; args.push_back(a); }
    for (;;) {
        while (*p == L' ' || *p == L'\t') ++p;
        if (!*p) break;
        std::wstring a;
        bool quoted = false;
        while (*p && (quoted || (*p != L' ' && *p != L'\t'))) {
            size_t bs = 0;
            while (*p == L'\\') { ++bs; ++p; }
            if (*p == L'"') {
                a.append(bs / 2, L'\\');
                if (bs % 2) a += L'"';
                else if (quoted && p[1] == L'"') { a += L'"'; ++p; }
                else quoted = !quoted;
                ++p;
            } else { a.append(bs, L'\\'); if (*p) a += *p++; }
        }
        args.push_back(a);
    }
    size_t chars = 0;
    for (auto& a : args) chars += a.size() + 1;
    auto* block = static_cast<unsigned char*>(malloc(sizeof(wchar_t*) * (args.size() + 1) + sizeof(wchar_t) * chars));
    if (!block) return nullptr;
    auto** v = reinterpret_cast<wchar_t**>(block);
    auto* s = reinterpret_cast<wchar_t*>(block + sizeof(wchar_t*) * (args.size() + 1));
    for (size_t i = 0; i < args.size(); ++i) {
        v[i] = s;
        memcpy(s, args[i].c_str(), sizeof(wchar_t) * (args[i].size() + 1));
        s += args[i].size() + 1;
    }
    v[args.size()] = nullptr;
    if (argc) *argc = (int)args.size();
    return v;
}
BOOL SetDefaultDllDirectories(DWORD) { return TRUE; }

// ---- console ----------------------------------------------------------------------------------------------------------------------------------
BOOL AttachConsole(DWORD) { return isatty(1) || isatty(2) ? TRUE : FALSE; }
BOOL WriteConsoleW(HANDLE h, const void* buf, DWORD n, LPDWORD written, LPVOID) {
    const std::string u = wgposix::ToUtf8(static_cast<const wchar_t*>(buf), n);
    DWORD w = 0;
    const BOOL ok = WriteFile(h, u.data(), (DWORD)u.size(), &w, nullptr);
    if (written) *written = n;
    return ok;
}
double _wtof(const wchar_t* s) { return wcstod(s, nullptr); }
int MulDiv(int a, int b, int c) { return c ? (int)std::llround((double)a * b / c) : -1; }
