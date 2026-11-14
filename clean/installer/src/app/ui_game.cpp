// SPDX-License-Identifier: MIT
// WrongGPU launcher window: a game's page (ui_shell.h) in option E, "Quiet": "< Library", then the game's header (its
// portrait cover from the player's Steam cache or a generated plate, the title, the status in words with its dot, the
// line under it, Play as the one filled capsule and More as a grey one with its menu). Once the actions scroll away a
// compact header stays under the toolbar (Back, a small cover, the name, the status, the same actions). Then two columns
// of inset grouped lists: what can be set on the left (Upscaler: the profile's Basic settings, the master switch first;
// OptiScaler: WrongGPU's setup, the FPS overlay, the editable keys this build has, the keys WrongGPU manages, outside
// edits), what is read on the right (Last run: a read-out from the log, warnings in plain language, the raw lines behind
// "Show"; Game folder: what else is there and what WrongGPU does about it). While settings differ from the game's profile
// a capsule floats over the page's bottom on blurred material (Discard, Apply). Written for this project.
#include "app/ui_shell.h"

#include "wg_core/checks.h"
#include "wg_core/gamerules.h"
#include "wg_core/rrconfig.h"

namespace wg::app::lui {

// "2026-10-17 19:19" -> "Today at 19:19", "Yesterday at ...", "Oct 14 at 16:26"
wstring When(const wstring& w) {
    if (w.size() < 16) return w;
    SYSTEMTIME now; GetLocalTime(&now);
    wchar_t today[16], yest[16];
    swprintf(today, 16, L"%04u-%02u-%02u", now.wYear, now.wMonth, now.wDay);
    FILETIME ft; SystemTimeToFileTime(&now, &ft);
    ULARGE_INTEGER u; u.LowPart = ft.dwLowDateTime; u.HighPart = ft.dwHighDateTime; u.QuadPart -= 864000000000ULL;
    ft.dwLowDateTime = u.LowPart; ft.dwHighDateTime = u.HighPart;
    SYSTEMTIME y; FileTimeToSystemTime(&ft, &y);
    swprintf(yest, 16, L"%04u-%02u-%02u", y.wYear, y.wMonth, y.wDay);
    const wstring day = w.substr(0, 10), time = w.substr(11, 5);
    // no-break spaces: a wrapped line never splits "Oct 17" or "at 19:19"
    if (day == today) return L"Today at\u00a0" + time;
    if (day == yest) return L"Yesterday at\u00a0" + time;
    static const wchar_t* const mon[] = {L"Jan", L"Feb", L"Mar", L"Apr", L"May", L"Jun", L"Jul", L"Aug", L"Sep", L"Oct", L"Nov", L"Dec"};
    const int m = _wtoi(w.substr(5, 2).c_str());
    return (m >= 1 && m <= 12 ? wstring(mon[m - 1]) : w.substr(5, 2)) + L"\u00a0" + std::to_wstring(_wtoi(w.substr(8, 2).c_str())) + L" at\u00a0" + time;
}

// WrongGPU's Ray Reconstruction can't run in this game with this build: the game's RR needs WrongGPU's OptiScaler edition
// (gamerules.h: the host restores the game's command-list state) and this build has none, so the RR plugin reports RR
// unsupported there and the game keeps its own ray tracing (install.cpp warns as it installs)
static bool RrBlocked(const GameEntry& e) { const GameRule* r = GameRuleFor(e.exe); return (r && r->needsHostRestore && !v.rrEdition) || (e.rrDirect && !v.rrEdition); }
// A game that creates Ray Reconstruction itself (launcher.h rrDirect) installed with an OptiScaler that is not WrongGPU's
// edition (the player's own download, used when the edition couldn't be downloaded): no OptiScaler there hands the game's RR
// to WrongGPU's plugin, so the card says so and shows no switch (a repair with the edition makes it work)
static bool RrNoEdition(const GameEntry& e) { return e.rrDirect && e.hasRecord && e.rrInstalled && !e.optiEdition && !RrBlocked(e); }
// WrongGPU's Ray Reconstruction takes RR in this game: its files are in, it is on, the RR data is ready, and this build can
// run it there
bool RrOn(const GameEntry& e) { return e.rrInstalled && e.rrEnabled && v.rrReady && !RrBlocked(e) && !RrNoEdition(e); }
bool RrAloneLastRun(const GameEntry& e) { return RrOn(e) && e.last.found && (e.last.rrOnly || e.last.rrUpscaled); }
static bool RrFailedRun(const LastRun& lr);  // (the Last run section, r7 #2)
wstring RrModeName(const wstring& mode) { return mode == L"exact" ? L"Exact" : mode == L"fast" ? L"Fast" : L"Accurate"; }

namespace {
TS Ts(float size, int weight, Col c, FontFace f = F_TEXT) { TS s; s.size = size; s.weight = weight; s.color = c; s.face = f; return s; }
TS Wrap(float size, Col c, float lineH) { TS s; s.size = size; s.color = c; s.wrap = true; s.lineH = lineH; return s; }

const char* const kExec[] = {"auto", "dx12", "hip"};
const char* const kFilter[] = {"default", "original", "analytic", "quality5", "quality9"};
// one name per choice wherever it shows (the row, its menu, the outside-change note), and what it does in a line: the
// menu item's second line and help text, and the row's sentence for the chosen one
// (the profile's quality5 / quality9 by what they look like: a number would read as a level on a scale)
const wchar_t* const kFilterNames[] = {L"Default", L"Original", L"Classic", L"Even", L"Even (wide)"};
const wchar_t* const kFilterHelp[] = {L"WrongGPU's standard look. Recommended.", L"The filter read from your DLSS file.", L"WrongGPU's earlier filter.",
                                      L"Spaces fine detail more evenly. No extra cost.", L"Like Even, with a wider filter. A little slower."};
const char* const kPre[] = {"scale", "original"};
const char* const kFps[] = {"keep", "on", "off"};
template <size_t N> int Index(const char* const (&vals)[N], const string& x) { for (size_t i = 0; i < N; ++i) if (x == vals[i]) return (int)i; return 0; }

bool Rec() { return v.cur.hasRecord; }
// an operation (Install, Update, Repair, Apply...) waits: one runs at a time, and none while the game runs
bool Locked() { return v.busy || v.listCached || v.cur.status == GameStatus::Running; }  // listCached: the last session's list (BusyWhy)
// the settings can be changed: WrongGPU is in the game, nothing that conflicts runs (PlayBlocked: the game itself, its files
// in use; an operation in it or queued for it; a task that changes every game) and the game is still there. Another game's
// operation leaves them free: Apply then waits for it, and says so.
// pageReading: the page opened before its files were read (OpenGame on a slow disk): the values shown are not the game's yet
bool CanEdit() { return Rec() && !PlayBlocked(v.cur) && v.cur.status != GameStatus::Orphaned && !v.pageReading; }
// A repair of an older WrongGPU is an update too (launcher_ui.cpp RepairUpdates, PrepareOp's question "Repair and update"):
// its button and menu item say so from the start, and its busy words are "Repairing and updating…"
bool RepairUpdatesHere(const GameEntry& e) {
    return UpdateDue(e, Ctx());  // an older version, or another build of this one (rc5)
}

// A profile key and value in the page's own words ("sparkle_guard", "on" -> "Sparkle guard", "On"), for the note about
// values changed outside the launcher.
wstring SettingLabel(const string& key) {
    static const std::pair<const char*, const wchar_t*> k[] = {{"wronggpu", L"Use WrongGPU"}, {"executor", L"Runs on"}, {"output_filter", L"Output filter"},
                                                              {"pre_exposure", L"Pre-exposure"}, {"sparkle_guard", L"Sparkle guard"}, {"fps_overlay", L"FPS overlay"}};
    for (auto& x : k) if (key == x.first) return x.second;
    return FromUtf8(key);
}
wstring SettingWords(const string& key, const string& value) {
    static const std::pair<const char*, const wchar_t*> w[] = {{"on", L"On"}, {"off", L"Off"}, {"auto", L"Auto"}, {"dx12", L"Direct3D 12"}, {"hip", L"HIP"},
                                                              {"keep", L"Default"}, {"scale", L"Scale"}, {"original", L"Original"}, {"default", L"Default"},
                                                              {"analytic", L"Classic"}, {"quality5", L"Even"}, {"quality9", L"Even (wide)"}};
    (void)key;
    for (auto& x : w) if (value == x.first) return x.second;
    return FromUtf8(value);
}

// ---- the rows of an inset grouped list (the mockup's .srow, .kv, .act) -------------------------------------------------------
constexpr float kLabelW = 84;   // a read-out's label column (.kv .k), 16 to its value
constexpr float kPinH = 60;     // the compact header pinned under the toolbar
constexpr float kColGap = 28;   // between the two columns (.gcols)

// A coloured tag's words (amber "Your own OptiScaler", "Asks first") a little toward the text colour: on the tag's
// grey fill the plain tokens are only just 4.5:1 in light; this keeps them clear of the limit (5.5:1 or more, both themes).
Col TagText(Col c) { return c.r == pal.text2.r && c.g == pal.text2.g && c.b == pal.text2.b ? c : Mix(c, pal.text, 0.18f); }

// The hairline at a row's top (none on a list's first row): it starts at the text.
void RowLine(Gfx& gx, float x, float y, float w, bool first) { if (!first) Hairline(gx, x + kCellPad, y, x + w); }

// A settings row: the name (+ a tag), a wrapped sentence under it, the control on the right (drawn by `ctrl` in its box,
// centred on the row). `dim`: the row cannot be changed now (before an install, a game that is running or gone, an
// operation in it): its name, its sentence and its tag in text3 (3:1 or better, as its disabled control), so the whole row
// reads as off, as the dimmed pills and buttons do (not a bright sentence under a dim name).
float SRow(Gfx& gx, float x, float y, float w, bool first, const wstring& title, const wstring& desc, float ctrlW, float ctrlH,
           const std::function<void(const Rc&)>& ctrl, bool dim, const wstring& tag = L"") {
    RowLine(gx, x, y, w, first);
    const float tx = x + kCellPad, tw = w - 2 * kCellPad - ctrlW - 20;
    const TS ds = Wrap(12.5f, dim ? pal.text3 : pal.text2, 17);
    const wstring sentence = NoWidow(desc);
    const float dh = desc.empty() ? 0 : gx.TextH(sentence, tw, ds);
    // the name's line 20 high and the sentence 1 under it, as Settings' rows have them: one rhythm for the inset lists of both
    // pages (a one-line row 60 high on each)
    const float textH = 20 + (desc.empty() ? 0 : 1 + dh);
    const float h = std::max(kSRowH, std::max(textH, ctrlH) + 22);
    const float ty = y + std::round((h - textH) / 2);
    const TS ns = Ts(14, 600, dim ? pal.text3 : pal.text);
    const float nw = std::min(tw, gx.TextW(title, ns) + 1);
    gx.Text(title, Rc{tx, ty, nw + 1, 20}, ns);
    if (!tag.empty()) Tag(gx, tx + nw + 8, ty + 10, tag, dim ? pal.text3 : pal.text2, pal.fill);
    if (!desc.empty()) gx.Text(sentence, Rc{tx, ty + 21, tw, dh + 2}, ds);
    const size_t before = v.hits.size();
    ctrl(Rc{x + w - kCellPad - ctrlW, y + std::round((h - ctrlH) / 2), ctrlW, ctrlH});
    // the sentence (and the tag: "+0.11 ms") for UI Automation: the help text of the row's control (each segment's too)
    const wstring help = desc + (tag.empty() ? wstring() : (desc.empty() ? wstring() : wstring(L" ")) + L"(" + tag + L")");
    for (size_t i = before; i < v.hits.size() && !help.empty(); ++i) if (!Passive(v.hits[i].role)) v.hits[i].help = help;
    return h;
}

// A folder that does not fit: whole folders dropped from its middle ("G:\...\Cyberpunk 2077\bin\x64": the drive and as many
// of the last folders as fit), else shortened in the middle by characters.
wstring ShortPath(Gfx& gx, const wstring& path, float width, const TS& s) {
    if (gx.TextW(path, s) <= width) return path;
    std::vector<wstring> parts;
    size_t at = 0;
    for (size_t i; (i = path.find(L'\\', at)) != wstring::npos; at = i + 1) parts.push_back(path.substr(at, i - at));
    parts.push_back(path.substr(at));
    for (size_t keep = parts.size() > 2 ? parts.size() - 2 : 0; keep >= 1; --keep) {
        wstring t = parts[0] + L"\\\u2026";
        for (size_t i = parts.size() - keep; i < parts.size(); ++i) t += L"\\" + parts[i];
        if (gx.TextW(t, s) <= width) return t;
    }
    return FitMiddle(gx, path, width, s);
}

// A read-out row: the label in text2, the value in the text colour (a path in mono, shortened in the middle; anything else
// wraps). Optional: a status dot before the value, a short note after it in text2 (under it when it does not fit), a tag
// and a sentence under it, a link (Show) at the right end.
struct KvOpt {
    bool mono = false;
    Col dot{0, 0, 0, 0};
    wstring suffix, tag, sub;
    Col tagFg{0, 0, 0, 0};
    int linkId = 0;
    wstring link, linkName;
    int textId = 0;  // the read-out's id for UI Automation (a path: its whole text, also when it is shortened)
    wstring tip;     // the hover tooltip and the end of its UI Automation name (a folder's whole path, a file's name)
};
float KvRow(Gfx& gx, float x, float y, float w, bool first, const wstring& label, const wstring& value, const KvOpt& o = KvOpt()) {
    RowLine(gx, x, y, w, first);
    const float lx = x + kCellPad, vx = lx + kLabelW + 16;
    const float linkW = o.link.empty() ? 0.f : ButtonW(gx, o.link, 0, 28, B_LINK) - 6 + 12;
    const float vw = x + w - kCellPad - linkW - vx;
    const float dx = o.dot.a > 0 ? 15.f : 0.f;
    TS vs = Ts(13.5f, 600, pal.text);
    if (o.mono) { vs.face = F_MONO; vs.size = 11.5f; vs.weight = 400; }
    // a path, or any mono value (a file name): one line, shortened in the middle (a file name never breaks at its hyphens)
    const bool path = o.mono || (value.size() > 2 && value[1] == L':' && value[2] == L'\\');
    const TS ss = Ts(13.5f, 400, pal.text2);
    wstring sub = o.sub;
    bool inlineSuffix = false;
    float vh = 19;
    wstring shown = value;
    vs.lineH = 19;  // every value on the label's baseline (mono too: the same uniform line)
    if (path) shown = ShortPath(gx, value, vw - dx, vs);
    else {
        if (!o.suffix.empty()) inlineSuffix = gx.TextW(value, vs) + 8 + gx.TextW(o.suffix, ss) <= vw - dx;
        if (!o.suffix.empty() && !inlineSuffix) sub = sub.empty() ? o.suffix : o.suffix + L"\n" + sub;  // the note leads the line(s) under it
        vs.wrap = true;
        vh = std::max(19.f, gx.TextH(value, vw - dx, vs));
    }
    const TS subS = Wrap(12.5f, pal.text2, 17);
    const float tagH = o.tag.empty() ? 0.f : 24.f;
    const float subH = sub.empty() ? 0.f : 2 + gx.TextH(sub, vw, subS);
    const float content = vh + tagH + subH;
    const float h = std::max(kKvH, content + 24);
    const float ty = y + std::round((h - content) / 2);
    gx.Text(label, Rc{lx, ty, kLabelW, 19}, Ts(13.5f, 400, pal.text2));
    if (dx > 0) StatusDot(gx, vx + 3.5f, ty + 10, o.dot);
    if (path) {
        TS ps = vs; ps.wrap = false;
        gx.Text(shown, Rc{vx + dx, ty, vw - dx, 19}, ps);
        if (o.textId) AddHit(o.textId, Rc{vx + dx, ty, std::min(vw - dx, gx.TextW(shown, ps) + 2), 19}, R_TEXT, label + L": " + value, true, -1, L"", false);
    }
    else gx.Text(value, Rc{vx + dx, ty, vw - dx, vh + 2}, vs);
    if (inlineSuffix) { TS is = ss; is.lineH = 19; gx.Text(o.suffix, Rc{vx + dx + gx.TextW(value, vs) + 8, ty, vw - dx, 19}, is); }
    float cy = ty + vh;
    if (!o.tag.empty()) { Tag(gx, vx, cy + 13, o.tag, o.tagFg.a > 0 ? TagText(o.tagFg) : pal.text2, pal.fill); cy += tagH; }
    if (!sub.empty()) gx.Text(sub, Rc{vx, cy + 2, vw, subH + 2}, subS);
    if (!(path && o.textId)) {  // the read-out for UI Automation (a path with a textId has its own), before the link that takes the click
        wstring spoken = o.tip;  // the tooltip's lines as sentences
        for (size_t at; (at = spoken.find(L'\n')) != wstring::npos;) spoken.replace(at, 1, L". ");
        // each part a sentence, one period between them (a part that ends with its own is not given a second: "clash.. File")
        wstring said = (label.empty() ? L"" : label + L": ") + value + (o.suffix.empty() ? L"" : L" " + o.suffix) + (o.tag.empty() ? L"" : L", " + o.tag);
        const wstring* const parts[] = {&o.sub, &spoken};
        for (const wstring* part : parts)
            if (!part->empty()) said += (said.empty() ? L"" : said.back() == L'.' ? L" " : L". ") + *part;
        const int tid = AddText(Rc{lx, ty, x + w - kCellPad - linkW - lx, content}, said, o.textId);
        if (tid && !o.tip.empty()) v.tips[tid] = o.tip;
    }
    if (!o.link.empty()) Button(gx, o.linkId, x + w - kCellPad + 6, ty + 10 - 14, o.link, B_LINK, 0, true, 28, 0, true, o.linkName.empty() ? o.link : o.linkName);
    return h;
}

// A sentence in a row: a status dot (or none) and the words, wrapped. `tip`: the hover tooltip (the files it is about),
// also read after the words by UI Automation.
float DotRow(Gfx& gx, float x, float y, float w, bool first, const wstring& text, Col dot, Col textCol, const wstring& tip = L"") {
    RowLine(gx, x, y, w, first);
    const float tx = x + kCellPad + (dot.a > 0 ? 15.f : 0.f), tw = x + w - kCellPad - tx;
    const TS ts = Wrap(13.5f, textCol, 19);
    const float th = gx.TextH(text, tw, ts);
    const float h = std::max(kKvH, th + 24);
    const float ty = y + std::round((h - th) / 2);
    if (dot.a > 0) StatusDot(gx, x + kCellPad + 3.5f, ty + 10, dot);
    gx.Text(text, Rc{tx, ty, tw, th + 2}, ts);
    wstring spoken = tip;  // the tooltip's lines as sentences
    for (size_t at; (at = spoken.find(L'\n')) != wstring::npos;) spoken.replace(at, 1, L". ");
    const int tid = AddText(Rc{tx, ty, tw, th + 2}, text + (spoken.empty() ? L"" : L" " + spoken + (spoken.back() == L'.' ? L"" : L".")));
    if (tid && !tip.empty()) v.tips[tid] = tip;
    return h;
}

// An action row (.act): the whole row is the button, its glyph at the right; the hover fill meets the list's corners.
float ActRow(Gfx& gx, float x, float y, float w, bool first, bool last, int id, const wstring& text, wchar_t glyph, bool enabled, Role role = R_BUTTON,
             const wstring& name = L"") {
    const Rc r{x, y, w, kActH};
    if (enabled && Hot(id)) RowFill(gx, r, first, last, Down(id));
    RowLine(gx, x, y, w, first);
    TS ts = Ts(13.5f, 400, enabled ? pal.text : pal.text3); ts.valign = 1;
    gx.Text(text, Rc{x + kCellPad, y, w - 2 * kCellPad - 24, kActH}, ts);
    gx.Icon(glyph, x + w - kCellPad - 6, r.cy() + (glyph == ic::down || glyph == ic::up ? 1.f : 0.f), glyph == ic::external || glyph == ic::zip ? 12.f : 10.f, enabled ? pal.text2 : pal.text3);
    FocusRing(gx, r.in(5), 6, id);
    AddHit(id, r, role, name.empty() ? text : name, enabled);
    return kActH;
}
// Collect logs (Last run's last row): it waits while an operation runs (it reads the game's files), and says why in its
// tooltip and for UI Automation, in More's words for the same item (a disabled row is no Tab stop: Narrator gets the reason)
static float CollectRow(Gfx& gx, float x, float y, float w) {
    const float h = ActRow(gx, x, y, w, false, true, ID_G_COLLECT, L"Collect logs", ic::zip, !v.busy);
    if (v.busy) {
        const wstring why = v.cur.status == GameStatus::Running ? L"Waits until " + BusyDoneWords() + L"." : BusyWhy(v.cur);
        v.tips[ID_G_COLLECT] = why;
        Describe(ID_G_COLLECT, why);
    }
    return h;
}

// A row that needs you instead of a section's controls: a status dot and the name in its colour (red: needs you), a
// sentence under it, one button on the right (B_TINTED: the action that needs you, a red pill as the Library's and the
// header's, so the page keeps Play as its one filled button). `working`: its task runs (the busy indicator before the
// button, as Settings has it; the button grey and disabled, saying what it does).
float NeedRow(Gfx& gx, float x, float y, float w, bool first, const wstring& title, const wstring& desc, Col dot, Col titleCol,
              int id, const wstring& btn, BtnKind kind, bool enabled, const wstring& btnName, bool working = false) {
    RowLine(gx, x, y, w, first);
    if (working) { kind = B_PLAIN; enabled = false; }
    const float bw = ButtonW(gx, btn, 0, 30, kind) + (working ? 22.f : 0.f);
    const float tx = x + kCellPad + 15, tw = x + w - kCellPad - bw - 20 - tx;
    const TS ds = Wrap(12.5f, pal.text2, 17);
    const wstring sentence = NoWidow(desc);
    const float dh = desc.empty() ? 0 : gx.TextH(sentence, tw, ds);
    const float textH = 20 + (desc.empty() ? 0 : 1 + dh);
    const float h = std::max(kSRowH, std::max(textH, 30.f) + 22);
    const float ty = y + std::round((h - textH) / 2);
    StatusDot(gx, x + kCellPad + 3.5f, ty + 10, dot);
    gx.Text(title, Rc{tx, ty, tw, 20}, Ts(14, 600, titleCol));
    if (!desc.empty()) gx.Text(sentence, Rc{tx, ty + 21, tw, dh + 2}, ds);
    AddText(Rc{tx, ty, tw, textH}, title + (desc.empty() ? L"" : L". " + desc));
    const float bx = x + w - kCellPad - Button(gx, id, x + w - kCellPad, y + std::round((h - 30) / 2), btn, kind, 0, enabled, 30, 0, true, btnName);
    if (working) BusyIndicator(gx, bx - 6 - 8, y + h / 2, 7, pal.text2);
    return h;
}

// A note about values changed outside the launcher, with its two ways back to one set of values inside it (as a failure's
// note keeps its Dismiss): the amber-tinted cell, the glyph, the words, and under them the buttons (30 high).
float ChoiceCallout(Gfx& gx, float x, float y, float w, const wstring& text, const std::vector<std::pair<int, wstring>>& btns, bool en) {
    const TS ts = Wrap(13, pal.text, 18);
    const float avail = w - 40 - 16;
    const float th = gx.TextH(text, avail, ts);
    const Rc r{x, y, w, 11 + th + 10 + 30 + 13};
    gx.FillRound(r, kGroupR, Mix(pal.cell, pal.amber, pal.dark ? 0.12f : 0.08f));  // the lists' radius
    gx.Icon(ic::warn, r.x + 21, r.y + 20, 13, pal.amberText);
    gx.Text(text, Rc{r.x + 40, r.y + 11, avail, th + 4}, ts);
    AddText(Rc{r.x + 40, r.y + 11, avail, th + 4}, text);
    // both write the game's files: not while an operation (another game's too) or Collect logs runs, nor while the game
    // runs; disabled then, and the tooltip says what they wait for, as Apply's does
    const bool wait = en && Locked();
    float bx = r.x + 40;
    for (auto& b : btns) {
        bx += Button(gx, b.first, bx, r.y + 11 + th + 10, b.second, B_PLAIN, 0, en && !wait, 30) + 8;
        if (wait) { v.tips[b.first] = BusyWhy(v.cur); Describe(b.first, BusyWhy(v.cur)); }
    }
    return r.h;
}

// Raw lines (the log's notes, the managed keys) in mono, inside a list under the row they belong to (.lognote).
float MonoLines(Gfx& gx, float x, float y, float w, const wstring& lines) {
    TS ms = Wrap(12, pal.text2, 17); ms.face = F_MONO;
    const float mh = gx.TextH(lines, w - 2 * kCellPad, ms);
    gx.Text(lines, Rc{x + kCellPad, y, w - 2 * kCellPad, mh + 4}, ms);
    AddText(Rc{x + kCellPad, y, w - 2 * kCellPad, mh + 4}, lines);
    return mh + 12;
}

// A section's heading with a tag after it (OptiScaler's "WrongGPU's setup"); returns 22.
float HeadingTag(Gfx& gx, float x, float y, float w, const wstring& title, const wstring& tag, Col tagFg) {
    GroupHeading(gx, x, y, w, title);
    if (!tag.empty()) {
        const float tx = x + 4 + gx.TextW(title, Ts(17, 600, pal.text, F_DISPLAY)) + 10;
        AddText(Rc{tx, y + 2, Tag(gx, tx, y + 12, tag, TagText(tagFg), pal.fill), 20}, tag);
    }
    return 22;
}

wstring KeyCodeName(int code) {
    struct K { int c; const wchar_t* n; };
    static const K keys[] = {{0x2D, L"Insert"}, {0x24, L"Home"}, {0x23, L"End"}, {0x2E, L"Delete"}, {0x21, L"Page Up"}, {0x22, L"Page Down"},
                             {0x13, L"Pause"}, {0x91, L"Scroll Lock"}, {0x70, L"F1"}, {0x71, L"F2"}, {0x72, L"F3"}, {0x73, L"F4"}, {0x74, L"F5"},
                             {0x75, L"F6"}, {0x76, L"F7"}, {0x77, L"F8"}, {0x78, L"F9"}, {0x79, L"F10"}, {0x7A, L"F11"}, {0x7B, L"F12"}};
    for (auto& k : keys) if (k.c == code) return k.n;
    // a key the menu does not list (set in OptiScaler.ini): its name on this keyboard, never a raw code ("Key 0x2A"); the key
    // under Esc (VK_OEM_3) too, which is a different character on each layout
    if (code > 0 && code < 0xFF) {
        const UINT sc = MapVirtualKeyW((UINT)code, MAPVK_VK_TO_VSC_EX);
        if (sc & 0xFF) {
            const LONG lp = (LONG)((sc & 0xFF) << 16) | ((sc & 0xFF00) == 0xE000 || (sc & 0xFF00) == 0xE100 ? (1L << 24) : 0L);  // extended
            wchar_t n[64] = {};
            if (GetKeyNameTextW(lp, n, 64) > 0) return n;
        }
    }
    return L"Another key";
}

// The proxy planner's reason in plain words (the full reason stays in the CLI and the logs).
wstring PlanLine(const wstring& why) {
    const size_t taken = why.find(L"taken: ");
    wstring line;
    if (why.rfind(L"the first free name", 0) == 0) line = L"The usual name; nothing else in this game uses it.";
    else if (why.rfind(L"the name of WrongGPU's earlier install", 0) == 0) line = L"The same name as WrongGPU's earlier install here.";
    else if (why.rfind(L"where your OptiScaler was", 0) == 0) line = L"Where your own OptiScaler was.";
    else if (taken != wstring::npos && why.find(L',', taken) == wstring::npos) line = L"The usual name is taken by another file in this game.";
    else line = L"The usual names are taken by other files in this game.";
    if (why.find(L"untested combination") != wstring::npos) line += L" Next to ReShade or Special K this is untested.";
    return line;
}

wstring RunsOnName(const wstring& executor) { return executor == L"dx12" ? L"Direct3D 12" : ExecutorName(executor); }

// A folder as a place in words (LIVE-FINDINGS 15: no paths as body text): from the game's own folder in a Steam library
// ("...\steamapps\common\Cyberpunk 2077\bin\x64" -> "Cyberpunk 2077 › bin › x64"), else its last three folders. The whole
// path goes to the row's tooltip and its UI Automation name.
wstring Place(const wstring& path) {
    std::vector<wstring> parts;
    size_t at = 0;
    for (size_t i; (i = path.find_first_of(L"\\/", at)) != wstring::npos; at = i + 1) if (i > at) parts.push_back(path.substr(at, i - at));
    if (at < path.size()) parts.push_back(path.substr(at));
    if (!parts.empty() && parts[0].size() == 2 && parts[0][1] == L':') parts.erase(parts.begin());  // the drive
    size_t from = parts.size() > 3 ? parts.size() - 3 : 0;
    for (size_t i = 0; i + 1 < parts.size(); ++i) if (IEq(parts[i], L"common") && i > 0 && IEq(parts[i - 1], L"steamapps")) from = i + 1;
    wstring s;
    for (size_t i = from; i < parts.size(); ++i) s += (s.empty() ? L"" : L" › ") + parts[i];
    return s.empty() ? path : s;
}

// The file name OptiScaler loads as, in words, read after the row's label "Installs as" ("Installed as" once it is in; the
// file name itself and the planner's reason go to the tooltip): "The usual file name", "The same name as your OptiScaler", ...
constexpr const wchar_t* kUsualName = L"The usual file name";
constexpr const wchar_t* kOtherName = L"Another free file name";
wstring PlanName(const wstring& why, const wstring& proxy) {
    if (why.rfind(L"the first free name", 0) == 0 || (why.empty() && IEq(proxy, kProxyOrder[0]))) return kUsualName;
    if (why.rfind(L"the name of WrongGPU's earlier install", 0) == 0) return L"WrongGPU's earlier file name";
    if (why.rfind(L"where your OptiScaler was", 0) == 0) return L"The same name as your OptiScaler";
    return kOtherName;
}
// A key WrongGPU's OptiScaler setup manages, by what it does for the player (the core's reasons, in a few words)
wstring ManagedWords(const string& section, const string& key) {
    struct M { const char* sec; const char* key; const wchar_t* words; };
    static const M kWords[] = {
        {"Upscalers", "Dx12Upscaler", L"Upscaling goes to WrongGPU"}, {"Libraries", "FfxDx12SRPath", L"Upscaling goes to WrongGPU"},
        {"FSR", "UpscalerIndex", L"Upscaling goes to WrongGPU"}, {"FSR", "Fsr4Update", L"OptiScaler doesn't replace WrongGPU's file"},
        {"CAS", "Enabled", L"No extra sharpening"}, {"Sharpness", "OverrideSharpness", L"No extra sharpening"},
        {"OutputScaling", "Enabled", L"Render size is left to the game"}, {"UpscaleRatio", "UpscaleRatioOverrideEnabled", L"Render size is left to the game"},
        {"Plugins", "LoadAsiPlugins", L"WrongGPU's plugin loads with OptiScaler"}, {"Plugins", "Path", L"WrongGPU's plugin loads with OptiScaler"},
        {"Log", "LogToFile", L"OptiScaler keeps a log for support"}, {"Log", "LogLevel", L"OptiScaler keeps a log for support"},
        {"Hotfix", "MipmapBiasOverride", L"Texture detail is left to WrongGPU"}, {"Menu", "ShowFps", L"The FPS overlay as set above"}};
    for (auto& m : kWords) if (_stricmp(m.sec, section.c_str()) == 0 && _stricmp(m.key, key.c_str()) == 0) return m.words;
    return L"Another setting WrongGPU needs";  // never a raw ini key on the surface (its line is in the tooltip)
}

// The line under it: why another name ("The usual name is taken by another file in this game."), and an untested pair
wstring PlanSub(const wstring& why, const wstring& proxy) {
    const bool untested = why.find(L"untested combination") != wstring::npos;
    if (PlanName(why, proxy) == kOtherName && !why.empty()) return PlanLine(why);
    return untested ? L"Next to ReShade or Special K this is untested." : L"";
}
// The row is worth showing: the usual name is taken, so OptiScaler takes another one (the line under it says why), or the
// pair is untested. Taking the name of the player's OptiScaler or of WrongGPU's earlier install is how a switch or an update
// goes (the section's sentence says it): the file name stays in its tooltip, not on the surface.
bool PlanNotable(const wstring& why, const wstring& proxy) { return PlanName(why, proxy) == kOtherName || !PlanSub(why, proxy).empty(); }
// The tooltip of that row: the file, and what the row does not say already ("dxgi.dll. Nothing else in this game uses it.").
// The name (PlanName) and the line under it (PlanSub: why another name) are on the row, and UI Automation reads the row and
// its tooltip as one, so they are not said twice ("The usual one. dxgi.dll. The usual name; ...").
wstring PlanTip(const wstring& why, const wstring& proxy) {
    const wstring more = why.rfind(L"the first free name", 0) == 0 ? L"Nothing else in this game uses it." : L"";
    return proxy.empty() ? more : more.empty() ? proxy : proxy + L". " + more;
}

// What else is in the game folder, in the player's words (SHIP-BAR 4; LIVE-FINDINGS 12): the row's name without its file
// ("ReShade 6.3.3", not "ReShade 6.3.3 as dxgi.dll") and a sentence for what WrongGPU does about it. The core's own text
// (the files, its message) goes to the tooltip and UI Automation; it stays as it is for the CLI, the logs and the installer.
struct FindingWords { wstring name, sub, tip; };
FindingWords WordsOf(const Finding& f) {
    FindingWords w;
    w.name = f.what;
    const size_t as = w.name.rfind(L" as ");  // "ReShade 6.3.3 as dxgi.dll": the name only
    if (as != wstring::npos && as > 0 && w.name.find(L'.', as) != wstring::npos) w.name.resize(as);
    const bool untested = f.message.find(L"untested with WrongGPU") != wstring::npos;
    switch (f.tool) {
        case Tool::OptiScaler:
            if (f.decision == Decision::AskSwitch) {
                w.sub = L"Backed up when you switch, and put back when WrongGPU is uninstalled.";
                if (f.message.find(L"older OptiScaler layout") != wstring::npos) w.sub += L" It's an older OptiScaler setup.";
                else if (untested) w.sub += L" This version isn't tested with WrongGPU.";
            } else if (f.decision == Decision::Refuse) {
                w.name = L"Two OptiScaler copies";
                w.sub = L"This game loads OptiScaler twice. Remove one of them yourself, then install WrongGPU.";
            }
            break;
        case Tool::ReShade: case Tool::SpecialK:
            if (f.decision == Decision::Chain) w.sub = L"It stays. If Model M falls back to AMD FSR next to it, Last run shows it.";  // the tag says untested
            else if (f.decision == Decision::Inform) { w.name = wstring(ToolName(f.tool)) + L" settings"; w.sub = wstring(ToolName(f.tool)) + L" itself isn't here. Nothing to do."; }
            else if (f.decision == Decision::Warn) w.sub = L"It may load into this game too. That pair isn't tested with WrongGPU.";
            break;
        case Tool::Dxvk: case Tool::Vkd3d:
            w.sub = L"It turns the game's graphics into Vulkan, and WrongGPU needs Windows' own Direct3D 12.";
            break;
        case Tool::AsiLoader: w.sub = L"It stays. WrongGPU keeps its plugin in a folder of its own, so it starts once."; break;
        case Tool::Cet: case Tool::Red4ext: case Tool::Ue4ss: w.sub = L"It stays. WrongGPU doesn't touch it."; break;
        case Tool::HipRuntime: w.name = L"Another HIP tool"; w.sub = L"Keep Runs on at Auto or Direct3D 12 so the two don't clash."; break;
        case Tool::NrOnAmd: w.name = L"Another upscaler mod"; w.sub = L"WrongGPU doesn't install next to it and doesn't touch it."; break;
        case Tool::VendorDll:
            if (f.what.rfind(L"A loose ", 0) == 0) { w.name = L"Another tool's file"; w.sub = L"WrongGPU leaves it alone and never writes over it."; }
            else { w.name = L"A copy of the game's DLSS file"; w.sub = L"From an older OptiScaler setup. WrongGPU leaves it alone."; }
            break;
        case Tool::FakeNvapi: w.name = ToolName(f.tool); w.sub = L"WrongGPU leaves it alone and never writes over it."; break;
        case Tool::ReFramework:  // the player's mod framework; a game that needs it for OptiScaler (RE Requiem) says so
            w.name = ToolName(f.tool);
            w.sub = f.message.find(L"needs it") != wstring::npos ? L"It stays. This game needs it for WrongGPU." : L"It stays. WrongGPU doesn't touch it.";
            break;
        case Tool::WrongGPULegacy: w.name = L"An older WrongGPU"; w.sub = L"Install removes it first, with your OK, and puts the game's own files back."; break;
        case Tool::GameApi: {
            const size_t p = f.what.find(L'(');
            const wstring api = p == wstring::npos ? wstring() : f.what.substr(p + 1, f.what.size() - p - 2);
            w.name = L"Not a Direct3D 12 game";
            w.sub = api.empty() ? L"WrongGPU works in Direct3D 12 games only." : L"It uses " + api + L", and WrongGPU works in Direct3D 12 games only.";
            break;
        }
        case Tool::AntiCheat: {  // the anti-cheat by its name (the Library's words), its marker in the tooltip
            GameEntry probe; probe.status = GameStatus::Refused; probe.statusText = L"Refused: " + f.what;
            const wstring s = ShortStatus(probe);
            w.name = s.rfind(L"Anti-cheat: ", 0) == 0 ? s.substr(12) : L"Anti-cheat";
            w.sub = L"WrongGPU stays out of games with anti-cheat.";
            break;
        }
        case Tool::Unknown:
            if (f.decision == Decision::Warn) { w.name = L"An unknown Direct3D 12 file"; w.sub = L"Left alone. If the game fails with WrongGPU, this file may be why."; }
            else { w.name = L"An unknown file"; w.sub = L"Left alone. WrongGPU's files take other names."; }
            break;
        default: w.sub = f.message; break;  // overlays: the core's sentences are the player's already ("Its overlay works next to WrongGPU.")
    }
    if (w.sub.empty()) w.sub = f.message;
    // the tooltip: its files, then the core's own sentence when the row says it in other words
    wstring files;
    for (auto& x : f.files) files += (files.empty() ? L"" : L", ") + x;
    w.tip = files.empty() ? wstring() : (f.files.size() > 1 ? L"Files: " : L"File: ") + files;
    // anti-cheat: where its marker is, once; the core's sentence names the same path again and the install policy, which the
    // row and the card above say in the player's words
    if (f.tool == Tool::AntiCheat) {
        if (!f.evidence.empty()) w.tip = L"Found: " + f.evidence;
        return w;
    }
    // (not a HIP tool's: the core's sentence is in its own terms, executors and proxy names, and the row says it in words)
    if (w.sub != f.message && !f.message.empty() && f.tool != Tool::HipRuntime) w.tip += (w.tip.empty() ? L"" : L"\n") + f.message;
    return w;
}

// WrongGPU's files that are missing or changed, counted ("2 of WrongGPU's files are missing or changed", "One of WrongGPU's
// files was changed"; no period): the header's line and the Game folder row. *count: how many; *tip: their names ("Missing:
// dxgi.dll\nChanged: ..."). Empty when none (the settings file and other problems have their own words).
wstring FilesProblem(const GameEntry& e, size_t* count, wstring* tip) {
    std::vector<wstring> missing, changed;
    for (auto& p : e.problems) {
        if (p.rfind(L"missing: ", 0) == 0) missing.push_back(p.substr(9));
        else if (p.rfind(L"changed: ", 0) == 0) changed.push_back(p.substr(9));
    }
    const size_t n = missing.size() + changed.size();
    if (count) *count = n;
    if (tip) {
        auto list = [](const std::vector<wstring>& f) { wstring o; for (auto& x : f) o += (o.empty() ? L"" : L", ") + x; return o; };
        *tip = missing.empty() ? wstring() : L"Missing: " + list(missing);
        if (!changed.empty()) *tip += (tip->empty() ? L"" : L"\n") + (L"Changed: " + list(changed));
    }
    if (n == 0) return L"";
    const wstring how = changed.empty() ? (n == 1 ? L"is missing" : L"are missing") : missing.empty() ? (n == 1 ? L"was changed" : L"were changed")
                                                                                                      : L"are missing or changed";
    return (n == 1 ? wstring(L"One of WrongGPU's files ") : std::to_wstring(n) + L" of WrongGPU's files ") + how;
}
}  // namespace

// What the running operation is waiting on, in words for "... once <this>" ("the update is done"; another game's:
// "Quarry Nine's update is done"): the game page's caption, and leaving it with unsaved changes (SHIP-BAR 4: no
// "operation").
wstring BusyDoneWords() {
    static const std::pair<const wchar_t*, const wchar_t*> kNoun[] = {
        {L"Installing", L"install"}, {L"Updating", L"update"}, {L"Repairing", L"repair"}, {L"Uninstalling", L"uninstall"},
        {L"Rolling back", L"rollback"}, {L"Cleaning up", L"clean-up"}, {L"Removing files", L"clean-up"}, {L"Adopting", L"adoption"}, {L"Resetting", L"reset"},
        {L"Setting up", L"set-up"}};  // the RR row's Set up (a repair that adds Ray Reconstruction)
    switch (v.task) {  // a task that changes every game or the model (PlayBlocked holds the page for it too), or Collect logs
        case T_COLLECT: return L"the logs are collected";
        case T_MODEL: return L"the Model M data is converted";
        case T_RRDATA: return L"the Ray Reconstruction data is converted";
        case T_SELFINSTALL: return L"WrongGPU is installed for this user";
        case T_FIRSTRUN: return v.first.installOffer && v.first.choice == 1 ? L"WrongGPU is installed for this user" : L"WrongGPU is set up";  // portable: no install
        case T_SELFUNINSTALL: return L"the launcher is uninstalled";
        default: break;
    }
    // the last session's list (BusyWhy): writes wait for the scan, or for a Rescan after one that failed
    if (!v.busy && v.listCached) return v.scanning ? L"WrongGPU finishes looking for games" : L"the games are read again with Rescan";
    wstring who;
    if (!IEq(v.busyDir, v.cur.dir)) for (auto& e : v.games) if (IEq(e.dir, v.busyDir)) { who = e.name; break; }
    if (v.busyVerb.rfind(L"Applying", 0) == 0) return who.empty() ? L"the settings are written" : who + L"'s settings are written";
    for (auto& n : kNoun)
        if (v.busyVerb.rfind(n.first, 0) == 0) return (who.empty() ? L"the " : who + L"'s ") + n.second + L" is done";
    return who.empty() ? L"WrongGPU is done" : L"WrongGPU is done with " + who;
}

const int kMenuKeys[20] = {0x2D, 0x24, 0x23, 0x2E, 0x21, 0x22, 0x13, 0x91, 0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x7B};
wstring KeyName(const string& value) {
    if (value.empty() || value == "auto") return L"Auto (Insert)";
    char* end = nullptr;
    const long c = strtol(value.c_str(), &end, 0);
    if (!end || *end) return FromUtf8(value);
    return KeyCodeName((int)c);
}

// Every write of the page's values waits by one rule (Apply on the capsule, Apply from leaving the page or from Play, the
// outside-change notes' buttons): Locked(): an operation, the last session's list, the game running
bool PageWritesWait() { return Locked(); }
// A write of the page that failed says "Your changes are still on the page" only when it carried changes made there (Apply);
// a callout's Keep these values / Use WrongGPU's values with nothing changed on the page has none to keep
bool PendingEdits() {
    const auto& p = v.pendingApply;
    if (p.edit.v != p.saved.v) return true;
    for (size_t i = 0; i < p.rows.size() && i < p.opti.size(); ++i)
        if (p.opti[i] != p.rows[i].value) return true;
    return false;
}

bool Dirty() {
    if (!Rec()) return false;
    if (v.edit.v != v.saved.v) return true;
    for (size_t i = 0; i < v.opti.editable.size() && i < v.optiValues.size(); ++i)
        if (v.optiValues[i] != v.opti.editable[i].value) return true;
    return false;
}

// How many settings differ from the game's profile (the unsaved capsule says it).
static int DirtyCount() {
    if (!Rec()) return 0;
    int n = 0;
    for (auto& kv : v.edit.v) { const auto it = v.saved.v.find(kv.first); n += it == v.saved.v.end() || it->second != kv.second; }
    for (auto& kv : v.saved.v) n += v.edit.v.find(kv.first) == v.edit.v.end();
    for (size_t i = 0; i < v.opti.editable.size() && i < v.optiValues.size(); ++i) n += v.optiValues[i] != v.opti.editable[i].value;
    return n;
}

// The output filter's choices for its menu: the name, what it does on a second line (and as its help text), the chosen one
// checked.
std::vector<MenuItem> FilterMenu() {
    std::vector<MenuItem> items;
    const string cur = v.edit.get("output_filter");
    for (int i = 0; i < 5; ++i) {
        MenuItem it{kFilterNames[i], M_FILTER0 + i, true, false, false, cur == kFilter[i]};
        it.help = kFilterHelp[i];
        items.push_back(it);
    }
    return items;
}

std::vector<MenuItem> MoreMenu() {
    const GameEntry& e = v.cur;
    const bool lock = Locked();
    // why the items that are off are off, each reason once: a note at the menu's end (a separator with words, ui_widgets.cpp)
    // and the help text of those items for UI Automation (SHIP-BAR 1: busy states say so in words; a disabled item takes no
    // keyboard focus, and a menu shows no tooltips)
    std::vector<wstring> why;
    std::vector<std::vector<wstring>> whyOf;  // the items each reason is about
    auto off = [&](bool enabled, const wstring& reason, const wstring& item) {
        if (enabled || reason.empty()) return enabled;
        const size_t k = (size_t)(std::find(why.begin(), why.end(), reason) - why.begin());
        if (k == why.size()) { why.push_back(reason); whyOf.emplace_back(); }
        whyOf[k].push_back(item);
        return enabled;
    };
    std::vector<MenuItem> m;
    m.push_back({L"Open folder", M_OPEN_FOLDER, true, false, false, false, ic::folder});
    // no log yet: unavailable, the note says so
    m.push_back({L"Open log", M_OPEN_LOG, off(e.hasRecord && e.last.found, e.hasRecord ? L"There's no log yet. It appears after the game runs with WrongGPU." : L"", L"Open log"),
                 false, false, false, ic::doc});
    // Reset OptiScaler settings (WrongGPU's OptiScaler setup here): an operation, it waits as the page's others do, and on a
    // game that is gone it is off for the reason Reset upscaler settings is (put after it below)
    const bool optiReset = v.opti.have && e.hasRecord;
    const wstring optiWhy = e.status == GameStatus::Orphaned ? L"The game is gone, so its settings no longer apply." : lock ? BusyWhy(e) : wstring();
    // waits while an operation writes a game's files
    // (a running game's reason is about its actions: this waits for the operation)
    const wstring collectWhy = !v.busy ? wstring() : e.status == GameStatus::Running ? L"Waits until " + BusyDoneWords() + L"." : BusyWhy(e);
    m.push_back({L"Collect logs", M_COLLECT, off(!v.busy, collectWhy, L"Collect logs"), false, false, false, ic::zip});
    MenuItem sep; sep.sep = true;
    std::vector<MenuItem> ops;
    const wstring lockWhy = lock ? BusyWhy(e) : wstring();
    // on an older WrongGPU a repair is an update too: named so from the start, as its question and its busy words say
    if (e.hasRecord && e.status != GameStatus::Orphaned)
        ops.push_back({RepairUpdatesHere(e) ? L"Repair and update" : L"Repair", M_REPAIR, off(!lock, lockWhy, L"Repair"), false, false, false, ic::wrench});
    if (e.hasRecord && e.hasPrevious) ops.push_back({L"Roll back to " + e.previous + L"\u2026", M_ROLLBACK, off(!lock, lockWhy, L"Roll back"), false, false, false, ic::undo});  // read with the entry
    // sentence case, as "Reset OptiScaler settings…" after it (no ellipsis: it only shows the defaults, Apply writes them)
    if (e.hasRecord) {
        const wstring resetWhy = e.status == GameStatus::Orphaned ? L"The game is gone, so its settings no longer apply."
                               : v.pageReading ? L"Waits until the game's settings are read." : BusyWhy(e);
        ops.push_back({L"Reset upscaler settings", M_RESET, off(CanEdit(), resetWhy, L"Reset upscaler settings"), false, false, false, ic::refresh});
    }
    if (optiReset)  // a reset, as the Upscaler's (undo is Roll back's)
        ops.push_back({L"Reset OptiScaler settings\u2026", M_OPTI0 - 1, off(!lock && e.status != GameStatus::Orphaned, optiWhy, L"Reset OptiScaler settings"), false, false, false, ic::refresh});
    if (!ops.empty()) { m.push_back(sep); for (auto& o : ops) m.push_back(o); }
    // a game that is gone: Remove files, as its header button says (Uninstall would read as uninstalling the game)
    if (e.hasRecord) {
        m.push_back(sep);
        m.push_back({e.status == GameStatus::Orphaned ? L"Remove WrongGPU's files\u2026" : L"Uninstall WrongGPU\u2026", M_UNINSTALL, off(!lock, lockWhy, e.status == GameStatus::Orphaned ? L"Remove WrongGPU's files" : L"Uninstall WrongGPU"), true, false, false, ic::trash});
    }
    // a game added with Add game, without WrongGPU in it: taken out of the Library again (a wrong exe picked by mistake);
    // with WrongGPU in it the launcher keeps managing it (Uninstall first, then this)
    else if (e.source == L"Added") {
        m.push_back(sep);
        m.push_back({L"Remove from Library\u2026", M_REMOVEGAME, off(!(v.busy && IEq(v.busyDir, e.dir)), BusyWhy(e), L"Remove from Library"), false, false, false, ic::minus});
    }
    if (!why.empty()) {
        // a tooltip's "Waits until ..." given its subject: the item it is about, or "Actions" for several (the header's word
        // while the game runs: "Actions wait until it closes")
        MenuItem note; note.sep = true;
        for (size_t i = 0; i < why.size(); ++i) {
            wstring w = why[i];
            if (w.rfind(L"Waits until ", 0) == 0) {
                const wstring name = whyOf[i].size() == 1 ? whyOf[i][0] : wstring();
                w = name.empty() ? L"Actions wait" + w.substr(5) : name + L" waits" + w.substr(5);
            }
            note.help += (note.help.empty() ? L"" : L"\n") + w;
        }
        m.push_back(note);
    }
    return m;
}

// ---- the header ----------------------------------------------------------------------------------------------------------------
// The header's actions, left to right: the one that leads is filled: Play in the accent (the player's pick) where WrongGPU
// is in the game, Install in the tint where it is not yet (IDENTITY.md, 2026-11-01: it turns WrongGPU on, the page's one
// filled button), otherwise the action that needs you in red (Update, Repair, Adopt, Clean up, Uninstall a game that is
// gone). The others are grey, and so is Play in a game without WrongGPU (the tint would say it is on there) or in one that
// needs a tool of the player's first (REFramework: the callout under the header leads there, with Learn more and Check
// again); Adopt as the alternative to Clean up is grey too (one lead per page). More (drawn by MoreButton) ends the row.
// While an operation runs on this game, the button that started it says what it is doing ("Update to 0.5.5" ->
// "Updating…"), as the Library's lockup does. Its width is kept from the longer of the two (HeadW), so Play and More
// never move while it runs.
struct HeadBtn { int id; wstring text; BtnKind kind; bool en; wchar_t icon; wstring idle; };
// The verb an action's operation runs under (StartOp's words): its button takes it while that operation runs.
static const wchar_t* BusyVerbOf(const wstring& action) {
    if (action == L"Install") return L"Installing\u2026";
    if (action.rfind(L"Update", 0) == 0) return L"Updating\u2026";
    if (action == L"Repair") return L"Repairing\u2026";
    if (action == L"Repair and update") return L"Repairing and updating\u2026";  // an older WrongGPU's repair (StartOp)
    if (action == L"Uninstall") return L"Uninstalling\u2026";
    if (action == L"Remove files") return L"Removing files\u2026";
    if (action == L"Clean up") return L"Cleaning up\u2026";
    if (action == L"Adopt") return L"Adopting\u2026";
    return L"";
}
// A header button's width: the longer of its words at rest and its busy verb ("Install" / "Installing…")
static float HeadW(Gfx& gx, const HeadBtn& b, float h) {
    const wchar_t* verb = BusyVerbOf(b.idle);
    return std::max(ButtonW(gx, b.text, b.icon, h, b.kind), *verb ? ButtonW(gx, verb, b.icon, h, b.kind) : 0.f);
}
static bool BusyHere() { return v.busy && IEq(v.busyDir, v.cur.dir); }
static std::vector<HeadBtn> HeadButtons() {
    const GameEntry& e = v.cur;
    wstring main, second;
    switch (e.status) {
        case GameStatus::Eligible: case GameStatus::PlayersOptiScaler: case GameStatus::OldInstall: main = L"Install"; break;
        case GameStatus::UpdateAvailable: main = L"Update to " + wstring(Ctx().data->version); break;
        case GameStatus::NeedsRepair: main = RepairUpdatesHere(e) ? L"Repair and update" : L"Repair"; break;
        case GameStatus::Orphaned: main = L"Remove files"; break;  // what a game that is gone needs, in red: WrongGPU's files out of its old folder
        case GameStatus::Leftovers: main = L"Clean up"; second = L"Adopt"; break;
        case GameStatus::NeedsPrereq: if (!e.hasRecord) main = L"Install"; break;  // held back (disabled) until the tool is there
        case GameStatus::Installed: case GameStatus::TurnedOff: case GameStatus::Running: if (e.adoptable) main = L"Adopt"; break;
        default: break;
    }
    // no Play for a game that is gone; Play waits (disabled) while an operation runs or the game is running already
    const bool play = gl.global.playButton && e.status != GameStatus::Orphaned && (v.fixture || e.exeFound);  // read with the entry, never per frame
    const bool lock = Locked();
    const bool playLeads = play && e.hasRecord && e.status != GameStatus::NeedsRepair && e.status != GameStatus::NeedsPrereq;
    const bool playFilled = playLeads;
    std::vector<HeadBtn> b;
    const bool install = e.status == GameStatus::Eligible || e.status == GameStatus::PlayersOptiScaler || e.status == GameStatus::OldInstall ||
                         (e.status == GameStatus::NeedsPrereq && !e.hasRecord);
    const bool heldBack = e.status == GameStatus::NeedsPrereq && !e.hasRecord;  // Install waits for the player's tool (REFramework): grey, disabled
    // the button of the operation running here says its verb (only that one: Play and the others just wait)
    auto words = [](const wstring& t) { const wchar_t* verb = BusyVerbOf(t); return BusyHere() && *verb && v.busyVerb == verb ? wstring(verb) : t; };
    // Play waits only for what conflicts with starting this game (PlayBlocked), not for another game's operation
    const bool playOn = !PlayBlocked(e);
    if (play && !playOn) v.tips[ID_G_PLAY] = BusyWhy(e);
    // grey Play in a game WrongGPU isn't in yet: what it starts (not "play with WrongGPU" beside Install)
    else if (play && !e.hasRecord) v.tips[ID_G_PLAY] = L"Starts the game without WrongGPU.";
    if (playFilled) b.push_back({ID_G_PLAY, L"Play", B_GO, playOn, ic::playLine});
    // beside a filled Play, what needs you is a red pill (the accent's tint), so the header keeps one filled button. An update
    // is routine, never red (UI review #3): the accent's fill on its own, a grey (glass on the art) button beside Play
    const bool update = e.status == GameStatus::UpdateAvailable;
    const BtnKind mainKind = heldBack ? B_SECONDARY : install ? B_GO : update ? (playLeads ? B_SECONDARY : B_GO) : playLeads ? B_TINTED : B_PRIMARY;
    if (!main.empty()) b.push_back({ID_G_PRIMARY, words(main), mainKind, !lock && !heldBack, 0, main});
    if (!main.empty() && lock && !BusyHere()) v.tips[ID_G_PRIMARY] = BusyWhy(e);  // waiting for another operation: why, in words
    if (heldBack) v.tips[ID_G_PRIMARY] = L"Waits until " + e.needs + L" is installed in this game.";
    if (play && !playFilled) b.push_back({ID_G_PLAY, L"Play", B_SECONDARY, playOn, ic::playLine});
    if (!second.empty()) b.push_back({ID_G_SECOND, words(second), B_SECONDARY, !lock, 0, second});  // the alternative to the lead: grey
    if (!second.empty() && lock) v.tips[ID_G_SECOND] = BusyWhy(e);
    return b;
}
// The header's disabled actions say why to UI Automation too (their help text), not only in the hover tooltip: a disabled
// button is no Tab stop, so neither Narrator nor the keyboard would ever get the tooltip. After the actions are drawn.
static void DescribeHead() {
    for (int id : {ID_G_PLAY, ID_G_PRIMARY, ID_G_SECOND}) {
        const auto it = v.tips.find(id);
        if (it != v.tips.end()) Describe(id, it->second);
    }
}

// More: a grey capsule, its label and a small chevron after it (.bigmore); pressed while its menu is open. glass: on the
// banner's art (.gwmore: translucent white, white words).
static float MoreButton(Gfx& gx, float x, float y, float h, bool alignRight, bool glass = false) {
    const TS ts0 = Ts(h >= 34 ? 14.f : 13.f, 600, glass ? pal.onArt : pal.text);
    const float tw = gx.TextW(L"More", ts0);
    const float w = std::ceil(16 + tw + 8 + 10 + 15);
    if (alignRight) x -= w;
    const Rc r{x, y, w, h};
    const bool open = v.menu.open && v.menu.owner == ID_G_MORE;
    if (glass) gx.FillRound(r, h / 2, Alpha(pal.white, Down(ID_G_MORE) || open ? 0.32f : Hot(ID_G_MORE) ? 0.26f : 0.18f));
    else gx.FillRound(r, h / 2, Down(ID_G_MORE) || open ? Mix(pal.fillHover, pal.text, 0.06f) : Hot(ID_G_MORE) ? pal.fillHover : pal.fill);
    if (glass) ArtHcEdge(gx, r, h / 2, Hot(ID_G_MORE) || open); else HcEdge(gx, r, h / 2, Hot(ID_G_MORE) || open);  // a Contrast theme: its edge
    TS ts = ts0; ts.valign = 1;
    gx.Text(L"More", Rc{r.x + 16, r.y, tw + 2, h}, ts);
    gx.Icon(ic::down, r.x + 16 + tw + 8 + 5, r.cy() + 1, 10, glass ? pal.onArt : pal.text2);
    FocusRing(gx, r, h / 2, ID_G_MORE);
    AddHit(ID_G_MORE, r, R_BUTTON, L"More", true);
    return w;
}

// While an operation runs on this game, the status line says what it does, verb first (its button's words: "Updating…"),
// then the step it is on in the secondary colour ("Copying WrongGPU's files (3 of 7)"), cut with an ellipsis where the
// line ends. Before the first step (the step is the verb itself) or after the last ("Done"), the verb alone.
static wstring BusyVerb() { return v.busyVerb.empty() ? wstring(L"Working\u2026") : v.busyVerb; }
// the whole line in words (UI Automation: the status text and the progress bar's name; the step is never cut there)
static wstring BusyWords() {
    const wstring verb = BusyVerb();
    return !v.step.empty() && v.step != verb ? verb + L" " + v.step : verb;
}
static void BusyLine(Gfx& gx, const Rc& r, const TS& verbS, TS stepS) {
    const wstring verb = BusyVerb();
    const float vw = std::min(r.w, gx.TextW(verb, verbS) + 1);
    TS vs = verbS; vs.trim = true;
    gx.Text(verb, Rc{r.x, r.y, vw + 1, r.h}, vs);
    const bool step = !v.step.empty() && v.step != verb;
    AddText(r, BusyWords(), ID_G_STATUS);
    if (!step || vw + 6 >= r.w) return;
    stepS.trim = true;
    gx.Text(v.step, Rc{r.x + vw + 6, r.y, r.w - vw - 6, r.h}, stepS);
}

// The status in words with its dot, and the line under it (what it means, the store). The status text itself is for the
// CLI and the logs. One typographic rule wherever the header shows them (the mockup's .gstat / .gsub): the status semibold,
// what follows it regular (the cover header: two lines; the banner: one line, the status first); the compact header's
// status line is the secondary line under the name there, all regular.
// said: big in UI Automation's words when they differ ("WrongGPU. Worked last time you played": the words shown, without the dot)
struct HeadStatus { Col dot; wstring big, sub, said; bool update = false; };  // update: an up arrow, not the dot (UpdateMark)
static void StatusMark(Gfx& gx, float cx, float cy, const HeadStatus& st, Col words) {
    if (st.update) UpdateMark(gx, cx, cy, words);
    else StatusDot(gx, cx, cy, st.dot);
}
static HeadStatus Status() {
    const GameEntry& e = v.cur;
    HeadStatus s;
    s.dot = pal.green;
    float fill = 0, cross = 0;
    BrandNow(fill, cross);
    switch (e.status) {
        case GameStatus::Installed:
            // the mark flips to RightGPU on a healthy game's page (BrandNow); the words keep the product's name (UI review, 0.6.0).
            // "Worked" only after a run where Model M upscaled a frame, the checklist's test (GuideShown, r2 #10): the runtime logs
            // "ready" before any frame, so a run that ends there gets the plain installed line, never "worked" beside a checklist
            // that says it hasn't yet (r2b #2). Ray Reconstruction's frames count where it did the upscaling (AnyUpscaled, r6 #1)
            if (cross >= 0.5f && e.last.AnyUpscaled() && e.last.upscaler != L"amd") {
                s.big = L"WrongGPU \u00b7 Worked last time you played";
                s.said = L"WrongGPU. Worked last time you played";
            } else s.big = L"WrongGPU " + e.installedVersion + L" is installed";
            if (RrFailedRun(e.last)) s.dot = pal.amber;  // Ray Reconstruction finished no frame in the last run: Last run's amber row says what to do (r7 #2)
            break;
        case GameStatus::TurnedOff: s.big = L"WrongGPU " + e.installedVersion + L" is installed and turned off"; s.dot = pal.greyDot; break;
        // an update ready is routine, not a state that worked or failed: an up arrow in the line's colour, as the Library's
        // banner has it (UI review #3; r2 #11: the accent's dot was the healthy green by default)
        case GameStatus::UpdateAvailable: s.big = L"Update ready: " + InstalledLabel(e) + L" \u2192 " + Ctx().data->version; s.dot = pal.greyDot; s.update = true; break;
        case GameStatus::NeedsRepair: s.big = L"WrongGPU " + e.installedVersion + L" needs a repair"; s.dot = pal.amber; break;
        case GameStatus::Running: s.big = L"The game is running"; break;
        case GameStatus::Refused: s.big = L"Not supported in this game"; s.dot = pal.greyDot; break;
        case GameStatus::ForeignMod: s.big = L"Another upscaler mod is in this game"; s.dot = pal.greyDot; break;
        case GameStatus::Eligible: case GameStatus::PlayersOptiScaler: s.big = L"Ready to install"; s.dot = pal.greyDot; break;
        case GameStatus::OldInstall: s.big = L"An older WrongGPU is here"; s.dot = pal.amber; break;
        case GameStatus::Leftovers: s.big = L"Leftover WrongGPU files"; s.dot = pal.amber; break;
        case GameStatus::Orphaned: s.big = L"The game is gone"; s.dot = pal.amber; break;
        // a tool of the player's comes first (REFramework for RE Requiem): amber, a warning in words, never red (Install
        // is held back, so nothing here is an action that needs you)
        case GameStatus::NeedsPrereq: s.big = L"Needs " + e.needs; s.dot = pal.amber; break;
        default: { bool lit = true; s.big = e.statusText; s.dot = StatusLed(e, lit); break; }
    }
    // the line under it: what it means, then the store after a middle dot ("Model M on Direct3D 12 · Steam"); a line that
    // says what to do is a sentence of its own, without the store (it would read as part of the instruction)
    bool instruction = false, rrLine = false;
    const bool adopt = e.adoptable && (e.status == GameStatus::Installed || e.status == GameStatus::TurnedOff);
    if (adopt) s.dot = pal.amber;
    if (e.status == GameStatus::Running) { s.sub = e.hasRecord ? L"Actions wait until it closes" : L"Install waits until it closes"; instruction = true; }
    else if (e.status == GameStatus::TurnedOff && !adopt) s.sub = L"AMD FSR upscales";  // the Library's words, never the last run's Model M
    else if (e.hasRecord) {
        // one pattern: the upscaler ("Model M on HIP", "AMD FSR"), a middle dot, then the store ("Model M on Direct3D 12 ·
        // Steam"); with WrongGPU's Ray Reconstruction on, "Model M and Ray Reconstruction" in their place, the Library
        // banner's words: its setup (the
        // line never says it ran where Last run says "Not used"), and as long as the plain line, so on the banner it stays
        // on the scrim's dark part, where white words read 4.5:1 (rr-notes.md)
        s.sub = e.last.found && e.last.upscaler == L"amd" ? L"AMD FSR" : L"Model M";
        // where it runs: the player's choice in Runs on once they made one (Direct3D 12, HIP; "auto" leaves it to the card),
        // else where the last run ran
        const string runsOn = v.pageReading ? string() : v.saved.get("executor");
        // the last run was Ray Reconstruction alone (r5 #4), or RR did the upscaling in it (r6 #1): Model M had no part in it
        if (RrAloneLastRun(e)) {
            s.sub = L"Ray Reconstruction";
            rrLine = true;
        } else if (RrOn(e) && !(e.last.found && e.last.upscaler == L"amd")) {
            s.sub += L" and Ray Reconstruction";
            rrLine = true;
        } else if (!(e.last.found && e.last.upscaler == L"amd") && (runsOn == "dx12" || runsOn == "hip")) s.sub += L" on " + RunsOnName(FromUtf8(runsOn));
        else if (e.last.ready && !e.last.executor.empty()) s.sub += L" on " + RunsOnName(e.last.executor);
        if (e.status == GameStatus::NeedsRepair) {  // a problem, not followed by the store; files in the Game folder row's words
            const wstring files = FilesProblem(e, nullptr, nullptr);
            s.sub = files.empty() || !e.realExe.empty() ? ProblemShort(e) : files;  // in the wrong place first (rc6 L2)
            instruction = true;
        }
        if (e.status == GameStatus::Orphaned) { s.sub = L"Remove files deletes what WrongGPU left in its folder"; instruction = true; }
        // the player's tool gone since the install: the same words as before an install (the callout says what to do)
        if (e.status == GameStatus::NeedsPrereq) { s.sub = L"WrongGPU can't load in the game without it"; instruction = true; }
        // where WrongGPU came from and what Adopt does, short enough for the banner's dark part ("it": the game's install,
        // brought into the launcher; never the launcher itself)
        if (adopt) { s.sub = L"Set up outside the launcher: Adopt brings it in"; instruction = true; }
    } else {
        s.sub = e.status == GameStatus::Eligible ? (e.statusText.find(L"experimental") != wstring::npos ? L"Not tested yet" : L"Tested with WrongGPU")
              : e.status == GameStatus::PlayersOptiScaler ? L"Install asks to switch to WrongGPU's OptiScaler (yours is backed up)"
              : e.status == GameStatus::OldInstall ? L"Install replaces it"
              : e.status == GameStatus::Leftovers ? L"Clean up or adopt them"
              : e.status == GameStatus::NeedsPrereq ? L"WrongGPU can't load in the game without it" : L"";  // "the game": one name per page
        instruction = e.status == GameStatus::PlayersOptiScaler || e.status == GameStatus::OldInstall || e.status == GameStatus::Leftovers ||
                      e.status == GameStatus::NeedsPrereq;
    }
    if (!instruction && !rrLine) s.sub += (s.sub.empty() ? L"" : L" \u00b7 ") + StoreOf(e);
    return s;
}

// The status and the line under it as sentences for UI Automation: each middle dot ends one, so every part keeps its
// capital ("Model M on Direct3D 12. Your folder.", not ", Your folder")
static wstring Spoken(const HeadStatus& st) {
    wstring sub = st.sub;
    for (size_t at; (at = sub.find(L" \u00b7 ")) != wstring::npos;) sub.replace(at, 3, L". ");
    const wstring s = (st.said.empty() ? st.big : st.said) + (sub.empty() ? L"" : L". " + sub);
    return s.empty() || s.back() == L'.' ? s : s + L".";
}

// The header from `top` (content DIPs): "< Library" on the column's edge, then the 88 x 132 cover and beside it, aligned to
// its bottom, the title (30 Display; a long name takes two lines at a smaller size), the status, the line under it and
// the actions. Returns its bottom; `btnY`: where the actions start; `pinY`: about the name's baseline (the compact header
// pins once it scrolls under the toolbar).
// "< Library" (.back): the accent's words after a small glyph, 30 high, the fill under the mouse; the glyph on the
// column's edge, where the compact header's Back has it.
static void BackLink(Gfx& gx, float x0, float y) {
    TS ts = Ts(14, 600, pal.tintText); ts.valign = 1;
    const float tw = gx.TextW(L"Library", ts);
    const Rc r{x0 - 8, y, std::ceil(8 + 13 + tw + 8), 30};
    if (Hot(ID_G_BACK)) gx.FillRound(r, 8, Down(ID_G_BACK) ? pal.fillHover : pal.fill);
    gx.Icon(ic::back, x0 + 3, r.cy(), 12, pal.tintText);  // the chevron's ink starts on the column's edge (its glyph box 3 DIPs before it)
    gx.Text(L"Library", Rc{x0 + 13, r.y, tw + 2, r.h}, ts);
    FocusRing(gx, r, 8, ID_G_BACK);
    AddHit(ID_G_BACK, r, R_BUTTON, L"Back to Library", true);  // not only "Library": the toolbar's tab has that name
}

// The banner header (BE-wide .gwide) of a game with Steam hero art: the column's width, 250 high, radius 16, the hero (its
// focus 30 % from the top) under a scrim from the left (0.76 -> 0.56 at 48 % -> 0 at 76 %: white words read 4.5:1 or better
// on the brightest art, Cyberpunk's yellow, as far as the status line goes), the logo bottom left (at most 280 x 90; the
// name in 28 Display without one), one status line (the dot, "WrongGPU 0.5.5-ea is installed · Model M on Direct3D 12 ·
// Steam"; "WrongGPU · Worked last time you played" where it is installed and healthy and Model M upscaled a frame in its last run), then the actions (filled Play, glass for the grey ones) and
// More. The name is still the page's heading for UI Automation (the logo is an image). The upscale wipe under the pointer
// on the art (not on its words or buttons). Returns its bottom; `btnY`: where the actions start.
constexpr float kBannerH = 250, kBannerR = 16;
// the part of the hero's height the banner may show, from its top (the Library's banner keeps the same): a Steam hero's
// dark bottom band never shows as a ragged strip where the banner is wide enough for the art's whole height (760 DIPs)
constexpr float kBannerKeep = 0.94f;
static ImageP HeroArt(const GameEntry& e) {
    if (e.appid.empty()) return nullptr;
    ImageP img = ArtBig(e.appid);
    return img ? img : ArtFor(e.appid).hero;
}
// The page's header is the banner when the game has hero art, decoded yet or not: while it decodes (a page opened before
// its art, after a DPI change, which decodes every image again) the banner waits on a plain plate instead of the cover
// header standing in for it, so the page never re-lays itself out by 120 DIPs and Play never moves under the pointer.
// Once decoded the art decides (a file that does not decode: the cover header).
static bool HasBanner(const GameEntry& e) {
    if (e.appid.empty()) return false;
    bool bigDone = true;
    const ImageP big = ArtBig(e.appid, &bigDone);
    const Art a = ArtFor(e.appid);
    if (big || a.hero) return true;
    if (bigDone && a.done) return false;
    return HeroOnDisk(e.appid);
}
// The header's part still under the compact header once it has pinned (heroRemnant): its art or cover only, scrolling on
// under the compact header, which has the name, the status and the actions (no second Play, no hits)
static bool heroRemnant = false;
// The header's words and actions only (BackLink, the logo or name, the status, the buttons): the compact header's cross-fade
// fades them out inside its plate faster than the art under them, so the two headers' words never show at once
static bool heroWordsOnly = false;
static float Banner(Gfx& gx, float x0, float cw, float top, float& btnY, float& pinY, const ImageP& img) {
    const GameEntry& e = v.cur;
    if (!heroRemnant) BackLink(gx, x0, top + 2);
    const ArtScope onArt;  // from here on the art: the status dot and the buttons' focus ring in the art's colours (both themes)
    const Rc r{x0, top + 40, cw, kBannerH};
    float wipe = 0;
    if (!heroWordsOnly) {
        gx.Shadow(r, kBannerR, 32, pal.dark ? 0.5f : 0.2f, 12, true);  // the hero or its plate covers its inner half
        // the sharp hero cross-fades in over the tile's; while the art decodes, a dark plate under the same words (HasBanner)
        const ImageP shown = HeroCover(gx, e.appid, r, kBannerR, 0.3f, [&] { gx.FillRound(r, kBannerR, pal.genB); }, nullptr, kBannerKeep);
        // the same part of the art; its split line and capsules over the scrim below (they stay white and light)
        if (!heroRemnant) wipe = UpscaleWipe(gx, ID_G_COVER, 0, r, kBannerR, shown ? shown : img, 0.3f, e.last, false, kBannerKeep, false);
    }
    const float lx = x0 + 32, room = cw - 64;
    const float by = r.b() - 26 - 36, sy = by - 16 - 20, logoB = sy - 14;
    btnY = by;
    pinY = logoB - 7;  // about the name's baseline (or the logo's foot): the compact header pins once it passes the toolbar
    const Art a = ArtFor(e.appid);
    // the status line in words (or the step while an operation runs here), and how far its words reach
    const bool busy = BusyHere();
    const HeadStatus st = busy ? HeadStatus{} : Status();
    const wstring line = busy ? wstring() : st.sub.empty() ? st.big : st.big + L" \u00b7 " + st.sub;
    const TS lineS = Ts(13.5f, 400, pal.onArt);
    // the words keep within 90 % of the banner (cut with an ellipsis after that: UI Automation hears the whole line)
    const float lineRoom = std::min(room, cw * 0.9f - 32 - 16) - 15;
    float wordsR = busy ? std::min(room, gx.TextW(BusyWords(), Ts(13.5f, 600, pal.onArt)) + 6)
                        : 15 + std::min(lineRoom, gx.TextW(st.big, Ts(13.5f, 600, pal.onArt)) + (st.sub.empty() ? 0.f : gx.TextW(L" \u00b7 " + st.sub, lineS)) + 2);
    if (a.logo) wordsR = std::max(wordsR, std::min(280.f, room));
    else wordsR = std::max(wordsR, std::min(gx.TextW(e.name, Ts(28, 600, pal.onArt, F_DISPLAY)), std::min(room, cw * 0.6f)));
    // the scrim from the left: 0.76 -> 0.56 -> 0, its 0.56 stop at 48 % of the banner or past the words' right end, whichever
    // is further (as the Library's banner has it), so white words read 4.5:1 or better on any art under them at every width
    // (white art under a 0.56 black: 4.95:1; SHIP-BAR 2). Whole hundredths: one brush per width, made once.
    if (!heroWordsOnly) {
        const float strong = std::min(0.9f, std::max(0.48f, std::ceil((lx - x0 + wordsR + 16) / cw * 100) / 100));
        const float clear = std::min(1.f, strong + 0.28f);
        if (clear < 1) gx.GradStops(r, kBannerR, {{0.f, Alpha(pal.black, 0.76f)}, {strong, Alpha(pal.black, 0.56f)}, {clear, Alpha(pal.black, 0)}, {1.f, Alpha(pal.black, 0)}}, 0, 0.5f, 1, 0.5f);
        else gx.GradStops(r, kBannerR, {{0.f, Alpha(pal.black, 0.76f)}, {strong, Alpha(pal.black, 0.56f)}, {1.f, Alpha(pal.black, 0)}}, 0, 0.5f, 1, 0.5f);
    }
    if (pal.dark && !heroWordsOnly) gx.StrokeRound(r.in(-0.5f), kBannerR + 0.5f, Alpha(pal.white, 0.07f), 1);
    if (heroRemnant) return r.b();
    WipeMarks(gx, r, wipe, e.last);  // clear of the words: the capsules sit under the buttons' row, at the banner's corners
    float th = 60;
    if (a.logo) gx.ImageContain(a.logo, Rc{lx, logoB - 90, std::min(280.f, room), 90}, 0, 2, 1, 10);
    else {
        TS ns = Ts(28, 600, pal.onArt, F_DISPLAY); ns.wrap = true; ns.lineH = 33; ns.trim = true;
        const float fullH = gx.TextH(e.name, std::min(room, cw * 0.6f), ns);
        th = std::min(66.f, fullH);
        gx.Text(e.name, Rc{lx, logoB - th, std::min(room, cw * 0.6f), th + 2}, ns);
        if (fullH > th + 1) v.tips[ID_G_TITLE] = e.name;  // cut: the whole name in the tooltip
    }
    AddHit(ID_G_TITLE, Rc{lx, logoB - th, std::min(280.f, room), th}, R_TEXT, e.name, true, 1, L"", false);  // the page's heading (level 1)
    if (busy) {  // what it is doing and its step in words, the progress under it
        BusyLine(gx, Rc{lx, sy, room, 20}, Ts(13.5f, 600, pal.onArt), Ts(13.5f, 400, Alpha(pal.onArt, 0.85f)));
        const Rc pb{lx, sy + 24, std::min(320.f, room), 4};
        ProgressBar(gx, pb, v.progress);
        AddProgress(ID_G_PROGRESS, pb, BusyWords(), v.progress > 0 ? v.progress : -1.f);
    } else {
        StatusMark(gx, lx + 3.5f, sy + 10.5f, st, pal.onArt);
        TS ls = lineS; ls.trim = true;  // the status semibold, the rest regular (HeadStatus)
        gx.TextLead(line, st.big.size(), pal.onArt, 600, Rc{lx + 15, sy, lineRoom, 20}, ls);
        AddText(Rc{lx, sy, 15 + lineRoom, 20}, Spoken(st), ID_G_STATUS);
        if (gx.TextW(st.big, Ts(13.5f, 600, pal.onArt)) + (st.sub.empty() ? 0.f : gx.TextW(L" \u00b7 " + st.sub, lineS)) > lineRoom)
            v.tips[ID_G_STATUS] = line;  // cut with an ellipsis: the whole line in the tooltip
    }
    float bx = lx;
    // on the art the grey actions are glass and a red pill is the art's own (B_ART_TINTED: the same in both themes, never a
    // pale capsule brighter than Play on dark art)
    for (auto& b : HeadButtons())
        bx += Button(gx, b.id, bx, by, b.text, b.kind == B_SECONDARY ? B_GLASS : b.kind == B_TINTED ? B_ART_TINTED : b.kind, b.icon, b.en, 36, HeadW(gx, b, 36)) + 10;
    MoreButton(gx, bx, by, 36, false, true);
    DescribeHead();
    return r.b();
}

static float Hero(Gfx& gx, float x0, float cw, float top, float& btnY, float& pinY) {
    const GameEntry& e = v.cur;
    if (HasBanner(e)) return Banner(gx, x0, cw, top, btnY, pinY, HeroArt(e));
    // without hero art: the cover and the title beside it
    if (!heroRemnant) BackLink(gx, x0, top + 2);
    const float heroTop = top + 40;
    const float mx = x0 + kBigCoverW + 24, mw = cw - kBigCoverW - 24;
    TS ts = Ts(30, 600, pal.text, F_DISPLAY); ts.wrap = true; ts.lineH = 35;
    while (ts.size > 22 && gx.TextH(e.name, mw, ts) > ts.lineH * 2 + 1) { ts.size -= 2; ts.lineH = std::round(ts.size * 1.17f); }
    ts.trim = true;  // still longer: an ellipsis on the second line
    const float th = std::min(gx.TextH(e.name, mw, ts), ts.lineH * 2);
    const float metaH = th + 6 + 20 + 2 + 18 + 14 + 34;
    const float bottom = heroTop + std::max(kBigCoverH, metaH + 2);
    const Rc cover{x0, heroTop, kBigCoverW, kBigCoverH};  // its top level with the title's, one line or two
    if (!heroWordsOnly) Cover(gx, cover, e.appid, e.name, 10, true);
    float my = bottom - 2 - metaH;
    pinY = my + th - std::round(ts.lineH * 0.2f);  // about the name's (last) baseline: the compact header pins once it passes the toolbar
    btnY = my + th + 6 + 20 + 2 + 18 + 14;
    if (heroRemnant) return bottom;
    if (!heroWordsOnly) UpscaleWipe(gx, ID_G_COVER, 0, cover, 10, e.appid.empty() ? nullptr : ArtFor(e.appid).cover, 0.5f, e.last, false);  // under the pointer (the cover takes no keyboard focus)
    gx.Text(e.name, Rc{mx, my, mw, th + 4}, ts);
    if (gx.TextH(e.name, mw, ts) > th + 1) v.tips[ID_G_TITLE] = e.name;  // cut on its second line: the whole name in the tooltip
    AddHit(ID_G_TITLE, Rc{mx, my, std::min(mw, gx.TextW(e.name, ts) + 2), th}, R_TEXT, e.name, true, 1, L"", false);  // the whole name, also when cut; the page's heading (level 1)
    my += th + 6;
    if (BusyHere()) {  // what it is doing and its step in words, the progress under it
        BusyLine(gx, Rc{mx, my, mw, 20}, Ts(14, 600, pal.text), Ts(14, 400, pal.text2));
        const Rc pb{mx, my + 22 + 7, std::min(320.f, mw), 4};
        ProgressBar(gx, pb, v.progress);  // no amount yet: a travelling segment, not a stub
        AddProgress(ID_G_PROGRESS, pb, BusyWords(), v.progress > 0 ? v.progress : -1.f);
    } else {
        const HeadStatus st = Status();
        StatusMark(gx, mx + 3.5f, my + 10.5f, st, pal.text);
        gx.Text(st.big, Rc{mx + 15, my, mw - 15, 20}, Ts(14, 600, pal.text));  // the status semibold, the line under it regular (HeadStatus)
        gx.Text(st.sub, Rc{mx, my + 22, mw, 18}, Ts(13, 400, pal.text2));
        AddText(Rc{mx, my, mw, 40}, Spoken(st), ID_G_STATUS);  // "Update ready: 0.5.0 -> 0.5.5. Actions wait until it closes"
    }
    my += 20 + 2 + 18 + 14;
    float bx = mx;
    for (auto& b : HeadButtons()) bx += Button(gx, b.id, bx, my, b.text, b.kind, b.icon, b.en, 34, HeadW(gx, b, 34)) + 10;
    MoreButton(gx, bx, my, 34, false);
    DescribeHead();
    return bottom;
}

// The compact header under the toolbar at `py` (content DIPs), on the page colour with a hairline under it: Back (its
// glyph where "< Library" has it), a small cover, the name and the status, the actions at the right. While WrongGPU is
// being installed its step replaces the status and the progress runs under the reading column, on the hairline. A name
// that does not fit ends in an ellipsis; its whole text is the title's name for UI Automation.
// pinFade: how far it has come in (PaintGame: 1 once the header's actions are under it). Its plate comes in with it, its
// words and actions only over the second half, so they never sit over the header's own words still showing through.
static float pinFade = 1;
static void PinnedHeader(Gfx& gx, const Rc& area, float x0, float cw, float py) {
    const GameEntry& e = v.cur;
    const Rc bar{area.x, py, area.w, kPinH};
    gx.Fill(bar, Alpha(pal.bg, pinFade));
    // its edge, one device pixel: the list hairline on the dark theme's black; on the light page that token is all but
    // invisible (1.1:1) and the cells under it look sliced, so a stronger separator there, 1.5:1 (Apple's opaque separator)
    // (the lists' weight: 2 device pixels at 150 %, inside the bar, so the page under it never covers half of it)
    const float px = std::max(1.f, gx.scale), line = HairlineW(gx);
    gx.Fill(Rc{area.x, std::round((bar.b() - line) * px) / px, area.w, line}, Alpha(pal.dark ? pal.sep : Mix(pal.bg, pal.text, 0.2f), pinFade));
    const float words = std::max(0.f, 2 * pinFade - 1);
    if (words < 1) gx.PushFade(words);
    const float cy = py + kPinH / 2;
    Button(gx, ID_G_BACK, x0 - 13, cy - 16, L"", B_GHOST, ic::back, true, 32, 32, false, L"Back to Library");  // the glyph where "< Library" has it
    const size_t backAt = v.hits.size();  // the name and the status are read after Back, before the actions (drawn first: they set the name's room)
    // the cover 12 DIPs after Back: Back's keyboard ring (5 DIPs outside it) keeps clear of the cover
    float tx = x0 - 13 + 32 + 12;
    Cover(gx, Rc{tx, cy - 18, 24, 36}, e.appid, e.name, 4, true);
    tx += 24 + 12;
    float bx = x0 + cw;
    bx -= MoreButton(gx, bx, cy - 16, 32, true) + 8;
    const std::vector<HeadBtn> btns = HeadButtons();
    for (size_t i = btns.size(); i-- > 0;) bx -= Button(gx, btns[i].id, bx, cy - 16, btns[i].text, btns[i].kind, btns[i].icon, btns[i].en, 32, HeadW(gx, btns[i], 32), true) + 8;
    DescribeHead();
    const float nw = std::max(40.f, bx - 8 - tx);
    TS ns = Ts(16, 600, pal.text, F_DISPLAY); ns.trim = true;
    gx.Text(e.name, Rc{tx, cy - 20, nw, 22}, ns);
    AddHit(ID_G_TITLE, Rc{tx, cy - 20, std::min(nw, gx.TextW(e.name, ns) + 2), 22}, R_TEXT, e.name, true, 1, L"", false);
    if (gx.TextW(e.name, ns) > nw) v.tips[ID_G_TITLE] = e.name;  // cut with an ellipsis: the whole name in the tooltip (SHIP-BAR 2)
    const size_t nameAt = v.hits.size() - 1;
    TS ss = Ts(12.5f, 400, pal.text2); ss.trim = true;
    if (BusyHere()) {
        BusyLine(gx, Rc{tx, cy + 2, nw, 18}, Ts(12.5f, 600, pal.text), ss);
        if (gx.TextW(BusyWords(), Ts(12.5f, 600, pal.text)) + 6 > nw) v.tips[ID_G_STATUS] = BusyWords();  // the step cut: its whole line
        const Rc pb{x0, bar.b() - 5, cw, 4};
        ProgressBar(gx, pb, v.progress);
        AddProgress(ID_G_PROGRESS, Rc{pb.x, pb.y, pb.w, 2}, BusyWords(), v.progress > 0 ? v.progress : -1.f);  // inside the plate (its hairline)
    } else {
        const HeadStatus st = Status();
        StatusMark(gx, tx + 3.5f, cy + 11, st, ss.color);
        gx.Text(st.big, Rc{tx + 13, cy + 2, nw - 13, 18}, ss);  // the secondary line under the name: regular (HeadStatus)
        if (gx.TextW(st.big, ss) > nw - 13) v.tips[ID_G_STATUS] = st.big;  // cut: the whole status in the tooltip
        AddText(Rc{tx, cy + 2, nw, 18}, Spoken(st), ID_G_STATUS);
    }
    // reading order (Tab, UI Automation): Back, the name, its status or step, then the actions (they do not overlap: hit
    // testing is unchanged)
    if (words < 1) gx.PopFade();
    if (backAt <= nameAt && nameAt < v.hits.size()) std::rotate(v.hits.begin() + (long long)backAt, v.hits.begin() + (long long)nameAt, v.hits.end());
}

// The page of a game shown before its files are read (OpenGame on a slow or sleeping disk: a click that changes nothing for
// seconds reads as ignored): the header has what the Library knows (the name, the art, the status, its actions), the
// settings wait (pageReading: a quiet row says so, nothing can be changed) and fill in when the read lands (ShowEntryWith),
// with no second entrance.
void ShowReadingPage(const GameEntry& e, int index) {
    v.menu.open = false;
    v.cur = e;
    v.saved = v.edit = gl.global.defaults;  // nothing differs: no unsaved capsule, nothing Apply could write
    v.opti = OptiView();
    v.outside.clear();
    v.optiValues.clear();
    v.detailsOpen = v.lockedOpen = v.optiMore = v.advOpen = false;
    v.scroll[P_GAME] = v.target[P_GAME] = 0;
    v.heroTall = -1;
    v.curIndex = index;
    v.pageChangedAt = NowMs();
    v.wipeAt.clear(); v.wipeOut.clear();
    v.page = P_GAME;
    v.pageReading = true;
    v.editFocusId = 0;
    v.focus = ID_G_BACK;
    Invalidate();
}

// More > Open folder and Open log: the folder looked at and opened on a worker (the window's thread never waits for a game's
// disk: a game on a network share that went offline, or a drive that dropped, would hold it for the share's timeout). A
// folder that is gone (the game moved or uninstalled since the scan) or no log yet is said in the app's words, as a note once
// no dialog is open, not in Windows' own error box with the path (SEE_MASK_FLAG_NO_UI). No owner window for the shell there:
// the worker never waits on the window's thread (which waits for the workers when the window closes).
static void OpenLater(const GameEntry& e, bool log) {
    if (v.fixture) return;
    const wstring dir = e.dir, name = e.name;
    const LastRun last = e.last;
    auto open = [dir, name, log, last](int& icon) -> wstring {  // what to say (empty: it opened, or the player cancelled)
        const HRESULT co = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
        // the log of the run Last run shows (r6 #3): the RR plugin's for a run known from it alone (r4 #1), even where an older
        // front-end log is there; else the front end's
        const wstring file = log ? LastRunLog(dir, last) : dir;
        const wstring gone = name + L"'s folder isn't there any more.\n\nThe game may have been moved or uninstalled. Rescan in the Library to find it again.";
        wstring say;
        icon = 1;
        const DWORD at = GetFileAttributesW(dir.c_str());
        if (at == INVALID_FILE_ATTRIBUTES || !(at & FILE_ATTRIBUTE_DIRECTORY)) say = gone;
        else if (log && !IsFile(file)) { say = L"There's no log yet. It appears after the game runs with WrongGPU."; icon = 2; }
        else {
            SHELLEXECUTEINFOW se{sizeof se};
            se.fMask = SEE_MASK_FLAG_NO_UI | SEE_MASK_NOASYNC;  // done before the worker ends
            se.lpVerb = L"open"; se.lpFile = file.c_str(); se.nShow = SW_SHOWNORMAL;
            if (!ShellExecuteExW(&se)) {
                const DWORD err = GetLastError();
                if (err == ERROR_FILE_NOT_FOUND || err == ERROR_PATH_NOT_FOUND || err == ERROR_BAD_NETPATH || err == ERROR_NOT_READY) say = gone;
                else if (err != ERROR_CANCELLED) say = L"Windows couldn't open " + name + (log ? L"'s log" : L"'s folder") + L".\n\nTry again in a moment.";
            }
        }
        if (SUCCEEDED(co)) CoUninitialize();
        return say;
    };
    if (!v.live || !v.hwnd) {  // no window to come back to: at once
        int icon = 1;
        const wstring say = open(icon);
        if (!say.empty()) Inform(say, icon);
        return;
    }
    const HWND hwnd = v.hwnd;
    OffThread([open, hwnd] {
        int icon = 1;
        const wstring say = open(icon);
        if (!say.empty()) NoteFrom(hwnd, say, icon);
    });
}
void OpenGameFolder(const GameEntry& e) { OpenLater(e, false); }
void OpenGameLog(const GameEntry& e) { OpenLater(e, true); }

// ---- sections ------------------------------------------------------------------------------------------------------------------
// Each returns the y where the next section's heading goes (kSectionGap under its last part).
// The segmented controls of the page's lists share one segment width (the widest label any of them has), so their left
// edges line up down a list. Runs on is the one exception: it names its choice the way the rest of the page does
// ("Direct3D 12": its own sentence, Last run, the header's line), so its segments take that name's width.
float PageSegW(Gfx& gx) { return SegmentW(gx, {L"Default", L"Auto", L"On", L"Off", L"HIP", L"Scale", L"Original"}); }

// "Turn it on in the game" (UI review #1): after an install, until the first run where Model M worked, what the player does
// in the game for WrongGPU to run: choose the upscaler there (DLSS on OptiScaler's route, FSR on the direct one; TAA or
// upscaling off never asks for one), then see it run. Each step ticks itself off from the game's log (LastRun: created when
// the game asked for an upscaler); the card goes once Model M has upscaled a frame (upscaled: the runtime logs "ready" before
// any frame, so ready alone isn't enough: r2 #10).
bool GuideShown() {
    const GameEntry& e = v.cur;
    if (!Rec() || (e.status != GameStatus::Installed && e.status != GameStatus::UpdateAvailable && e.status != GameStatus::Running)) return false;
    if (v.saved.get("wronggpu") == "off") return false;  // turned off on purpose: nothing to turn on
    return !e.last.AnyUpscaled();  // (Ray Reconstruction's frames where it did the upscaling: r6 #1)
}
// One step: its number in a circle (a check on the accent once done), its title, its sentence, and an optional button under
// them (id 0: none). Returns its height.
static float GuideStep(Gfx& gx, float x, float y, float w, int n, bool done, const wstring& title, const wstring& text, int id = 0,
                       const wstring& button = L"") {
    RowLine(gx, x, y, w, n == 1);
    const float cx = x + kCellPad + 10, tx = x + kCellPad + 32, tw = x + w - kCellPad - tx;
    TS ts = Ts(13.5f, 600, pal.text); ts.wrap = true; ts.lineH = 19;
    const TS bs = Wrap(13, pal.text2, 18);
    const float th = gx.TextH(title, tw, ts), bh = gx.TextH(text, tw, bs);
    float yy = y + 12;
    gx.Circle(cx, yy + 10, 10, done ? pal.tint : pal.fill);
    if (done) gx.Icon(ic::check, cx, yy + 10, 10, pal.onAccent);
    else { TS ns = Ts(11.5f, 700, pal.text2); ns.align = 1; ns.valign = 1; gx.Text(std::to_wstring(n), Rc{cx - 10, yy, 20, 20}, ns); }
    gx.Text(title, Rc{tx, yy, tw, th + 2}, ts);
    gx.Text(text, Rc{tx, yy + th + 2, tw, bh + 2}, bs);
    AddText(Rc{tx, yy, tw, th + bh + 4}, wstring(L"Step ") + std::to_wstring(n) + (done ? L", done: " : L": ") + title + L". " + text);
    yy += th + 2 + bh;
    if (id) { Button(gx, id, tx, yy + 10, button, B_SECONDARY, 0, CanEdit(), 30); yy += 10 + 30; }
    return yy + 12 - y;
}
static float GuideSection(Gfx& gx, float x, float y, float w) {
    const GameEntry& e = v.cur;
    const LastRun& lr = e.last;
    // the game asked for an upscaler in its last run (step 2 done); Ray Reconstruction upscaling in it counts (r6 #1), and so does an
    // RR attempt without a successful frame (r7 #1: RR needs the game's DLSS on already, as the RR-only run has it). The checklist's
    // completion stays on successful frames (GuideShown)
    const bool asked = lr.found && (lr.created || lr.rrUpscaled || lr.rrUsed);
    const int done = 1 + (asked ? 1 : 0);
    y += GroupHeading(gx, x, y, w, L"Turn it on in the game", std::to_wstring(done) + L" of 3 done") + kHeadGap;
    y += GroupList(gx, x, y, w, [&](float top) {
        float yy = top;
        const bool hosted = e.hosted;
        yy += GuideStep(gx, x, yy, w, 1, true, L"Installed", hosted ? L"WrongGPU and its OptiScaler are in the game's folder." : L"WrongGPU is in the game's folder.");
        yy += GuideStep(gx, x, yy, w, 2, asked, hosted ? L"Choose DLSS as the game's upscaler" : L"Choose FSR as the game's upscaler",
                        hosted ? L"In the game's graphics settings, choose DLSS (FSR or XeSS also work). With TAA or upscaling off, the game never asks for "
                                 L"an upscaler and WrongGPU stays idle."
                               : L"In the game's graphics settings, choose AMD FSR. With TAA or upscaling off, the game never asks for an upscaler and "
                                 L"WrongGPU stays idle.");
        // OptiScaler's FPS overlay names the upscaler running ("DLSS -> WrongGPU Model M"): offered here while it isn't on. It
        // is on once saved (or the game's file has it on and the launcher leaves it); an unsaved On turns it on with Apply.
        // An OptiScaler build without its key ([Menu] ShowFps, one of the managed rows) is left as it is (RenderOptiIni): no
        // promise and no button there (r2 #9)
        const OptiRow* showFps = nullptr;
        for (auto& r0 : v.opti.locked) if (r0.section == "Menu" && r0.key == "ShowFps") showFps = &r0;
        const string fe = v.edit.get("fps_overlay"), fs = v.saved.get("fps_overlay");
        const bool fileOn = showFps && _stricmp(showFps->value.c_str(), "true") == 0;
        const bool fpsOn = showFps && fe == fs && (fs == "on" || (fs == "keep" && fileOn));
        const bool fpsPending = showFps && fe == "on" && fs != "on";
        const bool offer = hosted && v.opti.have && showFps && !fpsOn && fe != "on";
        // the overlay names WrongGPU only with WrongGPU's edition of OptiScaler, which lists it by name; another OptiScaler shows
        // AMD FSR where Model M runs, as Last run's foot says (the same test: a run of WrongGPU not chosen by name) (r3 #3)
        const bool named = e.optiEdition && !(lr.found && lr.ready && !lr.optiPlugin);
        const wchar_t* fpsState = fpsOn ? L" (it is on)." : fpsPending ? L". It turns on when you apply this change." : L".";
        // why it shows AMD FSR is said once on the page: by Last run's foot once a run shows it (LastRunSection's test), else here (r5 #9)
        const bool footSays = lr.found && hosted && !lr.optiPlugin && lr.ready;
        // with WrongGPU's Ray Reconstruction on, the game's RR does the upscaling while it runs: the overlay of WrongGPU's edition
        // then names WrongGPU RR, and Last run shows Ray Reconstruction (r6 #4: Borderlands 4, Hogwarts Legacy, Cyberpunk 2077 with RR)
        const bool rr = RrOn(e);
        const wstring shows = rr ? L" After you quit, Last run here shows Model M, or Ray Reconstruction when it did the upscaling."
                                 : L" After you quit, Last run here shows Model M.";
        const wstring fpsText = !v.opti.have || !showFps
                                    ? shows.substr(1) + L" This OptiScaler has no FPS overlay setting the launcher can change."
                                : named ? wstring(L"OptiScaler's FPS overlay shows DLSS → WrongGPU Model M while it runs") +
                                              (rr ? L", or DLSS → WrongGPU RR and its mode while Ray Reconstruction runs" : L"") + fpsState + shows
                                        : wstring(L"OptiScaler's FPS overlay shows AMD FSR while Model M runs") +
                                              (footSays ? L"" : L", because this OptiScaler doesn't list WrongGPU by name") + fpsState +
                                              (rr ? L" After you quit, Last run here shows whether Model M or Ray Reconstruction ran."
                                                  : L" After you quit, Last run here shows whether Model M ran.");
        yy += GuideStep(gx, x, yy, w, 3, false, L"See it running",
                        hosted ? fpsText : L"WrongGPU's overlay in the game's corner says Model M while it runs." + shows,
                        offer ? ID_G_GUIDEFPS : 0, L"Turn on the FPS overlay");
        return yy - top;
    });
    return y + kSectionGap;
}

// The Upscaler's Advanced rows (Runs on, Output filter, Pre-exposure, Sparkle guard) shown when a page opens: one of them is
// off its default (saved or unsaved) or changed outside the launcher, so a value the player set is never hidden (UI review #5).
bool AdvancedWanted() {
    static const char* const kAdv[] = {"executor", "output_filter", "pre_exposure", "sparkle_guard"};
    for (const char* k : kAdv) {
        const SettingDef* d = FindSetting(k);
        const string def = d ? d->def : "";
        if (v.saved.get(k) != def || v.edit.get(k) != def) return true;
        for (auto& o : v.outside) if (o.key == k) return true;
    }
    return false;
}

static float UpscalerSection(Gfx& gx, float x, float y, float w) {
    const bool en = CanEdit();
    const float segW = PageSegW(gx);
    y += GroupHeading(gx, x, y, w, L"Upscaler") + kHeadGap;
    // what the controls do, or why they are off: before an install, a game that is gone, the game running, an operation
    const wchar_t* caption = !Rec() ? L"These can be changed once WrongGPU is installed in this game."
                             : v.cur.status == GameStatus::Orphaned ? L"These no longer apply: there is no game to start."
                             : v.cur.status == GameStatus::Running ? L"These can be changed once the game closes."
                             : PlayBlocked(v.cur) ? nullptr  // an operation (or a task) this game waits for: which, in words (BusyDoneWords)
                                      : L"Changes apply the next time the game starts.";
    y += GroupCaption(gx, x, y, w, caption ? wstring(caption) : L"These can be changed once " + BusyDoneWords() + L".");
    y += GroupList(gx, x, y, w, [&](float top) {
        float yy = top;
        const bool on = Rec() && v.edit.get("wronggpu") != "off";  // not in the game yet: off (the accent would say it is on)
        // what off does on the game's route (r2b #3): OptiScaler hands the game's upscaler back to AMD FSR; on the direct route
        // WrongGPU is the game's FSR file itself and passes the frames to the game's own AMD FSR (wronggpu.ini upscaler = amd)
        const wchar_t* offDoes = v.cur.hosted ? L"Model M upscales this game. Off: OptiScaler hands upscaling back to AMD FSR."
                                              : L"Model M upscales this game. Off: WrongGPU passes upscaling to the game's own AMD FSR.";
        yy += SRow(gx, x, yy, w, true, L"Use WrongGPU", offDoes, kSwitchW, kSwitchH,
                   [&](const Rc& r) { Switch(gx, ID_G_MASTER, r, on, en, L"Use WrongGPU (Model M)"); }, !en);
        // the rows under it apply only while it is on: dimmed and unavailable while it is off, as a dependent control is
        const bool sub = en && on;
        // the expert rows behind "Advanced settings (4)" (UI review #5): most players need only the switch; a page opens with
        // them shown when one is off its default, changed outside the launcher or unsaved (AdvancedWanted)
        // one of them changed outside the launcher (read again while the page is open, too): shown, no toggle, as the callout
        // under the list names it
        bool outsideAdv = false;
        for (auto& o : v.outside) outsideAdv = outsideAdv || o.key == "executor" || o.key == "output_filter" || o.key == "pre_exposure" || o.key == "sparkle_guard";
        if (!v.advOpen && !outsideAdv) {
            yy += ActRow(gx, x, yy, w, false, true, ID_G_ADVMORE, L"Advanced settings (4)", ic::down, true, R_LINK);
            return yy - top;
        }
        const std::vector<wstring> ex{L"Auto", L"Direct3D\u00a012", L"HIP"};
        const int exi = Index(kExec, v.edit.get("executor"));
        yy += SRow(gx, x, yy, w, false, L"Runs on", L"Auto uses Direct3D\u00a012 on RX\u00a09000-series cards. HIP is the other way Model\u00a0M can run.", SegmentedW(gx, ex, segW), 36,
                   [&](const Rc& r) { Segmented(gx, ID_G_EXEC0, r.r(), r.y, ex, exi, sub, L"Runs on", 28, {}, true, segW); }, !sub);
        const int fi = Index(kFilter, v.edit.get("output_filter"));
        float fw = 170;  // .select min-width 170, wider when a name needs it
        for (auto* n : kFilterNames) fw = std::max(fw, std::ceil(gx.TextW(n, Ts(13, 600, pal.text)) + 12 + 30));
        // the sentence says what the chosen filter does (each choice's line is in its menu too)
        yy += SRow(gx, x, yy, w, false, L"Output filter", kFilterHelp[fi], fw, 30,
                   [&](const Rc& r) { Select(gx, ID_G_FILTER, r, kFilterNames[fi], sub, L"Output filter"); }, !sub);
        const std::vector<wstring> pre{L"Scale", L"Original"};
        const int pi = Index(kPre, v.edit.get("pre_exposure"));
        yy += SRow(gx, x, yy, w, false, L"Pre-exposure", L"Scale keeps hair and fine detail steady. Original uses the game's brightness value as it is; in some games that makes hair and fine detail flicker.",
                   SegmentedW(gx, pre, segW), 36, [&](const Rc& r) { Segmented(gx, ID_G_PRE0, r.r(), r.y, pre, pi, sub, L"Pre-exposure", 28, {}, true, segW); }, !sub);
        const bool sp = v.edit.get("sparkle_guard") != "off";
        yy += SRow(gx, x, yy, w, false, L"Sparkle guard", L"Calms single-pixel flashes on foliage. Uses HIP, even when Model M runs on Direct3D\u00a012.", kSwitchW, kSwitchH,
                   [&](const Rc& r) { Switch(gx, ID_G_SPARKLE, r, sp, sub, L"Sparkle guard"); }, !sub, L"+0.11 ms");
        if (!outsideAdv) yy += ActRow(gx, x, yy, w, false, true, ID_G_ADVMORE, L"Fewer settings", ic::up, true, R_LINK);
        return yy - top;
    });
    // values changed outside the launcher: what they are, and the two ways back to one set of values
    if (!v.outside.empty()) {
        // in the page's words: "Sparkle guard On (WrongGPU's: Off)"; the same words and buttons as OptiScaler's note
        wstring t = L"Changed in the game's files, outside the launcher: ";
        for (size_t i = 0; i < v.outside.size(); ++i) {
            const OutsideEdit& o = v.outside[i];
            t += (i ? L", " : L"") + SettingLabel(o.key) + L" " + SettingWords(o.key, o.fileValue) + L" (WrongGPU's: " + SettingWords(o.key, o.profileValue) + L")";
        }
        y += 12;
        y += ChoiceCallout(gx, x, y, w, t + L".", {{ID_G_KEEP, L"Keep these values"}, {ID_G_REVERT, L"Use WrongGPU's values"}}, en);
    }
    return y + kSectionGap;
}

// Ray Reconstruction (rr-notes.md): only in a game that uses it (it ships DLSS Ray Reconstruction), and only where this
// build has the RR files or the game already has them; every other game's page is as it was. Use WrongGPU and Quality
// (Accurate / Fast; Exact in front of them with Settings > Exact mode, or while the game is set to it) are profile values
// like the Upscaler's (the unsaved bar, Apply). Without the RR data a red line and Convert take the rows' place; an
// install from before this build has Repair add the files.
static bool RrShown() { const GameEntry& e = v.cur; return e.rrUses && (e.rrInstalled || v.rrBuild); }
// rc5 (gamerules.h): the page's game takes RR only through its experimental opt-in, as the page's settings stand now
// (the saved profile, or this page's unsaved change)
static const char* RrExperimentalSetting() { const GameRule* r = GameRuleFor(v.cur.exe); return r ? r->rrExperimentalSetting : nullptr; }
bool RrHeldOffHere() { const char* s = RrExperimentalSetting(); return s && (!Rec() || v.edit.get(s) != "on"); }
bool RrExperimentalHere() { const char* s = RrExperimentalSetting(); return s && Rec() && v.edit.get(s) == "on"; }
// rc5: the game's menu has no RR option (LEGO Batman): one line where the RR section would be
static float RrNotOfferedSection(Gfx& gx, float x, float y, float w) {
    y += GroupHeading(gx, x, y, w, L"Ray Reconstruction") + kHeadGap;
    y += GroupList(gx, x, y, w, [&](float top) { return DotRow(gx, x, top, w, true, v.cur.rrWhy, pal.greyDot, pal.text); });  // (rc6: the entry's line)
    return y + kSectionGap;
}

// A game that uses the Ray Reconstruction data and is running (it holds the data's file: Convert can't replace it then,
// ConvertRr). Empty: none.
static wstring RunningRrName() {
    if (v.cur.status == GameStatus::Running && v.cur.rrInstalled) return v.cur.name;
    for (auto& x : v.games) if (x.status == GameStatus::Running && x.rrInstalled) return x.name;
    return L"";
}

static wstring RrMenuKey() {  // the game's OptiScaler menu key as the page's Menu key row names it (auto: Insert)
    for (auto& r0 : v.opti.editable)
        if (r0.def && r0.def->kind == OptiKind::Key) return r0.value.empty() || r0.value == "auto" ? wstring(L"Insert") : KeyName(r0.value);
    return L"Insert";
}

// A numbered step with its title, an optional button on the title's line (id 0: none) and an optional value under it in a
// monospaced box (a folder, a line to paste): set apart from the words, so it is read and typed exactly. Returns its height.
static float StepCodeRow(Gfx& gx, float x, float y, float w, int n, const wstring& title, const wstring& code, int id, const wstring& btn,
                         const wstring& btnName, bool enabled) {
    RowLine(gx, x, y, w, false);
    const float bw = id ? ButtonW(gx, btn, 0, 30, B_SECONDARY) : 0;
    const float cx = x + kCellPad + 10, tx = x + kCellPad + 32, tw = x + w - kCellPad - tx - (id ? bw + 16 : 0), cw = x + w - kCellPad - tx;
    TS ts = Ts(13.5f, 600, pal.text); ts.wrap = true; ts.lineH = 19;
    TS ms = Wrap(12.5f, pal.text, 18); ms.face = F_MONO;
    const float th = gx.TextH(title, tw, ts), mh = code.empty() ? 0.f : gx.TextH(code, cw - 20, ms);
    float yy = y + 14;
    gx.Circle(cx, yy + 10, 10, pal.fill);
    TS ns = Ts(11.5f, 700, pal.text2); ns.align = 1; ns.valign = 1;
    gx.Text(std::to_wstring(n), Rc{cx - 10, yy, 20, 20}, ns);
    gx.Text(title, Rc{tx, yy, tw, th + 2}, ts);
    if (id) Button(gx, id, x + w - kCellPad, yy - 5, btn, B_SECONDARY, 0, enabled, 30, 0, true, btnName);
    yy += std::max(th, 20.f) + 8;
    if (!code.empty()) {
        gx.FillRound(Rc{tx, yy, cw, mh + 14}, 6, pal.fill);
        gx.Text(code, Rc{tx + 10, yy + 7, cw - 20, mh + 2}, ms);
        yy += mh + 14;
    }
    AddText(Rc{tx, y + 14, cw, yy - y - 14}, L"Step " + std::to_wstring(n) + L": " + title + (code.empty() ? L"" : L" " + code));
    return yy + 14 - y;
}

// The file check's "Couldn't read Engine.ini" (RrConfigLineState 3, r3 #8): why, and what lets it be read again (not a fix:
// adding the line would fail the same way). Myself (mine): the player's own file, which the game may read fine (UTF-16): only
// WrongGPU's check can't read it, and how to check it by hand, never a request to save it another way (r4 #6)
static wstring CantReadWhy(bool mine, const RrConfigGame& rg) {
    const wstring why = v.rrcfgWhy.empty() ? wstring(L"it couldn't be read") : v.rrcfgWhy;
    const bool utf16 = why.find(L"UTF-16") != wstring::npos, inUse = why.find(L"another program") != wstring::npos;
    if (mine)
        return L"WrongGPU's check couldn't read it: " + why + L". " + (inUse ? L"Close the program that has it open, then check again, or c" : L"C") +
               L"heck it by hand: open Engine.ini in Notepad and look for the line under [SystemSettings]" +
               (rg.readOnly ? L", and in its Properties that Read-only is ticked." : L".");
    return L"WrongGPU couldn't read it: " + why + L". " +
           (utf16 ? L"Open it in Notepad, choose Save as, set Encoding to UTF-8, then check again."
            : inUse ? L"Close the program that has it open, then check again."
                    : L"Check that it opens in Notepad, then check again.");
}

// Borderlands 4 / Hogwarts Legacy (wg_core/rrconfig.h): the Engine.ini line their Ray Reconstruction needs, changeable at any
// time (UI review #2). WrongGPU (rr_config_fix=on: it adds the line, keeps Borderlands 4's file read-only, takes it out on
// uninstall) or Myself (off: WrongGPU leaves the file alone, takes its own line out; the steps, the folder, the line with
// Copy, and the file check with Check again). Not chosen yet (ask): neither segment. An Apply writes the choice.
static float RrConfigRows(Gfx& gx, float x, float y, float w, const RrConfigGame& rg, bool en, bool first) {
    const string want = v.edit.get("rr_config_fix"), saved = v.saved.get("rr_config_fix");
    const wstring name = rg.name;
    const int sel = want == "on" ? 0 : want == "off" ? 1 : -1;
    wstring desc;
    // the game hasn't made its settings folder yet (its first start): WrongGPU's choice waits for it, nothing is written (r2b #1)
    const bool noFolderYet = v.rrcfg == -1 && !v.rrcfgFile.empty();
    // a game in a protected folder: the line is added on the player's click once the folder is there, never by itself (r3 #4); the
    // same without WrongGPU's RR files, where the window never adds it by itself either (r7 #5: StartDue)
    const bool noRr = !v.cur.rrInstalled;
    const bool clickLater = (v.cur.rrConfigProtected || noRr) && !v.rrcfgFixed;
    if (want == "on")  // (the file as it is now: the row under it)
        desc = saved != "on" ? (noFolderYet && clickLater ? wstring(L"The game's settings folder isn't there yet. After you start the game once, click Fix it now here to add the line")
                                : noFolderYet ? wstring(L"The game's settings folder isn't there yet, so WrongGPU adds the line after you start the game once")
                                              : wstring(L"WrongGPU adds the line when you apply")) + (rg.readOnly ? L", and keeps the file read-only." : L".")
                             : clickLater ? wstring(L"WrongGPU adds the line when you click Fix it now") + (rg.readOnly ? L", and keeps the file read-only" : L"") +
                                                L". Uninstalling WrongGPU takes it out."
                                          : wstring(L"WrongGPU adds the line") + (rg.readOnly ? L" and keeps the file read-only" : L"") + L". Uninstalling WrongGPU takes it out.";
    else if (want == "off")  // led by what the line is for (r2 #8), as the question is
        desc = saved == "on" ? L"WrongGPU takes its line out when you apply, then leaves the file to you."
                             : RrConfigWhy(rg) + L" You add it yourself; WrongGPU leaves the file alone.";
    else
        desc = RrConfigWhy(rg) + L" Not chosen yet: WrongGPU can add it" + (rg.readOnly ? L" and keep the file read-only" : L"") + L", or you can add it yourself.";
    // r7 #5: the line is for WrongGPU's Ray Reconstruction, which isn't in this game's install: said first
    if (noRr) desc = L"WrongGPU's Ray Reconstruction isn't installed for this game, so this line does nothing for now. " + desc;
    const std::vector<wstring> opts{L"WrongGPU", L"Myself"};
    float yy = y;
    yy += SRow(gx, x, yy, w, first, L"Engine.ini setting", desc, SegmentedW(gx, opts, 0), 36,
               [&](const Rc& r) { Segmented(gx, ID_G_RRCFG0, r.r(), r.y, opts, sel, en, L"Engine.ini setting", 28, {}, true); }, !en);
    if (want == "on" && saved == "on") {
        // WrongGPU's choice, saved: the file as it is now, each state with what puts it right (r2 #6). Gone or not read-only:
        // Fix it again applies the choice again; not known (the game's settings folder isn't there yet): Check again later
        // Chosen before the game's first start (nothing written yet: r2b #1): it waits for the game's settings folder
        // The game in a protected folder (r2c): the window never adds it by itself (a permission prompt nobody clicked for);
        // once the folder is there it is "Ready to add", and Fix it now (the player's click) brings the prompt
        const int st = v.rrcfg;
        const bool waits = st == -1 && noFolderYet && !v.rrcfgFixed;
        const bool click = !v.rrcfgFixed && (v.cur.rrConfigProtected || noRr);  // (r7 #5: without the RR files too)
        const bool ready = st == 0 && click;
        const wstring askWhy = v.cur.rrConfigProtected ? L" Windows asks for permission first, because " + name + L" is installed in a protected folder." : L"";
        // (a protected game waits for the player's click, not for itself: r3 #4)
        const wstring t = st == 1 ? L"Engine.ini has the line" : st == 2 ? L"Not read-only" : st == 3 ? L"Couldn't read Engine.ini" : ready ? L"Ready to add"
                          : st == 0 ? L"Not in Engine.ini now"
                          : waits && click ? L"Waiting for the game's first start" : waits ? L"Will be added after you start the game once" : L"Couldn't check";
        const wstring d = st == 1 ? (rg.readOnly ? L"The file is read-only, so " + name + L" keeps it." : L"Ray Reconstruction can run in " + name + L".")
                          : st == 3 ? CantReadWhy(false, rg)
                          : st == 2 ? name + L" will remove it at its next start. " + (v.rrcfgFixed ? L"Fix it again" : L"Fix it now") + L" makes the file read-only."
                          : ready ? name + L" has made its settings folder. Click Fix it now to add the line." + askWhy
                          : st == 0 ? L"Until it is back, the game doesn't use Ray Reconstruction. " + wstring(v.rrcfgFixed ? L"Fix it again" : L"Fix it now") + L" adds it."
                          : waits && click ? name + L" makes its settings folder at its first start. Then click Fix it now here to add the line." + askWhy
                          : waits ? name + L" makes its settings folder at its first start. WrongGPU adds the line the next time the launcher opens or you open this page."
                                    : wstring(L"Start the game once, then check again.");
        const bool fix = st == 0 || st == 2;
        // "again" only where WrongGPU added the line before (the record has it); the player's own line, or none added yet: now.
        // Ready to add: Fix it now is the row's one action, on the accent's fill as Update (r3 #10); the line gone or the file not
        // read-only (a real problem: the game won't use Ray Reconstruction, or removes the line at its next start) is never less
        // emphasised than that routine step: its Fix it now / Fix it again on the same fill (r4 #7). Check again stays plain
        const bool again = fix && !ready && v.rrcfgFixed;
        yy += NeedRow(gx, x, yy, w, false, t, d, st == 1 ? pal.green : st == -1 ? pal.greyDot : pal.amber, pal.text, fix ? ID_G_RRCFG_FIX : ID_G_RRCFG_CHECK,
                      !fix ? L"Check again" : again ? L"Fix it again" : L"Fix it now", fix ? B_GO : B_PLAIN, fix ? en && !Locked() : true,
                      !fix ? L"Check Engine.ini again" : again ? L"Fix Engine.ini again" : L"Fix Engine.ini now");
        if (fix && Locked()) { v.tips[ID_G_RRCFG_FIX] = BusyWhy(v.cur); Describe(ID_G_RRCFG_FIX, BusyWhy(v.cur)); }
    }
    if (want != "off") return yy - y;
    // the steps (they stay on the page while the player works through them), then the file as it is now
    // before the game's first start (-1) its settings folder isn't there and its sub-folder is a guess (Hogwarts Legacy uses
    // WindowsNoEditor): step 1 names the folder the game makes it in and says to start the game first; Open folder makes
    // nothing (RrConfigAction) (r2 #7)
    const bool noFolder = v.rrcfg == -1;
    wstring folder = !v.rrcfgFile.empty() ? Parent(v.rrcfgFile)
                     : (rg.documents ? L"Documents\\" : L"%LOCALAPPDATA%\\") + wstring(rg.folder) + L"\\Windows";
    if (noFolder) folder = Parent(folder);
    // every step spelled out (r2 #5): Borderlands 4 deletes Engine.ini at each start, so the file and its section are usually
    // not there; the same steps as the release notes and the CLI's (RrConfigSteps)
    yy += StepCodeRow(gx, x, yy, w, 1, noFolder ? L"Start the game once, then come back (it makes its settings folder here)" : L"Open the game's settings folder",
                      folder, ID_G_RRCFG_OPEN, L"Open folder", L"Open the settings folder of " + name, true);
    yy += StepCodeRow(gx, x, yy, w, 2, L"Create or open Engine.ini there with Notepad", L"", 0, L"", L"", true);
    yy += StepCodeRow(gx, x, yy, w, 3, L"If the file has no [SystemSettings] line, add it at the end", L"", 0, L"", L"", true);
    yy += StepCodeRow(gx, x, yy, w, 4, L"Paste this line under [SystemSettings]", RrConfigLineW(), ID_G_RRCFG_COPY, L"Copy", L"Copy the line", true);
    yy += StepCodeRow(gx, x, yy, w, 5, L"Save the file", L"", 0, L"", L"", true);
    if (rg.readOnly)
        yy += StepCodeRow(gx, x, yy, w, 6, L"Right-click Engine.ini, choose Properties, tick Read-only and click OK (without it, " + name +
                                               L" deletes the file at its next start)", L"", 0, L"", L"", true);
    const int st = v.rrcfg;
    const wstring t = st == 1 ? L"Engine.ini has the line" : st == 3 ? L"Couldn't read Engine.ini" : st == 2 ? L"The line is there, but the file isn't read-only"
                      : st == -1 ? L"The game's settings folder isn't there yet"
                                                                                                                                   : L"Not in Engine.ini yet";
    const wstring d = st == 1 ? L"Ray Reconstruction can run in " + name + L"." : st == 3 ? CantReadWhy(true, rg)
                      : st == 2 ? name + L" deletes the file at its next start. Do step 6, then check again."
                      : st == -1 ? wstring(L"Start the game once, then come back and check again.") : L"Until it is, the game doesn't use Ray Reconstruction.";
    yy += NeedRow(gx, x, yy, w, false, t, d, st == 1 ? pal.green : st == -1 ? pal.greyDot : pal.amber, pal.text, ID_G_RRCFG_CHECK, L"Check again", B_PLAIN, true,
                  L"Check Engine.ini again");
    return yy - y;
}

// The Engine.ini setting's own list (a game it applies to, with WrongGPU in it): under the Ray Reconstruction section (also where
// that section says this build or this OptiScaler can't run it), or on its own under that heading where the section isn't
// shown (the setting is the game's, whatever this build has). Returns the y
// under its list (y itself for another game).
static float RrConfigCard(Gfx& gx, float x, float y, float w) {
    const RrConfigGame* rg = Rec() ? RrConfigGameFor(v.cur.exe) : nullptr;
    if (!rg) return y;
    const bool en = CanEdit();
    if (!cv.dry) v.rrcfgCardY = y;  // "Show me how" scrolls here (RevealRrConfigCard)
    return y + GroupList(gx, x, y, w, [&](float top) { return RrConfigRows(gx, x, top, w, *rg, en, true); });
}

static float RrSection(Gfx& gx, float x, float y, float w) {
    const GameEntry& e = v.cur;
    const bool en = CanEdit();
    const float segW = PageSegW(gx);
    y += GroupHeading(gx, x, y, w, L"Ray Reconstruction") + kHeadGap;
    y += GroupCaption(gx, x, y, w, e.rrPathTracingOnly ? L"Used when the game's path tracing asks for DLSS Ray Reconstruction."
                                                       : L"Used when the game's path tracing or ray tracing asks for DLSS Ray Reconstruction.");
    const bool blocked = RrBlocked(e), needData = !v.rrReady, needFiles = Rec() && !e.rrInstalled;
    y += GroupList(gx, x, y, w, [&](float top) {
        float yy = top;
        if (blocked) {  // this build can't run it in this game (no OptiScaler edition): said as it is, no switch that would do nothing
            yy += DotRow(gx, x, yy, w, true, L"This version of WrongGPU can't run Ray Reconstruction in this game. The game keeps its own ray tracing.", pal.greyDot, pal.text);
            return yy - top;
        }
        // rc5: a game whose RR only comes with its experimental opt-in (gamerules.h, Spider-Man 2): off, RR is off and not
        // switchable here, one plain line (nothing to convert, set up or prepare for it)
        if (RrHeldOffHere()) {
            yy += DotRow(gx, x, yy, w, true, GameRuleFor(e.exe)->rrHeldOffText, pal.greyDot, pal.text);
            return yy - top;
        }
        if (RrNoEdition(e)) {  // installed with another OptiScaler: said as it is, with what makes it work; no switch that would do nothing
            yy += DotRow(gx, x, yy, w, true, L"Ray Reconstruction isn't available here with this game's OptiScaler. This game creates Ray Reconstruction itself, "
                                             L"which only the WrongGPU edition of OptiScaler passes on to WrongGPU. Repair installs the edition when the launcher can "
                                             L"download it. Until then the game keeps its own ray tracing.", pal.greyDot, pal.text);
            return yy - top;
        }
        if (needData) {
            // the state in Settings' words and colour (amber: a warning in words; the red tint pill is the action that needs
            // you, as the RR Repair row's). While it converts (from here or from Settings) the row says so, with Settings'
            // busy indicator; its busy name keeps the focus (PaintAll)
            // it waits (disabled, the reason in its tooltip and help text, said before a press) for another task or
            // operation, and while a game that uses the data runs (it holds the data's file)
            const wstring running = RunningRrName();
            const bool working = v.task == T_RRDATA, can = v.task == T_NONE && !v.busy && running.empty();
            const wstring why = working || can ? wstring() : v.task != T_NONE || v.busy ? BusyWhy(v.cur)
                                                                 : L"Waits until " + running + L" closes. It uses the Ray Reconstruction data.";
            yy += NeedRow(gx, x, yy, w, true, working ? L"Converting the Ray Reconstruction data" : L"Ray Reconstruction data isn't converted yet",
                          e.rrPathTracingOnly ? L"WrongGPU converts it once from a DLSS Ray Reconstruction file on this PC. Until then, the game offers no path tracing."
                                              : L"WrongGPU converts it once from a DLSS Ray Reconstruction file on this PC. Until then, the game uses its own denoiser.",
                          working ? pal.greyDot : pal.amber, pal.text, ID_G_RRCONVERT, working ? L"Converting\u2026" : L"Convert", B_TINTED, can,
                          working ? L"Converting the Ray Reconstruction data\u2026" : L"Convert the Ray Reconstruction data", working);
            if (!why.empty()) { v.tips[ID_G_RRCONVERT] = why; Describe(ID_G_RRCONVERT, why); }
            return yy - top;
        }
        if (needFiles) {
            // an install from before this build: an optional addition, not a fault (a grey dot and a grey capsule, never the
            // red "needs you"); its operation is a repair, said as what it does ("Set up" -> "Setting up..." -> "Ray
            // Reconstruction is set up in ...")
            const bool working = BusyHere() && v.busyVerb == L"Setting up\u2026";
            const bool can = !Locked() && e.status != GameStatus::Orphaned;
            yy += NeedRow(gx, x, yy, w, true, L"Not set up in this game yet", L"Set up adds it to WrongGPU in this game.",
                          pal.greyDot, pal.text, ID_G_RRREPAIR, working ? L"Setting up\u2026" : L"Set up", B_PLAIN, can, working ? L"Setting up Ray Reconstruction\u2026" : L"Set up Ray Reconstruction", working);
            if (Locked() && !working) { v.tips[ID_G_RRREPAIR] = BusyWhy(e); Describe(ID_G_RRREPAIR, BusyWhy(e)); }
            return yy - top;
        }
        // rc5: the experimental opt-in on (Spider-Man 2, set by the command line): RR on, labelled Experimental with its
        // warning; the switch turns the opt-in off (ID_G_RRON, ui_activate.cpp)
        const bool experimental = RrExperimentalHere();
        const bool on = Rec() && v.edit.get("rr_enabled") != "off";
        // a game whose RR comes only with its path tracing (gamerules.h): off, it offers no path tracing at all
        const wchar_t* onDesc = experimental ? L"Experimental: the game may freeze or crash with it. Off: Model M only."
                              : e.rrPathTracingOnly ? L"Denoises and upscales the path-traced image. Off: the game offers no path tracing, only its own ray tracing levels."
                                                    : L"Denoises and upscales the ray-traced image. Off: the game falls back to its own denoiser.";
        yy += SRow(gx, x, yy, w, true, experimental ? L"Use WrongGPU (Experimental)" : L"Use WrongGPU", onDesc, kSwitchW, kSwitchH,
                   [&](const Rc& r) { Switch(gx, ID_G_RRON, r, on, en, experimental ? L"Use WrongGPU (Ray Reconstruction, experimental)" : L"Use WrongGPU (Ray Reconstruction)"); }, !en);
        const string mode = v.edit.get("rr_mode");
        // Exact only behind Settings' switch, or while this game is set to it (its state is never hidden)
        const bool exact = gl.global.exactMode || v.saved.get("rr_mode") == "exact" || mode == "exact";
        std::vector<wstring> q;
        if (exact) q.push_back(L"Exact");
        q.push_back(L"Accurate");
        q.push_back(L"Fast");
        // rc5 (owner 2026-11-05): Fast is the default and the recommended choice; Accurate is the higher quality, slower one
        const int qi = (mode == "accurate" ? 1 : mode == "exact" ? 0 : 2) - (exact ? 0 : 1);
        // Accurate is described against Exact only while Exact is there to see
        const wchar_t* desc = mode == "exact" ? L"The most precise. The slowest."
                            : mode == "accurate" ? (exact ? L"Looks the same as Exact, and faster. Higher quality than Fast, slower."
                                                          : L"Higher quality than Fast, slower.")
                                                 : L"Recommended. The fastest. Fine detail can differ slightly from Accurate in motion.";
        // it applies only while Use WrongGPU is on: dimmed and unavailable while it is off
        yy += SRow(gx, x, yy, w, false, L"Quality", desc, SegmentedW(gx, q, segW), 36,
                   [&](const Rc& r) { Segmented(gx, exact ? ID_G_RRMODE0 : ID_G_RRMODE0 + 1, r.r(), r.y, q, qi, en && on, L"Ray Reconstruction quality", 28, {}, true, segW); }, !(en && on));
        // release 0.6.0: its kernels prepared by the launcher for this game (rrprep.h): ready, or Prepare
        if (Rec() && on) {
            wstring why;
            const bool working = BusyHere() && v.busyVerb == L"Preparing Ray Reconstruction\u2026";
            const wstring bg = working ? L"" : RrBackgroundWords(e.exe);  // 0.6.0 polish: prepared again after a driver update
            // the game prepares them at its own first start (gamerules.h: its driver cache is its own; under Wine every game,
            // rrprep.h): no Prepare button, which would do nothing there
            if (RrKernelsInGame(e.exe))
                yy += DotRow(gx, x, yy, w, false,
                             UnderWine() ? L"Prepared by the game at its first start (under Proton its shader caches are its own)"
                                         : L"Prepared by the game at its first start (about a minute after a graphics driver update)",
                             pal.greyDot, pal.text);
            else if (!bg.empty())
                yy += NeedRow(gx, x, yy, w, false, bg, L"Not ready until it is done. Prepare does it now instead.",
                              pal.greyDot, pal.text, ID_G_RRPREP, L"Prepare", B_PLAIN, !Locked(), L"Prepare Ray Reconstruction", false);
            else if (!working && RrReadyForGame(e.dir, e.exe, &why))
                yy += DotRow(gx, x, yy, w, false, L"Prepared: the game starts it without waiting for the graphics driver", pal.green, pal.text);
            else
                yy += NeedRow(gx, x, yy, w, false, working ? L"Preparing Ray Reconstruction" : L"Not prepared yet",
                              working ? L"Esc cancels. What is done is kept." : L"Prepare compiles it now (about a minute), so the game doesn't wait for it at its first start.",
                              pal.greyDot, pal.text, ID_G_RRPREP, working ? L"Preparing\u2026" : L"Prepare", B_PLAIN, !Locked(), L"Prepare Ray Reconstruction", working);
        }
        return yy - top;
    });
    if (!blocked && !RrNoEdition(e) && !needData && !needFiles && v.opti.have)
        y += GroupFoot(gx, x, y, w, L"You can also switch it in the game: OptiScaler's menu (" + RrMenuKey() + L"), Ray Reconstruction.");
    // Borderlands 4 / Hogwarts Legacy: the Engine.ini line their Ray Reconstruction needs, its own list (UI review #2). Drawn
    // whenever WrongGPU is installed, whatever the OptiScaler edition or this build can do: the line is the game's own, the
    // question is asked after every install, repair or update, and it promises the choice can be changed here (r2 #2)
    if (Rec() && RrConfigGameFor(e.exe)) y = RrConfigCard(gx, x, y + 12, w);
    return y + kSectionGap;
}

// OptiScaler with WrongGPU's setup: what it is (version, the name it loads as), what can be set (the FPS overlay, the
// editable keys this build has: the menu key first, the rest behind "More OptiScaler settings"), the keys WrongGPU manages
// under the list.
static float OptiSection(Gfx& gx, float x, float y, float w) {
    const GameEntry& e = v.cur;
    const OptiView& o = v.opti;
    const bool en = CanEdit() && o.have;
    y += HeadingTag(gx, x, y, w, L"OptiScaler", L"WrongGPU's setup", pal.text2) + kHeadGap;
    // one text field per editable key; never fewer here (a field dropped here would leak its window: ShowEntry destroys the
    // ones another game's page does not need)
    if (v.edOpti.size() < o.editable.size()) v.edOpti.resize(o.editable.size());
    const float segW = PageSegW(gx);
    size_t more = 0;
    for (auto& r0 : o.editable) more += !(r0.def && r0.def->kind == OptiKind::Key);
    y += GroupList(gx, x, y, w, [&](float top) {
        float yy = top;
        bool first = true;
        auto kv = [&](const wstring& k, const wstring& val, const KvOpt& opt = KvOpt()) { if (val.empty()) return; yy += KvRow(gx, x, yy, w, first, k, val, opt); first = false; };
        kv(L"Version", o.version);
        {  // the name it loads as in words; the file and the planner's reason in the tooltip (SHIP-BAR 4: no DLL names). Only
           // when there is something to know (another name, an untested pair): the usual name says nothing to the player
            KvOpt lo;
            lo.tip = PlanTip(o.proxyReason, o.proxy);
            lo.sub = PlanSub(o.proxyReason, o.proxy);
            if (PlanNotable(o.proxyReason, o.proxy)) kv(L"Installed as", PlanName(o.proxyReason, o.proxy), lo);
        }
        if (!IEq(o.folder, e.dir)) { KvOpt fo; fo.tip = o.folder; kv(L"Folder", Place(o.folder), fo); }  // a place, the path in the tooltip
        const std::vector<wstring> fpsOpts{L"Default", L"On", L"Off"};
        const int fps = Index(kFps, v.edit.get("fps_overlay"));
        yy += SRow(gx, x, yy, w, first, L"FPS overlay", L"OptiScaler's frame time and upscaler time in the game.", SegmentedW(gx, fpsOpts, segW), 36,
                   [&](const Rc& r) { Segmented(gx, ID_G_FPS0, r.r(), r.y, fpsOpts, fps, en, L"FPS overlay", 28, {}, true, segW); }, !en);
        first = false;
        for (size_t i = 0; i < o.editable.size(); ++i) {
            const OptiRow& r0 = o.editable[i];
            if (!v.optiMore && !(r0.def && r0.def->kind == OptiKind::Key)) continue;
            const string val = i < v.optiValues.size() ? v.optiValues[i] : r0.value;
            const int id = ID_G_OPTI0 + (int)i * 10;
            const wstring label = r0.def ? r0.def->label : FromUtf8(r0.key), help = r0.def ? r0.def->help : L"";
            const OptiKind kind = r0.def ? r0.def->kind : OptiKind::Number;
            if (kind == OptiKind::Key) {
                yy += SRow(gx, x, yy, w, false, label, help, 150, 30, [&](const Rc& r) { Select(gx, id, r, KeyName(val), en, label); }, !en);
            } else if (kind == OptiKind::Bool) {
                const std::vector<wstring> opts{L"Auto", L"On", L"Off"};
                const int sel = val == "true" ? 1 : val == "false" ? 2 : 0;
                yy += SRow(gx, x, yy, w, false, label, help, SegmentedW(gx, opts, segW), 36,
                           [&](const Rc& r) { Segmented(gx, id, r.r(), r.y, opts, sel, en, label, 28, {}, true, segW); }, !en);
            } else {
                // locked with the row (the game running, an operation, a game that is gone): no keys, no focus, no edit
                yy += SRow(gx, x, yy, w, false, label, help, 110, 30, [&](const Rc& r) { FieldBox(gx, v.edOpti[i], id, r, FromUtf8(val), L"auto", 0, label, false, 0, en); }, !en);
            }
        }
        if (more > 0) {
            const wstring link = v.optiMore ? L"Fewer OptiScaler settings" : L"More OptiScaler settings (" + std::to_wstring(more) + L")";
            yy += ActRow(gx, x, yy, w, false, true, ID_G_OPTIMORE, link, v.optiMore ? ic::up : ic::down, true, R_LINK);
        }
        return yy - top;
    });
    // values OptiScaler's menu (or a hand edit) saved over ours
    if (!o.outside.empty()) {
        // in the page's words: "Menu key F1 (WrongGPU's: Insert)"
        auto words = [](const OptiKeyDef* d, const string& val) -> wstring {
            if (d && d->kind == OptiKind::Key) return KeyName(val);
            if (val == "true") return L"On";
            if (val == "false") return L"Off";
            if (val == "auto" || val.empty()) return L"Auto";
            return FromUtf8(val);
        };
        wstring t = L"Changed by OptiScaler's menu (or by hand): ";
        for (size_t i = 0; i < o.outside.size(); ++i) {
            const OptiOutside& x0 = o.outside[i];
            const OptiKeyDef* d = FindOptiKey(x0.section, x0.key);
            t += (i ? L", " : L"") + (d ? wstring(d->label) : FromUtf8(x0.key)) + L" " + words(d, x0.fileValue) + L" (WrongGPU's: " + words(d, x0.ours) + L")";
        }
        y += 12;
        y += ChoiceCallout(gx, x, y, w, t + L".", {{ID_G_OKEEP, L"Keep these values"}, {ID_G_OREVERT, L"Use WrongGPU's values"}}, en);
    }
    // the keys WrongGPU manages: a sentence under the list, Show / Hide, the keys in mono
    if (!o.locked.empty()) {
        // no count: Show lists what they do, and several keys do one thing ("Upscaling goes to WrongGPU"), so a count of keys
        // would not match the list
        const wstring t = L"WrongGPU manages some of OptiScaler's settings for you.";
        const wstring link = v.lockedOpen ? L"Hide" : L"Show";
        const TS fs = Ts(12.5f, 400, pal.text2);
        const float tw = std::min(w - 32 - 60, gx.TextW(t, fs) + 1);
        gx.Text(t, Rc{x + kCellPad, y + 8, tw + 1, 18}, fs);
        AddText(Rc{x + kCellPad, y + 8, tw + 1, 18}, t);
        Button(gx, ID_G_LOCKED, x + kCellPad + tw + 8 - 6, y + 8 + 9 - 14, link, B_LINK, 0, true, 28, 0, false, L"Managed by WrongGPU: " + link);
        y += 8 + 18;
        if (v.lockedOpen) {
            // what they do, in words ("Upscaling goes to WrongGPU · No extra sharpening · ..."), once each; the ini lines
            // themselves in the tooltip (SHIP-BAR 4: no developer text on the surface)
            std::vector<wstring> said;
            wstring lines;
            for (auto& r : o.locked) {
                lines += (lines.empty() ? L"" : L"\n") + (L"[" + FromUtf8(r.section) + L"] " + FromUtf8(r.key) + L" = " + FromUtf8(r.value));
                const wstring wd = ManagedWords(r.section, r.key);
                if (std::find(said.begin(), said.end(), wd) == said.end()) said.push_back(wd);
            }
            wstring words;
            for (wstring wd : said) {  // a line breaks only between them
                for (auto& ch : wd) if (ch == L' ') ch = L' ';
                words += (words.empty() ? L"" : L"  ·  ") + wd;  // a break after the dot, never before it
            }
            const TS ms = Wrap(12.5f, pal.text2, 18);
            const float mh = gx.TextH(words, w - 2 * kCellPad, ms);
            y += 10;
            gx.FillRound(Rc{x, y, w, mh + 20}, 10, pal.fill);
            gx.Text(words, Rc{x + kCellPad, y + 10, w - 2 * kCellPad, mh + 4}, ms);
            const int tid = AddText(Rc{x + kCellPad, y + 10, w - 2 * kCellPad, mh + 4}, words);
            if (tid) v.tips[tid] = lines;
            y += mh + 20;
        }
    }
    return y + kSectionGap;
}

// OptiScaler before WrongGPU's setup is in the game: it only reports (in the right column): what happens at the install,
// and the name it will load as.
static float OptiReportSection(Gfx& gx, float x, float y, float w) {
    const GameEntry& e = v.cur;
    wstring tag; Col tc = pal.text2;
    if (e.status == GameStatus::PlayersOptiScaler) { tag = L"Has your own OptiScaler"; tc = pal.amberText; }  // the Library's tile says the same words
    else tag = e.hasRecord ? L"Not installed" : L"Comes with Install";
    y += HeadingTag(gx, x, y, w, L"OptiScaler", tag, tc) + kHeadGap;
    wstring t;
    if (e.status == GameStatus::PlayersOptiScaler)  // what Install does about it (the tag says whose it is, once)
        t = L"Install switches it to WrongGPU's OptiScaler. Yours is backed up.";
    else if (e.hasRecord) t = L"WrongGPU's OptiScaler setup isn't installed in this game.";
    else t = L"WrongGPU runs through OptiScaler, from the OptiScaler download you choose. Its settings can be changed here once WrongGPU is installed.";
    y += GroupList(gx, x, y, w, [&](float top) {
        float yy = top;
        yy += DotRow(gx, x, yy, w, true, t, Col{0, 0, 0, 0}, pal.text2);
        if (!e.hasRecord && !e.proxyPlan.name.empty() && e.status != GameStatus::Refused && e.status != GameStatus::ForeignMod &&
            PlanNotable(e.proxyPlan.reason, e.proxyPlan.name)) {
            // the name in words, why under it when it is not the usual one; the file names (its own, the ones taken) in the tooltip
            const wstring& why = e.proxyPlan.reason;
            KvOpt lo; lo.tip = PlanTip(why, e.proxyPlan.name);
            lo.sub = PlanSub(why, e.proxyPlan.name);
            const size_t taken = why.find(L"taken: ");
            if (taken != wstring::npos) {
                wstring names = why.substr(taken + 7);
                const size_t cut = names.find(L" (next to ReShade");
                if (cut != wstring::npos) names.resize(cut);
                lo.tip += L"\nTaken: " + names;
            }
            yy += KvRow(gx, x, yy, w, false, L"Installs as", PlanName(why, e.proxyPlan.name), lo);
        }
        return yy - top;
    });
    return y + kSectionGap;
}

// A warning of the last run in the player's words, by its kind (the core's sentence is for the CLI and the logs: the raw
// lines behind Show have it). Empty: nothing to say (the list's foot says it already).
static wstring RunWarning(const wstring& kind, const wstring& text, bool footSaysIt) {
    if (kind == L"plugin-not-hooked") return footSaysIt ? wstring() : L"This OptiScaler version lists WrongGPU under AMD FSR in its menu. Nothing else changes.";
    if (kind == L"foreign-vtable") return L"An overlay or another mod (ReShade, Special K) took over some frames, so they used AMD FSR.";
    if (kind == L"xess-no-depth") return L"With XeSS chosen in the game, Model M gets less from the game and may look worse. Choose DLSS or FSR in the game's settings.";
    if (kind == L"no-depth") return L"The game gives the upscaler less than it needs, so Model M may look worse.";
    (void)text;  // a kind this launcher does not know: the core's own sentence stays behind Show, with the log's lines
    return L"WrongGPU noted a warning in this run. Select Show to read it.";
}

// The Denoiser row's words under "Ray Reconstruction" where the game asked for it but no frame succeeded (r6 #2): the RR log's
// reason and its fallback when it has them. mode: its mode's name first, on a line of its own (the RR-only row)
// The game asked for Ray Reconstruction in the last run and no frame of it succeeded (r6 #2): the page's amber row and the
// header's dot say so (r7 #2)
static bool RrFailedRun(const LastRun& lr) { return lr.found && lr.rrUsed && lr.rrFrames <= 0; }
// That amber row: what happened and the next step. The runtime's first-start case (its kernels still being prepared, a cold driver
// cache) clears by itself: play again; anything else: start again, and Collect logs if it happens again
static wstring RrFailedWarning(const LastRun& lr) {
    wstring s = L"Ray Reconstruction didn't finish a frame in this run" + wstring(lr.rrPassThrough ? L", so the game showed its image without it." : L".");
    if (lr.rrNote.find(L"kernels still being prepared") != wstring::npos) return s + L" The first start prepares its kernels. Play again; if it happens again, use Collect logs.";
    return s + L" Start the game again; if it happens again, use Collect logs.";
}
static wstring RrNoFrameText(const LastRun& lr, const wstring& mode) {
    wstring s = mode.empty() ? wstring() : mode + L"\n";
    s += L"No successful frame recorded in the log.";
    if (!lr.rrNote.empty()) s += L"\nThe log says: " + lr.rrNote + L".";
    if (lr.rrPassThrough) s += (lr.rrNote.empty() ? L"\n" : L" ") + wstring(L"The game's image was passed through without Ray Reconstruction.");
    return s;
}

// Last run: the time on the heading's right; a read-out (upscaler, runs on, the size in and out), warnings as sentences,
// the log's count with the raw lines behind Show; Open log and Collect logs as action rows.
static float LastRunSection(Gfx& gx, float x, float y, float w) {
    const LastRun& lr = v.cur.last;
    const wstring when = lr.found ? When(lr.when) : L"";
    const TS wts = Ts(13, 400, pal.text2);
    const float ww = when.empty() ? 0.f : gx.TextW(when, wts) + 2;
    GroupHeading(gx, x, y, w, L"Last run", L"", ww + kCellPad + 8);
    if (!when.empty()) { TS r = wts; r.align = 2; gx.Text(when, Rc{x + w - kCellPad - ww, y + 3, ww, 18}, r); AddText(Rc{x + w - kCellPad - ww, y + 3, ww, 18}, L"Last run: " + when); }
    y += 22 + kHeadGap;
    const bool raw = lr.found && (!lr.notes.empty() || !lr.support.empty() || !lr.warnings.empty());
    y += GroupList(gx, x, y, w, [&](float top) {
        float yy = top;
        if (!lr.found) {
            // the game's own store when it has one ("from Steam as usual"), else just "as usual"
            const wstring store = StoreOf(v.cur);
            const bool shop = store == L"Steam" || store == L"Epic" || store == L"GOG";
            const wstring t = L"Not played yet. Start it " + (shop ? L"from " + store + L" " : wstring()) + L"as usual" +
                              wstring(gl.global.playButton ? L" or with Play," : L"") + L" and its first run shows here.";
            yy += DotRow(gx, x, yy, w, true, t, Col{0, 0, 0, 0}, pal.text2);
            // More's glyph for it (a zip: the logs are packed into one file), not the external one: it opens no outside place
            yy += CollectRow(gx, x, yy, w);  // waits for an operation (it reads its files)
            return yy - top;
        }
        auto times = [](const wstring& s) { wstring o; for (wchar_t ch : s) o += ch == L'x' ? wstring(L"\u202f\u00d7\u202f") : wstring(1, ch); return o; };
        // the run known from the RR log alone (r4 #1: Borderlands 4, Hogwarts Legacy and RE Requiem run Ray Reconstruction only, and
        // the front end logs nothing there): the Denoiser row leads, with its mode and cost, then the size; no upscaler rows
        if (lr.rrOnly) {
            KvOpt dn;
            wstring m = lr.rrMode.empty() ? wstring() : RrModeName(lr.rrMode);
            if (lr.rrFrames <= 0) m = RrNoFrameText(lr, m);  // the game asked for it, no frame succeeded (r6 #2)
            else if (!lr.rrMs.empty()) m += (m.empty() ? L"" : L" \u00b7 ") + lr.rrMs + L" ms a frame";
            if (!v.rrReady) m += (m.empty() ? L"" : L"\n") + wstring(L"Its data is missing now, so the next run uses the game's own denoiser.");
            dn.sub = m;
            yy += KvRow(gx, x, yy, w, true, L"Denoiser", L"Ray Reconstruction", dn);
            if (!lr.renderSize.empty() && !lr.outputSize.empty()) yy += KvRow(gx, x, yy, w, false, L"Size", times(lr.renderSize) + L" \u2192 " + times(lr.outputSize));
            if (RrFailedRun(lr)) yy += DotRow(gx, x, yy, w, false, RrFailedWarning(lr), pal.amber, pal.text);  // r7 #2
            yy += ActRow(gx, x, yy, w, false, false, ID_G_LOG, L"Open log", ic::external, Rec());
            yy += CollectRow(gx, x, yy, w);
            return yy - top;
        }
        // Model M set up but not started: the game used AMD FSR, so that is the upscaler of the run, and what didn't start is
        // said under it (the row and the warning below tell one story)
        // A run where the game never asked for an upscaler (no context, no frame): WrongGPU wasn't used, and nothing failed. The
        // usual cause is the game's own setting (TAA, or upscaling off): what to choose there (UI review #9)
        const bool unused = lr.upscaler == L"network" && !lr.ready && !lr.created;
        const bool noStart = lr.upscaler == L"network" && !lr.ready && !unused;
        const bool turnedOff = lr.upscaler == L"amd";  // Use WrongGPU was off in that run (wronggpu.ini upscaler = amd)
        // Ray Reconstruction did the upscaling in that session, Model M none (r6 #1: it denoises and upscales in one pass): the row
        // names it, and nothing says Model M wasn't used or can't be confirmed (RR's frames are still never Model M's evidence)
        const bool rrUp = lr.rrUpscaled;
        const wstring up = rrUp ? L"Ray Reconstruction" : unused ? L"Not used" : turnedOff || noStart ? L"AMD FSR" : lr.ready ? L"Model M" : L"\u2014";
        const wstring ex = lr.ready && !rrUp && !lr.executor.empty() ? RunsOnName(lr.executor) : L"\u2014";
        KvOpt uo;
        // the runtime started, but the log shows no frame it upscaled (it logs "ready" before any frame): not a plain Model M (r3 #3)
        // (upscaled is the front end's own evidence here: RR's frames never count for Model M's row: r4 #1, r5 #3)
        const bool noFrame = lr.ready && !lr.upscaled && !turnedOff && !rrUp;
        if (rrUp) uo.sub = L"It denoises and upscales in one pass while it's on.";
        else if (noStart) uo.sub = L"Model M didn't start";
        // (only a context's first frames are logged unless log_dispatches is on, and those can go to AMD FSR while Model M starts:
        // a working late start looks the same, so this says what the log shows, not that nothing was upscaled: r4 #2. Never that
        // a right-looking game means Model M ran: AMD FSR looks right too, r5 #6)
        else if (noFrame) uo.sub = L"Started; no upscaled frame recorded in the log.\nThe log records only the first few frames, which can go to AMD FSR "
                                   L"while Model M starts, so it can't confirm whether Model M upscaled the frames after them.";
        // the game asked for Ray Reconstruction instead (it needs DLSS on already): nothing to choose, no warning (r7 #1)
        else if (unused && lr.rrUsed) uo.sub = L"Ray Reconstruction was asked for instead";
        else if (unused) uo.sub = L"The game didn't ask for an upscaler";
        else if (turnedOff) uo.sub = L"Use WrongGPU was off";
        yy += KvRow(gx, x, yy, w, true, L"Upscaler", up, uo);
        yy += KvRow(gx, x, yy, w, false, L"Runs on", ex);
        if (v.cur.rrInstalled) {  // the RR log of that run: the mode and its measured cost (none when the log has none)
            KvOpt dn;
            if (lr.rrUsed) {
                wstring m = lr.rrMode.empty() ? wstring() : RrModeName(lr.rrMode);  // its mode line can be out of the log's tail
                if (lr.rrFrames <= 0) m = RrNoFrameText(lr, m);  // an instance made (or tried), no frame succeeded (r6 #2)
                else if (!lr.rrMs.empty()) m += (m.empty() ? L"" : L" \u00b7 ") + lr.rrMs + L" ms a frame";
                // the RR data gone since that run: the page's Ray Reconstruction line says so, and this says it too
                if (!v.rrReady) m += (m.empty() ? L"" : L"\n") + wstring(L"Its data is missing now, so the next run uses the game's own denoiser.");
                dn.sub = m;
            } else if (lr.rrFound) dn.sub = L"The game didn't ask for it in this run.";  // the RR plugin ran, the game never made an instance
            yy += KvRow(gx, x, yy, w, false, L"Denoiser", lr.rrUsed ? L"Ray Reconstruction" : L"Not used", dn);
        }
        if (!lr.renderSize.empty() && !lr.outputSize.empty()) yy += KvRow(gx, x, yy, w, false, L"Size", times(lr.renderSize) + L" \u2192 " + times(lr.outputSize));
        else if (!lr.renderSize.empty()) yy += KvRow(gx, x, yy, w, false, L"Input", times(lr.renderSize));
        else if (!lr.outputSize.empty()) yy += KvRow(gx, x, yy, w, false, L"Output", times(lr.outputSize));
        // warnings in plain language (the core's own sentences, for the CLI and the logs, behind Show)
        std::vector<wstring> warn;
        const bool foot = v.cur.hosted && !lr.optiPlugin && lr.ready;  // the list's foot says it already (plugin-not-hooked)
        for (auto& wr : lr.warnings) {
            const wstring t = RunWarning(wr.kind, wr.text, foot);
            if (!t.empty() && std::find(warn.begin(), warn.end(), t) == warn.end()) warn.push_back(t);  // each sentence once
        }
        // the game asked for Ray Reconstruction and no frame succeeded: an amber row with the next step (r7 #2), after Model M's own
        if (RrFailedRun(lr)) warn.insert(warn.begin(), RrFailedWarning(lr));
        if (noStart) warn.insert(warn.begin(), wstring(L"Model M didn't start in this run, so the game used AMD FSR.") + (raw ? L" Select Show to read the log's lines." : L" Open log says why."));
        // what to choose follows the route, as the checklist's step 2 says it: DLSS through OptiScaler (FSR or XeSS also work),
        // AMD FSR on the direct route (WrongGPU is the game's FSR there; nothing else reaches it) (r2 #3)
        // Beside Ray Reconstruction's evidence (the game asked for RR, which needs the game's DLSS on already) there is none: the
        // Upscaler row says RR was asked for instead (r7 #1; r6 #1: no advice to choose DLSS). With rrUp there is no warning at all
        if (unused && !rrUp && !lr.rrUsed)
            warn.insert(warn.begin(), wstring(L"WrongGPU wasn't used in this run: the game didn't ask for an upscaler. In the game's graphics settings, ") +
                                          (v.cur.hosted ? L"choose DLSS as the upscaler (FSR or XeSS also work), then play again."
                                                        : L"choose AMD FSR as the upscaler, then play again."));
        for (auto& t : warn) yy += DotRow(gx, x, yy, w, false, t, pal.amber, pal.text);
        // the log: its count (or none), the raw lines behind Show
        KvOpt lo;
        wstring lv;
        if (raw) { lo.linkId = ID_G_DETAILS; lo.link = v.detailsOpen ? L"Hide" : L"Show"; lo.linkName = (v.detailsOpen ? L"Hide" : L"Show") + wstring(L" the log's lines"); }
        // the log's notes (fallbacks, foreign command lists, errors: lines without a kind the launcher could act on): counted
        // in grey, the lines behind Show; amber is for the warnings above, which say what happened and what to do
        if (!lr.notes.empty()) { lv = std::to_wstring(lr.notes.size()) + (lr.notes.size() == 1 ? L" note in the log" : L" notes in the log"); lo.dot = pal.greyDot; }
        else if (!lr.support.empty()) { lv = L"Support code " + lr.support; lo.dot = pal.greyDot; }
        else if (warn.empty() && (lr.ready || rrUp)) { lv = L"No warnings"; lo.dot = noFrame ? pal.greyDot : pal.green; }  // green once a frame was upscaled (r3 #3)
        else if (raw) lv = lr.warnings.size() == 1 ? L"More about the warning" : L"More about the warnings";  // the core's sentences behind Show
        if (!lv.empty()) yy += KvRow(gx, x, yy, w, false, L"Log", lv, lo);
        if (v.detailsOpen && raw) {
            wstring lines;
            for (auto& n : lr.notes) lines += n + L"\n";
            for (auto& wr : lr.warnings) lines += wr.kind + L": " + wr.text + L"\n";
            if (!lr.support.empty()) lines += L"support code: " + lr.support + L"\n";
            if (!lr.plugin.empty()) lines += L"plugin: " + lr.plugin + L"\n";
            while (!lines.empty() && lines.back() == L'\n') lines.pop_back();  // no empty last line
            yy += MonoLines(gx, x, yy - 4, w, lines) - 4;
        }
        yy += ActRow(gx, x, yy, w, false, false, ID_G_LOG, L"Open log", ic::external, Rec() && lr.found);  // as More's Open log: once there is a log
        yy += CollectRow(gx, x, yy, w);
        return yy - top;
    });
    if (lr.found && v.cur.hosted && !lr.optiPlugin && lr.ready)  // (plugin-not-hooked says the same: no row for it then)
        y += GroupFoot(gx, x, y, w, L"This OptiScaler version doesn't list WrongGPU by name: its menu shows AMD FSR where Model M runs. Nothing else changes.");
    return y + kSectionGap;
}

// Why WrongGPU stays out of a game it is not offered for (refused, another upscaler mod), in the player's words: the card's
// sentence (NotOfferedSection); *tip: the anti-cheat's marker. The Game folder card leaves out the finding it names.
static wstring StaysOutWhy(const GameEntry& e, wstring* tip) {
    wstring why = e.statusText.rfind(L"Refused: ", 0) == 0 ? e.statusText.substr(9) : e.statusText;
    if (!why.empty()) why[0] = towupper(why[0]);
    // anti-cheat: the Library's sentence ("It uses Easy Anti-Cheat. WrongGPU stays out of games with anti-cheat."); the
    // marker's full path only in the tooltip and for UI Automation
    const wstring shortStatus = e.status == GameStatus::Refused ? ShortStatus(e) : wstring();
    if (shortStatus.rfind(L"Anti-cheat", 0) == 0) {
        why = (shortStatus.rfind(L"Anti-cheat: ", 0) == 0 ? L"It uses " + shortStatus.substr(12) : wstring(L"It uses anti-cheat")) + L". WrongGPU stays out of games with anti-cheat.";
        for (auto& f : e.coexist) if (f.tool == Tool::AntiCheat && !f.evidence.empty() && tip) *tip = L"Found: " + f.evidence;
    }
    // the status line's own lead-in ("Other upscaler mod: X", "Other upscaler mod (X): not touched"): the mod's name
    if (e.status == GameStatus::ForeignMod && why.rfind(L"Other upscaler mod: ", 0) == 0) why = why.substr(20);
    else if (e.status == GameStatus::ForeignMod && why.rfind(L"Other upscaler mod (", 0) == 0 && why.find(L')') != wstring::npos) why = why.substr(20, why.find(L')') - 20);
    // the headline says another mod is here: the card names it and says why (not the headline again)
    if (e.status == GameStatus::ForeignMod) why = why + L" is in this game. WrongGPU stays out so the two don't fight over the upscaler.";
    else why += why.empty() || why.back() == L'.' ? L"" : L".";
    return why;
}

// Game folder: where the game is, what is wrong with WrongGPU's files and what fixes it, what else is in the folder and
// what WrongGPU does about it; Open folder.
static float FolderSection(Gfx& gx, float x, float y, float w) {
    const GameEntry& e = v.cur;
    const bool notOffered = !e.hasRecord && (e.status == GameStatus::Refused || e.status == GameStatus::ForeignMod);
    y += GroupHeading(gx, x, y, w, L"Game folder") + kHeadGap;
    y += GroupList(gx, x, y, w, [&](float top) {
        float yy = top;
        // where it is as a place in the UI font ("Cyberpunk 2077 › bin › x64"); the whole path and the game's file in the
        // tooltip and for UI Automation (LIVE-FINDINGS 15); Open folder below shows it in File Explorer
        KvOpt place;
        place.textId = ID_G_DIRTEXT;
        place.tip = e.dir;
        if (!e.exe.empty()) place.tip += L"\nThe game: " + (IEq(Parent(e.exe), e.dir) ? BaseName(e.exe) : e.exe);
        yy += KvRow(gx, x, yy, w, true, L"Folder", Place(e.dir), place);
        auto note = [&](const wstring& t, Col dot, const wstring& tip = L"") { yy += DotRow(gx, x, yy, w, false, t, dot, pal.text, tip); };
        // WrongGPU's files that are missing or changed: one row with their count, their names in the tooltip and for UI
        // Automation (no file names on the surface, the advice said once); the settings file and anything else as before.
        // What the header's line says already (a game that needs a repair: Status) is not said again: the row has the remedy
        // ("Repair puts WrongGPU's settings file back.").
        size_t nFiles = 0;
        wstring filesTip;
        const wstring filesT = FilesProblem(e, &nFiles, &filesTip);
        const bool headSays = e.status == GameStatus::NeedsRepair;
        const wstring headLine = headSays ? (filesT.empty() ? ProblemShort(e) : filesT) : wstring();
        for (auto& p : e.problems)
            if (p.rfind(L"missing: ", 0) != 0 && p.rfind(L"changed: ", 0) != 0) {
                const wstring s = ProblemSentence(p);
                if (IsMisplacedProblem(p)) note(s, pal.amber);  // it says what Repair does (rc6 L2)
                else if (p == L"wronggpu.ini is missing") note(s == headLine + L"." ? L"Repair puts WrongGPU's settings file back." : s + L" Repair puts it back.", pal.amber);
                else note(s == headLine + L"." ? L"Repair puts WrongGPU's files back." : s + L" Repair puts WrongGPU's files back.", pal.amber);
            }
        if (nFiles > 0)
            note(headLine == filesT ? (nFiles == 1 ? L"Repair puts WrongGPU's file back." : L"Repair puts WrongGPU's " + std::to_wstring(nFiles) + L" files back.")
                                    : filesT + (nFiles == 1 ? L". Repair puts it back." : L". Repair puts them back."),
                 pal.amber, filesTip);
        if (e.recordV1) note(L"Installed by the older WrongGPU setup (0.5). Repair or Update moves it to the launcher.", pal.greyDot);
        if (!e.leftovers.empty()) {
            // what they are in the player's words; their names in the tooltip and for UI Automation
            std::vector<wstring> vers;
            wstring names;
            for (auto& f : e.leftovers) {
                if (std::find(vers.begin(), vers.end(), f.version) == vers.end()) vers.push_back(f.version);
                names += (names.empty() ? L"" : L", ") + f.rel;
            }
            wstring ver;
            for (size_t k = 0; k < vers.size(); ++k) ver += (k == 0 ? L"" : k + 1 == vers.size() ? L" and " : L", ") + vers[k];
            const size_t n = e.leftovers.size();
            const wstring t = n == 1 ? L"A file from WrongGPU " + ver + L" is here that the launcher didn't install. Clean up removes only it; Adopt lets the launcher manage it."
                                     : std::to_wstring(n) + L" files from WrongGPU " + ver + L" are here that the launcher didn't install. Clean up removes only these; Adopt lets the launcher manage them.";
            const float rowY = yy;
            note(t, pal.amber);
            AddHit(ID_G_FOUND, Rc{x, rowY, w, yy - rowY}, R_TEXT, t + L" Files: " + names, true, -1, L"", false);
            v.tips[ID_G_FOUND] = (n == 1 ? L"File: " : L"Files: ") + names;
        }
        // the game's file is gone, but the header does not say so (it says it for a game that is gone: Uninstall)
        if (!e.exe.empty() && !v.fixture && !e.exeFound && e.status != GameStatus::Orphaned) note(L"The game was moved or uninstalled.", pal.amber);
        // what else is in the game folder, and what WrongGPU does about it, in the player's words (WordsOf); the core's
        // text, file names and folders in the tooltip
        bool firstFound = true;
        // a game WrongGPU stays out of: the finding the card above names ("It uses GameGuard. ...") is not listed again
        const wstring staysOut = notOffered ? StaysOutWhy(e, nullptr) : wstring();
        for (auto& f : e.coexist) {
            if (f.decision == Decision::Managed) continue;
            const FindingWords fw = WordsOf(f);
            if (notOffered && f.decision == Decision::Refuse && fw.name.size() > 2 && staysOut.find(fw.name) != wstring::npos) continue;
            KvOpt fo;
            fo.sub = notOffered && f.decision == Decision::Refuse ? wstring() : fw.sub;  // the reason is the section above's
            // a warning about how WrongGPU and the tool would meet (a HIP tool's "Keep Runs on at Auto..."): moot in a game
            // WrongGPU stays out of, where the page has no setting it could name
            const bool moot = notOffered && f.decision == Decision::Warn;
            if (moot) fo.sub = L"Nothing to do: WrongGPU stays out of this game.";
            fo.tip = fw.tip;
            // the tag says what WrongGPU does about it, in the player's words: none for what only informs, nor for what keeps it out
            // of a game it is not offered for (the card above says why: "Why WrongGPU stays out"), nor for a moot warning
            if (f.decision != Decision::Inform && !(notOffered && f.decision == Decision::Refuse) && !moot) {
                // a tool that stays: "Works next to it", or "Untested pair" where that pair is untested (ReShade, Special K:
                // the row's sentence says what happens then)
                const bool untested = f.tool == Tool::ReShade || f.tool == Tool::SpecialK || f.message.find(L"untested") != wstring::npos;
                fo.tag = f.decision == Decision::Refuse ? L"Stays out" : f.decision == Decision::AskSwitch || f.decision == Decision::Migrate ? L"Asks first"
                       : f.decision == Decision::Warn ? L"Needs a look" : f.decision == Decision::Chain ? (untested ? L"Untested pair" : L"Works next to it") : L"";
                // a tool that stays in grey: green says WrongGPU is working (IDENTITY.md), never a verdict on another tool
                fo.tagFg = f.decision == Decision::Chain || f.decision == Decision::Refuse ? pal.text2 : pal.amberText;
            }
            // a word in brackets after the name ("Special K (global injection)") stays after it in text2 when it fits; a
            // folder in brackets ("Cyber Engine Tweaks (bin\x64\plugins)") goes to the tooltip, as files do
            wstring what = fw.name;
            const size_t open = what.rfind(L" (");
            if (open != wstring::npos && open > 0 && what.back() == L')') {
                const wstring where = what.substr(open + 2, what.size() - open - 3);
                if (where.find(L'\\') != wstring::npos || where.find(L'/') != wstring::npos || where.find(L":") != wstring::npos) fo.tip = L"In " + where + (fo.tip.empty() ? L"" : L"\n" + fo.tip);
                else fo.suffix = L"(" + where + L")";
                what.resize(open);
            }
            yy += KvRow(gx, x, yy, w, false, firstFound ? L"Also here" : L"", what, fo);
            firstFound = false;
        }
        yy += ActRow(gx, x, yy, w, false, true, ID_G_FOLDER, L"Open folder", ic::external, true);
        return yy - top;
    });
    return y + kSectionGap;
}

// A game WrongGPU is not offered for (refused, another upscaler mod): the reason in plain words in a grey row (a final
// "no", not a warning: the accent is for actions, amber for what needs a look).
static float NotOfferedSection(Gfx& gx, float x, float y, float w) {
    const GameEntry& e = v.cur;
    wstring tip;
    const wstring why = StaysOutWhy(e, &tip);
    y += GroupHeading(gx, x, y, w, L"Why WrongGPU stays out") + kHeadGap;
    y += GroupList(gx, x, y, w, [&](float top) {
        // a glyph tile, the reason beside it (its measure capped, as everywhere)
        const float tx = x + 64, tw = std::min(640.f, x + w - kCellPad - tx);
        const TS ws = Wrap(13.5f, pal.text, 19);
        const wstring note = L"WrongGPU changes nothing in this game.";
        const float wh = gx.TextH(why, tw, ws), nh = gx.TextH(note, tw, Wrap(12.5f, pal.text2, 17));
        const float h = std::max(kSRowH, wh + 2 + nh + 24);
        const Rc tile{x + kCellPad, top + 13, 28, 28};
        gx.FillRound(tile, 7, pal.fill);
        gx.Icon(ic::shield, tile.cx(), tile.cy(), 14, pal.text2);
        gx.Text(why, Rc{tx, top + 12, tw, wh + 2}, ws);
        gx.Text(note, Rc{tx, top + 12 + wh + 2, tw, nh + 2}, Wrap(12.5f, pal.text2, 17));
        const int tid = AddText(Rc{tx, top + 12, tw, wh + 2}, why + (tip.empty() ? L"" : L" " + tip));
        if (tid && !tip.empty()) v.tips[tid] = tip;
        AddText(Rc{tx, top + 12 + wh + 2, tw, nh + 2}, note);
        return h;
    });
    return y + kSectionGap;
}

// A tool of the player's the game needs before WrongGPU can work in it (REFramework for RE Requiem, gamerules.h), missing,
// before an install or since one. The header says what the game needs and why; this says what to do, once, in plain words:
// the amber-tinted cell with the glyph, a title and a line, and the actions in one place: Learn more (the tool's public
// page in the browser; the page's one filled button, Install being held back and Play grey) and Check again (this game's
// folder read again, said as a toast). WrongGPU never installs, downloads or ships the tool.
static float NeedsCallout(Gfx& gx, float x, float y, float w) {
    const GameEntry& e = v.cur;
    const wstring title = e.hasRecord ? e.needs + L" isn't in the game's folder any more" : L"Add " + e.needs + L" to the game first";
    const wstring text = e.hasRecord ? L"WrongGPU can't run in the game until it's back. Add it to the game's folder again, then choose Check again."
                                     : L"WrongGPU doesn't install it for you. Add it to the game's folder, then choose Check again.";
    const TS ts = Wrap(13, pal.text, 18);
    const float avail = std::min(640.f, w - 40 - 16);
    const float th = gx.TextH(text, avail, ts);
    const Rc r{x, y, w, 12 + 20 + 3 + th + 12 + 30 + 14};
    gx.FillRound(r, kGroupR, Mix(pal.cell, pal.amber, pal.dark ? 0.12f : 0.08f));  // the lists' radius
    gx.Icon(ic::warn, r.x + 21, r.y + 22, 13, pal.amberText);
    gx.Text(title, Rc{r.x + 40, r.y + 12, avail, 20}, Ts(14, 600, pal.text));
    gx.Text(text, Rc{r.x + 40, r.y + 12 + 23, avail, th + 4}, ts);
    AddText(Rc{r.x + 40, r.y + 12, avail, 23 + th}, title + L". " + text);
    const float by = r.y + 12 + 23 + th + 12;
    const float lw = Button(gx, ID_G_LEARN, r.x + 40, by, L"Learn more", B_GO, ic::external, true, 30, 0, false, L"Learn more about " + e.needs);
    v.tips[ID_G_LEARN] = L"Opens " + e.needs + L"'s page in your browser.";
    Describe(ID_G_LEARN, v.tips[ID_G_LEARN]);
    // reading the folder while an operation writes it would show a step half done: it waits for that one
    const bool can = !BusyHere();
    Button(gx, ID_G_RECHECK, r.x + 40 + lw + 8, by, L"Check again", B_PLAIN, 0, can, 30, 0, false, L"Check again for " + e.needs);
    const wstring help = can ? L"Looks for " + e.needs + L" in the game's folder again." : BusyWhy(e);
    if (!can) v.tips[ID_G_RECHECK] = help;
    Describe(ID_G_RECHECK, help);
    return r.h;
}

// The sections under the header, from `y` (content DIPs). `pinAt`: the scroll at which the compact header pins. Returns
// the page's content height.
static float Cards(Gfx& gx, float x0, float cw, float y, float pinAt) {
    if (!cv.dry) v.rrcfgCardY = -1;  // set again where the Engine.ini card is drawn (RrConfigCard)
    // room under the last section: a toast's (as Settings has it), and the floating capsule's band while it shows. When the
    // capsule goes (Discard, an Apply) its band stays while it is in view: the page never drops by its height under a page
    // scrolled to its end (SHIP-BAR 3: no layout jumps), and the room goes once it is scrolled out of view
    auto tail = [](float end) { return end + (Dirty() ? 140.f : std::max(96.f, std::min(140.f, v.scroll[P_GAME] + v.H - end))); };
    // a failed operation's note, on the page of the game it ran on only (a success is a toast: one pattern for it)
    // (30 under it: the next heading as far under a callout as under the header)
    if (!v.notice.empty() && IEq(v.noticeDir, v.cur.dir)) y += Callout(gx, ID_G_NOTICE, x0, y, cw, v.notice, v.noticeTone, L"Dismiss") + 30;
    // a tool of the player's the game needs and does not have (REFramework for RE Requiem): first, before the settings
    if (!v.cur.needs.empty() && !v.cur.needsFound) y += NeedsCallout(gx, x0, y, cw) + 30;
    // the page shown before its files are read (a slow or sleeping disk: OpenGame): the settings wait, said in a quiet row with
    // the busy indicator, and fill in when the read lands
    if (v.pageReading) {
        y += GroupList(gx, x0, y, cw, [&](float top) {
            const wstring t = L"Reading the game's settings\u2026";
            BusyIndicator(gx, x0 + kCellPad + 7, top + kSRowH / 2, 7, pal.text2);
            TS ts = Ts(13.5f, 400, pal.text2); ts.valign = 1; ts.trim = true;
            gx.Text(t, Rc{x0 + kCellPad + 24, top, cw - 2 * kCellPad - 24, kSRowH}, ts);
            AddText(Rc{x0 + kCellPad + 24, top, cw - 2 * kCellPad - 24, kSRowH}, t);
            return kSRowH;
        });
        return tail(y);
    }
    if (!v.cur.hasRecord && (v.cur.status == GameStatus::Refused || v.cur.status == GameStatus::ForeignMod)) {
        // nothing to set up here: why, and what is in the folder
        y = NotOfferedSection(gx, x0, y, cw);
        y = FolderSection(gx, x0, y, cw);
        return tail(y - kSectionGap);
    }
    // what can be set on the left (the wider column, 560 of 920), what is read on the right (OptiScaler too while it has
    // nothing to set: before an install); below 980 DIPs of column one column, by importance: Upscaler, Last run,
    // OptiScaler, Game folder (a right column under 360 DIPs would cut the paths and file names it is there to show); Ray
    // Reconstruction, in a game that uses it, under the Upscaler in both
    const bool one = cw < 980;
    const float lw = one ? cw : std::floor((cw - kColGap) * 560.f / (920.f - kColGap)), rw = cw - kColGap - lw;
    // after an install, until Model M has run: what to do in the game, first (UI review #1); the right column (Last run)
    // stays level with it, so the steps and their result sit side by side (one column: the steps lead, full width)
    float ly = y;
    if (GuideShown()) { ly = GuideSection(gx, x0, ly, one ? cw : lw); if (one) y = ly; }
    ly = UpscalerSection(gx, x0, ly, lw);
    if (RrShown()) ly = RrSection(gx, x0, ly, lw);
    else if (v.cur.rrNotOffered) ly = RrNotOfferedSection(gx, x0, ly, lw);  // rc5: LEGO Batman (gamerules.h rrNotOffered)
    else if (Rec() && RrConfigGameFor(v.cur.exe)) {  // the Engine.ini setting under the section's heading (UI review #2)
        ly += GroupHeading(gx, x0, ly, lw, L"Ray Reconstruction") + kHeadGap;
        ly = RrConfigCard(gx, x0, ly, lw) + kSectionGap;
    }
    if (one) {
        if (v.cur.hasRecord) ly = LastRunSection(gx, x0, ly, cw);
        ly = v.opti.have ? OptiSection(gx, x0, ly, cw) : OptiReportSection(gx, x0, ly, cw);
        ly = FolderSection(gx, x0, ly, cw);
        return tail(ly - kSectionGap);
    }
    const bool optiLeft = v.opti.have;
    if (optiLeft) ly = OptiSection(gx, x0, ly, lw);
    const float rx = x0 + lw + kColGap;
    auto right = [&](float top) -> float {
        float yy = top;
        if (v.cur.hasRecord) yy = LastRunSection(gx, rx, yy, rw);
        if (!optiLeft) yy = OptiReportSection(gx, rx, yy, rw);
        yy = FolderSection(gx, rx, yy, rw);
        return yy - kSectionGap - top;
    };
    // the right column stays in view beside the settings: once the header is pinned it stops under it (taller than the
    // room between the header and the window's bottom: it scrolls until its end is in view and stops there) and goes on
    // up with the page only where the left column ends, so neither side is left empty. It stops 18 DIPs under the
    // header, or nearer where it already is nearer as the header pins (it never moves down: no jump at that moment).
    const float rh = Measure(gx, [&] { return right(y); });
    const float gap = std::max(0.f, std::min(18.f, y - pinAt - v.pinH));
    const float room = v.H - v.pinH - gap - GameBarH() - 16;
    float ry = y;
    if (v.pinH > 0) ry = std::max(y, std::min(cv.scroll + v.pinH + gap + std::min(0.f, room - rh), ly - kSectionGap - rh));
    // the same rule at any scroll, with the pinned header's height whether it is pinned now or not: Tab into this column
    // scrolls by where it will be (Reveal), so a control is never left under the header
    if (!cv.dry) {
        const float pinH = v.pinH > 0 ? v.pinH : cv.view.y + kPinH;
        const float pgap = std::max(0.f, std::min(18.f, y - pinAt - pinH)), proom = v.H - pinH - pgap - GameBarH() - 16;
        v.stickCol = View::StickyCol{true, y, pinH + pgap + std::min(0.f, proom - rh), ly - kSectionGap - rh, pinAt, pinH};
    }
    const size_t firstRight = v.hits.size();
    ry += right(ry);
    for (size_t k = firstRight; k < v.hits.size(); ++k) v.hits[k].sticky = true;
    return tail(std::max(ly - kSectionGap, ry));
}

// "Show me how" (ShowRrConfigSteps): once this game's Engine.ini steps show (the Apply that sets rr_config_fix=off is done), the
// page scrolls so their card sits under the pinned header, and the keyboard goes to the first step's Open folder. Once.
static void RevealRrConfigCard(const Rc& area) {
    if (v.rrcfgReveal.empty() || !IEq(v.rrcfgReveal, v.cur.dir) || v.pageReading || v.rrcfgCardY < 0) return;
    bool open = false;
    for (auto& h : v.hits) open = open || h.id == ID_G_RRCFG_OPEN;
    if (!open) return;  // not yet: the steps show once Myself is saved
    v.rrcfgReveal.clear();
    const float range = std::max(0.f, v.contentH[P_GAME] - v.H);
    v.target[P_GAME] = std::min(range, std::max(0.f, std::round(v.rrcfgCardY - (area.y + kPinH) - 8)));  // 8 under the header: the steps and the check fit
    if (!v.live || !MotionOn()) v.scroll[P_GAME] = v.target[P_GAME];
    v.paintScrolled = true;
    v.focus = ID_G_RRCFG_OPEN;
    v.focusVisible = true;
    UiaFocusChanged(ID_G_RRCFG_OPEN);
    Invalidate();
}

void PaintGame(Gfx& gx, const Rc& area) {
    const float x0 = ColX(area), cw = ColW(area.w);
    v.pageCx = x0 + cw / 2;
    // the compact header stays under the toolbar once the header's actions have scrolled away: the game's name, its status
    // and its actions never leave the page. What scrolls under it is clipped there, and so are its hits; the same for the
    // unsaved capsule's band at the bottom (nothing under either takes a click or shows a text field; Tab still reaches it
    // and scrolls it out from under them).
    // It pins once the game's name has scrolled under the toolbar (its baseline passes it), not only once the actions do: the
    // page never rests with Play and More in view and no name anywhere (Play always has its object in view). The header's
    // part still under it then scrolls on as art only (heroRemnant): the actions are the compact header's from that moment.
    // It comes in with the scroll, not at once: from the moment it pins until the header's actions have gone under it, the
    // compact header cross-fades in over the whole header, whose actions stay where they are (no button still in view is
    // taken away, and the part of them under the plate's edge still takes the click); the compact header's actions take the
    // clicks once the header's are under it (App Store's and Music's compact titles do the same).
    float btnY = 0, pinY = 0;
    const float heroB = Measure(gx, [&] { return Hero(gx, x0, cw, area.y, btnY, pinY); });
    const bool pinned = cv.scroll > pinY - area.y;
    const float settled = std::max(pinY - area.y + 1, btnY + (HasBanner(v.cur) ? 36.f : 34.f) - area.y - kPinH);  // the actions under the plate
    const float fade = pinned ? std::min(1.f, (cv.scroll - (pinY - area.y)) / (settled - (pinY - area.y))) : 0.f;
    // v.pinH: where the pinned header ends (window DIPs: the toolbar above it included); 0 while it is not pinned
    v.pinH = pinned ? area.y + kPinH : 0;
    if (pinned && fade < 1) {  // the cross-fade: the whole header under the compact one, its hits only below the plate
        // Below the plate the header as it is. Under the plate its art fades out as the plate fades in, and its words and
        // actions twice as fast (gone halfway, where the compact header's words begin): the two headers' words never show at
        // once, wherever the scroll stops (the fade follows the scroll, not the clock).
        const ContentView k0 = cv;
        cv.view = Rc{area.x, v.pinH, area.w, v.H - v.pinH};
        gx.PushClip(Rc{area.x, cv.scroll + v.pinH, area.w, v.H - v.pinH});
        Hero(gx, x0, cw, area.y, btnY, pinY);
        gx.PopClip();
        const size_t heroAt = v.hits.size();
        gx.PushClip(Rc{area.x, cv.scroll + area.y, area.w, kPinH});
        heroRemnant = true;
        Hero(gx, x0, cw, area.y, btnY, pinY);  // the art
        heroRemnant = false;
        if (fade < 0.5f) {
            gx.PushFade(1 - 2 * fade);
            heroWordsOnly = true;
            Hero(gx, x0, cw, area.y, btnY, pinY);
            heroWordsOnly = false;
            gx.PopFade();
        }
        gx.PopClip();
        v.hits.resize(heroAt);  // (the hits are the first pass's: one per control)
        cv = k0;
        const size_t plateAt = v.hits.size();
        pinFade = fade;
        PinnedHeader(gx, area, x0, cw, cv.scroll + area.y);
        pinFade = 1;
        v.hits.resize(plateAt);  // its actions are the header's until it has settled (one hit per control)
    } else if (pinned) PinnedHeader(gx, area, x0, cw, cv.scroll + area.y);
    else Hero(gx, x0, cw, area.y, btnY, pinY);
    const float top = pinned ? v.pinH : area.y;
    const ContentView keep = cv;
    if (pinned) {
        gx.PushClip(Rc{area.x, cv.scroll + v.pinH, area.w, v.H - v.pinH});
        if (fade >= 1 && cv.scroll + v.pinH < heroB + 48) {  // the header's art (and its shadow) below the compact header, until it has scrolled away
            heroRemnant = true;
            Hero(gx, x0, cw, area.y, btnY, pinY);
            heroRemnant = false;
        }
    }
    cv.view = Rc{area.x, top, area.w, v.H - top - GameBarH()};
    v.contentH[P_GAME] = Cards(gx, x0, cw, heroB + (HasBanner(v.cur) ? 30.f : 24.f), pinY - area.y);
    // a page that scrolls at all scrolls far enough for the compact header to pin: the name never rests half under the
    // toolbar (a page only a little taller than the window)
    if (v.contentH[P_GAME] > v.H + 0.5f) v.contentH[P_GAME] = std::max(v.contentH[P_GAME], pinY - area.y + 1 + v.H);
    cv = keep;
    if (!cv.dry) RevealRrConfigCard(area);
    if (pinned) gx.PopClip();
}

// The unsaved-changes capsule floats 18 DIPs above the window's bottom edge, 50 high: the band it sits in (68 DIPs) takes no
// clicks and shows no text field from the page under it.
float GameBarH() { return v.page == P_GAME && Dirty() ? 68.f : 0.f; }

// The capsule's coming and going, as a toast's (one material, one motion): it rises 8 DIPs as it fades in (kToastMs), and
// fades out where it was (kToastOutMs) with its words and buttons as they were, taking no clicks. Only for a change on the
// page it shows on: a page opened with unsaved changes (kept after a failed Apply) or another game's page has no motion to
// play. None with Animation effects off.
namespace {
struct CapsuleMotion { bool shown = false; ULONGLONG at = 0, page = 0; wstring dir; int n = 1; bool applyOn = true; };
CapsuleMotion capsule;
}  // namespace
bool GameMotion() {
    if (!capsule.at || v.page != P_GAME) return false;
    return (long long)NowMs() - (long long)capsule.at < (long long)(capsule.shown ? kToastMs : kToastOutMs) + 20;
}

// The unsaved-changes capsule: a floating capsule on the blurred page (Material): how many settings differ, the sentence
// when it fits, Discard and Apply (the page's prominent action while it shows). Centred on the reading column.
void PaintGameOverlay(Gfx& gx, const Rc& area) {
    const bool dirty = GameBarH() > 0;
    const bool samePage = IEq(capsule.dir, v.cur.dir) && capsule.page == v.pageChangedAt;
    if (dirty != capsule.shown || !samePage) {
        capsule.at = samePage && v.live && MotionOn() ? NowMs() : 0;
        capsule.shown = dirty; capsule.dir = v.cur.dir; capsule.page = v.pageChangedAt;
    }
    float a = 1;
    if (dirty) {
        capsule.n = std::max(1, DirtyCount());
        capsule.applyOn = !Locked();
        a = Entrance(capsule.at, kToastMs);  // (at 0: in place)
    } else {
        if (!capsule.at || !v.live || !MotionOn()) return;
        const float t = (float)((long long)NowMs() - (long long)capsule.at) / kToastOutMs;
        if (t >= 1) return;
        a = 1 - t * t;  // ease in: it leaves as the toast does
    }
    const int n = capsule.n;
    const wstring head = n == 1 ? L"1 unsaved change" : std::to_wstring(n) + L" unsaved changes";
    const wstring tail = n == 1 ? L"It takes effect the next time the game starts." : L"They take effect the next time the game starts.";
    TS hs = Ts(13.5f, 600, pal.text); hs.valign = 1;
    TS ts = Ts(13.5f, 400, pal.text2); ts.valign = 1;
    const float h = 50, btnH = 34;
    const float bw = ButtonW(gx, L"Discard", 0, btnH, B_GHOST) + 4 + std::max(84.f, ButtonW(gx, L"Apply", 0, btnH, B_GO));
    const float hw = gx.TextW(head, hs), tw = gx.TextW(tail, ts);
    const float maxW = ColW(area.w);
    const bool withTail = 20 + hw + 8 + tw + 18 + bw + 8 <= maxW;
    const float w = std::ceil(20 + hw + (withTail ? 8 + tw : 0.f) + 18 + bw + 8);
    const float cx = ColX(area) + ColW(area.w) / 2;
    const float rise = dirty ? std::round(8 * (1 - a) * gx.scale) / gx.scale : 0.f;
    const Rc r{std::round(cx - w / 2), v.H - 18 - h + rise, w, h};
    const size_t hitsBefore = v.hits.size();
    Material(gx, r, h / 2, 18, a);
    if (a < 1) gx.PushFade(a);
    gx.Text(head, Rc{r.x + 20, r.y, hw + 2, r.h}, hs);
    if (withTail) gx.Text(tail, Rc{r.x + 20 + hw + 8, r.y, tw + 2, r.h}, ts);
    AddText(Rc{r.x + 20, r.y, hw + (withTail ? 8 + tw : 0.f) + 2, r.h}, head + L". " + tail, ID_G_UNSAVED);  // before its buttons
    float bx = r.r() - 8;
    // Apply is an operation: not while the game runs or another operation works (one at a time; its tooltip says which)
    // in the accent's fill, as Update is: red is for repairs and failures (UI review r2 #12)
    bx -= Button(gx, ID_G_APPLY, bx, r.y + (h - btnH) / 2, L"Apply", B_GO, 0, capsule.applyOn, btnH, 84, true) + 4;
    if (dirty && Locked()) { v.tips[ID_G_APPLY] = BusyWhy(v.cur); Describe(ID_G_APPLY, BusyWhy(v.cur)); }
    // Discard only puts the page's values back (nothing is written): never waits, as leaving the page offers it any time
    Button(gx, ID_G_DISCARD, bx, r.y + (h - btnH) / 2, L"Discard", B_GHOST, 0, true, btnH, 0, true, L"", btnH / 2);
    if (a < 1) gx.PopFade();
    if (!dirty) v.hits.resize(hitsBefore);  // going: nothing on it takes a click or is read any more
}

}  // namespace wg::app::lui

// release 0.6.0: whether Ray Reconstruction is prepared for a game (rrprep.h): the window's own answer (launcher_ui.cpp)
namespace wg::app::win { bool RrReadyForGame(const wstring& dir, const wstring& exe, wstring* why); wstring RrBackgroundWords(const wstring& exe); }
namespace wg::app::lui {
bool RrReadyForGame(const wstring& dir, const wstring& exe, wstring* why) { return win::RrReadyForGame(dir, exe, why); }
wstring RrBackgroundWords(const wstring& exe) { return win::RrBackgroundWords(exe); }
}  // namespace wg::app::lui
