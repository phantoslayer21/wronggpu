// SPDX-License-Identifier: MIT
// WrongGPU launcher window: the Settings page and the first-run sheet (option E, "Quiet": build/launcher/ui-options/E-quiet,
// ?screen=settings and ?screen=firstrun). Settings is a short section index on the left (Model and runtime, Games,
// Launcher; the version and how the launcher runs under it) and one column of inset grouped lists. Every row has the same
// shape: a glyph tile, the name, a state word with its dot, one plain sentence, the controls on the right. Model and
// runtime (model data, HIP runtime), Games (game folders with a line per added folder, OptiScaler download, Play button),
// Launcher (appearance, release check with its address on a line of its own, the launcher's own install). The first run
// is a 560 DIP sheet centred over the blurred window: the mark and a welcome, this PC in one tinted list (a short summary,
// the graphics card, Model M and the games, with every problem; each check behind Show details), how the launcher runs
// (install for this user or portable, a two-row choice), and a full-width Continue that names what it will do. Its height
// eases to a new one instead of jumping. Written for this project.
#include "app/ui_shell.h"

#include "wg_core/model.h"
#include "wg_core/optiscaler.h"
#include "wg_core/rr.h"

namespace wg::app::lui {

namespace {
TS Ts(float size, int weight, Col c, FontFace f = F_TEXT) { TS s; s.size = size; s.weight = weight; s.color = c; s.face = f; return s; }
TS Wrap(float size, Col c, float lineH, int weight = 400) { TS s; s.size = size; s.weight = weight; s.color = c; s.wrap = true; s.lineH = lineH; return s; }

// name: for UI Automation when the words alone say too little. tint: the action that needs you (IDENTITY: red), on the
// accent's tint as the game page's Convert pill has it, in the row's own shape (radius 8)
struct Act { int id; wstring text; bool primary = false; bool enabled = true; wstring name; bool tint = false; };
// A row's state: a dot and a word in its colour, then the rest of the sentence ("Ready  Converted from ...").
// led: -1 no dot (the word in the text colour), 0 green (fine), 1 amber (needs a look), 2 grey (off, not there)
struct Status { int led = -1; wstring word, rest; };

// The mockup's settings row (.srow in a .group.icons): 11 DIPs above and below its words (54 at least), 16 inside the
// group's sides, a 28 DIP glyph tile, 20 between the tile and the words (they start 64 in, where the hairline starts),
// 20 between the words and the controls, 8 between buttons (30 high, radius 8).
// Two glyphs only this page uses: Appearance (a half-dark sun: light or dark) and Exact mode (a set square: precision), so
// no two rows share a glyph (Accent color keeps the palette, the HIP runtime the processor)
const wchar_t kGlyphAppearance = 0xE793;
const wchar_t kGlyphExact = 0xEB3C;
constexpr float kPadY = 11, kTile = 28, kTextX = kCellPad + kTile + 20, kBtnH = 30, kDescLine = 17;
// The state's dot sits in two en spaces at the start of the sentence (7 DIPs and a 6 DIP gap, the mockup's .state)
const wchar_t kDotRoom[] = L"\u2002\u2002";

// The section index (content y of each section's heading, the last frame's; ScrollBy uses it)
float secTop[3] = {0, 0, 0};
const wchar_t* const kSections[3] = {L"Model and runtime", L"Games", L"Launcher"};

// One row of a settings list: the glyph tile, the name, the state and sentence in one paragraph, and on the right
// buttons, a switch (switchId) or a segmented control (`seg`, ids from switchId). `border`: the hairline above it (not on
// a list's first row), from the words to the list's right edge. `textId`: the name and the sentence are one text for UI
// Automation (not focusable) with `tip` after them, which is also their hover tooltip (a file's or folder's whole path:
// the row says the place in words, LIVE-FINDINGS 15).
float SettingRow(Gfx& gx, float x, float y, float w, bool border, wchar_t icon, const wstring& title, const Status& st, const wstring& desc,
                 const std::vector<Act>& acts, int switchId = 0, int switchOn = -1,
                 const std::vector<wstring>& seg = {}, int segSel = 0, int textId = 0, const wstring& tip = L"", float ownW = 0) {
    float actW = ownW, ctrlH = ownW > 0 ? 28.f : 0.f;  // ownW: room for a control the caller draws (the accent swatches)
    // a conversion running: an activity indicator beside its button (it has no amount to show), 22 DIPs of the row
    auto working = [](const Act& a) {  // the one "Converting..."
        return (v.task == T_MODEL && (a.id == ID_S_RECONVERT || a.id == ID_S_MODELMORE)) || (v.task == T_RRDATA && (a.id == ID_S_RRCONVERT || a.id == ID_S_RRMORE));
    };
    for (auto& a : acts) { actW += ButtonW(gx, a.text, 0, kBtnH, B_PLAIN) + 8 + (working(a) ? 22.f : 0.f); ctrlH = kBtnH; }
    if (actW > 0) actW -= 8;
    if (switchId && seg.empty()) { actW = kSwitchW; ctrlH = kSwitchH; }
    if (!seg.empty()) { actW = SegmentedW(gx, seg); ctrlH = 32; }
    const float tx = x + kTextX, right = x + w - kCellPad;
    // buttons sit on the right; when that leaves the words too little room, they go on a line under the words
    const float side = right - tx - actW - 20;
    const bool below = !acts.empty() && side < 240;
    // beside controls the words keep a readable measure (about 90 characters) in a wide window, the controls at the row's
    // right end; a row without controls (the HIP runtime) has the whole row for its words, so they do not wrap early
    const float tw = actW == 0 ? right - tx : std::min(640.f, below ? right - tx : side);
    // the paragraph: the dot's room, the state word (semibold, its colour), then the rest and the sentence (text2)
    const bool hasSt = !st.word.empty();
    wstring lead = hasSt ? (st.led >= 0 ? wstring(kDotRoom) : wstring()) + st.word : wstring();
    wstring tail = st.rest;
    if (!desc.empty()) tail += (tail.empty() ? L"" : L" ") + desc;
    const wstring para = NoWidow(lead + (!lead.empty() && !tail.empty() ? L"\u00a0 " : L"") + tail);  // no word alone on the last line
    const TS ds = Wrap(12.5f, pal.text2, kDescLine);
    const float dH = para.empty() ? 0 : gx.TextH(para, tw, ds);
    const float textH = 20 + (dH > 0 ? 1 + dH : 0);
    const float body = textH + (below ? 10 + kBtnH : 0);
    const float inner = std::max({kTile + 2, body, below ? 0.f : ctrlH});
    const float h = std::max(kSRowH, kPadY + inner + kPadY);
    if (border) Hairline(gx, tx, y, x + w);
    const float ty = std::round(y + (h - inner) / 2 + (inner - body) / 2);
    // the glyph tile, level with the name
    const Rc ib{x + kCellPad, ty + 1, kTile, kTile};
    gx.FillRound(ib, 7, pal.fill);
    gx.Icon(icon, ib.cx(), ib.cy(), 14, pal.text2);
    gx.Text(title, Rc{tx, ty, tw, 20}, Ts(14, 600, pal.text));
    float my = ty + 20;
    if (dH > 0) {
        my += 1;
        const Rc pr{tx, my - 2, tw, dH + 2};  // DirectWrite sets the line 2 DIPs lower than the mockup's 1.38 line box
        if (hasSt) {
            const Col wc = st.led == 0 ? pal.greenText : st.led == 1 ? pal.amberText : st.led == 2 ? pal.text2 : pal.text;
            if (st.led >= 0) StatusDot(gx, tx + 3.5f, my + kDescLine / 2 - 1.5f, st.led == 0 ? pal.green : st.led == 1 ? pal.amber : pal.greyDot);
            gx.TextLead(para, lead.size(), wc, 600, pr, ds);
        } else gx.Text(para, pr, ds);
        my += dH;
    }
    // the row in words for UI Automation ("Model data: Ready. Converted from ..."), before its controls, once: with its
    // own id and the tooltip's whole path after the words (textId), else numbered; its sentence is the help text of its
    // switch or segments
    const wstring plain = (hasSt ? st.word : L"") + (hasSt && !tail.empty() ? L". " : L"") + tail;
    const wstring said = title + (plain.empty() ? L"" : L": " + plain);
    if (textId) {
        AddHit(textId, Rc{tx, ty, tw, my - ty}, R_TEXT, said + (tip.empty() ? L"" : L" " + tip), true, -1, L"", false);
        if (!tip.empty()) v.tips[textId] = tip;
    } else AddText(Rc{tx, ty, tw, textH}, said);
    const size_t before = v.hits.size();
    // the controls: centred on the row, or on their own line under the words
    const float cy = below ? my + 10 + kBtnH / 2 : y + h / 2;
    float bx = below ? tx + actW : right;
    if (switchId && seg.empty()) Switch(gx, switchId, Rc{bx - kSwitchW, std::round(cy - kSwitchH / 2), kSwitchW, kSwitchH}, switchOn == 1, true, title);
    if (!seg.empty()) Segmented(gx, switchId, bx, std::round(cy - 18), seg, segSel, true, title, 28, {}, true);
    for (size_t i = acts.size(); i-- > 0;) {
        bx -= Button(gx, acts[i].id, bx, std::round(cy - kBtnH / 2), acts[i].text, acts[i].primary ? B_PRIMARY : acts[i].tint ? B_TINTED : B_PLAIN, 0, acts[i].enabled,
                     kBtnH, 0, true, acts[i].name, acts[i].tint ? 8.f : -1.f) + 8;
        if (working(acts[i])) { BusyIndicator(gx, bx - 6, cy, 7, pal.text2); bx -= 22; }
    }
    for (size_t i = before; i < v.hits.size() && !plain.empty(); ++i) if (v.hits[i].role == R_TOGGLE || v.hits[i].role == R_RADIO) v.hits[i].help = plain;
    return h;
}

// A file or folder in words (LIVE-FINDINGS 15: no paths as body text, the whole path goes to the tooltip and UI
// Automation): its own name and the place it is in, from the drive (K:\Games\Itch: "Itch" in "K: > Games"), from the
// user folder for a path under it ("Downloads"); a deep place keeps its last two folders ("M: > ... > Ports").
// A drive alone is "Drive K:". `folder`: the last folder of the place only ("Downloads"), for a sentence.
struct PathWords { wstring name, in, folder; };
PathWords WordsFor(const wstring& path) {
    auto split = [](const wstring& p) {
        std::vector<wstring> parts;
        size_t at = 0;
        for (size_t i; (i = p.find_first_of(L"\\/", at)) != wstring::npos; at = i + 1) if (i > at) parts.push_back(p.substr(at, i - at));
        if (at < p.size()) parts.push_back(p.substr(at));
        return parts;
    };
    std::vector<wstring> parts = split(path);
    PathWords w;
    if (parts.empty()) { w.name = path; return w; }
    w.name = parts.back();
    parts.pop_back();
    if (parts.empty()) {
        if (w.name.size() == 2 && w.name[1] == L':') w.name = L"Drive " + w.name;
        return w;
    }
    w.folder = parts.back();
    if (w.folder.size() == 2 && w.folder[1] == L':') w.folder = L"drive " + w.folder;
    size_t from = 0;
    const std::vector<wstring> home = split(EnvVar(L"USERPROFILE"));
    bool inHome = !home.empty() && home.size() <= parts.size();
    for (size_t i = 0; inHome && i < home.size(); ++i) inHome = IEq(home[i], parts[i]);
    std::vector<wstring> shown;
    if (inHome) {
        from = home.size();
        if (from == parts.size()) { w.in = w.folder = L"your user folder"; return w; }
    } else {
        shown.push_back(parts[0]);  // the drive (or a share's server)
        from = 1;
    }
    if (parts.size() - from > 2) { shown.push_back(L"\u2026"); from = parts.size() - 2; }
    for (size_t i = from; i < parts.size(); ++i) shown.push_back(parts[i]);
    for (auto& s0 : shown) w.in += (w.in.empty() ? L"" : L" \u203a ") + s0;
    return w;
}

// A line under a row that belongs to it (an added game folder): the hairline from the words, a glyph in the words'
// column, then the folder in words (its name, the place it is in and how many games it holds, quiet) and its own Remove
// at the right end (the mockup's .folder: 40 high, the words 26 in from the row's words; Remove 12.5 semibold text2, 28
// high, the fill under the mouse). The whole path is the words' tooltip and in their UI Automation name (`textId`).
float SubRow(Gfx& gx, float x, float y, float w, wchar_t icon, const wstring& path, const wstring& note, int removeId, int textId) {
    const float h = 40, tx = x + kTextX;
    Hairline(gx, tx, y, x + w);
    gx.Icon(icon, tx + 7, y + h / 2, 14, pal.text2);
    float right = x + w - kCellPad;
    if (removeId) {
        TS rs = Ts(12.5f, 600, Hot(removeId) ? pal.text : pal.text2); rs.valign = 1;
        const float rw = std::ceil(gx.TextW(L"Remove", rs) + 20);
        const Rc rr{right - rw + 10, std::round(y + (h - 28) / 2), rw, 28};  // its words end on the row's content edge
        if (Hot(removeId)) gx.FillRound(rr, 7, Down(removeId) ? pal.fillHover : pal.fill);
        gx.Text(L"Remove", Rc{rr.x + 10, rr.y, rw - 18, rr.h}, rs);
        FocusRing(gx, rr, 7, removeId);
        // its name says a folder leaves the list ("Remove the folder "games" in M: > test-games from the list"), never that
        // games are removed; what stays is its help text
        const PathWords rw0 = WordsFor(path);
        AddHit(removeId, rr, R_BUTTON, L"Remove the folder \u201c" + rw0.name + L"\u201d" + (rw0.in.empty() ? L"" : L" in " + rw0.in) + L" from the list", true);
        Describe(removeId, L"WrongGPU stops looking for games in it. The folder and its games stay on this PC.");
        right = rr.x - 8;
    }
    // the name semibold in the text colour, then "K: > Games - 1 game" in text2 (a middle dot; the note never cut, a narrow row
    // shortens the place, then the name, in the middle)
    const PathWords pw = WordsFor(path);
    TS nms = Ts(13, 600, pal.text); nms.valign = 1;
    TS ns = Ts(13, 400, pal.text2); ns.valign = 1;
    const wstring note2 = (pw.in.empty() || note.empty() ? L"" : L" \u00b7 ") + note;
    const float room = std::max(40.f, right - (tx + 26));
    const float noteW = note2.empty() ? 0 : gx.TextW(note2, ns);
    const float nameFull = gx.TextW(pw.name, nms);
    const float nameW = std::min(nameFull, std::max(room * 0.6f, room - noteW - 10 - gx.TextW(pw.in, ns)));
    const wstring name = nameW < nameFull ? FitMiddle(gx, pw.name, nameW, nms) : pw.name;
    const float nw = std::min(nameW, gx.TextW(name, nms)) + 1;
    gx.Text(name, Rc{tx + 26, y, nw + 1, h}, nms);
    float qx = tx + 26 + nw + 10;
    const float qroom = std::max(0.f, right - qx - noteW);
    const wstring in = pw.in.empty() || qroom < 30 ? wstring() : gx.TextW(pw.in, ns) > qroom ? FitMiddle(gx, pw.in, qroom, ns) : pw.in;
    const wstring quiet = in + (in.empty() ? note : note2);
    const float qw = quiet.empty() ? 0 : std::min(right - qx, gx.TextW(quiet, ns) + 1);
    if (!quiet.empty()) gx.Text(quiet, Rc{qx, y, qw + 1, h}, ns);
    const Rc tr{tx + 26, y + 6, (quiet.empty() ? nw : qx + qw - (tx + 26)) + 1, h - 12};
    const wstring said = path + (note.empty() ? L"" : L", " + note);
    if (textId) {
        AddHit(textId, tr, R_TEXT, said, true, -1, L"", false);
        v.tips[textId] = path;
    } else AddText(tr, said);
    return h;
}

// The games found in an added folder ("1 game found", "No games found": the count never reads as the folder's name, a
// folder may be called "games"); nothing while it scans
wstring FolderNote(const wstring& folder) {
    if (v.scanning || folder.empty()) return L"";
    wstring f = folder;
    if (f.back() != L'\\') f += L'\\';
    int n = 0;
    for (auto& e : v.games) n += StartsWithI(e.dir + L"\\", f);
    return n == 0 ? L"No games found" : std::to_wstring(n) + (n == 1 ? L" game found" : L" games found");
}

// The OptiScaler version a download's or folder's name gives: "OptiScaler_0.9.0.7z" -> "0.9.0" (the first run of digits
// with two or three dot-separated parts); empty when the name has none.
wstring OptiNameVersion(const wstring& path) {
    const size_t slash = path.find_last_of(L"\\/");
    const wstring n = slash == wstring::npos ? path : path.substr(slash + 1);
    for (size_t i = 0; i < n.size(); ++i) {
        if (!iswdigit(n[i]) || (i > 0 && (iswdigit(n[i - 1]) || n[i - 1] == L'.'))) continue;
        size_t j = i, parts = 0;
        while (j < n.size() && iswdigit(n[j])) {
            size_t k = j;
            while (k < n.size() && iswdigit(n[k])) ++k;
            ++parts;
            j = k;
            if (parts == 3 || j + 1 >= n.size() || n[j] != L'.' || !iswdigit(n[j + 1])) break;
            ++j;
        }
        if (parts >= 2) return n.substr(i, j - i);
    }
    return L"";
}

// Settings > Launcher > Accent color (BE-wide notes.md): seven round swatches, 22 DIPs and 10 apart, the chosen one with a
// 2 DIP ring in the text colour 4 DIPs out; Windows is a four-colour swatch (it follows the Windows accent colour). One Tab
// stop (a radio group: the arrows choose), each swatch's hit 28 DIPs square; the keyboard's ring 7 DIPs out (the chosen
// ring is its own sign).
constexpr float kSwatch = 22, kSwatchGap = 10;
float AccentRow(Gfx& gx, float x, float y, float w, bool border) {
    const int sel = CurrentAccent();
    const float sw = A_COUNT * kSwatch + (A_COUNT - 1) * kSwatchGap;
    const float h = SettingRow(gx, x, y, w, border, ic::palette, L"Accent color", Status{},
                               sel == A_WINDOWS ? L"Follows your Windows accent color."
                                                : L"Play, switches and links. Green always means WrongGPU is working; red means something needs you.",
                               {}, 0, -1, {}, 0, 0, L"", sw);
    const float cy = std::round(y + h / 2);
    float sx = x + w - kCellPad - sw;
    const unsigned win = WindowsAccentRgb();
    for (int i = 0; i < A_COUNT; ++i, sx += kSwatch + kSwatchGap) {
        const int id = ID_S_ACCENT0 + i;
        const Rc r{sx, cy - kSwatch / 2, kSwatch, kSwatch};
        const float hl = Down(id) ? -0.08f : Hot(id) ? 0.1f : 0.f;
        if (i == A_WINDOWS) {  // the Windows logo's four colours, a quarter each
            static const unsigned q[4] = {0xf25022, 0x7fba00, 0x00a4ef, 0xffb900};
            gx.PushRoundClip(r, kSwatch / 2);
            for (int k = 0; k < 4; ++k) {
                const Col c = Hex(q[k]);
                gx.Fill(Rc{r.x + (k % 2) * kSwatch / 2, r.y + (k / 2) * kSwatch / 2, kSwatch / 2, kSwatch / 2}, hl >= 0 ? Mix(c, pal.white, hl) : Mix(c, pal.black, -hl));
            }
            gx.PopRoundClip();
        } else {
            const Col c = AccentSwatch(i, win);
            gx.Circle(r.cx(), r.cy(), kSwatch / 2, hl >= 0 ? Mix(c, pal.white, hl) : Mix(c, pal.black, -hl));
        }
        // its edge: dark, inset 0 0 0 1px rgba(255,255,255,.14); light, the swatch's own colour 30 % darker, so every swatch's
        // edge is 3:1 or more on the white cell (WCAG 1.4.11: green 3.7, cyan 4.6, pink 5.1, graphite 5.2)
        gx.Ring(r.cx(), r.cy(), kSwatch / 2 - 0.5f, pal.dark ? Alpha(pal.white, 0.14f) : Alpha(pal.black, 0.3f), 1);
        if (i == sel) gx.Ring(r.cx(), r.cy(), kSwatch / 2 + 4, pal.text, 2);
        if (Focused(id)) gx.Ring(r.cx(), r.cy(), kSwatch / 2 + 8, pal.focus, 2);
        AddHit(id, Rc{r.x - 3, r.y - 3, kSwatch + 6, kSwatch + 6}, R_RADIO, wstring(L"Accent color: ") + AccentName(i), true, i == sel ? 1 : 0);
        v.tips[id] = AccentName(i);
    }
    return h;
}

using RowFn = std::function<float(float, bool)>;
// An inset grouped list holding its rows (each draws itself at a y, its hairline when not first, and says how tall it is)
float Group(Gfx& gx, float x, float y, float w, const std::vector<RowFn>& rows) {
    return GroupList(gx, x, y, w, [&](float top) {
        float yy = top;
        for (size_t i = 0; i < rows.size(); ++i) yy += rows[i](yy, i > 0);
        return yy - top;
    });
}
}  // namespace

// A running game that uses the Model M data (`rr` false: a game with WrongGPU turned on in it) or the Ray Reconstruction data
// (a game with WrongGPU's RR files): it holds the data's file. From the Library's states (no process is looked at in a
// frame). A game with WrongGPU turned off (AMD FSR) does not wait for: a conversion that finds the file held says so in
// words (launcher_ui.cpp). Empty: none.
static wstring RunningUsing(bool rr) {
    for (auto& x : v.games) if (x.status == GameStatus::Running && (rr ? x.rrInstalled : x.hasRecord && !x.wgOff)) return x.name;
    return L"";
}

wstring PlaceWords(const wstring& path) {
    const PathWords w = WordsFor(path);
    return w.name + (w.in.empty() ? L"" : L" in " + w.in);
}

wstring HipRuntimeWords(const wstring& dll) {
    const size_t slash = dll.find_last_of(L"\\/");
    const wstring n = Lower(slash == wstring::npos ? dll : dll.substr(slash + 1));
    const wstring pre = L"amdhip64_";
    if (n.rfind(pre, 0) != 0 || n.size() <= pre.size() + 4 || n.substr(n.size() - 4) != L".dll") return L"";
    const wstring ver = n.substr(pre.size(), n.size() - 4 - pre.size());
    for (wchar_t c : ver) if (!iswdigit(c)) return L"";
    return L"HIP " + ver;
}

wstring ModelConvertedWords(bool again, bool chose) {
    const wstring game = v.modelSource.empty() ? wstring() : ModelSourceGame(v.modelSource);
    const wstring from = !game.empty() ? game + L"'s DLSS file" : chose ? wstring(L"the file you chose") : wstring(L"a DLSS file on this PC");
    return L"The Model M data is converted " + wstring(again ? L"again " : L"") + L"from " + from + L".";
}

float SettingsSectionScroll(int section) {
    if (section < 0 || section > 2) return 0;
    return std::max(0.f, secTop[section] - (kBarH + 18));  // its heading where the page's title starts
}

void PaintSettings(Gfx& gx, const Rc& area) {
    // the reading column every page shares (920 DIPs, centred); the index takes 180 of it and 40 to the lists, unless the
    // window is too narrow for both (then the lists take the column and the index is left out)
    const float x0 = ColX(area), cw = ColW(area.w);
    const bool index = cw >= 760;
    const float lx = index ? x0 + 220 : x0, lw = index ? cw - 220 : cw;
    float y = area.y + 18;
    TS h1 = Ts(34, 600, pal.text, F_DISPLAY); h1.spacing = -0.4f;
    PageTitle(gx, L"Settings", Rc{x0, y - 4, cw, 44}, h1);  // the mockup's 1.15 line box puts the glyphs 4 DIPs higher than DirectWrite's
    gx.Text(L"For the launcher. Each game has its own settings on its page.", Rc{x0, y + 43, cw, 20}, Ts(14, 400, pal.text2));
    AddText(Rc{x0, y + 43, std::min(cw, 420.f), 20}, L"For the launcher. Each game has its own settings on its page.");
    y += 43 + 20 + 26;
    const float colTop = y;
    const GlobalSettings& s = gl.global;

    // the section index: three links (the section in view on the fill), the version and how the launcher runs; it stays
    // in view while the lists scroll (the mockup's position: sticky)
    // read when the page opens and after an install (not per frame); a render leaves it false (its frame "settings-installed" sets it)
    const bool installed = v.launcherInstalled;
    const bool portable = gl.paths.portable || v.fixture;
    if (index) {
        const float iy = std::max(colTop, area.y + cv.scroll + 18);
        int on = 0;
        const float maxScroll = std::max(0.f, v.contentH[P_SETTINGS] - v.H);
        for (int i = 0; i < 3; ++i) if (secTop[i] > 0 && secTop[i] - cv.scroll <= kBarH + 18 + 60) on = i;
        if (maxScroll > 0 && cv.scroll >= maxScroll - 1) on = 2;
        for (int i = 0; i < 3; ++i) {
            const int id = ID_S_INDEX0 + i;
            const Rc r{x0, iy + 2 + i * 34, 180, 32};  // 2 below the lists' first heading, 34 apart (32 and a 2 DIP gap)
            const bool hot = Hot(id);
            if (Down(id)) gx.FillRound(r, 8, pal.fillHover);  // pressed: darker, as every control has it
            else if (i == on) gx.FillRound(r, 8, pal.fill);
            else if (hot) gx.FillRound(r, 8, Mix(pal.bg, pal.fill, 0.5f));
            TS ts = Ts(13.5f, 600, i == on || hot ? pal.text : pal.text2); ts.valign = 1;
            gx.Text(kSections[i], Rc{r.x + 12, r.y, r.w - 24, r.h}, ts);
            FocusRing(gx, r, 8, id);
            AddHit(id, r, R_LINK, kSections[i], true, i == on ? 1 : 0);  // state: the section in view (UI Automation: "Current section")
        }
        TS as = Ts(12, 400, pal.text2); as.lineH = 18;
        const float ay = iy + 2 + 3 * 34 - 2 + 20;
        gx.Text(L"WrongGPU " + wstring(Ctx().data->version), Rc{x0 + 12, ay, 168, 18}, as);
        gx.Text(installed ? L"Installed for this user" : portable ? L"Portable" : L"Not installed", Rc{x0 + 12, ay + 18, 168, 18}, as);
        AddText(Rc{x0 + 12, ay, 168, 36}, L"WrongGPU " + wstring(Ctx().data->version) + L", " + (installed ? L"installed for this user" : portable ? L"portable" : L"not installed"));
    }

    int sec = 0;
    auto section = [&](const std::vector<RowFn>& rows) {
        if (sec) y += kSectionGap;
        secTop[sec] = y;
        y += GroupHeading(gx, lx, y, lw, kSections[sec]) + kHeadGap;
        y += Group(gx, lx, y, lw, rows);
        ++sec;
    };

    // ---- model and runtime (the model's state, where it came from and the HIP runtime are read off the paint: at start-up,
    // after a conversion and after an operation; a frame never reads or hashes the model file)
    const bool ok = v.modelReady;
    // not ready: no model data yet, or a file this version cannot use (the core's reason is converter words: it goes to the
    // tooltip and UI Automation, the row says what to do)
    wstring why = v.fixture ? wstring() : v.modelWhy, src;
    const bool absent = why.empty() || IEq(why, L"no converted model on this PC");
    if (!why.empty()) why[0] = towupper(why[0]);
    if (ok) {
        const wstring game = v.fixture || v.modelSource.empty() ? wstring() : ModelSourceGame(v.modelSource);
        src = game.empty() ? L"Converted from a DLSS file already on this PC." : L"Converted from " + game + L"'s DLSS file.";
    }
    const wstring hip = v.fixture ? v.fixtureHip : v.hipLine;
    // the runtime an install recorded is a file's path: its version in words, the path in the tooltip and for UI Automation
    const bool hipPath = hip.find_first_of(L"\\/") != wstring::npos;
    const wstring hipWords = hipPath ? HipRuntimeWords(hip) : hip;
    const wstring modelPath = v.fixture ? L"C:\\Users\\Player\\AppData\\Local\\WrongGPU\\model\\model.wgcm" : ModelFile(Ctx());
    // Ready: nothing needs doing, so one quiet "Options..." (a menu: Choose a DLSS file..., Convert again). Not converted
    // yet: the two ways out side by side, a file or Convert (one more look through this PC's games, then the conversion:
    // the game page's Convert for Ray Reconstruction is the same action under the same name and in the same red tint: the
    // action that needs you); Convert again for a file this version cannot use. While it converts, one "Converting..." in
    // their place, whichever started it.
    const bool busy = v.busy;  // an operation may be copying the model file: it is not replaced under it
    // a running game that uses the data holds its file: its buttons wait (disabled, the reason in their tooltip and help
    // text), as the game page's Convert does, instead of a press that is refused (Model M data: every game with WrongGPU;
    // the RR data: every game with WrongGPU's RR files). A game the Library's states missed is still refused in words
    // (launcher_ui.cpp)
    const wstring modelRun = RunningUsing(false), rrRun = RunningUsing(true);
    const bool modelCan = !busy && modelRun.empty(), rrCan = !busy && rrRun.empty();
    // an operation at work (an install, an update, the other data's conversion, the launcher's own install): the buttons it
    // holds wait in words too, as the game page's and the Library's do ("Waits until WrongGPU finishes updating Quarry Nine.")
    const wstring busyWait = busy ? BusyWhy(GameEntry()) : wstring();
    const wstring modelWait = busy ? busyWait : modelRun.empty() ? wstring() : L"Waits until " + modelRun + L" closes. It uses the Model M data.";
    const wstring rrWait = busy ? busyWait : rrRun.empty() ? wstring() : L"Waits until " + rrRun + L" closes. It uses the Ray Reconstruction data.";
    // `own`: the task these buttons started is the one at work (their "Converting..." / "Installing..." says it; no reason
    // to give)
    auto waits = [](const std::vector<Act>& acts, const wstring& why, bool own) {
        if (why.empty() || own) return;
        for (auto& a : acts) if (!a.enabled) { v.tips[a.id] = why; Describe(a.id, why); }
    };
    // every button names its data (two "Convert again" on one page are told apart by Narrator and voice access)
    std::vector<Act> modelActs = ok ? std::vector<Act>{{ID_S_MODELMORE, L"Options\u2026", false, modelCan, L"Model M data options"}}
                                    : std::vector<Act>{{ID_S_MODELFILE, L"Choose a DLSS file\u2026", false, modelCan, L"Choose a DLSS file for the Model M data"},
                                                       {ID_S_RECONVERT, absent ? L"Convert" : L"Convert again", false, modelCan,
                                                        absent ? L"Convert the Model M data" : L"Convert the Model M data again", true}};
    // the search and the conversion run (a worker thread); "Converting..." takes the id of the button that started it, so the
    // keyboard focus stays on it (PaintAll keeps the focus on a control that went busy)
    if (v.task == T_MODEL) modelActs = {{ok ? ID_S_MODELMORE : v.focus == ID_S_MODELFILE ? ID_S_MODELFILE : ID_S_RECONVERT, L"Converting\u2026", false, false, L"", !ok}};
    const wstring modelTip = (ok || absent ? L"" : L"Why: " + why + L". ") + L"Model M data file: " + modelPath;
    // Ray Reconstruction data and Exact mode (rr-notes.md), right under Model data: in a build with the RR files, or while a
    // game has them. The same pattern and the same words as Model data: Ready with where it came from and its path, and
    // Options... (Choose a DLSS file..., Convert again), its path in the tooltip; not converted yet: the two ways side by
    // side (Choose a DLSS file..., Convert); a file this version cannot use: Choose a DLSS file... and Convert again. Never
    // downloaded or shipped. While a game with WrongGPU's RR files waits for it, the state words are amber as Model data's
    // (IDENTITY: amber, a warning in words) and Convert is the red tint pill's colour (red: the action that needs you, as
    // that game page's Convert); with no such game the data is optional: a grey dot and a plain Convert
    bool rrRows = v.rrBuild, rrWanted = false;
    for (auto& g0 : v.games) rrWanted = rrWanted || g0.rrInstalled;
    rrRows = rrRows || rrWanted;
    const bool rrOk = v.rrReady;
    wstring rrWhy = v.fixture ? wstring() : v.rrWhy;
    const bool rrAbsent = rrWhy.empty() || rrWhy.rfind(L"no converted", 0) == 0;
    if (!rrWhy.empty()) rrWhy[0] = towupper(rrWhy[0]);
    const wstring rrGame = rrOk && !v.rrSource.empty() ? ModelSourceGame(v.rrSource) : wstring();
    // where it came from in Model data's words ("Cyberpunk 2077's DLSS file": the row's name says which data)
    const wstring rrSrc = (rrGame.empty() ? wstring(L"Converted from a DLSS file already on this PC.")
                                          : L"Converted from " + rrGame + L"'s DLSS file.") + L" Every game with Ray Reconstruction uses it.";
    const wstring rrPath = v.fixture ? L"C:\\Users\\Player\\AppData\\Local\\WrongGPU\\model\\F\\model.wgcm" : RrDataFile();
    std::vector<Act> rrActs = rrOk ? std::vector<Act>{{ID_S_RRMORE, L"Options\u2026", false, rrCan, L"Ray Reconstruction data options"}}
                                   : std::vector<Act>{{ID_S_RRFILE, L"Choose a DLSS file\u2026", false, rrCan, L"Choose a DLSS file for the Ray Reconstruction data"},
                                                      {ID_S_RRCONVERT, rrAbsent ? L"Convert" : L"Convert again", false, rrCan,
                                                       rrAbsent ? L"Convert the Ray Reconstruction data" : L"Convert the Ray Reconstruction data again", rrWanted}};
    if (v.task == T_RRDATA) rrActs = {{rrOk ? ID_S_RRMORE : v.focus == ID_S_RRFILE ? ID_S_RRFILE : ID_S_RRCONVERT, L"Converting\u2026", false, false, L"", !rrOk && rrWanted}};
    const wstring rrTip = (rrOk || rrAbsent ? L"" : L"Why: " + rrWhy + L". ") + L"Ray Reconstruction data file: " + rrPath;
    std::vector<RowFn> modelRows{
        [&](float ry, bool b) {
            const float h = SettingRow(gx, lx, ry, lw, b, ic::cube, L"Model M data",
                                       ok ? Status{0, L"Ready", src}
                                          : absent ? Status{1, L"Not converted yet", L"Choose a DLSS file (version 310.5\u00a0or\u00a0newer), or Convert finds one in your games."}
                                                   : Status{1, L"Not ready", L"The Model M data on this PC can't be used. Convert it again."},
                                       L"", modelActs, 0, -1, {}, 0, ok || !absent ? ID_S_MODELTEXT : 0, ok || !absent ? modelTip : L"");
            waits(modelActs, modelWait, v.task == T_MODEL);
            return h;
        }};
    if (rrRows) {
        modelRows.push_back([&](float ry, bool b) {
            // not converted while no game waits for it: optional (a grey dot, as the HIP runtime's "Not found"), the sentence
            // says who needs it; data that can't be used stays amber (a file that is wrong, a warning in words)
            const int rrLed = rrWanted ? 1 : 2;
            const float h = SettingRow(gx, lx, ry, lw, b, ic::cube, L"Ray Reconstruction data",
                                       rrOk       ? Status{0, L"Ready", rrSrc}
                                       : rrAbsent ? Status{rrLed, L"Not converted yet", wstring(rrWanted ? L"" : L"Only games with Ray Reconstruction need it. ") +
                                                                                           L"Choose a game's DLSS Ray Reconstruction file, or Convert finds one."}
                                                  : Status{1, L"Not ready", L"The Ray Reconstruction data on this PC can't be used. Convert it again."},
                                       L"", rrActs, 0, -1, {}, 0, rrOk || !rrAbsent ? ID_S_RRTEXT : 0,
                                       rrOk || !rrAbsent ? rrTip : L"");
            waits(rrActs, rrWait, v.task == T_RRDATA);
            return h;
        });
        modelRows.push_back([&](float ry, bool b) {
            return SettingRow(gx, lx, ry, lw, b, kGlyphExact, L"Exact mode", Status{},
                              L"Adds Exact to each game's Ray Reconstruction quality choices. It's the most precise and the slowest.",
                              {}, ID_S_EXACT, s.exactMode ? 1 : 0);
        });
    }
    modelRows.push_back(
        [&](float ry, bool b) {
            return SettingRow(gx, lx, ry, lw, b, ic::processor, L"HIP runtime",
                              hip.empty() ? Status{2, L"Not found", L""} : Status{0, L"Found", hipWords.empty() ? L"" : hipWords + L"."},
                              hip.empty() ? L"Optional. Model M runs on Direct3D\u00a012 without it. Only Sparkle guard and the HIP choice in Runs\u00a0on need it."
                                          : L"Used by Sparkle guard and the HIP choice in Runs\u00a0on.",
                              {}, 0, -1, {}, 0, hipPath ? ID_S_HIPTEXT : 0, hipPath ? L"HIP runtime file: " + hip : L"");
        });
    section(modelRows);

    // ---- games
    // each added folder has its own Remove; with several, the whole list can go at once too (both ask first)
    std::vector<Act> folderActs{{ID_S_ADDFOLDER, L"Add folder\u2026"}};
    if (s.extraFolders.size() > 1) folderActs.insert(folderActs.begin(), {ID_S_CLEARFOLDERS, L"Clear folder list\u2026"});
    section({
        [&](float ry, bool b) {
            float h = SettingRow(gx, lx, ry, lw, b, ic::folder, L"Game folders", Status{}, L"Steam, Epic and GOG are found on their own. Add folders for anything else.",
                                 folderActs);
            for (size_t i = 0; i < s.extraFolders.size(); ++i) {
                const bool own = i < (size_t)kMaxFolderRows;
                h += SubRow(gx, lx, ry + h, lw, ic::folder, s.extraFolders[i], FolderNote(s.extraFolders[i]), own ? ID_S_FOLDERRM0 + (int)i : 0,
                            own ? ID_S_FOLDERTEXT0 + (int)i : 0);
            }
            return h;
        },
        [&](float ry, bool b) {
            const bool chosen = !s.optiPackage.empty();
            // the version the download's name gives ("OptiScaler_0.9.0.7z"), and the tested one only when it differs; the
            // download in words ("OptiScaler_0.9.0.7z in Downloads"), its whole path in the tooltip and for UI Automation
            const wstring ver = chosen ? OptiNameVersion(s.optiPackage) : wstring();
            const PathWords pw = chosen ? WordsFor(s.optiPackage) : PathWords{};
            const wstring what = pw.name + (pw.folder.empty() ? L"" : L" in " + pw.folder);
            // OptiScaler is its own project (GPL): installs use the WrongGPU edition of OptiScaler, which the launcher downloads
            // from that project's release page (optiedition.h); the player's own download only when that download doesn't work
            const wstring use = L"Only when the WrongGPU edition can't be downloaded: " +
                                (ver.empty() ? what : L"OptiScaler " + ver + L" from " + what) + L".";
            return SettingRow(gx, lx, ry, lw, b, ic::download, L"OptiScaler",
                              chosen ? Status{0, L"WrongGPU edition", use} : Status{0, L"WrongGPU edition", L""},
                              L"Installs use the WrongGPU edition of OptiScaler. OptiScaler is its own project (GPL); the launcher downloads it from "
                              L"github.com/phantoslayer21/OptiScaler. If that download doesn't work, you can choose your own OptiScaler download.",
                              {{ID_S_OPTIFILE, L"Choose file\u2026"}, {ID_S_OPTIDIR, L"Choose folder\u2026"}}, 0, -1, {}, 0,
                              chosen ? ID_S_OPTITEXT : 0, chosen ? L"OptiScaler download: " + s.optiPackage : L"");
        },
        [&](float ry, bool b) {
            return SettingRow(gx, lx, ry, lw, b, ic::playLine, L"Play button", Status{}, L"Start games from their page. They still work when started from Steam, Epic or GOG.",
                              {}, ID_S_PLAY, s.playButton ? 1 : 0);
        },
    });

    // ---- the launcher
    const int sel = s.appearance == L"light" ? 1 : s.appearance == L"dark" ? 2 : 0;
    // installed first: a portable copy on a PC where the launcher is installed for this user (Install for this user, or
    // installed before) says Installed, as the index line does. It is not that install: no uninstall here (it would remove
    // the other copy), the sentence says where to uninstall. Portable: its settings and logs next to it; the data the games
    // share (model data, the RR data, the games' list, compare.txt, hip-runtime.txt) stays in the user folder, where they read
    // it. Where its settings and logs are is said in words ("next to it, in WrongGPU-data", "in your user folder"); the whole
    // path is the row's tooltip and in its UI Automation name (LIVE-FINDINGS 15)
    Status inst = installed ? Status{-1, L"Installed for this user", L""} : portable ? Status{-1, L"Portable", L""} : Status{2, L"Not installed", L""};
    const wstring dataDir = v.fixture ? wstring(L"M:\\Games\\WrongGPU\\WrongGPU-data") : gl.paths.dataDir;
    wstring dataAt;
    {
        const wstring lad = EnvVar(L"LOCALAPPDATA");
        if (portable && (v.fixture || IEq(Parent(dataDir), Parent(gl.paths.exe)))) dataAt = L"next to it, in " + BaseName(dataDir);
        else if (!v.fixture && !lad.empty() && (IEq(dataDir, lad) || StartsWithI(dataDir, lad + L"\\"))) dataAt = L"in your user folder";
        else {
            const PathWords dw = WordsFor(dataDir);
            dataAt = L"in " + dw.name + (dw.in.empty() ? L"" : L" (" + dw.in + L")");
        }
    }
    // the sentences short, the place in plain words ("next to the app"); Windows 11 calls the uninstall list Installed apps
    const bool nextTo = portable && StartsWithI(dataAt, L"next to");
    const wstring kept = nextTo ? L"Settings and logs stay next to the app." : L"Settings and logs are " + dataAt + L".";
    wstring instDesc = installed ? portable ? L"In Start and Installed apps. This portable copy's " + wstring(nextTo ? L"settings and logs stay next to it."
                                                                                                                                          : L"settings and logs are " + dataAt + L".") +
                                                  L" To uninstall, use the installed WrongGPU or Installed apps."
                                            : L"In Start and in Installed apps. " + kept
                       : portable ? kept + L" The model data your games use stays in your user folder until you remove it here. Install it to add WrongGPU to Start and "
                                           L"Installed apps."
                                  : L"Runs from where it is. " + kept + L" Each game with WrongGPU gets its own entry in Installed apps.";
    if (v.fixture ? v.fixturePackaged : RunningPackaged()) {  // what it writes for games may be invisible to them: the one thing here that needs a look
        // in words a player knows (no "app package"): what happens and what to do first, then the usual sentence
        inst = Status{1, L"Opened from another app", L""};
        instDesc = L"Games may not see the files WrongGPU writes while it's opened this way. Open WrongGPU from File Explorer instead. " + instDesc;
    }
    // not while an operation works in a game (the launcher's install adopts the games; its uninstall can remove WrongGPU from
    // them): the reason in the tooltip and help text, as the data rows' (busyWait)
    std::vector<Act> instActs;
    // its question's look at the games first (LookingAtGames): the button says "Preparing...", as it says "Uninstalling..." after
    const bool looking = LookingAtGames();
    if (installed && !portable)
        instActs.push_back({ID_S_SELFUNINSTALL, v.task == T_SELFUNINSTALL ? L"Uninstalling\u2026" : looking ? L"Preparing\u2026" : L"Uninstall launcher\u2026", false, !v.busy});
    else if (!installed) instActs.push_back({ID_S_SELFINSTALL, v.task == T_SELFINSTALL ? L"Installing\u2026" : L"Install for this user", false, !v.busy});
    // a portable copy has no uninstall: the shared data it used goes by hand before its folder is deleted (RemoveSharedData;
    // its look at the games first, as the launcher's uninstall's); left of Install, away from the row's end
    const bool removing = v.busy && v.task == T_NONE && v.busyDir.empty() && v.busyVerb == L"Removing\u2026";
    if (portable && !installed)
        instActs.insert(instActs.begin(), Act{ID_S_REMOVEDATA, looking ? L"Preparing\u2026" : removing ? L"Removing\u2026" : L"Remove data\u2026", false, !v.busy,
                                                   L"Remove WrongGPU's data from this PC"});
    section({
        [&](float ry, bool b) {
            // Windows' own app mode (not the launcher's palette: set to Light, the launcher is light while Windows may be dark);
            // a render's palette is its theme
            const bool winDark = v.fixture ? pal.dark : ThemeIsDark(L"system");
            const wstring follows = L"Follows Windows' app mode (now " + wstring(winDark ? L"dark" : L"light") + L").";
            const float h = SettingRow(gx, lx, ry, lw, b, kGlyphAppearance, L"Appearance", Status{},
                                       sel == 0 ? follows : L"The launcher stays " + wstring(sel == 1 ? L"light" : L"dark") + L". System follows Windows' app mode.",
                                       {}, ID_S_THEME0, -1, {L"System", L"Light", L"Dark"}, sel);
            // each option's help text says what that option does (the row's sentence says what the chosen one does)
            Describe(ID_S_THEME0, follows);
            Describe(ID_S_THEME0 + 1, L"The launcher stays light.");
            Describe(ID_S_THEME0 + 2, L"The launcher stays dark.");
            return h;
        },
        [&](float ry, bool b) { return AccentRow(gx, lx, ry, lw, b); },
        [&](float ry, bool b) {
            float h = SettingRow(gx, lx, ry, lw, b, ic::sync, L"Check for new releases", Status{},
                                 s.updateCheck ? L"On: one request to the release address below when the launcher starts. Nothing else is sent."
                                               : L"Off: the launcher never uses the network.",
                                 {}, ID_S_UPDATE, s.updateCheck ? 1 : 0);
            if (s.updateCheck) {
                // the release address on a line of its own under the row: the field from the words, Save and Check now
                const float tx = lx + kTextX, right = lx + lw - kCellPad, ty = ry + h;
                Hairline(gx, tx, ty, lx + lw);
                const float bw = ButtonW(gx, L"Check now", 0, kBtnH, B_PLAIN) + ButtonW(gx, L"Save", 0, kBtnH, B_PLAIN) + 16;
                const float fy = ty + 11;
                FieldBox(gx, v.edUrl, ID_S_URL, Rc{tx, fy, right - tx - bw, kBtnH}, v.urlText, L"https://\u2026 (release page or version file)", ic::link,
                         L"Release address (https)");  // in the UI font, as the native field (not code)
                float bx = right;
                const float cw0 = ButtonW(gx, L"Check now", 0, kBtnH, B_PLAIN);  // "Checking..." in the same place while it waits
                bx -= Button(gx, ID_S_CHECKNOW, bx, fy, v.checkingUpdate ? L"Checking\u2026" : L"Check now", B_PLAIN, 0, !v.checkingUpdate, kBtnH, cw0, true) + 8;
                Button(gx, ID_S_SAVEURL, bx, fy, L"Save", B_PLAIN, 0, true, kBtnH, 0, true);
                h += 11 + kBtnH + 11;
            }
            return h;
        },
        [&](float ry, bool b) {
            const float h = SettingRow(gx, lx, ry, lw, b, ic::monitor, L"Launcher install", inst, instDesc, instActs, 0, -1, {}, 0,
                                       ID_S_DATATEXT, L"Settings and logs: " + dataDir);
            waits(instActs, busyWait, v.task == T_SELFINSTALL || v.task == T_SELFUNINSTALL || looking || removing);
            return h;
        },
    });

    // the foot: the build, and where the model data comes from (two centred lines under the lists)
    const wstring build = Ctx().data->buildId && *Ctx().data->buildId && wcscmp(Ctx().data->buildId, Ctx().data->version) ? wstring(L" (build ") + Ctx().data->buildId + L")" : L"";
    TS ns = Ts(12, 400, pal.text2); ns.align = 1;
    y += kSectionGap + 6;
    gx.Text(L"WrongGPU " + wstring(Ctx().data->version) + build, Rc{lx, y, lw, 19}, ns);
    gx.Text(L"No model data ships with WrongGPU. It's converted from DLSS files on this PC.", Rc{lx, y + 19, lw, 19}, ns);
    AddText(Rc{lx, y, lw, 38}, L"WrongGPU " + wstring(Ctx().data->version) + build + L". No model data ships with WrongGPU. It's converted from DLSS files on this PC.");
    v.pageCx = lx + lw / 2;  // a toast is centred under the lists, as the foot is
    v.contentH[P_SETTINGS] = y + 38 + 96;  // room under the foot for a toast ("Saved."): it never sits on the foot's lines
}

// ---- first run ---------------------------------------------------------------------------------------------------------------
namespace {
// A check line "Windows: Windows 11 Pro ..." as a label and a value; a line without a short label is all value.
void SplitCheck(const wstring& line, wstring& label, wstring& value) {
    const size_t c = line.find(L": ");
    if (c != wstring::npos && c <= 24) { label = line.substr(0, c); value = line.substr(c + 2); }
    else { label.clear(); value = line; }
    value = NoWidow(value);  // no short word alone on a wrapped value's last line ("need it.")
}
// The summary the sheet opens on: these three rows (each a placeholder until its check is done, so the list keeps its height
// while the checks arrive) and every problem; the rest (Windows, Direct3D 12, the HIP runtime, earlier installs, notes)
// behind Show details.
const wchar_t* const kBrief[3] = {L"Graphics card", L"Model M", L"Games"};
constexpr float kLinkFoot = 4;  // the checks list's room under Show details / Hide details (its focus ring clears the foot)
const wchar_t* const kBriefWait[3] = {L"Checking\u2026", L"Checking\u2026", L"Looking for your games\u2026"};
int BriefKey(const wstring& line) {
    for (int k = 0; k < 3; ++k) {
        const size_t n = wcslen(kBrief[k]);
        if (line.size() > n + 1 && line.compare(0, n, kBrief[k]) == 0 && line[n] == L':') return k;
    }
    return -1;
}
}  // namespace

void PaintFirstRun(Gfx& gx) {
    if (!v.first.open) return;
    // the entrance: the window blurs and dims while the sheet fades in and settles 10 DIPs
    const float e = Entrance(v.first.at, kFirstMs);
    Scrim(gx, e);
    const float w = std::min(560.f, v.W - 32);
    const float x = std::round((v.W - w) / 2), pad = w < 480 ? 20.f : 30.f, iw = w - 2 * pad, ix = x + pad;
    int passed = 0, problems = 0;
    for (auto& l : v.first.lines) { passed += l.first == 0; problems += l.first == 2; }
    const bool working = v.first.busy;
    // the rows shown: the summary (problems first, then the three rows, a grey dot and "Checking..." for one still running),
    // or after Show details every check as it came
    std::vector<std::pair<int, wstring>> brief;
    for (auto& l : v.first.lines) if (l.first == 2 && BriefKey(l.second) < 0) brief.push_back(l);
    for (int k = 0; k < 3; ++k) {
        bool have = false;
        for (auto& l : v.first.lines) if (BriefKey(l.second) == k) { brief.push_back(l); have = true; break; }
        if (!have && working) brief.push_back({3, wstring(kBrief[k]) + L": " + kBriefWait[k]});
    }
    const bool open = v.first.checksOpen && !working;
    const std::vector<std::pair<int, wstring>>& lines = open ? v.first.lines : brief;
    // Show details / Hide details on the list's last line: there whenever something is behind it, its room kept while the
    // checks run (the list is as tall then as once they are done)
    const bool link = !working && (open || v.first.lines.size() > brief.size());
    const bool linkRoom = link || working;
    const bool installing = v.task == T_FIRSTRUN;  // Continue is installing the launcher (a worker thread): the choice waits
    const float labW = iw < 440 ? 92.f : 104.f;
    const TS vs = Wrap(12.5f, pal.text, 17.5f);
    // the foot that stays put under the part that scrolls: Continue (full width, naming what it will do) and a hint
    // (an installed launcher can be uninstalled in Settings, not made portable; a portable one can be installed there)
    wstring hint = !v.first.installOffer ? L"" : v.first.choice == 1 ? L"You can uninstall it later in Settings." : L"You can install it later in Settings.";
    float choiceTop = -1;  // where "How it runs" starts in the sheet's content (its own coordinates), for the hint below
    const float footH = 16.f + 38.f + (hint.empty() ? 22.f : 9.f + 17.f + 30.f);  // the hint keeps the mockup's paragraph margin under it
    // a window too short for the whole sheet folds it a step at a time until the choice fits on the first screen (the
    // minimum window, 760 x 560, shows both ways to run, SHIP-BAR 3): 1 closes up its gaps, 2 leaves out the mark, 3 says
    // each way to run in one short line, 4 leaves their sentences out (each sentence stays the choice's tooltip and help
    // text). When even that is too tall (every check shown), the gaps close up and the part above the foot scrolls.
    int fold = 0;
    auto content = [&](float top) -> float {
        const float gap = fold ? 10.f : 16.f;
        float y = top + (fold ? 16.f : 23.f);
        // the mark, the welcome and one sentence, centred
        if (fold < 2) {
            AppMark(gx, Rc{std::round(x + (w - 36) / 2), y, 36, 36});
            y += 36 + (fold ? 6.f : 10.f);
        }
        TS ts = Ts(24, 600, pal.text, F_DISPLAY); ts.align = 1; ts.spacing = -0.2f;
        PageTitle(gx, L"Welcome to WrongGPU", Rc{ix, y - 2, iw, 32}, ts);
        y += 29 + 4;
        const wstring sub = v.first.installOffer ? L"A quick look at this PC, then choose how the launcher runs." : L"A quick look at this PC.";
        TS ss = Wrap(13.5f, pal.text2, 19); ss.align = 1;
        const float sh = gx.TextH(sub, iw, ss);
        gx.Text(sub, Rc{ix, y, iw, sh + 2}, ss);
        AddText(Rc{ix, y, iw, sh + 2}, sub);
        y += sh + gap;
        // This PC: what was checked, with a summary at the right end of the heading
        const wstring sum = working ? L"Checking\u2026" : problems ? std::to_wstring(problems) + (problems == 1 ? L" needs a look" : L" need a look")
                                                                    : std::to_wstring(passed) + (passed == 1 ? L" check passed" : L" checks passed");
        // amber means "needs a look" only: while it checks, the summary is text2 without a glyph
        TS hs = Ts(13, 600, pal.text);
        TS sumS = Ts(13, 600, working ? pal.text2 : problems ? pal.amberText : pal.greenText); sumS.align = 2;
        const float sw = gx.TextW(sum, sumS);
        gx.Text(L"This PC", Rc{ix + 4, y, iw - sw - 40, 18}, hs);
        gx.Text(sum, Rc{ix + iw - 4 - sw - 2, y, sw + 2, 18}, sumS);
        if (!working) gx.Icon(problems ? ic::warn : ic::check, ix + iw - 4 - sw - 11, y + 9.5f, 11, problems ? pal.amberText : pal.greenText);
        AddText(Rc{ix + 4, y, iw - 8, 18}, L"This PC: " + sum, 0, 2);  // a heading with its summary ("1 needs a look")
        y += 18 + 7;
        // the checks in one tinted list: a glyph (a check when done, a warning for a problem, an i for a note, a grey dot
        // while working), the label, the value
        const std::vector<std::pair<int, wstring>>& shown = lines;
        // the link's room: 2 above it, its 28, and 4 under it, so its keyboard focus ring (5 DIPs out at its outer edge)
        // stands 3 DIPs clear of the list's foot instead of on it
        float listH = 12 + (linkRoom ? 28.f + (shown.empty() ? 0 : 2 + kLinkFoot) : 0);
        for (auto& l : shown) {
            wstring lab, val;
            SplitCheck(l.second, lab, val);
            TS mt = vs; mt.weight = l.first == 2 ? 600 : 400;
            listH += std::max(17.5f, gx.TextH(val, iw - 50 - (lab.empty() ? 0 : labW + 10), mt)) + 4;
        }
        const Rc well{ix, y, iw, std::round(listH)};
        gx.FillRound(well, 10, pal.bg);
        float ly = well.y + 6 + 2;
        for (auto& l : shown) {
            wstring lab, val;
            SplitCheck(l.second, lab, val);
            const float vx = well.x + 36 + (lab.empty() ? 0 : labW + 10), vw = well.r() - 14 - vx;
            TS mt = vs; mt.weight = l.first == 2 ? 600 : 400;
            const float th = gx.TextH(val, vw, mt);
            const float gy = ly + 9.5f;
            if (l.first == 0) gx.Icon(ic::check, well.x + 20, gy, 11, pal.greenText);
            else if (l.first == 2) gx.Icon(ic::warn, well.x + 20, gy, 11, pal.amberText);
            else if (l.first == 1) gx.Icon(ic::info, well.x + 20, gy, 11, pal.text2);
            else BusyIndicator(gx, well.x + 20, gy, 6, pal.text2);  // still being checked: it moves (no amount to show)
            // a problem: its words semibold in the text colour (the glyph and the summary say amber); a note or
            // working: text2
            mt.color = l.first == 1 || l.first == 3 ? pal.text2 : pal.text;
            TS ls = vs; ls.color = pal.text2;  // the value's line height, so label and value share a baseline
            if (!lab.empty()) gx.Text(lab, Rc{well.x + 36, ly, labW, 19}, ls);
            gx.Text(val, Rc{vx, ly, vw, th + 2}, mt);
            // the check in words, with what it means ("Problem: Graphics card: ..."): the glyph says it to the eye
            AddText(Rc{well.x + 36, ly, well.r() - 14 - well.x - 36, th + 2},
                    (l.first == 2 ? L"Problem: " : l.first == 1 ? L"Note: " : L"") + (lab.empty() ? val : lab + L": " + val));
            ly += std::max(17.5f, th) + 4;
        }
        // the fold: a link on the list's last line, its chevron in the glyphs' column, its words in the labels'; 2 DIPs under
        // the lines, so its keyboard focus ring (4 DIPs out) clears the line above it, and kLinkFoot over the list's foot, so
        // the ring stands clear of the foot and of the left edge (3 DIPs in) rather than merging with them
        if (link)
            Button(gx, ID_FR_CHECKS, well.x + 14 - 6, shown.empty() ? ly - 5 : ly + 2, open ? L"Hide details" : L"Show details", B_LINK, open ? ic::up : ic::down, true, 28);
        y = well.b();
        bool modelProblem = false;
        for (auto& l : v.first.lines) modelProblem |= l.first == 2 && l.second.rfind(L"Model M", 0) == 0;
        if (modelProblem && !working) {  // the same two ways Settings offers without a model: a file, or one more look
            // off while Continue installs the launcher (it adopts the games: no second check runs beside it)
            const float bw = Button(gx, ID_FR_MODEL, ix, y + 12, L"Choose a DLSS file\u2026", B_SECONDARY, ic::cube, !installing, 32);
            Button(gx, ID_FR_SEARCH, ix + bw + 8, y + 12, L"Search your games", B_SECONDARY, ic::search, !installing, 32);
            y += 12 + 32;
        }
        // How it runs: install for this user, or portable (a radio choice in one tinted list; Continue carries it out).
        // While the checks run the rows are off: the marks and names in text3, no hover
        if (v.first.installOffer) {
            y += gap;
            choiceTop = y - top;
            gx.Text(L"How it runs", Rc{ix + 4, y, iw - 8, 18}, hs);
            AddText(Rc{ix + 4, y, iw - 8, 18}, L"How it runs", 0, 2);
            y += 18 + 7;
            struct Choice { int id; const wchar_t* t; const wchar_t* d; const wchar_t* brief; };
            const Choice ch[2] = {{ID_FR_INSTALL, L"Install for this user", L"Adds WrongGPU to Start and to Installed apps. If a game is in a protected folder, Windows asks for permission once.",
                                   L"Adds WrongGPU to Start and to Installed apps."},
                                  {ID_FR_PORTABLE, L"Run portable", L"Runs from this folder and keeps its settings here. The model data your games use stays in your user folder, "
                                   L"and each game with WrongGPU gets its own entry in Installed apps.",
                                   L"Keeps its settings here and the model data in your user folder."}};
            const TS cds = Wrap(12.5f, working ? pal.text3 : pal.text2, 17);
            float hs2[2], total = 0;
            wstring said[2];  // each way's sentence as shown (a folded sheet: short, or none)
            for (int i = 0; i < 2; ++i) {
                said[i] = fold >= 4 ? L"" : fold == 3 ? ch[i].brief : ch[i].d;
                hs2[i] = 11 + 19 + (said[i].empty() ? 0.f : gx.TextH(said[i], iw - 44 - 14, cds)) + 11;
                total += hs2[i];
            }
            const Rc grp{ix, y, iw, total};
            gx.FillRound(grp, 10, pal.bg);
            float cy = y;
            for (int i = 0; i < 2; ++i) {
                const Rc r{ix, cy, iw, hs2[i]};
                const bool on = v.first.choice == i + 1, hot = !working && !installing && Hot(ch[i].id);  // the choice stays shown (dimmed) while it works
                if (hot) gx.FillRound4(r, i == 0 ? 10.f : 0.f, i == 0 ? 10.f : 0.f, i == 1 ? 10.f : 0.f, i == 1 ? 10.f : 0.f,
                                       Down(ch[i].id) ? Mix(pal.bg, pal.fillHover, 0.8f) : Mix(pal.bg, pal.fill, 0.7f));
                if (i > 0) Hairline(gx, r.x + 44, r.y, r.r());
                // the radio: a ring in text2, chosen a thick ring in the accent (the mockup's 5.5 DIP border)
                const float rx = r.x + 14 + 9, ry = r.y + 11 + 10;
                if (on) { gx.Circle(rx, ry, 9, working ? Mix(pal.sheet, pal.tint, 0.45f) : pal.tint); gx.Circle(rx, ry, 3.5f, pal.sheet); }
                else gx.Ring(rx, ry, 8.25f, working ? pal.text3 : pal.text2, 1.5f);
                gx.Text(ch[i].t, Rc{r.x + 44, r.y + 11, iw - 58, 19}, Ts(13.5f, 600, working ? pal.text2 : pal.text));
                if (!said[i].empty()) gx.Text(said[i], Rc{r.x + 44, r.y + 11 + 19, iw - 58, hs2[i] - 30}, cds);
                FocusRing(gx, r.in(2), 8, ch[i].id);
                AddHit(ch[i].id, r, R_RADIO, ch[i].t, !working && !installing, on ? 1 : 0);
                Describe(ch[i].id, ch[i].d);  // what the choice means, whole, also when the sheet shows it short
                if (!cv.dry && said[i] != ch[i].d) v.tips[ch[i].id] = ch[i].d;  // not from a fold only measured
                cy += hs2[i];
            }
            y += total;
        }
        return y - top;
    };
    // the sheet stays 16 DIPs inside the window, and under the toolbar's band where it would reach the caption buttons
    // (they are drawn over it and take the press there: Minimize must not land on the sheet's corner or its scroll thumb)
    const float minTop = x + w > v.W - CaptionW() - 8 ? kBarH + 8 : 16.f;
    const float maxBody = v.H - minTop - 16 - footH;
    float ch = Measure(gx, [&] { return content(0); });
    if (ch > maxBody) {  // folded a step at a time, and only then the scroll
        while (fold < 4 && ch > maxBody) {
            ++fold;
            ch = Measure(gx, [&] { return content(0); });
        }
        if (ch > maxBody) {  // it scrolls anyway: everything shown, the gaps closed up
            fold = 1;
            ch = Measure(gx, [&] { return content(0); });
        }
    }
    // still too short: the sheet fills the window and the part above the foot scrolls (wheel, keyboard focus)
    const float bodyH = std::min(ch, maxBody);
    // a new height (the checks arriving, Show details, a problem's buttons) eases in over the entrance's time, centred, the
    // content clipped to the sheet meanwhile; at once offscreen, with reduce motion, and for a sheet just opened
    FirstRun& f = v.first;
    const float wantH = bodyH + footH;
    const ULONGLONG now = NowMs();
    auto shownH = [&] {
        if (!f.resizeAt) return f.toH;
        const float t = std::min(1.f, std::max(0.f, (float)((long long)now - (long long)f.resizeAt)) / kFirstMs);
        return f.fromH + (f.toH - f.fromH) * (1 - (1 - t) * (1 - t) * (1 - t));
    };
    if (!v.live || v.motionAt >= 0 || !MotionOn() || f.sizedAt != f.at || f.toH <= 0) { f.fromH = f.toH = wantH; f.resizeAt = 0; f.sizedAt = f.at; }
    else if (std::abs(wantH - f.toH) > 0.5f) { f.fromH = shownH(); f.toH = wantH; f.resizeAt = now; }
    float sheetH = shownH();
    if (f.resizeAt && now - f.resizeAt >= (ULONGLONG)kFirstMs) { f.resizeAt = 0; sheetH = f.toH; }
    const bool resizing = std::abs(sheetH - wantH) > 0.25f;
    const float shownBody = std::max(40.f, sheetH - footH);
    const float rise = std::round(10 * (1 - e));
    const float top = std::round(std::max(minTop, (v.H - sheetH) / 2)) + rise;
    v.first.maxScroll = std::max(0.f, ch - bodyH);
    v.first.scroll = std::min(std::max(0.f, v.first.scroll), v.first.maxScroll);
    const Rc r{x, top, w, std::round(shownBody + footH)};
    const Rc body{r.x + 1, r.y + 1, r.w - 2, std::round(shownBody) - 1};
    v.first.bodyTop = body.y; v.first.bodyBottom = body.b();
    if (e < 1) gx.PushFade(e);
    Sheet(gx, r, 16, 2.3f);
    const float s = std::round(v.first.scroll * gx.scale) / gx.scale;
    const ContentView keep = cv;
    const bool clip = v.first.maxScroll > 0 || resizing;
    if (clip) {
        cv = ContentView{body, s};
        gx.PushRoundClip(body, -3);
        gx.SetOffset(0, -s);
    }
    content(top);
    if (clip) {
        gx.SetOffset(0, 0);
        gx.PopRoundClip();
        cv = keep;
    }
    if (v.first.maxScroll > 0) {
        // the scroll position, a fade where more follows, and a hairline over the foot
        if (v.first.scroll < v.first.maxScroll - 1)
            gx.GradStops(Rc{body.x, body.b() - 40, body.w, 40}, 0, {{0.f, Alpha(pal.sheet, 0)}, {1.f, Alpha(pal.sheet, 0.95f)}}, 0, 0, 0, 1);
        const float th = std::max(36.f, body.h * body.h / ch), ty = body.y + 8 + (body.h - 16 - th) * (v.first.scroll / v.first.maxScroll);
        gx.FillRound(Rc{r.r() - 9, ty, 4, th}, 2, Alpha(pal.text2, 0.45f));
        BasePlate(gx, r, footH, 16);
    }
    // the foot: Continue saying what it will do, and when there is a choice, where to change it later
    const wstring go = working ? L"Checking\u2026" : installing ? (v.first.installOffer && v.first.choice == 1 ? L"Installing\u2026" : L"Setting up\u2026") : v.first.installOffer && v.first.choice == 1 ? L"Install and continue"
                     : v.first.installOffer && v.first.choice == 2 ? L"Continue portable" : L"Continue";
    const float by = r.b() - footH + 16;
    // the choice scrolled under the foot (a short window with the checks shown): the hint says it is chosen, and where
    if (!working && !resizing && v.first.installOffer && v.first.choice && choiceTop >= 0 && top + choiceTop + 18 - s > body.b() - 8)
        hint = v.first.choice == 1 ? L"Install for this user is chosen below. Scroll to change it." : L"Run portable is chosen below. Scroll to change it.";
    Button(gx, ID_FR_CONTINUE, ix, by, go, B_GO, 0, !working && !installing, 38, iw, false, L"", 10);  // the tint: carrying on, nothing needs you
    if (!hint.empty()) {
        TS hs = Ts(12.5f, 400, pal.text2); hs.align = 1;
        gx.Text(hint, Rc{ix, by + 38 + 9, iw, 17}, hs);
        AddText(Rc{ix, by + 38 + 9, iw, 17}, hint);
    }
    if (e < 1) gx.PopFade();
}

}  // namespace wg::app::lui
