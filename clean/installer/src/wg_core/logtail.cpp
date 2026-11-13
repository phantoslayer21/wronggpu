// SPDX-License-Identifier: MIT
// WrongGPU launcher core: "Last run" from wronggpu-fsr.log (logtail.h). Written for this project.
#include "logtail.h"

#include <algorithm>

#include "gamerules.h"
#include "optiscaler.h"

namespace wg {

static string TrimR(string s) { while (!s.empty() && (s.back() == '\r' || s.back() == '\n' || s.back() == ' ')) s.pop_back(); return s; }
// The text of a log line after its "[time pid] [tag] " prefix.
static string Body(const string& line) {
    size_t p = 0;
    for (int i = 0; i < 2 && p < line.size() && line[p] == '['; ++i) {
        const size_t e = line.find(']', p);
        if (e == string::npos) break;
        p = e + 1;
        while (p < line.size() && line[p] == ' ') ++p;
    }
    return line.substr(p);
}
// A log line's time of day in seconds: the front end's "[19:21:56.952 1000] ...", the RR plugin's "19:21:50.000 load: ...";
// -1 without one (r5 #2)
static int TimeOfDay(const string& line) {
    const size_t p = !line.empty() && line[0] == '[' ? 1 : 0;
    if (line.size() < p + 8 || line[p + 2] != ':' || line[p + 5] != ':') return -1;
    for (size_t i : {p, p + 1, p + 3, p + 4, p + 6, p + 7}) if (!isdigit((unsigned char)line[i])) return -1;
    const int h = (line[p] - '0') * 10 + line[p + 1] - '0', m = (line[p + 3] - '0') * 10 + line[p + 4] - '0', s = (line[p + 6] - '0') * 10 + line[p + 7] - '0';
    return h < 24 && m < 60 && s < 60 ? h * 3600 + m * 60 + s : -1;
}
static void Clocked(LastRun::Clock& c, const string& line) {
    const int t = TimeOfDay(line);
    if (t < 0) return;
    if (c.start < 0) c.start = t;
    c.end = t;
}
static string WordAfter(const string& s, const string& key) {
    size_t p = s.find(key);
    if (p == string::npos) return "";
    p += key.size();
    size_t e = p;
    while (e < s.size() && s[e] != ' ' && s[e] != ',' && s[e] != ';' && s[e] != ')') ++e;
    return s.substr(p, e - p);
}

// "ERROR" as a word of its own ("ERROR: ...", "upscaler ERROR: ...", "ERROR C1 ...", "WG-ERROR"), not inside a code
// name such as WG_ERROR_NOT_READY that an info line quotes.
static bool ErrorWord(const string& b) {
    for (size_t p = b.find("ERROR"); p != string::npos; p = b.find("ERROR", p + 1)) {
        const char before = p ? b[p - 1] : ' ', after = p + 5 < b.size() ? b[p + 5] : ' ';
        const bool lb = !(isalnum((unsigned char)before) || before == '_'), rb = !(isalnum((unsigned char)after) || after == '_');
        if (lb && rb) return true;
    }
    return false;
}

bool IsProblemLine(const string& b) {
    // information, never a warning: the settings line, hooks installed, upscaler/runtime info (they name WG_ERROR_* codes,
    // "fallback" and "AMD's FSR" while describing what would happen)
    if (b.rfind("settings:", 0) == 0 || b.find("hooks: installed") != string::npos || b.find(" info:") != string::npos || b.rfind("info:", 0) == 0 ||
        b.find("network runtime ready") != string::npos)
        return false;
    if (ErrorWord(b)) return true;                                                     // errors and WG-ERROR support incidents
    if (b.find(" refused (WG_ERROR_") != string::npos) return true;                     // a refused split (the frame went to AMD's FSR)
    if (b.find("goes to AMD FSR") != string::npos || b.find("stays on AMD") != string::npos || b.find("falls back to AMD") != string::npos ||
        b.find("fell back to AMD") != string::npos)
        return true;                                                                     // fallback reasons
    const bool fallback = (b.find("fallback") != string::npos && b.find("no fallback") == string::npos) || b.find("falls back") != string::npos;
    return fallback && (b.find("warning") != string::npos || b.find("->") != string::npos);
}

// A dispatch line of the direct route's FSR 3.0 hook (clean/fsr fsr30_hook.cpp) that says WrongGPU's network upscaled the
// frame: "-> FFX_OK (" then, after an optional "AMD for the frame-generation resources, then ", the upscaler's name (the
// runtime's "WrongGPU network", or "the WrongGPU upscaler" for one without a name). AMD's output kept, AMD alone, and the
// stand-in are not.
bool DirectOk(const string& b) {
    static const char kFfx[] = " -> FFX_OK (", kAmdFirst[] = "AMD for the frame-generation resources, then ";
    const size_t p = b.find(kFfx);
    if (p == string::npos || b.find("stand-in") != string::npos) return false;
    string rest = b.substr(p + sizeof kFfx - 1);
    if (rest.rfind(kAmdFirst, 0) == 0) rest = rest.substr(sizeof kAmdFirst - 1);
    return rest.rfind("WrongGPU network", 0) == 0 || rest.rfind("the WrongGPU upscaler", 0) == 0;
}

bool RoutineNote(const wstring& n) {
    return n.find(L"(WG_ERROR_NOT_READY)") != wstring::npos || n.find(L"until the runtime is ready") != wstring::npos;
}
void SplitRoutineNotes(LastRun& r) {
    if (!r.ready) return;  // Model M never started: its start-up refusals say why, they stay
    std::vector<wstring> keep;
    for (auto& n : r.notes) {
        const bool late = std::find(r.late.begin(), r.late.end(), n) != r.late.end();  // refused once it was ready: a real refusal
        (RoutineNote(n) && !late ? r.routine : keep).push_back(n);
    }
    r.notes = keep;
}

LastRun ParseLastRun(const string& text) {
    LastRun r;
    // the last session
    size_t start = 0;
    for (size_t p = text.find("==== session start"); p != string::npos; p = text.find("==== session start", p + 1)) start = p;
    if (text.find("==== session start") == string::npos && text.empty()) return r;
    const string s = text.substr(start);
    std::vector<wstring> notes;
    bool foreign = false, noDepth = false, readyYet = false;
    // the frame being dispatched (its "WrongGPU frame:" line up to the next one): logged once the runtime was ready, and nothing
    // after it said another upscaler took it (a stand-in note, a refusal, AMD FSR). Upscaled only on the front end's own word
    // that WrongGPU's dispatch of it succeeded ("-> OK (WrongGPU)"), never on an announcement before the dispatch (r3 #2)
    bool frameOpen = false, frameReady = false, frameElsewhere = false, prevStandIn = false;
    auto note = [&](const string& n) {
        wstring w = FromUtf8(n.size() > 200 ? n.substr(0, 197) + "..." : n);
        if (readyYet && std::find(r.late.begin(), r.late.end(), w) == r.late.end()) r.late.push_back(w);  // after the start (SplitRoutineNotes)
        if (std::find(notes.begin(), notes.end(), w) == notes.end()) notes.push_back(w);
    };
    for (size_t a = 0; a < s.size();) {
        size_t e = s.find('\n', a);
        if (e == string::npos) e = s.size();
        const string line = TrimR(s.substr(a, e - a));
        a = e + 1;
        if (line.empty()) continue;
        Clocked(r.clock, line);  // the session's first and last time (r5 #2)
        if (line.find("==== session start:") != string::npos) {
            r.found = true;
            const size_t c = line.find("session start:") + 14, pr = line.find(" (process", c);
            string exe = line.substr(c, pr == string::npos ? string::npos : pr - c);
            while (!exe.empty() && exe[0] == ' ') exe.erase(0, 1);
            r.process = FromUtf8(exe);
            if (pr != string::npos) r.clock.pid = strtoul(line.c_str() + pr + 9, nullptr, 10);
            continue;
        }
        const string b = Body(line);
        const bool standIn = b.find("stand-in") != string::npos;
        if (frameOpen && (standIn || b.find("forwarded to AMD") != string::npos || IsProblemLine(b))) frameElsewhere = true;
        if (b.find("WrongGPU frame:") != string::npos) { frameOpen = true; frameReady = readyYet; frameElsewhere = false; }
        // the evidence that Model M upscaled a frame (r3 #2): the dispatch of a frame logged once the runtime was ready ended
        // "-> OK (WrongGPU)" with nothing sending it elsewhere first; "WrongGPU upscales again after N frame(s)" (a frame it took
        // after a run of frames it could not; not right after a stand-in note); the runtime's timing line counting network frames.
        // The late start's "it upscales from this frame on" is logged before that frame's dispatch: not enough on its own
        static const char kOk[] = "-> OK (WrongGPU)";
        if (frameOpen && frameReady && !frameElsewhere && b.size() >= sizeof kOk - 1 && b.compare(b.size() - (sizeof kOk - 1), sizeof kOk - 1, kOk) == 0)
            r.upscaled = true;
        // ...the direct route's FSR 3.0 hook says it its way (r4 #3): "-> FFX_OK (WrongGPU network)", "-> FFX_OK (AMD for the
        // frame-generation resources, then WrongGPU network)", "-> FFX_OK (WrongGPU network; AMD's ... skipped: ...)"; never AMD's
        // output kept ("FFX_OK (AMD's output kept ...)") or the stand-in ("FFX_OK (WrongGPU stand-in)")
        if (frameOpen && frameReady && !frameElsewhere && DirectOk(b)) r.upscaled = true;
        if (readyYet && !prevStandIn && b.find("WrongGPU upscales again after") != string::npos) r.upscaled = true;
        if (readyYet && b.find("timing, last ") != string::npos && b.find("network GPU") != string::npos) {
            const size_t f = b.find(" frames)");
            size_t d = f;
            while (d != string::npos && d > 0 && isdigit((unsigned char)b[d - 1])) --d;
            if (f != string::npos && d > 0 && d < f && b[d - 1] == '(' && atoi(b.c_str() + d) > 0) r.upscaled = true;
        }
        prevStandIn = standIn;
        if (b.find("settings: upscaler=") != string::npos) { r.found = true; r.upscaler = FromUtf8(WordAfter(b, "upscaler=")); continue; }
        // the game asked for an upscaler: a create, a frame, a frame refused or sent to AMD FSR (a start that failed is not "not used")
        // (the direct route's FSR 3.0 hook logs its creates as ffxFsr3UpscalerContextCreate / ffxFsr3ContextCreate: r4 #3)
        if (b.find("ffxCreateContext") != string::npos || b.find("ffxFsr3UpscalerContextCreate") != string::npos || b.find("ffxFsr3ContextCreate") != string::npos ||
            b.find("WrongGPU frame:") != string::npos || b.find(" refused (WG_ERROR_") != string::npos ||
            b.find("goes to AMD FSR") != string::npos || b.find("ffxDispatch") != string::npos ||
            (IsProblemLine(b) && (b.find("create") != string::npos || b.find("context") != string::npos || b.find("frame") != string::npos)))
            r.created = true;
        if (b.find("network runtime ready: executor ") != string::npos) { r.ready = r.created = readyYet = true; r.executor = FromUtf8(WordAfter(b, "executor ")); continue; }
        if (b.rfind("executor dx12:", 0) == 0 && r.executor.empty()) { r.executor = L"dx12"; continue; }
        if (b.find("OptiScaler plugin:") != string::npos && b.find("selected in the host's upscaler list") != string::npos) { r.optiPlugin = true; continue; }
        // OptiScaler's custom upscaler slot (WrongGPU's edition): its upscaler list names WrongGPU, and the host chose it by that
        // name: the overlay shows WrongGPU, not AMD FSR (r3 #3; today's real runs log these lines, no plugin line)
        if (b.find("hosted in OptiScaler's custom upscaler folder") != string::npos || b.find(" the host chose WrongGPU ") != string::npos) r.optiPlugin = true;
        if (const size_t sp = b.find("[support] WG-"); sp != string::npos) { r.support = FromUtf8(WordAfter(b, "[support] ")); continue; }
        if (b.rfind("own table:", 0) == 0 && b.find("does not take over") != string::npos) {
            note("command lists record into a foreign table (another tool's hooks): " + b.substr(11));
            if (!foreign) {
                foreign = true;
                r.warnings.push_back({L"foreign-vtable", L"Another tool's hooks (ReShade, Special K, an overlay) give the game's command lists their own tables, "
                                                         L"which WrongGPU does not take over: those frames fell back to AMD FSR."});
            }
            continue;
        }
        if (b.find("WrongGPU frame:") != string::npos) {
            const string rs = WordAfter(b, " R="), os = WordAfter(b, " O=");
            if (!rs.empty() && rs.find('x') != string::npos) r.renderSize = FromUtf8(rs);
            if (!os.empty() && os.find('x') != string::npos) r.outputSize = FromUtf8(os);
        }
        if (b.find("WrongGPU frame:") != string::npos && b.find("notes:") != string::npos && b.find("no depth") != string::npos && !noDepth) {
            noDepth = true;
            r.warnings.push_back({L"no-depth", L"The game sends the upscaler no depth buffer: Model M runs without depth and may look worse."});
        }
        if (IsProblemLine(b)) note(b);
    }
    r.notes = notes;
    SplitRoutineNotes(r);  // before the last-three cut: a routine start-up line never pushes out a real problem
    if (r.notes.size() > 3) r.notes.erase(r.notes.begin(), r.notes.end() - 3);
    return r;
}

wstring LastRun::Summary() const {
    if (!found) return L"Not run yet";
    wstring s;
    if (upscaler == L"amd") s = L"AMD FSR (WrongGPU off)";
    else if (ready) s = L"Model M (" + (executor.empty() ? wstring(L"?") : executor) + L")";
    // the game asked for Ray Reconstruction instead (r7 #1: it needs the game's DLSS on already); with RR frames that is rrUpscaled below
    else if (upscaler == L"network" && !created && rrUsed) s = L"Ray Reconstruction: no successful frame";
    else if (upscaler == L"network" && !created) s = L"WrongGPU not used (the game asked for no upscaler)";
    else if (upscaler == L"network") s = L"Model M did not start";
    else s = L"started" + (upscaler.empty() ? wstring() : L" (" + upscaler + L")");
    if (rrGuardOff && upscaler.empty()) s = L"Ray Reconstruction: WrongGPU RR stayed off";  // 0.6.1: its guard (Last run says why)
    else if ((rrOnly && upscaler.empty()) || rrUpscaled)  // a Ray Reconstruction session alone (r4 #1), or RR upscaling where Model M didn't (r6 #1)
        s = rrFrames <= 0 ? wstring(L"Ray Reconstruction: no successful frame")  // an attempt without one (r6 #2)
                          : L"Ray Reconstruction" + (rrMs.empty() ? wstring() : L" (" + rrMs + L" ms a frame)");
    if (!support.empty()) s += L" \u00b7 " + support;
    else if (!notes.empty()) s += L" \u00b7 " + std::to_wstring(notes.size()) + L" warning(s)";
    if (!when.empty()) s += L" \u00b7 " + when;
    return s;
}

void ParseAsiLog(const string& text, LastRun& r) {
    size_t start = 0;
    for (size_t p = text.find("WrongGPU.asi loaded"); p != string::npos; p = text.find("WrongGPU.asi loaded", p + 1)) start = p;
    const string s = text.substr(start);
    for (size_t a = 0; a < s.size();) {
        size_t e = s.find('\n', a);
        if (e == string::npos) e = s.size();
        const string b = Body(TrimR(s.substr(a, e - a)));
        a = e + 1;
        if (b.rfind("not hooked:", 0) == 0) {
            r.plugin = L"not-hooked";
            r.warnings.push_back({L"plugin-not-hooked", L"WrongGPU's OptiScaler plugin did not hook this OptiScaler build (" + FromUtf8(TrimR(b.substr(11)).substr(0, 160)) +
                                                        L"): WrongGPU stays in OptiScaler's FSR sub-list instead of its own entry."});
        } else if (b.rfind("hooked:", 0) == 0) {
            r.plugin = L"hooked";
        }
    }
}
void ParseRrLog(const string& text, LastRun& r) {
    size_t start = string::npos;
    for (size_t p = text.find("load: process "); p != string::npos; p = text.find("load: process ", p + 1)) {
        const size_t eol = text.find('\n', p);
        if (text.substr(p, eol == string::npos ? string::npos : eol - p).find(" attached") != string::npos) start = text.rfind('\n', p) == string::npos ? 0 : text.rfind('\n', p) + 1;
    }
    if (start == string::npos) {
        // a long session: its "attached" line has scrolled out of the log's tail (ReadTail keeps 256 KB) while its instance
        // and timing lines are still there; with no other load line in the tail, the whole tail is that session
        if (text.find("load: process ") != string::npos ||
            (text.find("runtime: create instance") == string::npos && text.find(" ms per frame") == string::npos &&
             std::none_of(std::begin(kRrGuardLines), std::end(kRrGuardLines), [&](const char* g) { return text.find(g) != string::npos; })))
            return;
        start = 0;
    }
    r.rrFound = true;
    const string s = text.substr(start);
    auto mode = [](const string& w) -> wstring { return w == "exact" || w == "accurate" || w == "fast" ? FromUtf8(w) : wstring(); };
    long long timed = 0, evaluated = 0, finished = 0;
    wstring rrRender, rrOutput;
    for (size_t a = 0; a < s.size();) {
        size_t e = s.find('\n', a);
        if (e == string::npos) e = s.size();
        const string b = TrimR(s.substr(a, e - a));
        a = e + 1;
        Clocked(r.rrClock, b);  // the session's first and last time, its process id (r5 #2)
        if (const size_t lp = b.find("load: process "); lp != string::npos && b.find(" attached", lp) != string::npos && r.rrClock.pid == 0)
            r.rrClock.pid = strtoul(b.c_str() + lp + 14, nullptr, 10);
        if (b.find("runtime: create instance") != string::npos) r.rrUsed = true;
        // an attempt that failed (r6 #2): the game asked for RR all the same. The runtime's reason, for the page ("evaluate failed
        // before any pass: frame kernels still being prepared (...)", "create: failed (D3D12 objects)"); the plugin's fallback
        auto why = [](string w) {
            while (!w.empty() && (w.back() == '.' || w.back() == ' ')) w.pop_back();
            return FromUtf8(w.size() > 160 ? w.substr(0, 157) + "..." : w);
        };
        if (const size_t cf = b.find("runtime: create: failed ("); cf != string::npos) {
            r.rrUsed = true;
            string w = b.substr(cf + 25);
            if (!w.empty() && w.back() == ')') w.pop_back();
            r.rrNote = L"the instance couldn't be made (" + why(w) + L")";
        }
        if (const size_t ef = b.find(" evaluate failed before any pass: "); ef != string::npos) r.rrNote = why(b.substr(ef + 34));
        // the plugin's direct route (games that create RR themselves) refusing the game's RR feature before any instance (r7 #3: its
        // runtime or model missing, the game's device refused, inputs it can't take): the game asked for RR, and it didn't run. Not
        // where the game-interface route already runs RR in this game (that one's own lines count), and the plugin merely loading
        // ("RR provider registered") is no attempt
        static const string kRefused = "RR feature refused (the game keeps its own path): ";  // (after the route's own prefix)
        if (const size_t nr = b.find(kRefused); nr != string::npos && b.find("the game-interface route holds", nr) == string::npos) {
            r.rrUsed = true;
            r.rrNote = L"the game's request was refused (" + why(b.substr(nr + kRefused.size())) + L")";
        }
        if (b.find("colour passed through, success reported") != string::npos) r.rrPassThrough = true;
        // 0.6.1: the plugin's coexistence guard (another RR plugin, a second copy of WrongGPU RR, hooks replaced): its line from
        // the sentence on (the last one; its whole text up to 400 characters: the lines run to about 250)
        for (const char* g : kRrGuardLines)
            if (const size_t gp = b.find(g); gp != string::npos) {
                const string line = b.substr(gp);
                r.rrGuard = FromUtf8(line.size() > 400 ? line.substr(0, 397) + "..." : line);
                break;
            }
        // the mode: an Exact the plugin refused runs Accurate; a switch "rr_mode accurate -> fast"; "rr_mode=fast", "rr_mode fast"
        if (b.find("using accurate") != string::npos || b.find("accurate runs") != string::npos) r.rrMode = L"accurate";
        else if (const size_t arrow = b.find(" -> "); arrow != string::npos && b.find("rr_mode ") != string::npos && b.find("rr_mode ") < arrow) {
            const wstring m = mode(WordAfter(b.substr(arrow), " -> "));
            if (!m.empty()) r.rrMode = m;
        } else if (b.find("rr_mode=") != string::npos) {
            const wstring m = mode(WordAfter(b, "rr_mode="));
            if (!m.empty()) r.rrMode = m;
        } else if (b.find("rr_mode ") != string::npos) {
            const wstring m = mode(WordAfter(b, "rr_mode "));
            if (!m.empty()) r.rrMode = m;
        }
        const size_t gp = b.find(" timing, ");
        const size_t g = gp == string::npos ? string::npos : b.find(": GPU ", gp);
        if (g != string::npos && b.find(" ms per frame", g) != string::npos) {
            const double ms = atof(b.c_str() + g + 6);
            if (ms > 0 && ms < 1000) { wchar_t buf[16]; swprintf(buf, 16, L"%.1f", ms); r.rrMs = buf; r.rrUsed = true; }  // frames it ran
            // "timing, 300 frames from frame 35701: GPU ...": the frames it evaluated (r4 #1); its sizes ("render 642x361, output
            // 1920x1080") for Last run where the front end logged none
            const long long n = atoll(b.c_str() + gp + 9);
            if (n > 0 && b.compare(gp + 9 + std::to_string(n).size(), 7, " frames") == 0) timed += n;
            // (after the stage times, which have their own "output 0.413")
            const size_t rp = b.find(" render ", b.find(')', g) == string::npos ? g : b.find(')', g));
            const string rs = rp == string::npos ? string() : WordAfter(b.substr(rp), "render "), os = rp == string::npos ? string() : WordAfter(b.substr(rp), "output ");
            if (rs.find('x') != string::npos && os.find('x') != string::npos) { rrRender = FromUtf8(rs); rrOutput = FromUtf8(os); }
        }
        // only evidence of frames RR evaluated successfully counts (r5 #1). The plugin's evaluate counters in its exit summary
        // ("RR evaluates N", "... evaluates N") count calls before they are evaluated: refused inputs and failed evaluations the
        // plugin passed through count there too, so they never count here. The runtime's own count as an instance ends, its
        // failures taken off: "destroy instance 1 after 36044 frames (0 failed)" (each instance of the session)
        if (const size_t dp = b.find("destroy instance "); dp != string::npos) {
            const size_t af = b.find(" after ", dp), fp = af == string::npos ? string::npos : b.find(" frames (", af);
            if (fp != string::npos && isdigit((unsigned char)b[af + 7]))
                evaluated += std::max(0LL, atoll(b.c_str() + af + 7) - atoll(b.c_str() + fp + 9));
        }
        // ...and the plugin's line for an evaluate the runtime finished ("live: evaluate 1: runtime returned 0 (...)", logged for
        // the first one and the first after a failure): one frame each, at least
        if (b.find("live: evaluate ") != string::npos && b.find(": runtime returned 0 ") != string::npos) ++finished;
        // ...and the direct route's own line for one (r7 #4: "RR feature 1 first evaluate: render ... -> output ...; inputs: ...;
        // runtime returned 0 (N frame(s) passed through so far)", after the route's prefix; also "RR feature 1 evaluate 57: ..."),
        // logged for the first one and the first after a failure: one frame each, at least (its failures say "runtime returned <code>;")
        else if (const size_t fp = b.find("RR feature "); fp != string::npos && b.find(" evaluate", fp) != string::npos &&
                 b.find("; runtime returned 0 (", fp) != string::npos)
            ++finished;
    }
    // a timing line is frames the runtime read back from the GPU, so at least one frame even where its count is missing (an older
    // runtime's line): never "no successful frame" beside a measured cost (r6 #2)
    r.rrFrames = std::max({timed, evaluated, finished, r.rrMs.empty() ? 0LL : 1LL});
    if (r.rrFrames > 0) {
        r.rrUsed = true;
        if (r.renderSize.empty() && r.outputSize.empty()) { r.renderSize = rrRender; r.outputSize = rrOutput; }  // (an RR-only run's sizes)
    }
}
bool OptiLogMentionsXess(const string& text) {
    for (size_t i = 0; i + 4 <= text.size(); ++i)
        if ((text[i] | 32) == 'x' && (text[i + 1] | 32) == 'e' && (text[i + 2] | 32) == 's' && (text[i + 3] | 32) == 's') return true;
    return false;
}
static string ReadTail(const wstring& path, FILETIME* written) {
    HANDLE h = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, 0, nullptr);
    if (h == INVALID_HANDLE_VALUE) return string();
    LARGE_INTEGER size{};
    GetFileSizeEx(h, &size);
    const long long want = std::min<long long>(size.QuadPart, 256 * 1024);
    LARGE_INTEGER at; at.QuadPart = size.QuadPart - want;
    SetFilePointerEx(h, at, nullptr, FILE_BEGIN);
    string text((size_t)want, '\0');
    DWORD got = 0;
    if (want > 0) ReadFile(h, &text[0], (DWORD)want, &got, nullptr);
    text.resize(got);
    if (written) GetFileTime(h, nullptr, nullptr, written);
    CloseHandle(h);
    return text;
}

wstring FrontEndLog(const wstring& installDir) {
    const wstring here = Join(installDir, L"wronggpu-fsr.log"), slot = Join(Join(installDir, kOptiSlotDir), L"wronggpu-fsr.log");
    if (!IsFile(slot)) return here;
    if (!IsFile(here)) return slot;
    WIN32_FILE_ATTRIBUTE_DATA a{}, b{};
    if (!GetFileAttributesExW(here.c_str(), GetFileExInfoStandard, &a) || !GetFileAttributesExW(slot.c_str(), GetFileExInfoStandard, &b)) return here;
    return CompareFileTime(&b.ftLastWriteTime, &a.ftLastWriteTime) > 0 ? slot : here;
}

wstring LastRunLog(const wstring& installDir, const LastRun& r) {
    const wstring fe = FrontEndLog(installDir), rr = RrLogPath(installDir);
    if (r.rrOnly && IsFile(rr)) return rr;  // the newer RR session is the run, not the front end's older one (r6 #3)
    return !IsFile(fe) && IsFile(rr) ? rr : fe;
}

bool CountRrRun(LastRun& r) {
    // a session of the RR plugin where the game asked for RR is a run (r4 #1): the RR games (Borderlands 4, Hogwarts Legacy, RE
    // Requiem) run Ray Reconstruction only, and the front end writes no log there. An attempt counts whether or not a frame
    // succeeded (r6 #2: a failed one is a run that says so, never "Not played yet" or an older run's success); "upscaled" is RR
    // frames evaluated successfully (r5 #1)
    // 0.6.1: a session whose guard kept WrongGPU RR off (rrGuard) is a run too: Last run says why
    if (!r.rrFound || (!r.rrUsed && r.rrGuard.empty())) return false;
    // the same game session in the front end's log too (r5 #3): its own evidence stays as it is (created: the game asked for
    // Model M; upscaled: Model M upscaled a frame), so the page never says Model M worked, or failed, on RR's frames; RR's
    // frames show on the Denoiser row. Where Model M upscaled none, RR did the upscaling (it denoises and upscales in one pass,
    // r6 #1): the run worked, on RR's frames. Not with Model M turned off (AMD FSR), nor where it failed to start (created,
    // never ready: its failure stays the run's story)
    if (r.found) {
        const bool noStart = r.upscaler == L"network" && !r.ready && r.created;
        r.rrUpscaled = r.rrFrames > 0 && !r.upscaled && r.upscaler != L"amd" && !noStart;
        return true;
    }
    r.rrOnly = true;
    r.found = true;
    r.created = r.rrUsed;  // (a guard session alone: the game's request never reached WrongGPU RR)
    r.rrGuardOff = !r.rrUsed && !r.rrGuard.empty();
    r.upscaled = r.rrFrames > 0;
    return true;
}

wstring LastRunCacheFlags(const LastRun& r) {
    auto bit = [](bool b) { return b ? L'1' : L'0'; };
    return {bit(r.found), bit(r.ready), bit(r.optiPlugin), bit(r.upscaled), bit(r.rrOnly), bit(r.rrUpscaled), bit(r.rrGuardOff)};
}
bool ReadLastRunCacheFlags(const wstring& f, LastRun& r) {
    if (f.size() != 3 && f.size() != 5 && f.size() != 6 && f.size() != 7) return false;  // (3: before r4; 5: before r6; 6: before 0.6.1)
    for (wchar_t c : f) if (c != L'0' && c != L'1') return false;
    r.found = f[0] == L'1'; r.ready = f[1] == L'1'; r.optiPlugin = f[2] == L'1';
    if (f.size() >= 5) { r.upscaled = f[3] == L'1'; r.rrOnly = f[4] == L'1'; }
    if (f.size() >= 6) r.rrUpscaled = f[5] == L'1';
    if (f.size() >= 7) r.rrGuardOff = f[6] == L'1';
    return true;
}

wstring RrLogPath(const wstring& installDir) {
    const wstring here = Join(installDir, L"wronggpu-rr.log"), mirrored = Join(Join(installDir, kMirrorDir), L"wronggpu-rr.log");
    WIN32_FILE_ATTRIBUTE_DATA a{}, b{};
    if (GetFileAttributesExW(mirrored.c_str(), GetFileExInfoStandard, &b) &&
        (!GetFileAttributesExW(here.c_str(), GetFileExInfoStandard, &a) || CompareFileTime(&b.ftLastWriteTime, &a.ftLastWriteTime) > 0))
        return mirrored;
    return here;
}

static long long LocalSeconds(const FILETIME& ft) {
    FILETIME lt{};
    if (!FileTimeToLocalFileTime(&ft, &lt)) lt = ft;
    ULARGE_INTEGER u;
    u.LowPart = lt.dwLowDateTime;
    u.HighPart = lt.dwHighDateTime;
    return (long long)(u.QuadPart / 10000000ULL);
}
SessionSpan SpanOf(const LastRun::Clock& c, const FILETIME& written) {
    const long long w = LocalSeconds(written);
    SessionSpan s;
    s.pid = c.pid;
    s.start = s.end = w;
    if (c.end < 0) return s;  // no time of day in the log: its last write
    long long end = w - w % 86400 + c.end;
    if (end > w + 600) end -= 86400;  // later in the day than the file's last write: logged the day before (a copied log)
    s.end = end;
    s.start = c.start < 0 ? end : end - ((c.end - c.start + 86400) % 86400);
    return s;
}
RrJoin JoinRrSession(const SessionSpan& front, const SessionSpan& rr) {
    const long long kSlack = 120;
    const bool overlap = front.start <= rr.end + kSlack && rr.start <= front.end + kSlack;
    if (overlap && (!front.pid || !rr.pid || front.pid == rr.pid)) return RrJoin::Same;
    return rr.start > front.start ? RrJoin::RrNewer : RrJoin::FrontNewer;
}
// The RR session's evidence into the run (the same game session, or the RR session alone)
static void TakeRr(LastRun& r, const LastRun& rr) {
    r.rrFound = rr.rrFound; r.rrUsed = rr.rrUsed; r.rrMode = rr.rrMode; r.rrMs = rr.rrMs; r.rrFrames = rr.rrFrames; r.rrClock = rr.rrClock;
    r.rrNote = rr.rrNote; r.rrPassThrough = rr.rrPassThrough; r.rrGuard = rr.rrGuard;
    if (!rr.rrGuard.empty()) r.warnings.push_back({L"rr-guard", rr.rrGuard});  // 0.6.1: the RR plugin's guard line, shown in Last run
    if (r.renderSize.empty() && r.outputSize.empty()) { r.renderSize = rr.renderSize; r.outputSize = rr.outputSize; }  // (an RR-only run's sizes)
}

LastRun ReadLastRun(const wstring& installDir) {
    LastRun asi;
    for (const wchar_t* sub : {L"wronggpu-plugins", L"plugins", L""}) {  // the plugin's own folder, plugins\, next to OptiScaler
        const wstring p = Join(*sub ? Join(installDir, sub) : installDir, L"wronggpu-asi.log");
        if (IsFile(p)) { ParseAsiLog(ReadTail(p, nullptr), asi); break; }
    }
    const wstring path = FrontEndLog(installDir);
    FILETIME wt{};
    bool have = IsFile(path);
    LastRun r = have ? ParseLastRun(ReadTail(path, &wt)) : LastRun();
    // Ray Reconstruction (rr.h): the RR plugin's log next to the exe, or in REFramework's mirror where the game runs the
    // interface it hooks from there (gamerules.h); the newer one when both are there
    const wstring rrPath = RrLogPath(installDir);
    FILETIME rwt{};
    LastRun rr;
    if (IsFile(rrPath)) ParseRrLog(ReadTail(rrPath, &rwt), rr);
    // the two logs' last sessions: combined only when they are one game session; otherwise the newer alone (r5 #2): an RR session
    // newer than the front end's where the game asked for RR is the run (RR-only), with or without a frame that succeeded (r6 #2:
    // a failed attempt never leaves the older front end's success in its place); an older one, or a newer one where the game never
    // asked for RR, is left out (a front end's session of the same launch would have overlapped it)
    if (rr.rrFound) {
        const RrJoin join = !have || !r.found ? RrJoin::RrNewer : JoinRrSession(SpanOf(r.clock, wt), SpanOf(rr.rrClock, rwt));
        if (join == RrJoin::Same || (join == RrJoin::RrNewer && !(have && r.found))) TakeRr(r, rr);
        else if (join == RrJoin::RrNewer && (rr.rrUsed || !rr.rrGuard.empty())) { r = LastRun(); TakeRr(r, rr); have = false; }
    }
    r.plugin = asi.plugin;
    for (auto& w : asi.warnings) r.warnings.push_back(w);
    const bool rrRun = CountRrRun(r);
    for (auto& w : r.warnings)  // the cause of a missing depth buffer, when OptiScaler's log names XeSS
        if (w.kind == L"no-depth" && OptiLogMentionsXess(ReadTail(Join(installDir, L"OptiScaler.log"), nullptr))) {
            w.kind = L"xess-no-depth";
            w.text = L"The game's XeSS inputs give the upscaler no depth buffer (seen in Hogwarts Legacy): choose DLSS or FSR as the game's upscaler, or turn "
                     L"OptiScaler's XeSS inputs off on the OptiScaler tab.";
        }
    FILETIME lt; SYSTEMTIME st;
    if (rrRun && (!have || CompareFileTime(&rwt, &wt) > 0)) wt = rwt;  // the RR log is newer, or the only log: its time
    if ((have || rrRun) && FileTimeToLocalFileTime(&wt, &lt) && FileTimeToSystemTime(&lt, &st)) {
        wchar_t b[32]; swprintf(b, 32, L"%04u-%02u-%02u %02u:%02u", st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute);
        r.when = b;
    }
    return r;
}

}  // namespace wg
