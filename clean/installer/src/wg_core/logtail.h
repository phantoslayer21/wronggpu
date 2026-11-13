// SPDX-License-Identifier: MIT
// WrongGPU launcher core: "Last run" from the tail of a game's wronggpu-fsr.log (build/launcher-design.md section 3):
// which upscaler the front end used, whether the network runtime started and on which executor, fallbacks to AMD's
// FSR and the last support code; warnings the launcher shows (M2): command lists recorded into another tool's tables
// (foreign vtable: AMD FSR fallback), upscaler inputs without depth (XeSS inputs in some games), WrongGPU's OptiScaler
// plugin not hooked (wronggpu-asi.log). Read-only. Written for this project.
#pragma once
#include "util.h"

namespace wg {

struct LastRun {
    bool found = false;           // a log with at least one session
    wstring when;                 // the log's last write, local time "YYYY-MM-DD HH:MM"
    wstring process;              // the session's exe name
    wstring upscaler;             // network | amd | standin (settings: upscaler=...)
    wstring executor;             // dx12 | hip (network runtime ready: executor ...)
    bool ready = false;           // the network runtime started in this session
    bool created = false;         // the game asked for an upscaler in it (an ffxCreateContext, a frame, or the runtime ready):
                                  // a session without it is a run where WrongGPU was not used (UI review #9)
    bool upscaled = false;        // Model M upscaled at least one frame (the runtime logs "ready" before any frame, r2 #10): a
                                  // frame logged once it was ready whose dispatch ended "-> OK (WrongGPU)" with no stand-in,
                                  // refusal or AMD FSR line before it, "WrongGPU upscales again after", or the runtime's timing
                                  // line counting network frames (r3 #2: the late start's "it upscales from this frame on" is
                                  // logged before its frame's dispatch, so it alone is not enough; only the first frames of a
                                  // context are logged unless log_dispatches is on)
    bool optiPlugin = false;      // WrongGPU was selected through OptiScaler's upscaler list by name (the plugin, or the custom
                                  // upscaler slot of WrongGPU's edition: r3 #3)
    std::vector<wstring> notes;   // fallbacks, foreign command-list tables, errors (last few, oldest first)
    std::vector<wstring> routine; // start-up notes of a run whose runtime became ready (RoutineNote): not counted, not shown
    std::vector<wstring> late;    // notes (also) logged after the runtime became ready: never routine (UI review r2 #4)
    wstring support;              // the last [support] WG-... code
    wstring renderSize, outputSize;  // the last "WrongGPU frame:" line's R=WxH and O=WxH (the window's stat tiles)
    wstring plugin;               // WrongGPU.asi: "hooked", "not-hooked" (wronggpu-asi.log), empty without its log
    struct Warning { wstring kind, text; };  // kind: foreign-vtable, no-depth, xess-no-depth, plugin-not-hooked
    std::vector<Warning> warnings;
    // Ray Reconstruction (wronggpu-rr.log, the RR plugin's log next to the exe; its last session)
    bool rrFound = false;         // the RR plugin's log has a session
    bool rrUsed = false;          // the game asked for Ray Reconstruction in it (the RR runtime made an instance, or its create failed,
                                  // or the plugin's direct route refused the game's RR feature: r7 #3): an attempt, whether
                                  // or not a frame succeeded (rrFrames says that: r6 #2)
    wstring rrMode;               // exact | accurate | fast: the mode it ran last
    wstring rrMs;                 // the last timing line's GPU time a frame ("6.8"); empty when the log has none
    long long rrFrames = 0;       // the frames Ray Reconstruction evaluated successfully in that session (r5 #1): its timing lines'
                                  // counts ("timing, 300 frames"), the runtime's "destroy instance 1 after N frames (F failed)" as
                                  // N - F, or the plugin's "live: evaluate N: runtime returned 0" lines and its direct route's
                                  // "RR feature N first evaluate: ...; runtime returned 0" lines (r7 #4), whichever is larger; never
                                  // the plugin's evaluate counters ("RR evaluates N": calls, counted before they are evaluated)
    wstring rrNote;               // the RR log's last failure reason (r6 #2): the runtime's "evaluate failed before any pass: <why>" or
                                  // "create: failed (<why>)", or the direct route's "RR feature refused (...): <why>" (r7 #3); empty without one
    bool rrPassThrough = false;   // the plugin passed the game's colour through for a failed evaluate ("colour passed through")
    // A session of the RR plugin where the game asked for RR counts as a run (r4 #1: Borderlands 4, Hogwarts Legacy and RE Requiem
    // run Ray Reconstruction only, so the front end writes no log there; r6 #2: an attempt counts whether or not a frame succeeded):
    // without a front-end session ReadLastRun sets found and created for it, and upscaled only with frames evaluated ("upscaled" for
    // an RR game: RR frames evaluated successfully); with one (the same game session) found, created and upscaled stay the front
    // end's own (r5 #3). when from the RR log where it is newer or the only log.
    bool rrOnly = false;          // the run is known from the RR log alone (no front-end session): Last run shows the Denoiser
    // One game session in both logs where Ray Reconstruction evaluated frames and Model M upscaled none (r6 #1): RR did the
    // upscaling there (it denoises and upscales in one pass while it's on), so the run counts as working (the checklist, the
    // header's Worked) without being Model M's evidence: upscaled stays the front end's own. Not where Model M was turned off
    // (upscaler=amd) or failed to start (created, never ready): those keep their own story (CountRrRun)
    bool rrUpscaled = false;
    // Which game session each log's last session is (r5 #2): its process id (the front end's "session start: X (process N)", the
    // RR plugin's "load: process N attached"; 0: not in the log's tail) and its first and last logged time of day (seconds; -1:
    // none). ReadLastRun combines the two logs only for the same session
    struct Clock { unsigned long pid = 0; int start = -1, end = -1; };
    Clock clock;                  // the front end's session (ParseLastRun)
    Clock rrClock;                // the RR plugin's session (ParseRrLog)
    wstring Summary() const;      // one line for the game list
    // WrongGPU upscaled frames in the run: Model M's own frames (upscaled; for an RR-only run its RR frames), or Ray Reconstruction's
    // in one session with the front end where Model M upscaled none (rrUpscaled, r6 #1). The checklist and the header's Worked test
    bool AnyUpscaled() const { return found && (upscaled || rrUpscaled); }
};

// A log's last session on the PC's local clock (seconds): its process id, its start and its end. From the times of day the log
// has, on the day of the file's last write (the day before when the last one is later than the write: a copied log).
struct SessionSpan { unsigned long pid = 0; long long start = 0, end = 0; };
SessionSpan SpanOf(const LastRun::Clock& c, const FILETIME& written);
// Whether the front end's session and the RR plugin's session are one game session (r5 #2): both process ids known: the same id
// and the times overlap (an id can come again in a later launch); an id missing: the times overlap (two minutes' slack). Else
// the one that started later is the newer.
enum class RrJoin { Same, FrontNewer, RrNewer };
RrJoin JoinRrSession(const SessionSpan& front, const SessionSpan& rr);

// A log line's text (after its "[time pid] [tag]" prefix) that counts as a warning on the game page: an error (ERROR as
// a word, WG-ERROR), a refused split, a fallback to AMD's FSR with its reason. Info lines never count (hooks installed,
// "upscaler info:", the settings line, runtime ready), even when they quote WG_ERROR_* codes or mention AMD's FSR.
bool IsProblemLine(const string& body);
// A dispatch result of the direct route's FSR 3.0 hook saying WrongGPU's network upscaled the frame ("-> FFX_OK (WrongGPU
// network)", with AMD first for frame generation, or AMD's own call skipped); not AMD's output kept, not the stand-in (r4 #3).
bool DirectOk(const string& body);
// A problem line that can be routine (UI review #8): the first frames go to AMD FSR while the network runtime starts
// ("refused (WG_ERROR_NOT_READY)", "until the runtime is ready"). The same code also says a split was refused for a real
// reason (a list's table WrongGPU can't take over, a split lane in use), so it is routine only before the runtime is ready.
bool RoutineNote(const wstring& note);
// The notes of a run split for the page: once the runtime became ready, the routine kind logged only before that moves from
// notes to routine. One logged (again) after it (r.late) stays a note, and so does every one of a run whose runtime never
// started (r2 #4).
void SplitRoutineNotes(LastRun& r);
// Parses the text of a log (the part after the last "==== session start" line counts).
LastRun ParseLastRun(const string& text);
// The plugin's own log (wronggpu-asi.log, its last load): hooked or not, and why not.
void ParseAsiLog(const string& text, LastRun& r);
// The RR plugin's log (wronggpu-rr.log): its last session ("load: process N attached ..."): whether the RR runtime made an
// instance (the game asked for Ray Reconstruction), the mode it ran last (the plugin's start, the runtime's instance line,
// a switch in OptiScaler's menu or in the runtime), the last timing line's GPU time a frame. A session whose "attached" line has
// scrolled out of the tail (no other load line in it, but its instance or timing lines): the whole tail.
void ParseRrLog(const string& text, LastRun& r);
// OptiScaler.log names XeSS as the game's upscaler input (only used to name the cause of a "no depth" warning).
bool OptiLogMentionsXess(const string& text);
// Reads the last 256 KB of <dir>\wronggpu-fsr.log, the plugin's log (next to OptiScaler or in its own folder), the RR
// plugin's log (RrLogPath) and OptiScaler.log. An RR session where the game asked for RR counts as a run (LastRun::rrOnly),
// with or without a frame that succeeded (r6 #2). The two logs' last sessions are combined only when they are one game session
// (JoinRrSession); otherwise the newer one alone counts: an RR session newer than the front end's, where the game asked for RR,
// is an RR-only run (a failed attempt too: never the older front end's success in its place); an older one is left out (r5 #2).
LastRun ReadLastRun(const wstring& installDir);
// After ParseLastRun and ParseRrLog of one game session: an RR session where the game asked for RR (rrUsed: an attempt, r6 #2)
// counts for the run. Without a front-end session it is the run (found, created, rrOnly: r4 #1; upscaled only with frames
// evaluated, r5 #1); with one, the front end's created and upscaled stay as they are (r5 #3), and where Model M upscaled no frame
// and RR evaluated frames, RR did the upscaling (rrUpscaled, r6 #1). True when the RR session was an attempt.
bool CountRrRun(LastRun& r);
// The RR plugin's wronggpu-rr.log: next to the exe, or in REFramework's mirror (gamerules.h kMirrorDir); the newer one when
// both are there.
wstring RrLogPath(const wstring& installDir);
// The front end's wronggpu-fsr.log: next to the exe, or in OptiScaler 10's custom upscaler slot (optiscaler.h
// kOptiSlotDir); the newer one when both are there.
wstring FrontEndLog(const wstring& installDir);
// The log that Last run shows (Open log, r6 #3): the RR plugin's log for a run known from it alone (rrOnly: also where an older
// front-end log is there), else the front end's log, else the RR plugin's (a game that ran RR only). May not exist.
wstring LastRunLog(const wstring& installDir, const LastRun& r);

}  // namespace wg
