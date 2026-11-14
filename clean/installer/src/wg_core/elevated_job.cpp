// SPDX-License-Identifier: MIT
// The launcher's elevated step (elevated_job.h): the reply pipe, the worker's checks before it runs anything, and the
// launcher's side of one worker run. Written for this project.
#include "elevated_job.h"

#include <sddl.h>

#include <map>
#include <mutex>
#include <thread>

#include "detect.h"  // GameRoot
#include "package_snapshot.h"
#include "secure_fs.h"

namespace wg {

namespace {

const wchar_t* const kPipePrefix = L"\\\\.\\pipe\\WrongGPU-reply-";
const size_t kMaxReply = 4u << 20;     // bytes one connection may send
const DWORD kPipeBuffer = 64 * 1024;

// Records the worker sends: a type byte, a 32-bit job number (1-based; 0: none), a 32-bit length, that many bytes (UTF-8).
enum : char { kRecLog = 'L', kRecResult = 'R', kRecRefused = 'X', kRecEnd = 'E' };

struct HandleCloser { void operator()(HANDLE h) const { if (h && h != INVALID_HANDLE_VALUE) CloseHandle(h); } };
using Held = std::unique_ptr<void, HandleCloser>;
Held Hold(HANDLE h) { return Held(h == INVALID_HANDLE_VALUE ? nullptr : h); }

void PutU32(string& s, uint32_t v) { for (int i = 0; i < 4; ++i) s.push_back((char)((v >> (8 * i)) & 0xff)); }
uint32_t GetU32(const string& s, size_t at) {
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) v |= (uint32_t)(unsigned char)s[at + i] << (8 * i);
    return v;
}

// The records of one connection, in order, into a reply.
void ParseRecords(const string& bytes, size_t jobs, WorkerReply& out) {
    out.results.assign(jobs, {false, OpResult()});
    size_t at = 0;
    while (at + 9 <= bytes.size()) {
        const char type = bytes[at];
        const uint32_t n = GetU32(bytes, at + 1), len = GetU32(bytes, at + 5);
        if (len > bytes.size() - at - 9) break;
        const string body = bytes.substr(at + 9, len);
        at += 9 + len;
        if (out.complete) break;  // nothing counts after the end record
        if (type == kRecLog) out.log.push_back(FromUtf8(body));
        else if (type == kRecResult && n >= 1 && n <= jobs) {
            OpResult r;
            if (ParseResult(body, r)) out.results[n - 1] = {true, r};
        } else if (type == kRecRefused) {
            out.refused = ParseResult(body, out.refusal);
        } else if (type == kRecEnd) out.complete = true;
    }
}

}  // namespace

const wchar_t* const kElevationNotGranted = L"Administrator permission is needed to change this game folder and was not granted.";

ProcessIdentity IdentityOf(HANDLE process) {
    ProcessIdentity id;
    FILETIME created{}, exited{}, kernel{}, user{};
    if (!process || !GetProcessTimes(process, &created, &exited, &kernel, &user)) return id;
    id.pid = GetProcessId(process);
    id.started = ((unsigned long long)created.dwHighDateTime << 32) | created.dwLowDateTime;
    return id;
}

bool IsReplyPipeName(const wstring& name) {
    const wstring pre = kPipePrefix;
    if (name.size() != pre.size() + 32 || name.compare(0, pre.size(), pre) != 0) return false;
    for (size_t i = pre.size(); i < name.size(); ++i)
        if (!((name[i] >= L'0' && name[i] <= L'9') || (name[i] >= L'a' && name[i] <= L'f'))) return false;
    return true;
}

// ---- the reply pipe (the launcher's end) ------------------------------------------------------------------------------
struct ReplyChannel::Impl {
    wstring name;
    Held pipe, stop, io;
    std::thread reader;
    struct Connection { ProcessIdentity who; string bytes; bool ended = false; };
    std::mutex m;
    std::vector<Connection> done;

    // Waits for the pending operation on `ov`; false when stop was signalled first (the operation is then cancelled). The
    // operation's own completion is looked at first, so what a finished worker left in the pipe is never cut off.
    bool Wait(OVERLAPPED& ov, DWORD& got, bool& ok) {
        HANDLE hs[2] = {io.get(), stop.get()};
        const DWORD w = WaitForMultipleObjects(2, hs, FALSE, INFINITE);
        if (w == WAIT_OBJECT_0) { ok = GetOverlappedResult(pipe.get(), &ov, &got, FALSE) != FALSE; return true; }
        CancelIoEx(pipe.get(), &ov);
        GetOverlappedResult(pipe.get(), &ov, &got, TRUE);
        return false;
    }
    void Run() {
        for (;;) {
            OVERLAPPED ov{}; ov.hEvent = io.get();
            ResetEvent(io.get());
            DWORD got = 0;
            bool ok = ConnectNamedPipe(pipe.get(), &ov) != FALSE;
            DWORD e = ok ? 0 : GetLastError();
            // connected before this call (and maybe closed again: what it sent is still read)
            if (!ok && (e == ERROR_PIPE_CONNECTED || e == ERROR_NO_DATA)) ok = true;
            else if (!ok && e == ERROR_IO_PENDING) { if (!Wait(ov, got, ok)) return; }
            if (!ok) {
                if (WaitForSingleObject(stop.get(), 10) == WAIT_OBJECT_0) return;
                DisconnectNamedPipe(pipe.get());
                continue;
            }
            Connection c;
            ULONG pid = 0;
            if (GetNamedPipeClientProcessId(pipe.get(), &pid)) {
                Held p = Hold(OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid));
                c.who = IdentityOf(p.get());
            }
            bool stopped = false;
            std::vector<char> buf(kPipeBuffer);
            for (;;) {
                OVERLAPPED rv{}; rv.hEvent = io.get();
                ResetEvent(io.get());
                got = 0;
                ok = ReadFile(pipe.get(), buf.data(), (DWORD)buf.size(), &got, &rv) != FALSE;
                e = ok ? 0 : GetLastError();
                if (!ok && e == ERROR_IO_PENDING) { if (!Wait(rv, got, ok)) { stopped = true; break; } e = ok ? 0 : GetLastError(); }
                if (!ok) { c.ended = e == ERROR_BROKEN_PIPE; break; }
                c.bytes.append(buf.data(), got);
                if (c.bytes.size() > kMaxReply) break;
            }
            DisconnectNamedPipe(pipe.get());
            { std::lock_guard<std::mutex> l(m); done.push_back(std::move(c)); }
            if (stopped) return;
        }
    }
};

ReplyChannel::ReplyChannel() : impl_(new Impl) {}
ReplyChannel::~ReplyChannel() {
    if (impl_->reader.joinable()) { SetEvent(impl_->stop.get()); impl_->reader.join(); }
    delete impl_;
}
const wstring& ReplyChannel::Name() const { return impl_->name; }

bool ReplyChannel::Create(wstring& why) {
    // this account and administrators only; one instance, this computer only, made by this process first (a name that
    // is taken is never used)
    const wstring me = CurrentUserSid();
    if (me.empty()) { why = L"cannot read this account's SID"; return false; }
    const wstring sddl = L"D:P(A;;GA;;;" + me + L")(A;;GA;;;BA)";
    PSECURITY_DESCRIPTOR sd = nullptr;
    if (!ConvertStringSecurityDescriptorToSecurityDescriptorW(sddl.c_str(), SDDL_REVISION_1, &sd, nullptr)) {
        why = L"cannot make the reply pipe's permissions (" + ErrText(GetLastError()) + L")";
        return false;
    }
    SECURITY_ATTRIBUTES sa{sizeof(sa), sd, FALSE};
    for (int i = 0; i < 4 && !impl_->pipe; ++i) {
        impl_->name = kPipePrefix + RandomHex();
        impl_->pipe = Hold(CreateNamedPipeW(impl_->name.c_str(), PIPE_ACCESS_INBOUND | FILE_FLAG_OVERLAPPED | FILE_FLAG_FIRST_PIPE_INSTANCE,
                                            PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS, 1, kPipeBuffer, kPipeBuffer, 0, &sa));
    }
    const DWORD e = GetLastError();
    LocalFree(sd);
    if (!impl_->pipe) { why = L"cannot make the reply pipe (" + ErrText(e) + L")"; return false; }
    impl_->stop = Hold(CreateEventW(nullptr, TRUE, FALSE, nullptr));
    impl_->io = Hold(CreateEventW(nullptr, TRUE, FALSE, nullptr));
    if (!impl_->stop || !impl_->io) { why = L"cannot make the reply pipe's events"; impl_->pipe.reset(); return false; }
    impl_->reader = std::thread([this] { impl_->Run(); });
    return true;
}

WorkerReply ReplyChannel::Finish(const ProcessIdentity& worker, size_t jobs) {
    WorkerReply out;
    out.results.assign(jobs, {false, OpResult()});
    if (impl_->reader.joinable()) { SetEvent(impl_->stop.get()); impl_->reader.join(); }
    if (!worker.pid) return out;
    // exactly one connection of the worker that ended normally
    const Impl::Connection* from = nullptr;
    int count = 0;
    for (auto& c : impl_->done)
        if (c.who == worker) { ++count; from = &c; }
    if (count != 1 || !from->ended) return out;
    ParseRecords(from->bytes, jobs, out);
    out.connected = true;
    return out;
}

// ---- the launcher's side of one worker run ----------------------------------------------------------------------------
std::vector<OpResult> RunElevatedJobs(const std::vector<string>& jobTexts, const wstring& jobsDir, bool stop, const StartWorkerFn& start, const LogFn& log) {
    const size_t n = jobTexts.size();
    std::vector<OpResult> out(n);
    if (!n) return out;
    auto all = [&out](int code, const wstring& message) { for (auto& r : out) { r = OpResult(); r.code = code; r.message = message; } return out; };
    std::vector<wstring> files;
    const wstring stamp = NowStamp(true) + L"-" + RandomHex().substr(0, 8);
    wstring batch;
    auto cleanup = [&] { for (auto& f : files) DeleteFileW(f.c_str()); if (!batch.empty()) DeleteFileW(batch.c_str()); };
    string list;
    try {
        EnsureDir(jobsDir);
        // the folder as the worker finds it (it opens the files without following a link and checks their final path)
        const wstring dir = BoundPath(jobsDir);
        batch = Join(dir, L"batch-" + stamp + L".txt");
        std::vector<std::pair<wstring, wstring>> listed;
        for (size_t i = 0; i < n; ++i) {
            files.push_back(Join(dir, L"job-" + stamp + L"-" + std::to_wstring(i + 1) + L".txt"));
            WriteFileAtomic(files[i], jobTexts[i].data(), jobTexts[i].size());
            listed.push_back({files[i], Sha256Bytes(jobTexts[i].data(), jobTexts[i].size())});
        }
        list = BatchText(listed, stop);
        WriteFileAtomic(batch, list.data(), list.size());
    } catch (const SetupError& e) {
        cleanup();
        return all(EXIT_FAILED, L"The elevated step could not be prepared: " + e.msg);
    }
    ReplyChannel channel;
    wstring why;
    if (!channel.Create(why)) { cleanup(); return all(EXIT_FAILED, L"The elevated step could not be prepared: " + why + L"."); }
    DWORD code = 0;
    ProcessIdentity worker;
    const bool started = start(L"apply \"--batch=" + batch + L"\" --sha256=" + Sha256Bytes(list.data(), list.size()) + L" --pipe=" + channel.Name(), code, worker);
    const WorkerReply reply = channel.Finish(started ? worker : ProcessIdentity(), n);
    cleanup();
    if (log) for (auto& line : reply.log) log(L"elevated step: " + line);
    if (!started) return all(EXIT_REFUSED, kElevationNotGranted);
    if (reply.refused) return all(reply.refusal.code ? reply.refusal.code : EXIT_USAGE, reply.refusal.message);
    for (size_t i = 0; i < n; ++i) {
        if (reply.results[i].first) { out[i] = reply.results[i].second; continue; }
        out[i] = OpResult();
        out[i].code = EXIT_FAILED;
        out[i].message = L"The elevated step did not report a result for this game (exit " + std::to_wstring(code) + L"). Check the game before trying again.";
    }
    if (log && (!reply.connected || !reply.complete)) log(L"elevated step: no complete reply from the worker it started (exit " + std::to_wstring(code) + L")");
    return out;
}

// ---- the worker -------------------------------------------------------------------------------------------------------
namespace {

// The worker's end of the reply pipe.
class Reply {
public:
    bool Open(const wstring& name, wstring& why) {
        if (!IsReplyPipeName(name)) { why = L"no reply pipe was named"; return false; }
        for (int i = 0; i < 3 && !h_; ++i) {
            // identification only: the launcher's end never acts as this process
            h_ = Hold(CreateFileW(name.c_str(), GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, SECURITY_SQOS_PRESENT | SECURITY_IDENTIFICATION, nullptr));
            if (!h_ && GetLastError() == ERROR_PIPE_BUSY) WaitNamedPipeW(name.c_str(), 10000);
            else if (!h_) break;
        }
        if (!h_ || GetFileType(h_.get()) != FILE_TYPE_PIPE) { why = L"the launcher's reply pipe cannot be opened (" + ErrText(GetLastError()) + L")"; h_.reset(); return false; }
        return true;
    }
    void Send(char type, uint32_t n, const string& body) {
        std::lock_guard<std::mutex> l(m_);
        if (!h_) return;
        string rec(1, type);
        PutU32(rec, n);
        PutU32(rec, (uint32_t)body.size());
        rec += body;
        DWORD wrote = 0;
        if (!WriteFile(h_.get(), rec.data(), (DWORD)rec.size(), &wrote, nullptr) || wrote != rec.size()) h_.reset();
    }
    bool ok() const { return (bool)h_; }
private:
    Held h_;
    std::mutex m_;
};

// A file read once into memory (a job or batch file: at most 1 MiB), opened without following a link (secure_fs.h).
bool ReadOnce(const wstring& path, string& bytes, wstring& why) {
    if (!IsAbsolutePath(path)) { why = L"not an absolute path: " + path; return false; }
    FsGuard reader;  // no roots: reading only
    ScopedFsGuard active(&reader);
    Held h;
    try { h = Hold(OpenForRead(path, FILE_SHARE_READ)); }
    catch (const SetupError& e) { why = e.msg; return false; }
    if (!h) { why = L"cannot open " + path + L" (" + ErrText(GetLastError()) + L")"; return false; }
    LARGE_INTEGER size{};
    if (!GetFileSizeEx(h.get(), &size) || size.QuadPart > (1 << 20)) { why = path + L" is not a job file (size)"; return false; }
    bytes.resize((size_t)size.QuadPart);
    size_t done = 0;
    while (done < bytes.size()) {
        DWORD got = 0;
        if (!ReadFile(h.get(), bytes.data() + done, (DWORD)(bytes.size() - done), &got, nullptr) || !got) { why = L"cannot read " + path; return false; }
        done += got;
    }
    return true;
}

bool SameHash(const wstring& a, const wstring& b) { return a.size() == 64 && IEq(a, b); }
bool UsesPackage(Op op) { return op == Op::Install || op == Op::Update || op == Op::Repair; }

// The folders a job may change (secure_fs.h): the worker's private folder, the player's local app data folder, the game's
// folder (its root: GameRoot of the exe's folder or of the job's folder) and the player's Documents folder. Each is opened
// and held by the job's guard before any job runs; one that is not there is not a root (nothing in it is changed: the job
// then fails when it runs, as it would have), one with a link on its way or elsewhere than its path says is refused.
bool GuardJob(const WorkerJob& w, const wstring& privateDir, FsGuard& g, wstring& why) {
    std::vector<wstring> roots{privateDir}, maybe{w.localAppData, w.documents};
    if (!w.r.exe.empty()) maybe.push_back(GameRoot(Parent(w.r.exe)));
    if (!w.r.dir.empty()) maybe.push_back(GameRoot(w.r.dir));
    {
        FsGuard reader;
        ScopedFsGuard active(&reader);
        for (auto& m : maybe) {
            if (m.empty()) continue;
            try { if (IsDir(m)) roots.push_back(m); }
            catch (const SetupError& e) { why = e.msg; return false; }
        }
    }
    std::vector<wstring> added;
    for (auto& r : roots) {
        bool dup = false;
        for (auto& a : added) dup = dup || IEq(NormalPath(a), NormalPath(r));
        if (dup) continue;
        if (!g.AddRoot(r, why)) return false;
        added.push_back(r);
    }
    return true;
}

}  // namespace

WorkerRun RunWorker(const WorkerArgs& a, Context& ctx, const WorkerRunFn& run, const wstring& privateParent) {
    WorkerRun out;
    Reply reply;
    wstring why;
    out.last.code = EXIT_USAGE;
    if (!reply.Open(a.pipe, why)) {  // no way to report: nothing is run
        out.refused = true;
        out.last.message = L"The elevated step did not run: " + why + L".";
        return out;
    }
    // from here on the log lines go back to the launcher; the private folder and the log sink are undone on every return
    wstring privateDir;
    const auto oldSink = ctx.logSink;
    const wstring oldRoot = ctx.workRoot;
    {
        std::lock_guard<std::mutex> l(ctx.logMutex);
        ctx.logSink = [&reply](const wstring& s) { reply.Send(kRecLog, 0, ToUtf8(s)); };
    }
    struct Undo {
        Context& ctx; const std::function<void(const wstring&)>& sink; const wstring& root; wstring& dir;
        ~Undo() {
            if (!dir.empty()) {
                if (!ctx.opti.tempDir.empty() && StartsWithI(ctx.opti.tempDir, dir)) ctx.opti = OptiPackage();
                if (!ctx.opti.dir.empty() && StartsWithI(ctx.opti.dir, dir)) ctx.opti = OptiPackage();
                // removed below its parent's held handle (the jobs' guards, which hold the folder itself, are gone by now)
                try {
                    FsGuard g;
                    wstring why;
                    if (g.AddRoot(Parent(dir), why)) { ScopedFsGuard active(&g); RemoveTree(dir); }
                } catch (const SetupError&) {}
            }
            ctx.workRoot = root;
            std::lock_guard<std::mutex> l(ctx.logMutex);
            ctx.logSink = sink;
        }
    } undo{ctx, oldSink, oldRoot, privateDir};
    auto refuse = [&](const wstring& reason) {
        out.refused = true;
        out.ran = 0;
        out.last = OpResult();
        out.last.code = EXIT_USAGE;
        out.last.message = L"The elevated step did not run: " + reason + L".";
        LogLine(ctx, L"elevated worker: refused: " + reason);
        reply.Send(kRecRefused, 0, ResultText(out.last));
        reply.Send(kRecEnd, 0, "");
        return out;
    };

    // 1. the batch and every job file, each read once and checked against the request
    if (a.batch.empty() || a.sha256.empty()) return refuse(L"apply needs --batch=<file>, --sha256=<its SHA-256> and --pipe=<name>");
    string text;
    if (!ReadOnce(a.batch, text, why)) return refuse(why);
    if (!SameHash(Sha256Bytes(text.data(), text.size()), a.sha256))
        return refuse(BaseName(a.batch) + L" is not the file the launcher wrote for this request (SHA-256 mismatch)");
    std::vector<std::pair<wstring, wstring>> listed;
    bool stop = false;
    if (!ParseBatch(text, listed, stop, &why)) return refuse(L"bad batch file: " + why);
    out.jobs = (int)listed.size();
    std::vector<WorkerJob> jobs(listed.size());
    for (size_t i = 0; i < listed.size(); ++i) {
        const wstring label = L"job " + std::to_wstring(i + 1) + L" (" + BaseName(listed[i].first) + L")";
        string jt;
        if (!ReadOnce(listed[i].first, jt, why)) return refuse(label + L": " + why);
        if (!SameHash(Sha256Bytes(jt.data(), jt.size()), listed[i].second))
            return refuse(label + L" is not the job file the launcher wrote for this request (SHA-256 mismatch)");
        WorkerJob& w = jobs[i];
        if (!ParseJob(jt, w.r, w.localAppData, &why, &w.userSid, &w.perGameEntry, &w.documents)) return refuse(label + L": bad job file: " + why);
    }

    // 2. the private folder for this run's temporary files, and the OptiScaler packages copied into it and checked (the
    //    package read and the copy written through a guard whose one root is the private folder: secure_fs.h)
    if (!MakePrivateFolder(privateParent, privateDir, why)) { privateDir.clear(); return refuse(why); }
    ctx.workRoot = privateDir;
    {
        FsGuard snapshot;
        if (!snapshot.AddRoot(privateDir, why)) return refuse(why);
        ScopedFsGuard active(&snapshot);
        std::map<wstring, wstring> copies;  // package path + digest -> its checked copy
        for (size_t i = 0; i < jobs.size(); ++i) {
            OpRequest& r = jobs[i].r;
            if (!UsesPackage(r.op)) { r.optiPackage.clear(); r.optiDigest.clear(); continue; }
            if (r.optiPackage.empty()) continue;
            const wstring key = Lower(FullPath(r.optiPackage)) + L"|" + Lower(r.optiDigest);
            auto it = copies.find(key);
            if (it == copies.end()) {
                const wstring into = Join(privateDir, L"package-" + std::to_wstring(copies.size() + 1));
                wstring copy;
                try { EnsureDir(into); } catch (const SetupError& e) { return refuse(e.msg); }
                if (!SnapshotPackage(r.optiPackage, r.optiDigest, into, copy, why)) return refuse(L"job " + std::to_wstring(i + 1) + L": " + why);
                LogLine(ctx, L"elevated worker: OptiScaler package " + r.optiPackage + L" checked (" + r.optiDigest.substr(0, 12) + L")");
                it = copies.emplace(key, copy).first;
            }
            r.optiPackage = it->second;
        }
    }

    // 3. every job's folders opened and held without following a link, each where its path says (secure_fs.h): a job whose
    //    folder fails this runs, like any job that fails a check, no job at all
    std::vector<std::unique_ptr<FsGuard>> guards;
    for (size_t i = 0; i < jobs.size(); ++i) {
        guards.push_back(std::make_unique<FsGuard>());
        if (!GuardJob(jobs[i], privateDir, *guards.back(), why))
            return refuse(L"job " + std::to_wstring(i + 1) + L" (" + BaseName(listed[i].first) + L"): " + why);
    }

    // 4. every job checked: run them in order, each through its guard
    bool failed = false;
    out.last = OpResult();
    for (size_t i = 0; i < jobs.size(); ++i) {
        OpResult res;
        if (stop && failed) {
            res.code = kOpNotRun;
            res.message = L"not run: an earlier game failed";
        } else {
            ScopedFsGuard active(guards[i].get());
            try { res = run(jobs[i]); }
            catch (const SetupError& e) { res = OpResult(); res.code = e.code ? e.code : EXIT_FAILED; res.message = e.msg; }
            ++out.ran;
            if (res.code) { failed = true; out.last = res; }
        }
        reply.Send(kRecResult, (uint32_t)(i + 1), ResultText(res));
    }
    guards.clear();
    reply.Send(kRecEnd, 0, "");
    return out;
}

}  // namespace wg
