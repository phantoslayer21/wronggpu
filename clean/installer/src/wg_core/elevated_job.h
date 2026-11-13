// SPDX-License-Identifier: MIT
// The launcher's elevated step ("WrongGPU.exe apply"): the launcher's side (RunElevatedJobs) and the worker's (RunWorker).
//
// The launcher writes one job file per operation (launcher.h JobText) and a batch file listing each job file with its
// SHA-256 (BatchText) into its jobs folder, creates a reply pipe with a random name that only this account and
// administrators can open (ReplyChannel), and starts the worker with runas:
//     WrongGPU.exe apply --batch=<folder>\batch-<stamp>.txt --sha256=<hex> --pipe=\\.\pipe\WrongGPU-reply-<hex>
// The worker
//   - runs nothing without the reply pipe;
//   - reads the batch file and every job file once into memory, checks each SHA-256 against the command line and the batch
//     and parses those same bytes;
//   - checks every job before the first one runs: every path absolute, and the OptiScaler package of an install, update or
//     repair copied into a new folder only administrators can write, the copy's digest the job's (package_snapshot.h); the
//     install uses that copy. A batch with one job that fails these checks runs no job at all;
//   - writes no log or result file: its log lines and each job's result go back over the pipe, and the launcher writes the
//     lines to its own log.
// The launcher takes the reply only from the worker process it started (the pipe client's process id and start time) and
// only from this run: a job without a result on the pipe did not report.
#pragma once

#include <functional>

#include "launcher.h"

namespace wg {

// ---- both sides ---------------------------------------------------------------------------------------------------
// A process as the launcher recognises the worker: its id and start time.
struct ProcessIdentity {
    DWORD pid = 0;
    unsigned long long started = 0;
    bool operator==(const ProcessIdentity& o) const { return pid && pid == o.pid && started == o.started; }
};
ProcessIdentity IdentityOf(HANDLE process);

// The pipe name the launcher makes: \\.\pipe\WrongGPU-reply-<32 lower-case hex digits>; the worker opens nothing else.
bool IsReplyPipeName(const wstring& name);

// ---- the launcher's side ------------------------------------------------------------------------------------------
// What the worker sent over the pipe in this run.
struct WorkerReply {
    bool connected = false;       // the worker connected
    bool complete = false;        // its last record arrived
    bool refused = false;         // it refused the request before running any job (refusal says why)
    OpResult refusal;
    std::vector<std::pair<bool, OpResult>> results;  // per job, in order: received, the result
    std::vector<wstring> log;     // its log lines
};

// The reply pipe: made before the permission prompt (Create), read while the worker runs, finished after it exited.
class ReplyChannel {
public:
    ReplyChannel();
    ~ReplyChannel();
    ReplyChannel(const ReplyChannel&) = delete;
    ReplyChannel& operator=(const ReplyChannel&) = delete;
    bool Create(wstring& why);
    const wstring& Name() const;
    // The reply of `worker` (nothing when it is the empty identity: the worker was not started) for `jobs` jobs.
    WorkerReply Finish(const ProcessIdentity& worker, size_t jobs);
private:
    struct Impl;
    Impl* impl_;
};

// Starts the worker with these arguments and waits until it exits; false: it was not started (permission not given).
// `worker`: the process started; `exitCode`: its exit code.
using StartWorkerFn = std::function<bool(const wstring& args, DWORD& exitCode, ProcessIdentity& worker)>;
using LogFn = std::function<void(const wstring&)>;

// The launcher's elevated step for these job texts (JobText), in order, in one worker run (one permission prompt). stop:
// none runs after the first failure (kOpNotRun). A result per job; a job the worker did not report on gets a failure that
// says so. The job and batch files are written to `jobsDir` and deleted afterwards; the worker's log lines go to `log`.
std::vector<OpResult> RunElevatedJobs(const std::vector<string>& jobTexts, const wstring& jobsDir, bool stop, const StartWorkerFn& start, const LogFn& log);

extern const wchar_t* const kElevationNotGranted;

// ---- the worker's side --------------------------------------------------------------------------------------------
// One checked job, as the worker runs it.
struct WorkerJob {
    OpRequest r;
    wstring localAppData, userSid, documents;
    int perGameEntry = -1;
};
// Runs one job's operation (the launcher: environment, registry hive, RunOp). The tests pass their own.
using WorkerRunFn = std::function<OpResult(const WorkerJob&)>;

struct WorkerArgs { wstring batch, sha256, pipe; };
struct WorkerRun {
    OpResult last;          // the last failure, or OK; the refusal when refused
    int jobs = 0, ran = 0;  // ran: the jobs run (not the ones stop=1 skipped)
    bool refused = false;   // no job was run
};

// "apply --batch=<file> --sha256=<hex> --pipe=<name>". ctx: its log lines go over the pipe while it runs, and its
// temporary files under the worker's private folder (made in privateParent; empty: the Windows folder's Temp).
WorkerRun RunWorker(const WorkerArgs& a, Context& ctx, const WorkerRunFn& run, const wstring& privateParent = L"");

}  // namespace wg
