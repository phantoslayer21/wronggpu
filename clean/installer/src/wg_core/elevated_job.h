// SPDX-License-Identifier: MIT
// The elevated worker's side of "WrongGPU.exe apply": it runs only the job and batch files of the launcher's own request.
//
// The launcher writes the job files (launcher.h JobText) and, for several jobs, a batch file listing each job file with
// its SHA-256 (BatchText), and passes the SHA-256 of the file it names on the elevation command line:
//     apply --batch=<folder>\batch-<stamp>.txt --sha256=<hex>
//     apply --job=<folder>\job-<name>.txt --sha256=<hex>
// The worker reads each file once into memory, checks its SHA-256 and parses those same bytes. It works out every result
// path itself, in the real folder of the file it was given: result-<stamp>-<n>.txt for the n-th job of a batch,
// result-<name>.txt for a single job. That folder must not be a reparse point; it is held open (no delete sharing) while
// the worker runs, and each result is written to a new temporary file there and then moved over the result name, never
// through an existing link. A job's log= must be a file under the job's localappdata folder or under <data>\logs (the
// folder next to the jobs folder), checked the same way and held open while the job runs.
#pragma once

#include <functional>

#include "launcher.h"

namespace wg {

// One parsed job, as the worker runs it.
struct WorkerJob {
    OpRequest r;
    wstring localAppData, logFile, userSid, documents;
    int perGameEntry = -1;
};
// Runs one job's operation (the launcher: environment, registry hive, RunOp). The tests pass their own.
using WorkerRunFn = std::function<OpResult(const WorkerJob&)>;

struct WorkerRun {
    OpResult last;     // the batch's result: the last failure, or OK; for a single job, its result
    int jobs = 0, ran = 0;  // ran: the jobs taken up, refused or run (not the ones stop=1 skipped)
    bool refused = false;  // nothing was run and no result was written (the file given could not be verified)
};

// "apply --batch=<batch> --sha256=<hex>" (batch not empty) or "apply --job=<job> --sha256=<hex>".
WorkerRun RunWorkerFiles(const wstring& batch, const wstring& job, const wstring& sha256, const WorkerRunFn& run);

// The result file of the n-th (1-based) job of a batch, and of a single job, next to the file named on the command line
// (the launcher reads its results from these names; empty when the name is not batch-<stamp>.txt / job-<name>.txt).
wstring BatchResultPath(const wstring& batch, size_t n);
wstring JobResultPath(const wstring& job);

}  // namespace wg
