// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// Frame generation (issue #96): the session's create-time work (model check, weight packing, pipeline warm-up) on a
// worker thread with a bounded wait, following PendingCreate in clean/fsr/src/async_create.h.
//
// No lambda here or in its callers captures anything whose copy can throw: with MSVC's /Zc:lambda (on whenever
// /permissive- is, as in CI) such a capture terminates the process instead of unwinding. The work is a named callable
// that copies what it needs in its constructor (FgCopiedJob, FgModelJob), the worker entry holds only a shared_ptr,
// and the hold on the game's device is a function pointer plus a raw pointer (FgHold), which cannot throw at all.
// Every failure becomes a refusal: the worker could not start (WorkerStartFailed, the hold given back at once), the
// work threw (WorkerInitFailed), or the context went away first (WorkerAbandoned; the cleanup runs when the work ends).
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>

#include "fg_contract.h"
#include "fg_model.h"

namespace wgfg {

// One reference to give back (the worker's COM reference on the game's device): released exactly once. Move-only,
// noexcept throughout.
class FgHold {
public:
    using ReleaseFn = void (*)(void*);
    FgHold() noexcept = default;
    FgHold(ReleaseFn fn, void* object) noexcept : fn_(fn), object_(object) {}
    FgHold(FgHold&& o) noexcept : fn_(o.fn_), object_(o.object_) { o.fn_ = nullptr; }
    FgHold& operator=(FgHold&& o) noexcept {
        if (this != &o) {
            Release();
            fn_ = o.fn_;
            object_ = o.object_;
            o.fn_ = nullptr;
        }
        return *this;
    }
    FgHold(const FgHold&) = delete;
    FgHold& operator=(const FgHold&) = delete;
    ~FgHold() { Release(); }
    void Release() noexcept {
        ReleaseFn f = fn_;
        fn_ = nullptr;
        if (f) f(object_);
    }
    explicit operator bool() const noexcept { return fn_ != nullptr; }

private:
    ReleaseFn fn_ = nullptr;
    void* object_ = nullptr;
};

// A job that owns a copy of its callable, made in the constructor (a throwing copy unwinds normally there).
template <class F>
class FgCopiedJob {
public:
    explicit FgCopiedJob(const F& f) : f_(f) {}
    FgRefusal operator()() const { return f_(); }

private:
    F f_;
};

// The model check as a job: copies the path in its constructor.
class FgModelJob {
public:
    FgModelJob(const std::string& path, const FgModelExpectation& expect) : path_(path), expect_(expect) {}
    FgRefusal operator()() const {
        FgModelReport report;
        return FgValidateModelFile(path_, expect_, &report, nullptr);
    }

private:
    std::string path_;
    FgModelExpectation expect_;
};

class FgPendingInit {
public:
    using Work = std::function<FgRefusal()>;
    using Cleanup = void (*)(void* user, FgRefusal result);
    using Launcher = void (*)(std::function<void()> work);

    // Starts `work` on a new thread. Never throws: when the shared state or the thread cannot be made, the result is
    // a finished FgPendingInit with WorkerStartFailed (or null when even that allocation failed) and `hold` is
    // released before Start returns. Otherwise the worker releases `hold` as its last step.
    static std::shared_ptr<FgPendingInit> Start(Work work, FgHold hold) noexcept;
    // Tests: how the worker is launched (default: a detached std::thread). A launcher that throws is a worker that
    // could not be started. Null restores the default.
    static void SetLauncherForTest(Launcher launcher) noexcept;

    bool Wait(uint32_t ms);  // UINT32_MAX: until done; true once finished
    bool Done();
    FgRefusal Result();      // valid after Done()

    // The context goes away: `cleanup(user, result)` runs now if the work finished, else on the worker when it does.
    static void Abandon(const std::shared_ptr<FgPendingInit>& pending, Cleanup cleanup, void* user) noexcept;

    FgPendingInit() = default;

private:
    struct Entry;  // the worker's named entry callable (holds a shared_ptr only)
    void Run() noexcept;
    void Finish(FgRefusal r) noexcept;

    Work work_;
    FgHold hold_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool done_ = false;
    FgRefusal result_ = FgRefusal::WorkerStartFailed;
    Cleanup cleanup_ = nullptr;
    void* cleanup_user_ = nullptr;
};

}  // namespace wgfg
