// SPDX-License-Identifier: MIT
#include "fg_worker.h"

#include <atomic>
#include <thread>

namespace wgfg {

namespace {
std::atomic<FgPendingInit::Launcher> g_launcher{nullptr};
}

struct FgPendingInit::Entry {
    std::shared_ptr<FgPendingInit> self;  // copying a shared_ptr cannot throw
    explicit Entry(const std::shared_ptr<FgPendingInit>& p) noexcept : self(p) {}
    void operator()() const noexcept { self->Run(); }
};

void FgPendingInit::SetLauncherForTest(Launcher launcher) noexcept { g_launcher.store(launcher); }

std::shared_ptr<FgPendingInit> FgPendingInit::Start(Work work, FgHold hold) noexcept {
    std::shared_ptr<FgPendingInit> p;
    try {
        p = std::make_shared<FgPendingInit>();
    } catch (...) {
        hold.Release();  // nothing started: give the reference back now
        return nullptr;
    }
    p->work_ = std::move(work);  // std::function's move does not allocate
    p->hold_ = std::move(hold);
    try {
        std::function<void()> entry{Entry(p)};
        const Launcher launcher = g_launcher.load();
        if (launcher) launcher(std::move(entry));
        else std::thread(std::move(entry)).detach();
    } catch (...) {  // the worker could not be started: the work never ran
        p->Finish(FgRefusal::WorkerStartFailed);
        p->hold_.Release();
    }
    return p;
}

void FgPendingInit::Run() noexcept {
    FgRefusal r = FgRefusal::WorkerInitFailed;
    try {
        r = work_ ? work_() : FgRefusal::WorkerInitFailed;
    } catch (...) {
        r = FgRefusal::WorkerInitFailed;
    }
    Finish(r);
    hold_.Release();  // the last step: nothing of this worker touches the game's device after it
}

void FgPendingInit::Finish(FgRefusal r) noexcept {
    Cleanup cleanup = nullptr;
    void* user = nullptr;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (done_) return;
        result_ = r;
        done_ = true;
        cleanup = cleanup_;
        user = cleanup_user_;
        cleanup_ = nullptr;
    }
    cv_.notify_all();
    if (cleanup) cleanup(user, r);  // the context went away while the work ran
}

bool FgPendingInit::Wait(uint32_t ms) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (ms == UINT32_MAX) {
        while (!done_) cv_.wait(lock);
        return true;
    }
    const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
    while (!done_)
        if (cv_.wait_until(lock, until) == std::cv_status::timeout) break;
    return done_;
}

bool FgPendingInit::Done() {
    std::lock_guard<std::mutex> lock(mutex_);
    return done_;
}

FgRefusal FgPendingInit::Result() {
    std::lock_guard<std::mutex> lock(mutex_);
    return done_ ? result_ : FgRefusal::WorkerAbandoned;
}

void FgPendingInit::Abandon(const std::shared_ptr<FgPendingInit>& pending, Cleanup cleanup, void* user) noexcept {
    if (!pending) {
        if (cleanup) cleanup(user, FgRefusal::WorkerStartFailed);
        return;
    }
    FgRefusal r;
    {
        std::lock_guard<std::mutex> lock(pending->mutex_);
        if (!pending->done_) {
            pending->cleanup_ = cleanup;  // Finish calls it on the worker
            pending->cleanup_user_ = user;
            return;
        }
        r = pending->result_;
    }
    if (cleanup) cleanup(user, r);
}

}  // namespace wgfg
