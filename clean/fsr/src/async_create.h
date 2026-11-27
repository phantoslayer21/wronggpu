// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// A WrongGPU upscaler's create on a worker thread, so that a game thread waits for it only a bounded time.
//
// The network runtime's create starts HIP (its first stream starts HIP's worker threads). In some games that start
// does not finish while the game's thread that called the create keeps waiting for it (Borderlands 4: the create
// blocked in hipStreamCreateWithFlags and the game froze). With a bounded wait the game's call returns: the front end
// upscales with AMD's FSR meanwhile and takes the upscaler over once its create finishes. Plain C++, no Windows code.
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>

#include "../include/wronggpu_upscaler.h"

namespace wgfsr {

// Owns one thing to give back (the create worker's COM reference on the game's device, taken by the front end before the
// worker starts): released exactly once, by Release() or, if nobody did, when the owner goes away. So the reference is
// given back when the worker finishes, when the worker could not be started at all (Start), and when anything between taking
// the reference and starting the worker throws. Move-only.
class ScopedRelease {
public:
    ScopedRelease() = default;
    explicit ScopedRelease(std::function<void()> release) : release_(std::move(release)) {}
    ScopedRelease(ScopedRelease&& o) noexcept : release_(std::move(o.release_)) { o.release_ = nullptr; }
    ScopedRelease& operator=(ScopedRelease&& o) noexcept {
        if (this != &o) {
            Release();
            release_ = std::move(o.release_);
            o.release_ = nullptr;
        }
        return *this;
    }
    ScopedRelease(const ScopedRelease&) = delete;
    ScopedRelease& operator=(const ScopedRelease&) = delete;
    ~ScopedRelease() { Release(); }
    void Release() {
        std::function<void()> f = std::move(release_);
        release_ = nullptr;
        if (f) f();
    }
    explicit operator bool() const { return static_cast<bool>(release_); }

private:
    std::function<void()> release_;
};

// Takes the create worker's hold on the game's device: the owner is built FIRST (its callable is the only allocation), the
// reference is retained AFTER it, and the owner is returned by a noexcept move: nothing that can throw lies between the
// retain and the owner, so a throwing allocation never leaks the reference (it throws before anything was retained).
// The owner's callable is a named class, not a lambda that captures `release` by copy: with MSVC's /Zc:lambda (on whenever
// /permissive- is, as in the CI builds) a lambda whose capture copy throws terminates the process instead of unwinding (a
// defect in the closure's unwind state; reproduced with a callable whose copy throws, and std::bad_alloc from copying a large
// callable is the same case), while the same copy inside a class constructor unwinds normally.
struct DeviceRelease {
    std::function<void(void*)> release;
    void* device;
    DeviceRelease(const std::function<void(void*)>& r, void* d) : release(r), device(d) {}
    void operator()() const { release(device); }
};

inline ScopedRelease AcquireHold(void* device, const std::function<void(void*)>& retain, const std::function<void(void*)>& release) {
    ScopedRelease hold{std::function<void()>(DeviceRelease(release, device))};
    retain(device);
    return hold;
}

class PendingCreate {
public:
    // Starts backend->create(info) on a new thread. `info` and what it points to (options, log user) must stay valid
    // until the create finished, or until the cleanup given to Abandon ran. `hold`: released by the worker as its last step,
    // also after a timeout and after the context went away; released at once when the worker cannot be started (the create
    // then counts as failed: WG_ERROR_DEVICE, no exception reaches the game's thread) or when anything before it throws (the
    // front end takes a COM reference on the game's D3D12 device before the worker starts and gives it back through this, so a
    // late worker never meets a released device and a worker that never ran never leaks it).
    static std::shared_ptr<PendingCreate> Start(const WgUpscalerBackend* backend, const WgCreateInfo& info,
                                                ScopedRelease hold = {});
    // Tests: how the worker thread is launched (default: a detached std::thread). A launcher that throws is a worker that
    // could not be started. Null restores the default.
    using Launcher = void (*)(std::function<void()> work);
    static void SetLauncherForTest(Launcher launcher);

    // Waits up to `ms` milliseconds (UINT32_MAX: until it finishes); true when the create has finished.
    bool Wait(uint32_t ms);
    bool Done();
    // After Done(): the create's result and its upscaler (null unless WG_OK). Take() hands the upscaler over once.
    WgResult Result();
    WgUpscaler* Take();
    double Seconds();  // from Start until the create finished (until now while it runs)
    // Logs from a context's dispatch while the create runs ("still starting after 10 s"): true once per threshold.
    bool PassedThreshold(double seconds);

    // The context goes away. `cleanup` (destroys the upscaler if there is one, then frees what the create info points
    // to) runs now when the create has finished, else on the worker thread when it does.
    static void Abandon(const std::shared_ptr<PendingCreate>& pending, std::function<void(WgUpscaler*)> cleanup);

private:
    void Run(std::shared_ptr<PendingCreate> self);

    const WgUpscalerBackend* backend_ = nullptr;
    WgCreateInfo info_{};
    std::mutex mutex_;
    std::condition_variable cv_;
    bool done_ = false;
    bool taken_ = false;
    WgResult result_ = WG_ERROR_DEVICE;
    WgUpscaler* upscaler_ = nullptr;
    std::function<void(WgUpscaler*)> cleanup_;  // set by Abandon while the create runs
    ScopedRelease release_;                     // the worker's last step (Start's hold)
    double reported_ = 0;                        // highest threshold PassedThreshold answered true for
    std::chrono::steady_clock::time_point start_, end_;
};

}  // namespace wgfsr
