// SPDX-License-Identifier: MIT
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

class PendingCreate {
public:
    // Starts backend->create(info) on a new thread. `info` and what it points to (options, log user) must stay valid
    // until the create finished, or until the cleanup given to Abandon ran.
    static std::shared_ptr<PendingCreate> Start(const WgUpscalerBackend* backend, const WgCreateInfo& info);

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
    double reported_ = 0;                        // highest threshold PassedThreshold answered true for
    std::chrono::steady_clock::time_point start_, end_;
};

}  // namespace wgfsr
