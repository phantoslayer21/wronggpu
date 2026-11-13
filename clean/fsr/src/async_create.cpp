// SPDX-License-Identifier: MIT
#include "async_create.h"

#include <thread>

namespace wgfsr {

std::shared_ptr<PendingCreate> PendingCreate::Start(const WgUpscalerBackend* backend, const WgCreateInfo& info) {
    auto p = std::make_shared<PendingCreate>();
    p->backend_ = backend;
    p->info_ = info;
    p->start_ = std::chrono::steady_clock::now();
    std::thread([p] { p->Run(p); }).detach();
    return p;
}

void PendingCreate::Run(std::shared_ptr<PendingCreate> self) {
    WgUpscaler* upscaler = nullptr;
    const WgResult r = backend_->create(&info_, &upscaler);
    std::function<void(WgUpscaler*)> cleanup;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        result_ = r;
        end_ = std::chrono::steady_clock::now();
        upscaler_ = r == WG_OK ? upscaler : nullptr;
        done_ = true;
        cleanup = std::move(cleanup_);
        if (cleanup) taken_ = true;
    }
    cv_.notify_all();
    if (cleanup) cleanup(r == WG_OK ? upscaler : nullptr);  // the context went away while the create ran
}

bool PendingCreate::Wait(uint32_t ms) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (ms == UINT32_MAX) {
        cv_.wait(lock, [this] { return done_; });
        return true;
    }
    return cv_.wait_for(lock, std::chrono::milliseconds(ms), [this] { return done_; });
}

bool PendingCreate::Done() {
    std::lock_guard<std::mutex> lock(mutex_);
    return done_;
}

WgResult PendingCreate::Result() {
    std::lock_guard<std::mutex> lock(mutex_);
    return result_;
}

WgUpscaler* PendingCreate::Take() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!done_ || taken_) return nullptr;
    taken_ = true;
    return upscaler_;
}

double PendingCreate::Seconds() {
    std::lock_guard<std::mutex> lock(mutex_);
    return std::chrono::duration<double>((done_ ? end_ : std::chrono::steady_clock::now()) - start_).count();
}

bool PendingCreate::PassedThreshold(double seconds) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (done_ || reported_ >= seconds || std::chrono::steady_clock::now() - start_ < std::chrono::duration<double>(seconds)) return false;
    reported_ = seconds;
    return true;
}

void PendingCreate::Abandon(const std::shared_ptr<PendingCreate>& pending, std::function<void(WgUpscaler*)> cleanup) {
    WgUpscaler* upscaler = nullptr;
    {
        std::lock_guard<std::mutex> lock(pending->mutex_);
        if (!pending->done_) {
            pending->cleanup_ = std::move(cleanup);  // Run calls it
            return;
        }
        if (!pending->taken_) upscaler = pending->upscaler_;
        pending->taken_ = true;
    }
    cleanup(upscaler);
}

}  // namespace wgfsr
