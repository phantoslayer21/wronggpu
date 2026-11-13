// SPDX-License-Identifier: MIT
#include "split_progress.h"
#include <cstdio>
#include <thread>
#include <chrono>

namespace wgwin {
struct SplitProgress::Entry {
    uint64_t id, ticket, doneValue, resumeValue, started;
    const void* list;
    ID3D12Device* device;
    ID3D12Fence *order, *done, *resume;
    std::atomic<const void*> queue{nullptr};
    std::atomic<bool> cpuReleased{false};
    bool reported = false; // Poll has one consumer, independently of the game's submission thread.
    uint64_t lastReport = 0;
    ~Entry() { order->Release(); done->Release(); resume->Release(); device->Release(); }
};

bool SplitProgress::Track(uint64_t id, const void* list, ID3D12Device* device, ID3D12Fence* order, uint64_t ticket,
                          ID3D12Fence* done, uint64_t doneValue, ID3D12Fence* resume, uint64_t resumeValue, uint64_t nowUs) noexcept {
    try {
    if (!device || !order || !done || !resume) return false;
    auto e = std::make_shared<Entry>();
    e->id=id; e->list=list; e->device=device; e->order=order; e->ticket=ticket;
    e->done=done; e->doneValue=doneValue; e->resume=resume; e->resumeValue=resumeValue; e->started=nowUs;
    device->AddRef(); order->AddRef(); done->AddRef(); resume->AddRef();
    std::lock_guard lock(mutex_);
    for(auto& slot:entries_) if(!slot) { slot=std::move(e); return true; }
    return false; // Bounded metadata retention; never stall the game to make room.
    } catch(...) { return false; } // Diagnostics cannot turn an already queued split into a frame failure.
}

void SplitProgress::Submitted(uint64_t id, const void* queue) {
    std::lock_guard lock(mutex_);
    for(auto& e:entries_) if(e && e->id==id) { e->queue.store(queue); break; }
}
void SplitProgress::CpuReleased(uint64_t id) {
    std::lock_guard lock(mutex_);
    for(auto& e:entries_) if(e && e->id==id) { e->cpuReleased.store(true); break; }
}
size_t SplitProgress::Count() {
    std::lock_guard lock(mutex_); size_t n=0;
    for(auto& e:entries_) if(e) ++n;
    return n;
}

void SplitProgress::Poll(uint64_t nowUs, LogFn log, void* user) {
    std::array<std::shared_ptr<Entry>,64> snapshot;
    { std::lock_guard lock(mutex_); snapshot=entries_; }
    for(auto& e:snapshot) if(e) {
        // COM/driver queries and logging are outside the metadata lock: a blocked driver query must not block Track.
        const uint64_t order=e->order->GetCompletedValue(), done=e->done->GetCompletedValue(), resume=e->resume->GetCompletedValue();
        const HRESULT reason=e->device->GetDeviceRemovedReason();
        const bool sentinel=order==UINT64_MAX || done==UINT64_MAX || resume==UINT64_MAX;
        const bool lost=FAILED(reason) || sentinel;
        const bool complete=!lost && done>=e->doneValue && resume>=e->resumeValue && order>=e->ticket;
        const uint64_t age=nowUs>=e->started ? nowUs-e->started : 0;
        if(lost || (complete && e->reported) || (!complete && age>=1000000 &&
            (!e->reported || (nowUs>=e->lastReport && nowUs-e->lastReport>=1000000)))) {
            char text[640];
            const char* stage=lost?"device-status":complete?"recovered":order<e->ticket?"order":done<e->doneValue?"first-half":"resume";
            std::snprintf(text,sizeof text,
                "[split-progress-qa] S%llu list=%p queue=%p stage=%s age_us=%llu order=%llu/%llu done=%llu/%llu resume=%llu/%llu cpu_release=%d device_hr=0x%08lx sentinel=%d",
                static_cast<unsigned long long>(e->id),e->list,e->queue.load(),stage,static_cast<unsigned long long>(age),
                static_cast<unsigned long long>(order),static_cast<unsigned long long>(e->ticket),
                static_cast<unsigned long long>(done),static_cast<unsigned long long>(e->doneValue),
                static_cast<unsigned long long>(resume),static_cast<unsigned long long>(e->resumeValue),
                e->cpuReleased.load()?1:0,static_cast<unsigned long>(reason),sentinel?1:0);
            if(log) log(user,text);
            e->reported=true; e->lastReport=nowUs;
        }
        if(complete || lost) {
            std::shared_ptr<Entry> retired;
            { std::lock_guard lock(mutex_); for(auto& slot:entries_) if(slot==e) { retired=std::move(slot); break; } }
            // Final COM Release occurs outside the metadata lock as well.
        }
    }
}

SplitProgress& ProcessSplitProgress() { static auto* trace=new SplitProgress; return *trace; }
bool StartSplitProgressWorker(SplitProgress::LogFn log, void* user, uint64_t (*nowUs)()) {
    try {
        auto* trace=&ProcessSplitProgress();
        std::thread([log,user,nowUs,trace] {
            for(;;) {
                try { trace->Poll(nowUs(),log,user); }
                catch(...) {
                    try { if(log) log(user,"[split-progress-qa] diagnostic worker stopped after a polling/logging exception"); } catch(...) {}
                    return;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
            }
        }).detach();
        return true;
    } catch(...) { return false; }
}
}  // namespace wgwin
