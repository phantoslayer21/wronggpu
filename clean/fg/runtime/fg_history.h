// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// Frame generation (issue #96): title-neutral frame ingest, the history banks and the history epoch.
//
// FgFrameAssembler turns an adapter's tags (one texture per slot, each stamped with its frame token and viewport) into
// one FgFramePacket per game frame. It refuses a tag of another frame or viewport, an older (stale) tag, and a second
// different tag for the same slot of the same frame; at the end it refuses a frame without color, motion or depth, a
// frame without an optional input the session promised (declared Always, or a presence bit without the texture), and
// a frame carrying an optional input the session declared Never. A refused frame is closed: nothing of it lingers.
//
// FgHistory owns the two logical history roles, previous and current, over 2..4 physical banks:
// - only an accepted game frame (generated_index 0) writes and rotates history; a per-index evaluation (i = 1..N)
//   never does;
// - a bank still read by compute (marked with a fence value that has not completed) is never overwritten: when no
//   bank is free the frame is shown real-only (HistoryBusy) and the next frame primes again;
// - startup, reset / camera cut, a menu frame, a skipped frame token, an incompatible size or format change and a
//   device / queue / swap-chain replacement invalidate history (a new epoch); the first valid game frame after that
//   primes history and is shown real-only;
// - identity is exact: viewport, adapter LUID, queue and swap-chain epoch must match the session; tokens increase.
// Render-size changes inside the session's allocation are treated as incompatible (re-prime) until the history
// compatibility rule for them is qualified. Single caller (the present owner serialises it). Plain C++.
#include <cstdint>
#include <string>

#include "fg_contract.h"

namespace wgfg {

enum class FgTagSlot : uint32_t {
    Color = 0, Motion, Depth, Hudless, UiAlpha, Distortion, NoWarp,
};
inline constexpr uint32_t kFgTagSlots = 7;

class FgFrameAssembler {
public:
    explicit FgFrameAssembler(const FgCreateDesc& session) : session_(session) {}

    // Opens frame `token` (a still-open older frame is dropped). StaleTag for a token not above the last one.
    FgRefusal Begin(uint64_t token);
    FgRefusal Tag(FgTagSlot slot, const FgTextureRef& texture);
    // Builds the packet from `constants` (identity, rectangles, flags, preset, N, camera) and the tags; closes the
    // frame whether or not it was accepted.
    FgRefusal Finish(const FgFramePacket& constants, FgFramePacket* out);
    void Abort();
    bool open() const { return open_; }

private:
    FgCreateDesc session_;
    bool open_ = false;
    uint64_t token_ = 0;
    uint64_t last_token_ = 0;
    bool have_last_ = false;
    bool set_[kFgTagSlots] = {};
    FgTextureRef tags_[kFgTagSlots];
};

enum class FgBankRole : uint32_t { Free = 0, Current = 1, Previous = 2, Retired = 3 };

struct FgHistoryBank {
    FgBankRole role = FgBankRole::Free;
    uint64_t frame_token = 0;
    uint64_t epoch = 0;
    uint64_t read_fence = 0;  // 0: not read; else compute reads it until this fence value completes
    uint32_t color_format = 0;
    uint32_t output_width = 0, output_height = 0;
    FgRect render_rect;
    uint32_t writes = 0;      // how often this bank was written (tests)
};

struct FgIngestResult {
    FgFrameStatus status = FgFrameStatus::Refused;
    FgRefusal refusal = FgRefusal::None;
    FgFrameRealOnly real_only = FgFrameRealOnly::None;
    int previous_bank = -1, current_bank = -1;
    uint64_t epoch = 0;
    uint32_t generated_count = 0;  // N of the interval (0 unless Generated)
    uint32_t generated_index = 0;
    float t = 0;                   // Evaluate: t of index i
    FgPreset preset = FgPreset::A;
};

struct FgHistorySnapshot {
    uint32_t nbanks = 0;
    FgHistoryBank banks[4];
    int previous = -1, current = -1;
    uint64_t epoch = 0;
};
// The structural invariants of history: role bookkeeping, previous older than current, both of the current epoch, a
// retired bank only while compute still reads it. False with *why naming the first broken one.
bool FgCheckHistoryInvariants(const FgHistorySnapshot& s, std::string* why);

class FgHistory {
public:
    static constexpr uint32_t kMinBanks = 2, kMaxBanks = 4;
    // `banks` outside 2..4 is clamped into it.
    FgHistory(const FgCreateDesc& session, uint32_t banks);

    // A game frame (generated_index 0). completed_fence: the highest completed value of the compute fence.
    FgIngestResult Accept(const FgFramePacket& packet, uint64_t completed_fence);
    // An explicit evaluator's index i (1..N) of the interval Accept last opened. Never writes or rotates history.
    FgIngestResult Evaluate(const FgFramePacket& packet) const;
    // The current interval's banks (previous and current) are read by compute until `fence_value` completes.
    bool MarkRead(uint64_t fence_value);
    // Frees what completed: read marks at or below `completed_fence`, retired banks without a pending read.
    void Retire(uint64_t completed_fence);
    // The device / queue / swap chain was replaced, or a resize was forwarded: new identity, new epoch.
    void Rebind(uint64_t queue_epoch, uint64_t swapchain_epoch);
    void Invalidate();

    uint64_t epoch() const { return epoch_; }
    bool primed() const { return current_ >= 0; }
    bool interval_open() const { return previous_ >= 0 && current_ >= 0; }
    uint32_t banks() const { return nbanks_; }
    const FgHistoryBank& bank(uint32_t i) const { return banks_[i]; }
    int previous_bank() const { return previous_; }
    int current_bank() const { return current_; }
    uint64_t rotations() const { return rotations_; }
    uint64_t primes() const { return primes_; }
    uint64_t base_presents() const { return base_presents_; }
    FgHistorySnapshot Snapshot() const;

private:
    FgRefusal CheckIdentity(const FgFramePacket& p) const;
    FgRefusal CheckInputs(const FgFramePacket& p) const;
    void Drop(int bank);
    int FreeBank(uint64_t completed_fence, int keep) const;

    FgCreateDesc session_;
    uint32_t nbanks_;
    FgHistoryBank banks_[kMaxBanks];
    int previous_ = -1, current_ = -1;
    uint64_t epoch_ = 1;
    uint64_t queue_epoch_, swapchain_epoch_;
    uint64_t last_token_ = 0;
    bool have_token_ = false;
    uint32_t interval_n_ = 0;
    FgPreset interval_preset_ = FgPreset::A;
    uint64_t rotations_ = 0, primes_ = 0, base_presents_ = 0;
};

}  // namespace wgfg
