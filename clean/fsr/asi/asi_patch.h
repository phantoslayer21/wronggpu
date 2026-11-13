// SPDX-License-Identifier: MIT
#pragma once
// Original WrongGPU bounded call-patch transaction. Synthetic-memory testable; no host discovery.
// This supplies failure rollback, NOT exclusion or atomicity with respect to executing foreign threads.
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>

namespace wgasi {
class CallPatchTransaction {
public:
    static constexpr size_t kCapacity = 64;
    struct Patch {
        uint32_t offset = 0;
        std::array<uint8_t, 5> original{};
        std::array<uint8_t, 4> replacement{};
        std::array<uint8_t, 5> owned{};
        bool ownershipKnown = true;
    };
    // Exact receipt of stores completed by this writer, never a snapshot of arbitrary observed bytes.
    // Bits 0..3 correspond to the four requested operand bytes. Complete additionally requires all
    // four stores plus the writer's protection/cache obligations. Partial failures report exact bits.
    struct WriteResult { bool complete; uint8_t writtenMask; };
    enum class State { Rejected, Committed, RolledBack, RollbackFailed };
    struct Result {
        State state = State::Rejected;
        size_t attempted = 0;
        bool committed() const { return state == State::Committed; }
    };

    CallPatchTransaction(uint8_t* image, size_t size) : image_(image), size_(size) {}
    size_t count() const { return count_; }
    const Patch& patch(size_t index) const { return patches_[index]; }

    // No writes or dynamic allocations. A failed addition poisons the entire plan.
    bool Add(uint32_t offset, uint32_t expectedTarget, uintptr_t stub) {
        if (!valid_ || used_ || !image_ || count_ == kCapacity ||
            size_ < 5 || offset > size_ - 5 || expectedTarget >= size_)
            return valid_ = false;
        for (size_t i = 0; i < count_; ++i)
            if (uint64_t(offset) < uint64_t(patches_[i].offset) + 5 &&
                uint64_t(patches_[i].offset) < uint64_t(offset) + 5) return valid_ = false;
        const uint8_t* site = image_ + offset;
        int32_t originalRel = 0;
        std::memcpy(&originalRel, site + 1, 4);
        if (site[0] != 0xE8 || int64_t(offset) + 5 + originalRel != expectedTarget)
            return valid_ = false;
        const uintptr_t next = reinterpret_cast<uintptr_t>(site + 5);
        int32_t rel = 0;
        if (stub >= next) {
            if (stub - next > uint64_t(INT32_MAX)) return valid_ = false;
            rel = static_cast<int32_t>(stub - next);
        } else {
            if (next - stub > uint64_t(INT32_MAX) + 1) return valid_ = false;
            rel = static_cast<int32_t>(-static_cast<int64_t>(next - stub));
        }
        Patch& p = patches_[count_++];
        p.offset = offset;
        std::memcpy(p.original.data(), site, 5);
        std::memcpy(p.replacement.data(), &rel, 4);
        p.owned = p.original;
        return true;
    }

    // Writer(index, operand bytes, expected full site) must be noexcept and recheck expected bytes
    // immediately before its stores. Receipt plus original bytes defines the only restorable state.
    // Mismatches are preserved; read-check-write still is NOT atomic against external execution/writes.
    template<class Writer> Result Apply(Writer&& writer) {
        static_assert(noexcept(writer(size_t{}, static_cast<const uint8_t*>(nullptr), static_cast<const uint8_t*>(nullptr))),
                      "Patch writer must not throw after publication");
        if (used_) return {};
        used_ = true;
        if (!valid_ || !count_) return {};
        for (size_t i = 0; i < count_; ++i)
            if (!Original(i)) return {};
        for (size_t i = 0; i < count_; ++i) {
            // A prior writer may have allowed a foreign change. Never overwrite a newly changed site.
            if (!Original(i)) return Rollback(writer, i);
            Patch& p = patches_[i];
            const WriteResult receipt = writer(i, p.replacement.data(), p.original.data());
            p.ownershipKnown = (receipt.writtenMask & 0xf0) == 0;
            for (size_t byte = 0; byte < 4; ++byte)
                if (receipt.writtenMask & (1u << byte)) p.owned[byte + 1] = p.replacement[byte];
            if (!receipt.complete || receipt.writtenMask != 0x0f || image_[p.offset] != 0xE8 ||
                std::memcmp(image_ + patches_[i].offset + 1, patches_[i].replacement.data(), 4))
                return Rollback(writer, i + 1);
        }
        for (size_t i = 0; i < count_; ++i)
            if (image_[patches_[i].offset] != 0xE8 ||
                std::memcmp(image_ + patches_[i].offset + 1, patches_[i].replacement.data(), 4))
                return Rollback(writer, count_);
        return {State::Committed, count_};
    }

private:
    bool Original(size_t i) const {
        return std::memcmp(image_ + patches_[i].offset, patches_[i].original.data(), 5) == 0;
    }
    template<class Writer> Result Rollback(Writer& writer, size_t attempted) {
        bool restored = true;
        for (size_t n = attempted; n > 0; --n) {
            const size_t i = n - 1;
            const Patch& p = patches_[i];
            // Never overwrite an unexpected opcode/operand, even if it resembles another partial
            // patch. Only this writer's explicit receipt can establish the expected owned state.
            if (!p.ownershipKnown || std::memcmp(image_ + p.offset, p.owned.data(), 5)) {
                restored = false;
                continue;
            }
            const WriteResult written = writer(i, p.original.data() + 1, p.owned.data());
            if (!written.complete || written.writtenMask != 0x0f || !Original(i)) restored = false;
        }
        // A later restore must not silently disturb an earlier restore.
        for (size_t i = 0; i < attempted; ++i) if (!Original(i)) restored = false;
        return {restored ? State::RolledBack : State::RollbackFailed, attempted};
    }
    uint8_t* image_;
    size_t size_;
    std::array<Patch, kCapacity> patches_{};
    size_t count_ = 0;
    bool valid_ = true, used_ = false;
};
} // namespace wgasi
