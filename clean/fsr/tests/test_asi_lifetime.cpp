// SPDX-License-Identifier: MIT
// Original synthetic lifetime-publication tests: no module loading, host execution or GPU work.
#include "../asi/asi_lifetime.h"
#include <cstdio>

static int checks = 0, failures = 0;
#define CHECK(x) do { ++checks; if (!(x)) { ++failures; std::printf("FAIL %d: %s\n", __LINE__, #x); } } while (0)
using Tx = wgasi::CallPatchTransaction;

struct Fixture {
    std::array<uint8_t, 1024> memory{};
    std::array<uint8_t, 1024> before{};
    Tx transaction{memory.data(), memory.size()};
    explicit Fixture(size_t count) {
        memory.fill(0x90);
        for (size_t i = 0; i < count; ++i) {
            const uint32_t offset = static_cast<uint32_t>(8 + 8 * i);
            memory[offset] = 0xe8;
            const int32_t rel = 800 - static_cast<int32_t>(offset + 5);
            std::memcpy(memory.data() + offset + 1, &rel, 4);
            CHECK(transaction.Add(offset, 800, reinterpret_cast<uintptr_t>(memory.data()) + 900));
        }
        before = memory;
    }
};

static void TestPinFailurePreventsEveryWrite() {
    for (size_t count : {size_t(1), size_t(4), Tx::kCapacity}) {
        Fixture f(count);
        size_t pins = 0, writes = 0;
        auto pin = [&]() noexcept { ++pins; return false; };
        auto writer = [&](size_t i, const uint8_t* bytes, const uint8_t*) noexcept -> Tx::WriteResult {
            ++writes;
            std::memcpy(f.memory.data() + f.transaction.patch(i).offset + 1, bytes, 4);
            return {true, 0x0f};
        };
        const auto result = wgasi::PublishPinnedCalls(f.transaction, pin, writer);
        CHECK(!result.modulePinned && !result.patch.committed());
        CHECK(result.patch.state == Tx::State::Rejected && result.patch.attempted == 0);
        CHECK(pins == 1 && writes == 0 && f.memory == f.before);
    }
}

static void TestPinBeforePublicationAndRollback() {
    for (size_t count : {size_t(1), size_t(4), Tx::kCapacity})
    // Each possible forward-write failure plus success. Every rollback writer also needs pinning.
    for (size_t failAt = 0; failAt <= count; ++failAt) {
        Fixture f(count);
        bool pinned = false;
        size_t pins = 0, writes = 0;
        auto pin = [&]() noexcept {
            CHECK(writes == 0 && f.memory == f.before);
            ++pins;
            pinned = true;
            return true;
        };
        auto writer = [&](size_t i, const uint8_t* bytes, const uint8_t* expected) noexcept -> Tx::WriteResult {
            CHECK(pinned && pins == 1);
            uint8_t* site = f.memory.data() + f.transaction.patch(i).offset;
            CHECK(std::memcmp(site, expected, 5) == 0);
            const bool failed = writes++ == failAt;
            std::memcpy(site + 1, bytes, 4);
            return {!failed, 0x0f};
        };
        const auto result = wgasi::PublishPinnedCalls(f.transaction, pin, writer);
        CHECK(result.modulePinned && pinned && pins == 1);
        if (failAt == count) {
            CHECK(result.patch.committed() && writes == count);
            CHECK(result.patch.attempted == count);
            for (size_t i = 0; i < count; ++i)
                CHECK(std::memcmp(f.memory.data() + f.transaction.patch(i).offset + 1,
                                  f.transaction.patch(i).replacement.data(), 4) == 0);
        } else {
            CHECK(result.patch.state == Tx::State::RolledBack);
            CHECK(result.patch.attempted == failAt + 1 && writes == 2 * (failAt + 1));
            CHECK(f.memory == f.before);
        }
    }
}

static void TestRevalidationAfterPin() {
    Fixture f(4);
    size_t pins = 0, writes = 0;
    auto pin = [&]() noexcept {
        ++pins;
        // Another writer changes a planned site during the pin API's scheduling window.
        f.memory[f.transaction.patch(3).offset + 1] ^= 1;
        return true;
    };
    auto writer = [&](size_t, const uint8_t*, const uint8_t*) noexcept -> Tx::WriteResult {
        ++writes;
        return {true, 0x0f};
    };
    const auto result = wgasi::PublishPinnedCalls(f.transaction, pin, writer);
    CHECK(result.modulePinned && result.patch.state == Tx::State::Rejected);
    CHECK(pins == 1 && writes == 0 && result.patch.attempted == 0);
    auto expected = f.before;
    expected[f.transaction.patch(3).offset + 1] ^= 1;
    CHECK(f.memory == expected);
}

int main() {
    TestPinFailurePreventsEveryWrite();
    TestPinBeforePublicationAndRollback();
    TestRevalidationAfterPin();
    std::printf("ASI lifetime: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
