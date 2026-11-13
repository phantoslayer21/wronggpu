// SPDX-License-Identifier: MIT
// Original synthetic-memory transaction tests. No foreign images, host API calls, or executable patches.
#include "../asi/asi_patch.h"
#include <cstdio>

static int checks = 0, failures = 0;
#define CHECK(x) do { ++checks; if (!(x)) { ++failures; std::printf("FAIL %d: %s\n", __LINE__, #x); } } while (0)
using Tx = wgasi::CallPatchTransaction;
struct Fixture {
    std::array<uint8_t, 1024> memory{};
    std::array<uint8_t, 1024> before{};
    Tx tx{memory.data(), memory.size()};
    explicit Fixture(size_t count = 4) {
        memory.fill(0x90);
        for (size_t i = 0; i < count; ++i) {
            const size_t site = 8 * i + 8;
            memory[site] = 0xe8;
            const int32_t rel = 800 - static_cast<int32_t>(site + 5);
            std::memcpy(memory.data() + site + 1, &rel, 4);
            CHECK(tx.Add(static_cast<uint32_t>(site), 800,
                         reinterpret_cast<uintptr_t>(memory.data()) + 900));
        }
        before = memory;
    }
};

static void TestSuccessAndRevalidation() {
    Fixture f;
    size_t calls = 0;
    auto writer = [&](size_t i, const uint8_t* bytes, const uint8_t* expected) noexcept -> Tx::WriteResult {
        CHECK(std::memcmp(f.memory.data() + f.tx.patch(i).offset, expected, 5) == 0);
        ++calls; std::memcpy(f.memory.data() + f.tx.patch(i).offset + 1, bytes, 4); return {true, 0x0f};
    };
    auto result = f.tx.Apply(writer);
    CHECK(result.committed() && calls == 4 && result.attempted == 4);
    for (size_t i = 0; i < 4; ++i)
        CHECK(std::memcmp(f.memory.data() + f.tx.patch(i).offset + 1, f.tx.patch(i).replacement.data(), 4) == 0);
    CHECK(f.tx.Apply(writer).state == Tx::State::Rejected && calls == 4);
    Fixture changed;
    changed.memory[changed.tx.patch(3).offset + 1] ^= 1;
    const auto external = changed.memory;
    auto never = [&](size_t, const uint8_t*, const uint8_t*) noexcept -> Tx::WriteResult { ++calls; return {true, 0x0f}; };
    CHECK(changed.tx.Apply(never).state == Tx::State::Rejected);
    CHECK(calls == 4 && changed.memory == external);
    Fixture later;
    size_t laterCalls = 0;
    auto concurrent = [&](size_t i, const uint8_t* bytes, const uint8_t* expected) noexcept -> Tx::WriteResult {
        CHECK(std::memcmp(later.memory.data() + later.tx.patch(i).offset, expected, 5) == 0);
        ++laterCalls;
        std::memcpy(later.memory.data() + later.tx.patch(i).offset + 1, bytes, 4);
        if (laterCalls == 1) later.memory[later.tx.patch(1).offset + 1] ^= 1;
        return {true, 0x0f};
    };
    result = later.tx.Apply(concurrent);
    CHECK(result.state == Tx::State::RolledBack && !result.committed() && result.attempted == 1);
    CHECK(laterCalls == 2);
    CHECK(std::memcmp(later.memory.data() + 8, later.before.data() + 8, 5) == 0);
    CHECK(later.memory[later.tx.patch(1).offset + 1] != later.before[later.tx.patch(1).offset + 1]);
    Fixture disturbed;
    size_t disturbedCalls = 0;
    auto disturbEarlier = [&](size_t i, const uint8_t* bytes, const uint8_t* expected) noexcept -> Tx::WriteResult {
        CHECK(std::memcmp(disturbed.memory.data() + disturbed.tx.patch(i).offset, expected, 5) == 0);
        ++disturbedCalls;
        std::memcpy(disturbed.memory.data() + disturbed.tx.patch(i).offset + 1, bytes, 4);
        if (disturbedCalls == 4) disturbed.memory[disturbed.tx.patch(0).offset + 1] ^= 1;
        return {true, 0x0f};
    };
    result = disturbed.tx.Apply(disturbEarlier);
    CHECK(result.state == Tx::State::RollbackFailed && !result.committed());
    CHECK(disturbedCalls == 7); // Conflicted site 0 is never handed to the rollback writer.
    CHECK(disturbed.memory[disturbed.tx.patch(0).offset + 1] ==
          uint8_t(disturbed.tx.patch(0).replacement[0] ^ 1));
    for (size_t i = 1; i < 4; ++i)
        CHECK(std::memcmp(disturbed.memory.data() + disturbed.tx.patch(i).offset,
                          disturbed.before.data() + disturbed.tx.patch(i).offset, 5) == 0);
}

static void TestFailures() {
    // Inject failure at EVERY write, including first/last (the last stands in for optional preview).
    // Failure before modification, after a partial write, after a full write, and false success.
    for (size_t count : {size_t(1), size_t(4), Tx::kCapacity})
    for (size_t failure = 0; failure < count; ++failure)
    for (int kind = 0; kind < 4; ++kind) {
        Fixture f(count);
        size_t calls = 0;
        auto writer = [&](size_t i, const uint8_t* bytes, const uint8_t* expected) noexcept -> Tx::WriteResult {
            CHECK(std::memcmp(f.memory.data() + f.tx.patch(i).offset, expected, 5) == 0);
            const bool fail = calls++ == failure;
            const size_t length = fail ? (kind == 0 || kind == 3 ? 0 : kind == 1 ? 2 : 4) : 4;
            std::memcpy(f.memory.data() + f.tx.patch(i).offset + 1, bytes, length);
            return {!fail || kind == 3, static_cast<uint8_t>((1u << length) - 1)};
        };
        const auto result = f.tx.Apply(writer);
        CHECK(result.state == Tx::State::RolledBack && !result.committed());
        CHECK(result.attempted == failure + 1 && calls == 2 * (failure + 1));
        CHECK(f.memory == f.before);
    }
    // Every rollback position can fail, either leaving altered bytes or failing its cache/protection
    // obligation despite restoring bytes. Both MUST report RollbackFailed, never clean rollback.
    for (size_t failure = 0; failure < 4; ++failure)
    for (size_t rollbackFailure = 0; rollbackFailure <= failure; ++rollbackFailure)
    for (int restoreBytes = 0; restoreBytes < 2; ++restoreBytes) {
        Fixture f;
        size_t calls = 0;
        auto writer = [&](size_t i, const uint8_t* bytes, const uint8_t* expected) noexcept -> Tx::WriteResult {
            CHECK(std::memcmp(f.memory.data() + f.tx.patch(i).offset, expected, 5) == 0);
            const size_t call = calls++;
            const bool rollbackBad = call > failure && i == rollbackFailure;
            if (!rollbackBad || restoreBytes) std::memcpy(f.memory.data() + f.tx.patch(i).offset + 1, bytes, 4);
            return {call != failure && !rollbackBad, static_cast<uint8_t>(!rollbackBad || restoreBytes ? 0x0f : 0)};
        };
        const auto result = f.tx.Apply(writer);
        CHECK(result.state == Tx::State::RollbackFailed && !result.committed());
        CHECK(calls == 2 * (failure + 1)); // Still attempts every remaining restoration.
        CHECK((f.memory == f.before) == bool(restoreBytes));
    }
}

static void TestOwnershipConflicts() {
    // Foreign edit of every byte (including opcode), at every attempted site. Preserve it exactly,
    // continue restoring other owned sites, and never call the writer for the conflicted site.
    for (size_t conflictSite = 0; conflictSite < 4; ++conflictSite)
    for (size_t conflictByte = 0; conflictByte < 5; ++conflictByte) {
        Fixture f;
        size_t calls = 0;
        uint8_t foreign = 0;
        auto writer = [&](size_t i, const uint8_t* bytes, const uint8_t* expected) noexcept -> Tx::WriteResult {
            CHECK(std::memcmp(f.memory.data() + f.tx.patch(i).offset, expected, 5) == 0);
            const size_t call = calls++;
            CHECK(call < 4 || i != conflictSite);
            std::memcpy(f.memory.data() + f.tx.patch(i).offset + 1, bytes, 4);
            if (call == 3) {
                auto& changed = f.memory[f.tx.patch(conflictSite).offset + conflictByte];
                changed ^= 0x5a;
                foreign = changed;
            }
            return {call != 3, 0x0f};
        };
        const auto result = f.tx.Apply(writer);
        CHECK(result.state == Tx::State::RollbackFailed && !result.committed() && calls == 7);
        CHECK(f.memory[f.tx.patch(conflictSite).offset + conflictByte] == foreign);
        for (size_t i = 0; i < 4; ++i)
            if (i != conflictSite) CHECK(std::memcmp(f.memory.data() + f.tx.patch(i).offset,
                                                     f.before.data() + f.tx.patch(i).offset, 5) == 0);
    }
    // A partially failing writer owns only the bytes it reports, never arbitrary bytes observed
    // after the call. Exercise every sparse partial mask and a foreign change of each site byte.
    for (unsigned mask = 0; mask < 16; ++mask)
    for (size_t foreignByte = 0; foreignByte < 6; ++foreignByte) {
        Fixture f(1);
        size_t calls = 0;
        uint8_t foreign = 0;
        auto writer = [&](size_t i, const uint8_t* bytes, const uint8_t* expected) noexcept -> Tx::WriteResult {
            CHECK(std::memcmp(f.memory.data() + f.tx.patch(i).offset, expected, 5) == 0);
            const bool first = calls++ == 0;
            const unsigned ownMask = first ? mask : 15;
            for (size_t byte = 0; byte < 4; ++byte)
                if (ownMask & (1u << byte)) f.memory[f.tx.patch(i).offset + 1 + byte] = bytes[byte];
            if (first && foreignByte < 5) {
                auto& changed = f.memory[f.tx.patch(i).offset + foreignByte];
                changed ^= 0x5a;
                foreign = changed;
            }
            return {!first, static_cast<uint8_t>(ownMask)};
        };
        const auto result = f.tx.Apply(writer);
        CHECK(!result.committed());
        if (foreignByte < 5) {
            CHECK(result.state == Tx::State::RollbackFailed && calls == 1);
            CHECK(f.memory[f.tx.patch(0).offset + foreignByte] == foreign);
        } else {
            CHECK(result.state == Tx::State::RolledBack && calls == 2 && f.memory == f.before);
        }
    }
    // A conflict introduced by an earlier rollback is checked again before the next restore.
    Fixture duringRollback;
    size_t calls = 0;
    uint8_t foreign = 0;
    auto interfere = [&](size_t i, const uint8_t* bytes, const uint8_t* expected) noexcept -> Tx::WriteResult {
        CHECK(std::memcmp(duringRollback.memory.data() + duringRollback.tx.patch(i).offset, expected, 5) == 0);
        const size_t call = calls++;
        std::memcpy(duringRollback.memory.data() + duringRollback.tx.patch(i).offset + 1, bytes, 4);
        if (call == 4) {
            auto& changed = duringRollback.memory[duringRollback.tx.patch(1).offset + 1];
            changed ^= 0x5a; foreign = changed;
        }
        return {call != 3, 0x0f};
    };
    const auto result = duringRollback.tx.Apply(interfere);
    CHECK(result.state == Tx::State::RollbackFailed && !result.committed() && calls == 7);
    CHECK(duringRollback.memory[duringRollback.tx.patch(1).offset + 1] == foreign);
}
static void TestPreparation() {
    for (int kind = 0; kind < 8; ++kind) {
        Fixture f(1);
        bool added = true;
        const auto base = reinterpret_cast<uintptr_t>(f.memory.data());
        if (kind == 0) added = f.tx.Add(8, 800, base + 900); // duplicate
        if (kind == 1) added = f.tx.Add(9, 800, base + 900); // overlap
        if (kind == 2) added = f.tx.Add(1020, 800, base + 900); // truncated
        if (kind == 3) added = f.tx.Add(UINT32_MAX, 800, base + 900);
        if (kind == 4) added = f.tx.Add(32, 800, base + 900); // not E8
        if (kind == 5) { f.memory[32] = 0xe8; added = f.tx.Add(32, 800, base + 900); } // wrong target
        if (kind == 6) added = f.tx.Add(32, 1024, base + 900); // target outside image
        if (kind == 7) {
            f.memory[32] = 0xe8;
            const int32_t rel = 800 - 37; std::memcpy(f.memory.data() + 33, &rel, 4);
            added = f.tx.Add(32, 800, base + 37 + uint64_t(INT32_MAX) + 1);
        }
        CHECK(!added);
        const auto before = f.memory;
        size_t writes = 0;
        auto never = [&](size_t, const uint8_t*, const uint8_t*) noexcept -> Tx::WriteResult { ++writes; return {true, 0x0f}; };
        CHECK(f.tx.Apply(never).state == Tx::State::Rejected && writes == 0);
        CHECK(f.memory == before);
    }
    Fixture full(Tx::kCapacity);
    CHECK(!full.tx.Add(700, 800, reinterpret_cast<uintptr_t>(full.memory.data()) + 900));
    size_t writes = 0;
    auto never = [&](size_t, const uint8_t*, const uint8_t*) noexcept -> Tx::WriteResult { ++writes; return {true, 0x0f}; };
    CHECK(full.tx.Apply(never).state == Tx::State::Rejected && writes == 0);
    Tx empty(nullptr, 0);
    CHECK(empty.Apply(never).state == Tx::State::Rejected);
    CHECK(!empty.Add(0, 0, 0));
    // Both inclusive rel32 endpoints are accepted; one beyond each is refused. Keep distances
    // explicitly unsigned and independently check test-address arithmetic: MSVC Release optimized
    // the former mixed signed-loop-index ternary incorrectly, shifting positive cases down by 2^32.
    const std::array<uint64_t, 4> distances{{0x7fffffffULL, 0x80000000ULL, 0x80000000ULL, 0x80000001ULL}};
    for (int endpoint = 0; endpoint < 4; ++endpoint) {
        Fixture f(0);
        f.memory[8] = 0xe8;
        const int32_t original = 800 - 13; std::memcpy(f.memory.data() + 9, &original, 4);
        const uintptr_t next = reinterpret_cast<uintptr_t>(f.memory.data()) + 13;
        const uintptr_t distance = static_cast<uintptr_t>(distances[endpoint]);
        uintptr_t stub = 0;
        if (endpoint < 2) {
            stub = next + distance;
            CHECK(stub >= next && stub - next == distance);
        } else {
            CHECK(next >= distance);
            stub = next - distance;
            CHECK(stub <= next && next - stub == distance);
        }
        const bool added = f.tx.Add(8, 800, stub);
        if (added != (endpoint == 0 || endpoint == 2))
            std::printf("endpoint=%d next=%llx stub=%llx added=%d pointerBytes=%zu\n", endpoint,
                        static_cast<unsigned long long>(next), static_cast<unsigned long long>(stub),
                        int(added), sizeof(uintptr_t));
        CHECK(added == (endpoint == 0 || endpoint == 2));
    }
}
int main() {
    TestPreparation(); TestSuccessAndRevalidation(); TestFailures(); TestOwnershipConflicts();
    std::printf("ASI patch transaction: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
