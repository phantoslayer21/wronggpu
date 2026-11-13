// SPDX-License-Identifier: MIT
// RR guard: CPU unit test of the coexistence guard's pieces without process state (src/rr_guard.h).
//   - safe reads: a readable buffer; a PAGE_NOACCESS page, a reserved page, a guard page (the guard flag must survive the
//     check: nothing touched it); a read across a readable / unreadable boundary; the cached reader's structured-exception
//     path (a page made unreadable after its region was cached); the cache keeps VirtualQuery calls few;
//   - decode_jump: every jump form, and a jmp [rip+disp32] whose cell is not readable;
//   - classify: synthetic modules (ranges of one allocation) and anonymous relays: the interface's own code, a hook jump
//     to another module, a jump inside the interface, the linker's import thunk, a relay chain to the host, a relay loop
//     (more than kMaxHops), anonymous code, an unreadable target, a copy of WrongGPU RR, WrongGPU.asi, the exe's thunk;
//   - Verdict precedence, override parsing, the exact log lines;
//   - Decision: 32 threads make the "first call" at once: decide runs exactly once, every thread sees the same state.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <atomic>
#include <string>
#include <thread>
#include <vector>

#include "../src/rr_guard.h"
#include "test_util.h"

using namespace wgrr::guard;

static const size_t kPage = 4096;

static void put_rel32(uint8_t* at, const void* target) {  // E9 rel32 at `at`
  at[0] = 0xE9;
  const int32_t rel = (int32_t)((const uint8_t*)target - (at + 5));
  memcpy(at + 1, &rel, 4);
}
static void put_abs(uint8_t* at, const void* target) {  // FF 25 00000000 <abs64>
  at[0] = 0xFF;
  at[1] = 0x25;
  memset(at + 2, 0, 4);
  memcpy(at + 6, &target, 8);
}
static void put_rip_cell(uint8_t* at, const void* cell) {  // FF 25 <disp32 to cell>
  at[0] = 0xFF;
  at[1] = 0x25;
  const int32_t disp = (int32_t)((const uint8_t*)cell - (at + 6));
  memcpy(at + 2, &disp, 4);
}

int main() {
  // ---------------------------------------------------------------- safe reads
  {
    uint64_t v = 0x1122334455667788ull, out = 0;
    CHECK(safe_read(&v, &out, sizeof v) && out == v);
    CHECK(!safe_read(nullptr, &out, 8));
    CHECK(!safe_read(&v, nullptr, 8));

    uint8_t* blk = static_cast<uint8_t*>(VirtualAlloc(nullptr, 4 * kPage, MEM_RESERVE, PAGE_NOACCESS));
    CHECK(blk != nullptr);
    // page 0 readable, page 1 PAGE_NOACCESS, page 2 guard page, page 3 reserved only
    CHECK(VirtualAlloc(blk, kPage, MEM_COMMIT, PAGE_READWRITE) == blk);
    CHECK(VirtualAlloc(blk + kPage, kPage, MEM_COMMIT, PAGE_NOACCESS) == blk + kPage);
    CHECK(VirtualAlloc(blk + 2 * kPage, kPage, MEM_COMMIT, PAGE_READWRITE | PAGE_GUARD) == blk + 2 * kPage);
    memset(blk, 0xAB, kPage);
    uint8_t b[16] = {};
    CHECK(safe_read(blk + 8, b, 16) && b[0] == 0xAB && b[15] == 0xAB);
    CHECK(readable(blk, kPage));
    CHECK(!readable(blk + kPage, 1));
    CHECK(!safe_read(blk + kPage, b, 1));              // PAGE_NOACCESS
    CHECK(!safe_read(blk + 2 * kPage + 16, b, 8));     // guard page
    CHECK(!safe_read(blk + 3 * kPage, b, 8));          // reserved, not committed
    CHECK(!safe_read(blk + kPage - 4, b, 8));          // across readable -> unreadable
    MEMORY_BASIC_INFORMATION m;
    CHECK(VirtualQuery(blk + 2 * kPage, &m, sizeof m) && (m.Protect & PAGE_GUARD));  // never touched: still a guard page
    uint64_t p64 = 0;
    CHECK(!safe_get(blk + 3 * kPage, &p64));

    // the cached reader: many reads in one region take one VirtualQuery; a page made unreadable after its region was
    // cached is caught by the structured-exception path (no fault, false)
    Reader r;
    for (int i = 0; i < 200; ++i) CHECK(r.get(blk + 8 * i, &p64));
    CHECK(r.queries == 1);
    CHECK(!r.get(blk + kPage, &p64));
    Reader r2;
    CHECK(r2.get(blk + 64, &p64));
    DWORD old = 0;
    CHECK(VirtualProtect(blk, kPage, PAGE_NOACCESS, &old));
    CHECK(!r2.get(blk + 128, &p64));  // cached as readable, now faults: caught
    CHECK(VirtualProtect(blk, kPage, PAGE_READWRITE, &old));
    char s[8];
    memcpy(blk + kPage - 3, "abc", 3);  // no NUL before the unreadable page
    Reader r3;
    CHECK(!r3.str(blk + kPage - 3, s, sizeof s));
    memcpy(blk + 100, "name", 5);
    CHECK(r3.str(blk + 100, s, sizeof s) && strcmp(s, "name") == 0);
    CHECK(!r3.str(blk + 100, s, 4));  // longer than the buffer
    VirtualFree(blk, 0, MEM_RELEASE);
  }

  // ---------------------------------------------------------------- decode_jump
  {
    uint8_t code[32];
    const void* t = nullptr;
    const void* cell = nullptr;
    const uint8_t* at = reinterpret_cast<const uint8_t*>(0x10000000);
    code[0] = 0xE9;
    int32_t rel = 0x100;
    memcpy(code + 1, &rel, 4);
    CHECK(decode_jump(code, 5, at, &t, &cell) && t == at + 5 + 0x100 && cell == nullptr);
    CHECK(!decode_jump(code, 4, at, &t, &cell));  // too short
    code[0] = 0xEB;
    code[1] = 0xF0;  // -16
    CHECK(decode_jump(code, 2, at, &t, &cell) && t == at + 2 - 16);
    uint64_t target = 0x123456789ull;
    uint8_t mov_rax[12] = {0x48, 0xB8};
    memcpy(mov_rax + 2, &target, 8);
    mov_rax[10] = 0xFF;
    mov_rax[11] = 0xE0;
    CHECK(decode_jump(mov_rax, 12, at, &t, &cell) && (uint64_t)(uintptr_t)t == target);
    mov_rax[10] = 0x50;
    mov_rax[11] = 0xC3;
    CHECK(decode_jump(mov_rax, 12, at, &t, &cell) && (uint64_t)(uintptr_t)t == target);
    uint8_t mov_r11[13] = {0x49, 0xBB};
    memcpy(mov_r11 + 2, &target, 8);
    mov_r11[10] = 0x41;
    mov_r11[11] = 0xFF;
    mov_r11[12] = 0xE3;
    CHECK(decode_jump(mov_r11, 13, at, &t, &cell) && (uint64_t)(uintptr_t)t == target);
    // jmp [rip+0] with the cell right behind it (real memory: the cell is read)
    uint8_t* live = static_cast<uint8_t*>(VirtualAlloc(nullptr, 2 * kPage, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    put_abs(live, reinterpret_cast<const void*>(target));
    CHECK(decode_jump(live, 14, live, &t, &cell) && (uint64_t)(uintptr_t)t == target && cell == live + 6);
    // REX.W form
    live[100] = 0x48;
    put_abs(live + 101, reinterpret_cast<const void*>(target));
    CHECK(decode_jump(live + 100, 15, live + 100, &t, &cell) && (uint64_t)(uintptr_t)t == target);
    // a cell on an unreadable page: no decode (classify reports unknown)
    DWORD old = 0;
    put_rip_cell(live + 200, live + kPage + 8);
    CHECK(VirtualProtect(live + kPage, kPage, PAGE_NOACCESS, &old));
    CHECK(!decode_jump(live + 200, 6, live + 200, &t, &cell));
    uint8_t plain[4] = {0x48, 0x89, 0x5C, 0x24};  // mov [rsp+..], rbx: not a jump
    CHECK(!decode_jump(plain, 4, at, &t, &cell));
    VirtualFree(live, 0, MEM_RELEASE);
  }

  // ---------------------------------------------------------------- classify (synthetic modules)
  {
    // one executable allocation split into "modules": iface, foreign, host, own copy, WrongGPU.asi, exe, self;
    // the last pages are anonymous (no module); one page is made PAGE_NOACCESS
    const size_t kMod = 4 * kPage;
    const int kMods = 7;
    uint8_t* blk = static_cast<uint8_t*>(VirtualAlloc(nullptr, kMods * kMod + 4 * kPage, MEM_RESERVE | MEM_COMMIT,
                                                      PAGE_EXECUTE_READWRITE));
    CHECK(blk != nullptr);
    memset(blk, 0x90, kMods * kMod + 4 * kPage);  // nop: "not a jump"
    uint8_t* iface = blk;
    uint8_t* foreign = blk + kMod;
    uint8_t* host = blk + 2 * kMod;
    uint8_t* copy = blk + 3 * kMod;
    uint8_t* wgasi = blk + 4 * kMod;
    uint8_t* exe = blk + 5 * kMod;
    uint8_t* self = blk + 6 * kMod;
    uint8_t* anon = blk + kMods * kMod;  // 4 anonymous pages; the last one unreadable
    const char* names[kMods] = {"iface.dll", "other.asi", "host.dll", "WrongGPU-RR-copy.asi", "WrongGPU.asi", "game.exe",
                                "WrongGPU-RR.asi"};
    Env env;
    env.iface = iface;
    env.host = host;
    env.exe = exe;
    env.self = self;
    env.iface_iat_begin = iface + 3 * kPage;  // the interface's import table: its last page
    env.iface_iat_end = iface + 4 * kPage;
    env.module_of = [=](const void* a) -> const void* {
      const uint8_t* p = static_cast<const uint8_t*>(a);
      if (p < blk || p >= blk + kMods * kMod) return nullptr;
      return blk + ((p - blk) / kMod) * kMod;
    };
    env.exports = [=](const void* m, const char* name) {
      return (m == copy && strcmp(name, kOwnRrExport) == 0) || (m == wgasi && strcmp(name, kWrongGpuAsiExport) == 0);
    };
    env.name_of = [=](const void* m) { return std::string(names[(static_cast<const uint8_t*>(m) - blk) / kMod]); };
    DWORD old = 0;

    Where w = classify(iface + 16, env, true);  // the interface's own code
    CHECK(w.owner == Owner::kInterface && w.module == "iface.dll" && w.hops == 0);
    put_abs(iface + 64, foreign + 32);  // a hook on an export: FF25 to another module
    w = classify(iface + 64, env, true);
    CHECK(w.owner == Owner::kForeign && w.module == "other.asi" && w.hops == 1);
    w = classify(iface + 64, env, false);  // a slot value inside the interface (no following)
    CHECK(w.owner == Owner::kInterface);
    put_rel32(iface + 128, iface + 1024);  // a jump inside the interface: its own code
    CHECK(classify(iface + 128, env, true).owner == Owner::kInterface);
    void* foreign_fn = foreign + 48;
    memcpy(iface + 3 * kPage + 8, &foreign_fn, 8);  // the interface's import table cell
    put_rip_cell(iface + 192, iface + 3 * kPage + 8);  // the linker's import thunk: not a hook
    CHECK(classify(iface + 192, env, true).owner == Owner::kInterface);
    memcpy(iface + 2048, &foreign_fn, 8);
    put_rip_cell(iface + 256, iface + 2048);  // a cell outside the import table (a hook's cell): followed
    w = classify(iface + 256, env, true);
    CHECK(w.owner == Owner::kForeign && w.hops == 1);
    // export -> E9 -> anonymous relay (FF25) -> host: the host's own trampoline chain
    put_abs(anon + 16, host + 8);
    put_rel32(iface + 320, anon + 16);
    w = classify(iface + 320, env, true);
    CHECK(w.owner == Owner::kHost && w.hops == 2);
    // export -> relay -> relay -> foreign
    put_abs(anon + 64, anon + 128);
    put_abs(anon + 128, foreign + 64);
    put_rel32(iface + 384, anon + 64);
    w = classify(iface + 384, env, true);
    CHECK(w.owner == Owner::kForeign && w.module == "other.asi" && w.hops == 3);
    // a relay loop: more than kMaxHops -> unknown (the safe side)
    put_abs(anon + 256, anon + 512);
    put_abs(anon + 512, anon + 256);
    CHECK(classify(anon + 256, env, false).owner == Owner::kUnknown);
    // anonymous code that does not jump on
    CHECK(classify(anon + 1024, env, false).owner == Owner::kAnonymous);
    // an unreadable target: unknown, and an export whose jmp [rip] cell is unreadable: unknown
    uint8_t* dead = anon + 3 * kPage;
    CHECK(VirtualProtect(dead, kPage, PAGE_NOACCESS, &old));
    CHECK(classify(dead + 32, env, false).owner == Owner::kUnknown);
    put_rel32(iface + 448, dead + 32);
    CHECK(classify(iface + 448, env, true).owner == Owner::kUnknown);
    put_rip_cell(iface + 512, dead + 64);
    CHECK(classify(iface + 512, env, true).owner == Owner::kUnknown);
    CHECK(classify(nullptr, env, false).owner == Owner::kUnknown);
    // the other module kinds
    CHECK(classify(copy + 8, env, false).owner == Owner::kOwnCopy);
    CHECK(classify(wgasi + 8, env, false).owner == Owner::kWrongGpu);
    CHECK(classify(exe + 8, env, false).owner == Owner::kExeThunk);
    CHECK(classify(self + 8, env, false).owner == Owner::kSelf);
    CHECK(classify(host + 8, env, false).owner == Owner::kHost);
    // an export hooked straight into a copy of WrongGPU RR
    put_abs(iface + 576, copy + 16);
    w = classify(iface + 576, env, true);
    CHECK(w.owner == Owner::kOwnCopy && w.module == "WrongGPU-RR-copy.asi");

    // ---- Verdict
    Verdict v;
    v.add(classify(iface + 16, env, true), "the export A of iface.dll", false, false);
    v.add(classify(host + 8, env, false), "the import slot B of game.exe", false, false);
    v.add(classify(exe + 8, env, false), "the delay-load slot C of game.exe", false, true);
    v.add(classify(anon + 1024, env, false), "the host's next function of D", true, false);
    CHECK(v.kind == Verdict::kClean && v.reads == 4);
    v.add(classify(iface + 384, env, true), "the export E of iface.dll", false, false);
    CHECK(v.kind == Verdict::kOther);
    CHECK(v.detail == "the export E of iface.dll leads to other.asi (through 3 hook jumps)");
    v.add(classify(dead + 32, env, false), "the import slot F of game.exe", false, false);
    CHECK(v.detail == "the export E of iface.dll leads to other.asi (through 3 hook jumps)");  // the first one stays
    v.add(classify(copy + 8, env, false), "the import slot G of game.exe", false, false);
    CHECK(v.kind == Verdict::kOwnCopy && v.detail == "WrongGPU-RR-copy.asi");  // a copy of WrongGPU RR before all
    v.add(classify(foreign + 8, env, false), "the import slot H of game.exe", false, false);
    CHECK(v.kind == Verdict::kOwnCopy && v.reads == 8);
    Verdict u;
    u.add(classify(dead + 32, env, false), "the import slot F of game.exe", false, false);
    CHECK(u.kind == Verdict::kOther && u.detail == "the import slot F of game.exe could not be read");
    Verdict a;
    a.add(classify(anon + 1024, env, false), "the import slot I of game.exe", false, false);  // anon not accepted here
    CHECK(a.kind == Verdict::kOther && a.detail == "the import slot I of game.exe leads to code outside every module");
    Verdict e;
    e.add(classify(exe + 8, env, false), "the import slot J of game.exe", false, false);  // exe code not accepted here
    CHECK(e.kind == Verdict::kOther && e.detail == "the import slot J of game.exe leads to game.exe");
    CHECK(trusted(Owner::kWrongGpu, false, false) && !trusted(Owner::kUnknown, true, true) &&
          !trusted(Owner::kForeign, true, true) && !trusted(Owner::kOwnCopy, true, true));
    VirtualFree(blk, 0, MEM_RELEASE);
  }

  // ---------------------------------------------------------------- the override, the log lines
  {
    std::string note;
    CHECK(!override_take("", &note) && note.empty());
    CHECK(!override_take("off", &note) && note.empty());
    CHECK(override_take("take", &note) && note.empty());
    CHECK(override_take(" Take \r", &note) && note.empty());
    CHECK(!override_take("ours", &note) &&
          note == "rr_other_plugin=ours in wronggpu-rr.ini not recognised (off | take); off is used");
    // the beginnings the launcher's Last run page matches (HOOK-MODE.md): exact, no prefix
    const std::string stem = "another Ray Reconstruction plugin is active; WrongGPU RR stays off";
    CHECK(line_other("D") == stem + " (D; nothing patched; rr_other_plugin=take in wronggpu-rr.ini uses WrongGPU's "
                                    "instead)");
    CHECK(line_other("D", true) == stem + " (D; WrongGPU's handlers pass every call on unchanged; rr_other_plugin=take "
                                          "in wronggpu-rr.ini uses WrongGPU's instead)");
    CHECK(line_other_later("D") == stem + " (D; it loaded after WrongGPU: WrongGPU passes every call on unchanged; to "
                                          "use WrongGPU RR, move it aside)");
    CHECK(line_cannot_take("D") == "a plugin loaded after WrongGPU replaced its hooks; rr_other_plugin=take cannot be "
                                   "applied: WrongGPU RR stays off and passes every call on unchanged (D); move it aside "
                                   "to use WrongGPU RR");
    CHECK(line_take("D") == "another Ray Reconstruction plugin is active; rr_other_plugin=take: WrongGPU RR goes in "
                            "front of it (D; it gets the game's other calls, no RR work)");
    CHECK(line_take("D").compare(0, stem.size(), stem) != 0);  // RR is on: not a "stays off" line
    CHECK(line_own_copy("A", "B") == "a second copy of WrongGPU RR is loaded (A is active); this copy (B) stays off");
    CHECK(line_ngx_other() == stem + " (NGX route only: the host refused WrongGPU's RR provider, another provider "
                                     "holds it; the game-interface route is decided on its own)");
  }

  // ---------------------------------------------------------------- Decision: one decision under concurrent first calls
  for (int round = 0; round < 20; ++round) {
    Decision d;
    std::atomic<int> runs{0}, go{0}, wrong{0};
    std::vector<std::thread> th;
    for (int i = 0; i < 32; ++i)
      th.emplace_back([&] {
        while (!go.load()) std::this_thread::yield();
        d.freeze_once([&] {
          ++runs;
          Sleep(2);  // keep the others waiting inside
          return State::kPassThrough;
        });
        if (!d.frozen() || d.state() != State::kPassThrough || d.active()) ++wrong;  // every caller sees the decision
      });
    go.store(1);
    for (auto& t : th) t.join();
    CHECK_EQ(runs.load(), 1);
    CHECK_EQ(wrong.load(), 0);
    int later = 0;
    d.check([&] { ++later; return State::kActive; });  // frozen: a later check changes nothing
    d.freeze_once([&] { ++later; return State::kActive; });
    CHECK(later == 0 && d.state() == State::kPassThrough);
  }
  {
    Decision d;
    CHECK(d.active() && !d.frozen());
    d.check([] { return State::kTaken; });
    CHECK(d.active() && d.state() == State::kTaken && !d.frozen());
    d.set(State::kOff);
    CHECK(!d.active());
  }
  TEST_SUMMARY("test_rr_guard");
}
