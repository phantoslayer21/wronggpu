// SPDX-License-Identifier: MIT
// RR guard: CPU unit test of the coexistence guard's pieces without process state (src/rr_guard.h).
//   - safe reads: a readable buffer; a PAGE_NOACCESS page, a reserved page, a guard page (the guard flag must survive the
//     check, with no other thread changing the page meanwhile); a read across a readable / unreadable boundary; the
//     cached reader's structured-exception path (a page made unreadable after its region was cached); the cache keeps
//     VirtualQuery calls few;
//   - PeView: the real exe's headers; synthetic headers with a bad e_lfanew (negative, inside the DOS header, absurd, NT
//     headers past SizeOfImage), directories past SizeOfImage or wrapping, too few directories, PE32, no signature;
//   - decode_jump: every jump form, and a jmp [rip+disp32] whose cell is not readable;
//   - classify: synthetic modules (ranges of one allocation) and anonymous relays: the interface's own code, a hook jump
//     to another module, a jump inside the interface, the linker's import thunk, a relay chain to the host, exactly
//     kMaxHops jumps (followed) and one more (too many hops), a relay loop, anonymous code, an unreadable target, an
//     execute-only relay page (read), a copy of WrongGPU RR, WrongGPU.asi, the exe's thunk, the host's second module;
//   - Verdict: a copy of WrongGPU RR wins; a named foreign module is kept (the first one) as kForeign; unreadable memory,
//     too many hops, non-jumping anonymous code and the exe's own code are counted as unclear;
//   - decide: with NO second RR provider (a game-owned wrapper DLL, a host helper in a third image, an overlay-style
//     wrapper, a forwarded export, the heap-table and helper-table host layouts) RR stays active at every stage unless
//     rr_other_plugin=yield (take: in front, never off); a copy of WrongGPU RR switches it off in every mode; a decision
//     already taken stays; override parsing (off | take | yield), the exact log lines;
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
    CHECK(VirtualQuery(blk + 2 * kPage, &m, sizeof m) && (m.Protect & PAGE_GUARD));  // untouched by the check
    uint64_t p64 = 0;
    CHECK(!safe_get(blk + 3 * kPage, &p64));
    // an execute-only page counts as readable (a hooking library's trampolines); the copy is guarded either way
    uint8_t* xo = static_cast<uint8_t*>(VirtualAlloc(nullptr, kPage, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    CHECK(xo != nullptr);
    memset(xo, 0xC3, kPage);
    DWORD xold = 0;
    CHECK(VirtualProtect(xo, kPage, PAGE_EXECUTE, &xold));
    CHECK(readable(xo, 16));
    (void)safe_read(xo, b, 1);  // hardware without read-through-execute faults here: caught, false; no crash either way
    VirtualFree(xo, 0, MEM_RELEASE);

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
    // a relay loop: more than kMaxHops -> unknown (inconclusive: not a sign of another plugin)
    put_abs(anon + 256, anon + 512);
    put_abs(anon + 512, anon + 256);
    w = classify(anon + 256, env, false);
    CHECK(w.owner == Owner::kUnknown && w.too_many_hops && w.hops == kMaxHops);
    // exactly kMaxHops (4) jumps are followed to a foreign module; a fifth one is too many
    static_assert(kMaxHops == 4, "the chains below are written for four hops");
    put_abs(anon + 2048, anon + 2112);
    put_abs(anon + 2112, anon + 2176);
    put_abs(anon + 2176, anon + 2240);
    put_abs(anon + 2240, foreign + 72);
    w = classify(anon + 2048, env, false);
    CHECK(w.owner == Owner::kForeign && w.module == "other.asi" && w.hops == 4 && !w.too_many_hops);
    put_abs(anon + 2240, anon + 2304);  // lengthen it by one jump
    put_abs(anon + 2304, foreign + 72);
    w = classify(anon + 2048, env, false);
    CHECK(w.owner == Owner::kUnknown && w.too_many_hops && w.hops == 4);
    put_abs(anon + 2240, foreign + 72);  // (kept four hops for the Verdict checks below)
    // the host's API-table module (a separate loader called InitializeASI): a second host module
    {
      Env env2 = env;
      env2.host_alt = foreign;
      CHECK(classify(foreign + 8, env, false).owner == Owner::kForeign);
      CHECK(classify(foreign + 8, env2, false).owner == Owner::kHost);
      CHECK(classify(anon + 2048, env2, false).owner == Owner::kHost);  // a hook chain that ends in it
    }
    // an execute-only relay page is read like any other
    put_abs(anon + kPage + 16, foreign + 80);
    CHECK(VirtualProtect(anon + kPage, kPage, PAGE_EXECUTE, &old));
    w = classify(anon + kPage + 16, env, false);
    CHECK(w.owner == Owner::kForeign && w.hops == 1);
    CHECK(VirtualProtect(anon + kPage, kPage, PAGE_EXECUTE_READWRITE, &old));
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
    CHECK(v.kind == Verdict::kClean && v.reads == 4 && v.unclear == 0);
    // not positive findings: unreadable memory, a null slot, too many hops, anonymous code that does not jump on (not
    // accepted here), the exe's own code (not accepted here), a hook chain that ends in the exe: counted, kind stays clean
    v.add(classify(dead + 32, env, false), "the import slot F of game.exe", false, false);
    CHECK(v.kind == Verdict::kClean && v.unclear == 1 && v.unclear_detail == "the import slot F of game.exe could not be read");
    v.add(classify(nullptr, env, false), "the import slot N of game.exe", false, false);
    v.add(classify(anon + 256, env, false), "the import slot L of game.exe", false, false);
    v.add(classify(anon + 1024, env, false), "the import slot I of game.exe", false, false);
    v.add(classify(exe + 8, env, false), "the import slot J of game.exe", false, false);
    CHECK(v.kind == Verdict::kClean && v.unclear == 5 && v.reads == 9);
    CHECK(v.unclear_detail == "the import slot F of game.exe could not be read");  // the first one stays
    Verdict u;
    u.add(classify(anon + 256, env, false), "the import slot L of game.exe", false, false);
    CHECK(u.kind == Verdict::kClean && u.unclear == 1 &&
          u.unclear_detail == "the import slot L of game.exe leads through more than 4 hook jumps");
    Verdict a;
    a.add(classify(anon + 1024, env, false), "the import slot I of game.exe", false, false);
    CHECK(a.kind == Verdict::kClean && a.unclear == 1 &&
          a.unclear_detail == "the import slot I of game.exe leads to code outside every module");
    put_rel32(iface + 640, exe + 24);  // an export hooked into the exe's own code
    Verdict e;
    e.add(classify(exe + 8, env, false), "the import slot J of game.exe", false, false);
    e.add(classify(iface + 640, env, true), "the export K of iface.dll", false, false);
    CHECK(e.kind == Verdict::kClean && e.unclear == 2 &&
          e.unclear_detail == "the import slot J of game.exe leads into the exe's own code");
    Verdict e2;
    e2.add(classify(iface + 640, env, true), "the export K of iface.dll", false, false);
    CHECK(e2.unclear_detail == "the export K of iface.dll leads into the exe's own code (through 1 hook jump)");
    Verdict ok;  // the accepted places stay trusted
    ok.add(classify(anon + 1024, env, false), "the host's next function of D", true, false);
    ok.add(classify(exe + 8, env, false), "the delay-load slot C of game.exe", false, true);
    CHECK(ok.kind == Verdict::kClean && ok.unclear == 0);
    // positive findings
    v.add(classify(iface + 384, env, true), "the export E of iface.dll", false, false);
    CHECK(v.kind == Verdict::kForeign && v.other_module == foreign && v.module == "other.asi" && v.foreign == 1);
    CHECK(v.detail == "the export E of iface.dll leads to other.asi (through 3 hook jumps)");
    v.add(classify(foreign + 8, env, false), "the import slot M of game.exe", false, false);
    CHECK(v.detail == "the export E of iface.dll leads to other.asi (through 3 hook jumps)");  // the first one stays
    v.add(classify(copy + 8, env, false), "the import slot G of game.exe", false, false);
    CHECK(v.kind == Verdict::kOwnCopy && v.detail == "WrongGPU-RR-copy.asi");  // a copy of WrongGPU RR before all
    v.add(classify(foreign + 8, env, false), "the import slot H of game.exe", false, false);
    CHECK(v.kind == Verdict::kOwnCopy && v.reads == 13);
    Verdict f4;  // a decoded chain of exactly kMaxHops jumps that ends in a foreign module is positive
    f4.add(classify(anon + 2048, env, false), "the import slot P of game.exe", false, false);
    CHECK(f4.kind == Verdict::kForeign && f4.detail == "the import slot P of game.exe leads to other.asi (through 4 hook jumps)");
    Verdict f5;  // the unreadable first read does not hide a later positive finding
    f5.add(classify(dead + 32, env, false), "the import slot F of game.exe", false, false);
    f5.add(classify(foreign + 8, env, false), "the import slot Q of game.exe", false, false);
    CHECK(f5.kind == Verdict::kForeign && f5.unclear == 1 && f5.detail == "the import slot Q of game.exe leads to other.asi");
    CHECK(trusted(Owner::kWrongGpu, false, false) && !trusted(Owner::kUnknown, true, true) &&
          !trusted(Owner::kForeign, true, true) && !trusted(Owner::kOwnCopy, true, true));

    // ---- layouts with NO second RR provider: a named module on the path is inconclusive, RR stays active (off) at
    // every stage; only rr_other_plugin=yield switches it off, take puts WrongGPU RR in front while it still can
    {
      Env heap = env;  // the host's API table on the heap: no module, so nothing but InitializeASI's caller is the host
      heap.host_alt = nullptr;
      Env helper_table = env;  // the table in a helper image: a module, but not the one that patched
      helper_table.host_alt = wgasi;
      void* fwd_fn = foreign + 200;           // a forwarded export: GetProcAddress resolves into the implementation DLL
      struct Layout { const char* name; Where w; };
      const Layout layouts[] = {
          {"game-owned wrapper DLL", classify(foreign + 8, env, false)},           // the exe's slot wraps through it
          {"host helper in a third image", classify(foreign + 16, env, false)},    // the host is host.dll, not it
          {"overlay-style wrapper", classify(iface + 384, env, true)},             // export -> relay -> relay -> module
          {"forwarded export", classify(fwd_fn, env, true)},                       // the export's address is in it
          {"heap-table host layout", classify(foreign + 24, heap, false)},         // the real host's own patch
          {"helper-table host layout", classify(foreign + 32, helper_table, false)},
      };
      for (const Layout& l : layouts) {
        Verdict lv;
        lv.add(l.w, std::string("the import slot X of game.exe (") + l.name + ")", false, false);
        CHECK(lv.kind == Verdict::kForeign && lv.module == "other.asi");
        for (Stage st : {Stage::kLoad, Stage::kRoutes, Stage::kLate}) {
          CHECK(decide(lv.kind, Mode::kOff, st, State::kActive) == State::kActive);  // stays active
          CHECK(decide(lv.kind, Mode::kOff, st, State::kTaken) == State::kTaken);
          CHECK(decide(lv.kind, Mode::kTake, st, State::kActive) ==
                (st == Stage::kLate ? State::kActive : State::kTaken));  // never off with take
          CHECK(decide(lv.kind, Mode::kYield, st, State::kActive) ==
                (st == Stage::kLoad ? State::kOff : State::kPassThrough));
        }
      }
      // the host's own patch, when the host is known (caller of InitializeASI or the table's module), is not named
      Env split = env;
      split.host_alt = foreign;
      Verdict hv;
      hv.add(classify(foreign + 24, split, false), "the import slot X of game.exe", false, false);
      CHECK(hv.kind == Verdict::kClean && hv.unclear == 0 && hv.foreign == 0);
    }
    // ---- evidence that is positive by contract: a second copy of WrongGPU RR, at every stage and in every mode
    for (Mode m : {Mode::kOff, Mode::kTake, Mode::kYield}) {
      CHECK(decide(Verdict::kOwnCopy, m, Stage::kLoad, State::kActive) == State::kOwnCopy);
      CHECK(decide(Verdict::kOwnCopy, m, Stage::kRoutes, State::kActive) == State::kPassThrough);
      CHECK(decide(Verdict::kOwnCopy, m, Stage::kLate, State::kTaken) == State::kPassThrough);
      for (Stage st : {Stage::kLoad, Stage::kRoutes, Stage::kLate}) {
        CHECK(decide(Verdict::kClean, m, st, State::kActive) == State::kActive);  // clean: nothing changes
        CHECK(decide(Verdict::kClean, m, st, State::kTaken) == State::kTaken);
        for (State done : {State::kOff, State::kOwnCopy, State::kPassThrough})  // a decision already taken stays
          for (Verdict::Kind k : {Verdict::kClean, Verdict::kOwnCopy, Verdict::kForeign})
            CHECK(decide(k, m, st, done) == done);
      }
    }
    VirtualFree(blk, 0, MEM_RELEASE);
  }

  // ---------------------------------------------------------------- PeView: header and directory bounds
  {
    // the real exe: headers accepted, its import directory inside the image
    Reader r;
    PeView pe;
    CHECK(pe.open(GetModuleHandleW(nullptr), r) && pe.size_of_image() > 0);
    uint32_t rva = 0, size = 0;
    CHECK(pe.dir(IMAGE_DIRECTORY_ENTRY_IMPORT, &rva, &size) && pe.at(rva, size) != nullptr);
    CHECK(pe.at(pe.size_of_image(), 0) != nullptr && pe.at(pe.size_of_image(), 1) == nullptr);
    CHECK(pe.at(0xFFFFFFFFull, 2) == nullptr && pe.at(8, 0xFFFFFFFFFFFFFFF0ull) == nullptr);  // no wrap-around
    // a synthetic image: DOS header, NT headers at 0x80, SizeOfImage 0x2000, in 3 committed pages (one past the image)
    uint8_t* img = static_cast<uint8_t*>(VirtualAlloc(nullptr, 3 * kPage, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    CHECK(img != nullptr);
    auto* dos = reinterpret_cast<IMAGE_DOS_HEADER*>(img);
    auto* nt = reinterpret_cast<IMAGE_NT_HEADERS64*>(img + 0x80);
    auto reset = [&] {
      memset(img, 0, 3 * kPage);
      dos->e_magic = IMAGE_DOS_SIGNATURE;
      dos->e_lfanew = 0x80;
      nt->Signature = IMAGE_NT_SIGNATURE;
      nt->OptionalHeader.Magic = IMAGE_NT_OPTIONAL_HDR64_MAGIC;
      nt->OptionalHeader.SizeOfImage = 0x2000;
      nt->OptionalHeader.NumberOfRvaAndSizes = 16;
      nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT] = {0x1000, 0x100};
    };
    reset();
    PeView s;
    CHECK(s.open(img, r) && s.size_of_image() == 0x2000 && s.dir(IMAGE_DIRECTORY_ENTRY_IMPORT, &rva, &size) &&
          rva == 0x1000 && size == 0x100);
    // e_lfanew negative, inside the DOS header, absurdly far, or with the NT headers past SizeOfImage: refused
    for (LONG bad : {(LONG)-8, (LONG)4, (LONG)0x7FFFFFF0, (LONG)0x1FF0}) {
      reset();
      if (bad == 0x1FF0) {  // the NT headers copied there (readable memory, but outside the image's 0x2000 bytes)
        memcpy(img + 0x1FF0, nt, sizeof(IMAGE_NT_HEADERS64));
      }
      dos->e_lfanew = bad;
      PeView b;
      CHECK(!b.open(img, r));
    }
    // a directory whose RVA or size runs past SizeOfImage (or wraps): not handed out
    const IMAGE_DATA_DIRECTORY bad_dirs[] = {{0x2000, 1}, {0x1F00, 0x200}, {0xFFFFFFF0, 0x20}, {0x10, 0xFFFFFFF8}};
    for (const IMAGE_DATA_DIRECTORY& d : bad_dirs) {
      reset();
      nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT] = d;
      PeView b;
      CHECK(b.open(img, r) && !b.dir(IMAGE_DIRECTORY_ENTRY_IMPORT, &rva, &size));
    }
    // fewer directories than the index: absent; a PE32 optional header or a wrong signature: refused
    reset();
    nt->OptionalHeader.NumberOfRvaAndSizes = 1;
    PeView few;
    CHECK(few.open(img, r) && !few.dir(IMAGE_DIRECTORY_ENTRY_IMPORT, &rva, &size));
    reset();
    nt->OptionalHeader.Magic = IMAGE_NT_OPTIONAL_HDR32_MAGIC;
    PeView p32;
    CHECK(!p32.open(img, r));
    reset();
    nt->Signature = 0;
    PeView nosig;
    CHECK(!nosig.open(img, r));
    VirtualFree(img, 0, MEM_RELEASE);
  }

  // ---------------------------------------------------------------- the override, the log lines
  {
    std::string note;
    CHECK(parse_mode("", &note) == Mode::kOff && note.empty());
    CHECK(parse_mode("off", &note) == Mode::kOff && note.empty());
    CHECK(parse_mode("take", &note) == Mode::kTake && note.empty());
    CHECK(parse_mode(" Take \r", &note) == Mode::kTake && note.empty());
    CHECK(parse_mode("yield", &note) == Mode::kYield && note.empty());
    CHECK(parse_mode("YIELD\t", &note) == Mode::kYield && note.empty());
    CHECK(parse_mode("ours", &note) == Mode::kOff &&
          note == "rr_other_plugin=ours in wronggpu-rr.ini not recognised (off | take | yield); off is used");
    CHECK(strcmp(mode_word(Mode::kOff), "off") == 0 && strcmp(mode_word(Mode::kTake), "take") == 0 &&
          strcmp(mode_word(Mode::kYield), "yield") == 0);
    // the beginnings a reader matches (HOOK-MODE.md): exact, no prefix
    const std::string fstem = "another module hooks the game's RR calls: ";
    const std::string d = "the import slot X of Game.exe leads to Other.dll";
    CHECK(line_foreign("Other.dll", d, Tail::kStaysActive) ==
          fstem + "Other.dll (" + d + "); it is not known to be a Ray Reconstruction plugin: WrongGPU RR stays active "
                  "(rr_other_plugin=yield in wronggpu-rr.ini keeps WrongGPU RR off in this case)");
    CHECK(line_foreign("Other.dll", d, Tail::kTakeInFront) ==
          fstem + "Other.dll (" + d + "); rr_other_plugin=take: WrongGPU RR goes in front of it (it gets the game's "
                  "other calls, no RR work)");
    CHECK(line_foreign("Other.dll", d, Tail::kYieldOff) ==
          fstem + "Other.dll (" + d + "); rr_other_plugin=yield: WrongGPU RR stays off (nothing patched)");
    CHECK(line_foreign("Other.dll", d, Tail::kYieldPassing) ==
          fstem + "Other.dll (" + d + "); rr_other_plugin=yield: WrongGPU RR stays off (WrongGPU's handlers pass every "
                  "call on unchanged)");
    // only yield says "stays off"; nothing about a named module claims it is a Ray Reconstruction plugin
    const std::string stem = "another Ray Reconstruction plugin is active; WrongGPU RR stays off";
    for (Tail t : {Tail::kStaysActive, Tail::kTakeInFront, Tail::kYieldOff, Tail::kYieldPassing}) {
      const std::string l = line_foreign("M", "D", t);
      CHECK(l.compare(0, fstem.size(), fstem) == 0 && l.find("plugin is active") == std::string::npos);
      CHECK((l.find("stays off") != std::string::npos) == (t == Tail::kYieldOff || t == Tail::kYieldPassing));
    }
    CHECK(line_own_copy("A", "B") == "a second copy of WrongGPU RR is loaded (A is active); this copy (B) stays off");
    CHECK(line_ngx_other() == stem + " (NGX route only: the host refused WrongGPU's RR provider, another provider "
                                     "holds it; the game-interface route is decided on its own)");
    // the stay-on diagnostic: never one of the matched beginnings
    const std::string inc = line_inconclusive("at load", 2, "the import slot X of Game.exe could not be read");
    CHECK(inc == "hook: coexistence check at load: inconclusive for 2 address(es) (first: the import slot X of Game.exe "
                 "could not be read); no other plugin was positively found, WrongGPU RR stays active");
    CHECK(inc.compare(0, 6, "hook: ") == 0 && inc.find("plugin is active") == std::string::npos &&
          inc.find("stays off") == std::string::npos);
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
