// SPDX-License-Identifier: MIT
// CPU test of the RR runtime's per-list completion and retire list (src/rr_retire.h, issues #63 and #82): an object is
// released only when EVERY command list that may use it has proved complete through its own marker slot, never because
// a later list completed; a device switch keeps the old device's objects until its lists complete or it is removed.
// Fake objects count their Release calls; a plain array stands in for the mapped marker buffer, and "the GPU runs list
// s" is the write of s into s's slot. Built and run by test_rr_retire.py; exit status 0 when every check passes.
// Original WrongGPU code.
#include <stdio.h>

#include <map>
#include <vector>

#include "rr_retire.h"

namespace {

using wgrr_rt::ListMarks;
using wgrr_rt::Parked;
using wgrr_rt::ParkedSet;
using wgrr_rt::RetireList;

struct Fake {
  int releases = 0;
  unsigned long Release() { return (unsigned long)++releases; }
};

int g_failed = 0, g_checks = 0;
void check(bool ok, const char* what) {
  printf("%s: %s\n", ok ? "ok  " : "FAIL", what);
  ++g_checks;
  if (!ok) ++g_failed;
}

// a device's marker slots and its lists
struct Gpu {
  ListMarks marks;
  std::vector<uint32_t> words;
  std::map<uint64_t, uint32_t> slot_of;   // serial -> slot (a map: the wraparound test uses serials above 2^32)
  explicit Gpu(uint32_t n = 16) : marks(n), words(n, 0) {}
  uint64_t record() {   // the runtime records a list: serial and slot
    uint32_t s = 0;
    const uint64_t serial = marks.begin(&s);
    if (serial) {
      slot_of[serial] = s;
    }
    return serial;
  }
  void run(uint64_t serial) { words[slot_of[(size_t)serial]] = (uint32_t)serial; }   // the list executed (its last write)
  void poll() { marks.poll(words.data()); }
};

void own_list_proves_completion() {
  Gpu g;
  RetireList<Fake> l;
  Fake heap;
  const uint64_t a = g.record();
  int c = l.retire(&heap, {a}, g.marks, false);
  check(heap.releases == 0 && c == 0 && l.size() == 1, "heap used by list A (recorded, not run): not released");
  g.poll();
  c = l.pump(g.marks);
  check(heap.releases == 0 && l.size() == 1, "pump before A ran: not released");
  g.run(a);
  g.poll();
  c = l.pump(g.marks);
  check(heap.releases == 1 && c == 0 && l.empty(), "A wrote its own slot, pump: released once (not counted)");
  l.pump(g.marks);
  check(heap.releases == 1, "a later pump does not release it again");
}

void out_of_order_submission() {
  // The blocker of the #82 review: A is recorded first, B later; the game submits B first and it completes. A's
  // objects must NOT be released (a monotonic "marker >= serial" test would have released them).
  Gpu g;
  RetireList<Fake> l;
  Fake a_arena, a_heap, a_qh;
  const uint64_t a = g.record();   // instance X's frame, list A
  const uint64_t b = g.record();   // instance Y's frame, list B (recorded after A)
  check(a < b, "A recorded before B (smaller serial)");
  Fake* bufs[1] = {&a_arena};
  Fake* heaps[2] = {&a_heap, &a_qh};
  int c = wgrr_rt::retire_instance(l, bufs, 1, heaps, 2, std::vector<uint64_t>{a}, g.marks);   // X destroyed
  g.run(b);   // B submitted and completed first
  g.poll();
  c += l.pump(g.marks);
  check(g.marks.done(b) && !g.marks.done(a), "B complete, A not: completion is per list");
  check(c == 0 && a_arena.releases + a_heap.releases + a_qh.releases == 0 && l.size() == 3,
        "B completed first: A's buffer, descriptor heap and query heap NOT released");
  g.run(a);
  g.poll();
  c = l.pump(g.marks);
  check(c == 1 && a_arena.releases == 1 && a_heap.releases == 1 && a_qh.releases == 1 && l.empty(),
        "A completed: its three objects released once each, the buffer counted");

  // one instance with two lists in flight: the later one completing does not release the instance's objects
  Gpu h;
  RetireList<Fake> m;
  Fake arena;
  const uint64_t f1 = h.record(), f2 = h.record();
  m.retire(&arena, {f1, f2}, h.marks, true);
  h.run(f2);
  h.poll();
  m.pump(h.marks);
  check(arena.releases == 0, "instance with frames 1 and 2 in flight, frame 2 ran first: arena kept");
  h.run(f1);
  h.poll();
  check(m.pump(h.marks) == 1 && arena.releases == 1, "frame 1 ran too: arena released");
}

void never_submitted_list() {
  Gpu g(8);
  RetireList<Fake> l;
  Fake heap, buf;
  const uint64_t a = g.record();   // recorded, never submitted
  l.retire(&heap, {a}, g.marks, false);
  l.retire(&buf, {a}, g.marks, true);
  // many later lists run and complete; the pool cycles through the other slots
  bool slot_reused = false;
  for (int i = 0; i < 100; ++i) {
    const uint64_t s = g.record();
    if (!s) break;
    if (g.slot_of[(size_t)s] == g.slot_of[(size_t)a]) slot_reused = true;
    g.run(s);
    g.poll();
    l.pump(g.marks);
  }
  check(!slot_reused, "100 later lists: none was given the never-submitted list's marker slot");
  check(!g.marks.done(a) && heap.releases == 0 && buf.releases == 0 && l.size() == 2,
        "list never submitted, 100 later lists complete: its objects still kept");
  check(g.marks.pending() == 1, "only the never-submitted list is pending");
  const int c = l.release_all();   // device removed: nothing of it runs any more
  check(c == 1 && heap.releases == 1 && buf.releases == 1 && l.empty(), "device removed: released once each");
}

void slot_reuse_needs_the_holders_proof() {
  Gpu g(4);
  uint64_t s[4];
  for (uint64_t& v : s) v = g.record();
  check(s[3] != 0 && g.record() == 0, "4 slots, 4 lists in flight: no fifth list (nothing may be recorded)");
  g.run(s[3]);   // the LAST recorded list completes; the others not
  g.poll();
  const uint64_t n = g.record();
  check(n != 0 && g.slot_of[(size_t)n] == g.slot_of[(size_t)s[3]], "the next list takes the slot of the list that completed");
  g.poll();   // its slot still holds s[3]'s value: that proves nothing about the new holder
  check(!g.marks.done(n), "a stale value of the slot's previous holder does not prove the new list");
  check(!g.marks.done(s[0]) && !g.marks.done(s[1]) && !g.marks.done(s[2]), "the earlier lists are still in flight");
  // per-instance frame slots work the same way in the runtime: a slot is reused only when ITS list proved complete
  const uint64_t c = g.record();
  check(c == 0, "pool full again");
  g.marks.cancel(n);   // a list that recorded nothing gives its slot back and counts as complete
  check(g.marks.done(n) && g.record() != 0, "cancel: serial complete, slot free again");
}

void device_switch_keeps_old_objects() {
  // set-device with another device while a frame of the old one is recorded but not complete (the #82 review's major)
  ParkedSet<Fake> ps;
  Gpu old;
  const uint64_t a = old.record();
  Fake heap, arena, pso, marker, device;
  Parked<Fake> p;
  p.id = 1;
  p.marks = old.marks;   // the runtime moves its state; the fake GPU keeps a copy of the slot table
  p.words = old.words.data();
  p.deferred.retire(&heap, {a}, p.marks, false);
  p.deferred.retire(&arena, {a}, p.marks, true);
  p.deferred.retire(&pso, {a}, p.marks, false);
  p.last = {&marker, &device};
  p.last_counted = 1;
  std::vector<std::string> log;
  bool removed = false;
  auto is_removed = [&](Parked<Fake>&) { return removed; };
  auto no_sleep = [](uint32_t) {};
  int c = ps.park(std::move(p), is_removed, no_sleep, 0, &log);
  log.clear();
  c += ps.pump(is_removed, &log);
  check(c == 0 && heap.releases + arena.releases + pso.releases + marker.releases + device.releases == 0 && ps.size() == 1,
        "device switched with list A in flight: the old device's objects, marker buffer and device are all kept");
  old.run(a);   // the old device's queue runs A later
  c = ps.pump(is_removed, &log);
  check(c == 2 && heap.releases == 1 && arena.releases == 1 && pso.releases == 1 && marker.releases == 1 &&
            device.releases == 1 && ps.empty(),
        "A completed on the old device: every kept object released once (arena + marker counted), then the device");
  check(log.size() == 1, "one log line when the old device is released");

  // the old device is removed instead: everything goes
  Gpu old2;
  const uint64_t b = old2.record();
  Fake h2, m2, d2;
  Parked<Fake> q;
  q.id = 2;
  q.marks = old2.marks;
  q.words = old2.words.data();
  q.deferred.retire(&h2, {b}, q.marks, false);
  q.last = {&m2, &d2};
  q.last_counted = 1;
  c = ps.park(std::move(q), is_removed, no_sleep, 0, &log);
  c += ps.pump(is_removed, &log);
  check(c == 0 && h2.releases == 0, "second old device, list in flight, not removed: kept");
  removed = true;
  c = ps.pump(is_removed, &log);
  check(c == 1 && h2.releases == 1 && m2.releases == 1 && d2.releases == 1 && ps.empty(),
        "old device removed (GetDeviceRemovedReason): its objects released once each");
}

void wraparound_stale_marker() {
  // The #82 final review: the GPU writes only the low 32 bits of a serial and a free slot keeps its last value. A slot
  // that stays free while the serials go round 2^32 must not let that stale value prove its next holder.
  Gpu g(2);
  const uint64_t x = g.record();   // slot 0
  g.run(x);
  g.poll();                        // x complete: slot 0 free, still holding (uint32_t)x
  const uint64_t y = g.record();   // slot 1, kept in flight so that the next list goes to slot 0
  check(g.slot_of[(size_t)y] == 1 && !g.marks.done(y), "list Y holds slot 1, in flight");
  g.marks.skip_to(x + 0xffffffffull);   // the next serial would be x + 2^32: the same low 32 bits as slot 0's value
  const uint64_t z = g.record();
  check(z != 0 && g.slot_of[(size_t)z] == 0, "the next list takes slot 0 (the slot with the stale value)");
  check((uint32_t)z != (uint32_t)x && z > x + 0xffffffffull,
        "after 2^32 serials: the serial matching slot 0's stale value was skipped");
  g.poll();   // Z has not run; slot 0 still holds x's value
  check(!g.marks.done(z), "the stale value of slot 0 does not prove the new list (never submitted)");
  g.run(z);
  g.poll();
  check(g.marks.done(z) && !g.marks.done(y), "Z completes when it writes its own value; Y still in flight");

  // a fresh slot reads 0: a serial whose low 32 bits are 0 is never handed to it
  Gpu h(4);
  h.marks.skip_to(0xffffffffull);
  const uint64_t w = h.record();
  check(w != 0 && (uint32_t)w != 0 && h.words[h.slot_of[(size_t)w]] == 0, "a fresh slot never gets a serial ending in 32 zero bits");
  h.poll();
  check(!h.marks.done(w), "a fresh slot's 0 does not prove its first list");
}

// one old device with a list in flight (never completing on its own) and one counted object
struct OldDevice {
  Gpu gpu;
  uint64_t list = 0;
  Fake obj, marker, device;
  Parked<Fake> make(uint64_t id) {
    list = gpu.record();
    Parked<Fake> p;
    p.id = id;
    p.marks = gpu.marks;
    p.words = gpu.words.data();
    p.deferred.retire(&obj, {list}, p.marks, true);
    p.last = {&marker, &device};
    p.last_counted = 1;
    return p;
  }
  int releases() const { return obj.releases + marker.releases + device.releases; }
};

void device_switch_soft_cap() {
  // six device switches, each with a list still in flight on the device switched away from
  const size_t N = 6;
  std::vector<std::string> log;
  {   // none completes, none removed: all six stay tracked (the cap is soft), nothing released, each wait bounded
    ParkedSet<Fake> ps;
    std::vector<OldDevice> d(N);
    int sleeps = 0, c = 0;
    auto never = [](Parked<Fake>&) { return false; };
    auto count_sleep = [&](uint32_t ms) { sleeps += (int)ms; };
    for (size_t i = 0; i < N; ++i) c += ps.park(d[i].make(i + 1), never, count_sleep, 2000, &log);
    int rel = 0;
    for (const OldDevice& o : d) rel += o.releases();
    check(ps.size() == N && rel == 0 && c == 0,
          "6 switches with work pending, nothing completes: all 6 old devices stay tracked, nothing released");
    check(sleeps == 2 * 2000, "the 5th and 6th switch each waited at most 2000 ms (bounded) for the oldest");
    bool soft = false;
    for (const std::string& s : log) soft = soft || s.find("stays tracked") != std::string::npos;
    check(soft, "the soft cap is logged");
    for (OldDevice& o : d) o.gpu.run(o.list);   // later, every old device's list completes
    c = ps.pump(never, &log);
    bool once = true;
    for (const OldDevice& o : d) once = once && o.obj.releases == 1 && o.marker.releases == 1 && o.device.releases == 1;
    check(ps.empty() && once && c == (int)(2 * N), "their lists complete later: every object of all 6 released exactly once");
  }
  {   // the oldest one's list completes during the bounded wait of the 5th switch: it is released then
    ParkedSet<Fake> ps;
    std::vector<OldDevice> d(N);
    auto never = [](Parked<Fake>&) { return false; };
    int calls = 0;
    auto gpu_runs_oldest = [&](uint32_t) { if (++calls == 3) d[0].gpu.run(d[0].list); };
    int c = 0;
    for (size_t i = 0; i < 5; ++i) c += ps.park(d[i].make(i + 1), never, gpu_runs_oldest, 2000, &log);
    check(ps.size() == 4 && d[0].obj.releases == 1 && d[0].marker.releases == 1 && d[0].device.releases == 1 && c == 2,
          "5th switch: the oldest's list completes within the wait, it is released once (4 tracked)");
    check(calls == 3, "the wait ended as soon as the oldest completed");
    int rel = 0;
    for (size_t i = 1; i < 5; ++i) rel += d[i].releases();
    check(rel == 0, "the other old devices are untouched");
  }
  {   // the oldest one's device is removed: released at the 5th switch without waiting
    ParkedSet<Fake> ps;
    std::vector<OldDevice> d(N);
    int sleeps = 0;
    auto oldest_removed = [](Parked<Fake>& p) { return p.id == 1; };
    auto count_sleep = [&](uint32_t ms) { sleeps += (int)ms; };
    for (size_t i = 0; i < 4; ++i) ps.park(d[i].make(i + 1), [](Parked<Fake>&) { return false; }, count_sleep, 2000, &log);
    const int c = ps.park(d[4].make(5), oldest_removed, count_sleep, 2000, &log);
    check(ps.size() == 4 && d[0].releases() == 3 && c == 2 && sleeps == 0,
          "5th switch, the oldest's device removed: released at once, no wait");
  }
}

}  // namespace

int main() {
  own_list_proves_completion();
  out_of_order_submission();
  never_submitted_list();
  slot_reuse_needs_the_holders_proof();
  device_switch_keeps_old_objects();
  wraparound_stale_marker();
  device_switch_soft_cap();
  printf("%s (%d of %d checks failed)\n", g_failed ? "FAILED" : "passed", g_failed, g_checks);
  return g_failed ? 1 : 0;
}
