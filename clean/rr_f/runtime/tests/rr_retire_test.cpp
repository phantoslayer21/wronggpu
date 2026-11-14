// SPDX-License-Identifier: MIT
// CPU test of the RR runtime's retire list (src/rr_retire.h, issue #63): objects a recorded frame may use are released
// only when the frame marker proves that frame complete. Fake objects count their Release calls; the marker is a
// plain number. Built and run by test_rr_retire.py; exit status 0 when every check passes. Original WrongGPU code.
#include <stdio.h>

#include "rr_retire.h"

namespace {

struct Fake {
  int releases = 0;
  unsigned long Release() { return (unsigned long)++releases; }
};

int g_failed = 0;
void check(bool ok, const char* what) {
  printf("%s: %s\n", ok ? "ok  " : "FAIL", what);
  if (!ok) ++g_failed;
}

void heap_waits_for_marker() {
  wgrr_rt::RetireList<Fake> l;
  Fake heap;
  int c = l.retire(&heap, 5, 4, false);
  check(heap.releases == 0 && c == 0 && l.size() == 1, "heap retired at serial 5 with the marker at 4: not released");
  c = l.pump(4);
  check(heap.releases == 0 && c == 0 && l.size() == 1, "pump with the marker still at 4: not released");
  c = l.pump(5);
  check(heap.releases == 1 && c == 0 && l.empty(), "marker 5, pump: released once (not counted)");
  l.pump(6);
  check(heap.releases == 1, "a later pump does not release it again");
}

void completed_frame_releases_at_once() {
  wgrr_rt::RetireList<Fake> l;
  Fake buf;
  const int c = l.retire(&buf, 5, 5, true);
  check(buf.releases == 1 && c == 1 && l.empty(), "buffer retired at serial 5 with the marker at 5: released at once, counted");
  check(l.retire(nullptr, 9, 0, true) == 0 && l.empty(), "a null object is ignored");
}

void destroy_keeps_everything_until_complete() {
  // the instance destroy of issue #63: arena, upload, query readback, two output proxies (counted) and the
  // shader-visible descriptor heap and timestamp query heap, all used by the frame with serial 5
  wgrr_rt::RetireList<Fake> l;
  Fake arena, upload, qrb, proxy, heap, qh;
  Fake* bufs[5] = {&arena, &upload, &qrb, &proxy, nullptr};
  Fake* heaps[2] = {&heap, &qh};
  int c = wgrr_rt::retire_instance(l, bufs, 5, heaps, 2, 5, 4);
  check(c == 0, "destroy with the marker at 4: no counted object released");
  check(arena.releases + upload.releases + qrb.releases + proxy.releases == 0, "destroy with the marker at 4: no buffer released");
  check(heap.releases == 0, "destroy with the marker at 4: descriptor heap not released");
  check(qh.releases == 0, "destroy with the marker at 4: query heap not released");
  c = l.pump(5);
  check(c == 4, "marker 5, pump: the four buffers released and counted");
  check(arena.releases == 1 && upload.releases == 1 && qrb.releases == 1 && proxy.releases == 1 && heap.releases == 1 &&
            qh.releases == 1 && l.empty(),
        "marker 5, pump: every object released exactly once");
}

void destroy_after_completion_releases_at_once() {
  wgrr_rt::RetireList<Fake> l;
  Fake arena, heap, qh;
  Fake* bufs[1] = {&arena};
  Fake* heaps[2] = {&heap, &qh};
  const int c = wgrr_rt::retire_instance(l, bufs, 1, heaps, 2, 5, 7);
  check(c == 1 && arena.releases == 1 && heap.releases == 1 && qh.releases == 1 && l.empty(),
        "destroy with the frame complete: everything released at once");
}

void partial_pump_and_device_drop() {
  wgrr_rt::RetireList<Fake> l;
  Fake a, b, h;
  l.retire(&a, 3, 1, true);
  l.retire(&b, 8, 1, true);
  l.retire(&h, 8, 1, false);
  int c = l.pump(3);
  check(c == 1 && a.releases == 1 && b.releases == 0 && h.releases == 0 && l.size() == 2, "pump to 3 releases only serial 3");
  check(wgrr_rt::RetireList<Fake>::in_flight(8, 3) && !wgrr_rt::RetireList<Fake>::in_flight(3, 3), "in_flight compares serials");
  c = l.release_all();   // device dropped or removed: the frame can never run
  check(c == 1 && b.releases == 1 && h.releases == 1 && l.empty(), "release_all (device dropped / removed) releases the rest once");
}

}  // namespace

int main() {
  heap_waits_for_marker();
  completed_frame_releases_at_once();
  destroy_keeps_everything_until_complete();
  destroy_after_completion_releases_at_once();
  partial_pump_and_device_drop();
  printf("%s (%d failed)\n", g_failed ? "FAILED" : "passed", g_failed);
  return g_failed ? 1 : 0;
}
