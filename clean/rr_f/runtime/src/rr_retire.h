// SPDX-License-Identifier: MIT
// WrongGPU RR runtime: objects retired while command lists recorded by the runtime may still use them (issue #63).
// Original WrongGPU code.
//
// Completion is proven PER LIST. Every command list the runtime records into (a frame, a pass-through, a transient
// copy) gets a marker slot of its own and ends with a GPU write of its serial into that slot. A list is complete only
// when its own slot holds its serial. A later list's marker proves nothing about an earlier one: the game decides the
// submission order, and recording order is not submission order (a list recorded first can be submitted last, on
// another queue, or never). A slot is handed to a new list only after the list that held it proved complete, so a
// slot's value always belongs to its current holder.
//
// The rule (the same for the RR runtime, the front end and Model M's executor):
//   1. an object is released only when every list that may use it has proved complete, never because a wait timed
//      out, and never because some other list completed;
//   2. until then it stays on the list; when its device is removed nothing of it runs any more and everything left is
//      released then (at process exit the system reclaims the rest);
//   3. no caller waits here: retiring and pumping only read the marker slots.
//
// A device switch (set-device with another device) keeps the old device's lists, marker slots and waiting objects
// (ParkedSet) until each list proved complete or the old device is removed. Nothing is forgotten: beyond kMaxParked old
// devices, the oldest one is released if its lists complete within a bounded wait (or its device is removed), and kept
// tracked (with a log line) otherwise, never released early.
//
// Header-only and free of D3D12 so that the CPU test (clean/rr_f/runtime/tests/test_rr_retire.py) builds it with fake
// objects and a fake marker buffer on any host. T is any type with Release() (IUnknown in the runtime); "counted" marks
// the objects the runtime counts in its live-resource total (the buffers and textures it made, not its heaps), and the
// functions return how many counted objects they released so the caller can keep that total.
#pragma once

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace wgrr_rt {

// The marker slots of one device: which list holds which slot, and which lists have proved complete.
class ListMarks {
 public:
  explicit ListMarks(uint32_t slots = 1024) : occ_(slots, 0), held_(slots, 0) {}

  // A new list: its serial (> 0), and in *slot the marker word it must write that serial into (the low 32 bits) as
  // its last command. 0 when every slot is held by a list that has not proved complete (nothing may be recorded then).
  // The GPU writes only the low 32 bits, and a free slot still holds the value of its last completed list (0 when
  // fresh): a serial whose low 32 bits equal the value already in the chosen slot is skipped (never handed out), so
  // a stale value can never prove the new list, however many serials have gone by (32-bit wraparound).
  uint64_t begin(uint32_t* slot) {
    const uint32_t n = (uint32_t)occ_.size();
    for (uint32_t i = 0; i < n; ++i) {
      const uint32_t s = (hint_ + i) % n;
      if (occ_[s]) continue;
      ++last_;
      while ((uint32_t)last_ == held_[s]) ++last_;   // differs from the slot's current value (0 for a fresh slot)
      occ_[s] = last_;
      pending_.emplace(last_, s);
      hint_ = (s + 1) % n;
      *slot = s;
      return last_;
    }
    return 0;
  }

  // A list that recorded nothing after all (the caller failed before recording any command into it): its slot is free
  // again and the serial counts as complete. Never for a list that recorded commands.
  void cancel(uint64_t serial) {
    auto it = pending_.find(serial);
    if (it == pending_.end()) return;
    occ_[it->second] = 0;
    pending_.erase(it);
  }

  // Reads the marker slots (the device's mapped marker words): a list whose slot holds its serial is complete.
  void poll(const volatile uint32_t* words) {
    if (!words) return;
    for (auto it = pending_.begin(); it != pending_.end();) {
      if (words[it->second] == (uint32_t)it->first) {
        occ_[it->second] = 0;
        held_[it->second] = (uint32_t)it->first;   // what the free slot holds now
        it = pending_.erase(it);
      } else {
        ++it;
      }
    }
  }

  // true when list `serial` proved complete (as of the last poll), or was cancelled; serial 0 = no list.
  bool done(uint64_t serial) const { return serial == 0 || (serial <= last_ && pending_.find(serial) == pending_.end()); }

  size_t pending() const { return pending_.size(); }
  uint64_t last() const { return last_; }
  uint32_t slots() const { return (uint32_t)occ_.size(); }
  // Test hook (the wraparound test): the next serial handed out is above `serial`.
  void skip_to(uint64_t serial) {
    if (serial > last_) last_ = serial;
  }
  // the serials of every list that has not proved complete
  std::vector<uint64_t> pending_serials() const {
    std::vector<uint64_t> v;
    v.reserve(pending_.size());
    for (const auto& kv : pending_) v.push_back(kv.first);
    std::sort(v.begin(), v.end());
    return v;
  }

 private:
  std::vector<uint64_t> occ_;                       // the serial holding each slot, 0 = free
  std::vector<uint32_t> held_;                      // the value each slot holds as far as proven (0 = fresh)
  std::unordered_map<uint64_t, uint32_t> pending_;  // serial -> slot of every list not proved complete
  uint64_t last_ = 0;
  uint32_t hint_ = 0;
};

// Drops the serials of lists that have proved complete (users lists stay short).
inline void prune_done(std::vector<uint64_t>& users, const ListMarks& m) {
  users.erase(std::remove_if(users.begin(), users.end(), [&](uint64_t s) { return m.done(s); }), users.end());
}

template <class T>
class RetireList {
 public:
  struct Entry {
    std::vector<uint64_t> users;   // the lists that may use obj and have not proved complete
    T* obj;
    bool counted;
  };

  // Hands obj over: released at once when every list in `users` has proved complete, else kept until a pump() sees
  // them all complete. Returns the number of counted objects released now (0 or 1). A null obj is ignored.
  int retire(T* obj, std::vector<uint64_t> users, const ListMarks& m, bool counted) {
    if (!obj) return 0;
    prune_done(users, m);
    if (users.empty()) {
      obj->Release();
      return counted ? 1 : 0;
    }
    entries_.push_back({std::move(users), obj, counted});
    return 0;
  }

  // Releases every object whose lists have all proved complete. Returns the number of counted objects released.
  int pump(const ListMarks& m) {
    int counted = 0;
    size_t keep = 0;
    for (size_t i = 0; i < entries_.size(); ++i) {
      Entry& e = entries_[i];
      prune_done(e.users, m);
      if (e.users.empty()) {
        e.obj->Release();
        if (e.counted) ++counted;
      } else {
        if (keep != i) entries_[keep] = std::move(e);
        ++keep;
      }
    }
    entries_.resize(keep);
    return counted;
  }

  // The device is removed (its queues run nothing any more): releases everything left. Returns the number of counted
  // objects released.
  int release_all() {
    int counted = 0;
    for (Entry& e : entries_) {
      e.obj->Release();
      if (e.counted) ++counted;
    }
    entries_.clear();
    return counted;
  }

  size_t size() const { return entries_.size(); }
  bool empty() const { return entries_.empty(); }

 private:
  std::vector<Entry> entries_;
};

// Instance destroy: every object the instance's recorded lists may use goes to the list with the instance's lists not
// yet proved complete, the shader-visible descriptor heap and the timestamp query heap (the game's list dispatches with
// the one and runs EndQuery / ResolveQueryData on the other) exactly as the buffers; only the buffers are counted.
// Nothing waits. Returns the number of counted objects released now.
template <class T>
int retire_instance(RetireList<T>& l, T* const* buffers, size_t nb, T* const* heaps, size_t nh,
                    const std::vector<uint64_t>& users, const ListMarks& m) {
  int counted = 0;
  for (size_t i = 0; i < nb; ++i) counted += l.retire(buffers[i], users, m, true);
  for (size_t i = 0; i < nh; ++i) counted += l.retire(heaps[i], users, m, false);
  return counted;
}

// An old device after a device switch: its lists' marker slots, the objects waiting on them, and what goes last (the
// marker buffer the lists write, then the device reference) once every list proved complete or the device is removed.
template <class T>
struct Parked {
  uint64_t id = 0;                         // device generation, for the log
  ListMarks marks;
  RetireList<T> deferred;
  const volatile uint32_t* words = nullptr;   // the old device's marker slots (mapped; owned by `last`)
  std::vector<T*> last;                    // released after everything else, in order
  int last_counted = 0;                    // how many of `last` count as live resources
};

template <class T>
class ParkedSet {
 public:
  static constexpr size_t kMaxParked = 4;     // old devices tracked before park() tries to release the oldest
  static constexpr uint32_t kParkWaitMs = 2000;   // the bounded wait for the oldest one's lists at that point

  // Takes an old device's state; log lines go to *log. Nothing is ever forgotten: an old device leaves the set only
  // when it is released (every list proved complete, or the device removed). With more than kMaxParked tracked, the
  // oldest one is released at once when its marker slots show every list complete or removed(p) reports its device
  // removed; otherwise its marker slots are polled for at most wait_ms (sleep_ms(1) between polls) and it is released
  // when they complete; if it is still pending after that, it stays tracked (the cap is soft) and one line says so.
  // Returns the number of counted objects released.
  template <class Removed, class Sleep>
  int park(Parked<T>&& p, Removed removed, Sleep sleep_ms, uint32_t wait_ms, std::vector<std::string>* log) {
    if (log)
      log->push_back("device generation " + std::to_string(p.id) + " switched away with " +
                     std::to_string(p.marks.pending()) + " list(s) not proved complete: " +
                     std::to_string(p.deferred.size() + p.last.size()) +
                     " object(s) kept until each list completes or the device is removed");
    set_.push_back(std::move(p));
    int counted = 0;
    if (set_.size() <= kMaxParked) return counted;
    if (try_release(0, removed, log, &counted)) return counted;
    for (uint32_t t = 0; t < wait_ms; ++t) {
      sleep_ms(1);
      if (try_release(0, removed, log, &counted)) return counted;
    }
    if (log)
      log->push_back("more than " + std::to_string(kMaxParked) + " old devices kept: generation " +
                     std::to_string(set_.front().id) + " still has " + std::to_string(set_.front().marks.pending()) +
                     " list(s) not proved complete after " + std::to_string(wait_ms) +
                     " ms; it stays tracked (" + std::to_string(set_.size()) + " old devices tracked now)");
    return counted;
  }

  // Polls each old device's marker slots and releases what has proved complete; removed(p) tells whether p's device
  // is removed (then everything of it is released). An old device whose lists all proved complete is released in full.
  // Returns the number of counted objects released; log lines go to *log.
  template <class Removed>
  int pump(Removed removed, std::vector<std::string>* log) {
    int counted = 0;
    for (size_t i = 0; i < set_.size();) {
      Parked<T>& p = set_[i];
      p.marks.poll(p.words);
      counted += p.deferred.pump(p.marks);
      if (!try_release(i, removed, log, &counted)) ++i;
    }
    return counted;
  }

  size_t size() const { return set_.size(); }
  bool empty() const { return set_.empty(); }
  size_t objects() const {
    size_t n = 0;
    for (const Parked<T>& p : set_) n += p.deferred.size() + p.last.size();
    return n;
  }

 private:
  // Releases old device i in full (and removes it from the set) when every one of its lists proved complete or its
  // device is removed. True when released.
  template <class Removed>
  bool try_release(size_t i, Removed& removed, std::vector<std::string>* log, int* counted) {
    Parked<T>& p = set_[i];
    p.marks.poll(p.words);
    const bool gone = p.marks.pending() != 0 && removed(p);
    if (!gone && p.marks.pending() != 0) return false;
    const size_t n = p.deferred.size() + p.last.size();
    *counted += p.deferred.pump(p.marks);   // every object whose lists all proved complete
    *counted += p.deferred.release_all();   // the rest: only when the device is removed (else nothing is left)
    for (T* o : p.last)
      if (o) o->Release();
    *counted += p.last_counted;
    if (log)
      log->push_back("device generation " + std::to_string(p.id) +
                     (gone ? ": device removed, " : ": every list proved complete, ") + std::to_string(n) +
                     " kept object(s) released");
    set_.erase(set_.begin() + (ptrdiff_t)i);
    return true;
  }

  std::vector<Parked<T>> set_;
};

}  // namespace wgrr_rt
