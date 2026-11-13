// SPDX-License-Identifier: MIT
// WrongGPU RR runtime: objects retired while a frame recorded into the game's command list may still use them
// (issue #63). Original WrongGPU code.
//
// The rule (the same for the RR runtime, the front end and Model M's executor):
//   1. an object a recorded frame may use is released only when that frame's completion is proven (here: the
//      runtime-wide frame marker has reached the frame's serial), never because a wait timed out;
//   2. until then the object stays on this list; when the device is dropped or removed nothing of it runs any more,
//      and everything left is released then (at process exit the system reclaims the rest);
//   3. no caller waits here: retiring and pumping only compare serials.
//
// Header-only and free of D3D12 so that the CPU test (clean/rr_f/runtime/tests/test_rr_retire.py) builds it with a
// fake object on any host. T is any type with Release() (IUnknown in the runtime); "counted" marks the objects the
// runtime counts in its live-resource total (the buffers and textures it made, not its heaps), and the functions
// return how many counted objects they released so the caller can keep that total.
#pragma once

#include <stddef.h>
#include <stdint.h>

#include <vector>

namespace wgrr_rt {

template <class T>
class RetireList {
 public:
  struct Entry {
    uint64_t serial;   // the last frame that may use obj
    T* obj;
    bool counted;
  };

  // Hands obj over: released at once when the frame `serial` has completed (done >= serial), else kept until a
  // pump() sees it completed. Returns the number of counted objects released now (0 or 1). A null obj is ignored.
  int retire(T* obj, uint64_t serial, uint64_t done, bool counted) {
    if (!obj) return 0;
    if (serial <= done) {
      obj->Release();
      return counted ? 1 : 0;
    }
    entries_.push_back({serial, obj, counted});
    return 0;
  }

  // Releases every object whose frame has completed. Returns the number of counted objects released.
  int pump(uint64_t done) {
    int counted = 0;
    size_t keep = 0;
    for (size_t i = 0; i < entries_.size(); ++i) {
      Entry& e = entries_[i];
      if (e.serial <= done) {
        e.obj->Release();
        if (e.counted) ++counted;
      } else {
        entries_[keep++] = e;
      }
    }
    entries_.resize(keep);
    return counted;
  }

  // The device is gone (dropped, or removed: its queues run nothing any more): releases everything left. Returns the
  // number of counted objects released.
  int release_all() { return pump(UINT64_MAX); }

  // true when an object retired with `serial` would still be kept (its frame has not completed)
  static bool in_flight(uint64_t serial, uint64_t done) { return serial > done; }

  size_t size() const { return entries_.size(); }
  bool empty() const { return entries_.empty(); }

 private:
  std::vector<Entry> entries_;
};

// Instance destroy: the objects the instance's recorded frames may use. Buffers (counted) go to the list with the
// instance's last serial; the shader-visible descriptor heap and the timestamp query heap are released here (the
// caller has waited for the serial, bounded). Returns the number of counted objects released now.
template <class T>
int retire_instance(RetireList<T>& l, T* const* buffers, size_t nb, T* const* heaps, size_t nh, uint64_t serial,
                    uint64_t done) {
  int counted = 0;
  for (size_t i = 0; i < nb; ++i) counted += l.retire(buffers[i], serial, done, true);
  for (size_t i = 0; i < nh; ++i)
    if (heaps[i]) heaps[i]->Release();
  return counted;
}

}  // namespace wgrr_rt
