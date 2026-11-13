// SPDX-License-Identifier: MIT
#include "../runtime/library_handle.h"

#include <cstddef>
#include <cstdio>

namespace {
int closed[4] = {};
void fake_close(void* handle) {
  const auto i = reinterpret_cast<std::size_t>(handle);
  if (i < 4) ++closed[i];
}
int failures = 0;
void check(bool ok, const char* name) {
  if (!ok) {
    std::printf("FAIL %s\n", name);
    ++failures;
  }
}
}  // namespace

int main() {
  // HIP init can fail after loading the DLL but before the device is ready. Scope exit must still balance the load.
  {
    wgrt::LibraryHandle library(&fake_close);
    check(!library, "empty handle starts empty");
    library.reset(reinterpret_cast<void*>(1));
    check(library.get() == reinterpret_cast<void*>(1), "acquired handle is retained");
  }
  check(closed[1] == 1, "failed initialization releases its DLL reference");

  // Context recreation must close the previous per-context reference exactly once before adopting a new one.
  {
    wgrt::LibraryHandle library(&fake_close);
    library.reset(reinterpret_cast<void*>(2));
    library.reset(reinterpret_cast<void*>(3));
    check(closed[2] == 1, "replacement releases old reference once");
    library.reset();
    check(closed[3] == 1, "explicit empty reset releases current reference once");
  }
  check(closed[2] == 1 && closed[3] == 1, "destructor does not double-release cleared handle");
  std::printf("LibraryHandle: %s\n", failures ? "FAIL" : "PASS");
  return failures ? 1 : 0;
}
