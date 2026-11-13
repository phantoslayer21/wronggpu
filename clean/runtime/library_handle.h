// SPDX-License-Identifier: MIT
// Small move-free owner for a platform library reference. Kept separate so its acquire/release contract can be tested
// without initializing a GPU runtime.
#pragma once

namespace wgrt {

class LibraryHandle {
 public:
  using Close = void (*)(void*);

  explicit LibraryHandle(Close close) : close_(close) {}
  LibraryHandle(const LibraryHandle&) = delete;
  LibraryHandle& operator=(const LibraryHandle&) = delete;
  ~LibraryHandle() { reset(); }

  void reset(void* next = nullptr) {
    if (handle_ && close_) close_(handle_);
    handle_ = next;
  }
  void* get() const { return handle_; }
  explicit operator bool() const { return handle_ != nullptr; }

 private:
  Close close_ = nullptr;
  void* handle_ = nullptr;
};

}  // namespace wgrt
