// SPDX-License-Identifier: MIT
// Log of the probe build (WG_RR_PROBE; never part of the release build): names and numbers only, no image data.
// Every record has a key (what it is about), a stable part (compared) and an extra part (written, never compared:
// sequence numbers, thread ids, frame indices, jitter). During the first `full_frames` frames every record is written;
// afterwards a record is written only when its stable part differs from the last one written or seen for its key.
// The log stops at `cap_bytes` (one final line says so). Thread-safe. Platform-free: the DLL supplies the sink.
#pragma once
#include <stdint.h>

#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>

namespace wgrr {

class ProbeLog {
 public:
  using Sink = std::function<void(const std::string& line)>;  // one line, without the line break
  struct Limits {
    uint32_t full_frames = 600;          // frames logged in full
    uint64_t cap_bytes = 20ull << 20;    // 20 MB
  };

  explicit ProbeLog(Sink sink);
  ProbeLog(Sink sink, const Limits& limits);

  // A new frame begins (the probe calls this at every <new-frame-token>); frames are numbered from 1, the calls
  // before the first frame are frame 0. Frames 0..full_frames are logged in full.
  void next_frame();
  uint64_t frame() const;
  bool full_window() const;

  // Writes "f=<frame> <key>: <stable>[ ; <extra>]" subject to the rules above. Returns true if written.
  bool record(const std::string& key, const std::string& stable, const std::string& extra = std::string());
  // Always written (until the cap): start / stop markers and diagnostics.
  bool note(const std::string& text);

  uint64_t bytes() const;
  bool capped() const;
  uint64_t written() const;
  uint64_t suppressed() const;

 private:
  bool write_locked(const std::string& line);

  Sink sink_;
  Limits limits_;
  mutable std::mutex mu_;
  uint64_t frame_ = 0;
  uint64_t bytes_ = 0;
  uint64_t written_ = 0, suppressed_ = 0;
  bool capped_ = false;
  std::unordered_map<std::string, std::string> last_;
};

}  // namespace wgrr
