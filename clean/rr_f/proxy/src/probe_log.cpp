// SPDX-License-Identifier: MIT
// Probe-build log (see probe_log.h).
#include "probe_log.h"

namespace wgrr {

namespace {
const char kCapLine[] = "probe log: size cap reached, logging stopped";
}

ProbeLog::ProbeLog(Sink sink) : sink_(std::move(sink)) {}
ProbeLog::ProbeLog(Sink sink, const Limits& limits) : sink_(std::move(sink)), limits_(limits) {}

void ProbeLog::next_frame() {
  std::lock_guard<std::mutex> lock(mu_);
  ++frame_;
}

uint64_t ProbeLog::frame() const {
  std::lock_guard<std::mutex> lock(mu_);
  return frame_;
}

bool ProbeLog::full_window() const {
  std::lock_guard<std::mutex> lock(mu_);
  return frame_ <= limits_.full_frames;  // frames 1..full_frames (and the calls before the first frame)
}

bool ProbeLog::write_locked(const std::string& line) {
  if (capped_) return false;
  const uint64_t reserve = sizeof kCapLine + 2;
  if (bytes_ + line.size() + 2 + reserve > limits_.cap_bytes) {
    capped_ = true;
    bytes_ += sizeof kCapLine + 1;
    if (sink_) sink_(kCapLine);
    return false;
  }
  bytes_ += line.size() + 2;  // line break as written by the sink
  ++written_;
  if (sink_) sink_(line);
  return true;
}

bool ProbeLog::record(const std::string& key, const std::string& stable, const std::string& extra) {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = last_.find(key);
  const bool changed = it == last_.end() || it->second != stable;
  if (frame_ > limits_.full_frames && !changed) {
    ++suppressed_;
    return false;
  }
  if (it == last_.end()) last_.emplace(key, stable);
  else if (changed) it->second = stable;
  std::string line = "f=" + std::to_string(frame_) + " " + key + ": " + stable;
  if (!extra.empty()) line += " ; " + extra;
  return write_locked(line);
}

bool ProbeLog::note(const std::string& text) {
  std::lock_guard<std::mutex> lock(mu_);
  return write_locked("f=" + std::to_string(frame_) + " " + text);
}

uint64_t ProbeLog::bytes() const {
  std::lock_guard<std::mutex> lock(mu_);
  return bytes_;
}
bool ProbeLog::capped() const {
  std::lock_guard<std::mutex> lock(mu_);
  return capped_;
}
uint64_t ProbeLog::written() const {
  std::lock_guard<std::mutex> lock(mu_);
  return written_;
}
uint64_t ProbeLog::suppressed() const {
  std::lock_guard<std::mutex> lock(mu_);
  return suppressed_;
}

}  // namespace wgrr
