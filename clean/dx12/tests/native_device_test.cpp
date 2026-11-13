// SPDX-License-Identifier: MIT
// Original Device contract tests: native D3D12 allocations, copies, fill, events, binding and teardown.
// Reuses our synthetic device_test.hlsl. No model weights, HIP API or third-party implementation.
#include "../../runtime/device.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace wgrt;
static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { std::printf("FAIL: " __VA_ARGS__); std::printf("\n"); ++fails; } } while (0)
#define NEED(c, ...) do { if (!(c)) { std::printf("FAIL: " __VA_ARGS__); std::printf("\n"); return 1; } } while (0)

static int exercise(const char* build, unsigned cycle) {
  std::string err;
  auto d = make_d3d12_device(build, &err);
  NEED(d, "create cycle %u: %s", cycle, err.c_str());
  std::printf("cycle %u device: %s\nnote: %s\n", cycle, d->name().c_str(), d->note().c_str());
  const uint32_t n = cycle ? 259u : 66000u * 64u + 17u;
  const size_t bytes = size_t(n) * sizeof(uint32_t);
  DevPtr a = 0, b = 0;
  NEED(d->alloc(&a, bytes) && d->alloc(&b, bytes), "alloc: %s", d->err().c_str());
  std::vector<uint32_t> input(n), got(n);
  for (uint32_t i = 0; i < n; ++i) input[i] = i * 7u + cycle;
  NEED(d->upload(a, input.data(), bytes), "upload: %s", d->err().c_str());
  NEED(d->reserve_events(8) && d->record(0), "record start: %s", d->err().c_str());
  uint32_t count = n, k = 1;
  void* add[] = {&a, &count, &k};
  NEED(d->launch("wg_test_add", (n + 63) / 64, 64, add), "add: %s", d->err().c_str());
  k = 2;
  NEED(d->launch("wg_test_add", (n + 63) / 64, 64, add), "add2: %s", d->err().c_str());
  struct { uint32_t n, k; } params = {n, 5};
  DevPtr null = 0;
  void* shift[] = {&a, &b, &params, &null};
  NEED(d->launch("tst/wg_test_shift", (n + 63) / 64, 64, shift) && d->record(1),
       "shift and stop: %s", d->err().c_str());
  NEED(d->download(got.data(), b, bytes), "download: %s", d->err().c_str());
  bool identical = true;
  for (uint32_t i = 0; i < n; ++i) identical = identical && got[i] == input[(i + 1) % n] + 8;
  CHECK(identical, "consecutive dispatches, null and struct args, split grid");
  float ms = -1;
  CHECK(d->elapsed_ms(0, 1, &ms) && std::isfinite(ms) && ms >= 0, "elapsed timestamp: %s", d->err().c_str());
  CHECK(d->event_done(0) == 1 && d->event_done(1) == 1, "completed events");
  CHECK(d->elapsed_done_ms(0, 1, &ms) && std::isfinite(ms) && ms >= 0, "elapsed completed timestamp");
  CHECK(!d->record(-1), "negative event accepted");
  CHECK(!d->elapsed_ms(0, 7, &ms), "unrecorded timestamp accepted");
  CHECK(!d->elapsed_ms(0, 1, nullptr), "null timestamp destination accepted");
  const float previous_ms = ms;
  CHECK(d->reserve_events(130) && d->elapsed_done_ms(0, 1, &ms) && ms == previous_ms,
        "growing the timestamp pool did not preserve completed events");

  // Byte fill tests include zero, all alignments, a single byte and a multi-workgroup interval.
  std::vector<unsigned char> expected(bytes), observed(bytes);
  for (size_t i = 0; i < bytes; ++i) expected[i] = static_cast<unsigned char>(i * 13 + 1);
  NEED(d->upload(a, expected.data(), bytes), "upload byte pattern");
  const size_t offsets[] = {0, 1, 2, 3, 7, 1023};
  const size_t lengths[] = {0, 1, 2, 3, 1013, 9};
  for (unsigned j = 0; j < 6; ++j) {
    const size_t off = offsets[j], len = lengths[j];
    NEED(d->memset(a + off, 0xa0 + j, len), "unaligned fill %u: %s", j, d->err().c_str());
    std::fill(expected.begin() + off, expected.begin() + off + len, static_cast<unsigned char>(0xa0 + j));
  }
  NEED(d->download(observed.data(), a, bytes), "download fills");
  CHECK(observed == expected, "byte fills changed neighboring bytes or missed range");
  NEED(d->copy(b, a, bytes) && d->download(observed.data(), b, bytes), "cross-allocation copy");
  CHECK(observed == expected, "cross-allocation copy values");
  void* host = d->alloc_host(bytes);
  NEED(host, "alloc_host: %s", d->err().c_str());
  NEED(d->record(128) && d->download_async(host, b, bytes) && d->record(129) && d->sync(),
       "ordered asynchronous readback: %s", d->err().c_str());
  CHECK(std::memcmp(host, expected.data(), bytes) == 0, "asynchronous readback values");
  CHECK(d->elapsed_done_ms(128, 129, &ms) && std::isfinite(ms) && ms >= 0, "grown timestamp pool");
  CHECK(!d->download_async(observed.data(), b, bytes), "unowned asynchronous host destination accepted");
  d->free_host(host);
  // Copy contract: overlap can be rejected, but if accepted it must have memmove semantics.
  for (bool forward : {true, false}) {
    NEED(d->upload(a, expected.data(), bytes), "restore before overlap");
    auto moved = expected;
    const size_t dst = forward ? 3 : 0, src = forward ? 0 : 3;
    std::memmove(moved.data() + dst, moved.data() + src, bytes - 3);
    const bool copied = d->copy(a + dst, a + src, bytes - 3);
    NEED(d->download(observed.data(), a, bytes), "download overlap");
    CHECK(observed == (copied ? moved : expected), "overlap neither memmove nor clean rejection");
    std::printf("overlap %s: %s\n", forward ? "forward" : "reverse", copied ? "memmove" : "explicitly rejected");
  }
  CHECK(d->copy(a, a, bytes), "identity copy");
  DevPtr foreign = 16, past = a + bytes;
  void* bad_ptr[] = {&foreign, &count, &k};
  void* past_ptr[] = {&past, &count, &k};
  CHECK(!d->launch("wg_test_add", 1, 64, bad_ptr), "foreign pointer accepted");
  CHECK(!d->launch("wg_test_add", 1, 64, past_ptr), "one-past allocation pointer accepted");
  CHECK(!d->launch("wg_test_add", 1, 128, add), "wrong block size accepted");
  CHECK(!d->launch("unknown_kernel", 1, 64, add), "unknown kernel accepted");
  CHECK(!d->launch("wg_test_add", 1, 64, nullptr), "null argument array accepted");
  CHECK(!d->upload(a + bytes - 1, expected.data(), 2), "upload overrun accepted");
  CHECK(!d->download(observed.data(), a + bytes - 1, 2), "download overrun accepted");
  CHECK(!d->copy(b, a + bytes - 1, 2), "copy overrun accepted");
  CHECK(!d->memset(a + bytes - 1, 0, 2), "fill overrun accepted");
  CHECK(!d->upload(a, nullptr, 1), "null upload source accepted");
  CHECK(!d->download(nullptr, a, 1), "null download destination accepted");

  // Exact module binding must reject the wrong variant before inspecting its shorter argument array.
  NEED(d->load_module_tagged("post_w64__orig.co", "bound"), "bind original variant");
  NEED(d->launch("bound/wg_test_add", (n + 63) / 64, 64, add), "matching module binding");
  const bool wrong_loaded = d->load_module_tagged("post_w64.co", "bound");
  if (wrong_loaded) CHECK(!d->launch("bound/wg_test_add", 1, 64, nullptr), "mismatched code object accepted");
  std::printf("unsupported variant: %s\n", d->err().c_str());
  CHECK(!d->err().empty(), "unsupported variant had no diagnostic");
  NEED(d->sync(), "sync after errors: %s", d->err().c_str());
  d->free(a);
  CHECK(!d->memset(a, 0, 4), "freed allocation accepted");
  d->free(b);
  // Destroy with work outstanding: implementation must finish before releasing GPU resources.
  DevPtr pending = 0;
  NEED(d->alloc(&pending, 1024) && d->memset(pending, 0x72, 1024), "pending teardown work");
  return 0;
}

int main(int argc, char** argv) {
  if (argc != 2) { std::printf("usage: native_device_test <private test build>\n"); return 2; }
  for (unsigned cycle = 0; cycle < 5; ++cycle) if (exercise(argv[1], cycle)) return 1;
  std::printf("native_device_test: %s\n", fails ? "FAILED" : "all checks passed");
  return fails ? 1 : 0;
}
