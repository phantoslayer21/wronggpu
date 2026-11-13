// SPDX-License-Identifier: MIT
// A fake WrongGPU RR runtime behind the adapter of include/wg_rr_adapter.h: records every call, does no GPU work.
#pragma once
#include <stdint.h>
#include <string.h>

#include <functional>
#include <map>
#include <vector>

#include "../include/wg_rr_adapter.h"

struct FakeRuntime {
  uint32_t status_bits = WG_RR_RT_READY | WG_RR_RT_MODEL_LOADED;
  uint32_t evaluate_rc = 0;
  uint32_t create_rc = 0;
  WgRrFrameResult next_result{};       // fields copied into the result of evaluate (size kept)
  std::function<void()> on_evaluate;   // runs inside evaluate (settings snapshot tests)
  std::vector<uint8_t> supported_luid = {1, 2, 3, 4, 5, 6, 7, 8};
  std::map<void*, std::pair<uint32_t, uint32_t>> described;  // resource -> (format, samples)
  std::map<void*, std::pair<uint32_t, uint32_t>> sized;      // resource -> (width, height) (describe_resource_size)
  int passthroughs = 0;
  uint32_t passthrough_rc = 31;                              // the fail-safe copy: not recorded unless set to 0
  void* device = nullptr;
  int creates = 0, destroys = 0, evaluates = 0, retains = 0, releases = 0;
  std::vector<WgRrCreate> created;
  std::vector<WgRrFrame> frames;
  std::vector<void*> live;  // live instance handles
  void* last_transient_cmd = nullptr;
  int next_handle = 1;
  WgRrRuntimeV1 table{};

  FakeRuntime() {
    table.size = sizeof table;
    table.version = WG_RR_ADAPTER_VERSION;
    table.ctx = this;
    table.status = [](void* c) { return self(c)->status_bits; };
    table.adapter_supported = [](void* c, const void* luid, uint32_t n) -> uint32_t {
      FakeRuntime* r = self(c);
      return n == r->supported_luid.size() && memcmp(luid, r->supported_luid.data(), n) == 0;
    };
    table.release_version = [](void*, uint32_t out[3]) { out[0] = 0; out[1] = 5; out[2] = 0; };
    table.set_device = [](void* c, void* d) -> uint32_t { self(c)->device = d; return 0; };
    table.create_instance = [](void* c, const WgRrCreate* d, void** out) -> uint32_t {
      FakeRuntime* r = self(c);
      if (r->create_rc) return r->create_rc;
      r->creates++;
      r->created.push_back(*d);
      *out = reinterpret_cast<void*>((uintptr_t)(0x1000 * r->next_handle++));
      r->live.push_back(*out);
      return 0;
    };
    table.destroy_instance = [](void* c, void* h) {
      FakeRuntime* r = self(c);
      r->destroys++;
      for (size_t i = 0; i < r->live.size(); ++i)
        if (r->live[i] == h) { r->live.erase(r->live.begin() + i); break; }
    };
    table.instance_vram = [](void*, void* h) -> uint64_t { return 1000000ull + (uintptr_t)h; };
    table.evaluate = [](void* c, void*, const WgRrFrame* f, WgRrFrameResult* res) -> uint32_t {
      FakeRuntime* r = self(c);
      r->evaluates++;
      r->frames.push_back(*f);
      if (r->on_evaluate) r->on_evaluate();
      uint32_t size = res->size;
      *res = r->next_result;
      res->size = size;
      if (r->evaluate_rc == 0) { res->passes_started = 1; res->input_jitter_consumed = 1; }
      return r->evaluate_rc;
    };
    table.retain_transient = [](void* c, void* cmd, void* res, uint32_t, void** out) -> uint32_t {
      FakeRuntime* r = self(c);
      r->retains++;
      r->last_transient_cmd = cmd;
      *out = reinterpret_cast<uint8_t*>(res) + 0x7000;
      return 0;
    };
    table.release_transient = [](void* c, void*) { self(c)->releases++; };
    table.describe_resource = [](void* c, void* res, uint32_t* fmt, uint32_t* samples) -> uint32_t {
      FakeRuntime* r = self(c);
      auto it = r->described.find(res);
      if (it == r->described.end()) return 31;
      *fmt = it->second.first;
      *samples = it->second.second;
      return 0;
    };
    table.describe_resource_size = [](void* c, void* res, uint32_t* w, uint32_t* h) -> uint32_t {
      FakeRuntime* r = self(c);
      auto it = r->sized.find(res);
      if (it == r->sized.end()) return 31;
      *w = it->second.first;
      *h = it->second.second;
      return 0;
    };
    table.passthrough = [](void* c, const WgRrFrame*) -> uint32_t {
      FakeRuntime* r = self(c);
      if (r->passthrough_rc == 0) r->passthroughs++;
      return r->passthrough_rc;
    };
  }
  static FakeRuntime* self(void* c) { return static_cast<FakeRuntime*>(c); }
  const WgRrFrame& last() const { return frames.back(); }
};
