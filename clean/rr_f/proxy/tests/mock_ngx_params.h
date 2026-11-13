// SPDX-License-Identifier: MIT
// A mock of the game's NGX parameter block as the host hands it to the RR provider (wg_host_menu_abi.h,
// WgHostNgxParams): typed values by key. Getters follow the host's documented rules: false when the key is not set
// (output unchanged); an unsigned getter also answers an int key and the other way round (unless `strict`: then each
// getter answers its own type only, so the provider's own int / uint fallback is tested); the float getter answers
// float keys only. Counts every read so tests can check that the provider only reads (there is no setter).
#pragma once
#include <stdint.h>

#include <map>
#include <string>

#include "../include/wg_host_menu_abi.h"

struct MockNgxParams {
  std::map<std::string, void*> res, ptr;
  std::map<std::string, uint32_t> u;
  std::map<std::string, int32_t> i;
  std::map<std::string, float> f;
  int reads = 0;
  bool strict = false;  // each integer getter answers keys of its own type only

  void set_res(const std::string& k, void* r) { res[k] = r; }
  void set_ptr(const std::string& k, void* p) { ptr[k] = p; }
  void set_u(const std::string& k, uint32_t v) { u[k] = v; }
  void set_i(const std::string& k, int32_t v) { i[k] = v; }
  void set_f(const std::string& k, float v) { f[k] = v; }

  static MockNgxParams* self(void* b) { return static_cast<MockNgxParams*>(b); }
  WgHostNgxParams view() {
    WgHostNgxParams p;
    p.block = this;
    p.get_resource = [](void* b, const char* k, void** out) -> bool {
      MockNgxParams* m = self(b);
      ++m->reads;
      auto it = m->res.find(k);
      if (it == m->res.end() || !it->second) return false;
      *out = it->second;
      return true;
    };
    p.get_pointer = [](void* b, const char* k, void** out) -> bool {
      MockNgxParams* m = self(b);
      ++m->reads;
      auto it = m->ptr.find(k);
      if (it == m->ptr.end()) return false;
      *out = it->second;
      return true;
    };
    p.get_uint = [](void* b, const char* k, uint32_t* out) -> bool {
      MockNgxParams* m = self(b);
      ++m->reads;
      auto it = m->u.find(k);
      if (it != m->u.end()) { *out = it->second; return true; }
      if (m->strict) return false;
      auto jt = m->i.find(k);
      if (jt != m->i.end()) { *out = (uint32_t)jt->second; return true; }
      return false;
    };
    p.get_int = [](void* b, const char* k, int32_t* out) -> bool {
      MockNgxParams* m = self(b);
      ++m->reads;
      auto it = m->i.find(k);
      if (it != m->i.end()) { *out = it->second; return true; }
      if (m->strict) return false;
      auto jt = m->u.find(k);
      if (jt != m->u.end()) { *out = (int32_t)jt->second; return true; }
      return false;
    };
    p.get_float = [](void* b, const char* k, float* out) -> bool {
      MockNgxParams* m = self(b);
      ++m->reads;
      auto it = m->f.find(k);
      if (it == m->f.end()) return false;
      *out = it->second;
      return true;
    };
    return p;
  }
};
