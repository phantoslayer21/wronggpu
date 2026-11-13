// SPDX-License-Identifier: MIT
// A minimal JSON reader for the proxy tests (game profiles under tests/data). Strict enough for files written by
// tools/probe_log_to_profile.py: objects, arrays, strings with the common escapes, numbers, true / false / null.
#pragma once
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <utility>
#include <vector>

namespace mj {

struct Value {
  enum Kind { Null, Bool, Num, Str, Arr, Obj } kind = Null;
  bool b = false;
  double n = 0;
  std::string s;
  std::vector<Value> a;
  std::vector<std::pair<std::string, Value>> o;

  bool is_null() const { return kind == Null; }
  bool has(const char* k) const {
    for (const auto& kv : o)
      if (kv.first == k) return true;
    return false;
  }
  const Value& operator[](const char* k) const {
    static const Value none;
    for (const auto& kv : o)
      if (kv.first == k) return kv.second;
    return none;
  }
  const Value& operator[](size_t i) const {
    static const Value none;
    return i < a.size() ? a[i] : none;
  }
  const Value& operator[](int i) const { return (*this)[(size_t)(i < 0 ? a.size() : (size_t)i)]; }
  size_t size() const { return kind == Arr ? a.size() : o.size(); }
  int i() const { return (int)n; }
  uint32_t u() const { return (uint32_t)n; }
  double d() const { return n; }
  float f() const { return (float)n; }
};

class Parser {
 public:
  explicit Parser(const std::string& t) : t_(t) {}
  bool parse(Value* out) {
    ok_ = true;
    p_ = 0;
    *out = value();
    ws();
    return ok_ && p_ == t_.size();
  }

 private:
  void ws() {
    while (p_ < t_.size() && (t_[p_] == ' ' || t_[p_] == '\n' || t_[p_] == '\r' || t_[p_] == '\t')) ++p_;
  }
  bool eat(char c) {
    ws();
    if (p_ < t_.size() && t_[p_] == c) { ++p_; return true; }
    return false;
  }
  std::string str() {
    std::string r;
    if (!eat('"')) { ok_ = false; return r; }
    while (p_ < t_.size() && t_[p_] != '"') {
      char c = t_[p_++];
      if (c == '\\' && p_ < t_.size()) {
        char e = t_[p_++];
        switch (e) {
          case 'n': r += '\n'; break;
          case 't': r += '\t'; break;
          case 'r': r += '\r'; break;
          case 'b': r += '\b'; break;
          case 'f': r += '\f'; break;
          case 'u': {  // ASCII only (the profiles hold no other characters)
            unsigned v = (unsigned)strtoul(t_.substr(p_, 4).c_str(), nullptr, 16);
            p_ += 4;
            r += (char)(v < 128 ? v : '?');
            break;
          }
          default: r += e;
        }
      } else {
        r += c;
      }
    }
    if (p_ >= t_.size()) ok_ = false;
    ++p_;
    return r;
  }
  Value value() {
    Value v;
    ws();
    if (p_ >= t_.size()) { ok_ = false; return v; }
    char c = t_[p_];
    if (c == '{') {
      ++p_;
      v.kind = Value::Obj;
      if (eat('}')) return v;
      do {
        std::string k = str();
        if (!eat(':')) { ok_ = false; return v; }
        v.o.emplace_back(k, value());
      } while (ok_ && eat(','));
      if (!eat('}')) ok_ = false;
    } else if (c == '[') {
      ++p_;
      v.kind = Value::Arr;
      if (eat(']')) return v;
      do {
        v.a.push_back(value());
      } while (ok_ && eat(','));
      if (!eat(']')) ok_ = false;
    } else if (c == '"') {
      v.kind = Value::Str;
      v.s = str();
    } else if (t_.compare(p_, 4, "true") == 0) {
      v.kind = Value::Bool; v.b = true; p_ += 4;
    } else if (t_.compare(p_, 5, "false") == 0) {
      v.kind = Value::Bool; p_ += 5;
    } else if (t_.compare(p_, 4, "null") == 0) {
      p_ += 4;
    } else {
      char* end = nullptr;
      v.kind = Value::Num;
      v.n = strtod(t_.c_str() + p_, &end);
      if (end == t_.c_str() + p_) ok_ = false;
      p_ = (size_t)(end - t_.c_str());
    }
    return v;
  }
  const std::string& t_;
  size_t p_ = 0;
  bool ok_ = true;
};

inline bool load(const char* path, Value* out) {
  FILE* f = fopen(path, "rb");
  if (!f) return false;
  std::string t;
  char buf[65536];
  size_t n;
  while ((n = fread(buf, 1, sizeof buf, f)) > 0) t.append(buf, n);
  fclose(f);
  return Parser(t).parse(out);
}

}  // namespace mj
