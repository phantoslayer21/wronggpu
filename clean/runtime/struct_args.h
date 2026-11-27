// SPDX-License-Identifier: GPL-3.0-only
// Struct arguments of the D3D12 kernel ports with pointers inside (manifest argument kind "s", clean/dx12/build_dx12.py).
// WrongGPU's own code.
//
// A HIP kernel that takes a struct by value with 64-bit device pointers in it (model K's KBlkParams, KInParams,
// KOutParams) cannot be passed to its D3D12 port as plain bytes ("b<N>"): a D3D12 port addresses the arena by 32-bit
// byte offsets, and the hybrid device's pointers are HIP addresses. The argument token
//
//     s<N>:<field>.<field>...      e.g. s184:p19.b32 (KBlkParams: 27 constants),
//                                  s312:p4.b12.x24.b52.x12.b24.x4.p4.x8.p4.b8.p2.b56 (KOutParams without cbx..zby,
//                                  kappa / rho / lam, pad_ and exp2tab: 52 constants instead of 78)
//
// describes the N bytes of the struct as consecutive fields: p<n> = n pointers (8 bytes each; each becomes ONE root
// constant, the arena byte offset, 0 = null), b<n> = n bytes copied as they are (n a multiple of 4), x<n> = n bytes the
// port does not read (skipped: no root constant). The fields must add up to exactly N bytes. The port's HLSL struct
// is the compacted layout: one uint per pointer, the copied bytes, nothing for the skipped ones.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace wgrt {

struct StructField {
  char kind;       // 'p', 'b' or 'x'
  unsigned count;  // pointers ('p') or bytes ('b', 'x')
};

// Parses an "s<N>:<fields>" token. *bytes = N (the HIP struct's size), *dwords = the root constants it takes.
inline bool parse_struct_arg(const std::string& token, unsigned* bytes, unsigned* dwords,
                             std::vector<StructField>* fields) {
  fields->clear();
  const size_t colon = token.find(':');
  if (token.size() < 4 || token[0] != 's' || colon == std::string::npos || colon < 2) return false;
  auto number = [](const std::string& s, size_t a, size_t b, unsigned* v) {
    if (a >= b || b - a > 6) return false;
    unsigned n = 0;
    for (size_t i = a; i < b; ++i) {
      if (s[i] < '0' || s[i] > '9') return false;
      n = n * 10 + (unsigned)(s[i] - '0');
    }
    *v = n;
    return n > 0;
  };
  unsigned total = 0;
  if (!number(token, 1, colon, &total)) return false;
  unsigned seen = 0, words = 0;
  for (size_t a = colon + 1; a <= token.size();) {
    size_t b = token.find('.', a);
    if (b == std::string::npos) b = token.size();
    if (b <= a + 1) return false;
    const char kind = token[a];
    unsigned n = 0;
    if ((kind != 'p' && kind != 'b' && kind != 'x') || !number(token, a + 1, b, &n)) return false;
    if (kind == 'p') {
      seen += 8 * n;
      words += n;
    } else {
      if (n % 4) return false;
      seen += n;
      if (kind == 'b') words += n / 4;
    }
    fields->push_back({kind, n});
    a = b + 1;
  }
  if (seen != total || fields->empty()) return false;
  *bytes = total;
  *dwords = words;
  return true;
}

// Packs one struct argument (src: the HIP struct's bytes) into root constants. translate(pointer, &offset) turns a
// non-null device pointer into its arena byte offset (false: not an arena pointer). Returns the constants written, or
// -1 when a pointer could not be translated.
template <class Translate>
int pack_struct_arg(const std::vector<StructField>& fields, const void* src, uint32_t* out, Translate translate) {
  const unsigned char* s = static_cast<const unsigned char*>(src);
  int d = 0;
  for (const StructField& f : fields) {
    if (f.kind == 'p') {
      for (unsigned i = 0; i < f.count; ++i, s += 8) {
        uint64_t p = 0;
        for (int k = 0; k < 8; ++k) p |= (uint64_t)s[k] << (8 * k);
        uint32_t off = 0;
        if (p && !translate(p, &off)) return -1;
        out[d++] = p ? off : 0u;
      }
    } else if (f.kind == 'b') {
      for (unsigned i = 0; i < f.count / 4; ++i, s += 4)
        out[d++] = (uint32_t)s[0] | (uint32_t)s[1] << 8 | (uint32_t)s[2] << 16 | (uint32_t)s[3] << 24;
    } else {
      s += f.count;
    }
  }
  return d;
}

}  // namespace wgrt
