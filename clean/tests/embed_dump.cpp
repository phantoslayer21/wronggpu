// SPDX-License-Identifier: MIT
// Test harness for the runtime library's built-in file table (clean/runtime/embedded.cpp + a generated
// embedded_files.cpp, clean/tests/test_embedded_table.py): walks the table and reads every file through the runtime's
// own lookup, wg_read_file, from a folder without that file (so the built-in copy is returned), and writes the bytes to
// <out>/<name>. Prints the file count first.
//   embed_dump <empty folder> <out folder>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

#include "embedded.h"

namespace wg_embedded {
struct File {
  const char* name;
  const unsigned char* data;
  size_t size;
};
extern const File kFiles[];
extern const size_t kCount;
}  // namespace wg_embedded

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: embed_dump <empty folder> <out folder>\n");
    return 2;
  }
  const std::string empty = argv[1], out = argv[2];
  std::printf("count %zu\n", wg_embedded::kCount);
  for (size_t i = 0; i < wg_embedded::kCount; ++i) {
    const std::string name = wg_embedded::kFiles[i].name;
    std::vector<char> data;
    if (!wg_read_file(empty + "/" + name, &data)) {
      std::fprintf(stderr, "lookup failed: %s\n", name.c_str());
      return 1;
    }
    std::FILE* fp = std::fopen((out + "/" + name).c_str(), "wb");
    if (!fp) return 1;
    if (!data.empty() && std::fwrite(data.data(), 1, data.size(), fp) != data.size()) return 1;
    std::fclose(fp);
    std::printf("%s %zu\n", name.c_str(), data.size());
  }
  return 0;
}
