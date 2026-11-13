// SPDX-License-Identifier: MIT
// Files built into the runtime library: the release code objects and data (clean/runner/build.py writes the table,
// scrambled so the files are not stored as plain copies inside the library). Every read of a code object or data file
// goes through these two functions: the file on disk when it exists (development builds, in-game A/B payloads with
// extra code objects next to the library), else the built-in file with the same name.
#pragma once
#include <string>
#include <vector>

bool wg_read_file(const std::string& path, std::vector<char>* out);
bool wg_file_exists(const std::string& path);
