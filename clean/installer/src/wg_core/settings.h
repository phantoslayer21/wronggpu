// SPDX-License-Identifier: MIT
// WrongGPU launcher core: wronggpu.ini: reading and changing keys, the defaults of earlier versions. The per-game
// profile of build/launcher-design.md section 4 comes here (M1). Written for this project.
#pragma once
#include "context.h"

namespace wg {

struct StaleDefault { const wchar_t* upTo; const char* key; const char* old; const char* now; const wchar_t* why; };

string IniSet(const string& text, const string& section, const string& key, const string& value);
bool IniGetFrontEnd(const string& text, const string& key, string* value, string* section);
bool IniHasKey(const string& text, const string& section, const string& key, string* value = nullptr);
string IniRemoveFrontEnd(const string& text, const string& key);  // every line of `key` in the front end's sections
bool VersionAtMost(const wstring& v, const wstring& limit);
void MigrateSettings(Context& ctx, const wstring& dir, const wstring& oldVersion, Ui& ui);
}  // namespace wg
