// SPDX-License-Identifier: MIT
// Linux: the Windows registry as a small file-backed store (see win32_posix.h).
//
// HKEY_CURRENT_USER lives in $WRONGGPU_REGISTRY_FILE, else $XDG_DATA_HOME/WrongGPU/registry.ini (default ~/.local/share). A
// WRONGGPU_TESTING build keeps it in memory unless WRONGGPU_REGISTRY_FILE is set: a test run never writes the player's file.
// Hives opened with RegLoadAppKeyW are files of their own. HKEY_LOCAL_MACHINE, HKEY_USERS and HKEY_CLASSES_ROOT read as empty
// and refuse writes. HKCU\Software\Valve\Steam\SteamPath, when nothing was stored there, is the Linux Steam folder.
#include <sys/stat.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <map>
#include <memory>
#include <unordered_set>

#include "win32_posix_internal.h"

using namespace wgposix;

namespace {

struct Value { std::string name; DWORD type = REG_NONE; std::string data; };   // strings: UTF-8 text; the rest: raw bytes
struct Key { std::string name; std::map<std::string, Value> values; };         // values by lower-case name

std::string LowerAscii(std::string s) { for (auto& c : s) if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a'); return s; }

std::string Escape(const std::string& s) {
    std::string o;
    for (unsigned char c : s) {
        if (c < 0x20 || c == '%' || c == '=' || c == ']') { char b[4]; std::snprintf(b, sizeof(b), "%%%02X", c); o += b; }
        else o += (char)c;
    }
    return o;
}
std::string Unescape(const std::string& s) {
    std::string o;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '%' && i + 2 < s.size() && std::isxdigit((unsigned char)s[i + 1]) && std::isxdigit((unsigned char)s[i + 2])) {
            o += (char)std::strtol(s.substr(i + 1, 2).c_str(), nullptr, 16);
            i += 2;
        } else o += s[i];
    }
    return o;
}
std::string Hex(const std::string& d) { std::string o; char b[3]; for (unsigned char c : d) { std::snprintf(b, sizeof(b), "%02x", c); o += b; } return o; }
std::string Unhex(const std::string& h) { std::string o; for (size_t i = 0; i + 1 < h.size(); i += 2) o += (char)std::strtol(h.substr(i, 2).c_str(), nullptr, 16); return o; }
bool IsString(DWORD t) { return t == REG_SZ || t == REG_EXPAND_SZ; }

struct Store {
    std::mutex mu;
    std::string file;               // "" = memory only
    bool readOnly = false, loaded = false;
    std::map<std::string, Key> keys;   // by lower-case path "a\b\c"; "" is the root

    void Load() {
        if (loaded) return;
        loaded = true;
        keys[""];
        if (file.empty()) return;
        FILE* f = std::fopen(file.c_str(), "r");
        if (!f) return;
        std::string cur;
        bool have = false;
        char line[16384];
        while (std::fgets(line, sizeof(line), f)) {
            std::string l = line;
            while (!l.empty() && (l.back() == '\n' || l.back() == '\r')) l.pop_back();
            if (l.size() >= 2 && l.front() == '[' && l.back() == ']') {
                const std::string path = Unescape(l.substr(1, l.size() - 2));
                have = true;
                Make(path);
                cur = LowerAscii(path);
            } else if (have) {
                const size_t eq = l.find('=');
                if (eq == std::string::npos || l.size() < eq + 3 || l[eq + 2] != ':') continue;
                Value v;
                const std::string rawName = l.substr(0, eq), body = l.substr(eq + 3);
                v.name = rawName == "@" ? "" : Unescape(rawName);
                switch (l[eq + 1]) {
                    case 's': v.type = REG_SZ; v.data = Unescape(body); break;
                    case 'e': v.type = REG_EXPAND_SZ; v.data = Unescape(body); break;
                    case 'd': { v.type = REG_DWORD; const uint32_t n = (uint32_t)std::strtoul(body.c_str(), nullptr, 10); v.data.assign(reinterpret_cast<const char*>(&n), 4); break; }
                    case 'q': { v.type = REG_QWORD; const uint64_t n = std::strtoull(body.c_str(), nullptr, 10); v.data.assign(reinterpret_cast<const char*>(&n), 8); break; }
                    case 'b': v.type = REG_BINARY; v.data = Unhex(body); break;
                    case 'm': v.type = REG_MULTI_SZ; v.data = Unhex(body); break;
                    default: continue;
                }
                keys[cur].values[LowerAscii(v.name)] = v;
            }
        }
        std::fclose(f);
    }
    // the key and every key above it
    void Make(const std::string& path) {
        std::string lower;
        size_t i = 0;
        while (i <= path.size()) {
            size_t j = path.find('\\', i);
            if (j == std::string::npos) j = path.size();
            const std::string part = path.substr(i, j - i);
            if (!part.empty()) {
                lower += (lower.empty() ? "" : "\\") + LowerAscii(part);
                Key& k = keys[lower];
                if (k.name.empty()) k.name = part;
            }
            i = j + 1;
        }
    }
    // True when the file now holds the store (or the store has no file); `err` is the errno of the failure otherwise.
    bool Save(int& err) {
        err = 0;
        if (file.empty() || readOnly) return true;
        const size_t cut = file.find_last_of('/');
        if (cut != std::string::npos && cut > 0) {
            std::string dir;
            for (size_t i = 1; i <= cut; ++i) if (i == cut || file[i] == '/') { dir = file.substr(0, i); mkdir(dir.c_str(), 0777); }
        }
        const std::string tmp = file + ".write";
        FILE* f = std::fopen(tmp.c_str(), "w");
        if (!f) { err = errno; return false; }
        bool ok = true;
        auto put = [&](int n) { if (n < 0) ok = false; };
        std::map<std::string, std::string> shown;   // lower path -> path as written
        for (auto& [lower, k] : keys) {
            if (lower.empty()) continue;
            const size_t cut2 = lower.rfind('\\');
            shown[lower] = (cut2 == std::string::npos ? std::string() : shown[lower.substr(0, cut2)] + "\\") + k.name;
        }
        for (auto& [lower, k] : keys) {
            if (lower.empty()) { if (k.values.empty()) continue; }
            put(std::fprintf(f, "[%s]\n", Escape(lower.empty() ? std::string() : shown[lower]).c_str()));
            for (auto& [vl, v] : k.values) {
                const std::string n = v.name.empty() ? "@" : Escape(v.name);
                switch (v.type) {
                    case REG_SZ: put(std::fprintf(f, "%s=s:%s\n", n.c_str(), Escape(v.data).c_str())); break;
                    case REG_EXPAND_SZ: put(std::fprintf(f, "%s=e:%s\n", n.c_str(), Escape(v.data).c_str())); break;
                    case REG_DWORD: { uint32_t x = 0; std::memcpy(&x, v.data.data(), std::min<size_t>(4, v.data.size())); put(std::fprintf(f, "%s=d:%u\n", n.c_str(), x)); break; }
                    case REG_QWORD: { uint64_t x = 0; std::memcpy(&x, v.data.data(), std::min<size_t>(8, v.data.size())); put(std::fprintf(f, "%s=q:%llu\n", n.c_str(), (unsigned long long)x)); break; }
                    case REG_MULTI_SZ: put(std::fprintf(f, "%s=m:%s\n", n.c_str(), Hex(v.data).c_str())); break;
                    default: put(std::fprintf(f, "%s=b:%s\n", n.c_str(), Hex(v.data).c_str())); break;
                }
            }
        }
        if (std::fflush(f) != 0 || ferror(f)) ok = false;
        if (ok && fsync(fileno(f)) != 0) ok = false;
        if (!ok) err = errno ? errno : EIO;
        if (std::fclose(f) != 0 && ok) { ok = false; err = errno ? errno : EIO; }
        if (ok && std::rename(tmp.c_str(), file.c_str()) != 0) { ok = false; err = errno; }
        if (!ok) std::remove(tmp.c_str());
        return ok;
    }
};
// the Windows error for a failed save
DWORD SaveError(int err) { return err == EACCES || err == EPERM || err == EROFS ? ERROR_ACCESS_DENIED : ERROR_WRITE_FAULT; }
// Write the store out; when that fails, put `before` back so memory keeps matching the file, and give the error.
LSTATUS SaveOrUndo(Store* s, std::map<std::string, Key>& before) {
    int err = 0;
    if (s->Save(err)) return ERROR_SUCCESS;
    s->keys = std::move(before);
    return SaveError(err);
}

std::string UserFile() {
    if (const char* e = std::getenv("WRONGGPU_REGISTRY_FILE")) if (*e) return e;
#ifdef WRONGGPU_TESTING
    return "";
#else
    return XdgDataHome() + "/WrongGPU/registry.ini";
#endif
}
Store& UserStore() { static Store s; static std::once_flag o; std::call_once(o, [] { s.file = UserFile(); }); return s; }
Store& ReadOnlyStore() { static Store s; static std::once_flag o; std::call_once(o, [] { s.readOnly = true; }); return s; }
std::mutex g_hiveLock;
std::map<std::string, std::unique_ptr<Store>> g_hives;

}  // namespace

struct HKEY__ { Store* store; std::string path; REGSAM sam; bool loadedRoot; };

namespace {
std::mutex g_keyLock;
std::unordered_set<HKEY> g_keys;

HKEY NewKey(Store* s, const std::string& path, REGSAM sam, bool root = false) {
    HKEY k = new HKEY__{s, path, sam, root};
    std::lock_guard<std::mutex> g(g_keyLock);
    g_keys.insert(k);
    return k;
}
struct Resolved { Store* store = nullptr; std::string path; REGSAM sam = KEY_ALL_ACCESS; };
bool Resolve(HKEY key, Resolved& r) {
    if (key == HKEY_CURRENT_USER) { r.store = &UserStore(); return true; }
    if (key == HKEY_LOCAL_MACHINE || key == HKEY_USERS || key == HKEY_CLASSES_ROOT) { r.store = &ReadOnlyStore(); return true; }
    std::lock_guard<std::mutex> g(g_keyLock);
    if (!g_keys.count(key)) return false;
    r.store = key->store; r.path = key->path; r.sam = key->sam;
    return true;
}
std::string Normalize(LPCWSTR sub) {
    std::string out;
    if (!sub) return out;
    const std::string s = ToUtf8(sub, wcslen(sub));
    size_t i = 0;
    while (i <= s.size()) {
        size_t j = s.find_first_of("\\/", i);
        if (j == std::string::npos) j = s.size();
        if (j > i) out += (out.empty() ? "" : "\\") + s.substr(i, j - i);
        i = j + 1;
    }
    return out;
}
std::string JoinPath(const std::string& base, const std::string& sub) { return base.empty() ? sub : sub.empty() ? base : base + "\\" + sub; }
bool SteamKey(Store* s, const std::string& lowerPath) { return s == &UserStore() && lowerPath == "software\\valve\\steam"; }
// the value at (key, name), the Steam folder included
bool FindValue(Store* s, const std::string& lowerPath, const std::string& name, Value& out) {
    auto k = s->keys.find(lowerPath);
    if (k != s->keys.end()) {
        auto v = k->second.values.find(LowerAscii(name));
        if (v != k->second.values.end()) { out = v->second; return true; }
    }
    if (SteamKey(s, lowerPath) && LowerAscii(name) == "steampath") {
        const std::string root = SteamRoot();
        if (!root.empty()) { out = Value{"SteamPath", REG_SZ, "Z:" + root}; return true; }   // Steam writes it with "/" as well
    }
    return false;
}
std::vector<std::string> Children(Store* s, const std::string& lowerPath) {
    std::vector<std::string> out;
    const std::string prefix = lowerPath.empty() ? std::string() : lowerPath + "\\";
    for (auto& [p, k] : s->keys) {
        if (p.size() <= prefix.size() || p.compare(0, prefix.size(), prefix) != 0 || p.find('\\', prefix.size()) != std::string::npos) continue;
        out.push_back(k.name);
    }
    return out;
}
}  // namespace

LSTATUS RegOpenKeyExW(HKEY key, LPCWSTR sub, DWORD, REGSAM sam, PHKEY out) {
    Resolved r;
    if (!Resolve(key, r)) return ERROR_INVALID_HANDLE;
    const std::string path = LowerAscii(JoinPath(r.path, Normalize(sub)));
    std::lock_guard<std::mutex> g(r.store->mu);
    r.store->Load();
    if (!r.store->keys.count(path) && !(SteamKey(r.store, path) && !SteamRoot().empty())) return ERROR_FILE_NOT_FOUND;
    if (r.store == &ReadOnlyStore() && !path.empty()) return ERROR_FILE_NOT_FOUND;
    *out = NewKey(r.store, path, sam);
    return ERROR_SUCCESS;
}
LSTATUS RegCreateKeyExW(HKEY key, LPCWSTR sub, DWORD, LPWSTR, DWORD, REGSAM sam, LPSECURITY_ATTRIBUTES, PHKEY out, LPDWORD disposition) {
    Resolved r;
    if (!Resolve(key, r)) return ERROR_INVALID_HANDLE;
    if (r.store->readOnly || !(r.sam & KEY_CREATE_SUB_KEY)) return ERROR_ACCESS_DENIED;
    const std::string shown = JoinPath(r.path, Normalize(sub)), path = LowerAscii(shown);
    std::lock_guard<std::mutex> g(r.store->mu);
    r.store->Load();
    const bool existed = r.store->keys.count(path) != 0;
    if (!existed) {
        // the key's own spelling for the new parts, the parent's for the rest (parents keep the names they were made with)
        std::map<std::string, Key> before = r.store->keys;
        r.store->Make(shown);
        const LSTATUS saved = SaveOrUndo(r.store, before);
        if (saved != ERROR_SUCCESS) return saved;
    }
    if (disposition) *disposition = existed ? REG_OPENED_EXISTING_KEY : REG_CREATED_NEW_KEY;
    *out = NewKey(r.store, path, sam);
    return ERROR_SUCCESS;
}
LSTATUS RegCloseKey(HKEY key) {
    if (key == HKEY_CURRENT_USER || key == HKEY_LOCAL_MACHINE || key == HKEY_USERS || key == HKEY_CLASSES_ROOT) return ERROR_SUCCESS;
    {
        std::lock_guard<std::mutex> g(g_keyLock);
        if (!g_keys.erase(key)) return ERROR_INVALID_HANDLE;
    }
    delete key;
    return ERROR_SUCCESS;
}
LSTATUS RegQueryValueExW(HKEY key, LPCWSTR name, LPDWORD, LPDWORD type, LPBYTE data, LPDWORD size) {
    Resolved r;
    if (!Resolve(key, r)) return ERROR_INVALID_HANDLE;
    if (!(r.sam & KEY_QUERY_VALUE)) return ERROR_ACCESS_DENIED;
    Value v;
    {
        std::lock_guard<std::mutex> g(r.store->mu);
        r.store->Load();
        if (!FindValue(r.store, r.path, name ? ToUtf8(name, wcslen(name)) : std::string(), v)) return ERROR_FILE_NOT_FOUND;
    }
    if (type) *type = v.type;
    std::wstring wide;
    const char* bytes = v.data.data();
    size_t n = v.data.size();
    if (IsString(v.type)) { wide = FromUtf8(v.data); bytes = reinterpret_cast<const char*>(wide.c_str()); n = (wide.size() + 1) * sizeof(wchar_t); }
    if (!data) { if (size) *size = (DWORD)n; return ERROR_SUCCESS; }
    if (!size || *size < n) { if (size) *size = (DWORD)n; return ERROR_MORE_DATA; }
    std::memcpy(data, bytes, n);
    *size = (DWORD)n;
    return ERROR_SUCCESS;
}
LSTATUS RegGetValueW(HKEY key, LPCWSTR sub, LPCWSTR name, DWORD flags, LPDWORD type, PVOID data, LPDWORD size) {
    HKEY k = key;
    if (sub && *sub) { const LSTATUS s = RegOpenKeyExW(key, sub, 0, KEY_READ, &k); if (s != ERROR_SUCCESS) return s; }
    DWORD t = 0, cb = size ? *size : 0;
    LSTATUS s = RegQueryValueExW(k, name, nullptr, &t, static_cast<LPBYTE>(data), size ? &cb : nullptr);
    if (sub && *sub) RegCloseKey(k);
    if (s != ERROR_SUCCESS && s != ERROR_MORE_DATA) return s;
    const bool allowed = (flags & RRF_RT_ANY) == RRF_RT_ANY || ((flags & RRF_RT_REG_SZ) && t == REG_SZ) || ((flags & RRF_RT_REG_DWORD) && t == REG_DWORD);
    if (!allowed) return ERROR_UNSUPPORTED_TYPE;
    if (type) *type = t;
    if (size) *size = cb;
    return s;
}
LSTATUS RegSetValueExW(HKEY key, LPCWSTR name, DWORD, DWORD type, const BYTE* data, DWORD size) {
    Resolved r;
    if (!Resolve(key, r)) return ERROR_INVALID_HANDLE;
    if (r.store->readOnly || !(r.sam & KEY_SET_VALUE)) return ERROR_ACCESS_DENIED;
    Value v;
    v.name = name ? ToUtf8(name, wcslen(name)) : std::string();
    v.type = type;
    if (IsString(type)) {
        // callers size a string as (length + 1) * 2, the UTF-16 size: that many bytes / 2 = characters of the wide string
        const wchar_t* w = reinterpret_cast<const wchar_t*>(data);
        size_t n = size / 2, len = 0;
        while (len < n && w[len]) ++len;
        v.data = ToUtf8(w, len);
    } else if (data) {
        v.data.assign(reinterpret_cast<const char*>(data), size);
    }
    std::lock_guard<std::mutex> g(r.store->mu);
    r.store->Load();
    std::map<std::string, Key> before = r.store->keys;
    r.store->keys[r.path].values[LowerAscii(v.name)] = v;
    return SaveOrUndo(r.store, before);
}
LSTATUS RegDeleteKeyW(HKEY key, LPCWSTR sub) {
    Resolved r;
    if (!Resolve(key, r)) return ERROR_INVALID_HANDLE;
    const std::string path = LowerAscii(JoinPath(r.path, Normalize(sub)));
    if (path.empty()) return ERROR_ACCESS_DENIED;
    std::lock_guard<std::mutex> g(r.store->mu);
    r.store->Load();
    if (!r.store->keys.count(path)) return ERROR_FILE_NOT_FOUND;
    if (r.store->readOnly) return ERROR_ACCESS_DENIED;
    if (!Children(r.store, path).empty()) return ERROR_ACCESS_DENIED;   // a key with subkeys stays, as on Windows
    std::map<std::string, Key> before = r.store->keys;
    r.store->keys.erase(path);
    return SaveOrUndo(r.store, before);
}
LSTATUS RegEnumKeyExW(HKEY key, DWORD index, LPWSTR name, LPDWORD nameLen, LPDWORD, LPWSTR, LPDWORD, FILETIME*) {
    Resolved r;
    if (!Resolve(key, r)) return ERROR_INVALID_HANDLE;
    if (!(r.sam & KEY_ENUMERATE_SUB_KEYS)) return ERROR_ACCESS_DENIED;
    std::lock_guard<std::mutex> g(r.store->mu);
    r.store->Load();
    const std::vector<std::string> kids = Children(r.store, r.path);
    if (index >= kids.size()) return ERROR_NO_MORE_ITEMS;
    const std::wstring w = FromUtf8(kids[index]);
    if (!nameLen || *nameLen < w.size() + 1) { if (nameLen) *nameLen = (DWORD)w.size() + 1; return ERROR_MORE_DATA; }
    std::memcpy(name, w.c_str(), (w.size() + 1) * sizeof(wchar_t));
    *nameLen = (DWORD)w.size();
    return ERROR_SUCCESS;
}
LSTATUS RegLoadAppKeyW(LPCWSTR file, PHKEY out, REGSAM sam, DWORD, DWORD) {
    std::string p;
    DWORD e = 0;
    if (!ToPosix(file, p, e)) return e;
    Store* s;
    {
        std::lock_guard<std::mutex> g(g_hiveLock);
        auto& slot = g_hives[p];
        if (!slot) { slot.reset(new Store()); slot->file = p; }
        s = slot.get();
    }
    std::lock_guard<std::mutex> g(s->mu);
    s->keys.clear();   // what the file holds now, not what an earlier load saw
    s->loaded = false;
    s->Load();
    int saveErr = 0;
    s->Save(saveErr);   // an app hive file exists once it is loaded, as on Windows; if it cannot be written, the first write says so
    *out = NewKey(s, "", sam, true);
    return ERROR_SUCCESS;
}
