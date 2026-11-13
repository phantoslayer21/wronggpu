// SPDX-License-Identifier: MIT
// WrongGPU launcher core: a small JSON writer (owned-files.json, `status --json`, `--out` results). Writing only; no
// third-party code. Written for this project.
#pragma once
#include "util.h"

namespace wg {

// A JSON string literal (UTF-8, quotes included).
inline string JsonStr(const wstring& w) {
    const string s = ToUtf8(w);
    string o = "\"";
    for (unsigned char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
                else o += (char)c;
        }
    }
    return o + "\"";
}
inline string JsonStr(const string& s) { return JsonStr(FromUtf8(s)); }
inline string JsonStr(const wchar_t* s) { return JsonStr(wstring(s ? s : L"")); }

// Builds one JSON value with indentation: Obj/Arr open a container, Key names the next member, End closes.
class Json {
public:
    Json& Obj() { open('{'); return *this; }
    Json& Arr() { open('['); return *this; }
    Json& End() {
        const bool empty = first_.back();
        first_.pop_back();
        const char c = stack_.back(); stack_.pop_back();
        if (!empty) { out_ += "\n"; indent(); }
        out_ += c == '{' ? '}' : ']';
        return *this;
    }
    Json& Key(const char* k) { sep(); out_ += JsonStr(string(k)) + ": "; pendingKey_ = true; return *this; }
    Json& Str(const wstring& v) { value(JsonStr(v)); return *this; }
    Json& Str(const string& v) { value(JsonStr(v)); return *this; }
    Json& Str(const wchar_t* v) { value(JsonStr(v)); return *this; }
    Json& Num(long long v) { value(std::to_string(v)); return *this; }
    Json& Bool(bool v) { value(v ? "true" : "false"); return *this; }
    Json& Null() { value("null"); return *this; }
    // shorthands for object members
    Json& S(const char* k, const wstring& v) { return Key(k).Str(v); }
    Json& S(const char* k, const string& v) { return Key(k).Str(v); }
    Json& S(const char* k, const wchar_t* v) { return Key(k).Str(v); }
    Json& N(const char* k, long long v) { return Key(k).Num(v); }
    Json& B(const char* k, bool v) { return Key(k).Bool(v); }
    const string& str() const { return out_; }

private:
    void indent() { out_.append(stack_.size() * 2, ' '); }
    void sep() {
        if (stack_.empty()) return;
        if (!first_.back()) out_ += ",";
        first_.back() = false;
        out_ += "\n";
        indent();
    }
    void open(char c) {
        if (pendingKey_) pendingKey_ = false; else sep();
        out_ += c;
        stack_.push_back(c);
        first_.push_back(true);
    }
    void value(const string& v) {
        if (pendingKey_) pendingKey_ = false; else sep();
        out_ += v;
    }
    string out_;
    std::vector<char> stack_;
    std::vector<bool> first_;
    bool pendingKey_ = false;
};

}  // namespace wg
