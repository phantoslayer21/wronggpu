// SPDX-License-Identifier: MIT
// Minimal JSON reader (RFC 8259) for the converter map. Written for this project.
#include "json.h"

#include <set>

namespace wgc {

const Json* Json::find(const std::string& key) const {
    if (type != Type::Object) return nullptr;
    for (const auto& m : members) {
        if (m.first == key) return &m.second;
    }
    return nullptr;
}

bool Json::as_int64(std::int64_t& out) const {
    // Only plain integer literals are accepted ("12", "-3"); "1.0" or "1e3" are not integers here.
    if (type != Type::Number) return false;
    std::size_t i = 0;
    const bool neg = !text.empty() && text[0] == '-';
    if (neg) i = 1;
    if (i >= text.size() || text.size() - i > 19) return false;
    unsigned long long v = 0;
    for (; i < text.size(); ++i) {
        if (text[i] < '0' || text[i] > '9') return false;
        v = v * 10 + static_cast<unsigned>(text[i] - '0');
    }
    if (!neg && v > 9223372036854775807ULL) return false;
    if (neg && v > 9223372036854775808ULL) return false;
    out = neg ? static_cast<std::int64_t>(0 - v) : static_cast<std::int64_t>(v);
    return true;
}

namespace {

class Parser {
public:
    Parser(const std::string& s, std::string& err) : s_(s), err_(err) {}

    bool document(Json& out) {
        // Optional UTF-8 byte order mark.
        if (s_.size() >= 3 && static_cast<unsigned char>(s_[0]) == 0xEF &&
            static_cast<unsigned char>(s_[1]) == 0xBB && static_cast<unsigned char>(s_[2]) == 0xBF)
            pos_ = 3;
        ws();
        if (!value(out, 0)) return false;
        ws();
        if (pos_ != s_.size()) return fail("unexpected text after the JSON value");
        return true;
    }

private:
    bool fail(const std::string& what) {
        err_ = what + " at byte " + std::to_string(pos_);
        return false;
    }
    void ws() {
        while (pos_ < s_.size() && (s_[pos_] == ' ' || s_[pos_] == '\t' || s_[pos_] == '\n' || s_[pos_] == '\r'))
            ++pos_;
    }
    bool literal(const char* word) {
        std::size_t n = 0;
        while (word[n]) ++n;
        if (s_.compare(pos_, n, word) != 0) return fail("invalid literal");
        pos_ += n;
        return true;
    }

    bool value(Json& out, int depth) {
        if (depth > 64) return fail("nesting too deep");
        if (pos_ >= s_.size()) return fail("unexpected end of input");
        const char c = s_[pos_];
        if (c == '{') return object(out, depth);
        if (c == '[') return array(out, depth);
        if (c == '"') {
            out.type = Json::Type::String;
            return string(out.text);
        }
        if (c == 't') {
            out.type = Json::Type::Bool;
            out.boolean = true;
            return literal("true");
        }
        if (c == 'f') {
            out.type = Json::Type::Bool;
            out.boolean = false;
            return literal("false");
        }
        if (c == 'n') {
            out.type = Json::Type::Null;
            return literal("null");
        }
        if (c == '-' || (c >= '0' && c <= '9')) return number(out);
        return fail("unexpected character");
    }

    bool number(Json& out) {
        const std::size_t start = pos_;
        if (s_[pos_] == '-') ++pos_;
        if (pos_ >= s_.size()) return fail("bad number");
        if (s_[pos_] == '0') {
            ++pos_;
        } else if (s_[pos_] >= '1' && s_[pos_] <= '9') {
            while (pos_ < s_.size() && s_[pos_] >= '0' && s_[pos_] <= '9') ++pos_;
        } else {
            return fail("bad number");
        }
        if (pos_ < s_.size() && s_[pos_] == '.') {
            ++pos_;
            const std::size_t d = pos_;
            while (pos_ < s_.size() && s_[pos_] >= '0' && s_[pos_] <= '9') ++pos_;
            if (pos_ == d) return fail("bad number");
        }
        if (pos_ < s_.size() && (s_[pos_] == 'e' || s_[pos_] == 'E')) {
            ++pos_;
            if (pos_ < s_.size() && (s_[pos_] == '+' || s_[pos_] == '-')) ++pos_;
            const std::size_t d = pos_;
            while (pos_ < s_.size() && s_[pos_] >= '0' && s_[pos_] <= '9') ++pos_;
            if (pos_ == d) return fail("bad number");
        }
        out.type = Json::Type::Number;
        out.text = s_.substr(start, pos_ - start);
        return true;
    }

    bool hex4(unsigned& v) {
        if (pos_ + 4 > s_.size()) return fail("bad \\u escape");
        v = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = s_[pos_++];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= unsigned(c - '0');
            else if (c >= 'a' && c <= 'f') v |= unsigned(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= unsigned(c - 'A' + 10);
            else return fail("bad \\u escape");
        }
        return true;
    }

    static void put_utf8(std::string& o, unsigned cp) {
        if (cp < 0x80) {
            o.push_back(char(cp));
        } else if (cp < 0x800) {
            o.push_back(char(0xC0 | (cp >> 6)));
            o.push_back(char(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            o.push_back(char(0xE0 | (cp >> 12)));
            o.push_back(char(0x80 | ((cp >> 6) & 0x3F)));
            o.push_back(char(0x80 | (cp & 0x3F)));
        } else {
            o.push_back(char(0xF0 | (cp >> 18)));
            o.push_back(char(0x80 | ((cp >> 12) & 0x3F)));
            o.push_back(char(0x80 | ((cp >> 6) & 0x3F)));
            o.push_back(char(0x80 | (cp & 0x3F)));
        }
    }

    bool string(std::string& out) {
        ++pos_;  // opening quote
        out.clear();
        while (true) {
            if (pos_ >= s_.size()) return fail("unterminated string");
            const unsigned char c = static_cast<unsigned char>(s_[pos_]);
            if (c == '"') {
                ++pos_;
                return true;
            }
            if (c < 0x20) return fail("control character in string");
            if (c != '\\') {
                out.push_back(char(c));
                ++pos_;
                continue;
            }
            ++pos_;
            if (pos_ >= s_.size()) return fail("unterminated string");
            const char e = s_[pos_++];
            switch (e) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case 'u': {
                    unsigned cp = 0;
                    if (!hex4(cp)) return false;
                    if (cp >= 0xD800 && cp < 0xDC00) {
                        unsigned lo = 0;
                        if (pos_ + 2 > s_.size() || s_[pos_] != '\\' || s_[pos_ + 1] != 'u')
                            return fail("unpaired surrogate");
                        pos_ += 2;
                        if (!hex4(lo)) return false;
                        if (lo < 0xDC00 || lo >= 0xE000) return fail("unpaired surrogate");
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    } else if (cp >= 0xDC00 && cp < 0xE000) {
                        return fail("unpaired surrogate");
                    }
                    put_utf8(out, cp);
                    break;
                }
                default: return fail("bad escape");
            }
        }
    }

    bool array(Json& out, int depth) {
        out.type = Json::Type::Array;
        ++pos_;
        ws();
        if (pos_ < s_.size() && s_[pos_] == ']') {
            ++pos_;
            return true;
        }
        while (true) {
            out.items.emplace_back();
            ws();
            if (!value(out.items.back(), depth + 1)) return false;
            ws();
            if (pos_ >= s_.size()) return fail("unterminated array");
            if (s_[pos_] == ',') {
                ++pos_;
                continue;
            }
            if (s_[pos_] == ']') {
                ++pos_;
                return true;
            }
            return fail("expected ',' or ']'");
        }
    }

    bool object(Json& out, int depth) {
        out.type = Json::Type::Object;
        ++pos_;
        ws();
        if (pos_ < s_.size() && s_[pos_] == '}') {
            ++pos_;
            return true;
        }
        std::set<std::string> seen;
        while (true) {
            ws();
            if (pos_ >= s_.size() || s_[pos_] != '"') return fail("expected a key");
            std::string key;
            if (!string(key)) return false;
            if (!seen.insert(key).second) return fail("duplicate key \"" + key + "\"");
            ws();
            if (pos_ >= s_.size() || s_[pos_] != ':') return fail("expected ':'");
            ++pos_;
            ws();
            out.members.emplace_back(key, Json());
            if (!value(out.members.back().second, depth + 1)) return false;
            ws();
            if (pos_ >= s_.size()) return fail("unterminated object");
            if (s_[pos_] == ',') {
                ++pos_;
                continue;
            }
            if (s_[pos_] == '}') {
                ++pos_;
                return true;
            }
            return fail("expected ',' or '}'");
        }
    }

    const std::string& s_;
    std::string& err_;
    std::size_t pos_ = 0;
};

}  // namespace

bool parse_json(const std::string& input, Json& out, std::string& error) {
    out = Json();
    Parser p(input, error);
    return p.document(out);
}

}  // namespace wgc
