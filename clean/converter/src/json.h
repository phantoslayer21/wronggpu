// SPDX-License-Identifier: MIT
// Minimal JSON reader (RFC 8259) for the converter map. Written for this project.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace wgc {

class Json {
public:
    enum class Type { Null, Bool, Number, String, Array, Object };

    Type type = Type::Null;
    bool boolean = false;
    std::string text;              // String: decoded UTF-8; Number: the literal as written
    std::vector<Json> items;       // Array
    std::vector<std::pair<std::string, Json>> members;  // Object, in file order

    bool is_object() const { return type == Type::Object; }
    bool is_array() const { return type == Type::Array; }
    bool is_string() const { return type == Type::String; }
    bool is_number() const { return type == Type::Number; }
    bool is_bool() const { return type == Type::Bool; }

    // Object lookup; nullptr if absent or not an object.
    const Json* find(const std::string& key) const;

    // Number written as a plain integer literal in the int64 range ("12", "-3"; not "1.0", "1e3").
    bool as_int64(std::int64_t& out) const;
};

// Parses a complete JSON document. On error returns false and sets `error` (with a byte offset).
// Duplicate keys in one object are reported as errors.
bool parse_json(const std::string& input, Json& out, std::string& error);

}  // namespace wgc
