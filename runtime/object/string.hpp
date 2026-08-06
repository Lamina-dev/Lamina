//
// Created by meian on 2026/3/28.
//

#pragma once

#include "object.hpp"

namespace lmx::runtime {

class String : public Object {
    std::string data;
public:
    explicit String() noexcept;

    explicit String(const std::string& data) noexcept;

    explicit String(std::string&& data) noexcept;
    explicit String(const std::string& data, size_t index) noexcept;
    explicit String(const char* data, size_t size) noexcept;

    explicit String(const char* data) noexcept;

    ~String() noexcept;

    friend std::ostream& operator<<(std::ostream &os, const String &d) noexcept {
        return os << d.data;
    }

    String& operator+=(const String& other) noexcept;
    String operator+(const String& other) const noexcept;

    [[nodiscard]] bool operator==(const Object& other) const noexcept;
    [[nodiscard]] bool operator!=(const Object& other) const noexcept;

    [[nodiscard]] std::string to_string() const noexcept;
    [[nodiscard]] bool equals(const Object* other) const noexcept;
    [[nodiscard]] std::string type_info() const noexcept;

    [[nodiscard]] const char* c_str() const noexcept;
};


}

