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
    String(const std::string& data) noexcept;
    String(std::string&& data) noexcept;
    explicit String(const std::string& data, size_t index) noexcept;
    explicit String(const char* data, size_t size) noexcept;
    String(const char* data) noexcept;

    ~String() noexcept override;

    friend std::ostream& operator<<(std::ostream &os, const String &d) noexcept {
        return os << d.data;
    }

    String& operator+=(const String& other) noexcept;
    String operator+(const String& other) const noexcept;

    [[nodiscard]] bool operator==(const Object& other) const noexcept override;
    [[nodiscard]] bool operator!=(const Object& other) const noexcept override;

    [[nodiscard]] Object* clone() const noexcept override;
    [[nodiscard]] std::string to_string() const noexcept override;
    [[nodiscard]] bool equals(const Object* other) const noexcept override;
    [[nodiscard]] std::string type_info() const noexcept override;
};


}

