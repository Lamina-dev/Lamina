//
// Created by meian on 2026/3/28.
//

#pragma once

#include "object.hpp"

namespace lmx::runtime {

class StringObj : public Object {
    using StringObjImpl = std::string;
    StringObjImpl data;
public:
    explicit StringObj() noexcept;

    explicit StringObj(const StringObjImpl& data) noexcept;

    explicit StringObj(StringObjImpl&& data) noexcept;
    explicit StringObj(const StringObjImpl& data, size_t index) noexcept;
    explicit StringObj(const char* data, size_t size) noexcept;

    explicit StringObj(const char* data) noexcept;

    ~StringObj() noexcept;

    friend std::ostream& operator<<(std::ostream &os, const StringObj &d) noexcept {
        return os << d.data;
    }

    StringObj& operator+=(const StringObj& other) noexcept;
    StringObj operator+(const StringObj& other) const noexcept;

    [[nodiscard]] bool operator==(const Object& other) const noexcept;
    [[nodiscard]] bool operator!=(const Object& other) const noexcept;

    [[nodiscard]] std::string to_string() const noexcept;
    [[nodiscard]] bool equals(const Object* other) const noexcept;
    [[nodiscard]] std::size_t hash() const noexcept;
    [[nodiscard]] std::string type_info() const noexcept;

    [[nodiscard]] const char* c_str() const noexcept;
};


}

