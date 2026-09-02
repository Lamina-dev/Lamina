
#pragma once

#include "object.hpp"

namespace lmx::runtime {

class StringObj : public Object {
    using StringObjImpl = std::string;
    StringObjImpl data;
public:
    explicit StringObj();

    explicit StringObj(const StringObjImpl& data);

    explicit StringObj(StringObjImpl&& data) noexcept;
    explicit StringObj(const StringObjImpl& data, size_t index);
    explicit StringObj(const char* data, size_t size);

    explicit StringObj(const char* data);

    ~StringObj() noexcept;

    friend std::ostream& operator<<(std::ostream &os, const StringObj &d) noexcept {
        return os << d.data;
    }

    StringObj& operator+=(const StringObj& other);
    StringObj operator+(const StringObj& other) const;

    [[nodiscard]] bool operator==(const Object& other) const noexcept;
    [[nodiscard]] bool operator!=(const Object& other) const noexcept;

    [[nodiscard]] std::string to_string() const;
    [[nodiscard]] bool equals(const Object* other) const noexcept;
    [[nodiscard]] std::size_t hash() const noexcept;
    [[nodiscard]] std::string type_info() const noexcept;

    [[nodiscard]] const char* c_str() const noexcept;
};


}

