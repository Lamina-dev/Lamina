
#pragma once
#include "object.hpp"
#include "value.hpp"

#include <cstddef>
#include <string>

namespace lmx::runtime {
class TupleObj : public Object {
    Value* data{nullptr};
    std::size_t len_{0};
public:
    LMX_INLINE explicit TupleObj(Value* data, const std::size_t len) noexcept
        : Object(ObjectKind::Tuple), data(data), len_(len) {}
    LMX_INLINE explicit TupleObj(const std::size_t len) noexcept
        : Object(ObjectKind::Tuple), data(new Value[len]), len_(len) {}
    LMX_INLINE explicit TupleObj() noexcept : Object(ObjectKind::Tuple) {}

    LMX_INLINE ~TupleObj() noexcept {
        delete[] data;
    }

    [[nodiscard]] LMX_INLINE Value& get(const uint8_t i = 0) const noexcept {
        return data[i];
    }
    LMX_INLINE void set(const uint8_t i, const Value& v) const noexcept {
        data[i] = v;
    }

    LMX_INLINE Value& operator[](const uint8_t i) const noexcept {
        return data[i];
    }

    [[nodiscard]] LMX_INLINE std::size_t size() const noexcept { return len_; }
    [[nodiscard]] bool equals(const TupleObj& other) const noexcept;
    [[nodiscard]] std::size_t hash() const noexcept;
    [[nodiscard]] std::string to_string() const noexcept;
};
}
