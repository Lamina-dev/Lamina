//
// Created by meian on 2026/8/12.
//

#pragma once
#include "object.hpp"
#include "value.hpp"

namespace lmx::runtime {
class TupleObj : public Object {
    Value* data{nullptr};
public:
    LMX_INLINE explicit TupleObj(Value* data) noexcept : Object(ObjectKind::Tuple), data(data) {}
    LMX_INLINE explicit TupleObj(const size_t len) noexcept : Object(ObjectKind::Tuple), data(new Value[len]) {}
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

    static std::string to_string() noexcept;
};
}
