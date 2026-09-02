
#pragma once
#include "value.hpp"
#include <vector>

#include "value.hpp"
#include "../error.hpp"

namespace lmx::runtime {

class ArrayObj : public Object {
    std::vector<Value> arr{};

    [[nodiscard]] LMX_INLINE size_t get_index(const LmInt i) const noexcept {
        if (i < 0) return static_cast<LmInt>(arr.size()) + i;
        return i;
    }
    LMX_INLINE void check_index(const size_t i) const {
        if (i >= arr.size()) {
            VM_ERROR(RuntimeErrorType::IndexOutOfRange,
                "array len is " + std::to_string(arr.size()) + " but you index " + std::to_string(i)
                );
        }
    }
public:
    explicit ArrayObj() noexcept : Object(ObjectKind::Array) {}
    explicit ArrayObj(const size_t len) : Object(ObjectKind::Array), arr(len) {}
    explicit ArrayObj(const ArrayObj&) = default;
    explicit ArrayObj(ArrayObj&&) noexcept = default;
    ~ArrayObj() noexcept = default;

    LMX_INLINE void append(const Value& v) {
        arr.push_back(v);
    }
    LMX_INLINE void append(Value&& v) {
        arr.push_back(std::move(v));
    }

    [[nodiscard]] LMX_INLINE Value& at(const LmInt i) {
        const size_t i2 = get_index(i);
        check_index(i2);
        return arr[i2];
    }

    LMX_INLINE void store(const LmInt i, const Value& v) {
        const size_t i2 = get_index(i);
        check_index(i2);
        arr[i2] = v;
    }
    LMX_INLINE void store(const LmInt i, Value&& v) {
        const size_t i2 = get_index(i);
        check_index(i2);
        VALUE_DESTRUCT_UNSAFE(&arr[i2]);                  // 释放旧元素(含占有的对象)
        new (&arr[i2]) Value(std::move(v));  // 转移新值(浅拷贝)
    }

    [[nodiscard]] LMX_INLINE LmInt len() const noexcept {
        return static_cast<LmInt>(arr.size());
    }

    [[nodiscard]] LMX_INLINE const std::vector<Value>& values() const noexcept {
        return arr;
    }
};

}
