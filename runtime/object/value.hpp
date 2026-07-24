//
// Created by meian on 2026/3/29.
//

#pragma once
#include "code_module.hpp"
#include "fraction.hpp"
#include "object.hpp"

namespace lmx::runtime {
enum class ValueKind {
    Null, C_Ptr, Obj, Int, Bool, Fraction,
};
struct Value {
    ValueKind kind{ValueKind::Null};
    union {
        std::nullptr_t null_val;
        void* c_ptr{nullptr};
        Object* obj;
        int64_t int_val;
        bool bool_val;
        Fraction frac_val;
    };

    explicit Value()            noexcept;
    explicit Value(void* ptr)   noexcept;
    explicit Value(Object* obj) noexcept;
    explicit Value(int64_t val) noexcept;
    explicit Value(bool val)    noexcept;
    explicit Value(int num, int den);
    explicit Value(const Fraction& frac) noexcept;

    Object* operator->() const noexcept;

    Value& operator=(void* c_ptr)       noexcept;
    Value& operator=(Object* obj)       noexcept;
    Value& operator=(int64_t int_val)   noexcept;
    Value& operator=(bool bool_val)     noexcept;
    Value& operator=(const Value& other)noexcept;
    Value& operator=(Value&& other)     noexcept;
    Value &operator=(const Fraction& fraction);
    Value& operator=(std::nullptr_t)   noexcept;

    Value operator+(const Value& other) const noexcept;
    Value operator-(const Value& other) const noexcept;
    Value operator*(const Value& other) const noexcept;
    Value operator/(const Value& other) const noexcept;
    Value operator%(const Value& other) const noexcept;
    Value operator-() const noexcept;

    Value& operator+=(const Value& other) noexcept;
    Value& operator-=(const Value& other) noexcept;
    Value& operator*=(const Value& other) noexcept;
    Value& operator/=(const Value& other) noexcept;
    Value& operator%=(const Value& other) noexcept;

    bool operator==(const Value& other) const noexcept;
    bool operator!=(const Value& other) const noexcept;
    bool operator< (const Value& other) const noexcept;
    bool operator<=(const Value& other) const noexcept;
    bool operator> (const Value& other) const noexcept;
    bool operator>=(const Value& other) const noexcept;

    bool operator!() const noexcept;

    explicit operator bool() const noexcept;

    [[nodiscard]] std::string to_string() const noexcept;

    ~Value() noexcept;
};

}
