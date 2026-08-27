
#pragma once
#include "fraction.hpp"
#include "lmx.h"
#include "object.hpp"

#include <new>
#include <cstddef>
#include <utility>

namespace lmx::runtime {
enum class ValueKind : uint8_t {
    Null, C_Ptr, Obj, Int, Bool, Fraction, Real, Expr, C_VaList,
    C_ValueRef, Tuple, Set, Interval, Complex, Vector, Matrix, Table, Random,
    Quantity, Sparse, Tensor, Assumptions
};

LMX_INLINE constexpr bool is_object_value_kind(const ValueKind kind) noexcept {
    return kind == ValueKind::Obj || kind == ValueKind::Expr ||
           kind == ValueKind::Tuple || kind == ValueKind::Set ||
           kind == ValueKind::Interval || kind == ValueKind::Complex ||
           kind == ValueKind::Vector || kind == ValueKind::Matrix ||
           kind == ValueKind::Table || kind == ValueKind::Random ||
           kind == ValueKind::Quantity || kind == ValueKind::Sparse ||
           kind == ValueKind::Tensor || kind == ValueKind::Assumptions;
}

struct Value;
struct Value {
    ValueKind kind{ValueKind::Null};
    union {
        std::nullptr_t null_val;
        void* c_ptr{nullptr};
        Object* obj;
        LmInt int_val;
        bool bool_val;
        Fraction frac_val;
        double real_val;
    };

    explicit Value()            noexcept;
    explicit Value(void* ptr)   noexcept;
    explicit Value(Object* obj) noexcept;
    explicit Value(Object* obj, ValueKind kind) noexcept;
    explicit Value(LmInt val) noexcept;
    explicit Value(double val) noexcept;
    explicit Value(bool val)    noexcept;
    explicit Value(int num, int den) noexcept;
    explicit Value(const Fraction& frac) noexcept;
    Value(const Value& other) noexcept;
    Value(Value&& other) noexcept;

    LMX_INLINE Object* operator->() const noexcept;

    LMX_INLINE Value& operator=(void* c_ptr)       noexcept;
    LMX_INLINE Value& operator=(Object* obj)       noexcept;
    LMX_INLINE Value& operator=(LmInt int_val)     noexcept;
    LMX_INLINE Value& operator=(bool bool_val)     noexcept;
    LMX_INLINE Value& operator=(const Value& other)noexcept;
    LMX_INLINE Value& operator=(Value&& other)     noexcept;
    LMX_INLINE Value& operator=(const Fraction& fraction);
    LMX_INLINE Value& operator=(double val) noexcept;
    LMX_INLINE Value& operator=(std::nullptr_t)    noexcept;
    LMX_INLINE Value operator+(const Value& other) const noexcept;
    LMX_INLINE Value operator-(const Value& other) const noexcept;
    LMX_INLINE Value operator*(const Value& other) const noexcept;
    LMX_INLINE Value operator/(const Value& other) const noexcept;
    LMX_INLINE Value operator%(const Value& other) const noexcept;
    LMX_INLINE Value operator-() const noexcept;

    LMX_INLINE Value& operator+=(const Value& other) noexcept;
    LMX_INLINE Value& operator-=(const Value& other) noexcept;
    LMX_INLINE Value& operator*=(const Value& other) noexcept;
    LMX_INLINE Value& operator/=(const Value& other) noexcept;
    LMX_INLINE Value& operator%=(const Value& other) noexcept;

    bool operator==(const Value& other) const noexcept;
    [[nodiscard]] std::size_t hash() const noexcept;
    LMX_INLINE bool operator!=(const Value& other) const noexcept;
    LMX_INLINE bool operator< (const Value& other) const noexcept;
    LMX_INLINE bool operator<=(const Value& other) const noexcept;
    LMX_INLINE bool operator> (const Value& other) const noexcept;
    LMX_INLINE bool operator>=(const Value& other) const noexcept;

    LMX_INLINE bool operator!() const noexcept;

    LMX_INLINE explicit operator bool() const noexcept;

    [[nodiscard]] std::string to_string() const noexcept;

    LMX_INLINE ~Value() noexcept;
};

#define VALUE_DESTRUCT_UNSAFE(v) (v)->~Value()
LMX_INLINE Value::Value() noexcept = default;

LMX_INLINE Value::Value(const bool bool_val) noexcept : kind(ValueKind::Bool), bool_val(bool_val) {}

LMX_INLINE Value::Value(void *c_ptr) noexcept : kind(ValueKind::C_Ptr), c_ptr(c_ptr) {}

LMX_INLINE Value::Value(const LmInt int_val) noexcept : kind(ValueKind::Int), int_val(int_val) {}

LMX_INLINE Value::Value(Object *obj) noexcept : kind(ValueKind::Obj), obj(obj) {}

LMX_INLINE Value::Value(Object *obj, const ValueKind kind) noexcept
    : kind(is_object_value_kind(kind) ? kind : ValueKind::Obj), obj(obj) {}

LMX_INLINE Value::Value(const int num, const int den) noexcept : kind(ValueKind::Fraction), frac_val(num, den) {}

LMX_INLINE Value::Value(const Fraction& frac) noexcept : kind(ValueKind::Fraction), frac_val(frac) {}

LMX_INLINE Value::Value(double val) noexcept : kind(ValueKind::Real), real_val(val) {}

LMX_INLINE Value &Value::operator=(std::nullptr_t) noexcept {
    VALUE_DESTRUCT_UNSAFE(this);
    this->kind = ValueKind::Null;
    this->null_val = nullptr;
    return *this;
}

LMX_INLINE Object *Value::operator->() const noexcept {
    return obj;
}

LMX_INLINE Value &Value::operator=(double val) noexcept {
    VALUE_DESTRUCT_UNSAFE(this);
    this->kind = ValueKind::Real;
    this->real_val = val;
    return *this;
}

LMX_INLINE Value &Value::operator=(void *ptr) noexcept {
    VALUE_DESTRUCT_UNSAFE(this);
    this->kind = ValueKind::C_Ptr;
    this->c_ptr = ptr;
    return *this;
}

LMX_INLINE Value &Value::operator=(const bool val) noexcept {
    VALUE_DESTRUCT_UNSAFE(this);
    this->kind = ValueKind::Bool;
    this->bool_val = val;
    return *this;
}

LMX_INLINE Value &Value::operator=(const LmInt val) noexcept {
    VALUE_DESTRUCT_UNSAFE(this);
    this->kind = ValueKind::Int;
    this->int_val = val;
    return *this;
}

LMX_INLINE Value &Value::operator=(const Fraction &fraction) {
    VALUE_DESTRUCT_UNSAFE(this);
    this->kind = ValueKind::Fraction;
    this->frac_val = fraction;
    return *this;
}

LMX_INLINE Value &Value::operator=(Object *obj) noexcept {
    VALUE_DESTRUCT_UNSAFE(this);
    this->kind = ValueKind::Obj;
    this->obj = obj;
    return *this;
}



LMX_INLINE Value Value::operator%(const Value &other) const noexcept {
    return Value(this->int_val % other.int_val);
}

LMX_INLINE Value &Value::operator%=(const Value &other) noexcept {
    this->int_val %= other.int_val;
    return *this;
}

LMX_INLINE Value Value::operator*(const Value &other) const noexcept {
    return Value(this->int_val * other.int_val);
}

LMX_INLINE Value &Value::operator*=(const Value &other) noexcept {
    this->int_val *= other.int_val;
    return *this;
}

LMX_INLINE Value Value::operator/(const Value &other) const noexcept {
    return Value(this->int_val / other.int_val);
}

LMX_INLINE Value &Value::operator/=(const Value &other) noexcept {
    this->int_val /= other.int_val;
    return *this;
}

LMX_INLINE Value Value::operator+(const Value &other) const noexcept {
    return Value(int_val + other.int_val);
}

LMX_INLINE Value Value::operator-() const noexcept {
    return Value(-int_val);
}

LMX_INLINE Value &Value::operator+=(const Value &other) noexcept {
    this->int_val += other.int_val;
    return *this;
}

LMX_INLINE Value Value::operator-(const Value &other) const noexcept {
    return Value(int_val - other.int_val);
}

LMX_INLINE Value &Value::operator-=(const Value &other) noexcept {
    this->int_val -= other.int_val;
    return *this;
}

LMX_INLINE bool Value::operator<(const Value &other) const noexcept {
    return int_val < other.int_val;
}

LMX_INLINE bool Value::operator<=(const Value &other) const noexcept {
    return int_val <= other.int_val;
}

LMX_INLINE bool Value::operator>(const Value &other) const noexcept {
    return int_val > other.int_val;
}

LMX_INLINE bool Value::operator>=(const Value &other) const noexcept {
    return int_val >= other.int_val;
}

LMX_INLINE bool Value::operator!=(const Value &other) const noexcept {
    return !(*this == other);
}

LMX_INLINE bool Value::operator!() const noexcept {
    return !bool_val;
}

LMX_INLINE Value::operator bool() const noexcept {
    return bool_val;
}

LMX_INLINE Value::Value(const Value &other) noexcept : kind(ValueKind::Null), c_ptr(nullptr) {
    *this = other;
}

LMX_INLINE Value::Value(Value &&other) noexcept : kind(ValueKind::Null), c_ptr(nullptr) {
    *this = std::move(other);
}

LMX_INLINE Value &Value::operator=(const Value &other) noexcept {
    if (this == &other) return *this;
    this->~Value();
    if (is_object_value_kind(other.kind) && other.obj) {
        this->obj = other.obj->get();
        this->kind = other.kind;
    } else {
        this->kind = other.kind;
        switch (other.kind) {
        case ValueKind::Null: null_val = nullptr; break;
        case ValueKind::C_Ptr: c_ptr = other.c_ptr; break;
        case ValueKind::Int: int_val = other.int_val; break;
        case ValueKind::Bool: bool_val = other.bool_val; break;
        case ValueKind::Fraction: new (&frac_val) Fraction(other.frac_val); break;
        case ValueKind::Real: real_val = other.real_val; break;
        case ValueKind::C_VaList: c_ptr = nullptr; break;
        case ValueKind::C_ValueRef: c_ptr = nullptr; break;
        case ValueKind::Obj:
        case ValueKind::Expr:
        case ValueKind::Tuple:
        case ValueKind::Set:
        case ValueKind::Interval:
        case ValueKind::Complex:
        case ValueKind::Vector:
        case ValueKind::Matrix:
        case ValueKind::Table:
        case ValueKind::Random:
        case ValueKind::Quantity:
        case ValueKind::Sparse:
        case ValueKind::Tensor:
        case ValueKind::Assumptions:
            obj = nullptr;
            break;
        }
    }
    return *this;
}

LMX_INLINE Value &Value::operator=(Value &&other) noexcept {
    if (this == &other) return *this;
    this->~Value();
    this->kind = other.kind;
    switch (other.kind) {
    case ValueKind::Null: null_val = nullptr; break;
    case ValueKind::C_Ptr: c_ptr = other.c_ptr; break;
    case ValueKind::Obj:
    case ValueKind::Expr:
    case ValueKind::Tuple:
    case ValueKind::Set:
    case ValueKind::Interval:
    case ValueKind::Complex:
    case ValueKind::Vector:
    case ValueKind::Matrix:
    case ValueKind::Table:
    case ValueKind::Random:
    case ValueKind::Quantity:
    case ValueKind::Sparse:
    case ValueKind::Tensor:
    case ValueKind::Assumptions:
        obj = other.obj;
        break;
    case ValueKind::Int: int_val = other.int_val; break;
    case ValueKind::Bool: bool_val = other.bool_val; break;
    case ValueKind::Fraction: new (&frac_val) Fraction(other.frac_val); break;
    case ValueKind::Real: real_val = other.real_val; break;
    case ValueKind::C_VaList: c_ptr = nullptr; break;
    case ValueKind::C_ValueRef: c_ptr = nullptr; break;
    }
    if (is_object_value_kind(kind) || kind == ValueKind::C_Ptr ||
        kind == ValueKind::C_ValueRef) {
        other.kind = ValueKind::Null;
        other.c_ptr = nullptr;
    }
    return *this;
}

LMX_INLINE Value::~Value() noexcept {
    switch (this->kind) {
    case ValueKind::Obj:
    case ValueKind::Expr:
    case ValueKind::Tuple:
    case ValueKind::Set:
    case ValueKind::Interval:
    case ValueKind::Complex:
    case ValueKind::Vector:
    case ValueKind::Matrix:
    case ValueKind::Table:
    case ValueKind::Random:
    case ValueKind::Quantity:
    case ValueKind::Sparse:
    case ValueKind::Tensor:
    case ValueKind::Assumptions: {
        if (this->obj) this->obj->release();
        break;
    }
    default:{}
    }
    kind = ValueKind::Null;
    c_ptr = nullptr;
}
}
