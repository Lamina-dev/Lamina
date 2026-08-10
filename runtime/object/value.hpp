//
// Created by meian on 2026/3/29.
//

#pragma once
#include "fraction.hpp"
#include "lmx.h"
#include "object.hpp"

namespace lmx::runtime {
enum class ValueKind : uint8_t {
    Null, C_Ptr, Obj, Int, Bool, Fraction, C_VaList,
};
// #pragma pack(push, 1)
struct Value {
    ValueKind kind{ValueKind::Null};
    union {
        std::nullptr_t null_val;
        void* c_ptr{nullptr};
        Object* obj;
        LmInt int_val;
        bool bool_val;
        Fraction frac_val;
    };

    explicit Value()            noexcept;
    explicit Value(void* ptr)   noexcept;
    explicit Value(Object* obj) noexcept;
    explicit Value(LmInt val) noexcept;
    explicit Value(bool val)    noexcept;
    explicit Value(int num, int den);
    explicit Value(const Fraction& frac) noexcept;
    explicit Value(const Value&) = default;
    explicit Value(Value&&) = default;

    Object* operator->() const noexcept;

    Value& operator=(void* c_ptr)       noexcept;
    Value& operator=(Object* obj)       noexcept;
    Value& operator=(LmInt int_val)     noexcept;
    Value& operator=(bool bool_val)     noexcept;
    Value& operator=(const Value& other)noexcept;
    Value& operator=(Value&& other)     noexcept;
    Value& operator=(const Fraction& fraction);
    Value& operator=(std::nullptr_t)    noexcept;

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
// #pragma pack(pop)

LMX_INLINE Value::Value() noexcept = default;

LMX_INLINE Value::Value(const bool bool_val) noexcept : kind(ValueKind::Bool), bool_val(bool_val) {}

LMX_INLINE Value::Value(void *c_ptr) noexcept : kind(ValueKind::C_Ptr), c_ptr(c_ptr) {}

LMX_INLINE Value::Value(const LmInt int_val) noexcept : kind(ValueKind::Int), int_val(int_val) {}

LMX_INLINE Value::Value(Object *obj) noexcept : kind(ValueKind::Obj), obj(obj) {}

LMX_INLINE Value::Value(const int num, const int den) : kind(ValueKind::Fraction), frac_val(num, den) {}

LMX_INLINE Value::Value(const Fraction& frac) noexcept : kind(ValueKind::Fraction), frac_val(frac) {}

LMX_INLINE Value &Value::operator=(const Fraction& fraction) {
    this->~Value();
    // assert(this->kind == ValueKind::Fraction);
    this->kind = ValueKind::Fraction;
    this->frac_val = fraction;
    return *this;
}

LMX_INLINE Value &Value::operator=(std::nullptr_t) noexcept {
    this->~Value();
    // assert(this->kind == ValueKind::Null);
    this->kind = ValueKind::Null;
    this->null_val = nullptr;
    return *this;
}

LMX_INLINE Object *Value::operator->() const noexcept {
    // assert(this->kind == ValueKind::Obj);
    return obj;
}

LMX_INLINE Value &Value::operator=(void *ptr) noexcept {
    this->~Value();
    // assert(this->kind == ValueKind::C_Ptr);
    this->kind = ValueKind::C_Ptr;
    this->c_ptr = ptr;
    return *this;
}

LMX_INLINE Value &Value::operator=(const bool val) noexcept {
    this->~Value();
    // assert(this->kind == ValueKind::Bool);
    this->kind = ValueKind::Bool;
    this->bool_val = val;
    return *this;
}

LMX_INLINE Value &Value::operator=(const LmInt val) noexcept {
    this->~Value();
    // assert(this->kind == ValueKind::Int);
    this->kind = ValueKind::Int;
    this->int_val = val;
    return *this;
}

LMX_INLINE Value &Value::operator=(Object *obj) noexcept {
    this->~Value();
    // assert(this->kind == ValueKind::Obj);
    this->kind = ValueKind::Obj;
    this->obj = obj;
    return *this;
}

LMX_INLINE std::string Value::to_string() const noexcept {
    switch (kind) {
    case ValueKind::Bool: return bool_val ? "true" : "false";
    case ValueKind::Int: return std::to_string(int_val);
    case ValueKind::Obj: return Object::to_string(obj);
    case ValueKind::C_Ptr: return "RawPtr";
    case ValueKind::Null: return "Null";
    case ValueKind::Fraction: return frac_val.to_string();
    case ValueKind::C_VaList: return "VaList";
    }

    // 不可能到达这里
    return {};
}

LMX_INLINE Value Value::operator%(const Value &other) const noexcept {
    // assert(this->kind == ValueKind::Int);
    return Value(this->int_val % other.int_val);
}

LMX_INLINE Value &Value::operator%=(const Value &other) noexcept {
    // assert(this->kind == ValueKind::Int);
    this->int_val %= other.int_val;
    return *this;
}

LMX_INLINE Value Value::operator*(const Value &other) const noexcept {
    // assert(this->kind == ValueKind::Int);
    return Value(this->int_val * other.int_val);
}

LMX_INLINE Value &Value::operator*=(const Value &other) noexcept {
    // assert(this->kind == ValueKind::Int);
    this->int_val *= other.int_val;
    return *this;
}

LMX_INLINE Value Value::operator/(const Value &other) const noexcept {
    // assert(this->kind == ValueKind::Int);
    return Value(this->int_val / other.int_val);
}

LMX_INLINE Value &Value::operator/=(const Value &other) noexcept {
    // assert(this->kind == ValueKind::Int);
    this->int_val /= other.int_val;
    return *this;
}

LMX_INLINE Value Value::operator+(const Value &other) const noexcept {
    // assert(this->kind == ValueKind::Int);
    return Value(int_val + other.int_val);
}

LMX_INLINE Value Value::operator-() const noexcept {
    // assert(this->kind == ValueKind::Int);
    return Value(-int_val);
}

LMX_INLINE Value &Value::operator+=(const Value &other) noexcept {
    // assert(this->kind == ValueKind::Int);
    this->int_val += other.int_val;
    return *this;
}

LMX_INLINE Value Value::operator-(const Value &other) const noexcept {
    // assert(this->kind == ValueKind::Int);
    return Value(int_val - other.int_val);
}

LMX_INLINE Value &Value::operator-=(const Value &other) noexcept {
    // assert(this->kind == ValueKind::Int);
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

LMX_INLINE bool Value::operator==(const Value &other) const noexcept {
    return int_val == other.int_val;
}

LMX_INLINE bool Value::operator!=(const Value &other) const noexcept {
    return int_val != other.int_val;
}

LMX_INLINE bool Value::operator!() const noexcept {
    // assert(this->kind == ValueKind::Bool);
    return !bool_val;
}

LMX_INLINE Value::operator bool() const noexcept {
    // assert(this->kind == ValueKind::Bool);
    return bool_val;
}

LMX_INLINE Value &Value::operator=(const Value &other) noexcept {
    this->~Value();
    if (other.kind == ValueKind::Obj) {
        this->obj = other.obj->get();
        this->kind = ValueKind::Obj;
    } else {
        //memcpy(this, &other, sizeof(Value));
        this->kind = other.kind;
        this->int_val = other.int_val;
    }
    return *this;
}

LMX_INLINE Value &Value::operator=(Value &&other) noexcept = default;

LMX_INLINE Value::~Value() noexcept {
    switch (this->kind) {
    case ValueKind::Obj: {
        this->obj->release();
        break;
    }
    default:{}
    }
    kind = ValueKind::Null;
    c_ptr = nullptr;
}

}
