//
// Created by meian on 2026/3/29.
//

#pragma once
#include "fraction.hpp"
#include "lmx.h"
#include "object.hpp"

namespace lmx::runtime {
enum class ValueKind : uint8_t {
    Null, C_Ptr, Obj, Int, Bool, Fraction, C_VaList
};

struct Value;
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

    LMX_INLINE explicit Value()            noexcept;
    LMX_INLINE explicit Value(void* ptr)   noexcept;
    LMX_INLINE explicit Value(Object* obj) noexcept;
    LMX_INLINE explicit Value(LmInt val)   noexcept;
    LMX_INLINE explicit Value(bool val)    noexcept;
    LMX_INLINE explicit Value(int num, int den) noexcept;
    LMX_INLINE explicit Value(const Fraction& frac) noexcept;
    LMX_INLINE explicit Value(const Value&) noexcept = default;
    LMX_INLINE explicit Value(Value&&) noexcept = default;

    LMX_INLINE Object* operator->() const noexcept;

    LMX_INLINE Value& operator=(void* c_ptr)       noexcept;
    LMX_INLINE Value& operator=(Object* obj)       noexcept;
    LMX_INLINE Value& operator=(LmInt int_val)     noexcept;
    LMX_INLINE Value& operator=(bool bool_val)     noexcept;
    LMX_INLINE Value& operator=(const Value& other)noexcept;
    LMX_INLINE Value& operator=(Value&& other)     noexcept;
    LMX_INLINE Value& operator=(const Fraction& fraction);
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

    LMX_INLINE bool operator==(const Value& other) const noexcept;
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
// #pragma pack(pop)

#define VALUE_DESTRUCT_UNSAFE(v) (v)->~Value()
LMX_INLINE Value::Value() noexcept = default;

LMX_INLINE Value::Value(const bool bool_val) noexcept : kind(ValueKind::Bool), bool_val(bool_val) {}

LMX_INLINE Value::Value(void *c_ptr) noexcept : kind(ValueKind::C_Ptr), c_ptr(c_ptr) {}

LMX_INLINE Value::Value(const LmInt int_val) noexcept : kind(ValueKind::Int), int_val(int_val) {}

LMX_INLINE Value::Value(Object *obj) noexcept : kind(ValueKind::Obj), obj(obj) {}

LMX_INLINE Value::Value(const int num, const int den) noexcept : kind(ValueKind::Fraction), frac_val(num, den) {}

LMX_INLINE Value::Value(const Fraction& frac) noexcept : kind(ValueKind::Fraction), frac_val(frac) {}

LMX_INLINE Value &Value::operator=(std::nullptr_t) noexcept {
    VALUE_DESTRUCT_UNSAFE(this);
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
    VALUE_DESTRUCT_UNSAFE(this);
    // assert(this->kind == ValueKind::C_Ptr);
    this->kind = ValueKind::C_Ptr;
    this->c_ptr = ptr;
    return *this;
}

LMX_INLINE Value &Value::operator=(const bool val) noexcept {
    VALUE_DESTRUCT_UNSAFE(this);
    // assert(this->kind == ValueKind::Bool);
    this->kind = ValueKind::Bool;
    this->bool_val = val;
    return *this;
}

LMX_INLINE Value &Value::operator=(const LmInt val) noexcept {
    VALUE_DESTRUCT_UNSAFE(this);
    // assert(this->kind == ValueKind::Int);
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
    // assert(this->kind == ValueKind::Obj);
    this->kind = ValueKind::Obj;
    this->obj = obj;
    return *this;
}

inline std::string Value::to_string() const noexcept {
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
    VALUE_DESTRUCT_UNSAFE(this);
    switch (other.kind) {
    case ValueKind::Obj: {
        this->obj = other.obj->get();
        this->kind = ValueKind::Obj;
        break;
    }
    default: {
        //memcpy(this, &other, sizeof(Value));
        this->kind = other.kind;
        this->int_val = other.int_val;
    }
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
