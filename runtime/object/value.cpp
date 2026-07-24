//
// Created by meian on 2026/3/29.
//

#include "value.hpp"
#include <cassert>
#include <cstring>

using namespace lmx::runtime;

Value::Value() noexcept {}

Value::Value(const bool bool_val) noexcept : kind(ValueKind::Bool), bool_val(bool_val) {}

Value::Value(void *c_ptr) noexcept : kind(ValueKind::C_Ptr), c_ptr(c_ptr) {}

Value::Value(const int64_t int_val) noexcept : kind(ValueKind::Int), int_val(int_val) {}

Value::Value(Object *obj) noexcept : kind(ValueKind::Obj), obj(obj) {}

Value::Value(const int num, const int den) : kind(ValueKind::Fraction), frac_val(num, den) {}

Value::Value(const Fraction& frac) noexcept : kind(ValueKind::Fraction), frac_val(frac) {}

Value &Value::operator=(const Fraction& fraction) {
    // assert(this->kind == ValueKind::Fraction);
    this->kind = ValueKind::Fraction;
    this->frac_val = fraction;
    return *this;
}

Value &Value::operator=(std::nullptr_t) noexcept {
    // assert(this->kind == ValueKind::Null);
    this->kind = ValueKind::Null;
    this->null_val = nullptr;
    return *this;
}

Object *Value::operator->() const noexcept {
    // assert(this->kind == ValueKind::Obj);
    return obj;
}

Value &Value::operator=(void *ptr) noexcept {
    // assert(this->kind == ValueKind::C_Ptr);
    this->kind = ValueKind::C_Ptr;
    this->c_ptr = ptr;
    return *this;
}

Value &Value::operator=(const bool val) noexcept {
    // assert(this->kind == ValueKind::Bool);
    this->kind = ValueKind::Bool;
    this->bool_val = val;
    return *this;
}

Value &Value::operator=(const int64_t val) noexcept {
    // assert(this->kind == ValueKind::Int);
    this->kind = ValueKind::Int;
    this->int_val = val;
    return *this;
}

Value &Value::operator=(Object *obj) noexcept {
    // assert(this->kind == ValueKind::Obj);
    this->kind = ValueKind::Obj;
    this->obj = obj;
    return *this;
}

std::string Value::to_string() const noexcept {
    switch (kind) {
    case ValueKind::Bool: return bool_val ? "true" : "false";
    case ValueKind::Int: return std::to_string(int_val);
    case ValueKind::Obj: return obj->to_string();
    case ValueKind::C_Ptr: return "RawPtr";
    case ValueKind::Null: return "Null";
    case ValueKind::Fraction: return "frac";
    }

    // 不可能到达这里
    return {};
}

Value Value::operator%(const Value &other) const noexcept {
    // assert(this->kind == ValueKind::Int);
    return Value(this->int_val % other.int_val);
}

Value &Value::operator%=(const Value &other) noexcept {
    // assert(this->kind == ValueKind::Int);
    this->int_val %= other.int_val;
    return *this;
}

Value Value::operator*(const Value &other) const noexcept {
    // assert(this->kind == ValueKind::Int);
    return Value(this->int_val * other.int_val);
}

Value &Value::operator*=(const Value &other) noexcept {
    // assert(this->kind == ValueKind::Int);
    this->int_val *= other.int_val;
    return *this;
}

Value Value::operator/(const Value &other) const noexcept {
    // assert(this->kind == ValueKind::Int);
    return Value(this->int_val / other.int_val);
}

Value &Value::operator/=(const Value &other) noexcept {
    // assert(this->kind == ValueKind::Int);
    this->int_val /= other.int_val;
    return *this;
}

Value Value::operator+(const Value &other) const noexcept {
    // assert(this->kind == ValueKind::Int);
    return Value(int_val + other.int_val);
}

Value Value::operator-() const noexcept {
    // assert(this->kind == ValueKind::Int);
    return Value(-int_val);
}

Value &Value::operator+=(const Value &other) noexcept {
    // assert(this->kind == ValueKind::Int);
    this->int_val += other.int_val;
    return *this;
}

Value Value::operator-(const Value &other) const noexcept {
    // assert(this->kind == ValueKind::Int);
    return Value(int_val - other.int_val);
}

Value &Value::operator-=(const Value &other) noexcept {
    // assert(this->kind == ValueKind::Int);
    this->int_val -= other.int_val;
    return *this;
}

bool Value::operator<(const Value &other) const noexcept {
    return int_val < other.int_val;
}

bool Value::operator<=(const Value &other) const noexcept {
    return int_val <= other.int_val;
}

bool Value::operator>(const Value &other) const noexcept {
    return int_val > other.int_val;
}

bool Value::operator>=(const Value &other) const noexcept {
    return int_val >= other.int_val;
}

bool Value::operator==(const Value &other) const noexcept {
    return int_val == other.int_val;
}

bool Value::operator!=(const Value &other) const noexcept {
    return int_val != other.int_val;
}

bool Value::operator!() const noexcept {
    // assert(this->kind == ValueKind::Bool);
    return !bool_val;
}

Value::operator bool() const noexcept {
    // assert(this->kind == ValueKind::Bool);
    return bool_val;
}

Value &Value::operator=(const Value &other) noexcept {
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

Value &Value::operator=(Value &&other) noexcept = default;

Value::~Value() noexcept {
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
