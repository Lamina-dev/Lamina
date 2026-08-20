//
// Created by meian on 2026/3/29.
//
#include "value.hpp"

#include "adt.hpp"
#include "StringObj.hpp"
#include "literal.hpp"

#include <functional>

using namespace lmx::runtime;

bool Value::operator==(const Value &other) const noexcept {
    if (kind != other.kind) return false;
    switch (kind) {
    case ValueKind::Null: return true;
    case ValueKind::C_Ptr: return c_ptr == other.c_ptr;
    case ValueKind::Obj:
    case ValueKind::Expr:
        if (obj == other.obj) return true;
        if (!obj || !other.obj || obj->get_kind() != other.obj->get_kind()) return false;
        if (obj->get_kind() == ObjectKind::Adt) {
            return reinterpret_cast<const AdtObj*>(obj)->equals(*reinterpret_cast<const AdtObj*>(other.obj));
        }
        if (obj->get_kind() == ObjectKind::String) {
            return reinterpret_cast<const StringObj*>(obj)->equals(other.obj);
        }
        if (obj->get_kind() == ObjectKind::Literal) {
            return reinterpret_cast<const LiteralObj*>(obj)->equals(
                *reinterpret_cast<const LiteralObj*>(other.obj));
        }
        return false;
    case ValueKind::Int: return int_val == other.int_val;
    case ValueKind::Bool: return bool_val == other.bool_val;
    case ValueKind::Fraction: return frac_val == other.frac_val;
    case ValueKind::Real: return real_val == other.real_val;
    case ValueKind::C_VaList: return false;
    case ValueKind::C_ValueRef: return false;
    }
    return false;
}

std::size_t Value::hash() const noexcept {
    switch (kind) {
    case ValueKind::Null: return 0;
    case ValueKind::C_Ptr: return std::hash<void*>{}(c_ptr);
    case ValueKind::Int: return std::hash<LmInt>{}(int_val);
    case ValueKind::Bool: return std::hash<bool>{}(bool_val);
    case ValueKind::Fraction: {
        auto result = std::hash<int32_t>{}(frac_val.num);
        result ^= std::hash<int32_t>{}(frac_val.den) + 0x9e3779b9U +
                  (result << 6U) + (result >> 2U);
        return result;
    }
    case ValueKind::Real: return std::hash<double>{}(real_val);
    case ValueKind::Obj:
    case ValueKind::Expr:
        if (!obj) return 0;
        switch (obj->get_kind()) {
        case ObjectKind::String:
            return reinterpret_cast<const StringObj*>(obj)->hash();
        case ObjectKind::Literal:
            return reinterpret_cast<const LiteralObj*>(obj)->hash();
        case ObjectKind::Adt:
            return reinterpret_cast<const AdtObj*>(obj)->hash();
        default:
            return std::hash<const Object*>{}(obj);
        }
    case ValueKind::C_VaList:
    case ValueKind::C_ValueRef:
        return 0;
    }
    return 0;
}

namespace lmx::runtime {

std::string Value::to_string() const noexcept {
    switch (kind) {
    case ValueKind::Bool: return bool_val ? "true" : "false";
    case ValueKind::Int: return std::to_string(int_val);
    case ValueKind::Real: return std::to_string(real_val);
    case ValueKind::Obj: return Object::to_string(obj);
    case ValueKind::Expr: return Object::to_string(obj);
    case ValueKind::C_Ptr: return "RawPtr";
    case ValueKind::Null: return "Null";
    case ValueKind::Fraction: return frac_val.to_string();
    case ValueKind::C_VaList: return "VaList";
    case ValueKind::C_ValueRef: return "ValueRef";
    }

    // 不可能到达这里
    return {};
}

}
