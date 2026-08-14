//
// Created by meian on 2026/3/29.
//
#include "value.hpp"

#include "adt.hpp"
#include "StringObj.hpp"
#include "literal.hpp"

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
    }
    return false;
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
    }

    // 不可能到达这里
    return {};
}

}
