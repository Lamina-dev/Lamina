//
// Created by meian on 2026/3/29.
//
#include "value.hpp"

#include "adt.hpp"
#include "string.hpp"
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
            return reinterpret_cast<const String*>(obj)->equals(other.obj);
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


