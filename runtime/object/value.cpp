#include "value.hpp"

#include "adt.hpp"
#include "StringObj.hpp"
#include "lsr_expr_obj.hpp"
#include "literal.hpp"
#include "tuple.hpp"
#include "complex.hpp"
#include "vector.hpp"
#include "matrix.hpp"
#include "table.hpp"
#include "quantity.hpp"
#include "sparse.hpp"
#include "tensor.hpp"
#include "assumptions.hpp"

#include <functional>

using namespace lmx::runtime;

bool Value::operator==(const Value &other) const noexcept {
    if (kind != other.kind) return false;
    switch (kind) {
    case ValueKind::Null: return true;
    case ValueKind::C_Ptr: return c_ptr == other.c_ptr;
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
        if (obj->get_kind() == ObjectKind::Expr) {
            return reinterpret_cast<const ExprObj*>(obj)->equals(
                *reinterpret_cast<const ExprObj*>(other.obj));
        }
        if (obj->get_kind() == ObjectKind::Tuple) {
            return reinterpret_cast<const TupleObj*>(obj)->equals(
                *reinterpret_cast<const TupleObj*>(other.obj));
        }
        if (obj->get_kind() == ObjectKind::Complex) {
            return reinterpret_cast<const ComplexObj*>(obj)->equals(
                *reinterpret_cast<const ComplexObj*>(other.obj));
        }
        if (obj->get_kind() == ObjectKind::Vector) {
            return reinterpret_cast<const VectorObj*>(obj)->equals(
                *reinterpret_cast<const VectorObj*>(other.obj));
        }
        if (obj->get_kind() == ObjectKind::Matrix) {
            return reinterpret_cast<const MatrixObj*>(obj)->equals(
                *reinterpret_cast<const MatrixObj*>(other.obj));
        }
        if (obj->get_kind() == ObjectKind::Table) {
            return reinterpret_cast<const TableObj*>(obj)->equals(
                *reinterpret_cast<const TableObj*>(other.obj));
        }
        if (obj->get_kind() == ObjectKind::Quantity) {
            return reinterpret_cast<const QuantityObj*>(obj)->equals(
                *reinterpret_cast<const QuantityObj*>(other.obj));
        }
        if (obj->get_kind() == ObjectKind::Sparse) {
            return reinterpret_cast<const SparseMatrixObj*>(obj)->equals(
                *reinterpret_cast<const SparseMatrixObj*>(other.obj));
        }
        if (obj->get_kind() == ObjectKind::Tensor) {
            return reinterpret_cast<const TensorObj*>(obj)->equals(
                *reinterpret_cast<const TensorObj*>(other.obj));
        }
        if (obj->get_kind() == ObjectKind::Assumptions) {
            return reinterpret_cast<const AssumptionsObj*>(obj)->equals(
                *reinterpret_cast<const AssumptionsObj*>(other.obj));
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
    case ValueKind::Fraction: return frac_val.hash();
    case ValueKind::Real: return std::hash<double>{}(real_val);
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
        if (!obj) return 0;
        switch (obj->get_kind()) {
        case ObjectKind::String:
            return reinterpret_cast<const StringObj*>(obj)->hash();
        case ObjectKind::Literal:
            return reinterpret_cast<const LiteralObj*>(obj)->hash();
        case ObjectKind::Adt:
            return reinterpret_cast<const AdtObj*>(obj)->hash();
        case ObjectKind::Expr:
            return reinterpret_cast<const ExprObj*>(obj)->hash();
        case ObjectKind::Tuple:
            return reinterpret_cast<const TupleObj*>(obj)->hash();
        case ObjectKind::Complex:
            return reinterpret_cast<const ComplexObj*>(obj)->hash();
        case ObjectKind::Vector:
            return reinterpret_cast<const VectorObj*>(obj)->hash();
        case ObjectKind::Matrix:
            return reinterpret_cast<const MatrixObj*>(obj)->hash();
        case ObjectKind::Table:
            return reinterpret_cast<const TableObj*>(obj)->hash();
        case ObjectKind::Quantity:
            return reinterpret_cast<const QuantityObj*>(obj)->hash();
        case ObjectKind::Sparse:
            return reinterpret_cast<const SparseMatrixObj*>(obj)->hash();
        case ObjectKind::Tensor:
            return reinterpret_cast<const TensorObj*>(obj)->hash();
        case ObjectKind::Assumptions:
            return reinterpret_cast<const AssumptionsObj*>(obj)->hash();
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
    case ValueKind::Tuple: return Object::to_string(obj);
    case ValueKind::Set: return Object::to_string(obj);
    case ValueKind::Interval: return Object::to_string(obj);
    case ValueKind::Complex: return Object::to_string(obj);
    case ValueKind::Vector: return Object::to_string(obj);
    case ValueKind::Matrix: return Object::to_string(obj);
    case ValueKind::Table: return Object::to_string(obj);
    case ValueKind::Random: return Object::to_string(obj);
    case ValueKind::Quantity: return Object::to_string(obj);
    case ValueKind::Sparse: return Object::to_string(obj);
    case ValueKind::Tensor: return Object::to_string(obj);
    case ValueKind::Assumptions: return Object::to_string(obj);
    case ValueKind::C_Ptr: return "RawPtr";
    case ValueKind::Null: return "Null";
    case ValueKind::Fraction: return frac_val.to_string();
    case ValueKind::C_VaList: return "VaList";
    case ValueKind::C_ValueRef: return "ValueRef";
    }

    return {};
}

}
