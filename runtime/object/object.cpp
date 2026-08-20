//
// Created by meian on 2026/3/28.
//
#include "object.hpp"

#include "array.hpp"
#include "code_module.hpp"
#include "StringObj.hpp"
#include "lsr_expr_obj.hpp"
#include "adt.hpp"
#include "literal.hpp"
#include "tuple.hpp"
#include "complex.hpp"
#include "vector.hpp"
#include "matrix.hpp"
#include "table.hpp"
#include "random.hpp"
#include "quantity.hpp"

using namespace lmx::runtime;

Object::Object(const uint32_t kind) noexcept: kind(kind) {}

std::string Object::to_string(Object *obj) noexcept {
    if (!obj) return "Null";
    switch (obj->get_kind()) {
    case ObjectKind::Object: {
        return "<Object:" + std::to_string(reinterpret_cast<LmInt>(obj)) + ">";
    }
    case ObjectKind::Code  : {
        return reinterpret_cast<CodeModuleObj*>(obj)->to_string();
    }
    case ObjectKind::String: {
        return reinterpret_cast<StringObj*>(obj)->to_string();
    }
    case ObjectKind::Table : {
        return reinterpret_cast<TableObj*>(obj)->to_string();
    }
    case ObjectKind::Vector: {
        return reinterpret_cast<VectorObj*>(obj)->to_string();
    }
    case ObjectKind::Matrix: {
        return reinterpret_cast<MatrixObj*>(obj)->to_string();
    }
    case ObjectKind::Array : {
        return "";
    }
    case ObjectKind::Tuple: {
        return reinterpret_cast<TupleObj*>(obj)->to_string();
    }
    case ObjectKind::Expr: {
        return reinterpret_cast<ExprObj*>(obj)->to_string();
    }
    case ObjectKind::Adt: {
        return reinterpret_cast<AdtObj*>(obj)->to_string();
    }
    case ObjectKind::Literal: {
        return reinterpret_cast<LiteralObj*>(obj)->to_string();
    }
    case ObjectKind::Complex: {
        return reinterpret_cast<ComplexObj*>(obj)->to_string();
    }
    case ObjectKind::Random: {
        return reinterpret_cast<RandomObj*>(obj)->to_string();
    }
    case ObjectKind::Quantity: {
        return reinterpret_cast<QuantityObj*>(obj)->to_string();
    }
    default: {
        return "";
    }
    }
    return {};
}

Object::~Object() noexcept = default;

uint32_t Object::get_kind() const noexcept {
    return this->kind;
}

Object *Object::get() noexcept {
    rc++;
    return this;
}

void Object::release() noexcept {
    if (--rc <= 0) {
        switch (kind) {
        case ObjectKind::Object: {
            delete this;
            return;
        }
        case ObjectKind::Code  : {
            delete reinterpret_cast<CodeModuleObj*>(this);
            return;
        }
        case ObjectKind::String: {
            delete reinterpret_cast<StringObj*>(this);
            return;
        }
        case ObjectKind::Table : {
            delete reinterpret_cast<TableObj*>(this);
            return;
        }
        case ObjectKind::Vector: {
            delete reinterpret_cast<VectorObj*>(this);
            return;
        }
        case ObjectKind::Matrix: {
            delete reinterpret_cast<MatrixObj*>(this);
            return;
        }
        case ObjectKind::Array : {
            delete reinterpret_cast<ArrayObj*>(this);
            return;
        }
        case ObjectKind::Tuple: {
            delete reinterpret_cast<TupleObj*>(this);
            return;
        }
        case ObjectKind::Expr: {
            delete reinterpret_cast<ExprObj*>(this);
            return;
        }
        case ObjectKind::Adt: {
            delete reinterpret_cast<AdtObj*>(this);
            return;
        }
        case ObjectKind::Literal: {
            delete reinterpret_cast<LiteralObj*>(this);
            return;
        }
        case ObjectKind::Complex: {
            delete reinterpret_cast<ComplexObj*>(this);
            return;
        }
        case ObjectKind::Random: {
            delete reinterpret_cast<RandomObj*>(this);
            return;
        }
        case ObjectKind::Quantity: {
            delete reinterpret_cast<QuantityObj*>(this);
            return;
        }
        default: {
            return;
        }
        }
    }
}
