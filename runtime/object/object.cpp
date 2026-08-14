//
// Created by meian on 2026/3/28.
//
#include "object.hpp"
#include "code_module.hpp"
#include "string.hpp"
#include "lsr_ExprObj.hpp"
#include "adt.hpp"
#include "literal.hpp"

using namespace lmx::runtime;

Object::Object(const uint32_t kind) noexcept: kind(kind) {}



std::string Object::to_string(Object *obj) noexcept {
    if (!obj) return "Null";
    switch (obj->get_kind()) {
    case ObjectKind::Object: {
        return "<Object:" + std::to_string((int64_t)obj) + ">";
    }
    case ObjectKind::Code  : {
        return reinterpret_cast<CodeModule*>(obj)->to_string();
    }
    case ObjectKind::String: {
        return reinterpret_cast<String*>(obj)->to_string();
    }
    case ObjectKind::Table : {
        return "";
    }
    case ObjectKind::Vector: {
        return "";
    }
    case ObjectKind::Matrix: {
        return "";
    }
    case ObjectKind::Array : {
        return "";
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
    default: {
        return "";
    }
    }
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
            delete reinterpret_cast<CodeModule*>(this);
            return;
        }
        case ObjectKind::String: {
            delete reinterpret_cast<String*>(this);
            return;
        }
        case ObjectKind::Table : {
            return;
        }
        case ObjectKind::Vector: {
            return;
        }
        case ObjectKind::Matrix: {
            return;
        }
        case ObjectKind::Array : {
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
        default: {
            return;
        }
        }
    }
}
