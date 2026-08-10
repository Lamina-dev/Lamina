//
// Created by meian on 2026/3/28.
//
#include "object.hpp"

#include "array.hpp"
#include "code_module.hpp"
#include "StringObj.hpp"

using namespace lmx::runtime;

Object::Object(const uint32_t kind) noexcept: kind(kind) {}

std::string Object::to_string(Object *obj) noexcept {
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
            delete reinterpret_cast<CodeModuleObj*>(this);
            return;
        }
        case ObjectKind::String: {
            delete reinterpret_cast<StringObj*>(this);
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
            delete reinterpret_cast<ArrayObj*>(this);
            return;
        }
        default: {
            return;
        }
        }
    }
}
