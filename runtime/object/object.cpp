//
// Created by meian on 2026/3/28.
//
#include "object.hpp"

using namespace lmx::runtime;

Object::Object(const uint32_t kind) noexcept: kind(kind) {}


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
        delete this;
    }
}
