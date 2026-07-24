//
// Created by meian on 2026/4/10.
//

#include "gc.hpp"

#include "object/string.hpp"

using namespace lmx::runtime;

LmGCAllocator::~LmGCAllocator() noexcept {
    for (const auto& obj : objects) {
        delete obj;
    }
}

Object *LmGCAllocator::alloc_string(const char *str) noexcept {
    const auto ptr = new String(str);
    objects.push_back(ptr);
    return ptr;
}