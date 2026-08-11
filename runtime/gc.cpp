//
// Created by meian on 2026/4/10.
//

#include "gc.hpp"

#include "object/string.hpp"

using namespace lmx::runtime;

LmGCAllocator::~LmGCAllocator() noexcept {
    objects.clear();
}

Object *LmGCAllocator::alloc_string(const char *str, const uint32_t len) noexcept {
    const auto ptr = new String(str, len);
    objects.push_back(ptr);
    return ptr;
}
