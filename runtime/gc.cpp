//
// Created by meian on 2026/4/10.
//

#include "gc.hpp"

#include "object/array.hpp"
#include "object/StringObj.hpp"

using namespace lmx::runtime;

LmGCAllocator::~LmGCAllocator() noexcept {
    for (const auto& obj : objects) {
        obj->release();
    }
}

Object *LmGCAllocator::alloc_string(const char *str, const uint32_t len) noexcept {
    const auto ptr = new StringObj(str, len);
    objects.push_back(ptr);
    return ptr;
}

Object *LmGCAllocator::alloc_array(const size_t len) noexcept {
    const auto ptr = new ArrayObj(len);
    objects.push_back(ptr);
    return ptr;
}
