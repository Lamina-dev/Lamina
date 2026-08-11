//
// Created by meian on 2026/4/10.
// 这个作为后期实现，暂时这样吧

#pragma once
#include <list>
#include <cstdint>
#include <memory>

#include "object/array.hpp"
#include "object/object.hpp"
#include "object/StringObj.hpp"

namespace lmx::runtime {
class GC;

// using GCObject = std::shared_ptr<Object>;
class LmGCAllocator {
    std::list<Object*> objects;
public:
    LMX_INLINE Object* alloc_string(const char *str, uint32_t len) noexcept {
        const auto ptr = new StringObj(str, len);
        objects.push_back(ptr);
        return ptr;
    }
    LMX_INLINE Object* alloc_array(const size_t len) noexcept {
        const auto ptr = new ArrayObj(len);
        objects.push_back(ptr);
        return ptr;
    }
    ~LmGCAllocator() noexcept {
        for (const auto obj : objects) {
            if (obj->get_rc() > 0) obj->release();
        }
    };
};

}
