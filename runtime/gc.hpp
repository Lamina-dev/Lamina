
#pragma once
#include <cstdint>
#include <memory>

#include "object/array.hpp"
#include "object/object.hpp"
#include "object/StringObj.hpp"
#include "object/tuple.hpp"

namespace lmx::runtime {
class GC;

class LmGCAllocator {
public:
    LMX_INLINE Object* alloc_string(const char *str, uint32_t len) noexcept {
        return new StringObj(str, len);
    }
    LMX_INLINE Object* alloc_array(const size_t len) noexcept {
        return new ArrayObj(len);
    }
    LMX_INLINE Object* alloc_tuple(const size_t len) noexcept {
        return new TupleObj(len);
    }
    ~LmGCAllocator() noexcept = default;
};

}
