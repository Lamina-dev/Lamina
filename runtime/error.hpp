#pragma once
#include <iostream>
#include "lmx.h"
#include "object/object.hpp"

namespace lmx::runtime {

enum class RuntimeErrorType {
    ModuleLoad,
    CanNotCalling,
    IndexOutOfRange,
    Construct,
};
constexpr const char* error_str[] = {
    "ModuleLoaderError", "CanNotCalling", "IndexOutOfRange"
};

constexpr LMX_INLINE void VM_ERROR(const RuntimeErrorType type, const std::string& message) {
    std::cerr << error_str[static_cast<size_t>(type)] << "Error: " << message << std::endl;
    std::exit(1);
}

}
