#pragma once
#include <iostream>
#include "lmx.h"
#include "object/object.hpp"

namespace lmx::runtime {
const std::string VM_ERROR_ModLoad = "ModuleLoaderError";
const std::string VM_ERROR_CanNotCalling = "CanNotCallingError";

constexpr LMX_INLINE void VM_ERROR(const std::string& message) {
    std::cerr << message << std::endl;
    std::exit(1);
}

}
