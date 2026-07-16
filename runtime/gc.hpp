//
// Created by meian on 2026/4/10.
//

#pragma once
#include <vector>
#include <cstdint>
#include <memory>

#include "object/object.hpp"

namespace lmx::runtime {
class GC;

using GCObject = std::shared_ptr<Object>;

}