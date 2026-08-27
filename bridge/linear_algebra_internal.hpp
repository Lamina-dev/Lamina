#pragma once

#include "bridge/result.hpp"
#include "bridge/conversions.hpp"
#include "bridge/runtime_views.hpp"
#include "bridge/unit_bridge.hpp"

namespace lmx::bridge::linear_algebra {

MatrixObj* copied_matrix(MatrixObj* input);

} // namespace lmx::bridge::linear_algebra
