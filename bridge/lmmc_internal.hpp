#pragma once

#include "bridge/result.hpp"
#include "bridge/conversions.hpp"
#include "bridge/runtime_views.hpp"
#include "bridge/unit_bridge.hpp"

namespace lmx::bridge {

inline bool checked_nonnegative_ints(
    const ArrayObj* values, std::vector<std::size_t>& output,
    std::string& error) {
    if (!values) {
        error = "null index array";
        return false;
    }
    output.reserve(values->values().size());
    for (const auto& value : values->values()) {
        if (value.kind != ValueKind::Int || value.int_val < 0) {
            error = "index array contains an invalid value";
            return false;
        }
        output.push_back(static_cast<std::size_t>(value.int_val));
    }
    return true;
}

} // namespace lmx::bridge
