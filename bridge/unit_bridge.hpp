#pragma once

#include "bridge/conversions.hpp"

#include <string>

namespace lmx::bridge {

bool unit_power_expression(const std::string& unit, int multiplier,
                           std::string& result);
bool unit_product_expression(const std::string& lhs, const std::string& rhs,
                             bool divide, std::string& result);
AdtObj* quantity_result(double si_value, std::string unit,
                        const char* operation);

} // namespace lmx::bridge
