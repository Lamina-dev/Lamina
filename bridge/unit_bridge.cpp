#include "bridge/unit_bridge.hpp"

#include "runtime/object/quantity.hpp"
#include "lmmc/lsr_stdlib.h"


#include <cctype>
#include <utility>

namespace lmx::bridge {

bool unit_power_expression(const std::string& unit, const int multiplier,
                           std::string& result) {
    if (unit == "1") {
        result = "1";
        return true;
    }
    std::size_t cursor = 0;
    int operation_sign = 1;
    result.clear();
    while (cursor < unit.size()) {
        const auto begin = cursor;
        while (cursor < unit.size() &&
               (std::isalpha(static_cast<unsigned char>(unit[cursor])) ||
                unit[cursor] == '_')) ++cursor;
        if (cursor == begin) return false;
        const auto name = unit.substr(begin, cursor - begin);
        int exponent = 1;
        if (cursor < unit.size() && unit[cursor] == '^') {
            ++cursor;
            int sign = 1;
            if (cursor < unit.size() && (unit[cursor] == '+' || unit[cursor] == '-')) {
                if (unit[cursor] == '-') sign = -1;
                ++cursor;
            }
            if (cursor == unit.size() ||
                !std::isdigit(static_cast<unsigned char>(unit[cursor]))) return false;
            exponent = 0;
            while (cursor < unit.size() &&
                   std::isdigit(static_cast<unsigned char>(unit[cursor]))) {
                exponent = exponent * 10 + (unit[cursor++] - '0');
            }
            exponent *= sign;
        }
        exponent *= operation_sign * multiplier;
        if (exponent != 0) {
            if (!result.empty()) result += '*';
            result += name;
            if (exponent != 1) result += '^' + std::to_string(exponent);
        }
        if (cursor == unit.size()) break;
        if (unit[cursor] == '*') operation_sign = 1;
        else if (unit[cursor] == '/') operation_sign = -1;
        else return false;
        ++cursor;
    }
    if (result.empty()) result = "1";
    return true;
}

bool unit_product_expression(const std::string& lhs, const std::string& rhs,
                             const bool divide, std::string& result) {
    std::string right;
    if (!unit_power_expression(rhs, divide ? -1 : 1, right)) return false;
    if (lhs == "1") result = right;
    else if (right == "1") result = lhs;
    else result = lhs + '*' + right;
    int ignored = 0;
    return lmmc_lsr_units_is_dimensionless(result.c_str(), &ignored) == LMMC_STATUS_OK;
}

AdtObj* quantity_result(const double si_value, std::string unit,
                        const char* operation) {
    if (!std::isfinite(si_value)) {
        return result_error(MathErrorCode::NumericalFailure,
                            operation ? operation : "quantity",
                            "non-finite quantity value");
    }
    int ignored = 0;
    const auto status = lmmc_lsr_units_is_dimensionless(unit.c_str(), &ignored);
    if (status != LMMC_STATUS_OK)
        return result_error(status, operation ? operation : "quantity");
    return result_ok(new QuantityObj(si_value, std::move(unit)),
                     ValueKind::Quantity);
}

} // namespace lmx::bridge
