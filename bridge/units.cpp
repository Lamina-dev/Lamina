#include "bridge/result.hpp"
#include "bridge/conversions.hpp"
#include "bridge/runtime_views.hpp"
#include "bridge/unit_bridge.hpp"
#include <cstdarg>

#include <string>

#include <lmmc/lsr_stdlib.h>
using namespace lmx::bridge;

extern "C" LM_API AdtObj* lmx_units_from_cartesian(const double value, const char* unit) {
    if (!unit) return result_error(MathErrorCode::InvalidArgument, __func__, "quantity: null unit");
    lmmc_real_t si_value = 0.0;
    const auto status = lmmc_lsr_units_strip_num(value, unit, &si_value);
    if (status != LMMC_STATUS_OK) return result_error(status, "quantity");
    return quantity_result(si_value, unit, "quantity");
}

extern "C" LM_API AdtObj* lmx_units_convert(QuantityObj* value,
                                                  const char* target_unit) {
    if (!value || !target_unit) return result_error(MathErrorCode::InvalidArgument, __func__, "quantity_convert: invalid argument");
    lmmc_real_t ignored = 0.0;
    const auto status = lmmc_lsr_units_convert(1.0, value->unit().c_str(),
                                               target_unit, &ignored);
    if (status != LMMC_STATUS_OK) return result_error(status, "quantity_convert");
    return quantity_result(value->si_value(), target_unit, "quantity_convert");
}

extern "C" LM_API AdtObj* lmx_units_value(QuantityObj* value) {
    if (!value) return result_error(MathErrorCode::InvalidArgument, __func__, "quantity_value: null quantity");
    lmmc_real_t displayed = 0.0;
    const auto status = lmmc_lsr_units_convert_from_si(
        value->si_value(), value->unit().c_str(), &displayed);
    return lmmc_real_result("quantity_value", status, displayed);
}

extern "C" LM_API double lmx_units_strip(QuantityObj* value) {
    return value ? value->si_value() : 0.0;
}

extern "C" LM_API StringObj* lmx_units_unit(QuantityObj* value) {
    return new StringObj(value ? value->unit() : "");
}

extern "C" LM_API AdtObj* lmx_units_is_dimensionless(QuantityObj* value) {
    if (!value) return result_error(MathErrorCode::InvalidArgument, __func__, "quantity_is_dimensionless: null quantity");
    int result = 0;
    const auto status = lmmc_lsr_units_is_dimensionless(value->unit().c_str(), &result);
    if (status != LMMC_STATUS_OK) {
        return result_error(status, "quantity_is_dimensionless");
    }
    return result_ok(result != 0);
}

extern "C" LM_API AdtObj* lmx_units_add(QuantityObj* lhs, QuantityObj* rhs) {
    if (!lhs || !rhs) return result_error(MathErrorCode::InvalidArgument, __func__, "quantity_add: null quantity");
    lmmc_real_t ignored = 0.0;
    const auto status = lmmc_lsr_units_convert(1.0, rhs->unit().c_str(),
                                               lhs->unit().c_str(), &ignored);
    if (status != LMMC_STATUS_OK) return result_error(status, "quantity_add");
    return quantity_result(lhs->si_value() + rhs->si_value(), lhs->unit(), "quantity_add");
}

extern "C" LM_API AdtObj* lmx_units_subtract(QuantityObj* lhs, QuantityObj* rhs) {
    if (!lhs || !rhs) return result_error(MathErrorCode::InvalidArgument, __func__, "quantity_sub: null quantity");
    lmmc_real_t ignored = 0.0;
    const auto status = lmmc_lsr_units_convert(1.0, rhs->unit().c_str(),
                                               lhs->unit().c_str(), &ignored);
    if (status != LMMC_STATUS_OK) return result_error(status, "quantity_sub");
    return quantity_result(lhs->si_value() - rhs->si_value(), lhs->unit(), "quantity_sub");
}

extern "C" LM_API AdtObj* lmx_units_multiply(QuantityObj* lhs, QuantityObj* rhs) {
    if (!lhs || !rhs) return result_error(MathErrorCode::InvalidArgument, __func__, "quantity_mul: null quantity");
    std::string unit;
    if (!unit_product_expression(lhs->unit(), rhs->unit(), false, unit)) {
        return result_error(MathErrorCode::InvalidArgument, __func__, "quantity_mul: invalid resulting dimension");
    }
    return quantity_result(lhs->si_value() * rhs->si_value(), std::move(unit),
                           "quantity_mul");
}

extern "C" LM_API AdtObj* lmx_units_divide(QuantityObj* lhs, QuantityObj* rhs) {
    if (!lhs || !rhs) return result_error(MathErrorCode::InvalidArgument, __func__, "quantity_div: null quantity");
    if (rhs->si_value() == 0.0) return result_error(MathErrorCode::InvalidArgument, __func__, "quantity_div: division by zero");
    std::string unit;
    if (!unit_product_expression(lhs->unit(), rhs->unit(), true, unit)) {
        return result_error(MathErrorCode::InvalidArgument, __func__, "quantity_div: invalid resulting dimension");
    }
    return quantity_result(lhs->si_value() / rhs->si_value(), std::move(unit),
                           "quantity_div");
}

extern "C" LM_API AdtObj* lmx_units_power(QuantityObj* value,
                                              const LmInt exponent) {
    if (!value || exponent < -32 || exponent > 32) {
        return result_error(MathErrorCode::InvalidArgument, __func__, "quantity_pow: exponent must be in [-32, 32]");
    }
    if (value->si_value() == 0.0 && exponent < 0) {
        return result_error(MathErrorCode::InvalidArgument, __func__, "quantity_pow: division by zero");
    }
    std::string unit;
    if (!unit_power_expression(value->unit(), static_cast<int>(exponent), unit)) {
        return result_error(MathErrorCode::InvalidArgument, __func__, "quantity_pow: invalid resulting dimension");
    }
    return quantity_result(std::pow(value->si_value(), static_cast<double>(exponent)),
                           std::move(unit), "quantity_pow");
}
