#include "bridge/result.hpp"
#include "bridge/conversions.hpp"
#include "bridge/runtime_views.hpp"
#include "bridge/unit_bridge.hpp"
#include <cstdarg>

using namespace lmx::bridge;

extern "C" LM_API AdtObj* lmx_special_functions_hyperbolic_sine(const double value) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_unary_real_result("special.sinh", value, lmmc_sinh);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_special_functions_hyperbolic_cosine(const double value) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_unary_real_result("special.cosh", value, lmmc_cosh);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_special_functions_hyperbolic_tangent(const double value) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_unary_real_result("special.tanh", value, lmmc_tanh);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_special_functions_inverse_hyperbolic_sine(const double value) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_unary_real_result("special.asinh", value, lmmc_asinh);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_special_functions_inverse_hyperbolic_cosine(const double value) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_unary_real_result("special.acosh", value, lmmc_acosh);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_special_functions_inverse_hyperbolic_tangent(const double value) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_unary_real_result("special.atanh", value, lmmc_atanh);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_special_functions_exponential_minus_one(const double value) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_unary_real_result("special.expm1", value, lmmc_expm1);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_special_functions_logarithm_one_plus(const double value) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_unary_real_result("special.log1p", value, lmmc_log1p);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_special_functions_gamma(const double value) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_unary_real_result("special.gamma", value, lmmc_tgamma);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_special_functions_logarithmic_gamma(const double value) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_unary_real_result("special.lgamma", value, lmmc_lgamma);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_special_functions_digamma(const double value) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_unary_real_result("special.digamma", value, lmmc_digamma);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_special_functions_error_function(const double value) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_unary_real_result("special.erf", value, lmmc_erf);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_special_functions_complementary_error_function(const double value) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_unary_real_result("special.erfc", value, lmmc_erfc);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_special_functions_lambert_w(const double value) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_unary_real_result("special.lambert_w", value, lmmc_lambertw);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_special_functions_lambert_w_negative_one(const double value) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_unary_real_result(
        "special.lambert_wm1", value, lmmc_lambertw_wm1);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_special_functions_two_argument_arctangent(
    const double y, const double x) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_binary_real_result("special.atan2", y, x, lmmc_atan2);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_special_functions_beta(
    const double first, const double second) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_binary_real_result("special.beta", first, second, lmmc_beta);
} catch (...) {
    return c_abi_current_exception(__func__);
}

namespace {
AdtObj* special_predicate(const char* name, const double value,
                          lmmc_status_t (*operation)(double, int*)) {
    int result = 0;
    const auto status = operation(value, &result);
    if (status != LMMC_STATUS_OK) return result_error(status, name);
    return result_ok(result != 0);
}
} // namespace

extern "C" LM_API AdtObj* lmx_special_functions_is_finite(const double value) noexcept try {
    ensure_lmmc_runtime();
    return special_predicate("special.is_finite", value, lmmc_isfinite);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_special_functions_is_not_a_number(const double value) noexcept try {
    ensure_lmmc_runtime();
    return special_predicate("special.is_nan", value, lmmc_isnan);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_special_functions_is_infinite(const double value) noexcept try {
    ensure_lmmc_runtime();
    return special_predicate("special.is_inf", value, lmmc_isinf);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_special_functions_approximately_equal(
    const double lhs, const double rhs, const double epsilon) noexcept try {
    ensure_lmmc_runtime();
    int result = 0;
    const auto status = lmmc_approx_eq(lhs, rhs, epsilon, &result);
    if (status != LMMC_STATUS_OK)
        return result_error(status, "special.approx_equal");
    return result_ok(result != 0);
} catch (...) {
    return c_abi_current_exception(__func__);
}
