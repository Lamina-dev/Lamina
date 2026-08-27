#include "bridge/result.hpp"
#include "bridge/conversions.hpp"
#include "bridge/runtime_views.hpp"
#include "bridge/unit_bridge.hpp"
#include <cstdarg>

using namespace lmx::bridge;

extern "C" LM_API AdtObj* lmx_special_functions_hyperbolic_sine(const double value) {
    return lmmc_unary_real_result("special.sinh", value, lmmc_sinh);
}
extern "C" LM_API AdtObj* lmx_special_functions_hyperbolic_cosine(const double value) {
    return lmmc_unary_real_result("special.cosh", value, lmmc_cosh);
}
extern "C" LM_API AdtObj* lmx_special_functions_hyperbolic_tangent(const double value) {
    return lmmc_unary_real_result("special.tanh", value, lmmc_tanh);
}
extern "C" LM_API AdtObj* lmx_special_functions_inverse_hyperbolic_sine(const double value) {
    return lmmc_unary_real_result("special.asinh", value, lmmc_asinh);
}
extern "C" LM_API AdtObj* lmx_special_functions_inverse_hyperbolic_cosine(const double value) {
    return lmmc_unary_real_result("special.acosh", value, lmmc_acosh);
}
extern "C" LM_API AdtObj* lmx_special_functions_inverse_hyperbolic_tangent(const double value) {
    return lmmc_unary_real_result("special.atanh", value, lmmc_atanh);
}
extern "C" LM_API AdtObj* lmx_special_functions_exponential_minus_one(const double value) {
    return lmmc_unary_real_result("special.expm1", value, lmmc_expm1);
}
extern "C" LM_API AdtObj* lmx_special_functions_logarithm_one_plus(const double value) {
    return lmmc_unary_real_result("special.log1p", value, lmmc_log1p);
}
extern "C" LM_API AdtObj* lmx_special_functions_gamma(const double value) {
    return lmmc_unary_real_result("special.gamma", value, lmmc_tgamma);
}
extern "C" LM_API AdtObj* lmx_special_functions_logarithmic_gamma(const double value) {
    return lmmc_unary_real_result("special.lgamma", value, lmmc_lgamma);
}
extern "C" LM_API AdtObj* lmx_special_functions_digamma(const double value) {
    return lmmc_unary_real_result("special.digamma", value, lmmc_digamma);
}
extern "C" LM_API AdtObj* lmx_special_functions_error_function(const double value) {
    return lmmc_unary_real_result("special.erf", value, lmmc_erf);
}
extern "C" LM_API AdtObj* lmx_special_functions_complementary_error_function(const double value) {
    return lmmc_unary_real_result("special.erfc", value, lmmc_erfc);
}
extern "C" LM_API AdtObj* lmx_special_functions_lambert_w(const double value) {
    return lmmc_unary_real_result("special.lambert_w", value, lmmc_lambertw);
}
extern "C" LM_API AdtObj* lmx_special_functions_lambert_w_negative_one(const double value) {
    return lmmc_unary_real_result(
        "special.lambert_wm1", value, lmmc_lambertw_wm1);
}
extern "C" LM_API AdtObj* lmx_special_functions_two_argument_arctangent(
    const double y, const double x) {
    return lmmc_binary_real_result("special.atan2", y, x, lmmc_atan2);
}
extern "C" LM_API AdtObj* lmx_special_functions_beta(
    const double first, const double second) {
    return lmmc_binary_real_result("special.beta", first, second, lmmc_beta);
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

extern "C" LM_API AdtObj* lmx_special_functions_is_finite(const double value) {
    return special_predicate("special.is_finite", value, lmmc_isfinite);
}
extern "C" LM_API AdtObj* lmx_special_functions_is_not_a_number(const double value) {
    return special_predicate("special.is_nan", value, lmmc_isnan);
}
extern "C" LM_API AdtObj* lmx_special_functions_is_infinite(const double value) {
    return special_predicate("special.is_inf", value, lmmc_isinf);
}
extern "C" LM_API AdtObj* lmx_special_functions_approximately_equal(
    const double lhs, const double rhs, const double epsilon) {
    int result = 0;
    const auto status = lmmc_approx_eq(lhs, rhs, epsilon, &result);
    if (status != LMMC_STATUS_OK)
        return result_error(status, "special.approx_equal");
    return result_ok(result != 0);
}
