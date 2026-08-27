#include "bridge/result.hpp"
#include "bridge/conversions.hpp"
#include "bridge/runtime_views.hpp"
#include "bridge/unit_bridge.hpp"
#include <cstdarg>

#include <cstddef>
#include <string>
#include <lmmc/lsr_stdlib.h>

using namespace lmx::bridge;

extern "C" LM_API AdtObj* lmx_mathematics_hypotenuse(ExprObj* lhs, ExprObj* rhs) {
    double x = 0.0;
    double y = 0.0;
    std::string error;
    if (!expr_to_real(lhs, x, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    if (!expr_to_real(rhs, y, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    lmmc_real_t out = 0.0;
    const auto status = lmmc_hypot(x, y, &out);
    return lmmc_real_result("lmx_mathematics_hypotenuse", status, out);
}

extern "C" LM_API AdtObj* lmx_mathematics_binary_logarithm(ExprObj* expr) {
    double x = 0.0;
    std::string error;
    if (!expr_to_real(expr, x, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    lmmc_real_t out = 0.0;
    const auto status = lmmc_log2(x, &out);
    return lmmc_real_result("lmx_mathematics_binary_logarithm", status, out);
}

extern "C" LM_API AdtObj* lmx_mathematics_binary_exponential(ExprObj* expr) {
    double x = 0.0;
    std::string error;
    if (!expr_to_real(expr, x, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    lmmc_real_t out = 0.0;
    const auto status = lmmc_exp2(x, &out);
    return lmmc_real_result("lmx_mathematics_binary_exponential", status, out);
}

extern "C" LM_API AdtObj* lmx_mathematics_pi() {
    lmmc_real_t value = 0.0;
    const auto status = lmmc_lsr_math_pi(&value);
    return lmmc_real_result("math.pi", status, value);
}

extern "C" LM_API AdtObj* lmx_mathematics_euler_number() {
    lmmc_real_t value = 0.0;
    const auto status = lmmc_lsr_math_e(&value);
    return lmmc_real_result("math.e", status, value);
}

extern "C" LM_API AdtObj* lmx_mathematics_golden_ratio() {
    lmmc_real_t value = 0.0;
    const auto status = lmmc_lsr_math_phi(&value);
    return lmmc_real_result("math.phi", status, value);
}

extern "C" LM_API ArrayObj* lmx_mathematics_constants() {
    auto* result = new ArrayObj();
    const auto count = lmmc_lsr_constants_count();
    for (std::size_t index = 0; index < count; ++index) {
        const char* name = lmmc_lsr_constants_name(index);
        if (name) result->append(Value(new StringObj(name)));
    }
    return result;
}

extern "C" LM_API AdtObj* lmx_mathematics_constant(const char* name) {
    lmmc_real_t value = 0.0;
    const auto status = lmmc_lsr_constants_get(name, &value);
    return lmmc_real_result("math.constant", status, value);
}

extern "C" LM_API AdtObj* lmx_mathematics_constant_unit(const char* name) {
    const char* unit = lmmc_lsr_constants_unit(name);
    if (!unit) return result_error(MathErrorCode::InvalidArgument, __func__, "math.constant_unit: unknown constant");
    return result_ok(new StringObj(unit), ValueKind::Obj);
}

extern "C" LM_API AdtObj* lmx_mathematics_imaginary_unit() {
    lmmc_complex_t value{};
    const auto status = lmmc_lsr_math_I(&value);
    return lmmc_complex_result("math.I", status, value);
}

extern "C" LM_API AdtObj* lmx_mathematics_sine(const double value) {
    return lmmc_unary_real_result("math.sin", value, lmmc_lsr_math_sin);
}
extern "C" LM_API AdtObj* lmx_mathematics_cosine(const double value) {
    return lmmc_unary_real_result("math.cos", value, lmmc_lsr_math_cos);
}
extern "C" LM_API AdtObj* lmx_mathematics_tangent(const double value) {
    return lmmc_unary_real_result("math.tan", value, lmmc_lsr_math_tan);
}
extern "C" LM_API AdtObj* lmx_mathematics_inverse_sine(const double value) {
    return lmmc_unary_real_result("math.asin", value, lmmc_lsr_math_asin);
}
extern "C" LM_API AdtObj* lmx_mathematics_inverse_cosine(const double value) {
    return lmmc_unary_real_result("math.acos", value, lmmc_lsr_math_acos);
}
extern "C" LM_API AdtObj* lmx_mathematics_inverse_tangent(const double value) {
    return lmmc_unary_real_result("math.atan", value, lmmc_lsr_math_atan);
}
extern "C" LM_API AdtObj* lmx_mathematics_square_root(const double value) {
    return lmmc_unary_real_result("math.sqrt", value, lmmc_lsr_math_sqrt);
}
extern "C" LM_API AdtObj* lmx_mathematics_exponential(const double value) {
    return lmmc_unary_real_result("math.exp", value, lmmc_lsr_math_exp);
}
extern "C" LM_API AdtObj* lmx_mathematics_natural_logarithm(const double value) {
    return lmmc_unary_real_result("math.ln", value, lmmc_lsr_math_ln);
}
extern "C" LM_API AdtObj* lmx_mathematics_natural_logarithm_legacy(const double value) {
    return lmmc_unary_real_result("math.log", value, lmmc_lsr_math_log);
}
extern "C" LM_API AdtObj* lmx_mathematics_common_logarithm(const double value) {
    return lmmc_unary_real_result("math.log10", value, lmmc_lsr_math_log10);
}
extern "C" LM_API AdtObj* lmx_mathematics_absolute_value(const double value) {
    return lmmc_unary_real_result("math.abs", value, lmmc_lsr_math_abs);
}
extern "C" LM_API AdtObj* lmx_mathematics_floor(const double value) {
    return lmmc_unary_real_result("math.floor", value, lmmc_lsr_math_floor);
}
extern "C" LM_API AdtObj* lmx_mathematics_ceil(const double value) {
    return lmmc_unary_real_result("math.ceil", value, lmmc_lsr_math_ceil);
}
extern "C" LM_API AdtObj* lmx_mathematics_round(const double value) {
    return lmmc_unary_real_result("math.round", value, lmmc_lsr_math_round);
}
extern "C" LM_API AdtObj* lmx_mathematics_power(const double base,
                                         const double exponent) {
    return lmmc_binary_real_result(
        "math.pow", base, exponent, lmmc_lsr_math_pow);
}
extern "C" LM_API AdtObj* lmx_mathematics_logarithm(const double value,
                                              const double base) {
    return lmmc_binary_real_result(
        "math.log_base", value, base, lmmc_lsr_math_log_base);
}
extern "C" LM_API AdtObj* lmx_mathematics_clamp(const double value,
                                           const double lower,
                                           const double upper) {
    return lmmc_ternary_real_result(
        "math.clamp", value, lower, upper, lmmc_lsr_math_clamp);
}
