#include "bridge/result.hpp"
#include "bridge/conversions.hpp"
#include "bridge/runtime_views.hpp"
#include "bridge/unit_bridge.hpp"
#include <cstdarg>

#include <cstddef>
#include <string>
#include <lmmc/lsr_stdlib.h>

using namespace lmx::bridge;

extern "C" LM_API AdtObj* lmx_mathematics_hypotenuse(ExprObj* lhs, ExprObj* rhs) noexcept try {
    ensure_lmmc_runtime();
    double x = 0.0;
    double y = 0.0;
    std::string error;
    if (!expr_to_real(lhs, x, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    if (!expr_to_real(rhs, y, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    lmmc_real_t out = 0.0;
    const auto status = lmmc_hypot(x, y, &out);
    return lmmc_real_result("lmx_mathematics_hypotenuse", status, out);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_mathematics_binary_logarithm(ExprObj* expr) noexcept try {
    ensure_lmmc_runtime();
    double x = 0.0;
    std::string error;
    if (!expr_to_real(expr, x, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    lmmc_real_t out = 0.0;
    const auto status = lmmc_log2(x, &out);
    return lmmc_real_result("lmx_mathematics_binary_logarithm", status, out);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_mathematics_binary_exponential(ExprObj* expr) noexcept try {
    ensure_lmmc_runtime();
    double x = 0.0;
    std::string error;
    if (!expr_to_real(expr, x, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    lmmc_real_t out = 0.0;
    const auto status = lmmc_exp2(x, &out);
    return lmmc_real_result("lmx_mathematics_binary_exponential", status, out);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_mathematics_pi() noexcept try {
    ensure_lmmc_runtime();
    lmmc_real_t value = 0.0;
    const auto status = lmmc_lsr_math_pi(&value);
    return lmmc_real_result("math.pi", status, value);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_mathematics_euler_number() noexcept try {
    ensure_lmmc_runtime();
    lmmc_real_t value = 0.0;
    const auto status = lmmc_lsr_math_e(&value);
    return lmmc_real_result("math.e", status, value);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_mathematics_golden_ratio() noexcept try {
    ensure_lmmc_runtime();
    lmmc_real_t value = 0.0;
    const auto status = lmmc_lsr_math_phi(&value);
    return lmmc_real_result("math.phi", status, value);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API ArrayObj* lmx_mathematics_constants() noexcept try {
    ensure_lmmc_runtime();
    auto result = make_owned_object<ArrayObj>();
    const auto count = lmmc_lsr_constants_count();
    for (std::size_t index = 0; index < count; ++index) {
        const char* name = lmmc_lsr_constants_name(index);
        if (name) {
            result->append(take_object_value(
                make_owned_object<StringObj>(name), ValueKind::Obj));
        }
    }
    return result.release();
} catch (...) {
    return nullptr;
}

extern "C" LM_API AdtObj* lmx_mathematics_constant(const char* name) noexcept try {
    ensure_lmmc_runtime();
    lmmc_real_t value = 0.0;
    const auto status = lmmc_lsr_constants_get(name, &value);
    return lmmc_real_result("math.constant", status, value);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_mathematics_constant_unit(const char* name) noexcept try {
    ensure_lmmc_runtime();
    const char* unit = lmmc_lsr_constants_unit(name);
    if (!unit) return result_error(MathErrorCode::InvalidArgument, __func__, "math.constant_unit: unknown constant");
    return result_ok(new StringObj(unit), ValueKind::Obj);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_mathematics_imaginary_unit() noexcept try {
    ensure_lmmc_runtime();
    lmmc_complex_t value{};
    const auto status = lmmc_lsr_math_I(&value);
    return lmmc_complex_result("math.I", status, value);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_mathematics_sine(const double value) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_unary_real_result("math.sin", value, lmmc_lsr_math_sin);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_mathematics_cosine(const double value) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_unary_real_result("math.cos", value, lmmc_lsr_math_cos);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_mathematics_tangent(const double value) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_unary_real_result("math.tan", value, lmmc_lsr_math_tan);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_mathematics_inverse_sine(const double value) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_unary_real_result("math.asin", value, lmmc_lsr_math_asin);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_mathematics_inverse_cosine(const double value) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_unary_real_result("math.acos", value, lmmc_lsr_math_acos);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_mathematics_inverse_tangent(const double value) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_unary_real_result("math.atan", value, lmmc_lsr_math_atan);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_mathematics_square_root(const double value) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_unary_real_result("math.sqrt", value, lmmc_lsr_math_sqrt);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_mathematics_exponential(const double value) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_unary_real_result("math.exp", value, lmmc_lsr_math_exp);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_mathematics_natural_logarithm(const double value) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_unary_real_result("math.ln", value, lmmc_lsr_math_ln);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_mathematics_natural_logarithm_legacy(const double value) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_unary_real_result("math.log", value, lmmc_lsr_math_log);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_mathematics_common_logarithm(const double value) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_unary_real_result("math.log10", value, lmmc_lsr_math_log10);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_mathematics_absolute_value(const double value) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_unary_real_result("math.abs", value, lmmc_lsr_math_abs);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_mathematics_floor(const double value) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_unary_real_result("math.floor", value, lmmc_lsr_math_floor);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_mathematics_ceil(const double value) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_unary_real_result("math.ceil", value, lmmc_lsr_math_ceil);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_mathematics_round(const double value) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_unary_real_result("math.round", value, lmmc_lsr_math_round);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_mathematics_power(const double base,
                                         const double exponent) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_binary_real_result(
        "math.pow", base, exponent, lmmc_lsr_math_pow);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_mathematics_logarithm(const double value,
                                              const double base) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_binary_real_result(
        "math.log_base", value, base, lmmc_lsr_math_log_base);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_mathematics_clamp(const double value,
                                           const double lower,
                                           const double upper) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_ternary_real_result(
        "math.clamp", value, lower, upper, lmmc_lsr_math_clamp);
} catch (...) {
    return c_abi_current_exception(__func__);
}
