#include "bridge/result.hpp"
#include "bridge/conversions.hpp"
#include "bridge/runtime_views.hpp"

#include <cstddef>
#include <string>
#include <lmmc/stdlib.h>

using namespace lmx::bridge;

extern "C" LM_API AdtObj* lmx_math_hypot(ExprObj* lhs, ExprObj* rhs) noexcept try {
    ensure_lmmc_runtime();
    const auto x = expr_to_real(lhs, __func__);
    if (!x) return result_error(x.error());
    const auto y = expr_to_real(rhs, __func__);
    if (!y) return result_error(y.error());
    lmmc_real_t out = 0.0;
    const auto status = lmmc_hypot(x.value(), y.value(), &out);
    return lmmc_real_result("lmx_math_hypot", status, out);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_math_log2(ExprObj* expr) noexcept try {
    ensure_lmmc_runtime();
    const auto x = expr_to_real(expr, __func__);
    if (!x) return result_error(x.error());
    lmmc_real_t out = 0.0;
    const auto status = lmmc_log2(x.value(), &out);
    return lmmc_real_result("lmx_math_log2", status, out);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_math_exp2(ExprObj* expr) noexcept try {
    ensure_lmmc_runtime();
    const auto x = expr_to_real(expr, __func__);
    if (!x) return result_error(x.error());
    lmmc_real_t out = 0.0;
    const auto status = lmmc_exp2(x.value(), &out);
    return lmmc_real_result("lmx_math_exp2", status, out);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_math_pi() noexcept try {
    ensure_lmmc_runtime();
    lmmc_real_t value = 0.0;
    const auto status = lmmc_std_math_pi(&value);
    return lmmc_real_result("math.pi", status, value);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_math_e() noexcept try {
    ensure_lmmc_runtime();
    lmmc_real_t value = 0.0;
    const auto status = lmmc_std_math_e(&value);
    return lmmc_real_result("math.e", status, value);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_math_phi() noexcept try {
    ensure_lmmc_runtime();
    lmmc_real_t value = 0.0;
    const auto status = lmmc_std_math_phi(&value);
    return lmmc_real_result("math.phi", status, value);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API ArrayObj* lmx_math_constants() noexcept try {
    ensure_lmmc_runtime();
    auto result = make_owned_object<ArrayObj>();
    const auto count = lmmc_std_constants_count();
    for (std::size_t index = 0; index < count; ++index) {
        const char* name = lmmc_std_constants_name(index);
        if (name) {
            result->append(take_object_value(
                make_owned_object<StringObj>(name), ValueKind::Obj));
        }
    }
    return result.release();
} catch (...) {
    return nullptr;
}

extern "C" LM_API AdtObj* lmx_math_constant(const char* name) noexcept try {
    ensure_lmmc_runtime();
    lmmc_real_t value = 0.0;
    const auto status = lmmc_std_constants_get(name, &value);
    return lmmc_real_result("math.constant", status, value);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_math_constant_unit(const char* name) noexcept try {
    ensure_lmmc_runtime();
    const char* unit = lmmc_std_constants_unit(name);
    if (!unit) return result_error(MathErrorCode::InvalidArgument, __func__, "math.constant_unit: unknown constant");
    return result_ok(new StringObj(unit), ValueKind::Obj);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_math_i() noexcept try {
    ensure_lmmc_runtime();
    lmmc_complex_t value{};
    const auto status = lmmc_std_math_i(&value);
    return lmmc_complex_result("math.I", status, value);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_math_sin(const double value) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_unary_real_result("math.sin", value, lmmc_std_math_sin);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_math_cos(const double value) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_unary_real_result("math.cos", value, lmmc_std_math_cos);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_math_tan(const double value) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_unary_real_result("math.tan", value, lmmc_std_math_tan);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_math_asin(const double value) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_unary_real_result("math.asin", value, lmmc_std_math_asin);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_math_acos(const double value) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_unary_real_result("math.acos", value, lmmc_std_math_acos);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_math_atan(const double value) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_unary_real_result("math.atan", value, lmmc_std_math_atan);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_math_sqrt(const double value) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_unary_real_result("math.sqrt", value, lmmc_std_math_sqrt);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_math_exp(const double value) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_unary_real_result("math.exp", value, lmmc_std_math_exp);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_math_ln(const double value) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_unary_real_result("math.ln", value, lmmc_std_math_ln);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_math_log(const double value) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_unary_real_result("math.log", value, lmmc_std_math_log);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_math_log10(const double value) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_unary_real_result("math.log10", value, lmmc_std_math_log10);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_math_abs(const double value) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_unary_real_result("math.abs", value, lmmc_std_math_abs);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_math_floor(const double value) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_unary_real_result("math.floor", value, lmmc_std_math_floor);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_math_ceil(const double value) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_unary_real_result("math.ceil", value, lmmc_std_math_ceil);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_math_round(const double value) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_unary_real_result("math.round", value, lmmc_std_math_round);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_math_pow(const double base,
                                         const double exponent) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_binary_real_result(
        "math.pow", base, exponent, lmmc_std_math_pow);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_math_log_base(const double value,
                                              const double base) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_binary_real_result(
        "math.log_base", value, base, lmmc_std_math_log_base);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_math_clamp(const double value,
                                           const double lower,
                                           const double upper) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_ternary_real_result(
        "math.clamp", value, lower, upper, lmmc_std_math_clamp);
} catch (...) {
    return c_abi_current_exception(__func__);
}
