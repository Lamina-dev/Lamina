#include "bridge/result.hpp"
#include "bridge/conversions.hpp"
#include "bridge/runtime_views.hpp"
#include "bridge/unit_bridge.hpp"
#include <cstdarg>
#include "bridge/math_internal.hpp"
#include "numerical_integration.hpp"
#include "vector_calculus.hpp"
#include "symbolic_geometry.hpp"
using namespace lmx::bridge;

extern "C" LM_API AdtObj* lmx_computer_algebra_integrate_simpson_by_name(ExprObj* expression,
                                                  const char* variable,
                                                  ExprObj* lower,
                                                  ExprObj* upper,
                                                  const LmInt intervals) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    const auto* value = checked_expr(expression, error);
    if (!value) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    const auto* lower_value = checked_expr(lower, error);
    if (!lower_value) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    const auto* upper_value = checked_expr(upper, error);
    if (!upper_value) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    if (intervals <= 0 || intervals > std::numeric_limits<int>::max()) {
        return result_error(MathErrorCode::InvalidArgument, __func__, "Simpson integration: invalid interval count");
    }
    const auto result = LMCAS::quadrature_simpson_numeric(
        *value, variable ? variable : "", *lower_value, *upper_value,
        static_cast<int>(intervals));
    if (!result) return result_error(result.error());
    return result_ok(result.value().value);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_computer_algebra_integrate_gaussian_by_name(ExprObj* expression,
                                                   const char* variable,
                                                   ExprObj* lower,
                                                   ExprObj* upper,
                                                   const LmInt points) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    const auto* value = checked_expr(expression, error);
    if (!value) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    const auto* lower_value = checked_expr(lower, error);
    if (!lower_value) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    const auto* upper_value = checked_expr(upper, error);
    if (!upper_value) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    if (points <= 0 || points > std::numeric_limits<int>::max()) {
        return result_error(MathErrorCode::InvalidArgument, __func__, "Gaussian integration: invalid point count");
    }
    const auto result = LMCAS::quadrature_gaussian_numeric(
        *value, variable ? variable : "", *lower_value, *upper_value,
        static_cast<int>(points));
    if (!result) return result_error(result.error());
    return result_ok(result.value().value);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_computer_algebra_integrate_adaptive_by_name(ExprObj* expression,
                                                   const char* variable,
                                                   ExprObj* lower,
                                                   ExprObj* upper,
                                                   const double tolerance,
                                                   const LmInt max_depth) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    const auto* value = checked_expr(expression, error);
    if (!value) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    const auto* lower_value = checked_expr(lower, error);
    if (!lower_value) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    const auto* upper_value = checked_expr(upper, error);
    if (!upper_value) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    if (max_depth <= 0 || max_depth > std::numeric_limits<int>::max()) {
        return result_error(MathErrorCode::InvalidArgument, __func__, "Adaptive integration: invalid maximum depth");
    }
    const auto result = LMCAS::adaptive_simpson_numeric(
        *value, variable ? variable : "", *lower_value, *upper_value,
        tolerance, static_cast<int>(max_depth));
    if (!result) return result_error(result.error());
    return result_ok(result.value().value);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_computer_algebra_integrate_simpson_by_symbol(
    ExprObj* e, ExprObj* v, ExprObj* l, ExprObj* u, LmInt n) noexcept try {
    ensure_lmmc_runtime();
    std::string name, error;
    if (!checked_symbol_name(v, name, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return lmx_computer_algebra_integrate_simpson_by_name(e, name.c_str(), l, u, n);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_computer_algebra_integrate_gaussian_by_symbol(
    ExprObj* e, ExprObj* v, ExprObj* l, ExprObj* u, LmInt n) noexcept try {
    ensure_lmmc_runtime();
    std::string name, error;
    if (!checked_symbol_name(v, name, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return lmx_computer_algebra_integrate_gaussian_by_name(e, name.c_str(), l, u, n);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_computer_algebra_integrate_adaptive_by_symbol(
    ExprObj* e, ExprObj* v, ExprObj* l, ExprObj* u, double t, LmInt d) noexcept try {
    ensure_lmmc_runtime();
    std::string name, error;
    if (!checked_symbol_name(v, name, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return lmx_computer_algebra_integrate_adaptive_by_name(e, name.c_str(), l, u, t, d);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_computer_algebra_calculus_gradient_by_names(ExprObj* expression,
                                        ArrayObj* variables) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    const auto* value = checked_expr(expression, error);
    if (!value) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    std::vector<std::string> names;
    if (!array_strings(variables, names, error)) return result_error(MathErrorCode::InvalidArgument, __func__, error);
    return expr_array_result(LMCAS::gradient_checked(*value, names));
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_computer_algebra_calculus_divergence_by_names(
    ArrayObj* field, ArrayObj* variables) noexcept try {
    ensure_lmmc_runtime();
    std::vector<LMCAS::ExprPtr> expressions;
    std::vector<std::string> names;
    std::string error;
    if (!array_expressions(field, expressions, error) ||
        !array_strings(variables, names, error))
        return result_error(MathErrorCode::InvalidArgument, "lmx_computer_algebra_calculus_divergence_by_names",
                            std::move(error));
    return expr_result_ok(LMCAS::divergence_checked(expressions, names));
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_computer_algebra_calculus_curl_by_names(ArrayObj* field, ArrayObj* variables) noexcept try {
    ensure_lmmc_runtime();
    std::vector<LMCAS::ExprPtr> expressions;
    std::vector<std::string> names;
    std::string error;
    if (!array_expressions(field, expressions, error) ||
        !array_strings(variables, names, error)) return result_error(MathErrorCode::InvalidArgument, __func__, error);
    return expr_array_result(LMCAS::curl_checked(expressions, names));
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_computer_algebra_calculus_laplacian_by_names(
    ExprObj* expression, ArrayObj* variables) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    const auto* value = checked_expr(expression, error);
    std::vector<std::string> names;
    if (!value || !array_strings(variables, names, error))
        return result_error(MathErrorCode::InvalidArgument, "lmx_computer_algebra_calculus_laplacian_by_names",
                            std::move(error));
    return expr_result_ok(LMCAS::laplacian_checked(*value, names));
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_computer_algebra_calculus_jacobian_by_names(
    ArrayObj* functions, ArrayObj* variables) noexcept try {
    ensure_lmmc_runtime();
    std::vector<LMCAS::ExprPtr> expressions;
    std::vector<std::string> names;
    std::string error;
    if (!array_expressions(functions, expressions, error) ||
        !array_strings(variables, names, error))
        return result_error(MathErrorCode::InvalidArgument, "lmx_computer_algebra_calculus_jacobian_by_names",
                            std::move(error));
    return expr_result_ok(LMCAS::jacobian_checked(expressions, names));
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_computer_algebra_calculus_hessian_by_names(
    ExprObj* expression, ArrayObj* variables) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    const auto* value = checked_expr(expression, error);
    std::vector<std::string> names;
    if (!value || !array_strings(variables, names, error))
        return result_error(MathErrorCode::InvalidArgument, "lmx_computer_algebra_calculus_hessian_by_names",
                            std::move(error));
    return expr_result_ok(LMCAS::hessian_checked(*value, names));
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_computer_algebra_calculus_gradient_by_symbols(ExprObj* e, ArrayObj* v) noexcept try {
    ensure_lmmc_runtime();
    std::vector<std::string> names; std::string error;
    if (!math_internal::checked_symbol_names(v, names, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    const auto* value = checked_expr(e, error);
    if (!value) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return expr_array_result(LMCAS::gradient_checked(*value, names));
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_computer_algebra_calculus_divergence_by_symbols(ArrayObj* f, ArrayObj* v) noexcept try {
    ensure_lmmc_runtime();
    std::vector<LMCAS::ExprPtr> field; std::vector<std::string> names; std::string error;
    if (!array_expressions(f, field, error) || !math_internal::checked_symbol_names(v, names, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return expr_result_ok(LMCAS::divergence_checked(field, names));
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_computer_algebra_calculus_curl_by_symbols(ArrayObj* f, ArrayObj* v) noexcept try {
    ensure_lmmc_runtime();
    std::vector<LMCAS::ExprPtr> field; std::vector<std::string> names; std::string error;
    if (!array_expressions(f, field, error) || !math_internal::checked_symbol_names(v, names, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return expr_array_result(LMCAS::curl_checked(field, names));
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_computer_algebra_calculus_laplacian_by_symbols(ExprObj* e, ArrayObj* v) noexcept try {
    ensure_lmmc_runtime();
    std::vector<std::string> names; std::string error; const auto* value = checked_expr(e, error);
    if (!value || !math_internal::checked_symbol_names(v, names, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return expr_result_ok(LMCAS::laplacian_checked(*value, names));
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_computer_algebra_calculus_jacobian_by_symbols(ArrayObj* f, ArrayObj* v) noexcept try {
    ensure_lmmc_runtime();
    std::vector<LMCAS::ExprPtr> values; std::vector<std::string> names; std::string error;
    if (!array_expressions(f, values, error) || !math_internal::checked_symbol_names(v, names, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return expr_result_ok(LMCAS::jacobian_checked(values, names));
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_computer_algebra_calculus_hessian_by_symbols(ExprObj* e, ArrayObj* v) noexcept try {
    ensure_lmmc_runtime();
    std::vector<std::string> names; std::string error; const auto* value = checked_expr(e, error);
    if (!value || !math_internal::checked_symbol_names(v, names, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return expr_result_ok(LMCAS::hessian_checked(*value, names));
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_computer_algebra_calculus_volume_revolution_x(
    ExprObj* expression, ExprObj* lower, ExprObj* upper) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    const auto* value = checked_expr(expression, error);
    const auto* lower_value = checked_expr(lower, error);
    const auto* upper_value = checked_expr(upper, error);
    if (!value || !lower_value || !upper_value)
        return result_error(MathErrorCode::InvalidArgument,
                            "lmx_computer_algebra_calculus_volume_revolution_x", std::move(error));
    return expr_result_ok(LMCAS::volume_of_revolution_x_checked(
        *value, *lower_value, *upper_value));
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_computer_algebra_calculus_arc_length_x(
    ExprObj* expression, ExprObj* lower, ExprObj* upper) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    const auto* value = checked_expr(expression, error);
    const auto* lower_value = checked_expr(lower, error);
    const auto* upper_value = checked_expr(upper, error);
    if (!value || !lower_value || !upper_value)
        return result_error(MathErrorCode::InvalidArgument,
                            "lmx_computer_algebra_calculus_arc_length_x", std::move(error));
    return expr_result_ok(
        LMCAS::arc_length_x_checked(*value, *lower_value, *upper_value));
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API StringObj* lmx_computer_algebra_to_text(ExprObj* expr) noexcept try {
    ensure_lmmc_runtime();
    if (!expr) return new StringObj("CasError(InvalidArgument: null expr)");
    return new StringObj(expr->to_string());
} catch (...) {
    return nullptr;
}
