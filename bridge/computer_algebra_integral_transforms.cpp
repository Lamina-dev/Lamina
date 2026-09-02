#include "bridge/result.hpp"
#include "bridge/conversions.hpp"
#include "bridge/runtime_views.hpp"
#include "bridge/unit_bridge.hpp"
#include <cstdarg>
#include "bridge/math_internal.hpp"
#include "transform_engine.hpp"
#include "complex_analysis.hpp"
#include "series_engine.hpp"
#include "symbolic_implicit_diff.hpp"
using namespace lmx::bridge;

extern "C" LM_API AdtObj* lmx_computer_algebra_integral_transforms_laplace_by_names(
    ExprObj* expression, const char* time_variable,
    const char* frequency_variable) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    const auto* value = checked_expr(expression, error);
    if (!value)
        return result_error(MathErrorCode::InvalidArgument, "lmx_computer_algebra_integral_transforms_laplace_by_names",
                            std::move(error));
    return transform_engine_result_value(lamina::laplace_transform_checked(
        *value, time_variable ? time_variable : "",
        frequency_variable ? frequency_variable : ""));
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_computer_algebra_integral_transforms_inverse_laplace_by_names(
    ExprObj* expression, const char* frequency_variable,
    const char* time_variable) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    const auto* value = checked_expr(expression, error);
    if (!value)
        return result_error(MathErrorCode::InvalidArgument,
                            "lmx_computer_algebra_integral_transforms_inverse_laplace_by_names", std::move(error));
    return transform_engine_result_value(lamina::inverse_laplace_checked(
        *value, frequency_variable ? frequency_variable : "",
        time_variable ? time_variable : ""));
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_computer_algebra_integral_transforms_fourier_by_names(
    ExprObj* expression, const char* time_variable,
    const char* frequency_variable) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    const auto* value = checked_expr(expression, error);
    if (!value)
        return result_error(MathErrorCode::InvalidArgument, "lmx_computer_algebra_integral_transforms_fourier_by_names",
                            std::move(error));
    return transform_engine_result_value(lamina::fourier_transform_checked(
        *value, time_variable ? time_variable : "",
        frequency_variable ? frequency_variable : ""));
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_computer_algebra_integral_transforms_inverse_fourier_by_names(
    ExprObj* expression, const char* frequency_variable,
    const char* time_variable) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    const auto* value = checked_expr(expression, error);
    if (!value)
        return result_error(MathErrorCode::InvalidArgument,
                            "lmx_computer_algebra_integral_transforms_inverse_fourier_by_names", std::move(error));
    return transform_engine_result_value(lamina::inverse_fourier_transform_checked(
        *value, frequency_variable ? frequency_variable : "",
        time_variable ? time_variable : ""));
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_computer_algebra_integral_transforms_z_transform_by_names(
    ExprObj* expression, const char* index_variable,
    const char* frequency_variable) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    const auto* value = checked_expr(expression, error);
    if (!value)
        return result_error(MathErrorCode::InvalidArgument, "lmx_computer_algebra_integral_transforms_z_transform_by_names",
                            std::move(error));
    return transform_engine_result_value(lamina::z_transform_checked(
        *value, index_variable ? index_variable : "",
        frequency_variable ? frequency_variable : ""));
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_computer_algebra_integral_transforms_convolve_by_name(
    ExprObj* lhs, ExprObj* rhs, const char* variable) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    const auto* left = checked_expr(lhs, error);
    const auto* right = checked_expr(rhs, error);
    if (!left || !right)
        return result_error(MathErrorCode::InvalidArgument, "lmx_computer_algebra_integral_transforms_convolve_by_name",
                            std::move(error));
    return transform_engine_result_value(lamina::convolve_checked(
        *left, *right, variable ? variable : ""));
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_computer_algebra_residue_by_name(
    ExprObj* expression, const char* variable, ExprObj* point,
    const LmInt order) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    const auto* value = checked_expr(expression, error);
    const auto* point_value = checked_expr(point, error);
    if (!value || !point_value || order <= 0 ||
        order > std::numeric_limits<int>::max()) {
        return result_error(MathErrorCode::InvalidArgument, "lmx_computer_algebra_residue_by_name",
                            error.empty() ? "invalid order" : std::move(error));
    }
    return expr_result_ok(lamina::residue_checked(
        *value, variable ? variable : "", *point_value,
        static_cast<int>(order)));
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_computer_algebra_cauchy_integral_by_name(
    ExprObj* expression, const char* variable, ExprObj* point,
    const LmInt order) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    const auto* value = checked_expr(expression, error);
    const auto* point_value = checked_expr(point, error);
    if (!value || !point_value || order <= 0 ||
        order > std::numeric_limits<int>::max()) {
        return result_error(MathErrorCode::InvalidArgument,
                            "lmx_computer_algebra_cauchy_integral_by_name",
                            error.empty() ? "invalid order" : std::move(error));
    }
    return expr_result_ok(lamina::cauchy_integral_checked(
        *value, variable ? variable : "", *point_value,
        static_cast<int>(order)));
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_computer_algebra_is_analytic_by_name(ExprObj* expression,
                                           const char* variable) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    const auto* value = checked_expr(expression, error);
    if (!value) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    const auto result = lamina::is_analytic_checked(
        *value, variable ? variable : "");
    if (!result) return result_error(result.error());
    return result_ok(result.value());
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_computer_algebra_calculus_implicit_differentiate_by_names(
    ExprObj* expression, const char* independent, const char* dependent) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    const auto* value = checked_expr(expression, error);
    if (!value)
        return result_error(MathErrorCode::InvalidArgument,
                            "lmx_computer_algebra_calculus_implicit_differentiate_by_names", std::move(error));
    return expr_pointer_result(
        lamina::implicit_diff(*value, independent ? independent : "",
                              dependent ? dependent : ""),
        "lmx_computer_algebra_calculus_implicit_differentiate_by_names");
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_computer_algebra_series_laurent_series_by_name(
    ExprObj* expression, const char* variable, ExprObj* center,
    const LmInt negative_order, const LmInt positive_order) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    const auto* value = checked_expr(expression, error);
    const auto* center_value = checked_expr(center, error);
    if (!value || !center_value || negative_order < 0 || positive_order < 0 ||
        negative_order > std::numeric_limits<int>::max() ||
        positive_order > std::numeric_limits<int>::max()) {
        return result_error(MathErrorCode::InvalidArgument,
                            "lmx_computer_algebra_series_laurent_series_by_name",
                            error.empty() ? "invalid order" : std::move(error));
    }
    return expr_result_ok(lamina::laurent_series_checked(
        *value, variable ? variable : "", *center_value,
        static_cast<int>(negative_order), static_cast<int>(positive_order)));
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_computer_algebra_series_asymptotic_by_name(
    ExprObj* expression, const char* variable, const LmInt order) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    const auto* value = checked_expr(expression, error);
    if (!value || order <= 0 || order > std::numeric_limits<int>::max()) {
        return result_error(MathErrorCode::InvalidArgument, "lmx_computer_algebra_series_asymptotic_by_name",
                            error.empty() ? "invalid order" : std::move(error));
    }
    return expr_pointer_result(
        lamina::asymptotic_expand(*value, variable ? variable : "",
                                  static_cast<int>(order)),
        "lmx_computer_algebra_series_asymptotic_by_name");
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_computer_algebra_series_symbolic_sum_by_name(
    ExprObj* expression, const char* variable, ExprObj* lower,
    ExprObj* upper) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    const auto* value = checked_expr(expression, error);
    const auto* lower_value = checked_expr(lower, error);
    const auto* upper_value = checked_expr(upper, error);
    if (!value || !lower_value || !upper_value)
        return result_error(MathErrorCode::InvalidArgument,
                            "lmx_computer_algebra_series_symbolic_sum_by_name", std::move(error));
    return expr_pointer_result(
        lamina::symbolic_sum(*value, variable ? variable : "", *lower_value,
                             *upper_value),
        "lmx_computer_algebra_series_symbolic_sum_by_name");
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_computer_algebra_series_symbolic_product_by_name(
    ExprObj* expression, const char* variable, ExprObj* lower,
    ExprObj* upper) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    const auto* value = checked_expr(expression, error);
    const auto* lower_value = checked_expr(lower, error);
    const auto* upper_value = checked_expr(upper, error);
    if (!value || !lower_value || !upper_value)
        return result_error(MathErrorCode::InvalidArgument,
                            "lmx_computer_algebra_series_symbolic_product_by_name", std::move(error));
    return expr_pointer_result(
        lamina::symbolic_product(*value, variable ? variable : "",
                                 *lower_value, *upper_value),
        "lmx_computer_algebra_series_symbolic_product_by_name");
} catch (...) {
    return c_abi_current_exception(__func__);
}

/** @brief Symbol-argument Laplace transform. @param e Borrowed expression. @param a Borrowed time symbol. @param b Borrowed frequency symbol. @return Owning Expr or CasError. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_integral_transforms_laplace_by_symbols(ExprObj* e, ExprObj* a, ExprObj* b) noexcept try {
    ensure_lmmc_runtime();
    std::string x, y, error;
    if (!checked_symbol_name(a, x, error) || !checked_symbol_name(b, y, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return lmx_computer_algebra_integral_transforms_laplace_by_names(e, x.c_str(), y.c_str());
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Symbol-argument inverse Laplace transform. @param e Borrowed expression. @param a Borrowed frequency symbol. @param b Borrowed time symbol. @return Owning Expr or CasError. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_integral_transforms_inverse_laplace_by_symbols(ExprObj* e, ExprObj* a, ExprObj* b) noexcept try {
    ensure_lmmc_runtime();
    std::string x, y, error;
    if (!checked_symbol_name(a, x, error) || !checked_symbol_name(b, y, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return lmx_computer_algebra_integral_transforms_inverse_laplace_by_names(e, x.c_str(), y.c_str());
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Symbol-argument Fourier transform. @param e Borrowed expression. @param a Borrowed time symbol. @param b Borrowed frequency symbol. @return Owning Expr or CasError. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_integral_transforms_fourier_by_symbols(ExprObj* e, ExprObj* a, ExprObj* b) noexcept try {
    ensure_lmmc_runtime();
    std::string x, y, error;
    if (!checked_symbol_name(a, x, error) || !checked_symbol_name(b, y, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return lmx_computer_algebra_integral_transforms_fourier_by_names(e, x.c_str(), y.c_str());
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Symbol-argument inverse Fourier transform. @param e Borrowed expression. @param a Borrowed frequency symbol. @param b Borrowed time symbol. @return Owning Expr or CasError. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_integral_transforms_inverse_fourier_by_symbols(ExprObj* e, ExprObj* a, ExprObj* b) noexcept try {
    ensure_lmmc_runtime();
    std::string x, y, error;
    if (!checked_symbol_name(a, x, error) || !checked_symbol_name(b, y, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return lmx_computer_algebra_integral_transforms_inverse_fourier_by_names(e, x.c_str(), y.c_str());
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Symbol-argument Z transform. @param e Borrowed expression. @param a Borrowed index symbol. @param b Borrowed frequency symbol. @return Owning Expr or CasError. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_integral_transforms_z_transform_by_symbols(ExprObj* e, ExprObj* a, ExprObj* b) noexcept try {
    ensure_lmmc_runtime();
    std::string x, y, error;
    if (!checked_symbol_name(a, x, error) || !checked_symbol_name(b, y, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return lmx_computer_algebra_integral_transforms_z_transform_by_names(e, x.c_str(), y.c_str());
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Symbol-argument convolution. @param a Borrowed left expression. @param b Borrowed right expression. @param v Borrowed variable symbol. @return Owning Expr or CasError. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_integral_transforms_convolve_by_symbol(ExprObj* a, ExprObj* b, ExprObj* v) noexcept try {
    ensure_lmmc_runtime();
    std::string name, error;
    if (!checked_symbol_name(v, name, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return lmx_computer_algebra_integral_transforms_convolve_by_name(a, b, name.c_str());
} catch (...) {
    return c_abi_current_exception(__func__);
}

/** @brief Symbol-argument residue. @param e Borrowed expression. @param v Borrowed variable symbol. @param p Borrowed point. @param n Positive order. @return Owning Expr or CasError. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_residue_by_symbol(ExprObj* e, ExprObj* v, ExprObj* p, LmInt n) noexcept try {
    ensure_lmmc_runtime();
    std::string name, error;
    if (!checked_symbol_name(v, name, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return lmx_computer_algebra_residue_by_name(e, name.c_str(), p, n);
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Symbol-argument Cauchy integral. @param e Borrowed expression. @param v Borrowed variable symbol. @param p Borrowed point. @param n Positive order. @return Owning Expr or CasError. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_cauchy_integral_by_symbol(ExprObj* e, ExprObj* v, ExprObj* p, LmInt n) noexcept try {
    ensure_lmmc_runtime();
    std::string name, error;
    if (!checked_symbol_name(v, name, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return lmx_computer_algebra_cauchy_integral_by_name(e, name.c_str(), p, n);
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Symbol-argument analyticity query. @param e Borrowed expression. @param v Borrowed variable symbol. @return Owning Result bool or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_is_analytic_by_symbol(ExprObj* e, ExprObj* v) noexcept try {
    ensure_lmmc_runtime();
    std::string name, error;
    if (!checked_symbol_name(v, name, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return lmx_computer_algebra_is_analytic_by_name(e, name.c_str());
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Symbol-argument implicit differentiation. @param e Borrowed expression. @param a Borrowed independent symbol. @param b Borrowed dependent symbol. @return Owning Expr or CasError. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_calculus_implicit_differentiate_by_symbols(ExprObj* e, ExprObj* a, ExprObj* b) noexcept try {
    ensure_lmmc_runtime();
    std::string x, y, error;
    if (!checked_symbol_name(a, x, error) || !checked_symbol_name(b, y, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return lmx_computer_algebra_calculus_implicit_differentiate_by_names(e, x.c_str(), y.c_str());
} catch (...) {
    return c_abi_current_exception(__func__);
}

/** @brief Symbol-variable Laurent series. @param e Borrowed expression. @param v Borrowed symbol. @param c Borrowed center. @param n Negative order. @param p Positive order. @return Owning Expr or CasError. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_series_laurent_series_by_symbol(ExprObj* e, ExprObj* v, ExprObj* c, LmInt n, LmInt p) noexcept try {
    ensure_lmmc_runtime();
    std::string name, error;
    if (!checked_symbol_name(v, name, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return lmx_computer_algebra_series_laurent_series_by_name(e, name.c_str(), c, n, p);
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Symbol-variable asymptotic expansion. @param e Borrowed expression. @param v Borrowed symbol. @param n Positive order. @return Owning Expr or CasError. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_series_asymptotic_by_symbol(ExprObj* e, ExprObj* v, LmInt n) noexcept try {
    ensure_lmmc_runtime();
    std::string name, error;
    if (!checked_symbol_name(v, name, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return lmx_computer_algebra_series_asymptotic_by_name(e, name.c_str(), n);
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Symbol-variable symbolic sum. @param e Borrowed expression. @param v Borrowed symbol. @param l Borrowed lower bound. @param u Borrowed upper bound. @return Owning Expr or CasError. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_series_symbolic_sum_by_symbol(ExprObj* e, ExprObj* v, ExprObj* l, ExprObj* u) noexcept try {
    ensure_lmmc_runtime();
    std::string name, error;
    if (!checked_symbol_name(v, name, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return lmx_computer_algebra_series_symbolic_sum_by_name(e, name.c_str(), l, u);
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Symbol-variable symbolic product. @param e Borrowed expression. @param v Borrowed symbol. @param l Borrowed lower bound. @param u Borrowed upper bound. @return Owning Expr or CasError. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_series_symbolic_product_by_symbol(ExprObj* e, ExprObj* v, ExprObj* l, ExprObj* u) noexcept try {
    ensure_lmmc_runtime();
    std::string name, error;
    if (!checked_symbol_name(v, name, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return lmx_computer_algebra_series_symbolic_product_by_name(e, name.c_str(), l, u);
} catch (...) {
    return c_abi_current_exception(__func__);
}
