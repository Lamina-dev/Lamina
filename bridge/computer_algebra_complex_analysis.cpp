#include "bridge/result.hpp"
#include "bridge/conversions.hpp"
#include "bridge/runtime_views.hpp"
#include "bridge/unit_bridge.hpp"
#include <cstdarg>
#include "bridge/math_internal.hpp"
#include "complex_analysis.hpp"
#include "symbolic.hpp"
#include "symbolic_complex.hpp"

using namespace lmx::bridge;
using lmx::bridge::math_internal::checked_expression_operation;
using lmx::bridge::math_internal::checked_expr_result;

namespace {
lamina::lsr::ExprResult complex_expression(const lamina::ComplexSymbolic& value) {
    return lamina::lsr::complex(value.real, value.imag);
}
} // namespace

/** @brief Extracts symbolic complex argument. @param value Borrowed expression. @return Owning Result Expr or error. @ownership Input borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_complex_analysis_argument(ExprObj* value) noexcept try {
    ensure_lmmc_runtime();
    std::string error; const auto* expression = checked_expr(value, error);
    if (!expression) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    const auto real = lamina::real_part_checked(*expression);
    const auto imag = lamina::imag_part_checked(*expression);
    if (!real) return result_error(real.error());
    if (!imag) return result_error(imag.error());
    return checked_expr_result(lamina::complex_arg_checked(
        lamina::ComplexSymbolic{real.value(), imag.value()}));
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Builds exponential-form symbolic complex expression. @param radius Borrowed radius. @param angle Borrowed angle. @return Owning Result Expr or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_complex_analysis_exponential_form(
    ExprObj* radius, ExprObj* angle) noexcept try {
    ensure_lmmc_runtime();
    std::string error; const auto* r = checked_expr(radius, error);
    const auto* t = checked_expr(angle, error);
    if (!r || !t) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    const auto result = lamina::complex_exp_form_checked(*r, *t);
    if (!result) return result_error(result.error());
    const auto expression = complex_expression(result.value());
    if (!expression) return result_error(expression.error());
    return result_ok(new ExprObj(expression.value()), ValueKind::Expr);
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Builds trigonometric-form symbolic complex expression. @param radius Borrowed radius. @param angle Borrowed angle. @return Owning Result Expr or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_complex_analysis_trigonometric_form(
    ExprObj* radius, ExprObj* angle) noexcept try {
    ensure_lmmc_runtime();
    std::string error; const auto* r = checked_expr(radius, error);
    const auto* t = checked_expr(angle, error);
    if (!r || !t) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    const auto result = lamina::complex_trig_form_checked(*r, *t);
    if (!result) return result_error(result.error());
    const auto expression = complex_expression(result.value());
    if (!expression) return result_error(expression.error());
    return result_ok(new ExprObj(expression.value()), ValueKind::Expr);
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Computes unordered symbolic nth roots. @param value Borrowed radicand. @param degree Positive degree. @return Owning Result set or error. @ownership Input borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_complex_analysis_nth_roots(
    ExprObj* value, LmInt degree) noexcept try {
    ensure_lmmc_runtime();
    if (degree <= 0 || degree > std::numeric_limits<int>::max())
        return result_error(MathErrorCode::InvalidArgument, __func__, "complex.nth_roots: invalid degree");
    std::string error; const auto* expression = checked_expr(value, error);
    if (!expression) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    auto roots_result = lamina::solve_complex_nth_root_checked(
        *expression, static_cast<int>(degree));
    if (!roots_result ||
        roots_result.value().size() != static_cast<std::size_t>(degree)) {
        const auto approximate = lamina::lsr::evalf(**expression);
        if (approximate && approximate.value().is_finite()) {
            roots_result = lamina::solve_complex_nth_root_checked(
                SymbolicExpr::number(approximate.value().value),
                static_cast<int>(degree));
        }
    }
    if (!roots_result) return result_error(roots_result.error());
    auto roots = std::move(roots_result.value());
    if (roots.size() != static_cast<std::size_t>(degree)) {
        return result_error(
            MathErrorCode::Inconclusive, __func__,
            "CasError(Inconclusive in complex.nth_roots)");
    }
    std::vector<lamina::lsr::ExprPtr> values;
    for (const auto& root : roots) {
        const auto converted = complex_expression(root);
        if (!converted)
            return result_error(converted.error());
        values.push_back(converted.value());
    }
    return math_internal::unordered_expr_result(std::move(values));
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Computes unordered symbolic quadratic roots. @param a Borrowed quadratic coefficient. @param b Borrowed linear coefficient. @param c Borrowed constant. @return Owning Result set or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_complex_analysis_quadratic_roots(
    ExprObj* a, ExprObj* b, ExprObj* c) noexcept try {
    ensure_lmmc_runtime();
    std::string error; const auto* av = checked_expr(a, error);
    const auto* bv = checked_expr(b, error); const auto* cv = checked_expr(c, error);
    if (!av || !bv || !cv) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    const auto roots =
        lamina::solve_complex_quadratic_checked(*av, *bv, *cv);
    if (!roots) return result_error(roots.error());
    std::vector<lamina::lsr::ExprPtr> values;
    for (const auto& root : roots.value()) {
        const auto converted = complex_expression(root);
        if (!converted) return result_error(converted.error());
        values.push_back(converted.value());
    }
    return math_internal::unordered_expr_result(std::move(values));
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Performs symbolic analytic continuation. @param value Borrowed expression. @param variable Borrowed complex variable name. @return Owning Result Expr or error. @ownership Input borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_complex_analysis_analytic_continuation_by_name(
    ExprObj* value, const char* variable) noexcept try {
    ensure_lmmc_runtime();
    if (!variable || variable[0] == '\0')
        return result_error(MathErrorCode::InvalidArgument, __func__, "complex.analytic_continuation: empty variable");
    return checked_expression_operation("complex.analytic_continuation", value,
        [&](const auto& expression) {
            return lamina::analytic_continuation(expression, variable);
        });
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Extracts symbolic real part. @param value Borrowed expression. @return Owning Result Expr or error. @ownership Input borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_complex_analysis_real_part(ExprObj* value) noexcept try {
    ensure_lmmc_runtime();
    std::string error; const auto* expression = checked_expr(value, error);
    if (!expression) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return checked_expr_result(lamina::real_part_checked(*expression));
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Extracts symbolic imaginary part. @param value Borrowed expression. @return Owning Result Expr or error. @ownership Input borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_complex_analysis_imag_part(ExprObj* value) noexcept try {
    ensure_lmmc_runtime();
    std::string error; const auto* expression = checked_expr(value, error);
    if (!expression) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return checked_expr_result(lamina::imag_part_checked(*expression));
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Computes symbolic conjugate. @param value Borrowed expression. @return Owning Result Expr or error. @ownership Input borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_complex_analysis_conjugate(ExprObj* value) noexcept try {
    ensure_lmmc_runtime();
    std::string error; const auto* expression = checked_expr(value, error);
    if (!expression) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return checked_expr_result(lamina::conjugate_checked(*expression));
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_computer_algebra_complex_analysis_analytic_continuation_by_symbol(ExprObj* value, ExprObj* variable) noexcept try {
    ensure_lmmc_runtime();
    std::string name, error; if (!checked_symbol_name(variable, name, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return lmx_computer_algebra_complex_analysis_analytic_continuation_by_name(value, name.c_str());
} catch (...) {
    return c_abi_current_exception(__func__);
}
