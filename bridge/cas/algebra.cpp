#include "bridge/math_internal.hpp"
#include "symbolic.hpp"
#include "poly_utils.hpp"
#include "transcendental_factor.hpp"

using namespace lmx::bridge;

using lmx::bridge::math_internal::checked_expression_operation;

extern "C" LM_API AdtObj* lmx_cas_factor(ExprObj* value) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    const auto* expression = checked_expr(value, error);
    if (!expression)
        return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    LMCAS::ComputationContext context;
    auto result = (*expression)->factor_checked(context);
    if (!result) return result_error(result.error());
    return result_ok(new ExprObj(result.value()), ValueKind::Expr);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_cas_cancel(ExprObj* value) noexcept try {
    ensure_lmmc_runtime();
    return checked_expression_operation("algebra.cancel", value,
        [](const auto& expression) { return expression->cancel(); });
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_cas_simplify_trigonometric(ExprObj* value) noexcept try {
    ensure_lmmc_runtime();
    return checked_expression_operation("algebra.simplify_trig", value,
        [](const auto& expression) { return expression->simplify_trig(); });
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_cas_polynomial_gcd(
    ExprObj* lhs, ExprObj* rhs) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    const auto* left = checked_expr(lhs, error);
    const auto* right = checked_expr(rhs, error);
    if (!left || !right) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    LMCAS::ComputationContext context;
    auto result = LMCAS::symbolic_polynomial_gcd(
        **left, **right, context);
    if (!result) return result_error(result.error());
    return result_ok(new ExprObj(result.value()), ValueKind::Expr);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_cas_polynomial_resultant(
    ExprObj* lhs, ExprObj* rhs, const char* variable) noexcept try {
    ensure_lmmc_runtime();
    if (!variable)
        return result_error(MathErrorCode::InvalidArgument, __func__, "algebra.polynomial_resultant: null variable");
    std::string error;
    const auto* left = checked_expr(lhs, error);
    const auto* right = checked_expr(rhs, error);
    if (!left || !right) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return checked_expression_operation(
        "algebra.polynomial_resultant", *left, [&](const auto& expression) {
            return LMCAS::SymbolicExpr::poly_resultant(
                expression, *right, variable);
        });
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_cas_factor_multivariate(ExprObj* value) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    const auto* expression = checked_expr(value, error);
    if (!expression)
        return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return expr_result_ok((*expression)->factor_checked());
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_cas_factor_transcendental_by_name(
    ExprObj* value, const char* variable) noexcept try {
    ensure_lmmc_runtime();
    if (!variable || variable[0] == '\0')
        return result_error(MathErrorCode::InvalidArgument, __func__, "CasError(InvalidArgument in algebra.factor_transcendental: empty variable)");
    std::string error;
    const auto* expression = checked_expr(value, error);
    if (!expression) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    auto factored = LMCAS::factor_transcendental(*expression, variable);
    if (!factored) return result_error(factored.error());
    return math_internal::unordered_expr_result(factored.value());
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_cas_factor_transcendental_by_symbol(
    ExprObj* value, ExprObj* variable) noexcept try {
    ensure_lmmc_runtime();
    std::string name, error;
    if (!checked_symbol_name(variable, name, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return lmx_cas_factor_transcendental_by_name(value, name.c_str());
} catch (...) {
    return c_abi_current_exception(__func__);
}
