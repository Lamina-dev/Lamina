#include "bridge/result.hpp"
#include "bridge/conversions.hpp"
#include "bridge/runtime_views.hpp"
#include "bridge/unit_bridge.hpp"
#include <cstdarg>
#include "bridge/math_internal.hpp"
#include "symbolic.hpp"

using namespace lmx::bridge;

namespace {
template <typename Operation>
AdtObj* checked_expression_operation(
    const char* name, ExprObj* value, Operation operation) {
    std::string error;
    const auto* expression = checked_expr(value, error);
    if (!expression) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    try {
        auto output = operation(*expression);
        if (!output)
            return result_error(MathErrorCode::UnsupportedExpression, __func__, 
                std::string("CasError(UnsupportedExpression in ") + name + ")");
        return result_ok(new ExprObj(std::move(output)), ValueKind::Expr);
    } catch (const std::exception& error) {
        return result_error(MathErrorCode::InternalError, __func__, 
            std::string("CasError(InternalInvariant in ") + name +
            ": " + error.what() + ")");
    }
}
} // namespace

extern "C" LM_API AdtObj* lmx_computer_algebra_algebra_factor(ExprObj* value) {
    return checked_expression_operation("algebra.factor", value,
        [](const auto& expression) { return expression->factor(); });
}
extern "C" LM_API AdtObj* lmx_computer_algebra_algebra_cancel(ExprObj* value) {
    return checked_expression_operation("algebra.cancel", value,
        [](const auto& expression) { return expression->cancel(); });
}
extern "C" LM_API AdtObj* lmx_computer_algebra_algebra_simplify_trigonometric(ExprObj* value) {
    return checked_expression_operation("algebra.simplify_trig", value,
        [](const auto& expression) { return expression->simplify_trig(); });
}
extern "C" LM_API AdtObj* lmx_computer_algebra_algebra_polynomial_greatest_common_divisor(
    ExprObj* lhs, ExprObj* rhs) {
    std::string error;
    const auto* left = checked_expr(lhs, error);
    const auto* right = checked_expr(rhs, error);
    if (!left || !right) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return checked_expression_operation("algebra.polynomial_greatest_common_divisor", lhs, [&](const auto&) {
        return SymbolicExpr::poly_gcd(*left, *right);
    });
}
extern "C" LM_API AdtObj* lmx_computer_algebra_algebra_polynomial_resultant(
    ExprObj* lhs, ExprObj* rhs, const char* variable) {
    if (!variable)
        return result_error(MathErrorCode::InvalidArgument, __func__, "algebra.polynomial_resultant: null variable");
    std::string error;
    const auto* left = checked_expr(lhs, error);
    const auto* right = checked_expr(rhs, error);
    if (!left || !right) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return checked_expression_operation("algebra.polynomial_resultant", lhs, [&](const auto&) {
        return SymbolicExpr::poly_resultant(*left, *right, variable);
    });
}
