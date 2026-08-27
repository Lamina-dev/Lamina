#pragma once

#include "bridge/result.hpp"
#include "bridge/conversions.hpp"
#include "bridge/runtime_views.hpp"
#include "bridge/unit_bridge.hpp"

namespace lmx::bridge::math_internal {

ArrayObj* solution_tables(
    const std::vector<std::map<std::string, lamina::lsr::ExprPtr>>& solutions);
bool checked_symbol_names(
    ArrayObj* values, std::vector<std::string>& names, std::string& error);
bool nested_expressions(
    ArrayObj* rows,
    std::vector<std::vector<lamina::lsr::ExprPtr>>& output,
    std::string& error);
AdtObj* unordered_expr_result(std::vector<lamina::lsr::ExprPtr> values);
ArrayObj* symbol_text_array(ArrayObj* symbols, std::string& error);
AdtObj* checked_expr_result(const lamina::ExpressionResult& result);

template <typename Operation>
AdtObj* checked_expression_operation(
    const char* name, ExprObj* value, Operation operation) {
    std::string error;
    const auto* expression = checked_expr(value, error);
    if (!expression) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    try {
        auto output = operation(*expression);
        if (!output) {
            return result_error(MathErrorCode::UnsupportedExpression, __func__, 
                std::string("CasError(UnsupportedExpression in ") + name + ")");
        }
        return result_ok(new ExprObj(std::move(output)), ValueKind::Expr);
    } catch (const std::exception& exception) {
        return result_error(MathErrorCode::InternalError, __func__, 
            std::string("CasError(InternalInvariant in ") + name +
            ": " + exception.what() + ")");
    }
}

} // namespace lmx::bridge::math_internal

using lmx::runtime::AdtObj;
using lmx::runtime::ExprObj;

extern "C" {
AdtObj* lmx_computer_algebra_integrate_simpson_by_name(ExprObj*, const char*, ExprObj*, ExprObj*, LmInt);
AdtObj* lmx_computer_algebra_integrate_gaussian_by_name(ExprObj*, const char*, ExprObj*, ExprObj*, LmInt);
AdtObj* lmx_computer_algebra_integrate_adaptive_by_name(ExprObj*, const char*, ExprObj*, ExprObj*, double, LmInt);
AdtObj* lmx_computer_algebra_integral_transforms_laplace_by_names(ExprObj*, const char*, const char*);
AdtObj* lmx_computer_algebra_integral_transforms_inverse_laplace_by_names(ExprObj*, const char*, const char*);
AdtObj* lmx_computer_algebra_integral_transforms_fourier_by_names(ExprObj*, const char*, const char*);
AdtObj* lmx_computer_algebra_integral_transforms_inverse_fourier_by_names(ExprObj*, const char*, const char*);
AdtObj* lmx_computer_algebra_integral_transforms_z_transform_by_names(ExprObj*, const char*, const char*);
AdtObj* lmx_computer_algebra_integral_transforms_convolve_by_name(ExprObj*, ExprObj*, const char*);
AdtObj* lmx_computer_algebra_residue_by_name(ExprObj*, const char*, ExprObj*, LmInt);
AdtObj* lmx_computer_algebra_cauchy_integral_by_name(ExprObj*, const char*, ExprObj*, LmInt);
AdtObj* lmx_computer_algebra_is_analytic_by_name(ExprObj*, const char*);
AdtObj* lmx_computer_algebra_calculus_implicit_differentiate_by_names(ExprObj*, const char*, const char*);
AdtObj* lmx_computer_algebra_series_laurent_series_by_name(ExprObj*, const char*, ExprObj*, LmInt, LmInt);
AdtObj* lmx_computer_algebra_series_asymptotic_by_name(ExprObj*, const char*, LmInt);
AdtObj* lmx_computer_algebra_series_symbolic_sum_by_name(ExprObj*, const char*, ExprObj*, ExprObj*);
AdtObj* lmx_computer_algebra_series_symbolic_product_by_name(ExprObj*, const char*, ExprObj*, ExprObj*);
AdtObj* lmx_computer_algebra_complex_analysis_analytic_continuation_by_name(ExprObj*, const char*);
AdtObj* lmx_computer_algebra_ordinary_differential_equations_solve_separable_by_names(ExprObj*, const char*, const char*);
AdtObj* lmx_computer_algebra_algebra_polynomial_greatest_common_divisor(ExprObj*, ExprObj*);
AdtObj* lmx_computer_algebra_ordinary_differential_equations_solve_first_order_linear_by_names(ExprObj*, ExprObj*, const char*, const char*);
AdtObj* lmx_computer_algebra_ordinary_differential_equations_solve_second_order_linear_by_names(double, double, double, ExprObj*, const char*, const char*);
}
