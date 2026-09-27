#pragma once

#include "bridge/result.hpp"
#include "bridge/conversions.hpp"
#include "bridge/runtime_views.hpp"
#include "bridge/unit_bridge.hpp"

namespace lmx::bridge::math_internal {

ArrayObj* solution_tables(
    const std::vector<std::map<std::string, LMCAS::ExprPtr>>& solutions);
ArrayObj* solution_tables(
    const std::vector<std::map<std::string, LMCAS::SymbolicExpr>>& solutions);
bool checked_symbol_names(
    ArrayObj* values, std::vector<std::string>& names, std::string& error,
    const char* non_expression_message =
        "CasError(InvalidArgument: symbol array contains a non-expression value)");
bool nested_expressions(
    ArrayObj* rows,
    std::vector<std::vector<LMCAS::ExprPtr>>& output,
    std::string& error);
AdtObj* unordered_expr_result(std::vector<LMCAS::ExprPtr> values);
ArrayObj* symbol_text_array(ArrayObj* symbols, std::string& error);

template <typename Operation>
AdtObj* checked_expression_operation(
    const char* name, const LMCAS::ExprPtr& expression, Operation operation) {
    auto output = operation(expression);
    if (!output) {
        return result_error(MathErrorCode::UnsupportedExpression, __func__,
            std::string("CasError(UnsupportedExpression in ") + name + ")");
    }
    return result_ok(new ExprObj(std::move(output)), ValueKind::Expr);
}

template <typename Operation>
AdtObj* checked_expression_operation(
    const char* name, ExprObj* value, Operation operation) {
    std::string error;
    const auto* expression = checked_expr(value, error);
    if (!expression) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return checked_expression_operation(name, *expression, std::move(operation));
}

}
