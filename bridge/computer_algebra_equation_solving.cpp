#include "bridge/result.hpp"
#include "bridge/conversions.hpp"
#include "bridge/runtime_views.hpp"
#include "bridge/unit_bridge.hpp"
#include <cstdarg>

using namespace lmx::bridge;

extern "C" LM_API AdtObj* lmx_computer_algebra_equation_solving_solve_by_name(ExprObj* equation,
                                      const char* variable) {
    std::string error;
    const auto* value = checked_expr(equation, error);
    if (!value) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return expression_set_literal_result(
        lamina::lsr::solve(*value, variable ? variable : ""));
}

extern "C" LM_API AdtObj* lmx_computer_algebra_equation_solving_roots_by_name(ExprObj* expression,
                                      const char* variable) {
    std::string error;
    const auto* value = checked_expr(expression, error);
    if (!value) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return expression_set_literal_result(
        lamina::lsr::roots(*value, variable ? variable : ""));
}

/**
 * @brief Solves an equation for a variable supplied as a symbol expression.
 * @param equation Borrowed valid Lamina equation expression.
 * @param variable Borrowed Lamina expression containing exactly one symbol.
 * @return `Result.Ok(set<expr>)` for a determinate solution set, including empty;
 *         otherwise `Result.Err(text)`.
 * @ownership Caller owns the returned ADT and its set; inputs are borrowed.
 * @threadsafe Current VM thread only.
 */
extern "C" LM_API AdtObj* lmx_computer_algebra_equation_solving_symbol(ExprObj* equation, ExprObj* variable) {
    std::string name;
    std::string error;
    if (!checked_symbol_name(variable, name, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    const auto* value = checked_expr(equation, error);
    if (!value) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return expression_set_literal_result(lamina::lsr::solve(*value, name));
}

/**
 * @brief Finds roots using a variable supplied as a symbol expression.
 * @param expression Borrowed valid Lamina expression.
 * @param variable Borrowed Lamina expression containing exactly one symbol.
 * @return `Result.Ok(set<expr>)` for a determinate root set, including empty;
 *         otherwise `Result.Err(text)`.
 * @ownership Caller owns the returned ADT and its set; inputs are borrowed.
 * @threadsafe Current VM thread only.
 */
extern "C" LM_API AdtObj* lmx_computer_algebra_equation_solving_roots_by_symbol(ExprObj* expression, ExprObj* variable) {
    std::string name;
    std::string error;
    if (!checked_symbol_name(variable, name, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    const auto* value = checked_expr(expression, error);
    if (!value) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return expression_set_literal_result(lamina::lsr::roots(*value, name));
}
