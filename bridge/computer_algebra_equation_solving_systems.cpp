#include "bridge/result.hpp"
#include "bridge/conversions.hpp"
#include "bridge/runtime_views.hpp"
#include "bridge/unit_bridge.hpp"
#include <cstdarg>
#include "bridge/math_internal.hpp"
#include "symbolic.hpp"
#include "inequality_solver.hpp"

using namespace lmx::bridge;

// equation solving systems (moved from the former assumptions translation unit)

namespace {
ArrayObj* solution_tables(
    const std::vector<std::map<std::string,
        std::shared_ptr<SymbolicExpr>>>& solutions) {
    auto* result = new ArrayObj();
    for (const auto& solution : solutions) {
        std::vector<TableObj::Entry> entries;
        for (const auto& [name, expression] : solution)
            entries.emplace_back(
                name, Value(new ExprObj(expression), ValueKind::Expr));
        result->append(
            Value(new TableObj(std::move(entries)), ValueKind::Table));
    }
    return result;
}
} // namespace

extern "C" LM_API AdtObj* lmx_computer_algebra_equation_solving_system_by_names(
    ArrayObj* equations, ArrayObj* variables) {
    std::vector<lamina::lsr::ExprPtr> checked_equations;
    std::vector<std::string> checked_variables;
    std::string error;
    if (!array_expressions(equations, checked_equations, error) ||
        !array_strings(variables, checked_variables, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "solve.system: " + error);
    try {
        return result_ok(
            solution_tables(SymbolicExpr::solve_system(
                checked_equations, checked_variables)),
            ValueKind::Obj);
    } catch (const std::exception& error) {
        return result_error(MathErrorCode::InvalidArgument, __func__, std::string("solve.system: ") + error.what());
    }
}

/**
 * @brief Solves a symbolic equation system with its unknowns supplied as symbols.
 * @param equations Borrowed Lamina array of valid equation expressions.
 * @param variables Borrowed Lamina array whose elements are single-symbol expressions.
 * @return `Result.Ok(array<table>)` for solved systems or `Result.Err(text)` on invalid
 *         inputs, unsupported systems, or algorithm failure.
 * @ownership Caller owns the returned ADT and payload; inputs are borrowed.
 * @threadsafe Current VM thread only.
 */
extern "C" LM_API AdtObj* lmx_computer_algebra_equation_solving_system_by_symbols(
    ArrayObj* equations, ArrayObj* variables) {
    if (!variables) return result_error(MathErrorCode::InvalidArgument, __func__, "solve.system: null array");
    std::vector<std::string> names;
    names.reserve(static_cast<std::size_t>(variables->len()));
    std::string error;
    for (const auto& value : variables->values()) {
        if (value.kind != ValueKind::Expr || !value.obj) {
            return result_error(MathErrorCode::InvalidArgument, __func__, 
                "CasError(InvalidArgument: variable array contains a non-expression value)");
        }
        std::string name;
        if (!checked_symbol_name(
                reinterpret_cast<ExprObj*>(value.obj), name, error)) {
            return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
        }
        names.push_back(std::move(name));
    }
    std::vector<lamina::lsr::ExprPtr> checked_equations;
    if (!array_expressions(equations, checked_equations, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "solve.system: " + error);
    try {
        return result_ok(
            solution_tables(SymbolicExpr::solve_system(checked_equations, names)),
            ValueKind::Obj);
    } catch (const std::exception& exception) {
        return result_error(MathErrorCode::InvalidArgument, __func__, 
            std::string("solve.system: ") + exception.what());
    }
}


extern "C" LM_API AdtObj* lmx_computer_algebra_equation_solving_inequality(
    ExprObj* expression, const char* relation, const char* variable) {
    if (!relation || !variable)
        return result_error(MathErrorCode::InvalidArgument, __func__, "solve.inequality: invalid argument");
    std::string error;
    const auto* checked = checked_expr(expression, error);
    if (!checked) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    const std::string relation_name(relation);
    std::optional<lamina::InequalityType> type;
    if (relation_name == "<") type = lamina::InequalityType::LessThan;
    else if (relation_name == "<=") type = lamina::InequalityType::LessEqual;
    else if (relation_name == ">") type = lamina::InequalityType::GreaterThan;
    else if (relation_name == ">=") type = lamina::InequalityType::GreaterEqual;
    if (!type) return result_error(MathErrorCode::InvalidArgument, __func__, "solve.inequality: unknown relation");
    const auto result = lamina::InequalitySolver::solve_inequality_checked(
        *checked, *type, variable);
    if (!result) return result_error(result.error());
    const auto& intervals = result.value().intervals();
    if (intervals.size() == 1 && intervals.front().lower.value &&
        intervals.front().upper.value &&
        !intervals.front().lower.is_neg_infinity &&
        !intervals.front().upper.is_pos_infinity) {
        return result_ok(
            expr_from_result(lamina::lsr::interval(
                intervals.front().lower.value, intervals.front().upper.value,
                !intervals.front().lower.is_open,
                !intervals.front().upper.is_open)),
            ValueKind::Expr);
    }
    return result_ok(
        new ExprObj(result.value().to_expr(variable)), ValueKind::Expr);
}
