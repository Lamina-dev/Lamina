#include "bridge/math_internal.hpp"
#include "symbolic.hpp"
#include "solver.hpp"
#include "parametric_solver.hpp"

using namespace lmx::bridge;

namespace {

AdtObj* polynomial_system_result(
    const std::vector<LMCAS::ExprPtr>& equations,
    const std::vector<std::string>& variables,
    const char* operation) noexcept try {
    std::vector<LMCAS::SymbolicExpr> values;
    values.reserve(equations.size());
    for (const auto& equation : equations) values.push_back(*equation);
    const auto solutions =
        LMCAS::Solver::solve_polynomial_system_checked(values, variables);
    if (!solutions) return result_error(solutions.error());
    return result_ok(
        math_internal::solution_tables(solutions.value()), ValueKind::Obj);
} catch (...) {
    return c_abi_current_exception(operation);
}

AdtObj* parametric_piecewise_result(
    const std::vector<LMCAS::ExprPtr>& equations,
    const std::vector<std::string>& unknowns,
    const std::vector<std::string>& parameters,
    const char* operation) noexcept try {
    const auto piecewise = LMCAS::ParametricSolver::solve_system_piecewise(
        equations, unknowns, parameters);
    auto cases = make_owned_object<ArrayObj>();
    for (const auto& item : piecewise.cases) {
        if (!item.condition) {
            return result_error(MathErrorCode::InternalError, operation,
                "CasError(InternalInvariant in solve.parametric_piecewise: null condition)");
        }
        std::vector<Value> fields;
        fields.emplace_back(take_object_value(
            make_owned_object<ExprObj>(item.condition), ValueKind::Expr));
        fields.emplace_back(take_object_value(
            adopt_object(math_internal::solution_tables(item.solutions)),
            ValueKind::Obj));
        cases->append(take_object_value(
            make_owned_object<AdtObj>(
                "ParametricCase", "ParametricCase", std::move(fields)),
            ValueKind::Obj));
    }
    std::vector<Value> fields;
    fields.emplace_back(take_object_value(std::move(cases), ValueKind::Obj));
    return result_ok(new AdtObj(
        "ParametricPiecewise", "ParametricPiecewise", std::move(fields)),
        ValueKind::Obj);
} catch (...) {
    return c_abi_current_exception(operation);
}

}

extern "C" LM_API AdtObj* lmx_cas_system_by_names(
    ArrayObj* equations, ArrayObj* variables) noexcept try {
    ensure_lmmc_runtime();
    std::vector<LMCAS::ExprPtr> checked_equations;
    std::vector<std::string> checked_variables;
    std::string error;
    if (!array_expressions(equations, checked_equations, error) ||
        !array_strings(variables, checked_variables, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "solve.system: " + error);
    return result_ok(
        math_internal::solution_tables(LMCAS::SymbolicExpr::solve_system(
            checked_equations, checked_variables)),
        ValueKind::Obj);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_cas_system_by_symbols(
    ArrayObj* equations, ArrayObj* variables) noexcept try {
    ensure_lmmc_runtime();
    if (!variables) return result_error(MathErrorCode::InvalidArgument, __func__, "solve.system: null array");
    std::vector<std::string> names;
    std::string error;
    if (!math_internal::checked_symbol_names(
            variables, names, error,
            "CasError(InvalidArgument: variable array contains a non-expression value)")) {
        return result_error(
            MathErrorCode::InvalidArgument, __func__, std::move(error));
    }
    std::vector<LMCAS::ExprPtr> checked_equations;
    if (!array_expressions(equations, checked_equations, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "solve.system: " + error);
    return result_ok(
        math_internal::solution_tables(LMCAS::SymbolicExpr::solve_system(checked_equations, names)),
        ValueKind::Obj);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_cas_polynomial_system_by_names(
    ArrayObj* equations, ArrayObj* variables) noexcept try {
    ensure_lmmc_runtime();
    std::vector<LMCAS::ExprPtr> expressions;
    std::vector<std::string> names;
    std::string error;
    if (!array_expressions(equations, expressions, error) ||
        !array_strings(variables, names, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "solve.polynomial_system: " + error);
    return polynomial_system_result(expressions, names, __func__);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_cas_polynomial_system_by_symbols(
    ArrayObj* equations, ArrayObj* variables) noexcept {
    const char* exception_operation = __func__;
    try {
        ensure_lmmc_runtime();
        std::vector<std::string> names;
        std::string error;
        if (!math_internal::checked_symbol_names(variables, names, error))
            return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
        exception_operation = "lmx_cas_polynomial_system_by_names";
        std::vector<LMCAS::ExprPtr> expressions;
        if (!array_expressions(equations, expressions, error))
            return result_error(MathErrorCode::InvalidArgument, exception_operation, "solve.polynomial_system: " + error);
        return polynomial_system_result(expressions, names, exception_operation);
    } catch (...) {
        return c_abi_current_exception(exception_operation);
    }
}
extern "C" LM_API AdtObj* lmx_cas_parametric_system_by_names(
    ArrayObj* equations, ArrayObj* unknowns, ArrayObj* parameters) noexcept try {
    ensure_lmmc_runtime();
    std::vector<LMCAS::ExprPtr> values;
    std::vector<std::string> unknown_names, parameter_names;
    std::string error;
    if (!array_expressions(equations, values, error) ||
        !array_strings(unknowns, unknown_names, error) ||
        !array_strings(parameters, parameter_names, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "solve.parametric_system: " + error);
    return result_ok(math_internal::solution_tables(
        LMCAS::ParametricSolver::solve_system(
            values, unknown_names, parameter_names)), ValueKind::Obj);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_cas_parametric_system_by_symbols(
    ArrayObj* equations, ArrayObj* unknowns, ArrayObj* parameters) noexcept try {
    ensure_lmmc_runtime();
    std::vector<LMCAS::ExprPtr> values;
    std::vector<std::string> unknown_names, parameter_names;
    std::string error;
    if (!array_expressions(equations, values, error) ||
        !math_internal::checked_symbol_names(unknowns, unknown_names, error) ||
        !math_internal::checked_symbol_names(parameters, parameter_names, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "solve.parametric_system: " + error);
    return result_ok(math_internal::solution_tables(
        LMCAS::ParametricSolver::solve_system(
            values, unknown_names, parameter_names)), ValueKind::Obj);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_cas_parametric_piecewise_by_names(
    ArrayObj* equations, ArrayObj* unknowns, ArrayObj* parameters) noexcept try {
    ensure_lmmc_runtime();
    std::vector<LMCAS::ExprPtr> values;
    std::vector<std::string> unknown_names, parameter_names;
    std::string error;
    if (!array_expressions(equations, values, error) ||
        !array_strings(unknowns, unknown_names, error) ||
        !array_strings(parameters, parameter_names, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "solve.parametric_piecewise: " + error);
    return parametric_piecewise_result(
        values, unknown_names, parameter_names, __func__);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_cas_parametric_piecewise_by_symbols(
    ArrayObj* equations, ArrayObj* unknowns, ArrayObj* parameters) noexcept {
    const char* exception_operation = __func__;
    try {
        ensure_lmmc_runtime();
        std::vector<std::string> unknown_names, parameter_names;
        std::string error;
        if (!math_internal::checked_symbol_names(unknowns, unknown_names, error) ||
            !math_internal::checked_symbol_names(parameters, parameter_names, error))
            return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
        exception_operation = "lmx_cas_parametric_piecewise_by_names";
        std::vector<LMCAS::ExprPtr> values;
        if (!array_expressions(equations, values, error))
            return result_error(MathErrorCode::InvalidArgument, exception_operation, "solve.parametric_piecewise: " + error);
        return parametric_piecewise_result(
            values, unknown_names, parameter_names, exception_operation);
    } catch (...) {
        return c_abi_current_exception(exception_operation);
    }
}
