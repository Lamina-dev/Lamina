#include "bridge/result.hpp"
#include "bridge/conversions.hpp"
#include "bridge/runtime_views.hpp"
#include "bridge/unit_bridge.hpp"
#include <cstdarg>
#include "bridge/math_internal.hpp"
#include "runtime/object/assumptions.hpp"
#include "assumption_context.hpp"
#include "query_interface.hpp"
#include "solver.hpp"
#include "inequality_solver.hpp"
#include "parametric_solver.hpp"
#include "transcendental_factor.hpp"
#include "symbolic.hpp"
#include "symbolic_matrix.hpp"
#include "calculus_utils.hpp"
#include "multiple_integral.hpp"
#include "symbolic_complex.hpp"
#include "matrix_decomposition.hpp"
#include <array>
#include "symbolic_ode.hpp"
#include "symbolic_ode_engine.hpp"
#include "symbolic_vector_geometry.hpp"
#include "differential_geometry.hpp"

using namespace lmx::bridge;

namespace {
AdtObj* unordered_expr_result(std::vector<lamina::lsr::ExprPtr> values) {
    return expression_set_literal_result(lamina::lsr::ExprSet::make(std::move(values)));
}

ArrayObj* value_solution_tables(
    const std::vector<std::map<std::string, SymbolicExpr>>& solutions) {
    auto result = make_owned_object<ArrayObj>();
    for (const auto& solution : solutions) {
        std::vector<TableObj::Entry> entries;
        for (const auto& [name, expression] : solution) {
            auto value = take_object_value(
                make_owned_object<ExprObj>(
                    std::make_shared<SymbolicExpr>(expression)),
                ValueKind::Expr);
            entries.emplace_back(name, std::move(value));
        }
        result->append(take_object_value(
            make_owned_object<TableObj>(std::move(entries)), ValueKind::Table));
    }
    return result.release();
}

std::optional<lamina::InequalityType> checked_inequality_type(
    const char* relation) {
    const std::string name = relation ? relation : "";
    if (name == "<") return lamina::InequalityType::LessThan;
    if (name == "<=") return lamina::InequalityType::LessEqual;
    if (name == ">") return lamina::InequalityType::GreaterThan;
    if (name == ">=") return lamina::InequalityType::GreaterEqual;
    return std::nullopt;
}

AdtObj* interval_union_result(
    const lamina::Result<lamina::IntervalUnion>& result,
    const std::string& variable) {
    if (!result) return result_error(result.error());
    auto expression = result.value().to_expr(variable);
    if (!expression) {
        const auto empty = lamina::lsr::finite_set({});
        if (!empty) return result_error(empty.error());
        expression = empty.value();
    }
    return result_ok(new ExprObj(std::move(expression)), ValueKind::Expr);
}
} // namespace

/** @brief Applies the public final factorization workflow to a multivariate expression. @param value Borrowed polynomial expression. @return Owning Result Expr or error. @ownership Input borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_algebra_factor_multivariate(ExprObj* value) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    const auto* expression = checked_expr(value, error);
    if (!expression)
        return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return math_internal::checked_expr_result((*expression)->factor_checked());
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Computes a multivariate polynomial GCD through the public symbolic GCD API. @param lhs Borrowed polynomial. @param rhs Borrowed polynomial. @return Owning Result Expr or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_algebra_multivariate_greatest_common_divisor(
    ExprObj* lhs, ExprObj* rhs) noexcept try {
    ensure_lmmc_runtime();
    return lmx_computer_algebra_algebra_polynomial_greatest_common_divisor(lhs, rhs);
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Factors recognized transcendental structure. @param value Borrowed expression. @param variable Borrowed variable name. @return Owning Result unordered set of factors or error. @ownership Input borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_algebra_factor_transcendental_by_name(
    ExprObj* value, const char* variable) noexcept try {
    ensure_lmmc_runtime();
    if (!variable || variable[0] == '\0')
        return result_error(MathErrorCode::InvalidArgument, __func__, "CasError(InvalidArgument in algebra.factor_transcendental: empty variable)");
    std::string error;
    const auto* expression = checked_expr(value, error);
    if (!expression) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return unordered_expr_result(
        lamina::factor_transcendental(*expression, variable));
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Symbol-variable transcendental factorization. @param value Borrowed expression. @param variable Borrowed single symbol. @return Owning Result unordered set of factors or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_algebra_factor_transcendental_by_symbol(
    ExprObj* value, ExprObj* variable) noexcept try {
    ensure_lmmc_runtime();
    std::string name, error;
    if (!checked_symbol_name(variable, name, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return lmx_computer_algebra_algebra_factor_transcendental_by_name(value, name.c_str());
} catch (...) {
    return c_abi_current_exception(__func__);
}

/** @brief Solves a polynomial system with explicit text variables. @param equations Borrowed expression array. @param variables Borrowed text array. @return Owning Result array of solution tables or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_equation_solving_polynomial_system_full_by_names(
    ArrayObj* equations, ArrayObj* variables) noexcept try {
    ensure_lmmc_runtime();
    std::vector<lamina::lsr::ExprPtr> expressions;
    std::vector<std::string> names;
    std::string error;
    if (!array_expressions(equations, expressions, error) ||
        !array_strings(variables, names, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "solve.polynomial_system: " + error);
    std::vector<SymbolicExpr> values;
    values.reserve(expressions.size());
    for (const auto& expression : expressions) values.push_back(*expression);
    const auto solutions =
        lamina::Solver::solve_polynomial_system_checked(values, names);
    if (!solutions) return result_error(solutions.error());
    return result_ok(value_solution_tables(solutions.value()), ValueKind::Obj);
} catch (...) {
    return c_abi_current_exception(__func__);
}

/** @brief Solves a parametric system. @param equations Borrowed expression array. @param unknowns Borrowed unknown-name array. @param parameters Borrowed parameter-name array. @return Owning Result array of solution tables or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_equation_solving_parametric_system_by_names(
    ArrayObj* equations, ArrayObj* unknowns, ArrayObj* parameters) noexcept try {
    ensure_lmmc_runtime();
    std::vector<lamina::lsr::ExprPtr> values;
    std::vector<std::string> unknown_names, parameter_names;
    std::string error;
    if (!array_expressions(equations, values, error) ||
        !array_strings(unknowns, unknown_names, error) ||
        !array_strings(parameters, parameter_names, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "solve.parametric_system: " + error);
    return result_ok(math_internal::solution_tables(
        lamina::ParametricSolver::solve_system(
            values, unknown_names, parameter_names)), ValueKind::Obj);
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Solves a parametric system with symbol arrays. @param equations Borrowed expression array. @param unknowns Borrowed unknown-symbol array. @param parameters Borrowed parameter-symbol array. @return Owning Result array of solution tables or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_equation_solving_parametric_system_by_symbols(
    ArrayObj* equations, ArrayObj* unknowns, ArrayObj* parameters) noexcept try {
    ensure_lmmc_runtime();
    std::vector<lamina::lsr::ExprPtr> values;
    std::vector<std::string> unknown_names, parameter_names;
    std::string error;
    if (!array_expressions(equations, values, error) ||
        !math_internal::checked_symbol_names(unknowns, unknown_names, error) ||
        !math_internal::checked_symbol_names(parameters, parameter_names, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "solve.parametric_system: " + error);
    return result_ok(math_internal::solution_tables(
        lamina::ParametricSolver::solve_system(
            values, unknown_names, parameter_names)), ValueKind::Obj);
} catch (...) {
    return c_abi_current_exception(__func__);
}

/** @brief Computes an unordered Gröbner basis. @param polynomials Borrowed expression array. @param variables Borrowed text variable array. @return Owning Result set of expressions or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_equation_solving_groebner_basis_by_names(
    ArrayObj* polynomials, ArrayObj* variables) noexcept try {
    ensure_lmmc_runtime();
    std::vector<lamina::lsr::ExprPtr> expressions;
    std::vector<std::string> names;
    std::string error;
    if (!array_expressions(polynomials, expressions, error) ||
        !array_strings(variables, names, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "solve.groebner_basis: " + error);
    std::vector<SymbolicExpr> values;
    for (const auto& expression : expressions) values.push_back(*expression);
    std::vector<lamina::lsr::ExprPtr> result;
    for (auto& expression : lamina::Solver::groebner_basis(values, names))
        result.push_back(std::make_shared<SymbolicExpr>(std::move(expression)));
    return unordered_expr_result(std::move(result));
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Computes an unordered reduced Gröbner basis. @param polynomials Borrowed expression array. @param variables Borrowed text variable array. @return Owning Result set of expressions or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_equation_solving_reduced_groebner_basis_by_names(
    ArrayObj* polynomials, ArrayObj* variables) noexcept try {
    ensure_lmmc_runtime();
    std::vector<lamina::lsr::ExprPtr> expressions;
    std::vector<std::string> names;
    std::string error;
    if (!array_expressions(polynomials, expressions, error) ||
        !array_strings(variables, names, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "solve.reduced_groebner_basis: " + error);
    std::vector<SymbolicExpr> values;
    for (const auto& expression : expressions) values.push_back(*expression);
    std::vector<lamina::lsr::ExprPtr> result;
    for (auto& expression : lamina::Solver::reduced_groebner_basis(values, names))
        result.push_back(std::make_shared<SymbolicExpr>(std::move(expression)));
    return unordered_expr_result(std::move(result));
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Tests polynomial ideal membership. @param polynomial Borrowed polynomial. @param basis Borrowed basis array. @param variables Borrowed text variable array. @return Owning Result bool or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_equation_solving_ideal_membership_by_names(
    ExprObj* polynomial, ArrayObj* basis, ArrayObj* variables) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    const auto* checked = checked_expr(polynomial, error);
    std::vector<lamina::lsr::ExprPtr> basis_values;
    std::vector<std::string> names;
    if (!checked || !array_expressions(basis, basis_values, error) ||
        !array_strings(variables, names, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "solve.ideal_membership: " + error);
    std::vector<SymbolicExpr> values;
    for (const auto& expression : basis_values) values.push_back(*expression);
    return result_ok(lamina::Solver::ideal_membership(
        **checked, values, names));
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Computes an unordered elimination ideal. @param basis Borrowed basis array. @param variables Borrowed text variable array. @param count Number of leading variables to eliminate. @return Owning Result set of expressions or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_equation_solving_elimination_ideal_by_names(
    ArrayObj* basis, ArrayObj* variables, LmInt count) noexcept try {
    ensure_lmmc_runtime();
    if (count < 0 || count > std::numeric_limits<int>::max())
        return result_error(MathErrorCode::InvalidArgument, __func__, "solve.elimination_ideal: invalid elimination count");
    std::vector<lamina::lsr::ExprPtr> expressions;
    std::vector<std::string> names;
    std::string error;
    if (!array_expressions(basis, expressions, error) ||
        !array_strings(variables, names, error) ||
        static_cast<std::size_t>(count) > names.size())
        return result_error(MathErrorCode::InvalidArgument, __func__, "solve.elimination_ideal: " + error);
    std::vector<SymbolicExpr> values;
    for (const auto& expression : expressions) values.push_back(*expression);
    std::vector<lamina::lsr::ExprPtr> result;
    for (auto& expression : lamina::Solver::elimination_ideal(
             values, names, static_cast<int>(count)))
        result.push_back(std::make_shared<SymbolicExpr>(std::move(expression)));
    return unordered_expr_result(std::move(result));
} catch (...) {
    return c_abi_current_exception(__func__);
}

/** @brief Solves a conjunction of inequalities. @param expressions Borrowed left-side expressions. @param relations Borrowed relation-text array. @param variable Borrowed variable name. @return Owning Result Expr or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_equation_solving_inequalities_by_name(
    ArrayObj* expressions, ArrayObj* relations, const char* variable) noexcept try {
    ensure_lmmc_runtime();
    if (!variable || variable[0] == '\0')
        return result_error(MathErrorCode::InvalidArgument, __func__, "solve.inequalities: empty variable");
    std::vector<lamina::lsr::ExprPtr> values;
    std::vector<std::string> relation_names;
    std::string error;
    if (!array_expressions(expressions, values, error) ||
        !array_strings(relations, relation_names, error) ||
        values.size() != relation_names.size())
        return result_error(MathErrorCode::InvalidArgument, __func__, "solve.inequalities: invalid arrays");
    std::vector<std::pair<lamina::lsr::ExprPtr, lamina::InequalityType>> inputs;
    for (std::size_t index = 0; index < values.size(); ++index) {
        const auto type = checked_inequality_type(relation_names[index].c_str());
        if (!type) return result_error(MathErrorCode::InvalidArgument, __func__, "solve.inequalities: unknown relation");
        inputs.emplace_back(values[index], *type);
    }
    return interval_union_result(
        lamina::InequalitySolver::solve_inequalities_checked(inputs, variable),
        variable);
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Solves a rational inequality. @param numerator Borrowed numerator. @param denominator Borrowed denominator. @param relation Borrowed relation text. @param variable Borrowed variable name. @return Owning Result Expr or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_equation_solving_rational_inequality_by_name(
    ExprObj* numerator, ExprObj* denominator,
    const char* relation, const char* variable) noexcept try {
    ensure_lmmc_runtime();
    const auto type = checked_inequality_type(relation);
    if (!type || !variable || variable[0] == '\0')
        return result_error(MathErrorCode::InvalidArgument, __func__, "solve.rational_inequality: invalid argument");
    std::string error;
    const auto* n = checked_expr(numerator, error);
    const auto* d = checked_expr(denominator, error);
    if (!n || !d) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    auto result = lamina::InequalitySolver::solve_rational_inequality(
        *n, *d, *type, variable);
    auto expression = result.to_expr(variable);
    if (!expression) {
        const auto empty = lamina::lsr::finite_set({});
        if (!empty) return result_error(empty.error());
        expression = empty.value();
    }
    return result_ok(new ExprObj(std::move(expression)), ValueKind::Expr);
} catch (...) {
    return c_abi_current_exception(__func__);
}

namespace {
ArrayObj* symbol_text_array(ArrayObj* symbols, std::string& error) {
    std::vector<std::string> names;
    if (!math_internal::checked_symbol_names(symbols, names, error)) return nullptr;
    auto result = make_owned_object<ArrayObj>();
    for (auto& name : names) {
        result->append(take_object_value(
            make_owned_object<StringObj>(std::move(name)), ValueKind::Obj));
    }
    return result.release();
}
} // namespace

/** @brief Solves a polynomial system with symbol variables. @param equations Borrowed expression array. @param variables Borrowed symbol array. @return Owning Result array of tables or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_equation_solving_polynomial_system_full_by_symbols(
    ArrayObj* equations, ArrayObj* variables) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    auto* names = symbol_text_array(variables, error);
    if (!names) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    auto* result = lmx_computer_algebra_equation_solving_polynomial_system_full_by_names(equations, names);
    names->release();
    return result;
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Computes a Gröbner basis with symbol variables. @param polynomials Borrowed expression array. @param variables Borrowed symbol array. @return Owning Result set or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_equation_solving_groebner_basis_by_symbols(
    ArrayObj* polynomials, ArrayObj* variables) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    auto* names = symbol_text_array(variables, error);
    if (!names) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    auto* result = lmx_computer_algebra_equation_solving_groebner_basis_by_names(polynomials, names);
    names->release();
    return result;
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Computes a reduced Gröbner basis with symbol variables. @param polynomials Borrowed expression array. @param variables Borrowed symbol array. @return Owning Result set or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_equation_solving_reduced_groebner_basis_by_symbols(
    ArrayObj* polynomials, ArrayObj* variables) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    auto* names = symbol_text_array(variables, error);
    if (!names) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    auto* result = lmx_computer_algebra_equation_solving_reduced_groebner_basis_by_names(polynomials, names);
    names->release();
    return result;
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Tests ideal membership with symbol variables. @param polynomial Borrowed polynomial. @param basis Borrowed basis. @param variables Borrowed symbol array. @return Owning Result bool or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_equation_solving_ideal_membership_by_symbols(
    ExprObj* polynomial, ArrayObj* basis, ArrayObj* variables) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    auto* names = symbol_text_array(variables, error);
    if (!names) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    auto* result = lmx_computer_algebra_equation_solving_ideal_membership_by_names(polynomial, basis, names);
    names->release();
    return result;
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Computes an elimination ideal with symbol variables. @param basis Borrowed basis. @param variables Borrowed symbol array. @param count Elimination count. @return Owning Result set or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_equation_solving_elimination_ideal_by_symbols(
    ArrayObj* basis, ArrayObj* variables, LmInt count) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    auto* names = symbol_text_array(variables, error);
    if (!names) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    auto* result = lmx_computer_algebra_equation_solving_elimination_ideal_by_names(basis, names, count);
    names->release();
    return result;
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Solves inequalities for a symbol variable. @param expressions Borrowed expressions. @param relations Borrowed relation array. @param variable Borrowed symbol. @return Owning Result Expr or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_equation_solving_inequalities_by_symbol(
    ArrayObj* expressions, ArrayObj* relations, ExprObj* variable) noexcept try {
    ensure_lmmc_runtime();
    std::string name, error;
    if (!checked_symbol_name(variable, name, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return lmx_computer_algebra_equation_solving_inequalities_by_name(expressions, relations, name.c_str());
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Solves a rational inequality for a symbol variable. @param numerator Borrowed numerator. @param denominator Borrowed denominator. @param relation Borrowed relation. @param variable Borrowed symbol. @return Owning Result Expr or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_equation_solving_rational_inequality_by_symbol(
    ExprObj* numerator, ExprObj* denominator,
    const char* relation, ExprObj* variable) noexcept try {
    ensure_lmmc_runtime();
    std::string name, error;
    if (!checked_symbol_name(variable, name, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return lmx_computer_algebra_equation_solving_rational_inequality_by_name(
        numerator, denominator, relation, name.c_str());
} catch (...) {
    return c_abi_current_exception(__func__);
}

/** @brief Solves a parametric system into explicit piecewise cases. @param equations Borrowed expressions. @param unknowns Borrowed unknown-name array. @param parameters Borrowed parameter-name array. @return Owning Result ParametricPiecewise or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_equation_solving_parametric_piecewise_by_names(
    ArrayObj* equations, ArrayObj* unknowns, ArrayObj* parameters) noexcept try {
    ensure_lmmc_runtime();
    std::vector<lamina::lsr::ExprPtr> values;
    std::vector<std::string> unknown_names, parameter_names;
    std::string error;
    if (!array_expressions(equations, values, error) ||
        !array_strings(unknowns, unknown_names, error) ||
        !array_strings(parameters, parameter_names, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "solve.parametric_piecewise: " + error);
    const auto piecewise = lamina::ParametricSolver::solve_system_piecewise(
        values, unknown_names, parameter_names);
    auto cases = make_owned_object<ArrayObj>();
    for (const auto& item : piecewise.cases) {
        if (!item.condition) {
            return result_error(MathErrorCode::InternalError, __func__,
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
    return c_abi_current_exception(__func__);
}
/** @brief Solves a parametric system into piecewise cases using symbols. @param equations Borrowed expressions. @param unknowns Borrowed unknown symbols. @param parameters Borrowed parameter symbols. @return Owning Result ParametricPiecewise or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_equation_solving_parametric_piecewise_by_symbols(
    ArrayObj* equations, ArrayObj* unknowns, ArrayObj* parameters) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    auto* unknown_names = symbol_text_array(unknowns, error);
    if (!unknown_names) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    auto* parameter_names = symbol_text_array(parameters, error);
    if (!parameter_names) {
        unknown_names->release();
        return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    }
    auto* result = lmx_computer_algebra_equation_solving_parametric_piecewise_by_names(
        equations, unknown_names, parameter_names);
    unknown_names->release();
    parameter_names->release();
    return result;
} catch (...) {
    return c_abi_current_exception(__func__);
}
