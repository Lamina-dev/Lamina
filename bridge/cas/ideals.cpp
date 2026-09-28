#include "bridge/math_internal.hpp"
#include "symbolic.hpp"
#include "solver.hpp"
#include <limits>

using namespace lmx::bridge;

namespace {
std::vector<LMCAS::SymbolicExpr> symbolic_values(
    const std::vector<LMCAS::ExprPtr>& expressions) {
    std::vector<LMCAS::SymbolicExpr> values;
    values.reserve(expressions.size());
    for (const auto& expression : expressions) values.push_back(*expression);
    return values;
}

AdtObj* unordered_symbolic_result(
    std::vector<LMCAS::SymbolicExpr> expressions) {
    std::vector<LMCAS::ExprPtr> values;
    values.reserve(expressions.size());
    for (auto& expression : expressions) {
        values.push_back(
            std::make_shared<LMCAS::SymbolicExpr>(std::move(expression)));
    }
    return math_internal::unordered_expr_result(std::move(values));
}

AdtObj* groebner_basis_result(
    const std::vector<LMCAS::ExprPtr>& expressions,
    const std::vector<std::string>& names,
    const char* operation) noexcept try {
    const auto values = symbolic_values(expressions);
    return unordered_symbolic_result(
        LMCAS::Solver::groebner_basis(values, names));
} catch (...) {
    return c_abi_current_exception(operation);
}

AdtObj* reduced_groebner_basis_result(
    const std::vector<LMCAS::ExprPtr>& expressions,
    const std::vector<std::string>& names,
    const char* operation) noexcept try {
    const auto values = symbolic_values(expressions);
    return unordered_symbolic_result(
        LMCAS::Solver::reduced_groebner_basis(values, names));
} catch (...) {
    return c_abi_current_exception(operation);
}

AdtObj* ideal_membership_result(
    const LMCAS::ExprPtr& polynomial,
    const std::vector<LMCAS::ExprPtr>& basis,
    const std::vector<std::string>& names,
    const char* operation) noexcept try {
    const auto values = symbolic_values(basis);
    return result_ok(
        LMCAS::Solver::ideal_membership(*polynomial, values, names));
} catch (...) {
    return c_abi_current_exception(operation);
}

AdtObj* elimination_ideal_result(
    const std::vector<LMCAS::ExprPtr>& expressions,
    const std::vector<std::string>& names,
    const int count,
    const char* operation) noexcept try {
    const auto values = symbolic_values(expressions);
    return unordered_symbolic_result(
        LMCAS::Solver::elimination_ideal(values, names, count));
} catch (...) {
    return c_abi_current_exception(operation);
}

}

extern "C" LM_API AdtObj* lmx_cas_groebner_basis_by_names(
    ArrayObj* polynomials, ArrayObj* variables) noexcept try {
    ensure_lmmc_runtime();
    std::vector<LMCAS::ExprPtr> expressions;
    std::vector<std::string> names;
    std::string error;
    if (!array_expressions(polynomials, expressions, error) ||
        !array_strings(variables, names, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "solve.groebner_basis: " + error);
    return groebner_basis_result(expressions, names, __func__);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_cas_groebner_basis_by_symbols(
    ArrayObj* polynomials, ArrayObj* variables) noexcept try {
    ensure_lmmc_runtime();
    std::vector<std::string> names;
    std::string error;
    if (!math_internal::checked_symbol_names(variables, names, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    std::vector<LMCAS::ExprPtr> expressions;
    if (!array_expressions(polynomials, expressions, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "solve.groebner_basis: " + error);
    return groebner_basis_result(expressions, names, __func__);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_cas_reduced_groebner_basis_by_names(
    ArrayObj* polynomials, ArrayObj* variables) noexcept try {
    ensure_lmmc_runtime();
    std::vector<LMCAS::ExprPtr> expressions;
    std::vector<std::string> names;
    std::string error;
    if (!array_expressions(polynomials, expressions, error) ||
        !array_strings(variables, names, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "solve.reduced_groebner_basis: " + error);
    return reduced_groebner_basis_result(expressions, names, __func__);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_cas_reduced_groebner_basis_by_symbols(
    ArrayObj* polynomials, ArrayObj* variables) noexcept try {
    ensure_lmmc_runtime();
    std::vector<std::string> names;
    std::string error;
    if (!math_internal::checked_symbol_names(variables, names, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    std::vector<LMCAS::ExprPtr> expressions;
    if (!array_expressions(polynomials, expressions, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "solve.reduced_groebner_basis: " + error);
    return reduced_groebner_basis_result(expressions, names, __func__);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_cas_ideal_membership_by_names(
    ExprObj* polynomial, ArrayObj* basis, ArrayObj* variables) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    const auto* checked = checked_expr(polynomial, error);
    std::vector<LMCAS::ExprPtr> basis_values;
    std::vector<std::string> names;
    if (!checked || !array_expressions(basis, basis_values, error) ||
        !array_strings(variables, names, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "solve.ideal_membership: " + error);
    return ideal_membership_result(*checked, basis_values, names, __func__);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_cas_ideal_membership_by_symbols(
    ExprObj* polynomial, ArrayObj* basis, ArrayObj* variables) noexcept try {
    ensure_lmmc_runtime();
    std::vector<std::string> names;
    std::string error;
    if (!math_internal::checked_symbol_names(variables, names, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    const auto* checked = checked_expr(polynomial, error);
    std::vector<LMCAS::ExprPtr> basis_values;
    if (!checked || !array_expressions(basis, basis_values, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "solve.ideal_membership: " + error);
    return ideal_membership_result(*checked, basis_values, names, __func__);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_cas_elimination_ideal_by_names(
    ArrayObj* basis, ArrayObj* variables, LmInt count) noexcept try {
    ensure_lmmc_runtime();
    if (count < 0 || count > std::numeric_limits<int>::max())
        return result_error(MathErrorCode::InvalidArgument, __func__, "solve.elimination_ideal: invalid elimination count");
    std::vector<LMCAS::ExprPtr> expressions;
    std::vector<std::string> names;
    std::string error;
    if (!array_expressions(basis, expressions, error) ||
        !array_strings(variables, names, error) ||
        static_cast<std::size_t>(count) > names.size())
        return result_error(MathErrorCode::InvalidArgument, __func__, "solve.elimination_ideal: " + error);
    return elimination_ideal_result(
        expressions, names, static_cast<int>(count), __func__);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_cas_elimination_ideal_by_symbols(
    ArrayObj* basis, ArrayObj* variables, LmInt count) noexcept try {
    ensure_lmmc_runtime();
    std::vector<std::string> names;
    std::string error;
    if (!math_internal::checked_symbol_names(variables, names, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    if (count < 0 || count > std::numeric_limits<int>::max())
        return result_error(MathErrorCode::InvalidArgument, __func__, "solve.elimination_ideal: invalid elimination count");
    std::vector<LMCAS::ExprPtr> expressions;
    if (!array_expressions(basis, expressions, error) ||
        static_cast<std::size_t>(count) > names.size())
        return result_error(MathErrorCode::InvalidArgument, __func__, "solve.elimination_ideal: " + error);
    return elimination_ideal_result(
        expressions, names, static_cast<int>(count), __func__);
} catch (...) {
    return c_abi_current_exception(__func__);
}
