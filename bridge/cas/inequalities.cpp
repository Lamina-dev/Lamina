#include "bridge/result.hpp"
#include "bridge/conversions.hpp"
#include "symbolic.hpp"
#include "inequality_solver.hpp"

using namespace lmx::bridge;

namespace {
std::optional<LMCAS::InequalityType> checked_inequality_type(
    const char* relation) {
    const std::string name = relation ? relation : "";
    if (name == "<") return LMCAS::InequalityType::LessThan;
    if (name == "<=") return LMCAS::InequalityType::LessEqual;
    if (name == ">") return LMCAS::InequalityType::GreaterThan;
    if (name == ">=") return LMCAS::InequalityType::GreaterEqual;
    return std::nullopt;
}

AdtObj* interval_union_value(
    const LMCAS::IntervalUnion& intervals,
    const std::string& variable,
    const char* operation) {
    auto expression = intervals.to_expr(variable);
    if (!expression) {
        return result_error(
            MathErrorCode::InternalError, operation,
            "CasError(InternalInvariant: interval conversion returned null)");
    }
    return result_ok(new ExprObj(std::move(expression)), ValueKind::Expr);
}

AdtObj* interval_union_result(
    const LMCAS::Result<LMCAS::IntervalUnion>& result,
    const std::string& variable,
    const char* operation) {
    if (!result) return result_error(result.error());
    return interval_union_value(result.value(), variable, operation);
}
}

extern "C" LM_API AdtObj* lmx_cas_inequality(
    ExprObj* expression, const char* relation, const char* variable) noexcept try {
    ensure_lmmc_runtime();
    if (!relation || !variable)
        return result_error(MathErrorCode::InvalidArgument, __func__, "solve.inequality: invalid argument");
    std::string error;
    const auto* checked = checked_expr(expression, error);
    if (!checked) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    const auto type = checked_inequality_type(relation);
    if (!type) return result_error(MathErrorCode::InvalidArgument, __func__, "solve.inequality: unknown relation");
    const auto result = LMCAS::InequalitySolver::solve_inequality_checked(
        *checked, *type, variable);
    if (!result) return result_error(result.error());
    return interval_union_value(result.value(), variable, __func__);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_cas_inequalities_by_name(
    ArrayObj* expressions, ArrayObj* relations, const char* variable) noexcept try {
    ensure_lmmc_runtime();
    if (!variable || variable[0] == '\0')
        return result_error(MathErrorCode::InvalidArgument, __func__, "solve.inequalities: empty variable");
    std::vector<LMCAS::ExprPtr> values;
    std::vector<std::string> relation_names;
    std::string error;
    if (!array_expressions(expressions, values, error) ||
        !array_strings(relations, relation_names, error) ||
        values.size() != relation_names.size())
        return result_error(MathErrorCode::InvalidArgument, __func__, "solve.inequalities: invalid arrays");
    std::vector<std::pair<LMCAS::ExprPtr, LMCAS::InequalityType>> inputs;
    for (std::size_t index = 0; index < values.size(); ++index) {
        const auto type = checked_inequality_type(relation_names[index].c_str());
        if (!type) return result_error(MathErrorCode::InvalidArgument, __func__, "solve.inequalities: unknown relation");
        inputs.emplace_back(values[index], *type);
    }
    return interval_union_result(
        LMCAS::InequalitySolver::solve_inequalities_checked(inputs, variable),
        variable, __func__);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_cas_inequalities_by_symbol(
    ArrayObj* expressions, ArrayObj* relations, ExprObj* variable) noexcept try {
    ensure_lmmc_runtime();
    std::string name, error;
    if (!checked_symbol_name(variable, name, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return lmx_cas_inequalities_by_name(expressions, relations, name.c_str());
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_cas_rational_inequality_by_name(
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
    auto result = LMCAS::InequalitySolver::solve_rational_inequality(
        *n, *d, *type, variable);
    return interval_union_value(result, variable, __func__);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_cas_rational_inequality_by_symbol(
    ExprObj* numerator, ExprObj* denominator,
    const char* relation, ExprObj* variable) noexcept try {
    ensure_lmmc_runtime();
    std::string name, error;
    if (!checked_symbol_name(variable, name, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return lmx_cas_rational_inequality_by_name(
        numerator, denominator, relation, name.c_str());
} catch (...) {
    return c_abi_current_exception(__func__);
}
