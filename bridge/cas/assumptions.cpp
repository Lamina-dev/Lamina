#include "bridge/result.hpp"
#include "bridge/conversions.hpp"
#include "runtime/object/assumptions.hpp"
#include "assumption_context.hpp"
#include "query_interface.hpp"
#include "solver.hpp"
#include "symbolic.hpp"

using namespace lmx::bridge;

namespace {
using lmx::runtime::AssumptionsObj;

std::optional<LMCAS::Domain> assumption_domain(const char* value) {
    const std::string name = value ? value : "";
    if (name == "complex") return LMCAS::Domain::Complex;
    if (name == "real") return LMCAS::Domain::Real;
    if (name == "algebraic") return LMCAS::Domain::Algebraic;
    if (name == "rational") return LMCAS::Domain::Rational;
    if (name == "integer") return LMCAS::Domain::Integer;
    if (name == "natural") return LMCAS::Domain::Natural;
    if (name == "positive_int") return LMCAS::Domain::PositiveInt;
    return std::nullopt;
}

std::optional<LMCAS::Sign> assumption_sign(const char* value) {
    const std::string name = value ? value : "";
    if (name == "positive") return LMCAS::Sign::Positive;
    if (name == "negative") return LMCAS::Sign::Negative;
    if (name == "nonnegative") return LMCAS::Sign::NonNegative;
    if (name == "nonpositive") return LMCAS::Sign::NonPositive;
    if (name == "zero") return LMCAS::Sign::Zero;
    if (name == "nonzero") return LMCAS::Sign::NonZero;
    return std::nullopt;
}

AdtObj* assumptions_result(AssumptionsObj* value) {
    return result_ok(value, ValueKind::Assumptions);
}

AdtObj* truth_result(const LMCAS::Tribool value) {
    const char* constructor = value == LMCAS::Tribool::True
        ? "Proven" : value == LMCAS::Tribool::False
            ? "Disproven" : "Undetermined";
    return result_ok(
        new AdtObj("Truth", constructor, {}), ValueKind::Obj);
}

template <typename Result>
AdtObj* checked_truth(const Result& result) {
    if (!result) return result_error(result.error());
    return truth_result(result.value());
}

bool assumption_expr(
    ExprObj* expression, LMCAS::ExprPtr& output, std::string& error) {
    const auto* checked = checked_expr(expression, error);
    if (!checked) return false;
    output = *checked;
    return true;
}

}

extern "C" LM_API AssumptionsObj* lmx_cas_assumptions_empty() noexcept try {
    ensure_lmmc_runtime();
    return new AssumptionsObj();
} catch (...) {
    return nullptr;
}
extern "C" LM_API AdtObj* lmx_cas_assumptions_push(AssumptionsObj* value) noexcept try {
    ensure_lmmc_runtime();
    if (!value) return result_error(MathErrorCode::InvalidArgument, __func__, "assumptions.push: null context");
    auto result = adopt_object(value->copy());
    result->context().push();
    return assumptions_result(result.release());
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_cas_assumptions_pop(AssumptionsObj* value) noexcept try {
    ensure_lmmc_runtime();
    if (!value) return result_error(MathErrorCode::InvalidArgument, __func__, "assumptions.pop: null context");
    auto result = adopt_object(value->copy());
    auto popped = result->context().pop();
    if (!popped) return result_error(popped.error());
    return assumptions_result(result.release());
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_cas_assumptions_with_domain(
    AssumptionsObj* value, const char* symbol, const char* domain) noexcept try {
    ensure_lmmc_runtime();
    if (!value || !symbol)
        return result_error(MathErrorCode::InvalidArgument, __func__, "assumptions.with_domain: invalid argument");
    const auto checked_domain = assumption_domain(domain);
    if (!checked_domain)
        return result_error(MathErrorCode::InvalidArgument, __func__, "assumptions.with_domain: unknown domain");
    auto result = adopt_object(value->copy());
    const auto status =
        result->context().assume_domain_checked(symbol, *checked_domain);
    if (!status) return result_error(status.error());
    return assumptions_result(result.release());
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_cas_assumptions_with_sign(
    AssumptionsObj* value, const char* symbol, const char* sign) noexcept try {
    ensure_lmmc_runtime();
    if (!value || !symbol)
        return result_error(MathErrorCode::InvalidArgument, __func__, "assumptions.with_sign: invalid argument");
    const auto checked_sign = assumption_sign(sign);
    if (!checked_sign)
        return result_error(MathErrorCode::InvalidArgument, __func__, "assumptions.with_sign: unknown sign");
    auto result = adopt_object(value->copy());
    const auto status =
        result->context().assume_sign_checked(symbol, *checked_sign);
    if (!status) return result_error(status.error());
    return assumptions_result(result.release());
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_cas_assumptions_with_relation(
    AssumptionsObj* value, ExprObj* relation) noexcept try {
    ensure_lmmc_runtime();
    if (!value) return result_error(MathErrorCode::InvalidArgument, __func__, "assumptions.with_relation: null context");
    std::string error;
    LMCAS::ExprPtr expression;
    if (!assumption_expr(relation, expression, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    auto result = adopt_object(value->copy());
    const auto status = result->context().assume_checked(*expression);
    if (!status) return result_error(status.error());
    return assumptions_result(result.release());
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_cas_assumptions_with_conditional(
    AssumptionsObj* value, ExprObj* condition, ExprObj* conclusion) noexcept try {
    ensure_lmmc_runtime();
    if (!value) return result_error(MathErrorCode::InvalidArgument, __func__, "assumptions.with_conditional: null context");
    std::string error;
    LMCAS::ExprPtr checked_condition;
    LMCAS::ExprPtr checked_conclusion;
    if (!assumption_expr(condition, checked_condition, error) ||
        !assumption_expr(conclusion, checked_conclusion, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    auto result = adopt_object(value->copy());
    const auto status = result->context().assume_conditional_checked(
        *checked_condition, *checked_conclusion);
    if (!status) return result_error(status.error());
    return assumptions_result(result.release());
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_cas_assumptions_query(
    AssumptionsObj* value, ExprObj* expression, const char* property) noexcept try {
    ensure_lmmc_runtime();
    if (!value || !property)
        return result_error(MathErrorCode::InvalidArgument, __func__, "assumptions.query: invalid argument");
    std::string error;
    LMCAS::ExprPtr checked;
    if (!assumption_expr(expression, checked, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    const LMCAS::QueryInterface query(value->context());
    const std::string name(property);
    if (name == "positive") return checked_truth(query.query_positive_checked(*checked));
    if (name == "negative") return checked_truth(query.query_negative_checked(*checked));
    if (name == "nonnegative") return checked_truth(query.query_nonnegative_checked(*checked));
    if (name == "real") return checked_truth(query.query_real_checked(*checked));
    if (name == "integer") return checked_truth(query.query_integer_checked(*checked));
    if (name == "nonzero") return checked_truth(query.query_nonzero_checked(*checked));
    if (name == "algebraic") return checked_truth(query.query_algebraic_checked(*checked));
    if (name == "transcendental") return checked_truth(query.query_transcendental_checked(*checked));
    if (name == "finite") return checked_truth(query.query_finite_checked(*checked));
    if (name == "divergent") return checked_truth(query.query_divergent_checked(*checked));
    if (name == "positive_definite") return checked_truth(query.query_positive_definite_checked(*checked));
    if (name == "positive_semidefinite") return checked_truth(query.query_positive_semidefinite_checked(*checked));
    return result_error(MathErrorCode::InvalidArgument, __func__, "assumptions.query: unknown property");
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_cas_assumptions_query_periodic(
    AssumptionsObj* value, ExprObj* expression, const char* variable) noexcept try {
    ensure_lmmc_runtime();
    if (!value || !variable)
        return result_error(MathErrorCode::InvalidArgument, __func__, "assumptions.query_periodic: invalid argument");
    std::string error;
    LMCAS::ExprPtr checked;
    if (!assumption_expr(expression, checked, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    const LMCAS::QueryInterface query(value->context());
    return checked_truth(query.query_periodic_checked(*checked, variable));
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_cas_assumptions_period(
    AssumptionsObj* value, ExprObj* expression, const char* variable) noexcept try {
    ensure_lmmc_runtime();
    if (!value || !variable)
        return result_error(MathErrorCode::InvalidArgument, __func__, "assumptions.period: invalid argument");
    std::string error;
    LMCAS::ExprPtr checked;
    if (!assumption_expr(expression, checked, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    const LMCAS::QueryInterface query(value->context());
    const auto result = query.get_period_checked(*checked, variable);
    if (!result) return result_error(result.error());
    if (!result.value())
        return result_error(MathErrorCode::Inconclusive, __func__, "assumptions.period: undetermined");
    return result_ok(
        new ExprObj(std::make_shared<LMCAS::SymbolicExpr>(*result.value())),
        ValueKind::Expr);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API StringObj* lmx_cas_assumptions_serialize(
    AssumptionsObj* value) noexcept try {
    ensure_lmmc_runtime();
    return new StringObj(value ? value->context().serialize() : "");
} catch (...) {
    return nullptr;
}
extern "C" LM_API AdtObj* lmx_cas_assumptions_parse(const char* source) noexcept try {
    ensure_lmmc_runtime();
    const auto result =
        LMCAS::AssumptionContext::deserialize_checked(source ? source : "");
    if (!result) return result_error(result.error());
    return assumptions_result(new AssumptionsObj(result.value()));
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_cas_solve_with_assumptions(
    AssumptionsObj* assumptions, ExprObj* equation, const char* variable) noexcept try {
    ensure_lmmc_runtime();
    if (!assumptions || !variable)
        return result_error(MathErrorCode::InvalidArgument, __func__, "solve.with_assumptions: invalid argument");
    std::string error;
    const auto* checked = checked_expr(equation, error);
    if (!checked) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return expr_array_result(LMCAS::solve_with_assumptions_checked(
        *checked, variable, &assumptions->context()));
} catch (...) {
    return c_abi_current_exception(__func__);
}
