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
using lmx::runtime::AssumptionsObj;

std::optional<lamina::Domain> assumption_domain(const char* value) {
    const std::string name = value ? value : "";
    if (name == "complex") return lamina::Domain::Complex;
    if (name == "real") return lamina::Domain::Real;
    if (name == "algebraic") return lamina::Domain::Algebraic;
    if (name == "rational") return lamina::Domain::Rational;
    if (name == "integer") return lamina::Domain::Integer;
    if (name == "natural") return lamina::Domain::Natural;
    if (name == "positive_int") return lamina::Domain::PositiveInt;
    return std::nullopt;
}

std::optional<lamina::Sign> assumption_sign(const char* value) {
    const std::string name = value ? value : "";
    if (name == "positive") return lamina::Sign::Positive;
    if (name == "negative") return lamina::Sign::Negative;
    if (name == "nonnegative") return lamina::Sign::NonNegative;
    if (name == "nonpositive") return lamina::Sign::NonPositive;
    if (name == "zero") return lamina::Sign::Zero;
    if (name == "nonzero") return lamina::Sign::NonZero;
    return std::nullopt;
}

AdtObj* assumptions_result(AssumptionsObj* value) {
    return result_ok(value, ValueKind::Assumptions);
}

AdtObj* truth_result(const lamina::Tribool value) {
    const char* constructor = value == lamina::Tribool::True
        ? "Proven" : value == lamina::Tribool::False
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
    ExprObj* expression, lamina::lsr::ExprPtr& output, std::string& error) {
    const auto* checked = checked_expr(expression, error);
    if (!checked) return false;
    output = *checked;
    return true;
}

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

extern "C" LM_API AssumptionsObj* lmx_computer_algebra_assumptions_empty() {
    return new AssumptionsObj();
}
extern "C" LM_API AdtObj* lmx_computer_algebra_assumptions_push(AssumptionsObj* value) {
    if (!value) return result_error(MathErrorCode::InvalidArgument, __func__, "assumptions.push: null context");
    auto* result = value->copy();
    result->context().push();
    return assumptions_result(result);
}
extern "C" LM_API AdtObj* lmx_computer_algebra_assumptions_pop(AssumptionsObj* value) {
    if (!value) return result_error(MathErrorCode::InvalidArgument, __func__, "assumptions.pop: null context");
    auto* result = value->copy();
    if (result->context().depth() <= 1) {
        result->release();
        return result_error(MathErrorCode::InvalidArgument, __func__,
                            "assumptions.pop: cannot pop root scope");
    }
    result->context().pop();
    return assumptions_result(result);
}
extern "C" LM_API AdtObj* lmx_computer_algebra_assumptions_with_domain(
    AssumptionsObj* value, const char* symbol, const char* domain) {
    if (!value || !symbol)
        return result_error(MathErrorCode::InvalidArgument, __func__, "assumptions.with_domain: invalid argument");
    const auto checked_domain = assumption_domain(domain);
    if (!checked_domain)
        return result_error(MathErrorCode::InvalidArgument, __func__, "assumptions.with_domain: unknown domain");
    auto* result = value->copy();
    const auto status =
        result->context().assume_domain_checked(symbol, *checked_domain);
    if (!status) {
        result->release();
        return result_error(status.error());
    }
    return assumptions_result(result);
}
extern "C" LM_API AdtObj* lmx_computer_algebra_assumptions_with_sign(
    AssumptionsObj* value, const char* symbol, const char* sign) {
    if (!value || !symbol)
        return result_error(MathErrorCode::InvalidArgument, __func__, "assumptions.with_sign: invalid argument");
    const auto checked_sign = assumption_sign(sign);
    if (!checked_sign)
        return result_error(MathErrorCode::InvalidArgument, __func__, "assumptions.with_sign: unknown sign");
    auto* result = value->copy();
    const auto status =
        result->context().assume_sign_checked(symbol, *checked_sign);
    if (!status) {
        result->release();
        return result_error(status.error());
    }
    return assumptions_result(result);
}
extern "C" LM_API AdtObj* lmx_computer_algebra_assumptions_with_relation(
    AssumptionsObj* value, ExprObj* relation) {
    if (!value) return result_error(MathErrorCode::InvalidArgument, __func__, "assumptions.with_relation: null context");
    std::string error;
    lamina::lsr::ExprPtr expression;
    if (!assumption_expr(relation, expression, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    auto* result = value->copy();
    const auto status = result->context().assume_checked(*expression);
    if (!status) {
        result->release();
        return result_error(status.error());
    }
    return assumptions_result(result);
}
extern "C" LM_API AdtObj* lmx_computer_algebra_assumptions_with_conditional(
    AssumptionsObj* value, ExprObj* condition, ExprObj* conclusion) {
    if (!value) return result_error(MathErrorCode::InvalidArgument, __func__, "assumptions.with_conditional: null context");
    std::string error;
    lamina::lsr::ExprPtr checked_condition;
    lamina::lsr::ExprPtr checked_conclusion;
    if (!assumption_expr(condition, checked_condition, error) ||
        !assumption_expr(conclusion, checked_conclusion, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    auto* result = value->copy();
    const auto status = result->context().assume_conditional_checked(
        *checked_condition, *checked_conclusion);
    if (!status) {
        result->release();
        return result_error(status.error());
    }
    return assumptions_result(result);
}

extern "C" LM_API AdtObj* lmx_computer_algebra_assumptions_query(
    AssumptionsObj* value, ExprObj* expression, const char* property) {
    if (!value || !property)
        return result_error(MathErrorCode::InvalidArgument, __func__, "assumptions.query: invalid argument");
    std::string error;
    lamina::lsr::ExprPtr checked;
    if (!assumption_expr(expression, checked, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    const lamina::QueryInterface query(value->context());
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
    if (name == "periodic") return checked_truth(query.query_periodic_checked(*checked));
    if (name == "positive_definite") return checked_truth(query.query_positive_definite_checked(*checked));
    if (name == "positive_semidefinite") return checked_truth(query.query_positive_semidefinite_checked(*checked));
    return result_error(MathErrorCode::InvalidArgument, __func__, "assumptions.query: unknown property");
}

extern "C" LM_API AdtObj* lmx_computer_algebra_assumptions_period(
    AssumptionsObj* value, ExprObj* expression) {
    if (!value) return result_error(MathErrorCode::InvalidArgument, __func__, "assumptions.period: null context");
    std::string error;
    lamina::lsr::ExprPtr checked;
    if (!assumption_expr(expression, checked, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    const lamina::QueryInterface query(value->context());
    const auto result = query.get_period_checked(*checked);
    if (!result) return result_error(result.error());
    if (!result.value())
        return result_error(MathErrorCode::Inconclusive, __func__, "assumptions.period: undetermined");
    return result_ok(
        new ExprObj(std::make_shared<SymbolicExpr>(*result.value())),
        ValueKind::Expr);
}
extern "C" LM_API StringObj* lmx_computer_algebra_assumptions_serialize(
    AssumptionsObj* value) {
    return new StringObj(value ? value->context().serialize() : "");
}
extern "C" LM_API AdtObj* lmx_computer_algebra_assumptions_parse(const char* source) {
    const auto result =
        lamina::AssumptionContext::deserialize_checked(source ? source : "");
    if (!result) return result_error(result.error());
    return assumptions_result(new AssumptionsObj(result.value()));
}

extern "C" LM_API AdtObj* lmx_computer_algebra_equation_solving_with_assumptions(
    AssumptionsObj* assumptions, ExprObj* equation, const char* variable) {
    if (!assumptions || !variable)
        return result_error(MathErrorCode::InvalidArgument, __func__, "solve.with_assumptions: invalid argument");
    std::string error;
    const auto* checked = checked_expr(equation, error);
    if (!checked) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    try {
        const auto solutions = lamina::solve_with_assumptions(
            *checked, variable, &assumptions->context());
        auto* result = new ArrayObj();
        for (const auto& solution : solutions)
            result->append(Value(new ExprObj(solution), ValueKind::Expr));
        return result_ok(result, ValueKind::Obj);
    } catch (const std::exception& error) {
        return result_error(MathErrorCode::InvalidArgument, __func__, 
            std::string("solve.with_assumptions: ") + error.what());
    }
}
