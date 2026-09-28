#include "bridge/result.hpp"

#include <limits>
#include <utility>

namespace lmx::bridge {
namespace {

Value expr_value(const LMCAS::ExprPtr& expr) {
    return take_object_value(make_owned_object<ExprObj>(expr), ValueKind::Expr);
}

Value expr_array(const std::vector<LMCAS::ExprPtr>& expressions) {
    auto array = make_owned_object<ArrayObj>();
    for (const auto& expr : expressions) array->append(expr_value(expr));
    return take_object_value(std::move(array), ValueKind::Obj);
}

}


ExprObj* expr_from_result(const LMCAS::ExprResult& result) {
    if (!result) return new ExprObj(result.error());
    return new ExprObj(result.value());
}

AdtObj* expr_result_ok(const LMCAS::ExprResult& result) {
    if (!result) return result_error(result.error());
    if (!result.value()) {
        return result_error(MathErrorCode::InternalError, __func__,
            "CasError(InternalInvariant: null expression result)");
    }
    return result_ok(new ExprObj(result.value()), ValueKind::Expr);
}

AdtObj* expr_pointer_result(LMCAS::ExprPtr value,
                                const char* operation) {
    if (!value) {
        return result_error(MathErrorCode::UnsupportedExpression,
                            operation ? operation : "computer_algebra",
                            "operation produced no expression");
    }
    return result_ok(new ExprObj(std::move(value)), ValueKind::Expr);
}

AdtObj* expression_set_literal_result(const LMCAS::ExprSetResult& result) {
    if (!result) return result_error(result.error());
    std::vector<Value> values;
    values.reserve(result.value().size());
    for (const auto& expression : result.value().elements()) {
        values.emplace_back(take_object_value(
            make_owned_object<ExprObj>(
                std::make_shared<LMCAS::SymbolicExpr>(*expression)), ValueKind::Expr));
    }
    return result_ok(
        new lmx::runtime::LiteralObj(
            lmx::runtime::LiteralObj::Kind::Set, std::move(values)),
        ValueKind::Set);
}

AdtObj* solution_set_result(const LMCAS::SolveResult& result) {
    if (!result) return result_error(result.error());
    const auto& set = result.value();
    if (std::holds_alternative<LMCAS::EmptySolutions>(set))
        return result_ok(new AdtObj("SolutionSet", "Empty", {}), ValueKind::Obj);
    if (std::holds_alternative<LMCAS::UniversalSolutions>(set))
        return result_ok(new AdtObj("SolutionSet", "Universal", {}), ValueKind::Obj);

    std::vector<Value> fields;
    if (const auto* finite = std::get_if<LMCAS::FiniteSolutions>(&set)) {
        auto values = make_owned_object<ArrayObj>();
        for (const auto& item : finite->values) {
            if (item.multiplicity >
                static_cast<std::size_t>(std::numeric_limits<LmInt>::max()))
                return result_error(MathErrorCode::InvalidArgument, "LMCAS.solve_set",
                                    "solution multiplicity exceeds Lamina int range");
            std::vector<Value> entry;
            entry.emplace_back(expr_value(item.value));
            entry.emplace_back(static_cast<LmInt>(item.multiplicity));
            entry.emplace_back(expr_array(item.conditions));
            values->append(take_object_value(
                make_owned_object<AdtObj>(
                    "FiniteSolution", "FiniteSolution", std::move(entry)),
                ValueKind::Obj));
        }
        fields.emplace_back(take_object_value(std::move(values), ValueKind::Obj));
        return result_ok(new AdtObj(
            "SolutionSet", "Finite", std::move(fields)), ValueKind::Obj);
    }
    if (const auto* intervals = std::get_if<LMCAS::IntervalSolutions>(&set)) {
        auto values = make_owned_object<ArrayObj>();
        for (const auto& item : intervals->values) {
            std::vector<Value> entry;
            entry.emplace_back(expr_value(item.lower));
            entry.emplace_back(expr_value(item.upper));
            entry.emplace_back(item.lower_closed);
            entry.emplace_back(item.upper_closed);
            entry.emplace_back(expr_array(item.conditions));
            values->append(take_object_value(
                make_owned_object<AdtObj>(
                    "IntervalSolution", "IntervalSolution", std::move(entry)),
                ValueKind::Obj));
        }
        fields.emplace_back(take_object_value(std::move(values), ValueKind::Obj));
        return result_ok(new AdtObj(
            "SolutionSet", "Interval", std::move(fields)), ValueKind::Obj);
    }
    if (const auto* conditional = std::get_if<LMCAS::ConditionalSolutions>(&set)) {
        const auto& item = conditional->value;
        std::vector<Value> entry;
        entry.emplace_back(take_object_value(
            make_owned_object<StringObj>(item.variable), ValueKind::Obj));
        entry.emplace_back(expr_value(item.predicate));
        entry.emplace_back(expr_array(item.conditions));
        fields.emplace_back(take_object_value(
            make_owned_object<AdtObj>(
                "ConditionSet", "ConditionSet", std::move(entry)), ValueKind::Obj));
        return result_ok(new AdtObj(
            "SolutionSet", "Conditional", std::move(fields)), ValueKind::Obj);
    }
    const auto& parametric = std::get<LMCAS::ParametricSolutions>(set);
    auto values = make_owned_object<ArrayObj>();
    for (const auto& item : parametric.values) {
        auto parameters = make_owned_object<ArrayObj>();
        for (const auto& name : item.integer_parameters)
            parameters->append(take_object_value(
                make_owned_object<StringObj>(name), ValueKind::Obj));
        std::vector<Value> entry;
        entry.emplace_back(expr_value(item.value));
        entry.emplace_back(take_object_value(std::move(parameters), ValueKind::Obj));
        entry.emplace_back(expr_array(item.conditions));
        values->append(take_object_value(
            make_owned_object<AdtObj>(
                "ParametricSolution", "ParametricSolution", std::move(entry)),
            ValueKind::Obj));
    }
    fields.emplace_back(take_object_value(std::move(values), ValueKind::Obj));
    return result_ok(new AdtObj(
        "SolutionSet", "Parametric", std::move(fields)), ValueKind::Obj);
}

AdtObj* transform_engine_result_value(const LMCAS::TransformEngineResult& result) {
    if (!result) return result_error(result.error());
    if (!result.value().value.expression) {
        return result_error(MathErrorCode::UnsupportedExpression, "transform",
                            "transform produced no expression");
    }
    return result_ok(
        new ExprObj(result.value().value.expression), ValueKind::Expr);
}

}
