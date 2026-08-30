#include "bridge/result.hpp"
#include "bridge/conversions.hpp"
#include "bridge/runtime_views.hpp"
#include "bridge/unit_bridge.hpp"
#include "limit_result.hpp"
#include <cstdarg>

using namespace lmx::bridge;

extern "C" LM_API AdtObj* lmx_computer_algebra_simplify(ExprObj* expr) {
    std::string error;
    const auto* value = checked_expr(expr, error);
    if (!value)
        return result_error(MathErrorCode::InvalidArgument, "lmx_computer_algebra_simplify",
                            std::move(error));
    return expr_result_ok(lamina::lsr::simplify(*value));
}

extern "C" LM_API AdtObj* lmx_computer_algebra_expand(ExprObj* expr) {
    std::string error;
    const auto* value = checked_expr(expr, error);
    if (!value)
        return result_error(MathErrorCode::InvalidArgument, "lmx_computer_algebra_expand",
                            std::move(error));
    return expr_result_ok(lamina::lsr::expand(*value));
}

extern "C" LM_API AdtObj* lmx_computer_algebra_differentiate_by_name(ExprObj* expr, const char* variable) {
    std::string error;
    const auto* value = checked_expr(expr, error);
    if (!value)
        return result_error(MathErrorCode::InvalidArgument, "lmx_computer_algebra_differentiate_by_name",
                            std::move(error));
    return expr_result_ok(
        lamina::lsr::differentiate(*value, variable ? variable : ""));
}

/**
 * @brief Differentiates an expression with respect to a symbol expression.
 * @param expr Borrowed valid Lamina expression to differentiate.
 * @param variable Borrowed Lamina expression containing exactly one symbol.
 * @return A newly allocated expression containing the derivative or CasError.
 * @ownership Caller owns the returned ExprObj; inputs are borrowed.
 * @threadsafe Current VM thread only.
 */
extern "C" LM_API AdtObj* lmx_computer_algebra_differentiate_by_symbol(ExprObj* expr, ExprObj* variable) {
    std::string name;
    std::string error;
    if (!checked_symbol_name(variable, name, error))
        return result_error(MathErrorCode::InvalidArgument, "lmx_computer_algebra_differentiate_by_name",
                            std::move(error));
    return lmx_computer_algebra_differentiate_by_name(expr, name.c_str());
}

extern "C" LM_API AdtObj* lmx_computer_algebra_square_root(ExprObj* expr) {
    return unary_expression_result(expr, "lmx_computer_algebra_square_root", [](const auto& value) {
        return lamina::lsr::sqrt(value);
    });
}

extern "C" LM_API AdtObj* lmx_computer_algebra_sine(ExprObj* expr) {
    return unary_expression_result(expr, "lmx_computer_algebra_sine", [](const auto& value) {
        return lamina::lsr::sin(value);
    });
}

extern "C" LM_API AdtObj* lmx_computer_algebra_cosine(ExprObj* expr) {
    return unary_expression_result(expr, "lmx_computer_algebra_cosine", [](const auto& value) {
        return lamina::lsr::cos(value);
    });
}

extern "C" LM_API AdtObj* lmx_computer_algebra_tangent(ExprObj* expr) {
    return unary_expression_result(expr, "lmx_computer_algebra_tangent", [](const auto& value) {
        return lamina::lsr::tan(value);
    });
}

extern "C" LM_API AdtObj* lmx_computer_algebra_inverse_sine(ExprObj* expr) {
    return unary_expression_result(expr, "lmx_computer_algebra_inverse_sine", [](const auto& value) {
        return lamina::lsr::asin(value);
    });
}

extern "C" LM_API AdtObj* lmx_computer_algebra_inverse_cosine(ExprObj* expr) {
    return unary_expression_result(expr, "lmx_computer_algebra_inverse_cosine", [](const auto& value) {
        return lamina::lsr::acos(value);
    });
}

extern "C" LM_API AdtObj* lmx_computer_algebra_inverse_tangent(ExprObj* expr) {
    return unary_expression_result(expr, "lmx_computer_algebra_inverse_tangent", [](const auto& value) {
        return lamina::lsr::atan(value);
    });
}

extern "C" LM_API AdtObj* lmx_computer_algebra_exponential(ExprObj* expr) {
    return unary_expression_result(expr, "lmx_computer_algebra_exponential", [](const auto& value) {
        return lamina::lsr::exp(value);
    });
}

extern "C" LM_API AdtObj* lmx_computer_algebra_natural_logarithm(ExprObj* expr) {
    return unary_expression_result(expr, "lmx_computer_algebra_natural_logarithm", [](const auto& value) {
        return lamina::lsr::log(value);
    });
}

extern "C" LM_API AdtObj* lmx_computer_algebra_common_logarithm(ExprObj* expr) {
    return unary_expression_result(expr, "lmx_computer_algebra_common_logarithm", [](const auto& value) {
        return lamina::lsr::log10(value);
    });
}

extern "C" LM_API AdtObj* lmx_computer_algebra_floor(ExprObj* expr) {
    return unary_expression_result(expr, "lmx_computer_algebra_floor", [](const auto& value) {
        return lamina::lsr::floor(value);
    });
}

extern "C" LM_API AdtObj* lmx_computer_algebra_ceil(ExprObj* expr) {
    return unary_expression_result(expr, "lmx_computer_algebra_ceil", [](const auto& value) {
        return lamina::lsr::ceil(value);
    });
}

extern "C" LM_API AdtObj* lmx_computer_algebra_round(ExprObj* expr) {
    return unary_expression_result(expr, "lmx_computer_algebra_round", [](const auto& value) {
        return lamina::lsr::round(value);
    });
}

extern "C" LM_API AdtObj* lmx_computer_algebra_real_part(ExprObj* expr) {
    return unary_expression_result(expr, "lmx_computer_algebra_real_part", [](const auto& value) {
        return lamina::lsr::real(value);
    });
}

extern "C" LM_API AdtObj* lmx_computer_algebra_imaginary_part(ExprObj* expr) {
    return unary_expression_result(expr, "lmx_computer_algebra_imaginary_part", [](const auto& value) {
        return lamina::lsr::imag(value);
    });
}

extern "C" LM_API AdtObj* lmx_computer_algebra_conjugate(ExprObj* expr) {
    return unary_expression_result(expr, "lmx_computer_algebra_conjugate", [](const auto& value) {
        return lamina::lsr::conj(value);
    });
}

extern "C" LM_API AdtObj* lmx_computer_algebra_absolute_value(ExprObj* expr) {
    return unary_expression_result(expr, "lmx_computer_algebra_absolute_value", [](const auto& value) {
        return lamina::lsr::abs(value);
    });
}

extern "C" LM_API AdtObj* lmx_computer_algebra_clamp(ExprObj* expr, ExprObj* lower,
                                    ExprObj* upper) {
    std::string error;
    const auto* value = checked_expr(expr, error);
    const auto* lower_value = checked_expr(lower, error);
    const auto* upper_value = checked_expr(upper, error);
    if (!value || !lower_value || !upper_value)
        return result_error(MathErrorCode::InvalidArgument, "lmx_computer_algebra_clamp",
                            std::move(error));
    return expr_result_ok(
        lamina::lsr::clamp(*value, *lower_value, *upper_value));
}

extern "C" LM_API AdtObj* lmx_computer_algebra_integrate_by_name(ExprObj* expr,
                                        const char* variable) {
    std::string error;
    const auto* value = checked_expr(expr, error);
    if (!value)
        return result_error(MathErrorCode::InvalidArgument, "lmx_computer_algebra_integrate_by_name",
                            std::move(error));
    try {
        return expr_pointer_result(
            (*value)->integrate(variable ? variable : ""), "lmx_computer_algebra_integrate_by_name");
    } catch (const std::exception& exception) {
        return result_error(MathErrorCode::InternalError, "lmx_computer_algebra_integrate_by_name",
                            exception.what());
    }
}

/**
 * @brief Integrates an expression with respect to a symbol expression.
 * @param expr Borrowed valid Lamina expression to integrate.
 * @param variable Borrowed Lamina expression containing exactly one symbol.
 * @return A newly allocated antiderivative expression or CasError.
 * @ownership Caller owns the returned ExprObj; inputs are borrowed.
 * @threadsafe Current VM thread only.
 */
extern "C" LM_API AdtObj* lmx_computer_algebra_integrate_by_symbol(
    ExprObj* expr, ExprObj* variable) {
    std::string name;
    std::string error;
    if (!checked_symbol_name(variable, name, error))
        return result_error(MathErrorCode::InvalidArgument, "lmx_computer_algebra_integrate_by_name",
                            std::move(error));
    return lmx_computer_algebra_integrate_by_name(expr, name.c_str());
}

extern "C" LM_API AdtObj* lmx_computer_algebra_limit_by_name(
    ExprObj* expr, const char* variable, ExprObj* point,
    const char* direction) {
    std::string error;
    const auto* value = checked_expr(expr, error);
    const auto* point_value = checked_expr(point, error);
    if (!value || !point_value)
        return result_error(MathErrorCode::InvalidArgument, "lmx_computer_algebra_limit_by_name",
                            std::move(error));
    const std::string_view token = direction ? direction : "";
    LimitDirection parsed_direction = LimitDirection::Both;
    if (token == "-" || token == "below" || token == "left" ||
        token == "from_below") {
        parsed_direction = LimitDirection::FromBelow;
    } else if (token == "+" || token == "above" || token == "right" ||
               token == "from_above") {
        parsed_direction = LimitDirection::FromAbove;
    } else if (!token.empty() && token != "both") {
        return result_error(
            MathErrorCode::InvalidArgument,
            "lmx_computer_algebra_limit_by_name",
            "invalid limit direction");
    }
    auto result = lamina::limit_checked(
        *value, variable ? variable : "", *point_value, parsed_direction);
    if (!result) return result_error(result.error());
    const auto& outcome = result.value().value;
    std::shared_ptr<SymbolicExpr> expression;
    if (const auto* finite = std::get_if<lamina::FiniteLimit>(&outcome)) {
        expression = finite->value;
    } else if (std::holds_alternative<lamina::PositiveInfinityLimit>(
                   outcome)) {
        expression = SymbolicExpr::infinity(1);
    } else if (std::holds_alternative<lamina::NegativeInfinityLimit>(
                   outcome)) {
        expression = SymbolicExpr::infinity(-1);
    } else {
        return result_error(lamina::CasError{
            lamina::CasErrc::Inconclusive,
            "limit does not exist",
            "lmx_computer_algebra_limit_by_name"});
    }
    return result_ok(new ExprObj(std::move(expression)), ValueKind::Expr);
}

/**
 * @brief Evaluates a limit using a symbol expression as its variable.
 * @param expr Borrowed valid Lamina expression.
 * @param variable Borrowed Lamina expression containing exactly one symbol.
 * @param point Borrowed valid Lamina limit point.
 * @param direction Borrowed textual direction accepted by the existing limit binding.
 * @return A newly allocated limit expression or CasError.
 * @ownership Caller owns the return; all inputs are borrowed.
 * @threadsafe Current VM thread only.
 */
extern "C" LM_API AdtObj* lmx_computer_algebra_limit_by_symbol(
    ExprObj* expr, ExprObj* variable, ExprObj* point, const char* direction) {
    std::string name;
    std::string error;
    if (!checked_symbol_name(variable, name, error))
        return result_error(MathErrorCode::InvalidArgument, "lmx_computer_algebra_limit_by_name",
                            std::move(error));
    return lmx_computer_algebra_limit_by_name(expr, name.c_str(), point, direction);
}

extern "C" LM_API AdtObj* lmx_computer_algebra_series_by_name(
    ExprObj* expr, const char* variable, ExprObj* point, const LmInt order) {
    std::string error;
    const auto* value = checked_expr(expr, error);
    const auto* point_value = checked_expr(point, error);
    if (!value || !point_value)
        return result_error(MathErrorCode::InvalidArgument, "lmx_computer_algebra_series_by_name",
                            std::move(error));
    if (order < 0 || order > std::numeric_limits<int>::max())
        return result_error(MathErrorCode::InvalidArgument, "lmx_computer_algebra_series_by_name",
                            "invalid order");
    try {
        return expr_pointer_result(
            (*value)->series(variable ? variable : "", *point_value,
                             static_cast<int>(order)),
            "lmx_computer_algebra_series_by_name");
    } catch (const std::exception& exception) {
        return result_error(MathErrorCode::InternalError, "lmx_computer_algebra_series_by_name",
                            exception.what());
    }
}

/**
 * @brief Expands a series using a symbol expression as its variable.
 * @param expr Borrowed valid Lamina expression.
 * @param variable Borrowed Lamina expression containing exactly one symbol.
 * @param point Borrowed valid expansion point.
 * @param order Non-negative expansion order in Lamina integer range.
 * @return A newly allocated series expression or CasError.
 * @ownership Caller owns the return; all inputs are borrowed.
 * @threadsafe Current VM thread only.
 */
extern "C" LM_API AdtObj* lmx_computer_algebra_series_symbol(
    ExprObj* expr, ExprObj* variable, ExprObj* point, const LmInt order) {
    std::string name;
    std::string error;
    if (!checked_symbol_name(variable, name, error))
        return result_error(MathErrorCode::InvalidArgument, "lmx_computer_algebra_series_by_name",
                            std::move(error));
    return lmx_computer_algebra_series_by_name(expr, name.c_str(), point, order);
}

extern "C" LM_API AdtObj* lmx_computer_algebra_substitute(ExprObj* expr, AdtObj* binding) {
    std::string error;
    const auto* value = checked_expr(expr, error);
    if (!value || !binding || binding->type_name() != "Binding" ||
        binding->constructor() != "Binding" || binding->fields().size() != 2) {
        return result_error(MathErrorCode::InvalidArgument, "lmx_computer_algebra_substitute",
                            value ? "expected Binding<expr, expr>"
                                  : std::move(error));
    }
    const auto* symbol_field = binding->field(0);
    const auto* value_field = binding->field(1);
    if (!symbol_field || symbol_field->kind != ValueKind::Expr ||
        !value_field || value_field->kind != ValueKind::Expr) {
        return result_error(MathErrorCode::InvalidArgument, "lmx_computer_algebra_substitute",
                            "expected Binding<expr, expr>");
    }
    const auto* symbol = checked_expr(
        reinterpret_cast<ExprObj*>(symbol_field->obj), error);
    const auto* replacement = checked_expr(
        reinterpret_cast<ExprObj*>(value_field->obj), error);
    if (!symbol || !replacement)
        return result_error(MathErrorCode::InvalidArgument, "lmx_computer_algebra_substitute",
                            std::move(error));
    const auto checked_binding = lamina::lsr::binding(*symbol, *replacement);
    if (!checked_binding) return result_error(checked_binding.error());
    return expr_result_ok(
        lamina::lsr::substitute(*value, checked_binding.value()));
}

extern "C" LM_API AdtObj* lmx_computer_algebra_substitute_named_by_name(
    ExprObj* expr, const char* variable, ExprObj* replacement) {
    std::string error;
    const auto* value = checked_expr(expr, error);
    const auto* replacement_value = checked_expr(replacement, error);
    if (!value || !replacement_value || !variable || variable[0] == '\0') {
        return result_error(MathErrorCode::InvalidArgument, "lmx_computer_algebra_substitute",
                            error.empty() ? "empty variable" : std::move(error));
    }
    return expr_result_ok(
        lamina::lsr::substitute(*value, variable, *replacement_value));
}

/**
 * @brief Substitutes a value for a variable supplied as a symbol expression.
 * @param expr Borrowed valid Lamina expression.
 * @param variable Borrowed Lamina expression containing exactly one symbol.
 * @param replacement Borrowed valid Lamina replacement expression.
 * @return A newly allocated substituted expression or CasError.
 * @ownership Caller owns the return; all inputs are borrowed.
 * @threadsafe Current VM thread only.
 */
extern "C" LM_API AdtObj* lmx_computer_algebra_substitute_named_by_symbol(
    ExprObj* expr, ExprObj* variable, ExprObj* replacement) {
    std::string name;
    std::string error;
    if (!checked_symbol_name(variable, name, error))
        return result_error(MathErrorCode::InvalidArgument, "lmx_computer_algebra_substitute",
                            std::move(error));
    return lmx_computer_algebra_substitute_named_by_name(expr, name.c_str(), replacement);
}

extern "C" LM_API AdtObj* lmx_computer_algebra_substitute_many(
    ExprObj* expr, ArrayObj* variables, ArrayObj* replacements) {
    std::string error;
    const auto* value = checked_expr(expr, error);
    std::vector<std::string> names;
    std::vector<lamina::lsr::ExprPtr> values;
    if (!value || !array_strings(variables, names, error) ||
        !array_expressions(replacements, values, error) ||
        names.size() != values.size()) {
        return result_error(MathErrorCode::InvalidArgument, "lmx_computer_algebra_substitute",
                            error.empty() ? "binding count mismatch"
                                          : std::move(error));
    }
    std::vector<lamina::lsr::Binding> bindings;
    bindings.reserve(names.size());
    for (std::size_t index = 0; index < names.size(); ++index) {
        const auto symbol = lamina::lsr::sym(names[index]);
        if (!symbol) return result_error(symbol.error());
        const auto checked = lamina::lsr::binding(symbol.value(), values[index]);
        if (!checked) return result_error(checked.error());
        bindings.push_back(checked.value());
    }
    return expr_result_ok(lamina::lsr::substitute(*value, bindings));
}

extern "C" LM_API AdtObj* lmx_computer_algebra_finite_set(ArrayObj* elements) {
    std::vector<lamina::lsr::ExprPtr> values;
    std::string error;
    if (!array_expressions(elements, values, error))
        return result_error(MathErrorCode::InvalidArgument, "lmx_computer_algebra_finite_set",
                            std::move(error));
    return expr_result_ok(lamina::lsr::finite_set(std::move(values)));
}

extern "C" LM_API AdtObj* lmx_computer_algebra_interval(
    ExprObj* lower, ExprObj* upper, const bool lower_closed,
    const bool upper_closed) {
    std::string error;
    const auto* lower_value = checked_expr(lower, error);
    const auto* upper_value = checked_expr(upper, error);
    if (!lower_value || !upper_value)
        return result_error(MathErrorCode::InvalidArgument, "lmx_computer_algebra_interval",
                            std::move(error));
    return expr_result_ok(lamina::lsr::interval(
        *lower_value, *upper_value, lower_closed, upper_closed));
}

extern "C" LM_API AdtObj* lmx_computer_algebra_match(ExprObj* pattern, ExprObj* target,
                                     ArrayObj* wildcards) {
    std::string error;
    const auto* pattern_value = checked_expr(pattern, error);
    if (!pattern_value) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    const auto* target_value = checked_expr(target, error);
    if (!target_value) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    std::vector<std::string> names;
    if (!array_strings(wildcards, names, error)) return result_error(MathErrorCode::InvalidArgument, __func__, error);
    const auto result = lamina::lsr::expr_match(*pattern_value, *target_value,
                                                names);
    if (!result) return result_error(result.error());
    if (!result.value().matched) return result_error(MathErrorCode::InvalidArgument, __func__, "CAS pattern did not match");
    std::vector<TableObj::Entry> entries;
    entries.reserve(result.value().bindings.size());
    for (const auto& binding : result.value().bindings) {
        entries.emplace_back(binding.name,
                             Value(new ExprObj(binding.value), ValueKind::Expr));
    }
    return result_ok(new TableObj(std::move(entries)), ValueKind::Table);
}

extern "C" LM_API AdtObj* lmx_computer_algebra_table_expr(TableObj* value, const char* key) {
    if (!value || !key) return result_error(MathErrorCode::InvalidArgument, __func__, "CAS table lookup: invalid argument");
    const auto* field = value->find(key);
    if (!field || field->kind != ValueKind::Expr || !field->obj) {
        return result_error(MathErrorCode::InvalidArgument, __func__, "CAS table lookup: key is missing or is not an expression");
    }
    return result_ok(field->obj->get(), ValueKind::Expr);
}

extern "C" LM_API AdtObj* lmx_computer_algebra_evaluate_real(ExprObj* expr) {
    std::string error;
    const auto* value = checked_expr(expr, error);
    if (!value) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    const auto result = lamina::lsr::evalf(**value);
    if (!result) return result_error(result.error());
    return result_ok(result.value().value);
}

extern "C" LM_API AdtObj* lmx_computer_algebra_evaluate_real_with(ExprObj* expr, ArrayObj* variables,
                                          ArrayObj* values) {
    std::string error;
    const auto* expression = checked_expr(expr, error);
    if (!expression) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    std::vector<std::string> names;
    std::vector<double> numbers;
    if (!array_strings(variables, names, error) ||
        !array_numbers(values, numbers, error)) return result_error(MathErrorCode::InvalidArgument, __func__, error);
    if (names.size() != numbers.size()) {
        return result_error(MathErrorCode::InvalidArgument, __func__, "CAS numeric binding count mismatch");
    }
    lamina::NumericBindings bindings;
    for (std::size_t index = 0; index < names.size(); ++index) {
        bindings[names[index]] = numbers[index];
    }
    const auto result = lamina::lsr::evalf(**expression, bindings);
    if (!result) return result_error(result.error());
    return result_ok(result.value().value);
}

extern "C" LM_API AdtObj* lmx_computer_algebra_evaluate_complex(ExprObj* expr) {
    std::string error;
    const auto* value = checked_expr(expr, error);
    if (!value) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    const auto result = lamina::lsr::eval_complex(**value);
    if (!result) return result_error(result.error());
    return complex_result_ok({result.value().real.value,
                              result.value().imag.value});
}

extern "C" LM_API AdtObj* lmx_computer_algebra_evaluate_complex_with(ExprObj* expr,
                                                  ArrayObj* variables,
                                                  ArrayObj* values) {
    std::string error;
    const auto* expression = checked_expr(expr, error);
    if (!expression) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    std::vector<std::string> names;
    std::vector<double> numbers;
    if (!array_strings(variables, names, error) ||
        !array_numbers(values, numbers, error)) return result_error(MathErrorCode::InvalidArgument, __func__, error);
    if (names.size() != numbers.size()) {
        return result_error(MathErrorCode::InvalidArgument, __func__, "CAS numeric binding count mismatch");
    }
    lamina::NumericBindings bindings;
    for (std::size_t index = 0; index < names.size(); ++index) {
        bindings[names[index]] = numbers[index];
    }
    const auto result = lamina::lsr::eval_complex(**expression, bindings);
    if (!result) return result_error(result.error());
    return complex_result_ok({result.value().real.value,
                              result.value().imag.value});
}

extern "C" LM_API bool lmx_computer_algebra_structurally_equal(ExprObj* lhs, ExprObj* rhs) {
    std::string error;
    const auto* left = checked_expr(lhs, error);
    const auto* right = checked_expr(rhs, error);
    return left && right && lamina::lsr::structurally_equal(**left, **right);
}

extern "C" LM_API AdtObj* lmx_computer_algebra_equivalent(ExprObj* lhs, ExprObj* rhs) {
    std::string error;
    const auto* left = checked_expr(lhs, error);
    if (!left) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    const auto* right = checked_expr(rhs, error);
    if (!right) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    lamina::ComputationContext context;
    const auto result = lamina::lsr::equivalent(**left, **right, context);
    if (!result) return result_error(result.error());
    return result_ok(result.value());
}
