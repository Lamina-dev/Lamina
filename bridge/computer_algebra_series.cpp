#include "bridge/result.hpp"
#include "bridge/conversions.hpp"
#include "bridge/runtime_views.hpp"
#include "bridge/unit_bridge.hpp"
#include <cstdarg>
#include "bridge/math_internal.hpp"
#include "series_engine.hpp"
#include "symbolic_geometry.hpp"
#include "symbolic.hpp"
#include "calculus_utils.hpp"
#include "multiple_integral.hpp"

using namespace lmx::bridge;
using lmx::bridge::math_internal::checked_expression_operation;
using lmx::bridge::math_internal::checked_expr_result;

/** @brief Adds ordered power-series coefficients. @param lhs Borrowed coefficients. @param rhs Borrowed coefficients. @return Owning Result coefficient array or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_series_add(ArrayObj* lhs, ArrayObj* rhs) {
    std::vector<lamina::lsr::ExprPtr> a, b; std::string error;
    if (!array_expressions(lhs, a, error) || !array_expressions(rhs, b, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "series.add: " + error);
    try {
        auto* values = new ArrayObj();
        for (const auto& expression : lamina::power_series_add(a, b))
            values->append(Value(new ExprObj(expression), ValueKind::Expr));
        return result_ok(values, ValueKind::Obj);
    } catch (const std::exception& exception) {
        return result_error(MathErrorCode::InvalidArgument, __func__, std::string("series.add: ") + exception.what());
    }
}
/** @brief Multiplies ordered power-series coefficients. @param lhs Borrowed coefficients. @param rhs Borrowed coefficients. @param order Positive truncation order. @return Owning Result coefficient array or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_series_multiply(
    ArrayObj* lhs, ArrayObj* rhs, LmInt order) {
    if (order < 0 || order > std::numeric_limits<int>::max())
        return result_error(MathErrorCode::InvalidArgument, __func__, "series.multiply: invalid order");
    std::vector<lamina::lsr::ExprPtr> a, b; std::string error;
    if (!array_expressions(lhs, a, error) || !array_expressions(rhs, b, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "series.multiply: " + error);
    return expr_array_result(lamina::power_series_multiply_checked(
        a, b, static_cast<int>(order)));
}
/** @brief Composes ordered power-series coefficients. @param outer Borrowed outer coefficients. @param inner Borrowed inner coefficients. @param order Positive truncation order. @return Owning Result coefficient array or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_series_compose(
    ArrayObj* outer, ArrayObj* inner, LmInt order) {
    if (order < 0 || order > std::numeric_limits<int>::max())
        return result_error(MathErrorCode::InvalidArgument, __func__, "series.compose: invalid order");
    std::vector<lamina::lsr::ExprPtr> f, g; std::string error;
    if (!array_expressions(outer, f, error) || !array_expressions(inner, g, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "series.compose: " + error);
    return expr_array_result(lamina::power_series_compose_checked(
        f, g, static_cast<int>(order)));
}
/** @brief Builds a truncated Fourier series. @param value Borrowed function. @param variable Borrowed variable name. @param period Borrowed period. @param terms Positive term count. @return Owning Result Expr or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_series_fourier_by_name(
    ExprObj* value, const char* variable, ExprObj* period, LmInt terms) {
    if (!variable || variable[0] == '\0' || terms < 0 ||
        terms > std::numeric_limits<int>::max())
        return result_error(MathErrorCode::InvalidArgument, __func__, "series.fourier_series: invalid argument");
    std::string error; const auto* f = checked_expr(value, error);
    const auto* p = checked_expr(period, error);
    if (!f || !p) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    try {
        auto result = lamina::fourier_series(
            *f, variable, *p, static_cast<int>(terms));
        return result ? result_ok(new ExprObj(std::move(result)), ValueKind::Expr)
                      : result_error(MathErrorCode::UnsupportedExpression, __func__, "series.fourier_series: unsupported");
    } catch (const std::exception& exception) {
        return result_error(MathErrorCode::InvalidArgument, __func__, 
            std::string("series.fourier_series: ") + exception.what());
    }
}
/** @brief Computes convergence radius. @param coefficients Borrowed coefficients. @param variable Borrowed variable name. @return Owning Result Expr or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_series_convergence_radius_by_name(
    ArrayObj* coefficients, const char* variable) {
    if (!variable || variable[0] == '\0')
        return result_error(MathErrorCode::InvalidArgument, __func__, "series.convergence_radius: empty variable");
    std::vector<lamina::lsr::ExprPtr> values; std::string error;
    if (!array_expressions(coefficients, values, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "series.convergence_radius: " + error);
    return checked_expr_result(
        lamina::convergence_radius_checked(values, variable));
}
/** @brief Computes a convergence-test classification. @param value Borrowed general term. @param variable Borrowed index name. @return Owning Result ConvergenceInfo or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_series_convergence_test_by_name(
    ExprObj* value, const char* variable) {
    if (!variable || variable[0] == '\0')
        return result_error(MathErrorCode::InvalidArgument, __func__, "series.convergence_test: empty variable");
    std::string error; const auto* term = checked_expr(value, error);
    if (!term) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    const auto result = lamina::convergence_test_checked(*term, variable);
    if (!result) return result_error(result.error());
    const char* state = result.value().result == lamina::ConvergenceResult::Convergent
        ? "convergent" : result.value().result == lamina::ConvergenceResult::Divergent
        ? "divergent" : "inconclusive";
    std::vector<Value> fields;
    fields.emplace_back(new StringObj(state), ValueKind::Obj);
    fields.emplace_back(new StringObj(result.value().test_used), ValueKind::Obj);
    return result_ok(new AdtObj(
        "ConvergenceInfo", "ConvergenceInfo", std::move(fields)), ValueKind::Obj);
}
/** @brief Computes sequence lim inf. @param value Borrowed term. @param variable Borrowed index name. @return Owning Result Expr or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_series_limit_inferior_by_name(
    ExprObj* value, const char* variable) {
    if (!variable || variable[0] == '\0')
        return result_error(MathErrorCode::InvalidArgument, __func__, "series.limit_inferior: empty variable");
    return checked_expression_operation("series.limit_inferior", value,
        [&](const auto& term) { return lamina::lim_inf(term, variable); });
}
/** @brief Computes sequence lim sup. @param value Borrowed term. @param variable Borrowed index name. @return Owning Result Expr or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_series_limit_superior_by_name(
    ExprObj* value, const char* variable) {
    if (!variable || variable[0] == '\0')
        return result_error(MathErrorCode::InvalidArgument, __func__, "series.limit_superior: empty variable");
    return checked_expression_operation("series.limit_superior", value,
        [&](const auto& term) { return lamina::lim_sup(term, variable); });
}

/** @brief Computes total differential components. @param value Borrowed expression. @param variables Borrowed variable-name array. @return Owning Result array of component tables or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_calculus_total_differential_by_names(
    ExprObj* value, ArrayObj* variables) {
    std::string error; const auto* expression = checked_expr(value, error);
    std::vector<std::string> names;
    if (!expression || !array_strings(variables, names, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "calculus.total_differential: " + error);
    try {
        auto* components = new ArrayObj();
        for (const auto& [derivative, variable] :
             lamina::total_differential(*expression, names)) {
            if (!derivative) {
                components->release();
                return result_error(MathErrorCode::InvalidArgument, __func__, 
                    "calculus.total_differential: null derivative");
            }
            std::vector<TableObj::Entry> entries;
            entries.emplace_back("derivative", Value(
                new ExprObj(derivative), ValueKind::Expr));
            entries.emplace_back("variable", Value(
                new StringObj(variable), ValueKind::Obj));
            components->append(Value(
                new TableObj(std::move(entries)), ValueKind::Table));
        }
        return result_ok(components, ValueKind::Obj);
    } catch (const std::exception& exception) {
        return result_error(MathErrorCode::InvalidArgument, __func__, 
            std::string("calculus.total_differential: ") + exception.what());
    }
}
/** @brief Computes total differential components from symbols. @param value Borrowed expression. @param variables Borrowed symbol array. @return Owning Result array of component tables or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_calculus_total_differential_by_symbols(
    ExprObj* value, ArrayObj* variables) {
    std::string error; auto* names = math_internal::symbol_text_array(variables, error);
    if (!names) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    auto* result = lmx_computer_algebra_calculus_total_differential_by_names(value, names);
    names->release();
    return result;
}
/** @brief Applies logarithmic differentiation. @param value Borrowed expression. @param variable Borrowed variable name. @return Owning Result Expr or error. @ownership Input borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_calculus_log_differentiate_by_name(
    ExprObj* value, const char* variable) {
    if (!variable || variable[0] == '\0')
        return result_error(MathErrorCode::InvalidArgument, __func__, "calculus.log_differentiate: empty variable");
    return checked_expression_operation("calculus.log_differentiate", value,
        [&](const auto& expression) {
            return lamina::log_differentiate(expression, variable);
        });
}
/** @brief Applies logarithmic differentiation using a symbol. @param value Borrowed expression. @param variable Borrowed symbol. @return Owning Result Expr or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_calculus_log_differentiate_by_symbol(
    ExprObj* value, ExprObj* variable) {
    std::string name, error;
    if (!checked_symbol_name(variable, name, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return lmx_computer_algebra_calculus_log_differentiate_by_name(value, name.c_str());
}
/** @brief Computes inverse-function derivative. @param value Borrowed function. @param variable Borrowed variable name. @param point Borrowed target point. @return Owning Result Expr or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_calculus_inverse_derivative_by_name(
    ExprObj* value, const char* variable, ExprObj* point) {
    if (!variable || variable[0] == '\0')
        return result_error(MathErrorCode::InvalidArgument, __func__, "calculus.inverse_derivative: empty variable");
    std::string error; const auto* function = checked_expr(value, error);
    const auto* target = checked_expr(point, error);
    if (!function || !target) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return checked_expr_result(lamina::inverse_derivative_checked(
        *function, variable, *target));
}
/** @brief Computes inverse-function derivative using a symbol. @param value Borrowed function. @param variable Borrowed symbol. @param point Borrowed target point. @return Owning Result Expr or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_calculus_inverse_derivative_by_symbol(
    ExprObj* value, ExprObj* variable, ExprObj* point) {
    std::string name, error;
    if (!checked_symbol_name(variable, name, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return lmx_computer_algebra_calculus_inverse_derivative_by_name(value, name.c_str(), point);
}
/** @brief Solves an inverse function into unordered branches. @param value Borrowed function. @param variable Borrowed variable name. @param target Borrowed target expression. @return Owning Result set or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_calculus_inverse_function_by_name(
    ExprObj* value, const char* variable, ExprObj* target) {
    if (!variable || variable[0] == '\0')
        return result_error(MathErrorCode::InvalidArgument, __func__, "calculus.inverse_function: empty variable");
    std::string error; const auto* function = checked_expr(value, error);
    const auto* y = checked_expr(target, error);
    if (!function || !y) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    const auto result = lamina::inverse_function_checked(*function, variable, *y);
    if (!result) return result_error(result.error());
    return math_internal::unordered_expr_result(result.value());
}
/** @brief Solves an inverse function using a symbol. @param value Borrowed function. @param variable Borrowed symbol. @param target Borrowed target expression. @return Owning Result set or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_calculus_inverse_function_by_symbol(
    ExprObj* value, ExprObj* variable, ExprObj* target) {
    std::string name, error;
    if (!checked_symbol_name(variable, name, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return lmx_computer_algebra_calculus_inverse_function_by_name(value, name.c_str(), target);
}
/** @brief Computes asymptote components. @param value Borrowed function. @param variable Borrowed variable name. @return Owning Result Asymptotes or error. @ownership Input borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_calculus_asymptotes_by_name(
    ExprObj* value, const char* variable) {
    if (!variable || variable[0] == '\0')
        return result_error(MathErrorCode::InvalidArgument, __func__, "calculus.asymptotes: empty variable");
    std::string error; const auto* function = checked_expr(value, error);
    if (!function) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    const auto result = lamina::asymptotes_checked(*function, variable);
    if (!result) return result_error(result.error());
    auto* vertical = new ArrayObj();
    auto* horizontal = new ArrayObj();
    auto* oblique = new ArrayObj();
    for (const auto& expression : result.value().vertical)
        vertical->append(Value(new ExprObj(expression), ValueKind::Expr));
    for (const auto& expression : result.value().horizontal)
        horizontal->append(Value(new ExprObj(expression), ValueKind::Expr));
    for (const auto& [slope, intercept] : result.value().oblique) {
        auto* pair = new ArrayObj();
        pair->append(Value(new ExprObj(slope), ValueKind::Expr));
        pair->append(Value(new ExprObj(intercept), ValueKind::Expr));
        oblique->append(Value(pair, ValueKind::Obj));
    }
    std::vector<Value> fields;
    fields.emplace_back(vertical, ValueKind::Obj);
    fields.emplace_back(horizontal, ValueKind::Obj);
    fields.emplace_back(oblique, ValueKind::Obj);
    return result_ok(new AdtObj(
        "Asymptotes", "Asymptotes", std::move(fields)), ValueKind::Obj);
}
/** @brief Classifies continuity at a point. @param value Borrowed function. @param variable Borrowed variable name. @param point Borrowed point. @return Owning Result text or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_calculus_continuity_at_by_name(
    ExprObj* value, const char* variable, ExprObj* point) {
    if (!variable || variable[0] == '\0')
        return result_error(MathErrorCode::InvalidArgument, __func__, "calculus.continuity_at: empty variable");
    std::string error; const auto* function = checked_expr(value, error);
    const auto* target = checked_expr(point, error);
    if (!function || !target) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    try {
        const auto type = lamina::continuity_at(*function, variable, *target);
        const char* name = type == lamina::ContinuityType::Continuous ? "continuous"
            : type == lamina::ContinuityType::Removable ? "removable"
            : type == lamina::ContinuityType::Jump ? "jump" : "essential";
        return result_ok(new StringObj(name), ValueKind::Obj);
    } catch (const std::exception& exception) {
        return result_error(MathErrorCode::InvalidArgument, __func__, 
            std::string("calculus.continuity_at: ") + exception.what());
    }
}
/** @brief Finds unordered inflection points. @param value Borrowed function. @param variable Borrowed variable name. @return Owning Result set or error. @ownership Input borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_calculus_inflection_points_by_name(
    ExprObj* value, const char* variable) {
    if (!variable || variable[0] == '\0')
        return result_error(MathErrorCode::InvalidArgument, __func__, "calculus.inflection_points: empty variable");
    std::string error; const auto* function = checked_expr(value, error);
    if (!function) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    const auto result = lamina::inflection_points_checked(*function, variable);
    if (!result) return result_error(result.error());
    return math_internal::unordered_expr_result(result.value());
}
/** @brief Computes explicit-curve curvature. @param value Borrowed function. @param variable Borrowed variable name. @return Owning Result Expr or error. @ownership Input borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_calculus_curvature_by_name(
    ExprObj* value, const char* variable) {
    if (!variable || variable[0] == '\0')
        return result_error(MathErrorCode::InvalidArgument, __func__, "calculus.curvature: empty variable");
    std::string error; const auto* function = checked_expr(value, error);
    if (!function) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return checked_expr_result(lamina::curvature_checked(*function, variable));
}
/** @brief Computes parametric curvature. @param x Borrowed x component. @param y Borrowed y component. @param variable Borrowed parameter name. @return Owning Result Expr or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_calculus_curvature_parametric_by_name(
    ExprObj* x, ExprObj* y, const char* variable) {
    if (!variable || variable[0] == '\0')
        return result_error(MathErrorCode::InvalidArgument, __func__, "calculus.curvature_parametric: empty variable");
    std::string error; const auto* xv = checked_expr(x, error);
    const auto* yv = checked_expr(y, error);
    if (!xv || !yv) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return checked_expr_result(
        lamina::curvature_parametric_checked(*xv, *yv, variable));
}
/** @brief Computes surface area about x axis. @param value Borrowed function. @param variable Borrowed variable name. @param lower Borrowed lower bound. @param upper Borrowed upper bound. @return Owning Result Expr or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_calculus_surface_area_x_by_name(
    ExprObj* value, const char* variable, ExprObj* lower, ExprObj* upper) {
    if (!variable || variable[0] == '\0')
        return result_error(MathErrorCode::InvalidArgument, __func__, "calculus.surface_area_revolution_x: empty variable");
    std::string error; const auto* f = checked_expr(value, error);
    const auto* a = checked_expr(lower, error); const auto* b = checked_expr(upper, error);
    if (!f || !a || !b) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return checked_expr_result(
        lamina::surface_area_revolution_x_checked(*f, variable, *a, *b));
}
/** @brief Computes surface area about y axis. @param value Borrowed function. @param variable Borrowed variable name. @param lower Borrowed lower bound. @param upper Borrowed upper bound. @return Owning Result Expr or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_calculus_surface_area_y_by_name(
    ExprObj* value, const char* variable, ExprObj* lower, ExprObj* upper) {
    if (!variable || variable[0] == '\0')
        return result_error(MathErrorCode::InvalidArgument, __func__, "calculus.surface_area_revolution_y: empty variable");
    std::string error; const auto* f = checked_expr(value, error);
    const auto* a = checked_expr(lower, error); const auto* b = checked_expr(upper, error);
    if (!f || !a || !b) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return checked_expr_result(
        lamina::surface_area_revolution_y_checked(*f, variable, *a, *b));
}
/** @brief Computes volume about y axis. @param value Borrowed curve. @param lower Borrowed lower bound. @param upper Borrowed upper bound. @return Owning Result Expr or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_calculus_volume_y(
    ExprObj* value, ExprObj* lower, ExprObj* upper) {
    std::string error; const auto* f = checked_expr(value, error);
    const auto* a = checked_expr(lower, error); const auto* b = checked_expr(upper, error);
    if (!f || !a || !b) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return checked_expr_result(lamina::volume_of_revolution_y_checked(*f, *a, *b));
}
/** @brief Computes arc length in y representation. @param value Borrowed curve. @param lower Borrowed lower bound. @param upper Borrowed upper bound. @return Owning Result Expr or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_calculus_arc_length_y(
    ExprObj* value, ExprObj* lower, ExprObj* upper) {
    std::string error; const auto* f = checked_expr(value, error);
    const auto* a = checked_expr(lower, error); const auto* b = checked_expr(upper, error);
    if (!f || !a || !b) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return checked_expr_result(lamina::arc_length_y_checked(*f, *a, *b));
}
/** @brief Computes an iterated integral. @param value Borrowed integrand. @param variables Borrowed variable-name array. @param lowers Borrowed lower-bound array. @param uppers Borrowed upper-bound array. @return Owning Result Expr or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_calculus_integrate_multiple_by_names(
    ExprObj* value, ArrayObj* variables, ArrayObj* lowers, ArrayObj* uppers) {
    std::string error; const auto* integrand = checked_expr(value, error);
    std::vector<std::string> names; std::vector<lamina::lsr::ExprPtr> a, b;
    if (!integrand || !array_strings(variables, names, error) ||
        !array_expressions(lowers, a, error) || !array_expressions(uppers, b, error) ||
        names.size() != a.size() || names.size() != b.size())
        return result_error(MathErrorCode::InvalidArgument, __func__, "calculus.integrate_multiple: invalid arrays");
    std::vector<lamina::IntegrationStep> steps;
    for (std::size_t index = 0; index < names.size(); ++index)
        steps.push_back({names[index], a[index], b[index]});
    try {
        lamina::Integrator integrator;
        lamina::ComputationContext context;
        const auto result = lamina::integrate_multiple_checked(
            **integrand, steps, integrator, context);
        if (!result) return result_error(result.error());
        return result_ok(new ExprObj(
            std::make_shared<SymbolicExpr>(result.value())), ValueKind::Expr);
    } catch (const std::exception& exception) {
        return result_error(MathErrorCode::InvalidArgument, __func__, 
            std::string("calculus.integrate_multiple: ") + exception.what());
    }
}

/** @brief Computes asymptotes using a symbol variable. @param value Borrowed function. @param variable Borrowed symbol. @return Owning Result Asymptotes or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_calculus_asymptotes_by_symbol(ExprObj* value, ExprObj* variable) {
    std::string name, error; if (!checked_symbol_name(variable, name, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return lmx_computer_algebra_calculus_asymptotes_by_name(value, name.c_str());
}
/** @brief Classifies continuity using a symbol variable. @param value Borrowed function. @param variable Borrowed symbol. @param point Borrowed point. @return Owning Result text or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_calculus_continuity_at_by_symbol(ExprObj* value, ExprObj* variable, ExprObj* point) {
    std::string name, error; if (!checked_symbol_name(variable, name, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return lmx_computer_algebra_calculus_continuity_at_by_name(value, name.c_str(), point);
}
/** @brief Finds inflection points using a symbol variable. @param value Borrowed function. @param variable Borrowed symbol. @return Owning Result set or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_calculus_inflection_points_by_symbol(ExprObj* value, ExprObj* variable) {
    std::string name, error; if (!checked_symbol_name(variable, name, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return lmx_computer_algebra_calculus_inflection_points_by_name(value, name.c_str());
}
/** @brief Computes curvature using a symbol variable. @param value Borrowed function. @param variable Borrowed symbol. @return Owning Result Expr or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_calculus_curvature_by_symbol(ExprObj* value, ExprObj* variable) {
    std::string name, error; if (!checked_symbol_name(variable, name, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return lmx_computer_algebra_calculus_curvature_by_name(value, name.c_str());
}
/** @brief Computes parametric curvature using a symbol. @param x Borrowed x component. @param y Borrowed y component. @param variable Borrowed symbol. @return Owning Result Expr or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_calculus_curvature_parametric_by_symbol(ExprObj* x, ExprObj* y, ExprObj* variable) {
    std::string name, error; if (!checked_symbol_name(variable, name, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return lmx_computer_algebra_calculus_curvature_parametric_by_name(x, y, name.c_str());
}
/** @brief Computes x-axis surface area using a symbol. @param value Borrowed function. @param variable Borrowed symbol. @param lower Borrowed bound. @param upper Borrowed bound. @return Owning Result Expr or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_calculus_surface_area_x_by_symbol(ExprObj* value, ExprObj* variable, ExprObj* lower, ExprObj* upper) {
    std::string name, error; if (!checked_symbol_name(variable, name, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return lmx_computer_algebra_calculus_surface_area_x_by_name(value, name.c_str(), lower, upper);
}
/** @brief Computes y-axis surface area using a symbol. @param value Borrowed function. @param variable Borrowed symbol. @param lower Borrowed bound. @param upper Borrowed bound. @return Owning Result Expr or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_calculus_surface_area_y_by_symbol(ExprObj* value, ExprObj* variable, ExprObj* lower, ExprObj* upper) {
    std::string name, error; if (!checked_symbol_name(variable, name, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return lmx_computer_algebra_calculus_surface_area_y_by_name(value, name.c_str(), lower, upper);
}
/** @brief Computes an iterated integral using symbol variables. @param value Borrowed integrand. @param variables Borrowed symbol array. @param lowers Borrowed bounds. @param uppers Borrowed bounds. @return Owning Result Expr or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_calculus_integrate_multiple_by_symbols(ExprObj* value, ArrayObj* variables, ArrayObj* lowers, ArrayObj* uppers) {
    std::string error; auto* names = math_internal::symbol_text_array(variables, error);
    if (!names) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    auto* result = lmx_computer_algebra_calculus_integrate_multiple_by_names(value, names, lowers, uppers);
    names->release(); return result;
}
/** @brief Builds a Fourier series using a symbol. @param value Borrowed function. @param variable Borrowed symbol. @param period Borrowed period. @param terms Term count. @return Owning Result Expr or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_series_fourier_by_symbol(ExprObj* value, ExprObj* variable, ExprObj* period, LmInt terms) {
    std::string name, error; if (!checked_symbol_name(variable, name, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return lmx_computer_algebra_series_fourier_by_name(value, name.c_str(), period, terms);
}
/** @brief Computes convergence radius using a symbol. @param coefficients Borrowed coefficients. @param variable Borrowed symbol. @return Owning Result Expr or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_series_convergence_radius_by_symbol(ArrayObj* coefficients, ExprObj* variable) {
    std::string name, error; if (!checked_symbol_name(variable, name, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return lmx_computer_algebra_series_convergence_radius_by_name(coefficients, name.c_str());
}
/** @brief Computes convergence classification using a symbol. @param value Borrowed term. @param variable Borrowed symbol. @return Owning Result ConvergenceInfo or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_series_convergence_test_by_symbol(ExprObj* value, ExprObj* variable) {
    std::string name, error; if (!checked_symbol_name(variable, name, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return lmx_computer_algebra_series_convergence_test_by_name(value, name.c_str());
}
/** @brief Computes lim inf using a symbol. @param value Borrowed term. @param variable Borrowed symbol. @return Owning Result Expr or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_series_limit_inferior_by_symbol(ExprObj* value, ExprObj* variable) {
    std::string name, error; if (!checked_symbol_name(variable, name, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return lmx_computer_algebra_series_limit_inferior_by_name(value, name.c_str());
}
/** @brief Computes lim sup using a symbol. @param value Borrowed term. @param variable Borrowed symbol. @return Owning Result Expr or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_series_limit_superior_by_symbol(ExprObj* value, ExprObj* variable) {
    std::string name, error; if (!checked_symbol_name(variable, name, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return lmx_computer_algebra_series_limit_superior_by_name(value, name.c_str());
}
