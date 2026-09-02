#include "bridge/result.hpp"
#include "bridge/conversions.hpp"
#include "bridge/runtime_views.hpp"
#include "bridge/unit_bridge.hpp"
#include <cstdarg>
#include "bridge/math_internal.hpp"
#include "symbolic_ode_engine.hpp"
#include "symbolic_ode.hpp"
#include <array>

using namespace lmx::bridge;

namespace {
AdtObj* ode_solution_result(const lamina::ODESolutionResult& result) {
    if (!result) return result_error(result.error());
    if (!result.value().general_solution)
        return result_error(MathErrorCode::Inconclusive, __func__, "CasError(Inconclusive in ode: null solution)");
    return result_ok(
        new ExprObj(result.value().general_solution), ValueKind::Expr);
}

const char* ode_type_name(lamina::ODEType type) {
    switch (type) {
    case lamina::ODEType::Separable: return "separable";
    case lamina::ODEType::Linear1: return "linear1";
    case lamina::ODEType::Linear2_ConstCoeff: return "linear2_const_coeff";
    case lamina::ODEType::Homogeneous: return "homogeneous";
    case lamina::ODEType::Bernoulli: return "bernoulli";
    case lamina::ODEType::Exact: return "exact";
    case lamina::ODEType::HigherOrder_ConstCoeff: return "higher_order_const_coeff";
    case lamina::ODEType::Euler: return "euler";
    case lamina::ODEType::System: return "system";
    case lamina::ODEType::LaplaceMethod: return "laplace";
    case lamina::ODEType::Frobenius: return "frobenius";
    default: return "unknown";
    }
}
} // namespace

/** @brief Solves a Bernoulli ODE. @param p Borrowed P coefficient. @param q Borrowed Q coefficient. @param power Bernoulli power. @param independent Borrowed independent name. @param dependent Borrowed dependent name. @return Owning Result Expr or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_ordinary_differential_equations_solve_bernoulli_by_names(
    ExprObj* p, ExprObj* q, LmInt power,
    const char* independent, const char* dependent) noexcept try {
    ensure_lmmc_runtime();
    if (!independent || !dependent || power < std::numeric_limits<int>::min() ||
        power > std::numeric_limits<int>::max())
        return result_error(MathErrorCode::InvalidArgument, __func__, "ode.solve_bernoulli: invalid argument");
    std::string error; const auto* pv = checked_expr(p, error);
    const auto* qv = checked_expr(q, error);
    if (!pv || !qv) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return ode_solution_result(lamina::solve_bernoulli_ode_checked(
        *pv, *qv, static_cast<int>(power), independent, dependent));
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Solves an exact ODE. @param m Borrowed M coefficient. @param n Borrowed N coefficient. @param independent Borrowed independent name. @param dependent Borrowed dependent name. @return Owning Result Expr or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_ordinary_differential_equations_solve_exact_by_names(
    ExprObj* m, ExprObj* n, const char* independent, const char* dependent) noexcept try {
    ensure_lmmc_runtime();
    if (!independent || !dependent)
        return result_error(MathErrorCode::InvalidArgument, __func__, "ode.solve_exact: invalid variable");
    std::string error; const auto* mv = checked_expr(m, error);
    const auto* nv = checked_expr(n, error);
    if (!mv || !nv) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return ode_solution_result(lamina::solve_exact_ode_checked(
        *mv, *nv, independent, dependent));
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Solves a homogeneous first-order ODE. @param rhs Borrowed RHS. @param independent Borrowed independent name. @param dependent Borrowed dependent name. @return Owning Result Expr or error. @ownership Input borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_ordinary_differential_equations_solve_homogeneous_by_names(
    ExprObj* rhs, const char* independent, const char* dependent) noexcept try {
    ensure_lmmc_runtime();
    if (!independent || !dependent)
        return result_error(MathErrorCode::InvalidArgument, __func__, "ode.solve_homogeneous: invalid variable");
    std::string error; const auto* value = checked_expr(rhs, error);
    if (!value) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return ode_solution_result(lamina::solve_homogeneous_ode_checked(
        *value, independent, dependent));
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Solves a higher-order constant-coefficient ODE. @param coefficients Borrowed numeric coefficients. @param forcing Borrowed forcing expression. @param independent Borrowed independent name. @param dependent Borrowed dependent name. @return Owning Result Expr or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_ordinary_differential_equations_solve_higher_order_by_names(
    ArrayObj* coefficients, ExprObj* forcing,
    const char* independent, const char* dependent) noexcept try {
    ensure_lmmc_runtime();
    if (!independent || !dependent)
        return result_error(MathErrorCode::InvalidArgument, __func__, "ode.solve_higher_order: invalid variable");
    std::vector<double> values; std::string error;
    const auto* force = checked_expr(forcing, error);
    if (!force || !array_numbers(coefficients, values, error) ||
        values.empty() || values.size() > 7)
        return result_error(MathErrorCode::InvalidArgument, __func__, "ode.solve_higher_order: " + error);
    return ode_solution_result(lamina::solve_higher_order_ode_checked(
        values, *force, independent, dependent));
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Solves a Cauchy-Euler ODE. @param coefficients Borrowed numeric coefficients. @param forcing Borrowed forcing expression. @param independent Borrowed independent name. @param dependent Borrowed dependent name. @return Owning Result Expr or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_ordinary_differential_equations_solve_euler_by_names(
    ArrayObj* coefficients, ExprObj* forcing,
    const char* independent, const char* dependent) noexcept try {
    ensure_lmmc_runtime();
    if (!independent || !dependent)
        return result_error(MathErrorCode::InvalidArgument, __func__, "ode.solve_euler: invalid variable");
    std::vector<double> values; std::string error;
    const auto* force = checked_expr(forcing, error);
    if (!force || !array_numbers(coefficients, values, error) ||
        values.size() < 3 || values.size() > 4)
        return result_error(MathErrorCode::InvalidArgument, __func__, "ode.solve_euler: invalid coefficients");
    return ode_solution_result(lamina::solve_euler_ode_checked(
        values, *force, independent, dependent));
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Solves by variation of parameters. @param y1 Borrowed homogeneous solution. @param y2 Borrowed homogeneous solution. @param forcing Borrowed forcing expression. @param independent Borrowed variable name. @return Owning Result Expr or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_ordinary_differential_equations_solve_variation_of_parameters_by_name(
    ExprObj* y1, ExprObj* y2, ExprObj* forcing, const char* independent) noexcept try {
    ensure_lmmc_runtime();
    if (!independent)
        return result_error(MathErrorCode::InvalidArgument, __func__, "ode.solve_variation_of_parameters: invalid variable");
    std::string error; const auto* a = checked_expr(y1, error);
    const auto* b = checked_expr(y2, error); const auto* g = checked_expr(forcing, error);
    if (!a || !b || !g) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return ode_solution_result(lamina::solve_variation_of_parameters_checked(
        *a, *b, *g, independent));
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Computes a Frobenius series solution. @param p Borrowed P coefficient. @param q Borrowed Q coefficient. @param point Borrowed expansion point. @param independent Borrowed variable name. @param order Positive truncation order. @return Owning Result FrobeniusSolution or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_ordinary_differential_equations_solve_frobenius_by_name(
    ExprObj* p, ExprObj* q, ExprObj* point,
    const char* independent, LmInt order) noexcept try {
    ensure_lmmc_runtime();
    if (!independent || order <= 0 || order > std::numeric_limits<int>::max())
        return result_error(MathErrorCode::InvalidArgument, __func__, "ode.solve_frobenius: invalid argument");
    std::string error; const auto* pv = checked_expr(p, error);
    const auto* qv = checked_expr(q, error); const auto* x0 = checked_expr(point, error);
    if (!pv || !qv || !x0) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    const auto result = lamina::solve_frobenius_checked(
        *pv, *qv, *x0, independent, static_cast<int>(order));
    if (!result) return result_error(result.error());
    if (!result.value().series_solution)
        return result_error(MathErrorCode::InvalidArgument, __func__, "ode.solve_frobenius: null series");
    const char* point_type =
        result.value().point_type == lamina::ODESingularityType::Ordinary
        ? "ordinary" : result.value().point_type ==
        lamina::ODESingularityType::RegularSingular
        ? "regular_singular" : "irregular_singular";
    auto roots = make_owned_object<ArrayObj>();
    for (double root : result.value().indicial_roots)
        roots->append(Value(root));
    std::vector<Value> fields;
    fields.emplace_back(take_object_value(
        make_owned_object<ExprObj>(result.value().series_solution),
        ValueKind::Expr));
    fields.emplace_back(take_object_value(
        make_owned_object<StringObj>(point_type), ValueKind::Obj));
    fields.emplace_back(take_object_value(std::move(roots), ValueKind::Obj));
    fields.emplace_back(static_cast<LmInt>(result.value().truncation_order));
    return result_ok(new AdtObj(
        "FrobeniusSolution", "FrobeniusSolution", std::move(fields)),
        ValueKind::Obj);
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Classifies a first-order ODE. @param rhs Borrowed RHS. @param independent Borrowed independent name. @param dependent Borrowed dependent name. @return Owning Result text or error. @ownership Input borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_ordinary_differential_equations_classify_first_order_by_names(
    ExprObj* rhs, const char* independent, const char* dependent) noexcept try {
    ensure_lmmc_runtime();
    if (!independent || !dependent)
        return result_error(MathErrorCode::InvalidArgument, __func__, "ode.classify_first_order: invalid variable");
    std::string error; const auto* value = checked_expr(rhs, error);
    if (!value) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return result_ok(new StringObj(ode_type_name(
        lamina::classify_first_order_ode(*value, independent, dependent).type)),
        ValueKind::Obj);
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Classifies a higher-order ODE. @param coefficients Borrowed expression coefficients. @param forcing Borrowed forcing expression. @param independent Borrowed independent name. @param dependent Borrowed dependent name. @return Owning Result text or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_ordinary_differential_equations_classify_higher_order_by_names(
    ArrayObj* coefficients, ExprObj* forcing,
    const char* independent, const char* dependent) noexcept try {
    ensure_lmmc_runtime();
    if (!independent || !dependent)
        return result_error(MathErrorCode::InvalidArgument, __func__, "ode.classify_higher_order: invalid variable");
    std::vector<lamina::lsr::ExprPtr> values; std::string error;
    const auto* force = checked_expr(forcing, error);
    if (!force || !array_expressions(coefficients, values, error) || values.empty())
        return result_error(MathErrorCode::InvalidArgument, __func__, "ode.classify_higher_order: " + error);
    return result_ok(new StringObj(ode_type_name(
        lamina::classify_higher_order_ode(
            values, *force, independent, dependent).type)), ValueKind::Obj);
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Solves Bernoulli ODE using symbols. @param p Borrowed P. @param q Borrowed Q. @param power Power. @param x Borrowed independent symbol. @param y Borrowed dependent symbol. @return Owning Result Expr or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_ordinary_differential_equations_solve_bernoulli_by_symbols(ExprObj* p, ExprObj* q, LmInt power, ExprObj* x, ExprObj* y) noexcept try {
    ensure_lmmc_runtime();
    std::string a, b, error; if (!checked_symbol_name(x, a, error) || !checked_symbol_name(y, b, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return lmx_computer_algebra_ordinary_differential_equations_solve_bernoulli_by_names(p, q, power, a.c_str(), b.c_str());
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Solves exact ODE using symbols. @param m Borrowed M. @param n Borrowed N. @param x Borrowed independent symbol. @param y Borrowed dependent symbol. @return Owning Result Expr or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_ordinary_differential_equations_solve_exact_by_symbols(ExprObj* m, ExprObj* n, ExprObj* x, ExprObj* y) noexcept try {
    ensure_lmmc_runtime();
    std::string a, b, error; if (!checked_symbol_name(x, a, error) || !checked_symbol_name(y, b, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return lmx_computer_algebra_ordinary_differential_equations_solve_exact_by_names(m, n, a.c_str(), b.c_str());
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Solves homogeneous ODE using symbols. @param rhs Borrowed RHS. @param x Borrowed independent symbol. @param y Borrowed dependent symbol. @return Owning Result Expr or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_ordinary_differential_equations_solve_homogeneous_by_symbols(ExprObj* rhs, ExprObj* x, ExprObj* y) noexcept try {
    ensure_lmmc_runtime();
    std::string a, b, error; if (!checked_symbol_name(x, a, error) || !checked_symbol_name(y, b, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return lmx_computer_algebra_ordinary_differential_equations_solve_homogeneous_by_names(rhs, a.c_str(), b.c_str());
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Solves higher-order ODE using symbols. @param coefficients Borrowed coefficients. @param forcing Borrowed forcing. @param x Borrowed independent symbol. @param y Borrowed dependent symbol. @return Owning Result Expr or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_ordinary_differential_equations_solve_higher_order_by_symbols(ArrayObj* coefficients, ExprObj* forcing, ExprObj* x, ExprObj* y) noexcept try {
    ensure_lmmc_runtime();
    std::string a, b, error; if (!checked_symbol_name(x, a, error) || !checked_symbol_name(y, b, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return lmx_computer_algebra_ordinary_differential_equations_solve_higher_order_by_names(coefficients, forcing, a.c_str(), b.c_str());
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Solves Euler ODE using symbols. @param coefficients Borrowed coefficients. @param forcing Borrowed forcing. @param x Borrowed independent symbol. @param y Borrowed dependent symbol. @return Owning Result Expr or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_ordinary_differential_equations_solve_euler_by_symbols(ArrayObj* coefficients, ExprObj* forcing, ExprObj* x, ExprObj* y) noexcept try {
    ensure_lmmc_runtime();
    std::string a, b, error; if (!checked_symbol_name(x, a, error) || !checked_symbol_name(y, b, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return lmx_computer_algebra_ordinary_differential_equations_solve_euler_by_names(coefficients, forcing, a.c_str(), b.c_str());
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Solves variation of parameters using a symbol. @param y1 Borrowed solution. @param y2 Borrowed solution. @param forcing Borrowed forcing. @param x Borrowed independent symbol. @return Owning Result Expr or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_ordinary_differential_equations_solve_variation_of_parameters_by_symbol(ExprObj* y1, ExprObj* y2, ExprObj* forcing, ExprObj* x) noexcept try {
    ensure_lmmc_runtime();
    std::string name, error; if (!checked_symbol_name(x, name, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return lmx_computer_algebra_ordinary_differential_equations_solve_variation_of_parameters_by_name(y1, y2, forcing, name.c_str());
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Solves Frobenius series using a symbol. @param p Borrowed P. @param q Borrowed Q. @param point Borrowed point. @param x Borrowed independent symbol. @param order Order. @return Owning Result FrobeniusSolution or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_ordinary_differential_equations_solve_frobenius_by_symbol(ExprObj* p, ExprObj* q, ExprObj* point, ExprObj* x, LmInt order) noexcept try {
    ensure_lmmc_runtime();
    std::string name, error; if (!checked_symbol_name(x, name, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return lmx_computer_algebra_ordinary_differential_equations_solve_frobenius_by_name(p, q, point, name.c_str(), order);
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Classifies first-order ODE using symbols. @param rhs Borrowed RHS. @param x Borrowed independent symbol. @param y Borrowed dependent symbol. @return Owning Result text or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_ordinary_differential_equations_classify_first_order_by_symbols(ExprObj* rhs, ExprObj* x, ExprObj* y) noexcept try {
    ensure_lmmc_runtime();
    std::string a, b, error; if (!checked_symbol_name(x, a, error) || !checked_symbol_name(y, b, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return lmx_computer_algebra_ordinary_differential_equations_classify_first_order_by_names(rhs, a.c_str(), b.c_str());
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Classifies higher-order ODE using symbols. @param coefficients Borrowed coefficients. @param forcing Borrowed forcing. @param x Borrowed independent symbol. @param y Borrowed dependent symbol. @return Owning Result text or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_ordinary_differential_equations_classify_higher_order_by_symbols(ArrayObj* coefficients, ExprObj* forcing, ExprObj* x, ExprObj* y) noexcept try {
    ensure_lmmc_runtime();
    std::string a, b, error; if (!checked_symbol_name(x, a, error) || !checked_symbol_name(y, b, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return lmx_computer_algebra_ordinary_differential_equations_classify_higher_order_by_names(coefficients, forcing, a.c_str(), b.c_str());
} catch (...) {
    return c_abi_current_exception(__func__);
}

using lmx::bridge::math_internal::checked_expression_operation;

extern "C" LM_API AdtObj* lmx_computer_algebra_ordinary_differential_equations_solve_separable_by_names(
    ExprObj* rhs, const char* independent, const char* dependent) noexcept try {
    ensure_lmmc_runtime();
    if (!independent || !dependent)
        return result_error(MathErrorCode::InvalidArgument, __func__, "ordinary_differential_equations.solve_separable: invalid variable");
    return checked_expression_operation("ordinary_differential_equations.solve_separable", rhs,
        [&](const auto& expression) {
            return lamina::solve_separable_ode(
                expression, independent, dependent);
        });
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_computer_algebra_ordinary_differential_equations_solve_first_order_linear_by_names(
    ExprObj* coefficient, ExprObj* forcing,
    const char* independent, const char* dependent) noexcept try {
    ensure_lmmc_runtime();
    if (!independent || !dependent)
        return result_error(MathErrorCode::InvalidArgument, __func__, "ordinary_differential_equations.solve_first_order_linear: invalid variable");
    std::string error;
    const auto* checked_coefficient = checked_expr(coefficient, error);
    const auto* checked_forcing = checked_expr(forcing, error);
    if (!checked_coefficient || !checked_forcing)
        return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    auto result = lamina::solve_linear1_ode(
        *checked_coefficient, *checked_forcing,
        independent, dependent);
    if (!result)
        return result_error(MathErrorCode::Inconclusive, __func__, 
            "CasError(Inconclusive in ordinary_differential_equations.solve_first_order_linear)");
    result = result->simplify();
    return result_ok(new ExprObj(std::move(result)), ValueKind::Expr);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_computer_algebra_ordinary_differential_equations_solve_second_order_linear_by_names(
    const double a, const double b, const double c, ExprObj* forcing,
    const char* independent, const char* dependent) noexcept try {
    ensure_lmmc_runtime();
    if (!independent || !dependent)
        return result_error(MathErrorCode::InvalidArgument, __func__, "ordinary_differential_equations.solve_second_order_linear: invalid variable");
    std::string error;
    const auto* checked_forcing = checked_expr(forcing, error);
    if (!checked_forcing) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    const auto result = lamina::solve_linear2_ode_checked(
        a, b, c, *checked_forcing, independent, dependent);
    if (!result) return result_error(result.error());
    return result_ok(new ExprObj(result.value()), ValueKind::Expr);
} catch (...) {
    return c_abi_current_exception(__func__);
}

/** @brief Symbol-argument separable ODE solve. @param r Borrowed RHS. @param x Borrowed independent symbol. @param y Borrowed dependent symbol. @return Owning Result Expr or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_ordinary_differential_equations_solve_separable_by_symbols(ExprObj* r, ExprObj* x, ExprObj* y) noexcept try {
    ensure_lmmc_runtime();
    std::string a, b, error;
    if (!checked_symbol_name(x, a, error) || !checked_symbol_name(y, b, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return lmx_computer_algebra_ordinary_differential_equations_solve_separable_by_names(r, a.c_str(), b.c_str());
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Symbol-argument first-order linear ODE solve. @param c Borrowed coefficient. @param f Borrowed forcing. @param x Borrowed independent symbol. @param y Borrowed dependent symbol. @return Owning Result Expr or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_ordinary_differential_equations_solve_first_order_linear_by_symbols(ExprObj* c, ExprObj* f, ExprObj* x, ExprObj* y) noexcept try {
    ensure_lmmc_runtime();
    std::string a, b, error;
    if (!checked_symbol_name(x, a, error) || !checked_symbol_name(y, b, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return lmx_computer_algebra_ordinary_differential_equations_solve_first_order_linear_by_names(c, f, a.c_str(), b.c_str());
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Symbol-argument second-order linear ODE solve. @param a Coefficient. @param b Coefficient. @param c Coefficient. @param f Borrowed forcing. @param x Borrowed independent symbol. @param y Borrowed dependent symbol. @return Owning Result Expr or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_ordinary_differential_equations_solve_second_order_linear_by_symbols(double a, double b, double c, ExprObj* f, ExprObj* x, ExprObj* y) noexcept try {
    ensure_lmmc_runtime();
    std::string independent, dependent, error;
    if (!checked_symbol_name(x, independent, error) || !checked_symbol_name(y, dependent, error)) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return lmx_computer_algebra_ordinary_differential_equations_solve_second_order_linear_by_names(a, b, c, f, independent.c_str(), dependent.c_str());
} catch (...) {
    return c_abi_current_exception(__func__);
}
