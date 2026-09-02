#include "bridge/result.hpp"
#include "bridge/conversions.hpp"
#include "bridge/runtime_views.hpp"
#include "bridge/unit_bridge.hpp"
#include <cstdarg>
#include "bridge/math_internal.hpp"
#include "symbolic_vector_geometry.hpp"
#include "vector_calculus.hpp"
#include "differential_geometry.hpp"

using namespace lmx::bridge;

namespace {
bool checked_bounds(
    ArrayObj* values,
    std::pair<lamina::lsr::ExprPtr, lamina::lsr::ExprPtr>& output,
    std::string& error) {
    std::vector<lamina::lsr::ExprPtr> expressions;
    if (!array_expressions(values, expressions, error) || expressions.size() != 2) {
        error = "bounds must contain exactly two expressions";
        return false;
    }
    output = {expressions[0], expressions[1]};
    return true;
}

AdtObj* line_result(const lamina::LineSymbolicResult& line) {
    if (!line) return result_error(line.error());
    auto point = make_owned_object<ArrayObj>();
    auto direction = make_owned_object<ArrayObj>();
    for (const auto& value : line.value().point) {
        point->append(take_object_value(
            make_owned_object<ExprObj>(value), ValueKind::Expr));
    }
    for (const auto& value : line.value().direction) {
        direction->append(take_object_value(
            make_owned_object<ExprObj>(value), ValueKind::Expr));
    }
    std::vector<Value> fields;
    fields.emplace_back(take_object_value(std::move(point), ValueKind::Obj));
    fields.emplace_back(take_object_value(std::move(direction), ValueKind::Obj));
    return result_ok(new AdtObj("Line", "Line", std::move(fields)), ValueKind::Obj);
}

AdtObj* plane_result(const lamina::PlaneSymbolicResult& plane) {
    if (!plane) return result_error(plane.error());
    if (!plane.value().d)
        return result_error(MathErrorCode::InvalidArgument, __func__, "geometry: null plane offset");
    auto normal = make_owned_object<ArrayObj>();
    for (const auto& value : plane.value().normal) {
        normal->append(take_object_value(
            make_owned_object<ExprObj>(value), ValueKind::Expr));
    }
    std::vector<Value> fields;
    fields.emplace_back(take_object_value(std::move(normal), ValueKind::Obj));
    fields.emplace_back(take_object_value(
        make_owned_object<ExprObj>(plane.value().d), ValueKind::Expr));
    return result_ok(new AdtObj("Plane", "Plane", std::move(fields)), ValueKind::Obj);
}

const char* classification_name(
    lamina::CriticalPointClassification classification) {
    using Classification = lamina::CriticalPointClassification;
    switch (classification) {
        case Classification::LocalMinimum: return "minimum";
        case Classification::LocalMaximum: return "maximum";
        case Classification::Saddle: return "saddle";
        case Classification::Degenerate: return "degenerate";
        case Classification::Inconclusive: return "inconclusive";
    }
    return "inconclusive";
}

} // namespace

/** @brief Computes vector angle. @param lhs Borrowed vector. @param rhs Borrowed vector. @return Owning Result real or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_geometry_vector_angle(ArrayObj* lhs, ArrayObj* rhs) noexcept try {
    ensure_lmmc_runtime();
    std::vector<lamina::lsr::ExprPtr> a, b; std::string error;
    if (!array_expressions(lhs, a, error) || !array_expressions(rhs, b, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "geometry.vector_angle: " + error);
    const auto result = lamina::vector_angle_checked(a, b);
    if (!result) return result_error(result.error());
    return result_ok(result.value());
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Computes point-plane distance. @param point Borrowed point. @param normal Borrowed plane normal. @param offset Borrowed plane offset. @return Owning Result Expr or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_geometry_point_plane_distance(ArrayObj* point, ArrayObj* normal, ExprObj* offset) noexcept try {
    ensure_lmmc_runtime();
    std::vector<lamina::lsr::ExprPtr> p, n; std::string error;
    const auto* d = checked_expr(offset, error);
    if (!d || !array_expressions(point, p, error) || !array_expressions(normal, n, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "geometry.point_plane_distance: " + error);
    return math_internal::checked_expr_result(lamina::point_plane_distance_checked(p, {n, *d}));
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Constructs a symbolic line from two points. @param lhs Borrowed point. @param rhs Borrowed point. @return Owning Result Line or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_geometry_line_from_two_points(ArrayObj* lhs, ArrayObj* rhs) noexcept try {
    ensure_lmmc_runtime();
    std::vector<lamina::lsr::ExprPtr> a, b; std::string error;
    if (!array_expressions(lhs, a, error) || !array_expressions(rhs, b, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "geometry.line_from_two_points: " + error);
    return line_result(lamina::line_from_two_points_checked(a, b));
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Constructs a symbolic plane from three points. @param a Borrowed point. @param b Borrowed point. @param c Borrowed point. @return Owning Result Plane or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_geometry_plane_from_three_points(ArrayObj* a, ArrayObj* b, ArrayObj* c) noexcept try {
    ensure_lmmc_runtime();
    std::vector<lamina::lsr::ExprPtr> av, bv, cv; std::string error;
    if (!array_expressions(a, av, error) || !array_expressions(b, bv, error) ||
        !array_expressions(c, cv, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "geometry.plane_from_three_points: " + error);
    return plane_result(lamina::plane_from_three_points_checked(av, bv, cv));
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Computes an implicit-surface unit normal. @param surface Borrowed implicit equation. @param variables Borrowed coordinate symbols. @param point Borrowed point. @return Owning Result expression array or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_geometry_surface_normal(ExprObj* surface, ArrayObj* variables, ArrayObj* point) noexcept try {
    ensure_lmmc_runtime();
    std::string error; const auto* f = checked_expr(surface, error);
    std::vector<std::string> names; std::vector<lamina::lsr::ExprPtr> p;
    if (!f || !math_internal::checked_symbol_names(variables, names, error) ||
        !array_expressions(point, p, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "geometry.surface_normal: " + error);
    return expr_array_result(lamina::surface_normal_checked({*f, names}, p));
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Computes an implicit-surface tangent plane. @param surface Borrowed implicit equation. @param variables Borrowed coordinate symbols. @param point Borrowed point. @return Owning Result Plane or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_geometry_tangent_plane(ExprObj* surface, ArrayObj* variables, ArrayObj* point) noexcept try {
    ensure_lmmc_runtime();
    std::string error; const auto* f = checked_expr(surface, error);
    std::vector<std::string> names; std::vector<lamina::lsr::ExprPtr> p;
    if (!f || !math_internal::checked_symbol_names(variables, names, error) ||
        !array_expressions(point, p, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "geometry.tangent_plane: " + error);
    return plane_result(lamina::tangent_plane_checked({*f, names}, p));
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Computes a directional derivative. @param value Borrowed scalar field. @param variables Borrowed coordinate symbols. @param direction Borrowed direction vector. @param order Positive order. @return Owning Result Expr or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_geometry_directional_derivative(ExprObj* value, ArrayObj* variables, ArrayObj* direction, LmInt order) noexcept try {
    ensure_lmmc_runtime();
    if (order <= 0 || order > std::numeric_limits<int>::max())
        return result_error(MathErrorCode::InvalidArgument, __func__, "geometry.directional_derivative: invalid order");
    std::string error; const auto* f = checked_expr(value, error);
    std::vector<std::string> names; std::vector<lamina::lsr::ExprPtr> vector;
    if (!f || !math_internal::checked_symbol_names(variables, names, error) ||
        !array_expressions(direction, vector, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "geometry.directional_derivative: " + error);
    return math_internal::checked_expr_result(lamina::directional_derivative_checked(
        *f, names, vector, static_cast<int>(order)));
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Computes scalar curve integral. @param value Borrowed scalar field. @param path Borrowed parametrization. @param parameter Borrowed parameter symbol. @param lower Borrowed lower bound. @param upper Borrowed upper bound. @return Owning Result Expr or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_geometry_curve_integral_scalar(ExprObj* value, ArrayObj* path, ExprObj* parameter, ExprObj* lower, ExprObj* upper) noexcept try {
    ensure_lmmc_runtime();
    std::string name, error; const auto* f = checked_expr(value, error);
    const auto* a = checked_expr(lower, error); const auto* b = checked_expr(upper, error);
    std::vector<lamina::lsr::ExprPtr> r;
    if (!f || !a || !b || !checked_symbol_name(parameter, name, error) ||
        !array_expressions(path, r, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "geometry.curve_integral_scalar: " + error);
    return math_internal::checked_expr_result(lamina::curve_integral_scalar_checked(*f, r, name, *a, *b));
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Computes vector curve integral. @param field Borrowed vector field. @param path Borrowed parametrization. @param parameter Borrowed parameter symbol. @param lower Borrowed lower bound. @param upper Borrowed upper bound. @return Owning Result Expr or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_geometry_curve_integral_vector(ArrayObj* field, ArrayObj* path, ExprObj* parameter, ExprObj* lower, ExprObj* upper) noexcept try {
    ensure_lmmc_runtime();
    std::string name, error; const auto* a = checked_expr(lower, error);
    const auto* b = checked_expr(upper, error);
    std::vector<lamina::lsr::ExprPtr> f, r;
    if (!a || !b || !checked_symbol_name(parameter, name, error) ||
        !array_expressions(field, f, error) || !array_expressions(path, r, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "geometry.curve_integral_vector: " + error);
    return math_internal::checked_expr_result(lamina::curve_integral_vector_checked(f, r, name, *a, *b));
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Computes scalar surface integral. @param value Borrowed scalar field. @param path Borrowed parametrization. @param u Borrowed first parameter symbol. @param v Borrowed second parameter symbol. @param ub Borrowed u bounds. @param vb Borrowed v bounds. @return Owning Result Expr or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_geometry_surface_integral_scalar(ExprObj* value, ArrayObj* path, ExprObj* u, ExprObj* v, ArrayObj* ub, ArrayObj* vb) noexcept try {
    ensure_lmmc_runtime();
    std::string un, vn, error; const auto* f = checked_expr(value, error);
    std::vector<lamina::lsr::ExprPtr> r; std::pair<lamina::lsr::ExprPtr, lamina::lsr::ExprPtr> u_bounds, v_bounds;
    if (!f || !checked_symbol_name(u, un, error) || !checked_symbol_name(v, vn, error) ||
        !array_expressions(path, r, error) || !checked_bounds(ub, u_bounds, error) ||
        !checked_bounds(vb, v_bounds, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "geometry.surface_integral_scalar: " + error);
    return math_internal::checked_expr_result(lamina::surface_integral_scalar_checked(
        *f, r, un, vn, u_bounds.first, u_bounds.second,
        v_bounds.first, v_bounds.second));
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Computes vector surface integral. @param field Borrowed vector field. @param path Borrowed parametrization. @param u Borrowed first parameter symbol. @param v Borrowed second parameter symbol. @param ub Borrowed u bounds. @param vb Borrowed v bounds. @return Owning Result Expr or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_geometry_surface_integral_vector(ArrayObj* field, ArrayObj* path, ExprObj* u, ExprObj* v, ArrayObj* ub, ArrayObj* vb) noexcept try {
    ensure_lmmc_runtime();
    std::string un, vn, error; std::vector<lamina::lsr::ExprPtr> f, r;
    std::pair<lamina::lsr::ExprPtr, lamina::lsr::ExprPtr> u_bounds, v_bounds;
    if (!checked_symbol_name(u, un, error) || !checked_symbol_name(v, vn, error) ||
        !array_expressions(field, f, error) || !array_expressions(path, r, error) ||
        !checked_bounds(ub, u_bounds, error) || !checked_bounds(vb, v_bounds, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "geometry.surface_integral_vector: " + error);
    return math_internal::checked_expr_result(lamina::surface_integral_vector_checked(
        f, r, un, vn, u_bounds.first, u_bounds.second,
        v_bounds.first, v_bounds.second));
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Applies Green's theorem. @param p Borrowed P field. @param q Borrowed Q field. @param variables Borrowed two coordinate symbols. @param xb Borrowed x bounds. @param yb Borrowed y bounds. @return Owning Result Expr or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_geometry_greens_theorem(ExprObj* p, ExprObj* q, ArrayObj* variables, ArrayObj* xb, ArrayObj* yb) noexcept try {
    ensure_lmmc_runtime();
    std::string error; const auto* pv = checked_expr(p, error); const auto* qv = checked_expr(q, error);
    std::vector<std::string> names; std::pair<lamina::lsr::ExprPtr, lamina::lsr::ExprPtr> x_bounds, y_bounds;
    if (!pv || !qv || !math_internal::checked_symbol_names(variables, names, error) ||
        names.size() != 2 || !checked_bounds(xb, x_bounds, error) ||
        !checked_bounds(yb, y_bounds, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "geometry.greens_theorem: " + error);
    return math_internal::checked_expr_result(lamina::greens_theorem_checked(
        *pv, *qv, names, x_bounds, y_bounds));
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Applies divergence theorem. @param field Borrowed 3D field. @param variables Borrowed coordinate symbols. @param xb Borrowed x bounds. @param yb Borrowed y bounds. @param zb Borrowed z bounds. @return Owning Result Expr or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_geometry_divergence_theorem(ArrayObj* field, ArrayObj* variables, ArrayObj* xb, ArrayObj* yb, ArrayObj* zb) noexcept try {
    ensure_lmmc_runtime();
    std::string error; std::vector<lamina::lsr::ExprPtr> f; std::vector<std::string> names;
    std::pair<lamina::lsr::ExprPtr, lamina::lsr::ExprPtr> x_bounds, y_bounds, z_bounds;
    if (!array_expressions(field, f, error) || !math_internal::checked_symbol_names(variables, names, error) ||
        names.size() != 3 || !checked_bounds(xb, x_bounds, error) ||
        !checked_bounds(yb, y_bounds, error) || !checked_bounds(zb, z_bounds, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "geometry.divergence_theorem: " + error);
    return math_internal::checked_expr_result(lamina::divergence_theorem_checked(
        f, names, x_bounds, y_bounds, z_bounds));
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Applies Stokes' theorem. @param field Borrowed 3D field. @param variables Borrowed coordinate symbols. @param path Borrowed surface parametrization. @param u Borrowed parameter symbol. @param v Borrowed parameter symbol. @param ub Borrowed bounds. @param vb Borrowed bounds. @return Owning Result Expr or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_geometry_stokes_theorem(ArrayObj* field, ArrayObj* variables, ArrayObj* path, ExprObj* u, ExprObj* v, ArrayObj* ub, ArrayObj* vb) noexcept try {
    ensure_lmmc_runtime();
    std::string un, vn, error; std::vector<lamina::lsr::ExprPtr> f, r; std::vector<std::string> names;
    std::pair<lamina::lsr::ExprPtr, lamina::lsr::ExprPtr> u_bounds, v_bounds;
    if (!array_expressions(field, f, error) || !math_internal::checked_symbol_names(variables, names, error) ||
        !array_expressions(path, r, error) || !checked_symbol_name(u, un, error) ||
        !checked_symbol_name(v, vn, error) || !checked_bounds(ub, u_bounds, error) ||
        !checked_bounds(vb, v_bounds, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "geometry.stokes_theorem: " + error);
    return math_internal::checked_expr_result(lamina::stokes_theorem_checked(
        f, names, r, un, vn, u_bounds, v_bounds));
} catch (...) {
    return c_abi_current_exception(__func__);
}

/** @brief Finds classified extrema. @param value Borrowed scalar function. @param variables Borrowed coordinate symbols. @return Owning Result array of tables or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_geometry_find_extrema(ExprObj* value, ArrayObj* variables) noexcept try {
    ensure_lmmc_runtime();
    std::string error; const auto* f = checked_expr(value, error); std::vector<std::string> names;
    if (!f || !math_internal::checked_symbol_names(variables, names, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "geometry.find_extrema: " + error);
    const auto result = lamina::find_extrema_checked(*f, names);
    if (!result) return result_error(result.error());
    auto output = make_owned_object<ArrayObj>();
    for (const auto& critical : result.value()) {
        std::vector<TableObj::Entry> entries;
        for (const auto& [name, expression] : critical.point) {
            auto value = take_object_value(
                make_owned_object<ExprObj>(expression), ValueKind::Expr);
            entries.emplace_back(name, std::move(value));
        }
        auto classification = take_object_value(
            make_owned_object<StringObj>(
                classification_name(critical.classification)),
            ValueKind::Obj);
        entries.emplace_back("classification", std::move(classification));
        output->append(take_object_value(
            make_owned_object<TableObj>(std::move(entries)), ValueKind::Table));
    }
    return result_ok(output.release(), ValueKind::Obj);
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Finds Lagrange-multiplier candidates. @param value Borrowed objective. @param constraints Borrowed constraints. @param variables Borrowed coordinate symbols. @return Owning Result array of tables or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_geometry_lagrange_multipliers(ExprObj* value, ArrayObj* constraints, ArrayObj* variables) noexcept try {
    ensure_lmmc_runtime();
    std::string error; const auto* f = checked_expr(value, error);
    std::vector<lamina::lsr::ExprPtr> conditions; std::vector<std::string> names;
    if (!f || !array_expressions(constraints, conditions, error) ||
        !math_internal::checked_symbol_names(variables, names, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "geometry.lagrange_multipliers: " + error);
    const auto result = lamina::lagrange_multipliers_checked(*f, conditions, names);
    if (!result) return result_error(result.error());
    return result_ok(math_internal::solution_tables(result.value()), ValueKind::Obj);
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Computes first-kind Christoffel symbol. @param metric Borrowed metric. @param variables Borrowed coordinate symbols. @param k Index. @param i Index. @param j Index. @return Owning Result Expr or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_geometry_christoffel_first(ExprObj* metric, ArrayObj* variables, LmInt k, LmInt i, LmInt j) noexcept try {
    ensure_lmmc_runtime();
    if (k < 0 || i < 0 || j < 0 || k > std::numeric_limits<int>::max() ||
        i > std::numeric_limits<int>::max() || j > std::numeric_limits<int>::max())
        return result_error(MathErrorCode::InvalidArgument, __func__, "geometry.christoffel_first_kind: invalid index");
    std::string error; const auto* g = checked_expr(metric, error); std::vector<std::string> names;
    if (!g || !math_internal::checked_symbol_names(variables, names, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "geometry.christoffel_first_kind: " + error);
    return math_internal::checked_expr_result(lamina::christoffel_first_kind_checked(
        *g, names, static_cast<int>(k), static_cast<int>(i), static_cast<int>(j)));
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Computes second-kind Christoffel symbol. @param metric Borrowed metric. @param inverse Borrowed inverse metric. @param variables Borrowed coordinate symbols. @param k Index. @param i Index. @param j Index. @return Owning Result Expr or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_geometry_christoffel_second(ExprObj* metric, ExprObj* inverse, ArrayObj* variables, LmInt k, LmInt i, LmInt j) noexcept try {
    ensure_lmmc_runtime();
    if (k < 0 || i < 0 || j < 0 || k > std::numeric_limits<int>::max() ||
        i > std::numeric_limits<int>::max() || j > std::numeric_limits<int>::max())
        return result_error(MathErrorCode::InvalidArgument, __func__, "geometry.christoffel_second_kind: invalid index");
    std::string error; const auto* g = checked_expr(metric, error); const auto* inv = checked_expr(inverse, error);
    std::vector<std::string> names;
    if (!g || !inv || !math_internal::checked_symbol_names(variables, names, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "geometry.christoffel_second_kind: " + error);
    return math_internal::checked_expr_result(lamina::christoffel_second_kind_checked(
        *g, *inv, names, static_cast<int>(k), static_cast<int>(i), static_cast<int>(j)));
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Computes Riemann tensor component. @param metric Borrowed metric. @param variables Borrowed coordinate symbols. @param rho Index. @param sigma Index. @param mu Index. @param nu Index. @return Owning Result Expr or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_geometry_riemann(ExprObj* metric, ArrayObj* variables, LmInt rho, LmInt sigma, LmInt mu, LmInt nu) noexcept try {
    ensure_lmmc_runtime();
    if (rho < 0 || sigma < 0 || mu < 0 || nu < 0 ||
        rho > std::numeric_limits<int>::max() || sigma > std::numeric_limits<int>::max() ||
        mu > std::numeric_limits<int>::max() || nu > std::numeric_limits<int>::max())
        return result_error(MathErrorCode::InvalidArgument, __func__, "geometry.riemann_curvature_tensor: invalid index");
    std::string error; const auto* g = checked_expr(metric, error); std::vector<std::string> names;
    if (!g || !math_internal::checked_symbol_names(variables, names, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "geometry.riemann_curvature_tensor: " + error);
    return math_internal::checked_expr_result(lamina::riemann_curvature_tensor_checked(
        *g, names, static_cast<int>(rho), static_cast<int>(sigma),
        static_cast<int>(mu), static_cast<int>(nu)));
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Computes exterior derivative coefficients. @param coefficients Borrowed form coefficients. @param degree Form degree. @param variables Borrowed coordinate symbols. @return Owning Result expression array or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_geometry_exterior_derivative(ArrayObj* coefficients, LmInt degree, ArrayObj* variables) noexcept try {
    ensure_lmmc_runtime();
    if (degree < 0 || degree > std::numeric_limits<int>::max())
        return result_error(MathErrorCode::InvalidArgument, __func__, "geometry.exterior_derivative: invalid degree");
    std::string error; std::vector<lamina::lsr::ExprPtr> values; std::vector<std::string> names;
    if (!array_expressions(coefficients, values, error) ||
        !math_internal::checked_symbol_names(variables, names, error) ||
        static_cast<std::size_t>(degree) > names.size())
        return result_error(MathErrorCode::InvalidArgument, __func__, "geometry.exterior_derivative: " + error);
    return expr_array_result(lamina::exterior_derivative_checked(
        values, static_cast<int>(degree), names));
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Computes Lie derivative. @param value Borrowed scalar. @param field Borrowed vector field. @param variables Borrowed coordinate symbols. @param order Positive order. @return Owning Result Expr or error. @ownership Inputs borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_computer_algebra_geometry_lie_derivative(ExprObj* value, ArrayObj* field, ArrayObj* variables, LmInt order) noexcept try {
    ensure_lmmc_runtime();
    if (order <= 0 || order > std::numeric_limits<int>::max())
        return result_error(MathErrorCode::InvalidArgument, __func__, "geometry.lie_derivative: invalid order");
    std::string error; const auto* f = checked_expr(value, error);
    std::vector<lamina::lsr::ExprPtr> x; std::vector<std::string> names;
    if (!f || !array_expressions(field, x, error) ||
        !math_internal::checked_symbol_names(variables, names, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "geometry.lie_derivative: " + error);
    return math_internal::checked_expr_result(lamina::lie_derivative_checked(
        *f, x, names, static_cast<int>(order)));
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_computer_algebra_geometry_vector_dot(
    ArrayObj* lhs, ArrayObj* rhs) noexcept try {
    ensure_lmmc_runtime();
    std::vector<lamina::lsr::ExprPtr> left;
    std::vector<lamina::lsr::ExprPtr> right;
    std::string error;
    if (!array_expressions(lhs, left, error) ||
        !array_expressions(rhs, right, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "geometry.vector_dot: " + error);
    const auto result = lamina::vector_dot_checked(left, right);
    if (!result) return result_error(result.error());
    return result_ok(
        new ExprObj(result.value()->simplify()), ValueKind::Expr);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_computer_algebra_geometry_vector_cross(
    ArrayObj* lhs, ArrayObj* rhs) noexcept try {
    ensure_lmmc_runtime();
    std::vector<lamina::lsr::ExprPtr> left;
    std::vector<lamina::lsr::ExprPtr> right;
    std::string error;
    if (!array_expressions(lhs, left, error) ||
        !array_expressions(rhs, right, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "geometry.vector_cross: " + error);
    const auto result = lamina::vector_cross_checked(left, right);
    if (!result) return result_error(result.error());
    return expr_array_result(result);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_computer_algebra_geometry_line_plane_intersection(
    ArrayObj* point, ArrayObj* direction, ArrayObj* normal, ExprObj* offset) noexcept try {
    ensure_lmmc_runtime();
    lamina::LineSymbolic line;
    lamina::PlaneSymbolic plane;
    std::string error;
    if (!array_expressions(point, line.point, error) ||
        !array_expressions(direction, line.direction, error) ||
        !array_expressions(normal, plane.normal, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, 
            "geometry.line_plane_intersection: " + error);
    const auto* checked_offset = checked_expr(offset, error);
    if (!checked_offset) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    plane.d = *checked_offset;
    const auto result =
        lamina::line_plane_intersection_checked(line, plane);
    if (!result) return result_error(result.error());
    return expr_array_result(result);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_computer_algebra_geometry_metric_inverse(ExprObj* metric) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    const auto* checked = checked_expr(metric, error);
    if (!checked) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    const auto result = lamina::metric_inverse_checked(*checked);
    if (!result) return result_error(result.error());
    return result_ok(new ExprObj(result.value()), ValueKind::Expr);
} catch (...) {
    return c_abi_current_exception(__func__);
}
