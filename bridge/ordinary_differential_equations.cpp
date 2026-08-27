#include "bridge/result.hpp"
#include "bridge/conversions.hpp"
#include "bridge/runtime_views.hpp"
#include "bridge/unit_bridge.hpp"
#include <cstdarg>
#include "bridge/callback.hpp"
#include "lmmc/ode.h"

using namespace lmx::bridge;

namespace {
using OdeSolver = lmmc_status_t (*)(
    lmmc_ode_rhs_t, void*, std::size_t, double, double, double*,
    const lmmc_ode_config_t*, lmmc_ode_result_t*);

struct OdeCallbacks {
    OdeCallbackContext rhs;
    OdeMatrixCallbackContext jacobian;
    OdeCallbacks(const lmx::runtime::FuncObj* rhs_,
                 const lmx::runtime::FuncObj* jacobian_,
                 const std::size_t dimension)
        : rhs(rhs_, dimension), jacobian(jacobian_, dimension) {}
};

lmmc_status_t ode_rhs_adapter(
    const double time, const double* state, double* derivative,
    const std::size_t dimension, void* data) noexcept {
    return ode_callback_trampoline(
        time, state, derivative, dimension,
        &static_cast<OdeCallbacks*>(data)->rhs);
}
lmmc_status_t ode_jacobian_adapter(
    const double time, const double* state, double* jacobian,
    const std::size_t dimension, void* data) noexcept {
    return ode_matrix_callback_trampoline(
        time, state, jacobian, dimension,
        &static_cast<OdeCallbacks*>(data)->jacobian);
}

AdtObj* run_ode_solver(
    const char* name, const OdeSolver solver,
    const lmx::runtime::FuncObj* rhs,
    const lmx::runtime::FuncObj* jacobian, VectorObj* initial,
    const double start, const double end, const double initial_step,
    const double abs_tol, const double rel_tol, const LmInt max_steps) {
    if (!rhs || !initial || initial->size() == 0 || max_steps <= 0 ||
        !std::isfinite(start) || !std::isfinite(end) ||
        !std::isfinite(initial_step) || !std::isfinite(abs_tol) ||
        !std::isfinite(rel_tol) || initial_step <= 0.0 ||
        abs_tol < 0.0 || rel_tol < 0.0)
        return result_error(MathErrorCode::InvalidArgument, __func__, std::string(name) + ": invalid argument");
    lmmc_ode_config_t config{};
    auto status = lmmc_ode_default_config(
        start, end, initial->size(), &config);
    if (status != LMMC_STATUS_OK) return result_error(status, name);
    config.initial_step = initial_step;
    config.abs_tol = abs_tol;
    config.rel_tol = rel_tol;
    config.max_steps = static_cast<std::size_t>(max_steps);
    config.jacobian = jacobian ? ode_jacobian_adapter : nullptr;
    OdeCallbacks callbacks(rhs, jacobian, initial->size());
    auto state = initial->data();
    lmmc_ode_result_t result{};
    status = solver(
        ode_rhs_adapter, &callbacks, state.size(), start, end,
        state.data(), &config, &result);
    if (callbacks.rhs.failed())
        return result_error(MathErrorCode::CallbackFailure, name,
                            std::move(callbacks.rhs.error));
    if (jacobian && callbacks.jacobian.failed())
        return result_error(MathErrorCode::CallbackFailure, name,
                            std::move(callbacks.jacobian.error));
    if (status != LMMC_STATUS_OK)
        return result_error(status, name);
    if (result.num_steps > static_cast<std::size_t>(
                               std::numeric_limits<LmInt>::max()) ||
        result.num_rhs_evals > static_cast<std::size_t>(
                                  std::numeric_limits<LmInt>::max()))
        return result_error(MathErrorCode::NumericalFailure, __func__, std::string(name) + ": count overflow");
    std::vector<Value> fields;
    fields.emplace_back(new VectorObj(std::move(state)), ValueKind::Vector);
    fields.emplace_back(result.converged != 0);
    fields.emplace_back(static_cast<LmInt>(result.num_steps));
    fields.emplace_back(static_cast<LmInt>(result.num_rhs_evals));
    fields.emplace_back(result.final_t);
    fields.emplace_back(static_cast<lmx::runtime::Object*>(
        new StringObj(lmmc_ode_failure_string(result.failure_reason))));
    return result_ok(
        new AdtObj("OdeResult", "OdeResult", std::move(fields)),
        ValueKind::Obj);
}
} // namespace

extern "C" LM_API AdtObj* lmx_ordinary_differential_equations_euler(
    const lmx::runtime::FuncObj* rhs, VectorObj* initial,
    const double start, const double end, const double step,
    const double abs_tol, const double rel_tol, const LmInt max_steps) {
    return run_ode_solver(
        "ode.euler", lmmc_ode_euler_solve, rhs, nullptr, initial,
        start, end, step, abs_tol, rel_tol, max_steps);
}
extern "C" LM_API AdtObj* lmx_ordinary_differential_equations_runge_kutta_fourth_order(
    const lmx::runtime::FuncObj* rhs, VectorObj* initial,
    const double start, const double end, const double step,
    const double abs_tol, const double rel_tol, const LmInt max_steps) {
    return run_ode_solver(
        "ode.rk4", lmmc_ode_rk4_solve, rhs, nullptr, initial,
        start, end, step, abs_tol, rel_tol, max_steps);
}
extern "C" LM_API AdtObj* lmx_ordinary_differential_equations_runge_kutta_fourth_fifth_order(
    const lmx::runtime::FuncObj* rhs, VectorObj* initial,
    const double start, const double end, const double step,
    const double abs_tol, const double rel_tol, const LmInt max_steps) {
    return run_ode_solver(
        "ode.rk45", lmmc_ode_rk45_solve, rhs, nullptr, initial,
        start, end, step, abs_tol, rel_tol, max_steps);
}

#define LMX_ODE_IMPLICIT_EXPORT(export_name, solver_name) \
extern "C" LM_API AdtObj* lmx_ordinary_differential_equations_##export_name( \
    const lmx::runtime::FuncObj* rhs, VectorObj* initial, \
    const double start, const double end, const double step, \
    const double abs_tol, const double rel_tol, const LmInt max_steps) { \
    return run_ode_solver( \
        "ode." #export_name, solver_name, rhs, nullptr, initial, \
        start, end, step, abs_tol, rel_tol, max_steps); \
} \
extern "C" LM_API AdtObj* lmx_ordinary_differential_equations_##export_name##_with_jacobian( \
    const lmx::runtime::FuncObj* rhs, \
    const lmx::runtime::FuncObj* jacobian, VectorObj* initial, \
    const double start, const double end, const double step, \
    const double abs_tol, const double rel_tol, const LmInt max_steps) { \
    if (!jacobian) \
        return result_error(MathErrorCode::InvalidArgument, __func__, "ode." #export_name ": null Jacobian"); \
    return run_ode_solver( \
        "ode." #export_name "_with_jacobian", solver_name, rhs, jacobian, \
        initial, start, end, step, abs_tol, rel_tol, max_steps); \
}

LMX_ODE_IMPLICIT_EXPORT(implicit_euler, lmmc_ode_implicit_euler_solve)
LMX_ODE_IMPLICIT_EXPORT(trapezoidal, lmmc_ode_trapezoidal_solve)
LMX_ODE_IMPLICIT_EXPORT(singly_diagonally_implicit_runge_kutta_fourth_order, lmmc_ode_sdirk4_solve)
LMX_ODE_IMPLICIT_EXPORT(rosenbrock_generalized_runge_kutta_fourth_order, lmmc_ode_rosenbrock_grk4t_solve)
#undef LMX_ODE_IMPLICIT_EXPORT

extern "C" LM_API AdtObj* lmx_ordinary_differential_equations_final_state(AdtObj* result) {
    const auto* field = result ? result->field(0) : nullptr;
    if (!field || field->kind != ValueKind::Vector || !field->obj)
        return result_error(MathErrorCode::InvalidArgument, __func__, "ode: invalid OdeResult");
    return result_ok(field->obj->get(), ValueKind::Vector);
}
