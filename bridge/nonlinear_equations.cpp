#include "bridge/result.hpp"
#include "bridge/conversions.hpp"
#include "bridge/runtime_views.hpp"
#include "bridge/unit_bridge.hpp"
#include <cstdarg>
#include "bridge/callback.hpp"
#include "lmmc/nonlinear.h"

using namespace lmx::bridge;

namespace {
struct DualScalarContext {
    ScalarCallbackContext function;
    ScalarCallbackContext derivative;
    DualScalarContext(const lmx::runtime::FuncObj* function_,
                      const lmx::runtime::FuncObj* derivative_)
        : function(function_), derivative(derivative_) {}
};

double dual_function(const double value, void* data) noexcept {
    return scalar_callback_trampoline(
        value, &static_cast<DualScalarContext*>(data)->function);
}
double dual_derivative(const double value, void* data) noexcept {
    return scalar_callback_trampoline(
        value, &static_cast<DualScalarContext*>(data)->derivative);
}

AdtObj* root_result(
    const lmmc_nonlinear_result_t& result) {
    if (result.num_iter > static_cast<std::size_t>(
                              std::numeric_limits<LmInt>::max()))
        return result_error(MathErrorCode::NumericalFailure, __func__, "nonlinear: iteration count overflow");
    std::vector<Value> fields;
    fields.emplace_back(result.converged != 0);
    fields.emplace_back(static_cast<LmInt>(result.num_iter));
    fields.emplace_back(result.root);
    fields.emplace_back(result.function_value);
    fields.emplace_back(result.residual_norm);
    fields.emplace_back(static_cast<lmx::runtime::Object*>(
        new StringObj(lmmc_nonlinear_failure_string(result.failure_reason))));
    return result_ok(
        new AdtObj("RootResult", "RootResult", std::move(fields)),
        ValueKind::Obj);
}

AdtObj* run_root_solver(
    const int algorithm, const lmx::runtime::FuncObj* function,
    const lmx::runtime::FuncObj* derivative, const double first,
    const double second, const double abs_tol, const double rel_tol,
    const LmInt max_iterations) {
    if (!function || max_iterations <= 0 || !std::isfinite(first) ||
        !std::isfinite(second) || !std::isfinite(abs_tol) ||
        !std::isfinite(rel_tol) || abs_tol < 0.0 || rel_tol < 0.0)
        return result_error(MathErrorCode::InvalidArgument, __func__, "nonlinear: invalid argument");
    lmmc_nonlinear_config_t config{};
    auto status = lmmc_nonlinear_default_config(&config);
    if (status != LMMC_STATUS_OK)
        return result_error(status, "nonlinear");
    config.abs_tol = abs_tol;
    config.rel_tol = rel_tol;
    config.max_iter = static_cast<std::size_t>(max_iterations);
    DualScalarContext context(function, derivative);
    lmmc_nonlinear_result_t output{};
    if (algorithm == 0)
        status = lmmc_bisection_solve(
            dual_function, &context, first, second, &config, &output);
    else if (algorithm == 1)
        status = lmmc_newton_solve(
            dual_function, derivative ? dual_derivative : nullptr,
            &context, first, &config, &output);
    else
        status = lmmc_secant_solve(
            dual_function, &context, first, second, &config, &output);
    if (context.function.failed())
        return result_error(MathErrorCode::CallbackFailure, "nonlinear",
                            std::move(context.function.error));
    if (derivative && context.derivative.failed())
        return result_error(MathErrorCode::CallbackFailure, "nonlinear",
                            std::move(context.derivative.error));
    if (status != LMMC_STATUS_OK)
        return result_error(status, "nonlinear");
    return root_result(output);
}
} // namespace

extern "C" LM_API AdtObj* lmx_nonlinear_equations_bisection(
    const lmx::runtime::FuncObj* function, const double left,
    const double right, const double abs_tol, const double rel_tol,
    const LmInt max_iterations) {
    return run_root_solver(
        0, function, nullptr, left, right, abs_tol, rel_tol, max_iterations);
}
extern "C" LM_API AdtObj* lmx_nonlinear_equations_newton(
    const lmx::runtime::FuncObj* function, const double initial,
    const double abs_tol, const double rel_tol,
    const LmInt max_iterations) {
    return run_root_solver(
        1, function, nullptr, initial, 0.0,
        abs_tol, rel_tol, max_iterations);
}
extern "C" LM_API AdtObj* lmx_nonlinear_equations_newton_with_derivative(
    const lmx::runtime::FuncObj* function,
    const lmx::runtime::FuncObj* derivative, const double initial,
    const double abs_tol, const double rel_tol,
    const LmInt max_iterations) {
    if (!derivative)
        return result_error(MathErrorCode::InvalidArgument, __func__, "nonlinear.newton_with_derivative: null derivative");
    return run_root_solver(
        1, function, derivative, initial, 0.0,
        abs_tol, rel_tol, max_iterations);
}
extern "C" LM_API AdtObj* lmx_nonlinear_equations_secant(
    const lmx::runtime::FuncObj* function, const double first,
    const double second, const double abs_tol, const double rel_tol,
    const LmInt max_iterations) {
    return run_root_solver(
        2, function, nullptr, first, second,
        abs_tol, rel_tol, max_iterations);
}
extern "C" LM_API double lmx_nonlinear_equations_root(AdtObj* result) {
    const auto* field = result ? result->field(2) : nullptr;
    return field && field->kind == ValueKind::Real
        ? field->real_val : std::numeric_limits<double>::quiet_NaN();
}
