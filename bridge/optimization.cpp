#include "bridge/result.hpp"
#include "bridge/conversions.hpp"
#include "bridge/runtime_views.hpp"
#include "bridge/unit_bridge.hpp"
#include <cstdarg>
#include "bridge/callback.hpp"
#include "lmmc/optimize.h"

using namespace lmx::bridge;

namespace {
struct OptimizeCallbacks {
    VectorCallbackContext function;
    MatrixCallbackContext jacobian;
    VectorScalarCallbackContext objective;
    VectorCallbackContext gradient;

    OptimizeCallbacks(
        const lmx::runtime::FuncObj* function_,
        const lmx::runtime::FuncObj* jacobian_,
        const lmx::runtime::FuncObj* objective_,
        const lmx::runtime::FuncObj* gradient_,
        const std::size_t dimension)
        : function(function_, dimension, dimension),
          jacobian(jacobian_, dimension, dimension, dimension),
          objective(objective_, dimension),
          gradient(gradient_, dimension, dimension) {}
};

lmmc_status_t optimize_function_adapter(
    const lmmc_vec_t* input, lmmc_vec_t* output, void* data) noexcept {
    return vector_callback_trampoline(
        input, output, &static_cast<OptimizeCallbacks*>(data)->function);
}
lmmc_status_t optimize_jacobian_adapter(
    const lmmc_vec_t* input, lmmc_mat_t* output, void* data) noexcept {
    return matrix_callback_trampoline(
        input, output, &static_cast<OptimizeCallbacks*>(data)->jacobian);
}
double optimize_objective_adapter(
    const lmmc_vec_t* input, void* data) noexcept {
    return vector_scalar_callback_trampoline(
        input, &static_cast<OptimizeCallbacks*>(data)->objective);
}

lmmc_status_t optimize_gradient_adapter(
    const lmmc_vec_t* input, lmmc_vec_t* output, void* data) noexcept {
    return vector_callback_trampoline(
        input, output, &static_cast<OptimizeCallbacks*>(data)->gradient);
}

const char* optimize_failure_text(const lmmc_optimize_failure_t reason) {
    switch (reason) {
    case LMMC_OPT_FAILURE_NONE: return "none";
    case LMMC_OPT_FAILURE_MAX_ITER: return "max_iter";
    case LMMC_OPT_FAILURE_NUMERICAL_ISSUE: return "numerical_issue";
    case LMMC_OPT_FAILURE_LINE_SEARCH_FAILED: return "line_search_failed";
    case LMMC_OPT_FAILURE_SINGULAR_JACOBIAN: return "singular_jacobian";
    }
    return "unknown";
}

AdtObj* run_optimizer(
    const int algorithm, const lmx::runtime::FuncObj* function,
    const lmx::runtime::FuncObj* jacobian,
    const lmx::runtime::FuncObj* objective,
    const lmx::runtime::FuncObj* gradient, VectorObj* initial,
    const double abs_tol, const double rel_tol,
    const LmInt max_iterations) {
    if (!initial || initial->size() == 0 || max_iterations <= 0 ||
        !std::isfinite(abs_tol) || !std::isfinite(rel_tol) ||
        abs_tol < 0.0 || rel_tol < 0.0)
        return result_error(MathErrorCode::InvalidArgument, __func__, "optimize: invalid argument");
    lmmc_optimize_config_t config{};
    auto status = lmmc_optimize_default_config(&config);
    if (status != LMMC_STATUS_OK)
        return result_error(status, "optimize");
    config.abs_tol = abs_tol;
    config.rel_tol = rel_tol;
    config.max_iter = static_cast<std::size_t>(max_iterations);
    OptimizeCallbacks callbacks(
        function, jacobian, objective, gradient, initial->size());
    auto solution = initial->data();
    lmmc_vec_t x{solution.size(), solution.data(), 0};
    lmmc_optimize_result_t result{};
    if (algorithm == 0)
        status = lmmc_nleq_newton(
            optimize_function_adapter,
            jacobian ? optimize_jacobian_adapter : nullptr,
            &callbacks, &x, &config, &result);
    else if (algorithm == 1)
        status = lmmc_nleq_broyden(
            optimize_function_adapter, &callbacks, &x, &config, &result);
    else if (algorithm == 2)
        status = lmmc_minimize_lbfgs(
            optimize_objective_adapter, optimize_gradient_adapter,
            &callbacks, &x, &config, &result);
    else if (algorithm == 3)
        status = lmmc_minimize_levenberg_marquardt(
            optimize_function_adapter,
            jacobian ? optimize_jacobian_adapter : nullptr,
            &callbacks, &x, &config, &result);
    else
        status = lmmc_minimize_gradient_descent(
            optimize_objective_adapter, optimize_gradient_adapter,
            &callbacks, &x, &config, &result);
    if (function && callbacks.function.failed())
        return result_error(MathErrorCode::CallbackFailure, "optimize",
                            std::move(callbacks.function.error));
    if (jacobian && callbacks.jacobian.failed())
        return result_error(MathErrorCode::CallbackFailure, "optimize",
                            std::move(callbacks.jacobian.error));
    if (objective && callbacks.objective.failed())
        return result_error(MathErrorCode::CallbackFailure, "optimize",
                            std::move(callbacks.objective.error));
    if (gradient && callbacks.gradient.failed())
        return result_error(MathErrorCode::CallbackFailure, "optimize",
                            std::move(callbacks.gradient.error));
    if (status != LMMC_STATUS_OK)
        return result_error(status, "optimize");
    if (result.num_iter > static_cast<std::size_t>(
                              std::numeric_limits<LmInt>::max()))
        return result_error(MathErrorCode::NumericalFailure, __func__, "optimize: iteration count overflow");
    std::vector<Value> fields;
    fields.emplace_back(take_object_value(
        make_owned_object<VectorObj>(std::move(solution)), ValueKind::Vector));
    fields.emplace_back(result.converged != 0);
    fields.emplace_back(static_cast<LmInt>(result.num_iter));
    fields.emplace_back(result.final_residual);
    fields.emplace_back(take_object_value(
        make_owned_object<StringObj>(
            optimize_failure_text(result.failure_reason)),
        ValueKind::Obj));
    return result_ok(
        new AdtObj("OptimizeResult", "OptimizeResult", std::move(fields)),
        ValueKind::Obj);
}
} // namespace

extern "C" LM_API AdtObj* lmx_optimization_newton_system(
    const lmx::runtime::FuncObj* function, VectorObj* initial,
    const double abs_tol, const double rel_tol, const LmInt max_iterations) noexcept try {
    ensure_lmmc_runtime();
    return run_optimizer(
        0, function, nullptr, nullptr, nullptr, initial,
        abs_tol, rel_tol, max_iterations);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_optimization_newton_system_with_jacobian(
    const lmx::runtime::FuncObj* function,
    const lmx::runtime::FuncObj* jacobian, VectorObj* initial,
    const double abs_tol, const double rel_tol, const LmInt max_iterations) noexcept try {
    ensure_lmmc_runtime();
    return run_optimizer(
        0, function, jacobian, nullptr, nullptr, initial,
        abs_tol, rel_tol, max_iterations);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_optimization_broyden(
    const lmx::runtime::FuncObj* function, VectorObj* initial,
    const double abs_tol, const double rel_tol, const LmInt max_iterations) noexcept try {
    ensure_lmmc_runtime();
    return run_optimizer(
        1, function, nullptr, nullptr, nullptr, initial,
        abs_tol, rel_tol, max_iterations);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_optimization_limited_memory_broyden_fletcher_goldfarb_shanno(
    const lmx::runtime::FuncObj* objective,
    const lmx::runtime::FuncObj* gradient, VectorObj* initial,
    const double abs_tol, const double rel_tol, const LmInt max_iterations) noexcept try {
    ensure_lmmc_runtime();
    return run_optimizer(
        2, nullptr, nullptr, objective, gradient, initial,
        abs_tol, rel_tol, max_iterations);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_optimization_levenberg_marquardt(
    const lmx::runtime::FuncObj* residual, VectorObj* initial,
    const double abs_tol, const double rel_tol, const LmInt max_iterations) noexcept try {
    ensure_lmmc_runtime();
    return run_optimizer(
        3, residual, nullptr, nullptr, nullptr, initial,
        abs_tol, rel_tol, max_iterations);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_optimization_levenberg_marquardt_with_jacobian(
    const lmx::runtime::FuncObj* residual,
    const lmx::runtime::FuncObj* jacobian, VectorObj* initial,
    const double abs_tol, const double rel_tol, const LmInt max_iterations) noexcept try {
    ensure_lmmc_runtime();
    return run_optimizer(
        3, residual, jacobian, nullptr, nullptr, initial,
        abs_tol, rel_tol, max_iterations);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_optimization_gradient_descent(
    const lmx::runtime::FuncObj* objective,
    const lmx::runtime::FuncObj* gradient, VectorObj* initial,
    const double abs_tol, const double rel_tol, const LmInt max_iterations) noexcept try {
    ensure_lmmc_runtime();
    return run_optimizer(
        4, nullptr, nullptr, objective, gradient, initial,
        abs_tol, rel_tol, max_iterations);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_optimization_solution(AdtObj* result) noexcept try {
    ensure_lmmc_runtime();
    const auto* field = result ? result->field(0) : nullptr;
    if (!field || field->kind != ValueKind::Vector || !field->obj)
        return result_error(MathErrorCode::InvalidArgument, __func__, "optimize: invalid OptimizeResult");
    return result_ok(field->obj->get(), ValueKind::Vector);
} catch (...) {
    return c_abi_current_exception(__func__);
}
