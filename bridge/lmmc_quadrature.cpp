#include "bridge/result.hpp"
#include "bridge/conversions.hpp"
#include "bridge/runtime_views.hpp"
#include "bridge/unit_bridge.hpp"
#include <cstdarg>
#include "bridge/callback.hpp"
#include "lmmc/quadrature.h"

using namespace lmx::bridge;

extern "C" LM_API AdtObj* lmx_integration_adaptive(
    const lmx::runtime::FuncObj* function, const double lower,
    const double upper, const double absolute_tolerance,
    const double relative_tolerance, const LmInt max_depth) {
    if (!function) {
        return result_error(MathErrorCode::InvalidArgument, __func__, "integrate.adaptive: null callback");
    }
    if (!std::isfinite(lower)) {
        return result_error(MathErrorCode::InvalidArgument, __func__, "integrate.adaptive: non-finite lower bound");
    }
    if (!std::isfinite(upper)) {
        return result_error(MathErrorCode::InvalidArgument, __func__, "integrate.adaptive: non-finite upper bound");
    }
    if (!std::isfinite(absolute_tolerance)) {
        return result_error(MathErrorCode::InvalidArgument, __func__, "integrate.adaptive: non-finite absolute tolerance");
    }
    if (!std::isfinite(relative_tolerance)) {
        return result_error(MathErrorCode::InvalidArgument, __func__, "integrate.adaptive: non-finite relative tolerance");
    }
    if (absolute_tolerance < 0.0 || relative_tolerance < 0.0 ||
        max_depth < 1 || max_depth > 64) {
        return result_error(MathErrorCode::InvalidArgument, __func__, "integrate.adaptive: invalid tolerance or depth");
    }
    ScalarCallbackContext context(function);
    lmmc_quad_result_t output{};
    const auto status = lmmc_quad_adaptive(
        scalar_callback_trampoline, &context, lower, upper,
        absolute_tolerance, relative_tolerance,
        static_cast<std::size_t>(max_depth), &output);
    if (context.failed())
        return result_error(MathErrorCode::CallbackFailure,
                            "integrate.adaptive", std::move(context.error));
    if (status != LMMC_STATUS_OK &&
        status != LMMC_STATUS_WARNING_MAX_DEPTH) {
        return result_error(status, "integrate.adaptive");
    }
    if (!std::isfinite(output.value) || !std::isfinite(output.error) ||
        output.num_evals >
            static_cast<std::size_t>(std::numeric_limits<LmInt>::max())) {
        return result_error(MathErrorCode::NumericalFailure, __func__, "integrate.adaptive: invalid numerical result");
    }

    std::vector<Value> fields;
    fields.emplace_back(output.value);
    fields.emplace_back(output.error);
    fields.emplace_back(static_cast<LmInt>(output.num_evals));
    fields.emplace_back(static_cast<lmx::runtime::Object*>(
        new StringObj(status == LMMC_STATUS_OK ? "ok" : "warning_max_depth")));
    return result_ok(
        new AdtObj("IntegralResult", "IntegralResult", std::move(fields)),
        ValueKind::Obj);
}

extern "C" LM_API double lmx_integration_value(AdtObj* result) {
    if (!result || result->type_name() != "IntegralResult" ||
        result->constructor() != "IntegralResult") {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const auto* value = result->field(0);
    return value && value->kind == ValueKind::Real
        ? value->real_val : std::numeric_limits<double>::quiet_NaN();
}

extern "C" LM_API StringObj* lmx_integration_status(AdtObj* result) {
    if (!result || result->type_name() != "IntegralResult" ||
        result->constructor() != "IntegralResult") {
        return new StringObj("invalid");
    }
    const auto* value = result->field(3);
    if (!value || value->kind != ValueKind::Obj || !value->obj ||
        value->obj->get_kind() != lmx::runtime::ObjectKind::String) {
        return new StringObj("invalid");
    }
    return new StringObj(static_cast<StringObj*>(value->obj)->c_str());
}

namespace {
AdtObj* integral_estimate(
    const char* name, const lmmc_status_t status,
    const lmmc_quad_result_t& output) {
    if (status != LMMC_STATUS_OK &&
        status != LMMC_STATUS_WARNING_MAX_DEPTH)
        return result_error(status, name);
    if (!std::isfinite(output.value) || !std::isfinite(output.error) ||
        output.num_evals >
            static_cast<std::size_t>(std::numeric_limits<LmInt>::max()))
        return result_error(MathErrorCode::InvalidArgument, __func__, std::string(name) + ": invalid result");
    std::vector<Value> fields;
    fields.emplace_back(output.value);
    fields.emplace_back(output.error);
    fields.emplace_back(static_cast<LmInt>(output.num_evals));
    fields.emplace_back(static_cast<lmx::runtime::Object*>(
        new StringObj(status == LMMC_STATUS_OK ? "ok" : "warning_max_depth")));
    return result_ok(
        new AdtObj("IntegralResult", "IntegralResult", std::move(fields)),
        ValueKind::Obj);
}

AdtObj* fixed_integral(
    const char* name, const lmx::runtime::FuncObj* function,
    const double lower, const double upper, const LmInt order,
    const int algorithm) {
    if (!function || order <= 0 || !std::isfinite(lower) ||
        !std::isfinite(upper))
        return result_error(MathErrorCode::InvalidArgument, __func__, std::string(name) + ": invalid argument");
    ScalarCallbackContext context(function);
    double output = 0.0;
    lmmc_status_t status = LMMC_STATUS_INVALID_ARGUMENT;
    if (algorithm == 0)
        status = lmmc_quad_trapezoid(
            scalar_callback_trampoline, &context, lower, upper,
            static_cast<std::size_t>(order), &output);
    else if (algorithm == 1)
        status = lmmc_quad_simpson(
            scalar_callback_trampoline, &context, lower, upper,
            static_cast<std::size_t>(order), &output);
    else if (algorithm == 2)
        status = lmmc_quad_gauss_legendre(
            scalar_callback_trampoline, &context, lower, upper,
            static_cast<std::size_t>(order), &output);
    else if (algorithm == 3)
        status = lmmc_quad_gauss_hermite(
            scalar_callback_trampoline, &context,
            static_cast<std::size_t>(order), &output);
    else
        status = lmmc_quad_gauss_laguerre(
            scalar_callback_trampoline, &context,
            static_cast<std::size_t>(order), &output);
    if (context.failed())
        return result_error(MathErrorCode::CallbackFailure, name,
                            std::move(context.error));
    return lmmc_real_result(name, status, output);
}
} // namespace

extern "C" LM_API AdtObj* lmx_integration_trapezoid(
    const lmx::runtime::FuncObj* function, const double lower,
    const double upper, const LmInt intervals) {
    return fixed_integral(
        "integrate.trapezoid", function, lower, upper, intervals, 0);
}
extern "C" LM_API AdtObj* lmx_integration_simpson(
    const lmx::runtime::FuncObj* function, const double lower,
    const double upper, const LmInt intervals) {
    return fixed_integral(
        "integrate.simpson", function, lower, upper, intervals, 1);
}
extern "C" LM_API AdtObj* lmx_integration_gauss_legendre(
    const lmx::runtime::FuncObj* function, const double lower,
    const double upper, const LmInt order) {
    return fixed_integral(
        "integrate.gauss_legendre", function, lower, upper, order, 2);
}
extern "C" LM_API AdtObj* lmx_integration_gauss_hermite(
    const lmx::runtime::FuncObj* function, const LmInt order) {
    return fixed_integral(
        "integrate.gauss_hermite", function, 0.0, 0.0, order, 3);
}
extern "C" LM_API AdtObj* lmx_integration_gauss_laguerre(
    const lmx::runtime::FuncObj* function, const LmInt order) {
    return fixed_integral(
        "integrate.gauss_laguerre", function, 0.0, 0.0, order, 4);
}

extern "C" LM_API AdtObj* lmx_integration_romberg(
    const lmx::runtime::FuncObj* function, const double lower,
    const double upper, const double tolerance, const LmInt max_iterations) {
    if (!function || max_iterations <= 0 || !std::isfinite(tolerance))
        return result_error(MathErrorCode::InvalidArgument, __func__, "integrate.romberg: invalid argument");
    ScalarCallbackContext context(function);
    lmmc_quad_result_t output{};
    const auto status = lmmc_quad_romberg(
        scalar_callback_trampoline, &context, lower, upper, tolerance,
        static_cast<std::size_t>(max_iterations), &output);
    if (context.failed())
        return result_error(MathErrorCode::CallbackFailure,
                            "integrate.romberg", std::move(context.error));
    return integral_estimate("integrate.romberg", status, output);
}
extern "C" LM_API AdtObj* lmx_integration_tanh_sinh(
    const lmx::runtime::FuncObj* function, const double lower,
    const double upper, const double tolerance, const LmInt max_nodes) {
    if (!function || max_nodes <= 0 || !std::isfinite(tolerance))
        return result_error(MathErrorCode::InvalidArgument, __func__, "integrate.tanh_sinh: invalid argument");
    ScalarCallbackContext context(function);
    lmmc_quad_result_t output{};
    const auto status = lmmc_quad_tanh_sinh(
        scalar_callback_trampoline, &context, lower, upper, tolerance,
        static_cast<std::size_t>(max_nodes), &output);
    if (context.failed())
        return result_error(MathErrorCode::CallbackFailure,
                            "integrate.tanh_sinh", std::move(context.error));
    return integral_estimate("integrate.tanh_sinh", status, output);
}
