#include "bridge/result.hpp"
#include "bridge/conversions.hpp"
#include "bridge/runtime_views.hpp"
#include "bridge/unit_bridge.hpp"
#include <cstdarg>
#include "include/lmx_expr.h"

#include <cstdarg>

using namespace lmx::bridge;

namespace {
struct VaListEnd {
    va_list* args;
    ~VaListEnd() { va_end(*args); }
};
} // namespace

extern "C" LM_API ExprObj* lmx_computer_algebra_expression_symbol(
    const char* name) noexcept try {
    ensure_lmmc_runtime();
    return expr_from_result(lamina::lsr::sym(name ? name : ""));
} catch (...) {
    return nullptr;
}

extern "C" LM_API AdtObj* lmx_computer_algebra_symbol(const char* name) noexcept try {
    ensure_lmmc_runtime();
    return expr_result_ok(lamina::lsr::sym(name ? name : ""));
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_computer_algebra_parse(const char* source) noexcept try {
    ensure_lmmc_runtime();
    return expr_result_ok(lamina::lsr::parse_expr(source ? source : ""));
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_computer_algebra_pi() noexcept try {
    ensure_lmmc_runtime();
    return expr_result_ok(lamina::lsr::pi());
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_computer_algebra_euler_number() noexcept try {
    ensure_lmmc_runtime();
    return expr_result_ok(lamina::lsr::e());
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_computer_algebra_golden_ratio() noexcept try {
    ensure_lmmc_runtime();
    return expr_result_ok(lamina::lsr::phi());
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API ExprObj* lmx_computer_algebra_expression_imaginary_unit() noexcept try {
    ensure_lmmc_runtime();
    return expr_from_result(lamina::lsr::imaginary_unit());
} catch (...) {
    return nullptr;
}

extern "C" LM_API ExprObj* lmx_computer_algebra_expression_integer(const LmInt value) noexcept try {
    ensure_lmmc_runtime();
    return expr_from_result(lamina::lsr::integer(value));
} catch (...) {
    return nullptr;
}

extern "C" LM_API ExprObj* lmx_computer_algebra_expression_rational(const LmInt numerator,
                                               const LmInt denominator) noexcept try {
    ensure_lmmc_runtime();
    if (denominator == 0) return expression_internal_error("CasError(DivisionByZero: rational denominator is zero)");
    return expr_from_result(lamina::lsr::rational(Rational(
        BigInt(std::to_string(numerator)), BigInt(std::to_string(denominator)))));
} catch (...) {
    return nullptr;
}

extern "C" LM_API ExprObj* lmx_computer_algebra_expression_promote_value(const lmx::runtime::Value* value) noexcept try {
    ensure_lmmc_runtime();
    if (!value) return expression_internal_error("CasError(InvalidArgument: null Lamina value)");
    switch (value->kind) {
    case lmx::runtime::ValueKind::Int:
        return expr_from_result(lamina::lsr::integer(value->int_val));
    case lmx::runtime::ValueKind::Fraction:
        return expr_from_result(lamina::lsr::rational(Rational(
            value->frac_val.numerator(), value->frac_val.denominator())));
    case lmx::runtime::ValueKind::Real:
        return expr_from_result(lamina::lsr::approx_real(value->real_val));
    case lmx::runtime::ValueKind::Expr: {
        std::string error;
        const auto* expression = checked_expr(
            reinterpret_cast<ExprObj*>(value->obj), error);
        return expression ? new ExprObj(*expression) : expression_internal_error(std::move(error));
    }
    case lmx::runtime::ValueKind::Complex: {
        const auto* complex = reinterpret_cast<const ComplexObj*>(value->obj);
        if (!complex) {
            return expr_from_result(invalid_expr_operation(
                "null complex value cannot be promoted to Expr",
                "runtime.expr_value"));
        }
        const auto real = lamina::lsr::approx_real(complex->real());
        if (!real) return expr_from_result(real);
        const auto imag = lamina::lsr::approx_real(complex->imag());
        if (!imag) return expr_from_result(imag);
        return expr_from_result(lamina::lsr::complex(real.value(), imag.value()));
    }
    default:
        return expr_from_result(invalid_expr_operation(
            "Lamina value cannot be promoted to Expr", "runtime.expr_value"));
    }
} catch (...) {
    return nullptr;
}

extern "C" LM_API ExprObj* lmx_computer_algebra_expression_unary(const LmInt operation,
                                            ExprObj* operand) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    const auto* value = checked_expr(operand, error);
    if (!value) return expression_internal_error(std::move(error));
    switch (operation) {
    case LMX_EXPRESSION_OPERATION_NEG:
        return expr_from_result(lamina::lsr::neg(*value));
    case LMX_EXPRESSION_OPERATION_NOT:
        return expr_from_result(lamina::lsr::logical_not(*value));
    default:
        return expr_from_result(invalid_expr_operation(
            "unknown unary Expr operation", "runtime.expr_unary"));
    }
} catch (...) {
    return nullptr;
}

extern "C" LM_API ExprObj* lmx_computer_algebra_expression_binary(const LmInt operation,
                                             ExprObj* lhs, ExprObj* rhs) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    const auto* left = checked_expr(lhs, error);
    if (!left) return expression_internal_error(std::move(error));
    const auto* right = checked_expr(rhs, error);
    if (!right) return expression_internal_error(std::move(error));
    switch (operation) {
    case LMX_EXPRESSION_OPERATION_ADD: return expr_from_result(lamina::lsr::add(*left, *right));
    case LMX_EXPRESSION_OPERATION_SUB: return expr_from_result(lamina::lsr::sub(*left, *right));
    case LMX_EXPRESSION_OPERATION_MUL: return expr_from_result(lamina::lsr::mul(*left, *right));
    case LMX_EXPRESSION_OPERATION_DIV: return expr_from_result(lamina::lsr::div(*left, *right));
    case LMX_EXPRESSION_OPERATION_POW: return expr_from_result(lamina::lsr::pow(*left, *right));
    case LMX_EXPRESSION_OPERATION_EQ: return expr_from_result(lamina::lsr::relation(*left, *right, lamina::RelationOp::EQ));
    case LMX_EXPRESSION_OPERATION_NE: return expr_from_result(lamina::lsr::relation(*left, *right, lamina::RelationOp::NEQ));
    case LMX_EXPRESSION_OPERATION_GT: return expr_from_result(lamina::lsr::relation(*left, *right, lamina::RelationOp::GT));
    case LMX_EXPRESSION_OPERATION_GE: return expr_from_result(lamina::lsr::relation(*left, *right, lamina::RelationOp::GEQ));
    case LMX_EXPRESSION_OPERATION_LT: return expr_from_result(lamina::lsr::relation(*left, *right, lamina::RelationOp::LT));
    case LMX_EXPRESSION_OPERATION_LE: return expr_from_result(lamina::lsr::relation(*left, *right, lamina::RelationOp::LEQ));
    case LMX_EXPRESSION_OPERATION_AND: return expr_from_result(lamina::lsr::logical_and(*left, *right));
    case LMX_EXPRESSION_OPERATION_OR: return expr_from_result(lamina::lsr::logical_or(*left, *right));
    case LMX_EXPRESSION_OPERATION_IN: return expr_from_result(lamina::lsr::membership(*left, *right));
    case LMX_EXPRESSION_OPERATION_NOT_IN: return expr_from_result(lamina::lsr::membership(*left, *right, true));
    default:
        return expr_from_result(invalid_expr_operation(
            "unknown binary Expr operation", "runtime.expr_binary"));
    }
} catch (...) {
    return nullptr;
}

extern "C" LM_API ExprObj* lmx_computer_algebra_expression_function(const char* name,
                                               const LmInt count, ...) noexcept try {
    ensure_lmmc_runtime();
    va_list args;
    va_start(args, count);
    VaListEnd args_end{&args};
    std::vector<lamina::lsr::ExprPtr> values;
    std::string error;
    const bool valid = collect_expr_arguments(args, count, values, error);
    if (!valid) return expression_internal_error(std::move(error));
    return expr_from_result(lamina::lsr::function(name ? name : "", std::move(values)));
} catch (...) {
    return nullptr;
}

extern "C" LM_API ExprObj* lmx_computer_algebra_expression_set(const LmInt count, ...) noexcept try {
    ensure_lmmc_runtime();
    va_list args;
    va_start(args, count);
    VaListEnd args_end{&args};
    std::vector<lamina::lsr::ExprPtr> values;
    std::string error;
    const bool valid = collect_expr_arguments(args, count, values, error);
    if (!valid) return expression_internal_error(std::move(error));
    return expr_from_result(lamina::lsr::finite_set(std::move(values)));
} catch (...) {
    return nullptr;
}

extern "C" LM_API ExprObj* lmx_computer_algebra_expression_interval(ExprObj* lower, ExprObj* upper,
                                               const bool lower_closed,
                                               const bool upper_closed) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    const auto* lower_value = checked_expr(lower, error);
    if (!lower_value) return expression_internal_error(std::move(error));
    const auto* upper_value = checked_expr(upper, error);
    if (!upper_value) return expression_internal_error(std::move(error));
    return expr_from_result(lamina::lsr::interval(
        *lower_value, *upper_value, lower_closed, upper_closed));
} catch (...) {
    return nullptr;
}

extern "C" LM_API ExprObj* lmx_computer_algebra_expression_attach_unit(
    ExprObj* value, const char* display_unit, const char* dimension,
    const LmInt scale_numerator, const LmInt scale_denominator) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    const auto* expression = checked_expr(value, error);
    if (!expression) return expression_internal_error(std::move(error));
    auto definition = resolved_unit_definition(
        dimension, scale_numerator, scale_denominator, error);
    if (!definition) return expression_internal_error(std::move(error));
    lamina::ComputationContext context;
    return expr_from_result(lamina::lsr::with_unit_definition(
        *expression, display_unit ? display_unit : "1",
        std::move(*definition), context));
} catch (...) {
    return nullptr;
}

extern "C" LM_API ExprObj* lmx_computer_algebra_expression_convert_unit(
    ExprObj* value, const char* display_unit, const char* dimension,
    const LmInt scale_numerator, const LmInt scale_denominator) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    const auto* expression = checked_expr(value, error);
    if (!expression) return expression_internal_error(std::move(error));
    auto definition = resolved_unit_definition(
        dimension, scale_numerator, scale_denominator, error);
    if (!definition) return expression_internal_error(std::move(error));
    lamina::ComputationContext context;
    return expr_from_result(lamina::lsr::convert_to_unit_definition(
        *expression, display_unit ? display_unit : "1",
        std::move(*definition), context));
} catch (...) {
    return nullptr;
}

extern "C" LM_API ExprObj* lmx_computer_algebra_expression_strip_base_value(ExprObj* value) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    const auto* expression = checked_expr(value, error);
    if (!expression) return expression_internal_error(std::move(error));
    lamina::ComputationContext context;
    return expr_from_result(lamina::lsr::strip_to_base_value(
        *expression, context));
} catch (...) {
    return nullptr;
}

extern "C" LM_API ExprObj* lmx_computer_algebra_expression_strip_display_value(ExprObj* value) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    const auto* expression = checked_expr(value, error);
    if (!expression) return expression_internal_error(std::move(error));
    lamina::ComputationContext context;
    return expr_from_result(lamina::lsr::strip_to_display_value(
        *expression, context));
} catch (...) {
    return nullptr;
}
