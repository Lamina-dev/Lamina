#include "bridge/result.hpp"
#include "bridge/conversions.hpp"
#include <cstdarg>
#include "include/lmx_expr.h"

using namespace lmx::bridge;

namespace {
struct VaListEnd {
    va_list* args;
    ~VaListEnd() { va_end(*args); }
};

ExprObj* input_error(ExprObj* object, std::string message, const char* operation) {
    if (object && !object->ok()) return new ExprObj(object->error());
    return expr_from_result(invalid_expr_operation(message, operation));
}

ExprObj* boundary_error(const char* operation) noexcept {
    try {
        throw;
    } catch (const LMCAS::CasError& error) {
        try { return new ExprObj(error); } catch (...) { return nullptr; }
    } catch (const std::bad_alloc&) {
        return nullptr;
    } catch (const std::exception& error) {
        try {
            return new ExprObj(LMCAS::CasError{
                LMCAS::CasErrc::InternalInvariant, error.what(), operation});
        } catch (...) { return nullptr; }
    } catch (...) {
        try {
            return new ExprObj(LMCAS::CasError{
                LMCAS::CasErrc::InternalInvariant, "unknown exception", operation});
        } catch (...) { return nullptr; }
    }
}
}

extern "C" LM_API ExprObj* lmx_cas_expr_symbol(
    const char* name) noexcept try {
    ensure_lmmc_runtime();
    return expr_from_result(LMCAS::sym(name ? name : ""));
} catch (...) {
    return boundary_error(__func__);
}

extern "C" LM_API AdtObj* lmx_cas_symbol(const char* name) noexcept try {
    ensure_lmmc_runtime();
    return expr_result_ok(LMCAS::sym(name ? name : ""));
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_cas_parse(const char* source) noexcept try {
    ensure_lmmc_runtime();
    return expr_result_ok(LMCAS::parse_expr(source ? source : ""));
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_cas_serialize_expr(ExprObj* value) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    const auto* expression = checked_expr(value, error);
    if (!expression)
        return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    LMCAS::ComputationContext context;
    auto result = LMCAS::serialize_expr(*expression, context);
    if (!result) return result_error(result.error());
    return result_ok(new StringObj(std::move(result.value())), ValueKind::Obj);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_cas_parse_serialized_expr(
    const StringObj* source) noexcept try {
    ensure_lmmc_runtime();
    if (!source)
        return result_error(MathErrorCode::InvalidArgument, __func__, "null serialized expression");
    LMCAS::ComputationContext context;
    return expr_result_ok(LMCAS::parse_serialized_expr(source->to_string(), context));
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_cas_pi() noexcept try {
    ensure_lmmc_runtime();
    return expr_result_ok(LMCAS::pi());
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_cas_euler_number() noexcept try {
    ensure_lmmc_runtime();
    return expr_result_ok(LMCAS::e());
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_cas_golden_ratio() noexcept try {
    ensure_lmmc_runtime();
    return expr_result_ok(LMCAS::phi());
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API ExprObj* lmx_cas_expr_imaginary_unit() noexcept try {
    ensure_lmmc_runtime();
    return expr_from_result(LMCAS::imaginary_unit());
} catch (...) {
    return boundary_error(__func__);
}

extern "C" LM_API ExprObj* lmx_cas_expr_integer(const LmInt value) noexcept try {
    ensure_lmmc_runtime();
    return expr_from_result(LMCAS::integer(LMCAS::BigInt(value)));
} catch (...) {
    return boundary_error(__func__);
}

extern "C" LM_API ExprObj* lmx_cas_expr_rational(const LmInt numerator,
                                               const LmInt denominator) noexcept try {
    ensure_lmmc_runtime();
    if (denominator == 0) return expr_from_result(invalid_expr_operation(
        "rational denominator is zero", __func__));
    return expr_from_result(LMCAS::rational(LMCAS::Rational(
        LMCAS::BigInt(numerator), LMCAS::BigInt(denominator))));
} catch (...) {
    return boundary_error(__func__);
}

extern "C" LM_API ExprObj* lmx_cas_expr_promote_value(const lmx::runtime::Value* value) noexcept try {
    ensure_lmmc_runtime();
    if (!value) return expr_from_result(invalid_expr_operation(
        "null Lamina value", __func__));
    switch (value->kind) {
    case lmx::runtime::ValueKind::Int:
        return expr_from_result(LMCAS::integer(LMCAS::BigInt(value->int_val)));
    case lmx::runtime::ValueKind::Fraction:
        return expr_from_result(LMCAS::rational(LMCAS::Rational(
            value->frac_val.numerator(), value->frac_val.denominator())));
    case lmx::runtime::ValueKind::Real:
        return expr_from_result(LMCAS::approx_real(value->real_val));
    case lmx::runtime::ValueKind::Expr: {
        std::string error;
        const auto* expression = checked_expr(
            reinterpret_cast<ExprObj*>(value->obj), error);
        return expression ? new ExprObj(*expression) :
            input_error(reinterpret_cast<ExprObj*>(value->obj), std::move(error), __func__);
    }
    case lmx::runtime::ValueKind::Complex: {
        const auto* complex = reinterpret_cast<const ComplexObj*>(value->obj);
        if (!complex) {
            return expr_from_result(invalid_expr_operation(
                "null complex value cannot be promoted to Expr",
                "runtime.expr_value"));
        }
        const auto real = LMCAS::approx_real(complex->real());
        if (!real) return expr_from_result(real);
        const auto imag = LMCAS::approx_real(complex->imag());
        if (!imag) return expr_from_result(imag);
        return expr_from_result(LMCAS::complex(real.value(), imag.value()));
    }
    default:
        return expr_from_result(invalid_expr_operation(
            "Lamina value cannot be promoted to Expr", "runtime.expr_value"));
    }
} catch (...) {
    return boundary_error(__func__);
}

extern "C" LM_API ExprObj* lmx_cas_expr_unary(const LmInt operation,
                                            ExprObj* operand) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    const auto* value = checked_expr(operand, error);
    if (!value) return input_error(operand, std::move(error), __func__);
    switch (operation) {
    case LMX_EXPRESSION_OPERATION_NEG:
        return expr_from_result(LMCAS::neg(*value));
    case LMX_EXPRESSION_OPERATION_NOT:
        return expr_from_result(LMCAS::logical_not(*value));
    default:
        return expr_from_result(invalid_expr_operation(
            "unknown unary Expr operation", "runtime.expr_unary"));
    }
} catch (...) {
    return boundary_error(__func__);
}

extern "C" LM_API ExprObj* lmx_cas_expr_binary(const LmInt operation,
                                             ExprObj* lhs, ExprObj* rhs) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    const auto* left = checked_expr(lhs, error);
    if (!left) return input_error(lhs, std::move(error), __func__);
    const auto* right = checked_expr(rhs, error);
    if (!right) return input_error(rhs, std::move(error), __func__);
    switch (operation) {
    case LMX_EXPRESSION_OPERATION_ADD: return expr_from_result(LMCAS::add(*left, *right));
    case LMX_EXPRESSION_OPERATION_SUB: return expr_from_result(LMCAS::sub(*left, *right));
    case LMX_EXPRESSION_OPERATION_MUL: return expr_from_result(LMCAS::mul(*left, *right));
    case LMX_EXPRESSION_OPERATION_DIV: return expr_from_result(LMCAS::div(*left, *right));
    case LMX_EXPRESSION_OPERATION_POW: return expr_from_result(LMCAS::pow(*left, *right));
    case LMX_EXPRESSION_OPERATION_EQ: return expr_from_result(LMCAS::relation(*left, *right, LMCAS::RelationOp::EQ));
    case LMX_EXPRESSION_OPERATION_NE: return expr_from_result(LMCAS::relation(*left, *right, LMCAS::RelationOp::NEQ));
    case LMX_EXPRESSION_OPERATION_GT: return expr_from_result(LMCAS::relation(*left, *right, LMCAS::RelationOp::GT));
    case LMX_EXPRESSION_OPERATION_GE: return expr_from_result(LMCAS::relation(*left, *right, LMCAS::RelationOp::GEQ));
    case LMX_EXPRESSION_OPERATION_LT: return expr_from_result(LMCAS::relation(*left, *right, LMCAS::RelationOp::LT));
    case LMX_EXPRESSION_OPERATION_LE: return expr_from_result(LMCAS::relation(*left, *right, LMCAS::RelationOp::LEQ));
    case LMX_EXPRESSION_OPERATION_AND: return expr_from_result(LMCAS::logical_and(*left, *right));
    case LMX_EXPRESSION_OPERATION_OR: return expr_from_result(LMCAS::logical_or(*left, *right));
    case LMX_EXPRESSION_OPERATION_IN: return expr_from_result(LMCAS::membership(*left, *right));
    case LMX_EXPRESSION_OPERATION_NOT_IN: return expr_from_result(LMCAS::membership(*left, *right, true));
    default:
        return expr_from_result(invalid_expr_operation(
            "unknown binary Expr operation", "runtime.expr_binary"));
    }
} catch (...) {
    return boundary_error(__func__);
}

extern "C" LM_API ExprObj* lmx_cas_expr_function(const char* name,
                                               const LmInt count, ...) noexcept try {
    ensure_lmmc_runtime();
    va_list args;
    va_start(args, count);
    VaListEnd args_end{&args};
    std::vector<LMCAS::ExprPtr> values;
    std::string error;
    const bool valid = collect_expr_arguments(args, count, values, error);
    if (!valid) return expr_from_result(invalid_expr_operation(error, __func__));
    return expr_from_result(LMCAS::function(name ? name : "", std::move(values)));
} catch (...) {
    return boundary_error(__func__);
}

extern "C" LM_API ExprObj* lmx_cas_expr_set(const LmInt count, ...) noexcept try {
    ensure_lmmc_runtime();
    va_list args;
    va_start(args, count);
    VaListEnd args_end{&args};
    std::vector<LMCAS::ExprPtr> values;
    std::string error;
    const bool valid = collect_expr_arguments(args, count, values, error);
    if (!valid) return expr_from_result(invalid_expr_operation(error, __func__));
    return expr_from_result(LMCAS::finite_set(std::move(values)));
} catch (...) {
    return boundary_error(__func__);
}

extern "C" LM_API ExprObj* lmx_cas_expr_interval(ExprObj* lower, ExprObj* upper,
                                               const bool lower_closed,
                                               const bool upper_closed) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    const auto* lower_value = checked_expr(lower, error);
    if (!lower_value) return input_error(lower, std::move(error), __func__);
    const auto* upper_value = checked_expr(upper, error);
    if (!upper_value) return input_error(upper, std::move(error), __func__);
    return expr_from_result(LMCAS::interval(
        *lower_value, *upper_value, lower_closed, upper_closed));
} catch (...) {
    return boundary_error(__func__);
}

extern "C" LM_API ExprObj* lmx_cas_expr_attach_unit(
    ExprObj* value, const char* display_unit, const char* dimension,
    const LmInt scale_numerator, const LmInt scale_denominator) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    const auto* expression = checked_expr(value, error);
    if (!expression) return input_error(value, std::move(error), __func__);
    auto definition = resolved_unit_definition(
        dimension, scale_numerator, scale_denominator, error);
    if (!definition) return expr_from_result(invalid_expr_operation(error, __func__));
    LMCAS::ComputationContext context;
    return expr_from_result(LMCAS::with_unit_definition(
        *expression, display_unit ? display_unit : "1",
        std::move(*definition), context));
} catch (...) {
    return boundary_error(__func__);
}

extern "C" LM_API ExprObj* lmx_cas_expr_convert_unit(
    ExprObj* value, const char* display_unit, const char* dimension,
    const LmInt scale_numerator, const LmInt scale_denominator) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    const auto* expression = checked_expr(value, error);
    if (!expression) return input_error(value, std::move(error), __func__);
    auto definition = resolved_unit_definition(
        dimension, scale_numerator, scale_denominator, error);
    if (!definition) return expr_from_result(invalid_expr_operation(error, __func__));
    LMCAS::ComputationContext context;
    return expr_from_result(LMCAS::convert_to_unit_definition(
        *expression, display_unit ? display_unit : "1",
        std::move(*definition), context));
} catch (...) {
    return boundary_error(__func__);
}

extern "C" LM_API ExprObj* lmx_cas_expr_strip_base_value(ExprObj* value) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    const auto* expression = checked_expr(value, error);
    if (!expression) return input_error(value, std::move(error), __func__);
    LMCAS::ComputationContext context;
    return expr_from_result(LMCAS::strip_to_base_value(
        *expression, context));
} catch (...) {
    return boundary_error(__func__);
}

extern "C" LM_API ExprObj* lmx_cas_expr_strip_display_value(ExprObj* value) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    const auto* expression = checked_expr(value, error);
    if (!expression) return input_error(value, std::move(error), __func__);
    LMCAS::ComputationContext context;
    return expr_from_result(LMCAS::strip_to_display_value(
        *expression, context));
} catch (...) {
    return boundary_error(__func__);
}
