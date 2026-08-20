//
// Created by meian on 2026/4/8.
//

#include "include/lmx.h"
#include "include/lmx_expr.h"

#include "compiler/compiler.hpp"
#include "compiler/ast/ast_printer.hpp"
#include "compiler/hir/type_checker.hpp"
#include "compiler/parser.hpp"
#include "compiler/lexer.hpp"
#include "runtime/vm.hpp"
#include "runtime/object/lsr_expr_obj.hpp"
#include "runtime/object/StringObj.hpp"
#include "runtime/object/adt.hpp"
#include "runtime/object/complex.hpp"
#include "runtime/object/array.hpp"
#include "runtime/object/vector.hpp"
#include "runtime/object/matrix.hpp"
#include "runtime/object/table.hpp"
#include "runtime/object/random.hpp"
#include "runtime/object/quantity.hpp"

#include <cmath>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <fstream>
#include <limits>
#include <utility>
#include <cctype>

#include "compiler/mir/mir_printer.hpp"
#include "lmmc/numeric.h"
#include "lmmc/complex.h"
#include "lmmc/init.h"
#include "lmmc/dense.h"
#include "lmmc/stats.h"
#include "lmmc/random.h"
#include "lmmc/lsr_stdlib.h"

LmState global_state;

namespace {

class LmmcRuntimeLifetime {
public:
    LmmcRuntimeLifetime() noexcept { lmmc_init(); }
    ~LmmcRuntimeLifetime() noexcept { lmmc_deinit(); }

    LmmcRuntimeLifetime(const LmmcRuntimeLifetime&) = delete;
    LmmcRuntimeLifetime& operator=(const LmmcRuntimeLifetime&) = delete;
};

const LmmcRuntimeLifetime lmmc_runtime_lifetime;

using lmx::runtime::ExprObj;
using lmx::runtime::StringObj;
using lmx::runtime::AdtObj;
using lmx::runtime::ComplexObj;
using lmx::runtime::ArrayObj;
using lmx::runtime::VectorObj;
using lmx::runtime::MatrixObj;
using lmx::runtime::TableObj;
using lmx::runtime::RandomObj;
using lmx::runtime::QuantityObj;
using lmx::runtime::Value;
using lmx::runtime::ValueKind;

bool debug_dump_enabled() noexcept {
    const char* value = std::getenv("LMX_DEBUG_DUMP");
    return value && value[0] != '\0' && value[0] != '0';
}

std::string cas_error_text(const lamina::CasError& error) {
    std::string result = "CasError(";
    result += lamina::lsr::error_name(error);
    if (!error.operation.empty()) {
        result += " in ";
        result += error.operation;
    }
    if (!error.message.empty()) {
        result += ": ";
        result += error.message;
    }
    result += ")";
    return result;
}

ExprObj* expr_error(std::string message) {
    return new ExprObj(std::move(message));
}

ExprObj* expr_from_result(const lamina::lsr::ExprResult& result) {
    if (!result) return expr_error(cas_error_text(result.error()));
    return new ExprObj(result.value());
}

const lamina::lsr::ExprPtr* checked_expr(ExprObj* expr, std::string& error);

lamina::lsr::ExprResult invalid_expr_operation(const std::string& message,
                                               const char* operation) {
    return lamina::lsr::ExprResult::failure(
        lamina::CasErrc::InvalidArgument, message, operation);
}

bool collect_expr_arguments(va_list& args, const LmInt count,
                            std::vector<lamina::lsr::ExprPtr>& values,
                            std::string& error) {
    if (count < 0 || count > 65535) {
        error = "CasError(InvalidArgument: invalid Expr argument count)";
        return false;
    }
    values.reserve(static_cast<std::size_t>(count));
    for (LmInt i = 0; i < count; ++i) {
        auto* object = va_arg(args, ExprObj*);
        const auto* value = checked_expr(object, error);
        if (!value) return false;
        values.push_back(*value);
    }
    return true;
}

const lamina::lsr::ExprPtr* checked_expr(ExprObj* expr, std::string& error) {
    if (!expr) {
        error = "CasError(InvalidArgument: null expr)";
        return nullptr;
    }
    if (!expr->ok()) {
        error = expr->error();
        return nullptr;
    }
    return &expr->expr();
}

AdtObj* real_result_ok(const double value) {
    std::vector<lmx::runtime::Value> fields;
    fields.emplace_back(value);
    return new AdtObj("Result", "Ok", std::move(fields));
}

AdtObj* int_result_ok(const LmInt value) {
    std::vector<Value> fields;
    fields.emplace_back(value);
    return new AdtObj("Result", "Ok", std::move(fields));
}

AdtObj* bool_result_ok(const bool value) {
    std::vector<Value> fields;
    fields.emplace_back(value);
    return new AdtObj("Result", "Ok", std::move(fields));
}

AdtObj* real_result_error(std::string error) {
    std::vector<lmx::runtime::Value> fields;
    fields.emplace_back(static_cast<lmx::runtime::Object*>(
        new StringObj(std::move(error))));
    return new AdtObj("Result", "Err", std::move(fields));
}

AdtObj* object_result_ok(lmx::runtime::Object* value, const ValueKind kind) {
    std::vector<Value> fields;
    fields.emplace_back(value, kind);
    return new AdtObj("Result", "Ok", std::move(fields));
}

AdtObj* lmmc_object_error(const char* operation, const lmmc_status_t status) {
    std::string error = operation ? operation : "LMMC";
    error += ": ";
    error += lmmc_status_string(status);
    return real_result_error(std::move(error));
}

bool numeric_value(const Value& value, double& result) noexcept {
    switch (value.kind) {
    case ValueKind::Int: result = static_cast<double>(value.int_val); return true;
    case ValueKind::Fraction: result = value.frac_val.to_float(); return true;
    case ValueKind::Real: result = value.real_val; return true;
    default: return false;
    }
}

bool array_numbers(const ArrayObj* array, std::vector<double>& result,
                   std::string& error) {
    if (!array) {
        error = "null array";
        return false;
    }
    result.reserve(static_cast<std::size_t>(array->len()));
    for (const auto& value : array->values()) {
        double number = 0.0;
        if (!numeric_value(value, number)) {
            error = "array contains a non-numeric value";
            return false;
        }
        result.push_back(number);
    }
    return true;
}

lmmc_vec_t vector_view(VectorObj* value) noexcept {
    return {value ? value->size() : 0,
            value && !value->data().empty() ? value->data().data() : nullptr, 0};
}

lmmc_mat_t matrix_view(MatrixObj* value) noexcept {
    return {value ? value->rows() : 0, value ? value->cols() : 0,
            value ? value->cols() : 0,
            value && !value->data().empty() ? value->data().data() : nullptr, 0};
}

bool unit_power_expression(const std::string& unit, const int multiplier,
                           std::string& result) {
    if (unit == "1") {
        result = "1";
        return true;
    }
    std::size_t cursor = 0;
    int operation_sign = 1;
    result.clear();
    while (cursor < unit.size()) {
        const auto begin = cursor;
        while (cursor < unit.size() &&
               (std::isalpha(static_cast<unsigned char>(unit[cursor])) ||
                unit[cursor] == '_')) ++cursor;
        if (cursor == begin) return false;
        const auto name = unit.substr(begin, cursor - begin);
        int exponent = 1;
        if (cursor < unit.size() && unit[cursor] == '^') {
            ++cursor;
            int sign = 1;
            if (cursor < unit.size() && (unit[cursor] == '+' || unit[cursor] == '-')) {
                if (unit[cursor] == '-') sign = -1;
                ++cursor;
            }
            if (cursor == unit.size() ||
                !std::isdigit(static_cast<unsigned char>(unit[cursor]))) return false;
            exponent = 0;
            while (cursor < unit.size() &&
                   std::isdigit(static_cast<unsigned char>(unit[cursor]))) {
                exponent = exponent * 10 + (unit[cursor++] - '0');
            }
            exponent *= sign;
        }
        exponent *= operation_sign * multiplier;
        if (exponent != 0) {
            if (!result.empty()) result += '*';
            result += name;
            if (exponent != 1) result += '^' + std::to_string(exponent);
        }
        if (cursor == unit.size()) break;
        if (unit[cursor] == '*') operation_sign = 1;
        else if (unit[cursor] == '/') operation_sign = -1;
        else return false;
        ++cursor;
    }
    if (result.empty()) result = "1";
    return true;
}

bool unit_product_expression(const std::string& lhs, const std::string& rhs,
                             const bool divide, std::string& result) {
    std::string right;
    if (!unit_power_expression(rhs, divide ? -1 : 1, right)) return false;
    if (lhs == "1") result = right;
    else if (right == "1") result = lhs;
    else result = lhs + '*' + right;
    int ignored = 0;
    return lmmc_lsr_units_is_dimensionless(result.c_str(), &ignored) == LMMC_STATUS_OK;
}

AdtObj* quantity_result(const double si_value, std::string unit,
                        const char* operation) {
    if (!std::isfinite(si_value)) {
        return real_result_error(std::string(operation) + ": numerical failure");
    }
    int ignored = 0;
    const auto status = lmmc_lsr_units_is_dimensionless(unit.c_str(), &ignored);
    if (status != LMMC_STATUS_OK) return lmmc_object_error(operation, status);
    return object_result_ok(new QuantityObj(si_value, std::move(unit)),
                            ValueKind::Quantity);
}

AdtObj* lmmc_real_result(const char* operation, lmmc_status_t status,
                         lmmc_real_t value);

template <typename Operation>
AdtObj* vector_stat_result(const char* name, VectorObj* value, Operation operation) {
    if (!value) return real_result_error(std::string(name) + ": null vector");
    auto view = vector_view(value);
    lmmc_real_t result = 0.0;
    const auto status = operation(&view, &result);
    return lmmc_real_result(name, status, result);
}

AdtObj* lmmc_real_result(const char* operation, const lmmc_status_t status,
                         const lmmc_real_t value) {
    if (status == LMMC_STATUS_OK) return real_result_ok(value);
    std::string error = operation ? operation : "LMMC";
    error += ": ";
    error += lmmc_status_string(status);
    return real_result_error(std::move(error));
}

AdtObj* complex_result_ok(const lmmc_complex_t& value) {
    std::vector<lmx::runtime::Value> fields;
    fields.emplace_back(static_cast<lmx::runtime::Object*>(
        new ComplexObj(value.real, value.imag)),
        lmx::runtime::ValueKind::Complex);
    return new AdtObj("Result", "Ok", std::move(fields));
}

AdtObj* lmmc_complex_result(const char* operation,
                            const lmmc_status_t status,
                            const lmmc_complex_t& value) {
    if (status == LMMC_STATUS_OK) return complex_result_ok(value);
    std::string error = operation ? operation : "LMMC";
    error += ": ";
    error += lmmc_status_string(status);
    return real_result_error(std::move(error));
}

bool checked_complex(ComplexObj* value, lmmc_complex_t& result) noexcept {
    if (!value) return false;
    result.real = value->real();
    result.imag = value->imag();
    return true;
}

bool expr_to_real(ExprObj* expr, double& result, std::string& error) {
    const auto* value = checked_expr(expr, error);
    if (!value) return false;
    const auto evaluated = lamina::lsr::evalf(**value);
    if (!evaluated) {
        error = cas_error_text(evaluated.error());
        return false;
    }
    result = evaluated.value().value;
    return true;
}

} // namespace

extern "C" LM_API int lmx_printf(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    const int result = vprintf(fmt ? fmt : "", args);
    va_end(args);
    return result;
}

extern "C" LM_API ExprObj* cas_sym(const char* name) {
    return expr_from_result(lamina::lsr::sym(name ? name : ""));
}

extern "C" LM_API ExprObj* cas_parse(const char* source) {
    return expr_from_result(lamina::lsr::parse_expr(source ? source : ""));
}

extern "C" LM_API ExprObj* cas_expr_imaginary() {
    return expr_from_result(lamina::lsr::imaginary_unit());
}

extern "C" LM_API ExprObj* cas_expr_integer(const LmInt value) {
    return expr_from_result(lamina::lsr::integer(value));
}

extern "C" LM_API ExprObj* cas_expr_rational(const LmInt numerator,
                                               const LmInt denominator) {
    if (denominator == 0) return expr_error("CasError(DivisionByZero: rational denominator is zero)");
    return expr_from_result(lamina::lsr::rational(Rational(
        BigInt(std::to_string(numerator)), BigInt(std::to_string(denominator)))));
}

extern "C" LM_API ExprObj* cas_expr_value(const lmx::runtime::Value* value) {
    if (!value) return expr_error("CasError(InvalidArgument: null Lamina value)");
    switch (value->kind) {
    case lmx::runtime::ValueKind::Int:
        return expr_from_result(lamina::lsr::integer(value->int_val));
    case lmx::runtime::ValueKind::Fraction:
        return expr_from_result(lamina::lsr::rational(Rational(
            value->frac_val.num, value->frac_val.den)));
    case lmx::runtime::ValueKind::Real:
        return expr_from_result(lamina::lsr::approx_real(value->real_val));
    case lmx::runtime::ValueKind::Expr: {
        std::string error;
        const auto* expression = checked_expr(
            reinterpret_cast<ExprObj*>(value->obj), error);
        return expression ? new ExprObj(*expression) : expr_error(std::move(error));
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
}

extern "C" LM_API ExprObj* cas_expr_unary(const LmInt operation,
                                            ExprObj* operand) {
    std::string error;
    const auto* value = checked_expr(operand, error);
    if (!value) return expr_error(std::move(error));
    switch (operation) {
    case LMX_EXPR_NEG:
        return expr_from_result(lamina::lsr::neg(*value));
    case LMX_EXPR_NOT:
        return expr_from_result(lamina::lsr::logical_not(*value));
    default:
        return expr_from_result(invalid_expr_operation(
            "unknown unary Expr operation", "runtime.expr_unary"));
    }
}

extern "C" LM_API ExprObj* cas_expr_binary(const LmInt operation,
                                             ExprObj* lhs, ExprObj* rhs) {
    std::string error;
    const auto* left = checked_expr(lhs, error);
    if (!left) return expr_error(std::move(error));
    const auto* right = checked_expr(rhs, error);
    if (!right) return expr_error(std::move(error));
    switch (operation) {
    case LMX_EXPR_ADD: return expr_from_result(lamina::lsr::add(*left, *right));
    case LMX_EXPR_SUB: return expr_from_result(lamina::lsr::sub(*left, *right));
    case LMX_EXPR_MUL: return expr_from_result(lamina::lsr::mul(*left, *right));
    case LMX_EXPR_DIV: return expr_from_result(lamina::lsr::div(*left, *right));
    case LMX_EXPR_POW: return expr_from_result(lamina::lsr::pow(*left, *right));
    case LMX_EXPR_EQ: return expr_from_result(lamina::lsr::relation(*left, *right, lamina::RelationOp::EQ));
    case LMX_EXPR_NE: return expr_from_result(lamina::lsr::relation(*left, *right, lamina::RelationOp::NEQ));
    case LMX_EXPR_GT: return expr_from_result(lamina::lsr::relation(*left, *right, lamina::RelationOp::GT));
    case LMX_EXPR_GE: return expr_from_result(lamina::lsr::relation(*left, *right, lamina::RelationOp::GEQ));
    case LMX_EXPR_LT: return expr_from_result(lamina::lsr::relation(*left, *right, lamina::RelationOp::LT));
    case LMX_EXPR_LE: return expr_from_result(lamina::lsr::relation(*left, *right, lamina::RelationOp::LEQ));
    case LMX_EXPR_AND: return expr_from_result(lamina::lsr::logical_and(*left, *right));
    case LMX_EXPR_OR: return expr_from_result(lamina::lsr::logical_or(*left, *right));
    case LMX_EXPR_IN: return expr_from_result(lamina::lsr::membership(*left, *right));
    case LMX_EXPR_NOT_IN: return expr_from_result(lamina::lsr::membership(*left, *right, true));
    default:
        return expr_from_result(invalid_expr_operation(
            "unknown binary Expr operation", "runtime.expr_binary"));
    }
}

extern "C" LM_API ExprObj* cas_expr_function(const char* name,
                                               const LmInt count, ...) {
    va_list args;
    va_start(args, count);
    std::vector<lamina::lsr::ExprPtr> values;
    std::string error;
    const bool valid = collect_expr_arguments(args, count, values, error);
    va_end(args);
    if (!valid) return expr_error(std::move(error));
    return expr_from_result(lamina::lsr::function(name ? name : "", std::move(values)));
}

extern "C" LM_API ExprObj* cas_expr_set(const LmInt count, ...) {
    va_list args;
    va_start(args, count);
    std::vector<lamina::lsr::ExprPtr> values;
    std::string error;
    const bool valid = collect_expr_arguments(args, count, values, error);
    va_end(args);
    if (!valid) return expr_error(std::move(error));
    return expr_from_result(lamina::lsr::finite_set(std::move(values)));
}

extern "C" LM_API ExprObj* cas_expr_interval(ExprObj* lower, ExprObj* upper,
                                               const bool lower_closed,
                                               const bool upper_closed) {
    std::string error;
    const auto* lower_value = checked_expr(lower, error);
    if (!lower_value) return expr_error(std::move(error));
    const auto* upper_value = checked_expr(upper, error);
    if (!upper_value) return expr_error(std::move(error));
    return expr_from_result(lamina::lsr::interval(
        *lower_value, *upper_value, lower_closed, upper_closed));
}

extern "C" LM_API ExprObj* cas_simplify(ExprObj* expr) {
    std::string error;
    const auto* value = checked_expr(expr, error);
    if (!value) return expr_error(std::move(error));
    return expr_from_result(lamina::lsr::simplify(*value));
}

extern "C" LM_API ExprObj* cas_expand(ExprObj* expr) {
    std::string error;
    const auto* value = checked_expr(expr, error);
    if (!value) return expr_error(std::move(error));
    return expr_from_result(lamina::lsr::expand(*value));
}

extern "C" LM_API ExprObj* cas_diff(ExprObj* expr, const char* variable) {
    std::string error;
    const auto* value = checked_expr(expr, error);
    if (!value) return expr_error(std::move(error));
    return expr_from_result(lamina::lsr::differentiate(*value, variable ? variable : ""));
}

extern "C" LM_API ExprObj* cas_substitute(ExprObj* expr, AdtObj* binding) {
    std::string error;
    const auto* value = checked_expr(expr, error);
    if (!value) return expr_error(std::move(error));
    if (!binding || binding->type_name() != "Binding" ||
        binding->constructor() != "Binding" || binding->fields().size() != 2) {
        return expr_error("cas_substitute expects Binding<Expr, Expr>");
    }
    const auto* symbol_field = binding->field(0);
    const auto* value_field = binding->field(1);
    if (!symbol_field || symbol_field->kind != lmx::runtime::ValueKind::Expr ||
        !value_field || value_field->kind != lmx::runtime::ValueKind::Expr) {
        return expr_error("cas_substitute expects Binding<Expr, Expr>");
    }
    const auto* symbol = checked_expr(
        reinterpret_cast<ExprObj*>(symbol_field->obj), error);
    if (!symbol) return expr_error(std::move(error));
    const auto* replacement = checked_expr(
        reinterpret_cast<ExprObj*>(value_field->obj), error);
    if (!replacement) return expr_error(std::move(error));
    const auto checked_binding = lamina::lsr::binding(*symbol, *replacement);
    if (!checked_binding) return expr_error(cas_error_text(checked_binding.error()));
    return expr_from_result(lamina::lsr::substitute(*value, checked_binding.value()));
}

extern "C" LM_API AdtObj* cas_evalf(ExprObj* expr) {
    std::string error;
    const auto* value = checked_expr(expr, error);
    if (!value) return real_result_error(std::move(error));
    const auto result = lamina::lsr::evalf(**value);
    if (!result) return real_result_error(cas_error_text(result.error()));
    return real_result_ok(result.value().value);
}

extern "C" LM_API bool cas_is_ok(ExprObj* expr) {
    return expr && expr->ok();
}

extern "C" LM_API StringObj* cas_to_text(ExprObj* expr) {
    if (!expr) return new StringObj("CasError(InvalidArgument: null expr)");
    return new StringObj(expr->to_string());
}

extern "C" LM_API StringObj* cas_error(ExprObj* expr) {
    if (!expr) return new StringObj("CasError(InvalidArgument: null expr)");
    if (expr->ok()) return new StringObj("");
    return new StringObj(expr->error());
}

extern "C" LM_API AdtObj* lmmc_num_hypot(ExprObj* lhs, ExprObj* rhs) {
    double x = 0.0;
    double y = 0.0;
    std::string error;
    if (!expr_to_real(lhs, x, error)) return real_result_error(std::move(error));
    if (!expr_to_real(rhs, y, error)) return real_result_error(std::move(error));
    lmmc_real_t out = 0.0;
    const auto status = lmmc_hypot(x, y, &out);
    return lmmc_real_result("lmmc_num_hypot", status, out);
}

extern "C" LM_API AdtObj* lmmc_num_log2(ExprObj* expr) {
    double x = 0.0;
    std::string error;
    if (!expr_to_real(expr, x, error)) return real_result_error(std::move(error));
    lmmc_real_t out = 0.0;
    const auto status = lmmc_log2(x, &out);
    return lmmc_real_result("lmmc_num_log2", status, out);
}

extern "C" LM_API AdtObj* lmmc_num_exp2(ExprObj* expr) {
    double x = 0.0;
    std::string error;
    if (!expr_to_real(expr, x, error)) return real_result_error(std::move(error));
    lmmc_real_t out = 0.0;
    const auto status = lmmc_exp2(x, &out);
    return lmmc_real_result("lmmc_num_exp2", status, out);
}

extern "C" LM_API ComplexObj* lmx_complex_make(const double real,
                                                  const double imag) {
    lmmc_complex_t value{};
    const auto status = lmmc_complex_create(real, imag, &value);
    if (status != LMMC_STATUS_OK) return nullptr;
    return new ComplexObj(value.real, value.imag);
}

extern "C" LM_API double lmx_complex_real(ComplexObj* value) {
    return value ? value->real() : 0.0;
}

extern "C" LM_API double lmx_complex_imag(ComplexObj* value) {
    return value ? value->imag() : 0.0;
}

extern "C" LM_API AdtObj* lmx_complex_add(ComplexObj* lhs,
                                             ComplexObj* rhs) {
    lmmc_complex_t left{}, right{}, result{};
    if (!checked_complex(lhs, left) || !checked_complex(rhs, right)) {
        return real_result_error("lmx_complex_add: invalid argument");
    }
    const auto status = lmmc_complex_add(&left, &right, &result);
    return lmmc_complex_result("lmx_complex_add", status, result);
}

extern "C" LM_API AdtObj* lmx_complex_sub(ComplexObj* lhs,
                                             ComplexObj* rhs) {
    lmmc_complex_t left{}, right{}, result{};
    if (!checked_complex(lhs, left) || !checked_complex(rhs, right)) {
        return real_result_error("lmx_complex_sub: invalid argument");
    }
    const auto status = lmmc_complex_sub(&left, &right, &result);
    return lmmc_complex_result("lmx_complex_sub", status, result);
}

extern "C" LM_API AdtObj* lmx_complex_mul(ComplexObj* lhs,
                                             ComplexObj* rhs) {
    lmmc_complex_t left{}, right{}, result{};
    if (!checked_complex(lhs, left) || !checked_complex(rhs, right)) {
        return real_result_error("lmx_complex_mul: invalid argument");
    }
    const auto status = lmmc_complex_mul(&left, &right, &result);
    return lmmc_complex_result("lmx_complex_mul", status, result);
}

extern "C" LM_API AdtObj* lmx_complex_div(ComplexObj* lhs,
                                             ComplexObj* rhs) {
    lmmc_complex_t left{}, right{}, result{};
    if (!checked_complex(lhs, left) || !checked_complex(rhs, right)) {
        return real_result_error("lmx_complex_div: invalid argument");
    }
    const auto status = lmmc_complex_div(&left, &right, &result);
    return lmmc_complex_result("lmx_complex_div", status, result);
}

extern "C" LM_API AdtObj* lmx_complex_conj(ComplexObj* value) {
    lmmc_complex_t input{}, result{};
    if (!checked_complex(value, input)) {
        return real_result_error("lmx_complex_conj: invalid argument");
    }
    const auto status = lmmc_complex_conj(&input, &result);
    return lmmc_complex_result("lmx_complex_conj", status, result);
}

extern "C" LM_API AdtObj* lmx_complex_abs(ComplexObj* value) {
    lmmc_complex_t input{};
    if (!checked_complex(value, input)) {
        return real_result_error("lmx_complex_abs: invalid argument");
    }
    lmmc_real_t result = 0.0;
    const auto status = lmmc_complex_modulus(&input, &result);
    return lmmc_real_result("lmx_complex_abs", status, result);
}

extern "C" LM_API AdtObj* lmx_vector_from_array(ArrayObj* values) {
    std::vector<double> data;
    std::string error;
    if (!array_numbers(values, data, error)) {
        return real_result_error("vector: " + error);
    }
    if (data.empty()) return real_result_error("vector: empty vectors are not supported by LMMC");
    return object_result_ok(new VectorObj(std::move(data)), ValueKind::Vector);
}

extern "C" LM_API AdtObj* lmx_matrix_from_array(ArrayObj* rows) {
    if (!rows || rows->values().empty()) {
        return real_result_error("matrix: expected at least one row");
    }
    std::vector<double> data;
    std::size_t column_count = 0;
    for (const auto& row_value : rows->values()) {
        if (row_value.kind != ValueKind::Obj || !row_value.obj ||
            row_value.obj->get_kind() != lmx::runtime::ObjectKind::Array) {
            return real_result_error("matrix: each row must be an array");
        }
        std::vector<double> row;
        std::string error;
        if (!array_numbers(reinterpret_cast<ArrayObj*>(row_value.obj), row, error)) {
            return real_result_error("matrix: " + error);
        }
        if (column_count == 0) column_count = row.size();
        if (row.empty() || row.size() != column_count) {
            return real_result_error("matrix: rows must be non-empty and rectangular");
        }
        data.insert(data.end(), row.begin(), row.end());
    }
    return object_result_ok(new MatrixObj(rows->values().size(), column_count,
                                          std::move(data)), ValueKind::Matrix);
}

extern "C" LM_API LmInt lmx_vector_size(VectorObj* value) {
    return value ? static_cast<LmInt>(value->size()) : 0;
}

extern "C" LM_API AdtObj* lmx_vector_at(VectorObj* value, const LmInt index) {
    if (!value || index < 0 || static_cast<std::size_t>(index) >= value->size()) {
        return real_result_error("vector_at: index out of bounds");
    }
    return real_result_ok(value->data()[static_cast<std::size_t>(index)]);
}

extern "C" LM_API LmInt lmx_matrix_rows(MatrixObj* value) {
    return value ? static_cast<LmInt>(value->rows()) : 0;
}

extern "C" LM_API LmInt lmx_matrix_cols(MatrixObj* value) {
    return value ? static_cast<LmInt>(value->cols()) : 0;
}

extern "C" LM_API AdtObj* lmx_matrix_at(MatrixObj* value, const LmInt row,
                                           const LmInt column) {
    if (!value || row < 0 || column < 0 ||
        static_cast<std::size_t>(row) >= value->rows() ||
        static_cast<std::size_t>(column) >= value->cols()) {
        return real_result_error("matrix_at: index out of bounds");
    }
    return real_result_ok(value->data()[static_cast<std::size_t>(row) * value->cols() +
                                        static_cast<std::size_t>(column)]);
}

extern "C" LM_API LmInt lmx_table_size(TableObj* value) {
    return value ? static_cast<LmInt>(value->entries().size()) : 0;
}

extern "C" LM_API bool lmx_table_has(TableObj* value, const char* key) {
    return value && key && value->find(key) != nullptr;
}

extern "C" LM_API AdtObj* lmx_table_vector(TableObj* value, const char* key) {
    if (!value || !key) return real_result_error("table_vector: invalid argument");
    const auto* field = value->find(key);
    if (!field || field->kind != ValueKind::Vector || !field->obj) {
        return real_result_error("table_vector: key is missing or is not a vector");
    }
    return object_result_ok(field->obj->get(), ValueKind::Vector);
}

extern "C" LM_API AdtObj* lmx_vector_dot(VectorObj* lhs, VectorObj* rhs) {
    if (!lhs || !rhs) return real_result_error("vector_dot: null vector");
    auto left = vector_view(lhs);
    auto right = vector_view(rhs);
    lmmc_real_t result = 0.0;
    const auto status = lmmc_vec_dot(&left, &right, &result);
    return lmmc_real_result("vector_dot", status, result);
}

extern "C" LM_API AdtObj* lmx_vector_norm(VectorObj* value) {
    return vector_stat_result("vector_norm", value, lmmc_vec_norm2);
}

extern "C" LM_API AdtObj* lmx_matrix_norm(MatrixObj* value) {
    if (!value || !value->valid()) return real_result_error("matrix_norm: invalid matrix");
    auto view = matrix_view(value);
    lmmc_real_t result = 0.0;
    const auto status = lmmc_mat_norm_fro(&view, &result);
    return lmmc_real_result("matrix_norm", status, result);
}

extern "C" LM_API AdtObj* lmx_matrix_transpose(MatrixObj* value) {
    if (!value || !value->valid()) return real_result_error("matrix_transpose: invalid matrix");
    auto input = matrix_view(value);
    auto* result = new MatrixObj(value->cols(), value->rows(),
                                 std::vector<double>(value->rows() * value->cols()));
    auto output = matrix_view(result);
    const auto status = lmmc_mat_transpose_to(&input, &output);
    if (status != LMMC_STATUS_OK) {
        delete result;
        return lmmc_object_error("matrix_transpose", status);
    }
    return object_result_ok(result, ValueKind::Matrix);
}

extern "C" LM_API AdtObj* lmx_matrix_mul(MatrixObj* lhs, MatrixObj* rhs) {
    if (!lhs || !rhs || !lhs->valid() || !rhs->valid()) {
        return real_result_error("matrix_mul: invalid matrix");
    }
    if (lhs->cols() != rhs->rows()) {
        return real_result_error("matrix_mul: dimension mismatch");
    }
    auto left = matrix_view(lhs);
    auto right = matrix_view(rhs);
    auto* result = new MatrixObj(lhs->rows(), rhs->cols(),
                                 std::vector<double>(lhs->rows() * rhs->cols()));
    auto output = matrix_view(result);
    const auto status = lmmc_mat_mul(&left, &right, &output);
    if (status != LMMC_STATUS_OK) {
        delete result;
        return lmmc_object_error("matrix_mul", status);
    }
    return object_result_ok(result, ValueKind::Matrix);
}

extern "C" LM_API AdtObj* lmx_matrix_vector_mul(MatrixObj* matrix, VectorObj* vector) {
    if (!matrix || !vector || !matrix->valid()) {
        return real_result_error("matrix_vector_mul: invalid argument");
    }
    if (matrix->cols() != vector->size()) {
        return real_result_error("matrix_vector_mul: dimension mismatch");
    }
    auto input_matrix = matrix_view(matrix);
    auto input_vector = vector_view(vector);
    auto* result = new VectorObj(std::vector<double>(matrix->rows()));
    auto output = vector_view(result);
    const auto status = lmmc_mat_vec_mul(&input_matrix, &input_vector, &output);
    if (status != LMMC_STATUS_OK) {
        delete result;
        return lmmc_object_error("matrix_vector_mul", status);
    }
    return object_result_ok(result, ValueKind::Vector);
}

extern "C" LM_API AdtObj* lmx_stats_mean(VectorObj* value) {
    return vector_stat_result("stats_mean", value, lmmc_vec_mean);
}

extern "C" LM_API AdtObj* lmx_stats_variance(VectorObj* value) {
    return vector_stat_result("stats_variance", value, lmmc_vec_variance_sample);
}

extern "C" LM_API AdtObj* lmx_stats_variance_population(VectorObj* value) {
    return vector_stat_result("stats_variance_population", value,
                              lmmc_vec_variance_population);
}

extern "C" LM_API AdtObj* lmx_stats_stddev(VectorObj* value) {
    return vector_stat_result("stats_stddev", value, lmmc_vec_stddev_sample);
}

extern "C" LM_API AdtObj* lmx_stats_stddev_population(VectorObj* value) {
    return vector_stat_result("stats_stddev_population", value,
                              lmmc_vec_stddev_population);
}

extern "C" LM_API AdtObj* lmx_stats_median(VectorObj* value) {
    return vector_stat_result("stats_median", value, lmmc_vec_median);
}

extern "C" LM_API AdtObj* lmx_stats_quantile(VectorObj* value, const double p) {
    if (!value) return real_result_error("stats_quantile: null vector");
    auto input = vector_view(value);
    lmmc_real_t result = 0.0;
    const auto status = lmmc_vec_quantile(&input, p, &result);
    return lmmc_real_result("stats_quantile", status, result);
}

extern "C" LM_API AdtObj* lmx_stats_covariance(VectorObj* lhs, VectorObj* rhs) {
    if (!lhs || !rhs) return real_result_error("stats_covariance: null vector");
    auto left = vector_view(lhs);
    auto right = vector_view(rhs);
    lmmc_real_t result = 0.0;
    const auto status = lmmc_vec_covariance_sample(&left, &right, &result);
    return lmmc_real_result("stats_covariance", status, result);
}

extern "C" LM_API AdtObj* lmx_stats_correlation(VectorObj* lhs, VectorObj* rhs) {
    if (!lhs || !rhs) return real_result_error("stats_correlation: null vector");
    auto left = vector_view(lhs);
    auto right = vector_view(rhs);
    lmmc_real_t result = 0.0;
    const auto status = lmmc_vec_correlation_sample(&left, &right, &result);
    return lmmc_real_result("stats_correlation", status, result);
}

extern "C" LM_API AdtObj* lmx_stats_histogram(VectorObj* value, const LmInt bins) {
    if (!value || bins <= 0) return real_result_error("stats_histogram: invalid argument");
    auto input = vector_view(value);
    std::vector<double> edges(static_cast<std::size_t>(bins) + 1);
    std::vector<std::size_t> counts(static_cast<std::size_t>(bins));
    const auto status = lmmc_vec_histogram(&input, static_cast<std::size_t>(bins),
                                           edges.data(), counts.data());
    if (status != LMMC_STATUS_OK) return lmmc_object_error("stats_histogram", status);
    std::vector<double> count_values;
    count_values.reserve(counts.size());
    for (const auto count : counts) count_values.push_back(static_cast<double>(count));
    std::vector<TableObj::Entry> entries;
    entries.emplace_back("counts", Value(new VectorObj(std::move(count_values)), ValueKind::Vector));
    entries.emplace_back("edges", Value(new VectorObj(std::move(edges)), ValueKind::Vector));
    return object_result_ok(new TableObj(std::move(entries)), ValueKind::Table);
}

extern "C" LM_API AdtObj* lmx_rng_create(const LmInt seed) {
    lmmc_rng_t* rng = nullptr;
    auto status = lmmc_rng_create(&rng);
    if (status == LMMC_STATUS_OK) {
        status = lmmc_rng_seed(rng, static_cast<std::uint64_t>(seed));
    }
    if (status != LMMC_STATUS_OK) {
        lmmc_rng_destroy(rng);
        return lmmc_object_error("rng", status);
    }
    return object_result_ok(new RandomObj(rng), ValueKind::Random);
}

extern "C" LM_API AdtObj* lmx_rng_clone(RandomObj* source) {
    if (!source) return real_result_error("rng_clone: null rng");
    lmmc_rng_t* rng = nullptr;
    const auto status = lmmc_rng_clone(source->handle(), &rng);
    if (status != LMMC_STATUS_OK) return lmmc_object_error("rng_clone", status);
    return object_result_ok(new RandomObj(rng), ValueKind::Random);
}

extern "C" LM_API AdtObj* lmx_rng_jump(RandomObj* value) {
    if (!value) return real_result_error("rng_jump: null rng");
    const auto status = lmmc_rng_jump(value->handle());
    if (status != LMMC_STATUS_OK) return lmmc_object_error("rng_jump", status);
    return object_result_ok(value->get(), ValueKind::Random);
}

extern "C" LM_API AdtObj* lmx_rng_uniform(RandomObj* value, const double lower,
                                             const double upper) {
    if (!value) return real_result_error("random_uniform: null rng");
    lmmc_real_t result = 0.0;
    const auto status = lmmc_rng_uniform(value->handle(), lower, upper, &result);
    return lmmc_real_result("random_uniform", status, result);
}

extern "C" LM_API AdtObj* lmx_rng_normal(RandomObj* value, const double mean,
                                            const double stddev) {
    if (!value) return real_result_error("random_normal: null rng");
    lmmc_real_t result = 0.0;
    const auto status = lmmc_rng_normal(value->handle(), mean, stddev, &result);
    return lmmc_real_result("random_normal", status, result);
}

extern "C" LM_API AdtObj* lmx_rng_exponential(RandomObj* value, const double rate) {
    if (!value) return real_result_error("random_exponential: null rng");
    lmmc_real_t result = 0.0;
    const auto status = lmmc_rng_exponential(value->handle(), rate, &result);
    return lmmc_real_result("random_exponential", status, result);
}

extern "C" LM_API AdtObj* lmx_rng_int(RandomObj* value, const LmInt lower,
                                         const LmInt upper) {
    if (!value) return real_result_error("random_int: null rng");
    std::int64_t result = 0;
    const auto status = lmmc_rng_int_uniform(value->handle(), lower, upper, &result);
    if (status != LMMC_STATUS_OK) return lmmc_object_error("random_int", status);
    return int_result_ok(static_cast<LmInt>(result));
}

extern "C" LM_API AdtObj* lmx_rng_vector(RandomObj* value, const LmInt count,
                                            const double lower, const double upper) {
    if (!value || count <= 0) return real_result_error("random_vector: invalid argument");
    std::vector<double> data(static_cast<std::size_t>(count));
    const auto status = lmmc_rng_fill_uniform(value->handle(), lower, upper,
                                               data.data(), data.size());
    if (status != LMMC_STATUS_OK) return lmmc_object_error("random_vector", status);
    return object_result_ok(new VectorObj(std::move(data)), ValueKind::Vector);
}

extern "C" LM_API AdtObj* lmx_quantity_make(const double value, const char* unit) {
    if (!unit) return real_result_error("quantity: null unit");
    lmmc_real_t si_value = 0.0;
    const auto status = lmmc_lsr_units_strip_num(value, unit, &si_value);
    if (status != LMMC_STATUS_OK) return lmmc_object_error("quantity", status);
    return quantity_result(si_value, unit, "quantity");
}

extern "C" LM_API AdtObj* lmx_quantity_convert(QuantityObj* value,
                                                  const char* target_unit) {
    if (!value || !target_unit) return real_result_error("quantity_convert: invalid argument");
    lmmc_real_t ignored = 0.0;
    const auto status = lmmc_lsr_units_convert(1.0, value->unit().c_str(),
                                               target_unit, &ignored);
    if (status != LMMC_STATUS_OK) return lmmc_object_error("quantity_convert", status);
    return quantity_result(value->si_value(), target_unit, "quantity_convert");
}

extern "C" LM_API AdtObj* lmx_quantity_value(QuantityObj* value) {
    if (!value) return real_result_error("quantity_value: null quantity");
    lmmc_real_t displayed = 0.0;
    const auto status = lmmc_lsr_units_convert_from_si(
        value->si_value(), value->unit().c_str(), &displayed);
    return lmmc_real_result("quantity_value", status, displayed);
}

extern "C" LM_API double lmx_quantity_strip(QuantityObj* value) {
    return value ? value->si_value() : 0.0;
}

extern "C" LM_API StringObj* lmx_quantity_unit(QuantityObj* value) {
    return new StringObj(value ? value->unit() : "");
}

extern "C" LM_API AdtObj* lmx_quantity_is_dimensionless(QuantityObj* value) {
    if (!value) return real_result_error("quantity_is_dimensionless: null quantity");
    int result = 0;
    const auto status = lmmc_lsr_units_is_dimensionless(value->unit().c_str(), &result);
    if (status != LMMC_STATUS_OK) {
        return lmmc_object_error("quantity_is_dimensionless", status);
    }
    return bool_result_ok(result != 0);
}

extern "C" LM_API AdtObj* lmx_quantity_add(QuantityObj* lhs, QuantityObj* rhs) {
    if (!lhs || !rhs) return real_result_error("quantity_add: null quantity");
    lmmc_real_t ignored = 0.0;
    const auto status = lmmc_lsr_units_convert(1.0, rhs->unit().c_str(),
                                               lhs->unit().c_str(), &ignored);
    if (status != LMMC_STATUS_OK) return lmmc_object_error("quantity_add", status);
    return quantity_result(lhs->si_value() + rhs->si_value(), lhs->unit(), "quantity_add");
}

extern "C" LM_API AdtObj* lmx_quantity_sub(QuantityObj* lhs, QuantityObj* rhs) {
    if (!lhs || !rhs) return real_result_error("quantity_sub: null quantity");
    lmmc_real_t ignored = 0.0;
    const auto status = lmmc_lsr_units_convert(1.0, rhs->unit().c_str(),
                                               lhs->unit().c_str(), &ignored);
    if (status != LMMC_STATUS_OK) return lmmc_object_error("quantity_sub", status);
    return quantity_result(lhs->si_value() - rhs->si_value(), lhs->unit(), "quantity_sub");
}

extern "C" LM_API AdtObj* lmx_quantity_mul(QuantityObj* lhs, QuantityObj* rhs) {
    if (!lhs || !rhs) return real_result_error("quantity_mul: null quantity");
    std::string unit;
    if (!unit_product_expression(lhs->unit(), rhs->unit(), false, unit)) {
        return real_result_error("quantity_mul: invalid resulting dimension");
    }
    return quantity_result(lhs->si_value() * rhs->si_value(), std::move(unit),
                           "quantity_mul");
}

extern "C" LM_API AdtObj* lmx_quantity_div(QuantityObj* lhs, QuantityObj* rhs) {
    if (!lhs || !rhs) return real_result_error("quantity_div: null quantity");
    if (rhs->si_value() == 0.0) return real_result_error("quantity_div: division by zero");
    std::string unit;
    if (!unit_product_expression(lhs->unit(), rhs->unit(), true, unit)) {
        return real_result_error("quantity_div: invalid resulting dimension");
    }
    return quantity_result(lhs->si_value() / rhs->si_value(), std::move(unit),
                           "quantity_div");
}

extern "C" LM_API AdtObj* lmx_quantity_pow(QuantityObj* value,
                                              const LmInt exponent) {
    if (!value || exponent < -32 || exponent > 32) {
        return real_result_error("quantity_pow: exponent must be in [-32, 32]");
    }
    if (value->si_value() == 0.0 && exponent < 0) {
        return real_result_error("quantity_pow: division by zero");
    }
    std::string unit;
    if (!unit_power_expression(value->unit(), static_cast<int>(exponent), unit)) {
        return real_result_error("quantity_pow: invalid resulting dimension");
    }
    return quantity_result(std::pow(value->si_value(), static_cast<double>(exponent)),
                           std::move(unit), "quantity_pow");
}

LM_API LmState* lmx_newState() {
    auto* node = static_cast<LmLinkedNode *>(malloc(sizeof(LmLinkedNode)));
    memset(node, 0, sizeof(LmLinkedNode));
    global_state = LmState {.n = node, .vm = nullptr};
    return &global_state;
}
LM_API void lmx_deleteState(const LmState* state) {
    const LmLinkedNode* node = state->n;
    while (node != nullptr) {
        if (node->ptr != nullptr) free(node->ptr);
        const auto last = node->last;
        free((void*)node);
        node = last;
    }
    delete reinterpret_cast<lmx::runtime::LaminaVM*>(state->vm);
}
static LmLinkedNode* newLickedNode(LmLinkedNode* old) {
    auto* node = static_cast<LmLinkedNode *>(malloc(sizeof(LmLinkedNode)));
    node->last = old;
    return node;
}
static void lmx_state_addNode(LmState* state, void* ptr) {
    state->n = newLickedNode(state->n);
    state->n->ptr = ptr;
}

static LmModule* lmx_newCodeModule(LmState* state, std::vector<uint8_t>&& binary) {
    const auto storage = malloc(sizeof(lmx::runtime::CodeModuleObj));
    if (storage == nullptr) return nullptr;
    new (storage) lmx::runtime::CodeModuleObj(std::move(binary));
    lmx_state_addNode(state, storage);
    return static_cast<LmModule*>(storage);
}

void lmx_printASTFromString(LmState *state, FILE *file, const char *code, const char* name) {
    lmx::Compiler compiler;
    lmx::CompileResult result;
    if (!compiler.compile_source(code, name, result, lmx::CompileStage::Semantic)) return;

    const auto ast_str = lmx::AstPrinter::print(*result.module);
    if (fwrite(ast_str.c_str(), 1, ast_str.length(), file) != ast_str.length()) {
        fprintf(stderr, "Error writing AST to file\n");
    }
}

void lmx_printASTFromFile(LmState *state, FILE *file, const char *name) {
    lmx::Compiler compiler;
    lmx::CompileResult result;
    if (!compiler.compile_file(name, result, lmx::CompileStage::Semantic)) return;
    auto str = lmx::AstPrinter::print(*result.module);
    if (fwrite(str.c_str(), 1, str.length(), file) != str.length()) {
        fprintf(stderr, "Error writing AST to file\n");
    }
}

void lmx_printMIRFromString(LmState *state, FILE *file, const char *code, const char *name) {
    lmx::Compiler compiler;
    lmx::CompileResult result;
    if (!compiler.compile_source(code, name, result, lmx::CompileStage::Mir)) return;
    const auto mir_str = lmx::mir::MirPrinter::print(*result.mir);

    if (fwrite(mir_str.c_str(), 1, mir_str.length(), file) != mir_str.length()) {
        fprintf(stderr, "Error writing MIR to file\n");
    }
}
void lmx_printMIRFromFile(LmState *state, FILE *file, const char *name) {
    lmx::Compiler compiler;
    lmx::CompileResult result;
    if (!compiler.compile_file(name, result, lmx::CompileStage::Mir)) return;
    const auto mir_str = lmx::mir::MirPrinter::print(*result.mir);

    if (fwrite(mir_str.c_str(), 1, mir_str.length(), file) != mir_str.length()) {
        fprintf(stderr, "Error writing MIR to file\n");
    }
}

LaminaVM* lmx_newLaminaVM(LmState* state, int argc, char** argv) {
    auto* vm = new lmx::runtime::LaminaVM(argc, argv);
    if (state->vm) delete reinterpret_cast<lmx::runtime::LaminaVM*>(state->vm);
    state->vm = reinterpret_cast<LaminaVM*>(vm);
    return state->vm;
}

bool lmx_moduleToFile(LmState *state, LmModule *module, const char *name) {
    const std::filesystem::path path = name;
    std::filesystem::create_directories(path.parent_path());
    std::ofstream ofs(path.string() + lmx::file_suffix_binary, std::ios::binary | std::ios::trunc);
    const auto* mod = reinterpret_cast<lmx::runtime::CodeModuleObj*>(module);
    ofs.write(
        reinterpret_cast<const char*>(mod->raw_data.data()),
        static_cast<std::streamsize>(mod->raw_data.size())
        );
    if (!ofs) return false;
    ofs.close();
    return true;
}

LmModule *lmx_doString(LmState *state, const char *code, const char* name) {
    lmx::Compiler compiler;
    lmx::CompileResult result;
    if (!compiler.compile_source(code, name, result)) return nullptr;
    return lmx_newCodeModule(state, std::move(result.binary));
}
LmModule *lmx_doFile(LmState *state, const char* name, bool is_main_module) {
    lmx::Compiler compiler;
    lmx::CompileResult result;
    if (!compiler.compile_file(name, result, lmx::CompileStage::Binary,
                               is_main_module)) return nullptr;
#if !NDEBUG
    if (debug_dump_enabled()) {
        std::cout << lmx::AstPrinter::print(*result.module) << std::endl;
    }
#endif
#if !NDEBUG
    if (debug_dump_enabled()) {
        std::cout << lmx::mir::MirPrinter::print(*result.mir) << std::endl;
    }
#endif

    return lmx_newCodeModule(state, std::move(result.binary));
}

int lmx_vmRunModule(LmState* state, LaminaVM* vm, LmModule* module) {
    if (module == nullptr) return 1;
    return
    reinterpret_cast<lmx::runtime::LaminaVM*>(vm)
    ->
    run(reinterpret_cast<lmx::runtime::CodeModuleObj*>(module));
}

void lmx_vmEval(LmState *state, LaminaVM *vm, LmValue *result, const char *code) {
    std::string c = code;
    auto tks = lmx::Lexer(c).tokenize(c);
    auto node = lmx::Parser(tks).parse_stmt();
    lmx::hir::TypeCkContext().check_stmt(node);
}
