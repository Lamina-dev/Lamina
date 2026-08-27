#include "bridge/conversions.hpp"

#include "internal/symbolic_ast/arithmetic.hpp"

#include <cstdarg>
#include <iostream>
#include <utility>

namespace lmx::bridge {

namespace {
class LmmcRuntimeLifetime {
public:
    LmmcRuntimeLifetime() noexcept { lmmc_init(); }
    ~LmmcRuntimeLifetime() noexcept { lmmc_deinit(); }

    LmmcRuntimeLifetime(const LmmcRuntimeLifetime&) = delete;
    LmmcRuntimeLifetime& operator=(const LmmcRuntimeLifetime&) = delete;
};

const LmmcRuntimeLifetime lmmc_runtime_lifetime;
} // namespace

bool debug_dump_enabled() noexcept {
    const char* value = std::getenv("LMX_DEBUG_DUMP");
    return value && value[0] != '\0' && value[0] != '0';
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

bool checked_symbol_name(ExprObj* expr, std::string& name, std::string& error) {
    const auto* value = checked_expr(expr, error);
    if (!value) return false;
    const auto variable = std::dynamic_pointer_cast<const VariableNode>(
        lamina::detail::node(**value));
    if (!variable) {
        error = "CasError(InvalidArgument: expr must be a single symbol)";
        return false;
    }
    name = variable->name();
    return true;
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

bool array_expressions(const ArrayObj* array,
                       std::vector<lamina::lsr::ExprPtr>& result,
                       std::string& error) {
    if (!array) {
        error = "null array";
        return false;
    }
    result.reserve(static_cast<std::size_t>(array->len()));
    for (const auto& value : array->values()) {
        if (value.kind != ValueKind::Expr || !value.obj) {
            error = "array contains a non-expression value";
            return false;
        }
        const auto* expression = checked_expr(
            reinterpret_cast<ExprObj*>(value.obj), error);
        if (!expression) return false;
        result.push_back(*expression);
    }
    return true;
}

bool array_strings(const ArrayObj* array, std::vector<std::string>& result,
                   std::string& error) {
    if (!array) {
        error = "null array";
        return false;
    }
    result.reserve(static_cast<std::size_t>(array->len()));
    for (const auto& value : array->values()) {
        if (value.kind != ValueKind::Obj || !value.obj ||
            value.obj->get_kind() != lmx::runtime::ObjectKind::String) {
            error = "array contains a non-text value";
            return false;
        }
        result.emplace_back(reinterpret_cast<StringObj*>(value.obj)->c_str());
    }
    return true;
}

bool expr_to_real(ExprObj* expr, double& result, std::string& error) {
    const auto* value = checked_expr(expr, error);
    if (!value) return false;
    const auto evaluated = lamina::lsr::evalf(**value);
    if (!evaluated) {
        error = evaluated.error().message;
        return false;
    }
    result = evaluated.value().value;
    return true;
}

std::optional<lamina::lsr::NumberDomainSet> number_domain_for_name(
    const char* name) {
    const std::string domain = name ? name : "";
    if (domain == "integers") return lamina::lsr::integers();
    if (domain == "rationals") return lamina::lsr::rationals();
    if (domain == "reals") return lamina::lsr::reals();
    if (domain == "complexes") return lamina::lsr::complexes();
    if (domain == "expressions") return lamina::lsr::expressions();
    return std::nullopt;
}

bool checked_complex(ComplexObj* value, lmmc_complex_t& result) noexcept {
    if (!value) return false;
    result.real = value->real();
    result.imag = value->imag();
    return true;
}

AdtObj* complex_result_ok(const lmmc_complex_t& value) {
    return result_ok(new ComplexObj(value.real, value.imag),
                     lmx::runtime::ValueKind::Complex);
}

AdtObj* lmmc_real_result(const char* operation, const lmmc_status_t status,
                         const lmmc_real_t value) {
    if (status == LMMC_STATUS_OK) return result_ok(value);
    return result_error(status, operation ? operation : "LMMC");
}

AdtObj* lmmc_complex_result(const char* operation,
                            const lmmc_status_t status,
                            const lmmc_complex_t& value) {
    if (status == LMMC_STATUS_OK) return complex_result_ok(value);
    return result_error(status, operation ? operation : "LMMC");
}

std::optional<lamina::UnitDefinition> resolved_unit_definition(
    const char* dimension_text, const LmInt numerator,
    const LmInt denominator, std::string& error) {
    if (!dimension_text || denominator <= 0 || numerator <= 0) {
        error = "CasError(UnitInvalid: invalid unit definition)";
        return std::nullopt;
    }
    lamina::DimensionSignature::Exponents exponents;
    const std::string dimension(dimension_text);
    if (dimension != "1") {
        std::size_t cursor = 0;
        while (cursor < dimension.size()) {
            const auto separator = dimension.find('*', cursor);
            const auto factor = dimension.substr(
                cursor, separator == std::string::npos
                    ? std::string::npos : separator - cursor);
            const auto power = factor.rfind('^');
            const auto name = factor.substr(0, power);
            int exponent = 1;
            if (power != std::string::npos) {
                try {
                    std::size_t used = 0;
                    exponent = std::stoi(factor.substr(power + 1), &used);
                    if (used != factor.size() - power - 1) throw std::invalid_argument("unit exponent");
                } catch (...) {
                    error = "CasError(UnitInvalid: malformed dimension exponent)";
                    return std::nullopt;
                }
            }
            if (name.empty() || exponent == 0) {
                error = "CasError(UnitInvalid: malformed dimension signature)";
                return std::nullopt;
            }
            exponents.emplace(name, Rational(exponent));
            if (separator == std::string::npos) break;
            cursor = separator + 1;
        }
    }
    return lamina::UnitDefinition{
        lamina::DimensionSignature(std::move(exponents)),
        Rational(BigInt(std::to_string(numerator)),
                 BigInt(std::to_string(denominator)))};
}

} // namespace lmx::bridge
