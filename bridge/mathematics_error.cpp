#include "bridge/mathematics_error.hpp"

#include "runtime/object/StringObj.hpp"

#include <utility>
#include <exception>
#include <new>
#include <string_view>
#include <vector>

namespace lmx::bridge {
namespace {

const char* constructor_name(const MathErrorCode code) noexcept {
    switch (code) {
    case MathErrorCode::InvalidArgument: return "InvalidArgument";
    case MathErrorCode::ParseError: return "ParseError";
    case MathErrorCode::UnboundSymbol: return "UnboundSymbol";
    case MathErrorCode::DomainError: return "DomainError";
    case MathErrorCode::DimensionMismatch: return "DimensionMismatch";
    case MathErrorCode::UnitInvalid: return "UnitInvalid";
    case MathErrorCode::UnitStripTypeMismatch: return "UnitStripTypeMismatch";
    case MathErrorCode::SetElementTypeMismatch: return "SetElementTypeMismatch";
    case MathErrorCode::SetOperandTypeMismatch: return "SetOperandTypeMismatch";
    case MathErrorCode::SetElementNotHashable: return "SetElementNotHashable";
    case MathErrorCode::UnsupportedExpression: return "UnsupportedExpression";
    case MathErrorCode::Inconclusive: return "Inconclusive";
    case MathErrorCode::ResourceLimit: return "ResourceLimit";
    case MathErrorCode::Cancelled: return "Cancelled";
    case MathErrorCode::NumericalFailure: return "NumericalFailure";
    case MathErrorCode::SingularMatrix: return "SingularMatrix";
    case MathErrorCode::NotPositiveDefinite: return "NotPositiveDefinite";
    case MathErrorCode::ConvergenceFailure: return "ConvergenceFailure";
    case MathErrorCode::IndexOutOfBounds: return "IndexOutOfBounds";
    case MathErrorCode::EmptyInput: return "EmptyInput";
    case MathErrorCode::CallbackFailure: return "CallbackFailure";
    case MathErrorCode::InternalError: return "InternalError";
    }
    return "InternalError";
}

runtime::AdtObj* make_result_ok(std::vector<runtime::Value> fields) {
    return new runtime::AdtObj("Result", "Ok", std::move(fields));
}

} // namespace

MathErrorCode math_error_code(const lamina::CasErrc code) noexcept {
    switch (code) {
    case lamina::CasErrc::InvalidArgument: return MathErrorCode::InvalidArgument;
    case lamina::CasErrc::ParseError: return MathErrorCode::ParseError;
    case lamina::CasErrc::UnboundSymbol: return MathErrorCode::UnboundSymbol;
    case lamina::CasErrc::DomainError: return MathErrorCode::DomainError;
    case lamina::CasErrc::DimensionMismatch: return MathErrorCode::DimensionMismatch;
    case lamina::CasErrc::UnitInvalid: return MathErrorCode::UnitInvalid;
    case lamina::CasErrc::UnitStripTypeMismatch: return MathErrorCode::UnitStripTypeMismatch;
    case lamina::CasErrc::SetElementTypeMismatch: return MathErrorCode::SetElementTypeMismatch;
    case lamina::CasErrc::SetOperandTypeMismatch: return MathErrorCode::SetOperandTypeMismatch;
    case lamina::CasErrc::SetElementNotHashable: return MathErrorCode::SetElementNotHashable;
    case lamina::CasErrc::UnsupportedExpression: return MathErrorCode::UnsupportedExpression;
    case lamina::CasErrc::Inconclusive: return MathErrorCode::Inconclusive;
    case lamina::CasErrc::ResourceLimit: return MathErrorCode::ResourceLimit;
    case lamina::CasErrc::Cancelled: return MathErrorCode::Cancelled;
    case lamina::CasErrc::NumericFailure: return MathErrorCode::NumericalFailure;
    case lamina::CasErrc::InternalInvariant: return MathErrorCode::InternalError;
    }
    return MathErrorCode::InternalError;
}

MathErrorCode math_error_code(const lmmc_status_t status) noexcept {
    switch (status) {
    case LMMC_STATUS_INVALID_ARGUMENT: return MathErrorCode::InvalidArgument;
    case LMMC_STATUS_DIMENSION_MISMATCH: return MathErrorCode::DimensionMismatch;
    case LMMC_STATUS_ALLOCATION_FAILED: return MathErrorCode::ResourceLimit;
    case LMMC_STATUS_SINGULAR_MATRIX: return MathErrorCode::SingularMatrix;
    case LMMC_STATUS_REFERENCE_LIMIT: return MathErrorCode::ResourceLimit;
    case LMMC_STATUS_NOT_IMPLEMENTED: return MathErrorCode::UnsupportedExpression;
    case LMMC_STATUS_NUMERICAL_FAILURE: return MathErrorCode::NumericalFailure;
    case LMMC_STATUS_NOT_POSITIVE_DEFINITE: return MathErrorCode::NotPositiveDefinite;
    case LMMC_STATUS_CONVERGENCE_FAILED: return MathErrorCode::ConvergenceFailure;
    case LMMC_STATUS_OUT_OF_RANGE:
    case LMMC_STATUS_INDEX_OUT_OF_BOUNDS: return MathErrorCode::IndexOutOfBounds;
    case LMMC_STATUS_EMPTY_INPUT: return MathErrorCode::EmptyInput;
    case LMMC_STATUS_UNIT_STRIP_TYPE_MISMATCH: return MathErrorCode::UnitStripTypeMismatch;
    case LMMC_STATUS_UNIT_STRIP_OVERFLOW:
    case LMMC_STATUS_UNIT_STRIP_INVALID:
    case LMMC_STATUS_UNIT_STRIP_LEGACY_SYNTAX: return MathErrorCode::UnitInvalid;
    case LMMC_STATUS_OK:
    case LMMC_STATUS_WARNING_MAX_DEPTH: return MathErrorCode::InternalError;
    case LMMC_STATUS_NOT_INITIALIZED:
    case LMMC_STATUS_BUSY:
        return MathErrorCode::InternalError;
    }
    return MathErrorCode::InternalError;
}

runtime::AdtObj* make_mathematics_error(const MathErrorCode code,
                                        std::string operation,
                                        std::string message) {
    std::vector<runtime::Value> code_fields;
    auto code_value = make_owned_object<runtime::AdtObj>(
        "MathErrorCode", constructor_name(code), std::move(code_fields));
    std::vector<runtime::Value> error_fields;
    error_fields.emplace_back(take_object_value(std::move(code_value)));
    error_fields.emplace_back(take_object_value(
        make_owned_object<runtime::StringObj>(std::move(operation))));
    error_fields.emplace_back(take_object_value(
        make_owned_object<runtime::StringObj>(std::move(message))));
    return new runtime::AdtObj(
        "MathError", "MathError", std::move(error_fields));
}

runtime::AdtObj* result_error(const MathErrorCode code, std::string operation,
                              std::string message) {
    auto error = adopt_object(make_mathematics_error(
        code, std::move(operation), std::move(message)));
    std::vector<runtime::Value> fields;
    fields.emplace_back(take_object_value(std::move(error)));
    return new runtime::AdtObj("Result", "Err", std::move(fields));
}

runtime::AdtObj* result_error(const lamina::CasError& error) {
    return result_error(math_error_code(error.code), error.operation,
                        error.message);
}

runtime::AdtObj* result_error(const lmmc_status_t status,
                              std::string operation) {
    return result_error(math_error_code(status), std::move(operation),
                        lmmc_status_string(status));
}
runtime::AdtObj* c_abi_error(const MathErrorCode code,
                             const char* operation,
                             const char* message) noexcept {
    try {
        return result_error(
            code, operation ? operation : "bridge",
            message ? message : "unknown bridge failure");
    } catch (...) {
        return nullptr;
    }
}

runtime::AdtObj* c_abi_error(const lamina::CasError& error) noexcept {
    try {
        return result_error(error);
    } catch (...) {
        return nullptr;
    }
}

runtime::AdtObj* c_abi_current_exception(
    const char* operation) noexcept {
    try {
        throw;
    } catch (const lamina::detail::ResultPropagation& propagation) {
        return c_abi_error(propagation.error());
    } catch (const std::bad_alloc&) {
        return c_abi_error(MathErrorCode::ResourceLimit, operation,
                           "bridge allocation failed");
    } catch (const std::exception& error) {
        return c_abi_error(
            MathErrorCode::InternalError, operation, error.what());
    } catch (...) {
        return c_abi_error(MathErrorCode::InternalError, operation,
                           "unknown bridge exception");
    }
}

runtime::AdtObj* result_ok(const double value) {
    std::vector<runtime::Value> fields;
    fields.emplace_back(value);
    return make_result_ok(std::move(fields));
}

runtime::AdtObj* result_ok(const LmInt value) {
    std::vector<runtime::Value> fields;
    fields.emplace_back(value);
    return make_result_ok(std::move(fields));
}

runtime::AdtObj* result_ok(const bool value) {
    std::vector<runtime::Value> fields;
    fields.emplace_back(value);
    return make_result_ok(std::move(fields));
}

runtime::AdtObj* result_ok(runtime::Object* value,
                           const runtime::ValueKind kind) {
    auto owned = adopt_object(value);
    std::vector<runtime::Value> fields;
    fields.emplace_back(take_object_value(std::move(owned), kind));
    return make_result_ok(std::move(fields));
}

#if defined(LMX_BUILD_TESTS)
namespace {

runtime::AdtObj* allocation_failure_probe() noexcept try {
    throw std::bad_alloc{};
} catch (...) {
    return c_abi_current_exception(__func__);
}

runtime::AdtObj* checked_failure_probe() noexcept try {
    throw lamina::detail::ResultPropagation(lamina::CasError{
        lamina::CasErrc::DomainError,
        "checked failure",
        "checked.operation"});
} catch (...) {
    return c_abi_current_exception(__func__);
}

bool has_error(const runtime::AdtObj* result, const char* code,
               const char* operation, const char* message) {
    if (!result || result->type_name() != "Result" ||
        result->constructor() != "Err") {
        return false;
    }
    const auto* error_field = result->field(0);
    const auto* error =
        error_field && error_field->kind == runtime::ValueKind::Obj &&
                error_field->obj &&
                error_field->obj->get_kind() == runtime::ObjectKind::Adt
            ? static_cast<const runtime::AdtObj*>(error_field->obj)
            : nullptr;
    if (!error || error->type_name() != "MathError") return false;
    const auto* code_field = error->field(0);
    const auto* code_value =
        code_field && code_field->kind == runtime::ValueKind::Obj &&
                code_field->obj &&
                code_field->obj->get_kind() == runtime::ObjectKind::Adt
            ? static_cast<const runtime::AdtObj*>(code_field->obj)
            : nullptr;
    const auto string_field = [error](const std::size_t index) {
        const auto* field = error->field(index);
        return field && field->kind == runtime::ValueKind::Obj &&
                       field->obj &&
                       field->obj->get_kind() == runtime::ObjectKind::String
            ? static_cast<const runtime::StringObj*>(field->obj)
            : nullptr;
    };
    const auto* operation_value = string_field(1);
    const auto* message_value = string_field(2);
    return code_value && code_value->constructor() == code &&
           operation_value &&
           std::string_view(operation_value->c_str()) == operation &&
           message_value && std::string_view(message_value->c_str()) == message;
}

} // namespace

extern "C" LM_API int lmx_test_c_abi_exception_boundaries() noexcept {
    try {
        auto allocation = adopt_object(allocation_failure_probe());
        auto checked = adopt_object(checked_failure_probe());
        return has_error(allocation.get(), "ResourceLimit",
                         "allocation_failure_probe",
                         "bridge allocation failed") &&
                       has_error(checked.get(), "DomainError",
                                 "checked.operation", "checked failure")
            ? 0
            : 1;
    } catch (...) {
        return 2;
    }
}
#endif

} // namespace lmx::bridge
