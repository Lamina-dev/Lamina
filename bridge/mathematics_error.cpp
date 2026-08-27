#include "bridge/mathematics_error.hpp"

#include "runtime/object/StringObj.hpp"

#include <utility>
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
    }
    return MathErrorCode::InternalError;
}

runtime::AdtObj* make_mathematics_error(const MathErrorCode code,
                                        std::string operation,
                                        std::string message) {
    std::vector<runtime::Value> code_fields;
    auto* code_value = new runtime::AdtObj(
        "MathErrorCode", constructor_name(code), std::move(code_fields));
    std::vector<runtime::Value> error_fields;
    error_fields.emplace_back(code_value);
    error_fields.emplace_back(static_cast<runtime::Object*>(
        new runtime::StringObj(std::move(operation))));
    error_fields.emplace_back(static_cast<runtime::Object*>(
        new runtime::StringObj(std::move(message))));
    return new runtime::AdtObj("MathError", "MathError", std::move(error_fields));
}

runtime::AdtObj* result_error(const MathErrorCode code, std::string operation,
                              std::string message) {
    std::vector<runtime::Value> fields;
    fields.emplace_back(make_mathematics_error(
        code, std::move(operation), std::move(message)));
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
    std::vector<runtime::Value> fields;
    fields.emplace_back(value, kind);
    return make_result_ok(std::move(fields));
}

} // namespace lmx::bridge
