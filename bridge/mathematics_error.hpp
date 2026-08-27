#pragma once

#include "include/lmx.h"
#include "runtime/object/adt.hpp"
#include "runtime/object/object.hpp"
#include "runtime/object/value.hpp"
#include <result.hpp>
#include "lmmc/status.h"

#include <string>

namespace lmx::bridge {

enum class MathErrorCode {
    InvalidArgument,
    ParseError,
    UnboundSymbol,
    DomainError,
    DimensionMismatch,
    UnitInvalid,
    UnitStripTypeMismatch,
    SetElementTypeMismatch,
    SetOperandTypeMismatch,
    SetElementNotHashable,
    UnsupportedExpression,
    Inconclusive,
    ResourceLimit,
    Cancelled,
    NumericalFailure,
    SingularMatrix,
    NotPositiveDefinite,
    ConvergenceFailure,
    IndexOutOfBounds,
    EmptyInput,
    CallbackFailure,
    InternalError
};

MathErrorCode math_error_code(lamina::CasErrc code) noexcept;
MathErrorCode math_error_code(lmmc_status_t status) noexcept;

runtime::AdtObj* make_mathematics_error(MathErrorCode code,
                                        std::string operation,
                                        std::string message);
runtime::AdtObj* result_error(MathErrorCode code, std::string operation,
                              std::string message);
runtime::AdtObj* result_error(const lamina::CasError& error);
runtime::AdtObj* result_error(lmmc_status_t status, std::string operation);
runtime::AdtObj* result_ok(double value);
runtime::AdtObj* result_ok(LmInt value);
runtime::AdtObj* result_ok(bool value);
runtime::AdtObj* result_ok(runtime::Object* value, runtime::ValueKind kind);

} // namespace lmx::bridge
