#pragma once

#include "include/lmx.h"
#include "runtime/object/adt.hpp"
#include "runtime/object/object.hpp"
#include "runtime/object/value.hpp"
#include <result.hpp>
#include "lmmc/status.h"

#include <memory>
#include <utility>
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
template <typename T>
struct ObjectRelease {
    void operator()(T* value) const noexcept {
        if (value) value->release();
    }
};

template <typename T>
using OwnedObject = std::unique_ptr<T, ObjectRelease<T>>;

template <typename T, typename... Args>
OwnedObject<T> make_owned_object(Args&&... args) {
    return OwnedObject<T>(new T(std::forward<Args>(args)...));
}

template <typename T>
runtime::Value take_object_value(
    OwnedObject<T> value, runtime::ValueKind kind = runtime::ValueKind::Obj) noexcept {
    runtime::Value result(value.get(), kind);
    value.release();
    return result;
}

template <typename T>
OwnedObject<T> adopt_object(T* value) noexcept {
    return OwnedObject<T>(value);
}

MathErrorCode math_error_code(lamina::CasErrc code) noexcept;
MathErrorCode math_error_code(lmmc_status_t status) noexcept;

runtime::AdtObj* make_mathematics_error(MathErrorCode code,
                                        std::string operation,
                                        std::string message);
runtime::AdtObj* result_error(MathErrorCode code, std::string operation,
                              std::string message);
runtime::AdtObj* result_error(const lamina::CasError& error);
runtime::AdtObj* result_error(lmmc_status_t status, std::string operation);
runtime::AdtObj* c_abi_error(MathErrorCode code, const char* operation,
                             const char* message) noexcept;
runtime::AdtObj* c_abi_error(const lamina::CasError& error) noexcept;
runtime::AdtObj* c_abi_current_exception(
    const char* operation) noexcept;
runtime::AdtObj* result_ok(double value);
runtime::AdtObj* result_ok(LmInt value);
runtime::AdtObj* result_ok(bool value);
runtime::AdtObj* result_ok(runtime::Object* value, runtime::ValueKind kind);

} // namespace lmx::bridge
