#pragma once

#include "bridge/mathematics_error.hpp"

#include "include/lmx.h"

#include "compiler/compiler.hpp"
#include "runtime/object/lsr_expr_obj.hpp"
#include "runtime/object/StringObj.hpp"
#include "runtime/object/adt.hpp"
#include "runtime/object/array.hpp"
#include "runtime/object/complex.hpp"
#include "runtime/object/table.hpp"
#include "runtime/object/literal.hpp"

#include <result.hpp>
#include "transform_engine.hpp"

#include <memory>
#include <string>
#include <vector>


namespace lmx::bridge {

using runtime::AdtObj;
using runtime::ArrayObj;
using runtime::ExprObj;
using runtime::StringObj;
using runtime::Value;
using runtime::ValueKind;
const lamina::lsr::ExprPtr* checked_expr(ExprObj* expr, std::string& error);

// Ownership contract for this header family (result/conversions/runtime_views/
// unit_bridge): every exported function returning a runtime object pointer
// transfers ownership of that object (and any object graph it uniquely owns)
// to the VM caller; every object parameter is a borrowed reference valid for
// the duration of the call unless explicitly documented otherwise.

[[noreturn]] ExprObj* expression_internal_error(std::string message);
ExprObj* expr_from_result(const lamina::lsr::ExprResult& result);
AdtObj* expr_result_ok(const lamina::lsr::ExprResult& result);
AdtObj* expr_pointer_result(lamina::lsr::ExprPtr value,
                                const char* operation);
AdtObj* expression_set_literal_result(const lamina::lsr::ExprSetResult& result);
AdtObj* transform_engine_result_value(const lamina::TransformEngineResult& result);

template <typename Operation>
AdtObj* unary_expression_result(ExprObj* expr, const char* operation_name,
                       Operation operation) {
    std::string error;
    const auto* value = checked_expr(expr, error);
    if (!value)
        return result_error(MathErrorCode::InvalidArgument, operation_name,
                            std::move(error));
    return expr_result_ok(operation(*value));
}

template <typename ResultType>
AdtObj* expr_array_result(const ResultType& result) {
    if (!result) return result_error(result.error());
    auto values = make_owned_object<ArrayObj>();
    for (const auto& expression : result.value()) {
        values->append(take_object_value(
            make_owned_object<ExprObj>(expression), ValueKind::Expr));
    }
    return result_ok(values.release(), ValueKind::Obj);
}

} // namespace lmx::bridge
