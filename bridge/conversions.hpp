#pragma once

#include "bridge/mathematics_error.hpp"

#include "include/lmx.h"

#include "compiler/compiler.hpp"
#include "runtime/object/lsr_expr_obj.hpp"
#include "runtime/object/StringObj.hpp"
#include "runtime/object/adt.hpp"
#include "runtime/object/complex.hpp"
#include "runtime/object/array.hpp"
#include "runtime/object/matrix.hpp"
#include "runtime/object/quantity.hpp"
#include "runtime/object/table.hpp"
#include "runtime/object/vector.hpp"
#include "runtime/object/value.hpp"

#include <result.hpp>

#include <lmmc/numeric.h>
#include <lmmc/complex.h>

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace lmx::bridge {

using runtime::AdtObj;
using runtime::ArrayObj;
using runtime::ComplexObj;
using runtime::ExprObj;
using runtime::MatrixObj;
using runtime::QuantityObj;
using runtime::StringObj;
using runtime::TableObj;
using runtime::Value;
using runtime::ValueKind;
using runtime::VectorObj;

const lamina::lsr::ExprPtr* checked_expr(ExprObj* expr, std::string& error);
bool checked_symbol_name(ExprObj* expr, std::string& name,
                         std::string& error);
lamina::lsr::ExprResult invalid_expr_operation(const std::string& message,
                                               const char* operation);
bool collect_expr_arguments(va_list& args, LmInt count,
                            std::vector<lamina::lsr::ExprPtr>& values,
                            std::string& error);
bool numeric_value(const Value& value, double& result) noexcept;
bool array_numbers(const ArrayObj* array, std::vector<double>& result,
                   std::string& error);
bool array_expressions(const ArrayObj* array,
                       std::vector<lamina::lsr::ExprPtr>& result,
                       std::string& error);
bool array_strings(const ArrayObj* array, std::vector<std::string>& result,
                   std::string& error);
bool expr_to_real(ExprObj* expr, double& result, std::string& error);
std::optional<lamina::lsr::NumberDomainSet> number_domain_for_name(
    const char* name);
bool checked_complex(ComplexObj* value, lmmc_complex_t& result) noexcept;
AdtObj* complex_result_ok(const lmmc_complex_t& value);
AdtObj* lmmc_real_result(const char* operation, lmmc_status_t status,
                         lmmc_real_t value);
AdtObj* lmmc_complex_result(const char* operation, lmmc_status_t status,
                            const lmmc_complex_t& value);
std::optional<lamina::UnitDefinition> resolved_unit_definition(
    const char* dimension_text, LmInt numerator, LmInt denominator,
    std::string& error);

} // namespace lmx::bridge
