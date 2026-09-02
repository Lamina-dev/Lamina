#include "bridge/math_internal.hpp"

namespace lmx::bridge::math_internal {

ArrayObj* solution_tables(
    const std::vector<std::map<std::string, lamina::lsr::ExprPtr>>& solutions) {
    auto result = make_owned_object<ArrayObj>();
    for (const auto& solution : solutions) {
        std::vector<TableObj::Entry> entries;
        for (const auto& [name, expression] : solution) {
            auto value = take_object_value(
                make_owned_object<ExprObj>(expression), ValueKind::Expr);
            entries.emplace_back(name, std::move(value));
        }
        result->append(take_object_value(
            make_owned_object<TableObj>(std::move(entries)), ValueKind::Table));
    }
    return result.release();
}

bool checked_symbol_names(
    ArrayObj* values, std::vector<std::string>& names, std::string& error) {
    if (!values) {
        error = "CasError(InvalidArgument: null symbol array)";
        return false;
    }
    names.reserve(static_cast<std::size_t>(values->len()));
    for (const auto& value : values->values()) {
        if (value.kind != ValueKind::Expr || !value.obj) {
            error = "CasError(InvalidArgument: symbol array contains a non-expression value)";
            return false;
        }
        std::string name;
        if (!checked_symbol_name(
                reinterpret_cast<ExprObj*>(value.obj), name, error)) {
            return false;
        }
        names.push_back(std::move(name));
    }
    return true;
}

bool nested_expressions(
    ArrayObj* rows,
    std::vector<std::vector<lamina::lsr::ExprPtr>>& output,
    std::string& error) {
    if (!rows || rows->values().empty()) {
        error = "matrix requires at least one row";
        return false;
    }
    std::size_t columns = 0;
    for (const auto& row_value : rows->values()) {
        if (row_value.kind != ValueKind::Obj || !row_value.obj ||
            row_value.obj->get_kind() != lmx::runtime::ObjectKind::Array) {
            error = "matrix row is not an array";
            return false;
        }
        std::vector<lamina::lsr::ExprPtr> row;
        if (!array_expressions(
                static_cast<ArrayObj*>(row_value.obj), row, error)) {
            return false;
        }
        if (row.empty() || (columns != 0 && row.size() != columns)) {
            error = "matrix rows have inconsistent lengths";
            return false;
        }
        columns = row.size();
        output.push_back(std::move(row));
    }
    return true;
}

AdtObj* unordered_expr_result(std::vector<lamina::lsr::ExprPtr> values) {
    return expression_set_literal_result(
        lamina::lsr::ExprSet::make(std::move(values)));
}

ArrayObj* symbol_text_array(ArrayObj* symbols, std::string& error) {
    std::vector<std::string> names;
    if (!checked_symbol_names(symbols, names, error)) return nullptr;
    auto result = make_owned_object<ArrayObj>();
    for (auto& name : names) {
        result->append(take_object_value(
            make_owned_object<StringObj>(std::move(name)), ValueKind::Obj));
    }
    return result.release();
}

AdtObj* checked_expr_result(const lamina::ExpressionResult& result) {
    if (!result) return result_error(result.error());
    if (!result.value()) {
        return result_error(MathErrorCode::InternalError, __func__, 
            "CasError(InternalInvariant: null expression result)");
    }
    return result_ok(new ExprObj(result.value()), ValueKind::Expr);
}

} // namespace lmx::bridge::math_internal
