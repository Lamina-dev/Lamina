#include "bridge/math_internal.hpp"

namespace lmx::bridge::math_internal {

template <typename Expression, typename Convert>
ArrayObj* solution_tables_impl(
    const std::vector<std::map<std::string, Expression>>& solutions,
    Convert convert) {
    auto result = make_owned_object<ArrayObj>();
    for (const auto& solution : solutions) {
        std::vector<TableObj::Entry> entries;
        for (const auto& [name, expression] : solution) {
            entries.emplace_back(
                name, take_object_value(
                    make_owned_object<ExprObj>(convert(expression)),
                    ValueKind::Expr));
        }
        result->append(take_object_value(
            make_owned_object<TableObj>(std::move(entries)), ValueKind::Table));
    }
    return result.release();
}

ArrayObj* solution_tables(
    const std::vector<std::map<std::string, LMCAS::ExprPtr>>& solutions) {
    return solution_tables_impl(
        solutions, [](const LMCAS::ExprPtr& expression) { return expression; });
}

ArrayObj* solution_tables(
    const std::vector<std::map<std::string, LMCAS::SymbolicExpr>>& solutions) {
    return solution_tables_impl(solutions, [](const LMCAS::SymbolicExpr& expression) {
        return std::make_shared<LMCAS::SymbolicExpr>(expression);
    });
}

bool checked_symbol_names(
    ArrayObj* values, std::vector<std::string>& names, std::string& error,
    const char* non_expression_message) {
    if (!values) {
        error = "CasError(InvalidArgument: null symbol array)";
        return false;
    }
    names.reserve(static_cast<std::size_t>(values->len()));
    for (const auto& value : values->values()) {
        if (value.kind != ValueKind::Expr || !value.obj) {
            error = non_expression_message;
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
    std::vector<std::vector<LMCAS::ExprPtr>>& output,
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
        std::vector<LMCAS::ExprPtr> row;
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

AdtObj* unordered_expr_result(std::vector<LMCAS::ExprPtr> values) {
    return expression_set_literal_result(
        LMCAS::ExprSet::make(std::move(values)));
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

}
