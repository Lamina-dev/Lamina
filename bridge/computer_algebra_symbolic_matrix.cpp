#include "bridge/result.hpp"
#include "bridge/conversions.hpp"
#include "bridge/runtime_views.hpp"
#include "bridge/unit_bridge.hpp"
#include <cstdarg>
#include "bridge/math_internal.hpp"
#include "symbolic.hpp"
#include "symbolic_matrix.hpp"
#include "matrix_decomposition.hpp"
#include <array>

using namespace lmx::bridge;

namespace {
bool nested_expressions(
    ArrayObj* rows, std::vector<std::vector<std::shared_ptr<SymbolicExpr>>>& output,
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
                static_cast<ArrayObj*>(row_value.obj), row, error))
            return false;
        if (row.empty() || (columns != 0 && row.size() != columns)) {
            error = "matrix rows have inconsistent lengths";
            return false;
        }
        columns = row.size();
        output.emplace_back(row.begin(), row.end());
    }
    return true;
}

template <typename Result, typename Fields>
AdtObj* decomposition_result(
    const char* type_name, const Result& result, Fields fields) {
    if (!result) return result_error(result.error());
    std::vector<Value> values;
    for (const auto& expression : fields(result.value())) {
        if (!expression)
            return result_error(MathErrorCode::UnsupportedExpression, __func__, 
                std::string("CasError(UnsupportedExpression in ") +
                type_name + ")");
        values.emplace_back(new ExprObj(expression), ValueKind::Expr);
    }
    return result_ok(
        new AdtObj(type_name, type_name, std::move(values)), ValueKind::Obj);
}
} // namespace

extern "C" LM_API AdtObj* lmx_computer_algebra_symbolic_matrix_from_rows(ArrayObj* rows) {
    std::vector<std::vector<std::shared_ptr<SymbolicExpr>>> values;
    std::string error;
    if (!nested_expressions(rows, values, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "matrix.from_rows: " + error);
    try {
        return result_ok(
            new ExprObj(SymbolicExpr::matrix(values)), ValueKind::Expr);
    } catch (const std::exception& error) {
        return result_error(MathErrorCode::InvalidArgument, __func__, 
            std::string("matrix.from_rows: ") + error.what());
    }
}
extern "C" LM_API AdtObj* lmx_computer_algebra_symbolic_matrix_multiply(
    ExprObj* lhs, ExprObj* rhs) {
    std::string error;
    const auto* left = checked_expr(lhs, error);
    const auto* right = checked_expr(rhs, error);
    if (!left || !right) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    const auto result = lamina::matrix_multiply_checked(*left, *right);
    if (!result) return result_error(result.error());
    return result_ok(new ExprObj(result.value()), ValueKind::Expr);
}
extern "C" LM_API AdtObj* lmx_computer_algebra_symbolic_matrix_determinant(ExprObj* value) {
    std::string error;
    const auto* checked = checked_expr(value, error);
    if (!checked) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    const auto result = lamina::matrix_determinant_checked(*checked);
    if (!result) return result_error(result.error());
    return result_ok(new ExprObj(result.value()), ValueKind::Expr);
}
extern "C" LM_API AdtObj* lmx_computer_algebra_symbolic_matrix_inverse(ExprObj* value) {
    std::string error;
    const auto* checked = checked_expr(value, error);
    if (!checked) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    const auto result = lamina::matrix_inverse_checked(*checked);
    if (!result) return result_error(result.error());
    return result_ok(new ExprObj(result.value()), ValueKind::Expr);
}
extern "C" LM_API AdtObj* lmx_computer_algebra_symbolic_matrix_lower_upper_decomposition(ExprObj* value) {
    std::string error;
    const auto* checked = checked_expr(value, error);
    if (!checked) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return decomposition_result(
        "SymbolicLU", lamina::lu_decomposition_checked(*checked),
        [](const auto& result) { return std::array{result.L, result.U}; });
}
extern "C" LM_API AdtObj* lmx_computer_algebra_symbolic_matrix_orthogonal_triangular_decomposition(ExprObj* value) {
    std::string error;
    const auto* checked = checked_expr(value, error);
    if (!checked) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return decomposition_result(
        "SymbolicQR", lamina::qr_decomposition_checked(*checked),
        [](const auto& result) { return std::array{result.Q, result.R}; });
}
extern "C" LM_API AdtObj* lmx_computer_algebra_symbolic_matrix_cholesky(ExprObj* value) {
    std::string error;
    const auto* checked = checked_expr(value, error);
    if (!checked) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return decomposition_result(
        "SymbolicCholesky", lamina::cholesky_decomposition_checked(*checked),
        [](const auto& result) { return std::array{result.L}; });
}
extern "C" LM_API AdtObj* lmx_computer_algebra_symbolic_matrix_singular_value_decomposition(ExprObj* value) {
    std::string error;
    const auto* checked = checked_expr(value, error);
    if (!checked) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return decomposition_result(
        "SymbolicSvd", lamina::svd_decomposition_checked(*checked),
        [](const auto& result) {
            return std::array{result.U, result.S, result.V};
        });
}
extern "C" LM_API AdtObj* lmx_computer_algebra_symbolic_matrix_jordan(ExprObj* value) {
    std::string error;
    const auto* checked = checked_expr(value, error);
    if (!checked) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return decomposition_result(
        "JordanForm", lamina::jordan_form_checked(*checked),
        [](const auto& result) { return std::array{result.J, result.P}; });
}
