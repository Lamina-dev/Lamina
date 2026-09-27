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

template <typename Result, typename Fields>
AdtObj* decomposition_result(
    const char* operation_name, const char* type_name,
    const Result& result, Fields fields)
{
    if (!result) return result_error(result.error());
    std::vector<Value> values;
    for (const auto& expression : fields(result.value())) {
        if (!expression)
            return result_error(MathErrorCode::UnsupportedExpression,
                operation_name,
                std::string("CasError(UnsupportedExpression in ") +
                    type_name + ")");
        values.emplace_back(take_object_value(
            make_owned_object<ExprObj>(expression), ValueKind::Expr));
    }
    return result_ok(
        new AdtObj(type_name, type_name, std::move(values)), ValueKind::Obj);
}
}

extern "C" LM_API AdtObj* lmx_computer_algebra_symbolic_matrix_from_rows(ArrayObj* rows) noexcept try {
    ensure_lmmc_runtime();
    std::vector<std::vector<std::shared_ptr<LMCAS::SymbolicExpr>>> values;
    std::string error;
    if (!math_internal::nested_expressions(rows, values, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "matrix.from_rows: " + error);
    return result_ok(
        new ExprObj(LMCAS::SymbolicExpr::matrix(values)), ValueKind::Expr);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_computer_algebra_symbolic_matrix_multiply(
    ExprObj* lhs, ExprObj* rhs) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    const auto* left = checked_expr(lhs, error);
    const auto* right = checked_expr(rhs, error);
    if (!left || !right) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    const auto result = LMCAS::matrix_multiply_checked(*left, *right);
    if (!result) return result_error(result.error());
    return result_ok(new ExprObj(result.value()), ValueKind::Expr);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_computer_algebra_symbolic_matrix_determinant(ExprObj* value) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    const auto* checked = checked_expr(value, error);
    if (!checked) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    const auto result = LMCAS::matrix_determinant_checked(*checked);
    if (!result) return result_error(result.error());
    return result_ok(new ExprObj(result.value()), ValueKind::Expr);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_computer_algebra_symbolic_matrix_inverse(ExprObj* value) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    const auto* checked = checked_expr(value, error);
    if (!checked) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    const auto result = LMCAS::matrix_inverse_checked(*checked);
    if (!result) return result_error(result.error());
    return result_ok(new ExprObj(result.value()), ValueKind::Expr);
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_computer_algebra_symbolic_matrix_lower_upper_decomposition(ExprObj* value) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    const auto* checked = checked_expr(value, error);
    if (!checked) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return decomposition_result(
        __func__, "SymbolicLU", LMCAS::lu_decomposition_checked(*checked),
        [](const auto& result) { return std::array{result.P, result.L, result.U}; });
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_computer_algebra_symbolic_matrix_orthogonal_triangular_decomposition(ExprObj* value) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    const auto* checked = checked_expr(value, error);
    if (!checked) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return decomposition_result(
        __func__, "SymbolicQR", LMCAS::qr_decomposition_checked(*checked),
        [](const auto& result) { return std::array{result.Q, result.R}; });
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_computer_algebra_symbolic_matrix_cholesky(ExprObj* value) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    const auto* checked = checked_expr(value, error);
    if (!checked) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return decomposition_result(
        __func__, "SymbolicCholesky",
        LMCAS::cholesky_decomposition_checked(*checked),
        [](const auto& result) { return std::array{result.L}; });
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_computer_algebra_symbolic_matrix_singular_value_decomposition(ExprObj* value) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    const auto* checked = checked_expr(value, error);
    if (!checked) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return decomposition_result(
        __func__, "SymbolicSvd", LMCAS::svd_decomposition_checked(*checked),
        [](const auto& result) {
            return std::array{result.U, result.S, result.V};
        });
} catch (...) {
    return c_abi_current_exception(__func__);
}
extern "C" LM_API AdtObj* lmx_computer_algebra_symbolic_matrix_jordan(ExprObj* value) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    const auto* checked = checked_expr(value, error);
    if (!checked) return result_error(MathErrorCode::InvalidArgument, __func__, std::move(error));
    return decomposition_result(
        __func__, "JordanForm", LMCAS::jordan_form_checked(*checked),
        [](const auto& result) { return std::array{result.J, result.P}; });
} catch (...) {
    return c_abi_current_exception(__func__);
}
