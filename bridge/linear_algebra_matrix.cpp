#include "bridge/result.hpp"
#include "bridge/conversions.hpp"
#include "bridge/runtime_views.hpp"
#include "bridge/unit_bridge.hpp"
#include <cstdarg>
#include "bridge/linear_algebra_internal.hpp"
#include "lmmc/lsr_stdlib.h"
#include "lmmc/linear_algebra.h"

using namespace lmx::bridge;
using lmx::bridge::linear_algebra::copied_matrix;

extern "C" LM_API AdtObj* lmx_linear_algebra_matrix(ArrayObj* rows) noexcept try {
    ensure_lmmc_runtime();
    if (!rows || rows->values().empty()) {
        return result_error(MathErrorCode::InvalidArgument, __func__, "matrix: expected at least one row");
    }
    std::vector<double> data;
    std::size_t column_count = 0;
    for (const auto& row_value : rows->values()) {
        if (row_value.kind != ValueKind::Obj || !row_value.obj ||
            row_value.obj->get_kind() != lmx::runtime::ObjectKind::Array) {
            return result_error(MathErrorCode::InvalidArgument, __func__, "matrix: each row must be an array");
        }
        std::vector<double> row;
        std::string error;
        if (!array_numbers(reinterpret_cast<ArrayObj*>(row_value.obj), row, error)) {
            return result_error(MathErrorCode::InvalidArgument, __func__, "matrix: " + error);
        }
        if (column_count == 0) column_count = row.size();
        if (row.empty() || row.size() != column_count) {
            return result_error(MathErrorCode::InvalidArgument, __func__, "matrix: rows must be non-empty and rectangular");
        }
        data.insert(data.end(), row.begin(), row.end());
    }
    return result_ok(new MatrixObj(rows->values().size(), column_count,
                                          std::move(data)), ValueKind::Matrix);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API LmInt lmx_linear_algebra_row_count(MatrixObj* value) noexcept try {
    ensure_lmmc_runtime();
    return value ? static_cast<LmInt>(value->rows()) : 0;
} catch (...) {
    return 0;
}

extern "C" LM_API LmInt lmx_linear_algebra_column_count(MatrixObj* value) noexcept try {
    ensure_lmmc_runtime();
    return value ? static_cast<LmInt>(value->cols()) : 0;
} catch (...) {
    return 0;
}

extern "C" LM_API AdtObj* lmx_linear_algebra_element_at(MatrixObj* value, const LmInt row,
                                           const LmInt column) noexcept try {
    ensure_lmmc_runtime();
    if (!value || row < 0 || column < 0 ||
        static_cast<std::size_t>(row) >= value->rows() ||
        static_cast<std::size_t>(column) >= value->cols()) {
        return result_error(MathErrorCode::IndexOutOfBounds, __func__, "matrix_at: index out of bounds");
    }
    return result_ok(value->data()[static_cast<std::size_t>(row) * value->cols() +
                                        static_cast<std::size_t>(column)]);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API LmInt lmx_computer_algebra_table_size(TableObj* value) noexcept try {
    ensure_lmmc_runtime();
    return value ? static_cast<LmInt>(value->entries().size()) : 0;
} catch (...) {
    return 0;
}

extern "C" LM_API bool lmx_computer_algebra_table_has(TableObj* value, const char* key) noexcept try {
    ensure_lmmc_runtime();
    return value && key && value->find(key) != nullptr;
} catch (...) {
    return false;
}

extern "C" LM_API AdtObj* lmx_computer_algebra_table_vector(TableObj* value, const char* key) noexcept try {
    ensure_lmmc_runtime();
    if (!value || !key) return result_error(MathErrorCode::InvalidArgument, __func__, "table_vector: invalid argument");
    const auto* field = value->find(key);
    if (!field || field->kind != ValueKind::Vector || !field->obj) {
        return result_error(MathErrorCode::InvalidArgument, __func__, "table_vector: key is missing or is not a vector");
    }
    return result_ok(field->obj->get(), ValueKind::Vector);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_computer_algebra_table_matrix(TableObj* value, const char* key) noexcept try {
    ensure_lmmc_runtime();
    if (!value || !key) return result_error(MathErrorCode::InvalidArgument, __func__, "table_matrix: invalid argument");
    const auto* field = value->find(key);
    if (!field || field->kind != ValueKind::Matrix || !field->obj) {
        return result_error(MathErrorCode::InvalidArgument, __func__, "table_matrix: key is missing or is not a matrix");
    }
    return result_ok(field->obj->get(), ValueKind::Matrix);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_linear_algebra_matrix_norm(MatrixObj* value) noexcept try {
    ensure_lmmc_runtime();
    if (!value || !value->valid()) return result_error(MathErrorCode::InvalidArgument, __func__, "matrix_norm: invalid matrix");
    auto view = matrix_view(value);
    lmmc_real_t result = 0.0;
    const auto status = lmmc_mat_norm_fro(&view, &result);
    return lmmc_real_result("matrix_norm", status, result);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_linear_algebra_transpose(MatrixObj* value) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_matrix_unary_result("matrix_transpose", value,
                                    lmmc_lsr_linalg_transpose);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_linear_algebra_multiply(MatrixObj* lhs, MatrixObj* rhs) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_matrix_binary_result("matrix_mul", lhs, rhs,
                                     lmmc_lsr_linalg_matmul);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_linear_algebra_matrix_vector_product(MatrixObj* matrix, VectorObj* vector) noexcept try {
    ensure_lmmc_runtime();
    if (!matrix || !vector || !matrix->valid()) {
        return result_error(MathErrorCode::InvalidArgument, __func__, "matrix_vector_mul: invalid argument");
    }
    auto input_matrix = matrix_view(matrix);
    auto input_vector = vector_view(vector);
    lmmc_vec_t output{};
    const auto status = lmmc_lsr_linalg_matvec(&input_matrix, &input_vector,
                                                &output);
    return lmmc_vector_output("matrix_vector_mul", status, output);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_linear_algebra_identity_matrix(const LmInt size) noexcept try {
    ensure_lmmc_runtime();
    if (size <= 0) return result_error(MathErrorCode::InvalidArgument, __func__, "matrix_eye: size must be positive");
    lmmc_mat_t output{};
    const auto status = lmmc_lsr_linalg_eye(static_cast<std::size_t>(size),
                                             &output);
    return lmmc_matrix_output("matrix_eye", status, output);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_linear_algebra_diagonal_matrix(VectorObj* diagonal) noexcept try {
    ensure_lmmc_runtime();
    if (!diagonal) return result_error(MathErrorCode::InvalidArgument, __func__, "matrix_diag: null vector");
    auto input = vector_view(diagonal);
    lmmc_mat_t output{};
    const auto status = lmmc_lsr_linalg_diag(&input, &output);
    return lmmc_matrix_output("matrix_diag", status, output);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_linear_algebra_add(MatrixObj* lhs, MatrixObj* rhs) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_matrix_binary_result("matrix_add", lhs, rhs,
                                     lmmc_lsr_linalg_mat_add);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_linear_algebra_subtract(MatrixObj* lhs, MatrixObj* rhs) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_matrix_binary_result("matrix_sub", lhs, rhs,
                                     lmmc_lsr_linalg_mat_sub);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_linear_algebra_elementwise_multiply(MatrixObj* lhs,
                                                    MatrixObj* rhs) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_matrix_binary_result("matrix_mul_elements", lhs, rhs,
                                     lmmc_lsr_linalg_mat_mul_elem);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_linear_algebra_elementwise_divide(MatrixObj* lhs,
                                                    MatrixObj* rhs) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_matrix_binary_result("matrix_div_elements", lhs, rhs,
                                     lmmc_lsr_linalg_mat_div);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_linear_algebra_elementwise_power(MatrixObj* base,
                                                    MatrixObj* exponent) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_matrix_binary_result("matrix_pow_elements", base, exponent,
                                     lmmc_lsr_linalg_mat_pow_elem);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_linear_algebra_add_scalar(MatrixObj* value,
                                                  const double scalar) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_matrix_scalar_result("matrix_add_scalar", value, scalar,
                                     lmmc_lsr_linalg_mat_add_scalar);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_linear_algebra_subtract_scalar(MatrixObj* value,
                                                  const double scalar) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_matrix_scalar_result("matrix_sub_scalar", value, scalar,
                                     lmmc_lsr_linalg_mat_sub_scalar);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_linear_algebra_multiply_scalar(MatrixObj* value,
                                                  const double scalar) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_matrix_scalar_result("matrix_mul_scalar", value, scalar,
                                     lmmc_lsr_linalg_mat_mul_scalar);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_linear_algebra_divide_scalar(MatrixObj* value,
                                                  const double scalar) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_matrix_scalar_result("matrix_div_scalar", value, scalar,
                                     lmmc_lsr_linalg_mat_div_scalar);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_linear_algebra_power_scalar(MatrixObj* value,
                                                  const double scalar) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_matrix_scalar_result("matrix_pow_scalar", value, scalar,
                                     lmmc_lsr_linalg_mat_pow_scalar);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_linear_algebra_scale(MatrixObj* value,
                                             const double scalar) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_matrix_scalar_result("matrix_scale", value, scalar,
                                     lmmc_lsr_linalg_mat_scale);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_linear_algebra_integer_power(MatrixObj* value,
                                               const LmInt exponent) noexcept try {
    ensure_lmmc_runtime();
    if (!value || !value->valid()) {
        return result_error(MathErrorCode::InvalidArgument, __func__, "matrix_pow_int: invalid matrix");
    }
    auto input = matrix_view(value);
    lmmc_mat_t output{};
    const auto status = lmmc_lsr_linalg_mat_pow_int(&input, exponent, &output);
    return lmmc_matrix_output("matrix_pow_int", status, output);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_linear_algebra_adjoint(MatrixObj* value) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_matrix_unary_result("matrix_adjoint", value,
                                    lmmc_lsr_linalg_adjoint);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_linear_algebra_inverse(MatrixObj* value) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_matrix_unary_result("matrix_inverse", value,
                                    lmmc_lsr_linalg_inv);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_linear_algebra_det(MatrixObj* value) noexcept try {
    ensure_lmmc_runtime();
    if (!value || !value->valid()) return result_error(MathErrorCode::InvalidArgument, __func__, "matrix_det: invalid matrix");
    auto input = matrix_view(value);
    lmmc_real_t output = 0.0;
    const auto status = lmmc_lsr_linalg_det(&input, &output);
    return lmmc_real_result("matrix_det", status, output);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_linear_algebra_trace(MatrixObj* value) noexcept try {
    ensure_lmmc_runtime();
    if (!value || !value->valid()) return result_error(MathErrorCode::InvalidArgument, __func__, "matrix_trace: invalid matrix");
    auto input = matrix_view(value);
    lmmc_real_t output = 0.0;
    const auto status = lmmc_lsr_linalg_trace(&input, &output);
    return lmmc_real_result("matrix_trace", status, output);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_linear_algebra_rank(MatrixObj* value) noexcept try {
    ensure_lmmc_runtime();
    if (!value || !value->valid()) return result_error(MathErrorCode::InvalidArgument, __func__, "matrix_rank: invalid matrix");
    auto input = matrix_view(value);
    std::size_t output = 0;
    const auto status = lmmc_lsr_linalg_rank(&input, &output);
    if (status != LMMC_STATUS_OK) return result_error(status, "matrix_rank");
    return result_ok(static_cast<LmInt>(output));
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_linear_algebra_solve_left(MatrixObj* lhs,
                                                  MatrixObj* rhs) noexcept try {
    ensure_lmmc_runtime();
    return lmmc_matrix_binary_result("matrix_solve_left", lhs, rhs,
                                     lmmc_lsr_linalg_solve_left);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_linear_algebra_solve_right(MatrixObj* lhs,
                                                   MatrixObj* rhs) noexcept try {
    ensure_lmmc_runtime();
    if (!lhs || !rhs || !lhs->valid() || !rhs->valid()) {
        return result_error(MathErrorCode::InvalidArgument, __func__, "matrix_solve_right: invalid matrix");
    }
    auto left = matrix_view(lhs);
    auto right = matrix_view(rhs);
    lmmc_mat_t output{};
    const auto status = lmmc_lsr_linalg_solve_right(&left, &right, &output);
    return lmmc_matrix_output("matrix_solve_right", status, output);
} catch (...) {
    return c_abi_current_exception(__func__);
}

/** @brief Returns matrix shape as a two-element dense vector. @param value Borrowed valid matrix. @return Owning Result vector `[rows, cols]` or error. @ownership Input borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_linear_algebra_shape(MatrixObj* value) noexcept try {
    ensure_lmmc_runtime();
    if (!value || !value->valid())
        return result_error(MathErrorCode::InvalidArgument, __func__, "linalg.shape: invalid matrix");
    return result_ok(new VectorObj({
        static_cast<double>(value->rows()),
        static_cast<double>(value->cols())}), ValueKind::Vector);
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Computes matrix one-norm. @param value Borrowed valid matrix. @return Owning Result real or error. @ownership Input borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_linear_algebra_one_norm(MatrixObj* value) noexcept try {
    ensure_lmmc_runtime();
    if (!value || !value->valid())
        return result_error(MathErrorCode::InvalidArgument, __func__, "linalg.norm1: invalid matrix");
    auto input = matrix_view(value);
    lmmc_real_t output = 0.0;
    const auto status = lmmc_mat_norm1(&input, &output);
    return lmmc_real_result("linalg.norm1", status, output);
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Computes matrix infinity-norm. @param value Borrowed valid matrix. @return Owning Result real or error. @ownership Input borrowed; caller owns return. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_linear_algebra_infinity_norm(MatrixObj* value) noexcept try {
    ensure_lmmc_runtime();
    if (!value || !value->valid())
        return result_error(MathErrorCode::InvalidArgument, __func__, "linalg.norm_inf: invalid matrix");
    auto input = matrix_view(value); lmmc_real_t output = 0.0;
    const auto status = lmmc_mat_norm_inf(&input, &output);
    return lmmc_real_result("linalg.norm_inf", status, output);
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Solves a dense least-squares problem by QR decomposition. @param matrix Borrowed m-by-n matrix with m at least n. @param rhs Borrowed m-vector. @return Owning Result n-vector or error. @ownership Inputs borrowed; all LMMC temporaries destroyed on every path. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_linear_algebra_least_squares(
    MatrixObj* matrix, VectorObj* rhs) noexcept try {
    ensure_lmmc_runtime();
    if (!matrix || !matrix->valid() || !rhs ||
        matrix->rows() < matrix->cols() || rhs->size() != matrix->rows())
        return result_error(MathErrorCode::InvalidArgument, __func__, "linalg.least_squares: invalid dimensions");
    auto factor = adopt_object(copied_matrix(matrix));
    auto qr = matrix_view(factor.get());
    auto b = vector_view(rhs);
    lmmc_vec_t output{};
    std::vector<double> tau(matrix->cols());
    auto status = lmmc_qr_decompose_inplace(
        &qr, tau.data(), tau.size());
    if (status == LMMC_STATUS_OK)
        status = lmmc_vec_create(matrix->cols(), &output);
    if (status == LMMC_STATUS_OK)
        status = lmmc_qr_solve(&qr, tau.data(), &b, &output);
    return lmmc_vector_output("linalg.least_squares", status, output);
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Solves a dense triangular system. @param matrix Borrowed square triangular matrix. @param rhs Borrowed matching vector. @param upper True for upper triangular. @param unit_diagonal True for implicit unit diagonal. @return Owning Result vector or error. @ownership Inputs borrowed; output destroyed or adopted on every path. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_linear_algebra_solve_triangular(
    MatrixObj* matrix, VectorObj* rhs, bool upper, bool unit_diagonal) noexcept try {
    ensure_lmmc_runtime();
    if (!matrix || !matrix->valid() || !rhs ||
        matrix->rows() != matrix->cols() || rhs->size() != matrix->rows())
        return result_error(MathErrorCode::InvalidArgument, __func__, "linalg.solve_triangular: invalid dimensions");
    auto input = matrix_view(matrix); auto b = vector_view(rhs);
    lmmc_vec_t output{};
    auto status = lmmc_vec_create(rhs->size(), &output);
    if (status == LMMC_STATUS_OK)
        status = lmmc_solve_triangular(
            &input, upper ? 1 : 0, unit_diagonal ? 1 : 0, &b, &output);
    return lmmc_vector_output("linalg.solve_triangular", status, output);
} catch (...) {
    return c_abi_current_exception(__func__);
}
