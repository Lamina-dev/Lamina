#include "bridge/result.hpp"
#include "bridge/conversions.hpp"
#include "bridge/runtime_views.hpp"
#include "bridge/unit_bridge.hpp"
#include <cstdarg>
#include "bridge/lmmc_internal.hpp"
#include "runtime/object/sparse.hpp"
#include "lmmc/sparse.h"

using namespace lmx::bridge;

namespace {
using lmx::runtime::SparseMatrixObj;

AdtObj* sparse_output(
    const char* name, const lmmc_status_t status, lmmc_sparse_mat_t& output) {
    if (status != LMMC_STATUS_OK) {
        lmmc_sparse_destroy(&output);
        return result_error(status, name);
    }
    auto* result = new SparseMatrixObj(std::move(output));
    if (!result->valid()) {
        result->release();
        return result_error(MathErrorCode::InvalidArgument, __func__, std::string(name) + ": invalid CSR output");
    }
    return result_ok(result, ValueKind::Sparse);
}

bool checked_sparse(SparseMatrixObj* value, std::string& error) {
    if (!value || !value->valid()) {
        error = "invalid sparse matrix";
        return false;
    }
    return true;
}
} // namespace

extern "C" LM_API AdtObj* lmx_sparse_linear_algebra_from_dense(
    MatrixObj* value, const double epsilon) noexcept try {
    ensure_lmmc_runtime();
    if (!value || !value->valid() || epsilon < 0.0 ||
        !std::isfinite(epsilon))
        return result_error(MathErrorCode::InvalidArgument, __func__, "sparse.from_dense: invalid argument");
    auto input = matrix_view(value);
    lmmc_sparse_mat_t output{};
    return sparse_output(
        "sparse.from_dense",
        lmmc_sparse_from_dense(&input, epsilon, &output), output);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_sparse_linear_algebra_from_triplets(
    const LmInt rows, const LmInt cols, ArrayObj* row_indices,
    ArrayObj* col_indices, ArrayObj* values) noexcept try {
    ensure_lmmc_runtime();
    if (rows <= 0 || cols <= 0)
        return result_error(MathErrorCode::InvalidArgument, __func__, "sparse.from_triplets: invalid shape");
    std::vector<std::size_t> row_values;
    std::vector<std::size_t> col_values;
    std::vector<double> numeric_values;
    std::string error;
    if (!checked_nonnegative_ints(row_indices, row_values, error) ||
        !checked_nonnegative_ints(col_indices, col_values, error) ||
        !array_numbers(values, numeric_values, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "sparse.from_triplets: " + error);
    if (row_values.size() != col_values.size() ||
        row_values.size() != numeric_values.size())
        return result_error(MathErrorCode::InvalidArgument, __func__, 
            "sparse.from_triplets: array lengths differ");
    lmmc_sparse_builder_t* builder = nullptr;
    auto status = lmmc_sparse_builder_create(
        static_cast<std::size_t>(rows), static_cast<std::size_t>(cols),
        row_values.size(), &builder);
    if (status == LMMC_STATUS_OK) {
        for (std::size_t i = 0; i < row_values.size(); ++i) {
            if (!std::isfinite(numeric_values[i])) {
                status = LMMC_STATUS_NUMERICAL_FAILURE;
                break;
            }
            status = lmmc_sparse_builder_add(
                builder, row_values[i], col_values[i], numeric_values[i]);
            if (status != LMMC_STATUS_OK) break;
        }
    }
    lmmc_sparse_mat_t output{};
    if (status == LMMC_STATUS_OK)
        status = lmmc_sparse_builder_build(
            builder, LMMC_SPARSE_CSR, &output);
    lmmc_sparse_builder_destroy(builder);
    return sparse_output("sparse.from_triplets", status, output);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_sparse_linear_algebra_to_dense(SparseMatrixObj* value) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    if (!checked_sparse(value, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "sparse.to_dense: " + error);
    lmmc_mat_t output{};
    auto status = lmmc_mat_create(
        value->matrix().rows, value->matrix().cols, &output);
    if (status == LMMC_STATUS_OK)
        status = lmmc_sparse_to_dense(&value->matrix(), &output);
    return lmmc_matrix_output("sparse.to_dense", status, output);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API LmInt lmx_sparse_linear_algebra_row_count(SparseMatrixObj* value) noexcept try {
    ensure_lmmc_runtime();
    return value && value->matrix().rows <= static_cast<std::size_t>(
                                                std::numeric_limits<LmInt>::max())
        ? static_cast<LmInt>(value->matrix().rows) : -1;
} catch (...) {
    return -1;
}
extern "C" LM_API LmInt lmx_sparse_linear_algebra_column_count(SparseMatrixObj* value) noexcept try {
    ensure_lmmc_runtime();
    return value && value->matrix().cols <= static_cast<std::size_t>(
                                                std::numeric_limits<LmInt>::max())
        ? static_cast<LmInt>(value->matrix().cols) : -1;
} catch (...) {
    return -1;
}
extern "C" LM_API LmInt lmx_sparse_linear_algebra_nonzero_count(SparseMatrixObj* value) noexcept try {
    ensure_lmmc_runtime();
    return value && value->matrix().nnz <= static_cast<std::size_t>(
                                               std::numeric_limits<LmInt>::max())
        ? static_cast<LmInt>(value->matrix().nnz) : -1;
} catch (...) {
    return -1;
}

extern "C" LM_API AdtObj* lmx_sparse_linear_algebra_transpose(SparseMatrixObj* value) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    if (!checked_sparse(value, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "sparse.transpose: " + error);
    lmmc_sparse_mat_t output{};
    return sparse_output(
        "sparse.transpose",
        lmmc_sparse_transpose(&value->matrix(), &output), output);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_sparse_linear_algebra_add(
    SparseMatrixObj* lhs, SparseMatrixObj* rhs) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    if (!checked_sparse(lhs, error) || !checked_sparse(rhs, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "sparse.add: " + error);
    lmmc_sparse_mat_t output{};
    return sparse_output(
        "sparse.add",
        lmmc_sparse_add(
            1.0, &lhs->matrix(), 1.0, &rhs->matrix(), &output),
        output);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_sparse_linear_algebra_multiply(
    SparseMatrixObj* lhs, SparseMatrixObj* rhs) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    if (!checked_sparse(lhs, error) || !checked_sparse(rhs, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "sparse.mul: " + error);
    lmmc_sparse_mat_t output{};
    return sparse_output(
        "sparse.mul",
        lmmc_sparse_mat_mat_mul_sparse(
            &lhs->matrix(), &rhs->matrix(), &output),
        output);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_sparse_linear_algebra_matrix_vector_product(
    SparseMatrixObj* matrix, VectorObj* vector) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    if (!checked_sparse(matrix, error) || !vector)
        return result_error(MathErrorCode::InvalidArgument, __func__, "sparse.matvec: invalid argument");
    auto input = vector_view(vector);
    std::vector<double> data(matrix->matrix().rows);
    lmmc_vec_t output{data.size(), data.data(), 0};
    const auto status =
        lmmc_sparse_mat_vec_mul(&matrix->matrix(), &input, &output);
    if (status != LMMC_STATUS_OK)
        return result_error(status, "sparse.matvec");
    return result_ok(new VectorObj(std::move(data)), ValueKind::Vector);
} catch (...) {
    return c_abi_current_exception(__func__);
}

extern "C" LM_API AdtObj* lmx_sparse_linear_algebra_norm(SparseMatrixObj* value) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    if (!checked_sparse(value, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "sparse.norm: " + error);
    double output = 0.0;
    const auto status =
        lmmc_sparse_norm_fro(&value->matrix(), &output);
    return lmmc_real_result("sparse.norm", status, output);
} catch (...) {
    return c_abi_current_exception(__func__);
}

/** @brief Subtracts sparse matrices. @param lhs Borrowed sparse matrix. @param rhs Borrowed sparse matrix. @return Owning Result sparse matrix or error. @ownership Inputs borrowed; output destroyed or adopted on every path. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_sparse_linear_algebra_subtract(
    SparseMatrixObj* lhs, SparseMatrixObj* rhs) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    if (!checked_sparse(lhs, error) || !checked_sparse(rhs, error))
        return result_error(MathErrorCode::InvalidArgument, __func__, "sparse.sub: " + error);
    lmmc_sparse_mat_t output{};
    return sparse_output("sparse.sub", lmmc_sparse_add(
        1.0, &lhs->matrix(), -1.0, &rhs->matrix(), &output), output);
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Returns a scaled sparse matrix. @param value Borrowed sparse matrix. @param scalar Finite scale. @return Owning Result sparse matrix or error. @ownership Input borrowed; output destroyed or adopted on every path. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_sparse_linear_algebra_scale(
    SparseMatrixObj* value, double scalar) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    if (!checked_sparse(value, error) || !std::isfinite(scalar))
        return result_error(MathErrorCode::InvalidArgument, __func__, "sparse.scale: invalid argument");
    lmmc_sparse_mat_t output{};
    return sparse_output("sparse.scale", lmmc_sparse_add(
        scalar, &value->matrix(), 0.0, &value->matrix(), &output), output);
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Extracts sparse main diagonal. @param value Borrowed square sparse matrix. @return Owning Result vector or error. @ownership Input borrowed; output destroyed or adopted on every path. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_sparse_linear_algebra_diagonal(SparseMatrixObj* value) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    if (!checked_sparse(value, error) || value->matrix().rows != value->matrix().cols)
        return result_error(MathErrorCode::InvalidArgument, __func__, "sparse.diag: invalid square matrix");
    lmmc_vec_t output{};
    return lmmc_vector_output(
        "sparse.diag", lmmc_sparse_diag(&value->matrix(), &output), output);
} catch (...) {
    return c_abi_current_exception(__func__);
}
/** @brief Multiplies sparse by dense matrix. @param sparse Borrowed sparse matrix. @param dense Borrowed dense matrix. @return Owning Result dense matrix or error. @ownership Inputs borrowed; output destroyed or adopted on every path. @threadsafe Current VM thread only. */
extern "C" LM_API AdtObj* lmx_sparse_linear_algebra_matmul_dense(
    SparseMatrixObj* sparse, MatrixObj* dense) noexcept try {
    ensure_lmmc_runtime();
    std::string error;
    if (!checked_sparse(sparse, error) || !dense || !dense->valid() ||
        sparse->matrix().cols != dense->rows())
        return result_error(MathErrorCode::InvalidArgument, __func__, "sparse.matmul_dense: invalid dimensions");
    auto right = matrix_view(dense); lmmc_mat_t output{};
    auto status = lmmc_mat_create(
        sparse->matrix().rows, dense->cols(), &output);
    if (status == LMMC_STATUS_OK)
        status = lmmc_sparse_mat_mat_mul_dense(
            &sparse->matrix(), &right, &output);
    return lmmc_matrix_output("sparse.matmul_dense", status, output);
} catch (...) {
    return c_abi_current_exception(__func__);
}
