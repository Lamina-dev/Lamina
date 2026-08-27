#include "sparse.hpp"

#include <functional>

namespace lmx::runtime {
namespace {

void hash_combine(std::size_t& seed, const std::size_t value) noexcept {
    seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U);
}

} // namespace

SparseMatrixObj::SparseMatrixObj(lmmc_sparse_mat_t&& matrix) noexcept
    : Object(ObjectKind::Sparse), matrix_(matrix) {
    matrix = {};
}

SparseMatrixObj::~SparseMatrixObj() noexcept {
    lmmc_sparse_destroy(&matrix_);
}

bool SparseMatrixObj::valid() const noexcept {
    if (matrix_.format != LMMC_SPARSE_CSR || matrix_.rows == 0 ||
        matrix_.cols == 0 || !matrix_.row_ptr) return false;
    if (matrix_.row_ptr[0] != 0 || matrix_.row_ptr[matrix_.rows] != matrix_.nnz)
        return false;
    if (matrix_.nnz > 0 && (!matrix_.col_idx || !matrix_.values)) return false;
    for (std::size_t row = 0; row < matrix_.rows; ++row) {
        if (matrix_.row_ptr[row] > matrix_.row_ptr[row + 1] ||
            matrix_.row_ptr[row + 1] > matrix_.nnz) return false;
        std::size_t previous = 0;
        for (std::size_t index = matrix_.row_ptr[row];
             index < matrix_.row_ptr[row + 1]; ++index) {
            if (matrix_.col_idx[index] >= matrix_.cols ||
                (index > matrix_.row_ptr[row] &&
                 matrix_.col_idx[index] <= previous)) return false;
            previous = matrix_.col_idx[index];
        }
    }
    return true;
}

bool SparseMatrixObj::equals(const SparseMatrixObj& other) const noexcept {
    if (matrix_.rows != other.matrix_.rows || matrix_.cols != other.matrix_.cols ||
        matrix_.nnz != other.matrix_.nnz ||
        matrix_.format != other.matrix_.format) return false;
    for (std::size_t i = 0; i <= matrix_.rows; ++i)
        if (matrix_.row_ptr[i] != other.matrix_.row_ptr[i]) return false;
    for (std::size_t i = 0; i < matrix_.nnz; ++i) {
        if (matrix_.col_idx[i] != other.matrix_.col_idx[i] ||
            matrix_.values[i] != other.matrix_.values[i]) return false;
    }
    return true;
}

std::size_t SparseMatrixObj::hash() const noexcept {
    std::size_t result = std::hash<std::size_t>{}(matrix_.rows);
    hash_combine(result, std::hash<std::size_t>{}(matrix_.cols));
    hash_combine(result, std::hash<std::size_t>{}(matrix_.nnz));
    for (std::size_t i = 0; i <= matrix_.rows; ++i)
        hash_combine(result, std::hash<std::size_t>{}(matrix_.row_ptr[i]));
    for (std::size_t i = 0; i < matrix_.nnz; ++i) {
        hash_combine(result, std::hash<std::size_t>{}(matrix_.col_idx[i]));
        hash_combine(result, std::hash<double>{}(matrix_.values[i]));
    }
    return result;
}

std::string SparseMatrixObj::to_string() const noexcept {
    return "sparse(" + std::to_string(matrix_.rows) + "x" +
           std::to_string(matrix_.cols) + ", nnz=" +
           std::to_string(matrix_.nnz) + ")";
}

} // namespace lmx::runtime
