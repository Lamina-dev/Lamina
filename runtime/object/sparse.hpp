#pragma once

#include "object.hpp"
#include "lmmc/sparse.h"

#include <cstddef>
#include <string>

namespace lmx::runtime {

class SparseMatrixObj final : public Object {
    lmmc_sparse_mat_t matrix_{};

public:
    explicit SparseMatrixObj(lmmc_sparse_mat_t&& matrix) noexcept;
    ~SparseMatrixObj() noexcept;

    SparseMatrixObj(const SparseMatrixObj&) = delete;
    SparseMatrixObj& operator=(const SparseMatrixObj&) = delete;

    [[nodiscard]] const lmmc_sparse_mat_t& matrix() const noexcept { return matrix_; }
    [[nodiscard]] lmmc_sparse_mat_t& matrix() noexcept { return matrix_; }
    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] bool equals(const SparseMatrixObj& other) const noexcept;
    [[nodiscard]] std::size_t hash() const noexcept;
    [[nodiscard]] std::string to_string() const noexcept;
};

} // namespace lmx::runtime
