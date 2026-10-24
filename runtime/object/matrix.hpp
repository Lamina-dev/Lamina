#pragma once

#include "object.hpp"
#include "lmmc/dense_types.h"

#include <cstddef>
#include <limits>
#include <span>
#include <string>
#include <vector>
#include <variant>

namespace lmx::runtime {

class MatrixObj final : public Object {
    std::size_t rows_;
    std::size_t cols_;
    std::variant<std::vector<double>, lmmc_mat_t> data_;

public:
    MatrixObj(std::size_t rows, std::size_t cols, std::vector<double> data) noexcept
        : Object(ObjectKind::Matrix), rows_(rows), cols_(cols), data_(std::move(data)) {}
    explicit MatrixObj(lmmc_mat_t& data) noexcept;
    MatrixObj(std::size_t rows, std::size_t cols, lmmc_vec_t& data) noexcept;
    ~MatrixObj() noexcept;

    MatrixObj(const MatrixObj&) = delete;
    MatrixObj& operator=(const MatrixObj&) = delete;
    MatrixObj(MatrixObj&&) = delete;
    MatrixObj& operator=(MatrixObj&&) = delete;

    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::size_t cols() const noexcept { return cols_; }
    [[nodiscard]] std::span<const double> data() const noexcept;
    [[nodiscard]] std::span<double> data() noexcept;
    [[nodiscard]] bool valid() const noexcept {
        return (cols_ == 0 ||
                rows_ <= std::numeric_limits<std::size_t>::max() / cols_) &&
               data().size() == rows_ * cols_;
    }
    [[nodiscard]] bool equals(const MatrixObj& other) const noexcept;
    [[nodiscard]] std::size_t hash() const noexcept;
    [[nodiscard]] std::string to_string() const noexcept;
};

}
