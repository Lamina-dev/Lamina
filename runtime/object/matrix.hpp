#pragma once

#include "object.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace lmx::runtime {

class MatrixObj final : public Object {
    std::size_t rows_;
    std::size_t cols_;
    std::vector<double> data_;

public:
    MatrixObj(std::size_t rows, std::size_t cols, std::vector<double> data) noexcept
        : Object(ObjectKind::Matrix), rows_(rows), cols_(cols), data_(std::move(data)) {}

    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::size_t cols() const noexcept { return cols_; }
    [[nodiscard]] const std::vector<double>& data() const noexcept { return data_; }
    [[nodiscard]] std::vector<double>& data() noexcept { return data_; }
    [[nodiscard]] bool valid() const noexcept { return data_.size() == rows_ * cols_; }
    [[nodiscard]] bool equals(const MatrixObj& other) const noexcept;
    [[nodiscard]] std::size_t hash() const noexcept;
    [[nodiscard]] std::string to_string() const noexcept;
};

}
