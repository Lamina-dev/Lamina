#include "matrix.hpp"
#include "lmmc/dense_storage.h"

#include <algorithm>
#include <functional>
#include <sstream>

namespace lmx::runtime {

namespace {
void combine_hash(std::size_t& seed, const std::size_t value) noexcept {
    seed ^= value + 0x9e3779b9U + (seed << 6U) + (seed >> 2U);
}
}

MatrixObj::MatrixObj(lmmc_mat_t& data) noexcept
    : Object(ObjectKind::Matrix), rows_(data.rows), cols_(data.cols), data_(data) {
    data = {};
}

MatrixObj::MatrixObj(const std::size_t rows, const std::size_t cols,
                     lmmc_vec_t& data) noexcept
    : Object(ObjectKind::Matrix), rows_(rows), cols_(cols),
      data_(lmmc_mat_t{rows, cols, cols, data.data, data.owns_data}) {
    data = {};
}

MatrixObj::~MatrixObj() noexcept {
    if (auto* owner = std::get_if<lmmc_mat_t>(&data_))
        lmmc_mat_destroy(owner);
}

std::span<const double> MatrixObj::data() const noexcept {
    if (const auto* values = std::get_if<std::vector<double>>(&data_))
        return *values;
    const auto& owner = std::get<lmmc_mat_t>(data_);
    return {owner.data, owner.rows * owner.cols};
}

std::span<double> MatrixObj::data() noexcept {
    if (auto* values = std::get_if<std::vector<double>>(&data_))
        return *values;
    auto& owner = std::get<lmmc_mat_t>(data_);
    return {owner.data, owner.rows * owner.cols};
}

bool MatrixObj::equals(const MatrixObj& other) const noexcept {
    return rows_ == other.rows_ && cols_ == other.cols_ &&
           std::ranges::equal(data(), other.data());
}

std::size_t MatrixObj::hash() const noexcept {
    std::size_t result = std::hash<std::size_t>{}(rows_);
    combine_hash(result, std::hash<std::size_t>{}(cols_));
    for (const auto value : data()) combine_hash(result, std::hash<double>{}(value));
    return result;
}

std::string MatrixObj::to_string() const noexcept {
    const auto values = data();
    std::ostringstream out;
    out << '[';
    for (std::size_t row = 0; row < rows_; ++row) {
        if (row != 0) out << ", ";
        out << '[';
        for (std::size_t col = 0; col < cols_; ++col) {
            if (col != 0) out << ", ";
            out << values[row * cols_ + col];
        }
        out << ']';
    }
    out << ']';
    return out.str();
}

}
