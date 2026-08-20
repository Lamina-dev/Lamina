#include "matrix.hpp"

#include <functional>
#include <sstream>

namespace lmx::runtime {

namespace {
void combine_hash(std::size_t& seed, const std::size_t value) noexcept {
    seed ^= value + 0x9e3779b9U + (seed << 6U) + (seed >> 2U);
}
}

bool MatrixObj::equals(const MatrixObj& other) const noexcept {
    return rows_ == other.rows_ && cols_ == other.cols_ && data_ == other.data_;
}

std::size_t MatrixObj::hash() const noexcept {
    std::size_t result = std::hash<std::size_t>{}(rows_);
    combine_hash(result, std::hash<std::size_t>{}(cols_));
    for (const auto value : data_) combine_hash(result, std::hash<double>{}(value));
    return result;
}

std::string MatrixObj::to_string() const noexcept {
    std::ostringstream out;
    out << '[';
    for (std::size_t row = 0; row < rows_; ++row) {
        if (row != 0) out << ", ";
        out << '[';
        for (std::size_t col = 0; col < cols_; ++col) {
            if (col != 0) out << ", ";
            out << data_[row * cols_ + col];
        }
        out << ']';
    }
    out << ']';
    return out.str();
}

}
