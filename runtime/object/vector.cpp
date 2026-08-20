#include "vector.hpp"

#include <functional>
#include <sstream>

namespace lmx::runtime {

namespace {
void combine_hash(std::size_t& seed, const std::size_t value) noexcept {
    seed ^= value + 0x9e3779b9U + (seed << 6U) + (seed >> 2U);
}
}

bool VectorObj::equals(const VectorObj& other) const noexcept {
    return data_ == other.data_;
}

std::size_t VectorObj::hash() const noexcept {
    std::size_t result = std::hash<std::size_t>{}(data_.size());
    for (const auto value : data_) combine_hash(result, std::hash<double>{}(value));
    return result;
}

std::string VectorObj::to_string() const noexcept {
    std::ostringstream out;
    out << '[';
    for (std::size_t i = 0; i < data_.size(); ++i) {
        if (i != 0) out << ", ";
        out << data_[i];
    }
    out << ']';
    return out.str();
}

}
