
#include "tuple.hpp"

#include <functional>
#include <sstream>

namespace lmx::runtime {

namespace {
void combine_hash(std::size_t& seed, const std::size_t value) noexcept {
    seed ^= value + 0x9e3779b9U + (seed << 6U) + (seed >> 2U);
}
}

bool TupleObj::equals(const TupleObj& other) const noexcept {
    if (len_ != other.len_) return false;
    for (std::size_t i = 0; i < len_; ++i) {
        if (data[i] != other.data[i]) return false;
    }
    return true;
}

std::size_t TupleObj::hash() const noexcept {
    std::size_t result = std::hash<std::size_t>{}(len_);
    for (std::size_t i = 0; i < len_; ++i) combine_hash(result, data[i].hash());
    return result;
}

std::string TupleObj::to_string() const noexcept {
    std::ostringstream out;
    out << '(';
    for (std::size_t i = 0; i < len_; ++i) {
        if (i != 0) out << ", ";
        out << data[i].to_string();
    }
    if (len_ == 1) out << ',';
    out << ')';
    return out.str();
}
}
