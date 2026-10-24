#include "vector.hpp"
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

VectorObj::VectorObj(lmmc_vec_t& data) noexcept
    : Object(ObjectKind::Vector), data_(data) {
    data = {};
}

VectorObj::~VectorObj() noexcept {
    if (auto* owner = std::get_if<lmmc_vec_t>(&data_))
        lmmc_vec_destroy(owner);
}

std::span<const double> VectorObj::data() const noexcept {
    if (const auto* values = std::get_if<std::vector<double>>(&data_))
        return *values;
    const auto& owner = std::get<lmmc_vec_t>(data_);
    return {owner.data, owner.size};
}

std::span<double> VectorObj::data() noexcept {
    if (auto* values = std::get_if<std::vector<double>>(&data_))
        return *values;
    auto& owner = std::get<lmmc_vec_t>(data_);
    return {owner.data, owner.size};
}

void VectorObj::assign(const std::span<const double> values) {
    auto destination = data();
    if (destination.size() == values.size()) {
        if (destination.data() != values.data())
            std::copy(values.begin(), values.end(), destination.begin());
        return;
    }
    std::vector<double> replacement(values.begin(), values.end());
    if (auto* owner = std::get_if<lmmc_vec_t>(&data_))
        lmmc_vec_destroy(owner);
    data_ = std::move(replacement);
}

bool VectorObj::equals(const VectorObj& other) const noexcept {
    return std::ranges::equal(data(), other.data());
}

std::size_t VectorObj::hash() const noexcept {
    std::size_t result = std::hash<std::size_t>{}(size());
    for (const auto value : data()) combine_hash(result, std::hash<double>{}(value));
    return result;
}

std::string VectorObj::to_string() const noexcept {
    const auto values = data();
    std::ostringstream out;
    out << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) out << ", ";
        out << values[i];
    }
    out << ']';
    return out.str();
}

}
