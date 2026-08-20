#pragma once

#include "object.hpp"

#include <cstddef>
#include <string>

namespace lmx::runtime {

class QuantityObj final : public Object {
    double si_value_;
    std::string unit_;

public:
    QuantityObj(double si_value, std::string unit) noexcept
        : Object(ObjectKind::Quantity), si_value_(si_value), unit_(std::move(unit)) {}

    [[nodiscard]] double si_value() const noexcept { return si_value_; }
    [[nodiscard]] const std::string& unit() const noexcept { return unit_; }
    [[nodiscard]] bool equals(const QuantityObj& other) const noexcept;
    [[nodiscard]] std::size_t hash() const noexcept;
    [[nodiscard]] std::string to_string() const noexcept;
};

}
