#pragma once

#include "object.hpp"

#include <cstddef>
#include <string>

namespace lmx::runtime {

class ComplexObj : public Object {
    double real_;
    double imag_;

public:
    ComplexObj(double real, double imag) noexcept;

    [[nodiscard]] double real() const noexcept { return real_; }
    [[nodiscard]] double imag() const noexcept { return imag_; }
    [[nodiscard]] bool equals(const ComplexObj& other) const noexcept;
    [[nodiscard]] std::size_t hash() const noexcept;
    [[nodiscard]] std::string to_string() const noexcept;
};

}
