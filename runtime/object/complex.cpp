#include "complex.hpp"

#include <functional>
#include <sstream>

using namespace lmx::runtime;

ComplexObj::ComplexObj(const double real, const double imag) noexcept
    : Object(ObjectKind::Complex), real_(real), imag_(imag) {}

bool ComplexObj::equals(const ComplexObj& other) const noexcept {
    return real_ == other.real_ && imag_ == other.imag_;
}

std::size_t ComplexObj::hash() const noexcept {
    auto result = std::hash<double>{}(real_);
    result ^= std::hash<double>{}(imag_) + 0x9e3779b9U +
              (result << 6U) + (result >> 2U);
    return result;
}

std::string ComplexObj::to_string() const noexcept {
    std::ostringstream out;
    out << real_;
    if (imag_ >= 0.0) out << " + ";
    else out << " - ";
    out << (imag_ >= 0.0 ? imag_ : -imag_) << 'I';
    return out.str();
}
