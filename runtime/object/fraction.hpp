#pragma once
#include "object.hpp"

namespace lmx::runtime {

class Fraction {
public:
    using component_type = std::int32_t;

    Fraction() noexcept;
    ~Fraction() noexcept;
    explicit Fraction(component_type numerator, component_type denominator);
    explicit Fraction(const std::string& text);

    [[nodiscard]] component_type numerator() const noexcept;
    [[nodiscard]] component_type denominator() const noexcept;
    [[nodiscard]] Fraction clone() const noexcept;
    [[nodiscard]] std::size_t hash() const noexcept;
    [[nodiscard]] std::string to_string() const;
    [[nodiscard]] bool equals(const Fraction* other) const noexcept;
    [[nodiscard]] Fraction operator-() const;
    [[nodiscard]] Fraction operator+(const Fraction& other) const;
    [[nodiscard]] Fraction operator-(const Fraction& other) const;
    [[nodiscard]] Fraction operator*(const Fraction& other) const;
    [[nodiscard]] Fraction operator/(const Fraction& other) const;
    [[nodiscard]] Fraction operator%(const Fraction& other) const;

    Fraction& operator+=(const Fraction& other);
    Fraction& operator-=(const Fraction& other);
    Fraction& operator*=(const Fraction& other);
    Fraction& operator/=(const Fraction& other);

    [[nodiscard]] bool operator==(const Fraction& other) const noexcept;
    [[nodiscard]] bool operator!=(const Fraction& other) const noexcept;
    [[nodiscard]] bool operator>=(const Fraction& other) const noexcept;
    [[nodiscard]] bool operator<=(const Fraction& other) const noexcept;
    [[nodiscard]] bool operator>(const Fraction& other) const noexcept;
    [[nodiscard]] bool operator<(const Fraction& other) const noexcept;

    [[nodiscard]] double to_float() const noexcept;

    [[nodiscard]] Fraction operator+(int other) const;
    [[nodiscard]] Fraction operator-(int other) const;
    [[nodiscard]] Fraction operator*(int other) const;
    [[nodiscard]] Fraction operator/(int other) const;
    [[nodiscard]] std::int64_t operator%(std::int64_t modulus) const;

    Fraction& operator+=(int other);
    Fraction& operator-=(int other);
    Fraction& operator*=(int other);
    Fraction& operator/=(int other);

    [[nodiscard]] bool operator==(std::int64_t other) const noexcept;
    [[nodiscard]] bool operator!=(std::int64_t other) const noexcept;

private:
    component_type numerator_{0};
    component_type denominator_{1};

    static Fraction from_wide(std::int64_t numerator, std::int64_t denominator);
    void assign_normalized(std::int64_t numerator, std::int64_t denominator);
};
}
