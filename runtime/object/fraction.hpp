#pragma once
#include "object.hpp"

namespace lmx::runtime {

 struct Fraction {

     int32_t num {0};
     int32_t den {1};

     Fraction() noexcept;

     ~Fraction() noexcept;

     void simplify() noexcept;
     explicit Fraction(int32_t num, int32_t den) noexcept;
     explicit Fraction(const std::string& num_str) noexcept;

     [[nodiscard]] Fraction clone() const noexcept;
     [[nodiscard]] std::string to_string() const noexcept;
     [[nodiscard]] bool equals(const Fraction* other) const noexcept;
     [[nodiscard]] Fraction operator-() const noexcept;
     [[nodiscard]] Fraction operator+(const Fraction& other) const noexcept;
     [[nodiscard]] Fraction operator-(const Fraction& other) const noexcept;
     [[nodiscard]] Fraction operator*(const Fraction& other) const noexcept;
     [[nodiscard]] Fraction operator/(const Fraction& other) const noexcept;
     [[nodiscard]] Fraction operator%(const Fraction& other) const noexcept;

     Fraction& operator+=(const Fraction& other) noexcept;
     Fraction& operator-=(const Fraction& other) noexcept;
     Fraction& operator*=(const Fraction& other) noexcept;
     Fraction& operator/=(const Fraction& other) noexcept;

     [[nodiscard]] bool operator==(const Fraction& other) const noexcept;
     [[nodiscard]] bool operator!=(const Fraction& other) const noexcept;

     [[nodiscard]] double to_float() const noexcept;

     [[nodiscard]] Fraction operator+(int other) const noexcept;
     [[nodiscard]] Fraction operator-(int other) const noexcept;
     [[nodiscard]] Fraction operator*(int other) const noexcept;
     [[nodiscard]] Fraction operator/(int other) const noexcept;
     [[nodiscard]] int64_t operator%(int64_t other) const noexcept;

     Fraction& operator+=(int other) noexcept;
     Fraction& operator-=(int other) noexcept;
     Fraction& operator*=(int other) noexcept;
     Fraction& operator/=(int other) noexcept;

     [[nodiscard]] bool operator==(int64_t other) const noexcept;
     [[nodiscard]] bool operator!=(int64_t other) const noexcept;
};
}