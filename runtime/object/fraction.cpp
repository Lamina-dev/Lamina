#include "fraction.hpp"

#include <charconv>
#include <cctype>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string_view>

using namespace lmx::runtime;

namespace {
std::uint64_t magnitude(const std::int64_t value) noexcept {
    return value < 0
        ? static_cast<std::uint64_t>(-(value + 1)) + 1
        : static_cast<std::uint64_t>(value);
}

std::uint64_t gcd_unsigned(std::uint64_t lhs, std::uint64_t rhs) noexcept {
    while (rhs != 0) {
        const auto remainder = lhs % rhs;
        lhs = rhs;
        rhs = remainder;
    }
    return lhs;
}

std::string_view trim(const std::string_view text) noexcept {
    std::size_t first = 0;
    while (first < text.size() &&
           std::isspace(static_cast<unsigned char>(text[first])) != 0) {
        ++first;
    }
    std::size_t last = text.size();
    while (last > first &&
           std::isspace(static_cast<unsigned char>(text[last - 1])) != 0) {
        --last;
    }
    return text.substr(first, last - first);
}

std::int64_t parse_integer(std::string_view text) {
    if (text.empty()) throw std::invalid_argument("empty fraction component");

    bool explicit_plus = text.front() == '+';
    if (explicit_plus) {
        text.remove_prefix(1);
        if (text.empty()) throw std::invalid_argument("empty fraction component");
    }

    std::int64_t value = 0;
    const auto [end, error] =
        std::from_chars(text.data(), text.data() + text.size(), value);
    if (error == std::errc::result_out_of_range) {
        throw std::out_of_range("fraction component is outside int64 range");
    }
    if (error != std::errc{} || end != text.data() + text.size()) {
        throw std::invalid_argument("invalid fraction component");
    }
    return value;
}

std::uint64_t add_mod(
    const std::uint64_t lhs,
    const std::uint64_t rhs,
    const std::uint64_t modulus) noexcept {
    return lhs >= modulus - rhs ? lhs - (modulus - rhs) : lhs + rhs;
}

std::uint64_t multiply_mod(
    std::uint64_t lhs,
    std::uint64_t rhs,
    const std::uint64_t modulus) noexcept {
    std::uint64_t result = 0;
    while (rhs != 0) {
        if ((rhs & 1U) != 0) result = add_mod(result, lhs, modulus);
        rhs >>= 1U;
        if (rhs != 0) lhs = add_mod(lhs, lhs, modulus);
    }
    return result;
}

std::uint64_t modular_inverse(
    const std::uint64_t value,
    const std::uint64_t modulus) {
    std::uint64_t old_remainder = modulus;
    std::uint64_t remainder = value % modulus;
    std::uint64_t old_coefficient = 0;
    std::uint64_t coefficient = 1;

    while (remainder != 0) {
        const auto quotient = old_remainder / remainder;
        const auto next_remainder = old_remainder % remainder;
        const auto product = multiply_mod(quotient % modulus, coefficient, modulus);
        const auto next_coefficient = old_coefficient >= product
            ? old_coefficient - product
            : modulus - (product - old_coefficient);

        old_remainder = remainder;
        remainder = next_remainder;
        old_coefficient = coefficient;
        coefficient = next_coefficient;
    }

    if (old_remainder != 1) {
        throw std::domain_error("fraction denominator has no modular inverse");
    }
    return old_coefficient;
}
}

Fraction::Fraction() noexcept = default;
Fraction::~Fraction() noexcept = default;

Fraction::Fraction(
    const component_type numerator,
    const component_type denominator) {
    assign_normalized(numerator, denominator);
}

Fraction::Fraction(const std::string& input) {
    const auto text = trim(input);
    if (text.empty()) throw std::invalid_argument("empty fraction");

    const auto slash = text.find('/');
    if (slash != std::string_view::npos) {
        if (text.find('/', slash + 1) != std::string_view::npos) {
            throw std::invalid_argument("fraction contains multiple separators");
        }
        assign_normalized(
            parse_integer(text.substr(0, slash)),
            parse_integer(text.substr(slash + 1)));
        return;
    }

    std::size_t index = 0;
    bool negative = false;
    if (text[index] == '+' || text[index] == '-') {
        negative = text[index] == '-';
        if (++index == text.size()) {
            throw std::invalid_argument("fraction has no digits");
        }
    }

    bool seen_dot = false;
    bool seen_digit = false;
    std::uint64_t value = 0;
    std::int64_t scale = 1;
    const auto value_limit = negative
        ? std::uint64_t{1} << 63U
        : static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());

    for (; index < text.size(); ++index) {
        const char character = text[index];
        if (character == '.' && !seen_dot) {
            seen_dot = true;
            continue;
        }
        if (character < '0' || character > '9') {
            throw std::invalid_argument("invalid decimal fraction");
        }

        seen_digit = true;
        const auto digit = static_cast<std::uint64_t>(character - '0');
        if (value > (value_limit - digit) / 10U) {
            throw std::out_of_range("fraction numerator is outside int64 range");
        }
        value = value * 10U + digit;

        if (seen_dot) {
            if (scale > std::numeric_limits<std::int64_t>::max() / 10) {
                throw std::out_of_range("fraction scale is outside int64 range");
            }
            scale *= 10;
        }
    }
    if (!seen_digit) throw std::invalid_argument("fraction has no digits");

    const std::int64_t signed_value =
        negative && value == (std::uint64_t{1} << 63U)
            ? std::numeric_limits<std::int64_t>::min()
            : negative
                ? -static_cast<std::int64_t>(value)
                : static_cast<std::int64_t>(value);
    assign_normalized(signed_value, scale);
}

Fraction Fraction::from_wide(
    const std::int64_t numerator,
    const std::int64_t denominator) {
    Fraction result;
    result.assign_normalized(numerator, denominator);
    return result;
}

void Fraction::assign_normalized(
    std::int64_t numerator,
    std::int64_t denominator) {
    if (denominator == 0) {
        throw std::domain_error("fraction denominator cannot be zero");
    }
    if (numerator == 0) {
        numerator_ = 0;
        denominator_ = 1;
        return;
    }

    const auto common = gcd_unsigned(magnitude(numerator), magnitude(denominator));
    if (common == (std::uint64_t{1} << 63U)) {
        numerator_ = numerator == denominator ? 1 : -1;
        denominator_ = 1;
        return;
    }

    const auto signed_common = static_cast<std::int64_t>(common);
    numerator /= signed_common;
    denominator /= signed_common;
    if (denominator < 0) {
        if (numerator == std::numeric_limits<std::int64_t>::min()) {
            throw std::overflow_error("normalized fraction numerator is outside int64 range");
        }
        numerator = -numerator;
        denominator = -denominator;
    }

    if (numerator < std::numeric_limits<component_type>::min() ||
        numerator > std::numeric_limits<component_type>::max() ||
        denominator > std::numeric_limits<component_type>::max()) {
        throw std::overflow_error("fraction is outside its 32-bit representation");
    }

    numerator_ = static_cast<component_type>(numerator);
    denominator_ = static_cast<component_type>(denominator);
}

Fraction::component_type Fraction::numerator() const noexcept {
    return numerator_;
}

Fraction::component_type Fraction::denominator() const noexcept {
    return denominator_;
}

std::string Fraction::to_string() const {
    auto result = std::to_string(numerator_);
    if (denominator_ != 1) result += "/" + std::to_string(denominator_);
    return result;
}

Fraction Fraction::operator*(const Fraction& other) const {
    return from_wide(
        static_cast<std::int64_t>(numerator_) * other.numerator_,
        static_cast<std::int64_t>(denominator_) * other.denominator_);
}

Fraction Fraction::operator/(const Fraction& other) const {
    return from_wide(
        static_cast<std::int64_t>(numerator_) * other.denominator_,
        static_cast<std::int64_t>(denominator_) * other.numerator_);
}

Fraction Fraction::operator%(const Fraction& other) const {
    if (other.numerator_ == 0) {
        throw std::domain_error("fraction remainder divisor cannot be zero");
    }
    const auto dividend =
        static_cast<std::int64_t>(numerator_) * other.denominator_;
    const auto divisor =
        static_cast<std::int64_t>(denominator_) * other.numerator_;
    return from_wide(
        dividend % divisor,
        static_cast<std::int64_t>(denominator_) * other.denominator_);
}

Fraction Fraction::operator-() const {
    return from_wide(-static_cast<std::int64_t>(numerator_), denominator_);
}

Fraction& Fraction::operator+=(const Fraction& other) {
    *this = *this + other;
    return *this;
}

Fraction& Fraction::operator-=(const Fraction& other) {
    *this = *this - other;
    return *this;
}

Fraction& Fraction::operator*=(const Fraction& other) {
    *this = *this * other;
    return *this;
}

Fraction& Fraction::operator/=(const Fraction& other) {
    *this = *this / other;
    return *this;
}

Fraction Fraction::operator+(const Fraction& other) const {
    return from_wide(
        static_cast<std::int64_t>(numerator_) * other.denominator_ +
            static_cast<std::int64_t>(other.numerator_) * denominator_,
        static_cast<std::int64_t>(denominator_) * other.denominator_);
}

Fraction Fraction::operator-(const Fraction& other) const {
    return from_wide(
        static_cast<std::int64_t>(numerator_) * other.denominator_ -
            static_cast<std::int64_t>(other.numerator_) * denominator_,
        static_cast<std::int64_t>(denominator_) * other.denominator_);
}

Fraction Fraction::clone() const noexcept {
    return *this;
}

std::size_t Fraction::hash() const noexcept {
    auto result = std::hash<component_type>{}(numerator_);
    result ^= std::hash<component_type>{}(denominator_) + 0x9e3779b9U +
              (result << 6U) + (result >> 2U);
    return result;
}

bool Fraction::equals(const Fraction* other) const noexcept {
    return other != nullptr &&
           numerator_ == other->numerator_ &&
           denominator_ == other->denominator_;
}

bool Fraction::operator==(const Fraction& other) const noexcept {
    return equals(&other);
}

bool Fraction::operator!=(const Fraction& other) const noexcept {
    return !equals(&other);
}

double Fraction::to_float() const noexcept {
    return static_cast<double>(numerator_) / static_cast<double>(denominator_);
}

bool Fraction::operator!=(const std::int64_t other) const noexcept {
    return !(*this == other);
}

bool Fraction::operator==(const std::int64_t other) const noexcept {
    return denominator_ == 1 && numerator_ == other;
}

bool Fraction::operator>(const Fraction& other) const noexcept {
    return static_cast<std::int64_t>(numerator_) * other.denominator_ >
           static_cast<std::int64_t>(other.numerator_) * denominator_;
}

bool Fraction::operator<(const Fraction& other) const noexcept {
    return static_cast<std::int64_t>(numerator_) * other.denominator_ <
           static_cast<std::int64_t>(other.numerator_) * denominator_;
}

bool Fraction::operator>=(const Fraction& other) const noexcept {
    return !(*this < other);
}

bool Fraction::operator<=(const Fraction& other) const noexcept {
    return !(*this > other);
}

Fraction Fraction::operator*(const int other) const {
    return from_wide(
        static_cast<std::int64_t>(numerator_) * other,
        denominator_);
}

Fraction Fraction::operator/(const int other) const {
    return from_wide(
        numerator_,
        static_cast<std::int64_t>(denominator_) * other);
}

Fraction& Fraction::operator+=(const int other) {
    *this = *this + other;
    return *this;
}

Fraction& Fraction::operator-=(const int other) {
    *this = *this - other;
    return *this;
}

Fraction& Fraction::operator*=(const int other) {
    *this = *this * other;
    return *this;
}

Fraction& Fraction::operator/=(const int other) {
    *this = *this / other;
    return *this;
}

Fraction Fraction::operator+(const int other) const {
    return from_wide(
        numerator_ + static_cast<std::int64_t>(other) * denominator_,
        denominator_);
}

Fraction Fraction::operator-(const int other) const {
    return from_wide(
        numerator_ - static_cast<std::int64_t>(other) * denominator_,
        denominator_);
}

std::int64_t Fraction::operator%(const std::int64_t modulus) const {
    if (modulus <= 1) {
        throw std::domain_error("fraction modulus must be greater than one");
    }

    const auto unsigned_modulus = static_cast<std::uint64_t>(modulus);
    const auto inverse = modular_inverse(
        static_cast<std::uint64_t>(denominator_),
        unsigned_modulus);
    const auto numerator_magnitude =
        static_cast<std::uint64_t>(numerator_ < 0
            ? -static_cast<std::int64_t>(numerator_)
            : numerator_) % unsigned_modulus;
    const auto numerator_residue = numerator_ < 0 && numerator_magnitude != 0
        ? unsigned_modulus - numerator_magnitude
        : numerator_magnitude;
    return static_cast<std::int64_t>(
        multiply_mod(numerator_residue, inverse, unsigned_modulus));
}
