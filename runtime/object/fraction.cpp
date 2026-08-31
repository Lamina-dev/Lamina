#include "fraction.hpp"
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>

using namespace lmx::runtime;

Fraction::Fraction() noexcept = default;
Fraction::~Fraction() noexcept = default;

Fraction::Fraction(const int32_t num, const int32_t den) noexcept
: num(num), den(den) {simplify();}

Fraction::Fraction(const std::string& num_str) noexcept {
    const auto slash = num_str.find('/');
    if (slash != std::string::npos) {
        try {
            const auto parsed_numerator = std::stoll(num_str.substr(0, slash));
            const auto parsed_denominator = std::stoll(num_str.substr(slash + 1));
            if (parsed_numerator >= INT32_MIN && parsed_numerator <= INT32_MAX &&
                parsed_denominator > 0 && parsed_denominator <= INT32_MAX) {
                num = static_cast<int32_t>(parsed_numerator);
                den = static_cast<int32_t>(parsed_denominator);
                simplify();
                return;
            }
        } catch (...) {
        }
        num = 0;
        den = 0;
        return;
    }
    bool negative = false;
    bool has_dot = false;
    bool has_digit = false;
    int32_t scale = 1;
    int64_t value = 0;

    for (const char c : num_str) {
        if (std::isspace(static_cast<unsigned char>(c))) continue;
        if (!has_digit && value == 0 && !has_dot && (c == '+' || c == '-')) {
            negative = c == '-';
            continue;
        }
        if (std::isdigit(static_cast<unsigned char>(c))) {
            has_digit = true;
            value = value * 10 + (c - '0');
            if (has_dot) scale *= 10;
            continue;
        }
        if (c == '.' && !has_dot) {
            has_dot = true;
            continue;
        }
        break;
    }

    num = has_digit ? static_cast<int32_t>(negative ? -value : value) : 0;
    den = scale;
    simplify();
}

void Fraction::simplify() noexcept {
    if (den == 0) return;
    if (num == 0) {
        den = 1;
        return;
    }
    const int32_t g = std::gcd(num, den);
    if (g == 1) return;
    num /= g;

    den /= g;
}
std::string Fraction::to_string() const noexcept {

    auto result = std::to_string(num);
    if (den != 1) result += "/" + std::to_string(den);

    return result;
}

Fraction Fraction::operator*(const Fraction& other) const noexcept {
    return Fraction(num * other.num, den * other.den);
}
Fraction Fraction::operator/(const Fraction& other) const noexcept {
    return Fraction(num * other.den, den * other.num);
}

Fraction Fraction::operator%(const Fraction &other) const noexcept {
    const int32_t divisor = den * other.num;
    if (divisor == 0) return Fraction(0, 1);
    const auto rem = num * other.den % divisor;
    return Fraction(rem, other.den * den);
}

Fraction Fraction::operator-() const noexcept {
    return Fraction(-num, den);
}


#define ReDuce const int32_t lcm = std::lcm(den, other.den); \
const auto new_num1 = num * (lcm / den);\
const auto new_num2 = other.num * (lcm / other.den);


Fraction& Fraction::operator+=(const Fraction &other) noexcept {
    ReDuce
    num = new_num1 + new_num2;
    den = static_cast<int32_t>(lcm);
    simplify();
    return *this;
}

Fraction& Fraction::operator-=(const Fraction &other) noexcept {
    ReDuce
    num = new_num1 - new_num2;
    den = static_cast<int32_t>(lcm);
    simplify();
    return *this;
}

Fraction& Fraction::operator*=(const Fraction &other) noexcept {
    den *= other.den;
    num *= other.num;
    simplify();
    return *this;
}

Fraction& Fraction::operator/=(const Fraction &other) noexcept {
    den *= other.num;
    num *= other.den;
    simplify();
    return *this;
}

Fraction Fraction::operator+(const Fraction& other) const noexcept {
    ReDuce
    return Fraction(new_num1 + new_num2, lcm);
}
Fraction Fraction::operator-(const Fraction& other) const noexcept {
    ReDuce
    return Fraction(new_num1 - new_num2, lcm);
}
#undef ReDuce



Fraction Fraction::clone() const noexcept {
    return Fraction(num, den);
}

std::size_t Fraction::hash() const noexcept {
    auto result = std::hash<int32_t>{}(num);
    result ^= std::hash<int32_t>{}(den) + 0x9e3779b9U +
              (result << 6U) + (result >> 2U);
    return result;
}

bool Fraction::equals(const Fraction* other) const noexcept {
    return num == other->num && den == other->den;
}

bool Fraction::operator==(const Fraction& other) const noexcept {
    return equals(&other);
}
bool Fraction::operator!=(const Fraction& other) const noexcept {
    return !equals(&other);
}

double Fraction::to_float() const noexcept {
    return static_cast<double>(num) / static_cast<double>(den);
}

bool Fraction::operator!=(const int64_t other) const noexcept {
    return other * den != num;
}
bool Fraction::operator==(const int64_t other) const noexcept {
    return other * den == num;
}
bool Fraction::operator>(const Fraction &other) const noexcept {
    return num * other.den > other.num * den;
}
bool Fraction::operator<(const Fraction &other) const noexcept {
    return num * other.den < other.num * den;
}
bool Fraction::operator>=(const Fraction &other) const noexcept {
    return num * other.den >= other.num * den;
}
bool Fraction::operator<=(const Fraction &other) const noexcept {
    return num * other.den <= other.num * den;
}

Fraction Fraction::operator*(const int other) const noexcept {
    return Fraction(num * other, den);
}

Fraction Fraction::operator/(const int other) const noexcept {
    return Fraction(num, den * other);
}

Fraction& Fraction::operator+=(const int other) noexcept {
    num += den * other;
    simplify();
    return *this;
}

Fraction &Fraction::operator-=(const int other) noexcept {
    num -= den * other;
    simplify();
    return *this;
}

Fraction& Fraction::operator*=(const int other) noexcept {
    num *= other;
    simplify();
    return *this;
}

Fraction &Fraction::operator/=(const int other) noexcept {
    den *= other;
    simplify();
    return *this;
}

Fraction Fraction::operator+(const int other) const noexcept {
    return Fraction(num + other * den, den);
}

Fraction Fraction::operator-(const int other) const noexcept {
    return Fraction(num - other * den, den);
}

namespace {
int64_t mod_inverse(int64_t a, const int64_t m) noexcept {
    if (m <= 1) return 0;
    a %= m;
    if (a < 0) a += m;
    int64_t t = 0, newt = 1;
    int64_t r = m, newr = a;
    while (newr != 0) {
        const int64_t q = r / newr;
        const int64_t next_t = t - q * newt;
        t = newt;
        newt = next_t;
        const int64_t next_r = r - q * newr;
        r = newr;
        newr = next_r;
    }
    if (r != 1) return 0;
    if (t < 0) t += m;
    return t;
}

uint64_t add_mod_u(const uint64_t a, const uint64_t b, const uint64_t m) noexcept {
    if (a >= m - b) return a - (m - b);
    return a + b;
}

uint64_t mul_mod_u(uint64_t a, uint64_t b, const uint64_t m) noexcept {
    uint64_t result = 0;
    a %= m;
    b %= m;
    while (b != 0) {
        if ((b & 1U) != 0) result = add_mod_u(result, a, m);
        a = add_mod_u(a, a, m);
        b >>= 1U;
    }
    return result;
}
}

int64_t Fraction::operator%(const int64_t other) const noexcept {
    if (other == 0 || other == std::numeric_limits<int64_t>::min()) return 0;
    const int64_t m = other < 0 ? -other : other;
    const int64_t inv = mod_inverse(den, m);
    if (inv == 0) return 0;
    int64_t n = num;
    const bool negative = n < 0;
    if (negative) n = -n;
    auto result = static_cast<int64_t>(
        mul_mod_u(static_cast<uint64_t>(n), static_cast<uint64_t>(inv),
                  static_cast<uint64_t>(m)));
    if (negative && result != 0) result = m - result;
    return result;
}
