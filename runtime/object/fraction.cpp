#include "fraction.hpp"
#include <cmath>
#include <numeric>

using namespace lmx::runtime;

Fraction::Fraction() noexcept = default;
Fraction::~Fraction() noexcept = default;

Fraction::Fraction(const int32_t num, const int32_t den) noexcept
: num(num), den(den) {simplify();}

Fraction::Fraction(const std::string& num_str) noexcept {
    std::string num_s;
    std::string den_s;
    std::string* target = &num_s;
    bool has_dot = false;
    for (const char c : num_str) {
        if (isdigit(c)) *target += c;
        else if (c == '.') {
            if (has_dot) break;
            has_dot = true;
            target = &den_s;
        }
        else if (isspace(c)) continue;
        else break;
    }

    this->den = static_cast<int32_t>(std::pow(10, den_s.size()));

    this->num = std::stoi(num_s + den_s);
    simplify();
}
// static int64_t private_gcd(int64_t a, int64_t b) noexcept {
//     a = a < 0 ? -a : a;
//     b = b < 0 ? -b : b;
//
//     while (b != 0) {
//         const int64_t temp = b;
//         b = a % b;
//         a = temp;
//     }
//     return a;
// }
// static int64_t private_lcm(int64_t a, int64_t b) noexcept {
//     if (a == 0 || b == 0) return 0;
//     a = a < 0 ? -a : a;
//     b = b < 0 ? -b : b;
//
//     int64_t lcm = a;
//     int64_t step = a;
//
//     while (lcm % b != 0) {
//         if (lcm > INT64_MAX - step) return 0;
//         lcm += step;
//     }
//     return lcm;
// }

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
    const auto rem = num * other.den % (den * other.num);
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
    return static_cast<double>(num) / den;
}

bool Fraction::operator!=(const int64_t other) const noexcept {
    return other * den != num;
}

bool Fraction::operator==(const int64_t other) const noexcept {
    return other * den != num;
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

int64_t Fraction::operator%(const int64_t other) const noexcept {
    return num * static_cast<int64_t>(std::pow(den, -1)) % other;
}
