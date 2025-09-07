#pragma once
#include <algorithm>
#include <climits>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
class BigInt {
public:
    bool negative;
    std::vector<unsigned char> digits;// 存储数字，低位在前

    // 友元类声明
    friend class Fraction;
    // 构造函数
    BigInt() : negative(false), digits({0}) {}

    explicit BigInt(int n) : negative(n < 0) {
        if (n == 0) {
            digits.push_back(0);
            return;
        }

        // 处理 INT_MIN 的特殊情况
        if (n == INT_MIN) {
            // INT_MIN 的绝对值超出 int 范围，直接处理
            auto ln = static_cast<long long>(n);
            ln = -ln;// 现在安全了
            while (ln > 0) {
                digits.push_back(ln % 10);
                ln /= 10;
            }
        } else {
            n = std::abs(n);
            while (n > 0) {
                digits.push_back(n % 10);
                n /= 10;
            }
        }
    }

    explicit BigInt(const std::string& str) : negative(false), digits({}) {
        if (str.empty() || str == "0") {
            digits.push_back(0);
            return;
        }

        size_t start = 0;
        if (str[0] == '-') {
            negative = true;
            start = 1;
        } else if (str[0] == '+') {
            start = 1;
        }

        for (size_t i = str.length() - 1; i >= start and i < str.length(); i--) {
            if (str[i] >= '0' && str[i] <= '9') {
                digits.push_back(str[i] - '0');
            }
        }

        if (digits.empty()) {
            digits.push_back(0);
        }

        remove_leading_zeros();
    }

    // 移除前导零
    void remove_leading_zeros() {
        while (digits.size() > 1 && digits.back() == 0) {
            digits.pop_back();
        }
        if (digits.size() == 1 && digits[0] == 0) {
            negative = false;
        }
    }

    // 转换为字符串
    [[nodiscard]] std::string to_string() const {
        if (digits.size() == 1 && digits[0] == 0) {
            return "0";
        }

        std::string result;
        if (negative) result += "-";

        for (size_t i = digits.size() - 1; i < digits.size(); i--) {
            result += static_cast<char>('0' + digits[i]);
        }

        return result;
    }

    // 快速乘10
    void mul_pow10(size_t n /* 乘10次数 */) {
        if (n <= 0) return;
        remove_leading_zeros();
        for (size_t i = 0; i < n; i++) digits.push_back(0);
        for (size_t i = digits.size() - 1; i >= n; i--) {
            digits[i] = digits[i - n];
        }
        for (size_t i = n - 1; i < n; i--) digits[i] = 0;
    }

    // 返回末尾0的数量
    size_t count_end_zero() {
        if (is_zero()) return 0;
        size_t i = digits.size() - 1;
        while (!digits[i]) i--;
        return digits.size() - 1 - i;
    }

    //去除掉末尾的0
    void del_end_zero() {
        if (is_zero()) return;
        while (!digits.back()) digits.pop_back();
    }

    // 乘法
    BigInt operator*(const BigInt& other) const {
        BigInt result;
        BigInt numa = *this, numb = other;
        size_t end_zeros = numa.count_end_zero() + numb.count_end_zero();
        numa.del_end_zero();
        numb.del_end_zero();    //统计并去除末尾的0
        numa.negative = false;
        numb.negative = false;
        result.digits.assign(digits.size() + other.digits.size(), 0);

        if (numa.digits.size() >> 7 and numb.digits.size() >> 7) {
            // 等价于numa.digits.size() >= 128 and numb.digits.size() >= 128
            // 如果两数长度均大于50使用卡拉楚巴算法优化
            // 算法思想可以将两个四位数乘法看成
            // (100a + b)(100c + d) = 10000ac + 100(bc + ad) + bd
            // 时间复杂度O(n^1.585)   = 10000ac + 100[(a + b)(c + d) - ac - bd] + bd
            // 虽然n基本到了100位以上才有显著提升（一倍）,不过是针对pow才做的优化

            size_t moven = numb.digits.size() >> 1;
            if (numa.digits.size() > numb.digits.size()) moven = numa.digits.size() >> 1 /*除2*/;   //使用位数较大的一方作为移动基准
            BigInt a = numa >> moven;
            BigInt b = numa << (numa.digits.size() - moven);
            // b.remove_leading_zeros();
            BigInt c = numb >> moven;
            BigInt d = numb << (numb.digits.size() - moven);
            // d.remove_leading_zeros();

            BigInt bd = b * d;
            result = a * c; // 使用result记录ac节省内存
            BigInt ad_bc = (a + b) * (c + d) - result - bd;
            result.mul_pow10(moven);
            result = result + ad_bc;
            result.mul_pow10(moven);
            result = result + bd;

        } else {
            // 否则暴力计算
            for (size_t i = 0; i < numa.digits.size(); i++) {
                for (size_t j = 0; j < numb.digits.size(); j++) {
                    result.digits[i + j] += numa.digits[i] * numb.digits[j];
                    if (result.digits[i + j] >= 10) {
                        result.digits[i + j + 1] += result.digits[i + j] / 10;
                        result.digits[i + j] %= 10;
                    }
                }
            }
        }
        result.negative = (negative != other.negative);
        result.remove_leading_zeros();
        result.mul_pow10(end_zeros);

        return result;
    }

    // 加法
    BigInt operator+(const BigInt& other) const {
        if (negative == other.negative) {
            // 同号相加
            BigInt result;
            result.negative = negative;
            result.digits.clear();
            size_t maxn = other.digits.size();
            if (digits.size() > other.digits.size()) maxn = digits.size();
            result.digits.reserve(maxn + 1);
            int carry = 0;
            for (size_t i = 0; i < maxn || carry; ++i) {
                carry += (i < digits.size() ? digits[i] : 0) + (i < other.digits.size() ? other.digits[i] : 0);
                result.digits.push_back(carry % 10);
                carry /= 10;
            }
            result.remove_leading_zeros();
            return result;
        } else {
            // 异号相加，转换为减法
            // a + (-b) = a - b
            // (-a) + b = b - a
            BigInt other_copy = other;
            other_copy.negative = !other_copy.negative;
            return *this - other_copy;
        }
    }

    // 减法
    BigInt operator-(const BigInt& other) const {
        if (negative != other.negative) {
            // 异号相减
            // a - (-b) = a + b
            // (-a) - b = -(a + b)
            BigInt other_copy = other;
            other_copy.negative = !other_copy.negative;
            return *this + other_copy;
        }

        // 同号相减
        auto p1 = this;
        const BigInt* p2 = &other;
        bool result_negative = negative;

        if (abs_compare(*p1, *p2) < 0) {
            std::swap(p1, p2);
            result_negative = !result_negative;
        }

        BigInt result;
        result.negative = result_negative;
        result.digits.clear();
        int borrow = 0, diff;
        for (size_t i = 0; i < p1->digits.size(); ++i) {
            diff = p1->digits[i] - (i < p2->digits.size() ? p2->digits[i] : 0) - borrow;
            if (diff < 0) {
                diff += 10;
                borrow = 1;
            } else {
                borrow = 0;
            }
            result.digits.push_back(diff);
        }
        result.remove_leading_zeros();
        return result;
    }

    // 比较绝对值大小
    static int abs_compare(const BigInt& a, const BigInt& b) {
        if (a.digits.size() != b.digits.size()) {
            return a.digits.size() < b.digits.size() ? -1 : 1;
        }
        for (size_t i = a.digits.size() - 1; i < a.digits.size(); --i) {
            if (a.digits[i] != b.digits[i]) {
                return a.digits[i] < b.digits[i] ? -1 : 1;
            }
        }
        return 0;
    }

    // 转换为int（如果可能）
    [[nodiscard]] int to_int() const {
        if (is_zero()) return 0;

        if (digits.size() > 10) {// 太大了
            return negative ? INT_MIN : INT_MAX;
        }

        long long result = 0;

        for (size_t i = digits.size() - 1; i < digits.size(); i--) {
            result = result * 10 + digits[i];
            if (!negative && result > INT_MAX) {
                return INT_MAX;
            }
            // For negative numbers, the check is against -(long long)INT_MIN
            if (negative && result > -static_cast<long long>(INT_MIN)) {
                return INT_MIN;
            }
        }

        if (negative) {
            long long neg_res = -result;
            if (neg_res < INT_MIN) return INT_MIN;
            return static_cast<int>(neg_res);
        }

        return static_cast<int>(result);
    }

    // 检查是否为零
    [[nodiscard]] bool is_zero() const {
        return digits.size() == 1 && digits[0] == 0;
    }

    // 除法（整数除法）
    BigInt operator/(const BigInt& other) const {
        if (other.is_zero()) {
            throw std::runtime_error("Division by zero");
        }

        if (is_zero()) {
            return BigInt(0);
        }

        // 使用长除法算法
        BigInt dividend = *this;
        dividend.negative = false;
        BigInt divisor = other;
        divisor.negative = false;

        if (abs_compare(dividend, divisor) < 0) {
            return BigInt(0);
        }

        BigInt quotient;
        quotient.digits.clear();

        BigInt current(0);
        for (size_t i = dividend.digits.size() - 1; i < dividend.digits.size(); i--) {
            current.digits.insert(current.digits.begin(), dividend.digits[i]);
            current.remove_leading_zeros();

            int count = 0;
            while (abs_compare(current, divisor) >= 0) {
                current = current - divisor;
                count++;
            }
            quotient.digits.insert(quotient.digits.begin(), count);
        }

        quotient.negative = (negative != other.negative);
        quotient.remove_leading_zeros();
        return quotient;
    }

    // 取模运算
    BigInt operator%(const BigInt& other) const {
        if (other.is_zero()) {
            throw std::runtime_error("Modulo by zero");
        }

        BigInt quotient = *this / other;
        BigInt remainder = *this - (quotient * other);
        return remainder;
    }

    //十进制左移运算，注意不是补0，而是忽略左边的n位,补0请使用mul_pow10;
    BigInt operator<<(size_t n) const {
        if (n >= digits.size()) return BigInt(0);
        BigInt re = *this;
        while (n) {
            re.digits.pop_back();
            n--;
        }
        while (!re.digits.back() and re.digits.size() > 1) re.digits.pop_back();
        return re;
    }

    //十进制右移运算
    BigInt operator>>(size_t n) const {
        if (n >= digits.size()) return BigInt(0);
        BigInt re = *this;
        if (n == 0) return re;
        for (size_t i = n; i < re.digits.size(); i++) {
            re.digits[i - n] = re.digits[i];
        }
        while (n) {
            re.digits.pop_back();
            n--;
        }
        return re;
    }

    // 幂运算
    [[nodiscard]] BigInt power(const BigInt& exponent) const {
        if (exponent.negative) {
            throw std::runtime_error("Negative exponent not supported for integer power");
        }

        if (exponent.is_zero()) {
            return BigInt(1);
        }

        if (is_zero()) {
            return BigInt(0);
        }

        BigInt result(1);
        BigInt base = *this;
        BigInt exp = exponent;

        while (!exp.is_zero()) {
            if (exp.digits[0] % 2 == 1) {   // 如果指数是奇数
                result = result * base;
            }
            base = base * base;
            exp = exp / BigInt(2);
        }

        return result;
    }

    // 阶乘
    static BigInt factorial(const BigInt& n) {
        if (n.negative) {
            throw std::runtime_error("Factorial of negative number is undefined");
        }

        if (n.is_zero() || (n.digits.size() == 1 && n.digits[0] == 1)) {
            return BigInt(1);
        }

        BigInt result(1);
        BigInt current(1);

        while (abs_compare(current, n) <= 0) {
            result = result * current;
            current = current + BigInt(1);
        }

        return result;
    }

    // 比较运算符
    bool operator<(const BigInt& other) const {
        if (negative != other.negative) {
            return negative > other.negative;   // 负数小于正数
        }

        if (negative) {
            // 两个都是负数，绝对值大的反而小
            return abs_compare(*this, other) > 0;
        } else {
            // 两个都是正数
            return abs_compare(*this, other) < 0;
        }
    }

    bool operator<=(const BigInt& other) const {
        return *this < other || *this == other;
    }

    bool operator>(const BigInt& other) const {
        return !(*this <= other);
    }

    bool operator>=(const BigInt& other) const {
        return !(*this < other);
    }

    bool operator==(const BigInt& other) const {
        return negative == other.negative && digits == other.digits;
    }

    bool operator!=(const BigInt& other) const {
        return !(*this == other);
    }

    // Mathematical functions

    // Absolute value - returns a copy with positive sign
    [[nodiscard]] BigInt abs() const {
        BigInt result = *this;
        result.negative = false;
        return result;
    }

    // Negate - returns a copy with opposite sign
    [[nodiscard]] BigInt negate() const {
        BigInt result = *this;
        if (!result.is_zero()) {
            result.negative = !result.negative;
        }
        return result;
    }

    // Square root using Newton's method for integer square root
    // Returns the floor of the square root
    [[nodiscard]] BigInt sqrt() const {
        if (negative) {
            throw std::runtime_error("Square root of negative BigInt is undefined");
        }

        if (is_zero()) {
            return BigInt(0);
        }

        if (digits.size() == 1 && digits[0] == 1) {
            return BigInt(1);
        }

        // Newton's method for integer square root
        // Start with an initial guess
        BigInt x = *this;
        BigInt y = (*this + BigInt(1)) / BigInt(2);

        while (y < x) {
            x = y;
            y = (x + (*this / x)) / BigInt(2);
        }

        return x;
    }

    // Check if this BigInt is a perfect square
    [[nodiscard]] bool is_perfect_square() const {
        if (negative) {
            return false;
        }

        BigInt root = sqrt();
        return (root * root) == *this;
    }

    // Power function for non-negative integer exponents (alias for existing power method)
    [[nodiscard]] BigInt pow(const BigInt& exponent) const {
        return power(exponent);
    }

    // 将异或重载为幂
    BigInt operator^(const BigInt& b) {
        return power(b);
    }

    // Greatest Common Divisor using Euclidean algorithm
    static BigInt gcd(const BigInt& a, const BigInt& b) {
        BigInt abs_a = a.abs();
        BigInt abs_b = b.abs();

        if (abs_b.is_zero()) {
            return abs_a;
        }

        return gcd(abs_b, abs_a % abs_b);
    }

    // Least Common Multiple
    static BigInt lcm(const BigInt& a, const BigInt& b) {
        if (a.is_zero() || b.is_zero()) {
            return BigInt(0);
        }

        BigInt gcd_val = gcd(a, b);
        return (a.abs() / gcd_val) * b.abs();
    }

    // Convert to double (with potential precision loss warning)
    [[nodiscard]] double to_double() const {
        if (is_zero()) {
            return 0.0;
        }

        double result = 0.0;
        double multiplier = 1.0;

        for (size_t i = 0; i < digits.size(); ++i) {
            result += digits[i] * multiplier;
            multiplier *= 10.0;

            // Check for overflow - use a large but finite value
            if (multiplier > 1e308) {
                // Precision loss will occur
                break;
            }
        }

        return negative ? -result : result;
    }
};
