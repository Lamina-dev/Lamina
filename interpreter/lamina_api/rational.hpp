#pragma once
#include "bigint.hpp"
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <stdint.h>

class Rational {
private:
    BigInt numerator;
    BigInt denominator;

    // 求最大公约数
    static BigInt gcd(const BigInt& a, const BigInt& b) {
        BigInt abs_a = a;
        abs_a.negative = false;
        BigInt abs_b = b;
        abs_b.negative = false;

        while (!abs_b.is_zero()) {
            BigInt temp = abs_b;
            abs_b = abs_a % abs_b;
            abs_a = temp;
        }
        return abs_a;
    }

    // 化简分数
    void simplify() {
        if (denominator.is_zero()) {
            throw std::runtime_error("Denominator cannot be zero");
        }

        // 确保分母为正数
        if (denominator.negative) {
            numerator.negative = !numerator.negative;
            denominator.negative = false;
        }

        // 化简
        BigInt g = gcd(numerator, denominator);
        if (!g.is_zero() && !(g.digits.size() == 1 && g.digits[0] == 1)) {
            numerator = numerator / g;
            denominator = denominator / g;
        }
    }

public:
    // 构造函数
    Rational() : numerator(0), denominator(1) {}

    Rational(const BigInt& num) : numerator(num), denominator(1) {}

    Rational(const BigInt& num, const BigInt& den) : numerator(num), denominator(den) {
        simplify();
    }

    Rational(int num) : numerator(num), denominator(1) {}

    Rational(int num, int den) : numerator(num), denominator(den) {
        simplify();
    }

    Rational(const std::string& num):Rational() {
        BigInt &up = numerator, &down = denominator;
        BigInt bigint0to10[11];
        for (int i = 0; i < 11; i++) bigint0to10[i] = BigInt(i);
        uint64_t i = 0;
        while (i < num.size() and num[i] != '.' and num[i] != 'e' and num[i] != 'E') {//处理整数部分
            up *= bigint0to10[10];
            up += bigint0to10[num[i] - '0'];
            i++;
        }
        if (i == num.size() or num[i] == 'e' or num[i] == 'E') {
            if (i != num.size() and (num[i] == 'e' or num[i] == 'E')) {//如果为科学计数法
                i++;
                if (num[i] == '-') {
                    i++;
                    down *= bigint0to10[10].power(BigInt(std::string(num.begin() + i, num.end())));
                } else {
                    up *= bigint0to10[10].power(BigInt(std::string(num.begin() + i, num.end())));
                }
            }
            return ;//不是小数，直接跳出
        }
        i++;
        std::vector<short> n;//使用vector容器暂时存储小数位
        while (i < num.size() and num[i] != '.' and num[i] != 'e' and num[i] != 'E') {
            n.emplace_back(num[i] - '0');
            i++;
        }
        if (i == num.size() or ((num[i] == 'e' or num[i] == 'E'))) {//检查是否为循环小数
            //处理不循环小数
            //直接添加到末尾
            for (short& i: n) {
                up *= bigint0to10[10];
                down *= bigint0to10[10];
                up += bigint0to10[i];
            }
        } else {
            //处理循环小数
            //先检测循环部分，再将不循环部分直接添加，再加（循环部分/999...9（循环部分长度个9）/pow（10，len（不循环部分）））
            std::pair<uint64_t, uint64_t> xun = jiance(n);

            //用变量临时记录循环部分
            BigInt xup = BigInt(0), xdown = BigInt(0);
            for (uint64_t i = xun.first; i <= xun.second; i++) {
                xup *= bigint0to10[10];
                xup += bigint0to10[n[i]];
                xdown *= bigint0to10[10];
                xdown += bigint0to10[9];
            }

            //将不循环部分添加
            for (uint64_t i = 0; i < xun.first; i++) {
                up *= bigint0to10[10];
                up += bigint0to10[n[i]];
                down *= bigint0to10[10];
            }

            //通分
            up = up * xdown;
            down = down * xdown;

            //相加
            up = up + xup;
        }

        if (i != num.size() and (num[i] == 'e' or num[i] == 'E')) {//如果为科学计数法
            i++;
            if (num[i] == '-') {
                i++;
                down *= bigint0to10[10].power(BigInt(std::string(num.begin() + i, num.end())));
            } else {
                up *= bigint0to10[10].power(BigInt(std::string(num.begin() + i, num.end())));
            }
        }

        simplify();
    }

private:
    std::pair<uint64_t, uint64_t> jiance(std::vector<short>& n) {
        //检测循环部分，返回循环的开始和结束（左闭右闭）
        uint64_t h1[10] = {};//使用哈希表记录数字出现的次数
        uint64_t h2[10] = {};

        for (short& i: n) h1[i]++;//第一轮遍历记录每个数字的总出现次数（使用迭代器）
        bool flag = true;

        for (uint64_t i = 0; i < n.size(); i++) {//第二轮遍历
            flag = true;
            for (int j = 0; j < 10; j++) {
                if ((h1[j] - h2[j]) & 1) {
                    flag = false;
                    break;
                }
            }
            if (flag) {
                uint64_t i1 = i + 1, i2 = n.size();
                uint64_t ti1, ti2;
                bool ok = true;
                while (!((i2 - i1) & 1) and ok) {
                    //检测是否循环
                    //使用双指针
                    //这段改的比较多不写小注释了
                    //原理为不断将循环范围减半直到不循环为止如果第一轮不循环侧进入第二轮遍历
                    ti1 = i1;
                    ti2 = i1 + ((i2 - i1) >> 1);
                    while (ti2 < i2) {
                        if (n[ti1] != n[ti2]) {
                            ok = false;
                            break;
                        }
                        ti1++;
                        ti2++;
                    }
                    if (ok) {
                        i2 = i1 + ((i2 - i1) >> 1);
                    } else if (i2 == n.size()) {
                        break;
                    } else {
                        return {i1, i2 - 1};
                    }
                }
                if (ok) {//如果保持循环返回
                    return {i1 + 1, i2 - 1};
                }
            }
            h2[n[i]]++;
        }
        return {n.size() - 1, n.size() - 1};
    }

public:

    // 由double构造
    static Rational from_double(double value) {
        if (value == 0.0) {
            return Rational();
        }
        if (std::floor(value) == value) {   // 是整数，分母设为1
            return Rational(BigInt(std::to_string(value)));
        }
        std::ostringstream oss;
        oss << std::scientific << std::setprecision(15) << value;
        std::string str = oss.str();

        // 解析科学记数法
        size_t e_pos = str.find('e');
        std::string base = str.substr(0, e_pos);
        bool negative = false;
        if (base[0] == '-') {
            negative = true;
            base.erase(0, 1);
        }
        base.erase(1, 1);               // 删除小数点
        while (base.back() == '0') {    // 去除末尾0，不可能全为0
            base.pop_back();
        }
        int exponent = std::stoi(str.substr(e_pos + 1));

        size_t decimal_places = std::max(0, static_cast<int>(base.length()) - exponent - 1);    // 有几位小数

        BigInt num(base);
        if (negative) num = num.negate();
        BigInt den("1" + std::string(decimal_places, '0'));

        Rational r(num, den);
        r.simplify();
        return r;
    }

    // 获取分子和分母
    BigInt get_numerator() const { return numerator; }
    BigInt get_denominator() const { return denominator; }

    // 判断是否为整数
    bool is_integer() const {
        return denominator.digits.size() == 1 && denominator.digits[0] == 1;
    }

    // 判断是否为零
    bool is_zero() const {
        return numerator.is_zero();
    }

    // 转换为字符串
    std::string to_string() const {
        if (is_integer()) {
            return numerator.to_string();
        }
        return numerator.to_string() + "/" + denominator.to_string();
    }

    //向下舍入，n为保留几位小数（负数则忽略整数位）
    inline void floor(const BigInt& n) {
        if (n == BigInt(0)) return;
        if (n > BigInt(0)) {
            BigInt pow10n(10);
            pow10n = pow10n.power(n);
            numerator *= pow10n;
            numerator /= denominator;
            denominator = pow10n;
        } else {
            BigInt pow10n(10);
            pow10n = pow10n.power(n.abs());
            denominator *= pow10n;
            numerator /= denominator;
            numerator *= pow10n;
            denominator = BigInt(1);
        }
        simplify();
    }

    //将自己开根，n为保留几位小数（负数则忽略整数位）
    inline void sqrt_self(const BigInt& n) {
        if (numerator < BigInt(0)) throw std::runtime_error("Sqrt negative number");
        Rational t(numerator * denominator);
        
        //求解t的根，二分法
        BigInt ci = BigInt(t.numerator.digits.size() * 2+3) + n;
        const BigInt one(1);
        const BigInt nadd1(n + BigInt(1));
        Rational ans(t / 2);
        for (; !ci.is_zero(); ci -= one) {
            ans = (ans + t / ans) / 2;
            ans.floor(nadd1);
        }
        numerator = ans.numerator;
        denominator *= ans.denominator;
        floor(n);
    }

    inline Rational sqrt(const BigInt& n) {
        Rational re = *this;
        re.sqrt_self(n);
        return re;
    }

    // 转换为 BigInt（如果是整数）
    BigInt to_bigint() const {
        if (!is_integer()) {
            throw std::runtime_error("Cannot convert non-integer fraction to BigInt");
        }
        return numerator;
    }

    // 转换为 double（近似值）
    double to_double() const {
        return static_cast<double>(numerator.to_int()) / static_cast<double>(denominator.to_int());
    }

    // 加法
    Rational operator+(const Rational& other) const {
        BigInt new_num = numerator * other.denominator + other.numerator * denominator;
        BigInt new_den = denominator * other.denominator;
        return Rational(new_num, new_den);
    }

    // 减法
    Rational operator-(const Rational& other) const {
        BigInt new_num = numerator * other.denominator - other.numerator * denominator;
        BigInt new_den = denominator * other.denominator;
        return Rational(new_num, new_den);
    }

    // 乘法
    Rational operator*(const Rational& other) const {
        BigInt new_num = numerator * other.numerator;
        BigInt new_den = denominator * other.denominator;
        return Rational(new_num, new_den);
    }

    // 除法
    Rational operator/(const Rational& other) const {
        if (other.is_zero()) {
            throw std::runtime_error("Division by zero");
        }
        BigInt new_num = numerator * other.denominator;
        BigInt new_den = denominator * other.numerator;
        return Rational(new_num, new_den);
    }

    // 幂运算
    Rational power(const BigInt& exponent) const {
        if (exponent.negative) {
            // 负指数：(a/b)^(-n) = (b/a)^n
            if (is_zero()) {
                throw std::runtime_error("Cannot raise zero to negative power");
            }
            BigInt pos_exp = exponent;
            pos_exp.negative = false;
            return Rational(denominator.power(pos_exp), numerator.power(pos_exp));
        }

        if (exponent.is_zero()) {
            if (is_zero()) {
                throw std::runtime_error("0^0 is undefined");
            }
            return Rational(1);
        }

        return Rational(numerator.power(exponent), denominator.power(exponent));
    }

    // 比较运算符
    bool operator==(const Rational& other) const {
        return numerator * other.denominator == other.numerator * denominator;
    }

    bool operator!=(const Rational& other) const {
        return !(*this == other);
    }

    bool operator<(const Rational& other) const {
        BigInt left = numerator * other.denominator;
        BigInt right = other.numerator * denominator;
        return left < right;
    }

    bool operator<=(const Rational& other) const {
        return *this < other || *this == other;
    }

    bool operator>(const Rational& other) const {
        return !(*this <= other);
    }

    bool operator>=(const Rational& other) const {
        return !(*this < other);
    }

    // 取倒数
    Rational reciprocal() const {
        if (is_zero()) {
            throw std::runtime_error("Cannot take reciprocal of zero");
        }
        return Rational(denominator, numerator);
    }

    // 取绝对值
    Rational abs() const {
        Rational result = *this;
        result.numerator.negative = false;
        return result;
    }

    // 取负值
    Rational operator-() const {
        Rational result = *this;
        result.numerator.negative = !result.numerator.negative;
        return result;
    }
};
