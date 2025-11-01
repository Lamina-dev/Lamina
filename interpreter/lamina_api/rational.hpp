#pragma once
#include "bigint.hpp"
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <map>
#include <stdint.h>

class Rational {
private:
    BigInt numerator;
    BigInt denominator;

    // 求最大公约数
    static BigInt gcd(const BigInt& a, const BigInt& b) {
        BigInt abs_a = a.Abs();
        BigInt abs_b = b.Abs();

        while (abs_b) {
            BigInt temp = abs_b;
            abs_b = abs_a % abs_b;
            abs_a = temp;
        }
        return abs_a;
    }

    // 化简分数
    void simplify() {
        if (!denominator) {
            throw std::runtime_error("Denominator cannot be zero");
        }

        // 确保分母为正数
        if (denominator.IsNegative()) {
            numerator = -numerator;
            denominator = denominator.Abs();
        }

        // 化简
        BigInt g = gcd(numerator, denominator);
        if (g != 0 and g != 1) {
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
        if (num.empty() or num == "0") return;
        BigInt &up = numerator, &down = denominator;
        BigInt BigInt0to10[11];
        for (int i = 0; i < 11; i++) BigInt0to10[i] = BigInt(i);
        uint64_t i = 0;
        if (num[i] == '-') i++;
        while (i < num.size() and num[i] != '.' and num[i] != 'e' and num[i] != 'E') {//处理整数部分
            up *= BigInt0to10[10];
            up += BigInt0to10[num[i] - '0'];
            i++;
        }
        if (i == num.size() or num[i] == 'e' or num[i] == 'E') {
            if (i != num.size() and (num[i] == 'e' or num[i] == 'E')) {//如果为科学计数法
                i++;
                if (num[i] == '-') {
                    i++;
                    down *= BigInt0to10[10].power(BigInt(std::string(num.begin() + i, num.end())));
                } else {
                    up *= BigInt0to10[10].power(BigInt(std::string(num.begin() + i, num.end())));
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
                up *= BigInt0to10[10];
                down *= BigInt0to10[10];
                up += BigInt0to10[i];
            }
        } else {
            //处理循环小数
            //先检测循环部分，再将不循环部分直接添加，再加（循环部分/999...9（循环部分长度个9）/pow（10，len（不循环部分）））
            std::pair<uint64_t, uint64_t> xun = jiance(n);

            //用变量临时记录循环部分
            BigInt xup = BigInt(0), xdown = BigInt(0);
            for (uint64_t i = xun.first; i <= xun.second; i++) {
                xup *= BigInt0to10[10];
                xup += BigInt0to10[n[i]];
                xdown *= BigInt0to10[10];
                xdown += BigInt0to10[9];
            }

            //将不循环部分添加
            for (uint64_t i = 0; i < xun.first; i++) {
                up *= BigInt0to10[10];
                up += BigInt0to10[n[i]];
                down *= BigInt0to10[10];
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
                down *= BigInt0to10[10].power(BigInt(std::string(num.begin() + i, num.end())));
            } else {
                up *= BigInt0to10[10].power(BigInt(std::string(num.begin() + i, num.end())));
            }
        }

        if (num[0] == '-') up = BigInt(0) - up;

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
                    return {i1 - 1, i2 - 1};
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
        if (negative) num = -num;
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
        return denominator == BigInt(1);
    }

    // 判断是否为零
    bool is_zero() const {
        return !numerator;
    }

    // 转换为字符串
    std::string to_string() const {
        if (is_integer()) {
            return numerator.ToString();
        }
        return numerator.ToString() + "/" + denominator.ToString();
    }

    //转换成小数字符串，如果为循环小数循环节用括号圈出并在末尾添加...
    inline std::string to_float_string() {
        if (numerator % denominator == BigInt(0)) return (numerator / denominator).ToString();
        std::string re;
        const BigInt ten(10);
        const BigInt zero(0);
        if ((numerator < zero) ^ (denominator < zero)) re += '-';
        BigInt num = numerator.Abs(), den = denominator.Abs();
        re += (num / den).ToString();//先记录整数部分
        num %= den;
        re += '.';

        std::map<BigInt, size_t> num_index;

        while (num != zero and num_index.find(num) == num_index.end()) {
            num_index[num] = re.size();
            re += '0';
            num *= ten;
            while (num >= den) {
                num -= den;
                re.back()++;
            }
        }
        if (num_index.find(num) != num_index.end()) {//有循环节
            re.insert(re.begin() + num_index[num], '(');
            re += ")...";
        }
        return re;

    }

    //转换成小数字符串，n为保留几位小数（负数则忽略整数位）
    inline std::string to_float_string(int64_t n) const{
        if (n == 0) return (numerator / denominator).ToString();
        if (n > 0) {
            BigInt pow10n(10);
            pow10n = pow10n.power(BigInt(n));
            std::string re = (numerator * pow10n / denominator).ToString();
            if (n == re.size()) re.insert(re.end() - n, '0');//如果没有个位添加为0
            re.insert(re.end() - n,'.');
            while (re.back() == '0') re.pop_back();//去除末尾的0
            if (re.back() == '.') re.pop_back();//如果为小数点也排除
            return re;
        } else {
            BigInt pow10n(10);
            pow10n = pow10n.power(BigInt(n).Abs());
            std::string re = (numerator / (denominator * pow10n)).ToString();
            if (re.size() == 1 and re[0] == '0') return re;
            for (; n < 0; n++) re.push_back('0');
            return re;
        }
    }

    //向下舍入，n为保留几位小数（负数则忽略整数位）
    inline void floor(const int64_t& n) {
        floor_without_sim(n);
        simplify();
    }

private:

    inline void floor_without_sim(const int64_t& n) {
        if (n == 0) return;
        if (n > 0) {
            BigInt pow10n(10);
            pow10n = pow10n.power(BigInt(n));
            numerator *= pow10n;
            numerator /= denominator;
            denominator = pow10n;
        } else {
            BigInt pow10n(10);
            pow10n = pow10n.power(BigInt(n).Abs());
            denominator *= pow10n;
            numerator /= denominator;
            numerator *= pow10n;
            denominator = BigInt(1);
        }
    }

public:

    //将自己开根，n为保留几位小数（负数则忽略整数位）
    inline void sqrt_self(int64_t n) {
        if (numerator < BigInt(0)) throw std::runtime_error("Sqrt negative number");
        Rational t(numerator * denominator);
        
        //求解t的根，二分法
        const int64_t nadd1 = n + 1;
        const BigInt ten(10);
        Rational ans( (t * t + Rational(6) * t + Rational(1))  /  (Rational(4) * (t + Rational(1))) );//使用牛顿法寻找快速寻找近似值
        Rational temp;
        while (temp.numerator / ten != ans.numerator / ten) {
            temp = (ans + t / ans) / 2;
            temp.floor_without_sim(nadd1);
            ans = (temp + t / temp) / 2;
            ans.floor_without_sim(nadd1);
        }
        numerator = ans.numerator;
        denominator *= ans.denominator;
        floor(n);
    }

    inline Rational sqrt(int64_t n) const{
        Rational re = *this;
        re.sqrt_self(n);
        return re;
    }

    //将自身开radical次根，并保留n位小数
    inline void radicand_self(const BigInt& radical, int64_t n) {
        if (numerator < BigInt(0) and (radical % BigInt(2) == BigInt(0))) throw std::runtime_error("Radicand negative number");
        Rational t(numerator * denominator.power(radical - BigInt(1)));

        const int64_t nadd1 = n + 1;
        const Rational rat_radical(radical);
        const BigInt ten(10);
        Rational ans(Rational(t.numerator,radical) * (Rational(radical - BigInt(1)) + Rational(t.numerator, t.numerator.power(radical))));//使用牛顿法寻找快速寻找近似值
        //推到逼近公式：ans = ans/radical * ( (radical-1) + (t / ans^radical) )
        Rational temp;
        while (temp.numerator / ten != ans.numerator / ten) {
            temp = ans / rat_radical * ((rat_radical - Rational(1)) + (t / ans.power(radical)));
            temp.floor_without_sim(nadd1);
            ans = temp / rat_radical * ((rat_radical - Rational(1)) + (t / temp.power(radical)));
            ans.floor_without_sim(nadd1);
        }

        numerator = ans.numerator;
        denominator *= ans.denominator;
        floor(n);
    }

    inline Rational radicand(const BigInt& radical, int64_t n) {
        Rational re = *this;
        re.radicand_self(radical, n);
        return re;
    }


    // 转换为 BigInt（如果是整数）
    BigInt to_BigInt() const {
        if (!is_integer()) {
            throw std::runtime_error("Cannot convert non-integer fraction to BigInt");
        }
        return numerator;
    }

    // 转换为 double（近似值）
    double to_double() const {
        return std::stod(to_float_string(15));
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

    // 整数幂运算
    Rational power(const BigInt& exponent) const {
        if (exponent < BigInt(0)) {
            // 负指数：(a/b)^(-n) = (b/a)^n
            if (is_zero()) {
                throw std::runtime_error("Cannot raise zero to negative power");
            }
            BigInt pos_exp = -exponent;
            return Rational(denominator.power(pos_exp), numerator.power(pos_exp));
        }

        if (!exponent) {
            if (is_zero()) {
                throw std::runtime_error("0^0 is undefined");
            }
            return Rational(1);
        }

        return Rational(numerator.power(exponent), denominator.power(exponent));
    }

    // 有理数幂运算（开自己）
    inline void power_self(const Rational& exponent,size_t n) {
        *this = power(exponent.numerator);
        radicand_self(exponent.denominator,n);
    }

    // 有理数幂运算
    inline Rational power(const Rational& exponent, size_t n) {
        Rational re = *this;
        re.power_self(exponent, n);
        return re;
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
        result.numerator = result.numerator.Abs();
        return result;
    }

    // 取负值
    Rational operator-() const {
        Rational result = *this;
        result.numerator = -result.numerator;
        return result;
    }
};
