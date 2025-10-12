#pragma once
#define _USE_MATH_DEFINES
#include "symbolic.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_E
#define M_E 2.71828182845904523536
#endif

// 无理数类，支持常见无理数的精确表示
class Irrational {
public:
    // 转为符号表达式

    std::shared_ptr<SymbolicExpr> to_symbolic() const {
        switch (type) {
            case Type::SQRT: {
                // 只要系数不为0，始终输出表达式
                auto sqrtExpr = SymbolicExpr::sqrt(SymbolicExpr::number(static_cast<int>(radicand)));
                if (std::abs(coefficient) < 1e-15) {
                    return SymbolicExpr::number(0);
                } else if (std::abs(coefficient - 1.0) < 1e-15) {
                    return sqrtExpr;
                } else {
                    return SymbolicExpr::multiply(SymbolicExpr::number(::Rational(coefficient)), sqrtExpr);
                }
            }
            case Type::PI:
                // π 的所有情况都返回符号表达式
                if (std::abs(coefficient) < 1e-15) {
                    return SymbolicExpr::variable("π");
                } else if (std::abs(coefficient - 1.0) < 1e-15) {
                    return SymbolicExpr::variable("π");
                } else {
                    return SymbolicExpr::multiply(SymbolicExpr::number(::Rational(coefficient)), SymbolicExpr::variable("π"));
                }
            case Type::E:
                if (std::abs(coefficient) < 1e-15) {
                    return SymbolicExpr::number(0);
                } else if (std::abs(coefficient - 1.0) < 1e-15) {
                    return SymbolicExpr::variable("e");
                } else {
                    return SymbolicExpr::multiply(SymbolicExpr::number(::Rational(coefficient)), SymbolicExpr::variable("e"));
                }
            case Type::LOG:
                // log(n) 仅支持变量形式
                if (std::abs(coefficient) < 1e-15) {
                    return SymbolicExpr::number(0);
                } else {
                    return SymbolicExpr::multiply(SymbolicExpr::number(::Rational(coefficient)), SymbolicExpr::variable("log(" + std::to_string(radicand) + ")"));
                }
            case Type::COMPLEX:
                // 复杂形式暂不支持符号化，返回常数
                return SymbolicExpr::number(::Rational(constant_term));
            default:
                return SymbolicExpr::number(0);
        }
    }
    enum class Type {
        SQRT,  // √n 形式
        PI,    // π 的倍数
        E,     // e 的倍数
        LOG,   // log(n) 形式
        COMPLEX// 复合形式 (a*√b + c*π + d*e + ...)（暂时不适用，用 SymbolicExpr 代替）
    };

private:
    Type type;

    // 对于 √n 形式：coefficient * √radicand
    double coefficient;// 系数
    long long radicand;// 根号内的数

    // 对于复合形式：系数映射
    std::map<std::string, double> coefficients;
    double constant_term;// 常数项

    // 简化根号
    static std::pair<long long, long long> simplify_sqrt(long long n) {
        long long perfect_square = 1;
        long long remainder = n;

        for (long long i = 2; i * i <= n; ++i) {
            while (remainder % (i * i) == 0) {
                perfect_square *= i;
                remainder /= (i * i);
            }
        }
        return {perfect_square, remainder};
    }

public:
    // 构造函数
    Irrational() : type(Type::COMPLEX), coefficient(0), radicand(1), constant_term(0) {}

    // 创建 √n 形式的无理数
    static Irrational sqrt(long long n, double coeff = 1.0) {
        Irrational result;
        result.type = Type::SQRT;

        auto [perfect, remainder] = simplify_sqrt(n);
        result.coefficient = coeff * perfect;
        result.radicand = remainder;
        result.constant_term = 0;

        return result;
    }

    // 创建 π 的倍数
    static Irrational pi(double coeff = 1.0) {
        Irrational result;
        result.type = Type::PI;
        result.coefficient = coeff;
        result.radicand = 1;
        result.constant_term = 0;
        return result;
    }

    // 创建 e 的倍数
    static Irrational e(double coeff = 1.0) {
        Irrational result;
        result.type = Type::E;
        result.coefficient = coeff;
        result.radicand = 1;
        result.constant_term = 0;
        return result;
    }

    // 创建常数（可以退化为有理数）
    static Irrational constant(double value) {
        Irrational result;
        result.type = Type::COMPLEX;
        result.coefficient = 0;
        result.radicand = 1;
        result.constant_term = value;
        return result;
    }

    // 转换为复合形式
    void to_complex() {
        if (type == Type::COMPLEX) return;

        coefficients.clear();
        constant_term = 0;

        switch (type) {
            case Type::SQRT:
                if (radicand == 1) {
                    constant_term = coefficient;
                } else {
                    coefficients["sqrt" + std::to_string(radicand)] = coefficient;
                }
                break;
            case Type::PI:
                coefficients["pi"] = coefficient;
                break;
            case Type::E:
                coefficients["e"] = coefficient;
                break;
            default:
                break;
        }
        type = Type::COMPLEX;
    }

    // 加法
    Irrational operator+(const Irrational& other) const {
        Irrational result = *this;
        Irrational other_copy = other;

        result.to_complex();
        other_copy.to_complex();

        result.constant_term += other_copy.constant_term;

        for (const auto& [key, coeff]: other_copy.coefficients) {
            result.coefficients[key] += coeff;
        }

        return result;
    }

    // 减法
    Irrational operator-(const Irrational& other) const {
        Irrational result = *this;
        Irrational other_copy = other;

        result.to_complex();
        other_copy.to_complex();

        result.constant_term -= other_copy.constant_term;

        for (const auto& [key, coeff]: other_copy.coefficients) {
            result.coefficients[key] -= coeff;
        }

        return result;
    }

    // 标量乘法
    Irrational operator*(double scalar) const {
        Irrational result = *this;

        if (type == Type::COMPLEX) {
            result.constant_term *= scalar;
            for (auto& [key, coeff]: result.coefficients) {
                coeff *= scalar;
            }
        } else {
            result.coefficient *= scalar;
        }

        return result;
    }

    // 乘法（简化版本，主要处理常见情况）
    Irrational operator*(const Irrational& other) const {
        // 如果其中一个是常数
        if (type == Type::COMPLEX && coefficients.empty()) {
            return other * constant_term;
        }
        if (other.type == Type::COMPLEX && other.coefficients.empty()) {
            return *this * other.constant_term;
        }

        // √a * √b = √(ab)
        if (type == Type::SQRT && other.type == Type::SQRT) {
            return Irrational::sqrt(radicand * other.radicand,
                                    coefficient * other.coefficient);
        }

        // 其他情况转为近似值处理
        return Irrational::constant(to_double() * other.to_double());
    }

    // 除法（简化版本）
    Irrational operator/(const Irrational& other) const {
        // 如果除数是常数
        if (other.type == Type::COMPLEX && other.coefficients.empty() && other.constant_term != 0) {
            return *this * (1.0 / other.constant_term);
        }

        // 其他情况转为近似值处理
        double other_val = other.to_double();
        if (std::abs(other_val) < 1e-15) {
            throw std::runtime_error("Irrational: division by zero");
        }
        return Irrational::constant(to_double() / other_val);
    }

    // 负号
    Irrational operator-() const {
        return *this * (-1.0);
    }

    // 比较运算（基于近似值）
    bool operator==(const Irrational& other) const {
        return std::abs(to_double() - other.to_double()) < 1e-12;
    }

    bool operator<(const Irrational& other) const {
        return to_double() < other.to_double();
    }

    bool operator<=(const Irrational& other) const {
        return *this < other || *this == other;
    }

    bool operator>(const Irrational& other) const {
        return other < *this;
    }

    bool operator>=(const Irrational& other) const {
        return *this > other || *this == other;
    }

    // 转换为 double（近似值）
    double to_double() const {
        switch (type) {
            case Type::SQRT:
                if (radicand == 1) {
                    return coefficient;
                }
                return coefficient * std::sqrt(radicand);
            case Type::PI:
                return coefficient * M_PI;
            case Type::E:
                return coefficient * M_E;
            case Type::LOG:
                return coefficient * std::log(radicand);
            case Type::COMPLEX: {
                double result = constant_term;
                for (const auto& [key, coeff]: coefficients) {
                    if (key == "pi") {
                        result += coeff * M_PI;
                    } else if (key == "e") {
                        result += coeff * M_E;
                    } else if (key.substr(0, 4) == "sqrt") {
                        long long n = std::stoll(key.substr(4));
                        result += coeff * std::sqrt(n);
                    }
                }
                return result;
            }
            default:
                return 0.0;
        }
    }

    // 转换为字符串（精确表示）
    std::string to_string() const {
        switch (type) {
            case Type::SQRT:
                if (radicand == 1) {
                    // 处理整数系数的格式化
                    if (coefficient == static_cast<int>(coefficient)) {
                        return std::to_string(static_cast<int>(coefficient));
                    }
                    return std::to_string(coefficient);
                }
                if (coefficient == 1.0) {
                    return "√" + std::to_string(radicand);
                }
                if (coefficient == -1.0) {
                    return "-√" + std::to_string(radicand);
                }
                // 处理系数格式化
                if (std::abs(coefficient - std::round(coefficient)) < 1e-15) {
                    return std::to_string(static_cast<int>(std::round(coefficient))) + "√" + std::to_string(radicand);
                } else {
                    // 格式化小数，去掉末尾的0
                    std::ostringstream oss;
                    oss << std::fixed << std::setprecision(6) << coefficient;
                    std::string temp = oss.str();
                    temp.erase(temp.find_last_not_of('0') + 1);
                    if (temp.back() == '.') temp.pop_back();
                    return temp + "√" + std::to_string(radicand);
                }

            case Type::PI:
                if (coefficient == 1.0) {
                    return "π";
                }
                if (coefficient == -1.0) {
                    return "-π";
                }
                // 处理系数格式化
                if (std::abs(coefficient - std::round(coefficient)) < 1e-15) {
                    return std::to_string(static_cast<int>(std::round(coefficient))) + "π";
                } else {
                    // 格式化小数，去掉末尾的0
                    std::ostringstream oss;
                    oss << std::fixed << std::setprecision(6) << coefficient;
                    std::string temp = oss.str();
                    temp.erase(temp.find_last_not_of('0') + 1);
                    if (temp.back() == '.') temp.pop_back();
                    return temp + "π";
                }

            case Type::E:
                if (coefficient == 1.0) {
                    return "e";
                }
                if (coefficient == -1.0) {
                    return "-e";
                }
                // 处理系数格式化
                if (std::abs(coefficient - std::round(coefficient)) < 1e-15) {
                    return std::to_string(static_cast<int>(std::round(coefficient))) + "e";
                } else {
                    // 格式化小数，去掉末尾的0
                    std::ostringstream oss;
                    oss << std::fixed << std::setprecision(6) << coefficient;
                    std::string temp = oss.str();
                    temp.erase(temp.find_last_not_of('0') + 1);
                    if (temp.back() == '.') temp.pop_back();
                    return temp + "e";
                }

            case Type::LOG:
                if (coefficient == 1.0) {
                    return "log(" + std::to_string(radicand) + ")";
                }
                if (coefficient == -1.0) {
                    return "-log(" + std::to_string(radicand) + ")";
                }
                // 处理系数格式化
                if (std::abs(coefficient - std::round(coefficient)) < 1e-15) {
                    return std::to_string(static_cast<int>(std::round(coefficient))) + "log(" + std::to_string(radicand) + ")";
                } else {
                    // 格式化小数，去掉末尾的0
                    std::ostringstream oss;
                    oss << std::fixed << std::setprecision(6) << coefficient;
                    std::string temp = oss.str();
                    temp.erase(temp.find_last_not_of('0') + 1);
                    if (temp.back() == '.') temp.pop_back();
                    return temp + "log(" + std::to_string(radicand) + ")";
                }

            case Type::COMPLEX: {
                std::string result;
                bool first = true;

                // 常数项
                if (std::abs(constant_term) > 1e-15) {
                    if (std::abs(constant_term - std::round(constant_term)) < 1e-15) {
                        // 是整数或非常接近整数
                        result += std::to_string(static_cast<int>(std::round(constant_term)));
                    } else {
                        // 是真正的小数，格式化为最多6位小数但去掉末尾的0
                        std::ostringstream oss;
                        oss << std::fixed << std::setprecision(6) << constant_term;
                        std::string temp = oss.str();
                        // 去掉末尾的0和小数点
                        temp.erase(temp.find_last_not_of('0') + 1);
                        if (temp.back() == '.') temp.pop_back();
                        result += temp;
                    }
                    first = false;
                }

                // 其他项
                for (const auto& [key, coeff]: coefficients) {
                    if (std::abs(coeff) < 1e-15) continue;

                    if (!first && coeff > 0) result += " + ";
                    else if (!first && coeff < 0)
                        result += " - ";

                    double abs_coeff = std::abs(coeff);
                    std::string term;

                    if (key == "pi") {
                        if (abs_coeff == 1.0) {
                            term = "π";
                        } else if (abs_coeff == static_cast<int>(abs_coeff)) {
                            term = std::to_string(static_cast<int>(abs_coeff)) + "π";
                        } else {
                            term = std::to_string(abs_coeff) + "π";
                        }
                    } else if (key == "e") {
                        if (abs_coeff == 1.0) {
                            term = "e";
                        } else if (abs_coeff == static_cast<int>(abs_coeff)) {
                            term = std::to_string(static_cast<int>(abs_coeff)) + "e";
                        } else {
                            term = std::to_string(abs_coeff) + "e";
                        }
                    } else if (key.substr(0, 4) == "sqrt") {
                        long long n = std::stoll(key.substr(4));
                        if (abs_coeff == 1.0) {
                            term = "√" + std::to_string(n);
                        } else if (abs_coeff == static_cast<int>(abs_coeff)) {
                            term = std::to_string(static_cast<int>(abs_coeff)) + "√" + std::to_string(n);
                        } else {
                            term = std::to_string(abs_coeff) + "√" + std::to_string(n);
                        }
                    }

                    if (first && coeff < 0) result += "-";
                    result += term;
                    first = false;
                }

                return result.empty() ? "0" : result;
            }
            default:
                return "0";
        }
    }

    // 判断是否为零
    bool is_zero() const {
        return std::abs(to_double()) < 1e-15;
    }

    // 判断是否为有理数（即可以精确表示为分数）
    bool is_rational() const {
        if (type == Type::COMPLEX) {
            return coefficients.empty();
        }
        return false;
    }

    // 简化表示（去除系数为0的项）
    void simplify() {
        if (type == Type::COMPLEX) {
            auto it = coefficients.begin();
            while (it != coefficients.end()) {
                if (std::abs(it->second) < 1e-15) {
                    it = coefficients.erase(it);
                } else {
                    ++it;
                }
            }

            // 如果所有无理数项都被删除，只保留常数项
            if (coefficients.empty() && std::abs(constant_term) < 1e-15) {
                constant_term = 0.0;
            }
        }
    }

    // 判断是否为正数
    bool is_positive() const {
        return to_double() > 1e-15;
    }

    // 判断是否为负数
    bool is_negative() const {
        return to_double() < -1e-15;
    }

    // 绝对值
    Irrational abs() const {
        if (is_negative()) {
            return -*this;
        }
        return *this;
    }

    // 幂运算（仅支持整数幂）
    Irrational pow(int exponent) const {
        if (exponent == 0) {
            return Irrational::constant(1.0);
        }
        if (exponent == 1) {
            return *this;
        }
        if (exponent == 2 && type == Type::SQRT) {
            // (a√b)² = a²b
            return Irrational::constant(coefficient * coefficient * radicand);
        }

        // 其他情况使用近似值
        return Irrational::constant(std::pow(to_double(), exponent));
    }

    // 输出流重载
    friend std::ostream& operator<<(std::ostream& os, const Irrational& ir) {
        os << ir.to_string();
        return os;
    }

    // 获取类型
    Type get_type() const { return type; }
};
