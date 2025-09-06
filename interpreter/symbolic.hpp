#pragma once
#include <memory>
#include <string>
#include <vector>
#include <variant>
#include <cmath>
#include "bigint.hpp"
#include "rational.hpp"

// 符号表达式系统
// 支持精确的数学表达式，不进行数值近似

class SymbolicExpr {
public:
    enum class Type {
        Number,      // 数字 (BigInt, Rational, int)
        Sqrt,        // 平方根 √
        Root,        // n次方根 √[n]，未使用
        Power,       // 幂次 ^
        Multiply,    // 乘法 *
        Add,         // 加法 +
        Subtract,    // 减法 -，未使用
        Divide,      // 除法 /，未使用
        Variable     // 变量 (如 π, e)
    };

    Type type;
    
    // 数值存储
    std::variant<int, ::BigInt, ::Rational> number_value;
    
    // 表达式参数
    std::vector<std::shared_ptr<SymbolicExpr>> operands;
    
    // 字符串标识（用于变量名或操作符）
    std::string identifier;

    // 构造函数
    SymbolicExpr(Type t) : type(t) {}
    
    // 数字构造函数
    static std::shared_ptr<SymbolicExpr> number(int n) {
        auto expr = std::make_shared<SymbolicExpr>(Type::Number);
        expr->number_value = n;
        return expr;
    }
    
    static std::shared_ptr<SymbolicExpr> number(const ::BigInt& bi) {
        auto expr = std::make_shared<SymbolicExpr>(Type::Number);
        expr->number_value = bi;
        return expr;
    }
    
    static std::shared_ptr<SymbolicExpr> number(const ::Rational& r) {
        auto expr = std::make_shared<SymbolicExpr>(Type::Number);
        expr->number_value = r;
        return expr;
    }
    
    // 平方根构造函数
    static std::shared_ptr<SymbolicExpr> sqrt(std::shared_ptr<SymbolicExpr> operand) {
        auto expr = std::make_shared<SymbolicExpr>(Type::Sqrt);
        expr->operands.push_back(operand);
        return expr;
    }
    
    // 乘法构造函数
    static std::shared_ptr<SymbolicExpr> multiply(std::shared_ptr<SymbolicExpr> left, std::shared_ptr<SymbolicExpr> right) {
        auto expr = std::make_shared<SymbolicExpr>(Type::Multiply);
        if (right->is_number()) {
            // 将数字移至前端
            expr->operands.push_back(right);
            expr->operands.push_back(left);
            return expr;
        }
        expr->operands.push_back(left);
        expr->operands.push_back(right);
        return expr;
    }
    
    // 加法构造函数
    static std::shared_ptr<SymbolicExpr> add(std::shared_ptr<SymbolicExpr> left, std::shared_ptr<SymbolicExpr> right) {
        auto expr = std::make_shared<SymbolicExpr>(Type::Add);
        expr->operands.push_back(left);
        expr->operands.push_back(right);
        return expr;
    }
    
    // 幂次构造函数
    static std::shared_ptr<SymbolicExpr> power(std::shared_ptr<SymbolicExpr> base, std::shared_ptr<SymbolicExpr> exponent) {

        // 直接符号储存
        auto expr = std::make_shared<SymbolicExpr>(Type::Power);
        expr->operands.push_back(base);
        expr->operands.push_back(exponent);
        return expr;
    }
    
    // 变量构造函数
    static std::shared_ptr<SymbolicExpr> variable(const std::string& name) {
        auto expr = std::make_shared<SymbolicExpr>(Type::Variable);
        expr->identifier = name;
        return expr;
    }
    
    // 化简表达式
    std::shared_ptr<SymbolicExpr> simplify() const;
    
    // 转换为字符串表示
    std::string to_string() const;
    
    // 检查是否为数字
    bool is_number() const { return type == Type::Number; }

    // 检查是否为大整数
    bool is_big_int() const { return is_number() && std::holds_alternative<::BigInt>(number_value);}

    // 检查是否为分数（有理数）
    bool is_rational() const { return is_number() && std::holds_alternative<::Rational>(number_value);}

    // 检查是否为整数
    bool is_int() const { return is_number() && std::holds_alternative<int>(number_value);}
    
    // 获取数字值（如果是数字的话）
    std::variant<int, ::BigInt, ::Rational> get_number() const {
        if (is_number()) {
            return number_value;
        }
        throw std::runtime_error("Expression is not a number");
    }

    int get_int() const {
        if (is_int()) {
            return std::get<int>(get_number());
        }
        throw std::runtime_error("Expression is not a int");
    }
    ::BigInt get_big_int() const {
        if (is_big_int()) {
            return std::get<BigInt>(get_number());
        }
        throw std::runtime_error("Expression is not a BigInt");
    }
    ::Rational get_rational() const {
        if (is_rational()) {
            return std::get<Rational>(get_number());
        }
        throw std::runtime_error("Expression is not a Rational");
    }
    ::Rational convert_rational() const {
		if (!is_number()) {
			throw std::runtime_error("Expression cannot be converted into Rational");
		}
		if (is_rational()) return get_rational();
		else if (is_big_int()) return ::Rational(get_big_int());
		else return ::Rational(get_int());
	}
	
    // 尝试计算数值（如果可能的话）
    double to_double() const;
    
private:
    // 内部化简函数
    std::shared_ptr<SymbolicExpr> simplify_sqrt() const;
    std::shared_ptr<SymbolicExpr> simplify_multiply() const;
    std::shared_ptr<SymbolicExpr> simplify_add() const;
    std::shared_ptr<SymbolicExpr> simplify_power() const;
};
