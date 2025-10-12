#pragma once
#include "bigint.hpp"
#include "rational.hpp"
#include <memory>
#include <string>
#include <variant>
#include <cmath>
#include <limits>
#include <algorithm>
#include <map>
#include <functional>
#include <iostream>

#define _SYMBOLIC_DEBUG 0

#if _SYMBOLIC_DEBUG
#define err_stream std::cerr
#else
#define err_stream if (0) std::cerr
/*
class _NullBuffer : public std::streambuf {
	public:
		virtual int overflow(int c) {
			return c;
		}
};
_NullBuffer nbf;
std::ostream nullstream(&nbf);
#define err_stream nullstream
*/
#endif

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
        Infinity,      // 无限大（number_value 中的 int 决定是正或负）
        Variable     // 变量 (如 π, e)

    };
	
	/*
	哈希要保证的一些因素：
	- 乘法交换律（乘法：奇数二进制位参与运算）
	- 加法交换律（加法：偶数二进制位参与运算）
	- 加法和乘法区分
	这个函数在加法化简时使用。
	*/
	struct HashData {
#define _HASH_PARAMS		ODDBIT, EVENBIT, SQRBIT, HALFBIT
		using HashType = unsigned long long;
		// TODO: 允许多个 HashData 对象之间的不同以减少哈希冲突概率
		// 下方变量的名字不重要。
#define ODDBIT_D 0x555555555555555ull
#define EVENBIT_D 0xAAAAAAAAAAAAAAAull
#define SQRBIT_D 0xBDEEBD77BDEEBD7ull
#define HALFBIT_D 0x969969669699696ull
#define EMPTY 0ull
#define INFINITY_D 0xFFF7FFFFDEADBEEFull
#define PI_H 0x1451419810C0000ull
#define E_H 0x9198101145C0000ull
#define UNKNOWN_H 0xAD0AA0BEEFC0000ull
		
		HashType ODDBIT;
		HashType EVENBIT;
		HashType SQRBIT;
		HashType HALFBIT;
		
		::Rational k = ::Rational(1), ksqrt = ::Rational(1);
		HashType hash = EMPTY;
		std::shared_ptr<SymbolicExpr> hash_obj = SymbolicExpr::number(1);	// 因为是乘法，1为默认状态。此处存储被 hash 的项目对应的值
		
		static HashType bigint_hash(const BigInt& rt) {
			// 直接哈希所有 digits
			HashType weight = 1ull, ans = 0ull;
			for (auto &i : rt.digits) {
				ans = ans * weight + (i + 3ull);
				weight *= 17ull;			// 不用 10 减少哈希冲突
			}
			if (rt.negative) return ~ans;
			return ans;
		}
		
		static HashType rational_hash(const Rational& rt) {
			return bigint_hash(rt.get_numerator()) ^ bigint_hash(rt.get_denominator());
		}
		
		HashType to_single_hash() {
			return (rational_hash(k) & HALFBIT) ^ (rational_hash(ksqrt) & SQRBIT) ^ hash;
		}
		
		// TODO: 考虑优化
		std::shared_ptr<SymbolicExpr> get_combined_k() {
			return SymbolicExpr::multiply(SymbolicExpr::number(k), SymbolicExpr::sqrt(SymbolicExpr::number(ksqrt)))->simplify();
		}
		
		HashData() {
			
		}
		
		HashData(std::shared_ptr<SymbolicExpr> obj, 
			HashType ODDBIT = ODDBIT_D, HashType EVENBIT = EVENBIT_D, HashType SQRBIT = SQRBIT_D, HashType HALFBIT = HALFBIT_D)
			: ODDBIT(ODDBIT), EVENBIT(EVENBIT), SQRBIT(SQRBIT), HALFBIT(HALFBIT) {
			// Evaluate hash
			HashData ld, rd;
			HashType prehash = 0, rterm = 0;
			switch (obj->type) {
				case Type::Number:
					this->k = obj->convert_rational();
					err_stream << "[HPP Debug] Return as value " << k.to_string() << "\n";
					break;
				case Type::Infinity:
					this->hash = INFINITY_D;
					break;
				
				case Type::Sqrt:
					ld = HashData(obj->operands[0], _HASH_PARAMS);
					// sqrt 里面还有 sqrt，取值异或哈希
					this->ksqrt = ld.k;
					ld.k = ::Rational(0);
					this->hash = ld.to_single_hash() * SQRBIT;	// 表明这是个 sqrt，里面没东西则恰好为 0
					this->hash_obj = SymbolicExpr::sqrt(ld.hash_obj);
					break;
				case Type::Multiply:
					ld = HashData(obj->operands[0], _HASH_PARAMS);
					rd = HashData(obj->operands[1], _HASH_PARAMS);
					this->k = ld.k * rd.k;
					this->ksqrt = ld.ksqrt * rd.ksqrt;
					this->hash = (obj->operands[0]->is_number() ? 1 : ld.hash) * (obj->operands[1]->is_number() ? 1 : rd.hash);
					err_stream << "[HPP Debug] LDHash: " << ld.hash_obj->to_string() << ", RDHash: " << rd.hash_obj->to_string() << std::endl;
					err_stream << "[HPP Debug] My hash value is " << this->hash << std::endl;
					err_stream << "[HPP Debug] L applied: " << (obj->operands[0]->is_number() ? 1 : ld.hash) <<
						", R applied: " << (obj->operands[1]->is_number() ? 1 : rd.hash) << std::endl;
					if (!(ld.hash | rd.hash)) this->hash = 0;	// 里面没有东西
					this->hash_obj = SymbolicExpr::multiply(ld.hash_obj, rd.hash_obj)->simplify();
					break;
				case Type::Add:
					ld = HashData(obj->operands[0], _HASH_PARAMS);
					rd = HashData(obj->operands[1], _HASH_PARAMS);
					this->hash = ld.to_single_hash() + rd.to_single_hash();
					this->hash_obj = obj;	// 没有做任何处理
					break;
				case Type::Power:
					// TODO: 此处引入类似根式化简的机制，暂时直接 hash（可能有问题）
					ld = HashData(obj->operands[0], _HASH_PARAMS);
					rd = HashData(obj->operands[1], _HASH_PARAMS);
					// 不是特别恰当，但可以先这样
					// 保证 1，2，-1 等常见数值
					rterm = rd.to_single_hash() - 1;
					this->hash = ld.to_single_hash() ^ rterm ^ (rterm << 8) ^ (rterm << 16) ^ (rterm << 32);
					this->hash_obj = obj;	// 没有做任何处理
					break;
				case Type::Variable:
					if (obj->identifier == "π" || obj->identifier == "pi") this->hash = PI_H;
					else if (obj->identifier == "e") this->hash = E_H;
					else this->hash = UNKNOWN_H;
					this->hash_obj = obj;	// 没有做任何处理
					break;
				default:
					// 如果某个 hash 不能用就调过来
					defs: this->hash = EMPTY;
					for (auto &i : obj->operands) {
						if (obj->type == Type::Add) {
							this->hash += HashData(i).to_single_hash();	// 令其自然溢出，同时避免异或消除
						} else {
							this->hash *= HashData(i).to_single_hash() + 1;	// 令其自然溢出，同时避免异或消除
						}
					}
					this->hash ^= prehash;
					this->hash_obj = obj;
			}
			// TODO: 可能考虑在这里做根式化简
		}
#undef ODDBIT_D
#undef EVENBIT_D
#undef SQRBIT_D
#undef HALFBIT_D
#undef EMPTY
#undef INFINITY_D
#undef PI_H
#undef E_H
#undef UNKNOWN_H		
	};

    Type type;

    // 数值存储
    std::variant<int, ::BigInt, ::Rational> number_value;

    // 表达式参数
    std::vector<std::shared_ptr<SymbolicExpr>> operands;

    // 字符串标识（用于变量名或操作符）
    std::string identifier;

	// 是否已经化简完成
	bool already_simplified = false;

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

	static std::shared_ptr<SymbolicExpr> infinity(int k = 1) {
		auto expr = std::make_shared<SymbolicExpr>(Type::Infinity);
		expr->number_value = k;
		return expr;
	}

    // 平方根构造函数
    static std::shared_ptr<SymbolicExpr> sqrt(std::shared_ptr<SymbolicExpr> operands) {
        auto expr = std::make_shared<SymbolicExpr>(Type::Sqrt);
        expr->operands.push_back(operands);
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
    bool is_big_int() const { return is_number() && std::holds_alternative<::BigInt>(number_value); }

    // 检查是否为分数（有理数）
    bool is_rational() const { return is_number() && std::holds_alternative<::Rational>(number_value); }

    // 检查是否为整数
    bool is_int() const { return is_number() && std::holds_alternative<int>(number_value); }

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
