#include "symbolic.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <functional>
#include <iostream>

// 符号表达式的化简实现
std::shared_ptr<SymbolicExpr> SymbolicExpr::simplify() const {
	// !! TODO: !! 添加“化简”标记，避免 simplify 重复调用导致效率降低
    switch (type) {
        case Type::Number:
        case Type::Variable:
            return std::make_shared<SymbolicExpr>(*this);
            
        case Type::Sqrt:
            return simplify_sqrt();
            
        case Type::Multiply:
            return simplify_multiply();
            
        case Type::Add:
            return simplify_add();

        case Type::Power:
            return simplify_power();
        
        default:
            return std::make_shared<SymbolicExpr>(*this);
    }
}

std::shared_ptr<SymbolicExpr> SymbolicExpr::simplify_sqrt() const {
    if (operands.empty()) return std::make_shared<SymbolicExpr>(*this);
    
    auto simplified_operand = operands[0]->simplify();
	if (simplified_operand->type == SymbolicExpr::Type::Infinity) return simplified_operand;
    
    if (simplified_operand->is_number()) {
		
		auto scvrs = simplified_operand->convert_rational();
		
		if (simplified_operand->is_rational() && scvrs.get_denominator() == ::BigInt(1)) {
			
			// TODO: Debug output:
			std::cerr << "[Debug output] x/1 simplifier\n";
			
			::BigInt actual = scvrs.get_numerator();
			simplified_operand->number_value = actual;
		}
		
		// pair 格式：first 为系数，second 为根式下的值
		// 注意自行判断 second 为 1 的情况。
		auto num_process = [](int n) -> std::pair<int, int> {
			if (n < 0) throw std::runtime_error("Square root of negative number");
			if (n == 0 || n == 1) return std::make_pair(n, 1);
			int sqrt_n = static_cast<int>(std::sqrt(n));
            if (sqrt_n * sqrt_n == n) return std::make_pair(sqrt_n, 1);
            int factor = 1, remaining = n;
            for (int i = 2; i * i <= remaining; ++i) {
                while (remaining % (i * i) == 0) {
                    factor *= i;
                    remaining /= (i * i);
                }
            }
            return std::make_pair(factor, remaining);
		};
		
		auto in_simplify_range = [](const ::BigInt& bi) -> bool {
			return bi <= BigInt(INT_MAX) && bi >= BigInt(INT_MIN);
		};
		
		auto generate_component = [](const ::Rational& rat) -> std::shared_ptr<SymbolicExpr> {
			if (rat.get_denominator() == ::BigInt(1)) return SymbolicExpr::number(rat.get_numerator());
			else return SymbolicExpr::number(rat);
		};
		
        auto num_val = simplified_operand->get_number();
        if (std::holds_alternative<int>(num_val)) {
			// TODO: Debug output:
			//std::cerr << "[Debug output] numeric simplifier\n";
            int n = std::get<int>(num_val);
			auto res = num_process(n);
			if (res.second == 1) return SymbolicExpr::number(res.first);
			else if (res.first == 1) return SymbolicExpr::sqrt(SymbolicExpr::number(res.second));
			else return SymbolicExpr::multiply(SymbolicExpr::number(res.first), SymbolicExpr::sqrt(SymbolicExpr::number(res.second)));
        }
        if (std::holds_alternative<::BigInt>(num_val)) {
            const auto& bi = std::get<::BigInt>(num_val);
            if (bi.negative) throw std::runtime_error("Square root of negative number");
            if (bi.is_zero() || bi == BigInt(1)) return SymbolicExpr::number(bi);
			//std::cerr << "[Debug output] bigint simplifier init\n";
            if (bi.is_perfect_square()) return SymbolicExpr::number(bi.sqrt());
			// TODO: Debug output:
			//std::cerr << "[Debug output] bigint simplifier\n";
            // TODO: 这个等BigInt效率高点再说，或者换个算法
            /*BigInt factor(1), remaining(bi);
            for (BigInt i(2); i * i <= remaining; i = i + BigInt(1)) {
                while ((remaining % (i * i)).is_zero()) {
                    factor = factor * i;
                    remaining = remaining / (i * i);
                }
            }
            if (factor > BigInt(1)) {
                if (remaining == BigInt(1)) return SymbolicExpr::number(factor);
                else return SymbolicExpr::multiply(SymbolicExpr::number(factor), SymbolicExpr::sqrt(SymbolicExpr::number(remaining)));
            }*/
            // 暂时只判断可以转成int的
            if (in_simplify_range(bi)) {
                return SymbolicExpr::sqrt(SymbolicExpr::number(bi.to_int()))->simplify();
            }
        }
		// 分数化简中底数和指数分别判断
		if (std::holds_alternative<::Rational>(num_val)) {
			const auto &nobj = std::get<::Rational>(num_val);
			const auto &nume = nobj.get_numerator();
			const auto &deme = nobj.get_denominator();
			// 暂时只判断可以转成int的
			if (in_simplify_range(nume) && in_simplify_range(deme)) {
				auto numsimp = num_process(nume.to_int());
				auto demsimp = num_process(deme.to_int());
				::Rational numarea = ::Rational(numsimp.first, demsimp.first);
				::Rational sqarea = ::Rational(numsimp.second, demsimp.second);
				
				// TODO: Debug output:
				std::cerr << "[Debug output] numa = " << numarea.to_string() << "; sqa = " << sqarea.to_string() << std::endl;
				
				if (sqarea == ::Rational(1)) return SymbolicExpr::number(numarea);
				else if (numarea == ::Rational(1)) return SymbolicExpr::sqrt(SymbolicExpr::number(sqarea));
				return SymbolicExpr::multiply(generate_component(numarea), SymbolicExpr::sqrt(generate_component(sqarea)));
			}
		}
    }
	std::cerr << "[Debug output] end numeric sqrt simplifier\n";
    // sqrt(x*x) 或 sqrt(π*π) 直接返回 x 或 π
    if (simplified_operand->type == SymbolicExpr::Type::Multiply && simplified_operand->operands.size() == 2) {
        const auto& a = simplified_operand->operands[0];
        const auto& b = simplified_operand->operands[1];
        // sqrt(x*x) = x
        if (a->type == SymbolicExpr::Type::Variable && b->type == SymbolicExpr::Type::Variable && a->identifier == b->identifier) {
            return a;
        }
        // sqrt(π*π) = π
        if (a->type == SymbolicExpr::Type::Variable && b->type == SymbolicExpr::Type::Variable &&
            ((a->identifier == "π" && b->identifier == "π") || (a->identifier == "pi" && b->identifier == "pi"))) {
            return a;
        }
        if (a->to_string() == b->to_string()) {
            return a;
        }
        auto get_var = [](const std::shared_ptr<SymbolicExpr>& expr) -> std::string {
            if (expr->type == SymbolicExpr::Type::Variable) return expr->identifier;
            if (expr->type == SymbolicExpr::Type::Multiply && expr->operands.size() == 2) {
                if (expr->operands[1]->type == SymbolicExpr::Type::Variable) return expr->operands[1]->identifier;
            }
            return "";
        };
        std::string var_a = get_var(a);
        std::string var_b = get_var(b);
        if (!var_a.empty() && var_a == var_b) {
            // sqrt((c*π)*(d*π)) = sqrt((c*d)*π^2) = π*sqrt(c*d)
            auto pow2 = SymbolicExpr::power(SymbolicExpr::variable(var_a), SymbolicExpr::number(2));
            auto left_coeff = (a->type == SymbolicExpr::Type::Multiply && a->operands[0]->is_number()) ? a->operands[0] : SymbolicExpr::number(1);
            auto right_coeff = (b->type == SymbolicExpr::Type::Multiply && b->operands[0]->is_number()) ? b->operands[0] : SymbolicExpr::number(1);
            auto coeff_mul = SymbolicExpr::multiply(left_coeff, right_coeff)->simplify();
            // sqrt((c*π)*(d*π)) = π*sqrt(c*d)
            if (coeff_mul->is_number()) {
                auto sqrt_coeff = SymbolicExpr::sqrt(coeff_mul)->simplify();
                if (sqrt_coeff->is_number() && sqrt_coeff->to_string() == "1") {
                    return SymbolicExpr::variable(var_a);
                } else {
                    return SymbolicExpr::multiply(sqrt_coeff, SymbolicExpr::variable(var_a));
                }
            } else {
                // fallback: sqrt(π^2)
                return SymbolicExpr::sqrt(pow2)->simplify();
            }
        }
    }
    // sqrt(x^2) 或 sqrt(π^2) 直接返回 x 或 π
    if (simplified_operand->type == SymbolicExpr::Type::Power && simplified_operand->operands.size() == 2) {
		std::cerr << "[Debug output] power simplifier\n";
        const auto& base = simplified_operand->operands[0];
        const auto& exp = simplified_operand->operands[1];
        if (exp->is_number()) {
            auto exp_val = exp->get_number();
            int n = 0;
            BigInt big_n;// 0 for default
            if (std::holds_alternative<int>(exp_val)) n = std::get<int>(exp_val);
            else if (std::holds_alternative<::Rational>(exp_val)) {
                ::Rational r = std::get<::Rational>(exp_val);
 
                if (r.is_integer()) big_n = r.get_numerator();
            }
            if (n == 2 || big_n.to_string() == "2") {
                // sqrt(x^2) = x
                if (base->type == SymbolicExpr::Type::Variable && (base->identifier == "π" || base->identifier == "pi")) {
                    return base;
                }
                return base;
            }
        }
    }
    return SymbolicExpr::sqrt(simplified_operand);
}

std::shared_ptr<SymbolicExpr> SymbolicExpr::simplify_multiply() const {
    if (operands.size() != 2) return std::make_shared<SymbolicExpr>(*this);
    
    auto left = operands[0]->simplify();
    auto right = operands[1]->simplify();
	
	// TODO: 理论上为未定式
	if (left->type == SymbolicExpr::Type::Number && left->convert_rational() == ::Rational(0)) return left;
	if (right->type == SymbolicExpr::Type::Number && right->convert_rational() == ::Rational(0)) return right;
	
	if (left->type == SymbolicExpr::Type::Infinity) return left;
	if (right->type == SymbolicExpr::Type::Infinity) return right;
	
	// TODO: Debug output:
	std::cerr << "[Debug output] Init: Processing l:" << left->to_string() << ", r:" << right->to_string() << std::endl;
	
	// TODO: 1 或 -1 乘以某个内容，直接返回另一边
	// TODO: 加快运算速度，数字和 Multiply 相乘时，直接处理 Multiply 内的数字，不进入指数环节
	auto has_no_multiply_effect = [](const std::shared_ptr<SymbolicExpr>& obj) -> bool {
		return (obj->is_number() && obj->convert_rational() == ::Rational(1));
	};
	if (has_no_multiply_effect(left)) {
		std::cerr << "[Debug output] left has no effect\n";
		return right;
	}
	if (has_no_multiply_effect(right)) {
		std::cerr << "[Debug output] right has no effect\n";
		return left;
	}
	
    // 如果两个操作数都是数字，直接相乘
	// TODO: 有些计算时会出问题
    if (left->is_number() && right->is_number()) {
		std::cerr << "[Debug output] numeric calling in multiplier: ";
        auto left_num = left->get_number();
        auto right_num = right->get_number();
        
        if (std::holds_alternative<int>(left_num) && std::holds_alternative<int>(right_num)) {
            int result = std::get<int>(left_num) * std::get<int>(right_num);
			std::cerr << result << std::endl;
            return SymbolicExpr::number(result);
        } else {
			::Rational result = left->convert_rational() * right->convert_rational();
			std::cerr << result.to_string() << std::endl;
			return SymbolicExpr::number(result);
		}
    }
	
	// 加法运算特殊化简
	if ((left->type == SymbolicExpr::Type::Add) || (right->type == SymbolicExpr::Type::Add)) {
		
		auto res = SymbolicExpr::number(0);
		// TODO: *** 此处可能导致 res 被反复化简
		
		if ((left->type == SymbolicExpr::Type::Add) && (right->type == SymbolicExpr::Type::Add)) {
			for (auto &i : left->operands) {
				for (auto &j : right->operands) {
					auto adt = SymbolicExpr::multiply(i, j)->simplify();
					std::cerr << "[Debug output] [1] --- Adding term: " << adt->to_string() << std::endl;
					res = SymbolicExpr::add(res, adt);
				}
			}
		} else {
			if (left->type != SymbolicExpr::Type::Add)
				std::swap(left, right);
			
			for (auto &i : left->operands) {
				auto adt = SymbolicExpr::multiply(i, right)->simplify();
				std::cerr << "[Debug output] [2] --- Adding term: " << adt->to_string() << std::endl;
				res = SymbolicExpr::add(res, adt);
			}
		}

		std::cerr << "[Debug output] === Begin adder simplifier ===\n";
		return res->simplify();
		
	}
	
	// 注意，multiply 不属于这类类型，需要手动化简
	auto is_power_compatible = [](const std::shared_ptr<SymbolicExpr>& expr) -> bool {
		return expr->type == SymbolicExpr::Type::Number || expr->type == SymbolicExpr::Type::Sqrt
			|| expr->type == SymbolicExpr::Type::Power || expr->type == SymbolicExpr::Type::Variable;
	};
	
	// 注意，除法使用指数
	std::function<std::shared_ptr<SymbolicExpr>(const std::shared_ptr<SymbolicExpr>&)> power_compatible;
	power_compatible = [&](const std::shared_ptr<SymbolicExpr>& expr) -> std::shared_ptr<SymbolicExpr> {
		std::shared_ptr<SymbolicExpr> ret;
		if (expr->type == SymbolicExpr::Type::Number || expr->type == SymbolicExpr::Type::Variable) {
			return SymbolicExpr::power(expr, SymbolicExpr::number(1));
		} else if (expr->type == SymbolicExpr::Type::Sqrt) {
			ret = SymbolicExpr::power(power_compatible(expr->operands[0])->simplify(), SymbolicExpr::number(::Rational(1, 2)));
		} else if (expr->type == SymbolicExpr::Type::Power) {
			ret = SymbolicExpr::power(power_compatible(expr->operands[0]), power_compatible(expr->operands[1]))->simplify();
		} else {
			return expr;
		}
		if (ret->type == SymbolicExpr::Type::Number || ret->type == SymbolicExpr::Type::Variable)
			ret = SymbolicExpr::power(ret, SymbolicExpr::number(1));
		if (ret->type == SymbolicExpr::Type::Sqrt)
			ret = SymbolicExpr::power(ret->operands[0], SymbolicExpr::number(::Rational(1, 2)));
		return ret;
	};
	
	auto is_compounded_sqrt = [](const std::shared_ptr<SymbolicExpr>& expr) -> bool {
		if (expr->type == SymbolicExpr::Type::Multiply) {
			if (expr->operands.size() >= 2 && (expr->operands[0]->type == SymbolicExpr::Type::Number)
				&& (expr->operands[1]->type == SymbolicExpr::Type::Sqrt)) return true;
		}
		return false;
	};
	
	auto is_for_auxiliary = [is_compounded_sqrt](const std::shared_ptr<SymbolicExpr> &obj) -> bool {
		// 判断是否可用于 sqrt_and_auxiliary，如果不可用，直接尝试 flatten multiply
		return (obj->type == SymbolicExpr::Type::Number) || (obj->type == SymbolicExpr::Type::Sqrt)
			|| is_compounded_sqrt(obj);
	};
	
	// 考虑：和指数类型乘法合并/简化
	auto sqrt_and_auxiliary = [is_power_compatible, power_compatible, is_compounded_sqrt, is_for_auxiliary]
		(const std::shared_ptr<SymbolicExpr> &obj, bool no_simplify = false) -> std::shared_ptr<SymbolicExpr> {
		
		// TODO: Debug output:
		std::cerr << "[Debug output] Starting sqrt-and-aux process" << std::endl;
		
		// 此处可以考虑优化
		auto left = no_simplify ? obj->operands[0] : obj->operands[0]->simplify();
		auto right = no_simplify ? obj->operands[1] : obj->operands[1]->simplify();
		
		// TODO: Debug output:
		std::cerr << "[Debug output] Processing l:" << left->to_string() << ", r:" << right->to_string() << std::endl;
		
		if (is_for_auxiliary(left) && is_for_auxiliary(right)) {
			// 根式相乘的一般处理
			if (left->type == SymbolicExpr::Type::Sqrt || is_compounded_sqrt(left)
				|| right->type == SymbolicExpr::Type::Sqrt || is_compounded_sqrt(right)) {
				
				if ((!is_compounded_sqrt(left)) && (!is_compounded_sqrt(right))) {
					
					if (right->type == SymbolicExpr::Type::Sqrt)
						std::swap(left, right);
					
					// TODO: Debug output:
					std::cerr << "[Debug output] [4] Starting sqrt-and-aux process" << std::endl;
					
					bool negative = false;
					if (left->type == SymbolicExpr::Type::Number && left->convert_rational() < ::Rational(0, 1))
						negative = !negative;
					if (right->type == SymbolicExpr::Type::Number && right->convert_rational() < ::Rational(0, 1))
						negative = !negative;
					
					std::shared_ptr<SymbolicExpr> sresult = SymbolicExpr::multiply(
						(left->type == SymbolicExpr::Type::Number) ? (SymbolicExpr::multiply(left, left))
																	: left->operands[0],
						(right->type == SymbolicExpr::Type::Number) ? (SymbolicExpr::multiply(right, right))
																	: right->operands[0]);
					
					auto res = SymbolicExpr::sqrt(sresult)->simplify();
					
					if (negative) {
						// 终端化到最简，否则难以处理
						if (res->type == SymbolicExpr::Type::Multiply)
							return SymbolicExpr::multiply(SymbolicExpr::multiply(SymbolicExpr::number(-1), res->operands[0])->simplify(), res->operands[1]);
						else
							return SymbolicExpr::multiply(SymbolicExpr::number(-1), res);
						
					} else {
						return res;
					}
					
				} else {
					
					// 把右侧加入到左侧
					if (!is_compounded_sqrt(left))
						std::swap(left, right);
					
					// TODO: Debug output:
					std::cerr << "[Debug output] [5] Starting sqrt-and-aux process" << std::endl;
					
					bool negative = false;
					
					auto res = std::make_shared<SymbolicExpr>(*left);
					auto simplify_res = [&]() {
						if (is_compounded_sqrt(res->operands[1])) {
							res->operands[0] = SymbolicExpr::multiply(res->operands[0], res->operands[1]->operands[0])->simplify();
							res->operands[1] = res->operands[1]->operands[1];		// 不用再 sqrt
						}
						res->simplify();
					};
					
					if (right->type == SymbolicExpr::Type::Number) {
						std::cerr << "[Debug output] [5a] res: " << res->to_string() << std::endl;
						res->operands[0] = SymbolicExpr::multiply(res->operands[0], right)->simplify();
					} else if (right->type == SymbolicExpr::Type::Sqrt) {
						std::cerr << "[Debug output] [5b] res: " << res->to_string() << std::endl;
						res->operands[1] = SymbolicExpr::multiply(res->operands[1], right)->simplify();
						simplify_res();
					} else if (is_compounded_sqrt(right)) {
						std::cerr << "[Debug output] [5c] res: " << res->to_string() << std::endl;
						res->operands[0] = SymbolicExpr::multiply(res->operands[0], right->operands[0])->simplify();
						res->operands[1] = SymbolicExpr::multiply(res->operands[1], right->operands[1])->simplify();
						simplify_res();
					}
					
					if (negative) {
						std::cerr << "[Debug output] [5] Negative processor. res: " << res->to_string() << std::endl;
						if (res->type == SymbolicExpr::Type::Multiply) {
							// 这个需要化简
							res->operands[0] = SymbolicExpr::multiply(SymbolicExpr::number(-1), res->operands[0])->simplify();
						} else {
							// 不要再化简
							res = SymbolicExpr::multiply(SymbolicExpr::number(-1), res);
						}
					}
					//std::cerr << "[Debug output] [5] Finalize. res: " << res->to_string() << std::endl;
					return res;
					
				}
				
			}
		}
		
		// 回退到保留乘法
		return SymbolicExpr::multiply(left, right);
	};

	
	// 指数类型相乘的处理（可能考虑合并相同指数项）
	if ((left->type == SymbolicExpr::Type::Power || right->type == SymbolicExpr::Type::Power
		|| (left->type == right->type && left->type == SymbolicExpr::Type::Variable))
		|| (!is_for_auxiliary(left)) || (!is_for_auxiliary(right))) {
		
		if (is_power_compatible(left) && is_power_compatible(right)) {
			if (left->type != SymbolicExpr::Type::Power)
				std::swap(left, right);
			
			auto lcom = left;
			if (left->type != SymbolicExpr::Type::Power)
				lcom = power_compatible(left);
			
			auto rcom = power_compatible(right);
			
			// TODO: 提升为成员函数，判断相等
			std::function<bool(std::shared_ptr<SymbolicExpr>,std::shared_ptr<SymbolicExpr>)> is_power_equiv;
			is_power_equiv = [&](std::shared_ptr<SymbolicExpr> a, std::shared_ptr<SymbolicExpr> b) -> bool {
				if (a->type != b->type) 
					return false;
				if (a->type == SymbolicExpr::Type::Number)
					return a->convert_rational() == b->convert_rational();
				else if (a->type == SymbolicExpr::Type::Sqrt)
					return is_power_equiv(a->operands[0], b->operands[0]);
				else if (a->type == SymbolicExpr::Type::Variable)
					return a->identifier == b->identifier;
				else
					return false;
			};
			// TODO: Debug output:
			std::cerr << "[Debug output] [1] preparing to merge exponents" << std::endl;
			
			// TODO: 这边可能还需要测试
			if (lcom->operands[1]->is_number() && rcom->operands[1]->is_number()) {
				
				auto lcr = lcom->operands[1]->convert_rational();
				auto rcr = rcom->operands[1]->convert_rational();
				
				auto ldr = lcr.get_denominator();
				auto rdr = rcr.get_denominator();
				
				if (is_power_equiv(lcom->operands[0], rcom->operands[0])) {
					// TODO: Debug output:
					std::cerr << "[Debug output] [1] Merging bases" << std::endl;
					return SymbolicExpr::power(lcom->operands[0], SymbolicExpr::add(lcom->operands[1], rcom->operands[1]))->simplify();
				}
				
				if (ldr == rdr) {
					if (lcr == rcr) {
						// TODO: Debug output:
						std::cerr << "[Debug output] [1a] Merging exponents in a simplified way" << std::endl;
						
						if (lcom->operands[0]->type == SymbolicExpr::Type::Variable || rcom->operands[0]->type == SymbolicExpr::Type::Variable) {
							std::cerr << "[Debug output] [1a] Entering special reservation\n";
							return SymbolicExpr::power(SymbolicExpr::multiply(lcom->operands[0]->simplify(), rcom->operands[0]->simplify()), SymbolicExpr::number(lcr)); // 不要化简整体！
						} else {
							std::cerr << "[Debug output] [1a] No variable, normalizing\n";
							auto tmp = SymbolicExpr::power(SymbolicExpr::multiply(lcom->operands[0], rcom->operands[0]),
								SymbolicExpr::number(lcr));
							return tmp->simplify();
						}
					} else if (ldr == ::BigInt(1)) {
						// 没有分母（特殊处理，避免死循环）；此处，底数、指数均不能合并。
						// 此处不应再对整体进行 multiply 的化简调用。
						// 留给以后实现。
						// TODO: Debug output:
						std::cerr << "[Debug output] [1b] Give up merging" << std::endl;
					} else {
						// 注意这里的 multiply 不化简（power 的处理要跟上）
						// TODO: Debug output:
						std::cerr << "[Debug output] [1c] Merging exponents" << std::endl;
						auto new_base = SymbolicExpr::multiply(
								SymbolicExpr::power(lcom->operands[0], SymbolicExpr::number(lcr.get_numerator())),
								SymbolicExpr::power(rcom->operands[0], SymbolicExpr::number(rcr.get_numerator())));
						// 进行简化处理，同时避免死循环
						// 执行外部的 simplify 时内部的是不必要的
						new_base = new_base->simplify(); // new_base = sqrt_and_auxiliary(new_base, true);
						return SymbolicExpr::power(new_base, SymbolicExpr::number(::Rational(::BigInt(1), lcr.get_denominator())))->simplify();
					}
				}
			}
			
			// TODO: Debug output:
			std::cerr << "[Debug output] End of power-compatible process" << std::endl;
			
		} else {
			
			// 开始尝试 flatten
			std::function<bool(const std::shared_ptr<SymbolicExpr>&, std::shared_ptr<SymbolicExpr>)> flatten_multiply;
			// 暂时只支持有理指数化简
			// 键为底数的值，值为指数的值
			// TODO: 这样可能需要额外判断双层根号问题，以及分母有理化问题
			std::map<::Rational, ::Rational> base_ref;
			std::map<::Rational, std::shared_ptr<SymbolicExpr>> exponent_ref;
			std::vector<std::shared_ptr<SymbolicExpr>> result;
			flatten_multiply = [&](const std::shared_ptr<SymbolicExpr>& expr, std::shared_ptr<SymbolicExpr> pre_timing) -> bool {
				if (expr->type == SymbolicExpr::Type::Multiply) {
					for (auto &i : expr->operands) {
						if (!flatten_multiply(i, pre_timing)) return false;
					}
					return true;
				} else if (is_power_compatible(expr)) {
					auto current = power_compatible(expr);
					if (!has_no_multiply_effect(pre_timing))	// 略微加快速度
						current->operands[1] = SymbolicExpr::multiply(current->operands[1], pre_timing)->simplify();
					if (current->operands[0]->type == SymbolicExpr::Type::Multiply) {
						for (auto &i : current->operands[0]->operands) {
							if (!flatten_multiply(i, current->operands[1])) return false;
						}
					} else {
						// TODO: Debug output:
						std::cerr << "Converting " << expr->to_string() << " to " << current->to_string() << std::endl;
						result.push_back(current);
						return true;
					}
				}
				return false;
			};
			// 这样传递可能有性能问题
			// TODO: Debug output:
			std::cerr << "[Debug output] [2] Begin flat operation" << std::endl;
			bool able = flatten_multiply(std::make_shared<SymbolicExpr>(*this), SymbolicExpr::number(1));
			// TODO: Debug output:
			std::cerr << "[Debug output] [2] End flat operation with " << able << std::endl;
			if (able) {
				// 尝试合并指数
				bool exponent_merger = true, base_merger = true;
				std::cerr << "[Debug output] [2] Preloading" << std::endl;
				for (auto &cvt : result) {
					if (cvt->is_number()) continue;
					if (cvt->operands.size() < 2) {
						std::cerr << "[Debug output] [2] WARNING: Format error at " << cvt->to_string() << std::endl;
						continue;
					}
					if (!cvt->operands[0]->is_number()) {
						// TODO: Debug output:
						std::cerr << "[Debug output] [2] Flat: exponent fails at " << cvt->operands[0]->to_string() << std::endl;
						std::cerr << "[Debug output] [2] Flat: (of " << cvt->to_string() << ")\n";
						exponent_merger = false;
					}
					if (!cvt->operands[1]->is_number()) {
						// TODO: Debug output:
						std::cerr << "[Debug output] [2] Flat: fails at " << cvt->operands[1]->to_string() << std::endl;
						std::cerr << "[Debug output] [2] Flat: (of " << cvt->to_string() << ")\n";
						base_merger = false;
						exponent_merger = false;
					}
				}
				
				int exponent_merger_cnt = 0, base_merger_cnt = 0;
				
				if (exponent_merger) {
					std::cerr << "[Debug output] [2m] Ready for exp merger" << std::endl;
					// 同底数合并
					for (auto &cvt : result) {
						// 已经检查过了
						::Rational base = cvt->operands[0]->convert_rational();
						::Rational exponent = cvt->operands[1]->convert_rational();
						auto finder = base_ref.find(base);
						if (finder != base_ref.end()) {
							finder->second = finder->second + exponent;
							exponent_merger_cnt++;
						} else {
							base_ref[base] = exponent;
						}
					}
				}
				
				if (base_merger) {
					std::cerr << "[Debug output] [2n] Ready for base merger" << std::endl;
					// 同指数合并
					for (auto &cvt : result) {
						auto cvt_rational = cvt->operands[1]->convert_rational();
						auto finder = exponent_ref.find(cvt_rational);
						if (finder != exponent_ref.end()) {
							finder->second = SymbolicExpr::multiply(finder->second, cvt->operands[0])->simplify();
							base_merger_cnt++;
						} else {
							exponent_ref[cvt_rational] = cvt->operands[0];
						}
					}
				}
				
				if (exponent_merger && (exponent_merger_cnt >= base_merger_cnt)) {
					// 由于只允许两项乘法，此处需要递归构造
					// 为了后续处理方便，现在构造成 数值*后续表达式 的形式，如果要提升性能可以改二叉树
					auto res = SymbolicExpr::number(1);
					bool inits = true;
					for (auto &i : base_ref) {
						if (i.first == ::Rational(0)) return SymbolicExpr::number(0);
						if (i.second == ::Rational(0)) continue;
						if (i.first == ::Rational(1)) continue;
						// TODO: Debug output:
						std::cerr << "[Debug output] [2] exponent referring (" << i.first.to_string() << ")^(" << i.second.to_string() << ")\n";
						auto cres = SymbolicExpr::power(SymbolicExpr::number(i.first), SymbolicExpr::number(i.second))->simplify();
						if (inits) {
							res = cres;
							inits = false;
						}
						else res = SymbolicExpr::multiply(cres, res);
					}
					return res;
				} else if (base_merger && (base_merger_cnt >= exponent_merger_cnt)) {
					// 先这么复制下来，万一以后逻辑不同
					auto res = SymbolicExpr::number(1);
					bool inits = true;
					
					// TODO: Debug output:
					std::cerr << "[Debug output] [2] base merging\n";
					
					for (auto &i : exponent_ref) {
						// 不要化简
						
						// TODO: Debug output:
						std::cerr << "[Debug output] [2] base merging: {" << i.second->to_string() << "}^(" << i.first.to_string() << ")\n";
						
						if (i.first == ::Rational(0)) continue;
						std::shared_ptr<SymbolicExpr> cres;
						if (i.first == ::Rational(1)) cres = i.second;
						else cres = SymbolicExpr::power(i.second, SymbolicExpr::number(i.first))->simplify();
						if (inits) {
							res = cres;
							inits = false;
						}
						else res = SymbolicExpr::multiply(cres, res);
					}
					return res;
				}
				
				
			}
			// 否则无法化简，保留原表达式
		}
		
		// 到此处：未能化简，（以后这里可能引入欧拉公式等等）
		// TODO: 分母有理化
		
	}

	return sqrt_and_auxiliary(std::make_shared<SymbolicExpr>(*this), true);
}

std::shared_ptr<SymbolicExpr> SymbolicExpr::simplify_add() const {
    if (operands.size() != 2) return std::make_shared<SymbolicExpr>(*this);

    auto left = operands[0]->simplify();
    auto right = operands[1]->simplify();
	
	if (left->type == SymbolicExpr::Type::Infinity) return left;
	if (right->type == SymbolicExpr::Type::Infinity) return right;

    // 解析根号
	// 如果为根号，其中 coeff 为根式的系数，radicand 为根号下的值
	std::function<bool(const std::shared_ptr<SymbolicExpr>&,::Rational&,::Rational&)> extract_sqrt;
    extract_sqrt = [&extract_sqrt](const std::shared_ptr<SymbolicExpr>& expr, ::Rational& coeff, ::Rational& radicand) -> bool {
		if (expr->type == SymbolicExpr::Type::Number) {
			coeff = expr->convert_rational();
			radicand = ::Rational(1);
			return true;
		}
        if (expr->type == SymbolicExpr::Type::Sqrt && expr->operands.size() == 1 && expr->operands[0]->is_number()) {
            coeff = ::Rational(1);
            radicand = expr->operands[0]->convert_rational();
            return true;
        }
        if (expr->type == SymbolicExpr::Type::Multiply && expr->operands.size() == 2) {
			// 先特殊判断两项的情况
            if (expr->operands[0]->is_number() && expr->operands[1]->type == SymbolicExpr::Type::Sqrt && expr->operands[1]->operands.size() == 1 && expr->operands[1]->operands[0]->is_number()) {
                coeff = expr->operands[0]->convert_rational();
				//std::holds_alternative<::Rational>(expr->operands[0]->get_number()) ? std::get<::Rational>(expr->operands[0]->get_number()) : ::Rational(std::get<int>(expr->operands[0]->get_number()));
				
                radicand = expr->operands[1]->operands[0]->convert_rational();
                return true;
            }
			::Rational coeff1, radicand1, coeff2, radicand2;
			if (extract_sqrt(expr->operands[0], coeff1, radicand1) && extract_sqrt(expr->operands[1], coeff2, radicand2)) {
				coeff = coeff1 * coeff2;
				radicand = radicand1 * radicand2;
				return true;
			}
        }
        return false;
    };

    // TODO: 解析变量
    /*auto extract_variable = [](const std::shared_ptr<SymbolicExpr>& expr, ::Rational& coeff) -> bool {
        // type应该是变量或者乘法
        if (expr->type == SymbolicExpr::Type::Variable && expr->operands.size() == 1) {
            return true;
        }
    };*/

    // 解析数字
    auto extract_number = [](const std::shared_ptr<SymbolicExpr> expr, Rational& n) -> bool {
        if (expr->type != SymbolicExpr::Type::Number) return false;
		n = n + expr->convert_rational();
        return true;
    };

    std::vector<std::shared_ptr<SymbolicExpr>> terms;
    std::function<void(const std::shared_ptr<SymbolicExpr>&)> flatten_add;
	// 展开所有加法项
    flatten_add = [&](const std::shared_ptr<SymbolicExpr>& expr) {
        if (expr->type == SymbolicExpr::Type::Add && expr->operands.size() == 2) {
            flatten_add(expr->operands[0]);
            flatten_add(expr->operands[1]);
        } else {
            terms.push_back(expr);
        }
    };
    flatten_add(left);
    flatten_add(right);
    std::map<::Rational, ::Rational> sqrt_terms;// radicand -> sum of coeffs
    // std::map<std::string, ::Rational> variable_terms;// TODO
    Rational number_term(0);// 数字部分
    std::vector<std::shared_ptr<SymbolicExpr>> others;// other things
	
	std::cerr << "[Debug output] adder: end flatten add\n";
	
    for (const auto& term : terms) {
        ::Rational coeff;
        ::Rational radicand;
        if (extract_sqrt(term, coeff, radicand)) {
            sqrt_terms[radicand] = sqrt_terms[radicand] + coeff;
			std::cerr << "[Debug output] adder: got sqrt term " << coeff.to_string() << " at sqrt:" << radicand.to_string() << std::endl;
        } else if (extract_number(term, number_term)) {
            // Do nothing yet
        } else {
			std::cerr << "[Debug output] adder: undealt item " << term->to_string() << std::endl;
            others.push_back(term);
        }
    }
	
	std::cerr << "[Debug output] adder: number term = " << number_term.to_string() << std::endl;
	
    std::vector<std::shared_ptr<SymbolicExpr>> result_terms;
    for (const auto& [radicand, coeff] : sqrt_terms) {
        if (coeff.is_integer() && coeff.get_numerator().to_int() == 0) continue;
		
		std::cerr << "[Debug output] adder: sqrt term coeff:" << coeff.to_string() << "; radicand:" << radicand.to_string() << std::endl;
		
        if (radicand == ::Rational(1)) {
			result_terms.push_back(SymbolicExpr::number(coeff));
		} else if (coeff == ::Rational(1)) {
            result_terms.push_back(SymbolicExpr::sqrt(SymbolicExpr::number(radicand)));
        } else {
            result_terms.push_back(SymbolicExpr::multiply(SymbolicExpr::number(coeff), SymbolicExpr::sqrt(SymbolicExpr::number(radicand))));
        }
    }
    result_terms.insert(result_terms.end(), others.begin(), others.end());
    if (number_term != 0) {// 非0时才添加数字
        result_terms.push_back(SymbolicExpr::number(number_term));
    }
    if (result_terms.empty()) return SymbolicExpr::number(0);
    if (result_terms.size() == 1) return result_terms[0];

    std::shared_ptr<SymbolicExpr> sum = result_terms[0];
    for (size_t i = 1; i < result_terms.size(); ++i) {
        sum = SymbolicExpr::add(sum, result_terms[i]);
    }
    return sum;
}

std::shared_ptr<SymbolicExpr> SymbolicExpr::simplify_power() const {
    // 如果底数是数字，且指数是整数，用数字储存
    auto base = operands[0]->simplify();
    auto exponent = operands[1]->simplify();
	
	// TODO: 理论上有未定式
	if (exponent->type == SymbolicExpr::Type::Number && exponent->convert_rational() == ::Rational(0)) return SymbolicExpr::number(1);
	if (base->type == SymbolicExpr::Type::Number && base->convert_rational() == ::Rational(0)) return SymbolicExpr::number(0);
	
	if (base->type == SymbolicExpr::Type::Infinity) return base;
	if (exponent->type == SymbolicExpr::Type::Infinity) return exponent;
	
	// TODO: Debug output:
	std::cerr << "[Debug output] Simplifying power: base = " << base->to_string() << "; exponent = " << exponent->to_string()
		<< std::endl;
		
	if (exponent->is_number() && exponent->convert_rational() == ::Rational(1)) return base;
	
    if (base->is_number() && (exponent->is_int() || exponent->is_big_int())) {
		auto banum = base->convert_rational();
		auto exnum = exponent->convert_rational();
		
		if (exnum < ::Rational(0)) {
			return SymbolicExpr::power(SymbolicExpr::number(banum.reciprocal()), SymbolicExpr::number(::Rational(0) - exnum))->simplify();
		}
		
        auto expr = std::make_shared<SymbolicExpr>(Type::Number);
        // 底数是分数，结果为分数
        if (base->is_rational() || exponent->is_rational()) {
            if (exponent->is_int()) {
				expr->number_value = (base->get_rational()).power(BigInt(exponent->get_int()));
            } else if (exponent->is_big_int()) {
				expr->number_value = (base->get_rational()).power(exponent->get_big_int());
            }
        } else {// 否则结果为大整数
            BigInt b;
            if (base->is_int()) {
                b = BigInt(std::get<int>(base->get_number()));
            } else {
                b = std::get<BigInt>(base->get_number());
            }
            BigInt e;
            if (exponent->is_int()) {
                e = BigInt(std::get<int>(exponent->get_number()));
            } else {
                e = std::get<BigInt>(exponent->get_number());
            }
            if (e.to_int() >= 0) {
                expr->number_value = b.power(e);
            } else {
                
                expr->number_value = Rational(BigInt(1), b.power(e.negate()));
            }
        }

        return expr;
    } else if (base->is_number() && exponent->is_rational()) {
        // 底数是数字，指数是分数
        // 用符号储存
		// TODO:某些必要的化简如 8^(1/3)
		auto bsr = base->convert_rational();
		auto expr = exponent->convert_rational();
		
		auto in_range = [](const ::Rational& val) -> bool {
			auto vn = val.get_numerator(), vd = val.get_denominator();
			const ::BigInt lower = ::BigInt(INT_MIN), upper = ::BigInt(INT_MAX);
			return (vn >= lower && vn <= upper) && (vd >= lower && vd <= upper);
		};
		
		if (in_range(bsr) && in_range(expr)) {
			// TODO: Debug output:
			std::cerr << "[Debug output] Power simplifying (rational ^ rational) expressions" << std::endl;
			
			if (expr == ::Rational(1)) return SymbolicExpr::number(bsr);
			
			int bs_n = bsr.get_numerator().to_int(), bs_d = bsr.get_denominator().to_int();
			int es_n = expr.get_numerator().to_int(), es_d = expr.get_denominator().to_int();
			// TODO: Debug output:
			std::cerr << "[Debug output] bs = " << bs_n << "/" << bs_d << "; es = " << es_n << "/" << es_d << std::endl;
			
			// 如果成功，返回 true，origin 为修改后的值，保证 origin 不增大
			// 如果失败，返回 false，origin 不做修改
			// 注意，要保证既约分数（gcd(num, denom) = 1）
			auto simplify_inner = [](int& origin, const int& denom) -> bool {
				// 可以优化
				int answer = 1, target = origin;
				for (int i = 2; 1ll * i * i <= target; i++) {
					int exphere = 0;
					while (target % i == 0) {
						exphere++;
						target /= i;
					}
					if (exphere && (exphere % denom == 0)) {
						int contb = exphere / denom;
						for (int j = 0; j < contb; j++) answer *= i;
					} else return false;
				}
				if (target != 1) {
					if (denom != 1) return false;
					answer *= target;
				}
				// TODO: Debug output:
				std::cerr << "[Debug output] Denom = " << denom << ", Simplifying " << target << " to " << answer << std::endl;
				origin = answer;
				return true;
			};
			
			if (simplify_inner(bs_n, es_d) && simplify_inner(bs_d, es_d)) {
				// 化简成功
				// TODO: Debug output:
				std::cerr << "[Debug output] Post-operation bs = " << bs_n << "/" << bs_d << "; es = " << es_n << "/" << es_d << std::endl;
				std::cerr << "[Debug output] Power simplifying (rational ^ rational) - success" << std::endl;
				auto current_new_base = SymbolicExpr::number((::Rational(bs_n, bs_d)).power(::BigInt(es_n)));
				if (es_d == 1) return current_new_base;
				return SymbolicExpr::power(current_new_base, SymbolicExpr::number(::Rational(::BigInt(1), ::BigInt(es_d))));
			}
			// 否则化简失败，注意 bs_n 和 bs_d 可能需要重新获取
			
		}
		
		// 避免修改，重新获取
		auto rconv = exponent->convert_rational();
		
		if (rconv.get_denominator() == ::BigInt(2) && rconv.get_numerator() >= ::BigInt(-3) 
			&& rconv.get_numerator() <= ::BigInt(3)) {
			std::cerr << "[Debug output] call of sqrt simplifier\n";
			return SymbolicExpr::sqrt(SymbolicExpr::power(base, SymbolicExpr::number(rconv.get_numerator())))->simplify();
		}
    } else if (base->type == SymbolicExpr::Type::Power || base->type == SymbolicExpr::Type::Sqrt) {
		if (base->type == SymbolicExpr::Type::Sqrt) {
			base = SymbolicExpr::power(base->operands[0], SymbolicExpr::number(::Rational(1, 2)));
		}
		// TODO: Debug output:
		std::cerr << "[Debug output] Power simplifying embedded power / sqrt" << std::endl;
		auto pwr = SymbolicExpr::multiply(base->operands[1], exponent)->simplify();
		if (pwr->type == SymbolicExpr::Type::Number && pwr->convert_rational() == ::Rational(1))
			return base->operands[0]->simplify();
		return SymbolicExpr::power(base->operands[0]->simplify(), pwr);
	}
	
	if (exponent->is_int() || exponent->is_big_int()) {
		auto rconv = exponent->convert_rational();
		if (rconv == ::Rational(0)) return SymbolicExpr::number(1);
		if (rconv == ::Rational(1)) return std::make_shared<SymbolicExpr>(*base);
		if (rconv == ::Rational(-1)) {
			// 这里一定不是整数，尝试分母有理化
			
			std::function<bool(const std::shared_ptr<SymbolicExpr> &)> processable;
			processable = [&processable](const std::shared_ptr<SymbolicExpr> &obj) -> bool {
				return obj->type == SymbolicExpr::Type::Number || obj->type == SymbolicExpr::Type::Sqrt
					|| (obj->type == SymbolicExpr::Type::Multiply && obj->operands.size() == 2
						&& processable(obj->operands[0]) && processable(obj->operands[1]));
			};
			
			std::cerr << "[Debug output] begin rationalizing attempt\n";
			
			if (base->type == SymbolicExpr::Type::Add && base->operands.size() == 2 &&
				processable(base->operands[0]) && processable(base->operands[1])) {
				auto new_term = SymbolicExpr::multiply(SymbolicExpr::number(-1), base->operands[1])->simplify();
				// TODO: Debug output:
				std::cerr << "[Debug output] term processor ended\n";
				auto new_nume = SymbolicExpr::add(base->operands[0], new_term);
				std::cerr << "[Debug output] nume processor ended\n";
				auto new_denom = SymbolicExpr::multiply(base, new_nume)->simplify();
				std::cerr << "[Debug output] denom processor ended\n";
				if (new_denom->type == SymbolicExpr::Type::Number) {
					std::cerr << "[Debug output] term = " << new_term->to_string() << "; denom = " << new_denom->to_string() << std::endl;
					return SymbolicExpr::multiply(SymbolicExpr::number(new_denom->convert_rational().reciprocal()), 
							new_nume)->simplify();
				} else {
					// 有理化失败
					// TODO: Debug output
					std::cerr << "[Debug output] Pow: rationalize failed!\n";
				}
				
			}
		}
		// 防止死循环
		if (rconv.get_denominator() == ::BigInt(1) && rconv.get_numerator() >= ::BigInt(-3) && rconv.get_numerator() < ::BigInt(-1)) {
			// 转为倒数的情况
			return SymbolicExpr::power(SymbolicExpr::power(base, SymbolicExpr::number(rconv.get_numerator())), SymbolicExpr::number(-1))->simplify();
		}
		if (rconv.get_denominator() == ::BigInt(1) && rconv.get_numerator() > ::BigInt(1) && rconv.get_numerator() <= ::BigInt(4)) {
			int exps = rconv.get_numerator().to_int();
			std::shared_ptr<SymbolicExpr> result = std::make_shared<SymbolicExpr>(*base);
			for (int i = 2; i <= exps; i++)
				result = SymbolicExpr::multiply(result, base)->simplify();
			return result;
		}
	}

    return SymbolicExpr::power(base, exponent);
}


std::string SymbolicExpr::to_string() const {
	
	auto get_output = [](std::shared_ptr<const SymbolicExpr> expr) -> std::string {
		const std::string lbrace = std::string("("), rbrace = std::string(")");
		if ((expr->type == SymbolicExpr::Type::Number && (expr->convert_rational().get_denominator() == ::BigInt(1))) || expr->type == SymbolicExpr::Type::Variable
			|| expr->type == SymbolicExpr::Type::Sqrt) return expr->to_string();
		else return lbrace + expr->to_string() + rbrace;
	};
	
    switch (type) {
        case Type::Number:
            if (std::holds_alternative<int>(number_value)) {
                return std::to_string(std::get<int>(number_value));
            } else if (std::holds_alternative<::BigInt>(number_value)) {
                return std::get<::BigInt>(number_value).to_string();
            } else if (std::holds_alternative<::Rational>(number_value)) {
                return std::get<::Rational>(number_value).to_string();
            }
            return "0";
            
        case Type::Variable:
            return identifier;
			
		case Type::Infinity:
			if (std::get<int>(number_value) > 0) return "inf";
			else return "-inf";
            
        case Type::Sqrt:
            if (operands.empty()) return "√()";
            return "√" + get_output(operands[0]);
            
        case Type::Multiply:
            if (operands.size() < 2) return "*(?)";

            if (operands[1]->type == Type::Variable && operands[1]->identifier == "e") {
                // 当右操作数是 e 时，给左操作数加括号以明确优先级
                if (operands[0]->type == Type::Add) {
                    return "(" + operands[0]->to_string() + ")*" + operands[1]->to_string();
                }
            }
            
            if (operands[0]->is_number() && operands[1]->type == Type::Sqrt) {
                return operands[0]->to_string() + operands[1]->to_string();
            }
            return get_output(operands[0]) + "*" + get_output(operands[1]);
            
        case Type::Add: {
            if (operands.size() < 2) return "+(?)";
			
            std::vector<std::shared_ptr<SymbolicExpr>> terms;
            std::function<void(const std::shared_ptr<SymbolicExpr>&)> flatten_add;
            flatten_add = [&](const std::shared_ptr<SymbolicExpr>& expr) {
                if (expr->type == Type::Add && expr->operands.size() == 2) {
                    flatten_add(expr->operands[0]);
                    flatten_add(expr->operands[1]);
                } else {
                    terms.push_back(expr);
                }
            };
            flatten_add(std::make_shared<SymbolicExpr>(*this));

            std::vector<std::string> result_terms;
			result_terms.reserve(terms.size());
            for (auto &term : terms) {
                result_terms.push_back(get_output(term));
            }

            if (result_terms.empty()) return "0";
            std::string res = result_terms[0];
            for (size_t i = 1; i < result_terms.size(); ++i) {
                res += "+" + result_terms[i];
            }
            return res;
        }
            
        case Type::Power:
            if (operands.size() < 2) return "^(?)";
            return get_output(operands[0]) + "^" + get_output(operands[1]);
            
        default:
            return "Unknown";
    }
}

double SymbolicExpr::to_double() const {
    switch (type) {
        case Type::Number:
            if (std::holds_alternative<int>(number_value)) {
                return static_cast<double>(std::get<int>(number_value));
            } else if (std::holds_alternative<::BigInt>(number_value)) {
                return std::get<::BigInt>(number_value).to_double();
            } else if (std::holds_alternative<::Rational>(number_value)) {
                return std::get<::Rational>(number_value).to_double();
            }
            return 0.0;

        case Type::Variable:
            if (identifier == "π" || identifier == "pi") {
                #ifdef M_PI
                return M_PI;
                #else
                return 3.14159265358979323846;
                #endif
            }
            // 其他变量仍抛异常
            throw std::runtime_error("Symbolic variable cannot be converted to double");

		case Type::Infinity:
			if (std::get<int>(number_value) > 0) return std::numeric_limits<double>::infinity();
			else return -std::numeric_limits<double>::infinity();

        case Type::Sqrt:
            if (!operands.empty()) {
                return std::sqrt(operands[0]->to_double());
            }
            return 0.0;

        case Type::Multiply:
            if (operands.size() >= 2) {
                return operands[0]->to_double() * operands[1]->to_double();
            }
            return 0.0;
			
		case Type::Add:
			if (operands.size() >= 2) {
                return operands[0]->to_double() + operands[1]->to_double();
            }
            return 0.0;
			
		case Type::Power:
			return std::pow(operands[0]->to_double(), operands[1]->to_double());

        default:
            return 0.0;
    }
}