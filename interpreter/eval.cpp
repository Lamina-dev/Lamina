#include "interpreter.hpp"
#include "lamina.hpp"
Value Interpreter::eval_LiteralExpr(const LiteralExpr* node) {
    if (node->type == Value::Type::Int) {
        // Check if it contains scientific notation (e or E) or decimal point
        const std::string value = node->value;
        if (value.find('.') != std::string::npos ||
            value.find('e') != std::string::npos ||
            value.find('E') != std::string::npos) {
            // Parse as double for floating point numbers and scientific notation
            double d = std::stod(node->value);
            return Value(d);
            } else {
                // 先尝试用 int 解析，只有溢出时才用 BigInt
                try {
                    int i = std::stoi(node->value);
                    return Value(i);
                } catch (const std::out_of_range&) {
                    // int 溢出，使用 BigInt
                    ::BigInt big(node->value);
                    return Value(big);
                }
            }
    } else {
        // Check for boolean literals
        if (node->value == "true") return Value(true);
        if (node->value == "false") return Value(false);
        if (node->value == "null") return Value(nullptr);
        // Otherwise it's a string
        return Value(node->value);
    }
}

Value Interpreter::eval_CallExpr(const CallExpr* call) {
    std::string actual_callee = call->callee;
        //        std::cout << "DEBUG: Call expression with callee: '" << actual_callee << "'" << std::endl;

        // 检查调用的名称是否是一个参数，如果是，获取其实际值
        try {
            Value callee_value = get_variable(actual_callee);
            if (callee_value.is_string() && std::get<std::string>(callee_value.data).substr(0, 11) == "__function_") {
                // 这是一个函数参数，提取实际的函数名
                actual_callee = std::get<std::string>(callee_value.data).substr(11);
                //   std::cout << "DEBUG: Function parameter resolved to: '" << actual_callee << "'" << std::endl;
            }
        } catch (const RuntimeError&) {
            // 如果不是变量，保持原名称
            // std::cout << "DEBUG: Not a variable or parameter, using direct name: '" << actual_callee << "'" << std::endl;
        }

        // Check builtin functions first
        //   std::cout << "DEBUG: Looking for builtin function: '" << actual_callee << "'" << std::endl;
        //   std::cout << "DEBUG: Available builtin functions:" << std::endl;
        // for (const auto& pair : builtin_functions) {
        //     std::cout << "  - '" << pair.first << "'" << std::endl;
        // }

        auto builtin_it = builtin_functions.find(actual_callee);
        if (builtin_it != builtin_functions.end()) {
            //     std::cout << "DEBUG: Found builtin function: '" << actual_callee << "'" << std::endl;

            // 检查是否是模块函数
            bool is_module_func = actual_callee.find(".") != std::string::npos;
            if (is_module_func) {
                //       std::cout << "DEBUG: This is a module function with dot notation" << std::endl;
            }

            // Handle builtin call with stack frame and unified error handling
            push_frame(actual_callee, "<builtin>", 0);

            std::vector<Value> args;
            for (const auto& arg: call->args) {
                if (!arg) {
                    pop_frame();
                    RuntimeError err("Null argument in call to builtin function '" + actual_callee + "'");
                    err.stack_trace = get_stack_trace();
                    throw err;
                }
                args.push_back(eval(arg.get()));
            }

            Value result;
            try {
                result = builtin_it->second(args);
            } catch (...) {
                pop_frame();
                throw;
            }

            pop_frame();
            return result;
        }

        // Check user-defined functions
        auto it = functions.find(actual_callee);
        if (it != functions.end()) {
            FuncDefStmt* func = it->second;
            if (!func) {
                std::cerr << "Error: Function object for '" << actual_callee << "' is null" << std::endl;
                return Value("<func error>");
            }

            // Check recursion depth
            if (recursion_depth >= max_recursion_depth) {
                RuntimeError error("Maximum recursion depth exceeded (" + std::to_string(max_recursion_depth) + ")");
                error.stack_trace = get_stack_trace();
                throw error;
            }

            recursion_depth++;
            push_frame(actual_callee, "<script>", 0);// Add to call stack

            // Check parameter count
            if (call->args.size() > func->params.size()) {
                Interpreter::print_warning("Too many arguments provided to function '" + actual_callee +
                                                   "'. Expected " + std::to_string(func->params.size()) +
                                                   ", got " + std::to_string(call->args.size()),
                                           true);
            }

            // Calculate arguments
            std::vector<Value> arg_values(func->params.size());
            for (size_t j = 0; j < func->params.size(); ++j) {
                if (j < call->args.size()) {
                    if (call->args[j]) {
                        arg_values[j] = eval(call->args[j].get());
                    } else {
                        Interpreter::print_error("Null argument " + std::to_string(j + 1) + " in call to function '" + actual_callee + "'", true);
                        arg_values[j] = Value("<null arg>");
                    }
                } else {
                    Interpreter::print_warning("Missing argument '" + func->params[j] + "' in call to function '" + actual_callee + "'", true);
                    arg_values[j] = Value("<undefined>");
                }
            }

            push_scope();// Create scope here

            // Pass arguments
            for (size_t j = 0; j < func->params.size(); ++j) {
                set_variable(func->params[j], arg_values[j]);
            }

            // Execute function body, capture return
            try {
                for (const auto& stmt: func->body->statements) {
                    try {
                        execute(stmt);
                    } catch (const ReturnException& re) {
                        // Normal return handling
                        pop_frame();
                        pop_scope();
                        recursion_depth--;
                        return re.value;
                    } catch (const RuntimeError& re) {
                        // If the error doesn't have a stack trace yet, capture current state
                        RuntimeError enriched(re.message);
                        if (re.stack_trace.empty()) {
                            enriched.stack_trace = get_stack_trace();
                        } else {
                            enriched.stack_trace = re.stack_trace;// Preserve existing trace
                        }
                        pop_frame();
                        pop_scope();
                        recursion_depth--;
                        throw enriched;
                    } catch (const std::exception& e) {
                        // Wrap standard exception as RuntimeError
                        RuntimeError enriched("In function '" + actual_callee + "': " + std::string(e.what()));
                        enriched.stack_trace = get_stack_trace();// Get stack trace before cleanup
                        pop_frame();
                        pop_scope();
                        recursion_depth--;
                        throw enriched;
                    }
                }
            } catch (const ReturnException& re) {
                // Ensure cleanup in any case
                pop_frame();
                pop_scope();
                recursion_depth--;
                return re.value;
            }

            pop_frame();
            pop_scope();
            recursion_depth--;
            return Value();// Default value when no return
        }

        // Check if it's a module function before reporting undefined
        bool is_module_func = actual_callee.find(".") != std::string::npos;
        if (is_module_func) {
            // Prepare arguments for module function call
            std::vector<Value> args;
            for (const auto& arg: call->args) {
                if (!arg) {
                    std::cerr << "Error: Null argument in call to module function '" << actual_callee << "'" << std::endl;
                    args.push_back(Value());
                } else {
                    args.push_back(eval(arg.get()));
                }
            }

            // Try to call the module function
            Value result = call_module_function(actual_callee, args);
            if (result.to_string() != "null") {// 检查是否成功调用
                return result;
            }
            // If module function call failed, fall through to undefined function error
        }

        std::cerr << "Error: Call to undefined function '" << actual_callee << "'" << std::endl;
        return Value("<undefined function>");
}

Value Interpreter::eval_BinaryExpr(const BinaryExpr* bin) {
    Value l = eval(bin->left.get());
    Value r = eval(bin->right.get());

    // Handle arithmetic operations
    if (bin->op == "+") {
        // String concatenation
        if (l.is_string() || r.is_string()) {
            return Value(l.to_string() + r.to_string());
        }
        // Vector addition
        else if (l.is_array() && r.is_array()) {
            return l.vector_add(r);
        }
        // 只要有一方是 Irrational 或 Symbolic，优先生成符号表达式
        else if ((l.is_irrational() || l.is_symbolic() || r.is_irrational() || r.is_symbolic()) && l.is_numeric() && r.is_numeric()) {
            std::shared_ptr<SymbolicExpr> leftExpr;
            std::shared_ptr<SymbolicExpr> rightExpr;
            if (l.is_symbolic()) {
                leftExpr = std::get<std::shared_ptr<SymbolicExpr>>(l.data);
            } else if (l.is_irrational()) {
                leftExpr = std::get<::Irrational>(l.data).to_symbolic();
            } else if (l.is_rational()) {
                leftExpr = SymbolicExpr::number(std::get<::Rational>(l.data));
            } else if (l.is_bigint()) {
                leftExpr = SymbolicExpr::number(std::get<::BigInt>(l.data));
            } else if (l.is_int()) {
                leftExpr = SymbolicExpr::number(std::get<int>(l.data));
            } else if (l.is_float()) {
                leftExpr = SymbolicExpr::number(::Rational::from_double(std::get<double>(l.data)));
            } else {
                leftExpr = SymbolicExpr::number(0);
            }
            if (r.is_symbolic()) {
                rightExpr = std::get<std::shared_ptr<SymbolicExpr>>(r.data);
            } else if (r.is_irrational()) {
                rightExpr = std::get<::Irrational>(r.data).to_symbolic();
            } else if (r.is_rational()) {
                rightExpr = SymbolicExpr::number(std::get<::Rational>(r.data));
            } else if (r.is_bigint()) {
                rightExpr = SymbolicExpr::number(std::get<::BigInt>(r.data));
            } else if (r.is_int()) {
                rightExpr = SymbolicExpr::number(std::get<int>(r.data));
            } else if (r.is_float()) {
                rightExpr = SymbolicExpr::number(::Rational::from_double(std::get<double>(r.data)));
            } else {
                rightExpr = SymbolicExpr::number(0);
            }
            return Value(SymbolicExpr::add(leftExpr, rightExpr)->simplify());
        }
        // Numeric addition with irrational and rational number support
        else if (l.is_numeric() && r.is_numeric()) {
            // BigInt 优先：如果任一为 BigInt，结果为 BigInt
            if (l.is_bigint() || r.is_bigint()) {
                ::BigInt lb = l.is_bigint() ? std::get<::BigInt>(l.data) : ::BigInt(l.as_number());
                ::BigInt rb = r.is_bigint() ? std::get<::BigInt>(r.data) : ::BigInt(r.as_number());
                return Value(lb + rb);
            }
            // If either operand is rational, use rational arithmetic
            if (l.is_rational() || r.is_rational()) {
                ::Rational result = l.as_rational() + r.as_rational();
                return Value(result);
            }

            double result = l.as_number() + r.as_number();// Return int if both operands are int and result is the whole
            if (l.is_int() && r.is_int()) {
                // check overflow and underflow
                if (static_cast<int>(result) == INT_MAX ||
                        static_cast<int>(result) == INT_MIN) {
                    return Value(BigInt(l.as_number())+BigInt(r.as_number()));
                }
                return Value(static_cast<int>(result));
            }
            return Value(result);
        } else {
            error_and_exit("Cannot add " + l.to_string() + " and " + r.to_string());
        }
    }
    // Arithmetic operations (require numeric operands or vector operations)
    if (bin->op == "-" || bin->op == "*" || bin->op == "/" ||
            bin->op == "%" || bin->op == "^") {
        // Special handling for multiplication
        if (bin->op == "*") {
            // Vector and matrix operations
            if (l.is_array() && r.is_array()) {
                // Try dot product for same-size vectors
                const auto& la = std::get<std::vector<Value>>(l.data);
                const auto& ra = std::get<std::vector<Value>>(r.data);
                if (la.size() == ra.size()) {
                    return l.dot_product(r);
                }
            }
            // Matrix multiplication
            if (l.is_matrix() && r.is_matrix()) {
                return l.matrix_multiply(r);
            }
            // Scalar multiplication for vectors
            if (l.is_array() && r.is_numeric()) {
                return l.scalar_multiply(r.as_number());
            }
            if (l.is_numeric() && r.is_array()) {
                return r.scalar_multiply(l.as_number());
            }
            // 只要有一方是 Irrational 或 Symbolic，优先生成符号表达式
            if ((l.is_irrational() || r.is_irrational() || l.is_symbolic() || r.is_symbolic()) && l.is_numeric() && r.is_numeric()) {
                std::shared_ptr<SymbolicExpr> leftExpr;
                std::shared_ptr<SymbolicExpr> rightExpr;
                // 强制所有 Irrational 都转为 SymbolicExpr
                if (l.is_symbolic()) {
                    leftExpr = std::get<std::shared_ptr<SymbolicExpr>>(l.data);
                } else {
                    leftExpr = from_number_to_symbolic(l);
                }
                if (r.is_symbolic()) {
                    rightExpr = std::get<std::shared_ptr<SymbolicExpr>>(r.data);
                } else {
                    rightExpr = from_number_to_symbolic(r);
                }
                return Value(SymbolicExpr::multiply(leftExpr, rightExpr)->simplify());
            }
            // Regular multiplication (both must be numeric)
            if (l.is_numeric() && r.is_numeric()) {
                // BigInt 优先：如果任一为 BigInt，结果为 BigInt
                if (l.is_bigint() || r.is_bigint()) {
                    ::BigInt lb = l.is_bigint() ? std::get<::BigInt>(l.data) : ::BigInt(l.as_number());
                    ::BigInt rb = r.is_bigint() ? std::get<::BigInt>(r.data) : ::BigInt(r.as_number());
                    return Value(lb * rb);
                }
                // If either operand is irrational, use irrational arithmetic
                if (l.is_irrational() || r.is_irrational()) {
                    ::Irrational result = l.as_irrational() * r.as_irrational();
                    return Value(result);
                }
                // If either operand is rational, use rational arithmetic
                if (l.is_rational() || r.is_rational()) {
                    ::Rational result = l.as_rational() * r.as_rational();
                    return Value(result);
                }

                double result = l.as_number() * r.as_number();
                if (l.is_int() && r.is_int()) {
                    // check overflow and underflow
                    if (static_cast<int>(result) == INT_MAX ||
                            static_cast<int>(result) == INT_MIN) {
                        return Value(BigInt(l.as_number())*BigInt(r.as_number()));
                    }
                    return Value(static_cast<int>(result));
                }
                return Value(result);
            }
            // Error case
            error_and_exit("Cannot multiply " + l.to_string() + " and " + r.to_string());
        }

        // Special handling for minus
        if (bin->op == "-") {
            // Vector and matrix operations
            if (l.is_array() && r.is_array()) {
                // Try minus for same-size vectors
                const auto& la = std::get<std::vector<Value>>(l.data);
                const auto& ra = std::get<std::vector<Value>>(r.data);
                return l.vector_minus(r);                                // An exception can be raised inside
            }
            // Matrix multiplication
            if (l.is_matrix() && r.is_matrix()) {
                error_and_exit("Arithmetic operation '-' requires numeric or vector operands");
            }
            // Scalar multiplication for vectors
            if ((l.is_array() && r.is_numeric()) || (l.is_numeric() && r.is_array())) {
                error_and_exit("Arithmetic operation '-' requires same-type operands");
            }
            // 只要有一方是 Irrational 或 Symbolic，优先生成符号表达式
            if ((l.is_irrational() || r.is_irrational() || l.is_symbolic() || r.is_symbolic()) && l.is_numeric() && r.is_numeric()) {
                std::shared_ptr<SymbolicExpr> leftExpr;
                std::shared_ptr<SymbolicExpr> rightExpr;
                // 强制所有 Irrational 都转为 SymbolicExpr
                if (l.is_symbolic()) {
                    leftExpr = std::get<std::shared_ptr<SymbolicExpr>>(l.data);
                } else {
                    leftExpr = from_number_to_symbolic(l);
                }
                if (r.is_symbolic()) {
                    rightExpr = std::get<std::shared_ptr<SymbolicExpr>>(r.data);
                } else {
                    rightExpr = from_number_to_symbolic(r);
                }
                return Value(SymbolicExpr::add(leftExpr, SymbolicExpr::multiply(SymbolicExpr::number(-1), rightExpr)->simplify())->simplify());
            }
            // Regular multiplication (both must be numeric)
            if (l.is_numeric() && r.is_numeric()) {
                // BigInt 优先：如果任一为 BigInt，结果为 BigInt
                if (l.is_bigint() || r.is_bigint()) {
                    ::BigInt lb = l.is_bigint() ? std::get<::BigInt>(l.data) : ::BigInt(l.as_number());
                    ::BigInt rb = r.is_bigint() ? std::get<::BigInt>(r.data) : ::BigInt(r.as_number());
                    return Value(lb - rb);
                }
                // If either operand is irrational, use irrational arithmetic
                if (l.is_irrational() || r.is_irrational()) {
                    ::Irrational result = l.as_irrational() - r.as_irrational();
                    return Value(result);
                }
                // If either operand is rational, use rational arithmetic
                if (l.is_rational() || r.is_rational()) {
                    ::Rational result = l.as_rational() - r.as_rational();
                    return Value(result);
                }

                double result = l.as_number() - r.as_number();
                if (l.is_int() && r.is_int()) {
                    // check overflow and underflow
                    if (static_cast<int>(result) == INT_MAX ||
                            static_cast<int>(result) == INT_MIN) {
                        return Value(BigInt(l.as_number())-BigInt(r.as_number()));
                    }
                    return Value(static_cast<int>(result));
                }
                return (l.is_int() && r.is_int()) ? Value(static_cast<int>(result)) : Value(result);
            }
            // Error case
            error_and_exit("Cannot decrease " + l.to_string() + " by " + r.to_string());
        }

        // Other arithmetic operations require both operands to be numeric
        if (!l.is_numeric() || !r.is_numeric()) {
            error_and_exit("Arithmetic operation '" + bin->op + "' requires numeric operands");
        }

        // For division, always use rational arithmetic for precise results
        if (bin->op == "/") {
            // 只要有一方是 Irrational 或 Symbolic，优先生成符号表达式
            if ((l.is_irrational() || l.is_symbolic() || r.is_irrational() || r.is_symbolic()) && l.is_numeric() && r.is_numeric()) {
                std::shared_ptr<SymbolicExpr> leftExpr;
                std::shared_ptr<SymbolicExpr> rightExpr;
                if (l.is_symbolic()) {
                    leftExpr = std::get<std::shared_ptr<SymbolicExpr>>(l.data);
                } else {
                    leftExpr = from_number_to_symbolic(l);
                }
                if (r.is_symbolic()) {
                    rightExpr = std::get<std::shared_ptr<SymbolicExpr>>(r.data);
                } else {
                    rightExpr = from_number_to_symbolic(r);
                }
                auto divExpr = SymbolicExpr::multiply(leftExpr, SymbolicExpr::power(rightExpr, SymbolicExpr::number(-1)));
                return Value(divExpr->simplify());
            }
            // BigInt 优先：如果任一为 BigInt，结果为 BigInt（如果整除）或 Rational
            if (l.is_bigint() || r.is_bigint()) {
                ::BigInt lb = l.is_bigint() ? std::get<::BigInt>(l.data) : ::BigInt(l.as_number());
                ::BigInt rb = r.is_bigint() ? std::get<::BigInt>(r.data) : ::BigInt(r.as_number());
                if (rb.is_zero()) {
                    error_and_exit("Division by zero");
                }
                // 对于BigInt除法，如果能整除则返回BigInt，否则返回Rational
                try {
                    ::BigInt quotient = lb / rb;
                    ::BigInt remainder = lb - (quotient * rb);
                    if (remainder.is_zero()) {
                        return Value(quotient);
                    } else {
                        // 不能整除，返回有理数
                        return Value(::Rational(lb, rb));
                    }
                } catch (...) {
                    // 如果BigInt运算失败，回退到Rational
                    return Value(::Rational(lb, rb));
                }
            }
            // If either operand is irrational, use irrational arithmetic
			// TODO:这部分代码存在问题
            if (l.is_irrational() || r.is_irrational()) {
                ::Irrational lr = l.as_irrational();
                ::Irrational rr = r.as_irrational();
                if (rr.is_zero()) {
                    error_and_exit("Division by zero");
                }
                return Value(lr / rr);
            }

            ::Rational lr = l.as_rational();
            ::Rational rr = r.as_rational();
            if (rr.is_zero()) {
                error_and_exit("Division by zero");
            }
            return Value(lr / rr);
        }

        if (bin->op == "%") {
            if ((l.is_irrational() || l.is_symbolic() || r.is_irrational() || r.is_symbolic() || l.is_rational() || r.is_rational()) && l.is_numeric() && r.is_numeric()) {
                // 有小数，使用小数取模
                double ld = l.as_number();
                double rd = r.as_number();
                if (rd == 0.0) {
                    error_and_exit("Modulo by zero");
                }
                return Value(ld - rd * std::floor(ld / rd));// 保持结果非负
            }
            if (l.is_bigint() || r.is_bigint()) {
                // 有BigInt，使用BigInt内置方法
                ::BigInt lb = l.is_bigint() ? std::get<::BigInt>(l.data) : ::BigInt(l.as_number());
                ::BigInt rb = r.is_bigint() ? std::get<::BigInt>(r.data) : ::BigInt(r.as_number());
                return Value(lb % rb);
            }
            // 都为int
            return Value(static_cast<int>(l.as_number()) % static_cast<int>(r.as_number()));
        }

        if (bin->op == "^") {
            if ((l.is_irrational() || l.is_symbolic() || r.is_irrational() || r.is_symbolic()) && l.is_numeric() && r.is_numeric()) {
                // 只要有一方是 Irrational 或 Symbolic，优先生成符号表达式
                std::shared_ptr<SymbolicExpr> leftExpr;
                std::shared_ptr<SymbolicExpr> rightExpr;
                if (l.is_symbolic()) {
                    leftExpr = std::get<std::shared_ptr<SymbolicExpr>>(l.data);
                } else {
                    leftExpr = from_number_to_symbolic(l);
                }
                if (r.is_symbolic()) {
                    rightExpr = std::get<std::shared_ptr<SymbolicExpr>>(r.data);
                } else {
                    rightExpr = from_number_to_symbolic(r);
                }
                auto exponentExpr = SymbolicExpr::power(leftExpr, rightExpr);
                return Value(exponentExpr->simplify());
            }
            if (l.is_rational() && (r.is_bigint() || r.is_int())) {
                // 如果底数为Rational，指数为整数，结果为Rational
                ::BigInt rb = r.is_bigint() ? std::get<::BigInt>(r.data) : ::BigInt(r.as_number());
                return Value(l.as_rational().power(rb));
            }
            if ((l.is_bigint() || r.is_int())&& (r.is_bigint() || r.is_int())) {
                // 如果底数为整数，指数为整数，结果为BigInt
                ::BigInt lb = l.is_bigint() ? std::get<::BigInt>(l.data) : ::BigInt(l.as_number());
                ::BigInt rb = r.is_bigint() ? std::get<::BigInt>(r.data) : ::BigInt(r.as_number());
				if (rb < ::BigInt(0)) {
					return Value(l.as_rational().reciprocal().power(::BigInt(0) - rb));
				} else {
					return Value(lb.power(rb));
				}
            }
			if ((l.is_int() || l.is_bigint() || l.is_rational()) && r.is_rational()) {
				// 底数和指数均为Rational，考虑符号表达式
				::Rational lb;
				if (l.is_int()) lb = ::Rational(l.as_number());
				else if (l.is_bigint()) lb = ::Rational(std::get<::BigInt>(l.data));
				else lb = std::get<::Rational>(l.data);
				return Value(SymbolicExpr::power(SymbolicExpr::number(lb), SymbolicExpr::number(std::get<::Rational>(r.data)))->simplify());
			}
            // 有小数，采用小数幂
            double ld = l.as_number();
            double rd = r.as_number();

            return Value(std::pow(ld, rd));
        }

        error_and_exit("Unknown Arithmetic Operation.");
    }

    // Comparison operators
    if (bin->op == "==" || bin->op == "!=" || bin->op == "<" ||
        bin->op == "<=" || bin->op == ">" || bin->op == ">=") {
        // Handle different type combinations
        if (l.is_numeric() && r.is_numeric()) {
            // BigInt 比较优先
            if (l.is_bigint() || r.is_bigint()) {
                ::BigInt lb = l.is_bigint() ? std::get<::BigInt>(l.data) : ::BigInt(l.as_number());
                ::BigInt rb = r.is_bigint() ? std::get<::BigInt>(r.data) : ::BigInt(r.as_number());

                // 使用字符串比较来判断大小（这是一个简化的实现）
                std::string ls = lb.to_string();
                std::string rs = rb.to_string();

                if (bin->op == "==") return Value(ls == rs);
                if (bin->op == "!=") return Value(ls != rs);

                // 对于大小比较，需要考虑符号和长度
                bool lb_neg = ls[0] == '-';
                bool rb_neg = rs[0] == '-';

                if (lb_neg && !rb_neg) {
                    // 左负右正
                    if (bin->op == "<") return Value(true);
                    if (bin->op == "<=") return Value(true);
                    if (bin->op == ">") return Value(false);
                    if (bin->op == ">=") return Value(false);
                } else if (!lb_neg && rb_neg) {
                    // 左正右负
                    if (bin->op == "<") return Value(false);
                    if (bin->op == "<=") return Value(false);
                    if (bin->op == ">") return Value(true);
                    if (bin->op == ">=") return Value(true);
                } else {
                    // 同号比较：比较绝对值的长度和字典序
                    std::string labs = lb_neg ? ls.substr(1) : ls;
                    std::string rabs = rb_neg ? rs.substr(1) : rs;

                    bool abs_less;
                    if (labs.length() != rabs.length()) {
                        abs_less = labs.length() < rabs.length();
                    } else {
                        abs_less = labs < rabs;
                    }

                    bool result_less = lb_neg ? !abs_less : abs_less;

                    if (bin->op == "<") return Value(result_less);
                    if (bin->op == "<=") return Value(result_less || ls == rs);
                    if (bin->op == ">") return Value(!result_less && ls != rs);
                    if (bin->op == ">=") return Value(!result_less);
                }
            } else {
                double ld = l.as_number();
                double rd = r.as_number();

                if (bin->op == "==") return Value(ld == rd);
                if (bin->op == "!=") return Value(ld != rd);
                if (bin->op == "<") return Value(ld < rd);
                if (bin->op == "<=") return Value(ld <= rd);
                if (bin->op == ">") return Value(ld > rd);
                if (bin->op == ">=") return Value(ld >= rd);
            }
        } else if (l.is_string() && r.is_string()) {
            std::string ls = std::get<std::string>(l.data);
            std::string rs = std::get<std::string>(r.data);

            if (bin->op == "==") return Value(ls == rs);
            if (bin->op == "!=") return Value(ls != rs);
            if (bin->op == "<") return Value(ls < rs);
            if (bin->op == "<=") return Value(ls <= rs);
            if (bin->op == ">") return Value(ls > rs);
            if (bin->op == ">=") return Value(ls >= rs);
        } else if (l.is_bool() && r.is_bool()) {
            bool lb = std::get<bool>(l.data);
            bool rb = std::get<bool>(r.data);

            if (bin->op == "==") return Value(lb == rb);
            if (bin->op == "!=") return Value(lb != rb);
            // For booleans, false < true
            if (bin->op == "<") return Value(lb < rb);
            if (bin->op == "<=") return Value(lb <= rb);
            if (bin->op == ">") return Value(lb > rb);
            if (bin->op == ">=") return Value(lb >= rb);
        } else {
            // Type mismatch - only equality/inequality make sense
            if (bin->op == "==") return Value(false);// Different types are never equal
            if (bin->op == "!=") return Value(true); // Different types are always not equal

            error_and_exit("Cannot compare different types with operator '" + bin->op + "'");
            return Value();
        }
    }

        error_and_exit("Unknown binary operator '" + bin->op + "'");

}

Value Interpreter::eval_UnaryExpr(const UnaryExpr* unary) {
     Value v = eval(unary->operand.get());

    if (unary->op == "-") {
        if (v.type != Value::Type::Int && v.type != Value::Type::BigInt && v.type != Value::Type::Float) {
            RuntimeError error("Unary operator '-' requires integer, float or big integer operand");
            error.stack_trace = get_stack_trace();
            throw error;
        }
        if (v.type == Value::Type::Int) {
            int vi = std::get<int>(v.data);
            return Value(-vi);
        } else if (v.type == Value::Type::Float) {
            float vf = std::get<double>(v.data);
            return Value(-vf);
        } else {
            // For BigInt, negate directly
            ::BigInt big_val = std::get<::BigInt>(v.data);
            return Value(big_val.negate());
        }
    }

    if (unary->op == "!") {
        if (v.type != Value::Type::Int && v.type != Value::Type::BigInt) {
            RuntimeError error("Unary operator '!' requires integer or big integer operand");
            error.stack_trace = get_stack_trace();
            throw error;
        }
        int vi;
        if (v.type == Value::Type::Int) {
            vi = std::get<int>(v.data);
        } else {
            vi = std::get<::BigInt>(v.data).to_int();
        }

        if (vi < 0) {
            RuntimeError error("Cannot calculate factorial of negative number");
            error.stack_trace = get_stack_trace();
            throw error;
        }

        // Use BigInt for factorial if the number is large or result would be large
        if (vi > 20) {
            ::BigInt result(1);
            for (int j = 2; j <= vi; ++j) {
                result = result * ::BigInt(j);
            }
            return Value(result);
        } else {
            // Use regular int for small factorials
            int res = 1;
            for (int j = 1; j <= vi; ++j) res *= j;
            return Value(res);
        }
    }

    std::cerr << "Error: Unknown unary operator '" << unary->op << "'" << std::endl;
    return Value("<unknown op>");
}
