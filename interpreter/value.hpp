#pragma once
#include "bigint.hpp"
#include "irrational.hpp"
#include "rational.hpp"
#include "symbolic.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <variant>
#include <vector>

#ifdef _WIN32
#ifdef LAMINA_CORE_EXPORTS
#define LAMINA_API __declspec(dllexport)
#else
#define LAMINA_API __declspec(dllimport)
#endif
#else
#define LAMINA_API
#endif

class lStruct;
LAMINA_API std::string lStruct_to_string(const std::shared_ptr<lStruct>& lstruct);

class LAMINA_API Value {
public:
    enum class Type { Null,
					  Infinity,
                      Bool,
                      Int,
                      Float,
                      String,
                      Array,
                      Matrix,
                      BigInt,
                      Rational,
                      Irrational,
                      lStruct,
                      Symbolic };
    Type type;
	
	using DataType = std::variant<
            std::nullptr_t,
            bool, int, double, std::string,
            std::vector<Value>,
            std::vector<std::vector<Value>>,
            std::vector<std::pair<std::string, Value>>,
            ::BigInt, ::Rational, ::Irrational,
            std::shared_ptr<SymbolicExpr>,
            std::shared_ptr<lStruct>>;
    DataType data;

    virtual ~Value() = default;

    // Constructors
    Value() : type(Type::Null), data(std::in_place_index<0>, nullptr) {}// 0对应std::nullptr_t类型

    // 其他构造函数需要同步修改初始化方式
    Value(std::nullptr_t) : type(Type::Null), data(std::in_place_index<0>, nullptr) {}
    Value(bool b) : type(Type::Bool), data(std::in_place_index<1>, b) {}
    Value(int i) : type(Type::Int), data(std::in_place_index<2>, i) {}
    Value(const std::string& s) : type(Type::String), data(s) {}
    Value(const char* s) : type(Type::String), data(std::string(s)) {}
    Value(const ::BigInt& bi) : type(Type::BigInt), data(bi) {}
    Value(const ::Rational& r) : type(Type::Rational), data(r) {}
    Value(const ::Irrational& ir) : type(Type::Irrational), data(ir) {}
    Value(const std::shared_ptr<lStruct>& lstruct) : type(Type::lStruct), data(lstruct) {}
    Value(const std::shared_ptr<SymbolicExpr>& sym) : type(Type::Symbolic), data(sym) {}
	Value(double f) : type(Type::Float), data(std::in_place_index<3>, f) {
		int res = std::isinf(f);
		if (res) {
			if (f < 0) res = -1;
			this->type = Type::Infinity;
			this->data = DataType(std::in_place_index<2>, res);
		}
	}
    Value(const std::vector<Value>& arr) {
        // Check if this is a matrix (array of arrays)
        bool is_matrix = !arr.empty() && arr[0].is_array();
        if (is_matrix) {
            // Convert to matrix
            std::vector<std::vector<Value>> matrix;
            for (const auto& row: arr) {
                if (row.is_array()) {
                    matrix.push_back(std::get<std::vector<Value>>(row.data));
                } else {
                    // Mixed types, treat as array
                    type = Type::Array;
                    data = arr;
                    return;
                }
            }
            type = Type::Matrix;
            data = matrix;
        } else {
            type = Type::Array;
            data = arr;
        }
    }


    Value(const std::vector<std::vector<Value>>& mat) : type(Type::Matrix), data(mat) {}

    // Type checking helpers
    bool is_null() const { return type == Type::Null; }
	bool is_infinity() const { return type == Type::Infinity; }
    bool is_bool() const { return type == Type::Bool; }
    bool is_int() const { return type == Type::Int; }
    bool is_float() const { return type == Type::Float; }
    bool is_string() const { return type == Type::String; }
    bool is_array() const { return type == Type::Array; }
    bool is_matrix() const { return type == Type::Matrix; }
    bool is_bigint() const { return type == Type::BigInt; }
    bool is_rational() const { return type == Type::Rational; }
    bool is_irrational() const { return type == Type::Irrational; }
    bool is_lstruct() const { return type == Type::lStruct; }
    bool is_symbolic() const { return type == Type::Symbolic; }
    bool is_numeric() const { return type == Type::Int || type == Type::Float || type == Type::BigInt || type == Type::Rational || type == Type::Irrational || type == Type::Symbolic; }
    // Get numeric value as double
    double as_number() const {
		if (type == Type::Infinity) return (1.0 * std::get<int>(data) / 0.0);
        if (type == Type::Int) return static_cast<double>(std::get<int>(data));
        if (type == Type::Float) return std::get<double>(data);
        if (type == Type::BigInt) {
            // For BigInt, try to convert to int first, then to double
            const auto& bigint_val = std::get<::BigInt>(data);
            int int_val = bigint_val.to_int();
            if (int_val == INT_MAX || int_val == INT_MIN) {
                // BigInt was too large for int, use double conversion
                return bigint_val.to_double();
            }
            return static_cast<double>(int_val);
        }
        if (type == Type::Rational) {
            return std::get<::Rational>(data).to_double();
        }
        if (type == Type::Irrational) {
            return std::get<::Irrational>(data).to_double();
        }
        if (type == Type::Symbolic) {
            return std::get<std::shared_ptr<SymbolicExpr>>(data)->to_double();
        }
        return 0.0;
    }

    // Get numeric value as Rational (for precise calculations)
    ::Rational as_rational() const {
        if (type == Type::Rational) return std::get<::Rational>(data);
        if (type == Type::Int) return ::Rational(std::get<int>(data));
        if (type == Type::Float) return ::Rational::from_double(std::get<double>(data));
        if (type == Type::BigInt) {
            int int_val = std::get<::BigInt>(data).to_int();
            return ::Rational(int_val);
        }
        if (type == Type::Irrational) {
            return ::Rational::from_double(std::get<::Irrational>(data).to_double());
        }
        return ::Rational(0);
    }

    // Get numeric value as Irrational (for exact irrational calculations)
    ::Irrational as_irrational() const {
        if (type == Type::Irrational) return std::get<::Irrational>(data);
        if (type == Type::Int) return ::Irrational::constant(std::get<int>(data));
        if (type == Type::Float) return ::Irrational::constant(std::get<double>(data));
        if (type == Type::Rational) return ::Irrational::constant(std::get<::Rational>(data).to_double());
        if (type == Type::BigInt) {
            int int_val = std::get<::BigInt>(data).to_int();
            return ::Irrational::constant(int_val);
        }
        return ::Irrational::constant(0);
    }
	
	std::shared_ptr<SymbolicExpr> as_symbolic() const {
		if (type == Type::Infinity) return SymbolicExpr::infinity(std::get<int>(data));
		if (type == Type::Symbolic) return std::get<std::shared_ptr<SymbolicExpr>>(data);
		if (type == Type::Int || type == Type::Float || type == Type::Rational || type == Type::BigInt) {
			return SymbolicExpr::number(as_rational());
		}
		if (type == Type::Irrational) {
			return as_irrational().to_symbolic();
		}
		return SymbolicExpr::number(0);
	}
	
	bool as_symbolic_compatible() const {
		if (type == Type::Symbolic) return true;
		if (type == Type::Int || type == Type::Float || type == Type::Rational || type == Type::BigInt) return true;
		if (type == Type::Irrational) return true;
		return false;
	}
	
    // Get boolean value
    bool as_bool() const {
		if (type == Type::Infinity) return true;
        if (type == Type::Bool) return std::get<bool>(data);
        if (type == Type::Int) return std::get<int>(data) != 0;
        if (type == Type::Float) return std::get<double>(data) != 0.0;
        if (type == Type::BigInt) return !std::get<::BigInt>(data).is_zero();
        if (type == Type::Rational) return !std::get<::Rational>(data).is_zero();
        if (type == Type::Irrational) return !std::get<::Irrational>(data).is_zero();
        if (type == Type::String) return !std::get<std::string>(data).empty();
        if (type == Type::Array) return !std::get<std::vector<Value>>(data).empty();
        return false;
    }

    // String conversion
    std::string to_string() const {
        switch (type) {
			case Type::Infinity:
				return std::get<int>(data) > 0 ? "inf" : "-inf";
            case Type::Null:
                return "null";
            case Type::Bool:
                return std::get<bool>(data) ? "true" : "false";
            case Type::Int:
                return std::to_string(std::get<int>(data));
            case Type::Float: {
                double val = std::get<double>(data);
                // Remove trailing zeros for cleaner output
                std::string str = std::to_string(val);
                str.erase(str.find_last_not_of('0') + 1, std::string::npos);
                str.erase(str.find_last_not_of('.') + 1, std::string::npos);
                return str;
            }
            case Type::String:
                return std::get<std::string>(data);
            case Type::Array: {
                std::string res = "[";
                const auto& arr = std::get<std::vector<Value>>(data);
                for (size_t i = 0; i < arr.size(); ++i) {
                    if (i) res += ", ";
                    res += arr[i].to_string();
                }
                res += "]";
                return res;
            }
            case Type::Matrix: {
                std::string res = "[";
                const auto& mat = std::get<std::vector<std::vector<Value>>>(data);
                for (size_t i = 0; i < mat.size(); ++i) {
                    if (i) res += ", ";
                    res += "[";
                    for (size_t j = 0; j < mat[i].size(); ++j) {
                        if (j) res += ", ";
                        res += mat[i][j].to_string();
                    }
                    res += "]";
                }
                res += "]";
                return res;
            }
            case Type::BigInt: {
                return std::get<::BigInt>(data).to_string();
            }
            case Type::Rational: {
                return std::get<::Rational>(data).to_string();
            }
            case Type::Irrational: {
                return std::get<::Irrational>(data).to_string();
            }
            case Type::Symbolic: {
                return std::get<std::shared_ptr<SymbolicExpr>>(data)->to_string();
            }
            case Type::lStruct: {
                return lStruct_to_string(std::get<std::shared_ptr<lStruct>>(data));
            }
        }
        return "<unknown>";
    }

    // Vector operations
    Value vector_add(const Value& other) const {
        if (!is_array() || !other.is_array()) {
            std::cerr << "Error: Vector addition requires two arrays" << std::endl;
            return Value();
        }

        const auto& a = std::get<std::vector<Value>>(data);
        const auto& b = std::get<std::vector<Value>>(other.data);

        if (a.size() != b.size()) {
            std::cerr << "Error: Vector addition requires same dimensions" << std::endl;
            return Value();
        }

        std::vector<Value> result;
        for (size_t i = 0; i < a.size(); ++i) {
            if (a[i].is_numeric() && b[i].is_numeric()) {
                result.push_back(Value(a[i].as_number() + b[i].as_number()));
            } else {
                std::cerr << "Error: Vector elements must be numeric" << std::endl;
                return Value();
            }
        }
        return Value(result);
    }

    Value vector_minus(const Value& other) const {
        if (!is_array() || !other.is_array()) {
            std::cerr << "Error: Vector minus requires two arrays" << std::endl;
            return Value();
        }

        const auto& a = std::get<std::vector<Value>>(data);
        const auto& b = std::get<std::vector<Value>>(other.data);

        if (a.size() != b.size()) {
            std::cerr << "Error: Vector minus requires same dimensions" << std::endl;
            return Value();
        }

        std::vector<Value> result;
        for (size_t i = 0; i < a.size(); ++i) {
            if (a[i].is_numeric() && b[i].is_numeric()) {
                result.push_back(Value(a[i].as_number() - b[i].as_number()));
            } else {
                std::cerr << "Error: Vector elements must be numeric" << std::endl;
                return Value();
            }
        }
        return Value(result);
    }

    // Dot product
    Value dot_product(const Value& other) const {
        if (!is_array() || !other.is_array()) {
            std::cerr << "Error: Dot product requires two arrays" << std::endl;
            return Value();
        }

        const auto& a = std::get<std::vector<Value>>(data);
        const auto& b = std::get<std::vector<Value>>(other.data);

        if (a.size() != b.size()) {
            std::cerr << "Error: Dot product requires same dimensions" << std::endl;
            return Value();
        }

        double result = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            if (a[i].is_numeric() && b[i].is_numeric()) {
                result += a[i].as_number() * b[i].as_number();
            } else {
                std::cerr << "Error: Vector elements must be numeric" << std::endl;
                return Value();
            }
        }
        return Value(result);
    }
    // Scalar multiplication
    Value scalar_multiply(double scalar) const {
        if (!is_array()) {
            std::cerr << "Error: Scalar multiplication requires an array" << std::endl;
            return Value();
        }

        const auto& arr = std::get<std::vector<Value>>(data);
        std::vector<Value> result;

        for (const auto& elem: arr) {
            if (elem.is_numeric()) {
                result.push_back(Value(elem.as_number() * scalar));
            } else {
                std::cerr << "Error: Vector elements must be numeric" << std::endl;
                return Value();
            }
        }
        return Value(result);
    }

    // Cross product (for 3D vectors)
    Value cross_product(const Value& other) const {
        if (!is_array() || !other.is_array()) {
            std::cerr << "Error: Cross product requires two arrays" << std::endl;
            return Value();
        }

        const auto& a = std::get<std::vector<Value>>(data);
        const auto& b = std::get<std::vector<Value>>(other.data);

        if (a.size() != 3 || b.size() != 3) {
            std::cerr << "Error: Cross product requires 3D vectors" << std::endl;
            return Value();
        }

        for (size_t i = 0; i < 3; ++i) {
            if (!a[i].is_numeric() || !b[i].is_numeric()) {
                std::cerr << "Error: Vector elements must be numeric" << std::endl;
                return Value();
            }
        }

        double a1 = a[0].as_number(), a2 = a[1].as_number(), a3 = a[2].as_number();
        double b1 = b[0].as_number(), b2 = b[1].as_number(), b3 = b[2].as_number();

        std::vector<Value> result = {
                Value(a2 * b3 - a3 * b2),
                Value(a3 * b1 - a1 * b3),
                Value(a1 * b2 - a2 * b1)};
        return Value(result);
    }

    // Vector magnitude/norm
    Value magnitude() const {
        if (!is_array()) {
            std::cerr << "Error: Magnitude requires an array" << std::endl;
            return Value();
        }

        const auto& arr = std::get<std::vector<Value>>(data);
        double sum = 0.0;

        for (const auto& elem: arr) {
            if (elem.is_numeric()) {
                double val = elem.as_number();
                sum += val * val;
            } else {
                std::cerr << "Error: Vector elements must be numeric" << std::endl;
                return Value();
            }
        }
        return Value(std::sqrt(sum));
    }

    // Vector normalization
    Value normalize() const {
        Value mag = magnitude();
        if (!mag.is_numeric() || mag.as_number() == 0.0) {
            std::cerr << "Error: Cannot normalize zero vector" << std::endl;
            return Value();
        }
        return scalar_multiply(1.0 / mag.as_number());
    }

    // Matrix operations
    Value matrix_multiply(const Value& other) const {
        if (!is_matrix() || !other.is_matrix()) {
            std::cerr << "Error: Matrix multiplication requires two matrices" << std::endl;
            return Value();
        }

        const auto& a = std::get<std::vector<std::vector<Value>>>(data);
        const auto& b = std::get<std::vector<std::vector<Value>>>(other.data);

        if (a.empty() || b.empty() || a[0].size() != b.size()) {
            std::cerr << "Error: Invalid matrix dimensions for multiplication" << std::endl;
            return Value();
        }

        size_t rows = a.size();
        size_t cols = b[0].size();
        size_t inner = a[0].size();

        std::vector<std::vector<Value>> result(rows, std::vector<Value>(cols, Value(0.0)));

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < inner; ++k) {
                    if (!a[i][k].is_numeric() || !b[k][j].is_numeric()) {
                        std::cerr << "Error: Matrix elements must be numeric" << std::endl;
                        return Value();
                    }
                    sum += a[i][k].as_number() * b[k][j].as_number();
                }
                result[i][j] = Value(sum);
            }
        }
        return Value(result);
    }

    // Matrix determinant (2x2 and 3x3 only)
    Value determinant() const {
        if (!is_matrix()) {
            std::cerr << "Error: Determinant requires a matrix" << std::endl;
            return Value();
        }

        const auto& mat = std::get<std::vector<std::vector<Value>>>(data);

        if (mat.size() != mat[0].size()) {
            std::cerr << "Error: Determinant requires a square matrix" << std::endl;
            return Value();
        }

        size_t n = mat.size();

        if (n == 2) {
            // 2x2 determinant: ad - bc
            if (!mat[0][0].is_numeric() || !mat[0][1].is_numeric() ||
                !mat[1][0].is_numeric() || !mat[1][1].is_numeric()) {
                std::cerr << "Error: Matrix elements must be numeric" << std::endl;
                return Value();
            }
            double a = mat[0][0].as_number();
            double b = mat[0][1].as_number();
            double c = mat[1][0].as_number();
            double d = mat[1][1].as_number();
            return Value(a * d - b * c);
        } else if (n == 3) {
            // 3x3 determinant using rule of Sarrus
            for (size_t i = 0; i < 3; ++i) {
                for (size_t j = 0; j < 3; ++j) {
                    if (!mat[i][j].is_numeric()) {
                        std::cerr << "Error: Matrix elements must be numeric" << std::endl;
                        return Value();
                    }
                }
            }

            double a = mat[0][0].as_number(), b = mat[0][1].as_number(), c = mat[0][2].as_number();
            double d = mat[1][0].as_number(), e = mat[1][1].as_number(), f = mat[1][2].as_number();
            double g = mat[2][0].as_number(), h = mat[2][1].as_number(), i = mat[2][2].as_number();

            return Value(a * e * i + b * f * g + c * d * h - c * e * g - b * d * i - a * f * h);
        } else {
            std::cerr << "Error: Determinant only supported for 2x2 and 3x3 matrices" << std::endl;
            return Value();
        }
    }
};
