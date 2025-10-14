#include "cpu_backend.hpp"
#include <iostream>
#include <stdexcept>

// RuntimeError exception class
class RuntimeError : public std::exception {
public:
    std::string message;
    RuntimeError(const std::string& msg) : message(msg) {}
    const char* what() const noexcept override { return message.c_str(); }
};

CPUBackend::CPUBackend() {
    register_functions();
}

bool CPUBackend::initialize() {
    // CPU backend is always available
    return true;
}

void CPUBackend::register_functions() {
    functions_["matmul"] = [this](const std::vector<Value>& args) { return matmul(args); };
    functions_["add"] = [this](const std::vector<Value>& args) { return add(args); };
    functions_["sub"] = [this](const std::vector<Value>& args) { return sub(args); };
    functions_["mul"] = [this](const std::vector<Value>& args) { return mul(args); };
    functions_["div"] = [this](const std::vector<Value>& args) { return div(args); };
    functions_["dot"] = [this](const std::vector<Value>& args) { return dot(args); };
    functions_["cross"] = [this](const std::vector<Value>& args) { return cross(args); };
    functions_["transpose"] = [this](const std::vector<Value>& args) { return transpose(args); };
}

Value CPUBackend::call_function(const std::string& func_name, const std::vector<Value>& args) {
    auto it = functions_.find(func_name);
    if (it == functions_.end()) {
        throw RuntimeError("CPU backend: function '" + func_name + "' not found");
    }
    return it->second(args);
}

std::vector<std::string> CPUBackend::available_functions() const {
    std::vector<std::string> result;
    for (const auto& pair : functions_) {
        result.push_back(pair.first);
    }
    return result;
}

bool CPUBackend::has_function(const std::string& func_name) const {
    return functions_.find(func_name) != functions_.end();
}

// ============================================================================
// CPU operation implementations
// ============================================================================

Value CPUBackend::matmul(const std::vector<Value>& args) {
    if (args.size() != 2) {
        throw RuntimeError("matmul requires exactly 2 arguments");
    }

    const Value& a = args[0];
    const Value& b = args[1];

    // Matrix * Matrix
    if (a.is_matrix() && b.is_matrix()) {
        return a.matrix_multiply(b);
    }

    // Matrix * Vector (treat vector as column matrix)
    if (a.is_matrix() && b.is_array()) {
        const auto& mat = std::get<std::vector<std::vector<Value>>>(a.data);
        const auto& vec = std::get<std::vector<Value>>(b.data);

        if (mat.empty() || vec.empty()) {
            throw RuntimeError("matmul: empty matrix or vector");
        }

        size_t rows = mat.size();
        size_t cols = mat[0].size();

        if (cols != vec.size()) {
            throw RuntimeError("matmul: incompatible dimensions");
        }

        std::vector<Value> result(rows, Value(0.0));
        for (size_t i = 0; i < rows; ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < cols; ++j) {
                sum += mat[i][j].as_number() * vec[j].as_number();
            }
            result[i] = Value(sum);
        }
        return Value(result);
    }

    throw RuntimeError("matmul requires matrix or matrix-vector arguments");
}

Value CPUBackend::add(const std::vector<Value>& args) {
    if (args.size() != 2) {
        throw RuntimeError("add requires exactly 2 arguments");
    }

    const Value& a = args[0];
    const Value& b = args[1];

    // Vector addition
    if (a.is_array() && b.is_array()) {
        return a.vector_add(b);
    }

    // Scalar addition
    if (a.is_numeric() && b.is_numeric()) {
        return Value(a.as_number() + b.as_number());
    }

    throw RuntimeError("add requires numeric or array arguments");
}

Value CPUBackend::sub(const std::vector<Value>& args) {
    if (args.size() != 2) {
        throw RuntimeError("sub requires exactly 2 arguments");
    }

    const Value& a = args[0];
    const Value& b = args[1];

    // Vector subtraction
    if (a.is_array() && b.is_array()) {
        return a.vector_minus(b);
    }

    // Scalar subtraction
    if (a.is_numeric() && b.is_numeric()) {
        return Value(a.as_number() - b.as_number());
    }

    throw RuntimeError("sub requires numeric or array arguments");
}

Value CPUBackend::mul(const std::vector<Value>& args) {
    if (args.size() != 2) {
        throw RuntimeError("mul requires exactly 2 arguments");
    }

    const Value& a = args[0];
    const Value& b = args[1];

    // Scalar * Vector
    if (a.is_numeric() && b.is_array()) {
        return b.scalar_multiply(a.as_number());
    }

    // Vector * Scalar
    if (a.is_array() && b.is_numeric()) {
        return a.scalar_multiply(b.as_number());
    }

    // Scalar * Scalar
    if (a.is_numeric() && b.is_numeric()) {
        return Value(a.as_number() * b.as_number());
    }

    throw RuntimeError("mul requires numeric or array arguments");
}

Value CPUBackend::div(const std::vector<Value>& args) {
    if (args.size() != 2) {
        throw RuntimeError("div requires exactly 2 arguments");
    }

    const Value& a = args[0];
    const Value& b = args[1];

    // Vector / Scalar
    if (a.is_array() && b.is_numeric()) {
        double divisor = b.as_number();
        if (divisor == 0.0) {
            throw RuntimeError("division by zero");
        }
        return a.scalar_multiply(1.0 / divisor);
    }

    // Scalar / Scalar
    if (a.is_numeric() && b.is_numeric()) {
        double divisor = b.as_number();
        if (divisor == 0.0) {
            throw RuntimeError("division by zero");
        }
        return Value(a.as_number() / divisor);
    }

    throw RuntimeError("div requires numeric or array arguments");
}

Value CPUBackend::dot(const std::vector<Value>& args) {
    if (args.size() != 2) {
        throw RuntimeError("dot requires exactly 2 arguments");
    }

    const Value& a = args[0];
    const Value& b = args[1];

    if (a.is_array() && b.is_array()) {
        return a.dot_product(b);
    }

    throw RuntimeError("dot requires array arguments");
}

Value CPUBackend::cross(const std::vector<Value>& args) {
    if (args.size() != 2) {
        throw RuntimeError("cross requires exactly 2 arguments");
    }

    const Value& a = args[0];
    const Value& b = args[1];

    if (!a.is_array() || !b.is_array()) {
        throw RuntimeError("cross requires array arguments");
    }

    const auto& vec_a = std::get<std::vector<Value>>(a.data);
    const auto& vec_b = std::get<std::vector<Value>>(b.data);

    if (vec_a.size() != 3 || vec_b.size() != 3) {
        throw RuntimeError("cross product requires 3D vectors");
    }

    double a1 = vec_a[0].as_number();
    double a2 = vec_a[1].as_number();
    double a3 = vec_a[2].as_number();
    double b1 = vec_b[0].as_number();
    double b2 = vec_b[1].as_number();
    double b3 = vec_b[2].as_number();

    std::vector<Value> result = {
        Value(a2 * b3 - a3 * b2),
        Value(a3 * b1 - a1 * b3),
        Value(a1 * b2 - a2 * b1)
    };

    return Value(result);
}

Value CPUBackend::transpose(const std::vector<Value>& args) {
    if (args.size() != 1) {
        throw RuntimeError("transpose requires exactly 1 argument");
    }

    const Value& a = args[0];

    if (!a.is_matrix()) {
        throw RuntimeError("transpose requires a matrix argument");
    }

    const auto& mat = std::get<std::vector<std::vector<Value>>>(a.data);
    if (mat.empty()) {
        return a;
    }

    size_t rows = mat.size();
    size_t cols = mat[0].size();

    std::vector<std::vector<Value>> result(cols, std::vector<Value>(rows));
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[j][i] = mat[i][j];
        }
    }
    return Value(result);
}
