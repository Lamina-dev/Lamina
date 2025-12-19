#pragma once
#ifndef LAMINA_EXPORT
#ifdef __GNUC__
#define LAMINA_EXPORT __attribute__((visibility("default")))
#else
#define LAMINA_EXPORT
#endif
#endif

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

// Lamina 版本号
#include "version.hpp"
// Lamina 帮助文本
#include "help_text.hpp"
#include "value.hpp"

#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

/*
    对LAMINA核心资源操作的头文件
 */

class Value;
// 声明错误处理函数

template<class>
constexpr bool always_false = false;


#define LAMINA_BOOL(value) Value((bool) value)
#define LAMINA_INT(value) Value((int) value)
#define LAMINA_DOUBLE(value) Value((double) value)
#define LAMINA_STRING(value) Value((const char*) value)
#define LAMINA_BIGINT(value) Value((const ::BigInt&) value)
#define LAMINA_RATIONAL(value) Value((const ::Rational&) value)
#define LAMINA_IRRATIONAL(value) Value((const ::Irrational&) value)
#define LAMINA_COMPLEX(real, imag) Value(::lamina::Complex(real, imag))
#define LAMINA_ARR(value) Value(value)
#define LAMINA_MATRIX(value) Value(value)
#define LAMINA_NULL Value()


inline std::pair<std::string, Value> LAMINA_FUNC(
    const std::string& name,
    const std::function<Value(std::vector<Value>)>& fn_ptr
) {
    return {
        name, Value(std::make_shared<LmCppFunction>(fn_ptr))
    };
}

inline std::pair<std::string, Value> LAMINA_MODULE(
    const std::string& name,
    const std::string& version,
    const std::unordered_map<std::string, Value>& subitems
) {
    return {
        name, Value(std::make_shared<LmModule>( name, version, subitems ))
    };
}

class StdLibException final : public std::exception {
public:
    std::string message;
    explicit StdLibException(std::string  msg) : message(std::move(msg)) {}
    [[nodiscard]] const char* what() const noexcept override {
        return message.c_str();
    }
};

inline void L_ERR(const std::string& str) {
    throw StdLibException(str);
}

inline bool check_cpp_function_argv(const std::vector<Value>& argv, const size_t argc) {
    if (argv.size() != argc) {
		const std::string msg = "function need " + std::to_string(argc) + " argument(s), but got " + std::to_string(argv.size());
        throw StdLibException(msg);
    }
    return false;
}

inline bool check_cpp_function_argv_x(const std::vector<Value>& argv, const size_t min_argc, const size_t max_argc) {
    if (argv.size() < min_argc || argv.size() > max_argc) {
        const std::string msg = "function need " + std::to_string(min_argc) + " to " + std::to_string(max_argc) + "argument(s), but got " + std::to_string(argv.size());
        throw StdLibException(msg);
    }
    return false;
}

inline bool check_cpp_function_argv(
    const std::vector<Value>& argv,
    const std::vector<Value::Type>& except_type_vec
) {

    if (argv.size() != except_type_vec.size()) {
        const std::string msg = "function need " + std::to_string(except_type_vec.size()) + " arguments, but got " + std::to_string(argv.size());
        throw StdLibException(msg);
        return false;
    }

    size_t cnt = 0;
    for (const auto& arg: argv) {
        if (arg.type != except_type_vec[cnt]) {
            const std::string msg =  "callee used illegal type";
            throw StdLibException(msg);
            return false;
        }
        ++cnt;
    }
    return true;
}
