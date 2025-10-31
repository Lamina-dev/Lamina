#include "interpreter.hpp"
#include "lmStruct.hpp"


#include <fstream>
#include <cstdlib>
#include <exception>

#include "../../interpreter/lamina_api/lamina.hpp"
#include "../../interpreter/lamina_api/value.hpp"
#include "standard.hpp"

Value input(const std::vector<Value>& args) {
    std::string input_line;

    if (args.size() == 1) {
        std::cout << args[0].to_string();
    }

    if (std::getline(std::cin, input_line)) {
        return Value(input_line);
    }

    // Return empty string if input failed
    return Value("");
}

Value print(const std::vector<Value>& args) {
    for (size_t i = 0; i < args.size(); ++i) {
        std::cout << args[i].to_string();
        if (i != args.size() - 1) {
            std::cout << " ";
        }
    }

    std::cout << std::endl;
    return Value();
}

/**
 * 终端内执行系统指令
 */
Value system_(const std::vector<Value>& args) {
    if (args.empty()) {
        L_ERR("exec requires 1 argument: command");
    }

    if (!args[0].is_string()) {
        L_ERR("The first argument of exec must be a string (command).");
    }

    const std::string command = args[0].to_string();
    const int result = std::system(command.c_str());

    if (result != 0) {
        L_ERR("Command execution failed: " + command);
    }

    return Value(result);
}

Value lm_assert(const std::vector<Value>& args) {
    const bool cond = !args.empty()
                              ? args[0].as_bool()
                              : false;
    const auto msg = args.size() > 1
                             ? args[1].to_string()
                             : "None";
    if (!cond) {
        L_ERR("Assertion: " + msg);
    }

    return LAMINA_NULL;
}

// 局部变量表
Value locals(const std::vector<Value>& args) {
    auto table = Interpreter::variable_stack.back();
    std::vector<std::pair<std::string, Value>> init_vec = {};
    for (auto [key, value] : table) {
        init_vec.emplace_back(key, value);
    }
    return {std::make_shared<lmStruct>(init_vec)};
}

// 全局变量表
Value globals(const std::vector<Value>& args) {
    auto table = Interpreter::variable_stack[0];
    std::vector<std::pair<std::string, Value>> init_vec = {};
    for (auto [key, value] : table) {
        init_vec.emplace_back(key, value);
    }
    return {std::make_shared<lmStruct>(init_vec)};
}

// 变量表
Value vars(const std::vector<Value>& args) {
    const auto tables = Interpreter::variable_stack;
    std::vector<Value> results{};

    for (auto table: tables) {
        std::vector<std::pair<std::string, Value>> init_vec = {};
        for (auto [key, value] : table) {
            init_vec.emplace_back(key, value);
        }
        results.emplace_back(std::make_shared<lmStruct>(init_vec));
    }
    return results;
}

// 退出程序
Value exit_(const std::vector<Value>& args){
    if (args.empty()) return LAMINA_NULL;
    const auto err_code = std::get<int>(args[0].data);
    std::exit(err_code);
    return LAMINA_NULL;
}

// 错误处理函数
Value xpcall(const std::vector<Value>& args){
    if (args.size() < 2) return LAMINA_NULL;
    if (!args[0].is_lambda() and !args[1].is_lambda()) return LAMINA_NULL;
    const auto func = std::get<std::shared_ptr<LambdaDeclExpr>>(args[0].data);
    const auto handle = std::get<std::shared_ptr<LambdaDeclExpr>>(args[1].data);
    const auto new_args = std::vector(args.begin() + 2, args.end());
    try {
        Value result = Interpreter::call_function(
                func.get(), new_args);
        return result;
    }
    catch (std::exception& e) {
        try {
            return Interpreter::call_function(
                handle.get(), {Value(e.what())}
            );
        } catch (...) {
            return LAMINA_NULL;
        }
    }
    catch (...) {
        return LAMINA_NULL;
    }
}

Value typeof_(const std::vector<Value>& args) {
    if (args.empty()) {
        return LAMINA_NULL;
    }
    switch (args[0].type) {
        case Value::Type::Lambda:    return LAMINA_STRING("lambda");
        case Value::Type::lmStruct:  return LAMINA_STRING("struct");
        case Value::Type::Symbolic:  return LAMINA_STRING("symbolic");
        case Value::Type::Null:      return LAMINA_STRING("null");
        case Value::Type::Bool:      return LAMINA_STRING("bool");
        case Value::Type::Int:       return LAMINA_STRING("int");
        case Value::Type::Float:     return LAMINA_STRING("float");
        case Value::Type::BigInt:    return LAMINA_STRING("bigint");
        case Value::Type::Rational:  return LAMINA_STRING("rational");
        case Value::Type::Irrational:return LAMINA_STRING("irrational");
        case Value::Type::String:    return LAMINA_STRING("string");
        case Value::Type::Array:     return LAMINA_STRING("array");
        case Value::Type::Set:       return LAMINA_STRING("set");
        case Value::Type::Matrix:    return LAMINA_STRING("matrix");
        case Value::Type::lmCppFunction:return LAMINA_STRING("cpp_func");
        case Value::Type::lmModule:  return LAMINA_STRING("module");
        default: return LAMINA_NULL;
    }

}
