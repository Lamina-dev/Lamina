#pragma once
#ifdef _WIN32
#ifdef LAMINA_CORE_EXPORTS
#define LAMINA_API __declspec(dllexport)
#else
#define LAMINA_API __declspec(dllimport)
#endif
#else
#define LAMINA_API
#endif
#include "lamina_api/lamina.hpp"
#include "lamina_api/ast.hpp"
#include "lamina_api/value.hpp"
#include <functional>
#include <memory>
#include <set>
#include <stack>
#include <string>
#include <unordered_map>
#include <vector>

// 前向声明在 lamina.hpp 中已定义，无需重复声明

// Stack frame for function call tracking
struct StackFrame {
    std::string function_name;
    std::string file_name;
    int line_number;

    StackFrame(const std::string& func, const std::string& file, int line)
        : function_name(func), file_name(file), line_number(line) {}
};

// Enhanced runtime error class with stack trace support
class RuntimeError : public std::exception {
public:
    std::string message;
    std::vector<StackFrame> stack_trace;

    RuntimeError(const std::string& msg) : message(msg) {}
    RuntimeError(const std::string& msg, const std::vector<StackFrame>& trace)
        : message(msg), stack_trace(trace) {}

    [[nodiscard]] const char* what() const noexcept override {
        return message.c_str();
    }
};

// Exception for return statements
class ReturnException : public std::exception {
public:
    Value value;
    explicit ReturnException(const Value& v) : value(v) {}
};


// Exception for break statements
class BreakException : public std::exception {
public:
    BreakException() = default;
};


// Exception for continue statements
class ContinueException : public std::exception {
public:
    ContinueException() = default;
};


class LAMINA_API Interpreter {
    // 禁止拷贝，允许移动
    Interpreter(const Interpreter&) = delete;
    Interpreter& operator=(const Interpreter&) = delete;
    Interpreter(Interpreter&&) = default;
    Interpreter& operator=(Interpreter&&) = default;

public:
    Interpreter();
    ~Interpreter();
    static Value execute(const std::unique_ptr<Statement>& node);
    static Value eval(const ASTNode* node);

    static Value eval_LiteralExpr(const LiteralExpr* node);
    static Value eval_UnaryExpr(const UnaryExpr* unary);
    static Value eval_BinaryExpr(const BinaryExpr* bin);
    static Value eval_CallExpr(const CallExpr* call);
    static Value call_function(const LambdaDeclExpr* func, const std::vector<Value>& args, Value self=LAMINA_NULL, Value module=LAMINA_NULL);

    // Print all variables in current scope
    static void print_variables();

    // Save AST in REPL mode to keep function pointers valid
    static void save_repl_ast(std::unique_ptr<ASTNode> ast);

    // Stack trace management
    static void push_frame(const std::string& function_name, const std::string& file_name = "<script>", int line_number = 0);
    static void pop_frame();
    static std::vector<StackFrame> get_stack_trace();
    static void print_stack_trace(const RuntimeError& error, bool use_colors = true);

    // Utility functions for error display
    static bool supports_colors();
    static void print_error(const std::string& message, bool use_colors = true);
    static void print_warning(const std::string& message, bool use_colors = true);

    // Store builtins
    static std::unordered_map<std::string, Value> builtins;

    // Variable assignment
    static void set_variable(const std::string& name, const Value& val);

    // built global variable in interpreter
    static void set_global_variable(const std::string& name, const Value& val);

    // Variable lookup
    static Value get_variable(const std::string& name);

    // Variable scope stack, top is the current scope
    static std::vector<std::unordered_map<std::string, Value>> variable_stack;
	static std::vector<Value> module_stack;

	static void set_module_as(const Value& mod);

private:
    // Store REPL ASTs to keep function pointers valid in interactive mode
    static std::vector<std::unique_ptr<ASTNode>> repl_asts;

    // Stack trace for function calls
    static std::vector<StackFrame> call_stack;
    // Enter/exit scope
    static void push_scope();
    static void pop_scope();
    // Load and execute module
    static bool load_module(const std::string& module_path);
    static bool load_cpp_module(const std::string& module_path);

    // 将Number转为Symbolic(如果可能)
    static std::shared_ptr<SymbolicExpr> from_number_to_symbolic(const Value& v);
    // 移除静态成员变量声明，改用函数内静态变量
    // static std::vector<EntryFunction> entry_functions;
};

inline void print_them() {
    std::cout << "Thanks to all the contributors" << std::endl;
}

inline void print_logo() {
    // Output ASCII art logo
    std::cout << "   __                    _            \n";
    std::cout << "  / /   ____ _____ ___  (_)___  ____ _\n";
    std::cout << " / /   / __ `/ __ `__ \\/ / __ \\/ __ `/\n";
    std::cout << "/ /___/ /_/ / / / / / / / / / / /_/ / \n";
    std::cout << "/_____/\\__,_/_/ /_/ /_/_/_/ /_/\\__,_/  \n";
    std::cout << "                                       \n";
}
