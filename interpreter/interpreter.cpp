#include "interpreter.hpp"

#include "../extensions/standard/lmStruct.hpp"
#include "../extensions/standard/standard.hpp"
#include "cpp_module_loader.hpp"
#include "lamina_api/bigint.hpp"
#include "lamina_api/lamina.hpp"
#include "lexer.hpp"
#include "parser.hpp"
#include "utils/properties_parser.hpp"
#include "utils/src_manger.hpp"

#include <cmath>
#include <cstdlib>// For std::exit
#include <cstring>// For strcmp
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <ranges>
#include <sstream>
#ifdef _WIN32
#include <direct.h>
#include <io.h>
#include <windows.h>
#elif __linux__
#include <cstdlib>// For getenv
#include <dlfcn.h>
#include <limits.h>
#include <link.h>
#include <unistd.h>
#include <unistd.h>// For isatty
#else
#include <cstdlib>// For getenv
#include <dlfcn.h>
#include <limits.h>
#include <unistd.h>
#include <unistd.h>// For isatty
#endif

std::vector<std::unique_ptr<ASTNode>> Interpreter::repl_asts{};
std::vector<StackFrame> Interpreter::call_stack{};
std::unordered_map<std::string, Value> Interpreter::builtins = register_builtins;
std::vector<std::unordered_map<std::string, Value>> Interpreter::variable_stack{{}};

Value new_lm_struct(const std::vector<std::pair<std::string, Value>>& vec);

Interpreter::Interpreter()= default;

Interpreter::~Interpreter() {
    variable_stack = {{}};
    call_stack = {};
}

// Scope stack operations
void Interpreter::push_scope() {
    variable_stack.emplace_back();
}

void Interpreter::save_repl_ast(std::unique_ptr<ASTNode> ast) {
    repl_asts.push_back(std::move(ast));
}

void Interpreter::pop_scope() {
    if (variable_stack.size() > 1) variable_stack.pop_back();
}

Value Interpreter::get_variable(const std::string& name) {
    for (const auto & it : std::ranges::reverse_view(variable_stack)) {
        auto found = it.find(name);
        if (found != it.end()) return found->second;
    }

    // 如果是函数名找不到，查找builtins
    const auto builtins_it = builtins.find(name);
    if (builtins_it != builtins.end()) {
        return builtins_it->second;
    }

    RuntimeError error("Undefined variable '" + name + "'");
    error.stack_trace = get_stack_trace();
    throw error;
}

void Interpreter::set_variable(const std::string& name, const Value& val) {
    if (!variable_stack.empty()) {
        (*variable_stack.rbegin())[name] = val;
    }
}

void Interpreter::set_global_variable(const std::string& name, const Value& val) {
    if (!variable_stack.empty()) {
        variable_stack.front()[name] = val;
    }
}

Value Interpreter::execute(const std::unique_ptr<Statement>& node) {
	
	//std::cerr << "[Debug output] call of execute in interpreter!\n";
	
    if (!node) {
        return LAMINA_NULL;
    }

    if (auto* v = dynamic_cast<VarDeclStmt*>(node.get())) {
        if (v->expr) {
            Value val = eval(v->expr.get());
            set_variable(v->name, val);
        } else {
            RuntimeError error("Variable '" + v->name + "' declaration has null expression");
            error.stack_trace = get_stack_trace();
            throw error;
        }
    } else if (auto* d = dynamic_cast<DefineStmt*>(node.get())) {
        if (!d->value) {
            L_ERR("Null expression in define statement for '" + d->name + "'");
        }
        Value val = eval(d->value.get());
    } else if (auto* bi = dynamic_cast<BigIntDeclStmt*>(node.get())) {
        if (bi->init_value) {
            Value val = eval(bi->init_value.get());
            if (val.is_bigint()) {
                // 如果值已经是BigInt，直接使用
                set_variable(bi->name, val);
            } else if (val.is_int()) {
                // 将普通整数转换为BigInt
                ::BigInt big_val(std::get<int>(val.data));
                set_variable(bi->name, Value(big_val));
            } else if (val.is_string()) {
                // 从字符串创建BigInt
                try {
                    ::BigInt big_val(std::get<std::string>(val.data));
                    set_variable(bi->name, Value(big_val));
                } catch (const std::exception& e) {
                    L_ERR("Invalid BigInt string '" + std::get<std::string>(val.data) + "' in declaration of " + bi->name);
                }
            } else {
                // 默认初始化为0
                L_ERR("Cannot convert " + val.to_string() + " to BigInt in declaration of " + bi->name);
            }
        } else {
            set_variable(bi->name, Value(::BigInt(0)));
        }
    } else if (auto* a = dynamic_cast<AssignStmt*>(node.get())) {
        if (!a->expr) {
            L_ERR("Null expression in assignment to '" + a->name + "'");
        }
        Value val = eval(a->expr.get());
        set_variable(a->name, val);
    } else if (auto* a = dynamic_cast<StructDeclStmt*>(node.get())) {
        std::vector<std::pair<std::string, Value>> struct_init_val{};
        for (const auto& [n, e]: a->init_vec) {
            auto val = eval(e.get());
            struct_init_val.emplace_back(n, val);
        }
        set_variable(a->name, new_lm_struct(struct_init_val));
    } else if (auto* ifs = dynamic_cast<IfStmt*>(node.get())) {
        if (!ifs->condition) {
            L_ERR("Null condition in if statement");
        }
        Value cond = eval(ifs->condition.get());
        if (cond.as_bool()) {
            for (auto& stmt: ifs->thenBlock->statements) execute(stmt);
        } else if (ifs->elseBlock) {
            for (auto& stmt: ifs->elseBlock->statements) execute(stmt);
        }
    } else if (auto* ws = dynamic_cast<WhileStmt*>(node.get())) {
        if (!ws->condition) {
            RuntimeError error("Loop condition cannot be null");
            error.stack_trace = get_stack_trace();
            throw error;
        }

        // 更健壮的while循环实现

        try {
            while (true) {
                // 评估循环条件
                Value cond;
                try {
                    cond = eval(ws->condition.get());
                } catch (const std::exception& e) {
                    // 如果条件计算出错，优雅地退出循环
                    RuntimeError error("Loop condition error: " + std::string(e.what()));
                    error.stack_trace = get_stack_trace();
                    throw error;
                }

                if (!cond.as_bool()) {
                    // 条件为假，退出循环
                    break;
                }

                bool continue_encountered = false;

                // 执行循环体
                for (auto& stmt: ws->body->statements) {
                    if (continue_encountered) continue;

                    try {
                        execute(stmt);
                    } catch (const BreakException&) {
                        // 退出整个循环
                        return LAMINA_NULL;
                    } catch (const ContinueException&) {
                        // 跳到下一次迭代
                        continue_encountered = true;
                    }
                }
            }
        } catch (const BreakException&) {
            // 捕获可能从嵌套块冒泡上来的break异常
            return LAMINA_NULL;
        } catch (const ReturnException&) {
            // 函数返回，直接传递给上层
            throw;
        } catch (const std::exception& e) {
            // 所有其他异常都作为运行时错误处理
            RuntimeError error("Loop body execution error: " + std::string(e.what()));
            error.stack_trace = get_stack_trace();
            throw error;
        }
    } else if (auto* func = dynamic_cast<FuncDefStmt*>(node.get())) {
        set_variable(func->name, Value(std::make_shared<LambdaDeclExpr>(
            func->name, func->params, std::move(func->body))));
    } else if (auto* block = dynamic_cast<BlockStmt*>(node.get())) {
        for (auto& stmt: block->statements) execute(stmt);
    } else if (auto* rs = dynamic_cast<RepeatStmt*>(node.get())) {
        if (!rs->count) {
            RuntimeError error("Repeat count expression cannot be null");
            error.stack_trace = get_stack_trace();
            throw error;
        }

        // Evaluate count and coerce/validate various numeric types into a non-negative integer
        Value cntv = eval(rs->count.get());
        long long times = 0;

        // Integer literal
        if (cntv.is_int()) {
            times = static_cast<long long>(std::get<int>(cntv.data));
        }
        // BigInt -> convert if within range
        else if (cntv.is_bigint()) {
            const ::BigInt& bi = std::get<::BigInt>(cntv.data);
            int bi_int = bi.to_int();
            if (bi_int == INT_MAX || bi_int == INT_MIN) {
                RuntimeError error("Repeat count too large");
                error.stack_trace = get_stack_trace();
                throw error;
            }
            times = static_cast<long long>(bi_int);
        }
        // Rational: accept when it's an exact integer (denominator == 1)
        else if (cntv.is_rational()) {
            ::Rational r = std::get<::Rational>(cntv.data);
            if (!r.is_integer()) {
                RuntimeError error("Repeat count must be an integer (rational provided)");
                error.stack_trace = get_stack_trace();
                throw error;
            }
            // numerator may be big, convert cautiously
            try {
                ::BigInt num = r.get_numerator();
                int n = num.to_int();
                if (n == INT_MAX || n == INT_MIN) {
                    RuntimeError error("Repeat count too large");
                    error.stack_trace = get_stack_trace();
                    throw error;
                }
                times = static_cast<long long>(n);
            } catch (const std::exception& e) {
                RuntimeError error(std::string("Invalid repeat count: ") + e.what());
                error.stack_trace = get_stack_trace();
                throw error;
            }
        }
        // Float: accept only when it's an exact integer value
        else if (cntv.is_float()) {
            double d = std::get<double>(cntv.data);
            if (!std::isfinite(d)) {
                RuntimeError error("Repeat count must be a finite number");
                error.stack_trace = get_stack_trace();
                throw error;
            }
            double id = std::floor(d);
            if (d != id) {
                RuntimeError error("Repeat count must be an integer (float provided)");
                error.stack_trace = get_stack_trace();
                throw error;
            }
            if (d > static_cast<double>(LLONG_MAX) || d < static_cast<double>(LLONG_MIN)) {
                RuntimeError error("Repeat count out of range");
                error.stack_trace = get_stack_trace();
                throw error;
            }
            times = static_cast<long long>(d);
        }
        else {
            // Other types (Irrational, Symbolic, String, Array, etc.) are invalid here
            RuntimeError error("Repeat count must be an integer expression (int, bigint, rational or integer-valued float)");
            error.stack_trace = get_stack_trace();
            throw error;
        }

        // Final validation: non-negative
        if (times < 0) {
            RuntimeError error("Repeat count must be non-negative");
            error.stack_trace = get_stack_trace();
            throw error;
        }
        if (times == 0) return LAMINA_NULL;

        try {
            for (long long it = 0; it < times; ++it) {
                bool continue_encountered = false;
                for (auto& stmt: rs->body->statements) {
                    if (continue_encountered) continue;

                    try {
                        execute(stmt);
                    } catch (const BreakException&) {
                        // Exit the repeat entirely
                        return LAMINA_NULL;
                    } catch (const ContinueException&) {
                        // Proceed to next iteration
                        continue_encountered = true;
                    }
                }
            }
        } catch (const ReturnException&) {
            // propagate return
            throw;
        } catch (const std::exception& e) {
            RuntimeError error(std::string("Repeat body execution error: ") + e.what());
            error.stack_trace = get_stack_trace();
            throw error;
        }
    } else if (auto* ret = dynamic_cast<ReturnStmt*>(node.get())) {
        Value val = eval(ret->expr.get());
        throw ReturnException(val);
    } else if ([[maybe_unused]] auto* breakStmt = dynamic_cast<BreakStmt*>(node.get())) {
        throw BreakException();
    } else if ([[maybe_unused]] auto* contStmt = dynamic_cast<ContinueStmt*>(node.get())) {
        throw ContinueException();
    } else if (auto* includeStmt = dynamic_cast<IncludeStmt*>(node.get())) {
        const auto& path = includeStmt->module;
        bool ok_to_load = false;
        if (path.ends_with(".lm")){
            ok_to_load = load_module(path);
        }
        else if
        (   path.ends_with(".dll")
         or path.ends_with(".so")
         or path.ends_with(".dylib")
         or path.ends_with(".a")
        ){
            ok_to_load = load_cpp_module(path);
        }
        else if (
            auto it = register_std_libs.find(path);
            it != register_std_libs.end()
        ){
            set_variable(it->first, it->second);
            ok_to_load = true;
        }
        else {
            L_ERR("Failed to include module '" + path + "'");
        }
        if (!ok_to_load) {
            L_ERR("Failed to include module '" + path + "'");
        }
    } else if (auto* exprstmt = dynamic_cast<ExprStmt*>(node.get())) {
        if (exprstmt->expr) {
            try {
                // std::cerr << "DEBUG: Executing expression statement" << std::endl;
                Value result = eval(exprstmt->expr.get());
                // std::cerr << "DEBUG: Expression result: " << result.to_string() << std::endl;
                return result;
            } catch (const StdLibException& e) {
                throw;
            } catch (const std::exception& e) {
                std::cerr << "ERROR: Exception in expression statement: " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "ERROR: Unknown exception in expression statement" << std::endl;
            }
        } else {
            L_ERR("Empty expression statement");
        }
    } else if (auto* nullstmt = dynamic_cast<NullStmt*>(node.get())) {
        (void) nullstmt;
    } else {
        std::cout << "[Nothing to execute]" << std::endl;
    }
    return LAMINA_NULL;
}


Value Interpreter::eval(const ASTNode* node) {
    if (!node) {
        L_ERR("Attempted to evaluate null expression");
    }
    if (auto* lit = dynamic_cast<const LiteralExpr*>(node)) {
        return eval_LiteralExpr(lit);
    }
    if (auto* id = dynamic_cast<const IdentifierExpr*>(node)) {
        return Interpreter::get_variable(id->name);
    }
    if (auto* var = dynamic_cast<const VarExpr*>(node)) {
        return Interpreter::get_variable(var->name);
    }
    if (auto* bin = dynamic_cast<const BinaryExpr*>(node)) {
        return eval_BinaryExpr(bin);
    }
    if (auto* unary = dynamic_cast<const UnaryExpr*>(node)) {
        return eval_UnaryExpr(unary);
    }// Support function calls
    if (auto* call = dynamic_cast<const CallExpr*>(node)) {
        return eval_CallExpr(call);
    }
    if (auto* s_mem = dynamic_cast<const SetMemberExpr*>(node)) {
        auto left = eval(
            dynamic_cast<const GetMemberExpr*>( s_mem->g_mem.get() ) ->father.get()
        );
        auto val = eval(s_mem->val.get());
        if (left.is_lstruct()) {
            const auto lstruct_ = std::get<std::shared_ptr<lmStruct>>(left.data);

            auto attr_name = dynamic_cast<const GetMemberExpr*>(s_mem->g_mem.get())
                                ->child->name;
            lstruct_->insert(attr_name, val);
            return LAMINA_NULL;
        }
        L_ERR("Type of left can't get it member");
        return LAMINA_NULL;
    }
    if (auto* g_mem = dynamic_cast<const GetMemberExpr*>(node)) {
        auto left = eval(g_mem->father.get());
        if (left.is_lstruct()) {
            const auto lstruct_ = std::get<std::shared_ptr<lmStruct>>(left.data);

            const auto& attr_name = g_mem->child->name;
            auto res = lstruct_->find(attr_name);
            if (res == nullptr) {
                L_ERR("AttrError: struct hasn't attribute named " + attr_name);
                return LAMINA_NULL;
            }
            return res->value;
        }
        L_ERR("Type of left can't get it member");
        return LAMINA_NULL;
    }
    if (auto* g_item = dynamic_cast<const GetItemExpr*>(node)) {
        auto left = eval(g_item->father.get());
        if (g_item->params.empty()) {
            L_ERR("Getitem need one parameter");
            return LAMINA_NULL;
        }
        const auto& subscript = eval(g_item->params[0].get());
        if (left.is_array() and subscript.is_int()) {
            const auto& larray_ = std::get<std::vector<Value>>(left.data);
            Value val;
            try {
                val = larray_.at(std::get<int>(subscript.data));
            }
            catch (const std::out_of_range& e) {
                L_ERR("Index out of range");
                return LAMINA_NULL;
            }
            return val;
        }

        if (left.is_lstruct() and subscript.is_string()) {
            const auto& lstruct_ = std::get<std::shared_ptr<lmStruct>>(left.data);
            const auto& attr_name = std::get<std::string>(subscript.data);
            auto res = lstruct_->find(attr_name);
            if (res == nullptr) {
                L_ERR("AttrError: struct hasn't attribute named " + attr_name);
                return LAMINA_NULL;
            }
            return res->value;
        }
        L_ERR("Type of left is not subscriptable");
        return LAMINA_NULL;
    }
    if (auto* f = dynamic_cast<const LambdaDeclExpr*>(node)) {
        // 克隆 body
        std::unique_ptr<BlockStmt> cloned_body;
        if (f->body) {
            const auto stmt_ptr = f->body->clone_expr().release();
            cloned_body = std::unique_ptr<BlockStmt>(dynamic_cast<BlockStmt*>(stmt_ptr));
        }
        // Make LambdaDeclExpr
        auto func = std::make_shared<LambdaDeclExpr>(
            f->name,
            f->params,
            std::move(cloned_body)
        );
        return Value(func);
    }
    if (auto* ns_g_mem = dynamic_cast<const NameSpaceGetMemberExpr*>(node)) {
        auto left = eval(ns_g_mem->father.get());
        if (left.is_lmModule()) {
            const auto& lmodule_ = std::get<std::shared_ptr<LmModule>>(left.data);
            const auto& attr_name = ns_g_mem->child->name;
            Value val;
            try {
                val = lmodule_->sub_item.at(attr_name);
            }
            catch (const std::out_of_range& e) {
                L_ERR("Attr "+attr_name+" not in module "+lmodule_->module_name);
                return LAMINA_NULL;
            }
            return val;
        }
        L_ERR("Type of left is not a lamina module");
        return LAMINA_NULL;
    }
    if (auto* lm_struct = dynamic_cast<const LambdaStructDeclExpr*>(node)) {
        std::vector<std::pair<std::string, Value>> struct_init_val{};
        for (const auto& [n, e]: lm_struct->init_vec) {
            auto val = eval(e.get());
            struct_init_val.emplace_back(n, val);
        }
        return new_lm_struct(struct_init_val);
    }
    if (auto* arr = dynamic_cast<const ArrayExpr*>(node)) {
        std::vector<Value> elements;
        for (const auto& element: arr->elements) {
            if (element) {
                elements.push_back(eval(element.get()));
            } else {
                std::cerr << "Error: Null element in array literal" << std::endl;
                return {};
            }
        }
        return Value(elements);
    }
    std::cerr << "Error: Unsupported expression type" << std::endl;
    return Value("<type error>");
}

// Load and execute module
bool Interpreter::load_module(const std::string& module_path) {
    const auto file_content = open_lm_file(module_path);
    auto tokens = Lexer::tokenize(file_content);
    const auto parser = std::make_shared<Parser>(tokens);
    auto asts = parser->parse_program();


    // Check if AST generation succeeded
    if (asts.empty()) {
        // print_traceback("<stdin>", lineno);
        // return 1;
        std::cout << "[Nothing to execute]" << std::endl;
        return false;
    }

    // Only support AST of type BlockStmt (block statement)
    const auto block = std::make_unique<BlockStmt>(std::move(asts));
    push_frame(module_path, module_path, 0);   // Add to call stack
    push_scope();                              // Create scope here
    for (auto& stmt: block->statements) {
        try {
            execute(stmt);
        } catch (const std::exception& e) {
            std::cerr << "Error when import other file: " << e.what() << std::endl;
        } catch (...) { // 捕获非标准异常
            std::cerr << "Error: Unknown exception occurred while loading module" << std::endl;
            if (!call_stack.empty()) pop_frame();
            if (!variable_stack.empty()) pop_scope();
            return false;
        }
    }

    auto module_var_table = variable_stack.back();
    pop_scope();
    pop_frame();
    auto module_name = parser->get_module_name();
    if (module_name.empty()) {
        namespace fs = std::filesystem;
        const fs::path path(module_path);
        module_name = path.stem().string();
        std::cout << "[Debug] Path: " << module_path << " → module: " << module_name << std::endl;
    }

    std::cout << "\nProgram execution completed." << std::endl;
    set_variable(module_name,
        Value(
            std::make_shared<LmModule>(
                module_name,
                parser->get_module_version(),
                module_var_table
               )
            )
    );
    return true;
}

bool Interpreter::load_cpp_module(const std::string& module_path) {
    const auto& module = load_cppmodule(module_path);
    namespace fs = std::filesystem;
    const fs::path path(module_path);
    std::string module_name = path.stem().string();
    std::cout << "[Debug] Path: " << module_path << " → module: " << module_name << std::endl;
    
    for (const auto& [key, value] : module) {
        if (key == "lamina_init_module") {
            [[maybe_unused]] auto _ = std::get<std::shared_ptr<LmCppFunction>>(value.data)->function({}); // 初始化函数
        }
        std::cerr << "Debug: checking c++ function " << key << std::endl;
    }
    
    // 拼接properties文件名
    const size_t last_dot_pos = module_path.find_last_of('.');
    std::string properties_file_path;
    if (last_dot_pos != std::string::npos) {
        properties_file_path = module_path.substr(0, last_dot_pos + 1) + "properties"; // +1 保留 '.'
    } else {
        properties_file_path = module_path + "." + "properties";
    }

    const auto version = parse_properties(properties_file_path).at("version");
    set_variable(module_name,
        Value(
            std::make_shared<LmModule>(
                module_name,
                version.empty() ? version : "0.0.0" ,
                module
               )
            )
    );
    return true;
}

// 将Number转为Symbolic(如果可能)
std::shared_ptr<SymbolicExpr> Interpreter::from_number_to_symbolic(const Value& v) {
    std::shared_ptr<SymbolicExpr> expr;
    if (v.is_irrational()) {
        expr = std::get<::Irrational>(v.data).to_symbolic();
    } else if (v.is_rational()) {
        expr = SymbolicExpr::number(std::get<::Rational>(v.data));
    } else if (v.is_bigint()) {
        expr = SymbolicExpr::number(std::get<::BigInt>(v.data));
    } else if (v.is_int()) {
        expr = SymbolicExpr::number(std::get<int>(v.data));
    } else if (v.is_float()) {
        expr = SymbolicExpr::number(::Rational::from_double(std::get<double>(v.data)));
    } else {
        expr = SymbolicExpr::number(0);// 失败，返回0
    }

    return expr;
}

// 添加一个新的错误报告函数，只打印错误但不终止程序
// static void report_error(const std::string& msg) {
//     std::cerr << "Error: " << msg << std::endl;
// }

// 打印当前所有变量
void Interpreter::print_variables() {
    if (variable_stack.empty()) {
        std::cout << "No variables defined." << std::endl;
        return;
    }

    std::cout << "\nCurrent variable list:" << std::endl;
    std::cout << "--------------------" << std::endl;

    // 从最内层作用域开始打印
    bool hasVars = false;
    for (auto it = variable_stack.rbegin(); it != variable_stack.rend(); ++it) {
        const auto& scope = *it;
        for (const auto& [name, value]: scope) {
            std::cout << name << " = " << value.to_string() << std::endl;
            hasVars = true;
        }
    }

    if (!hasVars) {
        std::cout << "No variables defined." << std::endl;
    }
}

// Stack trace management functions
void Interpreter::push_frame(const std::string& function_name, const std::string& file_name, int line_number) {
    Interpreter::call_stack.emplace_back(function_name, file_name, line_number);
}

void Interpreter::pop_frame() {
    if (!Interpreter::call_stack.empty()) {
        Interpreter::call_stack.pop_back();
    }
}

std::vector<StackFrame> Interpreter::get_stack_trace() {
    return call_stack;
}

void Interpreter::print_stack_trace(const RuntimeError& error, const bool use_colors) {
    bool colors_enabled = supports_colors() && use_colors;

    // Use stack trace from error if available, otherwise use current call stack
    const auto& trace = error.stack_trace.empty() ? call_stack : error.stack_trace;

    // Print error header only if we have stack frames to show
    if (!trace.empty()) {
        if (colors_enabled) {
            std::cerr << "\033[1;31mTraceback (most recent call last):\033[0m\n";
        } else {
            std::cerr << "Traceback (most recent call last):\n";
        }

        // Print stack frames
        for (const auto& frame: trace) {
            if (colors_enabled) {
                std::cerr << "  File \"\033[1;34m" << frame.file_name << "\033[0m\", line "
                          << frame.line_number << ", in \033[1;33m" << frame.function_name << "\033[0m\n";
            } else {
                std::cerr << "  File \"" << frame.file_name << "\", line "
                          << frame.line_number << ", in " << frame.function_name << "\n";
            }
        }
    }

    // Print error message
    if (colors_enabled) {
        std::cerr << "\033[1;31mRuntimeError: " << error.message << "\033[0m\n";
    } else {
        std::cerr << "RuntimeError: " << error.message << "\n";
    }
}

bool Interpreter::supports_colors() {
    const bool is_terminal = isatty(fileno(stdout)) != 0;
    if (!is_terminal) {
        return false; // 非终端环境不支持颜色
    }

    // 检查环境变量, 部分终端通过 TERM 标记支持颜色
    const char* term = getenv("TERM");
    if (term != nullptr) {
        const std::string term_str(term);
        // 常见支持颜色的终端类型
        if (term_str == "dumb") { // "dumb" 终端不支持颜色
            return false;
        }
        // 其他如 xterm、linux、vt100 等通常支持颜色
        return true;
    }

    return true;
}

void Interpreter::print_error(const std::string& message, const bool use_colors) {
    bool colors_enabled = supports_colors() && use_colors;

    if (colors_enabled) {
        std::cerr << "\033[1;31mError: " << message << "\033[0m\n";
    } else {
        std::cerr << "Error: " << message << "\n";
    }
}

void Interpreter::print_warning(const std::string& message, bool use_colors) {
    bool colors_enabled = supports_colors() && use_colors;

    if (colors_enabled) {
        std::cerr << "\033[1;33mWarning: " << message << "\033[0m\n";
    } else {
        std::cerr << "Warning: " << message << "\n";
    }
}
