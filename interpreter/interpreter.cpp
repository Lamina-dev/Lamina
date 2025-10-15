#include "interpreter.hpp"
#include "../extensions/standard/lstruct.hpp"
#include "bigint.hpp"
#include "lamina.hpp"
#include "lexer.hpp"
#include "module_loader.hpp"
#include "parser.hpp"
#include "compute_backend/backend_interface.hpp"
#include "compute_backend/cpu/cpu_backend.hpp"
#ifdef ENABLE_VULKAN
#include "compute_backend/vulkan/vulkan_backend.hpp"
#endif
#ifdef ENABLE_CUDA
#include "compute_backend/cuda/cuda_backend.hpp"
#endif

#include <cmath>
#include <cstdlib>// For std::exit
#include <cstring>// For strcmp
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
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


// 这些异常类已经移到了 interpreter.hpp

// Scope stack operations
void Interpreter::push_scope() {
    variable_stack.emplace_back();
}

void Interpreter::add_function(const std::string& name, FuncDefStmt* func) {
    functions[name] = func;
}

void Interpreter::save_repl_ast(std::unique_ptr<ASTNode> ast) {
    repl_asts.push_back(std::move(ast));
}

void Interpreter::pop_scope() {
    if (variable_stack.size() > 1) variable_stack.pop_back();
}

Value Interpreter::get_variable(const std::string& name) const {
    for (auto it = variable_stack.rbegin(); it != variable_stack.rend(); ++it) {
        auto found = it->find(name);
        if (found != it->end()) return found->second;
    }

    // 如果变量找不到，检查是否是函数名
    auto func_it = functions.find(name);
    if (func_it != functions.end()) {
        // 返回一个函数类型的值（这里我们需要扩展Value类型）
        // 暂时返回一个特殊的字符串值表示函数
        return Value("__function_" + name);
    }

    RuntimeError error("Undefined variable '" + name + "'");
    error.stack_trace = get_stack_trace();
    throw error;
    return Value();// Unreachable, but suppress compiler warning
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

void Interpreter::execute(const std::unique_ptr<Statement>& node) {
    if (!node) return;

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
            error_and_exit("Null expression in define statement for '" + d->name + "'");
        }
        Value val = eval(d->value.get());
        // 删除递归限制
        /*if (d->name == "MAX_RECURSION_DEPTH" && val.is_int()) {
            int new_depth = std::get<int>(val.data);
            if (new_depth > 0 && new_depth <= 10000) {
                max_recursion_depth = new_depth;
                std::cout << "Recursion depth limit set to: " << new_depth << std::endl;
            } else {
                error_and_exit("Invalid recursion depth value: " + std::to_string(new_depth) + " (must be between 1 and 10000)");
            }
        } else {
            // 原：std::cerr << "Error: Unknown define constant: " << d->name << std::endl;
            // 已替换
        }*/
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
                    error_and_exit("Invalid BigInt string '" + std::get<std::string>(val.data) + "' in declaration of " + bi->name);
                }
            } else {
                // 默认初始化为0
                error_and_exit("Cannot convert " + val.to_string() + " to BigInt in declaration of " + bi->name);
            }
        } else {
            set_variable(bi->name, Value(::BigInt(0)));
        }
    } else if (auto* a = dynamic_cast<AssignStmt*>(node.get())) {
        if (!a->expr) {
            error_and_exit("Null expression in assignment to '" + a->name + "'");
        }
        Value val = eval(a->expr.get());
        set_variable(a->name, val);
    } else if (auto* a = dynamic_cast<StructDeclStmt*>(node.get())) {
        std::vector<std::pair<std::string, Value>> struct_init_val{};
        for (const auto& [n, e]: a->init_vec) {
            auto val = eval(e.get());
            struct_init_val.emplace_back(n, val);
        }
        set_variable(a->name, new_lstruct(struct_init_val));
    } else if (auto* ifs = dynamic_cast<IfStmt*>(node.get())) {
        if (!ifs->condition) {
            error_and_exit("Null condition in if statement");
        }
        Value cond = eval(ifs->condition.get());
        bool cond_true = cond.as_bool();
        if (cond_true) {
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
                        return;
                    } catch (const ContinueException&) {
                        // 跳到下一次迭代
                        continue_encountered = true;
                    }
                }
            }
        } catch (const BreakException&) {
            // 捕获可能从嵌套块冒泡上来的break异常
            return;
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
        functions[func->name] = func;
    } else if (auto* block = dynamic_cast<BlockStmt*>(node.get())) {
        for (auto& stmt: block->statements) execute(stmt);
    } else if (auto* ret = dynamic_cast<ReturnStmt*>(node.get())) {
        Value val = eval(ret->expr.get());
        throw ReturnException(val);
    } else if (auto* breakStmt = dynamic_cast<BreakStmt*>(node.get())) {
        (void) breakStmt;// 避免未使用变量警告
        throw BreakException();
    } else if (auto* contStmt = dynamic_cast<ContinueStmt*>(node.get())) {
        (void) contStmt;// 避免未使用变量警告
        throw ContinueException();
    } else if (auto* includeStmt = dynamic_cast<IncludeStmt*>(node.get())) {
        if (!load_module(includeStmt->module)) {
            error_and_exit("Failed to include module '" + includeStmt->module + "'");
        }
    } else if (auto* exprstmt = dynamic_cast<ExprStmt*>(node.get())) {
        if (exprstmt->expr) {
            try {
                // std::cerr << "DEBUG: Executing expression statement" << std::endl;
                Value result = eval(exprstmt->expr.get());
                // std::cerr << "DEBUG: Expression result: " << result.to_string() << std::endl;
            } catch (const StdLibException& e){
                throw StdLibException(e.what());
            } catch (const std::exception& e) {
                std::cerr << "ERROR: Exception in expression statement: " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "ERROR: Unknown exception in expression statement" << std::endl;
            }
        } else {
            error_and_exit("Empty expression statement");
        }
    } else if (auto* nullstmt = dynamic_cast<NullStmt*>(node.get())) {
        (void) nullstmt;
    } else if (auto* backend_block = dynamic_cast<BackendBlockStmt*>(node.get())) {
        // Handle @backend { } blocks
        auto& backend_mgr = BackendManager::instance();

        backend_mgr.push_default_backend(backend_block->backend_name);

        try {
            for (auto& stmt: backend_block->body->statements) {
                execute(stmt);
            }
            backend_mgr.pop_default_backend();
        } catch (...) {
            backend_mgr.pop_default_backend();
            throw;
        }
    }
}


Value Interpreter::eval(const ASTNode* node) {
    if (!node) {
        error_and_exit("Attempted to evaluate null expression");
    }
    if (auto* lit = dynamic_cast<const LiteralExpr*>(node)) {
        return eval_LiteralExpr(lit);
    }
    if (auto* id = dynamic_cast<const IdentifierExpr*>(node)) {
        return get_variable(id->name);
    }
    if (auto* var = dynamic_cast<const VarExpr*>(node)) {
        return get_variable(var->name);
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
    if (auto* arr = dynamic_cast<const ArrayExpr*>(node)) {
        std::vector<Value> elements;
        for (const auto& element: arr->elements) {
            if (element) {
                elements.push_back(eval(element.get()));
            } else {
                std::cerr << "Error: Null element in array literal" << std::endl;
                return Value();
            }
        }
        return Value(elements);
    }
    if (auto* idx = dynamic_cast<const IndexExpr*>(node)) {
        // Evaluate the array expression
        Value array_val = eval(idx->array.get());
        // Evaluate the index expression
        Value index_val = eval(idx->index.get());

        // Index must be an integer
        if (!index_val.is_int()) {
            RuntimeError error("Array index must be an integer, got " + index_val.to_string());
            error.stack_trace = get_stack_trace();
            throw error;
        }

        int index = std::get<int>(index_val.data);

        // Handle array indexing
        if (array_val.is_array()) {
            const auto& arr = std::get<std::vector<Value>>(array_val.data);
            if (index < 0 || index >= static_cast<int>(arr.size())) {
                RuntimeError error("Array index " + std::to_string(index) + " out of bounds (size: " + std::to_string(arr.size()) + ")");
                error.stack_trace = get_stack_trace();
                throw error;
            }
            return arr[index];
        } else if (array_val.is_matrix()) {
            // Handle matrix indexing - returns a row (as array)
            const auto& mat = std::get<std::vector<std::vector<Value>>>(array_val.data);
            if (index < 0 || index >= static_cast<int>(mat.size())) {
                RuntimeError error("Matrix row index " + std::to_string(index) + " out of bounds (rows: " + std::to_string(mat.size()) + ")");
                error.stack_trace = get_stack_trace();
                throw error;
            }
            return Value(mat[index]);
        } else {
            RuntimeError error("Cannot index non-array/non-matrix value: " + array_val.to_string());
            error.stack_trace = get_stack_trace();
            throw error;
        }
    }
    std::cerr << "Error: Unsupported expression type" << std::endl;
    return Value("<type error>");
}

// Load and execute module
bool Interpreter::load_module(const std::string& module_name) {
    // Check if already loaded to avoid circular imports
    if (loaded_modules.contains(module_name)) {
        return true;// Already loaded
    }
    // 立即插入，防止递归 include
    loaded_modules.insert(module_name);

    bool is_shared_lib = false;
    std::string clean_name = module_name;
    if (clean_name.rfind("lib", 0) == 0) {
        clean_name = clean_name.substr(3);
    }

// 平台相关动态库扩展名
#if defined(_WIN32)
    const std::string lib_ext = ".dll";
#elif defined(__APPLE__)
    const std::string lib_ext = ".dylib";
#else
    const std::string lib_ext = ".so";
#endif

    // Handle built-in hidden library "splash"
    if (module_name == "splash") {
        print_logo();
        return true;
    }
    // Handle built-in hidden library "them" (credits)
    if (module_name == "them") {
        print_them();
        return true;
    }

    // Build possible file paths
    std::string filename = module_name;
    bool has_lm = filename.find(".lm") != std::string::npos;
    bool has_lib = filename.find(lib_ext) != std::string::npos;

    // Try different paths: current directory, examples directory, etc.
    const std::vector<std::string> search_paths = {"", ".", "include", "extensions"};
    std::string full_path;

    // 如果指定了动态库扩展名，直接尝试加载动态库
    if (has_lib) {
        std::vector<std::string> tried_paths;
        for (const auto& path: search_paths) {
            full_path = (std::filesystem::path(path) / filename).string();
            std::ifstream testfile(full_path, std::ios::binary);
            tried_paths.push_back(full_path);
            if (testfile) {
                testfile.close();
                auto loader = std::make_unique<ModuleLoader>(full_path);
                if (loader->isLoaded()) {
                    if (loader->registerToInterpreter(*this)) {
                        // 保存模块加载器实例
                        module_loaders.push_back(std::move(loader));
                        return true;
                    }
                    std::cerr << "Failed to register module: " << full_path << std::endl;
                } else {
                    std::cerr << "Failed to load shared library: " << full_path << std::endl;
                }
            }
        }
        // 如果指定了.dll/.so/.dylib但没找到，直接返回失败
        if (tried_paths.empty()) {
            std::cerr << "Error: Cannot load module '" << module_name << "'\n";
        } else {
            std::cerr << "Error: Cannot load module '" << module_name << "'\nTried paths:\n";
            for (const auto& p : tried_paths) {
                std::cerr << "  - " << p << "\n";
            }
        }
        return false;
    }

    // 如果没有指定扩展名，默认添加.lm
    if (!has_lm) {
        filename += ".lm";
    }

    std::ifstream file;
    for (const auto& path: search_paths) {
        full_path = (std::filesystem::path(path) / filename).string();
        file.open(full_path);
        if (file) break;
    }

    // 如果没找到脚本文件，尝试查找动态库（lib前缀和无前缀，自动适配扩展名）
    if (!file) {
        std::vector<std::string> lib_names = {
                "lib" + clean_name + lib_ext,
                clean_name + lib_ext};
        is_shared_lib = true;
        bool found = false;
        for (const auto& libfile: lib_names) {
            for (const auto& path: search_paths) {
                full_path = (std::filesystem::path(path) / libfile).string();
                std::ifstream testfile(full_path, std::ios::binary);
                if (testfile) {
                    testfile.close();
                    auto loader = std::make_unique<ModuleLoader>(full_path);
                    if (loader->isLoaded()) {
                        if (loader->registerToInterpreter(*this)) {
                            // 保存模块加载器实例
                            module_loaders.push_back(std::move(loader));
                            found = true;
                            break;
                        }
                        std::cerr << "Failed to register module: " << full_path << std::endl;
                    } else {
                        std::cerr << "Failed to load shared library: " << full_path << std::endl;
                    }
                }
            }
            if (found) break;
        }
        if (!found) {
            std::cerr << "Error: Cannot load module '" << module_name << "'\n";
            std::cerr << "  Searched in: ";
            for (const auto& libfile: lib_names) {
                for (const auto& path: search_paths) {
                    std::cerr << (std::filesystem::path(path) / libfile).string() << " ";
                }
            }
            std::cerr << "\n";
            return false;
        }
        return true;
    }


    // Read file into string
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string source = buffer.str();
    file.close();


    // Lexical and syntax analysis
    auto tokens = Lexer::tokenize(source);
    auto ast = Parser::parse(tokens);

    if (!ast) {
        std::cerr << "Error: Failed to parse module '" << module_name << "'" << std::endl;
        loaded_modules.erase(module_name);// Remove from loaded modules on failure
        return false;
    }// Execute module code (should be a block statement)
    auto* block = dynamic_cast<BlockStmt*>(ast.get());
    if (block) {
        // Execute module code in the current scope (not a new scope)
        // This allows variables and functions to be accessible after inclusion
        try {
            for (auto& stmt: block->statements) {
                execute(stmt);
            }
        } catch (const std::exception& e) {
            std::cerr << "Error: Exception while executing module '" << module_name << "': " << e.what() << std::endl;
            loaded_modules.erase(module_name);// 失败时移除，防止死锁
            return false;
        }

        // Store the AST to keep function pointers valid
        loaded_module_asts.push_back(std::move(ast));
        return true;
    }
    std::cerr << "Error: Module '" << module_name << "' does not contain valid code" << std::endl;
    return false;
}

// 使用函数内的静态变量来避免DLL导出/导入问题
static std::vector<Interpreter::EntryFunction>& get_entry_functions() {
    static std::vector<Interpreter::EntryFunction> entry_functions;
    return entry_functions;
}

void Interpreter::register_entry(EntryFunction func) {
    get_entry_functions().push_back(func);
}

// Register builtin mathematical functions
void Interpreter::register_builtin_functions() {
    for (auto entry: get_entry_functions()) {
        entry(*this);
    }

    // Register compute backends
    try {
        auto& backend_mgr = BackendManager::instance();

        // Register CPU backend (always available)
        auto cpu_backend = std::make_shared<CPUBackend>();
        backend_mgr.register_backend(cpu_backend);

#ifdef ENABLE_VULKAN
        // Register Vulkan backend if compiled with support
        auto vulkan_backend = std::make_shared<VulkanBackend>();
        if (vulkan_backend->is_available()) {
            backend_mgr.register_backend(vulkan_backend);
        }
#endif

#ifdef ENABLE_CUDA
        // Register CUDA backend if compiled with support
        auto cuda_backend = std::make_shared<CudaBackend>();
        if (cuda_backend->is_available()) {
            backend_mgr.register_backend(cuda_backend);
        }
#endif
    } catch (const std::exception& e) {
        std::cerr << "Warning: Failed to register compute backends: " << e.what() << std::endl;
    }
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


// 在文件顶端添加错误处理函数
LAMINA_EXPORT void error_and_exit(const std::string& msg) {
    std::cerr << "Error: " << msg << std::endl;
    std::exit(EXIT_FAILURE);
}

// 添加一个新的错误报告函数，只打印错误但不终止程序
// static void report_error(const std::string& msg) {
//     std::cerr << "Error: " << msg << std::endl;
// }

// 打印当前所有变量
void Interpreter::printVariables() const {
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

    // 打印函数列表
    if (!functions.empty()) {
        std::cout << "\nDefined functions:" << std::endl;
        std::cout << "------------------" << std::endl;
        for (const auto& [name, _]: functions) {
            std::cout << name << "()" << std::endl;
        }
    }
}

// Stack trace management functions
void Interpreter::push_frame(const std::string& function_name, const std::string& file_name, int line_number) {
    call_stack.emplace_back(function_name, file_name, line_number);
}

void Interpreter::pop_frame() {
    if (!call_stack.empty()) {
        call_stack.pop_back();
    }
}

std::vector<StackFrame> Interpreter::get_stack_trace() const {
    return call_stack;
}

void Interpreter::print_stack_trace(const RuntimeError& error, bool use_colors) const {
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

void Interpreter::print_error(const std::string& message, bool use_colors) {
    bool colors_enabled = supports_colors() && use_colors;

    if (colors_enabled) {
        std::cerr << "\033[1;31mError: " << message << "\033[0m\n";
    } else {
        std::cerr << "Error: " << message << "\n";
    }
}

// Call module function implementation
Value Interpreter::call_module_function(const std::string& func_name, const std::vector<Value>& args) {
    // Try each loaded module
    for (auto& loader: module_loaders) {
        if (loader && loader->isLoaded()) {
            // Check if this module has the function (by namespace)
            size_t dot_pos = func_name.find(".");
            if (dot_pos != std::string::npos) {
                std::string ns = func_name.substr(0, dot_pos);
                const auto* module_info = loader->getModuleInfo();
                if (module_info && std::string(module_info->namespace_name) == ns) {
                    return loader->callModuleFunction(func_name, args);
                }
            }
        }
    }

    std::cerr << "ERROR: Module function '" << func_name << "' not found in any loaded module" << std::endl;
    return Value();// null value
}

void Interpreter::print_warning(const std::string& message, bool use_colors) {
    bool colors_enabled = supports_colors() && use_colors;

    if (colors_enabled) {
        std::cerr << "\033[1;33mWarning: " << message << "\033[0m\n";
    } else {
        std::cerr << "Warning: " << message << "\n";
    }
}
