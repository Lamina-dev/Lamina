//
// Created by meian on 2026/4/8.
//

#include "include/lmx.h"

#include "compiler/compiler.hpp"
#include "compiler/ast/ast_printer.hpp"
#include "compiler/hir/type_checker.hpp"
#include "compiler/parser.hpp"
#include "compiler/lexer.hpp"
#include "runtime/vm.hpp"
#include "runtime/object/lsr_ExprObj.hpp"
#include "runtime/object/string.hpp"
#include "runtime/object/adt.hpp"

#include <cmath>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <fstream>
#include <limits>
#include <utility>

#include "compiler/mir/mir_printer.hpp"
#include "lmmc/numeric.h"

LmState global_state;

namespace {

using lmx::runtime::ExprObj;
using lmx::runtime::String;

bool debug_dump_enabled() noexcept {
    const char* value = std::getenv("LMX_DEBUG_DUMP");
    return value && value[0] != '\0' && value[0] != '0';
}

std::string cas_error_text(const lamina::CasError& error) {
    std::string result = "CasError(";
    result += lamina::lsr::error_name(error);
    if (!error.operation.empty()) {
        result += " in ";
        result += error.operation;
    }
    if (!error.message.empty()) {
        result += ": ";
        result += error.message;
    }
    result += ")";
    return result;
}

ExprObj* expr_error(std::string message) {
    return new ExprObj(std::move(message));
}

ExprObj* expr_from_result(const lamina::lsr::ExprResult& result) {
    if (!result) return expr_error(cas_error_text(result.error()));
    return new ExprObj(result.value());
}

const lamina::lsr::ExprPtr* checked_expr(ExprObj* expr, std::string& error) {
    if (!expr) {
        error = "CasError(InvalidArgument: null expr)";
        return nullptr;
    }
    if (!expr->ok()) {
        error = expr->error();
        return nullptr;
    }
    return &expr->expr();
}

double lmmc_real_result(lmmc_status_t status, lmmc_real_t value) {
    if (status != LMMC_STATUS_OK) return std::numeric_limits<double>::quiet_NaN();
    return value;
}

double expr_to_real(ExprObj* expr) {
    std::string error;
    const auto* value = checked_expr(expr, error);
    if (!value) return std::numeric_limits<double>::quiet_NaN();
    const auto evaluated = lamina::lsr::evalf(**value);
    if (!evaluated) return std::numeric_limits<double>::quiet_NaN();
    return evaluated.value().value;
}

} // namespace

extern "C" LM_API int lmx_printf(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    const int result = vprintf(fmt ? fmt : "", args);
    va_end(args);
    return result;
}

extern "C" LM_API ExprObj* cas_sym(const char* name) {
    return expr_from_result(lamina::lsr::sym(name ? name : ""));
}

extern "C" LM_API ExprObj* cas_parse(const char* source) {
    return expr_from_result(lamina::lsr::parse_expr(source ? source : ""));
}

extern "C" LM_API ExprObj* cas_simplify(ExprObj* expr) {
    std::string error;
    const auto* value = checked_expr(expr, error);
    if (!value) return expr_error(std::move(error));
    return expr_from_result(lamina::lsr::simplify(*value));
}

extern "C" LM_API ExprObj* cas_expand(ExprObj* expr) {
    std::string error;
    const auto* value = checked_expr(expr, error);
    if (!value) return expr_error(std::move(error));
    return expr_from_result(lamina::lsr::expand(*value));
}

extern "C" LM_API ExprObj* cas_diff(ExprObj* expr, const char* variable) {
    std::string error;
    const auto* value = checked_expr(expr, error);
    if (!value) return expr_error(std::move(error));
    return expr_from_result(lamina::lsr::differentiate(*value, variable ? variable : ""));
}

extern "C" LM_API ExprObj* cas_substitute(ExprObj* expr, const char* variable, ExprObj* replacement) {
    std::string error;
    const auto* value = checked_expr(expr, error);
    if (!value) return expr_error(std::move(error));
    const auto* repl = checked_expr(replacement, error);
    if (!repl) return expr_error(std::move(error));
    return expr_from_result(lamina::lsr::substitute(*value, variable ? variable : "", *repl));
}

extern "C" LM_API double cas_evalf(ExprObj* expr) {
    std::string error;
    const auto* value = checked_expr(expr, error);
    if (!value) return std::numeric_limits<double>::quiet_NaN();
    const auto result = lamina::lsr::evalf(**value);
    if (!result) return std::numeric_limits<double>::quiet_NaN();
    return result.value().value;
}

extern "C" LM_API bool cas_is_ok(ExprObj* expr) {
    return expr && expr->ok();
}

extern "C" LM_API String* cas_to_text(ExprObj* expr) {
    if (!expr) return new String("CasError(InvalidArgument: null expr)");
    return new String(expr->to_string());
}

extern "C" LM_API String* cas_error(ExprObj* expr) {
    if (!expr) return new String("CasError(InvalidArgument: null expr)");
    if (expr->ok()) return new String("");
    return new String(expr->error());
}

extern "C" LM_API double lmmc_num_hypot(ExprObj* lhs, ExprObj* rhs) {
    const double x = expr_to_real(lhs);
    const double y = expr_to_real(rhs);
    if (!std::isfinite(x) || !std::isfinite(y)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    lmmc_real_t out = 0.0;
    const auto status = lmmc_hypot(x, y, &out);
    return lmmc_real_result(status, out);
}

extern "C" LM_API double lmmc_num_log2(ExprObj* expr) {
    const double x = expr_to_real(expr);
    if (!std::isfinite(x)) return std::numeric_limits<double>::quiet_NaN();
    lmmc_real_t out = 0.0;
    const auto status = lmmc_log2(x, &out);
    return lmmc_real_result(status, out);
}

extern "C" LM_API double lmmc_num_exp2(ExprObj* expr) {
    const double x = expr_to_real(expr);
    if (!std::isfinite(x)) return std::numeric_limits<double>::quiet_NaN();
    lmmc_real_t out = 0.0;
    const auto status = lmmc_exp2(x, &out);
    return lmmc_real_result(status, out);
}

extern "C" LM_API int printf(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    const int result = vprintf(fmt, args);
    va_end(args);
    return result;
}

LM_API LmState* lmx_newState() {
    auto* node = static_cast<LmLinkedNode *>(malloc(sizeof(LmLinkedNode)));
    memset(node, 0, sizeof(LmLinkedNode));
    global_state = LmState {.n = node, .vm = nullptr};
    return &global_state;
}
LM_API void lmx_deleteState(const LmState* state) {
    const LmLinkedNode* node = state->n;
    while (node != nullptr) {
        if (node->ptr != nullptr) free(node->ptr);
        const auto last = node->last;
        free((void*)node);
        node = last;
    }
    delete reinterpret_cast<lmx::runtime::LaminaVM*>(state->vm);
}
static LmLinkedNode* newLickedNode(LmLinkedNode* old) {
    auto* node = static_cast<LmLinkedNode *>(malloc(sizeof(LmLinkedNode)));
    node->last = old;
    return node;
}
static void lmx_state_addNode(LmState* state, void* ptr) {
    state->n = newLickedNode(state->n);
    state->n->ptr = ptr;
}

static LmModule* lmx_newCodeModule(LmState* state, std::vector<uint8_t>&& binary) {
    const auto storage = malloc(sizeof(lmx::runtime::CodeModule));
    if (storage == nullptr) return nullptr;
    new (storage) lmx::runtime::CodeModule(std::move(binary));
    lmx_state_addNode(state, storage);
    return static_cast<LmModule*>(storage);
}

void lmx_printASTFromString(LmState *state, FILE *file, const char *code, const char* name) {
    lmx::Compiler compiler;
    lmx::CompileResult result;
    if (!compiler.compile_source(code, name, result, lmx::CompileStage::Semantic)) return;

    const auto ast_str = lmx::AstPrinter::print(*result.module);
    if (fwrite(ast_str.c_str(), 1, ast_str.length(), file) != ast_str.length()) {
        fprintf(stderr, "Error writing AST to file\n");
    }
}

void lmx_printASTFromFile(LmState *state, FILE *file, const char *name) {
    lmx::Compiler compiler;
    lmx::CompileResult result;
    if (!compiler.compile_file(name, result, lmx::CompileStage::Semantic)) return;
    auto str = lmx::AstPrinter::print(*result.module);
    if (fwrite(str.c_str(), 1, str.length(), file) != str.length()) {
        fprintf(stderr, "Error writing AST to file\n");
    }
}

void lmx_printMIRFromString(LmState *state, FILE *file, const char *code, const char *name) {
    lmx::Compiler compiler;
    lmx::CompileResult result;
    if (!compiler.compile_source(code, name, result, lmx::CompileStage::Mir)) return;
    const auto mir_str = lmx::mir::MirPrinter::print(*result.mir);

    if (fwrite(mir_str.c_str(), 1, mir_str.length(), file) != mir_str.length()) {
        fprintf(stderr, "Error writing MIR to file\n");
    }
}
void lmx_printMIRFromFile(LmState *state, FILE *file, const char *name) {
    lmx::Compiler compiler;
    lmx::CompileResult result;
    if (!compiler.compile_file(name, result, lmx::CompileStage::Mir)) return;
    const auto mir_str = lmx::mir::MirPrinter::print(*result.mir);

    if (fwrite(mir_str.c_str(), 1, mir_str.length(), file) != mir_str.length()) {
        fprintf(stderr, "Error writing MIR to file\n");
    }
}

LaminaVM* lmx_newLaminaVM(LmState* state, int argc, char** argv) {
    auto* vm = new lmx::runtime::LaminaVM(argc, argv);
    if (state->vm) delete reinterpret_cast<lmx::runtime::LaminaVM*>(state->vm);
    state->vm = reinterpret_cast<LaminaVM*>(vm);
    return state->vm;
}

bool lmx_moduleToFile(LmState *state, LmModule *module, const char *name) {
    const std::filesystem::path path = name;
    std::filesystem::create_directories(path.parent_path());
    std::ofstream ofs(path.string() + lmx::file_suffix_binary, std::ios::binary | std::ios::trunc);
    const auto* mod = reinterpret_cast<lmx::runtime::CodeModule*>(module);
    ofs.write(
        reinterpret_cast<const char*>(mod->raw_data.data()),
        static_cast<std::streamsize>(mod->raw_data.size())
        );
    if (!ofs) return false;
    ofs.close();
    return true;
}

LmModule *lmx_doString(LmState *state, const char *code, const char* name) {
    lmx::Compiler compiler;
    lmx::CompileResult result;
    if (!compiler.compile_source(code, name, result)) return nullptr;
    return lmx_newCodeModule(state, std::move(result.binary));
}
LmModule *lmx_doFile(LmState *state, const char* name, bool is_main_module) {
    lmx::Compiler compiler;
    lmx::CompileResult result;
    if (!compiler.compile_file(name, result, lmx::CompileStage::Binary,
                               is_main_module)) return nullptr;
#if !NDEBUG
    if (debug_dump_enabled()) {
        std::cout << lmx::AstPrinter::print(*result.module) << std::endl;
    }
#endif
#if !NDEBUG
    if (debug_dump_enabled()) {
        std::cout << lmx::mir::MirPrinter::print(*result.mir) << std::endl;
    }
#endif

    return lmx_newCodeModule(state, std::move(result.binary));
}

int lmx_vmRunModule(LmState* state, LaminaVM* vm, LmModule* module) {
    if (module == nullptr) return 1;
    // std::cout << mod.disassemble() << std::endl;
    return
    reinterpret_cast<lmx::runtime::LaminaVM*>(vm)
    ->
    run(reinterpret_cast<lmx::runtime::CodeModule*>(module));
}

void lmx_vmEval(LmState *state, LaminaVM *vm, LmValue *result, const char *code) {
    std::string c = code;
    auto tks = lmx::Lexer(c).tokenize(c);
    auto node = lmx::Parser(tks).parse_stmt();
    lmx::hir::TypeCkContext().check_stmt(node);
}
