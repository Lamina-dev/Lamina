//
// Created by meian on 2026/4/8.
//

#include "include/lmx.h"
#include "include/lmx_expr.h"

#include "compiler/compiler.hpp"
#include "compiler/ast/ast_printer.hpp"
#include "compiler/hir/type_checker.hpp"
#include "compiler/parser.hpp"
#include "compiler/lexer.hpp"
#include "runtime/vm.hpp"
#include "runtime/object/lsr_expr_obj.hpp"
#include "runtime/object/StringObj.hpp"
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
using lmx::runtime::StringObj;
using lmx::runtime::AdtObj;

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

const lamina::lsr::ExprPtr* checked_expr(ExprObj* expr, std::string& error);

lamina::lsr::ExprResult invalid_expr_operation(const std::string& message,
                                               const char* operation) {
    return lamina::lsr::ExprResult::failure(
        lamina::CasErrc::InvalidArgument, message, operation);
}

bool collect_expr_arguments(va_list& args, const LmInt count,
                            std::vector<lamina::lsr::ExprPtr>& values,
                            std::string& error) {
    if (count < 0 || count > 65535) {
        error = "CasError(InvalidArgument: invalid Expr argument count)";
        return false;
    }
    values.reserve(static_cast<std::size_t>(count));
    for (LmInt i = 0; i < count; ++i) {
        auto* object = va_arg(args, ExprObj*);
        const auto* value = checked_expr(object, error);
        if (!value) return false;
        values.push_back(*value);
    }
    return true;
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

AdtObj* real_result_ok(const double value) {
    std::vector<lmx::runtime::Value> fields;
    fields.emplace_back(value);
    return new AdtObj("Result", "Ok", std::move(fields));
}

AdtObj* real_result_error(std::string error) {
    std::vector<lmx::runtime::Value> fields;
    fields.emplace_back(static_cast<lmx::runtime::Object*>(
        new StringObj(std::move(error))));
    return new AdtObj("Result", "Err", std::move(fields));
}

AdtObj* lmmc_real_result(const char* operation, const lmmc_status_t status,
                         const lmmc_real_t value) {
    if (status == LMMC_STATUS_OK) return real_result_ok(value);
    std::string error = operation ? operation : "LMMC";
    error += ": ";
    error += lmmc_status_string(status);
    return real_result_error(std::move(error));
}

bool expr_to_real(ExprObj* expr, double& result, std::string& error) {
    const auto* value = checked_expr(expr, error);
    if (!value) return false;
    const auto evaluated = lamina::lsr::evalf(**value);
    if (!evaluated) {
        error = cas_error_text(evaluated.error());
        return false;
    }
    result = evaluated.value().value;
    return true;
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

extern "C" LM_API ExprObj* cas_expr_imaginary() {
    return expr_from_result(lamina::lsr::imaginary_unit());
}

extern "C" LM_API ExprObj* cas_expr_integer(const LmInt value) {
    return expr_from_result(lamina::lsr::integer(value));
}

extern "C" LM_API ExprObj* cas_expr_rational(const LmInt numerator,
                                               const LmInt denominator) {
    if (denominator == 0) return expr_error("CasError(DivisionByZero: rational denominator is zero)");
    return expr_from_result(lamina::lsr::rational(Rational(
        BigInt(std::to_string(numerator)), BigInt(std::to_string(denominator)))));
}

extern "C" LM_API ExprObj* cas_expr_value(const lmx::runtime::Value* value) {
    if (!value) return expr_error("CasError(InvalidArgument: null Lamina value)");
    switch (value->kind) {
    case lmx::runtime::ValueKind::Int:
        return expr_from_result(lamina::lsr::integer(value->int_val));
    case lmx::runtime::ValueKind::Fraction:
        return expr_from_result(lamina::lsr::rational(Rational(
            value->frac_val.num, value->frac_val.den)));
    case lmx::runtime::ValueKind::Real:
        return expr_from_result(lamina::lsr::approx_real(value->real_val));
    case lmx::runtime::ValueKind::Expr: {
        std::string error;
        const auto* expression = checked_expr(
            reinterpret_cast<ExprObj*>(value->obj), error);
        return expression ? new ExprObj(*expression) : expr_error(std::move(error));
    }
    default:
        return expr_from_result(invalid_expr_operation(
            "Lamina value cannot be promoted to Expr", "runtime.expr_value"));
    }
}

extern "C" LM_API ExprObj* cas_expr_unary(const LmInt operation,
                                            ExprObj* operand) {
    std::string error;
    const auto* value = checked_expr(operand, error);
    if (!value) return expr_error(std::move(error));
    switch (operation) {
    case LMX_EXPR_NEG:
        return expr_from_result(lamina::lsr::neg(*value));
    case LMX_EXPR_NOT:
        return expr_from_result(lamina::lsr::logical_not(*value));
    default:
        return expr_from_result(invalid_expr_operation(
            "unknown unary Expr operation", "runtime.expr_unary"));
    }
}

extern "C" LM_API ExprObj* cas_expr_binary(const LmInt operation,
                                             ExprObj* lhs, ExprObj* rhs) {
    std::string error;
    const auto* left = checked_expr(lhs, error);
    if (!left) return expr_error(std::move(error));
    const auto* right = checked_expr(rhs, error);
    if (!right) return expr_error(std::move(error));
    switch (operation) {
    case LMX_EXPR_ADD: return expr_from_result(lamina::lsr::add(*left, *right));
    case LMX_EXPR_SUB: return expr_from_result(lamina::lsr::sub(*left, *right));
    case LMX_EXPR_MUL: return expr_from_result(lamina::lsr::mul(*left, *right));
    case LMX_EXPR_DIV: return expr_from_result(lamina::lsr::div(*left, *right));
    case LMX_EXPR_POW: return expr_from_result(lamina::lsr::pow(*left, *right));
    case LMX_EXPR_EQ: return expr_from_result(lamina::lsr::relation(*left, *right, lamina::RelationOp::EQ));
    case LMX_EXPR_NE: return expr_from_result(lamina::lsr::relation(*left, *right, lamina::RelationOp::NEQ));
    case LMX_EXPR_GT: return expr_from_result(lamina::lsr::relation(*left, *right, lamina::RelationOp::GT));
    case LMX_EXPR_GE: return expr_from_result(lamina::lsr::relation(*left, *right, lamina::RelationOp::GEQ));
    case LMX_EXPR_LT: return expr_from_result(lamina::lsr::relation(*left, *right, lamina::RelationOp::LT));
    case LMX_EXPR_LE: return expr_from_result(lamina::lsr::relation(*left, *right, lamina::RelationOp::LEQ));
    case LMX_EXPR_AND: return expr_from_result(lamina::lsr::logical_and(*left, *right));
    case LMX_EXPR_OR: return expr_from_result(lamina::lsr::logical_or(*left, *right));
    case LMX_EXPR_IN: return expr_from_result(lamina::lsr::membership(*left, *right));
    case LMX_EXPR_NOT_IN: return expr_from_result(lamina::lsr::membership(*left, *right, true));
    default:
        return expr_from_result(invalid_expr_operation(
            "unknown binary Expr operation", "runtime.expr_binary"));
    }
}

extern "C" LM_API ExprObj* cas_expr_function(const char* name,
                                               const LmInt count, ...) {
    va_list args;
    va_start(args, count);
    std::vector<lamina::lsr::ExprPtr> values;
    std::string error;
    const bool valid = collect_expr_arguments(args, count, values, error);
    va_end(args);
    if (!valid) return expr_error(std::move(error));
    return expr_from_result(lamina::lsr::function(name ? name : "", std::move(values)));
}

extern "C" LM_API ExprObj* cas_expr_set(const LmInt count, ...) {
    va_list args;
    va_start(args, count);
    std::vector<lamina::lsr::ExprPtr> values;
    std::string error;
    const bool valid = collect_expr_arguments(args, count, values, error);
    va_end(args);
    if (!valid) return expr_error(std::move(error));
    return expr_from_result(lamina::lsr::finite_set(std::move(values)));
}

extern "C" LM_API ExprObj* cas_expr_interval(ExprObj* lower, ExprObj* upper,
                                               const bool lower_closed,
                                               const bool upper_closed) {
    std::string error;
    const auto* lower_value = checked_expr(lower, error);
    if (!lower_value) return expr_error(std::move(error));
    const auto* upper_value = checked_expr(upper, error);
    if (!upper_value) return expr_error(std::move(error));
    return expr_from_result(lamina::lsr::interval(
        *lower_value, *upper_value, lower_closed, upper_closed));
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

extern "C" LM_API ExprObj* cas_substitute(ExprObj* expr, AdtObj* binding) {
    std::string error;
    const auto* value = checked_expr(expr, error);
    if (!value) return expr_error(std::move(error));
    if (!binding || binding->type_name() != "Binding" ||
        binding->constructor() != "Binding" || binding->fields().size() != 2) {
        return expr_error("cas_substitute expects Binding<Expr, Expr>");
    }
    const auto* symbol_field = binding->field(0);
    const auto* value_field = binding->field(1);
    if (!symbol_field || symbol_field->kind != lmx::runtime::ValueKind::Expr ||
        !value_field || value_field->kind != lmx::runtime::ValueKind::Expr) {
        return expr_error("cas_substitute expects Binding<Expr, Expr>");
    }
    const auto* symbol = checked_expr(
        reinterpret_cast<ExprObj*>(symbol_field->obj), error);
    if (!symbol) return expr_error(std::move(error));
    const auto* replacement = checked_expr(
        reinterpret_cast<ExprObj*>(value_field->obj), error);
    if (!replacement) return expr_error(std::move(error));
    const auto checked_binding = lamina::lsr::binding(*symbol, *replacement);
    if (!checked_binding) return expr_error(cas_error_text(checked_binding.error()));
    return expr_from_result(lamina::lsr::substitute(*value, checked_binding.value()));
}

extern "C" LM_API AdtObj* cas_evalf(ExprObj* expr) {
    std::string error;
    const auto* value = checked_expr(expr, error);
    if (!value) return real_result_error(std::move(error));
    const auto result = lamina::lsr::evalf(**value);
    if (!result) return real_result_error(cas_error_text(result.error()));
    return real_result_ok(result.value().value);
}

extern "C" LM_API bool cas_is_ok(ExprObj* expr) {
    return expr && expr->ok();
}

extern "C" LM_API StringObj* cas_to_text(ExprObj* expr) {
    if (!expr) return new StringObj("CasError(InvalidArgument: null expr)");
    return new StringObj(expr->to_string());
}

extern "C" LM_API StringObj* cas_error(ExprObj* expr) {
    if (!expr) return new StringObj("CasError(InvalidArgument: null expr)");
    if (expr->ok()) return new StringObj("");
    return new StringObj(expr->error());
}

extern "C" LM_API AdtObj* lmmc_num_hypot(ExprObj* lhs, ExprObj* rhs) {
    double x = 0.0;
    double y = 0.0;
    std::string error;
    if (!expr_to_real(lhs, x, error)) return real_result_error(std::move(error));
    if (!expr_to_real(rhs, y, error)) return real_result_error(std::move(error));
    lmmc_real_t out = 0.0;
    const auto status = lmmc_hypot(x, y, &out);
    return lmmc_real_result("lmmc_num_hypot", status, out);
}

extern "C" LM_API AdtObj* lmmc_num_log2(ExprObj* expr) {
    double x = 0.0;
    std::string error;
    if (!expr_to_real(expr, x, error)) return real_result_error(std::move(error));
    lmmc_real_t out = 0.0;
    const auto status = lmmc_log2(x, &out);
    return lmmc_real_result("lmmc_num_log2", status, out);
}

extern "C" LM_API AdtObj* lmmc_num_exp2(ExprObj* expr) {
    double x = 0.0;
    std::string error;
    if (!expr_to_real(expr, x, error)) return real_result_error(std::move(error));
    lmmc_real_t out = 0.0;
    const auto status = lmmc_exp2(x, &out);
    return lmmc_real_result("lmmc_num_exp2", status, out);
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
    const auto storage = malloc(sizeof(lmx::runtime::CodeModuleObj));
    if (storage == nullptr) return nullptr;
    new (storage) lmx::runtime::CodeModuleObj(std::move(binary));
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
    const auto* mod = reinterpret_cast<lmx::runtime::CodeModuleObj*>(module);
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
    run(reinterpret_cast<lmx::runtime::CodeModuleObj*>(module));
}

void lmx_vmEval(LmState *state, LaminaVM *vm, LmValue *result, const char *code) {
    std::string c = code;
    auto tks = lmx::Lexer(c).tokenize(c);
    auto node = lmx::Parser(tks).parse_stmt();
    lmx::hir::TypeCkContext().check_stmt(node);
}
