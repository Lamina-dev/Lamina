
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
#include "runtime/object/complex.hpp"
#include "runtime/object/array.hpp"
#include "runtime/object/vector.hpp"
#include "runtime/object/matrix.hpp"
#include "runtime/object/table.hpp"
#include "runtime/object/random.hpp"
#include "runtime/object/quantity.hpp"

#include <cmath>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <utility>
#include <cctype>

#include "compiler/mir/mir_printer.hpp"
#include "lmmc/numeric.h"
#include "lmmc/complex.h"
#include "lmmc/init.h"
#include "lmmc/dense.h"
#include "lmmc/stats.h"
#include "lmmc/random.h"
#include "lmmc/lsr_stdlib.h"
#include "complex_analysis.hpp"
#include "numerical_integration.hpp"
#include "series_engine.hpp"
#include "symbolic_geometry.hpp"
#include "symbolic_implicit_diff.hpp"
#include "transform_engine.hpp"
#include "vector_calculus.hpp"

LmState global_state;
namespace {

bool debug_dump_enabled() noexcept {
    const char* value = std::getenv("LMX_DEBUG_DUMP");
    return value && value[0] != '\0' && value[0] != '0';
}

} // namespace


extern "C" LM_API int lmx_printf(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    const int result = vprintf(fmt ? fmt : "", args);
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
    const auto storage = malloc(sizeof(lmx::runtime::CodeModuleObj));
    if (storage == nullptr) return nullptr;
    try {
        new (storage) lmx::runtime::CodeModuleObj(std::move(binary));
    } catch (...) {
        free(storage);
        throw;
    }
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
    try {
        lmx::Compiler compiler;
        lmx::CompileResult result;
        if (!compiler.compile_source(code, name, result)) return nullptr;
        return lmx_newCodeModule(state, std::move(result.binary));
    } catch (const lmx::runtime::VmFault& fault) {
        std::cerr << fault.what() << std::endl;
        return nullptr;
    }
}
LmModule *lmx_doFile(LmState *state, const char* name, bool is_main_module) {
    try {
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
    } catch (const lmx::runtime::VmFault& fault) {
        std::cerr << fault.what() << std::endl;
        return nullptr;
    }
}

int lmx_vmRunModule(LmState* state, LaminaVM* vm, LmModule* module) {
    if (module == nullptr) return 1;
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
