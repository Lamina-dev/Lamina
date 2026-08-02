//
// Created by meian on 2026/4/8.
//

#include "include/lmx.h"

#include "compiler/ast_printer.hpp"
#include "compiler/type_checker.hpp"
#include "compiler/parser.hpp"
#include "compiler/lexer.hpp"
#include "runtime/vm.hpp"

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <fstream>

#include "compiler/assembler.hpp"
#include "compiler/error.hpp"
#include "compiler/mir_builder.hpp"
#include "compiler/mir_printer.hpp"

LmState global_state;

LM_API LmState* lmx_newState() {
    auto* node = static_cast<LmLinkedNode *>(malloc(sizeof(LmLinkedNode)));
    memset(node, 0, sizeof(LmLinkedNode));
    global_state = LmState {node};
    return &global_state;
}
LM_API void lmx_deleteState(const LmState* state) {
    const LmLinkedNode* node = state->n;
    while (node->last) {
        if (node->ptr) free(node->ptr);
        const auto last = node->last;
        free((void*)node);
        node = last;
    }
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

void lmx_printASTFromString(LmState *state, FILE *file, const char *code, const char* name) {
    std::string c = code;
    auto tokens = lmx::Lexer(c).tokenize(c);
    if (errd) return;
    const auto node = lmx::Parser(tokens).parse_module(name);
    if (errd) return;
    lmx::hir::TypeCkContext().check_module(node);
    if (errd) return;

    const auto ast_str = lmx::AstPrinter::print(*node);
    if (fwrite(ast_str.c_str(), 1, ast_str.length(), file) != ast_str.length()) {
        fprintf(stderr, "Error writing AST to file\n");
    }
}

void lmx_printASTFromFile(LmState *state, FILE *file, const char *name) {

    const auto save_current_path = lmx::current_module_path;
    const auto this_module_path_name = std::filesystem::absolute(name).lexically_normal();
    lmx::current_module_path = this_module_path_name.parent_path();



    std::ifstream ifs(name);
    if (!ifs.is_open()) return;
    std::string c{
        std::istreambuf_iterator(ifs), std::istreambuf_iterator<char>()
    };
    ifs.close();
    auto tokens = lmx::Lexer(c).tokenize(c);
    const auto node = lmx::Parser(tokens).parse_module(name);
    if (errd) return;

    auto save_main_module = lmx::main_module;
    lmx::main_module = node;
    lmx::hir::TypeCkContext().check_module(node);

    if (errd) return;

    auto str = lmx::AstPrinter::print(*node.get());
    if (fwrite(str.c_str(), 1, str.length(), file) != str.length()) {
        fprintf(stderr, "Error writing AST to file\n");
    }
    lmx::current_module_path = save_current_path;
    lmx::main_module = save_main_module;
}

void lmx_printMIRFromString(LmState *state, FILE *file, const char *code, const char *name) {
    std::string c = code;
    auto tokens = lmx::Lexer(c).tokenize(c);
    const auto node = lmx::Parser(tokens).parse_module(name);
    if (errd) return;
    lmx::hir::TypeCkContext().check_module(node);
    if (errd) return;

    const auto mir = lmx::mir::MirBuilder::from_ast_module(node);
    if (errd) return;

    const auto mir_str = lmx::mir::MirPrinter::print(mir);

    if (fwrite(mir_str.c_str(), 1, mir_str.length(), file) != mir_str.length()) {
        fprintf(stderr, "Error writing MIR to file\n");
    }
}
void lmx_printMIRFromFile(LmState *state, FILE *file, const char *name) {
    std::ifstream ifs(name);
    if (!ifs.is_open()) return;
    std::string c{
        std::istreambuf_iterator(ifs), std::istreambuf_iterator<char>()
    };
    ifs.close();
    auto tokens = lmx::Lexer(c).tokenize(c);
    const auto node = lmx::Parser(tokens).parse_module(name);
    if (errd) return;
    lmx::hir::TypeCkContext().check_module(node);
    if (errd) return;

    const auto mir = lmx::mir::MirBuilder::from_ast_module(node);
    if (errd) return;

    const auto mir_str = lmx::mir::MirPrinter::print(mir);

    if (fwrite(mir_str.c_str(), 1, mir_str.length(), file) != mir_str.length()) {
        fprintf(stderr, "Error writing MIR to file\n");
    }
}

LaminaVM* lmx_newLaminaVM(LmState* state, int argc, char** argv) {
    auto* vm = static_cast<LaminaVM*>(malloc(sizeof(lmx::runtime::LaminaVM)));
    new (vm) lmx::runtime::LaminaVM(argc, argv);
    lmx_state_addNode(state, vm);
    return vm;
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
    std::string c = code;
    auto tokens = lmx::Lexer(c).tokenize(c);
    if (errd) return nullptr;

    const auto node = lmx::Parser(tokens).parse_module(name);
    if (errd) return nullptr;

    lmx::hir::TypeCkContext().check_module(node);
    if (errd) return nullptr;

    auto mir = lmx::mir::MirBuilder::from_ast_module(node);
    if (errd) return nullptr;

    auto binary = lmx::Assembler().asm_module(&mir);
    if (errd) return nullptr;

    const auto new_m = malloc(sizeof(lmx::runtime::CodeModule));
    if (new_m == nullptr) return nullptr;
    new (new_m) lmx::runtime::CodeModule(std::move(binary));
    lmx_state_addNode(state, new_m);
    return static_cast<LmModule*>(new_m);
}
LmModule *lmx_doFile(LmState *state, const char* name, bool is_main_module) {
    const auto save_current_path = lmx::current_module_path;
    const auto this_module_path_name = std::filesystem::absolute(name).lexically_normal();
    lmx::current_module_path = this_module_path_name.parent_path();
    std::ifstream ifs(this_module_path_name);
    if (!ifs.is_open()) return nullptr;
    std::string c{
        std::istreambuf_iterator(ifs), std::istreambuf_iterator<char>()
    };
    c += '\n';
    ifs.close();
    auto tokens = lmx::Lexer(c).tokenize(c);
    const auto node = lmx::Parser(tokens).parse_module(this_module_path_name.string());
    if (is_main_module) {
        if (lmx::main_module) {
            fprintf(stderr, "main module already exists: named \"%s\"\n", lmx::main_module->name.c_str());
            return nullptr;
        }
        lmx::main_module = node;
    }
    if (errd) return nullptr;
    lmx::hir::TypeCkContext().check_module(node);
#if !NDEBUG
    std::cout << lmx::AstPrinter::print(*node) << std::endl;
#endif
    if (errd) return nullptr;

    auto mir = lmx::mir::MirBuilder::from_ast_module(node);
#if !NDEBUG
    std::cout << lmx::mir::MirPrinter::print(mir) << std::endl;
#endif
    if (errd) return nullptr;
    auto binary = lmx::Assembler().asm_module(&mir);
    binary.shrink_to_fit();
    if (errd) return nullptr;

    const auto new_m = malloc(sizeof(lmx::runtime::CodeModule));
    new (new_m) lmx::runtime::CodeModule(std::move(binary));
    lmx_state_addNode(state, new_m);
    lmx::current_module_path = save_current_path;
    return static_cast<LmModule*>(new_m);
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
    const auto node = lmx::Parser(tks).parse_stmt();
    lmx::hir::TypeCkContext().check_stmt(node.get());
}
