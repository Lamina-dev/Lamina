//
// Created by meian on 2026/4/8.
//

#include "include/lmx.h"


#include "compiler/ast_printer.hpp"
#include "compiler/hir.hpp"
#include "compiler/parser.hpp"
#include "compiler/lexer.hpp"
#include "runtime/vm.hpp"
#include <iostream>
#include <cstdlib>
#include <cstring>

#include "compiler/assembler.hpp"
#include "compiler/error.hpp"
#include "compiler/mir_builder.hpp"
#include "compiler/mir_printer.hpp"

LmState lmx_newState() {
    auto* node = static_cast<LmLinkedNode *>(malloc(sizeof(LmLinkedNode)));
    memset(node, 0, sizeof(LmLinkedNode));
    return LmState {node};
}
void lmx_deleteState(const LmState* state) {
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
    lmx::hir::HirContext().check_module(node.get());
    if (errd) return;

    const auto ast_str = lmx::AstPrinter::print(*node);
    if (fwrite(ast_str.c_str(), 1, ast_str.length(), file) != ast_str.length()) {
        fprintf(stderr, "Error writing AST to file\n");
    }

}

void lmx_printMIRFromString(LmState *state, FILE *file, const char *code, const char *name) {
    std::string c = code;
    auto tokens = lmx::Lexer(c).tokenize(c);
    const auto node = lmx::Parser(tokens).parse_module(name);
    if (errd) return;
    lmx::hir::HirContext().check_module(node.get());
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

LmModule *lmx_doString(LmState *state, const char *code, const char* name) {
    std::string c = code;
    auto tokens = lmx::Lexer(c).tokenize(c);
    const auto node = lmx::Parser(tokens).parse_module(name);
    if (errd) return nullptr;
    lmx::hir::HirContext().check_module(node.get());
    if (errd) return nullptr;

    auto mir = lmx::mir::MirBuilder::from_ast_module(node);
    if (errd) return nullptr;
    const auto binary = lmx::Assembler().asm_module(&mir);
    const auto new_m = malloc(binary.size() * sizeof(binary[0]));
    memcpy(new_m, binary.data(), binary.size() * sizeof(binary[0]));
    lmx_state_addNode(state, new_m);
    return static_cast<LmModule*>(new_m);
}

int lmx_vmRunModule(LmState* state, LaminaVM* vm, LmModule* module) {
    if (module == nullptr) return 1;
    lmx::runtime::CodeModule mod(reinterpret_cast<uint8_t*>(module));
    return reinterpret_cast<lmx::runtime::LaminaVM*>(vm)->run(&mod);
}

void lmx_vmEval(LmState *state, LaminaVM *vm, LmValue *result, const char *code) {
    std::string c = code;
    auto tks = lmx::Lexer(c).tokenize(c);
    const auto node = lmx::Parser(tks).parse_stmt();
    lmx::hir::HirContext().check_stmt(node.get());
}
