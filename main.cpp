//
// Created by meian on 2026/4/8.
//

#include "compiler/ast_printer.hpp"
#include "compiler/hir.hpp"
#include "compiler/parser.hpp"
#include "compiler/lexer.hpp"
#include "runtime/vm.hpp"
#include "compiler/error.hpp"
#include <iostream>

int main(int argc, char** argv) {
    std::string code = R"(
func foo(a int, b int) {
    a + b
}
let v = foo(3, 7)
)";
    auto tokens = lmx::Lexer(code).tokenize(code);
    lmx::Parser parser(tokens);
    auto node = parser.parse_module("__test__");
    lmx::hir::HirContext().check_module(node.get());
    std::cout << lmx::AstPrinter::print(*node) << "\n";

    lmx::runtime::LaminaVM vm(nullptr, argc, argv);
    // while (true) {
    //     if (getchar() == 'c') {break;}
    // }
}
