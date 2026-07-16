//
// Created by meian on 2026/7/12.
//

#pragma once
#include <string>
#include <memory>
#include <sstream>
#include "ast.hpp"

namespace lmx {

class AstPrinter {
public:
    static std::string print(const ASTNode &node);
    static std::string print(const Module &module);

private:
    static void print_node(std::ostringstream &ss, const ASTNode &node, int indent);
    static void print_expr(std::ostringstream &ss, const ExprNode &node, int indent);
    static void print_stmt(std::ostringstream &ss, const StmtNode &node, int indent);
    static void print_type(std::ostringstream &ss, const Type &type, int indent);
    static void print_indent(std::ostringstream &ss, int indent);
};

} // namespace lmx
