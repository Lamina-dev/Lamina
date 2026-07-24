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
    static void print_node(std::ostringstream &ss, const ASTNode &node, const std::string &line_prefix, const std::string &child_prefix);
    static void print_expr(std::ostringstream &ss, const ExprNode &node, const std::string &line_prefix, const std::string &child_prefix);
    static void print_stmt(std::ostringstream &ss, const StmtNode &node, const std::string &line_prefix, const std::string &child_prefix);
    static void print_type(std::ostringstream &ss, const Type &type);
};

} // namespace lmx
