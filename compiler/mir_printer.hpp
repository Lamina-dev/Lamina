//
// Created by meian on 2026/7/20.
//

#pragma once
#include <string>
#include <memory>
#include <sstream>

#include "mir.hpp"

namespace lmx::mir {

class MirPrinter {
public:
    static std::string print(const MirModule &module);
    static std::string print(const MirNode &node);
    static std::string print(const MirExpr &expr);
    static std::string opcode_name(runtime::Opcode::Opcode op);

private:
    static void format_node(std::ostringstream &ss, const MirNode &node);
    static void format_expr(std::ostringstream &ss, const MirExpr &expr);
    static void format_binary(std::ostringstream &ss, const std::string &op_name,
                              const std::shared_ptr<MirExpr> &lhs,
                              const std::shared_ptr<MirExpr> &rhs);
    static void format_unary(std::ostringstream &ss, const std::string &op_name,
                             const std::shared_ptr<MirExpr> &e);
    static void format_operate(std::ostringstream &ss, const MirOperateExpr &op);
};

} // namespace lmx::mir
