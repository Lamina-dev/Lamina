//
// Created by meian on 2026/7/20.
//

#include "mir_printer.hpp"
#include "../runtime/opcode.hpp"

#include <memory>
#include <string>
#include <sstream>

namespace lmx::mir {

std::string MirPrinter::opcode_name(const runtime::Opcode::Opcode op) {
    switch (op) {
    case runtime::Opcode::Nop:        return "nop";
    case runtime::Opcode::New:        return "new";
    case runtime::Opcode::GetTrue:    return "get_true";
    case runtime::Opcode::GetFalse:   return "get_false";
    case runtime::Opcode::GetNull:    return "get_null";
    case runtime::Opcode::IConst:     return "iconst";
    case runtime::Opcode::CConst:     return "cconst";
    case runtime::Opcode::Pop:        return "pop";
    case runtime::Opcode::Halt:       return "halt";
    case runtime::Opcode::IAdd:       return "iadd";
    case runtime::Opcode::ISub:       return "isub";
    case runtime::Opcode::IMul:       return "imul";
    case runtime::Opcode::IDiv:       return "idiv";
    case runtime::Opcode::IMod:       return "imod";
    case runtime::Opcode::IPow:       return "ipow";
    case runtime::Opcode::INeg:       return "ineg";
    case runtime::Opcode::FuncCreate: return "func_create";
    case runtime::Opcode::CallVirtual: return "call_virtual";
    case runtime::Opcode::CCall:       return "ccall";
    case runtime::Opcode::CallFast:    return "call_fast";
    case runtime::Opcode::Ret:         return "ret";

    case runtime::Opcode::Goto:        return "goto";
    case runtime::Opcode::ICmpEq:      return "icmp_eq";
    case runtime::Opcode::ICmpNe:      return "icmp_ne";
    case runtime::Opcode::ICmpLt:      return "icmp_lt";
    case runtime::Opcode::ICmpLe:      return "icmp_le";
    case runtime::Opcode::ICmpGt:      return "icmp_gt";
    case runtime::Opcode::ICmpGe:      return "icmp_ge";
    case runtime::Opcode::IfTrue:      return "if_true";
    case runtime::Opcode::IfFalse:     return "if_false";
    case runtime::Opcode::LGet:        return "lget";
    case runtime::Opcode::LSet:        return "lset";
    case runtime::Opcode::GGet:        return "gget";
    case runtime::Opcode::GSet:        return "gset";
    case runtime::Opcode::FAdd:        return "fadd";
    case runtime::Opcode::FSub:        return "fsub";
    case runtime::Opcode::FMul:        return "fmul";
    case runtime::Opcode::FDiv:        return "fdiv";
    case runtime::Opcode::FMod:       return "fmodi";
    case runtime::Opcode::FNeg:        return "fneg";
    default:                           return "unknown";
    }
}

void MirPrinter::format_expr(std::ostringstream &ss, const MirExpr &expr) {
    switch (expr.kind) {
    case MirExprKind::Literal: {
        auto &lit = static_cast<const MirLiteralExpr &>(expr);
        ss << "literal ";
        switch (lit.literal_kind) {
        case MirLiteralKind::Integer: ss << "int";   break;
        case MirLiteralKind::Float:   ss << "float"; break;
        case MirLiteralKind::String:  ss << "str";   break;
        case MirLiteralKind::Boolean: ss << "bool";  break;
        }
        ss << " \"" << lit.data << "\"";
        break;
    }
    case MirExprKind::Ref: {
        auto &ref = static_cast<const MirRefExpr &>(expr);
        if (ref.is_temp) {
            ss << "%" << ref.name;
        } else {
            ss << "$" << ref.name;
        }
        break;
    }
    case MirExprKind::Operate: {
        format_operate(ss, static_cast<const MirOperateExpr &>(expr));
        break;
    }
    }
}

void MirPrinter::format_binary(std::ostringstream &ss, const std::string &op_name,
                                const std::shared_ptr<MirExpr> &lhs,
                                const std::shared_ptr<MirExpr> &rhs) {
    ss << op_name << " ";
    if (lhs) format_expr(ss, *lhs);
    ss << ", ";
    if (rhs) format_expr(ss, *rhs);
}

void MirPrinter::format_unary(std::ostringstream &ss, const std::string &op_name,
                               const std::shared_ptr<MirExpr> &e) {
    ss << op_name << " ";
    if (e) format_expr(ss, *e);
}

void MirPrinter::format_operate(std::ostringstream &ss, const MirOperateExpr &op) {
    if (op.operate_kind == MirOperateKind::RetVoid) {
        ss << "ret_void";
        return;
    }
    auto name = opcode_name(op.opcode);

    auto print_with = [&](const auto &derived, auto &&fn) {
        fn(ss, name, derived);
    };

#define DISPATCH(Type, field_dispatch) \
    { \
        auto &d = static_cast<const Type &>(op); \
        field_dispatch; \
        return; \
    }

    switch (op.opcode) {
    case runtime::Opcode::Nop:
        ss << name;
        break;
    case runtime::Opcode::Halt:
        ss << name;
        break;
    case runtime::Opcode::Ret:
        DISPATCH(MirRetExpr, format_unary(ss, name, d.value));

    case runtime::Opcode::New:
        DISPATCH(MirNewExpr, format_unary(ss, name, d.expr));
    case runtime::Opcode::IAdd:
        DISPATCH(MirIAddExpr, format_binary(ss, name, d.lhs, d.rhs));
    case runtime::Opcode::ISub:
        DISPATCH(MirISubExpr, format_binary(ss, name, d.lhs, d.rhs));
    case runtime::Opcode::IMul:
        DISPATCH(MirIMulExpr, format_binary(ss, name, d.lhs, d.rhs));
    case runtime::Opcode::IDiv:
        DISPATCH(MirIDivExpr, format_binary(ss, name, d.lhs, d.rhs));
    case runtime::Opcode::IMod:
        DISPATCH(MirIModExpr, format_binary(ss, name, d.lhs, d.rhs));
    case runtime::Opcode::IPow:
        DISPATCH(MirIPowExpr, format_binary(ss, name, d.lhs, d.rhs));
    case runtime::Opcode::INeg:
        DISPATCH(MirINegExpr, format_unary(ss, name, d.e));
    case runtime::Opcode::FAdd:
        DISPATCH(MirFAddExpr, format_binary(ss, name, d.lhs, d.rhs));
    case runtime::Opcode::FSub:
        DISPATCH(MirFSubExpr, format_binary(ss, name, d.lhs, d.rhs));
    case runtime::Opcode::FMul:
        DISPATCH(MirFMulExpr, format_binary(ss, name, d.lhs, d.rhs));
    case runtime::Opcode::FDiv:
        DISPATCH(MirFDivExpr, format_binary(ss, name, d.lhs, d.rhs));
    case runtime::Opcode::FMod:
        DISPATCH(MirFModExpr, format_binary(ss, name, d.lhs, d.rhs));
    case runtime::Opcode::FNeg:
        DISPATCH(MirFNegExpr, format_unary(ss, name, d.e));
    case runtime::Opcode::ICmpEq:
        DISPATCH(MirICmpEqExpr, format_binary(ss, name, d.lhs, d.rhs));
    case runtime::Opcode::ICmpNe:
        DISPATCH(MirICmpNeExpr, format_binary(ss, name, d.lhs, d.rhs));
    case runtime::Opcode::ICmpLt:
        DISPATCH(MirICmpLtExpr, format_binary(ss, name, d.lhs, d.rhs));
    case runtime::Opcode::ICmpLe:
        DISPATCH(MirICmpLeExpr, format_binary(ss, name, d.lhs, d.rhs));
    case runtime::Opcode::ICmpGt:
        DISPATCH(MirICmpGtExpr, format_binary(ss, name, d.lhs, d.rhs));
    case runtime::Opcode::ICmpGe:
        DISPATCH(MirICmpGeExpr, format_binary(ss, name, d.lhs, d.rhs));
    case runtime::Opcode::Goto: {
        auto &g = static_cast<const MirGotoExpr &>(op);
        ss << name << " " << g.label;
        break;
    }
    case runtime::Opcode::IfTrue: {
        auto &i = static_cast<const MirIfTrueExpr &>(op);
        ss << name << " ";
        if (i.cond) format_expr(ss, *i.cond);
        ss << " -> " << i.label;
        break;
    }
    case runtime::Opcode::IfFalse: {
        auto &i = static_cast<const MirIfFalseExpr &>(op);
        ss << name << " ";
        if (i.cond) format_expr(ss, *i.cond);
        ss << " -> " << i.label;
        break;
    }
    case runtime::Opcode::CallFast: {
        auto &c = static_cast<const MirCallFastExpr &>(op);
        ss << name << " " << c.name << "(";
        for (size_t i = 0; i < c.args.size(); ++i) {
            if (i > 0) ss << ", ";
            format_expr(ss, *c.args[i]);
        }
        ss << ")";
        break;
    }
    case runtime::Opcode::CallVirtual: {
        auto &c = static_cast<const MirCallVirtualExpr &>(op);
        ss << name << " reg=" << static_cast<int>(c.reg) << " argc=" << static_cast<int>(c.arg_count);
        break;
    }
    default:
        ss << name;
        break;
    }
}

void MirPrinter::format_node(std::ostringstream &ss, const MirNode &node) {
    switch (node.kind) {
    case MirNodeKind::Label: {
        auto &l = static_cast<const MirLabel &>(node);
        ss << l.name << ":";
        break;
    }
    case MirNodeKind::TempAssign: {
        auto &ta = static_cast<const MirTempAssign &>(node);
        ss << "%" << ta.name << " = ";
        if (ta.expr) format_expr(ss, *ta.expr);
        break;
    }
    case MirNodeKind::Assign: {
        auto &a = static_cast<const MirAssign &>(node);
        ss << "$" << a.name << " = ";
        if (a.expr) format_expr(ss, *a.expr);
        break;
    }
    case MirNodeKind::Expr: {
        auto &en = static_cast<const MirExprNode &>(node);
        if (en.expr) format_expr(ss, *en.expr);
        break;
    }
    case MirNodeKind::Func: {
        auto &f = static_cast<const MirFuncDefine &>(node);
        ss << "func " << f.name << "(";
        for (size_t i = 0; i < f.params.size(); ++i) {
            if (i > 0) ss << ", ";
            ss << "$" << f.params[i];
        }
        ss << ") {\n";
        for (auto &n : f.body) {
            ss << "    ";
            format_node(ss, *n);
            ss << "\n";
        }
        ss << "}";
        break;
    }
    }
}

std::string MirPrinter::print(const MirNode &node) {
    std::ostringstream ss;
    format_node(ss, node);
    return ss.str();
}

std::string MirPrinter::print(const MirExpr &expr) {
    std::ostringstream ss;
    format_expr(ss, expr);
    return ss.str();
}

std::string MirPrinter::print(const MirModule &module) {
    std::ostringstream ss;
    for (auto &node : module.nodes) {
        if (node->kind != MirNodeKind::Func) {
            ss << "  ";
        }
        format_node(ss, *node);
        ss << "\n";
    }
    return ss.str();
}

} // namespace lmx::mir
