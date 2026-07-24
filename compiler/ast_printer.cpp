#include "ast_printer.hpp"

#include <iostream>
#include <ranges>

namespace lmx {

void AstPrinter::print_type(std::ostringstream &ss, const Type &type) {
    if (!&type) return;
    switch (type.kind) {
    case TypeKind::Unknown:
        ss << "?";
        break;
    case TypeKind::None:
        ss << "none";
        break;
    case TypeKind::Basic: {
        auto &bt = static_cast<const BasicType &>(type);
        switch (bt.type) {
            case runtime::ValueKind::Null:   ss << "Null"; break;
            case runtime::ValueKind::C_Ptr:  ss << "CPtr"; break;
            case runtime::ValueKind::Obj:    ss << "Object"; break;
            case runtime::ValueKind::Int:    ss << "Int"; break;
            case runtime::ValueKind::Bool:   ss << "Bool"; break;
            case runtime::ValueKind::Fraction: ss << "Frac"; break;
        }
        break;
    }
    case TypeKind::String:
        ss << "String";
        break;
    case TypeKind::Function: {
        auto &ft = static_cast<const FunctionType &>(type);
        ss << "func(";
        for (size_t i = 0; i < ft.params_ty.size(); i++) {
            if (i > 0) ss << ", ";
            if (ft.params_ty[i]) print_type(ss, *ft.params_ty[i]);
        }
        ss << ")";
        if (ft.ret_ty) {
            ss << " -> ";
            print_type(ss, *ft.ret_ty);
        }
        break;
    }
    case TypeKind::Named: {
        auto &nt = static_cast<const NamedType &>(type);
        ss << nt.name;
        break;
    }
    case TypeKind::Array: {
        auto &at = static_cast<const ArrayType &>(type);
        ss << "[";
        if (at.type) print_type(ss, *at.type);
        ss << "; " << at.len << "]";
        break;
    }
    }
}

void AstPrinter::print_expr(std::ostringstream &ss, const ExprNode &node,
                             const std::string &line_prefix, const std::string &child_prefix) {

    auto print_kids = [&](const auto &children) {
        for (size_t i = 0; i < children.size(); i++) {
            const bool last = i + 1 == children.size();
            auto kid_pref = child_prefix + (last ? "└── " : "├── ");
            auto kid_cont = child_prefix + (last ? "    " : "│   ");
            print_expr(ss, *children[i], kid_pref, kid_cont);
        }
    };

    switch (node.kind) {
        case ASTKind::Literal: {
            auto &lit = static_cast<const LiteralNode &>(node);
            ss << line_prefix;
            // switch (lit.kind) {
            //     case LiteralNode::Kind::Integer: ss << "Int"; break;
            //     case LiteralNode::Kind::Float:   ss << "Float"; break;
            //     case LiteralNode::Kind::String:  ss << "String"; break;
            //     case LiteralNode::Kind::Boolean: ss << "Bool"; break;
            // }
            ss << " " << lit.val;
            if (node.type) { ss << " : "; print_type(ss, *node.type); }
            ss << "\n";
            break;
        }
        case ASTKind::Identifier: {
            auto &id = static_cast<const IdentifierNode &>(node);
            ss << line_prefix << "Identifier " << id.id;
            if (node.type) { ss << " : "; print_type(ss, *node.type); }
            ss << "\n";
            break;
        }
        case ASTKind::Unary: {
            auto &un = static_cast<const UnaryNode &>(node);
            ss << line_prefix << "Unary ";
            switch (un.op) {
                case UnaryNode::Op::Neg: ss << "-"; break;
            }
            if (node.type) { ss << " : "; print_type(ss, *node.type); }
            ss << "\n";
            if (un.expr) {
                std::vector kids = {un.expr.get()};
                for (size_t i = 0; i < kids.size(); i++) {
                    print_expr(ss, *kids[i], child_prefix + "└── ", child_prefix + "    ");
                }
            }
            break;
        }
        case ASTKind::Binary: {
            auto &bn = static_cast<const BinaryNode &>(node);
            ss << line_prefix;
            switch (bn.op) {
                case BinaryNode::Op::Add: ss << "+"; break;
                case BinaryNode::Op::Sub: ss << "-"; break;
                case BinaryNode::Op::Mul: ss << "*"; break;
                case BinaryNode::Op::Div: ss << "/"; break;
                case BinaryNode::Op::Mod: ss << "%"; break;
                case BinaryNode::Op::Pow: ss << "**"; break;
                case BinaryNode::Op::Gt: ss << ">"; break;
                case BinaryNode::Op::Ge: ss << ">="; break;
                case BinaryNode::Op::Lt: ss << "<"; break;
                case BinaryNode::Op::Le: ss << "<="; break;
                case BinaryNode::Op::Eq: ss << "=="; break;
                case BinaryNode::Op::Ne: ss << "!="; break;
                case BinaryNode::Op::And: ss << "and"; break;
                case BinaryNode::Op::Or: ss << "or"; break;
                case BinaryNode::Op::Dot: ss << "."; break;
                // case BinaryNode::Op::ColonColon: ss << "::"; break;
            }
            if (node.type) { ss << " : "; print_type(ss, *node.type); }
            ss << "\n";
            if (bn.lhs) print_expr(ss, *bn.lhs, child_prefix + "├── ", child_prefix + "│   ");
            if (bn.rhs) print_expr(ss, *bn.rhs, child_prefix + "└── ", child_prefix + "    ");
            break;
        }
        case ASTKind::Block: {
            auto &blk = static_cast<const BlockExprNode &>(node);
            ss << line_prefix << "Block";
            if (node.type) { ss << " : "; print_type(ss, *node.type); }
            ss << "\n";
            for (size_t i = 0; i < blk.stmts.size(); i++) {
                const bool last = i + 1 == blk.stmts.size();
                auto kid_pref = child_prefix + (last ? "└── " : "├── ");
                auto kid_cont = child_prefix + (last ? "    " : "│   ");
                print_stmt(ss, *blk.stmts[i], kid_pref, kid_cont);
            }
            break;
        }
        case ASTKind::Exprs: {
            auto &exprs = static_cast<const ExprsNode &>(node);
            ss << line_prefix << "Exprs";
            if (node.type) { ss << " : "; print_type(ss, *node.type); }
            ss << "\n";
            for (size_t i = 0; i < exprs.exprs.size(); i++) {
                const bool last = i + 1 == exprs.exprs.size();
                auto kid_pref = child_prefix + (last ? "└── " : "├── ");
                auto kid_cont = child_prefix + (last ? "    " : "│   ");
                print_expr(ss, *exprs.exprs[i], kid_pref, kid_cont);
            }
            break;
        }
        case ASTKind::SuffixParen: {
            auto &sp = static_cast<const SuffixParenNode &>(node);
            ss << line_prefix << "Call";
            if (node.type) { ss << " : "; print_type(ss, *node.type); }
            ss << "\n";
            print_expr(ss, *sp.expr, child_prefix + "├── ", child_prefix + "│   ");
            print_expr(ss, *sp.suffix, child_prefix + "└── ", child_prefix + "    ");
            break;
        }
        case ASTKind::SuffixBracket: {
            auto &sb = static_cast<const SuffixBracketNode &>(node);
            ss << line_prefix << "Index";
            if (node.type) { ss << " : "; print_type(ss, *node.type); }
            ss << "\n";
            print_expr(ss, *sb.expr, child_prefix + "├── ", child_prefix + "│   ");
            print_expr(ss, *sb.suffix, child_prefix + "└── ", child_prefix + "    ");
            break;
        }
        case ASTKind::IfExpr: {
            auto &ifn = static_cast<const IfExprNode &>(node);
            ss << line_prefix << "If";
            if (node.type) { ss << " : "; print_type(ss, *node.type); }
            ss << "\n";
            print_expr(ss, *ifn.cond, child_prefix + "├── ", child_prefix + "│   ");
            print_expr(ss, *ifn.then, child_prefix + "├── ", child_prefix + "│   ");
            if (ifn.els) {
                print_node(ss, *ifn.els, child_prefix + "└── ", child_prefix + "    ");
            }
            break;
        }
        case ASTKind::AsExpr: {
            auto &as = static_cast<const AsExprNode &>(node);
            ss << line_prefix << "As";
            if (node.type) { ss << " : "; print_type(ss, *node.type); }
            ss << "\n";
            if (as.expr) print_expr(ss, *as.expr, child_prefix + "├── ", child_prefix + "│   ");
            if (as.cast_type) {
                ss << child_prefix << "└── " << "Type ";
                print_type(ss, *as.cast_type);
                ss << "\n";
            }
            break;
        }
        default:
            ss << line_prefix << "UnknownExpr(" << static_cast<int>(node.kind) << ")\n";
            break;
    }
}

void AstPrinter::print_stmt(std::ostringstream &ss, const StmtNode &node,
                             const std::string &line_prefix, const std::string &child_prefix) {
    switch (node.kind) {
        case ASTKind::ExprStmt: {
            auto &es = static_cast<const ExprStmtNode &>(node);
            ss << line_prefix << "ExprStmt\n";
            if (es.expr) print_expr(ss, *es.expr, child_prefix + "└── ", child_prefix + "    ");
            break;
        }
        case ASTKind::Return: {
            auto &rn = static_cast<const ReturnNode &>(node);
            ss << line_prefix << "Return\n";
            if (rn.expr) print_expr(ss, *rn.expr, child_prefix + "└── ", child_prefix + "    ");
            break;
        }
        case ASTKind::TailReturn: {
            auto &tr = static_cast<const TailReturnNode &>(node);
            ss << line_prefix << "TailReturn\n";
            if (tr.expr) print_expr(ss, *tr.expr, child_prefix + "└── ", child_prefix + "    ");
            break;
        }
        case ASTKind::BreakStmt: {
            ss << line_prefix << "Break\n";
            break;
        }
        case ASTKind::ParamsDeclNode: {
            auto &pd = static_cast<const ParamsDeclNode &>(node);
            ss << line_prefix << "Params\n";
            for (size_t i = 0; i < pd.stmts.size(); i++) {
                const bool last = i + 1 == pd.stmts.size();
                auto kid_pref = child_prefix + (last ? "└── " : "├── ");
                auto &[name, type] = pd.stmts[i];
                ss << kid_pref << name << " : ";
                if (type) {
                    print_type(ss, *type);
                } else {
                    ss << "?";
                }
                ss << "\n";
            }
            break;
        }
        case ASTKind::FuncImpl: {
            auto &fn = static_cast<const FuncImplNode &>(node);
            ss << line_prefix << "Func " << fn.func_id;
            if (fn.return_type) {
                ss << " -> ";
                print_type(ss, *fn.return_type);
            }
            ss << "\n";
            if (fn.params) print_stmt(ss, *fn.params, child_prefix + "├── ", child_prefix + "│   ");
            if (fn.block) print_expr(ss, *fn.block, child_prefix + "└── ", child_prefix + "    ");
            break;
        }
        case ASTKind::VarDecl: {
            auto &vd = static_cast<const VarDeclNode &>(node);
            ss << line_prefix;
            if (vd.is_mutable) ss << "var "; else ss << "let ";
            ss << vd.id;
            if (vd.type) { ss << " : "; print_type(ss, *vd.type); }
            ss << "\n";
            if (vd.init_value) {
                print_expr(ss, *vd.init_value, child_prefix + "└── ", child_prefix + "    ");
            }
            break;
        }
        case ASTKind::AssignStmt: {
            auto &as = static_cast<const AssignStmtNode &>(node);
            ss << line_prefix << "Assign\n";
            if (as.lhs) print_expr(ss, *as.lhs, child_prefix + "├── ", child_prefix + "│   ");
            if (as.rhs) print_expr(ss, *as.rhs, child_prefix + "└── ", child_prefix + "    ");
            break;
        }
        default:
            ss << line_prefix << "UnknownStmt(" << static_cast<int>(node.kind) << ")\n";
            break;
    }
}

static bool is_expr_kind(ASTKind kind) {
    switch (kind) {
        case ASTKind::Literal:
        case ASTKind::Identifier:
        case ASTKind::Unary:
        case ASTKind::Binary:
        case ASTKind::Block:
        case ASTKind::Exprs:
        case ASTKind::SuffixParen:
        case ASTKind::SuffixBracket:
        case ASTKind::IfExpr:
        case ASTKind::AsExpr:
            return true;
        default:
            return false;
    }
}

void AstPrinter::print_node(std::ostringstream &ss, const ASTNode &node,
                             const std::string &line_prefix, const std::string &child_prefix) {
    if (is_expr_kind(node.kind)) {
        print_expr(ss, static_cast<const ExprNode &>(node), line_prefix, child_prefix);
    } else {
        print_stmt(ss, static_cast<const StmtNode &>(node), line_prefix, child_prefix);
    }
}

std::string AstPrinter::print(const ASTNode &node) {
    std::ostringstream ss;
    if (is_expr_kind(node.kind)) {
        print_expr(ss, static_cast<const ExprNode &>(node), "", "");
    } else {
        print_stmt(ss, static_cast<const StmtNode &>(node), "", "");
    }
    return ss.str();
}

std::string AstPrinter::print(const Module &module) {
    std::ostringstream ss;
    ss << "Module " << module.name << "\n";
    for (size_t i = 0; i < module.decls.size(); i++) {
        const bool last = i + 1 == module.decls.size();
        auto pref = last ? "└── " : "├── ";
        auto cont = last ? "    " : "│   ";
        print_stmt(ss, *module.decls[i], pref, cont);
    }
    return ss.str();
}

} // namespace lmx
