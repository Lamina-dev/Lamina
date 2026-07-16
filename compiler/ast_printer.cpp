//
// Created by meian on 2026/7/12.
//

#include "ast_printer.hpp"

namespace lmx {

void AstPrinter::print_indent(std::ostringstream &ss, int indent) {
    for (int i = 0; i < indent; ++i) {
        ss << "  ";
    }
}

void AstPrinter::print_type(std::ostringstream &ss, const Type &type, int indent) {
    print_indent(ss, indent);
    switch (type.kind) {
    case TypeKind::Unknown:
            unknown:
            ss << "UnknownType\n";
            break;
        case TypeKind::None:
            ss << "NoneType\n";
            break;
        case TypeKind::Basic: {
            auto &bt = static_cast<const BasicType &>(type);
            ss << "BasicType: ";
            switch (bt.type) {
                case runtime::ValueKind::Null:   ss << "Null"; break;
                case runtime::ValueKind::C_Ptr:  ss << "CPtr"; break;
                case runtime::ValueKind::Obj:    ss << "Obj"; break;
                case runtime::ValueKind::Int:    ss << "Int"; break;
                case runtime::ValueKind::Bool:   ss << "Bool"; break;
                case runtime::ValueKind::Fraction: ss << "Fraction"; break;
            }
            ss << "\n";
            break;
        }
        case TypeKind::String:
            ss << "StringType\n";
            break;
        case TypeKind::Function: {
            auto &ft = static_cast<const FunctionType &>(type);
            ss << "FunctionType\n";
            for (auto &p : ft.params_ty) {
                if (p) print_type(ss, *p, indent + 1);
            }
            if (ft.ret_ty) {
                print_indent(ss, indent + 1);
                ss << "-> ";
                print_type(ss, *ft.ret_ty, 0);
            }
            break;
        }
        case TypeKind::Named: {
            auto &nt = static_cast<const NamedType &>(type);
            ss << "NamedType: " << nt.name << "\n";
            break;
        }
        case TypeKind::Array: {
            auto &at = static_cast<const ArrayType &>(type);
            ss << "ArrayType [" << at.len << "]\n";
            if (at.type) print_type(ss, *at.type, indent + 1);
            break;
        }
    }
}

void AstPrinter::print_expr(std::ostringstream &ss, const ExprNode &node, int indent) {
    switch (node.kind) {
        case ASTKind::Literal: {
            auto &lit = static_cast<const LiteralNode &>(node);
            print_indent(ss, indent);
            ss << "Literal: ";
            switch (lit.kind) {
                case LiteralNode::Kind::Integer: ss << "Integer"; break;
                case LiteralNode::Kind::Float:   ss << "Float"; break;
                case LiteralNode::Kind::String:  ss << "String"; break;
                case LiteralNode::Kind::Boolean: ss << "Boolean"; break;
            }
            ss << " " << lit.val << "\n";
            break;
        }
        case ASTKind::Identifier: {
            auto &id = static_cast<const IdentifierNode &>(node);
            print_indent(ss, indent);
            ss << "Identifier: " << id.id << "\n";
            break;
        }
        case ASTKind::Unary: {
            auto &un = static_cast<const UnaryNode &>(node);
            print_indent(ss, indent);
            ss << "Unary: " << un.op << "\n";
            if (un.expr) print_expr(ss, *un.expr, indent + 1);
            break;
        }
        case ASTKind::Binary: {
            auto &bn = static_cast<const BinaryNode &>(node);
            print_indent(ss, indent);
            ss << "Binary: " << bn.op << "\n";
            if (bn.lhs) print_expr(ss, *bn.lhs, indent + 1);
            if (bn.rhs) print_expr(ss, *bn.rhs, indent + 1);
            break;
        }
        case ASTKind::Block: {
            auto &blk = static_cast<const BlockExprNode &>(node);
            print_indent(ss, indent);
            ss << "Block\n";
            for (auto &s : blk.stmts) {
                if (s) print_stmt(ss, *s, indent + 1);
            }
            break;
        }
        case ASTKind::Exprs: {
            auto &exprs = static_cast<const ExprsNode &>(node);
            print_indent(ss, indent);
            ss << "Exprs\n";
            for (auto &e : exprs.exprs) {
                if (e) print_expr(ss, *e, indent + 1);
            }
            break;
        }
        case ASTKind::SuffixParen: {
            auto &sp = static_cast<const SuffixParenNode &>(node);
            print_indent(ss, indent);
            ss << "SuffixParen\n";
            if (sp.expr) print_expr(ss, *sp.expr, indent + 1);
            if (sp.suffix) print_expr(ss, *sp.suffix, indent + 1);
            break;
        }
        case ASTKind::SuffixBracket: {
            auto &sb = static_cast<const SuffixBracketNode &>(node);
            print_indent(ss, indent);
            ss << "SuffixBracket\n";
            if (sb.expr) print_expr(ss, *sb.expr, indent + 1);
            if (sb.suffix) print_expr(ss, *sb.suffix, indent + 1);
            break;
        }
        case ASTKind::IfExpr: {
            auto &ifn = static_cast<const IfExprNode &>(node);
            print_indent(ss, indent);
            ss << "IfExpr\n";
            print_indent(ss, indent + 1);
            ss << "cond:\n";
            if (ifn.cond) print_expr(ss, *ifn.cond, indent + 2);
            print_indent(ss, indent + 1);
            ss << "then:\n";
            if (ifn.then) print_expr(ss, *ifn.then, indent + 2);
            print_indent(ss, indent + 1);
            ss << "else:\n";
            if (ifn.els) print_node(ss, *ifn.els, indent + 2);
            break;
        }
        default:
            print_indent(ss, indent);
            ss << "UnknownExpr(kind=" << static_cast<int>(node.kind) << ")\n";
            break;
    }
    if (node.type) print_type(ss, *node.type, indent);
}

void AstPrinter::print_stmt(std::ostringstream &ss, const StmtNode &node, int indent) {
    switch (node.kind) {
        case ASTKind::ExprStmt: {
            auto &es = static_cast<const ExprStmtNode &>(node);
            print_indent(ss, indent);
            ss << "ExprStmt\n";
            if (es.expr) print_expr(ss, *es.expr, indent + 1);
            break;
        }
        case ASTKind::Return: {
            auto &rn = static_cast<const ReturnNode &>(node);
            print_indent(ss, indent);
            ss << "Return\n";
            if (rn.expr) print_expr(ss, *rn.expr, indent + 1);
            break;
        }
        case ASTKind::TailReturn: {
            auto &tr = static_cast<const TailReturnNode &>(node);
            print_indent(ss, indent);
            ss << "TailReturn\n";
            if (tr.expr) print_expr(ss, *tr.expr, indent + 1);
            break;
        }
        case ASTKind::BreakStmt: {
            print_indent(ss, indent);
            ss << "Break\n";
            break;
        }
        case ASTKind::ParamsDeclNode: {
            auto &pd = static_cast<const ParamsDeclNode &>(node);
            print_indent(ss, indent);
            ss << "Params\n";
            for (auto &[name, type] : pd.stmts) {
                print_indent(ss, indent + 1);
                ss << name << ": ";
                if (type) {
                    print_type(ss, *type, 0);
                } else {
                    ss << "(null)\n";
                }
            }
            break;
        }
        case ASTKind::FuncImpl: {
            auto &fn = static_cast<const FuncImplNode &>(node);
            print_indent(ss, indent);
            ss << "FuncImpl\n";
            print_indent(ss, indent + 1);
            ss << fn.func_id << "\n";
            if (fn.params) print_stmt(ss, *fn.params, indent + 1);
            if (fn.return_type) {
                print_indent(ss, indent + 1);
                ss << "return: ";
                print_type(ss, *fn.return_type, 0);
            }
            if (fn.block) print_expr(ss, *fn.block, indent + 1);
            break;
        }
        case ASTKind::VarDecl: {
            auto &vd = static_cast<const VarDeclNode &>(node);
            print_indent(ss, indent);
            ss << "VarDecl (mutable=" << (vd.is_mutable ? "true" : "false") << ")\n";
            print_indent(ss, indent + 1);
            ss << vd.id << "\n";
            if (vd.type) {
                print_indent(ss, indent + 1);
                ss << "type: ";
                print_type(ss, *vd.type, 0);
            }
            if (vd.init_value) {
                print_indent(ss, indent + 1);
                ss << "init:\n";
                print_expr(ss, *vd.init_value, indent + 2);
            }
            break;
        }
        default:
            print_indent(ss, indent);
            ss << "UnknownStmt(kind=" << static_cast<int>(node.kind) << ")\n";
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
            return true;
        default:
            return false;
    }
}

void AstPrinter::print_node(std::ostringstream &ss, const ASTNode &node, int indent) {
    if (is_expr_kind(node.kind)) {
        print_expr(ss, static_cast<const ExprNode &>(node), indent);
    } else {
        print_stmt(ss, static_cast<const StmtNode &>(node), indent);
    }
}

std::string AstPrinter::print(const ASTNode &node) {
    std::ostringstream ss;
    print_node(ss, node, 0);
    return ss.str();
}

std::string AstPrinter::print(const Module &module) {
    std::ostringstream ss;
    ss << "Module: " << module.name << "\n";
    for (auto &d : module.decls) {
        if (d) print_stmt(ss, *d, 1);
    }
    return ss.str();
}

} // namespace lmx
