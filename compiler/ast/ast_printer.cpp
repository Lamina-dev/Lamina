#include "ast_printer.hpp"

#include <iostream>
#include <ranges>

namespace lmx {

void AstPrinter::print_type(std::ostringstream &ss, const Type &type) {
    switch (type.kind) {
    case TypeKind::Unknown:
        ss << "?";
        break;
    case TypeKind::None:
        ss << "none";
        break;
    case TypeKind::Basic: {
        switch (auto &bt = static_cast<const BasicType &>(type); bt.type) {
            case runtime::ValueKind::Null:   ss << "Null"; break;
            case runtime::ValueKind::C_Ptr:  ss << "CPtr"; break;
            case runtime::ValueKind::Obj:    ss << "Object"; break;
            case runtime::ValueKind::Int:    ss << "Int"; break;
            case runtime::ValueKind::Bool:   ss << "Bool"; break;
            case runtime::ValueKind::Fraction: ss << "Frac"; break;
            case runtime::ValueKind::Real: ss << "Real"; break;
            case runtime::ValueKind::Expr: ss << "Expr"; break;
            case runtime::ValueKind::C_VaList: ss << "..."; break;
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
        if (!nt.args.empty()) {
            ss << "<";
            for (size_t i = 0; i < nt.args.size(); ++i) {
                if (i) ss << ", ";
                print_type(ss, *nt.args[i]);
            }
            ss << ">";
        }
        break;
    }
    case TypeKind::Array: {
        auto &at = static_cast<const ArrayType &>(type);
        ss << "[";
        if (at.type) print_type(ss, *at.type);
        ss << "]";
        break;
    }
    case TypeKind::Tuple: {
        auto &tt = static_cast<const TupleType &>(type);
        ss << "(";
        for (size_t i = 0; i < tt.tys.size(); i++) {
            if (i > 0) ss << ", ";
            if (tt.tys[i]) print_type(ss, *tt.tys[i]);
        }
        ss << ")";
        break;
    }
    case TypeKind::NativeFunction: {
        auto &nt = static_cast<const NativeFunctionType &>(type);
        ss << "native func(";
        for (size_t i = 0; i < nt.params_ty.size(); i++) {
            if (i > 0) ss << ", ";
            if (nt.params_ty[i]) print_type(ss, *nt.params_ty[i]);
        }
        ss << ")";
        if (nt.ret_ty) {
            ss << " -> ";
            print_type(ss, *nt.ret_ty);
        }
        break;
    }
    case TypeKind::Module: {
        auto &mt = static_cast<const ModuleType &>(type);
        ss << "module(" << mt.target_path << ")";
        break;
    }
    case TypeKind::Nullable: {
        auto &nullable = static_cast<const NullableType &>(type);
        print_type(ss, *nullable.value_type);
        ss << "?";
        break;
    }
    case TypeKind::AdtConstructor: {
        auto &constructor = static_cast<const AdtConstructorType &>(type);
        ss << constructor.type_name << "." << constructor.constructor;
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
                print_expr(ss, *un.expr.get(), child_prefix + "└── ", child_prefix + "    ");
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
                case BinaryNode::Op::Pow: ss << "^"; break;
                case BinaryNode::Op::Gt: ss << ">"; break;
                case BinaryNode::Op::Ge: ss << ">="; break;
                case BinaryNode::Op::Lt: ss << "<"; break;
                case BinaryNode::Op::Le: ss << "<="; break;
                case BinaryNode::Op::Eq: ss << "=="; break;
                case BinaryNode::Op::Ne: ss << "!="; break;
                case BinaryNode::Op::And: ss << "and"; break;
                case BinaryNode::Op::Or: ss << "or"; break;
                case BinaryNode::Op::In: ss << "in"; break;
                case BinaryNode::Op::NotIn: ss << "not in"; break;
                case BinaryNode::Op::Bind: ss << "=>"; break;
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
        case ASTKind::LiteralPayload: {
            auto &payload = static_cast<const LiteralPayloadNode &>(node);
            if (payload.payload_kind == LiteralPayloadNode::Kind::Set) {
                ss << line_prefix << "Set";
            } else {
                ss << line_prefix << "Interval " << (payload.lower_closed ? "[" : "(") << (payload.upper_closed ? "]" : ")");
            }
            if (node.type) { ss << " : "; print_type(ss, *node.type); }
            ss << "\n";
            for (size_t i = 0; i < payload.elements.size(); i++) {
                const bool last = i + 1 == payload.elements.size();
                print_expr(ss, *payload.elements[i], child_prefix + (last ? "└── " : "├── "), child_prefix + (last ? "    " : "│   "));
            }
            break;
        }
        case ASTKind::MatchExpr: {
            auto &match = static_cast<const MatchExprNode &>(node);
            ss << line_prefix << "Match";
            if (node.type) { ss << " : "; print_type(ss, *node.type); }
            ss << "\n";
            if (match.target) print_expr(ss, *match.target, child_prefix + "└── ", child_prefix + "    ");
            break;
        }
        case ASTKind::DotExpr: {
            auto &d = static_cast<const DotExprNode &>(node);
            ss << line_prefix << "DotExpr";
            if (node.type) { ss << " : "; print_type(ss, *node.type); }
            ss << "\n";
            if (d.expr) print_expr(ss, *d.expr, child_prefix + "├── ", child_prefix + "│   ");
            if (d.rhs) {
                ss << child_prefix << "└── " << d.rhs->id << "\n";
            }
            break;
        }
        case ASTKind::NativeFuncCall: {
            ss << line_prefix << "NativeFuncCall\n";
            auto &nc = static_cast<const NativeFuncCallExpr &>(node);
            if (nc.expr) print_expr(ss, *nc.expr, child_prefix + "├── ", child_prefix + "│   ");
            if (nc.suffix) print_expr(ss, *nc.suffix, child_prefix + "└── ", child_prefix + "    ");
            break;
        }
        case ASTKind::ArrayLiteral: {
            auto &al = static_cast<const ArrayLiteralNode &>(node);
            ss << line_prefix << "ArrayLiteral";
            if (node.type) { ss << " : "; print_type(ss, *node.type); }
            ss << "\n";
            for (size_t i = 0; i < al.exprs.size(); i++) {
                const bool last = i + 1 == al.exprs.size();
                auto kid_pref = child_prefix + (last ? "└── " : "├── ");
                auto kid_cont = child_prefix + (last ? "    " : "│   ");
                print_expr(ss, *al.exprs[i], kid_pref, kid_cont);
            }
            break;
        }
        case ASTKind::TupleLiteral: {
            auto &tl = static_cast<const TupleLiteralNode &>(node);
            ss << line_prefix << "TupleLiteral";
            if (node.type) { ss << " : "; print_type(ss, *node.type); }
            ss << "\n";
            for (size_t i = 0; i < tl.exprs.size(); i++) {
                const bool last = i + 1 == tl.exprs.size();
                auto kid_pref = child_prefix + (last ? "└── " : "├── ");
                auto kid_cont = child_prefix + (last ? "    " : "│   ");
                print_expr(ss, *tl.exprs[i], kid_pref, kid_cont);
            }
            break;
        }
        case ASTKind::TupleGetExpr: {
            auto &tg = static_cast<const TupleGetExprNode &>(node);
            ss << line_prefix << "TupleGet ." << static_cast<int>(tg.i);
            if (node.type) { ss << " : "; print_type(ss, *node.type); }
            ss << "\n";
            if (tg.tup) print_expr(ss, *tg.tup, child_prefix + "└── ", child_prefix + "    ");
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
        case ASTKind::ContinueStmt: {
            ss << line_prefix << "Continue\n";
            break;
        }
        case ASTKind::LoopStmt: {
            auto &ls = static_cast<const LoopStmtNode &>(node);
            ss << line_prefix << "Loop\n";
            if (ls.expr) print_expr(ss, *ls.expr, child_prefix + "├── ", child_prefix + "│   ");
            for (size_t i = 0; i < ls.body.size(); i++) {
                const bool last = i + 1 == ls.body.size();
                auto kid_pref = child_prefix + (last ? "└── " : "├── ");
                auto kid_cont = child_prefix + (last ? "    " : "│   ");
                print_stmt(ss, *ls.body[i], kid_pref, kid_cont);
            }
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
        case ASTKind::NativeFuncDecl: {
            auto &nf = static_cast<const NativeFuncDeclNode &>(node);
            ss << line_prefix << "NativeFunc " << nf.func_id << " : " << nf.symbol;
            if (nf.return_type) {
                ss << " -> ";
                print_type(ss, *nf.return_type);
            }
            ss << "\n";
            if (nf.params) print_stmt(ss, *nf.params, child_prefix + "└── ", child_prefix + "    ");
            break;
        }
        case ASTKind::ImportStmt: {
            auto &imp = static_cast<const ImportStmtNode &>(node);
            ss << line_prefix << "Import " << imp.name << "\n";
            break;
        }
        case ASTKind::SymDecl: {
            auto &sym = static_cast<const SymDeclNode &>(node);
            ss << line_prefix << "sym ";
            for (size_t i = 0; i < sym.ids.size(); ++i) {
                if (i > 0) ss << ", ";
                ss << sym.ids[i];
            }
            ss << "\n";
            break;
        }
        case ASTKind::TypeDecl: {
            auto &declaration = static_cast<const TypeDeclNode &>(node);
            ss << line_prefix << "Type " << declaration.name << "\n";
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
        case ASTKind::DotExpr:
        case ASTKind::NativeFuncCall:
        case ASTKind::LiteralPayload:
        case ASTKind::MatchExpr:
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
    ss << "Module " << module.name;
    if (!module.lib_name.empty()) {
        ss << " [lib: " << module.lib_name << "]";
    }
    ss << "\n";

    const size_t total = module.decls.size() + module.native_funcs.size();
    size_t idx = 0;

    for (const auto &d : module.decls) {
        const bool last = ++idx == total;
        auto pref = last ? "└── " : "├── ";
        auto cont = last ? "    " : "│   ";
        print_stmt(ss, *d, pref, cont);
    }
    for (const auto &n : module.native_funcs) {
        const bool last = ++idx == total;
        auto pref = last ? "└── " : "├── ";
        auto cont = last ? "    " : "│   ";
        print_stmt(ss, *n, pref, cont);
    }
    return ss.str();
}

} // namespace lmx
