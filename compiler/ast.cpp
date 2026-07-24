//
// Created by meian on 2026/4/3.
//

#include "ast.hpp"
#include "ast.hpp"
#include <ranges>
#include <sstream>
#include <utility>

#include "ast_printer.hpp"

using namespace lmx;

Type::~Type() = default;

BasicType::~BasicType() = default;

ArrayType::~ArrayType() = default;

bool BasicType::equals(Type *other) const noexcept {
    if (!other) return false;
    if (other->kind != this->kind) return false;
    const auto *o = static_cast<BasicType *>(other);
    if (o->type != this->type) return false;
    return true;
}

bool ArrayType::equals(Type *other) const noexcept {
    if (!other) return false;
    if (other->kind != this->kind) return false;
    const auto *o = static_cast<ArrayType *>(other);
    if (this->type->equals(o->type.get())) return false;
    if (this->len != o->len) return false;
    return true;
}

bool UnknownType::equals(Type *other) const noexcept {
    return false;
}

bool StringType::equals(Type *other) const noexcept {
    if (!other) return false;
    if (other->kind != this->kind) return false;
    return true;
}

FunctionType::FunctionType(std::vector<std::shared_ptr<Type> > params_ty, std::shared_ptr<Type> ret_ty) noexcept
    : Type(TypeKind::Function), params_ty(std::move(params_ty)), ret_ty(std::move(ret_ty)) {
}

bool FunctionType::equals(Type *other) const noexcept {
    if (!other) return false;
    if (other->kind != this->kind) return false;
    const auto *o = static_cast<FunctionType *>(other);
    const auto params_len = params_ty.size();
    if (params_len != o->params_ty.size()) return false;
    for (size_t i = 0; i < params_len; i++) {
        if (!params_ty[i]->equals(o->params_ty[i].get())) return false;
    }
    if (!ret_ty->equals(o->ret_ty.get())) return false;
    return true;
}

bool NoneType::equals(Type *other) const noexcept {
    if (!other) return true;
    if (other->kind != this->kind) return false;
    return true;
}

std::string Type::to_string(const Type* kind) noexcept {
    std::ostringstream ss;
    AstPrinter::print_type(ss, *kind);
    return ss.str();
}

std::string BinaryNode::op_to_string(const Op op) noexcept {
    switch (op) {
    case Op::Add:
        return "+";
    case Op::Sub:
        return "-";
    case Op::Mul:
        return "*";
    case Op::Div:
        return "/";
    case Op::Mod:
        return "%";
    case Op::Pow:
        return "^";
    case Op::Gt:
        return ">";
    case Op::Ge:
        return ">=";
    case Op::Lt:
        return "<";
    case Op::Le:
        return "<=";
    case Op::Eq:
        return "==";
    case Op::Ne:
        return "!=";
    case Op::And:
        return "and";
    case Op::Or:
        return "or";
    case Op::Dot:
        return {};
        break;
    }
    return {};
}

ASTNode::ASTNode(const ASTKind kind, const size_t line, const size_t col) noexcept
    : kind(kind), line(line), col(col) {
}

ExprNode::ExprNode(const ASTKind kind, const size_t line, const size_t col) noexcept
    : ASTNode(kind, line, col) {
}

StmtNode::StmtNode(const ASTKind kind, const size_t line, const size_t col) noexcept
    : ASTNode(kind, line, col) {
}

Module::Module(std::string name, decltype(decls) decls) noexcept
        : name(std::move(name)), decls(std::move(decls)) {
    const auto decls_len = this->decls.size();
    for (auto i = 0; i < decls_len; i++) {
        const auto decl = this->decls[i];
        if (decl->kind == ASTKind::FuncImpl) {
            // auto node = std::reinterpret_pointer_cast<FuncImplNode>(decl);
            this->decls.erase(this->decls.begin() + i);
            this->decls.insert(this->decls.begin(), decl);
        }
    }
}

ExprStmtNode::ExprStmtNode(const size_t line, const size_t col, std::shared_ptr<ExprNode> expr) noexcept
    : StmtNode(ASTKind::ExprStmt, line, col), expr(std::move(expr)) {
}

ExprsNode::ExprsNode(const size_t line, const size_t col, std::vector<std::shared_ptr<ExprNode> > exprs) noexcept
    : ExprNode(ASTKind::Exprs, line, col), exprs(std::move(exprs)) {
}

LiteralNode::LiteralNode(const size_t line, const size_t col, std::string val, const Kind kind) noexcept
    : ExprNode(ASTKind::Literal, line, col), val(std::move(val)), kind(kind) {
}

IdentifierNode::IdentifierNode(const size_t line, const size_t col, std::string id) noexcept
    : ExprNode(ASTKind::Identifier, line, col), id(std::move(id)) {
}

SuffixParenNode::SuffixParenNode(const size_t line, const size_t col, std::shared_ptr<ExprNode> expr,
                                 std::shared_ptr<ExprsNode> suffix) noexcept
    : ExprNode(ASTKind::SuffixParen, line, col), expr(std::move(expr)), suffix(std::move(suffix)) {
}

SuffixBracketNode::SuffixBracketNode(const size_t line, const size_t col, std::shared_ptr<ExprNode> expr,
                                     std::shared_ptr<ExprNode> suffix) noexcept
    : ExprNode(ASTKind::SuffixBracket, line, col), expr(std::move(expr)), suffix(std::move(suffix)) {
}

UnaryNode::UnaryNode(const size_t line, const size_t col, const Op op, std::shared_ptr<ExprNode> expr) noexcept
    : ExprNode(ASTKind::Unary, line, col), op(op), expr(std::move(expr)) {
}
BinaryNode::BinaryNode(const size_t line, const size_t col, std::shared_ptr<ExprNode> lhs, const Op op,
                       std::shared_ptr<ExprNode> rhs) noexcept
    : ExprNode(ASTKind::Binary, line, col), lhs(std::move(lhs)), rhs(std::move(rhs)), op(op) {
}

ReturnNode::ReturnNode(const size_t line, const size_t col, std::shared_ptr<ExprNode> expr) noexcept
    : StmtNode(ASTKind::Return, line, col), expr(std::move(expr)) {
}

ReturnNode::ReturnNode(const size_t line, const size_t col, const std::shared_ptr<ExprStmtNode> &s) noexcept
    : StmtNode(ASTKind::Return, line, col), expr(s->expr) {
}

TailReturnNode::TailReturnNode(const size_t line, const size_t col, std::shared_ptr<ExprNode> expr) noexcept
    : StmtNode(ASTKind::TailReturn, line, col), expr(std::move(expr)) {}

TailReturnNode::TailReturnNode(const size_t line, const size_t col, const std::shared_ptr<ExprStmtNode> &s) noexcept
    : StmtNode(ASTKind::TailReturn, line, col), expr(s->expr) {}

BlockExprNode::BlockExprNode(const size_t line, const size_t col,
                             std::vector<std::shared_ptr<StmtNode> > stmts) noexcept
    : ExprNode(ASTKind::Block, line, col), stmts(std::move(stmts)) {
}

BreakStmtNode::BreakStmtNode(const size_t line, const size_t col) noexcept
    : StmtNode(ASTKind::BreakStmt, line, col) {}

ParamsDeclNode::ParamsDeclNode(const size_t line, const size_t col,
                               decltype(stmts) stmts) noexcept
    : StmtNode(ASTKind::ParamsDeclNode, line, col), stmts(std::move(stmts)) {
}

FuncImplNode::FuncImplNode(const size_t line, const size_t col,
    decltype(func_id) func_id,
    std::shared_ptr<ParamsDeclNode> params,
    std::shared_ptr<Type> return_type,
    std::shared_ptr<BlockExprNode> block) noexcept
    : StmtNode(ASTKind::FuncImpl, line, col), func_id(std::move(func_id)),params(std::move(params)),return_type(std::move(return_type)), block(std::move(block)) {}

IfExprNode::IfExprNode(const size_t line, const size_t col, std::shared_ptr<ExprNode> cond, std::shared_ptr<ExprNode> then, std::shared_ptr<ExprNode> els) noexcept
    : ExprNode(ASTKind::IfExpr, line, col), cond(std::move(cond)), then(std::move(then)), els(std::move(els)) {}

AsExprNode::AsExprNode(const size_t line, const size_t col,
                       std::shared_ptr<ExprNode> expr,
                       std::shared_ptr<Type> cast_type) noexcept
    : ExprNode(ASTKind::AsExpr, line, col), expr(std::move(expr)), cast_type(std::move(cast_type)) {}

VarDeclNode::VarDeclNode(const size_t line, const size_t col, decltype(id) id, std::shared_ptr<Type> type, const bool is_mutable) noexcept
    : StmtNode(ASTKind::VarDecl, line, col), id(std::move(id)), type(std::move(type)), is_mutable(is_mutable) {}

AssignStmtNode::AssignStmtNode(const size_t line, const size_t col,
                               std::shared_ptr<ExprNode> lhs,
                               std::shared_ptr<ExprNode> rhs) noexcept
    : StmtNode(ASTKind::AssignStmt, line, col), lhs(std::move(lhs)), rhs(std::move(rhs)) {}

std::shared_ptr<FunctionType> FuncImplNode::make_type() noexcept {
    decltype(FunctionType::params_ty) params_ty;
    for (const auto &type: params->stmts | std::views::values) {
        params_ty.push_back(type);
    }
    return std::make_shared<FunctionType>(params_ty, return_type);
}
