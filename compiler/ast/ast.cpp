//
// Created by meian on 2026/4/3.
//

#include "ast.hpp"
#include <ranges>
#include <sstream>
#include <utility>

#include "ast_printer.hpp"

using namespace lmx;

lmx::TypePool lmx::type_pool;

std::shared_ptr<Type> TypePool::basic(const runtime::ValueKind v) noexcept {
    for (const auto& t : types)
        if (t->kind == TypeKind::Basic && static_cast<BasicType*>(t.get())->type == v) return t;
    auto ty = std::make_shared<BasicType>(v);
    types.push_back(ty);
    return ty;
}

std::shared_ptr<Type> TypePool::string() noexcept {
    for (const auto& t : types) if (t->kind == TypeKind::String) return t;
    auto ty = std::make_shared<StringType>();
    types.push_back(ty);
    return ty;
}

std::shared_ptr<Type> TypePool::array(const std::shared_ptr<Type>& type, const size_t len) noexcept {
    for (const auto& t : types)
        if (t->kind == TypeKind::Array) {
            const auto* a = static_cast<ArrayType*>(t.get());
            if (a->len == len && a->type->equals(type.get())) return t;
        }
    auto ty = std::make_shared<ArrayType>(type, len);
    types.push_back(ty);
    return ty;
}

std::shared_ptr<Type> TypePool::function(std::vector<std::shared_ptr<Type>> params,
                                         std::shared_ptr<Type> ret) noexcept {
    for (const auto& t : types)
        if (t->kind == TypeKind::Function) {
            const auto* f = static_cast<FunctionType*>(t.get());
            if (!f->ret_ty->equals(ret.get()) || f->params_ty.size() != params.size()) continue;
            bool ok = true;
            for (size_t i = 0; i < params.size(); i++)
                if (!f->params_ty[i]->equals(params[i].get())) { ok = false; break; }
            if (ok) return t;
        }
    auto ty = std::make_shared<FunctionType>(std::move(params), std::move(ret));
    types.push_back(ty);
    return ty;
}

std::shared_ptr<Type> TypePool::native_function(std::vector<std::shared_ptr<Type>> params,
                                                std::shared_ptr<Type> ret,
                                                std::string name) noexcept {
    for (const auto& t : types)
        if (t->kind == TypeKind::NativeFunction) {
            const auto* f = static_cast<NativeFunctionType*>(t.get());
            if (f->name != name || !f->ret_ty->equals(ret.get()) || f->params_ty.size() != params.size()) continue;
            bool ok = true;
            for (size_t i = 0; i < params.size(); i++)
                if (!f->params_ty[i]->equals(params[i].get())) { ok = false; break; }
            if (ok) return t;
        }
    auto ty = std::make_shared<NativeFunctionType>(std::move(params), std::move(ret), std::move(name));
    types.push_back(ty);
    return ty;
}

std::shared_ptr<Type> TypePool::named(std::string name) noexcept {
    for (const auto& t : types)
        if (t->kind == TypeKind::Named && static_cast<NamedType*>(t.get())->name == name) return t;
    auto ty = std::make_shared<NamedType>(std::move(name));
    types.push_back(ty);
    return ty;
}

std::shared_ptr<Type> TypePool::module(std::string target_path, std::vector<hir::Scope::Var> exports) noexcept {
    for (const auto& t : types)
        if (t->kind == TypeKind::Module && static_cast<ModuleType*>(t.get())->target_path == target_path) return t;
    auto ty = std::make_shared<ModuleType>(std::move(target_path), std::move(exports));
    types.push_back(ty);
    return ty;
}

std::shared_ptr<Type> TypePool::unknown() noexcept {
    for (const auto& t : types) if (t->kind == TypeKind::Unknown) return t;
    auto ty = std::make_shared<UnknownType>();
    types.push_back(ty);
    return ty;
}

std::shared_ptr<Type> TypePool::none() noexcept {
    for (const auto& t : types) if (t->kind == TypeKind::None) return t;
    auto ty = std::make_shared<NoneType>();
    types.push_back(ty);
    return ty;
}

Type::~Type() = default;

BasicType::~BasicType() = default;

ArrayType::~ArrayType() = default;

bool ModuleType::equals(Type *other) const noexcept {
    if (is_null_type(other)) return false;
    if (other->kind != this->kind) return false;
    if (reinterpret_cast<ModuleType *>(other)->target_path != this->target_path) return false;
    return true;
}

bool BasicType::equals(Type *other) const noexcept {
    if (!other) return false;
    if (other->kind != this->kind) return false;
    const auto *o = reinterpret_cast<BasicType *>(other);
    if (o->type != this->type) return false;
    return true;
}

bool ArrayType::equals(Type *other) const noexcept {
    if (!other) return false;
    if (other->kind != this->kind) return false;
    const auto *o = reinterpret_cast<ArrayType *>(other);
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
    const auto *o = reinterpret_cast<FunctionType *>(other);
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
    if (!kind) return "?";
    std::ostringstream ss;
    AstPrinter::print_type(ss, *kind);
    return ss.str();
}

NativeFunctionType::NativeFunctionType(std::vector<std::shared_ptr<Type>> params_ty, std::shared_ptr<Type> ret_ty, std::string name) noexcept
    : Type(TypeKind::NativeFunction), params_ty(std::move(params_ty)), ret_ty(std::move(ret_ty)), name(std::move(name)) {
}

bool NativeFunctionType::equals(Type *other) const noexcept {
    if (is_null_type(other)) return false;
    if (other->kind != this->kind) return false;

    const auto *o = reinterpret_cast<NativeFunctionType *>(other);
    if (params_ty.size() != o->params_ty.size()) return false;
    for (size_t i = 0; i < params_ty.size(); i++) {
        if (!params_ty[i]->equals(o->params_ty[i].get())) return false;
    }
    if (!ret_ty->equals(o->ret_ty.get())) return false;
    return true;
}

bool NativeFunctionType::have_va_list() const noexcept {
    for (const auto& p : params_ty) {
        if (p->kind == TypeKind::Basic) {
            if (const auto t = reinterpret_cast<BasicType *>(p.get());
                t->type == runtime::ValueKind::C_VaList) return true;

        }
    }
    return false;
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
    : ExprNode(ASTKind::Binary, line, col), op(op), lhs(std::move(lhs)), rhs(std::move(rhs)) {
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
    :
    StmtNode(ASTKind::FuncImpl, line, col),
    func_id(std::move(func_id)),
    params(std::move(params)),
    return_type(std::move(return_type)),
    block(std::move(block)) {}

IfExprNode::IfExprNode(const size_t line, const size_t col,
    std::shared_ptr<ExprNode> cond,
    std::shared_ptr<ExprNode> then,
    std::shared_ptr<ExprNode> els) noexcept
    :
    ExprNode(ASTKind::IfExpr, line, col),
    cond(std::move(cond)),
    then(std::move(then)),
    els(std::move(els)) {}

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
    for (auto &type: params->stmts | std::views::values) {
        params_ty.push_back(type);
    }
    return std::reinterpret_pointer_cast<FunctionType>(type_pool.function(std::move(params_ty), return_type));
}

DotExprNode::DotExprNode(const size_t line, const size_t col, std::shared_ptr<ExprNode> expr, std::shared_ptr<IdentifierNode> rhs) noexcept
    : ExprNode(ASTKind::DotExpr, line, col), expr(std::move(expr)), rhs(std::move(rhs)) {}

NativeFuncDeclNode::NativeFuncDeclNode(
    const size_t line,
    const size_t col,
    decltype(func_id) func_id,
    std::shared_ptr<ParamsDeclNode> params,
    std::shared_ptr<Type> return_type,
    std::string symbol) noexcept
    : StmtNode(ASTKind::NativeFuncDecl, line, col), func_id(std::move(func_id)), params(std::move(params)), return_type(std::move(return_type)), symbol(std::move(symbol)) {}

std::shared_ptr<NativeFunctionType> NativeFuncDeclNode::make_type() noexcept {
    decltype(NativeFunctionType::params_ty) params_ty;
    for (const auto &type: params->stmts | std::views::values) {
        params_ty.push_back(type);
    }
    return std::reinterpret_pointer_cast<NativeFunctionType>(
        type_pool.native_function(std::move(params_ty), return_type, symbol));
}

NativeFuncCallExpr::NativeFuncCallExpr(const SuffixParenNode *sp) noexcept
    : ExprNode(ASTKind::NativeFuncCall, sp->line, sp->col), expr(sp->expr), suffix(sp->suffix) {}

LoopStmtNode::LoopStmtNode(const size_t line, const size_t col, decltype(expr) expr, std::vector<std::shared_ptr<StmtNode> > body) noexcept
    : StmtNode(ASTKind::LoopStmt, line, col), expr(std::move(expr)), body(std::move(body)) {}

ContinueStmtNode::ContinueStmtNode(const size_t line, const size_t col) noexcept
    : StmtNode(ASTKind::ContinueStmt, line, col) {}

ImportStmtNode::ImportStmtNode(size_t line, size_t col, decltype(name) name) noexcept
    : StmtNode(ASTKind::ImportStmt, line, col), name(std::move(name)) {}

SymDeclNode::SymDeclNode(size_t line, size_t col, std::vector<std::string> ids) noexcept
    : StmtNode(ASTKind::SymDecl, line, col), ids(std::move(ids)) {}
