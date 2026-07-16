//
// Created by meian on 2026/7/5.
//

#include "hir.hpp"

#include <functional>
#include <ranges>

#include "error.hpp"

using namespace lmx;
using namespace lmx::hir;

Scope::Scope(std::string name) noexcept : name(std::move(name)) {}

std::optional<Scope::Var *> HirContext::find_var(const std::string &name) noexcept {
    for (auto& i : scope_stack | std::views::reverse) {
        for (auto& j : i.vars) {
            if (j.name == name) return &j;
        }
    }
    return std::nullopt;
}

HirContext::HirContext() noexcept {
    scope_stack.emplace_back("@GLOBAL");
}

std::shared_ptr<Type> HirContext::inference_type(ExprNode* type) noexcept {
    if (!type) return nullptr;
    switch (type->kind) {
    case ASTKind::Literal: {
        const auto node = reinterpret_cast<LiteralNode*>(type);
        switch (node->kind) {
        case LiteralNode::Kind::Integer: {
            return std::make_shared<BasicType>(runtime::ValueKind::Int);
        }
        case LiteralNode::Kind::Float: {
            return std::make_shared<BasicType>(runtime::ValueKind::Fraction);
        }
        case LiteralNode::Kind::String: {
            return std::make_shared<StringType>();
        }
        case LiteralNode::Kind::Boolean: {
            return std::make_shared<BasicType>(runtime::ValueKind::Bool);
        }
        }
    }
    case ASTKind::Identifier: {
        const auto node = reinterpret_cast<IdentifierNode*>(type);
        if (node->type && node->type->kind != TypeKind::Unknown) return node->type;
        if (find_var(node->id).has_value())return (*find_var(node->id))->type;
        break;
    }
    case ASTKind::Unary: {
        const auto node = reinterpret_cast<UnaryNode*>(type);
        if (const auto t = inference_type(node->expr.get());
            t &&
            t->kind == TypeKind::Basic
            ) {
            if (const auto t2 = std::reinterpret_pointer_cast<BasicType>(t)->type;
                t2 == runtime::ValueKind::Int || t2 == runtime::ValueKind::Fraction) {

                return std::make_shared<BasicType>(t2);
            }
        }
        break;
    }
    case ASTKind::Binary: {
        const auto node = reinterpret_cast<BinaryNode*>(type);
        auto left_ty = inference_type(node->lhs.get());
        const auto right_ty = inference_type(node->rhs.get());
        if (left_ty->equals(right_ty.get())) return left_ty;
        break;
    }
    case ASTKind::Block: {
        const auto node = reinterpret_cast<BlockExprNode*>(type);
        if (node->stmts.back()->kind == ASTKind::TailReturn) {
            const auto tail_ret = std::reinterpret_pointer_cast<TailReturnNode>(node->stmts.back());
            if (tail_ret->expr &&
                !Type::is_null_type(tail_ret->expr->type.get()) &&
                tail_ret->expr->type->kind != TypeKind::Unknown

                ) return tail_ret->expr->type;
            return inference_type(tail_ret->expr.get());
        } //否则就是Block没有返回值
        break;
    }
    case ASTKind::SuffixParen: {
        const auto node = reinterpret_cast<SuffixParenNode*>(type);
        const auto left_ty = std::reinterpret_pointer_cast<FunctionType>(inference_type(node->expr.get()));
        return left_ty->ret_ty;
        break;
    }
    case ASTKind::SuffixBracket: {
        const auto node = reinterpret_cast<SuffixBracketNode*>(type);
        const auto left_t = inference_type(node->expr.get());
        if (left_t->kind == TypeKind::Array) {
            return std::reinterpret_pointer_cast<ArrayType>(left_t)->type;
        }
        break;
    }
    case ASTKind::IfExpr: {
        const auto node = reinterpret_cast<IfExprNode*>(type);

        return inference_type(node->then.get());
    }
    default: std::unreachable();
    }
    return std::make_shared<NoneType>();
}


// void HirContext::reset() noexcept {
//     scope_stack.clear();
// }

void HirContext::check_module(const Module *mod) noexcept {
    // reset();
    for (auto& node : mod->decls) {
        check_stmt(node.get());
    }
}
void HirContext::check_expr(ExprNode *expr) noexcept {
    switch (expr->kind) {
    case ASTKind::Literal: {
        expr->type = inference_type(expr);
        break;
    }
    case ASTKind::Identifier: {
        auto* node = reinterpret_cast<IdentifierNode*>(expr);
        auto re = find_var(node->id);
        if (!re.has_value())
            throw_error(ErrorType::Analysis, "undefined var `" + node->id + "`", node->line, node->col);
        node->type = (*re)->type;
        break;
    }
    case ASTKind::Unary: {
        auto* node = reinterpret_cast<UnaryNode*>(expr);
        const auto type = inference_type(node->expr.get());
        if (type->kind != TypeKind::Basic) goto unary_type_mismatch;
        if (const auto t2 = std::reinterpret_pointer_cast<BasicType>(type);
            t2->type != runtime::ValueKind::Int && t2->type != runtime::ValueKind::Fraction) goto unary_type_mismatch;
        node->type = type;
        break;
        unary_type_mismatch:
        throw_error(ErrorType::Analysis, "unary`-` cannot applied to this type", expr->line, expr->col);
        break;
    }
    case ASTKind::Binary: {
        auto* node = reinterpret_cast<BinaryNode*>(expr);
        check_expr(node->lhs.get());
        check_expr(node->rhs.get());
        const auto lty = inference_type(node->lhs.get());
        const auto rty = inference_type(node->rhs.get());
        node->lhs->type = lty;
        node->rhs->type = rty;
        if (!lty->equals(rty.get())) throw_error(ErrorType::Analysis, "binary operation type mismatch", expr->line, expr->col);

        if (lty->kind != TypeKind::Basic) goto binary_type_mismatch;
        if (const auto t2 = std::reinterpret_pointer_cast<BasicType>(lty);
            t2->type != runtime::ValueKind::Int && t2->type != runtime::ValueKind::Fraction) goto binary_type_mismatch;
        // static const std::map<runtime::ValueKind, std::unordered_map<std::string, runtime::ValueKind>> op_types = {
        //     {"+", runtime::ValueKind::Int},
        // };
        // todo!
        node->type = lty;
        break;
        binary_type_mismatch:
        throw_error(ErrorType::Analysis, "binary operation cannot applied to this type", expr->line, expr->col);
        break;
    }
    case ASTKind::Block: {
        auto node = reinterpret_cast<BlockExprNode*>(expr);
        for (auto& s : node->stmts) check_stmt(s.get());
        expr->type = inference_type(expr);
        break;
    }
    case ASTKind::SuffixParen: {
        const auto node = reinterpret_cast<SuffixParenNode*>(expr);
        check_expr(node->expr.get());
        const auto left = inference_type(node->expr.get());
        if (left->kind != TypeKind::Function) {
            throw_error(ErrorType::Analysis, "not a function type", node->line, node->col);
            break;
        }
        if (std::reinterpret_pointer_cast<FunctionType>(left)->params_ty.size() != node->suffix->exprs.size()) {
            throw_error(ErrorType::Analysis, "mismatch args count in function calling", node->line, node->col);
        }
        node->type = std::reinterpret_pointer_cast<FunctionType>(left)->ret_ty;
        break;
    }
    case ASTKind::SuffixBracket: {
        const auto node = reinterpret_cast<SuffixBracketNode*>(expr);
        check_expr(node->expr.get());
        const auto left = inference_type(node->expr.get());
        if (left->kind != TypeKind::Array) {
            throw_error(ErrorType::Analysis, "must be array type", node->line, node->col);
            break;
        }
        node->type = std::reinterpret_pointer_cast<ArrayType>(left)->type;
        break;
    }
    case ASTKind::IfExpr: {
        const auto node = reinterpret_cast<IfExprNode*>(expr);
        check_expr(node->cond.get());
        break;
    }
    default: std::unreachable();
    }
}

void HirContext::check_stmt(StmtNode* stmt) noexcept {
    switch (stmt->kind) {
    case ASTKind::ExprStmt: {
        const auto* node = reinterpret_cast<ExprStmtNode*>(stmt);
        check_expr(node->expr.get());
        break;
    }
    case ASTKind::Exprs:
        break;
    case ASTKind::ParamsDeclNode:
        break;
    case ASTKind::FuncImpl: {
        auto* node = reinterpret_cast<FuncImplNode*>(stmt);
        if (!is_global_scope()) throw_error(ErrorType::Analysis, "function only define in GlobalScope", stmt->line, stmt->col);

        new_global_var(node->func_id, node->make_type());
        auto& ref = scope_stack[0].vars.back();
        Scope scope;
        for (const auto& [name, ty] : node->params->stmts) {
            scope.vars.emplace_back(name, ty, true);
        }
        scope.name = "";
        scope.return_type = node->return_type;
        scope_stack.push_back(scope);

        check_expr(node->block.get());
        if (Type::is_null_type(node->return_type.get())) node->return_type = node->block->type;
        scope_stack.pop_back();
        ref.type = node->make_type();
        break;
    }
    case ASTKind::Return: {
        auto node = reinterpret_cast<ReturnNode*>(stmt);
        check_expr(node->expr.get());
        node->expr->type = inference_type(node->expr.get());
        break;
    }
    case ASTKind::TailReturn: {
        auto node = reinterpret_cast<TailReturnNode*>(stmt);
        check_expr(node->expr.get());
        // node->expr->type = inference_type(node->expr.get());
        break;
    }
    case ASTKind::VarDecl: {
        const auto node = reinterpret_cast<VarDeclNode*>(stmt);
        check_expr(node->init_value.get());
        if (Type::is_null_type(node->type.get())) {
            if (!node->init_value) {
                throw_error(ErrorType::Analysis, "the var `" + node->id + "` type not found", node->line, node->col);
            } else node->type = inference_type(node->init_value.get());
        } else {
            if (!node->type->equals(node->init_value->type.get())) {
                throw_error(ErrorType::Analysis, "the var `" + node->id + "` type mismatch with the initialization type", node->line, node->col);
            }
        }
        new_cur_scope_var(node->id, node->type);
        break;
    }
    case ASTKind::BreakStmt: {
        break;
    }
    default: std::unreachable();
    }
}

bool HirContext::is_global_scope() noexcept {
    return scope_stack.size() == 1;
}

void HirContext::new_var(std::string name, std::shared_ptr<Type> type, Scope *scope) noexcept {
    scope->vars.emplace_back(std::move(name), std::move(type));
}

void HirContext::new_cur_scope_var(std::string name, std::shared_ptr<Type> type) noexcept {
    scope_stack.back().vars.emplace_back(std::move(name), std::move(type));
}

void HirContext::new_global_var(std::string name, std::shared_ptr<Type> type) noexcept {
    scope_stack[0].vars.emplace_back(std::move(name), std::move(type));
}

//
// Scope *HirContext::parse_scope(ExprNode *node) noexcept {
//
//     switch (node->kind) {
//     case ASTKind::Binary: {
//         const auto id = reinterpret_cast<BinaryNode*>(node);
//         auto left = parse_scope(id->lhs.get());
//
//         // todo!
//         return std::nullopt;
//         break;
//     }
//     case ASTKind::Identifier: {
//         const auto id = reinterpret_cast<IdentifierNode*>(node);
//         return &scope_stack.back();
//     }
//     default: std::unreachable();
//     }
// }

