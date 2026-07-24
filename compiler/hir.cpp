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

Scope::Scope(const ScopeType scope) noexcept : scope(scope) {}

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
    if (!type) return std::make_shared<UnknownType>();
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
        break;
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
        if (const auto node = reinterpret_cast<BlockExprNode*>(type);
            node->stmts.back()->kind == ASTKind::TailReturn)
        {
            const auto tail_ret = std::reinterpret_pointer_cast<TailReturnNode>(node->stmts.back());
            if (tail_ret->expr &&
                !Type::is_null_type(tail_ret->expr->type.get()) &&
                tail_ret->expr->type->kind != TypeKind::Unknown

                ) return tail_ret->expr->type;
            return inference_type(tail_ret->expr.get());
        } //否则就是Block没有返回值
        return std::make_shared<NoneType>();
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
    case ASTKind::AsExpr: {
        const auto node = reinterpret_cast<AsExprNode*>(type);
        return node->cast_type;
    }
    default: std::unreachable();
    }
    return std::make_shared<UnknownType>();
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
    if (!expr) return;
    switch (expr->kind) {
    case ASTKind::Literal: {
        expr->type = inference_type(expr);
        break;
    }
    case ASTKind::Identifier: {
        auto* node = reinterpret_cast<IdentifierNode*>(expr);
        const auto re = find_var(node->id);
        if (!re.has_value()) {
            throw_error(ErrorType::Analysis, "undefined var `" + node->id + "`", node->line, node->col);
            break;
        }
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
        if (!lty->equals(rty.get())) {
            throw_error(
                ErrorType::Analysis,
                "binary operation type mismatch, (" +
                Type::to_string(lty.get()) + " " +
                BinaryNode::op_to_string(node->op) + " " + Type::to_string(rty.get()) + ")", expr->line, expr->col);
            break;
        }
        if (lty->kind != TypeKind::Basic) goto binary_type_mismatch;
        //if (const auto t2 = std::reinterpret_pointer_cast<BasicType>(lty);
         //   t2->type != runtime::ValueKind::Int && t2->type != runtime::ValueKind::Fraction) goto binary_type_mismatch;
        static const std::map<runtime::ValueKind, std::map<BinaryNode::Op, runtime::ValueKind>> op_types = {
            {runtime::ValueKind::Int, {
                {BinaryNode::Op::Add, runtime::ValueKind::Int},
                {BinaryNode::Op::Sub, runtime::ValueKind::Int},
                {BinaryNode::Op::Mul, runtime::ValueKind::Int},
                {BinaryNode::Op::Div, runtime::ValueKind::Fraction},
                {BinaryNode::Op::Mod, runtime::ValueKind::Int},
                {BinaryNode::Op::Pow, runtime::ValueKind::Int},
                {BinaryNode::Op::Eq, runtime::ValueKind::Bool},
                {BinaryNode::Op::Ne, runtime::ValueKind::Bool},
                {BinaryNode::Op::Gt, runtime::ValueKind::Bool},
                {BinaryNode::Op::Ge, runtime::ValueKind::Bool},
                {BinaryNode::Op::Lt, runtime::ValueKind::Bool},
                {BinaryNode::Op::Le, runtime::ValueKind::Bool},
            }},
            {runtime::ValueKind::Fraction, {
                {BinaryNode::Op::Add, runtime::ValueKind::Fraction},
                {BinaryNode::Op::Sub, runtime::ValueKind::Fraction},
                {BinaryNode::Op::Mul, runtime::ValueKind::Fraction},
                {BinaryNode::Op::Div, runtime::ValueKind::Fraction},
                {BinaryNode::Op::Mod, runtime::ValueKind::Fraction},
                {BinaryNode::Op::Pow, runtime::ValueKind::Fraction},
                {BinaryNode::Op::Eq, runtime::ValueKind::Bool},
                {BinaryNode::Op::Ne, runtime::ValueKind::Bool},
                {BinaryNode::Op::Gt, runtime::ValueKind::Bool},
                {BinaryNode::Op::Ge, runtime::ValueKind::Bool},
                {BinaryNode::Op::Lt, runtime::ValueKind::Bool},
                {BinaryNode::Op::Le, runtime::ValueKind::Bool},
            }},
            {runtime::ValueKind::Bool, {
                {BinaryNode::Op::And, runtime::ValueKind::Bool},
                {BinaryNode::Op::Or, runtime::ValueKind::Bool},
            }}
        };
        {
            const auto t2 = std::reinterpret_pointer_cast<BasicType>(lty)->type;
            const auto &type_map = op_types.at(t2);
            if (const auto it = type_map.find(node->op); it != type_map.end()) {
                node->type = std::make_shared<BasicType>(it->second);
            } else {
                goto binary_type_mismatch;
            }
        }
        break;
        binary_type_mismatch:
        throw_error(ErrorType::Analysis, "binary operation cannot applied to this type", expr->line, expr->col);
        break;
    }
    case ASTKind::Block: {
        const auto node = reinterpret_cast<BlockExprNode*>(expr);
        scope_stack.emplace_back(Scope::ScopeType::Block);
        for (auto& s : node->stmts) check_stmt(s.get());
        expr->type = scope_stack.back().return_type;
        scope_stack.pop_back();
        break;
    }
    case ASTKind::SuffixParen: {
        const auto node = reinterpret_cast<SuffixParenNode*>(expr);
        check_expr(node->expr.get());
        const auto left = inference_type(node->expr.get());
        if (Type::is_null_type(left.get())) break;
        if (left->kind != TypeKind::Function) {
            throw_error(ErrorType::Analysis, "not a function type", node->line, node->col);
            break;
        }
        const auto func_ty = std::reinterpret_pointer_cast<FunctionType>(left);
        if (func_ty->params_ty.size() != node->suffix->exprs.size()) {
            throw_error(ErrorType::Analysis, "mismatch args count in function calling", node->line, node->col);
            break;
        }
        const auto len = func_ty->params_ty.size();
        for (auto i = 0; i < len; i++) {
            const auto param = func_ty->params_ty[i];
            check_expr(node->suffix->exprs[i].get());
            if (!param->equals(node->suffix->exprs[i]->type.get())) {
                goto arg_type_mismatch;
            }
        }
        node->type = std::reinterpret_pointer_cast<FunctionType>(left)->ret_ty;
        break;
        arg_type_mismatch:
            throw_error(ErrorType::Analysis, "type mismatch arg in function calling", node->line, node->col);
        break;
    }
    case ASTKind::SuffixBracket: {
        const auto node = reinterpret_cast<SuffixBracketNode*>(expr);
        check_expr(node->expr.get());
        const auto left = inference_type(node->expr.get());
        if (left->kind != TypeKind::Array) {
            throw_error(ErrorType::Analysis, "must be array type but got `" + Type::to_string(left.get()) + "`", node->line, node->col);
            break;
        }
        node->type = std::reinterpret_pointer_cast<ArrayType>(left)->type;
        break;
    }
    case ASTKind::IfExpr: {
        const auto node = reinterpret_cast<IfExprNode*>(expr);
        check_expr(node->cond.get());
        if (node->cond->type->kind != TypeKind::Basic &&
            std::reinterpret_pointer_cast<BasicType>(node->cond->type)->type != runtime::ValueKind::Bool) {
            throw_error(ErrorType::Analysis, "must be bool type but got `" + Type::to_string(node->cond->type.get()), node->line, node->col);
        }
        check_expr(node->then.get());
        if (node->els) {
            check_expr(node->els.get());
            if (!node->then->type->equals(node->els->type.get())) {
                throw_error(ErrorType::Analysis, "if express then and else cannot type mismatch", node->line, node->col);
                break;
            }
        }
        break;
    }
    case ASTKind::AsExpr: {
        // const auto node = reinterpret_cast<AsExprNode*>(expr);
        // check_expr(node->expr.get());
        // node->type = node->cast_type;
        // todo!
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
    //case ASTKind::Exprs:
    //    break;
    //case ASTKind::ParamsDeclNode:
    //    break;
    case ASTKind::FuncImpl: {
        auto* node = reinterpret_cast<FuncImplNode*>(stmt);
        if (!is_global_scope()) throw_error(ErrorType::Analysis, "function only define in GlobalScope", stmt->line, stmt->col);

        new_global_var(node->func_id, node->make_type());
        auto& ref = scope_stack[0].vars.back();
        Scope scope;
        scope.name = node->func_id;
        scope.return_type = node->return_type;
        for (const auto& [name, ty] : node->params->stmts) {
            scope.vars.emplace_back(name, ty, true);
        }
        scope_stack.push_back(scope);

        if (node->block->kind == ASTKind::Block) {
            const auto* block = reinterpret_cast<BlockExprNode*>(node->block.get());
            for (auto& s : block->stmts) {
                check_stmt(s.get());
            }
        } else check_expr(node->block.get());
        if (!node->return_type->equals(scope_stack.back().return_type.get())) {
            node->return_type = scope_stack.back().return_type;
        }
        // if (Type::is_null_type(node->return_type.get())) {
        //     node->return_type = node->block->type;
        // }
        // else {
        //     if (!node->return_type->equals(node->block->type.get())) {
        //         throw_error(ErrorType::Analysis, "return type mismatch in function `" + node->func_id + "`", node->line, node->col);
        //         break;
        //     }
        // }

        scope_stack.pop_back();
        ref.type = node->make_type();
        break;
    }
    case ASTKind::Return: {
        const auto node = reinterpret_cast<ReturnNode*>(stmt);
        if (!node->expr) break;
        check_expr(node->expr.get());
        node->expr->type = inference_type(node->expr.get());
        for (auto& s : scope_stack | std::views::reverse) {
            if (s.scope == Scope::ScopeType::Function) {
                if (Type::is_null_type(s.return_type.get())) {
                    s.return_type = node->expr->type;
                    break;
                }
                if (!s.return_type->equals(node->expr->type.get())) {
                    throw_error(ErrorType::Analysis, "return type mismatch in function `" + s.name + "`", node->line, node->col);
                    goto return_fail_break;
                }
            }
        }
        return_fail_break:
        break;
    }
    case ASTKind::TailReturn: {
        const auto node = reinterpret_cast<TailReturnNode*>(stmt);
        check_expr(node->expr.get());
        scope_stack.back().return_type = node->expr->type;
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
        new_cur_scope_var(node->id, node->type, node->is_mutable);
        break;
    }
    case ASTKind::AssignStmt: {
        const auto node = reinterpret_cast<AssignStmtNode*>(stmt);
        check_expr(node->lhs.get());
        check_expr(node->rhs.get());
        node->lhs->type = inference_type(node->lhs.get());
        node->rhs->type = inference_type(node->rhs.get());
        if (node->lhs->kind != ASTKind::Identifier) {
            throw_error(ErrorType::Analysis, "left side of assignment must be an identifier", node->line, node->col);
            break;
        }
        const auto id = reinterpret_cast<IdentifierNode*>(node->lhs.get());
        const auto var = find_var(id->id);
        if (!var.has_value()) {
            throw_error(ErrorType::Analysis, "undefined var `" + id->id + "`", node->line, node->col);
            break;
        }
        if (!(*var)->is_mut) {
            throw_error(ErrorType::Analysis, "cannot assign to immutable var `" + id->id + "`", node->line, node->col);
            break;
        }
        if (!node->lhs->type->equals(node->rhs->type.get())) {
            throw_error(ErrorType::Analysis, "assignment type mismatch", node->line, node->col);
        }
        break;
    }
    case ASTKind::BreakStmt: {
        break;
    }
    default: std::unreachable();
    }
}

bool HirContext::is_global_scope() const noexcept {
    return scope_stack.size() == 1;
}

void HirContext::new_var(std::string name, std::shared_ptr<Type> type, Scope *scope, bool is_mut) noexcept {
    scope->vars.emplace_back(std::move(name), std::move(type), is_mut);
}

void HirContext::new_cur_scope_var(std::string name, std::shared_ptr<Type> type, bool is_mut) noexcept {
    scope_stack.back().vars.emplace_back(std::move(name), std::move(type), is_mut);
}

void HirContext::new_global_var(std::string name, std::shared_ptr<Type> type, bool is_mut) noexcept {
    scope_stack[0].vars.emplace_back(std::move(name), std::move(type), is_mut);
}

