//
// Created by meian on 2026/7/5.
//

#include "type_checker.hpp"

#include <fstream>
#include <ranges>
#include <map>

#include "../parser.hpp"
#include "../error.hpp"
#include "../../utils/utils.hpp"

using namespace lmx;
using namespace lmx::hir;

Scope::Scope(std::string name) noexcept : name(std::move(name)) {}

Scope::Scope(const ScopeType scope) noexcept : scope(scope) {}

std::optional<Scope::Var *> TypeCkContext::find_var(const std::string &name) noexcept {
    for (auto& i : scope_stack | std::views::reverse) {
        for (auto& j : i.vars) {
            if (j.name == name) return &j;
        }
    }
    return std::nullopt;
}
std::optional<Scope::Var *> TypeCkContext::find_global(const std::string &name) noexcept {
    for (auto& i : global_scope) {
        if (i.name == name) return &i;
    }
    return std::nullopt;
}

TypeCkContext::TypeCkContext() noexcept {
    scope_stack.emplace_back("@GLOBAL");
}

std::shared_ptr<Type> TypeCkContext::inference_type(ExprNode* type) noexcept {
    if (!type) return type_pool.unknown();
    switch (type->kind) {
    case ASTKind::Literal: {
        const auto node = reinterpret_cast<LiteralNode*>(type);
        switch (node->kind) {
        case LiteralNode::Kind::Integer: {
            return type_pool.basic(runtime::ValueKind::Int);
        }
        case LiteralNode::Kind::Float: {
            return type_pool.basic(runtime::ValueKind::Fraction);
        }
        case LiteralNode::Kind::String: {
            return type_pool.string();
        }
        case LiteralNode::Kind::Boolean: {
            return type_pool.basic(runtime::ValueKind::Bool);
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

                return type_pool.basic(t2);
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
        return type_pool.none();
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
    case ASTKind::DotExpr: {
        const auto node = reinterpret_cast<DotExprNode*>(type);
        const auto left = std::reinterpret_pointer_cast<ModuleType>(inference_type(node->expr.get()));
        return (*left->find_var(node->rhs->id))->type;
        break;
    }
    case ASTKind::NativeFuncCall: {
        const auto node = reinterpret_cast<NativeFuncCallExpr*>(type);
        const auto left_ty = std::reinterpret_pointer_cast<NativeFunctionType>(inference_type(node->expr.get()));
        return left_ty->ret_ty;
        break;
    }
    default: std::unreachable();
    }
    return type_pool.unknown();
}


static std::shared_ptr<StmtNode> sugar_loop_count(const std::shared_ptr<LoopStmtNode>& stmt) noexcept {
    std::string name = "@loop_cnt_id";
    auto var_cnt = std::make_shared<VarDeclNode>(0, 0, name, type_pool.basic(runtime::ValueKind::Int), true);
    var_cnt->init_value = std::move(stmt->expr);


    const auto lhs = std::make_shared<IdentifierNode>(0, 0, name);
    const auto rhs = std::make_shared<LiteralNode>(0, 0, "0", LiteralNode::Kind::Integer);
    auto break_cond = std::make_shared<BinaryNode>(0, 0, lhs, BinaryNode::Op::Eq, rhs);

    auto break_stmt_block = std::make_shared<BlockExprNode>(0, 0, decltype(BlockExprNode::stmts){std::make_shared<BreakStmtNode>(0, 0)});

    auto break_if = std::make_shared<IfExprNode>(0, 0, break_cond, break_stmt_block, nullptr);
    // if `@loop_cnt_id` == 0 { break }

    stmt->body.insert(stmt->body.begin(), std::make_shared<ExprStmtNode>(0, 0, break_if));

    const auto one = std::make_shared<LiteralNode>(0, 0, "1", LiteralNode::Kind::Integer);
    const auto dec_cnt = std::make_shared<AssignStmtNode>(0, 0, lhs, std::make_shared<BinaryNode>(0, 0, lhs, BinaryNode::Op::Sub, one));
    stmt->body.insert(stmt->body.end(), dec_cnt);
    decltype(BlockExprNode::stmts) block{var_cnt, stmt};
    auto result = std::make_shared<ExprStmtNode>(
        stmt->line, stmt->col, std::make_shared<BlockExprNode>(stmt->line, stmt->col, std::move(block)));
    return result;
}



// void HirContext::reset() noexcept {
//     scope_stack.clear();
// }

std::vector<Scope::Var> TypeCkContext::check_module(const std::shared_ptr<Module> &mod) noexcept {
    const auto save_cur_module = cur_module;
    cur_module = mod;

    if (!mod->native_funcs.empty() && mod->lib_name.empty()) {
        throw_error(ErrorType::Analysis, "module not `static` declare dynamic library, cannot declare native function", 0 , 0);

    }
    for (const auto& n : mod->native_funcs) {
        new_global_var(n->func_id, n->make_type());
    }
    // reset();
    for (auto& node : mod->decls) {
        check_stmt(node);
    }

    cur_module = save_cur_module;

    std::vector<Scope::Var> result;

    for (const auto& v : get_global()) {
        if (v.type->kind == TypeKind::Function) result.push_back(v);
    }
    return result;
}
void TypeCkContext::check_expr(std::shared_ptr<ExprNode>& expr) noexcept {
    if (!expr) return;
    switch (expr->kind) {
    case ASTKind::Literal: {
        expr->type = inference_type(expr.get());
        break;
    }
    case ASTKind::Identifier: {
        auto* node = reinterpret_cast<IdentifierNode*>(expr.get());
        if (const auto re = find_var(node->id); re.has_value()) {
            node->type = (*re)->type;
            break;
        }
        if (const auto re = find_global(node->id); re.has_value()) {
            node->type = (*re)->type;
            break;
        }
        throw_error(ErrorType::Analysis, "undefined var `" + node->id + "`", node->line, node->col);
        break;
    }
    case ASTKind::Unary: {
        auto* node = reinterpret_cast<UnaryNode*>(expr.get());
        check_expr(node->expr);
        const auto type = node->expr->type;
        if (type->kind != TypeKind::Basic) {
            throw_error(ErrorType::Analysis, "unary cannot applied to this type", expr->line, expr->col);
            break;
        }
        const auto t2 = std::reinterpret_pointer_cast<BasicType>(type);
        if (node->op == UnaryNode::Op::Neg) {
            if (
            t2->type != runtime::ValueKind::Int && t2->type != runtime::ValueKind::Fraction) {
                throw_error(ErrorType::Analysis, "unary`-` cannot applied to this type", expr->line, expr->col);
                break;
            }
        } else if (node->op == UnaryNode::Op::Not) {
            if (t2->type != runtime::ValueKind::Bool) {
                throw_error(ErrorType::Analysis, "unary`!` cannot applied to this type", expr->line, expr->col);
                break;
            }
        }
        node->type = type;
        break;
    }
    case ASTKind::Binary: {
        auto* node = reinterpret_cast<BinaryNode*>(expr.get());
        check_expr(node->lhs);
        check_expr(node->rhs);
        const auto lty = node->lhs->type;
        const auto rty = node->rhs->type;
        if (Type::is_null_type(lty.get()) || Type::is_null_type(rty.get())) break;
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
            const auto &type_map_it = op_types.find(t2);
            if (type_map_it == op_types.end()) {
                throw_error(
                    ErrorType::Analysis,
                    "binary operation type mismatch, (" +
                    Type::to_string(lty.get()) + " " +
                    BinaryNode::op_to_string(node->op) + " " + Type::to_string(rty.get()) + ")", expr->line, expr->col
                    );
                break;
            }
            const auto& type_map = type_map_it->second;
            if (const auto it = type_map.find(node->op); it != type_map.end()) {
                node->type = type_pool.basic(it->second);
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
        const auto node = reinterpret_cast<BlockExprNode*>(expr.get());
        scope_stack.emplace_back(Scope::ScopeType::Block);
        for (auto& s : node->stmts) check_stmt(s);
        expr->type = scope_stack.back().return_type;
        scope_stack.pop_back();
        break;
    }
    case ASTKind::SuffixParen: {
        const auto node = reinterpret_cast<SuffixParenNode*>(expr.get());
        if (node->expr->kind == ASTKind::Identifier) {
            const auto id = reinterpret_cast<IdentifierNode*>(node->expr.get());
            if (const auto re = find_global(id->id); re.has_value()) {
                node->can_fast = true;
            }
        } else {
            node->can_fast = false;
        }
        check_expr(node->expr);
        const auto left = node->expr->type;
        if (Type::is_null_type(left.get())) break;
        if (left->kind == TypeKind::Function) {
            const auto func_ty = std::reinterpret_pointer_cast<FunctionType>(left);
            if (func_ty->params_ty.size() != node->suffix->exprs.size()) {
                throw_error(ErrorType::Analysis,
                    "mismatch args count in function calling, (param(s)"
                    + std::to_string(func_ty->params_ty.size()) +
                    " != arg(s)" +
                    std::to_string(node->suffix->exprs.size()) + ")",
                    node->line, node->col
                    );
                break;
            }
            const auto len = func_ty->params_ty.size();
            for (auto i = 0; i < len; i++) {
                const auto param = func_ty->params_ty[i];
                check_expr(node->suffix->exprs[i]);
                if (!param->equals(node->suffix->exprs[i]->type.get())) {
                    throw_error(ErrorType::Analysis,
                        "type mismatch arg in function calling in arg(s) " + std::to_string(i) +
                        ": (" + Type::to_string(node->suffix->exprs[i]->type.get()) +
                        " != " + Type::to_string(node->suffix->exprs[i]->type.get()) + ")"
                        , node->line, node->col);
                    break;
                }
            }
            node->type = std::reinterpret_pointer_cast<FunctionType>(left)->ret_ty;
        } else if (left->kind == TypeKind::NativeFunction) {
            new (expr.get()) NativeFuncCallExpr(node);
            const auto node = reinterpret_cast<NativeFuncCallExpr*>(expr.get());
            const auto func_ty = std::reinterpret_pointer_cast<NativeFunctionType>(left);
            bool has_va_list = false;
            size_t fixed_arg_cnt = func_ty->params_ty.size();
            for (const auto& p : func_ty->params_ty) {
                if (p->kind == TypeKind::Basic &&
                    reinterpret_cast<BasicType*>(p.get())->type == runtime::ValueKind::C_VaList) {
                    if (p.get() != func_ty->params_ty.back().get()) {
                        // 如果变参不是最后一个类型，报错
                        throw_error(ErrorType::Analysis, "c_valist must be last type", node->line, node->col);
                        goto suffix_paren_break;
                    }
                    has_va_list = true;
                    fixed_arg_cnt = func_ty->params_ty.size() - 1;
                }
            }

            if (!has_va_list && func_ty->params_ty.size() != node->suffix->exprs.size()) {
                throw_error(ErrorType::Analysis,
                    "mismatch args count in function calling,(param(s): "
                    + std::to_string(func_ty->params_ty.size()) +
                    " != arg(s): " +
                    std::to_string(node->suffix->exprs.size()) + ")",
                    node->line, node->col
                    );
                break;
            }
            const auto len = node->suffix->exprs.size();
            size_t i = 0;
            for (; i < fixed_arg_cnt; i++) {
                const auto param = func_ty->params_ty[i];
                check_expr(node->suffix->exprs[i]);
                if (!param->equals(node->suffix->exprs[i]->type.get())) {
                    throw_error(ErrorType::Analysis, "type mismatch arg in function calling", node->line, node->col);
                    break;
                }
            }
            for (; i < len; i++) {
                // const auto param = func_ty->params_ty[i];
                check_expr(node->suffix->exprs[i]);
            }

            node->type = std::reinterpret_pointer_cast<NativeFunctionType>(left)->ret_ty;
        } else {
            throw_error(ErrorType::Analysis, "not a function type", node->line, node->col);
            break;
        }

        break;
        suffix_paren_break:
        break;
    }
    case ASTKind::SuffixBracket: {
        const auto node = reinterpret_cast<SuffixBracketNode*>(expr.get());
        check_expr(node->expr);
        const auto left = inference_type(node->expr.get());
        if (left->kind != TypeKind::Array) {
            throw_error(ErrorType::Analysis, "must be array type but got `" + Type::to_string(left.get()) + "`", node->line, node->col);
            break;
        }
        node->type = std::reinterpret_pointer_cast<ArrayType>(left)->type;
        break;
    }
    case ASTKind::IfExpr: {
        const auto node = reinterpret_cast<IfExprNode*>(expr.get());
        check_expr(node->cond);
        if (node->cond->type->kind != TypeKind::Basic &&
            std::reinterpret_pointer_cast<BasicType>(node->cond->type)->type != runtime::ValueKind::Bool) {
            throw_error(ErrorType::Analysis, "must be bool type but got `" + Type::to_string(node->cond->type.get()), node->line, node->col);
            break;
        }
        check_expr(node->then);
        if (node->els) {
            check_expr(node->els);
            if ( node->then->have_ret_value() && node->els->have_ret_value() &&
                !node->then->type->equals(node->els->type.get())) {
                throw_error(ErrorType::Analysis, "if express then and else cannot type mismatch", node->line, node->col);
                break;
            }
        }
        node->type = node->then->type;
        break;
    }
    case ASTKind::AsExpr: {
        // const auto node = reinterpret_cast<AsExprNode*>(expr);
        // check_expr(node->expr.get());
        // node->type = node->cast_type;
        // todo!
        break;
    }
    case ASTKind::DotExpr: {
        const auto node = reinterpret_cast<DotExprNode*>(expr.get());
        check_expr(node->expr);
        if (!Type::is_null_type(node->expr->type.get()) && node->expr->type->kind != TypeKind::Module) {
            throw_error(ErrorType::Analysis, "must be module type", node->line, node->col);
            break;
        }
        const auto left_ty = std::reinterpret_pointer_cast<ModuleType>(node->expr->type);
        if (Type::is_null_type(left_ty.get())) break;
        if (const auto result = left_ty->find_var(node->rhs->id); result.has_value()) {
            const auto* var = *result;
            node->rhs->type = var->type;
            node->type = var->type;
        } else {
            throw_error(ErrorType::Analysis, "module not have var `" + node->rhs->id + "`", node->line, node->col);
            break;
        }
        break;
    }
    case ASTKind::PipeExpr: {
        const auto node = reinterpret_cast<PipeExprNode*>(expr.get());
        check_expr(node->lhs);
        check_expr(node->rhs);
        std::shared_ptr<ExprNode> result;

        const auto& lhs_ty = node->lhs->type;
        const auto& rhs_ty = node->rhs->type;
        if (rhs_ty == nullptr || rhs_ty->kind != TypeKind::Function) {
            throw_error(ErrorType::Analysis, "`|>` op not return func on right", node->line, node->col);
            break;
        }
        const auto rhs_fty = std::reinterpret_pointer_cast<FunctionType>(rhs_ty);
        if (rhs_fty->params_ty.empty()) {
            throw_error(ErrorType::Analysis, "`|>` op right function calling not arg(1)", node->line, node->col);
            break;
        }
        if (!rhs_fty->params_ty[0]->equals(lhs_ty.get())) {
            throw_error(
                ErrorType::Analysis,
                "`|>` op in right, function arg type and left type mismatch, ("
                + Type::to_string(lhs_ty.get())
                + " |> "
                + Type::to_string(rhs_fty->params_ty[0].get())
                + ")",
                node->line, node->col
                );
            break;
        }
        decltype(ExprsNode::exprs) exprs;
        exprs.push_back(node->lhs);
        result = std::make_shared<SuffixParenNode>(
            node->line, node->col, node->rhs,
            std::make_shared<ExprsNode>(node->line, node->col, exprs)
            );
        result->type = rhs_fty->ret_ty;

        expr = result;
        break;
    }
    default: std::unreachable();
    }
}

void TypeCkContext::check_stmt(std::shared_ptr<StmtNode>& stmt) noexcept {
    switch (stmt->kind) {
    case ASTKind::ExprStmt: {
        auto* node = reinterpret_cast<ExprStmtNode*>(stmt.get());
        check_expr(node->expr);
        break;
    }
    case ASTKind::ImportStmt: {
        const auto* node = reinterpret_cast<ImportStmtNode*>(stmt.get());
        const auto found = find_module_name(node->name);
        if (!found) {
            throw_error(ErrorType::Analysis, "no module is called `" + node->name + "`", node->line, node->col);
            break;
        }
        const auto [path, abs_path] = *found;
        const auto abs_path_string = abs_path.string();

        if (cur_module->module_is_imported(abs_path_string)) {
            break;
        }

        std::vector<Scope::Var> exports;
        std::vector<uint8_t> compiled;
        do {
            std::ifstream ifs(abs_path);
            if (!ifs.is_open()) {
                throw_error(ErrorType::Analysis, "cannot open `" + abs_path.string() + "`", node->line, node->col);
            }
            std::string code{
                std::istreambuf_iterator(ifs),
                std::istreambuf_iterator<char>()
            };
            code += '\n';

            auto tokens = Lexer(code).tokenize(code);
            if (errd) break;

            auto ast = Parser(tokens).parse_module(abs_path_string);
            if (errd) break;

            exports = TypeCkContext().check_module(ast);
            if (errd) break;
            compiled = ast_to_binary(ast);
            if (errd) break;
        } while (false);
        if (errd) break;

        auto output_path = std::filesystem::path(main_module->name).parent_path() / module_cache_fold ;
        if (abs_path.filename() == std::string(file_default_mod) + file_suffix) {
            output_path /= path.string();
            output_path /= std::string(file_default_mod) + file_suffix_binary;
        } else {
            output_path /= path.string() + file_suffix_binary;
        }
        std::filesystem::create_directories(output_path.parent_path());
        std::ofstream ofs(output_path);
        ofs.write(reinterpret_cast<const char*>(compiled.data()), static_cast<std::streamsize>(compiled.size()));
        ofs.close();
        auto mod_ty = std::reinterpret_pointer_cast<ModuleType>(
        type_pool.module(output_path.string(), std::move(exports)));
        new_global_var(path.filename().string(), mod_ty);

        cur_module->imports[abs_path_string] = mod_ty;
        break;
    }
    //case ASTKind::Exprs:
    //    break;
    //case ASTKind::ParamsDeclNode:
    //    break;
    case ASTKind::FuncImpl: {
        auto* node = reinterpret_cast<FuncImplNode*>(stmt.get());
        if (!is_global_scope()) throw_error(ErrorType::Analysis, "function only define in GlobalScope", stmt->line, stmt->col);

        new_global_var(node->func_id, node->make_type());
        auto& ref = global_scope.back();
        Scope scope;
        scope.name = node->func_id;
        scope.return_type = node->return_type;
        for (const auto& [name, ty] : node->params->stmts) {
            scope.vars.emplace_back(name, ty, true);
        }
        scope_stack.push_back(scope);

        if (node->block->kind == ASTKind::Block) {
            for (auto* block = reinterpret_cast<BlockExprNode*>(node->block.get());
                auto& s : block->stmts) {

                check_stmt(s);
            }
        } else check_expr(node->block);
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
        const auto node = reinterpret_cast<ReturnNode*>(stmt.get());
        if (!node->expr) break;
        check_expr(node->expr);
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
        const auto node = reinterpret_cast<TailReturnNode*>(stmt.get());
        check_expr(node->expr);
        if (Type::is_null_type(scope_stack.back().return_type.get()))
            scope_stack.back().return_type = node->expr->type;
        else {
            if (!scope_stack.back().return_type->equals(node->expr->type.get())) {
                throw_error(ErrorType::Analysis, "return type is inconsistent with the above", node->line, node->col);
                break;
            }
        }
        // node->expr->type = inference_type(node->expr.get());
        break;
    }
    case ASTKind::VarDecl: {
        const auto node = reinterpret_cast<VarDeclNode*>(stmt.get());
        check_expr(node->init_value);
        if (Type::is_null_type(node->type.get())) {
            if (!node->init_value) {
                throw_error(ErrorType::Analysis, "the var `" + node->id + "` type not found", node->line, node->col);
                break;
            } else node->type = node->init_value->type;
        } else {
            if (!node->type->equals(node->init_value->type.get())) {
                throw_error(ErrorType::Analysis, "the var `" + node->id + "` type mismatch with the initialization type", node->line, node->col);
                break;
            }
        }
        new_cur_scope_var(node->id, node->type, node->is_mutable);
        break;
    }
    case ASTKind::AssignStmt: {
        const auto node = reinterpret_cast<AssignStmtNode*>(stmt.get());
        check_expr(node->lhs);
        check_expr(node->rhs);
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
    case ASTKind::BreakStmt:
    case ASTKind::ContinueStmt:{
        bool in_loop = false;
        for (const auto& s : scope_stack | std::views::reverse) {
            if (s.scope == Scope::ScopeType::Loop) {
                in_loop = true;
                break;
            }
        }
        if (!in_loop) {
            throw_error(ErrorType::Analysis, "break stmt must be in loop body", stmt->line, stmt->col);
            break;
        }
        break;
    }
    case ASTKind::LoopStmt: {
        auto node = std::reinterpret_pointer_cast<LoopStmtNode>(stmt);
        if (node->expr) {
            check_expr(node->expr);
            if (!Type::is_null_type(node->expr->type.get()) &&
                !node->expr->type->equals(type_pool.basic(runtime::ValueKind::Int).get())
                ) {

                throw_error(ErrorType::Analysis, "loop condition type must be int", node->line, node->col);
                break;
                }
        }
        scope_stack.emplace_back(Scope::ScopeType::Loop);
        for (auto& s : node->body) {
            check_stmt(s);
        }
        scope_stack.pop_back();
        if (node->expr) {
            stmt = sugar_loop_count(node);
        }
        break;
    }
    default: std::unreachable();
    }
}

bool TypeCkContext::is_global_scope() const noexcept {
    return scope_stack.size() == 1;
}

void TypeCkContext::new_var(std::string name, std::shared_ptr<Type> type, Scope *scope, bool is_mut) noexcept {
    scope->vars.emplace_back(std::move(name), std::move(type), is_mut);
}

void TypeCkContext::new_cur_scope_var(std::string name, std::shared_ptr<Type> type, bool is_mut) noexcept {
    scope_stack.back().vars.emplace_back(std::move(name), std::move(type), is_mut);
}

void TypeCkContext::new_global_var(std::string name, std::shared_ptr<Type> type, bool is_mut) noexcept {
    global_scope.emplace_back(std::move(name), std::move(type), is_mut);
}

std::vector<Scope::Var> &TypeCkContext::get_global() noexcept {
    return global_scope;
}

