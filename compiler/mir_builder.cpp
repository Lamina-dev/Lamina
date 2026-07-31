#include "mir_builder.hpp"

#include <cassert>
#include <memory>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

namespace lmx::mir {

namespace {

bool is_int_type(const Type *type) noexcept {
    return type && type->kind == TypeKind::Basic &&
           reinterpret_cast<const BasicType *>(type)->type == runtime::ValueKind::Int;
}

bool is_float_type(const Type *type) noexcept {
    return type && type->kind == TypeKind::Basic &&
           reinterpret_cast<const BasicType *>(type)->type == runtime::ValueKind::Fraction;
}

class Builder {
    std::vector<std::string> break_labels;
    std::vector<std::string> continue_labels;
    MirModule &module_;
    std::vector<std::shared_ptr<MirNode>> *emit_target_ = &module_.nodes;
    size_t temp_counter_ = 0;
    size_t label_counter_ = 0;

    std::string new_temp() noexcept {
        return "_" + std::to_string(temp_counter_++);
    }

    std::string new_label() noexcept {
        return ".L" + std::to_string(label_counter_++);
    }

    void emit(std::shared_ptr<MirNode> node) const noexcept {
        emit_target_->push_back(std::move(node));
    }

    void emit_label(const std::string &name) const noexcept {
        emit(std::make_shared<MirLabel>(name));
    }

    void emit_expr(std::shared_ptr<MirExpr> expr) const noexcept {
        emit(std::make_shared<MirExprNode>(std::move(expr)));
    }

    std::shared_ptr<MirRefExpr> ensure_temp(std::shared_ptr<MirExpr> expr) noexcept {
        if (expr->kind == MirExprKind::Ref) {
            if (const auto &ref = reinterpret_cast<MirRefExpr &>(*expr);
                ref.is_temp) return std::static_pointer_cast<MirRefExpr>(std::move(expr));
        }
        return temp_assign(std::move(expr));
    }

    std::shared_ptr<MirRefExpr> temp_assign(std::shared_ptr<MirExpr> expr) {
        auto name = new_temp();
        emit(std::make_shared<MirTempAssign>(name, std::move(expr)));
        return std::make_shared<MirRefExpr>(name, true);
    }

    [[nodiscard]] std::shared_ptr<MirRefExpr> emit_to_temp(const std::string &name, std::shared_ptr<MirExpr> expr) const {
        emit(std::make_shared<MirTempAssign>(name, std::move(expr)));
        return std::make_shared<MirRefExpr>(name, true);
    }

    static std::shared_ptr<MirExpr> eval_binary_arith(const BinaryNode::Op op, std::shared_ptr<MirExpr> lhs,
                                                std::shared_ptr<MirExpr> rhs, const bool is_float) {
        switch (op) {
        case BinaryNode::Op::Add:
            if (is_float) return std::make_shared<MirFAddExpr>(std::move(lhs), std::move(rhs));
            return std::make_shared<MirIAddExpr>(std::move(lhs), std::move(rhs));
        case BinaryNode::Op::Sub:
            if (is_float) return std::make_shared<MirFSubExpr>(std::move(lhs), std::move(rhs));
            return std::make_shared<MirISubExpr>(std::move(lhs), std::move(rhs));
        case BinaryNode::Op::Mul:
            if (is_float) return std::make_shared<MirFMulExpr>(std::move(lhs), std::move(rhs));
            return std::make_shared<MirIMulExpr>(std::move(lhs), std::move(rhs));
        case BinaryNode::Op::Div:
            if (is_float) return std::make_shared<MirFDivExpr>(std::move(lhs), std::move(rhs));
            return std::make_shared<MirIDivExpr>(std::move(lhs), std::move(rhs));
        case BinaryNode::Op::Mod:
            if (is_float) return std::make_shared<MirFModExpr>(std::move(lhs), std::move(rhs));
            return std::make_shared<MirIModExpr>(std::move(lhs), std::move(rhs));
        case BinaryNode::Op::Pow:
            if (is_float) return std::make_shared<MirFMulExpr>(std::move(lhs), std::move(rhs));
            return std::make_shared<MirIPowExpr>(std::move(lhs), std::move(rhs));
        case BinaryNode::Op::Lt: {
            return std::make_shared<MirICmpLtExpr>(std::move(lhs), std::move(rhs));
        }
        case BinaryNode::Op::Gt: {
            return std::make_shared<MirICmpGtExpr>(std::move(lhs), std::move(rhs));
        }
        case BinaryNode::Op::Ge:{
            return std::make_shared<MirICmpGeExpr>(std::move(lhs), std::move(rhs));
        }
        case BinaryNode::Op::Le:{
            return std::make_shared<MirICmpLeExpr>(std::move(lhs), std::move(rhs));
        }
        case BinaryNode::Op::Eq:{
            return std::make_shared<MirICmpEqExpr>(std::move(lhs), std::move(rhs));
        }
        case BinaryNode::Op::Ne: {
            return std::make_shared<MirICmpNeExpr>(std::move(lhs), std::move(rhs));
        }
        case BinaryNode::Op::And:{
            return std::make_shared<MirCmpAndExpr>(std::move(lhs), std::move(rhs));
        }
        case BinaryNode::Op::Or:{
            return std::make_shared<MirCmpOrExpr>(std::move(lhs), std::move(rhs));
        }
        }
        std::unreachable();
    }

    static std::shared_ptr<MirExpr> eval_binary_cmp(const BinaryNode::Op op, std::shared_ptr<MirExpr> lhs,
                                             std::shared_ptr<MirExpr> rhs) {
        switch (op) {
        case BinaryNode::Op::Eq:  return std::make_shared<MirICmpEqExpr>(std::move(lhs), std::move(rhs));
        case BinaryNode::Op::Ne:  return std::make_shared<MirICmpNeExpr>(std::move(lhs), std::move(rhs));
        case BinaryNode::Op::Lt:  return std::make_shared<MirICmpLtExpr>(std::move(lhs), std::move(rhs));
        case BinaryNode::Op::Le:  return std::make_shared<MirICmpLeExpr>(std::move(lhs), std::move(rhs));
        case BinaryNode::Op::Gt:  return std::make_shared<MirICmpGtExpr>(std::move(lhs), std::move(rhs));
        case BinaryNode::Op::Ge:  return std::make_shared<MirICmpGeExpr>(std::move(lhs), std::move(rhs));
        default: std::unreachable();
        }
    }

    std::shared_ptr<MirExpr> eval(ExprNode *expr);

    std::shared_ptr<MirExpr> process_block(const BlockExprNode *block) {
        const auto save_temp_counter = temp_counter_;
        const auto save_label_counter_ = label_counter_;
        std::shared_ptr<MirExpr> block_val;
        for (auto &stmt : block->stmts) {
            if (stmt->kind == ASTKind::TailReturn) {
                const auto *tr = reinterpret_cast<TailReturnNode *>(stmt.get());
                block_val = eval(tr->expr.get());
            } else {
                process(stmt.get());
            }
        }
        temp_counter_ = save_temp_counter;
        label_counter_ = save_label_counter_;
        return block_val;
    }

public:
    explicit Builder(MirModule &mod) noexcept : module_(mod)  {}

    void process(StmtNode *stmt) noexcept {
        switch (stmt->kind) {
        case ASTKind::ExprStmt: {
            const auto *node = reinterpret_cast<ExprStmtNode *>(stmt);
            eval(node->expr.get());
            break;
        }
        case ASTKind::Return: {
            const auto *node = reinterpret_cast<ReturnNode *>(stmt);
            auto val = eval(node->expr.get());
            emit_expr(std::make_shared<MirRetExpr>(ensure_temp(std::move(val))));
            break;
        }
        case ASTKind::TailReturn: {
            const auto *node = reinterpret_cast<TailReturnNode *>(stmt);
            auto val = eval(node->expr.get());
            emit_expr(std::make_shared<MirRetExpr>(ensure_temp(std::move(val))));
            break;
        }
        case ASTKind::BreakStmt: {
            emit(std::make_shared<MirExprNode>(std::make_shared<MirGotoExpr>(break_labels.back())));
            break;
        }
        case ASTKind::ContinueStmt: {
            emit(std::make_shared<MirExprNode>(std::make_shared<MirGotoExpr>(continue_labels.back())));
            break;
        }
        case ASTKind::VarDecl: {
            if (auto *node = reinterpret_cast<VarDeclNode *>(stmt); node->init_value) {
                auto val = eval(node->init_value.get());
                emit(std::make_shared<MirAssign>(node->id, std::move(val)));
            }
            break;
        }
        case ASTKind::AssignStmt: {
            const auto *node = reinterpret_cast<AssignStmtNode *>(stmt);
            auto val = eval(node->rhs.get());
            if (node->lhs->kind == ASTKind::Identifier) {
                auto *id = reinterpret_cast<IdentifierNode *>(node->lhs.get());
                emit(std::make_shared<MirAssign>(id->id, std::move(val)));
            } else {
                eval(node->lhs.get());
            }
            break;
        }
        case ASTKind::FuncImpl: {
            auto *func = reinterpret_cast<FuncImplNode *>(stmt);
            if (!func->block) break;
            const auto save_tc = temp_counter_;
            const auto save_lc = label_counter_;
            const auto save_target = emit_target_;

            std::vector<std::shared_ptr<MirNode>> body;
            emit_target_ = &body;

            auto body_val = process_block(func->block.get());
            if (body_val) {
                emit_expr(std::make_shared<MirRetExpr>(ensure_temp(std::move(body_val))));
            } else {
                emit_expr(std::make_shared<MirRetVoidExpr>());
            }

            emit_target_ = save_target;
            temp_counter_ = save_tc;
            label_counter_ = save_lc;

            std::vector<std::string> params;
            if (func->params) {
                for (auto &key: func->params->stmts | std::views::keys) {
                    params.push_back(key);
                }
            }
            emit(std::make_shared<MirFuncDefine>(func->func_id, std::move(params), std::move(body)));
            break;
        }
        case ASTKind::LoopStmt: {
            const auto *node = reinterpret_cast<LoopStmtNode *>(stmt);
            auto cl = new_label() + "_continue";
            auto bl = new_label() + "_break";
            continue_labels.push_back(cl);
            break_labels.push_back(bl);
            emit(std::make_shared<MirLabel>(cl));

            std::shared_ptr<MirExpr> left_result = std::make_shared<MirNopExpr>();
            if (node->expr) {
                left_result = eval(node->expr.get());

                auto zero = new_temp();
                auto zero_lit = std::make_shared<MirLiteralExpr>(MirLiteralKind::Integer, "0");
                emit(std::make_shared<MirTempAssign>(zero, zero_lit));
                // %zero = 0

                auto result_tmp = new_temp();
                emit(std::make_shared<MirTempAssign>(
                    result_tmp,
                    std::make_shared<MirICmpEqExpr>(
                            left_result, std::make_shared<MirRefExpr>(zero, true)
                        )
                ));
                // %result_tmp = ICmpEq %left_result, %zero

                auto if_expr = std::make_shared<MirIfTrueExpr>(std::make_shared<MirRefExpr>(result_tmp, true), bl);
                emit(std::make_shared<MirExprNode>(if_expr));
                // IfTrue %result_tmp, break
            }

            for (auto& s : node->body) {
                process(s.get());
            }
            // loop body


            if (node->expr) {
                auto one = new_temp();
                auto one_lit = std::make_shared<MirLiteralExpr>(MirLiteralKind::Integer, "1");
                emit(std::make_shared<MirTempAssign>(one, one_lit));
                // %one = 1

                emit(std::make_shared<MirTempAssign>(
                    reinterpret_cast<MirRefExpr*>(left_result.get())->name,
                    std::make_shared<MirISubExpr>(left_result, one_lit)
                    ));
            }


            emit(std::make_shared<MirExprNode>(std::make_shared<MirGotoExpr>(cl)));
            // continue

            emit(std::make_shared<MirLabel>(bl));
            // break_label

            continue_labels.pop_back();
            break_labels.pop_back();
            break;
        }
        default:
            break;
        }
    }

    void build_native_decl(NativeFuncDeclNode *node) const noexcept {
        std::vector<runtime::ValueKind> params;
        for (auto &ty: node->params->stmts | std::views::values) {
            if (ty->kind == TypeKind::Basic) {
                const auto t = std::static_pointer_cast<BasicType>(ty);
                params.push_back(t->type);
            } else {
                params.push_back(runtime::ValueKind::Obj);
            }
        }
        runtime::ValueKind ret_ty;
        if (node->return_type->kind == TypeKind::Basic) {
            const auto t = std::static_pointer_cast<BasicType>(node->return_type);
            ret_ty = t->type;
        } else {
            ret_ty = runtime::ValueKind::Obj;
        }
        emit(std::make_shared<MirNativeFuncDefine>(node->func_id, node->symbol, std::move(params), ret_ty));
    }


    void build(const Module *ast_mod) {

        for (auto& n : ast_mod->native_funcs) {
            build_native_decl(n.get());
        }

        for (auto &decl : ast_mod->decls) {
            process(decl.get());
        }
        // for (auto &func : ast_mod->top_func_def) {
        //     process(&func);
        // }
    }
};

std::shared_ptr<MirExpr> Builder::eval(ExprNode *expr) {
    switch (expr->kind) {
    case ASTKind::Literal: {
        auto *lit = reinterpret_cast<LiteralNode *>(expr);
        MirLiteralKind lk;
        switch (lit->kind) {
        case LiteralNode::Kind::Integer: lk = MirLiteralKind::Integer; break;
        case LiteralNode::Kind::Float:   lk = MirLiteralKind::Float;   break;
        case LiteralNode::Kind::String:  lk = MirLiteralKind::String;  break;
        case LiteralNode::Kind::Boolean: lk = MirLiteralKind::Boolean; break;
        }
        return std::make_shared<MirLiteralExpr>(lk, lit->val);
    }
    case ASTKind::Identifier: {
        auto *id = reinterpret_cast<IdentifierNode *>(expr);
        auto ref = std::make_shared<MirRefExpr>(id->id, false);
        return ref;
    }
    case ASTKind::Unary: {
        auto *un = reinterpret_cast<UnaryNode *>(expr);
        auto operand = ensure_temp(eval(un->expr.get()));
        if (is_int_type(expr->type.get())) {
            return temp_assign(std::make_shared<MirINegExpr>(std::move(operand)));
        }
        return temp_assign(std::make_shared<MirFNegExpr>(std::move(operand)));
    }
    case ASTKind::Binary: {
        switch (auto *bin = reinterpret_cast<BinaryNode *>(expr); bin->op) {
        case BinaryNode::Op::And: {
            auto false_label = new_label();
            auto end_label = new_label();
            auto result_name = new_temp();
            auto lhs = ensure_temp(eval(bin->lhs.get()));
            emit_expr(std::make_shared<MirIfFalseExpr>(lhs, false_label));
            auto _1 = emit_to_temp(result_name, eval(bin->rhs.get()));
            emit_expr(std::make_shared<MirGotoExpr>(end_label));
            emit_label(false_label);
            auto false_lit = std::make_shared<MirLiteralExpr>(MirLiteralKind::Boolean, "false");
            auto _2 = emit_to_temp(result_name, std::move(false_lit));
            emit_label(end_label);
            auto result_ref = std::make_shared<MirRefExpr>(result_name, true);
            return result_ref;
        }
        case BinaryNode::Op::Or: {
            auto true_label = new_label();
            auto end_label = new_label();
            auto result_name = new_temp();
            auto lhs = ensure_temp(eval(bin->lhs.get()));
            emit_expr(std::make_shared<MirIfTrueExpr>(lhs, true_label));
            auto _2 = emit_to_temp(result_name, eval(bin->rhs.get()));
            emit_expr(std::make_shared<MirGotoExpr>(end_label));
            emit_label(true_label);
            auto true_lit = std::make_shared<MirLiteralExpr>(MirLiteralKind::Boolean, "true");
            auto _ = emit_to_temp(result_name, std::move(true_lit));
            emit_label(end_label);
            auto result_ref = std::make_shared<MirRefExpr>(result_name, true);
            return result_ref;
        }
        default: {
            if (bin->op >= BinaryNode::Op::Eq && bin->op <= BinaryNode::Op::Ge) {
                auto lhs = ensure_temp(eval(bin->lhs.get()));
                auto rhs = ensure_temp(eval(bin->rhs.get()));
                return temp_assign(eval_binary_cmp(bin->op, std::move(lhs), std::move(rhs)));
            }
            bool is_float = is_float_type(expr->type.get());
            auto lhs = ensure_temp(eval(bin->lhs.get()));
            auto rhs = ensure_temp(eval(bin->rhs.get()));
            return temp_assign(eval_binary_arith(bin->op, std::move(lhs), std::move(rhs), is_float));
        }
        }
    }
    case ASTKind::Block: {
        auto *block = reinterpret_cast<BlockExprNode *>(expr);
        return process_block(block);
    }
    // case ASTKind::Exprs: {
    //     auto *exprs = static_cast<ExprsNode *>(expr);
    //     std::shared_ptr<MirExpr> last;
    //     for (auto &e : exprs->exprs) {
    //         last = eval(e.get());
    //     }
    //     return last;
    // }
    case ASTKind::SuffixParen: {
        auto *call = reinterpret_cast<SuffixParenNode *>(expr);

        std::vector<std::shared_ptr<MirRefExpr>> arg_refs;
        if (call->suffix) {
            for (auto &arg : call->suffix->exprs) {
                auto arg_val = eval(arg.get());
                arg_refs.push_back(ensure_temp(std::move(arg_val)));
            }
        }
        if (call->expr->kind == ASTKind::Identifier) {
            std::string func_name = reinterpret_cast<IdentifierNode *>(call->expr.get())->id;
            auto call_expr = std::make_shared<MirCallFastExpr>(std::move(func_name), std::move(arg_refs));
            return temp_assign(std::move(call_expr));
        }

        const auto reg_func = temp_assign(std::move(eval(call->expr.get())));
        auto call_expr = std::make_shared<MirCallExpr>(reg_func, std::move(arg_refs));
        return temp_assign(std::move(call_expr));
    }
    case ASTKind::SuffixBracket: {
        // auto *idx = static_cast<SuffixBracketNode *>(expr);
        // eval(idx->expr.get());
        // eval(idx->suffix.get());
        // auto lit = std::make_shared<MirLiteralExpr>(MirLiteralKind::Integer, "0");
        return std::make_shared<MirLiteralExpr>(MirLiteralKind::Integer, "0");
    }
    case ASTKind::IfExpr: {
        auto *if_expr = reinterpret_cast<IfExprNode *>(expr);
        auto else_label = new_label();
        auto end_label = new_label();
        auto result_name = new_temp();

        auto cond = ensure_temp(eval(if_expr->cond.get()));
        emit_expr(std::make_shared<MirIfFalseExpr>(cond, else_label));

        if (auto e = eval(if_expr->then.get()); e) {
            auto _ = emit_to_temp(result_name, e);
        }
        emit_expr(std::make_shared<MirGotoExpr>(end_label));

        emit_label(else_label);
        if (if_expr->els) {

            if (auto e = eval(if_expr->els.get()); e) {
                auto _ = emit_to_temp(result_name, e);
            }
        }
        emit_label(end_label);

        auto result_ref = std::make_shared<MirRefExpr>(result_name, true);
        return result_ref;
    }
    case ASTKind::AsExpr: {
        auto *as = reinterpret_cast<AsExprNode *>(expr);
        return eval(as->expr.get());
    }
    case ASTKind::NativeFuncCall: {
        auto *call = reinterpret_cast<NativeFuncCallExpr*>(expr);
        std::string func_name;
        if (call->expr->kind == ASTKind::Identifier) {
            func_name = reinterpret_cast<IdentifierNode *>(call->expr.get())->id;
        }
        std::vector<std::shared_ptr<MirRefExpr>> arg_refs;
        if (call->suffix) {
            for (auto &arg : call->suffix->exprs) {
                auto arg_val = eval(arg.get());
                arg_refs.push_back(ensure_temp(std::move(arg_val)));
            }
        }
        auto call_expr = std::make_shared<MirCCallExpr>(std::move(func_name), std::move(arg_refs));
        return temp_assign(std::move(call_expr));
    }
    case ASTKind::DotExpr: {
        const auto *dot = reinterpret_cast<DotExprNode *>(expr);
        std::shared_ptr<MirRefExpr> mod_ref;
        std::string mod_name;
         if (dot->expr->type->kind == TypeKind::Module) {
             const auto *id = reinterpret_cast<IdentifierNode *>(dot->expr.get());
             mod_ref = temp_assign(std::make_shared<MirGetModuleExpr>(id->id));  //
             mod_name = id->id;
         } else {
             mod_ref = temp_assign(eval(dot->expr.get()));
         }
         auto attr = std::make_shared<MirGetModuleAttrExpr>(mod_ref, std::move(mod_name), dot->rhs->id);  // %_1 = GetModuleAttr %_0, "foo"
         return temp_assign(attr);
         break;
    }
    default:
        std::unreachable();
        //return std::make_shared<MirLiteralExpr>(MirLiteralKind::Integer, "0");
    }
    std::unreachable();
    return nullptr;
}
} // namespace

MirModule MirBuilder::from_ast_module(const std::shared_ptr<Module> &ast) {
    MirModule mod;
    mod.lib_name = ast->lib_name;
    mod.imports = ast->imports;
    Builder builder(mod);
    builder.build(ast.get());
    return mod;
}

} // namespace lmx::mir
