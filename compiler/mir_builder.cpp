#include "mir_builder.hpp"
#include "../runtime/opcode.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace lmx::mir {

namespace {

bool is_int_type(const Type *type) {
    return type && type->kind == TypeKind::Basic &&
           static_cast<const BasicType *>(type)->type == runtime::ValueKind::Int;
}

bool is_float_type(const Type *type) {
    return type && type->kind == TypeKind::Basic &&
           static_cast<const BasicType *>(type)->type == runtime::ValueKind::Fraction;
}

class Builder {
    MirModule &module_;
    std::vector<std::shared_ptr<MirNode>> *emit_target_ = &module_.nodes;
    size_t temp_counter_ = 0;
    size_t label_counter_ = 0;

    std::string new_temp() {
        return "_" + std::to_string(temp_counter_++);
    }

    std::string new_label() {
        return ".L" + std::to_string(label_counter_++);
    }

    void emit(std::shared_ptr<MirNode> node) {
        emit_target_->push_back(std::move(node));
    }

    void emit_label(const std::string &name) {
        emit(std::make_shared<MirLabel>(name));
    }

    void emit_expr(std::shared_ptr<MirExpr> expr) {
        emit(std::make_shared<MirExprNode>(std::move(expr)));
    }

    std::shared_ptr<MirRefExpr> ensure_temp(std::shared_ptr<MirExpr> expr) {
        if (expr->kind == MirExprKind::Ref) {
            auto &ref = static_cast<MirRefExpr &>(*expr);
            if (ref.is_temp) return std::static_pointer_cast<MirRefExpr>(std::move(expr));
        }
        return temp_assign(std::move(expr));
    }

    std::shared_ptr<MirRefExpr> temp_assign(std::shared_ptr<MirExpr> expr) {
        auto name = new_temp();
        emit(std::make_shared<MirTempAssign>(name, std::move(expr)));
        return std::make_shared<MirRefExpr>(name, true);
    }

    std::shared_ptr<MirRefExpr> emit_to_temp(const std::string &name, std::shared_ptr<MirExpr> expr) {
        emit(std::make_shared<MirTempAssign>(name, std::move(expr)));
        return std::make_shared<MirRefExpr>(name, true);
    }

    std::shared_ptr<MirExpr> eval_binary_arith(BinaryNode::Op op, std::shared_ptr<MirExpr> lhs,
                                                std::shared_ptr<MirExpr> rhs, bool is_float) {
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
        case BinaryNode::Op::Dot:
            break;
        }
    }

    std::shared_ptr<MirExpr> eval_binary_cmp(BinaryNode::Op op, std::shared_ptr<MirExpr> lhs,
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

    std::shared_ptr<MirExpr> process_block(BlockExprNode *block) {
        std::shared_ptr<MirExpr> block_val;
        for (auto &stmt : block->stmts) {
            if (stmt->kind == ASTKind::TailReturn) {
                auto *tr = static_cast<TailReturnNode *>(stmt.get());
                block_val = eval(tr->expr.get());
            } else {
                process(stmt.get());
            }
        }
        return block_val;
    }

public:
    explicit Builder(MirModule &mod) : module_(mod) {}

    void process(StmtNode *stmt) {
        switch (stmt->kind) {
        case ASTKind::ExprStmt: {
            auto *node = static_cast<ExprStmtNode *>(stmt);
            eval(node->expr.get());
            break;
        }
        case ASTKind::Return: {
            auto *node = static_cast<ReturnNode *>(stmt);
            auto val = eval(node->expr.get());
            emit_expr(std::make_shared<MirRetExpr>(ensure_temp(std::move(val))));
            break;
        }
        case ASTKind::TailReturn: {
            auto *node = static_cast<TailReturnNode *>(stmt);
            auto val = eval(node->expr.get());
            emit_expr(std::make_shared<MirRetExpr>(ensure_temp(std::move(val))));
            break;
        }
        case ASTKind::BreakStmt: {
            break;
        }
        case ASTKind::VarDecl: {
            auto *node = static_cast<VarDeclNode *>(stmt);
            if (node->init_value) {
                auto val = eval(node->init_value.get());
                emit(std::make_shared<MirAssign>(node->id, std::move(val)));
            }
            break;
        }
        case ASTKind::AssignStmt: {
            auto *node = static_cast<AssignStmtNode *>(stmt);
            auto val = eval(node->rhs.get());
            if (node->lhs->kind == ASTKind::Identifier) {
                auto *id = static_cast<IdentifierNode *>(node->lhs.get());
                emit(std::make_shared<MirAssign>(id->id, std::move(val)));
            } else {
                eval(node->lhs.get());
            }
            break;
        }
        case ASTKind::FuncImpl: {
            auto *func = static_cast<FuncImplNode *>(stmt);
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
                for (auto &p : func->params->stmts) {
                    params.push_back(p.first);
                }
            }
            emit(std::make_shared<MirFuncDefine>(func->func_id, std::move(params), std::move(body)));
            break;
        }
        default:
            break;
        }
    }

    void build(Module *ast_mod) {
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
        auto *lit = static_cast<LiteralNode *>(expr);
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
        auto *id = static_cast<IdentifierNode *>(expr);
        auto ref = std::make_shared<MirRefExpr>(id->id, false);
        return ref;
    }
    case ASTKind::Unary: {
        auto *un = static_cast<UnaryNode *>(expr);
        auto operand = ensure_temp(eval(un->expr.get()));
        if (is_int_type(expr->type.get())) {
            return temp_assign(std::make_shared<MirINegExpr>(std::move(operand)));
        }
        return temp_assign(std::make_shared<MirFNegExpr>(std::move(operand)));
    }
    case ASTKind::Binary: {
        auto *bin = static_cast<BinaryNode *>(expr);
        switch (bin->op) {
        case BinaryNode::Op::Dot: {
            return eval(bin->rhs.get());
        }
        case BinaryNode::Op::And: {
            auto false_label = new_label();
            auto end_label = new_label();
            auto result_name = new_temp();
            auto lhs = ensure_temp(eval(bin->lhs.get()));
            emit_expr(std::make_shared<MirIfFalseExpr>(lhs, false_label));
            emit_to_temp(result_name, eval(bin->rhs.get()));
            emit_expr(std::make_shared<MirGotoExpr>(end_label));
            emit_label(false_label);
            auto false_lit = std::make_shared<MirLiteralExpr>(MirLiteralKind::Boolean, "false");
            emit_to_temp(result_name, std::move(false_lit));
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
            emit_to_temp(result_name, eval(bin->rhs.get()));
            emit_expr(std::make_shared<MirGotoExpr>(end_label));
            emit_label(true_label);
            auto true_lit = std::make_shared<MirLiteralExpr>(MirLiteralKind::Boolean, "true");
            emit_to_temp(result_name, std::move(true_lit));
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
        auto *block = static_cast<BlockExprNode *>(expr);
        return process_block(block);
    }
    case ASTKind::Exprs: {
        auto *exprs = static_cast<ExprsNode *>(expr);
        std::shared_ptr<MirExpr> last;
        for (auto &e : exprs->exprs) {
            last = eval(e.get());
        }
        return last;
    }
    case ASTKind::SuffixParen: {
        auto *call = static_cast<SuffixParenNode *>(expr);
        std::string func_name;
        if (call->expr->kind == ASTKind::Identifier) {
            func_name = static_cast<IdentifierNode *>(call->expr.get())->id;
        }
        std::vector<std::shared_ptr<MirRefExpr>> arg_refs;
        if (call->suffix) {
            for (auto &arg : call->suffix->exprs) {
                auto arg_val = eval(arg.get());
                arg_refs.push_back(ensure_temp(std::move(arg_val)));
            }
        }
        auto call_expr = std::make_shared<MirCallFastExpr>(std::move(func_name), std::move(arg_refs));
        return temp_assign(std::move(call_expr));
    }
    case ASTKind::SuffixBracket: {
        auto *idx = static_cast<SuffixBracketNode *>(expr);
        eval(idx->expr.get());
        eval(idx->suffix.get());
        auto lit = std::make_shared<MirLiteralExpr>(MirLiteralKind::Integer, "0");
        return lit;
    }
    case ASTKind::IfExpr: {
        auto *if_expr = static_cast<IfExprNode *>(expr);
        auto else_label = new_label();
        auto end_label = new_label();
        auto result_name = new_temp();

        auto cond = ensure_temp(eval(if_expr->cond.get()));
        emit_expr(std::make_shared<MirIfFalseExpr>(cond, else_label));

        emit_to_temp(result_name, eval(if_expr->then.get()));
        emit_expr(std::make_shared<MirGotoExpr>(end_label));

        emit_label(else_label);
        if (if_expr->els) {
            emit_to_temp(result_name, eval(if_expr->els.get()));
        }
        emit_label(end_label);

        auto result_ref = std::make_shared<MirRefExpr>(result_name, true);
        return result_ref;
    }
    case ASTKind::AsExpr: {
        auto *as = static_cast<AsExprNode *>(expr);
        return eval(as->expr.get());
    }
    default:
        return std::make_shared<MirLiteralExpr>(MirLiteralKind::Integer, "0");
    }
}

} // namespace

MirModule MirBuilder::from_ast_module(const std::shared_ptr<Module> &ast) {
    MirModule mod;
    Builder builder(mod);
    builder.build(ast.get());
    return mod;
}

} // namespace lmx::mir
