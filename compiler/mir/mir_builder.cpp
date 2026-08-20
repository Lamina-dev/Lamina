#include "mir_builder.hpp"
#include "lmx_expr.h"
#include <cassert>
#include <functional>
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

runtime::ValueKind native_value_kind(const std::shared_ptr<Type>& type) noexcept {
    if (!type) return runtime::ValueKind::Obj;
    if (type->kind == TypeKind::Basic) {
        return std::static_pointer_cast<BasicType>(type)->type;
    }
    if (type->kind == TypeKind::Tuple) return runtime::ValueKind::Tuple;
    if (type->kind == TypeKind::Named) {
        const auto& name = std::static_pointer_cast<NamedType>(type)->name;
        if (name == "set") return runtime::ValueKind::Set;
        if (name == "interval") return runtime::ValueKind::Interval;
    }
    return runtime::ValueKind::Obj;
}

class Builder {
    std::vector<std::string> break_labels;
    std::vector<std::string> continue_labels;
    MirModule &module_;
    std::vector<std::shared_ptr<MirNode>> *emit_target_ = &module_.nodes;
    size_t temp_counter_ = 0;
    size_t label_counter_ = 0;
    bool cas_runtime_declared_ = false;

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
        //if (!expr) return nullptr;
        auto name = new_temp();
        emit(std::make_shared<MirTempAssign>(name, std::move(expr)));
        return std::make_shared<MirRefExpr>(name, true);
    }

    [[nodiscard]] std::shared_ptr<MirRefExpr> emit_to_temp(const std::string &name, std::shared_ptr<MirExpr> expr) const {
        emit(std::make_shared<MirTempAssign>(name, std::move(expr)));
        return std::make_shared<MirRefExpr>(name, true);
    }

    void ensure_cas_runtime() noexcept {
        if (cas_runtime_declared_) return;
        cas_runtime_declared_ = true;
        if (module_.lib_name.empty()) {
            module_.lib_name = "laminaCore";
        }
        const auto expr = runtime::ValueKind::Expr;
        const auto text = runtime::ValueKind::Obj;
        const auto integer = runtime::ValueKind::Int;
        const auto boolean = runtime::ValueKind::Bool;
        const auto va_list = runtime::ValueKind::C_VaList;
        const auto value_ref = runtime::ValueKind::C_ValueRef;
        emit(std::make_shared<MirNativeFuncDefine>("__cas_sym", "cas_sym", std::vector{text}, expr));
        emit(std::make_shared<MirNativeFuncDefine>("__cas_expr_imaginary", "cas_expr_imaginary", std::vector<runtime::ValueKind>{}, expr));
        emit(std::make_shared<MirNativeFuncDefine>("__cas_expr_integer", "cas_expr_integer", std::vector{integer}, expr));
        emit(std::make_shared<MirNativeFuncDefine>("__cas_expr_rational", "cas_expr_rational", std::vector{integer, integer}, expr));
        emit(std::make_shared<MirNativeFuncDefine>("__cas_expr_value", "cas_expr_value", std::vector{value_ref}, expr));
        emit(std::make_shared<MirNativeFuncDefine>("__cas_expr_unary", "cas_expr_unary", std::vector{integer, expr}, expr));
        emit(std::make_shared<MirNativeFuncDefine>("__cas_expr_binary", "cas_expr_binary", std::vector{integer, expr, expr}, expr));
        emit(std::make_shared<MirNativeFuncDefine>("__cas_expr_function", "cas_expr_function", std::vector{text, integer, va_list}, expr));
        emit(std::make_shared<MirNativeFuncDefine>("__cas_expr_set", "cas_expr_set", std::vector{integer, va_list}, expr));
        emit(std::make_shared<MirNativeFuncDefine>("__cas_expr_interval", "cas_expr_interval", std::vector{expr, expr, boolean, boolean}, expr));
    }

    std::shared_ptr<MirCCallExpr> cas_call(std::string name, std::vector<std::shared_ptr<MirRefExpr>> args) noexcept {
        ensure_cas_runtime();
        return std::make_shared<MirCCallExpr>(std::move(name), std::move(args));
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

    static bool is_expr_type(const std::shared_ptr<Type>& type) noexcept {
        return type && type->kind == TypeKind::Basic &&
               std::reinterpret_pointer_cast<BasicType>(type)->type == runtime::ValueKind::Expr;
    }

    static std::string dotted_name(const ExprNode *expr) {
        if (expr->kind == ASTKind::Identifier) {
            return reinterpret_cast<const IdentifierNode *>(expr)->id;
        }
        if (expr->kind != ASTKind::DotExpr) {
            return {};
        }
        const auto *dot = reinterpret_cast<const DotExprNode *>(expr);
        auto lhs = dotted_name(dot->expr.get());
        if (lhs.empty()) {
            return {};
        }
        return lhs + "." + dot->rhs->id;
    }

    std::shared_ptr<MirExpr> eval(ExprNode *expr);

    std::shared_ptr<MirRefExpr> integer_arg(const long long value) {
        return ensure_temp(std::make_shared<MirLiteralExpr>(
            MirLiteralKind::Integer, std::to_string(value)));
    }

    static LmExprBinaryOp expr_binary_op(const BinaryNode::Op op) {
        switch (op) {
        case BinaryNode::Op::Add: return LMX_EXPR_ADD;
        case BinaryNode::Op::Sub: return LMX_EXPR_SUB;
        case BinaryNode::Op::Mul: return LMX_EXPR_MUL;
        case BinaryNode::Op::Div: return LMX_EXPR_DIV;
        case BinaryNode::Op::Pow: return LMX_EXPR_POW;
        case BinaryNode::Op::Eq: return LMX_EXPR_EQ;
        case BinaryNode::Op::Ne: return LMX_EXPR_NE;
        case BinaryNode::Op::Gt: return LMX_EXPR_GT;
        case BinaryNode::Op::Ge: return LMX_EXPR_GE;
        case BinaryNode::Op::Lt: return LMX_EXPR_LT;
        case BinaryNode::Op::Le: return LMX_EXPR_LE;
        case BinaryNode::Op::And: return LMX_EXPR_AND;
        case BinaryNode::Op::Or: return LMX_EXPR_OR;
        case BinaryNode::Op::In: return LMX_EXPR_IN;
        case BinaryNode::Op::NotIn: return LMX_EXPR_NOT_IN;
        case BinaryNode::Op::Mod:
        case BinaryNode::Op::Bind: std::unreachable();
        }
        std::unreachable();
    }

    std::shared_ptr<MirExpr> eval_as_expr(ExprNode *expr) {
        switch (expr->kind) {
        case ASTKind::Identifier: {
            const auto* id = reinterpret_cast<const IdentifierNode*>(expr);
            if (id->id == "I")
                return cas_call("__cas_expr_imaginary", {});
            auto value = std::make_shared<MirRefExpr>(id->id, false);
            if (is_expr_type(expr->type)) return value;
            return cas_call("__cas_expr_value", {ensure_temp(std::move(value))});
        }
        case ASTKind::Literal: {
            const auto* literal = reinterpret_cast<const LiteralNode*>(expr);
            if (literal->kind == LiteralNode::Kind::Integer) {
                return cas_call("__cas_expr_integer", {integer_arg(std::stoll(literal->val))});
            }
            if (literal->kind == LiteralNode::Kind::Float) {
                const runtime::Fraction value(literal->val);
                return cas_call("__cas_expr_rational",
                                {integer_arg(value.num), integer_arg(value.den)});
            }
            std::unreachable();
        }
        case ASTKind::Unary: {
            const auto* unary = reinterpret_cast<const UnaryNode*>(expr);
            return cas_call("__cas_expr_unary", {
                integer_arg(unary->op == UnaryNode::Op::Neg ? LMX_EXPR_NEG : LMX_EXPR_NOT),
                ensure_temp(eval_as_expr(unary->expr.get()))});
        }
        case ASTKind::Binary: {
            const auto* binary = reinterpret_cast<const BinaryNode*>(expr);
            return cas_call("__cas_expr_binary", {
                integer_arg(expr_binary_op(binary->op)),
                ensure_temp(eval_as_expr(binary->lhs.get())),
                ensure_temp(eval_as_expr(binary->rhs.get()))});
        }
        case ASTKind::SuffixParen: {
            const auto* call = reinterpret_cast<const SuffixParenNode*>(expr);
            if (!call->is_symbolic_call) return eval(expr);
            std::vector<std::shared_ptr<MirRefExpr>> args;
            args.push_back(ensure_temp(std::make_shared<MirLiteralExpr>(
                MirLiteralKind::String, dotted_name(call->expr.get()))));
            const auto count = call->suffix ? call->suffix->exprs.size() : 0;
            args.push_back(integer_arg(static_cast<long long>(count)));
            if (call->suffix) {
                for (const auto& argument : call->suffix->exprs)
                    args.push_back(ensure_temp(eval_as_expr(argument.get())));
            }
            return cas_call("__cas_expr_function", std::move(args));
        }
        case ASTKind::LiteralPayload: {
            const auto* payload = reinterpret_cast<const LiteralPayloadNode*>(expr);
            if (payload->payload_kind == LiteralPayloadNode::Kind::Interval) {
                return cas_call("__cas_expr_interval", {
                    ensure_temp(eval_as_expr(payload->elements[0].get())),
                    ensure_temp(eval_as_expr(payload->elements[1].get())),
                    ensure_temp(std::make_shared<MirLiteralExpr>(
                        MirLiteralKind::Boolean, payload->lower_closed ? "true" : "false")),
                    ensure_temp(std::make_shared<MirLiteralExpr>(
                        MirLiteralKind::Boolean, payload->upper_closed ? "true" : "false"))});
            }
            std::vector<std::shared_ptr<MirRefExpr>> args;
            args.push_back(integer_arg(static_cast<long long>(payload->elements.size())));
            for (const auto& element : payload->elements)
                args.push_back(ensure_temp(eval_as_expr(element.get())));
            return cas_call("__cas_expr_set", std::move(args));
        }
        case ASTKind::AsExpr:
            return eval_as_expr(reinterpret_cast<const AsExprNode*>(expr)->expr.get());
        default:
            std::unreachable();
        }
    }

    std::shared_ptr<MirExpr> eval_adt_binding(BinaryNode *node) {
        std::vector<std::shared_ptr<MirRefExpr>> fields;
        fields.push_back(ensure_temp(eval(node->lhs.get())));
        fields.push_back(ensure_temp(eval(node->rhs.get())));
        return std::make_shared<MirAdtNewExpr>("Binding", "Binding", std::move(fields));
    }

    std::shared_ptr<MirExpr> eval_match(MatchExprNode *node) {
        const auto target = ensure_temp(eval(node->target.get()));
        const auto end_label = new_label();
        const auto result_name = new_temp();

        std::function<void(const Pattern&, const std::shared_ptr<MirExpr>&, const std::string&)> emit_pattern;
        emit_pattern = [&](const Pattern& pattern, const std::shared_ptr<MirExpr>& value, const std::string& fail_label) {
            switch (pattern.kind) {
            case Pattern::Kind::Wildcard:
                break;
            case Pattern::Kind::Binding:
                emit(std::make_shared<MirAssign>(pattern.name, value));
                break;
            case Pattern::Kind::Literal: {
                auto literal = eval(pattern.literal.get());
                emit_expr(std::make_shared<MirIfFalseExpr>(
                    std::make_shared<MirICmpEqExpr>(value, literal), fail_label));
                break;
            }
            case Pattern::Kind::Constructor: {
                emit_expr(std::make_shared<MirIfFalseExpr>(
                    std::make_shared<MirAdtIsExpr>(value, pattern.adt_type_name, pattern.name), fail_label));
                for (size_t i = 0; i < pattern.fields.size(); ++i) {
                    auto field = temp_assign(std::make_shared<MirAdtGetExpr>(value, static_cast<uint8_t>(i)));
                    emit_pattern(pattern.fields[i], field, fail_label);
                }
                break;
            }
            }
        };

        for (size_t i = 0; i < node->arms.size(); ++i) {
            const auto next_label = new_label();
            emit_pattern(node->arms[i].pattern, target, next_label);
            if (node->arms[i].guard) {
                emit_expr(std::make_shared<MirIfFalseExpr>(eval(node->arms[i].guard.get()), next_label));
            }
            const auto result = emit_to_temp(result_name, eval(node->arms[i].value.get()));
            emit_expr(std::make_shared<MirGotoExpr>(end_label));
            emit_label(next_label);
        }
        emit_label(end_label);
        return std::make_shared<MirRefExpr>(result_name, true);
    }

    std::shared_ptr<MirExpr> process_block(const BlockExprNode *block) {
        std::shared_ptr<MirExpr> block_val;
        for (auto &stmt : block->stmts) {
            if (stmt->kind == ASTKind::TailReturn) {
                const auto *tr = reinterpret_cast<TailReturnNode *>(stmt.get());
                block_val = eval(tr->expr.get());
            } else {
                process(stmt.get());
            }
        }
        return block_val;
    }

public:
    explicit Builder(MirModule &mod) noexcept : module_(mod)  {}

    void process(StmtNode *stmt) noexcept {
        switch (stmt->kind) {
        case ASTKind::ExprStmt: {
            const auto *node = reinterpret_cast<ExprStmtNode *>(stmt);
            auto e = eval(node->expr.get());
            if (e) emit_expr(std::move(e));
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
            if (node->lhs->kind == ASTKind::SuffixBracket) {
                auto *sb = reinterpret_cast<SuffixBracketNode *>(node->lhs.get());
                emit(std::make_shared<MirArrStore>(
                    ensure_temp(eval(sb->expr.get())),
                    ensure_temp(eval(sb->suffix.get())),
                    std::move(val)));
            } else if (node->lhs->kind == ASTKind::TupleGetExpr) {
                auto *tg = reinterpret_cast<TupleGetExprNode *>(node->lhs.get());
                emit(std::make_shared<MirTupleStore>(
                    ensure_temp(eval(tg->tup.get())),
                    tg->i,
                    std::move(val)));
            } else if (node->lhs->kind == ASTKind::Identifier) {
                auto *id = reinterpret_cast<IdentifierNode *>(node->lhs.get());
                emit(std::make_shared<MirAssign>(id->id, std::move(val)));
            } else {
                eval(node->lhs.get());
            }
            break;
        }
        case ASTKind::SymDecl: {
            const auto* node = reinterpret_cast<SymDeclNode *>(stmt);
            for (const auto& id : node->ids) {
                std::vector<std::shared_ptr<MirRefExpr>> args;
                args.push_back(ensure_temp(std::make_shared<MirLiteralExpr>(MirLiteralKind::String, id)));
                emit(std::make_shared<MirAssign>(id, cas_call("__cas_sym", std::move(args))));
            }
            break;
        }
        case ASTKind::TypeDecl:
            break;
        case ASTKind::FuncImpl: {
            auto *func = reinterpret_cast<FuncImplNode *>(stmt);
            if (!func->block) break;
            const auto save_tc = temp_counter_;
            const auto save_lc = label_counter_;
            const auto save_target = emit_target_;

            std::vector<std::shared_ptr<MirNode>> body;
            emit_target_ = &body;

            auto body_val = process_block(reinterpret_cast<BlockExprNode*>(func->block.get()));
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
            params.push_back(native_value_kind(ty));
        }
        const auto ret_ty = native_value_kind(node->return_type);
        emit(std::make_shared<MirNativeFuncDefine>(node->func_id, node->symbol, std::move(params), ret_ty));
    }

    void build_imported_native_decls(const Module *ast_mod) const noexcept {
        bool has_imported_native = false;
        for (const auto& [path, mod_ty] : ast_mod->imports) {
            for (const auto& exported : mod_ty->exports) {
                if (exported.type->kind != TypeKind::NativeFunction) continue;
                const auto native_type =
                    std::static_pointer_cast<NativeFunctionType>(exported.type);
                std::vector<runtime::ValueKind> params;
                for (const auto& param : native_type->params_ty) {
                    params.push_back(native_value_kind(param));
                }
                const auto ret_ty = native_value_kind(native_type->ret_ty);
                emit(std::make_shared<MirNativeFuncDefine>(
                    mod_ty->binding_name + "." + exported.name,
                    native_type->name,
                    std::move(params),
                    ret_ty));
                has_imported_native = true;
            }
        }
        if (has_imported_native && module_.lib_name.empty()) module_.lib_name = "laminaCore";
    }


    void build(const Module *ast_mod) {

        for (auto& n : ast_mod->native_funcs) {
            build_native_decl(n.get());
        }
        build_imported_native_decls(ast_mod);

        for (auto &decl : ast_mod->decls) {
            process(decl.get());
        }
        // for (auto &func : ast_mod->top_func_def) {
        //     process(&func);
        // }
    }
};

std::shared_ptr<MirExpr> Builder::eval(ExprNode *expr) {
    if (is_expr_type(expr->type)) {
        if (expr->kind == ASTKind::Literal ||
            expr->kind == ASTKind::LiteralPayload ||
            expr->kind == ASTKind::AsExpr) {
            return eval_as_expr(expr);
        }
        if (expr->kind == ASTKind::SuffixParen &&
            reinterpret_cast<SuffixParenNode*>(expr)->is_symbolic_call)
            return eval_as_expr(expr);
        if (expr->kind == ASTKind::Identifier) {
            const auto *id = reinterpret_cast<const IdentifierNode *>(expr);
            if (id->id == "I") {
                return eval_as_expr(expr);
            }
        }
    }
    switch (expr->kind) {
    case ASTKind::Literal: {
        auto *lit = reinterpret_cast<LiteralNode *>(expr);
        MirLiteralKind lk;
        switch (lit->kind) {
        case LiteralNode::Kind::Integer: lk = MirLiteralKind::Integer; break;
        case LiteralNode::Kind::Float:   lk = MirLiteralKind::Float;   break;
        case LiteralNode::Kind::String:  lk = MirLiteralKind::String;  break;
        case LiteralNode::Kind::Boolean: lk = MirLiteralKind::Boolean; break;
        case LiteralNode::Kind::Null:    lk = MirLiteralKind::Null;    break;
        }
        return std::make_shared<MirLiteralExpr>(lk, lit->val);
    }
    case ASTKind::Identifier: {
        auto *id = reinterpret_cast<IdentifierNode *>(expr);
        if (id->is_zero_adt_constructor) {
            return std::make_shared<MirAdtNewExpr>(id->adt_type_name, id->id,
                                                   std::vector<std::shared_ptr<MirRefExpr>>{});
        }
        auto ref = std::make_shared<MirRefExpr>(id->id, false);
        return ref;
    }
    case ASTKind::Unary: {
        auto *un = reinterpret_cast<UnaryNode *>(expr);
        if (is_expr_type(expr->type)) {
            return eval_as_expr(expr);
        }
        auto operand = ensure_temp(eval(un->expr.get()));
        if (is_int_type(expr->type.get())) {
            return temp_assign(std::make_shared<MirINegExpr>(std::move(operand)));
        }
        return temp_assign(std::make_shared<MirFNegExpr>(std::move(operand)));
    }
    case ASTKind::Binary: {
        auto *bin = reinterpret_cast<BinaryNode *>(expr);
        if (is_expr_type(expr->type)) {
            return eval_as_expr(expr);
        }
        if (bin->op == BinaryNode::Op::Bind) {
            return eval_adt_binding(bin);
        }
        if (bin->op == BinaryNode::Op::In || bin->op == BinaryNode::Op::NotIn) {
            return temp_assign(std::make_shared<MirContainsExpr>(
                eval(bin->lhs.get()), eval(bin->rhs.get()),
                bin->op == BinaryNode::Op::NotIn));
        }
        switch (bin->op) {
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
            bool is_float = is_float_type(bin->lhs->type.get());
            auto lhs = ensure_temp(eval(bin->lhs.get()));
            auto rhs = ensure_temp(eval(bin->rhs.get()));
            return temp_assign(eval_binary_arith(bin->op, std::move(lhs), std::move(rhs), is_float));
        }
        }
    }
    case ASTKind::LiteralPayload: {
        auto *payload = reinterpret_cast<LiteralPayloadNode *>(expr);
        std::vector<std::shared_ptr<MirRefExpr>> elements;
        elements.reserve(payload->elements.size());
        for (auto& element : payload->elements) {
            elements.push_back(ensure_temp(eval(element.get())));
        }
        return std::make_shared<MirLiteralNewExpr>(
            payload->payload_kind, std::move(elements),
            payload->lower_closed, payload->upper_closed);
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
        if (call->is_adt_constructor) {
            return std::make_shared<MirAdtNewExpr>(call->adt_type_name,
                                                   call->adt_constructor,
                                                   std::move(arg_refs));
        }
        std::shared_ptr<MirExpr> call_result;
        if (call->can_fast) {
            std::string func_name = reinterpret_cast<IdentifierNode *>(call->expr.get())->id;
            call_result = std::make_shared<MirCallFastExpr>(std::move(func_name), std::move(arg_refs));
        } else {
            const auto reg_func = temp_assign(std::move(eval(call->expr.get())));
            call_result = std::make_shared<MirCallExpr>(reg_func, std::move(arg_refs));
        }
        if (is_expr_type(expr->type) && call->expr->type->kind == TypeKind::Function) {
            const auto function = std::static_pointer_cast<FunctionType>(call->expr->type);
            if (!is_expr_type(function->ret_ty)) {
                return cas_call("__cas_expr_value", {ensure_temp(std::move(call_result))});
            }
        }
        return call_result;
    }
    case ASTKind::SuffixBracket: {
        auto *idx = reinterpret_cast<SuffixBracketNode *>(expr);
        auto target = ensure_temp(eval(idx->expr.get()));
        auto index = ensure_temp(eval(idx->suffix.get()));
        return temp_assign(std::make_shared<MirArrLoadExpr>(std::move(target), std::move(index)));
    }
    case ASTKind::IfExpr: {
        auto *if_expr = reinterpret_cast<IfExprNode *>(expr);
        auto else_label = new_label();
        auto end_label = new_label();
        std::string result_name;
        if (expr->have_ret_value()) result_name = new_temp();


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
        if (expr->have_ret_value()) return std::make_shared<MirRefExpr>(result_name, true);
        return nullptr;
    }
    case ASTKind::MatchExpr:
        return eval_match(reinterpret_cast<MatchExprNode*>(expr));
    case ASTKind::AsExpr: {
        auto *as = reinterpret_cast<AsExprNode *>(expr);
        return eval(as->expr.get());
    }
    case ASTKind::NativeFuncCall: {
        auto *call = reinterpret_cast<NativeFuncCallExpr*>(expr);
        auto func_name = dotted_name(call->expr.get());
        std::vector<std::shared_ptr<MirRefExpr>> arg_refs;
        if (call->suffix) {
            for (auto &arg : call->suffix->exprs) {
                auto arg_val = eval(arg.get());
                arg_refs.push_back(ensure_temp(std::move(arg_val)));
            }
        }
        auto call_expr = std::make_shared<MirCCallExpr>(std::move(func_name), std::move(arg_refs));
        return std::move(call_expr);
    }
    case ASTKind::DotExpr: {
        const auto *dot = reinterpret_cast<DotExprNode *>(expr);
        if (dot->is_zero_adt_constructor) {
            return std::make_shared<MirAdtNewExpr>(dot->adt_type_name, dot->rhs->id,
                                                   std::vector<std::shared_ptr<MirRefExpr>>{});
        }
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
    case ASTKind::ArrayLiteral: {
        auto *arr = reinterpret_cast<ArrayLiteralNode *>(expr);
        std::vector<std::shared_ptr<MirExpr>> elements;
        elements.reserve(arr->exprs.size());
        for (auto &e : arr->exprs) {
            elements.push_back(eval(e.get()));
        }
        return std::make_shared<MirArrayExpr>(arr->is_constant(), std::move(elements));
    }
    case ASTKind::TupleLiteral: {
        auto* tup = reinterpret_cast<TupleLiteralNode*>(expr);
        std::vector<std::shared_ptr<MirExpr>> elements;
        elements.reserve(tup->exprs.size());
        for (auto &e : tup->exprs) {
            elements.push_back(eval(e.get()));
        }
        return std::make_shared<MirTupleExpr>(tup->is_constant(), std::move(elements));
    }
    case ASTKind::TupleGetExpr: {
        auto* tg = reinterpret_cast<TupleGetExprNode*>(expr);
        auto target = ensure_temp(eval(tg->tup.get()));
        return temp_assign(std::make_shared<MirTupleGetExpr>(std::move(target), tg->i));
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
