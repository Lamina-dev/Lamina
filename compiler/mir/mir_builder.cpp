#include "mir_builder.hpp"
#include "lmx_expr.h"
#include <cassert>
#include <cstdlib>
#include <functional>
#include <memory>
#include <optional>
#include <ranges>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace lmx::mir {

namespace {

bool is_int_type(const Type *type) noexcept {
    return type && type->kind == TypeKind::Basic &&
           reinterpret_cast<const BasicType *>(type)->type == runtime::ValueKind::Int;
}

bool is_float_type(const Type *type) noexcept {
    if (!type) return false;
    if (type->kind == TypeKind::Dimensioned) return true;
    if (type->kind != TypeKind::Basic) return false;
    const auto kind =
        reinterpret_cast<const BasicType *>(type)->type;
    return kind == runtime::ValueKind::Fraction ||
           kind == runtime::ValueKind::Real;
}

bool is_comparison_op(const BinaryNode::Op op) noexcept {
    switch (op) {
    case BinaryNode::Op::Eq:
    case BinaryNode::Op::Ne:
    case BinaryNode::Op::Lt:
    case BinaryNode::Op::Le:
    case BinaryNode::Op::Gt:
    case BinaryNode::Op::Ge:
        return true;
    default:
        return false;
    }
}

bool is_named_type(const std::shared_ptr<Type>& type,
                   const std::string_view name) noexcept {
    return type && type->kind == TypeKind::Named &&
           std::static_pointer_cast<NamedType>(type)->name == name;
}

std::optional<int> signed_integer_literal(const ExprNode* expression) noexcept {
    if (!expression) return std::nullopt;
    if (expression->kind == ASTKind::Literal) {
        const auto* literal = static_cast<const LiteralNode*>(expression);
        if (literal->kind != LiteralNode::Kind::Integer) return std::nullopt;
        try {
            std::size_t used = 0;
            const auto value = std::stoi(literal->val, &used);
            return used == literal->val.size()
                ? std::optional<int>(value) : std::nullopt;
        } catch (...) {
            return std::nullopt;
        }
    }
    if (expression->kind != ASTKind::Unary) return std::nullopt;
    const auto* unary = static_cast<const UnaryNode*>(expression);
    if (unary->op != UnaryNode::Op::Neg) return std::nullopt;
    const auto value = signed_integer_literal(unary->expr.get());
    return value ? std::optional<int>(-*value) : std::nullopt;
}

runtime::ValueKind native_value_kind(const std::shared_ptr<Type>& type) noexcept {
    if (!type) return runtime::ValueKind::Obj;
    if (type->kind == TypeKind::Basic) {
        return std::static_pointer_cast<BasicType>(type)->type;
    }
    if (type->kind == TypeKind::Dimensioned) return runtime::ValueKind::Fraction;
    if (type->kind == TypeKind::Tuple) return runtime::ValueKind::Tuple;
    if (type->kind == TypeKind::Function) return runtime::ValueKind::C_Ptr;
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
    bool expression_runtime_declared_ = false;

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

    void ensure_expression_runtime() noexcept {
        if (expression_runtime_declared_) return;
        expression_runtime_declared_ = true;
        if (module_.lib_name.empty()) {
            module_.lib_name = "liblamina";
        }
        const auto expr = runtime::ValueKind::Expr;
        const auto text = runtime::ValueKind::Obj;
        const auto integer = runtime::ValueKind::Int;
        const auto boolean = runtime::ValueKind::Bool;
        const auto va_list = runtime::ValueKind::C_VaList;
        const auto value_ref = runtime::ValueKind::C_ValueRef;
        const auto declare = [this](std::shared_ptr<MirNativeFuncDefine> definition) {
            module_.nodes.push_back(std::move(definition));
        };
        declare(std::make_shared<MirNativeFuncDefine>("__lmx_computer_algebra_expression_symbol", "lmx_computer_algebra_expression_symbol", std::vector{text}, expr));
        declare(std::make_shared<MirNativeFuncDefine>("__lmx_computer_algebra_expression_imaginary_unit", "lmx_computer_algebra_expression_imaginary_unit", std::vector<runtime::ValueKind>{}, expr));
        declare(std::make_shared<MirNativeFuncDefine>("__lmx_computer_algebra_expression_integer", "lmx_computer_algebra_expression_integer", std::vector{integer}, expr));
        declare(std::make_shared<MirNativeFuncDefine>("__lmx_computer_algebra_expression_rational", "lmx_computer_algebra_expression_rational", std::vector{integer, integer}, expr));
        declare(std::make_shared<MirNativeFuncDefine>("__lmx_computer_algebra_expression_promote_value", "lmx_computer_algebra_expression_promote_value", std::vector{value_ref}, expr));
        declare(std::make_shared<MirNativeFuncDefine>("__lmx_computer_algebra_expression_unary", "lmx_computer_algebra_expression_unary", std::vector{integer, expr}, expr));
        declare(std::make_shared<MirNativeFuncDefine>("__lmx_computer_algebra_expression_binary", "lmx_computer_algebra_expression_binary", std::vector{integer, expr, expr}, expr));
        declare(std::make_shared<MirNativeFuncDefine>("__lmx_computer_algebra_expression_function", "lmx_computer_algebra_expression_function", std::vector{text, integer, va_list}, expr));
        declare(std::make_shared<MirNativeFuncDefine>("__lmx_computer_algebra_expression_set", "lmx_computer_algebra_expression_set", std::vector{integer, va_list}, expr));
        declare(std::make_shared<MirNativeFuncDefine>("__lmx_computer_algebra_expression_interval", "lmx_computer_algebra_expression_interval", std::vector{expr, expr, boolean, boolean}, expr));
        declare(std::make_shared<MirNativeFuncDefine>("__lmx_computer_algebra_expression_attach_unit", "lmx_computer_algebra_expression_attach_unit",
            std::vector{expr, text, text, integer, integer}, expr));
        declare(std::make_shared<MirNativeFuncDefine>("__lmx_computer_algebra_expression_convert_unit", "lmx_computer_algebra_expression_convert_unit",
            std::vector{expr, text, text, integer, integer}, expr));
        declare(std::make_shared<MirNativeFuncDefine>("__lmx_computer_algebra_expression_strip_base_value", "lmx_computer_algebra_expression_strip_base_value", std::vector{expr}, expr));
        declare(std::make_shared<MirNativeFuncDefine>("__lmx_computer_algebra_expression_strip_display_value", "lmx_computer_algebra_expression_strip_display_value", std::vector{expr}, expr));
    }

    std::shared_ptr<MirCCallExpr> expression_call(std::string name, std::vector<std::shared_ptr<MirRefExpr>> args) noexcept {
        ensure_expression_runtime();
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
        case BinaryNode::Op::Lt:
            return is_float
                ? std::shared_ptr<MirExpr>(std::make_shared<MirFCmpLtExpr>(std::move(lhs), std::move(rhs)))
                : std::make_shared<MirICmpLtExpr>(std::move(lhs), std::move(rhs));
        case BinaryNode::Op::Gt:
            return is_float
                ? std::shared_ptr<MirExpr>(std::make_shared<MirFCmpGtExpr>(std::move(lhs), std::move(rhs)))
                : std::make_shared<MirICmpGtExpr>(std::move(lhs), std::move(rhs));
        case BinaryNode::Op::Ge:
            return is_float
                ? std::shared_ptr<MirExpr>(std::make_shared<MirFCmpGeExpr>(std::move(lhs), std::move(rhs)))
                : std::make_shared<MirICmpGeExpr>(std::move(lhs), std::move(rhs));
        case BinaryNode::Op::Le:
            return is_float
                ? std::shared_ptr<MirExpr>(std::make_shared<MirFCmpLeExpr>(std::move(lhs), std::move(rhs)))
                : std::make_shared<MirICmpLeExpr>(std::move(lhs), std::move(rhs));
        case BinaryNode::Op::Eq:
            return is_float
                ? std::shared_ptr<MirExpr>(std::make_shared<MirFCmpEqExpr>(std::move(lhs), std::move(rhs)))
                : std::make_shared<MirICmpEqExpr>(std::move(lhs), std::move(rhs));
        case BinaryNode::Op::Ne:
            return is_float
                ? std::shared_ptr<MirExpr>(std::make_shared<MirFCmpNeExpr>(std::move(lhs), std::move(rhs)))
                : std::make_shared<MirICmpNeExpr>(std::move(lhs), std::move(rhs));
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
                                             std::shared_ptr<MirExpr> rhs, const bool is_float) {
        switch (op) {
        case BinaryNode::Op::Eq:
            if (is_float) return std::make_shared<MirFCmpEqExpr>(std::move(lhs), std::move(rhs));
            return std::make_shared<MirICmpEqExpr>(std::move(lhs), std::move(rhs));
        case BinaryNode::Op::Ne:
            if (is_float) return std::make_shared<MirFCmpNeExpr>(std::move(lhs), std::move(rhs));
            return std::make_shared<MirICmpNeExpr>(std::move(lhs), std::move(rhs));
        case BinaryNode::Op::Lt:
            if (is_float) return std::make_shared<MirFCmpLtExpr>(std::move(lhs), std::move(rhs));
            return std::make_shared<MirICmpLtExpr>(std::move(lhs), std::move(rhs));
        case BinaryNode::Op::Le:
            if (is_float) return std::make_shared<MirFCmpLeExpr>(std::move(lhs), std::move(rhs));
            return std::make_shared<MirICmpLeExpr>(std::move(lhs), std::move(rhs));
        case BinaryNode::Op::Gt:
            if (is_float) return std::make_shared<MirFCmpGtExpr>(std::move(lhs), std::move(rhs));
            return std::make_shared<MirICmpGtExpr>(std::move(lhs), std::move(rhs));
        case BinaryNode::Op::Ge:
            if (is_float) return std::make_shared<MirFCmpGeExpr>(std::move(lhs), std::move(rhs));
            return std::make_shared<MirICmpGeExpr>(std::move(lhs), std::move(rhs));
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

    std::shared_ptr<MirRefExpr> string_arg(std::string value) {
        return ensure_temp(std::make_shared<MirLiteralExpr>(
            MirLiteralKind::String, std::move(value)));
    }

    std::vector<std::shared_ptr<MirRefExpr>> unit_arguments(
        std::shared_ptr<MirExpr> value, const UnitDefinition& unit) {
        return {ensure_temp(std::move(value)), string_arg(unit.display_unit),
                string_arg(unit.dimension.to_string()),
                integer_arg(unit.scale_to_base.numerator),
                integer_arg(unit.scale_to_base.denominator)};
    }

    std::shared_ptr<MirExpr> promote_runtime_value(
        std::shared_ptr<MirExpr> value, const std::shared_ptr<Type>& type) {
        auto promoted = expression_call("__lmx_computer_algebra_expression_promote_value", {
            ensure_temp(std::move(value))});
        if (!type || type->kind != TypeKind::Dimensioned) return promoted;
        const auto dimensioned = std::static_pointer_cast<DimensionedType>(type);
        return expression_call("__lmx_computer_algebra_expression_attach_unit",
                        unit_arguments(std::move(promoted), dimensioned->unit));
    }

    std::shared_ptr<MirExpr> apply_scale(std::shared_ptr<MirExpr> value,
                                         const RationalScale& scale) {
        if (scale.numerator == scale.denominator) return value;
        auto factor = std::make_shared<MirLiteralExpr>(
            MirLiteralKind::Float, scale.to_string());
        return std::make_shared<MirFMulExpr>(
            ensure_temp(std::move(value)), ensure_temp(std::move(factor)));
    }

    static LmExprBinaryOp expr_binary_op(const BinaryNode::Op op) {
        switch (op) {
        case BinaryNode::Op::Add: return LMX_EXPRESSION_OPERATION_ADD;
        case BinaryNode::Op::Sub: return LMX_EXPRESSION_OPERATION_SUB;
        case BinaryNode::Op::Mul: return LMX_EXPRESSION_OPERATION_MUL;
        case BinaryNode::Op::Div: return LMX_EXPRESSION_OPERATION_DIV;
        case BinaryNode::Op::Pow: return LMX_EXPRESSION_OPERATION_POW;
        case BinaryNode::Op::Eq: return LMX_EXPRESSION_OPERATION_EQ;
        case BinaryNode::Op::Ne: return LMX_EXPRESSION_OPERATION_NE;
        case BinaryNode::Op::Gt: return LMX_EXPRESSION_OPERATION_GT;
        case BinaryNode::Op::Ge: return LMX_EXPRESSION_OPERATION_GE;
        case BinaryNode::Op::Lt: return LMX_EXPRESSION_OPERATION_LT;
        case BinaryNode::Op::Le: return LMX_EXPRESSION_OPERATION_LE;
        case BinaryNode::Op::And: return LMX_EXPRESSION_OPERATION_AND;
        case BinaryNode::Op::Or: return LMX_EXPRESSION_OPERATION_OR;
        case BinaryNode::Op::In: return LMX_EXPRESSION_OPERATION_IN;
        case BinaryNode::Op::NotIn: return LMX_EXPRESSION_OPERATION_NOT_IN;
        case BinaryNode::Op::Mod:
        case BinaryNode::Op::Bind:
        case BinaryNode::Op::SetUnion:
        case BinaryNode::Op::SetIntersection:
        case BinaryNode::Op::SetSymmetricDifference:
        case BinaryNode::Op::Subset:
            std::unreachable();
        }
        std::unreachable();
    }

    std::shared_ptr<MirExpr> eval_as_expr(ExprNode *expr) {
        switch (expr->kind) {
        case ASTKind::Identifier: {
            const auto* id = reinterpret_cast<const IdentifierNode*>(expr);
            if (id->id == "I")
                return expression_call("__lmx_computer_algebra_expression_imaginary_unit", {});
            auto value = std::make_shared<MirRefExpr>(
                id->compiled_symbol.empty() ? id->id : id->compiled_symbol, false);
            if (is_expr_type(expr->type) && !expr->promoted_from_type) return value;
            return promote_runtime_value(
                std::move(value), expr->promoted_from_type
                    ? expr->promoted_from_type : expr->type);
        }
        case ASTKind::Literal: {
            const auto* literal = reinterpret_cast<const LiteralNode*>(expr);
            if (literal->kind == LiteralNode::Kind::Integer) {
                return expression_call("__lmx_computer_algebra_expression_integer", {integer_arg(std::stoll(literal->val))});
            }
            if (literal->kind == LiteralNode::Kind::Float) {
                const runtime::Fraction value(literal->val);
                return expression_call("__lmx_computer_algebra_expression_rational",
                                {integer_arg(value.num), integer_arg(value.den)});
            }
            std::unreachable();
        }
        case ASTKind::Unary: {
            const auto* unary = reinterpret_cast<const UnaryNode*>(expr);
            return expression_call("__lmx_computer_algebra_expression_unary", {
                integer_arg(unary->op == UnaryNode::Op::Neg ? LMX_EXPRESSION_OPERATION_NEG : LMX_EXPRESSION_OPERATION_NOT),
                ensure_temp(eval_as_expr(unary->expr.get()))});
        }
        case ASTKind::Binary: {
            const auto* binary = reinterpret_cast<const BinaryNode*>(expr);
            return expression_call("__lmx_computer_algebra_expression_binary", {
                integer_arg(expr_binary_op(binary->op)),
                ensure_temp(eval_as_expr(binary->lhs.get())),
                ensure_temp(eval_as_expr(binary->rhs.get()))});
        }
        case ASTKind::SuffixParen: {
            const auto* call = reinterpret_cast<const SuffixParenNode*>(expr);
            if (!call->is_symbolic_call) {
                std::vector<std::shared_ptr<MirRefExpr>> arguments;
                if (call->suffix) {
                    for (const auto& argument : call->suffix->exprs)
                        arguments.push_back(ensure_temp(eval(argument.get())));
                }
                std::shared_ptr<MirExpr> value;
                if (call->can_fast) {
                    const auto* identifier =
                        static_cast<IdentifierNode*>(call->expr.get());
                    value = std::make_shared<MirCallFastExpr>(
                        identifier->compiled_symbol.empty()
                            ? identifier->id : identifier->compiled_symbol,
                        std::move(arguments));
                } else {
                    value = std::make_shared<MirCallExpr>(
                        temp_assign(eval(call->expr.get())), std::move(arguments));
                }
                if (expr->promoted_from_type) {
                    return promote_runtime_value(
                        std::move(value), expr->promoted_from_type);
                }
                if (is_expr_type(expr->type)) return value;
                return promote_runtime_value(std::move(value), expr->type);
            }
            std::vector<std::shared_ptr<MirRefExpr>> args;
            args.push_back(ensure_temp(std::make_shared<MirLiteralExpr>(
                MirLiteralKind::String, dotted_name(call->expr.get()))));
            const auto count = call->suffix ? call->suffix->exprs.size() : 0;
            args.push_back(integer_arg(static_cast<long long>(count)));
            if (call->suffix) {
                for (const auto& argument : call->suffix->exprs)
                    args.push_back(ensure_temp(eval_as_expr(argument.get())));
            }
            return expression_call("__lmx_computer_algebra_expression_function", std::move(args));
        }
        case ASTKind::LiteralPayload: {
            const auto* payload = reinterpret_cast<const LiteralPayloadNode*>(expr);
            if (payload->payload_kind == LiteralPayloadNode::Kind::Interval) {
                return expression_call("__lmx_computer_algebra_expression_interval", {
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
            return expression_call("__lmx_computer_algebra_expression_set", std::move(args));
        }
        case ASTKind::NativeFuncCall: {
            const auto* call = reinterpret_cast<const NativeFuncCallExpr*>(expr);
            std::vector<std::shared_ptr<MirRefExpr>> arguments;
            if (call->suffix) {
                for (const auto& argument : call->suffix->exprs)
                    arguments.push_back(ensure_temp(eval(argument.get())));
            }
            auto name = dotted_name(call->expr.get());
            if (!call->adt_constructor.empty()) {
                name += "\x1f";
                name += call->adt_constructor;
            }
            auto value = std::make_shared<MirCCallExpr>(
                std::move(name), std::move(arguments));
            if (expr->promoted_from_type)
                return promote_runtime_value(std::move(value), expr->promoted_from_type);
            if (is_expr_type(expr->type)) return value;
            return promote_runtime_value(std::move(value), expr->type);
        }
        case ASTKind::UnitAnnotated: {
            const auto* unit = reinterpret_cast<const UnitAnnotatedExprNode*>(expr);
            return expression_call("__lmx_computer_algebra_expression_attach_unit",
                            unit_arguments(eval_as_expr(unit->value.get()), unit->resolved_unit));
        }
        case ASTKind::AsExpr: {
            const auto* as = reinterpret_cast<const AsExprNode*>(expr);
            auto value = eval_as_expr(as->expr.get());
            switch (as->cast_kind) {
            case AsExprNode::Kind::Unit:
                return expression_call("__lmx_computer_algebra_expression_convert_unit",
                                unit_arguments(std::move(value), as->resolved_unit));
            case AsExprNode::Kind::Num:
                return expression_call("__lmx_computer_algebra_expression_strip_base_value", {ensure_temp(std::move(value))});
            case AsExprNode::Kind::Scalar:
                return expression_call("__lmx_computer_algebra_expression_strip_display_value", {ensure_temp(std::move(value))});
            case AsExprNode::Kind::Type:
                return value;
            }
            std::unreachable();
        }
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
                emit(std::make_shared<MirAssign>(id, expression_call("__lmx_computer_algebra_expression_symbol", std::move(args))));
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
            emit(std::make_shared<MirFuncDefine>(
                func->compiled_symbol.empty() ? func->func_id
                                              : func->compiled_symbol,
                std::move(params), std::move(body)));
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

                auto result_tmp = new_temp();
                emit(std::make_shared<MirTempAssign>(
                    result_tmp,
                    std::make_shared<MirICmpEqExpr>(
                            left_result, std::make_shared<MirRefExpr>(zero, true)
                        )
                ));

                auto if_expr = std::make_shared<MirIfTrueExpr>(std::make_shared<MirRefExpr>(result_tmp, true), bl);
                emit(std::make_shared<MirExprNode>(if_expr));
            }

            for (auto& s : node->body) {
                process(s.get());
            }


            if (node->expr) {
                auto one = new_temp();
                auto one_lit = std::make_shared<MirLiteralExpr>(MirLiteralKind::Integer, "1");
                emit(std::make_shared<MirTempAssign>(one, one_lit));

                emit(std::make_shared<MirTempAssign>(
                    reinterpret_cast<MirRefExpr*>(left_result.get())->name,
                    std::make_shared<MirISubExpr>(left_result, one_lit)
                    ));
            }


            emit(std::make_shared<MirExprNode>(std::make_shared<MirGotoExpr>(cl)));

            emit(std::make_shared<MirLabel>(bl));

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
        const auto declare_exports = [this, &has_imported_native](
            const auto& self,
            const std::string& prefix,
            const std::shared_ptr<ModuleType>& module_type) -> void {
            for (const auto& exported : module_type->exports) {
                const auto qualified_name = prefix + "." + exported.name;
                if (exported.type->kind == TypeKind::Module) {
                    self(self, qualified_name,
                         std::static_pointer_cast<ModuleType>(exported.type));
                    continue;
                }
                if (exported.type->kind != TypeKind::NativeFunction) continue;
                const auto native_type =
                    std::static_pointer_cast<NativeFunctionType>(exported.type);
                std::vector<runtime::ValueKind> params;
                for (const auto& param : native_type->params_ty) {
                    params.push_back(native_value_kind(param));
                }
                const auto ret_ty = native_value_kind(native_type->ret_ty);
                const auto overload_name =
                    qualified_name + "\x1f" + native_type->name;
                emit(std::make_shared<MirNativeFuncDefine>(
                    qualified_name,
                    native_type->name,
                    params,
                    ret_ty));
                emit(std::make_shared<MirNativeFuncDefine>(
                    overload_name,
                    native_type->name,
                    std::move(params),
                    ret_ty));
                has_imported_native = true;
            }
        };
        for (const auto& [path, mod_ty] : ast_mod->imports) {
            declare_exports(declare_exports, mod_ty->binding_name, mod_ty);
        }
        if (has_imported_native && module_.lib_name.empty()) module_.lib_name = "liblamina";
    }


    void build(const Module *ast_mod) {

        for (auto& n : ast_mod->native_funcs) {
            build_native_decl(n.get());
        }
        build_imported_native_decls(ast_mod);

        for (auto &decl : ast_mod->decls) {
            process(decl.get());
        }
        for (const auto& builtin : ast_mod->builtin_functions) {
            std::vector<std::shared_ptr<MirNode>> body;
            switch (builtin.kind) {
            case SyntheticBuiltinKind::Raise:
                body.push_back(std::make_shared<MirExprNode>(
                    std::make_shared<MirRaiseExpr>(
                        std::make_shared<MirRefExpr>(
                            builtin.parameter_name, false))));
                break;
            }
            emit(std::make_shared<MirFuncDefine>(
                builtin.compiled_symbol,
                std::vector<std::string>{builtin.parameter_name},
                std::move(body)));
        }
    }
};

std::shared_ptr<MirExpr> Builder::eval(ExprNode *expr) {
    if (is_expr_type(expr->type)) {
        if (expr->promoted_from_type) return eval_as_expr(expr);
        if (expr->kind == ASTKind::Literal ||
            expr->kind == ASTKind::LiteralPayload ||
            expr->kind == ASTKind::UnitAnnotated ||
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
    case ASTKind::UnitAnnotated: {
        const auto* unit = reinterpret_cast<UnitAnnotatedExprNode*>(expr);
        return eval(unit->value.get());
    }
    case ASTKind::Identifier: {
        auto *id = reinterpret_cast<IdentifierNode *>(expr);
        if (id->is_zero_adt_constructor) {
            return std::make_shared<MirAdtNewExpr>(id->adt_type_name, id->id,
                                                   std::vector<std::shared_ptr<MirRefExpr>>{});
        }
        return std::make_shared<MirRefExpr>(
            id->compiled_symbol.empty() ? id->id : id->compiled_symbol, false);
    }
    case ASTKind::Unary: {
        auto *un = reinterpret_cast<UnaryNode *>(expr);
        if (is_expr_type(expr->type)) {
            return eval_as_expr(expr);
        }
        auto operand = ensure_temp(eval(un->expr.get()));
        if (un->op == UnaryNode::Op::Not) {
            auto false_lit = std::make_shared<MirLiteralExpr>(MirLiteralKind::Boolean, "false");
            auto false_ref = ensure_temp(std::move(false_lit));
            return temp_assign(std::make_shared<MirICmpEqExpr>(
                std::move(operand), std::move(false_ref)));
        }
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
            auto element = eval(bin->lhs.get());
            if (bin->lhs->type->kind == TypeKind::Dimensioned &&
                is_named_type(bin->rhs->type, "interval")) {
                const auto dimensioned =
                    std::static_pointer_cast<DimensionedType>(bin->lhs->type);
                element = apply_scale(std::move(element),
                                      dimensioned->unit.scale_to_base);
            }
            return temp_assign(std::make_shared<MirContainsExpr>(
                std::move(element), eval(bin->rhs.get()),
                bin->op == BinaryNode::Op::NotIn));
        }
        const bool set_difference =
            bin->op == BinaryNode::Op::Sub &&
            is_named_type(bin->lhs->type, "set");
        if (bin->op == BinaryNode::Op::SetUnion ||
            bin->op == BinaryNode::Op::SetIntersection ||
            bin->op == BinaryNode::Op::SetSymmetricDifference ||
            bin->op == BinaryNode::Op::Subset ||
            set_difference) {
            runtime::Opcode::Opcode opcode;
            switch (bin->op) {
            case BinaryNode::Op::SetUnion:
                opcode = runtime::Opcode::SetUnion;
                break;
            case BinaryNode::Op::SetIntersection:
                opcode = runtime::Opcode::SetIntersection;
                break;
            case BinaryNode::Op::SetSymmetricDifference:
                opcode = runtime::Opcode::SetSymmetricDifference;
                break;
            case BinaryNode::Op::Subset:
                opcode = runtime::Opcode::SetSubset;
                break;
            case BinaryNode::Op::Sub:
                opcode = runtime::Opcode::SetDifference;
                break;
            default:
                std::unreachable();
            }
            return temp_assign(std::make_shared<MirSetBinaryExpr>(
                opcode, eval(bin->lhs.get()), eval(bin->rhs.get())));
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
            if (is_comparison_op(bin->op)) {
                const bool is_float = is_float_type(bin->lhs->type.get()) ||
                                      is_float_type(bin->rhs->type.get());
                auto lhs = ensure_temp(eval(bin->lhs.get()));
                auto rhs = ensure_temp(eval(bin->rhs.get()));
                return temp_assign(eval_binary_cmp(bin->op, std::move(lhs), std::move(rhs), is_float));
            }
            const auto exponent_value = signed_integer_literal(bin->rhs.get());
            if (bin->op == BinaryNode::Op::Pow &&
                bin->lhs->type->kind == TypeKind::Dimensioned && exponent_value) {
                const auto exponent = *exponent_value;
                auto base = ensure_temp(eval(bin->lhs.get()));
                if (exponent == 0) {
                    return std::make_shared<MirLiteralExpr>(MirLiteralKind::Float, "1");
                }
                std::shared_ptr<MirExpr> result = base;
                for (int i = 1; i < std::abs(exponent); ++i) {
                    result = temp_assign(std::make_shared<MirFMulExpr>(
                        ensure_temp(std::move(result)), base));
                }
                if (exponent < 0) {
                    result = temp_assign(std::make_shared<MirFDivExpr>(
                        ensure_temp(std::make_shared<MirLiteralExpr>(
                            MirLiteralKind::Float, "1")),
                        ensure_temp(std::move(result))));
                }
                return result;
            }
            bool is_float = is_float_type(bin->lhs->type.get()) ||
                            is_float_type(bin->rhs->type.get()) ||
                            is_float_type(expr->type.get());
            auto lhs_value = eval(bin->lhs.get());
            auto rhs_value = eval(bin->rhs.get());
            if ((bin->op == BinaryNode::Op::Mul || bin->op == BinaryNode::Op::Div) &&
                expr->type->kind != TypeKind::Dimensioned &&
                bin->lhs->type->kind == TypeKind::Dimensioned &&
                bin->rhs->type->kind == TypeKind::Dimensioned) {
                const auto lhs_type =
                    std::static_pointer_cast<DimensionedType>(bin->lhs->type);
                const auto rhs_type =
                    std::static_pointer_cast<DimensionedType>(bin->rhs->type);
                lhs_value = apply_scale(std::move(lhs_value),
                                        lhs_type->unit.scale_to_base);
                rhs_value = apply_scale(std::move(rhs_value),
                                        rhs_type->unit.scale_to_base);
            }
            auto lhs = ensure_temp(std::move(lhs_value));
            auto rhs = ensure_temp(std::move(rhs_value));
            return temp_assign(eval_binary_arith(bin->op, std::move(lhs), std::move(rhs), is_float));
        }
        }
    }
    case ASTKind::LiteralPayload: {
        auto *payload = reinterpret_cast<LiteralPayloadNode *>(expr);
        std::vector<std::shared_ptr<MirRefExpr>> elements;
        elements.reserve(payload->elements.size());
        for (auto& element : payload->elements) {
            auto value = eval(element.get());
            if (payload->payload_kind == LiteralPayloadNode::Kind::Interval &&
                element->type->kind == TypeKind::Dimensioned) {
                const auto dimensioned =
                    std::static_pointer_cast<DimensionedType>(element->type);
                value = apply_scale(std::move(value),
                                    dimensioned->unit.scale_to_base);
            }
            elements.push_back(ensure_temp(std::move(value)));
        }
        return std::make_shared<MirLiteralNewExpr>(
            payload->payload_kind, std::move(elements),
            payload->lower_closed, payload->upper_closed);
    }
    case ASTKind::Block: {
        auto *block = reinterpret_cast<BlockExprNode *>(expr);
        return process_block(block);
    }
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
            const auto* identifier =
                static_cast<IdentifierNode*>(call->expr.get());
            const auto& func_name = identifier->compiled_symbol.empty()
                ? identifier->id : identifier->compiled_symbol;
            call_result = std::make_shared<MirCallFastExpr>(
                func_name, std::move(arg_refs));
        } else {
            const auto reg_func = temp_assign(std::move(eval(call->expr.get())));
            call_result = std::make_shared<MirCallExpr>(reg_func, std::move(arg_refs));
        }
        if (is_expr_type(expr->type) && call->expr->type->kind == TypeKind::Function) {
            const auto function = std::static_pointer_cast<FunctionType>(call->expr->type);
            if (!is_expr_type(function->ret_ty)) {
                return expression_call("__lmx_computer_algebra_expression_promote_value", {ensure_temp(std::move(call_result))});
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
        if (as->cast_kind == AsExprNode::Kind::Type) return eval(as->expr.get());
        if (is_expr_type(as->expr->type)) return eval_as_expr(expr);
        auto value = eval(as->expr.get());
        if (as->cast_kind == AsExprNode::Kind::Scalar) return value;
        if (as->expr->type->kind != TypeKind::Dimensioned) return value;
        const auto source = std::static_pointer_cast<DimensionedType>(as->expr->type);
        if (as->cast_kind == AsExprNode::Kind::Num) {
            return apply_scale(std::move(value), source->unit.scale_to_base);
        }
        const auto factor = source->unit.scale_to_base.divided_by(
            as->resolved_unit.scale_to_base);
        return factor ? apply_scale(std::move(value), *factor) : value;
    }
    case ASTKind::NativeFuncCall: {
        auto *call = reinterpret_cast<NativeFuncCallExpr*>(expr);
        std::vector<std::shared_ptr<MirRefExpr>> arg_refs;
        if (call->suffix) {
            for (auto &arg : call->suffix->exprs) {
                auto arg_val = eval(arg.get());
                arg_refs.push_back(ensure_temp(std::move(arg_val)));
            }
        }
        auto func_name = dotted_name(call->expr.get());
        if (!call->adt_constructor.empty()) {
            func_name += "\x1f";
            func_name += call->adt_constructor;
        }
        auto call_expr = std::make_shared<MirCCallExpr>(
            std::move(func_name), std::move(arg_refs));
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
            mod_name = dotted_name(dot->expr.get());
            mod_ref = temp_assign(
                std::make_shared<MirGetModuleExpr>(mod_name));
        } else {
            mod_ref = temp_assign(eval(dot->expr.get()));
        }
        const auto& symbol = dot->compiled_symbol.empty()
            ? dot->rhs->id : dot->compiled_symbol;
        auto attr = std::make_shared<MirGetModuleAttrExpr>(
            mod_ref, std::move(mod_name), symbol);
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
    }
    std::unreachable();
    return nullptr;
}
} // namespace

MirModule MirBuilder::from_ast_module(const std::shared_ptr<Module> &ast) {
    MirModule mod;
    mod.lib_name = ast->lib_name;
    const auto add_module = [&](const auto& self, const std::string& name,
                                const std::shared_ptr<ModuleType>& type) -> void {
        if (!mod.imports.emplace(name, type).second) return;
        for (const auto& exported : type->exports) {
            if (exported.type->kind != TypeKind::Module) continue;
            self(self, name + "." + exported.name,
                 std::static_pointer_cast<ModuleType>(exported.type));
        }
    };
    for (const auto& imported : ast->imports | std::views::values)
        add_module(add_module, imported->binding_name, imported);
    Builder builder(mod);
    builder.build(ast.get());
    return mod;
}

} // namespace lmx::mir
