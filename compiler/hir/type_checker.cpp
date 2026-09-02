
#include "type_checker.hpp"

#include <algorithm>
#include <functional>
#include <ranges>
#include <limits>
#include <string_view>
#include <unordered_set>

#include "../error.hpp"

using namespace lmx;
using namespace lmx::hir;

namespace {

bool is_basic_type(const std::shared_ptr<Type>& type, runtime::ValueKind kind) noexcept {
    return type && type->kind == TypeKind::Basic &&
           std::reinterpret_pointer_cast<BasicType>(type)->type == kind;
}

bool is_numeric_or_expr_type(const std::shared_ptr<Type>& type) noexcept {
    if (type && type->kind == TypeKind::Dimensioned) return true;
    if (!type || type->kind != TypeKind::Basic) return false;
    const auto kind = std::reinterpret_pointer_cast<BasicType>(type)->type;
    return kind == runtime::ValueKind::Int ||
           kind == runtime::ValueKind::Fraction ||
           kind == runtime::ValueKind::Real ||
           kind == runtime::ValueKind::Complex ||
           kind == runtime::ValueKind::Expr;
}

bool is_expr_type(const std::shared_ptr<Type>& type) noexcept {
    return is_basic_type(type, runtime::ValueKind::Expr);
}

bool is_expr_constructible(const std::shared_ptr<Type>& type) noexcept {
    if (type && type->kind == TypeKind::Dimensioned) return true;
    if (type && type->kind == TypeKind::Named) {
        return std::static_pointer_cast<NamedType>(type)->name == "interval";
    }
    if (!type || type->kind != TypeKind::Basic) return false;
    const auto kind = std::reinterpret_pointer_cast<BasicType>(type)->type;
    return kind == runtime::ValueKind::Int ||
           kind == runtime::ValueKind::Fraction ||
           kind == runtime::ValueKind::Real ||
           kind == runtime::ValueKind::Complex ||
           kind == runtime::ValueKind::Expr;
}

bool supports_basic_equality(const runtime::ValueKind kind) noexcept {
    switch (kind) {
    case runtime::ValueKind::Int:
    case runtime::ValueKind::Fraction:
    case runtime::ValueKind::Real:
    case runtime::ValueKind::Complex:
    case runtime::ValueKind::Vector:
    case runtime::ValueKind::Matrix:
    case runtime::ValueKind::Table:
    case runtime::ValueKind::Quantity:
    case runtime::ValueKind::Sparse:
    case runtime::ValueKind::Tensor:
    case runtime::ValueKind::Assumptions:
        return true;
    default:
        return false;
    }
}

bool is_int_or_fraction(const runtime::ValueKind kind) noexcept {
    return kind == runtime::ValueKind::Int ||
           kind == runtime::ValueKind::Fraction;
}

std::optional<runtime::ValueKind> basic_binary_result(
    const runtime::ValueKind operand,
    const BinaryNode::Op op) noexcept {
    switch (op) {
    case BinaryNode::Op::Add:
    case BinaryNode::Op::Sub:
    case BinaryNode::Op::Mul:
    case BinaryNode::Op::Mod:
    case BinaryNode::Op::Pow:
        if (is_int_or_fraction(operand) ||
            operand == runtime::ValueKind::Real) return operand;
        return std::nullopt;
    case BinaryNode::Op::Div:
        if (is_int_or_fraction(operand) ||
            operand == runtime::ValueKind::Real)
            return operand == runtime::ValueKind::Real
                ? runtime::ValueKind::Real : runtime::ValueKind::Fraction;
        return std::nullopt;
    case BinaryNode::Op::Eq:
    case BinaryNode::Op::Ne:
        if (supports_basic_equality(operand)) return runtime::ValueKind::Bool;
        return std::nullopt;
    case BinaryNode::Op::Gt:
    case BinaryNode::Op::Ge:
    case BinaryNode::Op::Lt:
    case BinaryNode::Op::Le:
        if (is_int_or_fraction(operand) ||
            operand == runtime::ValueKind::Real)
            return runtime::ValueKind::Bool;
        return std::nullopt;
    case BinaryNode::Op::And:
    case BinaryNode::Op::Or:
        if (operand == runtime::ValueKind::Bool)
            return runtime::ValueKind::Bool;
        return std::nullopt;
    default:
        return std::nullopt;
    }
}

bool is_named_type(const std::shared_ptr<Type>& type, const std::string_view name) noexcept {
    return type && type->kind == TypeKind::Named &&
           std::static_pointer_cast<NamedType>(type)->name == name;
}

bool is_dimensioned_type(const std::shared_ptr<Type>& type) noexcept;

std::optional<std::pair<bool, bool>> interval_constructor_bounds(
    const ExprNode* expression) noexcept {
    if (!expression || expression->kind != ASTKind::DotExpr) return std::nullopt;
    const auto* dot = static_cast<const DotExprNode*>(expression);
    if (!dot->expr || dot->expr->kind != ASTKind::Identifier || !dot->rhs) {
        return std::nullopt;
    }
    const auto* module = static_cast<const IdentifierNode*>(dot->expr.get());
    if (module->id != "std") return std::nullopt;
    if (dot->rhs->id == "interval_closed") return std::pair{true, true};
    if (dot->rhs->id == "interval_open") return std::pair{false, false};
    if (dot->rhs->id == "interval_closed_open") return std::pair{true, false};
    if (dot->rhs->id == "interval_open_closed") return std::pair{false, true};
    return std::nullopt;
}

bool is_interval_ordered_type(const std::shared_ptr<Type>& type) noexcept {
    if (!type) return false;
    if (type->kind == TypeKind::Dimensioned) return true;
    if (type->kind != TypeKind::Basic) return false;
    switch (std::static_pointer_cast<BasicType>(type)->type) {
    case runtime::ValueKind::Int:
    case runtime::ValueKind::Fraction:
    case runtime::ValueKind::Real:
        return true;
    default:
        return false;
    }
}

int numeric_rank(const std::shared_ptr<Type>& type) noexcept {
    if (!type || type->kind != TypeKind::Basic) return -1;
    switch (std::static_pointer_cast<BasicType>(type)->type) {
    case runtime::ValueKind::Int: return 0;
    case runtime::ValueKind::Fraction: return 1;
    case runtime::ValueKind::Real: return 2;
    default: return -1;
    }
}

std::shared_ptr<Type> unify_interval_bounds(const std::shared_ptr<Type>& lhs,
                                           const std::shared_ptr<Type>& rhs) noexcept {
    if (!lhs || !rhs) return nullptr;
    if (lhs->equals(rhs.get())) return lhs;
    const auto left_rank = numeric_rank(lhs);
    const auto right_rank = numeric_rank(rhs);
    if (left_rank >= 0 && right_rank >= 0) {
        const auto rank = std::max(left_rank, right_rank);
        const auto kind = rank == 0 ? runtime::ValueKind::Int
            : rank == 1 ? runtime::ValueKind::Fraction
                        : runtime::ValueKind::Real;
        return type_pool.basic(kind);
    }
    if (is_dimensioned_type(lhs) && is_dimensioned_type(rhs)) {
        const auto left = std::static_pointer_cast<DimensionedType>(lhs);
        const auto right = std::static_pointer_cast<DimensionedType>(rhs);
        if (left->unit.dimension == right->unit.dimension) return lhs;
    }
    return nullptr;
}

bool interval_member_assignable(const std::shared_ptr<Type>& expected,
                                const std::shared_ptr<Type>& actual) noexcept;

void mark_expr_promotion(const std::shared_ptr<ExprNode>& expression) {
    if (!expression || is_expr_type(expression->type)) return;
    expression->promoted_from_type = expression->type;
    expression->type = type_pool.basic(runtime::ValueKind::Expr);
}

bool is_dimensioned_type(const std::shared_ptr<Type>& type) noexcept {
    return type && type->kind == TypeKind::Dimensioned &&
           std::static_pointer_cast<DimensionedType>(type)->resolved;
}

std::optional<RationalScale> constant_numeric_value(const ExprNode* expression) noexcept {
    if (!expression) return std::nullopt;
    switch (expression->kind) {
    case ASTKind::Literal: {
        const auto* literal = static_cast<const LiteralNode*>(expression);
        if (literal->kind != LiteralNode::Kind::Integer &&
            literal->kind != LiteralNode::Kind::Float) return std::nullopt;
        return RationalScale::from_decimal(literal->val);
    }
    case ASTKind::Unary: {
        const auto* unary = static_cast<const UnaryNode*>(expression);
        if (unary->op != UnaryNode::Op::Neg) return std::nullopt;
        auto value = constant_numeric_value(unary->expr.get());
        if (!value) return std::nullopt;
        value->numerator = -value->numerator;
        return value;
    }
    case ASTKind::Binary: {
        const auto* binary = static_cast<const BinaryNode*>(expression);
        auto lhs = constant_numeric_value(binary->lhs.get());
        auto rhs = constant_numeric_value(binary->rhs.get());
        if (!lhs || !rhs) return std::nullopt;
        switch (binary->op) {
        case BinaryNode::Op::Mul: return lhs->multiplied_by(*rhs);
        case BinaryNode::Op::Div: return lhs->divided_by(*rhs);
        case BinaryNode::Op::Add: return lhs->added_to(*rhs);
        case BinaryNode::Op::Sub: return lhs->subtracted_by(*rhs);
        case BinaryNode::Op::Pow:
            if (rhs->denominator != 1 || rhs->numerator < -32 || rhs->numerator > 32)
                return std::nullopt;
            return lhs->raised_to(static_cast<int>(rhs->numerator));
        default: return std::nullopt;
        }
    }
    case ASTKind::UnitAnnotated: {
        const auto* unit = static_cast<const UnitAnnotatedExprNode*>(expression);
        auto value = constant_numeric_value(unit->value.get());
        return value ? value->multiplied_by(unit->resolved_unit.scale_to_base)
                     : std::nullopt;
    }
    default: return std::nullopt;
    }
}

std::optional<std::int64_t> signed_integer_literal(
    const ExprNode* expression) noexcept {
    if (!expression) return std::nullopt;
    if (expression->kind == ASTKind::Literal) {
        const auto* literal = static_cast<const LiteralNode*>(expression);
        if (literal->kind != LiteralNode::Kind::Integer) return std::nullopt;
        try {
            std::size_t used = 0;
            const auto value = std::stoll(literal->val, &used);
            return used == literal->val.size()
                ? std::optional<std::int64_t>(value) : std::nullopt;
        } catch (...) {
            return std::nullopt;
        }
    }
    if (expression->kind != ASTKind::Unary) return std::nullopt;
    const auto* unary = static_cast<const UnaryNode*>(expression);
    if (unary->op != UnaryNode::Op::Neg) return std::nullopt;
    const auto value = signed_integer_literal(unary->expr.get());
    if (!value || *value == std::numeric_limits<std::int64_t>::min())
        return std::nullopt;
    return -*value;
}

std::optional<UnitDefinition> combined_unit(const UnitDefinition& lhs,
                                            const UnitDefinition& rhs,
                                            const bool divide) {
    UnitDefinition result;
    result.dimension = divide ? lhs.dimension.divided_by(rhs.dimension)
                              : lhs.dimension.multiplied_by(rhs.dimension);
    const auto scale = divide ? lhs.scale_to_base.divided_by(rhs.scale_to_base)
                              : lhs.scale_to_base.multiplied_by(rhs.scale_to_base);
    if (!scale) return std::nullopt;
    result.scale_to_base = *scale;
    result.display_unit = lhs.display_unit + (divide ? "/" : "*") + rhs.display_unit;
    return result;
}

bool runtime_scale_representable(const RationalScale& scale) noexcept {
    return scale.numerator >= std::numeric_limits<std::int32_t>::min() &&
           scale.numerator <= std::numeric_limits<std::int32_t>::max() &&
           scale.denominator > 0 &&
           scale.denominator <= std::numeric_limits<std::int32_t>::max();
}

std::shared_ptr<Type> unify_types(const std::shared_ptr<Type>& lhs,
                                  const std::shared_ptr<Type>& rhs) noexcept;

std::shared_ptr<Type> literal_payload_type(const LiteralPayloadNode& node) noexcept {
    const bool has_expr = std::ranges::any_of(node.elements, [](const auto& element) {
        return is_expr_type(element->type);
    });
    if (node.payload_kind == LiteralPayloadNode::Kind::Interval) {
        if (has_expr) {
            const bool promotable = std::ranges::all_of(
                node.elements, [](const auto& element) {
                    return is_expr_constructible(element->type);
                });
            return promotable ? type_pool.basic(runtime::ValueKind::Expr)
                              : type_pool.unknown();
        }
        if (node.elements.size() != 2) return type_pool.unknown();
        auto element = unify_interval_bounds(node.elements[0]->type,
                                             node.elements[1]->type);
        return element ? type_pool.named("interval", {std::move(element)})
                       : type_pool.unknown();
    }

    if (node.elements.empty()) return type_pool.named("set", {type_pool.unknown()});
    auto element = node.elements.front()->type;
    for (std::size_t i = 1; i < node.elements.size(); ++i) {
        const auto& candidate = node.elements[i]->type;
        if (is_expr_type(element) || is_expr_type(candidate)) {
            if (!is_expr_constructible(element) ||
                !is_expr_constructible(candidate))
                return type_pool.unknown();
            element = type_pool.basic(runtime::ValueKind::Expr);
            continue;
        }
        auto unified = unify_types(element, candidate);
        if (!unified)
            unified = unify_interval_bounds(element, candidate);
        if (!unified) return type_pool.unknown();
        element = std::move(unified);
    }
    return type_pool.named("set", {std::move(element)});
}

using TypeBindings = std::unordered_map<std::string, std::shared_ptr<Type>>;

bool bind_adt_type(const std::shared_ptr<Type>& expected,
                   const std::shared_ptr<Type>& actual,
                   const std::unordered_set<std::string>& params,
                   TypeBindings& bindings) noexcept {
    if (!expected || !actual) return false;
    if (actual->kind == TypeKind::Unknown) return true;
    if (actual->kind == TypeKind::Never) return true;
    if (expected->kind == TypeKind::Nullable) {
        const auto nullable = std::static_pointer_cast<NullableType>(expected);
        if (actual->kind == TypeKind::Basic &&
            std::static_pointer_cast<BasicType>(actual)->type == runtime::ValueKind::Null) return true;
        if (actual->kind == TypeKind::Nullable)
            return bind_adt_type(nullable->value_type,
                                 std::static_pointer_cast<NullableType>(actual)->value_type,
                                 params, bindings);
        return bind_adt_type(nullable->value_type, actual, params, bindings);
    }
    if (expected->kind == TypeKind::Named) {
        const auto named = std::static_pointer_cast<NamedType>(expected);
        if (params.contains(named->name) && named->args.empty()) {
            const auto it = bindings.find(named->name);
            if (it == bindings.end()) {
                bindings[named->name] = actual;
                return true;
            }
            return it->second->equals(actual.get());
        }
        if (actual->kind != TypeKind::Named) return false;
        const auto actual_named = std::static_pointer_cast<NamedType>(actual);
        if (named->name != actual_named->name || named->args.size() != actual_named->args.size()) return false;
        for (size_t i = 0; i < named->args.size(); ++i) {
            if (!bind_adt_type(named->args[i], actual_named->args[i], params, bindings)) return false;
        }
        return true;
    }
    return expected->equals(actual.get());
}

bool type_assignable(const std::shared_ptr<Type>& expected,
                     const std::shared_ptr<Type>& actual) noexcept {
    if (!expected || !actual) return false;
    if (actual->kind == TypeKind::Never) return true;
    if (expected->kind == TypeKind::Function && actual->kind == TypeKind::Function) {
        const auto expected_function = std::static_pointer_cast<FunctionType>(expected);
        const auto actual_function = std::static_pointer_cast<FunctionType>(actual);
        if (expected_function->params_ty.size() != actual_function->params_ty.size()) return false;
        for (size_t i = 0; i < expected_function->params_ty.size(); ++i) {
            if (!expected_function->params_ty[i]->equals(actual_function->params_ty[i].get())) return false;
        }
        return actual_function->ret_ty->kind == TypeKind::Never ||
               expected_function->ret_ty->equals(actual_function->ret_ty.get());
    }
    if (expected->kind == TypeKind::Nullable) {
        const auto nullable = std::static_pointer_cast<NullableType>(expected);
        if (actual->kind == TypeKind::Basic &&
            std::static_pointer_cast<BasicType>(actual)->type == runtime::ValueKind::Null) return true;
        if (actual->kind == TypeKind::Nullable)
            return type_assignable(nullable->value_type,
                                   std::static_pointer_cast<NullableType>(actual)->value_type);
        return type_assignable(nullable->value_type, actual);
    }
    if (expected->kind != TypeKind::Named || actual->kind != TypeKind::Named)
        return expected->equals(actual.get());
    const auto expected_named = std::static_pointer_cast<NamedType>(expected);
    const auto actual_named = std::static_pointer_cast<NamedType>(actual);
    if (expected_named->name != actual_named->name || expected_named->args.size() != actual_named->args.size()) return false;
    for (size_t i = 0; i < expected_named->args.size(); ++i) {
        if (actual_named->args[i]->kind == TypeKind::Unknown) continue;
        if (!type_assignable(expected_named->args[i], actual_named->args[i])) return false;
    }
    return true;
}

bool interval_member_assignable(const std::shared_ptr<Type>& expected,
                                const std::shared_ptr<Type>& actual) noexcept {
    if (numeric_rank(expected) >= 0 && numeric_rank(actual) >= 0) return true;
    if (is_dimensioned_type(expected) && is_dimensioned_type(actual)) {
        return std::static_pointer_cast<DimensionedType>(expected)->unit.dimension ==
               std::static_pointer_cast<DimensionedType>(actual)->unit.dimension;
    }
    return type_assignable(expected, actual);
}

std::shared_ptr<Type> unify_types(const std::shared_ptr<Type>& lhs,
                                  const std::shared_ptr<Type>& rhs) noexcept {
    if (!lhs || !rhs) return nullptr;
    if (lhs->kind == TypeKind::Never)
        return rhs->kind == TypeKind::Never ? lhs : rhs;
    if (rhs->kind == TypeKind::Never) return lhs;
    if (lhs->kind == TypeKind::Unknown) return rhs;
    if (rhs->kind == TypeKind::Unknown) return lhs;
    if (numeric_rank(lhs) >= 0 && numeric_rank(rhs) >= 0)
        return unify_interval_bounds(lhs, rhs);
    const auto is_null = [](const std::shared_ptr<Type>& type) {
        return type->kind == TypeKind::Basic &&
               std::static_pointer_cast<BasicType>(type)->type == runtime::ValueKind::Null;
    };
    if (lhs->kind == TypeKind::Nullable) {
        const auto nullable = std::static_pointer_cast<NullableType>(lhs);
        if (is_null(rhs)) return lhs;
        auto value = rhs->kind == TypeKind::Nullable
            ? std::static_pointer_cast<NullableType>(rhs)->value_type : rhs;
        auto unified = unify_types(nullable->value_type, value);
        return unified ? type_pool.nullable(std::move(unified)) : nullptr;
    }
    if (rhs->kind == TypeKind::Nullable || is_null(lhs)) return unify_types(rhs, lhs);
    if (lhs->kind != TypeKind::Named || rhs->kind != TypeKind::Named)
        return lhs->equals(rhs.get()) ? lhs : nullptr;
    const auto left = std::static_pointer_cast<NamedType>(lhs);
    const auto right = std::static_pointer_cast<NamedType>(rhs);
    if (left->name != right->name || left->args.size() != right->args.size()) return nullptr;
    std::vector<std::shared_ptr<Type>> args;
    args.reserve(left->args.size());
    for (size_t i = 0; i < left->args.size(); ++i) {
        auto unified = unify_types(left->args[i], right->args[i]);
        if (!unified) return nullptr;
        args.push_back(std::move(unified));
    }
    return type_pool.named(left->name, std::move(args));
}

bool contains_unknown_type(const std::shared_ptr<Type>& type) noexcept {
    if (!type) return false;
    if (type->kind == TypeKind::Unknown) return true;
    if (type->kind == TypeKind::Nullable)
        return contains_unknown_type(std::static_pointer_cast<NullableType>(type)->value_type);
    if (type->kind == TypeKind::Tuple) {
        const auto tuple = std::static_pointer_cast<TupleType>(type);
        return std::any_of(tuple->tys.begin(), tuple->tys.end(), contains_unknown_type);
    }
    if (type->kind != TypeKind::Named) return false;
    const auto named = std::static_pointer_cast<NamedType>(type);
    return std::any_of(named->args.begin(), named->args.end(), contains_unknown_type);
}

std::shared_ptr<Type> instantiate_adt_type(const std::shared_ptr<Type>& type,
                                           const TypeBindings& bindings) noexcept {
    if (!type) return type;
    if (type->kind == TypeKind::Nullable)
        return type_pool.nullable(instantiate_adt_type(
            std::static_pointer_cast<NullableType>(type)->value_type, bindings));
    if (type->kind != TypeKind::Named) return type;
    const auto named = std::static_pointer_cast<NamedType>(type);
    if (const auto it = bindings.find(named->name); it != bindings.end() && named->args.empty()) return it->second;
    std::vector<std::shared_ptr<Type>> args;
    args.reserve(named->args.size());
    for (const auto& arg : named->args) args.push_back(instantiate_adt_type(arg, bindings));
    return type_pool.named(named->name, std::move(args));
}

} // namespace

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

TypeDeclNode* TypeCkContext::find_module_adt(ModuleType* module, const std::string& name) noexcept {
    if (!module) return nullptr;
    for (const auto& declaration : module->adt_exports) {
        if (declaration->name == name || declaration->qualified_name == name) return declaration.get();
    }
    return nullptr;
}

std::pair<TypeDeclNode*, AdtConstructorDecl*> TypeCkContext::find_module_constructor(
    ModuleType* module, const std::string& name) noexcept {
    if (!module) return {nullptr, nullptr};
    for (const auto& declaration : module->adt_exports) {
        for (auto& constructor : declaration->constructors) {
            if (constructor.name == name) return {declaration.get(), &constructor};
        }
    }
    return {nullptr, nullptr};
}

std::shared_ptr<Type> TypeCkContext::resolve_type(const std::shared_ptr<Type>& type) noexcept {
    if (!type) return type;
    if (type->kind == TypeKind::Dimensioned) {
        const auto dimensioned = std::static_pointer_cast<DimensionedType>(type);
        if (dimensioned->resolved) return type;
        const auto resolved = unit_system.resolve(dimensioned->syntax);
        if (!resolved) {
            throw_error(ErrorType::Analysis,
                        "UnitInvalid: unknown or invalid unit expression `" +
                            dimensioned->syntax.to_string() + "`", 0, 0);
            return type_pool.unknown();
        }
        return resolved->dimension.is_dimensionless()
            ? type_pool.basic(runtime::ValueKind::Fraction)
            : type_pool.dimensioned(*resolved);
    }
    if (type->kind == TypeKind::Named) {
        const auto named = std::static_pointer_cast<NamedType>(type);
        std::vector<std::shared_ptr<Type>> args;
        args.reserve(named->args.size());
        for (const auto& arg : named->args) args.push_back(resolve_type(arg));
        if (const auto it = adt_types.find(named->name); it != adt_types.end()) {
            if (args.size() != it->second->type_params.size()) {
                throw_error(ErrorType::Analysis, "type `" + named->name + "` expects " +
                            std::to_string(it->second->type_params.size()) + " argument(s)", 0, 0);
            }
            return type_pool.named(it->second->qualified_name, std::move(args));
        }
        if (named->name == "set") {
            if (args.size() != 1) {
                throw_error(ErrorType::Analysis,
                            "type `set` expects 1 argument(s)", 0, 0);
                return type_pool.named("set", std::move(args));
            }
            const bool unresolved_type_parameter =
                args.front()->kind == TypeKind::Named &&
                std::static_pointer_cast<NamedType>(args.front())->args.empty() &&
                !adt_types.contains(
                    std::static_pointer_cast<NamedType>(args.front())->name);
            if (args.front()->kind != TypeKind::Unknown &&
                !unresolved_type_parameter &&
                !is_equality_comparable(args.front())) {
                throw_error(ErrorType::Analysis,
                            "SetElementNotHashable", 0, 0);
            }
            return type_pool.named("set", std::move(args));
        }
        if (const auto dot = named->name.find('.'); dot != std::string::npos) {
            const auto module_name = named->name.substr(0, dot);
            const auto type_name = named->name.substr(dot + 1);
            if (const auto module_var = find_global(module_name);
                module_var.has_value() && (*module_var)->type->kind == TypeKind::Module) {
                auto module = std::static_pointer_cast<ModuleType>((*module_var)->type);
                if (auto* declaration = find_module_adt(module.get(), type_name)) {
                    if (args.size() != declaration->type_params.size()) {
                        throw_error(ErrorType::Analysis, "type `" + named->name + "` expects " +
                                    std::to_string(declaration->type_params.size()) + " argument(s)", 0, 0);
                    }
                    return type_pool.named(declaration->qualified_name, std::move(args));
                }
            }
        }
        return type_pool.named(named->name, std::move(args));
    }
    if (type->kind == TypeKind::Nullable) {
        const auto nullable = std::static_pointer_cast<NullableType>(type);
        return type_pool.nullable(resolve_type(nullable->value_type));
    }
    if (type->kind == TypeKind::Array) {
        const auto array = std::static_pointer_cast<ArrayType>(type);
        return type_pool.array(resolve_type(array->type));
    }
    if (type->kind == TypeKind::Tuple) {
        const auto tuple = std::static_pointer_cast<TupleType>(type);
        std::vector<std::shared_ptr<Type>> elements;
        elements.reserve(tuple->tys.size());
        for (const auto& element : tuple->tys) elements.push_back(resolve_type(element));
        return type_pool.tuple(std::move(elements));
    }
    if (type->kind == TypeKind::Function) {
        const auto function = std::static_pointer_cast<FunctionType>(type);
        std::vector<std::shared_ptr<Type>> params;
        params.reserve(function->params_ty.size());
        for (const auto& param : function->params_ty) params.push_back(resolve_type(param));
        return type_pool.function(std::move(params), resolve_type(function->ret_ty));
    }
    if (type->kind == TypeKind::NativeFunction) {
        const auto function = std::static_pointer_cast<NativeFunctionType>(type);
        std::vector<std::shared_ptr<Type>> params;
        params.reserve(function->params_ty.size());
        for (const auto& param : function->params_ty) params.push_back(resolve_type(param));
        return type_pool.native_function(std::move(params), resolve_type(function->ret_ty), function->name);
    }
    return type;
}
bool TypeCkContext::is_equality_comparable(const std::shared_ptr<Type>& type) noexcept {
    std::unordered_set<std::string> visiting;
    std::function<bool(const std::shared_ptr<Type>&)> comparable;
    comparable = [&](const std::shared_ptr<Type>& current) {
        if (!current || current->kind == TypeKind::Unknown || current->kind == TypeKind::None ||
            current->kind == TypeKind::Function || current->kind == TypeKind::NativeFunction ||
            current->kind == TypeKind::Module || current->kind == TypeKind::AdtConstructor ||
            current->kind == TypeKind::Array) return false;
        if (current->kind == TypeKind::Dimensioned) return true;
        if (current->kind == TypeKind::Nullable)
            return comparable(std::static_pointer_cast<NullableType>(current)->value_type);
        if (current->kind == TypeKind::Tuple) {
            const auto tuple = std::static_pointer_cast<TupleType>(current);
            return std::all_of(tuple->tys.begin(), tuple->tys.end(), comparable);
        }
        if (current->kind == TypeKind::Basic) {
            return std::static_pointer_cast<BasicType>(current)->type != runtime::ValueKind::C_VaList;
        }
        if (current->kind == TypeKind::String) return true;
        if (current->kind != TypeKind::Named) return false;
        const auto named = std::static_pointer_cast<NamedType>(current);
        if (named->name == "set" || named->name == "interval") {
            return named->args.size() == 1 && comparable(named->args.front());
        }
        const auto declaration_it = adt_types.find(named->name);
        if (declaration_it == adt_types.end()) return false;
        auto* declaration = declaration_it->second;
        if (named->args.size() != declaration->type_params.size()) return false;
        const auto key = Type::to_string(current.get());
        if (!visiting.insert(key).second) return true;
        TypeBindings bindings;
        for (size_t i = 0; i < declaration->type_params.size(); ++i)
            bindings[declaration->type_params[i]] = named->args[i];
        for (const auto& constructor : declaration->constructors) {
            for (const auto& field : constructor.fields) {
                if (!comparable(instantiate_adt_type(field, bindings))) {
                    visiting.erase(key);
                    return false;
                }
            }
        }
        visiting.erase(key);
        return true;
    };
    return comparable(type);
}
TypeCkContext::TypeCkContext(ModuleResolver* module_resolver) noexcept
    : module_resolver(module_resolver) {
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
        case LiteralNode::Kind::Null: {
            return type_pool.basic(runtime::ValueKind::Null);
        }
        }
        break;
    }
    case ASTKind::Identifier: {
        const auto node = reinterpret_cast<IdentifierNode*>(type);
        if (node->id == "I") {
            return type_pool.basic(runtime::ValueKind::Expr);
        }
        if (node->type && node->type->kind != TypeKind::Unknown) return node->type;
        if (find_var(node->id).has_value())return (*find_var(node->id))->type;
        if (find_global(node->id).has_value())
            return (*find_global(node->id))->type;
        break;
    }
    case ASTKind::Unary: {
        const auto node = reinterpret_cast<UnaryNode*>(type);
        if (const auto t = inference_type(node->expr.get());
            t &&
            t->kind == TypeKind::Basic
            ) {
            if (const auto t2 = std::reinterpret_pointer_cast<BasicType>(t)->type;
                t2 == runtime::ValueKind::Int ||
                t2 == runtime::ValueKind::Fraction ||
                t2 == runtime::ValueKind::Expr) {

                return type_pool.basic(t2);
            }
        }
        break;
    }
    case ASTKind::Binary: {
        const auto node = reinterpret_cast<BinaryNode*>(type);
        auto left_ty = inference_type(node->lhs.get());
        const auto right_ty = inference_type(node->rhs.get());
        if (node->op == BinaryNode::Op::Bind) {
            return type_pool.named("Binding");
        }
        if (node->op == BinaryNode::Op::In ||
            node->op == BinaryNode::Op::NotIn ||
            node->op == BinaryNode::Op::Subset) {
            return is_expr_type(left_ty) || is_expr_type(right_ty)
                ? type_pool.basic(runtime::ValueKind::Expr)
                : type_pool.basic(runtime::ValueKind::Bool);
        }
        if (node->op == BinaryNode::Op::SetUnion ||
            node->op == BinaryNode::Op::SetIntersection ||
            node->op == BinaryNode::Op::SetSymmetricDifference ||
            (node->op == BinaryNode::Op::Sub &&
             (is_named_type(left_ty, "set") ||
              is_named_type(right_ty, "set"))))
            return left_ty;
        if (is_expr_type(left_ty) || is_expr_type(right_ty)) {
            return type_pool.basic(runtime::ValueKind::Expr);
        }
        if (left_ty->equals(right_ty.get())) return left_ty;
        break;
    }
    case ASTKind::LiteralPayload: {
        return literal_payload_type(*reinterpret_cast<LiteralPayloadNode*>(type));
    }
    case ASTKind::UnitAnnotated:
        return type->type ? type->type : type_pool.unknown();
    case ASTKind::MatchExpr: {
        return type->type ? type->type : type_pool.unknown();
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
        return node->type ? node->type
                          : (node->cast_type ? node->cast_type : type_pool.unknown());
    }
    case ASTKind::DotExpr: {
        const auto node = reinterpret_cast<DotExprNode*>(type);
        const auto left_type = inference_type(node->expr.get());
        if (!left_type || left_type->kind != TypeKind::Module)
            return type_pool.unknown();
        const auto left = std::static_pointer_cast<ModuleType>(left_type);
        const auto found = left->find_var(node->rhs->id);
        return found ? (*found)->type : type_pool.unknown();
    }
    case ASTKind::NativeFuncCall: {
        const auto node = reinterpret_cast<NativeFuncCallExpr*>(type);
        const auto left_ty = std::reinterpret_pointer_cast<NativeFunctionType>(inference_type(node->expr.get()));
        return left_ty->ret_ty;
        break;
    }
    case ASTKind::ArrayLiteral: {
        const auto node = reinterpret_cast<ArrayLiteralNode*>(type);
        if (node->exprs.empty()) return type_pool.array(type_pool.unknown());
        const auto elem_ty = node->exprs[0]->type->kind == TypeKind::Unknown
            ? inference_type(node->exprs[0].get())
            : node->exprs[0]->type;
        for (size_t i = 1; i < node->exprs.size(); i++) {
            const auto& ety = node->exprs[i]->type->kind == TypeKind::Unknown
                ? inference_type(node->exprs[i].get())
                : node->exprs[i]->type;
            if (!elem_ty->equals(ety.get())) return type_pool.unknown();
        }
        if (Type::is_null_type(elem_ty.get())) return type_pool.unknown();
        return type_pool.array(elem_ty);
    }
    case ASTKind::TupleLiteral: {
        const auto node = reinterpret_cast<TupleLiteralNode*>(type);
        std::vector<std::shared_ptr<Type>> elements;
        elements.reserve(node->exprs.size());
        for (const auto& expression : node->exprs) {
            auto element = expression->type && expression->type->kind != TypeKind::Unknown
                ? expression->type : inference_type(expression.get());
            if (!element || Type::is_null_type(element.get())) return type_pool.unknown();
            elements.push_back(std::move(element));
        }
        return type_pool.tuple(std::move(elements));
    }
    case ASTKind::TupleGetExpr: {
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

    stmt->body.insert(stmt->body.begin(), std::make_shared<ExprStmtNode>(0, 0, break_if));

    const auto one = std::make_shared<LiteralNode>(0, 0, "1", LiteralNode::Kind::Integer);
    const auto dec_cnt = std::make_shared<AssignStmtNode>(0, 0, lhs, std::make_shared<BinaryNode>(0, 0, lhs, BinaryNode::Op::Sub, one));
    stmt->body.insert(stmt->body.end(), dec_cnt);
    decltype(BlockExprNode::stmts) block{var_cnt, stmt};
    auto result = std::make_shared<ExprStmtNode>(
        stmt->line, stmt->col, std::make_shared<BlockExprNode>(stmt->line, stmt->col, std::move(block)));
    return result;
}




std::vector<Scope::Var> TypeCkContext::check_module(const std::shared_ptr<Module> &mod) noexcept {
    const auto save_cur_module = cur_module;
    cur_module = mod;
    mod->adt_exports.clear();
    mod->unit_exports.clear();

    if (!mod->native_funcs.empty() && mod->lib_name.empty()) {
        throw_error(ErrorType::Analysis, "module not `static` declare dynamic library, cannot declare native function", 0 , 0);

    }
    static const auto builtin_adts = [] {
        std::vector<std::unique_ptr<TypeDeclNode>> declarations;
        declarations.push_back(std::make_unique<TypeDeclNode>(0, 0, "Option",
            std::vector<std::string>{"T"},
            std::vector<AdtConstructorDecl>{
                {"Some", {type_pool.named("T")}},
                {"None", {}}
            }));
        declarations.push_back(std::make_unique<TypeDeclNode>(0, 0, "Result",
            std::vector<std::string>{"T", "E"},
            std::vector<AdtConstructorDecl>{
                {"Ok", {type_pool.named("T")}},
                {"Err", {type_pool.named("E")}}
            }));
        declarations.push_back(std::make_unique<TypeDeclNode>(0, 0, "Binding",
            std::vector<std::string>{"K", "V"},
            std::vector<AdtConstructorDecl>{{"Binding", {
                type_pool.named("K"), type_pool.named("V")
            }}}));
        return declarations;
    }();
    for (const auto& declaration_ptr : builtin_adts) {
        auto* declaration = declaration_ptr.get();
        adt_types[declaration->name] = declaration;
        for (auto& constructor : declaration->constructors) {
            adt_constructors[constructor.name] = {declaration, &constructor};
            new_global_var(constructor.name, type_pool.adt_constructor(
                declaration->qualified_name, constructor.name, declaration->type_params, constructor.fields));
        }
    }
    for (auto& node : mod->decls) {
        if (node->kind == ASTKind::ImportStmt) check_stmt(node);
    }
    for (auto& node : mod->decls) {
        if (node->kind == ASTKind::UnitDecl) check_stmt(node);
    }
    for (const auto& node : mod->decls) {
        if (node->kind != ASTKind::TypeDecl) continue;
        auto* declaration = reinterpret_cast<TypeDeclNode*>(node.get());
        if (adt_types.contains(declaration->name)) {
            throw_error(ErrorType::Analysis, "duplicate ADT `" + declaration->name + "`", declaration->line, declaration->col);
            continue;
        }
        declaration->qualified_name = mod->name + "::" + declaration->name;
        adt_types[declaration->name] = declaration;
        adt_types[declaration->qualified_name] = declaration;
        mod->adt_exports.push_back(std::static_pointer_cast<TypeDeclNode>(node));
        for (auto& constructor : declaration->constructors) {
            if (adt_constructors.contains(constructor.name)) {
                throw_error(ErrorType::Analysis, "duplicate constructor `" + constructor.name + "`", declaration->line, declaration->col);
                continue;
            }
            adt_constructors[constructor.name] = {declaration, &constructor};
            new_global_var(constructor.name, type_pool.adt_constructor(
                declaration->qualified_name, constructor.name, declaration->type_params, constructor.fields));
        }
    }
    for (const auto& node : mod->decls) {
        if (node->kind != ASTKind::TypeDecl) continue;
        auto* declaration = reinterpret_cast<TypeDeclNode*>(node.get());
        std::unordered_set<std::string> parameters;
        for (const auto& parameter : declaration->type_params) {
            if (!parameters.insert(parameter).second)
                throw_error(ErrorType::Analysis, "duplicate type parameter `" + parameter + "`", declaration->line, declaration->col);
        }
        std::function<void(const std::shared_ptr<Type>&)> validate_type;
        validate_type = [&](const std::shared_ptr<Type>& type) {
            if (!type) return;
            if (type->kind == TypeKind::Nullable) {
                validate_type(std::static_pointer_cast<NullableType>(type)->value_type);
                return;
            }
            if (type->kind != TypeKind::Named) return;
            const auto named = std::static_pointer_cast<NamedType>(type);
            if (parameters.contains(named->name)) {
                if (!named->args.empty())
                    throw_error(ErrorType::Analysis, "type parameter `" + named->name + "` cannot have arguments", declaration->line, declaration->col);
                return;
            }
            if (named->name == "set" || named->name == "interval") {
                if (named->args.size() != 1) {
                    throw_error(ErrorType::Analysis, "type `" + named->name + "` expects 1 argument(s)",
                                declaration->line, declaration->col);
                    return;
                }
                validate_type(named->args.front());
                return;
            }
            const auto referenced = adt_types.find(named->name);
            if (referenced == adt_types.end()) {
                throw_error(ErrorType::Analysis, "unknown field type `" + named->name + "`", declaration->line, declaration->col);
                return;
            }
            if (named->args.size() != referenced->second->type_params.size()) {
                throw_error(ErrorType::Analysis, "type `" + named->name + "` expects " +
                            std::to_string(referenced->second->type_params.size()) + " argument(s)",
                            declaration->line, declaration->col);
                return;
            }
            for (const auto& argument : named->args) validate_type(argument);
        };
        for (auto& constructor : declaration->constructors) {
            for (auto& field : constructor.fields) {
                field = resolve_type(field);
                validate_type(field);
            }
            if (const auto constructor_it = adt_constructors.find(constructor.name);
                constructor_it != adt_constructors.end() && constructor_it->second.first == declaration) {
                constructor_it->second = {declaration, &constructor};
                if (const auto global = find_global(constructor.name); global.has_value()) {
                    (*global)->type = type_pool.adt_constructor(
                        declaration->qualified_name, constructor.name,
                        declaration->type_params, constructor.fields);
                }
            }
        }
    }
    std::unordered_set<std::string> local_adt_names;
    std::unordered_set<std::string> exported_adt_identities;
    for (const auto& declaration : mod->adt_exports) {
        local_adt_names.insert(declaration->name);
        exported_adt_identities.insert(declaration->qualified_name);
    }
    for (const auto& imported_module : mod->imports | std::views::values) {
        for (const auto& declaration : imported_module->adt_exports) {
            if (local_adt_names.contains(declaration->name) ||
                !exported_adt_identities.insert(
                    declaration->qualified_name).second)
                continue;
            mod->adt_exports.push_back(declaration);
        }
    }
    mod->function_slots.clear();
    std::unordered_map<std::string, std::vector<std::pair<size_t, FuncImplNode*>>>
        regular_function_groups;
    for (size_t declaration_order = 0; declaration_order < mod->decls.size();
         ++declaration_order) {
        const auto& declaration = mod->decls[declaration_order];
        if (declaration->kind != ASTKind::FuncImpl) continue;
        auto* function = static_cast<FuncImplNode*>(declaration.get());
        if (function->func_id == "raise") {
            throw_error(ErrorType::Analysis,
                        "cannot redefine builtin `raise`",
                        function->line, function->col);
        }
        for (auto& [name, type] : function->params->stmts) type = resolve_type(type);
        function->return_type = resolve_type(function->return_type);
        regular_function_groups[function->func_id].emplace_back(
            declaration_order, function);
    }
    for (const auto& [name, declarations] : regular_function_groups) {
        for (size_t i = 0; i < declarations.size(); ++i) {
            const auto* lhs = declarations[i].second;
            for (size_t j = i + 1; j < declarations.size(); ++j) {
                const auto* rhs = declarations[j].second;
                if (lhs->params->stmts.size() != rhs->params->stmts.size()) continue;
                bool duplicate = true;
                for (size_t parameter = 0; parameter < lhs->params->stmts.size();
                     ++parameter) {
                    if (!lhs->params->stmts[parameter].second->equals(
                            rhs->params->stmts[parameter].second.get())) {
                        duplicate = false;
                        break;
                    }
                }
                if (duplicate) {
                    throw_error(ErrorType::Analysis,
                                "duplicate function parameter signature `" + name + "`",
                                rhs->line, rhs->col);
                }
            }
        }
    }
    for (size_t declaration_order = 0; declaration_order < mod->decls.size();
         ++declaration_order) {
        const auto& declaration = mod->decls[declaration_order];
        if (declaration->kind != ASTKind::FuncImpl) continue;
        auto* function = static_cast<FuncImplNode*>(declaration.get());
        const auto& group = regular_function_groups[function->func_id];
        function->compiled_symbol = group.size() == 1
            ? function->func_id
            : function->func_id + "\x1f" + std::to_string(declaration_order);
        mod->function_slots.push_back(function->compiled_symbol);
    }
    mod->builtin_functions.clear();
    const auto add_raise_builtin = [&](std::string symbol,
                                       std::shared_ptr<Type> parameter_type,
                                       const bool is_export) {
        SyntheticBuiltinSpec builtin{
            SyntheticBuiltinKind::Raise,
            "raise",
            std::move(symbol),
            "value",
            std::move(parameter_type),
            type_pool.never(),
            is_export,
        };
        auto type = type_pool.function(
            {builtin.parameter_type}, builtin.return_type);
        new_global_var(builtin.source_name, std::move(type), false,
                       builtin.compiled_symbol, builtin.is_export);
        mod->function_slots.push_back(builtin.compiled_symbol);
        mod->builtin_functions.push_back(std::move(builtin));
    };
    add_raise_builtin("@builtin.raise.text", type_pool.string(), false);
    auto normalized_module_path = mod->name;
    std::ranges::replace(normalized_module_path, '\\', '/');
    if (normalized_module_path.ends_with(
            "modules/std/mathematics_error/module.lm")) {
        add_raise_builtin(
            "@builtin.raise.mathematics_error",
            resolve_type(type_pool.named("MathError")),
            true);
    }
    for (const auto& n : mod->native_funcs) {
        for (auto& [name, type] : n->params->stmts) type = resolve_type(type);
        n->return_type = resolve_type(n->return_type);
        new_global_var(n->func_id, n->make_type());
    }
    for (auto& node : mod->decls) {
        if (node->kind == ASTKind::ImportStmt || node->kind == ASTKind::UnitDecl) continue;
        check_stmt(node);
    }

    cur_module = save_cur_module;

    std::vector<Scope::Var> result;

    // Module exports include imported modules so packages can expose
    // hierarchical APIs such as std.cas and std.math.
    for (const auto& v : get_global()) {
        if (v.is_export &&
            (v.type->kind == TypeKind::Function ||
             v.type->kind == TypeKind::NativeFunction ||
             v.type->kind == TypeKind::Module)) {
            result.push_back(v);
        }
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
    case ASTKind::UnitAnnotated: {
        auto* node = static_cast<UnitAnnotatedExprNode*>(expr.get());
        check_expr(node->value);
        if (!is_basic_type(node->value->type, runtime::ValueKind::Int) &&
            !is_basic_type(node->value->type, runtime::ValueKind::Fraction)) {
            throw_error(ErrorType::Analysis,
                        "UnitTypeMismatch: units can only annotate numeric literals",
                        node->line, node->col);
            break;
        }
        const auto resolved = unit_system.resolve(node->unit_syntax);
        if (!resolved) {
            throw_error(ErrorType::Analysis,
                        "UnitInvalid: unknown or invalid unit expression `" +
                            node->unit_syntax.to_string() + "`",
                        node->line, node->col);
            break;
        }
        node->resolved_unit = *resolved;
        node->type = resolved->dimension.is_dimensionless()
            ? type_pool.basic(runtime::ValueKind::Fraction)
            : type_pool.dimensioned(*resolved);
        break;
    }
    case ASTKind::Identifier: {
        auto* node = reinterpret_cast<IdentifierNode*>(expr.get());
        if (node->id == "I") {
            node->type = type_pool.basic(runtime::ValueKind::Expr);
            break;
        }
        if (const auto re = find_var(node->id); re.has_value()) {
            node->type = (*re)->type;
            break;
        }
        Scope::Var* resolved = nullptr;
        size_t regular_function_count = 0;
        for (auto& global : global_scope) {
            if (global.name != node->id) continue;
            if (!resolved) resolved = &global;
            if (global.type->kind == TypeKind::Function)
                ++regular_function_count;
        }
        if (regular_function_count > 1) {
            throw_error(ErrorType::Analysis, "ambiguous overloaded function",
                        node->line, node->col);
            break;
        }
        if (resolved) {
            if (resolved->type->kind == TypeKind::AdtConstructor) {
                const auto constructor =
                    std::static_pointer_cast<AdtConstructorType>(resolved->type);
                if (!constructor->fields.empty()) {
                    node->type = resolved->type;
                    break;
                }
                node->is_zero_adt_constructor = true;
                node->adt_type_name = constructor->type_name;
                std::vector<std::shared_ptr<Type>> args(
                    constructor->type_params.size(), type_pool.unknown());
                node->type = type_pool.named(constructor->type_name,
                                             std::move(args));
                break;
            }
            node->type = resolved->type;
            node->compiled_symbol = resolved->symbol;
            break;
        }
        throw_error(ErrorType::Analysis, "undefined var `" + node->id + "`", node->line, node->col);
        break;
    }
    case ASTKind::Unary: {
        auto* node = reinterpret_cast<UnaryNode*>(expr.get());
        check_expr(node->expr);
        const auto type = node->expr->type;
        if (type->kind == TypeKind::Dimensioned) {
            if (node->op != UnaryNode::Op::Neg) {
                throw_error(ErrorType::Analysis, "unary `not` requires bool",
                            expr->line, expr->col);
                break;
            }
            node->type = type;
            break;
        }
        if (type->kind != TypeKind::Basic) {
            throw_error(ErrorType::Analysis, "unary cannot applied to this type", expr->line, expr->col);
            break;
        }
        const auto t2 = std::reinterpret_pointer_cast<BasicType>(type);
        if (node->op == UnaryNode::Op::Neg) {
            if (
            t2->type != runtime::ValueKind::Int &&
            t2->type != runtime::ValueKind::Fraction &&
            t2->type != runtime::ValueKind::Real &&
            t2->type != runtime::ValueKind::Expr) {
                throw_error(ErrorType::Analysis, "unary`-` cannot applied to this type", expr->line, expr->col);
                break;
            }
        } else if (node->op == UnaryNode::Op::Not) {
            if (t2->type != runtime::ValueKind::Bool &&
                t2->type != runtime::ValueKind::Expr) {
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
        if (node->op == BinaryNode::Op::Bind) {
            if (is_expr_type(lty) || is_expr_type(rty)) {
                if (!is_expr_constructible(lty) || !is_expr_constructible(rty))
                    goto binary_type_mismatch;
                mark_expr_promotion(node->lhs);
                mark_expr_promotion(node->rhs);
                node->type = type_pool.named("Binding", {
                    type_pool.basic(runtime::ValueKind::Expr),
                    type_pool.basic(runtime::ValueKind::Expr)});
                break;
            }
            node->type = type_pool.named("Binding", {lty, rty});
            break;
        }
        if ((is_dimensioned_type(lty) || is_dimensioned_type(rty)) &&
            !is_expr_type(lty) && !is_expr_type(rty) &&
            node->op != BinaryNode::Op::In &&
            node->op != BinaryNode::Op::NotIn) {
            const auto left_dimensioned = is_dimensioned_type(lty);
            const auto right_dimensioned = is_dimensioned_type(rty);
            const auto plain_numeric = [](const std::shared_ptr<Type>& type) {
                return is_basic_type(type, runtime::ValueKind::Int) ||
                       is_basic_type(type, runtime::ValueKind::Fraction);
            };
            if (left_dimensioned && right_dimensioned) {
                const auto left = std::static_pointer_cast<DimensionedType>(lty);
                const auto right = std::static_pointer_cast<DimensionedType>(rty);
                switch (node->op) {
                case BinaryNode::Op::Add:
                case BinaryNode::Op::Sub:
                case BinaryNode::Op::Mod:
                    if (!left->equals(right.get())) {
                        throw_error(ErrorType::Analysis,
                                    "DimensionMismatch: addition, subtraction, and remainder require identical units",
                                    node->line, node->col);
                        break;
                    }
                    node->type = lty;
                    break;
                case BinaryNode::Op::Eq:
                case BinaryNode::Op::Ne:
                case BinaryNode::Op::Gt:
                case BinaryNode::Op::Ge:
                case BinaryNode::Op::Lt:
                case BinaryNode::Op::Le:
                    if (!left->equals(right.get())) {
                        throw_error(ErrorType::Analysis,
                                    "DimensionMismatch: comparison requires identical units",
                                    node->line, node->col);
                        break;
                    }
                    node->type = type_pool.basic(runtime::ValueKind::Bool);
                    break;
                case BinaryNode::Op::Mul:
                case BinaryNode::Op::Div: {
                    auto unit = combined_unit(left->unit, right->unit,
                                              node->op == BinaryNode::Op::Div);
                    if (!unit) {
                        throw_error(ErrorType::Analysis, "UnitScaleOverflow",
                                    node->line, node->col);
                        break;
                    }
                    node->type = unit->dimension.is_dimensionless()
                        ? type_pool.basic(runtime::ValueKind::Fraction)
                        : type_pool.dimensioned(std::move(*unit));
                    break;
                }
                default:
                    throw_error(ErrorType::Analysis,
                                "binary operation cannot be applied to dimensioned values",
                                node->line, node->col);
                    break;
                }
                break;
            }
            if (left_dimensioned && node->op == BinaryNode::Op::Pow) {
                const auto exponent_value = signed_integer_literal(node->rhs.get());
                if (!is_basic_type(rty, runtime::ValueKind::Int) || !exponent_value) {
                    throw_error(ErrorType::Analysis,
                                "DimensionExponentMustBeConstantInteger",
                                node->line, node->col);
                    break;
                }
                if (*exponent_value < -32 || *exponent_value > 32) {
                    throw_error(ErrorType::Analysis,
                                "DimensionExponentOutOfRange", node->line, node->col);
                    break;
                }
                const auto exponent = static_cast<int>(*exponent_value);
                const auto left = std::static_pointer_cast<DimensionedType>(lty);
                UnitDefinition unit;
                unit.dimension = left->unit.dimension.raised_to(exponent);
                const auto scale = left->unit.scale_to_base.raised_to(exponent);
                if (!scale) {
                    throw_error(ErrorType::Analysis, "UnitScaleOverflow",
                                node->line, node->col);
                    break;
                }
                unit.scale_to_base = *scale;
                unit.display_unit = left->unit.display_unit + "^" + std::to_string(exponent);
                node->type = unit.dimension.is_dimensionless()
                    ? type_pool.basic(runtime::ValueKind::Fraction)
                    : type_pool.dimensioned(std::move(unit));
                break;
            }
            if ((node->op == BinaryNode::Op::Mul || node->op == BinaryNode::Op::Div) &&
                ((left_dimensioned && plain_numeric(rty)) ||
                 (right_dimensioned && plain_numeric(lty)))) {
                if (left_dimensioned) {
                    node->type = lty;
                } else if (node->op == BinaryNode::Op::Mul) {
                    node->type = rty;
                } else {
                    const auto right = std::static_pointer_cast<DimensionedType>(rty);
                    UnitDefinition unit;
                    unit.dimension = right->unit.dimension.raised_to(-1);
                    const auto scale = right->unit.scale_to_base.raised_to(-1);
                    if (!scale) {
                        throw_error(ErrorType::Analysis, "UnitScaleOverflow",
                                    node->line, node->col);
                        break;
                    }
                    unit.scale_to_base = *scale;
                    unit.display_unit = "1/" + right->unit.display_unit;
                    node->type = type_pool.dimensioned(std::move(unit));
                }
                break;
            }
            throw_error(ErrorType::Analysis,
                        "DimensionMismatch: incompatible dimensioned operands",
                        node->line, node->col);
            break;
        }
        if (node->op == BinaryNode::Op::Eq || node->op == BinaryNode::Op::Ne) {
            if (auto unified = unify_types(lty, rty)) {
                if (!is_equality_comparable(unified)) {
                    throw_error(ErrorType::Analysis,
                                "values are not equality comparable",
                                node->line, node->col);
                    break;
                }
                if (contains_unknown_type(lty)) node->lhs->type = unified;
                if (contains_unknown_type(rty)) node->rhs->type = unified;
                node->type = type_pool.basic(runtime::ValueKind::Bool);
                break;
            }
        }
        if (node->op == BinaryNode::Op::In || node->op == BinaryNode::Op::NotIn) {
            if (is_named_type(rty, "set")) {
                const auto container = std::static_pointer_cast<NamedType>(rty);
                if (container->args.size() != 1)
                    goto binary_type_mismatch;
                const auto& element = container->args.front();
                if (is_expr_type(element) && is_expr_constructible(lty))
                    mark_expr_promotion(node->lhs);
                const bool assignable =
                    type_assignable(element, node->lhs->type) ||
                    (numeric_rank(element) >= 0 &&
                     numeric_rank(node->lhs->type) >= 0);
                if (!assignable) goto binary_type_mismatch;
                node->type = type_pool.basic(runtime::ValueKind::Bool);
                break;
            }
            if (is_expr_type(lty) || is_expr_type(rty)) {
                if (is_expr_constructible(lty))
                    mark_expr_promotion(node->lhs);
                if (node->rhs->kind == ASTKind::LiteralPayload ||
                    is_expr_constructible(rty))
                    mark_expr_promotion(node->rhs);
                node->type = type_pool.basic(runtime::ValueKind::Expr);
                break;
            }
            if (!is_named_type(rty, "interval"))
                goto binary_type_mismatch;
            const auto container = std::static_pointer_cast<NamedType>(rty);
            if (container->args.size() != 1 ||
                !interval_member_assignable(container->args.front(), lty))
                goto binary_type_mismatch;
            node->type = type_pool.basic(runtime::ValueKind::Bool);
            break;
        }
        {
        const bool explicit_set_operation =
            node->op == BinaryNode::Op::SetUnion ||
            node->op == BinaryNode::Op::SetIntersection ||
            node->op == BinaryNode::Op::SetSymmetricDifference ||
            node->op == BinaryNode::Op::Subset;
        const bool set_difference =
            node->op == BinaryNode::Op::Sub &&
            (is_named_type(lty, "set") || is_named_type(rty, "set"));
        if (explicit_set_operation || set_difference) {
            if (!is_named_type(lty, "set") ||
                !is_named_type(rty, "set")) {
                throw_error(ErrorType::Analysis, "SetOperandTypeMismatch",
                            node->line, node->col);
                break;
            }
            const auto left_set = std::static_pointer_cast<NamedType>(lty);
            const auto right_set = std::static_pointer_cast<NamedType>(rty);
            if (left_set->args.size() != 1 || right_set->args.size() != 1) {
                throw_error(ErrorType::Analysis, "SetOperandTypeMismatch",
                            node->line, node->col);
                break;
            }
            auto element = unify_types(left_set->args.front(),
                                       right_set->args.front());
            if (!element)
                element = unify_interval_bounds(left_set->args.front(),
                                                right_set->args.front());
            if (!element) {
                throw_error(ErrorType::Analysis, "SetElementTypeMismatch",
                            node->line, node->col);
                break;
            }
            if (element->kind != TypeKind::Unknown &&
                !is_equality_comparable(element)) {
                throw_error(ErrorType::Analysis, "SetElementNotHashable",
                            node->line, node->col);
                break;
            }
            const auto set_type = type_pool.named("set", {element});
            if (contains_unknown_type(lty)) node->lhs->type = set_type;
            if (contains_unknown_type(rty)) node->rhs->type = set_type;
            node->type = node->op == BinaryNode::Op::Subset
                ? type_pool.basic(runtime::ValueKind::Bool)
                : set_type;
            break;
        }
        }
        if (is_expr_type(lty) || is_expr_type(rty)) {
            switch (node->op) {
            case BinaryNode::Op::Add:
            case BinaryNode::Op::Sub:
            case BinaryNode::Op::Mul:
            case BinaryNode::Op::Div:
            case BinaryNode::Op::Pow:
            case BinaryNode::Op::Eq:
            case BinaryNode::Op::Ne:
            case BinaryNode::Op::Gt:
            case BinaryNode::Op::Ge:
            case BinaryNode::Op::Lt:
            case BinaryNode::Op::Le:
                if (is_numeric_or_expr_type(lty) && is_numeric_or_expr_type(rty)) {
                    node->type = type_pool.basic(runtime::ValueKind::Expr);
                    break;
                }
                goto binary_type_mismatch;
            case BinaryNode::Op::And:
            case BinaryNode::Op::Or:
                if (is_expr_type(lty) && is_expr_type(rty)) {
                    node->type = type_pool.basic(runtime::ValueKind::Expr);
                    break;
                }
                goto binary_type_mismatch;
            default:
                goto binary_type_mismatch;
            }
            if (is_expr_type(node->type)) {
                node->type = type_pool.basic(runtime::ValueKind::Expr);
                break;
            }
        }
        {
        auto operand_type = lty;
        if (!lty->equals(rty.get())) {
            if (numeric_rank(lty) >= 0 && numeric_rank(rty) >= 0) {
                operand_type = unify_interval_bounds(lty, rty);
            } else {
                throw_error(
                    ErrorType::Analysis,
                    "binary operation type mismatch, (" +
                    Type::to_string(lty.get()) + " " +
                    BinaryNode::op_to_string(node->op) + " " +
                    Type::to_string(rty.get()) + ")",
                    expr->line, expr->col);
                break;
            }
        }
        if (!operand_type || operand_type->kind != TypeKind::Basic)
            goto binary_type_mismatch;
        {
            const auto operand =
                std::reinterpret_pointer_cast<BasicType>(operand_type)->type;
            const auto result = basic_binary_result(operand, node->op);
            if (!result) goto binary_type_mismatch;
            node->type = type_pool.basic(*result);
        }
        }
        break;
        binary_type_mismatch:
        throw_error(ErrorType::Analysis, "binary operation cannot applied to this type", expr->line, expr->col);
        break;
    }
    case ASTKind::LiteralPayload: {
        const auto node = reinterpret_cast<LiteralPayloadNode*>(expr.get());
        for (auto& element : node->elements) {
            check_expr(element);
        }
        node->type = literal_payload_type(*node);
        if (node->type->kind == TypeKind::Unknown) {
            const char* diagnostic = node->payload_kind == LiteralPayloadNode::Kind::Set
                ? "SetElementTypeMismatch"
                : "IntervalBoundTypeMismatch: interval bounds cannot be unified";
            throw_error(ErrorType::Analysis, diagnostic, node->line, node->col);
            break;
        }
        if (node->payload_kind == LiteralPayloadNode::Kind::Set) {
            const auto set = std::static_pointer_cast<NamedType>(node->type);
            if (set->args.size() == 1 &&
                is_expr_type(set->args.front())) {
                for (auto& element : node->elements) {
                    if (is_expr_constructible(element->type))
                        mark_expr_promotion(element);
                }
            }
            if (set->args.size() == 1 &&
                set->args.front()->kind != TypeKind::Unknown &&
                !is_equality_comparable(set->args.front())) {
                throw_error(ErrorType::Analysis, "SetElementNotHashable",
                            node->line, node->col);
                break;
            }
        }
        if (node->payload_kind == LiteralPayloadNode::Kind::Interval &&
            !is_expr_type(node->type)) {
            const auto interval = std::static_pointer_cast<NamedType>(node->type);
            if (interval->args.size() != 1 ||
                !is_interval_ordered_type(interval->args.front())) {
                throw_error(ErrorType::Analysis,
                            "IntervalBoundNotOrdered: interval bounds must be ordered values",
                            node->line, node->col);
                break;
            }
            const auto lower = constant_numeric_value(node->elements[0].get());
            const auto upper = constant_numeric_value(node->elements[1].get());
            if (lower && upper &&
                static_cast<long double>(lower->numerator) / lower->denominator >
                static_cast<long double>(upper->numerator) / upper->denominator) {
                throw_error(ErrorType::Analysis,
                            "IntervalBoundsReversed: lower bound exceeds upper bound",
                            node->line, node->col);
            }
        }
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
        if (const auto bounds = interval_constructor_bounds(node->expr.get())) {
            const auto standard_module = find_global("std");
            if (standard_module.has_value() &&
                (*standard_module)->type->kind == TypeKind::Module) {
                if (!node->suffix || node->suffix->exprs.size() != 2) {
                    throw_error(ErrorType::Analysis,
                                "IntervalArityMismatch: interval constructors require two bounds",
                                node->line, node->col);
                    break;
                }
                expr = std::make_shared<LiteralPayloadNode>(
                    node->line, node->col, LiteralPayloadNode::Kind::Interval,
                    std::move(node->suffix->exprs), bounds->first, bounds->second);
                check_expr(expr);
                break;
            }
        }
        if (node->expr->kind == ASTKind::Identifier) {
            const auto id = reinterpret_cast<IdentifierNode*>(node->expr.get());
            if (const auto found = find_global(id->id);
                found.has_value() && (*found)->type->kind == TypeKind::AdtConstructor) {
                const auto constructor = std::static_pointer_cast<AdtConstructorType>((*found)->type);
                if (constructor->fields.size() != node->suffix->exprs.size()) {
                    throw_error(ErrorType::Analysis, "constructor `" + constructor->constructor + "` expects " +
                                std::to_string(constructor->fields.size()) + " field(s)", node->line, node->col);
                    break;
                }
                const std::unordered_set<std::string> params(constructor->type_params.begin(), constructor->type_params.end());
                TypeBindings bindings;
                for (size_t i = 0; i < node->suffix->exprs.size(); ++i) {
                    check_expr(node->suffix->exprs[i]);
                    if (!bind_adt_type(constructor->fields[i], node->suffix->exprs[i]->type, params, bindings)) {
                        throw_error(ErrorType::Analysis, "constructor field type mismatch", node->line, node->col);
                    }
                }
                std::vector<std::shared_ptr<Type>> args;
                for (const auto& param : constructor->type_params) {
                    const auto it = bindings.find(param);
                    args.push_back(it == bindings.end() ? type_pool.unknown() : it->second);
                }
                node->is_adt_constructor = true;
                node->adt_type_name = constructor->type_name;
                node->adt_constructor = constructor->constructor;
                node->type = type_pool.named(constructor->type_name, std::move(args));
                break;
            }
        }
        if (node->expr->kind == ASTKind::Identifier) {
            const auto id = reinterpret_cast<IdentifierNode*>(node->expr.get());
            if (!find_var(id->id).has_value() && !find_global(id->id).has_value()) {
                bool has_expr_arg = false;
                for (auto& arg : node->suffix->exprs) {
                    check_expr(arg);
                    has_expr_arg = has_expr_arg || is_expr_type(arg->type);
                }
                if (has_expr_arg || node->allow_symbolic_call) {
                    node->is_symbolic_call = true;
                    node->expr->type = type_pool.basic(runtime::ValueKind::Expr);
                    node->type = type_pool.basic(runtime::ValueKind::Expr);
                    break;
                }
            }
        }
        node->can_fast = false;
        bool selected_callable = false;
        const std::function<std::shared_ptr<Type>(ExprNode*)>
            candidate_type = [&](ExprNode* argument) -> std::shared_ptr<Type> {
            if (!argument) return type_pool.unknown();
            if (argument->type && argument->type->kind != TypeKind::Unknown)
                return argument->type;
            if (argument->kind == ASTKind::Identifier ||
                argument->kind == ASTKind::Literal ||
                argument->kind == ASTKind::Unary ||
                argument->kind == ASTKind::Binary)
                return inference_type(argument);
            if (argument->kind != ASTKind::ArrayLiteral)
                return type_pool.unknown();
            const auto* array = static_cast<ArrayLiteralNode*>(argument);
            if (array->exprs.empty()) return type_pool.array(type_pool.unknown());
            const auto element = candidate_type(array->exprs.front().get());
            for (std::size_t index = 1; index < array->exprs.size(); ++index) {
                if (!element->equals(candidate_type(array->exprs[index].get()).get()))
                    return type_pool.unknown();
            }
            return type_pool.array(element);
        };
        const auto parameters_match = [&](const auto& params) {
            if (params.size() != node->suffix->exprs.size()) return false;
            for (std::size_t index = 0; index < params.size(); ++index) {
                const auto argument = candidate_type(node->suffix->exprs[index].get());
                if (argument->kind != TypeKind::Unknown &&
                    !type_assignable(params[index], argument) &&
                    !(is_expr_type(params[index]) && is_expr_constructible(argument)))
                    return false;
            }
            return true;
        };
        if (node->expr->kind == ASTKind::Identifier) {
            auto* id = static_cast<IdentifierNode*>(node->expr.get());
            if (!find_var(id->id).has_value()) {
                std::vector<Scope::Var*> regular_candidates;
                for (auto& global : global_scope) {
                    if (global.name == id->id &&
                        global.type->kind == TypeKind::Function)
                        regular_candidates.push_back(&global);
                }
                if (!regular_candidates.empty()) {
                    const auto selected = std::find_if(
                        regular_candidates.begin(), regular_candidates.end(),
                        [&](const auto* candidate) {
                            return parameters_match(
                                std::static_pointer_cast<FunctionType>(
                                    candidate->type)->params_ty);
                        });
                    auto* chosen = selected == regular_candidates.end()
                        ? regular_candidates.front() : *selected;
                    id->type = chosen->type;
                    id->compiled_symbol = chosen->symbol;
                    node->can_fast = true;
                    selected_callable = true;
                }
            }
        } else if (node->expr->kind == ASTKind::DotExpr) {
            auto* dot = static_cast<DotExprNode*>(node->expr.get());
            const auto owner_type = inference_type(dot->expr.get());
            if (owner_type && owner_type->kind == TypeKind::Module) {
                dot->expr->type = owner_type;
                const auto module = std::static_pointer_cast<ModuleType>(owner_type);
                std::vector<Scope::Var*> regular_candidates;
                std::vector<std::shared_ptr<NativeFunctionType>> native_candidates;
                for (auto& exported : module->exports) {
                    if (exported.name != dot->rhs->id) continue;
                    if (exported.type->kind == TypeKind::Function)
                        regular_candidates.push_back(&exported);
                    else if (exported.type->kind == TypeKind::NativeFunction)
                        native_candidates.push_back(
                            std::static_pointer_cast<NativeFunctionType>(exported.type));
                }
                if (!regular_candidates.empty()) {
                    const auto selected = std::find_if(
                        regular_candidates.begin(), regular_candidates.end(),
                        [&](const auto* candidate) {
                            return parameters_match(
                                std::static_pointer_cast<FunctionType>(
                                    candidate->type)->params_ty);
                        });
                    auto* chosen = selected == regular_candidates.end()
                        ? regular_candidates.front() : *selected;
                    dot->rhs->type = chosen->type;
                    dot->type = chosen->type;
                    dot->compiled_symbol = chosen->symbol;
                    selected_callable = true;
                } else if (!native_candidates.empty()) {
                    const auto selected = std::find_if(
                        native_candidates.begin(), native_candidates.end(),
                        [&](const auto& candidate) {
                            return parameters_match(candidate->params_ty);
                        });
                    const auto& chosen = selected == native_candidates.end()
                        ? native_candidates.front() : *selected;
                    dot->rhs->type = chosen;
                    dot->type = chosen;
                    if (native_candidates.size() > 1)
                        node->adt_constructor = chosen->name;
                    selected_callable = true;
                }
            }
        }
        if (!selected_callable) check_expr(node->expr);
        const auto left = node->expr->type;
        if (Type::is_null_type(left.get())) break;
        if (left->kind == TypeKind::AdtConstructor) {
            const auto constructor = std::static_pointer_cast<AdtConstructorType>(left);
            if (constructor->fields.size() != node->suffix->exprs.size()) {
                throw_error(ErrorType::Analysis, "constructor field count mismatch", node->line, node->col);
                break;
            }
            const std::unordered_set<std::string> params(constructor->type_params.begin(), constructor->type_params.end());
            TypeBindings bindings;
            for (size_t i = 0; i < node->suffix->exprs.size(); ++i) {
                check_expr(node->suffix->exprs[i]);
                if (!bind_adt_type(constructor->fields[i], node->suffix->exprs[i]->type, params, bindings))
                    throw_error(ErrorType::Analysis, "constructor field type mismatch", node->line, node->col);
            }
            std::vector<std::shared_ptr<Type>> args;
            for (const auto& param : constructor->type_params) {
                const auto it = bindings.find(param);
                args.push_back(it == bindings.end() ? type_pool.unknown() : it->second);
            }
            node->is_adt_constructor = true;
            node->adt_type_name = constructor->type_name;
            node->adt_constructor = constructor->constructor;
            node->type = type_pool.named(constructor->type_name, std::move(args));
        } else if (left->kind == TypeKind::Function) {
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
            bool symbolic_fallback = false;
            for (std::size_t i = 0; i < len; i++) {
                const auto param = func_ty->params_ty[i];
                check_expr(node->suffix->exprs[i]);
                if (is_expr_type(param) &&
                    is_expr_constructible(node->suffix->exprs[i]->type)) {
                    mark_expr_promotion(node->suffix->exprs[i]);
                }
                if (contains_unknown_type(node->suffix->exprs[i]->type) &&
                    type_assignable(param, node->suffix->exprs[i]->type)) {
                    node->suffix->exprs[i]->type = param;
                }
                if (!type_assignable(param, node->suffix->exprs[i]->type)) {
                    if (node->expr->kind == ASTKind::Identifier &&
                        is_expr_type(node->suffix->exprs[i]->type)) {
                        node->is_symbolic_call = true;
                        node->type = type_pool.basic(runtime::ValueKind::Expr);
                        symbolic_fallback = true;
                        break;
                    }
                    throw_error(ErrorType::Analysis,
                        "type mismatch arg in function calling in arg(s) " + std::to_string(i) +
                        ": (" + Type::to_string(node->suffix->exprs[i]->type.get()) +
                        " != " + Type::to_string(param.get()) + ")"
                        , node->line, node->col);
                    break;
                }
            }
            if (symbolic_fallback) break;
            node->type = std::reinterpret_pointer_cast<FunctionType>(left)->ret_ty;
        } else if (left->kind == TypeKind::NativeFunction) {
            const auto native_symbol = node->adt_constructor;
            new (expr.get()) NativeFuncCallExpr(node);
            const auto node = reinterpret_cast<NativeFuncCallExpr*>(expr.get());
            node->adt_constructor = native_symbol;
            const auto func_ty = std::reinterpret_pointer_cast<NativeFunctionType>(left);
            bool has_va_list = false;
            size_t fixed_arg_cnt = func_ty->params_ty.size();
            for (const auto& p : func_ty->params_ty) {
                if (p->kind == TypeKind::Basic &&
                    reinterpret_cast<BasicType*>(p.get())->type == runtime::ValueKind::C_VaList) {
                    if (p.get() != func_ty->params_ty.back().get()) {
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
                if (is_expr_type(param) &&
                    is_expr_constructible(node->suffix->exprs[i]->type)) {
                    mark_expr_promotion(node->suffix->exprs[i]);
                }
                if (!type_assignable(param, node->suffix->exprs[i]->type)) {
                    throw_error(
                        ErrorType::Analysis,
                        "type mismatch arg " + std::to_string(i) +
                            " in function calling: " +
                            Type::to_string(node->suffix->exprs[i]->type.get()) +
                            " is not assignable to " +
                            Type::to_string(param.get()),
                        node->line, node->col);
                    break;
                }
            }
            for (; i < len; i++) {
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
        check_expr(node->suffix);
        const auto left = node->expr->type;
        if (left->kind != TypeKind::Array) {
            throw_error(ErrorType::Analysis, "must be array type but got `" + Type::to_string(left.get()) + "`", node->line, node->col);
            break;
        }
        if (node->suffix->type->kind != TypeKind::Basic ||
            std::reinterpret_pointer_cast<BasicType>(node->suffix->type)->type != runtime::ValueKind::Int) {
            throw_error(ErrorType::Analysis, "array index must be int", node->line, node->col);
            break;
        }
        node->type = std::reinterpret_pointer_cast<ArrayType>(left)->type;
        break;
    }
    case ASTKind::IfExpr: {
        const auto node = reinterpret_cast<IfExprNode*>(expr.get());
        check_expr(node->cond);
        if (node->cond->type->kind != TypeKind::Basic ||
            std::reinterpret_pointer_cast<BasicType>(node->cond->type)->type != runtime::ValueKind::Bool) {
            throw_error(ErrorType::Analysis, "must be bool type but got `" + Type::to_string(node->cond->type.get()), node->line, node->col);
            break;
        }
        check_expr(node->then);
        if (node->els) {
            check_expr(node->els);
            if (node->then->have_ret_value() && node->els->have_ret_value()) {
                auto unified = unify_types(node->then->type, node->els->type);
                if (!unified) {
                    throw_error(ErrorType::Analysis, "if express then and else cannot type mismatch", node->line, node->col);
                    break;
                }
                node->type = std::move(unified);
            } else {
                node->type = node->then->type;
            }
        } else {
            node->type = node->then->type;
        }
        break;
    }
    case ASTKind::AsExpr: {
        auto* node = reinterpret_cast<AsExprNode*>(expr.get());
        if (node->cast_kind == AsExprNode::Kind::Unit) {
            check_expr(node->expr);
            const bool symbolic = is_expr_type(node->expr->type);
            if (!is_dimensioned_type(node->expr->type) && !symbolic) {
                throw_error(ErrorType::Analysis,
                            "UnitTypeMismatch: unit conversion requires a dimensioned value",
                            node->line, node->col);
                break;
            }
            const auto target = unit_system.resolve(node->unit_syntax);
            if (!target) {
                throw_error(ErrorType::Analysis,
                            "UnitInvalid: unknown or invalid target unit `" +
                                node->unit_syntax.to_string() + "`",
                            node->line, node->col);
                break;
            }
            const auto source = symbolic ? nullptr
                : std::static_pointer_cast<DimensionedType>(node->expr->type);
            if (source && source->unit.dimension != target->dimension) {
                throw_error(ErrorType::Analysis,
                            "DimensionMismatch: unit conversion requires equal dimensions",
                            node->line, node->col);
                break;
            }
            node->resolved_unit = *target;
            if (!symbolic) {
                const auto factor = source->unit.scale_to_base.divided_by(
                    target->scale_to_base);
                if (!factor || !runtime_scale_representable(*factor)) {
                    throw_error(ErrorType::Analysis, "UnitConversionOverflow",
                                node->line, node->col);
                    break;
                }
            }
            node->type = symbolic ? type_pool.basic(runtime::ValueKind::Expr)
                                  : type_pool.dimensioned(*target);
            break;
        }
        if (node->cast_kind == AsExprNode::Kind::Num ||
            node->cast_kind == AsExprNode::Kind::Scalar) {
            check_expr(node->expr);
            if (is_expr_type(node->expr->type)) {
                node->type = type_pool.basic(runtime::ValueKind::Expr);
                break;
            }
            if (is_dimensioned_type(node->expr->type)) {
                if (node->cast_kind == AsExprNode::Kind::Num) {
                    const auto dimensioned =
                        std::static_pointer_cast<DimensionedType>(node->expr->type);
                    if (!runtime_scale_representable(
                            dimensioned->unit.scale_to_base)) {
                        throw_error(ErrorType::Analysis, "UnitStripOverflow",
                                    node->line, node->col);
                        break;
                    }
                }
                node->type = type_pool.basic(runtime::ValueKind::Fraction);
                break;
            }
            if (is_basic_type(node->expr->type, runtime::ValueKind::Int) ||
                is_basic_type(node->expr->type, runtime::ValueKind::Fraction) ||
                is_basic_type(node->expr->type, runtime::ValueKind::Real)) {
                node->type = node->expr->type;
                break;
            }
            throw_error(ErrorType::Analysis, "UnitStripTypeMismatch",
                        node->line, node->col);
            break;
        }
        node->cast_type = resolve_type(node->cast_type);
        if (is_expr_type(node->cast_type) && node->expr->kind == ASTKind::SuffixParen)
            reinterpret_cast<SuffixParenNode*>(node->expr.get())->allow_symbolic_call = true;
        check_expr(node->expr);
        if (is_expr_type(node->cast_type) && is_expr_constructible(node->expr->type)) {
            mark_expr_promotion(node->expr);
        } else if (!node->cast_type->equals(node->expr->type.get())) {
            throw_error(ErrorType::Analysis, "cast type mismatch", node->line, node->col);
            break;
        }
        node->type = node->cast_type;
        break;
    }
    case ASTKind::DotExpr: {
        const auto node = reinterpret_cast<DotExprNode*>(expr.get());
        if (node->expr->kind == ASTKind::Identifier) {
            const auto* lhs = reinterpret_cast<IdentifierNode*>(node->expr.get());
            const auto type_it = adt_types.find(lhs->id);
            const auto constructor_it = adt_constructors.find(node->rhs->id);
            if (type_it != adt_types.end() && constructor_it != adt_constructors.end() &&
                constructor_it->second.first == type_it->second) {
                auto* declaration = type_it->second;
                auto* constructor = constructor_it->second.second;
                node->expr->type = type_pool.named(declaration->qualified_name);
                node->rhs->type = type_pool.adt_constructor(declaration->qualified_name, constructor->name,
                                                            declaration->type_params, constructor->fields);
                if (constructor->fields.empty()) {
                    node->is_zero_adt_constructor = true;
                    node->adt_type_name = declaration->qualified_name;
                    std::vector<std::shared_ptr<Type>> args(declaration->type_params.size(), type_pool.unknown());
                    node->type = type_pool.named(declaration->qualified_name, std::move(args));
                } else {
                    node->type = node->rhs->type;
                }
                break;
            }
        }
        check_expr(node->expr);
        if (!Type::is_null_type(node->expr->type.get()) && node->expr->type->kind != TypeKind::Module) {
            throw_error(ErrorType::Analysis, "must be module type", node->line, node->col);
            break;
        }
        const auto left_ty = std::reinterpret_pointer_cast<ModuleType>(node->expr->type);
        if (Type::is_null_type(left_ty.get())) break;
        if (const auto [declaration, constructor] = find_module_constructor(left_ty.get(), node->rhs->id);
            declaration && constructor) {
            node->rhs->type = type_pool.adt_constructor(
                declaration->qualified_name, constructor->name,
                declaration->type_params, constructor->fields);
            if (constructor->fields.empty()) {
                node->is_zero_adt_constructor = true;
                node->adt_type_name = declaration->qualified_name;
                std::vector<std::shared_ptr<Type>> args(
                    declaration->type_params.size(), type_pool.unknown());
                node->type = type_pool.named(declaration->qualified_name, std::move(args));
            } else {
                node->type = node->rhs->type;
            }
        } else {
            Scope::Var* resolved = nullptr;
            size_t regular_function_count = 0;
            for (auto& exported : left_ty->exports) {
                if (exported.name != node->rhs->id) continue;
                if (!resolved) resolved = &exported;
                if (exported.type->kind == TypeKind::Function)
                    ++regular_function_count;
            }
            if (regular_function_count > 1) {
                throw_error(ErrorType::Analysis, "ambiguous overloaded function",
                            node->line, node->col);
                break;
            }
            if (resolved) {
                node->rhs->type = resolved->type;
                node->type = resolved->type;
                node->compiled_symbol = resolved->symbol;
                break;
            }
            throw_error(ErrorType::Analysis, "module not have var `" +
                        node->rhs->id + "`", node->line, node->col);
            break;
        }
        break;
    }
    case ASTKind::MatchExpr: {
        auto* node = reinterpret_cast<MatchExprNode*>(expr.get());
        check_expr(node->target);
        bool catch_all = false;
        std::vector<Pattern> unguarded_patterns;
        std::shared_ptr<Type> result_type;
        auto target_named = node->target->type && node->target->type->kind == TypeKind::Named
            ? std::static_pointer_cast<NamedType>(node->target->type) : nullptr;

        for (auto& arm : node->arms) {
            if (catch_all) {
                throw_error(ErrorType::Analysis, "UnreachablePattern", arm.pattern.line, arm.pattern.col);
                continue;
            }
            scope_stack.emplace_back(Scope::ScopeType::Block);
            std::function<void(Pattern&, const std::shared_ptr<Type>&)> check_pattern;
            check_pattern = [&](Pattern& pattern, const std::shared_ptr<Type>& expected) {
                if (pattern.kind == Pattern::Kind::Wildcard) return;
                if (pattern.kind == Pattern::Kind::Binding) {
                    if (const auto constructor_it = adt_constructors.find(pattern.name);
                        constructor_it != adt_constructors.end() && constructor_it->second.second->fields.empty()) {
                        pattern.kind = Pattern::Kind::Constructor;
                        pattern.adt_type_name = constructor_it->second.first->qualified_name;
                        return;
                    }
                    new_cur_scope_var(pattern.name, expected);
                    return;
                }
                if (pattern.kind == Pattern::Kind::Literal) {
                    std::shared_ptr<ExprNode> literal = pattern.literal;
                    check_expr(literal);
                    if (!type_assignable(expected, literal->type)) {
                        throw_error(ErrorType::Analysis, "PatternTypeMismatch", pattern.line, pattern.col);
                    }
                    return;
                }
                TypeDeclNode* declaration = nullptr;
                AdtConstructorDecl* constructor = nullptr;
                if (!pattern.adt_type_name.empty()) {
                    if (const auto module_var = find_global(pattern.adt_type_name);
                        module_var.has_value() && (*module_var)->type->kind == TypeKind::Module) {
                        auto module = std::static_pointer_cast<ModuleType>((*module_var)->type);
                        std::tie(declaration, constructor) = find_module_constructor(module.get(), pattern.name);
                        if (declaration && constructor) pattern.adt_type_name = declaration->qualified_name;
                    }
                }
                if (!declaration || !constructor) {
                    const auto it = adt_constructors.find(pattern.name);
                    if (it != adt_constructors.end()) {
                        declaration = it->second.first;
                        constructor = it->second.second;
                    }
                }
                if (!declaration || !constructor) {
                    throw_error(ErrorType::Analysis, "unknown constructor `" + pattern.name + "`", pattern.line, pattern.col);
                    return;
                }
                if (!pattern.adt_type_name.empty() && pattern.adt_type_name != declaration->name &&
                    pattern.adt_type_name != declaration->qualified_name) {
                    throw_error(ErrorType::Analysis, "PatternTypeMismatch", pattern.line, pattern.col);
                    return;
                }
                pattern.adt_type_name = declaration->qualified_name;
                if (!expected || expected->kind != TypeKind::Named ||
                    std::static_pointer_cast<NamedType>(expected)->name != declaration->qualified_name) {
                    throw_error(ErrorType::Analysis, "PatternTypeMismatch", pattern.line, pattern.col);
                    return;
                }
                if (constructor->fields.size() != pattern.fields.size()) {
                    throw_error(ErrorType::Analysis, "constructor pattern field count mismatch", pattern.line, pattern.col);
                    return;
                }
                TypeBindings bindings;
                const auto expected_named = std::static_pointer_cast<NamedType>(expected);
                for (size_t i = 0; i < declaration->type_params.size() && i < expected_named->args.size(); ++i) {
                    bindings[declaration->type_params[i]] = expected_named->args[i];
                }
                for (size_t i = 0; i < pattern.fields.size(); ++i) {
                    check_pattern(pattern.fields[i], instantiate_adt_type(constructor->fields[i], bindings));
                }

            };
            check_pattern(arm.pattern, node->target->type);

            std::function<bool(const Pattern&, const Pattern&)> subsumes;
            subsumes = [&](const Pattern& previous, const Pattern& current) {
                if (previous.kind == Pattern::Kind::Wildcard || previous.kind == Pattern::Kind::Binding) return true;
                if (previous.kind != current.kind) return false;
                if (previous.kind == Pattern::Kind::Literal) {
                    return previous.literal && current.literal &&
                           previous.literal->kind == current.literal->kind &&
                           previous.literal->val == current.literal->val;
                }
                if (previous.kind != Pattern::Kind::Constructor ||
                    previous.adt_type_name != current.adt_type_name ||
                    previous.name != current.name ||
                    previous.fields.size() != current.fields.size()) return false;
                for (size_t i = 0; i < previous.fields.size(); ++i) {
                    if (!subsumes(previous.fields[i], current.fields[i])) return false;
                }
                return true;
            };
            if (std::any_of(unguarded_patterns.begin(), unguarded_patterns.end(),
                            [&](const Pattern& previous) { return subsumes(previous, arm.pattern); })) {
                throw_error(ErrorType::Analysis, "UnreachablePattern", arm.pattern.line, arm.pattern.col);
            }
            if (!arm.guard) unguarded_patterns.push_back(arm.pattern);
            if ((arm.pattern.kind == Pattern::Kind::Wildcard || arm.pattern.kind == Pattern::Kind::Binding) && !arm.guard) catch_all = true;
            if (arm.guard) {
                check_expr(arm.guard);
                if (!is_basic_type(arm.guard->type, runtime::ValueKind::Bool))
                    throw_error(ErrorType::Analysis, "match guard must be bool", arm.guard->line, arm.guard->col);
            }
            check_expr(arm.value);
            if (!result_type) {
                result_type = arm.value->type;
            } else if (auto unified = unify_types(result_type, arm.value->type)) {
                result_type = std::move(unified);
            } else {
                throw_error(ErrorType::Analysis, "MatchBranchTypeMismatch", arm.value->line, arm.value->col);
            }
            scope_stack.pop_back();
        }
        if (!catch_all) {
            using PatternRow = std::vector<Pattern>;
            using PatternMatrix = std::vector<PatternRow>;
            std::function<bool(const PatternMatrix&, const std::vector<std::shared_ptr<Type>>&)> exhaustive;
            exhaustive = [&](const PatternMatrix& matrix, const std::vector<std::shared_ptr<Type>>& types) -> bool {
                if (types.empty()) return !matrix.empty();
                const auto& head_type = types.front();
                auto tail_types = std::vector<std::shared_ptr<Type>>(types.begin() + 1, types.end());
                const auto make_wildcards = [](size_t count) {
                    PatternRow row;
                    row.reserve(count);
                    for (size_t i = 0; i < count; ++i)
                        row.emplace_back(Pattern::Kind::Wildcard, 0, 0);
                    return row;
                };
                const auto specialize_default = [&] {
                    PatternMatrix specialized;
                    for (const auto& row : matrix) {
                        if (row.empty()) continue;
                        if (row.front().kind != Pattern::Kind::Wildcard && row.front().kind != Pattern::Kind::Binding) continue;
                        specialized.emplace_back(row.begin() + 1, row.end());
                    }
                    return specialized;
                };

                auto default_matrix = specialize_default();
                if (!default_matrix.empty() && exhaustive(default_matrix, tail_types)) return true;

                if (head_type && head_type->kind == TypeKind::Named) {
                    const auto named = std::static_pointer_cast<NamedType>(head_type);
                    if (const auto declaration_it = adt_types.find(named->name); declaration_it != adt_types.end()) {
                        auto* declaration = declaration_it->second;
                        TypeBindings bindings;
                        for (size_t i = 0; i < declaration->type_params.size() && i < named->args.size(); ++i)
                            bindings[declaration->type_params[i]] = named->args[i];
                        for (const auto& constructor : declaration->constructors) {
                            PatternMatrix specialized;
                            for (const auto& row : matrix) {
                                if (row.empty()) continue;
                                PatternRow next;
                                const auto& head = row.front();
                                if (head.kind == Pattern::Kind::Wildcard || head.kind == Pattern::Kind::Binding) {
                                    next = make_wildcards(constructor.fields.size());
                                } else if (head.kind == Pattern::Kind::Constructor && head.name == constructor.name) {
                                    next = head.fields;
                                } else {
                                    continue;
                                }
                                next.insert(next.end(), row.begin() + 1, row.end());
                                specialized.push_back(std::move(next));
                            }
                            std::vector<std::shared_ptr<Type>> specialized_types;
                            specialized_types.reserve(constructor.fields.size() + tail_types.size());
                            for (const auto& field : constructor.fields)
                                specialized_types.push_back(instantiate_adt_type(field, bindings));
                            specialized_types.insert(specialized_types.end(), tail_types.begin(), tail_types.end());
                            if (!exhaustive(specialized, specialized_types)) return false;
                        }
                        return true;
                    }
                }

                if (is_basic_type(head_type, runtime::ValueKind::Bool)) {
                    for (const auto value : {"true", "false"}) {
                        PatternMatrix specialized;
                        for (const auto& row : matrix) {
                            if (row.empty()) continue;
                            const auto& head = row.front();
                            if (head.kind == Pattern::Kind::Wildcard || head.kind == Pattern::Kind::Binding ||
                                (head.kind == Pattern::Kind::Literal && head.literal && head.literal->val == value)) {
                                specialized.emplace_back(row.begin() + 1, row.end());
                            }
                        }
                        if (!exhaustive(specialized, tail_types)) return false;
                    }
                    return true;
                }

                return false;
            };

            PatternMatrix matrix;
            for (const auto& arm : node->arms) {
                if (!arm.guard) matrix.push_back({arm.pattern});
            }
            if (!exhaustive(matrix, {node->target->type}))
                throw_error(ErrorType::Analysis, "MissingWildcard", node->line, node->col);
        }
        node->type = result_type ? result_type : type_pool.none();
        break;
    }
    case ASTKind::ArrayLiteral: {
        auto* node = reinterpret_cast<ArrayLiteralNode*>(expr.get());
        std::shared_ptr<Type> element_type;
        for (auto& element : node->exprs) {
            check_expr(element);
            const auto& candidate = element->type;
            if (Type::is_null_type(candidate.get())) continue;
            if (!element_type) {
                element_type = candidate;
            } else if (!element_type->equals(candidate.get())) {
                throw_error(ErrorType::Analysis,
                    "array literal elements must be the same type, (" +
                    Type::to_string(element_type.get()) + " != " + Type::to_string(candidate.get()) + ")",
                    node->line, node->col);
                break;
            }
        }
        node->type = type_pool.array(element_type ? element_type : type_pool.unknown());
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
    case ASTKind::TupleLiteral: {
        const auto node = reinterpret_cast<TupleLiteralNode*>(expr.get());
        decltype(TupleType::tys) tys;
        for (auto& e : node->exprs) {
            check_expr(e);
            tys.push_back(e->type);
        }
        node->type = type_pool.tuple(std::move(tys));
        break;
    }
    case ASTKind::TupleGetExpr: {
        auto* node = reinterpret_cast<TupleGetExprNode*>(expr.get());
        check_expr(node->tup);
        if (node->tup->type->kind != TypeKind::Tuple) {
            throw_error(ErrorType::Analysis,
                        "TupleTypeMismatch: position access requires a tuple",
                        node->line, node->col);
            break;
        }
        const auto* tup_ty = reinterpret_cast<TupleType*>(node->tup->type.get());
        if (node->i >= tup_ty->tys.size()) {
            throw_error(ErrorType::Analysis,
                "TupleIndexOutOfBounds: tuple has " +
                    std::to_string(tup_ty->tys.size()) +
                    " elements but position " + std::to_string(node->i + 1) +
                    " was requested",
                node->line, node->col
                );
            break;
        }
        node->type = tup_ty->tys[node->i];
        break;
    }
    default: std::unreachable();
    }
}

void TypeCkContext::check_stmt(std::shared_ptr<StmtNode>& stmt) noexcept {
    switch (stmt->kind) {
    case ASTKind::TypeDecl:
        break;
    case ASTKind::ExprStmt: {
        auto* node = reinterpret_cast<ExprStmtNode*>(stmt.get());
        check_expr(node->expr);
        if (node->expr && contains_unknown_type(node->expr->type))
            throw_error(ErrorType::Analysis, "cannot infer ADT type arguments", node->line, node->col);
        break;
    }
    case ASTKind::ImportStmt: {
        const auto* node = reinterpret_cast<ImportStmtNode*>(stmt.get());
        if (!module_resolver) {
            throw_error(ErrorType::Analysis, "module resolver is unavailable for `" + node->name + "`", node->line, node->col);
            break;
        }
        const auto resolved = module_resolver->resolve_module({
            node->name,
            cur_module->name,
            node->line,
            node->col,
        });
        if (!resolved || errd) break;
        if (cur_module->module_is_imported(resolved->source_path)) break;

        for (const auto& declaration : resolved->type->adt_exports)
            adt_types[declaration->qualified_name] = declaration.get();
        for (const auto& [name, definition] : resolved->type->unit_exports)
            unit_system.import_unit(resolved->binding_name + "." + name, definition);
        new_global_var(resolved->binding_name, resolved->type);
        cur_module->imports[resolved->source_path] = resolved->type;
        break;
    }
    case ASTKind::UnitDecl: {
        auto* node = static_cast<UnitDeclNode*>(stmt.get());
        if (!is_global_scope()) {
            throw_error(ErrorType::Analysis, "unit declarations must be module scoped",
                        node->line, node->col);
            break;
        }
        if (!node->definition) {
            if (!unit_system.declare_base(
                    node->name, cur_module->name + "::" + node->name)) {
                throw_error(ErrorType::Analysis, "UnitRedefined: `" + node->name + "`",
                            node->line, node->col);
                break;
            }
            node->resolved_unit = *unit_system.resolve(node->name);
        } else {
            check_expr(node->definition);
            if (!is_dimensioned_type(node->definition->type)) {
                throw_error(ErrorType::Analysis,
                            "UnitInvalid: derived unit requires a dimensioned constant",
                            node->line, node->col);
                break;
            }
            const auto scale = constant_numeric_value(node->definition.get());
            if (!scale || scale->numerator <= 0) {
                throw_error(ErrorType::Analysis,
                            "UnitInvalid: derived unit scale must be a positive compile-time constant",
                            node->line, node->col);
                break;
            }
            const auto dimensioned =
                std::static_pointer_cast<DimensionedType>(node->definition->type);
            UnitDefinition definition{
                dimensioned->unit.dimension, *scale, node->name};
            if (!unit_system.declare_derived(node->name, definition)) {
                throw_error(ErrorType::Analysis, "UnitRedefined: `" + node->name + "`",
                            node->line, node->col);
                break;
            }
            node->resolved_unit = *unit_system.resolve(node->name);
        }
        cur_module->unit_exports.emplace_back(node->name, node->resolved_unit);
        break;
    }
    case ASTKind::SymDecl: {
        const auto* node = reinterpret_cast<SymDeclNode*>(stmt.get());
        for (const auto& id : node->ids) {
            if (id == "I") {
                throw_error(ErrorType::Analysis, "ImaginaryUnitReserved", node->line, node->col);
                break;
            }
            if (is_global_scope()) {
                new_global_var(id, type_pool.basic(runtime::ValueKind::Expr));
            } else {
                new_cur_scope_var(id, type_pool.basic(runtime::ValueKind::Expr));
            }
        }
        break;
    }
    case ASTKind::FuncImpl: {
        auto* node = reinterpret_cast<FuncImplNode*>(stmt.get());
        if (!is_global_scope()) throw_error(ErrorType::Analysis, "function only define in GlobalScope", stmt->line, stmt->col);

        for (auto& [name, type] : node->params->stmts) type = resolve_type(type);
        node->return_type = resolve_type(node->return_type);
        new_global_var(node->func_id, node->make_type(), false,
                       node->compiled_symbol);
        auto& ref = global_scope.back();
        Scope scope;
        scope.name = node->func_id;
        scope.return_type = node->return_type;
        for (const auto& [name, type] : node->params->stmts) {
            scope.vars.emplace_back(name, type, true);
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

        scope_stack.pop_back();
        ref.type = node->make_type();
        break;
    }
    case ASTKind::Return: {
        const auto node = reinterpret_cast<ReturnNode*>(stmt.get());
        if (!node->expr) break;
        check_expr(node->expr);
        for (auto& s : scope_stack | std::views::reverse) {
            if (s.scope == Scope::ScopeType::Function) {
                if (Type::is_null_type(s.return_type.get())) {
                    s.return_type = node->expr->type;
                    break;
                }
                if (contains_unknown_type(node->expr->type) && type_assignable(s.return_type, node->expr->type))
                    node->expr->type = s.return_type;
                if (is_expr_type(s.return_type) && is_expr_constructible(node->expr->type))
                    mark_expr_promotion(node->expr);
                if (!type_assignable(s.return_type, node->expr->type)) {
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
        if (is_expr_type(scope_stack.back().return_type) &&
            node->expr && node->expr->kind == ASTKind::SuffixParen)
            reinterpret_cast<SuffixParenNode*>(node->expr.get())->allow_symbolic_call = true;
        check_expr(node->expr);
        if (Type::is_null_type(scope_stack.back().return_type.get()))
            scope_stack.back().return_type = node->expr->type;
        else {
            if (contains_unknown_type(node->expr->type) &&
                type_assignable(scope_stack.back().return_type, node->expr->type))
                node->expr->type = scope_stack.back().return_type;
            if (is_expr_type(scope_stack.back().return_type) &&
                is_expr_constructible(node->expr->type))
                mark_expr_promotion(node->expr);
            if (!type_assignable(scope_stack.back().return_type, node->expr->type)) {
                throw_error(ErrorType::Analysis, "return type is inconsistent with the above", node->line, node->col);
                break;
            }
        }
        break;
    }
    case ASTKind::VarDecl: {
        const auto node = reinterpret_cast<VarDeclNode*>(stmt.get());
        if (node->id == "I") {
            throw_error(ErrorType::Analysis, "ImaginaryUnitReserved", node->line, node->col);
            break;
        }
        if (!Type::is_null_type(node->type.get())) node->type = resolve_type(node->type);
        if (is_expr_type(node->type) && node->init_value &&
            node->init_value->kind == ASTKind::SuffixParen)
            reinterpret_cast<SuffixParenNode*>(node->init_value.get())->allow_symbolic_call = true;
        check_expr(node->init_value);
        if (Type::is_null_type(node->type.get())) {
            if (!node->init_value) {
                throw_error(ErrorType::Analysis, "the var `" + node->id + "` type not found", node->line, node->col);
                break;
            } else {
                node->type = node->init_value->type;
                if (contains_unknown_type(node->type)) {
                    throw_error(ErrorType::Analysis, "cannot infer ADT type arguments for `" + node->id + "`", node->line, node->col);
                    break;
                }
            }
        } else {
            if (is_expr_type(node->type) && is_expr_constructible(node->init_value->type)) {
                mark_expr_promotion(node->init_value);
            } else if (contains_unknown_type(node->init_value->type) &&
                       type_assignable(node->type, node->init_value->type)) {
                node->init_value->type = node->type;
            } else if (!type_assignable(node->type, node->init_value->type)) {
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
        if (node->lhs->kind == ASTKind::SuffixBracket) {
        } else if (node->lhs->kind == ASTKind::TupleGetExpr) {
            throw_error(ErrorType::Analysis,
                        "TupleAssignment: tuple element bindings are immutable",
                        node->line, node->col);
            break;
        } else if (node->lhs->kind == ASTKind::Identifier) {
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
        } else {
            throw_error(ErrorType::Analysis, "left side of assignment must be an identifier", node->line, node->col);
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

void TypeCkContext::new_global_var(std::string name, std::shared_ptr<Type> type,
                                   bool is_mut, std::string symbol,
                                   bool is_export) noexcept {
    global_scope.emplace_back(std::move(name), std::move(type), is_mut,
                              std::move(symbol), is_export);
}

std::vector<Scope::Var> &TypeCkContext::get_global() noexcept {
    return global_scope;
}

