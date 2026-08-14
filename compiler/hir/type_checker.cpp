//
// Created by meian on 2026/7/5.
//

#include "type_checker.hpp"

#include <functional>
#include <ranges>
#include <map>
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
    if (!type || type->kind != TypeKind::Basic) return false;
    const auto kind = std::reinterpret_pointer_cast<BasicType>(type)->type;
    return kind == runtime::ValueKind::Int ||
           kind == runtime::ValueKind::Fraction ||
           kind == runtime::ValueKind::Real ||
           kind == runtime::ValueKind::Expr;
}

bool is_bool_or_expr_type(const std::shared_ptr<Type>& type) noexcept {
    if (!type || type->kind != TypeKind::Basic) return false;
    const auto kind = std::reinterpret_pointer_cast<BasicType>(type)->type;
    return kind == runtime::ValueKind::Bool ||
           kind == runtime::ValueKind::Expr;
}

bool is_expr_type(const std::shared_ptr<Type>& type) noexcept {
    return is_basic_type(type, runtime::ValueKind::Expr);
}

bool is_expr_constructible(const std::shared_ptr<Type>& type) noexcept {
    if (!type || type->kind != TypeKind::Basic) return false;
    const auto kind = std::reinterpret_pointer_cast<BasicType>(type)->type;
    return kind == runtime::ValueKind::Int ||
           kind == runtime::ValueKind::Fraction ||
           kind == runtime::ValueKind::Real ||
           kind == runtime::ValueKind::Expr;
}

bool is_named_type(const std::shared_ptr<Type>& type, const std::string_view name) noexcept {
    return type && type->kind == TypeKind::Named &&
           std::static_pointer_cast<NamedType>(type)->name == name;
}

std::shared_ptr<Type> unify_types(const std::shared_ptr<Type>& lhs,
                                  const std::shared_ptr<Type>& rhs) noexcept;

std::shared_ptr<Type> literal_payload_type(const LiteralPayloadNode& node) noexcept {
    if (node.payload_kind == LiteralPayloadNode::Kind::Interval) {
        if (node.elements.size() != 2) return type_pool.unknown();
        auto element = unify_types(node.elements[0]->type, node.elements[1]->type);
        return element ? type_pool.named("interval", {std::move(element)}) : type_pool.unknown();
    }

    if (node.elements.empty()) return type_pool.named("set", {type_pool.unknown()});
    auto element = node.elements.front()->type;
    for (std::size_t i = 1; i < node.elements.size(); ++i) {
        element = unify_types(element, node.elements[i]->type);
        if (!element) return type_pool.unknown();
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

std::shared_ptr<Type> unify_types(const std::shared_ptr<Type>& lhs,
                                  const std::shared_ptr<Type>& rhs) noexcept {
    if (!lhs || !rhs) return nullptr;
    if (lhs->kind == TypeKind::Unknown) return rhs;
    if (rhs->kind == TypeKind::Unknown) return lhs;
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
        return type_pool.array(resolve_type(array->type), array->len);
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
        if (current->kind == TypeKind::Nullable)
            return comparable(std::static_pointer_cast<NullableType>(current)->value_type);
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
        if (node->id == "i" || node->id == "I") {
            return type_pool.basic(runtime::ValueKind::Expr);
        }
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
        if (node->op == BinaryNode::Op::In || node->op == BinaryNode::Op::NotIn) {
            return is_expr_type(left_ty) || is_expr_type(right_ty)
                ? type_pool.basic(runtime::ValueKind::Expr)
                : type_pool.basic(runtime::ValueKind::Bool);
        }
        if (is_expr_type(left_ty) || is_expr_type(right_ty)) {
            return type_pool.basic(runtime::ValueKind::Expr);
        }
        if (left_ty->equals(right_ty.get())) return left_ty;
        break;
    }
    case ASTKind::LiteralPayload: {
        return literal_payload_type(*reinterpret_cast<LiteralPayloadNode*>(type));
    }
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
    mod->adt_exports.clear();

    if (!mod->native_funcs.empty() && mod->lib_name.empty()) {
        throw_error(ErrorType::Analysis, "module not `static` declare dynamic library, cannot declare native function", 0 , 0);

    }
    for (const auto& n : mod->native_funcs) {
        new_global_var(n->func_id, n->make_type());
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
    for (auto& node : mod->decls) {
        if (node->kind == ASTKind::ImportStmt) check_stmt(node);
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
    // reset();
    for (auto& node : mod->decls) {
        if (node->kind == ASTKind::ImportStmt) continue;
        check_stmt(node);
    }

    cur_module = save_cur_module;

    std::vector<Scope::Var> result;

    for (const auto& v : get_global()) {
        if (v.type->kind == TypeKind::Function ||
            v.type->kind == TypeKind::NativeFunction) {
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
    case ASTKind::Identifier: {
        auto* node = reinterpret_cast<IdentifierNode*>(expr.get());
        if (node->id == "i" || node->id == "I") {
            node->type = type_pool.basic(runtime::ValueKind::Expr);
            break;
        }
        if (const auto re = find_var(node->id); re.has_value()) {
            node->type = (*re)->type;
            break;
        }
        if (const auto re = find_global(node->id); re.has_value()) {
            if ((*re)->type->kind == TypeKind::AdtConstructor) {
                const auto constructor = std::static_pointer_cast<AdtConstructorType>((*re)->type);
                if (!constructor->fields.empty()) {
                    node->type = (*re)->type;
                    break;
                }
                node->is_zero_adt_constructor = true;
                node->adt_type_name = constructor->type_name;
                std::vector<std::shared_ptr<Type>> args(constructor->type_params.size(), type_pool.unknown());
                node->type = type_pool.named(constructor->type_name, std::move(args));
                break;
            }
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
            t2->type != runtime::ValueKind::Int &&
            t2->type != runtime::ValueKind::Fraction &&
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
            node->type = type_pool.named("Binding", {lty, rty});
            break;
        }
        if ((node->op == BinaryNode::Op::Eq || node->op == BinaryNode::Op::Ne) &&
            lty->kind == TypeKind::Named && rty->kind == TypeKind::Named) {
            if (auto unified = unify_types(lty, rty)) {
                if (!is_equality_comparable(unified)) {
                    throw_error(ErrorType::Analysis, "ADT fields are not equality comparable", node->line, node->col);
                    break;
                }
                if (contains_unknown_type(lty)) node->lhs->type = unified;
                if (contains_unknown_type(rty)) node->rhs->type = unified;
                node->type = type_pool.basic(runtime::ValueKind::Bool);
                break;
            }
        }
        if (node->op == BinaryNode::Op::In || node->op == BinaryNode::Op::NotIn) {
            if (is_expr_type(lty) || is_expr_type(rty)) {
                if (is_expr_constructible(lty)) {
                    node->lhs->type = type_pool.basic(runtime::ValueKind::Expr);
                }
                if (node->rhs->kind == ASTKind::LiteralPayload || is_expr_constructible(rty)) {
                    node->rhs->type = type_pool.basic(runtime::ValueKind::Expr);
                }
                node->type = type_pool.basic(runtime::ValueKind::Expr);
                break;
            }
            if (!is_named_type(rty, "set") && !is_named_type(rty, "interval")) {
                goto binary_type_mismatch;
            }
            const auto container = std::static_pointer_cast<NamedType>(rty);
            if (container->args.size() != 1 || !type_assignable(container->args.front(), lty)) {
                goto binary_type_mismatch;
            }
            node->type = type_pool.basic(runtime::ValueKind::Bool);
            break;
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
                if (is_bool_or_expr_type(lty) && is_bool_or_expr_type(rty)) {
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
    case ASTKind::LiteralPayload: {
        const auto node = reinterpret_cast<LiteralPayloadNode*>(expr.get());
        for (auto& element : node->elements) {
            check_expr(element);
        }
        node->type = literal_payload_type(*node);
        if (node->type->kind == TypeKind::Unknown) {
            const char* diagnostic = node->payload_kind == LiteralPayloadNode::Kind::Set
                ? "set elements must have one type"
                : "interval bounds must have one type";
            throw_error(ErrorType::Analysis, diagnostic, node->line, node->col);
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
                if (has_expr_arg) {
                    node->expr->type = type_pool.basic(runtime::ValueKind::Expr);
                    node->type = type_pool.basic(runtime::ValueKind::Expr);
                    break;
                }
            }
        }
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
            for (auto i = 0; i < len; i++) {
                const auto param = func_ty->params_ty[i];
                check_expr(node->suffix->exprs[i]);
                TypeBindings call_bindings;
                if (contains_unknown_type(node->suffix->exprs[i]->type) &&
                    type_assignable(param, node->suffix->exprs[i]->type)) {
                    node->suffix->exprs[i]->type = param;
                }
                if (!type_assignable(param, node->suffix->exprs[i]->type)) {
                    if (node->expr->kind == ASTKind::Identifier &&
                        is_expr_type(node->suffix->exprs[i]->type)) {
                        node->type = type_pool.basic(runtime::ValueKind::Expr);
                        break;
                    }
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
        const auto node = reinterpret_cast<AsExprNode*>(expr.get());
        check_expr(node->expr);
        if (is_expr_type(node->cast_type) && is_expr_constructible(node->expr->type)) {
            node->expr->type = type_pool.basic(runtime::ValueKind::Expr);
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
        } else if (const auto result = left_ty->find_var(node->rhs->id); result.has_value()) {
            const auto* var = *result;
            node->rhs->type = var->type;
            node->type = var->type;
        } else {
            throw_error(ErrorType::Analysis, "module not have var `" + node->rhs->id + "`", node->line, node->col);
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
        new_global_var(resolved->binding_name, resolved->type);
        cur_module->imports[resolved->source_path] = resolved->type;
        break;
    }
    case ASTKind::SymDecl: {
        const auto* node = reinterpret_cast<SymDeclNode*>(stmt.get());
        for (const auto& id : node->ids) {
            if (id == "i" || id == "I") {
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
    //case ASTKind::Exprs:
    //    break;
    //case ASTKind::ParamsDeclNode:
    //    break;
    case ASTKind::FuncImpl: {
        auto* node = reinterpret_cast<FuncImplNode*>(stmt.get());
        if (!is_global_scope()) throw_error(ErrorType::Analysis, "function only define in GlobalScope", stmt->line, stmt->col);

        for (auto& [name, type] : node->params->stmts) type = resolve_type(type);
        node->return_type = resolve_type(node->return_type);
        new_global_var(node->func_id, node->make_type());
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
        for (auto& s : scope_stack | std::views::reverse) {
            if (s.scope == Scope::ScopeType::Function) {
                if (Type::is_null_type(s.return_type.get())) {
                    s.return_type = node->expr->type;
                    break;
                }
                if (contains_unknown_type(node->expr->type) && type_assignable(s.return_type, node->expr->type))
                    node->expr->type = s.return_type;
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
        check_expr(node->expr);
        if (Type::is_null_type(scope_stack.back().return_type.get()))
            scope_stack.back().return_type = node->expr->type;
        else {
            if (contains_unknown_type(node->expr->type) &&
                type_assignable(scope_stack.back().return_type, node->expr->type))
                node->expr->type = scope_stack.back().return_type;
            if (!type_assignable(scope_stack.back().return_type, node->expr->type)) {
                throw_error(ErrorType::Analysis, "return type is inconsistent with the above", node->line, node->col);
                break;
            }
        }
        // node->expr->type = inference_type(node->expr.get());
        break;
    }
    case ASTKind::VarDecl: {
        const auto node = reinterpret_cast<VarDeclNode*>(stmt.get());
        if (node->id == "i" || node->id == "I") {
            throw_error(ErrorType::Analysis, "ImaginaryUnitReserved", node->line, node->col);
            break;
        }
        if (!Type::is_null_type(node->type.get())) node->type = resolve_type(node->type);
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
                node->init_value->type = type_pool.basic(runtime::ValueKind::Expr);
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

