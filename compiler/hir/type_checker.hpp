#pragma once
#include <optional>
#include <unordered_map>

#include "../ast/ast.hpp"

namespace lmx::hir {
/*
 * Lamina Compiler HIR
 * 这个阶段其实并不产生新东西，一开始想产生，后来发现必要性不大，
 * 现在它用来初步验证语法树正确性，填补空缺的数据类型
 * 完成后，产生一份符号表  
 */

using HirNode = ASTNode;

struct ResolvedModule {
    std::string source_path;
    std::string binding_name;
    std::shared_ptr<ModuleType> type;
};

struct ModuleRequest {
    std::string name;
    std::string importer;
    size_t line;
    size_t col;
};

class ModuleResolver {
public:
    virtual ~ModuleResolver() = default;
    virtual std::optional<ResolvedModule> resolve_module(
        const ModuleRequest& request) noexcept = 0;
};

class TypeCkContext {
    ModuleResolver* module_resolver;
    std::vector<Scope> scope_stack;

    std::vector<Scope::Var> global_scope;
    std::unordered_map<std::string, TypeDeclNode*> adt_types;
    std::unordered_map<std::string, std::pair<TypeDeclNode*, AdtConstructorDecl*>> adt_constructors;

    // Scope *parse_scope(ExprNode *node) noexcept;

    std::optional<Scope::Var *> find_var(const std::string &name) noexcept;

    std::optional<Scope::Var *> find_global(const std::string &name) noexcept;
    std::shared_ptr<Type> resolve_type(const std::shared_ptr<Type>& type) noexcept;
    bool is_equality_comparable(const std::shared_ptr<Type>& type) noexcept;
    TypeDeclNode* find_module_adt(ModuleType* module, const std::string& name) noexcept;
    std::pair<TypeDeclNode*, AdtConstructorDecl*> find_module_constructor(
        ModuleType* module, const std::string& name) noexcept;

    static void new_var(std::string name, std::shared_ptr<Type> type, Scope *scope, bool is_mut = false) noexcept;
    void new_cur_scope_var(std::string name, std::shared_ptr<Type> type, bool is_mut = false) noexcept;
    void new_global_var(std::string name, std::shared_ptr<Type> type, bool is_mut = false) noexcept;

    std::vector<Scope::Var> &get_global() noexcept;

    [[nodiscard]] bool is_global_scope() const noexcept;
public:
    explicit TypeCkContext(ModuleResolver* module_resolver = nullptr) noexcept;

    std::vector<Scope::Var> check_module(const std::shared_ptr<Module> &mod) noexcept;

    void check_expr(std::shared_ptr<ExprNode> &expr) noexcept;

    void check_stmt(std::shared_ptr<StmtNode> &stmt) noexcept;

    // void reset() noexcept;
    std::shared_ptr<Type> inference_type(ExprNode *type) noexcept;
};

}
