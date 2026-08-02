#pragma once
#include <optional>

#include "ast.hpp"

namespace lmx::hir {
/*
 * Lamina Compiler HIR
 * 这个阶段其实并不产生新东西，一开始想产生，后来发现必要性不大，
 * 现在它用来初步验证语法树正确性，填补空缺的数据类型
 * 完成后，产生一份符号表  
 */

using HirNode = ASTNode;


class TypeCkContext {
    std::vector<Scope> scope_stack;

    // Scope *parse_scope(ExprNode *node) noexcept;

    std::optional<Scope::Var *> find_var(const std::string &name) noexcept;

    void new_var(std::string name, std::shared_ptr<Type> type, Scope *scope, bool is_mut = false) noexcept;
    void new_cur_scope_var(std::string name, std::shared_ptr<Type> type, bool is_mut = false) noexcept;
    void new_global_var(std::string name, std::shared_ptr<Type> type, bool is_mut = false) noexcept;

    std::vector<Scope::Var> &get_global() noexcept;

    [[nodiscard]] bool is_global_scope() const noexcept;
public:
    explicit TypeCkContext() noexcept;

    std::vector<Scope::Var> check_module(const std::shared_ptr<Module> &mod) noexcept;

    void check_expr(std::shared_ptr<ExprNode> &expr) noexcept;

    void check_stmt(StmtNode *stmt) noexcept;

    // void reset() noexcept;
    std::shared_ptr<Type> inference_type(ExprNode *type) noexcept;
};

}
