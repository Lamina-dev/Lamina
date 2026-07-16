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
struct Scope {
    struct Var {
        std::string name;
        std::shared_ptr<Type> type;
        bool is_mut;
    };
    std::string name;
    std::vector<Var> vars;
    std::shared_ptr<Type> return_type;

    explicit Scope(std::string name) noexcept;

    explicit Scope() = default;
};

class HirContext {
    std::vector<Scope> scope_stack;

    // Scope *parse_scope(ExprNode *node) noexcept;

    std::optional<Scope::Var *> find_var(const std::string &name) noexcept;

    void new_var(std::string name, std::shared_ptr<Type> type, Scope *scope) noexcept;
    void new_cur_scope_var(std::string name, std::shared_ptr<Type> type) noexcept;
    void new_global_var(std::string name, std::shared_ptr<Type> type) noexcept;

    bool is_global_scope() noexcept;
public:
    explicit HirContext() noexcept;
    void check_module(const Module *mod) noexcept;

    void check_expr(ExprNode *expr) noexcept;

    void check_stmt(StmtNode *stmt) noexcept;



    void reset() noexcept;
    std::shared_ptr<Type> inference_type(ExprNode *type) noexcept;
};

}
