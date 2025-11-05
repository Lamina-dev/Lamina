#pragma once
#include "value.hpp"
#include <memory>
#include <string>
#include <utility>
#include <vector>

// AST 基类
struct ASTNode {
    int start_ln = 0;
    int end_ln = 0;
    int start_col = 0;
    int end_col = 0;
    virtual ~ASTNode() = default;
    [[nodiscard]] virtual std::unique_ptr<ASTNode> clone() const = 0;
};

// 表达式基类
struct Expression :  ASTNode {
    std::string source;// 保存表达式源码
    [[nodiscard]] virtual std::unique_ptr<Expression> clone_expr() const = 0;

    [[nodiscard]] std::unique_ptr<ASTNode> clone() const final {
        return std::unique_ptr<ASTNode>(clone_expr().release());
    }
};

// 语句基类
struct Statement :  ASTNode {
    [[nodiscard]] virtual std::unique_ptr<Statement> clone_expr() const = 0;

    [[nodiscard]] std::unique_ptr<ASTNode> clone() const final {
        return std::unique_ptr<ASTNode>(clone_expr().release());
    }
};

// 字面量
struct LiteralExpr final :  Expression {
    Value::Type type;
    std::string value;
    LiteralExpr(std::string v, Value::Type t)
        : type(t), value(std::move(v)) {}

    // 实现克隆：值类型直接拷贝
    [[nodiscard]] std::unique_ptr<Expression> clone_expr() const override {
        return std::make_unique<LiteralExpr>(value, type);
    }
};

// 标识符
struct IdentifierExpr final :  Expression {
    std::string name;
    explicit IdentifierExpr(std::string  n) : name(std::move(n)) {}
    [[nodiscard]] std::unique_ptr<Expression> clone_expr() const override {
        return std::make_unique<IdentifierExpr>(name);
    }
};

// 变量引用
struct VarExpr final :  Expression {
    std::string name;
    explicit VarExpr(std::string  n) : name(std::move(n)) {}
    [[nodiscard]] std::unique_ptr<Expression> clone_expr() const override {
        return std::make_unique<VarExpr>(name);
    }
};

// 二元运算
struct BinaryExpr final :  Expression {
    std::string op;
    std::unique_ptr<Expression> left, right;
    BinaryExpr(std::string  o, std::unique_ptr<Expression> l, std::unique_ptr<Expression> r)
        : op(std::move(o)), left(std::move(l)), right(std::move(r)) {}
    [[nodiscard]] std::unique_ptr<Expression> clone_expr() const override {
        auto cloned_left = left ? left->clone_expr() : nullptr;
        auto cloned_right = right ? right->clone_expr() : nullptr;
        return std::make_unique<BinaryExpr>(op, std::move(cloned_left), std::move(cloned_right));
    }
};

// 一元运算
struct UnaryExpr final :  Expression {
    std::string op;
    std::unique_ptr<Expression> operand;
    UnaryExpr(std::string  o, std::unique_ptr<Expression> e)
        : op(std::move(o)), operand(std::move(e)) {}
    [[nodiscard]] std::unique_ptr<Expression> clone_expr() const override {
        auto cloned_operand = operand ? operand->clone_expr() : nullptr;
        return std::make_unique<UnaryExpr>(op, std::move(cloned_operand));
    }
};

// 变量声明
struct VarDeclStmt final :  Statement {
    std::string name;
    std::unique_ptr<Expression> expr;
    VarDeclStmt(std::string  n, std::unique_ptr<Expression> e)
        : name(std::move(n)), expr(std::move(e)) {}
    [[nodiscard]] std::unique_ptr<Statement> clone_expr() const override {
        auto cloned_expr = expr ? expr->clone_expr() : nullptr;
        return std::make_unique<VarDeclStmt>(name, std::move(cloned_expr));
    }
};

// 赋值
struct AssignStmt final :  Statement {
    std::string name;
    std::unique_ptr<Expression> expr;
    AssignStmt(std::string  n, std::unique_ptr<Expression> e)
        : name(std::move(n)), expr(std::move(e)) {}
    [[nodiscard]] std::unique_ptr<Statement> clone_expr() const override {
        auto cloned_expr = expr ? expr->clone_expr() : nullptr;
        return std::make_unique<AssignStmt>(name, std::move(cloned_expr));
    }
};


// 复合语句块
struct BlockStmt final :  Statement {
    std::vector<std::unique_ptr<Statement>> statements{};
    explicit BlockStmt(std::vector<std::unique_ptr<Statement>> s) : statements(std::move(s)) {}
    [[nodiscard]] std::unique_ptr<Statement> clone_expr() const override {
        std::vector<std::unique_ptr<Statement>> cloned_stmts;
        for (const auto& stmt : statements) {
            cloned_stmts.push_back(stmt->clone_expr());
        }
        return std::make_unique<BlockStmt>(std::move(cloned_stmts));
    }
};

// if 语句
struct IfStmt final :  Statement {
    std::unique_ptr<Expression> condition;
    std::unique_ptr<BlockStmt> thenBlock;
    std::unique_ptr<BlockStmt> elseBlock;
    IfStmt(std::unique_ptr<Expression> cond, std::unique_ptr<BlockStmt> thenB, std::unique_ptr<BlockStmt> elseB)
        : condition(std::move(cond)), thenBlock(std::move(thenB)), elseBlock(std::move(elseB)) {}
    [[nodiscard]] std::unique_ptr<Statement> clone_expr() const override {
        // condition
        auto cloned_cond = condition ? condition->clone_expr() : nullptr;
        auto cloned_then = [this]() -> std::unique_ptr<BlockStmt> {
            if (!thenBlock) return nullptr;
            // 调用 BlockStmt::clone_expr 后强转，因返回的是 Statement 指针
            auto stmt_ptr = thenBlock->clone_expr().release();
            return std::unique_ptr<BlockStmt>(dynamic_cast<BlockStmt*>(stmt_ptr));
        }();

        auto cloned_else = [this]() -> std::unique_ptr<BlockStmt> {
            if (!elseBlock) return nullptr;
            auto stmt_ptr = elseBlock->clone_expr().release();
            return std::unique_ptr<BlockStmt>(dynamic_cast<BlockStmt*>(stmt_ptr));
        }();

        return std::make_unique<IfStmt>(std::move(cloned_cond), std::move(cloned_then), std::move(cloned_else));
    }
};

// while 语句
struct WhileStmt final :  Statement {
    std::unique_ptr<Expression> condition;
    std::unique_ptr<BlockStmt> body;
    WhileStmt(std::unique_ptr<Expression> cond, std::unique_ptr<BlockStmt> b)
        : condition(std::move(cond)), body(std::move(b)) {}
    [[nodiscard]] std::unique_ptr<Statement> clone_expr() const override {
        auto cloned_cond = condition ? condition->clone_expr() : nullptr;

        std::unique_ptr<BlockStmt> cloned_body;
        if (body) {
            const auto stmt_ptr = body->clone_expr().release();
            cloned_body = std::unique_ptr<BlockStmt>(dynamic_cast<BlockStmt*>(stmt_ptr));
        }

        return std::make_unique<WhileStmt>(std::move(cloned_cond), std::move(cloned_body));
    }
};

// repeat N times
struct RepeatStmt final :  Statement {
    // count expression (should evaluate to a positive integer literal in current grammar)
    std::unique_ptr<Expression> count;
    std::unique_ptr<BlockStmt> body;
    RepeatStmt(std::unique_ptr<Expression> c, std::unique_ptr<BlockStmt> b)
        : count(std::move(c)), body(std::move(b)) {}
    [[nodiscard]] std::unique_ptr<Statement> clone_expr() const override {
        auto cloned_count = count ? count->clone_expr() : nullptr;

        std::unique_ptr<BlockStmt> cloned_body;
        if (body) {
            const auto stmt_ptr = body->clone_expr().release();
            cloned_body = std::unique_ptr<BlockStmt>(dynamic_cast<BlockStmt*>(stmt_ptr));
        }

        return std::make_unique<RepeatStmt>(std::move(cloned_count), std::move(cloned_body));
    }
};

// 函数定义
struct FuncDefStmt final :  Statement {
    std::string name;
    std::vector<std::string> params;
    std::unique_ptr<BlockStmt> body;
    FuncDefStmt(std::string  n, const std::vector<std::string>& p, std::unique_ptr<BlockStmt> b)
        : name(std::move(n)), params(p), body(std::move(b)) {}
    [[nodiscard]] std::unique_ptr<Statement> clone_expr() const override {
        std::string cloned_name = name;
        std::vector<std::string> cloned_params = params;
        std::unique_ptr<BlockStmt> cloned_body;
        if (body) {
            const auto stmt_ptr = body->clone_expr().release();
            cloned_body = std::unique_ptr<BlockStmt>(dynamic_cast<BlockStmt*>(stmt_ptr));
        }

        return std::make_unique<FuncDefStmt>(std::move(cloned_name), cloned_params, std::move(cloned_body));
    }
};

struct GetMemberExpr;
// 函数调用
struct CallExpr final :  Expression {
    std::unique_ptr<Expression> callee;
    std::vector<std::unique_ptr<Expression>> args;
    CallExpr(std::unique_ptr<Expression> c, std::vector<std::unique_ptr<Expression>> a)
        : callee(std::move(c)), args(std::move(a)) {}
    [[nodiscard]] std::unique_ptr<Expression> clone_expr() const override {
        auto cloned_callee = callee ? callee->clone_expr() : nullptr;
        std::vector<std::unique_ptr<Expression>> cloned_args;
        for (const auto& arg : args) {
            cloned_args.push_back(arg->clone_expr());
        }

        return std::make_unique<CallExpr>(std::move(cloned_callee), std::move(cloned_args));
    }

};

// 数组字面量
struct ArrayExpr final :  Expression {
    std::vector<std::unique_ptr<Expression>> elements;
    explicit ArrayExpr(std::vector<std::unique_ptr<Expression>> elems)
        : elements(std::move(elems)) {}
    [[nodiscard]] std::unique_ptr<Expression> clone_expr() const override {
        // 克隆 elements 容器
        std::vector<std::unique_ptr<Expression>> cloned_elems;
        for (const auto& elem : elements) {
            cloned_elems.push_back(elem->clone_expr());
        }
        return std::make_unique<ArrayExpr>(std::move(cloned_elems));
    }
};

// return 语句
struct ReturnStmt final :  Statement {
    std::unique_ptr<Expression> expr;
    explicit ReturnStmt(std::unique_ptr<Expression> e) : expr(std::move(e)) {}
    [[nodiscard]] std::unique_ptr<Statement> clone_expr() const override {
        auto cloned_expr = expr ? expr->clone_expr() : nullptr;
        return std::make_unique<ReturnStmt>(std::move(cloned_expr));
    }
};

// include 语句
struct IncludeStmt final :  Statement {
    std::string module;
    explicit IncludeStmt(std::string  m) : module(std::move(m)) {}
    [[nodiscard]] std::unique_ptr<Statement> clone_expr() const override {
        return std::make_unique<IncludeStmt>(module);
    }
};

// 空语句
struct NullStmt final :  Statement {
    NullStmt() = default;
    [[nodiscard]] std::unique_ptr<Statement> clone_expr() const override {
        return std::make_unique<NullStmt>();
    }
};

// break 语句
struct BreakStmt final :  Statement {
    BreakStmt() = default;
    [[nodiscard]] std::unique_ptr<Statement> clone_expr() const override {
        return std::make_unique<BreakStmt>();
    }
};

// continue 语句
struct ContinueStmt final :  Statement {
    ContinueStmt() = default;
    [[nodiscard]] std::unique_ptr<Statement> clone_expr() const override {
        return std::make_unique<ContinueStmt>();
    }
};

// 表达式语句
struct ExprStmt final :  Statement {
    std::unique_ptr<Expression> expr;
    explicit ExprStmt(std::unique_ptr<Expression> e) : expr(std::move(e)) {}
    [[nodiscard]] std::unique_ptr<Statement> clone_expr() const override {
        auto cloned_expr = expr ? expr->clone_expr() : nullptr;
        return std::make_unique<ExprStmt>(std::move(cloned_expr));
    }
};

// Struct声明
struct StructDeclStmt final :  Statement {
    std::string name;
    std::vector<std::pair<std::string, std::unique_ptr<Expression>>> init_vec;
    std::vector<std::unique_ptr<IdentifierExpr>> includes;
    explicit StructDeclStmt(
            std::string  n,
            std::vector<std::pair<std::string, std::unique_ptr<Expression>>> v,
            std::vector<std::unique_ptr<IdentifierExpr>> i)
        : name(std::move(n)), init_vec(std::move(v)), includes(std::move(i)) {}
    [[nodiscard]] std::unique_ptr<Statement> clone_expr() const override {
        std::string cloned_name = name;
        std::vector<std::pair<std::string, std::unique_ptr<Expression>>> cloned_init_vec;
        for (const auto& pair : init_vec) {
            auto cloned_expr = pair.second ? pair.second->clone_expr() : nullptr;
            cloned_init_vec.emplace_back(pair.first, std::move(cloned_expr));
        }
        // 克隆 includes（需强转）
        std::vector<std::unique_ptr<IdentifierExpr>> cloned_includes;
        for (const auto& id : includes) {
            if (!id) {
                cloned_includes.push_back(nullptr);
                continue;
            }
            const auto expr_ptr = id->clone_expr().release();
            cloned_includes.push_back(std::unique_ptr<IdentifierExpr>(dynamic_cast<IdentifierExpr*>(expr_ptr)));
        }

        return std::make_unique<StructDeclStmt>(std::move(cloned_name), std::move(cloned_init_vec), std::move(cloned_includes));
    }
};

// 匿名Struct声明
struct LambdaStructDeclExpr final :  Expression {
    std::vector<std::pair<std::string, std::unique_ptr<Expression>>> init_vec;
    explicit LambdaStructDeclExpr(std::vector<std::pair<std::string, std::unique_ptr<Expression>>> v)
        : init_vec(std::move(v)) {}
    [[nodiscard]] std::unique_ptr<Expression> clone_expr() const override {
        std::vector<std::pair<std::string, std::unique_ptr<Expression>>> cloned_init_vec;
        for (const auto& pair : init_vec) {
            auto cloned_expr = pair.second ? pair.second->clone_expr() : nullptr;
            cloned_init_vec.emplace_back(pair.first, std::move(cloned_expr));
        }
        return std::make_unique<LambdaStructDeclExpr>(std::move(cloned_init_vec));
    }
};

// Define语句（用于设置常量，如递归深度）
struct DefineStmt final :  Statement {
    std::string name;
    std::unique_ptr<Expression> value;
    DefineStmt(std::string  n, std::unique_ptr<Expression> v)
        : name(std::move(n)), value(std::move(v)) {}
    [[nodiscard]] std::unique_ptr<Statement> clone_expr() const override {
        // 克隆 value
        auto cloned_value = value ? value->clone_expr() : nullptr;
        return std::make_unique<DefineStmt>(name, std::move(cloned_value));
    }
};

// BigInt变量声明
struct BigIntDeclStmt final :  Statement {
    std::string name;
    std::unique_ptr<Expression> init_value;
    explicit BigIntDeclStmt(std::string  n, std::unique_ptr<Expression> v = nullptr)
        : name(std::move(n)), init_value(std::move(v)) {}
    [[nodiscard]] std::unique_ptr<Statement> clone_expr() const override {
        // 克隆 init_value（允许为空）
        auto cloned_init = init_value ? init_value->clone_expr() : nullptr;
        return std::make_unique<BigIntDeclStmt>(name, std::move(cloned_init));
    }
};

// 获取成员
struct GetMemberExpr final :  Expression {
    std::unique_ptr<Expression> father;
    std::unique_ptr<IdentifierExpr> child;
    GetMemberExpr(std::unique_ptr<Expression> f, std::unique_ptr<IdentifierExpr> c)
        :father(std::move(f)), child(std::move(c)) {}
    [[nodiscard]] std::unique_ptr<Expression> clone_expr() const override {
        auto cloned_father = father ? father->clone_expr() : nullptr;
        std::unique_ptr<IdentifierExpr> cloned_child;
        if (child) {
            const auto expr_ptr = child->clone_expr().release();
            cloned_child = std::unique_ptr<IdentifierExpr>(dynamic_cast<IdentifierExpr*>(expr_ptr));
        }

        return std::make_unique<GetMemberExpr>(std::move(cloned_father), std::move(cloned_child));
    }
};

// 设置成员
struct SetMemberExpr final :  Expression {
    std::unique_ptr<Expression> g_mem;
    std::unique_ptr<Expression> val;
    SetMemberExpr(std::unique_ptr<Expression> g_mem, std::unique_ptr<Expression> val)
        : g_mem(std::move(g_mem)), val(std::move(val)) {}
    // 实现克隆方法
    [[nodiscard]] std::unique_ptr<Expression> clone_expr() const override {
        auto cloned_g_mem = g_mem ? g_mem->clone_expr() : nullptr;

        // 克隆 val
        std::unique_ptr<Expression> cloned_val;
        if (val) {
            // 先通过 clone_expr
            const auto expr_ptr = val->clone_expr().release();
            cloned_val = std::unique_ptr<Expression>(
                dynamic_cast<Expression*>(expr_ptr)
            );
        }

        // 用克隆后的成员构造新的 SetMemberExpr 并返回
        return std::make_unique<SetMemberExpr>(
            std::move(cloned_g_mem),
            std::move(cloned_val)
        );
    }
};

// 命名空间访问成员
struct NameSpaceGetMemberExpr final :  Expression {
    std::unique_ptr<Expression> father;
    std::unique_ptr<IdentifierExpr> child;
    NameSpaceGetMemberExpr(std::unique_ptr<Expression> f, std::unique_ptr<IdentifierExpr> c)
        :father(std::move(f)), child(std::move(c)) {}
    [[nodiscard]] std::unique_ptr<Expression> clone_expr() const override {
        // 逻辑同 GetMemberExpr
        auto cloned_father = father ? father->clone_expr() : nullptr;
        std::unique_ptr<IdentifierExpr> cloned_child;
        if (child) {
            auto expr_ptr = child->clone_expr().release();
            cloned_child = std::unique_ptr<IdentifierExpr>(dynamic_cast<IdentifierExpr*>(expr_ptr));
        }
        return std::make_unique<NameSpaceGetMemberExpr>(std::move(cloned_father), std::move(cloned_child));
    }
};

// 获取项
struct GetItemExpr final :  Expression {
    std::unique_ptr<Expression> father;
    std::vector<std::unique_ptr<Expression>> params;
    GetItemExpr(std::unique_ptr<Expression> f, std::vector<std::unique_ptr<Expression>> p)
        : father(std::move(f)), params(std::move(p)) {}
    [[nodiscard]] std::unique_ptr<Expression> clone_expr() const override {
        auto cloned_father = father ? father->clone_expr() : nullptr;
        std::vector<std::unique_ptr<Expression>> cloned_params;
        for (const auto& param : params) {
            cloned_params.push_back(param->clone_expr());
        }

        return std::make_unique<GetItemExpr>(std::move(cloned_father), std::move(cloned_params));
    }
};

// 声明匿名函数
struct LambdaDeclExpr final :  Expression {
    std::string name;
    std::vector<std::string> params;
    std::unique_ptr<BlockStmt> body;
    LambdaDeclExpr(std::string n, std::vector<std::string> p, std::unique_ptr<BlockStmt> b)
        : name(std::move(n)), params(std::move(p)), body(std::move(b)) {}
    [[nodiscard]] std::unique_ptr<Expression> clone_expr() const override {
        std::string cloned_name = name;
        std::vector<std::string> cloned_params = params;

        std::unique_ptr<BlockStmt> cloned_body;
        if (body) {
            auto stmt_ptr = body->clone_expr().release();
            cloned_body = std::unique_ptr<BlockStmt>(dynamic_cast<BlockStmt*>(stmt_ptr));
        }

        return std::make_unique<LambdaDeclExpr>(std::move(cloned_name), std::move(cloned_params), std::move(cloned_body));
    }
};