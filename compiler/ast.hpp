//
// Created by meian on 2026/4/3.
//

#pragma once
#include "../runtime/object/value.hpp"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
namespace lmx {
enum class ASTKind {
    ExprStmt,
    Literal,
    Identifier,
    Unary,
    Binary,
    Return,
    Block,
    Exprs,
    SuffixParen,
    SuffixBracket,
    ParamsDeclNode,
    FuncImpl, TailReturn, IfExpr, VarDecl, BreakStmt
};

enum class TypeKind {
    Basic, Array, Named, Unknown, String, Function, None
};

struct Type {
    TypeKind kind;

    explicit Type(const TypeKind kind) : kind(kind) {
    }

    virtual ~Type();

    [[nodiscard]] virtual bool equals(Type *other) const noexcept = 0;

    static bool is_null_type(const Type* kind) noexcept {
        return !kind || kind->kind == TypeKind::Unknown;
    }
};

struct UnknownType : Type {
    explicit UnknownType() : Type(TypeKind::Unknown) {}

    bool equals(Type *other) const noexcept override;
};
struct NoneType : Type {
    explicit NoneType() : Type(TypeKind::None) {}
    bool equals(Type *other) const noexcept override;
};

struct BasicType : Type {
    runtime::ValueKind type;

    explicit BasicType(const runtime::ValueKind type) noexcept : Type(TypeKind::Basic), type(type) {
    }

    ~BasicType() override;

    bool equals(Type *other) const noexcept override;
};

struct StringType : Type {
    explicit StringType() noexcept : Type(TypeKind::String) {}

    bool equals(Type *other) const noexcept override;
};
struct FunctionType : Type {
    std::vector<std::shared_ptr<Type>> params_ty;
    std::shared_ptr<Type> ret_ty;

    explicit FunctionType(std::vector<std::shared_ptr<Type>> params_ty, std::shared_ptr<Type> ret_ty) noexcept;
    bool equals(Type *other) const noexcept override;

};
struct NamedType : Type {
    std::string name;

    explicit NamedType(std::string name) noexcept : Type(TypeKind::Named), name(std::move(name)) {
    }

    bool equals(Type *other) const noexcept override{return false;}
};

struct ArrayType : Type {
    std::shared_ptr<Type> type;
    size_t len;

    explicit ArrayType(const std::shared_ptr<Type> &type, const size_t len)
        : Type(TypeKind::Array), type(type), len(len) {}

    ~ArrayType() override;

    bool equals(Type *other) const noexcept override;
};

struct ASTNode {
    ASTKind kind;
    size_t line, col;

    explicit ASTNode(ASTKind kind, size_t line, size_t col) noexcept;
};
struct ExprNode : ASTNode {
    std::shared_ptr<Type> type;
    explicit ExprNode(ASTKind kind, size_t line, size_t col) noexcept;
};

struct StmtNode : ASTNode {
    explicit StmtNode(ASTKind kind, size_t line, size_t col) noexcept;
};
struct FuncImplNode;
struct Module {
    std::string name;
    std::vector<std::shared_ptr<StmtNode>> decls;
    std::vector<FuncImplNode> top_func_def;
    Module(std::string name, decltype(decls) decls) noexcept;
};
struct ExprStmtNode : StmtNode {
    std::shared_ptr<ExprNode> expr;

    explicit ExprStmtNode(size_t line, size_t col, std::shared_ptr<ExprNode> expr) noexcept;
};

struct ExprsNode : ExprNode {
    std::vector<std::shared_ptr<ExprNode> > exprs;

    explicit ExprsNode(size_t line, size_t col, std::vector<std::shared_ptr<ExprNode> > exprs) noexcept;
};

struct LiteralNode : ExprNode {
    enum class Kind {
        Integer, Float, String, Boolean,
    };

    std::string val;
    Kind kind;

    explicit LiteralNode(size_t line, size_t col, std::string val, Kind kind) noexcept;
};

struct IdentifierNode : ExprNode {
    std::string id;

    explicit IdentifierNode(size_t line, size_t col, std::string id) noexcept;
};

struct SuffixParenNode : ExprNode {
    std::shared_ptr<ExprNode> expr;
    std::shared_ptr<ExprsNode> suffix;

    explicit SuffixParenNode(size_t line, size_t col, std::shared_ptr<ExprNode> expr,
                             std::shared_ptr<ExprsNode> suffix) noexcept;
};

struct SuffixBracketNode : ExprNode {
    std::shared_ptr<ExprNode> expr;
    std::shared_ptr<ExprNode> suffix;

    explicit SuffixBracketNode(size_t line, size_t col, std::shared_ptr<ExprNode> expr,
                               std::shared_ptr<ExprNode> suffix) noexcept;
};

struct UnaryNode : ExprNode {
    std::string op;
    std::shared_ptr<ExprNode> expr;

    explicit UnaryNode(size_t line, size_t col, std::string op, std::shared_ptr<ExprNode> expr) noexcept;
};

struct BinaryNode : ExprNode {
    std::string op;
    std::shared_ptr<ExprNode> lhs;
    std::shared_ptr<ExprNode> rhs;

    explicit BinaryNode(size_t line, size_t col, std::shared_ptr<ExprNode> lhs, std::string op,
                        std::shared_ptr<ExprNode> rhs) noexcept;
};

struct ReturnNode : StmtNode {
    std::shared_ptr<ExprNode> expr;

    explicit ReturnNode(size_t line, size_t col, std::shared_ptr<ExprNode> expr) noexcept;

    explicit ReturnNode(size_t line, size_t col, const std::shared_ptr<ExprStmtNode> &s) noexcept;
};

struct TailReturnNode : StmtNode {
    std::shared_ptr<ExprNode> expr;

    explicit TailReturnNode(size_t line, size_t col, std::shared_ptr<ExprNode> expr) noexcept;

    explicit TailReturnNode(size_t line, size_t col, const std::shared_ptr<ExprStmtNode> &s) noexcept;
};

struct BlockExprNode : ExprNode {
    std::vector<std::shared_ptr<StmtNode> > stmts;

    BlockExprNode(size_t line, size_t col, std::vector<std::shared_ptr<StmtNode> > stmts) noexcept;
};

struct BreakStmtNode : StmtNode {
    explicit BreakStmtNode(size_t line, size_t col) noexcept;
};

struct ParamsDeclNode : StmtNode {
    std::vector<std::pair<std::string, std::shared_ptr<Type>>> stmts;
    explicit ParamsDeclNode(size_t line, size_t col, decltype(stmts) stmts) noexcept;
};

/*
 * 部分情况
 * 这个结构体会给block = nullptr
 * 代表仅声明函数
 */
struct FuncImplNode : StmtNode {
    std::string func_id;
    std::shared_ptr<ParamsDeclNode> params;
    std::shared_ptr<Type> return_type;

    std::shared_ptr<BlockExprNode> block;

    explicit FuncImplNode(size_t line, size_t col,
        decltype(func_id) func_id,
        std::shared_ptr<ParamsDeclNode> params,
        std::shared_ptr<Type> return_type,
        std::shared_ptr<BlockExprNode> block
        ) noexcept;

    std::shared_ptr<FunctionType> make_type() noexcept;
};

struct VarDeclNode : StmtNode {
    std::string id; // ExprStmt<binary: ::> or identifier
    std::shared_ptr<Type> type;
    bool is_mutable;
    std::shared_ptr<ExprNode> init_value{nullptr};

    explicit VarDeclNode(size_t line, size_t col, decltype(id) id, std::shared_ptr<Type> type, bool is_mutable) noexcept;
};

struct IfExprNode : ExprNode {
    std::shared_ptr<ExprNode> cond;
    std::shared_ptr<ExprNode> then;
    std::shared_ptr<ASTNode> els;

    explicit IfExprNode(size_t line, size_t col, std::shared_ptr<ExprNode> cond, std::shared_ptr<ExprNode> then, std::shared_ptr<ASTNode> els) noexcept;
};

}