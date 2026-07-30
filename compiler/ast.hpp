//
// Created by meian on 2026/4/3.
//

#pragma once
#include "../runtime/object/value.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
namespace lmx {
struct NativeFuncDeclNode;

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
    FuncImpl, TailReturn, IfExpr, VarDecl, BreakStmt,
    AssignStmt, AsExpr, DotExpr,
    NativeFuncDecl,
    NativeFuncCall,
    LoopStmt,
    ContinueStmt,
    ImportStmt,
};

enum class TypeKind {
    Basic, Array, Named, Unknown, String, Function, None, NativeFunction,
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

    static std::string to_string(const Type* kind) noexcept;
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
struct NativeFunctionType : Type {
    std::vector<std::shared_ptr<Type>> params_ty;
    std::shared_ptr<Type> ret_ty;
    std::string name;

    explicit NativeFunctionType(std::vector<std::shared_ptr<Type>> params_ty, std::shared_ptr<Type> ret_ty, std::string name) noexcept;
    bool equals(Type *other) const noexcept override;

    [[nodiscard]] bool have_va_list() const noexcept;
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
    std::string lib_name;
    std::vector<std::shared_ptr<StmtNode>> decls;
    std::vector<std::shared_ptr<NativeFuncDeclNode>> native_funcs;

    // key 是展开后的源码绝对路径，value是输出的module编码文件路径
    std::unordered_map<std::string, std::string> sub_mods;


    Module(std::string name, decltype(decls) decls) noexcept;


    /*
     * name必须是绝对路径
     */
    [[nodiscard]] bool module_is_imported(const std::string& other_name) const noexcept {
        return sub_mods.contains(other_name);
    }
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
struct NativeFuncCallExpr : ExprNode {
    std::shared_ptr<ExprNode> expr;
    std::shared_ptr<ExprsNode> suffix;

    explicit NativeFuncCallExpr(const SuffixParenNode* sp) noexcept;
};
static_assert(sizeof(SuffixParenNode) == sizeof(NativeFuncCallExpr));

struct SuffixBracketNode : ExprNode {
    std::shared_ptr<ExprNode> expr;
    std::shared_ptr<ExprNode> suffix;

    explicit SuffixBracketNode(size_t line, size_t col, std::shared_ptr<ExprNode> expr,
                               std::shared_ptr<ExprNode> suffix) noexcept;
};

struct UnaryNode : ExprNode {
    enum class Op {
        Neg,
    };
    Op op;
    std::shared_ptr<ExprNode> expr;

    explicit UnaryNode(size_t line, size_t col, Op op, std::shared_ptr<ExprNode> expr) noexcept;
};

struct BinaryNode : ExprNode {
    enum class Op {
        Add, Sub, Mul, Div, Mod, Pow,
        Gt, Ge, Lt, Le, Eq, Ne, And, Or
    };
    Op op;
    std::shared_ptr<ExprNode> lhs;
    std::shared_ptr<ExprNode> rhs;

    explicit BinaryNode(size_t line, size_t col, std::shared_ptr<ExprNode> lhs, Op op,
                        std::shared_ptr<ExprNode> rhs) noexcept;

    static std::string op_to_string(Op op) noexcept;
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

struct NativeFuncDeclNode : StmtNode {
    std::string func_id;
    std::shared_ptr<ParamsDeclNode> params;
    std::shared_ptr<Type> return_type;

    std::string symbol;

    explicit NativeFuncDeclNode(size_t line, size_t col,
        decltype(func_id) func_id,
        std::shared_ptr<ParamsDeclNode> params,
        std::shared_ptr<Type> return_type,
        std::string symbol
        ) noexcept;

    std::shared_ptr<NativeFunctionType> make_type() noexcept;
};

struct VarDeclNode : StmtNode {
    std::string id; // ExprStmt<binary: ::> or identifier
    std::shared_ptr<Type> type;
    bool is_mutable;
    std::shared_ptr<ExprNode> init_value{nullptr};

    explicit VarDeclNode(size_t line, size_t col, decltype(id) id, std::shared_ptr<Type> type, bool is_mutable) noexcept;
};

struct AssignStmtNode : StmtNode {
    std::shared_ptr<ExprNode> lhs;
    std::shared_ptr<ExprNode> rhs;

    explicit AssignStmtNode(size_t line, size_t col,
                            std::shared_ptr<ExprNode> lhs,
                            std::shared_ptr<ExprNode> rhs) noexcept;
};

struct IfExprNode : ExprNode {
    std::shared_ptr<ExprNode> cond;
    std::shared_ptr<ExprNode> then;
    std::shared_ptr<ExprNode> els;

    explicit IfExprNode(size_t line, size_t col, std::shared_ptr<ExprNode> cond, std::shared_ptr<ExprNode> then, std::shared_ptr<ExprNode> els) noexcept;
};

struct AsExprNode : ExprNode {
    std::shared_ptr<ExprNode> expr;
    std::shared_ptr<Type> cast_type;

    explicit AsExprNode(size_t line, size_t col,
                        std::shared_ptr<ExprNode> expr,
                        std::shared_ptr<Type> cast_type) noexcept;
};
struct DotExprNode : ExprNode {
    std::shared_ptr<ExprNode> expr;
    std::shared_ptr<IdentifierNode> rhs;
    explicit DotExprNode(size_t line, size_t col, std::shared_ptr<ExprNode> expr, std::shared_ptr<IdentifierNode> rhs) noexcept;

};

struct LoopStmtNode : StmtNode {
    std::shared_ptr<ExprNode> expr;
    std::vector<std::shared_ptr<StmtNode>> body;

    explicit LoopStmtNode(size_t line, size_t col, decltype(expr) expr, std::vector<std::shared_ptr<StmtNode>> body) noexcept;
};

struct ContinueStmtNode : StmtNode {
    explicit ContinueStmtNode(size_t line, size_t col) noexcept;
};

struct ImportStmtNode : StmtNode {
    std::string name;

    explicit ImportStmtNode(size_t line, size_t col, decltype(name) name) noexcept;
};

}