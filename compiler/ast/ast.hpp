//
// Created by meian on 2026/4/3.
//

#pragma once

#include "../../runtime/object/value.hpp"
#include "unit.hpp"

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace lmx {
struct Type;
struct NativeFuncDeclNode;
struct TypeDeclNode;

namespace hir {
struct Scope {
    struct Var {
        std::string name;
        std::shared_ptr<Type> type;
        bool is_mut;
    };
    enum class ScopeType {
        Function, Block, Loop
    };
    ScopeType scope{ScopeType::Function};
    std::string name;
    std::vector<Var> vars;
    std::shared_ptr<Type> return_type;

    explicit Scope(std::string name) noexcept;
    explicit Scope(ScopeType scope) noexcept;
    explicit Scope() = default;



};
}


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
    SymDecl,
    PipeExpr,
    LiteralPayload,
    TypeDecl,
    MatchExpr,
    ArrayLiteral,
    TupleLiteral,
    TupleGetExpr,
    UnitAnnotated,
    UnitDecl,
};

enum class TypeKind {
    Basic, Array, Named, Unknown, String, Function, None, NativeFunction,
    Module, AdtConstructor, Nullable, Tuple, Dimensioned
};
struct Type {
    TypeKind kind;

protected:
    /*
     * 类型实例唯一性保证：构造函数仅供 TypePool 访问（friend），
     * 禁止任何其它代码裸建 Type，确保相同类型全局只有一个实例。
     */
    explicit Type(const TypeKind kind) noexcept : kind(kind) {
    }

public:
    virtual ~Type();

    [[nodiscard]] virtual bool equals(Type *other) const noexcept = 0;

    static bool is_null_type(const Type* kind) noexcept {
        return !kind || kind->kind == TypeKind::Unknown;
    }

    static std::string to_string(const Type* kind) noexcept;
};

struct ModuleType : Type {
    friend class TypePool;

    std::string target_path;
    std::string load_path;
    std::string binding_name;
    std::vector<hir::Scope::Var> exports;
    std::vector<std::shared_ptr<TypeDeclNode>> adt_exports;
    std::vector<std::pair<std::string, UnitDefinition>> unit_exports;
    explicit ModuleType(std::string target_path,
                        std::string load_path,
                        std::string binding_name,
                        std::vector<hir::Scope::Var> exports,
                        std::vector<std::shared_ptr<TypeDeclNode>> adt_exports = {},
                        std::vector<std::pair<std::string, UnitDefinition>> unit_exports = {}) noexcept
    : Type(TypeKind::Module), target_path(std::move(target_path)), load_path(std::move(load_path)),
      binding_name(std::move(binding_name)), exports(std::move(exports)),
      adt_exports(std::move(adt_exports)), unit_exports(std::move(unit_exports)) {}

public:
    bool equals(Type *other) const noexcept override;

    std::optional<hir::Scope::Var*> find_var(const std::string& n) noexcept {
        for (auto& var : exports) {
            if (var.name == n) return &var;
        }
        return std::nullopt;
    }
    [[nodiscard]] std::optional<size_t> find_func_idx(const std::string& n) const noexcept {
        size_t func_idx = 0;
        for (const auto& exported : exports) {
            if (exported.type->kind != TypeKind::Function) continue;
            if (exported.name == n) return func_idx;
            ++func_idx;
        }
        return std::nullopt;
    }
};

struct TupleType : Type {
    friend class TypePool;
    bool equals(Type *other) const noexcept override {
        if (is_null_type(other)) return false;
        if (this == other) return true;
        if (other->kind != this->kind) return false;
        const auto* o = reinterpret_cast<TupleType*>(other);
        if (tys.size() != o->tys.size()) return false;

        for (size_t i = 0; i < tys.size(); i++) {
            if (!tys[i]->equals(o->tys[i].get())) return false;
        }
        return true;
    }

    std::vector<std::shared_ptr<Type>> tys;

private:
    explicit TupleType(decltype(tys) tys) noexcept : Type(TypeKind::Tuple), tys(std::move(tys)) {}
};
struct UnknownType : Type {
    friend class TypePool;
private:
    explicit UnknownType() : Type(TypeKind::Unknown) {}

public:
    bool equals(Type *other) const noexcept override;
};
struct NoneType : Type {
    friend class TypePool;
private:
    explicit NoneType() : Type(TypeKind::None) {}

public:
    bool equals(Type *other) const noexcept override;
};

struct BasicType : Type {
    friend class TypePool;

    runtime::ValueKind type;
private:
    explicit BasicType(const runtime::ValueKind type) noexcept : Type(TypeKind::Basic), type(type) {
    }

public:
    ~BasicType() override;

    bool equals(Type *other) const noexcept override;
};

struct DimensionedType : Type {
    friend class TypePool;

    UnitSpec syntax;
    UnitDefinition unit;
    bool resolved{false};

private:
    explicit DimensionedType(UnitSpec syntax) noexcept
        : Type(TypeKind::Dimensioned), syntax(std::move(syntax)) {}
    explicit DimensionedType(UnitDefinition unit) noexcept
        : Type(TypeKind::Dimensioned), unit(std::move(unit)), resolved(true) {}

public:
    bool equals(Type* other) const noexcept override;
};

struct StringType : Type {
    friend class TypePool;
private:
    explicit StringType() noexcept : Type(TypeKind::String) {}

public:
    bool equals(Type *other) const noexcept override;
};
struct FunctionType : Type {
    friend class TypePool;

    std::vector<std::shared_ptr<Type>> params_ty;
    std::shared_ptr<Type> ret_ty;
private:
    explicit FunctionType(std::vector<std::shared_ptr<Type>> params_ty, std::shared_ptr<Type> ret_ty) noexcept;
public:
    bool equals(Type *other) const noexcept override;
};
struct NativeFunctionType : Type {
    friend class TypePool;

    std::vector<std::shared_ptr<Type>> params_ty;
    std::shared_ptr<Type> ret_ty;
    std::string name;
private:
    explicit NativeFunctionType(std::vector<std::shared_ptr<Type>> params_ty, std::shared_ptr<Type> ret_ty, std::string name) noexcept;
public:
    bool equals(Type *other) const noexcept override;

    [[nodiscard]] bool have_va_list() const noexcept;
};
struct NamedType : Type {
    std::string name;
    std::vector<std::shared_ptr<Type>> args;

    explicit NamedType(std::string name, std::vector<std::shared_ptr<Type>> args = {}) noexcept
        : Type(TypeKind::Named), name(std::move(name)), args(std::move(args)) {
    }

    bool equals(Type *other) const noexcept override;
};

struct NullableType : Type {
    std::shared_ptr<Type> value_type;

    explicit NullableType(std::shared_ptr<Type> value_type) noexcept
        : Type(TypeKind::Nullable), value_type(std::move(value_type)) {}

    bool equals(Type *other) const noexcept override;
};
struct AdtConstructorType : Type {
    std::string type_name;
    std::string constructor;
    std::vector<std::string> type_params;
    std::vector<std::shared_ptr<Type>> fields;

    AdtConstructorType(std::string type_name,
                       std::string constructor,
                       std::vector<std::string> type_params,
                       std::vector<std::shared_ptr<Type>> fields) noexcept;
    bool equals(Type *other) const noexcept override;
};

struct ArrayType : Type {
    friend class TypePool;

    std::shared_ptr<Type> type;
private:
    explicit ArrayType(const std::shared_ptr<Type> &type)
        : Type(TypeKind::Array), type(type) {}

public:
    ~ArrayType() override;

    bool equals(Type *other) const noexcept override;
};

struct ASTNode {
    virtual ~ASTNode() = default;

    ASTKind kind;
    size_t line, col;
    std::allocator<ASTKind> a;

    explicit ASTNode(ASTKind kind, size_t line, size_t col) noexcept;
};
struct ExprNode : ASTNode {
    std::shared_ptr<Type> type;
    std::shared_ptr<Type> promoted_from_type;
    explicit ExprNode(ASTKind kind, size_t line, size_t col) noexcept;

    [[nodiscard]] LMX_INLINE bool have_ret_value() const noexcept {
        return !Type::is_null_type(type.get()) && type->kind != TypeKind::None;
    }

    /*
     * 该表达式能否直接编码进常量池。
     * 字面量返回 true，布尔字面量因常量池没有对应 Tag 返回 false。
     */
    [[nodiscard]] virtual bool is_constant() const noexcept { return false; }
};

struct StmtNode : ASTNode {
    explicit StmtNode(ASTKind kind, size_t line, size_t col) noexcept;
};
struct FuncImplNode;

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
        Integer, Float, String, Boolean, Null,
    };

    std::string val;
    Kind kind;

    explicit LiteralNode(size_t line, size_t col, std::string val, Kind kind) noexcept;

    /*
     * 常量池没有 Bool Tag，布尔字面量无法直接入池。
     */
    [[nodiscard]] bool is_constant() const noexcept override {
        return kind != Kind::Boolean;
    }
};

struct IdentifierNode : ExprNode {
    std::string id;
    bool is_zero_adt_constructor{false};
    std::string adt_type_name;

    explicit IdentifierNode(size_t line, size_t col, std::string id) noexcept;
};

struct SuffixParenNode : ExprNode {
    std::shared_ptr<ExprNode> expr;
    std::shared_ptr<ExprsNode> suffix;
    bool can_fast{false};
    bool allow_symbolic_call{false};
    bool is_symbolic_call{false};
    bool is_adt_constructor{false};
    std::string adt_type_name;
    std::string adt_constructor;

    explicit SuffixParenNode(size_t line, size_t col, std::shared_ptr<ExprNode> expr,
                             std::shared_ptr<ExprsNode> suffix) noexcept;
};
struct NativeFuncCallExpr : ExprNode {
    std::shared_ptr<ExprNode> expr;
    std::shared_ptr<ExprsNode> suffix;
    bool can_fast{false};
    bool allow_symbolic_call{false};
    bool is_symbolic_call{false};
    bool is_adt_constructor{false};
    std::string adt_type_name;
    std::string adt_constructor;

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
        Not,
    };
    Op op;
    std::shared_ptr<ExprNode> expr;

    explicit UnaryNode(size_t line, size_t col, Op op, std::shared_ptr<ExprNode> expr) noexcept;
};

struct BinaryNode : ExprNode {
    enum class Op {
        Add, Sub, Mul, Div, Mod, Pow,
        Gt, Ge, Lt, Le, Eq, Ne, And, Or, In, NotIn, Bind
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

    std::shared_ptr<ExprNode> block;

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
    enum class Kind { Type, Unit, Num, Scalar };

    std::shared_ptr<ExprNode> expr;
    std::shared_ptr<Type> cast_type;
    Kind cast_kind{Kind::Type};
    UnitSpec unit_syntax;
    UnitDefinition resolved_unit;

    explicit AsExprNode(size_t line, size_t col,
                        std::shared_ptr<ExprNode> expr,
                        std::shared_ptr<Type> cast_type) noexcept;
    explicit AsExprNode(size_t line, size_t col,
                        std::shared_ptr<ExprNode> expr,
                        Kind cast_kind,
                        UnitSpec unit_syntax = {}) noexcept;
};

struct UnitAnnotatedExprNode : ExprNode {
    std::shared_ptr<ExprNode> value;
    UnitSpec unit_syntax;
    UnitDefinition resolved_unit;

    UnitAnnotatedExprNode(size_t line, size_t col,
                          std::shared_ptr<ExprNode> value,
                          UnitSpec unit_syntax) noexcept;
};

struct UnitDeclNode : StmtNode {
    std::string name;
    std::shared_ptr<ExprNode> definition;
    UnitDefinition resolved_unit;

    UnitDeclNode(size_t line, size_t col, std::string name,
                 std::shared_ptr<ExprNode> definition = nullptr) noexcept;
};

struct LiteralPayloadNode : ExprNode {
    enum class Kind {
        Set,
        Interval,
    };

    Kind payload_kind;
    std::vector<std::shared_ptr<ExprNode>> elements;
    bool lower_closed;
    bool upper_closed;

    explicit LiteralPayloadNode(size_t line, size_t col,
                                Kind payload_kind,
                                std::vector<std::shared_ptr<ExprNode>> elements,
                                bool lower_closed = false,
                                bool upper_closed = false) noexcept;
};

struct AdtConstructorDecl {
    std::string name;
    std::vector<std::shared_ptr<Type>> fields;
};

struct TypeDeclNode : StmtNode {
    std::string name;
    std::string qualified_name;
    std::vector<std::string> type_params;
    std::vector<AdtConstructorDecl> constructors;

    TypeDeclNode(size_t line, size_t col,
                 std::string name,
                 std::vector<std::string> type_params,
                 std::vector<AdtConstructorDecl> constructors) noexcept;
};

struct Pattern {
    enum class Kind { Wildcard, Binding, Literal, Constructor };

    Kind kind;
    size_t line;
    size_t col;
    std::string name;
    std::shared_ptr<LiteralNode> literal;
    std::vector<Pattern> fields;
    std::string adt_type_name;

    Pattern(Kind kind, size_t line, size_t col, std::string name = {}) noexcept;
};

struct MatchArm {
    Pattern pattern;
    std::shared_ptr<ExprNode> guard;
    std::shared_ptr<ExprNode> value;
};

struct MatchExprNode : ExprNode {
    std::shared_ptr<ExprNode> target;
    std::vector<MatchArm> arms;

    MatchExprNode(size_t line, size_t col,
                  std::shared_ptr<ExprNode> target,
                  std::vector<MatchArm> arms) noexcept;
};
struct DotExprNode : ExprNode {
    std::shared_ptr<ExprNode> expr;
    std::shared_ptr<IdentifierNode> rhs;
    bool is_zero_adt_constructor{false};
    std::string adt_type_name;
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

struct SymDeclNode : StmtNode {
    std::vector<std::string> ids;

    explicit SymDeclNode(size_t line, size_t col, std::vector<std::string> ids) noexcept;
};

struct PipeExprNode : ExprNode {
    std::shared_ptr<ExprNode> lhs;
    std::shared_ptr<ExprNode> rhs;

    explicit PipeExprNode(const size_t line, const size_t col, std::shared_ptr<ExprNode> lhs, std::shared_ptr<ExprNode> rhs) noexcept
        : ExprNode(ASTKind::PipeExpr, line, col), lhs(std::move(lhs)), rhs(std::move(rhs)) {};
};

struct ArrayLiteralNode : ExprNode {
    std::vector<std::shared_ptr<ExprNode>> exprs;

    explicit ArrayLiteralNode(size_t line, size_t col, decltype(exprs) exprs) noexcept;

    /*
     * 数组本身是否可入常量池，取决于全部元素是否可入池。
     */
    [[nodiscard]] bool is_constant() const noexcept override {
        for (const auto& e : exprs) if (!e->is_constant()) return false;
        return !exprs.empty();
    }
};

struct TupleLiteralNode : ExprNode {
    std::vector<std::shared_ptr<ExprNode>> exprs;

    explicit TupleLiteralNode(const size_t line, const size_t col, decltype(exprs) exprs) noexcept
        : ExprNode(ASTKind::TupleLiteral, line,  col), exprs(std::move(exprs)) {}

    [[nodiscard]] bool is_constant() const noexcept override {
        for (const auto& e : exprs) if (!e->is_constant()) return false;
        return !exprs.empty();
    }
};

struct TupleGetExprNode : ExprNode {
    std::shared_ptr<ExprNode> tup;
    uint8_t i;

    explicit TupleGetExprNode(const size_t line, const size_t col, decltype(tup) tup, const uint8_t i) noexcept
        : ExprNode(ASTKind::TupleGetExpr, line, col), tup(std::move(tup)), i(i) {}
};

struct Module {
    std::string name;
    std::string lib_name;
    std::vector<std::shared_ptr<StmtNode>> decls;
    std::vector<std::shared_ptr<NativeFuncDeclNode>> native_funcs;
    std::vector<std::shared_ptr<TypeDeclNode>> adt_exports;
    std::vector<std::pair<std::string, UnitDefinition>> unit_exports;

    // key 是展开后的源码绝对路径，value是输出的module编码文件路径
    std::unordered_map<std::string, std::shared_ptr<ModuleType>> imports;


    Module(std::string name, decltype(decls) decls) noexcept;


    /*
     * name必须是绝对路径
     */
    [[nodiscard]] bool module_is_imported(const std::string& other_name) const noexcept {
        return imports.contains(other_name);
    }
};

/*
 * 类型池：保证每个类型只有一个实例（interned）。
 * parse / hir / tyck 阶段统一从这里获取类型，不做裸 make_shared。
 */
class TypePool {
    std::vector<std::shared_ptr<Type>> types;
public:
    [[nodiscard]] std::shared_ptr<Type> basic(runtime::ValueKind v) noexcept;
    [[nodiscard]] std::shared_ptr<Type> dimensioned(UnitSpec syntax) noexcept;
    [[nodiscard]] std::shared_ptr<Type> dimensioned(UnitDefinition unit) noexcept;
    [[nodiscard]] std::shared_ptr<Type> string() noexcept;
    [[nodiscard]] std::shared_ptr<Type> array(const std::shared_ptr<Type>& type) noexcept;
    [[nodiscard]] std::shared_ptr<Type> function(std::vector<std::shared_ptr<Type>> params,
                                                 std::shared_ptr<Type> ret) noexcept;
    [[nodiscard]] std::shared_ptr<Type> native_function(std::vector<std::shared_ptr<Type>> params,
                                                        std::shared_ptr<Type> ret,
                                                        std::string name) noexcept;
    [[nodiscard]] std::shared_ptr<Type> named(std::string name,
                                              std::vector<std::shared_ptr<Type>> args = {}) noexcept;
    [[nodiscard]] std::shared_ptr<Type> nullable(std::shared_ptr<Type> value_type) noexcept;
    [[nodiscard]] std::shared_ptr<Type> adt_constructor(std::string type_name,
                                                        std::string constructor,
                                                        std::vector<std::string> type_params,
                                                        std::vector<std::shared_ptr<Type>> fields) noexcept;
    [[nodiscard]] std::shared_ptr<Type> module(std::string target_path,
                                               std::string load_path,
                                               std::string binding_name,
                                               std::vector<hir::Scope::Var> exports,
                                               std::vector<std::shared_ptr<TypeDeclNode>> adt_exports = {},
                                               std::vector<std::pair<std::string, UnitDefinition>> unit_exports = {}) noexcept;

    [[nodiscard]] std::shared_ptr<Type> unknown() noexcept;
    [[nodiscard]] std::shared_ptr<Type> none() noexcept;

    std::shared_ptr<Type> tuple(std::vector<std::shared_ptr<Type>> t) noexcept;
};

// 编译器前端共享的类型池，parse / hir / tyck 统一使用
extern TypePool type_pool;

}
