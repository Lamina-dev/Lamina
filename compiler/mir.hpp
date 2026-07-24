//
// Created by meian on 2026/7/16.
//

#pragma once
#include <memory>
#include <string>
#include <vector>

#include "ast.hpp"

namespace lmx::runtime::Opcode {
enum Opcode : uint8_t;
}

namespace lmx::mir {

enum class MirNodeKind {
    TempAssign, Assign, Label, Expr, Func
};
struct MirNode {
    MirNodeKind kind;

    explicit MirNode(MirNodeKind kind) noexcept;
};
enum class MirExprKind {
    Ref, Literal, Operate,
};
struct MirExpr {
    MirExprKind kind;
    explicit MirExpr(MirExprKind kind) noexcept;
};
enum class MirLiteralKind {
    Integer, Float, String, Boolean,
};

struct MirLiteralExpr : MirExpr {
    MirLiteralKind literal_kind;
    std::string data;

    explicit MirLiteralExpr(MirLiteralKind kind, std::string data) noexcept;
};
struct MirRefExpr : MirExpr {
    explicit MirRefExpr(std::string name, bool is_temp) noexcept;
    std::string name;
    bool is_temp;
};
struct MirLabel : MirNode {
    std::string name;
    explicit MirLabel(std::string name) noexcept;
};
struct MirFuncDefine : MirNode {
    std::string name;
    std::vector<std::string> params;
    std::vector<std::shared_ptr<MirNode>> body;

    explicit MirFuncDefine(std::string name, std::vector<std::string> params,  std::vector<std::shared_ptr<MirNode>> body) noexcept;
};

struct MirTempAssign : MirNode {
    std::string name;
    std::shared_ptr<MirExpr> expr;
    explicit MirTempAssign(std::string name, std::shared_ptr<MirExpr> expr) noexcept;
};

struct MirExprNode : MirNode {
    std::shared_ptr<MirExpr> expr;
    explicit MirExprNode(std::shared_ptr<MirExpr> expr) noexcept;
};
enum class MirOperateKind {
    Normal, RetVoid,
};

struct MirOperateExpr : MirExpr {
    runtime::Opcode::Opcode opcode;
    MirOperateKind operate_kind;

    explicit MirOperateExpr(runtime::Opcode::Opcode opcode, MirOperateKind kind = MirOperateKind::Normal) noexcept;
};

struct MirNopExpr : MirOperateExpr {
    explicit MirNopExpr() noexcept;
};

struct MirNewExpr : MirOperateExpr {
    std::shared_ptr<MirExpr> expr;
    explicit MirNewExpr(std::shared_ptr<MirExpr> expr) noexcept;
};

struct MirHaltExpr : MirOperateExpr {
    explicit MirHaltExpr() noexcept;
};

struct MirIAddExpr : MirOperateExpr {
    std::shared_ptr<MirExpr> lhs, rhs;
    explicit MirIAddExpr(std::shared_ptr<MirExpr> lhs, std::shared_ptr<MirExpr> rhs) noexcept;
};
struct MirISubExpr : MirOperateExpr {
    std::shared_ptr<MirExpr> lhs, rhs;
    explicit MirISubExpr(std::shared_ptr<MirExpr> lhs, std::shared_ptr<MirExpr> rhs) noexcept;
};
struct MirIMulExpr : MirOperateExpr {
    std::shared_ptr<MirExpr> lhs, rhs;
    explicit MirIMulExpr(std::shared_ptr<MirExpr> lhs, std::shared_ptr<MirExpr> rhs) noexcept;
};
struct MirIDivExpr : MirOperateExpr {
    std::shared_ptr<MirExpr> lhs, rhs;
    explicit MirIDivExpr(std::shared_ptr<MirExpr> lhs, std::shared_ptr<MirExpr> rhs) noexcept;
};

struct MirIModExpr : MirOperateExpr {
    std::shared_ptr<MirExpr> lhs, rhs;
    explicit MirIModExpr(std::shared_ptr<MirExpr> lhs, std::shared_ptr<MirExpr> rhs) noexcept;
};
struct MirIPowExpr : MirOperateExpr {
    std::shared_ptr<MirExpr> lhs, rhs;
    explicit MirIPowExpr(std::shared_ptr<MirExpr> lhs, std::shared_ptr<MirExpr> rhs) noexcept;
};
struct MirINegExpr : MirOperateExpr {
    std::shared_ptr<MirExpr> e;
    explicit MirINegExpr(std::shared_ptr<MirExpr> e) noexcept;
};
struct MirCallVirtualExpr : MirOperateExpr {
    uint8_t reg;
    uint8_t arg_count;
    explicit MirCallVirtualExpr(uint8_t reg, uint8_t arg_count) noexcept;
};
struct MirCallFastExpr : MirOperateExpr {
    std::string name;
    std::vector<std::shared_ptr<MirRefExpr>> args;
    explicit MirCallFastExpr(std::string name, std::vector<std::shared_ptr<MirRefExpr>> args) noexcept;
};
struct MirRetExpr : MirOperateExpr {
    std::shared_ptr<MirExpr> value;
    explicit MirRetExpr(std::shared_ptr<MirExpr> value) noexcept;
};
struct MirRetVoidExpr : MirOperateExpr {
    explicit MirRetVoidExpr() noexcept;
};
struct MirGotoExpr : MirOperateExpr {
    std::string label;
    explicit MirGotoExpr(std::string label) noexcept;
};
struct MirICmpEqExpr : MirOperateExpr {
    std::shared_ptr<MirExpr> lhs, rhs;
    explicit MirICmpEqExpr(std::shared_ptr<MirExpr> lhs, std::shared_ptr<MirExpr> rhs) noexcept;
};
struct MirICmpNeExpr : MirOperateExpr {
    std::shared_ptr<MirExpr> lhs, rhs;
    explicit MirICmpNeExpr(std::shared_ptr<MirExpr> lhs, std::shared_ptr<MirExpr> rhs) noexcept;
};
struct MirICmpLtExpr : MirOperateExpr {
    std::shared_ptr<MirExpr> lhs, rhs;
    explicit MirICmpLtExpr(std::shared_ptr<MirExpr> lhs, std::shared_ptr<MirExpr> rhs) noexcept;
};
struct MirICmpLeExpr : MirOperateExpr {
    std::shared_ptr<MirExpr> lhs, rhs;
    explicit MirICmpLeExpr(std::shared_ptr<MirExpr> lhs, std::shared_ptr<MirExpr> rhs) noexcept;
};
struct MirICmpGtExpr : MirOperateExpr {
    std::shared_ptr<MirExpr> lhs, rhs;
    explicit MirICmpGtExpr(std::shared_ptr<MirExpr> lhs, std::shared_ptr<MirExpr> rhs) noexcept;
};
struct MirICmpGeExpr : MirOperateExpr {
    std::shared_ptr<MirExpr> lhs, rhs;
    explicit MirICmpGeExpr(std::shared_ptr<MirExpr> lhs, std::shared_ptr<MirExpr> rhs) noexcept;
};
struct MirCmpAndExpr : MirOperateExpr {
    std::shared_ptr<MirExpr> lhs, rhs;
    explicit MirCmpAndExpr(std::shared_ptr<MirExpr> lhs, std::shared_ptr<MirExpr> rhs) noexcept;
};
struct MirCmpOrExpr : MirOperateExpr {
    std::shared_ptr<MirExpr> lhs, rhs;
    explicit MirCmpOrExpr(std::shared_ptr<MirExpr> lhs, std::shared_ptr<MirExpr> rhs) noexcept;
};
struct MirIfTrueExpr : MirOperateExpr {
    std::shared_ptr<MirExpr> cond;
    std::string label;
    explicit MirIfTrueExpr(std::shared_ptr<MirExpr> cond, std::string label) noexcept;
};
struct MirIfFalseExpr : MirOperateExpr {
    std::shared_ptr<MirExpr> cond;
    std::string label;
    explicit MirIfFalseExpr(std::shared_ptr<MirExpr> cond, std::string label) noexcept;
};
struct MirFAddExpr : MirOperateExpr {
    std::shared_ptr<MirExpr> lhs, rhs;
    explicit MirFAddExpr(std::shared_ptr<MirExpr> lhs, std::shared_ptr<MirExpr> rhs) noexcept;
};
struct MirFSubExpr : MirOperateExpr {
    std::shared_ptr<MirExpr> lhs, rhs;
    explicit MirFSubExpr(std::shared_ptr<MirExpr> lhs, std::shared_ptr<MirExpr> rhs) noexcept;
};
struct MirFMulExpr : MirOperateExpr {
    std::shared_ptr<MirExpr> lhs, rhs;
    explicit MirFMulExpr(std::shared_ptr<MirExpr> lhs, std::shared_ptr<MirExpr> rhs) noexcept;
};
struct MirFDivExpr : MirOperateExpr {
    std::shared_ptr<MirExpr> lhs, rhs;
    explicit MirFDivExpr(std::shared_ptr<MirExpr> lhs, std::shared_ptr<MirExpr> rhs) noexcept;
};
struct MirFModExpr : MirOperateExpr {
    std::shared_ptr<MirExpr> lhs, rhs;
    explicit MirFModExpr(std::shared_ptr<MirExpr> lhs, std::shared_ptr<MirExpr> rhs) noexcept;
};
struct MirFNegExpr : MirOperateExpr {
    std::shared_ptr<MirExpr> e;
    explicit MirFNegExpr(std::shared_ptr<MirExpr> e) noexcept;
};

struct MirAssign : MirNode {
    std::string name;
    std::shared_ptr<MirExpr> expr;
    explicit MirAssign(std::string name, std::shared_ptr<MirExpr> expr) noexcept;
};
struct MirModule {
    std::vector<std::shared_ptr<MirNode>> nodes;
};
}