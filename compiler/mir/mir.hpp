//
// Created by meian on 2026/7/16.
//

#pragma once
#include <memory>
#include <string>
#include <vector>

#include "../ast/ast.hpp"
#include "../../runtime/opcode.hpp"

namespace lmx::runtime::Opcode {
enum Opcode : uint8_t;
}

namespace lmx::mir {

enum class MirNodeKind {
    TempAssign, Assign, ArrStore, TupleStore, Label, Expr, Func, NativeFunc
};
struct MirNode {
    MirNodeKind kind;

    explicit MirNode(MirNodeKind kind) noexcept;
};
enum class MirExprKind {
    Ref, Literal, Operate, Array, Tuple
};
struct MirExpr {
    MirExprKind kind;
    explicit MirExpr(MirExprKind kind) noexcept;
};
enum class MirLiteralKind {
    Integer, Float, String, Boolean, Null,
};

struct MirLiteralExpr : MirExpr {
    MirLiteralKind literal_kind;
    std::string data;

    explicit MirLiteralExpr(MirLiteralKind kind, std::string data) noexcept;
};

/*
 * 数组构造。is_constant 为 true 表示元素均为可入池字面量
 * (Int/Frac/Str)，汇编时可整体编码进常量池；否则用
 * NewArray + ArrStore 指令逐元素构造。
 */
struct MirArrayExpr : MirExpr {
    bool is_constant;
    std::vector<std::shared_ptr<MirExpr>> elements;

    explicit MirArrayExpr(bool is_constant, std::vector<std::shared_ptr<MirExpr>> elements) noexcept;
};

struct MirTupleExpr : MirExpr {
    bool is_constant;
    std::vector<std::shared_ptr<MirExpr>> elements;

    explicit MirTupleExpr(const bool is_constant, std::vector<std::shared_ptr<MirExpr>> elements) noexcept
        : MirExpr(MirExprKind::Tuple), is_constant(is_constant), elements(std::move(elements)) {}
};

struct MirRefExpr : MirExpr {
    explicit MirRefExpr(std::string name, bool is_temp) noexcept;
    bool is_temp;
    std::string name;
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

struct MirNativeFuncDefine : MirNode {
    std::string name;
    std::string symbol;
    std::vector<runtime::ValueKind> params;
    runtime::ValueKind ret_ty;

    explicit MirNativeFuncDefine(std::string name, std::string symbol, std::vector<runtime::ValueKind> params, runtime::ValueKind ret_ty) noexcept;
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
struct MirCallExpr : MirOperateExpr {
    std::shared_ptr<MirRefExpr> func;
    std::vector<std::shared_ptr<MirRefExpr>> args;
    explicit MirCallExpr(std::shared_ptr<MirRefExpr> func,
    std::vector<std::shared_ptr<MirRefExpr>> args) noexcept;
};
struct MirCallFastExpr : MirOperateExpr {
    std::string name;
    std::vector<std::shared_ptr<MirRefExpr>> args;
    explicit MirCallFastExpr(std::string name, std::vector<std::shared_ptr<MirRefExpr>> args) noexcept;
};

/*
 * calling c function
 */
struct MirCCallExpr : MirOperateExpr {
    std::string name;
    std::vector<std::shared_ptr<MirRefExpr>> args;
    explicit MirCCallExpr(std::string name, std::vector<std::shared_ptr<MirRefExpr>> args) noexcept;
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
struct MirFCmpEqExpr : MirOperateExpr {
    std::shared_ptr<MirExpr> lhs, rhs;
    explicit MirFCmpEqExpr(std::shared_ptr<MirExpr> lhs, std::shared_ptr<MirExpr> rhs) noexcept;
};
struct MirFCmpNeExpr : MirOperateExpr {
    std::shared_ptr<MirExpr> lhs, rhs;
    explicit MirFCmpNeExpr(std::shared_ptr<MirExpr> lhs, std::shared_ptr<MirExpr> rhs) noexcept;
};
struct MirFCmpLtExpr : MirOperateExpr {
    std::shared_ptr<MirExpr> lhs, rhs;
    explicit MirFCmpLtExpr(std::shared_ptr<MirExpr> lhs, std::shared_ptr<MirExpr> rhs) noexcept;
};
struct MirFCmpLeExpr : MirOperateExpr {
    std::shared_ptr<MirExpr> lhs, rhs;
    explicit MirFCmpLeExpr(std::shared_ptr<MirExpr> lhs, std::shared_ptr<MirExpr> rhs) noexcept;
};
struct MirFCmpGtExpr : MirOperateExpr {
    std::shared_ptr<MirExpr> lhs, rhs;
    explicit MirFCmpGtExpr(std::shared_ptr<MirExpr> lhs, std::shared_ptr<MirExpr> rhs) noexcept;
};
struct MirFCmpGeExpr : MirOperateExpr {
    std::shared_ptr<MirExpr> lhs, rhs;
    explicit MirFCmpGeExpr(std::shared_ptr<MirExpr> lhs, std::shared_ptr<MirExpr> rhs) noexcept;
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
struct MirArrLoadExpr : MirOperateExpr {
    std::shared_ptr<MirExpr> target;
    std::shared_ptr<MirExpr> index;
    explicit MirArrLoadExpr(std::shared_ptr<MirExpr> target, std::shared_ptr<MirExpr> index) noexcept;
};
struct MirTupleGetExpr : MirOperateExpr {
    std::shared_ptr<MirExpr> target;
    uint8_t index;
    explicit MirTupleGetExpr(std::shared_ptr<MirExpr> target, uint8_t index) noexcept;
};
struct MirArrStore : MirNode {
    std::shared_ptr<MirExpr> target;
    std::shared_ptr<MirExpr> index;
    std::shared_ptr<MirExpr> value;
    explicit MirArrStore(std::shared_ptr<MirExpr> target, std::shared_ptr<MirExpr> index, std::shared_ptr<MirExpr> value) noexcept;
};
struct MirTupleStore : MirNode {
    std::shared_ptr<MirExpr> target;
    uint8_t index;
    std::shared_ptr<MirExpr> value;
    explicit MirTupleStore(std::shared_ptr<MirExpr> target, uint8_t index, std::shared_ptr<MirExpr> value) noexcept;
};
struct MirGetModuleExpr : MirOperateExpr {
    std::string name;
    explicit MirGetModuleExpr(std::string name) noexcept;
};
struct MirGetModuleAttrExpr : MirOperateExpr {
    std::shared_ptr<MirRefExpr> mod;
    std::string mod_name;
    std::string name;
    explicit MirGetModuleAttrExpr(std::shared_ptr<MirRefExpr> mod, std::string mod_name, std::string name) noexcept
        : MirOperateExpr(runtime::Opcode::Opcode::GetModuleAttr), mod(std::move(mod)),
          mod_name(std::move(mod_name)), name(std::move(name)) {}
};

struct MirAdtNewExpr : MirOperateExpr {
    std::string type_name;
    std::string constructor;
    std::vector<std::shared_ptr<MirRefExpr>> fields;
    MirAdtNewExpr(std::string type_name, std::string constructor,
                  std::vector<std::shared_ptr<MirRefExpr>> fields) noexcept;
};

struct MirAdtIsExpr : MirOperateExpr {
    std::shared_ptr<MirExpr> value;
    std::string type_name;
    std::string constructor;
    MirAdtIsExpr(std::shared_ptr<MirExpr> value,
                 std::string type_name,
                 std::string constructor) noexcept;
};

struct MirAdtGetExpr : MirOperateExpr {
    std::shared_ptr<MirExpr> value;
    uint8_t index;
    MirAdtGetExpr(std::shared_ptr<MirExpr> value, uint8_t index) noexcept;
};

struct MirLiteralNewExpr : MirOperateExpr {
    LiteralPayloadNode::Kind literal_kind;
    std::vector<std::shared_ptr<MirRefExpr>> elements;
    bool lower_closed;
    bool upper_closed;
    MirLiteralNewExpr(LiteralPayloadNode::Kind literal_kind,
                      std::vector<std::shared_ptr<MirRefExpr>> elements,
                      bool lower_closed = false,
                      bool upper_closed = false) noexcept;
};

struct MirContainsExpr : MirOperateExpr {
    std::shared_ptr<MirExpr> element;
    std::shared_ptr<MirExpr> container;
    MirContainsExpr(std::shared_ptr<MirExpr> element,
                    std::shared_ptr<MirExpr> container,
                    bool negate) noexcept;
};

struct MirModule {
    std::string lib_name{};
    std::vector<std::shared_ptr<MirNode>> nodes;
    std::unordered_map<std::string, std::shared_ptr<ModuleType>> imports;
};
}
