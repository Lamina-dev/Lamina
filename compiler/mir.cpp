//
// Created by meian on 2026/7/16.
//

#include "mir.hpp"
#include <utility>

#include "../runtime/opcode.hpp"

namespace lmx::mir {

MirNode::MirNode(const MirNodeKind kind) noexcept : kind(kind) {}
MirExpr::MirExpr(const MirExprKind kind) noexcept : kind(kind) {}
MirLiteralExpr::MirLiteralExpr(MirLiteralKind kind, std::string data) noexcept
    : MirExpr(MirExprKind::Literal), literal_kind(kind), data(std::move(data)) {}

MirRefExpr::MirRefExpr(std::string name, bool is_temp) noexcept : MirExpr(MirExprKind::Ref), name(std::move(name)), is_temp(is_temp) {}
MirTempAssign::MirTempAssign(std::string name, std::shared_ptr<MirExpr> expr) noexcept
    : MirNode(MirNodeKind::TempAssign), name(std::move(name)), expr(std::move(expr)) {}
MirExprNode::MirExprNode(std::shared_ptr<MirExpr> expr) noexcept : MirNode(MirNodeKind::Expr), expr(std::move(expr)) {}
MirAssign::MirAssign(std::string name, std::shared_ptr<MirExpr> expr) noexcept
    : MirNode(MirNodeKind::Assign), name(std::move(name)), expr(std::move(expr)) {}

MirFuncDefine::MirFuncDefine(std::string name, std::vector<std::string> params, std::vector<std::shared_ptr<MirNode> > body) noexcept :
    MirNode(MirNodeKind::Func), name(std::move(name)), params(std::move(params)), body(std::move(body)) {}

MirOperateExpr::MirOperateExpr(const runtime::Opcode::Opcode opcode, MirOperateKind kind) noexcept
    : MirExpr(MirExprKind::Operate), opcode(opcode), operate_kind(kind) {}

MirNopExpr::MirNopExpr() noexcept : MirOperateExpr(runtime::Opcode::Opcode::Nop) {}

MirNewExpr::MirNewExpr(std::shared_ptr<MirExpr> expr) noexcept
    : MirOperateExpr(runtime::Opcode::Opcode::New), expr(std::move(expr)) {}

MirHaltExpr::MirHaltExpr() noexcept : MirOperateExpr(runtime::Opcode::Opcode::Halt) {}

MirIAddExpr::MirIAddExpr(std::shared_ptr<MirExpr> lhs, std::shared_ptr<MirExpr> rhs) noexcept
   : MirOperateExpr(runtime::Opcode::Opcode::IAdd), lhs(std::move(lhs)), rhs(std::move(rhs)) {}

MirISubExpr::MirISubExpr(std::shared_ptr<MirExpr> lhs, std::shared_ptr<MirExpr> rhs) noexcept
   : MirOperateExpr(runtime::Opcode::Opcode::ISub), lhs(std::move(lhs)), rhs(std::move(rhs)) {}

MirIMulExpr::MirIMulExpr(std::shared_ptr<MirExpr> lhs, std::shared_ptr<MirExpr> rhs) noexcept
   : MirOperateExpr(runtime::Opcode::Opcode::IMul), lhs(std::move(lhs)), rhs(std::move(rhs)) {}

MirIDivExpr::MirIDivExpr(std::shared_ptr<MirExpr> lhs, std::shared_ptr<MirExpr> rhs) noexcept
    : MirOperateExpr(runtime::Opcode::Opcode::IDiv), lhs(std::move(lhs)), rhs(std::move(rhs)) {}

MirIModExpr::MirIModExpr(std::shared_ptr<MirExpr> lhs, std::shared_ptr<MirExpr> rhs) noexcept
   : MirOperateExpr(runtime::Opcode::Opcode::IMod), lhs(std::move(lhs)), rhs(std::move(rhs)) {}

MirIPowExpr::MirIPowExpr(std::shared_ptr<MirExpr> lhs, std::shared_ptr<MirExpr> rhs) noexcept
    : MirOperateExpr(runtime::Opcode::Opcode::IPow), lhs(std::move(lhs)), rhs(std::move(rhs)) {}

MirINegExpr::MirINegExpr(std::shared_ptr<MirExpr> e) noexcept
    : MirOperateExpr(runtime::Opcode::Opcode::INeg), e(std::move(e)) {}

MirCallVirtualExpr::MirCallVirtualExpr(uint8_t reg, uint8_t arg_count) noexcept
    : MirOperateExpr(runtime::Opcode::Opcode::CallVirtual), reg(reg), arg_count(arg_count) {}

MirCallFastExpr::MirCallFastExpr(std::string name, std::vector<std::shared_ptr<MirRefExpr>> args) noexcept
    : MirOperateExpr(runtime::Opcode::Opcode::CallFast), name(std::move(name)), args(std::move(args)) {}

MirRetExpr::MirRetExpr(std::shared_ptr<MirExpr> value) noexcept
    : MirOperateExpr(runtime::Opcode::Opcode::Ret), value(std::move(value)) {}
MirRetVoidExpr::MirRetVoidExpr() noexcept
    : MirOperateExpr(runtime::Opcode::Opcode::Nop, MirOperateKind::RetVoid) {}

MirGotoExpr::MirGotoExpr(std::string label) noexcept
    : MirOperateExpr(runtime::Opcode::Opcode::Goto), label(std::move(label)) {}

MirICmpEqExpr::MirICmpEqExpr(std::shared_ptr<MirExpr> lhs, std::shared_ptr<MirExpr> rhs) noexcept
    : MirOperateExpr(runtime::Opcode::Opcode::ICmpEq), lhs(std::move(lhs)), rhs(std::move(rhs)) {}

MirICmpNeExpr::MirICmpNeExpr(std::shared_ptr<MirExpr> lhs, std::shared_ptr<MirExpr> rhs) noexcept
    : MirOperateExpr(runtime::Opcode::Opcode::ICmpNe), lhs(std::move(lhs)), rhs(std::move(rhs)) {}

MirICmpLtExpr::MirICmpLtExpr(std::shared_ptr<MirExpr> lhs, std::shared_ptr<MirExpr> rhs) noexcept
    : MirOperateExpr(runtime::Opcode::Opcode::ICmpLt), lhs(std::move(lhs)), rhs(std::move(rhs)) {}

MirICmpLeExpr::MirICmpLeExpr(std::shared_ptr<MirExpr> lhs, std::shared_ptr<MirExpr> rhs) noexcept
    : MirOperateExpr(runtime::Opcode::Opcode::ICmpLe), lhs(std::move(lhs)), rhs(std::move(rhs)) {}

MirICmpGtExpr::MirICmpGtExpr(std::shared_ptr<MirExpr> lhs, std::shared_ptr<MirExpr> rhs) noexcept
    : MirOperateExpr(runtime::Opcode::Opcode::ICmpGt), lhs(std::move(lhs)), rhs(std::move(rhs)) {}

MirICmpGeExpr::MirICmpGeExpr(std::shared_ptr<MirExpr> lhs, std::shared_ptr<MirExpr> rhs) noexcept
    : MirOperateExpr(runtime::Opcode::Opcode::ICmpGe), lhs(std::move(lhs)), rhs(std::move(rhs)) {}

MirLabel::MirLabel(std::string name) noexcept
    : MirNode(MirNodeKind::Label), name(std::move(name)) {}

MirIfTrueExpr::MirIfTrueExpr(std::shared_ptr<MirExpr> cond, std::string label) noexcept
    : MirOperateExpr(runtime::Opcode::Opcode::IfTrue), cond(std::move(cond)), label(std::move(label)) {}

MirIfFalseExpr::MirIfFalseExpr(std::shared_ptr<MirExpr> cond, std::string label) noexcept
    : MirOperateExpr(runtime::Opcode::Opcode::IfFalse), cond(std::move(cond)), label(std::move(label)) {}

MirFAddExpr::MirFAddExpr(std::shared_ptr<MirExpr> lhs, std::shared_ptr<MirExpr> rhs) noexcept
    : MirOperateExpr(runtime::Opcode::Opcode::FAdd), lhs(std::move(lhs)), rhs(std::move(rhs)) {}

MirFSubExpr::MirFSubExpr(std::shared_ptr<MirExpr> lhs, std::shared_ptr<MirExpr> rhs) noexcept
    : MirOperateExpr(runtime::Opcode::Opcode::FSub), lhs(std::move(lhs)), rhs(std::move(rhs)) {}

MirFMulExpr::MirFMulExpr(std::shared_ptr<MirExpr> lhs, std::shared_ptr<MirExpr> rhs) noexcept
    : MirOperateExpr(runtime::Opcode::Opcode::FMul), lhs(std::move(lhs)), rhs(std::move(rhs)) {}

MirFDivExpr::MirFDivExpr(std::shared_ptr<MirExpr> lhs, std::shared_ptr<MirExpr> rhs) noexcept
    : MirOperateExpr(runtime::Opcode::Opcode::FDiv), lhs(std::move(lhs)), rhs(std::move(rhs)) {}

MirFModExpr::MirFModExpr(std::shared_ptr<MirExpr> lhs, std::shared_ptr<MirExpr> rhs) noexcept
    : MirOperateExpr(runtime::Opcode::Opcode::FMod), lhs(std::move(lhs)), rhs(std::move(rhs)) {}

MirFNegExpr::MirFNegExpr(std::shared_ptr<MirExpr> e) noexcept
    : MirOperateExpr(runtime::Opcode::Opcode::FNeg), e(std::move(e)) {}

MirCmpAndExpr::MirCmpAndExpr(std::shared_ptr<MirExpr> lhs, std::shared_ptr<MirExpr> rhs) noexcept
    : MirOperateExpr(runtime::Opcode::Opcode::And), lhs(std::move(lhs)), rhs(std::move(rhs)) {}
MirCmpOrExpr::MirCmpOrExpr(std::shared_ptr<MirExpr> lhs, std::shared_ptr<MirExpr> rhs) noexcept
    : MirOperateExpr(runtime::Opcode::Opcode::Or), lhs(std::move(lhs)), rhs(std::move(rhs)) {}
}
