
#include "assembler.hpp"
#include "lmx.h"

#include <algorithm>
#include <assert.h>
#include <cstring>
#include <ranges>

namespace lmx {
std::vector<uint8_t> RegAllocator::get_all_using() noexcept {
    std::vector<uint8_t> rs;
    rs.reserve(COMMON_REG_COUNT);
    for (uint8_t i = 0; i < COMMON_REG_COUNT; i++) {
        if (regs.test(i)) {
            rs.push_back(i + 1);
        }
    }
    return rs;
}

void InstEmitter::emit(InstSeq& s, const runtime::Opcode::Opcode op, const uint8_t a, const uint8_t b, const uint8_t c) noexcept {
    s.push_back({{static_cast<uint8_t>(op), a, b, c}});
}

void InstEmitter::emit(InstSeq& s, const runtime::Opcode::Opcode op, const uint8_t a, const uint8_t b) noexcept {
    s.push_back({{static_cast<uint8_t>(op), a, b, 0}});
}

void InstEmitter::emit(InstSeq& s, const runtime::Opcode::Opcode op, const uint16_t a, const uint8_t b) noexcept {
    s.push_back({{
        static_cast<uint8_t>(op),
        static_cast<uint8_t>(a & 0xFF),
        static_cast<uint8_t>((a >> 8) & 0xFF),
        b
    }});
}

void InstEmitter::emit(InstSeq& s, const runtime::Opcode::Opcode op, const uint8_t a, const uint16_t b) noexcept {
    s.push_back({{
        static_cast<uint8_t>(op),
        a,
        static_cast<uint8_t>(b & 0xFF),
        static_cast<uint8_t>((b >> 8) & 0xFF)
    }});
}

void InstEmitter::emit(InstSeq& s, const runtime::Opcode::Opcode op, const uint8_t a) noexcept {
    s.push_back({{static_cast<uint8_t>(op), a, 0, 0}});
}

void InstEmitter::emit(InstSeq &s, runtime::Opcode::Opcode op) noexcept {
    s.push_back({{static_cast<uint8_t>(op), 0, 0, 0}});
}

bool InstEmitter::inst_is_ret_reg(const runtime::Opcode::Opcode op) noexcept {
    switch (op) {
    case runtime::Opcode::Nop:
    case runtime::Opcode::Halt:
    case runtime::Opcode::CCall:
    case runtime::Opcode::CallFast:
    case runtime::Opcode::Ret:
    case runtime::Opcode::Goto:
    case runtime::Opcode::IfTrue:
    case runtime::Opcode::IfFalse:
    case runtime::Opcode::Call:
    case runtime::Opcode::TupleSet:
        return false;
    case runtime::Opcode::Raise:
        return false;
    case runtime::Opcode::FuncCreate:
    case runtime::Opcode::New:
    case runtime::Opcode::GetTrue:
    case runtime::Opcode::GetFalse:
    case runtime::Opcode::GetNull:
    case runtime::Opcode::IAdd:
    case runtime::Opcode::ISub:
    case runtime::Opcode::IMul:
    case runtime::Opcode::IDiv:
    case runtime::Opcode::IMod:
    case runtime::Opcode::IPow:
    case runtime::Opcode::INeg:
    case runtime::Opcode::IConst:
    case runtime::Opcode::NewTuple:
    case runtime::Opcode::ICmpEq:
    case runtime::Opcode::ICmpNe:
    case runtime::Opcode::ICmpLt:
    case runtime::Opcode::ICmpLe:
    case runtime::Opcode::ICmpGt:
    case runtime::Opcode::ICmpGe:
    case runtime::Opcode::LGet:
    case runtime::Opcode::LSet:
    case runtime::Opcode::GGet:
    case runtime::Opcode::GSet:
    case runtime::Opcode::FAdd:
    case runtime::Opcode::FSub:
    case runtime::Opcode::FMul:
    case runtime::Opcode::FDiv:
    case runtime::Opcode::FMod:
    case runtime::Opcode::FNeg:
    case runtime::Opcode::MovRR:
    case runtime::Opcode::And:
    case runtime::Opcode::Or:
    case runtime::Opcode::FCmpEq:
    case runtime::Opcode::FCmpNe:
    case runtime::Opcode::FCmpLt:
    case runtime::Opcode::FCmpLe:
    case runtime::Opcode::FCmpGt:
    case runtime::Opcode::FCmpGe:
    case runtime::Opcode::GetModule:
    case runtime::Opcode::GetModuleAttr:
    case runtime::Opcode::GetFunc:
    case runtime::Opcode::NewArray:
    case runtime::Opcode::ArrLoad:
    case runtime::Opcode::ArrStore:
    case runtime::Opcode::TupleGet:
    case runtime::Opcode::AdtNew:
    case runtime::Opcode::AdtIs:
    case runtime::Opcode::AdtGet:
    case runtime::Opcode::LiteralNew:
    case runtime::Opcode::Contains:
    case runtime::Opcode::NotContains:
    case runtime::Opcode::SetUnion:
    case runtime::Opcode::SetIntersection:
    case runtime::Opcode::SetDifference:
    case runtime::Opcode::SetSymmetricDifference:
    case runtime::Opcode::SetSubset:
        return true;

    }
    return false;
}

bool InstEmitter::inst_is_call(const runtime::Opcode::Opcode op) noexcept {
    switch (op) {
    case runtime::Opcode::CCall:
    case runtime::Opcode::CallFast:
    case runtime::Opcode::Call:
        return true;
    default:
        return false;
    }
}

std::optional<uint8_t> RegAllocator::alloc() noexcept {
    for (size_t i = 0; i < COMMON_REG_COUNT; i++) {
        if (!regs[i]) {
            regs.set(i);
            return static_cast<uint8_t>(i + 1);
        }
    }
    return std::nullopt;
}

void RegAllocator::free(const uint8_t reg) noexcept {
    if (reg == 0) return;
    regs.reset(reg - 1);
}


Assembler::Val::Val(const uint8_t reg, const bool is_tmp) noexcept
    : kind(Kind::Reg), is_tmp(is_tmp), reg(reg) {}

Assembler::Val::Val(const uint8_t var) noexcept
    : kind(Kind::Var), is_tmp(false), var(var) {}


void Assembler::write_u32(std::vector<uint8_t>& buf, const uint32_t value) noexcept {
    buf.push_back(static_cast<uint8_t>(value & 0xFF));
    buf.push_back(static_cast<uint8_t>((value >> 8) & 0xFF));
    buf.push_back(static_cast<uint8_t>((value >> 16) & 0xFF));
    buf.push_back(static_cast<uint8_t>((value >> 24) & 0xFF));
}

void Assembler::write_n(std::vector<uint8_t>& buf, const uint8_t* src, const size_t n) noexcept {
    buf.insert(buf.end(), src, src + n);
}

void Assembler::write_u64(std::vector<uint8_t>& buf, const uint64_t value) noexcept {
    buf.push_back(static_cast<uint8_t>(value & 0xFF));
    buf.push_back(static_cast<uint8_t>((value >> 8) & 0xFF));
    buf.push_back(static_cast<uint8_t>((value >> 16) & 0xFF));
    buf.push_back(static_cast<uint8_t>((value >> 24) & 0xFF));
    buf.push_back(static_cast<uint8_t>((value >> 32) & 0xFF));
    buf.push_back(static_cast<uint8_t>((value >> 40) & 0xFF));
    buf.push_back(static_cast<uint8_t>((value >> 48) & 0xFF));
    buf.push_back(static_cast<uint8_t>((value >> 56) & 0xFF));
}

std::optional<Assembler::Val*> Assembler::find_var(const std::string& name) noexcept {
    const auto it = vals.find(name);
    if (it != vals.end()) return &it->second;
    return std::nullopt;
}


uint16_t Assembler::write_cp_frac(const int32_t num, const int32_t frac) {
    using namespace lmx::runtime;
    cp.push_back(static_cast<uint8_t>(ConstantId::Frac));
    write_n(cp, reinterpret_cast<const uint8_t *>(&num), sizeof(num));
    write_n(cp, reinterpret_cast<const uint8_t *>(&frac), sizeof(frac));
    return cp_cnt++;
}

uint16_t Assembler::write_cp_i64(const int64_t num) {
    using namespace lmx::runtime;
    cp.push_back(static_cast<uint8_t>(ConstantId::Int));
    write_n(cp, reinterpret_cast<const uint8_t *>(&num), sizeof(num));
    return cp_cnt++;
}

uint16_t Assembler::write_cp_str(const uint32_t len, const std::string& str) {
    using namespace lmx::runtime;
    cp.push_back(static_cast<uint8_t>(ConstantId::Str));
    write_u32(cp, len);
    write_n(cp, reinterpret_cast<const uint8_t *>(str.c_str()), str.size());
    return cp_cnt++;
}

uint16_t Assembler::write_cp_adt_constructor(const std::string& type_name,
                                              const std::string& constructor,
                                              const uint8_t field_count) {
    using namespace lmx::runtime;
    const auto result = cp_cnt++;
    cp.push_back(static_cast<uint8_t>(ConstantId::AdtConstructor));
    const auto type_len = static_cast<uint16_t>(type_name.size());
    const auto constructor_len = static_cast<uint16_t>(constructor.size());
    write_n(cp, reinterpret_cast<const uint8_t*>(&type_len), sizeof(type_len));
    write_n(cp, reinterpret_cast<const uint8_t*>(&constructor_len), sizeof(constructor_len));
    cp.push_back(field_count);
    write_n(cp, reinterpret_cast<const uint8_t*>(type_name.data()), type_name.size());
    write_n(cp, reinterpret_cast<const uint8_t*>(constructor.data()), constructor.size());
    return result;
}

uint16_t Assembler::write_cp_arr(const uint8_t elem_tag,
                                 const std::vector<std::vector<uint8_t>>& elems) {
    using namespace lmx::runtime;
    cp.push_back(static_cast<uint8_t>(ConstantId::Arr));
    write_u32(cp, static_cast<uint32_t>(elems.size()));
    const bool inline_val = elem_tag == static_cast<uint8_t>(ConstantId::Int);
    for (const auto& element : elems) {
        cp.push_back(elem_tag);
        if (inline_val) {
            cp.insert(cp.end(), element.begin(), element.end());
        } else {
            cp.insert(cp.end(), sizeof(uint64_t), 0);
        }
    }
    if (!inline_val) {
        for (const auto& element : elems) {
            cp.insert(cp.end(), element.begin(), element.end());
        }
    }
    return cp_cnt++;
}


uint8_t Assembler::asm_mir_expr(InstEmitter::InstSeq& insts, mir::MirExpr* node) noexcept {
    switch (node->kind) {
    case mir::MirExprKind::Ref: {
        const auto e = reinterpret_cast<mir::MirRefExpr*>(node);
        if (const auto v_opt = find_var(e->name)) {
            const auto& v = *v_opt;
            if (v->kind == Val::Kind::Reg) {
                return v->reg;
            }
            const auto r = *reg.alloc();
            InstEmitter::emit(insts, runtime::Opcode::LGet, r, v->var);
            return r;
        }
        if (const auto v_opt = funcs.find(e->name); v_opt != funcs.end()) {
            const auto func_idx = static_cast<uint16_t>(v_opt->second);
            const auto r = *reg.alloc();
            InstEmitter::emit(insts, runtime::Opcode::GetFunc, r, func_idx);
            return r;
        }
        break;
    }

    case mir::MirExprKind::Literal: {
        const auto e = reinterpret_cast<mir::MirLiteralExpr*>(node);
        switch (e->literal_kind) {
        case mir::MirLiteralKind::Integer: {
            const auto r = *reg.alloc();
            const auto val = static_cast<int64_t>(std::stoi(e->data));
            if (val <= INT16_MAX) {
                const auto v = static_cast<int16_t>(val);
                InstEmitter::emit(insts, runtime::Opcode::IConst, r, std::bit_cast<uint16_t>(v));
            } else {
                const auto idx = write_cp_i64(val);
                InstEmitter::emit(insts, runtime::Opcode::New, r, idx);
            }

            return r;
        }
        case mir::MirLiteralKind::Float: {
            const runtime::Fraction f(e->data);
            const auto r1 = *reg.alloc();
            const auto r2 = *reg.alloc();
            const auto r3 = *reg.alloc();
            InstEmitter::emit(insts, runtime::Opcode::IConst, r1, static_cast<uint16_t>(f.denominator()));
            InstEmitter::emit(insts, runtime::Opcode::IConst, r2, static_cast<uint16_t>(f.numerator()));
            InstEmitter::emit(insts, runtime::Opcode::IDiv, r3, r2, r1);
            reg.free(r1);
            reg.free(r2);
            return r3;
        }
        case mir::MirLiteralKind::Boolean: {
            const auto r = *reg.alloc();
            if (e->data == "true") {
                InstEmitter::emit(insts, runtime::Opcode::GetTrue, r);
            } else {
                InstEmitter::emit(insts, runtime::Opcode::GetFalse, r);
            }
            return r;
        }
        case mir::MirLiteralKind::Null: {
            const auto r = *reg.alloc();
            InstEmitter::emit(insts, runtime::Opcode::GetNull, r);
            return r;
        }
        case mir::MirLiteralKind::String: {
            const auto r = *reg.alloc();
            const auto idx = write_cp_str(static_cast<uint32_t>(e->data.size()), e->data);
            InstEmitter::emit(insts, runtime::Opcode::New, r, idx);
            return r;
        }
        }
        break;
    }
    case mir::MirExprKind::Tuple: {
        const auto& tup = *reinterpret_cast<mir::MirTupleExpr*>(node);

        const auto r = *reg.alloc();
        InstEmitter::emit(insts, runtime::Opcode::NewTuple, r, static_cast<uint8_t>(tup.elements.size()));
        for (uint8_t i = 0; static_cast<size_t>(i) < tup.elements.size(); i++) {
            const auto rv = asm_mir_expr(insts, tup.elements[i].get());
            InstEmitter::emit(insts, runtime::Opcode::TupleSet, r, i, rv);
            reg.free(rv);
        }
        return r;

        break;
    }
    case mir::MirExprKind::Array: {
        const auto& arr = *reinterpret_cast<mir::MirArrayExpr*>(node);

        if (arr.is_constant && !arr.elements.empty()) {
            uint8_t elem_tag = 0;
            bool ok = true;
            std::vector<std::vector<uint8_t>> elems;
            elems.reserve(arr.elements.size());
            for (auto& e : arr.elements) {
                if (e->kind != mir::MirExprKind::Literal) { ok = false; break; }
                const auto& lit = *reinterpret_cast<mir::MirLiteralExpr*>(e.get());
                const bool first = elems.empty();
                uint8_t tag = 0;
                std::vector<uint8_t> d;
                switch (lit.literal_kind) {
                case mir::MirLiteralKind::Integer: {
                    tag = static_cast<uint8_t>(runtime::ConstantId::Int);
                    const auto v = static_cast<int64_t>(std::stoll(lit.data));
                    write_n(d, reinterpret_cast<const uint8_t*>(&v), sizeof(v));
                    break;
                }
                case mir::MirLiteralKind::Float: {
                    tag = static_cast<uint8_t>(runtime::ConstantId::Frac);
                    const runtime::Fraction f(lit.data);
                    const auto numerator = f.numerator();
                    const auto denominator = f.denominator();
                    write_n(d, reinterpret_cast<const uint8_t*>(&numerator), sizeof(numerator));
                    write_n(d, reinterpret_cast<const uint8_t*>(&denominator), sizeof(denominator));
                    break;
                }
                case mir::MirLiteralKind::String: {
                    tag = static_cast<uint8_t>(runtime::ConstantId::Str);
                    write_u32(d, static_cast<uint32_t>(lit.data.size()));
                    write_n(d, reinterpret_cast<const uint8_t*>(lit.data.c_str()), lit.data.size());
                    break;
                }
                default:
                    ok = false;
                    break;
                }
                if (!ok) break;
                if (first) elem_tag = tag;
                else if (elem_tag != tag) { ok = false; break; }
                elems.push_back(std::move(d));
            }
            if (ok) {
                const auto idx = write_cp_arr(elem_tag, elems);
                const auto r = *reg.alloc();
                InstEmitter::emit(insts, runtime::Opcode::New, r, idx);
                return r;
            }
        }

        const auto r = *reg.alloc();
        InstEmitter::emit(insts, runtime::Opcode::NewArray, r, static_cast<uint16_t>(arr.elements.size()));
        for (size_t i = 0; i < arr.elements.size(); i++) {
            const auto rv = asm_mir_expr(insts, arr.elements[i].get());
            const auto ri = *reg.alloc();
            InstEmitter::emit(insts, runtime::Opcode::IConst, ri, static_cast<uint16_t>(i));
            InstEmitter::emit(insts, runtime::Opcode::ArrStore, r, ri, rv);
            reg.free(ri);
            reg.free(rv);
        }
        return r;
    }

    case mir::MirExprKind::Operate: {
        const auto e = reinterpret_cast<mir::MirOperateExpr*>(node);

        if (e->operate_kind == mir::MirOperateKind::RetVoid) {
            InstEmitter::emit(insts, runtime::Opcode::Ret);
            return 0;
        }

        switch (e->opcode) {
        case runtime::Opcode::IAdd:
        case runtime::Opcode::ISub:
        case runtime::Opcode::IMul:
        case runtime::Opcode::IDiv:
        case runtime::Opcode::IMod:
        case runtime::Opcode::IPow:
        case runtime::Opcode::ICmpEq:
        case runtime::Opcode::ICmpNe:
        case runtime::Opcode::ICmpLt:
        case runtime::Opcode::ICmpLe:
        case runtime::Opcode::ICmpGt:
        case runtime::Opcode::ICmpGe:
        case runtime::Opcode::And:
        case runtime::Opcode::Or: {
            const auto& op = *reinterpret_cast<mir::MirIAddExpr*>(node);
            const auto rl = asm_mir_expr(insts, op.lhs.get());
            const auto rr = asm_mir_expr(insts, op.rhs.get());
            const auto rd = *reg.alloc();
            InstEmitter::emit(insts, e->opcode, rd, rl, rr);
            reg.free(rl);
            reg.free(rr);
            return rd;
        }
        case runtime::Opcode::SetUnion:
        case runtime::Opcode::SetIntersection:
        case runtime::Opcode::SetDifference:
        case runtime::Opcode::SetSymmetricDifference:
        case runtime::Opcode::SetSubset: {
            const auto& op = *reinterpret_cast<mir::MirSetBinaryExpr*>(node);
            const auto lhs = asm_mir_expr(insts, op.lhs.get());
            const auto rhs = asm_mir_expr(insts, op.rhs.get());
            const auto result = *reg.alloc();
            InstEmitter::emit(insts, e->opcode, result, lhs, rhs);
            reg.free(lhs);
            reg.free(rhs);
            return result;
        }

        case runtime::Opcode::FAdd:
        case runtime::Opcode::FSub:
        case runtime::Opcode::FMul:
        case runtime::Opcode::FDiv:
        case runtime::Opcode::FMod:
        case runtime::Opcode::FCmpEq:
        case runtime::Opcode::FCmpNe:
        case runtime::Opcode::FCmpLt:
        case runtime::Opcode::FCmpLe:
        case runtime::Opcode::FCmpGt:
        case runtime::Opcode::FCmpGe: {
            const auto& op = *reinterpret_cast<mir::MirFAddExpr*>(node);
            const auto rl = asm_mir_expr(insts, op.lhs.get());
            const auto rr = asm_mir_expr(insts, op.rhs.get());
            const auto rd = *reg.alloc();
            InstEmitter::emit(insts, e->opcode, rd, rl, rr);
            reg.free(rl);
            reg.free(rr);
            return rd;
        }

        case runtime::Opcode::INeg: {
            const auto& op = *reinterpret_cast<mir::MirINegExpr*>(node);
            const auto r = asm_mir_expr(insts, op.e.get());
            const auto rd = *reg.alloc();
            InstEmitter::emit(insts, runtime::Opcode::INeg, rd, r);
            reg.free(r);
            return rd;
        }
        case runtime::Opcode::FNeg: {
            const auto& op = *reinterpret_cast<mir::MirFNegExpr*>(node);
            const auto r = asm_mir_expr(insts, op.e.get());
            const auto rd = *reg.alloc();
            InstEmitter::emit(insts, runtime::Opcode::FNeg, rd, r);
            reg.free(r);
            return rd;
        }

        case runtime::Opcode::Ret: {
            const auto& ret = *reinterpret_cast<mir::MirRetExpr*>(node);
            const auto r = asm_mir_expr(insts, ret.value.get());
            InstEmitter::emit(insts, runtime::Opcode::MovRR, uint8_t{0}, r);
            reg.free(r);
            InstEmitter::emit(insts, runtime::Opcode::Ret);
            return 0;
        }
        case runtime::Opcode::Raise: {
            const auto& raise = *reinterpret_cast<mir::MirRaiseExpr*>(node);
            const auto source = asm_mir_expr(insts, raise.value.get());
            InstEmitter::emit(insts, runtime::Opcode::Raise, source);
            reg.free(source);
            return 0;
        }

        case runtime::Opcode::Goto: {
            const auto& g = *reinterpret_cast<mir::MirGotoExpr*>(node);
            const auto pos = insts.size() * 4;
            InstEmitter::emit(insts, runtime::Opcode::Goto, uint16_t{0}, uint8_t{0});
            pending_fixups.push_back({pos, runtime::Opcode::Goto, g.label});
            return 0;
        }

        case runtime::Opcode::IfTrue: {
            const auto& i = *reinterpret_cast<mir::MirIfTrueExpr*>(node);
            const auto r = asm_mir_expr(insts, i.cond.get());
            const auto pos = insts.size() * 4;
            InstEmitter::emit(insts, runtime::Opcode::IfTrue, r, uint16_t{0});
            reg.free(r);
            pending_fixups.push_back({pos, runtime::Opcode::IfTrue, i.label});
            return 0;
        }
        case runtime::Opcode::IfFalse: {
            const auto& i = *reinterpret_cast<mir::MirIfFalseExpr*>(node);
            const auto r = asm_mir_expr(insts, i.cond.get());
            const auto pos = insts.size() * 4;
            InstEmitter::emit(insts, runtime::Opcode::IfFalse, r, uint16_t{0});
            reg.free(r);
            pending_fixups.emplace_back(pos, runtime::Opcode::IfFalse, i.label);
            return 0;
        }

        case runtime::Opcode::CallFast: {
            const auto& c = *reinterpret_cast<mir::MirCallFastExpr*>(node);


            const auto argc = static_cast<uint8_t>(c.args.size());
            // Evaluate each arg and place in regs[255 - i]
            for (size_t i = 0; i < c.args.size(); ++i) {
                const auto rr = asm_mir_expr(insts, c.args[i].get());
                if (rr != static_cast<uint8_t>(LMX_VM_REG_COUNT - 1 - i)) {
                    InstEmitter::emit(insts, runtime::Opcode::MovRR,
                        static_cast<uint8_t>(LMX_VM_REG_COUNT - 1 - i), rr);
                }
                reg.free(rr);
            }

            const auto func_it = funcs.find(c.name);
            if (func_it == funcs.end()) return 0;
            const auto func_idx = static_cast<uint16_t>(func_it->second);

            InstEmitter::emit(insts, runtime::Opcode::CallFast, func_idx, argc);

            return 0;
        }
        case runtime::Opcode::CCall: {
            const auto& c = *reinterpret_cast<mir::MirCCallExpr*>(node);
            const auto argc = static_cast<uint8_t>(c.args.size());
            // Evaluate each arg and place in regs[255 - i]
            for (size_t i = 0; i < c.args.size(); ++i) {
                const auto rr = asm_mir_expr(insts, c.args[i].get());
                if (rr != static_cast<uint8_t>(LMX_VM_REG_COUNT - 1 - i)) {
                    InstEmitter::emit(insts, runtime::Opcode::MovRR,
                        static_cast<uint8_t>(LMX_VM_REG_COUNT - 1 - i), rr);
                }
                reg.free(rr);
            }

            const auto func_it = native_funcs.find(c.name);
            const auto func_idx = static_cast<uint16_t>(func_it->second);

            InstEmitter::emit(insts, runtime::Opcode::CCall, func_idx, argc);
            return 0;
        }
        case runtime::Opcode::Call: {
            const auto& c = *reinterpret_cast<mir::MirCallExpr*>(node);

            const auto argc = static_cast<uint8_t>(c.args.size());
            // Evaluate each arg and place in regs[255 - i]
            for (size_t i = 0; i < c.args.size(); ++i) {
                const auto rr = asm_mir_expr(insts, c.args[i].get());
                if (rr != static_cast<uint8_t>(LMX_VM_REG_COUNT - 1 - i)) {
                    InstEmitter::emit(insts, runtime::Opcode::MovRR,
                        static_cast<uint8_t>(LMX_VM_REG_COUNT - 1 - i), rr);
                }
                reg.free(rr);
            }

            const auto func_it = asm_mir_expr(insts, c.func.get());

            InstEmitter::emit(insts, runtime::Opcode::Call, func_it, argc);


            reg.free(func_it);
            return 0;
        }

        case runtime::Opcode::Halt: {
            InstEmitter::emit(insts, runtime::Opcode::Halt);
            return 0;
        }

        case runtime::Opcode::New: {
            const auto& n = *reinterpret_cast<mir::MirNewExpr*>(node);
            const auto r = asm_mir_expr(insts, n.expr.get());
            return r;
        }
        case runtime::Opcode::GetModule: {
            const auto& n = *reinterpret_cast<mir::MirGetModuleExpr*>(node);
            const auto r = *reg.alloc();
            const auto it = imports.find(n.name);
            InstEmitter::emit(insts, runtime::Opcode::GetModule, r, static_cast<uint16_t>(it->second.first));
            return r;
        }
        case runtime::Opcode::GetModuleAttr: {
            const auto& n = *reinterpret_cast<mir::MirGetModuleAttrExpr*>(node);
            auto mod_reg = *find_var(n.mod->name);
            if (!mod_reg) return 0;

            const auto it = imports.find(n.mod_name);
            if (it == imports.end()) return 0;
            const auto attr_idx = it->second.second->find_func_idx(n.name);
            if (!attr_idx) return 0;

            const auto r = *reg.alloc();
            InstEmitter::emit(insts, runtime::Opcode::MovRR, static_cast<uint8_t>(0), mod_reg->reg);
            reg.free(mod_reg->reg);
            InstEmitter::emit(insts, runtime::Opcode::GetModuleAttr, r, static_cast<uint16_t>(*attr_idx));
            return r;
            break;
        }
        case runtime::Opcode::AdtNew: {
            const auto& adt = *reinterpret_cast<mir::MirAdtNewExpr*>(node);
            if (adt.fields.size() > UINT8_MAX) return 0;
            for (size_t i = 0; i < adt.fields.size(); ++i) {
                const auto rr = asm_mir_expr(insts, adt.fields[i].get());
                const auto target = static_cast<uint8_t>(LMX_VM_REG_COUNT - 1 - i);
                if (rr != target) InstEmitter::emit(insts, runtime::Opcode::MovRR, target, rr);
                reg.free(rr);
            }
            const auto rd = *reg.alloc();
            const auto idx = write_cp_adt_constructor(adt.type_name, adt.constructor,
                                                      static_cast<uint8_t>(adt.fields.size()));
            InstEmitter::emit(insts, runtime::Opcode::AdtNew, rd, idx);
            return rd;
        }
        case runtime::Opcode::AdtIs: {
            const auto& adt = *reinterpret_cast<mir::MirAdtIsExpr*>(node);
            const auto value = asm_mir_expr(insts, adt.value.get());
            const auto constructor = *reg.alloc();
            const auto tag = adt.type_name + "\x1f" + adt.constructor;
            const auto constructor_idx = write_cp_str(static_cast<uint32_t>(tag.size()), tag);
            InstEmitter::emit(insts, runtime::Opcode::New, constructor, constructor_idx);
            const auto rd = *reg.alloc();
            InstEmitter::emit(insts, runtime::Opcode::AdtIs, rd, value, constructor);
            reg.free(constructor);
            return rd;
        }
        case runtime::Opcode::AdtGet: {
            const auto& adt = *reinterpret_cast<mir::MirAdtGetExpr*>(node);
            const auto value = asm_mir_expr(insts, adt.value.get());
            const auto rd = *reg.alloc();
            InstEmitter::emit(insts, runtime::Opcode::AdtGet, rd, value, adt.index);
            return rd;
        }
        case runtime::Opcode::LiteralNew: {
            const auto& literal = *reinterpret_cast<mir::MirLiteralNewExpr*>(node);
            if (literal.elements.size() > UINT8_MAX) return 0;
            for (size_t i = 0; i < literal.elements.size(); ++i) {
                const auto rr = asm_mir_expr(insts, literal.elements[i].get());
                const auto target = static_cast<uint8_t>(LMX_VM_REG_COUNT - 1 - i);
                if (rr != target) InstEmitter::emit(insts, runtime::Opcode::MovRR, target, rr);
                reg.free(rr);
            }
            const auto rd = *reg.alloc();
            const uint8_t flags =
                (literal.literal_kind == LiteralPayloadNode::Kind::Interval ? 1U : 0U) |
                (literal.lower_closed ? 2U : 0U) |
                (literal.upper_closed ? 4U : 0U);
            InstEmitter::emit(insts, runtime::Opcode::LiteralNew, rd,
                              static_cast<uint8_t>(literal.elements.size()), flags);
            return rd;
        }
        case runtime::Opcode::Contains:
        case runtime::Opcode::NotContains: {
            const auto& membership = *reinterpret_cast<mir::MirContainsExpr*>(node);
            const auto element = asm_mir_expr(insts, membership.element.get());
            const auto container = asm_mir_expr(insts, membership.container.get());
            const auto rd = *reg.alloc();
            InstEmitter::emit(insts, membership.opcode, rd, element, container);
            return rd;
        }
        case runtime::Opcode::ArrLoad: {
            const auto& op = *reinterpret_cast<mir::MirArrLoadExpr*>(node);
            const auto ra = asm_mir_expr(insts, op.target.get());
            const auto ri = asm_mir_expr(insts, op.index.get());
            const auto rd = *reg.alloc();
            InstEmitter::emit(insts, runtime::Opcode::ArrLoad, rd, ra, ri);
            reg.free(ra);
            reg.free(ri);
            return rd;
        }
        case runtime::Opcode::TupleGet: {
            const auto& op = *reinterpret_cast<mir::MirTupleGetExpr*>(node);
            const auto rt = asm_mir_expr(insts, op.target.get());
            const auto rd = *reg.alloc();
            InstEmitter::emit(insts, runtime::Opcode::TupleGet, rd, rt, op.index);
            reg.free(rt);
            return rd;
        }
        default:
            return 0;
        }
        break;
    }
    }
    return 0;
}


void Assembler::asm_mir_node(InstEmitter::InstSeq& result, mir::MirNode* node) noexcept {
    switch (node->kind) {
    case mir::MirNodeKind::Label: {
        const auto n = reinterpret_cast<mir::MirLabel*>(node);
        label_positions[n->name] = result.size() * 4;
        break;
    }

    case mir::MirNodeKind::TempAssign: {
        const auto n = reinterpret_cast<mir::MirTempAssign*>(node);
        auto r = asm_mir_expr(result, n->expr.get());
        if (const auto found = find_var(n->name);
            found.has_value()) {
            if ((*found)->reg != r) {
                InstEmitter::emit(result, runtime::Opcode::MovRR, (*found)->reg, r);
                reg.free(r);
            }
        } else {
            if (r == 0) {
                r = *reg.alloc();
                InstEmitter::emit(result, runtime::Opcode::MovRR, r, uint8_t{0});
            }
            vals[n->name] = Val(r, true);
        }
        break;
    }

    case mir::MirNodeKind::Assign: {
        const auto n = reinterpret_cast<mir::MirAssign*>(node);
        const auto r = asm_mir_expr(result, n->expr.get());

        const auto v_opt = find_var(n->name);
        if (!v_opt) {
            const auto var_idx = next_local_var++;
            vals[n->name] = Val(var_idx);
            InstEmitter::emit(result, runtime::Opcode::LSet, r, var_idx);
        } else {
            const auto& v = *v_opt;
            InstEmitter::emit(result, runtime::Opcode::LSet, r, v->var);
        }
        reg.free(r);
        break;
    }

    case mir::MirNodeKind::ArrStore: {
        const auto n = reinterpret_cast<mir::MirArrStore*>(node);
        const auto ra = asm_mir_expr(result, n->target.get());
        const auto ri = asm_mir_expr(result, n->index.get());
        const auto rv = asm_mir_expr(result, n->value.get());
        InstEmitter::emit(result, runtime::Opcode::ArrStore, ra, ri, rv);
        reg.free(ra);
        reg.free(ri);
        reg.free(rv);
        break;
    }

    case mir::MirNodeKind::TupleStore: {
        const auto n = reinterpret_cast<mir::MirTupleStore*>(node);
        const auto rt = asm_mir_expr(result, n->target.get());
        const auto rv = asm_mir_expr(result, n->value.get());
        InstEmitter::emit(result, runtime::Opcode::TupleSet, rt, n->index, rv);
        reg.free(rt);
        reg.free(rv);
        break;
    }

    case mir::MirNodeKind::Expr: {
        const auto n = reinterpret_cast<mir::MirExprNode*>(node);
        const auto r = asm_mir_expr(result, n->expr.get());
        if (r != 0) {
            reg.free(r);
        }
        break;
    }

    case mir::MirNodeKind::Func: {
        const auto n = reinterpret_cast<mir::MirFuncDefine*>(node);
        auto func_code = asm_func(n);
        break;
    }
    case mir::MirNodeKind::NativeFunc:
        break;
    }
}


std::vector<uint8_t> Assembler::asm_func(mir::MirFuncDefine* def) noexcept {
    vals.clear();
    reg = RegAllocator{};
    label_positions.clear();
    pending_fixups.clear();
    next_local_var = 0;

    InstEmitter::InstSeq insts;
    insts.reserve(128);

    for (size_t i = 0; i < def->params.size(); ++i) {
        vals[def->params[i]] = Val(static_cast<uint8_t>(i));
        if (next_local_var <= i) next_local_var = static_cast<uint8_t>(i + 1);
    }

    for (auto& n : def->body) {
        asm_mir_node(insts, n.get());
    }

    const auto terminal = !insts.empty() &&
        (insts.back().bytes[0] == static_cast<uint8_t>(runtime::Opcode::Ret) ||
         insts.back().bytes[0] == static_cast<uint8_t>(runtime::Opcode::Raise));
    if (!terminal) {
        InstEmitter::emit(insts, runtime::Opcode::Ret);
    }

    std::vector<uint8_t> code;
    code.reserve(insts.size() * 4);
    for (const auto&[bytes] : insts) {
        code.push_back(bytes[0]);
        code.push_back(bytes[1]);
        code.push_back(bytes[2]);
        code.push_back(bytes[3]);
    }

    resolve_fixups(code);

    return code;
}


void Assembler::resolve_fixups(std::vector<uint8_t>& code) noexcept {
    for (auto&[inst_pos, op, label] : pending_fixups) {
        const auto it = label_positions.find(label);
        if (it == label_positions.end()) continue;

        const auto target = static_cast<int64_t>(it->second);
        const auto origin = static_cast<int64_t>(inst_pos);
        const auto offset = static_cast<int16_t>(target - origin);

        switch (op) {
        case runtime::Opcode::Goto:
            // offset at bytes 1-2
            code[inst_pos + 1] = static_cast<uint8_t>(offset & 0xFF);
            code[inst_pos + 2] = static_cast<uint8_t>((offset >> 8) & 0xFF);
            break;
        case runtime::Opcode::IfTrue:
        case runtime::Opcode::IfFalse:
            // offset at bytes 2-3 (byte 1 is the register)
            code[inst_pos + 2] = static_cast<uint8_t>(offset & 0xFF);
            code[inst_pos + 3] = static_cast<uint8_t>((offset >> 8) & 0xFF);
            break;
        default:
            break;
        }
    }
}

std::shared_ptr<ModuleType> Assembler::get_module_type(const size_t idx) noexcept {
    for (auto &[i, t]: imports | std::views::values) {
        if (i == idx) return t;
    }
    return nullptr;
}


std::vector<uint8_t> Assembler::asm_module(mir::MirModule* mod) noexcept {
    std::vector<uint8_t> result;
    result.reserve(512);

    write_u32(result, LMX_MAGIC_NUM);
    write_u32(result, LMX_VERSION);

    struct CompiledFunc {
        std::string name;
        std::vector<uint8_t> code;
    };
    std::vector<CompiledFunc> compiled_funcs;

    struct EncodeNativeFunc {
        std::string name;
        std::vector<runtime::ValueKind> arg_ty;
        runtime::ValueKind ret_ty;
    };
    static auto encoder_native = [](EncodeNativeFunc& n) -> std::vector<uint8_t> {
        std::vector<uint8_t> n_re;

        n_re.insert(n_re.end(), n.name.begin(), n.name.end());
        n_re.push_back(0);
        n_re.push_back(static_cast<uint8_t>(n.arg_ty.size()));
        auto* arg_ty_ref = reinterpret_cast<std::vector<uint8_t>*>(&n.arg_ty);
        n_re.insert(n_re.end(), arg_ty_ref->begin(), arg_ty_ref->end());
        n_re.push_back(static_cast<uint8_t>(n.ret_ty));
        return n_re;
    };
    std::vector<EncodeNativeFunc> encode_native_funcs;

    funcs.clear();
    native_funcs.clear();
    cp.clear();
    imports.clear();
    cp_cnt = 0;
    size_t native_func_idx = 0;
    size_t func_idx = 0;
    std::vector<std::shared_ptr<mir::MirNode>> top_level_nodes;


    for (auto& [name, ty] : mod->imports) {
        const auto idx = imports.size();
        imports[name] = {idx, ty};
    }

    for (auto& node : mod->nodes) {
        if (node->kind == mir::MirNodeKind::Func) {
            const auto* f = reinterpret_cast<mir::MirFuncDefine*>(node.get());
            funcs[f->name] = func_idx++;
            compiled_funcs.push_back({.name = f->name, .code = std::vector<uint8_t>{}});
        } else if (node->kind == mir::MirNodeKind::NativeFunc) {
            const auto* f = reinterpret_cast<mir::MirNativeFuncDefine*>(node.get());
            native_funcs[f->name] = native_func_idx++;
            encode_native_funcs.push_back({.name = f->symbol, .arg_ty = f->params, .ret_ty = f->ret_ty});
        }
        else {
            top_level_nodes.push_back(node);
        }
    }

    vals.clear();
    reg = RegAllocator{};
    label_positions.clear();
    pending_fixups.clear();
    next_local_var = 0;

    InstEmitter::InstSeq entry_insts;
    for (auto& n : top_level_nodes) {
        asm_mir_node(entry_insts, n.get());
    }
    if (entry_insts.empty() || entry_insts.back().bytes[0] != static_cast<uint8_t>(runtime::Opcode::Ret)) {
        InstEmitter::emit(entry_insts, runtime::Opcode::Halt);
    }

    std::vector<uint8_t> entry_code;
    entry_code.reserve(entry_insts.size() * 4);
    for (auto& w : entry_insts) {
        entry_code.push_back(w.bytes[0]);
        entry_code.push_back(w.bytes[1]);
        entry_code.push_back(w.bytes[2]);
        entry_code.push_back(w.bytes[3]);
    }
    resolve_fixups(entry_code);

    for (auto&[name, code] : compiled_funcs) {
        for (auto& node : mod->nodes) {
            if (node->kind == mir::MirNodeKind::Func) {
                if (auto* f = reinterpret_cast<mir::MirFuncDefine*>(node.get());
                    f->name == name) {

                    code = asm_func(f);
                    break;
                }
            }
        }
    }

    std::vector<uint8_t> func_section;
    for (auto&[name, code] : compiled_funcs) {
        const auto func_len = static_cast<uint32_t>(code.size());
        func_section.push_back(static_cast<uint8_t>(func_len & 0xFF));
        func_section.push_back(static_cast<uint8_t>((func_len >> 8) & 0xFF));
        func_section.push_back(static_cast<uint8_t>((func_len >> 16) & 0xFF));
        func_section.push_back(static_cast<uint8_t>((func_len >> 24) & 0xFF));
        func_section.insert(func_section.end(), code.begin(), code.end());
    }

    write_u64(result, func_section.size());
    result.insert(result.end(), func_section.begin(), func_section.end());


    write_u64(result, cp.size());
    result.insert(result.end(), cp.begin(), cp.end());


    std::vector<uint8_t> native_decls;
    for (auto& n : encode_native_funcs) {
        auto data = encoder_native(n);
        native_decls.insert(native_decls.end(), data.begin(), data.end());
    }

    write_u64(result, native_decls.size() + mod->lib_name.size() + 1);
    if (!mod->lib_name.empty()) {
        result.insert(result.end(), mod->lib_name.begin(), mod->lib_name.end());
    }
    result.push_back(0);
    result.insert(result.end(), native_decls.begin(), native_decls.end());


    std::vector<uint8_t> import_data;
    for (auto &ty: mod->imports | std::views::values) {
        const std::string& out_path = ty->load_path;
        import_data.insert(import_data.end(), out_path.begin(), out_path.end());
        import_data.push_back(0);
    }
    write_u64(result, import_data.size());
    result.insert(result.end(), import_data.begin(), import_data.end());


    write_u64(result, entry_code.size());
    result.insert(result.end(), entry_code.begin(), entry_code.end());
    result.shrink_to_fit();
    return result;
}

}
