//
// Created by meian on 2026/7/20.
//

#include "assembler.hpp"
#include "lmx.h"

#include <algorithm>
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
    case runtime::Opcode::FuncCreate:
    case runtime::Opcode::CallVirtual:
    case runtime::Opcode::CCall:
    case runtime::Opcode::CallFast:
    case runtime::Opcode::Ret:
    case runtime::Opcode::Goto:
    case runtime::Opcode::IfTrue:
    case runtime::Opcode::IfFalse:
    case runtime::Opcode::Call:
    case runtime::Opcode::Push:
        return false;
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
    case runtime::Opcode::CConst:
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
    case runtime::Opcode::Pop:
    case runtime::Opcode::And:
    case runtime::Opcode::Or:
        return true;
        break;
    }
    return false;
}

bool InstEmitter::inst_is_call(const runtime::Opcode::Opcode op) noexcept {
    switch (op) {
    case runtime::Opcode::CCall:
    case runtime::Opcode::CallFast:
    case runtime::Opcode::Call:
    case runtime::Opcode::CallVirtual:
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
    regs.reset(reg - 1);
}

// --- Assembler::Val ---

Assembler::Val::Val(const uint8_t reg, const bool is_tmp) noexcept
    : kind(Kind::Reg), is_tmp(is_tmp), reg(reg) {}

Assembler::Val::Val(const uint8_t var) noexcept
    : kind(Kind::Var), is_tmp(false), var(var) {}

// --- Assembler ---

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
    if (it == vals.end()) return std::nullopt;
    return &it->second;
}

std::optional<Assembler::GlobalVar*> Assembler::find_global(const std::string& name) noexcept {
    const auto it = globals.find(name);
    if (it == globals.end()) return std::nullopt;
    return &it->second;
}

// ============================================================
//  Expression assembly: returns the register holding the result
// ============================================================

uint8_t Assembler::asm_mir_expr(InstEmitter::InstSeq& insts, mir::MirExpr* node) noexcept {
    switch (node->kind) {
    case mir::MirExprKind::Ref: {
        const auto e = reinterpret_cast<mir::MirRefExpr*>(node);
        const auto v_opt = find_var(e->name);
        if (!v_opt) return 0;
        const auto& v = *v_opt;
        if (v->kind == Val::Kind::Reg) {
            return v->reg;
        }
        // Var kind – emit LGet to load from local variable
        const auto r = *reg.alloc();
        InstEmitter::emit(insts, runtime::Opcode::LGet, r, v->var);
        return r;
    }

    case mir::MirExprKind::Literal: {
        const auto e = reinterpret_cast<mir::MirLiteralExpr*>(node);
        switch (e->literal_kind) {
        case mir::MirLiteralKind::Integer: {
            const auto r = *reg.alloc();
            const auto val = static_cast<int16_t>(std::stoi(e->data));
            InstEmitter::emit(insts, runtime::Opcode::IConst, r, static_cast<uint16_t>(val));
            return r;
        }
        case mir::MirLiteralKind::Float: {
            // Fraction encoding as two IConst + IDiv
            const runtime::Fraction f(e->data);
            const auto r1 = *reg.alloc();
            const auto r2 = *reg.alloc();
            const auto r3 = *reg.alloc();
            InstEmitter::emit(insts, runtime::Opcode::IConst, r1, static_cast<uint16_t>(f.den));
            InstEmitter::emit(insts, runtime::Opcode::IConst, r2, static_cast<uint16_t>(f.num));
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
        case mir::MirLiteralKind::String:
            // String constants not yet supported
            return 0;
        }
        break;
    }

    case mir::MirExprKind::Operate: {
        const auto e = reinterpret_cast<mir::MirOperateExpr*>(node);

        // RetVoid
        if (e->operate_kind == mir::MirOperateKind::RetVoid) {
            InstEmitter::emit(insts, runtime::Opcode::Ret);
            return 0;
        }

        switch (e->opcode) {
        // --- Binary integer ops ---
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

        // --- Binary float ops ---
        case runtime::Opcode::FAdd:
        case runtime::Opcode::FSub:
        case runtime::Opcode::FMul:
        case runtime::Opcode::FDiv:
        case runtime::Opcode::FMod: {
            const auto& op = *reinterpret_cast<mir::MirFAddExpr*>(node);
            const auto rl = asm_mir_expr(insts, op.lhs.get());
            const auto rr = asm_mir_expr(insts, op.rhs.get());
            const auto rd = *reg.alloc();
            InstEmitter::emit(insts, e->opcode, rd, rl, rr);
            reg.free(rl);
            reg.free(rr);
            return rd;
        }

        // --- Unary ops ---
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

        // --- Ret with value ---
        case runtime::Opcode::Ret: {
            const auto& ret = *reinterpret_cast<mir::MirRetExpr*>(node);
            const auto r = asm_mir_expr(insts, ret.value.get());
            InstEmitter::emit(insts, runtime::Opcode::MovRR, uint8_t{0}, r);
            reg.free(r);
            InstEmitter::emit(insts, runtime::Opcode::Ret);
            return 0;
        }

        // --- Goto ---
        case runtime::Opcode::Goto: {
            const auto& g = *reinterpret_cast<mir::MirGotoExpr*>(node);
            const auto pos = insts.size() * 4;
            InstEmitter::emit(insts, runtime::Opcode::Goto, uint16_t{0}, uint8_t{0});
            pending_fixups.push_back({pos, runtime::Opcode::Goto, g.label});
            return 0;
        }

        // --- Conditional branches ---
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

        // --- CallFast ---
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
            auto using_regs = reg.get_all_using();
            for (const auto r : using_regs) {
                InstEmitter::emit(insts, runtime::Opcode::Push, r);
            }

            const auto func_it = funcs.find(c.name);
            if (func_it == funcs.end()) return 0;
            const auto func_idx = static_cast<uint16_t>(func_it->second);

            const auto rd = *reg.alloc();
            InstEmitter::emit(insts, runtime::Opcode::CallFast, func_idx, argc);
            for (const auto r : using_regs | std::views::reverse) {
                InstEmitter::emit(insts, runtime::Opcode::Pop, r);
            }
            // CallFast result is left in regs[0] by convention
            InstEmitter::emit(insts, runtime::Opcode::MovRR, rd, uint8_t{0}); // return from r0
            return rd;
        }

        // --- Halt ---
        case runtime::Opcode::Halt: {
            InstEmitter::emit(insts, runtime::Opcode::Halt);
            return 0;
        }

        // --- New ---
        case runtime::Opcode::New: {
            const auto& n = *reinterpret_cast<mir::MirNewExpr*>(node);
            const auto r = asm_mir_expr(insts, n.expr.get());
            const auto rd = *reg.alloc();
            // Use constant pool index from the evaluated expression
            // For now, just move the result
            InstEmitter::emit(insts, runtime::Opcode::MovRR, rd, r);
            reg.free(r);
            return rd;
        }

        default:
            return 0;
        }
    }
    }
    return 0;
}

// ============================================================
//  Node assembly – appends instructions to `result`
// ============================================================

void Assembler::asm_mir_node(InstEmitter::InstSeq& result, mir::MirNode* node) noexcept {
    switch (node->kind) {
    case mir::MirNodeKind::Label: {
        const auto n = reinterpret_cast<mir::MirLabel*>(node);
        label_positions[n->name] = result.size() * 4;
        break;
    }

    case mir::MirNodeKind::TempAssign: {
        const auto n = reinterpret_cast<mir::MirTempAssign*>(node);
        const auto r = asm_mir_expr(result, n->expr.get());
        if (const auto found = find_var(n->name)) {
            InstEmitter::emit(result, runtime::Opcode::MovRR, (*found)->reg, r);
            reg.free(r);
        } else {
            vals[n->name] = Val(r, true);
        }
        break;
    }

    case mir::MirNodeKind::Assign: {
        const auto n = reinterpret_cast<mir::MirAssign*>(node);
        const auto r = asm_mir_expr(result, n->expr.get());

        const auto v_opt = find_var(n->name);
        if (!v_opt) {
            // New variable – allocate a local var slot
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

    case mir::MirNodeKind::Expr: {
        const auto n = reinterpret_cast<mir::MirExprNode*>(node);
        const auto r = asm_mir_expr(result, n->expr.get());
        // Free the result register if the expression produced one
        if (r != 0) {
            reg.free(r);
        }
        break;
    }

    case mir::MirNodeKind::Func: {
        const auto n = reinterpret_cast<mir::MirFuncDefine*>(node);
        auto func_code = asm_func(n);
        // Store the compiled function code (handled in asm_module)
        // For now, just skip – functions are collected at module level
        break;
    }
    }
}

// ============================================================
//  Function compilation
// ============================================================

std::vector<uint8_t> Assembler::asm_func(mir::MirFuncDefine* def) noexcept {
    vals.clear();
    reg = RegAllocator{};
    label_positions.clear();
    pending_fixups.clear();
    next_local_var = 0;

    InstEmitter::InstSeq insts;
    insts.reserve(128);

    // Register parameters as local variables
    for (size_t i = 0; i < def->params.size(); ++i) {
        vals[def->params[i]] = Val(static_cast<uint8_t>(i));
        if (next_local_var <= i) next_local_var = static_cast<uint8_t>(i + 1);
    }

    // Compile body nodes
    for (auto& n : def->body) {
        asm_mir_node(insts, n.get());
    }

    // Ensure function ends with a Ret
    if (insts.empty() || insts.back().bytes[0] != static_cast<uint8_t>(runtime::Opcode::Ret)) {
        InstEmitter::emit(insts, runtime::Opcode::Ret);
    }

    // Flatten instructions into raw bytecode
    std::vector<uint8_t> code;
    code.reserve(insts.size() * 4);
    for (const auto& w : insts) {
        code.push_back(w.bytes[0]);
        code.push_back(w.bytes[1]);
        code.push_back(w.bytes[2]);
        code.push_back(w.bytes[3]);
    }

    // Resolve label fixups
    resolve_fixups(code);

    return code;
}

// ============================================================
//  Fixup resolution
// ============================================================

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

// ============================================================
//  Module assembly (binary format)
// ============================================================

std::vector<uint8_t> Assembler::asm_module(mir::MirModule* mod) noexcept {
    std::vector<uint8_t> result;
    result.reserve(4096);

    // Magic number
    write_u32(result, LMX_MAGIC_NUM);
    // Version
    write_u32(result, LMX_VERSION);

    // ---- Collect functions and compile code ----
    struct CompiledFunc {
        std::string name;
        std::vector<uint8_t> code;
    };
    std::vector<CompiledFunc> compiled_funcs;

    // First pass: assign indices and compile all top-level functions
    funcs.clear();
    size_t func_idx = 0;
    std::vector<std::shared_ptr<mir::MirNode>> top_level_nodes;

    for (auto& node : mod->nodes) {
        if (node->kind == mir::MirNodeKind::Func) {
            auto* fdef = reinterpret_cast<mir::MirFuncDefine*>(node.get());
            funcs[fdef->name] = func_idx++;
            compiled_funcs.push_back({.name = fdef->name, .code = std::vector<uint8_t>{}});
        } else {
            top_level_nodes.push_back(node);
        }
    }

    // Compile top-level code as the entry point
    vals.clear();
    globals.clear();
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

    // Flatten entry code
    std::vector<uint8_t> entry_code;
    entry_code.reserve(entry_insts.size() * 4);
    for (auto& w : entry_insts) {
        entry_code.push_back(w.bytes[0]);
        entry_code.push_back(w.bytes[1]);
        entry_code.push_back(w.bytes[2]);
        entry_code.push_back(w.bytes[3]);
    }
    resolve_fixups(entry_code);

    // Compile each function body
    for (auto& cf : compiled_funcs) {
        for (auto& node : mod->nodes) {
            if (node->kind == mir::MirNodeKind::Func) {
                auto* fdef = reinterpret_cast<mir::MirFuncDefine*>(node.get());
                if (fdef->name == cf.name) {
                    cf.code = asm_func(fdef);
                    break;
                }
            }
        }
    }

    // ---- Write function section (user functions only) ----
    std::vector<uint8_t> func_section;

    for (auto& cf : compiled_funcs) {
        const auto func_len = static_cast<uint32_t>(cf.code.size());
        func_section.push_back(static_cast<uint8_t>(func_len & 0xFF));
        func_section.push_back(static_cast<uint8_t>((func_len >> 8) & 0xFF));
        func_section.push_back(static_cast<uint8_t>((func_len >> 16) & 0xFF));
        func_section.push_back(static_cast<uint8_t>((func_len >> 24) & 0xFF));
        func_section.insert(func_section.end(), cf.code.begin(), cf.code.end());
    }

    write_u64(result, func_section.size());
    result.insert(result.end(), func_section.begin(), func_section.end());

    // ---- Write constant pool section (empty for now) ----
    write_u64(result, 0);

    // ---- Write entry code section (after constants, loaded as prog->code) ----
    write_u64(result, entry_code.size());
    result.insert(result.end(), entry_code.begin(), entry_code.end());

    return result;
}

}
