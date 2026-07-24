//
// Created by meian on 2026/7/20.
//

#pragma once
#include <bitset>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "mir.hpp"
#include "../runtime/opcode.hpp"
#include "../runtime/vm.hpp"

namespace lmx {

class InstEmitter {
public:
    struct InstWord {
        uint8_t bytes[4];
    };
    using InstSeq = std::vector<InstWord>;
    static void emit(InstSeq& s, runtime::Opcode::Opcode op, uint8_t  a, uint8_t  b, uint8_t c) noexcept;
    static void emit(InstSeq& s, runtime::Opcode::Opcode op, uint8_t  a, uint8_t  b) noexcept;
    static void emit(InstSeq& s, runtime::Opcode::Opcode op, uint16_t a, uint8_t  b) noexcept;
    static void emit(InstSeq& s, runtime::Opcode::Opcode op, uint8_t  a, uint16_t b) noexcept;
    static void emit(InstSeq& s, runtime::Opcode::Opcode op, uint8_t  a) noexcept;
    static void emit(InstSeq& s, runtime::Opcode::Opcode op) noexcept;

    static bool inst_is_ret_reg(runtime::Opcode::Opcode op) noexcept;
    static bool inst_is_call(runtime::Opcode::Opcode op) noexcept;
};

class RegAllocator {
    static constexpr size_t COMMON_REG_COUNT = static_cast<size_t>(LMX_VM_REG_COUNT) - 1;
    std::bitset<COMMON_REG_COUNT> regs;
public:
    std::optional<uint8_t> alloc() noexcept;
    void free(uint8_t reg) noexcept;
    std::vector<uint8_t> get_all_using() noexcept;
};

struct PendingFixup {
    size_t inst_pos;
    runtime::Opcode::Opcode op;
    std::string label;
};

class Assembler {
public:
    size_t counter{0};
    struct Val {
        enum class Kind { Var, Reg };
        Kind kind;
        bool is_tmp;
        union {
            uint8_t reg;
            uint8_t var;
        };
        Val() noexcept : kind(Kind::Reg), is_tmp(false), reg(0) {}
        explicit Val(uint8_t reg, bool is_tmp) noexcept;
        explicit Val(uint8_t var) noexcept;
    };
    struct GlobalVar {
        uint16_t idx;
    };

    std::unordered_map<std::string, Val> vals;
    std::unordered_map<std::string, GlobalVar> globals;
    std::unordered_map<std::string, size_t> funcs; // func name -> index
    RegAllocator reg;

    std::optional<Val*> find_var(const std::string& name) noexcept;
    std::optional<GlobalVar*> find_global(const std::string& name) noexcept;

    static void write_u32(std::vector<uint8_t>& buf, uint32_t value) noexcept;
    static void write_u64(std::vector<uint8_t>& buf, uint64_t value) noexcept;
    static void write_n(std::vector<uint8_t>& buf, const uint8_t* src, size_t n) noexcept;

    // Module-level assembly
    std::vector<uint8_t> asm_module(mir::MirModule* mod) noexcept;

    // Per-function state
    std::unordered_map<std::string, size_t> label_positions;
    std::vector<PendingFixup> pending_fixups;
    uint8_t next_local_var;

    std::vector<uint8_t> asm_func(mir::MirFuncDefine* def) noexcept;
    uint8_t asm_mir_expr(InstEmitter::InstSeq& result, mir::MirExpr* node) noexcept;
    void asm_mir_node(InstEmitter::InstSeq& result, mir::MirNode* node) noexcept;
    void resolve_fixups(std::vector<uint8_t>& code) noexcept;
};

}
