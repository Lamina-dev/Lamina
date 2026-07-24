//
// Created by meian on 2026/4/6.
//

#include "code_module.hpp"
#include "lmx.h"
#include "../opcode.hpp"

#include <sstream>

using namespace lmx::runtime;

FuncObj::FuncObj(CodeModule *mod, const uint8_t *addr, const uint32_t bytecode_len) noexcept
   : mod(mod), addr(addr), bytecode_len(bytecode_len) {}

namespace {
class ModuleLoader {
public:
    static bool check_magic(const uint8_t*& p) noexcept {
        using Magic = decltype(LMX_MAGIC_NUM);
        // if (p == nullptr) return false;
        if (*reinterpret_cast<const Magic*>(p) != LMX_MAGIC_NUM) return false;
        p += sizeof(Magic);
        return true;
    }
    static bool check_version(const uint8_t*& p) noexcept {
        using Version = decltype(LMX_VERSION);
        // if (p == nullptr) return false;
        if (*reinterpret_cast<const Version*>(p) != LMX_VERSION) return false;
        p += sizeof(Version);
        return true;
    }
    static bool load_cp(std::vector<ConstantPoolInfo>& result, const uint8_t*& p) noexcept {
        const auto size = *reinterpret_cast<const uint64_t*>(p);
        p += sizeof(uint64_t);
        const auto over = p + size;
        while (p != over) {
            switch (static_cast<ConstantId>(*p++)) {
            case ConstantId::Int: {
                using IdType = int64_t;
                result.emplace_back(*reinterpret_cast<const IdType*>(p));
                p += sizeof(IdType);
                break;
            }
            case ConstantId::Frac: {
                using IdType = FracInfo;
                result.emplace_back(reinterpret_cast<const IdType*>(p));
                p += sizeof(IdType);
                break;
            }
            case ConstantId::Str: {
                using IdType = StringInfo;
                result.emplace_back(reinterpret_cast<const IdType*>(p));
                p += sizeof(IdType);
                p += result.back().str->length;
                break;
            }
            }
        }
        return true;
    }
    static bool load_funcs(CodeModule* mod, std::vector<FuncObj>& result, const uint8_t*& p) noexcept {
        const auto over = p + *reinterpret_cast<const uint64_t*>(p) + sizeof(uint64_t);
        p += sizeof(uint64_t);
        while (p != over) {
            const auto len = *reinterpret_cast<const uint32_t*>(p);
            p += sizeof(uint32_t);
            result.emplace_back(mod, p, len);
            p += len;
        }
        return true;
    }

    static bool load_entry_code(const uint8_t*& code, size_t& code_len, const uint8_t*& p) noexcept {
        code_len = *reinterpret_cast<const uint64_t*>(p);
        p += sizeof(uint64_t);
        code = p;
        p += code_len;
        return true;
    }

};
}



Object *CodeModule::clone() const noexcept {
    return nullptr;
}

CodeModule::CodeModule(std::vector<ConstantPoolInfo> cp, std::vector<TypeInfo> types, std::vector<FuncObj> funcs, const uint8_t *code, const size_t code_len) noexcept
    : Object(ObjectKind::Code), cp(std::move(cp)), funcs(std::move(funcs)), types(std::move(types)), code(code), code_len(code_len) {}

CodeModule::CodeModule(std::vector<ConstantPoolInfo>&& cp, std::vector<TypeInfo>&& types, std::vector<FuncObj>&& funcs, const uint8_t *code, const size_t code_len) noexcept
    : Object(ObjectKind::Code), cp(std::move(cp)), funcs(std::move(funcs)), types(std::move(types)), code(code), code_len(code_len) {}

CodeModule::CodeModule(const uint8_t *binary) noexcept : Object(ObjectKind::Code) {
    ModuleLoader::check_magic(binary);
    ModuleLoader::check_version(binary);
    ModuleLoader::load_funcs(this, funcs, binary);
    ModuleLoader::load_cp(cp, binary);
    ModuleLoader::load_entry_code(code, code_len, binary);
}

std::string CodeModule::to_string() const noexcept {
    return type_info();
}

std::string CodeModule::type_info() const noexcept {
    return "code";
}

bool CodeModule::equals(const Object *other) const noexcept {
    return false;
}

bool CodeModule::operator==(const Object &other) const noexcept {
    return false;
}

bool CodeModule::operator!=(const Object &other) const noexcept {
    return false;
}

// --- Disassembler ---

namespace {

struct InstInfo {
    const char* name;
    enum ArgFmt : uint8_t { None, Reg, RegReg, RegRegReg, RegImm16, Imm16, Imm16Reg, RegIdx, RegArgc };
    ArgFmt fmt;
};

static constexpr InstInfo INST_TABLE[] = {
    /* Nop        */ {"nop",       InstInfo::None},
    /* New        */ {"new",       InstInfo::RegImm16},
    /* GetTrue    */ {"get_true",  InstInfo::Reg},
    /* GetFalse   */ {"get_false", InstInfo::Reg},
    /* GetNull    */ {"get_null",  InstInfo::Reg},
    /* IConst     */ {"iconst",    InstInfo::RegImm16},
    /* CConst     */ {"cconst",    InstInfo::RegImm16},
    /* Pop        */ {"pop",       InstInfo::Reg},
    /* Push       */ {"push",      InstInfo::Reg},
    /* Halt       */ {"halt",      InstInfo::None},
    /* IAdd       */ {"iadd",      InstInfo::RegRegReg},
    /* ISub       */ {"isub",      InstInfo::RegRegReg},
    /* IMul       */ {"imul",      InstInfo::RegRegReg},
    /* IDiv       */ {"idiv",      InstInfo::RegRegReg},
    /* IMod       */ {"imod",      InstInfo::RegRegReg},
    /* IPow       */ {"ipow",      InstInfo::RegRegReg},
    /* INeg       */ {"ineg",      InstInfo::RegReg},
    /* FuncCreate */ {"func_create", InstInfo::RegImm16},
    /* CallVirtual*/ {"call_virtual",InstInfo::RegIdx},
    /* CCall      */ {"ccall",     InstInfo::RegImm16},
    /* CallFast   */ {"call_fast", InstInfo::Imm16Reg},
    /* Ret        */ {"ret",       InstInfo::Reg},
    /* Goto       */ {"goto",      InstInfo::Imm16},
    /* ICmpEq     */ {"icmp_eq",   InstInfo::RegRegReg},
    /* ICmpNe     */ {"icmp_ne",   InstInfo::RegRegReg},
    /* ICmpLt     */ {"icmp_lt",   InstInfo::RegRegReg},
    /* ICmpLe     */ {"icmp_le",   InstInfo::RegRegReg},
    /* ICmpGt     */ {"icmp_gt",   InstInfo::RegRegReg},
    /* ICmpGe     */ {"icmp_ge",   InstInfo::RegRegReg},
    /* IfTrue     */ {"if_true",   InstInfo::RegImm16},
    /* IfFalse    */ {"if_false",  InstInfo::RegImm16},
    /* LGet       */ {"lget",      InstInfo::RegIdx},
    /* LSet       */ {"lset",      InstInfo::RegIdx},
    /* GGet       */ {"gget",      InstInfo::RegImm16},
    /* GSet       */ {"gset",      InstInfo::RegImm16},
    /* FAdd       */ {"fadd",      InstInfo::RegRegReg},
    /* FSub       */ {"fsub",      InstInfo::RegRegReg},
    /* FMul       */ {"fmul",      InstInfo::RegRegReg},
    /* FDiv       */ {"fdiv",      InstInfo::RegRegReg},
    /* FMod       */ {"fmodi",     InstInfo::RegRegReg},
    /* FNeg       */ {"fneg",      InstInfo::RegReg},
    /* MovRR      */ {"movrr",     InstInfo::RegReg},
    /* Call       */ {"call",      InstInfo::RegArgc},
    /* And        */ {"and",       InstInfo::RegRegReg},
    /* Or         */ {"or",        InstInfo::RegRegReg},
};

static constexpr size_t INST_COUNT = sizeof(INST_TABLE) / sizeof(INST_TABLE[0]);

static void decode_inst(std::ostringstream& out, const uint8_t* p, size_t offset) {
    const auto op = static_cast<lmx::runtime::Opcode::Opcode>(p[0]);
    const uint8_t a = p[1], b = p[2], c = p[3];

    if (static_cast<size_t>(op) >= INST_COUNT) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "  0x%04zX:  .byte  %02x %02x %02x %02x\n", offset, p[0], p[1], p[2], p[3]);
        out << buf;
        return;
    }

    const auto& info = INST_TABLE[static_cast<size_t>(op)];

    // Imm16 at p[1..2]: for Imm16, Imm16Reg formats
    const auto u16_lo = static_cast<uint16_t>(a | (b << 8));
    const auto i16_lo = static_cast<int16_t>(u16_lo);

    // Imm16 at p[2..3]: for RegImm16, RegIdx formats
    const auto u16_hi = static_cast<uint16_t>(b | (c << 8));
    const auto i16_hi = static_cast<int16_t>(u16_hi);

    char buf[96];
    switch (info.fmt) {
    case InstInfo::None:
        std::snprintf(buf, sizeof(buf), "  0x%04zX:  %s\n", offset, info.name);
        break;
    case InstInfo::Reg:
        std::snprintf(buf, sizeof(buf), "  0x%04zX:  %s  r%u\n", offset, info.name, a);
        break;
    case InstInfo::RegReg:
        std::snprintf(buf, sizeof(buf), "  0x%04zX:  %s  r%u, r%u\n", offset, info.name, a, b);
        break;
    case InstInfo::RegRegReg:
        std::snprintf(buf, sizeof(buf), "  0x%04zX:  %s  r%u, r%u, r%u\n", offset, info.name, a, b, c);
        break;
    case InstInfo::RegImm16:
        std::snprintf(buf, sizeof(buf), "  0x%04zX:  %s  r%u, %d\n", offset, info.name, a, i16_hi);
        break;
    case InstInfo::RegIdx:
        std::snprintf(buf, sizeof(buf), "  0x%04zX:  %s  r%u, #%u\n", offset, info.name, a, b);
        break;
    case InstInfo::Imm16:
        std::snprintf(buf, sizeof(buf), "  0x%04zX:  %s  %d\t; -> 0x%04zX\n", offset, info.name, i16_lo, offset + i16_lo);
        break;
    case InstInfo::Imm16Reg:
        std::snprintf(buf, sizeof(buf), "  0x%04zX:  %s  %u, %u\n", offset, info.name, u16_lo, c);
        break;
    case InstInfo::RegArgc:
        std::snprintf(buf, sizeof(buf), "  0x%04zX:  %s  r%u, %u\n", offset, info.name, a, b);
        break;
    }
    out << buf;
}

} // anonymous namespace

std::string CodeModule::disassemble() const noexcept {
    std::ostringstream out;
    out << "Module: " << funcs.size() << " function(s), " << cp.size() << " constant(s)\n";

    // Disassemble functions
    for (size_t i = 0; i < funcs.size(); ++i) {
        const auto& f = funcs[i];
        out << "\n--- Func #" << i << " (" << f.bytecode_len << " bytes) ---\n";
        for (size_t off = 0; off + 4 <= f.bytecode_len; off += 4) {
            decode_inst(out, f.addr + off, off);
        }
    }

    // Disassemble entry-point code (top-level)
    out << "\n--- Entry Point ---\n";
    if (code && code_len > 0) {
        for (size_t off = 0; off + 4 <= code_len; off += 4) {
            decode_inst(out, code + off, off);
        }
    } else if (code) {
        // Fallback: iterate until Halt (limited)
        size_t off = 0;
        constexpr size_t MAX_ENTRY_BYTES = 4096;
        while (off < MAX_ENTRY_BYTES) {
            const auto op = code[off];
            decode_inst(out, code + off, off);
            off += 4;
            if (op == static_cast<uint8_t>(lmx::runtime::Opcode::Opcode::Halt)) break;
        }
    } else {
        out << "  (none)\n";
    }

    // Constant pool
    if (!cp.empty()) {
        out << "\n--- Constant Pool ---\n";
        for (size_t i = 0; i < cp.size(); ++i) {
            const auto& c = cp[i];
            switch (c.id) {
            case ConstantId::Int:
                out << "  #" << i << ": int " << c.int_value << "\n";
                break;
            case ConstantId::Frac:
                out << "  #" << i << ": frac " << c.frac_info->num << "/" << c.frac_info->den << "\n";
                break;
            case ConstantId::Str:
                out << "  #" << i << ": str \"" << c.str->str << "\"\n";
                break;
            }
        }
    }

    return out.str();
}

void CodeModule::disassemble_to_file(FILE* out) const noexcept {
    const auto s = disassemble();
    std::fwrite(s.c_str(), 1, s.size(), out);
}

