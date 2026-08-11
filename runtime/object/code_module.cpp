//
// Created by meian on 2026/4/6.
//

#include "code_module.hpp"

#include <complex>
#include <cstring>
#include <fstream>
#include <iostream>

#include "lmx.h"
#include "dynload/dynload.h"
#include "../opcode.hpp"
#include <sstream>

#include "../error.hpp"
#include "../../utils/utils.hpp"

using namespace lmx::runtime;

FuncObj::FuncObj(CodeModuleObj *mod, const uint8_t *addr, const uint32_t bytecode_len) noexcept
   : mod(mod), addr(addr), bytecode_len(bytecode_len) {}

NativeFuncObj::NativeFuncObj(
    const void *addr,
    const uint8_t args_ty_len,
    const ValueKind *args_ty,
    const ValueKind ret_ty,
    const char* name
    ) noexcept
    : addr(addr), args_ty_len(args_ty_len), ret_ty(ret_ty), args_ty(args_ty), name(name) {}

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

    static bool load_native_decl(std::vector<NativeFuncObj>& result, DLLib*& handle, const uint8_t*& p) noexcept {
        const auto size = *reinterpret_cast<const uint64_t*>(p);
        p += sizeof(uint64_t);
        const auto over = p + size;

        const auto lib_name = reinterpret_cast<const char *>(p);
        if (*lib_name == '\0') {
            p += 1;
            return false;
        }
        const auto real_name = std::string(lib_prefix) + lib_name + lib_suffix;

        handle = dlLoadLibrary(real_name.c_str());

        p += strlen(lib_name) + 1;
        if (handle == nullptr) {
            VM_ERROR(RuntimeErrorType::ModuleLoad, "`" + real_name + "` not found.");
            return false;
        }

        while (p != over) {
            const auto name = reinterpret_cast<const char *>(p);

            p += strlen(name) + 1;

            const uint8_t count = *p;
            const auto* args_ty = reinterpret_cast<const ValueKind*>(++p);
            p += count;
            const auto ret_ty = static_cast<ValueKind>(*p++);
            const void* addr = dlFindSymbol(handle, name);
            if (addr == nullptr) {
                std::cerr << "native symbol `" << name << "` not found" << std::endl;
                std::exit(1);
            }

            result.emplace_back(addr, count, args_ty, ret_ty, name);
        }

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
            case ConstantId::Arr: {
                auto* ai = reinterpret_cast<ArrayInfo*>(const_cast<uint8_t*>(p));
                const auto len = ai->len;
                p += sizeof(uint32_t) + len * sizeof(ConstantPoolInfo);
                // 修正 Frac/Str 元素指针，使其指向紧随 infos[] 的数据段
                auto* data = const_cast<uint8_t*>(p);
                for (uint32_t i = 0; i < len; i++) {
                    auto* e = &ai->infos[i];
                    switch (e->id) {
                    case ConstantId::Int:
                        break;
                    case ConstantId::Frac: {
                        ::new (static_cast<void*>(e)) ConstantPoolInfo(
                            reinterpret_cast<const FracInfo*>(data));
                        data += sizeof(FracInfo);
                        break;
                    }
                    case ConstantId::Str: {
                        const auto* si = reinterpret_cast<const StringInfo*>(data);
                        ::new (static_cast<void*>(e)) ConstantPoolInfo(si);
                        data += sizeof(StringInfo) + si->length;
                        break;
                    }
                    default:
                        break;
                    }
                }
                result.emplace_back(const_cast<const ArrayInfo*>(ai));
                p = data;
                break;
            }
            }
        }
        return true;
    }
    static bool load_funcs(CodeModuleObj* mod, std::vector<FuncObj>& result, const uint8_t*& p) noexcept {
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

    static bool load_imports(decltype(CodeModuleObj::imports)& mod, const uint8_t*& p) noexcept {
        const auto over = p + *reinterpret_cast<const uint64_t*>(p) + sizeof(uint64_t);
        p += sizeof(uint64_t);
        while (p != over) {
            std::string tmp = reinterpret_cast<const char *>(p);
            const size_t len = tmp.size();
            const auto path = (lmx::current_module_path / lmx::module_cache_fold / std::move(tmp)).string();

            p += len + 1;
            std::ifstream ifs(path, std::ios::binary);
            if (!ifs.is_open()) return false;
            ifs.seekg(0, std::ios::end);

            std::vector<uint8_t> data(ifs.tellg());
            ifs.seekg(0, std::ios::beg);
            ifs.read(reinterpret_cast<std::istream::char_type *>(data.data()), static_cast<std::streamsize>(data.size()));
            ifs.close();
            mod.push_back(std::make_unique<CodeModuleObj>(std::move(data)));
        }
        return true;
    }
};
}



CodeModuleObj::~CodeModuleObj() noexcept {
    if (native_lib_handle) {
        dlFreeLibrary(native_lib_handle);
    }
}


CodeModuleObj::CodeModuleObj(std::vector<uint8_t>&& data) noexcept : Object(ObjectKind::Code), raw_data(std::move(data)) {

    const uint8_t* binary = raw_data.data();
    ModuleLoader::check_magic(binary);
    ModuleLoader::check_version(binary);
    ModuleLoader::load_funcs(this, funcs, binary);
    ModuleLoader::load_cp(cp, binary);
    ModuleLoader::load_native_decl(native_funcs, native_lib_handle, binary);
    ModuleLoader::load_imports(imports, binary);
    ModuleLoader::load_entry_code(code, code_len, binary);
}

std::string CodeModuleObj::to_string() const noexcept {
    return type_info();
}

std::string CodeModuleObj::type_info() const noexcept {
    return "code";
}

bool CodeModuleObj::equals(const Object *other) const noexcept {
    return false;
}

bool CodeModuleObj::operator==(const Object &other) const noexcept {
    return false;
}

bool CodeModuleObj::operator!=(const Object &other) const noexcept {
    return false;
}

// --- Disassembler ---

namespace {

struct InstInfo {
    const char* name;
    enum ArgFmt : uint8_t { None, Reg, RegReg, RegRegReg, RegImm16, Imm16, Imm16Reg, RegIdx, RegArgc };
    ArgFmt fmt;
};

constexpr InstInfo INST_TABLE[] = {
    /* Nop        */ {.name = "nop",       .fmt = InstInfo::None},
    /* New        */ {.name = "new",       .fmt = InstInfo::RegImm16},
    /* GetTrue    */ {.name = "get_true",  .fmt = InstInfo::Reg},
    /* GetFalse   */ {.name = "get_false", .fmt = InstInfo::Reg},
    /* GetNull    */ {.name = "get_null",  .fmt = InstInfo::Reg},
    /* IConst     */ {.name = "iconst",    .fmt = InstInfo::RegImm16},
    /* CConst     */ {.name = "cconst",    .fmt = InstInfo::RegImm16},
    /* NewArrar   */ {.name = "new_array",       .fmt = InstInfo::RegImm16},
    /* ArrLoad    */ {.name = "arr_load",      .fmt = InstInfo::RegRegReg},
    /* Halt       */ {.name = "halt",      .fmt = InstInfo::None},
    /* IAdd       */ {.name = "iadd",      .fmt = InstInfo::RegRegReg},
    /* ISub       */ {.name = "isub",      .fmt = InstInfo::RegRegReg},
    /* IMul       */ {.name = "imul",      .fmt = InstInfo::RegRegReg},
    /* IDiv       */ {.name = "idiv",      .fmt = InstInfo::RegRegReg},
    /* IMod       */ {.name = "imod",      .fmt = InstInfo::RegRegReg},
    /* IPow       */ {.name = "ipow",      .fmt = InstInfo::RegRegReg},
    /* INeg       */ {.name = "ineg",      .fmt = InstInfo::RegReg},
    /* FuncCreate */ {.name = "func_create", .fmt = InstInfo::RegImm16},
    /* ArrStore   */ {.name = "arr_store",.fmt = InstInfo::RegRegReg},
    /* CCall      */ {.name = "ccall",     .fmt = InstInfo::Imm16Reg},
    /* CallFast   */ {.name = "call_fast", .fmt = InstInfo::Imm16Reg},
    /* Ret        */ {.name = "ret",       .fmt = InstInfo::Reg},
    /* Goto       */ {.name = "goto",      .fmt = InstInfo::Imm16},
    /* ICmpEq     */ {.name = "icmp_eq",   .fmt = InstInfo::RegRegReg},
    /* ICmpNe     */ {.name = "icmp_ne",   .fmt = InstInfo::RegRegReg},
    /* ICmpLt     */ {.name = "icmp_lt",   .fmt = InstInfo::RegRegReg},
    /* ICmpLe     */ {.name = "icmp_le",   .fmt = InstInfo::RegRegReg},
    /* ICmpGt     */ {.name = "icmp_gt",   .fmt = InstInfo::RegRegReg},
    /* ICmpGe     */ {.name = "icmp_ge",   .fmt = InstInfo::RegRegReg},
    /* IfTrue     */ {.name = "if_true",   .fmt = InstInfo::RegImm16},
    /* IfFalse    */ {.name = "if_false",  .fmt = InstInfo::RegImm16},
    /* LGet       */ {.name = "lget",      .fmt = InstInfo::RegIdx},
    /* LSet       */ {.name = "lset",      .fmt = InstInfo::RegIdx},
    /* GGet       */ {.name = "gget",      .fmt = InstInfo::RegImm16},
    /* GSet       */ {.name = "gset",      .fmt = InstInfo::RegImm16},
    /* FAdd       */ {.name = "fadd",      .fmt = InstInfo::RegRegReg},
    /* FSub       */ {.name = "fsub",      .fmt = InstInfo::RegRegReg},
    /* FMul       */ {.name = "fmul",      .fmt = InstInfo::RegRegReg},
    /* FDiv       */ {.name = "fdiv",      .fmt = InstInfo::RegRegReg},
    /* FMod       */ {.name = "fmodi",     .fmt = InstInfo::RegRegReg},
    /* FNeg       */ {.name = "fneg",      .fmt = InstInfo::RegReg},
    /* MovRR      */ {.name = "movrr",     .fmt = InstInfo::RegReg},
    /* Call       */ {.name = "call",      .fmt = InstInfo::RegArgc},
    /* And        */ {.name = "and",       .fmt = InstInfo::RegRegReg},
    /* Or         */ {.name = "or",        .fmt = InstInfo::RegRegReg},
    /* FCmpEq     */ {.name = "fcmp_eq",   .fmt = InstInfo::RegRegReg},
    /* FCmpNe     */ {.name = "fcmp_ne",   .fmt = InstInfo::RegRegReg},
    /* FCmpLt     */ {.name = "fcmp_lt",   .fmt = InstInfo::RegRegReg},
    /* FCmpLe     */ {.name = "fcmp_le",   .fmt = InstInfo::RegRegReg},
    /* FCmpGt     */ {.name = "fcmp_gt",   .fmt = InstInfo::RegRegReg},
    /* FCmpGe     */ {.name = "fcmp_ge",   .fmt = InstInfo::RegRegReg},
    /* GetModule  */ {.name = "get_module",  .fmt = InstInfo::RegImm16},
    /* GetModuleAttr*/{.name = "get_module_attr", .fmt = InstInfo::RegImm16},
};

constexpr size_t INST_COUNT = std::size(INST_TABLE);

void decode_inst(std::ostringstream& out, const uint8_t* p, const size_t offset) {
    const auto op = static_cast<Opcode::Opcode>(p[0]);
    const uint8_t a = p[1], b = p[2], c = p[3];

    if (static_cast<size_t>(op) >= INST_COUNT) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "  0x%04zX:  .byte  %02x %02x %02x %02x\n", offset, p[0], p[1], p[2], p[3]);
        out << buf;
        return;
    }

    const auto&[name, fmt] = INST_TABLE[static_cast<size_t>(op)];

    // Imm16 at p[1..2]: for Imm16, Imm16Reg formats
    const auto u16_lo = static_cast<uint16_t>(a | (b << 8));
    const auto i16_lo = static_cast<int16_t>(u16_lo);

    // Imm16 at p[2..3]: for RegImm16, RegIdx formats
    const auto u16_hi = static_cast<uint16_t>(b | (c << 8));
    const auto i16_hi = static_cast<int16_t>(u16_hi);

    char buf[96];
    switch (fmt) {
    case InstInfo::None:
        std::snprintf(buf, sizeof(buf), "  0x%04zX:  %s\n", offset, name);
        break;
    case InstInfo::Reg:
        std::snprintf(buf, sizeof(buf), "  0x%04zX:  %s  r%u\n", offset, name, a);
        break;
    case InstInfo::RegReg:
        std::snprintf(buf, sizeof(buf), "  0x%04zX:  %s  r%u, r%u\n", offset, name, a, b);
        break;
    case InstInfo::RegRegReg:
        std::snprintf(buf, sizeof(buf), "  0x%04zX:  %s  r%u, r%u, r%u\n", offset, name, a, b, c);
        break;
    case InstInfo::RegImm16:
        std::snprintf(buf, sizeof(buf), "  0x%04zX:  %s  r%u, %d\n", offset, name, a, i16_hi);
        break;
    case InstInfo::RegIdx:
        std::snprintf(buf, sizeof(buf), "  0x%04zX:  %s  r%u, #%u\n", offset, name, a, b);
        break;
    case InstInfo::Imm16:
        std::snprintf(buf, sizeof(buf), "  0x%04zX:  %s  %d\t; -> 0x%04zX\n", offset, name, i16_lo, offset + i16_lo);
        break;
    case InstInfo::Imm16Reg:
        std::snprintf(buf, sizeof(buf), "  0x%04zX:  %s  %u, %u\n", offset, name, u16_lo, c);
        break;
    case InstInfo::RegArgc:
        std::snprintf(buf, sizeof(buf), "  0x%04zX:  %s  r%u, %u\n", offset, name, a, b);
        break;
    }
    out << buf;
}

} // anonymous namespace

static const char* value_kind_name(ValueKind kind) noexcept {
    switch (kind) {
    case ValueKind::Null:     return "Null";
    case ValueKind::C_Ptr:    return "CPtr";
    case ValueKind::Obj:      return "Obj";
    case ValueKind::Int:      return "Int";
    case ValueKind::Bool:     return "Bool";
    case ValueKind::Fraction: return "Frac";
    case ValueKind::C_VaList: return "CVaList";
    default:                  return "?";
    }
}

std::string CodeModuleObj::disassemble() const noexcept {
    std::ostringstream out;
    out << "Module: " << funcs.size() << " function(s), "
        << cp.size() << " constant(s), "
        << native_funcs.size() << " native(s)\n";

    // Native functions
    if (!native_funcs.empty()) {
        out << "\n--- Native Functions ---\n";
        for (size_t i = 0; i < native_funcs.size(); ++i) {
            const auto& nf = native_funcs[i];
            out << "  #" << i << ": ";
            out << nf.name;
            if (nf.addr == nullptr) out << " [unresolved] ";
            out << "(" << static_cast<int>(nf.args_ty_len) << " args) -> " << value_kind_name(nf.ret_ty) << "\n";
            for (uint8_t j = 0; j < nf.args_ty_len; ++j) {
                out << "    arg" << static_cast<int>(j) << ": " << value_kind_name(nf.args_ty[j]) << "\n";
            }
        }
    }

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
            if (op == static_cast<uint8_t>(Opcode::Halt)) break;
        }
    } else {
        out << "  (none)\n";
    }

    // Constant pool
    if (!cp.empty()) {
        out << "\n--- Constant Pool ---\n";
        for (size_t i = 0; i < cp.size(); ++i) {
            switch (const auto& c = cp[i]; c.id) {
            case ConstantId::Int:
                out << "  #" << i << ": int " << c.int_value << "\n";
                break;
            case ConstantId::Frac:
                out << "  #" << i << ": frac " << c.frac_info->num << "/" << c.frac_info->den << "\n";
                break;
            case ConstantId::Str: {
                const std::string str(c.str->str, c.str->length);
                out << "  #" << i << ": str \"" << str << "\"\n";
                break;
            }
            case ConstantId::Arr: {
                out << "  #" << i << ": arr len=" << c.arr->len << " [";
                for (uint32_t j = 0; j < c.arr->len; ++j) {
                    if (j > 0) out << ", ";
                    const auto& e = c.arr->infos[j];
                    switch (e.id) {
                    case ConstantId::Int:
                        out << e.int_value;
                        break;
                    case ConstantId::Frac:
                        out << e.frac_info->num << "/" << e.frac_info->den;
                        break;
                    case ConstantId::Str:
                        out << '\"' << std::string(e.str->str, e.str->length) << '\"';
                        break;
                    default:
                        break;
                    }
                }
                out << "]\n";
                break;
            }
            }
        }
    }

    return out.str();
}

void CodeModuleObj::disassemble_to_file(FILE* out) const noexcept {
    const auto s = disassemble();
    std::fwrite(s.c_str(), 1, s.size(), out);
}

