//
// Created by meian on 2026/4/6.
//

#include "vm.hpp"

#include <cassert>
#include <chrono>
#include <cstdlib>

#include "object/fraction.hpp"
#include "object/adt.hpp"
#include "object/literal.hpp"
#include <cmath>
#include <iostream>
#include <ostream>
#include <ranges>

namespace lmx::runtime {
LaminaVM::LaminaVM(const int argc, char **argv) noexcept :
    // cp(cp),
    stack_storage(new Value[LMX_VM_REG_COUNT * LMX_CALLSTACK_MAX_COUNT * 2]),
    stack(stack_storage + LMX_VM_REG_COUNT * LMX_CALLSTACK_MAX_COUNT),
    regs(stack_storage),
    //local_vars_bp(new Value[LMX_LOCAL_VAR_COUNT * LMX_CALLSTACK_MAX_COUNT]),
    //local_vars_curp(local_vars_bp),
    // global_vars(/*new Value[65536]*/nullptr),
    // cur_frame(new Frame(nullptr, nullptr, local_vars_curp)),
    args(argv, argc),
    call_vm(dcNewCallVM(4096)) {}

LaminaVM::~LaminaVM() noexcept {
    delete[] stack_storage;
    // delete[] global_vars;
    //delete[] local_vars_bp;
    for (const auto frames : free_frames) delete frames;
    delete cur_frame;
    dcFree(call_vm);
}

Value &LaminaVM::get_reg(const uint8_t reg) const noexcept {
    return regs[reg];
}

Frame::Frame(Frame* last, CodeModule* mod ,const uint8_t *ret_addr) noexcept
    : last(last), mod(mod), ret_addr(ret_addr)
//, local_vars(local_vars)
{}

Frame::~Frame() noexcept = default;

#if defined(__GNUC__) || defined(__clang__)
#define VM_DISPATCH \
static const void* dispatch[] = {\
    &&opNop, &&opNew,\
    &&opGetTrue, &&opGetFalse, &&opGetNull,\
    &&opIConst, &&opCConst, &&opPop, &&opPush, &&opHalt,\
    &&opIAdd, &&opISub, &&opIMul, &&opIDiv, &&opIMod, &&opIPow, &&opINeg,\
    &&opFuncCreate,\
    &&opCallVirtual, &&opCCall, &&opCallFast, &&opRet,\
    &&opGoto,\
    &&opICmpEq, &&opICmpNe, &&opICmpLt, &&opICmpLe, &&opICmpGt, &&opICmpGe,\
    &&opIfTrue, &&opIfFalse,\
    &&opLGet, &&opLSet,\
    &&opGGet, &&opGSet,\
    &&opFAdd, &&opFSub, &&opFMul, &&opFDiv, &&opFMod, &&opFNeg,\
    &&opMovRR,&&opCall, &&opAnd, &&opOr,\
    &&opFCmpEq, &&opFCmpNe, &&opFCmpLt, &&opFCmpLe, &&opFCmpGt, &&opFCmpGe, \
    &&opGetModule, &&opGetModuleAttr, &&opGetFunc,\
    &&opAdtNew, &&opAdtIs, &&opAdtGet,\
    &&opLiteralNew, &&opContains, &&opNotContains\
};\
goto *dispatch[*ip];

#define VM_LABEL(name) op##name:
#define VM_END
#define VM_NEXT ip += 4; goto *dispatch[*ip];
#define VM_NEXT_RAW goto *dispatch[*ip];
#else
#define VM_DISPATCH for (;;) { switch (*ip) {
#define VM_END } }
#define VM_LABEL(name) case Opcode::name:
#define VM_NEXT ip += 4; break;
#define VM_NEXT_RAW break;
#endif
LMX_INLINE static constexpr int16_t read_i16(const uint8_t* p) {
    return static_cast<int16_t>(p[0] | p[1] << 8);
};
LMX_INLINE static constexpr uint16_t read_u16(const uint8_t* p) {
    return static_cast<uint16_t>(p[0] | (p[1] << 8));
};

static AdtObj* make_adt_value(const CodeModule* module, Value* regs, const uint16_t constant_index) {
    const auto* info = module->cp[constant_index].adt_constructor;
    const std::string type_name(info->data, info->type_name_length);
    const std::string constructor(info->data + info->type_name_length, info->constructor_length);
    std::vector<Value> fields;
    fields.reserve(info->field_count);
    for (uint8_t i = 0; i < info->field_count; ++i) {
        fields.push_back(regs[LMX_VM_REG_COUNT - 1 - i]);
    }
    return new AdtObj(type_name, constructor, std::move(fields));
}

static LiteralObj* make_literal_value(Value* regs, const uint8_t count,
                                      const uint8_t flags) {
    std::vector<Value> elements;
    elements.reserve(count);
    for (uint8_t i = 0; i < count; ++i) {
        elements.push_back(regs[LMX_VM_REG_COUNT - 1 - i]);
    }
    const auto kind = (flags & 1U) != 0
        ? LiteralObj::Kind::Interval : LiteralObj::Kind::Set;
    return new LiteralObj(kind, std::move(elements),
                          (flags & 2U) != 0, (flags & 4U) != 0);
}

int LaminaVM::run(CodeModule *prog) noexcept {
    cur_frame = new Frame(nullptr, prog, nullptr);
    const uint8_t* ip = prog->code;
    // assert((reinterpret_cast<uint64_t>(ip) % 4) == 0);
#if !NDEBUG
    const char* debug_dump = std::getenv("LMX_DEBUG_DUMP");
    if (debug_dump && debug_dump[0] != '\0' && debug_dump[0] != '0') {
        std::cout << prog->disassemble() << std::endl;
    }
#endif
    // return 0;
    //const auto start = std::chrono::high_resolution_clock::now();
    VM_DISPATCH


    VM_LABEL(Nop) {
        ip++;
        VM_NEXT_RAW
    }

    VM_LABEL(New) {

        switch (const auto& c = cur_frame->mod->cp[read_u16(ip + 2)]; c.id) {
        case ConstantId::Int: {
            regs[ip[1]] = c.int_value;
            break;
        }
        case ConstantId::Frac: {
            const auto frac = c.frac_info;
            new (&regs[ip[1]]) Value(frac->num, frac->den);
            break;
        }
        case ConstantId::Str: {
            regs[ip[1]] = allocator.alloc_string(c.str->str, c.str->length);
            break;
        }
        }
        VM_NEXT
    }

    VM_LABEL(GetTrue) {
        regs[ip[1]] = true;
        VM_NEXT
    }

    VM_LABEL(GetFalse) {
        regs[ip[1]] = false;
        VM_NEXT
    }

    VM_LABEL(GetNull) {
        regs[ip[1]] = nullptr;
        VM_NEXT
    }

    VM_LABEL(IConst) {
        regs[ip[1]] = Value(static_cast<int64_t>(read_i16(ip + 2)));
        VM_NEXT
    }

    VM_LABEL(CConst) {
        // 抛弃
        // switch (uint16_t idx = read_u16(ip + 2); cp[idx].id) {
        //     case ConstantId::Int:
        //         new (&regs[ip[1]]) Value(cp[idx].int_value);
        //         break;
        //     case ConstantId::Str:
        //         new (&regs[ip[1]]) Value(cp[idx].str);
        //         break;
        //     default:
        //         new (&regs[ip[1]]) Value();
        //         break;
        // }
        // VM_NEXT
    }

    VM_LABEL(Pop) {
        Value* slot = --stack;
        regs[ip[1]] = *slot;
        *slot = nullptr;
        VM_NEXT
    }

    VM_LABEL(Push) {
        *stack++ = regs[ip[1]];
        VM_NEXT
    }

    VM_LABEL(Halt) {
        //const auto end = std::chrono::high_resolution_clock::now();
        //std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start) << std::endl;
        return 0;
    }

    VM_LABEL(IAdd) {
        regs[ip[1]] = regs[ip[2]] + regs[ip[3]];
        VM_NEXT
    }

    VM_LABEL(ISub) {
        regs[ip[1]] = regs[ip[2]] - regs[ip[3]];
        VM_NEXT
    }

    VM_LABEL(IMul) {
        regs[ip[1]] = regs[ip[2]] * regs[ip[3]];
        VM_NEXT
    }

    VM_LABEL(IDiv) {
        new (&regs[ip[1]]) Value (regs[ip[2]].int_val, regs[ip[3]].int_val);
        VM_NEXT
    }

    VM_LABEL(IMod) {
        regs[ip[1]] = regs[ip[2]] % regs[ip[3]];
        VM_NEXT
    }

    VM_LABEL(IPow) {
        regs[ip[1]] = static_cast<int64_t>(std::pow(
            regs[ip[2]].int_val, regs[ip[3]].int_val
            ));
        VM_NEXT
    }

    VM_LABEL(INeg) {
        regs[ip[1]] = -regs[ip[2]];
        VM_NEXT
    }

    VM_LABEL(FuncCreate) { // create lambda func
        uint16_t code_idx = read_u16(ip + 2);
        // regs[ip[1]] = new CodeModule(code_idx, nullptr);
        VM_NEXT
    }

    VM_LABEL(CallVirtual) {
        VM_NEXT
    }

    VM_LABEL(CCall) {
        native_call(read_u16(ip + 1), ip[3]);
        VM_NEXT
    }

    VM_LABEL(CallFast) {
        const auto* func = &cur_frame->mod->funcs[read_u16(ip + 1)];
        new_frame(this, func->mod, ip + 4);
        for (uint8_t i = 0; i < ip[3]; ++i) {
            cur_frame->local_vars[i] = regs[LMX_VM_REG_COUNT - 1 - i];
        }

        regs += LMX_VM_REG_COUNT;

        ip = func->addr;
        VM_NEXT_RAW
    }

    VM_LABEL(Ret) {
        ip = pop_frame(this);

        const auto r0 = regs;

        regs -= LMX_VM_REG_COUNT;
        regs[0] = *r0;
        VM_NEXT_RAW
    }

    VM_LABEL(Goto) {
        ip += read_i16(ip + 1);
        VM_NEXT_RAW
    }

    VM_LABEL(ICmpEq) {
        regs[ip[1]] = regs[ip[2]] == regs[ip[3]];
        VM_NEXT
    }

    VM_LABEL(ICmpNe) {
        regs[ip[1]] = regs[ip[2]] != regs[ip[3]];
        VM_NEXT
    }

    VM_LABEL(ICmpLt) {
        regs[ip[1]] = regs[ip[2]] < regs[ip[3]];
        VM_NEXT
    }

    VM_LABEL(ICmpLe) {
        regs[ip[1]] = regs[ip[2]] <= regs[ip[3]];
        VM_NEXT
    }

    VM_LABEL(ICmpGt) {
        regs[ip[1]] = regs[ip[2]] > regs[ip[3]];
        VM_NEXT
    }

    VM_LABEL(ICmpGe) {
        regs[ip[1]] = regs[ip[2]] >= regs[ip[3]];
        VM_NEXT
    }

    VM_LABEL(IfTrue) {
        if (static_cast<bool>(regs[ip[1]])) {
            ip += read_i16(ip + 2);
            VM_NEXT_RAW
        } else {
            VM_NEXT
        }
    }

    VM_LABEL(IfFalse) {
        if (!static_cast<bool>(regs[ip[1]])) {
            ip += read_i16(ip + 2);
            VM_NEXT_RAW
        } else {
            VM_NEXT
        }
    }

    VM_LABEL(LGet) {
        regs[ip[1]] = cur_frame->local_vars[ip[2]];
        VM_NEXT
    }

    VM_LABEL(LSet) {
        cur_frame->local_vars[ip[2]] = regs[ip[1]];
        VM_NEXT
    }

    VM_LABEL(GGet) {
        // regs[ip[1]] = global_vars[read_u16(ip + 2)];
        VM_NEXT
    }

    VM_LABEL(GSet) {
        // global_vars[read_u16(ip + 2)] = regs[ip[1]];
        VM_NEXT
    }

    VM_LABEL(FAdd) {
        new (&regs[ip[1]]) Value(regs[ip[2]].frac_val + regs[ip[3]].frac_val);
        VM_NEXT
    }

    VM_LABEL(FSub) {
        new (&regs[ip[1]]) Value(regs[ip[2]].frac_val - regs[ip[3]].frac_val);
        VM_NEXT
    }

    VM_LABEL(FMul) {
        new (&regs[ip[1]]) Value(regs[ip[2]].frac_val * regs[ip[3]].frac_val);
        VM_NEXT
    }

    VM_LABEL(FDiv) {
        new (&regs[ip[1]]) Value(regs[ip[2]].frac_val / regs[ip[3]].frac_val);
        VM_NEXT
    }

    VM_LABEL(FMod) {
        new (&regs[ip[1]]) Value(regs[ip[2]].frac_val % regs[ip[3]].frac_val);
        VM_NEXT
    }

    VM_LABEL(FNeg) {
        regs[ip[1]].frac_val = -regs[ip[2]].frac_val;
        VM_NEXT
    }
    VM_LABEL(MovRR) {
        regs[ip[1]] = regs[ip[2]];
        VM_NEXT
    }
    VM_LABEL(Call) {
        const auto* func = static_cast<const FuncObj*>(regs[ip[1]].c_ptr);
        new_frame(this, func->mod, ip + 4);

        for (uint8_t i = 0; i < ip[2]; ++i) {
            cur_frame->local_vars[i] = regs[LMX_VM_REG_COUNT - 1 - i];
        }

        regs += LMX_VM_REG_COUNT;

        ip = func->addr;
        VM_NEXT_RAW
    }
    VM_LABEL(And) {
        regs[ip[1]] = regs[ip[2]] && regs[ip[3]];
        VM_NEXT
    }
    VM_LABEL(Or) {
        regs[ip[1]] = regs[ip[2]] || regs[ip[3]];
        VM_NEXT
    }
    VM_LABEL(FCmpEq) {
        regs[ip[1]] = regs[ip[2]].frac_val == regs[ip[3]].frac_val;
        VM_NEXT
    }
    VM_LABEL(FCmpNe) {
        regs[ip[1]] = regs[ip[2]].frac_val != regs[ip[3]].frac_val;
        VM_NEXT
    }
    VM_LABEL(FCmpLt) {
        regs[ip[1]] = regs[ip[2]].frac_val < regs[ip[3]].frac_val;
        VM_NEXT
    }
    VM_LABEL(FCmpLe) {
        regs[ip[1]] = regs[ip[2]].frac_val <= regs[ip[3]].frac_val;
        VM_NEXT
    }
    VM_LABEL(FCmpGt) {
        regs[ip[1]] = regs[ip[2]].frac_val > regs[ip[3]].frac_val;
        VM_NEXT
    }
    VM_LABEL(FCmpGe) {
        regs[ip[1]] = regs[ip[2]].frac_val >= regs[ip[3]].frac_val;
        VM_NEXT
    }
    VM_LABEL(GetModule) {
        regs[ip[1]] = cur_frame->mod->imports[read_u16(ip + 2)]->get();
        VM_NEXT
    }
    VM_LABEL(GetModuleAttr) {
        regs[ip[1]] = &reinterpret_cast<CodeModule*>(regs[0].obj)->funcs[read_u16(ip + 2)];
        VM_NEXT
    }
    VM_LABEL(GetFunc) {
        regs[ip[1]] = &cur_frame->mod->funcs[read_u16(ip + 2)];
        VM_NEXT
    }
    VM_LABEL(AdtNew) {
        regs[ip[1]] = make_adt_value(cur_frame->mod, regs, read_u16(ip + 2));
        VM_NEXT
    }
    VM_LABEL(AdtIs) {
        bool result = false;
        const auto& value = regs[ip[2]];
        const auto& constructor_value = regs[ip[3]];
        if (value.kind == ValueKind::Obj && value.obj && value.obj->get_kind() == ObjectKind::Adt &&
            constructor_value.kind == ValueKind::Obj && constructor_value.obj &&
            constructor_value.obj->get_kind() == ObjectKind::String) {
            const auto* adt = reinterpret_cast<const AdtObj*>(value.obj);
            const auto expected = reinterpret_cast<const String*>(constructor_value.obj)->to_string();
            result = adt->type_name().size() + adt->constructor().size() + 1 == expected.size() &&
                     expected.compare(0, adt->type_name().size(), adt->type_name()) == 0 &&
                     expected[adt->type_name().size()] == '\x1f' &&
                     expected.compare(adt->type_name().size() + 1, std::string::npos, adt->constructor()) == 0;
        }
        regs[ip[1]] = result;
        VM_NEXT
    }
    VM_LABEL(AdtGet) {
        const auto& value = regs[ip[2]];
        if (value.kind != ValueKind::Obj || !value.obj || value.obj->get_kind() != ObjectKind::Adt) {
            regs[ip[1]] = nullptr;
            VM_NEXT
        }
        const auto* field = reinterpret_cast<const AdtObj*>(value.obj)->field(ip[3]);
        regs[ip[1]] = field ? *field : Value{};
        VM_NEXT
    }
    VM_LABEL(LiteralNew) {
        regs[ip[1]] = make_literal_value(regs, ip[2], ip[3]);
        VM_NEXT
    }
    VM_LABEL(Contains) {
        const auto& container = regs[ip[3]];
        const bool result = container.kind == ValueKind::Obj && container.obj &&
            container.obj->get_kind() == ObjectKind::Literal &&
            reinterpret_cast<const LiteralObj*>(container.obj)->contains(regs[ip[2]]);
        regs[ip[1]] = result;
        VM_NEXT
    }
    VM_LABEL(NotContains) {
        const auto& container = regs[ip[3]];
        const bool result = container.kind == ValueKind::Obj && container.obj &&
            container.obj->get_kind() == ObjectKind::Literal &&
            reinterpret_cast<const LiteralObj*>(container.obj)->contains(regs[ip[2]]);
        regs[ip[1]] = !result;
        VM_NEXT
    }

    VM_END
}
}
#undef VM_DISPATCH
#undef VM_END
#undef VM_LABEL
#undef VM_NEXT
