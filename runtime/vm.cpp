//
// Created by meian on 2026/4/6.
//

#include "vm.hpp"

#include <chrono>

#include "object/fraction.hpp"
#include <cmath>
#include <iostream>
#include <ostream>
#include <ranges>

namespace lmx::runtime {
LaminaVM::LaminaVM(const int argc, char **argv) noexcept :
    // cp(cp),
    stack(new Value[LMX_VM_REG_COUNT * LMX_CALLSTACK_MAX_COUNT]),
    //local_vars_bp(new Value[LMX_LOCAL_VAR_COUNT * LMX_CALLSTACK_MAX_COUNT]),
    //local_vars_curp(local_vars_bp),
    global_vars(new Value[65536]),
    // cur_frame(new Frame(nullptr, nullptr, local_vars_curp)),
    args(argv, argc) {}

LaminaVM::~LaminaVM() noexcept {
    delete[] stack;
    delete[] global_vars;
    //delete[] local_vars_bp;
    for (const auto frames : free_frames) delete frames;
    delete cur_frame;
}

Value &LaminaVM::get_reg(const uint8_t reg) noexcept {
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
int LaminaVM::run(CodeModule *prog) noexcept {
    cur_frame = new Frame(nullptr, prog, nullptr);
    const uint8_t* ip = prog->code;

    // std::cout << prog->disassemble() << std::endl;

    // const auto start = std::chrono::high_resolution_clock::now();
    VM_DISPATCH

    VM_LABEL(Nop) {
        VM_NEXT
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
            regs[ip[1]] = allocator.alloc_string(c.str->str);
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
        regs[ip[1]] = *--stack;
        VM_NEXT
    }

    VM_LABEL(Push) {
        *stack++ = regs[ip[1]];
        VM_NEXT
    }

    VM_LABEL(Halt) {
        // const auto end = std::chrono::high_resolution_clock::now();
        // std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start) << std::endl;
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
        VM_NEXT
    }

    VM_LABEL(CallFast) {
        const auto* func = &cur_frame->mod->funcs[read_u16(ip + 1)];
        new_frame(this, func->mod, ip + 4);
        auto i = ip[3] - 1;
        while (i != 0) {
            cur_frame->local_vars[i] = regs[LMX_VM_REG_COUNT - 1 - i];
            i--;
        }
        cur_frame->local_vars[0] = regs[LMX_VM_REG_COUNT - 1];
        ip = func->addr;
        VM_NEXT_RAW
    }

    VM_LABEL(Ret) {
        ip = pop_frame(this);
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
        regs[ip[1]] = global_vars[read_u16(ip + 2)];
        VM_NEXT
    }

    VM_LABEL(GSet) {
        global_vars[read_u16(ip + 2)] = regs[ip[1]];
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

        auto i = ip[2];
        while (i != 0) {
            cur_frame->local_vars[i] = regs[LMX_VM_REG_COUNT - 1 - i];
            i--;
        }
        cur_frame->local_vars[0] = regs[LMX_VM_REG_COUNT - 1];
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

    VM_END
}
}
#undef VM_DISPATCH
#undef VM_END
#undef VM_LABEL
#undef VM_NEXT