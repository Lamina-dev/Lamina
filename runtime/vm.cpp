//
// Created by meian on 2026/4/6.
//

#include "vm.hpp"

#include "opcode.hpp"
#include <cmath>

using namespace lmx::runtime;

LaminaVM::LaminaVM(ConstantPoolInfo *cp, const int argc, char **argv) noexcept :
    cp(cp),
    stack(new Value[LMX_VM_REG_COUNT * LMX_CALLSTACK_MAX_COUNT]),
    local_vars_bp(new Value[LMX_LOCAL_VAR_COUNT * LMX_CALLSTACK_MAX_COUNT]),
    local_vars_curp(local_vars_bp),
    global_vars(new Value[65536]),
    cur_frame(new Frame(nullptr, nullptr, local_vars_curp)),
    args(argv, argc) {}

LaminaVM::~LaminaVM() noexcept {
    delete[] stack;
    delete[] global_vars;
    delete[] local_vars_bp;
    for (const auto frames : free_frames) delete frames;
    delete cur_frame;
}


void LaminaVM::new_frame(uint8_t *ret_addr) noexcept {
    local_vars_curp += LMX_LOCAL_VAR_COUNT;
    if (free_frames.empty()) {
        cur_frame = new Frame(cur_frame, ret_addr, local_vars_curp);
        //cur_frame = frame;
        return;
    }
    const auto frame = free_frames.back();
    free_frames.pop_back();
    frame->last = cur_frame;
    frame->local_vars = local_vars_curp;
    frame->ret_addr = ret_addr;
    cur_frame = frame;
}

uint8_t* LaminaVM::pop_frame() noexcept {
    local_vars_curp -= LMX_LOCAL_VAR_COUNT;
    free_frames.push_back(cur_frame);
    const auto ret = cur_frame->ret_addr;
    cur_frame = cur_frame->last;
    return ret;
}

Frame::Frame(Frame* last, uint8_t *ret_addr, Value* local_vars) noexcept
    : last(last), ret_addr(ret_addr), local_vars(local_vars) {}

Frame::~Frame() noexcept = default;

#if defined(__GNUC__) || defined(__clang__)
#define VM_DISPATCH \
static const void* dispatch[] = {\
    &&opNop, &&opNew,\
    &&opGetTrue, &&opGetFalse, &&opGetNull,\
    &&opIConst, &&opCConst, &&opPop, &&opHalt,\
    &&opIAdd, &&opISub, &&opIMul, &&opIDiv, &&opIMod, &&opIPow, &&opINeg,\
    &&opFuncCreate,\
    &&opCallVirtual, &&opCCall, &&opCallFast, &&opRet,\
    &&opGoto,\
    &&opICmpEq, &&opICmpNe, &&opICmpLt, &&opICmpLe, &&opICmpGt, &&opICmpGe,\
    &&opIfTrue, &&opIfFalse,\
    &&opLGet, &&opLSet,\
    &&opGGet, &&opGSet,\
    &&opFAddi, &&opFSUbi, &&opFMuli, &&opFDivi, &&opFModi, &&opFPow, &&opFNeg,\
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

int LaminaVM::run(Code *new_prog) noexcept {
    this->prog = new_prog;
    this->ip = prog->code;

    static auto read_i16 = [](const uint8_t* p) -> int16_t {
        return static_cast<int16_t>(p[0] | p[1] << 8);
    };
    static auto read_u16 = [](const uint8_t* p) -> uint16_t {
        return static_cast<uint16_t>(p[0] | (p[1] << 8));
    };

    VM_DISPATCH

    VM_LABEL(Nop) {
        VM_NEXT
    }

    VM_LABEL(New) {
        regs[ip[1]] = Value();
        VM_NEXT
    }

    VM_LABEL(GetTrue) {
        regs[ip[1]] = Value(true);
        VM_NEXT
    }

    VM_LABEL(GetFalse) {
        regs[ip[1]] = Value(false);
        VM_NEXT
    }

    VM_LABEL(GetNull) {
        regs[ip[1]] = Value();
        VM_NEXT
    }

    VM_LABEL(IConst) {
        regs[ip[1]] = Value(static_cast<int64_t>(read_i16(ip + 2)));
        VM_NEXT
    }

    VM_LABEL(CConst) {
        switch (uint16_t idx = read_u16(ip + 2); cp[idx].id) {
            case ConstantId::Int:
                regs[ip[1]] = Value(cp[idx].int_value);
                break;
            case ConstantId::Str:
                regs[ip[1]] = Value(cp[idx].str);
                break;
            default:
                regs[ip[1]] = Value();
                break;
        }
        VM_NEXT
    }

    VM_LABEL(Pop) {
        regs[ip[1]] = *--stack;
        VM_NEXT
    }

    VM_LABEL(Halt) {
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
        regs[ip[1]] = regs[ip[2]] / regs[ip[3]];
        VM_NEXT
    }

    VM_LABEL(IMod) {
        regs[ip[1]] = regs[ip[2]] % regs[ip[3]];
        VM_NEXT
    }

    VM_LABEL(IPow) {
        int64_t a = regs[ip[2]].int_val;
        int64_t b = regs[ip[3]].int_val;
        regs[ip[1]] = static_cast<int64_t>(std::pow(static_cast<double>(a), static_cast<double>(b)));
        VM_NEXT
    }

    VM_LABEL(INeg) {
        regs[ip[1]] = -regs[ip[2]];
        VM_NEXT
    }

    VM_LABEL(FuncCreate) {
        uint16_t code_idx = read_u16(ip + 2);
        regs[ip[1]] = new Code(code_idx, nullptr);
        VM_NEXT
    }

    VM_LABEL(CallVirtual) {
        VM_NEXT
    }

    VM_LABEL(CCall) {
        VM_NEXT
    }

    VM_LABEL(CallFast) {
        VM_NEXT
    }

    VM_LABEL(Ret) {
        ip = pop_frame();
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

    VM_LABEL(FAddi) {
        regs[ip[1]] = regs[ip[2]] + regs[ip[3]];
        VM_NEXT
    }

    VM_LABEL(FSUbi) {
        regs[ip[1]] = regs[ip[2]] - regs[ip[3]];
        VM_NEXT
    }

    VM_LABEL(FMuli) {
        regs[ip[1]] = regs[ip[2]] * regs[ip[3]];
        VM_NEXT
    }

    VM_LABEL(FDivi) {
        regs[ip[1]] = regs[ip[2]] / regs[ip[3]];
        VM_NEXT
    }

    VM_LABEL(FModi) {
        regs[ip[1]] = regs[ip[2]] % regs[ip[3]];
        VM_NEXT
    }

    VM_LABEL(FPow) {
        int64_t a = regs[ip[2]].int_val;
        int64_t b = regs[ip[3]].int_val;
        regs[ip[1]] = Value(static_cast<int64_t>(std::pow(static_cast<double>(a), static_cast<double>(b))));
        VM_NEXT
    }

    VM_LABEL(FNeg) {
        regs[ip[1]] = -regs[ip[2]];
        VM_NEXT
    }

    VM_END
}

#undef VM_DISPATCH
#undef VM_END
#undef VM_LABEL
#undef VM_NEXT