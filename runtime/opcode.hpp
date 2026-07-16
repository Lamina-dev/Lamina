//
// Created by meian on 2026/4/6.
//

#pragma once
#include <cstdint>

namespace lmx::runtime::Opcode {

enum Opcode : uint8_t {
    Nop,    // 0
    New,    // reg(1) constant_tag_idx(2)

    GetTrue,    // reg(1)
    GetFalse,   // reg(1)
    GetNull,    // reg(1)

    IConst,     // reg(1) imm(2)
    CConst,     // reg(1) constant_tag_idx(2)
    Pop,        // reg(1)
    Halt,
    IAdd, ISub, IMul, IDiv, IMod, IPow, INeg,    // reg(1) reg(1) reg(1)

    FuncCreate,     // reg(1) constant_tag_idx(2)

    CallVirtual, // reg(1) arg_count(1)
    CCall,
    CallFast,    // constant_tag_idx(2) arg_count(1)
    Ret,    //
    Goto,   // ip+(2)
    ICmpEq, ICmpNe, ICmpLt, ICmpLe, ICmpGt, ICmpGe,  // reg(1) reg(1) reg(1)
    IfTrue, IfFalse, // then ip+(2)

    LGet, LSet, // idx(1)
    GGet, GSet, // idx(2)
    FAddi, FSUbi, FMuli, FDivi, FModi, FPow, FNeg,
};

}
