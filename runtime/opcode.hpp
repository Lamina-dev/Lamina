//
// Created by meian on 2026/4/6.
//

#pragma once
#include <cstdint>

namespace lmx::runtime::Opcode {

enum Opcode : uint8_t {
    Nop = 0,    // 0
    New = 1,    // reg(1) constant_tag_idx(2)

    GetTrue,    // reg(1)
    GetFalse,   // reg(1)
    GetNull,    // reg(1)

    IConst,     // reg(1) imm(2)
    NewTuple,     // reg(1) count(1)
    NewArray,   // reg(1) len(2)
    ArrLoad,       // reg(1) reg(1) reg(1)
    Halt,
    IAdd, ISub, IMul, IDiv, IMod, IPow, INeg,    // reg(1) reg(1) reg(1)

    FuncCreate,     // reg(1) constant_tag_idx(2)

    ArrStore, //  reg(1) reg(1) reg(1)
    CCall,       // type_cpidx(2) arg_count(1)
    CallFast,    // idx(2) arg_count(1)
    Ret,    // reg(1)
    Goto,   // ip+(2)
    ICmpEq, ICmpNe, ICmpLt, ICmpLe, ICmpGt, ICmpGe,  // reg(1) reg(1) reg(1)
    IfTrue, IfFalse, // reg(1) then ip+(2)

    LGet, LSet, // reg(1) idx(1)
    GGet, GSet, // idx(2)
    FAdd, FSub, FMul, FDiv, FMod, FNeg,

    MovRR,
    Call,// reg(1) arg_count(1)
    And, Or,
    FCmpEq, FCmpNe, FCmpLt, FCmpLe, FCmpGt, FCmpGe,  // reg(1) reg(1) reg(1)

    GetModule, GetModuleAttr, GetFunc,
    TupleGet, // reg(1), obj_reg(1), idx(1)
    TupleSet, // obj_reg(1), idx(1), reg(1)
    AdtNew, AdtIs, AdtGet,
    LiteralNew, Contains, NotContains
};

}
