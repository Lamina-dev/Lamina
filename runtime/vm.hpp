//
// Created by meian on 2026/4/6.
//

#pragma once
#include <cstdint>
#include <vector>
#include <span>

#include "dyncall/dyncall.h"
#include "gc.hpp"
#include "lmx.h"
#include "object/code_module.hpp"
#include "object/value.hpp"

namespace lmx::runtime {

#define LMX_LOCAL_VAR_COUNT 256
#define LMX_CALLSTACK_MAX_COUNT 100
#define LMX_VM_REG_COUNT 256

struct Frame {
    Frame* last;
    CodeModule* mod;
    const uint8_t* ret_addr;
    Value local_vars[LMX_LOCAL_VAR_COUNT];
    explicit Frame(Frame* last, CodeModule* mod, const uint8_t* ret_addr) noexcept;
    ~Frame() noexcept;
};
class LaminaVM {

    std::vector<Frame*> free_frames;
    Value regs[LMX_VM_REG_COUNT];
    Value* stack;
    // Value* local_vars_bp;
    // Value* local_vars_curp;
    Value* global_vars;
    Frame* cur_frame{};
    LmGCAllocator allocator{};

    std::span<char*> args;
    DCCallVM* call_vm;


    // ConstantPoolInfo* cp;


public:
    explicit LaminaVM() noexcept = delete;
    explicit LaminaVM(int argc, char** argv) noexcept;
    ~LaminaVM() noexcept;

    int run(CodeModule* prog) noexcept;
    Value& get_reg(uint8_t reg) noexcept;

    friend LMX_INLINE void new_frame(LaminaVM* vm, CodeModule* mod, const uint8_t *ret_addr) noexcept {
        if (vm->free_frames.empty()) {
            vm->cur_frame = new Frame(vm->cur_frame, mod, ret_addr);
            //cur_frame = frame;
            //return;
        } else {
            const auto frame = vm->free_frames.back();
            vm->free_frames.pop_back();
            frame->last = vm->cur_frame;
            frame->mod = mod;
            frame->ret_addr = ret_addr;
            vm->cur_frame = frame;
        }
        // vm->local_vars_curp += LMX_LOCAL_VAR_COUNT;
    }
    friend LMX_INLINE const uint8_t *pop_frame(LaminaVM* vm) noexcept {
        auto* cur_frame = vm->cur_frame;
        // vm->local_vars_curp -= LMX_LOCAL_VAR_COUNT;
        // auto i = 0;
        vm->free_frames.push_back(cur_frame);
        vm->cur_frame = cur_frame->last;
        return cur_frame->ret_addr;
    }

    void native_call() noexcept;
};


}
