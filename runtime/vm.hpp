//
// Created by meian on 2026/4/6.
//

#pragma once
#include <cstdint>
#include <vector>
#include <span>

#include "error.hpp"
#include "dyncall/dyncall.h"
#include "gc.hpp"
#include "lmx.h"
#include "../utils/utils.hpp"
#include "object/code_module.hpp"
#include "object/string.hpp"
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
    // Value* global_vars;
    Frame* cur_frame{};
    LmGCAllocator allocator{};

    std::span<char*> args;
    DCCallVM* call_vm;


    // ConstantPoolInfo* cp;

    LMX_INLINE static void native_arg(DCCallVM* call_vm, const ValueKind k, const Value* v) noexcept {
        switch (k) {
        case ValueKind::Null: dcArgPointer(call_vm, nullptr); break;
        case ValueKind::C_Ptr: dcArgPointer(call_vm, v->c_ptr); break;
        case ValueKind::Obj: {
            const auto* o = v->obj;
            switch (o->get_kind()) {
            case ObjectKind::String: {
                dcArgPointer(call_vm, (DCpointer)reinterpret_cast<const String*>(o)->c_str()); break;
                break;
            }
            default: {
                dcArgPointer(call_vm, (DCpointer)o);
                break;
            }
            }

            break;
        }
        case ValueKind::Int: {
            dcArgLongLong(call_vm, v->int_val);
            break;
        }
        case ValueKind::Bool: {
            dcArgBool(call_vm, v->bool_val);
            break;
        }
        case ValueKind::Fraction: {
            dcArgDouble(call_vm, v->frac_val.to_float());
            break;
        }
        case ValueKind::C_VaList: {
            return;
        }
        }
    }
    LMX_INLINE static void native_arg(DCCallVM* call_vm, const Value* v) noexcept {
        return native_arg(call_vm, v->kind, v);
    }
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
            const auto frame = vm->free_frames[vm->free_frames.size() - 1];
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

    LMX_INLINE void native_call(const uint16_t idx, const uint8_t argc) noexcept {
        const auto mod = cur_frame->mod;
        if (!mod->native_lib_handle) {
            VM_ERROR(VM_ERROR_CanNotCalling + ": module not loaded dynamic library, cannot calling");
        }
        const auto* meta = &mod->native_funcs[idx];
        dcReset(call_vm);
        dcMode(call_vm, DC_CALL_C_DEFAULT);
        uint8_t va_list_len = 0;
        uint8_t i = 0;
        for (; i < argc; ++i) {
            const auto k = meta->args_ty[i];
            if (k == ValueKind::C_VaList) {
                va_list_len = argc - i;
                dcMode(call_vm, DC_CALL_C_ELLIPSIS);
                break;
            }
            native_arg(call_vm, k, &regs[LMX_VM_REG_COUNT - 1 - i]);
        }
        if (va_list_len > 0) {
            dcMode(call_vm, DC_CALL_C_ELLIPSIS_VARARGS);
            for (; i < argc; ++i) {
                native_arg(call_vm, &regs[LMX_VM_REG_COUNT - 1 - i]);
            }
        }

        if (meta->addr) {
            switch (meta->ret_ty) {
            case ValueKind::Null:   regs[0] = dcCallPointer(call_vm, (DCpointer)meta->addr); break;
            case ValueKind::C_Ptr:  regs[0] = dcCallPointer(call_vm, (DCpointer)meta->addr); break;
            case ValueKind::Obj:    regs[0] = static_cast<Object *>(dcCallPointer(call_vm, (DCpointer) meta->addr)); break;
            case ValueKind::Int:    regs[0] = static_cast<LmInt>(dcCallLongLong(call_vm, (DCpointer) meta->addr)); break;
            case ValueKind::Bool:   regs[0] = static_cast<bool>(dcCallBool(call_vm, (DCpointer) meta->addr)); break;
            case ValueKind::Fraction: dcCallVoid(call_vm, (DCpointer)meta->addr); break;
            case ValueKind::C_VaList:
                break;
            }
        }
    }
};


}
