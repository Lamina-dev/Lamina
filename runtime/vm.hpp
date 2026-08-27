
#pragma once
#include <cstdint>
#include <vector>
#include <span>
#include <expected>
#include <string>

#include "error.hpp"
#include "dyncall/dyncall.h"
#include "gc.hpp"
#include "lmx.h"
#include "../utils/utils.hpp"
#include "object/code_module.hpp"
#include "object/StringObj.hpp"
#include "object/value.hpp"

namespace lmx::runtime {

#define LMX_LOCAL_VAR_COUNT 256
#define LMX_CALLSTACK_MAX_COUNT 100
#define LMX_VM_REG_COUNT 256

struct Frame {
    Frame* last;
    CodeModuleObj* mod;
    const uint8_t* ret_addr;
    Value local_vars[LMX_LOCAL_VAR_COUNT];
    explicit Frame(Frame* last, CodeModuleObj* mod, const uint8_t* ret_addr) noexcept;
    ~Frame() noexcept;
};
class LaminaVM {
    std::vector<Frame*> free_frames;
    Value* stack_storage;
    Value* stack;
    Value* regs;
    Frame* cur_frame{};
    LmGCAllocator allocator{};

    std::span<char*> args;
    std::vector<DCCallVM*> call_vms;
    std::size_t native_depth = 0;
    std::size_t invoke_depth = 0;

    Value execute(const uint8_t* start, Frame* stop_frame);



    LMX_INLINE static void native_arg(DCCallVM* call_vm, const ValueKind k, const Value* v) noexcept {
        switch (k) {
        case ValueKind::Null: dcArgPointer(call_vm, nullptr); break;
        case ValueKind::C_Ptr: dcArgPointer(call_vm, v->c_ptr); break;
        case ValueKind::Obj: {
            const auto* o = v->obj;
            switch (o->get_kind()) {
            case ObjectKind::String: {
                dcArgPointer(call_vm, (DCpointer)reinterpret_cast<const StringObj*>(o)->c_str()); break;
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
        case ValueKind::Real: {
            dcArgDouble(call_vm, v->real_val);
            break;
        }
        case ValueKind::Expr: {
            dcArgPointer(call_vm, (DCpointer)v->obj);
            break;
        }
        case ValueKind::Tuple:
        case ValueKind::Set:
        case ValueKind::Interval:
        case ValueKind::Complex:
        case ValueKind::Vector:
        case ValueKind::Matrix:
        case ValueKind::Table:
        case ValueKind::Random:
        case ValueKind::Quantity:
        case ValueKind::Sparse:
        case ValueKind::Tensor:
        case ValueKind::Assumptions: {
            dcArgPointer(call_vm, (DCpointer)v->obj);
            break;
        }
        case ValueKind::C_VaList: {
            return;
        }
        case ValueKind::C_ValueRef: {
            dcArgPointer(call_vm, const_cast<Value*>(v));
            break;
        }
        }
    }
    LMX_INLINE static void native_arg(DCCallVM* call_vm, const Value* v) noexcept {
        return native_arg(call_vm, v->kind, v);
    }

    LMX_INLINE static int native_call_mode() noexcept {
#if (defined(_WIN32) || defined(_WIN64)) && defined(_M_X64)
        return DC_CALL_C_X64_WIN64;
#elif (defined(_WIN32) || defined(_WIN64)) && defined(__x86_64__)
        return DC_CALL_C_X64_WIN64;
#else
        return DC_CALL_C_DEFAULT;
#endif
    }
public:
    explicit LaminaVM() noexcept = delete;
    explicit LaminaVM(int argc, char** argv) noexcept;
    ~LaminaVM() noexcept;

    int run(CodeModuleObj* prog) noexcept;
    std::expected<Value, std::string> invoke(
        const FuncObj& function, std::span<const Value> arguments) noexcept;
    [[nodiscard]] static LaminaVM* current() noexcept;
    Value& get_reg(uint8_t reg) const noexcept;

    friend LMX_INLINE void new_frame(LaminaVM* vm, CodeModuleObj* mod, const uint8_t *ret_addr) noexcept {
        if (vm->free_frames.empty()) {
            vm->cur_frame = new Frame(vm->cur_frame, mod, ret_addr);
        } else {
            const auto frame = vm->free_frames[vm->free_frames.size() - 1];
            vm->free_frames.pop_back();
            frame->last = vm->cur_frame;
            frame->mod = mod;
            frame->ret_addr = ret_addr;
            vm->cur_frame = frame;
        }
    }
    friend LMX_INLINE const uint8_t *pop_frame(LaminaVM* vm) noexcept {
        auto* cur_frame = vm->cur_frame;
        for (auto& local : cur_frame->local_vars) local = Value{};
        vm->free_frames.push_back(cur_frame);
        vm->cur_frame = cur_frame->last;

        return cur_frame->ret_addr;
    }

    LMX_INLINE void native_call(const uint16_t idx, const uint8_t argc) {
        if (native_depth == call_vms.size()) {
            call_vms.push_back(dcNewCallVM(4096));
        }
        auto* call_vm = call_vms[native_depth];
        if (!call_vm) {
            VM_ERROR(RuntimeErrorType::CanNotCalling,
                     "cannot allocate native call state");
        }
        const auto mod = cur_frame->mod;
        if (!mod->native_lib_handle) {
            VM_ERROR(RuntimeErrorType::CanNotCalling, "module not loaded dynamic library, cannot calling");
        }
        const auto* meta = &mod->native_funcs[idx];
        dcReset(call_vm);
        dcMode(call_vm, native_call_mode());
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

        struct NativeDepthGuard {
            std::size_t& depth;
            ~NativeDepthGuard() noexcept { --depth; }
        };
        ++native_depth;
        const NativeDepthGuard native_depth_guard{native_depth};

        if (meta->addr) {
            switch (meta->ret_ty) {
            case ValueKind::Null:   regs[0] = dcCallPointer(call_vm, (DCpointer)meta->addr); break;
            case ValueKind::C_Ptr:  regs[0] = dcCallPointer(call_vm, (DCpointer)meta->addr); break;
            case ValueKind::Obj:    regs[0] = static_cast<Object *>(dcCallPointer(call_vm, (DCpointer) meta->addr)); break;
            case ValueKind::Int:    regs[0] = static_cast<LmInt>(dcCallLongLong(call_vm, (DCpointer) meta->addr)); break;
            case ValueKind::Bool:   regs[0] = static_cast<bool>(dcCallBool(call_vm, (DCpointer) meta->addr)); break;
            case ValueKind::Real:   regs[0] = dcCallDouble(call_vm, (DCpointer) meta->addr); break;
            case ValueKind::Expr:
            case ValueKind::Tuple:
            case ValueKind::Set:
            case ValueKind::Interval:
            case ValueKind::Complex:
            case ValueKind::Vector:
            case ValueKind::Matrix:
            case ValueKind::Table:
            case ValueKind::Random:
            case ValueKind::Quantity:
            case ValueKind::Sparse:
            case ValueKind::Tensor:
            case ValueKind::Assumptions:
                regs[0].~Value();
                regs[0].kind = meta->ret_ty;
                regs[0].obj = static_cast<Object *>(dcCallPointer(call_vm, (DCpointer) meta->addr));
                break;
            case ValueKind::Fraction: dcCallVoid(call_vm, (DCpointer)meta->addr); break;
            case ValueKind::C_VaList:
            case ValueKind::C_ValueRef:
                break;
            }
        }
    }
};


}
