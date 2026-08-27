
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

#include "object/array.hpp"
#include "object/tuple.hpp"

namespace lmx::runtime {
namespace {
thread_local LaminaVM* active_vm = nullptr;
}
LaminaVM::LaminaVM(const int argc, char **argv) noexcept :
    stack(new Value[LMX_VM_REG_COUNT * LMX_CALLSTACK_MAX_COUNT]),
    regs(stack),
    args(argv, argc),
    call_vms{dcNewCallVM(4096)} {}

LaminaVM::~LaminaVM() noexcept {
    delete[] stack;
    for (const auto frames : free_frames) delete frames;
    delete cur_frame;
    for (auto* call_vm : call_vms) dcFree(call_vm);
}

Value &LaminaVM::get_reg(const uint8_t reg) const noexcept {
    return regs[reg];
}

Frame::Frame(Frame* last, CodeModuleObj* mod ,const uint8_t *ret_addr) noexcept
    : last(last), mod(mod), ret_addr(ret_addr)
{}

Frame::~Frame() noexcept = default;

namespace {
void build_constant(LmGCAllocator &allocator, const ConstantPoolInfo &c, Value &dest);

Fraction as_fraction(const Value& value) noexcept {
    if (value.kind == ValueKind::Fraction) return value.frac_val;
    if (value.kind == ValueKind::Int) {
        return Fraction(static_cast<std::int32_t>(value.int_val), 1);
    }
    assert(false && "fraction opcode received a non-numeric value");
    return Fraction();
}

double as_real(const Value& value) noexcept {
    if (value.kind == ValueKind::Real) return value.real_val;
    if (value.kind == ValueKind::Fraction) return value.frac_val.to_float();
    if (value.kind == ValueKind::Int) return static_cast<double>(value.int_val);
    assert(false && "real opcode received a non-numeric value");
    return 0.0;
}

bool uses_real(const Value& lhs, const Value& rhs) noexcept {
    return lhs.kind == ValueKind::Real || rhs.kind == ValueKind::Real;
}
void make_elem(LmGCAllocator &allocator, ArrayObj *arr, const uint32_t idx, const ConstantPoolInfo &e) {
    switch (e.id) {
    case ConstantId::Int:
        arr->store(idx, Value(e.int_value));
        break;
    case ConstantId::Frac:
        arr->store(idx, Value(e.frac_info->num, e.frac_info->den));
        break;
    case ConstantId::Str: {
        Value v(allocator.alloc_string(e.str->str, e.str->length));
        arr->store(idx, std::move(v)); // store 内部已把 v 置空
        break;
    }
    case ConstantId::Arr: {
        Value v;
        build_constant(allocator, e, v);
        arr->store(idx, std::move(v));
        break;
    }
    default:
        break;
    }
}

void build_constant(LmGCAllocator &allocator, const ConstantPoolInfo &c, Value &dest) {
    switch (c.id) {
    case ConstantId::Int:
        dest = c.int_value;
        break;
    case ConstantId::Frac:
        dest = Value(c.frac_info->num, c.frac_info->den);
        break;
    case ConstantId::Str:
        dest = allocator.alloc_string(c.str->str, c.str->length);
        break;
    case ConstantId::Arr: {
        const auto *ai = c.arr;
        auto *arr = reinterpret_cast<ArrayObj *>(allocator.alloc_array(ai->len));
        for (uint32_t i = 0; i < ai->len; i++) {
            make_elem(allocator, arr, i, ai->infos[i]);
        }
        dest = arr;
        break;
    }
    default:
        break;
    }
}
} // namespace

#if defined(__GNUC__) || defined(__clang__)
#define VM_DISPATCH \
static const void* dispatch[] = {\
    &&opNop, &&opNew,\
    &&opGetTrue, &&opGetFalse, &&opGetNull,\
    &&opIConst, &&opNewTuple, &&opNewArray, &&opArrLoad, &&opHalt,\
    &&opIAdd, &&opISub, &&opIMul, &&opIDiv, &&opIMod, &&opIPow, &&opINeg,\
    &&opFuncCreate,\
    &&opArrStore, &&opCCall, &&opCallFast, &&opRet,\
    &&opGoto,\
    &&opICmpEq, &&opICmpNe, &&opICmpLt, &&opICmpLe, &&opICmpGt, &&opICmpGe,\
    &&opIfTrue, &&opIfFalse,\
    &&opLGet, &&opLSet,\
    &&opGGet, &&opGSet,\
    &&opFAdd, &&opFSub, &&opFMul, &&opFDiv, &&opFMod, &&opFNeg,\
    &&opMovRR,&&opCall, &&opAnd, &&opOr,\
    &&opFCmpEq, &&opFCmpNe, &&opFCmpLt, &&opFCmpLe, &&opFCmpGt, &&opFCmpGe, \
    &&opGetModule, &&opGetModuleAttr, &&opGetFunc,\
    &&opTupleGet, &&opTupleSet,\
    &&opAdtNew, &&opAdtIs, &&opAdtGet,\
    &&opLiteralNew, &&opContains, &&opNotContains,\
    &&opRaise,\
    &&opSetUnion, &&opSetIntersection, &&opSetDifference,\
    &&opSetSymmetricDifference, &&opSetSubset\
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

static AdtObj* make_adt_value(const CodeModuleObj* module, Value* regs, const uint16_t constant_index) {
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

static bool adt_matches(
    const AdtObj& value, const std::string& expected) noexcept {
    const auto separator = expected.find('\x1f');
    if (separator == std::string::npos ||
        expected.compare(separator + 1, std::string::npos,
                         value.constructor()) != 0)
        return false;
    const std::string_view expected_type(expected.data(), separator);
    if (expected_type == value.type_name()) return true;
    if (value.type_name().find("::") != std::string::npos ||
        expected_type.size() <= value.type_name().size() + 2)
        return false;
    const auto suffix = expected_type.substr(
        expected_type.size() - value.type_name().size());
    const auto prefix_end =
        expected_type.size() - value.type_name().size();
    return suffix == value.type_name() && prefix_end >= 2 &&
           expected_type.substr(prefix_end - 2, 2) == "::";
}
static Value make_literal_value(Value* regs, const uint8_t count,
                                const uint8_t flags) {
    std::vector<Value> elements;

    elements.reserve(count);
    for (uint8_t i = 0; i < count; ++i) {
        elements.push_back(regs[LMX_VM_REG_COUNT - 1 - i]);
    }
    const auto kind = (flags & 1U) != 0
        ? LiteralObj::Kind::Interval : LiteralObj::Kind::Set;
    auto* literal = new LiteralObj(kind, std::move(elements),
                                   (flags & 2U) != 0, (flags & 4U) != 0);
    return Value(literal, kind == LiteralObj::Kind::Interval
        ? ValueKind::Interval : ValueKind::Set);
}
static const LiteralObj* set_literal(const Value& value) noexcept {
    if (value.kind != ValueKind::Set || !value.obj ||
        value.obj->get_kind() != ObjectKind::Literal)
        return nullptr;
    const auto* literal = reinterpret_cast<const LiteralObj*>(value.obj);
    return literal->literal_kind() == LiteralObj::Kind::Set ? literal : nullptr;
}

static Value make_set_value(std::vector<Value> elements) {
    return Value(new LiteralObj(LiteralObj::Kind::Set, std::move(elements)),
                 ValueKind::Set);
}

LaminaVM* LaminaVM::current() noexcept {
    return active_vm;
}

int LaminaVM::run(CodeModuleObj* prog) noexcept {
    if (!prog) return 1;
    const auto* previous_vm = active_vm;
    active_vm = this;
    regs = stack;
    new_frame(this, prog, nullptr);
    if (const char* debug = std::getenv("LMX_DEBUG_DUMP");
        debug && debug[0] != '\0' && debug[0] != '0') {
        std::cout << prog->disassemble() << std::endl;
    }
    try {
        (void)execute(prog->code, nullptr);
        while (cur_frame) (void)pop_frame(this);
        for (std::size_t i = 0; i < LMX_VM_REG_COUNT; ++i) regs[i] = Value{};
        active_vm = const_cast<LaminaVM*>(previous_vm);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        while (cur_frame) (void)pop_frame(this);
        for (std::size_t i = 0; i < LMX_VM_REG_COUNT; ++i) regs[i] = Value{};
        active_vm = const_cast<LaminaVM*>(previous_vm);
        return 1;
    } catch (...) {
        std::cerr << "unknown VM failure" << std::endl;
        while (cur_frame) (void)pop_frame(this);
        for (std::size_t i = 0; i < LMX_VM_REG_COUNT; ++i) regs[i] = Value{};
        active_vm = const_cast<LaminaVM*>(previous_vm);
        return 1;
    }
}

std::expected<Value, std::string> LaminaVM::invoke(
    const FuncObj& function, const std::span<const Value> arguments) noexcept {
    if (!cur_frame || active_vm != this) {
        return std::unexpected("Lamina callback invoked outside an active VM");
    }
    if (arguments.size() > LMX_LOCAL_VAR_COUNT) {
        return std::unexpected("Lamina callback has too many arguments");
    }
    const auto register_offset = static_cast<std::size_t>(regs - stack);
    if (register_offset + LMX_VM_REG_COUNT >=
        LMX_VM_REG_COUNT * LMX_CALLSTACK_MAX_COUNT) {
        return std::unexpected("Lamina callback nesting limit exceeded");
    }

    Frame* const outer_frame = cur_frame;
    Value* const outer_regs = regs;
    Value* const callback_regs = regs + LMX_VM_REG_COUNT;
    new_frame(this, function.mod, nullptr);
    for (std::size_t i = 0; i < arguments.size(); ++i) {
        cur_frame->local_vars[i] = arguments[i];
    }
    regs = callback_regs;
    ++invoke_depth;
    try {
        auto result = execute(function.addr, outer_frame);
        while (cur_frame != outer_frame) (void)pop_frame(this);
        regs = outer_regs;
        for (std::size_t i = 0; i < LMX_VM_REG_COUNT; ++i) {
            callback_regs[i] = Value{};
        }
        --invoke_depth;
        return result;
    } catch (const std::exception& error) {
        while (cur_frame != outer_frame) (void)pop_frame(this);
        regs = outer_regs;
        for (std::size_t i = 0; i < LMX_VM_REG_COUNT; ++i) {
            callback_regs[i] = Value{};
        }
        --invoke_depth;
        return std::unexpected(error.what());
    } catch (...) {
        while (cur_frame != outer_frame) (void)pop_frame(this);
        regs = outer_regs;
        for (std::size_t i = 0; i < LMX_VM_REG_COUNT; ++i) {
            callback_regs[i] = Value{};
        }
        --invoke_depth;
        return std::unexpected("unknown Lamina callback failure");
    }
}

Value LaminaVM::execute(const uint8_t* ip, Frame* stop_frame) {
    VM_DISPATCH


    VM_LABEL(Nop) {
        ip++;
        VM_NEXT_RAW
    }

    VM_LABEL(New) {
        const auto &c = cur_frame->mod->cp[read_u16(ip + 2)];
        build_constant(allocator, c, regs[ip[1]]);
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

    VM_LABEL(NewTuple) {
        regs[ip[1]] = Value(allocator.alloc_tuple(ip[2]), ValueKind::Tuple);
        VM_NEXT
    }

    VM_LABEL(NewArray) {
        regs[ip[1]] = allocator.alloc_array(read_u16(ip + 2));
        VM_NEXT
    }

    VM_LABEL(ArrLoad) {
        regs[ip[1]] = reinterpret_cast<ArrayObj*>(regs[ip[2]].obj)->at(regs[ip[3]].int_val);
        VM_NEXT
    }

    VM_LABEL(Halt) {
        return Value{};
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
        new (&regs[ip[1]]) Value (
            static_cast<decltype(Fraction::num)>(regs[ip[2]].int_val),
            static_cast<decltype(Fraction::den)>(regs[ip[3]].int_val));
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

    VM_LABEL(FuncCreate) {
        uint16_t code_idx = read_u16(ip + 2);
        VM_NEXT
    }

    VM_LABEL(ArrStore) {
        reinterpret_cast<ArrayObj*>(regs[ip[1]].obj)->store(regs[ip[2]].int_val, std::move(regs[ip[3]]));
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
        const auto* return_address = pop_frame(this);
        auto* returned_regs = regs;
        regs -= LMX_VM_REG_COUNT;
        if (cur_frame == stop_frame) return std::move(returned_regs[0]);
        regs[0] = std::move(returned_regs[0]);
        ip = return_address;
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
        VM_NEXT
    }

    VM_LABEL(GSet) {
        VM_NEXT
    }

    VM_LABEL(FAdd) {
        regs[ip[1]] = uses_real(regs[ip[2]], regs[ip[3]])
            ? Value(as_real(regs[ip[2]]) + as_real(regs[ip[3]]))
            : Value(as_fraction(regs[ip[2]]) + as_fraction(regs[ip[3]]));
        VM_NEXT
    }

    VM_LABEL(FSub) {
        regs[ip[1]] = uses_real(regs[ip[2]], regs[ip[3]])
            ? Value(as_real(regs[ip[2]]) - as_real(regs[ip[3]]))
            : Value(as_fraction(regs[ip[2]]) - as_fraction(regs[ip[3]]));
        VM_NEXT
    }

    VM_LABEL(FMul) {
        regs[ip[1]] = uses_real(regs[ip[2]], regs[ip[3]])
            ? Value(as_real(regs[ip[2]]) * as_real(regs[ip[3]]))
            : Value(as_fraction(regs[ip[2]]) * as_fraction(regs[ip[3]]));
        VM_NEXT
    }

    VM_LABEL(FDiv) {
        regs[ip[1]] = uses_real(regs[ip[2]], regs[ip[3]])
            ? Value(as_real(regs[ip[2]]) / as_real(regs[ip[3]]))
            : Value(as_fraction(regs[ip[2]]) / as_fraction(regs[ip[3]]));
        VM_NEXT
    }

    VM_LABEL(FMod) {
        regs[ip[1]] = uses_real(regs[ip[2]], regs[ip[3]])
            ? Value(std::fmod(as_real(regs[ip[2]]), as_real(regs[ip[3]])))
            : Value(as_fraction(regs[ip[2]]) % as_fraction(regs[ip[3]]));
        VM_NEXT
    }

    VM_LABEL(FNeg) {
        regs[ip[1]] = regs[ip[2]].kind == ValueKind::Real
            ? Value(-regs[ip[2]].real_val)
            : Value(-as_fraction(regs[ip[2]]));
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
        regs[ip[1]] = uses_real(regs[ip[2]], regs[ip[3]])
            ? as_real(regs[ip[2]]) == as_real(regs[ip[3]])
            : as_fraction(regs[ip[2]]) == as_fraction(regs[ip[3]]);
        VM_NEXT
    }
    VM_LABEL(FCmpNe) {
        regs[ip[1]] = uses_real(regs[ip[2]], regs[ip[3]])
            ? as_real(regs[ip[2]]) != as_real(regs[ip[3]])
            : as_fraction(regs[ip[2]]) != as_fraction(regs[ip[3]]);
        VM_NEXT
    }
    VM_LABEL(FCmpLt) {
        regs[ip[1]] = uses_real(regs[ip[2]], regs[ip[3]])
            ? as_real(regs[ip[2]]) < as_real(regs[ip[3]])
            : as_fraction(regs[ip[2]]) < as_fraction(regs[ip[3]]);
        VM_NEXT
    }
    VM_LABEL(FCmpLe) {
        regs[ip[1]] = uses_real(regs[ip[2]], regs[ip[3]])
            ? as_real(regs[ip[2]]) <= as_real(regs[ip[3]])
            : as_fraction(regs[ip[2]]) <= as_fraction(regs[ip[3]]);
        VM_NEXT
    }
    VM_LABEL(FCmpGt) {
        regs[ip[1]] = uses_real(regs[ip[2]], regs[ip[3]])
            ? as_real(regs[ip[2]]) > as_real(regs[ip[3]])
            : as_fraction(regs[ip[2]]) > as_fraction(regs[ip[3]]);
        VM_NEXT
    }
    VM_LABEL(FCmpGe) {
        regs[ip[1]] = uses_real(regs[ip[2]], regs[ip[3]])
            ? as_real(regs[ip[2]]) >= as_real(regs[ip[3]])
            : as_fraction(regs[ip[2]]) >= as_fraction(regs[ip[3]]);
        VM_NEXT
    }
    VM_LABEL(GetModule) {
        const auto index = read_u16(ip + 2);
        if (index >= cur_frame->mod->imports.size() ||
            !cur_frame->mod->imports[index]) {
            VM_ERROR(RuntimeErrorType::ModuleLoad,
                     "imported module " + std::to_string(index) + " is unavailable");
        }
        regs[ip[1]] = cur_frame->mod->imports[index]->get();
        VM_NEXT
    }
    VM_LABEL(GetModuleAttr) {
        regs[ip[1]] = &reinterpret_cast<CodeModuleObj*>(regs[0].obj)->funcs[read_u16(ip + 2)];
        VM_NEXT
    }
    VM_LABEL(GetFunc) {
        regs[ip[1]] = &cur_frame->mod->funcs[read_u16(ip + 2)];
        VM_NEXT
    }
    VM_LABEL(TupleGet) {
        regs[ip[1]] = reinterpret_cast<TupleObj*>(regs[ip[2]].obj)->get(ip[3]);
        VM_NEXT
    }
    VM_LABEL(TupleSet) {
        reinterpret_cast<TupleObj*>(regs[ip[1]].obj)->set(ip[2], regs[ip[3]]);
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
            const auto expected = reinterpret_cast<const StringObj*>(constructor_value.obj)->to_string();
            result = adt_matches(*adt, expected);
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
        const bool result = (container.kind == ValueKind::Set ||
                             container.kind == ValueKind::Interval) && container.obj &&
            container.obj->get_kind() == ObjectKind::Literal &&
            reinterpret_cast<const LiteralObj*>(container.obj)->contains(regs[ip[2]]);
        regs[ip[1]] = result;
        VM_NEXT
    }
    VM_LABEL(NotContains) {
        const auto& container = regs[ip[3]];
        const bool result = (container.kind == ValueKind::Set ||
                             container.kind == ValueKind::Interval) && container.obj &&
            container.obj->get_kind() == ObjectKind::Literal &&
            reinterpret_cast<const LiteralObj*>(container.obj)->contains(regs[ip[2]]);
        regs[ip[1]] = !result;
        VM_NEXT
    }
    VM_LABEL(Raise) {
        const auto text = regs[ip[1]].to_string();
        VM_ERROR(RuntimeErrorType::Runtime, text);
    }
    VM_LABEL(SetUnion) {
        const auto* lhs = set_literal(regs[ip[2]]);
        const auto* rhs = set_literal(regs[ip[3]]);
        if (!lhs || !rhs)
            VM_ERROR(RuntimeErrorType::Construct,
                     "set union received a non-set operand");
        regs[ip[1]] = make_set_value(lhs->union_elements(*rhs));
        VM_NEXT
    }
    VM_LABEL(SetIntersection) {
        const auto* lhs = set_literal(regs[ip[2]]);
        const auto* rhs = set_literal(regs[ip[3]]);
        if (!lhs || !rhs)
            VM_ERROR(RuntimeErrorType::Construct,
                     "set intersection received a non-set operand");
        regs[ip[1]] = make_set_value(lhs->intersection_elements(*rhs));
        VM_NEXT
    }
    VM_LABEL(SetDifference) {
        const auto* lhs = set_literal(regs[ip[2]]);
        const auto* rhs = set_literal(regs[ip[3]]);
        if (!lhs || !rhs)
            VM_ERROR(RuntimeErrorType::Construct,
                     "set difference received a non-set operand");
        regs[ip[1]] = make_set_value(lhs->difference_elements(*rhs));
        VM_NEXT
    }
    VM_LABEL(SetSymmetricDifference) {
        const auto* lhs = set_literal(regs[ip[2]]);
        const auto* rhs = set_literal(regs[ip[3]]);
        if (!lhs || !rhs)
            VM_ERROR(RuntimeErrorType::Construct,
                     "set symmetric difference received a non-set operand");
        regs[ip[1]] =
            make_set_value(lhs->symmetric_difference_elements(*rhs));
        VM_NEXT
    }
    VM_LABEL(SetSubset) {
        const auto* lhs = set_literal(regs[ip[2]]);
        const auto* rhs = set_literal(regs[ip[3]]);
        if (!lhs || !rhs)
            VM_ERROR(RuntimeErrorType::Construct,
                     "set subset received a non-set operand");
        regs[ip[1]] = lhs->subset_of(*rhs);
        VM_NEXT
    }

    VM_END
}
}
#undef VM_DISPATCH
#undef VM_END
#undef VM_LABEL
#undef VM_NEXT
