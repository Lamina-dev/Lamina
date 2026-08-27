
#pragma once
#include <cstdint>
#include <memory>

#include "object.hpp"
#include "value.hpp"
#include "../binary.hpp"
#include "dyncall/dyncall_types.h"
#include "dynload/dynload.h"


namespace lmx::runtime {
class CodeModuleObj;
struct FuncObj {
    CodeModuleObj* mod;
    const uint8_t* addr;
    uint32_t bytecode_len;
    explicit FuncObj(CodeModuleObj* mod, const uint8_t* addr, uint32_t bytecode_len = 0) noexcept;
};
struct NativeFuncObj {
    const void* addr;

    uint8_t args_ty_len;
    ValueKind ret_ty;
    const ValueKind* args_ty;
    const char* name;

    explicit NativeFuncObj(
        const void* addr,
        uint8_t args_ty_len,
        const ValueKind* args_ty,
        ValueKind ret_ty,
        const char* name
    ) noexcept;
};
#if defined(_WIN32) || defined(_WIN64)
constexpr auto lib_prefix = "";
constexpr auto lib_suffix = ".dll";
#elif defined(__linux__) || defined(__unix__)
constexpr auto lib_prefix = "lib";
constexpr auto lib_suffix = ".so";
#elif defined(__APPLE__)
constexpr auto lib_prefix = "lib";
constexpr auto lib_suffix = ".dylib";
#else
#error "Unsupported platform from dynamic loading. What's your System?"
#endif

class CodeModuleObj : public Object {
public:
    DLLib* native_lib_handle{};
    std::vector<ConstantPoolInfo> cp;
    std::vector<FuncObj> funcs;
    std::vector<NativeFuncObj> native_funcs{};
    std::vector<std::unique_ptr<CodeModuleObj>> imports;
    std::vector<TypeInfo> types;
    const uint8_t* code{};
    size_t code_len{};
    std::vector<uint8_t> raw_data{};
    explicit CodeModuleObj(std::vector<uint8_t>&& data);
    ~CodeModuleObj() noexcept;


    [[nodiscard]] std::string   to_string   () const noexcept;
    [[nodiscard]] std::string   type_info   () const noexcept;
    [[nodiscard]] bool          equals(const Object* other) const noexcept;

    [[nodiscard]] bool operator==(const Object& other) const noexcept;
    [[nodiscard]] bool operator!=(const Object& other) const noexcept;

    [[nodiscard]] std::string disassemble() const noexcept;
    void disassemble_to_file(FILE* out) const noexcept;
};

}
