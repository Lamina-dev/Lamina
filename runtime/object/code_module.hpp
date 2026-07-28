//
// Created by meian on 2026/4/6.
//

#pragma once
#include <cstdint>
#include "object.hpp"
#include "value.hpp"
#include "../binary.hpp"
#include "dynload/dynload.h"


namespace lmx::runtime {
class CodeModule;
struct FuncObj {
    CodeModule* mod;
    const uint8_t* addr;
    uint32_t bytecode_len;
    explicit FuncObj(CodeModule* mod, const uint8_t* addr, uint32_t bytecode_len = 0) noexcept;
};
struct NativeFuncObj {
    // CodeModule* mod;
    const void* addr;

    uint8_t args_ty_len;
    const ValueKind* args_ty;
    ValueKind ret_ty;

    explicit NativeFuncObj(
        const void* addr,
        uint8_t args_ty_len,
        const ValueKind* args_ty,
        ValueKind ret_ty
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
class CodeModule : public Object {
public:
    DLLib* native_lib_handle{};
    std::vector<ConstantPoolInfo> cp;
    std::vector<FuncObj> funcs;
    std::vector<TypeInfo> types;
    std::vector<NativeFuncObj> native_funcs{};
    const uint8_t* code;
    size_t code_len;
    std::vector<uint8_t> raw_data{};
    explicit CodeModule(
        std::vector<ConstantPoolInfo> cp,
        std::vector<TypeInfo> types,
        std::vector<FuncObj> funcs,
        std::vector<NativeFuncObj> native_funcs,
        const char* lib_name,
        const uint8_t* code,
        size_t code_len = 0
        ) noexcept;
    explicit CodeModule(
        std::vector<ConstantPoolInfo>&& cp,
        std::vector<TypeInfo>&& types,
        std::vector<FuncObj>&& funcs,
        std::vector<NativeFuncObj>&& native_funcs,
        const char* lib_name,
        const uint8_t* code,
        size_t code_len = 0
        ) noexcept;
    explicit CodeModule(std::vector<uint8_t>&& data) noexcept;
    ~CodeModule() noexcept override;

    [[nodiscard]] Object*       clone       () const noexcept override;

    [[nodiscard]] std::string   to_string   () const noexcept override;
    [[nodiscard]] std::string   type_info   () const noexcept override;
    [[nodiscard]] bool          equals(const Object* other) const noexcept override;

    [[nodiscard]] bool operator==(const Object& other) const noexcept override;
    [[nodiscard]] bool operator!=(const Object& other) const noexcept override;

    [[nodiscard]] std::string disassemble() const noexcept;
    void disassemble_to_file(FILE* out) const noexcept;
};

}
