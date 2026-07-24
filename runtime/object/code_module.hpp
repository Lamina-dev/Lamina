//
// Created by meian on 2026/4/6.
//

#pragma once
#include <cstdint>
#include <cstdio>

#include "object.hpp"
#include "../binary.hpp"

namespace lmx::runtime {
class CodeModule;
struct FuncObj {
    CodeModule* mod;
    const uint8_t* addr;
    uint32_t bytecode_len;
    explicit FuncObj(CodeModule* mod, const uint8_t* addr, uint32_t bytecode_len = 0) noexcept;
};

class CodeModule : public Object {
public:
    std::vector<ConstantPoolInfo> cp;
    std::vector<FuncObj> funcs;
    std::vector<TypeInfo> types;
    const uint8_t* code;
    size_t code_len;
    explicit CodeModule(
        std::vector<ConstantPoolInfo> cp,
        std::vector<TypeInfo> types,
        std::vector<FuncObj> funcs,
        const uint8_t* code,
        size_t code_len = 0
        ) noexcept;
    explicit CodeModule(
        std::vector<ConstantPoolInfo>&& cp,
        std::vector<TypeInfo>&& types,
        std::vector<FuncObj>&& funcs,
        const uint8_t* code,
        size_t code_len = 0
        ) noexcept;
    explicit CodeModule(const uint8_t* binary) noexcept;
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
