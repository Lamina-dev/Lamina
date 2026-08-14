//
// Created by meian on 2026/4/6.
// 这一页是关于常量池编码的

#pragma once
#include <cstdint>
#include <vector>
namespace lmx::runtime {

enum class ConstantId : uint8_t {
    Int, Frac, Str, AdtConstructor
};
#pragma pack(push, 1)
struct FracInfo {
    int32_t num;
    int32_t den;
};
struct StringInfo {
    uint32_t length;
    char str[];
};
struct AdtConstructorInfo {
    uint16_t type_name_length;
    uint16_t constructor_length;
    uint8_t field_count;
    char data[];
};
enum class TypeTag : uint16_t {
    Func,
};
struct TypeInfo {

};
struct ConstantPoolInfo {
    ConstantId id;
    union {
        const int64_t int_value;
        const FracInfo* frac_info;
        const StringInfo* str;
        const AdtConstructorInfo* adt_constructor;
    };

    explicit ConstantPoolInfo(decltype(int_value) int_value) noexcept;
    explicit ConstantPoolInfo(decltype(frac_info) frac_info) noexcept;
    explicit ConstantPoolInfo(decltype(str) str) noexcept;
    explicit ConstantPoolInfo(decltype(adt_constructor) adt_constructor) noexcept;

};



#pragma pack(pop)
}
