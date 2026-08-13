//
// Created by meian on 2026/4/6.
// 这一页是关于常量池编码的

#pragma once
#include <cstdint>
#include <vector>
namespace lmx::runtime {

enum class ConstantId : uint8_t {
    Int, Frac, Str, Arr
};
struct ConstantPoolInfo;
#pragma pack(push, 1)
struct FracInfo {
    int32_t num;
    int32_t den;
};
struct StringInfo {
    uint32_t length;
    char str[];
};

enum class TypeTag : uint16_t {
    Func,
};
struct TypeInfo {

};
struct ArrayInfo;
struct ConstantPoolInfo {
    ConstantId id;
    union {
        const int64_t int_value;
        const FracInfo* frac_info;
        const StringInfo* str;
        const ArrayInfo* arr;
    };

    explicit ConstantPoolInfo(decltype(int_value) int_value) noexcept;
    explicit ConstantPoolInfo(decltype(frac_info) frac_info) noexcept;
    explicit ConstantPoolInfo(decltype(str) str) noexcept;
    explicit ConstantPoolInfo(const decltype(arr) arr) noexcept : id(ConstantId::Arr), arr(arr) {}
};

struct ArrayInfo {
    uint32_t len;
    ConstantPoolInfo infos[];
};

#pragma pack(pop)
}