//
// Created by meian on 2026/4/6.
// 这一页是关于常量池编码的

#pragma once
#include <cstdint>
#include <vector>
namespace lmx::runtime {

enum class ConstantId : uint8_t {
    Int, Frac, Str
};
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
struct ConstantPoolInfo {
    ConstantId id;
    union {
        const int64_t int_value;
        const FracInfo* frac_info;
        const StringInfo* str;
    };

    explicit ConstantPoolInfo(decltype(int_value) int_value) noexcept;
    explicit ConstantPoolInfo(decltype(frac_info) frac_info) noexcept;
    explicit ConstantPoolInfo(decltype(str) str) noexcept;

};




}