//
// Created by meian on 2026/4/6.
//

#include "binary.hpp"

namespace lmx::runtime {
ConstantPoolInfo::ConstantPoolInfo(decltype(frac_info) frac_info) noexcept
    : id(ConstantId::Frac), frac_info(frac_info) {}

ConstantPoolInfo::ConstantPoolInfo(decltype(int_value) int_value) noexcept
    : id(ConstantId::Int), int_value(int_value) {}

ConstantPoolInfo::ConstantPoolInfo(const decltype(str) str) noexcept
    : id(ConstantId::Str), str(str) {}
}

