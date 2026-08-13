//
// Created by meian on 2026/3/29.
//

#include "value.hpp"

namespace lmx::runtime {

std::string Value::to_string() const noexcept {
    switch (kind) {
    case ValueKind::Bool: return bool_val ? "true" : "false";
    case ValueKind::Int: return std::to_string(int_val);
    case ValueKind::Obj: return Object::to_string(obj);
    case ValueKind::C_Ptr: return "RawPtr";
    case ValueKind::Null: return "Null";
    case ValueKind::Fraction: return frac_val.to_string();
    case ValueKind::C_VaList: return "VaList";
    }

    // 不可能到达这里
    return {};
}

}