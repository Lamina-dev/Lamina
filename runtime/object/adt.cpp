#include "adt.hpp"

#include <sstream>
#include <functional>

using namespace lmx::runtime;

AdtObj::AdtObj(std::string type_name,
               std::string constructor,
               std::vector<Value> fields) noexcept
    : Object(ObjectKind::Adt),
      type_name_(std::move(type_name)),
      constructor_(std::move(constructor)),
           fields_(std::move(fields)) {}

const Value* AdtObj::field(const std::size_t index) const noexcept {
    if (index >= fields_.size()) return nullptr;
    return &fields_[index];
}

bool AdtObj::equals(const AdtObj& other) const noexcept {
    return type_name_ == other.type_name_ && constructor_ == other.constructor_ && fields_ == other.fields_;
}

namespace {
void combine_hash(std::size_t& seed, const std::size_t value) noexcept {
    seed ^= value + 0x9e3779b9U + (seed << 6U) + (seed >> 2U);
}

}

std::size_t AdtObj::hash() const noexcept {
    std::size_t result = std::hash<std::string>{}(type_name_);
    combine_hash(result, std::hash<std::string>{}(constructor_));
    for (const auto& field : fields_) combine_hash(result, field.hash());
    return result;
}

std::string AdtObj::to_string() const noexcept {
    std::ostringstream out;
    out << constructor_;
    if (!fields_.empty()) {
        out << "(";
        for (std::size_t i = 0; i < fields_.size(); ++i) {
            if (i != 0) out << ", ";
            out << fields_[i].to_string();
        }
        out << ")";
    }
    return out.str();
}
