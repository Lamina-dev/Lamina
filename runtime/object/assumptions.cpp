#include "assumptions.hpp"

#include <functional>
#include <utility>

namespace lmx::runtime {

AssumptionsObj::AssumptionsObj()
    : Object(ObjectKind::Assumptions) {}

AssumptionsObj::AssumptionsObj(lamina::AssumptionContext context)
    : Object(ObjectKind::Assumptions), context_(std::move(context)) {}

AssumptionsObj* AssumptionsObj::copy() const {
    return new AssumptionsObj(context_);
}

bool AssumptionsObj::equals(const AssumptionsObj& other) const {
    return context_.serialize() == other.context_.serialize();
}

std::size_t AssumptionsObj::hash() const {
    return std::hash<std::string>{}(context_.serialize());
}

std::string AssumptionsObj::to_string() const noexcept {
    return "assumptions(depth=" + std::to_string(context_.depth()) + ")";
}

} // namespace lmx::runtime
