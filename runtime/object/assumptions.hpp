#pragma once

#include "object.hpp"
#include "assumption_context.hpp"

#include <cstddef>
#include <string>

namespace lmx::runtime {

class AssumptionsObj final : public Object {
    lamina::AssumptionContext context_;

public:
    AssumptionsObj();
    explicit AssumptionsObj(lamina::AssumptionContext context);

    [[nodiscard]] lamina::AssumptionContext& context() noexcept { return context_; }
    [[nodiscard]] const lamina::AssumptionContext& context() const noexcept {
        return context_;
    }
    [[nodiscard]] AssumptionsObj* copy() const;
    [[nodiscard]] bool equals(const AssumptionsObj& other) const;
    [[nodiscard]] std::size_t hash() const;
    [[nodiscard]] std::string to_string() const noexcept;
};

} // namespace lmx::runtime
