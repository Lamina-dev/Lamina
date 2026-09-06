#pragma once

#include "object.hpp"
#include "assumption_context.hpp"

#include <cstddef>
#include <string>

namespace lmx::runtime {

class AssumptionsObj final : public Object {
    LMCAS::AssumptionContext context_;

public:
    AssumptionsObj();
    explicit AssumptionsObj(LMCAS::AssumptionContext context);

    [[nodiscard]] LMCAS::AssumptionContext& context() noexcept { return context_; }
    [[nodiscard]] const LMCAS::AssumptionContext& context() const noexcept {
        return context_;
    }
    [[nodiscard]] AssumptionsObj* copy() const;
    [[nodiscard]] bool equals(const AssumptionsObj& other) const;
    [[nodiscard]] std::size_t hash() const;
    [[nodiscard]] std::string to_string() const noexcept;
};

} // namespace lmx::runtime
