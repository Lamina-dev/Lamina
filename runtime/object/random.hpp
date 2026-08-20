#pragma once

#include "object.hpp"

struct lmmc_rng_t;

namespace lmx::runtime {

class RandomObj final : public Object {
    lmmc_rng_t* rng_;

public:
    explicit RandomObj(lmmc_rng_t* rng) noexcept
        : Object(ObjectKind::Random), rng_(rng) {}
    ~RandomObj() noexcept;

    [[nodiscard]] lmmc_rng_t* handle() const noexcept { return rng_; }
    [[nodiscard]] std::string to_string() const noexcept { return "<rng>"; }
};

}
