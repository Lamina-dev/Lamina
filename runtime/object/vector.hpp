#pragma once

#include "object.hpp"
#include "lmmc/dense_types.h"

#include <cstddef>
#include <span>
#include <string>
#include <vector>
#include <variant>

namespace lmx::runtime {

class VectorObj final : public Object {
    std::variant<std::vector<double>, lmmc_vec_t> data_;

public:
    explicit VectorObj(std::vector<double> data) noexcept
        : Object(ObjectKind::Vector), data_(std::move(data)) {}

    explicit VectorObj(lmmc_vec_t& data) noexcept;
    ~VectorObj() noexcept;

    VectorObj(const VectorObj&) = delete;
    VectorObj& operator=(const VectorObj&) = delete;
    VectorObj(VectorObj&&) = delete;
    VectorObj& operator=(VectorObj&&) = delete;

    [[nodiscard]] std::span<const double> data() const noexcept;
    [[nodiscard]] std::span<double> data() noexcept;
    [[nodiscard]] std::size_t size() const noexcept { return data().size(); }
    void assign(std::span<const double> values);
    [[nodiscard]] bool equals(const VectorObj& other) const noexcept;
    [[nodiscard]] std::size_t hash() const noexcept;
    [[nodiscard]] std::string to_string() const noexcept;
};

}
