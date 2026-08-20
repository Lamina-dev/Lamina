#pragma once

#include "object.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace lmx::runtime {

class VectorObj final : public Object {
    std::vector<double> data_;

public:
    explicit VectorObj(std::vector<double> data) noexcept
        : Object(ObjectKind::Vector), data_(std::move(data)) {}

    [[nodiscard]] const std::vector<double>& data() const noexcept { return data_; }
    [[nodiscard]] std::vector<double>& data() noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return data_.size(); }
    [[nodiscard]] bool equals(const VectorObj& other) const noexcept;
    [[nodiscard]] std::size_t hash() const noexcept;
    [[nodiscard]] std::string to_string() const noexcept;
};

}
