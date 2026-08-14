#pragma once

#include "object.hpp"
#include "value.hpp"

#include <string>
#include <cstddef>
#include <utility>
#include <vector>

namespace lmx::runtime {

class AdtObj : public Object {
    std::string type_name_;
    std::string constructor_;
    std::vector<Value> fields_;

public:
    AdtObj(std::string type_name,
           std::string constructor,
           std::vector<Value> fields) noexcept;

    [[nodiscard]] const std::string& type_name() const noexcept { return type_name_; }
    [[nodiscard]] const std::string& constructor() const noexcept { return constructor_; }
    [[nodiscard]] const std::vector<Value>& fields() const noexcept { return fields_; }
    [[nodiscard]] const Value* field(std::size_t index) const noexcept;
    [[nodiscard]] bool equals(const AdtObj& other) const noexcept;
    [[nodiscard]] std::size_t hash() const noexcept;
    [[nodiscard]] std::string to_string() const noexcept;
};

}
